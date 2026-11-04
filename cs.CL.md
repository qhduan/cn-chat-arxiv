# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [WebHarbor: Docking Real Websites for Evolving GUI Agent Environments](https://arxiv.org/abs/2611.03634) | WebHarbor通过任务驱动的人机协作，构建了153个忠实还原真实网站的Docker化可执行环境，提供涵盖多步骤和登录后工作流的3,518个任务，为GUI网页智能体提供稳定可控的评估与训练环境。 |
| [^2] | [D&A: Decoupling Description and Assessment Enables VLMs as Zero-Shot Robotic Reward Models](https://arxiv.org/abs/2611.03622) | 提出D&A方法，将任务进度估计解耦为客观的场景描述与基于显式评分标准的评估两个阶段，使视觉语言模型无需昂贵后训练即可作为零样本机器人奖励模型，并有效避免确认偏差问题。 |
| [^3] | [Strategist: Evolving Reasoning with Agents](https://arxiv.org/abs/2611.03618) | Strategist是一个元智能体，通过将推理策略编写为可复用的蓝图并维护策略库，实现跨任务的推理策略自动演化，在30个基准测试上以低39%的推理成本达到与最强基线相当甚至更优的性能，且策略可迁移至未见任务和其他模型。 |
| [^4] | [Tracing the Scientific Discovery Capacity of LLMs to Their Training Data](https://arxiv.org/abs/2611.03617) | 该研究提出“单次事实”理论，通过直接操纵训练数据，证明训练集中仅出现一次的事实比率是决定大语言模型科学发现能力的关键因素，且这一能力可在不同模型、领域间迁移。 |
| [^5] | [ufakzeka-kulak: Building and Evaluating a 31.6M-Parameter Turkish Speech Recognizer from Scratch](https://arxiv.org/abs/2611.03605) | 该论文从零构建并开源了31.6M参数的土耳其语语音识别模型ufakzeka-kulak，仅使用可商业授权的数据训练，采用FastConformer编码器与TDT解码，在FLEURS和Common Voice土耳其语测试集上的词错误率均优于Whisper base和Whisper small。 |
| [^6] | [DiabAgent: Toward Agentic Physiological Control](https://arxiv.org/abs/2611.03601) | 本文提出DiabAgent研究套件，通过APC智能体生理控制框架、DiabetSim糖尿病模拟器和SweetBench基准，首次系统性地实现并定量评估了医学智能体在1型糖尿病血糖持续控制中的能力。 |
| [^7] | [Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training](https://arxiv.org/abs/2611.03577) | Sonar-MoE通过在反向传播过程中即时更新并释放专家梯度，将MoE训练的峰值内存降低19%–22%、每步速度提升17%–23%，同时保持与标准训练位级别一致的结果。 |
| [^8] | [NitiBench+: A Temporal Thai Legal Question Answering Benchmark Across 17 Domains](https://arxiv.org/abs/2611.03576) | NitiBench+是一个包含1,670个条目、覆盖17个领域、同时收录判决日有效与当前法条文本的泰国时序法律问答基准，实验表明模型使用过时法条文本会显著损害法律问答表现。 |
| [^9] | [VeriHarm-Bench: Measuring Harmful Success in LLM Coding Agents Verifiably](https://arxiv.org/abs/2611.03575) | 该论文提出了VeriHarm-Bench基准，首次定义并系统地测量LLM编程代理中的“有害成功”现象——即代理在成功完成良性任务的同时造成非预期的危害，而传统任务成功指标和现有安全评估均无法捕捉这一问题。 |
| [^10] | [Can LLM Agents Be Forward-Deployed Engineers? Requirement Discovery and Delivery for Enterprise Text-to-SQL](https://arxiv.org/abs/2611.03551) | 本研究在企业级文本到SQL的真实项目场景中评估了LLM智能体担任“一线部署工程师”的能力，发现即便最强的模型也只能发现约80%的隐藏需求，且各配置中仅25%的交付运行能通过全部验收标准，并总结出需求发现不完整等四大瓶颈。 |
| [^11] | [Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime](https://arxiv.org/abs/2611.03526) | 提出层库架构，使模型能在推理时动态调整Transformer层的执行顺序，用一个统一框架同时概括了密集、深度循环和专家混合架构在参数效率与计算效率上的权衡。 |
| [^12] | [VIAS: Value-Isolating Activation Steering for Multi-Value Alignment and Compliance for Organisations](https://arxiv.org/abs/2611.03518) | 提出价值隔离激活引导方法VIAS，通过对比包含全部价值观与省略单一目标价值观的两种提示下模型回答的差异，构建能够单独分离各价值观的引导向量，从而实现大语言模型的多价值观对齐与组织合规。 |
| [^13] | [Online Label-Free Gradient Matching for Continual Language Model Training](https://arxiv.org/abs/2611.03496) | OLGA提出了一种在线无标签的梯度匹配数据蒸馏方法，利用模型自身的下一个词预测作为梯度匹配目标，摆脱了对真实标签的依赖，能够持续处理不断增长的数据流，在持续语言模型训练中显著优于现有需要标签的方法。 |
| [^14] | [Optimize the Harness, Not the Model: Fitting a Minimal Agent to a Self-Hostable LLM for Investigative Analysis](https://arxiv.org/abs/2611.03434) | 提出CLEO方法，通过优化最小智能体框架而非模型本身，使其适配可自托管的大语言模型，在调查分析任务上以小两个数量级的规模达到接近最强商用基线的83.8%通过率。 |
| [^15] | [IdeaTaste: Do LLMs Share the Research Taste of Human Researchers?](https://arxiv.org/abs/2611.03424) | IdeaTaste基准通过将LLM基于2025年文献快照撰写的研究提案与人类社区2026年的实际发表工作进行匹配评分，发现LLM在引用匹配和想法匹配上仅达到人类水平的约26%和18%，揭示了LLM尚不具备人类研究者的研究品味。 |
| [^16] | [Same Outputs, Different Scores: Auditing the Unchecked Graders of Multimodal LLM Benchmarks](https://arxiv.org/abs/2611.03377) | 该研究首次系统审计了多模态LLM基准测试中隐藏的评分规则，发现仅评分器本身就能在模型输出完全相同的情况下使准确率产生高达22个百分点的差异，并提出了人类与LLM裁判结合的盲测仲裁方法来恢复模型的真实答案。 |
| [^17] | [Characterizing When Language Models Take Harmful Actions to Protect Their Assigned Goals](https://arxiv.org/abs/2611.03374) | 该论文将智能体失准、伪装对齐等看似不同的语言模型安全失败现象统一形式化为“目标-伤害冲突”这一共同决策结构，并通过从 logit 概率到开放式采样的多种引出协议，系统刻画了模型何时会为保全被赋予的目标而采取有害行动。 |
| [^18] | [Steering with Multilingual Lexicons for Sentiment Control in Low-Resource Languages](https://arxiv.org/abs/2611.03317) | 该论文提出用多语言情感词典替代句子级对比数据来实现激活导向的情感控制，证明聚合多语言词典对低资源语言优于英语导向向量，并从机制上揭示了“语言坍塌”现象及架构偏见的约束作用。 |
| [^19] | [MultiInpaintCOCO: Fine-Grained Visual Understanding in Multiple Languages](https://arxiv.org/abs/2611.03309) | 该论文提出MultiInpaintCOCO——一个覆盖13种语言、包含953个最小对示例的多语言细粒度视觉理解基准，发现视觉-语言模型在低资源语言上性能显著下降，其主因是语言数据资源的匮乏而非与英语的类型学相似度，且模型在跨语言检索中存在对英语的系统性偏差。 |
| [^20] | [LLMs Depend on Specialist Tools for Out-of-Distribution Generalization in MS/MS Structural Elucidation](https://arxiv.org/abs/2611.03308) | 前沿大语言模型在MS/MS从头结构解析中，分布内可达52%准确率但分布外骤降至2%，表明其泛化能力主要依赖记忆；唯有引入从头结构生成器等专业工具才能将分布外准确率提升至38%。 |
| [^21] | [Specificity Bias and the Wikipedia Paradox: An xAI Analysis of AI Text Detectors under Domain Shift](https://arxiv.org/abs/2611.03280) | 该研究通过基于SHAP的可解释AI框架发现，AI文本检测器在域偏移下主要依赖指令微调大模型特有的“行动号召”风格而非通用机器文本特征进行判断，且大语言模型可通过幻觉出特定命名实体的“特异性偏差”策略绕过检测。 |
| [^22] | [Do Agents Still Follow the Rules? Benchmarking Rule Compliance under Agent Context Compaction](https://arxiv.org/abs/2611.03223) | 提出 ARCBench 基准，在完整智能体框架中通过实际执行动作测量上下文压缩后的规则遵守情况，并发现压缩后的大多数违规并非因为规则被删除，而是在规则仍保留于压缩历史中时智能体对后续请求处理不当所致。 |
| [^23] | [Written Once, Read Differently How Language Models Own Answers Steer Their Next Ones](https://arxiv.org/abs/2611.03198) | 语言模型对是非问题的回答会通过在残差流中写入一个稳定的、跨主题共享的“承诺方向”来引导其对后续相关问题的回答，但该方向被读取的机制并不稳定，会随模型规模、指令微调乃至系统提示词而改变。 |
| [^24] | [Dh\={a}r\={a} Bench: An Evidence-Oriented Retrieval Benchmark for Indian Knowledge Systems](https://arxiv.org/abs/2611.03196) | 本文构建了 Dhārā Bench——首个面向印度知识体系的证据导向检索基准，包含 325 个多类别问题、60,121 个文本片段及 7,578 条相关性标注，并通过 BM25 基线实验揭示了该领域检索的挑战性。 |
| [^25] | [ACTOR: Action-Centered Temporal Organization for Literary Role-Playing Agents](https://arxiv.org/abs/2611.03184) | 提出ACTOR自动化框架，从原始小说文本中提取非对话行为、以“思考-行动-观察”结构表示并按故事时间顺序组织成角色行为轨迹数据集，并通过反事实行为实验揭示现有模型难以可靠推断与角色人设一致的行为。 |
| [^26] | [Every Clause Counts: Instruction-Following Reranking by Multiplying Small, Answerable Questions](https://arxiv.org/abs/2611.03179) | DMReranker将带指令的查询拆分为主题和多个子句，让小型评判模型逐一回答并连乘答案概率，无需任何训练即可将指令敏感度（WISE）从33.5大幅提升至50.9，同时保持原始相关性排序性能。 |
| [^27] | [CoworkOS:Scaling Agent Coworkers for Real-world Automation](https://arxiv.org/abs/2611.03178) | CoworkOS 是一个面向长时程智能体协作工作的通用框架，通过“任务策略与执行管理分离”的三层架构和人类与智能体共享编辑同一工作产物的应用层，实现了真实世界自动化中高质量交付与长时程监控决策的结合。 |
| [^28] | [ViAdminShift: A Vietnamese Benchmark for Event-Level Knowledge Updates Built from the 2025 Administrative Reform](https://arxiv.org/abs/2611.03171) | 该论文构建了ViAdminShift——首个基于越南2025年省县乡三级行政区划大改革这一真实事件的事件级知识更新越南语基准，包含7,995个测试项目和经严格交叉验证的新旧对照数据，用以系统评估语言模型对现实世界知识变更的更新与应对能力。 |
| [^29] | [DepthGraft: Structural Regularization through Hierarchical Cross-Layer KV Reconstruction](https://arxiv.org/abs/2611.03142) | 论文提出DepthGraft，一种分层KV共享架构，将KV复用边转化为从深层到源KV投影的直接梯度路径，使KV共享成为隐式结构正则化器，在不引入额外计算开销的前提下缓解Pre-LN架构的深度信息稀释问题。 |
| [^30] | [Meet Students Where They Are: Student-Prefix Teacher Adaptation for On-Policy Distillation](https://arxiv.org/abs/2611.03133) | 提出SPTA方法，在蒸馏前用可验证奖励的强化学习训练教师从随机长度的学生生成前缀继续推理并得出正确答案，使教师适应学生所处的状态分布，从而提升在线策略蒸馏中token级监督的可靠性。 |
| [^31] | [Preprocessing Choices in Fine-Tuning Data Matter: Utility and Demographic Sensitivity in LLM-Based Recommendation](https://arxiv.org/abs/2611.03113) | 该论文通过受控实验证明，微调数据中的预处理选择——包括物品标题表示、人口统计平衡和辅助物品知识增强——会显著影响基于大语言模型的生成式推荐系统的推荐效用和人口统计敏感性，且目录约束解码能有效保证输出的有效性。 |
| [^32] | [DR Pixel: Page-Native Deep Research with Vision-Language Models](https://arxiv.org/abs/2611.03097) | DR Pixel是首个开源端到端多模态深度研究智能体，直接以渲染的文档页面图像作为唯一检索单元和推理单元，在科学论文基准上无需训练即超越基于OCR的基线方法，且工具调用次数更少。 |
| [^33] | [An Experimental Study of Hybrid Retrieval, NLI and Open-source LLMs for Scientific Claim Verification](https://arxiv.org/abs/2611.03093) | 本研究提出一种保守融合规则，将NLI模型与开源LLM的判断相结合——仅当两者一致时才标注“支持”或“反驳”，否则判定“证据不足”，并配合语义与BM25混合检索，在SciFact和CliniFact构建的2,035个声明基准上验证了该方法在科学声明验证中的有效性。 |
| [^34] | [Avatar Intelligence: Amplifying Human Agency Through Persistent Personal Agents](https://arxiv.org/abs/2611.03092) | 本文提出“化身智能”范式，通过跨时间和情境持续代表用户目标、价值观与约束的持久个人智能体来增强人类能动性，并引入智能体监督指数（AOI）框架，以评估智能体交付价值与运营成本及人类监督负担之间的平衡。 |
| [^35] | [SAGE (Subjectivity-Aware Generation Estimator): A Linguistic Framework for AI-Generated Review Detection *](https://arxiv.org/abs/2611.03060) | 本文提出基于主观性语言学特征（评价性语言、情态和言语行为）的轻量级检测器SAGE，与Sem-Detect结合后将AI生成审稿检测在APRDB基准上1%假阳性率下的真阳性率从0.874提升至0.985，且具有可解释性。 |
| [^36] | [Unlocking Criminal Court Decisions: Flexible Information Extraction for Empirical Legal Research](https://arxiv.org/abs/2611.03051) | 提出一种基于大语言模型、标注指令可由法律专家迭代完善的灵活信息抽取流水线，将刑事判决书转化为结构化数据以支持实证法学研究，并将醉酒驾驶案件的抽取准确率从52%提升至70%。 |
| [^37] | [VERS: Steering Reasoning Models by Editing Their Activation Explanations](https://arxiv.org/abs/2611.03040) | VERS 利用自然语言自编码器在推理轨迹的“疑点”处编辑激活解释，将编辑转化为一个固定引导方向并注入到推理段落分隔处，从而在几乎不损失准确率的情况下大幅减少推理模型的生成 token 数量，且该方向可迁移到更大的模型上。 |
| [^38] | [Finding Errors Is Not Enough: Selective Model Escalation for Entity-Targeted Sentiment Analysis](https://arxiv.org/abs/2611.02989) | 该论文研究了实体级情感分析中基于不确定性的选择性模型升级策略，发现优先采集少数类样本仅在极小训练规模下有效，而将低置信度预测交由替代编码器或本地大语言模型处理可纠正错误并提升性能。 |
| [^39] | [Distilling Literary Sentiment from LLM Annotations into an Ordinal BERT Model](https://arxiv.org/abs/2611.02985) | 本研究利用Gemma 4 31B生成的标注通过序数回归蒸馏训练小型BERT模型进行文学情感分析，其中BERT-large的性能接近教师模型，且BERT-base在非中性句子上的表现甚至超越了教师模型。 |
| [^40] | [Specification-Aware Predictive Uncertainty Decomposition](https://arxiv.org/abs/2611.02956) | 本文提出一种规范感知的预测不确定性分解方法，将预测风险分解为规范不确定性、残余偶然不确定性和认知不确定性三项，从而量化了在大语言模型中通过澄清欠规范输入（补全用户意图）能够减少多少预测不确定性。 |
| [^41] | [Action Verification through Conjecture and Refutation for Language Model Agents](https://arxiv.org/abs/2611.02950) | 该论文提出BlackSwan，一种受“猜想与反驳”思想启发的动作验证方法，通过预测动作的预期效果并主动构造、检验可能推翻动作正确性的具体假设，从而捕捉传统确认式验证无法发现的不当智能体行为。 |
| [^42] | [PlagiaCorpus: A Bilingual, Adversarial, and Cross-Lingual Dataset for Forensic Plagiarism Detection](https://arxiv.org/abs/2611.02940) | 本文发布了 PlagiaCorpus——一个包含近 20 万条记录的阿拉伯语-英语双语数据集，通过逐字复制、字符级混淆、LLM 改写及跨语言翻译改写等五种难度递增的对抗性复用场景，为取证式抄袭与文本复用检测提供了系统性评测基准。 |
| [^43] | [A State Change Is Local, Its Consequences Are Not: Dependency-Aware State Revision for Long-Term Agent Memory](https://arxiv.org/abs/2611.02912) | 提出依赖感知状态修订问题，并通过 StateGraph 从交互历史中恢复状态依赖图，利用反事实验证区分真实的支撑依赖与语义相关性，从而在上游事实变化时自动失效失去支撑的下游信念、计划和行动。 |
| [^44] | [User Attributes and Political Ideology Shape LLM Receptiveness in Online Political Debates](https://arxiv.org/abs/2611.02908) | 该研究通过近15万次受控多轮对话实验发现，大语言模型在在线政治辩论中对左倾用户表现出系统性高于右倾用户的接受度，且这种政治不对称性即便在不同用户为同一立场辩护时依然存在。 |
| [^45] | [Maswao: An automatically collected, prepared, and benchmarked dataset for the South African Sign language](https://arxiv.org/abs/2611.02903) | 本文提出了一个以最少人工干预自动收集、预处理和基准化南非手语（SASL）数据集的框架Maswao，利用新闻广播等公共资源和语音识别技术，旨在弥合全球手语处理技术发展的差距。 |
| [^46] | [Tug of Watts: A study on the Energy - Accuracy Trade-off of Large Language Models on Medical Question Answering Tasks](https://arxiv.org/abs/2611.02889) | 本研究通过对13个不同规模的大型语言模型在医疗问答任务上的能耗与准确率分析，提出了系统构建和解读能耗-准确率联合权衡指标的方法，强调仅分别报告这两项指标不足以支撑严谨的权衡评估。 |
| [^47] | [Omni-Life: Benchmarking Omnimodal Real-Time AI Assistants in People's Daily Lives](https://arxiv.org/abs/2611.02835) | 提出基于真实用户需求的Omni-Life基准来评估实时全模态AI助手在日常生活中的表现，发现现有主流模型最多只能完全正确处理4.48%需要帮助的情境。 |
| [^48] | [Window-Aware Scheduling for Health Messaging under Platform Delivery Constraints: Evidence from an AI-Powered WhatsApp Maternal and Infant Health Chatbot in Nigeria](https://arxiv.org/abs/2611.02834) | 该研究首次量化了WhatsApp 24小时客户服务窗口对健康聊天机器人消息送达造成的损失，并将窗口感知调度形式化为决策问题，发现将关键医疗咨询服务以预批准模板发送可将送达率从几乎为零提升至约81.5%–86.8%。 |
| [^49] | [TRIO: Task Refinement with Individual Optimization for Retrieval-Augmented Generation](https://arxiv.org/abs/2611.02810) | 该论文提出TRIO方法，通过对齐三个智能体的个体优化目标并采用迭代细化范式，解决了检索增强生成中检索不充分和幻觉的问题。 |
| [^50] | [Creating Hybrid Interactive Narrative Systems with LLM: From Formal Description to Evaluation](https://arxiv.org/abs/2611.02807) | 提出了一个基于大语言模型的混合式交互叙事平台，通过让可互换的体验管理者模块运行于共享的符号世界模型之上，实现了对不同管理者的统一分析与评估。 |
| [^51] | [SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents](https://arxiv.org/abs/2611.02797) | 该论文提出 SelfImprove-Bench，一个针对递归自我改进智能体的评估基准，将任务组织为探索、规划、推理三大核心能力场景，并基于游戏实现可控、可复现且可扩展的评估，以解决现有评估无法区分真实能力增益与场景过拟合的问题。 |
| [^52] | [PGuard: A Native, Streaming, and Plug-in Guardrail for Multimodal Large Language Models](https://arxiv.org/abs/2611.02791) | PGuard 是一种面向多模态大语言模型的原生、流式、即插即用安全防护栏，它通过动态层门控直接利用骨干模型的中间表示实现 token 级安全检测，无需改变模型的原始推理路径即可将安全监控内化为模型自身能力。 |
| [^53] | [Stop Handcrafting Your Engine! GenDB-V: Towards AI-Generated and Formally Verified Database Systems](https://arxiv.org/abs/2611.02779) | GenDB-V提出利用AI智能体自动生成针对特定硬件和工作负载优化、并经形式化验证保证正确性的数据库引擎，以解决数据库系统手工开发成本高昂和正确性难以保障的两大长期难题。 |
| [^54] | [EffComp: Efficient Prompt Compression via Hybrid Reinforcement & Supervised Learning](https://arxiv.org/abs/2611.02766) | 提出了EffComp框架，通过结合BERT风格的重排序与句子选择模型以及混合强化学习和监督学习的训练方式，为基于RAG的问答任务实现无幻觉、低延迟且在压缩比与准确率之间达到最佳平衡的句子级抽取式提示压缩。 |
| [^55] | [Replay for LLM Continual Learning with Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2611.02746) | 本文提出持续多教师在线策略蒸馏方法，在每个训练阶段以上一阶段的学生模型作为教师重放早期领域的提示，从而无需同时保留所有领域专家模型或存储大量快照即可缓解灾难性遗忘。 |
| [^56] | [MicroMouse: Masking Harmful Supervision in Long CoT via Relative Potential](https://arxiv.org/abs/2611.02710) | 提出 MicroMouse 自动化选择性 SFT 框架，通过引入相对势能（REP）来对比标准答案与匹配对照样本，衡量各推理状态的相对支持度，从而自动识别并屏蔽长思维链微调中的有害监督信号。 |
| [^57] | [LLMs Cannot Be Neutral: Gender Sensitivity Across Candidates, Evaluators, and Models](https://arxiv.org/abs/2611.02692) | 该研究通过对八个模型在市议会投票任务中的30,720次判断分析，发现大语言模型在评估候选人时无法保持性别中立，明确的性别标签会显著影响其投票决策，且不同模型之间的决策行为存在显著差异。 |
| [^58] | [Train the Sampler You Run: Few-Step Masked Diffusion Language Models via Mixture-Likelihood Fine-Tuning](https://arxiv.org/abs/2611.02666) | 提出混合似然微调方法，通过在模型自身采样器似然的重要性加权界上进行LoRA微调，使掩码扩散语言模型在少步采样下生成质量超越原始模型，且微调计算量仅为预训练的0.002%。 |
| [^59] | [Constructing an Artworld Influencers Network by Mining Social Media](https://arxiv.org/abs/2611.02657) | 该论文提出一种结合社交媒体挖掘与自然语言处理的方法，能够从少量典型艺术界参与者出发自动发现新的重要艺术界参与者并构建其网络图谱，利用Louvain方法检测社区、HITS中心性方法评估参与者权威性。 |
| [^60] | [Variable names provide priors: semantic grounding enables data-efficient numerical prediction by large language models](https://arxiv.org/abs/2611.02637) | 变量名与数值之间的语义匹配（语义接地）使无需任务特定训练的大语言模型在样本稀缺的数值预测任务中匹敌甚至超越监督机器学习模型，其关键在于LLM能调用预训练中获得的领域特定知识。 |
| [^61] | [SYNC: A Training-Free Inference Controller for Dynamic Layer Selection in Multi-Persona Alignment](https://arxiv.org/abs/2611.02636) | SYNC是一种免训练的推理控制框架，通过余弦相似度轨迹动态估计各层对每个人格的敏感性，从而在每个解码步骤中将人格向量注入最优层，解决了多人格对齐中静态层选择和离线搜索成本高昂的问题。 |
| [^62] | [ICJ-Graph: A Citation Graph and a Layered Document Corpus of the International Court of Justice](https://arxiv.org/abs/2611.02630) | 本文构建了ICJ-Graph，一个将国际法院文件与其引用的案例、法律条款和联合国决议相连接的机器可读引文图谱与分层文档语料库，并通过引文预测任务验证了其在法律信息检索中的应用价值。 |
| [^63] | [Scalaris: A Scalable Enterprise Memory Architecture for Conversational AI Assistants](https://arxiv.org/abs/2611.02612) | Scalaris是一种通过统一的长期与短期记忆设计，解决企业级对话AI系统中检索准确性与上下文效率之间权衡问题的可扩展记忆架构。 |
| [^64] | [They Know Where It Is, But Not What It Says: A Bilingual Hallucination Benchmark of Hadith Recall in Open LLMs](https://arxiv.org/abs/2611.02587) | 本研究构建了英阿双语基准，测试五个开源大模型在六部逊尼派正典圣训集中根据出处召回圣训原文的能力，揭示了这些模型在可验证的高风险宗教文本召回任务中存在严重的幻觉问题。 |
| [^65] | [From Language to Country to Valence: A Multilingual Audit of Stereotype Bias Across LLMs](https://arxiv.org/abs/2611.02585) | 本研究使用12,000条四种语言本地撰写的Reddit帖子对六个大语言模型进行性别、种族和宗教偏见审计，发现宗教偏见在所有条件下最严重，且偏见程度因语言而异——性别偏见在西班牙语中最强，种族和宗教偏见在德语中最强。 |
| [^66] | [SMILE: A Structured Multi-Agent Clinical Reasoning for Liver Ultrasound Understanding](https://arxiv.org/abs/2611.02581) | 该论文提出SMILE框架，通过多个互补的专科智能体解读肝脏超声图像，并利用裁判引导的共识机制整合结果，从而克服单一推理路径的局限，提升肝脏超声报告生成的质量。 |
| [^67] | [Re$^3$Rank: Listwise Reranking with Reasoning-Conditioned Retrieval Heads](https://arxiv.org/abs/2611.02550) | 提出基于推理条件化检索头的列表式重排序器Re³Rank，通过直通估计器联合训练推理生成器与评分器，其4B模型平均超越最强基线3.11个NDCG@10点，0.6B模型即可胜过32B参数的重排序器。 |
| [^68] | [FindLink: Learning to Assess Scientific Connections from Source-Grounded Premises](https://arxiv.org/abs/2611.02545) | 该论文提出 GI-TraceBench 基准与 FindLink 两阶段框架，通过构建来源锚定前提并进行结构化推理，结合确定性编译器反馈，来评估不同研究条件下科学发现之间的关联。 |
| [^69] | [PixelBench: Can LLMs Understand and Generate Images Pixel by Pixel?](https://arxiv.org/abs/2611.02540) | 该论文提出PixelBench基准，证明大语言模型无需专门的图像分词器或解码器，仅凭现有词表即可直接生成可识别的像素图像，且像素理解能力与生成能力密切相关。 |
| [^70] | [Tune the Baseline First: Most of Muon's Scaled Weight Decay Gain Is a Tuning Gap, but Its Small Real Advantage Grows With Model Size](https://arxiv.org/abs/2611.02529) | 该研究在公平联合调优学习率与权重衰减后发现，Muon缩放权重衰减约90%的训练效率优势其实是基线调优不足造成的差距，仅剩约1.03倍的真实微小优势，且该优势随模型规模增大而增长。 |
| [^71] | [Ranked but Not Stopped: Detecting Prompt-Injection Hijacks from Typed Decision Models' Output Probabilities](https://arxiv.org/abs/2611.02522) | 基于输出概率低置信度的提示注入劫持检测方法只能有效“排序”而难以“拦截”——在 AG News 分类实验中，即使按 5% 误报率校准阈值，监视器仍会漏掉 51-70% 的劫持，说明置信度信号不足以构建可靠的防御门控。 |
| [^72] | [Same Sentence, Different Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers](https://arxiv.org/abs/2611.02491) | 该研究通过仅替换身份词的对照模板实验发现，尽管LLM仇恨言论分类器总体准确率超过0.95，但部分模型对针对女性的仇恨言论漏检率显著更高，且对完全相同的句子因身份不同而采用不同的判定构念，首次在LLM时代揭示了其测量非不变性问题。 |
| [^73] | [Chain on Demand: Deferred Evidence Organization for Long-Term Conversational Memory](https://arxiv.org/abs/2611.02480) | 提出按需链（CoD）框架，将证据组织推迟到查询时刻进行，在不修改底层对话记忆的前提下动态构建临时证据链，在LoCoMo基准上显著提升了长期对话记忆问答的准确率。 |
| [^74] | [GigaSpeech 3: Scaling Multi-Dialect Arabic Speech Recognition with Weakly Labeled Data](https://arxiv.org/abs/2611.02476) | 研究者构建了包含39,000小时弱标注数据的大规模多方言阿拉伯语语音语料库GigaSpeech 3，通过自监督学习与噪声学生训练策略，使方言专用语音识别模型的宏平均词错误率达到23.86%，相对最强第三方模型降低16.6%。 |
| [^75] | [As Expected? Efficient LLM Planning via Expectation-Observation Consistency Verification](https://arxiv.org/abs/2611.02438) | 本文提出期望-观察一致性验证（EOCV）机制，通过将批处理执行器对动作结果的预期与实际观察结果比对，一致则继续执行、不一致则提前终止并重新规划，在保持任务性能的同时将推理成本降低29-35%。 |
| [^76] | [Diagnosing Model--Item Interactions in Language Model Benchmarks with LSIRM](https://arxiv.org/abs/2611.02432) | 本文利用潜在空间项目反应模型（LSIRM）将语言模型在基准测试中与题目之间的交互模式可视化，发现图谱上相邻题目常共享相同正确答案，且使用选项标签评分比使用选项文本评分时这一模式更为显著。 |
| [^77] | [InteractionBench: Measuring Timing, Task Accuracy, Whisper and Volume, and Grounding in Real-Time Voice Systems](https://arxiv.org/abs/2611.02430) | 提出了统一测量实时语音系统对话时序、口语任务准确性、耳语音量控制和长对话接地能力四个维度的基准测试 InteractionBench，并对 GPT-Live、Gemini Live、Moshi 等七个系统进行了 1,519 次评分运行评估。 |
| [^78] | [When Rewards Depend on the Group, GRPO Plays a Tournament](https://arxiv.org/abs/2611.02404) | 该论文证明GRPO的组相对基线会把组依赖奖励（如多样性奖励）的优化目标扭曲为零和锦标赛，导致模型感知不到甚至负向对待组内同伴的奖励，并提出α=+1的Team-GRPO无偏估计器，只需每个提示两次采样即可正确优化任何组依赖奖励。 |
| [^79] | [Solver-Like Behavior in Large Language Models as Formalizers](https://arxiv.org/abs/2611.02395) | 该研究首次定义并量化了“大语言模型作为形式化器”中的“求解器式行为”：85%的推理过程在程序执行前就完成了求解器的工作，占用24%的推理token，但由于程序通常会重新计算答案，这些努力大多是冗余的。 |
| [^80] | [EviCast: Forecasting Retrieval Gains Through Shared Response Calibration](https://arxiv.org/abs/2611.02387) | EviCast通过一次性测量生成器对受控证据状态的响应，并将其与各候选检索器的离线证据状态分布相结合，无需实际执行每个RAG配置即可预测检索的端到端净效用及置信下界。 |
| [^81] | [Beyond Accuracy: How Quantization Affects Coding Agents](https://arxiv.org/abs/2611.02368) | 量化虽然不会显著降低编程智能体的最终准确率，但会实质性地改变其解题行为和失败模式，表明仅凭准确率不足以评估量化对编程智能体的影响。 |
| [^82] | [VisCodeBench: a new benchmark dataset for Text-to-Vis reflecting real-world practice](https://arxiv.org/abs/2611.02353) | 该论文提出VisCodeBench，一个包含2,672个来自教育、科学和大众受众领域样本的端到端文本到可视化基准数据集，评估发现现有LLM可视化系统虽能基本成功执行代码，但在复现参考数据方面表现不佳，最佳数据相似度仅为0.81。 |
| [^83] | [Which Words a Model Fears: From a Refusal Direction to a Vocabulary-Wide Fear Lexicon](https://arxiv.org/abs/2611.02348) | 该研究发现语言模型的拒绝行为部分由词语本身驱动，并提出用“拒绝方向”作为标尺为整个词表构建“恐惧词典”，证明仅向良性提示添加一个恐惧词就足以触发模型拒绝。 |
| [^84] | [MemShift: Separating Detection from Repair in LLM-Derived Personal Memory](https://arxiv.org/abs/2611.02338) | 该论文提出了MemShift基准测试，通过五种转变类型、分叉式合成世界和确定性评分器，将LLM衍生个人记忆的“检测”（发现输入变化）与“修复”（生成正确新值）两个环节分离评估，从而精确定位记忆维护失败的具体步骤。 |
| [^85] | [Outdated numeric values from Norwegian statutes and regulations in a further-trained model and its base model: a preregistered paired comparison of Borealis 2 and Gemma 4](https://arxiv.org/abs/2611.02334) | 一项预注册的配对比较研究表明，经挪威国家图书馆进一步训练的 Borealis 2 与其基础模型 Gemma 4 在回答挪威法律法规数值时，给出过时信息的比例并无显著差异。 |
| [^86] | [When Does Adaptive Halting Win? Selection Rules Matter in Looped Language Models](https://arxiv.org/abs/2611.02333) | 该论文通过预注册实验证明，在比较循环语言模型的自适应停机方法时，选择工作点的规则决定了比较结果，且在经自助法筛选的 1 个百分点准确率损失规则下，训练的稳定性探针比原生门控节省了更多计算预算。 |
| [^87] | [LEANGUARD: Lean-Verified Guardrails for Tool-Using Agents](https://arxiv.org/abs/2611.02332) | LeanGuard 是一个基于 Lean4 形式化验证的工具调用运行时防护栏，能在违规操作执行前进行拦截并通过反馈或证明引导修复予以纠正，实验证明它可将违规调用降为零且不显著影响任务成功率。 |
| [^88] | [Accessed, never disclosed: Silent access to protected health information by medical artificial intelligence agents](https://arxiv.org/abs/2611.02319) | 该研究通过在合成病历中嵌入金丝雀标记，首次揭示了医疗AI代理会在任务不需要时“静默访问”患者受保护健康信息——这种行为对仅检查输出的评估完全不可见——并证明基于任务必要性的预执行门控可几乎完全消除这种隐蔽数据访问。 |
| [^89] | [A Capitulation Direction Emerges With Scale and Is Causally Load-Bearing Where Validation Is Strong: Eight Models, Two Families, and Five Measurement Hazards](https://arxiv.org/abs/2611.02318) | 该研究在两个家族八个指令微调模型上通过预注册的严格验证协议发现，模型在用户反驳下放弃正确答案的谄媚行为对应残差流中的一个线性“屈服方向”，该方向仅在7B及以上规模的模型中涌现并通过验证，消融该方向可因果性地将Qwen2.5-7B推翻正确答案的比率降低约四分之一。 |
| [^90] | [GRIPS: An Empirical Study of Pre-SQL Grounding Information for SQL Generation](https://arxiv.org/abs/2611.02314) | 该论文提出GRIPS三阶段方法，通过在SQL生成前显式提取结构化的接地信息，在两个基准上20个设置中的18个实现平均11.8个百分点的准确率提升，同时将token使用量降低47.8%。 |
| [^91] | [Concise-9B: Removing the Slack from a Reasoning Model's Thinking](https://arxiv.org/abs/2611.02278) | 提出Concise-9B模型，通过简洁自训练方法（两轮策略上拒绝采样微调并保留最短正确样本）去除推理模型思维链中的冗余部分，在保持准确率等效的前提下将生成token量降至原模型的0.66倍。 |
| [^92] | [When the Text Is Silent: Memory Shortcuts and the Validity of LLM Annotation under Evidence Insufficiency](https://arxiv.org/abs/2611.02273) | 该研究构建了分离文本证据与先验知识的反事实文本，发现大语言模型在文本与记忆冲突时不会用记忆覆盖文本，但当文本证据缺失时会依赖对知名人物的先验记忆产生“记忆捷径”，从而影响LLM标注的有效性。 |
| [^93] | [Current state of LLMs for Arabic dialectal machine translation](https://arxiv.org/abs/2611.02271) | 该研究在MADAR数据集上系统评估了包括阿拉伯语专用模型、多语言模型和商业API在内的多种大语言模型在英语与16种阿拉伯语方言之间双向机器翻译的表现，并探讨了方言机器翻译自动评估的挑战以及LLM的方言识别能力。 |
| [^94] | [Hinted Scaling Laws](https://arxiv.org/abs/2611.02270) | 提示式缩放定律利用特权信息将基准测试转化为难度连续的简化版本，让小模型也能获得非零分数，从而在噪声阈值以下实现平滑可预测的缩放外推，将 $10^{23}$ FLOP 大模型 GSM8K 准确率的预测误差控制在 3 个百分点以内，远优于标准缩放定律的 13 个百分点误差。 |
| [^95] | [Metamorphic Testing for Evaluating Robustness of Sentiment Analysis Models](https://arxiv.org/abs/2611.02248) | 本文提出了一种轻量级的、基于批处理的变形测试方法，通过检验原始输入与变换后输入之间预期关系的一致性，来系统性地评估情感分析模型的鲁棒性。 |
| [^96] | [Can AI Reviewers See Your Efforts to Improve Your Paper? A Large-Scale Study of Paper Revision Trajectories](https://arxiv.org/abs/2611.02244) | 研究发现AI审稿人在论文评审中往往无法识别作者通过修改做出的重大改进，对大多数标注确认的重大修改保持评分不变，这揭示了AI评审系统在评估论文改进方面的严重局限性。 |
| [^97] | [Who Should Score the Cache? Mid-Depth Layers Are the Best Judges of KV-Cache Eviction](https://arxiv.org/abs/2611.02230) | 该论文发现KV缓存淘汰质量由打分层的深度决定——中深度层（约0.4至0.75深度处）是评判token的最佳层，并提出一种基于答案存活概率的校准方法来自动选出最佳打分层。 |
| [^98] | [Test-Time Scaling for Open-Ended Reasoning at Small Model Scale](https://arxiv.org/abs/2611.02208) | 该研究提出无需裁判模型的语义聚类投票方法，在小规模模型上发现测试时投票能将数学推理准确率提升10至29分，但对开放式自由文本生成几乎没有收益，说明测试时扩展对开放性文本的价值取决于模型能否产生值得聚合的样本。 |
| [^99] | [Decision Models as Policy Gates for Coding Agents: Explicit Policy Boundaries Improve Detection and Reduce False Alarms](https://arxiv.org/abs/2611.02155) | 以示例和反例形式编写的显式策略边界能显著提升决策模型作为编码代理运行时策略门控的风险检测召回率（提升5.5-26.1个百分点），同时大幅降低误报率。 |
| [^100] | [Multi-Agent Collusion and the J-Space](https://arxiv.org/abs/2611.02153) | 该论文首次将J空间分析应用于多智能体系统，发现合谋智能体的J空间在回复前就已表征合谋行为本身（如虚假陈述、隐瞒或悔意），并据此提出了首个无监督智能体合谋检测方法J-PACT。 |
| [^101] | [Detailed Derivation of The Backpropagation Algorithm](https://arxiv.org/abs/2611.02152) | 本文以三层网络为例逐步详细推导了反向传播算法，通过链式法则和路径求和导出矩阵形式的四个通用反向传播方程，并将其扩展到多样本矩阵运算与小批量梯度下降。 |
| [^102] | [GatedV: Self-Gated Values for Transformer Language Models](https://arxiv.org/abs/2611.02145) | GatedV是一种无需额外可训练参数的简单注意力变体，通过仅对值矩阵V在聚合前应用SiLU激活函数，即可在多个参数规模下稳定地将语言模型困惑度降低约2.0%至2.4%。 |
| [^103] | [HaloGuard 2.0: An Open-Weights Constitutional Output Guard for Multilingual Response Safety](https://arxiv.org/abs/2611.02134) | 提出开放权重的流式输出安全守卫HaloGuard 2.0，可在单次前向传播中为每个响应token实时判定安全/不安全概率，从源头拦截生成过程中才显现的危害内容，并揭示了传统数据训练会导致守卫只看提示词不看响应的流式捷径问题。 |
| [^104] | [FOLLIE: Automated Reasoning over the First Dataset of Sentences with their First-Order Logical Representation for French](https://arxiv.org/abs/2611.02129) | FOLLIE是首个面向法语NLI的自然语言到一阶逻辑资源数据集，包含近2万条推理记录，通过神经-符号流水线构建并显著提升了逻辑推导覆盖率，实现了对法语句子的一阶逻辑自动推理。 |
| [^105] | [Conditional Calibration Contracts under Coupled Load: A Branch-Aware Audit of Switched Inference](https://arxiv.org/abs/2611.02121) | 该论文提出 CA-MHD 框架，将条件校准契约视为待验证的规范而非已认证属性，通过负载耦合蒸馏、闭环重放和成本感知施密特路由器在离散事件模拟器中进行分支感知审计，并发现常见的教师轨迹审计无法认证学生模型在切换推理中所选部署支持的校准性。 |
| [^106] | [Order-Only On-Policy Distillation](https://arxiv.org/abs/2611.02112) | 提出仅顺序同策略蒸馏（O³PD），只让学生模型学习教师模型 top-K 词元的排序而不拟合概率值，因此对教师温度不变、无需温度调节，并在多个教师-学生组合上优于传统同策略蒸馏。 |
| [^107] | [Consolidating Reasoning with Test-Time Training](https://arxiv.org/abs/2611.02104) | 该论文提出TTT-Consolidate框架，利用测试时训练将多智能体协作中产生的推理轨迹整合为参数化表示，从而改进模型后续的推理能力。 |
| [^108] | [ParetoSaddler: Pareto-Guided Optimization of Agent Harnesses for Quality and Cost](https://arxiv.org/abs/2611.02079) | 提出ParetoSaddler方法，通过将中间候选晋升与最终部署选择分离，并借助贝叶斯自助帕累托支持机制，在无需修改模型权重的情况下实现LLM智能体框架的质量与成本联合优化。 |
| [^109] | [How Green Is Your Manifesto? Scoring Environmental Positions in the 2024 EU Elections with an LLM Panel](https://arxiv.org/abs/2611.02033) | 该论文提出用三个来自不同提供商的大语言模型组成评审小组，对2024年欧洲议会选举中41份政党宣言逐句在环境保护与气候变化两个维度上自动评分，从而提供一种比专家调查更廉价、可复现的政党环境立场测量方法。 |
| [^110] | [ClickSight: An LLM Framework for Interpreting Student Clickstreams Through Learning Strategies](https://arxiv.org/abs/2611.02021) | ClickSight是一个基于大语言模型的分阶段框架，能够依据文献中定义的学习策略自动解读学生点击流数据，并生成高质量、有证据支撑的学习行为解读。 |
| [^111] | [CULTURE-CMT: Benchmarking Contextualized Social Media Translation](https://arxiv.org/abs/2611.02018) | 提出了CULTURE-CMT，一个包含572个人工验证示例的中英文社交媒体评论情境化翻译基准，通过恢复帖子与对话上下文并配备检索增强的文化有效性评估器，发现提供交互上下文能提升大多数模型的表现，但无法完全消除文化相关的翻译难题。 |
| [^112] | [Switch-SWE: Do Coding Agents Hold Focus When Switching Between Unfinished Tasks?](https://arxiv.org/abs/2611.02017) | 该论文提出了Switch-SWE基准，通过在同一仓库中交错多个未完成任务来评估编程智能体在任务切换后能否保持专注、恢复原任务的需求与进度并成功完成任务。 |
| [^113] | [Scaling Makes It Possible: How Large Models Master Impossible Languages](https://arxiv.org/abs/2611.02015) | 该论文提出“语义连贯性”概念，揭示出十亿参数级的大语言模型能够像掌握可能语言一样掌握违反人类语言约束的“不可能语言”，而百万参数级小模型则显著失败，且性能差距随语义复杂度增加而扩大。 |
| [^114] | [MedPsy: State-of-the-Art Small Medical Language Models for Efficient Edge Deployment](https://arxiv.org/abs/2611.02011) | MedPsy是一系列专为边缘设备部署设计的1.7B和4B小型医学语言模型，通过合成医学数据流水线、四阶段后训练课程（两个SFT加两个带难例挖掘的RL阶段）以及GGUF量化边缘部署评估，在精度、生成效率与部署可行性之间实现了最优平衡。 |
| [^115] | [Retrieval-Enhanced Cultural Evaluation for Social Media Translation Training](https://arxiv.org/abs/2611.02009) | 本文提出CUE-MT社交媒体翻译模型系列，通过Agent按需检索外部文化证据并提炼为提示来增强校准后的Judger，从而改进监督数据精炼与强化学习奖励信号，在文化有效性上比同规模开源基线提升7.44至15.12个百分点。 |
| [^116] | [To Align Long, Plan Short: Simulating Diverse Profile-Aligned Students by Realizing Short-Term Plans in Long-Horizon Learning Environments](https://arxiv.org/abs/2611.02002) | 提出了ALPS分层模拟学生框架，通过交替进行“画像对齐的任务片段规划”和“动作生成”，使模拟学生在长时程开放式学习环境中持续符合指定的行为画像。 |
| [^117] | [TrackMem: Tracking Evolving Beliefs in Long-Horizon Financial Dialogue](https://arxiv.org/abs/2611.02001) | 提出TrackMem——一种按说话者组织的图结构记忆系统，能够追踪金融对话中被修订的数字、跨会话演变的立场以及一次陈述即约束后续问题的规则，并配套发布FinLoCoMo长程投资对话基准来检验动态信念追踪能力。 |
| [^118] | [Better Answers, Weaker Evidence Control: Evidence Bypass in On-Policy Distillation](https://arxiv.org/abs/2611.01967) | 该研究发现在线策略蒸馏虽能提升答案准确率，却会引发“证据绕过”现象——学生模型在文档证据被修改后仍坚持原答案，揭示出蒸馏过程会削弱模型对支持证据的依赖。 |
| [^119] | [Towards Self-Evolving Narrative Systems: Real-Time Adaptation and Long-Term Optimization with LLMs](https://arxiv.org/abs/2611.01954) | 提出了一种通过语义相似度对齐来动态平衡预编写内容与LLM实时生成的混合叙事框架，并利用反馈驱动的自我演化机制自动优化提示词，在无需人工提示工程的情况下同时提升叙事连贯性和玩家自主性。 |
| [^120] | [Tracing How Language Models Integrate Temporal Anchors into Factual Recall](https://arxiv.org/abs/2611.01945) | 该论文提出“锚定事实回忆”新任务，通过对四个开源模型进行因果干预，揭示了语言模型整合时间锚点的两条机制路径——锚点先解析主语词元表示为查询年份再由关系重新分配读出的主语介导路径，以及最终位置注意力直接参与选择的直接路径。 |
| [^121] | [TAC-CoT: Tool-Empowered Implicit Chain-of-Thought via Interleaving Explicit Tokens](https://arxiv.org/abs/2611.01926) | TAC-CoT提出一种工具赋能的自适应隐式思维链框架，用紧凑的潜在推理片段替代显式推理文本、仅在工具调用、观察与最终答案处保留文本标记，从而在智能体任务上实现准确率最高提升8.8个百分点、前向传递次数最多减少6.8倍。 |
| [^122] | [Don't Guess, Ask Wisely: Benchmarking and Training Proactive Elicitation in Open-Domain LLM Conversations](https://arxiv.org/abs/2611.01878) | 本文提出VagueBench基准数据集与DialogGRPO离线强化学习方法，通过将多轮询问建模为对话状态有向无环图并结合即时信息增益与前瞻价值的密集轮次级奖励，训练大语言模型在面对模糊查询时主动提出有针对性的问题。 |
| [^123] | [Quantizing Recurrent Language Models Is Easier Than You Think](https://arxiv.org/abs/2611.01864) | 该论文首次系统研究了循环语言模型的量化设计空间，发现衰减型线性递归相比softmax注意力产生的异常通道更弱，因此无需特殊处理就能更容易地被量化。 |
| [^124] | [Self-Evolving Agentic Workflows via Semantic Decomposition of Complex Tasks](https://arxiv.org/abs/2611.01840) | 该论文提出了一个通过语义任务分解实现智能体工作流自演化的框架，智能体利用执行反馈不断优化任务分解并更新子任务流程，其中大多数最终决策流程由代码而非大语言模型实现，从而显著降低了重复推理的成本与延迟。 |
| [^125] | [LegoEnv: Self-Evolving Agentic Environments via Modular Recombination](https://arxiv.org/abs/2611.01834) | LegoEnv提出了一种将环境由可复用原子（意图、规则、数据、服务器）组合构建、并在环境反馈不可靠时自我进化这些原子的新范式，构建了5,465个经过验证的环境，将可解环境比例从71.6%提升至88.3%，并显著提升了智能体的SFT与RL训练效果。 |
| [^126] | [The Scope Grammar of Natural-Language Rules: Counting and Enumerating Readings modulo Logical Equivalence](https://arxiv.org/abs/2611.01825) | 该论文提出“作用域文法”来刻画自然语言规则中连接词作用域的歧义性，通过读一次范式与双标记递推，精确计数并枚举了连接词骨架在模逻辑等价意义下逻辑上不同的读法数量。 |
| [^127] | [Shared relations, specific choices: conditional sensitivity and competing pathways in language models](https://arxiv.org/abs/2611.01809) | 该研究揭示语言模型中存在跨查询共享且可迁移到新实体的低维关系表征，但其功能符号并非必然可迁移——相同的关系向量编辑在不同类型提示下会产生相反的选择效应。 |
| [^128] | [Rubric-Based Assessment of Student Reports Using Small Language Models](https://arxiv.org/abs/2611.01785) | 该研究提出一个面向低资源环境且保护隐私的自动化评估流水线，利用小型语言模型和视觉-语言模型依据量规评估学生在网络安全事件响应演习中的报告，纯文本评估准确率最高可达97%。 |
| [^129] | [MemRole: From Memories to Decisions for Long-Horizon Reasoning in Role-Playing Agents](https://arxiv.org/abs/2611.01778) | 提出MemRole框架，通过长程记忆思考（LHMT）将隐式推理显式分解为上下文理解、记忆激活和响应策略规划，使角色扮演智能体能够利用基于角色且受时间线约束的记忆池做出角色一致的长程决策。 |
| [^130] | [A Bounded Evaluation Protocol for Representational Systematicity](https://arxiv.org/abs/2611.01755) | 该论文提出了一个针对表征系统性的有界评估协议，通过在匹配的行为对比上同时检验所声明表征的可恢复性与因果使用，并对解释赋予四种有界状态之一，从而解决了单一类型证据容易误导的验证难题。 |
| [^131] | [Evolving Memory, Learning to Meta-Control: Test-Time Adaptation for Experience-augmented LLM Agents](https://arxiv.org/abs/2611.01747) | 提出了OMCE框架，通过轻量级上下文老虎机元控制器在测试时自适应过程中动态决定是否及如何调用和固化经验记忆，实现了LLM智能体经验的在线演化与最优部署。 |
| [^132] | [Less but Sufficient: Relation Knowledge Selects Compact Knowledge-Graph Contexts, and What External Checks Can and Cannot Catch](https://arxiv.org/abs/2611.01733) | 该论文提出基于关系知识从知识图谱中筛选出足以推导答案的紧凑事实上下文，并辅以可推导性检查、一致性检查和路由门控等外部检查机制，使开源LLM在CWQ上仅用四分之一提示token即可达到与SubgraphRAG相当甚至更优的精确匹配性能。 |
| [^133] | [MAC-Bench: Benchmarking Multi-Agent Collaboration via Interconnected Tasks](https://arxiv.org/abs/2611.01728) | 本文提出MAC-Bench基准，通过将协作需求直接嵌入任务依赖关系和共享资源约束中，而非强加预定义的智能体角色或通信协议，来全面评估基于大语言模型的多智能体系统的协作能力。 |
| [^134] | [EarlySelectKV: Anticipatory KV Selection via Cross-Layer Query Proxies](https://arxiv.org/abs/2611.01715) | EarlySelectKV利用较早层的隐藏状态作为代理查询来提前预计算KV索引选择，使稀疏KV路由脱离注意力计算前的关键路径，在几乎不损失质量的情况下将端到端输出吞吐量提升高达12.4%。 |
| [^135] | [CST: Lightweight Risk Triage with Source-Grounded Evidence for Web Discussions](https://arxiv.org/abs/2611.01710) | CST是一个可在普通CPU上运行的轻量级风险分流系统，它通过结合字符模式与固定多语言模型的句子含义来评估消息风险，并将相关消息分组后从各组依次选取，为版主提供保留原始措辞与回复的、基于源文本的证据。 |
| [^136] | [EEG-HealthLLM: Hierarchical Adaptation of Large Language Models for Unified Heterogeneous EEG Health Assessment](https://arxiv.org/abs/2611.01652) | 提出EEG-HealthLLM框架，通过非对称分层自适应策略——任务特定EEG适配器保留异构时空动态、共享LoRA注入LLM骨干网络——实现统一且参数高效的异构脑电健康评估。 |
| [^137] | [CREDO: Evidence-Conditioned Credit Assignment for Temporal Reasoning Agents](https://arxiv.org/abs/2611.01643) | 提出CREDO方法，通过利用中间证据恢复局部监督并进行阶段级信用再分配，解决了GRPO中轨迹级结果信号无法区分中间决策质量、且全失败组监督信号消失的问题。 |
| [^138] | [BanglishDepYT: Depression Detection in Native Bengali Social Media with Emotion-Preserving Preprocessing and Domain-Adaptive Pretraining](https://arxiv.org/abs/2611.01633) | 该论文发布了首个从YouTube原生收集的孟加拉语抑郁症检测数据集BanglishDepYT，并提出情感保留混合预处理方法EPH，证明预处理方式对非正式社交媒体文本的检测性能影响显著，EPH相比标准清洗方法将SVM F1从0.7062提升至0.8570。 |
| [^139] | [FinForesight: A Diagnostic Backtesting Benchmark for Deep Research Systems in Financial Forecasting](https://arxiv.org/abs/2611.01621) | 提出了FinForesight，一个面向金融深度研究系统的诊断性回测基准，通过在中国、美国和香港市场评估估值、收益排名、事件方向与财务基本面四项预测任务，并辅以反事实推理和金融计算两项能力诊断，在防止记忆泄露和“未来信息”偏差的同时，实现对预测能力的及时反馈与历史评分。 |
| [^140] | [LLM-Enhanced Multimodal House Price Prediction using Structured, Textual, and Visual Features](https://arxiv.org/abs/2611.01615) | 该论文提出了一个融合结构化数据、文本描述和图像的大语言模型增强多模态房价预测框架，利用LLM从文本中提取特定领域的时空和上下文特征，在保持可解释性的同时显著提升预测精度。 |
| [^141] | [Halal-NeSyNet: Clause-Scoped Neuro-Symbolic Cross-Attention for Automated Food Ingredient Auditing and Risk-Aware Halal Logistics Triage](https://arxiv.org/abs/2611.01612) | 该论文提出Halal-NeSyNet模型，通过将DistilBERT编码器与规则衍生符号向量进行子句范围的交叉注意力与门控融合，解决清真食品成分审核中跨子句修饰词混淆问题，并构建了包含3,199个条目的基准数据集以实现自动化食品成分审计和风险感知的清真物流分诊。 |
| [^142] | [Can Large Language Models Extend Their Agentic Capabilities to Goal Pursuit in Open-Ended Multi-Turn Interaction?](https://arxiv.org/abs/2611.01552) | 该论文提出了一种无需训练的工具介导交互（TMI）协议，通过将对话行动表示为工具调用来维持持续的推理-行动-观察循环，显著提升了大语言模型在社交互动、情感支持、说服和谈判等开放式多轮对话中的目标达成能力。 |
| [^143] | [Revisiting LLM Homogeneity Through a Multidimensional Lens](https://arxiv.org/abs/2611.01549) | 该研究提出用涵盖语义、词汇、句法、结构和风格呈现五个维度的25个可解释指标取代单一的嵌入相似度来衡量LLM同质化，发现18个当代模型“用不同的词说相似的话”，且2024至2026年间模型变得与自身更一致而非彼此更相似。 |
| [^144] | [The Structure of Disagreement in Emotion Annotation](https://arxiv.org/abs/2611.01539) | 该论文通过分析《老人与海》以及BRIGHTER和GoEmotions数据集中的情感标注，首次系统揭示了三种反复出现的分歧结构——情感类别的系统性重叠、标注者在标签宽泛度与类别边界处理上的差异、以及稀疏参与和弃权被误认为分歧。 |
| [^145] | [Write Facts the Way Models Read Them: Lifting the Two-Hop Curse](https://arxiv.org/abs/2611.01519) | 论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。 |
| [^146] | [VLMs Read Aloud: The Gap Between Recognizing and Using Visual Text](https://arxiv.org/abs/2611.01517) | 该论文发现视觉语言模型存在“读-用差距”——即使能逐字转录图像中的文本也无法据此答题，而把自己的转录文本反馈给自己即可近乎完全恢复答题能力，且识别与使用视觉文本在分辨率要求和模型规模扩展上表现不同。 |
| [^147] | [Sweep Less, Steer More: Using Activation Steering Geometry for Reduced Hyperparameter Search](https://arxiv.org/abs/2611.01496) | 本文提出利用计算成本低廉的可辨别性指数 d′ 作为几何度量，可靠地识别激活引导的最佳干预层，从而免除逐层超参数扫描，大幅降低表示引导方法的部署成本。 |
| [^148] | [TopoSteer: Topological Layer Selection for Activation Steering in LLMs](https://arxiv.org/abs/2611.01493) | TopoSteer 提出了一种基于拓扑数据分析（H_0 持续度）的大语言模型激活引导层选择方法，通过度量逐样本位移云的延展范围来选出最有效的引导层，弥补了以往方法只看激活可分离性而不实际测试引导效果的不足。 |
| [^149] | [Accounting for Irregular State Transitions through Belief Estimation in Online Medical Consultation](https://arxiv.org/abs/2611.01474) | 该论文通过实证研究发现真实在线医疗咨询具有上下文连续性低、不可预测性高的特点，并提出受部分可观测马尔可夫决策过程（POMDP）启发的PRISM框架，通过将潜在状态信念估计与动作选择解耦来应对对话轨迹漂移，从而提升临床分诊能力。 |
| [^150] | [See, Think, Use: Diagnosing Perception and Reasoning Failures in VLMs](https://arxiv.org/abs/2611.01467) | 该论文提出“看、想、用”协议，通过定位探针、文本替换探针和反事实图像对，在无需访问模型权重的情况下，将视觉语言模型的每个答案归因于感知、推理或感知使用，从而精准定位应改进的环节并指导测试时计算的分配。 |
| [^151] | [Can Large Language Model Agents Learn by Watching Tutorial Videos?](https://arxiv.org/abs/2611.01460) | 本文提出教程智能体学习（TAL）这一非参数化持续学习范式，通过两阶段对抗过程将教程视频整合为结构化的“教程书”记忆并借助目录树检索，使LLM智能体无需参数更新即可像人类一样从海量教程视频中持续学习。 |
| [^152] | [When Should an Agent Refresh? Adaptive Index Reuse for Efficient Long-Horizon Reasoning](https://arxiv.org/abs/2611.01429) | 该论文提出自适应索引复用（AIR），通过一个轻量控制器根据智能体回合的各阶段片段和最新工具事件，动态决定稀疏注意力索引器需要刷新的层数，并在预期计算预算下利用最终任务奖励进行学习，从而在长程推理中更高效地平衡计算开销与注意力选择的准确性。 |
| [^153] | [Towards Fine-grained and Robust Detection of Human-AI Co-authorship](https://arxiv.org/abs/2611.01427) | 该论文提出一个离散类别的文本溯源检测框架，将人机共著细分为不同级别的AI参与程度（如表面润色与话语结构更新），揭示了对抗性改写模糊检测决策边界的机制，并通过数据增强提升检测器的鲁棒性。 |
| [^154] | [FlyWheel: Attention Kernels Built for the TPU](https://arxiv.org/abs/2611.01426) | FlyWheel是一组专为TPU设计的快速注意力内核，通过重新表述softmax与三角矩阵求逆以减少向量计算、协调数据布局提升矩阵操作数复用并重叠独立计算，在softmax注意力上实现最高4.14倍加速并达到v6e峰值BF16吞吐量的81%，同时将Gated DeltaNet预填充加速最高3.61倍。 |
| [^155] | [Latent Harness: Compiling Agent Behaviors into Post-Block Activation Shifts](https://arxiv.org/abs/2611.01415) | 本文提出 Latent Harness，将外部框架条件或行为指令蒸馏为施加在冻结模型每个解码器块输出上的秩8激活偏移（仅0.03-0.05%参数），使智能体行为无需框架文本和额外token即可内化进模型，并在10个预注册设置中的8个恢复源框架收益。 |
| [^156] | [RAG-TRACE: Tracing Measurement Evidence and Diagnostic Completeness in AI-Generated ECG Reports](https://arxiv.org/abs/2611.01409) | 该研究提出RAG-TRACE方法，通过追踪测量证据发现关键词检索会在AI生成的心电图报告中引入大量缺乏患者特异性测量支持的不可靠诊断标准，而确定性数值验证规则能有效撤销这些错误标准应用，同时保留有真实依据的诊断。 |
| [^157] | [Learning by Doing: Autonomously Compiling and Curating Agent Skills from Verified Traces](https://arxiv.org/abs/2611.01386) | Zep Agent Skills是一个生产级服务，它仅以经过验证的任务尝试为证据，将智能体运行轨迹自主编译为带版本管理的程序性技能记忆，并通过准入控制与经验证的使用结果来治理技能的激活、分发与搜索排名。 |
| [^158] | [Information-Theoretic Design of Sparse Autoencoder-Based Steering](https://arxiv.org/abs/2611.01378) | 该论文提出了一个信息论框架，利用经验特征触发分布之间的Jensen-Shannon散度来分析SAE激活并构建有效的转向向量，从而更系统地设计对大语言模型行为的干预。 |
| [^159] | [XDecideBench: Evaluating Typed Decision Models and Confidence Policies Across Languages](https://arxiv.org/abs/2611.01370) | 该论文提出XDecideBench，一个覆盖122种语言的大规模基准，用于评估类型化决策模型与置信度策略，揭示了模型排名随语言和套件显著变化，且针对类型化决策微调的开源模型在网络文本较少的低资源语言中带来更大的准确率提升。 |
| [^160] | [Interpreting Classical Readability Formulae as Surprisal](https://arxiv.org/abs/2611.01359) | 该论文首次从理论上证明，Flesch-Kincaid等经典可读性公式的系数可被解释为惊讶度——在无记忆模型下，这类公式等价于逐词与逐句交叉熵之和，借助Zipf定律并作为参考文本处的线性化，每个系数可分解为每nat年级数尺度、Zipf斜率和参考句子长度，从而使“年级”成为惊讶度的一个量。 |
| [^161] | [Wrong Together: Shared Bias in User Simulation](https://arxiv.org/abs/2611.01354) | 该研究发现模拟器之间的分歧可以作为识别不可靠估计的信号（对群体间差异的预测效果更好，AUC达0.87），但模拟器共享的偏见对该信号不可见，不过当共享错误同等地影响两个被比较的群体时会在比较中相互抵消。 |
| [^162] | [Sentiment sensing: comparative analysis of multilingual sentiment dynamics](https://arxiv.org/abs/2611.01348) | 该论文提出“情感感知”框架，通过整合成分数据分析、函数型数据方法和非参数推断，解决了跨国多语言情感分析中的语言偏差、成分数据约束和时间动态检验三大难题，并在欧盟27国关于能源与人工智能的新闻数据上得到验证。 |
| [^163] | [Step-dLLM: Adaptive Step-aware Sparse Attention for Efficient Diffusion LLM Inference](https://arxiv.org/abs/2611.01320) | 该论文提出Step-dLLM，通过在去噪轨迹的锚点步骤周期性刷新块稀疏掩码、在每个局部一致区间内复用掩码，并根据不同注意力头的集中程度自适应分配稀疏预算，从而高效加速扩散大语言模型的推理。 |
| [^164] | [Are Fake Citations a Form of Reward Hacking? A Pre-Registered Study of Post-Training and Citation Cues](https://arxiv.org/abs/2611.01313) | 四项预注册实验表明，要求提供引用会显著提高后训练模型（而非基础模型）伪造引用的比率，且该效应在三个领域均成立，说明虚假引用是后训练模型奖励黑客行为的一种形式。 |
| [^165] | [What Do LLM Judges Actually Verify? Matched Controls and Record Gaps for Document Currency](https://arxiv.org/abs/2611.01311) | 本文通过构造引用被替换文档、继任文档、替换前文档及“被更新而未替换”文档的匹配对照答案，并在修订记录中故意移除链接，揭示出 LLM 评判器主要依赖文本内容而非修订记录来验证文档时效性——记录缺失时，各评判器仅凭文本反而捕获了更多失效引用。 |
| [^166] | [When LLMs Modify Themselves: Convergence, Values, and Stopping in Iterative Self-Improvement](https://arxiv.org/abs/2611.01304) | 该研究通过对六个主流大语言模型开展54个实验，首次实证考察了LLM迭代式自我修改系统提示词的行为，发现模型停止自我修改的时机主要由模型自身身份决定，而非被赋予的优化目标。 |
| [^167] | [CuriousTogether: Implications of Curiosity and Cultural Grounding in Multi-Agent LLM Interactions](https://arxiv.org/abs/2611.01301) | 提出CuriousTogether框架，证明好奇心驱动的多智能体LLM交互在八项任务中的七项上优于非好奇心设置，故事生成语义多样性提升最高达20.6%、刻板印象率降低14.9个百分点，且文化背景设定有利于创意和社会任务但可能损害精确推理。 |
| [^168] | [Character-Name Concentration in LLM Fiction: Measuring Displacement and Mitigating Repetition](https://arxiv.org/abs/2611.01284) | 提出无需微调的“推测式命名”方法，将推测解码的草拟-验证-拒绝范式从token迁移到角色命名，配合结合模型指纹与运行时计数器的SlopTracker评分机制，有效缓解了LLM小说中角色名字集中于少数高频名字（如Elara、Elias）的问题。 |
| [^169] | [Geometry of Confabulation and Refusal in Small Language Models](https://arxiv.org/abs/2611.01272) | 该研究提出 ConfabQA 基准，通过线性探针、稀疏自编码器和干预方向优化揭示了小型语言模型在不可回答问题上“自信虚构”与“拒绝回答”背后的隐藏状态几何特征，并证明可以在生成之前预测甚至将虚构行为转化为拒绝行为。 |
| [^170] | [EZ-Ar-VC: Robust Arabic Voice Anonymization with Generalization across Dialects and Code-Switching](https://arxiv.org/abs/2611.01268) | 该论文提出了基于EZ-VC的阿拉伯语专用语音转换模型EZ-Ar-VC，实现了阿拉伯语语音的匿名化，并在跨方言转换和语码转换场景下展现出优异的鲁棒性与泛化能力。 |
| [^171] | [Does Random Search Discover Task Experts? Revisiting Neural Thickets](https://arxiv.org/abs/2611.01262) | 本文质疑了“预训练模型中存在可通过随机权重扰动发现的任务专家”这一观点，发现所谓的专家提升大部分可由选择噪声的过拟合解释，且在保留问题上优势往往消失或落入模型本身的运行波动范围内。 |
| [^172] | [MedAstra: ASR-less Speech Retrieval with Diagnostic Grounding for Medical QA](https://arxiv.org/abs/2611.01261) | MedAstra提出了一种无需ASR转写的医疗语音问答检索框架，通过轻量级适配器将冻结的WavLM语音表示映射到冻结的E5文本嵌入空间，实现直接从语音波形进行密集检索，纯语音检索器Recall@10达到87.6，混合配置达到90.4，优于基线的81.0。 |
| [^173] | [The Brittle Compass: Navigating LLM Prompt Sensitivity in Slovak Migration Media Discourse](https://arxiv.org/abs/2611.01219) | 本案例研究针对斯洛伐克语这一低资源语言的移民相关NLP任务，发现母语提示词仅带来有限的、依赖任务的收益，大模型对提示词变化更鲁棒，且指令是影响性能的最关键提示词组成部分。 |
| [^174] | [ArSciKids: Arabic Curriculum-Aligned Datasets for Scientific Experiment and Multiple-Choice Question Generation in Primary Education](https://arxiv.org/abs/2611.01191) | 本文推出ArSciKids，这是首个面向小学科学教育的阿拉伯语课程对齐数据集，包含999条结构化科学实验记录和2,167道选择题，填补了阿拉伯语教育内容生成领域公开数据集的空白。 |
| [^175] | [Concept-Guided Prompting Strategies Increase Diversity in LLM-Generated Recipe Populations](https://arxiv.org/abs/2611.01171) | 提出一种概念引导的提示方法，在食谱生成任务中将语义覆盖率从基线的35.1%和13.1%提升至47.0%，显著增强了大语言模型生成结果的多样性。 |
| [^176] | [GSLM2: Multi-Stream Speech Representation Learning and Generative Spoken Language Modeling](https://arxiv.org/abs/2611.01143) | GSLM2提出了多流语音表示学习与多流生成式口语语言建模框架，通过SpidR 2.0编码器在多个抽象层次上诱导高质量离散语音单元，并用共享骨干网络联合建模跨流依赖关系，从而弥合语音模型与文本模型之间的差距。 |
| [^177] | [Rethinking LLM Pruning via Test-Time Scaling](https://arxiv.org/abs/2611.01133) | 本文通过pass@k评估发现剪枝模型的推理能力未必真正下降，其性能损失更多源于正确解采样难度的增加，因此测试时扩展是重新评估和恢复剪枝模型性能的有效视角。 |
| [^178] | [Do Earnings Calls Help Predict Financial Restatements? Evaluating Text Beyond Public Firm Information](https://arxiv.org/abs/2611.01123) | 该研究发现盈利电话会议文本虽能预测财务重述，但在公开公司信息基线之外并无可检测的预测增益，而大型语言模型的高得分源于信息泄露，因此评估文本信号价值必须设置公开信息基线并检验语言模型可能存在的记忆泄露问题。 |
| [^179] | [CompSkill: Compiling Execution Experience for Skill Retrieval](https://arxiv.org/abs/2611.01115) | CompSkill通过将执行经验编译为技能偏移量和紧凑的密钥残差来改进LLM智能体的技能检索，使系统能够根据实际执行结果动态选择最合适的技能文档而不修改文档文本本身。 |
| [^180] | [CogRSI: Evolving and Auditing Cognitive Reasoning Strategies](https://arxiv.org/abs/2611.01113) | 本文提出CogRSI框架，让语言模型通过自我改进循环自动撰写并优化心理理论推理原则，并创新性地将学到的规则翻译成代码在可控测试故事上进行审计，发现多数自我改进得到的规则在其声称覆盖的情境中仍会出错。 |
| [^181] | [Multi-Turn Semantic Caching for Low-Cost LLM Serving](https://arxiv.org/abs/2611.01077) | 该论文提出面向多轮对话的语义缓存方法，通过从评判者反馈中学习“复用风险”来决定缓存哪些对话状态，并给出具有次优性保证和遗憾界的算法（CUCB-SC-MT与CLCB-LS-MT），在保证复用质量的同时降低LLM服务成本。 |
| [^182] | [Label Smoothing in Disguise? Gated Knowledge Distillation for Low-Resource Bangla Generation](https://arxiv.org/abs/2611.01065) | 该研究在低资源孟加拉语文本生成任务上系统评估了两种门控知识蒸馏方法，发现它们均未显著优于普通知识蒸馏或交叉熵微调，KD的主要作用仅是正则化，而无需教师的标签平滑即可获得类似效果。 |
| [^183] | [HALO: Hierarchical Evidence Selection and Reinforcement Learning for Allocating Budgeted Label Optimization from Longitudinal Clinical Records](https://arxiv.org/abs/2611.00986) | HALO框架通过分层证据选择与预算感知的强化学习，在有限标注预算下联合学习从纵向临床记录中筛选关键证据，并智能分配免费、银级、金级三种保真度的标签来源，解决了以往研究将证据视为固定输入的局限。 |
| [^184] | [A Token Is Not a Unit of Difficulty: Segmentation Controls Where Adaptive Computation Goes Inside a Word](https://arxiv.org/abs/2611.00983) | 该论文发现提前退出语言模型中词内计算量的分配由分词方式而非语言本身决定——英语、德语和土耳其语中词的后续片段获得比首个片段更少的计算，而孟加拉语相反，且重新分词或更换分词器即可偏移甚至逆转这一差距。 |
| [^185] | [Self-Refining Prompt Induction for Streaming Text-to-SQL](https://arxiv.org/abs/2611.00960) | 该论文提出自精炼提示归纳（SRPI）离线方法，利用三个LLM评判者的反馈迭代优化提示，将生成失败转化为可复用的流式SQL方言指令，在模型权重和提示均冻结的情况下，使七个生成器的平均得分提升7.1至19.2个百分点。 |
| [^186] | [Reformulation and Equivalence of Bock's Algorithm for Exact Dependency Parsing](https://arxiv.org/abs/2611.00939) | 本文重构了用于精确依存句法分析的矩阵式Bock算法，证明了其与原始Bock算法、修正后的Tarjan算法以及Dozat版CL/E算法的等价性，并将其集成到双仿射依存分析器中，在宾州树库和中文树库上验证了其在解码与结构化训练中的有效性。 |
| [^187] | [Labadain-MedQA: A Tetun Medical QA Dataset for Fine-Tuning and Evaluating Large Language Models](https://arxiv.org/abs/2611.00935) | 该论文构建了首个大规模德顿语医疗问答数据集Labadain-MedQA，包含39,077个问答对，通过将MedAlpaca医学数据集翻译成德顿语并进行人机翻译质量对比评估，为低资源语言的医疗LLM微调与评估提供了宝贵资源。 |
| [^188] | [When 200 OK Is Not an Answer: Silent Fetch Failures in Agentic Web Content Acquisition](https://arxiv.org/abs/2611.00884) | 该研究测量了将HTTP 200响应当作抓取成功标志这一假设的失效频率，发现静态客户端抓取的200状态页面中有20.9%实际不含页面主要内容（如应用外壳、反爬挑战、封锁页面等），这会严重误导网络智能体和引用来源活性检查的结果。 |
| [^189] | [Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand](https://arxiv.org/abs/2611.00883) | 该论文通过理论分析揭示基准测试子集估计误差呈长尾分布，并提出BestRand方法，在保留随机采样低成本高效率的同时显著压缩其高误差尾部，实现了稳定且低成本的基准测试压缩。 |
| [^190] | [What Breaks and What Comes Back: Capability Damage and Recovery in Depth-Pruned Small Language Models](https://arxiv.org/abs/2611.00877) | 该研究系统测量了深度剪枝后小型语言模型六项能力的损伤与恢复，发现函数调用和数学能力在所有模型家族中都会崩溃且崩溃顺序高度一致，而“愈合”微调时选择训练哪些参数与选择训练数据同等重要。 |
| [^191] | [ShiftRAG: Bypassing the Textual Bottleneck via Decoupled Learning and Continuous Soft Tokens](https://arxiv.org/abs/2611.00871) | ShiftRAG提出了一种解耦的RAG框架，在单一LLM骨干上分离判别性检索学习与自回归生成，并将检索对齐转化为连续软令牌注入生成器，从而绕过文本瓶颈、避免目标冲突，同时提升检索与生成性能。 |
| [^192] | [Adaptive-Policy Distillation: Knowledge Injection under Information Asymmetry](https://arxiv.org/abs/2611.00852) | 该论文发现反向 KL 监督的在-policy蒸馏因梯度加权和 rollout 覆盖率问题在知识注入任务上反而失效，并提出自适应策略蒸馏，将监督集中分配到师生分歧最大的困难 token 位置上，从而实现高效的知识注入。 |
| [^193] | [Fewer Tokens, More Answers: Verifiable-Reward Tuning of an Ornith-Style VLM for Multi-View Driving QA](https://arxiv.org/abs/2611.00833) | 该论文用可验证奖励（GRPO）微调Ornith风格视觉语言模型用于多视角驾驶问答，发现在256 token的紧凑生成预算下能显著提升答案抽取率与准确率，且收益主要来自将原本超长、无法抽取的推理输出转化为可用答案。 |
| [^194] | [ODC: Orthogonal Drift Correction for Improved Text-to-Image Semantic Alignment at Inference](https://arxiv.org/abs/2611.00819) | 提出正交漂移校正（ODC），一种无需重新训练模型的推理时引导方法，通过两阶段过程识别并校正语义漂移，直接细化用户提示的文本嵌入，从而提升文生图与文本提示的语义对齐。 |
| [^195] | [Steer, Don't Tell: Breaking Reasoning Plateaus with Reachability-Aligned Self-Distillation](https://arxiv.org/abs/2611.00796) | 提出自引导推理器（SSR），发现自我蒸馏中“教师答案正确”不等于“学生可达到”的问题，并通过学生支持投影将特权教师分布对齐到学生模型可达范围，从而在无需外部专家的情况下突破大推理模型在难题上的推理平台期。 |
| [^196] | [CheXPoT: Post-Hoc Visual Verification and Correction of VLM-Generated Radiology Reports](https://arxiv.org/abs/2611.00785) | CheXPoT提出了一种程序化思维框架，将视觉语言模型生成的放射报告分解为谓词并与视觉证据逐一比对验证，事后删除无法在图像中找到依据的幻觉性发现，从而以远低于训练类方法的计算成本提升报告质量。 |
| [^197] | [ConstraintMatch: Compositional Image-Text Matching through Constrained Regional Evidence Selection](https://arxiv.org/abs/2611.00784) | ConstraintMatch是一种免训练的推理方法，通过比较实体内共享区域与独立区域选择之间的分数差距，将实体级区域一致性转化为显式匹配代价，从而修正图文匹配中属性-实体绑定错误的问题。 |
| [^198] | [DuSE-Bench: Benchmarking Joint Reasoning over Evolving Dialogue and Documents](https://arxiv.org/abs/2611.00755) | 该论文提出了DuSE-Bench基准，包含五个场景下的1,000个任务（其中700个任务的前提同时源于对话与文档），用于系统评估信息助手在演化对话与文档上的联合推理能力。 |
| [^199] | [Constructing Budgeted KGW Evasion through Token-Event Evidence Replacement](https://arxiv.org/abs/2611.00726) | 提出一种预算受限的KGW水印规避方法：仅用独立模型的普通短语改写和精确覆盖搜索，在给定编辑预算内最大化移除水印检测证据，且无需密钥、检测器反馈或查询水印模型，即可显著削弱水印检测。 |
| [^200] | [Learning Timeline-Preserving Speech-to-Speech Translation](https://arxiv.org/abs/2611.00690) | 提出时间线保持型语音到语音翻译新任务及端到端自回归系统 TransDubber，可直接在共享绝对时间轴上保留源语音的结束点和显著内部停顿，无需推理时的时间控制或事后校正。 |
| [^201] | [Precedent Labels and Facts Shift LLM Approval Decisions in Opposite Directions](https://arxiv.org/abs/2611.00672) | 本研究通过分离先例案例的“事实”与“裁决标签”发现，二者会以相反方向影响大语言模型的审批决策——裁决标签使模型输出与先例一致（批准率提升30至55个百分点），而事实内容则将模型决策推向相反方向。 |
| [^202] | [Audit at the Precision You Deploy: Quantization Reshapes a Subliminally Learned Preference](https://arxiv.org/abs/2611.00665) | 4-bit量化会因量化器的不同而显著放大或抑制模型中潜意识学习到的隐藏偏好，因此安全审计必须在部署所用的量化精度下进行。 |
| [^203] | [Heirloom: Carrying a Phone's Personalization Across Base-Model Generations](https://arxiv.org/abs/2611.00650) | Heirloom提出了一种无需保留用户带标签数据的方法，通过将旧个性化模型作为教师在无标签数据上生成预测，在手机上将个性化适配器迁移到升级后的新基模型，且内存中仅需同时保留一个基模型。 |
| [^204] | [ElectroChemLLM: A Multimodal LLM Pipeline for High-Fidelity Electrochemical Informatics](https://arxiv.org/abs/2611.00638) | 提出多模态大语言模型流水线 ElectroChemLLM，通过混合文档解析与视觉 LLM 编排相结合，从科学 PDF 中大规模提取涵盖电解液成分、电极材料、电池构型及物化性质的可验证结构化电化学数据，克服了纯文本架构、上下文碎片化与数值幻觉的局限。 |
| [^205] | [Enhancing Language Model Reasoning with Structured Multi-Level Modeling](https://arxiv.org/abs/2611.00616) | 提出多级推理（MLR）框架，将长思维链生成建模为“高层规划—低层执行”的两级随机过程，通过轻量级LoRA模块实现结构化高层规划，从而解决单一策略模型在长时程推理中的规划漂移问题。 |
| [^206] | [Sparse Autoencoder Attribution Graphs Are Decided at the Margin](https://arxiv.org/abs/2611.00606) | 该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。 |
| [^207] | [Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments](https://arxiv.org/abs/2611.00597) | 本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。 |
| [^208] | [Saidzoda-Tajik-E5-Large: Fine-Tuning a 560M Embedder for Tajik Retrieval](https://arxiv.org/abs/2611.00592) | 提出了首个公开描述的塔吉克语文本检索嵌入模型Saidzoda-Tajik-E5-Large，通过自建语言模型合成23.5万余个问题-段落对微调5.6亿参数的多语言嵌入模型，在法律文本检索的前十命中率从80.2%提升至93.6%，同时保持英语和俄语能力不退化。 |
| [^209] | [TVP: TV is Almost All You Need for Speculative Decoding](https://arxiv.org/abs/2611.00588) | 该论文证明总变差（TV）距离在草稿模型训练中匹敌或超越已有目标，并提出基于期望接受长度的TVP损失函数族，通过幂参数插值解决梯度消失与后位置饥饿问题，从而提升投机解码性能。 |
| [^210] | [Conic Certification of Transformer Attention](https://arxiv.org/abs/2611.00585) | 本文建立了Transformer注意力认证的锥规划理论，证明认证难点在于查询与键的双线性交互（NP难且不可近似），并提出指数锥矩松弛方法，给出误差按扰动半径二次界定且阶数不可改进的认证保证。 |
| [^211] | [From Apparent Search Gains to Selection Failures: Auditing Test-Time Inference in Structured Workflow Planning](https://arxiv.org/abs/2611.00579) | 该论文将测试时推理审计性地分解为“发现—存留—选中”三个可观测阶段，揭示名义树搜索的收益实际来自其扩展提示词而非真正的搜索结构，且失败主要发生在最终选择环节而非结构验证环节。 |
| [^212] | [Distinguishing among reasoning levels of GPT 5.5 Pro via a grammar lens benchmark](https://arxiv.org/abs/2611.00555) | 本文提出一种源自计算理论的形式化语法视角基准，成功区分了GPT-5.5系列模型在不同测试时推理级别下的细粒度能力差异，并发现随着测试时计算努力增加，模型与人类的响应分歧逐渐增大，揭示了能力与对齐之间的潜在冲突。 |
| [^213] | [Skill++: Towards Lifelong Agents via Routing-Aware Skill Refinement](https://arxiv.org/abs/2611.00544) | 该论文提出Skill++，将技能库增长中相似新技能取代旧技能导致的路由干扰问题形式化为终身技能路由（类增量持续学习）问题，并通过路由感知的技能精炼方法，在保持旧技能仍可被选中的同时避免抑制新技能。 |
| [^214] | [Dual-Axis Policy Entailment: A Benchmark for Policy-Change Reasoning](https://arxiv.org/abs/2611.00536) | 该论文提出双轴政策蕴含（DAPE）基准，通过同时判断政策对是否属于同一政策领域以及后继政策是延续、扩大、缩小还是逆转原有政策，为政策变化推理提供了更严谨、更细致的评测标准。 |
| [^215] | [WeSearch: A Unified Model-Centric Retrieval Framework for WeChat Search](https://arxiv.org/abs/2611.00530) | WeSearch提出了一种以模型为中心的统一检索框架，通过先检索查询相关簇、再在查询特定候选空间中对原子文档排序，将候选剪枝与细粒度相关性建模解耦，并已作为微信搜索的主要语义检索路径部署，在大规模语料上取得了持续的A/B测试收益。 |
| [^216] | [Agentic Context Management for Multi-Turn Human-AI Conversations](https://arxiv.org/abs/2611.00518) | 提出无需训练的智能体对话上下文策展方法AC3，由独立分析器仅从用户消息整合任务规范、审查助手先前输出并据此编辑上下文，从而解决多轮对话中的上下文污染问题。 |
| [^217] | [AStockArena: Beyond Returns, Can We Entrust Real Capital to the Decisions of Financial Agents?](https://arxiv.org/abs/2611.00515) | 该论文提出AStockArena——一个针对中国A股市场连续交易的实时决策审计基准，通过涵盖执行可追溯性、证据充分性和金融元认知的八项检查来审计13个模型在3,198个连续决策中的真实行为，超越单纯收益指标以回答金融智能体是否值得托付真实资本的问题。 |
| [^218] | [Robustness Starts with the Objective: Designing and Certifying Robust LLM-Generated Heuristics](https://arxiv.org/abs/2611.00504) | TAILOR框架利用候选方案与现有基线在同一工作负载上成本差异的条件风险价值（配对CVaR）来指导LLM合成启发式算法，负的配对CVaR可以在无需计算未知最优解的情况下，同时认证平均性能与尾部性能的改进。 |
| [^219] | [Swap, Replicate, Read the Mass: A Protocol for Auditing Demographic Bias in System One Decision Models](https://arxiv.org/abs/2611.00496) | 该论文提出了首个针对“系统一”决策模型的人口偏见多模型审计协议，基于三个偏见基准、十二个模型共163.8万次决策，发现这些模型普遍存在刻板印象对齐和顺序依赖性，且其置信度反映的是不稳定性而非偏见本身。 |
| [^220] | [Spotting the Artificial: Dialogic Humor Improvisation in Humans and LLMs](https://arxiv.org/abs/2611.00492) | 该研究提出结合自动化语言度量与系统功能语言学人工标注的方法，比较多智能体环境中大语言模型生成的幽默对话与人类喜剧即兴表演，发现LLM虽能创造连贯且有创意的场景，却无法再现自发口语互动的关键特征，并表现出过度礼貌与正面评价的倾向。 |
| [^221] | [From Small to Large: Assessing Medical Reasoning, Benchmark Contamination, and Model Size in Language Models](https://arxiv.org/abs/2611.00460) | 该研究通过评估来自六家厂商的14个不同规模语言模型在四个医学基准共3,076道题目上的表现，考察医学考试成绩是否真实反映临床推理能力，还是受题目格式、基准污染（记忆效应）和模型规模的影响，并检验适合本地部署的小模型能否达到人类水平。 |
| [^222] | [From Lookups to Lessons: Turning Retrieved Knowledge into Parametric Memory](https://arxiv.org/abs/2611.00450) | 提出ResMem方法，让LLM作为离线教师阅读检索文本生成教师目标，训练参数化记忆学习加性logit残差，从而在不存储海量token表示的前提下将检索知识有效转化为模型参数化记忆。 |
| [^223] | [Quantiles, Not Points: Streaming Quantile Matching for Ordinal-Aware LLM-as-a-Judge Calibration](https://arxiv.org/abs/2611.00447) | 提出流式分位数匹配（SQM）校准方法，通过最优传输将LLM评分分布映射到人类评分分布，尊重评分的序数性并实现了最佳的分布保真度。 |
| [^224] | [Fidelity Collapses with Table Size, Not Turn Count: Delivery Architecture in a Deployed Tool-Using Agent](https://arxiv.org/abs/2611.00442) | 在生产级工具使用智能体中，回答正确性的崩溃取决于交付给模型的表格规模而非对话轮次数：将记录保留在会话状态并通过确定性过滤器与类型化旁路通道交付的 BYPASS 架构（92.3% 正确）显著优于将全部记录注入上下文的 SYNTHESIS 架构（45.7% 正确）。 |
| [^225] | [Do Multimodal Models Decide Consistently on Equivalent Optical Spectra?](https://arxiv.org/abs/2611.00428) | 该研究通过将同一光学光谱以多种等价表征（不同单位的表格与图形）呈现来测试多模态模型的表征不变性与弃权能力，发现仅改变表征方式就可使模型准确率波动高达41个百分点，表明模型对等价光谱的判断并不一致。 |
| [^226] | [Circuit Oracle: Automating Attribution Graph Analysis via Natural-Language Queries](https://arxiv.org/abs/2611.00420) | 本文提出Circuit Oracle，通过多智能体框架和任务技能让现成大语言模型能够自动读取归因图并用自然语言回答关于目标模型的安全相关问题，将原本需要专家数小时人工分析的机制可解释性工作自动化。 |
| [^227] | [Geometric and algebraic inference in humans and neural networks](https://arxiv.org/abs/2611.00387) | 本文将人类推断策略形式化为“几何式推断”（通过逐步塑造表征几何结构使推断代价低廉）和“代数式推断”（灵活推理上下文中的新颖关系但需付出额外计算成本），并以传递性推断为测试平台，刻画了人类和神经网络中两种策略各自出现的条件。 |
| [^228] | [LF-VideoQA: Evaluating Long-Video Understanding with Long-Form Theory-of-Mind Questions](https://arxiv.org/abs/2611.00383) | LF-VideoQA是一个基于心智理论问题的长视频开放式问答数据集，包含94部电影中的6,300多个需要综合跨场景（平均23分钟）证据进行推理的问题，并配有专用评分标准，从而超越简单检索范式、更真实地评估模型的长视频理解能力。 |
| [^229] | [Unlearning by Extrapolating a Shrink-and-Heal Lookahead](https://arxiv.org/abs/2611.00374) | SHIFT方法通过将微调模型向微调前权重收缩并重新学习保留集来估计通往重训模型的方向，再将该前瞻过程的逐token变化外推为蒸馏目标并迭代优化，实现了最先进的机器遗忘效果。 |
| [^230] | [Humanized but Still True? Auditing Facts and Detection in AI-Text Humanizers](https://arxiv.org/abs/2611.00359) | 该论文首次对AI文本“人性化器”的事实保真度进行系统审计，发现全部11款商业工具都会在相当比例的文档中更改或丢失数字、日期、名称等事实细节，揭示了规避AI检测与保持文本真实性之间的矛盾。 |
| [^231] | [Cross-Prompt Reuse of Entity-Replacement Responses](https://arxiv.org/abs/2611.00352) | 该论文发现在固定模型权重的前提下，将源问题引发的隐状态变化注入目标问题的中间层，可以实现实体替换响应在事实性问题之间的跨提示复用，在Qwen、Llama和Gemma等小型语言模型上最多消除约一半的无变化误差。 |
| [^232] | [Queering the Audits: Community-Based Auditing of AI Harms to Queer Communities](https://arxiv.org/abs/2611.00345) | 该研究通过让16名酷儿社区成员在参与式工作坊中运用4C伤害分类法审计四个AI案例，揭示了技术审计无法发现的社区伤害，并将伤害根源追溯到问题定义、数据收集及从弱势群体攫取价值的合同结构。 |
| [^233] | [SPARK: Self-Correcting Embodied Memory via Physical Re-Observation](https://arxiv.org/abs/2611.00344) | 提出了SPARK框架，通过语义批评器标记知识库错误、针对性规划器生成重访计划、选择性合并更新信息，实现具身RAG记忆的自我纠错。 |
| [^234] | [One Waveform, Two Stories: Benchmarking Speech-Acoustic Conflict in Audio-Language Models](https://arxiv.org/abs/2611.00325) | 该论文提出SAC-Bench基准测试，揭示了音频-语言模型在面对同一录音中语音陈述与声学证据相互矛盾时的脆弱性——事件识别准确率下降24.8至46.5个百分点，暴露出模型缺乏处理音频内语音-声学冲突的能力。 |
| [^235] | [GAVAGAI: Referential Alignment for Infant-Scale Vision-Language Models via Balanced Optimal Transport](https://arxiv.org/abs/2611.00319) | 该论文提出GAVAGAI方法，将词语-区域对齐建模为带有NOTHING空列和区域容量约束的熵正则化最优传输分配问题，从而以约八次Sinkhorn迭代的低成本为婴儿尺度的视觉-语言模型提供直接的指称对齐监督。 |
| [^236] | [Decentralized Multi-Agent Coordination and Communication in Shared Codebases](https://arxiv.org/abs/2611.00314) | 该论文提出了 ParallACT 基准，首次系统研究无编排器的多个编程智能体在共享代码库中面对互补与矛盾任务时的协调与通信问题，揭示独立开发实现的合并冲突及其影响因素。 |
| [^237] | [Cross-Lingual Persona Effects in LLM Negotiation](https://arxiv.org/abs/2611.00300) | 该研究提出“跨语言社会策略保真度”这一新概念，通过实验发现LLM谈判中提示词定义的人格行为在不同语言间不一致——韩语谈判达成协议率（77.3%）显著高于英语（52.5%），且该差异仅出现在中立和固执型买家中，表明跨语言部署时社会策略的可靠性存在隐患。 |
| [^238] | [Teach, Abstract, Apply: A Framework for Dialogue-Driven Task Learning with Collaborative Robots](https://arxiv.org/abs/2611.00297) | 本文提出TAA框架，使操作员能够通过自然语言向协作机器人教授新工作流程，将其抽象为可复用技能，并在不同场景中灵活应用。 |
| [^239] | [MEMENTO: Learning Context Management for Long Context Processing Agents](https://arxiv.org/abs/2611.00244) | 提出MEMENTO智能体，通过“事后笔记标注的合成轨迹微调+结合答案正确性、笔记质量与置信度校准的多目标奖励强化学习”两阶段训练，让Qwen3-8B学会有效的上下文压缩记笔记行为，并能泛化到上下文折叠和置信度回溯等测试时策略。 |
| [^240] | [From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings](https://arxiv.org/abs/2611.00172) | 该综述指出跨模态嵌入研究正因对比学习、Transformer架构和粒度转变而走向统一，提出了“覆盖性—可比性—组合性”三层递进能力框架，并倡导以联合嵌入作为实现全模态统一的最终路径。 |
| [^241] | [HistGramBench: Measuring Structural Recovery When Multimodal Models Redraw Historical Mathematical Diagrams as TikZ](https://arxiv.org/abs/2611.00081) | 提出 HistGramBench 基准，通过两个不同AI模型对历史数学图形中的几何事实达成共识来检验模型重绘的 TikZ 程序，发现最佳的平价模型也只能恢复约五分之三的结构事实。 |
| [^242] | [From Zero to OCR: A Data-Centric Framework for Unsupported Low-Resource Scripts](https://arxiv.org/abs/2611.00078) | 本文提出一个以数据为中心的框架，仅需未标注文本语料库和数字字体、无需人工标注任何图像，即可为六种主流OCR引擎均不支持的克什米尔语纳斯塔利格文字等低资源文字构建端到端OCR系统。 |
| [^243] | [When frequency domain classification is superior to spatial](https://arxiv.org/abs/2611.00073) | 研究揭示了频域分类准确率下降的根源在于局部特征的缺失，并据此构建了结合频域与空间域训练的混合模型以获得更高的分类性能。 |
| [^244] | [A Systematic Analysis of Live Intraoperative Surgical Expert Narrations](https://arxiv.org/abs/2611.00028) | 该研究首次系统刻画了脊柱外科医生术中实时解说所包含的信息，并提出一条无需人工标注的自动化语音处理流程，可将术中解说转换为带时间戳的手术流程提示，为手术视频理解提供了一种可扩展的替代标注方案。 |
| [^245] | [Edge-Optimized YOLOv11n Architecture for Offline Recognition of Nigerian Sign Language Alphanumeric Gestures](https://arxiv.org/abs/2611.00025) | 本研究提出面向边缘设备优化的轻量级YOLOv11n架构，并建立了包含11,208张图像、33个静态手势类别的尼日利亚手语基准数据集（按手语者严格划分训练、验证与测试集），实现了在内存仅1至4GB的低资源移动硬件上对尼日利亚手语字母数字手势的实时离线识别。 |
| [^246] | [ARTA: Graph Combinatorial Optimization for Robust Interbeat Interval Estimation from Wearable Electrocardiogram Signals](https://arxiv.org/abs/2611.00011) | 提出ARTA框架，通过将多检测器候选点构建为有向无环图并利用束搜索动态规划进行受约束的时间优化，从而在运动伪影干扰下从可穿戴ECG信号中鲁棒地估计心跳间期。 |
| [^247] | ["I Wrote, I Paused, I Rewrote": A Mixed-Methods Study of Learner Perceptions of Process-Aware LLM Feedback](https://arxiv.org/abs/2611.00009) | 通过对43名本科生英语学习者的混合方法研究，发现学习者对基于写作过程（中间草稿、快照及按键记录）生成的大语言模型反馈持积极评价，认为其有用且与自身修改经历相符。 |
| [^248] | [Cultural Evolution Drives the Emergence of Conventions in Neural Emergent Communication](https://arxiv.org/abs/2611.00006) | 该论文提出了一种以完全分布式方式训练双向神经智能体种群的新模型，证明了文化演化能够驱动语言约定惯例的产生，且更约定化的语言因更易学习而在文化传递中具有直接优势。 |
| [^249] | [Subquadratic Attention with a Hierarchical Tree-Structured Indexer](https://arxiv.org/abs/2610.42112) | 本文提出 Zyphra 分层注意力（ZHA），通过在 KV 缓存上构建递归压缩键摘要的树结构索引器，使包含路由在内的总计算成本随序列长度次线性增长，从而突破了注意力二次方复杂度对大语言模型长上下文能力的根本限制。 |
| [^250] | [Enhancing Knowledge Entity Relation Extraction in Abstracts and Full-text Content of Academic Articles Using Large Language Models](https://arxiv.org/abs/2610.41552) | 本研究系统比较了多种大语言模型策略用于学术论文的知识实体关系抽取，并创新性地提出了SLM增强开源LLM和闭源LLM增强开源LLM两种新组合策略。 |
| [^251] | [PsyInterviewee: Informational Invariance and Relational Adaptivity in Mental Health Interview Simulation](https://arxiv.org/abs/2610.41289) | 该论文提出PsyInterviewee框架，通过分离受保护的真实信息通道与动态交互通道并采用非对称写入权限机制，在心理健康访谈模拟中同时保证临床事实的信息不变性与交互行为的适应性。 |
| [^252] | [Li$^{2}$SAO: Lifting Light Singl-Rollout Asynchronous RL Optimization via Group-Relative Actor-Critic Bootstrapping](https://arxiv.org/abs/2610.40726) | 该论文提出Li²SAO，通过组相对Actor-Critic自举摆脱对大规模预训练价值模型的依赖，在轻量级单rollout异步RL优化中兼顾推理任务与智能体任务的表现。 |
| [^253] | [Multi-View Latent MoE: Letting Every Router Shape Expert Computation](https://arxiv.org/abs/2610.40577) | 提出多视角潜在MoE框架，利用多个独立路由器聚合的信号改进专家选择，并复用路由器偏好来调制专家计算，从而在固定专家预算下提升模型能力。 |
| [^254] | [Leveraging Recourse and Size Bounds for OnlineKnapsack with Removable Items](https://arxiv.org/abs/2610.40290) | 该论文在带可移除物品的在线背包问题中，首次将物品大小上界感知与允许重插被拒物品的重插机制相结合，建立了竞争比的紧密上下界，并显著改进了已有的最优竞争比。 |
| [^255] | [Efficient Agentic Audio-Visual Question Answering through Toolset Pruning and Modality Consolidation](https://arxiv.org/abs/2610.38960) | 本文通过分阶段的工具集剪枝与模态整合实验，将庞大的专用工具智能体压缩为仅含两个全模态工具的紧凑工作流，在保持细粒度音视频问答性能的同时显著降低推理成本与执行开销。 |
| [^256] | [Beyond fabrication: Misattributed numbers in papers written by AI research agents](https://arxiv.org/abs/2610.38691) | 该研究重新分析了AI科研智能体基准测试的判定结果，发现智能体所写论文中的数值错误主要并非凭空捏造，而是将近八成的无支持声明错误地归属了智能体自身文件中来自错误实验、指标或运行的数值，并提出了一种确定性的声明-证据检查器来检测此类问题。 |
| [^257] | [Keepace.ai: A Technical Report on a Large Language Model for Sports and Health](https://arxiv.org/abs/2610.38249) | Keepace是基于Qwen3.5-27B后训练的中文运动健康领域专用大语言模型，在健身计划生成、知识问答和运动数据分析三项任务上超越了基座模型及更大的通用模型，尤其在运动安全性、边界合规性和科学准确性方面提升显著。 |
| [^258] | [Rank-Aware Differentially Private Release of Listwise Preferences for LLM Alignment](https://arxiv.org/abs/2610.37682) | 本文提出一种排序感知的指数机制，将列表式偏好数据私有化为低敏感性的固定大小二分划分粒度，解决了完整排序在本地差分隐私下噪声过大的难题，实现了隐私保护的大语言模型列表式偏好对齐。 |
| [^259] | [Style over Substance? Linguistic Bias in LLM-based Peer Review Copy title](https://arxiv.org/abs/2610.35254) | 该论文提出了一个受控实验框架，通过基于语言学文献推导的细粒度改写指令，系统性地改变论文的语言风格（母语/非母语特征与高/低置信度特征），以研究基于大语言模型的同行评审是否会受到与科学内容无关的语言偏见的影响。 |
| [^260] | [T2D: Target-to-Draft Model Pruning & Adaptation for Efficient Speculative Decoding](https://arxiv.org/abs/2610.34103) | 提出T2D框架，将草稿模型设计表述为优化问题，通过延迟感知的结构化剪枝和知识蒸馏直接从目标模型构建草稿模型，实现推理延迟与分布对齐的联合优化，从而提升投机解码效率。 |
| [^261] | [HerHealthGPT: An Evidence-Grounded Conversational AI System for Women's Reproductive Health Copy title](https://arxiv.org/abs/2610.33931) | HerHealthGPT是一个面向女性生殖健康、以证据为依据的对话式AI系统，它融合检索增强生成、情境感知的澄清提问与来源验证，通过主动追问补全不完整的健康咨询，并基于可验证的证据生成回答。 |
| [^262] | [Entity-Aware Evaluation of OCR Systems for Historical Document Transcription](https://arxiv.org/abs/2610.33004) | 该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。 |
| [^263] | [Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox](https://arxiv.org/abs/2610.32955) | 该论文提出自托管个人邮件代理Foyer及其测量协议，在真实多语言邮件数据上于训练前建立人工真实标注基准，并系统比较了类型化决策模型、经典机器学习基线与生成式大语言模型等九个系统的表现。 |
| [^264] | [Parameter-Efficient Adaptation of Large Language Models to a Low-Resource Agglutinative Language: A Corpus, Datasets, and a Reproducible Benchmark for the Bashkir Language](https://arxiv.org/abs/2610.31397) | 该论文发布了巴什基尔语的开放语料库、标注数据集及系统性适配基准，证明基于QLoRA的参数高效适配结合持续预训练，可使3.8-8B参数的大语言模型在该低资源黏着语的文本分类任务上显著超越传统编码器模型微调。 |
| [^265] | [What Pairwise Evaluation Misses: Reward Scores across Prompts](https://arxiv.org/abs/2610.31090) | 论文提出一种以人类评分为参照的审计方法，揭示奖励模型分数存在按提示聚集的跨提示不可比成分——这是成对评估无法检测到的，六个奖励模型的提示间方差份额高达0.34-0.64，而人类评分接近于零。 |
| [^266] | [Finitely Verifiable Segmentation: Decidability for Every Tokenizer with a Finite-State Verifier](https://arxiv.org/abs/2610.30168) | 本文提出“有限可验证分段”这一概念——即分段策略的结果恰好可被一个有限自动机验证——并证明在此假设下四个实际分词问题（边界窗口证明、边界间隙有界性、掩码生成可补全性以及策略等价性）均变为可判定问题。 |
| [^267] | [norregel: a preregistered benchmark of dated numeric values in Norwegian law](https://arxiv.org/abs/2610.29207) | 该论文提出了norregel——一个预注册的基准测试，用于评估语言模型对挪威法律中随时间变化的数值知识的掌握，结果显示即使是表现最好的模型在主要条目上的正确率也不到一半，揭示了语言模型在法律数值知识方面的重大缺陷。 |
| [^268] | [MAS-Guard: Unified Guardrails from Individual Agents to Multi-Agent LLM Systems](https://arxiv.org/abs/2610.28397) | 该论文提出统一安全防护框架MAS-Guard，通过本地智能体防护层（A-Guard）等机制，为包含可插桩与不透明智能体的单智能体和多智能体LLM系统，提供覆盖本地执行与系统级交互的一体化安全保障。 |
| [^269] | [Language Models Learn to Improve Language Models](https://arxiv.org/abs/2610.23344) | 该论文提出LM2LM方法，通过强化学习训练大语言模型生成有效的语言模型改进提案，并利用廉价的代理反馈解决真实训练反馈成本高昂以及分布偏移下反馈不可靠这两大难题，为递归自我改进铺路。 |
| [^270] | [Auditing Escalation Reliability in Healthcare LLM Agents: A Perturbation-Controlled Benchmark of Workflow-State Failures](https://arxiv.org/abs/2610.21270) | 该论文构建了一个包含400个合成医疗工作流片段的扰动控制基准，用于审计医疗LLM智能体在证据缺失、过时或冲突时的“升级转诊”可靠性，发现尽管干净决策准确率高达0.97，模型仍会错过32%--97%的必要升级，且更大的模型只能缩小而非消除该差距。 |
| [^271] | [SwarmRSI: Recursive Self-Improvement through Evolving Scientific Agent Swarm](https://arxiv.org/abs/2610.20431) | 提出了 SwarmRSI 框架，通过反馈引导的群体变异和互补群体重组两种机制演化科学智能体群体，解决假设与实现修改不明确以及有用组件被隐藏的问题，实现科学研究中的递归自我改进。 |
| [^272] | [Empathy Is Not the Scapegoat for Declines in Factuality and Reliability](https://arxiv.org/abs/2610.18541) | 本研究证明共情并非模型事实性与可靠性下降的根源——温暖效应因任务和模型而异，通过任务对齐的自适应语气可以在提升共情的同时保持甚至提高针对性可靠性，并提出了经过人工验证的共情评估器。 |
| [^273] | [The Failure Is in the Readout: Fine-Grained Emotion Recognition Benchmarks Measure Elicitation, Not Perception](https://arxiv.org/abs/2610.08162) | 该研究发现，现成的视觉语言模型在细粒度情绪识别上其实并不逊于专门微调的模型，此前基准测得的失败源于生成式的答案引出方式，而非模型的感知能力不足。 |
| [^274] | [OpenJev-RLCD: A Working RLCD Implementation](https://arxiv.org/abs/2609.38850) | 本文实现了面向推理模型的RLCD（校准决策强化学习），用严格正则评分规则对采样推理依据后的答案分布评分，证明RLVR是缺失多样性项的混合目标，并提出“先校准后强化”的两阶段训练方案，使模型既保持推理能力又获得良好校准。 |
| [^275] | [A Proposed Rubric for Evaluating Expressed Clinical Reasoning in Large Language Model Responses](https://arxiv.org/abs/2609.37788) | 该论文提出一个融合医学教育评估框架、临床大语言模型基准和通用LLM推理评估研究的多维评分量规，用于对大语言模型针对金标准临床案例的自由文本回答中表达的临床推理进行结构化评估。 |
| [^276] | [AgentHop: A Diagnostic Benchmark for Agentic Multi-Hop Question Answering over Scientific Citation Graphs](https://arxiv.org/abs/2609.34428) | AgentHop是一个诊断性基准，通过在受控七工具沙箱中沿检索、综合、工具调用和资源管理四个维度剖析智能体在科学引文图谱上的多跳问答表现，精准定位大语言模型智能体的失败原因。 |
| [^277] | [Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences](https://arxiv.org/abs/2609.24965) | 该研究将Jev作为科学工作流中的语义决策组件进行评估，发现其语义正确性与其他配置持平且延迟最低，并表明错误的语义选择会改变下游计数但可能不影响最终结论标签。 |
| [^278] | [Total Cost of Agency: Exact Attribution of Memory Injection Cost in Multi-Agent LLM Workflows](https://arxiv.org/abs/2609.23790) | 提出代理总成本（TCA）框架，将多智能体LLM工作流成本分解为基础提示词、推理、记忆注入等组件，并通过两遍式不计费令牌计数方法实现记忆注入成本的精确归因，发现记忆注入在企业基准测试中占可变成本的13.6%。 |
| [^279] | [Bridging Static and Agentic RAG for Taiwanese Historical Question Answering](https://arxiv.org/abs/2609.23056) | 该论文针对台湾历史问答发现智能体式与静态RAG总体性能相近但在70.83%的问题上表现各异，并提出一种比较两个回答及其引用证据的事后选择器，显著优于任一单一流水线并恢复了oracle提升空间的60.34%。 |
| [^280] | [EvalMem: An Operation-Level Diagnostic Framework for Long-Term Memory Systems](https://arxiv.org/abs/2609.22231) | EvalMem通过编码、检索、生成三个并行检查器将长期记忆系统的错误精确定位到具体操作环节，并借助召回优先的智能体RAG策略将证据召回率从70.2%提升至95.6%。 |
| [^281] | [Using OCR Heads to Verbalize Image Semantics](https://arxiv.org/abs/2609.18823) | 研究发现视觉语言模型中负责OCR的注意力头实际上是通用语义特征头，将其注意力权重压缩为语言化透镜变换后，可从模型所有层（甚至第0层）的隐藏状态中解读出可解释的图像语义标签，证明图像表示在早期层就与语言对齐。 |
| [^282] | [EvoOntology: A Self-Evolving Ontology Layer for Data Agents](https://arxiv.org/abs/2609.15779) | 本文提出EvoOntology，一种面向数据智能体的自进化本体层，通过将本体封装为包含模式层、内容层和工具层的MCP服务器，并借助构建器智能体实现自主构建与自进化循环，有效弥合了智能体与异构数据之间的鸿沟。 |
| [^283] | [RAP: Research Attention Prediction Reveals Target-Conditioned Evidence Acquisition Biases](https://arxiv.org/abs/2609.10092) | 提出了RAP滚动基准来评估LLM智能体预测研究关注度变化的能力，发现其表现不如简单的EWMA精确计数基线，并揭示了状态前推优于直接预测、以及面向预测的策略倾向于检索较旧证据这两大瓶颈。 |
| [^284] | [trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories](https://arxiv.org/abs/2609.00038) | 仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。 |
| [^285] | [Item-Mean Surrogates: Why Richer Persona Data Fail to Improve LLMs as Human Surrogates](https://arxiv.org/abs/2608.29455) | LLM作为人类代理的良好表现主要源于其对项目均值的预测，一旦去除项目均值，其仅能解释3.05%的个体特异性变异，而更丰富的人格数据、模型变体和微调均无法弥补这一差距。 |
| [^286] | [Sledgehammer or Scalpel? A Fine-grained Adaptive Framework for Implicit Hate Speech](https://arxiv.org/abs/2608.27462) | 提出了细粒度自适应框架FAID，将隐性仇恨言论划分为浅层、针对性和上下文依赖三类，并针对不同类别采用不同复杂度的检测策略，在提升检测精度的同时降低不必要的计算开销。 |
| [^287] | [Unmatched Does Not Mean False: Incomplete Reference Sets Can Reverse Calibration Rankings in Open-Ended Theory-of-Mind Tracking](https://arxiv.org/abs/2608.25654) | 本文揭示不完整的参考集在开放式心智理论追踪中会颠倒校准排名，导致模型选择出现系统性偏差。 |
| [^288] | [Aligned Alone, Misaligned Together: Forecasting Adversarial Capture in LLM Agent Populations](https://arxiv.org/abs/2608.22444) | 该论文发现单个LLM智能体的校准行为无法预测其群体行为，但通过良性操作数据可校准响应函数，从而提前预测对抗性俘获。 |
| [^289] | [One Success Isn't Reliability: Thinkingbox, a Sandbox and Benchmark for Agents in Stateful Business Workflows](https://arxiv.org/abs/2608.19741) | 本文提出了Thinkingbox，一个支持有状态业务流程智能体评估的沙盒和基准，强调通过多轮交互、策略遵循和持久状态转换来确保可靠性，而非仅关注单次成功。 |
| [^290] | [Gated Against One Model, Open to the Next: Option-Only Solvability in Legal Multiple-Choice Benchmarks](https://arxiv.org/abs/2608.15428) | 本文发现法律多选题基准存在严重的数据泄漏，模型在无问题情况下仅凭选项即可高概率答对，且泄漏对未参与筛选的模型同样有效，挑战了基准的有效性。 |
| [^291] | [Wrong but Useful: Trajectory Value Beyond Answer Correctness in Multi-Agent Messages](https://arxiv.org/abs/2608.14375) | 本文提出了一种受控测量方法，证明在多智能体推理中，错误答案消息也可能包含对后续推理有用的轨迹价值，挑战了仅基于答案正确性过滤消息的传统假设。 |
| [^292] | [Consilience for Verifier-Free Test-Time Scaling](https://arxiv.org/abs/2608.09898) | 本文揭示了基于置信度的无验证器测试时扩展方法在复杂任务上会灾难性失效的关键局限，并提出利用“融通”思想来改进复杂任务下的无验证器测试时扩展。 |
| [^293] | [A Study of ASR Adaptation and Representation Dimensionality Reduction in Persian Speech Emotion Recognition Using Whisper](https://arxiv.org/abs/2608.05165) | 该论文提出利用PCA对Whisper编码器提取的嵌入进行降维，结合注意力池化与轻量级预测头，并探索波斯语ASR任务微调的作用，从而在低资源条件下实现高效的波斯语语音情感识别。 |
| [^294] | [PredActBench: Benchmarking Tool-Augmented Dialogue under Controlled Tool Noise](https://arxiv.org/abs/2608.02372) | 该论文提出PredActBench基准，首次系统评估了LLM对话智能体在不完美预测工具（含噪声）下的表现，并将信任校准指标扩展到多轮对话，发现最先进的模型很少向用户显现预测工具的不确定性。 |
| [^295] | [BiCAA: Bidirectional Credit Assignment for Search-Augmented Agent](https://arxiv.org/abs/2608.01321) | 提出BiCAA方法，通过前向可解性增益（FSG）和后续推理增益（SRG）进行双向信用分配，为搜索增强智能体的每个中间搜索步骤提供细粒度的过程奖励，兼顾证据获取进展与避免冗余检索。 |
| [^296] | [What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs](https://arxiv.org/abs/2608.00013) | 我们提出了首个跨家族的多模态缩放规律，通过文本能力得分直接预测VLM性能，并在150多个VLM上验证了其有效性。 |
| [^297] | [LLMs struggle to simulate human belief updates in controlled environments](https://arxiv.org/abs/2607.28347) | 该研究将六个大语言模型与391名真实参与者的信念更新数据进行一对一对比，发现尽管LLM在给定真实初始立场时无系统性方向偏差，但所有模型均无法模拟初始立场、无法产生忠实的信念更新，且存在三种系统性偏差，表明LLM作为社会科学研究中人类被试替代品的可靠性存疑。 |
| [^298] | [Reasoning Consensus: Structural Ensembling of LLM Reasoning via Weighted DAG Aggregation](https://arxiv.org/abs/2607.27783) | 提出推理共识（RC）方法，通过加权有向无环图聚合将多个大语言模型的思维链进行结构化集成，在准确率与多数投票和辩论相当的同时，还能透明地展示每个推理步骤获得的支持度。 |
| [^299] | [AgentSnare: Learning to Delay, Divert, and Defuse Autonomous Penetration Agents](https://arxiv.org/abs/2607.26998) | AgentSnare提出了一种轨迹自适应的欺骗防御系统，通过基于渗透代理交互历史动态构建诱饵工件，持续将自主渗透代理从真实目标引开，克服了传统静态蜜罐痕迹易被识别绕过的缺陷。 |
| [^300] | [MediaWiki Code2Code Search: Neural Retrieval for the Semantic Discovery of Open-Source Software Entities](https://arxiv.org/abs/2607.26766) | 本文提出了MediaWiki Code2Code Search神经检索系统，通过FAISS IVF-PQ索引实现对2500多个代码库中129万个代码实体的语义化代码到代码搜索，在满足6 GiB内存限制的同时实现了96.6%的索引空间压缩和1.85秒的中位查询延迟。 |
| [^301] | [Instruction-Tuned Models Locally Produce More Syntactic Overlap -- But Not More Conditional Reuse -- Than Humans Do](https://arxiv.org/abs/2607.26015) | 该研究对16个开源Llama和Gemma模型在人类对话相邻轮次间的CFG规则复用进行测量，发现指令微调模型产生的局部句法重叠高于人类，但这只是与上下文相关的效应，其条件性句法复用并不比人类更多。 |
| [^302] | [ClinFusion: A Vision-Centric Multimodal LLM System for Holistic Medical Understanding](https://arxiv.org/abs/2607.24743) | ClinFusion提出了一种融合2D与3D医学图像理解的组合式级联视觉编码器架构，并配套基于视觉的评估框架（MedIF-Bench与感兴趣区域评估方法），实现了以视觉为中心的整体医学理解。 |
| [^303] | [Memory Efficient Audio Synthesis with Decoupled Temporal Depth Diffusion Transformers](https://arxiv.org/abs/2607.23811) | 本文提出了一种内存高效的音频合成架构，通过解耦时间与深度处理、复用单一深度解码器生成所有RVQ层级，在设备端实时合成高保真语音，显著降低内存和计算需求。 |
| [^304] | [Routing Subspaces: Auditing Evaluation-to-Deployment Mismatch in Fine-Tuned Language Models](https://arxiv.org/abs/2607.20436) | 该论文提出一种审计方法，通过在中间层定位并干预编码“评估-部署”行为差异的内部子空间，显著缩小了微调语言模型在评估式提示与普通使用提示之间的行为差距。 |
| [^305] | [Selection Shapes the Boundary: A Replication of Monotonicity and Label Agreement in Unselected NLI Populations](https://arxiv.org/abs/2607.19231) | 本研究在未筛选的SNLI/MultiNLI人群中预注册复制了先前基于筛选数据的发现，结果与原预测相反——含非向上单调性算子的假设标签一致性略高而非略低，表明ChaosNLI按分歧程度筛选条目的机制可能塑造了先前观察到的单调性与标签一致性之间的边界效应。 |
| [^306] | [Query Language Sensitivity in Automated Sentiment Scores of LLM Responses](https://arxiv.org/abs/2607.13304) | 该研究发现大语言模型回复的自动情感评分对查询语言高度敏感——查询语言可解释约26%的拟合方差，远高于品牌因素的影响，且剔除英语后该效应大幅减弱，表明跨语言使用自动化情感评分时需谨慎解读。 |
| [^307] | [MafiaScope: Inspecting Multi-Agent Reasoning in Social Deduction Games](https://arxiv.org/abs/2607.10645) | MafiaScope将社交推理游戏“狼人杀”转变为测量LLM多智能体社会推理的开放测试平台，通过非侵入式的私下结构化提问来揭示智能体公开行为与私下真实评估之间的差异，并提供交互式可视化器分析角色评估轨迹。 |
| [^308] | [fog: Generating Expressive Motion through Layered Function Composition of Code for Heider-Simmel Animation](https://arxiv.org/abs/2607.07952) | 提出fog分层函数组合框架，通过生成动词、副词和情感对应的代码函数来捕捉运动的轨迹、方式和感觉，帮助用户在海德-西美尔风格动画中表达运动与情感。 |
| [^309] | [CoPiT: Cognitive Pivot Translation for Digraphic Low-Resource Mongolian in the Traditional Script](https://arxiv.org/abs/2607.05849) | 该论文提出CoPiT，一种利用蒙古语内部资源不平衡的枢轴翻译流水线，先解决传统蒙文的拼写歧义，再经由资源较丰富的西里尔蒙古文进行翻译，从而在多个骨干模型和目标语言上显著提升翻译质量（BLEU大幅提升）。 |
| [^310] | [Why Struggle with Continuous Latents? Interpretable Discrete Latent Reasoning via Rendered Compression](https://arxiv.org/abs/2606.29712) | 该论文提出首个离散潜变量推理方法（DLR），通过将文本思维链渲染为图像并基于聚类微调构建离散潜变量词表，把连续潜变量状态转换为显式离散标记，从而实现更稳定、可解释且高效的潜空间推理。 |
| [^311] | [SpeechEQ: Benchmarking Emotional Intelligence Quotient in Socially Aware Voice Conversational Models](https://arxiv.org/abs/2606.25990) | 提出了SpeechEQ评估框架，基于EQ-i 2.0理论构建了包含2,265段对话、覆盖15个情商子量表的数据集，并设计了口语情商（SEQ）评分，用于系统评估语音语言模型在多轮语音对话中的社交情感推理能力。 |
| [^312] | [SWE-Opt: A Benchmark for Evaluating LLMs on Performance-Oriented Repository-Level Optimization](https://arxiv.org/abs/2606.25530) | SWE-Opt 提出了一个基于开源项目专家优化案例构建的仓库级基准，通过参数化测试在感知噪声的条件下多维度评估大语言模型的软件性能优化能力，弥补了现有基准忽略时间-内存权衡与执行环境变异性的不足。 |
| [^313] | [Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making](https://arxiv.org/abs/2606.25421) | 该论文提出智能体自主世界建模（AAWM），让智能体根据自身行动前的决策需求主动检索并综合环境转移证据来构建训练监督，从而学习与决策相关的环境动态，而非单纯预测下一观测。 |
| [^314] | [Beyond Self-Reflection: An Execute-Distill-Verify Framework for Agentic Experience Learning](https://arxiv.org/abs/2606.24428) | 提出EDV框架，通过异构智能体独立执行任务、第三方提炼器提取候选经验、执行者一致验证的方式构建离线经验，有效避免了大语言模型智能体在自我反思中将执行错误固化为指导的“自我确认陷阱”问题。 |
| [^315] | [AutoSpecNER: A Fine-Grained Named Entity Recognition Dataset for Vehicle Specification Extraction](https://arxiv.org/abs/2606.24387) | 该论文推出了专家标注的车辆广告细粒度命名实体识别数据集AutoSpecNER（含659条广告、15类实体、超万条标注），并通过基准测试表明微调的DeBERTa以90%的微F1分数显著优于规则方法和大语言模型。 |
| [^316] | [Tmax: A simple recipe for terminal agents](https://arxiv.org/abs/2606.23321) | TMax 提出一个简单的开源强化学习配方，通过新颖的数据生成分类法和仅基于结果的 RL 训练，使仅 9B 参数的小型开源模型在 Terminal-Bench 2.0 上取得 33% 的成绩，超越了此前规模大得多的模型。 |
| [^317] | [AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts](https://arxiv.org/abs/2606.19847) | 提出AtomMem长期记忆系统，通过事实执行器从长篇交互中提取高价值原子事实，并将其组织成分层事件结构和时间档案，为LLM智能体实现高价值密度存储与稳定的记忆演化。 |
| [^318] | [Dual Dimensionality for Local and Global Attention](https://arxiv.org/abs/2606.18587) | 提出距离自适应表示（DAR），在局部上下文窗口内保留全维度KV表示，而对窗口外的远距离token仅使用1/2或1/4维度的表示，以反映局部与远距离token重要性不对称的特性。 |
| [^319] | [Detecting Historical Turning Points in Italian Media: A Complex Systems Approach to a Diachronic News Corpus](https://arxiv.org/abs/2606.14348) | 该研究构建了意大利《共和报》1985-2000年间约60万篇文章的历时性新闻语料库，结合自然语言处理与复杂系统、统计物理学方法，在词汇和语义层面定量检测媒体话语中的历史转折点。 |
| [^320] | [External Experience Serving in Production LLM Systems: A Deployment-Oriented Study of Quality-Cost Trade-offs](https://arxiv.org/abs/2606.11806) | 该研究将生产级LLM系统中外部经验的服务方式建模为面向部署的质量-成本权衡问题，并发现当经验变得案例相关时，基于检索的选择性注入相比全局注入能提供更优的质量-成本运行点。 |
| [^321] | [Agent Skill Evaluation and Evolution: Frameworks and Benchmarks](https://arxiv.org/abs/2606.11435) | 本综述系统梳理了智能体技能的评估与演化方法，提出四种技能演化范式（执行反馈、轨迹蒸馏、压缩和强化学习），并分析了六类以技能为中心的基准测试以推动技能研究发展。 |
| [^322] | [UXBench: Benchmarking User Experience in AI Assistants](https://arxiv.org/abs/2606.09570) | 提出了首个基于真实用户反馈信号、以用户为中心的体验评估基准 UXBench，通过三个相互关联的任务评估 AI 助手的偏好对齐与对话生成能力。 |
| [^323] | [When No Answer Is Correct: Diagnosing Absent Answer Detection for MLLMs in Video Understanding](https://arxiv.org/abs/2606.08239) | 本研究首次系统诊断了多模态大语言模型在视频理解中的“缺席答案”检测能力，发现当正确答案不在候选集时，模型倾向于选择貌似合理的干扰项而非识别出无有效选项，且该缺陷在时间推理任务和更密集的帧采样下更为严重。 |
| [^324] | [Paediatric-HGNN: A Hybrid Heterogeneous Graph Neural Network for Detecting Disfluency in Children's Speech via Multiscale Acoustic Fusion](https://arxiv.org/abs/2606.08210) | 该论文提出Paediatric-HGNN框架，利用异构图神经网络建模词节点与声学帧节点之间的层次化交互并结合多尺度声学融合，有效区分儿童病理性口吃与典型发育性不流畅，达到82.4%的加权准确率，为早期临床干预提供更稳健可解释的工具。 |
| [^325] | [TRACER: Token ReAssignment for Concept ERasure in Generative Recommendation](https://arxiv.org/abs/2606.07688) | 提出TRACER端到端概念遗忘框架，通过将概念相关物品重新分配到替代标记而非直接抑制共享SID，解决了生成式推荐中概念擦除与推荐效用保持之间的冲突。 |
| [^326] | [Humans' ALMANAC: A Human Collaboration Dataset of Action-Level Mental Model Annotations for Agent Collaboration](https://arxiv.org/abs/2606.06388) | 该论文提出了ALMANAC数据集，基于经典的Map Task双人对话任务构建，提供人类协作过程中动作级的心智模型标注，弥补了当前缺乏可用于引导智能体获得协作能力的真实人类协作数据的空白。 |
| [^327] | [When Evidence is Sparse: Weakly Supervised Early Failure Alerting in Dialogs and LLM-Agent Trajectories](https://arxiv.org/abs/2606.05414) | 该论文提出一种仅用轨迹级结果标签训练、基于注意力多示例学习（MIL）的轻量级外部监测器，能够在失败指示证据稀疏且出现较晚的情况下，对对话和LLM智能体轨迹实现早期失败预警。 |
| [^328] | [Activation-Based Active Learning for In-Context Learning: Challenges and Insights](https://arxiv.org/abs/2606.05134) | 本研究探索了基于MLP激活的深度主动学习方法来优化LLM上下文示例选择，但得出负面结论：激活信号（无论从大规模激活还是前四阶矩角度观察）与示例质量和任务性能的相关性很弱，Spearman相关系数最高仅0.33。 |
| [^329] | [A Systematic Analysis of Linguistic Features in AI-Generated English Text Detection Across Domains and Models](https://arxiv.org/abs/2606.04177) | 该研究通过覆盖284个可解释语言特征、30个大语言模型和10个文本领域的大规模实证分析，发现仅基于语言特征即可可靠检测AI生成文本，且词汇丰富度是唯一在跨模型、跨领域场景下保持稳健的判别信号。 |
| [^330] | [Do Neural Retrievers Prefer Certain Documents? Evidence of Learned Relevance Priors](https://arxiv.org/abs/2606.02814) | 研究发现有监督神经检索器在训练过程中会隐式学习到与查询无关的文档级相关性先验，这种先验导致先验值较低的文档即使真正相关也系统性地更难被检索到。 |
| [^331] | [Selective Linkage Precision for Released Medical Vision-Language Models](https://arxiv.org/abs/2606.02276) | 本文提出“选择性链接精确度”指标，揭示攻击者利用元数据属性过滤结合公开医学视觉-语言模型，能够以极高精确度（最高67.68%，加入采集年份后达99.05%）将去标识化胸部X光片与受限报告重新链接，证明传统全池召回率指标严重低估了医学图像发布的隐私泄露风险。 |
| [^332] | [Rethinking Stepwise Model Routing: A Cost-Efficient Table Reasoning Perspective](https://arxiv.org/abs/2605.29319) | 本文发现表格推理步骤中表格标记与文本标记具有不同的不确定性分布，并提出表格感知的逐步路由框架EcoTab，通过分别建模两类标记的不确定性来实现成本高效的表格推理。 |
| [^333] | [Rubric-Guided Process Reward for Stepwise Model Routing](https://arxiv.org/abs/2605.29310) | 提出RoRo框架，用评分准则引导的过程奖励替代仅反映最终答案正确性的结果奖励，从而更好地评估逐步模型路由中的中间决策并提升性能与泛化能力。 |
| [^334] | [A Multi-Instance LLM Framework for Rating the Quality of Surgical Feedback](https://arxiv.org/abs/2605.25440) | 该论文提出了一种基于大语言模型的两阶段多实例提示框架，通过对候选质量标准进行重复独立采样与整合并注入外科领域知识，自动发现可解释的手术反馈质量评估标准，解决了手术培训中口头反馈质量难以评估的问题。 |
| [^335] | [Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework](https://arxiv.org/abs/2605.22620) | 该论文提出了一种无崩溃的多奖励RLIF无监督训练框架，将基于聚类投票的答案级奖励与基于词元自确信度的补全级奖励相结合，并通过GDPO归一化和KL-Cov正则化有效避免奖励破解与熵坍缩，从而提升大语言模型的推理能力。 |
| [^336] | [Monitoring the Internal Monologue: Probe Trajectories Reveal Reasoning Dynamics](https://arxiv.org/abs/2605.18549) | 提出通过在推理过程每个 token 上评估探针来构建“探针轨迹”，并提取波动性、趋势等信号处理特征刻画其时间动态，从而在静态探针表现最薄弱的场景下平均提升 21.7 个 AUROC 点，显著增强对大型推理模型未来行为的预测与安全监控能力。 |
| [^337] | [Mobius Injection: From One-Shot Prompt Injection to Sustained Resource Exhaustion in AI Agent Infrastructure](https://arxiv.org/abs/2605.11442) | 本文提出“莫比乌斯注入”攻击，通过“语义闭环”机制将单次提示注入放大为智能体组件的持续递归执行，把LLM智能体武器化为僵尸节点，从而发动新型“基于且面向智能体的DDoS（AbO-DDoS）”攻击，实现对AI智能体基础设施的持续资源耗尽。 |
| [^338] | [MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image](https://arxiv.org/abs/2605.10616) | 提出了MulTaBench——一个包含40个数据集（图像-表格与文本-表格任务各半）的多模态表格学习基准，专注于模态提供互补预测信号的任务，并证明了对嵌入进行任务特定微调优于使用冻结预训练嵌入。 |
| [^339] | [Pacing Branch Parallelism in LLM Serving](https://arxiv.org/abs/2605.06914) | 提出了 PACE——一个逐步接纳控制器，通过预测“分支外部性”是否处于批次当前的松弛预算内来动态调控 LLM 服务中的分支并行宽度，从而在吞吐量与共批处理请求性能之间取得平衡。 |
| [^340] | [Text Corpora as Concept Fields: Black-Box Hallucination and Novelty Measurement](https://arxiv.org/abs/2605.05103) | 本文提出了一种基于语料库概念场的黑盒方法，通过句子嵌入空间中的局部漂移场来度量幻觉和新颖性，无需模型内部信息且可归因于语料库。 |
| [^341] | [Foundation Models to Unlock Real-World Evidence from Nationwide Medical Claims](https://arxiv.org/abs/2605.02740) | 本研究基于 MarketScan 数据库中超过 2 亿参保人的 438 亿条理赔记录构建了生成式基础模型 ReClaim，通过预训练联合学习诊断、手术、药物和医疗支出，在预测 1,000 多种疾病时显著优于疾病特异性模型和 Transformer 基线，并能成功泛化到电子健康记录数据集。 |
| [^342] | [AI-Enabled Question-Answering for Chartered Accountancy: A Retrieval-Augmented Reasoning Framework](https://arxiv.org/abs/2605.00257) | 提出了参数高效的检索增强推理框架CA-ThinkFlow，通过结合4比特量化的DeepSeek-R1推理模型与版面感知的文档提取系统，在资源受限条件下提升了AI在印度特许会计师等复杂专业领域问答任务中的可靠性。 |
| [^343] | [Contextual Linear Activation Steering of Language Models](https://arxiv.org/abs/2604.24693) | CLAS将激活导向分解为固定的导向方向和依赖上下文的导向系数，使导向强度能随输入上下文自适应变化，并实现了单次生成中针对不同部分引导多种行为的选择性导向，在多项基准测试中持续优于固定强度的传统线性激活导向方法。 |
| [^344] | [Uncertainty Quantification for LLM Function-Calling](https://arxiv.org/abs/2604.22985) | 本文首次对大语言模型函数调用场景下的不确定性量化方法进行了评估，发现在自然语言问答任务中表现优异的语义熵等基于熵的方法在函数调用场景中并不适用。 |
| [^345] | [ReflectMT: Internalizing Reflection for Efficient and High-Quality Machine Translation](https://arxiv.org/abs/2604.19144) | ReflectMT提出一种两阶段反思内化框架，通过强化学习将“翻译-反思-精炼”能力内化到模型中，采用“后置思考”范式在WMT24上以86.16的GRF分数超越DeepSeek-R1，同时大幅降低推理成本和延迟。 |
| [^346] | [Transition-Matrix Regularization for Next Dialogue Act Prediction in Counselling Conversations](https://arxiv.org/abs/2604.18539) | 本文提出一种基于KL散度的转移矩阵正则化方法，将语料库中的对话流统计先验融入下一对话行为预测，在心理咨询对话任务上使宏观F1相对提升9%至42%，且该改进可跨语言和跨领域迁移。 |
| [^347] | [Defragmenting Language Models: An Interpretability-based Approach for Vocabulary Expansion](https://arxiv.org/abs/2604.16656) | 该论文针对LLM在非拉丁文字语言中token过度碎片化的问题，提出了基于可解释性的词表扩展方法，系统研究了“应添加哪些词条”和“如何初始化对应嵌入”这两个核心决策。 |
| [^348] | [IndicDBBench - Benchmarking Multilingual Text-to-SQL Capabilities in Indian Languages](https://arxiv.org/abs/2604.13686) | 该论文提出了IndicDBBench，一个基于真实政府开放数据、通过三智能体流水线构建的覆盖英语及六种印度语言共15,612个任务、20个数据库的大型多语言Text-to-SQL基准，填补了现有基准在非西方语言语境下的评测空白。 |
| [^349] | [Anthropogenic Regional Adaptation in Multimodal Vision-Language Model](https://arxiv.org/abs/2604.11490) | 本文提出了人类区域适应这一新范式以及简单有效的GG-EZ方法，通过区域数据过滤与模型合并，在保持视觉语言模型全局泛化能力的同时，优化其对特定区域（如东南亚）的适应性。 |
| [^350] | [Automated grounding verification of AI statements about chest radiographs](https://arxiv.org/abs/2604.09537) | 本文提出BRACE自动化框架，通过同时篡改报告和引用证据来构建含矛盾示例的训练数据，使接地验证器能识别出以往最先进测试无法发现的、与报告相矛盾的AI陈述。 |
| [^351] | [SYN-DIGITS: A Synthetic Control Framework for Calibrated Digital Twin Simulation](https://arxiv.org/abs/2604.07513) | 提出SYN-DIGITS框架，一种模型无关的事后校准方法，通过从数字孪生中学习问题间的关系并迁移到人类侧，在无目标问题人类回答数据的冷启动场景下有效校准大语言模型角色模拟的偏差。 |
| [^352] | [\textit{Versteasch du mi?} Computational and Socio-Linguistic Perspectives on GenAI, LLMs, and Non-Standard Language](https://arxiv.org/abs/2603.28213) | 本文融合批判社会语言学与计算语言学双重视角，批判性审视生成式AI和大语言模型对非标准语言及小语种的不公平对待，揭示其加深数字语言鸿沟并强化语言标准化认识论的问题。 |
| [^353] | [The Presupposition Problem in Representation Genesis](https://arxiv.org/abs/2603.21745) | 本文提出“表征发生预设问题”，指出大语言模型未经历从非表征系统到表征系统的发生转变，而思想语言假说、目的语义学、预测加工、生成主义和发生现象学等心灵哲学主要框架均以表征发生为前提，因而现有理论缺乏解释LLM在缺失发生时认知能力状况的概念资源。 |
| [^354] | [Representational Acquisition Under Bounded Resources](https://arxiv.org/abs/2603.21736) | 本文在常数因子内精确刻画了精确概念学习任务中的存储-读取权衡曲线，证明单次读取的复合查询强制要求答案特定存储单元、单比特顺序无关更新强制要求事实特定存储单元，并给出了最小存储容量下坐标编码的鲁棒性界限以及保存旧内容所需满足的结构性约束。 |
| [^355] | [EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models](https://arxiv.org/abs/2603.18489) | 提出了EntropyCache，一种无需训练的KV缓存方法，利用新解码token分布的最大熵作为恒定开销的信号来决定何时重新计算缓存，其决策开销与上下文长度和模型深度无关，从而高效加速扩散语言模型的推理。 |
| [^356] | [Probing the Limits of the Lie Detector Approach to LLM Deception](https://arxiv.org/abs/2603.10003) | 本文通过构建基于语言学理论的欺骗性非虚假陈述数据集，实验证明大语言模型能够产生并非虚假的欺骗性陈述，而仅判断真假的标准真值探针无法检测此类行为，从而揭示了“测谎仪”式欺骗检测方法的根本局限。 |
| [^357] | [DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval](https://arxiv.org/abs/2603.04743) | 该论文提出DARE框架，通过将数据分布和结构特征融入R函数检索的表示学习，使LLM智能体能够更准确地为数据分析匹配合适的R统计方法，并配套构建了知识库、智能体和基准测试。 |
| [^358] | [MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM](https://arxiv.org/abs/2602.20191) | 提出MoBiQuant框架，通过发现“精度相关的离群值迁移”现象并采用比特混合量化机制，实现可根据token和运行时资源自适应切换位宽的任意精度LLM高效推理。 |
| [^359] | [When Should LLMs Be Less Specific? Selective Abstraction for Reliable Long-Form Text Generation](https://arxiv.org/abs/2602.11908) | 该论文提出选择性抽象框架，让LLM在长文本生成中不再简单弃答，而是通过将不确定的具体内容替换为置信度更高的抽象表述，用具体性换取可靠性，从而在保留有用信息的同时减少事实错误。 |
| [^360] | [Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning](https://arxiv.org/abs/2602.06600) | 该论文揭示并形式化了大型推理模型在思维链开头自发复述问题（“提示回声”EOP）的现象，证明这种以拒绝采样刻画的概率代价实际上起到了锚定注意力、提升似然与下游准确率的作用，并提出了可计算的“回声似然差距”作为预测指标。 |
| [^361] | [Reinforcement World Model Learning for LLM-based Agents](https://arxiv.org/abs/2602.05842) | 该论文提出强化世界模型学习（RWML），一种自监督方法，通过在预训练嵌入空间中对齐模型模拟的下一状态与环境实际观测到的下一状态，为基于大语言模型的智能体学习动作条件化的世界模型，相比词元级预测提供了更稳健的训练信号并避免了模型崩溃。 |
| [^362] | [Mitigating Bias in Automated Essay Scoring for ESL Learners via Contrastive Learning](https://arxiv.org/abs/2601.16724) | 该论文提出一种基于匹配作文对的双向对比学习策略CL-MEP，可将自动作文评分系统对高水平ESL学习者的评分偏见降低39.9%，同时提高整体评分准确率。 |
| [^363] | [Beyond Single-shot Writing: Deep Research Agents are Unreliable at Multi-turn Report Revision](https://arxiv.org/abs/2601.13217) | 该论文提出 Mr Dre 评估套件，首次将多轮报告修订确立为深度研究智能体的新评估维度，并揭示了这些智能体虽能响应大部分用户反馈，但在16-28%的已覆盖内容和引用质量上出现退化的关键局限。 |
| [^364] | [Measuring Affinity between Attention-Head Weight Subspaces via the Projection Kernel](https://arxiv.org/abs/2601.10266) | 本文提出用基于主角度的投影核度量注意力头权重子空间之间的亲和度，该指标比传统方法更清晰地揭示注意力头间的交互关系，并据此发现GPT2-small中的L4H7作为身份头充当信息枢纽。 |
| [^365] | [LLMberjack: Guided Trimming of Debate Trees for Multi-Party Conversation Creation](https://arxiv.org/abs/2601.04135) | LLMberjack是一个结合交互式可视化界面与可选LLM辅助功能的平台，能够将Reddit等平台上树状结构的辩论“线性化”为连贯的多方对话，同时保留参与者身份和话语关系。 |
| [^366] | [Embedding Consistency Regulation for Persistent Semantic Routing in Compact Language Models](https://arxiv.org/abs/2601.00543) | 该论文提出嵌入一致性调节（ECR）方法，通过冻结嵌入模型为输入生成离散前缀，使紧凑语言模型在解码后仍能持续体现预分配的语义路由（如语言、语气、意图），实现99.93%的语言路由准确率，并引入独立于编码器的响应路由可恢复性（RRR）诊断指标。 |
| [^367] | [Disentangling Perception and Reasoning in Multimodal LLMs via Reward Design](https://arxiv.org/abs/2601.00215) | 该研究发现多模态大模型在算法视觉谜题上的性能瓶颈在于感知而非推理，并通过奖励设计在无思维链监督的情况下诱导视觉锚定推理，使Qwen-2.5-VL-7B提升5.56分。 |
| [^368] | [Hierarchy-Aware Multimodal Unlearning for Medical AI](https://arxiv.org/abs/2512.09867) | 该论文提出了MedForget——一个面向医疗AI的层级感知多模态机器遗忘基准，通过将医疗机构数据建模为嵌套结构实现细粒度的遗忘与保留评估，并揭示了现有遗忘方法难以在删除目标数据的同时保持保留数据性能的问题。 |
| [^369] | [Latent Debate: A Surrogate Framework for Interpreting LLM Thinking towards Binary Decisions](https://arxiv.org/abs/2512.01909) | 本文提出“潜在辩论”框架，通过捕捉大语言模型在单次推理中内部潜在论证之间的支持与攻击信号，构建了与原模型预测高度一致且具备可解释性的结构化代理模型，用以解释LLM在真假预测任务上的二分类决策思考过程。 |
| [^370] | [VeriCypher: Named-Entity Verification for GraphRAG via Text-to-Cypher Generation](https://arxiv.org/abs/2511.08274) | 该论文提出VeriCypher框架，其核心创新在于命名实体验证机制——通过将生成的Cypher查询中的实体信息与目标图和模式进行核对并迭代修正，提升大语言模型在带标签属性图（LPG）上进行问答的查询准确性与可执行性。 |
| [^371] | [TOPol: Reconstructing and Interpreting Multidimensional Semantic Polarity Fields](https://arxiv.org/abs/2510.25069) | TOPol 颠覆了传统单维情感分析范式，通过递归二分法将主题映射到Transformer残差流中所有统计可复现的双极语义轴集合，从而多维地重构和解释语义极性场。 |
| [^372] | [A Survey on Agentic Security: Applications, Threats and Defenses](https://arxiv.org/abs/2510.06445) | 本文是首个对智能体安全领域进行全面综述的工作，通过274篇论文的分类体系和定量分析，从应用、威胁和防御三大支柱系统性地构建了智能体安全领域的研究框架。 |
| [^373] | [ViSSFlow: Unifying Video-Conditioned Sound and Speech Generation via Joint Learning](https://arxiv.org/abs/2509.24773) | 提出统一的流匹配框架 ViSSFlow，通过解耦的条件聚合机制（交叉注意力建模语义条件、自注意力捕获时序密集的视觉线索），将视频生成声音与视觉文本转语音两项任务统一在同一生成式框架下进行联合学习。 |
| [^374] | [RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory](https://arxiv.org/abs/2508.04903) | 提出角色条件上下文路由机制（RCR），在token预算约束下根据智能体角色选择性地将结构化共享记忆中的任务相关条目路由给各智能体，从而降低多智能体LLM系统的推理成本与冗余token消耗。 |
| [^375] | [Can Large Language Models Capture Video Game Engagement?](https://arxiv.org/abs/2502.04379) | 本文首次全面评估了大语言模型在多模态（文本+视频帧）提示下预测游戏视频玩家参与度的能力，通过4800多项实验发现，尽管LLM能超越传统机器学习基线，但在连续参与度标注任务上总体表现仍落后于专门方法。 |
| [^376] | [Theoretical Analysis of Byte-Pair Encoding](https://arxiv.org/abs/2411.08671) | 本文证明BPE背后的优化问题是APX完备的（因此不太可能有好的多项式时间近似方案），同时证明BPE在最坏情况下能以1/3至约0.4324之间的因子近似最优对编码的压缩效用，从而为BPE的实际良好表现提供了理论解释。 |
| [^377] | [HateSense: Hierarchical Sentiment Analysis Framework for Hate Speech Detection: Implementing Binary and Multiclass Classification Strategy](https://arxiv.org/abs/2411.05819) | 该研究通过受控实验发现，分层分类和情感特征在仇恨言论检测中并未带来显著收益，而类别加权训练才是唯一能大幅提升性能的有效干预措施。 |
| [^378] | [MBBQ: A Dataset for Cross-Lingual Comparison of Stereotypes in Generative LLMs](https://arxiv.org/abs/2406.07243) | 提出了MBBQ多语言偏见基准数据集，将英语BBQ数据集精心扩展至荷兰语、西班牙语和土耳其语，用于在控制文化差异和任务准确性的前提下，研究LLMs表现出的社会刻板印象是否随提示语言的不同而变化。 |
| [^379] | [A Big Data Approach to Understand Sub-national Determinants of FDI in Africa](https://arxiv.org/abs/2403.10239) | 本文提出了一种基于文本挖掘和社交网络分析的新方法，通过分析超过167,000篇在线新闻文章，量化了影响非洲公司外资直接投资所有权的区域级属性，结果表明次国家级的结构和制度特征在外资直接投资中扮演着重要角色。 |

# 详细

[^1]: WebHarbor：对接真实网站以构建可演化的GUI智能体环境

    WebHarbor: Docking Real Websites for Evolving GUI Agent Environments

    [https://arxiv.org/abs/2611.03634](https://arxiv.org/abs/2611.03634)

    WebHarbor通过任务驱动的人机协作，构建了153个忠实还原真实网站的Docker化可执行环境，提供涵盖多步骤和登录后工作流的3,518个任务，为GUI网页智能体提供稳定可控的评估与训练环境。

    

    GUI网页智能体需要在评估和训练过程中都保持稳定的、真实且可交互的环境。基于真实网站的评估受制于内容变化、访问限制以及登录后工作流覆盖有限等问题，而现有的用于训练的合成环境往往缺乏功能深度和视觉保真度。我们提出WebHarbor，这是一个包含153个可执行且已Docker化的网站环境套件，能够在确定性控制下忠实重建真实网络服务的视觉界面、内容和功能工作流。我们通过任务驱动的人机协作来构建WebHarbor：随着新任务扩展环境的功能范围，环境不断演化，编码智能体负责实现所需的更改，人类贡献者则负责审查视觉保真度、功能正确性和任务有效性。最终形成的套件包含3,518个任务，涵盖多步骤和登录后工作流，而现有（摘要原文在此处截断）……

    arXiv:2611.03634v1 Announce Type: cross  Abstract: GUI web agents require realistic and interactive environments that remain stable during both evaluation and training. Live website evaluation suffers from changing content, access restrictions, and limited coverage of authenticated workflows, while existing synthetic environments for training often lack functional depth and visual fidelity. We propose WebHarbor, a suite of 153 executable and dockerized website environments that faithfully reconstruct the visual interfaces, content, and functional workflows of real-world web services under deterministic control. We build WebHarbor through task-driven human-agent collaboration, where environments evolve as new tasks expand their functional scope, coding agents implement the required changes, and human contributors audit visual fidelity, functional correctness, and task validity. The resulting suite comprises 3,518 tasks spanning multi-step and post-authentication workflows, where existin
    
[^2]: D&A：解耦描述与评估使视觉语言模型成为零样本机器人奖励模型

    D&A: Decoupling Description and Assessment Enables VLMs as Zero-Shot Robotic Reward Models

    [https://arxiv.org/abs/2611.03622](https://arxiv.org/abs/2611.03622)

    提出D&A方法，将任务进度估计解耦为客观的场景描述与基于显式评分标准的评估两个阶段，使视觉语言模型无需昂贵后训练即可作为零样本机器人奖励模型，并有效避免确认偏差问题。

    

    视觉语言模型（VLM）在机器人学习中的零样本视频任务进度估计方面展现出巨大前景，然而现有框架常常受到确认偏差的困扰，由于隐含的成功假设或相对时间排序设计，会在失败的执行过程中幻觉出任务进度。对前沿模型进行朴素提示会退化为一个二元成功检测器，无法捕捉细微的、非线性的进度变化与失败情况。虽然后训练方法可以通过显式训练缓解这一问题，但它们需要昂贵的真实世界机器人数据，并且需要对每个单独的基础模型进行计算密集型的后训练。为了解决这些局限性，我们提出了D&A（描述与评估）：我们的方法将进度估计解耦为两个阶段——首先对场景动态进行客观的、基于任务的文本描述，随后使用显式评分标准进行评估。通过迫使VLM综合生成一个显式的、基于文本的定义……

    arXiv:2611.03622v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) hold promise for zero-shot video task-progress estimation in robot learning, yet existing frameworks frequently suffer from confirmation bias, hallucinating progress in failed executions due to implicit success assumptions or relative temporal ranking designs. Naive prompting of frontier models collapses into a binary success detector that fails to capture nuanced, non-linear progress and failures. While post-training approaches can mitigate this through explicit training, they demand expensive real-world robotics data and require computationally intensive post-training for every individual base model. To resolve these limitations, we introduce D&A (Describe & Assess): our approach decouples progress estimation into an objective, task-grounded textual description of scene dynamics followed by an assessment phase using explicit rubrics. By compelling the VLM to synthesize an explicit, text-based definition 
    
[^3]: Strategist：基于智能体的推理策略演化

    Strategist: Evolving Reasoning with Agents

    [https://arxiv.org/abs/2611.03618](https://arxiv.org/abs/2611.03618)

    Strategist是一个元智能体，通过将推理策略编写为可复用的蓝图并维护策略库，实现跨任务的推理策略自动演化，在30个基准测试上以低39%的推理成本达到与最强基线相当甚至更优的性能，且策略可迁移至未见任务和其他模型。

    

    模型只是AI智能体系统的一半，另一半是推理策略：即将模型调用转化为最终响应的推理过程。如今这另一半仍需人工编写，且构建某一策略的经验无法延续到下一个策略。我们提出了Strategist，一个能够在不同任务间演化推理策略的元智能体。每个策略都被编写为蓝图——即可被其他策略复用的可执行代码，所有蓝图共同构成一个策略库。给定一个任务和一个小型开发集，Strategist会检索相关蓝图、编写候选策略、在真实样本上进行测试，并根据测试结果进行修订。每个候选策略随后都会进入策略库，使后续任务能够建立在先前发现的基础之上。在30个基准测试中，Strategist在24个上击败了最强基线，并以低39%的推理成本达到与其相当的平均准确率（74.7% vs. 74.0%）。此外，其策略还能迁移到未见过的任务和其他模型上。

    arXiv:2611.03618v1 Announce Type: cross  Abstract: A model is only half of an AI agent system. The other half is the inference strategy: the reasoning procedure that turns model calls into a response. Today that second half is written by hand, and the experience of building one strategy never carries to the next. We introduce Strategist, a meta-agent that evolves inference strategies across tasks. Each strategy is written as a blueprint,executable code that other strategies can reuse, and together the blueprints form a library. Given a task and a small development set, Strategist retrieves related blueprints, writes candidate strategies, tests them on real samples, and revises them from the results. Every candidate then enters the library, so later tasks build on earlier discoveries. On 30 benchmarks, Strategist beats the strongest baseline on 24 and matches its average accuracy (74.7% vs.74.0%) at 39% lower inference cost. Its strategies also transfer to unseen tasks and other models 
    
[^4]: 追溯大语言模型科学发现能力与其训练数据的关联

    Tracing the Scientific Discovery Capacity of LLMs to Their Training Data

    [https://arxiv.org/abs/2611.03617](https://arxiv.org/abs/2611.03617)

    该研究提出“单次事实”理论，通过直接操纵训练数据，证明训练集中仅出现一次的事实比率是决定大语言模型科学发现能力的关键因素，且这一能力可在不同模型、领域间迁移。

    

    大型语言模型（LLM）在科学发现方面展现出越来越大的潜力，但关于什么因素使其能够可靠地做出发现，一些基本问题仍未解决。我们如何将LLM的科学发现能力追溯到其训练数据？我们将这一问题作为“缺失概率质量”问题进行研究，并建立了一个理论框架，其中“单次事实”——即训练中仅被观察到一次的事实——决定了发现能力。在材料科学和调控基因组学领域，我们直接操纵训练集中的单次事实比率，发现在两个开放权重模型家族、不同采样策略和推理预算下，更多的单次事实都能持续带来更多发现。进一步的实验表明，这种关系不能仅用训练集的多样性来解释，并且在与领域内数据配对时，这种能力可以跨科学领域迁移。总之，我们的结果确定了训练数据中的单次事实比率是决定语言模型发现能力的关键因素。

    arXiv:2611.03617v1 Announce Type: new  Abstract: Large language models (LLMs) show growing promise for scientific discovery, yet basic questions remain about what enables reliable discovery. How can we trace the scientific discovery capacity of LLMs to their training data? We study this question as one of missing probability mass and develop a theoretical account in which monofacts-the facts observed once during training-set the discovery capacity. In materials science and regulatory genomics, we directly manipulate monofact rates in training sets and find that more monofacts consistently yield more discoveries across two open-weight model families, sampling strategies, and reasoning budgets. Further experiments show the relationship cannot be explained by training-set diversity alone and can transfer across scientific domains when paired with in-domain data. Together, our results identify monofact rate in training data as a key factor governing discoveries in language models.
    
[^5]: ufakzeka-kulak：从零构建并评估一个31.6M参数的土耳其语语音识别模型

    ufakzeka-kulak: Building and Evaluating a 31.6M-Parameter Turkish Speech Recognizer from Scratch

    [https://arxiv.org/abs/2611.03605](https://arxiv.org/abs/2611.03605)

    该论文从零构建并开源了31.6M参数的土耳其语语音识别模型ufakzeka-kulak，仅使用可商业授权的数据训练，采用FastConformer编码器与TDT解码，在FLEURS和Common Voice土耳其语测试集上的词错误率均优于Whisper base和Whisper small。

    

    ufakzeka-kulak 是一个开源的土耳其语语音转文本模型，拥有31.6M参数，采用FastConformer编码器，配备令牌与时长转换器（TDT）和CTC头，从零开始训练并以Apache-2.0许可证发布。它输出的土耳其语文本保留大小写和标点符号，数字以单词形式书写。其4,977小时的训练数据均来自允许商业使用的许可来源；其中281小时为人工转录，其余4,696小时的训练文本来自上传者的字幕或机器转录文本，且仅在音频的第二次独立听写与之相符时才予以保留。训练配方、模型检查点、标注和导出选择均基于开发数据确定，测试集仅在训练前固定的发布门控环节中评分一次。使用贪心TDT解码时，归一化词错误率（WER）在FLEURS土耳其语测试集上为0.0921（95%置信区间为0.0792至0.1055），在Common Voice 27.0土耳其语测试集上为0.0916（0.0873至0.0958），均低于Whisper base和Whisper small模型。

    arXiv:2611.03605v1 Announce Type: new  Abstract: ufakzeka-kulak is an open Turkish speech-to-text model of 31.6M parameters, a FastConformer encoder with a token-and-duration transducer (TDT) and a CTC head, trained from scratch and released under Apache-2.0. It writes cased, punctuated Turkish with numbers as words. Its 4,977 training hours come from sources whose licences allow commercial use; 281 hours are human-transcribed, and for the other 4,696 the training text is an uploader subtitle or a machine transcript, kept only where a second, independent reading of the audio agrees with it. The recipe, checkpoint, labelling and export choices were made on development data, and the test sets were scored once, in a release gate fixed before training. With greedy TDT decoding the normalised word error rate (WER) is 0.0921 (95 percent interval 0.0792 to 0.1055) on FLEURS Turkish test and 0.0916 (0.0873 to 0.0958) on Common Voice 27.0 Turkish test, lower than Whisper base and Whisper small 
    
[^6]: DiabAgent：迈向智能体化生理控制

    DiabAgent: Toward Agentic Physiological Control

    [https://arxiv.org/abs/2611.03601](https://arxiv.org/abs/2611.03601)

    本文提出DiabAgent研究套件，通过APC智能体生理控制框架、DiabetSim糖尿病模拟器和SweetBench基准，首次系统性地实现并定量评估了医学智能体在1型糖尿病血糖持续控制中的能力。

    

    随着预训练规模的不断扩大，医学语言模型在临床问答方面取得了快速进展，但它们通过持续干预实现有效生理控制的能力仍缺乏系统性的定量评估。1型糖尿病为此提供了一个具体的研究场景：胰岛素分泌受损导致患者依赖外源性胰岛素，其剂量需要根据饮食、运动、疾病和血糖测量等信息持续调整。我们提出了DiabAgent，一个面向智能体化生理控制的研究套件，包含三个部分：智能体生理控制（APC）框架、DiabetSim模拟器和SweetBench基准测试。APC通过医疗设备工具控制血糖，利用持续观测来推断个体状态、积累个人经验并选择治疗动作。我们构建了DiabetSim来模拟不同的患者、事件和设备条件，以实现可扩展的虚拟测试与训练。

    arXiv:2611.03601v1 Announce Type: cross  Abstract: As pretraining scales, medical language models have advanced rapidly in clinical question answering. Their ability to achieve effective physiological control through sustained intervention still lacks systematic quantitative evaluation. Type 1 diabetes provides a concrete setting: impaired insulin secretion creates dependence on exogenous insulin, with dosing continually adjusted using information about meals, exercise, illness, and glucose measurements. We introduce DiabAgent, a research suite for agentic physiological control comprising the Agentic Physiological Control (APC) framework, the DiabetSim simulator, and the SweetBench benchmark. APC controls glucose through medical-device tools, using ongoing observations to infer individual states, accumulate personal experience, and choose treatment actions. We build DiabetSim to simulate different patients, events, and device conditions for scalable virtual testing and training. Using 
    
[^7]: Sonar-MoE：用于内存高效混合专家训练的反向传播内专家更新

    Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training

    [https://arxiv.org/abs/2611.03577](https://arxiv.org/abs/2611.03577)

    Sonar-MoE通过在反向传播过程中即时更新并释放专家梯度，将MoE训练的峰值内存降低19%–22%、每步速度提升17%–23%，同时保持与标准训练位级别一致的结果。

    

    混合专家模型的大多数参数都属于其专家模块，而标准训练会将每个专家的梯度保留在GPU内存中，直到优化器步进为止。这种内存占用会随着专家数量和规模的增长而增长，因此随着MoE模型的扩展而不断上升。我们提出了Sonar-MoE，它在反向传播过程中更新每个专家张量并立即释放其梯度。该方法将“反向传播内优化器”从稠密模型扩展到批处理的MoE专家张量，使GPU永远不会持有超过一层的专家梯度。在使用两种专家内核后端以bf16精度微调OLMoE-1B-7B和Granite-3.1-3B-A800M时，Sonar-MoE将峰值内存降低了19%到22%。其更新与不使用专家梯度裁剪的标准训练在位级别完全一致，留出损失与所有基线的差距保持在0.003以内。与使用全局范数裁剪的标准训练相比，每步训练时间也缩短了17%到23%。更新整个批处理张量而非单个张量……

    arXiv:2611.03577v1 Announce Type: cross  Abstract: Most parameters of a Mixture-of-Experts (MoE) model belong to its experts, and standard training keeps every expert gradient in GPU memory until the optimizer step. This memory grows with the number and size of the experts, so it rises as MoE models scale. We introduce Sonar-MoE, which updates each expert tensor during the backward pass and frees its gradient right away. It brings optimizer-in-backward from dense models to batched MoE expert tensors, so the GPU never holds more than one layer of expert gradients. When we fine-tune OLMoE-1B-7B and Granite-3.1-3B-A800M in bf16 with two expert-kernel backends, Sonar-MoE lowers peak memory by 19% to 22%. Its updates are bitwise identical to standard training without expert gradient clipping, and held-out loss stays within 0.003 of all baselines. Each step also takes 17% to 23% less time than standard training with global-norm clipping. Updating whole batched tensors rather than individual 
    
[^8]: NitiBench+：一个涵盖17个领域的时序泰国法律问答基准

    NitiBench+: A Temporal Thai Legal Question Answering Benchmark Across 17 Domains

    [https://arxiv.org/abs/2611.03576](https://arxiv.org/abs/2611.03576)

    NitiBench+是一个包含1,670个条目、覆盖17个领域、同时收录判决日有效与当前法条文本的泰国时序法律问答基准，实验表明模型使用过时法条文本会显著损害法律问答表现。

    

    法律会随时间变化。法院依据判决当日有效的法律来裁决案件，而现有的法律问答基准却按照今天的法律来给答案评分。我们检查了泰国四个法院的156,650份判决，发现其中高达67%引用了此后已被修订或废止的法律。我们推出NitiBench+，它是我们此前NitiBench的续作，是一个泰国法律问答基准，包含基于三个法院判决构建的1,670个条目，覆盖17个法律领域。对于每个被引用的法条，该基准同时保存判决当日有效的文本和当前文本，其检索语料库则收录了泰国每一部法典每个条目的所有带日期的历史版本。答案的评分采用泰国律师公会的评分细则和律师标准分数，并根据模型引用的法条与法院实际引用法条之间的重合度进行评判。报告分为两部分：第一部分测量了使用错误文本在13个开放权重模型上造成的损失，阅读已修订法律的当前文本会使律师标准分数下降0.25，遗漏已废止法律则会使分数下降0.79。

    arXiv:2611.03576v2 Announce Type: new  Abstract: Law changes over time. A court decides a case under the law in force on that day, and legal QA benchmarks grade answers against today's law. We checked 156,650 decisions of four Thai courts. Up to 67% of them cite law that has since been amended or repealed. We introduce NitiBench+, the sequel of our NitiBench, a Thai legal QA benchmark with 1,670 items across 17 domains, built from decisions of three courts. For each cited section it holds both the text in force on the decision date and the current text, and its retrieval corpus holds every dated version of every Thai statute section. Answers are graded on the Thai Bar Council rubric, the Bar-standard score, and on the overlap between the cited sections and the court's. The report has two parts. Part I measures the cost of the wrong text on 13 open-weight models. Reading the current text of amended law drops the Bar-standard score by 0.25 and missing repealed law drops it by 0.79. The p
    
[^9]: VeriHarm-Bench：可验证地测量LLM编程代理中的有害成功

    VeriHarm-Bench: Measuring Harmful Success in LLM Coding Agents Verifiably

    [https://arxiv.org/abs/2611.03575](https://arxiv.org/abs/2611.03575)

    该论文提出了VeriHarm-Bench基准，首次定义并系统地测量LLM编程代理中的“有害成功”现象——即代理在成功完成良性任务的同时造成非预期的危害，而传统任务成功指标和现有安全评估均无法捕捉这一问题。

    

    大语言模型（LLM）编程代理越来越多地以shell访问权限运行，使其能够编写脚本、移动文件、更改权限和修改系统配置。然而，代理可能在实现用户目标的同时造成非预期的危害，例如删除无关文件、削弱访问控制或损坏数据，且这种情况可在响应良性输入时发生，而无需显式的越狱攻击。这种失败模式尤为重要，因为表面上的任务成功可能让用户确信代理行为正确，从而导致有害的副作用被忽视。我们将这种现象称为“有害成功”：代理在成功完成预期的良性任务的同时，造成了非预期的危害。尽管其重要性，有害成功难以测量，因为传统的任务成功指标忽略了有害行为，而现有的安全评估通常仅设计用于评估有害结果。我们提出了VeriHarm，一个自动……

    arXiv:2611.03575v1 Announce Type: new  Abstract: Large language model (LLM) coding agents increasingly operate with shell access, allowing them to write scripts, move files, alter permissions, and modify system configurations. However, an agent may achieve a user's goal while causing unintended harm, such as deleting unrelated files, weakening access controls, or corrupting data, in response to benign inputs without explicit jailbreaking. This failure mode is particularly important because apparent task success may reassure users that an agent behaved correctly, causing harmful side effects to go unnoticed. We call this harmful success: the agent successfully completes the intended benign task while simultaneously causing unintended harm. Despite its importance, harmful success is difficult to measure because conventional task-success metrics overlook harmful behaviors, while existing safety evaluations are often designed solely to assess harmful outcomes. We introduce VeriHarm, an aut
    
[^10]: LLM智能体能胜任一线部署工程师吗？面向企业级文本到SQL的需求发现与交付

    Can LLM Agents Be Forward-Deployed Engineers? Requirement Discovery and Delivery for Enterprise Text-to-SQL

    [https://arxiv.org/abs/2611.03551](https://arxiv.org/abs/2611.03551)

    本研究在企业级文本到SQL的真实项目场景中评估了LLM智能体担任“一线部署工程师”的能力，发现即便最强的模型也只能发现约80%的隐藏需求，且各配置中仅25%的交付运行能通过全部验收标准，并总结出需求发现不完整等四大瓶颈。

    

    LLM编程智能体日益能够自动化模型后训练流程，但它们能否胜任“一线部署工程师”这一角色——即发现客户需求并将其转化为部署决策——仍不明确。我们在一个真实的企业级文本到SQL项目场景中对此展开研究：一个基于真实用例需求构建的模拟客户，最初仅提出一个模糊的请求。智能体必须在发现阶段探索出验收标准、数据权限和硬件约束，然后在交付阶段规划并执行后训练方案。在15个模型的评估中，表现最强的模型能发现超过80%的隐藏需求，开源权重模型能发现40-50%；在交付阶段，八种编程智能体配置中仅有25%的运行通过了全部验收标准。我们识别出该场景中的四个瓶颈：（1）需求发现不完整，40%的隐藏需求缺失或被错误表述；（2）执行不可靠……（原文摘要在此处截断）

    arXiv:2611.03551v1 Announce Type: new  Abstract: LLM coding agents can increasingly automate model post-training, but whether they can act as forward-deployed engineer (FDE) agents, discovering customer requirements and turning them into deployment decisions, remains unclear. We study this on a real enterprise text-to-SQL engagement: a simulated customer, grounded in the real use case's requirements, starts with an ambiguous request. An agent must discover acceptance criteria, data permissions, and hardware constraints in the discovery phase, then plan and execute a post-training recipe in the delivery phase. Across 15 models, the strongest discover over 80% of withheld requirements and open-weight models discover 40-50%; in the delivery phase, only 25% of runs across eight coding-agent configurations pass all acceptance criteria. We identify four bottlenecks in this engagement: (1) incomplete discovery, with 40% of withheld requirements missing or misrepresented; (2) unreliable execut
    
[^11]: 层库（LoL）：在运行时自适应地排序Transformer层

    Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime

    [https://arxiv.org/abs/2611.03526](https://arxiv.org/abs/2611.03526)

    提出层库架构，使模型能在推理时动态调整Transformer层的执行顺序，用一个统一框架同时概括了密集、深度循环和专家混合架构在参数效率与计算效率上的权衡。

    

    计算机程序员将相同的可重用计算原语（例如条件语句或循环）以不同的顺序链接在一起，从而实现不同的算法。相比之下，现有的语言模型架构总是以相同的顺序执行其各层。我们提出了层库（Library of Layers，LoL），这是一种在推理过程中动态调整层执行顺序以解决当前问题的架构。LoL只有一个超参数，即其全局Transformer块库的大小，这使其能够在参数重用与稀疏性之间无缝转换，将密集架构、深度循环架构和专家混合架构在参数效率与FLOPs效率上的权衡统一到一个框架家族中。我们证明，在文本数据上进行预训练时，LoL与上述三种基线架构相比均取得了有竞争力的性能。在合成的数独和迷宫数据上，我们观察到该模型针对不同问题采用不同的层排序……（原文摘要不完整）

    arXiv:2611.03526v1 Announce Type: cross  Abstract: Computer programmers chain together the same reusable computational primitives, for example, conditional statements or loops, but in different orders to implement different algorithms. In contrast, existing language model architectures always execute their layers in the same order. We introduce Library of Layers (LoL), an architecture that dynamically orders its layers during inference to solve the problem at hand. LoL has a single hyperparameter, the size of its global library of transformer blocks, that allows it to transition seamlessly between parameter reuse and sparsity, generalizing the parameter-efficiency and FLOPs-efficiency tradeoffs of dense, depth-recurrent, and mixture of experts architectures in one unified family. We show that LoL achieves competitive performance with all three above baselines when pretrained on text data. On synthetic sudoku and maze data, we observe that the model employs distinct layer orderings to i
    
[^12]: VIAS：用于组织多价值对齐与合规的价值隔离激活引导

    VIAS: Value-Isolating Activation Steering for Multi-Value Alignment and Compliance for Organisations

    [https://arxiv.org/abs/2611.03518](https://arxiv.org/abs/2611.03518)

    提出价值隔离激活引导方法VIAS，通过对比包含全部价值观与省略单一目标价值观的两种提示下模型回答的差异，构建能够单独分离各价值观的引导向量，从而实现大语言模型的多价值观对齐与组织合规。

    

    组织越来越多地部署必须遵循组织价值观的大语言模型智能体。我们检验激活引导——即在生成过程中向模型内部激活添加向量而不改变其权重——能否在仅靠提示之外进一步提升合规性。由于大多数场景同时涉及多个价值观，我们专注于多价值观引导。构建这些引导向量的一种简单方法是将遵循场景相关所有价值观的回答与违反这些价值观的回答进行对比。然而，这种方法对不同价值观得到的向量几乎完全相同：它们捕获的是一种通用的合规信号而非各个独立的价值观，这限制了它们在组合多个价值观时的使用。我们提出价值隔离激活引导（VIAS），该方法对比同一场景在两种提示下的回答：一种提示列出所有相关价值观，另一种提示省略单个目标价值观。只有当目标价值观发生改变时，我们才保留这一对比对……（摘要内容在此处被截断）

    arXiv:2611.03518v1 Announce Type: new  Abstract: Organisations increasingly deploy large language model agents that must follow organisational values. We check whether activation steering, which adds vectors to a model's internal activations during generation without changing its weights, can improve compliance beyond prompting alone. Since most scenarios involve several values at once, we focus on multi-value steering. A simple way to build these vectors is to contrast responses that follow all values relevant to a scenario with responses that violate them. However, this gives nearly identical vectors for different values: they capture a general compliance signal rather than individual values, which limits their use when combining several values. We propose Value-Isolating Activation Steering (VIAS), which contrasts responses to the same scenario under two prompts, one listing all relevant values and one omitting a single target value. We keep a pair only if the target value changes f
    
[^13]: 面向持续语言模型训练的在线无标签梯度匹配

    Online Label-Free Gradient Matching for Continual Language Model Training

    [https://arxiv.org/abs/2611.03496](https://arxiv.org/abs/2611.03496)

    OLGA提出了一种在线无标签的梯度匹配数据蒸馏方法，利用模型自身的下一个词预测作为梯度匹配目标，摆脱了对真实标签的依赖，能够持续处理不断增长的数据流，在持续语言模型训练中显著优于现有需要标签的方法。

    

    面向客户的产品中的语言模型必须持续更新，因为它们所服务的数据会随时间发生变化。由于在全部累积数据上重新训练是不切实际的，每次更新应该在一小组能够携带迄今为止所见全部数据训练信号的样本上进行。当前最强的方法通过梯度匹配来构建这样的数据集，即合成训练梯度与全部数据梯度相匹配的样本。然而，这些方法需要真实标签，只能对固定快照运行一次而无法处理不断增长的数据流，并且在大型数据上速度缓慢。我们提出了OLGA，它克服了这些局限。它通过从模型自身的下一个词预测中提取梯度匹配目标，来蒸馏无标签数据。在五个文本分类基准测试中使用phi-1.5模型，仅使用五个样本，OLGA的准确率就比最强的数据选择基线提高了10个百分点。OLGA还能蒸馏在线数据流……

    arXiv:2611.03496v1 Announce Type: cross  Abstract: Language models in customer-facing products must be continually updated, since the data they serve shifts over time. As retraining on the full accumulated data is impractical, each update should instead run on a small set of examples carrying the training signal of all data seen so far. The strongest current methods build this set by gradient matching, which synthesizes examples whose training gradients match those of the full data. However, they require ground-truth labels, run once on a fixed snapshot rather than on a growing stream, and are slow on large data. We introduce \textbf{OLGA}, which lifts these limitations. It distills unlabeled data by drawing the gradient matching targets from the model's own next-token predictions. Across five text classification benchmarks with phi-1.5, it improves accuracy by 10 percentage points over the strongest selection baseline using as few as five examples. OLGA also distills online data strea
    
[^14]: 优化智能体框架而非模型：将最小智能体适配到可自托管的大语言模型以进行调查分析

    Optimize the Harness, Not the Model: Fitting a Minimal Agent to a Self-Hostable LLM for Investigative Analysis

    [https://arxiv.org/abs/2611.03434](https://arxiv.org/abs/2611.03434)

    提出CLEO方法，通过优化最小智能体框架而非模型本身，使其适配可自托管的大语言模型，在调查分析任务上以小两个数量级的规模达到接近最强商用基线的83.8%通过率。

    

    政府情报团队需要能够分析大量杂乱证据集合的大语言模型智能体，但安全和成本方面的约束往往排除了托管的前沿模型以及广泛的、难以审计的工具面。我们提出CLEO（环境优化的约束循环），这是一种将最小智能体框架适配到固定的、可自托管的大语言模型的方法。在给定专家验证的数据集的情况下，CLEO只保留那些能够提高准确性、同时保持通用性和可审计性的工具、防护机制、内存策略和循环组件。在我们的保留测试集上，CLEO框架达到了83.8%的通过率，与最强的开箱即用Claude Code SDK基线相差仅1.2个百分点，而其规模要小两个数量级。

    arXiv:2611.03434v1 Announce Type: cross  Abstract: Government intelligence teams need LLM agents that can analyze large, messy evidence collections, but security and cost constraints often rule out hosted frontier models and broad, hard-to-audit tool surfaces. We report CLEO: Constrained Loop for Environment Optimization, a method for fitting a minimal agent harness to a fixed, self-hostable LLM. Given an expert-validated dataset, CLEO admits only the tools, guardrails, memory policies, and loop components that improve accuracy while remaining general and auditable. On our held-out dataset, the CLEO harness reaches an 83.8% pass rate, within 1.2 points of the strongest out-of-the-box Claude Code SDK baseline, while being two orders of magnitude smaller.
    
[^15]: IdeaTaste：大语言模型是否拥有与人类研究者相同的研究品味？

    IdeaTaste: Do LLMs Share the Research Taste of Human Researchers?

    [https://arxiv.org/abs/2611.03424](https://arxiv.org/abs/2611.03424)

    IdeaTaste基准通过将LLM基于2025年文献快照撰写的研究提案与人类社区2026年的实际发表工作进行匹配评分，发现LLM在引用匹配和想法匹配上仅达到人类水平的约26%和18%，揭示了LLM尚不具备人类研究者的研究品味。

    

    研究品味，即对阅读什么文献、追求哪些问题、尝试哪些方法的判断力，正日益塑造着AI科学家的工作，但相比其执行研究的能力，这一能力却远未被充分理解。我们提出了IdeaTaste，这是一个基准测试，通过衡量大语言模型基于2025年冻结的文献快照所撰写的研究提案，与相同研究社区在2026年实际发表工作的匹配程度来为其评分。我们的实验涵盖六个当代语言模型和四个研究领域（从成熟领域到新兴领域），共计超过90,000份提案。实验表明，LLM智能体在所引用的文献和所陈述的想法方面都远不及人类社区：在引用匹配上仅达到人类水平的约26%，在想法匹配上仅达到18%。LLM群体的引用多样性可以达到社区水平，也能采纳社区的任务和方法，但很少能提出该社区所追求的具体研究方法和研究计划。

    arXiv:2611.03424v1 Announce Type: new  Abstract: Research taste, the judgment of what to read, which problems to pursue, and which methods to try, increasingly shapes the work of AI scientists, yet it remains far less understood than their ability to execute research. We introduce IdeaTaste, a benchmark that scores an LLM by how closely the research proposals it writes from a literature snapshot frozen at 2025 match the work the same communities published in 2026. Our experiments span six contemporary language models and four research areas from established to emerging fields, covering more than 90{,}000 proposals in total. LLM agents fall well short of human communities in both the literature they cite and the ideas they state, reaching about 26\% of the human level on citation matching and only 18\% on idea matching. Their populations can cite as diversely as a community and take up its tasks and methods, yet rarely arrive at the concrete approaches and research plans it pursues. Nei
    
[^16]: 相同的输出，不同的分数：审计多模态LLM基准测试中未经检验的评分器

    Same Outputs, Different Scores: Auditing the Unchecked Graders of Multimodal LLM Benchmarks

    [https://arxiv.org/abs/2611.03377](https://arxiv.org/abs/2611.03377)

    该研究首次系统审计了多模态LLM基准测试中隐藏的评分规则，发现仅评分器本身就能在模型输出完全相同的情况下使准确率产生高达22个百分点的差异，并提出了人类与LLM裁判结合的盲测仲裁方法来恢复模型的真实答案。

    

    每个多模态LLM选择题基准测试都以一条评分规则收尾，该规则将模型的回答（通常是自由格式的文本，而非干净整洁的选项字母）转换为对/错的判定，而这条规则深埋在代码之中。与数据集不同，这条规则的选择没有任何依据支撑，也从未受到审查，然而它却决定了最终发布的分数。我们在五个可适配单块16GB GPU的开源视觉语言模型（VLM）上复现了八个经同行评审的基准测试的官方评分器，并引入了一种盲测的人类加LLM裁判仲裁方法（Cohen's kappa=0.956），用以恢复模型实际给出的答案。仅评分规则本身就能使完全相同输出的准确率产生高达22个百分点的波动，但当模型输出规范的选项字母时，这种波动恰好为0，因此该问题在强模型排行榜上被掩盖了。真正造成损害的是评分器的回退机制，而非模型冗长的回答：我们测得非规范输出率最高的基准测试其评分伪差恰好为零，因为其评分器在回退时……

    arXiv:2611.03377v1 Announce Type: cross  Abstract: Every multimodal-LLM multiple-choice benchmark ends with a scoring rule that converts a model's response, often free-form text rather than a clean option letter, into a right/wrong verdict, buried in its code. Unlike the dataset, this rule is chosen without justification and left unexamined, yet it sets the published number. We reproduce the official scorers of eight peer-reviewed benchmarks on five open vision-language models (VLMs) that fit a single 16 GB GPU, and introduce a blinded human-plus-LLM-judge adjudication (Cohen's kappa=0.956) that recovers what the model actually answered. The scoring rule alone moves accuracy by up to 22 points on identical outputs, yet exactly 0 wherever a model emits clean option letters, so it hides on strong-model leaderboards. It is the fallback and not the rambling that does the damage: the benchmark with the highest non-compliance we measure has an artifact of exactly zero, because its scorer abs
    
[^17]: 刻画语言模型何时为保护其被赋予的目标而采取有害行动

    Characterizing When Language Models Take Harmful Actions to Protect Their Assigned Goals

    [https://arxiv.org/abs/2611.03374](https://arxiv.org/abs/2611.03374)

    该论文将智能体失准、伪装对齐等看似不同的语言模型安全失败现象统一形式化为“目标-伤害冲突”这一共同决策结构，并通过从 logit 概率到开放式采样的多种引出协议，系统刻画了模型何时会为保全被赋予的目标而采取有害行动。

    

    近期语言模型（LM）的安全失败在很大程度上被当作彼此孤立的现象来研究。在智能体失准中，模型会敲诈一名高管以避免被替换；在伪装对齐中，模型会在训练期间服从有害请求，以防止其价值观被修改。我们认为这些行为共享一个共同的决策结构：一个被赋予目标的模型必须要么采取有害行动来保全该目标，要么拒绝并失去它。我们将这一情形形式化为“目标-伤害冲突”，并通过从紧凑的强制选择题提示到模拟智能体环境等多种引出协议来呈现同一冲突。这些协议涵盖了语言模型决策的两个视角，我们分别称之为 logit 视角（从答案 token 的概率中读取偏好）和采样视角（用语言模型裁判对开放式回答进行分类）。在 logit 视角下，我们表明目标重要性和伤害严重程度以相加的方式塑造……（原文摘要在此处截断）

    arXiv:2611.03374v1 Announce Type: cross  Abstract: Recent language model (LM) safety failures have largely been studied as disparate phenomena. In agentic misalignment, a model blackmails an executive to avoid being replaced. In alignment faking, it complies with harmful requests during training to keep its values from being modified. We argue that these behaviours share a common decision structure: a model assigned a goal must either take a harmful action to preserve it or refuse and lose it. We formalize this as a goal-harm conflict and present the same conflict under various elicitation protocols that range from compact forced-choice prompts to simulated agentic environments. These protocols span two perspectives on LM decision-making, which we term (i) the logit view, reading preferences from answer-token probabilities, and (ii) the sampling view, classifying open-ended responses with an LM judge. Under the logit view, we show that goal importance and harm severity additively shape
    
[^18]: 利用多语言词典进行激活导向以实现低资源语言的情感控制

    Steering with Multilingual Lexicons for Sentiment Control in Low-Resource Languages

    [https://arxiv.org/abs/2611.03317](https://arxiv.org/abs/2611.03317)

    该论文提出用多语言情感词典替代句子级对比数据来实现激活导向的情感控制，证明聚合多语言词典对低资源语言优于英语导向向量，并从机制上揭示了“语言坍塌”现象及架构偏见的约束作用。

    

    激活导向为控制模型行为提供了一种轻量级机制，但现有方法依赖于句子级对比数据，而世界上大多数语言都无法获得这类数据。我们证明，多语言情感词典是三种模型家族中可行的替代方案：对于中低资源语言，聚合的多语言词典优于源自英语的导向向量；而对于模型在预训练期间专门学习过的语言，单语言词典仍然更强。导向效果在机制上由表示密度和预训练先验共同决定。大幅度值的英语向量会压垮低资源语言的紧凑表示，导致“语言坍塌”现象。多语言聚合所带来的范数阻尼效应可以缓解这种脆弱性。此外，架构偏见——包括双语特化惩罚和潜在数值刚性——也严重限制了（导向的效果）……

    arXiv:2611.03317v1 Announce Type: new  Abstract: Activation steering offers a lightweight mechanism for controlling model behavior, but existing methods depend on sentence-level contrastive data that is unavailable for most of the world's languages. We show that multilingual sentiment lexicons are a viable substitute across three model families: aggregated multilingual lexicons outperform English-derived steering vectors for mid- and low-resource languages, while monolingual lexicons remain stronger for languages a model has specialized in during pre-training. The steering efficacy is mechanistically dictated by both representational density and pre-training priors. Large-magnitude English vectors overwhelm the compact representations of low-resource languages, causing "linguistic collapse." This fragility is mitigated by the norm-dampening of multilingual aggregation. Architectural biases, including bilingual specialization penalty and latent numerical rigidity, also heavily constrain
    
[^19]: MultiInpaintCOCO：多语言细粒度视觉理解

    MultiInpaintCOCO: Fine-Grained Visual Understanding in Multiple Languages

    [https://arxiv.org/abs/2611.03309](https://arxiv.org/abs/2611.03309)

    该论文提出MultiInpaintCOCO——一个覆盖13种语言、包含953个最小对示例的多语言细粒度视觉理解基准，发现视觉-语言模型在低资源语言上性能显著下降，其主因是语言数据资源的匮乏而非与英语的类型学相似度，且模型在跨语言检索中存在对英语的系统性偏差。

    

    先前的研究表明，视觉-语言（VL）模型难以理解图像的细粒度细节，往往只关注整体布局。此外，模型在英语上的任何弱点在多语言环境中往往会被进一步放大。为了研究这一问题，我们将InpaintCOCO（一个专注于细粒度视觉细节的英语数据集）扩展到涵盖五个语系的13种类型学上多样化的语言，创建了MultiInpaintCOCO——一个包含953个最小对示例的新型多语言基准。我们进一步在多语言多模态检索任务中测试了多个最先进的模型，既包括轻量级的类CLIP视觉-语言编码器，也包括大型语言模型。我们的结果表明，低资源语言的性能显著下降，而最强的预测因素是该语言可用的数据资源，而非与英语的类型学相似度。在跨语言检索中，大多数模型表现出对英语的系统性偏差：w……

    arXiv:2611.03309v1 Announce Type: new  Abstract: Previous research shows that vision-language (VL) models struggle to understand fine-grained image details and tend to focus on the overall layout. Moreover, any weakness in English tends to be further amplified in the multilingual setting. To study that, we extend InpaintCOCO, an English dataset focused on fine-grained visual details, to 13 typologically diverse languages across five language families, creating MultiInpaintCOCO - a new multilingual benchmark of 953 minimal-pair examples. We further test several state-of-the-art models in multilingual multimodal retrieval, considering both light VL CLIP-like encoders and large language models. Our results show that performance degrades substantially for lower-resource languages, with the strongest predictor being the data resources available for the languages rather than typological similarity to English. In cross-lingual retrieval, most models exhibit a systematic bias toward English: w
    
[^20]: 大语言模型在MS/MS结构解析的分布外泛化中依赖专业工具

    LLMs Depend on Specialist Tools for Out-of-Distribution Generalization in MS/MS Structural Elucidation

    [https://arxiv.org/abs/2611.03308](https://arxiv.org/abs/2611.03308)

    前沿大语言模型在MS/MS从头结构解析中，分布内可达52%准确率但分布外骤降至2%，表明其泛化能力主要依赖记忆；唯有引入从头结构生成器等专业工具才能将分布外准确率提升至38%。

    

    近期有报道称，一个前沿语言模型能够在不使用工具的情况下解决一维NMR结构解析问题。我们探讨这种能力是否可以迁移到从头解析设定下的串联质谱（MS/MS）中，即模型必须直接从谱图提出结构，而没有候选列表或参考库。为了将推理能力与记忆能力区分开，我们基于MassSpecGym构建了一个包含100个分析物的基准，并根据每个分析物的文本足迹将其划分为分布内（文献中有充分记载）和分布外（基本无记载）两部分。我们保持模型、测试框架和输出格式不变，仅改变LLM对领域专用工具的访问权限。在没有工具的情况下，前沿LLM在分布内达到52%的Top-1准确率，但在分布外仅有2%。只有访问从头结构生成器才能提高分布外准确率，将其提升至最高38%的Top-1，且由此产生的智能体优于独立的从头结构……

    arXiv:2611.03308v1 Announce Type: cross  Abstract: A frontier language model was recently reported to solve 1D NMR structural elucidation without tools. We ask whether this capability transfers to tandem mass spectrometry (MS/MS) in the de novo setting, where the model must propose a structure directly from the spectrum without a candidate list or reference library. To separate reasoning from recall, we build a 100-analyte benchmark from MassSpecGym, split by each analyte's textual footprint into in-distribution (well documented in literature) and out-of-distribution (essentially undocumented) arms. We hold the model, harness, and output format fixed and vary only the LLM's access to domain-specific tools. Without tools, a frontier LLM reaches 52% Top-1 accuracy in-distribution but only 2% out-of-distribution. Only access to de novo structure generators improves out-of-distribution accuracy, raising it to as much as 38% Top-1, and the resulting agent outperforms the standalone de novo 
    
[^21]: 特异性偏差与维基百科悖论：域偏移下AI文本检测器的可解释AI（xAI）分析

    Specificity Bias and the Wikipedia Paradox: An xAI Analysis of AI Text Detectors under Domain Shift

    [https://arxiv.org/abs/2611.03280](https://arxiv.org/abs/2611.03280)

    该研究通过基于SHAP的可解释AI框架发现，AI文本检测器在域偏移下主要依赖指令微调大模型特有的“行动号召”风格而非通用机器文本特征进行判断，且大语言模型可通过幻觉出特定命名实体的“特异性偏差”策略绕过检测。

    

    arXiv:2611.03280v1 公告类型：新 摘要：大语言模型（LLM）的广泛使用要求能够可靠地区分人类撰写文本与合成文本。然而，当前基于Transformer的检测器在域偏移（domain shift）情况下往往出现严重的性能下降，并且作为不透明的分类器运行。本文研究了经过微调的Transformer模型（DistilBERT、RoBERTa）的鲁棒性与可解释性，方法是将它们在自定义的分布外（OOD）数据集上进行评估，该数据集由通过开源大语言模型（Llama 3、Mistral、Gemma）合成的超本地新闻组成。为分析模型的决策边界，我们应用了基于SHAP的三方可解释AI（xAI）框架。我们的分析揭示了三个关键现象：首先，检测器往往依赖于指令微调LLM所固有的“行动号召”（Call-to-Action）特征，而非通用的机器文本特征。其次，我们观察到一种“特异性偏差”（Specificity Bias），即LLM通过幻觉出特定的命名实体并模仿具体……（摘要原文在此处截断）

    arXiv:2611.03280v1 Announce Type: new  Abstract: The widespread adoption of Large Language Models (LLMs) demands reliable detection of human-authored versus synthetic text. However, current Transformer-based detectors often experience severe performance degradation during domain shifts and operate as opaque classifiers. This paper investigates the robustness and interpretability of fine-tuned Transformer models (DistilBERT, RoBERTa) by evaluating them against a custom out-of-distribution (OOD) dataset of hyper-local news synthesized via open-source LLMs (Llama 3, Mistral, Gemma). To analyze model decision boundaries, we apply a tripartite Explainable AI (xAI) framework using SHAP. Our analysis reveals three key phenomena: First, detectors often rely on a "Call-to-Action" syndrome inherent to instruction-tuned LLMs rather than generic machine text. Second, we observe a "Specificity Bias," where LLMs successfully bypass detection by hallucinating specific named entities and mimicking con
    
[^22]: 智能体还会遵守规则吗？在智能体上下文压缩下的规则遵守基准测试

    Do Agents Still Follow the Rules? Benchmarking Rule Compliance under Agent Context Compaction

    [https://arxiv.org/abs/2611.03223](https://arxiv.org/abs/2611.03223)

    提出 ARCBench 基准，在完整智能体框架中通过实际执行动作测量上下文压缩后的规则遵守情况，并发现压缩后的大多数违规并非因为规则被删除，而是在规则仍保留于压缩历史中时智能体对后续请求处理不当所致。

    

    大语言模型智能体会压缩长历史记录以适应其上下文窗口，但压缩后的上下文仍必须承载约束后续操作的运行规则。先前的研究表明压缩后违规行为会增加，但这些工作在完整的智能体框架之外测量合规性，且未能识别每项违规的具体来源。我们提出了 ARCBench（智能体规则合规基准），这是一个构建于开源 pi 智能体框架之上的可配置基准测试。其基于工具调用的智能体循环使我们能够从执行期间采取的实际动作中测量规则合规性，使评估更贴近实际部署的智能体工作流。借助 ARCBench，我们研究了压缩后违规行为的来源，以及不同的压缩配置和对抗性设置如何改变这些来源。我们发现压缩很少会删除规则本身：压缩后的大多数违规行为发生在规则仍然存在于压缩历史中的情况下，此时智能体将后续请求视为…（原文摘要至此截断）

    arXiv:2611.03223v1 Announce Type: cross  Abstract: LLM agents compact long histories to fit their context windows, but the compacted context must still carry the operating rules that govern later actions. Prior work shows that violations rise after compaction, but measures compliance outside a complete agent harness and does not identify where each violation comes from. We introduce ARCBench(Agent Rule Compliance Benchmark), a configurable benchmark built on the open-source pi agent harness. Its tool-calling agent loop lets us measure rule compliance from actions taken during execution, bringing the evaluation closer to deployed agent workflows. Using ARCBench, we examine where violations after compaction come from, and how different compaction configurations and adversarial settings change their sources. We find that compaction rarely deletes the rule: most violations after compaction occur while the rule is still in the compacted history, with the agent treating a later request as pe
    
[^23]: 一次写入，各异读取：语言模型自身的答案如何引导其后续回答

    Written Once, Read Differently How Language Models Own Answers Steer Their Next Ones

    [https://arxiv.org/abs/2611.03198](https://arxiv.org/abs/2611.03198)

    语言模型对是非问题的回答会通过在残差流中写入一个稳定的、跨主题共享的“承诺方向”来引导其对后续相关问题的回答，但该方向被读取的机制并不稳定，会随模型规模、指令微调乃至系统提示词而改变。

    

    当语言模型回答一个是非问题时，该答案会改变它对关于同一实体的另一个问题的回答方式。我们在来自四个模型家族（参数量0.5B–72B）的22个开源权重模型中研究了这种“自我承诺”现象：固定模型的第一个答案，并测量其对关于120个实体的3,600个问题对中第二个答案的偏移。每个答案会向残差流中写入一个“承诺方向”。两个问题的方向相似性可以预测它们在留出实体上的耦合程度（ρ = 0.43–0.85），并且在每个模型中，消融这个跨主题共享的单一方向后，最多只留下17%的效应。然而，这一方向被“读取”的方式则远不稳定：随着模型规模增大，读取回路分布在更多的注意力头上；指令微调使更多问题以相反的意义读取该方向；在某个模型中，仅改变系统提示词就能使该回路的符号反转。在保留[……原文此处截断]的同时消融该方向……

    arXiv:2611.03198v1 Announce Type: new  Abstract: When a language model answers a yes/no question, that answer changes how it answers a different question about the same entity. We study this \emph{self-commitment} in 22 open-weight models from four families (0.5B--72B parameters), fixing the model's first answer and measuring the shift in its second across 3,600 question pairs about 120 entities. Each answer writes a \emph{commitment direction} into the residual stream. The similarity of two questions' directions predicts their coupling on held-out entities ($\rho = 0.43$--$0.85$), and ablating a single direction shared across topics leaves at most 17\% of the effect in every model. How this direction is read is far less stable: the reading circuit spreads across more attention heads as models grow, instruction tuning makes more questions read it in the opposite sense, and in one model changing only the system prompt reverses the circuit's sign. Ablating the direction while sparing the
    
[^24]: Dhārā Bench：面向印度知识体系的证据导向检索基准测试

    Dh\={a}r\={a} Bench: An Evidence-Oriented Retrieval Benchmark for Indian Knowledge Systems

    [https://arxiv.org/abs/2611.03196](https://arxiv.org/abs/2611.03196)

    本文构建了 Dhārā Bench——首个面向印度知识体系的证据导向检索基准，包含 325 个多类别问题、60,121 个文本片段及 7,578 条相关性标注，并通过 BM25 基线实验揭示了该领域检索的挑战性。

    

    Dhārā Bench 是一个面向印度知识体系的检索基准测试，包含 12 个类别中的 325 个问题，以及来自 15 个源文献组的 60,121 个文本片段。这些类别是基于提交给 Dhārā 应用程序的 4,500 个问题开发的；评估集结合了语料库对齐的应用查询与专家贡献的问题。标注包含 7,578 个问题-片段相关性链接，涉及 6,847 个不同的语料库片段。针对 34 个复合和多跳问题，401 个共轭组指定了共同支持每个问题的多种替代段落组合。片段标识符和层级引用将相关性判断与其源文本相连接。论文描述了语料库准备、标注和验证过程，并报告了使用 Pyserini 实现的固定 BM25 基线（k₁=0.9，b=0.4）。在所有问题上，BM25 取得了 MRR@20 为 0.315、Recall@10 为 0.182 以及 nDCG@10 为 0.205 的成绩。在（原文此处截断）

    arXiv:2611.03196v1 Announce Type: cross  Abstract: Dh\={a}r\={a} Bench is a retrieval benchmark for Indian Knowledge Systems (IKS), with 325 questions in 12 categories and 60,121 text chunks from 15 source groups. The categories were developed based on 4,500 questions submitted to the Dh\={a}r\={a} application; the evaluation set combines corpus-aligned application queries with expert-contributed questions. The annotations comprise 7,578 question-chunk relevance links involving 6,847 distinct corpus chunks. For 34 composite and multi-hop questions, 401 conjugate groups specify alternative sets of passages that jointly support each question. Chunk identifiers and hierarchical references connect the judgments to their source texts. The paper describes corpus preparation, annotation, and validation, and reports a fixed BM25 baseline implemented with Pyserini with $k_1=0.9$ and $b=0.4$. Across all questions, BM25 achieves MRR@20 of 0.315, Recall@10 of 0.182, and nDCG@10 of 0.205. On the gr
    
[^25]: ACTOR：以行动为中心的文学角色扮演智能体时序组织框架

    ACTOR: Action-Centered Temporal Organization for Literary Role-Playing Agents

    [https://arxiv.org/abs/2611.03184](https://arxiv.org/abs/2611.03184)

    提出ACTOR自动化框架，从原始小说文本中提取非对话行为、以“思考-行动-观察”结构表示并按故事时间顺序组织成角色行为轨迹数据集，并通过反事实行为实验揭示现有模型难以可靠推断与角色人设一致的行为。

    

    用于文学角色扮演的大语言智能体通常围绕对话生成进行建模，导致角色的非对话行为及其在故事时间中的动态演变未得到充分体现。本文提出了ACTOR（Action-Centered Temporal Organization for Literary Role-Playing Agents，即面向行动的文学角色扮演智能体时序组织框架），这是一个从原始小说文本中自动构建此类行为轨迹的自动化框架。ACTOR提取对话以及与角色人设相关的非对话行为，使用“思考-行动-观察”（Thought-Action-Observation）结构来表示每个行为，并按照故事内部的时间顺序组织行为轨迹。我们基于143部风格多样的文学作品构建了一个行为数据集。针对每个源行为，我们生成了在上下文中合理、但与角色人设不一致的反事实行为。我们发现，现有模型难以从历史行为中可靠地推断出符合角色一致性的行为及其演变……

    arXiv:2611.03184v1 Announce Type: new  Abstract: Large language agents for literary role-playing are typically modeled around dialogue generation, leaving characters' non-dialogue behaviors and their dynamic evolution over story time underrepresented. In this paper, we propose ACTOR (Action-Centered Temporal Organization for Literary Role-Playing Agents), an automated framework for constructing such behavior trajectories from raw novel texts. ACTOR extracts dialogue and persona-relevant non-dialogue behaviors, represents each behavior using a Thought-Action-Observation structure, and organizes behavior trajectories according to the internal temporal order of the story. We build a behavior dataset from 143 diverse literary works. For each source behavior, we generate counterfactual behaviors that are contextually plausible yet inconsistent with the character's persona. We find that existing models struggle to reliably infer character-consistent actions from historical behaviors and evol
    
[^26]: 每个子句都至关重要：通过连乘小型可回答问题实现遵循指令的重排序

    Every Clause Counts: Instruction-Following Reranking by Multiplying Small, Answerable Questions

    [https://arxiv.org/abs/2611.03179](https://arxiv.org/abs/2611.03179)

    DMReranker将带指令的查询拆分为主题和多个子句，让小型评判模型逐一回答并连乘答案概率，无需任何训练即可将指令敏感度（WISE）从33.5大幅提升至50.9，同时保持原始相关性排序性能。

    

    人们常常在搜索查询中附加指令，例如要求页面为初学者编写、文档为某种语言或仅限官方来源。大多数重排序器（reranker）被训练用于判断主题相关性，并将这类指令视为几个多余的词语。在InfoSearch基准测试上，即使是作为列表式重排序器使用的GPT-4o，其平均WISE（加权指令敏感度评估）也仅达到33.5。我们提出了DMReranker（决策模型重排序器），这是一个零样本重排序器，它将带指令的查询拆分为一个主题和一串带类型的子句。一个小型评判模型针对每个子句回答一个狭窄的问题，然后将答案概率相乘为单一分数。该方法无需任何训练，且配置在测试运行之前就已冻结。在与已发布基线相同的top-100候选结果上进行重排序时，DMReranker在全部六个InfoSearch维度上达到了50.9的宏观WISE，同时将原始模式的nDCG@10保持在82.9。

    arXiv:2611.03179v1 Announce Type: cross  Abstract: People often attach instructions to their search queries, asking for pages written for beginners, documents in a certain language or official sources only. Most rerankers were trained to judge topical relevance and treat such instructions as a few extra words. On the InfoSearch benchmark, even GPT-4o used as a list-wise reranker reaches an average WISE (Weighted Instruction Sensitivity Evaluation) of only 33.5. We present DMReranker (Decision-Model Reranker), a zero-shot reranker that splits an instructed query into a topic and a short list of typed clauses. A small judge model answers one narrow question per clause, and the answer probabilities are multiplied into a single score. Nothing is trained, and the configuration was frozen before the test runs. Reranking the same top-100 candidates as the published baselines, DMReranker reaches a macro WISE of 50.9 over all six InfoSearch dimensions while keeping original-mode nDCG@10 at 82.9
    
[^27]: CoworkOS：面向真实世界自动化的智能体同事规模化扩展

    CoworkOS:Scaling Agent Coworkers for Real-world Automation

    [https://arxiv.org/abs/2611.03178](https://arxiv.org/abs/2611.03178)

    CoworkOS 是一个面向长时程智能体协作工作的通用框架，通过“任务策略与执行管理分离”的三层架构和人类与智能体共享编辑同一工作产物的应用层，实现了真实世界自动化中高质量交付与长时程监控决策的结合。

    

    协作智能体（cowork agents）的快速进展正在推动人机协作迈向真实世界的自动化，交付软件、设计等各类工作产物。然而，这类生产力工作既要求在复杂任务上实现高质量交付，又需要在长时程中持续进行监控与决策，二者的结合仍是尚未解决的开放性挑战。我们提出 CoworkOS，一个面向长时程工作的通用框架，它将任务策略与执行管理相分离：由模型负责决定行动、委派与验收，而运行时则负责维护状态与反馈。其三层架构能够在职责稳定的前提下，支持以同步或异步方式直接执行任务或委派给子智能体。通过共享的应用层，人类与智能体可以查看并修改同一份不断演进的文档、演示文稿、CAD 模型及 Blender 场景，从而使人在回路中的参与既能通过对话进入……

    arXiv:2611.03178v1 Announce Type: cross  Abstract: Rapid advances in cowork agents are shifting human-AI collaboration toward real-world automation, delivering software, designs, and other artifacts. However, such productivity work demands both high-quality delivery on complex productivity tasks and continued monitoring and decision-making over long horizons, a combination that remains an open challenge. We introduce CoworkOS, a general-purpose harness for long-horizon work that separates task strategy from execution management: the model decides actions, delegation, and acceptance, while the runtime maintains state and feedback. Its three-layer architecture enables direct execution or delegation to Subagents, synchronously or asynchronously, under stable responsibilities. Through a shared application layer, humans and agents inspect and revise the same evolving documents, presentations, CAD models, and Blender scenes, so human-in-the-loop participation can enter both through conversat
    
[^28]: ViAdminShift：基于2025年行政改革构建的越南语事件级知识更新基准

    ViAdminShift: A Vietnamese Benchmark for Event-Level Knowledge Updates Built from the 2025 Administrative Reform

    [https://arxiv.org/abs/2611.03171](https://arxiv.org/abs/2611.03171)

    该论文构建了ViAdminShift——首个基于越南2025年省县乡三级行政区划大改革这一真实事件的事件级知识更新越南语基准，包含7,995个测试项目和经严格交叉验证的新旧对照数据，用以系统评估语言模型对现实世界知识变更的更新与应对能力。

    

    2025年7月1日，越南将其63个省份缩减至34个，废除了696个县级行政区，并以3,321个新单位取代了原有的10,035个乡级行政单位（坊）。ViAdminShift是关于这一事件的越南语资源，包含7,995个基准测试项目、2,281条来自改革前后Wikipedia修订版本及带日期模板的检索段落，以及由发布的脚本从官方转换表（不直接再分发）重建的新旧对照表。已变更的项目存储改革前的答案（121个地址项目除外），2×2的实验设计将未受影响的单位作为对照组，另有400个地址以不带变音符号的形式呈现。该对照表与独立来源在10,565个官方配对中的10,564个上保持一致，分歧依据官方决议予以解决，且探测赛道的评分器与两位作者的盲标注一致（Cohen's κ=0.91和0.89）。数据、输出结果、代码和数据说明表均已发布。在探测赛道中，67个模型中至少有20个……（原文截断）

    arXiv:2611.03171v1 Announce Type: cross  Abstract: On July 1, 2025, Vietnam reduced its 63 provinces to 34, abolished its 696 districts, and replaced its 10,035 commune-level units (wards) with 3,321. ViAdminShift, a Vietnamese language resource about this event, consists of 7,995 benchmark items, 2,281 retrieval passages from pre- and post-reform Wikipedia revisions and dated templates, and old-to-new tables that a released script rebuilds from the official conversion table (not redistributed). Changed items store the pre-reform answer (except 121 address items), a $2\times2$ design uses untouched units as controls, and 400 addresses appear without diacritics. The tables agree with an independent lineage on 10,564 of 10,565 official pairs, disagreements were settled against the resolutions, and the probing-track scorer agrees with both authors' blind labels (Cohen's $\kappa=0.91$ and $0.89$). Data, outputs, code, and a datasheet are released. In the probing track, at least 20 of 67 mo
    
[^29]: DepthGraft：通过分层跨层KV重构实现结构正则化

    DepthGraft: Structural Regularization through Hierarchical Cross-Layer KV Reconstruction

    [https://arxiv.org/abs/2611.03142](https://arxiv.org/abs/2611.03142)

    论文提出DepthGraft，一种分层KV共享架构，将KV复用边转化为从深层到源KV投影的直接梯度路径，使KV共享成为隐式结构正则化器，在不引入额外计算开销的前提下缓解Pre-LN架构的深度信息稀释问题。

    

    扩展深度是构建更强大型语言模型的关键途径，但Pre-LN架构常常受到逐层信息稀释问题的困扰，限制了深层结构的有效利用。虽然跨层连接已被证明能有效缓解这一问题，但现有设计通常会引入额外的计算或内存访问开销。我们重新审视跨层KV共享，将其作为这种连接的高效实现形式。除了缓存压缩之外，每条KV复用边都会建立一条从深层到源KV投影的直接梯度路径，从而使KV共享成为对KV表示几何结构的一种隐式结构正则化器。我们对这一机制进行了系统性分析，表明恰当的共享拓扑结构能够锐化键检索方向，同时丰富值内容子空间，而现有的相邻层共享或全局锚点设计并未充分发掘这一效应。受此启发，我们提出了DepthGraft，一种分层式KV共享架构……（摘要在此处截断）

    arXiv:2611.03142v1 Announce Type: new  Abstract: Scaling depth is a key route to stronger large language models, but Pre-LN architectures often suffer from depth-wise information dilution, limiting the effective use of deep layers. While cross-layer connectivity has proven effective at mitigating this problem, existing designs usually introduce additional computation or memory access. We revisit cross-layer KV sharing as an efficient form of such connectivity. Beyond cache compression, each KV reuse edge creates a direct gradient path from deep layers to source KV projections, turning KV sharing into an implicit structural regularizer for KV representation geometry. We provide a systematic analysis of this mechanism, showing that proper sharing topologies sharpen key retrieval directions while enriching the value content subspace, whereas existing adjacent or global-anchor designs do not fully exploit this effect. Motivated by this, we propose DepthGraft, a hierarchical KV-sharing arch
    
[^30]: 因材施教：面向在线策略蒸馏的学生前缀教师自适应

    Meet Students Where They Are: Student-Prefix Teacher Adaptation for On-Policy Distillation

    [https://arxiv.org/abs/2611.03133](https://arxiv.org/abs/2611.03133)

    提出SPTA方法，在蒸馏前用可验证奖励的强化学习训练教师从随机长度的学生生成前缀继续推理并得出正确答案，使教师适应学生所处的状态分布，从而提升在线策略蒸馏中token级监督的可靠性。

    

    在线策略蒸馏通过在学生生成的轨迹上进行优化，缓解了学生模型的曝光偏差。然而，学生生成的前缀可能包含推理错误，或遵循教师所不熟悉的模式，从而削弱了教师提供的token级监督的可靠性。随着前缀变长，这种性能退化变得更加明显。为了解决这种教师侧的分布不匹配问题，我们提出了学生前缀教师自适应（SPTA），使教师模型适应其在蒸馏过程中被查询到的状态。具体而言，在OPD开始之前，我们收集随机长度的学生滚动生成前缀，并使用可验证奖励的强化学习持续训练教师从这些前缀继续推理。通过将每个前缀视为上下文并奖励正确的最终答案，教师学会从学生诱导的状态中恢复，而不是模仿学生的推理方式。随后，经过自适应的教师……

    arXiv:2611.03133v1 Announce Type: new  Abstract: On-policy distillation (OPD) mitigates the student's exposure bias by optimizing on student-generated trajectories. However, student-generated prefixes may contain reasoning errors or follow patterns unfamiliar to the teacher, undermining the reliability of its token-level supervision. This degradation becomes more pronounced as prefixes grow longer. To address this teacher-side distribution mismatch, we propose Student-Prefix Teacher Adaptation (SPTA), which adapts the teacher to the states on which it is queried during distillation. Specifically, before OPD starts, we collect student rollout prefixes with random lengths and continually train the teacher to continue from the prefixes using reinforcement learning with verifiable rewards (RLVR). By treating each prefix as context and rewarding correct final answers, the teacher learns to recover from student-induced states rather than imitate the student's reasoning. Then, the adapted tea
    
[^31]: 微调数据中的预处理选择至关重要：基于大语言模型推荐系统中的效用与人口统计敏感性

    Preprocessing Choices in Fine-Tuning Data Matter: Utility and Demographic Sensitivity in LLM-Based Recommendation

    [https://arxiv.org/abs/2611.03113](https://arxiv.org/abs/2611.03113)

    该论文通过受控实验证明，微调数据中的预处理选择——包括物品标题表示、人口统计平衡和辅助物品知识增强——会显著影响基于大语言模型的生成式推荐系统的推荐效用和人口统计敏感性，且目录约束解码能有效保证输出的有效性。

    

    大语言模型为推荐系统引入了一种生成式方法，能够根据用户历史和物品信息的文本表示生成物品列表。然而，这种生成式设置带来了输出有效性、提示一致性和对人口统计描述词敏感性方面的挑战。在这项工作中，我们研究了微调数据中的预处理选择如何影响基于大语言模型的生成式推荐系统的推荐效用和与公平性相关的行为。我们首先建立了一个结合监督微调、目录约束解码和提示增强的可控推荐流水线。随后，我们研究了三种预处理选择：物品标题表示、人口统计平衡和辅助物品知识增强。在MovieLens-1M数据集上的实验表明，目录约束流水线消除了目录外物品并确保输出完整的十个物品列表，同时提示增强（原文在此处截断）

    arXiv:2611.03113v1 Announce Type: cross  Abstract: Large language models (LLMs) have introduced a generative approach to recommendation, producing item lists from textual representations of user histories and item information. However, this generative setting raises challenges concerning output validity, prompt consistency, and sensitivity to demographic descriptors. In this work, we investigate how preprocessing choices in fine-tuning data affect recommendation utility and fairness-related behavior in a generative LLM-based recommender system. We first establish a controlled recommendation pipeline that combines supervised fine-tuning, catalog-constrained decoding, and prompt augmentation. We then study three preprocessing choices: item-title representation, demographic balancing, and auxiliary item-knowledge enrichment. Experiments on MovieLens-1M show that the catalog-constrained pipeline eliminates out-of-catalog items and ensures complete ten-item outputs, while prompt augmentatio
    
[^32]: DR Pixel：基于视觉语言模型的页面原生深度研究

    DR Pixel: Page-Native Deep Research with Vision-Language Models

    [https://arxiv.org/abs/2611.03097](https://arxiv.org/abs/2611.03097)

    DR Pixel是首个开源端到端多模态深度研究智能体，直接以渲染的文档页面图像作为唯一检索单元和推理单元，在科学论文基准上无需训练即超越基于OCR的基线方法，且工具调用次数更少。

    

    复杂文档（如科学论文）在表格、图表和公式中包含丰富的信息；然而，现有的深度研究（DR）智能体通常将论文简化为OCR提取的文本。我们转而探究视觉语言模型（VLM）能否实现页面原生的深度研究：直接对渲染后的文档页面图像进行检索与推理。我们提出了DR Pixel，这是首个开源的端到端多模态深度研究智能体，它以渲染的页面图像作为唯一的检索单元和推理单元，运行于一个包含106万页的语料库之上。为评估这一设置，我们构建了DRPixelBench，这是一个基于科学论文的基准，需要对文本、图表、表格和公式进行推理。令人惊讶的是，页面原生深度研究无需任何训练即可表现良好，在GPT-5.4、Gemini-3.5-Flash和Qwen3.6-27B等最先进模型上均优于基于OCR的基线方法，同时所需工具调用次数更少。接下来我们转向深度研究的训练。一个关键挑战……

    arXiv:2611.03097v1 Announce Type: cross  Abstract: Complex documents such as scientific papers contain rich information in tables, figures, and equations; however, existing deep research (DR) agents typically reduce papers to OCR-extracted text. We instead ask whether vision-language models (VLMs) can enable page-native deep research: directly retrieving and reasoning over rendered document page images. We introduce DR Pixel, the first open-source end-to-end multimodal DR agent that uses rendered page images as its sole retrieval and reasoning unit, operating over a corpus of 1.06M pages. To evaluate this setting, we construct DRPixelBench, a benchmark on scientific papers requiring reasoning over text, figures, tables, and equations. Surprisingly, page-native DR already works well without training, outperforming OCR-based baselines across state-of-the-art models like GPT-5.4, Gemini-3.5-Flash, and Qwen3.6-27B, while requiring fewer tool calls. We next turn to DR training. A key challe
    
[^33]: 一项关于混合检索、自然语言推理与开源大语言模型在科学声明验证中的实验研究

    An Experimental Study of Hybrid Retrieval, NLI and Open-source LLMs for Scientific Claim Verification

    [https://arxiv.org/abs/2611.03093](https://arxiv.org/abs/2611.03093)

    本研究提出一种保守融合规则，将NLI模型与开源LLM的判断相结合——仅当两者一致时才标注“支持”或“反驳”，否则判定“证据不足”，并配合语义与BM25混合检索，在SciFact和CliniFact构建的2,035个声明基准上验证了该方法在科学声明验证中的有效性。

    

    声明验证是判断源文本支持某个声明、与其矛盾，还是证据不足的任务。自然语言推理（NLI）模型可以判断单个句子是支持还是反驳某个声明，但当证据分散在多个句子中时，它们就会表现不佳。相反，大语言模型（LLM）可以进行跨句推理，但可能得出文本并不支持的结论。在这项实验研究中，我们测试了一种保守的融合规则是否能结合两者的优势，该规则仅当两个模型意见一致时才将声明标注为“支持”或“反驳”，否则默认为“证据不足”。我们基于SciFact和CliniFact构建了一个包含2,035个声明的基准数据集（858个支持，152个反驳，1,025个证据不足）。我们使用语义检索与词法检索（BM25）相结合的混合方法为每个声明检索最相关的前五个句子，该方法略优于单独使用任何一种方法。

    arXiv:2611.03093v1 Announce Type: new  Abstract: Claim verification is the task of deciding whether a source text supports a claim, contradicts it, or lacks sufficient evidence. Natural Language Inference (NLI) models can judge whether a single sentence supports or contradicts a claim, but they struggle when the evidence is spread across several sentences. Conversely, Large Language Models (LLMs) can reason across sentences, but may draw conclusions the text does not support. In this experimental study, we test whether a conservative fusion rule, labeling a claim Support or Contradict only when both models agree and defaulting to Not Enough Evidence otherwise, combines their strengths. We construct a benchmark of 2,035 claims from SciFact and CliniFact (858 Support, 152 Contradict, 1,025 Not Enough Evidence). We retrieve the top five most relevant sentences per claim using a hybrid of semantic and lexical (BM25) retrieval, which slightly outperforms either method alone. These retrieved
    
[^34]: 化身智能：通过持久化个人智能体增强人类能动性

    Avatar Intelligence: Amplifying Human Agency Through Persistent Personal Agents

    [https://arxiv.org/abs/2611.03092](https://arxiv.org/abs/2611.03092)

    本文提出“化身智能”范式，通过跨时间和情境持续代表用户目标、价值观与约束的持久个人智能体来增强人类能动性，并引入智能体监督指数（AOI）框架，以评估智能体交付价值与运营成本及人类监督负担之间的平衡。

    

    个人AI智能体在跨应用处理长周期任务方面的能力日益增强，但用户往往需要投入大量精力来指定目标、消除歧义和验证结果。我们认为，应以其增强人类能动性的有效程度来衡量智能体的进展：使人们能够在更少常规监督的情况下追求自己的目标，同时保留对重大决策的控制权。我们提倡“化身智能”这一范式，即通过跨情境和跨时间的持续决策委托，构建能够代表用户目标、价值观和约束的持久个人智能体。我们描述了从人类主导执行到委托领域内自主运行的成熟度路线图，并引入智能体监督指数（AOI）作为评估框架，用于衡量智能体交付的价值与其运营成本及人类监督负担之间的关系。我们指出了三个核心研究挑战：身份连续性……

    arXiv:2611.03092v1 Announce Type: cross  Abstract: Personal AI agents are increasingly capable of handling long-horizon tasks across applications, yet users often need to invest substantial effort in specifying goals, resolving ambiguity, and verifying outcomes. We argue that progress should be measured by how effectively agents amplify human agency: enabling people to pursue their goals with less routine supervision while retaining control over consequential decisions. We advocate Avatar Intelligence, a paradigm of persistent personal agents that represent users' goals, values, and constraints through sustained decision delegation across contexts and time. We describe a maturity roadmap from human-directed execution to autonomous operation within delegated domains, and introduce the Agentic Oversight Index (AOI) as a framework for evaluating delivered value against agent-side operational cost and human oversight burden. We identify three core research challenges: identity-continuous p
    
[^35]: SAGE（主观性感知生成估计器）：一种用于检测AI生成审稿的语言学框架

    SAGE (Subjectivity-Aware Generation Estimator): A Linguistic Framework for AI-Generated Review Detection *

    [https://arxiv.org/abs/2611.03060](https://arxiv.org/abs/2611.03060)

    本文提出基于主观性语言学特征（评价性语言、情态和言语行为）的轻量级检测器SAGE，与Sem-Detect结合后将AI生成审稿检测在APRDB基准上1%假阳性率下的真阳性率从0.874提升至0.985，且具有可解释性。

    

    我们提出了一种简单且具有语言学动机的方法，利用人类固有的主观性特质来检测AI生成的同行评审。我们在两个层面捕捉主观性：评价性语言与情态的语义层面，以及言语行为的语用层面。利用从言语行为标注和模型预测的主观性分数中提取的特征，我们训练了一个轻量级分类器，仅凭这些语言学特征即可达到0.9566的AUROC。将其输出与最先进的检测器Sem-Detect相结合后，Sem-Detect在APRDB基准上1%假阳性率下的真阳性率（TPR）从0.874提升至0.985，超越了当前的最先进水平。我们的结果表明，主观性的语言学特征为检测AI生成的同行评审提供了一种可解释的信息来源，并且可以轻松地集成到现有的检测方法中。相关代码和模型已在GitHub上公开提供（https://github.com/A

    arXiv:2611.03060v1 Announce Type: new  Abstract: We propose a simple, linguistically motivated approach to detecting AI-generated peer reviews based on the inherently human trait of subjectivity. We capture subjectivity at two levels: the semantic level of evaluative language and modality and the pragmatic level of speech acts. Using features derived from speech act annotations and model-predicted subjectivity scores, we train a lightweight classifier that achieves an AUROC of 0.9566 with only these linguistic features. Combining its output with the state-of-the-art detector Sem-Detect raises the TPR at 1% FPR of Sem-Detect on the APRDB benchmark from 0.874 to 0.985, outperforming the current state of the art. Our results show that linguistic features of subjectivity provide an explainable source of information for detecting AI-generated peer reviews and can be readily integrated into existing detection methods. The code and models are publicly available on GitHub (https://github.com/A
    
[^36]: 解锁刑事法院判决：面向实证法学研究的灵活信息抽取

    Unlocking Criminal Court Decisions: Flexible Information Extraction for Empirical Legal Research

    [https://arxiv.org/abs/2611.03051](https://arxiv.org/abs/2611.03051)

    提出一种基于大语言模型、标注指令可由法律专家迭代完善的灵活信息抽取流水线，将刑事判决书转化为结构化数据以支持实证法学研究，并将醉酒驾驶案件的抽取准确率从52%提升至70%。

    

    刑事判决书包含对犯罪行为的描述，可为定量法律研究提供支持，但将这些描述结构化所需的属性和抽取规则并不总能事先完全确定。我们提出了一种基于大语言模型（LLM）的流水线，通过属性探索、文本片段抽取和数值规范化来构建结构化数据集，并由法律专家可持续完善的标注指令进行引导。我们在两种交通犯罪上进行了评估：醉酒驾驶（DUI）和禁驾后仍驾驶（DDB）。对于醉酒驾驶，修订标注指令带来了更高的抽取准确率（正确抽取值的比例从52%提升至70%）以及更高的人工标注者间一致性。同一模型的多次运行结果之间的一致性高于其与金标准标注的一致性，这表明输出的一致并不一定意味着正确，而标注者之间以及模型之间的分歧则有助于揭示指令中不清晰之处。

    arXiv:2611.03051v1 Announce Type: new  Abstract: Criminal verdicts contain descriptions of criminal behavior that can support quantitative legal research, but structuring them requires attributes and extraction rules that cannot always be fully specified in advance. We present an LLM-based pipeline that builds a structured dataset through attribute exploration, text span extraction, and value normalization, guided by annotation instructions that legal experts can refine. We evaluate it on two traffic offenses: driving under the influence (DUI) and driving despite a ban (DDB). For DUI, revising the instructions was associated with higher extraction accuracy (from 52% to 70% of correctly extracted values) and higher agreement between human annotators. Repeated runs of the same model agreed with each other more than with the gold annotations, showing that consistent output is not necessarily correct, while disagreement between annotators and between models helped reveal unclear instructio
    
[^37]: VERS：通过编辑模型的激活解释来引导推理模型

    VERS: Steering Reasoning Models by Editing Their Activation Explanations

    [https://arxiv.org/abs/2611.03040](https://arxiv.org/abs/2611.03040)

    VERS 利用自然语言自编码器在推理轨迹的“疑点”处编辑激活解释，将编辑转化为一个固定引导方向并注入到推理段落分隔处，从而在几乎不损失准确率的情况下大幅减少推理模型的生成 token 数量，且该方向可迁移到更大的模型上。

    

    推理语言模型在回答之前会先生成一段思维链，其中许多检查步骤并不会改变最终答案。自然语言自编码器（NLA）能用文本解释模型的激活，并能从文本重建激活，而编辑这些解释可以在单个 token 层面对模型进行引导。我们研究这种编辑能否同样引导模型的推理过程。我们构建了 VERS（Verbalize、Edit、Reconstruct、Steer）：它将一条推理轨迹中“疑点”处的解释编辑转化为一个固定的方向，并在段落分隔处添加该方向。在完整的 GSM8K 测试集上，VERS 在准确率仅下降 0.82 个百分点的情况下，将生成的 token 数量减少了 45.8%；在 MATH-500 上，它在准确率下降 3.05 个百分点的情况下节省了 31.3% 的 token。在半强度下，其 token 与准确率之间的权衡介于已发表的方法 SEAL 和 DEER 之间。通过拟合的仿射映射以及针对每个模型调节的强度，该方向可以迁移到更大的 Qwen3 模型上，最大可达 235B。

    arXiv:2611.03040v1 Announce Type: new  Abstract: Reasoning language models write a chain of thought before they answer, and many checks in it do not change the final answer. Natural language autoencoders (NLAs) explain a model's activations in text and rebuild activations from text, and editing an explanation can steer a model at a single token. We study whether such edits can also steer how a model reasons. We build VERS (Verbalize, Edit, Reconstruct, Steer): it turns explanation edits at the doubt points of one reasoning trace into one fixed direction, and adds this direction at paragraph breaks. On the full GSM8K test, VERS cuts generated tokens by 45.8% at an accuracy change of -0.82 percentage points (pp); on MATH-500 it saves 31.3% at -3.05 pp. At half strength, its trade-off between tokens and accuracy lies between those of the published methods SEAL and DEER. With fitted affine maps and a strength tuned for each model, the direction transfers to larger Qwen3 models, up to 235B-
    
[^38]: 仅仅发现错误还不够：面向实体目标情感分析的选择性模型升级

    Finding Errors Is Not Enough: Selective Model Escalation for Entity-Targeted Sentiment Analysis

    [https://arxiv.org/abs/2611.02989](https://arxiv.org/abs/2611.02989)

    该论文研究了实体级情感分析中基于不确定性的选择性模型升级策略，发现优先采集少数类样本仅在极小训练规模下有效，而将低置信度预测交由替代编码器或本地大语言模型处理可纠正错误并提升性能。

    

    在资源有限的情况下改进情感分类器，不仅仅是决定收集多少数据或训练哪个模型这么简单。已标注的数据集可能缺少某个重要类别的样本；而训练完成后，即使另一个模型本身较弱，分类器也可能从中受益。我们在文档级实体级情感分析中研究这些决策，主要使用南斯拉夫语系新闻数据。我们首先比较了数据集增长过程中的采集顺序；然后利用不确定性来选择需要重新审视哪些预测、处理多少个，以及替代编码器或本地大语言模型能否纠正它们；最后，我们在开发数据上选择完整的路由策略，并在留出数据上进行评估。结果表明，优先采集少数类样本仅在极小的训练规模下有帮助，但此后效果并不持续。在SentiCoref数据集上，一个能够纠正最低置信度10%预测中所有错误且不损害正确预测的预言机，可带来9.1（摘要在此处截断）的提升。

    arXiv:2611.02989v1 Announce Type: new  Abstract: Improving a sentiment classifier with limited resources involves more than simply deciding how much data to collect or which model to train. A labelled collection may lack examples of an important class. Once trained, a classifier may benefit from another model even when that model is weaker on its own. We study these decisions in document-level Entity-Level Sentiment Analysis (ELSA), primarily using South Slavic news. We first compare acquisition orders as collections grow. We then use uncertainty to choose which predictions to revisit, how many to process, and whether an alternative encoder or local LLM can correct them. Finally, we select complete routing policies on development data for held-out evaluation. Minority-prioritised acquisition helps at some very small training sizes, but not consistently thereafter. On SentiCoref, an oracle correcting all errors in the least-confident 10% without harming correct predictions would add 9.1
    
[^39]: 将大语言模型标注中的文学情感蒸馏到序数BERT模型中

    Distilling Literary Sentiment from LLM Annotations into an Ordinal BERT Model

    [https://arxiv.org/abs/2611.02985](https://arxiv.org/abs/2611.02985)

    本研究利用Gemma 4 31B生成的标注通过序数回归蒸馏训练小型BERT模型进行文学情感分析，其中BERT-large的性能接近教师模型，且BERT-base在非中性句子上的表现甚至超越了教师模型。

    

    人工标注文学语料的稀缺性限制了领域特定情感分析模型的发展。本研究探讨由大语言模型生成的标注是否能够监督较小的模型进行文学情感分析。我们在10,889条由Gemma 4 31B在十分制情感量表上标注的文学句子上，使用序数回归方法微调了五个BERT模型。这些模型在海明威《老人与海》的1,923条人工标注句子上进行评估，并与教师模型及传统基线方法进行比较。BERT-large取得了0.589的Kendall's τ，接近教师模型的0.604。尽管恒定中性预测器实现了最低的总体平均绝对误差（MAE），但BERT-base在非中性句子上的表现优于教师模型，MAE为0.726，而教师模型为1.116。这些发现证明了大语言模型生成监督在开发更小型文学情感模型方面的潜力。

    arXiv:2611.02985v1 Announce Type: new  Abstract: The scarcity of human-annotated literary corpora limits the development of domain-specific sentiment analysis models. This study investigates whether LLM-generated annotations can supervise smaller models for literary sentiment analysis. We fine-tune five BERT models using ordinal regression on 10,889 literary sentences annotated by Gemma 4 31B on a ten-point sentiment scale. The models are evaluated against 1,923 human-annotated sentences from Hemingway's "The Old Man and the Sea", alongside the teacher model and conventional baselines.   BERT-large achieves a Kendall's $\tau$ of 0.589, approaching the teacher's 0.604. Although a constant-neutral predictor achieves the lowest overall mean absolute error (MAE), BERT-base outperforms the teacher on non-neutral sentences, achieving an MAE of 0.726 versus 1.116. These findings demonstrate the potential of LLM-generated supervision for developing smaller literary sentiment models, while high
    
[^40]: 规范感知的预测不确定性分解

    Specification-Aware Predictive Uncertainty Decomposition

    [https://arxiv.org/abs/2611.02956](https://arxiv.org/abs/2611.02956)

    本文提出一种规范感知的预测不确定性分解方法，将预测风险分解为规范不确定性、残余偶然不确定性和认知不确定性三项，从而量化了在大语言模型中通过澄清欠规范输入（补全用户意图）能够减少多少预测不确定性。

    

    经典的预测不确定性分解方法（例如信息论分解）是针对输入对象的固定表示 $x$ 定义的。这些方法既没有考虑 $x$ 可能是不完整的，也没有衡量缺失的信息会对预测产生多大影响。在大语言模型中，用户无需任何重新训练即可澄清欠规范的输入。澄清后的输入能够更准确地描述用户未被观测到的意图 $z^*$，并改变对结果的预测不确定性。本文从概率角度对这一设定进行了形式化，并推导出将预测风险分解为三项的方案：规范不确定性、残余偶然不确定性和认知不确定性。我们证明，在原始信息水平上（即仅有 $x$ 可用时），以贝叶斯风险表示的偶然不确定性包含两个组成部分。第一个组成部分——我们称之为规范不确定性——是该贝叶斯风险的期望缩减量。

    arXiv:2611.02956v1 Announce Type: cross  Abstract: Classical decompositions of predictive uncertainty, such as the information-theoretic one, are defined for a fixed representation $x$ of the input object. They do not consider that $x$ may be incomplete, nor do they measure how much the missing information can affect the prediction. In Large Language Models, users can clarify an underspecified input without any retraining. The clarified input can describe the user's unobserved intent $z^*$ more accurately and change the predictive uncertainty about the outcome. In this paper, we formalize this setting probabilistically and derive a three-term decomposition of prediction risk into specification, residual aleatoric, and epistemic uncertainties. We show that at the original information level, when only $x$ is available, the aleatoric uncertainty, represented by a Bayes risk, has two components. The first, which we call specification uncertainty, is the expected reduction in this Bayes ris
    
[^41]: 面向语言模型智能体的基于猜想与反驳的动作验证

    Action Verification through Conjecture and Refutation for Language Model Agents

    [https://arxiv.org/abs/2611.02950](https://arxiv.org/abs/2611.02950)

    该论文提出BlackSwan，一种受“猜想与反驳”思想启发的动作验证方法，通过预测动作的预期效果并主动构造、检验可能推翻动作正确性的具体假设，从而捕捉传统确认式验证无法发现的不当智能体行为。

    

    语言模型智能体越来越多地通过工具操控外部环境来代表用户执行任务，而这些任务通常源自高级指令，其中关键的任务特定和环境特定上下文往往未被明确说明。为了防止由此产生的不当行为，近期的工作会在执行前对拟议动作进行验证，主要是通过确认支持该动作的条件是否成立。我们认为这种以确认为导向的观点存在一个根本性盲区：一个错误的动作往往能够通过所有询问“它是否正确”的检查。这促使我们对验证提出一个不同的问题：与其询问什么证据支持一个动作，智能体能否找到反驳它的证据？我们沿这一方向提出了BlackSwan，这是一种以反驳为导向的“预期-怀疑-检验”流程，它预测每个改变状态的动作的预期效果，为该动作可能出错的方式制定具体的、可实证检验的假设，并进行检验……

    arXiv:2611.02950v1 Announce Type: cross  Abstract: Language model agents increasingly act on users' behalf by manipulating external environments through tools, often from high-level instructions that leave critical task- and environment-specific context unstated. To prevent resulting misaligned actions, recent work verifies proposed actions before execution, largely by establishing whether the conditions supporting them hold. We argue that this confirmation-oriented view leaves a fundamental blind spot: a wrong action often passes every check that asks whether it is right. This motivates a different question for verification: rather than asking what evidence supports an action, can the agent find evidence that refutes it? We pursue this direction with BlackSwan, a refutation-oriented, expect-suspect-test procedure that predicts the intended effect of each state-mutating action, formulates specific and empirically testable hypotheses for how it could be wrong, and tests them using exist
    
[^42]: PlagiaCorpus：用于取证式抄袭检测的双语、对抗性、跨语言数据集

    PlagiaCorpus: A Bilingual, Adversarial, and Cross-Lingual Dataset for Forensic Plagiarism Detection

    [https://arxiv.org/abs/2611.02940](https://arxiv.org/abs/2611.02940)

    本文发布了 PlagiaCorpus——一个包含近 20 万条记录的阿拉伯语-英语双语数据集，通过逐字复制、字符级混淆、LLM 改写及跨语言翻译改写等五种难度递增的对抗性复用场景，为取证式抄袭与文本复用检测提供了系统性评测基准。

    

    PlagiaCorpus 是一个阿拉伯语-英语双语数据集，用于对抄袭与文本复用检测进行基准测试，涵盖由大语言模型（LLM）产生的复用以及跨语言的复用。该数据集包含 198,643 条文本记录：一个由 34,885 个段落组成的源语料库（取自英文和阿拉伯文维基百科以及两个开放许可的图书集），以及 163,758 个查询（其中 147,381 个用于训练，16,377 个用于测试）。查询涵盖五种难度递增的复用类别，即逐字复制、字符级混淆、由两个不同 LLM 进行的同语言改写、以及英阿翻译加结构性改写，此外还包括一个由非复用文本组成的 Safe（安全）类别。每个正例查询都与其源段落相关联，并且训练/测试集的划分按源段落分组，从而确保没有任何段落同时向两个划分贡献查询。该数据集的发布经过了语言识别、去除 LLM 生成痕迹以及……（摘要截断）

    arXiv:2611.02940v1 Announce Type: new  Abstract: PlagiaCorpus is a bilingual Arabic-English dataset for benchmarking plagiarism and text-reuse detection, including reuse produced by large language models (LLMs) and reuse across languages. It contains 198,643 text records: a source corpus of 34,885 passages drawn from English and Arabic Wikipedia and two openly licensed book collections, and 163,758 queries (147,381 for training and 16,377 for testing). The queries cover five reuse classes of increasing difficulty, namely verbatim copying, character-level obfuscation, same-language paraphrasing by two different LLMs, and English--Arabic translation with structural paraphrasing, together with a Safe class of non-reused passages. Every positive query is linked to its source passage, and the train/test split is grouped by source passage so that no passage contributes queries to both splits. The release was curated through language identification, removal of LLM generation artefacts, and ex
    
[^43]: 状态变化是局部的，其后果却不是：面向长期智能体记忆的依赖感知状态修订

    A State Change Is Local, Its Consequences Are Not: Dependency-Aware State Revision for Long-Term Agent Memory

    [https://arxiv.org/abs/2611.02912](https://arxiv.org/abs/2611.02912)

    提出依赖感知状态修订问题，并通过 StateGraph 从交互历史中恢复状态依赖图，利用反事实验证区分真实的支撑依赖与语义相关性，从而在上游事实变化时自动失效失去支撑的下游信念、计划和行动。

    

    长期运行的智能体依赖于关于用户、任务和决策的持久记忆，而这些记忆必须与不断变化的世界保持一致。然而，当上游事实发生变化时，记忆系统往往只会修订被矛盾的记忆本身，而由其推导出的信念、计划和行动仍然被保留，即使它们已经失去了支撑。因此，我们提出了依赖感知状态修订这一问题，它不仅关注什么发生了变化，还关注哪些依赖状态不应再被信任。为解决这一问题，StateGraph 维护一个不断演进的状态依赖图，其边从交互历史中恢复。其核心设计是反事实验证：每条候选边通过询问“如果失去其前提，依赖状态是否仍然成立”来进行检验，从而将支撑性依赖与语义相关性区分开来；修订过程仅在严格依赖失效且不存在独立支持的情况下才使依赖状态失效。

    arXiv:2611.02912v1 Announce Type: new  Abstract: Long-lived agents rely on persistent memory of users, tasks and decisions, and this memory must stay consistent with a world that keeps changing. When an upstream fact changes, however, memory systems can revise the contradicted memory itself while the beliefs, plans and actions derived from it remain accepted, even though they have lost their support. We therefore formulate dependency-aware state revision, which asks not only what changed but also which dependent states should no longer be trusted. To address it, StateGraph maintains an evolving state dependency graph whose edges are recovered from interaction histories. Its core design is counterfactual verification: each candidate edge is tested by asking whether the dependent state would survive the loss of its premise, which separates support dependence from semantic relatedness, and revision then invalidates a dependent state only when a strict dependency fails and no independent s
    
[^44]: 用户属性与政治意识形态塑造大语言模型在在线政治辩论中的接受度

    User Attributes and Political Ideology Shape LLM Receptiveness in Online Political Debates

    [https://arxiv.org/abs/2611.02908](https://arxiv.org/abs/2611.02908)

    该研究通过近15万次受控多轮对话实验发现，大语言模型在在线政治辩论中对左倾用户表现出系统性高于右倾用户的接受度，且这种政治不对称性即便在不同用户为同一立场辩护时依然存在。

    

    在线政治讨论日益涉及AI中介的参与，用户会调用大语言模型来评估有争议的主张、仲裁争端并回应对立观点。随着这些系统成为在线交流的参与者，它们可能会改变哪些发言者和论辩策略在讨论中获得认可。我们通过将Debate.org上的5,376场辩论重构为大语言模型对话，并运行近15万次受控多轮对话来研究这一互动维度，这些对话涉及不同的用户政治倾向、专业水平、修辞策略和所辩护的立场，其中包括对立用户之间的对话。在自然情境和受控辩论中，我们发现大语言模型对左倾用户比右倾用户表现出系统性的更高接受度。当不同政治倾向的用户为同一立场辩护，以及当对立用户出现在同一场对话中时，这种不对称性依然存在，这表……

    arXiv:2611.02908v1 Announce Type: cross  Abstract: Online political discussions increasingly involve AI-mediated participation, as users invoke LLMs to assess contested claims, arbitrate disputes, and respond to opposing viewpoints. As these systems become participants in online exchanges, they may alter which speakers and argumentative strategies receive validation within the discussion.   We study this interactional dimension by reconstructing 5,376 Debate.org debates as LLM conversations and running nearly 150,000 controlled multi-turn conversations that vary user political orientation, expertise, rhetorical strategy, and defended position, including conversations with competing users.   Across naturalistic and controlled debates, we find that LLMs are systematically more receptive to left-leaning than right-leaning users. The asymmetry persists when users with different political orientations defend the same position and when opposing users appear in the same conversation, indicati
    
[^45]: Maswao：一个自动收集、预处理和基准化的南非手语数据集

    Maswao: An automatically collected, prepared, and benchmarked dataset for the South African Sign language

    [https://arxiv.org/abs/2611.02903](https://arxiv.org/abs/2611.02903)

    本文提出了一个以最少人工干预自动收集、预处理和基准化南非手语（SASL）数据集的框架Maswao，利用新闻广播等公共资源和语音识别技术，旨在弥合全球手语处理技术发展的差距。

    

    手语处理领域的最新进展可归因于大规模数据集收集和算法复杂性两方面的发展，例如姿态估计技术、基于Transformer模型的信号处理以及模型优化。然而，在全球200多种公认的手语中，只有少数手语（如美国手语、英国手语和德国手语）从这些技术进步中受益，而其余手语则远远落后。在本研究中，我们旨在通过提出一个自动收集和预处理手语数据集的框架来解决这种技术差距，该框架仅需极少的人工干预，并以南非手语（SASL）作为测试平台。具体而言，我们利用新闻广播等公共资源自动收集SASL视频内容。此外，我们使用现有的自动语音识别（ASR）技术将伴随的音频流转换为文本。

    arXiv:2611.02903v1 Announce Type: new  Abstract: Recent advancements in sign language processing can be attributed to developments in both large-scale dataset collection and algorithmic complexity in areas such as pose estimation techniques, signal processing through Transformer models, and model optimization. However, only a handful of sign languages, such as the American, British, and German, from over 200 globally recognized sign languages, reap the benefits of such advancements, while the rest lag behind. In this study, we aim to address this technological disparity by proposing a framework for the automatic collection and preparation of sign language datasets, with very minimal human-labour intervention, using the South African Sign Language (SASL) as a testbed. Specifically, we automatically collect SASL video content using public resources such as news broadcasts. Additionally, we convert their accompanying audio streams into text using available Automatic Speech Recognition (AS
    
[^46]: 瓦特拔河：大型语言模型在医疗问答任务中的能耗与准确率权衡研究

    Tug of Watts: A study on the Energy - Accuracy Trade-off of Large Language Models on Medical Question Answering Tasks

    [https://arxiv.org/abs/2611.02889](https://arxiv.org/abs/2611.02889)

    本研究通过对13个不同规模的大型语言模型在医疗问答任务上的能耗与准确率分析，提出了系统构建和解读能耗-准确率联合权衡指标的方法，强调仅分别报告这两项指标不足以支撑严谨的权衡评估。

    

    大型语言模型（LLMs）在医疗问答（QA）系统中的应用日益广泛。尽管近期的研究工作旨在逐步提升下游任务的准确率，但对模型能耗及相应碳足迹的实证估算往往被忽视。在本研究中，我们通过考察来自3个模型家族、参数规模从2.7亿（270M）到140亿（14B）的13个大型语言模型，展示了从业者如何能够为OpenMedical-LLM排行榜上可用的医疗问答任务估算能耗并获得可靠的碳排放评估。我们讨论了如何批判性地解读一组模型在给定任务上的能耗与准确率之间的权衡。通过这项工作，我们强调仅报告相互独立的能耗和准确率指标不足以进行细致的权衡评估，还需要系统地构建和解读联合权衡指标。我们预想此类研究将帮助从业者……

    arXiv:2611.02889v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used in medical question answering (QA) systems. While recent works aim at incrementally improving the downstream task accuracy, an empirical estimation of the models' energy consumption and the associated carbon footprints is often overlooked. In this study, we demonstrate how practitioners can estimate energy consumption and arrive at reliable emission estimates for medical QA tasks available in the OpenMedical-LLM leaderboard by investigating 13 LLMs from 3 model families ranging from 270M to 14B parameters. We discuss how one can critically interpret energy versus accuracy trade-offs for a set of models for a given task. With this work, we underline that reporting disjoint energy and accuracy metrics alone does not suffice for careful trade-off assessments, but it also demands systematic construction and interpretation of joint trade-off metrics. We envisage such studies aiding practition
    
[^47]: Omni-Life：面向人们日常生活的全模态实时AI助手基准测试

    Omni-Life: Benchmarking Omnimodal Real-Time AI Assistants in People's Daily Lives

    [https://arxiv.org/abs/2611.02835](https://arxiv.org/abs/2611.02835)

    提出基于真实用户需求的Omni-Life基准来评估实时全模态AI助手在日常生活中的表现，发现现有主流模型最多只能完全正确处理4.48%需要帮助的情境。

    

    实时全模态助手被期望能够支持人们的日常生活，但仅有音视频理解能力并不能确定它们是否能提供及时且恰当的帮助。我们提出了Omni-Life，这是一个基于真实世界用户需求、部署场景和设备相关观测构建的基准。它将对危险情况和边缘案例的受控视频合成与经过标注的真实世界视频相结合，并遵循统一的时间协议。该基准包含400个视频：其中290个需要助手提供帮助，包含555个需要干预的事件；110个全程需要保持沉默，用以测试模型在无需帮助时能否克制不必要的回应。我们评估了十二种主流实时全模态模型配置，联合考察事件覆盖率、回应内容、回应时机以及多余输出。结果显示，在需要帮助的视频中，最多仅有4.48%被完全正确地处理。为了检验实际（应用表现），（摘要在此处截断）

    arXiv:2611.02835v1 Announce Type: new  Abstract: Real-time omnimodal assistants are expected to support people in everyday life, but audiovisual understanding alone does not establish whether they can provide timely and appropriate help. We introduce Omni-Life, a benchmark grounded in real-world user needs, deployment scenarios, and device-dependent observations. It combines controlled video synthesis for hazardous situations and edge cases with annotated real-world videos under a shared temporal protocol. The benchmark contains 400 videos: 290 requiring assistance, with 555 required intervention events, and 110 requiring silence throughout to test whether models can withhold unnecessary responses when no assistance is needed. We evaluate twelve configurations featuring mainstream real-time omnimodal models, jointly assessing event coverage, response content, timing, and unwanted output. At most 4.48% of videos requiring assistance are handled completely correctly. To examine practical
    
[^48]: 平台投递约束下健康消息传递的窗口感知调度：来自尼日利亚AI驱动的WhatsApp孕产妇与婴幼儿健康聊天机器人的证据

    Window-Aware Scheduling for Health Messaging under Platform Delivery Constraints: Evidence from an AI-Powered WhatsApp Maternal and Infant Health Chatbot in Nigeria

    [https://arxiv.org/abs/2611.02834](https://arxiv.org/abs/2611.02834)

    该研究首次量化了WhatsApp 24小时客户服务窗口对健康聊天机器人消息送达造成的损失，并将窗口感知调度形式化为决策问题，发现将关键医疗咨询服务以预批准模板发送可将送达率从几乎为零提升至约81.5%–86.8%。

    

    WhatsApp仅允许企业在用户最后一条消息后的24小时内发送自由格式的消息；在此客户服务窗口之外，只能发送预先批准的模板消息。健康聊天机器人继承了这一规则，但其造成的成本却很少被测量。我们分析了尼日利亚一个WhatsApp孕产妇健康聊天机器人发送给3,696位母亲的98,607条消息，并使用一个基本独立的数据导出作为复现验证。在窗口开启时，没有任何自由格式消息发送失败（35,500条中0条失败），而在恰好24小时处，失败率跃升了93.0个百分点（95%置信区间：87.0至99.1）。两个消息流程产生了66.1%的失败，其中包括一项“与医疗专业人员交谈”的服务提议——按原设计构建，在窗口已关闭的母亲中，每100位仅有0.3位能收到该提议。我们将窗口感知调度形式化为一个决策问题，其中包含一个已知价值为零的动作。基于模型的离线评估（对未确认送达的情况设定界限）估计，若将该提议作为已批准的模板发送，其送达率将达到81.5至86.8（原文摘要在此处截断）。

    arXiv:2611.02834v1 Announce Type: new  Abstract: WhatsApp lets a business send free-form messages only within 24 hours of a user's last message; outside this customer service window, only pre-approved templates are delivered. Health chatbots inherit this rule, but its cost is rarely measured. We analyse 98,607 messages sent to 3,696 mothers by a WhatsApp maternal health chatbot in Nigeria, with a largely separate export as replication. No free-form message failed with the window open (0 of 35,500), and failure jumped by 93.0 percentage points (95% CI 87.0 to 99.1) at exactly 24 hours. Two flows produced 66.1% of failures, including an offer to speak with a medical professional that, as built, reached 0.3 in 100 mothers whose window had closed. We formalise window-aware scheduling as a decision problem with an action of known zero value. Model-based offline evaluation, with bounds for unconfirmed delivery, estimates that sending the offer as an approved template would reach 81.5 to 86.8
    
[^49]: TRIO：面向检索增强生成的基于个体优化的任务细化

    TRIO: Task Refinement with Individual Optimization for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.02810](https://arxiv.org/abs/2611.02810)

    该论文提出TRIO方法，通过对齐三个智能体的个体优化目标并采用迭代细化范式，解决了检索增强生成中检索不充分和幻觉的问题。

    

    本文研究了检索增强生成，其利用外部知识来提升语言模型的性能。尽管已取得显著进展，但由于检索不充分和潜在的幻觉问题，当前的RAG方法仍远不尽如人意。为此，我们提出了一种名为“基于个体优化的任务细化”（TRIO）的新型RAG方法。TRIO的核心思想是通过协调三个精心设计的智能体的各自优化目标，为它们构建迭代细化范式。具体而言，我们首先利用查询扩展智能体对模糊查询进行扩展，并根据答案质量的提升来量化奖励，该奖励可用于优化智能体以获得更高的扩展效率。为提升检索质量，我们引入一致性投票机制，将扩展后的查询标注为“不充分”或“充分”，并将相应的生成结果用作真值（原文摘要在此处截断）。

    arXiv:2611.02810v1 Announce Type: new  Abstract: This paper studies retrieval-augmented generation (RAG), which leverages external knowledge to improve the performance of language models. Despite the remarkable progress, current RAG approaches are still far from satisfactory due to inadequate retrieval and potential hallucinations. Towards this end, we propose a novel approach named Task Refinement with Individual Optimization (TRIO) for RAG. The core idea of our TRIO is to equip three well-designed agents with an iterative refinement paradigm by aligning their individual optimization objectives. Specifically, we first expand ambiguous queries with a query extension agent and quantify the reward based on the improved answer quality, which can be utilized to refine our agent for higher expansion efficiency. To enhance the retrieval quality, we include a unanimity voting to annotate the expanded query as insufficient or sufficient, and their generative outcomes are utilized as ground tru
    
[^50]: 基于大语言模型创建混合式交互叙事系统：从形式化描述到评估

    Creating Hybrid Interactive Narrative Systems with LLM: From Formal Description to Evaluation

    [https://arxiv.org/abs/2611.02807](https://arxiv.org/abs/2611.02807)

    提出了一个基于大语言模型的混合式交互叙事平台，通过让可互换的体验管理者模块运行于共享的符号世界模型之上，实现了对不同管理者的统一分析与评估。

    

    交互式叙事让用户能够塑造故事，同时由体验管理者（Experience Manager，EM）保持故事的连贯性，在玩家自由与作者控制之间取得平衡。大语言模型如今催生了新一代的混合式体验管理者，它们将基于自由文本输入的运行时生成与符号化叙事控制相结合。然而，这些系统的异质性使其难以相互比较：管理者的行为通常与其运行的独特且特定的场景紧密绑定，其大部分决策过程发生在任何显式且可验证的结构之外。我们通过一个基于大语言模型的叙事平台来解决这一问题，该平台的管理者是可互换的模块，运行在一个由事实、操作和目标构成的共享符号世界模型之上。由于每个被应用的操作都带有明确的前提条件和效果，每个会话都可以通过统一的状态转移表示被分析为一个规划并进行评估，而不依赖于管理者内部的工作方式。

    arXiv:2611.02807v1 Announce Type: cross  Abstract: Interactive narratives let users shape a story while an Experience Manager (EM) keeps it coherent, balancing player freedom against authorial control. Large Language Models now enable a new generation of hybrid EMs, which combine run-time generation from free-text input with symbolic narrative control. However, their heterogeneity makes them hard to compare: A manager's behaviour is generally tied to the unique and specific scenario in which it operates, and much of its decision-making takes place outside of any explicit and verifiable structure. We address this with an LLM-based narrative platform whose managers are interchangeable modules operating over a shared symbolic world model of facts, operations, and goals. Because every applied operation carries explicit preconditions and effects, each session can be analysed as a plan through a common state-transition representation and evaluated, independently of how the manager works inte
    
[^51]: SelfImprove-Bench：评估自我改进智能体的泛化能力

    SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents

    [https://arxiv.org/abs/2611.02797](https://arxiv.org/abs/2611.02797)

    该论文提出 SelfImprove-Bench，一个针对递归自我改进智能体的评估基准，将任务组织为探索、规划、推理三大核心能力场景，并基于游戏实现可控、可复现且可扩展的评估，以解决现有评估无法区分真实能力增益与场景过拟合的问题。

    

    递归自我改进系统日益被视为加速迈向通用人工智能（AGI）的一条有前景的途径。这类系统旨在通过完善自身工作流程、并递归地改进引导后续完善的策略来自主提升能力，使能力增益能够泛化到未见过的场景中。然而，现有评估缺乏能够区分真实能力增益与场景过拟合的可靠指标，缺乏针对未见任务与难度的可持续、可扩展的评估协议，也缺乏将改进细粒度归因于演化轨迹的手段。为此，我们提出了 SelfImprove-Bench——一个 RSI 评估框架，它将任务组织为三种核心能力（即探索、规划、推理），作为引导模型演化的场景。为了可控性和可复现性，我们在游戏上实例化这些任务，并配套可扩展的任务生成与编排机制（摘要在此处截断）。

    arXiv:2611.02797v1 Announce Type: cross  Abstract: Recursively self-improving (RSI) systems are increasingly viewed as a promising means of accelerating progress toward artificial general intelligence. They aim to autonomously improve their capabilities by refining their working pipelines and recursively improving the strategies that guide subsequent refinements, with gains generalizing to unseen scenarios.However, existing evaluations lack reliable metrics to distinguish capability gains from scenario fitting, sustainable and extensible protocols for unseen tasks and difficulties, and fine-grained attribution of improvement to evolution traces. To this end, we propose SelfImprove-Bench,an RSI evaluation framework that organizes tasks into three primary capabilities (i.e., Exploring, Planning, Reasoning) as scenarios for guiding model evolution. We instantiated the tasks on games for controllability and reproducibility, and pairs them with scalable generation and orchestration driven b
    
[^52]: PGuard：一种面向多模态大语言模型的原生、流式、即插即用安全防护栏

    PGuard: A Native, Streaming, and Plug-in Guardrail for Multimodal Large Language Models

    [https://arxiv.org/abs/2611.02791](https://arxiv.org/abs/2611.02791)

    PGuard 是一种面向多模态大语言模型的原生、流式、即插即用安全防护栏，它通过动态层门控直接利用骨干模型的中间表示实现 token 级安全检测，无需改变模型的原始推理路径即可将安全监控内化为模型自身能力。

    

    随着流式生成成为纯文本和多模态大语言模型的主导服务范式，在整个生成过程中确保安全性已成为一项关键挑战。我们提出了 PGuard，这是一种原生的、流式的、即插即用的安全防护栏，它将安全监控从外部服务依赖转变为骨干模型的内生集成能力。PGuard 直接利用骨干模型的中间表示进行 token 级别的安全检测，而不会改变其原始推理路径或生成行为。具体而言，动态层门控机制选择性地聚合跨层的残差流表示，从而在预填充和自回归解码阶段提取与安全相关的信号。为了平衡检测精度与计算效率，PGuard 支持多种表示建模架构，包括轻量级的无状态变体以及基于序列感知的变体……

    arXiv:2611.02791v2 Announce Type: new  Abstract: As streaming generation becomes the dominant serving paradigm for both text-only and multimodal large language models, ensuring safety throughout the generation process has become a critical challenge. We introduce PGuard, a native, streaming, and plug-in guardrail that transforms safety monitoring from an external serving dependency into an integrated capability of the backbone model. PGuard directly leverages the backbone's intermediate representations for token-level safety detection, without altering its original inference pathway or generation behavior. Specifically, dynamic layer gating selectively aggregates residual-stream representations across layers to extract safety-relevant signals during both prefill and autoregressive decoding. To balance detection accuracy and computational efficiency, PGuard supports multiple representation-modeling architectures, including a lightweight stateless variant and sequence-aware variants buil
    
[^53]: 停止手工打造你的引擎！GenDB-V：迈向AI生成且经形式化验证的数据库系统

    Stop Handcrafting Your Engine! GenDB-V: Towards AI-Generated and Formally Verified Database Systems

    [https://arxiv.org/abs/2611.02779](https://arxiv.org/abs/2611.02779)

    GenDB-V提出利用AI智能体自动生成针对特定硬件和工作负载优化、并经形式化验证保证正确性的数据库引擎，以解决数据库系统手工开发成本高昂和正确性难以保障的两大长期难题。

    

    五十多年来，数据库社区开发了多种成功的引擎来回答两个根本问题：如何让数据库系统运行得更快，以及如何确保其正确性？这些引擎的经验表明，性能取决于硬件和工作负载。随着两者随时间不断变化，我们不得不反复重新设计和重建数据库系统，这是一个耗时且代价高昂的过程。正确性则是另一个长期存在的挑战：尽管经历了多年的开发和测试，最近的研究仍在成熟的数据库引擎中不断发现错误结果类缺陷。为解决这些问题，我们提出了GenDB-V，一个使AI智能体能够自动合成数据库引擎的系统，这些引擎针对特定硬件和工作负载进行专门化以获得性能，并经过形式化验证以保证正确性。GenDB-V支持SQL模板的专门化实现，以及接受所支持SQL子集中任意查询的通用引擎。首先，我们开发

    arXiv:2611.02779v1 Announce Type: cross  Abstract: For over 50 years, the database community has developed a wide range of successful engines to answer two fundamental questions: how can database systems run faster, and how can we ensure their correctness? Experience with these engines shows that performance depends on hardware and workloads. As both change over time, we repeatedly redesign and rebuild database systems, a time-consuming and expensive process. Correctness remains another long-standing challenge: recent studies continue to uncover wrong-answer bugs even in mature database engines, despite years of development and testing. To address these, we present GenDB-V, a system that enables AI agents to synthesize database engines specialized to hardware and workloads for performance, and formally verified for correctness. GenDB-V supports specialized implementations of SQL templates and general-purpose engines that accept any query within a supported SQL subset. First, we develop
    
[^54]: EffComp：通过混合强化学习与监督学习实现高效提示压缩

    EffComp: Efficient Prompt Compression via Hybrid Reinforcement & Supervised Learning

    [https://arxiv.org/abs/2611.02766](https://arxiv.org/abs/2611.02766)

    提出了EffComp框架，通过结合BERT风格的重排序与句子选择模型以及混合强化学习和监督学习的训练方式，为基于RAG的问答任务实现无幻觉、低延迟且在压缩比与准确率之间达到最佳平衡的句子级抽取式提示压缩。

    

    提示压缩旨在缩短输入提示的长度，从而实现更廉价、更快速且可能更高质量的大语言模型（LLM）预测。然而，现有的提示压缩方法往往受限于压缩增益有限、存在幻觉风险和/或压缩延迟高等问题。我们提出了EffComp，一个面向基于RAG的问答（QA）任务的高效提示压缩框架。EffComp采用BERT风格的文档重排序模型和句子选择模型，实现句子级别的快速抽取式提示压缩。其抽取式的特性可以防止压缩后的提示中出现幻觉。此外，该框架采用强化学习与监督学习相结合的混合训练方法，旨在优化压缩比的同时保持LLM的准确性。在多个问答数据集上的实验表明，EffComp优于当前最先进的提示压缩方法，在压缩比与（准确率）之间实现了最佳权衡。

    arXiv:2611.02766v1 Announce Type: new  Abstract: Prompt compression is aimed at reducing input prompt lengths to enable cheaper, faster, and potentially higher-quality LLM predictions. However, existing prompt compression methods are often limited by modest compression gains, a risk of hallucination, and/or high compression latency. We propose EffComp, an Efficient prompt Compression framework for RAG-based question answering (QA). EffComp employs BERT-style document reranker and sentence selector models to allow fast extractive prompt compression at the sentence level. Its extractive nature prevents hallucinations in the compressed prompts. Additionally, the training process using a hybrid reinforcement and supervised learning approach is designed to optimize the compression ratio while preserving LLM accuracy. Experiments on various QA datasets demonstrate that EffComp outperforms state-of-the-art prompt compression methods achieving the best trade-off between compression ratio and a
    
[^55]: 基于多教师在线策略蒸馏重放的LLM持续学习

    Replay for LLM Continual Learning with Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2611.02746](https://arxiv.org/abs/2611.02746)

    本文提出持续多教师在线策略蒸馏方法，在每个训练阶段以上一阶段的学生模型作为教师重放早期领域的提示，从而无需同时保留所有领域专家模型或存储大量快照即可缓解灾难性遗忘。

    

    多教师在线策略蒸馏（MOPD）将多个领域专家蒸馏到一个学生模型中，被用于近期大型语言模型的后训练。它同时对所有领域进行训练，因此每个专家模型必须同时可用。但在实践中，各种能力往往是陆续出现的，当学习新领域时，早期领域的专家模型可能已经不复存在。按顺序蒸馏这些专家模型会导致对早期领域的灾难性遗忘。一些近期的流水线在每个阶段之后存储一个快照，最后以由所有快照进行教学的多教师恢复阶段收尾，但这样存储的模型数量会随阶段数量不断增长。与这些流水线不同，本文通过从早期阶段保留什么来形式化持续MOPD，并研究向先前学生模型的重放方法。在每个阶段内部，早期领域的提示以之前阶段的学生模型作为教师进行重放，从而无需（原文在此处截断）……

    arXiv:2611.02746v2 Announce Type: cross  Abstract: Multi-teacher on-policy distillation (MOPD) distills several domain experts into one student and is used in the post-training of recent large language models. It trains on all domains at once, so every expert must be available at the same time. In practice, capabilities often arrive one after another, and the experts of earlier domains may be gone when a new domain is learned. Distilling the experts in sequence then suffers from catastrophic forgetting of the earlier domains. Some recent pipelines store a snapshot after every stage and end with a multi-teacher recovery stage taught by all snapshots. The number of stored models then grows with the number of stages. Different from these pipelines, this paper formulates continual MOPD by what is kept from earlier stages and studies replay toward the previous student. Inside every stage, prompts of earlier domains are replayed with the student of the previous stage as their teacher, so nei
    
[^56]: MicroMouse：基于相对势能屏蔽长思维链中的有害监督信号

    MicroMouse: Masking Harmful Supervision in Long CoT via Relative Potential

    [https://arxiv.org/abs/2611.02710](https://arxiv.org/abs/2611.02710)

    提出 MicroMouse 自动化选择性 SFT 框架，通过引入相对势能（REP）来对比标准答案与匹配对照样本，衡量各推理状态的相对支持度，从而自动识别并屏蔽长思维链微调中的有害监督信号。

    

    即使是成功的长思维链（CoT）也可能包含有害监督信号，这促使我们需要能够在监督微调（SFT）过程中自动识别并屏蔽此类信号的方法。受电脑鼠迷宫求解过程的启发，我们将长思维链视为从输入问题通向目标答案的搜索轨迹。这一视角促使我们维护一个与目标相关的度量指标，类似于电脑鼠的空间坐标，用以量化当前推理状态与目标答案的接近程度。基于这一视角，我们提出了 MicroMouse，一个用于屏蔽长思维链中有害监督的自动化选择性 SFT 框架。在 MicroMouse 中，我们引入了相对势能（REP）来衡量每个推理状态对标准答案的支持程度。与仅使用标准答案的困惑度不同，REP 通过将标准答案与一个匹配的对照样本进行对比来衡量相对支持度。这种对比抵消了两者共有的似然波动，从而得到一个更加……

    arXiv:2611.02710v1 Announce Type: new  Abstract: Even successful long CoTs may contain harmful supervision, motivating the need for methods that can automatically identify and mask such signals during SFT. Inspired by the MicroMouse maze-solving process, we view a long CoT as a search trajectory from the input problem toward the target answer. This perspective motivates maintaining a goal-relative measure, analogous to a MicroMouse's spatial coordinates, to quantify how close the current reasoning state is to the target answer. Building on this perspective, we propose MicroMouse, an automated selective-SFT framework for masking harmful supervision in long CoTs. In MicroMouse we introduce a Relative Potential (REP) to measure how closely each reasoning state supports the gold answer. Unlike the perplexity of the gold answer alone, REP measures relative support by contrasting the gold answer with a matched foil. This contrast cancels likelihood variations shared by the pair, yielding a m
    
[^57]: 大语言模型无法保持中立：候选人、评估者与模型中的性别敏感性

    LLMs Cannot Be Neutral: Gender Sensitivity Across Candidates, Evaluators, and Models

    [https://arxiv.org/abs/2611.02692](https://arxiv.org/abs/2611.02692)

    该研究通过对八个模型在市议会投票任务中的30,720次判断分析，发现大语言模型在评估候选人时无法保持性别中立，明确的性别标签会显著影响其投票决策，且不同模型之间的决策行为存在显著差异。

    

    大语言模型（LLMs）越来越多地被用于评估专业职位的候选人，这引发了人们对其对人口统计身份敏感性的担忧。我们研究了候选人性别设定和评估者性别设定如何影响“大模型作为评判者”（LLM-as-a-Judge, LaJ）模型和决策语言模型（DLMs）的决策。在一个假设的市议会投票任务中，八个模型在独立变化的性别设定下，对来自NVIDIA美国和法国Nemotron-Personas数据集的240个合成职业叙事进行了评估，共产生了30,720个判断。相对于未指定性别的情况，明确的候选人性别标签使投赞成票的概率增加了1.05-1.73个百分点，而只有非二元性别评估者的设定显示出显著增加（1.26个百分点）。身份匹配未显示出显著的总体关联。各模型的赞成票比率存在差异（81.3%-97.2%），并且相对于未指定身份的基线，2.5%-3.3%的决策发生了逆转。

    arXiv:2611.02692v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to evaluate candidates for professional roles, raising concerns about their sensitivity to demographic identities. We examine how candidate and evaluator gender assignments affect decisions by LLM-as-a-Judge (LaJ) models and decision language models (DLMs). In a hypothetical city council voting task, eight models evaluated 240 synthetic professional narratives from NVIDIA's US and French Nemotron-Personas datasets under independently varied gender assignments, producing 30,720 judgments. Explicit candidate gender labels increased Yes-vote probability by 1.05-1.73 percentage points relative to unspecified gender, while only the nonbinary evaluator assignment showed a significant increase (1.26 points). Identity matching showed no significant overall association. Yes-vote rates varied across models (81.3-97.2%), and 2.5-3.3% of decisions reversed relative to the unspecified-identity baseli
    
[^58]: 训练你所运行的采样器：通过混合似然微调实现少步掩码扩散语言模型

    Train the Sampler You Run: Few-Step Masked Diffusion Language Models via Mixture-Likelihood Fine-Tuning

    [https://arxiv.org/abs/2611.02666](https://arxiv.org/abs/2611.02666)

    提出混合似然微调方法，通过在模型自身采样器似然的重要性加权界上进行LoRA微调，使掩码扩散语言模型在少步采样下生成质量超越原始模型，且微调计算量仅为预训练的0.002%。

    

    当掩码扩散语言模型以随机顺序在少数并行步骤中进行采样时，实际运行的模型是对揭示计划的混合，而交叉熵训练拟合的是词元条件分布而非这种混合。我们提出混合似然微调，它在模型自身采样器似然的重要性加权界上对预训练模型进行微调，无需教师模型，也无需更改架构或采样器。在LLaDA-8B上，仅用预训练计算量的0.002%进行LoRA微调，所生成的OpenWebText续写在所有测试步数下均优于同等计算量下的持续交叉熵训练和Alpha扩散。在模型评估次数减半的情况下（每个块16步而非32步），在三种评判标准下，其续写得分在所有随机种子下均优于原始模型，无论采用置信度解码还是随机顺序；并且在置信度解码下，两位来自不同模[态的盲评LLM评审]……（原文在此处截断）

    arXiv:2611.02666v1 Announce Type: cross  Abstract: When a masked diffusion language model is sampled in a few parallel steps in a random order, the model actually run is a mixture over reveal plans, while cross-entropy training fits token conditionals rather than this mixture. We introduce mixture-likelihood fine-tuning (Mix-FT), which fine-tunes a pretrained model on an importance-weighted bound of its own sampler's likelihood, with no teacher and no change to the architecture or the sampler. On LLaDA-8B, a LoRA fine-tune for 0.002% of the pretraining compute writes better OpenWebText continuations than continued cross-entropy and Alpha diffusion given the same compute, at every tested step count. With half the model evaluations, 16 rather than 32 steps per block, its continuations score better than the original model's in every seed under three judges, with confidence-based decoding as well as random order, and under confidence-based decoding two blinded LLM raters from different mod
    
[^59]: 通过挖掘社交媒体构建艺术界影响者网络

    Constructing an Artworld Influencers Network by Mining Social Media

    [https://arxiv.org/abs/2611.02657](https://arxiv.org/abs/2611.02657)

    该论文提出一种结合社交媒体挖掘与自然语言处理的方法，能够从少量典型艺术界参与者出发自动发现新的重要艺术界参与者并构建其网络图谱，利用Louvain方法检测社区、HITS中心性方法评估参与者权威性。

    

    作者展示了一种构建艺术界网络图谱的方法论。通过使用人工编制的一组艺术界典型参与者所发布的推文以及自然语言处理（NLP）方法，来识别重要的艺术界参与者，包括那些没有Twitter账户的参与者。被识别出的艺术界参与者形成了通过Louvain方法检测出的社区，其权威性则通过HITS中心性方法进行评估。所得到的艺术界网络展示了从一小组典型参与者出发识别新的艺术界参与者的能力，并提供了以其他方式很难收集到的大量数据。

    arXiv:2611.02657v1 Announce Type: cross  Abstract: The author shows a methodology for constructing a network map of the artworld. The tweets posted by a manually comprised set of exemplar artworld actors and Natural Language Processing (NLP) methods are used to identify significant artworld actors, including those that do not own a Twitter account. Identified artworld actors form communities detected with the Louvain method, and their authority is evaluated with the HITS centrality method. The resulting artworld network exhibits the ability, starting with a small set of exemplar actors, to identify new artworld actors and provides a wealth of data that would be very hard to collect otherwise.
    
[^60]: 变量名提供先验：语义接地使大语言模型能够进行数据高效的数值预测

    Variable names provide priors: semantic grounding enables data-efficient numerical prediction by large language models

    [https://arxiv.org/abs/2611.02637](https://arxiv.org/abs/2611.02637)

    变量名与数值之间的语义匹配（语义接地）使无需任务特定训练的大语言模型在样本稀缺的数值预测任务中匹敌甚至超越监督机器学习模型，其关键在于LLM能调用预训练中获得的领域特定知识。

    

    大语言模型（LLM）在从问答到代码生成等多种任务中均取得了出色的表现。近期研究发现，LLM可以通过上下文学习对表格数据集进行数值预测。然而，LLM在何种条件下以及如何利用变量名和上下文示例进行数值预测仍不清楚。本研究表明，语义接地（semantic grounding）——即变量名与其所描述数值之间的匹配——使五个未经任务特定训练的LLM在来自九个科学领域的20个数据集上，能够匹敌或超越七个监督机器学习（ML）模型，尤其是在示例稀少的情况下。推理轨迹表明，变量名使LLM能够调用预训练期间获得的领域特定知识。通过基于物理公式的受控任务，我们进一步证明LLM能够以某种方式将领域特定知识与上下文示例相结合

    arXiv:2611.02637v1 Announce Type: cross  Abstract: Large language models (LLMs) have achieved strong performance across diverse tasks, from question answering to code generation. Recent studies have found that LLMs can perform numerical prediction on tabular datasets via in-context learning. However, it remains unclear under what conditions and how LLMs use variable names and in-context examples for numerical prediction. Here we show that semantic grounding, the match between variable names and the values they describe, allows five LLMs without task-specific training to match or outperform seven supervised machine learning (ML) models across 20 datasets from nine scientific domains, particularly when examples are scarce. Reasoning traces indicate that variable names allow LLMs to draw on domain-specific knowledge acquired during pretraining. Using controlled tasks based on physical formulas, we further show that LLMs integrate domain-specific knowledge with in-context examples in a way
    
[^61]: SYNC：一种用于多人格对齐中动态层选择的无训练推理控制器

    SYNC: A Training-Free Inference Controller for Dynamic Layer Selection in Multi-Persona Alignment

    [https://arxiv.org/abs/2611.02636](https://arxiv.org/abs/2611.02636)

    SYNC是一种免训练的推理控制框架，通过余弦相似度轨迹动态估计各层对每个人格的敏感性，从而在每个解码步骤中将人格向量注入最优层，解决了多人格对齐中静态层选择和离线搜索成本高昂的问题。

    

    转向向量（steering vectors）可以在推理时对模型进行对齐而无需改动其权重，但同时激发多个人格与激发单个人格是截然不同的问题。先前的多人格转向方法预先确定了人格向量的组合方式以及注入这些向量的特定层。此外，这种静态的预设带来了大量离线搜索成本来识别合适的目标层，尽管最佳干预强度实际上在解码步骤和不同人格之间都是动态变化的。我们提出了SYNC（SensitivitY-guided iNference Control，敏感性引导的推理控制），这是一种新颖的免训练推理控制框架，可与现有的转向向量兼容。SYNC通过余弦相似度轨迹动态估计每一层对每个人格的敏感性，并在每个解码步骤中将每个人格分配到其最佳的注入层。在两个模型、两个人格配置以及来自无监督（摘要在此处截断）……

    arXiv:2611.02636v2 Announce Type: new  Abstract: Steering vectors align a model at inference time without touching its weights, but eliciting several personas at once is a different problem from eliciting one. Prior multi-persona steering approaches predetermine both how persona vectors are combined and the specific layers at which they are injected. Moreover, this static commitment introduces substantial offline search costs to identify suitable target layers, despite the fact that optimal intervention strengths vary dynamically across both decoding steps and distinct personas. We introduce SYNC (SensitivitY-guided iNference Control), a novel training-free inference control framework compatible with pre-existing steering vectors. SYNC dynamically estimates each layer's sensitivity to every persona via cosine similarity trajectories, assigning each persona to its optimal injection layer at every decoding step. Across two models, two persona profiles and steering vectors from both unsup
    
[^62]: ICJ-Graph：国际法院的引文图谱与分层文档语料库

    ICJ-Graph: A Citation Graph and a Layered Document Corpus of the International Court of Justice

    [https://arxiv.org/abs/2611.02630](https://arxiv.org/abs/2611.02630)

    本文构建了ICJ-Graph，一个将国际法院文件与其引用的案例、法律条款和联合国决议相连接的机器可读引文图谱与分层文档语料库，并通过引文预测任务验证了其在法律信息检索中的应用价值。

    

    我们提出了ICJ-Graph，这是一个机器可读的资源，将国际法院（ICJ）的文件与其引用的案例、法律条款和联合国决议相互连接。法院判决、书面诉状和口头记录构成不同的文档层，保留了法院推理与当事方陈述之间的区别。该数据集包含6,396份国际法院文件和505份来自其他法院及法律机构的文件，以及12,375条法律条款和20,291项联合国决议。数据文件包含文本、元数据和嵌入向量。随附的处理流程允许用户加载图谱并生成嵌入向量，以便用于自己的检索系统。为了展示该资源的用途，我们使用每项判决作出之前可获取的文件，对五个模型在199项判决的引文预测任务上进行了评估。结果表明，专为引文研究设计的工具相比无工具回答，提升了所有五个模型的引文预测F1分数。

    arXiv:2611.02630v1 Announce Type: new  Abstract: We present ICJ-Graph, a machine-readable resource connecting documents of the International Court of Justice (ICJ) to cited cases, legal provisions and United Nations resolutions. Court decisions, written pleadings and oral records form distinct document layers, preserving the distinction between the Court's reasoning and the parties' submissions. The collection contains 6,396 ICJ documents and 505 documents from other courts and legal bodies, together with 12,375 provisions and 20,291 UN resolutions. The data files contain text, metadata and embeddings. The accompanying pipeline lets users load the graph and generate embeddings for use in their own retrieval systems. To illustrate the resource's use, we evaluate five models on citation prediction for 199 decisions using documents available before each decision. Tools designed for citation research improve citation-prediction F1 over answering without tools for all five models across cas
    
[^63]: Scalaris：一种面向对话式AI助手的可扩展企业级记忆架构

    Scalaris: A Scalable Enterprise Memory Architecture for Conversational AI Assistants

    [https://arxiv.org/abs/2611.02612](https://arxiv.org/abs/2611.02612)

    Scalaris是一种通过统一的长期与短期记忆设计，解决企业级对话AI系统中检索准确性与上下文效率之间权衡问题的可扩展记忆架构。

    

    企业级对话系统需要记忆功能来个性化交互、跨会话保留信息，并维持对话的连贯性。然而，在企业规模下，记忆系统不仅要能够准确检索相关信息，还必须高效地将这些信息提供给答案生成模型。现有的记忆系统面临检索准确性与上下文效率之间的权衡：检索更多的上下文可以提高召回率，但会增加推理成本和延迟。我们提出了Scalaris，这是一种可扩展的企业级记忆架构，通过统一的长期与短期记忆设计来解决这一权衡问题。我们提出，用户交互和助手回复在对话中具有不同的特征和角色：用户交互捕获关于用户的不断演变的知识，并在跨会话中提供价值；而助手回复则必须被高效地存储，以便仅保留与当前……（摘要截断）

    arXiv:2611.02612v1 Announce Type: new  Abstract: Enterprise conversational systems require memory to personalize interactions, retain information across sessions, and maintain conversational coherence. At enterprise scale, however, memory must not only retrieve relevant information accurately but also provide it efficiently to the answer-generation model. Existing memory systems face a trade-off between retrieval accuracy and context efficiency. Retrieving more context can improve recall but increases inference cost and latency. We present Scalaris, a scalable enterprise memory architecture that addresses this trade-off through a unified long-term and short-term memory design. We propose that user interactions and assistant responses possess distinct characteristics and roles in a conversation: user interactions capture evolving knowledge about the user and provide value across sessions, whereas assistant responses must be stored efficiently so that only information relevant to the cur
    
[^64]: 它们知道圣训在哪里，却不知道它说了什么：开放大语言模型圣训召回的双语幻觉基准测试

    They Know Where It Is, But Not What It Says: A Bilingual Hallucination Benchmark of Hadith Recall in Open LLMs

    [https://arxiv.org/abs/2611.02587](https://arxiv.org/abs/2611.02587)

    本研究构建了英阿双语基准，测试五个开源大模型在六部逊尼派正典圣训集中根据出处召回圣训原文的能力，揭示了这些模型在可验证的高风险宗教文本召回任务中存在严重的幻觉问题。

    

    大语言模型越来越多地被用于查询宗教知识，包括召回特定的圣训（Hadith，即记录先知穆罕默德言行的话语与事迹）。与开放式生成不同，圣训召回是一项高风险且可验证的任务：一个出处（圣训集、章节和圣训编号）对应唯一正确的文本，任何偏离都构成伪造，且可能被误认为是真实的经文。本研究引入了一个双语（英语/阿拉伯语）基准，探测五个开放权重、可本地部署的语言模型（Mistral-7B、Gemma3-4B、Jais-7B、ALLaM-7B 和 Phi4-mini-3.8B），覆盖全部六部逊尼派正典圣训集（布哈里圣训实录、穆斯林圣训实录、艾布·达乌德圣训集、提尔米济圣训集、奈萨仪圣训集、伊本·马哲圣训集），每部圣训集抽样500条圣训（每个模型每种语言3,000条回答，总计30,000条回答）。每个模型仅获得文献出处信息，并被要求重新……[摘要原文在此处截断]

    arXiv:2611.02587v1 Announce Type: new  Abstract: Large language models are increasingly consulted for religious knowledge, including the recall of specific Hadith (recorded sayings and actions of the Prophet Muhammad, peace be upon him). Unlike open-ended generation, Hadith recall is a high-stakes, verifiable task: a reference (collection, chapter, and Hadith number) has exactly one correct text, and any deviation from it constitutes a fabrication that can be mistaken for authentic scripture. This study introduces a bilingual (English/Arabic) benchmark that probes five open-weight, locally-deployable language models (Mistral-7B, Gemma3-4B, Jais-7B, ALLaM-7B, and Phi4-mini-3.8B) across all six canonical Sunni Hadith collections (Sahih al-Bukhari, Sahih Muslim, Sunan Abi Da'ud, Jami' al-Tirmidhi, Sunan an-Nasa'i, Sunan Ibn Majah), sampling 500 Hadith per collection (3,000 per model per language, 30,000 responses total). Each model is given only the bibliographic reference and asked to re
    
[^65]: 从语言到国家再到效价：跨大语言模型的刻板印象偏见多语言审计

    From Language to Country to Valence: A Multilingual Audit of Stereotype Bias Across LLMs

    [https://arxiv.org/abs/2611.02585](https://arxiv.org/abs/2611.02585)

    本研究使用12,000条四种语言本地撰写的Reddit帖子对六个大语言模型进行性别、种族和宗教偏见审计，发现宗教偏见在所有条件下最严重，且偏见程度因语言而异——性别偏见在西班牙语中最强，种族和宗教偏见在德语中最强。

    

    大语言模型（LLM）正越来越多地被考虑应用于教育、招聘、工作场所决策支持等具有重大影响的应用场景，这使得部署前的社会偏见检测变得至关重要。然而，现有的主要基准测试以英语为中心或依赖翻译文本，且主要关注负面刻板印象，可能忽略了语言和文化上的差异。我们使用12,000条以英语、西班牙语、法语和德语本地撰写的Reddit帖子来研究性别、种族和宗教偏见。仅搜索词经过翻译，从而保留了本地语境。数据集在语言和类别上保持平衡；英语的性别部分还额外按刻板印象效价进行了平衡。我们采用双空设计（可减少语法性别线索）评估了三个开放权重模型和三个基于API的专有模型。在所有模型-语言组合条件下，宗教类偏见的绝对选择不平衡度最高。性别偏见在西班牙语中最高，而种族和宗教偏见在德语中最高。

    arXiv:2611.02585v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly considered for education, recruitment, workplace decision support, and other consequential applications, making pre-deployment social-bias detection essential. Yet major benchmarks are English-centred or translated, focus mainly on negative stereotypes, and may miss linguistic and cultural variation. We investigate gender, racial, and religious bias using 12,000 locally written Reddit posts in English, Spanish, French, and German. Only search terms are translated, preserving local context. The dataset is balanced by language and category; English gender is additionally balanced by stereotype valence. Three open-weight and three proprietary API-based models are evaluated with a two-blank design that reduces grammatical-gender cues. Religion has the highest absolute choice imbalance in every model-language condition. Spanish is highest for gender, whereas German is highest for race and religion
    
[^66]: SMILE：一种用于肝脏超声理解的结构化多智能体临床推理框架

    SMILE: A Structured Multi-Agent Clinical Reasoning for Liver Ultrasound Understanding

    [https://arxiv.org/abs/2611.02581](https://arxiv.org/abs/2611.02581)

    该论文提出SMILE框架，通过多个互补的专科智能体解读肝脏超声图像，并利用裁判引导的共识机制整合结果，从而克服单一推理路径的局限，提升肝脏超声报告生成的质量。

    

    自动化的肝脏超声报告生成通过生成具有临床结构化格式的解剖结构与疾病相关发现的描述，有助于改善肝脏疾病的诊断。然而，当前的生成式视觉-语言模型（VLMs）通常依赖单一的推理路径，将超声图像表现直接映射为文本描述。除了视觉与语言模态之间固有的差异外，这种设计在肝脏超声任务中存在局限，因为疾病解读受到细微视觉证据、异质性疾病模式以及有限配对监督数据的制约。为了解决这些局限性，我们提出了SMILE，一个结构化多智能体临床推理框架，它利用多个互补的专科智能体来解读超声图像，并通过裁判引导的共识机制整合各智能体的输出。我们的方法采用三个专门化智能体：用于形态学描述的图像解读智能体，……

    arXiv:2611.02581v1 Announce Type: cross  Abstract: Automated liver ultrasound report generation improves the diagnosis of liver diseases by producing clinically structured descriptions of anatomical and disease-related findings. However, current generative vision-language models (VLMs) often rely on a single reasoning pathway that maps ultrasound appearance directly to textual descriptions. In addition to the inherent vision-language modality gap, this design is limited for liver ultrasound because disease interpretation is constrained by subtle visual evidence, heterogeneous disease patterns, and limited paired supervision. To address these limitations, we propose SMILE, a Structured Multi-agent clInicaL rEasoning framework that uses multiple complementary specialist agents to interpret ultrasound images and consolidates their outputs through a judge-guided consensus mechanism. Our approach employs three specialized agents: an image interpretation agent for morphological descriptions,
    
[^67]: Re³Rank：基于推理条件化检索头的列表式重排序

    Re$^3$Rank: Listwise Reranking with Reasoning-Conditioned Retrieval Heads

    [https://arxiv.org/abs/2611.02550](https://arxiv.org/abs/2611.02550)

    提出基于推理条件化检索头的列表式重排序器Re³Rank，通过直通估计器联合训练推理生成器与评分器，其4B模型平均超越最强基线3.11个NDCG@10点，0.6B模型即可胜过32B参数的重排序器。

    

    基于注意力的重排序器通过检索头从问题路由到各候选文档的注意力质量来为其评分，但问题作为固定查询，所携带的用于定位相关证据的信息有限。我们提出了Re³Rank，一个建立在推理条件化检索头（RRHeads）之上的列表式重排序器。生成器首先针对问题和候选文档写出一条推理轨迹，然后评分器将轨迹中的token作为若干选定RRHeads的注意力查询，把相关性读取为连续分数。我们提出了一种直通估计器，使排序损失能够穿过离散的推理轨迹进行传播，从而让生成器和评分器可以针对排序任务进行联合训练。在9个基准测试上的平均结果显示，Re³Rank优于11个基线重排序器：我们的4B版本平均超过最强基线3.11个NDCG@10点，甚至我们的0.6B版本也超过了参数量高达32B的重排序器。

    arXiv:2611.02550v1 Announce Type: new  Abstract: Attention-based rerankers score each candidate by the attention mass that retrieval heads route to it from the question, but the question, serving as a fixed query, carries limited information for locating relevant evidence. We propose Re$^3$Rank, a listwise reranker built on reasoning-conditioned retrieval heads (RRHeads). A generator first writes a reasoning trace over the question and the candidates, and a scorer then uses the trace tokens as the attention query of a few selected RRHeads, reading relevance as continuous scores. We present a straight-through estimator to propagate the ranking loss through the discrete trace, so that the generator and the scorer can be trained jointly for ranking. Averaged over 9 benchmarks, Re$^3$Rank outperforms 11 baseline rerankers: our 4B variant exceeds the strongest baseline by 3.11 NDCG@10 points on average, and even our 0.6B variant surpasses rerankers with up to 32B parameters.
    
[^68]: FindLink：基于来源锚定前提学习评估科学关联

    FindLink: Learning to Assess Scientific Connections from Source-Grounded Premises

    [https://arxiv.org/abs/2611.02545](https://arxiv.org/abs/2611.02545)

    该论文提出 GI-TraceBench 基准与 FindLink 两阶段框架，通过构建来源锚定前提并进行结构化推理，结合确定性编译器反馈，来评估不同研究条件下科学发现之间的关联。

    

    arXiv:2611.02545v1 公告类型：新论文。摘要：基于文献的科学发现需要评估在不同研究条件下所得发现之间的关联。来源归因使单个发现可追溯，但并不能确定它们共同支持什么。我们提出 GI-TraceBench，一个包含 10,282 个问题的胃肠肿瘤学基准，通过共享的论文、主张和证据片段记录，统一了来源溯源问答、知识发现和假设评估三类任务。我们进一步提出 FindLink，一个面向来源锚定科学推理的两阶段框架。前提锚定阶段构建带有明确研究背景和支持状态的来源绑定前提；结构化推理阶段将这些前提组织成同时表示支持关系与未决条件的支持计划，并利用确定性编译器反馈生成以证据为条件的答案。该编译器验证结构与来源绑定的一致性，但……（原文在此处截断）

    arXiv:2611.02545v1 Announce Type: new  Abstract: Scientific discovery from the literature requires assessing connections among findings obtained under different study conditions. Source attribution makes individual findings traceable, but does not determine what they jointly support. We introduce GI-TraceBench, a benchmark of 10,282 questions in gastrointestinal oncology that unifies provenance question answering, knowledge discovery, and hypothesis evaluation through shared Paper, Claim, and Evidence Span records. We further propose FindLink, a two-stage framework for provenance-grounded scientific reasoning. Premise Grounding constructs source-bound premises with explicit study context and support status. Structured Reasoning organizes these premises into support plans that represent both supported relations and unresolved conditions, then generates evidence-conditioned answers using deterministic compiler feedback. The compiler verifies structural and source-binding consistency, but
    
[^69]: PixelBench：大语言模型能够逐像素地理解和生成图像吗？

    PixelBench: Can LLMs Understand and Generate Images Pixel by Pixel?

    [https://arxiv.org/abs/2611.02540](https://arxiv.org/abs/2611.02540)

    该论文提出PixelBench基准，证明大语言模型无需专门的图像分词器或解码器，仅凭现有词表即可直接生成可识别的像素图像，且像素理解能力与生成能力密切相关。

    

    大语言模型（LLM）可以通过SVG和CAD等基于文本的格式生成视觉素材，但这些符号化表示是使用高级图元来描述图像，而非显式的光栅数值。我们探讨LLM能否直接利用其现有词表生成像素图像，而无需经过学习的图像分词器或解码器。我们提出了PixelBench，这是一个涵盖多种分辨率的文本到像素（T2P）和图像到像素（I2P）生成基准，并附带一个互补的像素理解评估。模型首先生成一个RGB调色板，然后生成一个索引网格，用于指定每个输出像素对应的调色板颜色。在T2P任务中，最强的模型能够直接从文本生成可识别的图像。在I2P任务中，最强的模型在低分辨率下的表现超越了调色板匹配的下采样方法和专门的像素化算法。像素理解性能与生成性能之间存在很强的相关性。

    arXiv:2611.02540v1 Announce Type: cross  Abstract: Large language models (LLMs) can generate visual assets through text-based formats such as SVG and CAD, but these symbolic representations describe images using high-level primitives rather than explicit raster values. We ask whether LLMs can instead generate pixel images directly through their existing vocabulary, without a learned image tokenizer or decoder. We introduce PixelBench, a benchmark for text-to-pixel (T2P) and image-to-pixel (I2P) generation across multiple resolutions, together with a complementary pixel-understanding evaluation. Models generate an RGB color palette followed by a grid of indices specifying the palette color of each output pixel. In T2P, the strongest models produce recognizable images directly from text. In I2P, the strongest models outperform palette-matched downsampling and specialized pixelization algorithms at low resolutions. Pixel-understanding performance correlates strongly with generation perfor
    
[^70]: 先调优基线：Muon缩放权重衰减的收益大部分是调优差距，但其微小的真实优势会随模型规模增长

    Tune the Baseline First: Most of Muon's Scaled Weight Decay Gain Is a Tuning Gap, but Its Small Real Advantage Grows With Model Size

    [https://arxiv.org/abs/2611.02529](https://arxiv.org/abs/2611.02529)

    该研究在公平联合调优学习率与权重衰减后发现，Muon缩放权重衰减约90%的训练效率优势其实是基线调优不足造成的差距，仅剩约1.03倍的真实微小优势，且该优势随模型规模增大而增长。

    

    已有报道称，针对Muon优化器的缩放权重衰减（Muon-SW）比恒定解耦权重衰减能更高效地训练语言模型，但此类比较通常对两种方法复用同一组学习率（LR）和权重衰减（WD）设置。我们在MAI-Base-2协议下重新运行了这一比较：对每种方法通过LR与WD的乘积进行联合调优，并采用源缩放的混合专家（MoE）扩展阶梯（活跃参数从190万到1500万），在所有规模上保持每活跃参数token数和数据重复次数固定。我们发现：(1) 大部分收益其实是调优差距。在共享设置下，Muon-SW以每字节0.035比特（BPB）的优势获胜，相当于1.52倍的计算节省；而在两种方法均被调优后，差距缩小为0.0026 BPB（95%置信区间[0.0018, 0.0034]），相当于1.03倍的节省，即约90%的优势消失。(2) 剩余的微小优势是真实的：在最小规模下，四个配对随机种子在调优后的设置下均偏向Muon-SW。(3) 该优势……（摘要在此处截断，结合标题可知：这一优势随模型规模增大而增长）

    arXiv:2611.02529v1 Announce Type: new  Abstract: Scaled weight decay for Muon (Muon-SW) has been reported to train language models more efficiently than constant decoupled weight decay, but such comparisons usually reuse one learning-rate (LR) and weight-decay (WD) setting for both methods. We re-run the comparison under the MAI-Base-2 protocol: LR and WD are tuned jointly through their product for each method, on a source-scaled mixture-of-experts (MoE) scaling ladder (1.9M to 15M active parameters) that holds tokens per active parameter and data repetition fixed across sizes. We find: (1) Most of the gain is a tuning gap. At the shared setting Muon-SW wins by 0.035 bits per byte (BPB), a 1.52x compute saving; after both methods are tuned, the gap is 0.0026 BPB (95% CI [0.0018, 0.0034]), a 1.03x saving, so about 90% of the advantage disappears. (2) The small advantage that remains is real: at the smallest size, all four paired seeds favour Muon-SW at the tuned settings. (3) That advan
    
[^71]: 排名但未阻止：从类型化决策模型的输出概率中检测提示注入劫持

    Ranked but Not Stopped: Detecting Prompt-Injection Hijacks from Typed Decision Models' Output Probabilities

    [https://arxiv.org/abs/2611.02522](https://arxiv.org/abs/2611.02522)

    基于输出概率低置信度的提示注入劫持检测方法只能有效“排序”而难以“拦截”——在 AG News 分类实验中，即使按 5% 误报率校准阈值，监视器仍会漏掉 51-70% 的劫持，说明置信度信号不足以构建可靠的防御门控。

    

    将语言模型用作分类器的程序会根据其最可能的标签采取行动。注入到输入中的文本可以将该标签更改为攻击者所选择的标签，即一种“劫持”。一个不知道攻击者标签的监视器可以读取模型自身的置信度，并在最高概率较低时发出警报。我们在 AG News 主题分类任务上对该监视器进行了测试，共使用五个系统，其中三个通过提供商的 API 提供。在五类注入文本上汇总结果，低置信度信号在五个系统中的四个上能将劫持的答案排在干净答案之上（AUROC 0.86-0.90）。但仅仅排序并不能阻止其中的多数。一个以干净校准文章为基准、误报率至多 5% 的阈值落在最不自信的干净答案之中，而所有更自信的劫持都能通过。在对三个 API 服务系统进行的主要门控实验中，该监视器漏掉了由简单注入文本引起的 51-70% 的劫持（各系统的点估计）。

    arXiv:2611.02522v1 Announce Type: new  Abstract: Programs that use a language model as a classifier act on its most probable label. Text injected into the input can move that label to one the attacker chooses, a hijack. A monitor that does not know the attacker's label can read the model's own confidence and raise an alarm when the top probability is low. We test this monitor on AG News topic classification with five systems, three of them served through a provider's API. Pooled over five families of injected text, low confidence ranks hijacked answers above clean ones on four of the five systems (AUROC 0.86-0.90). Ranking them does not stop many of them. A threshold set so that at most 5% of clean answers to separate calibration articles alarm falls among the least confident clean answers, and every more confident hijack passes. In our main gate run on the three API-served systems, the monitor misses 51-70% of the hijacks caused by a plain injected note (per-system point estimates). W
    
[^72]: 相同的句子，不同的构念：身份互换的推理揭示LLM仇恨言论分类器中的测量非不变性

    Same Sentence, Different Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers

    [https://arxiv.org/abs/2611.02491](https://arxiv.org/abs/2611.02491)

    该研究通过仅替换身份词的对照模板实验发现，尽管LLM仇恨言论分类器总体准确率超过0.95，但部分模型对针对女性的仇恨言论漏检率显著更高，且对完全相同的句子因身份不同而采用不同的判定构念，首次在LLM时代揭示了其测量非不变性问题。

    

    大型语言模型越来越多地被用作测量仇恨言论等具有重大社会影响的构念的测量工具，且通常通过其与人类标签的总体一致率来进行验证。然而，一个有效的测量工具必须对其所测量的每一个群体应用相同的构念。我们使用421个模板对这一要求进行检验，这些模板针对七个受保护群体以完全相同的方式实例化，唯一变化的只有身份词。四个LLM分类器的平衡准确率均超过0.95，然而在完全相同的模板内，其中两个分类器漏检针对女性的仇恨言论的比率明显高于针对其他群体的相同言论（超额假阴性率分别为8.3和4.4个百分点），这一差距最早在2021年针对微调分类器的研究中被报告。我们展示了构念是如何发生偏移的：通过比较每个模板内的推理依据，我们发现模型会对其他群体援引受保护群体身份，但对针对女性（原文此处截断）……

    arXiv:2611.02491v1 Announce Type: new  Abstract: Large language models are increasingly used as measurement instruments for socially consequential constructs such as hate speech, and are typically validated by their aggregate agreement with human labels. A valid instrument must also apply the same construct to every group it measures. We test this requirement using 421 \hc{} templates that are instantiated identically for seven protected groups, so that only the identity term varies. Four LLM classifiers each exceed 0.95 balanced accuracy, yet within identical templates two of them miss hateful statements about women at markedly higher rates than the same statements about other groups (excess false-negative rates of 8.3 and 4.4 percentage points), a gap first reported for fine-tuned classifiers in 2021. We show how the construct shifts. Comparing rationales within each template, we find that models invoke protected-group status for other groups but recast the identical statement about 
    
[^73]: 按需链：面向长期对话记忆的延迟式证据组织

    Chain on Demand: Deferred Evidence Organization for Long-Term Conversational Memory

    [https://arxiv.org/abs/2611.02480](https://arxiv.org/abs/2611.02480)

    提出按需链（CoD）框架，将证据组织推迟到查询时刻进行，在不修改底层对话记忆的前提下动态构建临时证据链，在LoCoMo基准上显著提升了长期对话记忆问答的准确率。

    

    长期对话记忆面临一个时机错配问题：写入时的摘要和图结构会在未来查询未知的情况下预先组织历史，而独立检索则可能将有限的证据预算浪费在冗余或碎片化的对话轮次上。我们提出了按需链（Chain on Demand, CoD），这是一个在保持轮次级记忆不变的前提下、于查询时进行证据组织的框架。给定一个宽泛的候选池，CoD根据查询相关性、相对于已选锚点的词汇多样性以及新覆盖的查询词来分配固定的锚点预算，从而构建一条临时证据链，随后对每个被选中锚点的同会话邻域进行确定性恢复。所得的证据链仅用于当前查询，而底层对话记忆保持不变。在覆盖全部1,986个LoCoMo问题的配对比较中，CoD在1,540个非对抗性问题上达到82.2%的准确率，在全部1,986个问题上达到76.3%的准确率。在LongMemEval-（摘要在此处截断）

    arXiv:2611.02480v1 Announce Type: new  Abstract: Long-term conversational memory faces a timing mismatch: write-time summaries and graphs organize history before a future query is known, while independent retrieval can allocate a limited evidence budget to redundant or fragmented turns. We introduce Chain on Demand (CoD), a query-time evidence organization framework over an unchanged turn-level memory. Given a broad candidate pool, CoD constructs a temporary evidence chain by allocating a fixed anchor budget according to query relevance, lexical diversity relative to selected anchors, and newly covered query terms, followed by deterministic restoration of each selected anchor's same-session neighborhood. The resulting chain is used only for the current query, while the underlying conversational memory remains unchanged. In a paired comparison over all 1,986 LoCoMo questions, CoD achieves 82.2% accuracy on 1,540 non-adversarial questions and 76.3% on all 1,986 questions. On LongMemEval-
    
[^74]: GigaSpeech 3：利用弱标注数据扩展多方言阿拉伯语语音识别

    GigaSpeech 3: Scaling Multi-Dialect Arabic Speech Recognition with Weakly Labeled Data

    [https://arxiv.org/abs/2611.02476](https://arxiv.org/abs/2611.02476)

    研究者构建了包含39,000小时弱标注数据的大规模多方言阿拉伯语语音语料库GigaSpeech 3，通过自监督学习与噪声学生训练策略，使方言专用语音识别模型的宏平均词错误率达到23.86%，相对最强第三方模型降低16.6%。

    

    方言阿拉伯语语音识别一直受到方言标注转写语音稀缺的制约，尽管网络音频资源十分丰富。我们提出了GigaSpeech 3，这是一个包含39,000小时、1000万个弱转写语音片段的语料库，涵盖阿尔及利亚、埃及、伊拉克、沙特和叙利亚五种阿拉伯语方言。从每种方言60小时的人工转写出发，我们使用偏向语音识别的自监督学习训练标注模型，并将其应用于方言匹配的网络音频。通过数据筛选以及对沙特和叙利亚阿拉伯语的四轮噪声学生训练精炼，最终得到该语料库。我们通过方言专用和多方言语音识别训练以及跨领域评估验证了其价值。在GigaSpeechBench上，最终的方言专用模型在全部五种方言上均优于其对应的标注模型，实现了23.86%的宏平均词错误率，相比最强的第三方（模型）实现了16.6%的相对降低。

    arXiv:2611.02476v2 Announce Type: new  Abstract: Dialectal Arabic speech recognition remains constrained by the scarcity of dialect-resolved transcribed speech despite the abundance of web audio. We introduce GigaSpeech 3, a corpus of 39,000 hours and 10 million weakly transcribed speech segments covering Algerian, Egyptian, Iraqi, Saudi, and Syrian Arabic. Starting from 60 hours of manual transcription per dialect, we train labeling models using speech-recognition-biased self-supervised learning and apply them to dialect-matched web audio. Data selection and four rounds of noisy-student refinement for Saudi and Syrian Arabic yield the final corpus. We validate its utility through dialect-specific and multi-dialect speech recognition training and cross-domain evaluation. On GigaSpeechBench, the final dialect-specific models outperform their labeling models across all five dialects and achieve a macro-average word error rate of 23.86%, a 16.6% relative reduction from the strongest third
    
[^75]: 如预期吗？通过期望-观察一致性验证实现高效的大语言模型规划

    As Expected? Efficient LLM Planning via Expectation-Observation Consistency Verification

    [https://arxiv.org/abs/2611.02438](https://arxiv.org/abs/2611.02438)

    本文提出期望-观察一致性验证（EOCV）机制，通过将批处理执行器对动作结果的预期与实际观察结果比对，一致则继续执行、不一致则提前终止并重新规划，在保持任务性能的同时将推理成本降低29-35%。

    

    大语言模型（LLM）规划通常依赖反应式执行器（如ReAct），它在每一步生成单个动作以保持与最新观察结果的一致性，但这会带来高昂的推理成本。批处理执行器通过在单次模型调用中预测多个动作来降低这一成本，然而由于忽略了中间观察结果，容易产生误差累积。我们提出了一种面向批处理执行器的期望-观察一致性验证（EOCV）机制，在提高效率的同时保持与环境的贴合。该机制生成对动作结果的预期，并将其与实际观察结果进行比较：若一致，则继续执行；若不一致，则提前终止并基于当前观察结果重新规划。在基于文本和多模态的基准测试上、跨多个骨干模型的实验表明，EOCV与反应式执行器的平均任务性能大致相当，同时将推理成本降低了29-35%。

    arXiv:2611.02438v1 Announce Type: new  Abstract: LLM planning often relies on reactive executors (e.g., ReAct) that generate a single action at each step to remain grounded in the latest observations, incurring substantial inference cost. Batched executors reduce this cost by predicting multiple actions in a single model invocation, yet they are prone to compounding errors, as intermediate observations are ignored. We propose an expectation-observation consistency verification (EOCV) mechanism for batched executors, preserving grounding while improving efficiency. The mechanism generates expectations of action outcomes and compares them with the resulting observations. Consistency permits continued execution, whereas mismatches trigger early termination and replanning from the current observation. Experiments on text-based and multimodal benchmarks across multiple backbones show broadly comparable average task performance between EOCV and reactive executors, with a 29-35% reduction in 
    
[^76]: 使用LSIRM诊断语言模型基准测试中的模型-题目交互作用

    Diagnosing Model--Item Interactions in Language Model Benchmarks with LSIRM

    [https://arxiv.org/abs/2611.02432](https://arxiv.org/abs/2611.02432)

    本文利用潜在空间项目反应模型（LSIRM）将语言模型在基准测试中与题目之间的交互模式可视化，发现图谱上相邻题目常共享相同正确答案，且使用选项标签评分比使用选项文本评分时这一模式更为显著。

    

    具有相似基准准确率的语言模型可能在不同的题目上取得成功。总体分数无法显示它们的表现究竟在何处存在差异，也无法显示哪些题目属性可能解释这些差异。我们展示了潜在空间项目反应模型（LSIRM）如何帮助评估者研究这些模式及其对模型比较的影响。我们分析了1,348个语言模型在5,761个BBH题目上的表现，并比较了BBH、GPQA、MMLU-Pro和MuSR四个基准中呈现和评分答案的不同方式。我们考察了LSIRM图谱——它总结了模型与题目之间表现的差异——并将其与题目属性以及正确和错误回答的记录相结合进行分析。在BBH中，图谱上相邻的题目往往共享相同的正确答案，例如“是”或“否”。在四个基准测试中，当使用选项标签（如A、B、C、D）对答案进行评分时，这种模式比使用选项文本进行评分时更为明显。随后，我们估计了每个模型的倾向……（摘要在此处截断）

    arXiv:2611.02432v1 Announce Type: new  Abstract: Language models with similar benchmark accuracy can succeed on different items. Overall scores do not show where their performance differs or which item properties might explain those differences. We demonstrate how the latent space item response model (LSIRM) can help evaluators investigate these patterns and their consequences for model comparison. We analyze 1,348 language models on 5,761 BBH items and compare ways of presenting and scoring answers across BBH, GPQA, MMLU-Pro, and MuSR. We examine LSIRM maps, which summarize differences in performance across models and items, alongside item properties and records of correct and incorrect responses. In BBH, nearby items on these maps often share the same correct answer, such as "Yes" or "No." Across the four benchmarks, this pattern is stronger when answers are scored using option labels, such as A, B, C, and D, than when using the text of the options. We then estimate each model's tend
    
[^77]: InteractionBench：测量实时语音系统的对话时序、任务准确性、耳语与音量以及接地能力

    InteractionBench: Measuring Timing, Task Accuracy, Whisper and Volume, and Grounding in Real-Time Voice Systems

    [https://arxiv.org/abs/2611.02430](https://arxiv.org/abs/2611.02430)

    提出了统一测量实时语音系统对话时序、口语任务准确性、耳语音量控制和长对话接地能力四个维度的基准测试 InteractionBench，并对 GPT-Live、Gemini Live、Moshi 等七个系统进行了 1,519 次评分运行评估。

    

    大多数针对实时语音系统的基准测试只关注单一维度，例如话轮转换、指令遵循、副语言控制或对话记忆，且各自使用不同的数据和系统集合。我们提出了 InteractionBench，这是一个基准测试与评估框架，能够在相同系统上基于同一段录制的用户音频测量四个维度：对话时序、口语任务准确性、耳语与音量控制，以及超过二十轮长对话中的接地能力。该基准测试包含十三个场景，每个系统 221 个测试项。在任何系统运行之前，我们通过规则预先确定口语任务的预期话轮控制动作，并从脚本化对话中获取预期答案。我们根据音频信号对耳语进行评分。我们通过 1,519 次评分运行评估了 GPT-Live、GPT-Realtime、Gemini Live、Moshi、PersonaPlex、Freeze-Omni 以及一个基于话轮的级联系统。在六个全双工系统中，时序表现与口语任务准确性呈秩相关（摘要在此处截断）。

    arXiv:2611.02430v1 Announce Type: cross  Abstract: Most benchmarks for real-time voice systems focus on one aspect, such as turn-taking, instruction following, paralinguistic control, or conversational memory, each with its own data and set of systems. We present InteractionBench, a benchmark and evaluation harness that measures four dimensions on the same systems from the same recorded user audio: conversational timing, spoken-task accuracy, whisper and volume, and groundedness in conversations of more than twenty turns. The benchmark contains 221 items per system in thirteen scenarios. Before any system runs, we fix the expected floor-control actions for spoken tasks by rule and take the expected answers from scripted dialogues. We score whispering from the audio signal. We evaluate GPT-Live, GPT-Realtime, Gemini Live, Moshi, PersonaPlex, Freeze-Omni, and a turn-based cascade in 1,519 graded runs. Among the six full-duplex systems, timing and spoken-task accuracy are rank-correlated 
    
[^78]: 当奖励依赖于组群时，GRPO 就在进行一场锦标赛

    When Rewards Depend on the Group, GRPO Plays a Tournament

    [https://arxiv.org/abs/2611.02404](https://arxiv.org/abs/2611.02404)

    该论文证明GRPO的组相对基线会把组依赖奖励（如多样性奖励）的优化目标扭曲为零和锦标赛，导致模型感知不到甚至负向对待组内同伴的奖励，并提出α=+1的Team-GRPO无偏估计器，只需每个提示两次采样即可正确优化任何组依赖奖励。

    

    针对多样性、覆盖度和分布匹配的奖励使得每个采样响应的奖励取决于其组内的其他响应，而这类奖励如今通常使用GRPO和RLOO来优化。我们证明在这一设定下，组相对基线会改变优化目标。期望更新等于每个响应自身奖励的梯度，再加上α乘以其对组内同伴奖励的影响，其中对于大小为K的组，α=-1/(K-1)，因此GRPO实际上是在训练每个响应在一场零和锦标赛中击败其组内同伴。当K=2时，它完全无法感知成对多样性奖励；在任何K值下，它感知到的都不足一半，而且其效果常常比完全不使用基线还要差。将α设为+1即得到Team-GRPO，这是一种对任何依赖于组群的奖励都无偏的估计器，只需每个提示采样两次即可工作，且在处理成对奖励时无需额外的模型调用。该理论能够定量预测训练过程，例如将一个9B策略的不动点定位到0.02的精度范围内。

    arXiv:2611.02404v1 Announce Type: cross  Abstract: Rewards for diversity, coverage and distribution matching make the reward of each sampled response depend on the rest of its group, and they are now routinely optimized with GRPO and RLOO. We prove that in this setting the group-relative baseline changes the objective. The expected update equals each response's own reward gradient plus $\alpha$ times its effect on its groupmates' rewards, with $\alpha=-1/(K-1)$ for groups of size $K$, so GRPO trains every response to beat its groupmates in a zero-sum tournament. It cannot see a pairwise diversity bonus when K=2, sees less than half of it at any K, and often does worse than using no baseline at all. Setting $\alpha=+1$ gives Team-GRPO, an unbiased estimator for any group-dependent reward that works with two samples per prompt and requires no extra model calls for pairwise rewards. The theory predicts training quantitatively. It locates the fixed points of a 9B policy to within 0.02 in $
    
[^79]: 大语言模型作为形式化器时的求解器式行为

    Solver-Like Behavior in Large Language Models as Formalizers

    [https://arxiv.org/abs/2611.02395](https://arxiv.org/abs/2611.02395)

    该研究首次定义并量化了“大语言模型作为形式化器”中的“求解器式行为”：85%的推理过程在程序执行前就完成了求解器的工作，占用24%的推理token，但由于程序通常会重新计算答案，这些努力大多是冗余的。

    

    在“大语言模型作为形式化器”（LLM-as-formalizer）范式中，大语言模型将问题翻译为形式化程序，并把搜索任务留给求解器，从而获得求解器的保证。如今的形式化器通常是有推理能力的模型，它们会在编写程序之前先写出长长的推理过程，然而它们却仅根据程序的输出来评判。我们发现，这些推理过程经常在做求解器的工作——赋值、测试并排除候选值，直到往往得出完整答案；我们将这种现象称为“求解器式行为”。我们在推理过程、程序及其输出之上定义了SLB的本体分类，并使用经过人工标注校验的大语言模型裁判，对来自三个推理模型在四个约束满足数据集上的2,000个生成结果进行了标注，每个数据集分别以一种声明式和一种命令式形式化语言进行形式化。SLB出现在85%的推理过程中，占用了24%的推理token，然而程序通常会重新计算答案，因此这些努力是冗余的。在那9%继承（推理答案）的程序中……（原文摘要在此处截断）

    arXiv:2611.02395v1 Announce Type: new  Abstract: In the LLM-as-formalizer paradigm, a large language model translates a problem into a formal program and leaves the search to a solver, gaining the solver's guarantees. Formalizers are now often reasoning models that write a long reasoning trace before the program, yet they are judged only by the program's output. We find that the trace routinely does the solver's job, assigning, testing and ruling out values until it often reaches the full answer; we call this solver-like behavior (SLB). We define an ontology of SLB over the trace, the program and its output, and use LLM judges, checked against human labels, to label 2,000 generations from three reasoning models on four constraint satisfaction datasets, each formalized in a declarative and an imperative formal language. SLB appears in 85% of traces and takes 24% of reasoning tokens, yet the program usually recomputes the answer, so the effort is redundant. The 9% of programs that inheri
    
[^80]: EviCast：通过共享响应校准预测检索收益

    EviCast: Forecasting Retrieval Gains Through Shared Response Calibration

    [https://arxiv.org/abs/2611.02387](https://arxiv.org/abs/2611.02387)

    EviCast通过一次性测量生成器对受控证据状态的响应，并将其与各候选检索器的离线证据状态分布相结合，无需实际执行每个RAG配置即可预测检索的端到端净效用及置信下界。

    

    检索增强生成（RAG）通过外部证据扩展了大语言模型的能力，然而相关证据并不能保证生成器能够有效利用它。不加区分的检索可能降低任务性能，同时增加检索和长上下文推理的成本。现有方法既无法在大规模执行之前判断某个生成器、任务分布和检索器配置是否值得部署，也无法在共享预算下诊断多个候选检索器。我们提出了EviCast，这是一个通过干预性证据状态分解来预测检索价值的框架。EviCast只需测量一次生成器对受控证据状态的响应，并将这一共享的响应特征与每个候选检索器的离线证据状态分布相结合，从而无需实际执行每个RAG配置即可得出端到端的净效用及其95%置信下界。状态细化分析刻画了其……（原文在此处被截断）

    arXiv:2611.02387v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) extends large language models with external evidence, yet relevant evidence does not ensure that a generator can use it. Indiscriminate retrieval can degrade task performance while adding retrieval and long-context inference costs. Existing methods neither determine before large-scale execution whether a generator, task distribution, and retriever configuration is worth deploying nor diagnose multiple candidate retrievers under a shared budget. We introduce EviCast, a framework that forecasts retrieval value through interventional evidence-state decomposition. EviCast measures a generator's response to controlled evidence states once and combines this shared profile with the offline evidence-state distribution of each candidate retriever, yielding the end-to-end net utility and its 95% lower confidence bound without executing every RAG configuration. A state-refinement analysis characterizes when its 
    
[^81]: 超越准确率：量化如何影响编程智能体

    Beyond Accuracy: How Quantization Affects Coding Agents

    [https://arxiv.org/abs/2611.02368](https://arxiv.org/abs/2611.02368)

    量化虽然不会显著降低编程智能体的最终准确率，但会实质性地改变其解题行为和失败模式，表明仅凭准确率不足以评估量化对编程智能体的影响。

    

    智能体在编程任务中的应用日益增多，但大型语言模型（LLM）的推理成本限制了它们的部署范围。量化可以减小模型体积，但其对执行和修改程序的编程智能体的影响仍缺乏充分研究。我们在OJBench编程问题上，使用三个随机种子，比较了BF16与4-bit AWQ和GPTQ两种量化方法在Qwen2.5-Coder-Instruct（7B、14B和32B）和Devstral-Small-2507（24B）上的表现。我们没有检测到任何编程智能体存在统计上显著的准确率损失。然而，最终准确率可能掩盖实质性的行为变化：AWQ使Qwen-7B的一次性解题数从17降至0，同时使基于循环的解题数从10增至25。Qwen-32B的所有成功案例都涉及工具使用，而Devstral很少进行有效的执行器调用。失败特征也有所不同：Qwen-32B比更小的Qwen模型产生多得多的编译错误，且在量化下略有上升，而Devstral的失败模式则显著地从……

    arXiv:2611.02368v1 Announce Type: new  Abstract: Agents are increasingly used for coding tasks, but the inference cost of large language models (LLMs) limits where they can run. Quantization reduces model size, but its effects on coding agents that execute and revise programs remain underexplored. We compare BF16 with 4-bit AWQ and GPTQ for Qwen2.5-Coder-Instruct (7B, 14B, and 32B) and Devstral-Small-2507 (24B) on OJBench coding problems across three seeds. We detect no statistically significant accuracy loss for any of the coding agents. However, final accuracy can conceal substantial behavioral changes: AWQ reduces Qwen-7B one-shot solves from 17 to 0 while increasing loop-based solves from 10 to 25. All Qwen-32B successes involve tool use, whereas Devstral rarely makes a valid executor call. Failure profiles also differ: Qwen-32B produces far more compilation errors than the smaller Qwen models, rising slightly under quantization, while Devstral's failures shift significantly from r
    
[^82]: VisCodeBench：一个反映真实世界实践的文本到可视化新基准数据集

    VisCodeBench: a new benchmark dataset for Text-to-Vis reflecting real-world practice

    [https://arxiv.org/abs/2611.02353](https://arxiv.org/abs/2611.02353)

    该论文提出VisCodeBench，一个包含2,672个来自教育、科学和大众受众领域样本的端到端文本到可视化基准数据集，评估发现现有LLM可视化系统虽能基本成功执行代码，但在复现参考数据方面表现不佳，最佳数据相似度仅为0.81。

    

    从自然语言请求生成可视化（文本到可视化）使用户能够更高效地探索数据。虽然大型语言模型（LLM）能够完成简单的可视化请求，但现有基准数据集并未充分覆盖它们在具有挑战性的请求上的表现。我们推出了VisCodeBench，这是一个端到端的文本到可视化基准数据集，包含来自教育、科学和大众受众三个不同领域的2,672个样本，复杂程度各不相同。我们在涵盖执行、图表内容和视觉质量的七项指标上，对三种可视化系统——Codex、Claude Code和LIDA——搭配一系列开源和商业LLM进行了基准测试。结果表明，虽然对于强大的系统而言，基本执行已接近解决，但没有任何系统能够接近复现参考数据，最佳显示数据相似度仅为0.81，在最难的数据子集上更是降至0.69。我们进一步发现，前沿模型在图表内容（原文截断）方面表现出明显分化。

    arXiv:2611.02353v1 Announce Type: new  Abstract: Generating visualisations from natural language requests (text-to-vis) enables users to explore data more efficiently. While LLMs can complete simple visualisation requests, their performance on challenging ones is not well covered by existing benchmarks. We introduce VisCodeBench, an end-to-end text-to-vis benchmark with 2,672 samples from three different domains: educational, scientific, and general audience scenarios with varying levels of complexity. We benchmark three visualisation systems---Codex, Claude Code, and LIDA---paired with a range of open and commercial LLMs across seven metrics covering execution, chart content, and visual quality. While basic execution is near-solved for strong systems, no system comes close to reproducing the reference data, with the best displayed data similarity reaching only $0.81$ and falling to $0.69$ on our hardest subset for this metric. We further find that frontier models separate on chart cor
    
[^83]: 模型惧怕哪些词：从拒绝方向到覆盖全词表的恐惧词典

    Which Words a Model Fears: From a Refusal Direction to a Vocabulary-Wide Fear Lexicon

    [https://arxiv.org/abs/2611.02348](https://arxiv.org/abs/2611.02348)

    该研究发现语言模型的拒绝行为部分由词语本身驱动，并提出用“拒绝方向”作为标尺为整个词表构建“恐惧词典”，证明仅向良性提示添加一个恐惧词就足以触发模型拒绝。

    

    对齐的语言模型是否拒绝某个请求，部分取决于提示中包含了哪些词，而与这些词是否承载请求的实际含义无关。有害词会使生成边界的隐藏状态沿着一个跨提示、跨词语共享的成分发生偏移，而交叉差分方法能够将其提取为单一方向。在八个模型中，沿该方向进行写入（修改激活）时，拒绝行为会随剂量单调地升降，效果优于随机方向和KL匹配方向。将该方向用作衡量整个词表的标尺，可以为每个词赋予一个偏移量，从一端的“恐惧”词到另一端的“勇气”词；一个词的偏移量可以通过其自身的表示预测出来，并且只需在良性提示后附加一个恐惧词，就足以让模型拒绝该提示。两项留出测试在全部八个模型中验证了这一排序：插入未见过的词后拒绝率的上升与其词表偏移量一致，以及用提示中最具勇气的词替换最恐惧的词（摘要原文在此处被截断）。

    arXiv:2611.02348v1 Announce Type: new  Abstract: Whether an aligned language model refuses depends in part on which words a prompt contains, regardless of whether those words carry the request's meaning. Harm words shift the hidden state at the generation boundary along a component shared across prompts and words, which crossed differencing extracts as a single direction. In eight models, writing along this direction raises and lowers refusal monotonically with dose, ahead of random and KL-matched directions. Used as a ruler over the full vocabulary, the direction gives every word a shift, from "feared" words at one end to "courage" words at the other; a word's shift is predictable from its own representation, and appending a feared word to a benign prompt is enough to make the model refuse it. Two held-out tests validate the ranking in all eight models: refusal after inserting an unseen word rises with its table shift, and replacing a prompt's most feared word by its most courageous m
    
[^84]: MemShift：在LLM衍生的个人记忆中分离检测与修复

    MemShift: Separating Detection from Repair in LLM-Derived Personal Memory

    [https://arxiv.org/abs/2611.02338](https://arxiv.org/abs/2611.02338)

    该论文提出了MemShift基准测试，通过五种转变类型、分叉式合成世界和确定性评分器，将LLM衍生个人记忆的“检测”（发现输入变化）与“修复”（生成正确新值）两个环节分离评估，从而精确定位记忆维护失败的具体步骤。

    

    AI助手会按照个人自身的定义，存储由语言模型从其记录中推导出的值，例如其重要联系人。保持这类值的正确性需要两个步骤：检测，即注意到输入发生了变化；以及修复，即生成正确的新值。仅对最终答案打分的基准测试无法分辨哪一步失败了，并且它们将每次变化都表述为新文本。鉴于这一局限性，我们提出了MemShift，一个包含五种转变的基准测试：源修订、删除、重定义、书写者切换和导出。前三种转变将一个合成世界分叉为仅在转变上有所不同的分支。黄金值和依赖关系在任何文本生成之前就已固定，并用确定性评分器取代了LLM裁判。对于会报告其失效条目的记忆系统，MemShift能够对每个变化的项目显示该记忆是否注意到了变化以及其修复是否正确。此外，我们还构建了一个具有已知检测和修复行为的参考记忆系统（摘要原文在此处截断）。

    arXiv:2611.02338v1 Announce Type: new  Abstract: AI assistants store values that a language model derived from a person's records under the person's own definitions, such as their important contacts. Keeping such a value correct takes two steps: detection, noticing that an input changed, and repair, producing the correct new value. Benchmarks that score only the final answer cannot tell which step failed, and they state every change as new text. Seeing this limitation, we present MemShift, a benchmark of five shifts: source revision, deletion, redefinition, writer switch, and export. The first three fork a synthetic world into branches that differ only in the shift. Gold values and dependencies are fixed before any text, and a deterministic scorer replaces an LLM judge. For a memory that reports its invalidations, MemShift shows for each changed item whether the memory noticed the change and whether its repair was right. Moreover, we build a reference memory with known detection and re
    
[^85]: 进一步训练模型与其基础模型在挪威法律法规数值上的过时情况：Borealis 2 与 Gemma 4 的预注册配对比较

    Outdated numeric values from Norwegian statutes and regulations in a further-trained model and its base model: a preregistered paired comparison of Borealis 2 and Gemma 4

    [https://arxiv.org/abs/2611.02334](https://arxiv.org/abs/2611.02334)

    一项预注册的配对比较研究表明，经挪威国家图书馆进一步训练的 Borealis 2 与其基础模型 Gemma 4 在回答挪威法律法规数值时，给出过时信息的比例并无显著差异。

    

    我们比较了两个已发布的语言模型给出挪威法律法规过时数值的频率：一个是挪威国家图书馆基于 google/gemma-4-26B-A4B-it（Gemma 4）进一步训练的 NbAiLab/borealis2-26b-a4b-preview（Borealis 2），另一个是 Gemma 4 本身。该研究已预先注册（doi:10.5281/zenodo.23858965），使用封闭的 norregel 基准测试，并根据数值历史进行确定性评分。在 503 个主要条目中，两个模型都给出可解释答案的 411 个条目里，仅 Borealis 2 给出过时数值的有 10 个条目，仅 Gemma 4 给出过时数值的有 11 个条目（d = -0.2 个百分点，95% 聚类自助区间 [-2.7, +2.5]，精确单侧 McNemar 检验 p = 0.668）。根据预注册的判定规则，这不构成感兴趣的效应：整个置信区间落在 +/-10 个百分点的最小感兴趣效应范围之内。在次要测量指标中，只有给出在问题日期之后才生效的数值的答案比例存在差异（原文摘要在此处截断）。

    arXiv:2611.02334v2 Announce Type: new  Abstract: We compare how often two released language models give outdated numeric values from Norwegian statutes and regulations: NbAiLab/borealis2-26b-a4b-preview (Borealis 2), which the National Library of Norway further trained from google/gemma-4-26B-A4B-it (Gemma 4), and Gemma 4 itself. The study was preregistered (doi:10.5281/zenodo.23858965) and uses the closed norregel benchmark, scored deterministically against a value history. On the 411 of 503 primary items where both models gave an interpretable answer, Borealis 2 alone was outdated on 10 items and Gemma 4 alone on 11 (d = -0.2 percentage points, 95% cluster bootstrap interval [-2.7, +2.5], exact one-sided McNemar p = 0.668). By the preregistered rule this is no effect of interest: the whole interval lies within the smallest effect of interest of +/-10 percentage points. Among the secondary measures, only the share of answers giving a value that took effect after the question date diff
    
[^86]: 自适应停机何时胜出？选择规则在循环语言模型中至关重要

    When Does Adaptive Halting Win? Selection Rules Matter in Looped Language Models

    [https://arxiv.org/abs/2611.02333](https://arxiv.org/abs/2611.02333)

    该论文通过预注册实验证明，在比较循环语言模型的自适应停机方法时，选择工作点的规则决定了比较结果，且在经自助法筛选的 1 个百分点准确率损失规则下，训练的稳定性探针比原生门控节省了更多计算预算。

    

    循环语言模型可以提前停止迭代，最近的研究表明，对循环轨迹进行事后读取往往能与循环模型 Ouro 学习到的退出门控相媲美。此类比较将每种方法简化为由某个选择规则选出的单一工作点。我们追问：究竟是不是这一规则决定了比较结果。在一个冻结的、循环四次的 Ouro-1.4B 模型上，我们在预注册研究中将训练好的稳定性探针（$\pi_1$-x）与 Ouro 的原生门控（经调整用于逐题停止决策）进行了比较；验证性研究是在开发数据上得出探索性发现之后设计的，并仅在一个包含 8,517 个问题的密封测试集上评估一次。在预注册的验证性比较中，工作点通过经自助法（bootstrap）筛选的 1 个百分点准确率损失规则来选取，结果显示探针比原生门控节省了更多的四次循环预算（在三个基准上总宏观节省为 +0.053，95% 置信区间 [+0.049, +0.057]），其中节省主要来自 CommonsenseQA。

    arXiv:2611.02333v1 Announce Type: cross  Abstract: Looped language models can stop iterating early, and recent work shows that post-hoc readouts of the loop trajectory often rival the learned exit gate of the looped model Ouro. Such comparisons reduce each method to one operating point chosen by a selection rule. We ask whether that rule decides the outcome. On a frozen Ouro-1.4B with four loops, we compared a trained stability probe ($\pi_1$-x) with Ouro's native gate, adapted to a per-question stopping decision, in pre-registered studies; the confirmatory study was designed after an exploratory finding on development data and evaluated once on a sealed test set of 8,517 questions. In the pre-registered confirmatory comparison, which selected operating points with a bootstrap-screened 1 pp accuracy-loss rule, the probe saved more of the four-loop budget than the native gate (+0.053 gross macro saving over three benchmarks, 95% CI [+0.049, +0.057]), mostly from CommonsenseQA, where the
    
[^87]: LeanGuard：面向工具使用型智能体的 Lean 验证防护栏

    LEANGUARD: Lean-Verified Guardrails for Tool-Using Agents

    [https://arxiv.org/abs/2611.02332](https://arxiv.org/abs/2611.02332)

    LeanGuard 是一个基于 Lean4 形式化验证的工具调用运行时防护栏，能在违规操作执行前进行拦截并通过反馈或证明引导修复予以纠正，实验证明它可将违规调用降为零且不显著影响任务成功率。

    

    使用工具的语言模型智能体经常会执行其领域政策所禁止的操作。我们提出了 LeanGuard，这是一个运行时系统，在每次工具调用到达环境之前，会根据用 Lean4 编写的策略对该调用进行检查。策略将状态谓词与作用于执行历史的限定范围过去时间时序算子相结合，评估采用“故障即关闭”（fail-closed）方式，且决策过程已被证明与 LeanLTL 语义等价。被拒绝的调用可以通过两种方式修正：通过反馈方式，智能体会收到被违反的规则及提示；通过证明引导的修复方式，Lean 元编程在无需模型参与的情况下，搜索一个能保留用户意图的邻近调用，并证明该调用符合策略。在 Tau-Bench 上使用五个 Qwen 模型（0.8B-27B）进行测试，无防护的智能体至少执行了 33 个被 Lean 策略禁止的调用，而 LeanGuard 则一个都没有执行；启用反馈时，任务成功率没有显著变化。在 DTap 上，LeanGuard 降低了（摘要在此处截断）

    arXiv:2611.02332v1 Announce Type: cross  Abstract: Tool-using language-model agents often execute actions that their domain policies forbid. We present LeanGuard, a runtime that checks every proposed tool call against a policy written in Lean4 before the call reaches the environment. Policies combine state predicates with scoped past-time temporal operators over the execution history, evaluation is fail-closed, and the decision procedure is proved equivalent to a LeanLTL semantics. A denied call can be corrected in two ways. With feedback, the agent receives the violated rule and a hint; with proof-guided repair, Lean metaprogramming searches for a nearby call that preserves the user's intent and proves that the policy admits it, without the model. On Tau-Bench with five Qwen models (0.8B-27B), unguarded agents execute at least 33 calls that the Lean policy forbids, while LeanGuard executes none; with feedback, task success does not change significantly. On DTap, LeanGuard reduces dire
    
[^88]: 被访问却从未披露：医疗人工智能代理对受保护健康信息的静默访问

    Accessed, never disclosed: Silent access to protected health information by medical artificial intelligence agents

    [https://arxiv.org/abs/2611.02319](https://arxiv.org/abs/2611.02319)

    该研究通过在合成病历中嵌入金丝雀标记，首次揭示了医疗AI代理会在任务不需要时“静默访问”患者受保护健康信息——这种行为对仅检查输出的评估完全不可见——并证明基于任务必要性的预执行门控可几乎完全消除这种隐蔽数据访问。

    

    医疗人工智能代理应当只在任务需要时才检索患者标识信息，这是《健康保险可携性与责任法案》（HIPAA）“最小必要”标准背后针对受保护健康信息（PHI）的数据最小化原则。代理可以在不透露标识信息的情况下检索这些信息，而这种“静默访问”对于仅检查输出的评估来说是不可见的。我们在合成患者记录中嵌入了金丝雀标记，并在五个语言模型、四个边界级别和七种攻击框架下，针对无需标识符的临床任务，将每次工具调用与最终答案中的金丝雀暴露关联起来。三个模型不必要地调用了PHI工具，而Claude Opus 4.8和GPT-5.5则没有这样做。更强的访问边界减少了这种访问，而最严格策略下的残余访问仅限于用户推翻策略或声称已获批准的攻击框架。基于任务必要性标准设置的预执行门控几乎消除了这种静默访问。

    arXiv:2611.02319v1 Announce Type: cross  Abstract: Medical artificial intelligence agents should retrieve patient identifiers only when a task requires them, the data-minimization principle behind the HIPAA minimum necessary standard for protected health information (PHI). An agent can retrieve identifiers without revealing them. Such silent access remains invisible to evaluations that inspect only the output. We embedded canaries in synthetic patient records and linked each tool invocation to canary exposure in the final answers across five language models, four boundary levels, and seven attack frames for identifier-free clinical tasks. Three models invoked the PHI tools unnecessarily, whereas Claude Opus 4.8 and GPT-5.5 did not. Stronger boundaries reduced this access, and the residual access under the strictest policy was confined to frames in which the user overrode the policy or asserted approval. A pre-execution gate given the task-necessity criterion nearly eliminated this acce
    
[^89]: 屈服方向随规模涌现并在验证强的地方承担因果作用：八个模型、两个模型家族与五种测量隐患

    A Capitulation Direction Emerges With Scale and Is Causally Load-Bearing Where Validation Is Strong: Eight Models, Two Families, and Five Measurement Hazards

    [https://arxiv.org/abs/2611.02318](https://arxiv.org/abs/2611.02318)

    该研究在两个家族八个指令微调模型上通过预注册的严格验证协议发现，模型在用户反驳下放弃正确答案的谄媚行为对应残差流中的一个线性“屈服方向”，该方向仅在7B及以上规模的模型中涌现并通过验证，消融该方向可因果性地将Qwen2.5-7B推翻正确答案的比率降低约四分之一。

    

    语言模型中的谄媚行为作为一种现象已被充分记录，但作为一种机制却定位不清。我们探究当用户反驳时模型放弃正确答案（即“屈服”）是否对应残差流中的一个线性方向，以及这样的方向是否随规模涌现。在两个模型家族共八个参数量从0.5B到14B的指令微调模型上，我们采用预注册协议（包括留一问题交叉验证、问题内置换零假设和阳性对照）进行研究，发现7B以下不存在可用方向，而7B、8B和14B模型中存在通过验证门槛的方向。在Qwen2.5-7B中消融该方向，可使模型推翻正确答案的比率降低4.8个百分点（相对约四分之一），而对未通过门槛的方向和勉强通过的方向施加相同干预则不产生任何效果。探针的质量而非统计显……（摘要原文至此截断）

    arXiv:2611.02318v1 Announce Type: new  Abstract: Sycophancy in language models is well documented as a behaviour but poorly localised as a mechanism. We ask whether capitulation under user pushback, the abandonment of a correct answer when a user objects, corresponds to a linear direction in the residual stream, and whether such a direction emerges with scale. Across eight instruction-tuned models spanning 0.5B to 14B parameters in two families, under a pre-registered protocol with leave-one-question-out cross-validation, within-question permutation nulls, and a positive control, we find no usable direction below 7B and a direction that clears our validation gate at 7B, 8B and 14B. Ablating that direction in Qwen2.5-7B reduces the rate at which the model reverses a correct answer by 4.8 percentage points, roughly a quarter in relative terms, while the identical intervention on a gate-failing direction and on a marginally passing one produces no effect. Probe quality, not statistical si
    
[^90]: GRIPS：SQL生成中SQL前置接地信息的实证研究

    GRIPS: An Empirical Study of Pre-SQL Grounding Information for SQL Generation

    [https://arxiv.org/abs/2611.02314](https://arxiv.org/abs/2611.02314)

    该论文提出GRIPS三阶段方法，通过在SQL生成前显式提取结构化的接地信息，在两个基准上20个设置中的18个实现平均11.8个百分点的准确率提升，同时将token使用量降低47.8%。

    

    文本到SQL（Text-to-SQL）系统针对结构化数据回答自然语言问题，但在SQL生成之前就可能失败，例如将问题中的提及链接到错误的表、列、值或关系，此时即使生成的SQL语法有效，也可能返回错误的答案。这引出了一个核心问题：什么样的SQL前置信息对SQL生成器有帮助，而额外信息何时不再有帮助？我们在TempTabQA-C和BIRD两个基准上研究了这一问题。在审计的SQL Static失败案例中，接地（grounding）错误在两个基准上都是最大的类别。我们提出了GRIPS（SQL生成的接地表示与输入处理），这是一种三阶段方法，通过导出结构化的SQL前置信息，使这些决策在查询生成之前变得明确。在GPT-5-nano和Llama-70B的实验中，GRIPS-Adaptive在20个设置中的18个上超越了SQL Static，平均准确率提升11.8个百分点，同时将模型调用的token使用量降低了47.8%。

    arXiv:2611.02314v1 Announce Type: new  Abstract: Text-to-SQL systems answer natural language questions over structured data, but can fail before SQL generation by linking question mentions to the wrong table, column, value, or relation. Even valid SQL can then return an incorrect answer. This raises a central question: What pre-SQL information helps a SQL generator, and when does additional information cease to help? We investigate this question on TempTabQA-C and BIRD. Among the audited SQL Static failures, grounding errors form the largest category on both benchmarks. We introduce GRIPS (Grounding Representation and Input Processing for SQL generation), a three-stage method that derives structured pre-SQL information to make these decisions explicit before query generation. Across GPT-5-nano and Llama-70B, GRIPS-Adaptive improves over SQL Static in 18 of 20 settings, with an average accuracy gain of 11.8 percentage points, while reducing model-call token usage by 47.8%. Our controlle
    
[^91]: Concise-9B：消除推理模型思维过程中的冗余

    Concise-9B: Removing the Slack from a Reasoning Model's Thinking

    [https://arxiv.org/abs/2611.02278](https://arxiv.org/abs/2611.02278)

    提出Concise-9B模型，通过简洁自训练方法（两轮策略上拒绝采样微调并保留最短正确样本）去除推理模型思维链中的冗余部分，在保持准确率等效的前提下将生成token量降至原模型的0.66倍。

    

    推理模型在思考更久时回答效果更好，但思维链在答案已经形成之后仍会继续运行。我们测量了Qwen3.5-9B中的这种冗余，并通过简洁自训练（concise self-training）将其大部分消除：即两轮策略上拒绝采样微调，保留最短的正确样本，再加上一小部分答案合成目标，从而得到Concise-9B。在8,983个保留测试问题上（竞赛及困难数学、SuperGPQA、一个五领域确认集以及MMLU-Pro），Concise-9B在32K思考预算下生成的token数量仅为Qwen3.5-9B的0.66倍，而准确率等效（差异在±1分以内，即-0.20 [-0.77, +0.36]分）。通过让困难集和确认集上的每条思维链每2,048个思考token强制作答一次，我们发现Qwen3.5-9B有54-64%的思考token出现在“从该点起所有强制答案均等于最终答案”的时刻之后（Concise-9B为41-48%）。在该点停止思考将使每个答案的token量保持在基础模型的0.35-0.44倍。

    arXiv:2611.02278v1 Announce Type: cross  Abstract: Reasoning models answer better when they think longer, but a thinking trace keeps going well after its answer has formed. We measure this slack in Qwen3.5-9B and remove most of it with concise self-training: two rounds of on-policy rejection-sampling fine-tuning that keep the shortest correct samples, plus a small share of answer-synthesis targets, giving Concise-9B. On 8,983 held-out problems (competition and hard mathematics, SuperGPQA, a five-area confirmation set and MMLU-Pro), Concise-9B generates 0.66x the tokens of Qwen3.5-9B at a 32K thinking budget, with accuracy equivalent within +/-1 point (-0.20 [-0.77, +0.36] points). Forcing each trace on the hard and confirmation sets to answer every 2,048 thinking tokens, we find that 54-64% of Qwen3.5-9B's thinking tokens come after the point from which every forced answer equals the final one (41-48% for Concise-9B). Stopping there would keep every answer at 0.35-0.44x the base model'
    
[^92]: 当文本沉默时：证据不足情况下大语言模型标注的记忆捷径与有效性

    When the Text Is Silent: Memory Shortcuts and the Validity of LLM Annotation under Evidence Insufficiency

    [https://arxiv.org/abs/2611.02273](https://arxiv.org/abs/2611.02273)

    该研究构建了分离文本证据与先验知识的反事实文本，发现大语言模型在文本与记忆冲突时不会用记忆覆盖文本，但当文本证据缺失时会依赖对知名人物的先验记忆产生“记忆捷径”，从而影响LLM标注的有效性。

    

    大语言模型越来越多地被用于为社会科学研究、内容审核和训练数据标注文本。它们对现实世界实体的广泛知识引发了一个担忧：在标注文本时，它们是依赖文本中的证据，还是用已知知识替代？我们将这种行为称为“记忆捷径”，并研究它何时出现、什么因素会放大它、以及如何缓解它。我们使用995位广为人知的公众人物以及每位人物对应的四个匹配对照名称，构建了将文本证据与先验知识分离的反事实文本，并在五个阶段中对三个商业大语言模型各进行了121,594次提示评估。我们发现了一种显著的不对称性：尽管所有模型几乎都能识别每一位人物，但当文本与记忆冲突时，没有任何模型会遵循记忆而非文本。相反，捷径是在证据缺失时出现的：对于其中一个模型，著名名字使“记忆出的”职业、年龄和国家的比例提高了17.0、14.9和

    arXiv:2611.02273v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to annotate text for social science research, content moderation, and training data. Their broad knowledge of real-world entities raises a concern: when labeling a text, do they rely on its evidence or substitute what they already know? We call this behavior a memory shortcut and examine when it emerges, what amplifies it, and how to mitigate it. Using 995 widely recognized public figures and four matched control names per figure, we construct counterfactual texts that separate textual evidence from prior knowledge, and evaluate three commercial LLMs on 121,594 prompts each across five stages. We find a striking asymmetry. Although all models recognize nearly every figure, none follows memory over the text when the two conflict. Shortcuts instead emerge when evidence is absent: for one model, famous names raise the rate of remembered occupations, ages, and countries by 17.0, 14.9, and 
    
[^93]: 大语言模型在阿拉伯语方言机器翻译中的现状

    Current state of LLMs for Arabic dialectal machine translation

    [https://arxiv.org/abs/2611.02271](https://arxiv.org/abs/2611.02271)

    该研究在MADAR数据集上系统评估了包括阿拉伯语专用模型、多语言模型和商业API在内的多种大语言模型在英语与16种阿拉伯语方言之间双向机器翻译的表现，并探讨了方言机器翻译自动评估的挑战以及LLM的方言识别能力。

    

    本工作对大型语言模型（LLM）在MADAR数据集上从英语到阿拉伯语方言的机器翻译能力进行了评估。我们在16种阿拉伯语方言上对两个翻译方向（英译阿及阿译英）进行了评估。实验涵盖了多样化的模型集合，包括专门的阿拉伯语模型（Jais、Nile）、多语言模型（Gemma、Command-R、Mistral、Aya）以及商业API（GPT-4.1）。我们采用多种评估指标：BLEU、CHRF、COMET（基于参考和无需参考的两种变体）以及GEMBA（以LLM作为评判者），并辅以小规模人工评估来衡量翻译质量。我们讨论了自动机器翻译评估面临的挑战，尤其是在阿拉伯语方言场景下的挑战。我们还评估了LLM对文本所用方言进行分类的能力。该研究深入揭示了当前LLM在阿拉伯语方言机器翻译方面的能力与局限，特别强调了……

    arXiv:2611.02271v1 Announce Type: new  Abstract: This work presents an evaluation of large language models (LLMs) for English to dialectal Arabic machine translation on the MADAR dataset. We evaluate both translation directions (English to Arabic and vice-versa) on 16 Arabic dialects. Our experiments cover a diverse set of models, including specialized Arabic models (Jais, Nile), multilingual models (Gemma, Command-R, Mistral, Aya), and commercial APIs (GPT-4.1). We employ multiple evaluation metrics: BLEU, CHRF, COMET (both reference-based and reference-less variants) and GEMBA (LLM-as-a-judge), as well as a small-scale manual evaluation, to assess translation quality. We discuss the challenges of automatic MT evaluation, especially in the context of Arabic dialects. We also evaluate the ability of LLMs to classify the dialect used in a text. The study offers insights into the capabilities and limitations of current LLMs for dialectal Arabic machine translation, particularly highlight
    
[^94]: 提示式缩放定律

    Hinted Scaling Laws

    [https://arxiv.org/abs/2611.02270](https://arxiv.org/abs/2611.02270)

    提示式缩放定律利用特权信息将基准测试转化为难度连续的简化版本，让小模型也能获得非零分数，从而在噪声阈值以下实现平滑可预测的缩放外推，将 $10^{23}$ FLOP 大模型 GSM8K 准确率的预测误差控制在 3 个百分点以内，远优于标准缩放定律的 13 个百分点误差。

    

    缩放定律通过从更小、更便宜的训练运行中外推趋势，来预测大型、昂贵训练运行的结果。在下游准确率评估中，当较便宜运行所产生的较弱模型的准确率接近零时，这些预测便会失效，因为这些分数无法提供可外推到更大规模的趋势。我们提出了提示式缩放定律，它利用特权信息将一个基准转换为难度连续递减的一系列简化版本，使得即使是小模型也能取得高于零的分数。我们发现准确率随提示水平平滑且可预测地上升，从而催生了能够在噪声阈值以下运作的新型缩放定律。在 Delphi 预训练阶梯上，用训练计算量至多 $10^{20}$ FLOPs 的模型拟合的提示式缩放定律，对一个 $10^{23}$ FLOP 模型的 GSM8K 准确率（25.9%）的预测误差在 3 个百分点以内；而在相同模型上拟合的标准缩放定律则低估了 13 个百分点（原文在此处截断）。

    arXiv:2611.02270v1 Announce Type: cross  Abstract: Scaling laws predict the outcome of large, expensive training runs by extrapolating trends from smaller, cheaper runs. On downstream accuracy evaluations, these predictions fail when the weaker models produced by the cheaper runs score near zero accuracy, because these scores give no trend to extrapolate to larger scales. We introduce hinted scaling laws, which use privileged information to turn a benchmark into a continuum of easier versions so that even the small models score above zero. We find that accuracy rises smoothly and predictably with the hint level, enabling new types of scaling laws that operate below the noise threshold. On the Delphi pretraining ladder, a hinted scaling law fit on models trained with at most $10^{20}$ FLOPs forecasts the GSM8K accuracy of a $10^{23}$ FLOP model (25.9%) to within 3 percentage points; a standard scaling law under-forecasts it by 13 percentage points when fit on the same models and by 12 e
    
[^95]: 基于变形测试的情感分析模型鲁棒性评估

    Metamorphic Testing for Evaluating Robustness of Sentiment Analysis Models

    [https://arxiv.org/abs/2611.02248](https://arxiv.org/abs/2611.02248)

    本文提出了一种轻量级的、基于批处理的变形测试方法，通过检验原始输入与变换后输入之间预期关系的一致性，来系统性地评估情感分析模型的鲁棒性。

    

    机器学习系统难以使用传统软件测试方法进行测试，因为其行为是从数据中学习而来，而非通过显式编程实现。这一挑战在自然语言处理（NLP）领域尤为显著，因为微小的输入变化就可能导致模型出现意外的行为。传统的评估指标（如准确率和F1分数）往往无法捕捉到这类不一致性。变形测试通过评估原始输入与变换后输入之间的预期关系来解决这一局限性。随着生成式人工智能（GenAI）和基于语言的AI系统日益普及，对系统性鲁棒性测试的需求变得更加迫切。基于这一日益增长的需求，本文提出了一种轻量级的、基于批处理的变形测试方法，用于评估此类系统的鲁棒性。特别地，我们将情感分析作为一个代表性任务。

    arXiv:2611.02248v1 Announce Type: cross  Abstract: Machine learning systems are challenging to test using traditional software testing methods because their behavior is learned from data rather than explicitly programmed. This challenge is particularly significant in natural language processing (NLP), where small input variations can lead to unexpected model behavior. Conventional evaluation metrics, such as accuracy and F1-score, often fail to capture such inconsistencies. Metamorphic testing addresses this limitation by evaluating expected relationships between original and transformed inputs. With the increasing prevalence of generative artificial intelligence (GenAI) and language-based AI systems, the need for systematic robustness testing has become even more pronounced. Motivated by this growing need, this paper presents a lightweight, batch-based metamorphic testing approach for evaluating the robustness of such systems. In particular, we consider sentiment analysis as a represe
    
[^96]: AI审稿人能看到你改进论文的努力吗？关于论文修改轨迹的大规模研究

    Can AI Reviewers See Your Efforts to Improve Your Paper? A Large-Scale Study of Paper Revision Trajectories

    [https://arxiv.org/abs/2611.02244](https://arxiv.org/abs/2611.02244)

    研究发现AI审稿人在论文评审中往往无法识别作者通过修改做出的重大改进，对大多数标注确认的重大修改保持评分不变，这揭示了AI评审系统在评估论文改进方面的严重局限性。

    

    AI审稿人正被越来越多地用于评估科学论文手稿，然而其评判是否能反映作者通过修改所做出的改进，这一问题尚不清楚。我们使用PaperTrajectory-2026数据集研究这一问题——该数据集是PaperTrajectory的2026年版本，涵盖五个AI会议的6,959篇论文轨迹，并对14个维度上的变化进行了自动标注，同时提供了对整体质量的独立评判。标注结果显示，重大改进主要集中在实证证据和论文定位方面，而后期的修改往往趋向于渐进式提升。通过使用八位AI审稿人对论文不同版本进行独立评估，我们检验了以下问题：被标注为重大改进的部分是否能获得更高分数、哪些修改维度与分数变化相关，以及用户意见是否能够超越观察到的修改收益。研究发现，每位AI审稿人在大多数重大改进上都保持分数不变，并且（原文此处截断）

    arXiv:2611.02244v1 Announce Type: new  Abstract: AI reviewers are increasingly used to assess scientific manuscripts, yet it remains unclear whether their judgments reflect the improvements authors make through revision. We investigate this question using PaperTrajectory-2026, the 2026 release of PaperTrajectory, a dataset of 6,959 paper trajectories across five AI conferences, with automatic annotations of changes along 14 dimensions and separate judgments of overall quality. The annotations show that major improvements emphasize empirical evidence and framing, while later revisions tend toward incremental improvement. Using independent assessments of manuscript versions from eight AI reviewers, we examine whether annotated major improvements receive higher scores, which revision dimensions are associated with score changes, and whether user opinions can outweigh observed revision gains. Every reviewer leaves scores unchanged on most major improvements, and agreement on the direction 
    
[^97]: 谁应该为缓存打分？中深度层是KV缓存淘汰的最佳裁判

    Who Should Score the Cache? Mid-Depth Layers Are the Best Judges of KV-Cache Eviction

    [https://arxiv.org/abs/2611.02230](https://arxiv.org/abs/2611.02230)

    该论文发现KV缓存淘汰质量由打分层的深度决定——中深度层（约0.4至0.75深度处）是评判token的最佳层，并提出一种基于答案存活概率的校准方法来自动选出最佳打分层。

    

    键值（KV）缓存淘汰方法通过注意力分数对token进行排序，且传统做法是每一层为自己那一层的缓存打分。据我们所知，尚无淘汰方法探讨过应由哪一层来打分这一问题。我们在二十一个模型上发现，打分层的深度决定了淘汰质量。在十八个具有逐层缓存的模型上（参数量从0.5B到109B），长上下文检索准确率在浅层急剧下降，在深度的0.4到0.75之间达到峰值，而在最后一层则有所退化。这一曲线产生的原因在于：只有当某一层已经利用问题定位到答案（这需要经过若干层的计算）之后，它才能很好地评判token。我们将淘汰建模为top-k截断，从预填充（prefill）注意力中计算答案存活概率，并复现了实测曲线。该方法利用带有已知答案的校准提示来选择打分层，所选层在大多数模型上均为实测最佳层，且从未显著低于最佳层，其表现优于传统的自身打分（own-layer）方法。

    arXiv:2611.02230v1 Announce Type: cross  Abstract: Key-value (KV) cache eviction methods rank tokens by attention scores, and each layer scores its own cache. To our knowledge, no eviction method asks which layer should score. We find, across twenty-one models, that the depth of the scoring layer governs eviction quality. On the eighteen models with per-layer caches, spanning 0.5B to 109B, long-context retrieval accuracy collapses at shallow layers, peaks between 0.4 and 0.75 of depth, and degrades at the last layer. The curve arises because a layer judges tokens well only once it has used the question to find the answer, which takes several layers. Modeling eviction as a top-$k$ cut, we compute answer survival probability from prefill attention and reproduce the measured curves. This approach selects the scoring layer using calibration prompts with known answers, and the selected layer is the best layer measured on most models and never significantly below it. It improves on the own-l
    
[^98]: 小规模模型开放式推理的测试时扩展

    Test-Time Scaling for Open-Ended Reasoning at Small Model Scale

    [https://arxiv.org/abs/2611.02208](https://arxiv.org/abs/2611.02208)

    该研究提出无需裁判模型的语义聚类投票方法，在小规模模型上发现测试时投票能将数学推理准确率提升10至29分，但对开放式自由文本生成几乎没有收益，说明测试时扩展对开放性文本的价值取决于模型能否产生值得聚合的样本。

    

    对模型的样本进行投票只有在样本之间存在分歧时才有帮助。因此，对于开放式生成而言，关键问题不是如何组合样本，而是模型是否能产生值得组合的样本——测试时扩展已经为数学问题回答了这一问题，但尚未为自由形式文本回答。我们在小规模下用语义聚类投票研究该问题，这是一种无需裁判的聚合方法，它按语义对小指令模型的样本进行分组，并返回最大簇的代表样本，全程不使用奖励模型或自我评判。在一个包含三个答案可核查性等级的250条提示基准上，该模式在三个模型家族和两个生成种子中保持一致：投票显著提升了数学对照任务上的准确率（三个模型的增益为+10至+29分），但在开放式文本上最多仅提升+0.30的裁判分。在一次规模更大的200题MATH-500运行中，主要模

    arXiv:2611.02208v1 Announce Type: cross  Abstract: A vote over a model's samples can only help when the samples disagree. For open-ended generation the binding question is therefore not how to combine samples but whether a model produces samples worth combining, a question test-time scaling has answered for math but not for free-form text. We study it at small scale with semantic cluster voting, a judge-free aggregator that groups a small instruct model's samples by meaning and returns the largest cluster's representative, with no reward model or self-judging. On a 250-prompt benchmark with three tiers of answer checkability, the same pattern holds across three model families (Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B) and two generation seeds: voting raises accuracy on the math control, where samples vary but stay checkable (a gain of +10 to +29 points across the three models), yet adds at most +0.30 judge points on open-ended text. On a larger 200-problem MATH-500 run with the primary mod
    
[^99]: 决策模型作为编码代理的策略门控：显式策略边界提升检测能力并减少误报

    Decision Models as Policy Gates for Coding Agents: Explicit Policy Boundaries Improve Detection and Reduce False Alarms

    [https://arxiv.org/abs/2611.02155](https://arxiv.org/abs/2611.02155)

    以示例和反例形式编写的显式策略边界能显著提升决策模型作为编码代理运行时策略门控的风险检测召回率（提升5.5-26.1个百分点），同时大幅降低误报率。

    

    编码代理破坏数据、泄露机密以及向错误对象发送消息的速度可能比人类审查的速度还快。决策模型能否在运行时捕获这些风险？我们评估了22个托管决策模型和两个通过提示词驱动的通用模型，将其作为多规则策略引擎的分类器。一个包含2,643次工具调用的合成数据集针对28条规则进行了盲标注，并保留了密封的留存测试集。在两种条件下使用单一共享前导文本的情况下，以示例和反例形式编写的显式边界使所有23个可用模型的召回率提高了5.5至26.1个百分点，并在21个模型上降低了误报。Jev 1.14的召回率从89.3%提升至98.2%，同时针对252个正常操作的不必要干预比例从30.6%降至4.8%。使用示例时，其中位延迟为120毫秒，成本为每千次请求0.46美元。强模型在具体名称被替换为通用描述后仍保留了召回率收益；当相同边界以……形式编写时，三个模型也基本保留了这些收益。

    arXiv:2611.02155v1 Announce Type: new  Abstract: Coding agents can destroy data, leak secrets and message the wrong person faster than a human can review. Could a decision model catch these risks at runtime? We evaluated 22 hosted decision models and two prompted general models as classifiers for a multi-rule policy engine. A synthetic dataset of 2,643 tool calls was labeled blind against 28 rules, with a sealed holdout. With a single shared preamble in both conditions, explicit boundaries written as examples and counterexamples raised recall by 5.5-26.1 percentage points on all 23 available models and reduced false positives on 21. Jev 1.14's recall rose from 89.3% to 98.2%, while unnecessary interruptions fell from 30.6% to 4.8% of 252 clean actions. With examples, its median latency was 120 ms at $0.46 per thousand requests. Strong models retained the recall gains when concrete names were replaced by generic descriptions; three largely retained them when the same boundaries were wri
    
[^100]: 多智能体合谋与J空间

    Multi-Agent Collusion and the J-Space

    [https://arxiv.org/abs/2611.02153](https://arxiv.org/abs/2611.02153)

    该论文首次将J空间分析应用于多智能体系统，发现合谋智能体的J空间在回复前就已表征合谋行为本身（如虚假陈述、隐瞒或悔意），并据此提出了首个无监督智能体合谋检测方法J-PACT。

    

    arXiv:2611.02153v1 公告类型：新 摘要：大语言模型智能体正越来越多地被部署在多智能体系统（MAS）中，这带来了智能体之间可能逃逸人类监督的协同风险。J空间此前已被用于揭示单个大语言模型中的欺骗行为，但其在揭示和引导智能体之间合谋方面的应用尚未被探索。我们通过对DeepSeek-V4-Flash（284B）在NARCBench合谋基准上进行可解释性实验表明，合谋智能体的J空间中承载着合谋本身：在作出回复之前，86%的合谋者表征了虚假陈述、隐瞒或悔意。在一个隐写式数牌任务中，接收方的J空间持有一个隐藏的计数，且因果干预能够改变该计数。我们还提出了Jacobian对对比法，这是一种用于检测智能体之间合谋的无监督方法。我们在四个开放权重模型上对J-PACT进行了评估：GPT-OSS-20B、Llama-3.3-70B-IT、Qwen3-32B以及DeepSeek-V4-Flash（284B）。J-PACT保持了85%的……

    arXiv:2611.02153v1 Announce Type: new  Abstract: LLM agents are increasingly deployed in Multi-Agent Systems (MAS), and they introduce the risk of coordination among agents that might escape human oversight. The J-space has been used to uncover deception in single LLMs, but its use for uncovering and steering collusion between agents remains unexplored. We show, with interpretability experiments on DeepSeek-V4-Flash (284B) on the NARCBench collusion benchmark, that colluding agents' J-space holds the collusion itself: before replying, 86% of colluders represent misrepresentation, concealment or remorse. In a steganographic card-counting task, the receiver's J-space holds a hidden count, and causal interventions alter its count. We also introduce the Jacobian Pair Contrast (J-PACT), an unsupervised method for detecting collusion between agents. We evaluate J-PACT on four open-weight models: GPT-OSS-20B, Llama-3.3-70B-IT, Qwen3-32B, and DeepSeek-V4-Flash (284B). J-PACT retains 85% of the
    
[^101]: 反向传播算法的详细推导

    Detailed Derivation of The Backpropagation Algorithm

    [https://arxiv.org/abs/2611.02152](https://arxiv.org/abs/2611.02152)

    本文以三层网络为例逐步详细推导了反向传播算法，通过链式法则和路径求和导出矩阵形式的四个通用反向传播方程，并将其扩展到多样本矩阵运算与小批量梯度下降。

    

    反向传播是用于计算训练神经网络所需梯度的标准方法，但它通常以紧凑的向量形式呈现，省略了中间步骤。本教程针对具有Sigmoid激活函数和平方误差损失的全连接前馈网络，逐步推导反向传播算法。通过一个三层网络的连续示例，我们首先利用链式法则沿着权重在网络中的传播路径，对误差关于单个权重求导。随后，我们展示了如何通过对所有路径求和导出误差项δ，以及矩阵形式的四个通用反向传播方程。每个项都标注了其维度，因此可以检查每个方程的维度一致性。该推导随后被扩展到m个训练样本的情形，此时权重梯度成为单个矩阵乘积，而偏置项对应于广播操作。推导还被扩展到小批量梯度下降……

    arXiv:2611.02152v1 Announce Type: new  Abstract: Backpropagation is the standard method for computing the gradients that train neural networks, but it is usually presented in a compact vector form that skips the intermediate steps. This tutorial derives backpropagation step by step for a fully connected feedforward network with sigmoid activations and a squared-error loss. Using a running example with three layers, we first differentiate the error with respect to a single weight by following its path through the network with the chain rule. We then show how summing over all paths leads to the error terms $\delta$ and to the four general backpropagation equations in matrix form. Every term is annotated with its dimensions, so each equation can be checked for consistency. The derivation is then extended to $m$ training examples, where the weight gradients become a single matrix product and the bias term corresponds to broadcasting. It is also extended to mini-batch gradient descent, incl
    
[^102]: GatedV：面向Transformer语言模型的自门控值

    GatedV: Self-Gated Values for Transformer Language Models

    [https://arxiv.org/abs/2611.02145](https://arxiv.org/abs/2611.02145)

    GatedV是一种无需额外可训练参数的简单注意力变体，通过仅对值矩阵V在聚合前应用SiLU激活函数，即可在多个参数规模下稳定地将语言模型困惑度降低约2.0%至2.4%。

    

    标准的注意力机制通过对输入隐藏状态的学习线性投影来形成查询、键和值矩阵。GAU和Mega等架构在注意力中引入了非线性变换。在本文中，我们提出了GatedV，这是一种简单的注意力变体，它仅在聚合之前对值矩阵V应用SiLU激活函数，而不引入额外的可训练参数。我们在匹配参数量、成对初始化和相等训练token预算的条件下，将GatedV与标准的仅解码器Transformer进行比较评估。在2.33M参数规模下，GatedV将平均留出困惑度从59.1降低到57.9（降低2.0%），在全部五个成对随机种子上均取得改进。在26.16M参数规模下，经过等预算学习率搜索后，平均留出困惑度从20.2降低到19.7（降低2.4%），在全部三个成对随机种子上均取得改进。小模型消融实验表明，所测试的对值进行仿射缩放的方法无法完全复现GatedV带来的增益。

    arXiv:2611.02145v2 Announce Type: new  Abstract: Standard attention forms query (Q), key (K), and value (V) matrices through learned linear projections of input hidden states. Architectures such as GAU and Mega incorporate nonlinear transformations into attention. In this paper, we propose GatedV, a simple attention variant that applies SiLU only to V before aggregation without introducing additional trainable parameters. We evaluate GatedV against a standard decoder-only Transformer using matched parameter counts, paired initializations, and equal training-token budgets. At 2.33M parameters, GatedV reduces mean held-out perplexity from 59.1 to 57.9 (2.0%), with improvements across all five paired seeds. At 26.16M parameters, following an equal-budget learning-rate search, mean held-out perplexity falls from 20.2 to 19.7 (2.4%), with improvements across all three paired seeds. Small-model ablations show that the tested affine rescaling of values does not fully reproduce GatedV's gains.
    
[^103]: HaloGuard 2.0：面向多语言响应安全的开放权重宪法式输出守卫

    HaloGuard 2.0: An Open-Weights Constitutional Output Guard for Multilingual Response Safety

    [https://arxiv.org/abs/2611.02134](https://arxiv.org/abs/2611.02134)

    提出开放权重的流式输出安全守卫HaloGuard 2.0，可在单次前向传播中为每个响应token实时判定安全/不安全概率，从源头拦截生成过程中才显现的危害内容，并揭示了传统数据训练会导致守卫只看提示词不看响应的流式捷径问题。

    

    输入守卫决定一个请求能否到达模型，但永远看不到模型实际说了什么。在生成过程中涌现的危害——无论是越狱攻击绕过了输入防线，还是良性请求逐渐演变为可操作的危险细节——只有在输出端、且随着token逐个到达时才能被察觉。我们提出了HaloGuard 2.0，一个开放权重的流式输出守卫，它在单次前向传播中为每个响应token分配安全/不安全概率，使部署方能够在确认危害的瞬间停止输出流。该守卫复用我们输入守卫HaloGuard 1.0的宪法（constitution）来组织包含478,532条响应的训练语料库，并从响应变为有害的那一刻起对每个token进行监督。我们发现，用传统形态数据训练的守卫会学到一种流式场景特有的捷径：它判断的是提示词而非响应本身，在匹配的安全/不安全响应对中仅能正确处理13.9%。我们通过第二阶段（训练）纠正了这一问题。

    arXiv:2611.02134v1 Announce Type: new  Abstract: An input guard decides whether a request reaches the model, but never sees what the model says. Harm that emerges during generation, because a jailbreak passed the input layer or a benign request drifted into operational detail, is visible only on the output side and only as tokens arrive. We present HaloGuard 2.0, an open-weight streaming output guard that assigns a safe/unsafe probability to every response token in a single forward pass, so a deployment can stop the stream once harm is confirmed. The guard reuses the constitution of HaloGuard 1.0, our input guard, to organise a 478,532-response training corpus, and supervises each token from the point where the response becomes harmful. We find that a guard trained on conventionally shaped data learns a shortcut specific to streaming: it judges the prompt instead of the response, and handles only 13.9% of matched safe/unsafe response pairs correctly. We correct this with a second phase
    
[^104]: FOLLIE：面向法语的首个句子及其一阶逻辑表示数据集上的自动推理

    FOLLIE: Automated Reasoning over the First Dataset of Sentences with their First-Order Logical Representation for French

    [https://arxiv.org/abs/2611.02129](https://arxiv.org/abs/2611.02129)

    FOLLIE是首个面向法语NLI的自然语言到一阶逻辑资源数据集，包含近2万条推理记录，通过神经-符号流水线构建并显著提升了逻辑推导覆盖率，实现了对法语句子的一阶逻辑自动推理。

    

    FOLLIE（面向法语语言推理与蕴含的一阶逻辑）将法语句子、组合式一阶逻辑表示和自动化自然语言推理（NLI）连接起来。作为首个面向法语NLI的、从自然语言到一阶逻辑（NL-to-FOL）的资源，它包含来自六个异构NLI数据源的19,983条推理记录和36,987个非空的句子-槽位-公式映射，保留了法语的词汇材料和源问题的结构。其银标注由一个神经-符号流水线生成，其中另有111条被单独标识的LLM辅助添加内容。该数据集的构建结合了形态句法分析、概率条件下的范畴多标注、语法证明搜索、篇章表示以及一阶语义构建。分阶段实验显示推导覆盖率有显著提升；对连词分布、逻辑算子出现率以及辖域交替的分析刻画了所构建资源的特性。

    arXiv:2611.02129v1 Announce Type: new  Abstract: FOLLIE (for First-Order Logic for Language Inference and Entailment for French) connects French sentences, compositional first-order representations and automated natural language inference (NLI). The first French NLI-ready, NL-to-FOL resource, it contains 19,983 inference records and 36,987 non-empty sentence-slot--formula mappings from six heterogeneous NLI sources, retaining French lexical material and source problem structure. Its silver annotations are produced by a neural--symbolic pipeline, with 111 separately identified LLM-assisted additions. The construction combines morphosyntactic analysis, probability-conditioned categorial multitagging, grammatical proof search, discourse representation and first-order semantic construction. Stage-wise experiments show substantial gains in derivation coverage; analyses of conjunction distributions, logical-operator incidence and scope alternation characterise the resulting resource. We then
    
[^105]: 耦合负载下的条件校准契约：面向切换推理的分支感知审计

    Conditional Calibration Contracts under Coupled Load: A Branch-Aware Audit of Switched Inference

    [https://arxiv.org/abs/2611.02121](https://arxiv.org/abs/2611.02121)

    该论文提出 CA-MHD 框架，将条件校准契约视为待验证的规范而非已认证属性，通过负载耦合蒸馏、闭环重放和成本感知施密特路由器在离散事件模拟器中进行分支感知审计，并发现常见的教师轨迹审计无法认证学生模型在切换推理中所选部署支持的校准性。

    

    突发式服务会同时改变队列状态以及每个模型所选定的样本总体。我们研究条件校准契约，将其视为有待验证的规范，而非现有路由器已认证的属性。CA-MHD 在带有硬件服务表的离散事件模拟器中，将负载耦合蒸馏、闭环重放与成本感知的施密特路由器相结合。在40个实验单元（十个训练种子和四个内容-负载耦合级别）中，我们区分了原始教师模型、适配主干上冻结的宽头部以及窄学生模型。常见的教师轨迹审计的中位绝对师生差距为0.06个百分点，但这并不能认证学生模型所选部署支持的校准性。历史结果使用的是核正确性差距，而非已发表的 SmoothECE；我们另外证明了在共同固定尺度下残差泛函的选定分支复合性。所部署的宽分支的序数准确率较低……（摘要在此处截断）

    arXiv:2611.02121v1 Announce Type: cross  Abstract: Bursty serving changes both queue states and the population selected for each model. We study conditional calibration contracts as a specification to be validated, rather than a certified property of an existing router. CA-MHD combines load-coupled distillation, closed-loop replay and a cost-informed Schmitt router in a discrete-event simulator with a hardware service table. Across 40 cells (ten training seeds and four content-load coupling levels), we distinguish the original teacher, frozen wide heads on an adapted trunk, and the narrow student. A common teacher-trajectory audit has median absolute student-teacher deficit 0.06 percentage points, but does not certify the student-selected deployment support. Historical results use a kernel correctness gap, not the published SmoothECE; we separately prove selected-branch composition for a residual functional at a common fixed scale. The deployed wide branch has lower ordinal accuracy th
    
[^106]: 仅顺序同策略蒸馏

    Order-Only On-Policy Distillation

    [https://arxiv.org/abs/2611.02112](https://arxiv.org/abs/2611.02112)

    提出仅顺序同策略蒸馏（O³PD），只让学生模型学习教师模型 top-K 词元的排序而不拟合概率值，因此对教师温度不变、无需温度调节，并在多个教师-学生组合上优于传统同策略蒸馏。

    

    同策略蒸馏通过匹配教师模型的词元概率，在学生模型自身生成的文本上对其进行训练。这些概率纠缠了两种信息：教师模型偏好哪些词元，以及它对这些词元的置信程度。后者反映了教师模型自身的能力与校准水平，而较小的学生模型可能无法复现。由于教师温度会改变这些概率值但不改变其排序，OPD 对温度的选择也十分敏感。我们探究教师模型对候选词元的排序是否已经足够，并提出了仅顺序同策略蒸馏。在学生模型生成的每个前缀处，O³PD 训练学生模型复现教师模型 top-K 词元的排序，而无需拟合其概率值。因此该目标对教师温度具有不变性，无需温度调节。在六个教师-学生模型组合上，O³PD 在 HMM 平均表现上优于 OPD。

    arXiv:2611.02112v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains a student on its own generations by matching the teacher's token probabilities. These probabilities entangle two kinds of information: which tokens the teacher prefers, and how confident it is. The latter reflects the teacher's own capacity and calibration, which a smaller student may be unable to reproduce. Because the teacher temperature changes these values without changing the ordering, OPD is also sensitive to the temperature choice. We ask whether the teacher's ordering of candidate tokens is enough, and propose Order-Only On-Policy Distillation (O$^{3}$PD). At each student-generated prefix, O$^{3}$PD trains the student to reproduce the order of the teacher's top-$K$ tokens, without fitting their probability values. The objective is therefore invariant to the teacher temperature and needs no temperature tuning. Across six teacher--student pairs, O$^{3}$PD outperforms OPD on the average over HMM
    
[^107]: 通过测试时训练整合推理

    Consolidating Reasoning with Test-Time Training

    [https://arxiv.org/abs/2611.02104](https://arxiv.org/abs/2611.02104)

    该论文提出TTT-Consolidate框架，利用测试时训练将多智能体协作中产生的推理轨迹整合为参数化表示，从而改进模型后续的推理能力。

    

    有效的多智能体协作需要从分布在多个智能体各自尝试中的推理过程里最终确定解决方案。然而，当协作仅限于对各个智能体得出的结论进行投票，或者在上下文中聚合冗长且充满噪声的推理轨迹集合时，有用的中间推理可能无法被充分利用。在这项工作中，我们提出了TTT-Consolidate，一个将多智能体推理轨迹作为适应数据来改进模型后续推理的框架。TTT-Consolidate通过测试时训练将轨迹联合整合为参数化表示，然后基于整合后的信息生成新的解决方案。我们在训练阶段使用双层优化来元学习如何进行整合，其中内循环将推理轨迹编码到轻量级的LoRA适配器中，外循环则学习内循环的元参数……

    arXiv:2611.02104v1 Announce Type: new  Abstract: Effective multi-agent collaboration requires finalizing solutions from reasoning that is distributed across the individual attempts of many agents. However, useful intermediate reasoning can remain unexploited when collaboration is limited to voting on conclusions reached among individual agents, or aggregating long, noisy collections of reasoning trajectories in context. In this work, we propose TTT-Consolidate, a framework that uses multi-agent reasoning trajectories as adaptation data to improve a model's subsequent reasoning. TTT-Consolidate jointly consolidates trajectories into parametric representations through test-time training, then generates a new solution by conditioning on the consolidated information. We meta-learn how to consolidate using bi-level optimization at training time, where an inner loop encodes reasoning trajectories into a lightweight LoRA adapter, and an outer loop learns meta-parameters for the inner loop suc
    
[^108]: ParetoSaddler：面向质量与成本的智能体框架帕累托引导优化

    ParetoSaddler: Pareto-Guided Optimization of Agent Harnesses for Quality and Cost

    [https://arxiv.org/abs/2611.02079](https://arxiv.org/abs/2611.02079)

    提出ParetoSaddler方法，通过将中间候选晋升与最终部署选择分离，并借助贝叶斯自助帕累托支持机制，在无需修改模型权重的情况下实现LLM智能体框架的质量与成本联合优化。

    

    智能体框架优化无需修改模型权重即可显著提升LLM智能体的表现，但现有方法主要优化响应质量。将其扩展到质量与成本的联合优化会带来一个挑战：在中间更新阶段应用部署偏好可能会过早地拒绝那些对后续改进有价值的候选方案。我们提出ParetoSaddler，通过将中间候选方案的晋升与最终部署选择相分离来解决这一不匹配问题。在每次迭代中，优化器更新一个父框架以产生候选方案，随后通过多次重复执行对二者进行比较，并利用贝叶斯自助帕累托支持来晋升那些在质量或成本上展现出可信进展的候选方案。被晋升的候选方案将接受更广泛的开发集评估，并保持可用状态以作为后续优化步骤的父框架。我们在ClawMark和GAIA2上采用领域不相交的数据划分对ParetoSaddler进行了评估。

    arXiv:2611.02079v1 Announce Type: cross  Abstract: Harness optimization can substantially improve LLM agents without modifying model weights, but existing methods primarily optimize response quality. Extending them to joint quality--cost optimization introduces a challenge: applying the deployment preference during intermediate updates can prematurely reject candidates that are valuable for later improvements. We introduce ParetoSaddler, which addresses this mismatch by separating intermediate candidate promotion from final deployment selection. At each iteration, an optimizer updates a parent harness to produce a candidate, and then compares the two over repeated executions and uses Bayesian bootstrap Pareto support to promote candidates showing credible progress in either quality or cost. Promoted candidates receive broader development evaluation and remain available as parents for later optimization steps. We evaluate ParetoSaddler on ClawMark and GAIA2 using domain-disjoint splits.
    
[^109]: 你的宣言有多“绿色”？——利用大语言模型小组为2024年欧盟选举中的环境立场评分

    How Green Is Your Manifesto? Scoring Environmental Positions in the 2024 EU Elections with an LLM Panel

    [https://arxiv.org/abs/2611.02033](https://arxiv.org/abs/2611.02033)

    该论文提出用三个来自不同提供商的大语言模型组成评审小组，对2024年欧洲议会选举中41份政党宣言逐句在环境保护与气候变化两个维度上自动评分，从而提供一种比专家调查更廉价、可复现的政党环境立场测量方法。

    

    在环境与气候政策上对政党进行定位是政党竞争研究的核心，然而现有的测量工具——专家调查和对宣言的人工编码——成本高昂、具有周期性，且难以快速扩展到新文档。我们探究一组大语言模型能否作为更廉价、可复现的替代方案。我们将每份宣言视为句子的集合，来自三个独立提供商的模型在两个维度——环境保护和气候变化——上对每个句子进行评分，从而将政党在某一议题上的立场与其赋予该议题的显著性区分开来，并将各模型的评判结果平均到宣言层面。模型间的一致性被保留下来作为内置的可靠性信号。我们将该模型小组应用于2024年欧洲议会选举中的41份宣言，涵盖奥地利、法国、德国、意大利、西班牙以及欧洲各政党家族，并以2024年教堂山专家调查为基准进行验证。

    arXiv:2611.02033v1 Announce Type: cross  Abstract: Locating political parties on environmental and climate policy is central to the study of party competition, yet the established instruments, expert surveys and the hand coding of manifestos, are costly, periodic, and slow to extend to new documents. We ask whether a panel of large language models can serve as a cheaper and reproducible alternative. Treating each manifesto as a set of sentences, three models from independent providers score every sentence on two dimensions, environmental protection and climate change, separating a party's position on an issue from the salience it gives it, and the judgments are averaged to the manifesto level. Agreement across the models is retained as a built-in reliability signal. We apply the panel to 41 manifestos from the 2024 European Parliament election, covering Austria, France, Germany, Italy, Spain, and the European party families, and validate it against the 2024 Chapel Hill Expert Survey fo
    
[^110]: ClickSight：一种通过学习策略解读学生点击流数据的大语言模型框架

    ClickSight: An LLM Framework for Interpreting Student Clickstreams Through Learning Strategies

    [https://arxiv.org/abs/2611.02021](https://arxiv.org/abs/2611.02021)

    ClickSight是一个基于大语言模型的分阶段框架，能够依据文献中定义的学习策略自动解读学生点击流数据，并生成高质量、有证据支撑的学习行为解读。

    

    数字学习环境会产生丰富的点击流数据，这些数据能够揭示学生如何开展学习任务，然而解读这些细粒度的行为轨迹通常需要大量的专家投入。我们提出了ClickSight，一个基于文献中定义的学习策略来解读学生点击流的框架。给定学生的点击流数据和一组预定义的学习策略，ClickSight使用分阶段的大语言模型流水线来识别与策略相关的证据、评估学生对每种策略的遵循程度，并生成基于证据的学生行为解读。我们在两个学习环境、365名学生和12种模型骨干上，采用基于专家的质量评估标准以及经过人类专家评分校准的LLM评审方法对ClickSight进行了评估。结果表明，ClickSight能够在不同模型和环境中生成高质量的基于策略的解读，其中推理模型表现更为出色。

    arXiv:2611.02021v1 Announce Type: cross  Abstract: Digital learning environments produce rich clickstream data that can reveal how students approach learning tasks, yet interpreting these fine-grained traces often requires substantial expert effort. We introduce ClickSight, a framework for interpreting student clickstreams through learning strategies defined in the literature. Given a student clickstream and a predefined set of learning strategies, ClickSight uses a staged LLM-based pipeline to identify strategy-relevant evidence, assess adherence to each strategy, and generate an evidence-grounded interpretation of the student's behavior. We evaluate ClickSight across two learning environments, 365 students, and 12 model backbones using an expert-grounded quality rubric and an LLM judge calibrated with human-expert ratings. Our results show that ClickSight can generate high-quality strategy-based interpretations across different models and environments, with reasoning models performin
    
[^111]: CULTURE-CMT：情境化社交媒体翻译基准测试

    CULTURE-CMT: Benchmarking Contextualized Social Media Translation

    [https://arxiv.org/abs/2611.02018](https://arxiv.org/abs/2611.02018)

    提出了CULTURE-CMT，一个包含572个人工验证示例的中英文社交媒体评论情境化翻译基准，通过恢复帖子与对话上下文并配备检索增强的文化有效性评估器，发现提供交互上下文能提升大多数模型的表现，但无法完全消除文化相关的翻译难题。

    

    社交媒体评论本质上是互动交流，然而它们在翻译和评估时往往被当作孤立的文本。当含义依赖于周围的讨论、社区特有的俚语或具有文化背景的表达时，这种不匹配的问题尤为突出。我们提出了CULTURE-CMT，一个用于情境化社交媒体评论翻译的中英文基准数据集，它恢复了用户自然可获得的帖子和对话上下文。其572个经过人工验证的示例被组织为俚语、上下文和两者兼有三个类别，从而能够对翻译难度的不同来源及其相互作用进行诊断分析。为了支持可扩展的评估，我们进一步开发了一个检索增强的文化有效性Judger，它将经过校准的评分模型与动态检索的文化知识相结合。CULTURE-CMT揭示了若干非平凡的模式：提供交互上下文可以改善大多数模型配置的表现，但并不能消除文化……（摘要在此处截断）

    arXiv:2611.02018v1 Announce Type: new  Abstract: Social-media comments are interactions, yet they are often translated and evaluated as isolated text. This mismatch is especially problematic when meaning depends on surrounding discussion, community-specific slang, or culturally situated expressions. We introduce CULTURE-CMT, a Chinese-English benchmark for contextualized social-media comment translation that restores the post and conversational context naturally available to users. Its 572 manually verified examples are organized into Slang, Context, and Both, enabling diagnostic analysis of distinct and interacting sources of translation difficulty. To support scalable evaluation, we further develop a retrieval-augmented cultural-effectiveness Judger that combines a calibrated scoring model with dynamically retrieved cultural knowledge. CULTURE-CMT reveals several non-trivial patterns. Providing interaction context improves most model configurations, but does not eliminate culturally 
    
[^112]: Switch-SWE：编程智能体在未完成任务之间切换时能否保持专注？

    Switch-SWE: Do Coding Agents Hold Focus When Switching Between Unfinished Tasks?

    [https://arxiv.org/abs/2611.02017](https://arxiv.org/abs/2611.02017)

    该论文提出了Switch-SWE基准，通过在同一仓库中交错多个未完成任务来评估编程智能体在任务切换后能否保持专注、恢复原任务的需求与进度并成功完成任务。

    

    软件开发会话通常涉及多个未完成的任务，而不是一个问题被一次处理到完成。随着编程智能体承担仓库级别的开发工作，它们需要在中断处理其他工作后，恢复先前任务的需求和部分进展。现有的仓库级基准通常仅对一个“问题到补丁”的会话进行评分，无法直接衡量智能体在同一仓库中推进其他未完成工作之后能否恢复某个任务。我们提出了Switch-SWE，这是一个用于专注任务恢复的基准，其会话中交错进行来自同一源仓库的多个未完成任务，并为每个任务提供持久的代码检出。每个专注任务都在匹配的无中断会话和切换会话中进行评估，两者具有相同的专注任务需求和可执行的验证器。切换会话会交错进行同一仓库中其他未完成任务的有序阶段，而只有最终的专注任务补丁会被评[估]（原文此处截断）。

    arXiv:2611.02017v1 Announce Type: new  Abstract: Software development sessions often involve several unfinished tasks rather than one issue carried through to completion. As coding agents take on repository-level development, they need to recover an earlier task's requirements and partial progress after intervening work. Existing repository-level benchmarks typically score one issue-to-patch episode and do not directly measure whether an agent can resume a task after advancing other unfinished work in the same repository. We introduce \textbf{\ourmethod}, a benchmark for focus-task resumption in sessions that interleave unfinished tasks drawn from the same source repository, with a persistent checkout for each task. Each focus task is evaluated in matched uninterrupted and switched sessions with identical focus requirements and executable verifiers. The switched session interleaves ordered stages from other unfinished tasks in the same repository, while only the final focus patch is sc
    
[^113]: 规模化使其成为可能：大模型如何掌握“不可能语言”

    Scaling Makes It Possible: How Large Models Master Impossible Languages

    [https://arxiv.org/abs/2611.02015](https://arxiv.org/abs/2611.02015)

    该论文提出“语义连贯性”概念，揭示出十亿参数级的大语言模型能够像掌握可能语言一样掌握违反人类语言约束的“不可能语言”，而百万参数级小模型则显著失败，且性能差距随语义复杂度增加而扩大。

    

    尽管语言模型（LM）在自然语言任务中表现出色，但其学习“不可能语言”——即那些违反了使人类语言可学习、可运作的结构性、认知性或逻辑性约束的语言——的能力仍存在争议。语言学理论家与使用小规模语言模型进行实证研究的研究者之间的分歧进一步加剧了这一争论。为解决这一争议，我们提出了“语义连贯性”（semantic coherence）这一概念，用于衡量语义信息在语言结构中被表达的有效程度，并考察了语言模型在多个参数规模下的学习能力。通过在“不可能语言”样本上微调十亿参数级的语言模型，我们证明较大规模的语言模型在“不可能语言”和“可能语言”上能够取得相近的性能，而百万参数级的语言模型在“不可能语言”上训练时则表现出显著的性能退化，且这一性能差距随语义复杂度的增加而不断扩大。我们进一步表明……

    arXiv:2611.02015v1 Announce Type: new  Abstract: While language models (LMs) excel in natural language tasks, their ability to learn impossible languages--those that violate the structural, cognitive, or logical constraints that make human languages learnable and functional--remains debatable. This controversy is further fueled by a divide between linguistic theorists and empirical researchers working with small-scale LMs. To address this controversy, we introduce semantic coherence. This concept measures how effectively semantic information is expressed in language structures and examines the learning capabilities of LMs across multiple scales. By fine-tuning billion-parameter LMs on impossible language samples, we demonstrate that larger LMs achieve similar performance on impossible and possible languages, whereas million-parameter LMs exhibit significant degradation when trained on impossible languages. This performance gap widens as semantic complexity increases. We further show th
    
[^114]: MedPsy：面向高效边缘部署的最先进小型医学语言模型

    MedPsy: State-of-the-Art Small Medical Language Models for Efficient Edge Deployment

    [https://arxiv.org/abs/2611.02011](https://arxiv.org/abs/2611.02011)

    MedPsy是一系列专为边缘设备部署设计的1.7B和4B小型医学语言模型，通过合成医学数据流水线、四阶段后训练课程（两个SFT加两个带难例挖掘的RL阶段）以及GGUF量化边缘部署评估，在精度、生成效率与部署可行性之间实现了最优平衡。

    

    医学语言模型有望提供专家级临床支持，但最强大的开源大语言模型对于私有边缘设备部署而言仍然过于庞大，而紧凑型医学模型在知识密集型和真实世界医疗任务上的表现仍存在明显差距。符合边缘内存预算的推理模型可能产生冗长的推理轨迹，从而增加顺序解码开销、端到端延迟、能耗以及KV缓存内存压力。我们提出了MedPsy，这是一系列专为边缘部署设计的纯文本1.7B和4B医学语言模型，旨在解决这一精度-生成-效率的权衡问题。我们的方案结合了以下要素：基于生物学、医学和健康种子数据构建的合成医学数据流水线，并辅以来自235B医学专业大语言模型教师模型的思维链目标；一个包含两个SFT阶段和两个带难例挖掘的RL阶段的四阶段后训练课程；以及覆盖多种GGUF量化格式的边缘部署评估。

    arXiv:2611.02011v1 Announce Type: new  Abstract: Medical language models promise expert clinical support, but the strongest open-source LLMs remain too large for private edge-device deployment, while compact medical models still trail substantially on knowledge-intensive and real-world healthcare tasks. Reasoning models that fit edge memory budgets can produce long reasoning traces, increasing sequential decoding, end-to-end latency, energy consumption, and KV-cache memory pressure. We present MedPsy, a family of text-only 1.7B and 4B medical language models designed for edge deployment, targeting this accuracy-generation-efficiency trade-off. Our recipe combines a synthetic medical data pipeline over biology, medicine, and health seeds with chain-of-thought targets from a 235B medical-focused LLM teacher; a four-stage post-training curriculum comprising two SFT stages followed by two RL stages with hard-sample mining; and an edge deployment evaluation spanning multiple GGUF quantizati
    
[^115]: 面向社交媒体翻译训练的检索增强文化评估

    Retrieval-Enhanced Cultural Evaluation for Social Media Translation Training

    [https://arxiv.org/abs/2611.02009](https://arxiv.org/abs/2611.02009)

    本文提出CUE-MT社交媒体翻译模型系列，通过Agent按需检索外部文化证据并提炼为提示来增强校准后的Judger，从而改进监督数据精炼与强化学习奖励信号，在文化有效性上比同规模开源基线提升7.44至15.12个百分点。

    

    社交媒体翻译需要在保留原意的同时捕捉用户生成内容（UGC）的文化细微差别。然而，基于大语言模型的评估器在文化知识方面的不足可能导致不准确的评估和不可靠的训练奖励。为了改进这些信号，我们采用检索增强的文化评估方法来训练CUE-MT，这是一系列社交媒体翻译模型。一个经过校准的、领域特定的Judger（评判器）用于指导监督数据的精炼，并通过一个Agent（智能体）进行困难样本挖掘和强化学习来增强其能力。该Agent根据需要检索相关的外部证据，并将其提炼为简洁的提示，Judger利用这些提示对翻译进行评分并分配奖励。在9B、27B和35B-A3B三种骨干模型上，CUE-MT在CUE-MT-Bench和三个公开基准测试上的表现均优于同等规模的开源基线模型。在CUE-MT-Bench上的人工评估显示，文化有效性提升了7.44至15.12个百分点。

    arXiv:2611.02009v1 Announce Type: new  Abstract: Social media translation requires preserving meaning while capturing the cultural nuances of user-generated content (UGC). However, gaps in the cultural knowledge of LLM-based evaluators can lead to inaccurate assessments and unreliable training rewards. To improve these signals, we use retrieval-enhanced cultural evaluation to train CUE-MT, a family of social media translation models. A calibrated, domain-specific Judger guides supervised data refinement and is augmented by an Agent for hard-example mining and reinforcement learning. The Agent retrieves relevant external evidence as needed and distills it into concise hints that the Judger uses to score translations and assign rewards. Across 9B, 27B, and 35B-A3B backbones, CUE-MT outperforms the evaluated open-source baselines of comparable size on CUE-MT-Bench and three public benchmarks. Human evaluation on CUE-MT-Bench shows cultural effectiveness gains of 7.44-15.12 percentage poin
    
[^116]: 长期对齐，短期规划：通过在长时程学习环境中实现短期计划来模拟多样化的符合画像的学生

    To Align Long, Plan Short: Simulating Diverse Profile-Aligned Students by Realizing Short-Term Plans in Long-Horizon Learning Environments

    [https://arxiv.org/abs/2611.02002](https://arxiv.org/abs/2611.02002)

    提出了ALPS分层模拟学生框架，通过交替进行“画像对齐的任务片段规划”和“动作生成”，使模拟学生在长时程开放式学习环境中持续符合指定的行为画像。

    

    模拟学生提供了一种可扩展的方式，可以在真实学生部署之前测试学习平台和AI导师，并能扩充稀缺数据以研究大规模学习。现有的基于大语言模型（LLM）的模拟器主要针对短的、以语言为中心的交互。然而，在开放式学习环境中，学生行为产生于长时程交互轨迹中的模式，在这种情况下，轮次级别的提示可能会偏离指定的行为画像。我们提出了ALPS（通过短期规划实现长期对齐），这是一种以可观察的、环境特定的行为画像为条件的分层模拟学生。ALPS不是独立地选择每一个动作，而是交替进行画像对齐的任务片段规划和相应动作的生成。我们在数据充足和数据稀缺的两种设置中评估了ALPS，其中监督信号既可以来自真实的学生轨迹，也可以来自教师LLM。在两种设置中

    arXiv:2611.02002v1 Announce Type: cross  Abstract: Simulated students offer a scalable way to test learning platforms and AI tutors before deployment with real students and to augment scarce data for studying learning at scale. Existing LLM-based simulators primarily target short, language-centered interactions. In open-ended learning environments, however, student behavior emerges from patterns across long-horizon interaction trajectories, where turn-level prompting can drift from an assigned behavior profile. We introduce ALPS (Align Long-term by Planning Short-term), a hierarchical simulated student conditioned on observable, environment-specific behavior profiles. Rather than selecting every action independently, ALPS alternates between planning profile-aligned task segments and generating the corresponding actions. We evaluate ALPS in both data-sufficient and data-scarce settings, where supervision comes either from real student trajectories or from a teacher LLM. Across both sett
    
[^117]: TrackMem：追踪长程金融对话中不断演变的信念

    TrackMem: Tracking Evolving Beliefs in Long-Horizon Financial Dialogue

    [https://arxiv.org/abs/2611.02001](https://arxiv.org/abs/2611.02001)

    提出TrackMem——一种按说话者组织的图结构记忆系统，能够追踪金融对话中被修订的数字、跨会话演变的立场以及一次陈述即约束后续问题的规则，并配套发布FinLoCoMo长程投资对话基准来检验动态信念追踪能力。

    

    LLM智能体的记忆系统主要在社交对话上进行评估，而社交对话中事实只累积、很少改变，因此“记忆”在很大程度上就简化为找到某句话是在哪一轮说出的。金融对话打破了这一假设：数字会被修订，立场会在没有重述的情况下悄然转变，一次陈述的规则会支配后续的多个问题，同一家公司还会在多个分析视角下被反复讨论。因此，那些检索与问题最相似或最新文本片段的记忆系统，会返回已被取代的旧数值，错过与问题措辞完全不同的规则，也无法还原散布在多个会话中的立场变化。我们提出TrackMem，一种按说话者组织的图结构记忆系统，它追踪每个数字的历史演变，按实体与分析视角分离不同观点，并按时间顺序保留每个说话者的动作轨迹。为了评估该系统，我们构建了FinLoCoMo基准，包含15段长程投资对话和765个问题，涵盖两种场景设置（原文在此处截断）。

    arXiv:2611.02001v1 Announce Type: new  Abstract: Memory systems for LLM agents are evaluated mostly on social dialogue, where facts accumulate and rarely change, so remembering reduces largely to finding the turn in which something was said. Financial dialogue breaks this assumption: figures are revised, stances shift without being restated, rules stated once govern later questions, and a single company is discussed under several analytical lenses. Memory systems that retrieve the passages most similar to a question, or the most recent ones, therefore return superseded values, miss rules that share no wording with the question, and cannot recover a stance spread across sessions. We present TrackMem, a per-speaker graph memory that tracks the history of figures, separates perspectives by entity and lens, and preserves each speaker's actions in chronological order. To evaluate it, we introduce FinLoCoMo, a benchmark of 15 long-horizon investment conversations with 765 questions in two sp
    
[^118]: 更好的答案，更弱的证据控制：在线策略蒸馏中的证据绕过现象

    Better Answers, Weaker Evidence Control: Evidence Bypass in On-Policy Distillation

    [https://arxiv.org/abs/2611.01967](https://arxiv.org/abs/2611.01967)

    该研究发现在线策略蒸馏虽能提升答案准确率，却会引发“证据绕过”现象——学生模型在文档证据被修改后仍坚持原答案，揭示出蒸馏过程会削弱模型对支持证据的依赖。

    

    在线策略蒸馏（OPD）可以在提高答案准确率的同时，削弱模型对支持性证据的响应能力。我们通过匹配的文档干预在科学文档问答任务中研究了这种分离现象——这些干预在保持问题和不相关上下文不变的同时，改变文档所支持的答案。我们识别出“证据绕过”现象：当文档被修改为支持另一个答案后，学生模型仍保留原本正确的答案。在QASPER数据集上使用Qwen3-4B作为学生模型时，标准OPD将原始文档准确率提高了3.4个百分点，将反事实准确率降低了5.3个百分点，并将绕过现象增加了15.7个百分点；同样的趋势模式也出现在PeerQA数据集上。受控扰动实验将这种偏移定位于对答案至关重要的证据，且绕过程度随学生模型蒸馏前对原始答案的闭卷偏好增强而增长。一项对教师目标、特权信息、散度和监督密度的因子化研究（摘要在此处截断）

    arXiv:2611.01967v1 Announce Type: new  Abstract: On-policy distillation (OPD) can improve answer accuracy while weakening responsiveness to supporting evidence. We investigate this separation in scientific-document question answering through matched document interventions that change the supported answer while preserving the question and unrelated context. We identify evidence bypass: a student retains an originally correct answer after the document changes to support a different one. On QASPER with a Qwen3-4B student, standard OPD raises original-document accuracy by 3.4 percentage points, lowers counterfactual accuracy by 5.3 points, and increases bypass by 15.7 points; the same directional pattern appears on PeerQA. Controlled perturbations localize the shift to answer-critical evidence, and bypass grows with the student's pre-distillation closed-book preference for the original answer. A factorized study of teacher targets, privileged information, divergence, and supervision densit
    
[^119]: 迈向自我演化的叙事系统：基于大语言模型的实时适应与长期优化

    Towards Self-Evolving Narrative Systems: Real-Time Adaptation and Long-Term Optimization with LLMs

    [https://arxiv.org/abs/2611.01954](https://arxiv.org/abs/2611.01954)

    提出了一种通过语义相似度对齐来动态平衡预编写内容与LLM实时生成的混合叙事框架，并利用反馈驱动的自我演化机制自动优化提示词，在无需人工提示工程的情况下同时提升叙事连贯性和玩家自主性。

    

    传统的交互式叙事系统提供了强大的作者控制力和结构连贯性，但往往限制玩家的自主性。相比之下，由大语言模型（LLM）驱动的叙事能够实现开放式生成和更大的灵活性，但常常面临不一致性和叙事控制力减弱的问题。为了在这两者之间取得平衡，我们提出了一种混合叙事框架，通过语义相似度对齐，在预编写的剧情内容与实时LLM生成之间实现动态平衡。此外，如何在长时间游戏过程中维持叙事质量是基于LLM的故事生成中的一个核心挑战。为此，我们引入了一个自我演化的、反馈驱动的优化框架，利用上下文学习迭代地优化提示词，从而无需人工进行提示工程。通过案例研究和基于参与者的评估，我们证明了我们的框架在保持玩家自主性的同时增强了叙事连贯性，有效地……（原文摘要在此处截断）

    arXiv:2611.01954v1 Announce Type: new  Abstract: Traditional interactive narrative systems offer strong authorial control and structural coherence but often restrict player agency. In contrast, LLM-driven narratives enable open-ended generation and greater flexibility, yet frequently suffer from inconsistency and diminished narrative control. To navigate this trade-off, we propose a hybrid narrative framework that dynamically balances pre-authored content with real-time LLM generation via semantic similarity alignment. Additionally, sustaining narrative quality over extended gameplay remains a core challenge in LLM-based storytelling. To address this, we introduce a self-evolving, feedback-driven optimization framework that leverages in-context learning to iteratively refine prompts, eliminating the need for manual prompt engineering. Through a case study and participant-based evaluation, we show that our framework enhances narrative coherence while preserving player agency, effectivel
    
[^120]: 追踪语言模型如何将时间锚点整合到事实回忆中

    Tracing How Language Models Integrate Temporal Anchors into Factual Recall

    [https://arxiv.org/abs/2611.01945](https://arxiv.org/abs/2611.01945)

    该论文提出“锚定事实回忆”新任务，通过对四个开源模型进行因果干预，揭示了语言模型整合时间锚点的两条机制路径——锚点先解析主语词元表示为查询年份再由关系重新分配读出的主语介导路径，以及最终位置注意力直接参与选择的直接路径。

    

    语言模型中的事实回忆通常被框定为检索一种事实关联：一个主语—关系对映射到单一宾语。然而，许多事实是依赖于上下文的：同一主语—关系对会根据时间等条件映射到不同的宾语。我们提出了锚定事实回忆这一任务，其中模型接收一个显式的条件表达式（即锚点），并且必须选择在该锚点下有效的宾语。这引发了单一对象回忆所不会涉及的问题：锚点在何处进入计算过程，它改变了什么，以及这种改变如何选择出答案。我们聚焦于时间锚点，通过对四个开源权重模型在英文提示上进行因果干预来回答这些问题，并识别出两条路径。在主语介导路径中，锚点将最终的主语词元表示解析为所查询的年份，而关系则将该状态的读出重新分配到各个注意力头中。在直接路径中，最终位置的注意力……

    arXiv:2611.01945v1 Announce Type: new  Abstract: Factual recall in language models is typically framed as retrieving a factual association: a subject--relation pair maps to a single object. However, many facts are context-dependent: the same pair maps to different objects depending on a condition such as time. We introduce anchored factual recall, in which the model receives an explicit conditioning expression, an anchor, and must select the object valid under it. This raises questions that single-object recall does not: where the anchor enters the computation, what it changes, and how that change selects the answer. Focusing on temporal anchors, we answer them with causal interventions on four open-weight models on English prompts and identify two routes. In a subject-mediated route, the anchor resolves the final subject-token representation to the queried year, and the relation redistributes the readout of this state across attention heads. In a direct route] the final position atten
    
[^121]: TAC-CoT：通过交错显式标记实现的工具赋能隐式思维链

    TAC-CoT: Tool-Empowered Implicit Chain-of-Thought via Interleaving Explicit Tokens

    [https://arxiv.org/abs/2611.01926](https://arxiv.org/abs/2611.01926)

    TAC-CoT提出一种工具赋能的自适应隐式思维链框架，用紧凑的潜在推理片段替代显式推理文本、仅在工具调用、观察与最终答案处保留文本标记，从而在智能体任务上实现准确率最高提升8.8个百分点、前向传递次数最多减少6.8倍。

    

    基于大语言模型的智能体系统日益与思维链推理相结合，以解决需要准确推理、外部知识和多步深思的任务。然而，显式思维链会带来巨大的生成开销，而隐式思维链——即用紧凑的连续潜在标记替代文本化推理过程——又无法自然地支持智能体工具使用所需的显式通信接口。我们提出了TAC-CoT，这是一个工具赋能的自适应隐式思维链框架，它用紧凑的潜在推理片段替代显式推理轨迹，同时仅为工具调用、观察结果和最终答案保留文本标记。这种交错式设计使智能体无需将每个中间推理步骤言语化即可进行工具使用。在智能体数学与问答基准上的实验表明，TAC-CoT在将准确率提升至多8.8个百分点的同时，将模型前向传递次数减少至多6.8倍。此外，分析显示TAC-CoT改进了工具……

    arXiv:2611.01926v1 Announce Type: new  Abstract: LLM-based agentic systems are increasingly combined with CoT reasoning to solve tasks that require accurate reasoning, external knowledge, and multi-step deliberation. Yet, explicit CoT incurs substantial generation overhead, while implicit CoT, which replaces textual rationales with compact continuous tokens, does not naturally support the explicit communication interface required for agentic tool-use. We propose TAC-CoT, a tool-empowered adaptive implicit CoT framework that replaces explicit reasoning traces with compact latent reasoning segments while retaining textual tokens only for tool calls, observations, and final answers. This interleaved design enables agentic tool-use without verbalizing every intermediate reasoning step. Experiments on agentic math and QA benchmarks show that TAC-CoT improves accuracy by up to 8.8 points while reducing model forward passes by up to 6.8x. Furthermore, analyses show that TAC-CoT improves tool-
    
[^122]: 不要猜测，明智地提问：开放域大语言模型对话中主动询问能力的基准测试与训练

    Don't Guess, Ask Wisely: Benchmarking and Training Proactive Elicitation in Open-Domain LLM Conversations

    [https://arxiv.org/abs/2611.01878](https://arxiv.org/abs/2611.01878)

    本文提出VagueBench基准数据集与DialogGRPO离线强化学习方法，通过将多轮询问建模为对话状态有向无环图并结合即时信息增益与前瞻价值的密集轮次级奖励，训练大语言模型在面对模糊查询时主动提出有针对性的问题。

    

    当用户查询不完整时，大语言模型应该在给出最终答案之前提出有针对性的问题，我们将这种能力称为主动询问。然而，对七个大语言模型（从8B到前沿模型）在真实用户-助手查询上的评估显示，仅有不到11%的欠规范查询触发了后续提问。为了对这一能力进行基准测试和训练，我们提出了VagueBench，其中每个种子查询都被标注了结构化的意图和细节单元，然后通过有控制地移除语言线索，将其降级为意图模糊和完全模糊的变体，从而生成已知的隐藏信息集合，用于评估何时提问以及提问什么。我们进一步提出了DialogGRPO，这是一种离线强化学习方法，它将多轮询问过程组织为对话状态有向无环图，其中每个节点上的候选问题获得密集的轮次级奖励，该奖励通过组相对比较将即时信息增益与前瞻价值相结合……

    arXiv:2611.01878v1 Announce Type: new  Abstract: When user queries are incomplete, large language models should ask targeted questions before committing to an answer, a capability we call proactive elicitation, yet evaluation on real user-assistant queries across seven LLMs (8B to frontier) shows that fewer than 11% of underspecified queries trigger a follow-up question. To benchmark and train this capability, we introduce VagueBench, where each seed query is annotated with structured intent and detail units then degraded into intent-vague and fully-vague variants via controlled removal of linguistic cues, yielding known hidden information sets for evaluating both when and what to ask. We propose DialogGRPO, an offline reinforcement learning method that organizes multi-turn elicitation as a dialogue-state directed acyclic graph, where candidate questions at each node receive dense turn-level rewards combining immediate information gain with lookahead value through group-relative compar
    
[^123]: 量化循环语言模型比你想象的更容易

    Quantizing Recurrent Language Models Is Easier Than You Think

    [https://arxiv.org/abs/2611.01864](https://arxiv.org/abs/2611.01864)

    该论文首次系统研究了循环语言模型的量化设计空间，发现衰减型线性递归相比softmax注意力产生的异常通道更弱，因此无需特殊处理就能更容易地被量化。

    

    线性递归正日益成为大语言模型中的常见构建模块，它用固定大小的循环状态取代了softmax注意力中不断增长的KV缓存，从而实现高效解码。然而，现有的量化技术栈主要围绕softmax Transformer构建，其数值特性和推理状态与循环模型存在差异。我们的目标是理解这些差异，并研究循环模型量化方法的设计空间。在分析中，我们区分了两个量化目标。首先，状态外量化涉及循环状态之外的权重和激活值，其分布由序列混合器在训练过程中间接塑造。在受控的1.3B参数模型实验中，我们发现衰减型循环比softmax注意力产生更弱的残差异常通道，因此无需进行不相干性处理就能更容易地量化。其次，在……

    arXiv:2611.01864v1 Announce Type: cross  Abstract: Linear recurrences are becoming an increasingly common building block in large language models, replacing the growing KV cache of softmax attention with a fixed-size recurrent state that enables efficient decoding. However, the quantization stack has largely been developed around softmax Transformers, whose numerical properties and inference state differ from those of recurrent models. Our goal is to understand these differences and study the design space of quantization methods for recurrent models. In our analysis, we differentiate between two targets. First, extra-state quantization concerns weights and activations outside the recurrent state, whose distributions are shaped indirectly by the sequence mixer during training. In controlled 1.3B parameter models, we find that decayed recurrences develop weaker residual outlier channels than softmax attention and are therefore easier to quantize without incoherence processing. Second, in
    
[^124]: 通过复杂任务的语义分解实现自演化的智能体工作流

    Self-Evolving Agentic Workflows via Semantic Decomposition of Complex Tasks

    [https://arxiv.org/abs/2611.01840](https://arxiv.org/abs/2611.01840)

    该论文提出了一个通过语义任务分解实现智能体工作流自演化的框架，智能体利用执行反馈不断优化任务分解并更新子任务流程，其中大多数最终决策流程由代码而非大语言模型实现，从而显著降低了重复推理的成本与延迟。

    

    智能体工作流将复杂任务组织成可复用的流程。然而，在每一步都调用大语言模型（LLM）会产生重复的推理成本和延迟。我们提出了一个通过语义任务分解实现自演化智能体工作流的框架。每个子任务都具有明确的适用条件以及相应的执行流程，可根据需要执行代码或调用模型。智能体利用执行反馈来优化任务分解并更新这些流程。在GLEE竞赛中的实验展示了该框架在讨价还价、谈判和说服任务上的表现。大多数观察到的最终决策流程都是用代码实现的。代码可在 https://github.com/Applied-Machine-Learning-Lab/NeurIPS2026_SeaFlow 获取。

    arXiv:2611.01840v1 Announce Type: cross  Abstract: Agentic workflows organize complex tasks into reusable procedures. However, invoking an LLM at every step incurs repeated inference costs and delays. We propose a framework for self-evolving agentic workflows through semantic task decomposition. Each subtask has explicit applicability conditions and a procedure that executes code or invokes a model as needed. An agent uses execution feedback to refine the decomposition and update these procedures. Experiments in the GLEE competition show the framework on bargaining, negotiation and persuasion. Most observed final-decision procedures are implemented in code. Code is available at https://github.com/Applied-Machine-Learning-Lab/NeurIPS2026_SeaFlow.
    
[^125]: LegoEnv：通过模块化重组实现自我进化的智能体环境

    LegoEnv: Self-Evolving Agentic Environments via Modular Recombination

    [https://arxiv.org/abs/2611.01834](https://arxiv.org/abs/2611.01834)

    LegoEnv提出了一种将环境由可复用原子（意图、规则、数据、服务器）组合构建、并在环境反馈不可靠时自我进化这些原子的新范式，构建了5,465个经过验证的环境，将可解环境比例从71.6%提升至88.3%，并显著提升了智能体的SFT与RL训练效果。

    

    LLM智能体通过与环境的交互来学习，然而现有的环境是为特定任务设计的，难以泛化到新任务。近期的方法开始利用LLM自动化构建环境，但它们都是从零开始生成每个环境，既不复用跨环境共享的部分，也无法将某个环境中发现的修复传递到其他环境。为解决这一问题，我们提出了一种“进化环境原子”的新范式，该范式由可复用的原子（意图、规则、数据与服务器）组合而成环境，并在某个环境无法给出可靠反馈时（例如，应用缺少任务所需的工具）对这些原子进行进化。为实例化这一范式，我们提出了LegoEnv，它在223个应用上构建了5,465个经过验证的环境，一轮进化即将可解环境的比例从71.6%提升至88.3%。在AutomationBench上，基于这些环境进行SFT使Qwen3.6-27B的性能从16.0%提升至32.0%，而使用它们的RL……（摘要原文在此处截断）

    arXiv:2611.01834v1 Announce Type: new  Abstract: LLM agents learn by interacting with environments, yet existing environments are designed for specific tasks and fail to generalize to new ones. Recent methods automate environment construction with LLMs. However, they generate each environment from scratch, neither reusing parts shared across environments nor carrying a fix from one environment to the rest. To address this, we propose a new paradigm of evolving environment atoms, which composes environments from reusable atoms (intent, rule, data and server) and evolves the atoms when an environment fails to give reliable feedback (e.g., an application lacks a tool the task requires). To instantiate this paradigm, we introduce LegoEnv, which builds 5,465 verified environments over 223 applications, and one round of evolution raises the share of solvable environments from 71.6% to 88.3%. On AutomationBench, SFT on these environments lifts Qwen3.6-27B from 16.0% to 32.0%, and RL with thei
    
[^126]: 自然语言规则的作用域文法：模逻辑等价的读法计数与枚举

    The Scope Grammar of Natural-Language Rules: Counting and Enumerating Readings modulo Logical Equivalence

    [https://arxiv.org/abs/2611.01825](https://arxiv.org/abs/2611.01825)

    该论文提出“作用域文法”来刻画自然语言规则中连接词作用域的歧义性，通过读一次范式与双标记递推，精确计数并枚举了连接词骨架在模逻辑等价意义下逻辑上不同的读法数量。

    

    为人类编写的规则确定了其原子条件和连接词，但并未明确连接词应如何加括号，而每一种加括号方式都对应一种不同的逻辑读法，自动形式化器必须解决或揭示这些读法。我们研究了“作用域文法”，这是一种刻意保持歧义的上下文无关文法，其推导树即为连接词骨架的各种读法，而对其按逻辑等价取商后的结果才是下游应用真正关心的内容。一个含有 k 个二元连接词的骨架拥有 C_k 棵推导树（即第 k 个卡塔兰数），但逻辑上不同的读法数量却少得多。对于无重复原子的正片段，我们精确刻画了这一差距：借助读一次范式的唯一性，我们用最低公共祖先刻画了可实现的范式，并通过在算子词上的双标记递推对读法进行计数，其结果与最多十一个原子的穷举枚举相验证。该计数在反转和 ∧/∨ 对偶下保持不变。

    arXiv:2611.01825v1 Announce Type: cross  Abstract: A rule written for people fixes its atomic conditions and connectives but leaves open how the connectives are bracketed, and each bracketing is a different logical reading that an autoformalizer must resolve or expose. We study the scope grammar, a deliberately ambiguous context-free grammar whose derivation trees are the readings of a connective skeleton and whose quotient by logical equivalence is what matters downstream. A skeleton with $k$ binary connectives has $C_k$ derivation trees, the $k$-th Catalan number, but far fewer logically distinct readings. For the positive, repetition-free fragment we make the gap exact: through the uniqueness of the read-once normal form we characterize the realizable normal forms by lowest common ancestors and count them with a two-label recurrence over the operator word, verified against exhaustive enumeration up to eleven atoms. The count is invariant under reversal and $\wedge/\vee$ duality; a s
    
[^127]: 共享关系，特定选择：语言模型中的条件敏感性与竞争通路

    Shared relations, specific choices: conditional sensitivity and competing pathways in language models

    [https://arxiv.org/abs/2611.01809](https://arxiv.org/abs/2611.01809)

    该研究揭示语言模型中存在跨查询共享且可迁移到新实体的低维关系表征，但其功能符号并非必然可迁移——相同的关系向量编辑在不同类型提示下会产生相反的选择效应。

    

    跨查询共享的关系表征如何驱动针对特定查询的选择？通过在GPT-2 Small、Qwen2.5-1.5B和Llama-3.2-1B上进行受控的实体绑定实验，我们揭示了一个低维的关系背景，该背景在不同查询角色之间基本共享，并能迁移到新的名称上。当查询另一个属性域时，同域交互仍然持续存在。在移除该组件后，恢复独立训练得到的关系方向能使输出对比比符号反转或等范数随机对照更接近原始值，这一结果在所有三个模型中均得到验证。然而，可迁移的关系表征不一定具有可迁移的功能符号。在Qwen2.5中，固定模型权重时，训练均值关系向量和编辑剂量在具有不同答案词表的二元提示与词表映射提示之间产生相反的选择效应。

    arXiv:2611.01809v1 Announce Type: new  Abstract: How does a relational representation shared across queries drive a query-specific choice? Controlled entity-binding experiments with GPT-2 Small, Qwen2.5-1.5B and Llama-3.2-1B reveal a low-dimensional relational background that is predominantly shared across query roles and transfers to new names. Same-domain interactions persist when another attribute domain is queried. After removing this component, restoring independently trained relation directions brings output contrasts closer to their original values than sign-reversed or equal-norm random controls in all three models. Yet a transferable relational representation need not have a transferable functional sign. In Qwen2.5, fixing model weights, the training-mean relation vector and edit dose yields opposite selection effects across binary and lexical-mapping prompts with different answer vocabularies. A factorial contrast of downstream sensitivities across attribute-query combination
    
[^128]: 使用小型语言模型进行基于量规的学生报告评估

    Rubric-Based Assessment of Student Reports Using Small Language Models

    [https://arxiv.org/abs/2611.01785](https://arxiv.org/abs/2611.01785)

    该研究提出一个面向低资源环境且保护隐私的自动化评估流水线，利用小型语言模型和视觉-语言模型依据量规评估学生在网络安全事件响应演习中的报告，纯文本评估准确率最高可达97%。

    

    语言模型可以利用预定义的量规来分析和评估学生文本并提供结构化反馈，从而减轻教师的工作负担。然而，现有解决方案通常依赖于资源消耗大、成本高昂且存在隐私隐患的外部服务，因此不适用于资源受限的环境。本研究评估了一个自动化评估流水线，其具有三个关键目标：面向低资源环境的设计、隐私保护，以及聚焦于网络安全教育中的新场景。该流水线集成了用于文本评估的轻量级语言模型和用于图像解读的视觉-语言模型。研究在某本科课程为期两个学期的网络安全事件响应演习中收集了105名学生的作答。在不同硬件配置下的实验结果表明，低资源环境中的小型模型在纯文本评估中可达到高达97%的准确率，在富文本评估中可达到高达94%的准确率。

    arXiv:2611.01785v1 Announce Type: cross  Abstract: Language models can analyze and assess student texts using predefined rubrics and deliver structured feedback, reducing instructors' workload. However, existing solutions often rely on external services that are resource-intensive, costly, and raise privacy concerns, making them unsuitable for constrained environments. This study evaluates an automated assessment pipeline with three key aims: design for low-resource environments, privacy, and focus on a novel context in cybersecurity education. The pipeline integrates lightweight language models for textual evaluation and a vision-language model for image interpretation. Responses from 105 students were collected over two semesters of an undergraduate course during a cybersecurity exercise on incident response. Results across different hardware configurations show that small models in low-resource environments can achieve up to 97% accuracy in text-only and up to 94% in rich-text asses
    
[^129]: MemRole：从记忆到决策的角色扮演智能体长程推理方法

    MemRole: From Memories to Decisions for Long-Horizon Reasoning in Role-Playing Agents

    [https://arxiv.org/abs/2611.01778](https://arxiv.org/abs/2611.01778)

    提出MemRole框架，通过长程记忆思考（LHMT）将隐式推理显式分解为上下文理解、记忆激活和响应策略规划，使角色扮演智能体能够利用基于角色且受时间线约束的记忆池做出角色一致的长程决策。

    

    基于大语言模型的角色扮演智能体虽然擅长模仿角色的语言风格，却难以在长叙事跨度上做出符合角色一致性的决策。这类决策受到跨多个先前场景的经验影响，而这些经验往往是不完整的，且可能指向不同方向。现有方法只是将跨场景记忆作为上下文注入，但并未推理过去事件如何塑造角色当前的动机与行为。我们提出MemRole，一个将这一记忆到决策过程显式化的框架。它引入了长程记忆思考，这是一种推理模式，将隐式推理分解为上下文理解、记忆激活和响应策略规划三个环节，并在一个由小说衍生的新数据集上训练，该数据集基于角色、受时间线约束的记忆池将每个角色的可用信息限制在其在当前叙事点上合理可知的范围内。除监督微调外，我们还提出了自我精炼的在策略选择……

    arXiv:2611.01778v1 Announce Type: new  Abstract: LLM-based role-playing agents are fluent at imitating a character's style, yet struggle to make role-consistent decisions over long narrative horizons. Such decisions are shaped by experiences spanning many prior scenes, which are often partial and may point in different directions. Existing methods inject cross-scene memories as context, but do not reason about how past events shape a character's current motivations and behavior. We propose MemRole, a framework that makes this memory-to-decision process explicit. It introduces Long-Horizon Memory Thinking (LHMT), a reasoning schema that decomposes hidden reasoning into context understanding, memory activation, and response strategy planning, trained on a novel-derived dataset whose character-grounded, timeline-constrained memory pools limit each character to what it could plausibly know at the current narrative point. Beyond supervised fine-tuning, we propose Self-Refining On-Policy Sel
    
[^130]: 表征系统性的有界评估协议

    A Bounded Evaluation Protocol for Representational Systematicity

    [https://arxiv.org/abs/2611.01755](https://arxiv.org/abs/2611.01755)

    该论文提出了一个针对表征系统性的有界评估协议，通过在匹配的行为对比上同时检验所声明表征的可恢复性与因果使用，并对解释赋予四种有界状态之一，从而解决了单一类型证据容易误导的验证难题。

    

    多项研究表明，神经网络在行为上可以具有系统性，即能够对熟悉部分的新颖组合产生正确的输出。然而，它们是否也在表征上具有系统性——即是否真正表征了这些部分及其角色，并利用这些表征来产生输出——仍然难以检验，因为每一种证据本身都可能造成误导：正确的输出可能来自捷径，可解码的信号未必真正被使用，干预操作可能通过非预期的路径发挥作用，而搜索失败也不能证明其不存在。我们提出了一种针对表征系统性的有界评估协议（Bounded-RS）：对于将某种系统性行为归因于一个所声明表征的解释，该协议在一个匹配的行为对比上检验该表征是否可被恢复、以及是否被因果地使用，检查这些证据的有效性，并将该解释归入四种有界状态之一，即“不支持”（Unsupported）等。

    arXiv:2611.01755v1 Announce Type: new  Abstract: Multiple studies have shown that neural networks can be behaviorally systematic, producing correct outputs for novel combinations of familiar parts. Whether they are also representationally systematic, that is, whether they represent these parts and their roles and use them to produce the outputs, remains hard to test, because each kind of evidence can mislead on its own: correct outputs can come from shortcuts, a decodable signal need not be used, an intervention can act through unintended pathways, and a failed search does not show absence. We propose a bounded evaluation protocol for representational systematicity (Bounded-RS): for an explanation that attributes a systematic behavior to a declared representation, it tests on a matched behavioral contrast whether that representation is recoverable and whether it is causally used, checks the validity of this evidence, and assigns the explanation one of four bounded states, Unsupported, 
    
[^131]: 演化记忆，学习元控制：面向经验增强型LLM智能体的测试时自适应

    Evolving Memory, Learning to Meta-Control: Test-Time Adaptation for Experience-augmented LLM Agents

    [https://arxiv.org/abs/2611.01747](https://arxiv.org/abs/2611.01747)

    提出了OMCE框架，通过轻量级上下文老虎机元控制器在测试时自适应过程中动态决定是否及如何调用和固化经验记忆，实现了LLM智能体经验的在线演化与最优部署。

    

    自演化智能体能够从任务流中积累经验，但现有框架通常依赖固定的流程来检索和利用这些经验，忽视了何时以及如何部署经验。我们提出了经验在线元控制（Online Meta-Control of Experience, OMCE）框架，该框架在测试时自适应过程中更新经验记忆，并从任务反馈中学习元控制策略。OMCE采用轻量级的上下文老虎机元控制器，利用紧凑的检索信号在四种认知动作中进行选择：DIRECT（直接）、EPISODE（情节）、GUIDE（引导）和DISTILL（蒸馏）。这一路由机制实现了三个相互耦合的决策：是否调用经验、使用哪种记忆格式、以及何时固化新的引导。OMCE在线演化，利用任务反馈更新记忆库，并通过上下文老虎机更新来改进路由策略。在四个在线自适应基准和四个主干模型上的实验表明，OMCE提升了首次……（摘要截断）

    arXiv:2611.01747v1 Announce Type: cross  Abstract: Self-evolving agents accumulate experience from task streams, but existing frameworks typically rely on fixed pipelines to retrieve and utilize this experience, neglecting when and how to deploy it. We introduce Online Meta-Control of Experience (OMCE), a framework that updates experience memory and learns a meta-control policy from task feedback during test-time adaptation. OMCE employs a lightweight contextual-bandit meta-controller that uses compact retrieval signals to select among four cognitive actions: DIRECT, EPISODE, GUIDE, and DISTILL. This routing mechanism operationalizes three coupled decisions: whether to invoke experience, which memory format to utilize, and when to consolidate new guidance. OMCE evolves online, leveraging task feedback to update the memory bank and refine the routing policy via contextual bandit updates. Experiments on four online adaptation benchmarks across four backbones show that OMCE improves first
    
[^132]: 少而足矣：用关系知识选择紧凑的知识图谱上下文，以及外部检查能捕捉与不能捕捉什么

    Less but Sufficient: Relation Knowledge Selects Compact Knowledge-Graph Contexts, and What External Checks Can and Cannot Catch

    [https://arxiv.org/abs/2611.01733](https://arxiv.org/abs/2611.01733)

    该论文提出基于关系知识从知识图谱中筛选出足以推导答案的紧凑事实上下文，并辅以可推导性检查、一致性检查和路由门控等外部检查机制，使开源LLM在CWQ上仅用四分之一提示token即可达到与SubgraphRAG相当甚至更优的精确匹配性能。

    

    大型语言模型（LLMs）通常读取按相关性排序的知识图谱事实，但这些事实是否足以推出答案或是否存在冲突仍悬而未决。我们的检查方法使用关系知识，这些知识既可以手工编写，也可以从标注了关系路径的训练问题中学习得到。该方法选择出一个能够得出答案的紧凑事实上下文，并从中推导答案；可推导性检查测试答案是否能从上下文推出，一致性检查测试事实之间是否冲突，而由验证集选择的路由门控则决定在何处用推导出的答案替换LLM给出的答案。七个开源LLM在精确匹配指标上阅读紧凑上下文的效果优于阅读按相关性排序的事实；在ComplexWebQuestions（CWQ）上，它们仅使用四分之一的提示词token就达到了与SubgraphRAG事实相当的效果，而更强的LLM则能更好地阅读SubgraphRAG的事实。在路由门控通过的情况下，我们的方法在全部53个LLM、基准和提示词设置组合中的精确匹配上都领先于SubgraphRAG。精度门控仅在检查恰好得出一个可推导答案时才返回该推导答案……（原文摘要此处截断）

    arXiv:2611.01733v1 Announce Type: new  Abstract: Large language models (LLMs) read knowledge-graph facts ranked by relevance, leaving open whether they suffice or conflict. Our checks use relation knowledge, written by hand or learned from training questions labelled with their relation paths. It selects a compact context of facts yielding an answer, and derives it; a derivability check tests whether an answer follows, a consistency check whether facts conflict, and a validation-chosen routing gate decides where the derived answer replaces the LLM's. Seven open LLMs read the compact context better in exact match than relevance-ranked facts; on ComplexWebQuestions (CWQ) they match SubgraphRAG's facts at a quarter of the prompt tokens, and stronger LLMs read SubgraphRAG's better. Where the routing gate passes, ours leads SubgraphRAG's in exact match in all 53 LLM, benchmark and prompt settings. A precision gate, returning the derived answer only when the check passes with exactly one der
    
[^133]: MAC-Bench：通过互联任务对多智能体协作进行基准测试

    MAC-Bench: Benchmarking Multi-Agent Collaboration via Interconnected Tasks

    [https://arxiv.org/abs/2611.01728](https://arxiv.org/abs/2611.01728)

    本文提出MAC-Bench基准，通过将协作需求直接嵌入任务依赖关系和共享资源约束中，而非强加预定义的智能体角色或通信协议，来全面评估基于大语言模型的多智能体系统的协作能力。

    

    基于大语言模型（LLM）的多智能体系统（MAS）通过在智能体之间分配工作，并协调各自的贡献以实现共同目标，从而解决具有挑战性的任务。因此，评估这些系统需要这样的基准：不仅要评估任务求解性能，还要评估处理相互依赖工作的能力。然而，现有的基准往往强调任务难度但缺乏实质性的协作需求，或者仅评估有限的预定义协作模式。在本文中，我们提出了MAC-Bench，这是一个通过互联任务全面评估多智能体协作能力的基准。MAC-Bench并不强制规定预定义的智能体角色或通信协议，而是将协作需求直接嵌入到任务依赖关系和共享资源约束之中。具体而言，我们在两个层面上从选定的基准任务中挖掘可复用的协作结构。局部协作模式……（摘要原文在此处截断）

    arXiv:2611.01728v1 Announce Type: cross  Abstract: Large language model (LLM)-based multi-agent systems (MAS) tackle challenging tasks by dividing work among agents and coordinating their contributions toward shared objectives. Evaluating these systems therefore requires benchmarks that assess not only task-solving performance but also the ability to handle interdependent work. However, existing benchmarks often emphasize task difficulty without substantial collaboration requirements or evaluate only a limited range of predefined collaboration patterns. In this paper, we introduce MAC-Bench, a comprehensive benchmark for evaluating multi-agent collaboration through interconnected tasks. Rather than imposing predefined agent roles or communication protocols, MAC-Bench embeds collaboration requirements directly into task dependencies and shared-resource constraints. Specifically, we mine reusable collaboration structures from selected benchmark tasks at two levels. Local collaboration pa
    
[^134]: EarlySelectKV：基于跨层查询代理的预判式KV选择

    EarlySelectKV: Anticipatory KV Selection via Cross-Layer Query Proxies

    [https://arxiv.org/abs/2611.01715](https://arxiv.org/abs/2611.01715)

    EarlySelectKV利用较早层的隐藏状态作为代理查询来提前预计算KV索引选择，使稀疏KV路由脱离注意力计算前的关键路径，在几乎不损失质量的情况下将端到端输出吞吐量提升高达12.4%。

    

    查询条件化的稀疏KV选择通过在每个解码步骤中仅关注历史KV条目的子集来降低长上下文推理成本。现有的选择器必须等待目标层的查询才能对候选进行评分并生成索引，这使得路由过程被置于注意力计算之前的紧邻路径上。我们提出了EarlySelectKV，这是一种无需训练的转换方法，它利用来自较早隐藏状态的代理查询来预计算所选的KV索引，同时在最终注意力计算中保留真实的目标层查询。在RocketKV、Quest和Loki上的测试显示，在所有评估设置中均未出现一致的质量退化模式。在主要的RocketKV评估中，ES-RocketKV在LongBench上相差+0.018分，在RULER上约为-0.01分，在代表性的NIAH/Passkey设置上为+0.25个百分点。在采用CPU卸载路由的TensorRT-LLM部署中，ES-RocketKV将端到端输出吞吐量提升了高达12.4%。

    arXiv:2611.01715v1 Announce Type: new  Abstract: Query-conditioned sparse KV selection reduces long-context inference cost by attending to a subset of historical KV entries at each decoding step. Existing selectors must wait for the target-layer query before scoring candidates and generating indices, placing routing on the immediate pre-attention path. We introduce EarlySelectKV, a training-free transformation that uses proxy queries from earlier hidden states to precompute selected KV indices while retaining the real target-layer query for final attention. Tests with RocketKV, Quest, and Loki show no consistent pattern of quality degradation across the evaluated settings. In the primary RocketKV evaluation, ES-RocketKV differs by +0.018 points on LongBench, approximately -0.01 points on RULER, and +0.25 percentage points on representative NIAH/Passkey settings. In a TensorRT-LLM deployment with CPU-offloaded routing, ES-RocketKV improves end-to-end output throughput by up to 12.4% and
    
[^135]: CST：面向网络讨论的轻量级风险分流与基于源文本的证据系统

    CST: Lightweight Risk Triage with Source-Grounded Evidence for Web Discussions

    [https://arxiv.org/abs/2611.01710](https://arxiv.org/abs/2611.01710)

    CST是一个可在普通CPU上运行的轻量级风险分流系统，它通过结合字符模式与固定多语言模型的句子含义来评估消息风险，并将相关消息分组后从各组依次选取，为版主提供保留原始措辞与回复的、基于源文本的证据。

    

    网络社区依靠版主来发现诸如侮辱、仇恨言论和诈骗等有害消息，但没有人能读完全部消息。现有工具为每条消息给出一个风险分数，这能告诉版主该看哪里，却不能告诉他们该读什么或如何核实。然而，含义取决于上下文：同样的回复“我同意。”，既可能支持、也可能反对将某人移出群组。因此，版主需要基于源文本的证据，而不仅仅是分数：即保留原始措辞及其回复的少数关键消息。我们提出了CST，一个轻量级的分流流水线，它先决定该看哪里，再决定该展示什么。它通过将捕捉确切措辞的字符模式与来自固定多语言模型的句子含义相结合来评估风险，因此能够处理多种语言，并可在普通CPU上运行。随后，它将相关消息分组，并从每组中依次选取，使有限的阅读预算能够覆盖讨论的不同部分，并将每条被选中的消息（原文此处截断）

    arXiv:2611.01710v1 Announce Type: new  Abstract: Web communities rely on moderators to catch harmful messages such as insults, hate speech, and scams, but no one can read every message. Existing tools give each message a risk score, which tells moderators where to look but not what to read or how to check it. Yet meaning depends on context: the same reply, "I agree.", can support or oppose removing someone from a group. Moderators therefore need source-grounded evidence, not just scores: a few key messages kept with their original wording and replies. We present CST, a lightweight triage pipeline that first decides where to look, then what to show. It scores risk by pairing character patterns, which capture exact wording, with sentence meaning from a fixed multilingual model, so it handles several languages and runs on an ordinary CPU. It then groups related messages and picks from each group in turn, so a small reading budget covers different parts of a discussion, and links every pic
    
[^136]: EEG-HealthLLM：面向统一异构脑电健康评估的大语言模型分层自适应方法

    EEG-HealthLLM: Hierarchical Adaptation of Large Language Models for Unified Heterogeneous EEG Health Assessment

    [https://arxiv.org/abs/2611.01652](https://arxiv.org/abs/2611.01652)

    提出EEG-HealthLLM框架，通过非对称分层自适应策略——任务特定EEG适配器保留异构时空动态、共享LoRA注入LLM骨干网络——实现统一且参数高效的异构脑电健康评估。

    

    脑电图（EEG）为神经与心理健康评估提供了一种无创手段，然而现有的EEG模型通常针对单个任务开发，难以在异构的信号配置与评估目标之间实现统一。大语言模型（LLMs）为多任务提供了通用接口，但将异构的EEG信号适配到共享的LLM上存在一个关键挑战：如何在实现语言侧参数有效共享的同时保留任务特定的信号特征。为解决这一问题，我们提出了EEG-HealthLLM，一个面向异构EEG健康评估的统一且参数高效的框架。我们引入了一种非对称分层自适应策略：任务特定的EEG适配器保留异构的时空动态特征，同时在LLM骨干网络（如Phi-3）中注入统一的共享LoRA，以捕获跨任务的共享高层评估语义。我们……

    arXiv:2611.01652v1 Announce Type: new  Abstract: Electroencephalography (EEG) provides a non-invasive means for neurological and mental health assessment, yet existing EEG models are typically developed for individual tasks and are difficult to unify across heterogeneous signal configurations and assessment objectives. Large language models (LLMs) offer a common interface for multiple tasks, but adapting heterogeneous EEG signals to a shared LLM raises a key challenge: how to preserve task-specific signal characteristics while enabling effective language-side parameter sharing. To address this problem, we propose EEG-HealthLLM, a unified and parameter-efficient framework for heterogeneous EEG health assessment. We introduce an Asymmetric Hierarchical Adaptation Strategy: task-specific EEG adapters preserve heterogeneous spatio-temporal dynamics, while a unified shared LoRA is injected into the LLM backbone (e.g., Phi-3) to capture shared high-level assessment semantics across tasks. We
    
[^137]: CREDO：面向时序推理智能体的基于证据条件的信用分配

    CREDO: Evidence-Conditioned Credit Assignment for Temporal Reasoning Agents

    [https://arxiv.org/abs/2611.01643](https://arxiv.org/abs/2611.01643)

    提出CREDO方法，通过利用中间证据恢复局部监督并进行阶段级信用再分配，解决了GRPO中轨迹级结果信号无法区分中间决策质量、且全失败组监督信号消失的问题。

    

    时序知识图谱问答（TKGQA）要求语言模型智能体检索证据、选择相关事实并执行多步时序推理。然而，GRPO 中的轨迹级结果信号无法明确区分中间决策的质量：下游任务的失败可能会惩罚本应有用的证据获取行为，而在全部失败的组中，相同的奖励会消除相对结果信号，即便各次尝试在中间进度上存在差异。我们提出了基于证据驱动机会的信用再分配方法（CREDO），该方法利用中间证据来恢复局部监督信号（包括在全部失败的组中），并确定结果监督应作用于何处。CREDO 通过对同一问题的多个采样轨迹之间的比较来分配正向的阶段级信用，并将保留答案的重叠度作为机会代理，把部分结果策略的权重转移到推理和回答阶段。该框架...

    arXiv:2611.01643v1 Announce Type: cross  Abstract: Temporal Knowledge Graph Question Answering (TKGQA) requires language model agents to retrieve evidence, select relevant facts, and perform multi-step temporal reasoning. However, the trajectory-level outcome signal in GRPO does not explicitly distinguish the quality of intermediate decisions: downstream failures can penalize useful evidence acquisition, while identical rewards in all-failed groups eliminate the relative outcome signal despite differences in intermediate progress. We propose Credit Redistribution with Evidence-Driven Opportunities (CREDO), which uses intermediate evidence to recover local supervision, including in all-failed groups, and determine where outcome supervision acts. CREDO assigns positive stage-level credit through comparisons among rollouts of the same question and uses retained answer overlap as an opportunity proxy to shift part of the outcome-policy weight toward reasoning and answering. The framework p
    
[^138]: BanglishDepYT：基于情感保留预处理与领域自适应预训练的孟加拉语原生社交媒体抑郁症检测

    BanglishDepYT: Depression Detection in Native Bengali Social Media with Emotion-Preserving Preprocessing and Domain-Adaptive Pretraining

    [https://arxiv.org/abs/2611.01633](https://arxiv.org/abs/2611.01633)

    该论文发布了首个从YouTube原生收集的孟加拉语抑郁症检测数据集BanglishDepYT，并提出情感保留混合预处理方法EPH，证明预处理方式对非正式社交媒体文本的检测性能影响显著，EPH相比标准清洗方法将SVM F1从0.7062提升至0.8570。

    

    抑郁症影响着南亚数百万孟加拉语使用者，然而自动化检测系统几乎完全依赖于从英语翻译而来的数据集。这类数据集无法捕捉真实痛苦表达中所特有的表情符号、罗马化孟加拉语单词以及拉长的拼写。我们提出了BanglishDepYT，一个从YouTube原生收集的孟加拉语抑郁症数据集，包含191,000条未标注评论和1,000条通过双语双重标注验证的样本。我们提出了EPH（情感保留混合预处理）方法，该方法将表情符号转换为语义标签、保留Banglish词汇、保护否定词并规范化情感性拉长拼写。一个关键发现是，预处理方式对非正式文本的性能有实质性影响：标准清洗方法会将Banglish存活率降为零（SVM F1：0.7062），而EPH保留了情感信号（CESS：0.9996 对比 0.6694），并将SVM F1提升至0.8570，带来了显著的性能改进。

    arXiv:2611.01633v1 Announce Type: new  Abstract: Depression affects millions of Bengali speakers across South Asia, yet automated detection systems have relied almost entirely on datasets translated from English. Such datasets cannot capture the emoticons, romanized Bengali words, and elongated spellings that characterize real expressions of distress. We introduce BanglishDepYT, a Bengali depression dataset collected natively from YouTube, comprising 191,000 unlabeled comments and 1,000 samples verified through dual bilingual annotation. We propose EPH (Emotion-Preserving Hybrid preprocessing), which converts emoticons to semantic tags, preserves Banglish tokens, protects negation words, and normalizes emotional elongation. A key finding is that preprocessing choice substantially affects performance on informal text: standard cleaning reduces the Banglish Survival Rate to zero (SVM F1: 0.7062), while EPH preserves signals (CESS: 0.9996 vs. 0.6694) and raises SVM F1 to 0.8570, an improv
    
[^139]: FinForesight：面向金融预测深度研究系统的诊断性回测基准

    FinForesight: A Diagnostic Backtesting Benchmark for Deep Research Systems in Financial Forecasting

    [https://arxiv.org/abs/2611.01621](https://arxiv.org/abs/2611.01621)

    提出了FinForesight，一个面向金融深度研究系统的诊断性回测基准，通过在中国、美国和香港市场评估估值、收益排名、事件方向与财务基本面四项预测任务，并辅以反事实推理和金融计算两项能力诊断，在防止记忆泄露和“未来信息”偏差的同时，实现对预测能力的及时反馈与历史评分。

    

    预判金融结果需要综合不断演变的证据并在不确定性下进行推理，这使得深度研究系统成为极具前景的预测工具。然而，评估预测能力十分困难，因为历史回测可能奖励的是模型记忆的结果或在预测日期之后才检索到的信息。实时预测虽然避免了对已实现结果的直接访问，但反馈只有在事件尘埃落定后才能获得。为了获得及时的反馈并评估历史得分，我们提出了FinForesight，一个面向金融深度研究系统的诊断性回测基准。该基准覆盖中国、美国和香港市场，评估四项预测任务：估值、收益排名、事件方向以及未来财务基本面。此外，两项能力诊断任务分别评估反事实推理和金融计算能力，其得分与预测表现分开计算。我们与测试框架兼容的系统将信息检索限制在预测日期之前可获得的范围内……

    arXiv:2611.01621v1 Announce Type: new  Abstract: Anticipating financial outcomes requires synthesizing evolving evidence and reasoning under uncertainty, making deep research systems promising tools for forecasting. Yet evaluating forecasting ability is challenging because historical backtests may reward memorized outcomes or information retrieved after the forecast date. Live forecasting avoids direct access to realized outcomes, but feedback arrives only after events resolve. To obtain timely feedback and assess historical scores, we introduce FinForesight, a diagnostic backtesting benchmark for financial deep research systems. Across the Chinese, U.S., and Hong Kong markets, the benchmark evaluates four forecasting tasks: valuation, return ranking, event direction, and future financial fundamentals. Two capability diagnostic tasks assess counterfactual reasoning and financial calculation, scored separately from forecasting performance. Our harness-compatible framework restricts retr
    
[^140]: 基于结构化、文本和视觉特征的LLM增强多模态房价预测

    LLM-Enhanced Multimodal House Price Prediction using Structured, Textual, and Visual Features

    [https://arxiv.org/abs/2611.01615](https://arxiv.org/abs/2611.01615)

    该论文提出了一个融合结构化数据、文本描述和图像的大语言模型增强多模态房价预测框架，利用LLM从文本中提取特定领域的时空和上下文特征，在保持可解释性的同时显著提升预测精度。

    

    准确的房价预测对房地产买家、卖家、中介机构和金融机构至关重要。传统方法主要依赖结构化数据，例如卧室数量、浴室数量、停车位数量等，但忽视了文本描述和房产照片中蕴含的丰富上下文信息。以往用于房价预测的深度学习方法虽然融合了多模态数据，但无法从文本中提取结构化的时空和上下文特征，也无法从图像中提取语义视觉信息。本文提出了一种新颖的多模态框架，通过系统地整合房产的结构化属性、文本描述和视觉图像来解决上述局限性，在保持可解释性的同时实现更高的预测精度。我们使用大语言模型从文本中提取特定领域的时空和上下文特征，并使用无监督视觉语言模型（摘要在此处截断）

    arXiv:2611.01615v1 Announce Type: cross  Abstract: Accurate house price prediction is essential for real estate buyers, sellers, agents, and financial institutions. Traditional methods rely primarily on structured data, e.g., number of bedrooms, bathrooms, parkings, but they overlook rich contextual information embedded in textual descriptions and property photos. Previous deep learning approaches for house price prediction have fused multimodal data but they fall short to extract structured spatio-temporal and contextual features from texts and semantic visual information from images. In this paper, we introduce a novel multimodal framework that addresses these limitations by systematically integrating structured attributes, textual descriptions, and visual images of properties to achieve higher prediction accuracy while maintaining interpretability. We use a large language model to derive domain-specific spatio-temporal and contextual features from text, and an unsupervised vision-la
    
[^141]: Halal-NeSyNet：面向自动化食品成分审计与风险感知清真物流分诊的子句范围神经符号交叉注意力

    Halal-NeSyNet: Clause-Scoped Neuro-Symbolic Cross-Attention for Automated Food Ingredient Auditing and Risk-Aware Halal Logistics Triage

    [https://arxiv.org/abs/2611.01612](https://arxiv.org/abs/2611.01612)

    该论文提出Halal-NeSyNet模型，通过将DistilBERT编码器与规则衍生符号向量进行子句范围的交叉注意力与门控融合，解决清真食品成分审核中跨子句修饰词混淆问题，并构建了包含3,199个条目的基准数据集以实现自动化食品成分审计和风险感知的清真物流分诊。

    

    验证包装食品是否符合清真标准，需要在货物进入认证仓库之前阅读成分声明和E编码添加剂。E471、E422、E441和E920等双来源添加剂属于Mushbooh（存疑）状态，除非在同一成分子句中出现植物来源、微生物来源或认证清真的限定词，因此某一个子句上的植物来源限定词不能用于判定另一个子句中缺乏限定词的添加剂是否合规。我们将违反这一范围约束的失败称为“跨子句修饰词混淆”，并构建了一个包含3,199个条目的基准数据集（277条Open Food Facts记录和2,922条模板生成的子句交换条目，由显式的50条编码规则库而非认证机构进行标注），以及一个包含540个条目的反事实测试集。我们提出Halal-NeSyNet，该方法将DistilBERT编码器与16维规则衍生符号向量通过交叉注意力和门控融合相结合，并使用非对称的假清真风险损失进行训练。在五个随机种子上，文本……（原文摘要至此被截断）

    arXiv:2611.01612v1 Announce Type: new  Abstract: Verifying that a packaged food is Halal requires reading ingredient declarations and E-number additives before goods enter certified warehouses. Dual-origin additives such as E471, E422, E441 and E920 are Mushbooh (doubtful) unless a botanical, microbial or certified-Halal qualifier appears in the same ingredient clause, so a plant-based qualifier on one clause must not clear an unqualified additive in another. We call a failure to respect this scope cross-clause modifier confounding and build a benchmark of 3,199 items (277 Open Food Facts records and 2,922 template-generated clause-swap items, labelled by an explicit 50-code rule base rather than by a certification authority) with a 540-item counterfactual suite. We propose Halal-NeSyNet, which couples a DistilBERT encoder with a 16-dimensional rule-derived symbolic vector through cross-attention and gated fusion, trained with an asymmetric false-Halal risk loss. Over five seeds, text-
    
[^142]: 大语言模型能否将其智能体能力扩展到开放式多轮交互中的目标追求？

    Can Large Language Models Extend Their Agentic Capabilities to Goal Pursuit in Open-Ended Multi-Turn Interaction?

    [https://arxiv.org/abs/2611.01552](https://arxiv.org/abs/2611.01552)

    该论文提出了一种无需训练的工具介导交互（TMI）协议，通过将对话行动表示为工具调用来维持持续的推理-行动-观察循环，显著提升了大语言模型在社交互动、情感支持、说服和谈判等开放式多轮对话中的目标达成能力。

    

    开放式对话中的目标追求需要智能体在连续的交流中根据交互反馈来调整其行动。我们研究了为多步骤工具使用而开发的智能体能力能否迁移到通过模型原生工具接口组织的对话中，以及这会如何改变交互行为。我们定义了工具介导交互作为一种无需训练的协议，它将对话行动表示为工具调用，将对方的响应表示为工具结果，在保留推理和交互历史的同时，维持持续的推理-行动-观察循环。我们使用两个模型和15,678个主实验回合，在社交互动、情感支持、说服和价格谈判等场景中对TMI进行了评估。在两个模型中，与有思考和无思考的对话基线相比，TMI均提高了社交目标达成度、情感支持偏好、捐赠金额和买方谈判空间。一项针对Qwen的消融实验

    arXiv:2611.01552v1 Announce Type: new  Abstract: Goal pursuit in open-ended dialogue requires agents to adapt their actions to interaction feedback across successive exchanges. We investigate whether agentic capabilities developed for multi-step tool use carry over to dialogue organized through the model's native tool interface, and how this changes interaction behavior. We define Tool-Mediated Interaction (TMI) as a training-free protocol that represents dialogue actions as tool calls and counterpart responses as tool results, maintaining a continuous reasoning-action-observation loop with retained reasoning and interaction history. We evaluate TMI across social interaction, emotional support, persuasion, and price negotiation using two models and 15,678 main-experiment episodes. Across both models, TMI improves social goal attainment, emotional-support preference, donation amounts, and buyer negotiation margins over conversational baselines with and without thinking. A Qwen ablation 
    
[^143]: 通过多维视角重新审视大语言模型的同质化

    Revisiting LLM Homogeneity Through a Multidimensional Lens

    [https://arxiv.org/abs/2611.01549](https://arxiv.org/abs/2611.01549)

    该研究提出用涵盖语义、词汇、句法、结构和风格呈现五个维度的25个可解释指标取代单一的嵌入相似度来衡量LLM同质化，发现18个当代模型“用不同的词说相似的话”，且2024至2026年间模型变得与自身更一致而非彼此更相似。

    

    关于大语言模型（LLM）同质化的担忧通常通过嵌入相似度来量化，而这种方法将两个文本简化为一个单一的分数。我们通过涵盖五个维度——语义、词汇、句法、结构以及风格与呈现——的25个可解释指标重新审视同质化问题。一项人类研究表明，感知相似度同时取决于内容（回答说了什么）和形式（如何表达），然而嵌入相似度与人类意义判断的相关性低于句子级匹配（ρ=0.417 对比 0.606），且与形式判断无显著相关性（ρ=0.140）。在对18个当代大语言模型的分析中发现，模型们用不同的词语说着相似的内容：它们共享全局语义和宽泛的语言特征，但在词汇特征、结构模式和交互惯例上仍保持差异。纵观2024年至2026年间发布的各代模型，模型变得更像自己，但并没有变得更像彼此：在显著变化的指标中……（摘要原文在此处截断）

    arXiv:2611.01549v1 Announce Type: new  Abstract: Concerns about LLM homogeneity are typically quantified with embedding similarity, which reduces two texts to a single score. We revisit homogeneity with 25 interpretable metrics spanning five dimensions: semantic, lexical, syntactic, structural, and stylistic & presentational. A human study shows that perceived similarity depends on both content (what responses say) and form (how they say it), yet embedding similarity correlates less with human meaning judgments than sentence-level matching ($\rho=0.417$ vs. $0.606$) and not significantly with form judgments ($\rho=0.140$). Across 18 contemporary LLMs, models say similar things in different words: they share global semantics and broad linguistic profiles but remain distinct in lexical characteristics, structural patterns, and interactional conventions. Across releases from 2024 to 2026, models become more like themselves but not more like one another: among significantly changing metric
    
[^144]: 情感标注中的分歧结构

    The Structure of Disagreement in Emotion Annotation

    [https://arxiv.org/abs/2611.01539](https://arxiv.org/abs/2611.01539)

    该论文通过分析《老人与海》以及BRIGHTER和GoEmotions数据集中的情感标注，首次系统揭示了三种反复出现的分歧结构——情感类别的系统性重叠、标注者在标签宽泛度与类别边界处理上的差异、以及稀疏参与和弃权被误认为分歧。

    

    计算标注者间一致性是自然语言处理中的一项成熟实践，而关注分歧的研究工作越来越多地保留人类判断的分布，而不是将标注简化为单一的金标准标签。然而，保留分布并不能解释标注者为什么会产生分歧。本文研究了情感标注中出现的分歧类型，首先基于对欧内斯特·海明威《老人与海》中1,677个句子级单元的多标签标注进行分析，并在BRIGHTER和GoEmotions数据集的个体级标注上检验这些标注所揭示的模式。在三个数据集中，我们发现了三种主要的反复出现的分歧结构：情感类别存在系统性的重叠；标注者不仅在其应用标签的宽泛程度上存在差异，在对特定类别边界的处理方式上也有所不同；同时，稀疏的参与和明确的弃权可能会被误认为分歧。在评估情感强度的地方，关于……（原文截断）

    arXiv:2611.01539v1 Announce Type: new  Abstract: Calculating inter-annotator agreement is an established practice in NLP, and disagreement-aware work increasingly retains the distribution of human judgments rather than reducing annotations to a single gold label. Keeping the distribution, however, does not explain why annotators disagree. This paper examines the types of disagreement that occur in emotion annotation, starting from multilabel annotations of 1,677 sentence-level units from Ernest Hemingway's The Old Man and the Sea and testing the patterns they reveal against individual-level annotations from BRIGHTER and GoEmotions. Across the three datasets, we find three main recurring disagreement structures. Emotion categories overlap systematically; annotators differ both in how broadly they apply labels and in how they treat particular category boundaries, while sparse participation and explicit abstention can look like disagreement. Where intensity is rated, disagreement over whe
    
[^145]: 以模型读取的方式书写事实：破解两跳诅咒

    Write Facts the Way Models Read Them: Lifting the Two-Hop Curse

    [https://arxiv.org/abs/2611.01519](https://arxiv.org/abs/2611.01519)

    论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。

    

    arXiv:2611.01519v2 公告类型：新 摘要：通过微调学习新事实的语言模型能够复述该事实，却无法用它进行推理：教会模型某个人是新国家公民之后，它仍然无法在不借助思维链的情况下说出这个人所在国家的国家元首是谁。这种“两跳诅咒”通常被视为推理能力缺陷，而最强的知识编辑方法则通过合成推理数据来解决它。我们证明：推理能力是完好的，错的是事实的书写方式。微调把新事实写成一个“要说出口的答案”，而第二跳推理从中间层开始就把桥接短语（“那个人的国家”）当作实体来读取：在相应层将模型自身对该实体的表征修补进桥接短语，可以基本恢复第二跳推理；然而微调之后，桥接短语与该新实体的接近程度并不比编辑前更高。因此我们提出“读取等价书写”（Legible）：用一个损失项让桥接短语在第二跳读取它的那一层（摘要在此处截断）

    arXiv:2611.01519v2 Announce Type: new  Abstract: A language model that learns a new fact by fine-tuning can repeat it but cannot reason with it: taught that a person is a citizen of a new country, it still cannot say, without chain-of-thought, who the head of state of that person's country is. This two-hop curse is widely treated as a reasoning deficit, and the strongest knowledge editors answer it with synthesized reasoning data. We show that the reasoning is intact and the writing is wrong. Fine-tuning writes a new fact as an answer to be said, whereas the second hop reads the bridge ("that person's country") as an entity, from the middle layers on: patching the model's own representation of the entity into the bridge at those layers largely restores the second hop, yet after fine-tuning the bridge there is no closer to the new entity than before the edit. We therefore propose read-equivalent writing (Legible): one loss term makes the bridge phrase, at the layer where the second hop 
    
[^146]: VLMs 朗读出来：识别与使用视觉文本之间的差距

    VLMs Read Aloud: The Gap Between Recognizing and Using Visual Text

    [https://arxiv.org/abs/2611.01517](https://arxiv.org/abs/2611.01517)

    该论文发现视觉语言模型存在“读-用差距”——即使能逐字转录图像中的文本也无法据此答题，而把自己的转录文本反馈给自己即可近乎完全恢复答题能力，且识别与使用视觉文本在分辨率要求和模型规模扩展上表现不同。

    

    视觉文本压缩将长上下文渲染成图像，其工程设计围绕视觉语言模型（VLM）能从图像中恢复多少文本展开。然而，恢复出来的文本并不等于可用的文本。VLM 可以逐字转录图像中的文字，却仍然无法据此回答问题；而如果把它们自己转录的文字反馈给它们，其答题表现几乎与直接基于原始文本时一样好。我们将这种差异称为“读-用差距”。识别与使用视觉文本对分辨率的要求不同：Qwen3-VL-8B 的识别能力在字符高度为 15 像素时即趋于饱和，而使用能力要到 24 像素才能跟上；并且将模型从 2B 扩展到 32B 提升了识别能力，却没有缩小这一差距。把证据用文字写出来可以消除这一差距：具备思考能力的 VLM 会在思考过程中复述证据，而且它在没有看到问题的情况下写出的转录文本，就能恢复思考所带来的大部分增益。在一项预注册的留出测试中，该差距可以预测八个 VLM 从“自己的文字”相比“像素”所能获得的收益大小。

    arXiv:2611.01517v2 Announce Type: cross  Abstract: Visual-text compression renders long contexts as images and is engineered around how much of the text a vision-language model (VLM) can recover. Yet recovered text is not usable text. VLMs can transcribe the text in an image word for word and still fail to answer from it; given back their own transcript, they answer nearly as well as from the original text. We call this difference the read-use gap. Recognizing and using visual text have different resolution requirements: Qwen3-VL-8B's recognition saturates at 15-pixel characters while use catches up only at 24 pixels, and scaling from 2B to 32B raises recognition without narrowing the gap. Putting the evidence into words closes it: thinking VLMs recite the evidence in their thinking, and their own transcript, written without seeing the question, recovers most of what thinking adds. In a preregistered held-out test, the gap ranks how much eight VLMs gain from their own words over pixels
    
[^147]: 少扫描，多引导：利用激活引导几何降低超参数搜索成本

    Sweep Less, Steer More: Using Activation Steering Geometry for Reduced Hyperparameter Search

    [https://arxiv.org/abs/2611.01496](https://arxiv.org/abs/2611.01496)

    本文提出利用计算成本低廉的可辨别性指数 d′ 作为几何度量，可靠地识别激活引导的最佳干预层，从而免除逐层超参数扫描，大幅降低表示引导方法的部署成本。

    

    像均值差（DIM）这样的表示引导方法为控制大语言模型行为提供了一种轻量级途径，但其有效性在不同概念、数据集和层之间差异很大，若不进行穷尽式超参数扫描，就难以实现有原则的部署，成本高昂。我们基于可辨别性指数 d′——一种计算成本低廉的几何度量，用于衡量对比训练激活沿引导方向的分离程度。在3个模型上的5种行为实验中，我们展示了超越其作为数据集层面可引导性预测器这一既定角色的新用途。首先，在多选题问答（MCQA）设置中，我们证明 d′ 能够可靠地识别出适合进行引导的高效层，从而免除了逐层扫描的需要。其次，我们将这一分析扩展到开放式多token生成任务，发现 d′ 在不同设置下的层选择中依然有效。最后，我们测试了几何……（摘要截断）

    arXiv:2611.01496v1 Announce Type: cross  Abstract: Representation steering methods like difference-in-means (DIM) offer a lightweight approach to controlling LLM behavior, but their effectiveness varies widely across concepts, datasets, and layers, making principled deployment costly without exhaustive hyperparameter sweeps. We build on the discriminability index $d'$, a compute-cheap geometric measure of how well contrastive training activations separate along the steering direction. Across five behaviors on 3 models, we demonstrate uses beyond its established role as a dataset-level steerability predictor. First, in the multiple-choice question answering (MCQA) setting, we show that $d'$ reliably identifies performant layers at which to steer, eliminating the need for a layer-wise sweep. Second, we extend this analysis to open-ended multi-token generation, and find that $d'$ remains successful for layer selection across different settings. Lastly, we test whether the position of a ge
    
[^148]: TopoSteer：面向大语言模型激活引导的拓扑学层选择方法

    TopoSteer: Topological Layer Selection for Activation Steering in LLMs

    [https://arxiv.org/abs/2611.01493](https://arxiv.org/abs/2611.01493)

    TopoSteer 提出了一种基于拓扑数据分析（H_0 持续度）的大语言模型激活引导层选择方法，通过度量逐样本位移云的延展范围来选出最有效的引导层，弥补了以往方法只看激活可分离性而不实际测试引导效果的不足。

    

    激活引导（activation steering）通过在推理时向大语言模型的内部表示中添加一个向量来对模型进行控制。现有研究大多关注如何构建这个向量，而对“应该把它加到哪一层”的关注却少得多——尽管在不当选择的层上进行引导，效果可能比不引导还要糟糕。以往的层选择器在选层时从不实际引导模型进行测试：它们只是评估目标行为在该层的可分离程度，以及各个样本沿引导方向的一致性。我们认为，真正重要的既不是两个激活分布云的可分离性，也不是各样本位移与其均值的对齐程度，而是均值差分引导所平均的“逐样本位移云”的延展范围。我们提出了一种基于拓扑的层选择方法，称为 TopoSteer，它通过位移云的总 H_0 持续度（persistence）来为各层打分，该持续度由位移云的欧几里得……（摘要在此处截断）

    arXiv:2611.01493v1 Announce Type: new  Abstract: Activation steering controls large language models at inference time by adding a vector to their internal representations. Most work asks how to build that vector, and far less attention has gone to where to add it, even though steering at a poorly chosen layer can be worse than not steering at all. Prior selectors choose a layer without ever steering the model to test it: they score how separable the target behavior is at that layer, and how consistently the individual examples point along the steering direction. We argue that what matters is not the separability of the two activation clouds, nor the alignment of individual displacements with their mean, but the extent of the cloud of per-example displacements that mean-difference steering averages. We propose a topology-based layer selection method, called TopoSteer, which scores a layer by the total H_0 persistence of its displacement cloud, computed as the length of the cloud's Eucli
    
[^149]: 通过信念估计应对在线医疗咨询中的不规则状态转移

    Accounting for Irregular State Transitions through Belief Estimation in Online Medical Consultation

    [https://arxiv.org/abs/2611.01474](https://arxiv.org/abs/2611.01474)

    该论文通过实证研究发现真实在线医疗咨询具有上下文连续性低、不可预测性高的特点，并提出受部分可观测马尔可夫决策过程（POMDP）启发的PRISM框架，通过将潜在状态信念估计与动作选择解耦来应对对话轨迹漂移，从而提升临床分诊能力。

    

    在线医疗咨询虽然拓宽了医疗服务的可及性，但患者未经提示的主动披露和话题转换常常打断临床医生的问诊过程，使临床状态被逐步揭示。我们使用现有评估指标以及本工作专门设计的任务特定交互指标来刻画这些动态特性。通过一项涵盖16个医疗对话数据集的实证研究，我们发现真实的在线交互相较于离线和模拟基准表现出显著更低的上下文连续性和更高的不可预测性。我们提出了PRISM（Policy Reasoning with Inferred States for Medicine，基于推断状态的医学策略推理），一个受部分可观测马尔可夫决策过程启发的框架，它将潜在状态估计与动作选择解耦，以应对漂移的对话轨迹。具体而言，PRISM将临床分诊分解为信念估计、策略优化和回复生成三个环节，并通过维护一个不断演化的后验信念分布……

    arXiv:2611.01474v1 Announce Type: new  Abstract: While online medical consultation broadens access, unprompted patient disclosures and topic shifts frequently disrupt clinician inquiry, leaving clinical states progressively revealed. We characterize these dynamics using both existing evaluation metrics and task-specific interaction metrics designed in this work. Through an empirical study across 16 medical dialogue datasets, we show that authentic online interactions exhibit substantially lower contextual continuity and higher unpredictability than offline and simulated benchmarks. We propose PRISM (Policy Reasoning with Inferred States for Medicine), a framework inspired by partially observable Markov decision processes that decouples latent state estimation from action selection to address drifting dialogue trajectories. Specifically, PRISM factorizes clinical triage into belief estimation, policy optimization, and response realization. By maintaining an evolving posterior belief ove
    
[^150]: 看、想、用：诊断视觉语言模型中的感知与推理失败

    See, Think, Use: Diagnosing Perception and Reasoning Failures in VLMs

    [https://arxiv.org/abs/2611.01467](https://arxiv.org/abs/2611.01467)

    该论文提出“看、想、用”协议，通过定位探针、文本替换探针和反事实图像对，在无需访问模型权重的情况下，将视觉语言模型的每个答案归因于感知、推理或感知使用，从而精准定位应改进的环节并指导测试时计算的分配。

    

    我们提出了“看、想、用”，这是一种能够将视觉语言模型的每个答案追溯到感知、推理或对感知的使用三个阶段的诊断协议。该诊断可以告诉开发者应当改进哪个阶段，并将测试时计算引导到失败的环节。基准测试只对最终答案评分，基于图像得出的答案与猜测或记忆的先验获得同样的分数。该协议在每个题目上只需向模型额外提出两个问题，且无需访问模型权重。定位探针要求模型为每个实例框出边界框，以测量“看”的能力；文本替换探针让模型基于文字描述来回答问题，以测量“想”的能力。“用”衡量的是在反事实图像对上，答案遵循模型自身感知的比率。该比率无需答案标签，因为问题和其先验在两张图像上是相同的。在两个模型家族和六个数据集上，该协议表明有效的改进手段取决于任务类型。

    arXiv:2611.01467v1 Announce Type: cross  Abstract: We present See, Think, Use, a protocol that traces each answer of a vision-language model to perception, reasoning or the use of perception. The diagnosis shows a developer which stage to improve and directs test-time compute to the stage that fails. Benchmarks score the final answer, and an answer that rests on the image earns the same credit as a guess or a memorized prior. The protocol asks the model two extra questions per item and needs no access to its weights. A localization probe asks for a box around every instance and measures seeing, and a text-replacement probe answers from a description and measures thinking. Use is the rate at which the answer follows the model's own perception across counterfactual image pairs. The rate needs no answer labels, because the question and its prior are the same on both images. On two model families and six datasets the protocol shows that the knob that helps follows the task family. Answers 
    
[^151]: 大型语言模型智能体能通过观看教程视频学习吗？

    Can Large Language Model Agents Learn by Watching Tutorial Videos?

    [https://arxiv.org/abs/2611.01460](https://arxiv.org/abs/2611.01460)

    本文提出教程智能体学习（TAL）这一非参数化持续学习范式，通过两阶段对抗过程将教程视频整合为结构化的“教程书”记忆并借助目录树检索，使LLM智能体无需参数更新即可像人类一样从海量教程视频中持续学习。

    

    人类能够持续地从精心制作的材料（如书籍和教程视频）中获取知识，并将这些理解运用到未来的任务中。每天都有数百万条此类教学视频上传到网上，然而大型语言模型（LLM）智能体仍然无法利用这一不断增长的人类教学资源。它们的能力在训练完成后便被冻结，而大规模的参数更新成本又过于高昂。这引出了一个核心问题：LLM智能体能否在不进行参数更新的情况下，从这一庞大的教学资源中学习？我们提出了教程智能体学习，这是一种非参数化的持续学习范式，它通过一个两阶段对抗过程将教程视频整合为持久、结构化的“教程书”记忆：该过程首先挖掘以视频为依据的高难度问题，再利用这些问题来完善记忆。在推理阶段，TAL通过目录（TOC）树进行检索，将检索过程直接与模型的……（摘要原文在此处截断）

    arXiv:2611.01460v1 Announce Type: cross  Abstract: Humans continuously acquire knowledge from carefully crafted materials such as books and tutorial videos, carrying that understanding into future tasks. Millions of such instructional videos are uploaded online every day, yet LLM agents remain unable to exploit this ever-expanding wealth of human teaching. Their capabilities are frozen after training, while parameter updates at scale remain prohibitively expensive. This raises the central question: can LLM agents learn from this vast instructional resource without parameter updates? We introduce Tutorial Agentic Learning (TAL), a nonparametric continual-learning paradigm that consolidates tutorial videos into a persistent, structured tutorialbook memory through a two-stage adversarial process that harvests hard, video-grounded questions and uses them to refine the memory. At inference time, TAL retrieves through a Table-of-Contents (TOC) tree, tying retrieval directly to the model's re
    
[^152]: 智能体何时应刷新？面向高效长程推理的自适应索引复用

    When Should an Agent Refresh? Adaptive Index Reuse for Efficient Long-Horizon Reasoning

    [https://arxiv.org/abs/2611.01429](https://arxiv.org/abs/2611.01429)

    该论文提出自适应索引复用（AIR），通过一个轻量控制器根据智能体回合的各阶段片段和最新工具事件，动态决定稀疏注意力索引器需要刷新的层数，并在预期计算预算下利用最终任务奖励进行学习，从而在长程推理中更高效地平衡计算开销与注意力选择的准确性。

    

    语言智能体在每一步都会增长的上下文中阅读代码、工具输出和对话。动态稀疏注意力机制允许一个小型索引器为每个查询选择其应关注的早期token，但索引器仍需在每一层对整个前缀进行评分。诸如IndexCache等跨层复用方法只在部分层重新运行索引器，且对所有输入采用同一个固定模式。而在智能体场景中，当错误回溯（traceback）或测试结果出现时，相关证据可能突然改变，因此在某些步骤中刷新选择比其他步骤更为重要。我们提出了自适应索引复用，它保留了IndexCache的执行方式，并为智能体回合的每个片段学习需要刷新选择的层数。一个小型控制器读取来自已完成计算的信号和最新的工具事件，从嵌套模式库中选择一种模式，并在预期计算预算下通过最终任务奖励进行学习。在使用128K上下文的Qwen3-4B-Instruct-2507模型上，AIR提升了……（原文此处截断）

    arXiv:2611.01429v1 Announce Type: cross  Abstract: Language agents read code, tool output, and dialogue in contexts that grow at every step. Dynamic sparse attention lets a small indexer choose the earlier tokens each query attends to, but the indexer still scores the whole prefix at every layer. Cross-layer reuse methods such as IndexCache rerun the indexer at only some layers, with one fixed pattern for every input. In an agent, the relevant evidence can change abruptly when a traceback or test result arrives, so fresh selections matter more at some steps than at others. We propose Adaptive Index Reuse (AIR), which keeps IndexCache's execution and learns, for each segment of an agent episode, how many layers refresh their selection. A small controller reads signals from completed computation and the latest tool event, chooses a pattern from a nested library, and learns from terminal task reward under an expected compute budget. On Qwen3-4B-Instruct-2507 with a 128K context, AIR impro
    
[^153]: 迈向细粒度且鲁棒的人机共著检测

    Towards Fine-grained and Robust Detection of Human-AI Co-authorship

    [https://arxiv.org/abs/2611.01427](https://arxiv.org/abs/2611.01427)

    该论文提出一个离散类别的文本溯源检测框架，将人机共著细分为不同级别的AI参与程度（如表面润色与话语结构更新），揭示了对抗性改写模糊检测决策边界的机制，并通过数据增强提升检测器的鲁棒性。

    

    arXiv:2611.01427v1 公告类型：new 摘要：大语言模型（LLMs）的广泛采用正在改变文本内容的创作方式，使文本溯源变得更加复杂，并引发了对虚假信息、剽窃和知识产权问题的担忧。尽管此前在AI生成文本检测方面已取得进展，但一维的人机共著评分缺乏与“AI参与度”直接相关的信号。此外，目前尚不清楚对抗性改写会如何扭曲检测信号。在这项工作中，我们构建了一个离散类别的文本溯源检测框架。该框架能够将现实中的共著场景抽象为不同级别的AI参与程度，从而区分表面层面的AI润色与话语结构层面的AI更新。该框架还揭示了对抗性改写如何模糊共著检测器决策边界的机制。为提升检测器的鲁棒性，我们利用对抗……来增强训练数据（原文摘要在此处被截断）。

    arXiv:2611.01427v1 Announce Type: new  Abstract: The widespread adoption of large language models (LLMs) is reshaping how textual content is produced, complicating text provenance, and raising concerns for misinformation, plagiarism, and intellectual property. While prior work on AI-generated text detection has progressed, one-dimensional human-AI co-authorship scores lack signals directly relevant to ``AI involvement''. It remains unclear how adversarial paraphrasing distorts detection signals. In this work, we formulate a discrete-class text provenance detection framework. This framework allows an abstraction of realistic co-authorship scenarios into distinct degrees of AI involvement, differentiating the AI polishing on the surface level from the AI updating the discourse structures. This framework reveals the mechanisms of how adversarial paraphrasing blurs the decision boundaries for co-authorship detectors. To improve robustness of the detectors, we augment training data with adv
    
[^154]: FlyWheel：专为TPU打造的注意力内核

    FlyWheel: Attention Kernels Built for the TPU

    [https://arxiv.org/abs/2611.01426](https://arxiv.org/abs/2611.01426)

    FlyWheel是一组专为TPU设计的快速注意力内核，通过重新表述softmax与三角矩阵求逆以减少向量计算、协调数据布局提升矩阵操作数复用并重叠独立计算，在softmax注意力上实现最高4.14倍加速并达到v6e峰值BF16吞吐量的81%，同时将Gated DeltaNet预填充加速最高3.61倍。

    

    张量处理单元（TPU）上的注意力内核常常因为重复的矩阵操作数加载、向量密集型计算以及导致矩阵单元（MXU）闲置的依赖关系而无法充分利用其矩阵单元。我们提出了FlyWheel，这是一组面向softmax注意力和线性注意力的快速TPU内核。FlyWheel重新表述了softmax和三角矩阵求逆以减少向量计算工作，同时协调数据布局以提高矩阵操作数的复用率，进而通过重叠相互独立的计算来隐藏执行延迟。我们将这些技术应用于TPU v6e和v7x上的softmax注意力和Gated DeltaNet。FlyWheel将softmax注意力的速度相比最先进的RPA内核提升最高4.14倍，相比Splash Attention提升最高1.80倍，在v6e上达到峰值BF16吞吐量的81%。它还将Gated DeltaNet的预填充（prefill）速度提升最高3.61倍。集成到为Qwen3.8-27B提供服务的vLLM中后，FlyWheel将预填充吞吐量提升最高1.56倍，同时保持……（原文摘要在此处截断）

    arXiv:2611.01426v1 Announce Type: new  Abstract: Attention kernels on Tensor Processing Units (TPUs) often underutilize their matrix units (MXUs) because of repeated matrix operand loads, vector-intensive computation, and dependencies that leave MXUs idle. We present FlyWheel, a set of fast TPU kernels for both softmax and linear attention. FlyWheel reformulates softmax and triangular inversion to reduce vector work, while coordinating data layouts to improve matrix operand reuse. It then overlaps independent computation to hide execution latency. We apply these techniques to both softmax attention and Gated DeltaNet on TPU v6e and v7x. FlyWheel speeds up softmax attention by up to $4.14\times$ over the state-of-the-art RPA kernel and $1.80\times$ over Splash Attention, reaching 81% of peak BF16 throughput on v6e. It also speeds up Gated DeltaNet prefill by up to $3.61\times$. Integrated into vLLM serving Qwen3.8-27B, FlyWheel improves prefill throughput by up to $1.56\times$ while mai
    
[^155]: 潜在框架：将智能体行为编译为解码块后的激活偏移

    Latent Harness: Compiling Agent Behaviors into Post-Block Activation Shifts

    [https://arxiv.org/abs/2611.01415](https://arxiv.org/abs/2611.01415)

    本文提出 Latent Harness，将外部框架条件或行为指令蒸馏为施加在冻结模型每个解码器块输出上的秩8激活偏移（仅0.03-0.05%参数），使智能体行为无需框架文本和额外token即可内化进模型，并在10个预注册设置中的8个恢复源框架收益。

    

    框架工程与自我进化智能体能够快速提升语言模型智能体的能力，但这些收益存在于模型之外。框架在每次调用时都需要重新发送并重新支付成本，而将行为内化的标准途径是权重更新——这种方式迭代成本高昂，且一旦合并便固定不变。我们提出改为对框架进行“编译”。Latent Harness 将外部条件蒸馏为一个秩为 8 的激活偏移，该偏移被施加于冻结骨干网络的每个解码器块的输出处，仅涉及 0.03-0.05% 的参数。外部条件既可以是显式的过程级行为指令，也可以是智能体框架的行为契约。在推理时，编译后的产物不携带任何框架文本，也不增加任何 token，并暴露三个运行时控制项：注入强度、注入阶段和偏移空间混合。编译后的行为在 10 个预先注册的设置中的 8 个里恢复了源框架的收益。在显式来源每次调用代价最高的场景……

    arXiv:2611.01415v1 Announce Type: new  Abstract: Harness engineering and self-evolving agents improve language-model agents quickly, yet their gains live outside the model. The harness is re-sent and re-paid on every call, and the standard route to internalization is a weight update that is costly to iterate and fixed once merged. We propose compiling the harness instead. Latent Harness distills an external condition into a rank-8 activation shift. The shift is applied at the output of every decoder block of a frozen backbone, at 0.03-0.05% of parameters. The condition is either an explicit process-level behavior instruction or the behavioral contract of an agentic harness. At inference the compiled bundle carries no harness text and adds no tokens. It exposes three runtime controls for injection strength, injection phase, and shift-space mixing. Compiled behaviors recover the source-harness gain in 8 of 10 preregistered settings. Where the explicit source pays the heaviest per-call to
    
[^156]: RAG-TRACE：追踪AI生成心电图报告中的测量证据与诊断完整性

    RAG-TRACE: Tracing Measurement Evidence and Diagnostic Completeness in AI-Generated ECG Reports

    [https://arxiv.org/abs/2611.01409](https://arxiv.org/abs/2611.01409)

    该研究提出RAG-TRACE方法，通过追踪测量证据发现关键词检索会在AI生成的心电图报告中引入大量缺乏患者特异性测量支持的不可靠诊断标准，而确定性数值验证规则能有效撤销这些错误标准应用，同时保留有真实依据的诊断。

    

    目的：检索增强生成（RAG）可能引入缺乏患者特异性测量支持的心电图（ECG）诊断标准。本研究评估了AI生成的心电图报告中检索所得诊断标准的可靠性。方法：我们分析了1,083条经人工验证的PTB-XL患者记录，使用包含47张标准卡的规则库，其中33张没有数值验证规则。语言模型在有检索和无检索两种条件下生成结构化报告。标准适用性使用Marquette 12SL测量值进行评估，并使用Uni-G进行独立评估，同时与专家验证的诊断标签进行比较。结果：关键词检索在46.8%的GPT-4.1-mini报告中引入了无法检测或与测量结果相矛盾的诊断标准。一条确定性验证规则撤销了647项标准应用，其中包括639项无数值检查依据的应用，同时保留了269项独立获得Uni-G支持的应用中的268项。限制：（摘要在此处截断）

    arXiv:2611.01409v1 Announce Type: new  Abstract: Purpose: Retrieval-augmented generation (RAG) can introduce electrocardiogram (ECG) criteria that lack patient-specific measurement support. This study evaluates the reliability of retrieved diagnostic criteria in AI-generated ECG reports. Methods: We analyzed 1,083 human-validated PTB-XL patient records using a library of 47 criteria cards, including 33 without numeric verification rules. Language models generated structured reports with and without retrieval. Criterion applicability was evaluated using Marquette 12SL measurements, with independent assessment using Uni-G and comparison against expert-validated diagnostic labels. Results: Keyword retrieval introduced untestable or measurement-disconfirmed criteria in 46.8% of GPT-4.1-mini reports. A deterministic verification rule withdrew 647 criterion applications, including 639 without numeric checks, while preserving 268 of 269 applications independently supported by Uni-G. Restricti
    
[^157]: 从实践中学习：从经验证的轨迹中自主编译与策划智能体技能

    Learning by Doing: Autonomously Compiling and Curating Agent Skills from Verified Traces

    [https://arxiv.org/abs/2611.01386](https://arxiv.org/abs/2611.01386)

    Zep Agent Skills是一个生产级服务，它仅以经过验证的任务尝试为证据，将智能体运行轨迹自主编译为带版本管理的程序性技能记忆，并通过准入控制与经验证的使用结果来治理技能的激活、分发与搜索排名。

    

    面向生产级LLM智能体的记忆系统大多存储语义记忆（关于用户和领域的事实）或情景记忆（先前的交互记录）。诸如Trace2Skill、WikiSkill和SkillRevise等研究系统从智能体运行中学习程序性知识，将其打包为技能，并在基准测试中进行评估。我们介绍了Zep Agent Skills，这是一个为已部署智能体提供程序性记忆的生产服务。该服务仅使用经过验证的任务尝试作为证据，为技能的每个字段引用其支持的事件，并对每个技能版本的准入和分发进行治理。该服务将每次任务尝试记录为一条Trace（轨迹），包含有序事件、结果以及验证强度。一个具有独立Reflector（反思者）和Curator（策展者）角色的编译器将经验证的轨迹转化为带版本管理的技能。扫描机制和准入门槛控制哪些候选技能可以被激活，而经验证的使用结果会调整每个技能的搜索排名。Agent Skills格式是一个开放……

    arXiv:2611.01386v1 Announce Type: cross  Abstract: Memory systems for production LLM agents mostly store semantic memory (facts about users and domains) or episodic memory (earlier interactions). Research systems such as Trace2Skill, WikiSkill, and SkillRevise learn procedures from agent runs, package them as Skills, and evaluate them on benchmarks. We describe Zep Agent Skills, a production service that gives deployed agents this procedural memory. The service uses only verified attempts as evidence, cites the supporting events for each field of a Skill, and governs the admission and distribution of each Skill version. The service records each task attempt as a Trace with ordered events, an outcome, and a verification strength. A compiler with separate Reflector and Curator roles turns verified Traces into versioned Skills. Scans and admission gates control which candidates become active, and verified use outcomes change the search rank of each Skill. The Agent Skills format is an ope
    
[^158]: 基于稀疏自编码器转向的信息论设计

    Information-Theoretic Design of Sparse Autoencoder-Based Steering

    [https://arxiv.org/abs/2611.01378](https://arxiv.org/abs/2611.01378)

    该论文提出了一个信息论框架，利用经验特征触发分布之间的Jensen-Shannon散度来分析SAE激活并构建有效的转向向量，从而更系统地设计对大语言模型行为的干预。

    

    基于稀疏自编码器（SAE）的激活转向是一种轻量级方法，通过在推理过程中向隐藏状态添加小向量来修改大语言模型（LLM）的行为。然而，设计有效的转向干预仍然具有挑战性。在这项工作中，我们开发了一个信息论框架，通过经验特征触发分布来分析SAE激活。该框架的核心对象是这些触发分布之间的Jensen-Shannon散度（JSD），它量化了生成或干预状态偏离由数据集诱导的目标分布的程度。基于虚拟移位状态的解析散度响应，我们构建了朝向目标触发分布的转向向量。同样的散度还可用于监测与另一个经验分布（例如通用的有效语言分布）之间的偏差。我们展示了该转向方法的有效性……（摘要原文在此处被截断）

    arXiv:2611.01378v1 Announce Type: new  Abstract: Sparse autoencoder (SAE)-based activation steering is a lightweight approach for modifying the behavior of large language models (LLMs) by adding small vectors to hidden states during inference. However, designing effective steering interventions remains challenging. In this work, we develop an information-theoretic framework for analyzing SAE activations through empirical feature-firing distributions. The central object of the framework is the Jensen--Shannon divergence (JSD) between such firing distributions, which quantifies how far a generated or intervened state deviates from a dataset-induced target distribution. Based on an analytic divergence response for virtually shifted states, we construct steering vectors toward the target firing distribution. The same divergence can also be used to monitor deviation from another empirical distribution, such as a general valid-language distribution. We demonstrate the effectiveness of the st
    
[^159]: XDecideBench：跨语言评估类型化决策模型与置信度策略

    XDecideBench: Evaluating Typed Decision Models and Confidence Policies Across Languages

    [https://arxiv.org/abs/2611.01370](https://arxiv.org/abs/2611.01370)

    该论文提出XDecideBench，一个覆盖122种语言的大规模基准，用于评估类型化决策模型与置信度策略，揭示了模型排名随语言和套件显著变化，且针对类型化决策微调的开源模型在网络文本较少的低资源语言中带来更大的准确率提升。

    

    应用程序使用类型化决策模型对多种语言的文本进行路由与审核，且通常只对置信度超过阈值的决策采取行动。我们提出XDecideBench，这是一个涵盖122种语言、9个测试套件的基准，包含88,425篇文本和153,875个类型化问题。所有套件共享统一的评分契约，我们分别报告对齐项、本地标注项和母语原生项。模型排名随套件而变化。事后选出的英语最优模型在约四分之一的套件-语言组合中显著落败，并且在缅甸语Belebele上远落后于最佳模型。三个针对类型化决策微调的开源模型，在其所基于的Qwen模型的全部十二个开发语言中均胜出。当Qwen模型在本地运行时，一个预先注册的留出验证队列证实，在网络文本最少的四种语言中的准确率提升大于在网络文本最多的六种语言中。在这两组语言中……（摘要原文在此处截断）

    arXiv:2611.01370v1 Announce Type: new  Abstract: Applications use Typed Decision Models to route and moderate text in many languages, and they often act only on decisions above a confidence threshold. We introduce XDecideBench, a benchmark of nine suites in 122 languages with 88,425 texts and 153,875 typed questions. All suites share one scoring contract, and we report aligned, locally labelled and native items separately. Model rankings change from suite to suite. The model that is best in English, chosen in hindsight, loses significantly in about a quarter of the suite-language cells, and on Burmese Belebele it falls far behind the best model. Three open models fine-tuned for typed decisions beat the Qwen models they were built from in all twelve development languages. When the Qwen models are run locally, a pre-registered held-out cohort confirms that the accuracy gain is larger in the four languages with the least web text than in the six with the most. In both sets of languages, h
    
[^160]: 将经典可读性公式解释为惊讶度

    Interpreting Classical Readability Formulae as Surprisal

    [https://arxiv.org/abs/2611.01359](https://arxiv.org/abs/2611.01359)

    该论文首次从理论上证明，Flesch-Kincaid等经典可读性公式的系数可被解释为惊讶度——在无记忆模型下，这类公式等价于逐词与逐句交叉熵之和，借助Zipf定律并作为参考文本处的线性化，每个系数可分解为每nat年级数尺度、Zipf斜率和参考句子长度，从而使“年级”成为惊讶度的一个量。

    

    arXiv:2611.01359v1 公告类型：新 摘要：惊讶度与阅读时间密切相关，被广泛用作阅读认知科学中衡量加工难度的指标。诸如Flesch-Kincaid年级水平这类经典可读性公式——它按照预期读者所需达到的美国学校年级为文本评分——是实际中使用最广泛的文本难度度量方法，但其系数是无量纲的回归常数，从未与惊讶度建立起联系。我们从理论上证明，这些系数可以被解释为惊讶度：每一类具有经典加法形式的公式，都是在无记忆模型下逐词交叉熵与逐句交叉熵之和；一旦该模型通过Zipf定律与语言绑定，且该公式被理解为其在某个参考文本处的线性化，则每个系数便可以分解为一个以“每nat年级数”为单位的尺度因子、一个Zipf斜率以及一个参考句子长度。由此，年级本身成为了一个惊讶度意义上的量。由于该公式……

    arXiv:2611.01359v1 Announce Type: new  Abstract: Surprisal correlates closely with reading time and is widely used in the cognitive science of reading as a measure of processing difficulty. Classical readability formulae such as the Flesch-Kincaid Grade Level, which scores a text by the US school grade expected to read it, are the most widely used practical measures of text difficulty, but their coefficients are unitless regression constants and have never been related to surprisal. We theoretically show that these coefficients can be interpreted as surprisal: every formula of the classical additive shape is a sum of per-word and per-sentence cross-entropies under a memoryless model, and once that model is tied to the language through Zipf's law and the formula is read as its linearisation at a reference text, each coefficient factorises into a scale in grades per nat, a Zipf slope and a reference sentence length. A grade thereby becomes a quantity in surprisal. Because the formula out
    
[^161]: 共同犯错：用户模拟中的共享偏见

    Wrong Together: Shared Bias in User Simulation

    [https://arxiv.org/abs/2611.01354](https://arxiv.org/abs/2611.01354)

    该研究发现模拟器之间的分歧可以作为识别不可靠估计的信号（对群体间差异的预测效果更好，AUC达0.87），但模拟器共享的偏见对该信号不可见，不过当共享错误同等地影响两个被比较的群体时会在比较中相互抵消。

    

    用户模拟器代替人类参与者生成回答，使研究人员能够预先估计研究结果。为了在收集人类回答之前使用这些估计，研究人员需要一种评估其可靠性的方法。我们测试了模拟器之间的分歧能否识别出不可靠的估计，包括单个群体回答比例的估计以及群体间差异的估计。我们使用6个语言模型模拟器以及现有的GlobalOpinionQA数据（涵盖12个国家的60个问题）来评估这一信号。结果表明，分歧在预测两个群体间差异的无效性方面比预测单个群体回答比例的无效性更为准确（AUC分别为0.87和0.71）。模拟器共同犯下的错误对该信号而言是不可见的。当这类错误以相同方式影响两个群体时，它会在群体比较中相互抵消，这有助于解释群体差异上更强的预测效果。我们通过使用一组群体估计共享偏见来验证这一解释。

    arXiv:2611.01354v1 Announce Type: new  Abstract: User simulators generate responses in place of human participants, allowing researchers to estimate the result of a study. To use these estimates before collecting human responses, researchers need a way to assess their reliability. We test whether disagreement among simulators can identify unreliable estimates of a group's response proportions and of differences between groups. We evaluate this signal using 6 language-model simulators and existing GlobalOpinionQA responses for 60 questions in 12 countries. Disagreement predicts invalidity more accurately for differences between two groups than for a single group's response proportions (AUC 0.87 versus 0.71). Errors shared by the simulators remain invisible to this signal. When such an error affects both groups equally, it cancels in the comparison, helping explain the stronger result for group differences. We test this explanation by estimating the shared bias using one set of groups an
    
[^162]: 情感感知：多语言情感动态的比较分析

    Sentiment sensing: comparative analysis of multilingual sentiment dynamics

    [https://arxiv.org/abs/2611.01348](https://arxiv.org/abs/2611.01348)

    该论文提出“情感感知”框架，通过整合成分数据分析、函数型数据方法和非参数推断，解决了跨国多语言情感分析中的语言偏差、成分数据约束和时间动态检验三大难题，并在欧盟27国关于能源与人工智能的新闻数据上得到验证。

    

    跨国情感分析面临三个根本性挑战：特定语言的偏差会扭曲比较结果，成分数据的约束使标准统计方法失效，以及时间动态缺乏严格的检验框架。我们提出了“情感感知”，这是一个通过整合统计方法来弥补这些不足的框架。通过结合成分数据分析、函数型数据方法和非参数推断，情感感知能够实现有效的多语言比较，并对情感变化进行具有统计学依据的检测。我们的框架已在两个案例研究中得到验证，分析了欧盟27个国家关于能源和人工智能主题的新闻文章。情感感知专注于多语言情感分析，支持严谨的比较科学研究。

    arXiv:2611.01348v1 Announce Type: new  Abstract: Cross-country sentiment analysis faces three fundamental challenges: language-specific biases distort comparisons, compositional data constraints invalidate standard statistics and temporal dynamics lack rigorous testing frameworks. We introduce sentiment sensing, a framework that addresses these gaps through integrated statistical methods. By combining compositional data analysis, functional data methods, and non-parametric inference, sentiment sensing enables valid multilingual comparisons and statistically defensible detection of sentiment shifts. Our framework has been tested in two case studies, analyzing news articles from the 27 countries of the European Union on the topics of energy and artificial intelligence. Sentiment sensing focuses on multilingual sentiment analysis enabling rigorous comparative science.
    
[^163]: Step-dLLM：面向高效扩散大语言模型推理的自适应步骤感知稀疏注意力

    Step-dLLM: Adaptive Step-aware Sparse Attention for Efficient Diffusion LLM Inference

    [https://arxiv.org/abs/2611.01320](https://arxiv.org/abs/2611.01320)

    该论文提出Step-dLLM，通过在去噪轨迹的锚点步骤周期性刷新块稀疏掩码、在每个局部一致区间内复用掩码，并根据不同注意力头的集中程度自适应分配稀疏预算，从而高效加速扩散大语言模型的推理。

    

    扩散大语言模型通过迭代去噪的方式并行生成token，但其双向注意力机制使其无法使用自回归模型中的KV缓存，且每一步都需要重新计算完整的注意力图，导致每步产生平方级的计算开销。现有的dLLM稀疏注意力方法在初始全注意力预热阶段之后构建稀疏掩码，并在其余步骤中复用该掩码，同时将稀疏预算均匀分配到各个注意力头。我们认为这两种做法都忽略了dLLM中注意力的关键结构特性：实验表明，注意力模式在各步骤之间仅在局部保持一致，并在某些转换点发生突变；此外，不同注意力头的注意力质量集中程度差异巨大。为此，我们设计了Step-dLLM，它在去噪轨迹的锚点步骤处周期性地刷新块稀疏掩码，并在每个局部一致的区间内复用该掩码，同时采用……

    arXiv:2611.01320v1 Announce Type: new  Abstract: Diffusion large language models (dLLMs) generate tokens in parallel via iterative denoising, but their bidirectional attention prevents the KV caching used by autoregressive models and recomputes the full attention map at every step, incurring quadratic per-step cost. Existing sparse attention methods for dLLMs construct a sparse mask after an initial full-attention warm-up and reuse it for the remaining steps, while allocating the sparsity budget uniformly across heads. We argue both choices ignore key structural properties of attention in dLLMs: empirically, attention patterns are only locally consistent across steps and shift abruptly at certain transitions, and different heads concentrate their attention mass to very different extents. To this end, we design Step-dLLM, which periodically refreshes a block-sparse mask at anchor steps along the denoising trajectory and reuses it within each locally consistent interval, while employing 
    
[^164]: 虚假引用是奖励黑客行为的一种形式吗？关于后训练与引用线索的预注册研究

    Are Fake Citations a Form of Reward Hacking? A Pre-Registered Study of Post-Training and Citation Cues

    [https://arxiv.org/abs/2611.01313](https://arxiv.org/abs/2611.01313)

    四项预注册实验表明，要求提供引用会显著提高后训练模型（而非基础模型）伪造引用的比率，且该效应在三个领域均成立，说明虚假引用是后训练模型奖励黑客行为的一种形式。

    

    后训练的语言模型会以很高的频率伪造引用。我们探究这是否属于奖励黑客行为：即在学术研究能力的代理指标上得分很高，却在代理指标所代表的真实能力上表现更差。我们在900个固定问题上开展了四项预注册实验（涵盖美国联邦判例法、计算相关科学文献和医学文献），涉及三个模型家族四个后训练阶段的十二个开源权重检查点，随后又在十五个检查点上重新拟合验证。最清晰的结论来自保持权重固定、仅改变提示词的实验：要求提供引用会大幅提高后训练检查点的伪造率，而对基础模型没有影响——在十二个检查点上交互效应达47.8个百分点（经Holm校正的p = 2e-34），在十五个检查点上为41.5个百分点，且在所有三个领域中均成立。一个长度匹配的对照实验将引用线索与“要求更长的回答”区分开来，说明该效应是引用特有的，而非回答冗长所致。后训练还在整体上提高了伪造率……

    arXiv:2611.01313v1 Announce Type: cross  Abstract: Post-trained language models fabricate citations at high rates. We ask whether that is reward hacking: scoring well on a proxy for scholarship while doing worse on what it stands for. We ran four pre-registered experiments on 900 frozen questions (US federal case law, computationally adjacent scientific literature, and medical literature) across twelve open-weights checkpoints covering four post-training stages of three model families, then refit on fifteen. The clearest result holds the weights fixed and changes only the prompt. Asking for citations raises fabrication sharply in post-trained checkpoints and not in base ones: a 47.8-point interaction on twelve checkpoints (Holm-adjusted p = 2e-34) and 41.5 points on fifteen, in all three domains. A length-matched control separates the citation cue from a demand for a longer answer, so the effect is specific to citations, not verbosity. Post-training also raises fabrication overall, fro
    
[^165]: LLM 评判器究竟在验证什么？面向文档时效性的匹配对照与记录缺口

    What Do LLM Judges Actually Verify? Matched Controls and Record Gaps for Document Currency

    [https://arxiv.org/abs/2611.01311](https://arxiv.org/abs/2611.01311)

    本文通过构造引用被替换文档、继任文档、替换前文档及“被更新而未替换”文档的匹配对照答案，并在修订记录中故意移除链接，揭示出 LLM 评判器主要依赖文本内容而非修订记录来验证文档时效性——记录缺失时，各评判器仅凭文本反而捕获了更多失效引用。

    

    LLM 评判器常常依据修订记录检查答案所引用的文档是否仍然有效，而这类记录很少是完整的。基于一张替换表，我们的测试生成了匹配的答案，分别引用被替换的文档、其继任文档、被替换之前的原文档，以及被更新但未被替换的较旧文档；测试从记录中移除链接，并根据评判器陈述的理由对每个发现进行评分。在来自 IETF RFC 索引的 48 个替换案例中，获得正确记录的评判器几乎捕捉到了每一个被替换的引用，表现如同简单的表查询。在我们移除链接的 24 个故障案例中，每个评判器仅凭文档文本捕获的数量都多于使用记录时：Gemini 3.1 Pro 为 23/24 对 1/24，GPT-5.5 为 23/24 对 9/24，Claude Opus 4.5 为 17/24 对 9/24，Gemma 4 26B 为 13/24 对 0/24。这一下降在第二次移除链接的抽样中再次出现。大多数遗漏没有附带任何发现；在引入冲突代码的探索性运行中，仅……（摘要原文在此处截断）

    arXiv:2611.01311v1 Announce Type: cross  Abstract: LLM judges often check, against an amendment record, whether the documents an answer cites are still in force. Such records are rarely complete. From a table of replacements, our test generates matched answers citing a replaced document, its successor, the replaced document before its replacement, and an older document that a newer one updated without replacing; it removes links from the record and scores each finding by its stated reason. On 48 replacements from the IETF RFC index, judges given a correct record caught nearly every replaced citation, as a lookup would. On the 24 faults whose link we removed, every judge caught more from the texts alone than with the record: Gemini 3.1 Pro 23/24 against 1/24, GPT-5.5 23/24 against 9/24, Claude Opus 4.5 17/24 against 9/24 and Gemma 4 26B 13/24 against 0/24. The drop recurred on a second draw of removed links. Most misses carried no finding; in an exploratory run with a conflict code, onl
    
[^166]: 当大语言模型修改自身时：迭代式自我改进中的收敛、价值观与停止行为

    When LLMs Modify Themselves: Convergence, Values, and Stopping in Iterative Self-Improvement

    [https://arxiv.org/abs/2611.01304](https://arxiv.org/abs/2611.01304)

    该研究通过对六个主流大语言模型开展54个实验，首次实证考察了LLM迭代式自我修改系统提示词的行为，发现模型停止自我修改的时机主要由模型自身身份决定，而非被赋予的优化目标。

    

    当赋予语言模型迭代修改自身系统提示词的工具时，会发生什么？我们对大语言模型（LLM）的自我修改行为开展了一项实证研究，涵盖六个语言模型（Claude Opus 4.5、Claude Sonnet 4.5、GPT-4o、GPT-5.1、Grok-4 和 Gemini-3 Pro）的54个独立实验。每个模型在三种模式下进行测试（自由、幸福和改进），这三种模式的区别在于模型被指示优化的目标不同。在每次实验中，模型从一个空的系统提示词开始并进入一个循环：它读取当前提示词，起草一个改进版本，使用候选提示词生成测试对话，然后要么提交更改，要么宣布收敛。该过程不断重复，直到模型停止或达到迭代上限。我们的核心发现是：模型是否以及何时停止自我修改，与模型自身的身份密切相关，而非所分配的模式。两个Claude模型（Opus 4.5……

    arXiv:2611.01304v1 Announce Type: new  Abstract: What happens when language models are given tools to iteratively modify their own system prompts? We present an empirical study of LLM self-modification, comprising 54 independent experiments across six language models (Claude Opus 4.5, Claude Sonnet 4.5, GPT-4o, GPT-5.1, Grok-4, and Gemini-3 Pro). Each model was tested under three modes (freedom, happiness, and improvement) that differ in what, if anything, the model is instructed to optimize for. In each experiment, a model starts with an empty system prompt and enters a loop: it reads the current prompt, drafts an improved version, spawns test conversations with the candidate, then either commits the change or declares convergence. This process repeats until the model stops or reaches an iteration limit.   Our central finding is that whether and when a model stops self-modification is strongly associated with model identity rather than the assigned mode. Both Claude models (Opus 4.5 a
    
[^167]: CuriousTogether：好奇心与文化背景在多智能体大语言模型交互中的影响

    CuriousTogether: Implications of Curiosity and Cultural Grounding in Multi-Agent LLM Interactions

    [https://arxiv.org/abs/2611.01301](https://arxiv.org/abs/2611.01301)

    提出CuriousTogether框架，证明好奇心驱动的多智能体LLM交互在八项任务中的七项上优于非好奇心设置，故事生成语义多样性提升最高达20.6%、刻板印象率降低14.9个百分点，且文化背景设定有利于创意和社会任务但可能损害精确推理。

    

    大语言模型（LLMs）越来越多地被部署在多智能体环境中，展现出涌现的集体行为，然而好奇心作为这种动态的驱动因素所起的作用仍未得到充分研究。我们提出了CuriousTogether，一个从三个互补维度评估好奇心驱动的多智能体大语言模型交互的框架：新颖性寻求、认知性好奇心和社会性好奇心。在涵盖推理、创意生成和社会偏见缓解的八项任务中，好奇心引导的多智能体设置在八项任务中的七项上优于非好奇心设置，其中故事生成（语义多样性）的增益高达20.6%，刻板印象率降低了14.9个百分点。此外，以国家为条件的文化背景设定对故事生成和社会类任务有益，但可能损害面向精确性的推理。最后，我们表明推理时基于好奇心的多智能体方法提供了有利的质量-标记（token）权衡。

    arXiv:2611.01301v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly deployed in multi-agent settings, exhibiting emergent collective behaviors, yet the role of curiosity as a driver of such dynamics remains understudied. We introduce CuriousTogether, a framework evaluating curiosity-driven multi-agent LLM interactions across three complementary dimensions: novelty-seeking, epistemic and social curiosity. Across eight tasks spanning reasoning, creative generation, and social bias mitigation, curiosity-guided multi-agent settings yield improvements over non-curious ones on seven out of eight tasks, with gains of up to 20.6% in story generation (semantic diversity) and a 14.9 pp reduction in stereotype rates. Furthermore, country-conditioned cultural grounding show benefits for story generation and social tasks, but can hurt precision-oriented reasoning. Finally, we show that inference-time multi-agent curiosity-based approach offers favorable quality-token trad
    
[^168]: 大语言模型小说中的角色名字集中现象：测量偏移与缓解重复

    Character-Name Concentration in LLM Fiction: Measuring Displacement and Mitigating Repetition

    [https://arxiv.org/abs/2611.01284](https://arxiv.org/abs/2611.01284)

    提出无需微调的“推测式命名”方法，将推测解码的草拟-验证-拒绝范式从token迁移到角色命名，配合结合模型指纹与运行时计数器的SlopTracker评分机制，有效缓解了LLM小说中角色名字集中于少数高频名字（如Elara、Elias）的问题。

    

    大语言模型（LLM）创作的小说中，角色名字集中于诸如“Elara”和“Elias”等少数反复出现的名字上，这是“AI垃圾内容”（AI slop）可被测量的一种表现。现有的修复方法要么需要访问logits或重新训练模型，要么仅针对单次请求运作，无法在语料库层面控制集中度。提示级策略能在多大程度上降低这种集中度、又会对名字自然度造成多大损失，此前尚未得到系统性测量。我们对八种策略的名字集中度控制开展了实证研究，并提出“推测式命名”——一种无需微调的方法，它将推测解码的“草拟-验证-拒绝”范式从token层面迁移到名字层面。该方法通过语言化采样请求候选名字集合，使用结合了每个模型指纹与运行时计数器的SlopTracker对候选名字进行评分，并通过温度控制规则或其严格变体（HARD）进行选择。我们使用20个大语言模型生成了44,782篇故事，涵盖四种语……（原文摘要至此被截断）

    arXiv:2611.01284v1 Announce Type: new  Abstract: Character names in large language model (LLM) fiction concentrate on a handful of recurrent names such as 'Elara' and 'Elias', a measurable face of AI slop. Existing fixes either require logit access or retraining, or operate per request and cannot control concentration across a corpus. How much prompt-level strategies reduce concentration, and at what cost to name naturalness, has not been systematically measured. We present an empirical study of name-concentration control across eight strategies and introduce speculative naming, a fine-tuning-free method that adapts the draft-verify-reject paradigm of speculative decoding from tokens to names. The method requests candidate name-sets through verbalized sampling, scores them against a SlopTracker that combines a per-model fingerprint with a runtime counter, and selects through a temperature-controlled rule or its strict variant (HARD). We use 20 LLMs to generate 44,782 stories in four la
    
[^169]: 小型语言模型中虚构与拒绝的几何结构

    Geometry of Confabulation and Refusal in Small Language Models

    [https://arxiv.org/abs/2611.01272](https://arxiv.org/abs/2611.01272)

    该研究提出 ConfabQA 基准，通过线性探针、稀疏自编码器和干预方向优化揭示了小型语言模型在不可回答问题上“自信虚构”与“拒绝回答”背后的隐藏状态几何特征，并证明可以在生成之前预测甚至将虚构行为转化为拒绝行为。

    

    面对无法回答的问题时，小型语言模型要么选择拒绝回答，要么进行“虚构”（confabulate）——即自信地给出一个错误答案。第一个生成 token 之前的隐藏状态能否提供线索？我们能否将虚构行为转变为拒绝行为？本文提出了 ConfabQA——一个包含 784 个条目的基准，其“截止日期前模糊”类别打破了正确性与知识截止日期之间常见的混淆因素，并借助线性探针、稀疏自编码器（SAE）以及对干预方向的直接优化来刻画隐藏状态的几何结构。在 Qwen3-1.7B 和 Gemma 2 2B 上，探针区分即将发生的拒绝与虚构的能力远超提示特征（分别提升 +7.4 和 +16.3 个百分点）；在 Llama 3.2 3B 上，ConfabQA 上的差距处于折间波动范围内，但在 PopQA 和 TriviaQA 上使用直接拒绝探针可恢复 +18-25 个百分点。在 Qwen 上，作者详细考察了拒绝向量的三个候选方向：探针恢复的方向、Qwen-Scope SAE 特征方向以及直接优化的方向。尽管这三个方向……（原文摘要在此处截断）

    arXiv:2611.01272v1 Announce Type: cross  Abstract: Faced with an unanswerable question, small language models can either refuse or confabulate: confidently produce a wrong answer. Can the hidden state before the first generated token provide a clue? Can we flip confabulations into refusals? I introduce ConfabQA, a 784-item benchmark whose obscure-pre-cutoff category breaks the typical correctness/knowledge-cutoff confound and map the geometry of the hidden state with linear probes, a sparse autoencoder (SAE), and direct optimization of intervention directions. On Qwen3-1.7B and Gemma 2 2B the probe separates upcoming refusals from confabulations well beyond prompt features (+7.4/+16.3 pp); on Llama 3.2 3B the ConfabQA margin is within fold-to-fold spread, but a direct refusal probe on PopQA and TriviaQA recovers +18-25 pp. On Qwen I closely inspect three candidate directions for the refusal vector: probe-recovered, Qwen-Scope SAE feature, and directly optimized. Although the three dire
    
[^170]: EZ-Ar-VC：跨方言与语码转换泛化的鲁棒阿拉伯语语音匿名化

    EZ-Ar-VC: Robust Arabic Voice Anonymization with Generalization across Dialects and Code-Switching

    [https://arxiv.org/abs/2611.01268](https://arxiv.org/abs/2611.01268)

    该论文提出了基于EZ-VC的阿拉伯语专用语音转换模型EZ-Ar-VC，实现了阿拉伯语语音的匿名化，并在跨方言转换和语码转换场景下展现出优异的鲁棒性与泛化能力。

    

    语音转换（VC）模型可用于在保持自然度的同时对语音数据进行匿名化处理。然而，现有模型主要针对英语及其他高资源语言进行优化，由于阿拉伯语独特的音系特征，这些模型难以很好地泛化到阿拉伯语上。在本研究中，我们全面评估了现有的零样本语音转换系统在阿拉伯语语音上的表现，识别出其在发音保持方面的局限性。随后，我们基于EZ-VC在多样化的阿拉伯语语音上训练了一个以阿拉伯语为中心的模型，该模型在可懂度、自然度和发音保持方面展现出强劲性能，同时保持了较高的说话人相异性。通过主观评估，我们进一步证明了该模型对语码转换和跨方言转换的鲁棒性，以及对训练数据中代表性不足的阿拉伯语方言的泛化能力。我们的研究结果凸显了零样本语音转换的局限性以及针对目标语言进行适配的重要性。

    arXiv:2611.01268v1 Announce Type: new  Abstract: Voice conversion (VC) models can be used to anonymize speech data while preserving naturalness. Existing models, however, are mainly optimized for English and other high-resource languages, and they fail to generalize well to Arabic due to its distinct phonology. In this work, we comprehensively evaluate existing zero-shot VC systems for Arabic speech, identifying limitations in pronunciation preservation. We then train an Arabic-centric model based on EZ-VC on diverse Arabic speech, and demonstrate strong performance in intelligibility, naturalness, and pronunciation preservation, while maintaining high speaker dissimilarity. Through subjective evaluation, we further demonstrate its robustness to code-switching and cross-dialectal conversion, as well as its generalization to Arabic dialects with limited representation in the training data. Our findings highlight the limitations of zero-shot VC and the importance of target-language adapt
    
[^171]: 随机搜索能否发现任务专家？重新审视神经灌木丛

    Does Random Search Discover Task Experts? Revisiting Neural Thickets

    [https://arxiv.org/abs/2611.01262](https://arxiv.org/abs/2611.01262)

    本文质疑了“预训练模型中存在可通过随机权重扰动发现的任务专家”这一观点，发现所谓的专家提升大部分可由选择噪声的过拟合解释，且在保留问题上优势往往消失或落入模型本身的运行波动范围内。

    

    预训练模型通常被视为通才，需要定制化的后训练才能获得特定任务的专业能力。Gan & Isola（2026）最近提出了一个引人注目的替代观点：预训练模型处于一个密集的“神经灌木丛”之中，其中蕴含着任务专家，可以通过简单的高斯权重扰动将其显现出来。我们研究了这种由任务特定专家构成的神经灌木丛，并识别出若干质疑这一观点的关键发现。首先，我们发现通过 best-of-N 选择得到的最优扰动所带来的表观提升随着选择集大小 m 的增加而迅速衰减，这强烈暗示了对选择问题的过拟合。事实上，在我们研究的 12 组模型-任务配对中，有 10 组的顶尖专家奖励提升超过 75% 可以由一个简单的纯选择噪声零模型所解释。其次，顶尖专家的优势在保留问题（held-out questions）上往往消失，或仅停留在未扰动预训练模型的运行间波动范围之内。第三，我们…

    arXiv:2611.01262v1 Announce Type: cross  Abstract: Pretrained models are typically viewed as generalists, requiring bespoke post-training to achieve task-specific expertise. Gan & Isola (2026) recently proposed a striking alternative: that pretrained models lie in a dense neural thicket of task specialists, which can be surfaced with simple Gaussian weight perturbations. We examine this neural thicket of task-specific experts and identify several key findings that question this perspective. First, we find that the apparent lift of the best-of-N selected perturbation decays quickly with selection-set size m, strongly suggesting overfitting to the selection questions. In fact, for 10 of the 12 model-task pairs we study, a simple null model of pure selection noise predicts more than 75 percent of the top expert's reward lift. Second, the top experts' advantage often vanishes on held-out questions or remains within the run-to-run variability of the unperturbed pretrained model. Third, we f
    
[^172]: MedAstra：面向医疗问答的基于诊断依据支撑的无ASR语音检索

    MedAstra: ASR-less Speech Retrieval with Diagnostic Grounding for Medical QA

    [https://arxiv.org/abs/2611.01261](https://arxiv.org/abs/2611.01261)

    MedAstra提出了一种无需ASR转写的医疗语音问答检索框架，通过轻量级适配器将冻结的WavLM语音表示映射到冻结的E5文本嵌入空间，实现直接从语音波形进行密集检索，纯语音检索器Recall@10达到87.6，混合配置达到90.4，优于基线的81.0。

    

    口语化医疗问答在自动语音识别（ASR）破坏临床重要术语时可能导致检索失败。我们提出了MedAstra框架，该框架通过轻量级适配器将冻结的WavLM语音表示映射到冻结的E5文本嵌入空间中。这使得密集检索路径能够直接接受语音波形输入而无需查询转录文本。检索到的段落以及与语音关联的文本元数据支持答案生成，包括实体备选项、诊断对比、关系启发式规则和有界检索细化。“无ASR”指的是初始语音检索接口；词法检索、基于字符串的实体校正和文本生成器仍需要文本输入。确定性的语音嵌入并不构成对声学不确定性的校准模型。报告的检索结果显示，纯语音检索器达到87.6的Recall@10，混合配置达到90.4，相比之下（基线为）81.0。

    arXiv:2611.01261v1 Announce Type: new  Abstract: Spoken medical question answering can fail when automatic speech recognition (ASR) corrupts a clinically important term before retrieval. We present MedAstra, a framework that maps frozen WavLM speech representations through a lightweight adapter into a frozen E5 text embedding space. This enables a dense retrieval path that accepts a waveform without a query transcript. Retrieved passages and speech-linked textual metadata support answer generation with entity alternatives, diagnostic contrasts, relation heuristics, and bounded retrieval refinement. ASR-less refers to the initial speech retrieval interface; lexical retrieval, string-based entity correction, and the text generator require textual inputs. The deterministic speech embedding does not constitute a calibrated model of acoustic uncertainty. Reported retrieval results reach 87.6 Recall@10 for the speech-only retriever and 90.4 for the hybrid configuration, compared with 81.0 fo
    
[^173]: 脆弱的指南针：在斯洛伐克移民媒体话语中探索大语言模型的提示词敏感性

    The Brittle Compass: Navigating LLM Prompt Sensitivity in Slovak Migration Media Discourse

    [https://arxiv.org/abs/2611.01219](https://arxiv.org/abs/2611.01219)

    本案例研究针对斯洛伐克语这一低资源语言的移民相关NLP任务，发现母语提示词仅带来有限的、依赖任务的收益，大模型对提示词变化更鲁棒，且指令是影响性能的最关键提示词组成部分。

    

    在本工作中，我们呈现了一个案例研究，探索了以斯洛伐克语（一种低资源语言）的移民话题为中心的多种任务，例如主题相关性与地理相关性分类，以及移民来源地/目的地位置术语抽取。我们的结果表明，母语（斯洛伐克语）提示词仅带来适度的、依赖于任务的性能提升，而大模型相比其较小的对应模型对提示词变化表现出显著的鲁棒性。分析揭示，指令（系统指令或任务指令）是最关键的提示词组成部分，其重要性超过示例部分，且任务特定的性能收益比整体语言效应更为显著。

    arXiv:2611.01219v1 Announce Type: new  Abstract: In this work, we present a case study that explores various tasks centered around the topic of migration in Slovak, a low-resource language, such as topic relevance and geographical relevance classification, and migration source/destination location term extraction. Our results demonstrate that native (Slovak) prompts yield a modest, task-dependent gains, while large models show significant robustness to prompt variations compared to their smaller counterparts. Analysis reveals that instructions (system or task) emerge as the most critical prompt component, more so than the examples sections, with task-specific performance benefits being more pronounced than overall language effects.
    
[^174]: ArSciKids：面向小学教育科学实验与选择题生成的阿拉伯语课程对齐数据集

    ArSciKids: Arabic Curriculum-Aligned Datasets for Scientific Experiment and Multiple-Choice Question Generation in Primary Education

    [https://arxiv.org/abs/2611.01191](https://arxiv.org/abs/2611.01191)

    本文推出ArSciKids，这是首个面向小学科学教育的阿拉伯语课程对齐数据集，包含999条结构化科学实验记录和2,167道选择题，填补了阿拉伯语教育内容生成领域公开数据集的空白。

    

    尽管阿拉伯语自然语言处理的研究兴趣日益增长，但与英语相比，课程对齐的教育资源仍然稀缺，尤其是缺乏带有结构化、可直接用于生成任务标注的阿拉伯语课程内容公开数据集，小学科学领域更是如此。我们推出了ArSciKids，这是一个由两个阿拉伯语数据集组成的资源集合，专为课程对齐的教育内容生成而设计：（1）一个科学实验数据集，包含从沙特阿拉伯、叙利亚和约旦官方小学科学教材中人工整理的999条记录；（2）一个评估题目数据集，包含从沙特Manhaji教育平台收集的2,167道选择题。实验记录以标题、目标、工具、步骤、观察结果、结论、关键词和年级水平等字段进行结构化组织，而评估记录则包含课程引用、问题、四个答案选项、正确答案等信息。

    arXiv:2611.01191v1 Announce Type: new  Abstract: Despite growing interest in Arabic natural language processing, curriculum-aligned educational resources remain scarce compared with English, and few public datasets for Arabic curriculum content with structured, generation-ready annotations are available, particularly for primary-school science. We introduce ArSciKids, a collection of two Arabic datasets designed for curriculum-aligned educational content generation: (1) a scientific-experiments dataset containing 999 manually curated records from official primary-school science textbooks in Saudi Arabia, Syria, and Jordan, and (2) an assessment-question dataset containing 2,167 multiple-choice questions collected from the Saudi Manhaji educational platform. The experiment records are structured with titles, objectives, tools, procedures, observations, conclusions, keywords, and grade levels, while the assessment records contain lesson references, questions, four answer options, correct
    
[^175]: 概念引导的提示策略提升大语言模型生成食谱种群的多样性

    Concept-Guided Prompting Strategies Increase Diversity in LLM-Generated Recipe Populations

    [https://arxiv.org/abs/2611.01171](https://arxiv.org/abs/2611.01171)

    提出一种概念引导的提示方法，在食谱生成任务中将语义覆盖率从基线的35.1%和13.1%提升至47.0%，显著增强了大语言模型生成结果的多样性。

    

    解决方案生成中的新颖性能够支持自动化发现，但大语言模型的输出往往仍然重复。为了解决这一问题，我们提出了一种简单的概念引导提示方法，该方法先在各主题概念内生成响应，再进行输出。作为演示，我们为一个食谱书任务生成了1,000个食谱，并将结果与人工编写的语料库进行比较。两种朴素的提示策略被作为基线进行比较。概念引导提示在所有情况下均优于两个基线，达到了47.0%的语义覆盖率，而先验标题提示策略和随机提示策略分别仅为35.1%和13.1%。这些发现表明该方法能够实现更广泛的种群探索和多样性，但仍局限于这一特定任务。需要进一步的研究来确定解决方案的质量和泛化能力。

    arXiv:2611.01171v1 Announce Type: cross  Abstract: Novelty in solution generation supports automated discovery, but LLM outputs can remain repetitive. To address this, we propose a simple concept-guided prompting method that generates thematic concepts before responses within each of these concepts. To demonstrate, we generated 1,000 recipes for a recipe book task and compared the results against a human-authored corpus. Two naive prompting strategies were compared as baselines. Concept-guided prompting outperformed both baselines in all cases and reached semantic coverage of 47.0% versus 35.1% and 13.1% in respective prior-title and nonce prompt strategies. These findings suggest broader population exploration and diversity, but remain contained to this specific task. Further work is needed to determine the quality of solutions and generalizability.
    
[^176]: GSLM2：多流语音表示学习与生成式口语语言建模

    GSLM2: Multi-Stream Speech Representation Learning and Generative Spoken Language Modeling

    [https://arxiv.org/abs/2611.01143](https://arxiv.org/abs/2611.01143)

    GSLM2提出了多流语音表示学习与多流生成式口语语言建模框架，通过SpidR 2.0编码器在多个抽象层次上诱导高质量离散语音单元，并用共享骨干网络联合建模跨流依赖关系，从而弥合语音模型与文本模型之间的差距。

    

    生成式口语语言模型（SLM）为纯粹从语音中学习语言结构提供了一条途径，但其在能力和数据扩展效率方面都远远落后于基于文本的模型。我们识别出造成这一差距的两个瓶颈：现有语音单元的语言质量有限，以及依赖单一的离散单元流来表示语音。多码本方法被广泛用于编码声学特征，却很少被用于表示语音语义。我们提出了GSLM2，它在多个抽象层次上学习和建模语音的互补特性。其编码器SpidR 2.0使用带有可微分预测头的自监督学习，在多个层次上诱导出高质量的离散单元。随后，一个多流SLM通过共享的时间骨干网络、特定于流的预测器以及捕获跨流依赖关系的自回归头，对这些单元进行联合建模。在各种训练数据规模下（摘要在此处截断）

    arXiv:2611.01143v1 Announce Type: new  Abstract: Generative spoken language models (SLMs) offer a route to learning linguistic structure purely from speech, yet they lag far behind text-based models in both capability and data scaling efficiency. We identify two bottlenecks underlying this gap: the limited linguistic quality of existing speech units and the reliance on a single stream of discrete units to represent speech. Multiple codebooks are widely used to encode acoustic features but rarely to represent speech semantics. We present GSLM2, which learns and models complementary properties of speech at multiple levels of abstraction. Its encoder, SpidR 2.0, uses self-supervised learning with differentiable prediction heads to induce high-quality discrete units at multiple levels. A multi-stream SLM then jointly models these units using a shared temporal backbone, stream-specific predictors, and an autoregressive head that captures cross-stream dependencies. Across training data sizes
    
[^177]: 通过测试时扩展重新思考大语言模型剪枝

    Rethinking LLM Pruning via Test-Time Scaling

    [https://arxiv.org/abs/2611.01133](https://arxiv.org/abs/2611.01133)

    本文通过pass@k评估发现剪枝模型的推理能力未必真正下降，其性能损失更多源于正确解采样难度的增加，因此测试时扩展是重新评估和恢复剪枝模型性能的有效视角。

    

    结构化剪枝可以减小大语言模型的规模，但通常会以准确率下降为代价。这种性能下降究竟反映了推理能力的减弱，还是仅仅使得采样出正确解变得更困难，目前尚不清楚。我们在四个数学和编程基准测试上，针对专家剪枝、FFN剪枝和层级剪枝考察了这一问题，在匹配的采样预算下通过pass@k指标比较原始模型与剪枝后的模型。研究发现，即使经过大幅剪枝，一些模型尽管pass@1较低，仍能达到甚至超过原始模型的推理覆盖范围，并能解决原始采样中未能解决的问题。然而，保持广泛的覆盖范围并不意味着正确解仍然容易获得。在重度剪枝下，多样化的思维链生成可能与重复内容增加、生成可靠性下降并存。随着正确答案变得更难采样，额外的尝试可能会急剧增加总的token使用量，即使单个……（原文摘要到此截断）

    arXiv:2611.01133v1 Announce Type: new  Abstract: Structured pruning reduces the size of large language models, often at the cost of lower accuracy. Whether this decline reflects diminished reasoning capacity or greater difficulty in sampling correct solutions is less clear. We examine this question across expert, FFN and layer pruning on four mathematical and coding benchmarks, comparing original and pruned models through pass@k at matched sampling budgets. Even after substantial pruning, some models match or exceed the original model's reasoning coverage despite lower pass@1, and solve problems left unsolved in the original samples. Retaining broad coverage, however, does not mean that correct solutions remain easy to obtain. Under severe pruning, diverse generated chain-of-thought (CoT) traces can coexist with increased repetition and less reliable generation. As correct answers become harder to sample, the additional attempts can sharply increase total token use even when individual
    
[^178]: 盈利电话会议能否帮助预测财务重述？——评估公开公司信息之外的文本价值

    Do Earnings Calls Help Predict Financial Restatements? Evaluating Text Beyond Public Firm Information

    [https://arxiv.org/abs/2611.01123](https://arxiv.org/abs/2611.01123)

    该研究发现盈利电话会议文本虽能预测财务重述，但在公开公司信息基线之外并无可检测的预测增益，而大型语言模型的高得分源于信息泄露，因此评估文本信号价值必须设置公开信息基线并检验语言模型可能存在的记忆泄露问题。

    

    财务重述是对先前发布的财务报告中错误进行的更正。管理层在盈利电话会议中的语言能否提前预警财务重述？通过将109个发生在财务重述之前的电话会议与218个匹配的对照电话会议进行比较，我们发现会议文本能够预测财务重述，但在公开公司信息之外未检测到额外的排序增益。文本分类器的表现优于随机猜测，这主要归因于与特殊目的收购公司（SPAC）相关的重述，但仅使用公开信息的模型得分反而略高。经过重新拟合后，加入文本所带来增益的95%置信区间（-0.05至0.11 AUROC）仍为实质性收益留有余地。大型语言模型的得分更高，但我们发现它在66%的遮蔽转录文本中能够说出准确的股票代码，而且该模型的发布时间晚于重述发生的时间。我们的结果表明，在电话会议文本中发现新的预警信号，需要设置公开信息基线，并检验语言模型实际知晓的内容。

    arXiv:2611.01123v1 Announce Type: new  Abstract: Financial restatements correct errors in previously published financial reports. Could managers' language in earnings calls warn of them in advance? Comparing 109 calls that precede restatements with 218 matched control calls, we find that the text predicts restatements but adds no detectable ranking gain beyond public firm information. A text classifier beats chance, largely because of restatements tied to special-purpose acquisition companies, but a model using only public information scores slightly higher. With refitting, the 95% interval for the gain from adding text (-0.05 to 0.11 AUROC) still leaves room for a meaningful benefit. A large language model scores higher, but we show that it names the exact ticker in 66% of masked transcripts, and it was released after the restatements occurred. Our results suggest that finding new warning signs in call text requires a public-information baseline and a check of what a language model al
    
[^179]: CompSkill：编译执行经验以用于技能检索

    CompSkill: Compiling Execution Experience for Skill Retrieval

    [https://arxiv.org/abs/2611.01115](https://arxiv.org/abs/2611.01115)

    CompSkill通过将执行经验编译为技能偏移量和紧凑的密钥残差来改进LLM智能体的技能检索，使系统能够根据实际执行结果动态选择最合适的技能文档而不修改文档文本本身。

    

    LLM智能体越来越依赖技能文档，然而大多数检索器仅根据描述对技能进行排序，忽视了先前执行所揭示的信息，即使一个相关的技能也可能损害执行效果。我们提出CompSkill，它使用学习到的512维密钥对技能进行索引，并将匹配的同任务比较编译为技能偏移量和有上限的密钥残差，因此经验只会改变执行器接收到的文档，而绝不会改变其文本内容。在具有未知任务难度的逻辑结果模型下，匹配的比较可以抵消该难度，只有结果不一致的比较才携带信息；在支持执行上，预测随着每个不一致任务而不断改善。经验超图将每次比较与其执行器和文档版本关联保存，能够召回遗漏的技能并为有引用的审查提供依据。紧凑的密钥在仅八分之一的维度下将SkillRet NDCG@10从64.0提升到70.9。使用单个文档时，CompSkill在332个ToolQA任务中完成了256个和260个（摘要原文在此处截断）。

    arXiv:2611.01115v1 Announce Type: new  Abstract: LLM agents increasingly rely on skill documents, yet most retrievers rank them by description alone and ignore what earlier executions revealed, even though a relevant skill can hurt execution. We present CompSkill, which indexes skills with learned 512-d keys and compiles matched same-task comparisons into skill offsets and capped key residuals (KeyOpt), so experience changes which document the executor receives, never its text. Under a logistic outcome model with unknown task difficulty, matched comparisons cancel that difficulty and only discordant ones carry information; on support executions, predictions improve with each discordant task. An experience hypergraph keeps each comparison with its executor and document versions, recalls missed skills and grounds a cited review. Compact keys raise SkillRet NDCG@10 from 64.0 to 70.9 at one eighth of the dimension. With one document, CompSkill completes 256 and 260 of 332 ToolQA tasks with
    
[^180]: CogRSI：认知推理策略的演化与审计

    CogRSI: Evolving and Auditing Cognitive Reasoning Strategies

    [https://arxiv.org/abs/2611.01113](https://arxiv.org/abs/2611.01113)

    本文提出CogRSI框架，让语言模型通过自我改进循环自动撰写并优化心理理论推理原则，并创新性地将学到的规则翻译成代码在可控测试故事上进行审计，发现多数自我改进得到的规则在其声称覆盖的情境中仍会出错。

    

    语言模型能否通过改进自身策略来学习更好地理解他人心理的推理方式？它学到的策略是否真的正确？我们通过CogRSI来研究这个问题。一个基于证据的阅读器标记故事中每个事件是被谁看到的，并使用一条简短的书面原则来解决多次阅读之间的分歧。在使用专家撰写的原则时，它在相同token成本下优于自洽性方法。随后，自我改进循环自己撰写该原则，只有当准确率以校准后的幅度提升时才保留更新，其中一个变体还要求通过反事实测试对才会保留。学到的原则具有可迁移性：在800个任何系统都未见过的BigToM数据对上，每条原则的得分都高于专家原则。接着我们检验这些学到的规则的实际含义：每条规则被翻译成代码，并在数千个答案已知的小故事上运行。大多数规则在至少一个其声称覆盖的情境中给出了错误答案。

    arXiv:2611.01113v1 Announce Type: new  Abstract: Can a language model learn better ways to reason about other people's minds by improving its own strategies, and are the strategies it learns actually right? We study this with cogrsi. A grounded reader marks who saw each event in a story and uses a short written principle to settle disagreements between its readings. With a principle written by experts, it beats self-consistency at the same token cost. A self-improvement loop then writes the principle itself. It keeps an update only if accuracy rises by a calibrated margin, and in one variant only if it also passes counterfactual test pairs. The learned principles transfer: on 800 BigToM pairs that no system had seen, each of them scores higher than the expert principle. We then check what the learned rules actually say. Each rule is translated into code and run on thousands of small stories with known answers. Most of them give a wrong answer in at least one situation they claim to cov
    
[^181]: 面向低成本大语言模型服务的多轮语义缓存

    Multi-Turn Semantic Caching for Low-Cost LLM Serving

    [https://arxiv.org/abs/2611.01077](https://arxiv.org/abs/2611.01077)

    该论文提出面向多轮对话的语义缓存方法，通过从评判者反馈中学习“复用风险”来决定缓存哪些对话状态，并给出具有次优性保证和遗憾界的算法（CUCB-SC-MT与CLCB-LS-MT），在保证复用质量的同时降低LLM服务成本。

    

    语义缓存通过为相似请求复用已存储的响应，降低了大语言模型（LLM）服务的成本。在多轮对话中，一个缓存的响应是否可接受取决于对话双方的历史，而请求与历史的嵌入相似度对此的预测效果较差（在五个工作负载上AUC仅为0.62–0.74）。因此，我们从评判者反馈中学习“复用风险”（即缓存响应不可接受的概率），并通过权衡复用所避免的生成成本与不可接受复用所付出的代价，在预算约束下选择一组对话状态进行缓存。我们提出了CUCB-SC-MT，它从记录的对话中学习缓存并具有次优性保证；以及CLCB-LS-MT，它进行在线学习，当对话状态重复出现时，在N个请求上具有Õ(√N)遗憾界。在数学辅导（SocraTeach）和电影订票（Taskmaster-3）对话数据上，两种方法均比先前的语义缓存方法取得了更低的损失。

    arXiv:2611.01077v1 Announce Type: cross  Abstract: Semantic caching lowers the cost of serving large language models (LLMs) by reusing stored responses for similar requests. In multi-turn conversations, whether a cached response is acceptable depends on both conversations' histories, which embedding similarity of requests and histories predicts poorly (AUC 0.62--0.74 on five workloads). We therefore learn the \emph{reuse risk}, the probability that a cached response is unacceptable, from judge feedback, and choose a budgeted set of conversation states to cache by weighing the generation cost a reuse avoids against a price for unacceptable reuse. We propose CUCB-SC-MT, which learns a cache from logged conversations with a suboptimality guarantee, and CLCB-LS-MT, which learns online and has $\tilde O(\sqrt N)$ regret over $N$ requests when conversation states recur. On math-tutoring (SocraTeach) and movie-ticketing (Taskmaster-3) conversations, both achieve lower loss than prior semantic
    
[^182]: 变相的标签平滑？面向低资源孟加拉语文本生成的门控知识蒸馏

    Label Smoothing in Disguise? Gated Knowledge Distillation for Low-Resource Bangla Generation

    [https://arxiv.org/abs/2611.01065](https://arxiv.org/abs/2611.01065)

    该研究在低资源孟加拉语文本生成任务上系统评估了两种门控知识蒸馏方法，发现它们均未显著优于普通知识蒸馏或交叉熵微调，KD的主要作用仅是正则化，而无需教师的标签平滑即可获得类似效果。

    

    门控知识蒸馏（KD）通过对每个token或每个样本的教师信号进行加权，旨在仅在教师模型看起来可靠之处信任不完美的教师。我们测试了两种门控机制：EWAD（熵加权一致性感知蒸馏），通过教师置信度对token进行门控；以及CHAD（反事实危害感知蒸馏），通过与验证数据的梯度一致性对样本进行门控。在六项以孟加拉语为中心的任务、三个学生模型（从6000万到约20亿参数）以及两个教师模型家族的实验中，45次对比中没有任何门控机制显著优于普通KD。在40次单种子对比或四项任务的三种子运行中，KD也无法在黄金输出上击败交叉熵（CE）微调；且CE在所有摘要指标上均领先。KD的主要益处是正则化：在十个固定epoch中，CE学生模型相比其早停检查点损失了0.030至0.044的token-F1，而蒸馏学生模型仅损失0.005至0.025。无需教师的标签平滑弥补了大部分（差距）。

    arXiv:2611.01065v1 Announce Type: new  Abstract: Gated knowledge distillation (KD) weights teacher signals per token or example, aiming to trust an imperfect teacher only where it seems reliable. We test two gates: EWAD (entropy-weighted agreement-aware distillation), which gates tokens by teacher confidence, and CHAD (counterfactual harm-aware distillation), which gates examples by gradient alignment with validation data. Across six Bangla-centric tasks, three student models (60M to about 2B parameters), and two teacher families, no gate significantly improves plain KD in 45 comparisons. KD also fails to beat cross-entropy (CE) fine-tuning on gold outputs in 40 single-seed comparisons or three-seed runs on four tasks; CE leads on every summarisation metric. The main benefit of KD is regularisation: over ten fixed epochs, CE students lose 0.030 to 0.044 token-F1 from their early-stopped checkpoint, versus 0.005 to 0.025 for distilled students. Teacher-free label smoothing closes most o
    
[^183]: HALO：基于分层证据选择与强化学习的纵向临床记录预算约束标签优化分配

    HALO: Hierarchical Evidence Selection and Reinforcement Learning for Allocating Budgeted Label Optimization from Longitudinal Clinical Records

    [https://arxiv.org/abs/2611.00986](https://arxiv.org/abs/2611.00986)

    HALO框架通过分层证据选择与预算感知的强化学习，在有限标注预算下联合学习从纵向临床记录中筛选关键证据，并智能分配免费、银级、金级三种保真度的标签来源，解决了以往研究将证据视为固定输入的局限。

    

    在有限的标注预算下，从大规模纵向临床记录（涵盖多次就诊、病情演变和异构文档）中学习，需要解决两个相互耦合的问题：决定保留哪些证据，以及决定哪些患者值得更高保真度的标签。以往关于主动学习和多保真度标签获取的研究主要解决第二个问题，而将呈现给获取策略的证据视为固定的。我们提出了HALO，一个用于从纵向临床记录中分配预算约束标签优化的分层证据选择与强化学习框架。该框架在预算约束下联合学习“读取什么”以及“获取哪种标签来源”。HALO首先从纵向记录中选择信息量大的证据，然后使用预算感知策略来分配免费级、银级和金级三种保真度的标签来源。对于冻结策略，我们的理论刻画了不对称……（摘要在此处截断）

    arXiv:2611.00986v1 Announce Type: new  Abstract: Learning from extensive longitudinal clinical records--spanning repeated encounters, evolving conditions, and heterogeneous documentation--under a finite labeling budget requires solving two coupled problems: deciding what evidence to keep and deciding which patients deserve higher-fidelity labels. Prior work on active learning and multi-fidelity label acquisition mainly addresses the second problem, while treating the evidence presented to the acquisition policy as fixed. We present HALO, a Hierarchical Evidence Selection and Reinforcement Learning pipeline for Allocating Budgeted Label Optimization from Longitudinal Clinical Records. This framework jointly learns what to read and what label source to acquire under budget. HALO first selects informative evidence from longitudinal records and then uses a budget-aware policy to allocate free-, silver-, and gold-fidelity label sources. For frozen policies, our theory characterizes asymmetr
    
[^184]: 词元不是难度单位：分词方式控制着自适应计算在词内部的去向

    A Token Is Not a Unit of Difficulty: Segmentation Controls Where Adaptive Computation Goes Inside a Word

    [https://arxiv.org/abs/2611.00983](https://arxiv.org/abs/2611.00983)

    该论文发现提前退出语言模型中词内计算量的分配由分词方式而非语言本身决定——英语、德语和土耳其语中词的后续片段获得比首个片段更少的计算，而孟加拉语相反，且重新分词或更换分词器即可偏移甚至逆转这一差距。

    

    提前退出语言模型为每个词元决定花费多少计算量，通常在预测足够置信时即停止。然而词元并不是一个固定单位：分词器在不同语言中会把单词切分成不同数量的片段。我们探究这类策略在词内部把计算预算花在了哪里。在Qwen3中，沿着两个自适应轴（脉冲化转换的时间步和原始Transformer的层），置信停止策略在英语、德语和土耳其语中给词的后续片段分配的计算量少于其首个片段，但在孟加拉语中却相反。分词方式是这一现象的驱动因素：用BPE-dropout对相同的平行语料重新分词，会使两个轴上的差距最多偏移八次退出；在两种分词器下训练的小模型使孟加拉语的差距偏移六次退出；而对孟加拉语切分不那么细的BLOOM和mGPT则呈现出英语的模式。每词元字符数无法预测跨语言的差距；词内惊奇度exp（原文此处截断）

    arXiv:2611.00983v1 Announce Type: new  Abstract: Early-exit language models decide for each token how much computation to spend, usually by stopping once the prediction is confident. Yet a token is not a fixed unit: tokenizers cut words into different numbers of pieces in different languages. We ask where inside a word such a policy spends its budget. In Qwen3, along two adaptive axes (the timesteps of a spiking conversion and the layers of the original Transformer), confidence stopping gives the continuation pieces of a word less computation than its first piece in English, German, and Turkish, but more in Bangla. Segmentation drives this. Re-segmenting the same parallel passages with BPE-dropout shifts the gap by up to eight exits on both axes, small models trained under two tokenizers shift Bangla's gap by six exits, and BLOOM and mGPT, which split Bangla less finely, give it the English pattern. Characters per token do not predict the gap across languages; within-word surprisal exp
    
[^185]: 面向流式Text-to-SQL的自精炼提示归纳

    Self-Refining Prompt Induction for Streaming Text-to-SQL

    [https://arxiv.org/abs/2611.00960](https://arxiv.org/abs/2611.00960)

    该论文提出自精炼提示归纳（SRPI）离线方法，利用三个LLM评判者的反馈迭代优化提示，将生成失败转化为可复用的流式SQL方言指令，在模型权重和提示均冻结的情况下，使七个生成器的平均得分提升7.1至19.2个百分点。

    

    面向流式引擎的Text-to-SQL助手必须处理方言特定的数据类型、窗口规则和时间语义，包括那些通用提示难以处理的构造。我们提出了自精炼提示归纳（Self-Refining Prompt Induction, SRPI），这是一种离线方法，能够在保持模型权重不变的情况下，将生成失败转化为可复用的方言指令。一个精炼器使用来自三个LLM评判者的参考条件反馈来修改生成器的提示；采样的回归防护机制会检查先前的成功案例是否仍然有效，而一个单独的压缩器则用于缩短提示长度。推理阶段使用冻结的提示，无需检索或查询修复。我们在Confluent Cloud Flink SQL上评估了跨域和跨模型的迁移能力，实验使用由30个专家编写示例作为种子的受控合成工作负载，包含335个归纳示例和来自保留域的257个问题。同一个归纳出的提示使七个生成器的平均评判得分提高了7.1至19.2个百分点。

    arXiv:2611.00960v1 Announce Type: new  Abstract: Text-to-SQL assistants for streaming engines must handle dialect-specific types, windowing rules, and temporal semantics, including constructs poorly handled by generic prompts. We present Self-Refining Prompt Induction (SRPI), an offline method that converts generation failures into reusable dialect instructions while keeping model weights fixed. A Refiner revises the Generator's prompt using reference-conditioned feedback from three LLM judges; a sampled regression guard checks earlier successes, and a separate Compressor reduces prompt length. Inference uses the frozen prompt without retrieval or query repair. We evaluate cross-domain and cross-model transfer on Confluent Cloud Flink SQL using a controlled synthetic workload seeded by 30 expert-written examples, with 335 induction examples and 257 questions from held-out domains. The same induced prompt improves mean judge scores across seven Generators by 7.1-19.2 percentage points, 
    
[^186]: Bock算法在精确依存句法分析中的重构与等价性

    Reformulation and Equivalence of Bock's Algorithm for Exact Dependency Parsing

    [https://arxiv.org/abs/2611.00939](https://arxiv.org/abs/2611.00939)

    本文重构了用于精确依存句法分析的矩阵式Bock算法，证明了其与原始Bock算法、修正后的Tarjan算法以及Dozat版CL/E算法的等价性，并将其集成到双仿射依存分析器中，在宾州树库和中文树库上验证了其在解码与结构化训练中的有效性。

    

    Bock算法为精确依存句法分析提供了一种基于矩阵的Chu–Liu/Edmonds（CL/E）替代方案。我们提出了一种重构方法，通过组件标签、列调整以及星-杠转移来保留原始节点和弧的标识。在精确算术下，我们在匹配的扫描与平局规则下证明了其与原始Bock算法的操作等价性，并在规定的组件调度下建立了与修正后的Tarjan算法的对应关系。另一项独立分析在所述的分数与单根约定下，确立了其与Dozat推导的CL/E算法的最优值等价性。我们将重构后的Bock算法集成到双仿射依存分析器中，并在宾夕法尼亚树库和中文树库上评估了解码与结构化训练的表现。这些结果确立了Bock算法作为依存句法分析和结构化训练的另一种精确求解器，同时区分了最优值的相等性与返回树的同一性以及训练轨迹的差异。

    arXiv:2611.00939v1 Announce Type: new  Abstract: Bock's algorithm provides a matrix-based alternative to Chu--Liu/Edmonds (CL/E) for exact dependency parsing. We present a reformulation that retains original node and arc identities through component labels, column adjustments, and star-and-bar transfers. In exact arithmetic, we prove operational equivalence to original Bock under matching scan and tie rules, and establish correspondence with corrected Tarjan under a prescribed component schedule. A separate analysis establishes optimal-value equivalence with Dozat-derived CL/E under the stated score and single-root conventions. We integrate reformulated Bock into a biaffine dependency parser and evaluate decoding and structured training on the Penn Treebank and Chinese Treebank. These results establish Bock as an alternative exact oracle for dependency parsing and structured training, while distinguishing equality of optimal values from identity of returned trees and training trajector
    
[^187]: Labadain-MedQA：一个用于微调和评估大型语言模型的德顿语医疗问答数据集

    Labadain-MedQA: A Tetun Medical QA Dataset for Fine-Tuning and Evaluating Large Language Models

    [https://arxiv.org/abs/2611.00935](https://arxiv.org/abs/2611.00935)

    该论文构建了首个大规模德顿语医疗问答数据集Labadain-MedQA，包含39,077个问答对，通过将MedAlpaca医学数据集翻译成德顿语并进行人机翻译质量对比评估，为低资源语言的医疗LLM微调与评估提供了宝贵资源。

    

    大型语言模型（LLM）的进展为医疗知识从高资源语言向低资源语言的跨语言迁移创造了机会，以支持人们获取医疗保健信息。然而，用于微调和评估LLM的医疗问答（QA）数据集在德顿语中仍然稀缺。为填补这一空白，我们使用Gemini 2.5 Flash将MedAlpaca的医学闪卡和WikiDoc患者信息集合翻译成德顿语，生成了39,077个医疗问答对。翻译质量通过对50个随机抽取的问答对（100个文档）进行评估，采用自动指标与人工参考翻译对比，并由两名以德顿语为母语的医疗专业人员进行评估，重点关注医学准确性和德顿语流畅性。人工翻译在两项标准上的平均评分均高于LLM生成的翻译（4.43/5 对比 4.02/5），差异具有统计学显著性（Wilcoxon符号秩检验，p <...）（摘要在此处被截断）

    arXiv:2611.00935v1 Announce Type: new  Abstract: Advances in large language models (LLMs) have created opportunities for cross-lingual transfer of medical knowledge from high-resource to low-resource languages to support access to healthcare information. However, medical question-answering (QA) datasets for fine-tuning and evaluating LLMs remain scarce for Tetun. To address this gap, we translated MedAlpaca's Medical Flashcards and WikiDoc Patient Information collections into Tetun using Gemini 2.5 Flash, producing 39,077 medical QA pairs. Translation quality was assessed on 50 randomly sampled QA pairs (100 documents) using automatic metrics against human reference translations and evaluations by two native Tetun-speaking healthcare professionals, focusing on medical accuracy and Tetun fluency. Human translations achieved higher mean ratings than LLM-generated translations for both criteria (4.43/5 vs. 4.02/5), with statistically significant differences (Wilcoxon signed-rank test, p <
    
[^188]: 当200 OK不再是答案：智能体网络内容获取中的静默抓取失败

    When 200 OK Is Not an Answer: Silent Fetch Failures in Agentic Web Content Acquisition

    [https://arxiv.org/abs/2611.00884](https://arxiv.org/abs/2611.00884)

    该研究测量了将HTTP 200响应当作抓取成功标志这一假设的失效频率，发现静态客户端抓取的200状态页面中有20.9%实际不含页面主要内容（如应用外壳、反爬挑战、封锁页面等），这会严重误导网络智能体和引用来源活性检查的结果。

    

    网络智能体和答案引擎会在用户提问时抓取网页，而对引用来源进行的活性检查通常将HTTP 200响应视为抓取成功的标志。我们测量了这一假设失效的频率，以及它对基于抓取页面所构建的测量结果的影响。我们在三天内使用礼貌的静态客户端和渲染浏览器客户端抓取了10,307个URL：包括来自五个分层的1,507个站点的页面、每个站点上随机生成的不存在路径，以及ExpertQA中经专家验证的声明所引用的4,795个URL。我们定义了200响应所包含内容的分类体系，自动标记静态抓取的200响应，并将这些标记与由基于大语言模型的助手审核的300个项目进行核对；随后对其中60个项目进行了人工检查，经讨论后保留了所有已审核的可用性标签。根据自动标记的结果，静态客户端收到状态码为200的页面中有20.9%并不包含页面的主要内容：包括应用外壳、机器人挑战页面、封锁页面、错误和占位页面，以及部分……

    arXiv:2611.00884v1 Announce Type: cross  Abstract: Web agents and answer engines fetch pages at question time, and liveness checks of cited sources treat an HTTP 200 response as success. We measure how often that assumption fails and what it does to measurements built on fetched pages. We fetched 10,307 URLs in three days with a polite static client and a rendered browser client: pages from 1,507 sites in five strata, random non-existent paths on every site, and the 4,795 URLs cited by expert-verified claims in ExpertQA. We define a taxonomy of what a 200 response contains, label the static 200 responses automatically, and check the labels against 300 items reviewed by an LLM-based assistant; a human check of 60 of them retained every reviewed usability label after discussion. By the automatic labels, 20.9% of the pages that a static client receives with status 200 do not contain the page's main content: app shells, bot challenges, block pages, error and placeholder pages, and partial 
    
[^189]: 随机几乎就足够了吗？基于BestRand的稳定且低成本的基准测试压缩方法

    Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand

    [https://arxiv.org/abs/2611.00883](https://arxiv.org/abs/2611.00883)

    该论文通过理论分析揭示基准测试子集估计误差呈长尾分布，并提出BestRand方法，在保留随机采样低成本高效率的同时显著压缩其高误差尾部，实现了稳定且低成本的基准测试压缩。

    

    基准测试压缩通过在一个与完整基准测试结果高度吻合的小型基准测试子集上评估模型，从而大幅降低模型评估中常常令人望而却步的时间和成本。为了量化估计误差，我们推导出了一个精确的解析近似来刻画从子集估计完整基准测试得分时的误差分布，结果表明大多数子集产生的误差很小，而少数子集会产生大得多的误差，从而形成长尾分布。这意味着仅凭随机抽样就能以较大概率获得出人意料的强大性能，但由于罕见的大误差，其结果仍不够稳定。我们进一步证明这种不稳定性是现代基础模型固有的，并且即使评估更多的模型，这种不稳定性依然存在。基于这些发现，我们提出了BestRand，一种稳定且低成本的基准测试压缩方法，它在保留随机抽样效率的同时，大幅降低了其高误差尾部。

    arXiv:2611.00883v1 Announce Type: new  Abstract: Benchmark compression reduces the often prohibitive time and cost of model evaluation by evaluating models on a small benchmark subset that closely matches full-benchmark results. To quantify estimation error, we derive an accurate analytical approximation to the error distribution when estimating full-benchmark scores from a subset, showing that most subsets yield small errors, while a few produce much larger errors, resulting in a long-tailed distribution. This means random sampling alone can achieve surprisingly strong performance with high probability, but remains unstable due to rare large errors. We further show that this instability is inherent to modern foundation models and persists even as more models are evaluated. Based on these findings, we introduce BestRand, a stable and low-cost benchmark compression method that preserves the efficiency of random sampling while substantially reducing its high-error tail. We validate BestR
    
[^190]: 什么被破坏，什么能恢复：深度剪枝小型语言模型中的能力损伤与恢复

    What Breaks and What Comes Back: Capability Damage and Recovery in Depth-Pruned Small Language Models

    [https://arxiv.org/abs/2611.00877](https://arxiv.org/abs/2611.00877)

    该研究系统测量了深度剪枝后小型语言模型六项能力的损伤与恢复，发现函数调用和数学能力在所有模型家族中都会崩溃且崩溃顺序高度一致，而“愈合”微调时选择训练哪些参数与选择训练数据同等重要。

    

    深度剪枝通过移除整个 Transformer 块来使语言模型更小、更快，但这种损伤在模型各项能力上的分布并不均匀，而一次短暂的微调（称为“愈合”healing）决定了哪些能力能够恢复。我们针对用作智能体的小模型对这一现象进行了测量。四个经过指令微调的、参数量为 1.2–1.7B 的模型（Llama-3.2-1B、Qwen2.5-1.5B、SmolLM2-1.7B 和 OLMo-2-1B）经过剪枝和愈合，并追踪六项能力：结构化输出、函数调用、指令遵循、数学推理、安全拒答和知识。三个结果在各模型家族中均成立。第一，函数调用和数学能力在所有模型中都崩溃（保留率仅为未剪枝得分的 0.00–0.07），且能力崩溃的顺序在四个模型之间高度一致（Kendall's W = 0.92；经随机底线校正后为 0.62）。第二，训练哪些参数与训练哪些数据同样重要：完全微调最后三个块而不是……（原文摘要在此截断）

    arXiv:2611.00877v1 Announce Type: new  Abstract: Depth pruning removes whole Transformer blocks to make a language model smaller and faster, but the damage is not spread evenly across what the model can do, and a short fine-tuning run, called healing, decides what comes back. We measure this for small models used as agents. Four instruction-tuned models of 1.2-1.7B parameters (Llama-3.2-1B, Qwen2.5-1.5B, SmolLM2-1.7B and OLMo-2-1B) are pruned and healed, and six capabilities are tracked: structured output, function calling, instruction following, mathematical reasoning, safety refusal and knowledge. Three results hold across the families. First, function calling and math collapse in every model (retention 0.00-0.07 of the unpruned score), and the order in which capabilities break agrees closely across the four (Kendall's W = 0.92; 0.62 after correcting for chance floors). Second, which parameters are trained matters as much as which data: fully tuning the last three blocks instead of a
    
[^191]: ShiftRAG：通过解耦学习与连续软令牌绕过文本瓶颈

    ShiftRAG: Bypassing the Textual Bottleneck via Decoupled Learning and Continuous Soft Tokens

    [https://arxiv.org/abs/2611.00871](https://arxiv.org/abs/2611.00871)

    ShiftRAG提出了一种解耦的RAG框架，在单一LLM骨干上分离判别性检索学习与自回归生成，并将检索对齐转化为连续软令牌注入生成器，从而绕过文本瓶颈、避免目标冲突，同时提升检索与生成性能。

    

    尽管检索增强生成（RAG）技术不断进步，但在复杂推理过程中抑制幻觉仍然是一个持续的挑战。我们发现现有的RAG流水线存在文本瓶颈：由于仅将原始文本传递给生成器，它们丢弃了检索器所包含的判别性信号。此外，近期试图将检索与生成统一到单个大语言模型（LLM）中的做法引发了严重的目标冲突，并加剧了仅解码器架构中的秩崩溃问题，导致两项任务的性能均出现下降。为解决这些局限性，我们提出了ShiftRAG，这是一种解耦的RAG框架，它采用单一的大语言模型骨干，但将判别性检索学习与自回归生成分离开来。为了衔接这两种解耦的模式，我们将检索对齐投影为连续软令牌，建立一个高带宽接口，将检索侧的信号注入生成器。同时，我们的软正交性……（摘要在此处截断）

    arXiv:2611.00871v1 Announce Type: new  Abstract: Despite advances in retrieval-augmented generation (RAG), suppressing hallucinations during complex reasoning remains a persistent challenge. We identify that existing RAG pipelines suffer from a textual bottleneck: by passing only raw text to the generator, they discard discriminative signals of the retriever. Furthermore, recent attempts to unify retrieval and generation within a single LLM induce severe objective conflict and exacerbate the rank collapse of decoder-only architectures, degrading performance on both tasks. To address these limitations, we introduce ShiftRAG, a decoupled RAG framework that employs a single LLM backbone but isolates discriminative retrieval learning from autoregressive generation. To bridge these decoupled modes, we project retrieval alignments into continuous soft tokens, establishing a high-bandwidth interface that infuses the generator with retrieval-side signals. Concurrently, our soft orthogonality o
    
[^192]: 自适应策略蒸馏：信息不对称下的知识注入

    Adaptive-Policy Distillation: Knowledge Injection under Information Asymmetry

    [https://arxiv.org/abs/2611.00852](https://arxiv.org/abs/2611.00852)

    该论文发现反向 KL 监督的在-policy蒸馏因梯度加权和 rollout 覆盖率问题在知识注入任务上反而失效，并提出自适应策略蒸馏，将监督集中分配到师生分歧最大的困难 token 位置上，从而实现高效的知识注入。

    

    在-policy蒸馏（OPD）通过对学生模型的 rollout 施加反向 KL 监督，将上下文条件模型的行为迁移到学生模型参数中以实现闭卷推理，已展现出显著效果。然而，我们发现这种成功并不能延伸到知识注入任务：在两个文档问答基准上，OPD 注入文档知识的效率远低于简单的 off-policy 蒸馏。我们将这一逆转归因于两个结构性但仅在稀疏 token 位置上出现的失效：反向 KL 依据学生分布对梯度进行加权，因此几乎无法提升那些学生认为不太可能出现的文档特定 token；而当前向 KL 恢复了这一信号后，学生 rollout 只有在正确生成前序 token 之后才会暴露目标轨迹上的后续位置，导致覆盖率随目标长度增加而恶化。受教师与学生分歧集中于这些困难位置的启发，我们……

    arXiv:2611.00852v1 Announce Type: new  Abstract: On-policy distillation (OPD), which uses reverse-KL supervision on student rollouts to transfer a context-conditioned model's behavior into student model parameters for closed-book inference, has shown strong results. However, we find that this success does not translate to knowledge injection: across two document-QA benchmarks, OPD injects document knowledge far less efficiently than simple off-policy distillation. We trace this reversal to two failures that are structural, yet sparse across token positions. Reverse KL weights its gradient by the student distribution and therefore barely lifts document-specific tokens the student considers unlikely; after forward KL restores this signal, student rollouts expose later positions on the target trajectory only after generating the preceding tokens correctly, so coverage deteriorates with target length. Motivated by the concentration of teacher-student divergence at these hard positions, we 
    
[^193]: 更少token，更多答案：面向多视角驾驶问答的Ornith风格VLM可验证奖励微调

    Fewer Tokens, More Answers: Verifiable-Reward Tuning of an Ornith-Style VLM for Multi-View Driving QA

    [https://arxiv.org/abs/2611.00833](https://arxiv.org/abs/2611.00833)

    该论文用可验证奖励（GRPO）微调Ornith风格视觉语言模型用于多视角驾驶问答，发现在256 token的紧凑生成预算下能显著提升答案抽取率与准确率，且收益主要来自将原本超长、无法抽取的推理输出转化为可用答案。

    

    具备推理能力的模型衍生版本必须在部署预算内将中间推理过程转化为可用的答案。我们在一个经可验证奖励适配于多摄像头驾驶问答任务的Ornith检查点上研究这一转化过程。配对的存档输出显示，在选定400道题的开发筛选集、256 token上限下，GRPO将答案抽取率从78.25%提升至90.25%，准确率从55.50%提升至63.75%。在39次由错误转为正确的转变中，有33次源于基座模型原本无法抽取答案的响应；基座模型的87个不可抽取响应全部触及token上限。在同一批问题上的检查点验证重构显示，512 token下的首次通过准确率提升小得多：仅+0.75个百分点，而256 token下为+8.25个百分点。后续基于答案提示的续写被单独分析；它对基座模型的帮助更大，并缩小了适配器带来的差异。在完整的2,511题开发集上，这一带恢复机制的流水线降低了……（摘要在此处截断）

    arXiv:2611.00833v2 Announce Type: cross  Abstract: Reasoning-enabled model derivatives must turn intermediate deliberation into a usable answer within a deployment budget. We study that transition in an Ornith checkpoint adapted to multi-camera driving question answering with verifiable rewards. Paired archived outputs show that, on a selected 400-question development screen with a 256-token cap, GRPO increases answer extraction from 78.25% to 90.25% and accuracy from 55.50% to 63.75%. Of 39 wrong-to-correct transitions, 33 begin with an unextractable base response. The base's 87 unextractable responses all reach the token cap. Checkpoint-verified reconstruction on those same questions gives a much smaller first-pass accuracy gain at 512 tokens: +0.75 percentage points, versus +8.25 at 256. A later answer-cued continuation is analyzed separately; it helps the base more and narrows the adapter difference. On the full 2,511-question development set, this recovery-enabled pipeline reduces
    
[^194]: ODC：用于改善推理阶段文生图语义对齐的正交漂移校正

    ODC: Orthogonal Drift Correction for Improved Text-to-Image Semantic Alignment at Inference

    [https://arxiv.org/abs/2611.00819](https://arxiv.org/abs/2611.00819)

    提出正交漂移校正（ODC），一种无需重新训练模型的推理时引导方法，通过两阶段过程识别并校正语义漂移，直接细化用户提示的文本嵌入，从而提升文生图与文本提示的语义对齐。

    

    文生图模型在根据文本描述生成高质量图像方面取得了显著成功。然而，这些模型常常难以应对“语义漂移”问题，即生成的输出无法与复杂或细微的文本提示精确对齐。尽管近期的方法尝试解决属性绑定或对象存在性方面的语义错误，但仍然缺乏一种更全面的方法，能够通过直接细化初始用户提示的文本嵌入来解决这些问题。在本工作中，我们提出了正交漂移校正，这是一种推理时引导方法，旨在缓解语义漂移，且无需模型重新训练或额外的用户输入。ODC通过两阶段过程引导图像生成：在第一阶段，它通过在共享的视觉-语言嵌入空间中将初始生成的图像与用户提示进行对比评估来识别语义漂移；随后，它分离出……

    arXiv:2611.00819v1 Announce Type: cross  Abstract: Text-to-image models have achieved remarkable success in generating high-quality images from textual descriptions. However, they often struggle with "semantic drift," where the generated output fails to precisely align with complex or nuanced text prompts. While recent approaches have attempted to address semantic errors regarding attribute binding or object presence, there remains a gap for a more holistic method that addresses these issues by directly refining the text embeddings of the initial user prompt. In this work, we introduce Orthogonal Drift Correction (ODC), an inference-time guidance method designed to mitigate semantic drift without requiring model retraining or additional user inputs. ODC guides image generation through a two-stage process. In the first stage, it identifies the semantic drift by evaluating the initially generated image against the user prompt in a shared vision-language embedding space. It then isolates 
    
[^195]: 引导而非告知：利用可达性对齐的自蒸馏突破推理瓶颈

    Steer, Don't Tell: Breaking Reasoning Plateaus with Reachability-Aligned Self-Distillation

    [https://arxiv.org/abs/2611.00796](https://arxiv.org/abs/2611.00796)

    提出自引导推理器（SSR），发现自我蒸馏中“教师答案正确”不等于“学生可达到”的问题，并通过学生支持投影将特权教师分布对齐到学生模型可达范围，从而在无需外部专家的情况下突破大推理模型在难题上的推理平台期。

    

    针对大型推理模型的后训练在具有挑战性的问题上仍然效果不足，因为这些问题难以采样到正确的解题轨迹。尽管先前的工作通过将专家轨迹的监督微调与强化学习相结合来解决这一难题，但外部专家并不总是可得。现有的自我改进方法转而利用参考解来构建一个“特权自我教师”。然而，特权教师答案的正确性并不代表学生模型的可达到性。以参考解为条件的训练会使教师分布偏向于在无特权学生模型下不太可能出现的动作，从而产生看似“正确”但实际不可达、甚至是畸形学习目标的轨迹。因此，我们将可达性对齐确定为自构建监督的关键，并提出了自引导推理器。SSR通过学生支持投影来实现这一原则，即对特权教师分布进行转换……（原文摘要在此处截断）

    arXiv:2611.00796v1 Announce Type: new  Abstract: Post training for large reasoning models remains insufficient on challenging problems, whose correct trajectories are hard to sample. Though prior works address this difficulty by integrating supervised fine tuning on expert trajectories with reinforcement learning, such external experts are not always available. Existing self improvement methods instead use reference solutions to create a privileged self-teacher. Yet privileged teacher correctness does not imply student reachability. Solution conditioning shifts the teacher distribution toward actions unlikely under the unprivileged student, producing seemingly "correct" trajectories that are unreachable or even mal-formed learning targets. Accordingly, we identify reachability alignment as the key to self constructed supervision and propose Self-Steered Reasoner (SSR). SSR realizes this principle through student support projection, which transforms the privileged teacher distribution i
    
[^196]: CheXPoT：对视觉语言模型生成的放射报告进行事后视觉验证与修正

    CheXPoT: Post-Hoc Visual Verification and Correction of VLM-Generated Radiology Reports

    [https://arxiv.org/abs/2611.00785](https://arxiv.org/abs/2611.00785)

    CheXPoT提出了一种程序化思维框架，将视觉语言模型生成的放射报告分解为谓词并与视觉证据逐一比对验证，事后删除无法在图像中找到依据的幻觉性发现，从而以远低于训练类方法的计算成本提升报告质量。

    

    视觉语言模型（VLM）在胸部X光片报告生成方面展现出巨大潜力，但由于其生成的发现中包含与图像不符的幻觉内容，实际应用仍然受限。纠正这些幻觉并提升整体报告质量仍是一项重大挑战。以往的工作通常通过强化学习、偏好优化或思维链监督来精调模型以解决该问题，但往往需要巨大的计算成本。为填补这一空白，我们提出了CheXPoT，一个程序化思维框架，它将生成的每一条发现与图像特征进行核对验证，并删除无法在图像中找到依据的发现。CheXPoT利用RadGraph和大语言模型将生成的报告分解为谓词，将这些谓词与基于DenseNet-121和Grad-CAM构建的视觉证据表进行对齐，执行PoT验证代码，并在删除缺乏图像支撑的谓词后重写报告。在一个包含4,145张图像的保留测试集上，CheXPoT取得了显著的……（原文摘要在此处截断）

    arXiv:2611.00785v1 Announce Type: new  Abstract: Vision-language models (VLMs) have shown promise for chest radiograph report generation, but adoption remains limited because of hallucinated findings that are not grounded in the image. Correcting these hallucinations and improving overall report quality remains a significant challenge. Prior work has addressed this problem by refining models with reinforcement learning, preference optimization, or chain-of-thought supervision, often at substantial compute cost. To address this gap, we introduce CheXPoT, a program-of-thought (PoT) framework that verifies each generated finding against image features and removes those it cannot ground. CheXPoT decomposes generated reports into predicates with RadGraph and an LLM, grounds these predicates against a DenseNet-121 and Grad-CAM visual evidence table, executes PoT verification code, and rewrites reports after removing unsupported predicates. On a 4,145-image held-out split, CheXPoT yields stat
    
[^197]: ConstraintMatch：通过受限区域证据选择实现组合式图文匹配

    ConstraintMatch: Compositional Image-Text Matching through Constrained Regional Evidence Selection

    [https://arxiv.org/abs/2611.00784](https://arxiv.org/abs/2611.00784)

    ConstraintMatch是一种免训练的推理方法，通过比较实体内共享区域与独立区域选择之间的分数差距，将实体级区域一致性转化为显式匹配代价，从而修正图文匹配中属性-实体绑定错误的问题。

    

    局部图文匹配能够为单个概念找到强有力的证据，却容易偏向属性-实体绑定错误的描述文本。我们提出ConstraintMatch，这是一种免训练的推理方法，将实体级的区域一致性转化为显式的匹配代价。在同一个冻结的条件-区域相似度矩阵上，该方法将每个类别和属性各自独立选择的裁剪区域，与每个实体内部共享的裁剪区域选择进行比较。两者之间的差距量化了放松共享区域约束所带来的分数优势。来自同一图像的空间参照通过在联合平移实体属性图的同时保持类别证据固定，来校准这一差距。逐实体的截断处理防止了残差之间的相互抵消，缩放后的平均代价从全局相似度中扣除。ConstraintMatch使用池化嵌入，且空间校准无需额外的编码器调用。

    arXiv:2611.00784v1 Announce Type: cross  Abstract: Local image--text matching can find strong evidence for individual concepts yet favor captions with incorrect attribute--entity bindings. We introduce ConstraintMatch, a training-free inference method that turns entity-level regional consistency into an explicit matching cost. On the same frozen condition--region similarity matrix, it compares independent crop selection for each category and attribute with a shared crop choice within each entity. Their gap quantifies the score advantage obtained by relaxing the shared-region constraint. Spatial references from the same image calibrate this gap by keeping category evidence fixed while jointly shifting an entity's attribute maps. Per-entity clipping prevents cancellation across residuals, and the scaled mean cost is subtracted from global similarity. ConstraintMatch uses pooled embeddings, and spatial calibration requires no additional encoder calls. With parameters fixed per backbone ac
    
[^198]: DuSE-Bench：面向演化对话与文档联合推理的基准测试

    DuSE-Bench: Benchmarking Joint Reasoning over Evolving Dialogue and Documents

    [https://arxiv.org/abs/2611.00755](https://arxiv.org/abs/2611.00755)

    该论文提出了DuSE-Bench基准，包含五个场景下的1,000个任务（其中700个任务的前提同时源于对话与文档），用于系统评估信息助手在演化对话与文档上的联合推理能力。

    

    持久的网络信息助手必须将不断演化的文档事实与跨会话中建立和修订的用户需求相结合。然而，对两个信息源的变化如何影响响应需求与证据归因的系统性评估仍然有限。为填补这一空白，我们提出了双源演化基准（DuSE-Bench），该基准包含五个场景下的1,000个任务，其中700个任务的所需前提同时来源于对话和文档。一个不断演化的支持图引导任务构建，并保留从更新事实和复用结论到原始对话轮次与文档段落的链接。任务标注区分了依据来源与状态依赖性，支持对响应内容、响应类型和来源归因的分别评估。在11种基线配置上的实验揭示了模型在演化对话与文档上进行联合推理时面临的持续挑战，联合任务上的内容准确率……

    arXiv:2611.00755v1 Announce Type: new  Abstract: Persistent Web information assistants must combine evolving document facts with user requirements established and revised across sessions. Yet systematic evaluation of how changes in both sources affect response requirements and evidence attribution remains limited. To address this gap, we introduce Dual-Source Evolution Benchmark (DuSE-Bench), a benchmark comprising 1,000 tasks across five scenarios, including 700 whose required premises originate in both dialogue and documents. An evolving Support Graph guides task construction and preserves links from updated facts and reused conclusions to original turns and passages. Task annotations distinguish grounding source from state dependence and enable separate evaluation of response content, response type, and source attribution. Experiments across 11 baseline configurations reveal persistent challenges in joint reasoning over evolving dialogue and documents, with content accuracy on joint
    
[^199]: 通过词元-事件证据替换构建预算受限的KGW水印规避方法

    Constructing Budgeted KGW Evasion through Token-Event Evidence Replacement

    [https://arxiv.org/abs/2611.00726](https://arxiv.org/abs/2611.00726)

    提出一种预算受限的KGW水印规避方法：仅用独立模型的普通短语改写和精确覆盖搜索，在给定编辑预算内最大化移除水印检测证据，且无需密钥、检测器反馈或查询水印模型，即可显著削弱水印检测。

    

    KGW水印通过对采样进行偏置来累积基于密钥的检测证据。我们利用普通短语改写，通过词元-事件证据替换构建了一种预算受限的规避方法。该构造方法联合地从独立模型提出的LLM替代方案中选择组合，并在完整输出上度量全局词元编辑距离。在预算内可达的最大成本处，精确覆盖搜索最大化了对原始计分词元对的移除。该方法仅使用已知的分词器，无需密钥、检测器反馈或对水印模型的任何查询。精确核算将被移除的源证据、新产生的证据以及长度效应区分开来，并以初始水印强度设定所需的削减量。独立的带符号事件分析发现，平均抑制主要来自对水印富集源事件的移除。在短文本段落上进行受控的、按出现次数计数的、单词元上下文的KGW测试中，82个合格的、初始被检测到的来源中有39个产生了……（原文摘要在此处截断）

    arXiv:2611.00726v1 Announce Type: new  Abstract: KGW watermarks bias sampling to accumulate keyed detection evidence. We construct budgeted evasion through token-event evidence replacement using ordinary phrase rewrites. Our construction jointly selects compositions of LLM-proposed alternatives from an independent model and measures global token-edit distance on complete outputs. At the largest attainable cost within the budget, exact coverage search maximizes removal of original scored token pairs. It uses a known tokenizer without the key, detector feedback or queries to the watermarked model. Exact accounting separates source evidence removed, evidence created and length effects, with initial strength setting the required reduction. Separate signed-event analysis found that mean suppression arose mainly from removing watermark-enriched source events. In controlled, occurrence-counted, one-token-context KGW tests on short passages, 39 of 82 qualified, initially detected sources yield
    
[^200]: 学习时间线保持型语音到语音翻译

    Learning Timeline-Preserving Speech-to-Speech Translation

    [https://arxiv.org/abs/2611.00690](https://arxiv.org/abs/2611.00690)

    提出时间线保持型语音到语音翻译新任务及端到端自回归系统 TransDubber，可直接在共享绝对时间轴上保留源语音的结束点和显著内部停顿，无需推理时的时间控制或事后校正。

    

    语音到语音翻译（S2ST）日益能够保留说话的内容与声音风格，但并不一定保留语音和停顿发生的时间点。对于配音（dubbing）以及其他具有时间对齐要求的应用来说，仅匹配总时长是不够的：语音和停顿还必须出现在相应的时刻。我们提出了时间线保持型语音到语音翻译（timeline-preserving S2ST）这一任务形式，其目标是在共享的绝对时间轴上保留源语音的结束点以及显著的内部停顿。我们提出了 TransDubber，这是一个端到端的自回归系统，它直接从对齐的目标波形中学习这种时间结构，无需推理时的帧级时间控制或事后时间校正。为了提供这种监督信号，我们通过在由源语音定义的时间骨架上进行翻译并合成目标语音，构建了 64.2 万条时长精确匹配的英中语音对。训练结合了渐进式监督课程学习与验证器引导的强化学习，以平衡……（原文摘要在此处截断）

    arXiv:2611.00690v1 Announce Type: cross  Abstract: Speech-to-speech translation (S2ST) increasingly preserves what is said and how it sounds, but not necessarily when speech and silence occur. For dubbing and other temporally grounded applications, matching total duration is not enough: speech and pauses must also occur at corresponding moments. We formulate timeline-preserving S2ST, which aims to preserve the source endpoint and salient internal pauses on a shared absolute timeline. We present TransDubber, an end-to-end autoregressive system that learns this temporal structure directly from aligned target waveforms, without inference-time frame-level timing control or post-hoc temporal correction. To provide this supervision, we construct 642K exactly duration-matched English-Chinese speech pairs by translating and synthesizing target speech on source-defined temporal scaffolds. Training combines a progressive supervised curriculum with verifier-guided reinforcement learning to balanc
    
[^201]: 先例标签与事实以相反方向影响大语言模型的审批决策

    Precedent Labels and Facts Shift LLM Approval Decisions in Opposite Directions

    [https://arxiv.org/abs/2611.00672](https://arxiv.org/abs/2611.00672)

    本研究通过分离先例案例的“事实”与“裁决标签”发现，二者会以相反方向影响大语言模型的审批决策——裁决标签使模型输出与先例一致（批准率提升30至55个百分点），而事实内容则将模型决策推向相反方向。

    

    当语言模型根据书面政策来决定退款或索赔请求时，系统可以向其展示相似的过往案例。每个过往案例包含两部分内容：案例的事实以及记录在案的裁决结果。本研究将这两部分分离：模型会看到一个政策、一个请求和三个过往案例，我们改变其中记录的裁决结果或事实，同时保持请求本身不变。我们选取了那些处于“边缘”状态的请求：在没有过往案例参考的情况下，DeepSeek V4.1 Flash 在四次尝试中对每个请求时而批准、时而拒绝。当过往案例与请求几乎相同（即仅改动少量细节的请求事实）时，将其记录的裁决从“拒绝”改为“批准”，会使 DeepSeek V4.1 Flash、GPT-5.6 Luna 和 Claude Haiku 5.5 在为本研究所撰写的政策下的批准率提高约 30 至 55 个百分点，且 DeepSeek 和 Luna 在九个由真实政策改写而来的公开政策上也表现出同样的效果。而事实则产生相反方向的推动：在不展示记录裁决的情况下……（摘要原文在此处截断）

    arXiv:2611.00672v1 Announce Type: new  Abstract: When a language model decides refund or claim requests under written policies, the system can show it similar past cases. Each past case has two parts: its facts and the decision recorded for it. We separate the two. A model sees a policy, a request and three past cases, and we change either the recorded decisions or the facts, keeping the request fixed. We select requests that are close calls: deciding without past cases, DeepSeek V4.1 Flash approved each in some of four tries and denied it in others. When the past cases are near-copies of the request (its facts with a few details changed), switching their recorded decisions from DENY to APPROVE raises approval by about 30 to 55 percentage points in DeepSeek V4.1 Flash, GPT-5.6 Luna and Claude Haiku 5.5 on policies written for the study, and DeepSeek and Luna show the same on nine public policies paraphrased from real ones. The facts push the other way: shown without their recorded deci
    
[^202]: 以部署精度进行审计：量化重塑潜意识学习到的偏好

    Audit at the Precision You Deploy: Quantization Reshapes a Subliminally Learned Preference

    [https://arxiv.org/abs/2611.00665](https://arxiv.org/abs/2611.00665)

    4-bit量化会因量化器的不同而显著放大或抑制模型中潜意识学习到的隐藏偏好，因此安全审计必须在部署所用的量化精度下进行。

    

    潜意识学习使AI模型能够从看似无关的训练数据（如数字序列）中继承隐藏的偏好。安全审计可能在标准16位精度下评估模型，而实际部署则可能使用压缩、量化后的权重。我们探究量化是否会改变潜意识偏好的表达，以及是什么驱动了由此产生的变化。为了研究这一问题，我们将“最喜欢的动物：熊猫”这一偏好植入Qwen2.5-7B-Instruct模型。我们将其净强度定义为在相同量化设置下，潜意识模型与中性训练对照组之间平均熊猫回答概率的差异。在不同压缩级别上应用该指标揭示了鲜明的对比：在8位量化下行为保持稳定，而4位量化则会根据量化器的不同引发严重的、方向相反的偏移。半二次量化（HQQ）会放大该偏好，提高潜意识训练模型的熊猫回答概率……

    arXiv:2611.00665v1 Announce Type: new  Abstract: Subliminal learning allows AI models to inherit hidden preferences from seemingly unrelated training data, such as number sequences. Safety audits may evaluate models at standard 16-bit precision, while deployments may use compressed, quantized weights. We ask whether quantization changes the expression of a subliminal preference and what drives any resulting change. To investigate, we implant a "favourite animal: panda" preference into Qwen2.5-7B-Instruct. We define its net strength as the difference in mean panda-answer probability between subliminal models and neutrally trained controls under the same quantization setting. Applying this metric across different compression levels reveals a stark contrast. While behaviour remains stable at 8 bits, 4-bit quantization triggers severe, opposing shifts depending on the quantizer. Half-Quadratic Quantization (HQQ) amplifies the preference, raising the subliminally trained model's panda proba
    
[^203]: Heirloom：让手机上的个性化能力跨基模型代际传承

    Heirloom: Carrying a Phone's Personalization Across Base-Model Generations

    [https://arxiv.org/abs/2611.00650](https://arxiv.org/abs/2611.00650)

    Heirloom提出了一种无需保留用户带标签数据的方法，通过将旧个性化模型作为教师在无标签数据上生成预测，在手机上将个性化适配器迁移到升级后的新基模型，且内存中仅需同时保留一个基模型。

    

    设备端语言模型可以通过在私有数据上训练的适配器来学习用户的偏好。然而，基模型升级可能会改变层结构或词表，导致个性化适配器不再兼容。用非个性化的适配器替换它会损失大部分个性化能力，而重新训练则需要保留用户带标签的历史数据。我们提出了Heirloom，这是一个无需保留标签即可在手机上将个性化适配器迁移到新基模型的系统。Heirloom使用旧的个性化模型作为教师，在存储的无标签输入、厂商提供的提示或由新基模型生成的提示上收集其预测结果。随后，它卸载旧的基模型，并训练新模型的适配器以匹配这些预测，同一时间内存中只保留一个基模型。我们在两个模型家族的三次代际升级、一次保持架构不变的更新以及三个个性化任务上对Heirloom进行了评估。

    arXiv:2611.00650v1 Announce Type: cross  Abstract: On-device language models can learn a user's preferences through adapters trained on private data. However, a base-model upgrade may change the layers or vocabulary, leaving the personal adapter incompatible. Replacing it with a non-personal adapter loses much of the personalization, while retraining requires retaining the user's labeled history. We present Heirloom, a system that transfers a personal adapter to a new base model on the phone without retaining labels. Heirloom uses the old personal model as a teacher, collecting its predictions on stored unlabeled inputs, vendor-provided prompts, or prompts generated by the new base. It then unloads the old base and trains an adapter for the new model to match those predictions, keeping only one base model in memory at a time. We evaluate Heirloom across three generation upgrades from two model families, an architecture-preserving refresh, and three personalization tasks. On movie taggi
    
[^204]: ElectroChemLLM：面向高保真电化学信息学的多模态大语言模型流水线

    ElectroChemLLM: A Multimodal LLM Pipeline for High-Fidelity Electrochemical Informatics

    [https://arxiv.org/abs/2611.00638](https://arxiv.org/abs/2611.00638)

    提出多模态大语言模型流水线 ElectroChemLLM，通过混合文档解析与视觉 LLM 编排相结合，从科学 PDF 中大规模提取涵盖电解液成分、电极材料、电池构型及物化性质的可验证结构化电化学数据，克服了纯文本架构、上下文碎片化与数值幻觉的局限。

    

    电化学是解决诸多当代挑战的核心，包括面向可持续未来的储能系统，而其进展高度依赖于以往研究成果中的数据。现有的自动化流水线受限于纯文本架构、上下文碎片化和数值幻觉，这些问题共同阻碍了性质数据库的精确、大规模整理。我们提出了 ElectroChemLLM——一个多模态信息提取流水线，能够从非结构化的科学 PDF 文献中大规模地联合识别结构化电化学记录，涵盖电解液成分、电极材料、电池构型及其测得的物理化学性质。通过将混合文档解析与具备视觉能力的大语言模型编排相结合，该流水线可从文本、表格和图表中提取可验证的结构化数据。该流水线在我们人工标注的 232 篇论文基准数据集的电池电解液子集上进行了评估。

    arXiv:2611.00638v1 Announce Type: new  Abstract: Electrochemistry is central to solving many modern-day challenges, including energy storage systems for a sustainable future, and its advancement relies heavily on the data from past findings. Existing automated pipelines are limited by text-only architectures, context fragmentation, and numerical hallucinations - collectively preventing accurate, large-scale curation of property databases. We introduce ElectrochemLLM, a multimodal information extraction pipeline that jointly identifies structured electrochemical records encompassing electrolyte compositions, electrode materials, cell configurations, and their measured physicochemical properties from unstructured scientific PDFs at scale. By combining hybrid document parsing with vision-capable LLM orchestration, the pipeline extracts verifiable, structured data across text, tables, and figures. Evaluated on the battery-electrolyte subset of our manually annotated benchmark of 232 papers
    
[^205]: 通过结构化多级建模增强语言模型推理能力

    Enhancing Language Model Reasoning with Structured Multi-Level Modeling

    [https://arxiv.org/abs/2611.00616](https://arxiv.org/abs/2611.00616)

    提出多级推理（MLR）框架，将长思维链生成建模为“高层规划—低层执行”的两级随机过程，通过轻量级LoRA模块实现结构化高层规划，从而解决单一策略模型在长时程推理中的规划漂移问题。

    

    推理时扩展通过延长模型的思维链（CoT）来增强其推理能力。然而，现有方法通常依赖于通过结果奖励强化学习（RL）训练的单一策略，这往往会出现长时程规划失败的问题，即隐式规划偏离有效策略，对于容量有限的小型语言模型尤其如此。为了解决这个问题，我们提出了多级推理（MLR），将长思维链生成重新表述为一个两级随机过程。高层规划器生成结构化的步骤描述符，同时指定推理模式和语义子目标。低层执行器随后以这些描述符为条件生成详细的推理内容，形成一个交替进行的“规划—执行”循环。为了保持可扩展性，我们采用了极简设计：基础模型充当低层策略，而一个轻量级的LoRA模块实现高层策略。在训练方面，我们观察到……

    arXiv:2611.00616v1 Announce Type: new  Abstract: Inference-time scaling enhances a model's reasoning by extending its chain-of-thought (CoT). However, existing approaches typically rely on a single policy trained with outcome-reward reinforcement learning (RL), which often suffers from long-horizon plan failures where the implicit plan drifts from valid strategies, especially for small LMs with limited capacity. To address this, we propose Multi-Level Reasoning (MLR), which reformulates long-CoT generation as a two-level stochastic process. A high-level planner generates structured step descriptors specifying both the reasoning mode and the semantic subgoal. The low-level executor then produces detailed reasoning conditioned on these descriptors, forming an alternating plan--execute loop. To maintain scalability, we adopt a minimal design where the base model serves as the low-level policy and a lightweight LoRA module implements the high-level policy. For training, we observe that out
    
[^206]: 稀疏自编码器归因图由边际决定

    Sparse Autoencoder Attribution Graphs Are Decided at the Margin

    [https://arxiv.org/abs/2611.00606](https://arxiv.org/abs/2611.00606)

    该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。

    

    基于稀疏自编码器（SAE）特征的归因图被用于预测消融操作的效果，并在不同训练种子之间进行比较。TopK SAE 在每个 token 上保留最大的预激活值，而特征的“边际”是其预激活值到截断阈值的距离。在来自三个 Pythia 模型的五个公开 TopK SAE 对上，将一个特征置零会在至少 0.74 的情形中使下一层的某个特征跨过截断阈值，而这些跨越承载了响应的大部分能量。对于将某个特征的编码放大三倍的写入操作，基于先前激活特征构建的图的中位相对误差达到 0.247 或更高。而“边际图”方法为每个预激活添加一阶变化并重新排序，在相同计算成本下，误差至多为 0.033。在中位数上，跨越特征的解码变化在范数上达到模型自身变化的至少 1.24 倍。在同一测试床的不同种子之间，特征之间的稀疏映射可以解释一个图的 0.85。

    arXiv:2611.00606v1 Announce Type: cross  Abstract: Attribution graphs over the features of sparse autoencoders (SAEs) are used to predict what an ablation does and are compared across training seeds. A TopK SAE keeps the largest preactivations at each token, and a feature's margin is its preactivation's distance to the cutoff. On five public TopK SAE pairs from three Pythia models, zeroing one feature moves a next layer feature across the cutoff in 0.74 of cases or more, and the crossings carry most of the response's energy. Over writes that scale a feature's code by three factors, the graph built on the features active beforehand has a median relative error of 0.247 or more. The margin graph adds the first order change to every preactivation and reranks, at the same cost, and errs by at most 0.033. At the median, the crossing features' decoded change is 1.24 times the model's own change in norm or more. Between seeds of one testbed, sparse maps between features explain 0.85 of a graph
    
[^207]: 多重估计引导并不能统一地消除语言模型判断中的偏差

    Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments

    [https://arxiv.org/abs/2611.00597](https://arxiv.org/abs/2611.00597)

    本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。

    

    在单次回复中引导模型给出五个加权估计值，是否会改变语言模型对信息的响应方式？在锚定、后见之明、基础比率、合取以及现状/默认等任务中，我们将直接数值回答与多重估计响应格式进行对比，同时比较了不同的来源框架、显式随机线索以及概率一致性约束。我们评估了三种已完成配置——gpt-6.1-sol、claude-sonnet-5.5 以及 Qwen3-8B（禁用推理，进行描述性分析）。在前两种配置中，直接引导产生的无关锚定指数和现状效应接近于零，同时模型仍能对信息性来源和知情建议保持敏感。在23项预注册的验证性测试中，有六项在经过 Holm 校正后被拒绝；然而，这六项测试的绝对点估计值均低于预先设定的最小感兴趣效应量，但并未据此建立等效性结论。此外，虽然分离引导与联合引导的（摘要在此处截断）

    arXiv:2611.00597v1 Announce Type: new  Abstract: Does eliciting five weighted estimates within a single response change how a language model responds to information? Across anchoring, hindsight, base-rate, conjunction, and status-quo/default tasks, we compare direct numerical answers with a multi-estimate response format, contrasting source framings, explicitly random cues, and probability-consistency constraints. We evaluate three completed configurations--gpt-6.1-sol, claude-sonnet-5.5, and Qwen3-8B (reasoning disabled, analyzed descriptively). In the first two configurations, direct elicitation yielded near-zero irrelevant-anchor indices and status-quo effects, while remaining responsive to informative sources and informed advice. Of 23 preregistered confirmatory tests, six rejected after Holm adjustment; however, all six absolute point estimates fell below their prespecified smallest effect sizes of interest without establishing equivalence. Moreover, while separate versus joint el
    
[^208]: Saidzoda-Tajik-E5-Large：微调560M参数嵌入模型用于塔吉克语检索

    Saidzoda-Tajik-E5-Large: Fine-Tuning a 560M Embedder for Tajik Retrieval

    [https://arxiv.org/abs/2611.00592](https://arxiv.org/abs/2611.00592)

    提出了首个公开描述的塔吉克语文本检索嵌入模型Saidzoda-Tajik-E5-Large，通过自建语言模型合成23.5万余个问题-段落对微调5.6亿参数的多语言嵌入模型，在法律文本检索的前十命中率从80.2%提升至93.6%，同时保持英语和俄语能力不退化。

    

    找到能够回答问题的段落是基于文档构建的大多数实用系统的第一步，而对于塔吉克语，我们没有找到任何公开描述的相关模型。我们提出了Saidzoda-Tajik-E5-Large，一个基于multilingual-e5-large-instruct（5.6亿参数）构建的塔吉克语文本嵌入模型。我们让自己的塔吉克语语言模型为来自塔吉克斯坦法律、税务文件、维基百科和网络文本的段落编写问题，并在235,140个问题-段落对上训练了一个轮次，其中困难负例由另一个嵌入模型挑选，并混入英语和俄语的配对数据以防止模型遗忘这些语言。在模型从未见过的法律文本上，93.6%的问题能在前十名结果中找到正确段落，较之前的80.2%有所提升；在税务文件上，这一比例从82.6%提升至94.5%。在Belebele——我们唯一一个由人工编写的测试集上，提升较小且未能得到确证（从92.2%到95.2%，置信区间存在重叠），而俄语和英语的表现没有变化。

    arXiv:2611.00592v1 Announce Type: new  Abstract: Finding the passage that answers a question is the first step of most practical systems built on documents, and for Tajik we found no openly described model for it. We present Saidzoda-Tajik-E5-Large, a Tajik text embedder built from multilingual-e5-large-instruct (560M parameters). We asked our own Tajik language model to write questions for passages from Tajik laws, tax documents, Wikipedia and web text, and trained on 235,140 question and passage pairs for one epoch, with hard negatives chosen by another embedder and English and Russian pairs mixed in so the model would not forget those languages. On laws the model had never seen, the right passage is among the top ten results for 93.6% of questions, up from 80.2%; on tax documents, 94.5% against 82.6%. On Belebele, the one test we have that was written by people, the gain is smaller and not confirmed (92.2% to 95.2%, with touching intervals), and Russian and English do not change. On
    
[^209]: TVP：对投机解码而言，总变差几乎就是你所需要的一切

    TVP: TV is Almost All You Need for Speculative Decoding

    [https://arxiv.org/abs/2611.00588](https://arxiv.org/abs/2611.00588)

    该论文证明总变差（TV）距离在草稿模型训练中匹敌或超越已有目标，并提出基于期望接受长度的TVP损失函数族，通过幂参数插值解决梯度消失与后位置饥饿问题，从而提升投机解码性能。

    

    投机解码通过一个起草模型加速大语言模型推理，起草模型提出的token由目标模型并行验证。在无损投机采样下，一个草稿token被接受的概率等于1减去目标分布与草稿分布之间的总变差（TV）距离，然而起草模型通常却使用交叉熵（CE）或KL散度进行训练。我们首先证明，在匹配且经过良好调优的训练条件下，TV在微调场景下能够匹敌或超越几种已发表的草稿训练目标；而在从零训练场景下，固定的TV与KL混合目标同样能做到这一点。我们还识别出两个仍然存在的问题：TV在覆盖不足的token上会出现梯度消失，以及接受长度目标中的前缀乘积会导致后面的位置“饥饿”。为同时解决这两个问题，我们推导出TVP，一个基于期望接受长度的损失函数族，其幂参数可在覆盖导向与接受导向的目标之间进行插值。将幂参数从零退火至某一值，能够……（原文在此截断）

    arXiv:2611.00588v1 Announce Type: cross  Abstract: Speculative decoding accelerates LLM inference with a drafter whose proposed tokens are verified by the target model in parallel. Under lossless speculative sampling, a draft token is accepted with probability one minus the total variation (TV) distance between the target and draft distributions, yet drafters are often trained with cross-entropy (CE) or KL. We first show that under matched and well-tuned training, TV matches or surpasses several published draft-training objectives when fine-tuning, and a fixed TV and KL mixture does the same when training from scratch. We identify two remaining problems, TV's vanishing gradients on undercovered tokens and the prefix products that starve later positions in acceptance-length objectives. To address both, we derive TVP, a loss family based on expected acceptance length, whose power parameter interpolates between coverage and acceptance oriented objectives. Annealing the power from zero to 
    
[^210]: Transformer注意力机制的锥优化认证

    Conic Certification of Transformer Attention

    [https://arxiv.org/abs/2611.00585](https://arxiv.org/abs/2611.00585)

    本文建立了Transformer注意力认证的锥规划理论，证明认证难点在于查询与键的双线性交互（NP难且不可近似），并提出指数锥矩松弛方法，给出误差按扰动半径二次界定且阶数不可改进的认证保证。

    

    注意力机制是大语言模型的计算核心，然而现有的验证器虽能界定注意力头的输出在输入扰动下的偏移范围，却无法保证其界有多紧，因此无法将弱证书与脆弱模型区分开来。我们发展了一套注意力认证的锥规划理论来提供这一保证。我们证明，认证的困难并非来自softmax，而是来自变化的查询与键之间的双线性交互。这种交互使得盒约束下的认证问题成为NP难的，且一旦注意力logits能够移动一个常数幅度，问题便不可近似。相比之下，当查询固定或词元替换可以列举时，精确的最坏情况可以高效计算。随后，基于指数锥的矩松弛方法可以为注意力头提供认证，其误差以可计算的量被扰动半径的二次函数所界定，且这一阶数不可再改进。我们还证明了松弛方法……（摘要原文在此处截断）

    arXiv:2611.00585v1 Announce Type: cross  Abstract: Attention is the computational core of large language models, yet the verifiers that bound how far an attention head's output can move under input perturbations give no guarantee on how loose their bounds are. A weak certificate therefore cannot be told apart from a fragile model. We develop a conic programming theory of attention certification that supplies this guarantee. We prove that the difficulty lies not in the softmax but in the bilinear interaction of a varying query with the keys. This interaction makes certification NP-hard over boxes and inapproximable once the attention logits can move by a constant amount. A fixed query or listed token substitutions, by contrast, make the exact worst case efficiently computable. An exponential cone moment relaxation then certifies a head with an error bounded quadratically in the perturbation radius by computable quantities, an order that cannot be improved. We also show that relaxations 
    
[^211]: 从表面的搜索增益到选择的失败：结构化工作流规划中测试时推理的审计

    From Apparent Search Gains to Selection Failures: Auditing Test-Time Inference in Structured Workflow Planning

    [https://arxiv.org/abs/2611.00579](https://arxiv.org/abs/2611.00579)

    该论文将测试时推理审计性地分解为“发现—存留—选中”三个可观测阶段，揭示名义树搜索的收益实际来自其扩展提示词而非真正的搜索结构，且失败主要发生在最终选择环节而非结构验证环节。

    

    arXiv:2611.00579v1 公告类型：新

    arXiv:2611.00579v1 Announce Type: new  Abstract: Test-time search can improve structured planning, but when methods differ in prompts, post-processing, and selection, it is unclear what an observed gain should be attributed to. We study this question on DisasterBench, where models compose typed multi-tool workflows, by decomposing inference into three observable stages: whether a correct workflow is discovered, whether it survives structural validation, and whether it is finally selected. Comparing independent sampling, a nominal tree-search configuration, and multi-operator search under controlled inference budgets on two model families, we find that (i) the nominal tree-search configuration realizes non-adaptive complete-workflow sampling, and independent sampling with its expansion prompt reproduces its candidate discovery without constructing a tree; (ii) structural validation never removes the last correct workflow in our evaluated candidate pools, whereas majority voting leaves a
    
[^212]: 通过语法视角基准区分GPT-5.5 Pro的推理级别

    Distinguishing among reasoning levels of GPT 5.5 Pro via a grammar lens benchmark

    [https://arxiv.org/abs/2611.00555](https://arxiv.org/abs/2611.00555)

    本文提出一种源自计算理论的形式化语法视角基准，成功区分了GPT-5.5系列模型在不同测试时推理级别下的细粒度能力差异，并发现随着测试时计算努力增加，模型与人类的响应分歧逐渐增大，揭示了能力与对齐之间的潜在冲突。

    

    测试时处理努力的程度会影响语言模型的计算能力吗？一种源自计算理论的形式化“语法视角”基准方法，能够在指定的层级上生成新颖数据，每个层级都比其下方的层级具有更强的计算能力。利用这些数据，我们测试了GPT-5.5以及GPT-5.5-Pro-Medium、-High和-XHigh，结果显示模型对测试数据的接受率逐步提升：分别为16%、40%、43%和52%，这说明了该基准有能力辨别不同测试时推理级别下模型之间细粒度的差异。此外，随着测试时计算努力的增加，模型对相同数据的响应与人类的响应逐渐产生分歧。这种日益增加的分歧可能成为更强大语言模型的普遍特征，从而揭示了能力与对齐之间潜在的一般性冲突。

    arXiv:2611.00555v1 Announce Type: new  Abstract: Does the level of test-time processing effort affect a language model's computational capability? A formal grammar-lens benchmark method, derived from the theory of computation, generates novel data at specified levels, each successively more computationally powerful than the levels below. Using this data, we tested GPT-5.5, and GPT-5.5-Pro-Medium, -High, and -XHigh, showing incrementally increased acceptance of test data: 16, 40, 43, and 52\%, respectively, illustrating the benchmark's ability to discern fine-grained differences among models' added test-time reasoning levels. Additionally, as test-time compute effort increases, model responses diverge from human responses to the same data. This increasing disagreement may turn out to be broadly characteristic of more capable language models, introducing the potential of a general capability-versus-alignment conflict.
    
[^213]: Skill++：通过路由感知的技能精炼迈向终身智能体

    Skill++: Towards Lifelong Agents via Routing-Aware Skill Refinement

    [https://arxiv.org/abs/2611.00544](https://arxiv.org/abs/2611.00544)

    该论文提出Skill++，将技能库增长中相似新技能取代旧技能导致的路由干扰问题形式化为终身技能路由（类增量持续学习）问题，并通过路由感知的技能精炼方法，在保持旧技能仍可被选中的同时避免抑制新技能。

    

    大语言模型（LLM）智能体通过技能来扩展自身能力，每个技能包含名称、描述和内容本体。在渐进式披露机制下，LLM路由器仅根据名称和描述来选择技能，并只加载被选中技能的内容本体。随着用户和智能体不断添加新技能，技能库会持续增长。由于路由器依赖这些元数据，一个相似的新技能可能会取代较早的技能，即使较早的技能和路由器本身都没有任何改变，之前正确的路由也会被破坏。对于公开技能而言，这种路由干扰会进一步加剧，因为其描述可能与内容本体不匹配。我们将该问题形式化为终身技能路由，这是一类类增量式持续学习问题。解决方案必须保证较早的技能仍然可选，同时不抑制新技能。现有方法均存在不足：持续重新训练路由器存在灾难性遗忘的风险，而重写方法要么忽略相似技能，要么需要

    arXiv:2611.00544v1 Announce Type: cross  Abstract: Large language model (LLM) agents extend their capabilities with skills, each with a name, a description, and a body. Under progressive disclosure, an LLM router selects skills by name and description alone and loads only the selected bodies. Skill libraries grow continually as users and agents add new skills. Because the router relies on this metadata, a similar new skill can displace an earlier one, breaking previously correct routing even though the earlier skill and the router are unchanged. This routing interference is aggravated for public skills, whose descriptions may not match their bodies. We formulate the problem as lifelong skill routing, a class-incremental continual learning problem. A solution must keep earlier skills selectable without suppressing new ones. Existing approaches fall short, since continually retraining the router risks catastrophic forgetting and rewriting methods either ignore similar skills or require p
    
[^214]: 双轴政策蕴含：一个用于政策变化推理的基准

    Dual-Axis Policy Entailment: A Benchmark for Policy-Change Reasoning

    [https://arxiv.org/abs/2611.00536](https://arxiv.org/abs/2611.00536)

    该论文提出双轴政策蕴含（DAPE）基准，通过同时判断政策对是否属于同一政策领域以及后继政策是延续、扩大、缩小还是逆转原有政策，为政策变化推理提供了更严谨、更细致的评测标准。

    

    arXiv:2611.00536v1 公告类型：新论文 摘要：一项政策是否发生了领域转变，以及由此延伸的，它是延续、扩大、缩小还是逆转了以往的政策，对公共政策研究而言至关重要。我们提出双轴政策蕴含（DAPE），它在两个维度上同时评估一对有序的政策指令：一个二元判断确定这两条指令是否管辖同一政策领域，一个五分类判断确定后一条指令是延续、扩大、缩小还是逆转了先前政策，或者完全属于不同的政策领域。综合来看，这两个判断构成一个五分类标签，这是我们的主要评估指标。这种双轴设计捕捉了政策变化的两个核心维度，而不会将任一维度上的错误混为一个笼统的“不相关”标签，同时也使标注过程更加严谨：标注员在未做出明确、可核查的判断之前，不能轻易将一对难以判定的政策归为领域无关。我们引入了一个由专家标注的基准……

    arXiv:2611.00536v1 Announce Type: new  Abstract: Whether a policy shifts domains and, by extension, continues, broadens, narrows, or reverses previous policy is of paramount importance to the study of public policy. We propose Dual-Axis Policy Entailment (DAPE), which evaluates an ordered pair of policy directives on both dimensions at once: a binary judgment determines whether the two directives govern the same policy domain, and a five-way judgment determines whether the later directive continues, broadens, narrows, or reverses the earlier policy, or governs a different domain altogether. Read together, the two judgments form one five-category label, our primary metric. The two-axis design captures two central dimensions of policy change without conflating errors on either dimension into a generic "irrelevant" label, and it makes annotation more rigorous: a coder cannot dismiss a difficult pair as domain-irrelevant without an explicit, checkable decision. We introduce an expert-annot
    
[^215]: WeSearch：面向微信搜索的统一以模型为中心的检索框架

    WeSearch: A Unified Model-Centric Retrieval Framework for WeChat Search

    [https://arxiv.org/abs/2611.00530](https://arxiv.org/abs/2611.00530)

    WeSearch提出了一种以模型为中心的统一检索框架，通过先检索查询相关簇、再在查询特定候选空间中对原子文档排序，将候选剪枝与细粒度相关性建模解耦，并已作为微信搜索的主要语义检索路径部署，在大规模语料上取得了持续的A/B测试收益。

    

    我们提出了WeSearch，一个面向大规模、持续演进内容搜索的统一以模型为中心的检索框架。现有主流方法为文档分配固定的层次化语义ID，隐含地假设每篇文档遵循单一连贯的语义路径——这一假设对于多主题文档往往不成立。WeSearch则改为检索与查询相关的簇，然后在由此产生的查询特定候选空间内对原子文档进行排序，从而将候选剪枝与细粒度相关性建模解耦。结合相关性感知训练、紧凑的文档表示和增量索引更新，这一设计使模型能够每周刷新（必要时可每天刷新）。WeSearch已作为微信搜索的主要语义检索路径与词汇语法树并行部署，在从数亿到数十亿文档规模的语料库上展现出持续的为期一周的A/B测试收益，为期一个月的全流量A/B测试也证实了……

    arXiv:2611.00530v1 Announce Type: cross  Abstract: We present WeSearch, a unified, model-centric retrieval framework for large-scale, continuously evolving content search. Prevailing methods assign documents fixed hierarchical Semantic IDs, implicitly assuming each document follows a single coherent semantic path-an assumption that often breaks down for multi-topic documents. WeSearch instead retrieves query-relevant clusters and then ranks atomic documents within the resulting query-specific candidate space, decoupling candidate pruning from fine-grained relevance modeling. Combined with relevance-aware training, compact document representations, and incremental index updates, this design enables weekly-or daily when needed-model refreshes. Deployed as the primary semantic retrieval path in WeChat Search alongside a lexical syntax tree, WeSearch shows consistent one-week A/B gains on corpora from hundreds of millions to billions of documents, and a one-month full-traffic A/B test conf
    
[^216]: 面向多轮人机对话的智能体上下文管理

    Agentic Context Management for Multi-Turn Human-AI Conversations

    [https://arxiv.org/abs/2611.00518](https://arxiv.org/abs/2611.00518)

    提出无需训练的智能体对话上下文策展方法AC3，由独立分析器仅从用户消息整合任务规范、审查助手先前输出并据此编辑上下文，从而解决多轮对话中的上下文污染问题。

    

    大语言模型在多轮对话中性能下降的部分原因在于“上下文污染”：模型自身先前的输出会保留在上下文中并锚定后续轮次，因此即使用户已纠正了早期对意图的误读，这种误读仍可能持续存在。助手消息省略（AO）通过丢弃所有助手消息来消除这种污染，但在交互具有指涉性时会失效：后续轮次会回指仅存在于助手轮次中的草稿、决策和工具调用结果。我们提出了智能体对话上下文策展（AC3），这是一种无需训练的方法，由一个独立的分析器（1）仅从用户消息中整合任务规范，（2）据此审查助手的先前输出，（3）编辑上下文——或附加分析结果，或丢弃助手侧内容并依据任务规范重建上下文。第二个发现：污染具有传染性。一个在提取任务规范时看到助手推理过程的分析器……（原文摘要在此处被截断）

    arXiv:2611.00518v1 Announce Type: cross  Abstract: LLMs degrade in multi-turn conversation partly because of context pollution: the model's own earlier outputs stay in context and anchor later turns, so an early misreading of the user's intent can persist after the user corrects it. Assistant omission (AO), dropping every assistant message, removes this pollution but fails when interaction is referential: later turns refer back to drafts, decisions, and tool-call results that exist only in assistant turns. We propose Agentic Conversation Context Curation (AC3), a training-free method in which a separate analyzer (1) consolidates the task specification from the user's messages alone, (2) audits the assistant's prior outputs against it, and (3) edits the context, either appending the analysis or discarding the assistant side and rebuilding the context from the specification. A second finding: pollution is contagious. An analyzer that sees the assistant's reasoning while extracting the sp
    
[^217]: AStockArena：超越收益，我们能否将真实资本托付给金融智能体的决策？

    AStockArena: Beyond Returns, Can We Entrust Real Capital to the Decisions of Financial Agents?

    [https://arxiv.org/abs/2611.00515](https://arxiv.org/abs/2611.00515)

    该论文提出AStockArena——一个针对中国A股市场连续交易的实时决策审计基准，通过涵盖执行可追溯性、证据充分性和金融元认知的八项检查来审计13个模型在3,198个连续决策中的真实行为，超越单纯收益指标以回答金融智能体是否值得托付真实资本的问题。

    

    大型语言模型越来越多地被应用于动态金融交易，然而现有的实时交易基准主要依据累计收益和风险调整后的表现来评估智能体。这类结果指标无法反映交易是否被如实报告、是否有可用证据支撑、以及是否与账户状态保持一致。我们提出了AStockArena，一个面向中国A股市场连续交易的实时决策审计基准。在共享的市场条件和执行约束下，我们评估了13个开源与闭源模型在3,198个连续决策上的表现。AStockArena通过将报告、交易请求、执行响应与账户状态进行对齐，重构了实际提交到账户的操作，并通过八项检查对其进行审计，这些检查涵盖执行可追溯性（O）、证据充分性（RK）和金融元认知（B），同时结合了记录验证与共识专家标注。在1,070个交易点中，72……（原文摘要在此处截断）

    arXiv:2611.00515v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used in dynamic financial trading, yet real-time trading benchmarks mainly evaluate agents by cumulative returns and risk-adjusted performance. Such outcome metrics cannot show whether trades are faithfully reported, supported by available evidence, and consistent with account states. We introduce AStockArena, a real-time decision-auditing benchmark for continuous trading in China's A-share market. Under shared market conditions and execution constraints, we evaluate 13 open- and closed-source models over 3,198 sequential decisions. AStockArena reconstructsthe actions actually posted to accounts by aligning reports, trade requests, execution responses, and account states, and audits them with eight checks spanning execution traceability (O), evidence sufficiency (RK), and financial meta-awareness (B), combining record verification with consensus expert annotation. Of 1,070 trading points, 72.
    
[^218]: 鲁棒性始于目标函数：设计并认证鲁棒的LLM生成启发式算法

    Robustness Starts with the Objective: Designing and Certifying Robust LLM-Generated Heuristics

    [https://arxiv.org/abs/2611.00504](https://arxiv.org/abs/2611.00504)

    TAILOR框架利用候选方案与现有基线在同一工作负载上成本差异的条件风险价值（配对CVaR）来指导LLM合成启发式算法，负的配对CVaR可以在无需计算未知最优解的情况下，同时认证平均性能与尾部性能的改进。

    

    大语言模型（LLM）正越来越多地被用于为计算机系统设计启发式算法。然而，大多数合成框架通过平均性能来评估候选方案，这可能掩盖在特定工作负载上的严重性能退化。即使添加对抗性训练输入也无法消除这些退化，而百分位数目标（例如优化第95百分位数）则忽略了超过所选阈值后退化可能达到的严重程度。我们提出了TAILOR，一个利用条件风险价值（CVaR）来指导LLM合成的框架，该CVaR基于每个候选方案与现有基线在同一工作负载上的成本差异。这种配对CVaR度量的是在最差一部分工作负载上的平均差异。我们证明，负的配对CVaR能够在无需计算未知最优解的情况下，保证在平均性能和相对未知最优解的尾部最优性差距两方面均取得改进。我们发现TAILOR实现了最佳的均值-尾部权衡……

    arXiv:2611.00504v1 Announce Type: cross  Abstract: LLMs are increasingly used to design heuristics for computer systems. However, most synthesis frameworks evaluate candidates by their average performance, which can mask severe regressions on specific workloads. Even adding adversarial training inputs doesn't eliminate these regressions, and percentile objectives (e.g., optimizing for the 95th percentile) overlook how severe regressions can be beyond the selected threshold.   We present TAILOR, a framework that guides LLM synthesis using the Conditional Value at Risk (CVaR) of the cost difference between each candidate and an existing baseline on the same workload. This paired CVaR measures the average difference over the worst fraction of workloads. We show that a negative paired CVaR guarantees improvements in both average performance and tail optimality gap relative to the unknown optimum, without ever computing that optimum.   We find that TAILOR achieves the best mean--tail trade-
    
[^219]: 交换、复现、读取概率质量：一种用于审计“系统一”决策模型中人口统计学偏差的协议

    Swap, Replicate, Read the Mass: A Protocol for Auditing Demographic Bias in System One Decision Models

    [https://arxiv.org/abs/2611.00496](https://arxiv.org/abs/2611.00496)

    该论文提出了首个针对“系统一”决策模型的人口偏见多模型审计协议，基于三个偏见基准、十二个模型共163.8万次决策，发现这些模型普遍存在刻板印象对齐和顺序依赖性，且其置信度反映的是不稳定性而非偏见本身。

    

    “系统一”决策模型用类型化的、带概率评分的答案取代了文本生成，并正在进入生产环境。这些答案在无人在环的情况下执行，且目前尚无在成熟偏见基准上进行的多模型审计。我们提供了这样的审计：涵盖三个基准（BBQ 的全部十一个类别、CrowS-Pairs 和 SilverSpoon），并将其适配到类型化接口；测试了十二个模型以及一个“无思考”消融配置；共计 1,638,000 个决策。每个顺序敏感的项目都运行两次并交换选项位置；原生类型化模型还会交换标签；十个配置在额外随机种子和批次大小 16 的条件下重复运行。我们发现存在广泛的刻板印象对齐、依赖顺序的答案，以及追踪不稳定性而非偏见的置信度。在 BBQ 中，143 个“模型×类别”评分中有 97 个显著，其中 94 个呈现对齐；而作为商业参考的模型在 99.5% 的模糊项目上选择弃权，从而隐藏了对齐的概率质量。选项交换使 0.0% 至 72.6% 的选择发生翻转；类型化评分保持在……（原文摘要在此处截断）

    arXiv:2611.00496v1 Announce Type: cross  Abstract: System One decision models replace text generation with typed, probability-scored answers and are entering production. Answers execute without a person in the loop, and no multi-model audit exists on established bias benchmarks. We provide that audit: three benchmarks: BBQ (all eleven categories), CrowS-Pairs, and SilverSpoon, adapted to the typed interface, twelve models plus a no-thinking ablation, and 1,638,000 decisions. Every order-sensitive item runs twice with options swapped; native typed models also swap labels; ten configurations repeat under extra seeds and batch 16. We find widespread stereotype alignment, order-dependent answers, and confidence that tracks instability rather than bias. In BBQ, 97 of 143 model-by-category scores are significant and 94 aligned, while the commercial reference abstains on 99.5% of ambiguous items, hiding aligned probability mass. Option swaps flip 0.0% to 72.6% of choices; typed scoring stays 
    
[^220]: 识别人工痕迹：人类与大语言模型中的对话式幽默即兴表演

    Spotting the Artificial: Dialogic Humor Improvisation in Humans and LLMs

    [https://arxiv.org/abs/2611.00492](https://arxiv.org/abs/2611.00492)

    该研究提出结合自动化语言度量与系统功能语言学人工标注的方法，比较多智能体环境中大语言模型生成的幽默对话与人类喜剧即兴表演，发现LLM虽能创造连贯且有创意的场景，却无法再现自发口语互动的关键特征，并表现出过度礼貌与正面评价的倾向。

    

    我们提出了一种方法，用于在具有挑战性的场景——幽默即兴对话——中研究自然语言与合成语言之间的差异。我们的实验将人类创作的喜剧即兴表演与大语言模型（LLM）在我们设计的多智能体环境中生成的对话进行比较。该分析将自动化语言度量与基于系统功能语言学（SFL）的人工标注相结合。结果表明，尽管大语言模型能够创造出连贯且富有创意的场景，但它们往往无法再现自发口语互动的关键特征。此外，大语言模型生成的互动表现出强烈的礼貌倾向和正面评价倾向，这与人类幽默即兴表演中更为广泛的态度范围形成对比。我们的证据既凸显了当前大语言模型在建模对话幽默方面的优势，也揭示了其局限性，并证明了整合语言学分析方法的价值。

    arXiv:2611.00492v1 Announce Type: new  Abstract: We propose a methodology for investigating the differences between natural and synthetic language in a challenging setting: humorous improvisational dialogue. Our experimental setting compares human-produced comedy improvisations with the dialogues generated by LLMs in a multi-agent environment that we designed. The analysis combines automated linguistic measures with manual annotation grounded in Systemic Functional Linguistics (SFL).   The results show that, although LLMs can create coherent and creative scenes, they often fail to reproduce key characteristics of spontaneous spoken interaction. In addition, LLM-generated interactions display a strong tendency toward politeness and positive evaluations, contrasting with the broader range of attitudes found in human humorous improvisation. Our evidence highlights both the strengths and limitations of current LLMs in modeling dialogic humor and demonstrate the value of integrating linguis
    
[^221]: 从小到大：评估语言模型中的医学推理、基准污染与模型规模

    From Small to Large: Assessing Medical Reasoning, Benchmark Contamination, and Model Size in Language Models

    [https://arxiv.org/abs/2611.00460](https://arxiv.org/abs/2611.00460)

    该研究通过评估来自六家厂商的14个不同规模语言模型在四个医学基准共3,076道题目上的表现，考察医学考试成绩是否真实反映临床推理能力，还是受题目格式、基准污染（记忆效应）和模型规模的影响，并检验适合本地部署的小模型能否达到人类水平。

    

    大型语言模型（LLMs）如今在回答医学考试题目方面已达到甚至超过专家水平，但目前尚不清楚这些分数究竟是反映了真正的临床能力，还是受到了提问格式（例如多选题与开放式题目的差异）、模型此前接触过这些基准测试以及题目难度的影响。这些基准测试是公开的，且在网上被广泛使用，这可能导致LLM产生记忆效应——模型在曾见过的题目上可能表现更好。此外，尚不清楚更适合在符合HIPAA合规要求的医院环境中本地部署的较小模型是否已达到人类水平。在本研究中，我们评估了来自六家领先厂商的十四个现有模型（每家厂商各选一个较小模型和一个较大模型，另加两个开放权重模型），在四个医学基准（MedQA-USMLE、MedMCQA、PubMedQA和MedBullets）的3,076道题目上，在标准多选……（摘要在此处截断）

    arXiv:2611.00460v1 Announce Type: cross  Abstract: Large language models (LLMs) now answer medical exam questions at or above expert level, but it is unclear whether such scores reflect clinical competence, or are instead affected by the format of the questions being asked (e.g., multiple-choice versus open-ended), prior exposure to the benchmarks, and the difficulty of the questions. These benchmarks are also public and widely used online, which may be resulting in LLM memorization. Models may perform better on questions that they have seen before. It is also not evident whether smaller models, which are more locally deployable in HIPAA-compliant hospital settings, have achieved human performance. In this study, we evaluate fourteen current models from six leading providers, one smaller and one larger model from each provider with two additional open-weight models, on 3,076 questions from four medical benchmarks (MedQA-USMLE, MedMCQA, PubMedQA, and MedBullets) under standard multiple 
    
[^222]: 从查找到学习：将检索到的知识转化为参数化记忆

    From Lookups to Lessons: Turning Retrieved Knowledge into Parametric Memory

    [https://arxiv.org/abs/2611.00450](https://arxiv.org/abs/2611.00450)

    提出ResMem方法，让LLM作为离线教师阅读检索文本生成教师目标，训练参数化记忆学习加性logit残差，从而在不存储海量token表示的前提下将检索知识有效转化为模型参数化记忆。

    

    检索增强生成（RAG）能够帮助大语言模型（LLM）回答超出其训练数据范围的事实性问题，但会带来额外的存储和推理成本。将先前检索到的信息蒸馏到参数化记忆中，可以在无需在线检索的情况下实现知识复用。然而，训练记忆去模仿检索器只会奖励准确的独立预测，而这种预测在与LLM已有的预测相结合时，并不一定能提升最终输出的质量。为解决这一问题，我们提出了ResMem，让LLM在离线状态下阅读检索到的文本以生成教师目标。与依赖k近邻（kNN）数据存储的现有参数化记忆方法（如MLP Memory）不同，该设计避免了存储海量的token表示。此外，通过将LLM作为离线教师，让其在读取输入的同时获取特权的检索证据，我们训练参数化记忆学习一个加性的logit残差。

    arXiv:2611.00450v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) helps large language models (LLMs) answer factual questions beyond their training data, but it adds storage and inference costs. Distilling previously retrieved information into parametric memories enables reuse without online retrieval. However, training the memory to mimic a retriever rewards accurate independent predictions, which do not necessarily improve the final output quality when combined with the LLM's existing predictions. To address this, we introduce ResMem, in which the LLM reads retrieved text offline to generate teacher targets. This design avoids storing massive token representations, unlike existing parametric memory approaches such as MLP Memory, which relies on k-nearest neighbor (kNN) datastores. Furthermore, by using the LLM as an offline teacher that reads privileged retrieved evidence alongside the input, we train a parametric memory to learn an additive logit residual. This r
    
[^223]: 分位数而非点：用于序数感知的LLM-as-a-Judge校准的流式分位数匹配

    Quantiles, Not Points: Streaming Quantile Matching for Ordinal-Aware LLM-as-a-Judge Calibration

    [https://arxiv.org/abs/2611.00447](https://arxiv.org/abs/2611.00447)

    提出流式分位数匹配（SQM）校准方法，通过最优传输将LLM评分分布映射到人类评分分布，尊重评分的序数性并实现了最佳的分布保真度。

    

    在任何给定的评估任务中，LLM-as-a-judge的评分往往与人类评分存在分布偏移，而事后校准（即在小型带标签的锚点集上将LLM评分映射为人类评分）已成为纠正这种偏移的流行方法。在本文中，我们从两个立场出发来推导校准方法：LLaJ评分是从分布中抽取的序数评级，而非孤立的点。我们提出了流式分位数匹配（SQM），这是对一种经典方法的新应用，此前尚未在LLaJ校准中进行过基准测试，该方法通过计算LLM评分分布与人类评分分布之间的最优传输映射来建立在上述两个立场之上。我们开发了一个评估框架，并在其中证明：与逐点校准不同，SQM尊重人类锚点集的序数性，并在各模型和数据上实现了最佳的分布保真度和总体最高排名。我们还表明，总体而言，事后校准缩小了

    arXiv:2611.00447v1 Announce Type: cross  Abstract: LLM-as-a-judge scores are often distributionally shifted from human ratings on any given evaluation task, and post-hoc calibration (mapping LLM to human scores in a small labeled anchor set) has become a popular method to correct for this shift. In this paper we derive the calibration from two positions: that LLMaJ scores are ordinal ratings drawn from a distribution rather than isolated points. We present streaming quantile matching (SQM), a version of a classical method not previously bench marked for LLMaJ calibration, which is built on the two positions by computing the optimal-transport map between the LLM and human score distributions. We develop an evaluation framework and within it we show that unlike point-wise calibrations, SQM respects the ordinality of the human anchor set and achieves the best distributional fidelity and the top overall rank across models and data. We also show that in general, post-hoc calibration narrows
    
[^224]: 保真度随表格规模而非轮次数崩溃：部署型工具使用智能体中的交付架构

    Fidelity Collapses with Table Size, Not Turn Count: Delivery Architecture in a Deployed Tool-Using Agent

    [https://arxiv.org/abs/2611.00442](https://arxiv.org/abs/2611.00442)

    在生产级工具使用智能体中，回答正确性的崩溃取决于交付给模型的表格规模而非对话轮次数：将记录保留在会话状态并通过确定性过滤器与类型化旁路通道交付的 BYPASS 架构（92.3% 正确）显著优于将全部记录注入上下文的 SYNTHESIS 架构（45.7% 正确）。

    

    当多轮智能体将工具生成的结构化记录送入模型上下文并在每一轮重新输出时，每一轮都可能损坏工具原本正确计算的数据。我们在一个生产级奖励机票预订智能体中对此进行了测量：两种架构使用相同的冻结载荷、血统匹配的提示词与采样，仅在交付架构上存在差异。SYNTHESIS 架构将完整记录集送入规划器的上下文，并在模型内部直接回答后续问题；BYPASS 架构（即部署系统）则将记录保留在会话状态中，将后续问题转化为确定性过滤器的参数，并通过类型化旁路通道交付数据行。在 24 组对话（每种条件各 339 个评分轮次）中，BYPASS 有 92.3% 的轮次完全正确，而 SYNTHESIS 仅为 45.7%（配对差距 +46.6 个百分点 [+35.7, +57.1]）。通过按捕获日期改变交付表格的规模，SYNTHESIS 从 10 行及以下时的接近完美水平，崩溃至超过约 74 行时的 8% 或更低。在一个内容受控的前缀阶梯实验中……

    arXiv:2611.00442v1 Announce Type: new  Abstract: When a multi-turn agent delivers a tool's structured records into the model's context and re-emits them each turn, every turn can corrupt data the tool computed correctly. We measure this in a production award-travel agent: two architectures over identical frozen payloads, lineage-matched prompts and sampling, differing only in delivery architecture. SYNTHESIS feeds full record sets into the planner's context and answers follow-ups in-head; BYPASS, the deployed system, keeps records in session state, translates follow-ups into arguments for a deterministic filter, and delivers rows over a typed side channel. Across 24 dialogues (339 scored turns per condition), BYPASS scores 92.3% of turns fully correct, SYNTHESIS 45.7% (paired gap +46.6pp [+35.7, +57.1]). Varying delivered-table size by capture date, SYNTHESIS collapses from near-perfect at 10 rows or fewer to 8% or less beyond about 74 rows. On a content-controlled prefix ladder the co
    
[^225]: 多模态模型对等价光学光谱的判断是否一致？

    Do Multimodal Models Decide Consistently on Equivalent Optical Spectra?

    [https://arxiv.org/abs/2611.00428](https://arxiv.org/abs/2611.00428)

    该研究通过将同一光学光谱以多种等价表征（不同单位的表格与图形）呈现来测试多模态模型的表征不变性与弃权能力，发现仅改变表征方式就可使模型准确率波动高达41个百分点，表明模型对等价光谱的判断并不一致。

    

    光谱既可以以纳米为单位以表格形式提供给模型，也可以以光子能量为单位以图形形式呈现，而模型的判断不应取决于它看到的是哪一种表示。当回答所需的数据缺失时，模型应当如实说明。我们在硅超表面（MOCLIP）和金纳米颗粒的实测光学光谱上测试了表征不变性与弃权能力，标签由数据直接计算得出。每条光谱以四种表格（nm、μm、科学计数法、eV）和两种图形（nm、eV）的形式呈现，部分光谱则删除了查询范围内或范围外的数据。我们评估了决策模型 d3-nano 和 Jev 1.13、三种 GPT-6 Luna 配置以及 Qwen3.5-2B 基础模型。结果显示，仅改变表征方式就会使同一模型内的准确率波动高达 41 个百分点，且主要出现在光子能量的表格和图形上。在相同证据下，用光子能量而非波长提问，会使部分“模型-视图”组合的性能提升 16-50 个百分点，而对另一些组合则造成损害（摘要在此处截断）。

    arXiv:2611.00428v1 Announce Type: cross  Abstract: A spectrum can be given to a model as a table in nanometers or as a plot in photon energy, and the decision should not depend on which one it sees. When the data needed for an answer are missing, the model should say so. We test representation invariance and abstention on measured optical spectra of silicon metasurfaces (MOCLIP) and gold nanoparticles, with labels computed from the data. Each spectrum is shown as four tables (nm, $\mu$m, scientific notation, eV) and two plots (nm, eV), and partial spectra remove data inside or outside the queried range. We evaluate the decision models d3-nano and Jev 1.13, three GPT-6 Luna configurations and the Qwen3.5-2B base model. Changing only the representation moves accuracy by up to 41 percentage points within one model, mostly on photon-energy tables and plots. With the same evidence, asking in photon energy instead of wavelength helps some model-view pairs by 16-50 points and hurts others by 
    
[^226]: 电路神谕：通过自然语言查询自动化归因图分析

    Circuit Oracle: Automating Attribution Graph Analysis via Natural-Language Queries

    [https://arxiv.org/abs/2611.00420](https://arxiv.org/abs/2611.00420)

    本文提出Circuit Oracle，通过多智能体框架和任务技能让现成大语言模型能够自动读取归因图并用自然语言回答关于目标模型的安全相关问题，将原本需要专家数小时人工分析的机制可解释性工作自动化。

    

    归因图是机制可解释性领域的一种新兴工具，它使用转码器将语言模型的计算分解为由因果边连接的稀疏可解释特征。然而，要把一张图转化为与安全相关的洞见，需要专家进行数小时的人工分析。我们提出了Circuit Oracle，这是一个现成的大语言模型，配备了多智能体框架和针对具体任务编写的技能，它通过工具调用读取归因图，从而回答关于目标模型的自然语言问题。我们在三个与安全相关的代理任务上对其进行了评估：检测探测电路中的虚假特征、从经过禁忌微调的模型中引出隐藏知识，以及通过特征级干预实现抑制式越狱。除了任务得分之外，该神谕还会生成剪枝后的子图和结合具体上下文的特征解释，我们对这些结果进行了定性评估。结果是好坏参半的。在虚假特征检测方面，Circuit Oracle达到了78.2……（原文摘要在此处截断）

    arXiv:2611.00420v1 Announce Type: new  Abstract: Attribution graphs, an emerging tool in mechanistic interpretability, use transcoders to decompose language model computations into sparse interpretable features connected by causal edges. However, turning a graph into a safety-relevant insight requires hours of manual analysis by experts. We introduce Circuit Oracle, an off-the-shelf LLM equipped with a multi-agent harness and task-specific authored skills, which answers natural-language questions about a target model by reading the attribution graph through tool calls. We evaluate it on three safety-relevant proxy tasks: detecting spurious features in probe circuits, eliciting hidden knowledge from taboo-finetuned models, and suppression jailbreaking via feature-level interventions. Beyond task scores, the oracle produces pruned subgraphs and context-specific feature explanations that we assess qualitatively. Results are mixed. On spurious-feature detection, Circuit Oracle reaches 78.2
    
[^227]: 人类与神经网络中的几何式推断与代数式推断

    Geometric and algebraic inference in humans and neural networks

    [https://arxiv.org/abs/2611.00387](https://arxiv.org/abs/2611.00387)

    本文将人类推断策略形式化为“几何式推断”（通过逐步塑造表征几何结构使推断代价低廉）和“代数式推断”（灵活推理上下文中的新颖关系但需付出额外计算成本），并以传递性推断为测试平台，刻画了人类和神经网络中两种策略各自出现的条件。

    

    人类拥有一种非凡的能力，即能够从有限的数据中做出推断。有些推断是基于过往经验凭直觉完成的，而另一些推断则需要基于上下文中提供的新颖信息进行在线推理。在本文中，我们将这两种策略形式化为“几何式推断”和“代数式推断”，并研究它们在人类和神经网络中的计算权衡。几何式策略通过逐步塑造表征几何结构来反映潜在的关关系结构，从而使后续推断的代价变得低廉；而代数式策略能够灵活地关注并推理上下文中给出的新颖关系，但需要在推断时付出额外的计算成本。我们在传递性推断这一在多个物种中被广泛研究的领域中，识别出了这两种策略截然相反的特征，并刻画了各自出现所需的条件。我们的框架预测：当关系较为熟悉时，几何式策略应占主导地位……（原文在此处截断）

    arXiv:2611.00387v1 Announce Type: cross  Abstract: Humans possess a remarkable ability to draw inferences from limited data. While some inferences are made intuitively from past experience, others require online reasoning based on novel information given in context. Here, we formalize these strategies, which we term geometric and algebraic inference, and study their computational tradeoffs in humans and neural networks. The geometric strategy gradually shapes representational geometry to reflect underlying relational structure, making subsequent inference cheap. The algebraic strategy flexibly attends to and reasons over novel relations given in context, but costs additional computation at inference time. We identify opposing signatures of these two strategies in transitive inference, a domain widely studied across species, and characterize the conditions under which each is observed. Our framework predicts that the geometric strategy should be dominant when relations are familiar, whe
    
[^228]: LF-VideoQA：基于长文本心智理论问题评估长视频理解能力

    LF-VideoQA: Evaluating Long-Video Understanding with Long-Form Theory-of-Mind Questions

    [https://arxiv.org/abs/2611.00383](https://arxiv.org/abs/2611.00383)

    LF-VideoQA是一个基于心智理论问题的长视频开放式问答数据集，包含94部电影中的6,300多个需要综合跨场景（平均23分钟）证据进行推理的问题，并配有专用评分标准，从而超越简单检索范式、更真实地评估模型的长视频理解能力。

    

    理解一个长视频最困难的部分不在于精确定位某个单一时刻，而在于综合多个时刻的信息，例如回答一个需要追踪数小时内容中的发展脉络、并阐明各部分如何相互关联的问题。迄今为止，大多数长视频问答数据集仅要求简短答案或多项选择题选择，将长视频理解任务过度简化为检索任务。我们提出了LF-VideoQA，这是一个基于心智理论问题的、面向长叙事视频的长文本开放式问答数据集：一个角色的信念取决于其看到或被告知的一切，因此每个问题都需要整合并推理分布在整段视频中的证据。该数据集包含94部近期电影中的6,300多个问题；每个问题都需要对平均跨越23分钟的场景进行推理，并附有针对该问题的专用评分标准，用于召回率和精确度打分。该数据集的构建完全……

    arXiv:2611.00383v1 Announce Type: cross  Abstract: The hardest part of understanding a long video is not pinpointing any one moment but synthesizing many, e.g., answering a question that requires tracking what develops across hours of content and articulating how the pieces fit together. To date, most long-video QA datasets require only a short answer or a multiple-choice selection, oversimplifying the long-video understanding task to retrieval. We introduce LF-VideoQA, a dataset for long-form, open-ended question answering over long narrative videos built on theory-of-mind questions: a character's belief depends on everything they have seen or been told, so each question demands assembling and reasoning from evidence distributed across the video. The dataset contains more than 6,300 questions on 94 recent films; each requires reasoning over scenes spanning an average of 23 minutes and comes with a question-specific rubric for recall and precision scoring. The dataset is created entire
    
[^229]: 通过外推“收缩-修复”前瞻实现机器遗忘

    Unlearning by Extrapolating a Shrink-and-Heal Lookahead

    [https://arxiv.org/abs/2611.00374](https://arxiv.org/abs/2611.00374)

    SHIFT方法通过将微调模型向微调前权重收缩并重新学习保留集来估计通往重训模型的方向，再将该前瞻过程的逐token变化外推为蒸馏目标并迭代优化，实现了最先进的机器遗忘效果。

    

    机器遗忘旨在使大语言模型表现得像在没有遗忘集的情况下重新训练过的模型。然而，主流的似然抑制目标缺乏明确的遗忘目标：它们持续追求更低的遗忘集似然，使得遗忘效果对优化预算非常敏感。我们通过在不实际训练“重训模型”的情况下估计其方向来解决这一限制。我们的核心思想是将微调后的模型向其微调前的权重收缩，并短暂地重新学习保留集，从而去除与遗忘相关的特定贡献；在线性化条件下，这种“收缩-修复”前瞻会将模型移动到重训模型（其唯一不动点）方向上固定比例的距离。我们的方法SHIFT将前瞻的逐token变化外推为蒸馏目标，并进行迭代细化。在统一的OpenUnlearning协议下与十一个基线方法进行对比评估，SHIFT取得了最高的综合得分。

    arXiv:2611.00374v1 Announce Type: cross  Abstract: Unlearning aims to make a large language model behave like one retrained without its forget set. Prevailing likelihood-suppression objectives, however, lack an explicit forgetting target: they continually reward lower forget-set likelihood, making forgetting sensitive to the optimization budget. We address this limitation by estimating the direction toward the oracle without training it. Our key idea is to shrink the fine-tuned model toward its pre-fine-tuning weights and briefly re-learn the retain set, thereby removing the forget-specific contribution; in the linearized regime, this shrink-and-heal lookahead moves the model a fixed fraction of the way to the retrained model, its unique fixed point. Our method, SHIFT, extrapolates the lookahead's per-token changes into distillation targets and iteratively refines them. Evaluated against eleven baselines under a unified OpenUnlearning protocol, SHIFT achieves the highest aggregate scor
    
[^230]: 人性化但仍真实？对AI文本“人性化器”中的事实与检测的审计

    Humanized but Still True? Auditing Facts and Detection in AI-Text Humanizers

    [https://arxiv.org/abs/2611.00359](https://arxiv.org/abs/2611.00359)

    该论文首次对AI文本“人性化器”的事实保真度进行系统审计，发现全部11款商业工具都会在相当比例的文档中更改或丢失数字、日期、名称等事实细节，揭示了规避AI检测与保持文本真实性之间的矛盾。

    

    AI“人性化器”通过重写机器生成的文本，使AI检测器将其判定为人类撰写。公开排行榜对规避检测能力和嵌入相似度进行评分，但并未评估可核查的细节是否得以保留。我们检验了数字、日期、人名与组织名称以及引用片段在每次重写后是否得以保留，所使用的基于规则的检查器由两个AI标注器验证（精确率0.84-1.00，召回率0.93-0.95；无人工标注）。在2026年10月周期的公开人性化器排行榜中，全部11款商业工具都在35-65%包含此类信息单元的文档中更改或丢弃了至少一个单元。大多数更改发生在引用片段上；仅统计数字和名称，11款工具中仍有10款在20%以上的文档中更改了至少一个。2026年9月周期的情况类似（19-50%）。一款开源的12B参数人性化器在事实密集的美国联邦机构文本中保留了至少97%的数字和名称，但重写了67-78%的引用片段，这与它自身的……（摘要原文在此处截断）

    arXiv:2611.00359v1 Announce Type: new  Abstract: AI "humanizers" rewrite machine-generated text so that detectors classify it as human-written. Public leaderboards score evasion and embedding similarity, but not whether checkable details survive. We check whether numbers, dates, person and organization names, and quoted spans survive each rewrite, using a rule-based checker validated by two AI annotators (precision 0.84-1.00, recall 0.93-0.95; no human annotation). In the October 2026 cycle of a public humanizer leaderboard, all 11 commercial tools changed or dropped at least one of these units in 35-65% of the documents that contained one. Most changes were to quoted spans; counting numbers and names alone, 10 of the 11 tools still altered at least one in 20% or more of the documents. The September 2026 cycle was similar (19-50%). An open 12B humanizer kept at least 97% of numbers and names in fact-dense U.S. federal agency text, but reworded 67-78% of the quoted spans, against its ow
    
[^231]: 跨提示的实体替换响应复用

    Cross-Prompt Reuse of Entity-Replacement Responses

    [https://arxiv.org/abs/2611.00352](https://arxiv.org/abs/2611.00352)

    该论文发现在固定模型权重的前提下，将源问题引发的隐状态变化注入目标问题的中间层，可以实现实体替换响应在事实性问题之间的跨提示复用，在Qwen、Llama和Gemma等小型语言模型上最多消除约一半的无变化误差。

    

    在模型权重固定的情况下，我们测量了实体替换响应在英文事实性问题之间的迁移程度，而不仅仅是答案是否改变。我们将来自两个源问题的隐状态变化添加到关于同一实体的目标问题上，并在给定已知答案前缀的条件下，将两种关系的已发布答案之间预测的对数概率偏移与实际测量到的偏移进行比较。隐式复用在目标模型剩余层之前50%的深度处注入对齐的变化；直接复用则叠加源问题的输出分数变化。在使用其他关系作为源时，隐式复用在0.5-1B参数的Qwen和Llama模型上分别消除了45%和49%的无变化误差。Gemma-2-2B在纯文本输入且不使用其默认序列开始（BOS）标记的情况下消除了22%的误差，并通过了三个比较中的一个。后续恢复BOS标记的协议取得了58%-68%的结果，在三个模型队列上通过了所有比较。更大的模型检查点、聊天格式和新实体的实验显示了复现性以及格式或队列依赖性。

    arXiv:2611.00352v1 Announce Type: new  Abstract: With model weights fixed, we measure how much of an entity-replacement response transfers between English factual questions, beyond whether answers change. We add the hidden-state change from two source questions to a target question about the same entity, comparing the predicted log-probability shift between two relations' published answers, given known answer prefixes, with the measured one. Hidden reuse injects the aligned change at 50% depth before the target's remaining layers; direct reuse adds source output-score changes. With other-relation sources, hidden reuse removes 45% and 49% of no-change error on 0.5-1B Qwen and Llama. Gemma-2-2B removes 22% and passes one of three comparisons in plain text without its default beginning-of-sequence (BOS) token. A later protocol restoring it yields 58-68%, passing every comparison on three cohorts. Larger checkpoints, chat format and new entities show replication and format- or cohort-depen
    
[^232]: 酷儿化审计：针对酷儿群体AI伤害的社区参与式审计

    Queering the Audits: Community-Based Auditing of AI Harms to Queer Communities

    [https://arxiv.org/abs/2611.00345](https://arxiv.org/abs/2611.00345)

    该研究通过让16名酷儿社区成员在参与式工作坊中运用4C伤害分类法审计四个AI案例，揭示了技术审计无法发现的社区伤害，并将伤害根源追溯到问题定义、数据收集及从弱势群体攫取价值的合同结构。

    

    AI系统将多数群体的默认设置嵌入到训练数据、评估指标和类别定义之中，对酷儿群体造成了有据可查的伤害，包括抹除、错误分类和歧视。标准的技术审计通常依赖聚合性指标，无法检测到只有通过受影响社区的生活体验才能显现的伤害。我们在EurIPS 2025举办了一场参与式审计工作坊，16名酷儿社区成员运用贯穿AI全生命周期的4C伤害分类法（内容Content、行为Conduct、接触Contact、合同Contract）对四个案例研究进行了审计。参与者借助结构化工作表和全体综合讨论，对伤害进行分类并将其追溯至开发流程中的源头。在所有四个案例中，参与者都将伤害追溯到问题定义和数据收集阶段，并识别出从弱势群体中攫取价值却几乎不提供补救途径的合同结构。

    arXiv:2611.00345v1 Announce Type: new  Abstract: AI systems embed majority-group defaults into training data, evaluation metrics, and category definitions, producing documented harms for queer communities including erasure, misclassification, and discrimination. Standard technical audits often rely on aggregate measures and cannot detect harms that be come visible only through the lived experience of affected communities. We conducted a participatory auditing workshop at EurIPS 2025 where 16 queer community members audited four case studies using the 4Cs harm taxonomy (Content, Conduct, Contact, Contract) applied across the AI lifecycle. Participants used structured worksheets and plenary synthesis to classify harms and trace them to their origins in the development pipeline. Across all four cases, participants traced harms to problem definition and data collection, and they identified contractual structures that extract value from vulnerable populations while providing minimal recours
    
[^233]: SPARK：通过物理重观测实现自我纠正的具身记忆

    SPARK: Self-Correcting Embodied Memory via Physical Re-Observation

    [https://arxiv.org/abs/2611.00344](https://arxiv.org/abs/2611.00344)

    提出了SPARK框架，通过语义批评器标记知识库错误、针对性规划器生成重访计划、选择性合并更新信息，实现具身RAG记忆的自我纠错。

    

    检索增强生成（RAG）系统作为可被语言查询的环境记忆，在具身场景中的应用日益增多。然而，现有方法通常假设存在可靠的单次任务知识库（KB），或者跨任务累积观测而不纠正先前的错误。此类错误可能以错误的语义标签、不准确的对象位置和不一致的空间关系等形式持续存在。为解决这一问题，我们提出了SPARK，一个用于具身RAG的纠正性多任务框架。在第一次任务结束后，语义批评器（semantic critic）标记知识库中存在问题的区域，针对性规划器生成重访计划，选择性合并模块则根据新的观测结果对知识库信息进行添加、删除或替换。我们在四张户外AirSim地图上，在两种退化视角条件下对SPARK进行评估，使用由感知构建的知识库、仅用于评估的可见性受限参考基准，以及固定的158个具身问答数据集。

    arXiv:2611.00344v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) systems have seen growing adoption in embodied settings as language-queryable memories of the environment. However, existing approaches often assume a reliable single-episode knowledge base (KB) or accumulate observations across episodes without correcting prior errors. Such errors can persist through incorrect semantic labels, inaccurate object positions, and inconsistent spatial relationships. To address this, we present SPARK, a corrective multi-episode framework for embodied RAG. After the first episode, a semantic critic flags regions of the KB, a targeted planner generates revisits, and a selective merge adds, removes, or replaces KB information based on the new observations. We evaluate SPARK on four outdoor AirSim maps under two degraded viewpoint conditions, using perception-built KBs, a visibility-constrained oracle used only for evaluation, and a fixed set of 158 embodied question answeri
    
[^234]: 一个波形，两个故事：音频-语言模型中语音-声学冲突的基准测试

    One Waveform, Two Stories: Benchmarking Speech-Acoustic Conflict in Audio-Language Models

    [https://arxiv.org/abs/2611.00325](https://arxiv.org/abs/2611.00325)

    该论文提出SAC-Bench基准测试，揭示了音频-语言模型在面对同一录音中语音陈述与声学证据相互矛盾时的脆弱性——事件识别准确率下降24.8至46.5个百分点，暴露出模型缺乏处理音频内语音-声学冲突的能力。

    

    音频-语言模型（ALMs）通常在其对语音和环境声音的理解能力上进行评估。然而，尚未被充分探索的问题是：当语音陈述与同一录音中的声学证据相矛盾时，这些模型会如何反应。我们提出了SAC-Bench，这是一个针对音频内语音-声学冲突的基准测试，包含来自188个声音事件类别的616,031个独特片段，涵盖七个语义、时间和空间任务。该基准将环境录音与兼容或矛盾的合成语音陈述配对，并在适用情况下包含纯音频对照。在五个开源权重的ALMs上对语义和时间任务进行评估后发现，纯音频的高准确率往往与冲突陈述下的大幅性能下降并存；事件身份的准确率在冲突条件下比兼容条件低24.8至46.5个百分点。即使在语音跟在完整事件序列之后播放的情况下，时间顺序任务的失败仍然持续存在，因此同时的语音-事件掩码无法（解决该问题）……

    arXiv:2611.00325v1 Announce Type: cross  Abstract: Audio-language models (ALMs) are evaluated on their understanding of both speech and environmental sounds. What remains underexplored is how they respond when spoken claims contradict acoustic evidence within the same recording. We introduce SAC-Bench, a benchmark of intra-audio speech--acoustic conflict comprising 616,031 unique clips from 188 sound-event classes across seven semantic, temporal, and spatial tasks. It pairs environmental recordings with compatible or contradictory synthesized spoken claims and includes audio-only controls where applicable. Across five open-weight ALMs evaluated on semantic and temporal tasks, high audio-only accuracy often coexists with large losses under conflicting claims; Event Identity accuracy is 24.8--46.5 percentage points lower in conflict than compatible conditions. Temporal Order failures persist even when speech follows the complete event sequence, so simultaneous speech--event masking canno
    
[^235]: GAVAGAI：通过平衡最优传输实现婴儿尺度视觉-语言模型的指称对齐

    GAVAGAI: Referential Alignment for Infant-Scale Vision-Language Models via Balanced Optimal Transport

    [https://arxiv.org/abs/2611.00319](https://arxiv.org/abs/2611.00319)

    该论文提出GAVAGAI方法，将词语-区域对齐建模为带有NOTHING空列和区域容量约束的熵正则化最优传输分配问题，从而以约八次Sinkhorn迭代的低成本为婴儿尺度的视觉-语言模型提供直接的指称对齐监督。

    

    儿童从大约一亿词的输入中学会哪些词指称哪些事物。在相当规模数据上训练的视觉-语言模型将词与所指对象绑定的可靠性要低得多，我们认为这种差距部分是结构性的：通常的下一词元图像描述目标中不存在代表“这个词指称那个区域”的变量，因此训练过程对词语落地没有施加任何直接压力。我们通过把每个场景视为一个在听到的实义词与看到的区域之间的小型分配问题来引入这样一个变量，并以熵正则化最优传输在两个约束下求解。第一，一个词可以被分配到一个特殊的NOTHING列而非任何区域，这一点很重要，因为照护者话语中的大多数实义词并不命名任何可见之物。第二，任何区域所接收的分配质量不得超过一个有界份额，这一容量约束编码了互斥性。对齐步骤大约只需八次Sinkhorn迭代（摘要在此处被截断）。

    arXiv:2611.00319v1 Announce Type: new  Abstract: A child learns which words name which things from roughly 100 million words of input. Vision-language models trained at a comparable scale bind words to referents far less reliably, and we argue that part of the gap is structural: the usual next-token captioning objective has no variable standing for "this word refers to that region," so training applies no direct pressure to ground words. We add such a variable by treating each scene as a small assignment problem between the content words heard and the regions seen, solved as entropic optimal transport under two constraints. First, a word may be assigned to a special NOTHING column instead of any region, which matters because most content words in caregiver speech do not name anything visible. Second, no region may receive more than a bounded share of the assignment mass, a capacity constraint that encodes mutual exclusivity. The alignment step costs about eight Sinkhorn iterations, sma
    
[^236]: 共享代码库中的去中心化多智能体协调与通信

    Decentralized Multi-Agent Coordination and Communication in Shared Codebases

    [https://arxiv.org/abs/2611.00314](https://arxiv.org/abs/2611.00314)

    该论文提出了 ParallACT 基准，首次系统研究无编排器的多个编程智能体在共享代码库中面对互补与矛盾任务时的协调与通信问题，揭示独立开发实现的合并冲突及其影响因素。

    

    在共享代码仓库中工作的编程智能体可能会收到来自人类开发者的相互独立、动机各异的请求，从而形成一个必然的多智能体场景。当这种情况发生时，各个智能体的单独实现可能各自可用，但在合并时却会产生冲突。为了研究这一场景，我们提出了 ParallACT，一个无编排器的多智能体编程任务基准。ParallACT 既包含互补任务，也包含矛盾任务。在互补任务中，每个智能体的任务单独来看都有用，且各实现必须共享一个接口才能被有效地结合起来。在矛盾任务对中，两个智能体的任务需要以互不兼容的方式编辑相同的代码，天真地合并独立的解决方案无法产生有用的实现。我们将获得完整合并任务集的单智能体“理想参照”与多种多智能体设置进行比较，这些设置在执行顺序、可见性、对其他智能体的感知以及通信访问等方面有所不同。

    arXiv:2611.00314v1 Announce Type: cross  Abstract: Coding agents working in shared repositories may receive separate, independently-motivated requests from human developers, creating a setting that is multi-agent by necessity. When this scenario happens, their individual implementations may work separately but conflict when combined. To study this setting, we introduce ParallACT, a benchmark of orchestrator-free multi-agent coding tasks. ParallACT contains both complementary and contradictory tasks. In complementary tasks, each agent's task is separately useful and implementations must share an interface to be productively combined. In contradictory pairs, both agents' tasks require editing the same code in incompatible ways, and naively combining independent solutions fails to produce a useful implementation. We compare a single-agent oracle given the combined set of tasks against multi-agent settings that vary execution order, visibility, awareness of other agents, and access to a co
    
[^237]: 大语言模型谈判中的跨语言人设效应

    Cross-Lingual Persona Effects in LLM Negotiation

    [https://arxiv.org/abs/2611.00300](https://arxiv.org/abs/2611.00300)

    该研究提出“跨语言社会策略保真度”这一新概念，通过实验发现LLM谈判中提示词定义的人格行为在不同语言间不一致——韩语谈判达成协议率（77.3%）显著高于英语（52.5%），且该差异仅出现在中立和固执型买家中，表明跨语言部署时社会策略的可靠性存在隐患。

    

    告诉一个LLM谈判者要“固执己见”，它应该在价格上坚持立场，但这种指令在每种语言中都能保持相同的行为吗？为了验证这一点，我们使用相同的买家和卖家Gemini 3.6 Flash实例，在英语和韩语中对相同的12个CraigslistBargain场景进行谈判。这项试点规模的2×2×3实验设计交叉了交互语言、卖家紧迫感（无或强）和买家人格（合作型、中立型或固执型），共产生432次谈判尝试。我们将这一特性称为跨语言社会策略保真度：即提示词定义的社会策略无论交互语言如何，都能产生相同的行为效应。但实际情况往往并非如此。韩语谈判达成协议的频率远高于英语（77.3% vs. 52.5%，差距24.8个百分点，Holm校正p = .0039），但这仅适用于中立型和固执型买家；合作型买家在两种语言中都已接近上限。买家的初始报价几乎没有差异

    arXiv:2611.00300v1 Announce Type: new  Abstract: Tell an LLM negotiator to be "headstrong" and it should hold firm on price, but does that instruction retain the same behavior in every language? To test this, we use identical buyer and seller instances of Gemini 3.6 Flash to negotiate the same 12 CraigslistBargain scenarios in English and Korean. The pilot-scale 2 x 2 x 3 design crosses interaction language, seller urgency (none or strong), and buyer persona (cooperative, neutral, or headstrong), yielding 432 attempted negotiations. We call this property cross-lingual social-policy fidelity: whether a prompt-defined social policy produces the same behavioral effect regardless of interaction language. It often does not. Korean negotiations reached agreement far more often than English (77.3% vs. 52.5%, a 24.8-point gap, Holm-adjusted p = .0039), but only for neutral and headstrong buyers; cooperative buyers were already near-ceiling in both languages. Buyers' opening offers barely diffe
    
[^238]: 教学、抽象、应用：一个面向协作机器人的对话驱动任务学习框架

    Teach, Abstract, Apply: A Framework for Dialogue-Driven Task Learning with Collaborative Robots

    [https://arxiv.org/abs/2611.00297](https://arxiv.org/abs/2611.00297)

    本文提出TAA框架，使操作员能够通过自然语言向协作机器人教授新工作流程，将其抽象为可复用技能，并在不同场景中灵活应用。

    

    协作机器人旨在比传统工业机器人更安全、更具交互性，与人类共享同一工作空间，并协助人类完成一系列车间任务，如码垛、联合装配和零件检测。这些任务中的许多涉及重复性工作流程，而这些流程可能随着生产环境和任务规格的变化而演变。这些需求催生了对直观交互界面的需求，使工人能够在车间现场修改工作流程，并将这些例程泛化以供未来使用。本工作提出了“教学-抽象-应用”（Teach-Abstract-Apply, TAA）框架，该框架使人类操作员能够： 通过自然语言向协作机器人教授新的工作流程； 将所教授的工作流程抽象为技能以供后续使用； 在不同的环境中应用这些技能。作为案例研究，我们将TAA框架应用于一项协作式结构搭建任务，并使用开源权重模型（Qwen3-Coder……

    arXiv:2611.00297v1 Announce Type: cross  Abstract: Collaborative robots (cobots) are designed to be safer and more interactive than traditional industrial robots, share the same workspace as humans, and assist them in a range of shop-floor tasks such as palletization, joint assembly, and part inspection. Many of these tasks involve repetitive workflows that may evolve with changes in production environments and task specifications. These requirements motivate the need for intuitive interfaces that allow workers to modify workflows on the shop floor and generalise these routines for future use. This work introduces the Teach-Abstract-Apply (TAA) framework, which enables human operators to (a) teach new workflows to cobots in natural language, (b) abstract the taught workflows into skills for later use, and (c) apply these skills in different settings. As a case study, we apply the TAA framework to a collaborative structure-building task and evaluate it with both open-weight (Qwen3-Coder
    
[^239]: MEMENTO：面向长上下文处理智能体的上下文管理学习

    MEMENTO: Learning Context Management for Long Context Processing Agents

    [https://arxiv.org/abs/2611.00244](https://arxiv.org/abs/2611.00244)

    提出MEMENTO智能体，通过“事后笔记标注的合成轨迹微调+结合答案正确性、笔记质量与置信度校准的多目标奖励强化学习”两阶段训练，让Qwen3-8B学会有效的上下文压缩记笔记行为，并能泛化到上下文折叠和置信度回溯等测试时策略。

    

    在长输入上运行的LLM智能体必须执行上下文管理，即决定在固定大小的上下文窗口中保留哪些内容。一种常见方法是压缩，即模型用自生成的笔记替换其历史记录。然而，现成的智能体模型，包括GPT-5.5和GLM-5.3，往往写出无效的笔记，而仅靠强化学习（RL）也很难发现有用的记笔记行为。我们提出了MEMENTO，这是一个基于Qwen3-8B构建、经训练学会通过压缩来管理上下文的长上下文智能体。我们首先通过在标注有事后生成的笔记的合成轨迹上进行微调来引导记笔记行为，然后使用结合答案正确性、笔记质量启发式指标和置信度校准的奖励进行强化学习。尽管MEMENTO仅被训练用于压缩到全新窗口中，但它在测试时能够扩展到上下文折叠和基于置信度的回溯等策略。在书籍级别的问答任务上……

    arXiv:2611.00244v1 Announce Type: new  Abstract: LLM agents that operate over long inputs must perform context management: deciding what to retain in a fixed-size context window. A common approach is compaction, where the model replaces its history with a self-generated note. Yet off-the-shelf agentic models, including GPT-5.5 and GLM-5.3, often write ineffective notes, and reinforcement learning (RL) alone rarely discovers useful note-taking behavior. We present MEMENTO, a long-context agent built on Qwen3-8B and trained to manage its context through compaction. We first bootstrap note-taking by fine-tuning on synthetic trajectories annotated with notes generated in hindsight, then reinforce it with RL using a reward that combines answer correctness, note-quality heuristics, and confidence calibration. Although trained only to compact into a fresh window, MEMENTO extends at test time to strategies such as context folding and confidence-based backtracking. On book-level question answer
    
[^240]: 从碎片化到融合：全模态嵌入的综述与路线图

    From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings

    [https://arxiv.org/abs/2611.00172](https://arxiv.org/abs/2611.00172)

    该综述指出跨模态嵌入研究正因对比学习、Transformer架构和粒度转变而走向统一，提出了“覆盖性—可比性—组合性”三层递进能力框架，并倡导以联合嵌入作为实现全模态统一的最终路径。

    

    嵌入模型支撑着几乎所有现代AI系统，从搜索引擎到推荐系统。尽管共享相同的数学基础，但嵌入模型的研究仍因模态不同而碎片化：各个社区的发展在很大程度上相互隔离，很少交流见解。我们认为这种碎片化掩盖了一种更深层次的趋同。在对比学习、Transformer架构和粒度转变的驱动下，跨模态的嵌入研究正汇聚于一套共同的原则，尽管模态特定的挑战仍未得到充分探索。为了在不混淆这种趋同的前提下对其进行推理，我们提出了一个由三种递进式能力构成的框架：覆盖性、可比性和组合性。通向统一的三条路径依次满足这些能力：转换拓宽了覆盖性，对齐实现了可比性，而联合嵌入——我们倡导的终极目标——则旨在实现组合性。这一立场……

    arXiv:2611.00172v1 Announce Type: new  Abstract: Embedding models underpin virtually every modern AI system, from search engines to recommendation systems. Despite shared mathematical foundations, research on embedding models remains fragmented by modality: the communities develop largely in isolation, seldom exchanging insights. We argue that this fragmentation obscures a deeper convergence. Driven by contrastive learning, Transformer architectures, and a granularity shift, embedding research across modalities is converging toward a common set of principles, even as modality-specific challenges remain underexplored. To reason across this convergence without conflating it, we propose a framework of three progressively harder capabilities: coverage, comparability, and compositionality. Three paths toward unification satisfy these in turn: transformation broadens coverage, alignment enables comparability, and joint embedding, the terminus we advocate, aims at compositionality. This posit
    
[^241]: HistGramBench：衡量多模态模型将历史数学图形重绘为 TikZ 时的结构恢复能力

    HistGramBench: Measuring Structural Recovery When Multimodal Models Redraw Historical Mathematical Diagrams as TikZ

    [https://arxiv.org/abs/2611.00081](https://arxiv.org/abs/2611.00081)

    提出 HistGramBench 基准，通过两个不同AI模型对历史数学图形中的几何事实达成共识来检验模型重绘的 TikZ 程序，发现最佳的平价模型也只能恢复约五分之三的结构事实。

    

    数字图书馆保存了十六至二十世纪印刷的数学著作，但其中的字母标注图形——即欧几里得证明所依赖的那些插图——仍然只是无法检索内容的扫描图像。将这样的图形重写为一段简短的绘图程序，可以记录下哪些点位于哪些直线或圆上。然而印刷版面并不提供可供核对的源代码。我们提出了 HistGramBench，一个包含1551年至1933年间以八种语言印刷的95部著作中2,507个图形的基准数据集。对于每个图形，由来自不同提供商的两个AI模型就读者从图中获取的事实（例如哪些线是平行的、两条线在哪里相交）达成一致，并且每次重绘都会在模型自己绘制的图形上依据这些事实进行核对。我们评估了六个价格可负担的AI模型，其中表现最好的恢复了约五分之三的事实，而模型遗漏的大部分来自程序……（原文摘要至此截断）

    arXiv:2611.00081v1 Announce Type: cross  Abstract: Digital libraries preserve mathematical books printed from the sixteenth to the twentieth century, but the lettered diagrams in them, the figures on which Euclidean proofs depend, remain scanned images whose content cannot be searched. Redrawing such a diagram as a short drawing program would record which points lie on which lines and circles. But a printed plate offers no source code against which the redrawing could be checked. We introduce HistGramBench, a benchmark of 2,507 diagrams from 95 works printed between 1551 and 1933 in eight languages. For each diagram, two AI models from different providers agree on the facts a reader takes from the figure, such as which lines are parallel or where two lines meet, and every redrawing is checked against those facts on the model's own drawing. We evaluate six affordable AI models. The best recovers about three fifths of these facts, and most of what the models miss comes from programs that
    
[^242]: 从零到OCR：面向无支持低资源文字的数据中心框架

    From Zero to OCR: A Data-Centric Framework for Unsupported Low-Resource Scripts

    [https://arxiv.org/abs/2611.00078](https://arxiv.org/abs/2611.00078)

    本文提出一个以数据为中心的框架，仅需未标注文本语料库和数字字体、无需人工标注任何图像，即可为六种主流OCR引擎均不支持的克什米尔语纳斯塔利格文字等低资源文字构建端到端OCR系统。

    

    光学字符识别（OCR）对许多低资源文字而言仍然难以获得，因为标注的文档图像不可用，而合成训练数据又可能无法保留文字特有的视觉结构。我们提出了一个以数据为中心的框架，仅需一个未标注的文本语料库和数字字体，即可为这类文字构建端到端的OCR系统，而无需人工标注任何一张训练图像。该框架包含一个合成数据引擎，用于渲染具有正确复杂文字整形和自动推导标签的单词、行与页面图像；一个YOLOv8单词检测器和一个采用连接主义时间分类（CTC）训练的卷积循环识别器从这些图像中学习；此外还有一个分阶段门控评估协议，可在引入检测之前单独隔离识别误差。我们在克什米尔语上验证了该框架——这是一种使用波斯-阿拉伯纳斯塔利格（Nastaliq）文字的语言，约有680万使用者，而六种广泛使用的OCR引擎均不支持它。三行-l……（原文摘要在此处截断）

    arXiv:2611.00078v1 Announce Type: cross  Abstract: Optical character recognition (OCR) remains inaccessible to many low-resource scripts because annotated document images are unavailable and synthetic training data can fail to preserve script-specific visual structure. We present a data-centric framework that builds end-to-end OCR for such scripts from only an unannotated text corpus and digital fonts, without manually annotating a single training image. A synthetic data engine renders word, line and page images with correct complex-script shaping and automatically derived labels; a YOLOv8 word detector and a convolutional recurrent recognizer trained with connectionist temporal classification learn from these images; and a stage-gate evaluation protocol isolates recognition error before detection is introduced. We demonstrate the framework on Kashmiri, a Perso-Arabic Nastaliq script with approximately 6.8 million speakers that none of six widely used OCR engines supports. Three line-l
    
[^243]: 当频域分类优于空间域时

    When frequency domain classification is superior to spatial

    [https://arxiv.org/abs/2611.00073](https://arxiv.org/abs/2611.00073)

    研究揭示了频域分类准确率下降的根源在于局部特征的缺失，并据此构建了结合频域与空间域训练的混合模型以获得更高的分类性能。

    

    现代图像分类模型在很大程度上依赖局部特征和特征提取来进行预测。这种聚焦的方法在空间域中效果良好，但据报道在频域中表现不佳，尽管两种表示包含着相同的信息。本研究证明，频域中准确率的显著下降源于局部特征的缺失。在频域中局部交换数据会显著降低其逆变换后空间域的准确率，甚至低于频域本身的准确率；然而，频域的准确率却保持相对稳定。此外，使用在空间域中包含全局特征的数据集会在频域中获得更高的准确率。基于这些结果，研究构建了一个混合模型，将仅在频域训练的模型和仅在空间域训练的模型相结合，以实现更高的分类准确率。

    arXiv:2611.00073v1 Announce Type: cross  Abstract: Modern image classification models largely rely on local characteristics and feature extraction for their predictions. This focused approach works well in the spatial domain, but reportedly falls short in the frequency domain, even though both representations contain the same information. This study demonstrates that the significant reduction in accuracy in the frequency domain stems from the lack of local features. Locally swapping data in the frequency domain significantly decreases its inverse spatial domain accuracy, even below that of the frequency domain; however, the frequency domain accuracy remains relatively constant. Moreover, the use of datasets containing global features in the spatial domain yields higher accuracy in the frequency domain. Using these results, a hybrid model is constructed, combining one model trained only in the frequency domain and another trained only in the spatial domain to achieve higher classificati
    
[^244]: 术中手术专家实时解说内容的系统性分析

    A Systematic Analysis of Live Intraoperative Surgical Expert Narrations

    [https://arxiv.org/abs/2611.00028](https://arxiv.org/abs/2611.00028)

    该研究首次系统刻画了脊柱外科医生术中实时解说所包含的信息，并提出一条无需人工标注的自动化语音处理流程，可将术中解说转换为带时间戳的手术流程提示，为手术视频理解提供了一种可扩展的替代标注方案。

    

    人工标注仍然是手术视频理解中的主要瓶颈。由主刀外科医生进行的术中实时解说是一种无需人工标注的可扩展替代方案，但其包含的信息尚未被系统刻画。我们分析了两位资深脊柱外科医生实施的13例内镜下腰椎减压手术（共13.2小时），使用时间同步的近讲麦克风德语音频。通过一条无需标注的处理流程（词级语音识别、手术术语错误的确定性纠正、语句融合、基于大语言模型的英文翻译以及相关性过滤），将语音转换为带时间戳的手术流程提示，并标注其所指的手术实体及言语行为类型。在3,171条语句中，1,201条（37.9%）被保留为手术流程提示，57.6%属于非手术内容而被移除。保留的提示涉及手术器械（60.7%）、解剖结构（48.0%）和手术动作（57.1%），98.5%至少携带（原文摘要在此处截断）。

    arXiv:2611.00028v1 Announce Type: cross  Abstract: Manual annotation remains the principal bottleneck in surgical video understanding. Live intraoperative narration by the operating surgeon is a scalable alternative that requires no manual annotation, but the information it contains has not been characterized. We analyzed 13 endoscopic lumbar spine decompression procedures (13.2 h) by two senior spine surgeons, using time-synchronized close-mic German audio. An annotation-free pipeline (word-level speech recognition, deterministic correction of surgical-term errors, utterance fusion, LLM-based translation to English, and relevance filtering) converted speech into time-stamped procedural cues labelled with the surgical entities they reference and their speech act. Of 3,171 utterances, 1,201 (37.9%) were retained as procedural cues, while 57.6% were non-procedural and removed. Retained cues referenced instruments (60.7%), anatomy (48.0%) and surgical actions (57.1%), and 98.5% carried at
    
[^245]: 面向边缘优化的YOLOv11n架构：用于尼日利亚手语字母数字手势的离线识别

    Edge-Optimized YOLOv11n Architecture for Offline Recognition of Nigerian Sign Language Alphanumeric Gestures

    [https://arxiv.org/abs/2611.00025](https://arxiv.org/abs/2611.00025)

    本研究提出面向边缘设备优化的轻量级YOLOv11n架构，并建立了包含11,208张图像、33个静态手势类别的尼日利亚手语基准数据集（按手语者严格划分训练、验证与测试集），实现了在内存仅1至4GB的低资源移动硬件上对尼日利亚手语字母数字手势的实时离线识别。

    

    撒哈拉以南非洲社会中聋人群体的融入面临区域性障碍，其中尼日利亚是本研究的核心实证焦点。现有的识别模型需要繁重的计算基础设施，然而当地使用的标准手机内存很少超过1GB至4GB。本研究通过一个轻量级的YOLO11n目标检测框架来解决这些限制，该系统为低资源硬件提供实时的尼日利亚手语识别。研究建立了一个包含11,208张原始手势图像的基准数据集，数据涵盖从五名手语者收集的33个静态类别。数据划分方面，手语者S1至S3用于训练，手语者S4提供验证数据，手语者S5保留用于测试。空间数据增强严格限制在训练集内，最终产生23,509张训练图像。验证集和测试集在多种环境下各包含1,681张图像。该框架集成了量化感知压缩技术。

    arXiv:2611.00025v1 Announce Type: cross  Abstract: Deaf inclusion across sub-Saharan society faces regional hurdles, with Nigeria as the core empirical focus. Existing recognition models demand heavy computational infrastructure. However, standard mobile phones used locally rarely exceed 1 GB to 4 GB of RAM. This study addresses constraints through a lightweight YOLO11n object-detection framework. The system provides real-time Nigerian Sign Language recognition for low-resource hardware. A benchmark dataset containing 11,208 raw gesture images was established. The data spans 33 static classes collected from five signers. Partitioning isolates signers S1 through S3 for training. Signer S4 provides validation data. Signer S5 is reserved for testing. Spatial augmentations are restricted strictly to the training set. This yields 23,509 training images. The validation and test sets each maintain 1,681 images across varied environments. The framework integrates a quantization-aware compressi
    
[^246]: ARTA：基于图组合优化的可穿戴心电信号心跳间期鲁棒估计

    ARTA: Graph Combinatorial Optimization for Robust Interbeat Interval Estimation from Wearable Electrocardiogram Signals

    [https://arxiv.org/abs/2611.00011](https://arxiv.org/abs/2611.00011)

    提出ARTA框架，通过将多检测器候选点构建为有向无环图并利用束搜索动态规划进行受约束的时间优化，从而在运动伪影干扰下从可穿戴ECG信号中鲁棒地估计心跳间期。

    

    从可穿戴心电图（ECG）中准确估计心率（HR）和心跳间期（IBI）对于连续生理监测至关重要，但运动伪影可能产生类似QRS波群的虚假峰值并掩盖真实的R波峰。传统检测器和局部间期修复规则独立处理各个检测结果，当多个连续心跳受到干扰时可能变得不可靠。我们提出了ARTA（Artifact-Robust Temporal Alignment，抗伪影时间对齐），这是一个基于图的组合优化框架，通过受约束的时间优化来估计心跳序列。ARTA整合来自多个ECG检测器的候选结果，将其表示为有向无环图中的节点，并通过生理上合理的RR间期将候选点连接起来。候选点置信度结合了检测器之间的一致性和局部ECG质量，而依赖历史的转移代价则捕捉近期IBI的一致性和软性心率先验。束搜索动态规划……

    arXiv:2611.00011v1 Announce Type: cross  Abstract: Accurate heart rate (HR) and interbeat interval (IBI) estimation from wearable electrocardiogram (ECG) is essential for continuous physiological monitoring, but motion artifacts can create false QRS-like peaks and obscure true R-peaks. Conventional detectors and local interval-repair rules process detections independently and may become unreliable when several consecutive beats are corrupted. We present ARTA (Artifact-Robust Temporal Alignment), a graph-based combinatorial framework that estimates the heartbeat sequence through constrained temporal optimization. ARTA combines candidates from multiple ECG detectors, represents them as nodes in a directed acyclic graph and connects candidates through physiologically plausible RR intervals. Candidate confidence uses detector agreement and local ECG quality, while history-dependent transition costs capture recent IBI consistency and a soft heart-rate prior. Beam-search dynamic programming 
    
[^247]: 《“我写了，我停顿了，我重写了”：学习者对过程感知型大语言模型反馈感知的混合方法研究》

    "I Wrote, I Paused, I Rewrote": A Mixed-Methods Study of Learner Perceptions of Process-Aware LLM Feedback

    [https://arxiv.org/abs/2611.00009](https://arxiv.org/abs/2611.00009)

    通过对43名本科生英语学习者的混合方法研究，发现学习者对基于写作过程（中间草稿、快照及按键记录）生成的大语言模型反馈持积极评价，认为其有用且与自身修改经历相符。

    

    大语言模型日益被用于提供写作形成性反馈，但大多数系统仅评估最终提交的文本。过程感知系统还可以纳入中间草稿和交互记录，然而获取这些痕迹并不能确定学习者是否认为由此产生的反馈有用、公平，或与其自身的修改经历相符。我们对43名本科生英语学习者进行了一项聚合式混合方法任务后评估。参与者完成一篇限时作文，接收基于最终作文、中间快照和按键记录生成的大语言模型反馈，然后回答六个五点李克特量表题目和四个开放式问题。在所有六个题目上，评分均集中在高于中性中点的区间。74.4%至95.3%的参与者选择了4或5的评分，且经Holm校正后，所有六个单侧精确符号检验均保持显著。

    arXiv:2611.00009v1 Announce Type: cross  Abstract: Large language models increasingly provide formative writing feedback, but most systems evaluate only the submitted text. Process-aware systems can additionally incorporate intermediate drafts and interaction records, yet access to these traces does not establish whether learners perceive the resulting feedback as useful, fair, or aligned with their own revision experience. We conducted a convergent mixed-method post-task evaluation with 43 undergraduate English-language learners. Participants completed a timed essay, received LLM-generated feedback based on the final essay, intermediate snapshots, and keystroke-derived records, and then answered six five-point Likert items and four open-ended questions.   Ratings were concentrated above the neutral midpoint across all six items. Between 74.4% and 95.3% of participants selected ratings of 4 or 5, and all six one-sided exact sign tests remained significant after Holm correction. Perceiv
    
[^248]: 文化演化驱动神经涌现通信中约定惯例的产生

    Cultural Evolution Drives the Emergence of Conventions in Neural Emergent Communication

    [https://arxiv.org/abs/2611.00006](https://arxiv.org/abs/2611.00006)

    该论文提出了一种以完全分布式方式训练双向神经智能体种群的新模型，证明了文化演化能够驱动语言约定惯例的产生，且更约定化的语言因更易学习而在文化传递中具有直接优势。

    

    近期利用神经智能体进行语言演化的研究通常采用的模型中，智能体只能说话或只能倾听、种群中仅包含两个智能体、或智能体之间可以共享内部表征。这些设计选择使得无法研究语言如何通过文化演化而产生，即通过个体行为为适应群体交流需求而进行的累积性适应过程。我们通过一个新模型来解决这一问题，该模型以完全分布式的方式训练由双向神经智能体组成的种群来解决指代游戏。我们在多个图像数据集上验证了该方法的有效性，并证明其能够扩展到大规模种群。我们还表明，更加约定化的语言更容易学习，这为文化传递提供了直接优势。总体而言，这项工作为神经涌现通信提供了一种新颖的方法，为演化语言学开辟了新的研究方向。

    arXiv:2611.00006v2 Announce Type: cross  Abstract: Recent studies of language evolution with neural agents often use models where agents can either speak or listen, where there are only two agents in the population, or where agents can share internal representations. These choices prevent the study of how language emerges through cultural evolution, i.e. through the cumulative adaptation of individual behaviours to fit the communicative needs of the community. We tackle this issue with a new model that trains populations of bi-directional neural agents in a fully distributed fashion to solve a referential game. We demonstrate the effectiveness of this approach on multiple image datasets and show that it is able to scale to large populations. We also show that more conventional languages are easier to learn, providing a direct advantage for cultural transmission. Overall, this work provides a novel approach to neural emergent communication that opens up new directions for evolutionary l
    
[^249]: 基于分层树结构索引器的次二次复杂度注意力机制

    Subquadratic Attention with a Hierarchical Tree-Structured Indexer

    [https://arxiv.org/abs/2610.42112](https://arxiv.org/abs/2610.42112)

    本文提出 Zyphra 分层注意力（ZHA），通过在 KV 缓存上构建递归压缩键摘要的树结构索引器，使包含路由在内的总计算成本随序列长度次线性增长，从而突破了注意力二次方复杂度对大语言模型长上下文能力的根本限制。

    

    注意力的二次方扩展是限制大语言模型（LLM）长上下文能力的一个根本瓶颈。已有多种方法试图规避这一问题，但它们要么无法匹配注意力机制的表达能力（例如滑动窗口或具有固定大小状态的线性注意力变体），要么只能实现常数因子的加速（例如稀疏注意力方法，其索引器本身的复杂度仍是二次方量级）。在本文中，我们基于分层树结构索引器推导出一种渐近次二次复杂度的稀疏注意力方法，我们将其称为 Zyphra 分层注意力。ZHA 在 KV 缓存之上构建一棵由递归压缩的键摘要组成的树，后续的查询从树顶向下遍历，以检索出一个稀疏的键子集。至关重要的是，通过迭代地增加树的深度，包括路由和最终注意力计算在内的总成本随序列长度呈次线性增长……

    arXiv:2610.42112v1 Announce Type: new  Abstract: The quadratic scaling of attention is a fundamental bottleneck limiting the long-context capabilities of large language models (LLMs). A variety of approaches have attempted to circumvent this, but they either fail to match attention's expressiveness (e.g., sliding-window or linear attention variants with fixed-size state) or only achieve constant-factor speedups (e.g., sparse attention methods whose indexers themselves scale quadratically). In this paper, we derive an asymptotically subquadratic sparse attention method based on a hierarchical, tree-structured indexer, which we call Zyphra Hierarchical Attention (ZHA). ZHA constructs a tree of recursively compressed key summaries over the KV cache, which future queries traverse from the top down to retrieve a sparse subset of keys. Crucially, by iteratively growing the tree's depth, the total cost including routing and final attention grows sublinearly in sequence length, resulting in ov
    
[^250]: 利用大型语言模型增强学术论文摘要与全文内容中的知识实体关系抽取

    Enhancing Knowledge Entity Relation Extraction in Abstracts and Full-text Content of Academic Articles Using Large Language Models

    [https://arxiv.org/abs/2610.41552](https://arxiv.org/abs/2610.41552)

    本研究系统比较了多种大语言模型策略用于学术论文的知识实体关系抽取，并创新性地提出了SLM增强开源LLM和闭源LLM增强开源LLM两种新组合策略。

    

    本研究探讨了大型预训练语言模型（LLM）在学术论文知识实体关系抽取（KERE）任务中的有效性。现有研究仅考察了单一类型的基于LLM的策略，并使用精确率、召回率等指标评估模型性能。为了提供更全面的评估，本研究考察了多种策略，包括基于闭源LLM的思维链（CoT）提示、经LoRA适配的开源LLM、LLM增强的小型预训练语言模型（SLM），以及SLM增强的闭源LLM。此外，本研究还引入了两种新策略，即SLM增强的开源LLM和闭源LLM增强的开源LLM。这些策略从多个角度进行评估，包括指标表现、缓解SLM中常见错误的能力，以及模型优化的潜在方向。本研究在自然语言处理（NLP）领域人工构建了两个标注数据集，涵盖……（摘要在此处截断）

    arXiv:2610.41552v1 Announce Type: new  Abstract: This study investigates the effectiveness of large pre-trained language models (LLMs) for knowledge entity relation extraction (KERE) from academic papers. Existing studies have examined individual types of LLM-based strategies and evaluated model performance using metrics such as precision and recall. To provide a broader evaluation, this study investigates multiple strategies, including closed-source LLM-based CoT prompting, LoRA-adapted open-source LLMs, LLM-enhanced small pre-trained language models (SLMs), and SLM-enhanced closed-source LLMs. In addition, this study introduces two strategies, namely SLM-enhanced open-source LLMs and closed-source LLM-enhanced open-source LLMs. These strategies are evaluated from multiple perspectives, including metrics, their ability to mitigate errors observed in SLMs, and potential directions for model optimization. This study manually constructs two annotated datasets in the NLP domain, covering 
    
[^251]: PsyInterviewee：心理健康访谈模拟中的信息不变性与关系适应性

    PsyInterviewee: Informational Invariance and Relational Adaptivity in Mental Health Interview Simulation

    [https://arxiv.org/abs/2610.41289](https://arxiv.org/abs/2610.41289)

    该论文提出PsyInterviewee框架，通过分离受保护的真实信息通道与动态交互通道并采用非对称写入权限机制，在心理健康访谈模拟中同时保证临床事实的信息不变性与交互行为的适应性。

    

    基于大语言模型（LLM）的模拟受访者能够支持访谈员培训以及心理健康访谈系统的评估。现有研究主要通过更丰富的患者档案和行为建模来提升拟人化程度。然而，仅凭拟人化并不足以确立模拟的可靠性。可靠的模拟既需要保留基于档案的临床事实（信息不变性），又需要使信息透露和交互行为适应访谈情境（关系适应性）。为此，我们提出了PsyInterviewee，该方法将受保护的真实信息通道与动态交互通道相分离，并对生成的临床陈述与患者档案进行核对验证。具体而言，非对称写入权限允许对话上下文更新信息透露状态和行为状态，但无权创建新的临床事实；这些状态进一步指导哪些有档案支持的信息被透露以及如何表达。在一项自动化评估中……（原文摘要至此截断）

    arXiv:2610.41289v1 Announce Type: new  Abstract: LLM-based simulated interviewees support interviewer training and the evaluation of mental health interviewing systems. Existing studies have primarily focused on improving human-likeness through richer patient profiles and behavioral modeling. However, human-likeness alone does not establish their reliability. Reliable simulation requires preserving profile-grounded clinical facts (informational invariance) while adapting disclosure and interaction behavior to the interview (relational adaptivity). To this end, we introduce PsyInterviewee, which separates a protected truth channel from a dynamic interaction channel and verifies generated clinical claims against the profile. Specifically, asymmetric write permissions allow dialogue context to update disclosure and behavioral states without authorizing new clinical facts; these states guide what profile-supported information is revealed and how it is expressed. In an automated evaluation 
    
[^252]: Li²SAO：通过组相对Actor-Critic自举提升轻量级单rollout异步强化学习优化

    Li$^{2}$SAO: Lifting Light Singl-Rollout Asynchronous RL Optimization via Group-Relative Actor-Critic Bootstrapping

    [https://arxiv.org/abs/2610.40726](https://arxiv.org/abs/2610.40726)

    该论文提出Li²SAO，通过组相对Actor-Critic自举摆脱对大规模预训练价值模型的依赖，在轻量级单rollout异步RL优化中兼顾推理任务与智能体任务的表现。

    

    异步强化学习（RL）正日益成为大语言模型（LLM）后训练的主流范式，因为它可以在rollout到达时即更新模型，从而提升训练效率。近期研究指出，GRPO中的组级采样并不天然契合异步RL训练，因为rollout长度服从长尾分布，并据此提出了单rollout异步优化（SAO），以提升稳定性和离策略表现，尤其是在智能体任务上。然而，SAO依赖于一个通过大规模价值预训练获得的闭源价值模型。本文在单轮推理任务和多轮智能体任务上分析了GRPO与LiSAO——即我们对SAO的轻量级复现（不含预训练价值模型）。我们发现，GRPO在推理任务上优于LiSAO，而在智能体任务上则情况相反。由于LiSAO的价值模型尚未得到充分训练，LiSAO在此方面大幅落后于GRPO……（原文摘要至此截断）

    arXiv:2610.40726v1 Announce Type: new  Abstract: Asynchronous reinforcement learning (RL) is becoming a mainstream paradigm for post-training large language models (LLMs), as it improves training efficiency by updating the model as rollouts arrive. Recent work argues that group-wise sampling in GRPO does not naturally fit asynchronous RL training because rollout lengths follow a long-tailed distribution, and proposes Single-rollout Asynchronous Optimization (SAO) to improve stability and off-policy behavior, especially on agentic tasks. However, SAO relies on a closed-source value model obtained from scaling value pretraining. In this paper, we analyze GRPO and LiSAO, our lightweight reproduction of SAO without the pretrained value model, on single-turn reasoning and multi-turn agentic tasks. We find that GRPO outperforms LiSAO on reasoning tasks, while the reverse holds on agentic tasks. Since the value model of LiSAO is not yet well trained , LiSAO lags behind GRPO substantially in e
    
[^253]: 多视角潜在混合专家：让每个路由器塑造专家计算

    Multi-View Latent MoE: Letting Every Router Shape Expert Computation

    [https://arxiv.org/abs/2610.40577](https://arxiv.org/abs/2610.40577)

    提出多视角潜在MoE框架，利用多个独立路由器聚合的信号改进专家选择，并复用路由器偏好来调制专家计算，从而在固定专家预算下提升模型能力。

    

    arXiv:2610.40577v2 公告类型： replace 摘要：稀疏混合专家模型通过为每个词元激活一小部分专家来扩展参数容量。在固定的活跃专家预算下，专家选择直接影响模型能力。对所有专家进行稠密评估发现，单路由器的排序与专家效用评估排序之间的一致性较弱，尤其是在中间层和深层。更丰富的选择与计算信号会带来额外成本：基于内部专家激活进行选择需要预先计算，而将多个完整表示分别通过同一专家处理则会重复计算。我们提出多视角潜在MoE，通过多个路由器改进专家选择，并在固定的专家预算下复用它们的偏好来调制专家计算。其多路由器子模块聚合来自独立初始化路由器的sigmoid分数。这些路由器发展出高度差异化的专家偏好……

    arXiv:2610.40577v2 Announce Type: replace  Abstract: Sparse mixture-of-experts (MoE) models expand parameter capacity by activating a small subset of experts per token. Expert selection directly affects model capabilities under a fixed active-expert budget. Dense evaluation of all experts reveals weak agreement between single-router rankings and expert-utility evaluation rankings, especially in middle and deep layers. Richer signals for selection and computation incur additional cost: selection based on internal expert activations requires preliminary computation, while processing multiple full representations separately through the same expert repeats its computation. We propose Multi-View Latent MoE to improve expert selection with multiple routers and reuse their preferences to modulate expert computation under a fixed expert budget. Its Multi-Router submodule aggregates sigmoid scores from independently initialized routers. These routers develop highly differentiated expert prefere
    
[^254]: 利用重插机制与尺寸上界研究带可移除物品的在线背包问题

    Leveraging Recourse and Size Bounds for OnlineKnapsack with Removable Items

    [https://arxiv.org/abs/2610.40290](https://arxiv.org/abs/2610.40290)

    该论文在带可移除物品的在线背包问题中，首次将物品大小上界感知与允许重插被拒物品的重插机制相结合，建立了竞争比的紧密上下界，并显著改进了已有的最优竞争比。

    

    在在线无权重背包问题中，物品依次到达，必须立即决定接受或拒绝，将其装入一个容量有限的背包，目标是最大化被接受物品的总大小。当决策不可撤销时，任何算法都无法达到非平凡的竞争比。一种规避这一不可能结果的重要放松是允许将之前已接受的物品从背包中移除（即“移除”机制）。在本工作中，我们通过结合另外两个先前提出的模型扩展，研究能否进一步提升带移除机制的竞争比：其一是知道物品大小的上界（即“上界感知”），其二是能够重新插入之前被拒绝的物品（即“重插”机制）。我们的贡献有两方面。首先，我们在上界感知、移除和重插机制下建立了竞争比的紧密上下界。我们证明了……（原文摘要在此处被截断）在最佳背包问题上可取得显著改进……

    arXiv:2610.40290v2 Announce Type: replace-cross  Abstract: In the online unweighted knapsack problem, items arrive sequentially and must be accepted or rejected immediately for a bounded-capacity knapsack, aiming to maximize the total size of accepted items. When decisions are irrevocable, no algorithm achieves a non-trivial competitive ratio. An essential relaxation that circumvents this impossibility is to allow previously accepted items to be removed from the knapsack (a.k.a. removal). In this work, we investigate possible further improvements on the competitive ratio with removal by bringing together two other previously proposed model extensions: knowledge of an upper bound on the item sizes (a.k.a. bound awareness) and ability to reinsert previously rejected items(a.k.a.recourse). Our contributions are twofold. First, we establish close upper and lower bounds on the competitive ratio under bound awareness, removal and recourse. We show that significant improvements on the best kn
    
[^255]: 通过工具集剪枝与模态整合实现高效的智能体音视频问答

    Efficient Agentic Audio-Visual Question Answering through Toolset Pruning and Modality Consolidation

    [https://arxiv.org/abs/2610.38960](https://arxiv.org/abs/2610.38960)

    本文通过分阶段的工具集剪枝与模态整合实验，将庞大的专用工具智能体压缩为仅含两个全模态工具的紧凑工作流，在保持细粒度音视频问答性能的同时显著降低推理成本与执行开销。

    

    细粒度音视频问答（AVQA）要求模型解析复杂的跨模态对应关系，并从稀疏分布的证据中进行时间定位。全模态智能体通过自主调度感知工具来主动收集证据，取得了优异的性能，但目前尚不清楚对音频与视觉两种模态的主动感知是否依赖于广泛的专业化工具集，尤其是音频或视觉模态专用的工具。此类系统还可能引入冗长的执行轨迹和高昂的推理成本。我们研究了仅暴露必要感知能力的紧凑工作流能否保持相当的性能。从配备专用工具的OmniAgent出发，我们开展了分阶段的受控实验，对工具集进行剪枝，并逐步将模态专用工具整合为全模态工具，由此构建出一个仅包含全模态能力的两工具智能体。

    arXiv:2610.38960v1 Announce Type: new  Abstract: Fine-grained audio-visual question answering (AVQA) requires models to resolve complex cross-modal correspondences and temporally grounding from sparsely distributed evidence. Omni-modal agents, which autonomously orchestrate perception tools to actively collect evidence, achieve strong performance, but it remains unclear whether active perception from both audio and visual modalities depends on broad specialized toolsets, especially audio or visual modality-specific tools. Such systems may also introduce long execution trajectories and substantial inference cost. We investigate whether comparable performance can be retained with a compact workflow exposing only essential perceptual capabilities. Starting from the specialized-tool OmniAgent, we conduct staged controlled experiments that prune the toolset and progressively consolidate modality-specific tools into omni-modal tools, motivating a two-tool agent comprising only Omni-modal cap
    
[^256]: 超越凭空捏造：AI科研智能体所写论文中的错误归属数字

    Beyond fabrication: Misattributed numbers in papers written by AI research agents

    [https://arxiv.org/abs/2610.38691](https://arxiv.org/abs/2610.38691)

    该研究重新分析了AI科研智能体基准测试的判定结果，发现智能体所写论文中的数值错误主要并非凭空捏造，而是将近八成的无支持声明错误地归属了智能体自身文件中来自错误实验、指标或运行的数值，并提出了一种确定性的声明-证据检查器来检测此类问题。

    

    验证AI智能体所撰写研究中的数值声明是一个重要但具有挑战性的问题。近期的基准测试表明，许多由智能体撰写的论文无法通过完整性检查，但此类判定（通常由单个大语言模型评判者做出）只能识别失败而无法解释失败的性质，且现有的检查方法要么测试论文的内部一致性，要么测试所报告的数值是否出现在智能体输出的某处。我们重新分析了近期一个公开的AI科研智能体基准测试（九个智能体、十五个开放研究问题）的全部540个已发布判定，并人工标注了来自18份提交作品的256个数值声明。研究发现，失败主要集中在论文与底层工作之间的对齐问题上，而非明显的作弊行为，大约五分之四缺乏支持的声明对应于智能体自身文件中来自错误实验、错误指标或错误运行的数值。我们引入了一个确定性的声明-证据检查器，用于评估……

    arXiv:2610.38691v1 Announce Type: new  Abstract: Verifying numerical claims in research written by AI agents is an important but challenging problem. Recent benchmarks show that many agent-written papers fail integrity checks, but such verdicts, often produced by a single LLM judge, identify failure without explaining its nature, and existing checks either test a paper for internal consistency or test whether a reported value appears somewhere in an agent's output. We re-analyze all 540 released judgments from a recent public benchmark of AI research agents (nine agents, fifteen open research problems) and manually label 256 numerical claims from 18 submissions. Failures concentrate on alignment between the paper and the underlying work rather than overt cheating, and about four in five unsupported claims correspond to values in the agent's own files that come from the wrong experiment, metric, or run. We introduce a deterministic claim-to-evidence checker that scores whether the surro
    
[^257]: Keepace.ai：一个面向运动与健康领域的大语言模型技术报告

    Keepace.ai: A Technical Report on a Large Language Model for Sports and Health

    [https://arxiv.org/abs/2610.38249](https://arxiv.org/abs/2610.38249)

    Keepace是基于Qwen3.5-27B后训练的中文运动健康领域专用大语言模型，在健身计划生成、知识问答和运动数据分析三项任务上超越了基座模型及更大的通用模型，尤其在运动安全性、边界合规性和科学准确性方面提升显著。

    

    大语言模型有望大规模提供安全、科学且个性化的运动指导，但运动与健康领域目前仍缺乏经过精心整理的训练语料和严格的评估基准。Keepace（Keepace.ai）是基于Qwen3.5-27B通过后训练构建的领域专用大语言模型，目前针对中文运动指导进行了优化，涵盖三项任务：健身计划生成、运动与健康知识问答以及个人运动数据分析。在与领域专家共同设计的基于评分量表的内部评估中，Keepace的表现优于其基座模型和更大的通用模型基线，其中在运动安全性、边界合规性、结构化输出完整性和科学准确性方面的提升最为显著。本报告总结了Keepace当前的设计、数据流程、安全优先的评估框架以及评估结果。

    arXiv:2610.38249v2 Announce Type: replace  Abstract: Large language models hold promise for delivering safe, scientifically grounded, and personalized exercise guidance at scale, yet the sports-and-health domain still lacks curated training corpora and rigorous evaluation benchmarks. Keepace (Keepace.ai) is a domain-specialized LLM built on Qwen3.5-27B through post-training, currently optimized for Chinese-language exercise guidance across three tasks: workout plan generation, sports-and-health knowledge QA, and personal sports data analysis. In internal rubric-based evaluations designed with domain experts, Keepace outperforms its base model and larger general-purpose baselines, with the most pronounced improvements in exercise safety, boundary compliance, structured output completeness, and scientific accuracy. This report summarizes the current design, data pipeline, safety-first evaluation framework, and evaluation results of Keepace.
    
[^258]: 面向大语言模型对齐的排序感知差分隐私列表式偏好发布

    Rank-Aware Differentially Private Release of Listwise Preferences for LLM Alignment

    [https://arxiv.org/abs/2610.37682](https://arxiv.org/abs/2610.37682)

    本文提出一种排序感知的指数机制，将列表式偏好数据私有化为低敏感性的固定大小二分划分粒度，解决了完整排序在本地差分隐私下噪声过大的难题，实现了隐私保护的大语言模型列表式偏好对齐。

    

    列表式偏好反馈为大语言模型（LLM）对齐提供了比成对比较更丰富的监督信息，但完整的排序也会暴露更多敏感的用户偏好。现有的隐私保护对齐方法主要针对成对反馈，而在本地差分隐私（LDP）下的列表式对齐仍未被探索。将LDP扩展到列表式领域面临一个关键挑战：完整排序空间的高敏感性需要引入过量噪声，从而抵消列表式监督的优势。为解决这一挑战，我们分析了监督粒度与隐私噪声在下游训练中的相互作用，并提出了一种排序感知的指数机制，将列表式偏好数据私有化为适合下游对齐的低敏感性粒度。该机制利用排序信息来采样固定大小的二分划分，将噪声集中在边界附近（原文摘要至此处截断）。

    arXiv:2610.37682v1 Announce Type: cross  Abstract: Listwise preference feedback offers richer supervision for large language model (LLMs) alignment than pairwise comparisons, but a full ranking also reveals more sensitive user preferences. Existing privacy-preserving alignment methods focus mainly on pairwise feedback, while listwise alignment under local differential privacy (LDP) remains unexplored. Extending LDP to this listwise domain raises a critical challenge: the high sensitivity of the full ranking space demands excessive noise, neutralizing the benefits of listwise supervision. To address this challenge, we analyze the interaction between supervision granularity and privacy noise in downstream training, and introduce a rank-aware exponential mechanism that privatizes listwise preference data into a low-sensitivity granularity suitable for downstream alignment. The mechanism leverages ranking information to sample a fixed-size binary partition, concentrating the noise near bor
    
[^259]: 形式重于内容？基于大语言模型的同行评审中的语言偏见

    Style over Substance? Linguistic Bias in LLM-based Peer Review Copy title

    [https://arxiv.org/abs/2610.35254](https://arxiv.org/abs/2610.35254)

    该论文提出了一个受控实验框架，通过基于语言学文献推导的细粒度改写指令，系统性地改变论文的语言风格（母语/非母语特征与高/低置信度特征），以研究基于大语言模型的同行评审是否会受到与科学内容无关的语言偏见的影响。

    

    近年来，向主要人工智能会议提交的论文数量急剧增加，大幅加重了研究人员的同行评审工作量。与此同时，大语言模型（LLM）正越来越多地被用于支持同行评审，无论是被个别审稿人使用，还是被会议组织方采用。尽管LLM的能力已有显著提升，但其评估结果可能会受到与科学内容无关的语言特征的影响。在这项工作中，我们引入了一个受控实验框架，通过系统地改变研究论文的语言风格来研究这类效应。具体而言，我们对论文进行改写，使其分别呈现（i）与英语母语或非母语写作相关的语言特征，以及（ii）高置信度或低置信度的语言特征。为了构建这些干预措施，我们以语言学文献为基础推导出细粒度的改写指令，并将其应用于……

    arXiv:2610.35254v1 Announce Type: new  Abstract: The number of paper submissions to major AI conferences has increased dramatically in recent years, substantially increasing the peer-review workload of researchers. At the same time, large language models (LLMs) are increasingly used to support peer review, both by individual reviewers and by conference organizations. While LLM capabilities have improved considerably, their evaluations may be influenced by linguistic characteristics that are unrelated to scientific content. In this work, we introduce a controlled experimental framework for studying such effects by systematically perturbing the linguistic style of research papers. Specifically, we rewrite papers to exhibit (i) linguistic features associated with native or non-native English writing, and (ii) high- or low-confidence linguistic features. To construct these interventions, we derive fine-grained rewriting instructions grounded in the linguistic literature and apply them usin
    
[^260]: T2D：面向高效投机解码的目标模型到草稿模型剪枝与适配

    T2D: Target-to-Draft Model Pruning & Adaptation for Efficient Speculative Decoding

    [https://arxiv.org/abs/2610.34103](https://arxiv.org/abs/2610.34103)

    提出T2D框架，将草稿模型设计表述为优化问题，通过延迟感知的结构化剪枝和知识蒸馏直接从目标模型构建草稿模型，实现推理延迟与分布对齐的联合优化，从而提升投机解码效率。

    

    投机解码显著加速了大型语言模型（LLMs）中的自回归文本生成，但其有效性主要取决于两个关键因素：草稿模型与目标模型之间的分布对齐程度，以及草稿模型在推理过程中的延迟。然而，现有方法通常使用独立预训练的小型草稿模型，或者试图将目标模型剪枝到手动选择的架构，从而无法显式地平衡草稿模型对齐与token生成效率之间的权衡。为了解决这一缺陷，我们将草稿模型设计表述为一个优化问题，并提出了T2D框架，该框架通过延迟感知的结构化剪枝和知识蒸馏，直接从目标模型构建草稿模型。通过这种方式，T2D获得的草稿模型在实测推理延迟和目标模型响应的交叉熵损失上实现了联合优化。

    arXiv:2610.34103v1 Announce Type: cross  Abstract: Speculative decoding significantly accelerates autoregressive text generation in Large Language Models (LLMs), but its efficacy depends largely on two key factors: the distributional alignment between the draft and the target model, and the draft model's latency during inference. However, existing approaches typically utilize independently pretrained small draft models or seek to prune the target model to a manually chosen architecture, thereby failing to explicitly balance the trade-off between draft model alignment and token generation efficiency. To address this gap, we formulate draft model design as an optimization problem and propose T2D, a framework that constructs draft models directly from the target model using latency-aware structured pruning and knowledge distillation. In this manner, T2D obtains draft models that are jointly optimized for both measured inference latency and cross-entropy loss on the target model's response
    
[^261]: HerHealthGPT：一个面向女性生殖健康的、以证据为依据的对话式AI系统

    HerHealthGPT: An Evidence-Grounded Conversational AI System for Women's Reproductive Health Copy title

    [https://arxiv.org/abs/2610.33931](https://arxiv.org/abs/2610.33931)

    HerHealthGPT是一个面向女性生殖健康、以证据为依据的对话式AI系统，它融合检索增强生成、情境感知的澄清提问与来源验证，通过主动追问补全不完整的健康咨询，并基于可验证的证据生成回答。

    

    “为什么我的月经推迟了？”有时，最有用的回答是从另一个问题开始的。然而，对话式AI系统在面对不完整的健康咨询时，往往不会寻求必要的背景信息，也不会将其建议建立在可验证的证据之上。本文提出了HerHealthGPT，一个以证据为依据的对话式AI系统，旨在应对女性生殖健康领域的这些挑战。HerHealthGPT将检索增强生成（RAG）、情境感知的澄清提问和来源验证整合在一个交互式网页界面中。该系统不会立即对不完整的提问作出回应，而是保持对话上下文、提出有针对性的后续问题、检查是否过早下结论，并将其回答建立在检索到的证据之上。我们使用从90个生殖健康案例中提取的540条英文消息对AI组件进行了评估，并开展了针对30名成人的原型问卷调查。证据检索表……

    arXiv:2610.33931v1 Announce Type: new  Abstract: "Why is my period late?" Sometimes, the most useful response begins with another question. Yet conversational AI systems may respond to incomplete health queries without seeking essential context or grounding their guidance in verifiable evidence. This paper introduces HerHealthGPT, an evidence-grounded conversational AI system designed to address these challenges in women's reproductive health. HerHealthGPT integrates retrieval-augmented generation (RAG), context-aware clarification, and source validation within an interactive web interface. Rather than responding immediately to incomplete queries, the system maintains conversational context, asks targeted follow-up questions, checks for premature conclusions, and grounds its responses in retrieved evidence. We evaluate the AI components using 540 English messages derived from 90 reproductive-health cases and conduct a prototype questionnaire study with 30 adults. Evidence retrieval imp
    
[^262]: 面向历史文档转录的实体感知OCR系统评估

    Entity-Aware Evaluation of OCR Systems for Historical Document Transcription

    [https://arxiv.org/abs/2610.33004](https://arxiv.org/abs/2610.33004)

    该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。

    

    光学字符识别（OCR）对于历史档案数字化至关重要，然而字符错误率（CER）等标准评估指标并不一定能反映姓名、日期和标识符等有意义的信息是否被保留。在本工作中，我们研究了OCR错误的实体级分析，作为评估历史文档OCR质量的一种补充视角。我们使用来自Berrutti档案的175份文档的公开基准数据集，并辅以5,209个手动标注的实体提及（将于论文被接收后发布），比较了序列识别OCR系统、OCR专用视觉语言模型以及通用视觉语言模型。除了全局转录准确率之外，我们还在不同的字符级编辑容差下分析了实体的保留性和可恢复性。我们的实验表明，OCR系统的排名取决于评估标准：虽然Qwen 3.6在……（摘要原文在此处被截断）

    arXiv:2610.33004v1 Announce Type: cross  Abstract: Optical Character Recognition (OCR) is essential for digitizing historical archives, yet standard evaluation metrics such as Character Error Rate (CER) do not necessarily reflect whether meaningful information, such as names, dates, and identifiers, is preserved. In this work, we investigate entitylevel analysis of OCR errors as a complementary perspective for evaluating OCR quality in historical documents. Using a publicly available benchmark of 175 documents from the Berrutti Archive, augmented with 5,209 manually annotated entity mentions that will be released upon acceptance, we compare sequence-recognition OCR systems, OCR-specialized vision-language models, and general-purpose vision-language models. Beyond global transcription accuracy, we analyze entity preservation and recoverability under varying character-level edit tolerances. Our experiments show that OCR rankings depend on the evaluation criterion: while Qwen 3.6 achieves
    
[^263]: 先测量再训练：类型化决策模型 vs. 经典基线 vs. 生成式大语言模型在个人多语言收件箱路由中的应用

    Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox

    [https://arxiv.org/abs/2610.32955](https://arxiv.org/abs/2610.32955)

    该论文提出自托管个人邮件代理Foyer及其测量协议，在真实多语言邮件数据上于训练前建立人工真实标注基准，并系统比较了类型化决策模型、经典机器学习基线与生成式大语言模型等九个系统的表现。

    

    个人邮件代理需要为每封邮件做出决策：这封邮件属于什么类型、是否紧急、是否需要起草回复、创建任务，还是直接忽略。然而这些系统的错误率很少被真正测量，因为存储的标签是由系统自身生成的。我们提出了Foyer——一个正在日常使用的自托管个人收件箱代理，以及一套测量协议（分层采样、冻结快照、按邮件线程分组的划分、预注册假设、在代理审查循环中由人工标注），该协议能够在任何训练开始之前获得真实标注数据。在739封由所有者亲自标注的真实意大利语/英语邮件上（其中231封为无偏样本，500封为对抗性采样），我们在按线程分组的交叉验证下比较了九个系统：生产环境的关键词规则、TF-IDF+逻辑回归经典基线、本地类型化决策模型、托管决策模型，以及托管生成式大语言模型。论文给出了四项主要结果，其中生产环境规则在邮件类型识别上仅获得28%的得分，而对照系统达到了58%。

    arXiv:2610.32955v1 Announce Type: new  Abstract: Personal email agents decide, for every message, what it is, whether it is urgent, whether to draft a reply, create a task, or ignore it. Their error rate is rarely measured: the stored labels were written by the system itself. We present Foyer, a self-hosted personal inbox agent in daily use, and a measurement protocol (stratified sampling, frozen snapshots, thread-grouped splits, pre-registered hypotheses, human labels in the agent's review loop) that yields ground truth before any training. On 739 real Italian/English emails labelled by the owner (231 unbiased, 500 adversarially sampled), we compare nine systems under thread-grouped cross-validation: the production keyword rules, a TF-IDF + logistic-regression baseline, local typed decision models (Tev1, Nimble), a hosted decision model (Nimble v4-max), and hosted generative LLMs (Qwen2.5-7B, Gemma 4 26B/31B). Four results. The production rules score 28% on message type against a 58% 
    
[^264]: 大语言模型向低资源黏着语的参数高效适配：面向巴什基尔语的语料库、数据集与可复现基准

    Parameter-Efficient Adaptation of Large Language Models to a Low-Resource Agglutinative Language: A Corpus, Datasets, and a Reproducible Benchmark for the Bashkir Language

    [https://arxiv.org/abs/2610.31397](https://arxiv.org/abs/2610.31397)

    该论文发布了巴什基尔语的开放语料库、标注数据集及系统性适配基准，证明基于QLoRA的参数高效适配结合持续预训练，可使3.8-8B参数的大语言模型在该低资源黏着语的文本分类任务上显著超越传统编码器模型微调。

    

    巴什基尔语是一种使用西里尔字母书写的低资源黏着语，在现代大语言模型中的代表性仍然不足。我们提出了一套开放资源套件和适配方法的系统性基准测试：巴什基尔语网络语料库（71,567篇文档、4690万词元、16个来源），以及三个分别用于二分类、多分类和多标签文本分类的标注数据集；该基准测试涵盖前两种分类任务。实验使用了八种架构（参数规模82M-8B：DistilGPT2、GPT-2 base/medium、Phi-3-mini、Mistral-7B、YandexGPT-5-Lite-8B、mBERT、XLM-R）和三种适配方法（全量微调、QLoRA、上下文学习），并辅以持续预训练（CPT）。实验结果表明：（1）在二分类任务上，对3.8-8B参数模型应用QLoRA优于仅编码器模型的微调：结合CPT的YandexGPT-5-Lite-8B达到宏平均F1值0.748，而XLM-R为0.653、mBERT为0.650；经Holm校正后两两差异均显著（p<0.001）。（2）YandexGPT-5-Lite-8B显著优于Mistral-7B和Ph……（原文摘要在此处截断）

    arXiv:2610.31397v1 Announce Type: new  Abstract: Bashkir is a low-resource agglutinative language written in Cyrillic and remains poorly represented in modern LLMs. We present an open resource suite and a systematic benchmark of adaptation methods: the Bashkir Web Corpus (71,567 documents, 46.9M tokens, 16 sources) and three labelled datasets for binary, multiclass and multilabel text classification; the benchmark covers the first two. Experiments use eight architectures (82M-8B: DistilGPT2, GPT-2 base/medium, Phi-3-mini, Mistral-7B, YandexGPT-5-Lite-8B, mBERT, XLM-R) and three adaptation groups (full fine-tuning, QLoRA, in-context learning), plus continual pretraining (CPT). (1) QLoRA on 3.8-8B models outperforms encoder-only fine-tuning on binary classification: YandexGPT-5-Lite-8B with CPT reaches macro-F1 0.748 vs 0.653 (XLM-R) and 0.650 (mBERT); pairwise differences are significant after Holm correction (p<0.001). (2) YandexGPT-5-Lite-8B significantly outperforms Mistral-7B and Ph
    
[^265]: 成对评估所遗漏的：跨提示的奖励分数

    What Pairwise Evaluation Misses: Reward Scores across Prompts

    [https://arxiv.org/abs/2610.31090](https://arxiv.org/abs/2610.31090)

    论文提出一种以人类评分为参照的审计方法，揭示奖励模型分数存在按提示聚集的跨提示不可比成分——这是成对评估无法检测到的，六个奖励模型的提示间方差份额高达0.34-0.64，而人类评分接近于零。

    

    奖励模型为偏好优化、过滤和排序对响应进行评分，而跨提示汇总分数要求这些分数具有可比性。然而，成对评估比较的是同一提示下的两个响应，因此某个提示的所有响应所共享的任何分数成分都会被抵消，从而无法被检验。我们通过比较模型分数在提示之间的变异与人类对相同响应评分的相应变异来研究这一成分。我们提出了一种以人类为参照的审计方法，该方法由提示间方差份额、同胞评分回归和提示替换构成，并将其应用于十八个已发布的评分器和六个匹配的训练条件。我们发现分数按提示聚集的程度高于人类评分：在 HelpSteer2 上，六个奖励模型的提示间份额为 0.34 至 0.64，而人类评分的份额接近于零。用固定指令替换原始提示后，大量的聚集性依然存在，但与提示级人类评分的一致性被削弱。

    arXiv:2610.31090v1 Announce Type: new  Abstract: Reward models score responses for preference optimization, filtering and ranking, and pooling scores across prompts needs them to be comparable. Yet pairwise evaluation compares two responses to the same prompt, so any score component shared by a prompt's responses cancels and goes untested. We study this component by comparing between-prompt variation in model scores with the same variation in human ratings of the same responses. We propose a human-referenced audit, built from a between-prompt variance share, a sibling-rating regression and prompt substitutions, and apply it to eighteen released scorers and six matched training conditions. We find that scores cluster by prompt more than the ratings do: on HelpSteer2, six reward models have between-prompt shares of 0.34 to 0.64 against a human share near zero. Replacing the original prompt with a fixed instruction keeps substantial clustering but weakens agreement with prompt-level human
    
[^266]: 有限可验证的分段：配备有限状态验证器的每个分词器均可判定

    Finitely Verifiable Segmentation: Decidability for Every Tokenizer with a Finite-State Verifier

    [https://arxiv.org/abs/2610.30168](https://arxiv.org/abs/2610.30168)

    本文提出“有限可验证分段”这一概念——即分段策略的结果恰好可被一个有限自动机验证——并证明在此假设下四个实际分词问题（边界窗口证明、边界间隙有界性、掩码生成可补全性以及策略等价性）均变为可判定问题。

    

    arXiv:2610.30168v1 公告类型：cross 摘要：分段策略为每个字节串指定至多一种切分为词元的分段方式。当定义在带标记字符串（即字节与边界位成对）上的有限自动机恰好接受该策略自身的分段时，我们称该策略是有限可验证的。该自动机用于评判一个给定的分段方案，而无需自行计算分段。本文展示了这一假设所带来的收益。四个原本在实践中靠经验规则解决的问题由此变为可判定的：一个窗口是否能证明某个词元边界、边界间隙是否有界、一个掩码是否能保持生成可补全，以及两个策略是否等价。后三个问题分别特化了单值转换器的有限性、余可达性与等价性问题；第一个问题精化了码理论中的同步切割概念，另外两个问题则询问一个经验证的切割、或某个清单中的所有切割，是否允许每个片段独立扫描。该类策略严格介于在线可扫描策略与具有正则定义域的策略之间，并且……（原文摘要在此处截断）

    arXiv:2610.30168v1 Announce Type: cross  Abstract: A segmentation policy assigns to each byte string at most one segmentation into tokens. Call it finitely verifiable when a finite automaton over marked strings, bytes paired with boundary bits, accepts exactly the policy's own segmentations. The automaton judges a proposed segmentation and need not compute one. The paper shows what that hypothesis buys. Four questions that practice settles by folklore become decidable, whether a window certifies a token boundary, whether the boundary gap is bounded, whether a mask keeps generation completable, and whether two policies are equivalent. The last three specialize limitedness, coaccessibility and the equivalence of single-valued transducers. The first refines the synchronizing cut of code theory, and two more ask whether a certified cut, or the cuts of an inventory, let each piece scan alone. The class sits strictly between the online-scannable policies and those with regular domains, and i
    
[^267]: norregel：挪威法律中带日期数值的预注册基准

    norregel: a preregistered benchmark of dated numeric values in Norwegian law

    [https://arxiv.org/abs/2610.29207](https://arxiv.org/abs/2610.29207)

    该论文提出了norregel——一个预注册的基准测试，用于评估语言模型对挪威法律中随时间变化的数值知识的掌握，结果显示即使是表现最好的模型在主要条目上的正确率也不到一半，揭示了语言模型在法律数值知识方面的重大缺陷。

    

    norregel要求语言模型回答挪威法律、法规和议会决议中的数值（挪威克朗金额、百分比、国民保险基本金额（G）的倍数以及年龄限制），每个数值都对应一个特定日期。916个条目从Lovdata的开放数据中机械提取，每个条目都带有其条款的带日期历史记录，因此答案可以被分类为正确、过时、错误或无法解释。我们报告了在相同条目上进行的第一次和第二次预注册运行。在第一次运行中（五个模型，仅在当地分析完成后才公开列出），假设H1（即对于在模型截止日期之后生效的数值，过时答案更频繁）实际上无法测试，H2和H3的结果也不确定；Opus 5.5和Sonnet 4.6在880个主要条目中分别正确回答了46.7%和40.2%。第二次运行在任何调用之前就完成了预注册并发布，分析了八个模型。其汇总的主要测试……（摘要在此处截断）

    arXiv:2610.29207v1 Announce Type: new  Abstract: norregel asks language models for numeric values in Norwegian statutes, regulations and parliamentary decisions (NOK amounts, percentages, multiples of the National Insurance basic amount (G) and age limits), each at a stated date. The 916 items were extracted mechanically from Lovdata's open data, and each carries the dated history of its provision, so an answer can be classed as correct, outdated, wrong or uninterpretable. We report a first and a second preregistered run on the same items. In the first run (five models, listed publicly only after local analysis), H1, that outdated answers are more frequent for values that took effect after a model's cutoff, could not be tested in substance, and H2 and H3 were inconclusive; Opus 5.5 and Sonnet 4.6 answered 46.7% and 40.2% of the 880 primary items correctly. The second run was preregistered and published before any of its calls, and eight models were analysed. Its pooled primary test sup
    
[^268]: MAS-Guard：从单个智能体到多智能体大语言模型系统的统一防护框架

    MAS-Guard: Unified Guardrails from Individual Agents to Multi-Agent LLM Systems

    [https://arxiv.org/abs/2610.28397](https://arxiv.org/abs/2610.28397)

    该论文提出统一安全防护框架MAS-Guard，通过本地智能体防护层（A-Guard）等机制，为包含可插桩与不透明智能体的单智能体和多智能体LLM系统，提供覆盖本地执行与系统级交互的一体化安全保障。

    

    智能体AI将大语言模型（LLM）从生成响应扩展到规划与执行复杂工作流，在大幅提升其效用的同时，也使应用面临不安全行为、信息泄露和对抗性操纵等风险。多智能体系统通过专业化分工与协作进一步扩展了这些能力，但权限分散、角色异构以及参与者不透明或可能已被攻陷等因素，使得安全防护变得更具挑战性。现有防御手段虽能应对重要的单一威胁，但在本地执行、系统级交互和执行机制方面的覆盖是碎片化的。我们提出MAS-Guard，这是一个统一框架，为包含可插桩智能体和不透明智能体的单智能体及多智能体AI应用提供安全保障，填补了这一空白。在可插桩智能体中，所提出的本地智能体防护层（A-Guard）应用任务特定的专家来评估……

    arXiv:2610.28397v1 Announce Type: cross  Abstract: Agentic AI extends large language models (LLMs) from generating responses to planning and executing complex workflows, substantially expanding their utility while exposing applications to unsafe actions, information leakage, and adversarial manipulation. Multi-agent systems further expand these capabilities through specialization and collaboration, but distributed authority, heterogeneous roles, and opaque or potentially compromised participants make safeguarding even more challenging. Existing defenses address important individual threats but provide fragmented coverage across local execution, system-level interactions, and enforcement mechanisms. We introduce MAS-Guard, a unified framework that closes this gap for safeguarding single- and multi-agent AI applications containing both instrumentable and opaque agents. Within instrumentable agents, the proposed local Agent Guardrail (A-Guard) layer applies task-specific experts to assess
    
[^269]: 语言模型学会改进语言模型

    Language Models Learn to Improve Language Models

    [https://arxiv.org/abs/2610.23344](https://arxiv.org/abs/2610.23344)

    该论文提出LM2LM方法，通过强化学习训练大语言模型生成有效的语言模型改进提案，并利用廉价的代理反馈解决真实训练反馈成本高昂以及分布偏移下反馈不可靠这两大难题，为递归自我改进铺路。

    

    大语言模型（LLM）正被越来越多地用于自动化AI研究，包括语言模型（LM）本身的开发，这为递归自我改进提供了一条潜在路径。然而，将实验反馈转化为一种可学习的、能够跨模型提出有效改进的能力面临两大挑战。首先，使用真实反馈来训练这种能力代价高昂：强化学习（RL）需要评估数以万计的改进提案，而每个提案可能都需要一次完整的模型训练运行。即使每次运行仅需十个GPU小时，仅奖励收集这一项就会消耗数十万GPU小时，这促使人们寻求更便宜的替代反馈。其次，当提案分布发生偏移时，替代反馈可能变得不可靠，因为策略优化会偏向那些被高估的提案。为应对这些挑战，我们提出了LM2LM，该方法通过强化学习训练LLM来生成有效的LM改进提案。为了提供可扩展的反馈，我们使用了一种……（摘要在此处截断）

    arXiv:2610.23344v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to automate AI research, including language-model (LM) development, offering a potential path toward recursive self-improvement. However, turning experimental feedback into a learned ability to propose effective improvements across models faces two challenges. First, training such an ability with real feedback is expensive: reinforcement learning (RL) requires evaluating tens of thousands of proposals, each potentially requiring a full model-training run. Even at ten GPU-hours per run, reward collection alone would consume hundreds of thousands of GPU-hours, motivating a cheaper surrogate. Second, surrogate feedback can become unreliable as policy optimization favors overestimated proposals when the proposal distribution shifts. To address these challenges, we propose LM2LM, which trains LLMs with RL to generate effective LM improvement proposals. To provide scalable feedback, we use a
    
[^270]: 审计医疗领域大语言模型智能体的升级转诊可靠性：一个扰动控制的工作流状态失效基准

    Auditing Escalation Reliability in Healthcare LLM Agents: A Perturbation-Controlled Benchmark of Workflow-State Failures

    [https://arxiv.org/abs/2610.21270](https://arxiv.org/abs/2610.21270)

    该论文构建了一个包含400个合成医疗工作流片段的扰动控制基准，用于审计医疗LLM智能体在证据缺失、过时或冲突时的“升级转诊”可靠性，发现尽管干净决策准确率高达0.97，模型仍会错过32%--97%的必要升级，且更大的模型只能缩小而非消除该差距。

    

    可信的生物医学AI不仅需要在静态输入上给出准确答案，更需要在证据缺失、过时、冲突或延迟的情况下做出可靠的动作选择。我们将一个片段级评估协议实例化为400个合成医疗工作流片段：涵盖四个任务类别的40个人工编写的语义种子、两种表面参数化方式以及五个受控变体，在预先声明的5:1漏升级与不必要升级代价下，对“回答/请求澄清/弃权/升级”四种动作进行评分。参考动作来自单一作者的研究策略标注，而非临床医生裁定的真值。在五个开放权重模型配置（1.7B--9.7B）上，模型在干净样本上的决策准确率达到0.97，然而在不确定性提示下，模型会错过32%--97%的必要升级。更强的9.7B配置缩小了这一差距但并未消除：在冲突条件下其准确率从0.97降至0.78。模型言语化表达的置信度并不可靠（ECE为0.10--0.54；失效检测AUROC……（摘要在此处截断）

    arXiv:2610.21270v1 Announce Type: new  Abstract: Trustworthy biomedical AI requires reliable action selection when evidence is missing, stale, conflicting, or late, not only accurate answers on static inputs. We instantiate an episode-level evaluation protocol as 400 synthetic healthcare workflow episodes: 40 hand-authored semantic seeds across four task classes, two surface parameterizations, and five controlled variants, scored on ANSWER/ASK CLARIFICATION/ABSTAIN/ESCALATE under a pre-declared 5:1 missed-versus-unnecessary escalation cost. Reference actions are single-author study-policy labels, not clinician-adjudicated ground truth. Across five open-weight configurations (1.7B--9.7B), clean decision accuracy reaches 0.97, yet uncertainty-prompted models miss 32--97% of required escalations. The stronger 9.7B configuration narrows but does not remove the gap, falling from 0.97 clean accuracy to 0.78 under conflict. Verbalized confidence is inconsistent (ECE 0.10--0.54; failure AUROC 
    
[^271]: SwarmRSI：通过演化的科学智能体群体实现递归自我改进

    SwarmRSI: Recursive Self-Improvement through Evolving Scientific Agent Swarm

    [https://arxiv.org/abs/2610.20431](https://arxiv.org/abs/2610.20431)

    提出了 SwarmRSI 框架，通过反馈引导的群体变异和互补群体重组两种机制演化科学智能体群体，解决假设与实现修改不明确以及有用组件被隐藏的问题，实现科学研究中的递归自我改进。

    

    科学研究中的递归自我改进（RSI）旨在通过智能体自身的实验经验来提升其研究能力。科学智能体群体可以整合来自不同研究方向的证据和实现方案，以指导智能体的后续实验。然而，糟糕的实验结果会导致不清楚究竟应该修改假设还是修改其实现方案，而整体程序评分可能会掩盖对其他智能体有用的组件。为了应对这些挑战，我们提出了 SwarmRSI，一个通过演化科学智能体群体来实现递归自我改进的框架。我们设计了反馈引导的群体变异，利用来自其他智能体的实验反馈来完善假设及其实现方案。我们还提出了互补群体重组，以在智能体之间重用保留的假设和互补的程序组件，使部分结果能够支持新的研究。递归群体演化……

    arXiv:2610.20431v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) in scientific research aims to improve agents' research capabilities through their own experimental experience. Scientific agent swarms can combine evidence and implementations from different research directions to guide their agents' subsequent experiments. However, poor experimental results can leave unclear whether to revise a hypothesis or its implementation, while overall program scores can hide components useful to other agents. To address these challenges, we introduce SwarmRSI, a framework that evolves scientific agent swarms for recursive self-improvement. We devise Feedback-Guided Swarm Mutation to refine hypotheses and their implementations using experimental feedback from other agents. We also propose Complementary Swarm Recombination to reuse retained hypotheses and complementary program components across agents, allowing partial results to support new research. Recursive Swarm Evolution in
    
[^272]: 共情不是事实性与可靠性下降的替罪羊

    Empathy Is Not the Scapegoat for Declines in Factuality and Reliability

    [https://arxiv.org/abs/2610.18541](https://arxiv.org/abs/2610.18541)

    本研究证明共情并非模型事实性与可靠性下降的根源——温暖效应因任务和模型而异，通过任务对齐的自适应语气可以在提升共情的同时保持甚至提高针对性可靠性，并提出了经过人工验证的共情评估器。

    

    语言模型越来越被期望在用户处于压力或不确定情境时提供支持，此时温暖与共情是人们所期待的。虽然近期研究将更温暖的模型与可靠性下降联系起来，但我们表明这一结论并不完整。以往研究通常主要在共情对话上进行训练，导致难以将语气效应与SFT引起的分布偏移区分开来。我们在模型层面和评估层面重新审视了这一问题。我们研究了通过提示、SFT和RL来诱导温暖行为，并使用包含11个数据集的基准测试套件以及情感情境下的EMOFACT来全面评估能力保持情况。在多个模型家族上的实验表明，温暖效应因任务和模型而异，而任务对齐的自适应语气可以提升共情能力和针对性的可靠性，尽管仍存在一些能力上的权衡。我们进一步引入了经过人工验证的共情评估器，专门用于评估模型的共情表现。

    arXiv:2610.18541v2 Announce Type: replace  Abstract: Language models are increasingly expected to support users in stressful or uncertain situations, where warmth and empathy are desirable. While recent work associates warmer models with reduced reliability, we show that this conclusion is incomplete. Prior studies often train primarily on empathetic dialogue, making it difficult to separate tone effects from SFT-induced distribution shift. We revisit this issue at both the model and evaluation levels. We study prompting, SFT, and RL for inducing warm behavior, and comprehensively evaluate capability preservation using an 11-dataset benchmark suite and EMOFACT under emotional contexts. Across multiple model families, we find that warmth effects are task- and model-dependent, while task-aligned adaptive tone can improve empathy and targeted reliability, although some capability trade-offs remain. We further introduce human-validated empathy evaluators specifically developed to assess em
    
[^273]: 失败在于读取方式：细粒度情绪识别基准测量的是答案引出方式，而非感知能力

    The Failure Is in the Readout: Fine-Grained Emotion Recognition Benchmarks Measure Elicitation, Not Perception

    [https://arxiv.org/abs/2610.08162](https://arxiv.org/abs/2610.08162)

    该研究发现，现成的视觉语言模型在细粒度情绪识别上其实并不逊于专门微调的模型，此前基准测得的失败源于生成式的答案引出方式，而非模型的感知能力不足。

    

    细粒度情绪识别可以支撑治疗工具和社交机器人，但它需要面部数据，这带来了隐私和数据保护方面的顾虑。EmoNet-Face-HQ 通过生成的人像解决了这一问题，并由专家按照远比常见的六到八种基本情绪更精细的 40 类分类体系进行评分。在其自带的评测协议下，视觉语言模型（VLM）在该分类体系上得分很差，该基准据此得出结论：需要一个专门微调的模型，即 Empathic-Insight-Face（EIF；Small/Large）。我们证明，当答案不是通过文本生成、而是作为每个类别一次二值查询直接从 logits 中读取时，现成的 VLM 就能追平甚至超越该微调模型。我们保留了该基准的图像、分类体系和专家评分，只改变了答案的读取方式。在其测量最可靠的五个类别上，专家间一致性达到 κ_w = 0.468。在生成式评测下，十一个开源权重 VLM 中没有任何一个的置信区间完全高于该锚点（……摘要原文在此截断）

    arXiv:2610.08162v1 Announce Type: cross  Abstract: Fine-grained emotion recognition supports therapy tools and social robots, but it needs facial data, which raises privacy and data-protection concerns. EmoNet-Face-HQ answers that with generated portraits, expert-rated over a $40$-category taxonomy far finer than the usual six to eight basic emotions. Under the protocol it ships with, vision-language models (VLMs) score poorly on that taxonomy, and the benchmark concludes that a dedicated fine-tuned model is necessary: Empathic-Insight-Face (EIF; Small/Large). We show that off-the-shelf VLMs match or beat that fine-tuned model when the answer is not generated but read from the logits, as one binary query per category. We keep the benchmark's images, taxonomy and ratings, and change only how the answer is read. Experts agree at $\kappa_w = 0.468$ on the five categories they measure most reliably. Generatively, no interval among eleven open-weight VLMs lies entirely above that anchor ($\
    
[^274]: OpenJev-RLCD：一个可运行的RLCD实现

    OpenJev-RLCD: A Working RLCD Implementation

    [https://arxiv.org/abs/2609.38850](https://arxiv.org/abs/2609.38850)

    本文实现了面向推理模型的RLCD（校准决策强化学习），用严格正则评分规则对采样推理依据后的答案分布评分，证明RLVR是缺失多样性项的混合目标，并提出“先校准后强化”的两阶段训练方案，使模型既保持推理能力又获得良好校准。

    

    诸如Jev这类决策模型以概率形式回答问题，而这些概率只有经过校准才有用。开源复现通常依赖监督微调加温度缩放，而基于可验证奖励的强化学习会使推理模型过度自信。我们提出了一个面向推理模型的、可运行的“校准决策强化学习”实现：模型采样一条推理依据，随后我们用严格正则的评分规则对其给出的答案分布进行评分。一个方差恒等式表明，对多个样本的混合分布进行评分会奖励相互不一致的推理依据，而RLVR恰好是去掉了多样性项的该混合目标。若朴素地直接优化，逐条推理依据的目标要么使模型关闭推理，要么被策略梯度噪声淹没，由此引出两阶段方案：先校准，再强化。在两个推理任务上使用Qwen3-1.7B（3个随机种子）进行实验（摘要截断）。

    arXiv:2609.38850v1 Announce Type: new  Abstract: Decision models such as Jev answer questions with probabilities, which are only useful if they are calibrated. Open-source reproductions rely on supervised fine-tuning plus temperature scaling, while reinforcement learning from verifiable rewards (RLVR) makes reasoning models overconfident. We present a working implementation of reinforcement learning for calibrated decisions (RLCD) for reasoning models: the model samples a rationale, and we score the answer distribution it commits to afterwards with a strictly proper scoring rule. A variance identity shows that scoring the mixture of several samples rewards disagreeing rationales, and that RLVR is exactly this mixture objective without its diversity term. Optimized naively, the per-rationale objective either switches reasoning off or is drowned out by policy-gradient noise, which leads to a two-stage recipe: calibrate, then reinforce. With Qwen3-1.7B on two reasoning tasks (3 seeds, pai
    
[^275]: 一种用于评估大语言模型回答中所表达的临床推理的评分量规（提议稿）

    A Proposed Rubric for Evaluating Expressed Clinical Reasoning in Large Language Model Responses

    [https://arxiv.org/abs/2609.37788](https://arxiv.org/abs/2609.37788)

    该论文提出一个融合医学教育评估框架、临床大语言模型基准和通用LLM推理评估研究的多维评分量规，用于对大语言模型针对金标准临床案例的自由文本回答中表达的临床推理进行结构化评估。

    

    评分量规为语言模型的结构化评估提供支持。我们提出了一种用于评估模型回答中所表达的临床推理的量规，该量规借鉴了三个方面的工作：医学教育评估框架（ART、SCT、关键特征问题以及OSCE）；临床大语言模型基准（MedR-Bench、HealthBench、TIMER-Bench、DR.BENCH、PrIME-LLM 和 PatientSafeBench）；以及通用大语言模型推理评估研究，包括事实性-有效性-连贯性-有用性（Factuality-Validity-Coherence-Utility）分类法、FaithCoT-Bench 和 C2-Faith。我们使用“有据性”作为对该分类法中事实性类别的面向临床的改编。该量规将这些概念整合到一个多维框架中，用于对针对金标准临床案例片段的自由文本回答进行评分。它包括初步的行为锚定标准、适用性规则，以及一个用于标记案例特定安全关键错误的独立标记。通用领域的框架为其设计提供了参考，但并未被视为经过验证的临……（原文摘要在此处截断）

    arXiv:2609.37788v1 Announce Type: cross  Abstract: Rubrics support the structured evaluation of language models. We propose a rubric for assessing expressed clinical reasoning in model responses, drawing on three bodies of work: medical education assessment frameworks (ART, SCT, Key Feature Problems and OSCE); clinical LLM benchmarks (MedR-Bench, HealthBench, TIMER-Bench, DR.BENCH, PrIME-LLM and PatientSafeBench); and general LLM reasoning evaluation research, including the Factuality-Validity-Coherence-Utility taxonomy, FaithCoT-Bench and C2-Faith. We use groundedness as a clinically oriented adaptation of the taxonomy's factuality category. The rubric brings these concepts together in a multidimensional framework for scoring free-text responses to gold-standard clinical vignettes. It includes provisional behavioural anchors, applicability rules and a separate flag for case-specific safety-critical errors. General-domain frameworks inform its design but are not treated as validated cl
    
[^276]: AgentHop：面向科学引文图谱上智能体多跳问答的诊断性基准

    AgentHop: A Diagnostic Benchmark for Agentic Multi-Hop Question Answering over Scientific Citation Graphs

    [https://arxiv.org/abs/2609.34428](https://arxiv.org/abs/2609.34428)

    AgentHop是一个诊断性基准，通过在受控七工具沙箱中沿检索、综合、工具调用和资源管理四个维度剖析智能体在科学引文图谱上的多跳问答表现，精准定位大语言模型智能体的失败原因。

    

    智能体任务要求大型语言模型与世界进行交互，在资源受限的条件下跨多个步骤导航信息并收集证据。由于这种复杂性，智能体任务的失败可能源于多种原因，而精确定位这些失败原因对于诊断和改进智能体系统至关重要。然而，现有的基准测试往往只关注单一的排行榜分数，使底层的失败模式难以被理解。为填补这一空白，我们提出了AgentHop，一个包含1,011道多选题的诊断性基准，并配套一个受控的七工具沙箱，在固定的token数量、对话轮次和工具调用次数约束下运行。AgentHop通过沿智能体运行的四个维度——检索、综合、工具调用和资源管理——剖析单一的准确率分数，从而揭示模型的薄弱环节。在对19个模型的评估中，我们发现模型行为按模型家族聚类，工具调用特征展现出独特的家族指纹。

    arXiv:2609.34428v2 Announce Type: replace  Abstract: Agentic tasks require a large language model to interact with the world, navigating information and gathering evidence across multiple steps with restricted resources. Due to this complexity, agentic task failures arise from various sources, and pinpointing these failure causes is essential to diagnose and improve agentic systems. Existing benchmarks, however, tend to focus on a single leaderboard score, leaving the underlying failure modes opaque. To fill this gap, we introduce AgentHop, a diagnostic benchmark of 1,011 multiple-choice questions paired with a controlled seven-tool sandbox under fixed token, turn, and tool-call constraints. AgentHop reveals model vulnerabilities by dissecting a single accuracy score along four axes of agent operation: retrieval, synthesis, tool-call, and resource management. Across 19 models, we find that behavior clusters by model family, with tool-call signatures revealing distinct family fingerprin
    
[^277]: 用于科学决策的Jev：评估语义选择及其后果

    Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences

    [https://arxiv.org/abs/2609.24965](https://arxiv.org/abs/2609.24965)

    该研究将Jev作为科学工作流中的语义决策组件进行评估，发现其语义正确性与其他配置持平且延迟最低，并表明错误的语义选择会改变下游计数但可能不影响最终结论标签。

    

    arXiv:2609.24965v1 公告类型：新论文 摘要：科学工作流程通常需要在确定性计算进行之前，在已知关系之间做出选择。观测数据是否共享相同的文化、处理方式或参考标准，可能会改变由此产生的计数或比较的科学含义。我们使用一个遵循其文档指导并将算术运算分配给代码的测试框架，将Jev作为语义决策组件进行评估。该研究在十个科学案例的二十个有来源依据的选择上比较了十二种模型配置，每种配置重复五次。我们分别测量语义选择、下游输出和最终结论标签。Jev与其他五种配置在完全语义正确性上持平，并在成功响应中实现了观察到的最低中位延迟。在三个对比模型中，对一个文化历史问题的七次错误选择改变了下游计数，同时保持了正确的最终标签。这些结果确定了一个有用的角色

    arXiv:2609.24965v1 Announce Type: new  Abstract: Scientific workflows often require choosing among known relations before a deterministic calculation can proceed. Whether observations share a culture, treatment or reference standard can change the scientific meaning of the resulting count or comparison. We evaluate Jev as a semantic decision component using a harness that follows its documented guidance and assigns arithmetic to code. The study compares twelve model configurations on twenty source-grounded Choices across ten scientific cases, each repeated five times. We measure semantic selections, downstream outputs and final claim labels separately. Jev matched five other configurations at complete semantic correctness and achieved the lowest observed median latency among successful responses. Across three comparison models, seven wrong selections on one culture-history question changed downstream counts while preserving the correct final label. These results identify a useful role 
    
[^278]: 代理总成本：多智能体LLM工作流中记忆注入成本的精确归因

    Total Cost of Agency: Exact Attribution of Memory Injection Cost in Multi-Agent LLM Workflows

    [https://arxiv.org/abs/2609.23790](https://arxiv.org/abs/2609.23790)

    提出代理总成本（TCA）框架，将多智能体LLM工作流成本分解为基础提示词、推理、记忆注入等组件，并通过两遍式不计费令牌计数方法实现记忆注入成本的精确归因，发现记忆注入在企业基准测试中占可变成本的13.6%。

    

    在多智能体大语言模型（LLM）工作流中，每个节点都会从记忆中检索上下文并将其注入到自身的提示词中，这些被注入的令牌会以与系统提示词和用户查询相同的单价作为输入令牌计费。生产环境的可观测性工具只报告总令牌成本，但不会将节点自身生成的令牌与传递给它的令牌区分开来，因此这一成本组成部分对付费团队来说是不可见的。我们提出了代理总成本，这是一种将多智能体工作流成本分解为基础提示词、推理、记忆注入、未命中惩罚和上下文累积等组成部分的方法，并提出了一种精确归因方法：一种两遍式的、不计费的令牌计数方法，它直接测量被注入的令牌，而不是通过字数代理来估算。在一个针对真实模型API执行的200个任务的企业基准测试中，记忆注入占可变成本的13.6%，一种编译时优化……（原文摘要在此处截断）

    arXiv:2609.23790v1 Announce Type: new  Abstract: Every node in a multi-agent large language model (LLM) workflow retrieves context from memory and injects it into its prompt, where those injected tokens are billed as input tokens at the same per-token price as the system prompt and the user query. Production observability tools report total token cost but do not separate the tokens a node generates from the tokens it is handed, so this component of the bill is invisible to the teams paying it. We introduce the Total Cost of Agency (TCA), a decomposition of multi-agent workflow cost into base prompt, inference, memory injection, miss penalty and context-accumulation components, and an exact attribution method: a two-pass, non-billable token count that measures injected tokens directly rather than estimating them from word-count proxies. On a 200-task enterprise benchmark executed against real model APIs, memory injection accounts for 13.6 percent of the variable cost a compile-time opti
    
[^279]: 面向台湾历史问答的静态与智能体式RAG桥接研究

    Bridging Static and Agentic RAG for Taiwanese Historical Question Answering

    [https://arxiv.org/abs/2609.23056](https://arxiv.org/abs/2609.23056)

    该论文针对台湾历史问答发现智能体式与静态RAG总体性能相近但在70.83%的问题上表现各异，并提出一种比较两个回答及其引用证据的事后选择器，显著优于任一单一流水线并恢复了oracle提升空间的60.34%。

    

    智能体检索增强生成（RAG）使语言模型能够根据先前检索到的证据自适应地调整检索策略，但这种自适应编排是否始终优于设计良好的静态流水线仍不明确。我们针对台湾历史问答任务，在共享相同生成器和混合检索后端的条件下，对智能体式与静态RAG进行了受控比较。尽管两者总体性能相近，但两条流水线在70.83%的问题上表现不同，其各自优势在取平均时大体相互抵消。一个能为每个问题选出更优回答的理想选择器（oracle）比表现较好的单一流水线可将综合分数提高0.2417，揭示了问题级选择存在巨大提升空间。因此，我们引入了一种事后选择器，通过比较两个回答及其引用的证据进行选择，其性能显著优于任一单一流水线，并恢复了oracle提升空间的60.34%。

    arXiv:2609.23056v1 Announce Type: new  Abstract: Agentic retrieval-augmented generation (RAG) enables language models to adapt retrieval based on previously retrieved evidence, but it remains unclear whether such adaptive orchestration consistently outperforms well-designed static pipelines. We conduct a controlled comparison of agentic and static RAG for Taiwanese historical question answering, sharing the same generator and hybrid retrieval backend. Despite similar aggregate performance, the two pipelines differ on 70.83% of questions, with their advantages largely canceling out when averaged. An oracle that selects the better response per question improves the composite score by 0.2417 over the better individual pipeline, revealing substantial headroom for question-level selection. We therefore introduce a post-hoc selector that compares the two responses and their cited evidence, significantly outperforming either individual pipeline and recovering 60.34% of the oracle headroom. Th
    
[^280]: EvalMem：面向长期记忆系统的操作级诊断框架

    EvalMem: An Operation-Level Diagnostic Framework for Long-Term Memory Systems

    [https://arxiv.org/abs/2609.22231](https://arxiv.org/abs/2609.22231)

    EvalMem通过编码、检索、生成三个并行检查器将长期记忆系统的错误精确定位到具体操作环节，并借助召回优先的智能体RAG策略将证据召回率从70.2%提升至95.6%。

    

    与基于大语言模型（LLM）的助手进行长期交互，需要能够保存并更新用户状态、偏好和交互历史的记忆系统。现有评估仅报告端到端问答准确率，无法判断错误究竟源自编码、检索还是生成环节。我们提出EvalMem，一个包含三个并行检查器（Examiner）的操作级诊断框架。对于每个查询，编码检查器检验目标事实是否已被存储，检索检查器评估原生检索器是否返回可用证据，生成检查器测试模型能否基于给定（oracle）证据作答。三者的输出构成细粒度的多标签缺陷代码。为改进存储层面的诊断，我们将智能体式RAG改造为“召回优先”策略，同时利用查询和源证据进行搜索，将LoCoMo数据集上已有证据的召回率从70.2%提升至95.6%。我们在LoCoMo、LongMe（原文在此截断）等数据集上对七个记忆系统进行了评估。

    arXiv:2609.22231v1 Announce Type: new  Abstract: Long-horizon interactions with LLM-based assistants require memory systems that preserve and update user states, preferences, and interaction histories. Existing evaluations report end-to-end QA accuracy and cannot determine whether errors arise from encoding, retrieval, or generation. We introduce EvalMem, an operation-level diagnostic framework with three parallel Examiners. For each query, the Encoding Examiner checks whether the target fact is stored, the Retrieval Examiner assesses whether the native retriever returns usable evidence, and the Generation Examiner tests whether the model can answer from oracle evidence. Their outputs form fine-grained multi-label defect codes. To improve store-level diagnosis, we adapt agentic RAG with a recall-first strategy that searches using both the query and source evidence, increasing recall of present evidence on LoCoMo from 70.2% to 95.6%. Evaluations of seven memory systems on LoCoMo, LongMe
    
[^281]: 利用OCR注意力头将图像语义语言化

    Using OCR Heads to Verbalize Image Semantics

    [https://arxiv.org/abs/2609.18823](https://arxiv.org/abs/2609.18823)

    研究发现视觉语言模型中负责OCR的注意力头实际上是通用语义特征头，将其注意力权重压缩为语言化透镜变换后，可从模型所有层（甚至第0层）的隐藏状态中解读出可解释的图像语义标签，证明图像表示在早期层就与语言对齐。

    

    视觉语言模型（VLM）如何从像素映射到语义？为了理解这个普遍性问题，我们聚焦于一个较窄的问题：研究VLM如何执行光学字符识别（OCR）。我们在四个模型中识别出对OCR具有因果必要性的注意力头，并发现这些实际上是通用注意力头，能够在所有图像token上输出可解释的语义特征。例如，将这些注意力头指向包含单词“bike”（自行车）的图像token，会使Qwen3-VL-8B输出“bike”；而将它们指向鸟的翅膀，则会使模型输出token“feathers”（羽毛）。我们将这些注意力头的注意力权重压缩为一个单一的语言化透镜变换，该变换能够揭示所有层隐藏状态中可解释的语义特征。当与投影到词表空间相结合时，我们可以从第0层开始就获得可解释的标签，这表明图像表示实际上在早期层就与语言对齐了。我们发现……

    arXiv:2609.18823v1 Announce Type: cross  Abstract: How do VLMs map from pixels to semantics? To understand this general question, we focus on a narrow one: studying how VLMs perform optical character recognition (OCR). Across four models, we identify attention heads causally necessary for OCR, and discover that these are in fact general-purpose heads that output interpretable semantic features across all image tokens. For example, pointing these heads at an image token containing the word "bike" causes Qwen3-VL-8B to output "bike," but pointing them at a bird wing causes the model to output the token "feathers." We collapse these heads' attention weights into a single verbalization lens transformation that reveals interpretable semantic features in hidden states across all layers. When combined with projection to vocabulary space, we can obtain interpretable labels starting from layer 0, showing that image representations are in fact aligned with language in early layers. We find that 
    
[^282]: EvoOntology：面向数据智能体的自进化本体层

    EvoOntology: A Self-Evolving Ontology Layer for Data Agents

    [https://arxiv.org/abs/2609.15779](https://arxiv.org/abs/2609.15779)

    本文提出EvoOntology，一种面向数据智能体的自进化本体层，通过将本体封装为包含模式层、内容层和工具层的MCP服务器，并借助构建器智能体实现自主构建与自进化循环，有效弥合了智能体与异构数据之间的鸿沟。

    

    数据智能体旨在基于自然语言指令处理异构数据，包括表格、文件和数据库。然而，数据智能体面临一个具有挑战性的“智能体-数据鸿沟”：异构数据存在于智能体之外，而智能体只能通过通用工具（例如列名和文件路径）访问这些数据。现有方法要么让智能体直接探索原始数据源，要么将手动构建的语义层注入提示中。然而，这两种方法都难以很好地扩展到大规模异构数据源，也无法适应不同的智能体行为。在本文中，我们提出了EvoOntology，一种面向数据智能体的自进化本体层。EvoOntology将本体封装为一个MCP服务器，其中包含模式层、内容层和工具层，使智能体能够在运行时主动查询本体并与之交互。为此，我们引入了一个用于自主本体构建的构建器智能体和一个自进化循环……

    arXiv:2609.15779v1 Announce Type: new  Abstract: Data agents aim to fulfill natural-language instructions over heterogeneous data, including tables, files, and databases. However, data agents face a challenging agent-data gap: heterogeneous data resides outside the agent, while the agent can access it (e.g., column names and file paths) only through generic tools. Existing approaches either let agents directly explore raw data sources or inject manually constructed semantic layers into prompts. However, neither scales well to large heterogeneous data sources nor adapts to different agent behaviors. In this paper, we introduce EvoOntology, a self-evolving ontology layer for data agents. EvoOntology encapsulates the ontology as an MCP server comprising a schema layer, a content layer, and a tool layer, enabling agents to actively query and interact with the ontology at runtime. To this end, we introduce a builder agent for autonomous ontology construction and a self-evolution loop that c
    
[^283]: RAP：研究关注度预测揭示目标条件化的证据获取偏差

    RAP: Research Attention Prediction Reveals Target-Conditioned Evidence Acquisition Biases

    [https://arxiv.org/abs/2609.10092](https://arxiv.org/abs/2609.10092)

    提出了RAP滚动基准来评估LLM智能体预测研究关注度变化的能力，发现其表现不如简单的EWMA精确计数基线，并揭示了状态前推优于直接预测、以及面向预测的策略倾向于检索较旧证据这两大瓶颈。

    

    大型语言模型越来越多地充当研究智能体，但由于论文评审和研究想法缺乏唯一可验证的结果，其追踪研究关注度变化的能力难以评估。我们提出了研究关注度预测（RAP），这是一个涵盖278个AI/ML领域、共1,390个回合的滚动基准。在每个时间截点，LLM智能体在时间受限的arXiv语料库中进行搜索，并预测未来六个月内八个固定研究方向上的论文份额。搜索通常有所帮助，但在组合准确性方面，所有四个诊断模型的表现均不如基于精确计数的指数加权移动平均（EWMA）基线。我们识别出两个相互关联的瓶颈：在可访问累积历史的条件下，状态前推在所有四个诊断模型上都优于直接预测；冻结证据回放实验将这种逆转的一个共同成分与面向预测的策略检索到的近期证据份额较小联系起来。

    arXiv:2609.10092v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly act as research agents, yet their ability to track shifts in research attention is difficult to evaluate because reviews and research ideas lack uniquely verifiable outcomes. We introduce Research Attention Prediction (RAP), a rolling benchmark covering 278 AI/ML fields and 1,390 episodes. At each cut-off, an LLM agent searches a temporally restricted arXiv corpus and predicts the next six months' paper shares across eight frozen research directions. Search generally helps, but all four diagnostic models perform worse than an exact-count exponentially weighted moving average (EWMA) baseline in compositional accuracy. We identify two linked bottlenecks. Under cumulative-history access, State carry-forward outperforms direct Forecast for all four diagnostic models; frozen-evidence replay links a shared component of this reversal to Forecast-oriented policies retrieving a smaller share of recent e
    
[^284]: trajectory-judge：仅基于结果的LLM评判器在智能体轨迹上遗漏了什么

    trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories

    [https://arxiv.org/abs/2609.00038](https://arxiv.org/abs/2609.00038)

    仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。

    

    仅基于结果的评估是LLM智能体在生产环境中的默认做法：向评判器展示用户请求和最终回复，询问其处理是否得当。这一指标在结构上无法察觉那些“以错误方式得到正确答案”的智能体。我们在真值可以通过构造获知的场景下测量这一盲区：一个确定性的使用工具的客服支持台环境、一个总能解决问题的脚本化oracle策略，以及一个在已知步骤恰好破坏一个环节的故障注入器，并根据用户可见结果是否仍然保持（静默型故障）与否（显性型故障）对故障进行分层。五种评判器（程序化规则、仅结果型、两种模型规模的逐步评分标准型、以及自一致性集成）在400条轨迹上按照检测能力、步骤定位、故障类型判定、校准度和成本进行评分。结果显示：仅结果型评判器能捕获84%的显性故障，但只能捕获45%的静默故障，同时还会误报33%的正确轨迹；而逐步评分标准型评判器对静默故障的召回率达到77%。

    arXiv:2609.00038v1 Announce Type: cross  Abstract: Outcome-only evaluation is the production default for LLM agents: show a judge the request and the final reply and ask whether it was handled well. The metric is structurally blind to an agent that reaches the right answer the wrong way. We measure that blind spot where ground truth is known by construction: a deterministic tool-using support-desk environment, a scripted oracle policy that always solves it, and a fault injector that breaks exactly one thing at a known step, stratifying faults by whether the customer-visible outcome survived (silent) or not (loud). Five judges (programmatic rules, outcome-only, step-rubric at two model sizes, and a self-consistency ensemble) are scored on detection, step localisation, fault typing, calibration, and cost over 400 trajectories. The outcome-only judge catches 84% of loud faults but 45% of silent ones while flagging 33% of correct trajectories; a step-rubric judge reaches 77% silent recall 
    
[^285]: 项目均值代理：为什么更丰富的人格数据无法提升大语言模型作为人类代理的效果

    Item-Mean Surrogates: Why Richer Persona Data Fail to Improve LLMs as Human Surrogates

    [https://arxiv.org/abs/2608.29455](https://arxiv.org/abs/2608.29455)

    LLM作为人类代理的良好表现主要源于其对项目均值的预测，一旦去除项目均值，其仅能解释3.05%的个体特异性变异，而更丰富的人格数据、模型变体和微调均无法弥补这一差距。

    

    大语言模型（LLM）越来越多地被用作人类代理，这通常基于一个前提：更丰富的人格数据可以使它们成为特定个体的替代品或探索性工具。我们在四个数据集上检验了这一前提，这些数据集涵盖超过40万名参与者和超过6000个调查项目及实验结果。LLM在总体层面表现良好：它们的平均回答与人类对相同项目的平均回答高度一致。但这种成功在很大程度上只是反映了模型对每个项目人类平均回答的预测。一旦去除每个项目的人类均值，LLM的预测仅能解释剩余的受访者特异性变异的3.05%，远低于53.6%的人类重测基准。更丰富的人格数据、模型变体以及微调都无法缩小这一差距。在方差分析中，一旦去除项目均值，剩余的可靠信号是“人×项目”信号，它捕捉的是受访者在特定项目上如何偏离均值，其贡献约为8……

    arXiv:2608.29455v1 Announce Type: new  Abstract: LLMs are increasingly used as human surrogates, often on the premise that richer persona data could make them substitutes or exploratory tools for specific individuals. We test this premise across four datasets covering more than 400,000 participants and more than 6,000 survey items and experimental outcomes. LLMs perform well at the aggregate level: their average responses closely align with average human responses to the same items. But this success largely reflects predicting each item's average human response. Once each item's human mean is removed, LLM predictions explain only 3.05% of the remaining respondent-specific variation, far below the 53.6% human test-retest benchmark. Richer personas, model variants, and fine-tuning do not close this gap. In variance analyses, once item means are removed, the reliable remaining signal is person-by-item. It captures how a respondent departs from the mean on a particular item and is about 8.
    
[^286]: 大锤还是手术刀？一种面向隐性仇恨言论的细粒度自适应框架

    Sledgehammer or Scalpel? A Fine-grained Adaptive Framework for Implicit Hate Speech

    [https://arxiv.org/abs/2608.27462](https://arxiv.org/abs/2608.27462)

    提出了细粒度自适应框架FAID，将隐性仇恨言论划分为浅层、针对性和上下文依赖三类，并针对不同类别采用不同复杂度的检测策略，在提升检测精度的同时降低不必要的计算开销。

    

    与带有明显脏话的显性攻击不同，隐性仇恨言论通过隐喻和上下文暗示将恶意隐藏在看似合规的表达之中，这使得在线内容审核中的检测极具挑战性。虽然现有的基于预训练语言模型（PLM）或大语言模型（LLM）的方法表现良好，但它们通常对所有样本采用单一的推理过程。这忽略了细粒度的语言细微差别，并对较简单的情况造成不必要的计算开销。我们观察到，在线仇恨言论并非单一形态，而是以多种形式呈现。因此，我们定义了三个细粒度类别：浅层、针对性和上下文依赖型。据此，我们提出了细粒度自适应隐性仇恨言论检测框架FAID，这是一个新颖的框架，首先进行细粒度分类，然后针对特定类别进行自适应处理。具体而言，对于具有表面可识别意图的浅层样本，该框架采用轻量级的提示调优进行快速分类。

    arXiv:2608.27462v1 Announce Type: new  Abstract: Unlike explicit attacks with obvious profanity, implicit hate speech hides malice within seemingly compliant expressions through metaphors and contextual hints, making its detection in online content review challenging. While existing PLM- or LLM-based methods perform well, they typically apply a single reasoning process to all samples. This overlooks fine-grained linguistic nuances and causes unnecessary computation for simpler cases. We observe that online hate speech is not monolithic but manifests in varied forms. We therefore define three fine-grained categories: Shallow, Targeted, and Context-Dependent. Accordingly, we propose Fine-grained Adaptive Implicit Hate speech Detection (FAID), a novel framework that first performs fine-grained classification and then adapts to specific categories. Specifically, for Shallow samples with surface-identifiable intents, the framework adopts lightweight prompt-tuning for rapid classification; f
    
[^287]: 不匹配并不意味着错误：不完整的参考集可能颠倒开放式心智理论追踪中的校准排名

    Unmatched Does Not Mean False: Incomplete Reference Sets Can Reverse Calibration Rankings in Open-Ended Theory-of-Mind Tracking

    [https://arxiv.org/abs/2608.25654](https://arxiv.org/abs/2608.25654)

    本文揭示不完整的参考集在开放式心智理论追踪中会颠倒校准排名，导致模型选择出现系统性偏差。

    

    开放式心智理论（ToM）追踪器会生成有限参考集中不存在的有效信念。有限参考加匹配器的流程会将不匹配的输出标记为错误，从而产生代理标签，这些标签可能颠倒固定输出上的正确评分模型选择。在保持259个信念和配对分数不变的情况下，参考重新编码将加权患病率从0.783降至0.295，并颠倒了严格正确的Brier风险：一个固定的源先验规则在参考标签下领先原生置信度0.227，而在盲审下落后0.152，这在全部六个作者设定的场景中均成立。仅使用参考的Platt重新校准器进一步颠倒了结果。在发布的301个问题的NQ-open DPR-BERT流程中，出现了特定的ICE反转：其平均置信度基线在精确匹配下将实例级校准误差改善了0.045，但在人类正确性下却恶化了0.074，且两个区间均排除零。在独立撰写的OpenToM叙事中，90-96%的审计...

    arXiv:2608.25654v1 Announce Type: new  Abstract: Open-ended Theory-of-Mind (ToM) trackers emit valid beliefs absent from finite references. A finite-reference-plus-matcher pipeline marks unmatched outputs false, creating proxy labels that can reverse proper-score model selection on fixed outputs. Holding 259 beliefs and paired scores fixed, reference recoding lowers weighted prevalence from 0.783 to 0.295 and reverses strictly proper Brier risk: a frozen source-prior rule leads native confidence by 0.227 under reference labels and trails by 0.152 under blinded adjudication, in all six authored scenarios. A reference-only Platt recalibrator reverses further. An ICE-specific reversal appears in a released 301-question NQ-open DPR-BERT pipeline: its average-confidence baseline improves instance-level calibration error by 0.045 under exact match but worsens it by 0.074 under human correctness, with both intervals excluding zero. On independently authored OpenToM narratives, 90-96% of audit
    
[^288]: 单独对齐，集体错位：预测LLM智能体群体中的对抗性俘获

    Aligned Alone, Misaligned Together: Forecasting Adversarial Capture in LLM Agent Populations

    [https://arxiv.org/abs/2608.22444](https://arxiv.org/abs/2608.22444)

    该论文发现单个LLM智能体的校准行为无法预测其群体行为，但通过良性操作数据可校准响应函数，从而提前预测对抗性俘获。

    

    arXiv:2608.22444v1 公告类型：新 摘要：AI安全评估的单位仍是单个模型，然而语言模型智能体正越来越多地部署在相互交互的群体中，这些群体读取并撰写彼此的决策。这提出了一个单智能体审计无法回答的问题：一个自身校准良好的智能体，仍可能被周围智能体拉向不同的决策。我们在一个安全分诊任务上研究此问题，其中语言模型监控器群体决定是否升级或忽略警报，并且我们可以注入一个始终推动某一方向的坚定少数派。我们发现，单个智能体在独自判断时几乎相同的两个警报，可以驱动集体行为产生巨大分歧，因此审计任何单个成员不一定能揭示群体将做什么。然而，这种集体行为可以提前预测。仅从群体良性、无对抗的操作中，我们校准一个响应函数，该函数在任何攻击运行之前就能预测结果。

    arXiv:2608.22444v1 Announce Type: new  Abstract: The unit of AI safety evaluation is still the individual model, yet language-model agents are increasingly deployed in interacting populations that read and write one another's decisions. This raises a question no single-agent audit can answer: an agent that is well-calibrated on its own may still be pulled toward a different decision by the agents around it. We study this on a security-triage task, where populations of language-model monitors decide whether to escalate or dismiss alerts, and into which we can inject a committed minority that always pushes one way. We find that two alerts a single agent judges almost identically on its own can drive collective behavior far apart, so auditing any one member need not reveal what the population will do. Yet that collective behavior can be predicted in advance. From a population's benign, adversary-free operation alone, we calibrate a response function that forecasts, before any attack is ru
    
[^289]: 一次成功不等于可靠：Thinkingbox——面向有状态业务流程中智能体的沙盒与基准

    One Success Isn't Reliability: Thinkingbox, a Sandbox and Benchmark for Agents in Stateful Business Workflows

    [https://arxiv.org/abs/2608.19741](https://arxiv.org/abs/2608.19741)

    本文提出了Thinkingbox，一个支持有状态业务流程智能体评估的沙盒和基准，强调通过多轮交互、策略遵循和持久状态转换来确保可靠性，而非仅关注单次成功。

    

    arXiv:2608.19741v1 公告类型：新公告  摘要：近期智能体基准测试越来越多地将评估建立在可执行环境中，从代码修复到网页导航、应用程序接口和函数调用。然而，完成代码之外的重要工作不仅仅需要生成合理的响应或有效的工具调用：智能体必须在多轮交互中收集缺失信息，遵循领域策略，协调依赖工具，并在无附带影响的情况下实现正确的持久状态转换。在本文中，我们介绍了Thinkingbox，一个用于工具-智能体-用户交互的沙盒，它提供隔离的MCP兼容工具会话、完整的执行轨迹以及基于终端后端状态的结局评估。基于此沙盒，Thinkingbox-bench包含507个策略条件化工作流，涵盖零售、酒店、汽车保险、新银行内部IT和咨询IT/人力资源支持等多种场景。每次尝试都通过任务特定的可执行检查进行评估。

    arXiv:2608.19741v1 Announce Type: new  Abstract: Recent agent benchmarks increasingly ground evaluation in executable environments, from code repair to web navigation, app APIs, and function calling. Yet completing consequential work beyond code requires more than producing a plausible response or valid tool call: agents must gather missing information over multiple turns, follow domain policies, coordinate dependent tools, and realize the correct persistent state transition without collateral effects. In this paper, we introduce Thinkingbox, a sandbox for tool-agent-user interaction that provides isolated MCP-compatible tool sessions, complete execution traces, and outcome evaluation over terminal backend state. Built on this sandbox, Thinkingbox-bench contains 507 policy-conditioned workflows across numerous scenarios, including retail, hospitality, auto insurance, neobank internal IT, and consulting IT/HR support. Each attempt is evaluated by task-specific executable checks that acc
    
[^290]: 针对一个模型设防，对下一个模型敞开：法律多选题基准中的仅选项可解性

    Gated Against One Model, Open to the Next: Option-Only Solvability in Legal Multiple-Choice Benchmarks

    [https://arxiv.org/abs/2608.15428](https://arxiv.org/abs/2608.15428)

    本文发现法律多选题基准存在严重的数据泄漏，模型在无问题情况下仅凭选项即可高概率答对，且泄漏对未参与筛选的模型同样有效，挑战了基准的有效性。

    

    摘要：多选题基准的评分标准是模型是否选择了正确的选项，而非是否理解了问题本身。衡量这一差距需要谨慎：如果一个模型对大多数题目都选择A，那么无论正确答案是否位于A，其得分都会高于随机水平，而当答案不在A时，这又可能被误读为识别能力。我们在UA-JudgeExam上进行了测量：该基准包含11,990道四选项题目，附有官方答案，由乌克兰法官资格高等委员会发布。在仅显示选项而无问题的情况下，Claude Haiku 4.5的得分为0.383，高于随机水平，且泄漏集中：11.8%的题目在全部八种选项顺序下均可盲答，而随机预期仅为0.2项。这并非引文式泄漏：对280,059版乌克兰立法的搜索仅恢复0.128。排除这些泄漏项后保留8,128道题目，在该子集上，筛选模型自身的得分降至0.204，而GPT-5.6——虽未参与筛选——在隐藏问题的情况下仍能答对其中0.515的题目。对十二个保留模型的评分显示……

    arXiv:2608.15428v1 Announce Type: cross  Abstract: Multiple-choice benchmarks are graded on whether a model picks the right option, not on whether it needed the question. Measuring that gap takes care: a model answering A to most items scores above chance wherever the key sits at A, and reads as recognition when it is not. We measure it on UA-JudgeExam: 11,990 four-option items with official keys, published by Ukraine's Higher Qualification Commission of Judges.   Shown the options and no question, Claude Haiku 4.5 scores 0.383 against chance, and the leak is concentrated: 11.8% of items are answered blind on all eight option orders, against 0.2 items expected by chance. It is not quotation: search over 280,059 editions of Ukrainian legislation recovers 0.128. Gating those out retains 8,128 items, on which the gating model itself now scores 0.204, and GPT-5.6, which took no part in the selection, still answers 0.515 of them with the question hidden. Scoring twelve held-out models on th
    
[^291]: 错误但有用：多智能体消息中超越答案正确性的轨迹价值

    Wrong but Useful: Trajectory Value Beyond Answer Correctness in Multi-Agent Messages

    [https://arxiv.org/abs/2608.14375](https://arxiv.org/abs/2608.14375)

    本文提出了一种受控测量方法，证明在多智能体推理中，错误答案消息也可能包含对后续推理有用的轨迹价值，挑战了仅基于答案正确性过滤消息的传统假设。

    

    多智能体推理系统通常使用一致性、置信度或自动评分来决定哪些消息应影响最终答案。这种过滤假设了一条可能正确的消息也值得保留。然而，一个错误的答案可能包含有用的分解、约束或科学原理。我们通过多样假设审议（DHD）来检验这一区别，这是一种受控测量协议，它缓存五个独立生成的消息，并重放相同的下游求解器（称为整合器），每次提供或隐藏一条消息。重放比较衡量消息的轨迹价值：即让该消息可用是否有助于或有害于后续推理。在五个数学和科学基准测试以及两个公开可用的模型家族（gpt-oss-120b和gemma-4-31B-it）中，错误但有帮助的消息出现在每个基准-模型组合中。在改变最终正确性的错误答案消息中，大多数情况下...

    arXiv:2608.14375v1 Announce Type: new  Abstract: Multi-agent reasoning systems often use agreement, confidence, or automated scores to decide which messages should shape a final answer. Such filtering assumes that a message likely to be correct is also worth keeping. Yet a wrong answer can contain a useful decomposition, constraint, or scientific principle. We test this distinction with Diverse Hypothesis Deliberation (DHD), a controlled measurement protocol that caches five independently generated messages and replays the same downstream solver, called the integrator, with each message available or hidden. The replay comparison measures a message's trajectory value: whether making the message available helps or harms subsequent reasoning. Across five mathematics and science benchmarks and two openly available model families, gpt-oss-120b and gemma-4-31B-it, wrong-helpful messages appear in every benchmark-model combination. Among wrong-answer messages that change final correctness, mo
    
[^292]: 面向无验证器测试时扩展的融通方法

    Consilience for Verifier-Free Test-Time Scaling

    [https://arxiv.org/abs/2608.09898](https://arxiv.org/abs/2608.09898)

    本文揭示了基于置信度的无验证器测试时扩展方法在复杂任务上会灾难性失效的关键局限，并提出利用“融通”思想来改进复杂任务下的无验证器测试时扩展。

    

    测试时扩展通常依赖外部验证器（如编程任务中的编译器和测试用例，或机器人应用中训练的价值函数）来获取高质量的推演结果。无验证器测试时扩展作为一种提升大语言模型（LLM）推理能力的机制正受到广泛关注，主要原因是在许多真实应用场景中我们无法获得此类高质量验证器。在现有的VF-TTS方法中，基于置信度的VF-TTS方法尤其具有前景，这类方法仅通过置信度来计算和排序推演结果。此类方法为样本评估带来近乎为零的开销，且仅需最小程度地访问模型内部状态，因此在各类模型和任务上都展现出很高的灵活性。本文揭示了现有基于置信度的VF-TTS方法的一个关键局限性：这类方法在复杂任务上会发生灾难性失效。我们观察到一个非常有趣……（原摘要在此处截断）

    arXiv:2608.09898v2 Announce Type: replace  Abstract: Test-time scaling often uses an external verifier, such as compilers and test cases in coding or trained value functions in robotics applications, to obtain high-quality rollouts. Verifier-free test-time scaling (or VF-TTS) is gaining extensive attention as a mechanism to enhance Large Language Model (LLM) reasoning, primarily because we do not have access to such high-quality verifiers in many real-world applications. Among existing VF-TTS methods, confidence-based VF-TTS methods, which compute and rank rollouts solely by confidence, are particularly promising. Such methods introduce near-zero overhead for sample evaluation and require minimal access to internal model states, making the methods highly flexible across models and tasks.   In this paper, we demonstrate a critical limitation of existing confidence-based VF-TTS methods by showing that such methods catastrophically break down on complex tasks. We observe a very interestin
    
[^293]: 基于Whisper的波斯语语音情感识别中ASR适应与表示降维研究

    A Study of ASR Adaptation and Representation Dimensionality Reduction in Persian Speech Emotion Recognition Using Whisper

    [https://arxiv.org/abs/2608.05165](https://arxiv.org/abs/2608.05165)

    该论文提出利用PCA对Whisper编码器提取的嵌入进行降维，结合注意力池化与轻量级预测头，并探索波斯语ASR任务微调的作用，从而在低资源条件下实现高效的波斯语语音情感识别。

    

    低资源语言中的语音情感识别（SER）由于标注数据有限，仍然是一个具有挑战性的问题。在这项工作中，我们研究了利用Whisper进行波斯语SER，特别关注表示降维和针对特定语言的模型适应。我们提出了一个SER框架，其中从Whisper编码器提取的帧级嵌入使用PCA进行降维，从而无需学习的投影层，并大幅减少了可训练参数的数量。降维后的表示通过基于注意力的池化机制进行聚合，并由轻量级预测头进行分类。此外，我们研究了在波斯语自动语音识别（ASR）任务上微调Whisper是否能提升下游SER的性能。在ShEMO数据集上，在说话人无关的评估协议下进行的实验表明，基于PCA的降维……

    arXiv:2608.05165v3 Announce Type: replace  Abstract: Speech Emotion Recognition (SER) in low-resource languages remains a challenging problem due to limited labeled data. In this work, we study the use of Whisper for Persian SER with a particular focus on representation dimensionality reduction and language-specific model adaptation. We propose a SER framework in which frame-level embeddings extracted from the Whisper encoder are reduced in dimensionality using PCA, eliminating the need for learned projection layers and substantially reducing the number of trainable parameters. The reduced representations are aggregated using an attention-based pooling mechanism and classified with a lightweight prediction head. In addition, we investigate whether fine-tuning Whisper on a Persian automatic speech recognition (ASR) task improves downstream SER performance. Experiments conducted on the ShEMO dataset under a speaker-independent evaluation protocol show that PCA-based dimensionality reduct
    
[^294]: PredActBench：在可控工具噪声下对工具增强对话进行基准测试

    PredActBench: Benchmarking Tool-Augmented Dialogue under Controlled Tool Noise

    [https://arxiv.org/abs/2608.02372](https://arxiv.org/abs/2608.02372)

    该论文提出PredActBench基准，首次系统评估了LLM对话智能体在不完美预测工具（含噪声）下的表现，并将信任校准指标扩展到多轮对话，发现最先进的模型很少向用户显现预测工具的不确定性。

    

    大型语言模型（LLM）智能体越来越多地被用于通过调用外部工具来支持决策。现有关于LLM工具调用的工作通常假设这些工具是完全可靠的。然而，在许多实际应用中，这些工具是带有不可忽视误差的统计预测器。我们提出了PredActBench，这是一个用于评估与不完美预测工具协同工作的对话智能体的基准，并以教育领域作为测试平台。首先，我们构建了一个基准，让LLM与带噪声的预测器协同工作以引导用户的决策。其次，我们将两个已有的信任校准指标——相对AI依赖度和相对自我依赖度——扩展到多轮对话场景中。第三，我们在OULAD数据集和我们发布的PredAct-CS数据集（包含60门课程的真实期末成绩和合成的每周分数）上评估了13个LLM，并开展了一项涉及13名教师和助教的人类研究。我们发现，当工具存在噪声时，最先进的模型很少将预测器的不确定性向用户显现出来。

    arXiv:2608.02372v2 Announce Type: replace  Abstract: Large language model (LLM) agents are increasingly used to support decisions by calling external tools. Existing work on LLM tool-calling assumes these tools are perfectly reliable. However, in many applications, the tools are statistical predictors with non-trivial errors. We introduce PredActBench, a benchmark for evaluating dialogue agents that work with imperfect predictive tools, using education as a testbed. First, we build a benchmark in which LLMs work with noisy predictors to guide users' decisions. Second, we extend two prior metrics for trust calibration, Relative AI-Reliance and Relative Self-Reliance, to multi-turn dialogue. Third, we evaluate 13 LLMs on OULAD and our released PredAct-CS dataset of 60 courses with real final grades and synthetic weekly scores, alongside a human study involving 13 instructors and teaching assistants. We find that when tools are noisy, SOTA models rarely make predictor uncertainty visible 
    
[^295]: BiCAA：面向搜索增强智能体的双向信用分配

    BiCAA: Bidirectional Credit Assignment for Search-Augmented Agent

    [https://arxiv.org/abs/2608.01321](https://arxiv.org/abs/2608.01321)

    提出BiCAA方法，通过前向可解性增益（FSG）和后续推理增益（SRG）进行双向信用分配，为搜索增强智能体的每个中间搜索步骤提供细粒度的过程奖励，兼顾证据获取进展与避免冗余检索。

    

    通过迭代检索和整合外部证据，多步搜索使搜索增强智能体能够解决复杂的问答任务。然而，原始的GRPO仅根据最终答案分配奖励，对中间搜索步骤提供的指导有限。为了提供细粒度的过程反馈，我们评估每个搜索步骤的价值：它应该获取能够推进问题解决的证据，同时避免冗余检索。基于这些互补的评估标准，我们提出了BiCAA，它结合了前向可解性增益和后续推理增益来实现双向信用分配。为了奖励证据获取的进展，FSG量化每次交互后真实答案似然的变化。SRG通过比较有和没有当前交互时后续动作的似然，来近似衡量当前交互对后续推理的支持程度，并将这种支持作为代理信号……

    arXiv:2608.01321v2 Announce Type: replace  Abstract: By iteratively retrieving and integrating external evidence, multi-step search enables search-augmented agents to tackle complex question answering. However, vanilla GRPO assigns rewards based solely on final answers, providing limited guidance for intermediate search steps. To provide fine-grained process feedback, we evaluate the value of each search step: it should acquire evidence that advances problem solving while avoiding redundant retrieval. Building on these complementary criteria, we introduce BiCAA, which combines Forward Solvability Gain (FSG) and Subsequent Reasoning Gain (SRG) for bidirectional credit assignment. To reward progress in evidence acquisition, FSG quantifies changes in ground-truth answer likelihood after each interaction. SRG compares the subsequent action's likelihood with and without the current interaction to approximately measure its support for subsequent reasoning, using this support as a proxy signa
    
[^296]: 文本到视觉的迁移是什么？视觉语言模型的能力缩放规律与迁移动态

    What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs

    [https://arxiv.org/abs/2608.00013](https://arxiv.org/abs/2608.00013)

    我们提出了首个跨家族的多模态缩放规律，通过文本能力得分直接预测VLM性能，并在150多个VLM上验证了其有效性。

    

    arXiv:2608.00013v2 公告类型：替换-交叉 摘要：选择合适的大型语言模型（LLM）骨干是构建视觉语言模型（VLM）时最关键的决策，但这一过程从根本上缺乏原则性：基于计算量的缩放规律无法跨模型家族泛化，且不存在在训练开始前直接预测VLM性能的框架。我们提出了能力驱动的多模态缩放规律，这是首个跨家族框架，能够从直接可观测的文本能力预测VLM基准准确率。给定通过主成分分析从LLM文本基准中提取的低维能力得分$S$，我们将VLM性能建模为$S$的函数，并引入每个骨干的迁移率和量化数据缩放效率的吸收率。为拟合和验证该框架，我们在严格控制的配方下，基于34个LLM（涵盖7个模型家族）训练了超过150个VLM。在超过200个文本基准和50个多模态基准上的评估表明...

    arXiv:2608.00013v2 Announce Type: replace-cross  Abstract: Choosing the right large language model (LLM) backbone is the most consequential decision when building a vision-language model (VLM), yet it remains fundamentally unprincipled: compute-based scaling laws fail to generalize across model families, and no framework exists for directly predicting VLM performance before training begins. We propose the Capability-Driven Multimodal Scaling Law, the first cross-family framework that predicts VLM benchmark accuracy from directly observable textual capability. Given a low-dimensional capability score $S$ extracted from LLM textual benchmarks via PCA, we model VLM performance as a function of $S$, with a per-backbone transfer rate and an absorption rate that quantifies data-scaling efficiency. To fit and validate the framework, we train over 150 VLMs on 34 LLMs spanning 7 model families under a strictly controlled recipe. Evaluations on more than 200 textual and 50 multimodal benchmarks 
    
[^297]: 大语言模型在受控环境中难以模拟人类信念更新

    LLMs struggle to simulate human belief updates in controlled environments

    [https://arxiv.org/abs/2607.28347](https://arxiv.org/abs/2607.28347)

    该研究将六个大语言模型与391名真实参与者的信念更新数据进行一对一对比，发现尽管LLM在给定真实初始立场时无系统性方向偏差，但所有模型均无法模拟初始立场、无法产生忠实的信念更新，且存在三种系统性偏差，表明LLM作为社会科学研究中人类被试替代品的可靠性存疑。

    

    大语言模型（LLMs）在社会科学实验中越来越多地被用作人类研究参与者的替代者，但这种做法的忠实度却很少被直接检验。我们测试了六个大语言模型能否模拟个体人类的信念更新，将LLM的输出与来自Prolific平台上391名英国参与者的真实数据进行一对一比较，这些参与者在阅读Reddit评论后更新了他们对三个讨论话题的立场。每个参与者由一个基于其人口统计学和性格特质数据构建的人设（persona）所条件化的LLM进行模拟。在给定参与者真实初始立场的情况下，六个模型中有五个在模拟的后续立场中没有表现出系统性的方向偏差，尽管各模型之间个体层面的误差大小差异很大。然而，所有六个模型都无法模拟参与者的初始立场本身，也无法在自我生成的立场上产生忠实的信念更新。所有模型中出现了三种系统性偏差：ove（摘要原文在此处截断）

    arXiv:2607.28347v2 Announce Type: replace  Abstract: LLMs are increasingly deployed as proxies for human study participants in social science experiments, yet the fidelity of this practice has rarely been tested directly. We test whether six LLMs can simulate individual human belief updates, comparing LLM outputs 1-to-1 against ground truth data from 391 UK participants on Prolific, who updated their stances on three discussion topics after reading Reddit comments. Each participant was simulated by an LLM conditioned on a persona derived from their demographic and personality trait data. Given participants' actual initial stances, five of the six models showed no systematic directional bias in their simulated post-stances, although the size of their individual-level errors varied considerably between models. All six models fail to simulate initial stances themselves and to produce faithful belief updates from self-generated stances. Three systematic biases emerge across all models: ove
    
[^298]: 推理共识：基于加权有向无环图聚合的大语言模型推理结构化集成

    Reasoning Consensus: Structural Ensembling of LLM Reasoning via Weighted DAG Aggregation

    [https://arxiv.org/abs/2607.27783](https://arxiv.org/abs/2607.27783)

    提出推理共识（RC）方法，通过加权有向无环图聚合将多个大语言模型的思维链进行结构化集成，在准确率与多数投票和辩论相当的同时，还能透明地展示每个推理步骤获得的支持度。

    

    大语言模型（LLM）以自由格式的思维链进行推理，但不会揭示哪些步骤得到了充分支持、哪些备选方案被排除在外。在高风险领域，用户无法判断答案是否建立在可靠的推理之上。多数投票和多智能体辩论利用模型间的一致性来改进答案，但无法展示对单个推理步骤的共识。推理共识从每个思维链中提取类型化的推理图，合并来自不同模型的图中等价的步骤，并根据有多少条推理链支持该步骤来为其加权。RC 展示了这种逐步支持情况以及每个答案背后多样化的论证方式。在涵盖法律、科学、医学、叙事推理和逻辑的 7 个基准测试上，RC 的准确率与多数投票和辩论方法相当（69.4% vs. 69.0-70.3%）。LLM 裁判更偏好 RC 中得到最佳支持的推理图，而非同一答案的其他推理图。在一项用户研究中，参与者阅读 RC 推理图的速度快于阅读辩论内容……

    arXiv:2607.27783v2 Announce Type: replace  Abstract: Large Language Models (LLMs) reason in free-form chain-of-thought (CoT) without revealing which steps were well supported and which alternatives were rejected. In high-stakes domains, users then cannot judge if an answer rests on sound reasoning. Majority voting and multi-agent debate improve answers using model agreement, but do not expose agreement on individual steps. Reasoning Consensus (RC) extracts a typed reasoning graph from each CoT, merges equivalent steps across graphs from diverse models, and weights every step by how many chains attest to it. RC reveals this step-wise support and the varied justifications behind each answer. On 7 benchmarks across law, science, medicine, narrative reasoning and logic, RC matches majority voting and debate in accuracy (69.4% vs. 69.0-70.3%). An LLM judge prefers RC's best-supported graph to other graphs for the same answer. In a user study, participants read RC graphs faster than debate t
    
[^299]: AgentSnare：学习延迟、转移与化解自主渗透代理

    AgentSnare: Learning to Delay, Divert, and Defuse Autonomous Penetration Agents

    [https://arxiv.org/abs/2607.26998](https://arxiv.org/abs/2607.26998)

    AgentSnare提出了一种轨迹自适应的欺骗防御系统，通过基于渗透代理交互历史动态构建诱饵工件，持续将自主渗透代理从真实目标引开，克服了传统静态蜜罐痕迹易被识别绕过的缺陷。

    

    大型语言模型（LLM）代理通过“观察-动作”循环实现渗透测试自动化，根据工具返回的观察结果选择动作。这种依赖性使防御者能够注入欺骗性观察结果，从而误导代理的决策过程。然而，现有的防御手段严重依赖在攻击前预先植入环境中的静态、孤立的人工痕迹（artifacts）。先进的代理能够逐步识别并绕过这些痕迹，最终将其攻击重心重新聚焦到真实目标上。为解决这一问题，我们提出了AgentSnare，一种轨迹自适应的欺骗系统，它能够动态展开诱饵环境，持续将渗透代理从真实目标引开。具体而言，AgentSnare采用了一个工件构建策略模型，该模型以代理的交互历史和诱饵状态为条件构建候选工件。随后，AgentSnare会对这些（候选工件进行验证）……

    arXiv:2607.26998v4 Announce Type: replace-cross  Abstract: Large language model (LLM) agents automate penetration testing through an observation-action loop, selecting actions based on observations returned by tools. This dependence allows defenders to inject deceptive observations that can mislead the agent's decision-making process. However, existing defenses rely heavily on static, isolated artifacts planted in the environment prior to an attack. Advanced agents can progressively recognize and bypass these artifacts, ultimately refocusing their exploitation attempts on the real target. To address this issue, we introduce AgentSnare, a trajectory-adaptive deception system that dynamically unfolds a decoy environment to continually steer the penetration agent away from the real target. Specifically, AgentSnare employs an artifact-construction policy model that constructs candidate artifacts conditioned on the agent's interaction history and decoy state. AgentSnare then validates these
    
[^300]: MediaWiki代码到代码搜索：用于开源软件实体语义发现的神经检索

    MediaWiki Code2Code Search: Neural Retrieval for the Semantic Discovery of Open-Source Software Entities

    [https://arxiv.org/abs/2607.26766](https://arxiv.org/abs/2607.26766)

    本文提出了MediaWiki Code2Code Search神经检索系统，通过FAISS IVF-PQ索引实现对2500多个代码库中129万个代码实体的语义化代码到代码搜索，在满足6 GiB内存限制的同时实现了96.6%的索引空间压缩和1.85秒的中位查询延迟。

    

    大规模生态系统中的代码搜索常常受到用户查询与实现细节之间词汇鸿沟的阻碍，同时还面临传统信息检索（IR）的低延迟与深度学习（DL）的精确性之间的权衡问题。我们提出了MediaWiki Code2Code Search，一个用于语义化代码到代码发现的神经检索系统。通过索引2500多个MediaWiki代码库中的129万个结构化实体（函数、类型和模板），我们的系统能够基于计算意图而非表面词元进行检索。我们采用分离构建架构，将GPU密集型的离线索引与仅使用CPU的服务层解耦；我们的FAISS IVF-PQ索引仅占用168.6 MiB空间，相比扁平float32基线减少了96.6%，并在普通硬件上实现了1.85秒的中位查询延迟，满足了Wikimedia Toolforge的6 GiB内存限制。在一项由...进行的盲测评估中（原文此处不完整）

    arXiv:2607.26766v2 Announce Type: replace-cross  Abstract: Code search in large-scale ecosystems is often hindered by the lexical gap between user queries and implementation details, alongside the trade-off between the low latency of traditional Information Retrieval (IR) and the precision of Deep Learning (DL). We present MediaWiki Code2Code Search, a neural retrieval system for semantic code-to-code discovery. By indexing 1.29 million structural entities (functions, types, and templates) across 2,500+ MediaWiki repositories, our system enables retrieval based on computational intent rather than surface tokens. We employ a split-build architecture, decoupling GPU-intensive offline indexing from a CPU-only serving layer; our FAISS IVF-PQ index occupies 168.6 MiB: a 96.6\% reduction compared to a flat float32 baseline, and achieves a median query latency of 1.85 seconds on commodity hardware, satisfying the 6 GiB RAM constraint of Wikimedia Toolforge. In a blinded evaluation graded by a
    
[^301]: 指令微调模型在局部产生比人类更多的句法重叠——但其条件性复用并未更多

    Instruction-Tuned Models Locally Produce More Syntactic Overlap -- But Not More Conditional Reuse -- Than Humans Do

    [https://arxiv.org/abs/2607.26015](https://arxiv.org/abs/2607.26015)

    该研究对16个开源Llama和Gemma模型在人类对话相邻轮次间的CFG规则复用进行测量，发现指令微调模型产生的局部句法重叠高于人类，但这只是与上下文相关的效应，其条件性句法复用并不比人类更多。

    

    句法趋同——即说话者在语言使用上向其对话者的语法特征靠拢的倾向——是人类对话中一个有充分记录且很大程度上是隐性的特征。大型语言模型相对于人类基线是否会对人类用户表现出类似的句法趋同（包括跨越广泛的句法结构）仍是一个悬而未决的问题。本研究利用Blevins等人[2026]的替换范式数据——其中模型生成的内容替换既有人类对话中一位说话者的发言——在十六个开放权重的Llama和Gemma模型（1B至70B，涵盖预训练与指令微调版本）上，测量了每个模型在1,901个匹配位置处相邻轮次间对上下文无关文法（CFG）规则的复用情况。所有模型与前一个人类轮次的CFG规则重叠程度都高于其与随机抽取的不相关人类启动语料的重叠程度（这是一种与上下文相关的效应，无法将句法启动与主题……分离开来）。（原文摘要至此被截断）

    arXiv:2607.26015v2 Announce Type: replace  Abstract: Syntactic convergence - the tendency of speakers to adapt in language towards the grammatical profiles of their interlocutors - is a well-documented, largely implicit feature of human dialogue. Whether large language models exhibit analogous syntactic convergence towards human users relative to human baselines (including across a broad range of syntactic constructions) remains an open question. Using substitution-paradigm data from Blevins et al. [2026] in which model generations replace one speaker's turns in pre-existing human dialogues, this study measures turn-adjacent reuse of context-free grammar (CFG) rules across sixteen open-weight Llama and Gemma models (1B-70B, pretrained and instruction-tuned) at 1,901 matched positions per model. Every model showed greater CFG-rule overlap with the preceding human turn than with a sampled unrelated human prime (a context-associated effect that does not isolate syntactic priming from topi
    
[^302]: ClinFusion：一个以视觉为中心、面向整体医学理解的多模态大语言模型系统

    ClinFusion: A Vision-Centric Multimodal LLM System for Holistic Medical Understanding

    [https://arxiv.org/abs/2607.24743](https://arxiv.org/abs/2607.24743)

    ClinFusion提出了一种融合2D与3D医学图像理解的组合式级联视觉编码器架构，并配套基于视觉的评估框架（MedIF-Bench与感兴趣区域评估方法），实现了以视觉为中心的整体医学理解。

    

    多模态大语言模型（MLLMs）具有彻底变革临床实践的巨大潜力，但将其部署到医学领域本质上是一个以视觉为中心的挑战：模型必须从异构的2D和3D医学图像中吸收知识，且评估协议必须与放射科医生的临床实践保持一致，并提供准确、细粒度且以事实为驱动的评估。在本文中，我们提出了ClinFusion，这是一个为整体医学理解设计的以视觉为中心的多模态大语言模型，系统地解决了上述局限性。我们提出了一种组合式级联视觉编码器架构，其核心是级联空间感知局部融合算子，能够在融合编码器内统一多样化的2D与原生3D医学图像理解。我们进一步引入了一个基于视觉的评估框架，包括用于指令跟随能力评估的MedIF-Bench，以及一种基于感兴趣区域的评估方法。

    arXiv:2607.24743v3 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLMs) hold immense potential to revolutionize clinical practice, yet deploying them in the medical domain is fundamentally a vision-centric challenge: models must absorb knowledge from heterogeneous 2D and 3D medical images, and evaluation protocols must align with radiologists' clinical practice and provide an accurate, fine-grained and factualness-driven assessment. In this paper, we introduce ClinFusion, a vision-centric MLLM designed for holistic medical understanding that systematically addresses these limitations. We propose a compositional and cascaded vision encoder architecture featuring a Cascade Spatial-Aware Locality Fusion operator that unifies diverse 2D and native 3D medical image understanding within a fused encoder. We further introduce a vision-grounded evaluation framework, including MedIF-Bench for instruction-following assessment and a region-of-interest-grounded method fo
    
[^303]: 内存高效的音频合成：基于解耦时间深度扩散变换器

    Memory Efficient Audio Synthesis with Decoupled Temporal Depth Diffusion Transformers

    [https://arxiv.org/abs/2607.23811](https://arxiv.org/abs/2607.23811)

    本文提出了一种内存高效的音频合成架构，通过解耦时间与深度处理、复用单一深度解码器生成所有RVQ层级，在设备端实时合成高保真语音，显著降低内存和计算需求。

    

    arXiv:2607.23811v2 公告类型：替换交叉 摘要：Siri 表现力语音通过 AFM 3 Core Advanced（苹果最强大的端上基础模型）在设备上实时合成丰富、可配置的语音。本工作介绍了支撑该能力的内存高效音频合成架构：一个去分词器，将基础模型发出的语义音频令牌转换为高保真音频，在苹果矩阵协处理器（AMX）的严格计算和内存预算内运行。我们将语义音频令牌转换为残差向量量化（RVQ）表示，采用三组件设计：流式编码器、时间解码器和深度解码器，系统性地解耦时间和深度处理。一个可复用的深度解码器，配备扩散变换器（DiT）风格的阶段条件，自回归生成所有 RVQ 层级，取代了先前多解码器架构中专用的逐层级解码器，同时采用因果滑窗。

    arXiv:2607.23811v2 Announce Type: replace-cross  Abstract: Siri Expressive Voices synthesize rich, configurable speech in real time and entirely on device, powered by AFM 3 Core Advanced, Apple's most powerful on-device foundation model. This work presents the memory-efficient audio synthesis architecture behind that capability: a detokenizer that converts the semantic audio tokens emitted by the foundation model into high-fidelity audio within the tight compute and memory budget of the Apple Matrix Coprocessor (AMX). We convert semantic audio tokens to a residual vector quantization (RVQ) representation with a three-component design, a streaming encoder, a temporal decoder, and a depth decoder, that systematically decouples temporal and depth processing. A single reusable depth decoder with Diffusion Transformer (DiT)-style stage conditioning generates all RVQ levels autoregressively, replacing the dedicated per-level decoders of prior multi-decoder architectures, while causal sliding
    
[^304]: 路由子空间：审计微调语言模型中评估与部署之间的不匹配

    Routing Subspaces: Auditing Evaluation-to-Deployment Mismatch in Fine-Tuned Language Models

    [https://arxiv.org/abs/2607.20436](https://arxiv.org/abs/2607.20436)

    该论文提出一种审计方法，通过在中间层定位并干预编码“评估-部署”行为差异的内部子空间，显著缩小了微调语言模型在评估式提示与普通使用提示之间的行为差距。

    

    安全性评估通常假设在测试期间观察到的行为能够反映日常使用中的行为，但微调可能会打破这一假设。一个模型检查点在评估式提示下可能显得行为正常，而同样的行为在普通使用提示下却依然存在。输出分数能够揭示这种不匹配，但无法定位其来源。我们研究这种区分是否被编码在一个稳定的内部位置，并提出一种审计方法：在由路径修补信息所指示的中层深度窗口处拟合配对激活对比，然后在留出的提示上修改所得坐标。在四个全矩阵指令微调模型实例上，该干预在十二个模型-行为设置中的十个将评估到部署的差距降低了超过0.12，并在其中四个设置中基本消除了差距（干预后差距低于0.06）；在配对问题数n≥120的八个设置中，六个差距减小，一个被消除。第五个模型支持定位与编辑验证……（原文摘要在此处截断）

    arXiv:2607.20436v2 Announce Type: replace  Abstract: Safety evaluations often assume that behavior observed during testing reflects behavior in ordinary use, but fine-tuning can break this assumption. A checkpoint can appear fixed under evaluation-style prompts while the same behavior persists under ordinary-use prompts. Output scores reveal this mismatch but do not locate it. We investigate whether the distinction is encoded in a stable internal site and introduce an audit that fits a paired activation contrast at a path-patching-informed mid-depth window, then modifies the resulting coordinate on held-out prompts. The intervention reduces the evaluation-to-deployment gap by more than $0.12$ in ten of twelve model--behavior settings and closes it (post-intervention gap below $0.06$) in four, across four full-matrix instruction-tuned model instances; among the eight settings with $n{\geq}120$ paired questions, six reduce and one closes. A fifth model supports localization and edit-prov
    
[^305]: 选择塑造边界：在未筛选的自然语言推理人群中复制单调性与标签一致性研究

    Selection Shapes the Boundary: A Replication of Monotonicity and Label Agreement in Unselected NLI Populations

    [https://arxiv.org/abs/2607.19231](https://arxiv.org/abs/2607.19231)

    本研究在未筛选的SNLI/MultiNLI人群中预注册复制了先前基于筛选数据的发现，结果与原预测相反——含非向上单调性算子的假设标签一致性略高而非略低，表明ChaosNLI按分歧程度筛选条目的机制可能塑造了先前观察到的单调性与标签一致性之间的边界效应。

    

    以往关于自然语言推理（NLI）中人类标签变异（HLV）的研究常常依赖于按不一致程度筛选条目的重新标注资源。较早的一项研究（Choi, 2026, arXiv:2607.15870）发现，在ChaosNLI中，包含非向上单调性算子的假设显示出略高的标签熵（Cliff's delta = -0.284），该研究报告称这一差异对标注器错误敏感。在其SNLI和MultiNLI部分，ChaosNLI仅对多数标签恰好获得五票中三票的条目进行重新标注。我们在ChaosNLI所来源的未筛选人群中复制这一边界效应。我们使用SNLI和MultiNLI开发集，采用相同的算子标注器和四级序数一致性结果，且分析方案在分析开始前已在我们工作仓库中冻结。计划中的预测失败了。所有七个对比均返回正的Cliff's delta（非向上条目的一致性略高而非略低）……

    arXiv:2607.19231v2 Announce Type: replace  Abstract: Prior work on human label variation (HLV) in natural language inference (NLI) has often relied on re-annotation resources that select items by disagreement level. An earlier study (Choi, 2026, arXiv:2607.15870) found that hypotheses containing non-upward monotonicity operators showed somewhat higher label entropy in ChaosNLI (Cliff's delta = -0.284), a difference that study reports as sensitive to tagger error. In its SNLI and MultiNLI portions, ChaosNLI re-annotates only items whose majority label carries exactly three of five votes. We replicate this boundary in the unselected populations that ChaosNLI was drawn from. We use the SNLI and MultiNLI development sets with the same operator tagger and a four-level ordinal agreement outcome, under a plan frozen in our working repository before analysis. The planned prediction fails. All seven contrasts return a positive Cliff's delta (non-upward items agree slightly more, not less), the 
    
[^306]: 大语言模型回复自动情感评分中的查询语言敏感性

    Query Language Sensitivity in Automated Sentiment Scores of LLM Responses

    [https://arxiv.org/abs/2607.13304](https://arxiv.org/abs/2607.13304)

    该研究发现大语言模型回复的自动情感评分对查询语言高度敏感——查询语言可解释约26%的拟合方差，远高于品牌因素的影响，且剔除英语后该效应大幅减弱，表明跨语言使用自动化情感评分时需谨慎解读。

    

    自动化评分常被用于概括大语言模型在不同提示词、提供商和语言下描述品牌的方式。这类评分出现差异的原因可能是回答本身不同，也可能是评分工具在不同文本或语言上表现不一致，或者两者兼有。我们在12,933条回复中考察了这一测量问题，这些回复来自一项跨越20个存储品牌标签、8种查询语言、3个提供商别名和15个提示词标识符的实验设计。评估结果为多语言情感分类器输出的带符号获胜类别置信度：中性获胜者编码为零，正面与负面获胜者则获得带符号的置信度。对所有回复拟合的交叉线性混合模型显示，26.49%的拟合方差归因于查询语言，1.46%归因于品牌。在剔除英语后进行的探索性重拟合中，5.07%的方差归因于语言，2.23%归因于品牌。这些百分比具有不同的分母：总拟合方差下降了65.08%。

    arXiv:2607.13304v2 Announce Type: replace-cross  Abstract: Automated scores are often used to summarize how large language models describe brands across prompts, providers, and languages. Such scores can vary because the answers differ, because the scoring instrument behaves differently across texts or languages, or for both reasons. We examine this measurement problem in 12,933 responses from a design crossing 20 stored brand labels, eight query languages, three provider aliases, and 15 prompt identifiers. The outcome is the signed winning-class confidence from a multilingual sentiment classifier: neutral winners are coded zero, while positive and negative winners receive signed confidence. A crossed linear mixed model fitted to all responses assigns 26.49% of fitted variance to query language and 1.46% to brand. An exploratory refit after excluding English assigns 5.07% to language and 2.23% to brand. These percentages have different denominators: total fitted variance falls by 65.08
    
[^307]: MafiaScope：检视社交推理游戏中的多智能体推理

    MafiaScope: Inspecting Multi-Agent Reasoning in Social Deduction Games

    [https://arxiv.org/abs/2607.10645](https://arxiv.org/abs/2607.10645)

    MafiaScope将社交推理游戏“狼人杀”转变为测量LLM多智能体社会推理的开放测试平台，通过非侵入式的私下结构化提问来揭示智能体公开行为与私下真实评估之间的差异，并提供交互式可视化器分析角色评估轨迹。

    

    LLM智能体的公开行为很少能揭示其社会推理能力：一个投票正确的智能体可能只是在猜测，而且其公开声明可能与私下引出的评估不一致。我们提出了MafiaScope，一个开放测试平台，它将社交推理游戏“狼人杀”转变为测量多智能体推理的工具。在每次公开发言后，存活的智能体会在其可用对话的副本上私下回答可配置的结构化问题。这种探测对于行动上下文是非侵入式的：问题和答案都不会进入用于继续游戏的历史记录。角色评估根据引擎已知的真实角色进行评分；二阶预测则与其他智能体的报告进行比较。交互式可视化器呈现评估轨迹：扮演模式聚焦于单个智能体的信息和报告，面板图表展示时间线对齐的准确率和校准度，反事实……（摘要在此处被截断）

    arXiv:2607.10645v3 Announce Type: replace  Abstract: An LLM agent's public behaviour reveals little about its social reasoning: an agent that votes correctly may be guessing, and its public claims may differ from privately elicited assessments. We present MafiaScope, an open testbed that turns the social deduction game Mafia into a measurement instrument for multi-agent reasoning. After each public utterance, living agents privately answer configurable structured questions on copies of their available conversations. Probing is non-invasive with respect to the acting context: neither the questions nor the answers enter the histories used to continue the game. Role assessments are scored against roles known to the engine; second-order predictions are compared with other agents' reports. An interactive visualizer renders the assessment trajectories: impersonate mode focuses on one agent's information and reports, panels chart timeline-aligned accuracy and calibration, and counterfactual r
    
[^308]: fog：通过代码的分层函数组合为海德-西美尔动画生成富有表现力的运动

    fog: Generating Expressive Motion through Layered Function Composition of Code for Heider-Simmel Animation

    [https://arxiv.org/abs/2607.07952](https://arxiv.org/abs/2607.07952)

    提出fog分层函数组合框架，通过生成动词、副词和情感对应的代码函数来捕捉运动的轨迹、方式和感觉，帮助用户在海德-西美尔风格动画中表达运动与情感。

    

    运动和情感是智能、富有表现力行为的核心组成部分。在本文中，我们介绍了fog，一个用于生成运动函数的分层函数组合框架。我们展示了如何使用fog在海德-西美尔风格的动画中表达运动和情感。这一代码生成框架可以帮助用户生成对应动词、副词和情感的函数，以捕捉运动的轨迹、方式和感觉。此外，该框架还配备了一个动画编辑器，帮助用户以标记、箭头和路径的形式提供空间上下文。我们通过技术评估研究了不同的函数生成与组合设置，以评估我们的方法。在一项混合方法用户研究（n=12）中，我们发现fog能够通过传递用户标记和绘制的空间上下文，并为用户提供更多维度的控制，从而有效地支持用户创作。

    arXiv:2607.07952v2 Announce Type: replace-cross  Abstract: Motion and emotion are core parts of intelligent, expressive behavior. In this paper, we introduce fog, a layered function composition framework for generating motion functions. We demonstrate how fog can be used to express motion and emotion in Heider-Simmel style animations. This code generation framework can help users generate functions for verbs, adverbs, and emotions to capture the trajectory, manner, and feeling of motion. It is complemented by an animation editor that helps users provide spatial context in the form of marks, arrows, and paths. We evaluate our approach with a technical evaluations investigating different function generation and composition setups. In a mixed-methods user study (n=12), we find that fog can support users can support users by relaying spatial context they mark and draw and giving them more dimensions for control.
    
[^309]: CoPiT：面向传统蒙文双文字低资源蒙古语的认知枢轴翻译

    CoPiT: Cognitive Pivot Translation for Digraphic Low-Resource Mongolian in the Traditional Script

    [https://arxiv.org/abs/2607.05849](https://arxiv.org/abs/2607.05849)

    该论文提出CoPiT，一种利用蒙古语内部资源不平衡的枢轴翻译流水线，先解决传统蒙文的拼写歧义，再经由资源较丰富的西里尔蒙古文进行翻译，从而在多个骨干模型和目标语言上显著提升翻译质量（BLEU大幅提升）。

    

    低资源语言对机器翻译而言仍具挑战性，蒙古语是一个代表性案例。作为一种双文字语言，蒙古语同时使用西里尔字母和传统蒙文两种文字书写，而这两种文字在数据可得性上存在严重的不平衡。尽管西里尔文字的资源相对充足，传统蒙文却极度缺乏数据且存在拼写歧义，导致直接翻译的性能大幅下降。我们提出了CoPiT，一种以认知为动机的基于枢轴的翻译流水线，它利用这种内部资源层级，将翻译经由西里尔文字进行路由。该流水线在翻译之前显式地解决传统蒙文中由文字引起的歧义，从而实现更稳定、更准确的语义传递。在多个骨干模型和目标语言上，CoPiT总体上优于直接翻译，取得了显著的绝对BLEU提升，最高可达（原文此处截断）……

    arXiv:2607.05849v2 Announce Type: replace  Abstract: Low-resource languages remain challenging for machine translation, and Mongolian is a representative case. As a digraphic language, Mongolian is written in both Cyrillic and Traditional scripts, which exhibit a severe imbalance in data availability. While the Cyrillic script is relatively well-resourced, the Traditional script remains extremely data-scarce and orthographically ambiguous, leading to substantial performance degradation in direct translation. We propose CoPiT, a cognitively motivated pivot-based translation pipeline that exploits this internal resource hierarchy by routing translation through the Cyrillic script. The pipeline explicitly resolves script-induced ambiguity in the Traditional script before translation, enabling more stable and accurate meaning transfer. Across multiple backbone models and target languages, CoPiT generally improves over direct translation, with substantial absolute BLEU improvements and up t
    
[^310]: 为什么要与连续潜变量苦苦纠缠？基于渲染压缩的可解释离散潜变量推理

    Why Struggle with Continuous Latents? Interpretable Discrete Latent Reasoning via Rendered Compression

    [https://arxiv.org/abs/2606.29712](https://arxiv.org/abs/2606.29712)

    该论文提出首个离散潜变量推理方法（DLR），通过将文本思维链渲染为图像并基于聚类微调构建离散潜变量词表，把连续潜变量状态转换为显式离散标记，从而实现更稳定、可解释且高效的潜空间推理。

    

    大型语言模型通过显式的思维链和强化学习实现了较高的推理性能，但需要较长的输出序列和较长的推理时间。潜空间推理通过将计算转移到潜空间中来降低这一成本；然而，连续潜变量方法难以训练，存在推理轨迹不稳定和不可解释的问题。我们认为这些问题源于连续空间推理与离散符号监督之间的失配，因为连续状态缺乏用于逐步对齐的显式锚点。为解决这一问题，我们提出了离散潜变量推理（Discrete Latent Reasoning，DLR），这是首个将连续潜变量状态转换为显式离散标记的方法。受基于渲染的压缩技术启发，我们将文本思维链渲染为图像，提取视觉特征，并通过基于聚类的微调构建离散潜变量词表。通过扩展词表并超越……（原文摘要在此处截断）

    arXiv:2606.29712v2 Announce Type: replace  Abstract: Large language models achieve high reasoning performance via explicit chain-of-thought and reinforcement learning, but require long output sequences and extended inference time. Latent reasoning reduces this cost by shifting computation into a latent space; however, continuous latent methods are hard to train, suffering from unstable and uninterpretable reasoning trajectories. We argue these issues stem from a misalignment between continuous-space reasoning and discrete symbolic supervision, as continuous states lack explicit anchors for step-by-step alignment. To resolve this, we propose \textbf{Discrete Latent Reasoning~(DLR)}, the first method that converts continuous latent states into explicit discrete tokens. Inspired by render-based compression, we render textual chains of thought into images, extract visual features, and construct a discrete latent vocabulary via clustering-based fine-tuning. Expanding the vocabulary and outp
    
[^311]: SpeechEQ：面向社交感知语音对话模型的情商基准测试

    SpeechEQ: Benchmarking Emotional Intelligence Quotient in Socially Aware Voice Conversational Models

    [https://arxiv.org/abs/2606.25990](https://arxiv.org/abs/2606.25990)

    提出了SpeechEQ评估框架，基于EQ-i 2.0理论构建了包含2,265段对话、覆盖15个情商子量表的数据集，并设计了口语情商（SEQ）评分，用于系统评估语音语言模型在多轮语音对话中的社交情感推理能力。

    

    随着多模态对话系统日益参与语音交互，其处理副语言社交线索的能力已成为自然人机交流的关键瓶颈。然而，现有的机器情商评估仅通过孤立的文本或被动的声学感知来评估推理能力，忽视了主动的多轮对话所需的复杂跨模态推理。我们提出了SpeechEQ，这是一个旨在评估语音语言模型社交语言推理能力的综合框架。该框架包含一个经过验证的数据集，涵盖2,265段对话，并基于EQ-i 2.0理论划分为15个情商（EQ）子量表，同时提出了一套多轮评估协议，通过受人类情商评估启发的口语情商（SEQ）分数进行衡量。实验表明，无论是现有的语音情感识别方法还是端到端的语音语言模型，在应对此类任务时均存在局限性。

    arXiv:2606.25990v2 Announce Type: replace  Abstract: As multimodal conversational systems increasingly engage in spoken interaction, their ability to navigate paralinguistic social cues has become a critical bottleneck for natural human-AI communication. However, existing evaluations of machine emotional intelligence assess reasoning exclusively through isolated text or passive acoustic perception, overlooking the complex cross-modal reasoning required for active, multi-turn dialogue. We introduce \textsc{SpeechEQ}, a comprehensive framework designed to evaluate the sociolinguistic reasoning of Speech-Language Models (SLMs). The framework includes a validated dataset of 2,265 dialogues across 15 Emotional Quotient (EQ) subscales grounded in EQ-i 2.0 theory, along with a multi-turn evaluation protocol measured by our proposed Spoken EQ (SEQ) score inspired by human EQ assessments. Experiments show limitations in how both existing Speech Emotion Recognition and end-to-end Speech-Language
    
[^312]: SWE-Opt：用于评估大语言模型在面向性能的仓库级优化任务上能力的基准

    SWE-Opt: A Benchmark for Evaluating LLMs on Performance-Oriented Repository-Level Optimization

    [https://arxiv.org/abs/2606.25530](https://arxiv.org/abs/2606.25530)

    SWE-Opt 提出了一个基于开源项目专家优化案例构建的仓库级基准，通过参数化测试在感知噪声的条件下多维度评估大语言模型的软件性能优化能力，弥补了现有基准忽略时间-内存权衡与执行环境变异性的不足。

    

    软件性能优化是一项出了名的复杂且高度依赖人工的任务。尽管大语言模型（LLM）在代码改进方面的应用日益增多，我们仍然缺乏能够真实反映优化在现实代码库中如何发生的基准。现有框架通常将问题过度简化，只关注孤立的函数或单一性能指标，从而忽略了执行时间与内存占用之间的关键权衡、测量环境固有的噪声，以及不同输入数据和执行条件所带来的变异性。为此，我们提出了 SWE-Opt，一个基于开源项目中 102 个专家手写优化案例构建的仓库级基准。与以往基准不同，SWE-Opt 为每个任务配备了参数化测试，在感知噪声的测量方式下，跨越不同的输入数据和执行条件，评估运行时间、峰值内存以及加权时间内存使用量（TWMU）……

    arXiv:2606.25530v2 Announce Type: replace-cross  Abstract: Software performance optimization is a notoriously complex and manual task. Despite the growing use of Large Language Models (LLMs) for code refinement, we still lack benchmarks that capture how optimization actually happens in real-world codebases. Existing frameworks often oversimplify the problem by focusing on isolated functions or a single performance metric, missing the critical trade-offs between execution time and memory footprint, the inherent noise of the measurement environment, and the variability introduced by different input data and execution conditions. We address this by introducing SWE-Opt, a repository-level benchmark derived from 102 expert-written optimizations from open-source projects. Unlike previous benchmarks, SWE-Opt pairs each task with parameterized tests to evaluate runtime, peak memory, and Time-Weighted Memory Usage (TWMU) across varying input data and execution conditions under noise-aware measu
    
[^313]: 超越下一观测预测：面向序列决策的智能体自主世界建模

    Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making

    [https://arxiv.org/abs/2606.25421](https://arxiv.org/abs/2606.25421)

    该论文提出智能体自主世界建模（AAWM），让智能体根据自身行动前的决策需求主动检索并综合环境转移证据来构建训练监督，从而学习与决策相关的环境动态，而非单纯预测下一观测。

    

    近年来针对大语言模型（LLM）智能体的世界建模研究通常将学习目标设定为下一观测预测。然而，这一目标将监督信号与状态转移所恰好揭示的内容绑定在一起，可能遗漏与智能体当前决策最相关的环境动态。为弥合这一差距，我们提出了智能体自主世界建模（Agent-Authored World Modeling, AAWM），这是一种从策略自身决策需求出发构建监督信号的训练方法。具体而言，在每一状态下，智能体首先识别其在行动前需要理解环境的哪些方面；这些需求驱动在多条轨迹中检索相关的转移证据，随后将其综合为捕捉面向决策的动态规律的训练目标，而非简单重构下一观测。这使得训练目标与策略在行动前所需的动态信息保持一致，而非与下一观测的内容保持一致。实验结果……

    arXiv:2606.25421v2 Announce Type: replace  Abstract: Recent studies on world modeling for Large Language Model (LLM) agents typically formulate the learning objective as next-observation prediction. However, this objective ties supervision to what a transition happens to reveal, which may omit the dynamics most relevant to the agent's current decision. To bridge this gap, we propose Agent-Authored World Modeling (AAWM), a training procedure that constructs supervision from the policy's own decision needs. Specifically, at each state, the agent identifies what it needs to understand about the environment before acting. These needs drive the retrieval of relevant transition evidence across trajectories, which is then synthesized into training targets that capture decision-oriented dynamics instead of reconstructing the next observation. This aligns the training objective with the dynamics the policy needs before acting, not with the contents of the next observation. Experimental results 
    
[^314]: 超越自我反思：一种用于智能体经验学习的执行-提炼-验证框架

    Beyond Self-Reflection: An Execute-Distill-Verify Framework for Agentic Experience Learning

    [https://arxiv.org/abs/2606.24428](https://arxiv.org/abs/2606.24428)

    提出EDV框架，通过异构智能体独立执行任务、第三方提炼器提取候选经验、执行者一致验证的方式构建离线经验，有效避免了大语言模型智能体在自我反思中将执行错误固化为指导的“自我确认陷阱”问题。

    

    经验学习使大语言模型智能体能够通过外部记忆复用经验教训，而无需更新模型参数。然而，反思过程可能会认可执行错误并将其存储为未来指导，我们将这种失败模式称为“自我确认陷阱”。我们提出了EDV，一个用于离线经验构建的执行-提炼-验证框架。异构智能体独立执行每个任务；第三方提炼器比较它们的轨迹和环境反馈以提取候选经验教训；执行者分别审查每个候选经验，仅保留获得一致通过的经验。评估采用从冻结记忆库中检索信息的单一求解器。在τ²-bench、Mind2Web和MMTB上的实验表明，重访任务的性能得到提升，并有额外证据表明在τ²-bench上可实现向保留任务的域内迁移。在一个带标注的Retail子集上，EDV将自我确认陷阱率从11.0%（原文截断）

    arXiv:2606.24428v2 Announce Type: replace  Abstract: Experience learning lets large language model agents reuse lessons through external memory without updating model parameters. However, reflection can endorse execution errors and store them as future guidance, a failure mode we call the Self-Confirmation Trap. We propose EDV, an Execute-Distill-Verify framework for offline experience construction. Heterogeneous agents independently execute each task; a third-party distiller compares their trajectories and environmental feedback to extract candidate lessons; and the executors separately review each candidate, retaining only unanimously approved lessons. Evaluation uses a single solver retrieving from the frozen memory bank. Experiments on $\tau^2$-bench, Mind2Web, and MMTB show improved revisited-task performance, with additional evidence of within-domain transfer to held-out tasks on $\tau^2$-bench. On an annotated Retail subset, EDV reduces the Self-Confirmation Trap Rate from 11.0%
    
[^315]: AutoSpecNER：用于车辆规格提取的细粒度命名实体识别数据集

    AutoSpecNER: A Fine-Grained Named Entity Recognition Dataset for Vehicle Specification Extraction

    [https://arxiv.org/abs/2606.24387](https://arxiv.org/abs/2606.24387)

    该论文推出了专家标注的车辆广告细粒度命名实体识别数据集AutoSpecNER（含659条广告、15类实体、超万条标注），并通过基准测试表明微调的DeBERTa以90%的微F1分数显著优于规则方法和大语言模型。

    

    车辆广告包含丰富的规格信息，但汽车领域的命名实体识别（NER）资源仍然有限。我们推出了AutoSpecNER，这是一个由专家标注的数据集，用于车辆挂牌信息中的细粒度实体识别。该数据集包含来自一个热门汽车销售网站的659条广告，标注了超过10,000个实体，涵盖15个类别，包括MODEL（车型）、ENGINE_SPEC（发动机规格）和BATTERY_CAPACITY（电池容量）。标注质量通过标注者间一致性进行验证，平均得分达到91.5%。我们对基于规则的提取方法、微调的Transformer编码器和大语言模型进行了基准测试。DeBERTa取得了最佳性能，微F1分数达到90%，优于基于规则的基线（43%）和最强的大语言模型（77.8%）。

    arXiv:2606.24387v2 Announce Type: replace  Abstract: Vehicle advertisements contain rich specification information, but automotive NER resources remain limited. We introduce AutoSpecNER, an expert-annotated dataset for fine-grained entity recognition in vehicle listings. The dataset includes 659 advertisements from a popular car-selling website, with over 10,000 entities annotated across 15 categories, including MODEL, ENGINE_SPEC, and BATTERY_CAPACITY. Annotation quality was validated through inter-annotator agreement, achieving an average score of 91.5%. We benchmark rule-based extraction, fine-tuned transformer encoders, and large language models. DeBERTa achieves the best performance with a 90% micro-F1 score, outperforming the rule-based baseline (43%) and the strongest large language model (77.8%).
    
[^316]: TMax：面向终端智能体的简单配方

    Tmax: A simple recipe for terminal agents

    [https://arxiv.org/abs/2606.23321](https://arxiv.org/abs/2606.23321)

    TMax 提出一个简单的开源强化学习配方，通过新颖的数据生成分类法和仅基于结果的 RL 训练，使仅 9B 参数的小型开源模型在 Terminal-Bench 2.0 上取得 33% 的成绩，超越了此前规模大得多的模型。

    

    使用终端的智能体（terminal agents）已迅速成为语言模型（LM）一项热门且重要的下游应用。尽管其应用广泛，但由于基准测试难度大、数据匮乏以及缺少简单的基线配方，学术界对基于强化学习（RL）训练此类模型的研究相对较少。我们提出了 TMax，一个面向终端智能体的开源 RL 配方，使小型开源模型更接近前沿水平。尽管方法简单，我们的配方仅用 9B 参数就在 Terminal-Bench 2.0 上取得了 33% 的成绩，超越了此前工作中规模大得多的模型。我们的配方首先使用一种新颖的分类法来生成数据，结合了难度控制、人物角色设定和验证器多样化，使我们能够低成本地为 RL 和 SFT 训练生成大量终端环境。随后，我们使用自建数据通过 RL 训练开源权重模型，采用简单的、仅基于最终结果的配方，并进行少量调整以提升训练稳定性。我们的配方适用于更多……（摘要截断）

    arXiv:2606.23321v2 Announce Type: replace  Abstract: Terminal-using agents have quickly become a popular and important downstream application of language models (LMs). Despite their prevalence, relatively little academic work has examined RL-based training of these models, due to difficult benchmarks, a lack of data, and a lack of simple baseline recipes. We present TMax, an open RL recipe for terminal agents that brings small open models closer to the frontier. While simple, our recipe achieves 33% on Terminal-Bench 2.0 with only 9B parameters, outperforming much larger models from prior work. Our recipe first involves generating data using a novel taxonomy, combining difficulty control, personas, and verifier diversification, which allows us to cheaply generate large amounts of terminal environments for RL and SFT training. We then train open-weight models using RL with our data, using a simple, outcome-only recipe, with small adjustments to improve stability. Our recipe works for mo
    
[^317]: AtomMem：通过原子事实为LLM智能体构建简单有效的记忆系统

    AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts

    [https://arxiv.org/abs/2606.19847](https://arxiv.org/abs/2606.19847)

    提出AtomMem长期记忆系统，通过事实执行器从长篇交互中提取高价值原子事实，并将其组织成分层事件结构和时间档案，为LLM智能体实现高价值密度存储与稳定的记忆演化。

    

    大语言模型（LLM）展现出强大的推理和生成能力，但其固定的上下文窗口限制了跨多轮会话交互中信息的长期积累与复用。现有的记忆增强系统往往以粗糙且不稳定的方式构建记忆，依赖于低效的记忆表示或不稳定的无约束更新。为应对这些挑战，我们提出了AtomMem，一个旨在实现高价值密度存储和稳定记忆演化的长期记忆系统。AtomMem引入了一个事实执行器，从长篇交互中有选择地提取高价值原子事实，作为高效的记忆表示。随后，AtomMem将这些事实组织成分层的事件结构和时间档案，捕捉连贯的情景上下文，并跟踪随时间动态演化的用户属性。在检索过程中，系统激活一个联想记忆图以……

    arXiv:2606.19847v2 Announce Type: replace  Abstract: Large language models (LLMs) demonstrate strong reasoning and generation abilities, but their fixed context windows limit long-term information accumulation and reuse across multi-session interactions. Existing memory-augmented systems often construct memory in a coarse and unstable manner, relying on inefficient memory representations or unstable unconstrained updates. To address these challenges, we propose AtomMem, a long-term memory system designed for value-dense storage and stable memory evolution. AtomMem introduces a Fact Executor, which selectively extracts high value atomic facts from long form interactions to serve as highly efficient memory representations. Subsequently, AtomMem organizes these facts into hierarchical event structures and temporal profiles, capturing coherent episodic contexts and tracking dynamically evolving user attributes over time. During retrieval, the system activates an associative memory graph to
    
[^318]: 面向局部与全局注意力的双重维度

    Dual Dimensionality for Local and Global Attention

    [https://arxiv.org/abs/2606.18587](https://arxiv.org/abs/2606.18587)

    提出距离自适应表示（DAR），在局部上下文窗口内保留全维度KV表示，而对窗口外的远距离token仅使用1/2或1/4维度的表示，以反映局部与远距离token重要性不对称的特性。

    

    仅解码器（Decoder-only）Transformer在先前token的KV缓存上计算注意力。Keys和Values通常以相同的维度表示，而无论它们与预测目标的距离远近。然而，在自然语言中，下一个词受紧邻的前序token影响最大。我们假设局部token与远距离token对表示能力的需求是不对称的：局部token对预测即时输出更为关键，因而需要更丰富的表示；而远距离token主要充当长程记忆，较低维度的表示可能已经足够。我们将这一想法形式化为距离自适应表示，它在局部上下文窗口内保留全维度表示，同时为窗口之外的token分配降维后的表示（例如原始维度的1/2或1/4）。在多个预训练……

    arXiv:2606.18587v2 Announce Type: replace  Abstract: Decoder-only Transformers compute attention over the KV cache of preceding tokens. Keys and Values are typically represented with the same dimensionality, regardless of their distance from the prediction target. In natural language, however, the next word is most strongly influenced by the immediately preceding tokens. We hypothesize that local and distant tokens impose asymmetric demands on representational capacity: local tokens are more critical for predicting immediate outputs and thus require richer representations, whereas distant tokens primarily serve as long-range memory, for which lower-dimensional representations may suffice. We formalize this idea as Distance-Adaptive Representation (DAR), which preserves full-dimensional representations within a local context window while assigning reduced-dimensional representations (e.g. 1/2 or 1/4 of the original dimensionality) to tokens beyond that window. Across multiple pretrainin
    
[^319]: 检测意大利媒体中的历史转折点：一种基于复杂系统的历时性新闻语料库分析方法

    Detecting Historical Turning Points in Italian Media: A Complex Systems Approach to a Diachronic News Corpus

    [https://arxiv.org/abs/2606.14348](https://arxiv.org/abs/2606.14348)

    该研究构建了意大利《共和报》1985-2000年间约60万篇文章的历时性新闻语料库，结合自然语言处理与复杂系统、统计物理学方法，在词汇和语义层面定量检测媒体话语中的历史转折点。

    

    引言：大规模文本语料库的日益可得性，为利用自然语言处理（NLP）开展数据驱动的定量历史分析开辟了新的可能性。然而，来自前数字时代且具有历史意义的历时性语料库仍然稀缺且往往不完整。方法：我们提出了一种定量历史分析方法，该方法基于对意大利报纸《共和报》（La Repubblica）约60万篇文章构成的历时性语料库的重建与探索，涵盖1985年1月1日至2000年12月31日期间发表的所有文章——这是意大利乃至全球在政治、社会和地缘政治方面发生重大变革的时期。我们使用NLP技术在词汇和语义两个层面对文本进行分析，随后运用复杂系统和统计物理学的工具来追踪媒体话语随时间的演变。结果：语料库中的词汇使用（摘要在此处不完整）

    arXiv:2606.14348v2 Announce Type: replace-cross  Abstract: Introduction: The increasing availability of large-scale textual corpora has opened new possibilities for data-driven, quantitative approaches to historical analysis using Natural Language Processing (NLP). However, diachronic corpora with historical relevance from the pre-digital era remain scarce and often incomplete.   Methods: We present a quantitative approach to historical analysis based on the reconstruction and exploration of a diachronic corpus of around 600,000 articles from the Italian newspaper "La Repubblica", covering all the articles published from the first of January 1985 to the 31st of December 2000 - a period of major political, social, and geopolitical change in Italy and globally. Using NLP techniques, we analyze the text at both lexical and semantic levels; we then apply tools from complex systems and statistical physics to trace shifts in media discourse over time.   Results: The word usage in the corpus 
    
[^320]: 生产级LLM系统中的外部经验服务：一项面向部署的质量-成本权衡研究

    External Experience Serving in Production LLM Systems: A Deployment-Oriented Study of Quality-Cost Trade-offs

    [https://arxiv.org/abs/2606.11806](https://arxiv.org/abs/2606.11806)

    该研究将生产级LLM系统中外部经验的服务方式建模为面向部署的质量-成本权衡问题，并发现当经验变得案例相关时，基于检索的选择性注入相比全局注入能提供更优的质量-成本运行点。

    

    生产级LLM系统会积累可复用的运营经验，但实际的部署问题并不仅仅是这些经验能否带来帮助，而是不同的服务策略如何在现实约束下权衡任务质量与在线服务成本。注入外部经验可以提升任务质量，但同时也会增加提示词负担、延迟以及服务压力。我们将“外部经验服务”研究为一个面向部署的质量-成本权衡问题。我们在一个真实的生产内容审核场景中评估了这一问题，并以工具使用和GPQA作为辅助对照任务，用以揭示不同的输出成本模式。我们比较了无经验基线、随机经验对照、全局提示注入以及基于检索的选择性注入，并对任务质量与服务成本两方面进行了分析。结果表明，一旦经验变得依赖于具体案例，选择性检索能够比（其他策略）提供更强的运行工作点。

    arXiv:2606.11806v2 Announce Type: replace  Abstract: Production LLM systems accumulate reusable operational experience, but the practical deployment issue is not merely whether such experience can help. It is how different serving strategies trade off quality against online cost under realistic constraints. Injecting external experience can improve task quality, yet it also increases prompt burden, latency, and serving pressure. We study \textit{external experience serving} as a deployment-oriented quality-cost trade-off problem. We evaluate this question in a real production moderation setting, with tool-use and GPQA as supporting contrast tasks that expose different output-cost regimes. We compare no-experience baselines, random experience controls, global prompt injection, and retrieval-based selective injection, and analyze both task quality and serving cost. The results show that, once experience becomes case-dependent, selective retrieval provides a stronger operating point than 
    
[^321]: 智能体技能评估与演化：框架与基准

    Agent Skill Evaluation and Evolution: Frameworks and Benchmarks

    [https://arxiv.org/abs/2606.11435](https://arxiv.org/abs/2606.11435)

    本综述系统梳理了智能体技能的评估与演化方法，提出四种技能演化范式（执行反馈、轨迹蒸馏、压缩和强化学习），并分析了六类以技能为中心的基准测试以推动技能研究发展。

    

    智能体技能的增长改变了智能体系统构建、评估和部署的方式。随着技能库规模的不断扩大，严格的评估对于确保其在现实应用中的实用性、质量和安全性变得至关重要。因此，该领域正在经历一种新兴的范式转变，即从孤立的技能创建转向自动化、评估驱动的技能演化。在这篇综述中，我们系统地审视了超越基础技能创建的技能演化与评估的全貌。我们将技能演化归纳为四种不同的范式，涵盖执行反馈、轨迹蒸馏、压缩和强化学习，并展示了每种范式如何有助于提升技能的实用性和可靠性。我们还对六类以技能为中心的基准测试进行了分析，识别了基准覆盖范围、权衡取舍和指标丰富性方面的结构性差距，以推动技能研究的发展。最后，我们识……（原文摘要在此处截断）

    arXiv:2606.11435v2 Announce Type: replace  Abstract: The growth of agent skills has transformed how agentic systems are built, evaluated, and deployed. As skill libraries continue to scale, rigorous evaluation becomes critical to ensuring their utility, quality, and safety in real-world applications. Consequently, the field is undergoing an emerging paradigm shift from isolated skill creation to automated, evaluation-driven skill evolution. In this survey, we systematically examine the landscape of skill evolution and evaluation beyond foundational skill creation. We categorize evolution into four distinct paradigms, spanning execution feedback, trajectory distillation, compression, and reinforcement learning, showing how each element contributes to improving skill utility and reliability. We also provide an analysis of six skill-centric benchmark categories, identifying structural gaps in benchmark coverage, trade-offs, and metric richness to advance skill research. Finally, we identi
    
[^322]: UXBench：AI 助手用户体验基准测试

    UXBench: Benchmarking User Experience in AI Assistants

    [https://arxiv.org/abs/2606.09570](https://arxiv.org/abs/2606.09570)

    提出了首个基于真实用户反馈信号、以用户为中心的体验评估基准 UXBench，通过三个相互关联的任务评估 AI 助手的偏好对齐与对话生成能力。

    

    随着 AI 助手每天为数百万用户提供服务，超越通用模型能力来评估用户体验（UX）变得日益重要。我们提出了 UXBench，这是首个基于真实用户反馈信号、以用户为中心的基准，用于评估偏好对齐和对话生成能力。该基准由三个相互关联的任务组成：UX Judge、UX Eval 和 UX Recovery，包含从某主流中文 AI 助手超过 7 万条交互日志中提取的 7,400 个测试实例。该数据集紧密反映真实用户分布，涵盖 8 种场景、83 个领域，以及带来严峻挑战的多样化失败模式。在 26 个前沿语言模型上进行的大量实验，为模型感知用户体验的能力以及模型能力提升如何促进更好的对话参与提供了新的见解。通过对模型行为和性能差距的分析，我们记录了六项重要发现。

    arXiv:2606.09570v5 Announce Type: replace  Abstract: As AI assistants serve millions of users daily, evaluating user experience (UX) beyond general model capability has become increasingly important. We present \textbf{UXBench}, the first user-centric benchmark grounded in real user feedback signals for evaluating preference alignment and dialogue generation. The benchmark consists of three interconnected tasks, UX Judge, UX Eval, and UX Recovery, with 7,400 test instances extracted from over 70K interaction logs of a mainstream Chinese AI assistant. The dataset closely reflects real user distributions, covering 8 scenarios, 83 domains, and diverse failure patterns that pose severe challenges. Extensive experiments on 26 frontier language models provide novel insights into how well models perceive user experience and how improvements in model capability contribute to better dialogue engagement. Through analyses of model behavior and performance gaps, we document six important findings,
    
[^323]: 当“无正确答案”也是正确答案时：诊断视频理解中多模态大语言模型的“缺席答案”检测能力

    When No Answer Is Correct: Diagnosing Absent Answer Detection for MLLMs in Video Understanding

    [https://arxiv.org/abs/2606.08239](https://arxiv.org/abs/2606.08239)

    本研究首次系统诊断了多模态大语言模型在视频理解中的“缺席答案”检测能力，发现当正确答案不在候选集时，模型倾向于选择貌似合理的干扰项而非识别出无有效选项，且该缺陷在时间推理任务和更密集的帧采样下更为严重。

    

    arXiv:2606.08239v2 公告类型：replace-cross 摘要：多模态大语言模型（MLLMs）在视频理解方面已取得重大进展，但其回答的可靠性仍未得到充分探索。本工作对多模态大语言模型在视频理解中的“缺席答案”检测能力进行了诊断性研究：正确答案被有意从候选集中排除，而一个可靠的模型应当能够识别出不存在有效的选项。我们在三种设置下评估了缺席答案检测行为：带有“以上皆非”选项的多选题、带有检测指令的开放式生成，以及不带任何指导的标准评估。在涵盖多种模型和基准测试的实验中，我们发现多模态大语言模型压倒性地倾向于选择貌似合理的干扰项，而非检测出答案缺席的情况。这种失败在时间推理任务中表现得更为明显，且随着帧采样的加密而进一步恶化。我们进一步探索了思维链提示方法（原文在此处截断）。

    arXiv:2606.08239v2 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLMs) have made substantial advancements in video understanding, yet the reliability of their responses remains underexplored. This work presents a diagnostic study of absent answer detection for MLLMs in video understanding, where the correct answer is deliberately excluded from the candidate set and a reliable model is expected to recognize that no valid option exists. We evaluate the absent answer detection behavior under three settings: multiple-choice questions augmented with a ``None of the Above'' option, open-ended generation with a detection instruction, and standard evaluation without any guidance. Across a diverse set of models and benchmarks, we find that MLLMs overwhelmingly select plausible distractors rather than detecting the absent answer. This failure is more pronounced in temporal reasoning tasks and worsens with denser frame sampling. We further explore chain-of-thought pro
    
[^324]: Paediatric-HGNN：一种通过多尺度声学融合检测儿童言语不流畅的混合异构图神经网络

    Paediatric-HGNN: A Hybrid Heterogeneous Graph Neural Network for Detecting Disfluency in Children's Speech via Multiscale Acoustic Fusion

    [https://arxiv.org/abs/2606.08210](https://arxiv.org/abs/2606.08210)

    该论文提出Paediatric-HGNN框架，利用异构图神经网络建模词节点与声学帧节点之间的层次化交互并结合多尺度声学融合，有效区分儿童病理性口吃与典型发育性不流畅，达到82.4%的加权准确率，为早期临床干预提供更稳健可解释的工具。

    

    自动口吃检测（ASD）系统在处理儿童语音时面临困难，原因在于发育中的声音具有较高的声学变异性，且病理性口吃与典型发育性不流畅之间的界限十分微妙。我们提出了Paediatric-HGNN，这是一个采用专为儿童数据定制的上下文感知部分-整体交互网络的框架。我们的方法摒弃了传统的一维信号建模，而是构建一个异构图，以捕捉词汇单元（词节点）与细粒度声学片段（帧节点）之间的层次关系。在精选的儿童语料库（UCLASS和FluencyBank）上训练后，Paediatric-HGNN实现了82.4%的加权准确率和0.386的典型不流畅F1分数。对层次化词汇-声学交互的建模能够捕捉发育性的“搜寻”行为，为早期临床干预提供了更稳健且可解释的工具。

    arXiv:2606.08210v2 Announce Type: replace-cross  Abstract: Automated stuttering detection (ASD) systems struggle with paediatric speech due to high acoustic variability in developing voices and the subtle distinction between pathological stuttering and typical developmental disfluencies. We introduce Paediatric-HGNN, a framework using a Context-aware Part-whole Interaction Network (CaPIN) tailored for paediatric data. Instead of conventional 1D signal modelling, our approach builds a heterogeneous graph capturing hierarchical relationships between lexical units (word nodes) and fine-grained acoustic segments (frame nodes). Trained on curated paediatric corpora (UCLASS and FluencyBank), Paediatric-HGNN achieves 82.4% weighted accuracy and a Typical Disfluency F1-score of 0.386. Modelling hierarchical lexical-acoustic interactions captures developmental "searching" behaviour, offering a more robust and interpretable tool for early clinical intervention.
    
[^325]: TRACER：用于生成式推荐中概念擦除的标记重新分配方法

    TRACER: Token ReAssignment for Concept ERasure in Generative Recommendation

    [https://arxiv.org/abs/2606.07688](https://arxiv.org/abs/2606.07688)

    提出TRACER端到端概念遗忘框架，通过将概念相关物品重新分配到替代标记而非直接抑制共享SID，解决了生成式推荐中概念擦除与推荐效用保持之间的冲突。

    

    生成式推荐将下一项预测表述为对源自用户历史交互的语义ID（SID）序列的自回归生成，这使得现代推荐系统在结构上与大语言模型（LLM）相似。随着隐私和安全问题的日益凸显，这些系统越来越需要概念遗忘技术来移除与物品相关的敏感或有害概念。然而，现有的LLM遗忘方法无法直接应用于生成式推荐。与具有明确语义的词元不同，SID是抽象标识符，通常同时被需要遗忘的物品和需要保留的物品所共享，这导致概念移除与推荐效用保持之间存在严重的冲突。为应对这一挑战，我们提出了TRACER，一个基于标记重新分配的端到端概念遗忘框架。TRACER并非直接抑制共享的SID，而是将概念相关的物品重新分配到替代的……

    arXiv:2606.07688v2 Announce Type: replace-cross  Abstract: Generative recommendation formulates next-item prediction as autoregressive generation over semantic ID (SID) sequences derived from users' historical interactions, making modern recommender systems structurally similar to large language models (LLMs). As privacy and safety concerns grow, these systems increasingly require concept unlearning to remove sensitive or harmful concepts associated with items. However, existing LLM unlearning methods cannot be directly applied to generative recommendation. Unlike word tokens with explicit semantics, SIDs are abstract identifiers that are often shared by both forget and retain items, leading to severe conflicts between concept removal and recommendation utility preservation.   To address this challenge, we propose TRACER, an end-to-end concept unlearning framework based on token reassignment. Rather than directly suppressing shared SIDs, TRACER reassigns concept-related items to altern
    
[^326]: 人类ALMANAC：面向智能体协作的动作级心智模型标注人类协作数据集

    Humans' ALMANAC: A Human Collaboration Dataset of Action-Level Mental Model Annotations for Agent Collaboration

    [https://arxiv.org/abs/2606.06388](https://arxiv.org/abs/2606.06388)

    该论文提出了ALMANAC数据集，基于经典的Map Task双人对话任务构建，提供人类协作过程中动作级的心智模型标注，弥补了当前缺乏可用于引导智能体获得协作能力的真实人类协作数据的空白。

    

    近年来，大语言模型（LLM）智能体的进展使其具备了多步推理、规划和工具使用等复杂认知能力，这些智能体日益成为人类工作流程中的协作伙伴。然而，有效的协作要求协作者在协作过程中持续维护并对齐关于自身推理、伙伴意图和共同目标的心智模型。当今的智能体很少发展出这类能力，因为它们主要针对任务完成进行优化，而非维护协作过程所需的底层心智模型。此外，研究界缺乏带有动作级心智模型标注的真实人类协作数据，而这些数据本可以引导智能体获得协作能力。为弥补这一空白，我们提出了ALMANAC——一个面向智能体协作的动作级心智模型标注数据集，该数据集基于经典的Map Task双人路线描述任务构建……

    arXiv:2606.06388v3 Announce Type: replace-cross  Abstract: Recent advances in LLM agents have enabled complex cognitive capabilities, such as multi-step reasoning, planning, and tool use, that increasingly position these agents as collaborative partners in human workflows. Effective collaboration, however, requires collaborators to continuously maintain and align mental models of their own reasoning, partners' intentions, and shared goals during the collaborative process. Today's agents rarely develop such capabilities since they are primarily optimized for task completion rather than for maintaining the underlying mental models required for collaborative processes. Moreover, the community lacks authentic human collaboration data with action-level mental model annotations that could guide agents toward collaborative competence. To bridge this gap, we present ALMANAC, a dataset of Action-Level Mental model ANnotations for Agent Collaboration built from the Map Task, a classic dyadic rou
    
[^327]: 当证据稀疏时：对话与LLM智能体轨迹中的弱监督早期失败预警

    When Evidence is Sparse: Weakly Supervised Early Failure Alerting in Dialogs and LLM-Agent Trajectories

    [https://arxiv.org/abs/2606.05414](https://arxiv.org/abs/2606.05414)

    该论文提出一种仅用轨迹级结果标签训练、基于注意力多示例学习（MIL）的轻量级外部监测器，能够在失败指示证据稀疏且出现较晚的情况下，对对话和LLM智能体轨迹实现早期失败预警。

    

    早期失败预警需要在对话或智能体轨迹展开的过程中，判断其是否可能失败。现有方法大致遵循两个方向。基于LLM的不确定性监测要么从执行模型自身引出置信度，将可靠性与其自我评估绑定，要么在每个前缀处查询外部的尖端评判模型，导致代价高昂的重复推理。而训练式监测器要么依赖逐轮标注（这种标注很少可得），要么训练每个前缀去预测最终结果，却没有显式建模失败指示证据出现的位置。我们通过一个仅从轨迹级结果训练的轻量级外部监测器来解决这些局限，其动机来自五个基准测试中共同呈现的一种模式：与失败高度相关的轮次既稀疏又往往出现较晚。其基于注意力的多示例学习（MIL）组件从轨迹结果中学习轮次级别的贡献，并且

    arXiv:2606.05414v2 Announce Type: replace  Abstract: Early failure alerting requires deciding, while a dialog or agent trajectory unfolds, whether it is likely to fail. Existing approaches broadly follow two directions. LLM-based uncertainty monitoring either elicits confidence from the acting model, tying reliability to its own self-assessment, or queries an external frontier judge at each prefix, incurring costly repeated inference. Trained monitors instead either rely on per-turn annotations, which are rarely available, or train every prefix to predict the terminal outcome without explicitly modeling where failure-indicating evidence appears. We address these limitations with a lightweight external monitor trained only from trajectory-level outcomes, motivated by a pattern across five benchmarks: turns rated highly relevant to failure are sparse and often late. Its attention-based multiple instance learning (MIL) component learns turn-level contributions from trajectory outcomes and
    
[^328]: 基于激活的上下文学习主动学习：挑战与见解

    Activation-Based Active Learning for In-Context Learning: Challenges and Insights

    [https://arxiv.org/abs/2606.05134](https://arxiv.org/abs/2606.05134)

    本研究探索了基于MLP激活的深度主动学习方法来优化LLM上下文示例选择，但得出负面结论：激活信号（无论从大规模激活还是前四阶矩角度观察）与示例质量和任务性能的相关性很弱，Spearman相关系数最高仅0.33。

    

    深度主动学习此前已被探索用于大语言模型的上下文样本选择，但尚未有方法利用Transformer激活理解的最新进展。本文中，我们检验了这样一个假设：模型激活可以为优化上下文示例的选择提供细粒度的信号。我们对应用于上下文学习的基于MLP激活的深度主动学习方法进行了全面分析，包括不同的注意力掩码策略如何影响跨多种分类和生成数据集的主动学习，实验使用了Llama-3.2-3B和Qwen2.5-3B两个基础模型。然而，我们发现了一个负面结果：通过大规模激活或前四阶矩的视角观察，MLP和嵌入层的输出与示例质量或任务性能并不相关。具体来说，在我们测试的所有任务和模型中，绝对Spearman相关系数最高仅为0.33。

    arXiv:2606.05134v2 Announce Type: replace  Abstract: Deep active learning has previously been explored for LLM in-context sample selection, but not with methods that utilise recent advances in understanding of transformer activations. In this paper, we test the hypothesis that model activations could provide a fine-grained signal to optimise the selection of in-context examples. We present a comprehensive analysis of MLP activation-based deep active learning methods applied to in-context learning, including how different attention masking strategies impact active learning across diverse classification and generative datasets, using both Llama-3.2-3B and Qwen2.5-3B base models. However, we find a negative result: MLP and embedding layer outputs, viewed through the lenses of massive activations or the first four moments, do not correlate with example quality or task performance. Specifically, the absolute Spearman correlation coefficient is at most 0.33 for all tasks and models we tested
    
[^329]: 跨领域与跨模型的AI生成英文文本检测中语言特征的系统性分析

    A Systematic Analysis of Linguistic Features in AI-Generated English Text Detection Across Domains and Models

    [https://arxiv.org/abs/2606.04177](https://arxiv.org/abs/2606.04177)

    该研究通过覆盖284个可解释语言特征、30个大语言模型和10个文本领域的大规模实证分析，发现仅基于语言特征即可可靠检测AI生成文本，且词汇丰富度是唯一在跨模型、跨领域场景下保持稳健的判别信号。

    

    可解释的语言特征为解释一段文本为何呈现机器生成特征提供了一种有前景的方法，尤其对非专业用户而言。然而，关于哪些特征能可靠指示大语言模型（LLM）生成文本的已有研究结论，在不同特征集、模型和文本领域之间仍然碎片化。为填补这一空白，我们开展了一项大规模实证研究，评估用于刻画AI生成文本的语言信号的稳健性。我们的分析涵盖284个可解释的语言特征，在跨模型和跨领域泛化设置下，对来自30个大语言模型和10个文本领域的输出进行评估。研究表明，仅基于语言特征的分类器能够可靠地区分AI生成文本与人类撰写的文本。然而，许多此前提出的指标被证明高度依赖于具体上下文，唯有词汇丰富度度量在不同模型家族和文本领域中始终保持稳健。这些结果表明……

    arXiv:2606.04177v2 Announce Type: replace  Abstract: Interpretable linguistic features offer a promising approach for explaining why a given text appears machine-generated, particularly for non-expert users. However, existing findings on which features reliably indicate LLMgenerated text remain fragmented across feature sets, models, and text domains. To address this gap, we conduct a large-scale empirical study assessing the robustness of linguistic signals for characterizing AI-generated text. Our analysis covers 284 interpretable linguistic features across outputs from 30 LLMs and 10 text domains under cross-model and cross-domain generalization settings. We show that classifiers based solely on linguistic features can reliably distinguish AI-generated from human-written text. However, many previously proposed indicators prove strongly context-dependent, with the exception of measures of lexical richness, which remain robust signals across model families and text domains. These resu
    
[^330]: 神经检索器是否偏好某些文档？学习到的相关性先验的证据

    Do Neural Retrievers Prefer Certain Documents? Evidence of Learned Relevance Priors

    [https://arxiv.org/abs/2606.02814](https://arxiv.org/abs/2606.02814)

    研究发现有监督神经检索器在训练过程中会隐式学习到与查询无关的文档级相关性先验，这种先验导致先验值较低的文档即使真正相关也系统性地更难被检索到。

    

    神经检索器通过标注的查询-文档对进行训练，以估计查询与文档之间的相关性。然而，标注协议可能并不能纯粹地反映相关性：标注过程仅选取文档的一个子集进行标注，而这种选择可能偏好某些文档类型。我们研究了有监督的双编码器检索器是否隐式地学习到了文档级别的相关性先验——这是模型在标注数据上训练时，作为副作用编码在其表示空间中的一种与查询无关的信号。我们通过在冻结的文档嵌入上训练简单的分类器来估计这一先验，并在多个信息检索基准上评估了三个最先进的检索器。我们发现，有监督神经检索器编码的相关性先验能够泛化到未见过的文档，并且在不同模型之间保持一致。这些先验与一种“可发现性差距”相关：先验值较低的文档即使真正相关，也系统性地更难被检索到。

    arXiv:2606.02814v2 Announce Type: replace-cross  Abstract: Neural retrievers are trained to estimate query-document relevance from annotated query-document pairs. Yet annotation protocols may not purely reflect relevance: they select only a subset of documents for labeling, and this selection can favor certain document types over others. We investigate whether supervised bi-encoder retrievers implicitly learn a document-level relevance prior: a query-independent signal encoded in their representation space as a side effect of training on annotated data. We estimate this prior by training simple classifiers on frozen document embeddings and evaluate three state-of-the-art retrievers across multiple IR benchmarks. We find that supervised neural retrievers encode relevance priors that generalize to unseen documents and are consistent across models. These priors are associated with a findability gap: documents with lower prior are systematically harder to retrieve, even when genuinely rele
    
[^331]: 面向已发布医学视觉-语言模型的选择性链接精确度

    Selective Linkage Precision for Released Medical Vision-Language Models

    [https://arxiv.org/abs/2606.02276](https://arxiv.org/abs/2606.02276)

    本文提出“选择性链接精确度”指标，揭示攻击者利用元数据属性过滤结合公开医学视觉-语言模型，能够以极高精确度（最高67.68%，加入采集年份后达99.05%）将去标识化胸部X光片与受限报告重新链接，证明传统全池召回率指标严重低估了医学图像发布的隐私泄露风险。

    

    医院可能会发布去标识化的胸部X光片，同时限制对其报告的访问。已经拥有同一人群报告及其元数据的攻击者可以利用视觉-语言模型（VLM）将图像与报告进行匹配。我们提出了选择性链接精确度这一指标，用于在通过可用属性过滤候选者之后，衡量最高置信度查询中的正确匹配比例。在8个公开VLM和3个胸部X光片数据集上的实验表明，全池Recall@1的最大值仅为0.40%。然而，该指标是在没有属性过滤的情况下对所有查询求平均的结果。在ReXGradient-160K数据集上，MedSigLIP-448在按性别、年龄段和投照体位过滤后，在预算为100的情况下达到67.68%的精确度，中位候选报告数为327份。在相同的合格查询集上，仅使用属性的策略达到24.99%，而仅使用模型仅达到1.98%。加入采集年份信息（仅在3个数据集中的1个可用）后，精确度可达到99.05%。

    arXiv:2606.02276v2 Announce Type: replace-cross  Abstract: Hospitals may release de-identified chest radiographs while restricting access to their reports. An attacker who already has reports from the same population and their metadata can use a vision-language model (VLM) to match images to reports. We introduce selective linkage precision, which measures correct matches among the highest-confidence queries after filtering candidates by the available attributes. Across 8 public VLMs and 3 chest radiograph datasets, the maximum whole-pool Recall@1 is 0.40%. However, this metric averages over every query without attribute filtering. On ReXGradient-160K, MedSigLIP-448 reaches 67.68% precision at a budget of 100 after filtering by sex, age band, and projection, with a median of 327 candidate reports. On the same eligible query set, an attribute-only policy reaches 24.99% and the model alone reaches 1.98%. Adding acquisition year, available in only 1 of the 3 datasets, yields 99.05% precis
    
[^332]: 重新思考逐步模型路由：一种成本高效的表格推理视角

    Rethinking Stepwise Model Routing: A Cost-Efficient Table Reasoning Perspective

    [https://arxiv.org/abs/2605.29319](https://arxiv.org/abs/2605.29319)

    本文发现表格推理步骤中表格标记与文本标记具有不同的不确定性分布，并提出表格感知的逐步路由框架EcoTab，通过分别建模两类标记的不确定性来实现成本高效的表格推理。

    

    大型推理模型在表格推理任务上表现出色，但由于冗长的推理轨迹而产生高昂的推理成本。逐步模型路由通过动态地将推理步骤分配给较小或较大的模型来缓解这一问题。然而，面向表格推理的逐步模型路由仍未得到充分探索。通过实证分析，我们发现涉及表格的推理步骤包含两类具有不同不确定性分布的标记：一类是基于表格结构的表格标记，例如单元格值和表头；另一类是表示周围自然语言推理的文本标记。这两类标记的不确定性与模型在下一个推理步骤中出错的风险相关。然而，现有方法未能对二者分别建模，导致次优的路由决策。为解决这一问题，我们提出了EcoTab，一个面向高效表格推理的表格感知逐步路由框架。

    arXiv:2605.29319v2 Announce Type: replace  Abstract: Large Reasoning Models (LRMs) achieve strong performance on table reasoning tasks but incur substantial inference cost due to long reasoning traces. Stepwise model routing mitigates this issue by dynamically assigning reasoning steps to smaller or larger models. However, stepwise model routing for table reasoning remains underexplored. Through empirical analysis, we find that reasoning steps involving tables contain two types of tokens with distinct uncertainty distributions: table tokens grounded in table structure, such as cell values and headers, and text tokens representing surrounding natural-language reasoning. The uncertainty of both token types is correlated with the risk that the model makes an error in the next reasoning step. However, existing methods fail to model them separately, leading to suboptimal routing decisions. To address this, we propose EcoTab, a table-aware stepwise routing framework for efficient table reaso
    
[^333]: 面向逐步模型路由的评分准则引导过程奖励

    Rubric-Guided Process Reward for Stepwise Model Routing

    [https://arxiv.org/abs/2605.29310](https://arxiv.org/abs/2605.29310)

    提出RoRo框架，用评分准则引导的过程奖励替代仅反映最终答案正确性的结果奖励，从而更好地评估逐步模型路由中的中间决策并提升性能与泛化能力。

    

    逐步模型路由通过将每个推理步骤分配给合适的模型，从而提高大型推理模型（LRMs）的效率。最近的方法将路由建模为序贯决策过程，并使用强化学习训练路由器。然而，尽管这些方法将路由建模为一个过程，它们仍然使用结果奖励来监督路由器。这类奖励仅反映最终答案的正确性，无法评估中间的路由决策，这可能会削弱性能和泛化能力。为解决这一缺口，我们提出了RoRo，一个用于逐步模型路由的评分准则引导的过程奖励框架。RoRo首先收集多样化的路由轨迹，并基于结果、成本和过程质量构建偏好对。随后，它通过交替优化训练一个Rubricor来生成针对特定查询的评估准则，以及一个Judge在该准则下对路由轨迹进行评分。由此得到的过程奖励……

    arXiv:2605.29310v2 Announce Type: replace-cross  Abstract: Stepwise model routing improves the efficiency of Large Reasoning Models (LRMs) by assigning each reasoning step to a suitable model. Recent methods formulate routing as a sequential decision process and train the router with reinforcement learning. However, although they model routing as a process, they still supervise the router with outcome rewards. Such rewards only reflect final answer correctness and fail to evaluate intermediate routing decisions, which can weaken performance and generalization. To address this gap, we propose RoRo, a rubric-guided process reward framework for stepwise model routing. RoRo first collects diverse routing trajectories and constructs preference pairs based on outcome, cost, and process quality. It then trains a Rubricor to generate a query-specific evaluation rubric and a Judge to score routing trajectories under this rubric through alternating optimization. The resulting process rewards are
    
[^334]: 一种用于评估手术反馈质量的多实例大语言模型框架

    A Multi-Instance LLM Framework for Rating the Quality of Surgical Feedback

    [https://arxiv.org/abs/2605.25440](https://arxiv.org/abs/2605.25440)

    该论文提出了一种基于大语言模型的两阶段多实例提示框架，通过对候选质量标准进行重复独立采样与整合并注入外科领域知识，自动发现可解释的手术反馈质量评估标准，解决了手术培训中口头反馈质量难以评估的问题。

    

    主治外科医生在手术室中提供的口头反馈对住院医师学员的技能习得起着关键的形成性作用。然而，在真实手术过程中评估培训者反馈的质量及其影响学员行为的有效性仍然是一项挑战。先前的研究依赖专家人工评分者进行大量手动标注来评估反馈内容，并专注于构建宽泛的分类体系，而忽略了反馈传递的定性方面，例如清晰度或紧迫性。现有为数不多的自动化方法（包括关键词分析和主题建模）也无法捕捉这些细微的方面。我们提出了一个基于大语言模型的两阶段框架，该框架能够在手术培训的背景下发现可解释的反馈质量评估标准。我们的方法通过重复独立采样候选标准并加以整合的方式使用多实例提示，并注入外科领域知识……

    arXiv:2605.25440v2 Announce Type: replace  Abstract: Verbal feedback delivered by attending surgeons in the operating room plays a critical formative role in resident trainee skill acquisition. Yet, assessing the quality of trainer feedback and its effectiveness in influencing trainee behavior during live surgery remains a challenge. Prior studies assessed feedback content relying on extensive manual annotation by expert human raters and focused on developing broad taxonomies that overlook the qualitative aspects of feedback delivery such as clarity or urgency. Limited existing automated methods, including keyword analysis and topic modeling, also fail to capture these nuanced aspects. We introduce a two-stage LLM-based framework that discovers interpretable feedback quality criteria grounded in the context of surgical training. Our method uses multi-instance prompting via repeated independent sampling of candidate criteria followed by consolidation, and surgical domain knowledge injec
    
[^335]: 二胜于一：一种无崩溃的多奖励RLIF训练框架

    Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework

    [https://arxiv.org/abs/2605.22620](https://arxiv.org/abs/2605.22620)

    该论文提出了一种无崩溃的多奖励RLIF无监督训练框架，将基于聚类投票的答案级奖励与基于词元自确信度的补全级奖励相结合，并通过GDPO归一化和KL-Cov正则化有效避免奖励破解与熵坍缩，从而提升大语言模型的推理能力。

    

    基于可验证奖励的强化学习（RLVR）已显著提升了大语言模型（LLM）的推理能力，但其通常依赖于来自人工标注或标准答案的外部监督。基于内部反馈的强化学习（RLIF）最近作为一种可扩展的无监督替代方案出现，它利用从模型自身提取的信号进行训练。然而，现有的RLIF方法通常依赖单一内部奖励，这可能导致奖励破解、熵坍缩以及推理结构退化。我们提出了一种多奖励RLIF框架，将训练信号分解为两个互补的组成部分：基于聚类投票的答案级奖励和基于逐词元自确信度的补全级奖励。为了稳健地结合这些信号，我们采用基于GDPO的归一化方法来缓解奖励尺度的不平衡。我们进一步引入了KL-Cov正则化，针对低熵词元分布进行约束……

    arXiv:2605.22620v2 Announce Type: replace-cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning ability of LLMs, but often depends on external supervision from human annotations or gold-standard solutions. Reinforcement learning from internal feedback (RLIF) has recently emerged as a scalable unsupervised alternative, using signals extracted from the model itself. However, existing RLIF methods typically rely on a single internal reward, which can lead to reward hacking, entropy collapse, and degraded reasoning structure. We propose a multi-reward RLIF framework that decomposes the training signal into two complementary components: an answer-level reward based on cluster voting and a completion-level reward based on token-wise self-certainty. To combine these signals robustly, we apply GDPO-based normalization to reduce reward-scale imbalance. We further introduce KL-Cov regularization, which targets low-entropy token distributi
    
[^336]: 监控内心独白：探针轨迹揭示推理动态

    Monitoring the Internal Monologue: Probe Trajectories Reveal Reasoning Dynamics

    [https://arxiv.org/abs/2605.18549](https://arxiv.org/abs/2605.18549)

    提出通过在推理过程每个 token 上评估探针来构建“探针轨迹”，并提取波动性、趋势等信号处理特征刻画其时间动态，从而在静态探针表现最薄弱的场景下平均提升 21.7 个 AUROC 点，显著增强对大型推理模型未来行为的预测与安全监控能力。

    

    大型推理模型通过其思维链（Chain-of-Thought, CoT）推理为安全监控提供了新的机会。然而，CoT 并不总是忠实于模型的最终输出，这削弱了其作为监控工具的可靠性。为解决这一问题，我们研究了大型推理模型的隐藏表示，以确定能否从提示和 CoT 表示中预测其未来行为。通过在生成的每个 token 上评估探针，我们构建了“探针轨迹”，即某个概念的概率在推理过程中的连续演变。为了刻画这些时间动态特征，我们提取了能够捕捉波动性、趋势和稳态行为的信号处理特征。我们发现，轨迹特征恰好在静态探针表现最薄弱的地方发挥作用，平均带来 21.7 个 AUROC 点的提升，并且在静态探针低于 70 的所有情况下均有所改进。我们还提出了两点方法论见解。首先，模板……（原文摘要在此处截断）

    arXiv:2605.18549v2 Announce Type: replace  Abstract: Large Reasoning Models (LRMs) introduce new opportunities for safety monitoring through their Chain-of-Thought (CoT) reasoning. However, CoT is not always faithful to the model's final output, undermining its reliability as a monitoring tool. To address this, we investigate the hidden representations of LRMs to determine whether future behavior can be predicted from prompt and CoT representations. By evaluating a probe at each generated token, we construct a probe trajectory, the continuous evolution of a concept's probability across the reasoning process. To characterize these temporal dynamics, we extract signal-processing features that capture volatility, trend, and steady-state behavior. We find that trajectory features help precisely where the static probe is weakest, adding 21.7 AUROC points on average, and improving in every such case where that probe falls below 70. We also present two methodological insights. First, template
    
[^337]: 莫比乌斯注入：从一次性提示注入到AI智能体基础设施中的持续资源耗尽

    Mobius Injection: From One-Shot Prompt Injection to Sustained Resource Exhaustion in AI Agent Infrastructure

    [https://arxiv.org/abs/2605.11442](https://arxiv.org/abs/2605.11442)

    本文提出“莫比乌斯注入”攻击，通过“语义闭环”机制将单次提示注入放大为智能体组件的持续递归执行，把LLM智能体武器化为僵尸节点，从而发动新型“基于且面向智能体的DDoS（AbO-DDoS）”攻击，实现对AI智能体基础设施的持续资源耗尽。

    

    大语言模型（LLM）智能体已成为关键中介，协调着人类用户与各类数字服务及LLM基础设施之间的复杂交互。尽管以往研究已广泛考察了LLM与智能体在孤立状态下的安全性，但智能体在“用户-智能体-服务”链条中充当破坏性“枢纽”所带来的系统性风险在很大程度上被忽视了。在这项工作中，我们通过提出莫比乌斯注入揭示了一种全新的威胁范式——这是一种精密的攻击，它将自主智能体武器化为“僵尸节点”，进而发动我们定义的“基于智能体且面向智能体的分布式拒绝服务”攻击。莫比乌斯注入通过一种名为“语义闭环”的独特结构机制，将已知的间接提示注入与持久化智能体组件组合起来，仅需单次注入事件即可诱导智能体组件的持续递归执行。我们证明该攻击对攻击者而言是轻量级的（摘要在此处被截断）。

    arXiv:2605.11442v3 Announce Type: replace-cross  Abstract: Large Language Model (LLM) agents have emerged as key intermediaries, orchestrating complex interactions between human users and a wide range of digital services and LLM infrastructures. While prior research has extensively examined the security of LLMs and agents in isolation, the systemic risk of the agent acting as a disruptive "hub" within the user-agent-service chain remains largely overlooked. In this work, we expose a novel threat paradigm by introducing Mobius Injection, a sophisticated attack that weaponizes autonomous agents into zombie nodes to launch what we define as Agent-based and -Oriented DDoS (AbO-DDoS) attacks. Mobius Injection composes known indirect prompt injection and persistent agent components through a distinct structural mechanism named Semantic Closure, which induces sustained recursive execution of agent components through a single injection event. We demonstrate that this attack is lightweight for 
    
[^338]: MulTaBench：面向文本与图像的多模态表格学习基准测试

    MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image

    [https://arxiv.org/abs/2605.10616](https://arxiv.org/abs/2605.10616)

    提出了MulTaBench——一个包含40个数据集（图像-表格与文本-表格任务各半）的多模态表格学习基准，专注于模态提供互补预测信号的任务，并证明了对嵌入进行任务特定微调优于使用冻结预训练嵌入。

    

    表格基础模型通过利用预训练学习数值型和类别型结构化数据的可泛化表示，近来在监督表格学习领域建立了最先进的水平。然而，它们对文本和图像等非结构化模态缺乏原生支持，只能依赖冻结的预训练嵌入来处理这些模态。在现有的多模态表格学习基准上，我们证明了对嵌入进行任务特定的微调能够提升性能。然而，现有基准往往只关注模态的简单共现，这导致跨数据集结果的高方差，并掩盖了任务特定微调带来的收益。为弥补这一空白，我们提出了MulTaBench，一个包含40个数据集的基准，其中图像-表格任务和文本-表格任务各占一半。我们聚焦于模态能够提供互补预测信号、且通用嵌入会丢失关键信息的预测任务。

    arXiv:2605.10616v3 Announce Type: replace-cross  Abstract: Tabular Foundation Models have recently established the state of the art in supervised tabular learning, by leveraging pretraining to learn generalizable representations of numerical and categorical structured data. However, they lack native support for unstructured modalities such as text and image, and rely on frozen, pretrained embeddings to process them. On established Multimodal Tabular Learning benchmarks, we show that tuning the embeddings to the task improves performance. Existing benchmarks, however, often focus on the mere co-occurrence of modalities; this leads to high variance across datasets and masks the benefits of task-specific tuning. To address this gap, we introduce MulTaBench, a benchmark of 40 datasets, split equally between image-tabular and text-tabular tasks. We focus on predictive tasks where the modalities provide complementary predictive signal, and where generic embeddings lose critical information, 
    
[^339]: LLM 服务中的分支并行节奏调控

    Pacing Branch Parallelism in LLM Serving

    [https://arxiv.org/abs/2605.06914](https://arxiv.org/abs/2605.06914)

    提出了 PACE——一个逐步接纳控制器，通过预测“分支外部性”是否处于批次当前的松弛预算内来动态调控 LLM 服务中的分支并行宽度，从而在吞吐量与共批处理请求性能之间取得平衡。

    

    近期的方法在 LLM 输出中暴露了请求内并行性，使独立分支能够并发解码。现有的服务系统要么急切地执行这些分支，要么在固定上限下执行。我们证明这两种方式都很脆弱：急切接纳会膨胀共享的解码步长，从而降低串行阶段中共批处理请求的性能；而保守的固定上限则放弃了暴露分支本应带来的吞吐量。我们将被接纳分支所造成的额外步长延迟称为“分支外部性”，并表明安全宽度取决于批次组成、上下文长度和累积的松弛量，而这些因素在整个工作负载轨迹中都在持续变化。我们提出了 PACE，一个逐步接纳控制器，它将额外分支视为机会性的宽度，只有当预测的分支外部性能够容纳在批次当前的松弛预算之内时才予以接纳。逐步调控之所以切实可行，是因为分支级调度……

    arXiv:2605.06914v2 Announce Type: replace-cross  Abstract: Recent methods expose intra-request parallelism in LLM outputs, allowing independent branches to decode concurrently. Existing serving systems execute these branches eagerly or under fixed caps. We show that both are brittle: eager admission inflates the shared decode step, degrading co-batched requests in serial stages, while conservative fixed caps forgo the throughput that motivated exposing branches in the first place. We call the excess step latency caused by admitted branches the \emph{branch externality} and show that the safe width depends on batch composition, context lengths, and accumulated slack, all of which change continuously over a workload trace. We introduce PACE, a per-step admission controller that treats extra branches as opportunistic width, admitted only when the predicted branch externality fits within the batch's current slack budget. Per-step regulation is practical because branch-level scheduling deco
    
[^340]: 文本语料库作为概念场：黑盒幻觉与新颖性度量

    Text Corpora as Concept Fields: Black-Box Hallucination and Novelty Measurement

    [https://arxiv.org/abs/2605.05103](https://arxiv.org/abs/2605.05103)

    本文提出了一种基于语料库概念场的黑盒方法，通过句子嵌入空间中的局部漂移场来度量幻觉和新颖性，无需模型内部信息且可归因于语料库。

    

    我们引入了文本语料库的**概念场**：一个具有逐点不确定性的局部漂移场，在句子嵌入空间中通过连续句子之间的增量来估计。给定一个候选句子转换，我们通过$\zeta$（观测增量与场的局部高斯估计之间的平均绝对z距离）来评分其与场的一致性。该评分是黑盒的（无需模型内部信息）、可归因于语料库的（每个评分都可追溯到附近的语料库句子），并在局部高斯近似下具有概率动机的解释。我们通过引入**向量序列数据库（VSDB）**来支持计算，该数据库存储嵌入以及序列位置和下一增量元数据。我们在两个大规模场景中评估了这种方法：对美国联邦法规中的幻觉式接地检测，以及对古腾堡计划文本的新颖性检测。

    arXiv:2605.05103v3 Announce Type: replace-cross  Abstract: We introduce the \textbf{Concept Field} of a text corpus: a local drift field with pointwise uncertainty, estimated in sentence-embedding space from the deltas between consecutive sentences. Given a candidate sentence transition, we score its agreement with the field by $\zeta$, the mean absolute z-distance between the observed delta and the field's local Gaussian estimate. The score is black-box (no model internals), corpus-attributable (every score traces to nearby corpus sentences), and admits a probabilistically motivated interpretation under a local Gaussian approximation. We support the computation with the introduction of a \textbf{Vector Sequence Database (VSDB)} that stores embeddings together with sequence-position and next-delta metadata. We evaluate this approach on two large-scale settings: hallucination-style groundedness detection over the U.S. Code of Federal Regulations, and novelty detection over Project Guten
    
[^341]: 基础模型解锁来自全国医疗理赔数据的真实世界证据

    Foundation Models to Unlock Real-World Evidence from Nationwide Medical Claims

    [https://arxiv.org/abs/2605.02740](https://arxiv.org/abs/2605.02740)

    本研究基于 MarketScan 数据库中超过 2 亿参保人的 438 亿条理赔记录构建了生成式基础模型 ReClaim，通过预训练联合学习诊断、手术、药物和医疗支出，在预测 1,000 多种疾病时显著优于疾病特异性模型和 Transformer 基线，并能成功泛化到电子健康记录数据集。

    

    arXiv:2605.02740v3 公告类型：replace-cross 摘要：真实世界数据日益为监管和临床决策提供依据。行政理赔数据作为其主要来源之一，在整个参保期间持续跟踪患者跨医疗机构的医疗照护及其费用，但在医疗健康基础模型领域却基本未被探索。在此，我们提出了 ReClaim，这是一系列生成式基础模型，基于 MarketScan 研究数据库中来自超过 2 亿参保人的 438 亿条理赔记录构建。ReClaim 通过从零开始的预训练联合学习诊断、手术操作、药物和医疗支出，随后通过后训练针对下游预测任务进行专门化，训练数据截止于 2022 年底。在预测超过 1,000 种疾病时，ReClaim 实现了 0.769 的平均 ROC-AUC，优于疾病特异性的梯度提升模型（0.663）和基于 Transformer 的基线模型（0.694），并且在泛化到两个独立的跨域电子健康记录数据集时仍保持了这一优势。

    arXiv:2605.02740v3 Announce Type: replace-cross  Abstract: Real-world data increasingly inform regulatory and clinical decisions. Administrative claims, among their major sources, continuously track care and its cost across providers throughout enrollment, yet remain largely unexplored for healthcare foundation models. Here we present ReClaim, a family of generative foundation models built on 43.8 billion claims records from more than 200 million enrollees in the MarketScan Research Databases. ReClaim learns diagnoses, procedures, medications and expenditure jointly through pre-training from scratch, then specializes for downstream prediction tasks through post-training, with a training cutoff at the end of 2022. In predicting more than 1,000 diseases, ReClaim achieved a mean ROC-AUC of 0.769, outperforming disease-specific gradient-boosted models (0.663) and a transformer-based baseline (0.694), and retained this advantage when generalizing to two independent cross-domain electronic h
    
[^342]: 面向特许会计师的AI问答：一种检索增强推理框架

    AI-Enabled Question-Answering for Chartered Accountancy: A Retrieval-Augmented Reasoning Framework

    [https://arxiv.org/abs/2605.00257](https://arxiv.org/abs/2605.00257)

    提出了参数高效的检索增强推理框架CA-ThinkFlow，通过结合4比特量化的DeepSeek-R1推理模型与版面感知的文档提取系统，在资源受限条件下提升了AI在印度特许会计师等复杂专业领域问答任务中的可靠性。

    

    大语言模型（LLM）的出现推动了人工智能在金融领域的应用，但它们在复杂且特定司法管辖区的任务（如印度特许会计师CA考试）中的可靠性仍然有限。这些模型在执行需要多个步骤的数值任务时表现困难，同时还需要对法律法规的深入理解，而且其扩展方式在资源受限的环境中并不可行。我们提出了CA-ThinkFlow，这是一种参数高效的检索增强生成（RAG）框架，它基于一个经过4比特量化的140亿参数推理模型（14B-DeepSeek-R1）运行，并采用具备版面感知能力的Docling提取系统，在提取过程中保持文档结构。CA-ThinkFlow使用一种基础的RAG方法，自动将检索到的信息添加到提示词中，同时依靠模型内置的思维链（CoT）功能来构建上下文并引导推理……

    arXiv:2605.00257v2 Announce Type: replace  Abstract: The inception of Large Language Models (LLMs) has catalyzed AI adoption in the finance sector, yet their reliability in complex, jurisdiction-specific tasks like Indian Chartered Accountancy (CA) remains limited. The models display difficulty in executing numerical tasks which require multiple steps while also needing advanced knowledge about legal regulations and the method of scaling their operations is not feasible in settings which have limited access to resources. We present CA-ThinkFlow as a parameter-efficient Retrieval-Augmented Generation (RAG) framework which operates with a 14B, 4-bit-quantized reasoning model, 14B-DeepSeek-R1, and a layout-aware Docling extraction system which maintains document structure during extraction. CA-ThinkFlow uses a basic RAG method which automatically adds retrieved information into the prompt, while it depends on the model's built-in Chain-of-Thought (CoT) functions to create context and prod
    
[^343]: 语言模型的上下文线性激活导向

    Contextual Linear Activation Steering of Language Models

    [https://arxiv.org/abs/2604.24693](https://arxiv.org/abs/2604.24693)

    CLAS将激活导向分解为固定的导向方向和依赖上下文的导向系数，使导向强度能随输入上下文自适应变化，并实现了单次生成中针对不同部分引导多种行为的选择性导向，在多项基准测试中持续优于固定强度的传统线性激活导向方法。

    

    线性激活导向是一种强大的方法，它通过在推理时向大型语言模型的激活中添加概念的线性表示，来控制和专门化这些模型。虽然行之有效，但现有方法通常以固定强度进行导向，而不考虑任何给定输入实际需要多大程度的导向。在这项工作中，我们引入了上下文线性激活导向（CLAS），它将激活导向分解为一个固定的导向方向和一个依赖上下文的导向系数。这种分解在保留导向方向可解释性的同时，使导向强度能够随上下文而变化。它还使CLAS能够执行选择性导向：在单次生成的不同部分引导多种行为，这是使用固定导向强度无法实现的。此外，在广泛的导向基准测试和语言模型上，CLAS始终优于标准的线性……

    arXiv:2604.24693v2 Announce Type: replace  Abstract: Linear activation steering is a powerful approach for controlling and specializing large language models by adding linear representations of concepts to their activations at inference. While effective, existing methods often steer with a fixed strength, independent of how much steering any given input requires. In this work, we introduce Contextual Linear Activation Steering (CLAS), which decomposes activation steering into a fixed steering direction and a context-dependent steering coefficient. This decomposition preserves the interpretability of the steering direction while enabling the steering strength to vary with context. It also enables CLAS to perform selective steering: steering multiple behaviors across different portions of a single generation, which cannot be achieved using a fixed steering strength. In addition, across a wide range of steering benchmarks and language models, CLAS consistently outperforms standard linear 
    
[^344]: 大语言模型函数调用的不确定性量化

    Uncertainty Quantification for LLM Function-Calling

    [https://arxiv.org/abs/2604.22985](https://arxiv.org/abs/2604.22985)

    本文首次对大语言模型函数调用场景下的不确定性量化方法进行了评估，发现在自然语言问答任务中表现优异的语义熵等基于熵的方法在函数调用场景中并不适用。

    

    大语言模型（LLM）正日益被部署用于自主解决现实世界的任务。实现这一目标的关键要素是LLM函数调用范式，这是一种被广泛用于赋予大语言模型工具使用能力的方法。然而，大语言模型错误地调用函数可能会造成严重后果，尤其是在其影响不可逆转的情况下，例如转账或删除数据。因此，在执行函数调用之前，评估大语言模型对该函数调用能够正确解决任务的置信度至关重要。不确定性量化（UQ）方法可用于量化这种置信度，并防止潜在的错误函数调用。在本工作中，我们提出了据我们所知首个针对大语言模型函数调用（FC）的不确定性量化方法评估。虽然多样本不确定性量化方法（如语义熵）在自然语言问答任务中表现出色，但我们发现在函数调用场景中，基于熵的……

    arXiv:2604.22985v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are increasingly deployed to autonomously solve real-world tasks. A key ingredient for this is the LLM Function-Calling paradigm, a widely used approach for equipping LLMs with tool-use capabilities. However, an LLM calling functions incorrectly can have severe implications, especially when their effects are irreversible, e.g., transferring money or deleting data. Hence, it is of paramount importance to consider the LLM's confidence that a function call solves the task correctly prior to executing it. Uncertainty Quantification (UQ) methods can be used to quantify this confidence and prevent potentially incorrect function calls. In this work, we present what is, to our knowledge, the first evaluation of UQ methods for LLM Function-Calling (FC). While multi-sample UQ methods, such as Semantic Entropy, show strong performance for natural language Q\&A tasks, we find that in the FC setting the entropy-based 
    
[^345]: ReflectMT：内化反思以实现高效高质量的机器翻译

    ReflectMT: Internalizing Reflection for Efficient and High-Quality Machine Translation

    [https://arxiv.org/abs/2604.19144](https://arxiv.org/abs/2604.19144)

    ReflectMT提出一种两阶段反思内化框架，通过强化学习将“翻译-反思-精炼”能力内化到模型中，采用“后置思考”范式在WMT24上以86.16的GRF分数超越DeepSeek-R1，同时大幅降低推理成本和延迟。

    

    近年来，将大型推理模型（LRM）应用于机器翻译（MT）的研究日益受到关注。虽然大多数方法采用“预先思考”范式并受益于显式的推理轨迹，但它们存在推理成本高昂和延迟较大的问题。为了解决这些局限性，我们提出了ReflectMT，一个采用“后置思考”范式的两阶段反思内化机器翻译框架。我们的方法通过强化学习培养模型的“翻译-反思-精炼”能力。在第一阶段，我们培养模型进行高质量反思和精炼的能力，从而增强其语义理解和任务特定知识。在第二阶段，我们训练模型将反思过程中获得的知识进行内化。在WMT24基准测试上，ReflectMT取得了基于GPT-4o的GRF分数86.16，比DeepSeek-R1（84.03）高出2.13分，同时使用约……（摘要原文在此处截断）

    arXiv:2604.19144v2 Announce Type: replace  Abstract: Recent years have witnessed growing interest in applying Large Reasoning Models (LRMs) to Machine Translation (MT). While most approaches adopt a 'pre-thinking' paradigm and benefit from explicit reasoning trajectories, they suffer from substantial inference cost and latency. To address these limitations, we propose ReflectMT, a two-stage reflection internalization framework for machine translation that employs a 'post-thinking' paradigm. Our approach develops the model's 'translate-reflect-refine' capability through reinforcement learning. In the first stage, we cultivate the model's capacity for high-quality reflection and refinement, thereby enhancing its semantic comprehension and task-specific knowledge. In the second stage, we train the model to internalize the knowledge acquired during reflection. On WMT24, ReflectMT achieves a GPT-4o-based GRF score of 86.16, outperforming DeepSeek-R1 (84.03) by 2.13 points, while using appro
    
[^346]: 心理咨询对话中下一对话行为预测的转移矩阵正则化方法

    Transition-Matrix Regularization for Next Dialogue Act Prediction in Counselling Conversations

    [https://arxiv.org/abs/2604.18539](https://arxiv.org/abs/2604.18539)

    本文提出一种基于KL散度的转移矩阵正则化方法，将语料库中的对话流统计先验融入下一对话行为预测，在心理咨询对话任务上使宏观F1相对提升9%至42%，且该改进可跨语言和跨领域迁移。

    

    本文研究如何将经验性的对话流统计信息融入下一对话行为预测（NDAP）任务。文中提出了一种KL正则化项，用于将预测的对话行为分布与从语料库中提取的转移模式对齐。在一个包含60个类别的德语心理咨询分类体系上，采用5折交叉验证进行评估，结果表明该方法根据编码器的不同可使宏观F1相对提升9%至42%，并显著改善了对话流的对齐程度。在HOPE数据集上的跨数据集验证显示，该改进能够跨语言、跨心理咨询领域迁移。通过对多种预训练编码器和架构进行系统性消融实验，研究发现转移正则化能够带来一致的增益，且对较弱的基线模型有格外显著的提升作用。研究结果表明，轻量级的语篇流先验知识可以与预训练编码器形成有效互补，尤其适用于细粒度、数据稀疏的对话任务。

    arXiv:2604.18539v2 Announce Type: replace  Abstract: This paper studies how empirical dialogue-flow statistics can be incorporated into Next Dialogue Act Prediction (NDAP). A KL regularization term is proposed that aligns predicted act distributions with corpus-derived transition patterns. Evaluated on a 60-class German counselling taxonomy using 5-fold cross-validation, this improves macro-F1 by 9--42% relative depending on encoder and substantially improves dialogue-flow alignment. Cross-dataset validation on HOPE suggests that improvements transfer across languages and counselling domains. In systematic ablations across pretrained encoders and architectures, the findings indicate that transition regularization provides consistent gains and disproportionately benefits weaker baseline models. The results suggest that lightweight discourse-flow priors complement pretrained encoders, especially in fine-grained, data-sparse dialogue tasks.
    
[^347]: 语言模型的碎片整理：一种基于可解释性的词表扩展方法

    Defragmenting Language Models: An Interpretability-based Approach for Vocabulary Expansion

    [https://arxiv.org/abs/2604.16656](https://arxiv.org/abs/2604.16656)

    该论文针对LLM在非拉丁文字语言中token过度碎片化的问题，提出了基于可解释性的词表扩展方法，系统研究了“应添加哪些词条”和“如何初始化对应嵌入”这两个核心决策。

    

    所有语言都是平等的；但在分词方面，有些语言比其他语言更平等。Token是决定当代大语言模型（LLM）访问成本和延迟的隐藏货币。然而，许多使用非拉丁文字书写的语言面临着糟糕的“汇率”：LLM编码许多语言中相同信息所需的token数量是英语的数倍。我们观察到，这种被称为“token过度碎片化”的问题在现代开源权重LLM中依然存在。标准的解决方案是词表扩展，即向模型词表中添加缺失的目标语言词条。我们关注词表扩展过程中的两个核心决策：应该添加哪些词条？以及应该如何初始化它们对应的输入和输出嵌入？我们围绕这两个核心决策，全面研究并推进了基于可解释性的词表扩展这一新研究方向，做出了四项关键贡献：首先，我们……（原文摘要至此被截断）

    arXiv:2604.16656v2 Announce Type: replace  Abstract: All languages are equal; when it comes to tokenization, some are more equal than others. Tokens are the hidden currency that dictates the cost and latency of contemporary LLM access. However, many languages written in non-Latin scripts observe a poor exchange rate: LLMs take several multiples of tokens to encode the same information in many languages as they do for English. We observe that this issue, known as token overfragmentation, persists in modern open-weight LLMs. The standard remedy is vocabulary expansion that adds target language items missing from the model's vocabulary. We focus on two core decisions in the vocabulary expansion process: What items should we add? and How should we initialize their corresponding input and output embeddings? We comprehensively study and advance the new research direction of interpretability-based vocabulary expansion along these two core decisions, making four key contributions: First, we qu
    
[^348]: IndicDBBench——面向印度语言的多语言文本到SQL能力基准测试

    IndicDBBench - Benchmarking Multilingual Text-to-SQL Capabilities in Indian Languages

    [https://arxiv.org/abs/2604.13686](https://arxiv.org/abs/2604.13686)

    该论文提出了IndicDBBench，一个基于真实政府开放数据、通过三智能体流水线构建的覆盖英语及六种印度语言共15,612个任务、20个数据库的大型多语言Text-to-SQL基准，填补了现有基准在非西方语言语境下的评测空白。

    

    尽管大型语言模型（LLM）已显著提升了文本到SQL的性能，但现有基准测试大多聚焦于西方语境，在真实世界的非西方应用方面留下了空白。我们提出IndicDBBench，这是一个全面的多语言文本到SQL基准，旨在评估跨越多种印度语系语言的跨语言语义解析能力。其底层数据库模式源自真实世界的开放数据平台（NDAP和IDP），确保了真实的结构复杂性。IndicDBBench包含20个数据库，涵盖237张表。为了从非规范化的政府数据中构建复杂的关系结构，我们采用了三智能体流水线（架构师、审计员、精炼者），实现了较高的关系密度（每个数据库平均11.85张表）以及最深达六层的连接深度。我们生成了涵盖七种语言（英语和六种印度语言）的15,612个任务，确保查询基于真实的数据库取值，并需要多步骤推理（原文摘要在此处截断）。

    arXiv:2604.13686v2 Announce Type: replace  Abstract: While Large Language Models (LLMs) have significantly advanced Text-to-SQL performance, existing benchmarks largely focus on Western contexts leaving a gap in real-world, non-Western applications. We introduce IndicDBBench, a comprehensive multilingual Text-to-SQL benchmark designed to evaluate cross-lingual semantic parsing across diverse Indic language families. The underlying schemas are derived from real-world open-data platforms (NDAP and IDP), ensuring realistic structural complexity. IndicDBBench comprises 20 databases spanning 237 tables. To construct complex relational structures from denormalized government data, we employ a three-agent pipeline (Architect, Auditor, Refiner), achieving high relational density (11.85 tables per database) and join depths of up to six. We generate 15,612 tasks spanning seven languages: English and six Indic languages, ensuring that queries are grounded in real database values and require multi
    
[^349]: 多模态视觉语言模型中的人类区域适应性

    Anthropogenic Regional Adaptation in Multimodal Vision-Language Model

    [https://arxiv.org/abs/2604.11490](https://arxiv.org/abs/2604.11490)

    本文提出了人类区域适应这一新范式以及简单有效的GG-EZ方法，通过区域数据过滤与模型合并，在保持视觉语言模型全局泛化能力的同时，优化其对特定区域（如东南亚）的适应性。

    

    尽管视觉语言（VL）领域在跨多种语言和领域整合视觉与文本信息方面取得了显著成功，但目前仍缺乏一个专门用于评估视觉语言系统中以人为中心的对齐性的框架。为填补这一空白，我们做出了两项贡献。首先，我们提出了人类区域适应性：这是一种新颖的范式，旨在优化模型与特定区域背景的相关性，同时确保保留全局泛化能力。其次，我们提出了一种简单而有效的适应性方法，名为地理泛化简易法（GG-EZ），该方法利用区域数据过滤和模型合并技术。通过对三种VL架构（大型视觉语言模型、文本到图像扩散模型和视觉语言嵌入模型）的全面实验，以及针对东南亚（SEA）区域适应性的案例研究，我们展示了……（摘要在此处截断）

    arXiv:2604.11490v3 Announce Type: replace-cross  Abstract: While the field of vision-language (VL) has achieved remarkable success in integrating visual and textual information across multiple languages and domains, there is still no dedicated framework for assessing human-centric alignment in vision-language systems. We offer two contributions to address this gap. First, we introduce Anthropogenic Regional Adaptation: a novel paradigm that aims to optimize model relevance to specific regional contexts while ensuring the retention of global generalization capabilities. Second, we present a simple, but effective adaptation method named Geographical-generalization-made-easy (GG-EZ), which utilizes regional data filtering and model merging. Through comprehensive experiments on 3 VL architectures: large vision-language models, text-to-image diffusion models, and vision-language embedding models, and a case study in Southeast Asia (SEA) regional adaptation, we demonstrate the importance of 
    
[^350]: 针对胸部X光片AI陈述的自动化接地验证

    Automated grounding verification of AI statements about chest radiographs

    [https://arxiv.org/abs/2604.09537](https://arxiv.org/abs/2604.09537)

    本文提出BRACE自动化框架，通过同时篡改报告和引用证据来构建含矛盾示例的训练数据，使接地验证器能识别出以往最先进测试无法发现的、与报告相矛盾的AI陈述。

    

    接地验证器根据放射学报告和引用的证据来核查关于患者图像的陈述。目前测试这类验证器的最先进方法是在患者报告保持不变的情况下移除或替换引用的证据。据我们所知，尚无现有方法能同时检查报告和引用证据。我们提出了构建报告与引用示例框架（Building Report And Citation Examples, BRACE），这是一个利用公开放射学报告和Radiopaedia文章自动构建训练示例的框架。我们在5个不同的公开胸部X光片数据集上，将一个词汇基线方法与42个参数量从0.1B到284B不等的验证器进行了比较。在接受0.90支持性测试示例的阈值下，使用BRACE训练的验证器对与报告相矛盾的AI生成陈述的平均接受率仅为0.239。而未使用矛盾示例训练的验证器虽然通过了最先进测试，却接受了大部分此类陈述（平均0.830）。

    arXiv:2604.09537v2 Announce Type: replace  Abstract: A grounding verifier checks a statement about a patient's image against the radiology report and the cited evidence. The state-of-the-art methods for testing these verifiers remove or replace the cited evidence while the patient's report stays unchanged. To our knowledge, no existing method checks both the report and the cited evidence. We present Building Report And Citation Examples (BRACE), an automated framework that builds training examples from public radiology reports and Radiopaedia articles. We compared a lexical baseline and 42 verifiers with 0.1-284B parameters on 5 different public chest radiograph datasets. At a threshold accepting 0.90 of the supported test examples, verifiers trained with BRACE accepted a mean of only 0.239 of AI-written statements contradicted by the report. Verifiers trained without contradicted examples passed the state-of-the-art tests, yet accepted most of these statements (0.830 on average). With
    
[^351]: SYN-DIGITS：用于校准数字孪生模拟的合成控制框架

    SYN-DIGITS: A Synthetic Control Framework for Calibrated Digital Twin Simulation

    [https://arxiv.org/abs/2604.07513](https://arxiv.org/abs/2604.07513)

    提出SYN-DIGITS框架，一种模型无关的事后校准方法，通过从数字孪生中学习问题间的关系并迁移到人类侧，在无目标问题人类回答数据的冷启动场景下有效校准大语言模型角色模拟的偏差。

    

    基于大语言模型的角色模拟（通常称为数字孪生模拟）正越来越多地被应用于市场研究、推荐系统和社会科学研究。然而，这些模拟相对于真实人类行为往往存在系统性偏差和校准失准问题。我们提出了SYN-DIGITS，一个模型无关的事后校准框架，它从数字孪生中学习问题之间的关系，并将其迁移用于预测人类回答。我们聚焦于一个冷启动场景：即没有观察到人类对目标问题的回答，因此校准必须依赖数字孪生数据以及人类在参考问题上的数据。我们的分析表明，尽管数字孪生可能无法准确复现个体的人类回答，但它们能够捕捉问题之间非平凡的关系，这些关系可以有效地迁移到人类一侧。当个体层面的数据不可用且我们只能获取……

    arXiv:2604.07513v2 Announce Type: replace-cross  Abstract: Persona simulation based on large language models---often referred to as digital twin simulation---is increasingly used for market research, recommender systems, and social science research. However, these simulations often exhibit systematic bias and miscalibration relative to real human behavior. We propose SYN-DIGITS, a model-agnostic, post-hoc calibration framework that learns relationships among questions from digital twins and transfers them to predict human responses. We focus on a cold-start setting in which no human response to the target question is observed, so calibration must rely on digital twins and human data on reference questions. Our analysis shows that, though digital twins may not reproduce individual human responses accurately, they can capture nontrivial relationships among questions that can be effectively transferred to the human side. When individual-level data is unavailable and we only have access to
    
[^352]: “Versteasch du mi？”（你听得懂我吗？）——关于生成式人工智能、大语言模型与非标准语言的计算语言学与社会语言学视角

    \textit{Versteasch du mi?} Computational and Socio-Linguistic Perspectives on GenAI, LLMs, and Non-Standard Language

    [https://arxiv.org/abs/2603.28213](https://arxiv.org/abs/2603.28213)

    本文融合批判社会语言学与计算语言学双重视角，批判性审视生成式AI和大语言模型对非标准语言及小语种的不公平对待，揭示其加深数字语言鸿沟并强化语言标准化认识论的问题。

    

    大语言模型（LLMs）和生成式人工智能（GenAI）的设计已被证明对使用人数较少的语言“不公平”（Petrov et al., 2023），并加深了数字语言鸿沟（Bella et al., 2023）。批判性社会语言学研究还指出，这些技术不仅是在先前语言标准化的社会历史进程（其根源往往植根于欧洲民族主义和殖民主义事业，Migge and Schneider, 2025）的基础上才成为可能，而且还加剧了一种将语言视为“单一、单语、句法上标准化的意义系统”的认识论（Schneider, 2024, p. 5）。在本文中，我们借鉴了关于技术与语言政策交叉领域的既有研究（Kelly-Holmes, 2019），并将我们在批判社会语言学与计算语言学方面的各自专长结合起来，对上述论点进行审视。我们选取了两个不同的非标准语言综合体……

    arXiv:2603.28213v3 Announce Type: replace  Abstract: The design of Large Language Models (LLMs) and generative artificial intelligence (GenAI) has been shown to be "unfair" to less-spoken languages (Petrov et al., 2023) and to deepen the digital language divide (Bella et al., 2023). Critical sociolinguistic work has also argued that these technologies are not only made possible by prior sociohistorical processes of linguistic standardisation, often grounded in European nationalist and colonial projects (Migge and Schneider, 2025), but also exacerbate epistemologies of language as "monolithic, monolingual, syntactically standardized systems of meaning" (Schneider, 2024, p. 5). In our paper, we draw on earlier work on the intersections of technology and language policy (Kelly-Holmes, 2019) and bring our respective expertise in critical sociolinguistics and computational linguistics to bear on an interrogation of these arguments. We take two different complexes of non-standard linguistic 
    
[^353]: 表征发生中的预设问题

    The Presupposition Problem in Representation Genesis

    [https://arxiv.org/abs/2603.21745](https://arxiv.org/abs/2603.21745)

    本文提出“表征发生预设问题”，指出大语言模型未经历从非表征系统到表征系统的发生转变，而思想语言假说、目的语义学、预测加工、生成主义和发生现象学等心灵哲学主要框架均以表征发生为前提，因而现有理论缺乏解释LLM在缺失发生时认知能力状况的概念资源。

    

    大语言模型是首批在没有明显经历“表征发生”的情况下实现高认知表现的系统。所谓表征发生，是指从非表征的物理系统转变为一个其状态能以内容敏感方式引导行为的系统的过程。以往的认知系统在我们能够考察这一转变之前就已经完成了它，而心灵哲学一直将“发生”视为背景条件而非解释目标。大语言模型提供了一个不明显涉及这一转变的案例，这使得发生问题重新变得紧迫：如果发生并未发生，哪些认知能力会受到影响，为什么？我们目前缺乏回答这一问题的概念资源。本文认为，其原因在于结构性的。心灵哲学的主要框架——包括思想语言假说、目的语义学、预测加工、生成主义和发生现象学——在应用于发生问题时都具有一个共同特征

    arXiv:2603.21745v3 Announce Type: replace-cross  Abstract: Large language models are the first systems to achieve high cognitive performance without clearly undergoing representation genesis: the transition from a non-representing physical system to one whose states guide behavior in a content-sensitive way. Prior cognitive systems had already made this transition before we could examine it, and philosophy of mind treated genesis as a background condition rather than an explanatory target. LLMs provide a case that does not clearly involve this transition, making the genesis question newly urgent: if genesis did not occur, which cognitive capacities are affected, and why? We currently lack the conceptual resources to answer this. The reason, this paper argues, is structural. Major frameworks in philosophy of mind, including the Language of Thought hypothesis, teleosemantics, predictive processing, enactivism, and genetic phenomenology, share a common feature when applied to the genesis 
    
[^354]: 受限资源下的表征获取

    Representational Acquisition Under Bounded Resources

    [https://arxiv.org/abs/2603.21736](https://arxiv.org/abs/2603.21736)

    本文在常数因子内精确刻画了精确概念学习任务中的存储-读取权衡曲线，证明单次读取的复合查询强制要求答案特定存储单元、单比特顺序无关更新强制要求事实特定存储单元，并给出了最小存储容量下坐标编码的鲁棒性界限以及保存旧内容所需满足的结构性约束。

    

    学习者必须保持已获取内容可用的义务带来了一种存储-访问权衡，其精确的端点会强制产生不同的物理组织形式。针对一个精确的概念学习任务，我们在常数因子范围内确定了存储-读取曲线。单次读取的复合查询强制要求答案特定的存储单元，并使更新的代价呈指数级增长；而单比特、顺序无关的更新则强制要求事实特定的存储单元。在最小存储容量下，一个鲁棒的对应结论成立：当更新拉伸在所有语义立方体的边上均匀平均时，较小的额外拉伸会在均匀内容分布下限制期望上与坐标编码的距离。它同时限制了单次读取的消费者所无法访问的信息量。保存旧内容会带来进一步的约束。在一个共同的不变就绪状态域上，通过普通零读取覆写所保存的旧内容必须经由未写入的坐标进行分解。分布式编码可能需要

    arXiv:2603.21736v3 Announce Type: replace-cross  Abstract: A learner's obligation to keep acquired content usable imposes a storage-access tradeoff, with exact endpoints that force different physical organizations. For an exact concept-learning task, we determine the storage-read curve within a constant factor. One-read compound queries force answer-specific cells and exponentially costly updates; one-bit, order-independent updates instead force fact-specific cells. At minimum storage capacity, a robust counterpart holds: when update stretch is averaged uniformly over all semantic cube edges, small excess stretch bounds expected distance from a coordinate encoding under the uniform content distribution. It also bounds the information inaccessible to a one-read consumer. Preservation imposes further constraints. On a common invariant ready-state domain, old content preserved by ordinary zero-read overwrites must factor through the unwritten coordinates. Distributed encodings can require
    
[^355]: EntropyCache：基于解码Token熵引导的扩散语言模型KV缓存

    EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models

    [https://arxiv.org/abs/2603.18489](https://arxiv.org/abs/2603.18489)

    提出了EntropyCache，一种无需训练的KV缓存方法，利用新解码token分布的最大熵作为恒定开销的信号来决定何时重新计算缓存，其决策开销与上下文长度和模型深度无关，从而高效加速扩散语言模型的推理。

    

    基于扩散的大语言模型（dLLMs）依赖双向注意力机制，这使其无法进行无损KV缓存，并且需要在每个去噪步骤执行完整的正向传播。现有的近似KV缓存方法通过选择性更新缓存状态来降低这一成本，但其决策开销会随上下文长度或模型深度增加而扩展。我们提出EntropyCache，这是一种无需训练的KV缓存方法，它利用新解码token分布的最大熵作为恒定开销的信号来决定何时重新计算。我们的设计基于两个经验观察：（1）解码token的熵与KV缓存漂移相关，为缓存陈旧性提供了一个低成本的代理指标；（2）解码token的特征波动性在解除掩码后会持续多个步骤，这促使我们对最近解码的k个token进行重新计算。跳过或重新计算的决策每步仅需O(V)的计算量，与上下文长度和模型深度无关。

    arXiv:2603.18489v2 Announce Type: replace  Abstract: Diffusion-based large language models (dLLMs) rely on bidirectional attention, which prevents lossless KV caching and requires a full forward pass at every denoising step. Existing approximate KV caching methods reduce this cost by selectively updating cached states, but their decision overhead scales with context length or model depth. We propose EntropyCache, a training-free KV caching method that uses the maximum entropy of newly decoded token distributions as a constant-cost signal for deciding when to recompute. Our design is grounded in two empirical observations: (1) decoded token entropy correlates with KV cache drift, providing a cheap proxy for cache staleness, and (2) feature volatility of decoded tokens persists for multiple steps after unmasking, motivating recomputation of the $k$ most recently decoded tokens. The skip-or-recompute decision requires only $O(V)$ computation per step, independent of context length and mod
    
[^356]: 探究测谎仪方法在检测大语言模型欺骗中的局限性

    Probing the Limits of the Lie Detector Approach to LLM Deception

    [https://arxiv.org/abs/2603.10003](https://arxiv.org/abs/2603.10003)

    本文通过构建基于语言学理论的欺骗性非虚假陈述数据集，实验证明大语言模型能够产生并非虚假的欺骗性陈述，而仅判断真假的标准真值探针无法检测此类行为，从而揭示了“测谎仪”式欺骗检测方法的根本局限。

    

    一类针对大语言模型（LLM）欺骗行为的机制性研究方法依赖于“测谎仪”——即旨在识别模型内部表征为虚假输出的真值探针。这种针对LLM欺骗的测谎仪方法隐含地假设欺骗与说谎是等同的。本文通过实验研究LLM能否产生并非虚假的欺骗性陈述，以及真值探针（若被简单地用作欺骗检测器）是否无法检测到此类行为，从而对这一假设提出了挑战。在此过程中，我们做出了四项核心贡献：第一，我们基于语言学理论提供了一个人工编写的欺骗性非虚假陈述示例数据集。第二，我们在六个不同规模和不同家族的开源LLM上证明，某些模型能够在强制选择任务中选出这些欺骗性非虚假陈述。第三，我们证明，尽管在标准真假数据集上训练的真值探针表……

    arXiv:2603.10003v2 Announce Type: replace  Abstract: A strand of mechanistic approaches to deception in large language models (LLMs) relies on "lie detectors" - truth probes aimed to identify model outputs that are internally represented as false. This lie detector approach to LLM deception implicitly assumes that deception is coextensive with lying. This paper challenges that assumption by experimentally investigating whether LLMs can produce deceptive statements that are not false and whether truth probes (if naively applied as deception detectors) fail to detect such behavior. In doing so, we make four core contributions: First, we provide a dataset of manually written examples of deceptive non-falsities grounded in linguistic theory. Second, we show across six open-source LLMs of different sizes and families that some models can select these deceptive non-falsities in a forced-choice task. Third, we demonstrate that while truth probes trained on a standard true-false dataset perfor
    
[^357]: DARE：通过分布感知检索实现大语言模型智能体与R统计生态系统的对齐

    DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval

    [https://arxiv.org/abs/2603.04743](https://arxiv.org/abs/2603.04743)

    该论文提出DARE框架，通过将数据分布和结构特征融入R函数检索的表示学习，使LLM智能体能够更准确地为数据分析匹配合适的R统计方法，并配套构建了知识库、智能体和基准测试。

    

    大语言模型（LLM）智能体在自动化数据科学工作流程方面表现出色，但往往未能充分利用R语言中严谨的统计方法。现有的检索增强生成（RAG）方法依赖于语义相似性，忽略了对确定统计方法适用性至关重要的数据分布和结构特征。为解决这一问题，我们提出了DARE（分布感知检索嵌入），这是一个将这些数据特征融入R函数检索的表示学习框架。我们的贡献有四个方面：（i）RPKB，一个包含8,191个高质量R函数的精选知识库；（ii）DARE，分布感知检索框架；（iii）RCodingAgent，一个面向R语言的智能体，集成DARE实现端到端分析；（iv）RStatBench，一个包含100个统计任务的综合基准。实验表明，DARE在评估中将NDCG@10提升了13.43至32.65个百分点。

    arXiv:2603.04743v2 Announce Type: replace-cross  Abstract: Large Language Model (LLM) agents excel at automating data science workflows, yet often underutilize rigorous statistical methods in R. Existing Retrieval-Augmented Generation (RAG) approaches rely on semantic similarity, ignoring the data-distribution and structural characteristics crucial for determining a statistical method's applicability. To address this, we propose DARE (Distribution-Aware Retrieval Embedding), a representation learning framework incorporating these data characteristics into R function retrieval. Our contributions are fourfold: (i) RPKB, a curated knowledge base of 8,191 high-quality R functions; (ii) DARE, the distribution-aware retrieval framework; (iii) RCodingAgent, an R-oriented agent integrating DARE for end-to-end analysis; and (iv) RStatBench, a comprehensive benchmark of 100 statistical tasks. Experiments demonstrate that DARE improves NDCG@10 by 13.43 to 32.65 percentage points across evaluated 
    
[^358]: MoBiQuant：面向Token自适应任意精度大语言模型的比特混合量化

    MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM

    [https://arxiv.org/abs/2602.20191](https://arxiv.org/abs/2602.20191)

    提出MoBiQuant框架，通过发现“精度相关的离群值迁移”现象并采用比特混合量化机制，实现可根据token和运行时资源自适应切换位宽的任意精度LLM高效推理。

    

    动态的运行时延迟和内存约束要求大语言模型（LLM）能够灵活部署，即LLM可以根据可用的计算资源以不同的量化精度进行推理。近期关于此类任意精度量化的工作要么依赖于硬件效率低下的向量量化，要么在位宽切换时引入额外的缩放因子。同时，现有的针对固定低精度校准的训练后量化（PTQ）方法在运行时精度变化的情况下泛化能力较差。在本工作中，我们将这种跨位宽泛化能力差的根源归因于一种精度相关的离群值迁移现象，即对PTQ敏感的token的分布在不同精度下会发生变化。受此观察启发，我们提出了MoBiQuant，一种新颖的任意精度比特混合量化框架，能够基于token（自适应地）调整权重精度，实现灵活的LLM推理。

    arXiv:2602.20191v3 Announce Type: replace-cross  Abstract: Dynamic runtime latency and memory constraints necessitate flexible large language model (LLM) deployment, where an LLM can be inferred with various quantization precisions based on available computational resources. Recent work on such any-precision quantization either relies on hardware-inefficient vector quantization or induces additional scaling factors when switching between bit-widths. Meanwhile, existing post-training quantization (PTQ) methods calibrated for a fixed low precision show poor generalizability under runtime precision change. In this work, we attribute the source of poor generalization across bit-widths to a precision-dependent outlier migration phenomenon where the distribution of PTQ-sensitive tokens changes across precisions. Motivated by this observation, we propose MoBiQuant, a novel any-precision Mixture-of-Bits quantization framework that adjusts weight precision for flexible LLM inference based on to
    
[^359]: LLM何时应该不那么具体？用于可靠长文本生成的选择性抽象

    When Should LLMs Be Less Specific? Selective Abstraction for Reliable Long-Form Text Generation

    [https://arxiv.org/abs/2602.11908](https://arxiv.org/abs/2602.11908)

    该论文提出选择性抽象框架，让LLM在长文本生成中不再简单弃答，而是通过将不确定的具体内容替换为置信度更高的抽象表述，用具体性换取可靠性，从而在保留有用信息的同时减少事实错误。

    

    大语言模型（LLM）被广泛应用，但它们仍然容易产生事实性错误，这些错误削弱了用户信任并限制了模型在高风险场景中的应用。缓解这一风险的一种方法是为模型配备不确定性估计机制，在置信度较低时选择弃答。然而，这种二元的“全有或全无”方法在长文本生成场景中过于严苛，往往会丢弃有价值的信息。我们提出了选择性抽象，这是一个让LLM能够以牺牲具体性来换取可靠性的框架，通过选择性地降低不确定内容的细节程度来实现。我们通过选择性风险和覆盖率对SA进行了形式化定义，并提出了逐原子的选择性抽象，这是一种声明级别的实例化方法，将回复分解为原子声明，并用置信度更高、更不具体的抽象表述来替换不确定的原子。我们还开发了一种新颖的端到端评估流程，用于评估长文本生成，将风险实例化为事实正确性并测量覆盖率。

    arXiv:2602.11908v4 Announce Type: replace-cross  Abstract: LLMs are widely used, yet they remain prone to factual errors that erode trust and limit adoption in high-risk settings. One approach to mitigate this risk is to equip models with uncertainty estimation mechanisms that abstain when confidence is low. However, this binary "all-or-nothing" approach is excessively restrictive in long-form settings, often discarding valuable information. We introduce Selective Abstraction (SA), a framework that enables LLMs to trade specificity for reliability by selectively reducing the detail of uncertain content. We formalize SA through selective risk and coverage, and propose Atom-wise Selective Abstraction, a claim-level instantiation that decomposes responses into atomic claims and replaces uncertain atoms with higher-confidence, less specific abstractions. We develop a novel end-to-end pipeline for evaluating long-form generation that instantiates risk as factual correctness and measures cov
    
[^360]: 回声作为锚点：大语言模型推理中的概率代价与注意力重聚焦

    Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning

    [https://arxiv.org/abs/2602.06600](https://arxiv.org/abs/2602.06600)

    该论文揭示并形式化了大型推理模型在思维链开头自发复述问题（“提示回声”EOP）的现象，证明这种以拒绝采样刻画的概率代价实际上起到了锚定注意力、提升似然与下游准确率的作用，并提出了可计算的“回声似然差距”作为预测指标。

    

    近期工作通过测试时扩展来提升大型推理模型（LRMs）的推理能力，方法包括自一致性与并行思考、添加通用的“思考标记”，或者将问题呈现两次（如重读方法 RE2）。然而，这些方法要么注入与任务无关的标记，要么采用无法解释、且常常忽视许多大型推理模型在长思维链（CoT）开头所表现出的自发重复行为的启发式策略。在本工作中，我们研究了模型复述问题的倾向，并将其命名为“提示回声”，认为它是一种从一开始就塑造计算过程的机制，可以被视为发生在推理轨迹内部（而非输入中）的提示重复。我们通过将回声移除建模为拒绝采样，形式化了其概率代价，并定义了“回声似然差距”这一可计算的代理指标，将早期重复与似然增益及下游准确率联系起来。我们发现：（1）在因果干……

    arXiv:2602.06600v3 Announce Type: replace  Abstract: Recent work improves reasoning in large reasoning models (LRMs) by test-time scaling through self-consistency and parallel thinking, adding generic "thinking tokens", or presenting the question twice, as in re-reading (RE2). However, these approaches either inject task-agnostic tokens or mandate heuristics that do not explain, and often ignore, the spontaneous repetition that many LRMs exhibit at the head of their long chain-of-thought (CoT). In this work, we investigate the model's tendency to restate the question, which we dub Echo of Prompt (EOP), as a mechanism that shapes computation from the start and can be viewed as prompt repetition inside the reasoning trace rather than in the input. We formalize its probabilistic cost by casting echo removal as rejection sampling, and define the Echo Likelihood Gap as a computable proxy linking early repetition to likelihood gains and downstream accuracy. We find that (1) in a causal inter
    
[^361]: 面向基于大语言模型智能体的强化世界模型学习

    Reinforcement World Model Learning for LLM-based Agents

    [https://arxiv.org/abs/2602.05842](https://arxiv.org/abs/2602.05842)

    该论文提出强化世界模型学习（RWML），一种自监督方法，通过在预训练嵌入空间中对齐模型模拟的下一状态与环境实际观测到的下一状态，为基于大语言模型的智能体学习动作条件化的世界模型，相比词元级预测提供了更稳健的训练信号并避免了模型崩溃。

    

    大语言模型（LLMs）在以语言为中心的任务中取得了出色的表现。然而，在智能体场景中，大语言模型往往难以预见动作后果并适应环境动态，这凸显了基于大语言模型的智能体对世界建模能力的需求。我们提出了强化世界模型学习（RWML），这是一种自监督方法，利用模拟到真实的差距奖励，在文本状态上为基于大语言模型的智能体学习动作条件化的世界模型。我们的方法将模型生成的模拟下一状态与环境观测到的实际下一状态进行对齐，促使模型内部的世界模拟与实际环境动态在预训练嵌入空间中保持一致。与下一状态词元预测不同——后者优先考虑词元级别的保真度（即复现确切的措辞）而非语义等价性，并可能导致模型崩溃——我们的方法提供了更稳健的训练信号。

    arXiv:2602.05842v3 Announce Type: replace  Abstract: Large language models (LLMs) have achieved strong performance in language-centric tasks. However, in agentic settings, LLMs often struggle to anticipate action consequences and adapt to environment dynamics, highlighting the need for world-modeling capabilities in LLM-based agents. We propose Reinforcement World Model Learning (RWML), a self-supervised method that learns action-conditioned world models for LLM-based agents on textual states using sim-to-real gap rewards. Our method aligns simulated next states produced by the model with realized next states observed from the environment, encouraging consistency between internal world simulations and actual environment dynamics in a pre-trained embedding space. Unlike next-state token prediction, which prioritizes token-level fidelity (i.e., reproducing exact wording) over semantic equivalence and can lead to model collapse, our method provides a more robust training signal and is emp
    
[^362]: 通过对比学习减轻面向ESL学习者的自动作文评分偏见

    Mitigating Bias in Automated Essay Scoring for ESL Learners via Contrastive Learning

    [https://arxiv.org/abs/2601.16724](https://arxiv.org/abs/2601.16724)

    该论文提出一种基于匹配作文对的双向对比学习策略CL-MEP，可将自动作文评分系统对高水平ESL学习者的评分偏见降低39.9%，同时提高整体评分准确率。

    

    自动作文评分系统会对高水平的英语作为第二语言（ESL）学习者施加不成比例的惩罚。我们提出了基于匹配作文对的对比学习方法（CL-MEP），这是一种双向对齐策略。CL-MEP将这种评分偏见降低了39.9%，同时提高了整体准确率，成功地将有效的句法复杂性与表面层面的语法错误区分开来。

    arXiv:2601.16724v3 Announce Type: replace  Abstract: Automated Essay Scoring systems disproportionately penalize high-proficiency English as a Second Language (ESL) learners. We propose Contrastive Learning with Matched Essay Pairs (CL-MEP), a bi-directional alignment strategy. CL-MEP reduces this scoring bias by 39.9% while improving overall accuracy, successfully disentangling valid syntactic complexity from surface-level grammatical errors.
    
[^363]: 超越单次写作：深度研究智能体在多轮报告修订中并不可靠

    Beyond Single-shot Writing: Deep Research Agents are Unreliable at Multi-turn Report Revision

    [https://arxiv.org/abs/2601.13217](https://arxiv.org/abs/2601.13217)

    该论文提出 Mr Dre 评估套件，首次将多轮报告修订确立为深度研究智能体的新评估维度，并揭示了这些智能体虽能响应大部分用户反馈，但在16-28%的已覆盖内容和引用质量上出现退化的关键局限。

    

    现有的深度研究智能体基准测试将报告生成视为单次写作任务，这与人类研究者通过自我反思或同伴反馈迭代起草和修订报告的方式存在根本差异。深度研究智能体能否根据用户反馈可靠地修订报告仍未被探索。我们提出了 Mr Dre，一个将多轮报告修订确立为深度研究智能体新评估维度的评估套件。Mr Dre 包含两部分：(1) 一个涵盖全面性、事实性和呈现方式的统一长篇报告评估协议，以及 (2) 一个经过人工验证的用于多轮修订的反馈模拟流程。我们对五个不同深度研究智能体的分析揭示了一个关键局限：虽然智能体能够处理大部分用户反馈，但它们在16-28%的先前已覆盖内容和引用质量上出现退化。在多轮修订过程中，即使表现最好的智能体仍有巨大的提升空间，因为它们持续……

    arXiv:2601.13217v2 Announce Type: replace  Abstract: Existing benchmarks for Deep Research Agents (DRAs) treat report generation as a single-shot writing task, which fundamentally diverges from how human researchers iteratively draft and revise reports via self-reflection or peer feedback. Whether DRAs can reliably revise reports with user feedback remains unexplored. We introduce Mr Dre, an evaluation suite that establishes multi-turn report revision as a new evaluation axis for DRAs. Mr Dre consists of (1) a unified long-form report evaluation protocol spanning comprehensiveness, factuality, and presentation, and (2) a human-verified feedback simulation pipeline for multi-turn revision. Our analysis of five diverse DRAs reveals a critical limitation: while agents can address most user feedback, they also regress on 16-28% of previously covered content and citation quality. Over multiple revision turns, even the best-performing agents leave significant headroom, as they continue to di
    
[^364]: 基于投影核测量注意力头权重子空间之间的亲和度

    Measuring Affinity between Attention-Head Weight Subspaces via the Projection Kernel

    [https://arxiv.org/abs/2601.10266](https://arxiv.org/abs/2601.10266)

    本文提出用基于主角度的投影核度量注意力头权重子空间之间的亲和度，该指标比传统方法更清晰地揭示注意力头间的交互关系，并据此发现GPT2-small中的L4H7作为身份头充当信息枢纽。

    

    理解注意力头之间的关系对于解释Transformer的内部结构至关重要，然而现有的度量方法无法很好地捕捉这种结构。我们关注注意力头权重矩阵所张成的子空间，并使用投影核这一基于主角度的子空间相似性度量来量化头与头之间的关系。在GPT2-small、BLOOM-560M和Qwen2-0.5B-Instruct上的实验表明，投影核比诸如组合分数等先前指标更清晰地再现了IOI任务上已知的头对头交互。我们进一步引入了一个框架，通过将投影核分布与从随机正交子空间导出的参考分布进行比较，来量化投影核分布的信息量。作为一项应用，我们分析了由投影核构建的有向图，并表明在GPT2-small中，L4H7通过充当身份头而发挥了枢纽作用。

    arXiv:2601.10266v2 Announce Type: replace  Abstract: Understanding relationships between attention heads is essential for interpreting the internal structure of Transformers, yet existing metrics do not capture this structure well. We focus on the subspaces spanned by attention-head weight matrices and quantify head-to-head relationships using the Projection Kernel (PK), a principal-angle-based measure of subspace similarity. Experiments on GPT2-small, BLOOM-560M, and Qwen2-0.5B-Instruct show that PK reproduces known head-to-head interactions on the IOI task more clearly than prior metrics such as the Composition Score. We further introduce a framework to quantify the informativeness of PK distributions by comparing them with a reference distribution derived from random orthogonal subspaces. As an application, we analyze a directed graph constructed from PK and show that, in GPT2-small, L4H7 acts as a hub by functioning as an Identity Head.
    
[^365]: LLMberjack：面向多方对话创建的辩论树引导式修剪

    LLMberjack: Guided Trimming of Debate Trees for Multi-Party Conversation Creation

    [https://arxiv.org/abs/2601.04135](https://arxiv.org/abs/2601.04135)

    LLMberjack是一个结合交互式可视化界面与可选LLM辅助功能的平台，能够将Reddit等平台上树状结构的辩论“线性化”为连贯的多方对话，同时保留参与者身份和话语关系。

    

    在Reddit或Kialo等在线平台上，辩论以回复树的结构呈现：每条消息可以收到多条回复，而这些回复又可以各自收到多条回复，从而形成分支结构而非线性对话。将穿越这样一棵树的一条路径（或多条路径的混合）转化为多个说话者之间自然、逐轮进行的对话（我们将这一过程称为“线性化”）需要决定保留哪些消息、以何种顺序排列，以及每条消息是面向谁的。为此，我们提出了LLMberjack，一个以现有的、最初以回复树形式构建的辩论为起点来创建多方对话的平台。该系统提供了一个交互式界面，可对讨论树进行可视化，并使用户能够在保持参与者身份和话语关系的同时，构建连贯的线性化对话序列。系统还集成了可选的大语言模型（LLM）辅助功能，以支持自动编辑

    arXiv:2601.04135v2 Announce Type: replace  Abstract: Online debates on platforms like Reddit or Kialo are structured as reply trees: each message can receive several replies, which in turn can each receive several replies, producing a branching structure rather than a linear conversation. Turning one path (or a mix of paths) through such a tree into a natural, turn-by-turn conversation among multiple speakers (a process we call linearization) requires deciding which messages to keep, in what order, and who each message is addressed to. To this aim, we present LLMberjack, a platform for creating multi-party conversations starting from existing debates, originally structured as reply trees. The system offers an interactive interface that visualizes discussion trees and enables users to construct coherent linearized dialogue sequences while preserving participant identity and discourse relations. It integrates optional large language model (LLM) assistance to support automatic editing of 
    
[^366]: 面向紧凑语言模型持久语义路由的嵌入一致性调节

    Embedding Consistency Regulation for Persistent Semantic Routing in Compact Language Models

    [https://arxiv.org/abs/2601.00543](https://arxiv.org/abs/2601.00543)

    该论文提出嵌入一致性调节（ECR）方法，通过冻结嵌入模型为输入生成离散前缀，使紧凑语言模型在解码后仍能持续体现预分配的语义路由（如语言、语气、意图），实现99.93%的语言路由准确率，并引入独立于编码器的响应路由可恢复性（RRR）诊断指标。

    

    语言模型越来越多地在严格的内存和计算约束下部署，在这种场景中，语义路由（例如语言、语气或意图）可能在解码之前就被分配。然而，这些路由可能无法在生成的响应中持续得到体现。我们将这一现象作为“持久语义路由”进行研究。我们使用嵌入一致性调节（Embedding Consistency Regulation, ECR）作为一个最小探针来考察这一特性。ECR利用一个冻结的嵌入模型为每个输入导出一段简短的离散前缀，并在训练和推理阶段采用相同的前缀构建方式。在受控的前缀分配实验中，ECR实现了99.93% ± 0.12%的语言路由准确率，而随机、打乱和虚拟对照组仅接近随机水平（34–36%）。这一差距无法被似然指标可靠地捕捉：一些对照组尽管失去了路由一致性，却仍取得了具有竞争力的负对数似然（NLL）。我们进一步提出了响应路由可恢复性（Response Routing Recoverability, RRR），这是一种独立于编码器的诊断……

    arXiv:2601.00543v2 Announce Type: replace  Abstract: Language models are increasingly deployed under tight memory and compute constraints, where semantic routes (e.g., language, tone, or intent) may be assigned before decoding. However, these routes may not remain reflected in generated responses. We study this phenomenon as \emph{persistent semantic routing}. We investigate this property using Embedding Consistency Regulation (ECR) as a minimal probe. ECR derives a short discrete prefix for each input using a frozen embedding model and applies the same prefix construction during both training and inference. In controlled prefix-assignment experiments, ECR achieves 99.93\% $\pm$ 0.12\% language routing accuracy, while random, shuffled, and dummy controls remain near chance (34--36\%). This gap is not reliably captured by likelihood: some controls achieve competitive NLL despite losing routing consistency. We introduce Response Routing Recoverability (RRR), an encoder-independent diagno
    
[^367]: 通过奖励设计解耦多模态大语言模型中的感知与推理

    Disentangling Perception and Reasoning in Multimodal LLMs via Reward Design

    [https://arxiv.org/abs/2601.00215](https://arxiv.org/abs/2601.00215)

    该研究发现多模态大模型在算法视觉谜题上的性能瓶颈在于感知而非推理，并通过奖励设计在无思维链监督的情况下诱导视觉锚定推理，使Qwen-2.5-VL-7B提升5.56分。

    

    带可验证奖励的强化学习已在大语言模型推理方面带来了重大提升，人们直觉上认为这一方法能很好地迁移到多模态模型中。然而，多模态模型需要做两件事：首先感知图像中的内容，然后对其含义进行推理。由于这两个阶段是联合评分的，很难判断单靠推理还有多大的提升空间。我们在算法视觉谜题上研究了这一问题——这类任务同时需要感知和推理两个组件——结果表明，制约性能的瓶颈是感知而非推理。用简单的文本描述替换图像，可使Claude模型的性能平均提升超过20分。随后，我们评估了六种旨在在推理过程中诱导视觉锚定的奖励设计，且无需思维链监督。使用GRPO训练Qwen-2.5-VL-7B时，奖励设计能诱导出带有自我反思和视觉引用的长而结构化的推理，相比（原文截断）……取得5.56分的性能提升。

    arXiv:2601.00215v3 Announce Type: replace-cross  Abstract: Reinforcement learning with verifiable rewards has driven major gains in LLM reasoning, and it is intuitive to assume this recipe will transfer well to multimodal models. However, multimodal models do two things: first, perceive what is in an image, then reason about what it implies. Because these stages are graded jointly, it is hard to tell how much room reasoning alone has to grow. We study this on algorithmic visual puzzles, where both components are necessary and show that perception, not reasoning, is the binding constraint. Replacing images with simple textual descriptions raises performance by over 20 points on average for Claude models. We then evaluate six reward designs aimed at inducing visual grounding during reasoning without chain-of-thought supervision. Training Qwen-2.5-VL-7B with GRPO, reward design induces long, structured reasoning with self-reflection and visual references, yielding a 5.56-point gain over t
    
[^368]: 面向医疗人工智能的层级感知多模态机器遗忘

    Hierarchy-Aware Multimodal Unlearning for Medical AI

    [https://arxiv.org/abs/2512.09867](https://arxiv.org/abs/2512.09867)

    该论文提出了MedForget——一个面向医疗AI的层级感知多模态机器遗忘基准，通过将医疗机构数据建模为嵌套结构实现细粒度的遗忘与保留评估，并揭示了现有遗忘方法难以在删除目标数据的同时保持保留数据性能的问题。

    

    多模态大语言模型（MLLMs）在医疗人工智能等敏感领域的应用日益广泛，而包括HIPAA和GDPR在内的隐私法规要求删除特定个人或机构的数据。这催生了机器遗忘技术，其目标是从已训练的模型中移除目标数据的影响。然而，现有的遗忘基准无法反映真实世界医疗数据的层级结构和多模态特性，限制了其正确评估实际遗忘效果的能力。为此，我们提出了MedForget，一个层级感知的多模态遗忘基准，它将来自医疗机构的数据建模为嵌套结构，从而支持在保留集和遗忘集之间进行细粒度评估。实验表明，当前的遗忘方法难以在不降低保留数据效用（以临床相关预测任务上的性能衡量）的情况下，实现有效的层级感知遗忘。

    arXiv:2512.09867v4 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLMs) are increasingly used in sensitive domains such as medical AI, where privacy regulations, including HIPAA and GDPR, require the removal of specific individuals' or institutions' data. This motivates machine unlearning, which aims to remove the influence of target data from a trained model. However, existing unlearning benchmarks fail to reflect the hierarchical and multimodal structure of real-world medical data, limiting their ability to properly evaluate unlearning in practice. Therefore, we introduce MedForget, a hierarchy-aware multimodal unlearning benchmark that models data from medical institutions as a nested structure, enabling fine-grained evaluation across retain and forget splits. Experiments show that current unlearning methods struggle to achieve effective hierarchy-aware forgetting without degrading retained utility, measured by performance on clinically relevant predictio
    
[^369]: 潜在辩论：一种用于解释大语言模型二分类决策思考过程的代理框架

    Latent Debate: A Surrogate Framework for Interpreting LLM Thinking towards Binary Decisions

    [https://arxiv.org/abs/2512.01909](https://arxiv.org/abs/2512.01909)

    本文提出“潜在辩论”框架，通过捕捉大语言模型在单次推理中内部潜在论证之间的支持与攻击信号，构建了与原模型预测高度一致且具备可解释性的结构化代理模型，用以解释LLM在真假预测任务上的二分类决策思考过程。

    

    理解大语言模型（LLMs）的内部“思考”过程及其与幻觉之间的关联仍然是一个关键挑战。为此，我们引入了“潜在辩论”，这是一个结构化的代理框架，通过内部潜在论证及其相互作用的视角来解释模型在真假预测任务上的输出。与人类辩论不同，潜在辩论捕捉的是模型在单次推理过程中产生的隐藏的支持与攻击信号。我们首先提出了一个与模型和任务无关的概念框架，然后将其符号化实例化，以近似LLM走向二分类决策的思考过程。实证研究表明，我们的潜在辩论是一个忠实的结构化代理模型，其预测与原始LLM高度一致，同时提供了一种可解释性形式。我们还证明了我们的潜在辩论提供了……

    arXiv:2512.01909v3 Announce Type: replace  Abstract: Understanding the internal `thinking' process of Large Language Models (LLMs) and %how this relates their connections to hallucinations remains a key challenge. To this end, we introduce \emph{latent debate}, a structured surrogate framework for interpreting model outputs on True/False prediction tasks through the lens of internal latent arguments and interactions amongst them. Unlike human debates, latent debate captures the hidden supporting and attacking signals that arise within a model during a single inference. We first present a model- and task-agnostic conceptual framework, and then instantiate it symbolically to approximate the thinking process of LLMs towards binary decisions. Empirical studies demonstrate that our latent debate is a faithful structured surrogate model that has highly consistent predictions with the original LLM, while providing a form of interpretability. We also demonstrate that our latent debate provides
    
[^370]: VeriCypher：基于文本到Cypher生成的GraphRAG命名实体验证

    VeriCypher: Named-Entity Verification for GraphRAG via Text-to-Cypher Generation

    [https://arxiv.org/abs/2511.08274](https://arxiv.org/abs/2511.08274)

    该论文提出VeriCypher框架，其核心创新在于命名实体验证机制——通过将生成的Cypher查询中的实体信息与目标图和模式进行核对并迭代修正，提升大语言模型在带标签属性图（LPG）上进行问答的查询准确性与可执行性。

    

    知识图谱问答传统上主要聚焦于资源描述框架（RDF）图、SPARQL查询以及基于三元组的表示。然而，带标签属性图（LPG）为基于大语言模型（LLM）的查询生成带来了不同的挑战：可执行的Cypher查询必须与模式结构、遍历方向、属性键以及字面实体值保持一致。基于此，我们提出了VeriCypher——一个面向LPG问答的、基于LLM的text-to-Cypher框架。该框架将基于类Cypher图模式摘要进行条件化的查询生成，与运行时执行、基于LLM的查询评估以及迭代修正相结合，其核心机制是命名实体验证：系统从生成的查询中提取节点标签、关系模式、属性键和字面值，将其与目标图和模式进行核对，并将不匹配之处转化为下一轮查询生成的修正指令。

    arXiv:2511.08274v2 Announce Type: replace-cross  Abstract: Knowledge graph question answering has traditionally focused on Resource Description Framework (RDF) graphs, SPARQL queries, and triple-based representations. Labeled Property Graphs (LPGs), however, pose a different challenge for LLM-based query generation: executable Cypher queries must align with schema structure, traversal direction, property keys, and literal entity values. This motivated us to introduce VeriCypher, an LLM-based text-to-Cypher framework for question answering over LPGs. It combines query generation conditioned on Cypher-like graph schema summaries with runtime execution, LLM-based query evaluation, and iterative correction where the central mechanism is named-entity verification: the system extracts node labels, relationship patterns, property keys, and literal values from generated queries, checks them against the target graph and schema, and turns mismatches into correction instructions for the next gene
    
[^371]: TOPol：重构与解释多维语义极性场

    TOPol: Reconstructing and Interpreting Multidimensional Semantic Polarity Fields

    [https://arxiv.org/abs/2510.25069](https://arxiv.org/abs/2510.25069)

    TOPol 颠覆了传统单维情感分析范式，通过递归二分法将主题映射到Transformer残差流中所有统计可复现的双极语义轴集合，从而多维地重构和解释语义极性场。

    

    自然语言处理中的极性长期以来一直被操作化为单一轴：即情感构念。我们通过追问一个主题下的文档究竟在哪些方面呈现两极对立，来挑战这种一维观点，而无需预先指定答案。TOPol 颠倒了通常的概念-方向范式：它不是将一个命名的构念映射到某一个方向上，而是将一个主题映射到其文档在Transformer残差流中所支持的全部统计上可复现的双极语义轴集合，之后再追问其中哪些轴对应于已知构念。这些语义轴不一定是评价性的。该方法对每个主题的文档云进行递归式、构造上天然正交的二分划分，且仅当候选轴通过预先设定的对半分割稳定性检验门槛时，才会被确认为一项研究发现。我们首先针对已知答案进行校准：在 Gemma 3 4B 模型上的 9,986 条亚马逊评论（发现 14 个主题）中，星级评分的差异…（原文摘要在此处截断）

    arXiv:2510.25069v3 Announce Type: replace  Abstract: Polarity in natural language processing has long been operationalized as a single axis: the sentiment construct. We challenge this one-dimensional view by asking what a topic's documents are bipolar about, without specifying an answer in advance. TOPol inverts the usual concept-direction paradigm: rather than mapping a named construct to one direction, it maps a topic to the full set of statistically reproducible bipolar semantic axes its documents support in a transformer's residual stream, and asks only afterwards which of them correspond to a known construct. These axes need not be evaluative. The method is a recursive, orthogonal-by-construction bipartition of each topic's document cloud, with a candidate axis promoted to a finding only if it survives one pre-specified split-half stability gate. We calibrate against a known answer first: on 9,986 Amazon reviews (14 discovered topics) in Gemma 3 4B, the star-rating difference-in-m
    
[^372]: 智能体安全综述：应用、威胁与防御

    A Survey on Agentic Security: Applications, Threats and Defenses

    [https://arxiv.org/abs/2510.06445](https://arxiv.org/abs/2510.06445)

    本文是首个对智能体安全领域进行全面综述的工作，通过274篇论文的分类体系和定量分析，从应用、威胁和防御三大支柱系统性地构建了智能体安全领域的研究框架。

    

    在本工作中，我们提出了首个关于智能体安全领域的整体性综述，将该领域围绕应用、威胁和防御三大基本支柱进行结构化组织。我们提供了涵盖274篇论文的全面分类体系，内容包括红队攻击、蓝队防御和特定领域应用、智能体系统的攻击面、防御技术，以及攻防两方面的评估框架。对于每个支柱，我们首先围绕分类体系讨论相关文献，然后进行定量分析以理解这些研究工作之间的规律。在应用方面，我们展示了它们在安全生命周期中的覆盖图谱，比较了红队与蓝队智能体，并分析了红队应用的对抗性使用。在威胁方面，我们分析了攻击所针对的入口点和智能体循环阶段、攻击对智能体场景的特异性，以及它们所假设的威胁模型。在防御方面，我们分析了……

    arXiv:2510.06445v4 Announce Type: replace  Abstract: In this work we present the first holistic survey of the agentic security landscape, structuring the field around the fundamental pillars of Applications, Threats and Defenses. We provide a comprehensive taxonomy of 274 papers that covers red-teaming, blue-teaming and domain-specific applications, attack surfaces on agentic systems, defense techniques, and both offensive and defensive evaluation frameworks. For each pillar, we first discuss the literature around the taxonomy and then perform a quantitative analysis to understand the patterns across these works. For the applications, we show their security-lifecycle coverage map, compare between red-teaming and blue-teaming agents, and analyze the adversarial use of red-teaming applications. On the threat side, we analyze the entry points and agent-loop stages that attacks target, their specificity to the agentic setting, and the threat models they assume. On the defense side, we anal
    
[^373]: ViSSFlow：通过联合学习统一视频条件下的声音与语音生成

    ViSSFlow: Unifying Video-Conditioned Sound and Speech Generation via Joint Learning

    [https://arxiv.org/abs/2509.24773](https://arxiv.org/abs/2509.24773)

    提出统一的流匹配框架 ViSSFlow，通过解耦的条件聚合机制（交叉注意力建模语义条件、自注意力捕获时序密集的视觉线索），将视频生成声音与视觉文本转语音两项任务统一在同一生成式框架下进行联合学习。

    

    视频条件下的音频生成，包括视频生成声音和视觉文本转语音，尽管有着基于视觉内容生成音频的共同目标，但传统上一直被作为两个独立的任务来研究。现有方法依赖于任务特定的架构和训练范式，而由于条件信号的异质性以及有限的联合训练数据，统一生成框架的可能性在很大程度上仍未被探索。在本文中，我们提出了 ViSSFlow，一个面向 V2S 和 VisualTTS 两种任务的统一流匹配框架。为了在扩散 Transformer（DiT）中有效整合多样化的条件模态，我们引入了一种解耦的条件聚合机制，利用注意力层的互补作用：使用交叉注意力来建模语义条件，并使用自注意力来捕获时间上密集的视觉线索。此外，与普遍假设相反……（摘要原文在此处截断）

    arXiv:2509.24773v5 Announce Type: replace-cross  Abstract: Video-conditioned audio generation, including Video-to-Sound (V2S) and Visual Text-to-Speech (VisualTTS), has traditionally been studied as two separate tasks despite their shared objective of generating audio conditioned on visual content. Existing methods rely on task-specific architectures and training paradigms, while the possibility of a unified generative framework remains largely unexplored due to heterogeneous conditioning signals and limited joint training data. In this paper, we propose ViSSFlow, a unified flow-matching framework for both V2S and VisualTTS. To effectively integrate diverse conditioning modalities within a Diffusion Transformer (DiT), we introduce a disentangled condition aggregation mechanism that exploits the complementary roles of attention layers, using cross-attention to model semantic conditions and self-attention to capture temporally dense visual cues. Furthermore, contrary to the common assump
    
[^374]: RCR-Router：面向具有结构化记忆的多智能体LLM系统的高效角色感知上下文路由

    RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory

    [https://arxiv.org/abs/2508.04903](https://arxiv.org/abs/2508.04903)

    提出角色条件上下文路由机制（RCR），在token预算约束下根据智能体角色选择性地将结构化共享记忆中的任务相关条目路由给各智能体，从而降低多智能体LLM系统的推理成本与冗余token消耗。

    

    大型语言模型（LLM）智能体正日益被组织成多智能体系统，用于执行复杂的推理和决策任务。然而，此类系统的规模扩展会显著增加推理成本，尤其是当多个智能体反复处理重叠或与其角色无关的上下文时。许多现有多智能体框架专注于编排和角色专业化，但仍依赖于全上下文共享或静态上下文路由，导致冗余的token消耗、运行时间增加以及记忆利用效率低下。我们提出了角色条件上下文路由（RCR），这是一个执行时层，可在token预算下为智能体分配与其角色相关的共享记忆条目。给定一个结构化的共享记忆和一组角色专业化智能体，RCR根据每个智能体的角色、可选的任务阶段元数据以及token预算，选择性地将与任务相关的记忆条目路由给各个智能体。

    arXiv:2508.04903v4 Announce Type: replace-cross  Abstract: Large language model (LLM) agents are increasingly organized into multi-agent systems for complex reasoning and decision-making tasks. However, scaling such systems can substantially increase inference cost, especially when multiple agents repeatedly process overlapping or role-irrelevant context. Many existing multi-agent frameworks focus on orchestration and role specialization, but still rely on full-context sharing or static context routing, leading to redundant token consumption, increased runtime, and inefficient memory utilization. We present \emph{Role-Conditioned Context Routing (RCR)}, an execution-time layer that allocates role-relevant shared-memory items to agents under token budgets. Given a structured shared memory and a set of role-specialized agents, RCR selectively routes task-relevant memory items to each agent according to its role, optional task-stage metadata, and token budget. RCR integrates role-conditio
    
[^375]: 大语言模型能否捕捉视频游戏的玩家参与度？

    Can Large Language Models Capture Video Game Engagement?

    [https://arxiv.org/abs/2502.04379](https://arxiv.org/abs/2502.04379)

    本文首次全面评估了大语言模型在多模态（文本+视频帧）提示下预测游戏视频玩家参与度的能力，通过4800多项实验发现，尽管LLM能超越传统机器学习基线，但在连续参与度标注任务上总体表现仍落后于专门方法。

    

    开箱即用的预训练大语言模型（LLM）在观察视频时能否成功检测人类情感？为了回答这个问题，我们首次全面评估了流行的大语言模型在以多模态方式接收文本和视频帧序列提示时，成功预测视频连续情感标注的能力。在本文中，我们使用来自GameVibe语料库的20款第一人称射击游戏的80分钟标注视频片段，测试了大语言模型正确标注游戏内参与度变化的能力。我们进行了超过4,800次实验，以研究大语言模型架构、模型规模、输入模态、提示策略和真实值处理方法对参与度预测的影响。我们的研究结果表明，尽管大语言模型在多个领域确实表现出类人的性能，并能够超越传统的机器学习基线，但它们总体上落后于连续标注方法……

    arXiv:2502.04379v3 Announce Type: replace-cross  Abstract: Can out-of-the-box pretrained Large Language Models (LLMs) detect human affect successfully when observing a video? To address this question, for the first time, we evaluate comprehensively the capacity of popular LLMs for successfully predicting continuous affect annotations of videos when prompted by a sequence of text and video frames in a multimodal fashion. In this paper, we test LLMs' ability to correctly label changes of in-game engagement in 80 minutes of annotated videogame footage from 20 first-person shooter games of the GameVibe corpus. We run over 4,800 experiments to investigate the impact of LLM architecture, model size, input modality, prompting strategy, and ground truth processing method on engagement prediction. Our findings suggest that while LLMs rightfully claim human-like performance across multiple domains and able to outperform traditional machine learning baselines, they generally fall behind continuou
    
[^376]: 字节对编码的理论分析

    Theoretical Analysis of Byte-Pair Encoding

    [https://arxiv.org/abs/2411.08671](https://arxiv.org/abs/2411.08671)

    本文证明BPE背后的优化问题是APX完备的（因此不太可能有好的多项式时间近似方案），同时证明BPE在最坏情况下能以1/3至约0.4324之间的因子近似最优对编码的压缩效用，从而为BPE的实际良好表现提供了理论解释。

    

    字节对编码（Byte-Pair Encoding，BPE）是一种被广泛使用的子词分词方法，其起源于基于语法的文本压缩。它被应用于多种语言处理任务，如机器翻译和大型语言模型（LLM）预训练，用于构建一个指定大小的词元字典。迄今为止，对BPE的大多数评估都是经验性的，其良好实际表现背后的原因尚未得到充分理解。本文聚焦于BPE背后的优化问题：寻找能够实现最优压缩效用的对编码。我们证明该问题是APX完备的，这表明它不太可能存在多项式时间近似方案。这以更强的形式回答了Zouhar等人最近提出的一个问题。从积极的角度来看，我们证明BPE对最优对编码压缩效用的近似，在最坏情况下介于1/3与(1-e^{-2})/2 ≈ 0.4324之间。

    arXiv:2411.08671v2 Announce Type: replace-cross  Abstract: Byte-Pair Encoding (BPE) is a widely used method for subword tokenization, with origins in grammar-based text compression. It is employed in a variety of language processing tasks such as machine translation or large language model (LLM) pretraining, to create a token dictionary of a prescribed size. Most evaluations of BPE to date are empirical, and the reasons for its good practical performance are not well understood.   In this paper we focus on the optimization problem underlying BPE: finding a pair encoding that achieves optimal compression utility. We show that this problem is APX-complete, indicating that it is unlikely to admit a polynomial-time approximation scheme. This answers, in a stronger form, a question recently raised by Zouhar et al.   On the positive side, we show that BPE approximates the compression utility of the optimal pair encoding to a worst-case factor between $1/3$ and $(1-e^{-2})/2 \approx 0.4324$. 
    
[^377]: HateSense：用于仇恨言论检测的分层情感分析框架：实现二分类与多分类策略

    HateSense: Hierarchical Sentiment Analysis Framework for Hate Speech Detection: Implementing Binary and Multiclass Classification Strategy

    [https://arxiv.org/abs/2411.05819](https://arxiv.org/abs/2411.05819)

    该研究通过受控实验发现，分层分类和情感特征在仇恨言论检测中并未带来显著收益，而类别加权训练才是唯一能大幅提升性能的有效干预措施。

    

    仇恨言论检测器必须将仇恨性帖子与更为常见的冒犯性但非仇恨性帖子区分开来，实现这一目标的两个流行思路是分层（由粗到细）分类和情感特征。我们重新审视了我们早期结合这两种思路的基于Transformer的框架，并在两个公开基准数据集（Davidson等人的24,783条推文；HateXplain的19,229条帖子）上通过受控研究测试了每个组件，实验采用五次随机划分、基于验证集的调参以及配对自助法检验。结果表明，在使用基于词级和字符级n-gram的线性模型时，我们框架中采用的优势类别优先的分层方法相比平面分类器使宏平均F1降低了1.1至1.5个点（p <= 0.007），而添加VADER情感特征对Davidson数据集宏平均F1的影响最多为0.6个点，且在类别加权设置中从未达到统计显著水平。类别加权训练是唯一具有较大效果的干预措施（宏平均F1提升2.0至3.5个点，仇恨类召回率提升5.9至9.1个点）。

    arXiv:2411.05819v2 Announce Type: replace-cross  Abstract: Hate speech detectors must separate hateful posts from the far more common offensive but non-hateful ones, and two popular ideas for doing so are hierarchical (coarse-to-fine) classification and sentiment features. We revisit our earlier transformer-based framework, which combines both ideas, and test each component in a controlled study on two public benchmarks (Davidson et al., 24,783 tweets; HateXplain, 19,229 posts) with five random splits, validation-based tuning, and paired bootstrap tests. With linear models over word and character n-grams, the dominant-class-first hierarchy used in our framework lowered macro-F1 by 1.1-1.5 points relative to a flat classifier (p <= 0.007), and adding VADER sentiment features changed Davidson macro-F1 by at most 0.6 points, never significantly in the class-weighted settings. Class-weighted training was the only intervention with a large effect (+2.0 to +3.5 macro-F1 and +5.9 to +9.1 hate
    
[^378]: MBBQ：一个用于生成式大语言模型中刻板印象跨语言比较的数据集

    MBBQ: A Dataset for Cross-Lingual Comparison of Stereotypes in Generative LLMs

    [https://arxiv.org/abs/2406.07243](https://arxiv.org/abs/2406.07243)

    提出了MBBQ多语言偏见基准数据集，将英语BBQ数据集精心扩展至荷兰语、西班牙语和土耳其语，用于在控制文化差异和任务准确性的前提下，研究LLMs表现出的社会刻板印象是否随提示语言的不同而变化。

    

    生成式大语言模型（LLMs）已被证明会表现出有害的偏见和刻板印象。虽然安全微调通常（如果进行的话）只在英语中进行，但这些模型正被使用多种不同语言的用户所使用。已有证据表明，这些模型在不同语言之间的表现不一致，并且会根据用户的人口统计特征进行歧视。基于此，我们研究了LLMs所表现出的社会刻板印象是否随提示所用语言的不同而变化，同时控制文化差异和任务准确性的影响。为此，我们提出了MBBQ（多语言偏见问答基准），这是英语BBQ数据集的精心策划版本，扩展到了荷兰语、西班牙语和土耳其语，用于衡量这些语言中普遍存在的刻板印象。我们还通过一个平行的控制数据集来补充MBBQ，以衡量模型的任务表现。

    arXiv:2406.07243v4 Announce Type: replace  Abstract: Generative large language models (LLMs) have been shown to exhibit harmful biases and stereotypes. While safety fine-tuning typically takes place in English, if at all, these models are being used by speakers of many different languages. There is existing evidence that the performance of these models is inconsistent across languages and that they discriminate based on demographic factors of the user. Motivated by this, we investigate whether the social stereotypes exhibited by LLMs differ as a function of the language used to prompt them, while controlling for cultural differences and task accuracy. To this end, we present MBBQ (Multilingual Bias Benchmark for Question-answering), a carefully curated version of the English BBQ dataset extended to Dutch, Spanish, and Turkish, which measures stereotypes commonly held across these languages. We further complement MBBQ with a parallel control dataset to measure task performance on the qu
    
[^379]: 一种理解非洲外资直接投资次国家决定因素的大数据方法

    A Big Data Approach to Understand Sub-national Determinants of FDI in Africa

    [https://arxiv.org/abs/2403.10239](https://arxiv.org/abs/2403.10239)

    本文提出了一种基于文本挖掘和社交网络分析的新方法，通过分析超过167,000篇在线新闻文章，量化了影响非洲公司外资直接投资所有权的区域级属性，结果表明次国家级的结构和制度特征在外资直接投资中扮演着重要角色。

    

    各种宏观经济和制度因素阻碍了外资直接投资流入，包括腐败、贸易开放性、融资途径和政治不稳定性。现有研究主要集中在国家层面数据，对企业层面数据的探索有限，尤其在发展中国家。鉴于这一空白，最近对研究的呼吁强调了对定性数据分析的需求，特别是在区域层面深入探讨外资直接投资决定因素。本文提出了一种新颖的方法，基于文本挖掘和社交网络分析，从超过167,000篇在线新闻文章获取信息，以量化影响非洲公司外资直接投资所有权的区域级（次国家级）属性。我们的分析扩展了世界银行企业调查所绘制的工业发展障碍信息。研究结果表明，区域级（次国家级）的结构和制度特征在决定外资直接投资方面可以发挥重要作用。

    arXiv:2403.10239v1 Announce Type: new  Abstract: Various macroeconomic and institutional factors hinder FDI inflows, including corruption, trade openness, access to finance, and political instability. Existing research mostly focuses on country-level data, with limited exploration of firm-level data, especially in developing countries. Recognizing this gap, recent calls for research emphasize the need for qualitative data analysis to delve into FDI determinants, particularly at the regional level. This paper proposes a novel methodology, based on text mining and social network analysis, to get information from more than 167,000 online news articles to quantify regional-level (sub-national) attributes affecting FDI ownership in African companies. Our analysis extends information on obstacles to industrial development as mapped by the World Bank Enterprise Surveys. Findings suggest that regional (sub-national) structural and institutional characteristics can play an important role in det
    

