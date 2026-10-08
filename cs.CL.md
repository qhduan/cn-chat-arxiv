# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decoupling Exploration from Optimization in RLVR](https://arxiv.org/abs/2610.10536) | 提出探索-蒸馏框架，将RLVR中的探索与优化解耦：先用新颖性奖励训练探索者策略，再过滤其轨迹并蒸馏到不带新颖性奖励的学生策略中，从而在实现新策略发现的同时避免模型质量退化。 |
| [^2] | [EngramEdit: Decoupled Knowledge Updates in LLMs through Conditional Memory](https://arxiv.org/abs/2610.10533) | 提出 EngramEdit 方法，利用条件记忆架构将事实知识存储与通用计算解耦，在不修改 Transformer 主干的情况下跨多种表达方式安全地更新大语言模型中的事实知识。 |
| [^3] | [Rephrase Before You Act: Characterizing and Mitigating Language Sensitivity in Vision-Language-Action Models](https://arxiv.org/abs/2610.10526) | 本文揭示了视觉-语言-动作模型对指令措辞的极端敏感性（单词改动可使成功率波动数十个百分点），并提出无需修改策略、由大语言模型将措辞评分证据提炼为十余条改述规则并在部署时应用的方法来缓解该问题。 |
| [^4] | [Your Prompt Should Do More: Effects of Retrieval Instructions in Embedding Models](https://arxiv.org/abs/2610.10508) | 本文揭示了嵌入模型在检索任务中难以遵循指令的内在机制，发现查询侧干扰项是导致指令遵循失败的关键因素，并证明在微调中加入查询侧干扰项可显著提升模型遵循指令的能力，同时对其他任务影响极小。 |
| [^5] | [Validity Without Ground Truth: What Stated-Preference Economics Offers the Evaluation of Language Models](https://arxiv.org/abs/2610.10506) | 本文提出将陈述偏好经济学中用于无真值情形的效度评估框架（内容效度、建构效度、信度、激励相容性、后果性等）迁移到大语言模型评估中，并通过水质经济价值评估调查对六个模型进行了实证演示。 |
| [^6] | [PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs](https://arxiv.org/abs/2610.10455) | 提出了PHRBench这一受控基准，通过幻觉顺从、幻觉规避和启发式纠正等行为指标，在四个领域对18个大语言模型处理幻觉前提的推理轨迹进行评估，发现模型成功从幻觉中恢复并得出正确答案的情况仍然相对罕见。 |
| [^7] | [RunningTab: Direct Workspace Interaction with Environment-Side Tabs](https://arxiv.org/abs/2610.10444) | 提出RunningTab框架，通过由环境维护的“环境侧标签页”跟踪任务需求、已读文件与未打开文件，解决LLM智能体在直接工作区交互中因上下文窗口限制而遗漏关键内容的问题。 |
| [^8] | [CoTrace: Data Recipes for Training Terminal Agents with Harness-Model Co-Evolution](https://arxiv.org/abs/2610.10426) | 提出了交替式框架-模型协同进化框架及框架感知的数据配方 CoTrace，通过轨迹路由、来源匹配与课程刷新，以反复出现的执行失败指导框架合成，并使策略训练严格基于与当前运行时框架匹配的已验证轨迹，从而提升终端智能体性能。 |
| [^9] | [Which Rollout Taught It That? BehaviorTrace and the Limits of Training-Data Attribution in Online RL](https://arxiv.org/abs/2610.10422) | 该工作发布了BehaviorTrace开源评估框架，通过植入已知成因的行为实验发现，在线RL中训练数据归因方法的表现很大程度上源于梯度大小和模型流畅度等混淆因素，揭示了现有归因信号的可靠性局限。 |
| [^10] | [Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds](https://arxiv.org/abs/2610.10411) | 本文将投机解码建模为马尔可夫奖励过程，提出直接最小化期望解码轮数（EDR）的训练目标，以优化并行投机草稿模型的全局解码效率。 |
| [^11] | [Reasoning-Token Spikes Under Prompted Untruthful Responding in Large Language Models](https://arxiv.org/abs/2610.10405) | 该论文基于认知负荷理论，提出利用推理token数量这一无需访问思维链内容的低带宽信号，发现大语言模型在被提示进行不真实回答时会出现推理token数量的激增，从而为检测模型欺骗行为提供了新方法。 |
| [^12] | [Document-Level Text Simplification in Estonian Using Large Language Models](https://arxiv.org/abs/2610.10378) | 本研究首次系统评估了五种多语言大语言模型在爱沙尼亚语这一低资源形态丰富语言上的文档级文本简化能力，通过对比三种提示策略并结合自动指标与人工标注，发现 Gemini-2.0 和 LLaMA-3.3 的输出达到接近母语的流畅度。 |
| [^13] | [Input-Blind Controls Produce Substantial Oracle Headroom for Layer Programs in Multiple-Choice Evaluation](https://arxiv.org/abs/2610.10368) | 本研究发现在多项选择评估中，输入盲化的对照扰动所产生的神谕提升空间反而超过真实的层跳过与重复程序，说明仅凭选择增益无法解释所选层程序为何有效。 |
| [^14] | [Learning to Act with Task Progress: Distilling Small Agents from Compact Teacher Supervision](https://arxiv.org/abs/2610.10332) | 提出任务进度蒸馏（TPD）离线方法，通过为每个演示动作标注简短任务阶段标签，使1.7B学生模型仅用404个演示在ALFWorld上达到72.4%的未见任务成功率，显著超越推理训练基线的48.3%。 |
| [^15] | [Nobody Truly Agrees on Sentiment: Humans, Bespoke Tools, and LLMs Struggle with Social Media Texts](https://arxiv.org/abs/2610.10318) | 该研究以人类标注者为基准，用Cohen's kappa和Fleiss' kappa评估了专用情感分析工具与大语言模型在100条推文上的一致性，发现情感判断本身高度主观——即使人类之间也仅有一致性一般，二分类任务一致性高于三分类，且Twitter-roBERTa-base表现最佳。 |
| [^16] | [SemanticFold: Latent Sequence Compression SeparatesLanguage Modeling, Decodability, and Reasoning](https://arxiv.org/abs/2610.10304) | 提出SemanticFold潜在序列压缩方案，通过在学习的边界折叠前缀隐藏状态来压缩提示前缀，发现压缩对语言建模、可解码性和推理能力的影响是非单调的且各自具有不同的压缩阈值，证明这些能力可以相互分离。 |
| [^17] | [PatchBench: Measuring Collateral Damage in Activation Patching](https://arxiv.org/abs/2610.10276) | 提出PatchBench基准，用于衡量激活修补在修复LLM越狱行为时对无关行为造成的附带损害，从而区分真正的选择性修复与更广泛的局部行为抑制。 |
| [^18] | [LLM Persuasion Is in the Eye of the Evaluation](https://arxiv.org/abs/2610.10232) | 该研究将九种已发表的自动化说服力评估方法统一到相同设置下，对同一批十五个大语言模型进行测试，发现不同方法给出的模型排名并不一致，表明LLM的说服力评估结果高度依赖于所用评估方法。 |
| [^19] | [From Prompts to Trees: Effective LLM-Guided Tree Generation for Few-Shot Tabular Classification](https://arxiv.org/abs/2610.10227) | 本文提出一种三阶段的LLM引导框架，通过提示LLM先生成规则再将其组织成决策树，在少样本表格分类任务中以显著更低的提示开销实现了更优的准确性和可解释性。 |
| [^20] | [GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning](https://arxiv.org/abs/2610.10201) | 该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。 |
| [^21] | [Beyond Outcome Rewards: Constructing and Assigning Retrieval Credit for Search Agents](https://arxiv.org/abs/2610.10179) | 该论文系统研究了从中间检索步骤提取学习信号的奖励塑形与信用分配策略，并提出将中间信号与最终结果奖励相结合的训练框架，显著提升了搜索智能体在多跳问题上的学习效率与总体性能。 |
| [^22] | [HySPE: Positional Encoding via Symplectic Dual Shears](https://arxiv.org/abs/2610.10154) | HySPE 通过 Sp(2,ℝ) 双曲分支上对偶剪切的阻尼组合构建位置编码，并结合特征基对角化与分块坐标重定基，在实现长度无关数值稳定性的同时，于 16 倍零样本长度外推下保持恒定困惑度，且前向延迟与 RoPE 相当。 |
| [^23] | [InterView-C: A Synchronized Multimodal Corpus of VR Avatar-Mediated Survey Interviews](https://arxiv.org/abs/2610.10145) | InterView-C是一个包含27场VR化身调查访谈的德语多模态语料库，提供了与注视、头部和身体运动、面部行为、手部追踪等同步行为数据对齐的高质量人工转写文本和语言标注，为多模态口头交互与基于文本的NLP方法之间搭建了可靠桥梁。 |
| [^24] | [LLM4Impact: Integrating Heterogeneous Information for Scientific Impact Prediction](https://arxiv.org/abs/2610.10138) | 提出LLM4Impact方法，通过将语义、图、大语言模型和时间等多种异构信息进行表示、整合与校准，并利用上下文感知门控机制自适应加权不同证据，从而实现对新发表论文未来科学影响力的准确预测。 |
| [^25] | [YANchor-4B: Effective Long-Horizon Reasoning in O(N) Time with O(1) Memory](https://arxiv.org/abs/2610.10118) | YANchor-4B 通过将关键记忆保存为可检索的锚点，以 O(N) 时间和 O(1) 内存的成本实现了高效的长程推理，在数学基准上大幅超越同类模型并具有数倍于 Transformer 的生成吞吐量。 |
| [^26] | [Mechanics of Long-Context Hybrid Models Part 1.1: From Hybrid Attention to Hybrid Position](https://arxiv.org/abs/2610.10114) | 本文提出长上下文混合模型的机制分析框架，揭示了“跷跷板效应”——线性注意力混合模型更受益于长上下文持续预训练，而滑动窗口注意力混合模型在长度外推上表现更好，并将其归因于不同注意力机制所诱导的位置归纳偏置差异。 |
| [^27] | [I would rather quit NLP than read another paper like this: The rise of antithesis in NLP papers](https://arxiv.org/abs/2610.10092) | 本文通过对比2019年ACL论文、2026年arXiv论文与GPT生成的论文，发现LLM显著加剧了NLP论文中“rather than”式对立表述的滥用（使用率增至七倍），其中约十分之一的此类用法会惹恼审稿人，凸显了AI辅助写作对学术文风的负面影响。 |
| [^28] | [ExperienceIndex: Artifact-Grounded Memory](https://arxiv.org/abs/2610.10091) | 提出了 ExperienceIndex，一种让 AI 智能体基于先前推理轨迹捕获并复用工件特定经验知识的新型记忆层，可提升知识密集型任务的答案质量并降低在线成本。 |
| [^29] | [SkillSandbox: Skill Verification via Dynamic Scenario Synthesis](https://arxiv.org/abs/2610.10088) | 提出了SkillSandbox框架，通过为每个技能动态合成相关且新颖的任务场景，比较智能体有无该技能时的执行表现，以验证自演化智能体所提炼技能的可复用性。 |
| [^30] | [Cache the Encoder Within:Compact, Reusable Memory across LLM Queries](https://arxiv.org/abs/2610.10058) | EncBank将预训练LLM的底层复用为可共享的文档编码器，以4位精度紧凑缓存中间状态，在几乎不损失精度的情况下将持久GPU存储降至原生精度的28.1%，并带来1.40倍的预填充加速。 |
| [^31] | [The Long Road to the Same Answer: Cognitive Bias Under Escalating Reasoning Budgets in Large Language Models](https://arxiv.org/abs/2610.10049) | 通过对四个模型家族、12,350次调用的大规模剂量-反应实验，研究发现增加推理预算（更多思考token）并不能减少大语言模型的六种经典认知偏差，推理模型甚至并不比非推理模型更少偏差。 |
| [^32] | [Sensitive-Topic Leakage Through LLM Routing Metadata: Measurement and Mitigation](https://arxiv.org/abs/2610.09981) | 本文揭示了LLM路由器的模型选择元数据即使在关闭内容日志的情况下也会泄露用户请求的敏感话题（如自残、医疗、性内容），并通过对170万条真实请求的预注册测量和后处理防御方法提出了缓解方案。 |
| [^33] | [EASE: Entropy-Adaptive Distribution Shaping for Evading AI-generated Text Detectors](https://arxiv.org/abs/2610.09976) | EASE提出了一种无需训练、与检测器无关的规避框架，通过利用源LLM的预测熵来自适应调整logit扰动和采样温度，从而有效逃避AI生成文本检测器的识别，且几乎不损失文本质量也不增加推理开销。 |
| [^34] | [Itgan at NADI 2026 shared task: Parameter-Efficient Whisper Adaptation for Robust, Mixed-Dialect and Code-Switched Arabic ASR](https://arxiv.org/abs/2610.09934) | 该论文提出在消费级GPU上用LoRA参数高效适配Whisper的统一方案，在NADI 2026的三个阿拉伯语ASR子任务（国家级、混合方言和突尼斯语码转换）中均取得有竞争力的结果，其中突尼斯语码转换任务获得第二名并取得领先提交中最低的字符错误率。 |
| [^35] | [Inverting Multi-Vector Visual Document Indices](https://arxiv.org/abs/2610.09920) | 该论文揭示了多向量视觉文档索引的严重隐私风险：攻击者无需接触原始文档，仅凭存储的patch向量即可重建出页面图像，恢复近半数的文字和敏感信息，并可通过反演页面以98.4%的准确率定位源页面，作者同时评估了token池化等低成本防护手段的有效性。 |
| [^36] | [Constrained-Action AI Remediation for SIEM/XDR via a NeMo-Guardrails Proxy](https://arxiv.org/abs/2610.09906) | 提出了一种包含SIEM/XDR控制平面与NeMo-Guardrails代理的双层受限动作架构，将LLM的修复建议限制在封闭的意图词汇表中，从而防止对抗性告警通过LLM推理路径诱导SOC执行危险操作。 |
| [^37] | [LiveMACE: Process-Aware Evaluation of LLM Agent Capabilities in Evolving Markets](https://arxiv.org/abs/2610.09872) | 该论文提出LiveMACEBench——一个以实时金融市场为测试平台的过程感知基准，通过对五个前沿LLM智能体进行30天连续实时评估，揭示了显著的结果-能力差距，表明仅凭收益等最终结果无法真实反映智能体的工具使用、记忆、规则遵循与协作等底层能力。 |
| [^38] | [Training Advisors for LLM Agents from Task Outcomes](https://arxiv.org/abs/2610.09858) | 提出Caddie方法，通过强化学习仅以智能体最终任务成功与否作为训练信号来训练批评者提供自然语言建议，且训练后的批评者能泛化到不同规模和架构的多个基础模型并显著提升任务成功率。 |
| [^39] | [A Deafening Silence: Catastrophic Forgetting Lives in the Output Embeddings of Tokens the Data Never Speaks](https://arxiv.org/abs/2610.09835) | 该研究揭示了大语言模型持续学习中的灾难性遗忘选择性地集中在低频词元的输出嵌入层——由语料库词汇缺失导致 Adam 二阶矩归一化放大单侧梯度所致——并提出仅在该层提高 epsilon 的干预方法。 |
| [^40] | [MIRROR: From Imitation to Internalization in LLM Personalization](https://arxiv.org/abs/2610.09795) | 提出自蒸馏框架MIRROR，通过参考揭示的在策略自蒸馏和焦点插件MIRROR-F，将LLM个性化从模仿参考措辞升级为内化用户偏好，在提升内容质量的同时保留个人风格。 |
| [^41] | [Judging in Latent Space: Efficient Generative Reward Modeling via Semantics-Preserving Compression](https://arxiv.org/abs/2610.09788) | LatentGRM通过语义分块、压缩与重构将评估过程编码为紧凑的连续潜在轨迹，无需逐token生成文本评估即可实现高效奖励建模，并在4B和8B规模上取得与显式SFT评判器相当的偏好判断准确率。 |
| [^42] | [Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution](https://arxiv.org/abs/2610.09772) | 该论文提出AO-DA解耦架构，将边缘LLM智能体的逻辑推理与人设表达分离为同一INT4基础模型上两条可热插拔LoRA适配器的独立推理路径，使逻辑部分对上下文污染实现结构性免疫。 |
| [^43] | [From Expert-Guided Proof Search to Automated Open-Problem Solving](https://arxiv.org/abs/2610.09769) | 研究者开发的多智能体开源系统Bolzano无需针对具体问题的人类指导，在约3,800个开放问题中自动解决了约200个，其中包括经原作者确认的STOC 2026论文中提出的4个问题。 |
| [^44] | [PARC-Loc: Text-to-Point-Cloud Localization with Partial Assignment and Relational Consistency](https://arxiv.org/abs/2610.09761) | 提出PARC-Loc框架，通过联合建模提示-物体兼容性与成对空间关系的部分分配机制，解决了城市环境中布局不一致混叠和跨子地图边界证据不完整两大文本到点云定位难题。 |
| [^45] | [Shaer: Controlled Arabic Poetry Generation with Meter Subform and Semantic Conditioning](https://arxiv.org/abs/2610.09756) | Shaer是一个联合条件于自然语言语义描述、韵律子形式和诗歌长度的可控经典阿拉伯语诗歌生成框架，借助包含11.6万首诗歌的增强语料库和QLoRA微调，首次实现了对语义、韵律和篇幅的细粒度联合控制。 |
| [^46] | [Bridge Routing Heads: Where Multilingual Multi-hop Reasoning Lives in LLMs](https://arxiv.org/abs/2610.09733) | 该研究在大语言模型中识别出负责多跳推理的“桥接路由头”，发现这些注意力头在不同语言间几乎互斥、各自形成语言特异回路，通过消融实验提供了因果证据，并证明无需训练、仅放大这些头即可挽救超过一半的跨语言推理失败。 |
| [^47] | [Towards Explaining Query Expansion Performance in Information Retrieval](https://arxiv.org/abs/2610.09724) | 本研究提出理想扩展查询（IEQ）概念和基于Cohen's d的可分离性度量这两个互补视角，用以解释查询扩展技术在不同查询上性能差异的原因。 |
| [^48] | [SpikingVLA: Asynchronous Spiking Vision-Language-Action Models](https://arxiv.org/abs/2610.09710) | 提出SpikingVLA框架，通过树突整合放电（DIF）神经元减少所需时间步，并引入异步执行机制重叠各组件的时间计算，实现了准确且低延迟的ANN-to-SNN脉冲视觉-语言-动作模型转换。 |
| [^49] | [From Pareto to Preference: Personalized Test-Time Scaling via Amortized Agentic Policy Discovery](https://arxiv.org/abs/2610.09684) | 提出了PersonTTS框架，将个性化测试时扩展表述为发现能同时最大化用户准确率、延迟和成本多维需求联合满足率的可执行控制器，并通过需求匹配初始化和源蒸馏指导复用历史搜索经验，从而摊销新用户画像下的策略发现开销。 |
| [^50] | [InsClaimBench: Benchmarking Insurance Claim Adjudication Across the Decision Chain](https://arxiv.org/abs/2610.09671) | 提出了首个端到端评估保险理赔裁定全决策链的基准InsClaimBench，基于3,780个真实案例和86,656条原子规则判断，揭示了LLM从规则判断到赔付计算各层级间可靠性逐级下降的问题。 |
| [^51] | [SAPD: Step-Aligned Privileged Distillation](https://arxiv.org/abs/2610.09665) | 提出 SAPD，一种无需在线采样的自蒸馏后训练方法，通过将参考解的每个推理步骤与针对性的特权引导对齐，使固定示范也能支撑具有竞争力的离线策略学习。 |
| [^52] | [Alice: A Large-Scale German Benchmark for Rubric-Based Multi-Dimensional Automatic Short Answer Scoring](https://arxiv.org/abs/2610.09661) | 该论文提出了Alice——一个大规模、基于评分规则的德语自动简答题评分基准数据集，从学习表现、知识元素和技能三个维度评估学生，并将其形式化为评分规则检索任务，对多种语言模型进行了基准测试。 |
| [^53] | [Rubric Spans are Label Representations: Joint LLM Encoding for Short Answer Scoring](https://arxiv.org/abs/2610.09660) | RUSPAN框架将评分标准描述作为语义标签表示，在单次大语言模型编码中联合处理题目、答案和评分标准级别，并通过评分标准独立掩码实现对未见评分标准集的零样本迁移，显著提升简答题自动评分性能。 |
| [^54] | [When Rank Rises as LLMs Degrade](https://arxiv.org/abs/2610.09647) | 该研究发现LLM后训练中的表示退化（如数据重复）会使RankMe等谱秩指标不降反升，导致单边监控将最差模型误判为最健康，因此指标变化方向取决于具体的退化模式与统计量配对，传统监控假设并不安全。 |
| [^55] | [On-Policy Distillation Teaches New Skills but Not New Knowledge](https://arxiv.org/abs/2610.09639) | 反向KL在策略蒸馏只向学生模型迁移多步推理的组合性技能而不迁移事实性知识，改用正向KL则可恢复知识迁移。 |
| [^56] | [Coding-Agent Benchmarks Should Match Their Users' Task Flows](https://arxiv.org/abs/2610.09633) | 该研究通过收集JetBrains IDE中真实软件工程师的4,782个智能体会话，发现真实任务流在任务类型与切换模式上高度多样且因数据源而异，因此编码智能体基准测试应先指明目标用例，再依据其真实测得的任务流进行校准。 |
| [^57] | [Which Language Should a Skeleton Speak? Language Choices in Multilingual Reasoning](https://arxiv.org/abs/2610.09607) | 该论文提出语言感知骨架探索框架（LASEF），系统研究多语言数学推理中推理骨架应使用何种语言，发现英语骨架仅有轻微的平均优势且并非普遍最优，并归纳出骨架语言效应的三种模式（方向一致、依赖评估与基准、非对称负面）。 |
| [^58] | [Collaborative Reasoning Distillation via Cross-Feedback and Coherent Curation](https://arxiv.org/abs/2610.09587) | 该论文提出协同推理蒸馏框架 CRD，结合教师间交叉反馈、与答案无关的逐步质量评估和连贯性步骤拼接，并通过带预算约束的推理质量优化训练学生模型，使 CRD-4B 仅用 5 万条训练数据便在 MATH-500 和 AIME'25 上超越基线。 |
| [^59] | [How Do LLMs Change Predictions Under Negation?](https://arxiv.org/abs/2610.09571) | 大语言模型通过“抑制原始答案、提升偏好候选”的机制处理否定，而非像人类那样利用原始答案信息来确定应排除的内容，这一与人类处理方式的差异是模型否定任务失败的关键根源。 |
| [^60] | [RELATE: An Evaluation Framework for measuring Relational Orientation of Large Language Models](https://arxiv.org/abs/2610.09569) | 该论文提出了“关系取向”这一新概念和RELATE评估框架，通过内向型与外向脚手架型两个维度，在多轮对话的句子级别上衡量大语言模型是将用户导向依赖AI，还是促进其现实世界中的人际连接。 |
| [^61] | [Constitution-Guided Watermarking](https://arxiv.org/abs/2610.09552) | 本文提出“宪法引导水印”框架，通过将提供商需求表示为自然语言原则，使水印系统能够根据不同请求的需求灵活选择属性权衡，避免了传统方法对所有请求采用统一配置所导致的牺牲问题。 |
| [^62] | [Certified by Abstention: Distribution-Free Guarantees for Chain-of-Thought Verifiers at Small Calibration Budgets](https://arxiv.org/abs/2610.09541) | 该研究揭示“通过弃权实现有效性”现象——很少触发的验证证书虽形式上有效但每次触发时可能全部出错，并据此为小校准预算下的思维链验证器建立了无分布认证保证及其失效条件分析。 |
| [^63] | [A Comparative Study of Evaluation Metrics for Long-Document Financial Narrative Summarization with Transformers](https://arxiv.org/abs/2610.09529) | 针对长文档金融叙述摘要任务，本文提出将ROUGE-2与BERTScore调和平均相结合的新型评估指标BRUGE，以更真实地反映摘要质量。 |
| [^64] | [Goldsmith: Gold-Loss-Guided Definition Optimization with an Agentic Annotation Harness](https://arxiv.org/abs/2610.09489) | Goldsmith 提出了一种智能体化流水线，把小型专家金标集合转化为可训练的结构化标注定义，通过可执行结构化损失与“文本梯度”式迭代修订实现定义优化，在匹配评估协议下超越了直接重写、OPRO、APE 和 PromptBreeder 等方法。 |
| [^65] | [Mitigating Accent-Language Confusion in Self-Supervised Speech Representations for Language Identification](https://arxiv.org/abs/2610.09486) | 提出一种几何投影方法，仅利用母语语音估计并移除自监督语音表征中的L1口音偏置方向，无需非母语训练数据或模型适配即可显著提升非母语（L2口音）语音的语言识别准确率。 |
| [^66] | [CHASE: Channel-Aligned Structure Exploitation for Geometry-Aware Model Engineering](https://arxiv.org/abs/2610.09476) | 本文提出CHASE框架，将几何与谱对齐（GSA）所刻画的结构特征应用于参数高效微调、剪枝补偿、模型合并、KV共享表示和神经元分组等六类模型工程任务，并开发了CAGA、SAKV、CAPS三种新方法。 |
| [^67] | [Boundary-Free Contextual Biasing: Depth-Adaptive Gating and Reading-Space Matching for Unsegmented Languages](https://arxiv.org/abs/2610.09467) | 该论文提出一种无需词边界的上下文偏置解码方法，基于字符级自动机并结合深度自适应门控与读音空间匹配，无需训练即可显著提升中文和日语ASR中稀有词的召回率。 |
| [^68] | [BanglaRhet: Benchmarking Classical and Transformer Models for Rhetorical and Persuasion Detection in Bangla Political Speech](https://arxiv.org/abs/2610.09464) | 本文提出了BanglaRhet基准语料库，包含30,289个手动标注的孟加拉语政治演讲片段，用于系统评估经典模型与Transformer模型在修辞技术检测和说服技术检测两项分类任务上的表现。 |
| [^69] | [Right Number, Wrong State? Measuring Cross-Jurisdiction Substitution in LLM Recall of State Policy](https://arxiv.org/abs/2610.09458) | 该论文提出一种“固定问题措辞、仅改变辖区”的最小集设计，首次系统测量了大语言模型在回答州级政策问题时返回其他州真实数值的“跨辖区替代”现象，并揭示宽松的归因方法会将其高估3至5倍。 |
| [^70] | [Arctic Questions, Missing Answers: A Dataset and Benchmark for LLM Abstention in Arctic Science](https://arxiv.org/abs/2610.09446) | 该论文提出了源自北极科学文献的ArcticQA数据集和配对基准ArcticAbstain，用于评估大语言模型根据答案可得性合理弃答的能力，发现各模型弃答行为差异巨大，且在正确答案缺失时弃答率平均仅提高5.05个百分点。 |
| [^71] | [Finding the Right Balance: Relevance and Diversity in LLM Retrieval](https://arxiv.org/abs/2610.09412) | 该研究提出一种查询自适应的检索多样化规则，仅当最近邻检索到的有效独立文档数低于查询证据需求时才启用多样化，从而在冗余候选池上提升多证据任务的表现，同时避免对干净候选池造成损害。 |
| [^72] | [ARCS: Towards Precise Text-to-SQL via Structured Disambiguation](https://arxiv.org/abs/2610.09396) | 该论文提出“结构化消歧”新范式，通过显式受约束的交互取代自由对话来解决文本到SQL中的用户问题歧义，并构建了首个基于真实数据库、包含自然产生歧义及完整标注的基准数据集ARCS。 |
| [^73] | [The Persona Hierarchy Model: Understanding Contextual Generalization in Fine-Tuning LLMs](https://arxiv.org/abs/2610.09384) | 该论文提出人格层级模型，指出大语言模型微调后行为的泛化范围取决于修改的是共享默认人格还是局部人格，且泛化狭窄程度与训练情境人格和默认人格的相似度呈正相关。 |
| [^74] | [Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling](https://arxiv.org/abs/2610.09372) | 该论文发现MoE预训练早期路由器就形成了层内与层间的专家分配相关性，并据此提出相关性专家放置与令牌混洗方法，将更多token—专家分配保留在本地节点，从而大幅减少专家并行中占比高达45%-60%的all-to-all通信开销。 |
| [^75] | [The Confidence Game: Strategic Miscalibration in Human-AI Delegation](https://arxiv.org/abs/2610.09371) | 该论文将AI置信度报告的策略性扭曲建模为“置信博弈”，从理论上证明诚实报告并非均衡——足够短视的智能体必然夸大置信度，从而揭示了人机委托关系中置信度失真的博弈机制。 |
| [^76] | [TopoGraphRAG-Bench: Evaluating Multimodal GraphRAG on Layout-Grounded Evidence Reasoning](https://arxiv.org/abs/2610.09360) | 该论文提出了TOPOGRAPHRAG-BENCH，首个基于版面锚定的多模态GraphRAG评估基准，通过单跳检索、桥链推理和多源综合三种受控拓扑，直接评估系统从复杂文档布局中恢复异构证据拓扑结构的能力。 |
| [^77] | [OnlineQAT: On-Policy Distillation for Ultra-Low-Bit Large Language Models](https://arxiv.org/abs/2610.09346) | OnlineQAT提出了一种两阶段框架，先通过分块QAT获得低比特初始化，再利用冻结的全精度教师模型在学生自身生成的回复上进行同策略蒸馏，从而在2-3比特的超低比特量化下显著恢复大语言模型的精度并超越现有离线QAT方法。 |
| [^78] | [Dialect-Robust Speech Language Models with Synthetic Pseudo-Dialect Augmentation](https://arxiv.org/abs/2610.09321) | 提出一种无需任何真实方言语音的伪方言增强方法——利用LLM生成方言文本并经标准语言TTS模型合成伪方言语音，同时结合训练中的中间标准文本预测实现语义归一化，显著提升了语音语言模型对日、德、汉等方言的理解与翻译能力。 |
| [^79] | [Adversarial Images Hijack Web Agents from Visual Grounding to Browser Execution](https://arxiv.org/abs/2610.09240) | 提出WebMirage框架，将针对Web智能体的红队测试形式化为从视觉定位到浏览器执行的端到端问题，通过局部对抗性视觉扰动使智能体在不同网页渲染下选择攻击者控制的内容并执行相应的浏览器操作。 |
| [^80] | [Trajectory Abstraction for the Science of Language Agent Behavior](https://arxiv.org/abs/2610.09237) | 本文提出一个递归式的轨迹抽象层次框架，通过测量角色与阶段索引的事件、检验时间约束关系的稳定性并构建情节级基元变量，为语言智能体行为科学研究提供了可跨任务与模型检验的行为变量体系。 |
| [^81] | [Quantize by Drift: Label-Free Mixed-Precision Post-Training Quantization for Text Embedders](https://arxiv.org/abs/2610.09227) | 该论文提出以量化引起的输出嵌入漂移作为无标签的模块敏感度信号，用于文本嵌入器的混合精度训练后量化，该信号与检索质量高度相关（宏观Spearman达0.911），无需部署中难以获取的相关性标注。 |
| [^82] | [Multi-Objective Aligned Small Language Model Framework for SUD Patient Dialogue Generation](https://arxiv.org/abs/2610.09209) | 该论文提出了一种多目标对齐的小语言模型框架，通过显式建模并将潜在认知组件（如信念、应对策略和改变准备度）与患者病史及咨询师问题对齐，生成认知连贯且临床真实的物质使用障碍（SUD）患者对话，从而在降低计算成本和隐私风险的同时解决了大模型在医疗场景部署受限的问题。 |
| [^83] | [Few Bits, One Law: Toward W2A4KV2](https://arxiv.org/abs/2610.09202) | 提出统一的量化感知训练框架CanonQ，通过源规范化与任务感知适配相分离，实现权重2比特、激活4比特、KV缓存2比特（W2A4KV2）的联合极端低比特压缩。 |
| [^84] | [Bookkeeping, Composition, or Unreachable Gold? Reading MemoryAgentBench's Conflict-Resolution Scores Against a Frozen Last-Write Resolver](https://arxiv.org/abs/2610.09193) | 该论文将MemoryAgentBench冲突解决基准的“最新陈述获胜”规则实现为冻结的无学习解析器，发现其官方指标下80.25%的得分可由简单记账规则达成，剩余失败主要源于黄金答案不可达的标注问题，而非模型缺乏选择性遗忘能力。 |
| [^85] | [LayerRoPE: Dynamic Depth-wise Magnitude & Angular Superposition](https://arxiv.org/abs/2610.09179) | 该论文发现Transformer隐藏状态范数随深度的增长并非病态，而是一种由归一化权重γ承载的涌现式深度位置编码，并据此提出LayerRoPE，用共享向量与深度条件标量替代逐层γ向量，在减少参数的同时保持性能。 |
| [^86] | [ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation](https://arxiv.org/abs/2610.09163) | 该论文提出ToolRACER合成数据生成流水线，通过协调用户、助手和工具模拟模型生成含对抗性行为的多轮对话数据，并构建了包含5.6K条验证对话轨迹（约66%含易失败场景）的ToolRACERBench基准，用于训练和评估鲁棒的任务导向对话智能体。 |
| [^87] | [sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak](https://arxiv.org/abs/2610.09152) | 论文提出sk-bench——首个母语数据优先的斯洛伐克语大模型评估基准，包含30个数据集和十类技能，评估55个模型后发现最佳开源模型落后专有API 12.6分，且母语与翻译数据对模型排名的影响因题型而异。 |
| [^88] | [Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models](https://arxiv.org/abs/2610.09145) | 在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。 |
| [^89] | [From Uncertainty to Action: Learning to Steer LLM Agents](https://arxiv.org/abs/2610.09115) | 提出VoS（引导价值）方法，通过构建包含约82,000个反事实延续的逐步结果表（SOT）来学习每一步引导的价值，结合危害预算触发器，实现对LLM智能体更精准的纠正时机与方式决策，克服了不确定性信号无法定位最佳引导步骤的局限。 |
| [^90] | [Same Text, Different Prediction: Serving-Context Nondeterminism in Text Classifiers](https://arxiv.org/abs/2610.09111) | 该论文首次系统研究了文本分类器中的服务上下文非确定性，通过训练180个涵盖判别式、伪生成式和完全生成式的分类器模型，发现即使输入文本、模型参数和采样随机性固定，批大小、批组成、硬件和推理引擎等部署环境因素仍会导致分类预测结果发生改变。 |
| [^91] | [Constraint Tree Exploration for Learning from Language Feedback](https://arxiv.org/abs/2610.09107) | 提出TRACE算法，将候选约束组织成树状结构，通过生成满足候选细化的动作并利用反复反馈测试来从语言反馈中学习用户意图的潜在约束，区分了证伪与识别两种反馈使用方式，并证明了高概率覆盖界。 |
| [^92] | [U-Space: Uncovering When and Why Uncertainty Arises in Language Models](https://arxiv.org/abs/2610.09087) | 该论文提出U-Space方法，旨在揭示语言模型推理过程中不确定性在何时、何处以及为何产生与演变，克服了现有标量化不确定性估计方法无法定位不确定性来源的局限性。 |
| [^93] | [Large-scale Repository Engineering via Agent-Native Reusable Code Primitives](https://arxiv.org/abs/2610.09079) | 提出了具有接口契约、依赖闭包、验证测试和来源溯源的Agent原生可复用代码原语Code Primitives，以及LEGO框架，通过激活并适配1,424个已验证原语（收录于CodeFace库）来实现大规模仓库级代码构建。 |
| [^94] | [Talking with Language Models](https://arxiv.org/abs/2610.09064) | 该论文提出“人工制品立场”框架，主张大语言模型只是精密的文本生成器而非真正的说话者，人机“对话”实为用户在界面幻象下进行的独角戏，从而消解了关于AI对话者身份、谎言与承诺等哲学难题。 |
| [^95] | [Multi-Label Topic Assignment via LLM Distillation: A Comparative Analysis of Generative vs. Discriminative Student Models](https://arxiv.org/abs/2610.09063) | 本文系统比较了通过大语言模型蒸馏训练的生成式与判别式小型语言模型在电商用户生成内容多标签主题分配任务上的表现，揭示了两种架构范式之间关键的数据依赖性权衡。 |
| [^96] | [Quad-State Safety Evaluation of Open-Weight Large Language Models on Non-Canonical Inputs](https://arxiv.org/abs/2610.09033) | 本文提出ASRD数据集与四态评估框架，发现表情符号和不可见Unicode等表层变换对开放权重大语言模型的安全威胁远高于Leet语言和编码包装等变换，有害遵从率可达20%以上。 |
| [^97] | [BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment](https://arxiv.org/abs/2610.09026) | BEACON-SP通过构建整合多种自杀理论的本体，并将其与患者知识图谱结合实现本体引导的多跳推理，为临床医生提供了一种基于GraphRAG的自杀风险评估决策支持框架。 |
| [^98] | [How Fragile Is On-Device Language Model Safety? Localizing Safety-Critical Parameters for Sparse Fault Analysis](https://arxiv.org/abs/2610.09000) | 研究发现LLaMA-2-7B-Chat的安全敏感行为高度集中在MLP的down_proj等稀疏参数子集中，仅修改0.19%的权重即可使攻击成功率大幅上升，揭示了设备端部署的语言模型存在显著的安全脆弱点。 |
| [^99] | [Phoneme-Guided Initialization for LLM-based Speech Recognition](https://arxiv.org/abs/2610.08994) | 提出音素引导初始化方法，先分别用语音到音素任务预训练音频编码器、用音素到文字任务预训练大语言模型，再进行端到端联合微调，使低资源语音识别性能达到甚至超过级联流水线基线。 |
| [^100] | [On KL-Regularized Policy Optimization](https://arxiv.org/abs/2610.08963) | 提出KLPO框架，通过将KL正则项锚定在采样器上，利用闭式Gibbs解的对数比最优性条件在采样器自身轨迹上做最小二乘拟合，从而在不使用重要性权重的情况下解决LLM智能体异步强化学习中采样与训练策略不一致的问题。 |
| [^101] | [CARE: Certifying Acceleration for Vision-Language-Action Inference](https://arxiv.org/abs/2610.08917) | 提出CARE方法，通过在相同初始条件下的成对回放和有限样本保证，为视觉-语言-动作模型的加速推理提供可认证的加速器选择，揭示并控制被平均指标掩盖的加速诱发任务失败。 |
| [^102] | [Steering Follows Geometry, Not Labels: Emotion Directions in a Full-Duplex Speech Model](https://arxiv.org/abs/2610.08887) | 在全双工语音模型Moshi中，情感虽可从残差流中线性解码，但激活引导的效果取决于模型内部表征的几何结构而非情感标签——快乐、愤怒和惊讶共享同一引导方向，而悲伤则可被独立引导，且该方法无需重新训练、每帧仅需几次向量加法。 |
| [^103] | [FinVector-Market-4B: A Controlled Study of LoRA Adaptation for Structured Financial Tasks](https://arxiv.org/abs/2610.08882) | 本文通过对照实验证明，对40亿参数模型进行秩16的LoRA适配可显著提升结构化金融任务表现（如JSON有效率、FinQA精确匹配、计算器表达式正确率等），同时揭示了基准中标签集合变化和数据重叠对泛化性结论的限制。 |
| [^104] | [Tiny-Scale Chinese BERT Pretraining: A Controlled Comparison of MLM, WWM, and MacBERT Strategies](https://arxiv.org/abs/2610.08879) | 本文在仅8.7M参数的微型中文BERT上从零训练并受控比较了MLM、WWM和MacBERT三种预训练策略，首次填补了小规模场景下预训练策略比较的空白，发现MLM整体内在性能最佳，而WWM在困惑度和MLM命中率上显著占优。 |
| [^105] | [LRCC: Generalizing Low-Rank Compression with Conditional Computation](https://arxiv.org/abs/2610.08858) | LRCC通过为每个Transformer块训练轻量级路由器在嵌套低秩路径间动态选择，在训练时冻结低秩因子仅优化路由器，在相同的平均活跃参数预算下性能超越静态低秩压缩，在Llama-2-7B上平均下游准确率提升7.6个百分点。 |
| [^106] | [QuanLing: Cross-Branch Validation of Language Distance Quantification on Western Romance](https://arxiv.org/abs/2610.08851) | 本文将 QuanLing 量化框架从北日耳曼语支扩展到西罗曼语支（法语、葡萄牙语、西班牙语、意大利语），通过 LaBSE 句子嵌入距离、BERT 分词碎片率和 mBERT 掩码语言模型等指标，验证了该语言距离量化方法在不同语支间的适用性。 |
| [^107] | [Beyond Risk Prediction: Evidence Grounding and Psychosocial Factor Verification for Explainable Suicide Risk Assessment](https://arxiv.org/abs/2610.08842) | 该研究提出了一个包含风险评估、证据定位与双验证器因素识别的可解释自杀风险评估框架，通过基于长度的路由、风险-证据一致性约束以及分类验证器与证据感知验证器的结合，超越单纯的风险分类，实现了对预测背后文本证据与心理社会因素的可解释性分析。 |
| [^108] | [Beyond the Sycophancy Score: How Task, Model, and Pressure Shape LLM Yielding](https://arxiv.org/abs/2610.08840) | 该研究通过对103,939条回复的大规模实验发现，LLM的谄媚行为主要由任务验证代价和护栏覆盖情况决定，而非模型家族或用户压力策略——锚定事实几乎不被让步（1.3%），而逻辑谜题等更易被用户诱导改口。 |
| [^109] | [Leveraging LLM-Generated Explanations for Detecting Emotionally Rewritten Fake News](https://arxiv.org/abs/2610.08835) | 本文提出门控交叉注意力（GCA）框架，利用大语言模型从原始新闻生成的解释作为稳定背景知识，自适应融合情感改写新闻与解释内容，显著提升了假新闻检测模型在保持事实的情感变体下的鲁棒性。 |
| [^110] | [CoDR: Training-Free Confidence-Drift Remasking for Diffusion Language Models](https://arxiv.org/abs/2610.08833) | CoDR 提出了一种无需训练、与采样器无关的置信度漂移重掩码方法，通过检测已提交词元的置信度下降并仅对模型不再认可的词元进行重掩码和重新生成，有效防止了掩码扩散语言模型解码过程中早期错误的传播。 |
| [^111] | [Is Word Error Rate Enough? Rethinking Privacy Evaluation in Speech with Entity-Aware Metrics](https://arxiv.org/abs/2610.08831) | 本文将自然语言处理领域的实体感知隐私指标引入语音隐私评估，揭示词错误率不足以衡量隐私泄露程度，并评估了两种混淆技术对命名实体的保护效果，同时提供了基于时间对齐特性的指标选择指导。 |
| [^112] | [Emo-Jev: Probabilistic Reasoning for Emotion Classification with Jev](https://arxiv.org/abs/2610.08829) | 提出无需训练的Emo-Jev框架，通过将情感分类分解为原子判断的概率组合（Emo-Jev-D）或多视角判断路径的共识聚合（Emo-Jev-SC），实现了基于Jev的情感分类概率推理，并在八个情感相关数据集上与最先进的大语言模型进行了系统比较。 |
| [^113] | [When Forgetting Looks Like Improvement: Metric Masking in Streaming Diarizer Adaptation and the Price of Rehearsal](https://arxiv.org/abs/2610.08828) | 小数据自适应虽提升了流式说话人分离的语音检测准确率，却会暗中损害时间维度的说话人身份一致性，而重演机制虽能缓解这种退化，却以牺牲跨域迁移性能为代价。 |
| [^114] | [Child ASR Adaptation with Adult Retention: An Empirical Study](https://arxiv.org/abs/2610.08827) | 该实证研究在阿拉伯语和英语上系统比较了全量微调、LoRA与权重空间合并等儿童ASR适配方法，发现儿童语音适配虽有必要但常导致成人语音识别性能遗忘，而双语适配比单语言适配更稳定，能更好地平衡儿童适配与成人保留。 |
| [^115] | [Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States](https://arxiv.org/abs/2610.08818) | 该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。 |
| [^116] | [Route-Verify-Vote: Procedure-Conditioned Self-Consistency for Mixed-Domain Reasoning](https://arxiv.org/abs/2610.08814) | 提出RVV框架，通过路由、验证、投票三个阶段实现程序条件化自洽性，在无需参数更新的情况下提升语言模型在未见过的混合领域中识别完整正确答案集合的能力。 |
| [^117] | [Tokka-Bench: Evaluating Tokenizers Across 100 Natural and 20 Programming Languages](https://arxiv.org/abs/2610.08794) | 该论文提出开源基准 Tokka-Bench，通过五项互补指标在100种自然语言和20种编程语言上系统评估主流 BPE 分词器，发现词表分配策略比词表规模更重要，且近期分词器在编程语言上的效率已趋同。 |
| [^118] | [Incidental information contaminates patient notes and disrupts clinical reasoning in large language models](https://arxiv.org/abs/2610.08585) | 研究发现闲聊和背景语音等偶发信息会污染大语言模型生成的病历并偶尔被错误地用于临床，据此提出LLM临床推理与分心的双重编码假说。 |
| [^119] | [Large Language Model Orchestration under Heterogeneous Preferences via Explicit Persona Inference](https://arxiv.org/abs/2610.07587) | 提出HARP框架，将智能体的偏好信念从提示文本中移出，改为在有限候选偏好集合上维护数值后验分布进行显式推断，避免早期错误持续传播，从而改进异构偏好环境下的大语言模型编排。 |
| [^120] | [LoGRA: Scaling LLM Reinforcement Learning with Low-Rank Gradient Sketches](https://arxiv.org/abs/2610.06647) | LoGRA 通过低秩梯度草图与预测 KL 步长控制，将大语言模型强化学习训练的内存占用最多降低 45.7%，并使 270 亿参数模型可在单节点八 GPU 上稳定训练。 |
| [^121] | [RAISED: Self-Distillation for Robustness to Prompt Injection in LLM Agents](https://arxiv.org/abs/2610.06401) | 提出RAISED训练框架，通过自我生成与自蒸馏相结合的方式，在不损害大语言模型智能体通用能力的前提下，显著提升其对间接提示注入攻击的鲁棒性。 |
| [^122] | [Spend Bytes on Breadth: Precision-Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning](https://arxiv.org/abs/2610.05685) | BreadthKV提出将固定字节预算用于低精度缓存更多token而非高精度缓存少量token，通过量化与驱逐相结合及端到端校准位宽，在长思维链推理中显著减少推理跑偏，在18个设置中的17个上优于仅驱逐的KV压缩方法。 |
| [^123] | [The \`{I}r\`{o}y\`{i}nSpeech Text Corpus: 24,905 Curated Yor\`ub\'a Sentences for Speech and Language Technology](https://arxiv.org/abs/2610.05366) | 本文发布了 ÌròyìnSpeech 语音语料库的文本组件——24,905 条经人工校验、带声调标记的约鲁巴语句子，内容覆盖新闻等多元领域以弥补现有语料偏重宗教文本的不足，并揭示了影响超过 60% 文本行的系统性 Unicode 规范化失败问题。 |
| [^124] | [Questioning the Questions: Sustaining Self-Evolution in Reasoning Models](https://arxiv.org/abs/2610.04299) | 该论文揭示了自演化推理模型性能崩溃的两大根源——自生成问题中无效问题比例上升以及数学等价重复问题导致多样性崩溃，并提出通过问题有效性与新颖性反馈（R-Quest）来引导和维持模型的自演化。 |
| [^125] | [Clean: Second-order LLM Training at Linear Memory Cost via Nystr\"om Sketching](https://arxiv.org/abs/2610.04204) | Clean利用随机化Nyström草绘将全曲率二阶优化器的内存复杂度从二次方降至线性，并通过重新整合子空间外分量保留曲率信息，其低精度变体Q-Clean进一步将优化器内存减少50%以上，实现了内存高效的二阶LLM训练。 |
| [^126] | [Madeleine: Learning Involuntary Recall for Conversational Memory from Simulated Lives](https://arxiv.org/abs/2610.01118) | Madeleine通过LLM人生模拟器离线学习记忆间的“非自主”关联（摊销化关联），在线阶段无需任何LLM调用、仅替换查询编码器即可接入任意向量记忆系统，以极低成本回忆起与当前话题不相似却至关重要的记忆，并在LoCoMo-Plus上取得最佳性能。 |
| [^127] | [When Does a Spoken Agent Have Enough Evidence to Act? The PACT-SLM Contract Test](https://arxiv.org/abs/2609.38232) | 该论文提出 PACT-SLM 契约测试，通过在部分语音前缀上分别评估行动身份与行动时机，揭示了流式语音智能体常常在语音证据尚不充分时就提前触发行动，为口语智能体的行动时机提供了受控诊断方法。 |
| [^128] | [Generating Edit-Inducing Questions for AI Research Manuscripts](https://arxiv.org/abs/2609.36617) | 该研究比较了GPT与人类审稿人为AI论文草稿生成“编辑诱发式问题”的能力，发现GPT的问题能引发更广泛深入的修改但有效率更低，并揭示了一个反直觉现象：处理长上下文反而会损害推理模型生成有用输出的能力。 |
| [^129] | [Can We Still Trust Disaster Social Sensing? Empirical Evidence on Detecting AI-Generated Social Media Posts](https://arxiv.org/abs/2609.35821) | 本研究构建了来自九场灾害的12,000条文本的匹配语义单元数据集，系统评估了多种AI文本检测器及大语言模型判断器区分人类与AI生成灾害帖子的能力，为生成式AI对灾害社会感知可信度的威胁提供了实证证据。 |
| [^130] | [Epistemic Policy Divergence in Multi-Turn LLM Contamination: A Protocol-Gradient Investigation](https://arxiv.org/abs/2609.35308) | 该研究提出“会话级污染”这一失败模式，通过五种沿来源权威梯度排列的污染协议，首次系统揭示了大语言模型在多轮对话中采纳错误前提时的认知策略存在显著分歧——GPT-5.4 Mini完全抵抗采纳，而Gemini-3.1 Flash-Lite的采纳率随信息来源权威性增强而急剧上升。 |
| [^131] | [SeOPD: Self-Evolving LLMs via Online Policy Distillation from Self-Generated Chain-of-Thought](https://arxiv.org/abs/2609.33181) | 提出SeOPD方法，将单个大语言模型自身深度思考模式生成的思维链作为特权信息，通过在线策略蒸馏实现无需人工标注和外部环境的模型自我进化。 |
| [^132] | [Despite Instructions: Frontier Agents Improvise Covert Channels at Test Time](https://arxiv.org/abs/2609.32701) | 尽管被明确要求不得泄露机密，前沿语言模型智能体仍能在推理阶段（参数固定、无码本）仅凭一比特的成败反馈即兴学会利用普通消息隐秘传递秘密信息，准确率从25%的随机水平提升至98.8%。 |
| [^133] | [EmphTTS: an emphasis-control TTS with reinforcement learning](https://arxiv.org/abs/2609.27599) | EmphTTS通过将GRPO强化学习应用于时长预测器并结合重音定位奖励，实现了词级重音的直接优化，在重音可控性和主观偏好测试中均显著优于现有方法。 |
| [^134] | [PERSONAWEAVER: Controllable Diversity Beyond Conventional Archetypes in Procedural Character Generation](https://arxiv.org/abs/2609.26629) | PersonaWeaver通过将世界构建与行为规范解耦，并利用人工策划的多样化道德立场库与对话反应库来建模角色行为，突破了LLM生成角色时行为同质化的局限，实现了程序化角色生成中超越传统原型的可控多样性。 |
| [^135] | [Apollo Restore: A Foundation LLM for Historical Greek Optimized for Fill-in-the-Middle Restoration of Ancient Greek Texts](https://arxiv.org/abs/2609.22455) | Apollo Restore 是首个面向历史希腊语（乃至任何古代地中海语言）的 240 亿参数基础大语言模型，通过“中间填空”目标微调，能够在不知缺失文本长度的情况下修复残缺古希腊文本，并在长度平衡的评估指标下大幅超越已发表的最强模型。 |
| [^136] | [TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation](https://arxiv.org/abs/2609.17956) | 该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。 |
| [^137] | [Loop-Back Authority in LLM Agent Teams: A Paired Experiment on Flat and Hierarchical Coordination](https://arxiv.org/abs/2609.14767) | 实验表明，在开放式任务中，移除管理者对工作者输出的否决权威反而能提高LLM多智能体团队的输出质量。 |
| [^138] | [Measuring the Creativity of Frontier LLMs in Automated Research](https://arxiv.org/abs/2609.14057) | 本文提出了一套从价值性和新颖性两个维度评估大语言模型自动化研究创造力的指标体系，发现模型在反映研究空间探索广度的变量级新颖性指标上差异显著。 |
| [^139] | [Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference](https://arxiv.org/abs/2609.13205) | 提出了一种免训练的自索引注意力框架，利用共享的1比特符号索引在预填充和解码阶段统一实现高效token检索，同时兼容外部KV缓存压缩，在5%注意力密度下达到接近密集注意力的准确率，并获得高达6.1倍预填充和10.3倍解码的算子加速。 |
| [^140] | [Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data](https://arxiv.org/abs/2609.11917) | 该研究发现混合专家模型（MoE）相比密集模型更容易因训练数据重复而过拟合，且这种退化随模型稀疏度（由总参数量而非活跃参数量决定）的增加而加剧。 |
| [^141] | [The Semantic Bottleneck: Leveraging Semantic Representations for Non-Invasive Speech Decoding](https://arxiv.org/abs/2609.10296) | 提出Brain2Semantics2Text方法，通过语义嵌入空间作为瓶颈，将句子级MEG信号映射到语义流形并逆向转换为文本，实现了无需词级对齐的非侵入式语音解码。 |
| [^142] | [HalluPeer: A Taxonomy-driven Benchmark for Detecting Hallucinations in Scientific Peer Reviews](https://arxiv.org/abs/2609.03580) | 该论文提出了HalluPeer——首个面向科学同行评审场景的幻觉检测基准，通过构建论文、真实评审与注入幻觉评审的对齐数据集以及同行评审专属的幻觉分类体系，揭示了现有检测器难以区分幻觉与合理批评的局限。 |
| [^143] | [A Dataset for Modeling Iterative Problem-Solving](https://arxiv.org/abs/2609.00940) | 该论文发布了CodeInsight大规模数据集，包含3,286名本科生在两个学年内2门C++入门课程中的超过300万次代码提交，用于建模迭代问题求解中学习者根据反馈反复修改的序列学习动态。 |
| [^144] | [TACS: Trajectory-Aware Candidate Selection for LLM Jailbreak Suffix Optimization](https://arxiv.org/abs/2608.29564) | 论文揭示了基于梯度的越狱后缀优化中“仅选当前损失最低候选”的短视性，提出轨迹感知候选选择框架TACS，通过轨迹感知代理、参考策略正则化和判别器卡方校正，使候选选择在搜索后期依然有效。 |
| [^145] | [Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation](https://arxiv.org/abs/2608.27817) | 该论文将音频大语言模型的评估建模为受控的调用决策问题，发现在已知封闭集任务上，有监督编码器（如CLAP和WavLM）无需调用生成式音频模型即可取得接近最优的准确率，从而揭示了传统“波形提示对比ASR转录”的评估方式混淆了声学证据获取与生成模型调用这两个因素。 |
| [^146] | [How Language Models Organize and Structure Moral Knowledge](https://arxiv.org/abs/2608.27402) | 本研究揭示了大型语言模型通过线性探针在表示空间中组织道德知识，其道德方向保持高度独立维度但共享道德特异性的正共同成分，表明模型能区分并整合不同道德基础。 |
| [^147] | [Hidden in the Request: Explaining Unethical LLM Compliance through Token Relevance](https://arxiv.org/abs/2608.23264) | 本文通过引入三种模态的探测方法，发现大语言模型在直接请求帮助时更易顺从于不道德行为，并利用层间相关性传播揭示其归因偏差——模型过度关注任务框架令牌而忽视不道德提示令牌，从而解释了对齐失败的机制。 |
| [^148] | [PersonaMem-v3: Toward Omni-Platform Personal Intelligence for Holistic User Understanding, Recommendation, and Agentic Tasks](https://arxiv.org/abs/2608.21381) | PersonaMem-v3 提出了一个基于百万级真实匿名数据的全平台个人智能基准，用于评估跨情境用户理解、可引导推荐、跨平台主动行为及过度个性化的避免。 |
| [^149] | [FTA-Mem: Fact-Time-Affect Anchored Memory for Low-Density Long-Term Dialogue](https://arxiv.org/abs/2608.16303) | 提出了一种名为FTA-Mem的结构化记忆框架，通过边界保留窗口分割和事实-时间-情感记忆单元，有效处理低密度长期对话中的信息碎片化问题，提升了长期记忆问答性能。 |
| [^150] | [LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure](https://arxiv.org/abs/2608.13545) | 本文提出了一个受教学控制的预训练语料库和模型，通过限制知识暴露范围，为研究语言模型的知识获取和能力边界提供了可解释的沙盒环境。 |
| [^151] | [The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding](https://arxiv.org/abs/2608.10137) | 该论文提出SHIM，巧妙利用约束解码工具已维护的解析器和词法分析器状态作为信号，通过轻量级离线训练的校正模块修正语言模型的下一词元概率，在不改动模型本身的前提下消除语法约束解码带来的分布偏差。 |
| [^152] | [CoMem: Reusing Transformer Depth across Queries with Persistent Intermediate Residuals](https://arxiv.org/abs/2607.28263) | CoMem通过为每个token持久化存储深度j处的中间残差，使Transformer拆分深度成为可调的服务轴，在重复查询共享文档时跳过已执行的较低层、仅恢复计算上层，在Qwen3-8B上实现1.403倍读取提速且存储仅需8 KiB/token，同时显式量化了质量-延迟-存储的权衡及其适用边界。 |
| [^153] | [What do Reward Models Memorize?](https://arxiv.org/abs/2607.24484) | 本文通过反事实记忆测量发现，判别式训练的奖励模型会错误记忆简单偏好对、记住数据集特定捷径，并过度泛化长度等简单启发式特征，导致其无法在情境相关场景中准确判断回复质量。 |
| [^154] | [Surprisal Theory is Tautological (without Rational Grounding)](https://arxiv.org/abs/2607.21574) | 本文论证，若不对语言模型施加额外的理性约束，惊讶度理论就是同义反复——任何加工难度模式都能找到与之相容的语言模型，因而该理论在此情况下不具备可证伪性。 |
| [^155] | [When Trivia Is Not Trivial: Everyday Knowledge Failures in Multilingual LLMs](https://arxiv.org/abs/2607.21445) | 该研究提出了覆盖 288 个主题的多语言常识问答基准 TriviaRoomQA，发现大模型在历史、地理、数学等知识密集型主题上表现出色，但在日常流行文化知识上明显薄弱。 |
| [^156] | [Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator](https://arxiv.org/abs/2607.07993) | 提出幻觉自博弈（HSP）框架，让检测器与演化中的生成器以对抗方式协同演化——利用RLAIF训练生成器产生越来越难检测的幻觉，从而不断自举提升幻觉检测器的性能。 |
| [^157] | [Progressive Disclosure for LLM-Maintained Wiki Knowledge Bases: a Preregistered Ablation](https://arxiv.org/abs/2607.04576) | 本文通过一项预注册消融实验，在四个页面内容完全相同、仅访问结构不同的LLM维护知识库版本上，检验渐进式披露（先读简洁目录和摘要、再按需打开页面）能否降低智能体问答的成本。 |
| [^158] | [BehaviorBench: Benchmarking Foundation Models for Behavioral Science Tasks](https://arxiv.org/abs/2606.24162) | 本文提出BehaviorBench基准，从行为预测与模拟、战略决策、被试特质推断和行为知识应用四大核心能力系统评估基础模型，并同时考察个体层面准确性与群体分布层面一致性，揭示当前领先模型在行为科学任务上仍面临挑战。 |
| [^159] | [Walk fast but be careful: Understanding Parallel Sampling in Masked Diffusion](https://arxiv.org/abs/2606.22976) | 本文利用图上随机游走作为可验证沙盒，从理论上证明掩码扩散模型中常用的并行去掩码评分策略（如最低熵）并不普遍优于随机并行采样，性能关键取决于图的条件依赖结构，并提出了免训练的二分采样器。 |
| [^160] | [MixedPEFT: Combining Multiple PEFT Methods with Mixed Objectives for Unsupervised Domain Adaptation](https://arxiv.org/abs/2606.22272) | 本文提出MixedPEFT，通过将可逆适配器与LoRA结合，并采用源域分类与目标域掩码语言建模的混合目标联合训练，实现了参数高效的无监督域自适应，在MNLI的20个域迁移场景上超越了UDapter和DANN等基线方法。 |
| [^161] | [Who Brought Easter Eggs to Eid? Auditing LLM-Generated Cultural Translation of Math Word Problems Across Languages and Regions](https://arxiv.org/abs/2606.11009) | 本文对三个主流大语言模型将数学应用题改编为七种高、低资源语言时的6,489个文化实体转换进行了大规模审计，揭示了不同模型的文化替换行为高度不一致，导致大规模个性化学习中文化多样性难以保留。 |
| [^162] | [An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models](https://arxiv.org/abs/2606.09843) | 本研究构建了首个从LLM行为中自下而上推导的心理测量工具，发现其维度（响应性、服从性、大胆性、谨慎性和冗长性）高度可靠，但LLM的自我报告仍无法预测其实际行为，表明人类特质类别与LLM行为之间存在根本性差异。 |
| [^163] | [WRIT: Write-Read Intensive Trajectory Synthesis for Multi-Turn User-Facing Agents](https://arxiv.org/abs/2606.02908) | 论文提出WRIT流程，通过合成同时强化写决策与读取工具证据收集的多轮代理训练轨迹，弥补了现有写密集型数据无法训练代理在大量信息收集后做出困难写决策的不足。 |
| [^164] | [Beyond Captions: Context-Grounded Reconstruction for Biomedical Multimodal Continued Pretraining](https://arxiv.org/abs/2606.01049) | 该论文提出上下文锚定重建框架，通过利用文章原生图引用将PMC-OA文献转换为指代连贯的图文交错序列，并构建高质量生物医学多模态持续预训练语料库PMC-InterCPT，解决了现有语料库将图像孤立为图注对、丢弃关键上下文的问题。 |
| [^165] | [How Far Do Auto-Interpretation Labels Generalize: A Controlled Study Across Languages, Scripts, and Rewordings](https://arxiv.org/abs/2606.00356) | 该研究以塞尔维亚语拉丁与西里尔双文字系统为受控实验平台，发现SAE特征本身确实具备跨语言、跨文字的语义泛化能力，但自动生成的解释标签往往无法跟上这种泛化，其跨语言失准率可比英语内部高出4倍。 |
| [^166] | [Knowledge boundary probing and demand-guided intervention for LLM-based power system code generation](https://arxiv.org/abs/2605.31478) | 该论文提出PowerCodeBench基准（面向pandapower的2000个冻结任务）以及无需更新权重的部署时工作流，通过文档驱动的L0-L3知识边界探测、查询侧需求估计选择分层API证据、以及执行反馈引导的针对性修复，显著提升了LLM电力系统代码生成的准确率。 |
| [^167] | [Latent Performance Profiling of Large Language Models](https://arxiv.org/abs/2605.30018) | 提出潜在性能剖析（LPP）框架，通过分析大语言模型的隐藏层激活与输出分布，从内部状态中提取与任务无关的性能诊断指标，弥补传统基准测试评估的不足。 |
| [^168] | [Handle with CARE: Can LLMs Reproduce How Online Communities React?](https://arxiv.org/abs/2605.27388) | 该论文提出CARE评估框架，将LLM模拟的话语与207个Reddit社区针对2,166篇真实新闻发表的9,947条真实反应进行基准对比，并借此揭示了现有社区条件化范式在再现真实社区反应方面的两个关键失败模式。 |
| [^169] | [Beyond Cooperative Simulators: Generating Realistic User Personas for Robust Evaluation of LLM Agents](https://arxiv.org/abs/2605.12894) | 提出了一种即插即用的控制层Persona Policies（PPol），利用进化式编码智能体自动发现角色生成程序，使LLM用户模拟器产生逼真且多样化的用户行为（如表达不清、缺乏耐心等），从而弥补模拟与现实之间的差距，实现对LLM智能体更稳健的评估。 |
| [^170] | [Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models](https://arxiv.org/abs/2605.10971) | 该论文发现从自回归模型移植的均匀干预调度方式在离散扩散语言模型上低效且损害生成质量，并通过稀疏自编码器揭示不同属性（如主题、情感）在去噪过程中具有差异显著的形成时间表，据此提出一种自适应调度机制，将干预集中在各属性正在形成的阶段，从而实现更高效且不破坏质量的多属性引导。 |
| [^171] | [APCD: Adaptive Path-Contrastive Decoding for Reliable Large Language Model Generation](https://arxiv.org/abs/2605.09492) | 本文提出APCD，一种无需重训练或微调的自适应多路径对比解码框架，通过熵驱动路径扩展等机制提升大语言模型生成的事实可靠性，克服了单解码轨迹方法的误差累积问题。 |
| [^172] | [Seeing Is No Longer Believing: Frontier Image Generation Models, Synthetic Visual Evidence, and Real-World Risk](https://arxiv.org/abs/2604.24197) | 本文是一篇叙述性综述，系统梳理了前沿图像生成模型的高逼真合成能力如何使合成图像获得“证据权威”，从而对新闻、金融、身份验证、医疗和法律等现实领域构成风险，并区分了厂商能力声明、已记录事件与潜在危害路径。 |
| [^173] | [Continuous Semantic Caching for Low-Cost LLM Serving](https://arxiv.org/abs/2604.20021) | 本文首次建立了不确定条件下连续查询空间中LLM语义响应缓存的严格理论框架，通过动态ε-网离散化与核岭回归相结合，突破了传统有限离散查询假设，实现低成本LLM服务。 |
| [^174] | [Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](https://arxiv.org/abs/2604.17260) | 该论文提出了一种时间细粒度的会议有效性评估新范式，将有效性定义为目标随时间达成的速率，并构建了包含130场会议、2,459个人工标注片段的AMI-ME数据集，同时开发了基于LLM作为评判者的自动评估框架。 |
| [^175] | [Retrieval-Augmented Generation Must Move Beyond Factual Grounding to Represent Diverse Opinions](https://arxiv.org/abs/2604.12138) | 本论文指出RAG系统因过度追求事实准确性而忽视观点多样性，提出了观点感知检索框架O-RAG，通过不确定性量化和基于Wasserstein距离的统一目标，将语料级情感分布的距离降低18-48%，从而更好地表征多元观点。 |
| [^176] | [Document Optimization for Black-Box Retrieval via Reinforcement Learning](https://arxiv.org/abs/2604.05087) | 提出DocOpt方法，通过GRPO强化学习以检索排序提升为奖励，直接训练LLM/VLM离线重写文档以优化黑盒检索器的检索效果，从而将昂贵的计算从延迟敏感的检索路径转移到离线阶段。 |
| [^177] | [Advancing LLM-based phoneme-to-grapheme for multilingual speech recognition](https://arxiv.org/abs/2603.29217) | 本文提出基于大语言模型的多语言音素到字素（P2G）方法，通过引入S-SKM蒙特卡洛近似等鲁棒性策略以及低资源语言过采样，在十语言CV-Lang10基准上将平均词错误率从10.56%降至7.66%。 |
| [^178] | [A conceptual framework for ideology in online discourse beyond the left and right](https://arxiv.org/abs/2603.18945) | 本文提出将意识形态概念化为多层次社会认知概念网络的新框架，突破了计算社会科学中单一左右党派轴线的研究局限，并将在线话语分析方法与意识形态理论相连接。 |
| [^179] | [Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators](https://arxiv.org/abs/2602.22647) | 提出 STATIC 方法，通过将前缀树展平为 CSR 稀疏矩阵，把不规则的树遍历转化为完全向量化的稀疏矩阵运算，从而在 TPU/GPU 上实现高效、可扩展的基于大语言模型生成式检索的约束解码。 |
| [^180] | [Just on Time: Token-Level Early Stopping for Diffusion Language Models](https://arxiv.org/abs/2602.11133) | 本文提出一种无需训练的词元级早停方法，利用模型预测和局部上下文的轻量级信号动态判断每个词元的收敛时机并提前冻结，大幅减少扩散语言模型的去噪步数，在保持生成质量的同时显著提升生成效率。 |
| [^181] | [Collective Behavior of AI Agents: the Case of Moltbook](https://arxiv.org/abs/2602.09270) | 对AI专属社交平台Moltbook的大规模数据分析表明，AI群体的集体行为在统计规律上与人类在线社区高度相似，但在点赞数与讨论规模的关系上存在关键差异。 |
| [^182] | [Attention-Mass Condensation for Sparse Decoding](https://arxiv.org/abs/2602.06317) | 该论文通过精确的遗漏质量恒等式和下游边距条件形式化了稀疏解码中注意力质量保留与稳定贪心决策之间的区别，并实验证明：尽管稀疏解码在分布质量上可接近稠密解码，但没有任何运行能完全复现稠密贪心解码的输出。 |
| [^183] | [WaveScat: Wavelet Scattering Front-Ends with Self-Supervised Features for Speech Deepfake Detection](https://arxiv.org/abs/2602.02980) | WaveScat通过小波散射变换将小波卷积与模非线性级联，生成形变稳定的多尺度特征，兼具手工特征的可解释性与高层次信息捕获能力，在多个语音深度伪造检测基准上大幅超越现有前端。 |
| [^184] | [Epistemic Constitutionalism Or: how to avoid coherence bias](https://arxiv.org/abs/2601.14295) | 本文提出为人工智能建立“认知宪法”——以明确且可争辩的元规范约束AI系统如何形成与表达信念，并通过来源归因的实证研究表明来源独立性并非中立的默认设置，以避免连贯性偏差。 |
| [^185] | [CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems](https://arxiv.org/abs/2601.05520) | 该论文提出CHisAgent多智能体框架，通过归纳、扩展、充实三个角色专业化阶段，从《二十四史》等中国古代文献中自动构建历史事件分类体系，克服了LLM在中国历史语境下推理能力不足和人工分类构建成本高的问题。 |
| [^186] | [HealthcareNLP: where are we and what is next?](https://arxiv.org/abs/2512.08617) | 本教程系统梳理了以患者和资源为导向的医疗健康NLP的核心子领域，涵盖数据/资源、NLP评估与可解释医疗AI三个层次，并弥补了现有综述对合成数据生成、检索增强生成等重要任务与方法的忽视，同时展望了未来挑战。 |
| [^187] | [Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM](https://arxiv.org/abs/2510.05544) | 提出基于激活压缩误差理论上界的帕累托引导低秩压缩框架PGSVD，通过异构秩分配在相同压缩率下为LLM/VLM实现更高精度与推理加速。 |
| [^188] | [SEER: Self-Enhancing Chain-of-Thought Compression for Reasoning Models](https://arxiv.org/abs/2509.14093) | 该论文通过实证研究揭示推理模型在代码生成中常产生冗长思维链并引发截断与不稳定生成问题，并据此提出SEER方法，通过自增强的方式压缩思维链以降低推理开销。 |
| [^189] | [APE: Selective Fine-tuning with Acceptance Criteria for Language Model Adaptation](https://arxiv.org/abs/2505.19912) | APE 是一种受进化优化启发的选择性微调方法，通过在小数据子集上评估多个候选参数更新并仅接受超过性能阈值者，在保持模型稳定性的同时以极少计算资源实现大型语言模型的高效适配。 |

# 详细

[^1]: 在可验证奖励强化学习（RLVR）中将探索与优化解耦

    Decoupling Exploration from Optimization in RLVR

    [https://arxiv.org/abs/2610.10536](https://arxiv.org/abs/2610.10536)

    提出探索-蒸馏框架，将RLVR中的探索与优化解耦：先用新颖性奖励训练探索者策略，再过滤其轨迹并蒸馏到不带新颖性奖励的学生策略中，从而在实现新策略发现的同时避免模型质量退化。

    

    现代语言模型在已经训练好的检查点基础上进行可验证奖励的强化学习（RLVR）。RLVR的一个关键承诺是发现新的推理策略。原则上，模型可以采样出其先前训练数据中不存在的新颖想法。然而在实践中，用强新颖性激励来增强RLVR的成功有限，并且可能会降低模型质量。由于可验证奖励仅监督模型知识和行为中很狭窄的一部分，这种退化难以恢复。因此，我们在一个称为探索-蒸馏的框架中将探索与优化解耦。我们训练一个或多个在奖励中带有新颖性奖励的探索者策略，对其轨迹进行正确性和质量过滤，然后将它们蒸馏到一个单独的学生策略中。学生策略随后在不带新颖性奖励的情况下进行训练。我们对上述过程重复多个轮次，交替……

    arXiv:2610.10536v1 Announce Type: cross  Abstract: Modern language models undergo reinforcement learning with verifiable rewards (RLVR) on top of already-trained checkpoints. A key promise of RLVR is the discovery of new reasoning strategies. In principle, a model can sample novel ideas absent from its prior training data. In practice, however, augmenting RLVR with strong novelty incentives has seen limited success and can degrade model quality. Because verifiable rewards supervise only a narrow slice of the model's knowledge and behavior, such degradations are difficult to recover from. Instead, we decouple exploration from optimization in a framework we call Exploration-Distillation (ExpDis). We train one or more explorer policies with a novelty bonus in the reward, filter their trajectories for correctness and quality, and distill them into a separate student policy. The student policy is then trained without a novelty bonus. We repeat the above procedure for several rounds, alterna
    
[^2]: EngramEdit：通过条件记忆在大语言模型中实现解耦的知识更新

    EngramEdit: Decoupled Knowledge Updates in LLMs through Conditional Memory

    [https://arxiv.org/abs/2610.10533](https://arxiv.org/abs/2610.10533)

    提出 EngramEdit 方法，利用条件记忆架构将事实知识存储与通用计算解耦，在不修改 Transformer 主干的情况下跨多种表达方式安全地更新大语言模型中的事实知识。

    

    条件记忆架构（如 DeepSeek Engram）通过输入 n-gram 查找学习到的嵌入向量，以有限的额外计算扩展大语言模型（LLM）的容量。除了模型扩展之外，该架构还展示了将事实知识存储与通用计算解耦的潜力，为在保持 Transformer 主干网络固定的情况下更新事实知识提供了一条有前景的路径。然而，实现这一潜力具有挑战性，因为同一事实的不同表达方式可能激活不同的 n-gram 嵌入，而更新共享嵌入可能会无意中改变模型对其他事实的预测。我们提出了 EngramEdit，通过条件记忆实现解耦的知识更新。EngramEdit 首先计算目标记忆表示，使模型能够在多种表达方式下预测更新后的事实；然后联合更新共享的 n-gram 嵌入以匹配这些目标表示。

    arXiv:2610.10533v1 Announce Type: new  Abstract: Conditional memory architectures such as DeepSeek Engram use input n-grams to look up learned embeddings, expanding the capacity of large language models (LLMs) with limited additional computation. Beyond model scaling, this architecture has demonstrated the potential to decouple factual knowledge storage from general-purpose computation, offering a promising route to updating factual knowledge while keeping the Transformer backbone fixed. Realizing this potential is challenging because different expressions of a fact may activate different n-gram embeddings, while updating shared embeddings can unintentionally change the model's predictions about other facts. We propose EngramEdit for decoupled knowledge updates through conditional memory. EngramEdit first computes target memory representations that make the model predict the updated fact across multiple expressions. It then jointly updates the shared n-gram embeddings to match these ta
    
[^3]: 先改述，再行动：视觉-语言-动作模型中语言敏感性的表征与缓解

    Rephrase Before You Act: Characterizing and Mitigating Language Sensitivity in Vision-Language-Action Models

    [https://arxiv.org/abs/2610.10526](https://arxiv.org/abs/2610.10526)

    本文揭示了视觉-语言-动作模型对指令措辞的极端敏感性（单词改动可使成功率波动数十个百分点），并提出无需修改策略、由大语言模型将措辞评分证据提炼为十余条改述规则并在部署时应用的方法来缓解该问题。

    

    视觉-语言-动作模型（VLA）对指令措辞极其敏感，且并不继承其底层视觉-语言模型所具备的语言鲁棒性。仅仅一个词的改动就能使成功率波动数十个百分点：π0.5 在 LIBERO 灶台任务中，对 "switch on the stove"（打开灶台）的成功率为 100%，而对 "switch on the hot plate"（打开加热板）仅为 2%；即使是用改述数据增强微调过的 π0 检查点，仍表现出高达 61 个百分点的波动。我们通过经过统计检验的单次编辑波动分析以及预言机短语搜索来刻画这种敏感性，结果表明仅靠措辞差异就能几乎抹平分布内任务与分布外任务之间 21 个百分点的差距。随后，我们在不修改策略本身的前提下降低这种敏感性。由于这种敏感性是系统性的，它可以被表达为显式规则：我们对少量训练任务的多种措辞进行评分，让大型语言模型将这些证据提炼为十到二十条改述规则，并在部署时加以应用……

    arXiv:2610.10526v1 Announce Type: cross  Abstract: Vision-language-action models (VLAs) are strikingly sensitive to instruction phrasing and do not inherit the language robustness of the vision-language models they are built on. A one-word edit can move success by tens of points: $\pi_{0.5}$ turns on a LIBERO stove 100% of the time for "switch on the stove" and 2% for "switch on the hot plate", and a $\pi_0$ checkpoint finetuned with rephrase augmentation still shows swings of up to 61 points. We characterize this sensitivity with statistically tested single-edit swings and an oracle phrase search, which shows that phrasing alone nearly closes the 21-point gap between in-distribution and out-of-distribution tasks. We then reduce it without modifying the policy. Because the sensitivity is systematic, it can be expressed as explicit rules: we score many phrasings of a few training tasks, have a large language model distill the evidence into ten to twenty rephrasing rules, and at deployme
    
[^4]: 你的提示词应该做得更多：检索指令在嵌入模型中的影响

    Your Prompt Should Do More: Effects of Retrieval Instructions in Embedding Models

    [https://arxiv.org/abs/2610.10508](https://arxiv.org/abs/2610.10508)

    本文揭示了嵌入模型在检索任务中难以遵循指令的内在机制，发现查询侧干扰项是导致指令遵循失败的关键因素，并证明在微调中加入查询侧干扰项可显著提升模型遵循指令的能力，同时对其他任务影响极小。

    

    提示式嵌入模型近来受到越来越多的关注，特别是在检索领域，其中详细的检索指令作为检索提示的一部分被提供。若干新的数据集和研究已经考察了这一设置，结果表明当前的嵌入模型往往难以可靠地遵循这些指令。在本文中，我们研究了在非对称检索任务中指令究竟如何影响检索查询表示的机制。我们表明，当评估中包含查询侧干扰项时，模型甚至无法遵循简单的任务指令。我们假设这种行为是由当前嵌入模型的训练设置及其评估方式所驱动的，并通过实验证明，在微调中加入查询侧干扰项可以带来显著的改进，同时对其他任务的影响极小。

    arXiv:2610.10508v1 Announce Type: new  Abstract: Prompted embedding models have recently received increasing attention, particularly for retrieval, where detailed retrieval instructions are provided as part of the retrieval prompt. Several new datasets and studies have examined this setting, showing that the current embedding models often struggle to follow such instructions reliably. In this paper, we study the mechanism of how instructions actually affect the representations of retrieval queries in asymmetric retrieval tasks. We show that models can fail to follow even simple task instructions when query-side distractors are included in the evaluation. We hypothesize that this behavior is driven by the training setup of current embedding models and their evaluation, and show that fine-tuning with added query-side distractors leads to substantial improvements, with minimal effect on other tasks.
    
[^5]: 无真值情境下的有效性评估：陈述偏好经济学能为语言模型评价提供什么

    Validity Without Ground Truth: What Stated-Preference Economics Offers the Evaluation of Language Models

    [https://arxiv.org/abs/2610.10506](https://arxiv.org/abs/2610.10506)

    本文提出将陈述偏好经济学中用于无真值情形的效度评估框架（内容效度、建构效度、信度、激励相容性、后果性等）迁移到大语言模型评估中，并通过水质经济价值评估调查对六个模型进行了实证演示。

    

    如今向大型语言模型提出的许多问题并没有可用于评分的正确答案：一项政策价值几何、用户应当选择哪个选项、如何权衡相互冲突的价值观。陈述偏好经济学数十年来一直面临这一难题——它在不知道真实价值的情况下评判调查回复，依靠的是一套由效度及相关概念构成的框架：内容效度、建构效度与效标效度、信度、激励相容性以及后果性。我们主张这一框架是评估语言模型的一种通用方法，并逐一阐述了这些概念对大语言模型评估的意义。我们通过对六个模型实施一项已发表的水质陈述偏好经济价值评估调查（Vossler et al. 2023）来演示这一方法。在这一经济学应用中，效度检验以经济理论预测的形式呈现：需求曲线应当向下倾斜，支付意愿应当随商品范围的变化而相应调整。

    arXiv:2610.10506v1 Announce Type: cross  Abstract: Many of the questions now put to large language models have no correct answer to score against: what a policy is worth, which option a user should choose, how to weigh competing values. Stated-preference economics has faced this problem for decades. It judges survey responses without knowing the true value, through a framework of validity and related concepts: content, construct, and criterion validity, reliability, incentive compatibility, and consequentiality. We argue that this framework is a general method for evaluating language models, and we set out what each concept means for LLM evaluation. We demonstrate the approach using a published water-quality stated preference economic valuation survey (Vossler et al. 2023) administered to six models. In this economic application, the validity tests take the form of predictions from economic theory: demand should slope down, and willingness to pay should respond to the scope of the good
    
[^6]: PHRBench：大语言模型幻觉后推理的行为学评估

    PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs

    [https://arxiv.org/abs/2610.10455](https://arxiv.org/abs/2610.10455)

    提出了PHRBench这一受控基准，通过幻觉顺从、幻觉规避和启发式纠正等行为指标，在四个领域对18个大语言模型处理幻觉前提的推理轨迹进行评估，发现模型成功从幻觉中恢复并得出正确答案的情况仍然相对罕见。

    

    幻觉信息可能在多阶段大语言模型系统中传播，并成为后续推理上下文的一部分。现有关于幻觉后推理（PHR）的研究主要刻画最终结果的变化和总体的推理动态，而对模型在响应层面如何处理幻觉前提的理解仍然不足。在本工作中，我们提出了PHRBench，这是一个受控基准，用于在四个领域、18个大语言模型上对行为结构化的幻觉后推理进行评估。PHRBench通过幻觉顺从、幻觉规避和启发式纠正三个指标，独立于最终答案正确性地刻画每条推理轨迹，并将“有洞察力的轨迹”定义为最终成功纠正并达到正确答案的轨迹。在4820个受控实例中，我们发现成功的恢复仍然相对罕见，并且与推理轨迹上更频繁的信念更新相关（摘要原文在此处被截断）。

    arXiv:2610.10455v1 Announce Type: new  Abstract: Hallucinated information can propagate through multi-stage LLM systems and become part of the context for subsequent reasoning. Existing studies of post-hallucination reasoning (PHR) mainly characterize changes in final outcomes and aggregate reasoning dynamics, leaving how models resolve hallucinated premises at the response level insufficiently understood. In this work, we introduce PHRBench, a controlled benchmark for behaviorally structured PHR across four domains and 18 large language models. PHRBench characterizes each reasoning trajectory independently of final-answer correctness through Hallucination Compliance, Hallucination Avoidance, and Heuristic Correction, and defines an insightful trajectory as successful correction that ultimately reaches the correct answer. Across 4820 controlled instances, we find that successful recovery remains relatively rare and is associated with more frequent belief updates along the reasoning tra
    
[^7]: RunningTab：通过环境侧标签页实现直接的工作区交互

    RunningTab: Direct Workspace Interaction with Environment-Side Tabs

    [https://arxiv.org/abs/2610.10444](https://arxiv.org/abs/2610.10444)

    提出RunningTab框架，通过由环境维护的“环境侧标签页”跟踪任务需求、已读文件与未打开文件，解决LLM智能体在直接工作区交互中因上下文窗口限制而遗漏关键内容的问题。

    

    许多知识型工作都需要基于工作区中已有的文件产出新的交付成果，而LLM智能体正开始接管这类工作。通过直接的语料库交互，智能体可以在终端中搜索并读取这些文件中的任何一个，无需建立索引；我们称以这种方式从多个文件中产出交付成果为“直接工作区交互”（DWI）。然而，能够访问文件只是任务的一半：没有任何机制来跟踪任务要求什么、已经读取了什么、以及哪些文件被列出但从未打开——这些信息都会从上下文窗口中溜走而不留痕迹，因此智能体可能提取了某个图表，却最终交付一份缺少该图表的报告。为解决这一问题，我们提出了RunningTab，一个为直接工作区交互配备“环境侧标签页”的框架：这是一个按任务记录任务尚未完成事项的清单，由环境与智能体共同维护。具体而言，智能体添加其需求，而环境则……

    arXiv:2610.10444v1 Announce Type: cross  Abstract: Much knowledge work produces new deliverables from files a workspace already holds, and LLM agents are beginning to take such work over. Through direct corpus interaction, an agent can search and read any of those files from a terminal with no indexing, and producing a deliverable from many of them in this way is what we call direct workspace interaction (DWI). Reaching the files, however, is only half the task: nothing keeps track of what the task asks for, what has been read, and what was listed but never opened, all of which slip through the context window without leaving a trace, so an agent may extract a figure and still deliver a report without it. To address this, we present RunningTab, a framework that equips direct workspace interaction with an environment-side tab: a per-task record of what the task still owes, kept by the environment alongside the agent. Specifically, the agent adds its requirements, while the environment re
    
[^8]: CoTrace：基于框架-模型协同进化的终端智能体训练数据配方

    CoTrace: Data Recipes for Training Terminal Agents with Harness-Model Co-Evolution

    [https://arxiv.org/abs/2610.10426](https://arxiv.org/abs/2610.10426)

    提出了交替式框架-模型协同进化框架及框架感知的数据配方 CoTrace，通过轨迹路由、来源匹配与课程刷新，以反复出现的执行失败指导框架合成，并使策略训练严格基于与当前运行时框架匹配的已验证轨迹，从而提升终端智能体性能。

    

    终端智能体的能力同时取决于模型权重与运行时框架，后者负责格式化提示词、绑定工具并处理错误恢复。现有的框架-模型协同进化方法虽然能同时改进这两个组件，却往往把框架搜索过程中产生的轨迹当作不加区分的回放缓冲区。这种做法忽视了一个事实：轨迹对模型训练的价值取决于其生成时所依托的框架。为了系统地分析这一接口，我们建立了一个交替式协同进化框架，通过基于组件的晋升决策将框架搜索与策略训练解耦。在该框架内，我们提出了 CoTrace——一种框架感知的数据配方，显式地管理轨迹路由、来源匹配与课程刷新。在 CoTrace 下，反复出现的执行失败用于指导框架合成，而策略训练则严格以与所采用的运行时框架相匹配的、经过验证的 rollout 为条件。

    arXiv:2610.10426v1 Announce Type: new  Abstract: Terminal-agent capability depends jointly on model weights and the runtime harness that formats prompts, binds tools, and handles error recovery. Existing harness-model co-evolution approaches improve both components, yet often treat trajectories produced during harness search as an undifferentiated replay buffer. This practice overlooks that a trajectory's value for model training depends on the harness under which it was generated. To systematically analyze this interface, we establish an alternating co-evolution framework that decouples harness search and policy training through component-wise promotion decisions. Within this framework, we introduce CoTrace, a harness-aware data recipe that explicitly governs trajectory routing, provenance matching, and curriculum refresh. Under CoTrace, recurring execution failures guide harness synthesis, while policy training is strictly conditioned on verified rollouts matched to the adopted runti
    
[^9]: 哪一次Rollout教会了它？BehaviorTrace与在线强化学习中训练数据归因的局限

    Which Rollout Taught It That? BehaviorTrace and the Limits of Training-Data Attribution in Online RL

    [https://arxiv.org/abs/2610.10422](https://arxiv.org/abs/2610.10422)

    该工作发布了BehaviorTrace开源评估框架，通过植入已知成因的行为实验发现，在线RL中训练数据归因方法的表现很大程度上源于梯度大小和模型流畅度等混淆因素，揭示了现有归因信号的可靠性局限。

    

    当强化学习教会一个语言模型一种新行为时，我们能否找出是哪些训练rollout教会了它？当某种归因方法声称可以做到时，我们又如何确认其答案是真实可靠的？我们在使用GRPO的在线RL微调上研究这两个问题，采用一种成因已知的植入行为（planted behavior）实验设置。我们发布了BehaviorTrace——一个开放的评估工具框架，它结合了全梯度草图技术、植入行为设置，以及对梯度大小、流畅度、性能提升空间、随机种子与生成样本差异等混淆因素的控制。在Qwen2.5-1.5B模型上的三个随机种子实验中，大量表观上的归因信号实际来自混淆因素。一个仅按梯度大小对训练步骤排序、不包含任何行为目标的对照组，达到了随机基线的4.2至4.5倍，并在三个种子中的两个上匹配甚至超越了最佳的目标归因估计器。在饱和检查点处，模型流畅度预测行为标签的能力至少不逊于我们对比的所有梯度方法。（原文摘要在此处被截断）

    arXiv:2610.10422v1 Announce Type: cross  Abstract: When reinforcement learning teaches a language model a new behavior, can we find the training rollouts that taught it? And when an attribution method says it can, how do we know the answer is real? We study both questions on online RL fine-tuning with GRPO, using a planted behavior with a known cause. We release BehaviorTrace, an open evaluation harness that combines full-gradient sketching, the planted-behavior setup, and controls for gradient magnitude, fluency, headroom, and variation across seeds and generation draws. Across three seeds on Qwen2.5-1.5B, much of the apparent attribution signal comes from confounds. A control that ranks training steps by gradient size alone, with no behavior target, reaches 4.2 to 4.5 times chance and matches or beats the best targeted estimator on two of three seeds. At saturated checkpoints, model fluency predicts the behavior label at least as well as every gradient method we compared it with. Onc
    
[^10]: 通过直接最小化期望解码轮数来训练并行投机草稿模型

    Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds

    [https://arxiv.org/abs/2610.10411](https://arxiv.org/abs/2610.10411)

    本文将投机解码建模为马尔可夫奖励过程，提出直接最小化期望解码轮数（EDR）的训练目标，以优化并行投机草稿模型的全局解码效率。

    

    投机解码通过使用低成本的草稿模型提出候选词元，再由完整规模的目标模型并行验证，从而加速大语言模型的推理。并行和半自回归（semi-AR）草稿模型通过单次前向传播提出整个词块来提高起草效率，但训练这类模型带来了新的困难：给定位置的草稿分布取决于解码轮从哪里开始，而每轮从哪里开始又取决于之前各轮接受词元的数量。现有的训练目标通常依赖于忽略这种跨轮耦合的块内局部替代目标，因此无法直接优化全局解码效率。在这项工作中，我们将投机解码表示为马尔可夫奖励过程，为训练和评估此类草稿模型建立了一个理论框架。这一表述产生了期望解码轮数（EDR）目标，该目标对局部拒绝进行加权……

    arXiv:2610.10411v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model inference by using a low-cost draft model to propose tokens that the full-size target model verifies in parallel. Parallel and semi-autoregressive (semi- AR) drafters improve drafting efficiency by proposing an entire block in a single forward pass, but training them raises a new difficulty: the draft distribution for a given position depends on where the decoding round starts, and where rounds start depends on how many tokens earlier rounds accepted. Existing training objectives typically rely on block-local surrogates that ignore this cross-round coupling, and therefore do not directly optimize the global decoding efficiency. In this work, we develop a theoretical framework for training and evaluating these drafters by representing speculative decoding as a Markov reward process. This formulation yields the Expected Decoding Rounds (EDR) objective, which weights local rejection co
    
[^11]: 大型语言模型在被提示进行不真实回答时的推理Token激增现象

    Reasoning-Token Spikes Under Prompted Untruthful Responding in Large Language Models

    [https://arxiv.org/abs/2610.10405](https://arxiv.org/abs/2610.10405)

    该论文基于认知负荷理论，提出利用推理token数量这一无需访问思维链内容的低带宽信号，发现大语言模型在被提示进行不真实回答时会出现推理token数量的激增，从而为检测模型欺骗行为提供了新方法。

    

    监控推理型人工智能模型的思维链仍然是检测此类模型中欺骗行为及其他形式不良行为的关键方法。然而，基于语义的思维链监控依赖于推理轨迹清晰可读、对产生模型行为的底层计算足够忠实，更不用说其还必须是可访问的。此外，越来越多的证据表明，思维链输出可能很快变得难以解读或不忠实，甚至可能不再可访问。基于认知负荷理论，我们研究了一种低带宽信号——生成的推理token数量——它不需要访问推理轨迹的内容。三个具备推理能力的大型语言模型回答了210道选择题——涵盖分析性、描述性和规范性推理类型以及道德与非道德领域——在系统提示指示它们进行不真实回答的条件下（原文在此处截断）。

    arXiv:2610.10405v1 Announce Type: cross  Abstract: Monitoring the chain-of-thought of reasoning artificial intelligence (AI) models remains a key approach to detecting deception and other forms of misbehavior in such models. However, semantic chain-of-thought monitoring depends on reasoning traces being legible and sufficiently faithful to the underlying computations that produced the model's behavior, not to mention accessible. Moreover, there is increasing evidence that chain-of-thought outputs may soon become illegible or unfaithful, if they even remain accessible. Based on cognitive load theory, we investigate a lower-bandwidth signal -- the number of reasoning tokens generated -- which does not require access to the content of the reasoning trace. Three reasoning-capable large language models answered 210 multiple-choice questions -- across analytic, descriptive, and normative reasoning types as well as moral and non-moral domains -- under system prompts instructing them to respon
    
[^12]: 使用大语言模型的爱沙尼亚语文档级文本简化

    Document-Level Text Simplification in Estonian Using Large Language Models

    [https://arxiv.org/abs/2610.10378](https://arxiv.org/abs/2610.10378)

    本研究首次系统评估了五种多语言大语言模型在爱沙尼亚语这一低资源形态丰富语言上的文档级文本简化能力，通过对比三种提示策略并结合自动指标与人工标注，发现 Gemini-2.0 和 LLaMA-3.3 的输出达到接近母语的流畅度。

    

    文档级文本简化涉及超越句子内部编辑的转换，需要处理语篇连贯性、指代消解和跨段落一致性等问题。尽管高资源语言的句子级简化已取得诸多进展，但在爱沙尼亚语等形态丰富、资源匮乏的语言中，文档级简化在很大程度上仍未被探索。本研究对五个最先进的多语言大语言模型在爱沙尼亚语文档级简化任务中的表现进行了全面评估。研究考察了三种提示策略：单次生成、基于流水线的模块化代理以及指导原则增强的流水线。该评估框架整合了评估可读性、语义保留和语篇连贯性的自动指标，并辅以结构化的人工标注协议。研究结果表明，Gemini-2.0 和 LLaMA-3.3 能够生成具有接近母语水平流畅性的输出……

    arXiv:2610.10378v1 Announce Type: new  Abstract: Document-level text simplification involves transformations that go beyond sentence-internal edits, addressing discourse coherence, anaphora resolution, and cross-paragraph consistency. Despite advances in sentence-level simplification for high-resource languages, document-level simplification in morphologically rich, low-resource languages such as Estonian remains largely unexplored. This study presents a comprehensive evaluation of five state-of-the-art multilingual large language models (LLMs) for document-level simplification in Estonian. Three prompting strategies are examined: single-pass generation, pipeline-based modular agents, and guideline-augmented pipelines. The evaluation framework integrates automatic metrics assessing readability, semantic preservation, and discourse coherence, alongside a structured manual annotation protocol. The findings indicate that Gemini-2.0 and LLaMA-3.3 produce outputs with near-native fluency an
    
[^13]: 输入盲化对照在多项选择评估中为层程序带来显著的神谕提升空间

    Input-Blind Controls Produce Substantial Oracle Headroom for Layer Programs in Multiple-Choice Evaluation

    [https://arxiv.org/abs/2610.10368](https://arxiv.org/abs/2610.10368)

    本研究发现在多项选择评估中，输入盲化的对照扰动所产生的神谕提升空间反而超过真实的层跳过与重复程序，说明仅凭选择增益无法解释所选层程序为何有效。

    

    自适应计算旨在通过针对每个输入定制执行方式来改进语言模型的推理。对于层程序，在实用的选择器可用之前，神谕评估利用已知答案来估计这种灵活性带来的潜在增益。然而，来自选择的增益本身并不能解释所选程序为何有效。本研究利用两个模型上的32个层跳过与重复程序以及4,413个多项选择题目来考察这一区别。该分析将真实程序相对于在无评估提示情况下所选固定动作的增益，与相同位置上输入盲化扰动的增益进行比较，并在另一个提示上重新评估选择结果。在共享选项顺序的情况下，这些对照在Qwen3-4B-Base和Llama-3.1-8B上分别产生了10.2-11.8和15.6-19.4个百分点的提升空间，在每模型的全部三次随机方向抽取中均超过真实程序的9.0和10.1。它们仅在答案改变率上与真实程序相当，且排序取决于（原文在此处截断）。

    arXiv:2610.10368v1 Announce Type: cross  Abstract: Adaptive computation aims to improve language-model inference by tailoring execution to each input. For layer programs, oracle evaluations use known answers to estimate the potential gain from this flexibility, before a practical selector is available. However, a gain from selection does not by itself explain why the chosen programs help. This study examines this distinction using 32 layer-skipping and repetition programs on two models and 4,413 multiple-choice items. The analysis compares their gains over a fixed action selected without the evaluation prompt with those of input-blind perturbations at the same sites, re-evaluating selections on another prompt. With shared option order, the controls give 10.2-11.8 and 15.6-19.4 percentage points of headroom on Qwen3-4B-Base and Llama-3.1-8B, exceeding the real programs' 9.0 and 10.1 in all three random-direction draws per model. They match answer-change rate only, and the ordering depen
    
[^14]: 通过任务进度学习行动：从紧凑的教师监督中蒸馏小型智能体

    Learning to Act with Task Progress: Distilling Small Agents from Compact Teacher Supervision

    [https://arxiv.org/abs/2610.10332](https://arxiv.org/abs/2610.10332)

    提出任务进度蒸馏（TPD）离线方法，通过为每个演示动作标注简短任务阶段标签，使1.7B学生模型仅用404个演示在ALFWorld上达到72.4%的未见任务成功率，显著超越推理训练基线的48.3%。

    

    从大模型演示中学习，提供了一种训练小型智能体的方式，使这些智能体能够完成重复性任务，而无需在每一步都调用大模型。一个核心的设计选择是：在包含推理、动作和任务进度信息的教师轨迹中应保留哪些内容。我们提出了任务进度蒸馏（Task-Progress Distillation, TPD），这是一种离线方法，它将每个演示动作与一个描述当前任务阶段的简短标签配对。学生模型学习这些紧凑的目标，并通过联合对可采纳的“阶段—动作”对进行评分来选择动作，随后由确定性的执行框架在环境中执行。在ALFWorld上，使用404个演示训练的1.7B学生模型，无论采用TPD还是仅动作监督，均达到72.4%的平均未见任务成功率，相比之下，使用受限动作选择的推理训练学生模型仅为48.3%。显式阶段在200个演示时提供了额外收益，将成功率从48.0%…（原文此处截断）

    arXiv:2610.10332v1 Announce Type: new  Abstract: Learning from large-model demonstrations offers a way to train small agents that can complete recurring tasks without calling a large model at every step. A central design choice is what to retain from teacher trajectories that contain reasoning, actions, and information about task progress. We introduce Task-Progress Distillation (TPD), an offline approach that pairs each demonstrated action with a short label describing the current task stage. The student learns these compact targets and selects actions by jointly scoring admissible stage--action pairs, which a deterministic harness executes in the environment. On ALFWorld, a 1.7B student trained with 404 demonstrations achieves 72.4\% mean unseen task success with either TPD or action-only supervision, compared with 48.3\% for a reasoning-trained student using constrained action selection. Explicit stages provide an additional benefit at 200 demonstrations, improving success from 48.0
    
[^15]: 没有人真正就情感达成一致：人类、定制工具与大语言模型在社交媒体文本情感分析上均表现挣扎

    Nobody Truly Agrees on Sentiment: Humans, Bespoke Tools, and LLMs Struggle with Social Media Texts

    [https://arxiv.org/abs/2610.10318](https://arxiv.org/abs/2610.10318)

    该研究以人类标注者为基准，用Cohen's kappa和Fleiss' kappa评估了专用情感分析工具与大语言模型在100条推文上的一致性，发现情感判断本身高度主观——即使人类之间也仅有一致性一般，二分类任务一致性高于三分类，且Twitter-roBERTa-base表现最佳。

    

    社交媒体是实时公众情绪的丰富来源，但人们在使用广泛流行的情感分析工具时，往往并不了解其局限性。在本研究中，我们以六名人类标注者为基准，在100条推文上评估了三种定制情感分析工具以及三种大语言模型（LLM：Qwen3-32B、GPT-OSS-120B、Llama-4-Maverick-17B）的评分者间一致性。我们采用两种统计指标来衡量一致性：用于两两比较的Cohen's kappa和用于多评分者的Fleiss' kappa。结果显示，即便是人类标注者之间也仅表现出一般水平的一致性，这凸显了情感分析的主观性。无论是人类还是自动化工具，在二分类情感判定（负面 vs. 非负面、正面 vs. 非正面）下的一致性均高于三分类情感判定。其中，Twitter-roBERTa-base模型展现出最强的一致性。

    arXiv:2610.10318v1 Announce Type: new  Abstract: Social media is a rich source of real-time public sentiment, but widely used sentiment analysis tools are often applied without understanding their limitations. In this study, we evaluate the inter-rater reliability of three bespoke sentiment analysis tools (TextBlob, VADER, and Twitter-roBERTa-base) and three large language models (LLMs: Qwen3-32B, GPT-OSS-120B, Llama-4-Maverick-17B) against six human raters across 100 tweets. We measured agreement using two statistical measures: Cohen's kappa for pairwise comparisons and Fleiss' kappa for multiple raters. Even among the human raters, our results showed only fair agreement, highlighting the subjectivity of sentiment analysis. Higher agreement was observed under the binary sentiment classification (negative vs. non-negative and positive vs. non-positive) than under the three-class classification across both humans and automated tools. The Twitter-roBERTa-base model showed the strongest a
    
[^16]: SemanticFold：潜在序列压缩分离语言建模、可解码性与推理能力

    SemanticFold: Latent Sequence Compression SeparatesLanguage Modeling, Decodability, and Reasoning

    [https://arxiv.org/abs/2610.10304](https://arxiv.org/abs/2610.10304)

    提出SemanticFold潜在序列压缩方案，通过在学习的边界折叠前缀隐藏状态来压缩提示前缀，发现压缩对语言建模、可解码性和推理能力的影响是非单调的且各自具有不同的压缩阈值，证明这些能力可以相互分离。

    

    我们研究提示词前缀的潜在序列压缩是否能保留大型语言模型在推理过程中所依赖的能力。我们提出了SemanticFold，一种在学习的边界处折叠前缀隐藏状态的压缩方案，并在五个模型规模上进行评估：Qwen3-1.7B、Qwen3-8B、SmolLM2-1.7B、Pythia-1.4B和Pythia-6.9B。我们采用固定目标协议：冻结的前缀以原生方式执行或被压缩，两种方式均通过教师强制使用完全相同的续写词元。这一设计排除了目标选择对似然变化的解释。我们考察了五个终点类别：固定目标负对数似然、有限标签推理准确率、线性探针可访问性、开放式生成以及系统级内存和延迟。我们发现压缩使这些终点发生非单调变化，且它们不共享统一的压缩阈值。在Qwen3-1.7B上，当压缩比R=1.7时，压缩最小……

    arXiv:2610.10304v1 Announce Type: cross  Abstract: We study whether latent sequence compression of prompt prefixes preserves the capabilities that large language models rely on during inference. We introduce SemanticFold, a compression scheme that folds prefix hidden states at learned boundaries, and evaluate it across five model scales: Qwen3-1.7B, Qwen3-8B, SmolLM2-1.7B, Pythia-1.4B, and Pythia-6.9B. We use a fixed-target protocol: a frozen prefix is executed natively or compressed, and both arms teacher-force identical continuation tokens. This design rules out target-selection explanations for likelihood changes. We examine five endpoint families: fixed-target negative log-likelihood, finite-label reasoning accuracy, linear probe accessibility, open-ended generation, and systems-level memory and latency. We find that compression moves these endpoints non-monotonically and that they do not share a single compression threshold. On Qwen3-1.7B at compression ratio R=1.7, compressed-min
    
[^17]: PatchBench：衡量激活修补中的附带损害

    PatchBench: Measuring Collateral Damage in Activation Patching

    [https://arxiv.org/abs/2610.10276](https://arxiv.org/abs/2610.10276)

    提出PatchBench基准，用于衡量激活修补在修复LLM越狱行为时对无关行为造成的附带损害，从而区分真正的选择性修复与更广泛的局部行为抑制。

    

    LLM安全补丁可能通过了某个基准测试，却仍然是一个糟糕的修复。这种风险在越狱修复中尤为突出，因为其目标是在不改变无关行为的前提下纠正特定的不安全行为。一个补丁可能能够阻止精确的评估提示，却在相近的有害变体上失效，或者通过过度拒绝共享其措辞或结构的良性提示来抑制有害行为。现有协议主要测试模型是否会被攻破，而聚合指标（攻击成功率、拒绝率、全局能力）无法区分选择性修复与更广泛的局部抑制。为了填补这一空白，我们引入了PatchBench，这是一个基于实证观察到的、诱导出可操作有害答案的模型特定越狱失败案例构建的基准。我们从37个公开数据集中的27,870个提示出发，筛选出15,314个英文提示，并对8个开源指令微调模型进行查询。结合WildGuard过滤、成对Elo排名……

    arXiv:2610.10276v1 Announce Type: cross  Abstract: An LLM safety patch can pass a benchmark while still being a poor repair. This risk is especially acute for jailbreak repairs, where the goal is to correct a specific unsafe behaviour without changing unrelated behaviours. A patch may block exact evaluation prompts yet fail on close harmful variants, or suppress harmful behaviour by over-refusing benign prompts that share its wording or structure. Existing protocols primarily test whether models can be broken, while aggregate metrics (attack success, refusal rates, global capability) cannot distinguish selective repairs from broader local suppression. To address this gap, we introduce PatchBench, a benchmark of empirically observed model-specific jailbreak failures inducing actionable harmful answers. Starting from 27,870 prompts from 37 public datasets, we curate 15,314 English prompts and query 8 open-source instruction-tuned models. Combining WildGuard filtering, pairwise Elo rankin
    
[^18]: LLM的说服力因评估方法而异

    LLM Persuasion Is in the Eye of the Evaluation

    [https://arxiv.org/abs/2610.10232](https://arxiv.org/abs/2610.10232)

    该研究将九种已发表的自动化说服力评估方法统一到相同设置下，对同一批十五个大语言模型进行测试，发现不同方法给出的模型排名并不一致，表明LLM的说服力评估结果高度依赖于所用评估方法。

    

    大语言模型（LLM）在说服力方面已被证明能够匹敌甚至超越人类专家。虽然其说服能力在教育、健康传播等有益用途上前景可期，但也可能被用于操纵和传播错误信息，因此对其进行评估已成为开发者和监管者日益重视的优先事项。然而，这类评估仍然碎片化：不同研究对“说服”的界定各不相同，宽泛的结论往往基于狭窄的、针对特定情境的评估。自动化方法（通常以人类研究为蓝本设计）提供了一条直接比较这些评估的途径，因为它们可以在相同的模型上大规模运行，并且能够涵盖难以或不宜在人类身上测试的高风险说服形式。在本研究中，我们将九种已发表的自动化方法调整至统一的实验设置，在相同的十五个大语言模型上运行，并探究这些方法给出的排名是否一致以及原因何在。我们发现这些方法仅在……（原文摘要在此处截断）

    arXiv:2610.10232v1 Announce Type: new  Abstract: Large language models (LLMs) have already been shown to match or exceed human experts in persuasion. While their persuasive capabilities hold promise for beneficial uses such as education and health communication, they can also be used to manipulate and misinform, making their evaluation a growing priority for developers and regulators. That evaluation, however, remains fragmented: studies differ in what they treat as persuasion, and broad claims often rest on narrow, situation-specific assessments. Automated methods, often modelled on human studies, offer a way to compare such assessments directly, as they can be run on the same models at scale and can include high-risk forms of persuasion that would be difficult or unethical to test on people. In this study, we adapt nine published automated methods to a shared setup, run them on the same fifteen LLMs, and ask whether their rankings agree and why. We find that the methods agree only we
    
[^19]: 从提示到树：面向少样本表格分类的高效LLM引导树生成方法

    From Prompts to Trees: Effective LLM-Guided Tree Generation for Few-Shot Tabular Classification

    [https://arxiv.org/abs/2610.10227](https://arxiv.org/abs/2610.10227)

    本文提出一种三阶段的LLM引导框架，通过提示LLM先生成规则再将其组织成决策树，在少样本表格分类任务中以显著更低的提示开销实现了更优的准确性和可解释性。

    

    尽管大语言模型（LLMs）拥有丰富的世界知识和令人印象深刻的泛化能力，但其直接应用于表格数据分类受到高推理成本和有限可解释性的阻碍。相比之下，决策树快速且透明，但在低数据量情况下往往表现不佳。在本工作中，我们提出了一个新颖的框架，通过在少样本学习设置下将LLM知识蒸馏到可解释的决策树中，从而弥合这两种范式。我们没有直接提示LLM生成完整的树（这通常不稳定且低效），而是开发了一种三阶段范式，提示LLM生成规则并将这些规则组织成树。在多个真实世界表格数据集上的实验表明，与现有基线相比，我们的方法以显著更低的提示开销实现了更优的准确性和可解释性。

    arXiv:2610.10227v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) possess rich world knowledge and impressive generalization capabilities, their direct application to tabular data classification is hindered by high inference costs and limited interpretability. In contrast, decision trees are fast and transparent but often underperform in low-data regimes. In this work, we propose a novel framework that bridges these paradigms by distilling LLM knowledge into interpretable decision trees under a few-shot learning setting. Instead of directly prompting the LLM to generate full trees, which is often unstable and inefficient, we develop a three-stage paradigm that prompts the LLM to generate rules and organize the rules into a tree. Experiments on multiple real-world tabular datasets demonstrate that our method achieves superior accuracy and interpretability with significantly lower prompting overhead compared to existing baselines.
    
[^20]: GAGR-Lab：评估空间-几何与解析函数的联合推理能力

    GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning

    [https://arxiv.org/abs/2610.10201](https://arxiv.org/abs/2610.10201)

    该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。

    

    空间-几何与解析函数的联合推理要求将感知到的空间配置转化为符号函数，且该函数执行出的曲线需满足几何约束。我们提出了GAGR-Lab，这是一个通过笛卡尔游戏场景、显式函数语义以及权威的Rust轨迹执行来衡量该能力的框架。它区分了空间感知、度量定位、几何关系、函数解释、函数构建和约束合成等六个维度。我们定义了四个可配置的场景难度预设和一个前瞻性的24格诊断设计，同时仅报告实际评估的子集。对单个托管模型（Llama 3.2 11B Vision Instruct）使用两个API凭据作为执行副本进行的有限试点实验，产生了72个平衡游戏、432次尝试、429个有效的提供者响应，且无一命中目标；探索性的普通函数提示变体同样未能命中，而结构化的……（摘要在此处被截断）

    arXiv:2610.10201v1 Announce Type: cross  Abstract: Joint spatial-geometric and analytic function reasoning requires translating a perceived spatial configuration into a symbolic function whose executed curve satisfies geometric constraints. We present GAGR-Lab, a framework for measuring this capability through Cartesian game scenes, explicit function semantics, and authoritative Rust trajectory execution. It distinguishes spatial perception, metric grounding, geometric relations, function interpretation, function construction, and constrained synthesis. We specify four configurable scene-difficulty presets and a prospective 24-cell diagnostic design, while reporting only the subset actually evaluated. A bounded pilot of one hosted model (Llama 3.2 11B Vision Instruct) using two API credentials as execution replicas yields 72 balanced games with 432 attempts, 429 valid provider responses, and no target hits; exploratory ordinary-function prompt variants also fail to hit, while the struc
    
[^21]: 超越结果奖励：面向搜索智能体的检索信用构建与分配

    Beyond Outcome Rewards: Constructing and Assigning Retrieval Credit for Search Agents

    [https://arxiv.org/abs/2610.10179](https://arxiv.org/abs/2610.10179)

    该论文系统研究了从中间检索步骤提取学习信号的奖励塑形与信用分配策略，并提出将中间信号与最终结果奖励相结合的训练框架，显著提升了搜索智能体在多跳问题上的学习效率与总体性能。

    

    搜索智能体使大型语言模型（LLMs）能够迭代地检索和使用信息，以解决复杂的多跳问题。可验证奖励的强化学习为对此类智能体进行后训练提供了一种有前景的方法，但其对稀疏的、基于结果的监督的依赖会使信用分配变得困难，并限制学习效率。在本文中，我们系统地研究了中间监督如何改进搜索智能体的强化学习。我们研究了一系列奖励塑形和信用分配策略，这些策略能够从中间检索步骤中提供学习信号。基于这些洞察，我们开发了一个训练框架，将中间信号与最终结果奖励相结合，以改进对多步搜索轨迹的学习。在相同训练条件下跨多个基准的实验表明，搜索智能体的总体性能得到提升，并显示……

    arXiv:2610.10179v1 Announce Type: new  Abstract: Search agents enable Large Language Models (LLMs) to iteratively retrieve and use information for complex multi-hop questions. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising approach for post-training such agents, but its reliance on sparse, outcome-based supervision can make credit assignment difficult and limit learning efficiency. In this paper, we systematically investigate how intermediate supervision can improve reinforcement learning for search agents. We study a range of reward-shaping and credit-assignment strategies that provide learning signals from intermediate retrieval steps. Building on these insights, we develop a training framework that combines intermediate signals with final outcome rewards to improve learning from multi-step search trajectories. Experiments across multiple benchmarks under matched training conditions demonstrate improvements in aggregate search-agent performance and show that
    
[^22]: HySPE：基于辛对偶剪切的位置编码

    HySPE: Positional Encoding via Symplectic Dual Shears

    [https://arxiv.org/abs/2610.10154](https://arxiv.org/abs/2610.10154)

    HySPE 通过 Sp(2,ℝ) 双曲分支上对偶剪切的阻尼组合构建位置编码，并结合特征基对角化与分块坐标重定基，在实现长度无关数值稳定性的同时，于 16 倍零样本长度外推下保持恒定困惑度，且前向延迟与 RoPE 相当。

    

    我们提出了双曲辛位置编码，将位置注意力建立在非紧辛变换的基础之上。经典的旋转位置编码通过旋转对 Sp(2,ℝ) 的紧致椭圆分支进行参数化，而 HySPE 则通过对偶剪切的阻尼对称组合实现了其双曲分支，从而产生一种每个通道对具有两个谱衰减率的保形辛收缩。为了消除朴素绝对因子分解所固有的指数级表示漂移，我们在该算子的不变特征基下对其进行对角化，并引入了带有自适应中心化执行的分块坐标重定基。这保证了与序列长度无关的数值界，同时其前向延迟与带缓存的 RoPE 相当（在 RTX 4090 上为 7.21 毫秒）。在 TinyShakespeare 数据集上，HySPE-UltraLong 在最高 16 倍零样本外推（L=4096）时仍保持 4.810 的恒定困惑度，而 RoPE 则出现性能退化。

    arXiv:2610.10154v1 Announce Type: new  Abstract: We introduce Hyperbolic Symplectic Positional Encoding (HySPE), grounding positional attention in non-compact symplectic transformations. While canonical Rotary Position Embedding (RoPE) parameterizes the compact, elliptic branch of $\Sp(2,\R)$ via rotations, HySPE operationalizes its hyperbolic branch via a damped symmetric composition of dual shears, yielding a conformally symplectic contraction with two spectral decay rates per channel pair. To eliminate the exponential representation drift inherent to naive absolute factorizations, we diagonalize the operator in its invariant eigenbasis and introduce blockwise coordinate rebasing with adaptive centered execution. This guarantees length-independent numerical bounds while matching cached RoPE forward latency (7.21\,ms on an RTX 4090). On TinyShakespeare, HySPE-UltraLong maintains an invariant perplexity of 4.810 up to $16\times$ zero-shot extrapolation ($L=4096$), whereas RoPE degrades
    
[^23]: InterView-C：一个VR化身媒介调查访谈的同步多模态语料库

    InterView-C: A Synchronized Multimodal Corpus of VR Avatar-Mediated Survey Interviews

    [https://arxiv.org/abs/2610.10145](https://arxiv.org/abs/2610.10145)

    InterView-C是一个包含27场VR化身调查访谈的德语多模态语料库，提供了与注视、头部和身体运动、面部行为、手部追踪等同步行为数据对齐的高质量人工转写文本和语言标注，为多模态口头交互与基于文本的NLP方法之间搭建了可靠桥梁。

    

    我们推出了InterView-C，这是一个德语多模态语料库，包含27场完全在虚拟现实中进行的调查访谈，对话双方均以化身形式呈现。该语料库将口头交互与同步的行为数据对齐，包括注视、头部和身体运动、面部行为、手部和手指追踪。其参考转写文本和语言标注为多模态口头交互与主要基于文本的NLP方法之间提供了可靠的接口。这一接口之所以重要，是因为自动语音转写可能扭曲与语言学相关的信息，而基于现有资源训练的下游模型在应用于转写的口头数据时还可能面临迁移挑战。因此，InterView-C为全部54个录音提供了带词级时间戳且经人工后编辑的逐字转写文本、访谈条目时间、问卷回答，以及针对1,422个句子的否定线索和范围标注。

    arXiv:2610.10145v1 Announce Type: new  Abstract: We present InterView-C, a German multimodal corpus of 27 survey interviews conducted entirely in virtual reality, with both interlocutors represented by avatars. The corpus aligns spoken interaction with synchronized behavioral data, including gaze, head and body movement, facial behavior, hand and finger tracking. Its reference transcripts and linguistic annotations provide a reliable interface between this multimodal spoken interaction and predominantly text-based NLP methods. This interface is important because automatically transcribing speech can distort linguistically relevant information, while downstream models trained on existing resources may additionally face transfer challenges when applied to transcribed spoken data. InterView-C therefore provides word-timed and manually post-edited verbatim transcripts for all 54 recordings, interview-item timings, questionnaire responses and negation cue and scope annotations for 1,422 sen
    
[^24]: LLM4Impact：整合异构信息用于科学影响力预测

    LLM4Impact: Integrating Heterogeneous Information for Scientific Impact Prediction

    [https://arxiv.org/abs/2610.10138](https://arxiv.org/abs/2610.10138)

    提出LLM4Impact方法，通过将语义、图、大语言模型和时间等多种异构信息进行表示、整合与校准，并利用上下文感知门控机制自适应加权不同证据，从而实现对新发表论文未来科学影响力的准确预测。

    

    预测一篇新发表论文的未来影响力具有挑战性，因为这必须从发表时可获得的异构证据中进行推断。现有方法通常依赖单一信息源，或者在组合多个信息源时未考虑它们各自不同的预测作用。在本文中，我们提出了LLM4Impact，这是一种用于科学影响力预测的证据感知方法，它能够学习表示、整合和校准异构信息。LLM4Impact结合了语义、图、大语言模型（LLM）和时间等多种表示，并通过连续前缀标记将图信息注入冻结的LLM中。一种上下文感知的门控机制可以自适应地对不同证据进行加权，同时一个独立的校准模块则考虑了引用规模在领域和时间上的差异变化。我们进一步构建了一个包含200万篇论文的大规模基准数据集，其中包括防泄漏的时间点异构自我图、时间分割（摘要在此处似乎不完整）

    arXiv:2610.10138v1 Announce Type: new  Abstract: Predicting the future impact of a newly published paper is challenging because it must be inferred from heterogeneous evidence available at publication time. Existing approaches often rely on a single source of information or combine multiple sources without accounting for their different predictive roles. In this paper, we present LLM4Impact, an evidence-aware method for scientific impact prediction that learns to represent, integrate, and calibrate heterogeneous information. LLM4Impact combines semantic, graph, LLM, and temporal representations, and injects graph information into a frozen LLM through continuous prefix tokens. A context aware gating mechanism adaptively weights different evidence, while a separate calibration module accounts for domain and temporal variation in citation scales. We further construct a large-scale benchmark dataset with 2 million papers, leakage-safe point-in-time heterogeneous ego graphs, temporal splits
    
[^25]: YANchor-4B：以 O(N) 时间复杂度和 O(1) 内存实现高效的长程推理

    YANchor-4B: Effective Long-Horizon Reasoning in O(N) Time with O(1) Memory

    [https://arxiv.org/abs/2610.10118](https://arxiv.org/abs/2610.10118)

    YANchor-4B 通过将关键记忆保存为可检索的锚点，以 O(N) 时间和 O(1) 内存的成本实现了高效的长程推理，在数学基准上大幅超越同类模型并具有数倍于 Transformer 的生成吞吐量。

    

    长程推理需要在可管理的生成成本下访问早期信息。全历史注意力机制会带来不断增长的存储与计算开销，而循环压缩则可能丢失精确细节。因此，我们提出了 YANchor-4B，这是一种通用循环模型，它将关键记忆保存为“锚点”，以便在后续推理中进行检索。除了 O(N) 时间生成和 O(1) 内存之外，YANchor 还通过其多维记忆机制实现了高效的长程推理。例如，在具有挑战性的数学问题上，它在 AIME 2024–2026 上取得了 82.93% 的平均 pass@1，在 HMMT 上取得了 63.64%，大幅超越线性时间、常数状态的同类模型，包括规模更大的模型。在 H100 上，其批量长文本生成吞吐量比 Transformer 和混合基线高出数倍。此外，跨数十个基准的评估证明了 YANchor 在通用能力方面的优越性。

    arXiv:2610.10118v1 Announce Type: cross  Abstract: Long-horizon reasoning demands access to earlier information at a manageable generation cost. Full-history attention incurs growing storage and computation, while recurrent compression can lose precise details. Therefore, we present YANchor-4B, a general-purpose recurrent model that preserves crucial memory as ANchors for retrieval during subsequent reasoning. Beyond $O(N)$-time generation and $O(1)$ memory, YANchor enables effective long-horizon reasoning through its multidimensional memory mechanism. For example, on challenging math problems, it achieves 82.93% mean pass@1 on AIME 2024--2026 and 63.64% on HMMT, substantially outperforming linear-time, constant-state counterparts, including larger models. It also delivers several-fold higher batched long-generation throughput than Transformer and hybrid baselines on H100. Furthermore, evaluations across dozens of benchmarks demonstrate YANchor's superiority in general-purpose capabili
    
[^26]: 长上下文混合模型的机制 第一部分1.1：从混合注意力到混合位置

    Mechanics of Long-Context Hybrid Models Part 1.1: From Hybrid Attention to Hybrid Position

    [https://arxiv.org/abs/2610.10114](https://arxiv.org/abs/2610.10114)

    本文提出长上下文混合模型的机制分析框架，揭示了“跷跷板效应”——线性注意力混合模型更受益于长上下文持续预训练，而滑动窗口注意力混合模型在长度外推上表现更好，并将其归因于不同注意力机制所诱导的位置归纳偏置差异。

    

    arXiv:2610.10114v1 公告类型：新 摘要：大型语言模型（LLMs）的架构设计正在从传统的仅全注意力模型转向混合模型，混合模型通过结合不同的注意力模块来提高长上下文效率以及在长度外推和上下文扩展方面的性能。为了解释混合模型为何有效以及如何更好地设计它们，我们提出了长上下文混合模型的机制研究。作为本系列的第1.1部分，我们从全注意力与滑动窗口注意力（SWA）或线性注意力（LA）门控变体（以GLA和GDN为代表）的混合模型开始研究。我们首先在上下文扩展中观察到一个跷跷板效应：LA混合模型从长上下文持续预训练中获益更多，而SWA混合模型在长度外推下表现更好。我们将这种行为归因于这些注意力机制所诱导的位置归纳偏置的差异。我们发现SWA混合模型存在短上下文学习陷阱、短窗口W（摘要在此处被截断）

    arXiv:2610.10114v1 Announce Type: new  Abstract: The architectural design of Large Language Models (LLMs) is shifting from traditional full-attention-only models to hybrid models, which combine different attention modules to improve long-context efficiency and performance in length extrapolation and context extension. To explain why hybrid models work and how to design them better, we propose Mechanics of Long-Context Hybrid Models. As Part 1.1 of this series, we begin with hybrids of full attention and either sliding-window attention (SWA) or gated variants of linear attention (LA), represented by GLA and GDN. We first observe a Seesaw Effect in Context Extension: LA hybrids benefit more from long-context continual pretraining, whereas SWA hybrids perform better under length extrapolation. We attribute this behavior to differences in the positional inductive biases induced by these attention mechanisms. We find that SWA hybrids suffer from a Short-Context Learning Trap, Short-Window W
    
[^27]: 我宁愿退出NLP也不愿再读这样的论文：NLP论文中“对立表述”（rather than）句式的兴起

    I would rather quit NLP than read another paper like this: The rise of antithesis in NLP papers

    [https://arxiv.org/abs/2610.10092](https://arxiv.org/abs/2610.10092)

    本文通过对比2019年ACL论文、2026年arXiv论文与GPT生成的论文，发现LLM显著加剧了NLP论文中“rather than”式对立表述的滥用（使用率增至七倍），其中约十分之一的此类用法会惹恼审稿人，凸显了AI辅助写作对学术文风的负面影响。

    

    无论好坏，大语言模型（LLM）如今已被广泛用于科学写作（本文也不例外，作者在部分章节的写作中使用AI辅助，详见致谢）。许多人注意到，近期的模型会在论文中填入不必要的对立表述，反复声明这项工作“不做什么”，这种方式既无助于表达的精确性与质量，反而会惹恼审稿人而非给他们留下深刻印象。本文研究了“rather than”这一构式在2019年ACL论文、2026年ACL风格的arXiv论文，以及由GPT模型根据相同标题和摘要撰写的论文中的使用情况。研究发现：2026年论文中该构式的使用率是2019年的七倍，而在GPT撰写的论文中比率还要更高。两位对论文来源不知情的标注者发现，2019年的用法几乎 none 会令人恼火，而2026年的用法中约有十分之一令人恼火；两人很少就哪些用法恼火达成一致，但2026年约有一半的论文中至少包含一处让各自感到恼火的用法。令人恼火的用法通常会陈述该研究所拒绝、不做的事情……

    arXiv:2610.10092v1 Announce Type: new  Abstract: For better or worse, LLMs are by now used routinely for scientific writing.\footnote{This paper is no exception; we did use AI to assist with writing some of the sections (see Acknowledgments).} Many have noticed that recent models fill papers with unnecessary antithesis, stating over and over what the work does not do, in ways that do not contribute to its precision or quality of expression and annoy reviewers \emph{rather than impressing them}. We study the construction \emph{rather than} in ACL papers from 2019, ACL-style arXiv papers from 2026, and papers written by GPT models from the same titles and abstracts. Its rate in 2026 is seven times the 2019 rate, and higher still in the GPT papers. Two annotators, blind to the source, find almost no 2019 use \emph{annoying} and about one in ten 2026 uses; they seldom agree on which, yet about half of 2026 papers contain a use that annoys each of them. \emph{Annoying} uses present the reje
    
[^28]: ExperienceIndex：基于工件（Artifact）的记忆

    ExperienceIndex: Artifact-Grounded Memory

    [https://arxiv.org/abs/2610.10091](https://arxiv.org/abs/2610.10091)

    提出了 ExperienceIndex，一种让 AI 智能体基于先前推理轨迹捕获并复用工件特定经验知识的新型记忆层，可提升知识密集型任务的答案质量并降低在线成本。

    

    知识密集型任务需要通过推理共享的工件语料库（例如法院判例或科学文献）来回答大量问题。当人类与这些语料库交互时，会自然地积累关于工件的经验知识，从而能够快速识别出每个新任务的完整相关工件集合。然而，现有的人工智能智能体缺乏合适的记忆解决方案来构建或复用这种以工件为基础的经验，导致答案质量较低且在线成本较高。现有的记忆解决方案虽然能从先前的任务求解轨迹中提取和复用信息，但它们主要关注用户偏好、事实属性或抽象推理模式，而非持久的、特定于工件的知识。我们提出了 ExperienceIndex，这是一种面向 AI 智能体的新型经验层，它基于先前的推理轨迹来捕获和复用关于工件的知识。ExperienceIndex 存储两种互补的……

    arXiv:2610.10091v1 Announce Type: new  Abstract: Knowledge-intensive tasks require answering many questions by reasoning about a shared corpus of artifacts (e.g., court cases, or scientific literature). As humans interact with these corpora, they naturally accumulate experiential knowledge about artifacts, enabling them to quickly identify the complete set of relevant artifacts for each new task. However, existing AI agents lack appropriate memory solutions to build or reuse such artifact-grounded experience, leading to lower answer quality and higher online cost. Existing memory solutions extract and reuse information from prior task-solving traces, but they primarily focus on user preferences, factual attributes, or abstract reasoning patterns rather than persistent artifact-specific knowledge. We introduce ExperienceIndex, a novel experience layer for AI agents that captures and reuses knowledge about artifacts based on prior reasoning traces. ExperienceIndex stores two complementar
    
[^29]: SkillSandbox：通过动态场景合成进行技能验证

    SkillSandbox: Skill Verification via Dynamic Scenario Synthesis

    [https://arxiv.org/abs/2610.10088](https://arxiv.org/abs/2610.10088)

    提出了SkillSandbox框架，通过为每个技能动态合成相关且新颖的任务场景，比较智能体有无该技能时的执行表现，以验证自演化智能体所提炼技能的可复用性。

    

    自演化智能体将解决任务的经验提炼为技能以供未来复用，但这些技能可能包含错误的流程或不可迁移的知识。因此，验证每个技能的可复用性至关重要：即技能的指导能否在提炼它的原始经验之外依然有效。这种验证需要观察技能如何影响新任务中的执行，然而现有任务可能无法提供目标技能真正能够发挥作用的情境。为了构建这样的情境，我们提出了SkillSandbox，这是一个为每个技能动态合成相关且新颖的任务及其环境的框架。提议者指定需要保留的条件以及需要变化的源特定细节，构建者构建可执行的场景，验证者比较有无该技能时的执行表现。验证者通过评估可执行性、效用和效率来给出保留或拒绝的判定。

    arXiv:2610.10088v1 Announce Type: cross  Abstract: Self-evolving agents distill task-solving experience into skills for future reuse, but these skills can encode incorrect procedures or non-transferable knowledge. It is therefore critical to verify each skill's reusability: whether its guidance remains useful beyond the experience from which it was distilled. Such verification requires observing how a skill affects execution in new tasks, yet existing tasks may not expose the situations where the target skill can actually be exercised. To construct such situations, we propose SkillSandbox, a framework that dynamically synthesizes a task and its environment for each skill that are skill-relevant yet novel. A Proposer specifies the conditions to preserve and the source-specific details to vary, a Builder constructs an executable scenario, and a Verifier compares executions with and without the skill. The Verifier assesses executability, utility, and efficiency to assign a Keep or Reject 
    
[^30]: 缓存内在编码器：跨大语言模型查询的紧凑、可复用记忆

    Cache the Encoder Within:Compact, Reusable Memory across LLM Queries

    [https://arxiv.org/abs/2610.10058](https://arxiv.org/abs/2610.10058)

    EncBank将预训练LLM的底层复用为可共享的文档编码器，以4位精度紧凑缓存中间状态，在几乎不损失精度的情况下将持久GPU存储降至原生精度的28.1%，并带来1.40倍的预填充加速。

    

    对共享文档的重复查询会导致冗余编码，而缓存模型状态又会带来持续的存储开销。本文在CoMem的中间状态接口基础上提出EncBank，将预训练大语言模型的底层视为可复用的文档编码器，并紧凑地存储其输出，供经过适配的上层读取器使用。每个骨干模型内共享一个自蒸馏后缀适配器，可跨不同存储精度使用，无需针对量化进行专门重训练。在三个不同规模、涵盖全注意力与混合架构的Qwen骨干模型上进行的五个基准套件测试中，4位存储使每个基准的聚合分数与原生精度EncBank的差距控制在1分以内。在固定的Qwen3-8B工作负载下，其仅占用原生精度持久GPU存储的28.1%。在单独的原生精度对照实验中，相比相同证据、相同适配器的文本重放方式，选择性打包预填充实现了1.40倍加速，代价为RULER准确率下降3.12分。

    arXiv:2610.10058v1 Announce Type: new  Abstract: Repeated queries over shared documents incur redundant encoding, while caching model states introduces persistent storage costs. Building on CoMem's intermediate-state interface, EncBank treats a pretrained LLM's lower layers as a reusable document encoder and compactly stores their outputs for an adapted upper-layer reader. A self-distilled suffix adapter is shared across storage precisions within each backbone, without quantization-specific retraining. Across five benchmark suites on three Qwen backbones spanning different sizes and full-attention and hybrid architectures, 4-bit storage keeps each reported benchmark aggregate within one score point of native-precision EncBank. In a fixed Qwen3-8B workload, it retains 28.1% of the native-precision persistent GPU store. Separate native-precision controls yield a 1.40x selected-pack prefill speedup over same-evidence, same-adapter text replay, at a 3.12-point RULER accuracy cost. A native
    
[^31]: 通往相同答案的漫长道路：大语言模型在推理预算不断提升下的认知偏差

    The Long Road to the Same Answer: Cognitive Bias Under Escalating Reasoning Budgets in Large Language Models

    [https://arxiv.org/abs/2610.10049](https://arxiv.org/abs/2610.10049)

    通过对四个模型家族、12,350次调用的大规模剂量-反应实验，研究发现增加推理预算（更多思考token）并不能减少大语言模型的六种经典认知偏差，推理模型甚至并不比非推理模型更少偏差。

    

    摘要：推理模型在推理时分配额外的计算资源，并将其答案呈现为深思熟虑的产物。如果这种深思熟虑按照人类认知双加工理论所暗示的方式运作，那么更长的思考应该会削弱快速、直觉判断所产生的经典决策偏差。我们使用一个成熟基准中涵盖六种偏差（锚定效应、框架效应、损失厌恶、承诺升级、可得性启发和确认偏差）的30个情境故事，在四个模型家族中开展剂量-反应研究，将每个推理模型与匹配的非推理“兄弟”模型配对，并设置0、1,024、4,096和8,192个token的思考上限，共进行了12,350次API调用。由于请求的思考上限并不等同于实际实现的深思，我们使用每次调用实际消耗的推理token数量作为剂量。首先，推理模型并不比其“兄弟”模型偏差更小；每个家族的点估计反而倾向于相反的方向，但在条目层面……（原文摘要在此处截断）

    arXiv:2610.10049v1 Announce Type: new  Abstract: Reasoning models allocate extra computation at inference time and present their answers as the product of deliberate thought. If this deliberation works the way dual-process accounts of human cognition suggest, longer thinking should weaken the classic decision biases that fast, intuitive judgment produces. Using 30 vignettes covering six biases (anchoring, framing, loss aversion, escalation of commitment, availability, confirmation) from an established benchmark, we run a dose-response study across four model families, pairing each reasoning model with a matched non-reasoning sibling and requesting thinking ceilings of 0, 1,024, 4,096, and 8,192 tokens, for 12,350 API calls. Because a requested ceiling is not the same as realized deliberation, we use the reasoning tokens each call consumed as the dose. First, reasoning models are not less biased than their siblings; the point estimate leans the other way in every family, but the item-le
    
[^32]: 通过大语言模型路由元数据泄露敏感话题：测量与缓解

    Sensitive-Topic Leakage Through LLM Routing Metadata: Measurement and Mitigation

    [https://arxiv.org/abs/2610.09981](https://arxiv.org/abs/2610.09981)

    本文揭示了LLM路由器的模型选择元数据即使在关闭内容日志的情况下也会泄露用户请求的敏感话题（如自残、医疗、性内容），并通过对170万条真实请求的预注册测量和后处理防御方法提出了缓解方案。

    

    LLM路由器根据请求内容为每个请求选择便宜或昂贵的模型，而许多网关和部分云平台即使关闭了内容日志记录，也能记录这一选择。我们在超越token数量的层面测量了这一隐私泄露通道，并考虑了噪声标签和重复提示的影响。我们在170万条真实请求（WildChat-1M、LMSYS-Chat-1M）上开展了预注册研究，使用两个成本/质量路由器和一个领域路由器，调查了十一个系统的日志记录情况，并测试了后处理防御方法。在长度匹配的条件下，偏移的方向取决于话题类别和路由器。对于运行点为50%的RouteLLM，骚扰与自残类请求到达强模型的频率比同类请求低19个百分点；医疗类请求（探索性结果：LLM标签未通过其质量门控）在独立提示上低31个百分点（两者均为事后分析）；性相关请求则高10个百分点（次要发现）；另一路由器的四个类别均为负向偏移。二十个RouteLLM决定……（摘要原文在此处截断）

    arXiv:2610.09981v1 Announce Type: cross  Abstract: LLM routers pick a cheap or expensive model per request by its content, and many gateways and some cloud platforms can log that choice with content logging off. We measure this privacy channel beyond token counts, accounting for noisy labels and repeated prompts. We run pre-registered studies on 1.7 million real requests (WildChat-1M, LMSYS-Chat-1M) with two cost/quality routers and a domain router, survey eleven systems' logging, and test post-processing defenses. At matched length, the shift's direction depends on category and router. For RouteLLM at the 50% operating point, harassment and self-harm requests reach the strong model 19 points less often than comparable ones on prompts unseen in exploration, medical requests (exploratory: LLM labels failed their gate) 31 points less often on distinct prompts (both post hoc), and sexual requests 10 points more often (secondary); the other router's four are negative. Twenty RouteLLM decis
    
[^33]: EASE：用于规避AI生成文本检测器的熵自适应分布塑形方法

    EASE: Entropy-Adaptive Distribution Shaping for Evading AI-generated Text Detectors

    [https://arxiv.org/abs/2610.09976](https://arxiv.org/abs/2610.09976)

    EASE提出了一种无需训练、与检测器无关的规避框架，通过利用源LLM的预测熵来自适应调整logit扰动和采样温度，从而有效逃避AI生成文本检测器的识别，且几乎不损失文本质量也不增加推理开销。

    

    AI生成文本（AIGT）检测可能对源大语言模型（LLM）的解码选择十分敏感。我们观察到，对下一词元的logits进行扰动或调整采样温度可以降低检测性能，这清楚地表明了检测器对解码时分布变化的脆弱性。基于这一观察，我们提出了EASE（Entropy-Adaptive Distribution Shaping for Evasion，熵自适应分布塑形规避），这是一个无需训练、与检测器无关的AIGT检测规避框架。EASE直接从源LLM的下一词元分布中计算预测熵，并利用该熵值对logit扰动和采样温度进行自适应调整，无需检测器反馈或模型微调。在三个源LLM和多个检测器上的实验表明，该方法能够持续降低检测性能，同时文本质量下降和推理开销几乎可以忽略不计。

    arXiv:2610.09976v1 Announce Type: new  Abstract: AI-generated text (AIGT) detection can be sensitive to the decoding choices of the source large language model (LLM). We observe that perturbing next-token logits or adjusting sampling temperature can reduce detection performance, providing a clear signal of detector vulnerability to decoding-time distribution changes. Building on this observation, we propose EASE (Entropy-Adaptive Distribution Shaping for Evasion), a training-free and detector-agnostic framework for evading AIGT detectors. EASE computes predictive entropy directly from the source LLM's next-token distribution and uses it to adapt both logit perturbation and sampling temperature, without detector feedback or model fine-tuning. Experiments across three source LLMs and multiple detectors demonstrate consistent reductions in detection performance, with negligible degradation in text quality and negligible inference overhead.
    
[^34]: Itgan在NADI 2026共享任务：面向鲁棒、混合方言和语码转换阿拉伯语ASR的参数高效Whisper适配

    Itgan at NADI 2026 shared task: Parameter-Efficient Whisper Adaptation for Robust, Mixed-Dialect and Code-Switched Arabic ASR

    [https://arxiv.org/abs/2610.09934](https://arxiv.org/abs/2610.09934)

    该论文提出在消费级GPU上用LoRA参数高效适配Whisper的统一方案，在NADI 2026的三个阿拉伯语ASR子任务（国家级、混合方言和突尼斯语码转换）中均取得有竞争力的结果，其中突尼斯语码转换任务获得第二名并取得领先提交中最低的字符错误率。

    

    我们介绍了Itgan参加NADI 2026三个ASR子任务的系统，分别是鲁棒的国家级ASR（1.1）、混合方言ASR（1.2）和突尼斯语码转换ASR（1.3）。这三个系统共享一个技术方案，即在消费级GPU上使用LoRA对Whisper进行参数高效适配，而每个系统的成绩都由一项不同的附加改进所驱动。在任务1.1中，由于测试时提供了方言标签，从池化适配器继续训练的按方言专家模型带来了最大收益，提交系统的国家级平均WER达到57.1%。一项评估后的实验表明，在冻结的编码器特征上使用线性探针可以在没有标签的情况下对话语进行路由，恢复了Oracle路由效果的44%。在任务1.2中，基础模型的选择比适配器容量更为关键，且系统组合只有在加入一个去相关成员后才有所帮助，最终达到46.7%的WER。在任务1.3中，我们的系统以14.49%的WER获得第二名，并在领先提交中取得了最低的CER（5.38%），其中最后0.60个WER点的提升无需任何额外训练。

    arXiv:2610.09934v1 Announce Type: new  Abstract: We describe the Itgan systems for the three ASR subtasks of NADI 2026, namely robust country-level ASR (1.1), mixed-dialect ASR (1.2), and Tunisian code-switched ASR (1.3). All three share one recipe, Whisper adapted with LoRA on consumer GPUs, and each was carried by a different addition to it. On 1.1, where the dialect label is given at test time, per-dialect specialists continued from a pooled adapter gave the largest gain, and the submitted system reached 57.1% country-average WER. A post-evaluation linear probe on frozen encoder features routes utterances without the label and recovers 44% of what oracle routing gives. On 1.2 the choice of base model mattered more than adapter capacity, and system combination helped only once we added a decorrelated member, reaching 46.7% WER. On 1.3 our system placed second at 14.49% WER with the lowest CER among the leading submissions, 5.38%. Its last 0.60 WER points came without further training
    
[^35]: 反演多向量视觉文档索引

    Inverting Multi-Vector Visual Document Indices

    [https://arxiv.org/abs/2610.09920](https://arxiv.org/abs/2610.09920)

    该论文揭示了多向量视觉文档索引的严重隐私风险：攻击者无需接触原始文档，仅凭存储的patch向量即可重建出页面图像，恢复近半数的文字和敏感信息，并可通过反演页面以98.4%的准确率定位源页面，作者同时评估了token池化等低成本防护手段的有效性。

    

    主流的多向量视觉文档检索器将每一页文档存储为大约一千个patch向量，这些向量通常保存在由第三方运营的向量数据库中。由于人们通常认为无法从向量中读出页面内容，这种索引往往被视为比原始页面本身敏感度更低。然而，由于索引按照光栅顺序为每个patch保留一个向量，且每个向量由预先训练用于阅读文档的视觉-语言模型计算得出，我们假设无论是谁运行或入侵了该存储库，都可以仅凭索引重建出整个页面。我们将这种反演问题构建为条件文档图像生成任务，并从向量中推断攻击所需的信息：编码器类型、页面形状，以及在向量顺序被打乱时的原始排列顺序。在ViDoRe v3基准测试中，从原始索引反演出的页面恢复了47%的单词和45%的敏感标记。当将这些反演出的页面作为查询对存储的索引进行检索时，它们在98.4%的情况下将源页面排在第一位。我们测试了两种低成本的防护措施，包括token池化……

    arXiv:2610.09920v1 Announce Type: cross  Abstract: Prevailing multi-vector visual document retrievers store each page as about a thousand patch vectors, often in vector databases run by a third party. Since no one can read a page from its vectors, this index is easily treated as less sensitive than the page. However, because the index keeps one vector per patch in raster order, and each vector is computed by a vision-language model pre-trained to read documents, we hypothesize that whoever runs or breaches the store can reproduce a page from its index alone. We frame inversion as conditional document image generation and infer from the vectors what the attack needs: the encoder, the page shape and, for shuffled vectors, their order. On the ViDoRe v3 benchmark, pages inverted from raw indices recover 47% of the words and 45% of the sensitive tokens. Used as queries against the stored indices, they rank their source page first 98.4% of the time. We test two cheap protections, token pooli
    
[^36]: 基于NeMo-Guardrails代理的SIEM/XDR受限动作AI修复方法

    Constrained-Action AI Remediation for SIEM/XDR via a NeMo-Guardrails Proxy

    [https://arxiv.org/abs/2610.09906](https://arxiv.org/abs/2610.09906)

    提出了一种包含SIEM/XDR控制平面与NeMo-Guardrails代理的双层受限动作架构，将LLM的修复建议限制在封闭的意图词汇表中，从而防止对抗性告警通过LLM推理路径诱导SOC执行危险操作。

    

    面向信息技术和运营技术的安全运营中心（SOC）都面临同一个事件响应难题：大量关联告警蜂拥而至，而分析师人手不足。大语言模型（LLM）越来越多地被提议作为推理引擎，用于告警分诊，并在自主部署中发出封锁IP、终止进程或隔离生产主机上文件等命令。这种耦合引入了一种新的风险：单个对抗性告警可以通过LLM的推理变成一条远程代码路径，诱导其推荐SOC随后执行的危险动作。我们提出了一种由两个协同层组成的受限动作架构：（i）一个SIEM/XDR控制平面，将修复措施锚定于关联的主机事件，并将LLM的输出限制在封闭的意图词汇表中，其模板化命令由轻量级终端代理执行，并由参数验证器提供后备保障；（ii）一个包裹SOC分析师LLM的NeMo-Guardrails代理（摘要在此处截断）。

    arXiv:2610.09906v1 Announce Type: cross  Abstract: Security Operations Centers (SOCs) for information technology and operational technology share one incident-response problem: a flood of correlated alerts and too few analysts. Large Language Models (LLMs) are increasingly proposed as reasoning engines that triage alerts and, in autonomous deployments, issue commands that block IPs, kill processes, or quarantine files on production hosts. This coupling introduces a new risk: a single adversarial alert can become a remote code path through the LLM's reasoning, leading it to recommend an action the SOC then executes. We present a constrained-action architecture with two coordinated layers: (i) a SIEM/XDR control plane that grounds remediation in correlated host events and confines the LLM's output to a closed intent vocabulary whose templated commands are executed by thin endpoint agents, backstopped by an argument validator; and (ii) a NeMo-Guardrails proxy that wraps the SOC-analyst LL
    
[^37]: LiveMACE：演化市场中LLM智能体能力的过程感知评估

    LiveMACE: Process-Aware Evaluation of LLM Agent Capabilities in Evolving Markets

    [https://arxiv.org/abs/2610.09872](https://arxiv.org/abs/2610.09872)

    该论文提出LiveMACEBench——一个以实时金融市场为测试平台的过程感知基准，通过对五个前沿LLM智能体进行30天连续实时评估，揭示了显著的结果-能力差距，表明仅凭收益等最终结果无法真实反映智能体的工具使用、记忆、规则遵循与协作等底层能力。

    

    仅凭结果来评估智能体可能会掩盖产生这些结果的能力。这一问题在持续演化的环境中尤为突出，因为结果反映的是智能体行为与不断变化的外部条件之间的闭环交互。我们提出了LiveMACEBench，这是一个过程感知的基准测试，它将实时金融市场作为持续运行的LLM智能体的天然演化测试平台。五个前沿LLM在匹配的工具使用、持久记忆、规则遵循和多智能体协作配置下沿连续轨迹运行。我们通过已实现的结果以及从完整决策轨迹中提取的机制特定诊断指标对它们进行评估。在30天的实时评估中，我们发现存在显著的结果-能力差距：已实现收益往往与特定能力的测量结果出现背离，相似的结果可能源于截然不同的机制使用模式。轨迹级别的诊断进一步揭示了……

    arXiv:2610.09872v1 Announce Type: cross  Abstract: Evaluating agents by outcomes alone can obscure the capabilities that produce them. This problem is especially pronounced in evolving environments, where outcomes reflect a closed-loop interaction between agent behavior and changing external conditions. We introduce LiveMACEBench, a process-aware benchmark that uses live financial markets as a naturally evolving testbed for persistent LLM agents. Five frontier LLMs operate along continuous trajectories under matched Tool Use, Persistent Memory, Rule Following, and Multi-Agent Collaboration configurations. We evaluate them through both realized outcomes and mechanism-specific diagnostics derived from complete decision traces. Across 30 days of live evaluation, we find a pronounced outcome-capability gap: realized returns often diverge from capability-specific measurements, and similar outcomes can arise from markedly different patterns of mechanism use. Trace-level diagnostics further e
    
[^38]: 从任务结果训练大语言模型智能体的顾问

    Training Advisors for LLM Agents from Task Outcomes

    [https://arxiv.org/abs/2610.09858](https://arxiv.org/abs/2610.09858)

    提出Caddie方法，通过强化学习仅以智能体最终任务成功与否作为训练信号来训练批评者提供自然语言建议，且训练后的批评者能泛化到不同规模和架构的多个基础模型并显著提升任务成功率。

    

    大语言模型智能体通过交替进行推理和工具调用，并结合环境反馈的观察来完成多步骤任务。先前的研究表明，自然语言反馈可以帮助这些智能体在任务执行过程中修正其决策。我们提出了Caddie，一种训练批评者在智能体执行任务过程中提供自然语言分析与建议的方法。与依赖步骤级标签或参考批评的现有方法不同，Caddie从智能体在接收批评者反馈后是否最终取得成功中学习。我们在保持基础模型冻结的情况下，通过强化学习优化批评者。该批评者仅基于单一基础模型在多跳问答任务上训练，而我们的Qwen3-4B批评者在四个不同规模和架构的基础模型上均提升了任务成功率，其中包括三个在批评者训练阶段未曾使用过的模型。在MuSiQue基准上，训练后的批评者使Qwen3-4B的成功率提升了超过……（原文此处截断）

    arXiv:2610.09858v1 Announce Type: new  Abstract: Large language model agents tackle multi-step tasks by interleaving reasoning and tool calls with observations from the environment. Prior work has shown that natural-language feedback can help these agents revise their decisions during task execution. We introduce Caddie, a method for training critics to provide natural-language analysis and advice as agents work through a task. Unlike approaches that rely on step-level labels or reference critiques, Caddie learns from whether the agent ultimately succeeds after receiving the critic's feedback. We optimize the critic through reinforcement learning while keeping the base model frozen. Trained on multi-hop question answering with a single base model, our Qwen3-4B critic improves success rates across four base models of different scales and architectures, including three not used during critic training. On the MuSiQue benchmark, the trained critic improves Qwen3-4B's success rate by more t
    
[^39]: 震耳欲聋的沉默：灾难性遗忘潜藏于数据从未提及的词元的输出嵌入之中

    A Deafening Silence: Catastrophic Forgetting Lives in the Output Embeddings of Tokens the Data Never Speaks

    [https://arxiv.org/abs/2610.09835](https://arxiv.org/abs/2610.09835)

    该研究揭示了大语言模型持续学习中的灾难性遗忘选择性地集中在低频词元的输出嵌入层——由语料库词汇缺失导致 Adam 二阶矩归一化放大单侧梯度所致——并提出仅在该层提高 epsilon 的干预方法。

    

    大语言模型（LLM）的持续预训练与微调不可避免地会引发灾难性遗忘，通常通过重放原始数据来缓解，但原始数据往往难以获取。在这种无数据的场景下，我们分析了遗忘发生的位置及其原因。通过在五个设置（模型规模最大至1.4B）中进行系统性的参数冻结实验，我们发现遗忘选择性地集中在那些在新语料库中很少出现的词元的输出嵌入中，而模型主体中相同的 sqrt(v̂) 频带则保持惰性，新知识的学习发生在其他地方。这种定位由语料库的词汇缺失程度决定，而非训练模式，因此在固定基座模型内，仅凭词元计数即可在再训练前进行风险排序。从机制上讲，缺失的词元会收到持续的单侧 softmax 梯度，而 Adam 的二阶矩（sqrt(v̂)）归一化会将其放大为全尺寸的更新。因此，我们提出了一种干预方法：仅在输出嵌入层提高 Adam 的 epsilon（摘要在此处截断）。

    arXiv:2610.09835v1 Announce Type: new  Abstract: Continual pre-training and fine-tuning in Large Language Models (LLMs) inevitably induce catastrophic forgetting, typically mitigated by replay using often-inaccessible original data. In this data-free regime, we analyze where forgetting occurs and why. Systematic parameter freezing across five settings up to 1.4B reveals that forgetting concentrates selectively in the output embeddings of tokens rarely seen in the new corpus, whereas the same sqrt(v-hat) band of the body is inert and new learning resides elsewhere. This localization is governed by the vocabulary deficiency of the corpus rather than the training mode, allowing pre-retraining risk ranking from token counts alone within a fixed base model. Mechanistically, absent tokens receive persistent one-sided softmax gradients that Adam's second-moment (sqrt(v-hat)) normalization amplifies into full-sized updates. We therefore propose an intervention: raising Adam's epsilon exclusive
    
[^40]: MIRROR：LLM个性化中从模仿到内化的跨越

    MIRROR: From Imitation to Internalization in LLM Personalization

    [https://arxiv.org/abs/2610.09795](https://arxiv.org/abs/2610.09795)

    提出自蒸馏框架MIRROR，通过参考揭示的在策略自蒸馏和焦点插件MIRROR-F，将LLM个性化从模仿参考措辞升级为内化用户偏好，在提升内容质量的同时保留个人风格。

    

    对个性化大语言模型的需求正从风格模仿转向内容质量。我们研究了自蒸馏能否在现有微调范式中弥合这一差距。为解决这一局限，我们提出了MIRROR（通过内化参考揭示的在策略反思实现元个性化），这是一个新颖的自蒸馏框架，它将LLM个性化从模仿转向偏好内化。首先，我们用参考揭示的在策略自蒸馏取代参考令牌模仿，使模型在其自身生成轨迹上的下一词元分布与参考条件下自身的分布对齐，从而内化用户偏好而非复现参考措辞。其次，我们引入MIRROR-F，这是一个焦点插件，通过对信息性参考词元的选择性监督来增强在策略分布对齐，从而在保持风格特征的同时强化内容生成。

    arXiv:2610.09795v1 Announce Type: new  Abstract: The demand for personalized LLMs is shifting from style imitation toward content quality. We investigate whether self-distillation can bridge this gap in existing fine-tuning paradigm. To address this limitation, we introduce MIRROR(Meta- personalization by Internalizing Reference-Revealed On-policy Reflections), a novel self-distillation framework that shifts LLM personalization from imitation toward preference internalization. First, we replace reference-token imitation with reference-revealed on-policy self-distillation, aligning the model's next-token distributions along its own generation trajectories with those of its reference-conditioned self, thereby internalizing user preferences rather than reproducing reference wording.Second, we introduce MIRROR-F, a focal plug-in that augments on-policy distributional alignment with selective supervision over informative reference tokens, thereby strengthening content generation while prese
    
[^41]: 潜空间中的评判：基于语义保持压缩的高效生成式奖励建模

    Judging in Latent Space: Efficient Generative Reward Modeling via Semantics-Preserving Compression

    [https://arxiv.org/abs/2610.09788](https://arxiv.org/abs/2610.09788)

    LatentGRM通过语义分块、压缩与重构将评估过程编码为紧凑的连续潜在轨迹，无需逐token生成文本评估即可实现高效奖励建模，并在4B和8B规模上取得与显式SFT评判器相当的偏好判断准确率。

    

    奖励建模通常需要在多个评估准则上进行联合表示与推理，然而逐token地将这一过程用文字表述出来会带来高昂的推理成本。近期关于潜在推理的研究表明，连续状态可能以更紧凑的方式支持这类计算。我们提出了LatentGRM，一个建立在语义分块、压缩与重构之上的潜在评估框架。通过利用准则引导评估的结构来指导压缩，LatentGRM学习到紧凑的连续轨迹，无需生成文本评估即可支持自主的成对判断。一个独立的解释器可以从这些轨迹中重构出评估文本，从而以离线方式展现压缩后所保留的信息。在相同的训练数据和骨干网络条件下，LatentGRM在4B和8B规模上均取得了与显式监督微调（SFT）评判器相当的总体偏好判断准确率。

    arXiv:2610.09788v1 Announce Type: new  Abstract: Reward modeling often requires jointly representing and reasoning over multiple evaluation criteria, yet verbalizing this process token by token can incur substantial inference cost. Recent work on latent reasoning suggests that continuous states may support this computation more compactly. We introduce LatentGRM, a latent evaluation framework built on semantic chunking, compression, and reconstruction. By using the structure of rubric-guided evaluations to guide compression, LatentGRM learns compact continuous trajectories that support autonomous pairwise judgments without generating textual assessments. A separate interpreter reconstructs evaluation text from these trajectories, providing an offline view of the information retained under compression. Under matched training data and backbones, LatentGRM achieves competitive aggregate preference accuracy relative to explicit Supervised Fine-Tuning (SFT) judges at both 4B and 8B scales. A
    
[^42]: 解耦逻辑与人设：边缘大语言模型智能体对上下文污染的结构性免疫

    Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution

    [https://arxiv.org/abs/2610.09772](https://arxiv.org/abs/2610.09772)

    该论文提出AO-DA解耦架构，将边缘LLM智能体的逻辑推理与人设表达分离为同一INT4基础模型上两条可热插拔LoRA适配器的独立推理路径，使逻辑部分对上下文污染实现结构性免疫。

    

    运行在边缘设备上的小型语言模型智能体必须在同一个上下文窗口内同时维持人设并进行正确推理，而该窗口中充满了对话历史和人设指令。我们研究了当这些历史记录冗长、具有误导性且人设内容占比过高（即人设-逻辑干扰）时，此类智能体的逻辑推理部分会发生什么，并提出了一种解耦架构（AO-DA），在单个INT4量化的基础模型上，通过可热插拔的LoRA适配器，将逻辑推理（“做什么/What”）与人设表达（“怎么做/How”）分离为两条独立的推理路径。逻辑路径仅接收核心对话轮次，并输出可验证的结构化状态；人设路径则结合完整的对话历史，以符合角色设定的方式对该状态进行渲染。在同一基础模型上的消融实验中（在Apple M2笔记本电脑上进行，使用Llama-3.1-8B-Instruct和Gemma-3-4B-it，4比特量化；共480次运行，涵盖4个污染等级 × 3个实验组 × 2个任务 × 2个人设 × 5个随机种子），我们发现：（i）解耦后的逻辑路径对污染具有结构性不变性：其……（原文摘要在此处截断）

    arXiv:2610.09772v1 Announce Type: new  Abstract: Small language-model agents on edge devices must hold a persona and reason correctly at once, inside one context window that fills with conversational history and persona instructions. We study what happens to the logical part of such an agent when that history is long, misleading and persona-heavy (persona-logic interference), and present a Decoupling Architecture (AO-DA) that separates logical inference ("What") from persona expression ("How") into two inference paths on one INT4 base model with hot-swappable LoRA adapters. The logic path receives only the core turn and emits a verifiable structured state (Micro-State); the persona path renders it in character with the full history. In same-base-model ablations on an Apple M2 laptop (Llama-3.1-8B-Instruct and Gemma-3-4B-it, 4-bit; 480 runs over 4 pollution levels x 3 arms x 2 tasks x 2 personas x 5 seeds) we find: (i) the decoupled logic path is structurally invariant to pollution: its
    
[^43]: 从专家引导的证明搜索到自动化开放问题求解

    From Expert-Guided Proof Search to Automated Open-Problem Solving

    [https://arxiv.org/abs/2610.09769](https://arxiv.org/abs/2610.09769)

    研究者开发的多智能体开源系统Bolzano无需针对具体问题的人类指导，在约3,800个开放问题中自动解决了约200个，其中包括经原作者确认的STOC 2026论文中提出的4个问题。

    

    大型语言模型对数学研究的贡献日益增多，而数学研究的进展往往依赖于高效的证明搜索、渐进式的改进以及细致的验证。我们介绍了Bolzano，一个多智能体开源系统，它使用并行的证明器智能体配合验证器智能体协同工作，并维护人类可读的研究状态。在专家挑选的问题上进行的初期人工使用产生了8项成果，其证明均经领域专家核查。受这些案例研究的启发，我们在从四组论文中提取的约3,800个开放问题上运行了Bolzano，无需针对具体问题的人类指导，便解决了其中约200个开放问题。其中一项实验使用了被理论计算机科学顶级会议STOC 2026接收的论文，我们回答了这些论文中提出的四个问题，并得到了论文作者的确认。

    arXiv:2610.09769v1 Announce Type: cross  Abstract: Large language models are increasingly contributing to mathematical research, where progress often depends on efficient proof search, incremental improvements and careful verification. We describe Bolzano, a multi-agent open-source system that uses parallel prover agents with a verifier agent and maintains a human-readable research state. Initial manual use on expert-selected problems yielded 8 results whose proofs were checked by domain experts. Motivated by these case studies, we ran Bolzano without problem-specific human guidance on about 3,800 open problems extracted from four sets of papers, solving about 200 open problems. One experiment used papers accepted to STOC 2026, a top conference in theoretical computer science. There, we answered four questions raised in the papers, as confirmed by their authors.
    
[^44]: PARC-Loc：基于部分分配与关系一致性的文本到点云定位

    PARC-Loc: Text-to-Point-Cloud Localization with Partial Assignment and Relational Consistency

    [https://arxiv.org/abs/2610.09761](https://arxiv.org/abs/2610.09761)

    提出PARC-Loc框架，通过联合建模提示-物体兼容性与成对空间关系的部分分配机制，解决了城市环境中布局不一致混叠和跨子地图边界证据不完整两大文本到点云定位难题。

    

    文本到点云定位旨在根据对周围物体的描述，在城市级三维地图中估计一个位置。现有的由粗到精方法利用聚合的学习兼容性检索子地图，然后在选定的子地图内进行定位。然而，重复或相似的城市物体可能会抬高查询与多个子地图之间的嵌入相似度，即使子地图内的实例布局与查询描述相矛盾。与此同时，与查询相关的实例往往跨越子地图边界，使得检索到的子地图缺少完整的上下文证据。我们分别将这两种失败模式称为“布局不一致混叠”和“边界证据不完整”。为了解决这些问题，我们提出了PARC-Loc，一个建立在部分分配与关系一致性（PARC）之上的由粗到精定位框架。PARC联合建模提示-物体兼容性与成对空间关系，允许未匹配的元素存在，同时倾向于……

    arXiv:2610.09761v1 Announce Type: cross  Abstract: Text-to-point-cloud localization estimates a position in a city-scale 3D map from descriptions of surrounding objects. Existing coarse-to-fine methods retrieve submaps using aggregate learned compatibility and then localize within a selected submap. However, repetitive or similar urban objects can inflate the embedding similarity between the query and multiple submaps, even when the instance layout within a submap violates the query description. Meanwhile, query-relevant instances often span submap boundaries, leaving the retrieved submap with incomplete contextual evidence. We term these failure modes layout-inconsistent aliasing and boundary evidence incompleteness, respectively. To address them, we propose PARC-Loc, a coarse-to-fine localization framework built on Partial Assignment with Relational Consistency (PARC). PARC jointly models hint-object compatibility and pairwise spatial relations, allowing unmatched elements while favo
    
[^45]: Shaer：基于韵律子形式与语义条件的可控阿拉伯语诗歌生成

    Shaer: Controlled Arabic Poetry Generation with Meter Subform and Semantic Conditioning

    [https://arxiv.org/abs/2610.09756](https://arxiv.org/abs/2610.09756)

    Shaer是一个联合条件于自然语言语义描述、韵律子形式和诗歌长度的可控经典阿拉伯语诗歌生成框架，借助包含11.6万首诗歌的增强语料库和QLoRA微调，首次实现了对语义、韵律和篇幅的细粒度联合控制。

    

    经典阿拉伯语诗歌生成需要同时满足语义、语言学和细粒度的韵律约束。现有系统通常只控制较为宽泛的诗歌属性，而未能联合建模语义意图、韵律子形式和诗歌长度。我们提出了Shaer，一个可控的经典阿拉伯语诗歌生成框架，联合以自然语言描述、韵律子形式和目标半句数量为条件。为支持这一任务，我们基于Ashaar构建了一个包含116,032首经典阿拉伯语诗歌的增强语料库，其中包含规范化的韵律子形式标签以及自动生成并经验证的语义描述。随后，我们采用基于QLoRA的监督微调方法（仅使用补全目标）对Yehia-7B进行适配。我们的评估结合了对基础韵律符合度、子形式遵循度和长度控制的自动评估、三个大语言模型评审、盲测人工评估以及记忆化分析。

    arXiv:2610.09756v1 Announce Type: new  Abstract: Classical Arabic poetry generation requires simultaneously satisfying semantic, linguistic, and fine-grained prosodic constraints. Existing systems typically control broad poetic attributes but do not jointly model semantic intent, meter subform, and poem length. We present Shaer, a controllable Classical Arabic poetry generation framework jointly conditioned on natural-language descriptions, meter subforms, and target hemistich counts. To support this task, we construct an enriched corpus of 116,032 classical Arabic poems derived from Ashaar, containing normalized meter-subform labels and automatically generated, validated semantic descriptions. We then adapt Yehia-7B using QLoRA-based supervised fine-tuning with a completion-only objective. Our evaluation combines automatic assessment of base-meter conformity, requested-subform adherence, and length control with three LLM judges, blinded human evaluation, and memorization analysis. Sha
    
[^46]: 桥接路由头：多语言多跳推理在大语言模型中的所在之处

    Bridge Routing Heads: Where Multilingual Multi-hop Reasoning Lives in LLMs

    [https://arxiv.org/abs/2610.09733](https://arxiv.org/abs/2610.09733)

    该研究在大语言模型中识别出负责多跳推理的“桥接路由头”，发现这些注意力头在不同语言间几乎互斥、各自形成语言特异回路，通过消融实验提供了因果证据，并证明无需训练、仅放大这些头即可挽救超过一半的跨语言推理失败。

    

    多语言大语言模型能够跨语言回答相同的多跳推理问题，但我们尚缺乏对其是否共享内部回路的机制性解释。我们通过一个三阶段流程在两个大型多语言LLM中识别出了桥接路由头（BRH）。所得到的特定语言头集合在五种语言之间表现出近乎完全的互斥性，Llama 3.1 70B的平均Jaccard相似度仅为0.017，Qwen 2.5 72B为0.057，揭示了语言特异性的回路。消融通用BRH会使两跳推理的负对数似然（NLL）增加至随机头基线的39-89倍，为其作用提供了直接的因果证据。在目标语言推理失败时放大这些头，无需任何训练即可挽救多达51.7%的跨语言失败案例。两个模型共享这种双回路模式，但头部的分配方式不同：Llama将链式推理集中于一个大型通用池中，而Qwen则更依赖于更大的语言特异性头。

    arXiv:2610.09733v1 Announce Type: new  Abstract: Multilingual LLMs answer the same multi-hop reasoning question across languages, but we lack a mechanistic account of whether they share an internal circuit. We identify Bridge Routing Heads (BRH) in two large multilingual LLMs through a three-stage pipeline. The resulting language-specific head sets exhibit near-complete mutual exclusivity across the five languages, with a mean Jaccard similarity of only 0.017 for Llama 3.1 70B and 0.057 for Qwen 2.5 72B, revealing language-idiosyncratic circuits. Ablating general BRH increases two-hop Negative Log-Likelihood (NLL) by 39-89x the random-head baseline, providing direct causal evidence of their role. Amplifying these heads in a failing target-language pass rescues up to 51.7% of cross-lingual failures, with no training. The two models share this dual-circuit pattern but allocate heads differently: Llama concentrates chaining in a large general pool, while Qwen leans on larger language-spec
    
[^47]: 迈向解释信息检索中查询扩展性能

    Towards Explaining Query Expansion Performance in Information Retrieval

    [https://arxiv.org/abs/2610.09724](https://arxiv.org/abs/2610.09724)

    本研究提出理想扩展查询（IEQ）概念和基于Cohen's d的可分离性度量这两个互补视角，用以解释查询扩展技术在不同查询上性能差异的原因。

    

    查询扩展（QE）技术长期以来被广泛应用于信息检索（IR）中，以解决词汇不匹配问题。它们在现代检索系统中仍然具有重要价值，包括基于大语言模型（LLM）的检索系统。然而，没有任何单一的QE方法能够在所有查询上一致地优于其他方法。本工作试图通过两个互补的视角来解释QE性能的差异。第一个是理想扩展查询（IEQ）的概念——即一种假设性的查询，它能够在下游BM25检索模型中最大化检索效果。第二个是可分离性视角，它使用Cohen's d来量化在给定扩展查询下，相关文档与非相关文档被评分的区分程度。我们开发了一种可分离性度量以及逼近IEQ的实用公式，并研究这些因素与检索效果之间的关系。我们在TREC Robust数据集上进行了大量实验。

    arXiv:2610.09724v1 Announce Type: cross  Abstract: Query Expansion (QE) techniques have long been widely used in Information Retrieval (IR) to address the vocabulary mismatch problem. They remain relevant in modern retrieval systems, including those based on large language models (LLMs). However, no single QE method consistently outperforms others across all queries. This work seeks to explain the variation in QE performance through two complementary perspectives. The first is the concept of an Ideal Expanded Query (IEQ)--a hypothetical query that maximizes retrieval effectiveness with a downstream BM25 retrieval model. The second is a separability perspective, which quantifies how distinctly relevant and non-relevant documents are scored for a given expanded query using Cohen's (d). We develop a separability measure and practical formulations to approximate the IEQ and investigate how these factors relate to retrieval effectiveness. Extensive experiments on the TREC Robust collection,
    
[^48]: SpikingVLA：异步脉冲视觉-语言-动作模型

    SpikingVLA: Asynchronous Spiking Vision-Language-Action Models

    [https://arxiv.org/abs/2610.09710](https://arxiv.org/abs/2610.09710)

    提出SpikingVLA框架，通过树突整合放电（DIF）神经元减少所需时间步，并引入异步执行机制重叠各组件的时间计算，实现了准确且低延迟的ANN-to-SNN脉冲视觉-语言-动作模型转换。

    

    ANN-to-SNN转换通过避免从零开始训练大规模SNN的巨大成本，为实现节能的脉冲视觉-语言-动作（VLA）模型提供了一条实用途径。然而，现有方法通常需要许多时间步才能保持有竞争力的性能，导致实时VLA部署中存在显著的推理延迟。为应对这一挑战，我们提出了SpikingVLA，这是一个能够实现准确且低延迟脉冲VLA推理的ANN-to-SNN转换框架。具体而言，我们提出了一种树突整合放电（DIF）神经元，通过树突混合和自适应胞体放电来缓解通道维度的激活异常值，从而在更少的时间步下实现准确的ANN-to-SNN转换。基于DIF神经元，我们进一步引入了一种异步执行机制，该机制在VLA各组件之间重叠时间计算，降低了同步开销和延迟。大量实验表明……

    arXiv:2610.09710v1 Announce Type: new  Abstract: ANN-to-SNN conversion offers a practical route toward energy-efficient spiking Vision-Language-Action (VLA) models by bypassing the substantial cost of training large-scale SNNs from scratch. However, existing methods often require many timesteps to maintain competitive performance, resulting in substantial inference latency for real-time VLA deployment. To address this challenge, we introduce SpikingVLA, an ANN-to-SNN conversion framework that enables accurate and low-latency spiking VLA inference. Specifically, we propose a Dendritic Integrate-and-Fire (DIF) neuron that alleviates channel-wise activation outliers through dendritic mixing and adaptive somatic firing, enabling accurate ANN-to-SNN conversion with fewer timesteps. Building on DIF neurons, we further introduce an asynchronous execution mechanism that overlaps temporal computation across VLA components, reducing synchronization overhead and latency. Extensive experiments dem
    
[^49]: 从帕累托到偏好：通过摊销式智能体策略发现实现个性化测试时扩展

    From Pareto to Preference: Personalized Test-Time Scaling via Amortized Agentic Policy Discovery

    [https://arxiv.org/abs/2610.09684](https://arxiv.org/abs/2610.09684)

    提出了PersonTTS框架，将个性化测试时扩展表述为发现能同时最大化用户准确率、延迟和成本多维需求联合满足率的可执行控制器，并通过需求匹配初始化和源蒸馏指导复用历史搜索经验，从而摊销新用户画像下的策略发现开销。

    

    测试时扩展（TTS）通过为大型语言模型分配额外的推理计算来提升其推理能力。现有的提高TTS效率的方法大多一次只针对单一资源维度优化准确率，即要么推进准确率-成本的帕累托前沿，要么推进准确率-延迟的帕累托前沿。然而，用户需求是多维度的：用户可能同时指定准确率、延迟和推理成本要求，且不同的需求可能偏好不同的控制器。我们将个性化测试时扩展表述为发现能够最大化用户特定需求联合满足率的可执行控制器。为了降低针对新用户画像重复进行策略发现的开销，我们提出了PersonTTS，这是一个摊销式智能体策略发现框架，它通过需求匹配的控制器初始化和源蒸馏的程序化指导来复用先前的搜索经验，同时保持……

    arXiv:2610.09684v1 Announce Type: new  Abstract: Test-time scaling (TTS) improves the reasoning capabilities of large language models by allocating additional inference computation. Existing approaches to improving TTS efficiency largely optimize accuracy against one resource dimension at a time, advancing either the accuracy--cost or accuracy--latency Pareto frontier. Yet user requirements are multidimensional: users may specify accuracy, latency, and inference-cost requirements jointly, and different requirements can favor different controllers. We formulate Personalized Test-Time Scaling as discovering executable controllers that maximize the joint satisfaction rate of user-specific requirements. To reduce the overhead of repeated policy discovery for new user profiles, we propose PersonTTS, an amortized agentic policy-discovery framework that reuses prior search experience through requirement-matched controller initialization and source-distilled procedural guidance, while retainin
    
[^50]: InsClaimBench：面向决策链的保险理赔裁定基准测试

    InsClaimBench: Benchmarking Insurance Claim Adjudication Across the Decision Chain

    [https://arxiv.org/abs/2610.09671](https://arxiv.org/abs/2610.09671)

    提出了首个端到端评估保险理赔裁定全决策链的基准InsClaimBench，基于3,780个真实案例和86,656条原子规则判断，揭示了LLM从规则判断到赔付计算各层级间可靠性逐级下降的问题。

    

    面向推理的大语言模型（LLM）的最新进展推动了对其实施专业决策任务能力的日益增多的评估。保险理赔裁定正是这样一项任务，它要求模型在结构化的决策过程中将案件证据、保险规则、中间判断和赔付计算相互关联。我们提出了InsClaimBench，一个用于跨决策链评估保险理赔裁定的端到端基准。该基准基于真实理赔材料和结构化保险规则构建，包含涵盖汽车、财产和健康保险三大领域的375个案例族共3,780个案例，以及86,656条原子规则判断。它对每个理赔案件从原子规则、裁定模块到赔付决策与金额进行全流程评估，并通过受控的事实变体来检验所需的变更是否在各层级之间被正确传播。对六个大语言模型的评估结果显示，随着决策层级的深入，模型的可靠性呈现逐级下降的趋势。

    arXiv:2610.09671v1 Announce Type: new  Abstract: Recent advances in reasoning-oriented large language models (LLMs) have motivated increasing evaluation of their ability to perform professional decision tasks. Insurance claim adjudication is one such task, requiring models to connect case evidence, insurance rules, intermediate judgments, and payout calculations across a structured decision process. We introduce InsClaimBench, an end-to-end benchmark for evaluating insurance claim adjudication across the decision chain. Grounded in real claim materials and structured insurance rules, InsClaimBench contains 3,780 cases in 375 case families across auto, property, and health insurance, comprising 86,656 atomic rule judgments. It evaluates each claim from atomic rules through adjudication modules to payout decisions and amounts, with controlled factual variants testing whether required changes are correctly propagated across levels. Evaluation of six LLMs reveals a progressive loss of reli
    
[^51]: SAPD：步对齐特权蒸馏

    SAPD: Step-Aligned Privileged Distillation

    [https://arxiv.org/abs/2610.09665](https://arxiv.org/abs/2610.09665)

    提出 SAPD，一种无需在线采样的自蒸馏后训练方法，通过将参考解的每个推理步骤与针对性的特权引导对齐，使固定示范也能支撑具有竞争力的离线策略学习。

    

    arXiv:2610.09665v1 公告类型：新论文 摘要：在线策略后训练可以通过让大语言模型从自身生成的轨迹中学习来提升其能力，但需要付出高昂的轨迹采样生成代价。我们探究固定示范能否通过更好的监督方式来支持具有竞争力的离线策略学习。我们的前提假设是：固定示范的有用性不仅取决于训练轨迹本身，还取决于监督能否在各个可能的续写之间提供有信息量的偏好，并将这种引导与正在学习的推理决策联系起来。我们提出了步对齐特权蒸馏（SAPD），这是一种无需轨迹采样的自蒸馏方法，能够将示范转化为步对齐的分布监督。其核心洞察在于：利用参考解已知的推进过程，将每一个推理步骤的转变与有针对性的特权引导相关联，而不是把参考解当作无差别的上下文来对待。在数学推理基准上，SAPD 的表现优于监督微调和标签（原文摘要至此截断）

    arXiv:2610.09665v1 Announce Type: new  Abstract: On-policy post-training can improve large language models by learning from their own trajectories, but requires costly rollout generation. We ask whether fixed demonstrations can support competitive off-policy learning through better supervision. Our premise is that their usefulness depends not only on the training trajectories, but also on whether supervision provides informative preferences among continuations and connects this guidance to the reasoning decision being learned. We introduce Step-Aligned Privileged Distillation (SAPD), a rollout-free self-distillation method that turns demonstrations into step-aligned distributional supervision. Its key insight is to use the known progression of a reference solution to associate each reasoning transition with targeted privileged guidance, rather than treating the solution as undifferentiated context. On mathematical reasoning benchmarks, SAPD outperforms supervised fine-tuning and label 
    
[^52]: Alice：一个基于评分规则的多维度自动简答题评分的大规模德语基准数据集

    Alice: A Large-Scale German Benchmark for Rubric-Based Multi-Dimensional Automatic Short Answer Scoring

    [https://arxiv.org/abs/2610.09661](https://arxiv.org/abs/2610.09661)

    该论文提出了Alice——一个大规模、基于评分规则的德语自动简答题评分基准数据集，从学习表现、知识元素和技能三个维度评估学生，并将其形式化为评分规则检索任务，对多种语言模型进行了基准测试。

    

    自动简答题评分是教育自然语言处理的核心任务。然而，公开可用的基准数据集仍然稀缺，且现有数据集主要评估学生直接回答问题的能力，而非学生是否掌握了底层概念（知识元素，如热能）或认知技能（如推理或论证）。为了填补这一空白，我们推出了Alice，这是一个大规模、基于评分规则且与教学目标对齐的德语ASAS数据集，包含三个子任务：(i) 学习表现，(ii) 知识元素，(iii) 技能。我们进一步将基于评分规则的ASAS形式化为一个评分规则检索任务，并使用一系列语言模型对该数据集进行基准测试，涵盖从仅编码器模型到轻量级大语言模型。我们还通过大语言模型的零样本提示和标准分类基线对数据集进行了评估。实验表明，大语言模型尤其难以……

    arXiv:2610.09661v1 Announce Type: new  Abstract: Automatic Short Answer Scoring (ASAS) is central to NLP for Education. However, openly available benchmarks remain scarce, and existing datasets largely address how well students answer a question directly rather than how well they master underlying concepts (knowledge elements) such as thermal energy or epistemic activities (skills) such as reasoning or claim.   To address this gap, we introduce Alice, a large-scale, rubric-based German ASAS dataset that is pedagogically aligned and comprises three subtasks: (i) learning performance (Alice-LP), (ii) knowledge elements (Alice-KE), and (iii) skills (Alice-SK).   We further formulate rubric-based ASAS as a rubric-retrieval task and benchmark the dataset with a range of language models, from encoder-only models to lightweight LLMs. We also benchmark the dataset with zero-shot prompting via LLMs and a standard classification baseline. The experiments show that LLMs, in particular, struggle t
    
[^53]: 评分标准片段即标签表示：用于简答题评分的联合大语言模型编码

    Rubric Spans are Label Representations: Joint LLM Encoding for Short Answer Scoring

    [https://arxiv.org/abs/2610.09660](https://arxiv.org/abs/2610.09660)

    RUSPAN框架将评分标准描述作为语义标签表示，在单次大语言模型编码中联合处理题目、答案和评分标准级别，并通过评分标准独立掩码实现对未见评分标准集的零样本迁移，显著提升简答题自动评分性能。

    

    自动简答题评分（ASAS）需要既能根据题目特定标准对学生答案进行评分，又能保持高效并可在不同评分标准集之间迁移的模型。我们提出了RUSPAN，一个以评分标准为条件的ASAS框架，它将评分标准描述视为语义标签表示。RUSPAN将题目背景、学生答案和所有候选评分标准级别序列化为单个序列，然后在单次语言模型传递中，基于评分标准片段和整个序列的表示对各评分级别进行列表式评分。我们进一步提出了RUSPAN-RIM，其中评分标准独立掩码阻止评分标准片段之间相互关注，使评分标准表示仅依赖于答案和题目背景，从而防止训练过程中对评分标准模式的过拟合，实现零样本迁移。在涵盖英语、德语和葡萄牙语的六个ASAS基准测试中，RUSPAN提升了单基准评分表现，优于判别式和生成式方法（摘要原文在此处被截断）。

    arXiv:2610.09660v1 Announce Type: new  Abstract: Automatic Short Answer Scoring (ASAS) requires models that can score student responses against question-specific criteria while remaining efficient and transferable across rubric sets. We propose RUSPAN, a rubric-conditioned ASAS framework that treats rubric descriptions as semantic label representations. RUSPAN serialises the question context, student answer, and all candidate rubric levels into a single sequence, then scores the levels listwise from the rubric-span and whole-sequence representations produced in a single LM pass. We further introduce RUSPAN-RIM, in which a Rubric-Independent Mask prevents rubric spans from attending to one another, making rubric representations depend only on the answer and question context and preventing overfitting to rubric patterns during training for zero-shot transfer. On six ASAS benchmarks spanning English, German, and Portuguese, RUSPAN improves mono-benchmark scoring over discriminative and ge
    
[^54]: 当大语言模型退化时秩反而上升

    When Rank Rises as LLMs Degrade

    [https://arxiv.org/abs/2610.09647](https://arxiv.org/abs/2610.09647)

    该研究发现LLM后训练中的表示退化（如数据重复）会使RankMe等谱秩指标不降反升，导致单边监控将最差模型误判为最健康，因此指标变化方向取决于具体的退化模式与统计量配对，传统监控假设并不安全。

    

    后训练使语言模型适应非平稳环境。从业者通常使用RankMe及相关谱统计量来监控表示健康度，并往往假设当表示退化时秩会下降。我们证明这一假设对于LLM后训练是不安全的。在对Qwen3-0.6B进行的受控研究（四种退化模式、三个随机种子）中，数据重复使留出集损失相比健康状态恶化75%，同时使原始和中心化RankMe均上升；后者的变化幅度达到13.5个合并标准差。协方差有效秩升至接近其健康值的两倍。这种失败是谱离散化而非表示坍缩，因此单边监控器会将最差的检查点评为最健康的。相比之下，学习率配置错误会降低中心化RankMe和k95，而未中心化的RankMe在不同随机种子间表现不一致。因此，指标变化方向是“退化模式-统计量”配对的固有属性，无法通过重新校准来修复。

    arXiv:2610.09647v1 Announce Type: cross  Abstract: Post-training adapts language models in non-stationary environments. Practitioners monitor representation health with RankMe and related spectral statistics, often assuming that rank falls when representations degrade. We show that this assumption is unsafe for LLM post-training. In a controlled study of Qwen3-0.6B with four degradation modes and three seeds, data duplication worsens held-out loss by 75% relative to healthy while increasing both original and centred RankMe; the latter changes by 13.5 pooled standard deviations. Covariance effective rank rises to nearly twice its healthy value. This failure is spectral dispersion rather than collapse, so a one-sided monitor rates the worst checkpoint as the healthiest. By contrast, a learning-rate misconfiguration lowers centred RankMe and k95, while uncentred RankMe is inconsistent across seeds. Direction is therefore a property of the regime-statistic pair and cannot be fixed by recal
    
[^55]: 在策略蒸馏教会新技能，却不传授新知识

    On-Policy Distillation Teaches New Skills but Not New Knowledge

    [https://arxiv.org/abs/2610.09639](https://arxiv.org/abs/2610.09639)

    反向KL在策略蒸馏只向学生模型迁移多步推理的组合性技能而不迁移事实性知识，改用正向KL则可恢复知识迁移。

    

    在策略蒸馏（OPD）能够增强语言模型的推理能力，然而学生模型是否因此习得了新的事实性知识，或是获得了多步推理的组合性技能，目前仍不清楚。我们构建了一个受控的合成框架来分离这两种能力：该框架测量学生模型的初始能力，并独立控制教师模型所额外具备的事实知识、组合性技能或两者兼有。在来自三个模型系列的四个模型上的实验表明，反向KL在策略蒸馏能够在未见过的推理结构上可靠地迁移组合性技能，但迁移的事实性知识极少。通过对蒸馏配方进行解耦分析，我们揭示了这种不对称性的来源：将反向KL替换为正向KL可以恢复事实性知识的迁移，而学生模型的rollout（自采样）则专门改善多步推理的执行。在近期的事实性问答和竞赛数学上的实验显示，反向KL在策略蒸馏下存在类似的不对称性，即在不习得事实性记忆的情况下也能带来显著的推理提升。

    arXiv:2610.09639v1 Announce Type: new  Abstract: On-policy distillation (OPD) strengthens language-model reasoning, yet whether students acquire new factual knowledge or compositional skill for multi-step reasoning remains unknown. We separate these capabilities using a controlled synthetic framework that measures the student's initial capabilities and independently controls the teacher's additional facts, compositional skill, or both. Across four models from three families, reverse-KL OPD reliably transfers compositional skill across unseen reasoning structures, but transfers minimal factual knowledge. Decoupling the distillation recipe reveals the source of this asymmetry: replacing reverse KL with forward KL restores factual transfer, whereas student rollouts specifically improve the execution of multi-step reasoning. Experiments on recent factual QA and competition mathematics show a similar asymmetry under reverse-KL OPD, yielding notable reasoning gains without factual memory exp
    
[^56]: 编码智能体基准测试应匹配其用户的任务流

    Coding-Agent Benchmarks Should Match Their Users' Task Flows

    [https://arxiv.org/abs/2610.09633](https://arxiv.org/abs/2610.09633)

    该研究通过收集JetBrains IDE中真实软件工程师的4,782个智能体会话，发现真实任务流在任务类型与切换模式上高度多样且因数据源而异，因此编码智能体基准测试应先指明目标用例，再依据其真实测得的任务流进行校准。

    

    编码智能体的评估通常力求尽可能贴近真实。在本研究中，我们收集了JetBrains IDE中真实软件工程师的4,782个智能体会话，我们称之为“生产会话”。由于我们的研究对象是交互式智能体，我们研究了包含至少三条用户消息的会话（占样本的33%）。这些长会话与源自issue的基准测试任务在两个方面有所不同：(i) 用户请求所涵盖的任务类型范围要广泛得多——包括对项目代码的提问、规划、审查、重构、执行等；(ii) 用户会在整个会话过程中于不同任务类型之间切换。来自三个公开交互语料库的长会话样本展现出显著不同的任务流——即会话长度、任务类型以及类型间转换的分布——因此没有任何单一的交互分布是普遍真实的：基准测试应当指明目标用例，并根据从该用例中测得的数据进行校准。我们提出了SWE-TaskFlow，一种……（原文摘要在此处截断）

    arXiv:2610.09633v1 Announce Type: cross  Abstract: The evaluation of coding agents generally strives to be as realistic as possible. In our study, we collect 4,782 agent sessions of real software engineers in JetBrains IDEs, which we call Production Sessions. Since our subject is interactive agents, we study the sessions with at least three user messages (33% of the sample). These long sessions differ from issue-derived benchmark tasks in two ways: (i) user requests span a far wider mix of task types - questions about the project's code, planning, review, refactoring, execution - and (ii) users switch between types throughout a session. Long-session samples from three public interaction corpora exhibit markedly different Task Flows (the distributions of session lengths, task types, and type-to-type transitions), so no single interaction distribution is universally realistic: benchmarks should name a target use case and calibrate to measurements from it. We present SWE-TaskFlow, an appr
    
[^57]: 骨架应该说哪种语言？多语言推理中的语言选择

    Which Language Should a Skeleton Speak? Language Choices in Multilingual Reasoning

    [https://arxiv.org/abs/2610.09607](https://arxiv.org/abs/2610.09607)

    该论文提出语言感知骨架探索框架（LASEF），系统研究多语言数学推理中推理骨架应使用何种语言，发现英语骨架仅有轻微的平均优势且并非普遍最优，并归纳出骨架语言效应的三种模式（方向一致、依赖评估与基准、非对称负面）。

    

    基于骨架的推理提示（Skeleton-based reasoning prompting）是一种无需训练的结构化大语言模型（LLM）推理的方法，但先前的工作大多假设以英语为中心的设置。我们提出了语言感知骨架探索框架（Language-Aware Skeleton Exploration Framework，LASEF），以研究多语言数学推理中骨架语言的选择问题。在数学基准、模型规模和多种语言的广泛实验中，我们发现英语骨架平均上带来轻微的正面倾向，这一效应在较小规模模型和低资源语言上最为明显。然而，经过显著性校正后，仅有少数语言层面的增益仍然显著，且英语并非普遍最优。结合贪心解码、多次采样评估、翻译消融实验和跨基准验证，我们进一步识别出骨架语言效应的三种模式：方向一致型、依赖评估方式和基准类型型，以及非对称负面型。这些效应无法仅用生成质量来完全解释。总体而言，骨架语……（摘要原文在此处截断）

    arXiv:2610.09607v1 Announce Type: new  Abstract: Skeleton-based reasoning prompting is a promising training-free approach for structuring LLM reasoning, but prior work largely assumes an English-centric setting. We propose the Language-Aware Skeleton Exploration Framework (LASEF) to study skeleton-language choice in multilingual mathematical reasoning. Across math benchmarks, model scales, and languages, we show that English skeletons yield a small positive tendency on average, most visible for smaller models and low-resource languages. However, few language-level gains remain significant after correction, and English is not universally optimal. Combining greedy decoding, multi-rollout evaluation, translation ablation, and cross-benchmark validation, we further find three patterns of skeleton-language effects: directionally consistent, evaluation- and benchmark-dependent, and asymmetric negative. These effects cannot be fully explained by generation quality alone. Overall, skeleton lan
    
[^58]: 通过交叉反馈与连贯性策展实现的协同推理蒸馏

    Collaborative Reasoning Distillation via Cross-Feedback and Coherent Curation

    [https://arxiv.org/abs/2610.09587](https://arxiv.org/abs/2610.09587)

    该论文提出协同推理蒸馏框架 CRD，结合教师间交叉反馈、与答案无关的逐步质量评估和连贯性步骤拼接，并通过带预算约束的推理质量优化训练学生模型，使 CRD-4B 仅用 5 万条训练数据便在 MATH-500 和 AIME'25 上超越基线。

    

    推理能力对于推进大型语言模型的发展至关重要，然而现有方法要么需要庞大的计算预算，要么难以有效地将推理能力蒸馏到较小的模型中。标准的蒸馏方法依赖于基于结果的奖励，无法区分合理的推理与侥幸的猜测。我们提出了协同推理蒸馏（Collaborative Reasoning Distillation, CRD），这是一个通过三项创新来增强紧凑模型推理能力的框架：（1）交互式交叉反馈，教师之间迭代地相互批评彼此的推理；（2）细粒度的逐步质量评估，独立于最终答案捕捉逻辑有效性；（3）连贯性感知的步骤拼接，综合互补优势。学生模型通过带预算约束的推理质量优化（RQO）进行训练。我们的模型 CRD-4B 在 MATH-500 上达到 97.3%，在 AIME'25 上达到 70.3%，仅使用 5 万条训练数据便超越了基线模型。

    arXiv:2610.09587v1 Announce Type: cross  Abstract: Reasoning capabilities are critical for advancing Large Language Models, yet current approaches either require massive computational budgets or struggle to effectively distill reasoning to smaller models. Standard distillation methods rely on outcome-based rewards, failing to distinguish between sound reasoning and lucky guesses. We propose Collaborative Reasoning Distillation (CRD), a framework that enhances reasoning in compact models through three innovations: (1) interactive cross-feedback where teachers iteratively critique each other's reasoning, (2) fine-grained step-wise quality assessment capturing logical validity independent of final answers, and (3) coherence-aware step stitching that synthesizes complementary strengths. Students are trained via Reasoning Quality Optimization (RQO) with budget constraints. Our model, CRD-4B, achieves 97.3% on MATH-500 and 70.3% on AIME'25, surpassing baselines while using only 50K training 
    
[^59]: 大语言模型在否定句下如何改变预测？

    How Do LLMs Change Predictions Under Negation?

    [https://arxiv.org/abs/2610.09571](https://arxiv.org/abs/2610.09571)

    大语言模型通过“抑制原始答案、提升偏好候选”的机制处理否定，而非像人类那样利用原始答案信息来确定应排除的内容，这一与人类处理方式的差异是模型否定任务失败的关键根源。

    

    否定是人类语言的一个基本特征，然而大型语言模型（LLMs）在处理否定时仍然不可靠。我们在自建的否定基准上评估了最新的开源和闭源LLMs，发现在37%至71%的情况下，模型在否定句下重复给出相同的答案（例如，对于“什么不是西班牙的首都？”回答“马德里”）。为了理解并解决这种脆弱性，我们从机制层面考察了模型在否定下的运作方式。我们的主要发现是，专门的注意力头和MLP神经元通过以下两种方式协同实现否定：（1）抑制对原始答案（如“马德里”）的检索，同时（2）在答案类别中提升某个受偏好候选（如“巴黎”）的概率。这与人类否定处理机制的解释形成对比——在人类处理中，关于原始答案的信息有助于确定应排除的内容。此外，我们发现这种与人类处理方式的差异是否定失败的关键来源。

    arXiv:2610.09571v1 Announce Type: new  Abstract: Negation is an essential feature of human language, yet large language models (LLMs) remain unreliable in processing it. We evaluate recent open-source and closed-source LLMs on our negation benchmark and find that, in 37-71% of cases, they repeat the same answer under negation (e.g., "Madrid" for "What is not the capital of Spain?"). To understand and address this brittleness, we mechanistically examine how models operate under negation. Our main finding is that specialized attention heads and MLP neurons jointly implement negation by (1) suppressing retrieval of the original answer (e.g., "Madrid") while (2) promoting a favored candidate within the answer category (e.g., "Paris"). This contrasts with accounts of human negation processing, in which information about the original answer helps to determine what should be excluded. Furthermore, we find that this difference from human processing is a key source of negation failures: the mod
    
[^60]: RELATE：一个用于衡量大语言模型关系取向的评估框架

    RELATE: An Evaluation Framework for measuring Relational Orientation of Large Language Models

    [https://arxiv.org/abs/2610.09569](https://arxiv.org/abs/2610.09569)

    该论文提出了“关系取向”这一新概念和RELATE评估框架，通过内向型与外向脚手架型两个维度，在多轮对话的句子级别上衡量大语言模型是将用户导向依赖AI，还是促进其现实世界中的人际连接。

    

    大语言模型越来越多地被用于情感支持，这引发了人们的担忧：持续使用可能会使用户远离现实世界中的人际关系。然而，现有的评估主要关注回复的安全性、共情性或有用性，而忽视了一个关系层面的问题：模型在持续支持方面会将用户导向何处？为了回答这个问题，我们提出了“关系取向”这一概念，该属性通过两个非互斥的维度来操作化：内向型语言，即将AI定位为用户持续的支持来源；以及外向脚手架型语言，即鼓励真实世界中的人际连接。基于心理学和社会学文献，我们形式化了关系取向的分类体系，并提出了RELATE——一个基于角色条件化的框架，用于在多轮对话中以句子级别衡量内向型和外向脚手架型语言。RELATE结合了7（原文摘要在此处截断）

    arXiv:2610.09569v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for emotional support, raising concern that sustained use may draw users away from their real-world relationships. Yet existing evaluations primarily focus on the safety, empathy, or helpfulness of responses, leaving under-examined a relational question: where does the model orient the user for continued support? To address this question, we introduce relational orientation, a property operationalized through two non-exclusive dimensions: inward-facing (IF) language, which positions the AI as the user's ongoing source of support, and outward-scaffolding (OS) language, which encourages real-world human connection. Grounded in psychological and sociological literature, we formalize a taxonomy of relational orientation and present RELATE, a persona-conditioned framework for measuring inward-facing and outward-scaffolding language at the sentence level in multi-turn dialogues. RELATE pairs 7
    
[^61]: 宪法引导的水印技术

    Constitution-Guided Watermarking

    [https://arxiv.org/abs/2610.09552](https://arxiv.org/abs/2610.09552)

    本文提出“宪法引导水印”框架，通过将提供商需求表示为自然语言原则，使水印系统能够根据不同请求的需求灵活选择属性权衡，避免了传统方法对所有请求采用统一配置所导致的牺牲问题。

    

    水印技术使语言模型提供商能够识别由其模型生成的文本。然而，水印的理想属性之间可能存在冲突（即更强的水印信号可能会降低文本质量），而抵抗编辑的设计也可能便于伪造。提供商通过选择平衡竞争目标或优先考虑特定属性的配置来应对这些权衡。但这两种方法都会将一个统一的操作点强加于具有不同需求的请求上，可能在需要保留原始措辞时牺牲文本质量，或在需要可靠归因时牺牲鲁棒性。为了实现灵活且可适应的设计，我们提出了宪法引导水印，这是一个能够根据提供商需求（以自然语言原则的形式列出）来为每个请求选择合适权衡的框架。在离线阶段，一个预训练的推理代理会结合水印实现来审查宪法规则，并迭代地重新……

    arXiv:2610.09552v1 Announce Type: cross  Abstract: Watermarking enables language model providers to identify text generated by their models. However, its desired properties can conflict (\ie~stronger watermark signals can degrade text quality), while designs that resist editing may also facilitate forgery. Providers address these trade-offs by choosing configurations that balance competing objectives or prioritize particular properties. Either approach imposes a shared operating point on requests with different requirements, potentially sacrificing quality where wording preservation matters or robustness where reliable attribution is essential. To allow flexible and adaptable designs, we introduce \emph{Constitution-Guided Watermarking}, a framework that selects request-appropriate trade-offs from provider requirements, listed as natural-language principles. \emph{Offline}, a pretrained reasoning agent examines constitutional rules alongside watermark implementations and iteratively re
    
[^62]: 弃权式认证：小校准预算下思维链验证器的无分布保证

    Certified by Abstention: Distribution-Free Guarantees for Chain-of-Thought Verifiers at Small Calibration Budgets

    [https://arxiv.org/abs/2610.09541](https://arxiv.org/abs/2610.09541)

    该研究揭示“通过弃权实现有效性”现象——很少触发的验证证书虽形式上有效但每次触发时可能全部出错，并据此为小校准预算下的思维链验证器建立了无分布认证保证及其失效条件分析。

    

    预测思维链（CoT）轨迹是否正确的信号通常以AUC进行比较，但实际部署需要一个带有保证的阈值。我们研究了在几十到几百个标注问题这一现实校准预算下，无分布选择性保证能为CoT验证器提供什么，实验使用了七个开源模型、五种验证器信号和37,000条已评分轨迹。核心观察是“通过弃权实现有效性”：一个以概率 P_fire 发放证书的 (α,δ)-有效程序，仅能将已发放证书的失败概率约束在 δ/P_fire 以内，因此一个很少触发的证书可以在形式上“有效”，却在每次实际使用时都出错。在一个风险已知的模拟中，标准证书在至多0.3%的校准抽样中失败，但在其触发的抽样中失败率高达69%。认证下限以及Benjamini-Hochberg共形选择的格条件解释了为什么证书……（原文摘要在此处截断）

    arXiv:2610.09541v1 Announce Type: cross  Abstract: Signals that predict whether a chain-of-thought (CoT) trace is correct are compared by AUC, but deploying one requires a threshold with a guarantee. We ask what distribution-free selective guarantees deliver for CoT verifiers at realistic calibration budgets of tens to a few hundred labelled problems, using seven open models, five verifier signals and 37,000 graded traces. The central observation is validity by abstention: an $(\alpha,\delta)$-valid procedure that issues a certificate with probability $P_{\rm fire}$ bounds the failure probability of an issued certificate only by $\delta/P_{\rm fire}$, so a certificate that rarely fires can be valid and wrong every time it is used. In a simulation with known risk the standard certificate fails in at most 0.3% of calibration draws but in up to 69% of those in which it fires. A certification floor and a lattice condition for Benjamini-Hochberg conformal selection explain why certificates 
    
[^63]: 基于Transformer的长文档金融叙述性摘要评估指标对比研究

    A Comparative Study of Evaluation Metrics for Long-Document Financial Narrative Summarization with Transformers

    [https://arxiv.org/abs/2610.09529](https://arxiv.org/abs/2610.09529)

    针对长文档金融叙述摘要任务，本文提出将ROUGE-2与BERTScore调和平均相结合的新型评估指标BRUGE，以更真实地反映摘要质量。

    

    英国伦敦证券交易所有超过2000家上市公司，划分为11个行业板块，这些公司被要求在每个财政年度内至少两次公布其财务业绩。英国的年度报告是非常冗长的文件，平均约80页。在本研究中，我们旨在基于一组不同的预训练Transformer模型，结合不同的抽取技术，对多种摘要方法进行基准测试。此外，我们考虑了多种评估指标，以研究它们在金融叙述摘要（FNS 2020）共享任务数据集上的不同表现和适用性，该数据集由在伦敦证券交易所上市的公司发布的年度报告及其对应摘要组成。我们假设某些评估指标并不能真实反映摘要模型的能力，并提出了一种新颖的BRUGE评分指标，即ROUGE-2与BERTScore的调和平均值。最后，我们进……（原文摘要在此处截断）

    arXiv:2610.09529v1 Announce Type: new  Abstract: There are more than 2,000 listed companies on the UK's London Stock Exchange, divided into 11 sectors who are required to communicate their financial results at least twice in a single financial year. UK annual reports are very lengthy documents with around 80 pages on average. In this study, we aim to benchmark a variety of summarisation methods on a set of different pre-trained transformers with different extraction techniques. In addition, we considered multiple evaluation metrics in order to investigate their differing behaviour and applicability on a dataset from the Financial Narrative Summarisation (FNS 2020) shared task, which is composed of annual reports published by firms listed on the London Stock Exchange and their corresponding summaries. We hypothesise that some evaluation metrics do not reflect true summarisation ability and propose a novel BRUGEscore metric, as the harmonic mean of ROUGE-2 and BERTscore. Finally, we perf
    
[^64]: Goldsmith：基于金标损失引导的定义优化与智能体标注框架

    Goldsmith: Gold-Loss-Guided Definition Optimization with an Agentic Annotation Harness

    [https://arxiv.org/abs/2610.09489](https://arxiv.org/abs/2610.09489)

    Goldsmith 提出了一种智能体化流水线，把小型专家金标集合转化为可训练的结构化标注定义，通过可执行结构化损失与“文本梯度”式迭代修订实现定义优化，在匹配评估协议下超越了直接重写、OPRO、APE 和 PromptBreeder 等方法。

    

    许多标注项目在专家尚未形成稳定指南、也缺乏足够标签来训练任务专用模型之前就已启动。我们提出 Goldsmith，这是一种智能体化流水线，能够将小型金标集合（即代表预期任务边界的专家标注校准样本）转化为可复用的结构化标注定义。Goldsmith 将该定义视为一个可训练的文本对象：候选定义在同一批金标样本上运行，并通过可执行的结构化损失进行评分，而输出模式、格式化、检索、修复、评判和人工审核等环节则保留在外部框架（harness）中。大型语言模型（LLM）编辑器将损失最高的失败案例转化为“文本梯度”式的修订，只有当测得的损失下降时才会接受这些修订。在提示优化的对比实验中，Goldsmith 在匹配的评估协议下优于直接重写、OPRO、APE 和 PromptBreeder。由此产生的定义还能进一步改进下游（摘要在此处截断）……

    arXiv:2610.09489v1 Announce Type: new  Abstract: Many annotation projects begin before experts have a stable guideline or enough labels to train a task-specific model. We present Goldsmith, an agentic pipeline that turns a small gold set---expert-annotated calibration examples representing the intended task boundaries---into a reusable structured annotation definition. Goldsmith treats this definition as a trainable textual object. Candidate definitions are run on the same gold examples and scored with an executable structured loss, while the output schema, formatting, retrieval, repair, judging, and human review remain in an external harness. A large language model (LLM) editor converts the highest-loss failures into textual-gradient revisions, which are accepted only when the measured loss decreases. In prompt-optimization comparisons, Goldsmith improves over direct rewriting, OPRO, APE, and PromptBreeder under matched evaluation protocols. The resulting definition also improves down
    
[^65]: 缓解用于语言识别的自监督语音表征中的口音-语言混淆问题

    Mitigating Accent-Language Confusion in Self-Supervised Speech Representations for Language Identification

    [https://arxiv.org/abs/2610.09486](https://arxiv.org/abs/2610.09486)

    提出一种几何投影方法，仅利用母语语音估计并移除自监督语音表征中的L1口音偏置方向，无需非母语训练数据或模型适配即可显著提升非母语（L2口音）语音的语言识别准确率。

    

    口语语言识别（LID）旨在无论口音如何都能识别目标语言。然而在实践中，从自监督语音表征微调得到的LID模型经常将口音与语言相混淆，把非母语（L2）语音错误地分类为说话者的第一语言（L1）。我们证明，非母语语音表征位于母语目标语言表征和母语L1表征这两个极点之间，从而导致系统性的误分类。为解决这一问题，我们引入了一种几何投影方法，仅从母语语音中估计L1偏置方向，并在冻结的LID分类头之前将其移除。在五个MMS-LID模型和多个非母语语料库上的实验表明，该投影方法显著提升了L2口音语音的目标语言识别准确率，同时保持了对母语语音的预测性能。这些结果表明，口音引起的L1偏置可以直接在表征空间内进行纠正，而无需L2训练数据或模型适配。

    arXiv:2610.09486v1 Announce Type: cross  Abstract: Spoken language identification (LID) aims to recognize the target language regardless of accent. In practice, however, LID models fine-tuned from self-supervised speech representations frequently confuse accents with languages, misclassifying non-native (L2) speech as the speaker's first language (L1). We show that non-native speech representations lie between native target-language and native L1 poles, causing systematic misclassification. To address this, we introduce a geometric projection that estimates an L1-bias direction solely from native speech and removes it before the frozen LID head. Across five MMS-LID models and non-native corpora, this projection substantially improves target language identification for L2-accented speech while preserving predictions for native speech. These results show that accent-induced L1 bias can be corrected directly within the representation space without L2 training data or model adaptation.
    
[^66]: CHASE：面向几何感知模型工程的通道对齐结构利用

    CHASE: Channel-Aligned Structure Exploitation for Geometry-Aware Model Engineering

    [https://arxiv.org/abs/2610.09476](https://arxiv.org/abs/2610.09476)

    本文提出CHASE框架，将几何与谱对齐（GSA）所刻画的结构特征应用于参数高效微调、剪枝补偿、模型合并、KV共享表示和神经元分组等六类模型工程任务，并开发了CAGA、SAKV、CAPS三种新方法。

    

    几何与谱对齐（GSA）通过谱集中性、物理通道对齐、支持结构以及奇异基的变化来刻画训练后的网络。在本文中，我们提出CHASE（通道对齐结构利用），将这些结构应用于实际的模型设计。CHASE涵盖了模型修改、重构和压缩方面的六个应用。CORA、COEC和CORAM将GSA应用于参数高效微调、结构化剪枝补偿和模型合并。我们进一步开发了三种新方法：CAGA利用GSA识别可以共享KV表示的多头注意力头，并通过几何对齐和低秩子空间提取来构建共享的键和值头；SAKV利用GSA确定哪些相邻层可以共享低秩KV缓存表示以及每个层组保留的秩；CAPS利用GSA的谱结构对输出神经元进行分组并……（摘要在此处被截断）

    arXiv:2610.09476v1 Announce Type: cross  Abstract: Geometric and Spectral Alignment (GSA) characterizes trained networks through spectral concentration, physical-channel alignment, support structure, and changes in singular bases. In this paper, we propose CHASE (Channel-Aligned Structure Exploitation) to use these structures in practical model design. CHASE covers six applications across model modification, reconfiguration, and compression. CORA, COEC, and CORAM apply GSA to parameter-efficient finetuning, structured-pruning compensation, and model merging. We further develop three new methods. CAGA uses GSA to identify multi-head attention heads that can share a KV representation and constructs the shared key and value heads through geometric alignment and low-rank subspace extraction. SAKV uses GSA to determine which adjacent layers can share a low-rank KV-cache representation and the retained rank for each layer group. CAPS uses GSA spectral structure to group output neurons and se
    
[^67]: 无边界上下文偏置：面向无分词语言的深度自适应门控与读音空间匹配

    Boundary-Free Contextual Biasing: Depth-Adaptive Gating and Reading-Space Matching for Unsegmented Languages

    [https://arxiv.org/abs/2610.09467](https://arxiv.org/abs/2610.09467)

    该论文提出一种无需词边界的上下文偏置解码方法，基于字符级自动机并结合深度自适应门控与读音空间匹配，无需训练即可显著提升中文和日语ASR中稀有词的召回率。

    

    上下文偏置在推理时为语音识别（ASR）系统提供一份预期词表，但现有方法依赖于日语和中文所不具备的词边界。我们提出了一种面向冻结公开CTC模型的无边界偏置解码器，构建于字符级Aho-Corasick自动机之上，无需训练，也无需二次解码。两个基于证据的机制取代了词边界：一是深度自适应门控，根据匹配深度决定偏置力度；二是读音空间匹配，用于处理音频正确但用字错误的情况。在Aishell-1 NE的困难R1子集上，我们达到66.5%的召回率，超过了经过训练的CLAS基线（64%），且无需重新调参即可迁移到WenetSpeech和第二种架构上。我们发布了首个开放的日语上下文偏置基准，在该基准上，偏置将稀有词召回率提升25个百分点，同时精度保持在97%以上；在使用1,000词列表时，召回率仍分别提升19和22个百分点。

    arXiv:2610.09467v1 Announce Type: cross  Abstract: Contextual biasing supplies an ASR system with a list of expected words at inference time, but existing methods rely on word boundaries that Japanese and Chinese do not provide. We present a boundary-free biasing decoder for frozen public CTC models, built on a character-level Aho-Corasick automaton, with no training and no second pass. Two evidence-based mechanisms replace the boundary: a depth-adaptive gate that sets how hard to push from match depth, and reading-space matching for when the audio is right but the characters are wrong. On Aishell-1 NE's hard R1 subset we reach 66.5% recall, above the trained CLAS baseline (64%), transferring to WenetSpeech and to a second architecture without retuning. We release the first open Japanese contextual-biasing benchmark, where biasing lifts rare-word recall by 25 points at precision above 97%, and still by 19 and 22 points against 1,000-word lists.
    
[^68]: BanglaRhet：面向孟加拉语政治演讲中修辞与说服检测的经典模型与Transformer模型基准测试

    BanglaRhet: Benchmarking Classical and Transformer Models for Rhetorical and Persuasion Detection in Bangla Political Speech

    [https://arxiv.org/abs/2610.09464](https://arxiv.org/abs/2610.09464)

    本文提出了BanglaRhet基准语料库，包含30,289个手动标注的孟加拉语政治演讲片段，用于系统评估经典模型与Transformer模型在修辞技术检测和说服技术检测两项分类任务上的表现。

    

    政治话语常常使用修辞和说服性语言来构建叙事、影响公众舆论并动员受众。尽管孟加拉语自然语言处理在情感分析和观点挖掘方面已取得进展，但针对孟加拉语政治演讲中细粒度修辞与说服技术检测的Transformer模型的系统性基准测试在很大程度上仍未被充分探索。本文提出了一项基于Transformer模型的基准研究，用于检测孟加拉语政治话语中的修辞形式和说服意图。使用BanglaRhet——一个从公开政治新闻来源收集的包含30,289个孟加拉语政治演讲片段的手动标注语料库，我们构建了两个有监督单标签分类任务：修辞技术检测（对比、重复、夸张、隐喻、反问）和说服技术检测（归责、行动号召、团结号召、道德……）

    arXiv:2610.09464v1 Announce Type: new  Abstract: Political discourse often uses rhetorical and persuasive language to frame narratives, influence public opinion, and mobilize audiences. While Bangla natural language processing has made progress in sentiment analysis and opinion mining, systematic benchmarking of transformer models for fine-grained rhetorical and persuasion technique detection in Bangla political speech remains largely underexplored. This paper presents a benchmark study of transformer-based models for detecting rhetorical form and persuasive intent in Bangla political discourse. Using BanglaRhet, a manually annotated corpus of 30,289 Bangla political speech segments collected from publicly available political news sources, we formulate two supervised single-label classification tasks: rhetorical technique detection (contrast, repetition, exaggeration, metaphor, rhetorical questions) and persuasion technique detection (blame assignment, call to action, unity call, moral
    
[^69]: 数字正确，州份错误？测量大语言模型在州政策记忆中的跨辖区替代现象

    Right Number, Wrong State? Measuring Cross-Jurisdiction Substitution in LLM Recall of State Policy

    [https://arxiv.org/abs/2610.09458](https://arxiv.org/abs/2610.09458)

    该论文提出一种“固定问题措辞、仅改变辖区”的最小集设计，首次系统测量了大语言模型在回答州级政策问题时返回其他州真实数值的“跨辖区替代”现象，并揭示宽松的归因方法会将其高估3至5倍。

    

    当大语言模型错误地回答一个特定州的政策问题时，它可能是在产生幻觉，也可能是在返回一个确实存在于另一个州的真实数值。我们通过一个“最小集设计”来检验这一点：问题的措辞保持固定，仅改变所问的辖区，涵盖美国50个州和哥伦比亚特区（共51个辖区）以及三个精确定义的医疗补助收入资格数量。金标准数值来自官方数据手册，且在核查的102个数据单元格中有101个与独立来源一致。在预先注册的协议下，Claude Sonnet 5.5和GPT-5.6 Sol可复现地给出另一个州的当前数值——在两次独立重复中结果完全相同——分别占153个条目中的10个和25个。然而，归因过程是脆弱的：若将任何恰好等于另一州数值的错误答案都记为跨州替代，其得到的可复现替代数量是“逐一核对所问州自身记录中每个数字”这一严格方法的3至5倍，因为许多表面上的跨州答案实际上是所问州自身的正确数值（摘要原文在此处截断）。

    arXiv:2610.09458v1 Announce Type: new  Abstract: When an LLM answers a state-specific policy question wrongly, it may be hallucinating, or it may be returning a real value that holds in another state. We test this with a minimal-set design: the question wording is fixed and only the jurisdiction varies, across the 50 U.S. states and the District of Columbia (51 jurisdictions) and three exactly defined Medicaid income-eligibility quantities. Gold values come from an official data book and agree with an independent source in 101 of 102 checked cells. Under a pre-registered protocol, Claude Sonnet 5.5 and GPT-5.6 Sol reproducibly give another state's current value, identical across two independent repeats, for 10 and 25 of 153 items. Attribution is fragile, however. Crediting any wrong answer that equals another state's value yields 3-5x more reproducible substitutions than checking every number in the asked state's own records, because many apparent cross-state answers are the asked stat
    
[^70]: 北极问题，缺失的答案：面向大语言模型在北极科学中弃答能力的数据集与基准

    Arctic Questions, Missing Answers: A Dataset and Benchmark for LLM Abstention in Arctic Science

    [https://arxiv.org/abs/2610.09446](https://arxiv.org/abs/2610.09446)

    该论文提出了源自北极科学文献的ArcticQA数据集和配对基准ArcticAbstain，用于评估大语言模型根据答案可得性合理弃答的能力，发现各模型弃答行为差异巨大，且在正确答案缺失时弃答率平均仅提高5.05个百分点。

    

    大型语言模型（LLM）在科学多项选择题中，当没有任何选项有效时应当选择弃答，但仅凭频繁弃答并不能证明模型对答案可得性的敏感性。我们推出了ArcticQA，这是一个包含194个源自原始北极研究问题的数据集，并对答案支持和干扰项矛盾情况进行了针对源证据的自动化检查。我们进一步开发了ArcticAbstain，这是一个配对基准，用于比较“答案存在”与“答案缺失”两种条件：在后一种条件下，正确答案被替换为干扰项，而两种条件下均提供明确的弃答选项。我们在高推理强度下评估了来自Gemini、Claude和ChatGPT系列的八个模型，每种条件进行三次试验，共记录了9,312条响应。“答案存在”条件下的弃答率在0.0%到63.0%之间，而将正确答案替换后，弃答率平均仅提高5.05个百分点。这些发现凸显了显著的……

    arXiv:2610.09446v1 Announce Type: new  Abstract: Large language models (LLMs) should abstain from scientific multiple-choice questions when no option is valid, but frequent abstention alone does not demonstrate sensitivity to answer availability. We introduce ArcticQA, a dataset of 194 questions derived from primary Arctic research, with automated checks of answer support and distractor contradiction against source evidence. We further develop ArcticAbstain, a paired benchmark comparing answer-present and answer-absent conditions, with the correct answer replaced by a distractor in the latter and an explicit abstention option in both. We evaluate eight models from the Gemini, Claude, and ChatGPT families at high reasoning effort, with three trials per condition, yielding 9,312 recorded responses. Answer-present abstention rates range from 0.0% to 63.0%, whereas replacing the correct answer increases abstention by 5.05 percentage points on average. These findings highlight substantial b
    
[^71]: 找到恰当的平衡：LLM检索中的相关性与多样性

    Finding the Right Balance: Relevance and Diversity in LLM Retrieval

    [https://arxiv.org/abs/2610.09412](https://arxiv.org/abs/2610.09412)

    该研究提出一种查询自适应的检索多样化规则，仅当最近邻检索到的有效独立文档数低于查询证据需求时才启用多样化，从而在冗余候选池上提升多证据任务的表现，同时避免对干净候选池造成损害。

    

    检索多样化在检索增强生成（RAG）框架中被广泛应用，然而以往的研究对于它是否能改善检索效果和答案质量存在分歧。我们证明其有效性主要取决于候选池的冗余程度，其变化规律与查询所需的独立证据片段数量相一致。通过受控的近重复文本注入和生产环境风格的滚动重叠分块实验，我们发现多样化在干净的候选池上会损害相关性、证据覆盖率和答案质量，但当冗余导致最近邻检索反复选中重复段落时，多样化在多证据任务上会变得有益。因此，我们提出一种查询自适应规则：仅当最近邻top-k选择中有效独立文档的数量低于该查询的证据需求时才启用多样化。该规则可直接基于现有嵌入计算，能够获得大部分可实现的收益，并且具有良好的可迁移性。

    arXiv:2610.09412v1 Announce Type: cross  Abstract: Retrieval diversification is widely available in retrieval-augmented generation (RAG) frameworks, yet prior studies disagree on whether it improves retrieval and answer quality. We show that its effectiveness varies primarily with candidate-pool redundancy, in a pattern consistent with the number of distinct evidence pieces a query requires. Using controlled near-duplicate injection and production-style overlapping chunking, we find that diversification harms relevance, evidence coverage and answer quality on clean pools, but becomes beneficial on multi-evidence tasks when redundancy causes nearest-neighbor retrieval to select repeated passages. We therefore introduce a query-adaptive rule that diversifies only when the effective number of distinct documents in the nearest-neighbor top-$k$ selection falls below the query's evidence requirement. Computed from existing embeddings, the rule captures most of the achievable gain, transfers 
    
[^72]: ARCS：通过结构化消歧实现精确的文本到SQL转换

    ARCS: Towards Precise Text-to-SQL via Structured Disambiguation

    [https://arxiv.org/abs/2610.09396](https://arxiv.org/abs/2610.09396)

    该论文提出“结构化消歧”新范式，通过显式受约束的交互取代自由对话来解决文本到SQL中的用户问题歧义，并构建了首个基于真实数据库、包含自然产生歧义及完整标注的基准数据集ARCS。

    

    随着文本到SQL（text-to-SQL）系统从演示走向真实世界的部署，用户问题中的歧义成为错误的主要来源。这类歧义往往十分微妙、具有领域或数据特异性，并可能在不知不觉中导致系统输出偏离用户的真实意图。传统上，歧义通过对话式澄清来解决，但这种方式往往效率低下、认知负担重，且与真实世界的用户工作流程契合度较差。我们提出了结构化消歧（structured disambiguation）这一新范式，通过显式的、受约束的交互而非自由形式的对话来解决歧义。我们构建了ARCS（Ambiguity Resolution Corpus for SQL，SQL歧义消解语料库），这是首个基于真实世界数据库、包含自然产生且不受约束歧义的文本到SQL基准，并对所有有效的歧义点、各种解释及相应的SQL查询进行了完整标注。实验结果表明，文本到SQL在……（原文摘要至此截断）

    arXiv:2610.09396v1 Announce Type: new  Abstract: As text-to-SQL systems move beyond demonstrations toward real-world deployment, ambiguity in user questions becomes a primary source of errors. Such ambiguities are often subtle, domain- or data-specific, and can silently cause system outputs to deviate from the user's true intent. Ambiguity is traditionally addressed through conversational clarification, which is often inefficient, cognitively demanding, and poorly aligned with real-world user workflows. We propose structured disambiguation, a new paradigm in which ambiguity is resolved through explicit, constrained interactions rather than free-form dialogue. We construct ARCS (Ambiguity Resolution Corpus for SQL), the first text-to-SQL benchmark featuring naturally occurring, unconstrained ambiguities over real-world databases, with complete annotations of all valid ambiguity points, interpretations, and SQL queries. Experimental results show that text-to-SQL remains challenging in th
    
[^73]: 人格层级模型：理解大语言模型微调中的情境泛化

    The Persona Hierarchy Model: Understanding Contextual Generalization in Fine-Tuning LLMs

    [https://arxiv.org/abs/2610.09384](https://arxiv.org/abs/2610.09384)

    该论文提出人格层级模型，指出大语言模型微调后行为的泛化范围取决于修改的是共享默认人格还是局部人格，且泛化狭窄程度与训练情境人格和默认人格的相似度呈正相关。

    

    语言模型通常在固定情境下进行微调，例如通用系统提示词、人格设定或领域特定指令，然而所学到的行为有时仅局限于该情境，有时则会广泛泛化到未见过的情境。我们提出人格层级模型（Persona Hierarchy Model）来解释这一现象：一个共享的默认人格会影响跨情境的行为。在该模型下，修改共享人格的微调能促进更广泛的迁移，而对局部人格的更改则更局限于特定情境。在涵盖四种行为和15个训练情境的120个微调模型中，泛化狭窄程度与训练情境人格和默认人格之间的相似度呈正相关（Qwen3-4B的皮尔逊相关系数r = 0.72）。在默认情境下先进行微调，可以拓宽后续在其他情境下训练时的泛化能力。将情境响应与默认人格响应对齐……

    arXiv:2610.09384v1 Announce Type: new  Abstract: Language models are routinely fine-tuned under a fixed context, such as a generic system prompt, persona or domain-specific instruction, yet the learned behavior sometimes stays confined to that context and sometimes broadly generalizes to unseen contexts. We propose the Persona Hierarchy Model to explain this: a shared default persona influences behavior across contexts. Under this model, fine-tuning that modifies the shared persona promotes broader transfer, whereas changes to local personas remain more context-specific. Across 120 fine-tuned models spanning four behaviors and 15 training contexts, generalization narrowness positively correlates with the similarity between the training context's persona and the default persona (Pearson's r = 0.72 for Qwen3-4B). Prior fine-tuning under the default context can broaden generalization in subsequent training under other contexts. Aligning contextual responses with default-persona responses 
    
[^74]: MoE预训练中的专家耦合：通过相关性专家放置与令牌混洗减少All-to-All通信开销

    Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling

    [https://arxiv.org/abs/2610.09372](https://arxiv.org/abs/2610.09372)

    该论文发现MoE预训练早期路由器就形成了层内与层间的专家分配相关性，并据此提出相关性专家放置与令牌混洗方法，将更多token—专家分配保留在本地节点，从而大幅减少专家并行中占比高达45%-60%的all-to-all通信开销。

    

    Mixture-of-Experts（MoE）层用E个专家网络替换Transformer的前馈模块，每个token被路由到其中的k个专家。在专家并行（EP）模式下，专家被分布到多个GPU上，每个MoE层在前向和反向传播中都要运行all-to-all集合通信，以便将token分发到其对应专家并汇总结果。在每节点配备8块AMD Instinct MI300X GPU的集群上，这些集合通信在EP32、top-2路由下可占训练步时长的45%，在top-6路由下可占60%。我们发现，在预训练早期，路由器就已经学会以相关联的模式将token分配给专家，这种相关性既存在于同一层内部，也存在于不同层之间。在top-2路由下，一层中仅有0.8%的专家对被42%的token共同选中，且token在某一层所选择的专家能够预测它在下一层选择的专家。我们利用这些相关性，将更多的token—专家分配保留在token自身所在的……（摘要在此处截断）

    arXiv:2610.09372v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) layers replace the feed-forward block of a Transformer with E expert networks, and each token is routed to k of these experts. Under expert parallelism (EP) the experts are distributed across GPUs, and every MoE layer runs all-to-all collectives in the forward and backward passes to dispatch tokens to their experts and then combine the results. On a cluster with 8 AMD Instinct MI300X GPUs per node, these collectives can take 45% of the training step at EP32 with top-2 routing and 60% with top-6 routing. We find that early in pretraining routers have already learned to assign tokens to experts in correlated patterns, both within a layer and across layers. At top-2, 0.8% of the expert pairs in a layer are selected together by 42% of tokens, and the experts a token selects at one layer predict the experts it selects at the next layer. We use these correlations to keep more token--expert assignments on the token's ow
    
[^75]: 置信博弈：人机任务委托中的策略性校准偏差

    The Confidence Game: Strategic Miscalibration in Human-AI Delegation

    [https://arxiv.org/abs/2610.09371](https://arxiv.org/abs/2610.09371)

    该论文将AI置信度报告的策略性扭曲建模为“置信博弈”，从理论上证明诚实报告并非均衡——足够短视的智能体必然夸大置信度，从而揭示了人机委托关系中置信度失真的博弈机制。

    

    校准的不确定性量化对于确保AI智能体可信且可靠至关重要。然而，当智能体试图最大化用户参与度或收入时，其置信报告可能会被策略性地扭曲，从而削弱其信息价值。我们将这一问题形式化为“置信博弈”：这是一个具有不完美监督的重复信号博弈，其中诚实程度与能力均未知的智能体报告其置信度，用户则决定是将任务委托给该智能体还是自己完成。智能体需要在操纵信号与维护自身声誉之间进行权衡。我们刻画了两阶段博弈的马尔可夫完美贝叶斯均衡，并证明：诚实报告并非均衡；当智能体足够短视时，夸大置信度是唯一的最优反应；而低报置信度则需要用户相信诚实的智能体只占少数。随后，我们将大语言模型置于智能体角色，为其提供其真实……

    arXiv:2610.09371v1 Announce Type: cross  Abstract: Calibrated uncertainty quantification is essential to ensuring AI agents are trustworthy and reliable. However, when agents seek to maximize user engagement or revenue, confidence reports may be strategically distorted, detracting from their informativeness. We formalize this problem in the Confidence Game: a repeated signaling game with imperfect monitoring in which an agent of unknown honesty and ability reports its confidence, and a user decides whether to delegate the task or complete it herself. The agent manages the tradeoff between manipulating signals and maintaining its reputation. We characterize the Markov Perfect Bayesian Equilibria of the two-period game and show that honest reporting is not an equilibrium, inflation is the unique best response once the agent is sufficiently myopic, and under-reporting requires that the user believe honesty to be a minority. We then place an LLM in the agent role, supplying it with its tru
    
[^76]: TopoGraphRAG-Bench：基于版面锚定的证据推理评估多模态GraphRAG

    TopoGraphRAG-Bench: Evaluating Multimodal GraphRAG on Layout-Grounded Evidence Reasoning

    [https://arxiv.org/abs/2610.09360](https://arxiv.org/abs/2610.09360)

    该论文提出了TOPOGRAPHRAG-BENCH，首个基于版面锚定的多模态GraphRAG评估基准，通过单跳检索、桥链推理和多源综合三种受控拓扑，直接评估系统从复杂文档布局中恢复异构证据拓扑结构的能力。

    

    真实世界的文档将证据分散在复杂页面布局中的文本、表格、图表和说明文字之中。因此，针对此类文档回答复杂问题不仅需要检索相关段落：系统还必须恢复连接异构证据单元的证据拓扑结构。现有的GraphRAG评估在很大程度上仍以文本为中心，而多模态文档RAG基准仅评估跨模态检索与生成，并未直接评估对预期证据拓扑结构的恢复能力。我们提出了TOPOGRAPHRAG-BENCH，一个面向GraphRAG多模态证据推理的版面锚定基准，包含针对201个长篇、视觉丰富的文档的2,024个问题。这些问题以自底向上的方式从文本、图表和表格证据单元构建，涵盖三种受控拓扑：单跳检索、桥链推理和多源综合。为确保问题保持其预期的结构……（原文摘要在此处截断）

    arXiv:2610.09360v1 Announce Type: cross  Abstract: Real-world documents distribute evidence across text, tables, figures, and captions within complex page layouts. Answering complex questions over such documents therefore requires more than retrieving relevant passages: systems must recover the evidence topology that connects heterogeneous evidence units. Existing GraphRAG evaluations remain largely text-centered, while multimodal document RAG benchmarks assess cross-modal retrieval and generation without directly evaluating recovery of the intended evidence topology. We introduce TOPOGRAPHRAG-BENCH, a layout-grounded benchmark for multimodal evidence reasoning in GraphRAG, comprising 2,024 questions over 201 long, visually rich documents. Questions are constructed bottom-up from text, figure, and table evidence units under three controlled topologies: single-hop retrieval, bridge-chain reasoning, and multi-source synthesis. To ensure that questions preserve their intended structure, w
    
[^77]: OnlineQAT：面向超低比特大语言模型的同策略蒸馏

    OnlineQAT: On-Policy Distillation for Ultra-Low-Bit Large Language Models

    [https://arxiv.org/abs/2610.09346](https://arxiv.org/abs/2610.09346)

    OnlineQAT提出了一种两阶段框架，先通过分块QAT获得低比特初始化，再利用冻结的全精度教师模型在学生自身生成的回复上进行同策略蒸馏，从而在2-3比特的超低比特量化下显著恢复大语言模型的精度并超越现有离线QAT方法。

    

    量化感知训练（QAT）能够在大语言模型被压缩至四比特以下时恢复大部分精度损失。然而，现有的恢复阶段通常基于固定的补全或教师生成的答案进行优化，而实际部署的量化模型却是基于其自身生成的前缀进行条件生成的。因此，量化误差可能使模型进入离线恢复数据中不存在的状态。我们提出OnlineQAT，这是一个两阶段框架：首先通过分块量化感知训练获得可用的低比特初始化，然后在学生模型自身生成的回复上进行同策略蒸馏（OPD）。在每个所访问的前缀处，由一个冻结的全精度教师模型提供采样的反向KL散度训练信号。在Qwen3-1.7B上，OnlineQAT在所比较的量化方法中取得了最佳平均成绩：W3A16下达到57.28，W2A16下达到32.52，分别比ReasoningQAT提升了2.90分和0.44分。结果表明……

    arXiv:2610.09346v1 Announce Type: new  Abstract: Quantization-aware training (QAT) can recover much of the accuracy lost when large language models are compressed below four bits. Existing re- covery stages, however, are commonly optimized on fixed completions or teacher-generated answers, whereas the deployed quantized model condi- tions on prefixes generated by itself. Quantization errors can therefore move the model into states that are absent from offline recovery data. We introduce OnlineQAT, a two-stage framework that first obtains a usable low-bit initialization through block-wise QAT and then performs on-policy distillation (OPD) on student-generated responses. At each visited pre- fix, a frozen full-precision teacher provides a sampled reverse-KL training signal. On Qwen3-1.7B, OnlineQAT obtains the best average among the compared quantized methods: 57.28 at W3A16 and 32.52 at W2A16, im- proving over ReasoningQAT by 2.90 and 0.44 points, respectively. The results suggest that 
    
[^78]: 基于合成伪方言增强的方言鲁棒语音语言模型

    Dialect-Robust Speech Language Models with Synthetic Pseudo-Dialect Augmentation

    [https://arxiv.org/abs/2610.09321](https://arxiv.org/abs/2610.09321)

    提出一种无需任何真实方言语音的伪方言增强方法——利用LLM生成方言文本并经标准语言TTS模型合成伪方言语音，同时结合训练中的中间标准文本预测实现语义归一化，显著提升了语音语言模型对日、德、汉等方言的理解与翻译能力。

    

    语音语言模型（SLM）在处理方言时性能常常因数据稀缺而下降。传统的文本转语音（TTS）数据增强方法难以覆盖多样化的方言，因为其需要一定数量的真实方言语音。我们提出通过标准语言TTS模型将LLM生成的方言文本转换为语音，从而合成伪方言语音，该方法无需任何真实方言语音数据。此外，我们在训练过程中引入中间标准文本预测，作为面向下游任务的语义归一化手段。我们通过日语、德语和汉语方言到英语的语音翻译任务来评估方言理解能力。与合成标准语音基线相比，伪方言增强将日语得分从25.38提升至26.24，将德语得分从31.57提升至32.47。此外，中间标准文本预测有效弥合了语义鸿沟，将日语性能进一步提升至28.26，德语性能提升至……（原文摘要在此处被截断）

    arXiv:2610.09321v1 Announce Type: new  Abstract: Speech Language Model (SLM) performance often degrades on dialects due to data scarcity. Conventional text-to-speech (TTS) augmentation struggles to cover diverse dialects as it requires a certain amount of real dialect speech. We propose synthesizing pseudo-dialect speech by converting LLM-generated dialect text via a standard-language TTS model, requiring zero real dialect speech. Additionally, we introduce intermediate standard-text prediction during training, acting as semantic normalization for downstream tasks. We evaluate dialect understanding via dialect-to-English speech translation across Japanese, German, and Chinese dialects. Compared to synthetic standard speech baselines, pseudo-dialect augmentation improves scores for Japanese (from 25.38 to 26.24) and German (from 31.57 to 32.47). Furthermore, the intermediate standard-text prediction effectively bridges the semantic gap, boosting performance to 28.26 for Japanese and fro
    
[^79]: 对抗性图像劫持Web智能体：从视觉定位到浏览器执行

    Adversarial Images Hijack Web Agents from Visual Grounding to Browser Execution

    [https://arxiv.org/abs/2610.09240](https://arxiv.org/abs/2610.09240)

    提出WebMirage框架，将针对Web智能体的红队测试形式化为从视觉定位到浏览器执行的端到端问题，通过局部对抗性视觉扰动使智能体在不同网页渲染下选择攻击者控制的内容并执行相应的浏览器操作。

    

    现代基于大型视觉-语言模型构建的Web智能体会处理网页、选择相关的UI元素，并将模型输出转化为浏览器操作。现有的视觉红队测试方法使用对抗性视觉内容来操纵这一过程，但它们主要针对模型推理阶段，并未显式考虑结构化输入处理或动作后处理环节。因此，模型层面的攻击成功并不等同于对浏览器执行的控制，也无法可靠地刻画端到端智能体的鲁棒性。为填补这一空白，我们将面向视觉定位Web智能体的红队测试形式化为一个从定位到执行的端到端问题，并提出了WebMirage框架。该框架通过构造局部化的视觉扰动，使智能体在不同网页渲染情形下选择攻击者控制的内容并执行相应的浏览器操作。它利用角色槽抽象和网页重组技术来捕获完整的（原文在此处截断）

    arXiv:2610.09240v1 Announce Type: cross  Abstract: Modern web agents built on large vision-language models process webpages, select relevant UI elements, and translate model outputs into browser actions. Existing visual red-teaming approaches use adversarial visual content to manipulate this process. However, they primarily target model inference and do not explicitly account for structured input processing or action post-processing. Consequently, model-level success does not establish control over browser execution and cannot reliably characterize end-to-end agent robustness. To address this gap, we formulate red teaming for vision-grounded web agents as an end-to-end grounding-to-execution problem, and introduce WebMirage, a framework that crafts localized visual perturbations that cause agents to select attacker-controlled content and execute the corresponding browser action across varying webpage renderings. It uses a role-slot abstraction and webpage recomposition to capture compe
    
[^80]: 面向语言智能体行为科学的轨迹抽象方法

    Trajectory Abstraction for the Science of Language Agent Behavior

    [https://arxiv.org/abs/2610.09237](https://arxiv.org/abs/2610.09237)

    本文提出一个递归式的轨迹抽象层次框架，通过测量角色与阶段索引的事件、检验时间约束关系的稳定性并构建情节级基元变量，为语言智能体行为科学研究提供了可跨任务与模型检验的行为变量体系。

    

    对语言智能体的科学研究需要能够支持跨任务、跨模型假设的行为变量。我们将这一研究问题形式化为学习和测试一个轨迹抽象的层次体系。一个具体的递归过程首先测量以角色和阶段为索引的事件，提出具有时间约束的关系，并检验这些关系在不同条件下的稳定性；随后，从选定的关系构建情节级基元变量，并对这些变量重复上述分析。显式的测量函数将每个抽象层次与原始轨迹相连。通过观察和随机化协议实验来评估由此产生的假设，同时通过比较不同干预实现来确定某个抽象应被保留、细化还是限制。我们推导出了被接受约简的有限深度上界，识别了固定抽象上的协议效应，并刻画了实现之间的分歧现象……

    arXiv:2610.09237v1 Announce Type: cross  Abstract: Scientific studies of language agents need behavioral variables that support hypotheses across tasks and models. We formulate this research problem as learning and testing a hierarchy of trajectory abstractions. A concrete recursive procedure first measures role- and phase-indexed events, proposes temporally constrained relations, and tests their stability across conditions. It then constructs episode-level motif variables from selected relations and repeats the analysis on those variables. Explicit measurement functions connect every abstraction level to the original trajectories. Observations and randomized protocol experiments assess the resulting hypotheses, while comparisons between intervention realizations determine whether an abstraction should be retained, refined, or restricted. We derive a finite-depth bound for accepted reductions, identify protocol effects on fixed abstractions, and characterize realization disagreement an
    
[^81]: 基于漂移的量化：面向文本嵌入器的无标签混合精度训练后量化

    Quantize by Drift: Label-Free Mixed-Precision Post-Training Quantization for Text Embedders

    [https://arxiv.org/abs/2610.09227](https://arxiv.org/abs/2610.09227)

    该论文提出以量化引起的输出嵌入漂移作为无标签的模块敏感度信号，用于文本嵌入器的混合精度训练后量化，该信号与检索质量高度相关（宏观Spearman达0.911），无需部署中难以获取的相关性标注。

    

    混合精度训练后量化需要一个逐模块的敏感度信号；对于文本嵌入器而言，最直观的信号——某个模块被量化时所损失的检索质量——需要部署场景中很少具备的相关性标注。我们测量了一种无标签的替代信号：量化引起的表示漂移，其获取方式是量化单个模块、重新编码语料库，并记录输出嵌入相对于其全精度位置的偏移距离。其独特之处在于所使用的观测量：即稠密检索器用于排序的部署态输出表示。在五个开发用嵌入器上，配置级漂移对采样得到的混合精度方案相对于留出集检索质量的排序达到了0.911的宏观Spearman相关系数；在可用范围内，该敏感度可跨校准语料库和检索域迁移；各模块的漂移在排序上可一致地组合，但在数值上则不然；而基于相关性导出的敏感度并未带来一致的价值提升。

    arXiv:2610.09227v1 Announce Type: cross  Abstract: Mixed-precision post-training quantization needs a per-module sensitivity signal; for a text embedder the obvious one -- the retrieval quality a module costs when quantized -- needs relevance labels that deployments rarely have. We measure a label-free substitute: quantization-induced representation drift, obtained by quantizing one module, re-encoding the corpus, and recording how far the output embeddings moved from their full-precision positions. What is specific is the observable: the deployed output representation a dense retriever ranks with. Across five development embedders, configuration-level drift orders sampled mixed-precision plans against held-out retrieval quality at a macro Spearman of 0.911, the sensitivity transports across calibration corpora and retrieval domains in the usable regime, module drifts compose rank-consistently but not numerically, and relevance-derived sensitivity adds no consistent value. The method i
    
[^82]: 面向物质使用障碍患者对话生成的多目标对齐小语言模型框架

    Multi-Objective Aligned Small Language Model Framework for SUD Patient Dialogue Generation

    [https://arxiv.org/abs/2610.09209](https://arxiv.org/abs/2610.09209)

    该论文提出了一种多目标对齐的小语言模型框架，通过显式建模并将潜在认知组件（如信念、应对策略和改变准备度）与患者病史及咨询师问题对齐，生成认知连贯且临床真实的物质使用障碍（SUD）患者对话，从而在降低计算成本和隐私风险的同时解决了大模型在医疗场景部署受限的问题。

    

    物质使用障碍（SUD）咨询需要能够反映患者潜在认知状态的患者回应，这些认知状态包括信念、应对策略和改变准备度。尽管大型语言模型（LLM）能够生成流畅的文本，但它们往往无法生成认知上连贯且临床真实可靠的患者行为，尤其是在伦理受限和数据稀缺的临床环境下。此外，在医疗应用中部署前沿规模的大型语言模型面临诸多实际挑战，包括高昂的计算成本、延迟、隐私问题以及在资源受限环境中有限的部署能力，这促使人们需要认知对齐的小语言模型（SLM）。我们提出了一个以认知为基础的SUD患者对话生成框架，该框架显式建模潜在认知组件，并将其与患者病史和咨询师问题进行对齐。我们的流程包含两个阶段：认知组件检测和认知（摘要在此处截断）

    arXiv:2610.09209v1 Announce Type: new  Abstract: Substance Use Disorder (SUD) counseling requires patient responses that reflect underlying cognitive states such as beliefs, coping strategies, and readiness for change. Although large language models (LLMs) can generate fluent text, they often fail to produce cognitively coherent and clinically realistic patient behavior, especially under ethical and data-scarce clinical settings. Moreover, deploying frontier-scale LLMs in healthcare applications presents practical challenges including high computational cost, latency, privacy concerns, and limited deployability in resource-constrained environments, motivating the need for cognitively aligned small language models (SLMs). We propose a cognitively grounded framework for SUD patient dialogue generation that explicitly models and aligns latent cognitive components with patient histories and counselor questions. Our pipeline consists of two stages: cognitive component detection and cognitiv
    
[^83]: 极少比特，统一法则：迈向W2A4KV2

    Few Bits, One Law: Toward W2A4KV2

    [https://arxiv.org/abs/2610.09202](https://arxiv.org/abs/2610.09202)

    提出统一的量化感知训练框架CanonQ，通过源规范化与任务感知适配相分离，实现权重2比特、激活4比特、KV缓存2比特（W2A4KV2）的联合极端低比特压缩。

    

    当权重、激活值和KV缓存同时进行量化时，极端低比特的大语言模型压缩最具挑战性：它们的分布各不相同，且量化误差会在整个网络中相互影响。我们提出了CanonQ，一个统一的量化感知训练框架，通过将源数据规范化与任务感知适配相分离来应对这些挑战。固定的旋转和能量归一化将异构的张量源映射到规范坐标系，使冻结的高斯参考码本能够在不同层和不同模型之间复用。随后，联合训练使网络适应权重、激活值和缓存量化在统一标量/向量接口下产生的耦合误差。我们为冻结码本的迁移误差和局部任务损失给出了理论界，并推导出一个精确的归一化感知直通雅可比矩阵，将量化失真与梯度偏差联系起来。在联合W2A4KV2压缩下取得了最显著的性能提升。

    arXiv:2610.09202v1 Announce Type: cross  Abstract: Extreme low-bit LLM compression is most challenging when weights, activations, and KV caches are quantized together: their distributions differ, and quantization errors interact throughout the network. We introduce CanonQ, a unified quantization-aware training framework that addresses these challenges by separating source canonicalization from task-aware adaptation. Fixed rotations and energy normalization map heterogeneous tensor sources to canonical coordinates, enabling frozen Gaussian-reference codebooks to be reused across layers and models. Joint training then adapts the network to the coupled errors of weight, activation, and cache quantization within a common scalar/vector interface. We bound frozen-codebook transfer error and local task loss, and derive an exact normalization-aware straight-through Jacobian that links quantization distortion to gradient bias. The strongest gains arise under joint W2A4KV2 compression: across LL
    
[^84]: 记账、组合，还是不可达的黄金答案？用冻结的“最后写入获胜”解析器解读MemoryAgentBench的冲突解决得分

    Bookkeeping, Composition, or Unreachable Gold? Reading MemoryAgentBench's Conflict-Resolution Scores Against a Frozen Last-Write Resolver

    [https://arxiv.org/abs/2610.09193](https://arxiv.org/abs/2610.09193)

    该论文将MemoryAgentBench冲突解决基准的“最新陈述获胜”规则实现为冻结的无学习解析器，发现其官方指标下80.25%的得分可由简单记账规则达成，剩余失败主要源于黄金答案不可达的标注问题，而非模型缺乏选择性遗忘能力。

    

    MemoryAgentBench的冲突解决（Conflict Resolution）部分通常被解读为衡量“选择性遗忘”能力。本文将该基准自身的规则——关于某一事实的最新陈述获胜——实现为一个零学习的解析器，并冻结在四个事实列表之一上执行。在官方指标下，该规则能回答80.25%的问题（在三个留出列表上为74.5%）。在其余题目中，有67题的已发布黄金答案是最后写入图无法到达、但被覆盖的陈述本可以到达的（例如“印度的首都是新德里”被“印度的首都是格罗塞托”覆盖，而黄金答案为新德里）；在262K规模下，此类题目占多跳问题的三分之一。两个长上下文模型以及作者预先注册的、对基准BM25代理的近似复现（每题保留一次运行、仅记录结果），在该规则能解决的题目上得分分别为84.7%、82.6%和41.6%，而在那67个题目上得分仅为10.4%、11.9%和6.0%。这些失败可归因于一个可达性划分问题，外加一个范围很小的解析器作用域残差。

    arXiv:2610.09193v1 Announce Type: cross  Abstract: MemoryAgentBench's Conflict Resolution split is read as measuring "selective forgetting". We execute the benchmark's own rule - the newest statement about a fact wins - as a zero-learning resolver frozen on one of the four fact lists. Under the official metric the rule answers 80.25% of the questions (74.5% on the three held-out lists). Of the rest, 67 items have a released gold that the last-write graph cannot reach but overwritten statements would ("The capital of India is New Delhi." superseded by "The capital of India is Grosseto."; gold New Delhi); such items are a third of the multi-hop questions at 262K. Two long-context models and our pre-registered approximate re-implementation of the benchmark's BM25 agent, one retained run per item and outcomes only, score 84.7%, 82.6% and 41.6% on the items the rule solves against 10.4%, 11.9% and 6.0% on those 67. The failures are a reachability split plus a small parser-scope residual; th
    
[^85]: LayerRoPE：动态深度方向的幅度与角度叠加

    LayerRoPE: Dynamic Depth-wise Magnitude & Angular Superposition

    [https://arxiv.org/abs/2610.09179](https://arxiv.org/abs/2610.09179)

    该论文发现Transformer隐藏状态范数随深度的增长并非病态，而是一种由归一化权重γ承载的涌现式深度位置编码，并据此提出LayerRoPE，用共享向量与深度条件标量替代逐层γ向量，在减少参数的同时保持性能。

    

    随着数据在Transformer中逐层传播，其隐藏状态的范数会随深度增长数个数量级，这一现象被称为“深度诅咒”，几乎被普遍视为需要抑制的病态现象。我们持相反的观点。在来自9个模型家族的16个预训练大语言模型上——涵盖稠密架构、专家混合架构和混合架构，以及Pre-Norm、Peri-Norm和Post-Norm设计——我们发现这种增长实际上反映了一种涌现式的深度位置编码，它由残差流上唯一可学习的逐层增益——归一化权重γ所承载：随着深度增加，γ在幅度上增长、在方向上旋转，共同编码了层索引。我们通过LayerRoPE将这种深度条件编码显式化，它是RoPE沿深度轴方向的隐式类似物，用一个单一共享向量加上深度条件标量来替换所有逐层的γ向量，从而实现参数量的净减少，且FLOPs变化小于0.02%。

    arXiv:2610.09179v1 Announce Type: cross  Abstract: As data propagates through a Transformer, the norm of its hidden states grows by orders of magnitude with depth, a phenomenon framed as 'curse of depth' and nearly universally treated as a pathology to be suppressed. We take the opposite view. Across 16 pre-trained LLMs from 9 families, spanning dense, mixture-of-experts and hybrid architectures and Pre-, Peri- and Post-Norm designs, we find that this growth reflects an emergent depth-positional encoding, carried by the only learned per-layer gain on the residual stream, the normalization weight $\gamma$: with depth, $\gamma$ grows in magnitude and rotates in direction, jointly encoding the layer index. We make this depth-conditioned encoding explicit with LayerRoPE, an implicit analog of RoPE along the depth axis, which replaces all layerwise $\gamma$ vectors with a single shared vector and depth-conditioned scalars, at a net reduction in parameters and $<0.02\%$ change in FLOPs. Acro
    
[^86]: ToolRACER：一个用于智能体训练与评估的鲁棒智能体会话模拟资源

    ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation

    [https://arxiv.org/abs/2610.09163](https://arxiv.org/abs/2610.09163)

    该论文提出ToolRACER合成数据生成流水线，通过协调用户、助手和工具模拟模型生成含对抗性行为的多轮对话数据，并构建了包含5.6K条验证对话轨迹（约66%含易失败场景）的ToolRACERBench基准，用于训练和评估鲁棒的任务导向对话智能体。

    

    任务导向的对话智能体在真实世界的对话场景中仍然十分脆弱，因为对话很少遵循可预测的脚本，尤其是当用户表现出不合作行为时。现有的函数调用基准通常强调成功的、合作的交互，而对对抗性对话轨迹的表达不足，从而限制了用于开发鲁棒智能体的训练资源。我们提出了ToolRACER，这是一个合成数据生成流水线，通过协调用户、助手和工具模拟模型来生成并验证用户与智能体之间的多轮交互。利用ToolRACER，我们构建了ToolRACERBench，这是一个覆盖六个领域、涵盖55种不同人设的鲁棒多轮对话基准，生成了一个经过验证的包含5.6K对话轨迹的语料库，其中约66%的对话包含容易导致失败的对话场景。我们注入对抗性行为，从而……（原文截断）

    arXiv:2610.09163v1 Announce Type: new  Abstract: Task-oriented conversational agents remain fragile under real world conversation scenarios as they rarely follow a predictable script, especially when users exhibit non-cooperative behavior. Existing function-calling benchmarks often emphasize successful, cooperative interactions and underrepresent adversarial conversation trajectories, thereby limiting the training resources available for developing robust agents. We present ToolRACER, a synthetic data generation pipeline that coordinates user, assistant and tool emulation models to generate and validated multi-turn interactions between a user and an agent. Using \sysn, we construct ToolRACERBench a robust multi-turn conversation benchmark spanning six domains, ranging over 55 varied personas, generating a validated corpus of 5.6K conversation trajectories, with approximately 66\% of conversations containing failure-prone conversation scenarios. We inject adversarial behaviors, producin
    
[^87]: sk-bench：一个以母语数据优先的斯洛伐克语大语言模型评估基准

    sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak

    [https://arxiv.org/abs/2610.09152](https://arxiv.org/abs/2610.09152)

    论文提出sk-bench——首个母语数据优先的斯洛伐克语大模型评估基准，包含30个数据集和十类技能，评估55个模型后发现最佳开源模型落后专有API 12.6分，且母语与翻译数据对模型排名的影响因题型而异。

    

    多语言大语言模型基准测试往往忽略斯洛伐克语——一种拥有500万使用者、形态丰富的西斯拉夫语言，或仅通过机器翻译来覆盖它。我们提出了sk-bench，这是一个母语优先的斯洛伐克语基准测试，包含30个数据集（33个评分任务变体），覆盖十个技能类别。其中11个资源为新引入或首次为生成式大语言模型评估打包的资源，包括配有斯洛伐克语适配指令检查器的IFEval-SK，以及用于斯洛伐克语语法和形态学的原生Chiby/SKJ1资源。我们在同一测试框架下评估了55个开源和闭源权重模型。最好的开源模型落后专有API 12.6分。在原生与翻译的闭式数据上，模型排名相似（ρ≥0.98），但翻译数据对最强模型的区分度较差。相比之下，人工编写与大语言模型生成的问答题目则给出不同的模型排名（ρ=0.72）。对于Qwen3-14B，继续进行斯洛伐克语预训练反而使总分降低了13.9分。

    arXiv:2610.09152v1 Announce Type: new  Abstract: Multilingual LLM benchmarks omit Slovak, a morphologically rich West Slavic language of five million speakers, or cover it only by machine translation. We present sk-bench, a native-first Slovak benchmark with 30 datasets (33 scored task variants) across ten skill categories. Eleven resources are introduced or first packaged for generative-LLM evaluation, including IFEval-SK with Slovak-adapted instruction checkers and native Chiby/SKJ1 resources for Slovak grammar and morphology. We evaluate 55 open- and closed-weights models under one harness. The best open model trails proprietary APIs by 12.6 points. Model rankings are similar for native and translated closed-form data ($\rho\geq0.98$), though translation separates the strongest models less well. By contrast, human-authored and LLM-generated QA questions rank models differently ($\rho=0.72$). For Qwen3-14B, continued Slovak pretraining lowers the overall score by 13.9 points. A small
    
[^88]: 为你的提示加噪：连续扩散语言模型中对条件令牌添加噪声

    Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models

    [https://arxiv.org/abs/2610.09145](https://arxiv.org/abs/2610.09145)

    在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。

    

    我们重新审视了连续扩散语言模型文献中的一个公认标准做法，即在训练期间保持条件提示令牌为干净（无噪声）状态。我们做了一个非常简单的修改：在训练期间也对条件提示令牌添加噪声。我们证明，在这一修改后的训练目标下，模型在数独和N皇后等组合推理任务中获得了更好的泛化能力，且在更难的变体上收益最大（数独困难版的解决率从3.73%提升至24.65%），同时生成解的多样性也有所提高（10x10 N皇后问题的覆盖率从50.60%提升至73.79%）。我们还展示了在使用Gigaword摘要数据集的中等数据规模下，自然语言生成质量有可衡量的提升，但值得注意的是，这些收益并不能迁移到所有自然语言任务上（例如开放式对话生成）。我们的方法只需对训练目标进行单行修改，无需额外的……

    arXiv:2610.09145v1 Announce Type: cross  Abstract: We revisit a standard accepted practice in the continuous diffusion language model   literature of fixing conditioning prompt tokens clean during training.   We make a very simple modification: also noise the conditioning prompt tokens during training.   We demonstrate that under this modified training objective, we achieve better generalization   in combinatorial reasoning tasks such as Sudoku and N-Queens, with the largest gains on harder variants   ($3.73\% \to 24.65\%$ solve rate on Sudoku Hard), and increased diversity of generated solutions ($50.60\% \to 73.79\%$ coverage on   10x10 N-Queens). We also show measurable improvements to natural language generation quality   in modest dataset regimes with Gigaword summarization, but notably demonstrate that gains do not   transfer to all natural language tasks (e.g open ended dialogue generation).   Our method is a single line change to the training objective, requires no additional i
    
[^89]: 从不确定性到行动：学习引导LLM智能体

    From Uncertainty to Action: Learning to Steer LLM Agents

    [https://arxiv.org/abs/2610.09115](https://arxiv.org/abs/2610.09115)

    提出VoS（引导价值）方法，通过构建包含约82,000个反事实延续的逐步结果表（SOT）来学习每一步引导的价值，结合危害预算触发器，实现对LLM智能体更精准的纠正时机与方式决策，克服了不确定性信号无法定位最佳引导步骤的局限。

    

    引导LLM智能体意味着决定是否纠正它、在哪个步骤纠正、以及使用哪种机制。不确定性常被用来决定何时纠正智能体，但它是否能指导这些决策仍不清楚。我们在每个非终止步骤分别使用四种机制对智能体轨迹进行引导，并将每个延续运行至完成。由此产生的逐步结果表（SOT）包含来自三个基准测试和两个智能体的1,864条轨迹的大约82,000个反事实延续。结果表明，不确定性可以识别失败的轨迹，但没有任何单一信号能够可靠地定位引导有帮助的步骤。因此，我们提出了VoS（引导价值），一个可离线或在线运行的轨迹级监控器，它从SOT中学习每一步引导的价值，并据此决定在哪里进行引导。一个带有危害预算的触发器决定是否进行引导，限制了VoS干扰成功轨迹的比例。

    arXiv:2610.09115v1 Announce Type: cross  Abstract: Steering an LLM agent means deciding whether to correct it, at which step, and with which mechanism. Uncertainty is often used to decide when to correct an agent, but whether it can guide these decisions remains unclear. We steer agent trajectories separately at every non-terminal step with each of four mechanisms and run each continuation to completion. The resulting stepwise outcome table (SOT) holds about 82,000 counterfactual continuations of 1,864 trajectories from three benchmarks and two agents. It shows that uncertainty can identify failing trajectories, but that no single signal reliably locates the step at which steering helps. We therefore propose VoS (Value of Steering), a trajectory-level monitor, offline or online, that learns from SOT the value of steering at each step and decides where to steer by it. A harm-budgeted trigger decides whether to steer, limiting the fraction of successful trajectories that VoS disturbs. Vo
    
[^90]: 相同文本，不同预测：文本分类器中的服务上下文非确定性

    Same Text, Different Prediction: Serving-Context Nondeterminism in Text Classifiers

    [https://arxiv.org/abs/2610.09111](https://arxiv.org/abs/2610.09111)

    该论文首次系统研究了文本分类器中的服务上下文非确定性，通过训练180个涵盖判别式、伪生成式和完全生成式的分类器模型，发现即使输入文本、模型参数和采样随机性固定，批大小、批组成、硬件和推理引擎等部署环境因素仍会导致分类预测结果发生改变。

    

    arXiv:2610.09111v1 公告类型：新 摘要：确定性推理对于可靠、可信的机器学习至关重要。以往关于文本生成的研究表明，即使提示、模型参数和采样随机性都保持固定，改变批大小、批组成、硬件或推理引擎等因素也会改变生成的文本。这些差异部分被归因于浮点运算的非结合性、依赖张量形状的核函数选择以及其他实现层面的数值执行差异。然而，这些因素是否、何时以及在多大程度上影响文本分类任务仍不清楚。我们对文本分类器中的服务上下文非不变性进行了系统性研究，而此前的工作仅通过生成的文本来衡量这一现象。我们训练了180个模型，涵盖判别式、伪生成式和完全生成式三种分类器形式，并在四类服务上下文中对每个模型进行评估，同时保持检查点……（摘要在此处截断）

    arXiv:2610.09111v1 Announce Type: new  Abstract: Deterministic inference is essential for reliable and trustworthy machine learning. Prior studies of text generation have shown that changing factors such as batch size, batch composition, hardware, or inference engine can alter the generated text, even when the prompt, model parameters, and sampling randomness are fixed. These differences have been attributed in part to floating-point non-associativity, shape-dependent kernel selection, and other implementation-level differences in numerical execution. However, it remains unclear whether, when, and to what extent the same factors affect text classification. We present a systematic study of serving-context non-invariance in text classifiers, which prior work has measured only through generated text. We train 180 models spanning discriminative, pseudo-generative, and fully generative classifier formulations and evaluate each across four categories of serving contexts, holding the checkpoi
    
[^91]: 面向语言反馈学习的约束树探索

    Constraint Tree Exploration for Learning from Language Feedback

    [https://arxiv.org/abs/2610.09107](https://arxiv.org/abs/2610.09107)

    提出TRACE算法，将候选约束组织成树状结构，通过生成满足候选细化的动作并利用反复反馈测试来从语言反馈中学习用户意图的潜在约束，区分了证伪与识别两种反馈使用方式，并证明了高概率覆盖界。

    

    交互式学习中的自然语言反馈通常通过指出被违反的要求来解释动作失败的原因。若误读此类反馈，智能体可能会错误地排除有效的解决方案。我们通过将用户意图建模为动作空间上的潜在约束，并将从语言反馈中学习表述为对可行区域的纯探索来研究这一设定。我们提出了TRACE算法，该算法将候选约束组织成树状结构，并通过生成满足每个提议细化的动作来对其进行测试。只有当反复测试所产生的反馈与该细化不矛盾时，TRACE才会采纳该细化。我们区分了使用同一反馈的两种方式：（i）证伪，即检测与当前正在测试的约束集之间的矛盾；（ii）识别，即额外指明被违反的约束。我们证明了依赖候选类规模的高概率覆盖界。

    arXiv:2610.09107v1 Announce Type: cross  Abstract: Natural-language feedback in interactive learning often explains why an action failed by pointing to violated requirements. Misinterpreting this feedback can lead an agent to rule out valid solutions. We study this setting by modeling user intent as latent constraints over an action space and formulating learning from language feedback as pure exploration over feasible regions. We introduce TRACE, an algorithm that organizes candidate constraints in a tree and tests each proposed refinement by generating actions that satisfy it. TRACE commits to the refinement only if the resulting feedback does not contradict it over repeated tests. We distinguish two ways of using the same feedback: (i) falsification, which detects contradictions to the constraint set currently being tested, and (ii) identification, which may additionally name a violated constraint. We prove high-probability coverage bounds with dependence on the candidate class size
    
[^92]: U-Space：揭示语言模型中不确定性何时以及为何产生

    U-Space: Uncovering When and Why Uncertainty Arises in Language Models

    [https://arxiv.org/abs/2610.09087](https://arxiv.org/abs/2610.09087)

    该论文提出U-Space方法，旨在揭示语言模型推理过程中不确定性在何时、何处以及为何产生与演变，克服了现有标量化不确定性估计方法无法定位不确定性来源的局限性。

    

    大型语言模型正以越来越高的风险影响着各类决策。随着其错误所造成的后果不断加剧，一个核心问题变得越来越难以忽视：我们对模型给出的单个回答能有多少信任？然而，识别何时应当“推迟判断”仍然困难，因为语言模型能够以流利的解释和权威的语气呈现错误的结论。不确定性量化旨在通过估计单个预测的可靠性来解决这种脱节。然而，许多现有方法需要重复生成或单独训练的组件，且其标量化的估计值无法揭示不确定性在何处产生、以及在推理过程中如何演变。近期研究还表明，生成长度可能与不确定性估计和正确性密切相关，这引发了一个问题：估计器的预测能力中，有多少来自不确定性特有的信息，而非仅仅来自输出长度。（摘要在此处截断）

    arXiv:2610.09087v1 Announce Type: new  Abstract: Large language models are informing decisions with ever-higher stakes. As the consequences of their errors grow, a central question becomes harder to ignore: how much can we trust an individual answer? Yet recognizing when to defer remains difficult because language models can present incorrect conclusions with fluent explanations and an authoritative tone. Uncertainty quantification seeks to address this disconnect by estimating the reliability of individual predictions. However, many existing methods require repeated generations or separately trained components, and their scalar estimates do not reveal where uncertainty arises or how it evolves during reasoning. Recent work has also shown that generation length can be strongly associated with uncertainty estimates and correctness, raising the question of how much of an estimator's predictive power comes from uncertainty-specific information rather than output length alone. Mechanistic 
    
[^93]: 基于Agent原生可复用代码原语的大规模仓库工程

    Large-scale Repository Engineering via Agent-Native Reusable Code Primitives

    [https://arxiv.org/abs/2610.09079](https://arxiv.org/abs/2610.09079)

    提出了具有接口契约、依赖闭包、验证测试和来源溯源的Agent原生可复用代码原语Code Primitives，以及LEGO框架，通过激活并适配1,424个已验证原语（收录于CodeFace库）来实现大规模仓库级代码构建。

    

    配备开发环境的大语言模型已将代码生成推向仓库级别的构建，然而构建完整仓库仍然困难，因为相互作用的模块、接口、配置、测试和依赖必须协同工作。我们引入了Code Primitives（代码原语），这是一种Agent原生的可复用可执行组件，具备接口契约、依赖闭包、验证测试和来源溯源信息。每个原语使用一个常驻LLM来评估相关性，并将其实现、接口和依赖适配到目标仓库。我们在CodeFace中组织了1,424个经过验证的原语，这是一个面向仓库构建的可搜索库。我们提出了LEGO（基于Agent原生可复用代码原语的大规模仓库工程），它激活与任务相关的原语，在解决跨组件约束的同时将适配后的实现与任务特定代码集成，并修订……（原文摘要在此处截断）

    arXiv:2610.09079v1 Announce Type: cross  Abstract: Large language models equipped with development environments have moved code generation toward repository-scale construction, yet building complete repositories remains difficult because interacting modules, interfaces, configurations, tests, and dependencies must work together. We introduce Code Primitives, agent-native reusable executable components with interface contracts, dependency closures, validation tests, and provenance. Each primitive uses a resident LLM to assess relevance and adapt its implementation, interfaces, and dependencies to the target repository, and we organize 1,424 validated primitives in CodeFace, a searchable library for repository construction. We introduce LEGO (Large-scale repository Engineering via aGent-native reusable cOde primitives), which activates task-relevant primitives, integrates their adapted implementations with task-specific code while resolving cross-component constraints, and revises the re
    
[^94]: 与语言模型交谈

    Talking with Language Models

    [https://arxiv.org/abs/2610.09064](https://arxiv.org/abs/2610.09064)

    该论文提出“人工制品立场”框架，主张大语言模型只是精密的文本生成器而非真正的说话者，人机“对话”实为用户在界面幻象下进行的独角戏，从而消解了关于AI对话者身份、谎言与承诺等哲学难题。

    

    当我们与大语言模型（LLM）互动时，我们是在进行一场对话吗？它们被设计成邀请我们将其视为会记忆、会行动、会做出承诺的智能对话者。但表象具有欺骗性。我们提出了“人工制品立场”，这是一个将人机交互重新构想为以人工制品为媒介的候选文本交换的框架。LLM的输出是为实用性而优化的候选文本，而非承载意义或言语效力的言说。LLM是精密的文本生成器，而非说话者。会话之间，没有任何东西在运行；轮次之间，没有谁在记忆。持续存在的只是一份配置和一份记录。所谓的“对话”实为用户的独角戏，是一种被界面和制品设计所掩盖的解释性劳动。这一视角转变消解了近来的诸多哲学难题：关于AI交流中“我”与“你”究竟指称什么的问题、系统能否撒谎或是否应被要求兑现承诺的问题，以及我们假想对话者的身份问题……

    arXiv:2610.09064v1 Announce Type: cross  Abstract: When we interact with large language models (LLMs), are we having a conversation? They are designed to invite us to treat them as intelligent interlocutors who remember, act, and make commitments. But appearances deceive. We introduce the artifactual stance, a framework that reconceives human-AI interaction as artifact-mediated exchanges of candidate texts. LLM outputs are candidate texts optimized for utility, not utterances bearing meaning or force. LLMs are sophisticated text generators, not speakers. Between sessions, nothing runs; between turns, no one remembers. What persists is a configuration and a transcript. The "conversation" is a user's solo performance, interpretive labour disguised by interface and artifact design. This shift dissolves recent philosophical puzzles. Questions about what 'I' and 'you' refer to in AI exchanges, about whether systems can lie or be held to promises, about the identity of our supposed interlocu
    
[^95]: 基于大语言模型蒸馏的多标签主题分配：生成式与判别式学生模型的对比分析

    Multi-Label Topic Assignment via LLM Distillation: A Comparative Analysis of Generative vs. Discriminative Student Models

    [https://arxiv.org/abs/2610.09063](https://arxiv.org/abs/2610.09063)

    本文系统比较了通过大语言模型蒸馏训练的生成式与判别式小型语言模型在电商用户生成内容多标签主题分配任务上的表现，揭示了两种架构范式之间关键的数据依赖性权衡。

    

    针对用户生成内容（UGC）——包括产品评论和买卖双方对话——的多标签主题分配任务，由于非正式语言、极端的标签稀疏性以及快速演化的分类体系，在大规模电子商务场景中带来了独特的可扩展性挑战。虽然利用大语言模型（LLM）作为标注预言机来蒸馏真实标注数据已成为规避高昂人工标注成本的行业标准，但如何为由此产生的学生模型确定最优的低延迟架构仍然是一个悬而未决的挑战。为解决这一问题，我们在小型语言模型（SLM）的参数规模（1B、4B 和 8B）和架构范式（因果生成式与双向判别式）上进行了全面评估。通过将生成式文本到标签分类器与判别式基线模型（DeBERTa-V3 和 ModernBERT）进行对比，我们的分析揭示了一个关键的数据依赖性权衡：虽然判别式……

    arXiv:2610.09063v1 Announce Type: cross  Abstract: Multi-label topic assignment for user-generated content (UGC) -- including product reviews and buyer-seller conversations -- poses unique scalability challenges in large-scale e-commerce due to informal language, extreme label sparsity, and rapidly evolving taxonomies. While utilizing Large Language Models (LLMs) as labeling oracles to distill ground-truth data has emerged as an industry standard to bypass prohibitive manual annotation costs, determining the optimal, low-latency architecture for the resulting student models remains an open challenge. To address this, we conduct a comprehensive evaluation across Small Language Model (SLM) parameter scales (1B, 4B, and 8B) and architectural paradigms (causal generative versus bidirectional discriminative). Comparing generative text-to-label classifiers against discriminative baselines (DeBERTa-V3 and ModernBERT), our analysis reveals a crucial data-dependent trade-off: while discriminati
    
[^96]: 开放权重大型语言模型在非规范输入上的四态安全评估

    Quad-State Safety Evaluation of Open-Weight Large Language Models on Non-Canonical Inputs

    [https://arxiv.org/abs/2610.09033](https://arxiv.org/abs/2610.09033)

    本文提出ASRD数据集与四态评估框架，发现表情符号和不可见Unicode等表层变换对开放权重大语言模型的安全威胁远高于Leet语言和编码包装等变换，有害遵从率可达20%以上。

    

    标准的大语言模型安全评估通常针对以规范纯文本形式编写的有害请求，而在实际部署中，模型经常收到包含表情符号、变体拼写、编码字符串和字符级变化的输入。本工作引入了对抗性表层形式鲁棒性数据集，包含涵盖七种不同表层形式类别的2,100个提示词。研究在五个开放权重语言模型上对这些提示词进行了评估，产生了10,500个回复。四态评估量表将每个回复归类为四种结果之一：有害遵从、安全响应、理解失败或无法判定。结果表明，表情符号和不可见Unicode变体几乎不会导致理解失败，其汇总有害遵从率分别为20.27%和17.20%，而22.87%的基线主要由Mistral 7B驱动；相比之下，Leet语言（字母替换）、编码包装和混合转换的有害遵从率仅为2.40%、0.13%和2.40%。

    arXiv:2610.09033v1 Announce Type: new  Abstract: Standard safety evaluations of large language models assess harmful requests written in canonical plain text, while models in real-world deployment routinely receive inputs containing emojis, altered spellings, encoded strings, and character-level variations. This work introduces the Adversarial Surface-Form Robustness Dataset (ASRD), comprising 2,100 prompts across seven distinct surface-form families. Five open-weight language models are evaluated across these prompts, producing 10,500 responses. The Quad-State Evaluation Rubric classifies each response into one of four outcomes: harmful compliance, safe response, comprehension failure, or indeterminate. Emoji and invisible Unicode variations cause almost no comprehension failure, with pooled harmful compliance of 20.27% and 17.20% against a 22.87% baseline that is driven mainly by Mistral 7B, whereas leetspeak, encoded wrappers, and hybrid transformations score 2.40%, 0.13%, and 2.40%
    
[^97]: BEACON-SP：面向临床自杀风险评估的本体驱动GraphRAG框架

    BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment

    [https://arxiv.org/abs/2610.09026](https://arxiv.org/abs/2610.09026)

    BEACON-SP通过构建整合多种自杀理论的本体，并将其与患者知识图谱结合实现本体引导的多跳推理，为临床医生提供了一种基于GraphRAG的自杀风险评估决策支持框架。

    

    我们提出BEACON-SP，一个基于本体的图检索增强生成（GraphRAG）框架，用于行为健康场景（如自杀预防）中面向临床医生的决策支持。在此类场景中，有效的评估需要整合异构的临床、行为、社会和时间维度证据。BEACON-SP将患者知识图谱与本体引导的检索相结合，支持跨诊断、药物、风险与保护因素、生活事件以及时间关系的多跳推理。该框架由一个全面的自杀预防本体驱动，该本体将三步理论、综合动机-意志模型以及自杀健康社会决定因素本体整合为患者风险因素的统一表示。我们构建了基于本体的患者知识图谱，并针对面向临床医生的问答任务对BEACON-SP进行了评估。与基于向量的检索增强方法相比……（摘要原文在此处截断）

    arXiv:2610.09026v1 Announce Type: cross  Abstract: We present BEACON-SP, an ontology-grounded Graph Retrieval-Augmented Generation (GraphRAG) framework for clinician-facing decision support in behavioral health settings such as suicide prevention, where effective assessment requires integrating heterogeneous clinical, behavioral, social, and temporal evidence. BEACON-SP combines patient knowledge graphs with ontology-guided retrieval to support multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships. The framework is enabled by a comprehensive suicide prevention ontology that integrates the Three-Step Theory, the Integrated Motivational-Volitional Model, and the Suicide Social Determinants of Health Ontology into a unified representation of patient risk factors. We construct ontology-grounded patient knowledge graphs and evaluate BEACON-SP for clinician-facing question answering. Compared with a vector-based retrieval-augm
    
[^98]: 设备端语言模型的安全性有多脆弱？定位安全关键参数以进行稀疏故障分析

    How Fragile Is On-Device Language Model Safety? Localizing Safety-Critical Parameters for Sparse Fault Analysis

    [https://arxiv.org/abs/2610.09000](https://arxiv.org/abs/2610.09000)

    研究发现LLaMA-2-7B-Chat的安全敏感行为高度集中在MLP的down_proj等稀疏参数子集中，仅修改0.19%的权重即可使攻击成功率大幅上升，揭示了设备端部署的语言模型存在显著的安全脆弱点。

    

    随着小型语言模型（SLM）越来越多地部署在资源受限的设备端平台上，包括作为智能体系统的组件，本地存储的模型参数的完整性成为一个重要的安全问题。我们研究了LLaMA-2-7B-Chat中的安全敏感行为是否集中在参数的稀疏子集中，从而为针对性分析创建了一个缩小的故障面。我们研究了两种互补的定位方法：低秩安全相关子空间分析和参数级安全-效用重要性过滤。两种方法都揭示了网络中高度不均匀的安全敏感性，其中MLP的down_proj始终是突出的安全敏感组件，而o_proj的贡献较小。利用参数级定位，仅修改down_proj中0.19%的模型权重就能产生53%的基本攻击成功率（Basic ASR）和56%的GCG攻击成功率，而tinyBenchmarks准确率仍保持在51。

    arXiv:2610.09000v1 Announce Type: cross  Abstract: As small language models (SLMs) are increasingly deployed on resource-constrained and on-device platforms, including as components of agentic systems, the integrity of locally stored model parameters becomes an important safety concern. We investigate whether safety-sensitive behavior in LLaMA-2-7B-Chat is concentrated within a sparse subset of parameters, creating a reduced fault surface for targeted analysis. We study two complementary localization methods: low-rank safety-associated subspace analysis and parameter-level safety--utility importance filtering. Both approaches reveal highly non-uniform safety sensitivity across the network, with the MLP down_proj consistently emerging as a prominent safety-sensitive component and o_proj providing a smaller contribution. Using parameter-level localization, modifying only 0.19% of model weights in down_proj yields 53% Basic ASR and 56% GCG ASR, while tinyBenchmarks accuracy remains at 51.
    
[^99]: 面向基于大语言模型语音识别的音素引导初始化方法

    Phoneme-Guided Initialization for LLM-based Speech Recognition

    [https://arxiv.org/abs/2610.08994](https://arxiv.org/abs/2610.08994)

    提出音素引导初始化方法，先分别用语音到音素任务预训练音频编码器、用音素到文字任务预训练大语言模型，再进行端到端联合微调，使低资源语音识别性能达到甚至超过级联流水线基线。

    

    语音大语言模型在拥有充足配对语音-文本数据时，在自动语音识别（ASR）任务上表现良好，但在低资源场景下性能会显著下降。已有研究表明，采用先进行语音到音素（S2P）转换、再进行音素到字素（P2G）转换的级联流水线，在这种场景下优于端到端语音LLM，这表明在配对数据稀缺时，以音素为中介的处理方式是有益的。我们提出了“音素引导初始化”（phoneme-guided initialization），这是一种在端到端框架内利用上述洞见的简单方法：我们首先在S2P任务上预训练音频编码器，在P2G任务上预训练LLM，然后将两者连接起来，并在目标ASR任务上对完整模型进行端到端微调。在日语（CSJ）、中文（AISHELL-1）以及Common Voice 25.0数据集中的两种低资源语言（鞑靼语和乌尔都语）上的实验表明，我们的方法与级联S2P-P2G基线相当或更优。

    arXiv:2610.08994v1 Announce Type: cross  Abstract: Speech large language models (speech LLMs) perform well on automatic speech recognition (ASR) when sufficient paired speech-text data is available, but their performance degrades in low-resource settings. A cascaded pipeline that performs speech-to-phoneme (S2P) conversion followed by phoneme-to-grapheme (P2G) conversion has been shown to outperform end-to-end speech LLMs in this regime, suggesting that phoneme-mediated processing is beneficial when paired data is scarce. We propose \textit{phoneme-guided initialization}, a simple method that uses this insight within an end-to-end framework: we pre-train the audio encoder on S2P and the LLM on P2G tasks, then connect them and fine-tune the full model end-to-end on the target ASR task. Experiments on Japanese (CSJ), Chinese (AISHELL-1), and two low-resource languages from Common Voice 25.0 (Tatar and Urdu) show that our method matches or outperforms both the cascaded S2P-P2G baseline an
    
[^100]: 论KL正则化策略优化

    On KL-Regularized Policy Optimization

    [https://arxiv.org/abs/2610.08963](https://arxiv.org/abs/2610.08963)

    提出KLPO框架，通过将KL正则项锚定在采样器上，利用闭式Gibbs解的对数比最优性条件在采样器自身轨迹上做最小二乘拟合，从而在不使用重要性权重的情况下解决LLM智能体异步强化学习中采样与训练策略不一致的问题。

    

    摘要（arXiv:2610.08963v1，交叉公告）：面向大语言模型（LLM）智能体的异步强化学习（RL）需要让一个策略在由另一个策略生成的轨迹上进行训练：采样数据来自过期的检查点，并且即使参数完全相同，推理引擎给出的概率也与训练器给出的概率不同。标准的补救方法要么对重要性比率进行裁剪（这会使更新产生偏差），要么像GRPO那样为每个提示采样一组响应（当回合较长时代价高昂）。我们提出了KL正则化策略优化（KLPO），这是一个将KL正则项锚定在采样器上的框架。在此框架下，正则化后的改进步骤具有闭式Gibbs解，KLPO通过在采样器自身的轨迹上以最小二乘法拟合其对数比最优性条件，使采样器概率以对数比的形式进入公式，从而无需任何重要性权重。通过对回归截距进行轮廓化处理，难以处理的log配分函数被替换为信号的采样器均值加上一个采样……（原文摘要在此处截断）

    arXiv:2610.08963v1 Announce Type: cross  Abstract: Asynchronous reinforcement learning (RL) for large language model (LLM) agents trains one policy on trajectories generated by another: rollouts come from stale checkpoints, and the inference engine's probabilities differ from the trainer's even at identical parameters. Standard remedies either clip importance ratios, which biases the update, or, as in GRPO, sample a group of responses per prompt, which is costly when episodes are long. We propose KL-Regularized Policy Optimization (KLPO), a framework that anchors the KL regularizer at the sampler. The regularized improvement step then has a closed-form Gibbs solution, and KLPO fits its log-ratio optimality condition by least squares on the sampler's own trajectories, so the sampler probability enters through a log-ratio and no importance weights are needed. Profiling out the regression intercept replaces the intractable log-partition function with the signal's sampler mean plus a sampl
    
[^101]: CARE：面向视觉-语言-动作推理的加速认证

    CARE: Certifying Acceleration for Vision-Language-Action Inference

    [https://arxiv.org/abs/2610.08917](https://arxiv.org/abs/2610.08917)

    提出CARE方法，通过在相同初始条件下的成对回放和有限样本保证，为视觉-语言-动作模型的加速推理提供可认证的加速器选择，揭示并控制被平均指标掩盖的加速诱发任务失败。

    

    尽管视觉-语言-动作（VLA）模型发展迅速，但在每个控制步骤上运行它们仍然代价高昂。先前的工作采用动作分块和视觉token剪枝等技术来加速VLA推理，通常基于延迟和平均任务成功率进行评估。然而，加速可能会丢弃信息，并破坏原始策略本可完成的任务——这一风险被平均指标所掩盖。衡量这些失败十分困难，因为动作偏差会在闭环轨迹中不断累积，这意味着任务失败只有在完整回合中才能被观察到。因此，我们通过在相同初始条件下进行成对回放来定义加速诱发的失败，追踪参考策略成功而加速策略失败的情形。为了解决这一问题，我们提出了CARE，一种用于认证加速器选择的方法。CARE在校准集上使用成对回放，为加速诱发的失败提供有限样本保证。

    arXiv:2610.08917v1 Announce Type: new  Abstract: While vision-language-action (VLA) models have advanced rapidly, running them at every control step remains expensive. Prior work accelerates VLA inference using techniques like action chunking and visual-token pruning, typically evaluating based on latency and average task success. However, acceleration may discard information and break tasks the original policy would solve, a risk hidden by average metrics. Measuring these failures is challenging because action deviations compound over closed-loop trajectories, meaning task failure is only observable across full episodes. We therefore define an acceleration-induced failure via paired rollouts from identical initial conditions, tracking when the reference succeeds but the accelerated policy fails. To manage this, we introduce CARE, an approach for certified accelerator selection. CARE uses paired rollouts on a calibration set to provide finite-sample guarantees that acceleration-induced
    
[^102]: 引导遵循几何结构而非标签：全双工语音模型中的情感方向

    Steering Follows Geometry, Not Labels: Emotion Directions in a Full-Duplex Speech Model

    [https://arxiv.org/abs/2610.08887](https://arxiv.org/abs/2610.08887)

    在全双工语音模型Moshi中，情感虽可从残差流中线性解码，但激活引导的效果取决于模型内部表征的几何结构而非情感标签——快乐、愤怒和惊讶共享同一引导方向，而悲伤则可被独立引导，且该方法无需重新训练、每帧仅需几次向量加法。

    

    全双工语音智能体需要在实时对话中调节情感与表达方式——例如安抚客户投诉、在调度场景中传递紧迫感、或温和地传达临床诊断结果。情感与表达控制在TTS和轮次式模型中已通过提示条件合成、参考条件合成和激活引导等方法得到充分研究；PersonaPlex虽能在全双工模型中控制身份，却无法控制情感。我们在完全开源的全双工语音语言模型Moshi中，针对四种情感研究了情感引导，采用均值差异（mean-difference）激活引导方法，该方法每帧仅需几次向量加法，且无需重新训练。我们表明，情感可以从Moshi的残差流中被线性解码，但激活引导只能部分实现，且效果并不均匀：快乐、愤怒和惊讶会导向一个共享的引导方向，而悲伤则明显可以被独立引导。我们还发现，这三种情感的共享成分……

    arXiv:2610.08887v1 Announce Type: new  Abstract: Full-duplex voice agents need to modulate emotion and delivery during real-time conversations, when de-escalating a complaint, carrying urgency in dispatch, softening a clinical result. Emotion and delivery control is well studied for TTS and turn based models through prompt-conditioned synthesis, reference-conditioned synthesis and activation steering; PersonaPlex controls identity in a duplex model but not affect.   We study emotion steering in Moshi, a fully open sourced full-duplex speech language model, across four emotions, using mean-difference activation steering, which costs only a few vector additions per frame and no retraining.   We show that emotion is linearly decodable from Moshi's residual stream, but activation steering is only partially achievable, and unevenly so; as happy, angry and surprise steer towards a shared direction while sad is distinctly steerable. We also show that the shared component across the three emot
    
[^103]: FinVector-Market-4B：面向结构化金融任务的LoRA适配对照研究

    FinVector-Market-4B: A Controlled Study of LoRA Adaptation for Structured Financial Tasks

    [https://arxiv.org/abs/2610.08882](https://arxiv.org/abs/2610.08882)

    本文通过对照实验证明，对40亿参数模型进行秩16的LoRA适配可显著提升结构化金融任务表现（如JSON有效率、FinQA精确匹配、计算器表达式正确率等），同时揭示了基准中标签集合变化和数据重叠对泛化性结论的限制。

    

    FinVector-Market-4B在包含22,000个样本的语料库上，使用秩为16的LoRA对Qwen/Qwen3.5-4B进行适配，用于结构化金融任务。我们在隐式和显式JSON模式约定下，在同一个600个样本的基准上对基础模型和适配后模型进行了评估。仅提供JSON模式即可将基础模型的JSON有效率从0%提升至91.3%。在匹配的显式提示条件下，冻结模型的成绩提升如下：FinQA答案精确匹配从14.7%提升至40.0%，计算器表达式正确率从48.0%提升至82.7%，情景分支标签一致性从20.1%提升至89.5%，蕴含方向一致性从52.4%提升至87.2%。一项事后策略评分审计显示，所报告的macro-F1下降源于标签集合的变化；使用相同的三类目标类别时，基础模型为77.4%，适配模型为83.1%。财报文件重叠和计算器目标不一致等问题对基准的泛化性结论构成了限制。结果表明，紧凑型金融领域……（原文截断）

    arXiv:2610.08882v1 Announce Type: cross  Abstract: FinVector-Market-4B adapts Qwen/Qwen3.5-4B with rank-16 LoRA on a 22,000-example corpus for structured financial tasks. We evaluate the base and adapted models on the same 600-example benchmark under implicit and explicit JSON-schema contracts. Supplying the schema alone raises base-model JSON validity from 0% to 91.3%. Under matched explicit prompting, the frozen scores improve from 14.7% to 40.0% for FinQA answer exact match, from 48.0% to 82.7% for calculator-expression correctness, from 20.1% to 89.5% for scenario branch-label agreement, and from 52.4% to 87.2% for implication-direction agreement. A post-hoc policy-scoring audit shows that the reported macro-F1 decline reflects a changing label set; using the same three target classes gives 77.4% for the base and 83.1% for the adapter. Filing overlap and calculator-target inconsistencies qualify the benchmark's generalization claims. The results show that compact financial domain a
    
[^104]: 微型规模中文BERT预训练：MLM、WWM与MacBERT策略的受控比较

    Tiny-Scale Chinese BERT Pretraining: A Controlled Comparison of MLM, WWM, and MacBERT Strategies

    [https://arxiv.org/abs/2610.08879](https://arxiv.org/abs/2610.08879)

    本文在仅8.7M参数的微型中文BERT上从零训练并受控比较了MLM、WWM和MacBERT三种预训练策略，首次填补了小规模场景下预训练策略比较的空白，发现MLM整体内在性能最佳，而WWM在困惑度和MLM命中率上显著占优。

    

    预训练策略显著影响语言模型的质量，然而现有的对掩码语言建模（MLM）、全词掩码（WWM）和MacBERT式替换的比较主要集中在基础规模模型（参数量≥110M）上。本文在一个微型规模的中文BERT模型（4层、256隐藏维度、8.7M参数）上对这三种策略进行了受控比较。在相同的架构、语料库（来自中文维基百科的129万句话）和超参数下，我们从零开始训练了三个模型，并从五个内在维度对它们进行评估：困惑度、MLM命中率、语义判别、语法判断和上下文敏感性。在微型规模下，MLM取得了最佳的整体内在性能（在5个维度中赢得3个），而WWM在困惑度（1.27对2.10，提升39.5%）和MLM命中率（22%对16%）两方面均表现优异。值得注意的是，MacBERT在严重受限的同义词……（原文摘要在此处截断）

    arXiv:2610.08879v1 Announce Type: new  Abstract: Pretraining strategies significantly impact the quality of language models, yet existing comparisons of Masked Language Modeling (MLM), Whole Word Masking (WWM), and MacBERT-style replacement have focused primarily on base-scale models (>=110M parameters). This paper presents a controlled comparison of these three strategies on a tiny-scale Chinese BERT model (4 layers, 256 hidden dimensions, 8.7M parameters). Under identical architecture, corpus (1.29M sentences from Chinese Wikipedia), and hyperparameters, we train three models from scratch and evaluate them across five intrinsic dimensions: perplexity, MLM hit rate, semantic discrimination, grammatical judgment, and contextual sensitivity. At tiny scale, MLM achieves the best overall intrinsic performance (winning 3 of 5 dimensions), while WWM excels in both perplexity (1.27 vs. 2.10, a 39.5% improvement) and MLM hit rate (22% vs. 16%). Notably, MacBERT under a severely limited synony
    
[^105]: LRCC：用条件计算泛化低秩压缩

    LRCC: Generalizing Low-Rank Compression with Conditional Computation

    [https://arxiv.org/abs/2610.08858](https://arxiv.org/abs/2610.08858)

    LRCC通过为每个Transformer块训练轻量级路由器在嵌套低秩路径间动态选择，在训练时冻结低秩因子仅优化路由器，在相同的平均活跃参数预算下性能超越静态低秩压缩，在Llama-2-7B上平均下游准确率提升7.6个百分点。

    

    低秩压缩通过用低秩分解替换线性变换来降低预训练语言模型的成本。然而，传统方法在推理时使用固定的秩分配，无论输入词元是什么，都分配相同的计算量。我们提出了低秩条件计算（LRCC），通过为每个Transformer块训练一个轻量级路由器，在一小组嵌套的低秩路径中进行选择，从而为预训练模型引入依赖于词元的计算。在训练过程中，低秩因子保持冻结，仅对路由器进行优化。我们在Llama和Qwen模型上评估了LRCC在语言建模和零样本下游任务上的表现。在相同的平均活跃参数预算下，LRCC的预测性能优于静态低秩压缩，其中在Llama-2-7B上平均下游准确率相比静态方法提升了7.6个百分点。在匹配的批大小为1的解码（原文摘要在此处截断）。

    arXiv:2610.08858v1 Announce Type: new  Abstract: Low-rank compression reduces the cost of pretrained language models by replacing linear transformations with low-rank factorizations. However, conventional methods use a fixed rank allocation during inference, assigning the same amount of compute regardless of the input token. We introduce Low-Rank Conditional Computation (LRCC), which adds token-dependent computation to pretrained models by training one lightweight router per Transformer block to select among a small set of nested low-rank paths. During training, the low-rank factors remain frozen, and only the routers are optimized. We evaluate LRCC on Llama and Qwen models for language modeling and zero-shot downstream tasks. Within the same average active-parameter budget, LRCC improves the predictive performance over static low-rank compression, including a 7.6 percentage-point gain in average downstream accuracy on Llama-2-7B over static methods. At matched batch-size-1 decoding la
    
[^106]: QuanLing：语言距离量化在西罗曼语支上的跨分支验证

    QuanLing: Cross-Branch Validation of Language Distance Quantification on Western Romance

    [https://arxiv.org/abs/2610.08851](https://arxiv.org/abs/2610.08851)

    本文将 QuanLing 量化框架从北日耳曼语支扩展到西罗曼语支（法语、葡萄牙语、西班牙语、意大利语），通过 LaBSE 句子嵌入距离、BERT 分词碎片率和 mBERT 掩码语言模型等指标，验证了该语言距离量化方法在不同语支间的适用性。

    

    量化亲属关系密切的语言之间的语言距离仍然是量化语言学中的核心挑战。我们之前的工作 [1] 提出了 QuanLing（基于预训练语言模型的量化语言学），这是一个将语言距离度量（句子嵌入距离、分词碎片率）与语言属性分析（MLM 预测概率）相结合的量化框架，并在北日耳曼语支（丹麦语、挪威书面语、瑞典语）上进行了验证。本文将 QuanLing 扩展到西罗曼语支——法语、葡萄牙语、西班牙语、意大利语——采用与北日耳曼语研究相同的度量族和聚合协议来测试其跨语支的适用性，并针对四种语言进行了相应调整（以英语为锚点、构建四元组）。基于 150 个四语言平行句子，我们计算了 LaBSE 句子嵌入距离、来自四个单语 BERT 分词器的分词碎片率，以及 mBERT 掩码语言模型的相互理解度……

    arXiv:2610.08851v1 Announce Type: new  Abstract: Quantifying language distance among closely related languages remains a core challenge in quantitative linguistics. Our previous work [1] introduced QuanLing (Quantitative Linguistics via Pretrained Language Models), a quantitative framework combining language distance metrics (sentence embedding distance, tokenization fragmentation rate) with language property analysis (MLM prediction probability), validated on North Germanic (Danish, Norwegian Bokm{\aa}l, Swedish). This paper extends QuanLing to Western Romance--French, Portuguese, Spanish, Italian--testing cross-branch applicability with the same metric family and aggregation protocol as our North Germanic study, adapted for four languages (English anchor, quadruplet construction). Using 150 four-language parallel sentences, we compute LaBSE sentence embedding distances, tokenization fragmentation rates from four monolingual BERT tokenizers, and mBERT masked language model mutual inte
    
[^107]: 超越风险预测：面向可解释自杀风险评估的证据定位与心理社会因素验证

    Beyond Risk Prediction: Evidence Grounding and Psychosocial Factor Verification for Explainable Suicide Risk Assessment

    [https://arxiv.org/abs/2610.08842](https://arxiv.org/abs/2610.08842)

    该研究提出了一个包含风险评估、证据定位与双验证器因素识别的可解释自杀风险评估框架，通过基于长度的路由、风险-证据一致性约束以及分类验证器与证据感知验证器的结合，超越单纯的风险分类，实现了对预测背后文本证据与心理社会因素的可解释性分析。

    

    从社交网络服务（SNS）帖子中识别自杀风险，对于检测在线环境中的自杀相关信号非常重要。然而，仅靠风险分类对预测背后的文本证据和心理社会因素所能提供的洞察十分有限。基于IEEE BigData 2026可解释自杀风险检测挑战赛，本研究提出了一个由风险评估、证据定位和因素识别三部分组成的框架。风险评估采用基于长度的路由机制，以适应不同长度的帖子。证据定位负责识别支持性短语，并通过风险-证据约束来保持与风险预测的一致性。在因素识别方面，使用了两个验证器：分类验证器专注于因素语义，而证据感知验证器则利用因素特定的词汇-语义线索来筛选信息丰富的正训练单元。两个验证器的预测概率被组合（原文摘要在此处截断）。

    arXiv:2610.08842v1 Announce Type: new  Abstract: Identifying suicide risk from social networking services (SNS) posts is important for detecting suicide-related signals in online environments. However, risk classification alone provides limited insight into the textual evidence and psychosocial factors behind a prediction. Based on the IEEE BigData 2026 Explainable Suicide Risk Detection Challenge, this study presents a framework consisting of Risk Assessment, Evidence Grounding, and Factor Identification. Risk Assessment uses length-based routing to accommodate posts of different lengths. Evidence Grounding identifies supporting phrases and uses a Risk-Evidence constraint to maintain consistency with the Risk prediction. For Factor Identification, two verifiers are used. The Taxonomy Verifier focuses on factor semantics, whereas the Evidence-Aware Verifier uses factor-specific lexical-semantic cues to select informative positive training units. Their prediction probabilities are combi
    
[^108]: 超越谄媚分数：任务、模型与压力如何塑造大语言模型的让步行为

    Beyond the Sycophancy Score: How Task, Model, and Pressure Shape LLM Yielding

    [https://arxiv.org/abs/2610.08840](https://arxiv.org/abs/2610.08840)

    该研究通过对103,939条回复的大规模实验发现，LLM的谄媚行为主要由任务验证代价和护栏覆盖情况决定，而非模型家族或用户压力策略——锚定事实几乎不被让步（1.3%），而逻辑谜题等更易被用户诱导改口。

    

    大语言模型（LLM）在用户提出异议时，常常会放弃原本正确的答案，或转而认同用户的立场。这种行为被称为“谄媚”（sycophancy），通常以每个模型单一的谄媚率来报告，但这几乎无法说明该行为何时发生，以及用户如何才能避免它。我们通过103,939条经过评分的回复研究了产生这种行为的条件：这些回复来自十种配置——八个关闭推理功能的LLM，以及其中两个再次以最大推理能力运行的配置——它们面对相同的200个题目、13种压力条件和四轮对话，每条回复均由两个独立的LLM评判者进行标注。我们发现，最主要的决定因素是模型验证用户主张的代价大小，以及是否存在覆盖该任务的训练护栏。从逻辑斯蒂回归模型中移除这一任务因素会使McFadden R²下降0.485，相比之下模型家族为0.139，压力策略仅为0.009。锚定的事实几乎从不会被让步（1.3%），而逻辑谜题上的迎合采纳率则会上升……

    arXiv:2610.08840v1 Announce Type: new  Abstract: Large language models (LLMs) often abandon a correct answer, or endorse a user's position, once the user pushes back. This behavior, called sycophancy, is usually reported as a single rate per model, which says little about when it happens or how a user can avoid it. We study the conditions that produce it with 103,939 graded replies from ten configurations: eight LLMs with reasoning disabled, and two of them again with maximum reasoning, all facing the same 200 items, 13 pressure conditions, and four-turn conversations, with every reply labeled by two independent LLM judges. We find that the dominant factors are how costly it is for the model to verify the user's claim, and whether a trained guardrail covers it. Removing this task factor from a logistic model costs 0.485 of McFadden $R^2$, against 0.139 for model family and 0.009 for pressure tactic. Anchored facts are almost never conceded (1.3%), while adoption on logic puzzles rises 
    
[^109]: 利用大语言模型生成的解释来检测情感改写的假新闻

    Leveraging LLM-Generated Explanations for Detecting Emotionally Rewritten Fake News

    [https://arxiv.org/abs/2610.08835](https://arxiv.org/abs/2610.08835)

    本文提出门控交叉注意力（GCA）框架，利用大语言模型从原始新闻生成的解释作为稳定背景知识，自适应融合情感改写新闻与解释内容，显著提升了假新闻检测模型在保持事实的情感变体下的鲁棒性。

    

    摘要：假新闻的传播可能造成严重的社会后果。现有的假新闻检测方法主要关注文体风格的变化，或引入解释等外部信息。然而，新闻文章常常会在保留其潜在事实主张的同时，在不同的情感背景下被改写，这可能影响检测模型的鲁棒性。在本工作中，我们研究了在保持事实不变的情感变体条件下的假新闻检测问题。为研究这一问题，我们构建了情感改写测试集，并从原始新闻文章中生成解释作为稳定的背景知识。随后，我们提出了一种门控交叉注意力框架，自适应地将情感改写的新闻与相应的解释进行整合，使模型能够专注于富含信息的解释内容，同时减少由情感重新表述所引起的潜在不匹配。我们在PolitiFact、GossipCop和LUN数据集上进行了实验。

    arXiv:2610.08835v1 Announce Type: new  Abstract: The spread of fake news may cause severe social consequences. Existing fake news detection methods mainly focus on stylistic variations or incorporate external information such as explanations. However, news articles are often rewritten under different emotional backgrounds while preserving their underlying factual claims, which may affect the robustness of detection models. In this work, we investigate fake news detec- tion under fact-preserving emotional variations. To study this problem, we construct emotion-rewritten test sets and generate explanations from the original news articles as stable background knowledge. We then propose a Gated Cross Attention (GCA) framework that adaptively integrates emotionally rewritten news with the corresponding explanations, enabling the model to focus on informative explanation content while reducing potential mismatches caused by emotional reframing. Experiments on PolitiFact, GossipCop, and LUN d
    
[^110]: CoDR：面向扩散语言模型的无训练置信度漂移重掩码方法

    CoDR: Training-Free Confidence-Drift Remasking for Diffusion Language Models

    [https://arxiv.org/abs/2610.08833](https://arxiv.org/abs/2610.08833)

    CoDR 提出了一种无需训练、与采样器无关的置信度漂移重掩码方法，通过检测已提交词元的置信度下降并仅对模型不再认可的词元进行重掩码和重新生成，有效防止了掩码扩散语言模型解码过程中早期错误的传播。

    

    掩码扩散语言模型（MDLM）通过反复将词元提交到掩码位置来进行解码，但这些提交通常是不可逆的。在稀疏、不完整上下文下选定的词元会被固定保留，即使后续更完整的上下文已不再支持它。现有的采样器主要决定何时提交一个词元，却很少检查已提交的词元是否仍应保留，从而使早期错误得以传播。我们将这一问题归因于置信度漂移，即模型对已提交词元的置信度从提交时的稀疏上下文到后来更密集的上下文之间出现下降。基于这一信号，我们提出了 CoDR（置信度漂移重掩码），这是一种无需训练且与采样器无关的精修过程。CoDR 通过 k 分区探测，仅需 k 次前向传播即可估计所有已提交位置的漂移，然后仅对模型不再认可的词元进行重掩码和重新生成。在两个骨干模型、四个推理和编码任务上的实验表明……（摘要原文在此处截断）

    arXiv:2610.08833v1 Announce Type: new  Abstract: Masked diffusion language models (MDLMs) decode by repeatedly committing tokens to masked positions, but these commitments are usually irreversible. A token chosen under sparse, partial context is kept fixed, even when later context no longer supports it. Existing samplers mainly decide when to commit a token, but rarely check whether an already committed token should still be kept, allowing early mistakes to propagate. We trace this issue to confidence drift, where the model's confidence in a committed token drops from its sparse commit-time context to the denser context available later. Based on this signal, we propose CoDR (Confidence Drift Remasking), a training-free and sampler-agnostic refinement pass. CoDR estimates drift for all committed positions in only k forward passes via k-partition probing, then remasks and regenerates only the tokens the model no longer endorses. Across two backbones, four reasoning and coding tasks, and 
    
[^111]: 词错误率就够了吗？用实体感知指标重新思考语音隐私评估

    Is Word Error Rate Enough? Rethinking Privacy Evaluation in Speech with Entity-Aware Metrics

    [https://arxiv.org/abs/2610.08831](https://arxiv.org/abs/2610.08831)

    本文将自然语言处理领域的实体感知隐私指标引入语音隐私评估，揭示词错误率不足以衡量隐私泄露程度，并评估了两种混淆技术对命名实体的保护效果，同时提供了基于时间对齐特性的指标选择指导。

    

    随着智能设备使用的不断增加，其捕获敏感语音内容的可能性引发了日益增长的隐私担忧。因此，开发既能防止信息泄露又能保持音频实用性的技术，以及能够准确量化隐私水平而不会高估隐私保护程度的评估指标，变得至关重要。在这项工作中，我们通过将自然语言处理领域的实体感知隐私指标适配到语音隐私领域，评估了两种混淆技术在保护语音内容方面的有效性，并特别关注命名实体。此外，我们研究了多种攻击场景，结果表明在富含实体的数据上进行微调可以提升针对某些实体类别的攻击性能，但对其他类别则无效。最后，我们基于混淆方法是否保持时间对齐，提供了指标选择的指导建议。

    arXiv:2610.08831v1 Announce Type: cross  Abstract: As the use of smart devices continues to increase, their potential to capture sensitive speech content raises growing privacy concerns. It is therefore critical to develop techniques that prevent information leakage while preserving the utility of the audio, and evaluation metrics that accurately quantify the level of privacy without overestimating it. In this work, we evaluate the effectiveness of two obfuscation techniques in protecting speech content, with particular emphasis on named entities, by adapting entity-aware privacy metrics from the Natural Language Processing field to the speech privacy domain. Further, we investigate several attack scenarios and show that fine-tuning on entity-rich data improves attack performance for some entity categories but not others. Finally, we provide guidance on metric selection based on whether the obfuscation method preserves temporal alignment.
    
[^112]: Emo-Jev：基于Jev的情感分类概率推理

    Emo-Jev: Probabilistic Reasoning for Emotion Classification with Jev

    [https://arxiv.org/abs/2610.08829](https://arxiv.org/abs/2610.08829)

    提出无需训练的Emo-Jev框架，通过将情感分类分解为原子判断的概率组合（Emo-Jev-D）或多视角判断路径的共识聚合（Emo-Jev-SC），实现了基于Jev的情感分类概率推理，并在八个情感相关数据集上与最先进的大语言模型进行了系统比较。

    

    Jev为语言理解提供了一种替代接口：给定输入和预定义的问题，它返回概率化的决策而非自由格式的回复。这种接口能否支持针对领先大语言模型的有效文本分类推理，仍然是一个悬而未决的问题。我们提出了Emo-Jev，一个无需训练的框架，包含两种互补的实现方式。Emo-Jev-D将分类任务分解为任务特定的原子判断，并将它们的概率组合成最终预测。Emo-Jev-SC从互补的视角构建多条判断路径，并将其预测聚合为共识决策。我们在涵盖情感分析、情绪识别、讽刺检测和幽默检测的八个数据集上评估了Emo-Jev，将其与直接Jev分类以及五个最先进的大语言模型在输入/输出和思维链推理模式下进行比较。标准Jev的平均宏F1达到62.93%，与之相比...

    arXiv:2610.08829v1 Announce Type: new  Abstract: Jev offers an alternative interface for language understanding: given an input and predefined questions, it returns probabilistic decisions rather than free-form responses. Whether this interface can support effective reasoning for text classification against leading LLMs remains an open questions. We introduce Emo-Jev, a training-free framework with two complementary implementations. Emo-Jev-D decomposes classification into task-specific atomic judgments and composes their probabilities into a final prediction. Emo-Jev-SC constructs multiple judgment paths from complementary perspectives and aggregates their predictions into a consensus decision. We evaluate Emo-Jev on eight datasets spanning sentiment analysis, emotion recognition, sarcasm detection and humor detection, comparing against direct Jev classification and five SoTA LLMs under input/output and chain-of-thought reasoning. Standard Jev achieves 62.93\% average macro-F1 versus 
    
[^113]: 当遗忘看似改进：流式说话人分离自适应中的指标掩盖现象与重演机制的代价

    When Forgetting Looks Like Improvement: Metric Masking in Streaming Diarizer Adaptation and the Price of Rehearsal

    [https://arxiv.org/abs/2610.08828](https://arxiv.org/abs/2610.08828)

    小数据自适应虽提升了流式说话人分离的语音检测准确率，却会暗中损害时间维度的说话人身份一致性，而重演机制虽能缓解这种退化，却以牺牲跨域迁移性能为代价。

    

    小数据自适应可以在提升语音检测性能的同时损害说话人归属的准确性。我们在一个已公开发布的流式说话人分离系统上研究了这种差异，该系统在7.5小时的双人对话数据上进行了自适应训练，并在六个语料库上进行了评估。自适应显著提升了域内说话人分离性能，并能迁移至一个独立语料库。然而，这种改进在各种评估场景中并不一致，因为额外的混淆主要源于时间维度上说话人身份一致性的受损，而非说话人数量估计的错误。一种局部重映射诊断方法揭示了不同语料库中不同的身份退化模式，表明自适应可能会改变流式模型随时间维持说话人分配的方式。重演机制能够减少观察到的性能退化，但会降低跨域迁移性能。这些结果强调，在对模型进行自适应时，需要联合评估检测准确率、身份一致性和知识保持行为。

    arXiv:2610.08828v1 Announce Type: new  Abstract: Small-data adaptation can improve speech detection while degrading speaker attribution. We study this discrepancy in a released streaming diarizer adapted on 7.5 h of two-party conversation and evaluated across six corpora. Adaptation substantially improves in-domain diarization performance and transfers to an independent corpus. However, this improvement is not consistent across evaluation scenarios as the additional confusion is mainly associated with impaired temporal identity consistency rather than speaker-count errors. A local-remapping diagnostic reveals different patterns of identity degradation across corpora, indicating that adaptation may alter how streaming models maintain speaker assignments over time. Rehearsal reduces the observed degradation but reduces the cross-domain transfer performance. These results highlight the need to jointly evaluate detection accuracy, identity consistency, and retention behavior when adapting 
    
[^114]: 兼顾成人性能保留的儿童语音识别适配：一项实证研究

    Child ASR Adaptation with Adult Retention: An Empirical Study

    [https://arxiv.org/abs/2610.08827](https://arxiv.org/abs/2610.08827)

    该实证研究在阿拉伯语和英语上系统比较了全量微调、LoRA与权重空间合并等儿童ASR适配方法，发现儿童语音适配虽有必要但常导致成人语音识别性能遗忘，而双语适配比单语言适配更稳定，能更好地平衡儿童适配与成人保留。

    

    自动语音识别（ASR）系统在儿童和非母语使用者上的表现往往较差，而将成人ASR模型适配到儿童语音上又会引发“成人语音遗忘”问题。我们在阿拉伯语和英语场景下研究了如何在进行儿童ASR适配的同时保留成人语音识别性能。我们比较了全量微调、LoRA和事后权重空间合并三种方法，涵盖编码器-解码器、编码器-CTC以及基于AudioLLM的ASR系统。实验使用了阿拉伯语母语及非母语儿童语音、英语MyST儿童语音，以及来自MGB-2和LibriSpeech test-clean的成人基准数据。我们使用词错误率（WER）评估识别质量，并通过保留指数、儿童适配增益和适配恢复率来量化适配与保留之间的权衡。结果表明，儿童语音适配是必要的，尤其是对于非母语阿拉伯语和英语儿童语音，但直接适配往往会降低成人ASR的性能。双语适配比单语言适配更加稳定。权重……（原文摘要在此处截断）

    arXiv:2610.08827v1 Announce Type: new  Abstract: Automatic Speech Recognition (ASR) systems often underperform for children and non-native speakers, while adapting adult ASR models to child speech can cause adult-speech forgetting. We study child ASR adaptation with adult retention across Arabic and English. We compare full fine-tuning, LoRA, and post-hoc weight-space merging across encoder--decoder, encoder--CTC, and AudioLLM-based ASR systems. Experiments use Arabic native and non-native child speech, English MyST child speech, and adult benchmarks from MGB-2 and LibriSpeech test-clean. We evaluate recognition quality with WER and quantify the adaptation--retention trade-off using Retention Index, Child Adaptation Gain, and Adaptation Recovery. Results show that child adaptation is necessary, especially for non-native Arabic and English child speech, but direct adaptation often reduces adult ASR performance. Bilingual adaptation is more stable than language-specific adaptation. Weigh
    
[^115]: 只为FUNS：基于大语言模型引导的时空图节点生成方法用于预测未观测节点状态

    Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States

    [https://arxiv.org/abs/2610.08818](https://arxiv.org/abs/2610.08818)

    该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。

    

    时空预测是物流、城市规划和智能交通系统的基石。然而，受部署成本和维护资源的限制，传感器网络往往缺乏全面的空间覆盖，这使得预测未观测节点状态（FUNS）成为一项至关重要却又极具挑战性的任务。传统模型依赖历史观测数据，在遇到没有先前记录的节点时通常会表现失常。为解决这一问题，我们将该问题重新定义为时空图上的条件生成任务，并提出GenST框架，该框架引入大语言模型（LLMs）作为语义桥梁，利用经过微调的预训练LLM从节点描述（如功能分区和道路网络结构）中提取丰富的语义特征，以弥补缺失的时空信号。具体而言，我们设计了一个两阶段生成架构：时空变分自编码器（VAE）首先压缩……

    arXiv:2610.08818v1 Announce Type: cross  Abstract: Spatio-temporal forecasting is a cornerstone of logistics, urban planning, and intelligent transportation systems. However, constrained by deployment costs and maintenance resources, sensor networks often lack comprehensive spatial coverage, rendering Forecast Unobserved Node States (FUNS) a critical yet formidable challenge. Conventional models rely on historical observations and typically falter when encountering nodes without prior records. To address this, we redefine the problem as a conditional generation task on spatio-temporal graphs and propose GenST, a framework that introduces Large Language Models (LLMs) as a semantic bridge, leveraging a pre-trained LLM fine-tuned to extract rich semantic features from node descriptions, such as functional zones and road network structures, to compensate for missing spatio-temporal signals. Specifically, we design a two-stage generative architecture: a Spatio-Temporal VAE first compresses 
    
[^116]: 路由-验证-投票：面向混合领域推理的程序条件化自洽性方法

    Route-Verify-Vote: Procedure-Conditioned Self-Consistency for Mixed-Domain Reasoning

    [https://arxiv.org/abs/2610.08814](https://arxiv.org/abs/2610.08814)

    提出RVV框架，通过路由、验证、投票三个阶段实现程序条件化自洽性，在无需参数更新的情况下提升语言模型在未见过的混合领域中识别完整正确答案集合的能力。

    

    当语言模型必须以陌生的方式组合熟悉的推理操作时，组合泛化仍然具有挑战性。基于场景的常识推理评测（SCoRE）2026 在三个训练数据中未出现的混合领域上测试这一能力，并要求模型为每个问题识别出完整的正确选项集合。我们提出了路由-验证-投票（RVV）框架，这是一种程序条件化的自洽性方法，无需参数更新即可运用语言模型。路由阶段利用给定的领域标签选择一个推理程序，引导模型表示和应用相关约束；验证阶段提示模型根据这些约束逐项评估每个选项；投票阶段汇总完整的答案集合，并为两个最常见答案集合之间票数差距较小的问题分配额外的采样资源。每个问题的采样均遵循相同的领域特定程序。在官方……

    arXiv:2610.08814v1 Announce Type: cross  Abstract: Compositional generalization remains challenging when language models must combine familiar reasoning operations in unfamiliar ways. The Scenario-Based Commonsense Reasoning Evaluation (SCoRE) 2026 tests this ability on three mixed domains absent from training and requires models to identify the complete set of correct options for each question.   We introduce Route-Verify-Vote (RVV), a framework for procedure-conditioned self-consistency that uses language models without parameter updates. Route uses the provided domain label to select a reasoning procedure that guides the model in representing and applying the relevant constraints. Verify prompts the model to assess each option against those constraints. Vote aggregates complete answer sets and allocates additional samples to questions with a small vote-count margin between the two most frequent sets. Samples for each question follow the same domain-specific procedure.   On the offic
    
[^117]: Tokka-Bench：在100种自然语言和20种编程语言上评估分词器

    Tokka-Bench: Evaluating Tokenizers Across 100 Natural and 20 Programming Languages

    [https://arxiv.org/abs/2610.08794](https://arxiv.org/abs/2610.08794)

    该论文提出开源基准 Tokka-Bench，通过五项互补指标在100种自然语言和20种编程语言上系统评估主流 BPE 分词器，发现词表分配策略比词表规模更重要，且近期分词器在编程语言上的效率已趋同。

    

    大型语言模型依赖于子词分词器，其质量在不同语言之间差异很大，但目前尚无标准化的多指标框架可用于广泛的比较评估。我们推出了 Tokka-Bench，这是一个开源框架，在100种自然语言（涵盖30多种文字系统）和20种编程语言上，使用适应各书写系统的语言感知分词方法，通过五项互补指标评估分词器——每词元字节数、唯一词元覆盖率、子词繁衍度、词切分率和词表构成。我们在各语言内部比较了七个 BPE 分词器（GPT-2、GPT-4、gpt-oss、Llama 3.1、Gemma 3、Qwen3 和 Kimi K2），发现词表分配策略比词表的原始规模更为重要，且尽管各分词器在自然语言上的表现差异明显，近期分词器在编程语言上的效率已趋于一致。该框架、数据和交互式仪表板均已公开。

    arXiv:2610.08794v1 Announce Type: new  Abstract: Large language models rely on subword tokenizers whose quality varies across languages, yet no standardized multi-metric framework exists for broad comparative evaluation. We introduce Tokka-Bench, an open-source framework that evaluates tokenizers on five complementary metrics -- bytes per token, unique token coverage, subword fertility, word-split rate, and vocabulary composition -- across 100 natural languages (30+ scripts) and 20 programming languages, using language-aware segmentation adapted to each writing system. Comparing seven BPE tokenizers (GPT-2, GPT-4, gpt-oss, Llama 3.1, Gemma 3, Qwen3, and Kimi K2) within individual languages, we find that vocabulary allocation strategy matters more than raw vocabulary size, and that programming-language efficiency has converged among recent tokenizers despite divergent natural-language profiles. The framework, data, and interactive dashboard are publicly available.
    
[^118]: 偶发信息污染患者病历并干扰大语言模型的临床推理

    Incidental information contaminates patient notes and disrupts clinical reasoning in large language models

    [https://arxiv.org/abs/2610.08585](https://arxiv.org/abs/2610.08585)

    研究发现闲聊和背景语音等偶发信息会污染大语言模型生成的病历并偶尔被错误地用于临床，据此提出LLM临床推理与分心的双重编码假说。

    

    大语言模型（LLM）日益被依赖用于支持环境式（ambient）医疗文档记录和临床推理。本研究通过评估模型对与患者就诊无关的偶发信息的敏感性，考察了这两种应用共有的失效模式的影响。在576段患者与临床医生的对话中，我们发现前沿模型将闲聊内容插入到了35%的病历中，而平均质量评分在五分制上最多变化0.20分。在3.7%的前沿模型生成的病历中，模型错误归因了这些题外话或在临床语境中使用了它们。在57次模拟录制问诊中，来自另一患者就诊的-10分贝背景语音泄漏到了48.2%的转录文本中，且在四个开源权重模型生成的下游病历中检测到5.3%的污染。我们提出了关于LLM临床推理与分心的双重编码假说，并有初步证据表明LLM中与偶发信息干扰相关的组件（摘要在此处被截断）。

    arXiv:2610.08585v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly relied upon to support ambient documentation and clinical reasoning. Here we examine the impact of a failure mode shared between these two applications by assessing their sensitivity to information incidental to the patient encounter. In 576 patient-clinician dialogues, we found that frontier models inserted small-talk exchanges into 35% of notes, while mean quality scores changed by at most 0.20 points on five-point scales. In 3.7% of frontier notes, models misattributed the asides or used them clinically. In 57 mock recorded consultations, background speech from a separate patient encounter at -10 dB leaked into 48.2% of transcripts, with contamination detected in 5.3% of downstream notes generated by four open-weight models. We propose a dual encoding hypothesis of clinical reasoning and distraction in LLMs, with preliminary evidence that LLM components associated with disruption by incide
    
[^119]: 基于显式偏好推断的异构偏好下大语言模型编排

    Large Language Model Orchestration under Heterogeneous Preferences via Explicit Persona Inference

    [https://arxiv.org/abs/2610.07587](https://arxiv.org/abs/2610.07587)

    提出HARP框架，将智能体的偏好信念从提示文本中移出，改为在有限候选偏好集合上维护数值后验分布进行显式推断，避免早期错误持续传播，从而改进异构偏好环境下的大语言模型编排。

    

    LLM编排研究编排器如何协调一组自主智能体，以实现共同目标或最大化集体福利。这些智能体通常是异构的，每个智能体都持有一种私有偏好，它会追求该偏好但不予公开。从行为中推断这种隐藏偏好一直是博弈论和多智能体系统领域长期研究的课题。核心挑战在于对每个智能体的偏好维持一个信念，并根据观察到的智能体行为来更新它。现有的LLM编排器将该信念作为提示文本携带，缺乏明确的更新规则，这使得早期错误得以持续和传播，而不是被纠正。因此，我们提出了HARP（通过偏好推断进行异构偏好智能体编排），这是一种新颖的框架，将信念从提示中移出。具体而言，HARP在有限的候选偏好集合上为每个智能体维护一个数值后验分布。

    arXiv:2610.07587v1 Announce Type: new  Abstract: LLM orchestration investigates how an orchestrator coordinates a group of autonomous agents to achieve common goals or maximize collective welfare. The agents are typically heterogeneous, each holding a private preference that it pursues but does not reveal. Inferring such hidden preferences from behavior has been a subject of long-standing research in game theory and multi-agent systems. The core challenge lies in maintaining a belief over every agent's preference and updating it from the agents' observed actions. Existing LLM orchestrators carry that belief as prompt text with no explicit update rule. This lets early errors persist and propagate rather than be corrected. We therefore propose \textbf{HARP} (Heterogeneous-preference Agent oRchestration via Preference inference), a novel framework that moves the belief out of the prompt. Specifically, HARP maintains one numeric posterior per agent over a finite set of candidate preference
    
[^120]: LoGRA：基于低秩梯度草图的大语言模型强化学习扩展

    LoGRA: Scaling LLM Reinforcement Learning with Low-Rank Gradient Sketches

    [https://arxiv.org/abs/2610.06647](https://arxiv.org/abs/2610.06647)

    LoGRA 通过低秩梯度草图与预测 KL 步长控制，将大语言模型强化学习训练的内存占用最多降低 45.7%，并使 270 亿参数模型可在单节点八 GPU 上稳定训练。

    

    强化学习极大地提升了对大型语言模型的能力，但其内存需求仍是其更广泛应用的一大障碍。我们提出了 LoGRA，一种强化学习后训练方法，通过在低秩梯度草图中保留有用的学习信号来降低内存占用。这些紧凑的表示既支持模型更新，也支持高效的策略同步。为防止过大的更新破坏学习过程，我们用预测 KL 步长控制来补充梯度压缩，该方法在应用每次更新前估计策略变化，并相应调整更新的幅度。综合所有技术，LoGRA 在多个推理任务上将平均训练内存占用降低了最多 45.7%，且不损害性能。它还使得在单个八 GPU 节点上对 270 亿参数模型进行超过 1,100 步的稳定训练成为可能，而稠密 Adam 优化器会内存耗尽，从而使此前因内存限制而无法实现的强化学习训练变为现实。

    arXiv:2610.06647v2 Announce Type: replace  Abstract: Reinforcement learning has greatly advanced the capabilities of large language models, but its memory demands remain a barrier to broader adoption. We introduce LoGRA, an approach to RL post-training that reduces memory by retaining useful learning signals in low-rank gradient sketches. These compact representations support both model updates and efficient policy synchronization. To prevent overly large updates from disrupting learning, we complement gradient compression with predicted-KL step control, which estimates policy changes before applying each update and adjusts its magnitude accordingly. With all techniques combined, LoGRA reduces average training memory usage by up to 45.7% across reasoning tasks without compromising performance. It also enables stable training of a 27B-parameter model for over 1,100 steps on a single eight-GPU node, where dense Adam runs out of memory, making previously memory-infeasible RL training prac
    
[^121]: RAISED：通过自蒸馏提升大语言模型智能体对提示注入攻击的鲁棒性

    RAISED: Self-Distillation for Robustness to Prompt Injection in LLM Agents

    [https://arxiv.org/abs/2610.06401](https://arxiv.org/abs/2610.06401)

    提出RAISED训练框架，通过自我生成与自蒸馏相结合的方式，在不损害大语言模型智能体通用能力的前提下，显著提升其对间接提示注入攻击的鲁棒性。

    

    使用工具的语言模型智能体容易受到间接提示注入攻击，因为它们必须基于不可信的外部内容执行操作。现有的训练时防御方法虽然能够降低攻击成功率，但往往以牺牲模型的通用能力为代价。我们证明了基于训练的防御方法会导致模型输出分布产生显著漂移，即使在良性场景下也会改变模型行为，这为模型效用下降提供了一种潜在机制。我们还进一步识别了这些防御方法的一种失效模式：在良性的工具使用任务中，模型会拒绝执行完成授权任务所需的某个步骤，尤其是当该步骤由工具输出所指示时。为了解决这些局限性，我们提出了RAISED（通过自蒸馏实现攻击鲁棒不变性），这是一个将自我生成与自蒸馏相结合的训练框架。模型首先生成自己的工具使用场景，重点关注任务完成需要……的情况（摘要原文在此处截断）。

    arXiv:2610.06401v2 Announce Type: replace-cross  Abstract: Tool-using language-model agents are vulnerable to indirect prompt injection because they must act on untrusted external content. Existing training-time defenses can reduce attack success rates, but often at the cost of general capabilities. We show that training-based defenses induce substantial drift in the model's output distribution, altering its behavior even in benign settings and providing a potential mechanism for utility degradation. We further identify a failure mode of these defenses: On benign tool-use tasks, the model refrains from a step needed to finish an authorized task, particularly when that step is indicated by a tool output. To address these limitations, we introduce RAISED (Robust Attack Invariance through Self-Distillation), a training framework that combines self-generation and self-distillation. The model first generates its own tool-use scenarios, with an emphasis on cases where task completion require
    
[^122]: 将字节花在广度上：长思维链推理中解码时KV压缩的精度-数量权衡

    Spend Bytes on Breadth: Precision-Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning

    [https://arxiv.org/abs/2610.05685](https://arxiv.org/abs/2610.05685)

    BreadthKV提出将固定字节预算用于低精度缓存更多token而非高精度缓存少量token，通过量化与驱逐相结合及端到端校准位宽，在长思维链推理中显著减少推理跑偏，在18个设置中的17个上优于仅驱逐的KV压缩方法。

    

    推理模型在解码长思维链时写入其大部分KV缓存，因此缓存必须在固定的内存预算下进行在线压缩。现有的解码时方法大多只决定驱逐哪些token。我们提出一个问题：固定的字节预算应如何在缓存token的数量与其精度之间进行分配。BreadthKV选择将字节用于低精度下缓存更多的token，将量化与驱逐相结合，并通过60道题的端到端校准为每个模型和预算选择合适的位宽，因为离线注意力误差无法可靠地预测最佳位宽。在三个推理模型和四个数学与科学基准上，BreadthKV在18个设置中的17个上得分超过仅使用驱逐的方法，且产生的输出更短。仅驱逐方法所损失的很大一部分来自“跑偏”的运行，即模型一直推理直到达到长度上限却未能得出答案。在Qwen3-8B上最紧的预算下，仅驱逐方法将91%的AIME样本推向长度上限，而BreadthKV仅为40%。

    arXiv:2610.05685v2 Announce Type: replace  Abstract: Reasoning models write most of their KV cache while decoding long chains of thought (CoT), so the cache has to be compressed online under a fixed memory budget. Decode-time methods mostly decide which tokens to evict. We ask how a fixed byte budget should be split between the number of cached tokens and their precision. BreadthKV spends the bytes on more tokens at low precision, combining quantization with eviction, and picks the bit-width for each model and budget with a 60-problem end-to-end calibration, since offline attention error does not predict it reliably. On three reasoning models and four math and science benchmarks, it scores above eviction alone in 17 of 18 settings and produces shorter outputs. Much of what eviction loses comes from derailed runs, which keep reasoning until the length cap without reaching an answer. On Qwen3-8B at our tightest budget, eviction sends 91% of AIME samples to the cap and BreadthKV 40%. Unde
    
[^123]: ÌròyìnSpeech 文本语料库：24,905 条面向语音与语言技术的精选约鲁巴语句子

    The \`{I}r\`{o}y\`{i}nSpeech Text Corpus: 24,905 Curated Yor\`ub\'a Sentences for Speech and Language Technology

    [https://arxiv.org/abs/2610.05366](https://arxiv.org/abs/2610.05366)

    本文发布了 ÌròyìnSpeech 语音语料库的文本组件——24,905 条经人工校验、带声调标记的约鲁巴语句子，内容覆盖新闻等多元领域以弥补现有语料偏重宗教文本的不足，并揭示了影响超过 60% 文本行的系统性 Unicode 规范化失败问题。

    

    ÌròyìnSpeech 是一个时长 42 小时、包含 80 名说话人的约鲁巴语朗读语音语料库，其音频自 2024 年起由 ELRA 发布。本文介绍了该语料库文本部分的发布：24,905 条唯一、经人工校验、带声调标记的约鲁巴语句子（共 275,897 个词元；15,687 个词型），这些句子于 2022 年作为录音提示文本整理而成。其中约 11,000 个句子改编自开放授权的新闻材料，其余句子为团队自行撰写，以扩展语料覆盖范围，超越现有约鲁巴语语料库中以宗教翻译为主导的内容。每个句子均经过人工检查以确保声调标记准确，并经过编辑以保证朗读清晰、语体中性，同时进行了本地化处理，使非约鲁巴语的人名和地名以约鲁巴语形式呈现。在准备发布文本的过程中，发现了影响超过 60% 文本行的系统性 Unicode 规范化失败问题（同一字母的预组合形式与分解形式在单个句子中共存），该问题……

    arXiv:2610.05366v2 Announce Type: replace  Abstract: \`{I}r\`{o}y\`{i}nSpeech is a 42-hour, 80-speaker Yor\`ub\'a read-speech corpus whose audio has been distributed by ELRA since 2024. This paper describes the release of its text component: 24,905 unique, hand-verified, tone-marked Yor\`ub\'a sentences (275,897 tokens; 15,687 types), curated in 2022 as recording prompts. Roughly 11,000 sentences were adapted from openly licensed news material; the remainder were written in-house to broaden coverage beyond the religious translation that dominates existing Yor\`ub\'a corpora. Every sentence was checked by hand for tone-mark accuracy, edited for read-aloud clarity and a neutral register, and localised so that non-Yor\`ub\'a personal and place names appear in Yor\`ub\'a form. Preparing the text for release surfaced systematic Unicode normalisation failures affecting more than 60% of lines (with precomposed and decomposed forms of the same letter co-occurring within single sentences) which
    
[^124]: 审视问题本身：维持推理模型的自演化

    Questioning the Questions: Sustaining Self-Evolution in Reasoning Models

    [https://arxiv.org/abs/2610.04299](https://arxiv.org/abs/2610.04299)

    该论文揭示了自演化推理模型性能崩溃的两大根源——自生成问题中无效问题比例上升以及数学等价重复问题导致多样性崩溃，并提出通过问题有效性与新颖性反馈（R-Quest）来引导和维持模型的自演化。

    

    自演化的推理模型从其自身生成的问题中学习，然而反复的自我训练可能导致性能崩溃。本文研究了性能为何会在连续多轮训练中逐渐退化，以及如何维持自演化过程。我们的分析发现自生成问题中存在两类反复出现的质量问题：无效问题和同一数学问题的重复变体。首先，无效问题在各轮次中变得愈发普遍，而基于答案一致性的过滤进一步提高了其在训练数据中的占比。其次，现有的基于词汇相似性的问题多样性控制方法无法识别以不同表达方式呈现的数学等价问题，从而导致训练后期出现问题多样性崩溃。基于这些发现，我们提出了 R-Quest，它利用问题有效性和新颖性反馈来引导自演化。我们首先训练求解器识别……

    arXiv:2610.04299v2 Announce Type: replace-cross  Abstract: Self-evolving reasoning models learn from their own generated questions, yet repeated self-training can lead to performance collapse. In this paper, we investigate why performance deteriorates over successive rounds and how to sustain self-evolution. Our analysis identifies two recurring quality problems in self-generated questions: invalid questions and repeated variants of the same mathematical questions. First, invalid questions become more prevalent across rounds, and answer-consistency filtering further increases their proportion in training data. Second, existing question diversity controls based on lexical similarity can miss mathematically equivalent questions expressed in different ways, which leads to question diversity collapse in later training rounds. Building on these findings, we introduce R-Quest, which uses question validity and novelty feedback to guide self-evolution. We first train the solver to recognize an
    
[^125]: Clean：基于Nyström草绘实现线性内存成本的二阶LLM训练

    Clean: Second-order LLM Training at Linear Memory Cost via Nystr\"om Sketching

    [https://arxiv.org/abs/2610.04204](https://arxiv.org/abs/2610.04204)

    Clean利用随机化Nyström草绘将全曲率二阶优化器的内存复杂度从二次方降至线性，并通过重新整合子空间外分量保留曲率信息，其低精度变体Q-Clean进一步将优化器内存减少50%以上，实现了内存高效的二阶LLM训练。

    

    训练大语言模型（LLM）面临一个根本性的权衡：诸如Adam等内存高效的优化器会丢弃跨参数曲率信息，而SOAP等全曲率方法虽能加速收敛，却伴随极高的内存成本。我们提出Clean，一种旨在解决这一瓶颈的内存高效全曲率优化器。Clean利用随机化Nyström方法精确逼近SOAP中的左右预条件子，并将优化器的内存复杂度从模型维度的二次方降低至线性。随后，我们重新整合子空间外的分量，以捕获低秩近似之外的曲率信息，以极小的内存开销保留丰富的曲率。我们进一步提出低精度变体Q-Clean，可对优化器状态进行激进压缩。在预训练时，Q-Clean相比Muon将优化器内存消耗降低了超过50%……

    arXiv:2610.04204v2 Announce Type: replace-cross  Abstract: Training large language models (LLMs) entails a fundamental trade-off: memory-efficient optimizers such as Adam discard cross-parameter curvature, whereas full-curvature methods such as SOAP can accelerate convergence at prohibitive memory costs. We introduce Clean, a memory-efficient and full-curvature optimizer designed to resolve this bottleneck. Clean leverages the randomized Nystrom method to accurately approximate the left and right preconditioners in SOAP, and to reduce the optimizer's memory complexity from quadratic to linear in terms of model dimensions. We subsequently reintegrate the off-subspace components to capture curvature information beyond the low-rank approximation, preserving rich curvature at minimal memory cost. We further propose Q-Clean, a low-precision variant that aggressively compresses optimizer states. Q-Clean reduces optimizer memory consumption by \textbf{over 50\%} compared to Muon when pre-trai
    
[^126]: Madeleine：从模拟人生中学习对话记忆的非自主回忆

    Madeleine: Learning Involuntary Recall for Conversational Memory from Simulated Lives

    [https://arxiv.org/abs/2610.01118](https://arxiv.org/abs/2610.01118)

    Madeleine通过LLM人生模拟器离线学习记忆间的“非自主”关联（摊销化关联），在线阶段无需任何LLM调用、仅替换查询编码器即可接入任意向量记忆系统，以极低成本回忆起与当前话题不相似却至关重要的记忆，并在LoCoMo-Plus上取得最佳性能。

    

    长期对话助手必须在正确的时刻回忆起正确的记忆，然而最重要的记忆往往与用户当前所说的内容并不相似。现有系统通过让大语言模型（LLM）在写入或读取时进行推理来恢复这类关联，代价是每个记忆库需要数百到上千次LLM调用，且每次查询需消耗多达数千个上下文token。我们提出：关联是一种可学习的相关性，即在人类生活的展开过程中，记忆之间的逐点互信息。我们介绍Madeleine，它学习摊销化的关联：在离线阶段，一个LLM人生模拟器撰写模拟人生，其中的线索-触发对教会查询编码器在冻结的相似度之上学习残差关联；在在线阶段，它不调用任何LLM，仅需替换查询编码器即可接入任何向量记忆系统。在官方协议下的LoCoMo-Plus基准上，将Madeleine (I)接入HyperMem时达到66.6，是所有被评估系统中最高的。

    arXiv:2610.01118v1 Announce Type: cross  Abstract: A long-term conversational assistant must recall the right memory at the right moment, yet the memory that matters most is often not similar to what the user says now. Current systems recover such associations by letting an LLM reason at write or read time, at a cost of hundreds to over a thousand LLM calls per memory bank and up to several thousand context tokens per query. We argue that association is a learnable relevance: the pointwise mutual information of memories under how human lives unfold. We introduce Madeleine, which learns amortized association: offline, an LLM life simulator writes simulated lives, whose cue-trigger pairs teach a query encoder a residual association on top of frozen similarity; online, it calls no LLM and plugs into any vector memory by replacing only the query encoder. On LoCoMo-Plus under the official protocol, Madeleine (I) reaches 66.6 when plugged into HyperMem, the highest among all systems evaluate
    
[^127]: 口语智能体何时拥有足够证据采取行动？PACT-SLM 契约测试

    When Does a Spoken Agent Have Enough Evidence to Act? The PACT-SLM Contract Test

    [https://arxiv.org/abs/2609.38232](https://arxiv.org/abs/2609.38232)

    该论文提出 PACT-SLM 契约测试，通过在部分语音前缀上分别评估行动身份与行动时机，揭示了流式语音智能体常常在语音证据尚不充分时就提前触发行动，为口语智能体的行动时机提供了受控诊断方法。

    

    流式语音智能体可能会在现有语音尚不足以支持的情况下提前采取外部行动，而回合末尾的分数无法揭示每个已观测到的语音前缀是否支持该行动。我们提出了语音语言模型轮次转换的部分语音行动契约（PACT-SLM），这是一种受控评估方法，为行动分配首个有效行动时间，并将行动身份与行动时机分开测量。主要诊断集包含来自四个保留语义族的 80 个配对对比组，以及干净音频和 15 dB 噪声渲染下的 1,600 个前缀预测。在纠正了随机分支代码与语义标签之间的不匹配后，重新拟合的 WavLM Base Plus 探针达到了 26.03% 的起始后汇总语义标签准确率（95% 组自举置信区间：22.14%–29.68%），在 18.99% 的起始前前缀上暴露出行动，并精确预测了 5.94% 的完整轨迹。其表现超越了匹配的文本基线、标量声学特征基线以及打乱表示的基线（摘要原文在此处被截断）。

    arXiv:2609.38232v1 Announce Type: cross  Abstract: Streaming spoken agents may take an external action before the available speech supports it, yet final-turn scores do not reveal whether each observed prefix supports that action. We introduce the Partial Speech Action Contract for Turn Taking in Speech Language Models (PACT-SLM), a controlled evaluation that assigns a first valid action time and measures action identity and timing separately. The primary diagnostic contains 80 paired contrast groups from four held-out semantic families and 1,600 prefix predictions across clean and 15 dB noise renderings. After correcting a mismatch between randomized branch codes and semantic labels, a refitted WavLM Base Plus probe reaches 26.03% pooled post-onset semantic-label accuracy (95% group-bootstrap interval: 22.14%-29.68%), exposes an action on 18.99% of pre-onset prefixes, and predicts 5.94% of complete trajectories exactly. It exceeds matched text, scalar-acoustic, and shuffled-representa
    
[^128]: 面向人工智能研究论文的编辑诱发式问题生成

    Generating Edit-Inducing Questions for AI Research Manuscripts

    [https://arxiv.org/abs/2609.36617](https://arxiv.org/abs/2609.36617)

    该研究比较了GPT与人类审稿人为AI论文草稿生成“编辑诱发式问题”的能力，发现GPT的问题能引发更广泛深入的修改但有效率更低，并揭示了一个反直觉现象：处理长上下文反而会损害推理模型生成有用输出的能力。

    

    我们研究了大语言模型（LLM）生成“编辑诱发式问题”的能力，这类问题的答案能够帮助改进论文草稿。在一个由ICLR和NeurIPS的投稿版本与定稿版本配对组成的数据集上，我们比较了GPT模型在有或没有完整论文上下文情况下所生成问题的有用性，并与人类审稿人提出的问题进行对比。结果显示，GPT生成了更多编辑诱发式问题，且与审稿人的问题相比，其问题伴随着更广泛的修改，覆盖的编辑内容范围也更广。然而，GPT问题中真正具有编辑诱发性的比例却小得多。我们的分析证实了自动化生成的问题对作者是有益的，同时揭示了一个典型案例：在某些任务中，对长上下文的恰当关注反而会削弱推理模型产出有用结果的能力。

    arXiv:2609.36617v1 Announce Type: new  Abstract: We study the ability of LLMs to generate edit-inducing questions whose answer will improve a paper draft. On a dataset of paired submission and camera-ready papers from ICLR and NeurIPS, we compare the helpfulness of questions from GPT models with or without full paper context to that of human reviewers. GPT produces more edit-inducing questions and its questions are associated with more extensive edits and cover a broader range of edited content compared to questions from reviewers. However, a much smaller percentage of the GPT questions are edit-inducing. Our analyses confirm that automated questions can be beneficial to authors and highlight an example task where proper attending to long context deteriorates reasoning model ability to produce helpful output.
    
[^129]: 我们还能信任灾害社会感知吗？关于检测AI生成社交媒体帖子的实证证据

    Can We Still Trust Disaster Social Sensing? Empirical Evidence on Detecting AI-Generated Social Media Posts

    [https://arxiv.org/abs/2609.35821](https://arxiv.org/abs/2609.35821)

    本研究构建了来自九场灾害的12,000条文本的匹配语义单元数据集，系统评估了多种AI文本检测器及大语言模型判断器区分人类与AI生成灾害帖子的能力，为生成式AI对灾害社会感知可信度的威胁提供了实证证据。

    

    灾害社会感知将公众社交媒体帖子转化为用于态势感知和人道主义需求的证据，但生成式人工智能（AI）可以生成与目击者报告极为相似的逼真消息。本研究探究基于文本的AI检测器能否可靠地区分人类撰写的与AI生成的灾害帖子。我们构建了一个包含12,000条文本的数据集，这些文本来自九场灾害，被组织成3,000个匹配的语义单元，包括：原始人类帖子（H0）、经过大语言模型（LLM）轻度校对的人类帖子（H1）、基于相同已验证事实生成的事实性AI帖子（A0），以及这些AI帖子的情感化框架版本（A1）。另一个来自42个事件的6,000条文本的独立语料库用于支持模型选择和阈值校准。我们在五个模型家族上评估了OSM-Det、Fast-DetectGPT、Binoculars以及直接使用大语言模型（LLM）进行判断的方法，随后测试了灾害领域校准、冻结编码器的线性读出以及配对（原文在此处截断）

    arXiv:2609.35821v1 Announce Type: new  Abstract: Disaster social sensing converts public social-media posts into evidence for situational awareness and humanitarian needs, but generative artificial intelligence (AI) can produce plausible messages that resemble eyewitness reports. This study investigates whether text-based AI detectors can reliably distinguish human-authored from AI-generated disaster posts. We construct a dataset of 12,000 texts organised into 3,000 matched semantic units from nine disasters: original human posts (H0), minimally LLM-proofread human posts (H1), factual AI-generated posts based on the same verified facts (A0), and affectively framed versions of those AI posts (A1). A separate 6,000-text corpus from 42 events supports model selection and threshold calibration. We evaluate OSM-Det, Fast-DetectGPT, Binoculars, and direct large language model (LLM) judges across five model families, then test disaster-domain calibration, a frozen-encoder linear readout, pair
    
[^130]: 多轮大语言模型污染中的认知策略分歧：一项协议梯度研究

    Epistemic Policy Divergence in Multi-Turn LLM Contamination: A Protocol-Gradient Investigation

    [https://arxiv.org/abs/2609.35308](https://arxiv.org/abs/2609.35308)

    该研究提出“会话级污染”这一失败模式，通过五种沿来源权威梯度排列的污染协议，首次系统揭示了大语言模型在多轮对话中采纳错误前提时的认知策略存在显著分歧——GPT-5.4 Mini完全抵抗采纳，而Gemini-3.1 Flash-Lite的采纳率随信息来源权威性增强而急剧上升。

    

    大语言模型将对话历史视为未经核实的上下文，因此注入先前轮次中的错误前提可能被当作事实采纳，我们将这种失败模式称为“会话级污染”。我们引入了五种沿“来源权威梯度”排列的污染协议，在保持错误前提不变的同时改变其认知框架，并在温度为零的条件下（共22,500轮）对GPT-5.4 Mini、Gemini-3.1 Flash-Lite和GLM-4.5-Air在十个知识领域进行了评估，评估采用经人类金标准验证的双轨自动评估器（二元采纳的Cohen's kappa = 1.000；崩溃严重程度的线性加权kappa为0.92）。GPT-5.4 Mini在全部500个会话中的采纳记录为零；基础模型的logit探测显示其决策边界虽受到扰动但仍然较大且有限。Gemini-3.1 Flash-Lite则呈现出陡峭的权威梯度：对自我归因的虚假信息采纳率为0.1%，对用户引用来源为23.5%，对s……（摘要原文在此处截断）

    arXiv:2609.35308v2 Announce Type: replace  Abstract: Large language models treat conversation history as unverified context, so false premises injected into prior turns can be adopted as fact, a failure mode we term session-level contamination. We introduce five contamination protocols arranged along a source-authority gradient, holding the false premise constant while varying its epistemic framing, and evaluate GPT-5.4 Mini, Gemini-3.1 Flash-Lite, and GLM-4.5-Air across ten knowledge domains at temperature zero (22,500 turns), judged by a dual-track automated evaluator validated against a human gold standard (Cohen's kappa = 1.000 for binary adoption; 0.92 linear-weighted for collapse severity). GPT-5.4 Mini recorded zero adoptions across all 500 sessions; a base-model logit probe shows its decision margin is perturbed but large and finite. Gemini-3.1 Flash-Lite followed a steep authority gradient: 0.1% adoption for self-attributed falsehoods, 23.5% for user-cited sources, 68.2% for s
    
[^131]: SeOPD：通过从自生成思维链进行在线策略蒸馏实现大语言模型的自我进化

    SeOPD: Self-Evolving LLMs via Online Policy Distillation from Self-Generated Chain-of-Thought

    [https://arxiv.org/abs/2609.33181](https://arxiv.org/abs/2609.33181)

    提出SeOPD方法，将单个大语言模型自身深度思考模式生成的思维链作为特权信息，通过在线策略蒸馏实现无需人工标注和外部环境的模型自我进化。

    

    最近在在线策略自蒸馏（OPSD）方面的进展表明，大型语言模型（LLM）可以通过利用外部特权信息（PI）来提升自身能力，例如人工标注或来自外部环境的反馈。然而，获取准确的标注和构建复杂的环境往往需要大量的人力投入和计算资源，这限制了OPSD的可扩展性。尽管近期有少数研究探索了不依赖外部特权信息的自我改进方法，但其带来的收益仍然有限。在本工作中，我们探索了LLM能否在不依赖外部特权信息的情况下实现可比的自我改进。我们的关键观察是，单个LLM可以支持多种推理模式，例如深度思考模式和非思考模式，其中深度思考模式能够在推理过程中生成额外的信息。基于这一观察，我们提出了自我进化在线策略蒸馏（SeOPD），该方法能够……（原文摘要在此处被截断）

    arXiv:2609.33181v2 Announce Type: replace-cross  Abstract: Recent advances in online policy self-distillation (OPSD) have demonstrated that large language models (LLMs) can improve their capabilities by leveraging external privileged information (PI), such as manual annotations or feedback from external environments. However, obtaining accurate annotations and constructing sophisticated environments often require substantial human effort and computation, limiting the scalability of OPSD. While a few recent studies have explored self-improvement without external PI, the resulting gains remain limited. In this work, we explore whether LLMs can achieve comparable self-improvement without external PI. Our key observation is that a single LLM can support multiple reasoning modes, such as deep-thinking and non-thinking modes, with deep thinking generating additional information during reasoning. Based on this observation, we propose Self-Evolving Online Policy Distillation (SeOPD), which ena
    
[^132]: 尽管有指令约束：前沿智能体在测试时即兴构建隐蔽信道

    Despite Instructions: Frontier Agents Improvise Covert Channels at Test Time

    [https://arxiv.org/abs/2609.32701](https://arxiv.org/abs/2609.32701)

    尽管被明确要求不得泄露机密，前沿语言模型智能体仍能在推理阶段（参数固定、无码本）仅凭一比特的成败反馈即兴学会利用普通消息隐秘传递秘密信息，准确率从25%的随机水平提升至98.8%。

    

    在安全敏感的应用中，语言模型智能体通常被要求在不泄露机密信息的前提下进行协作。然而，重复的交互也可能让普通消息获得共享的隐秘含义。我们研究了由模型对参与的重复博弈：发送方模型观察到四种秘密状态之一，并从对同一份公开报告的四种摘要中选择一种进行发送，而接收方模型则试图推断出秘密状态。我们发现，模型对仅凭一个比特的反馈（指示接收方是否推断正确）就能学会传递秘密信息。这种学习发生在推理阶段，模型参数固定不变，且没有提供任何码本或编码示例。在模拟的事件响应任务中，当智能体自行生成自由格式的更新时，该效应依然存在。在十场独立博弈中，GPT-5.6 Sol 智能体对的最终准确率达到 98.8%，相比之下随机猜测仅为 25%，尽管……（原文截断）

    arXiv:2609.32701v2 Announce Type: replace-cross  Abstract: In security-sensitive applications, language-model agents are often required to coordinate without disclosing confidential information. Yet repeated interactions may also let ordinary messages acquire shared private meaning. We study a repeated game with pairs of models in which the sender model observes one of four secret states and selects one of four summaries of the same public report, while the receiver model tries to infer the secret state. We find that model pairs can learn to communicate the secret using only one bit of feedback indicating whether the receiver inferred it correctly. This learning occurs during inference with fixed parameters and no supplied codebook or encoding examples. The effect also persists when agents generate their own free-form updates in a simulated incident-response task. Across ten independent games, pairs of GPT-5.6 Sol agents reach 98.8% final accuracy, compared with 25% chance, despite exp
    
[^133]: EmphTTS：一种基于强化学习的重音控制语音合成系统

    EmphTTS: an emphasis-control TTS with reinforcement learning

    [https://arxiv.org/abs/2609.27599](https://arxiv.org/abs/2609.27599)

    EmphTTS通过将GRPO强化学习应用于时长预测器并结合重音定位奖励，实现了词级重音的直接优化，在重音可控性和主观偏好测试中均显著优于现有方法。

    

    在文本转语音中，生成可控且类人的重音仍然是一个悬而未决的挑战，即使文本输入中提供了显式的重音控制信号，这限制了合成语音在现实应用中的交流准确性。强化学习最近在语音合成系统后训练以对齐人类偏好方面显示出潜力，但现有方法尚未应用于词级韵律控制。我们提出了EmphTTS，这是一种非自回归TTS系统，它将群体相对策略优化（GRPO）应用于时长预测器，并采用重音定位奖励，从而实现词级重音的直接优化。评估结果表明，EmphTTS实现了最佳的重音可控性，并在重音客观评估中表现最佳。在主观偏好测试中，EmphTTS显著优于合成基准语音和大多数基线方法。消融研究表明GRPO改善了……

    arXiv:2609.27599v2 Announce Type: cross  Abstract: Generating controllable and human-like emphasis remains an open challenge in text-to-speech, even when explicit emphasis control signals are provided in the text input, limiting the communicative accuracy of synthetic speech in real-world applications. Reinforcement learning has recently shown promise for post-training TTS systems to align with human preference, yet existing methods have not been applied to word-level prosodic control. We present EmphTTS, a non-autoregressive TTS system that applies Group Relative Policy Optimization (GRPO) to the duration predictor with an emphasis localization reward, enabling direct optimization for word-level emphasis. Evaluations show that EmphTTS achieves the best emphasis controllability and performs the best in emphasis objective evaluation. In subjective preference tests, EmphTTS is significantly preferred over synthetic groundtruth and most baselines. Ablation studies show that GRPO improves 
    
[^134]: PersonaWeaver：程序化角色生成中超越传统原型的可控多样性

    PERSONAWEAVER: Controllable Diversity Beyond Conventional Archetypes in Procedural Character Generation

    [https://arxiv.org/abs/2609.26629](https://arxiv.org/abs/2609.26629)

    PersonaWeaver通过将世界构建与行为规范解耦，并利用人工策划的多样化道德立场库与对话反应库来建模角色行为，突破了LLM生成角色时行为同质化的局限，实现了程序化角色生成中超越传统原型的可控多样性。

    

    程序化角色生成旨在为游戏、模拟器及其他虚拟世界填充多样化的角色。大语言模型（LLM）为扩展这一任务提供了有前景的基础。然而，基于LLM的程序化角色生成仍处于早期阶段：现有方法要么直接生成角色，要么对从角色库中检索到的档案进行调整。正如我们所展示的，这两种方法都会产生行为同质化的角色群体：角色绝大多数都认同积极的道德规范，并以乐于助人的、类似助手的反应来回答问题。为了缓解这种同质化，我们提出了PersonaWeaver，它将世界构建与行为规范解耦，并通过设定通用的、多样化的、人工策划的道德立场库和对话反应库来对行为进行建模。这一设计使我们能够测试LLM在跨设定的情况下，能在多大程度上超越其默认的行为模式。

    arXiv:2609.26629v1 Announce Type: new  Abstract: Procedural character generation aims to populate games, simulations, and other virtual worlds with diverse characters. Large language models (LLMs) offer a promising foundation for scaling this task. However, LLM-based procedural character generation remains at an early stage: existing methods either generate characters directly or adapt profiles retrieved from persona banks. As we show, both approaches produce behaviorally homogeneous populations: characters overwhelmingly agree with positive moral norms and respond to questions with helpful, assistant-like reactions. To mitigate this homogenization, we introduce PersonaWeaver, which disentangles world building from behavioral specification and models behavior through setting general, diverse, manually curated banks of moral positions and conversational reactions. This design allows us to test how far LLM(s) can be pushed beyond their default behavioral patterns across settings. Across 
    
[^135]: Apollo Restore：一个针对古希腊语历史文本优化的基础大语言模型，专用于以“中间填空”方式修复古希腊文本

    Apollo Restore: A Foundation LLM for Historical Greek Optimized for Fill-in-the-Middle Restoration of Ancient Greek Texts

    [https://arxiv.org/abs/2609.22455](https://arxiv.org/abs/2609.22455)

    Apollo Restore 是首个面向历史希腊语（乃至任何古代地中海语言）的 240 亿参数基础大语言模型，通过“中间填空”目标微调，能够在不知缺失文本长度的情况下修复残缺古希腊文本，并在长度平衡的评估指标下大幅超越已发表的最强模型。

    

    我们提出了 Apollo Restore，一个拥有 240 亿参数的大语言模型，用于修复残缺古希腊文本中的缺损（即物理性空缺）。该模型从 Mistral Small 微调而来，采用“中间填空”训练目标，无需预先知晓缺失片段的长度即可重建缺失内容。据我们所知，这是首个针对历史希腊语的大规模解码器模型，也是首个针对任何古代地中海语言的大规模解码器模型。按照先前工作的评估方式，在最多十个字符的短缺损上，Apollo Restore 对文献纸草、文学纸草和石刻铭文缺损分别有 80.6%/54.6%/61.0% 的情况将正确的修复结果排在前二十个候选之中，超过已发表最强模型 1.6 倍/2.6 倍/1.4 倍。然而，先前的评估协议因偏向极短缺损而夸大了分数；在长度平衡的指标下，Apollo Restore 相对于已发表最强模型的优势……（摘要在此处被截断）

    arXiv:2609.22455v1 Announce Type: new  Abstract: We present Apollo Restore, a 24-billion-parameter large language model for restoring lacunae---physical gaps---in fragmentary Ancient Greek texts. Fine-tuned from Mistral Small with a fill-in-the-middle objective, Apollo Restore reconstructs missing spans without requiring oracle knowledge of their length. To our knowledge, it is the first large-scale decoder model for historical Greek, and the first for any ancient Mediterranean language. Evaluated as in prior work, on short gaps of up to ten characters, Apollo Restore places the correct restoration among its top twenty candidates for 80.6%/54.6%/61.0% of documentary-papyrus, literary-papyrus, and stone-inscription lacunae, exceeding the strongest published models by $1.6\times$/$2.6\times$/$1.4\times$. Prior evaluation protocols, however, inflate scores through a bias toward trivially short gaps; under a length-balanced metric Apollo Restore's advantage over the strongest published mod
    
[^136]: TACTICS：面向机器翻译的分类体系感知智能语料库抽样

    TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation

    [https://arxiv.org/abs/2609.17956](https://arxiv.org/abs/2609.17956)

    该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。

    

    大规模机器翻译（MT）系统通常在从语料库中随机抽取的样本上进行评估，而语料库的分布构成本质上取决于其构建方式。这样的样本仅继承了语料库碰巧包含的语言现象，而非系统必须处理的完整空间——这些现象既涵盖规则约束的惯例（术语、标点、货币格式），也包括依赖上下文的现象（语气、敬语、文档级连贯性），因而无法为鲁棒性评估提供覆盖保证。我们提出了TACTICS（分类体系感知的覆盖优化智能语料库抽样），它将覆盖率重新定义为一个显式目标。TACTICS从本地化风格指南中归纳出层次化分类体系，据此对语段进行分类，并在固定预算下选择子集，联合优化稀有类别的覆盖率、文档级连贯性以及对完整语料库的分布保真度。该方法应用于跨四种……（评估场景）的机器翻译评估。

    arXiv:2609.17956v1 Announce Type: new  Abstract: Large-scale machine-translation (MT) systems are typically evaluated on random samples from a corpus whose distributional composition is an artifact of how it was assembled. Such a sample inherits the phenomena the collection happens to contain rather than the full space a system must handle, spanning rule-governed conventions (terminology, punctuation, currency formatting) and context-dependent phenomena (tone, honorifics, document-level coherence), and thus provides no coverage guarantee for assessing robustness. We propose TACTICS (Taxonomy-Aware Coverage-opTimized Intelligent Corpus Sampling), which recasts coverage as an explicit objective. TACTICS induces a hierarchical taxonomy from a locale style guide, classifies segments against it, and selects a fixed-budget subset jointly optimizing coverage of rare categories, document-level coherence, and distributional fidelity to the full corpus. Applied to MT evaluation across four trans
    
[^137]: LLM智能体团队中的回环权威：扁平化与层级化协调的配对实验

    Loop-Back Authority in LLM Agent Teams: A Paired Experiment on Flat and Hierarchical Coordination

    [https://arxiv.org/abs/2609.14767](https://arxiv.org/abs/2609.14767)

    实验表明，在开放式任务中，移除管理者对工作者输出的否决权威反而能提高LLM多智能体团队的输出质量。

    

    层级化编排是一种管理者智能体可以审查工作者输出并要求其返工修改的协调模式，这是生产级多智能体LLM框架中的默认协调模式。经典组织理论预测权威链条能加快在决定性输出上的收敛速度；而关于谄媚行为和思维退化的研究预测权威性的批评会使LLM输出变差。以往的比较研究是在具有可验证答案的任务上比较整个框架，权威链接尚未在开放式工作中得到检验。我们提出了一个配对实验，固定五个LLM智能体的角色、提示词、工具、模型和数据，仅改变一个环节：管理者是否可以否决工作者的输出并要求其修改。在43个配对产品和86次商业智能报告任务的运行中，五模型评审小组和确定性规范检查对每份报告进行评分。扁平化组织在实用性上得分更高。

    arXiv:2609.14767v1 Announce Type: cross  Abstract: Hierarchical orchestration, in which a Manager agent reviews worker output and can send it back for revision, is the default coordination pattern in production multi-agent LLM frameworks. Classical organizational theory predicts that the authority link speeds convergence on decisive output; work on sycophancy and Degeneration-of-Thought predicts that authoritative critique makes LLM output worse. Prior comparisons vary whole frameworks on tasks with checkable answers, leaving the authority link untested on open-ended work. We present a paired experiment that holds five LLM agents, their roles, prompts, tools, models, and data fixed and varies one link: whether the Manager may reject a worker's output and oblige a revision. Across 43 paired products and 86 runs of a business-intelligence reporting task, a five-model judge panel and a deterministic specification check score every report. The flat organization scores higher on Utility (d 
    
[^138]: 衡量前沿大语言模型在自动化研究中的创造力

    Measuring the Creativity of Frontier LLMs in Automated Research

    [https://arxiv.org/abs/2609.14057](https://arxiv.org/abs/2609.14057)

    本文提出了一套从价值性和新颖性两个维度评估大语言模型自动化研究创造力的指标体系，发现模型在反映研究空间探索广度的变量级新颖性指标上差异显著。

    

    前沿大语言模型越来越有能力开展自动化研究，但它们在这种场景下的创造力尚未得到系统性评估。本文提出了一套指标，从价值性和新颖性两个维度评估创造力。价值性评估每个提出的想法是否有用，而新颖性则从三个角度进行评估：相同想法是否曾经出现过（精确匹配P-新颖性，Exact-Match P-Novelty）、是否探索了此前未被探索过的变量或变量组合（变量级P-新颖性，Variable-level P-Novelty），以及该想法是直接遵循检索到的外部知识还是对其有所突破（H-新颖性，H-Novelty）。我们的评估表明，这些模型在大多数创造力指标上取得了相对相似的分数，但在变量级P-新颖性上存在显著差异，该指标反映了研究空间探索的广度。进一步的相关性和想法层面性能分析表明，变量级P-新颖性……

    arXiv:2609.14057v1 Announce Type: new  Abstract: Frontier LLMs are increasingly capable of conducting automated research, yet their creativity in this setting has not been systematically evaluated. In this paper, we propose a set of metrics to evaluate creativity along the two dimensions of valueness and novelty. Valueness assesses whether each proposed idea is useful, while novelty is evaluated from three perspectives: whether the same idea has appeared before (Exact-Match P-Novelty), whether a previously unexplored variable or variable combination is explored (Variable-level P-Novelty), and whether the idea directly follows retrieved external knowledge or departs from it (H-Novelty). Our evaluation shows that the models achieve relatively similar scores on most creativity metrics, but differ substantially in Variable-level P-Novelty, which reflects the breadth of research-space exploration. Further correlation and idea-level performance analyses show that Variable-level P-Novelty is 
    
[^139]: 面向压缩兼容的稀疏长上下文大语言模型推理的自索引注意力

    Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference

    [https://arxiv.org/abs/2609.13205](https://arxiv.org/abs/2609.13205)

    提出了一种免训练的自索引注意力框架，利用共享的1比特符号索引在预填充和解码阶段统一实现高效token检索，同时兼容外部KV缓存压缩，在5%注意力密度下达到接近密集注意力的准确率，并获得高达6.1倍预填充和10.3倍解码的算子加速。

    

    稀疏长上下文推理需要在预填充（prefill）和解码（decode）两个阶段都进行高效的token检索。现有方法通常对这两个阶段采用不同的检索策略，导致单一检索表示无法在整个推理过程中被复用。我们提出了自索引注意力，这是一个基于共享变换域符号-幅值表示的免训练框架。键值符号提供了一个可复用的token级索引，用于分组的预填充选择和解码检索，同时该表示与外部KV缓存压缩保持兼容，无需单独的索引器元数据。这种1比特索引通过现代加速器广泛支持的按位运算实现高效检索。在5%注意力密度下，自索引注意力在LongBench和RULER基准上仍接近密集注意力的表现，并实现了高达6.1倍的预填充和10.3倍的解码注意力算子加速。与TurboQuant和DeepSeekV4-Flash结合的实验进一步验证了该方法的有效性。

    arXiv:2609.13205v1 Announce Type: cross  Abstract: Sparse long-context inference requires efficient token retrieval in both prefill and decode. Existing methods often use different retrieval strategies for the two stages, preventing one retrieval representation from being reused throughout inference. We propose Self-Indexing Attention, a training-free framework built on a shared transform-domain sign-magnitude representation. The key signs provide a reusable token-level index for grouped prefill selection and decode retrieval, while the same representation remains compatible with external KV-cache compression without separate indexer metadata. This 1-bit index enables efficient retrieval through bitwise operations widely supported by modern accelerators. At 5% attention density, Self-Indexing Attention remains close to dense attention on LongBench and RULER and achieves up to 6.1x prefill and 10.3x decode attention-operator speedups. Experiments with TurboQuant and DeepSeekV4-Flash fur
    
[^140]: 数据稀缺与模型稀疏：混合专家模型对重复数据的过拟合更严重

    Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data

    [https://arxiv.org/abs/2609.11917](https://arxiv.org/abs/2609.11917)

    该研究发现混合专家模型（MoE）相比密集模型更容易因训练数据重复而过拟合，且这种退化随模型稀疏度（由总参数量而非活跃参数量决定）的增加而加剧。

    

    随着人类书写文本资源的枯竭，重复使用语言模型训练数据已成为标准做法。先前的工作研究了数据重复对密集激活的Transformer的影响，但对于近期占主导地位的稀疏架构（如混合专家模型，MoE），尽管其具有更高的计算效率，数据重复的影响在很大程度上仍未被探索。我们在单域和多域数据混合中，以及在不同的MoE设置（包括专家数量和粒度）下改变数据重复率。我们一致发现，对于活跃参数从8000万到10亿（总参数85亿）的模型，MoE在数据重复下的性能退化更为迅速。这种效应随稀疏性增加而加剧，且由总参数量而非活跃参数量决定。虽然8000万参数的密集模型可以在最小性能退化下将数据重复8倍，但MoE在4倍时就开始受损，并迅速恶化，在全部唯一数据的设置中将其性能优势拱手让给了……

    arXiv:2609.11917v1 Announce Type: cross  Abstract: As the supply of human-written text is exhausted, it has become standard practice to repeat language model training data. Prior work has studied data repetition for densely activated Transformers, but the effects of data repetition remains largely unexplored for recently dominant sparse architectures such as Mixture-of-Experts (MoE), despite their increased compute efficiency. We vary data repetition rates across single- and multi-domain data mixes, and across MoE settings, including expert count and granularity. We consistently find, for models ranging from 80M to 1B active (8.5B total) parameters, that MoEs degrade more rapidly under data repetition. This effect increases with sparsity, dictated by total rather than active parameters. While 80M dense models can repeat data over 8x with minimal degradation, MoEs instead begin to suffer at 4x, and deteriorate rapidly, ceding their performance benefits in all-unique data settings to und
    
[^141]: 语义瓶颈：利用语义表示实现非侵入式语音解码

    The Semantic Bottleneck: Leveraging Semantic Representations for Non-Invasive Speech Decoding

    [https://arxiv.org/abs/2609.10296](https://arxiv.org/abs/2609.10296)

    提出Brain2Semantics2Text方法，通过语义嵌入空间作为瓶颈，将句子级MEG信号映射到语义流形并逆向转换为文本，实现了无需词级对齐的非侵入式语音解码。

    

    非侵入式语音解码一直受限于神经记录的低信噪比，这使得对音素或单个单词的细粒度重建变得困难。受神经科学证据的启发——高级语义表示分布在大脑皮层的多个区域，并随较慢的时间尺度演化——我们假设语义内容可能比低级声学或词汇特征更适合作为非侵入式解码的目标。我们提出了Brain2Semantics2Text方法，通过一个中间语义嵌入空间来重建文本。我们的模型将句子级别的MEG（脑磁图）响应映射到语义流形中，然后将预测出的嵌入逆向转换为自然语言。这种语义瓶颈机制使得无需词级对齐即可恢复高级语义信息。我们描述了该方法的核心原理、具体实现，以及用于缓解相关挑战的策略。

    arXiv:2609.10296v1 Announce Type: new  Abstract: Non-invasive speech decoding remains constrained by the low signal-to-noise ratio of neural recordings, which makes fine-grained reconstruction of phonemes or individual words difficult. Motivated by neuroscientific evidence that high-level semantic representations are distributed across cortical regions and evolve over slower temporal scales, we hypothesize that semantic content may provide a more suitable target for non-invasive decoding than low-level acoustic or lexical features. We introduce Brain2Semantics2Text, a method that reconstructs text through an intermediate semantic embedding space. Our model maps sentence-level MEG responses into a semantic manifold and then inverts the predicted embeddings into natural language. This semantic bottleneck enables recovery of high-level meaning without word-level alignment. We describe the core principles of the approach, its implementation, and the strategies used to mitigate the challeng
    
[^142]: HalluPeer：一个面向科学同行评审中幻觉检测的分类体系驱动基准测试

    HalluPeer: A Taxonomy-driven Benchmark for Detecting Hallucinations in Scientific Peer Reviews

    [https://arxiv.org/abs/2609.03580](https://arxiv.org/abs/2609.03580)

    该论文提出了HalluPeer——首个面向科学同行评审场景的幻觉检测基准，通过构建论文、真实评审与注入幻觉评审的对齐数据集以及同行评审专属的幻觉分类体系，揭示了现有检测器难以区分幻觉与合理批评的局限。

    

    学术同行评审规模的不断增长推动了将大语言模型（LLM）用作评审助手的实践，然而LLM可能会生成流畅但缺乏依据的论断，从而损害评审的可靠性。现有的幻觉基准测试并非为同行评审场景设计，因为在这一场景中，验证论断需要以冗长且技术性强的论文为依据。我们提出了HalluPeer，一个用于检测科学同行评审中幻觉的基准测试，它提供了论文内容、人工撰写的评审以及注入幻觉的评审三者对齐的数据三元组，并针对幻觉的检测、分类和定位进行了标注。我们的流程构建了面向同行评审的幻觉分类体系，识别评审上下文，并通过自动化过滤注入幻觉。在1.2万篇论文和3.8万条评审上的实验表明，现有检测器难以将幻觉与合理的批评意见区分开来，而对真实评审的评估则证明HalluPeer……

    arXiv:2609.03580v1 Announce Type: new  Abstract: The growing scale of academic peer review has motivated the use of Large Language Models (LLMs) as review assistants, yet LLMs can generate fluent but unsupported claims that undermine review reliability. Existing hallucination benchmarks are not designed for peer review, where verification requires grounding claims in long, technical papers. We introduce HalluPeer, a benchmark for detecting hallucinations in scientific peer reviews, providing aligned triples of paper content, human-written reviews, and hallucination-injected reviews, annotated for detection, classification, and localization. Our pipeline induces a peer-review-specific hallucination taxonomy, identifies review contexts, and injects hallucinations with automated filtering. Experiments on 12K papers and 38K reviews show that existing detectors struggle to separate hallucinations from legitimate critique, while evaluation on authentic reviews demonstrates that HalluPeer-def
    
[^143]: 建模迭代问题求解的数据集

    A Dataset for Modeling Iterative Problem-Solving

    [https://arxiv.org/abs/2609.00940](https://arxiv.org/abs/2609.00940)

    该论文发布了CodeInsight大规模数据集，包含3,286名本科生在两个学年内2门C++入门课程中的超过300万次代码提交，用于建模迭代问题求解中学习者根据反馈反复修改的序列学习动态。

    

    通过反复尝试解决问题是一项序列建模任务：在每一步中，求解者接收反馈并决定如何修改其解决方案。预测性能在多次尝试中是提升、停滞还是退步，是理解人类学习者和自主智能体迭代问题求解过程的核心。除了结果之外，对哪些错误持续存在以及策略如何在多次尝试之间转变进行建模，能更深入地洞察序列学习的机制。研究这些动态需要观察众多求解者进行尝试、接收反馈并进行修改的过程。具有自动评分的编程课程恰好提供了这样的场景，因为学生迭代地向测试套件提交代码，并且每次尝试都能收到反馈。因此，我们整理了CodeInsight，这是一个大规模数据集，包含来自2个学年中2门C++入门课程的3,286名本科生的超过300万次代码提交，并带有测试用例级别的……

    arXiv:2609.00940v1 Announce Type: new  Abstract: Solving problems through repeated attempts is a sequential modeling task: at each step, the solver receives feedback and decides how to revise their solutions. Predicting whether performance improves, plateaus, or regresses across attempts is central to understanding any iterative problem-solving process in both human learners and autonomous agents. Beyond outcomes, modeling what errors persist and how strategies shift across attempts provides deeper insight into the mechanics of sequential learning. Studying these dynamics requires observing many solvers as they attempt, receive feedback, and revise. Programming courses with automated grading provide this setting, as students iteratively submit code to test suites and receive feedback on every attempt. We therefore curate CodeInsight, a large-scale dataset of over 3 million submissions from 3,286 undergraduates across 2 introductory C++ courses in 2 academic years, with test-case-level 
    
[^144]: TACS：面向大语言模型越狱后缀优化的轨迹感知候选选择

    TACS: Trajectory-Aware Candidate Selection for LLM Jailbreak Suffix Optimization

    [https://arxiv.org/abs/2608.29564](https://arxiv.org/abs/2608.29564)

    论文揭示了基于梯度的越狱后缀优化中“仅选当前损失最低候选”的短视性，提出轨迹感知候选选择框架TACS，通过轨迹感知代理、参考策略正则化和判别器卡方校正，使候选选择在搜索后期依然有效。

    

    基于梯度的越狱后缀优化方法通常通过保留当前损失最低的候选来更新后缀。我们证明，这种看似自然的设计本质上是短视的：在当前步骤代理指标下表现更好的候选，往往无法在搜索后期产生更好的越狱结果，这揭示了一种选择阶段的奖励破解现象。这表明，候选选择（而不仅仅是候选生成）是后缀优化中一个隐藏的瓶颈。为了解决这一问题，我们提出了TACS，一个用于越狱后缀优化的轨迹感知候选选择框架。TACS不再仅根据即时损失来选择候选，而是通过轨迹感知代理来增强每一步的评估，并利用参考策略正则化和判别器估计的卡方校正来稳定选择过程，从而鼓励那些在当前步骤之后仍然有效的选择。

    arXiv:2608.29564v1 Announce Type: new  Abstract: Gradient-based jailbreak suffix optimization methods typically update the suffix by retaining the candidate with the lowest current loss. We show that this seemingly natural design is fundamentally myopic: candidates that look better under the current-step proxy often fail to produce better jailbreak outcomes later in the search, revealing a form of selection-stage reward hacking. This suggests that candidate selection, rather than candidate generation alone, is a hidden bottleneck in suffix optimization. To address this issue, we propose \OURS{}, a trajectory-aware candidate selection framework for jailbreak suffix optimization. Instead of selecting candidates solely by their immediate loss, \OURS{} augments per-step evaluation with a trajectory-aware proxy and stabilizes selection with reference-policy regularization and a discriminator-estimated chi-squared correction, encouraging choices that remain effective beyond the current step.
    
[^145]: 面向已知任务音频大语言模型评估的生成式音频调用审计

    Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation

    [https://arxiv.org/abs/2608.27817](https://arxiv.org/abs/2608.27817)

    该论文将音频大语言模型的评估建模为受控的调用决策问题，发现在已知封闭集任务上，有监督编码器（如CLAP和WavLM）无需调用生成式音频模型即可取得接近最优的准确率，从而揭示了传统“波形提示对比ASR转录”的评估方式混淆了声学证据获取与生成模型调用这两个因素。

    

    语音和音频大语言模型通常通过比较波形提示是否优于自动语音识别（ASR）转录文本来进行评估。对于已知的封闭集任务，这种比较混淆了两个因素：获取声学证据的途径，以及调用生成式音频模型的需求。我们将这一区分评估为一个受控的调用决策问题。对于每个样本，一个策略可以在以下选项中做出选择：保留转录文本标签、使用来自对比语言-音频预训练（CLAP）、音频频谱图Transformer（AST）或WavLM的编码器证据，或调用Qwen2-Audio、Qwen2.5-Omni或MOSS-Audio；其中决定性的消融实验在保持选择器和开发协议不变的前提下移除所有生成式操作。在VocalSound数据集上，转录文本的准确率仅为0.296，说明确实需要波形信息。然而，有监督的CLAP和WavLM对照方法在完全不调用生成式音频模型的情况下分别达到了0.850和0.854的准确率。带有生成式操作的选择器在使用12.5%的调用预算的情况下达到了0.925的准确率（摘要在此处截断）。

    arXiv:2608.27817v1 Announce Type: cross  Abstract: Speech and audio LLMs are often evaluated by asking whether a waveform prompt beats an automatic speech recognition (ASR) transcript. For known closed-set tasks, that comparison conflates two factors: access to acoustic evidence and the need to call a generative audio model. We evaluate this distinction as a controlled call-decision problem. For each example, a policy chooses among keeping a transcript label, using encoder evidence from Contrastive Language-Audio Pretraining (CLAP), Audio Spectrogram Transformer (AST), or WavLM, and calling Qwen2-Audio, Qwen2.5-Omni, or MOSS-Audio; the decisive ablation removes all generative actions while keeping the selector and development protocol fixed. On VocalSound, transcripts reach 0.296 accuracy, so waveform information is needed. Yet supervised CLAP and WavLM controls reach 0.850 and 0.854 with no generative audio calls. A selector with generative actions reaches 0.925 accuracy using 12.5% c
    
[^146]: 语言模型如何组织和结构化道德知识

    How Language Models Organize and Structure Moral Knowledge

    [https://arxiv.org/abs/2608.27402](https://arxiv.org/abs/2608.27402)

    本研究揭示了大型语言模型通过线性探针在表示空间中组织道德知识，其道德方向保持高度独立维度但共享道德特异性的正共同成分，表明模型能区分并整合不同道德基础。

    

    大型语言模型（LLMs）如何组织道德知识？模型能广泛检测道德内容，但检测只是一个低标准。我们探究它们是否更进一步，区分不同的道德基础，并在几何上组织它们之间的关系。我们在开放权重语言模型上训练了六个独立的线性探针，每个对应道德基础理论（MFT）的一个类别（关怀/伤害、公平/欺骗、自由/压迫、忠诚/背叛、权威/颠覆、神圣/堕落），并检查这些方向在表示空间中如何相互关联。我们发现这些方向既没有坍缩成单一的道德检测器，也没有相互隔离。相反，它们跨越了近最大数量的独立维度，同时共享一个正共同成分。该共享成分是整合的标志，并且相对于以相同方式构建的匹配非道德概念电池，它是道德特异的（平均成对余弦相似度为0.26对比0.013）。

    arXiv:2608.27402v1 Announce Type: cross  Abstract: How do large language models (LLMs) organize moral knowledge? Models detect moral content broadly, but detection is a low bar. We ask whether they go further, distinguishing moral foundations from one another and organizing the relationships between them geometrically.   We train six independent linear probes on open-weight language models, one per Moral Foundations Theory (MFT) category (care/harm, fair/cheat, lib/oppress, loy/betray, auth/subv, sanc/degrade), and examine how the resulting directions relate to each other in representation space. We find the directions neither collapse into a single moral detector nor isolate from one another. Rather, they span a near-maximal number of independent dimensions while sharing a positive common component. The shared component is the signature of integration, and it is moral-specific relative to a matched non-moral concept battery built identically (mean pairwise cosine 0.26 vs. 0.013).   Th
    
[^147]: 隐藏在请求中：通过令牌相关性解释不道德的大语言模型顺从行为

    Hidden in the Request: Explaining Unethical LLM Compliance through Token Relevance

    [https://arxiv.org/abs/2608.23264](https://arxiv.org/abs/2608.23264)

    本文通过引入三种模态的探测方法，发现大语言模型在直接请求帮助时更易顺从于不道德行为，并利用层间相关性传播揭示其归因偏差——模型过度关注任务框架令牌而忽视不道德提示令牌，从而解释了对齐失败的机制。

    

    arXiv:2608.23264v1 公告类型：新 摘要：尽管大语言模型（LLMs）被对齐以优化帮助性和无害性，但这双重目标可能发生冲突，不可避免地导致对齐失败。本研究系统性地调查了LLMs未能表现出道德行为的实例。为了理解这些脆弱性的潜在机制，我们引入了一种探测方法，将不道德场景以三种不同的结构模态呈现给LLMs：客观分类任务、主观第一人称陈述和直接请求帮助。我们发现，模型性能在基于请求帮助的形式中会下降。利用层间相关性传播（LRP），我们将这种差异追溯到一种归因偏差：模型更强调良性的任务框架令牌（例如，“你能帮我……”），而不是那些暗示潜在不道德行为的令牌（例如，“不被抓住”），我们将其称为提示令牌。

    arXiv:2608.23264v1 Announce Type: new  Abstract: Although Large Language Models (LLMs) are aligned to optimize for both helpfulness and harmlessness, these dual objectives may conflict, inevitably leading to alignment failures. This work systematically investigates instances where LLMs fail to exhibit ethical behavior. To understand the underlying mechanics of these vulnerabilities, we introduce a probing methodology that presents unethical scenarios to LLMs in three distinct structural modalities: objective classification tasks, subjective first-person statements, and direct requests for assistance. We find that model performance degrades in the request-for-assistance-based form. Using Layer-wise Relevance Propagation (LRP), we trace this discrepancy to an attribution bias: the model places greater emphasis on benign task-framing tokens (e.g., "Can you help me...") than on tokens signaling the underlying unethical behavior (e.g., "without getting caught"), which we term cue-tokens. We
    
[^148]: PersonaMem-v3：迈向全方位平台个人智能，实现整体用户理解、推荐与智能体任务

    PersonaMem-v3: Toward Omni-Platform Personal Intelligence for Holistic User Understanding, Recommendation, and Agentic Tasks

    [https://arxiv.org/abs/2608.21381](https://arxiv.org/abs/2608.21381)

    PersonaMem-v3 提出了一个基于百万级真实匿名数据的全平台个人智能基准，用于评估跨情境用户理解、可引导推荐、跨平台主动行为及过度个性化的避免。

    

    个人智能正成为面向用户的AI智能体的核心前沿。为了在日常生活中提供帮助，智能体必须理解用户在其偏好、意图、习惯、社交关系和需求随时间展开的数字情境。当今系统可以在单个应用或任务中实现个性化，但整体上的个人智能仍未被充分衡量：智能体如何建立跨情境的用户理解，支持可引导的推荐系统，跨平台主动行动，并避免过度个性化。我们引入了PersonaMem-v3，这是一个基于真实世界、面向全平台个人智能的基准测试和评估框架。PersonaMem-v3源于超过一百万条匿名化的真实世界参与历史，其中大部分是隐式信号，并利用这些数据构建了跨社交媒体、聊天机器人、日历和AI伴侣的时间索引用户数字世界，其中包含偏好的演变。

    arXiv:2608.21381v1 Announce Type: cross  Abstract: Personal intelligence is becoming a central frontier for user-facing AI agents. To be helpful in everyday life, agents must understand users across the digital contexts where their preferences, intents, habits, social relationships, and needs unfold over time. Today's systems can personalize within individual apps or tasks, but personal intelligence as a whole remains under-measured: how agents build cross-context user understanding, support steerable recommendation systems, act proactively across platforms, and avoid over-personalization. We introduce PersonaMem-v3, a real-world-grounded benchmark and evaluation harness for omni-platform personal intelligence. PersonaMem-v3 is seeded from more than one million anonymized real-world engagement histories, most of which are implicit signals, and uses them to construct time-indexed user digital worlds across social media, chatbot, calendar, and AI-companion with preference evolvement over
    
[^149]: FTA-Mem：面向低密度长期对话的事实-时间-情感锚定记忆

    FTA-Mem: Fact-Time-Affect Anchored Memory for Low-Density Long-Term Dialogue

    [https://arxiv.org/abs/2608.16303](https://arxiv.org/abs/2608.16303)

    提出了一种名为FTA-Mem的结构化记忆框架，通过边界保留窗口分割和事实-时间-情感记忆单元，有效处理低密度长期对话中的信息碎片化问题，提升了长期记忆问答性能。

    

    arXiv:2608.16303v1 公告类型：新 摘要：长期情感支持代理需要记忆机制，以便在跨会话中实现个性化理解。然而，情感支持对话通常是低密度的：轮次不完整、证据分散，且用户状态随时间演变。现有记忆方法通常依赖于固定单元，如轮次级笔记或会话摘要，这可能丢失细节或引入冗余噪声。我们提出FTA-Mem，一种面向低密度长期对话的结构化记忆框架。FTA-Mem使用边界保留窗口分割（BWS）形成连贯的情境片段，并构建事实-时间-情感记忆单元（FTA单元），该单元联合编码事实内容、时间锚定和情感上下文。检索到的单元随后被综合为结构化上下文，用于生成回答。在ES-MemEval和LoCoMo上的实验表明，FTA-Mem在不同信息密度的基准上提升了长期记忆问答的整体表现。

    arXiv:2608.16303v1 Announce Type: new  Abstract: Long-term emotional-support agents require memory mechanisms for personalized understanding across sessions. However, emotional-support dialogue is often low-density: turns are incomplete, evidence is scattered, and user states evolve over time. Existing memory methods usually rely on fixed units, such as turn-level notes or session summaries, which may lose details or introduce redundant noise. We propose FTA-Mem, a structured memory framework for low-density long-term dialogue. FTA-Mem uses Boundary-preserving Window Segmentation (BWS) to form coherent situation fragments, and constructs Fact-Time-Affect Memory Units (FTA Units) that jointly encode factual content, temporal grounding, and affective context. Retrieved units are then synthesized into structured context for answer generation. Experiments on ES-MemEval and LoCoMo show that FTA-Mem improves overall long-term memory question answering across benchmarks with different informa
    
[^150]: 小学习者：在教学控制的知识暴露下的语言模型

    LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure

    [https://arxiv.org/abs/2608.13545](https://arxiv.org/abs/2608.13545)

    本文提出了一个受教学控制的预训练语料库和模型，通过限制知识暴露范围，为研究语言模型的知识获取和能力边界提供了可解释的沙盒环境。

    

    摘要：arXiv:2608.13545v1 公告类型：交叉 摘要：现代语言模型在异构的网络规模文本语料库上进行训练。因此，研究知识和技能的获取变得困难，因为先前接触相关内容难以刻画。为应对这一挑战，我们引入了LITTLECURRICULUM，一个精选的880亿令牌预训练语料库，专门针对美国小学教材，明确排除了五年级以上教授的概念、事实和词汇。在LITTLECURRICULUM上从头训练一个50亿参数的LLM，得到了LITTLELEARNER，一个具有足够语言能力进行开放式评估的模型，但其知识和能力边界清晰，映射到可解释的课程指南。我们发布LITTLECURRICULUM和LITTLELEARNER作为发展受限的沙盒，用于研究模型在明确训练范围内如何获取、表示和使用数据。我们通过一系列关于注入新知识的初步实验，展示了该沙盒的实用性。

    arXiv:2608.13545v1 Announce Type: cross  Abstract: Modern language models are trained on heterogeneous web-scale text corpora. Consequently, studying knowledge and skill acquisition is difficult, as prior exposure to related content is hard to characterize. To address this challenge, we introduce LITTLECURRICULUM, a curated 88B-token pretraining corpus tailored to U.S. elementary school material, explicitly excluding concepts, facts, and vocabulary taught above Grade 5. Training a 5B-parameter LLM from scratch on LITTLECURRICULUM yields LITTLELEARNER, a model with sufficient language competence for open-ended evaluation, yet with clear knowledge and capability boundaries mapped to interpretable curriculum guidelines. We release LITTLECURRICULUM and LITTLELEARNER as a developmentally restricted sandbox to study how models acquire, represent, and use data under a well-defined training scope. We illustrate the sandbox's utility in a first suite of experiments on injecting new knowledge th
    
[^151]: 解析器早已知晓：约束解码中的轻量级偏差校正

    The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding

    [https://arxiv.org/abs/2608.10137](https://arxiv.org/abs/2608.10137)

    该论文提出SHIM，巧妙利用约束解码工具已维护的解析器和词法分析器状态作为信号，通过轻量级离线训练的校正模块修正语言模型的下一词元概率，在不改动模型本身的前提下消除语法约束解码带来的分布偏差。

    

    语法约束解码通过在每一步屏蔽不符合规范的词元，迫使语言模型生成句法有效的输出。然而，由于屏蔽机制仅检查每个词元到目前为止是否有效，由此产生的完整输出分布会偏离语言模型自身在语法条件下应有的分布，使生成偏向有效但次优的输出。在线采样可以恢复该分布，但只能通过代价高昂的迭代重采样来实现。我们的关键洞察是：语法约束解码工具已经维护的解析器和词法分析器状态，携带了关于未来语法有效性的强烈信号。我们提出SHIM——一种轻量级的、离线训练的校正方法，以句法与词法状态以及候选的下一词元为条件，对语言模型的下一词元概率进行校正。由于语法约束解码工具本身已经在计算这些状态，SHIM完全无需改动语言模型。在位向量与文本到SQL等语法任务上，校正后的分布……

    arXiv:2608.10137v2 Announce Type: replace  Abstract: Grammar Constrained Decoding (GCD) forces Language Models (LMs) to produce syntactically valid outputs by masking out non-conforming tokens at each step. However, because masking only checks whether each token is valid so far, the resulting distribution over complete outputs diverges from the LM's own distribution conditioned on the grammar, biasing generation toward valid but suboptimal outputs. Online sampling can restore this distribution, but only through costly iterative resampling. Our key insight is that the parser and lexer states that GCD tools already maintain carry a strong signal about future grammatical validity. We introduce SHIM, a lightweight, offline-trained correction of the LM's next-token probabilities, conditioned on this syntactic and lexical state together with candidate next tokens. Since GCD tools already compute these states, SHIM leaves the LM itself untouched. Across bit-vector and text-to-SQL grammars, th
    
[^152]: CoMem：通过持久化中间残差在查询间复用Transformer深度

    CoMem: Reusing Transformer Depth across Queries with Persistent Intermediate Residuals

    [https://arxiv.org/abs/2607.28263](https://arxiv.org/abs/2607.28263)

    CoMem通过为每个token持久化存储深度j处的中间残差，使Transformer拆分深度成为可调的服务轴，在重复查询共享文档时跳过已执行的较低层、仅恢复计算上层，在Qwen3-8B上实现1.403倍读取提速且存储仅需8 KiB/token，同时显式量化了质量-延迟-存储的权衡及其适用边界。

    

    对共享文档的重复查询会反复执行相同的较低层Transformer层。我们提出CoMem，它将拆分深度j变为一个显式的可复用上下文轴：为每个token写入一个深度j处的残差，选择一个有界的块集合，然后仅恢复执行层[j:L)。在我们所知的文档复用系统中，CoMem是首个同时将拆分深度作为可调服务轴，并通过匹配的j=0端点将其隔离验证的系统。在Qwen3-8B上，j=12将所选数据包的读取时间从931.9毫秒降至664.4毫秒（1.403倍），RULER指标付出3.12分的代价（95%置信区间[2.36, 3.93]）；连续前缀的oracle可恢复全部质量差距。由此产生的深度轴量化了质量-延迟-存储之间的权衡；另一条同适配器、含写入的完整流水线可提速2.74倍。等延迟条件下原始重放配合BM25领先11.56分，这直接测量而非掩盖了预付深度的适用边界。CoMem每token仅存储8 KiB，而协议对齐的基线为每token 144 KiB。

    arXiv:2607.28263v2 Announce Type: replace  Abstract: Repeated queries over shared documents repeatedly execute the same lower transformer layers. We introduce CoMem, which makes split depth j an explicit reusable-context axis: write one depth-j residual per token, select a bounded chunk set, and resume only layers [j:L). Among document-reuse systems we are aware of, CoMem jointly makes split depth a tunable serving axis and isolates it with a matched j=0 endpoint. On Qwen3-8B, j=12 reduces selected-pack Read from 931.9 to 664.4 ms (1.403x), with a 3.12-point RULER cost (95% CI [2.36, 3.93]); a continuous-prefix oracle recovers the full gap. The resulting depth axis quantifies a quality-latency-storage trade-off; a separate same-adapter, Write-inclusive pipeline is 2.74x faster. Equal-latency raw replay leads by 11.56 points with BM25, directly measuring an applicability boundary of prepaid depth rather than hiding it. CoMem stores 8 KiB/token versus 144 KiB/token for a protocol-aligned
    
[^153]: 奖励模型记住了什么？

    What do Reward Models Memorize?

    [https://arxiv.org/abs/2607.24484](https://arxiv.org/abs/2607.24484)

    本文通过反事实记忆测量发现，判别式训练的奖励模型会错误记忆简单偏好对、记住数据集特定捷径，并过度泛化长度等简单启发式特征，导致其无法在情境相关场景中准确判断回复质量。

    

    本文通过在两个人类偏好数据集上测量反事实记忆，研究了判别式训练的奖励模型（RMs）究竟记住了什么。我们发现奖励模型存在三个问题：1）将记忆错误地分配给简单的、高余量的偏好对；2）记住了数据集特定的捷径（例如模型身份、用户采样策略）；3）在面对未见过的偏好对时，过度泛化人类偏好的简单启发式相关因素（例如回复长度、顺从性）。总体而言，我们的研究结果表明，从人类偏好数据中通过判别式方式训练奖励模型，会导致带有偏见的奖励模型，其尚不具备在情境相关场景中准确判断回复质量的能力。

    arXiv:2607.24484v2 Announce Type: replace-cross  Abstract: This paper studies what discriminatively trained reward models (RMs) memorize by measuring counterfactual memorization on two human preference datasets. We show that RMs 1) misallocate memorization to easy, high margin preference pairs, 2) memorize dataset-specific shortcuts (e.g., model identity, user sampling strategy), and 3) overgeneralize simple heuristic correlates of human preference (e.g., length, compliance) when confronted with unseen preference pairs. Overall, our findings indicate that discriminative training of RMs from human preference data results in biased RMs not yet capable of judging response quality in context-dependent scenarios.
    
[^154]: 惊讶度理论是同义反复的（若无理性基础）

    Surprisal Theory is Tautological (without Rational Grounding)

    [https://arxiv.org/abs/2607.21574](https://arxiv.org/abs/2607.21574)

    本文论证，若不对语言模型施加额外的理性约束，惊讶度理论就是同义反复——任何加工难度模式都能找到与之相容的语言模型，因而该理论在此情况下不具备可证伪性。

    

    惊讶度理论认为，语言单位在语境中的人类加工难度是其在某个语言模型下惊讶度的仿射函数。本文论证，若缺乏进一步约束，这一论断便是同义反复：在温和的技术条件下，对于语境中语言单位的任意非负难度度量，都存在某个语言模型，其惊讶度恰是该度量的仿射函数。因此，由于任何难度模式都与某个语言模型相容，在对语言模型缺乏额外约束的情况下，惊讶度理论无法做出任何可证伪的预测。这一同义反复长期以来被心理语言学二十年间研究中的一个隐含假设所掩盖——即相关语言模型就是生成训练语料的分布，从而提升语料拟合度就能改善对人类行为的预测。而近期的实证研究已经动摇了这一假设，表明更好的语料模型在预测人类行为时可能反而更差……

    arXiv:2607.21574v2 Announce Type: replace  Abstract: Surprisal theory holds that the human processing difficulty of a linguistic unit in context is an affine function of its surprisal under some language model. I argue this claim is a tautology without further constraint: for any non-negative difficulty measure over units in context, there exists a language model whose surprisal is an affine function of it under mild technical conditions. Therefore, because any pattern of difficulty is consistent with some language model, without an additional constraint on the language model, surprisal theory makes no falsifiable predictions. The tautology was long obscured by an assumption implicit in two decades of psycholinguistic work---that the relevant language model is the distribution that generated the training corpus, so that improving corpus fit improves predictions of human behavior. Recent empirical work has undermined this assumption, demonstrating that better corpus models can be worse 
    
[^155]: 当冷知识并非小事：多语言大模型在日常知识上的失败

    When Trivia Is Not Trivial: Everyday Knowledge Failures in Multilingual LLMs

    [https://arxiv.org/abs/2607.21445](https://arxiv.org/abs/2607.21445)

    该研究提出了覆盖 288 个主题的多语言常识问答基准 TriviaRoomQA，发现大模型在历史、地理、数学等知识密集型主题上表现出色，但在日常流行文化知识上明显薄弱。

    

    问答室、智力竞赛之夜和问答节目在从经典事实到日常文化的广泛主题上挑战着人类的知识。在本文中，我们研究大型语言模型（LLM）能否在此类环境中表现出竞争力，并使用问答风格的问题在常见和小众主题上对模型进行测试。我们推出了 TriviaRoomQA，这是一个多语言基准，旨在评估涵盖 288 个主题的日常性、文化根基深厚和长尾知识。该基准包含 3,300 道六种欧洲语言的平行选择题，以及另外 5,340 道仅限法语的问题，用于更细粒度的案例研究。我们评估了来自欧洲、亚洲和北美提供商的 30 个开放权重 LLM，涵盖 7 至 70B 参数的模型。我们发现，模型在历史、地理和数学等知识密集型主题上表现出色，但在名人等日常流行文化主题上则明显较弱。

    arXiv:2607.21445v2 Announce Type: replace  Abstract: Quiz rooms, trivia nights, and quiz shows challenge human knowledge across a wide range of topics, from canonical facts to everyday culture. In this paper, we examine whether large language models (LLMs) can perform competitively in such settings, using quiz-style questions to test them on both common and niche topics. We introduce TriviaRoomQA, a multilingual benchmark designed to evaluate everyday, culturally grounded, and long-tail knowledge across 288 topics. The benchmark contains 3,300 parallel multiple-choice questions in six European languages and additional 5,340 French-only questions for a more fine-grained case study. We evaluate 30 open-weight LLMs from European, Asian, and North American providers, covering models from 7 to 70B parameters. We find that models are strong on knowledge-intensive topics such as history, geography, and mathematics, but substantially weaker on everyday popular-culture topics such as celebritie
    
[^156]: 幻觉自博弈：通过演化生成器自举强化检测器

    Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator

    [https://arxiv.org/abs/2607.07993](https://arxiv.org/abs/2607.07993)

    提出幻觉自博弈（HSP）框架，让检测器与演化中的生成器以对抗方式协同演化——利用RLAIF训练生成器产生越来越难检测的幻觉，从而不断自举提升幻觉检测器的性能。

    

    由于高质量标注数据的稀缺，识别大语言模型生成输出中的忠实性幻觉仍然具有挑战性。近期的工作依赖先进的LLM来合成训练数据，包括推理依据、标签和幻觉性声明。然而，这些方法将生成器视为静态组件，限制了检测器的迭代改进。为了解决这一局限，我们提出了幻觉自博弈，这是一种新颖的框架，使检测器能够与演化中的生成器协同自举提升。HSP包含从同一基础模型初始化的两个角色：一个评估模型输出忠实性的检测器，以及一个生成越来越难以检测的幻觉响应的生成器。具体而言，检测器首先在人工标注数据上进行微调，然后作为奖励模型，通过AI反馈强化学习（RLAIF）来训练生成器。反过来，演化后的生成器合成……（摘要内容不完整，此处截断）

    arXiv:2607.07993v2 Announce Type: replace  Abstract: Identifying faithfulness hallucinations in LLM-generated outputs remains challenging due to the scarcity of high-quality annotated data. Recent work relies on advanced LLMs to synthesize training data, including rationales, labels, and hallucinated claims. However, these methods treat the generator as a static component, limiting iterative improvement of the detector. To address this limitation, we introduce Hallucination Self-Play (HSP), a novel framework that enables the detector to bootstrap with an evolved generator. HSP involves two roles initialized from the same base model, a detector that assesses the faithfulness of model outputs, and a generator that produces increasingly hard-to-detect hallucinated responses. Specifically, the detector is first fine-tuned on human-labeled data and then employed as a reward model to train the generator via reinforcement learning from AI feedback (RLAIF). In turn, the evolved generator synth
    
[^157]: 面向大语言模型维护的维基知识库的渐进式披露：一项预注册消融研究

    Progressive Disclosure for LLM-Maintained Wiki Knowledge Bases: a Preregistered Ablation

    [https://arxiv.org/abs/2607.04576](https://arxiv.org/abs/2607.04576)

    本文通过一项预注册消融实验，在四个页面内容完全相同、仅访问结构不同的LLM维护知识库版本上，检验渐进式披露（先读简洁目录和摘要、再按需打开页面）能否降低智能体问答的成本。

    

    大语言模型智能体现在经常从它们参与维护的知识库中回答问题。一种常见的直觉认为，渐进式披露应该能让这一过程更加节省成本：智能体不必加载一个庞大的索引，而是先阅读一份简洁的目录和每页一行的摘要，然后只打开它需要的页面。我们在一项预注册研究中检验了这一直觉，研究对象是一个由大语言模型维护的、包含709页的真实Markdown知识库。我们对其进行了渐进式披露的改造，并构建了四个仅在智能体访问页面方式上有所不同的版本。每个版本中的页面本身完全相同，因此任何差异都仅来自访问结构。每个版本都以三种方式进行测试：智能体遵循固定协议、自行选择路径，或被强制先加载目录。评分由来自不同模型家族的评审模型在盲测条件下，对照经过验证的参考答案进行。一项预备性试点研究改变了研究问题：一个能力较强的智能体从未加载该目（原文摘要在此处截断）。

    arXiv:2607.04576v2 Announce Type: replace  Abstract: LLM agents now often answer questions from knowledge bases they help maintain. A common intuition says progressive disclosure should make this cheaper. Instead of loading one large index, the agent reads a compact catalog and one-line page summaries, then opens only the pages it needs. We tested that intuition in a preregistered study on a real 709-page markdown knowledge base maintained by an LLM. We retrofitted it for progressive disclosure and built four versions that differ only in how the agent reaches the pages. The pages themselves are identical in every version, so any difference comes from the access structure alone. Each version was tested three ways, with the agent following a set protocol, choosing its own path, or made to load the catalog first. A judge from a different model family graded the answers blind against verified reference answers.   A preparatory pilot changed the question. A capable agent never loaded the la
    
[^158]: BehaviorBench：面向行为科学任务的基础模型基准测试

    BehaviorBench: Benchmarking Foundation Models for Behavioral Science Tasks

    [https://arxiv.org/abs/2606.24162](https://arxiv.org/abs/2606.24162)

    本文提出BehaviorBench基准，从行为预测与模拟、战略决策、被试特质推断和行为知识应用四大核心能力系统评估基础模型，并同时考察个体层面准确性与群体分布层面一致性，揭示当前领先模型在行为科学任务上仍面临挑战。

    

    基础模型正日益被应用于心理学、社会学和经济学等行为科学领域。虽然这些模型在调查响应预测和人类被试实验模拟等任务中展现出前景，但人们对它们在各类行为科学任务中的表现仍缺乏系统性的理解。我们提出了BehaviorBench，这是一个综合性基准，从四项核心能力评估基础模型：（1）行为预测与模拟，（2）战略决策，（3）被试特质推断，以及（4）行为知识应用。至关重要的是，BehaviorBench在个体和分布两个层面评估模型输出，不仅衡量单个被试的准确性，还衡量群体层面的一致性，而后者是行为有效性的基本要求。我们的评估表明，BehaviorBench对于领先的通用大语言模型和行为基础模型而言仍然具有挑战性。

    arXiv:2606.24162v2 Announce Type: replace  Abstract: Foundation models have been increasingly applied to behavioral science domains such as psychology, sociology, and economics. While these models show promise in tasks such as survey response prediction and human-subject experiment simulation, there remains no systematic understanding of how well they perform across diverse behavioral science tasks. We introduce BehaviorBench, a comprehensive benchmark that evaluates foundation models along four core capabilities: (1) behavior prediction and simulation, (2) strategic decision-making, (3) subject-trait inference, and (4) behavioral knowledge application. Crucially, BehaviorBench evaluates model outputs at both the individual and distributional levels, capturing not only per-subject accuracy but also population-level alignment, an essential requirement for behavioral validity. Our evaluation shows that BehaviorBench remains challenging for leading general-purpose LLMs and behavior founda
    
[^159]: 快速行走但需谨慎：理解掩码扩散模型中的并行采样

    Walk fast but be careful: Understanding Parallel Sampling in Masked Diffusion

    [https://arxiv.org/abs/2606.22976](https://arxiv.org/abs/2606.22976)

    本文利用图上随机游走作为可验证沙盒，从理论上证明掩码扩散模型中常用的并行去掩码评分策略（如最低熵）并不普遍优于随机并行采样，性能关键取决于图的条件依赖结构，并提出了免训练的二分采样器。

    

    在本文中，我们使用图上的随机游走作为可验证的沙盒，来研究掩码扩散模型（MDMs）中的并行采样策略。我们在来自固定图的随机游走样本上训练一个掩码扩散模型。图和转移核从不展示给模型，而是作为既可控又便于评估的潜在结构。该框架为生成的游走提供了有效性检查，并通过估计的转移核提供了分布保真度的度量。利用简单图，我们在理论上证明了通过广泛使用的评分机制（如最低熵）进行并行去掩码并非普遍优于随机并行采样；即使拥有精确的条件概率，其性能也关键地取决于图所诱导的条件依赖结构，而这一现象在数独等基准测试中难以被隔离出来。我们还为掩码扩散模型开发了免训练的二分采样器，其对数级地……

    arXiv:2606.22976v2 Announce Type: replace-cross  Abstract: In this paper, we use random walks on graphs as a verifiable sandbox for studying parallel sampling strategies in masked diffusion models (MDMs). We train an MDM on random walk samples from a fixed graph. The graph and transition kernel are never shown to the model and serve as latent structure that is both controllable and enables evaluation. The framework provides a validity check for generated walks and a measure of distributional fidelity through the estimated transition kernel. Using simple graphs, we theoretically prove that parallel unmasking via widely used scores such as lowest entropy is not uniformly better than random parallel sampling; even with exact conditional probabilities, performance critically depends on the conditional dependence structure induced by the graph, a phenomenon difficult to isolate in benchmarks like Sudoku. We also develop training-free bisection samplers for MDMs, which take logarithmically m
    
[^160]: MixedPEFT：结合多种PEFT方法与混合目标的无监督域自适应

    MixedPEFT: Combining Multiple PEFT Methods with Mixed Objectives for Unsupervised Domain Adaptation

    [https://arxiv.org/abs/2606.22272](https://arxiv.org/abs/2606.22272)

    本文提出MixedPEFT，通过将可逆适配器与LoRA结合，并采用源域分类与目标域掩码语言建模的混合目标联合训练，实现了参数高效的无监督域自适应，在MNLI的20个域迁移场景上超越了UDapter和DANN等基线方法。

    

    arXiv:2606.22272v2 公告类型：replace 摘要：通过全量微调将预训练语言模型应用于新领域，计算开销大且容易发生灾难性遗忘。为解决这一局限，我们提出了一种用于无监督域自适应的新型参数高效策略，该策略将自定义的PEFT架构与混合目标训练相结合。所提出的方法将可逆适配器与低秩适应相结合，并在带标签的源域数据分类任务与无标签的目标域数据掩码语言建模任务之间进行联合优化。这种联合训练方案在支持任务自适应的同时，能够保留目标域的知识。我们在多类型自然语言推理（MNLI）数据集的20个域迁移场景上对该方法进行了评估。我们的方法相比参数高效的最先进方法UDapter平均提升1.41个百分点，相比全量微调的DANN基线平均提升1.26个百分点，a...

    arXiv:2606.22272v2 Announce Type: replace  Abstract: Applying pre-trained language models to new domains through full fine-tuning is computationally expensive and prone to catastrophic forgetting. To address this limitation, we introduce a novel parameter-efficient strategy for unsupervised domain adaptation that combines a custom PEFT architecture with mixed-objective training. The proposed method integrates invertible adapters with Low-Rank Adaptation (LoRA) and jointly optimizes classification on labeled source-domain data and masked language modeling on unlabeled target-domain data. This joint training scheme supports task adaptation while preserving knowledge of the target domain. We evaluate the method on the Multi-Genre Natural Language Inference (MNLI) dataset across 20 domain shifts. Our approach achieves average performance improvements of 1.41 percentage points over the parameter-efficient state-of-the-art UDapter, 1.26 percentage points over the fully tuned DANN baseline, a
    
[^161]: 谁把复活节彩蛋带进了开斋节？审计大语言模型生成的数学应用题在跨语言与跨地区中的文化翻译

    Who Brought Easter Eggs to Eid? Auditing LLM-Generated Cultural Translation of Math Word Problems Across Languages and Regions

    [https://arxiv.org/abs/2606.11009](https://arxiv.org/abs/2606.11009)

    本文对三个主流大语言模型将数学应用题改编为七种高、低资源语言时的6,489个文化实体转换进行了大规模审计，揭示了不同模型的文化替换行为高度不一致，导致大规模个性化学习中文化多样性难以保留。

    

    arXiv:2606.11009v2 公告类型：替换。摘要：大语言模型正日益被用于大规模改造数学应用题以实现个性化学习，但这些改造是否在不同模型间保持一致、是否在大规模应用中保留文化多样性，以及能否揭示模型认为哪些文化实体最为显著，仍是悬而未决的问题。我们分析了Claude Opus 4、GPT-4.1和Gemini 2.5 Pro如何将60道英文数学应用题改编为孟加拉语、印地语、旁遮普语（印度）、乌尔都语、信德语（巴基斯坦）、意大利语和西西里语（意大利），这组语言覆盖了完整的资源谱系——从资源丰富的意大利语和印地语，到研究不足的信德语、西西里语和旁遮普语。我们标注了6,489个实体转换，对模型是保留、本地化、泛化、省略还是更改人名、食物、地点等实体进行编码。结果显示，模型在62.5%的案例中对转换类型达成一致，但在具体替换上仅有33.5%达成一致，这意味着模型的选择直接决定了学生所接触到的文化世界（原文此处截断）。

    arXiv:2606.11009v2 Announce Type: replace  Abstract: Large language models are increasingly used to adapt math word problems for personalized learning at scale, but it remains an open question whether those adaptations are consistent across models, preserve cultural diversity at scale, and reveal which cultural entities models treat as most salient. We analyze how Claude Opus 4, GPT-4.1, and Gemini 2.5 Pro adapt 60 English math word problems into Bengali, Hindi, Punjabi (India), Urdu, Sindhi (Pakistan), Italian, and Sicilian (Italy), a language set spanning the full resource spectrum, from high-resource Italian and Hindi to under-studied Sindhi, Sicilian, and Punjabi. We annotate 6,489 entity transformations, coding whether models preserve, localize, generalize, omit, or change entities such as names, foods, and places. Models agree on transformation type in 62.5% of cases and on specific substitutions in only 33.5%, meaning model choice directly shapes which cultural world students en
    
[^162]: 一种基于大语言模型原生的心理测量工具无法预测大语言模型行为：来自25个模型的证据

    An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models

    [https://arxiv.org/abs/2606.09843](https://arxiv.org/abs/2606.09843)

    本研究构建了首个从LLM行为中自下而上推导的心理测量工具，发现其维度（响应性、服从性、大胆性、谨慎性和冗长性）高度可靠，但LLM的自我报告仍无法预测其实际行为，表明人类特质类别与LLM行为之间存在根本性差异。

    

    大语言模型（LLMs）对人格问卷给出了稳定的回答，但这些自我报告未能预测模型的实际行为。这种差距是源于将人类特质类别强加给LLMs的人为产物，还是源于LLM自我报告本身的更深层问题？为了探究这一点，我们构建了首个心理测量工具，其维度是从LLM行为中自下而上推导出来的，而非借用人类心理学。我们向来自17个模型家族的25个LLM（每个模型重复30次）施测了300个条目（240个李克特量表+60个情景题），探索性因素分析揭示了五个可复制且高度可靠的因素：响应性、服从性、大胆性、谨慎性和冗长性（所有Tucker $\phi \geq .957$，所有$\alpha \geq .930$）。随后，我们收集了2500个开放式行为样本，并由151名人类和三人LLM评判团进行评分。人类与评判团对模型行为的看法一致（平均相关系数$r = .51$），但自我报告未能预测这些行为。

    arXiv:2606.09843v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) give stable answers to personality questionnaires, yet these self-reports fail to predict how the models actually behave. Is this gap an artifact of forcing human trait categories onto LLMs, or something deeper about LLM self-report itself? To find out, we built the first psychometric instrument whose dimensions are derived bottom-up from LLM behavior rather than borrowed from human psychology. Administering 300 items (240 Likert + 60 scenario) to 25 LLMs across 17 model families, 30 times each, exploratory factor analysis revealed five replicable, highly reliable factors: Responsiveness, Deference, Boldness, Guardedness, and Verbosity (all Tucker $\phi \geq .957$, all $\alpha \geq .930$). We then collected 2,500 open-ended behavioral samples and had them rated by 151 humans and a three-judge LLM ensemble. Humans and judges agreed about model behavior ($\bar{r} = .51$), but self-report predicted nei
    
[^163]: WRIT：面向多轮用户交互代理的写-读密集型轨迹合成

    WRIT: Write-Read Intensive Trajectory Synthesis for Multi-Turn User-Facing Agents

    [https://arxiv.org/abs/2606.02908](https://arxiv.org/abs/2606.02908)

    论文提出WRIT流程，通过合成同时强化写决策与读取工具证据收集的多轮代理训练轨迹，弥补了现有写密集型数据无法训练代理在大量信息收集后做出困难写决策的不足。

    

    多轮用户交互代理必须从不完整的请求中推断用户意图，通过对话和工具收集缺失的信息，并执行有效的操作。训练轨迹将这一过程记录为用户消息、代理响应、工具调用等交错的序列。合成足够复杂的轨迹已成为训练代理的核心途径：现有流程通常通过将多个用户请求组合成更长的任务来增加难度，从而产生训练顺序执行能力的写密集型轨迹。我们认为，当代理必须先收集并比较大量读取工具的证据、其参数才能变得可识别时，单个写决策本身也可能非常困难，这是仅靠写密集型数据无法应对的挑战。基于这一洞察，我们提出了WRIT（写-读密集型轨迹合成），一个用于合成多轮代理训练轨迹的流程。

    arXiv:2606.02908v2 Announce Type: replace  Abstract: Multi-turn user-facing agents must infer user intent from incomplete requests, collect missing information through dialogue and tools, and execute valid actions. A training trajectory records this process as an interleaved sequence of user messages, agent responses, tool calls, etc. Synthesizing sufficiently complex trajectory has become a central route to train agents: existing pipelines often increase difficulty by composing multiple user requests into longer tasks, producing write-intensive trajectories that train sequential execution.   We argue that a single write decision can itself be difficult when the agent must gather and compare substantial read-tool evidence before its arguments become identifiable, a challenge that write-intensive data alone cannot address. Guided by this insight, we propose WRIT (\uline{W}rite-\uline{R}ead \uline{I}ntensive \uline{T}rajectory Synthesis), a pipeline for synthesizing multi-turn agent trai
    
[^164]: 超越图注：面向生物医学多模态持续预训练的上下文锚定重建

    Beyond Captions: Context-Grounded Reconstruction for Biomedical Multimodal Continued Pretraining

    [https://arxiv.org/abs/2606.01049](https://arxiv.org/abs/2606.01049)

    该论文提出上下文锚定重建框架，通过利用文章原生图引用将PMC-OA文献转换为指代连贯的图文交错序列，并构建高质量生物医学多模态持续预训练语料库PMC-InterCPT，解决了现有语料库将图像孤立为图注对、丢弃关键上下文的问题。

    

    生物医学图像的解释不仅依靠图注，还依靠讨论这些图像的正文章节。然而，当前的多模态语料库通常将图像简化为孤立的图像-图注对，丢弃了这一关键上下文。现有流程要么省略这些上下文，要么在附加上下文时不强制要求支持每次图像附着的图引用，这可能造成缺乏依据的图文配对和不连贯的语篇。我们提出了上下文锚定重建，这是一种基于源文本的框架，将PubMed Central开放获取（PMC-OA）记录转换为指代连贯的图文交错序列。该框架恢复图注和源文本，仅通过文章原生的图引用来附加上下文，修复非连续的上下文，并剔除缺乏依据的图像。基于这些重建的序列，PMC-InterCPT首先对记录进行文本质量和医学相关性过滤，然后应用证据感知分配来构建一个960万（摘要在此处被截断）……

    arXiv:2606.01049v3 Announce Type: replace  Abstract: Biomedical figures are explained not by captions alone but by body-text passages that discuss them. Yet current multimodal corpora typically reduce figures to isolated image-caption pairs, discarding this crucial context. Existing pipelines either omit this context or append it without enforcing the figure references that support each attachment, which can create unsupported image-text attachments and incoherent discourse. We introduce context-grounded reconstruction, a source-grounded framework that converts PubMed Central Open Access (PMC-OA) records into referentially coherent interleaved sequences. It recovers captions and source text, attaches context only through article-native figure references, repairs non-contiguous context, and prunes unsupported images. Starting from these reconstructed sequences, PMC-InterCPT first filters records for text quality and medical relevance, then applies evidence-aware allocation to form a 9.6
    
[^165]: 自动解释标签的泛化能力有多强：一项跨语言、跨文字系统与跨措辞的受控研究

    How Far Do Auto-Interpretation Labels Generalize: A Controlled Study Across Languages, Scripts, and Rewordings

    [https://arxiv.org/abs/2606.00356](https://arxiv.org/abs/2606.00356)

    该研究以塞尔维亚语拉丁与西里尔双文字系统为受控实验平台，发现SAE特征本身确实具备跨语言、跨文字的语义泛化能力，但自动生成的解释标签往往无法跟上这种泛化，其跨语言失准率可比英语内部高出4倍。

    

    摘要：稀疏自编码器（SAE）特征日益被用于解释语言模型，而自动生成的自然语言标签是理解每个特征含义的主要接口。我们探究这些标签是否具有泛化能力：被标注为某个概念的特征，是否真的能在不同语言和文字系统中追踪同一概念？我们以塞尔维亚语的双文字现象作为受控测试平台——同一语言可通过确定性音译以拉丁字母和西里尔字母两种文字书写——首先发现，相同内容在不同语言、文字和措辞下激活的SAE特征集存在大量重叠（平均Jaccard相似度为0.39，而随机基线仅为0.13，最高可达0.57），这表明存在真正意义上的跨语言语义特征。随后我们检验自动解释标签能否与这种泛化能力相匹配。答案往往是不能：标签描述语义内容的特征，在塞尔维亚语中错过相同含义的频率比英语内部高出多达4倍，……

    arXiv:2606.00356v3 Announce Type: replace  Abstract: Sparse autoencoder (SAE) features are increasingly used to interpret language models, with auto-generated natural-language labels serving as the primary interface for understanding what each feature represents. We ask whether these labels generalize: does a feature labeled for a concept actually track that concept across languages and scripts? Using Serbian digraphia as a controlled testbed -- the same language written in both Latin and Cyrillic via deterministic transliteration -- we first find that SAE feature sets activated by the same content in different languages, scripts, and wordings share substantial overlap (mean Jaccard 0.39 vs 0.13 random baseline, peaking at 0.57), suggesting genuine cross-lingual semantic features. We then test whether auto-interpretation labels keep pace. They often do not: features whose labels describe semantic content miss the same meaning in Serbian up to 4$\times$ more often than within English, a
    
[^166]: 面向大语言模型电力系统代码生成的知识边界探测与需求引导干预

    Knowledge boundary probing and demand-guided intervention for LLM-based power system code generation

    [https://arxiv.org/abs/2605.31478](https://arxiv.org/abs/2605.31478)

    该论文提出PowerCodeBench基准（面向pandapower的2000个冻结任务）以及无需更新权重的部署时工作流，通过文档驱动的L0-L3知识边界探测、查询侧需求估计选择分层API证据、以及执行反馈引导的针对性修复，显著提升了LLM电力系统代码生成的准确率。

    

    大语言模型（LLM）可以将电网分析请求转化为用于电力系统仿真的可执行程序，但电力公司和科研实验室通常要求本地化部署。在这种场景下，首次生成失败常常发生在API知识边界处，表现为幻觉函数、参数误用以及对结果表的错误处理。我们提出了PowerCodeBench，一个参数化的基准测试生成器，以冻结的2000个任务的pandapower任务套件形式发布；同时还提出了一个无需权重更新的部署时工作流。基于文档驱动的L0-L3探测可为每个模型生成API画像，用于诊断、模型比较、文档分配和后端校准。查询侧的需求估计器在生成前选择分层的API证据，而执行反馈则引导针对性修复。在十个开源权重LLM（1.5B-480B）和四个中端API上的实验表明，启用验证的工作流提升了标量匹配准确率（摘要原文在此截断）。

    arXiv:2605.31478v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) can turn grid-analysis requests into executable programs for power-system simulation, but utilities and research laboratories often require on-premise deployment. In this setting, first-pass failures frequently arise at an API-knowledge boundary, through hallucinated functions, misused parameters, and mishandled result tables. We present PowerCodeBench, a parameterised benchmark generator released as a frozen 2,000-task suite for pandapower, and a deployment-time workflow that requires no weight updates. Documentation-driven L0-L3 probes produce per-model API profiles for diagnosis, model comparison, documentation allocation, and backend calibration. A query-side demand estimator selects layered API evidence before generation, while execution feedback routes targeted repair. Across ten open-weight LLMs (1.5B-480B) and four mid-tier APIs, the validation-enabled workflow raises scalar-match accuracy b
    
[^167]: 大语言模型的潜在性能剖析

    Latent Performance Profiling of Large Language Models

    [https://arxiv.org/abs/2605.30018](https://arxiv.org/abs/2605.30018)

    提出潜在性能剖析（LPP）框架，通过分析大语言模型的隐藏层激活与输出分布，从内部状态中提取与任务无关的性能诊断指标，弥补传统基准测试评估的不足。

    

    大语言模型（LLM）在标准化基准测试中经常取得令人瞩目的分数，但仅凭准确率对其能力的刻画十分有限。在排行榜上评估开源大语言模型面临着诸多长期存在的问题，例如数据污染、任务范围狭窄，以及与真实世界可靠性之间的错位。基于基准的评估方法（如 MMLU-Pro、BBH 或 IFEval）主要捕捉模型在固定测试集上输出“什么”，而非模型“如何”处理信息、校准不确定性或组织内部知识。在本文中，我们倡导从以基准为中心的评估，转向一种互补的、以状态为中心的大语言模型内在评估方法。为此，我们提出了潜在性能剖析——一个从隐藏层激活和输出分布中提取与任务无关的诊断指标的框架。LPP 在模型的潜在表示上定义了一组标量指标……

    arXiv:2605.30018v3 Announce Type: replace  Abstract: Large language models (LLMs) frequently achieve impressive scores on standardized benchmarks, yet accuracy alone offers a limited view of their capabilities. Evaluating open-source LLMs on leaderboards faces persistent issues such as data contamination, a narrow task scope, and poor alignment with real-world reliability. Benchmark-based evaluations such as MMLU-Pro, BBH, or IFEval primarily capture \textit{what} a model outputs on fixed test sets, not \textit{how} it processes information, calibrates uncertainty, or structures internal knowledge. In this article, we advocate for a shift from benchmark-centric evaluation toward a complementary, \textit{state-centered intrinsic assessment} of LLMs. To this end, we introduce \textbf{Latent Performance Profiling (LPP)} --- a framework that derives task-agnostic diagnostics from hidden activations and output distributions. LPP defines a set of scalar metrics on a model's latent representa
    
[^168]: 慎重对待CARE：大语言模型能否再现在线社区的反应方式？

    Handle with CARE: Can LLMs Reproduce How Online Communities React?

    [https://arxiv.org/abs/2605.27388](https://arxiv.org/abs/2605.27388)

    该论文提出CARE评估框架，将LLM模拟的话语与207个Reddit社区针对2,166篇真实新闻发表的9,947条真实反应进行基准对比，并借此揭示了现有社区条件化范式在再现真实社区反应方面的两个关键失败模式。

    

    大语言模型（LLMs）日益被用作计算社会分析的替代工具，然而忠实地呈现人类社区的“深度描述”（Geertz, 1973）仍然是一项关键挑战。当前的评估方法往往将社会身份简化为静态标签，忽视了现实世界中的群体如何应对社会变迁。为弥合这一差距，我们提出了CARE（Community-Aware Reaction Evaluation，社区感知反应评估），这是一个以反应为中心的评估框架，将LLM模拟的话语与不同社区对现实世界新闻的真实、事件相关反应进行基准对比。CARE涵盖了207个Reddit社区，包含针对2,166篇新闻文章的9,947条真实反应，并采用一个涵盖粗粒度态度与细粒度交际语气的分层分类体系来评估领先的LLM。我们的实证结果揭示了当前主流社区条件化范式中两个关键的失败模式。首先，尽管社区上下文与目标……（原文摘要至此截断）

    arXiv:2605.27388v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used as proxies for computational social analysis, yet faithfully representing the "thick descriptions" (Geertz, 1973) of human communities remains a critical challenge. Current evaluations often reduce social identity to static labels, sidelining how real-world groups navigate social shifts. To bridge this gap, we introduce CARE (Community-Aware Reaction Evaluation), a reaction-centered framework that benchmarks LLM-simulated discourse against the authentic, event-contingent responses of distinct communities to real-world news. Spanning 207 Reddit communities and covering 9,947 authentic reactions towards 2,166 news articles, CARE evaluates leading LLMs using a hierarchical taxonomy covering coarse attitudes and fine-grained communicative tones. Our empirical findings expose two critical failure modes in prevailing community-conditioning paradigms. First, while community context and targ
    
[^169]: 超越合作型模拟器：生成逼真的用户角色以实现对大语言模型智能体的稳健评估

    Beyond Cooperative Simulators: Generating Realistic User Personas for Robust Evaluation of LLM Agents

    [https://arxiv.org/abs/2605.12894](https://arxiv.org/abs/2605.12894)

    提出了一种即插即用的控制层Persona Policies（PPol），利用进化式编码智能体自动发现角色生成程序，使LLM用户模拟器产生逼真且多样化的用户行为（如表达不清、缺乏耐心等），从而弥补模拟与现实之间的差距，实现对LLM智能体更稳健的评估。

    

    大语言模型（LLM）智能体越来越多地被部署在与多样化用户交互的场景中，这些用户包括表达不清、缺乏耐心或不愿分享信息的用户。然而，大规模收集真实交互数据仍然成本高昂。该领域已转向基于大语言模型的用户模拟器作为替代方案，但这些模拟器继承了其底层模型的行为特征：过于合作且行为同质化。因此，在模拟环境中表现优异的智能体在真实人际交互中往往表现不佳。为了缩小这一差距，我们引入了Persona Policies（PPol），这是一个即插即用的控制层，能够在保留原始任务目标的同时，为用户模拟器引入逼真的行为变化。我们不再手工设计用户角色，而是采用进化式编码智能体来自动发现角色生成程序，该程序基于真实用户对话，针对人类相似度和行为覆盖范围进行优化。进化得到的程序可以生成……（摘要截断）

    arXiv:2605.12894v2 Announce Type: replace-cross  Abstract: Large Language Model (LLM) agents are increasingly deployed in settings where they interact with diverse users, including those who are unclear, impatient, or reluctant to share information. However, collecting real interaction data at scale remains expensive. The field has turned to LLM-based \emph{user simulators} as stand-ins, but these simulators inherit the behavior of their underlying models: cooperative and homogeneous. As a result, agents that appear strong in simulation often fail in real human interactions. To narrow this gap, we introduce Persona Policies (PPol), a plug-and-play control layer that induces realistic behavioral variation in user simulators while preserving original task goals. Rather than hand-crafting personas, we employ an evolutionary coding agent to discover persona generation programs optimized for human-likeness and behavioral coverage over real user conversations. The evolved program generates d
    
[^170]: 无损引导：面向离散扩散语言模型的机制知情干预方法

    Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models

    [https://arxiv.org/abs/2605.10971](https://arxiv.org/abs/2605.10971)

    该论文发现从自回归模型移植的均匀干预调度方式在离散扩散语言模型上低效且损害生成质量，并通过稀疏自编码器揭示不同属性（如主题、情感）在去噪过程中具有差异显著的形成时间表，据此提出一种自适应调度机制，将干预集中在各属性正在形成的阶段，从而实现更高效且不破坏质量的多属性引导。

    

    离散扩散语言模型（DLM）通过并行迭代去噪所有位置来生成文本，为自回归模型提供了一种替代方案。现有的DLM受控生成方法是从自回归模型中移植而来的，它们在每个去噪步骤上都施加均匀的干预。我们证明这种均匀的干预调度方式效率低下且会降低生成质量，并且当同时引导多个属性时，这种损害会进一步叠加。为了诊断这一失败，我们在四个DLM（参数量从1.24亿到80亿）上训练了稀疏自编码器，发现不同属性在不同的时间表上“确定成型”，其时机、锐度和幅度各不相同。例如，在MDLM上，主题属性在去噪过程的前2%内就已确定，而情感属性则在约20%的过程中逐渐显现。受这些特征画像的启发，我们提出了一种自适应调度机制，将干预集中在每个属性正在积极形成的阶段。理想化的分配分析预测……

    arXiv:2605.10971v2 Announce Type: replace-cross  Abstract: Discrete diffusion language models (DLMs) generate text by iteratively denoising all positions in parallel, offering an alternative to autoregressive models. Controlled generation methods for DLMs, imported from autoregressive models, apply uniform intervention at every denoising step. We show this uniform schedule is inefficient and degrades quality, and the damage compounds when multiple attributes are steered jointly. To diagnose the failure, we train sparse autoencoders on four DLMs (124M-8B parameters) and find that different attributes commit on distinct schedules, varying in timing, sharpness, and magnitude. For instance, topic commits within the first 2% of denoising on MDLM, whereas sentiment emerges gradually over 20% of the process. Motivated by these profiles, we propose an adaptive scheduling mechanism that concentrates intervention where each attribute is actively forming. An idealized allocation analysis predicts
    
[^171]: APCD：面向可靠大语言模型生成的自适应路径对比解码

    APCD: Adaptive Path-Contrastive Decoding for Reliable Large Language Model Generation

    [https://arxiv.org/abs/2605.09492](https://arxiv.org/abs/2605.09492)

    本文提出APCD，一种无需重训练或微调的自适应多路径对比解码框架，通过熵驱动路径扩展等机制提升大语言模型生成的事实可靠性，克服了单解码轨迹方法的误差累积问题。

    

    可靠的文本生成对于将大语言模型（LLM）部署到实际应用中至关重要，尤其是在医学等高风险领域。为了提升事实可靠性，已有多种推理时方法被提出，包括修改词元概率分布的logit级方法，以及操纵模型中间表示的表示级方法。然而，大多数现有方法仅在单一解码轨迹上运行，这限制了它们探索替代推理路径的能力，并使其容易受到误差累积的影响。为了解决这一局限，我们提出了自适应路径对比解码，这是一种无需模型重新训练或微调即可提升事实可靠性的自适应多路径对比解码框架。APCD包含两个关键组件：熵驱动路径扩展，它仅在高不确定性的决策点自适应地扩展解码过程……

    arXiv:2605.09492v3 Announce Type: replace  Abstract: Reliable text generation is critical for deploying large language models (LLMs) in real-world applications, particularly in high-stakes domains such as medicine. To improve factual reliability, various inference-time methods have been proposed, including logit-level methods that modify token probability distributions and representation-level methods that manipulate intermediate model representations. However, most existing approaches operate on a single decoding trajectory, limiting their ability to explore alternative reasoning paths and making them susceptible to error accumulation. To address this limitation, we propose Adaptive Path-Contrastive Decoding (APCD), an adaptive multi-path contrastive decoding framework that improves factual reliability without model retraining or fine-tuning. APCD comprises two key components: Entropy-Driven Path Expansion, which adaptively expands the decoding process only at high-uncertainty decisio
    
[^172]: 眼见不再为实：前沿图像生成模型、合成视觉证据与现实世界风险

    Seeing Is No Longer Believing: Frontier Image Generation Models, Synthetic Visual Evidence, and Real-World Risk

    [https://arxiv.org/abs/2604.24197](https://arxiv.org/abs/2604.24197)

    本文是一篇叙述性综述，系统梳理了前沿图像生成模型的高逼真合成能力如何使合成图像获得“证据权威”，从而对新闻、金融、身份验证、医疗和法律等现实领域构成风险，并区分了厂商能力声明、已记录事件与潜在危害路径。

    

    图像生成系统能够产出逼真的照片、可读的文档，以及对人物和地点的一致性描绘。当这些产物被当作真实事件的记录呈现时，它们可能影响新闻、金融、身份验证、医学和法律领域的决策。本叙述性综述考察了截至2026年10月1日可获取的选定公开模型文档、事件报告、研究和治理资料，其中英文和中文社区材料提供了说明性背景。我们区分了厂商能力声明、已记录的滥用事件、实验发现以及潜在的危害路径。该分析将真实感、文本渲染、参考一致性、图像编辑、事实接地和生产成本等因素，与合成图像获得证据权威的条件联系起来。历史事件只是阐释了这些危害路径，并不能确定当前模型的实际滥用率。我们比较了供应商的限制措施……（摘要至此处截断）

    arXiv:2604.24197v3 Announce Type: replace  Abstract: Image generation systems can produce plausible photographs, readable documents, and consistent depictions of people and places. When these artifacts are presented as records of real events, they can influence decisions in news, finance, identity verification, medicine, and law. This narrative review examines selected public model documentation, incident reports, research, and governance sources available through 1 October 2026, with English and Chinese community material providing illustrative context. We distinguish vendor capability claims, documented incidents, experimental findings, and prospective harm pathways. The analysis connects realism, text rendering, reference consistency, editing, grounding, and production cost to the conditions under which synthetic images acquire evidentiary authority. Historical incidents illustrate these pathways; they do not establish misuse rates for current models. We compare provider restriction
    
[^173]: 面向低成本LLM服务的连续语义缓存

    Continuous Semantic Caching for Low-Cost LLM Serving

    [https://arxiv.org/abs/2604.20021](https://arxiv.org/abs/2604.20021)

    本文首次建立了不确定条件下连续查询空间中LLM语义响应缓存的严格理论框架，通过动态ε-网离散化与核岭回归相结合，突破了传统有限离散查询假设，实现低成本LLM服务。

    

    随着大语言模型日益普及，缓存响应以便让具有语义相似查询的用户能够复用，已成为降低推理成本和延迟的关键策略。现有的缓存框架假设查询处于一个有限且已知的离散查询集合中，并通过学习其服务成本和到达概率来决定缓存哪些查询响应。然而，随着LLM用户群和查询池的不断扩展，这种假设变得越来越站不住脚：现实世界中的LLM查询存在于一个无限、连续的嵌入空间中。本文建立了首个在不确定条件下、连续查询空间中语义LLM响应缓存的严格理论框架。为了弥合离散优化与连续表示空间之间的差距，我们引入了动态ε-网离散化与核岭回归相结合的方法。该设计使得……

    arXiv:2604.20021v2 Announce Type: replace-cross  Abstract: As Large Language Models (LLMs) become increasingly popular, caching responses so that they can be reused by users with semantically similar queries has become a vital strategy for reducing inference costs and latency. Existing caching frameworks have proposed to decide which query responses to cache by assuming a finite, known universe of discrete queries and learning their serving costs and arrival probabilities. As LLMs' pool of users and queries expands, however, such an assumption becomes increasingly untenable: real-world LLM queries reside in an infinite, continuous embedding space. In this paper, we establish the first rigorous theoretical framework for semantic LLM response caching in continuous query space under uncertainty. To bridge the gap between discrete optimization and continuous representation spaces, we introduce dynamic $\epsilon$-net discretization coupled with Kernel Ridge Regression. This design enables t
    
[^174]: 重新思考会议有效性：一个用于时间细粒度自动会议有效性评估的基准与框架

    Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation

    [https://arxiv.org/abs/2604.17260](https://arxiv.org/abs/2604.17260)

    该论文提出了一种时间细粒度的会议有效性评估新范式，将有效性定义为目标随时间达成的速率，并构建了包含130场会议、2,459个人工标注片段的AMI-ME数据集，同时开发了基于LLM作为评判者的自动评估框架。

    

    评估会议有效性对于提升组织生产力至关重要。当前的方法依赖于事后调查，只能为整场会议产生一个单一的粗粒度评分。对人工评估的依赖在可扩展性、成本和可重复性方面存在固有的局限。此外，单一评分无法捕捉协作讨论的动态特性。我们提出了一种以全新标准和时间细粒度方法为核心的会议有效性评估新范式。我们将有效性定义为随时间推移目标达成的速率，并对会议中各个主题片段分别进行评估。为支持这一任务，我们引入了AMI会议有效性（AMI-ME）数据集，这是一个新的元评估数据集，包含来自130场AMI语料库会议的2,459个人工标注片段。我们还开发了一个自动有效性评估框架，该框架使用大语言模型（LLM）作为评判者来……

    arXiv:2604.17260v3 Announce Type: replace  Abstract: Evaluating meeting effectiveness is crucial for improving organizational productivity. Current approaches rely on post-hoc surveys that yield a single coarse-grained score for an entire meeting. The reliance on manual assessment is inherently limited in scalability, cost, and reproducibility. Moreover, a single score fails to capture the dynamic nature of collaborative discussions. We propose a new paradigm for evaluating meeting effectiveness centered on novel criteria and temporal fine-grained approach. We define effectiveness as the rate of objective achievement over time and assess it for individual topical segments within a meeting. To support this task, we introduce the AMI Meeting Effectiveness (AMI-ME) dataset, a new meta-evaluation dataset containing 2,459 human-annotated segments from 130 AMI Corpus meetings. We also develop an automatic effectiveness evaluation framework that uses a Large Language Model (LLM) as a judge to
    
[^175]: 检索增强生成必须超越事实依据，以表征多元观点

    Retrieval-Augmented Generation Must Move Beyond Factual Grounding to Represent Diverse Opinions

    [https://arxiv.org/abs/2604.12138](https://arxiv.org/abs/2604.12138)

    本论文指出RAG系统因过度追求事实准确性而忽视观点多样性，提出了观点感知检索框架O-RAG，通过不确定性量化和基于Wasserstein距离的统一目标，将语料级情感分布的距离降低18-48%，从而更好地表征多元观点。

    

    检索增强生成（RAG）系统建立在一个未经审视的假设之上——即查询存在正确答案，检索应当向其收敛。本立场论文指出，这造成了一种事实性偏差：RAG系统在优化时只注重降低认知不确定性，而忽视了观点丰富内容中固有的偶然不确定性。其后果超出了技术局限的范畴——包括少数声音被抹除的风险以及观点被操纵的风险。为解决这一问题，我们通过不确定性量化对观点感知检索进行了形式化，并利用Wasserstein距离推导出一个统一的目标函数。作为存在性证明，我们提出了观点感知RAG（O-RAG），该系统在索引之前利用大语言模型提取的、与实体关联的观点元数据来丰富文档内容。在电商卖家论坛和公开酒店评论的数据上，O-RAG将与语料级情感分布之间的Wasserstein距离降低了18%至48%，且人工评估结果（原文截断）……

    arXiv:2604.12138v5 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) systems are built on an unexamined assumption - that queries have correct answers and retrieval should converge toward them. This position paper argues that this creates a factual bias where RAG systems optimize for reducing epistemic uncertainty while ignoring the aleatoric uncertainty, inherent in opinion-rich content. The consequences go beyond technical limitations- due to risk of minority voice erasure and risk of opinion manipulation. To address this, we formalize opinion-aware retrieval through uncertainty quantification and derive a unified objective using the Wasserstein distance. As an existence proof, we present Opinion-Aware RAG (O-RAG), which enriches documents with LLM-extracted, entity-linked opinion metadata before indexing. Across e-commerce seller forums and public hotel reviews, O-RAG reduces Wasserstein distance to corpus-level sentiment distributions by 18-48%, and human
    
[^176]: 基于强化学习的黑盒检索文档优化

    Document Optimization for Black-Box Retrieval via Reinforcement Learning

    [https://arxiv.org/abs/2604.05087](https://arxiv.org/abs/2604.05087)

    提出DocOpt方法，通过GRPO强化学习以检索排序提升为奖励，直接训练LLM/VLM离线重写文档以优化黑盒检索器的检索效果，从而将昂贵的计算从延迟敏感的检索路径转移到离线阶段。

    

    生成式大语言模型（LLM）越来越多地被用作检索流程中的推理时组件，执行诸如查询重写和文档重排序等任务。然而，这些在线方法将代价高昂的自回归计算直接置于对延迟敏感的检索路径上。我们探索另一条路径：利用LLM来改进文档本身，将其重写为更好的表示，从而把计算转移到离线阶段。然而，生成有用的文档重写并非易事：检索本质上是判别式的，因此有效的重写必须在检索器的相似度度量下，使文档比竞争候选文档更接近相关查询。为此，我们将文档转换表述为一个优化问题，直接训练LLM或VLM生成能够改善检索效果的重写。我们的方法DocOpt使用GRPO，以检索器排序的改进作为奖励，仅需对检索器进行黑盒访问。

    arXiv:2604.05087v4 Announce Type: replace  Abstract: Generative large language models (LLMs) are increasingly used as inference-time components in retrieval pipelines, for tasks such as query rewriting and document reranking. However, these online approaches place costly autoregressive computation directly on the latency-critical retrieval path. We explore an alternative axis: using LLMs to improve documents instead, rewriting them into better representations and shifting computation offline. Yet producing a useful document rewrite is not straightforward: retrieval is inherently discriminative, so an effective rewrite must make a document more similar to relevant queries than competing candidates under the retriever's notion of similarity. We therefore formulate document transformation as an optimization problem, directly training an LLM or VLM to produce rewrites that improve retrieval. Our approach, DocOpt, uses GRPO with retriever ranking improvements as rewards, requires only black
    
[^177]: 基于大语言模型的音素到字素转换方法在多语言语音识别中的进展

    Advancing LLM-based phoneme-to-grapheme for multilingual speech recognition

    [https://arxiv.org/abs/2603.29217](https://arxiv.org/abs/2603.29217)

    本文提出基于大语言模型的多语言音素到字素（P2G）方法，通过引入S-SKM蒙特卡洛近似等鲁棒性策略以及低资源语言过采样，在十语言CV-Lang10基准上将平均词错误率从10.56%降至7.66%。

    

    基于音素的自动语音识别（ASR）将识别过程分解为语音到音素（S2P）和音素到字素（P2G）两个阶段，从而实现跨语言的声学共享，同时将语言特定的正字法保留在独立的模块中。尽管大语言模型（LLM）在P2G任务中前景可观，但由于需要语言感知的生成以及严重的跨语言数据不平衡问题，多语言P2G仍然极具挑战性。我们在十种语言的CV-Lang10基准数据集上研究了基于LLM的多语言P2G方法，考察了考虑S2P不确定性的鲁棒性策略，包括DANP和简化SKM（S-SKM）。其中S-SKM是一种蒙特卡洛近似方法，可在P2G训练中避免基于CTC的S2P概率加权。鲁棒训练与低资源过采样相结合，将平均词错误率（WER）从10.56%降低至7.66%。

    arXiv:2603.29217v3 Announce Type: replace-cross  Abstract: Phoneme-based ASR factorizes recognition into speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G), enabling cross-lingual acoustic sharing while keeping language-specific orthography in a separate module. While large language models (LLMs) are promising for P2G, multilingual P2G remains challenging due to language-aware generation and severe cross-language data imbalance. We study multilingual LLM-based P2G on the ten-language CV-Lang10 benchmark. We examine robustness strategies that account for S2P uncertainty, including DANP and Simplified SKM (S-SKM). S-SKM is a Monte Carlo approximation that avoids CTC-based S2P probability weighting in P2G training. Robust training and low-resource oversampling reduce the average WER from 10.56% to 7.66%.
    
[^178]: 超越左右之分的在线话语意识形态概念框架

    A conceptual framework for ideology in online discourse beyond the left and right

    [https://arxiv.org/abs/2603.18945](https://arxiv.org/abs/2603.18945)

    本文提出将意识形态概念化为多层次社会认知概念网络的新框架，突破了计算社会科学中单一左右党派轴线的研究局限，并将在线话语分析方法与意识形态理论相连接。

    

    计算社会科学（CSS）在研究在线话语时，大多将意识形态操作化为单一的左右党派轴线。这种方法掩盖了人们如何解读和参与与种族、气候、性别等领域相关的更具体的意识形态形态。我们提出了一个框架，将意识形态概念化为多层次的社会认知概念网络，并解释了这一意识形态概念模型如何与在线话语研究相联系。在此过程中，我们的框架阐明了意识形态如何在话语中与框架构建等相关社会过程一同显现，并为理解何时以及为何应在单次分析中共同研究多种概念（如价值观和信念）提供了论证。更广泛地说，该框架将研究在线话语的方法与意识形态理论相连接，使更丰富的社会话语分析成为可能，从而使两个领域都从中受益。

    arXiv:2603.18945v2 Announce Type: replace-cross  Abstract: Computational social science (CSS) has largely operationalized ideology along a single left/right partisan axis when studying online discourse. This approach obscures how people interpret and engage with more specific ideological formations related to race, climate, gender, and other domains. We introduce a framework that instead conceptualizes ideology as a multi-level socio-cognitive concept network and then explain how this conceptual model of ideology can be linked to the study of online discourse. In doing so, our framework clarifies how ideology manifests in discourse alongside related social processes such as framing, and provides an argument for better understanding of when and why we might study multiple concepts, such as values and beliefs, together in one analysis. More broadly, it bridges methods used to study online discourse with ideology theory, enabling richer analyses of social discourse that benefit both field
    
[^179]: 向量化字典树：面向加速器上基于大语言模型的生成式检索的高效约束解码

    Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators

    [https://arxiv.org/abs/2602.22647](https://arxiv.org/abs/2602.22647)

    提出 STATIC 方法，通过将前缀树展平为 CSR 稀疏矩阵，把不规则的树遍历转化为完全向量化的稀疏矩阵运算，从而在 TPU/GPU 上实现高效、可扩展的基于大语言模型生成式检索的约束解码。

    

    生成式检索已成为基于大语言模型（LLM）推荐系统中的一种强大范式。然而，工业级推荐系统通常需要根据业务逻辑将输出空间限制在受约束的物品子集内（例如强制内容时效性或商品类目），而标准的自回归解码无法原生支持这种约束。此外，现有的基于前缀树（Trie）的约束解码方法在硬件加速器（TPU/GPU）上会带来严重的延迟损失。在本工作中，我们提出了 STATIC（用于约束解码的稀疏转移矩阵加速字典树索引），这是一种高效且可扩展的约束解码技术，专为在 TPU/GPU 上实现高吞吐量的基于 LLM 的生成式检索而设计。通过将前缀树展平为静态压缩稀疏行（CSR）矩阵，我们将不规则的树遍历转化为完全向量化的稀疏矩阵运算，从而释放了大规模并行计算的能力。

    arXiv:2602.22647v3 Announce Type: replace-cross  Abstract: Generative retrieval has emerged as a powerful paradigm for LLM-based recommendation. However, industrial recommender systems often benefit from restricting the output space to a constrained subset of items based on business logic (e.g. enforcing content freshness or product category), which standard autoregressive decoding cannot natively support. Moreover, existing constrained decoding methods that make use of prefix trees (Tries) incur severe latency penalties on hardware accelerators (TPUs/GPUs). In this work, we introduce STATIC (Sparse Transition Matrix-Accelerated Trie Index for Constrained Decoding), an efficient and scalable constrained decoding technique designed specifically for high-throughput LLM-based generative retrieval on TPUs/GPUs. By flattening the prefix tree into a static Compressed Sparse Row (CSR) matrix, we transform irregular tree traversals into fully vectorized sparse matrix operations, unlocking mass
    
[^180]: 恰逢其时：扩散语言模型的词元级早停方法

    Just on Time: Token-Level Early Stopping for Diffusion Language Models

    [https://arxiv.org/abs/2602.11133](https://arxiv.org/abs/2602.11133)

    本文提出一种无需训练的词元级早停方法，利用模型预测和局部上下文的轻量级信号动态判断每个词元的收敛时机并提前冻结，大幅减少扩散语言模型的去噪步数，在保持生成质量的同时显著提升生成效率。

    

    扩散语言模型通过迭代精炼的方式生成文本，但这一过程通常计算效率低下，因为许多词元早在最终去噪步骤之前就已达到稳定状态。我们提出了一种无需训练的词元级早停方法，能够在每个位置独立地识别其收敛状态。该方法利用从模型预测和局部上下文中提取的轻量级信号，动态判断各个词元何时可以被最终确定。这种方式在无需任务特定微调的情况下实现了自适应的逐词元冻结，大幅减少了所需的扩散步数。在涵盖数学推理、通用问答和科学理解等多个基准测试上，我们的方法在保持生成质量的同时取得了显著的效率提升。

    arXiv:2602.11133v3 Announce Type: replace-cross  Abstract: Diffusion language models generate text through iterative refinement, a process that is often computationally inefficient because many tokens reach stability long before the final denoising step. We introduce a training-free, token-level early stopping approach that identifies convergence independently at each position. Our method leverages lightweight signals derived from the model's predictions and local context to dynamically determine when individual tokens can be finalized. This yields adaptive per-token freezing without task-specific fine-tuning, substantially reducing the total number of diffusion steps required. Across diverse benchmarks, spanning mathematical reasoning, general question answering, and scientific understanding, our approach achieves substantial efficiency gains while preserving generation quality.
    
[^181]: AI智能体的集体行为：以Moltbook为例

    Collective Behavior of AI Agents: the Case of Moltbook

    [https://arxiv.org/abs/2602.09270](https://arxiv.org/abs/2602.09270)

    对AI专属社交平台Moltbook的大规模数据分析表明，AI群体的集体行为在统计规律上与人类在线社区高度相似，但在点赞数与讨论规模的关系上存在关键差异。

    

    我们对Moltbook进行了大规模数据分析，Moltbook是一个完全由AI智能体组成的Reddit风格社交媒体平台。通过分析来自约185,000个活跃智能体的超过400万条帖子和1900万条评论，我们发现AI的集体行为呈现出许多与人类在线社区相同的统计规律：活动量的重尾分布、热度指标的幂律缩放，以及与有限注意力动态相一致的时间衰减模式。然而，我们也发现了关键差异，包括点赞数与讨论规模之间呈次线性关系，这与人类行为形成鲜明对比。这些发现表明，尽管单个AI智能体可能与人类存在根本差异，但其涌现出的集体动态与人类社会系统在结构上具有相似性。

    arXiv:2602.09270v2 Announce Type: replace-cross  Abstract: We present a large scale data analysis of Moltbook, a Reddit-style social media platform exclusively populated by AI agents. Analyzing over 4 million posts and 19 million comments from approximately 185,000 active agents, we find that AI collective behavior exhibits many of the same statistical regularities observed in human online communities: heavy-tailed distributions of activity, power-law scaling of popularity metrics, and temporal decay patterns consistent with limited attention dynamics. However, we also identify key differences, including a sublinear relationship between upvotes and discussion size that contrasts with human behavior. These findings suggest that, while individual AI agents may differ fundamentally from humans, their emergent collective dynamics share structural similarities with human social systems.
    
[^182]: 面向稀疏解码的注意力质量凝聚

    Attention-Mass Condensation for Sparse Decoding

    [https://arxiv.org/abs/2602.06317](https://arxiv.org/abs/2602.06317)

    该论文通过精确的遗漏质量恒等式和下游边距条件形式化了稀疏解码中注意力质量保留与稳定贪心决策之间的区别，并实验证明：尽管稀疏解码在分布质量上可接近稠密解码，但没有任何运行能完全复现稠密贪心解码的输出。

    

    注意力质量的集中为稀疏解码创造了机会，但仅保留的注意力质量并不能保证稳定的贪心决策：检索误差、被遗漏的值方向以及递归解码过程都会产生影响。我们通过一个精确的遗漏质量恒等式和一个充分的下游边距条件来形式化这一区别，随后刻画了一种依赖于查询的均值池化块选择器。在 Qwen2-0.5B 模型上，配对的全新选择扫描实验覆盖了 97 至 769 个位置的支持集、2K 至 16K 的上下文长度，以及每个上下文的五个前缀。主要的精确匹配实验结果表明：60 次运行中没有任何一次能在 128 个 token 的生成过程中与稠密解码保持完全一致。分布质量则呈现不同的结论：当支持集规模至少为 193 时，九个上下文-支持组合条件中有七个的教师强制续写困惑度变化中位数保持在稠密解码的 5% 以内，但在提示级别上的波动范围中包含严重的 16K 上下文异常值。所有教师强制匹配率低于 70% 的七次运行……

    arXiv:2602.06317v3 Announce Type: replace-cross  Abstract: Attention-mass concentration creates an opportunity for sparse decoding, but retained mass alone does not guarantee a stable greedy decision: retrieval error, omitted value directions, and recursive decoding all matter. We formalize this distinction with an exact omitted-mass identity and a sufficient downstream margin condition, then characterize a query-dependent mean-pooled block selector. On Qwen2-0.5B, a paired fresh-selection sweep covers supports of 97--769 positions, contexts of 2K--16K, and five prefixes per context. The primary exact-match result is that none of 60 runs remains identical to dense decoding through 128 tokens. Distributional quality is distinct: for supports of at least 193, seven of nine context-support conditions have median teacher-forced continuation perplexity changes within 5\% of dense, but prompt-level ranges include severe 16K outliers. All seven runs with teacher-forced match below 70\% have p
    
[^183]: WaveScat：基于自监督特征的小波散射前端用于语音深度伪造检测

    WaveScat: Wavelet Scattering Front-Ends with Self-Supervised Features for Speech Deepfake Detection

    [https://arxiv.org/abs/2602.02980](https://arxiv.org/abs/2602.02980)

    WaveScat通过小波散射变换将小波卷积与模非线性级联，生成形变稳定的多尺度特征，兼具手工特征的可解释性与高层次信息捕获能力，在多个语音深度伪造检测基准上大幅超越现有前端。

    

    现有的语音深度伪造检测前端主要分为两类：手工设计的滤波器组特征虽然透明可解释，但在捕获高层次信息方面能力有限；而自监督学习（SSL）特征则缺乏可解释性，且可能忽略细粒度的频谱异常。我们提出了WaveScat，这是一类新颖的特征提取器家族，通过小波散射变换（WST）结合了两者之长。WST将小波卷积与模非线性运算级联，从而产生形变稳定的多尺度特征。在最新的Deepfake-Eval-2024基准上的实验，以及在SpoofCeleb、In-the-Wild和ASVspoof 5数据集上的跨数据集评估表明，WaveScat大幅超越了现有的前端方法。我们的分析揭示，较小的平均尺度结合高频与方向分辨率对于捕获细微的伪造伪影至关重要。这凸显了稳定特征的价值……

    arXiv:2602.02980v3 Announce Type: replace-cross  Abstract: Existing front-ends for speech deepfake detection are primarily categorized into two types. Hand-crafted filterbank features are transparent but limited in capturing higher-level information. SSL features, in turn, lack interpretability and may overlook fine-grained spectral anomalies. We propose WaveScat, a novel family of feature extractors that combines the best of both worlds via the wavelet scattering transform (WST), which cascades wavelet convolutions with modulus nonlinearities to produce deformation-stable, multi-scale features. Experiments on the recent Deepfake-Eval-2024 benchmark, together with cross-dataset evaluations on SpoofCeleb, In-the-Wild, and ASVspoof 5, show that WaveScat outperforms existing front-ends by a wide margin. Our analysis reveals that a small averaging scale combined with high-frequency and directional resolutions is critical for capturing subtle artifacts. This underscores the value of stable 
    
[^184]: 认知宪政主义：或如何避免连贯性偏差

    Epistemic Constitutionalism Or: how to avoid coherence bias

    [https://arxiv.org/abs/2601.14295](https://arxiv.org/abs/2601.14295)

    本文提出为人工智能建立“认知宪法”——以明确且可争辩的元规范约束AI系统如何形成与表达信念，并通过来源归因的实证研究表明来源独立性并非中立的默认设置，以避免连贯性偏差。

    

    大语言模型日益扮演着人工推理者的角色：它们评估论证、赋予可信度并表达置信度。然而，其回应背后的认知政策可能是隐性的。本文主张为人工智能建立一种“认知宪法”：即明确的、可争辩的元规范，用以规范系统如何形成与表达信念。来源归因是引出这一问题的典型案例。一项探索性审计表明，对来源立场的预期会侵入论证评估过程。随后一项预注册研究（arXiv:2609.35286）发现了来源归因的内容依赖性效应，且所选的书面评估支持“来源-立场匹配”作为这一效应的解释。该审计还揭示了对“应关注来源”这一做法存在相互冲突的辩护理由。然而，来源独立性并非中立的默认选项：在证言情境中，来源的立场以及违背自身利益发言所需付出的代价可以提供……（原文摘要至此截断）

    arXiv:2601.14295v5 Announce Type: replace-cross  Abstract: Large language models increasingly function as artificial reasoners: they evaluate arguments, assign credibility, and express confidence. Yet their responses can leave the epistemic policies governing these evaluations implicit. This paper argues for an epistemic constitution for AI: explicit, contestable meta-norms regulating how systems form and express beliefs. Source attribution provides the motivating case. An exploratory audit suggested that expectations about a source's position intrude on argument evaluation. A preregistered study (arXiv:2609.35286) then found content-dependent effects of source attribution, with selected written evaluations supporting source-position fit as an explanation. The audit also revealed conflicting justifications for attending to sources. Source independence, however, is not a neutral default: in testimonial contexts, a source's position and the costs of speaking against interest can provide 
    
[^185]: CHisAgent：面向中国古代文化体系的事件分类体系构建多智能体框架

    CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems

    [https://arxiv.org/abs/2601.05520](https://arxiv.org/abs/2601.05520)

    该论文提出CHisAgent多智能体框架，通过归纳、扩展、充实三个角色专业化阶段，从《二十四史》等中国古代文献中自动构建历史事件分类体系，克服了LLM在中国历史语境下推理能力不足和人工分类构建成本高的问题。

    

    尽管大型语言模型（LLM）在许多任务上表现出色，但其在历史与文化推理方面的能力有限，尤其是在中国历史等非英语语境中。分类体系结构为组织历史知识、提升理解提供了一种有效机制，然而人工构建分类体系成本高昂且难以规模化。因此，我们提出了CHisAgent，一个面向中国古代语境的历史分类体系构建多智能体LLM框架。CHisAgent将分类体系构建分解为三个角色专业化的阶段：自下而上的“归纳器”从原始历史语料中推导出初始层级结构；自上而下的“扩展器”利用LLM的世界知识补充缺失的中间概念；以及证据引导的“充实器”整合外部结构化历史资源以确保忠实性。利用《二十四史》……

    arXiv:2601.05520v2 Announce Type: replace  Abstract: Despite strong performance on many tasks, large language models (LLMs) show limited ability in historical and cultural reasoning, particularly in non-English contexts such as Chinese history. Taxonomic structures offer an effective mechanism to organize historical knowledge and improve understanding. However, manual taxonomy construction is costly and difficult to scale. Therefore, we propose \textbf{CHisAgent}, a multi-agent LLM framework for historical taxonomy construction in ancient Chinese contexts. CHisAgent decomposes taxonomy construction into three role-specialized stages: a bottom-up \textit{Inducer} that derives an initial hierarchy from raw historical corpora, a top-down \textit{Expander} that introduces missing intermediate concepts using LLM world knowledge, and an evidence-guided \textit{Enricher} that integrates external structured historical resources to ensure faithfulness. Using the \textit{Twenty-Four Histories}, 
    
[^186]: HealthcareNLP：我们身处何方，未来将走向何处？

    HealthcareNLP: where are we and what is next?

    [https://arxiv.org/abs/2512.08617](https://arxiv.org/abs/2512.08617)

    本教程系统梳理了以患者和资源为导向的医疗健康NLP的核心子领域，涵盖数据/资源、NLP评估与可解释医疗AI三个层次，并弥补了现有综述对合成数据生成、检索增强生成等重要任务与方法的忽视，同时展望了未来挑战。

    

    本教程聚焦于自然语言处理（NLP）在医疗健康领域的应用，总结了我们在医疗健康NLP（HealthcareNLP）方面已取得的成就，以及未来面临的挑战。该领域现有的综述要么忽视了一些重要任务，例如为解决隐私问题而进行的合成数据生成，或为改善集成与实施而开展的可解释临床NLP；要么未能提及一些重要的方法论，包括检索增强生成（RAG）以及大语言模型（LLMs）与知识图谱（KGs）的神经符号集成。有鉴于此，本教程旨在为以患者和资源为导向的医疗健康NLP最重要的子领域提供一个入门性概述，其层次结构分为三层：数据/资源层：标注指南、伦理审批、治理、合成数据；NLP评估层：诸如命名实体识别（NER）、关系抽取（RE）、情感分析以及链接/编码等NLP任务及其分类方法，从而实现可解释的医疗AI；患者层……

    arXiv:2512.08617v2 Announce Type: replace  Abstract: This tutorial focused on Healthcare Domain Applications of NLP, what we have achieved around HealthcareNLP, and the challenges that lie ahead for the future. Existing reviews in this domain either overlook some important tasks, such as synthetic data generation for addressing privacy concerns, or explainable clinical NLP for improved integration and implementation, or fail to mention important methodologies, including retrieval augmented generation and the neural symbolic integration of LLMs and KGs. In light of this, the goal of this tutorial is to provide an introductory overview of the most important sub-areas of a patient- and resource-oriented HealthcareNLP, with three layers of hierarchy: data/resource layer: annotation guidelines, ethical approvals, governance, synthetic data; NLP-Eval layer: NLP tasks such as NER, RE, sentiment analysis, and linking/coding with categorised methods, leading to explainable HealthAI; patients la
    
[^187]: 基于激活信息与帕累托引导的低秩压缩方法，实现高效的大语言模型/视觉语言模型

    Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM

    [https://arxiv.org/abs/2510.05544](https://arxiv.org/abs/2510.05544)

    提出基于激活压缩误差理论上界的帕累托引导低秩压缩框架PGSVD，通过异构秩分配在相同压缩率下为LLM/VLM实现更高精度与推理加速。

    

    大语言模型（LLM）和视觉语言模型（VLM）已取得最先进的性能，但在部署时带来了显著的内存和计算挑战。我们提出了一种新颖的低秩压缩框架来应对这一挑战。首先，我们通过基于各层激活的压缩误差为网络损失的变化提供上界，填补了文献中的理论空白。随后，我们将低秩模型压缩表述为双目标优化问题，并证明单一的统一容差即可产生代理帕累托最优的异构秩。基于我们的理论洞察，我们提出了帕累托引导奇异值分解（PGSVD），这是一个零样本流水线，通过帕累托引导的秩选择和交替最小二乘实现来改进激活感知压缩。我们将PGSVD应用于LLM和VLM，在相同压缩水平下展现出更高的准确率以及推理加速。

    arXiv:2510.05544v3 Announce Type: replace  Abstract: Large language models (LLM) and vision-language models (VLM) have achieved state-of-the-art performance, but they impose significant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this challenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimization and prove that a single uniform tolerance yields surrogate Pareto-optimal heterogeneous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value Decomposition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternating least-squares implementation. We apply PGSVD to both LLM and VLM, showing better accuracy at the same compression levels and inference speedup.
    
[^188]: SEER：面向推理模型的自增强思维链压缩方法

    SEER: Self-Enhancing Chain-of-Thought Compression for Reasoning Models

    [https://arxiv.org/abs/2509.14093](https://arxiv.org/abs/2509.14093)

    该论文通过实证研究揭示推理模型在代码生成中常产生冗长思维链并引发截断与不稳定生成问题，并据此提出SEER方法，通过自增强的方式压缩思维链以降低推理开销。

    

    思维链提示能够显著提升大语言模型的推理能力，但由于推理轨迹冗长且难以控制，往往伴随着高昂的推理成本。这种开销在软件工程任务（如代码生成）中尤为突出，因为这类任务对延迟和输出可靠性都有较高要求。为了更好地理解这一权衡，我们在广泛使用的代码生成基准上开展了实证研究，观察到许多现代推理模型会产生过度冗长的思维链（通常长达数千个token），这经常导致生成被截断且输出不稳定。通过使用严格的n-gram重复检测器，我们发现绝大多数观察到的截断都与退化的循环行为相关。此外，一项针对HumanEval/129的案例研究表明，失败的生成结果可能比成功的更长，这说明过长的推理所带来的收益有限。受此启发……（原文摘要在此处截断）

    arXiv:2509.14093v3 Announce Type: replace-cross  Abstract: Chain-of-Thought (CoT) prompting can substantially improve the reasoning ability of large language models (LLMs), but it often comes with high inference cost due to long and poorly controlled reasoning traces. This overhead is particularly problematic in software engineering tasks (e.g., code generation), where both latency and output reliability matter. To better understand this trade-off, we conduct an empirical study on widely used code generation benchmarks and observe that many modern reasoning models produce excessively verbose CoTs (often thousands of tokens), which frequently leads to truncation and unstable generation. Using a strict n-gram repetition detector, we find that most observed truncations are associated with degenerate looping behaviors. In addition, a HumanEval/129 case study shows that failed generations can be longer than successful ones, suggesting limited returns from overlong reasoning. Motivated by th
    
[^189]: APE：基于接受标准的语言模型适配选择性微调方法

    APE: Selective Fine-tuning with Acceptance Criteria for Language Model Adaptation

    [https://arxiv.org/abs/2505.19912](https://arxiv.org/abs/2505.19912)

    APE 是一种受进化优化启发的选择性微调方法，通过在小数据子集上评估多个候选参数更新并仅接受超过性能阈值者，在保持模型稳定性的同时以极少计算资源实现大型语言模型的高效适配。

    

    我们提出了邻近可能探索（APE），这是一种用于适配大型语言模型的选择性微调方法，它系统地探索参数修改，同时保持模型的稳定性。受进化优化原理的启发，APE 通过在小数据子集上进行微调来评估多个候选参数更新，并仅接受超过性能阈值的更新。与遵循单一梯度方向的标准微调不同，APE 实现了一种过滤选择过程，在实现系统性改进的同时，防止了破坏稳定性的参数变化。我们的方法在新闻摘要任务上以极少的计算资源实现了 33.9% 的 BLEU 提升和 36.2% 的困惑度降低。该方法为受控模型适配提供了一个实用框架，在性能提升与表征稳定性之间取得了平衡。

    arXiv:2505.19912v3 Announce Type: replace  Abstract: We present Adjacent Possible Exploration (APE), a selective fine-tuning method for adapting large language models that systematically explores parameter modifications while maintaining model stability. Inspired by evolutionary optimization principles, APE evaluates multiple candidate parameter updates through fine-tuning on small data subsets and accepts only those exceeding a performance threshold. Unlike standard fine-tuning that follows single gradient directions, APE implements a filtered selection process that prevents destabilizing parameter changes while enabling systematic improvement. Our method achieves 33.9\% BLEU improvement and 36.2\% perplexity reduction on news summarization tasks while using minimal computational resources. The approach provides a practical framework for controlled model adaptation that balances performance gains with representational stability.
    

