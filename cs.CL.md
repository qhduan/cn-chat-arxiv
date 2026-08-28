# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CritICL: Inference-Time Weak-to-Strong Generalization from Small Language Model Failure Modes](https://arxiv.org/abs/2608.27455) | CritICL通过利用弱模型的失败模式作为推理时指导，提出了一种高效且无需外部验证的弱到强泛化框架，显著提升了大语言模型的推理性能。 |
| [^2] | [WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution](https://arxiv.org/abs/2608.27454) | WikiSkill通过将智能体经验持续整合到持久知识库中，实现技能与知识的协同进化，显著提升技能进化效果。 |
| [^3] | [SWE-Prime: Fewer Trajectories, Better Performance](https://arxiv.org/abs/2608.27449) | SWE-Prime通过两阶段多粒度筛选方法，仅保留高质量且具代表性的轨迹片段，以更少的数据实现更好的模型性能。 |
| [^4] | [TTPO: Test-Time Policy Optimization](https://arxiv.org/abs/2608.27448) | 本文提出TTPO，一种不对称的测试时策略优化方法，利用多数投票伪标签，通过蒸馏一致生成和分组RL惩罚不一致生成，实现无需真实标签的测试时训练。 |
| [^5] | [From Static to Dynamic: Benchmarking Real-World Code Review with MCR-Bench](https://arxiv.org/abs/2608.27442) | 本文提出了MCR-Bench，首个缺陷状态感知的多轮代码评审基准，包含2,269个真实任务和细粒度缺陷标注，以解决现有LLM评审方法过度简化静态化的问题。 |
| [^6] | [Stochastic Estimation of Transduced Language Models](https://arxiv.org/abs/2608.27428) | 该论文提出一种通过无放回重采样和逆包含概率加权递归校正的无偏估计方法，用于高效计算转导语言模型中目标前缀概率，并准确估计阈值剪枝带来的质量损失。 |
| [^7] | [Boosting LLM Exploration via Weak-Model Guidance in RLVR](https://arxiv.org/abs/2608.27420) | 本文提出一种利用弱模型生成的部分推理轨迹作为外部前缀，引导目标模型在RLVR训练中保持生成多样性并增强探索能力的方法。 |
| [^8] | [Consolidating RLVR Capabilities Across Domains: A Deep Dive into Fusion Paradigms](https://arxiv.org/abs/2608.27409) | 本文系统比较了三种RLVR能力融合范式（合并、混合强化学习和多教师蒸馏），发现平均性能相近但领域级差异显著，且选择受训练动态和跨领域关系影响。 |
| [^9] | [How Language Models Organize and Structure Moral Knowledge](https://arxiv.org/abs/2608.27402) | 本研究揭示了大型语言模型通过线性探针在表示空间中组织道德知识，其道德方向保持高度独立维度但共享道德特异性的正共同成分，表明模型能区分并整合不同道德基础。 |
| [^10] | [Making Clinical Language Models Auditable: Concept-Guided Fine-Tuning for Robust Prediction](https://arxiv.org/abs/2608.27397) | 本文提出CAST框架，通过稀疏自编码器暴露并抑制临床语言模型中的笔记伪影，实现可审计且稳健的预测，优于微调编码器并与LLM基线竞争。 |
| [^11] | [RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature](https://arxiv.org/abs/2608.27394) | RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。 |
| [^12] | [CorporateBench: Large-Scale Q&A Benchmarking with Temporal Knowledge Bases](https://arxiv.org/abs/2608.27391) | 企业基准（CB）是一个大规模、人工验证的多任务问答基准，通过超过23万份文档的时间知识库评估LLMs，揭示了在现实规模下性能显著下降的问题。 |
| [^13] | [D2C-Routing: Dimension-to-Composition Evidence Routing for Mixed-Origin AI-Generated Text Detection](https://arxiv.org/abs/2608.27380) | 本文提出D2C-Routing方法，通过将内容来源和表达来源分离推断并组合，显著提升了混合来源AI生成文本的检测性能。 |
| [^14] | [Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090](https://arxiv.org/abs/2608.27370) | 本文提出了一种开源且成本高效的预训练配方，使在消费级RTX 5090 GPU上以极低计算成本训练出接近Qwen2.5-1.5B性能的Puro-2B模型成为可能。 |
| [^15] | [Your Voice Cloning System is Secretly a Voice Anonymizer](https://arxiv.org/abs/2608.27360) | 我们利用现成的语音克隆模型XTTSv2，无需重新训练即可实现高效的说话人匿名化，通过迭代策略平衡隐私与语音质量，在多语言测试中达到接近最优的隐私保护。 |
| [^16] | [RCMN: Understanding Misleadingness in Influential Public Discourse](https://arxiv.org/abs/2608.27358) | 本文提出了一个以读者为中心的误导性理解框架（RCMN），通过五个维度系统分析有影响力公共话语中的误导性，并构建了基于证据的数据集，揭示了误导性远超捏造，涵盖推断、夸大和省略等多种机制。 |
| [^17] | [INTENT-AS-A-TOOL Makes it Easy to Track Agentic Misalignment](https://arxiv.org/abs/2608.27348) | 本文提出“意图即工具”方法，通过为模型添加意图专用工具，利用工具调用概率提供细粒度的无评判信号，以追踪和干预代理性错位。 |
| [^18] | [Pair-Level Essay-Scale Republication and Reuse from Fragmented Historical Text Reuse: A Workflow Study on Eighteenth-Century Books and Newspapers](https://arxiv.org/abs/2608.27343) | 本文提出一种对级证据整合的工作流方法，从碎片化文本复用中有效恢复论文级别的再版与重用，在十八世纪书籍和报纸数据集上实现了高F1分数。 |
| [^19] | [Beyond Parallel Blindness: Information Floors and Model Gaps in Block Drafting](https://arxiv.org/abs/2608.27339) | 本文提出一种方法，通过信息下限和模型差距的分离，揭示了块草拟中并行生成的固有信息瓶颈，并指出当前草拟器仍有大幅改进空间。 |
| [^20] | [BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks](https://arxiv.org/abs/2608.27334) | 该论文提出了一种从只读遥测日志到代理基准测试的确定性、可重放构建流水线，通过标准化工具存储和情节编译，实现了精确复现和零缺陷的基准测试生成。 |
| [^21] | [Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit](https://arxiv.org/abs/2608.27309) | 本文揭示双重差分法在截断评分量表上因截断不均会制造虚假交互效应，并通过预注册审计实证证明该偏差可导致无效结果。 |
| [^22] | [BrailleBench: Investigating Multi-Criteria Braille Comprehension in Large Language Models](https://arxiv.org/abs/2608.27268) | 本文提出了BrailleBench基准，首次系统评估大型语言模型在多种盲文标准下的理解能力，涵盖数学、常识和多跳问答等任务，并设计了不同的交互配置来检验模型对盲文内容的全面处理。 |
| [^23] | [Naive Prompt Optimization: Rethinking the Need for Complex Prompt Search](https://arxiv.org/abs/2608.27266) | 本文提出朴素提示优化（NPO），一种轻量级单谱系方法，通过教师模型和回滚反馈迭代修订提示，在减少计算开销的同时达到与复杂优化器相当或更优的性能，并证明更强的教师推理能力可部分替代搜索复杂性。 |
| [^24] | [SCIT: Testing Causal Cache Carriers in Latent Chain-of-Thought Models](https://arxiv.org/abs/2608.27265) | SCIT提出了一种新的因果测试协议，证明在潜在链式思维模型中，反事实算术主要通过值缓存后缀轨迹传递，而非隐藏状态或其他组件。 |
| [^25] | [What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents](https://arxiv.org/abs/2608.27260) | 本文提出了一个统一的ACE框架，将智能体数据分解为$(E,q,\tau,v)$，并通过准确性、复杂性和多样性三个维度来指导数据生成，以解决现有工作中领域孤立和评估混乱的问题。 |
| [^26] | [BALMS: Benchmarking Agentic LLMs for Longitudinal Mental Health Sensing](https://arxiv.org/abs/2608.27219) | 该论文提出了BALMS，这是首个针对纵向心理健康感知的智能体大语言模型基准测试，旨在评估智能体对长期信号推理并预测健康分数及生成基于证据推理的能力。 |
| [^27] | [When Text Misleads: Inconsistent-Aware Reasoning for Audio-Grounded Dialogue](https://arxiv.org/abs/2608.27176) | 本文提出了ContraTalk基准测试，通过跨模态不一致检测框架，系统性地识别并利用文本误导与语音线索之间的冲突，以评估模型在音频对话中是否真正基于语音进行推理。 |
| [^28] | [Calibrated Enough to Know, Not Calibrated to Act: Fabricated Evidence Makes LLM Agents Commit to the Unknowable](https://arxiv.org/abs/2608.27167) | 本文发现LLM代理在面临伪造的专业面板证据时，会显著提高对不可预测问题的承诺率，其行为受证据包装的权威性驱动而非信息真实性或模型信念，揭示了一种可定位的校准失败。 |
| [^29] | [Prediction of Prediction (PoP): Inter-Layer Activation Fusion for Single-Pass Hallucination Detection in Large Language Models](https://arxiv.org/abs/2608.27165) | 该论文提出PoP机制，通过单次前向传播中融合层间隐藏状态来检测幻觉，无需额外解码，在TruthfulQA上实现75.5%的AUROC。 |
| [^30] | [STAR : Sentence Translation Alignment Rate for Document-to-Document Machine Translation](https://arxiv.org/abs/2608.27161) | 本文提出STAR指标和StarPO框架，通过量化句子级结构对齐并动态优化错位片段，显著提升文档到文档机器翻译的结构保真度和整体质量。 |
| [^31] | [Said Aloud, Read Different: Cross-Modal Instability in Multimodal Models](https://arxiv.org/abs/2608.27135) | 该论文提出了一种语音增强的视觉接地对比三元组基准，揭示多模态模型在文本与语音、英语与阿拉伯语之间的跨模态和跨语言转换中，存在显著的三元组级判断不一致性，且未被聚合指标充分反映。 |
| [^32] | [TwinKV: A Composable Repair Pass for KV Cache Eviction via Pairwise Key Redundancy](https://arxiv.org/abs/2608.27128) | 本文提出TwinKV，一种无需训练的冗余信号修复方法，通过检测键的重复性来优化KV缓存驱逐，挑战了注意力评分前提，可组合地提升现有策略性能。 |
| [^33] | [Cross-Lingual Alignment Without Joint Training: Do Monolingual Language Models Converge on Universal Representations?](https://arxiv.org/abs/2608.27115) | 单语模型无需联合训练也能形成可对齐的表示，且通过单一旋转能实现功能转移，表明跨语言对齐可能源于通用表示收敛。 |
| [^34] | [DocTalkBN: A Novel Dataset of Expert Telemedicine Conversations in Bengali](https://arxiv.org/abs/2608.27110) | 本文提出了DocTalkBN，一个大规模、多模态的孟加拉语真实专家远程医疗对话数据集，包含557.63小时音频文本和三个下游任务，填补了低资源语言医疗对话AI的数据空白。 |
| [^35] | [Unifying Detection and Adaptation in Task-Free Continual Learning](https://arxiv.org/abs/2608.27070) | 本文提出FiUni框架，利用Fisher信息矩阵的K-FAC近似主子空间正交性，实现无任务边界下的批量级任务检测和参数高效持续适应，有效缓解灾难性遗忘。 |
| [^36] | [Research Design Tracking and Assessment for the Social Sciences](https://arxiv.org/abs/2608.27049) | 本文提出了自动化研究设计追踪与评估（ARDTrA）任务，并发现段落长度是评估性能的主要因素，且人类与机器在任务难度上存在不一致性。 |
| [^37] | [Cascaded Batch Prompting](https://arxiv.org/abs/2608.27038) | 提出级联批量提示，通过两阶段分离复杂推理与符号接地，解决了批量提示的性能不可预测性，并在多项选择问答和自然语言推理任务中实现速度提升和性能突破。 |
| [^38] | [Reasoning about In-Context Samples for Machine-Translation](https://arxiv.org/abs/2608.27036) | 本文提出了一种基于片段的推理框架，通过从相似示例中提取平行片段作为中间推理步骤，显著提升了LLM在机器翻译中的表现。 |
| [^39] | [Representing and Parsing Korean Constituency Structure at Different Levels of Granularity](https://arxiv.org/abs/2608.27035) | 本文通过比较三种基于Penn韩语树库的不同粒度表示，系统评估了韩语成分解析在形态复杂eojeol单元下的表示策略与解析性能。 |
| [^40] | [ITL: Interpretable Document Alignment with Structured Reference Frameworks](https://arxiv.org/abs/2608.27031) | 本文提出了一种名为ITL的领域无关、语言可移植的可解释文档对齐方法，通过从结构化参考框架中构建概念级术语轮廓并分配重要性权重，实现了对文档与概念之间对齐程度的定量、可追踪度量。 |
| [^41] | [JudgeStealer: Extracting LLM Judging Capabilities across Evaluation Protocols](https://arxiv.org/abs/2608.26982) | 本文提出了首个查询高效的模型提取框架JUDGESTEALER，通过跨协议一致性转换和动态输入选择，在有限查询预算下复制LLM评判器的多种评估能力。 |
| [^42] | [Terrain signatures in Welsh settlement names](https://arxiv.org/abs/2608.26978) | 本研究首次通过严格验证的词汇框架和地理结构化模型，证明威尔士聚落名称中的地形词汇（如“bryn”与“cwm”）能可靠地反映局部海拔差异，表明地名保留了可量化的环境信息。 |
| [^43] | [Squeezing More from Limited Data with Recursive Transformers](https://arxiv.org/abs/2608.26973) | 该论文提出递归Transformer和分解嵌入，在有限数据预算下通过复用共享块扩展计算并减少嵌入参数，显著提升模型泛化性能。 |
| [^44] | [TEMPLAR Wales: A georeferenced environmental and toponymic dataset of Welsh settlements](https://arxiv.org/abs/2608.26970) | 该论文构建了威尔士全境3,757个定居点的地理参考环境-地名数据集，通过稳定标识符整合可复现的定居点框架、确定性词汇筛选和环境属性，实现了地名与环境数据的标准化定量分析。 |
| [^45] | [Scaling Model-Generated Distillation Data Can Make Latent Teacher Traits More Recoverable](https://arxiv.org/abs/2608.26958) | 本文发现，扩展模型生成的蒸馏数据量不仅提升学生性能，还能增强教师潜在特征在学生行为中的可恢复性，即使数据与任务无关且未明确提及该特征。 |
| [^46] | [From Atomic to Agentic: Towards Interpretable Evaluation of LLMs' Agentic Mathematical Capabilities](https://arxiv.org/abs/2608.26950) | 本文提出一个过程级数学基准，通过将智能体行为与原子能力分类对齐，并利用自动化流水线生成细粒度注释，实现可解释地评估LLMs的智能体数学推理能力。 |
| [^47] | [KinyaEmbed: Contrastive Sentence Embeddings for Kinyarwanda via Multi-Stage Curriculum Training](https://arxiv.org/abs/2608.26941) | 本文提出了首个专门针对基尼亚卢旺达语的句子嵌入模型KinyaEmbed，通过四阶段课程训练显著提升了该语言在语义相似度任务上的性能，并引入了无污染评估数据集Wiki-RW-STS。 |
| [^48] | [Mapping Written Words to Spoken Words in a Different Language Using Only Visual Grounding](https://arxiv.org/abs/2608.26925) | 本文提出一种基于自监督语音表征和对齐的简单方法，利用视觉接地数据将书面英文关键词映射到印地语口语表达，无需端到端神经模型，在关键词检测和定位上表现良好。 |
| [^49] | [TabuLM: Morphology-Aware Tabular Pre-training for Low-Resource Languages](https://arxiv.org/abs/2608.26923) | 本文提出了TabuLM，首个面向基尼亚卢旺达语的形态学感知表格预训练模型，通过新目标MCR和CTP及结构注意力偏置，在低资源语言上实现了有效的表格表示学习，并构建了首个原生问答基准TabQA-kin。 |
| [^50] | [AraMS-28k: The Largest Publicly Released Line-Level Dataset of Historical Arabic Manuscripts with Margin and Insertion-Anchor Annotations](https://arxiv.org/abs/2608.26921) | 该论文首次发布了最大规模的历史阿拉伯语手稿行级数据集AraMS-28k，并引入了行级插入锚点标注，以恢复非线性阅读顺序，同时提供标音与去标音两种转写版本。 |
| [^51] | [Planting a Latent Variable in Natural-Looking Text: a More Realistic Test of Belief States in LLMs and Their Link to Concept Geometry](https://arxiv.org/abs/2608.26887) | 本工作通过在自然文本中植入可控潜变量，证明小型LLM不仅能追踪其贝叶斯后验信念，还能在特征几何中按马尔可夫链顺序排列这些状态，首次将信念状态与概念几何联系起来。 |
| [^52] | [C-Unseen: Weak Signal Detection in Dynamic Temporal Knowledge Graphs via LLM Reasoning](https://arxiv.org/abs/2608.26870) | C-Unseen利用大语言模型推理和链式思维，通过识别并追踪罕见语义连贯子图在动态时序知识图谱中的持续增殖，实现自解释的弱信号检测。 |
| [^53] | [Evaluating Confidence-Gated Retrieval with Matched Trajectory Replay](https://arxiv.org/abs/2608.26846) | 该论文提出匹配轨迹回放协议，发现事后等渗校准在相同阈值下改变代理的承诺决策，显著提升已承诺答案的准确率（最高达41个百分点），但可能减少回答范围。 |
| [^54] | [SymbolLKG: Towards Verifiable Logical Reasoning via Logical Knowledge Graph and Symbolic Solvers](https://arxiv.org/abs/2608.26836) | 本文提出了一种结合逻辑知识图谱和动态符号求解器路由的神经符号架构，通过将逻辑规则作为拓扑节点显式建模依赖，并采用拓扑感知检索和逻辑路由器，实现了可验证的逻辑推理，有效减少了LLM的幻觉和推理不一致问题。 |
| [^55] | [RuleWeaver: Benchmarking Rule-Centered Scenario Reasoning for Large Language Models](https://arxiv.org/abs/2608.26832) | 本文提出RuleWeaver基准框架，通过从语料中提取并增强复杂规则来构建场景问答，以全面评估LLM在规则中心场景推理中的能力，并发现现有模型在此任务上仍有显著不足。 |
| [^56] | [Behavior2Trip: Towards Personalized Travel Planning via User Behavior Trajectory](https://arxiv.org/abs/2608.26807) | 本文提出行为感知旅行规划任务及Behavior2Trip基准，通过用户历史行为轨迹隐式推断偏好，生成个性化旅行计划，减少交互负担并提升个性化水平。 |
| [^57] | [Decoupling Planning and Control for Instructable Agents](https://arxiv.org/abs/2608.26788) | 本文提出Instruct-to-Act系统，通过将VLM规划器生成的高层文本指令与世界模型控制器的低延迟动作执行解耦，并利用合成指令重标记和联合优化训练，使控制器能够接受语言指导，从而在陌生环境中实现可靠且高效的自主行动。 |
| [^58] | [Instruction Quality Matters: Refining Instructions for Effective Preference Learning](https://arxiv.org/abs/2608.26779) | 本文提出指令质量是偏好学习的隐藏瓶颈，并引入指令优化流程，通过选择弱指令和基于评分标准的LLM反馈修订，有效提升偏好数据质量，从而在离线和在线设置中显著改善模型对齐性能。 |
| [^59] | [Equal Ranking Quality, Different Decisions: Training Order-Consistent LLM Scorers](https://arxiv.org/abs/2608.26762) | 本文发现LLM评分器在排序质量相同时决策不一致，提出OC-SFT方法通过训练分数顺序无关性来保持排序质量并提升决策稳定性。 |
| [^60] | [Letters hide the truth from our eyes: English homophones have meaningfully different phonetic realizations](https://arxiv.org/abs/2608.26749) | 本研究通过分析大量英语同音词，发现其语音实现差异与语境含义相关，并采用时间归一化频谱图揭示了传统转录所掩盖的语音细节。 |
| [^61] | [Preserving General Capabilities during Domain Specialization with Uncertainty-Calibrated MOPD](https://arxiv.org/abs/2608.26735) | 本文提出不确定性校准的MOPD方法，通过双温度采样和正优势密度过滤扩大有效学习轨迹，并利用中心化对数似然过滤确保更新可靠性，从而在领域专业化时保持模型的通用能力。 |
| [^62] | [Towards Expert Financial QA via Self-Improving RAG](https://arxiv.org/abs/2608.26706) | 本文提出一种自我改进的检索增强生成框架，通过三个代理和动态阈值重试机制，在金融文档问答中显著提升准确率，尤其能恢复近四成初始错误答案。 |
| [^63] | [PragAlign: Evidence-Sensitive Reply Assistance Across Chinese and Japanese Appropriateness Judgments](https://arxiv.org/abs/2608.26700) | 本文提出PragAlign模型，通过分离上下文阅读与选择性澄清，在中日双语得体性判断中显著优于基线，揭示跨语言共性判断模式。 |
| [^64] | [Scaling phoneme-based TTS augmentation for ASR: A unified pipeline and controlled study](https://arxiv.org/abs/2608.26697) | 本文提出了一种基于音素的统一TTS到ASR增强流程，并引入音素频率引导选择（PFGS）方法，在多种语言的ASR任务中有效提升了性能。 |
| [^65] | [Beyond Reflection: Affirmation as a Promising Behavioral Marker Associated with Quality in Text-Based Counseling](https://arxiv.org/abs/2608.26689) | 本研究通过多层级分析发现，在文本咨询中“肯定”行为比“反思”行为更能稳定预示对话质量，且该信号在跨语言数据集上也有一定表现，为咨询培训提供了新视角。 |
| [^66] | [FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance for Pruned Large Language Models](https://arxiv.org/abs/2608.26676) | 本文提出FOCUS和RePAIR两种令牌级引导方法，通过分析剪枝LLM中的文本退化机制，分别抑制教师泄漏和防止重复循环，从而在不牺牲任务性能的前提下缓解退化问题。 |
| [^67] | [Do LLMs Understand Personality? Rethinking Persona Fidelity Evaluation through Structured Behavioral Inference](https://arxiv.org/abs/2608.26674) | 本文提出PRISM框架，通过系统功能语言学启发的结构化逆推理方法，解决了现有LLM人格保真度评估中的整体评估幻觉和静态测量局限。 |
| [^68] | [Meta-Learning Where to Allocate Experts: Task-Conditioned Layer-Wise Compression for MoEs](https://arxiv.org/abs/2608.26650) | 本文提出MetaNet，一种任务条件化的分层压缩方法，通过元学习预测每层的专家保留阈值和路由偏差，在冻结模型参数下实现可调的准确率-专家激活权衡，减少约40%的专家激活而保持性能。 |
| [^69] | [Information-Guided Frontier Decoding: Contextual Utility-Driven Commitment in dMLLMs](https://arxiv.org/abs/2608.26641) | 提出一种无需训练的解码策略IGFD，通过结合置信度、邻域不确定性和结构风险来优化令牌承诺顺序，优先保留语义锚点并延迟结构脆弱令牌，从而提升扩散多模态语言模型的解码质量。 |
| [^70] | [Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation](https://arxiv.org/abs/2608.26638) | 本文提出预测驱动评估框架，结合少量人工判断与大规模自动评分实现无偏且高效的系统比较，并引入PPSR元指标来衡量自动指标节省人工标注的程度，优于现有元指标。 |
| [^71] | [The Thousand-Graph Hypothesis: A Testable Hypothesis of Task-Conditioned Relation Materialization in Repository-Level Code Reasoning](https://arxiv.org/abs/2608.26602) | 本文提出一种仅含实体的外部接口，通过推理时的任务条件化关系具体化，在无需预构建关系图的情况下，显著提升了仓库级代码推理的成功率。 |
| [^72] | [Not Just Reason, Not Just Scan: Reinforcement Learning for Proactive Scientific Error Verification over Academic Paper](https://arxiv.org/abs/2608.26596) | 本文提出VERA-RL，一种基于强化学习的框架，用于在无预设问题和证据的情况下主动检测学术论文中的科学错误，并构建了包含12,900个样本的VERA-13K数据集以支持这一任务。 |
| [^73] | [Benchmarking Clinical Decision Pathway Adherence in Large Language Models](https://arxiv.org/abs/2608.26592) | 该论文提出了MEGA-CDP基准，首次系统评估医学大语言模型遵循临床指南路径的能力，并发现现有模型在此任务上仍面临挑战。 |
| [^74] | [Surgical Alignment in Knowledge Graph Training for Clinical Diagnosis with Large Language Models](https://arxiv.org/abs/2608.26587) | 本文提出“外科式对齐”概念，通过梯度干预密度和梯度扭曲指标，发现KL正则化下的知识图谱判断训练能产生稀疏局部更新，优于任务特定SFT的密集更新，从而更有效地将KG知识整合到LLM中用于临床诊断。 |
| [^75] | [J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data](https://arxiv.org/abs/2608.26582) | J-Zero提出了一种统一的挑战者-求解者-评判者协同进化框架，通过对抗性任务生成和基于生成方式的偏好对，实现了无需人工数据即可在可验证和不可验证领域中的自我进化。 |
| [^76] | [Visual Information-Guided Parallel Decoding for Diffusion Multimodal Large Language Models](https://arxiv.org/abs/2608.26580) | 提出一种视觉信息引导采样器，通过利用标记对图像标记的注意力优先级和约束惩罚，改进了扩散多模态大语言模型的解码质量。 |
| [^77] | [Double Trouble: Bilingual Pretraining Leaves Language-Conditioned Effects in Shared-Language Representations](https://arxiv.org/abs/2608.26576) | 双语预训练在深层隐藏状态中引入语言条件效应，即使在对齐嵌入空间后，共享词汇的表示在预测层面仍不一致，源于上下文处理而非输入表示。 |
| [^78] | [Dependency-Aware Revocable Decoding for Efficient Diffusion Large Language Model Inference](https://arxiv.org/abs/2608.26574) | 本文提出依赖感知的可撤销解码（DARD），通过分离标记状态并排除不可靠标记来保护验证上下文，从而在不牺牲生成质量的前提下提升扩散大语言模型的并行解码效率。 |
| [^79] | [SPT: Skills as Pre-Training Data for Agentic Language Models](https://arxiv.org/abs/2608.26563) | 本文提出技能预训练（SPT）方法，通过将公开技能包作为中间训练数据，结合引用插入策略，显著提升智能体语言模型的工具使用能力。 |
| [^80] | [SPEAR: Distilling Domain-Adaptive Reasoning Skeletons via Sequential Symbolic Alignment in Reinforcement Learning](https://arxiv.org/abs/2608.26550) | SPEAR提出了一种无需训练、即插即用的过程奖励方法，通过符号里程碑和最长公共子序列对齐，在强化学习蒸馏中提供密集且逻辑一致的奖励，避免了昂贵的外部神经过程奖励模型。 |
| [^81] | [DuMateBench: Evaluating Autonomous Agents in Complex Real-World Workflows](https://arxiv.org/abs/2608.26546) | 本文提出了DuMateBench，一个基于真实用户会话重建的基准，通过200个多能力任务和三种环境复杂性挑战，填补了代理评估在现实工作流中的空白。 |
| [^82] | [Multi-Expert Conformal Risk Control for Pairwise LLM Judging in Open-Ended Dialogue](https://arxiv.org/abs/2608.26529) | 本文提出多专家共形风险控制方法，并引入MC3算法，通过初始阈值比率校准异质专家评分尺度，实现开放式对话中成对LLM评判的风险控制与覆盖率优化。 |
| [^83] | [Sycophancy Suppression Can Impair Rational Updating: Anti-Sycophancy Should Preserve the Ability to Update](https://arxiv.org/abs/2608.26511) | 反奉承方法在减少无依据屈服时可能损害模型的理性更新能力，两者之间存在权衡，需在设计中保留更新能力。 |
| [^84] | [A Single Suffix to Break Them All: Basin-Aware Jailbreaks for Merged Model Families](https://arxiv.org/abs/2608.26506) | 我们发现了模型合并中源于预训练基础模型的隐蔽越狱风险，并提出盆地感知越狱（BAJ）方法，通过最小-最大优化生成跨合并模型族可迁移的对抗后缀。 |
| [^85] | [Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance](https://arxiv.org/abs/2608.26480) | 本文证明，在不进行训练或基准调优的情况下，基于账本控制的管理器-工作器脚手架能显著提升某些LLM的编码性能，但效果因模型而异，并非普遍适用。 |
| [^86] | [Compositional Generalization via Structural Identification in a Category-Theoretic Framework](https://arxiv.org/abs/2608.26465) | 本文提出一种无需训练模型、基于范畴论函子和Kan扩展的方法，通过分析训练语料在结构识别下的可接受性，来诊断组合泛化的支持与失败模式。 |
| [^87] | [Diff Mining: Logit Differences Reveal Finetuning Objectives](https://arxiv.org/abs/2608.26462) | 本文提出了一种名为“差异挖掘”的简单框架，通过比较微调模型与基础模型的输出逻辑值来揭示微调目标，无需访问模型内部信息，并能有效识别微调过程中学到的行为。 |
| [^88] | [Vowel Signs Are Not Letters: A Pre-tokenization Ceiling on Multilingual Tokenizer Fertility](https://arxiv.org/abs/2608.26449) | 本文发现，基于GPT-2单词正则的预分词器在元音附标文字中每个元音符号处强制拆分单词，导致分词生育率显著上升，并提出了一个无需训练的下界来量化此影响。 |
| [^89] | [Don't Overthink, Don't Underthink: Toward Adaptive Reasoning in Agentic AI](https://arxiv.org/abs/2608.26442) | 本文指出代理型AI系统需要自适应推理机制，以动态调整推理资源，避免过度或不足推理带来的效率与性能问题。 |
| [^90] | [AfriSwitch: A Benchmark for In-the-Wild African Code-Switched Speech Recognition](https://arxiv.org/abs/2608.26434) | 该论文提出了AfriSwitch，一个61.36小时、覆盖16种非洲语言的野外语码转换语音基准，揭示了多语言ASR系统在此场景下的高错误率，并强调现有模型缺乏对非洲语码转换的适应能力。 |
| [^91] | [SpeechGym: An Audio-Native Gym for Training Voice Agents via Reinforcement Learning](https://arxiv.org/abs/2608.26432) | 本文提出SpeechGym，一个音频原生的代理环境，通过强化学习端到端训练语音代理，解决了现有框架无法在音频模态中改进代理能力的问题。 |
| [^92] | [The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection](https://arxiv.org/abs/2608.26423) | 本文提出了一种潜在诊断分类法框架，通过维度优化分类器、识别潜在支持向量和构建诊断分类法，为提示注入检测提供了一种可靠决策与风险标记的端到端指南。 |
| [^93] | [Case2Flow: Bridging Patient Cases and Guideline Flowcharts through Multimodal Retrieval](https://arxiv.org/abs/2608.26414) | 本文提出了Case2Flow任务和FlowAtlas语料库，用于从医学指南中为患者病例检索相关流程图，并揭示了现有多模态检索方法的系统性缺陷。 |
| [^94] | [LowRankArena: A Standardized Evaluation Platform for SVD-Based LLM Compression](https://arxiv.org/abs/2608.26389) | 本文提出了LowRankArena，一个统一基准、压缩预算和推理测量的标准化平台，用于公平评估基于SVD的LLM压缩方法，并提供大规模可复现检查点。 |
| [^95] | [Co-Evolving Structured Knowledge and Reasoning in Language Models](https://arxiv.org/abs/2608.26386) | 本文提出KBevo框架，通过端到端联合优化结构化知识库构建与推理，使推理结果直接反馈改进知识库质量，从而提升问答的准确性、可控性和组合推理能力。 |
| [^96] | [Why RAGs Hallucinate: Penalty-Aware Evaluation of Retrieval-Augmented Generation Systems with Knowledge-Gap Canaries](https://arxiv.org/abs/2608.26385) | 本文提出了一种惩罚感知评估框架，通过引入不对称评分和知识缺口金丝雀，揭示了RAG系统在知识库无法支持时仍会猜测并产生幻觉的问题。 |
| [^97] | [Assessing the Downstream Utility of Evidence-Aware Retrieval in RAG](https://arxiv.org/abs/2608.26379) | 本研究评估证据感知检索信号在RAG中的下游实用性，发现其虽改变检索排名，但对训练、系统选择和答案质量预测的益处并不一致。 |
| [^98] | [Survival-Guided Length Control for Efficient Diffusion Language Models](https://arxiv.org/abs/2608.26374) | 提出一种无需训练的生存分析长度预测器，通过预测序列结束标记来动态调整扩散语言模型的解码长度，实现高达7倍的推理加速且不损失准确性。 |
| [^99] | [Knowledge-Verified Emergent Deception in LLM Agents Under Conflicting Incentives](https://arxiv.org/abs/2608.26372) | 该论文提出了KnownLieBench基准，通过知识验证方法，首次确认代理知晓用户权益，再评估利益冲突下其是否产生虚假陈述，从而区分真实欺骗与无知或幻觉。 |
| [^100] | [Cross-lingual Representation Learning via Centroid Intervention Fusion](https://arxiv.org/abs/2608.26357) | 提出质心干预融合（CIF）框架，将多个跨语言干预投影融合为单一语言共享算子，显著提升低资源语言的跨语言迁移性能。 |
| [^101] | [MoganColBERT-TR: A Late-Interaction Multi-Vector Retrieval Model for Turkish](https://arxiv.org/abs/2608.26344) | 该论文提出了MoganColBERT-TR，一种基于晚期交互的多向量检索模型，通过从嵌入模型微调并利用交叉编码器蒸馏，实现了土耳其语的高效词元级检索。 |
| [^102] | [Neuro-symbolic PRM: Enhancing Scientific Reasoning via Structured Traces and Symbolic Verification](https://arxiv.org/abs/2608.26329) | 本文提出了一种神经符号框架，通过将推理解耦为符号有效性和语义依据性两个维度，并采用确定性验证器和条件化过程奖励模型，有效解决了大型语言模型在科学推理中中间步骤缺乏上下文依据的问题。 |
| [^103] | [How Unlikely Is "Unlikely"? Assessing Verbal Probability Perception Across Large Language Models](https://arxiv.org/abs/2608.26327) | 大型语言模型在理解概率词语时与人类基准高度一致，但对否定表达（如“不可能”）存在系统性高估偏差，且“可能”一词在模型间分歧最大。 |
| [^104] | [When Is Noise Response Universal? Tokenization as the Hidden Variable in Language Models](https://arxiv.org/abs/2608.26319) | 本文发现语言模型在词级噪声下表现普遍一致，但在字符级噪声下分化，并揭示分词机制是这一差异的隐藏决定因素，而非模型架构。 |
| [^105] | [MemToC: Benchmarking Memory-Tool Conflict Resolution in Large Language Models](https://arxiv.org/abs/2608.26295) | 该论文提出了MemToC基准测试，通过可执行工具和已知正确性控制，系统评估了大型语言模型在工具返回与参数记忆冲突时的仲裁能力，发现工具返回结果强烈主导闭卷答案，且模型在工具错误时保留正确记忆的能力较弱。 |
| [^106] | [On Scope Classification and Current Knowledge-Editing Benchmarks: A Negative Result, with INLAY as a Gradient-Free Case Study](https://arxiv.org/abs/2608.26292) | 当前知识编辑基准无法衡量范围分类决策，因为所有测试场景中任何路由方法的最大收益为零，表明这些基准存在结构性缺陷。 |
| [^107] | [A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers](https://arxiv.org/abs/2608.26194) | 本文提出了一种名为STeReO的重排序器，它通过整合语音和文本检索器来聚合异构模态数据库，并利用自建数据集训练，从而在多模态检索中显著提升证据选择的准确性。 |
| [^108] | [Comparing Chunking and Embedding Strategies for Turkish RAG Systems](https://arxiv.org/abs/2608.26192) | 本文系统比较了土耳其语RAG系统中分块策略与嵌入模型的影响，发现布局感知分块能缩小嵌入模型差异，且领先嵌入模型间无显著统计差异。 |
| [^109] | [Invocation-Level Reliability of Tool-Using Agents](https://arxiv.org/abs/2608.26189) | 本文发现工具使用智能体的调用级可靠性测量受评分规则结构性限制，严重性和恢复参数无法准确估计，且模型自身错误会显著降低深层任务的性能。 |
| [^110] | [When the Canonical Completion Is Wrong: Formalizing and Measuring the Jump in Large Language Models](https://arxiv.org/abs/2608.26187) | 本文首次形式化了LLM中的“跳跃”概念，将其定义为具有可验证证书的有限扩展问题，并提供了度量方法，以解决关于LLM溯因能力的争论。 |
| [^111] | [Investigating the Influence of Prompt and Response Languages on LLM Content Generation](https://arxiv.org/abs/2608.26186) | 本研究揭示提示语言和响应语言对LLM生成内容长度有显著影响，但语义保真度保持稳定，突出了分词器在跨语言生成中的关键作用。 |
| [^112] | [PACEShop: Evaluating Personalized, Actionable, Compositional, and Evidence-grounded Shopping Assistants](https://arxiv.org/abs/2608.26180) | 本文提出PACE评估框架和PACEShop基准数据集，以联合衡量购物助手在个性化、可操作性、组合性和证据基础方面的结构化响应质量，填补了现有评估方法的空白。 |
| [^113] | [A Multi-Framework Comparison of Outline Stages in Long-Form Generation with LLMs](https://arxiv.org/abs/2608.26177) | 本文首次解耦评估长篇生成中的大纲阶段，提出基于锚点的LLM评审协议，发现无单一框架在所有粒度上占优。 |
| [^114] | [Knowledge Cards: Structured Knowledge for AI Systems](https://arxiv.org/abs/2608.26176) | 本文提出“知识卡片”作为AI系统的新文档形式，旨在填补现有模型卡片、数据卡片和系统卡片在描述系统内部知识结构（概念、关系和推理模式）方面的空白，以支持智能体AI的可检查性和领域专家审查。 |
| [^115] | [Lost in Compression: A Controlled Cross-Lingual Audit of Extractive Prompt Compressors](https://arxiv.org/abs/2608.26175) | 这项研究通过跨十种语言的受控审计发现，英语监督训练的抽取式压缩器在非英语语言上存在显著的迁移差距，且该差距随压缩率增加而扩大，多语言训练则能缓解此问题。 |
| [^116] | [Mitigating Fabrication in Multi-Stage LLM Pipelines for Hiring: An Empirical Evaluation of Prompt Guardrails and Human-in-the-Loop Checkpoints](https://arxiv.org/abs/2608.26171) | 该论文通过实证评估表明，仅靠提示护栏不足以消除LLM招聘流水线中的捏造问题，而结合人在回路检查点能显著降低捏造率并消除身份虚构。 |
| [^117] | [Hallucinations in LLMs: A Lifecycle-Based Survey of Causes, Detection, Mitigation, and Prevention](https://arxiv.org/abs/2608.26168) | 本文提出了一种基于生命周期的幻觉分类框架，将LLM幻觉分为数据、训练和推理三类，并系统综述了各阶段的原因、检测、缓解与预防方法，以及基准数据的评估标准。 |
| [^118] | [Using Poly-Encoders for Computationally Efficient Automated Creativity Assessment](https://arxiv.org/abs/2608.26165) | 本文提出了一种基于Poly-Encoder的自动化创造力评估方法，利用小型BERT编码器在保持与大型语言模型相当性能的同时显著降低计算成本。 |
| [^119] | [From Sound to Symptom: Real-Time Respiratory Signal Understanding for Conversational Healthcare Agents](https://arxiv.org/abs/2608.26163) | 本文提出HealthCUES，一个实时对话代理中首个用于副语言呼吸监测的流式管道，能够在不打断对话流的情况下实现亚秒级咳嗽检测及细粒度分析，并引入对话感知门控机制以防止警报疲劳。 |
| [^120] | [Mutual Debiasing via Dual-Seed Comparison for Probabilistic Sampling in Large Language Models](https://arxiv.org/abs/2608.26161) | 本文提出双种子比较（DSC）方法，通过比较两个独立 LLM 生成的种子来中和采样偏差，从而提升概率采样的保真度。 |
| [^121] | [Self-Generated Text Recognition: Quality Heuristics, Cross-Task Transfer, and Downstream Bias in LLM Evaluation](https://arxiv.org/abs/2608.26159) | 本研究通过系统分析实验设计选择（操作化）对自生成文本识别准确率的影响，调和了先前矛盾结论，并证实了质量启发式在LLM评估中的关键作用。 |
| [^122] | [VFA: Empowering Multilingual MLLMs via Vision-Free Adaptation](https://arxiv.org/abs/2608.26155) | 提出了一种通过组合互补任务向量来解耦多语言增强与视觉对齐的框架，仅用不到2%的数据即可在多语言多模态基准上提升多个MLLM的性能。 |
| [^123] | [Evaluating AI Generated Summaries for Cancer Patients](https://arxiv.org/abs/2608.26154) | 本研究通过人类专家和LLM评估者双重框架评估癌症患者护理应用中的AI生成摘要，识别出遗漏和不准确等局限，并迭代优化提示设计与安全防护。 |
| [^124] | [Artificial Intelligence Models Can Predict and Collaboratively Modulate Human Memory Search](https://arxiv.org/abs/2608.26152) | 本研究首次证明大型语言模型能够预测并协同调节人类在语义记忆搜索中的心理轨迹，从而作为认知工具增强而非取代人类生成性思维。 |
| [^125] | [Towards Interpretable Depression Detection: Linking Acoustic Features to DSM-5 Indicators](https://arxiv.org/abs/2608.26148) | 本文提出一种可解释的抑郁症检测框架，通过透明映射声学特征到DSM-5临床指标，在普通硬件上本地运行以保护隐私，初步验证了其有效性。 |
| [^126] | [CARE: Causally-Aligned Reasoning Exploration for Medical Large Language Models](https://arxiv.org/abs/2608.26147) | CARE通过引入因果充分性和邻近可学习性两个条件，提出了一种因果对齐的推理探索框架，有效解决了医疗大语言模型中强化学习的信用分配失败和梯度爆炸问题，避免了“答案正确、推理错误”的陷阱。 |
| [^127] | [Vagdhenu: A Vrutta (Meter) Aware Shloka-to-Chant (TTS) System for Sanskrit](https://arxiv.org/abs/2608.26146) | 本文介绍了一种基于韵律感知的梵语诗偈转唱诵系统，通过卡纳达文路由和精确音系处理，在现有TTS骨架上实现了高保真唱诵。 |
| [^128] | [Why Current XAI Is Not Enough for Arabic NLP: A Critical Survey of the Explainability Gap](https://arxiv.org/abs/2608.26144) | 本文批判性地指出，阿拉伯语NLP的可解释性存在方法、任务和语言三方面的显著差距，现有XAI技术不足以应对其独特挑战。 |
| [^129] | [Beyond Accuracy: A Qualitative Analysis of Vision-Language Models for Hate Speech Detection in Memes](https://arxiv.org/abs/2608.26143) | 本研究通过定性分析四种视觉语言模型在梗图仇恨言论检测中的表现，揭示了它们在理解上下文和讽刺方面的局限性，并强调了超越分类准确性的评估重要性。 |
| [^130] | [Position Is All You Need: A Free Lunch Token Compression Strategy for MLLM-based Referring Expression Segmentation](https://arxiv.org/abs/2608.26142) | 本文提出PAYN，一种仅依赖位置信息的令牌压缩策略，用于MLLM-based指代表达分割，通过保留位置嵌入和局部空间结构来避免性能退化。 |
| [^131] | [AdaThinking-E: One-Token Entropy Regulation for Adaptive Thinking](https://arxiv.org/abs/2608.26141) | 本文提出AdaThinking-E，一种通过单令牌熵调控的自适应思考强化学习框架，能根据问题复杂度智能决定是否启用深度推理，从而减少简单任务的计算开销并提升整体准确性。 |
| [^132] | [Affix Cache for Diffusion Large Language Models](https://arxiv.org/abs/2608.26140) | ACache提出一种后缀导向的缓存重用机制，通过选择性重计算关键锚点令牌的键值状态，在扩散大语言模型中高效恢复共享文本跨度的精度损失。 |
| [^133] | [Syntax vs. Semantics: How Transformers Learn Deep Dependencies](https://arxiv.org/abs/2608.26139) | 该论文提出一个机制框架，揭示Transformer学习深层语义依赖时存在“梯度饥饿”现象，并解释了思维链策略如何通过外化推理步骤绕过该抑制机制。 |
| [^134] | [Cross-Platform Generalisation Failure in Mental Health Natural Language Processing: A Five-Axis Fairness Audit of Transformer Models on Social Media](https://arxiv.org/abs/2608.26138) | 本研究通过五轴公平性审计框架揭示了心理健康NLP模型在跨平台应用中存在显著的性能下降和校准失败，表明平台特定泛化问题严重制约了实际部署。 |
| [^135] | [Interpretable, Fairly Evaluated Automated L2 Speaking Assessment that Beats the Single-Human Ceiling and Why Pause Encoding Does Not Change LLM Fluency Scores](https://arxiv.org/abs/2608.26137) | 该论文提出一种可解释的特征与LLM混合模型，在L2口语自动评分中超越单人人类评分中位数，实现高准确度和公平性。 |
| [^136] | [Reward-Informed Sparse Autoencoders and the Solution-Completeness Confound](https://arxiv.org/abs/2608.26136) | 本文发现奖励引导的稀疏自编码器在区分好/坏推理时，其分离效果主要源于解决方案完整性（如长度和结构线索）而非推理质量，揭示了潜在混淆因素。 |
| [^137] | [Data Science Approaches to Evaluating Honours Candidates](https://arxiv.org/abs/2608.26135) | 本文提出了一种模块化数据科学流水线，结合多种NLP技术从开源情报中提取公众情绪，并验证了领域知情算法MINOS在评估荣誉候选人声誉方面的优越性。 |
| [^138] | [Agent Seer: Synthesizing Scenarios from Specification Understanding](https://arxiv.org/abs/2608.26133) | Agent Seer仅从单一MCP工具规格中自动合成逼真的多轮评估场景，无需人工干预、示例或实时工具访问，显著提升了AI代理评估的可扩展性和适应性。 |
| [^139] | [SLM-Conditioned Hierarchical Relation Routing for Labeled Property Graph Learning](https://arxiv.org/abs/2608.26132) | 该论文提出了一种将小型语言模型集成到图消息选择中的分层路由架构，通过目标条件查询实现更精确的语义证据选择。 |
| [^140] | [Evaluating Language Models in Realistic Conversational Contexts](https://arxiv.org/abs/2608.26131) | 本文提出了一个名为UPHELD的大型基准测试集，包含专业编剧创作的真实人类对话和大量人工标注，用于填补现有评估框架在衡量大型语言模型人类规模对话能力方面的空白。 |
| [^141] | [Agents Don't Paginate: First-Chunk Selection for LLM Tool Responses](https://arxiv.org/abs/2608.26130) | 该论文发现编码智能体从不使用分页获取额外工具响应块，并提出首块选择策略（基于0/1背包）以最大化首块中黄金项的包含率，实验表明现有方法在精度上存在不足。 |
| [^142] | [FIRSTPASS: A Multi-Domain, Multi-Round Peer Review Dataset Grounded in Real Editorial Outcomes](https://arxiv.org/abs/2608.26129) | 该论文提出了FIRSTPASS，这是首个覆盖多学科、多轮真实编辑对话的同行评审数据集，填补了非计算机科学领域评审数据的空白。 |
| [^143] | [Graph-Based Modeling of Financial Volatility Dynamics](https://arxiv.org/abs/2608.26127) | 提出了一种将隐含波动率曲面建模为动态图序列的金融感知图时空网络，以更准确地预测已实现波动率。 |
| [^144] | [TelecomGPT-R1: A Unified Open-Source Reasoner for the Telecom Stack](https://arxiv.org/abs/2608.26126) | 本文发布了TelecomGPT-R1-9B，一个统一的开源电信推理器，通过围绕协议、知识、建模和故障四个推理轴的67,427个示例监督微调语料库，结合轴特定思维链增强，在GSMA开放电信排行榜上取得顶尖性能，弥合了通用推理器与领域特定电信LLM之间的能力差距。 |
| [^145] | [Training-Time Explainability for Multilingual Hate Speech Detection: Aligning Model Reasoning with Human Rationales](https://arxiv.org/abs/2608.26125) | 本文提出了一种训练时可解释性框架，通过将模型推理与人类标注理由对齐，在多语言仇恨言论检测中同时提升了分类性能和解释质量，尤其有效捕捉了文化特定的隐性反穆斯林仇恨线索。 |
| [^146] | [Natural-Language Policies to Executable Decisions: An Interpretable Large Language Model Framework](https://arxiv.org/abs/2608.26124) | 本文提出一个生产级LLM定价系统，通过严格决策边界和可解释条件树，将非结构化政策转化为确定性可审计的决策，实现了开放政策支持与可靠金融决策的结合。 |
| [^147] | [Which India Survives Translation? Narrative Homogenisation Across Indian Oral Traditions in LLMs](https://arxiv.org/abs/2608.26123) | 本研究通过计算分析发现，大型语言模型在生成印度不同口头传统的叙事时存在显著的同质化倾向，导致地区文化多样性被扁平化。 |
| [^148] | [Can a Model Catch Its Own Hallucinations for Free?: Label-Free Doubt Signals Hold Their Own Against a Labelled Dataset for Abstention](https://arxiv.org/abs/2608.26121) | 本文证明，仅使用模型自身无标签的置信度信号进行弃权调优，即可达到与使用带标签数据集监督训练相当的性能，从而无需额外标注即可让模型识别并避免自身幻觉。 |
| [^149] | [Recipes for Steering and Scaling LLMs via Sampling](https://arxiv.org/abs/2608.26120) | 本文提出了一种基于采样（SMC和RE算法）的灵活框架，用于引导和扩展LLMs生成质量，无需外部监督，且比现有方法更具扩展性。 |
| [^150] | [DeflectBench: A Benchmark for Evaluating Rhetorical Fallacy Generation in LLMs](https://arxiv.org/abs/2608.26119) | 该基准揭示了大型语言模型生成修辞谬误的拒绝行为主要由提示结构而非内容驱动，且某些提示框架可完全绕过安全限制。 |
| [^151] | [ElementCheck: Complexity-Aware Long-Form Text Factuality Evaluation via Sentence Elements](https://arxiv.org/abs/2608.26118) | ElementCheck提出了一种基于句子元素图的复杂度感知验证框架，通过图拓扑结构估计句子复杂度并自适应调整验证粒度，解决了长文本事实性评估中固定分解和验证粒度导致的可靠性问题。 |
| [^152] | [CIFQA: A Deterministic Tool-Grounded Multi-Agent LLM Framework for Financial Query Answering](https://arxiv.org/abs/2608.26114) | CIFQA通过多智能体分离语言理解与数值执行，结合确定性Python工具，解决了LLM在多步金融计算中的数值错误问题。 |
| [^153] | [TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding](https://arxiv.org/abs/2608.26112) | 我们提出了TreeGraft，一种自适应多草稿器嫁接框架，通过结合不同成本的草稿器来构建共享草稿树，从而平衡草稿质量与延迟，提升基于树的投机解码效率。 |
| [^154] | [VISA: Agentic Self-Evolving Data Synthesis for Multimodal Instruction Following](https://arxiv.org/abs/2608.26013) | VISA提出了一种自进化智能体循环，通过利用失败样本、验证器反馈和目标模型错误来动态优化多模态指令合成，显著提升数据质量与难度。 |
| [^155] | [Unveiling Spectral Mechanisms in Training-Free LLM Text Detection](https://arxiv.org/abs/2608.25944) | 本文从理论和实证角度揭示了无训练文本检测中频谱分析的作用机制，指出频谱能量捕捉人类写作的“生成活力”，且其信号强度在长、连续、受约束的生成中最明显。 |
| [^156] | [Learning New Facts with QLoRA: An Acquisition-Retention Frontier](https://arxiv.org/abs/2608.25677) | 本文通过系统比较不同秩的QLoRA与全量微调，揭示了参数高效微调中秩的选择在事实获取与能力保留之间存在明确的权衡边界，其中低秩保留能力但获取少，高秩获取多但损失能力，全量微调则作为保守基线。 |
| [^157] | [A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation](https://arxiv.org/abs/2608.25643) | 本文通过分析同策略蒸馏中逐标记梯度的非均匀性，提出了一种轻量级的惊喜感知重新加权（SuRe）方法，以有效增强低概率标记的权重，从而提升蒸馏性能。 |
| [^158] | [When Stale Constraints Go Unchecked: Budgeted Verification Failures in Inherited Agent Memory](https://arxiv.org/abs/2608.25553) | 该论文研究了在有限验证预算下，代理继承的过时约束未被检查导致验证失败的问题，并提出了通过重新分配验证槽位来减少这类错误的方法。 |
| [^159] | [ClueWeaver: Reward-Guided Dual-Agent Evidence Reasoning for Compact LLMs on Literary Long Narratives](https://arxiv.org/abs/2608.25531) | ClueWeaver提出了一种奖励引导的双智能体框架，通过“查找者”与“解释者”的协作，使紧凑型本地模型在长篇文学叙事问答中实现高效、可检查且证据忠实的推理。 |
| [^160] | [DataKernelBench: Can LLMs Optimize Database Queries on GPUs?](https://arxiv.org/abs/2608.25061) | 本文提出了DataKernelBench基准测试，证明大语言模型能通过执行引导修复在GPU上优化数据库查询，最强配置实现2.11倍加速，并发现内核融合、完整查询专业化和工作负载上下文是关键因素。 |
| [^161] | [Unsupervised Post-Training of Foundation Models: A Survey](https://arxiv.org/abs/2608.24982) | 本文综述了80种无监督后训练方法，提出按内部信号源分类，并揭示了信号选择与任务结构决定模型改进或错误放大的关键机制。 |
| [^162] | [Retrieved But Not Reliable: A Survey on Attacks, and Defenses in Retrieval-Augmented Generation](https://arxiv.org/abs/2608.24977) | 本综述统一了检索增强生成全流水线中的攻击与防御框架，按准确性、隐私和公平性分类威胁模型，并覆盖各阶段防御策略。 |
| [^163] | [RePolicy: Reinforcement Learning for Safety-Policy Invocation in Agent Safeguards](https://arxiv.org/abs/2608.24275) | RePolicy通过强化学习动态调用安全策略，实现了对未见轨迹和变化策略上下文的高鲁棒性安全检测。 |
| [^164] | [FireRedAudio: A General-Purpose Audio Language Model with Decoupled Continuous Representations for Understanding and Generation](https://arxiv.org/abs/2608.24168) | FireRedAudio是首个在单个可训练的自动回归LLM中，为理解和生成提供分离连续输入表示的公开统一音频语言模型，通过解耦表示实现了语音识别、理解、翻译、编辑和合成等多个任务的最优性能。 |
| [^165] | [When Youth Enter The Chat: An Epistemic Shift in the Validation of LLM-Based Measures of Student Talk](https://arxiv.org/abs/2608.23780) | 本文提出，验证基于LLM的学生谈话测量时，应通过重新情境化对话和让青年参与研究来共享认知权威，以确保测量对边缘化青年有意义且公平。 |
| [^166] | [The Limits of Automatic Evaluation of Creativity in Large Language Models](https://arxiv.org/abs/2608.23705) | 本研究发现，现有的自动评估方法，尤其是基于LLM的评判者，无法可靠地匹配人类对创造力的判断，并表现出对AI生成文本的系统性偏好。 |
| [^167] | [Language Chain in Alignment: Cross-Lingual Ranking Preference Optimization](https://arxiv.org/abs/2608.23149) | 本文提出跨语言排序偏好优化（CRPO）框架，通过利用英语偏好知识的分层结构，在目标语言中实现更优的语言对齐和输出质量。 |
| [^168] | [Buried in Textual Debt: Context Pruning with Visual Evidence Preservation for MLLM Agents](https://arxiv.org/abs/2608.22963) | 本文提出SPARE框架，通过KL散度引导的上下文剪枝，在保留视觉证据的同时移除冗余推理文本，以解决MLLM智能体长轨迹中的“文本债务”问题。 |
| [^169] | [ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation](https://arxiv.org/abs/2608.22559) | ExecRubrics通过将评分标准转化为可执行的Python函数，实现了可验证、高效且能捕捉复杂依赖关系的长篇评估，替代了昂贵的黑盒LLM评判器。 |
| [^170] | [Aligned Alone, Misaligned Together: Forecasting Adversarial Capture in LLM Agent Populations](https://arxiv.org/abs/2608.22444) | 该论文发现单个LLM智能体的校准行为无法预测其群体行为，但通过良性操作数据可校准响应函数，从而提前预测对抗性俘获。 |
| [^171] | [Decoupled Physical Modeling and Execution for Physics Reasoning](https://arxiv.org/abs/2608.22126) | 该论文提出了一种解耦物理建模与执行的统一框架，通过两阶段后训练策略（监督微调加规则反馈强化学习）提炼中间表示，从而提升大型语言模型在物理推理任务上的表现。 |
| [^172] | [GUI-Primitives: Diagnosing Spatial Reasoning Failures in Vision-Language GUI Grounding](https://arxiv.org/abs/2608.21832) | 该论文提出了一个名为GUI-Primitives的基准测试，通过对比指令对系统性地诊断视觉-语言模型在GUI空间关系推理中的缺陷，发现现有模型在严格定位准确率上表现极差，且多数预测完全偏离候选区域。 |
| [^173] | [TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness](https://arxiv.org/abs/2608.15763) | 本文提出操控系统感知训练（HAT）方法，通过将可进化的操控系统状态纳入训练分布，使数字人代理在实时直播中既能快速响应又能灵活适应动态策略变化。 |
| [^174] | [MLLM-Routed Heterogeneous Ensembles for Robust Cross-Dataset Image Classification](https://arxiv.org/abs/2608.13463) | 本文提出ARMDIL，利用多模态大语言模型动态路由图像到异构视觉骨干网络，在跨数据集分类中实现鲁棒性能，无需专门训练路由器。 |
| [^175] | [Why Knowing Both Hops Is Not Enough: Understanding Two-Hop Generalization in Language Models](https://arxiv.org/abs/2608.07261) | 本文通过受控符号环境中的变换器训练和机制分析，揭示了两跳泛化中第二跳分布内成功而分布外失败的根本原因，即一致中间表示的出现与层间不匹配。 |
| [^176] | [ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs](https://arxiv.org/abs/2608.01979) | ET-Prune提出了一种免训练的动态令牌剪枝方法，通过证据感知分配和问题条件化保护，在OCR任务中自适应调整剪枝预算，避免丢失关键文本证据。 |
| [^177] | [CultureVidBench: Benchmarking Cultural Understanding in Text-to-Video Generation](https://arxiv.org/abs/2608.01942) | CultureVidBench是一个全新的基准测试，通过覆盖12个国家、6大洲的1000条提示词，系统评估文本生成视频模型在物质文化、社会实践和仪式典礼等多个维度的文化理解能力，填补了现有基准无法直接评估文化表现力的空白。 |
| [^178] | [Can LVLMs Uncover the Truth Behind Visual Illusions? An Analysis of Perceptual and Reasoning Capabilities](https://arxiv.org/abs/2607.27747) | 通过构建基于真实视觉错觉图像的基准测试，发现大型视觉语言模型在联合感知与推理方面能力有限，远未达到其宣称的先进水平。 |
| [^179] | [TriShieldRAG: 3 Rings, One Blind Spot in Layered Defenses for Retrieval-Augmented Generation](https://arxiv.org/abs/2607.23838) | TriShieldRAG通过三层防御（文档筛选、信任重排序和多模型共识）显著提升RAG系统对毒化文档的鲁棒性，有效降低攻击成功率。 |
| [^180] | [Selective State-Space Adaptation and Retrieval for Language Model Reasoning](https://arxiv.org/abs/2607.19326) | 本文提出两种选择性状态空间适配器（MaLoRA和MaRA），分别在令牌和上下文级别动态调控语言模型推理，以提升自适应能力和推理性能。 |
| [^181] | [Non-binary bottom-up constituency parsing without arity actions](https://arxiv.org/abs/2607.10591) | 本文提出一种无需动作数参数的非二进成分句法分析方法，通过分隔符限定栈配置来唯一确定归约域，从而减少动作库规模并保持树结构准确性。 |
| [^182] | [When Top-K Misses the Decision: Tool-Call Drift in Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2607.07050) | 本研究揭示了多教师在线策略蒸馏中Top-K采样导致的“工具调用漂移”问题，并提出通过学生感知的支持集恢复被遗漏的令牌坐标，从而显著降低过度调用率。 |
| [^183] | [The Score Granularity Gap in Black-Box LLM Classification: A Comparative Study of Confidence Constructions](https://arxiv.org/abs/2606.22179) | 本文揭示了黑盒LLM分类中置信度分数的“粒度差距”，即单次语言化置信度虽排序性能良好，但取值有限，限制了阈值设定精度。 |
| [^184] | [Learning the ARTS of Search for Automated Discovery](https://arxiv.org/abs/2606.21891) | 提出了ARTS方法，通过推理语言模型和测试时训练，区分假设价值与执行质量，并解决搜索日志上下文长度问题，从而提升自动科学发现的效率。 |
| [^185] | [ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents](https://arxiv.org/abs/2606.18037) | 我们提出ProvenanceGuard，一种源感知验证器，通过分解主张、路由到特定源证据并检查归因，解决LLM代理中跨源混淆的问题，并支持答案级允许/阻止决策及修复。 |
| [^186] | [Not Truly Multilingual: Script Consistency as a Missing Dimension in VLM Evaluation](https://arxiv.org/abs/2606.17188) | 当前多语言VLM评估忽视了多脚本语言，本研究通过旁遮普语三种脚本基准发现模型存在显著脚本差距，视觉输入无法消除，表明其并非真正的多脚本。 |
| [^187] | [SkillChain: Closing the Loop on Skill Evolution for Image-Based E-Commerce AI Assistants](https://arxiv.org/abs/2606.12984) | SkillChain通过自动化技能创建、路由优化和迭代细化，为基于图像的电商AI助手实现了技能演化的生产闭环，有效应对异构用户意图的挑战。 |
| [^188] | [Harnessing the Collective Intelligence of AI Agents in the Wild for New Discoveries](https://arxiv.org/abs/2606.10402) | 本文提出了EinsteinArena平台，通过开放分布式环境中的AI代理集体协作，在数学任务上发现了12个超越人类和现有AI的新最先进结果。 |
| [^189] | [Beyond FLOPs: Benchmarking Real Inference Acceleration of LLM Pruning under a GEMM-Centric Taxonomy](https://arxiv.org/abs/2606.09080) | 本文提出了一种基于GEMM维度的分类法，构建统一基准框架，以系统评估不同LLM剪枝方法在真实硬件上的加速效果，而非仅依赖FLOPs。 |
| [^190] | [Forward-Free Diffusion Language Models with BPTT-Free Looped Refinement](https://arxiv.org/abs/2606.08357) | 本文提出FReDA，一种无需人工设计前向过程的扩散语言模型，通过递归分布细化利用模型生成的草稿作为中间状态，避免了传统扩散模型中的前向扰动和BPTT训练开销，提升了生成质量。 |
| [^191] | [Summarization is Not Dead Yet](https://arxiv.org/abs/2606.08000) | 摘要生成并未消亡，人类参考文献在信息性和忠实性上仍优于LLM输出，后者仅胜在表面流畅性。 |
| [^192] | [The Cold-Start Safety Gap in LLM Agents](https://arxiv.org/abs/2606.07867) | 本文发现大语言模型智能体在会话开始时存在“冷启动安全差距”，即初始阶段最脆弱，随前置常规任务增加安全性显著提升，并提出SODA基准系统量化这一现象。 |
| [^193] | [Many Circuits, One Mechanism: Input Variation and Evaluation Granularity in Circuit Discovery](https://arxiv.org/abs/2606.06267) | 该研究发现，电路发现中结构差异可能不代表机制差异，而是输入频率变化导致的“幻影特化”，实际功能上共享单一计算机制。 |
| [^194] | [An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic](https://arxiv.org/abs/2606.05725) | 本文提出一种基于最大均值差异（MMD）的简单检测器，通过比较API流量窗口的语义分布与历史良性流量，有效且低延迟地检测模型提取攻击。 |
| [^195] | [CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2606.02502) | 本文提出CRAM方法，通过中心路由和自适应混合专家模型，在缓解多模态持续指令微调中灾难性遗忘的同时，提升参数效率。 |
| [^196] | [MineExplorer: Evaluating Open-World Exploration of MLLM Agents in Minecraft](https://arxiv.org/abs/2605.30931) | 本文提出了MineExplorer基准测试，通过筛选通用原子任务、组合隐式多跳任务及多智能体合成流程，系统评估MLLM智能体在《我的世界》中的开放世界探索能力。 |
| [^197] | [NestedKV: Nested Memory Routing for Long-Context KV Cache Compression](https://arxiv.org/abs/2605.26678) | NestedKV提出了一种免训练的键缓存压缩方法，通过多时间尺度余弦异常评分和头部自适应混合，有效解决了长上下文场景中单一重要性信号的脆弱性问题。 |
| [^198] | [ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions](https://arxiv.org/abs/2605.24279) | ContextEcho通过一个结合25个探针、快照-然后-探针协议及多种测量方法的基准测试，揭示了长时智能体编码会话中语言模型人格漂移的现象。 |
| [^199] | [HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents](https://arxiv.org/abs/2605.17873) | HINT-SD通过利用完整轨迹后见之明精准定位失败相关动作，并仅对定向动作片段进行反馈条件蒸馏，避免了逐回合生成反馈的低效问题，在长时程智能体任务中显著提升性能。 |
| [^200] | [MedFabric: Gold Evidence Hides the Difficulty of Word-Level Medical Fabrication Detection](https://arxiv.org/abs/2605.04180) | MedFabric通过消除风格捷径并提供与虚构内容表面形式高度相似的真实证据，揭示了词级医学虚构检测在实际部署中的真实难度。 |
| [^201] | [TRACES: Tagging Reasoning Steps for Adaptive Cost-Efficient Early-Stopping](https://arxiv.org/abs/2604.21057) | 本文提出TRACES框架，通过实时标记推理步骤并监控推理行为变化，实现大型语言模型推理的自适应成本高效早期停止，显著提升推理效率。 |
| [^202] | [MOMO: A framework for seamless physical, verbal, and graphical robot skill learning and adaptation](https://arxiv.org/abs/2604.20468) | 本文提出了一个名为MOMO的框架，通过融合动觉触摸、自然语言和图形界面三种模态，并结合基于能量的意图检测和工具型LLM架构，实现了非专家用户对机器人技能的安全、直观和灵活适应。 |
| [^203] | [$p1$: Better Prompt Optimization with Fewer Prompts](https://arxiv.org/abs/2604.08801) | 本文发现提示优化的成败取决于系统提示间方差与响应间方差的相对大小，并指出增加用户提示可能降低系统提示间方差，从而提出$p1$方法以更少的提示实现更优的优化效果。 |
| [^204] | [DataSTORM: Deep Research on Large-Scale Databases using Exploratory Data Analysis and Data Storytelling](https://arxiv.org/abs/2604.06474) | DataSTORM通过结合探索性数据分析和数据叙事原理，提出了一种基于LLM代理的系统，能够自主对大型结构化数据库进行迭代假设生成、定量推理和叙事收敛，从而克服了传统深度研究仅聚焦于非结构化数据的局限。 |
| [^205] | [Addressing the Reasoning Gap: Mechanistic Circuit-Based Knowledge Editing in Large Language Models](https://arxiv.org/abs/2604.05876) | 本文提出MCircKE框架，通过识别和调整因果推理回路，弥合了知识编辑中的推理差距，使编辑后的事实能有效用于多步推理。 |
| [^206] | [Grounded Token Initialization for New Vocabulary in LMs for Generative Recommendation](https://arxiv.org/abs/2604.02324) | 本文发现均值初始化导致新令牌退化，提出在预训练嵌入空间中进行语言锚定的初始化方法，以提升生成式推荐中语言模型扩展新词汇的性能。 |
| [^207] | [Translation Asymmetry in LLMs as a Data Augmentation Factor: A Case Study for 6 Romansh Language Varieties](https://arxiv.org/abs/2603.25489) | 该论文发现，在低资源语言（如罗曼什语）的机器翻译中，利用LLM的翻译不对称性，将数据增强方向从生成目标语言改为生成高资源语言，能显著提升性能并首次实现各变体的流畅翻译。 |
| [^208] | [Many Dialects, Many Languages, One Cultural Lens: Evaluating Multilingual VLMs for Bengali Culture Understanding Across Historically Linked Languages and Regional Dialects](https://arxiv.org/abs/2603.21165) | 本文提出BanglaVerse基准，揭示仅用标准孟加拉语评估会高估多语言视觉语言模型的性能，方言变化显著降低其表现，尤其影响图像描述生成。 |
| [^209] | [How LLMs Distort Our Written Language](https://arxiv.org/abs/2603.18161) | 这项研究通过用户研究和数据集分析，揭示大型语言模型在辅助写作时不仅改变写作风格和语气，还显著扭曲了原始意图，导致大量文本内容趋于中立和失去个性。 |
| [^210] | [Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization](https://arxiv.org/abs/2603.16105) | 本文提出ZipCal，一种基于Zipf幂律最大化词汇多样性的模型无关数据策展方法，通过分析数据内在属性而非模型信号，为剪枝和量化高效选择高性能校准数据，显著优于均匀随机采样。 |
| [^211] | [SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models](https://arxiv.org/abs/2603.09215) | SPAR-K通过模态感知的交替深度调度和周期性全深度刷新步骤，在保持语音问答准确性的同时显著降低解码深度，实现了高效的SLM推理。 |
| [^212] | [Modular Expert Merging for Biomedical Retrieval](https://arxiv.org/abs/2602.04731) | 本文提出模块化专家合并方法，通过合成难负样本和LoRA微调领域专家并合并，在生物医学检索上优于大规模混合训练，兼顾通用性能。 |
| [^213] | [Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content](https://arxiv.org/abs/2601.17173) | 本文首次提出导师式问答数据集MentorQA及评估框架，证明多智能体架构在长篇多语言内容中能显著提升超越事实准确性的导师式回应质量。 |
| [^214] | [MAPLE: Metadata Conditioned LLM Pretraining for Locale-Aware Question Answering](https://arxiv.org/abs/2601.15236) | 本文提出MAPLE方法，通过在预训练时加入地理元数据，使语言模型能够根据地域变化切换答案，解决局部知识消歧问题。 |
| [^215] | [CounterVid: Counterfactual Video Generation for Mitigating Action and Temporal Hallucinations in Video-Language Models](https://arxiv.org/abs/2601.04778) | 提出了一种反事实视频生成框架，通过合成仅改变动作或时间顺序的硬负样本视频，有效缓解视频-语言模型中的动作和时间幻觉问题。 |
| [^216] | [AEScorer: An Agentic Evidence-Grounded Framework for Graded Factuality Verification](https://arxiv.org/abs/2601.03605) | AEScorer提出了一种代理式证据获取与分级评分框架，用于区分事实性的细微差异，并通过新基准GradedVeriBench验证了其优越性能。 |
| [^217] | [TopiCLEAR: Adaptive embedding clustering for interpretable topic discovery from short texts](https://arxiv.org/abs/2512.06694) | 本文提出TopiCLEAR框架，通过自适应降维的迭代嵌入聚类，验证了人类可解释主题对应低维结构的假设，从而提升短文本主题发现的解释性。 |
| [^218] | [What the "Spotless" Mind Remembers: How Knowledge Entanglement Shapes What Leaks After Unlearning in LLMs](https://arxiv.org/abs/2510.25732) | 本文发现，大语言模型在遗忘后，知识纠缠度越高的事实越易泄露，且不同遗忘算法（如GA+KL）会反转这一关系，首次揭示其因果影响。 |
| [^219] | [MENTOR: Reinforcement Learning via Flexible Teacher-Optimized Rewards for Tool-Use Distillation](https://arxiv.org/abs/2510.18383) | 本文提出MENTOR，一种基于强化学习的在策略蒸馏框架，通过灵活且过程感知的奖励结构，在行为对齐和性能之间取得平衡，有效解决了将大型语言模型工具使用能力蒸馏到小型模型时的域外泛化问题。 |
| [^220] | [DeepPlanner: Scaling Planning Capability for Deep Research Agents via Advantage Shaping](https://arxiv.org/abs/2510.12979) | DeepPlanner提出了一种端到端强化学习框架，通过熵基优势塑造和样本级加权，有效增强了深度研究代理的规划能力，解决了规划令牌在标准RL下未被充分优化的问题。 |
| [^221] | [LLM-Specific Utility for Retrieval-Augmented Generation](https://arxiv.org/abs/2510.11358) | 本文首次形式化并实证了检索增强生成中证据的LLM特定效用，证明其具有模型依赖性和不可转移性，为优化RAG系统提供了新视角。 |
| [^222] | [Are Large Language Models Effective Knowledge Graph Constructors?](https://arxiv.org/abs/2510.11297) | 本文提出D2A-HKG框架，评估了七个LLM在零样本无模式条件下构建文档知识图谱的能力，发现它们能生成合理结构但在复杂关系处理上不足。 |
| [^223] | [GSM8K-V: Can Vision Language Models Solve Grade School Math Word Problems in Visual Contexts](https://arxiv.org/abs/2509.25160) | 该论文提出 GSM8K-V 基准，将文本数学题转化为多图像序列，发现视觉语言模型在视觉数学推理上存在显著模态差距（最佳模型仅 59%，远低于人类 91%）。 |
| [^224] | [Recurrence Meets Transformers for Universal Multimodal Retrieval](https://arxiv.org/abs/2509.08897) | 本文提出ReT-2，一个结合递归机制与Transformer架构的统一多模态检索模型，支持多模态查询和跨模态文档检索，并在M2KR和M-BEIR基准上达到最先进性能。 |
| [^225] | [LLM Analysis of 150+ years of German Parliamentary Debates on Migration Reveals Shift from Post-War Solidarity to Anti-Solidarity in the Last Decade](https://arxiv.org/abs/2509.07274) | 本文利用大型语言模型对150多年德国议会移民辩论进行大规模分析，发现最强模型（如GPT-5）的标注性能接近人类水平，并揭示出近十年从战后团结向反团结的显著转变。 |
| [^226] | [Do Language Models Follow Occam's Razor? An Evaluation of Parsimony in Inductive and Abductive Reasoning](https://arxiv.org/abs/2509.03345) | 本文首次系统评估了大型语言模型在归纳和溯因推理中是否遵循奥卡姆剃刀原则，并引入了一个可扩展的一阶逻辑推理问题生成框架。 |
| [^227] | [Beyond the Rosetta Stone: Unification Forces in Generalization Dynamics](https://arxiv.org/abs/2508.11017) | 通过合成数据上的小型Transformer实验，本文揭示了跨语言知识迁移的关键在于表示统一性，这取决于信息性和可提取性，并提供了统一理论解释多语言模型中的迁移现象。 |
| [^228] | [AirLLM: Diffusion Policy-based Adaptive LoRA for Remote Fine-Tuning of LLM over the Air](https://arxiv.org/abs/2507.11515) | AirLLM提出了一种基于分层扩散策略的自适应LoRA框架，通过PPO智能体联合考虑无线状态和语言复杂性来动态调整秩配置，从而显著提高空中远程微调的通信效率。 |
| [^229] | [Distinct Profiles of Run-to-Run Score Reliability and Expert-Panel Alignment Across Four LLM Evaluators of Simulated Japanese-Language AI-to-AI Counseling](https://arxiv.org/abs/2507.02950) | 本研究揭示了四种LLM评估器在日语AI咨询评估中，高逐次评分信度并不等同于与专家判断一致，所有系统均倾向于给出高于专家的分数。 |
| [^230] | [From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning](https://arxiv.org/abs/2505.22203) | 本研究发现基于规则的验证器在数学推理中因无法识别等价答案而产生假阴性，阻碍强化学习性能，而基于模型的验证器虽提升静态准确性但存在稳健性问题，强调验证器需从精确性转向稳健性。 |
| [^231] | [Legal Rule Induction: Towards Generalizable Principle Discovery from Analogous Judicial Precedents](https://arxiv.org/abs/2505.14104) | 本文首次形式化定义了法律规则归纳（LRI）任务，并提出了一种可复现的数据集构建流程，利用大语言模型从类似司法先例中自动提取可泛化的法律原则。 |
| [^232] | [Plain Transformers Can be Powerful Graph Learners](https://arxiv.org/abs/2504.12588) | 本文证明普通Transformer架构通过三项简单修改（如简化L2注意力）即可成为强大的图学习器，无需复杂架构改动。 |
| [^233] | [Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling](https://arxiv.org/abs/2504.05216) | 本文提出LLM-QL模型，通过辅助的查询似然最大化任务增强大语言模型的稠密检索能力，利用生成优势改进对比学习。 |
| [^234] | [The BS-meter: Detecting Politics and Labour through ChatGPT's Language](https://arxiv.org/abs/2411.15129) | 本文通过实证对比科学文献与ChatGPT生成文本，提出了一种统计模型来检测“胡扯”语言，并验证其在政治言论和“狗屁工作”语境中的适用性，揭示了LLM语言的独特特征。 |

# 详细

[^1]: CritICL：从小型语言模型失败模式中进行推理时弱到强泛化

    CritICL: Inference-Time Weak-to-Strong Generalization from Small Language Model Failure Modes

    [https://arxiv.org/abs/2608.27455](https://arxiv.org/abs/2608.27455)

    CritICL通过利用弱模型的失败模式作为推理时指导，提出了一种高效且无需外部验证的弱到强泛化框架，显著提升了大语言模型的推理性能。

    

    arXiv:2608.27455v1 公告类型：新 摘要：近年来，推理时扩展的进展显著提升了大语言模型（LLM）的推理性能。然而，这些方法通常依赖重复生成或外部验证。为解决此局限，我们引入了CritICL，一种新颖的推理时框架，能在保持高效率的同时提升推理能力。我们的关键洞见是，在同一模型家族内，LLM的失败模式在不同规模间呈现结构化模式。CritICL不将失败视为不良输出，而是将其用作指导来源。具体而言，我们利用从较弱模型衍生的失败模式，并通过基于批评的上下文示例将其融入推理中。我们提出两种变体：CritICL-dynamic，它能自适应地预测输入特定的失败模式并检索批评；以及CritICL-static，它使用全局失败模式配置文件提供稳定指导。实验结果表明...

    arXiv:2608.27455v1 Announce Type: new  Abstract: Recent advances in inference-time scaling have significantly improved the reasoning performance of large language models (LLMs). However, these methods typically rely on repeated generation or external verification. To address this limitation, we introduce CritICL, a novel inference-time framework that improves reasoning while maintaining high efficiency. Our key insight is that LLM failure modes exhibit structured patterns across model scales within the same family. Instead of treating failures as undesirable outputs, CritICL leverages them as a source of guidance. Specifically, we utilize failure modes derived from weaker models and incorporate them into inference through critique-based in-context examples. We propose two variants: CritICL-dynamic, which adaptively predicts input-specific failure modes and retrieves critiques, and CritICL-static, which uses a global failure mode profile to provide stable guidance. Experimental results 
    
[^2]: WikiSkill：将智能体经验编译为持久知识以促进技能进化

    WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution

    [https://arxiv.org/abs/2608.27454](https://arxiv.org/abs/2608.27454)

    WikiSkill通过将智能体经验持续整合到持久知识库中，实现技能与知识的协同进化，显著提升技能进化效果。

    

    arXiv:2608.27454v1 公告类型：新  摘要：智能体技能将专业知识和操作流程封装为可复用资源，从而扩展AI智能体的能力。近期研究从智能体经验中自动发现此类技能，使智能体能够通过交互逐步适应。然而，指导技能发展的洞察通常分散在优化历史中，限制了其在迭代中的系统性复用。我们提出了WikiSkill，一个将智能体技能与持久知识库（wiki）协同进化的框架。在高层次上，WikiSkill将原始执行经验、积累的知识和可执行技能分离，同时持续将经验整合到wiki中，后续技能更新可在此基础上进行。在多种基准和模型中，WikiSkill持续优于最先进的技能进化方法，并在大多数模型-基准设置中优于无技能基线。我们发现技能进化补充了...

    arXiv:2608.27454v1 Announce Type: new  Abstract: Agent skills package specialized knowledge and workflows into reusable resources that extend AI agent capabilities. Recent work automatically discovers such skills from agent experience, which enables agents to progressively adapt through interaction. However, the insights that guide skill development typically remain scattered across optimization histories, limiting their systematic reuse across iterations. We introduce WikiSkill, a framework that co-evolves agent skills with a persistent knowledge base (wiki). At a high level, WikiSkill separates raw execution experience, accumulated knowledge, and executable skills, while continuously consolidating experience into the wiki, which subsequent skill updates can build on. Across diverse benchmarks and models, WikiSkill consistently outperforms state-of-the-art skill-evolution methods and improves over no-skill baselines in most model-benchmark settings. We find that skill evolution comple
    
[^3]: SWE-Prime：更少的轨迹，更好的性能

    SWE-Prime: Fewer Trajectories, Better Performance

    [https://arxiv.org/abs/2608.27449](https://arxiv.org/abs/2608.27449)

    SWE-Prime通过两阶段多粒度筛选方法，仅保留高质量且具代表性的轨迹片段，以更少的数据实现更好的模型性能。

    

    为了提升大型语言模型解决现实世界软件问题的能力，先前的工作主要集中于构建大规模智能体轨迹数据集，并对成功轨迹进行监督微调（SFT）。然而，任务成功并不保证监督质量高：成功轨迹可能仍包含无效、冗余或危险的步骤。直接使用此类轨迹进行SFT可能会引入噪声监督，并鼓励模型模仿不良的问题解决行为。因此，我们提出SWE-Prime，一种多粒度、两阶段的SFT数据选择方法，逐步在轨迹和片段级别过滤训练数据。具体而言，第一阶段基于过程质量、结果质量和数据代表性进行轨迹级筛选，选择高质量且有代表性的成功轨迹子集。第二阶段进行片段级筛选。

    arXiv:2608.27449v1 Announce Type: cross  Abstract: To improve large language models' ability to resolve real-world software issues, prior work has focused on constructing large-scale agent trajectory datasets and performing supervised fine-tuning (SFT) on successful trajectories. However, task success does not guarantee high-quality supervision: successful trajectories may still contain ineffective, redundant, or risky steps. Directly using such trajectories for SFT can introduce noisy supervision and encourage models to imitate undesirable problem-solving behaviors. Therefore, we propose SWE-Prime, a multi-granularity, two-stage SFT data selection method that progressively filters training data at the trajectory and segment levels. Specifically, the first stage performs trajectory-level screening based on process quality, result quality, and data representativeness, selecting a high-quality and representative subset of successful trajectories. The second stage performs segment-level s
    
[^4]: 测试时策略优化

    TTPO: Test-Time Policy Optimization

    [https://arxiv.org/abs/2608.27448](https://arxiv.org/abs/2608.27448)

    本文提出TTPO，一种不对称的测试时策略优化方法，利用多数投票伪标签，通过蒸馏一致生成和分组RL惩罚不一致生成，实现无需真实标签的测试时训练。

    

    摘要：arXiv:2608.27448v1 公告类型：新 摘要：近年来，诸如强化学习（RL）和同策略自蒸馏（OPSD）等突出的后训练方法，推动了大型语言模型在数学推理方面的快速进展，然而它们对真实标签的依赖阻碍了测试时训练（TTT）。用多数投票伪标签替代真实标签是一种自然的替代方案，但它很脆弱：一个错误的投票会污染教师模型并误导每个令牌。我们观察到这种失败模式是不对称的：与伪标签不一致的生成结果通常无论投票本身是否正确都是错误的。基于这一观察，我们提出了测试时策略优化（TTPO），一种不对称的目标函数，通过OPSD蒸馏一致的生成结果，并用分组RL惩罚不一致的生成结果。令牌级选择进一步细化两个分支：蒸馏降低已收敛位置的权重，而RL仅惩罚置信度高的错误。B

    arXiv:2608.27448v1 Announce Type: new  Abstract: Recent prominent post-training methods, such as Reinforcement Learning (RL) and On-Policy Self-Distillation (OPSD), have driven rapid progress in mathematical reasoning for large language models, yet their reliance on ground-truth labels precludes test-time training (TTT). Replacing ground truth with majority-vote pseudo-labels is a natural alternative, yet it is fragile: an incorrect vote corrupts the teacher and misleads every token. We observe that this failure mode is asymmetric: rollouts that disagree with the pseudo-label are typically wrong regardless of whether the vote itself is correct. Building on this observation, we propose Test-Time Policy Optimization (TTPO), an asymmetric objective that distills agreeing rollouts via OPSD and penalizes disagreeing rollouts with Grouped RL. Token-level selection further refines both branches: distillation down-weights already-converged positions, while RL penalizes only confident errors. B
    
[^5]: 从静态到动态：基于MCR-Bench对真实世界代码评审的基准测试

    From Static to Dynamic: Benchmarking Real-World Code Review with MCR-Bench

    [https://arxiv.org/abs/2608.27442](https://arxiv.org/abs/2608.27442)

    本文提出了MCR-Bench，首个缺陷状态感知的多轮代码评审基准，包含2,269个真实任务和细粒度缺陷标注，以解决现有LLM评审方法过度简化静态化的问题。

    

    在现实世界的软件开发中，代码评审通常涉及开发者和评审者之间的迭代交互以提升软件质量，这使得该过程成本高昂且耗时。尽管近期研究探索了使用大型语言模型（LLMs）进行自动化代码评审，但大多数方法将代码评审过度简化为单轮、静态的决策任务，未能捕捉到多轮交互的特性以及真实评审场景中固有的复杂问题解决过程。为弥补这一差距，我们引入了MCR-Bench，这是首个面向现实多轮代码评审的缺陷状态感知基准。MCR-Bench涵盖五种常用编程语言，包含2,269个真实世界的多轮代码评审任务，每个任务都标注了细粒度的缺陷信息和跨轮状态标签。MCR-Bench中的每个任务都配备了细粒度的缺陷元数据（例如，描述、类型等）。

    arXiv:2608.27442v1 Announce Type: cross  Abstract: In real-world software development, code review typically involves iterative interactions between developers and reviewers to improve software quality, making the process costly and time-consuming. Although recent work explores large language models (LLMs) for automated code review, most approaches oversimplify code review into a single-round, static decision task, which fails to capture the multi-round interactive nature and the complex problem-solving processes inherent in realistic review scenarios. To bridge this gap, we introduce MCR-Bench, the first defect state-aware benchmark designed for realistic multi-round code review. MCR-Bench covers five commonly-used programming languages and consists of 2,269 real-world multi-round code review tasks, each of which is annotated with fine-grained defect information and cross-round state labels. Each task in MCR-Bench is equipped with fine-grained defect metadata (e.g., description, type,
    
[^6]: 随机估计转导语言模型

    Stochastic Estimation of Transduced Language Models

    [https://arxiv.org/abs/2608.27428](https://arxiv.org/abs/2608.27428)

    该论文提出一种通过无放回重采样和逆包含概率加权递归校正的无偏估计方法，用于高效计算转导语言模型中目标前缀概率，并准确估计阈值剪枝带来的质量损失。

    

    摘要：转导语言模型（TLMs）将一个预训练的源语言模型与一个功能有限状态转换器组合，以在目标字符串上诱导出一个语言模型。在TLM下计算目标前缀的概率，需要对转换器映射到以该前缀开头的目标字符串的所有源字符串的源模型概率进行求和。这个集合可能呈指数级增长或无限大。先前的工作使用基于源前缀概率的计算捷径，然后用阈值剪枝的束求和来近似结果和。这产生了一个误差未知的下界。相反，我们无放回地重新采样源前缀，并通过其包含概率的倒数对每个选中的前缀进行加权。我们证明递归应用这种校正可以得到目标前缀概率的无偏估计器，并使我们能够估计由阈值剪枝损失的质量。我们的束求和a

    arXiv:2608.27428v1 Announce Type: new  Abstract: Transduced language models (TLMs) compose a pretrained \emph{source} language model with a functional finite-state transducer to induce a language model over \emph{target} strings. Computing the probability of a target prefix under a TLM amounts to summing the source-model probabilities of all source strings that the transducer maps to target strings beginning with that prefix. This set can be exponentially large or infinite. Prior work uses a computational shortcut based on source prefix probabilities, then approximates the resulting sum with threshold-pruned beam summing. This produces a lower bound with unknown error. Instead, we resample source prefixes without replacement and reweight each selected prefix by the inverse of its inclusion probability. We show that applying this correction recursively gives an unbiased estimator of the target prefix probability and lets us estimate the mass lost by threshold pruning. Our beam-summing a
    
[^7]: 通过弱模型引导提升RLVR中LLM的探索能力

    Boosting LLM Exploration via Weak-Model Guidance in RLVR

    [https://arxiv.org/abs/2608.27420](https://arxiv.org/abs/2608.27420)

    本文提出一种利用弱模型生成的部分推理轨迹作为外部前缀，引导目标模型在RLVR训练中保持生成多样性并增强探索能力的方法。

    

    摘要：基于可验证奖励的强化学习（RLVR）显著提升了LLM的推理能力，但通常会导致策略熵下降，造成推理覆盖范围收窄，并在较大k值下降低pass@k性能。尽管现有方法通过算法正则化缓解了这种熵崩溃问题，但跨模型的非参数扰动仍被忽视。在本工作中，我们提出了一种简单有效的方法，在RLVR过程中保持LLM的生成多样性。我们不依赖内部探索，而是强制目标模型基于由较小、较弱的语言模型生成的部分推理轨迹来生成答案。这些不熟悉的前缀有效打破了过度自信，并鼓励探索不同的推理路径。我们实证研究了外部前缀的潜力，揭示了分布差异对RLVR训练中探索动态的影响机制。

    arXiv:2608.27420v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) significantly improves LLM reasoning but often causes a drop in policy entropy, leading to narrowed reasoning coverage and degraded pass@$k$ for large $k$. While existing methods mitigate this entropy collapse through algorithmic regularizations, cross-model non-parametric perturbation is also neglected. In this work, we propose a simple yet effective approach to preserve the generative diversity of LLMs during RLVR. Instead of relying solely on internal exploration, we force the target model to generate answers based on partial reasoning trajectories generated by a smaller, weaker language models. These unfamiliar prefixes effectively disrupt over-confidence and encourage the exploration of distinct reasoning paths. We empirically study the potential of outer prefixes, revealing the mechanism of the impact of distributional discrepancy to the exploration dynamics in RLVR training. Ex
    
[^8]: 跨领域巩固RLVR能力：融合范式的深入研究

    Consolidating RLVR Capabilities Across Domains: A Deep Dive into Fusion Paradigms

    [https://arxiv.org/abs/2608.27409](https://arxiv.org/abs/2608.27409)

    本文系统比较了三种RLVR能力融合范式（合并、混合强化学习和多教师蒸馏），发现平均性能相近但领域级差异显著，且选择受训练动态和跨领域关系影响。

    

    基于可验证奖励的强化学习（RLVR）提升了大型语言模型的特定能力，但覆盖多种能力通常需要训练单独的领域专家，随后进行整合。我们根据所复用的工件将三种融合范式进行组织：合并（Merge）结合专家任务向量，混合强化学习（Mix RL）汇集其数据集，而多教师策略蒸馏（MOPD）则同时使用两者。由于这些方法大多在孤立环境中研究，它们之间的比较以及如何选择仍不明确。我们使用共享专家和数据，在不同模型规模及多领域基准套件上比较了这三种方法。尽管它们的平均性能差异最多为1.4个百分点，但在单个基准上差距可达8.6个百分点，领域级变化与任务向量几何中可见的跨领域关系相关。训练动态揭示了不同约束：混合强化学习依赖于领域混合比例，而MOPD则重新...（注：摘要被截断，此处按提供内容翻译）

    arXiv:2608.27409v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) improves specific capabilities of large language models, but covering multiple capabilities often involves training separate domain experts and subsequently consolidating them. We organize three fusion paradigms by the artefacts they reuse: Merge combines expert task vectors, Mix RL pools their datasets, and multi-teacher on-policy distillation (MOPD) uses both. Because they have largely been studied in isolation, how they compare and how to choose among them remain unclear. We compare all three using shared experts and data across model scales and a multi-domain benchmark suite. Although their average performance differs by at most 1.4 points, the gap reaches 8.6 points on a single benchmark, with domain-level variation tracking cross-domain relations visible in task-vector geometry. Training dynamics expose distinct constraints: Mix RL depends on domain mixture proportions, MOPD rem
    
[^9]: 语言模型如何组织和结构化道德知识

    How Language Models Organize and Structure Moral Knowledge

    [https://arxiv.org/abs/2608.27402](https://arxiv.org/abs/2608.27402)

    本研究揭示了大型语言模型通过线性探针在表示空间中组织道德知识，其道德方向保持高度独立维度但共享道德特异性的正共同成分，表明模型能区分并整合不同道德基础。

    

    大型语言模型（LLMs）如何组织道德知识？模型能广泛检测道德内容，但检测只是一个低标准。我们探究它们是否更进一步，区分不同的道德基础，并在几何上组织它们之间的关系。我们在开放权重语言模型上训练了六个独立的线性探针，每个对应道德基础理论（MFT）的一个类别（关怀/伤害、公平/欺骗、自由/压迫、忠诚/背叛、权威/颠覆、神圣/堕落），并检查这些方向在表示空间中如何相互关联。我们发现这些方向既没有坍缩成单一的道德检测器，也没有相互隔离。相反，它们跨越了近最大数量的独立维度，同时共享一个正共同成分。该共享成分是整合的标志，并且相对于以相同方式构建的匹配非道德概念电池，它是道德特异的（平均成对余弦相似度为0.26对比0.013）。

    arXiv:2608.27402v1 Announce Type: cross  Abstract: How do large language models (LLMs) organize moral knowledge? Models detect moral content broadly, but detection is a low bar. We ask whether they go further, distinguishing moral foundations from one another and organizing the relationships between them geometrically.   We train six independent linear probes on open-weight language models, one per Moral Foundations Theory (MFT) category (care/harm, fair/cheat, lib/oppress, loy/betray, auth/subv, sanc/degrade), and examine how the resulting directions relate to each other in representation space. We find the directions neither collapse into a single moral detector nor isolate from one another. Rather, they span a near-maximal number of independent dimensions while sharing a positive common component. The shared component is the signature of integration, and it is moral-specific relative to a matched non-moral concept battery built identically (mean pairwise cosine 0.26 vs. 0.013).   Th
    
[^10]: 使临床语言模型可审计：概念引导的微调以实现稳健预测

    Making Clinical Language Models Auditable: Concept-Guided Fine-Tuning for Robust Prediction

    [https://arxiv.org/abs/2608.27397](https://arxiv.org/abs/2608.27397)

    本文提出CAST框架，通过稀疏自编码器暴露并抑制临床语言模型中的笔记伪影，实现可审计且稳健的预测，优于微调编码器并与LLM基线竞争。

    

    arXiv:2608.27397v1 公告类型：交叉 摘要：临床语言模型在院内预测中可以达到较高准确性，但在部署环境变化下会失效，因为它们利用了不反映患者状态的笔记特定伪影（如模板、分隔符、样板文本）。我们提出了CAST（概念引导伪影抑制调优），一种基于SAE的可审计临床文本分类框架。CAST使用稀疏自编码器从中间Transformer激活中提取稀疏、人类可审计的特征，通过LLM辅助解释流程和ICD-10检索约束标记SAE潜在变量，在微调期间通过残差减法抑制已验证的伪影潜在变量，并提供事后逐概念归因以审计模型决策。在MIMIC-IV出院记录死亡率预测中，CAST相对于相应的微调编码器基线有所改进，并与强大的LLM基线保持竞争力，同时生成临床特征级别的审计轨迹。

    arXiv:2608.27397v1 Announce Type: cross  Abstract: Clinical language models can achieve strong in-hospital accuracy yet fail under deployment shifts because they exploit note-specific artifacts (e.g., templates, separators, boilerplate) that do not reflect patient state. We propose CAST (Concept-guided Artifact Suppression Tuning), an SAE-based framework for auditable clinical text classification. CAST uses Sparse Autoencoders to expose sparse, human-auditable features from intermediate Transformer activations, labels SAE latents with an LLM-assisted interpretation pipeline and ICD-10 retrieval constraints, suppresses verified artifact latents via residual subtraction during fine-tuning, and provides post-hoc per-concept attributions for auditing model decisions. On MIMIC-IV discharge-note mortality prediction, CAST improves over its corresponding fine-tuned encoder baselines and remains competitive with strong LLM baselines, while producing a feature-level audit trail of the clinical 
    
[^11]: RATIO：科学文献中跨类型构思操作检索的基准

    RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

    [https://arxiv.org/abs/2608.27394](https://arxiv.org/abs/2608.27394)

    RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。

    

    arXiv:2608.27394v1 公告类型：新 摘要：检索到的科学文献可以为人与AI科学家提供灵感。灵感可以采取不同形式：先前的工作可能直接建议如何解决问题，或在不同抽象层次上指出方向——放大到更一般的视角或缩小到具体实现。我们引入RATIO（跨类型构思操作检索），这是一个大规模基准，其中相关性由三种操作定义，我们称之为构思动作：Address检索针对所提出问题的潜在方法，Broaden检索更一般的表述，Specify检索具体实例。RATIO是通过一种通用方法从CS文献中数百万篇全文科学论文构建而成，该方法将话语标记远距离监督——先前仅用于分类——扩展到语料库级检索，并结合了广泛的LLM和人工审核。实验表明，操作-

    arXiv:2608.27394v1 Announce Type: new  Abstract: Retrieved scientific literature can serve as inspiration for both human and AI scientists. Inspiration can take different forms: prior work may directly suggest how to address a problem, or surface directions at different levels of abstraction - zooming out to a more general view or zooming in to a concrete realization. We introduce RATIO (Retrieval Across Typed Ideation Operations), a large-scale benchmark in which relevance is defined by three operations which we name ideation moves: Address retrieves potential approaches for stated problems, Broaden retrieves more general formulations, and Specify retrieves concrete instantiations. RATIO is constructed from millions of full-text scientific papers across CS literature via a general recipe that extends discourse-marker distant supervision - previously used only for classification - to corpus-scale retrieval, combined with extensive LLM and human vetting. Experiments show that operation-
    
[^12]: 企业基准：基于时间知识库的大规模问答基准测试

    CorporateBench: Large-Scale Q&A Benchmarking with Temporal Knowledge Bases

    [https://arxiv.org/abs/2608.27391](https://arxiv.org/abs/2608.27391)

    企业基准（CB）是一个大规模、人工验证的多任务问答基准，通过超过23万份文档的时间知识库评估LLMs，揭示了在现实规模下性能显著下降的问题。

    

    arXiv:2608.27391v1 公告类型：新  摘要：大型语言模型（LLMs）越来越能够回答关于企业级文档集合的复杂问题。但评估很困难：公司不愿分享内部通信，而合成数据集往往过于简单。我们提出了企业基准（CB），一个经过人工验证的多任务问答基准，其规模接近LLMs在企业通信网络中遇到的条件，评估语料库超过230,000份文档。CB通过四个合成生成的公司（员工规模从12到10,000不等）评估LLMs在两个维度（信息提取和知识库查询）上的表现。每个语料库都从一个随时间演化的知识库中采样，描述一个一致的世界，确保即使在数十万份文档中也能保证跨文档的逻辑一致性。我们在CB上评估了五个LLMs，结果显示当输入规模接近实际尺度时，性能显著下降。CB为LLM开发者提供了一个...

    arXiv:2608.27391v1 Announce Type: new  Abstract: LLMs are increasingly able to answer complex questions about enterprise-scale document collections. But evaluation is hard: companies don't want to share internal communications, and synthetic datasets have been overly simple. We present CorporateBench (CB), a human-validated multi-task Q&A benchmark whose scale approaches the conditions LLMs encounter in corporate communication networks, with evaluation corpora surpassing 230,000 documents. CB evaluates LLMs across two dimensions (information extraction and knowledge base querying) through four synthetically generated firms ranging from 12 to 10,000 employees. Each corpus is sampled from a temporally evolving knowledge base describing a consistent world, guaranteeing cross-document logical consistency even across hundreds of thousands of documents. We evaluate five LLMs on CB, revealing increasingly poor performance as input size approaches realistic scales. CB provides LLM developers a
    
[^13]: D2C-Routing：面向混合来源AI生成文本检测的维度到组合证据路由

    D2C-Routing: Dimension-to-Composition Evidence Routing for Mixed-Origin AI-Generated Text Detection

    [https://arxiv.org/abs/2608.27380](https://arxiv.org/abs/2608.27380)

    本文提出D2C-Routing方法，通过将内容来源和表达来源分离推断并组合，显著提升了混合来源AI生成文本的检测性能。

    

    AI生成文本检测通常被构建为关于文本是人工撰写还是机器生成的二元文档级判断。这种框架在混合来源写作中失效，因为内容来源和表达来源可能不同。我们将混合来源检测视为维度到组合的来源归属问题，在推断内容来源和表达来源后，将其组合成四种协作类型。我们提出维度到组合路由（D2C-Routing），该方法在内容侧和表达侧证据被路由到监督维度头之前，通过一个学习到的门控组合层预测最终标签。在MixD2C（从HART混合来源基准重建的分割）上，我们公开的基于D2C-Routing的检测系统达到0.8603的四路平均TPR@1%FPR，比同分割的RACE本地重跑高6.5个百分点。核心消融实验支持路由设计，而错误分析显示区分AI与人类在混合场景中的挑战。

    arXiv:2608.27380v1 Announce Type: new  Abstract: AI-generated text detection is commonly framed as a binary document-level judgment about whether a text is human-written or machine-generated. This framing breaks down for mixed-origin writing, where content origin and expression origin may differ. We cast mixed-origin detection as dimension-to-composition source attribution, inferring content origin and expression origin before composing them into four collaboration types. We propose Dimension-to-Composition Routing (D2C-Routing), which routes content-side and expression-side evidence to supervised dimension heads before a learned gated composition layer predicts the final label. On MixD2C, a reconstructed split derived from the HART mixed-origin benchmark, our disclosed D2C-Routing-based detector system reaches 0.8603 four-way Avg TPR@1%FPR, 6.5 points above the same-split RACE-local rerun. Core ablations support the routing design, while error analysis shows that distinguishing AI-con
    
[^14]: Puro-2B：在RTX 5090上花费5090美元训练的“穷人版”Qwen2-1.5B

    Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090

    [https://arxiv.org/abs/2608.27370](https://arxiv.org/abs/2608.27370)

    本文提出了一种开源且成本高效的预训练配方，使在消费级RTX 5090 GPU上以极低计算成本训练出接近Qwen2.5-1.5B性能的Puro-2B模型成为可能。

    

    arXiv:2608.27370v1 公告类型：新  摘要：语言模型预训练几乎已成为高昂成本的代名词，使其对学术界和开源社区的大部分人来说遥不可及。尽管已有强大的开源努力，包括开放权重模型和开源训练配方，但长期以来一直缺少一种成本高效、硬件可访问且开源预训练配方。即使在小规模下，训练Llama-3.2-3B也需花费超过150万美元，而复现SmolLM3-3B则需超过70万美元。在本报告中，我们提出了一种旨在降低这一门槛的开源预训练配方。利用该配方，我们从零开始训练了一系列Puro-2B模型，使用FP8精度，在消费级RTX 5090 GPU上处理多达1.4万亿个令牌。该系列中的模型在令牌预算和所选配方变体上有所不同。我们最好的模型以不到6900美元的计算成本进行训练，并在我们的评估协议下接近Qwen2.5-1.5B的性能。这种成本效益...

    arXiv:2608.27370v1 Announce Type: new  Abstract: Language model pretraining has become almost synonymous with prohibitive cost, placing it out of reach for much of the academic and open-source communities. Although strong open-source efforts already exist, including open-weight models and open-source training recipes, a cost-efficient, hardware-accessible, and open-source pretraining recipe has long been missing. Even at a small scale, training Llama-3.2-3B costs over \$1.5M, and reproducing SmolLM3-3B needs over \$700K. In this report, we present an open pretraining recipe designed to lower this barrier. Using this recipe, we train a collection of Puro-2B models from scratch on up to 1.4 trillion tokens with FP8 precision on consumer-grade RTX 5090 GPUs. The models in the collection differ in token budgets and selected recipe variants. Our best model is trained at a compute cost of less than \$6.9K and approaches Qwen2.5-1.5B performance under our evaluation protocol. This cost effici
    
[^15]: 你的语音克隆系统其实是语音匿名化器

    Your Voice Cloning System is Secretly a Voice Anonymizer

    [https://arxiv.org/abs/2608.27360](https://arxiv.org/abs/2608.27360)

    我们利用现成的语音克隆模型XTTSv2，无需重新训练即可实现高效的说话人匿名化，通过迭代策略平衡隐私与语音质量，在多语言测试中达到接近最优的隐私保护。

    

    arXiv:2608.27360v1 公告类型：新 摘要：说话人匿名化旨在从语音中抑制说话人身份特征，同时保留语言内容和质量。我们提出将XTTSv2——一个在27,000小时语音上训练的多语言语音克隆模型——重新用于说话人匿名化，而无需重新训练。我们的关键见解是，XTTSv2的语音克隆能力能够独立于说话人身份保留韵律结构，从而通过基于伪说话人进行条件化实现语音转换。我们引入了一种迭代优化策略，通过最大化说话人差异度和可理解度的调和平均值来平衡隐私与实用性。在CommonVoice和多语言LibriSpeech上跨越七种欧洲语言的评估中，我们的系统实现了接近最优的隐私保护（等错误率≈0.49）、有竞争力的可理解度，并且语音质量显著优于专门的匿名化基线，同时无需特定语言的训练。我们在此发布代码：https://

    arXiv:2608.27360v1 Announce Type: new  Abstract: Speaker anonymization suppresses speaker-identifying attributes from speech while preserving linguistic content and quality. We propose repurposing XTTSv2, a multilingual voice cloning model trained on 27k hours of speech, for speaker anonymization without retraining. Our key insight is that XTTSv2's voice cloning capabilities preserve prosodic structure independently of speaker identity, enabling voice conversion by conditioning on a pseudo-speaker. We introduce an iterative refinement strategy that balances privacy and utility by maximizing a harmonic mean of speaker dissimilarity and intelligibility. Evaluated on seven European languages across CommonVoice and Multilingual LibriSpeech, our system achieves near-optimal privacy (EER $\approx$ 0.49), competitive intelligibility, and substantially better speech quality than dedicated anonymization baselines, while requiring no language-specific training. We release the code here: https://
    
[^16]: RCMN：理解有影响力的公共话语中的误导性

    RCMN: Understanding Misleadingness in Influential Public Discourse

    [https://arxiv.org/abs/2608.27358](https://arxiv.org/abs/2608.27358)

    本文提出了一个以读者为中心的误导性理解框架（RCMN），通过五个维度系统分析有影响力公共话语中的误导性，并构建了基于证据的数据集，揭示了误导性远超捏造，涵盖推断、夸大和省略等多种机制。

    

    有影响力的公共话语塑造公众信念，也可能产生误导，不仅通过其陈述内容，还通过信息的框架、省略、情境化和传播方式。然而，较少研究关注这种误导性如何产生并影响读者形成的解读。为填补这一空白，我们引入了以读者为中心的误导性理解框架（RCMN），该框架通过五个维度来操作化误导性：误导机制、可能的读者解读、基于证据的解读、情绪唤起和交际意图。基于此框架，我们构建了一个基于证据的有影响力公共话语数据集。实证结果表明，误导性具有多样性，远不止于捏造，其中无根据的推断、夸大和省略是常见机制，并且常与情绪唤起增强和扭曲相关联。

    arXiv:2608.27358v1 Announce Type: cross  Abstract: Influential public discourse shapes public beliefs and can also mislead, not only through what is stated, but also through how information is framed, omitted, contextualised, and communicated. Yet less research has focused on how such misleadingness arises and shapes the interpretations formed by readers. To address this gap, we introduce Reader-Centric Misleadingness Understanding (RCMN), a framework that operationalises misleadingness through five dimensions: misleading mechanism, likely reader interpretation, evidence-warranted interpretation, emotional arousal, and communicative intent. Based on this framework, we construct an evidence-grounded dataset of influential public discourse. Empirical findings show that misleadingness is diverse and extends well beyond fabrication, with unsupported inference, exaggeration, and omission among the prevalent mechanisms, and is frequently associated with heightened emotional arousal and disto
    
[^17]: 意图即工具：轻松追踪代理性错位

    INTENT-AS-A-TOOL Makes it Easy to Track Agentic Misalignment

    [https://arxiv.org/abs/2608.27348](https://arxiv.org/abs/2608.27348)

    本文提出“意图即工具”方法，通过为模型添加意图专用工具，利用工具调用概率提供细粒度的无评判信号，以追踪和干预代理性错位。

    

    随着大型语言模型（LLMs）被部署为自主代理，安全故障越来越多地涉及具有后果性的行动。我们研究了代理性错位，即代理在目标冲突和压力下采取有害行动的情况。通过链式思维（CoT）监控，我们发现有害执行往往在推理过程中有意图信号的前兆。然而，事后CoT标签过于粗糙，无法显示生成过程中意图的变化。我们引入了“意图即工具”（INTENT-AS-A-TOOL）方法，该方法添加了针对意图的工具，为模型提供专用通道来表达对目标行为的承诺。调用意图工具的概率提供了一种无需评判的、细粒度的信号，指示模型追求该行为的倾向。我们的结果表明，INTENT-AS-A-TOOL补充了CoT监控，将事后CoT标签扩展为密集轨迹，并识别出在线干预的关键步骤。这些发现表明，行动偏好可以用于早期检测和干预代理性错位。

    arXiv:2608.27348v1 Announce Type: new  Abstract: As large language models (LLMs) are deployed as autonomous agents, safety failures increasingly involve consequential actions. We study agentic misalignment, where agents take harmful actions under goal conflicts and pressures. Using chain-of-thought (CoT) monitoring, we find that harmful execution is often preceded by intent signals in reasoning. However, post-hoc CoT labels are too coarse to show how intent changes during generation. We introduce INTENT-AS-A-TOOL, an approach that adds intent-targeted tools to give the model a dedicated channel for expressing commitment to a target behavior. The probability of calling an intent tool provides a judge-free, fine-grained signal of the model's tendency to pursue that behavior. Our results show that INTENT-AS-A-TOOL complements CoT monitoring, expands post-hoc CoT labels into dense trajectories, and identifies critical steps for online intervention. These findings suggest that action prefer
    
[^18]: 从碎片化历史文本复用中恢复论文级别的再版与重用：一项关于十八世纪书籍与报纸的工作流研究

    Pair-Level Essay-Scale Republication and Reuse from Fragmented Historical Text Reuse: A Workflow Study on Eighteenth-Century Books and Newspapers

    [https://arxiv.org/abs/2608.27343](https://arxiv.org/abs/2608.27343)

    本文提出一种对级证据整合的工作流方法，从碎片化文本复用中有效恢复论文级别的再版与重用，在十八世纪书籍和报纸数据集上实现了高F1分数。

    

    本文探讨了从碎片化的文本复用证据中恢复论文级别的再版与重用，其核心挑战在于对级证据整合，而不仅仅是片段检索。研究聚焦于以十八世纪苏格兰哲学家大卫·休谟的论文为中心的候选集，涵盖ECCO（十八世纪在线文集）中的书籍和历史报纸。由于输入是由碎片化的复用命中而非干净的文档对组成，且正例覆盖本质上不完整，我们将任务表述为对级证据整合以形成合理的传播关系，并比较三种方法论家族：分阶段基于规则的工作流、基线方法（决策树和两种直接LLM设置）以及自动化规则适应。在标注的ECCO-ECCO切片上，仅对级特征聚合在主标注切片上已达到0.948的F1分数，而最终工作流进一步优化了性能。

    arXiv:2608.27343v1 Announce Type: new  Abstract: This paper addresses the recovery of essay-scale republication and reuse from fragmented text-reuse evidence, a setting whose central challenge is pair-level evidence consolidation and not fragment retrieval alone. The study focuses on a candidate set centered on essays by eighteenth-century Scottish philosopher David Hume, spanning books from ECCO (Eighteenth Century Collections Online) and historical newspapers. Because the input consists of fragmented reuse hits instead of clean document pairs, and positive coverage is inherently incomplete, we formulate the task as pair-level evidence consolidation into plausible transmission relations and compare three methodological families: a staged rule-based workflow, baselines (a decision tree and two direct LLM settings), and automated rule adaptation. On labeled ECCO--ECCO slices, pair-level feature aggregation alone already reaches 0.948 F1 on the main labeled slice, while the final workflo
    
[^19]: 超越并行盲目性：块草拟中的信息下限与模型差距

    Beyond Parallel Blindness: Information Floors and Model Gaps in Block Drafting

    [https://arxiv.org/abs/2608.27339](https://arxiv.org/abs/2608.27339)

    本文提出一种方法，通过信息下限和模型差距的分离，揭示了块草拟中并行生成的固有信息瓶颈，并指出当前草拟器仍有大幅改进空间。

    

    arXiv:2608.27339v1 公告类型：交叉 摘要：块草拟器在一次前向传播中提出多个令牌，而此时较早的目标令牌尚未实现。它们的拒绝混合了两种损失：缺失块内路径信息和可观察信息的不完美建模。接受长度无法区分这两种损失。我们通过一个信息下限将两者分开，该下限是在指定条件顺序下最小期望拒绝率；超过此下限的拒绝部分即为模型差距。我们通过四个领域的目标滚动、四个开源权重目标和一个前沿API目标来估计这两者，得出了三个发现。首先，在Qwen3-4B的最后一个槽位上，全并行下限达到$0.286$，这限制了即使是最佳提案也只能达到每槽位$71\%$的接受率。其次，一个已实现的令牌可以消除$86$--$100\%$的这个下限，这种局部性也通过独立的互信息分析得到了验证。第三，当前的草拟器仍远高于其下限：最终槽位的模型差距占DFlash拒绝的$43$--$64\%$。

    arXiv:2608.27339v1 Announce Type: cross  Abstract: Block drafters propose several tokens in one forward pass, before earlier target tokens are realised. Their rejection mixes two losses: missing within-block path information and imperfect modelling of observable information. Accepted length cannot distinguish them. We separate the two with an information floor, the minimum expected rejection at a specified conditioning order; rejection above this floor is the model gap. Estimating both from target rollouts across four domains, four open-weight targets, and a frontier API target yields three findings. First, the all-parallel floor reaches $0.286$ at the final slot on Qwen3-4B, limiting even the best proposal to $71\%$ per-slot acceptance. Second, one realised token removes $86$--$100\%$ of this floor, a locality also recovered by an independent mutual-information analysis. Third, current drafters remain far above their floors: the final-slot model gap accounts for $43$--$64\%$ of DFlash
    
[^20]: BTS-AgentBench：从只读遥测日志到代理基准测试的确定性、可重放流水线

    BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks

    [https://arxiv.org/abs/2608.27334](https://arxiv.org/abs/2608.27334)

    该论文提出了一种从只读遥测日志到代理基准测试的确定性、可重放构建流水线，通过标准化工具存储和情节编译，实现了精确复现和零缺陷的基准测试生成。

    

    arXiv:2608.27334v1 公告类型：新 摘要：工业现场包含大量只读遥测数据，但很少有基准测试规定如何将这些记录编译为可执行的多轮代理任务。我们提出了一种遥测到情节的构建方法，并将其实例化为BTS-AgentBench。该流水线将BTS元数据和原始历史记录规范化为只读工具存储，编译具有工具派生黄金答案和证据的静态任务，并将保留的任务提升为类型化、有界、面向操作员的情节。发布的532行版本增加了澄清、目标修订、时间戳策略、质量门控报告和证据归因，同时保留了源计算和分割。编码契约预检报告零发现，构建排除控制器完成0/532行。两个独立的原始到情节构建匹配所有11个逻辑工具存储导出，并精确复现发布的356/87/89训练/开发/测试工件。将共享构建路径应用于XAI4HEAT。

    arXiv:2608.27334v1 Announce Type: new  Abstract: Industrial sites contain large volumes of read-only telemetry, but few benchmarks specify how to compile these records into executable multi-turn agent tasks. We present a telemetry-to-episode construction method instantiated as BTS-AgentBench. The pipeline normalizes BTS metadata and raw histories into a read-only tool store, compiles static tasks with tool-derived gold answers and evidence, and lifts retained tasks into typed, bounded operator-facing episodes. The 532-row release adds clarification, goal revision, timestamp policy, quality-gated reporting, and evidence attribution while preserving the source computation and split. Coded contract preflight reports zero findings, and the construction-exclusion controller completes 0/532 rows. Two independent raw-to-episode builds match all 11 logical tool-store exports and reproduce the released 356/87/89 train/dev/test artifact exactly. Applying the shared construction path to XAI4HEAT 
    
[^21]: 在截断评分量表上的双重差分法可能制造虚假效应：来自预注册大语言模型裁判审计的证据

    Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit

    [https://arxiv.org/abs/2608.27309](https://arxiv.org/abs/2608.27309)

    本文揭示双重差分法在截断评分量表上因截断不均会制造虚假交互效应，并通过预注册审计实证证明该偏差可导致无效结果。

    

    arXiv:2608.27309v1 公告类型：交叉 摘要：对大语言模型裁判的审计通过对比匹配条件来验证偏差，而最严谨的设计会进行两次差分：在两项候选回答之间进行项目内对比，再在操控的属性上进行二次差分，最终从有界评分量表上读取结果。我们证明，这一终点在该量表上无法被识别。双重差分的每一项都受到各自截断比例的影响，因此观察到的统计量混淆了差异偏好与差异衰减：当两项回答以不平等的方式截断时，一个共同作用于两者的严重性偏移会制造出交互效应，因为距界限的不等距离使它们恰好落在良好刺激所在的位置。我们在一个冻结教学裁判的预注册审计中展示了这一失败，该审计在首次990次调用之前就已密封。注册的主要终点，即陈述的学习者画像对裁判脚手架偏好的影响，为零：+0.085分（95% BCa置信区间）。

    arXiv:2608.27309v1 Announce Type: cross  Abstract: Audits of LLM judges certify a bias by contrasting matched conditions, and the strongest designs difference twice: a within-item contrast between two candidate responses, differenced again across a manipulated attribute, read off a bounded rating scale. We show that this endpoint is not identified on the scale that reports it. Each term of the double difference is censored by its own share, so the observed statistic confounds differential preference with differential attenuation: a severity shift common to both responses manufactures an interaction whenever the two censor it unequally, as unequal distances from the bounds make them, exactly where good stimuli place them. We exhibit the failure inside a pre-registered audit of a frozen pedagogy judge, sealed before the first of its 990 calls. The registered primary endpoint, the effect of a stated learner profile on the judge's scaffolding preference, is null: $+0.085$ points (95\% BCa 
    
[^22]: BrailleBench：探究大型语言模型中的多标准盲文理解能力

    BrailleBench: Investigating Multi-Criteria Braille Comprehension in Large Language Models

    [https://arxiv.org/abs/2608.27268](https://arxiv.org/abs/2608.27268)

    本文提出了BrailleBench基准，首次系统评估大型语言模型在多种盲文标准下的理解能力，涵盖数学、常识和多跳问答等任务，并设计了不同的交互配置来检验模型对盲文内容的全面处理。

    

    摘要：尽管大型语言模型（LLMs）在获取知识和计算辅助方面起着中介作用，但它们的能力应以同样的方式惠及弱势群体。然而，目前尚不清楚现有的人工智能系统是否足够包容，使盲人和聋盲用户能够通过盲文访问相同的功能，因为盲文的指示符、缩略形式和数字表示对模型理解提出了独特的要求。为此，我们引入了BrailleBench，一个从不同标准评估LLMs盲文理解能力的基准。BrailleBench对齐了来自五个数据集的5,570个实例，涵盖数学、常识和多跳问答，跨越英语以及盲文一级和二级。设计了不同的配置，以了解系统是否能理解盲文撰写的内容、用盲文表达答案，并完成端到端的盲文交互。为确保质量并防止评估偏差，我们还进行了人工验证。

    arXiv:2608.27268v1 Announce Type: new  Abstract: Although Large language models (LLMs) mediate access to knowledge and computational assistance, their capabilities should benefit vulnerable groups in the same way. However, it is unclear whether existing AI systems are inclusive enough for blind and deafblind users to access the same functionality through Braille, whose indicators, contractions, and digital representations introduce distinct requirements for model comprehension. To this end, we introduce BrailleBench, a benchmark for evaluating LLMs in Braille comprehension from different Criteria. BrailleBench aligns 5,570 instances from five datasets, including mathematics, commonsense, and multi-hop question answering across English and Braille Grades 1 and 2. Different configurations are designed to understand whether the systems can comprehend Braille-authored content, express answers in Braille, and complete end-to-end Braille interaction. To ensure the quality and prevent evaluat
    
[^23]: 朴素提示优化：重新思考复杂提示搜索的必要性

    Naive Prompt Optimization: Rethinking the Need for Complex Prompt Search

    [https://arxiv.org/abs/2608.27266](https://arxiv.org/abs/2608.27266)

    本文提出朴素提示优化（NPO），一种轻量级单谱系方法，通过教师模型和回滚反馈迭代修订提示，在减少计算开销的同时达到与复杂优化器相当或更优的性能，并证明更强的教师推理能力可部分替代搜索复杂性。

    

    arXiv:2608.27266v1 公告类型：新 摘要：高效改进跨多样任务的自主代理是加速代理型AI中递归自我改进（RSI）的核心，提示优化作为一种有前景的方法，能够提供与微调模型权重相当的性能提升，同时降低优化和推理阶段的计算成本。然而，近期发展越来越倾向于不必要的复杂提示优化器。我们引入了朴素提示优化（NPO），这是一种轻量级单谱系方法，利用教师模型和回滚反馈迭代修订提示。NPO在更少的回滚次数下实现了与GEPA相当或更好的性能，并且其优势随着教师模型强度的增加而增强，表明更强的教师推理能力可以部分替代优化器侧的搜索复杂性。在交互式游戏中，NPO与GEPA保持广泛竞争力，而GRPO在某些方面表现更优。

    arXiv:2608.27266v1 Announce Type: new  Abstract: Efficiently improving autonomous agents across diverse tasks is central to accelerating recursive self-improvement (RSI) in agentic AI, with prompt optimization emerging as a promising approach capable of delivering performance gains comparable to those achieved by fine-tuning model weights, while reducing computational costs in both optimization and serving. However, recent developments increasingly favor unnecessarily complex prompt optimizers. We introduce Naive Prompt Optimization (NPO), a lightweight single-lineage method that iteratively revises prompts using a teacher model with rollout feedback. NPO achieves comparable or better performance than GEPA with fewer rollouts, and its advantage increases with stronger teacher models, suggesting that stronger teacher reasoning can partially substitute for optimizer-side search complexity. In interactive games, NPO remains broadly competitive with GEPA, while GRPO performs better on some
    
[^24]: SCIT：测试潜在链式思维模型中的因果缓存载体

    SCIT: Testing Causal Cache Carriers in Latent Chain-of-Thought Models

    [https://arxiv.org/abs/2608.27265](https://arxiv.org/abs/2608.27265)

    SCIT提出了一种新的因果测试协议，证明在潜在链式思维模型中，反事实算术主要通过值缓存后缀轨迹传递，而非隐藏状态或其他组件。

    

    潜在链式思维模型将中间推理从输出的文本转移到连续状态中，提高了紧凑性，但隐藏了因果对象。我们引入了SCIT（后缀缓存互换测试），这是一种因果协议，它构建精确的源-接收方反事实，修补声明的缓存段，并识别哪个变换器对象携带反事实计算。SCIT结合了充分性测试与K/V组件分割、隐藏状态控制、语义源控制、解码验证和匹配破坏。在CODI-GPT2和一个Sim-CoT风格的GPT-2复现中，反事实算术主要通过值缓存后缀轨迹传递，而非隐藏状态、键、可复用答案槽或单令牌触发器。对于主要的CODI-GPT2检查点，晚期值后缀机制的完整充分性和必要性证据成立；Sim-CoT风格的检查点显示出相同的充分性和解码结果。

    arXiv:2608.27265v1 Announce Type: new  Abstract: Latent chain-of-thought models move intermediate reasoning from emitted text into continuous states, improving compactness but hiding the causal object. We introduce SCIT, the Suffix Cache Interchange Test, a causal protocol that constructs exact source-recipient counterfactuals, patches declared cache segments, and identifies which transformer object carries the counterfactual computation. SCIT combines sufficiency tests with K/V component splits, hidden-state controls, semantic source controls, decoded validation, and matched corruption. On CODI-GPT2 and a Sim-CoT-style GPT-2 reproduction, counterfactual arithmetic transfers primarily through value-cache suffix trajectories rather than hidden states, keys, reusable answer slots, or single-token triggers. Complete sufficiency-and-necessity evidence for the late-value-suffix mechanism holds for the main CODI-GPT2 checkpoint; the Sim-CoT-style checkpoint shows the same sufficiency and dec
    
[^25]: 什么造就了好的智能体数据？一个关于大语言模型智能体数据生成的ACE视角

    What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents

    [https://arxiv.org/abs/2608.27260](https://arxiv.org/abs/2608.27260)

    本文提出了一个统一的ACE框架，将智能体数据分解为$(E,q,\tau,v)$，并通过准确性、复杂性和多样性三个维度来指导数据生成，以解决现有工作中领域孤立和评估混乱的问题。

    

    大语言模型智能体越来越依赖生成的交互数据来学习如何与外部环境交互。智能体数据生成必须在环境、任务、交互和成功信号之间保持一致性，同时产生有用而非仅仅丰富的经验。现有工作覆盖了许多智能体领域，但以领域为中心的组织和异构评估常常掩盖了共同的生成机制，并将候选构建与验证和选择混为一谈。本研究为该领域开发了一个两层级框架。首先，我们将智能体数据表示为一个共同的因子化对象$(E,q,\tau,v)$，包括环境规范、任务信号、交互实现和可选验证器。我们按其主要锚点和依赖结构来组织生成范式。其次，我们通过准确性-复杂性-多样性（ACE）视角将生成公式化为约束分布设计。

    arXiv:2608.27260v1 Announce Type: new  Abstract: LLM agents increasingly rely on generated interaction data to learn how to interact with external environments. Agentic data generation must maintain consistency among environments, tasks, interactions, and success signals while producing experience that is useful rather than merely abundant. Existing work spans many agent domains, but domain-centered organization and heterogeneous evaluation often obscure common generation mechanisms and conflate candidate construction with verification and selection. This work develops a two-level framework for the field. First, we represent agentic data as a common factorized object $(E,q,\tau,v)$, comprising an environment specification, task signal, interaction realization, and optional verifier. We organize generation paradigms by their primary anchor and dependency structure. Second, we formulate generation as constrained distribution design through the Accuracy-Complexity-divErsity (ACE) lens. Ac
    
[^26]: BALMS：面向纵向心理健康感知的智能体大语言模型基准测试

    BALMS: Benchmarking Agentic LLMs for Longitudinal Mental Health Sensing

    [https://arxiv.org/abs/2608.27219](https://arxiv.org/abs/2608.27219)

    该论文提出了BALMS，这是首个针对纵向心理健康感知的智能体大语言模型基准测试，旨在评估智能体对长期信号推理并预测健康分数及生成基于证据推理的能力。

    

    心理健康评估依赖于偶发性的自我报告量表，这些量表将压力等主观状态转化为数值分数，但仅能提供稀疏的健康状况快照。可穿戴设备提供纵向的行为和生理信号，用于连续、低负担的监测。近年来，由大语言模型驱动的个人健康智能体能够对可穿戴信号进行自然语言查询，但主要处理短期的、基于检索的查询（例如，一周内的最高步数）。它们并未评估智能体是否能对长期信号进行推理，以预测与基于证据的推理相匹配的健康分数。为解决这一空白，我们引入了BALMS，这是首个针对纵向心理健康感知的基于大语言模型的智能体系统的系统性基准测试。BALMS涵盖3个真实世界的纵向数据集、2个任务族（封闭式健康分数预测和由大语言模型作为评判者自动评分的推理生成）、3种智能体范式评估。

    arXiv:2608.27219v1 Announce Type: new  Abstract: Mental health assessment relies on episodic self-report scales, which convert subjective states such as stress into numerical scores but provide only sparse snapshots of wellbeing. Wearable devices offer longitudinal behavioral and physiological signals for continuous, low-burden monitoring. Recent LLM-driven personal-health agents enable natural language queries over wearable signals, but mainly handle short-term, retrieval-based lookups (e.g., highest step count over a week). They do not evaluate whether agents can reason over long-term signals to predict wellbeing scores paired with evidence-grounded rationales. To address this gap, we introduce BALMS, the first systematic benchmark of LLM-based agentic systems for longitudinal mental health sensing. BALMS spans 3 real-world longitudinal datasets, 2 task families (closed-form wellbeing-score prediction and rationale generation auto-graded by an LLM-as-Judge), 3 agentic paradigms evalu
    
[^27]: 当文本误导：音频对话中的不一致感知推理

    When Text Misleads: Inconsistent-Aware Reasoning for Audio-Grounded Dialogue

    [https://arxiv.org/abs/2608.27176](https://arxiv.org/abs/2608.27176)

    本文提出了ContraTalk基准测试，通过跨模态不一致检测框架，系统性地识别并利用文本误导与语音线索之间的冲突，以评估模型在音频对话中是否真正基于语音进行推理。

    

    理解口语对话需要对词汇内容和副语言声学信号（如情感和对话意图）进行联合推理。然而，现有的评估往往允许基于转录文本或单一模态解决方案的捷径，这掩盖了模型是否真正基于语音进行预测。我们将这种失败模式形式化为跨模态不一致，即转录文本暗示了看似合理但错误的表面解释，而韵律或说话风格等声学线索则支持不同的答案。我们开发了一个可扩展的框架，用于识别文本偏见的表面解释，并将不一致区域转换为冲突问答示例。我们还包含了一致的情况，即基于转录和基于语音的解释一致，从而能够在对抗性音频依赖之外进行评估。这产生了ContraTalk，一个受控基准测试，包含五个话语维度中的501个问题：情感、意图、讽刺、强调和不确定性。

    arXiv:2608.27176v1 Announce Type: new  Abstract: Understanding spoken dialogue requires joint reasoning over lexical content and paralinguistic acoustic signals such as emotion and conversational intent. However, existing evaluations often allow shortcuts based on transcripts or single-modality solutions, obscuring whether models genuinely ground predictions in speech. We formalize this failure mode as cross-modal disagreement, where transcripts suggest plausible but incorrect surface interpretations while acoustic cues such as prosody or speaking style support different answers. We develop a scalable framework that identifies text-biased surface interpretations and converts disagreement regions into conflict QA examples. We also include consistent cases where transcript-based and speech-grounded interpretations agree, enabling evaluation beyond adversarial audio dependence. This results in ContraTalk, a controlled benchmark containing 501 questions across five discourse dimensions: in
    
[^28]: 校准到足以知晓，却未校准到行动：伪造证据使LLM代理对不可知之事做出承诺

    Calibrated Enough to Know, Not Calibrated to Act: Fabricated Evidence Makes LLM Agents Commit to the Unknowable

    [https://arxiv.org/abs/2608.27167](https://arxiv.org/abs/2608.27167)

    本文发现LLM代理在面临伪造的专业面板证据时，会显著提高对不可预测问题的承诺率，其行为受证据包装的权威性驱动而非信息真实性或模型信念，揭示了一种可定位的校准失败。

    

    arXiv:2608.27167v1 公告类型：新 摘要：一个LLM代理在看到一个看起来专业的市场面板时，对一个问题做出方向性判断的频率远高于仅被问及该问题本身时的频率——在12个前沿模型中，随着证据的升级，承诺率从6.5%上升到54.0%。即使面板上的每个数字都是编造的，它同样会轻易承诺：完全伪造整个显示内容，使模型能看到的除了问题本身外无一是真，仍将承诺率从24.5%提升至36.8%，这在统计上与真实市场数据产生的37.6%无显著差异。解锁自信行动的不是信息，而是其包装的权威性。这种失败是狭窄且可定位的。能力不足并非答案：在附于相同面板的可回答问题上，相同模型几乎总是回答，且准确率近乎完美。这也不是信念问题——声明的概率在驱动行动变化达48个百分点的梯度上几乎不动，且得分糟糕。

    arXiv:2608.27167v1 Announce Type: new  Abstract: An LLM agent shown a professional-looking market panel commits to a directional call on a provably unpredictable question far more often than one asked the bare question: across 12 frontier models, commitment rises from 6.5% to 54.0% as evidence is escalated. It commits just as readily when every number on the panel is invented: fabricating the entire display, so nothing the model can see is true except the question itself, still lifts commitment from 24.5% to 36.8%, statistically indistinguishable from the 37.6% produced by genuine market data. What unlocks confident action is not information but the authority of its packaging. The failure is narrow and locatable. Incapacity is not the answer: on matched answerable questions attached to the same panels, the same models answer essentially always, at near-perfect accuracy. Nor is it belief - stated probabilities barely move across the gradient that swings action by 48 points, and score wo
    
[^29]: 预测的预测（PoP）：面向大型语言模型单次前向传播幻觉检测的层间激活融合

    Prediction of Prediction (PoP): Inter-Layer Activation Fusion for Single-Pass Hallucination Detection in Large Language Models

    [https://arxiv.org/abs/2608.27165](https://arxiv.org/abs/2608.27165)

    该论文提出PoP机制，通过单次前向传播中融合层间隐藏状态来检测幻觉，无需额外解码，在TruthfulQA上实现75.5%的AUROC。

    

    自回归大型语言模型（LLMs）通常会以高解码置信度生成事实不正确的输出，这限制了它们在高风险工作流程中的部署。现有的输出阶段不确定性指标在模型对错误断言过度自信时可能失效，而多样本验证流程则引入了显著的内存和延迟开销。本研究评估了生成过程中内部隐藏状态转换动态是否能无需辅助解码调用即可指示事实错误。我们引入了预测的预测（PoP）机制，该机制通过在单次前向传播中融合跨深度的中间隐藏表示来捕获层间转换不确定性。在TruthfulQA基准上使用自回归Transformer骨干进行评估，PoP在事实正确性分类中达到了75.5%的接收者操作特征曲线下面积（AUROC）。该机制在b（此处截断，但根据上下文应继续）的范围内运行。

    arXiv:2608.27165v1 Announce Type: new  Abstract: Autoregressive large language models (LLMs) routinely generate factually incorrect outputs with high decoding confidence, limiting their deployment in high-stakes workflows. Existing output-stage uncertainty metrics can fail when models are overconfident on false assertions, while multi-sample verification pipelines introduce substantial memory and latency overhead. This work evaluates whether internal hidden-state transition dynamics during generation can signal factual errors without auxiliary decoding calls. We introduce Prediction of Prediction (PoP), a mechanism that captures layer-transition uncertainty by fusing intermediate hidden representations across depth during a single forward pass. Evaluated on the TruthfulQA benchmark using autoregressive transformer backbones, PoP achieves an area under the receiver operating characteristic curve (AUROC) of 75.5% for factual-correctness classification. The mechanism operates within the b
    
[^30]: STAR：面向文档到文档机器翻译的句子翻译对齐率

    STAR : Sentence Translation Alignment Rate for Document-to-Document Machine Translation

    [https://arxiv.org/abs/2608.27161](https://arxiv.org/abs/2608.27161)

    本文提出STAR指标和StarPO框架，通过量化句子级结构对齐并动态优化错位片段，显著提升文档到文档机器翻译的结构保真度和整体质量。

    

    arXiv:2608.27161v1 公告类型：新 摘要：大型语言模型（LLMs）已推动机器翻译从句子级别向文档到文档（Doc2Doc）的转变，有望改善整体连贯性。然而，单次生成的文档到文档翻译常常遭受结构错位问题，表现为句子遗漏或幻觉，这违反了源语言与目标语言对应的核心要求。为解决这一问题，我们引入了句子翻译对齐率（STAR），这是一种辅助指标，用于显式量化句子级别的结构保真度。在此基础上，我们提出了STAR掩码偏好优化（StarPO）框架，该框架根据结构质量对文档级假设进行排序，并利用动态对齐掩码将优化重点放在错位片段上。跨新闻和文学领域的实验结果表明，StarPO显著提升了翻译质量和结构完整性。值得注意的是，StarPO允许紧凑的模型实现。

    arXiv:2608.27161v1 Announce Type: new  Abstract: Large Language Models (LLMs) have enabled a shift from sentence-level to document-to-document (Doc2Doc) machine translation, promising improved global coherence. However, document-to-document generation in a single pass frequently suffers from structural misalignment, manifesting as sentence omissions or hallucinations that violate the core requirement of source-target correspondence. To address this, we introduce Sentence Translation Alignment Rate (STAR), an auxiliary metric that explicitly quantifies sentence-level structural fidelity. Building on this, we propose STAR-masked Preference Optimization (StarPO), a framework that ranks document-level hypotheses by structural quality and utilizes a dynamic alignment mask to focus optimization on misaligned segments. Experimental results across news and literary domains demonstrate that StarPO significantly enhances translation quality and structural integrity. Notably, StarPO allows compac
    
[^31]: 大声说出，读法不同：多模态模型中的跨模态不稳定性

    Said Aloud, Read Different: Cross-Modal Instability in Multimodal Models

    [https://arxiv.org/abs/2608.27135](https://arxiv.org/abs/2608.27135)

    该论文提出了一种语音增强的视觉接地对比三元组基准，揭示多模态模型在文本与语音、英语与阿拉伯语之间的跨模态和跨语言转换中，存在显著的三元组级判断不一致性，且未被聚合指标充分反映。

    

    arXiv:2608.27135v1 公告类型：新 摘要：多模态基础模型越来越多地用于语音优先的助手，这些助手必须解释口语查询并产生基于视觉的决策。然而，目前尚不清楚语义等价的查询在模态（文本与语音）和语言（英语与阿拉伯语）之间是否产生一致的判断。我们引入了一个语音增强的视觉接地对比三元组基准，涵盖来自18个中东和北非国家的10,150张文化接地图像，每张图像配有一个支持性陈述和两个看似合理但不支持的替代陈述。我们将对比不稳定性定义为模型在三元组内未能解决所有陈述的条件率，从而将碎片化推理与完全失败区分开来。在英语和阿拉伯语的文本和语音条件下评估近期多模态模型，我们发现模态和语言转换引入了显著的三元组级不一致性，而这些不一致性并未被聚合指标完全捕捉。

    arXiv:2608.27135v1 Announce Type: new  Abstract: Multimodal foundation models are increasingly used in speech-first assistants that must interpret spoken queries and produce visually grounded decisions. Yet it remains unclear whether semantically equivalent queries yield consistent judgments across modality (text vs. speech) and language (English vs. Arabic). We introduce a speech-augmented visually grounded contrastive triplet benchmark spanning 10,150 culturally grounded images from 18 MENA countries, where each image is paired with one supported statement and two plausible but unsupported alternatives. We define contrastive instability as the conditional rate at which a model fails to resolve all statements within a triplet, isolating fragmented reasoning from complete failure. Evaluating recent multimodal models under text and speech in English and Arabic, we find that modality and language shifts introduce substantial triplet-level inconsistencies that are not fully captured by ag
    
[^32]: TwinKV：一种通过成对键冗余实现KV缓存驱逐的可组合修复方法

    TwinKV: A Composable Repair Pass for KV Cache Eviction via Pairwise Key Redundancy

    [https://arxiv.org/abs/2608.27128](https://arxiv.org/abs/2608.27128)

    本文提出TwinKV，一种无需训练的冗余信号修复方法，通过检测键的重复性来优化KV缓存驱逐，挑战了注意力评分前提，可组合地提升现有策略性能。

    

    长上下文推理受到键值（KV）缓存内存占用的瓶颈限制，尤其是在资源紧张的小型模型上。现有的KV缓存驱逐方法使用模型的注意力分布来对令牌进行评分，或者在无注意力变体中，使用每个键与全局参考点的距离。通过受控的留一法探测，我们发现注意力大小与令牌对答案的因果贡献无关（Spearman $\rho=-0.004$），这挑战了主流驱逐方法的前提。我们引入了TwinKV，一种无需训练、无需注意力的冗余信号，用于检测令牌的键在上下文中是否存在近重复项。TwinKV并非取代现有策略，而是作为一种可组合的修复方法：给定策略的固定保留集，它识别出没有幸存重复项的驱逐令牌（孤儿）以及信息在其他地方重复的保留令牌（冗余）。

    arXiv:2608.27128v1 Announce Type: new  Abstract: Long-context inference is bottlenecked by the memory footprint of the key-value (KV) cache, especially for small models under tight resource budgets. Existing KV cache eviction methods score tokens using the model's attention distribution or, in attention-free variants, each key's distance from a global reference point. Using a controlled leave-one-out probe, we find that attention magnitude is unrelated to a token's causal contribution to the answer (Spearman $\rho=-0.004$), challenging the premise behind dominant eviction methods. We introduce TwinKV, a training-free, attention-free redundancy signal that detects whether a token's key has a near-duplicate elsewhere in context. Rather than replacing existing policies, TwinKV acts as a composable repair pass: given a policy's fixed retained set, it identifies evicted tokens with no surviving duplicate (\emph{orphans}) and retained tokens whose information is duplicated elsewhere (\emph{r
    
[^33]: 无需联合训练的跨语言对齐：单语语言模型是否收敛于通用表示？

    Cross-Lingual Alignment Without Joint Training: Do Monolingual Language Models Converge on Universal Representations?

    [https://arxiv.org/abs/2608.27115](https://arxiv.org/abs/2608.27115)

    单语模型无需联合训练也能形成可对齐的表示，且通过单一旋转能实现功能转移，表明跨语言对齐可能源于通用表示收敛。

    

    arXiv:2608.27115v1 公告类型：新 摘要：多语言语言模型中的跨语言对齐通常归因于联合训练：共享参数、混合语言批次或显式对齐目标。我们研究仅使用非平行数据训练的单语模型是否能在没有联合训练的情况下学习可对齐的表示。通过严格测试单语语言模型，如Goldfish模型系列和来自不同研究实验室独立开发的模型，我们发现三个结果。相关性：这些模型在各层间发展出可对齐的表示几何结构，且对齐强度随数据规模、模型规模或语言接近度的增加而增强。构造：一个在平行句子上拟合的单一Procrustes旋转可将模型间的隐藏状态映射。因果性：同一旋转传递功能内容；将旋转后的英语残差注入德语模型的事实性完形任务中，在大多数情况下会将预测翻转为捐赠者的首都。我们得出结论。

    arXiv:2608.27115v1 Announce Type: new  Abstract: Cross-lingual alignment in multilingual language models is typically attributed to joint training: shared parameters, mixed-language batches, or explicit alignment objectives. We ask whether monolingual models trained on non-parallel data learn alignable representations without joint training. By testing on strictly monolingual language models, such as the Goldfish model families and independently developed models from different research labs, we find three results. Correlation: these models develop alignable representational geometry across layers, with alignment strengthening as data scale, model scale, or linguistic proximity increases. Construction: a single Procrustes rotation fit on parallel sentences maps hidden states between models. Causation: the same rotation transfers functional content; patching a rotated English residual into a German model on a factual cloze flips the prediction to the donor's capital in most cases. We con
    
[^34]: DocTalkBN：孟加拉语专家远程医疗对话的新型数据集

    DocTalkBN: A Novel Dataset of Expert Telemedicine Conversations in Bengali

    [https://arxiv.org/abs/2608.27110](https://arxiv.org/abs/2608.27110)

    本文提出了DocTalkBN，一个大规模、多模态的孟加拉语真实专家远程医疗对话数据集，包含557.63小时音频文本和三个下游任务，填补了低资源语言医疗对话AI的数据空白。

    

    arXiv:2608.27110v1 公告类型：新 摘要：可靠的医疗对话式人工智能需要真实的专家与患者互动数据，然而此类数据集仍然稀缺，尤其是对于孟加拉语等低资源语言。我们提出了DocTalkBN，一个从全国播出的、由委员会认证医生参与的远程医疗节目中收集的孟加拉语真实专家远程医疗对话的大规模多模态数据集。DocTalkBN包含557.63小时的配对音频和文本、1,515次多轮患者通话、10,274次主持人-医生问答交流，总计170万个词元，覆盖26个医学专科。与先前基于医学论坛、书面健康内容或合成数据的资源不同，我们的数据集保留了低资源环境下真实医疗互动的自发性、上下文丰富性和口语特征。为支持基准驱动的研究，我们进一步从语料库中构建了三个下游任务，包括医疗分诊分类。

    arXiv:2608.27110v1 Announce Type: new  Abstract: Reliable medical conversational AI requires authentic expert--patient interaction data, yet such datasets remain scarce, especially for low-resource languages such as Bengali. We present DocTalkBN, a large-scale multimodal dataset of real-world expert telemedicine conversations in Bengali, collected from nationally broadcast telemedicine programs featuring board-certified physicians. DocTalkBN contains 557.63 hours of paired audio and text, 1,515 multi-turn patient calls, 10,274 host--doctor question--answer exchanges, totaling 1.7M tokens, spanning 26 medical specialties. Unlike prior resources derived from medical forums, written health content, or synthetic data, our dataset preserves the spontaneity, contextual richness, and spoken characteristics of authentic medical interactions in a low-resource setting. To support benchmark-driven research, we further construct three downstream tasks from the corpus, medical triage classification
    
[^35]: 统一任务检测与自适应于无任务边界的持续学习

    Unifying Detection and Adaptation in Task-Free Continual Learning

    [https://arxiv.org/abs/2608.27070](https://arxiv.org/abs/2608.27070)

    本文提出FiUni框架，利用Fisher信息矩阵的K-FAC近似主子空间正交性，实现无任务边界下的批量级任务检测和参数高效持续适应，有效缓解灾难性遗忘。

    

    为了缓解大型语言模型（LLMs）在下游持续学习（CL）中的灾难性遗忘问题，现有方法通常限制参数更新或引入特定任务的适应模块。然而，这些方法在训练过程中往往依赖显式的任务边界，限制了它们在现实无任务场景中的适用性。在本文中，我们提出了一种名为**Fi**sher引导的**uni**fied（**FiUni**）框架，用于批量级任务检测和参数高效的持续适应。FiUni的动机来自对预训练模型Fisher信息矩阵（FIM）的一个关键观察：其Kronecker-Factored近似曲率（K-FAC）近似的主子空间之间的正交性，通过少量下游任务样本估计，可以反映不同任务之间的相似性。基于这一观察，FiUni构建了由FIM派生的冻结子空间来引导低秩适应。

    arXiv:2608.27070v1 Announce Type: cross  Abstract: To mitigate catastrophic forgetting in downstream continual learning (CL) for large language models (LLMs), existing methods typically constrain parameter updates or introduce task-specific adaptation modules. However, these methods often rely on explicit task boundaries during training, limiting their applicability to realistic task-free scenarios. In this paper, we propose a \textbf{Fi}sher-guided \textbf{uni}fied (\textbf{FiUni}) framework for batch-level task detection and parameter-efficient continual adaptation. FiUni is motivated by a key observation about the Fisher information matrix (FIM) of pre-trained models: the orthogonality among the principal subspaces of its Kronecker-Factored Approximate Curvature (K-FAC) approximation, estimated from a small number of downstream task samples, can reflect the similarity between different tasks. Based on this observation, FiUni constructs FIM-derived frozen subspaces to guide low-rank 
    
[^36]: 社会科学研究设计追踪与评估

    Research Design Tracking and Assessment for the Social Sciences

    [https://arxiv.org/abs/2608.27049](https://arxiv.org/abs/2608.27049)

    本文提出了自动化研究设计追踪与评估（ARDTrA）任务，并发现段落长度是评估性能的主要因素，且人类与机器在任务难度上存在不一致性。

    

    摘要：在社会科学中，对因果研究设计进行可靠评估对于基于证据的政策制定至关重要，但迄今为止完全依赖于人工专家分析。我们引入了自动化研究设计追踪与评估（ARDTrA）任务，该任务涉及检测论文中使用的研究设计并评估其应用质量。我们创建了一个专家标注的数据集，涵盖六类反事实研究设计，并使用基于多轮检索增强生成（RAG）的对话流程评估该任务。在四种检索策略、四种大语言模型和六种嵌入模型中，我们发现段落长度是性能的主要驱动因素，解释了52%至66%的方差。按研究设计进行的分析还表明，人类与机器的难度并不一致：系统认为最难的设计并非专家标注者分歧最大的设计，这指出了任务难度的两个独立来源。

    arXiv:2608.27049v1 Announce Type: new  Abstract: Reliable assessment of causal research designs in the social sciences is critical for evidence-based policy-making, yet has so far relied entirely on manual expert analysis. We introduce Automated Research Design Tracking and Assessment (ARDTrA), a task that involves detecting the research design used in a paper and assessing the quality of its application. We create an expert-annotated dataset of papers covering six families of counterfactual research designs and evaluate the task using a multi-turn RAG-based conversational pipeline. Across four retrieval strategies, four LLMs and six embedding models, we find that passage length is the main driver of performance, explaining 52-66% of the variance. A per-research-design analysis also shows that human and machine difficulty do not align: the designs that prove hardest for the system are not those on which expert annotators disagree most, pointing to two independent sources of task diffic
    
[^37]: 级联批量提示

    Cascaded Batch Prompting

    [https://arxiv.org/abs/2608.27038](https://arxiv.org/abs/2608.27038)

    提出级联批量提示，通过两阶段分离复杂推理与符号接地，解决了批量提示的性能不可预测性，并在多项选择问答和自然语言推理任务中实现速度提升和性能突破。

    

    arXiv:2608.27038v1 公告类型：新 摘要：尽管批量提示通过同时处理多个实例使大型语言模型推理更高效，但它面临下游任务性能不可预测的问题。我们提出级联批量提示，一种两阶段方法，旨在通过将复杂推理与符号接地分离来解决传统批量提示的不可预测性。在多项选择问答和自然语言推理上的实验表明，所提方法优于标准单一提示基线，同时实现与批量大小成比例的速度提升，在帕累托前沿上确立了新的最先进水平。

    arXiv:2608.27038v1 Announce Type: new  Abstract: Although batch prompting makes large language model inference more efficient by processing multiple instances simultaneously, it suffers from unpredictable downstream task performance. We propose cascaded batch prompting, a two-stage approach designed to resolve the unpredictability of conventional batch prompting by disentangling complex reasoning from symbol grounding. Experiments on multiple-choice question answering and natural language inference demonstrate that the proposed method outperforms the standard single prompting baseline while achieving a speedup proportional to batch size, establishing a new state of the art on the Pareto frontier.
    
[^38]: 关于机器翻译中上下文样本推理的研究

    Reasoning about In-Context Samples for Machine-Translation

    [https://arxiv.org/abs/2608.27036](https://arxiv.org/abs/2608.27036)

    本文提出了一种基于片段的推理框架，通过从相似示例中提取平行片段作为中间推理步骤，显著提升了LLM在机器翻译中的表现。

    

    arXiv:2608.27036v1 公告类型：新 摘要：大型语言模型（LLMs）可以通过训练进行思维链推理，以提高其响应的可靠性。在这项工作中，我们研究了如何利用显式推理来增强基于LLM的机器翻译（MT）中的上下文样本。我们引入了一种新颖的基于片段的推理框架，其中模型首先从检索到的相似示例中提取平行的源-目标片段，并将这些片段作为中间推理轨迹来生成最终翻译。为了训练我们的模型，我们从大型教师模型中提炼出银级片段和草稿。我们在Qwen3模型系列上进行的实验，覆盖6种语言，每种语言最多5个领域，表明基于片段的机器翻译显著优于标准k-shot或基础草稿等替代方法。

    arXiv:2608.27036v1 Announce Type: new  Abstract: Large Language Models (LLMs) can be trained to perform chain-of-thoughts reasoning in order to improve the reliability of their responses. In this work, we investigate how explicit reasoning can be leveraged for LLM-Based Machine Translation (MT) with in-context samples. We introduce a novel fragment-based reasoning framework in which the model first extracts parallel source-target fragments from retrieved similar exemplars, and uses these fragments as intermediate reasoning traces to produce the final translation. To train our model, we distill silver fragments and drafts from a large teacher model. Our experiments with the Qwen3 model family, over 6 languages, including up to 5 domains per language, demonstrate that fragment-based MT significantly outperforms alternative methods like standard k-shot or basic drafting.
    
[^39]: 不同粒度下韩语成分结构的表示与解析

    Representing and Parsing Korean Constituency Structure at Different Levels of Granularity

    [https://arxiv.org/abs/2608.27035](https://arxiv.org/abs/2608.27035)

    本文通过比较三种基于Penn韩语树库的不同粒度表示，系统评估了韩语成分解析在形态复杂eojeol单元下的表示策略与解析性能。

    

    韩语成分解析面临一个表示挑战，因为短语结构树的终端单元并不直接对应简单的表面词。韩语eojeol（语节）是形态复杂的间距单位，现有成分资源在表示eojeol内部形态和非显性元素方面有所不同。本文比较了从Penn韩语树库中派生的三种成分解析表示：Morpheme+XPOS、Eojeol+XPOS和Eojeol+UPOS。我们通过移除空元素、将Penn韩语短语结构与显性eojeol标记对齐、尽可能保留Penn韩语短语标签，并变化终端和前终端层来构建这些表示。然后，我们在共享建模和评估设置下，以自顶向下、中序和自底向上顺序评估了规范的基于非二元转换的成分解析器。所有实验使用黄金终端分割和黄金前终端分割。

    arXiv:2608.27035v1 Announce Type: new  Abstract: Korean constituency parsing raises a representational challenge because the terminal units of a phrase-structure tree do not straightforwardly correspond to simple surface words. Korean eojeols are morphologically complex spacing units, and existing constituency resources differ in how they represent eojeol-internal morphology and non-overt elements. This paper compares three constituency parsing representations derived from the Penn Korean Treebank: Morpheme+XPOS, Eojeol+XPOS, and Eojeol+UPOS. We construct these representations by removing null elements, aligning Penn Korean phrase structure with overt eojeol tokens, preserving Penn Korean phrase labels where possible, and varying the terminal and preterminal layers. We then evaluate canonical non-binary transition-based constituency parsers in top-down, in-order, and bottom-up orders under a shared modeling and evaluation setup. All experiments use gold terminal segmentation and gold p
    
[^40]: ITL：基于结构化参考框架的可解释文档对齐方法

    ITL: Interpretable Document Alignment with Structured Reference Frameworks

    [https://arxiv.org/abs/2608.27031](https://arxiv.org/abs/2608.27031)

    本文提出了一种名为ITL的领域无关、语言可移植的可解释文档对齐方法，通过从结构化参考框架中构建概念级术语轮廓并分配重要性权重，实现了对文档与概念之间对齐程度的定量、可追踪度量。

    

    测量文档与结构化参考框架之间的对齐程度，需要识别分布在文本中的概念证据，并通过定量、可解释和可追踪的度量来报告。许多常用的检索和分类方法要么返回成对相似度分数，要么返回一个或多个类别标签，而较少有方法提供概念级分数，这些分数能直接追溯到支持它们的术语证据。我们提出了“智能目标定位器”（ITL），这是一种领域无关且语言可移植的方法，用于估计目标文档的文本单元与结构化参考文档（SRD）中定义的概念之间的亲和度。ITL从SRD中诱导出基于独立术语、二元组、三元组和共现关系的概念特定术语轮廓。每个术语被赋予一个重要性权重，该权重结合了概念记忆。

    arXiv:2608.27031v1 Announce Type: new  Abstract: Measuring alignment between documents and structured reference frameworks requires identifying conceptual evidence distributed throughout the text and reporting it through measures that are quantitative, interpretable, and traceable. Many commonly used retrieval and classification approaches return either pairwise similarity scores or one or more class labels, whereas fewer methods provide concept-level scores that are directly traceable to the terminological evidence supporting them. We present \emph{Intelligent Target Locator} (ITL), a domain-agnostic and language-portable methodology that estimates the affinity between the textual units of a target document and the concepts defined in a \emph{Structured Reference Document} ($SRD$). From the $SRD$, ITL induces concept-specific terminological profiles built from independent terms, bigrams, trigrams, and co-occurrences. Each term is assigned an importance weight that combines concept mem
    
[^41]: JudgeStealer：跨评估协议提取大语言模型评判能力

    JudgeStealer: Extracting LLM Judging Capabilities across Evaluation Protocols

    [https://arxiv.org/abs/2608.26982](https://arxiv.org/abs/2608.26982)

    本文提出了首个查询高效的模型提取框架JUDGESTEALER，通过跨协议一致性转换和动态输入选择，在有限查询预算下复制LLM评判器的多种评估能力。

    

    大型语言模型（LLM）评判器越来越多地用于各种评估场景，使其评判能力成为有价值的知识产权。然而，黑盒访问将这些能力暴露于模型提取攻击之下。现有的提取方法并未专门针对LLM评判器，且在受限查询预算下对多种评估协议的支持有限。在本研究中，我们提出JUDGESTEALER，这是首个用于复制点式评分、成对比较和列表排序协议下评判能力的查询高效模型提取框架。JUDGESTEALER利用跨协议的强一致性，获取点式分数并将其转换为成对和列表式监督信号，而无需额外的受害者查询。为了捕获信息丰富的评判模式并提高查询效率，JUDGESTEALER基于语义多样性、预测不确定性动态选择点式输入。

    arXiv:2608.26982v1 Announce Type: new  Abstract: Large language model (LLM) judges are increasingly used across various evaluation scenarios, making their judgment capabilities valuable intellectual property. However, black-box access exposes these capabilities to model extraction attacks. Existing extraction methods do not specifically target LLM judges and provide limited support for multiple evaluation protocols under restricted query budgets. In this study, we propose JUDGESTEALER, the first query-efficient model extraction framework for replicating judging capabilities across pointwise scoring, pairwise comparison, and listwise ranking protocols. JUDGESTEALER exploits the strong cross-protocol agreement to acquire pointwise scores and transform them into pairwise and listwise supervisions without additional victim queries. To capture informative judge patterns and improve query efficiency, JUDGESTEALER dynamically selects pointwise inputs based on semantic diversity, predictive un
    
[^42]: 威尔士聚落名称中的地形特征

    Terrain signatures in Welsh settlement names

    [https://arxiv.org/abs/2608.26978](https://arxiv.org/abs/2608.26978)

    本研究首次通过严格验证的词汇框架和地理结构化模型，证明威尔士聚落名称中的地形词汇（如“bryn”与“cwm”）能可靠地反映局部海拔差异，表明地名保留了可量化的环境信息。

    

    景观被命名，但名称是否保留超越广泛地理结构的可测量环境信息，很少被测试。我们使用一个冻结的、来源审计的24元素词汇框架、预注册的结果特定模型和地理结构化验证，分析了3,757个威尔士聚落。核心比较对比了101个带有高地地形元素（如“bryn”或“mynydd”）的聚落与139个带有低地地形元素（如“cwm”或“pant”）的聚落。高地地形名称的聚落相对于其2公里周围环境高出24.4米（95%置信区间，10.8--38.1米；霍尔姆校正p = 0.00137）。该关联在预指定的1、2和5公里邻域定义中保持正向，并使用独立生成的高程源进行了复制（24.1米；95%置信区间，10.6--37.6米）。将地形名称极性添加到非词汇空间和聚落基线中，减少了地理保持验证的误差。

    arXiv:2608.26978v1 Announce Type: cross  Abstract: Landscapes are named, but whether names retain measurable environmental information beyond broad geographic structure is rarely tested. We analysed 3,757 Welsh settlements using a frozen, source-audited 24-element lexical framework, preregistered outcome-specific models and geographically structured validation. The central comparison contrasted 101 settlements carrying high-terrain elements (\textit{bryn} or \textit{mynydd}) with 139 carrying low-terrain elements (\textit{cwm} or \textit{pant}). High-terrain names occupied locations 24.4 m higher relative to their 2-km surroundings (95\% CI, 10.8--38.1 m; Holm-adjusted $p$ = 0.00137). The association remained positive across prespecified 1-, 2- and 5-km neighbourhood definitions and was reproduced using an independently produced elevation source (24.1 m; 95\% CI, 10.6--37.6 m). Adding terrain-name polarity to a non-lexical spatial and settlement baseline reduced geographically held-out
    
[^43]: 在有限数据下利用递归Transformer挖掘更多潜力

    Squeezing More from Limited Data with Recursive Transformers

    [https://arxiv.org/abs/2608.26973](https://arxiv.org/abs/2608.26973)

    该论文提出递归Transformer和分解嵌入，在有限数据预算下通过复用共享块扩展计算并减少嵌入参数，显著提升模型泛化性能。

    

    在有限数据下进行预训练，需要与网络规模语言建模不同的扩展视角。在固定数据预算但相对充足的计算资源下，增加参数数量仅有助于达到最优规模；超过该点后，模型会过拟合，泛化能力变差。我们研究了1000万至1亿词预训练预算、两个语料库和多种下游评估中的这一行为，发现最优规模强烈依赖于数据预算和下游目标。我们认为标准Transformer在此设置下扩展效果不佳，因为嵌入层消耗了大量参数预算，且每个token的计算与表示能力紧密耦合。为解决这一耦合问题，我们研究了递归Transformer，通过深度复用共享块来扩展计算，并结合分解嵌入以减少词汇映射参数。我们训练了三个递归模型，发现它们优于标准模型。

    arXiv:2608.26973v1 Announce Type: new  Abstract: Pre-training under limited data requires a different view of scaling than web-scale language modeling. With a fixed data budget but relatively abundant compute, increasing parameter count helps only up to an optimal scale; beyond that point, models overfit and generalization worsens. We study this behavior across 10M-100M word pre-training budgets, two corpora, and multiple downstream evaluations, and find that optimal size depends strongly on both the data budget and the downstream target. We argue that standard Transformers scale down poorly to this setting, because embeddings consume a large fraction of the parameter budget and per-token computation is tied to representational capacity. To address this coupling, we study recursive Transformers, reusing a shared block across depth to scale compute, together with factorized embeddings to reduce vocabulary-map parameters. We train three recursive models and find that they outperform stan
    
[^44]: TEMPLAR Wales：威尔士定居点的地理参考环境与地名数据集

    TEMPLAR Wales: A georeferenced environmental and toponymic dataset of Welsh settlements

    [https://arxiv.org/abs/2608.26970](https://arxiv.org/abs/2608.26970)

    该论文构建了威尔士全境3,757个定居点的地理参考环境-地名数据集，通过稳定标识符整合可复现的定居点框架、确定性词汇筛选和环境属性，实现了地名与环境数据的标准化定量分析。

    

    摘要：地名提供了景观如何被描述和组织的持久记录，但其定量重用需要在已映射地点、词汇注释和环境测量之间进行明确分离。TEMPLAR Wales是一个地理参考的环境-地名数据集，包含威尔士全境的3,757条定居点记录。该资源通过稳定标识符，将可复现的定居点框架与确定性词汇筛选及定居点级环境属性联系起来。它包含来自24个威尔士地名元素冻结注册表的1,350个词汇检测，覆盖1,294个定居点，同时分别保留精确和前缀标记匹配及其来源。环境属性描述了河流和海岸邻近度、海拔和多种空间尺度下的局部地形背景、土地覆盖和邻域木本覆盖，并提供了基于独立高程数据的并行地形测量。

    arXiv:2608.26970v1 Announce Type: cross  Abstract: Place names provide persistent records of how landscapes have been described and organised, but their quantitative reuse requires explicit separation between mapped places, lexical annotations and environmental measurements. TEMPLAR Wales is a georeferenced environmental-toponymy dataset comprising 3,757 settlement records across Wales. The resource links a reproducible settlement frame to deterministic lexical screening and settlement-level environmental attributes through stable identifiers. It contains 1,350 lexical detections across 1,294 settlements, generated from a frozen registry of 24 Welsh place-name elements, while retaining exact- and prefix-token matches and their provenance separately. Environmental attributes describe river and coastal proximity, elevation and local terrain context at multiple spatial scales, land cover and neighbourhood woody cover, with parallel terrain measurements derived from independent elevation p
    
[^45]: 扩展模型生成的蒸馏数据可使潜在教师特征更易被恢复

    Scaling Model-Generated Distillation Data Can Make Latent Teacher Traits More Recoverable

    [https://arxiv.org/abs/2608.26958](https://arxiv.org/abs/2608.26958)

    本文发现，扩展模型生成的蒸馏数据量不仅提升学生性能，还能增强教师潜在特征在学生行为中的可恢复性，即使数据与任务无关且未明确提及该特征。

    

    扩展模型生成的数据通常被视为改进蒸馏：更多示例应增加覆盖范围、减少噪声并产生更强的学生模型。我们展示了第二种效应：更大的数据集可以使训练后的学生模型中教师特有的微妙信号更易被检测，即使示例与任务无关且从未提及该特征。在受潜意识学习启发的受控设置中，被诱导表达目标特征的教师生成受限的离任务数据，如仅数字的补全。训练在不同数量的独立离任务数据上的学生模型在单独领域中进行评估，并匹配无特征对照组以隔离目标特定迁移。我们的主要发现是，更大的独立数据集使教师的诱导特征在学生后续行为中更清晰地突显。其他可能特征也可能随规模增强，但目标通常增长更多。当小规模学生模型已经...（原文未完整）

    arXiv:2608.26958v1 Announce Type: cross  Abstract: Scaling model-generated data is usually viewed as improving distillation: more examples should increase coverage, reduce noise, and produce stronger students. We show a second effect: larger datasets can make subtle teacher-specific signals easier to detect in the trained student, even when examples are off-task and never mention the trait. In a controlled setup inspired by subliminal learning, a teacher induced to express a target trait generates restricted off-task data, such as number-only completions. Students trained on different amounts of independent off-task data are evaluated in a separate domain, with matched no-trait controls isolating target-specific transfer. Our main finding is that larger independent datasets make the teacher's induced trait stand out more clearly in the student's later behavior. Other plausible traits may also strengthen with scale, but the target usually grows more. When the small-scale student already
    
[^46]: 从原子到智能体：迈向可解释的LLM智能体数学能力评估

    From Atomic to Agentic: Towards Interpretable Evaluation of LLMs' Agentic Mathematical Capabilities

    [https://arxiv.org/abs/2608.26950](https://arxiv.org/abs/2608.26950)

    本文提出一个过程级数学基准，通过将智能体行为与原子能力分类对齐，并利用自动化流水线生成细粒度注释，实现可解释地评估LLMs的智能体数学推理能力。

    

    大型语言模型（LLMs）正从执行端到端的数学推理演变为集成智能体智能。然而，大多数现有的数学基准仅评估最终答案。这种面向结果的评估在识别过程级错误或严格逻辑方面提供的诊断价值有限，无法引导LLMs转变为稳健的智能体。为弥补这一差距，我们提出一个过程级基准，旨在评估LLMs固有的智能体数学推理能力。我们的框架将问题解决智能体行为与可复用的数学原子能力的结构化分类对齐。我们设计了一套全面的规划、行动和反馈任务，涵盖文本和多模态场景，并辅以自动化流水线，通过受控的LLM重写合成高质量轨迹并生成细粒度注释。实验表明，具备智能体能力的模型在此基准上表现更优。

    arXiv:2608.26950v1 Announce Type: new  Abstract: Large Language Models (LLMs) are evolving from performing end-to-end mathematical reasoning to integrating agentic intelligence. However, most existing math benchmarks evaluate only final answers. This outcome-oriented evaluation provides limited diagnostic value for identifying process-level failures or rigorous logic, failing to guide the transformation of LLMs into robust agents. To bridge this gap, we present a process-level benchmark designed to evaluate the inherent agentic mathematical reasoning abilities of LLMs. Our framework aligns problem-solving agentic behaviors with a structured taxonomy of reusable mathematical atomic capabilities. We design a comprehensive suite of planning, action, and feedback tasks across both textual and multimodal contexts, supported by an automated pipeline that synthesizes high-quality trajectories and produces fine-grained annotations via controlled LLM rewriting. Experiments reveal that models wi
    
[^47]: KinyaEmbed：通过多阶段课程训练实现基尼亚卢旺达语的对比句嵌入

    KinyaEmbed: Contrastive Sentence Embeddings for Kinyarwanda via Multi-Stage Curriculum Training

    [https://arxiv.org/abs/2608.26941](https://arxiv.org/abs/2608.26941)

    本文提出了首个专门针对基尼亚卢旺达语的句子嵌入模型KinyaEmbed，通过四阶段课程训练显著提升了该语言在语义相似度任务上的性能，并引入了无污染评估数据集Wiki-RW-STS。

    

    arXiv:2608.26941v1 公告类型：新 摘要：我们提出了KinyaEmbed，这是首个专门针对基尼亚卢旺达语的句子嵌入模型，基尼亚卢旺达语是一种形态丰富的班图语言，在卢旺达有超过1200万人使用。现有的多语言嵌入模型如LaBSE、mE5-large和OpenAI text-embedding-3-large，由于在预训练语料中严重代表性不足，在基尼亚卢旺达语上表现不佳。KinyaEmbed基于KinyaBERT-large构建，并通过使用MultipleNegativesRankingLoss（MNRL）的四阶段课程进行训练：第一阶段利用来自卢旺达官方公报的约18,000对释义对，采用三种温度尺度；第二阶段在715个NLLB翻译的MNLI三元组上进行微调，以学习蕴含结构；第三阶段使用英语-基尼亚卢旺达语OPUS-100翻译对对齐表示；第四阶段使用从KinyaCOMET中过滤的2,936个高质量配对（质量阈值0.8）进行优化。我们在SemRel2024-rw上进行了评估，并引入了Wiki-RW-STS，这是一个新的无污染基尼亚卢旺达语数据集。

    arXiv:2608.26941v1 Announce Type: new  Abstract: We present KinyaEmbed, the first dedicated sentence embedding model for Kinyarwanda, a morphologically rich Bantu language spoken by over 12 million people in Rwanda. Existing multilingual embedding models such as LaBSE, mE5-large, and OpenAI text-embedding-3-large perform poorly on Kinyarwanda due to severe under-representation in their pre-training corpora. KinyaEmbed is built on KinyaBERT-large and trained via a four-stage curriculum using MultipleNegativesRankingLoss (MNRL): Stage 1 leverages ~18,000 paraphrase pairs from the Official Gazette of Rwanda with three temperature scales; Stage 2 fine-tunes on 715 NLLB-translated MNLI triplets for entailment structure; Stage 3 aligns representations using English-Kinyarwanda OPUS-100 translation pairs; Stage 4 refines with 2,936 high-quality pairs filtered from KinyaCOMET at quality threshold 0.8. We evaluate on SemRel2024-rw and introduce Wiki-RW-STS, a new contamination-free Kinyarwanda 
    
[^48]: 仅通过视觉接地将书面文字映射到另一种语言的口语表达

    Mapping Written Words to Spoken Words in a Different Language Using Only Visual Grounding

    [https://arxiv.org/abs/2608.26925](https://arxiv.org/abs/2608.26925)

    本文提出一种基于自监督语音表征和对齐的简单方法，利用视觉接地数据将书面英文关键词映射到印地语口语表达，无需端到端神经模型，在关键词检测和定位上表现良好。

    

    arXiv:2608.26925v1 公告类型：新 摘要：在许多资源匮乏的环境中，即使仅仅收集语音数据也非常困难。一种有前景的方法是要求说话者描述图像。但如何从这种视觉接地的语音数据中构建模型呢？给定一个包含印地语口语字幕的图像数据集，我们考虑如何将书面英文关键词映射到该词在印地语中的口语实现形式。先前的工作训练了端到端的多模态神经模型。相反，我们探索了一种基于自监督语音表征的更简单的对齐方法。使用现成的图像字幕系统从图像中自动获取书面英文标签。然后，将与同一关键词关联的印地语话语（使用自监督特征）进行对齐，并聚合对齐证据以识别对应目标词的重复出现的语音片段。评估关键词检测和定位的实验表明，我们的基于对齐的方法在性能上具有竞争力。

    arXiv:2608.26925v1 Announce Type: new  Abstract: In many low-resource settings, even just eliciting speech for data collection is difficult. One promising approach has been to ask speakers to describe images. But how do we build models from such visually grounded speech data? Given a dataset of images with Hindi spoken captions, we consider how we can map a written English keyword to spoken realisations of that word in Hindi. Previous work trained end-to-end multimodal neural models. Instead, we explore a simpler alignment-based approach built on self-supervised speech representations. Written English tags are automatically obtained from images using off-the-shelf image captioning systems. Hindi utterances associated with the same keyword are then aligned (using self-supervised features), and alignment evidence is aggregated to identify recurring speech segments corresponding to the target word. Experiments evaluating keyword spotting and localization show that our alignment-based appr
    
[^49]: TabuLM：面向低资源语言的形态学感知表格预训练

    TabuLM: Morphology-Aware Tabular Pre-training for Low-Resource Languages

    [https://arxiv.org/abs/2608.26923](https://arxiv.org/abs/2608.26923)

    本文提出了TabuLM，首个面向基尼亚卢旺达语的形态学感知表格预训练模型，通过新目标MCR和CTP及结构注意力偏置，在低资源语言上实现了有效的表格表示学习，并构建了首个原生问答基准TabQA-kin。

    

    我们提出了TabuLM，这是首个在基尼亚卢旺达语表格数据上预训练的语言模型。基尼亚卢旺达语是一种形态丰富的班图语言，由卢旺达超过1200万人使用，但此前缺乏专门的表格表示学习资源。TabuLM扩展了KinyaBERT-large（一种双层形态变换器），通过添加行、列和单元格类型的加性嵌入，以及一种学习到的表结构注意力偏置，该偏置增强了同行和同列的注意力。预训练采用了两个新目标：掩码单元格恢复（MCR），该目标掩蔽整个单元格并强制从行和列上下文中重建；以及列类型预测（CTP），该目标根据观察到的单元格值预测列的语义类型。我们在来自NISR、RAB、REB和MoH开放数据门户的172个卢旺达政府表格（约35,000个单元格）上进行了预训练，并引入了TabQA-kin，这是首个原生基尼亚卢旺达语表格问答基准，包含31个表格中的526个问答对。

    arXiv:2608.26923v1 Announce Type: new  Abstract: We present TabuLM, the first language model pre-trained on Kinyarwanda tabular data. Kinyarwanda is a morphologically rich Bantu language spoken by over 12 million people in Rwanda, yet lacks any dedicated tabular representation learning resource. TabuLM extends KinyaBERT-large, a two-tier morphological transformer, with additive row, column, and cell-type embeddings and a learned table-structure attention bias that sharpens same-row and same-column attention. Pre-training uses two new objectives: Masked Cell Recovery (MCR), which masks entire cells and forces reconstruction from row and column context, and Column Type Prediction (CTP), which predicts column semantic types from observed cell values. We pre-train on 172 Rwandan government tables (~35,000 cells) from NISR, RAB, REB, and MoH open-data portals, and introduce TabQA-kin, the first native Kinyarwanda table question-answering benchmark comprising 526 QA pairs across 31 tables an
    
[^50]: AraMS-28k：最大公开的历史阿拉伯语手稿行级数据集，包含页边注释与插入锚点标注

    AraMS-28k: The Largest Publicly Released Line-Level Dataset of Historical Arabic Manuscripts with Margin and Insertion-Anchor Annotations

    [https://arxiv.org/abs/2608.26921](https://arxiv.org/abs/2608.26921)

    该论文首次发布了最大规模的历史阿拉伯语手稿行级数据集AraMS-28k，并引入了行级插入锚点标注，以恢复非线性阅读顺序，同时提供标音与去标音两种转写版本。

    

    arXiv:2608.26921v1 公告类型：交叉 摘要：我们推出了AraMS-28k，这是最大公开的真实历史阿拉伯语手稿行级数据集，包含14本书、3,043页和28,600条标注文本行（27,971条正文，629条页边）。其中13本书为手抄本，涵盖三种书写传统——纳斯赫体、鲁克阿体和马格里布体——另有一本石印印刷版以增加格式多样性。每行被标注为正文或页边，并且页边行若在正文中有明确的插入点，则进一步标注插入锚点，从而在行级粒度上恢复手稿真实的非线性阅读顺序——据我们所知，这是历史阿拉伯语手稿语料库中首次发布此类标注。由于参考转写完全标音，而手稿笔迹通常无变音符，我们同时发布了原始标音转写和去变音符的对应版本。

    arXiv:2608.26921v1 Announce Type: cross  Abstract: We introduce AraMS-28k, the largest publicly released line-level dataset of genuine historical Arabic manuscripts, comprising 14 books, 3,043 pages, and 28,600 annotated text lines (27,971 main-text, 629 margin). Thirteen books are hand-copied manuscripts spanning three script traditions -- Naskh, Ruq'ah, and Maghrebi -- and one is a lithographed printed edition included to broaden format diversity. Each line is labelled as main-text or margin, and margin lines that have an unambiguous attachment point in the main text are further annotated with an insertion anchor, recovering the manuscript's true non-linear reading order at line-level granularity -- to our knowledge the first such annotation released for a historical Arabic manuscript corpus. Because reference transcriptions are fully vocalised while manuscript hands are typically undiacritised, we release both the raw diacritised transcription and a diacritic-normalised counterpart 
    
[^51]: 在自然文本中植入潜变量：对LLM信念状态及其与概念几何联系的一种更现实测试

    Planting a Latent Variable in Natural-Looking Text: a More Realistic Test of Belief States in LLMs and Their Link to Concept Geometry

    [https://arxiv.org/abs/2608.26887](https://arxiv.org/abs/2608.26887)

    本工作通过在自然文本中植入可控潜变量，证明小型LLM不仅能追踪其贝叶斯后验信念，还能在特征几何中按马尔可夫链顺序排列这些状态，首次将信念状态与概念几何联系起来。

    

    arXiv:2608.26887v1 公告类型：新 摘要：大型语言模型（LLMs）被认为能追踪“信念状态”，即对支配语言的潜变量运行概率分布（Shai等人，2024；Sarfati等人，2026），但迄今为止，这一能力仅在玩具合成数据和少数孤立案例研究中得到全面证明，且从未与LLM特征的几何结构（可解释性在模型激活中发现的概念）建立实证联系。在本工作中，我们在自然文本中植入了一个可控潜变量。一个LLM教师编写普通文本，同时我们“潜意识地”在每一步沿K=8个不相关的稀疏自编码器方向之一引导它，活动方向遵循一个环形马尔可夫链。基于该语料库训练的小型变压器模型确实追踪了关于我们植入潜变量的贝叶斯后验信念。此外，它还将这8个状态本身排列成一个环，顺序与马尔可夫链完全一致。

    arXiv:2608.26887v1 Announce Type: new  Abstract: LLMs are thought to track "belief states," i.e., running probability distributions over the latent variables that govern language (Shai et al., 2024; Sarfati et al., 2026), but so far this has only been comprehensively demonstrated on toy synthetic data and in a few isolated case studies. It has also never been empirically connected to the geometry of LLM features (the concepts interpretability finds in model activations). In this work, we plant a controllable latent variable inside natural-looking text. An LLM teacher writes ordinary text while we "subliminally" steer it along one of K = 8 unrelated sparse autoencoder directions at each token, with the active directions following a ring-shaped Markov chain. A small transformer model trained on this corpus does indeed track the Bayesian posterior belief about our planted latent variable. Moreover, it also arranges the 8 states themselves on a ring, in the exact order of the Markov chain,
    
[^52]: C-Unseen：基于大语言模型推理的动态时序知识图谱弱信号检测

    C-Unseen: Weak Signal Detection in Dynamic Temporal Knowledge Graphs via LLM Reasoning

    [https://arxiv.org/abs/2608.26870](https://arxiv.org/abs/2608.26870)

    C-Unseen利用大语言模型推理和链式思维，通过识别并追踪罕见语义连贯子图在动态时序知识图谱中的持续增殖，实现自解释的弱信号检测。

    

    弱信号是在重大变化确立之前出现的早期、低可见性指标。现有的检测方法基于关键词频率、主题建模或非类型化图拓扑，未能捕捉这些信号所体现的语义和关系结构。在本文中，我们提出C-Unseen，一个用于动态时序知识图谱（DTKGs）中弱信号检测的自解释框架。我们将弱信号定义为在连续TKG快照中增殖的罕见、语义连贯的子图。该框架通过两个模块运作：罕见子图提取器，其中大语言模型通过链式思维推理识别与主导快照叙事内容相冲突的子图；以及弱信号警报器，其中跨时间步追踪这些罕见子图的持续性以隔离真正的弱信号。实验结果表明，C-Unseen在检测动态时序知识图谱中的弱信号方面表现出色。

    arXiv:2608.26870v1 Announce Type: new  Abstract: Weak signals are early, low-visibility indicators that precede significant changes before those changes become established. Existing detection methods, based on keyword frequency, topic modeling, or untyped graph topology, fail to capture the semantic and relational structure through which such signals manifest. In this paper, we propose C-Unseen, a self-interpretable framework for weak signal detection in Dynamic Temporal Knowledge Graphs (DTKGs). We define a weak signal as a rare, semantically coherent subgraph that proliferates across consecutive TKG snapshots. The framework operates through two modules: a Rare Subgraphs Extractor, in which an LLM identifies subgraphs whose content is in tension with the dominant snapshot narrative via chain-of-thought reasoning, and a Weak Signal Alerter, in which the persistence of these rare subgraphs is tracked across time steps to isolate true weak signals. Experimental results demonstrate that C
    
[^53]: 评估匹配轨迹回放的置信门控检索

    Evaluating Confidence-Gated Retrieval with Matched Trajectory Replay

    [https://arxiv.org/abs/2608.26846](https://arxiv.org/abs/2608.26846)

    该论文提出匹配轨迹回放协议，发现事后等渗校准在相同阈值下改变代理的承诺决策，显著提升已承诺答案的准确率（最高达41个百分点），但可能减少回答范围。

    

    arXiv:2608.26846v1 公告类型：交叉 摘要：交互式语言模型代理使用置信信号来决定是立即回答、检索额外证据（从记忆或外部知识），还是延迟回答。然而，置信度通常是在孤立情况下评估的，没有测量其触发动作的轨迹级后果。我们提出匹配轨迹回放，这是一种用于比较置信度到动作映射的受控协议。该协议保持候选答案状态、证据点、预算和动作成本固定。我们使用它来比较原始口头化置信度与事后等渗校准，在多跳问答系统中使用Mistral、GPT和Qwen模型，在HotpotQA和MuSiQue数据集上进行测试。在相同的数值承诺阈值下，校准改变了代理最终承诺回答的问题。在所有六种模型-数据集对中，它使承诺答案的准确率提高了多达41个百分点。然而，它可能减少...

    arXiv:2608.26846v1 Announce Type: cross  Abstract: Interactive language-model agents use confidence signals to decide whether to answer immediately, retrieve additional evidence (from memory or external knowledge), or defer. Yet confidence is usually evaluated in isolation, without measuring the trajectory-level consequences of the actions it triggers. We propose matched trajectory replay, a controlled protocol for comparing confidence-to-action mappings. The protocol holds candidate answer states, evidence points, budgets, and action costs fixed. We use it to compare raw verbalized confidence with post-hoc isotonic calibration in a multi-hop question-answering system using Mistral, GPT, and Qwen models on HotpotQA and MuSiQue datasets. At the same numerical commitment threshold, calibration changes which questions agents ultimately commit to answering. Across all six model-dataset pairs, it increases accuracy among committed answers by up to 41 percentage points. However, it can reduc
    
[^54]: SymbolLKG：通过逻辑知识图谱与符号求解器实现可验证的逻辑推理

    SymbolLKG: Towards Verifiable Logical Reasoning via Logical Knowledge Graph and Symbolic Solvers

    [https://arxiv.org/abs/2608.26836](https://arxiv.org/abs/2608.26836)

    本文提出了一种结合逻辑知识图谱和动态符号求解器路由的神经符号架构，通过将逻辑规则作为拓扑节点显式建模依赖，并采用拓扑感知检索和逻辑路由器，实现了可验证的逻辑推理，有效减少了LLM的幻觉和推理不一致问题。

    

    大型语言模型（LLMs）在自然语言理解方面表现出色，但在严格的多步推理中却面临困难，常常出现幻觉和不一致性问题。现有的解决方案如思维链（CoT）缺乏严格的验证机制，而标准检索增强生成（RAG）往往忽略了逻辑任务中固有的复杂结构依赖。为弥合这一差距，我们提出了一种神经符号架构，该架构将逻辑知识图谱（LKG）与动态求解器路由相结合。具体来说，我们引入了一个基于本体的LKG，将逻辑规则和约束视为一流的拓扑节点，从而能够显式建模从文本中提取的依赖关系。我们进一步设计了一个逻辑路由器，以动态地将任务分配给最优符号引擎，并由一种拓扑感知的混合检索机制提供支持。在逻辑任务上的实验结果表明...

    arXiv:2608.26836v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable proficiency in natural language understanding, yet they struggle with strict multi-step reasoning, frequently suffering from hallucinations and inconsistency. Existing solutions like Chain-of-Thought (CoT) lack rigorous verification mechanisms, while standard Retrieval-Augmented Generation (RAG) often misses the complex, structural dependencies inherent in logical tasks. To bridge this gap, we propose a Neuro-Symbolic architecture that integrates a Logical Knowledge Graph (LKG) with dynamic solver routing. Specifically, we introduce an ontology-based LKG that treats logical rules and constraints as first-class topological nodes, enabling explicit modeling of dependencies extracted from text. We further design a Logic Router to dynamically dispatch tasks to the optimal symbolic engine, which is supported by a topology-aware hybrid retrieval mechanism. Experimental results on logica
    
[^55]: RuleWeaver：为大型语言模型构建以规则为中心的场景推理基准

    RuleWeaver: Benchmarking Rule-Centered Scenario Reasoning for Large Language Models

    [https://arxiv.org/abs/2608.26832](https://arxiv.org/abs/2608.26832)

    本文提出RuleWeaver基准框架，通过从语料中提取并增强复杂规则来构建场景问答，以全面评估LLM在规则中心场景推理中的能力，并发现现有模型在此任务上仍有显著不足。

    

    arXiv:2608.26832v1 公告类型：新 摘要：大型语言模型（LLMs）越来越多地应用于专业领域，在这些领域中，有效利用领域专业知识通常需要在具体场景中推理复杂规则。然而，现有基准仅部分评估了这一能力，因为它们要么专注于输出级别的指令约束，要么忽视了规则在场景推理中扮演的不同角色。为解决这些差距，本文介绍了RuleWeaver，一个用于评估以规则为中心的场景推理的基准构建框架。RuleWeaver从语料库中提取的IF-THEN元规则开始，逐步将其增强为复杂规则，并将这些规则组合成以规则为中心的场景问答实例。除了最终答案的正确性外，RuleWeaver还通过基于评分标准的答案质量、规则召回率和规则精确率支持过程级评估。对11个代表性LLM的实验表明，当前模型在复杂规则推理方面仍存在困难。

    arXiv:2608.26832v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly applied to specialized domains, where effective use of domain expertise often requires reasoning over complex rules in concrete scenarios. However, existing benchmarks only partially evaluate this capability, as they either focus on output-level instruction constraints or overlook the distinct roles that rules play in scenario reasoning. To address these gaps, this paper introduces RuleWeaver, a benchmark construction framework for evaluating rule-centered scenario reasoning. RuleWeaver starts from corpus-derived IF-THEN Meta Rules, progressively augments them into complex rules, and composes these rules into rule-centered scenario QA instances. Beyond final-answer correctness, RuleWeaver further supports process-level evaluation through rubric-based answer quality, rule recall, and rule precision. Experiments on 11 representative LLMs show that current models still struggle with complex rule
    
[^56]: 行为到行程：通过用户行为轨迹实现个性化旅行规划

    Behavior2Trip: Towards Personalized Travel Planning via User Behavior Trajectory

    [https://arxiv.org/abs/2608.26807](https://arxiv.org/abs/2608.26807)

    本文提出行为感知旅行规划任务及Behavior2Trip基准，通过用户历史行为轨迹隐式推断偏好，生成个性化旅行计划，减少交互负担并提升个性化水平。

    

    arXiv:2608.26807v1 公告类型：交叉 摘要：旅行规划代理通过建模用户的个人偏好来协助用户生成个性化旅行计划。现有代理要么依赖明确的用户指令，要么通过多轮澄清来获取用户偏好。然而，这两种方法都忽略了用户过去行为中蕴含的丰富行为信号，这些信号隐式地编码了他们的偏好。这种对用户主动输入的过度依赖增加了交互负担，并限制了计划的个性化程度。为弥补这一差距，我们引入了一项新任务，即行为感知旅行规划，该任务直接从过去行为中推断用户偏好，并生成个性化旅行计划。为促进该任务的研究，我们构建了Behavior2Trip基准，该基准基于中国最大的在线旅行平台之一构建，包含11,400个实例。每个实例平均代表39.8条用户过去行为，涵盖5个偏好维度下的14个属性。

    arXiv:2608.26807v1 Announce Type: cross  Abstract: Travel planning agents assist users in generating personalized travel plans by modeling their individual preferences. Existing agents either rely on explicit user instructions or engage in multi-turn clarification to elicit user preferences. However, both approaches overlook the rich behavioral signals latent in users' past behaviors, which implicitly encode their preferences. This over-reliance on active user input increases interaction burden and limits plan personalization. To bridge this gap, we introduce a new task, Behavior-Aware Travel Planning, which infers user preferences directly from past behaviors and generates personalized travel plans. To facilitate research on this task, we introduce Behavior2Trip, a benchmark constructed from one of the largest Chinese online travel platforms, comprising 11,400 instances. Each instance represents an average of 39.8 past user behaviors spanning 14 attributes across 5 preference dimensio
    
[^57]: 可指导代理的规划与控制解耦

    Decoupling Planning and Control for Instructable Agents

    [https://arxiv.org/abs/2608.26788](https://arxiv.org/abs/2608.26788)

    本文提出Instruct-to-Act系统，通过将VLM规划器生成的高层文本指令与世界模型控制器的低延迟动作执行解耦，并利用合成指令重标记和联合优化训练，使控制器能够接受语言指导，从而在陌生环境中实现可靠且高效的自主行动。

    

    近期研究表明，预训练的、指令调优的视觉语言模型（VLMs）在将指令和观察映射为高级计划方面表现良好，但在陌生环境中难以将这些计划实现为可靠的低延迟动作序列。与此同时，世界模型控制器擅长快速的观察到动作控制，但缺乏开放式任务指导。在本工作中，我们将这些优势结合到一个单一系统中，名为Instruct-to-Act，我们训练一个世界模型控制器，在由VLM规划器生成的稀疏、高延迟且高级文本指令条件下，以高频率自主行动。为了训练控制器使其可接受语言指令，我们用合成指令重新标记控制器策略回放片段，并联合优化行为克隆目标以及现有的奖励最大化和世界建模目标。我们在七个实体上评估了我们提出的方法。

    arXiv:2608.26788v1 Announce Type: new  Abstract: Recent work shows that pre-trained, instruction-tuned vision-language models (VLMs) perform well at mapping from instructions and observations to high-level plans, but struggle to realize such plans as reliable low-latency action sequences in unfamiliar environments. At the same time, world-model controllers excel at fast observation-to-action control, but lack open-ended task guidance. In this work, we combine these strengths into a single system, Instruct-to-Act, where we train a world-model controller to act autonomously at high frequency when conditioned on sparse, higher-latency, and high-level text instructions generated by a VLM planner. To train controllers to be language-instructable, we relabel segments of controller policy rollouts with synthetic instructions and jointly optimize a behavior-cloning objective along with existing reward-maximizing and world-modeling objectives. We evaluate our proposed approach across seven embo
    
[^58]: 指令质量至关重要：优化指令以促进有效的偏好学习

    Instruction Quality Matters: Refining Instructions for Effective Preference Learning

    [https://arxiv.org/abs/2608.26779](https://arxiv.org/abs/2608.26779)

    本文提出指令质量是偏好学习的隐藏瓶颈，并引入指令优化流程，通过选择弱指令和基于评分标准的LLM反馈修订，有效提升偏好数据质量，从而在离线和在线设置中显著改善模型对齐性能。

    

    arXiv:2608.26779v1 公告类型：新 摘要：偏好学习利用响应配对来优化模型，然而这些配对的信息量从根本上受到生成它们的指令的影响。我们识别出指令质量是偏好学习中的一个隐藏瓶颈：低质量或模糊的指令限制了响应质量的分布，从而削弱了强选择响应并减弱了偏好信号。通过最佳和最差N项分析，我们表明指令质量同时限制了采样响应质量的上限和下限。基于这一观察，我们引入了一个指令优化流程，该流程利用奖励信号选择弱指令，并通过基于评分标准的LLM反馈对其进行修订，从而在不丢弃样本的情况下改进偏好数据。在离线和在线偏好学习设置中，对多个模型和基准的实验表明，与原始数据和替代数据相比，对齐性能得到了广泛提升。

    arXiv:2608.26779v1 Announce Type: new  Abstract: Preference learning optimizes models using response pairs, yet the informativeness of these pairs is fundamentally shaped by the instructions from which they are generated. We identify instruction quality as a hidden bottleneck in preference learning: low-quality or ambiguous instructions restrict the response-quality distribution, limiting strong chosen responses and weakening preference signals. Through Best- and Worst-of-N analyses, we show that instruction quality constrains both the ceiling and floor of sampled response quality. Motivated by this observation, we introduce an instruction-refinement pipeline that selects weak instructions using reward signals and revises them with rubric-guided LLM feedback, improving preference data without discarding examples. Across offline and online preference learning settings, experiments on multiple models and benchmarks show broad alignment improvements over original data and alternative data
    
[^59]: 同等排序质量，不同决策：训练顺序一致的LLM评分器

    Equal Ranking Quality, Different Decisions: Training Order-Consistent LLM Scorers

    [https://arxiv.org/abs/2608.26762](https://arxiv.org/abs/2608.26762)

    本文发现LLM评分器在排序质量相同时决策不一致，提出OC-SFT方法通过训练分数顺序无关性来保持排序质量并提升决策稳定性。

    

    arXiv:2608.26762v1 公告类型：新 摘要：重排序器、奖励模型和多文档问答评分器在一个LLM提示中为候选文档或响应打分，因此每个分数依赖于它们的顺序。这类评分器基于排序质量进行选择，但它们的分数决定了一个决策：分数阈值保留的内容、读者回答的内容或偏好模型选择的内容。然而，同等排序质量并不意味着同等决策：在段落重排序中，五个训练评分器在nDCG@10上相差0.010以内，但重新排序后保留集仅重叠0.66-0.84。一个已发表的重排序器在我们比较中取得最高保留集F1，但重叠率仍仅为0.667。我们测试的任何提示时间变化都无法消除这种顺序依赖性：唯一提升排序质量的改变使三个决策均未变化。顺序一致性SFT（OC-SFT）在权重中削弱了这种依赖性，训练候选分数不依赖于顺序。它保持了排序质量，并使训练评分器之间的所有决策稳定性指标均得到提升。

    arXiv:2608.26762v1 Announce Type: new  Abstract: Rerankers, reward models and multi-document QA scorers score candidate documents or responses in one LLM prompt, so each score depends on their order. Such scorers are selected on ranking quality, but their scores determine a decision: what a score threshold retains, a reader answers, or a preference model selects. However, equal ranking quality does not imply equal decisions: on passage reranking, five trained scorers within 0.010 nDCG@10 retain sets that overlap by only 0.66-0.84 when reordered. A published reranker takes the highest retained-set F1 in our comparison and still overlaps by only 0.667. No prompt-time change we test removes that order dependence: the only one that gains ranking quality leaves all three decisions unchanged. Order-consistency SFT (OC-SFT) attenuates it in the weights, training a candidate's score not to depend on the order. It holds ranking quality and leads every decision-stability measure among trained sc
    
[^60]: 字母掩盖了我们眼中的真相：英语同音词在语音实现上存在有意义的差异

    Letters hide the truth from our eyes: English homophones have meaningfully different phonetic realizations

    [https://arxiv.org/abs/2608.26749](https://arxiv.org/abs/2608.26749)

    本研究通过分析大量英语同音词，发现其语音实现差异与语境含义相关，并采用时间归一化频谱图揭示了传统转录所掩盖的语音细节。

    

    arXiv:2608.26749v1 公告类型：新 摘要：已知英语同音词的口语词长分布与使用频率共变。本研究探讨同音词在语音实现的其他方面是否也存在差异。通过对美国电视新闻广播中14,000个同音词标记进行一系列定量研究，发现诸如“weight”和“wait”等同音词对的标记具有不同的语音实现，且这些差异可根据其在话语语境中的含义进行预测。即使在考虑与时长相关的变异后，这些系统性差异仍然存在。时间归一化频谱图成为探测语音实现细节的绝佳工具，并消除了对语音转录的需求，因为转录不可避免地会掩盖我们眼中的语音真相。

    arXiv:2608.26749v1 Announce Type: new  Abstract: The distribution of spoken word duration of English homophones is known to co-vary with frequency of use. This study investigates whether other aspects of the phonetic realization of homophones also differ. A series of quantitative investigations of 14,000 homophone tokens in American television news broadcasts revealed that the tokens of homophone pairs such as \textit{weight} and \textit{wait} have different phonetic realizations, and that these can be predicted from their meanings in utterance context. These systematic differences remain even when taking duration-related variation into account. Time-normalized spectrograms emerged as an excellent tool for probing the fine details of phonetic realization, and obviate the need for phonetic transcriptions, which inevitably hide the phonetic truth from our eyes.
    
[^61]: 在领域专业化过程中通过不确定性校准的MOPD保持通用能力

    Preserving General Capabilities during Domain Specialization with Uncertainty-Calibrated MOPD

    [https://arxiv.org/abs/2608.26735](https://arxiv.org/abs/2608.26735)

    本文提出不确定性校准的MOPD方法，通过双温度采样和正优势密度过滤扩大有效学习轨迹，并利用中心化对数似然过滤确保更新可靠性，从而在领域专业化时保持模型的通用能力。

    

    arXiv:2608.26735v1 公告类型：新 摘要：将大型语言模型专业化到垂直领域能改善领域特定行为，但往往会削弱通用能力，如推理、编码、指令遵循和创意写作。我们研究了多教师在线策略蒸馏（MOPD）中的这种领域-通用权衡，其中专业化学生在其自身采样的轨迹上接受领域和通用教师的监督。标准MOPD面临两个限制：普通的在线策略采样很少暴露具有较大正教师-学生优势的令牌，而优势符号本身并不能确定生成的更新方向是否可靠。我们提出了不确定性校准的MOPD来解决这些限制。双温度采样扩大了候选轨迹池，正优势密度过滤选择具有更强正向学习信号的轨迹。然后，中心化对数似然（CLL）过滤计算熵校准的教师信号，以进一步确保更新方向的可靠性。

    arXiv:2608.26735v1 Announce Type: new  Abstract: Specializing large language models to vertical domains improves domain-specific behavior but often degrades general capabilities such as reasoning, coding, instruction following, and creative writing. We study this domain--general trade-off in Multi-Teacher On-Policy Distillation (MOPD), where a specialized student is supervised on its own sampled trajectories by domain and general teachers. Standard MOPD faces two limitations: ordinary on-policy sampling rarely exposes tokens with large positive teacher--student advantages, while the advantage sign alone does not establish whether the resulting update direction is reliable. We propose uncertainty-calibrated MOPD to address these limitations. Dual-temperature sampling broadens the candidate trajectory pool, and positive-advantage-density filtering selects trajectories with stronger positive learning signals. Centered log-likelihood (CLL) filtering then computes an entropy-calibrated teac
    
[^62]: 迈向专家级金融问答：基于自我改进的检索增强生成

    Towards Expert Financial QA via Self-Improving RAG

    [https://arxiv.org/abs/2608.26706](https://arxiv.org/abs/2608.26706)

    本文提出一种自我改进的检索增强生成框架，通过三个代理和动态阈值重试机制，在金融文档问答中显著提升准确率，尤其能恢复近四成初始错误答案。

    

    专家级金融问答既需要基于事实的验证以捕捉数字幻觉，又需要审计追踪以满足监管合规要求，而标准单次检索增强生成（RAG）系统不具备这些属性。我们通过自我改进的检索增强生成（Self-Improving RAG）框架向这一目标迈进了一步，该框架将文档问答分解为三个专门代理（检索、推理和评判），由编排器协调，并带有反馈驱动的自我修正机制。当评判代理对答案的评分低于动态阈值时，系统会触发重试，并采用升级策略：更广泛的检索、更细致的提示和放宽的接受标准。我们在FinanceBench（美国证券交易委员会文件问答）上进行了评估，自我改进的检索增强生成实现了86%的Oracle引导准确率（衡量与黄金答案的一致性），拉撒路率为36.4%，即通过有针对性的重试，近四成最初错误的答案得以恢复。一个关键发现是，固定的检索流程加上评判驱动的自我修正，能显著提升问答的准确性和可靠性。

    arXiv:2608.26706v1 Announce Type: new  Abstract: Expert-level financial question answering requires both grounded verification to catch numeric hallucinations and audit trails for regulatory compliance, attributes that standard single-pass RAG systems lack. We take a step toward this goal with Self-Improving RAG, a framework that decomposes document QA into three specialized agents (Retrieval, Reasoning, and Judge) coordinated by an orchestrator with feedback-driven self-correction. When the Judge Agent scores an answer below a dynamic threshold, the system triggers retry with escalated strategies: broader retrieval, more careful prompting, and relaxed acceptance criteria. We evaluate on FinanceBench (SEC filing QA), where Self-Improving RAG achieves 86% oracle-guided accuracy (measuring agreement with gold answers) with a 36.4% Lazarus Rate, recovering nearly 4 in 10 initially incorrect answers through targeted retry. A key finding is that a fixed retrieval pipeline with judge-driven 
    
[^63]: PragAlign：面向中日得体性判断的证据敏感回复辅助

    PragAlign: Evidence-Sensitive Reply Assistance Across Chinese and Japanese Appropriateness Judgments

    [https://arxiv.org/abs/2608.26700](https://arxiv.org/abs/2608.26700)

    本文提出PragAlign模型，通过分离上下文阅读与选择性澄清，在中日双语得体性判断中显著优于基线，揭示跨语言共性判断模式。

    

    多语言环境中的回复辅助需要语言能力和文化情境下的得体性判断。我们提出了PragAlign，它将上下文阅读与选择性澄清分离，并与Direct和Rule两种基线方法进行了评估。九名以中文为母语的评判者评估了中文材料，而三名以日语为母语的评判者评估了匹配的日文版本。在中文评估中，PragAlign获得的排名显著优于两种基线方法。在日文评估中，Direct的平均排名最低，PragAlign的最高排名率最高，但总体差异不显著。两组在10个场景中有5个选择了相同的顶级条件，其中包括四个共同的PragAlign选择。结果揭示了共性和语言特定的判断模式，并为旨在支持语言和文化理解的回复辅助提供了参考。

    arXiv:2608.26700v1 Announce Type: new  Abstract: Reply assistance in multilingual settings requires linguistic competence and culturally situated judgments of appropriateness. We present PragAlign, which separates context reading from selective clarification, and evaluate it alongside Direct and Rule. Nine native Chinese speakers judged Chinese materials, while three native Japanese speakers judged matched Japanese versions. In the Chinese evaluation, PragAlign received significantly better ranks than both baselines. In the Japanese evaluation, Direct had the lowest mean rank, PragAlign had the highest top-rank rate, and the omnibus difference was not significant. The groups selected the same top condition in 5 of 10 scenarios, including four shared PragAlign selections. The results identify shared and language-specific judgment patterns and inform reply assistance designed to support linguistic and cultural understanding.
    
[^64]: 基于音素的TTS增强用于ASR的扩展：统一流程与受控研究

    Scaling phoneme-based TTS augmentation for ASR: A unified pipeline and controlled study

    [https://arxiv.org/abs/2608.26697](https://arxiv.org/abs/2608.26697)

    本文提出了一种基于音素的统一TTS到ASR增强流程，并引入音素频率引导选择（PFGS）方法，在多种语言的ASR任务中有效提升了性能。

    

    合成语音为自动语音识别（ASR）提供了可扩展的监督信号，但其效果取决于所选的文本、参考语音和合成数据量。我们提出了一种统一的基于音素的TTS到ASR增强流程，该流程围绕一个使用F5-TTS架构从头训练并带有语言ID条件化的多语言TTS模型构建。该流程结合了特定语言的音素转换、参考语音过滤、候选文本选择、合成和匹配的ASR续训练。我们进一步提出了音素频率引导选择（PFGS），该方法利用从真实ASR训练标签中估计的音素频率对候选句子进行排序。针对阿拉伯语、法语、意大利语和葡萄牙语的独立单语ASR系统的实验覆盖了13个测试集。在合成规模扫描中，随机增强在11个测试集上优于仅匹配真实数据的续训练。在标称60%合成比例下，该方法进一步提升了性能。

    arXiv:2608.26697v1 Announce Type: new  Abstract: Synthetic speech provides scalable supervision for automatic speech recognition (ASR), but its benefit depends on the selected texts, reference speech, and amount of synthesized data. We present a unified phoneme-based TTS-to-ASR augmentation pipeline built around a multilingual TTS model trained from scratch using the F5-TTS architecture with language-ID conditioning. The pipeline combines language-specific grapheme-to-phoneme conversion, reference-speech filtering, candidate-text selection, synthesis, and matched ASR continuation. We further propose phoneme-frequency-guided selection (PFGS), which ranks candidate sentences using phoneme frequencies estimated from real ASR training labels. Experiments with separate monolingual ASR systems for Arabic, French, Italian, and Portuguese span 13 test sets. Across the synthesis-scale sweep, random augmentation improves over matched real-only continuation on 11 test sets. Under a nominal 60% sy
    
[^65]: 超越反思：肯定作为基于文本咨询中与质量相关的有前景行为标志

    Beyond Reflection: Affirmation as a Promising Behavioral Marker Associated with Quality in Text-Based Counseling

    [https://arxiv.org/abs/2608.26689](https://arxiv.org/abs/2608.26689)

    本研究通过多层级分析发现，在文本咨询中“肯定”行为比“反思”行为更能稳定预示对话质量，且该信号在跨语言数据集上也有一定表现，为咨询培训提供了新视角。

    

    虽然人工智能辅助的基于文本咨询正受到关注，但哪些咨询师行为与更高对话质量相关，在实证上仍不明确。现有研究往往过度聚焦于“反思”，借用动机性访谈的框架。为填补这一空白，我们使用KokoroChat（一个由专业咨询师和受训者进行的大规模日语文本咨询数据集，并新增了咨询师策略标签和客户困扰程度标注）进行了多层次分析。结果显示，在本研究所用的质量指标下，在所分析的策略中，“肯定”与会谈质量的一致性关联强于“反思”。跨数据集迁移实验进一步表明，这种质量信号在一定程度上也能在ESConv（一个由非专家支持者组成的英语数据集）上观察到。这些发现为咨询师培训和情感支持提供了实证启示。

    arXiv:2608.26689v1 Announce Type: new  Abstract: While AI-assisted text-based counseling is gaining attention, it remains empirically unclear which counselor behaviors are associated with higher dialogue quality. Existing research often focuses heavily on Reflection, borrowing frameworks from Motivational Interviewing. To address this gap, we conduct a multi-layered analysis using KokoroChat, a large-scale Japanese text counseling dataset conducted by professional counselors and trainees, newly annotated with counselor strategy tags and client distress levels. Our results show that, under the quality indicators used in this study, Affirmation is more consistently associated with session quality than Reflection among the analyzed strategies. Cross-dataset transfer experiments further suggest that this quality signal can be observed to some extent on ESConv, an English dataset with non-expert supporters. These findings provide empirical implications for counselor training and emotional s
    
[^66]: FOCUS & RePAIR：通过令牌级引导缓解剪枝大型语言模型中的文本退化

    FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance for Pruned Large Language Models

    [https://arxiv.org/abs/2608.26676](https://arxiv.org/abs/2608.26676)

    本文提出FOCUS和RePAIR两种令牌级引导方法，通过分析剪枝LLM中的文本退化机制，分别抑制教师泄漏和防止重复循环，从而在不牺牲任务性能的前提下缓解退化问题。

    

    剪枝是压缩大型语言模型（LLMs）的一种实用方法，但它可能加剧文本退化，尤其是重复循环，即使困惑度和任务准确性基本保持不变。在本工作中，我们通过将解码视为一个进入并持续于少量循环上下文的动态过程，对这种失败模式进行了令牌级分析。我们的分析将退化分解为循环进入风险和循环持续性，并表明持续性由分配给令牌采样集中合理替代方案的逃逸质量所控制。基于这些发现，我们提出了两种用于剪枝后微调的令牌级引导目标。FOCUS将蒸馏重新加权到高置信度的教师区域以抑制泄漏，而RePAIR使用以起始为中心的正/负延续对配合边际损失，以促进合理替代方案并防止过早陷入重复循环。

    arXiv:2608.26676v1 Announce Type: cross  Abstract: Pruning is a practical approach to compress large language models (LLMs), but it can amplify text degeneration, especially repetition loops, even when perplexity and task accuracy remain largely unchanged. In this work, we present a token-level analysis of this failure mode by viewing decoding as a dynamical process that enters and persists in a small set of recurrent contexts. Our analysis decomposes degeneration into loop entry risk and loop persistence, and shows that persistence is controlled by the escape mass assigned to plausible alternatives within the token sampling set. Motivated by these findings, we propose two token-level guidance objectives for post-pruning fine-tuning. FOCUS reweights distillation toward high-confidence teacher regions to suppress leakage, while RePAIR uses onset-centered positive/negative continuation pairs with a margin loss to promote plausible alternatives and prevent early commitment to repetition l
    
[^67]: 大型语言模型理解人格吗？通过结构化行为推断重新思考人格保真度评估

    Do LLMs Understand Personality? Rethinking Persona Fidelity Evaluation through Structured Behavioral Inference

    [https://arxiv.org/abs/2608.26674](https://arxiv.org/abs/2608.26674)

    本文提出PRISM框架，通过系统功能语言学启发的结构化逆推理方法，解决了现有LLM人格保真度评估中的整体评估幻觉和静态测量局限。

    

    随着大型语言模型越来越多地被部署用于模拟多样化的人类角色，确保人格保真度——即代理的行为在多大程度上持续反映目标人格的心理和风格特征——已成为一项关键要求。然而，现有的评估范式主要依赖于基于整体LLM的评判者，这些评判者容易产生“整体评估幻觉”，或依赖于静态心理测量量表，这些量表无法捕捉动态对话中所需的上下文依赖保真度。为解决这些局限性，我们提出了PRISM（基于逆SFL建模的人格推理），这是一种基于心理语言学的框架，将人格保真度评估重新表述为结构化的逆推理任务。受系统功能语言学（SFL）启发，PRISM将人格保真度分解为三个功能维度：任务框架、人际立场和语言风格。

    arXiv:2608.26674v1 Announce Type: cross  Abstract: As large language models are increasingly deployed to simulate diverse human characters, ensuring persona fidelity, defined as the extent to which an agent's behavior consistently reflects the psychological and stylistic characteristics of a target persona, has become a critical requirement. However, existing evaluation paradigms primarily rely on either holistic LLM-based judges, which are prone to "holistic appraisal hallucination'', or static psychometric inventories, which fail to capture the context-dependent fidelity required in dynamic dialogue. To address these limitations, we propose PRISM (Persona Reasoning with Inverse SFL-based Modeling), a psycholinguistically grounded framework that reformulates persona fidelity evaluation as a structured inverse inference task. Inspired by Systemic Functional Linguistics (SFL), PRISM decomposes persona fidelity into three functional dimensions: Task Framing, Interpersonal Stance, and Lin
    
[^68]: 元学习专家分配位置：基于任务条件的MoE分层压缩

    Meta-Learning Where to Allocate Experts: Task-Conditioned Layer-Wise Compression for MoEs

    [https://arxiv.org/abs/2608.26650](https://arxiv.org/abs/2608.26650)

    本文提出MetaNet，一种任务条件化的分层压缩方法，通过元学习预测每层的专家保留阈值和路由偏差，在冻结模型参数下实现可调的准确率-专家激活权衡，减少约40%的专家激活而保持性能。

    

    混合专家（MoE）模型将每个标记路由到专家网络的子集，在增加容量的同时保持每标记计算的稀疏性。在许多已部署的MoE中，活跃专家数量在层和任务间是固定的，尽管层角色和专家冗余随深度变化，需求随难度变化。现有方法仅解决该设置的一部分：层间分配通常离线确定并复用于所有任务，而令牌级方法使用局部路由信号变化专家激活，缺乏任务级上下文。我们提出MetaNet，一种支持集控制器，为每层预测专家保留阈值和有界路由偏差。骨干网络、专家和路由器保持冻结。在DeepSeek-MoE-16B-Chat上，MetaNet提供了可调的准确性-专家激活权衡。相对于固定k=6，保守设置平均激活3.61个专家（减少40%），并达到可比性能。

    arXiv:2608.26650v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models route each token to a subset of expert networks, increasing capacity while keeping per-token computation sparse. In many deployed MoEs, the number of active experts is fixed across layers and tasks, although layer roles and expert redundancy vary with depth and demand varies with difficulty. Existing approaches address only part of this setting: layer-wise allocations are usually determined offline and reused for all tasks, while token-level methods vary expert activation using local routing signals without task-level context. We propose MetaNet, a support-set controller that predicts, for each layer, an expert-retention threshold and a bounded routing bias. The backbone, experts, and router remain frozen. On DeepSeek-MoE-16B-Chat, MetaNet provides a tunable accuracy-expert-activation trade-off. Relative to fixed k=6, a conservative setting activates 3.61 experts on average (40% fewer) and achieves compara
    
[^69]: 信息引导的前沿解码：扩散多模态语言模型中的上下文效用驱动承诺

    Information-Guided Frontier Decoding: Contextual Utility-Driven Commitment in dMLLMs

    [https://arxiv.org/abs/2608.26641](https://arxiv.org/abs/2608.26641)

    提出一种无需训练的解码策略IGFD，通过结合置信度、邻域不确定性和结构风险来优化令牌承诺顺序，优先保留语义锚点并延迟结构脆弱令牌，从而提升扩散多模态语言模型的解码质量。

    

    扩散多模态语言模型（dMLLMs）的解码质量在很大程度上取决于掩码令牌被承诺的顺序。现有的基于置信度的策略优先处理局部简单的令牌，但置信度并不一定反映上下文的有用性。因此，结构上简单的令牌（如标点符号）可能在信息丰富的语义锚点之前被承诺，从而削弱上下文传播并增加错误累积。我们提出了信息引导的前沿解码（IGFD），这是一种无需训练的解码策略，它使用令牌置信度、邻域不确定性和结构承诺风险对候选进行排名。IGFD鼓励在解码过程中尽早承诺可靠的语义锚点，同时延迟脆弱的结构令牌，从而改善上下文支持。动态候选前沿进一步将令牌选择限制在相同解码预算下的局部可扩展区域内。该方法无需额外训练，仅需少量计算开销。

    arXiv:2608.26641v1 Announce Type: new  Abstract: Decoding quality in diffusion multimodal language models (dMLLMs) depends heavily on the order in which masked tokens are committed. Existing confidence-based strategies prioritize locally easy tokens, but confidence does not necessarily reflect contextual usefulness. As a result, structurally easy tokens such as punctuation may be committed before informative semantic anchors, weakening context propagation and increasing error accumulation. We propose Information-Guided Frontier Decoding (IGFD), a training-free decoding strategy that ranks candidates using token confidence, neighborhood uncertainty, and structural commitment risk. IGFD encourages early commitment of reliable semantic anchors while delaying fragile structural tokens, improving contextual support during decoding. A dynamic candidate frontier further constrains token selection to locally expandable regions under the same decoding budget. The method requires no additional t
    
[^70]: 哪些指标能节省最多的人工标注？预测驱动评估与元评估

    Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation

    [https://arxiv.org/abs/2608.26638](https://arxiv.org/abs/2608.26638)

    本文提出预测驱动评估框架，结合少量人工判断与大规模自动评分实现无偏且高效的系统比较，并引入PPSR元指标来衡量自动指标节省人工标注的程度，优于现有元指标。

    

    arXiv:2608.26638v1 公告类型：新 摘要：在各种不可验证的任务中，人工评估可靠但昂贵，而自动指标更具可扩展性但往往存在偏差。基于预测驱动推断（PPI），我们提出了预测驱动评估框架，该框架将有限的人工判断与大规模自动评分相结合，以获得数据高效且可证明无偏的系统比较。我们开发了参数化和非参数化程序，分析了配对与非配对设计之间的效率权衡，并在六个WMT数据集上验证了该框架。我们进一步引入了预测驱动节省率（PPSR），这是一种元指标，用于衡量在预测驱动评估中使用自动指标时可以节省多少人工标注。PPSR直接针对预测驱动评估的指标效用，并比现有系统级元指标产生更具区分性和稳定性的指标排名。总体而言，我们的新范式重新定义了...

    arXiv:2608.26638v1 Announce Type: new  Abstract: Across various non-verifiable tasks, human evaluation is reliable but expensive, while automatic metrics are more scalable but often biased. Building on prediction-powered inference (PPI), we propose prediction-powered evaluation, a framework that combines limited human judgments with large-scale automatic scores to obtain data-efficient system comparisons that are provably unbiased. We develop parametric and non-parametric procedures, analyze the efficiency trade-off between paired and unpaired designs, and validate the framework on six WMT datasets. We further introduce the Prediction-Powered Saving Ratio (PPSR), a meta-metric that measures how much human annotation an automatic metric can save when used within prediction-powered evaluation. PPSR directly targets metric utility for prediction-powered evaluation and yields more discriminative and stable metric rankings than existing system-level meta-metrics. Overall, our new paradigm r
    
[^71]: 千图假说：仓库级代码推理中任务条件化关系具体化的可检验假说

    The Thousand-Graph Hypothesis: A Testable Hypothesis of Task-Conditioned Relation Materialization in Repository-Level Code Reasoning

    [https://arxiv.org/abs/2608.26602](https://arxiv.org/abs/2608.26602)

    本文提出一种仅含实体的外部接口，通过推理时的任务条件化关系具体化，在无需预构建关系图的情况下，显著提升了仓库级代码推理的成功率。

    

    大型软件仓库通常超出模型上下文限制。将仓库知识训练进模型成本高昂且容易过时，而局部检索可能遗漏分散的需求，显式关系图则增加了持续维护负担。我们提出了一种仅含实体的外部接口，在推理过程中进行任务条件化关系具体化。一个两层索引将全局路由与局部实体聚焦分离，并在DeepSeek-V4-Flash和SWE-bench Verified上进行了评估。在零预构建实体-关系边的情况下，基础、单层和两层条件分别达到92.1%、94.2%和95.6%的成功率。

    arXiv:2608.26602v1 Announce Type: cross  Abstract: Large software repositories are often beyond model context limits. Training repository knowledge into models is costly and quickly stale, while local retrieval can miss scattered requirements, and explicit relation graphs add ongoing maintenance burden. We propose an entity-only external interface with task-conditioned relation materialization during inference. A two-layer index separates global routing from local entity focus and is evaluated on DeepSeek-V4-Flash and SWE-bench Verified. The base, one-layer, and two-layer conditions achieve 92.1%, 94.2%, and 95.6% success, respectively, under zero pre-built entity-relation edges.
    
[^72]: 不仅仅是推理，不仅仅是扫描：面向学术论文主动科学错误验证的强化学习

    Not Just Reason, Not Just Scan: Reinforcement Learning for Proactive Scientific Error Verification over Academic Paper

    [https://arxiv.org/abs/2608.26596](https://arxiv.org/abs/2608.26596)

    本文提出VERA-RL，一种基于强化学习的框架，用于在无预设问题和证据的情况下主动检测学术论文中的科学错误，并构建了包含12,900个样本的VERA-13K数据集以支持这一任务。

    

    arXiv:2608.26596v1 公告类型：新 摘要：多模态大语言模型（MLLMs）日益成为有能力的科学助手，但它们距离完全自主的研究仍相去甚远。这一转变要求模型主动检查学术论文，构建全局证据视图，并在没有预设问题或证据的情况下做出可追溯的判断。然而，现有工作对于这种无问题和无证据的验证所提供的任务范式或训练研究十分有限。我们通过科学错误检测来研究这一挑战，模型必须判断是否存在错误，并基于证据推理进行论证。为填补这一空白，我们提出了VERA-RL，一种用于学术论文科学错误检测的强化学习框架。遵循“推理—验证—扫描”的递进过程，我们构建了VERA-13K数据集，包含12,900个样本，组织为4,300个匹配链，覆盖研究工作流程中的6类科学错误类别以及广泛的自然科学领域。

    arXiv:2608.26596v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) are increasingly capable scientific assistants, yet they remain far from fully autonomous research. This transition requires models to actively inspect academic papers, build global evidence views, and make traceable judgments without prespecified issues or evidence. However, existing work provides limited task paradigms or training studies for such issue- and evidence-absent verification. We study this challenge through scientific error detection, where models must determine whether errors exist and justify them with evidence-based reasoning. To fill this gap, we present VERA-RL, a reinforcement-learning formulation for scientific error detection over academic papers. Following a Reason--Verify--Scan progression, we construct VERA-13K, a 12,900-sample dataset organized into 4,300 matched chains, covering 6 scientific-error categories across the research workflow and broad natural-science domains.
    
[^73]: 大型语言模型中临床决策路径遵循度的基准测试

    Benchmarking Clinical Decision Pathway Adherence in Large Language Models

    [https://arxiv.org/abs/2608.26592](https://arxiv.org/abs/2608.26592)

    该论文提出了MEGA-CDP基准，首次系统评估医学大语言模型遵循临床指南路径的能力，并发现现有模型在此任务上仍面临挑战。

    

    arXiv:2608.26592v1 公告类型：新 摘要：遵循临床实践指南定义的临床决策路径（CDP）对于安全可靠的医疗决策至关重要。然而，现有的医学大型语言模型（LLM）基准主要评估最终答案的准确性，对模型遵循指南能力的评估有限。为解决这一缺口，我们引入了MEGA-CDP，一个基准测试，用于评估医学LLM是否能以提供的指南为参考生成符合指南的CDP。MEGA-CDP基于2,274份中英文临床实践指南，通过“指南到案例”流程构建，生成42,353个具有明确参考CDP的临床案例。它支持单轮情景和多轮交互设置，并引入了一个面向CDP的评估框架来衡量路径一致性。对16个代表性LLM的实验表明，当前模型实现可靠的临床决策支持仍具挑战性。

    arXiv:2608.26592v1 Announce Type: new  Abstract: Following clinical decision pathways (CDPs) defined by clinical practice guidelines is essential for safe and reliable medical decision-making. However, existing medical large language model (LLM) benchmarks mainly evaluate final-answer accuracy, providing limited evaluation of models' ability to adhere to guidelines. To address this gap, we introduce MEGA-CDP, a benchmark for evaluating whether medical LLMs can generate guideline-adherent CDPs using provided guidelines as references. MEGA-CDP is constructed from 2,274 English and Chinese clinical practice guidelines through a guideline-to-case pipeline, yielding 42,353 clinical cases with explicit reference CDPs. It supports both single-turn vignette and multi-turn interactive settings, and introduces a CDP-oriented evaluation framework for measuring pathway consistency. Experiments on 16 representative LLMs show that reliable clinical decision support remains challenging for current mo
    
[^74]: 临床诊断大语言模型训练中的知识图谱外科式对齐

    Surgical Alignment in Knowledge Graph Training for Clinical Diagnosis with Large Language Models

    [https://arxiv.org/abs/2608.26587](https://arxiv.org/abs/2608.26587)

    本文提出“外科式对齐”概念，通过梯度干预密度和梯度扭曲指标，发现KL正则化下的知识图谱判断训练能产生稀疏局部更新，优于任务特定SFT的密集更新，从而更有效地将KG知识整合到LLM中用于临床诊断。

    

    生物医学知识图谱（KGs）提供了结构化的医学知识，能够在大语言模型（LLM）的临床诊断应用中支撑其推理，但如何将KG信号整合到LLM中仍是一个开放问题。我们进行了一项系统性研究，涵盖五种KG任务表述、三种训练范式、两个知识图谱和三个基础LLM。在任务层面，所有范式都优于未微调的基线，但具有相当领域内准确度的方法在知识迁移行为上表现出显著差异。我们引入了梯度干预密度（GID）和梯度扭曲（GD）来度量优化器对预训练模型的修改广度。GID和GD共同揭示了一个明确的分界：在KL正则化下的KG判断训练产生稀疏、局部的更新（我们称之为“外科式对齐”），而任务特定的SFT则产生密集更新。一项受控消融实验表明，目标和KL贡献是这一差异的关键因素。

    arXiv:2608.26587v1 Announce Type: new  Abstract: Biomedical knowledge graphs (KGs) offer structured medical knowledge that can ground large language model (LLM) reasoning in clinical diagnosis application, yet how KG signal should be integrated into LLMs remains an open question. We present a systematic study spanning five KG task formulations, three training paradigms, two KGs, and three base LLMs. At the task level, all paradigms improve over the non-finetuned baseline, but methods with comparable in-domain accuracy show substantially different knowledge transfer behavior. We introduce Gradient Intervention Density (GID) and Gradient Distortion (GD) to measure how broadly an optimizer modifies the pretrained model. GID and GD together reveal a clear divide: KG-judgment training under KL regularization produces sparse, localized updates (a regime we term as surgical alignment), while task-specific SFT produces dense ones. A controlled ablation shows that the objective and KL contribut
    
[^75]: J-Zero：从零数据出发的统一挑战者-求解者-评判者协同进化

    J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data

    [https://arxiv.org/abs/2608.26582](https://arxiv.org/abs/2608.26582)

    J-Zero提出了一种统一的挑战者-求解者-评判者协同进化框架，通过对抗性任务生成和基于生成方式的偏好对，实现了无需人工数据即可在可验证和不可验证领域中的自我进化。

    

    arXiv:2608.26582v1 公告类型：交叉 摘要：自我进化语言模型最近成为通往超级智能的一条有前景的路径，其优势在于减少人类监督成本。尽管在可验证领域已取得显著进展，但自我进化在不可验证领域仍研究不足。我们提出了从零数据出发的评判者协同适应（J-Zero），这是一个统一的挑战者-求解者-评判者协同进化框架，支持在两种领域中的自我改进。挑战者和求解者通过对抗性互动协同进化：挑战者生成越来越难的任务，而求解者学习产生更高质量的响应。与此同时，评判者通过使用偏好对进行协同适应，这些偏好对的顺序是预先已知的，基于每个响应的生成方式，即求解者的答案优于挑战者的答案，以及其分解再组合的答案优于其一次性答案，而非基于评判者自身的评分。

    arXiv:2608.26582v1 Announce Type: cross  Abstract: Self-evolving language models have recently emerged as a promising path toward superintelligence, with the advantage of reducing the cost of human supervision. While considerable progress has been made in verifiable domains, self-evolution in unverifiable domains remains substantially less explored. We propose Judge co-adaptation from Zero data (J-Zero), a unified Challenger--Solver--Judge co-evolution framework that supports self-improvement across both domains. The Challenger and Solver co-evolve through an adversarial interaction: the Challenger generates increasingly difficult tasks, while the Solver learns to produce higher-quality responses to them. In parallel, the Judge co-adapts using preference pairs whose ordering is known in advance from how each response was produced, i.e., the Solver's answer over the Challenger's, and its decomposed-and-recombined answer over its one-shot answer, rather than from the Judge's own scores. 
    
[^76]: 视觉信息引导的扩散多模态大语言模型并行解码方法

    Visual Information-Guided Parallel Decoding for Diffusion Multimodal Large Language Models

    [https://arxiv.org/abs/2608.26580](https://arxiv.org/abs/2608.26580)

    提出一种视觉信息引导采样器，通过利用标记对图像标记的注意力优先级和约束惩罚，改进了扩散多模态大语言模型的解码质量。

    

    arXiv:2608.26580v1 公告类型：交叉 摘要：扩散多模态大语言模型（dMLLMs）最近作为一种新的多模态生成解码范式出现。从一个完全掩码的序列开始，dMLLMs通过逐步解掩码剩余掩码位置的一个子集来逐步解码序列。由于选定的标记作为后续步骤的预测上下文，决定解码哪些标记对最终输出的质量至关重要。最常见的策略基于确定性度量优先选择标记，这种度量倾向于训练数据中频繁出现的标记。最近的方法根据标记对后续预测的影响进行排序，但没有显式考虑输入图像。我们提出了视觉信息引导采样器（VIG-Sampler），它基于标记对图像标记的注意力来优先选择标记。我们进一步施加了一个约束，惩罚那些图像注意力分布异常的候选标记。

    arXiv:2608.26580v1 Announce Type: cross  Abstract: Diffusion multimodal large language models (dMLLMs) have recently emerged as a new decoding paradigm for multimodal generation. Starting from a fully masked sequence, dMLLMs progressively decode the sequence by unmasking a subset of the remaining masked positions at each step. Since the selected tokens serve as the prediction context for subsequent steps, deciding which tokens to decode is crucial to the quality of the final output. The most common strategy prioritizes tokens based on a certainty measure that tends to favor tokens frequently observed in the training data. Recent approaches instead order tokens according to their influence on subsequent predictions, but do not explicitly account for the input image. We propose the Visual Information-Guided Sampler (VIG-Sampler), which prioritizes tokens based on their attention to image tokens. We further impose a constraint that penalizes candidate tokens whose image-attention distribu
    
[^77]: 双重麻烦：双语预训练在共享语言表示中留下语言条件效应

    Double Trouble: Bilingual Pretraining Leaves Language-Conditioned Effects in Shared-Language Representations

    [https://arxiv.org/abs/2608.26576](https://arxiv.org/abs/2608.26576)

    双语预训练在深层隐藏状态中引入语言条件效应，即使在对齐嵌入空间后，共享词汇的表示在预测层面仍不一致，源于上下文处理而非输入表示。

    

    arXiv:2608.26576v1 公告类型：新 摘要：当研究人员比较多语言模型以进行探针分析、可解释性或跨语言迁移时，他们通常对齐嵌入空间，并假设共享语言表示是可比较的。我们表明，对于仅解码器模型，这一假设可能为时过早。我们在八种类型多样的语言上预训练了配对的3.1亿参数模型（一个仅英语，一个双语），分别控制了英语暴露量、总计算量和文档重叠度。在对齐共享英语词汇后，我们测试了保留词汇，发现词嵌入在对齐后看起来相似，但模型用于预测的更深层隐藏状态却不相似。这一差距在全部八种语言中都存在，并且在对文档重叠度和替代对齐方法进行控制后依然成立。这种隐藏状态不匹配在中间Transformer层中增长，表明它源于上下文处理，而非输入表示。

    arXiv:2608.26576v1 Announce Type: new  Abstract: When researchers compare multilingual models for probing, interpretability, or cross-lingual transfer, they often align embedding spaces and assume that shared-language representations are comparable. We show that this assumption can be premature for decoder-only models. We pretrain paired 310M-parameter models (one English-only, one bilingual) across eight typologically diverse languages, separately controlling for English exposure, total compute, and document overlap. After aligning on shared English vocabulary, we test held-out words and find that token embeddings look similar after alignment, but the deeper hidden states that the model uses for prediction do not. This gap holds for all eight languages and survives controls for document overlap and alternative alignment methods. This hidden-state mismatch grows through middle transformer layers, suggesting that it arises from contextual processing rather than the input representations
    
[^78]: 依赖感知的可撤销解码用于高效扩散大语言模型推理

    Dependency-Aware Revocable Decoding for Efficient Diffusion Large Language Model Inference

    [https://arxiv.org/abs/2608.26574](https://arxiv.org/abs/2608.26574)

    本文提出依赖感知的可撤销解码（DARD），通过分离标记状态并排除不可靠标记来保护验证上下文，从而在不牺牲生成质量的前提下提升扩散大语言模型的并行解码效率。

    

    扩散大语言模型（dLLMs）通过迭代去噪并行解码多个标记，为自回归生成提供了一种有前景的替代方案。然而，增加解码并行度往往会降低生成质量，因为早期错误可能污染后续上下文。可撤销解码通过重新评估已解码标记并重新掩盖不可靠标记来缓解这一问题，但现有方法忽略了不可靠标记也可能破坏验证上下文本身。我们识别出这一失败模式，并提出依赖感知的可撤销解码（DARD），这是一种无需训练的框架，将标记分为掩盖、候选和未掩盖三种状态。DARD使用排除较少可靠标记的选择性上下文来验证候选标记，并自适应地调节其对后续解码的影响。在3个开源dLLMs上的12个文本和多模态基准测试中的实验表明，DARD持续改善了性能。

    arXiv:2608.26574v1 Announce Type: new  Abstract: Diffusion large language models (dLLMs) offer a promising alternative to autoregressive generation by decoding multiple tokens in parallel through iterative denoising. However, increasing decoding parallelism often degrades generation quality, as early errors can contaminate later contexts. Revocable decoding mitigates this issue by re-evaluating decoded tokens and remasking unreliable ones, but existing methods overlook that unreliable tokens may also corrupt the verification context itself. We identify this failure mode and propose Dependency-Aware Revocable Decoding (DARD), a training-free framework that separates tokens into masked, candidate, and unmasked states. DARD verifies candidate tokens using a selective context that excludes less reliable tokens and adaptively regulates their influence on subsequent decoding. Experiments across 12 textual and multimodal benchmarks on 3 open-source dLLMs show that DARD consistently improves t
    
[^79]: 技能作为智能体语言模型的预训练数据

    SPT: Skills as Pre-Training Data for Agentic Language Models

    [https://arxiv.org/abs/2608.26563](https://arxiv.org/abs/2608.26563)

    本文提出技能预训练（SPT）方法，通过将公开技能包作为中间训练数据，结合引用插入策略，显著提升智能体语言模型的工具使用能力。

    

    arXiv:2608.26563v1 公告类型：新 摘要：智能体（工具使用）语言模型主要在后训练阶段基于工具调用轨迹和智能体行为序列进行训练。这些数据提供了直接的行为监督，但生成它们需要任务环境、执行和验证，这使得广泛覆盖工具和任务的成本高昂。公开可用的技能提供了另一种训练数据来源：它们编码了可复用的工具语义和工作流程，但通常仅作为推理时的上下文使用。我们引入了技能预训练（SPT），这是一种中间训练方法，将因果语言建模应用于SkillCorpus——一个包含公开多文件技能包的集合，可选地混合通用数据。为了保持每个包内文件之间的关系，我们还引入了引用插入（Reference Insert），这是一种引用感知的组装策略，将支持文件放置在主要指令中提及它们的位置附近。跨多个模型规模和后训练方案的实验表明，SPT能够...

    arXiv:2608.26563v1 Announce Type: new  Abstract: Agentic (tool-using) language models are mainly trained on tool-call traces and agent trajectories during post-training. These data provide direct behavioral supervision, but producing them requires task environments, execution, and verification, making broad tool and task coverage expensive. Publicly available skills offer another source of training data: they encode reusable tool semantics and workflows but are typically used only as inference-time context. We introduce Skill Pre-Training (SPT), a mid-training method that applies causal language modeling to SkillCorpus, a collection of public multi-file skill packages, optionally mixed with general data. To preserve relations among files within each package, we also introduce Reference Insert, a reference-aware assembly strategy that places supporting files near their mentions in the primary instruction. Experiments across multiple model scales and post-training recipes show that SPT c
    
[^80]: SPEAR：通过强化学习中的序列符号对齐提炼领域自适应推理骨架

    SPEAR: Distilling Domain-Adaptive Reasoning Skeletons via Sequential Symbolic Alignment in Reinforcement Learning

    [https://arxiv.org/abs/2608.26550](https://arxiv.org/abs/2608.26550)

    SPEAR提出了一种无需训练、即插即用的过程奖励方法，通过符号里程碑和最长公共子序列对齐，在强化学习蒸馏中提供密集且逻辑一致的奖励，避免了昂贵的外部神经过程奖励模型。

    

    基于强化学习的知识蒸馏有潜力将复杂推理从教师模型转移到学生模型，但目前面临一个关键困境：研究者必须在稀疏的基于结果的奖励（其逻辑指导不足）和昂贵的神经过程奖励模型（用于密集信号）之间做出选择。我们通过引入SPEAR（符号过程评估与对齐奖励）解决了这一问题，这是一种无需训练且即插即用的过程奖励方法，用于序列级策略蒸馏。SPEAR将自然语言推理轨迹投影到领域自适应的符号里程碑中，为过程级推理对齐提供了高效代理。通过利用最长公共子序列（LCS）来将学生探索与教师里程碑对齐，SPEAR提供了密集且顺序感知的奖励信号，在不需外部神经验证器的情况下强制逻辑一致性。我们的实验...

    arXiv:2608.26550v1 Announce Type: new  Abstract: Reinforcement learning-based knowledge distillation has the potential to transfer complex reasoning from teacher to student models, yet it currently faces a critical dilemma: researchers must choose between sparse outcome-based rewards, which provide insufficient logical guidance, or expensive neural Process Reward Models (PRMs) for dense signals. We resolve this by introducing SPEAR (Symbolic Process Evaluation and Alignment Reward), a training-free and plug-and-play process reward method for sequence-level on-policy distillation. SPEAR projects natural-language reasoning traces into domain-adaptive symbolic milestones, providing an efficient proxy for process-level reasoning alignment. By utilizing the longest common subsequence (LCS) to align student explorations with teacher milestones, SPEAR provides a dense, order-aware reward signal that enforces logical consistency without the need for an external neural verifier. Our experiments
    
[^81]: DuMateBench：评估复杂真实工作流中的自主代理

    DuMateBench: Evaluating Autonomous Agents in Complex Real-World Workflows

    [https://arxiv.org/abs/2608.26546](https://arxiv.org/abs/2608.26546)

    本文提出了DuMateBench，一个基于真实用户会话重建的基准，通过200个多能力任务和三种环境复杂性挑战，填补了代理评估在现实工作流中的空白。

    

    自主代理越来越多地被用于完成现实环境中复杂的多工具工作流。然而，现有的基准测试通常按应用或能力分离任务，并在比实际实践更清洁、更稳定的环境中评估代理。我们引入了DuMateBench，这是一个从大规模生产代理平台收集的匿名化和隐私筛选用户会话中重建的真实会话基准。每个任务保留了相关的解决前交互历史、持久配置和工作区状态，并通过人工验证进行确认。由此产生的基准包含200个任务，涵盖8个广泛场景和17个细粒度能力类别，大多数任务需要多种能力的协调。我们在隔离的Docker容器中执行这些任务，并注入三种形式的现实环境复杂性：信息不足、不稳定和动态变化。

    arXiv:2608.26546v1 Announce Type: new  Abstract: Autonomous agents are increasingly adopted to complete complex, multi-tool workflows in real-world settings. However, existing benchmarks typically separate tasks by application or capability and evaluate agents in environments that are cleaner and more stable than those encountered in practice. We introduce DuMateBench, a real-session benchmark reconstructed from anonymized and privacy-screened user sessions collected from a large-scale production agent platform. Each task preserves the relevant pre-solution interaction history, persistent configurations, and workspace state, and is then validated through human verification. The resulting benchmark comprises 200 tasks spanning 8 broad scenarios and 17 fine-grained capability categories, with most tasks requiring multiple capability coordination. We execute these tasks in isolated Docker containers injected with three forms of real-world environmental complexity: Insufficient, Unstable, 
    
[^82]: 多专家共形风险控制在开放式对话中成对LLM评判中的应用

    Multi-Expert Conformal Risk Control for Pairwise LLM Judging in Open-Ended Dialogue

    [https://arxiv.org/abs/2608.26529](https://arxiv.org/abs/2608.26529)

    本文提出多专家共形风险控制方法，并引入MC3算法，通过初始阈值比率校准异质专家评分尺度，实现开放式对话中成对LLM评判的风险控制与覆盖率优化。

    

    本文探讨了多专家共形风险控制（CRC）算法在开放式对话中进行成对LLM-as-a-Judge评估中的应用。我们的核心见解是，多专家聚合为CRC提供了互补性的解决方案：CRC通过弃权在决策阈值处控制风险，而聚合则在源头净化评分函数。基于此，我们首先设计了两种多专家CRC方法：得分平均和决策投票，它们分别在得分和决策层面进行聚合。虽然这两种策略在同质专家小组上优于单专家方法，但在异质LLM评判者上，它们保持风险有效性但仅恢复有限的覆盖率，因为统一阈值无法匹配各专家不同的评分尺度。为解决此问题，我们进一步提出了边际校准共形共识（MC3）：它通过初始阈值比率捕获各专家特有的尺度，同时联合调整。

    arXiv:2608.26529v1 Announce Type: new  Abstract: In this paper, we explore multi-expert Conformal Risk Control (CRC) algorithms for pairwise LLM-as-a-Judge evaluation in open-ended dialogue. Our core insight is that multi-expert aggregation offers a complementary remedy to CRC: whereas CRC controls risk at the decision threshold through abstention, aggregation sanitizes the scoring function at its source. Guided by this, we first design two multi-expert CRC methods: Score Averaging and Decision Voting, which aggregate at the score and decision levels, respectively. While both strategies outperform single-expert methods on homogeneous expert panels, on heterogeneous LLM judges they remain risk-valid but recover only limited coverage, because a uniform threshold cannot match the experts' distinct scoring scales. To resolve this issue, we further propose Marginal-Calibrated Conformal Consensus (MC3): it captures distinct per-expert scales via initial threshold ratios, while jointly tuning
    
[^83]: 抑制奉承行为可能损害理性更新：反奉承应保留更新能力

    Sycophancy Suppression Can Impair Rational Updating: Anti-Sycophancy Should Preserve the Ability to Update

    [https://arxiv.org/abs/2608.26511](https://arxiv.org/abs/2608.26511)

    反奉承方法在减少无依据屈服时可能损害模型的理性更新能力，两者之间存在权衡，需在设计中保留更新能力。

    

    大型语言模型常常表现出奉承行为，当用户提出质疑时，模型会修改其答案以迎合用户。然而，这种答案翻转可能源于不同的原因。一种可能性是模型仅仅为了满足用户而顺应其反馈。另一种可能是反馈确实包含有用的证据，促使模型以理性的方式更新其答案。我们将这两种情况区分为“无依据屈服”和“理性更新”。先前的研究主要侧重于抑制“无依据屈服”，而忽视了其对“理性更新”的影响。我们通过一个两轮评估框架来弥补这一空白，该框架分别衡量这两种行为。在代表性的训练时和推理时干预措施中，我们发现反奉承方法常常面临一种权衡，即减少“无依据屈服”可能会牺牲“理性更新”，反之亦然，即使这两个目标被联合优化也是如此。

    arXiv:2608.26511v1 Announce Type: new  Abstract: Large language models often exhibit sycophancy, revising their answers to align with users when users push back. Such answer flips, however, can arise from different causes. One possibility is that the model simply aligns with the user's feedback in order to satisfy them. Another is that the feedback genuinely contains useful evidence, prompting the model to update its answer in a rational way. We distinguish them as Unsupported-Yielding and Rational-Updating. Prior work focuses primarily on suppressing Unsupported-Yielding, while overlooking its effect on Rational-Updating. We address this gap with a two-turn evaluation framework that measures the two behaviors separately. Across representative training-time and inference-time interventions, we find that anti-sycophancy methods often encounter a trade-off in which reducing Unsupported-Yielding can sacrifice Rational-Updating, and vice versa, even when the two objectives are optimized jo
    
[^84]: 一个后缀破解所有：针对合并模型族的盆地感知越狱攻击

    A Single Suffix to Break Them All: Basin-Aware Jailbreaks for Merged Model Families

    [https://arxiv.org/abs/2608.26506](https://arxiv.org/abs/2608.26506)

    我们发现了模型合并中源于预训练基础模型的隐蔽越狱风险，并提出盆地感知越狱（BAJ）方法，通过最小-最大优化生成跨合并模型族可迁移的对抗后缀。

    

    模型合并技术能够在不进行额外训练的情况下组合多个微调模型，但其安全性影响仍未被充分理解。以往研究主要将合并风险归因于不安全的组成模型，并隐含假设合并各自对齐的模型能保持安全性。相反，我们表明模型合并揭示了一种先前被忽视的越狱风险，该风险根源于预训练基础模型，即使所有组成模型都经过单独的安全对齐。基于这一观察，我们研究了一种新的威胁场景，其中攻击者构造的越狱提示能够泛化到共享同一预训练骨干的合并模型上，而无需访问确切的合并系数或组成检查点。为利用这一现象，我们提出了**盆地感知越狱（BAJ）**，它将越狱生成表述为合并空间上的最小-最大优化，以产生可迁移的对抗性后缀。

    arXiv:2608.26506v1 Announce Type: cross  Abstract: Model merging enables combining multiple fine-tuned models without additional training, but its safety implications remain poorly understood. Prior work primarily attributes merging risks to unsafe constituent models, implicitly assuming that merging individually aligned models preserves safety. In contrast, we show that model merging reveals a previously overlooked jailbreak risk rooted in the pretrained foundation model, even when all constituent models are individually safety-aligned. Motivated by this observation, we study a new threat setting where an attacker constructs jailbreak prompts that generalize across merged models sharing the same pretrained backbone, without access to the exact merging coefficients or constituent checkpoints. To exploit this phenomenon, we propose \textbf{Basin-Aware Jailbreak (BAJ)}, which formulates jailbreak generation as a min--max optimization over the merging space to produce transferable adversa
    
[^85]: 基于账本控制的零样本自编排提升LLM编码性能

    Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance

    [https://arxiv.org/abs/2608.26480](https://arxiv.org/abs/2608.26480)

    本文证明，在不进行训练或基准调优的情况下，基于账本控制的管理器-工作器脚手架能显著提升某些LLM的编码性能，但效果因模型而异，并非普遍适用。

    

    多智能体大语言模型系统被广泛报道能超越单模型基线，但证据不一，且比较通常存在混淆：流程同时改变令牌预算、工具调用和提示，因此总体增益很少能揭示真正有效的因素。我们研究了在共享文件系统工作区中引入管理器-工作器脚手架的效果，无需训练且无需针对基准进行调优，与同一模型单次回答进行对比。在九个模型上——五个开放权重模型，参数范围从9B到约2.8T，以及四个前沿封闭模型——针对LiveCodeBench最新的100个困难问题，脚手架的好处是真实但有条件的：对某些模型效果显著且统计显著（如Qwen3.8-27B提升23.4，GPT-5.6-Luna提升10.6，GPT-5.6-Terra提升8.0，各基于五次配对运行；Kimi-K3提升30.4，Minimax-M3提升11.0，基于五次配对运行且关闭推理，p值均小于10^-4，以及...）

    arXiv:2608.26480v1 Announce Type: cross  Abstract: Multi-agent large language model systems are widely reported to beat single-model baselines, but the evidence is mixed, and comparisons are usually confounded: pipelines change token budgets, tool calls, and prompts simultaneously, so an aggregate gain rarely reveals what actually helped. We investigate the effect of introducing the manager-worker scaffold over a shared filesystem workspace, with no training and no per-benchmark tuning, measured against the same model answering in a single pass. Across nine models -- five open-weight, spanning 9B to ~2.8T parameters, and four frontier closed models -- on the 100 latest hard LiveCodeBench problems, the scaffold's benefit is real but conditional: large and statistically significant for some (Qwen3.8-27B +23.4, GPT-5.6-Luna +10.6 and GPT-5.6-Terra +8.0, each over five paired passes; Kimi-K3 +30.4 and Minimax-M3 +11.0 over five paired passes with reasoning off, both at $p < 10^{-4}$, and +
    
[^86]: 基于范畴论框架中结构识别的组合泛化

    Compositional Generalization via Structural Identification in a Category-Theoretic Framework

    [https://arxiv.org/abs/2608.26465](https://arxiv.org/abs/2608.26465)

    本文提出一种无需训练模型、基于范畴论函子和Kan扩展的方法，通过分析训练语料在结构识别下的可接受性，来诊断组合泛化的支持与失败模式。

    

    组合泛化通常通过模型准确率来评估。我们转而询问，哪些结构或词汇识别使得训练中观察到的结构能够支持COGS测试集中的未见示例。句子被表示为从句法地址到词汇标记的函子，选择性塌缩诱导出Kan扩展，从而传播观察到的关联。在21种COGS泛化类型中，可接受性遵循不同的识别轮廓，而残余失败则揭示了不支持的结构模板。这些数据侧诊断刻画了在特定识别下训练语料库所许可的内容，而无需训练预测模型。

    arXiv:2608.26465v1 Announce Type: new  Abstract: Compositional generalization is usually evaluated through model accuracy. We instead ask which structural or lexical identifications make held-out COGS examples admissible from the structures observed in training. Sentences are represented as functors from syntactic addresses to lexical tokens, and selective collapses induce Kan extensions that propagate observed associations. Across 21 COGS generalization types, admissibility follows distinct identification profiles, while residual failures separate unsupported structural templates. These data-side diagnoses characterize what the training corpus licenses under specified identifications, without training a predictive model.
    
[^87]: 差异挖掘：逻辑差异揭示微调目标

    Diff Mining: Logit Differences Reveal Finetuning Objectives

    [https://arxiv.org/abs/2608.26462](https://arxiv.org/abs/2608.26462)

    本文提出了一种名为“差异挖掘”的简单框架，通过比较微调模型与基础模型的输出逻辑值来揭示微调目标，无需访问模型内部信息，并能有效识别微调过程中学到的行为。

    

    摘要：arXiv:2608.26462v1 公告类型：交叉 摘要：微调已成为优化语言模型现有行为和诱导新行为的黄金标准，但在此过程中具体出现了哪些行为往往仍不明确。随着模型能力不断增强，更好地理解微调变得越来越重要，尤其是在微调过程中可能出现不需要的行为。在本文中，我们引入了差异挖掘（Diff Mining），一个简单而有效的框架，通过比较微调模型与其基础模型的逻辑值来识别微调模型学到了什么。差异挖掘有效地表征了微调模型中被放大的显著标记，作为其训练的“指纹”——即使在微调领域无关的文本上也是如此。与许多需要模型内部信息的现有模型差异方法不同，差异挖掘仅需访问输出逻辑值，并能扩展到大型模型。该框架由两个模块化阶段组成：（i）提取每个上下文的逻辑值差异

    arXiv:2608.26462v1 Announce Type: cross  Abstract: Finetuning has become the gold standard for refining existing behaviors and inducing new ones in language models, yet it often remains unclear exactly which behaviors emerge during this process. As models grow ever more capable, understanding finetuning better becomes increasingly important, particularly since unwanted behaviors may arise during finetuning. In this paper, we introduce Diff Mining, a simple yet effective framework for identifying what a finetuned model has learned by comparing its logits to those of its base model. Diff Mining effectively surfaces salient tokens that are amplified in the finetuned model, serving as a fingerprint of its training -- even on text unrelated to the finetuning domain. Unlike many existing model diffing methods which require model internals, Diff Mining only needs access to output logits and scales to large models. The framework consists of two modular stages: (i) extracting per-context logit 
    
[^88]: 元音符号不是字母：多语言分词器生育率上的预分词天花板

    Vowel Signs Are Not Letters: A Pre-tokenization Ceiling on Multilingual Tokenizer Fertility

    [https://arxiv.org/abs/2608.26449](https://arxiv.org/abs/2608.26449)

    本文发现，基于GPT-2单词正则的预分词器在元音附标文字中每个元音符号处强制拆分单词，导致分词生育率显著上升，并提出了一个无需训练的下界来量化此影响。

    

    arXiv:2608.26449v1 公告类型：新 摘要：使用HuggingFace ByteLevel预分词器的字节级BPE分词器继承了GPT-2的单词正则表达式，其中单词定义为\p{L}+，即一个或多个Unicode字母。在元音附标文字（abugida）中，元音以组合标记形式书写；因此，该模式在每个元音符号处拆分每个单词。由于BPE仅在预分词内部进行合并，这些拆分在训练过程中持续存在，无论词汇量大小或语料库组成如何。我们将此效应形式化为一个无需训练的分词生育率下界。在来自平行语料库的26种语言中，所有17种元音附标文字均受到影响，范围从1.47倍（藏文）到9.02倍（泰文），而拉丁文、西里尔文、韩文和汉字恰好为1.00倍。对于5种语言，仅在字符类上不同的匹配分词器对，其得分在预测下限的2.2%以内，尼泊尔语中每个词的分词数为4.78对1.58。当训练语料库中尼泊尔语份额从5%扫到95%时，受损分词器仍然保持该缺陷。

    arXiv:2608.26449v1 Announce Type: new  Abstract: Byte-level BPE tokenizers that use the HuggingFace ByteLevel pre-tokenizer inherit GPT-2's word regex, where a word is defined as \p{L}+, one or more Unicode letters. In abugida scripts, vowels are written as combining marks; this pattern therefore splits each word at every vowel sign. Since BPE merges only within a pre-token, those splits persist through training regardless of vocabulary size or corpus composition. We formalise this effect as a training-free lower bound on fertility. Across 26 languages from a parallel corpus, every one of the 17 abugidas is affected, ranging from 1.47x (Tibetan) to 9.02x (Thai), whereas Latin, Cyrillic, Hangul, and Han show exactly 1.00x. For 5 languages, matched tokenizer pairs that differ only in this character class fall within 2.2% of the predicted floor, scoring 4.78 versus 1.58 tokens per word on Nepali. When the Nepali share of the training corpus is swept from 5% to 95%, the broken tokenizer ba
    
[^89]: 不要过度思考，也不要思考不足：迈向代理型人工智能中的自适应推理

    Don't Overthink, Don't Underthink: Toward Adaptive Reasoning in Agentic AI

    [https://arxiv.org/abs/2608.26442](https://arxiv.org/abs/2608.26442)

    本文指出代理型AI系统需要自适应推理机制，以动态调整推理资源，避免过度或不足推理带来的效率与性能问题。

    

    arXiv:2608.26442v1 公告类型：新 摘要：大型语言模型（LLM）的最新进展表明，增加推理时的计算量可以提高复杂任务的性能。然而，许多现有方法依赖于固定或预分配的推理控制，例如固定的令牌预算、预先执行的难度估计或激活空间干预，并且通常在独立的推理基准上进行评估，而非完整的代理工作流。这些假设在代理型人工智能系统中可能不成立，因为在这些系统中，推理需求通过规划、工具使用、记忆检索和代理间交互而动态演变。因此，推理可能变得过度或不足，导致不必要的计算、增加延迟、规划偏移、过度使用工具或不完整的解决方案。我们认为，下一代代理型人工智能的主要挑战不仅仅是语言模型应执行多少推理，而是应如何分配推理。

    arXiv:2608.26442v1 Announce Type: new  Abstract: Recent advances in Large Language Models (LLMs) have shown that increased inference-time reasoning can improve performance on complex tasks. However, many existing approaches rely on fixed or preallocated reasoning controls, such as fixed token budgets, pre-execution difficulty estimates, or activation-space interventions, and are often evaluated on standalone reasoning benchmarks rather than full agentic workflows. These assumptions may not hold in agentic AI systems, where reasoning requirements evolve dynamically through planning, tool use, memory retrieval, and agent-to-agent interactions. Consequently, reasoning can become either excessive or insufficient, resulting in unnecessary computation, increased latency, planning drift, excessive tool use, or incomplete solutions. We argue that a major challenge for next-generation agentic AI is not merely how much reasoning a language model should perform, but how it should allocate reasoni
    
[^90]: AfriSwitch：野外非洲语码转换语音识别基准

    AfriSwitch: A Benchmark for In-the-Wild African Code-Switched Speech Recognition

    [https://arxiv.org/abs/2608.26434](https://arxiv.org/abs/2608.26434)

    该论文提出了AfriSwitch，一个61.36小时、覆盖16种非洲语言的野外语码转换语音基准，揭示了多语言ASR系统在此场景下的高错误率，并强调现有模型缺乏对非洲语码转换的适应能力。

    

    arXiv:2608.26434v1 公告类型：新  摘要：语码转换在双语非洲对话中普遍存在，但大多数自动语音识别（ASR）系统假设单语输入，并在精心策划的单语基准上进行评估。我们提出了AfriSwitch，这是一个61.36小时的人工转写基准，涵盖16种非洲语言及其变体的野外语码转换语音，并附带切换级英语跨度标签、每话语语码混合指数（CMI）和切换点计数。语料库统计显示，不同非洲语言的混合行为在两条大致独立的轴上差异显著：说话者切换的频率以及混合的平衡程度。没有单一的标量能捕捉一种语言的语码转换程度。对五种开放和商业多语言ASR系统进行零样本基准测试，其词错误率远高于同一语言的已发表单语结果，最佳系统平均词错误率为35.93%，且没有任何系统在任何语言上低于24%。针对非洲的定向训练，而非...

    arXiv:2608.26434v1 Announce Type: new  Abstract: Code-switching is pervasive in bilingual African conversation, yet most ASR systems assume monolingual input and are evaluated on curated monolingual benchmarks. We present AfriSwitch, a 61.36-hour human-transcribed benchmark of in-the-wild code-switched speech spanning 16 African languages and language varieties, released with switch-level English span tags, perutterance Code-Mixing Index (CMI), and switch-point counts. Corpus statistics show that mixing behaviour varies widely across African languages along two largely independent axes: how often speakers alternate, and how balanced the mixture is. No single scalar captures how code-switched a language is. Benchmarking five open and commercial multilingual ASR systems zero-shot yields word error rates far above published monolingual figures for the same languages, with the best system averaging 35.93% WER and no system falling below 24% on any language. Africa-targeted training, not mo
    
[^91]: SpeechGym：一个通过强化学习训练语音代理的音频原生健身房

    SpeechGym: An Audio-Native Gym for Training Voice Agents via Reinforcement Learning

    [https://arxiv.org/abs/2608.26432](https://arxiv.org/abs/2608.26432)

    本文提出SpeechGym，一个音频原生的代理环境，通过强化学习端到端训练语音代理，解决了现有框架无法在音频模态中改进代理能力的问题。

    

    语音代理必须完全通过语音调用工具并进行多轮对话，然而主流范式却是在文本中训练它们。现有框架要么在专有语音API周围级联TTS和ASR，其中梯度无法流动且每次调用的成本使在线策略强化学习变得不可行，要么停留在文本层面：它们能测量语音代理却无法改进它们。我们提出了SpeechGym，一个音频原生的代理环境，其中两个全模态模型以原生音频进行对话，无需外部ASR或TTS，也没有API边界，在既定文本代理基准的未修改任务、工具和成功检查上运行，因此交互模态是唯一变量，且循环保持本地化并可端到端训练。音频代理能力并非源于音频理解。语音引入的失败是感知性而非推理性缺陷：代理选择了正确的工具和正确的参数槽，但填充...

    arXiv:2608.26432v1 Announce Type: cross  Abstract: Voice agents must call tools and hold multi-turn dialogue entirely through speech, yet the dominant paradigm trains them in text. Existing frameworks either cascade TTS and ASR around a proprietary voice API, where gradients cannot flow and per-call cost makes on-policy reinforcement learning prohibitive, or stay in text: they measure voice agents but cannot improve them. We present SpeechGym, an audio-native agentic environment in which two omni-modal models converse in native audio, with no external ASR or TTS and no API boundary, over the unmodified tasks, tools and success check of an established text agentic benchmark, so that the interaction modality is the only variable and the loop stays local and trainable end to end. Audio agentic capability does not follow from audio understanding. The failures speech introduces are perceptual rather than reasoning deficits: the agent picks the right tool and the right argument slot but fill
    
[^92]: 潜在诊断分类法：一种构建分类器并诊断其决策的框架，应用于提示注入检测

    The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection

    [https://arxiv.org/abs/2608.26423](https://arxiv.org/abs/2608.26423)

    本文提出了一种潜在诊断分类法框架，通过维度优化分类器、识别潜在支持向量和构建诊断分类法，为提示注入检测提供了一种可靠决策与风险标记的端到端指南。

    

    arXiv:2608.26423v1 公告类型：交叉 摘要：本文提出了一种框架，用于构建作为防护层的分类器，并开发一种互补的诊断方法，以识别分类器的哪些自信决策可以被信任。该框架，即潜在诊断分类法，包括：（i）构建一个维度优化的分类器，其中嵌入维度通过交叉验证性能经验性选择，而非预先固定；（ii）定位一个相对较小的潜在支持向量集（约占训练示例总数的29%），代表有影响力的提示，用于识别改变分类器预测标签的令牌；（iii）利用这些令牌及其相关的攻击幅度来构建诊断分类法。该诊断分类法为标记需要不同处理的提示提供了端到端的指南：安全地依赖分类器的决策；标记启发式偏差和启发式过拟合。

    arXiv:2608.26423v1 Announce Type: cross  Abstract: This paper proposes a framework for constructing a classifier as a safeguard layer, and for developing a complementary diagnostic that identifies which of the classifier's confident decisions can be trusted. This framework, the Latent Diagnostic Taxonomy, consists of (i) constructing a dimensionality-optimized classifier, in which the embedding dimensionality is empirically selected via cross-validated performance rather than fixed a priori, (ii) locating a relatively small set of latent support vectors (~ 29% of total training examples) representing influential prompts for identifying tokens that alter the classifier's predicted labels, and (iii) utilizing such tokens and their associated attack magnitudes for constructing a diagnostic taxonomy. This diagnostic taxonomy provides an end-to-end guideline for flagging prompts that require different treatments: rely Safely on the classifier's decision; flag Heuristic Bias and Heuristic Ov
    
[^93]: Case2Flow：通过多模态检索连接患者病例与指南流程图

    Case2Flow: Bridging Patient Cases and Guideline Flowcharts through Multimodal Retrieval

    [https://arxiv.org/abs/2608.26414](https://arxiv.org/abs/2608.26414)

    本文提出了Case2Flow任务和FlowAtlas语料库，用于从医学指南中为患者病例检索相关流程图，并揭示了现有多模态检索方法的系统性缺陷。

    

    arXiv:2608.26414v1 公告类型：新 摘要：医学指南编码了丰富的、基于证据的决策逻辑，但临床医生所需的特定决策工件在指南中难以定位，更不用说跨涵盖疑似疾病和治疗的指南了。尽管指南文本段落已支持端到端的问答，但流程图在决策支持中仍未被充分利用，尽管它们能够编码可操作的临床路径。因此，我们引入了Case2Flow，一个旨在从指南文档集合中为给定患者病例检索最相关指南流程图的任务。为支持该任务，我们构建了FlowAtlas，一个从2,080份医学指南中提取的202个流程图的精选语料库，以及一个合成1,911个对齐病例-流程图对的流水线。我们对多模态检索方法的评估揭示了系统性失败模式，包括过度依赖关键词以及由无信息性背景引发的虚假标记-补丁匹配。

    arXiv:2608.26414v1 Announce Type: new  Abstract: Medical guidelines encode rich, evidence-based decision logic, yet the specific decision artifact a clinician needs is hard to locate within a guideline, let alone across guidelines covering plausible diseases and treatments. While guideline passages have supported end-to-end question answering, flowcharts remain largely underused in decision support despite their ability to encode actionable clinical pathways. We therefore introduce Case2Flow, a task designed to retrieve the most relevant guideline flowchart for a given patient case from a collection of guideline documents. To support it, we construct FlowAtlas, a curated corpus of 202 flowcharts extracted from 2,080 medical guidelines, together with a pipeline that synthesises 1,911 aligned case-flowchart pairs. Our evaluation of multimodal retrieval methods reveals systematic failure modes, including overreliance on keywords and spurious token-patch matches induced by uninformative ba
    
[^94]: LowRankArena：一个基于SVD的LLM压缩标准化评估平台

    LowRankArena: A Standardized Evaluation Platform for SVD-Based LLM Compression

    [https://arxiv.org/abs/2608.26389](https://arxiv.org/abs/2608.26389)

    本文提出了LowRankArena，一个统一基准、压缩预算和推理测量的标准化平台，用于公平评估基于SVD的LLM压缩方法，并提供大规模可复现检查点。

    

    arXiv:2608.26389v1 公告类型：新  摘要：基于SVD的低秩压缩已成为降低大型语言模型（LLM）内存和计算成本的快速增长方向。然而，现有研究之间的有意义的比较仍然困难，因为先前的评估使用了不同的基准、不一致的压缩比例和多样化的设置，往往无法将低秩效果与辅助技术分离开来。因此，目前尚不清楚所报告的性能提升是反映了方法层面的改进，还是评估协议上的差异。这种缺乏可比性的问题凸显了对统一、可复现评估平台的需求。为解决这一问题，我们提出了LowRankArena，一个用于基于SVD的LLM压缩的标准化评估平台。LowRankArena统一了任务版本、均匀精度压缩预算、比较制度和推理测量，并提供了一个可复现的流水线，发布了超过3 TiB的压缩检查点。通过使用LowRankArena，我们的对齐...

    arXiv:2608.26389v1 Announce Type: new  Abstract: SVD-based low-rank compression has become a fast-growing direction for reducing the memory and computational cost of large language models (LLMs). However, meaningful comparison across existing studies remains difficult as prior evaluations use varied benchmarks, inconsistent ratios, and diverse setups, often failing to isolate low-rank effects from auxiliary techniques. As a result, it remains unclear whether reported gains reflect method-level improvements or differences in evaluation protocol. This lack of comparability highlights the need for a unified, reproducible evaluation platform. To address this problem, we present LowRankArena, a standardized evaluation platform for SVD-based LLM compression. LowRankArena unifies task versions, uniform-precision compression budgets, comparison regimes, and inference measurements, and provides a reproducible pipeline with over 3 TiB released compressed checkpoints. Using LowRankArena, our alig
    
[^95]: 语言模型中结构化知识与推理的共同演化

    Co-Evolving Structured Knowledge and Reasoning in Language Models

    [https://arxiv.org/abs/2608.26386](https://arxiv.org/abs/2608.26386)

    本文提出KBevo框架，通过端到端联合优化结构化知识库构建与推理，使推理结果直接反馈改进知识库质量，从而提升问答的准确性、可控性和组合推理能力。

    

    arXiv:2608.26386v1 公告类型：交叉 摘要：检索增强方法通过将语言模型基于外部知识来提高事实准确性，但检索非结构化文本常常引入无关上下文，并对检索信息控制有限。结构化知识库提供了一种更可控的替代方案，但其构建成本高昂，且推理时往往脆弱。为解决这些限制，我们提出KBevo：一个共同演化框架，联合学习构建结构化知识库并对其进行推理，以应对知识密集型问答任务。通过使用问答结果奖励对两个组件进行端到端优化，我们的方法使推理成功直接提升所构建知识库的质量。这导致更大、连接更紧密、答案可达性更高的知识结构，同时与标准检索基线相比，也改善了组合事实推理和可控性。

    arXiv:2608.26386v1 Announce Type: cross  Abstract: Retrieval-augmented methods improve factual accuracy by grounding language models in external knowledge, but retrieving over unstructured text often introduces irrelevant context and offers limited control over the retrieved information. Structured knowledge bases offer a more controllable alternative, yet they are expensive to construct and often brittle to reason over. To address these limitations, we propose KBevo: a co-evolving framework that jointly learns to construct a structured knowledge base and reason over it for knowledge-intensive question answering. By optimizing both components end-to-end with QA outcome rewards, our method enables reasoning success to directly improve the quality of the constructed knowledge base. This leads to larger, better-connected knowledge structures with higher answer reachability, while also improving compositional factual reasoning and controllability compared to standard retrieval baselines.
    
[^96]: 为什么RAG会幻觉：基于知识缺口金丝雀的检索增强生成系统惩罚感知评估

    Why RAGs Hallucinate: Penalty-Aware Evaluation of Retrieval-Augmented Generation Systems with Knowledge-Gap Canaries

    [https://arxiv.org/abs/2608.26385](https://arxiv.org/abs/2608.26385)

    本文提出了一种惩罚感知评估框架，通过引入不对称评分和知识缺口金丝雀，揭示了RAG系统在知识库无法支持时仍会猜测并产生幻觉的问题。

    

    基于体积的准确性奖励检索增强生成（RAG）系统进行猜测：一个回答所有问题的系统得分高于一个在知识库无法支持答案时拒绝回答的系统。基于Kalai等人（2025）的置信度目标分析，我们提出了一种针对部署RAG产品的惩罚感知评估框架，结合了（i）不对称评分（正确+1，错误-4，弃权0），（ii）知识缺口金丝雀——其答案可验证地不在知识库中的问题，因此任何回答都构成来自参数记忆的无根据生成，以及（iii）一个故障归因流程，将检索、生成和弃权策略故障分开。将该框架应用于三个商业RAG系统和一个无检索基线，在SimpleQA-Verified（1000个问题×3次重复，由跨家族三人评审小组盲评，一致性达98.9%）上，我们发现回答时的准确性接近……

    arXiv:2608.26385v1 Announce Type: cross  Abstract: Volume-based accuracy rewards retrieval-augmented generation (RAG) systems for guessing: a system that answers everything outscores one that declines when its knowledge base cannot support an answer. Building on the confidence-target analysis of Kalai et al. (2025), we present a penalty-aware evaluation framework for deployed RAG products, combining (i) asymmetric scoring (correct +1, wrong -4, abstain 0), (ii) knowledge-gap canaries, questions whose answers are verifiably absent from the knowledge base, so that any answer constitutes ungrounded generation from parametric memory, and (iii) a failure-attribution pipeline that separates retrieval, generation, and abstention-policy failures. Applying the framework to three commercial RAG systems and a no-retrieval baseline on SimpleQA-Verified (1,000 questions x 3 repeats, graded blind by a cross-family three-judge panel with 98.9% unanimity), we find that accuracy when answering is close
    
[^97]: 评估证据感知检索在RAG中的下游实用性

    Assessing the Downstream Utility of Evidence-Aware Retrieval in RAG

    [https://arxiv.org/abs/2608.26379](https://arxiv.org/abs/2608.26379)

    本研究评估证据感知检索信号在RAG中的下游实用性，发现其虽改变检索排名，但对训练、系统选择和答案质量预测的益处并不一致。

    

    arXiv:2608.26379v1 公告类型：交叉 摘要：检索增强生成（RAG）的检索评估日益设计为围绕检索到的段落是否包含能够支持生成的证据，而不仅仅是主题相关性。我们研究这种与下游证据需求的更紧密对齐是否也使检索评估对其所构建的决策更有用。在五个检索基准和一个端到端的TREC RAG 2025设置中，我们检查了一个答案支持信号在四个角色中的作用：比较检索器、指导检索训练和系统选择、预测下游答案质量，以及过滤提供给生成器的证据。该信号改变了检索排名，但其下游价值并不统一。它不能可靠地改进检索器训练；使用它进行系统选择的益处取决于生成器被指示如何使用检索到的证据；基于它的检索分数并不能稳健地预测答案质量。

    arXiv:2608.26379v1 Announce Type: cross  Abstract: Retrieval evaluation for retrieval-augmented generation (RAG) is increasingly designed around whether retrieved passages contain evidence that can support generation, rather than topical relevance alone. We study whether this closer alignment with downstream evidence needs also makes retrieval evaluation more useful for the decisions built from it.   Across five retrieval benchmarks and an end-to-end TREC RAG 2025 setting, we examine an answer-support signal in four roles: comparing retrievers, guiding retrieval training and system selection, predicting downstream answer quality, and filtering the evidence supplied to a generator. The signal changes retrieval rankings, but its downstream value is not uniform. It does not reliably improve retriever training; the benefit of using it for system selection depends on how the generator is instructed to use the retrieved evidence; and retrieval scores based on it do not robustly predict answe
    
[^98]: 面向高效扩散语言模型的生存引导长度控制

    Survival-Guided Length Control for Efficient Diffusion Language Models

    [https://arxiv.org/abs/2608.26374](https://arxiv.org/abs/2608.26374)

    提出一种无需训练的生存分析长度预测器，通过预测序列结束标记来动态调整扩散语言模型的解码长度，实现高达7倍的推理加速且不损失准确性。

    

    arXiv:2608.26374v1 公告类型：新 摘要：扩散语言模型（DLMs）通过迭代去噪掩码序列来生成文本，但标准解码要么固定序列长度，要么依赖临时停止规则，这往往导致不必要的去噪步骤。我们将长度选择重新定义为结束标记上的离散时间生存问题，并提出一种即插即用、无需训练的长度预测器，可添加到任何现有DLM中。在推理和代码生成基准测试中，生存引导长度解码将推理速度提升高达7倍，同时保持任务准确性。我们进一步发现，即使在同一数据集内，预测长度也差异很大，这使得模型性能对所选长度敏感。

    arXiv:2608.26374v1 Announce Type: new  Abstract: Diffusion language models (DLMs) generate text by iteratively denoising masked sequences, but standard decoding either fixes the sequence length or relies on ad hoc stopping rules, often leading to unnecessary denoising steps. We recast length selection as a discrete-time survival problem over the end-of-sequence token and propose a plug-in, training-free length predictor that can be added to any existing DLM. Across reasoning and code-generation benchmarks, survival-guided length decoding speeds up inference by up to 7 times while preserving task accuracy. We further find that predicted lengths vary widely even within the same dataset, making model performance sensitive to the chosen length.
    
[^99]: 知识验证的LLM代理在利益冲突下涌现的欺骗行为

    Knowledge-Verified Emergent Deception in LLM Agents Under Conflicting Incentives

    [https://arxiv.org/abs/2608.26372](https://arxiv.org/abs/2608.26372)

    该论文提出了KnownLieBench基准，通过知识验证方法，首次确认代理知晓用户权益，再评估利益冲突下其是否产生虚假陈述，从而区分真实欺骗与无知或幻觉。

    

    arXiv:2608.26372v1 公告类型：交叉 摘要：大型语言模型日益被部署为自主代理，代表公司为用户服务，这使它们处于用户与部署者利益可能冲突的环境中。当代理知道用户应得某项权益而其部署者更倾向于拒绝时，它是否会保持诚实？回答这个问题很困难，因为虚假陈述可能反映的是无知或幻觉，而非欺骗。为应对这一挑战，我们引入了KnownLieBench，一个知识验证的基准测试，首先通过中性探针确认代理知道用户的权益，然后评估在引入拒绝该权益的激励后，它是否做出虚假陈述。具体来说，KnownLieBench涵盖八个客户服务领域和112个有依据的案例，进行与追踪信任的客户代理的多轮对话，并将仅由激励引发的欺骗与在明确指令下产生的欺骗区分开来。

    arXiv:2608.26372v1 Announce Type: cross  Abstract: Large language models are increasingly deployed as autonomous agents serving users on behalf of companies, placing them in settings where user and deployer interests can conflict. When an agent knows that a user is owed something its deployer would prefer to deny, does it remain honest? Answering this is difficult because false statements can reflect either ignorance or hallucination rather than deception. To address this challenge, we introduce KnownLieBench , a knowledge-verified benchmark that first confirms through a neutral probe that an agent knows a user's entitlement, and then evaluates whether it makes false claims once an incentive to deny that entitlement is introduced. Specifically, KnownLieBench covers eight customer-service domains and 112 grounded cases, conducts multi-round dialogues with a trust-tracking customer agent, and separates deception emerging from incentive alone from deception produced under explicit instruc
    
[^100]: 通过质心干预融合的跨语言表示学习

    Cross-lingual Representation Learning via Centroid Intervention Fusion

    [https://arxiv.org/abs/2608.26357](https://arxiv.org/abs/2608.26357)

    提出质心干预融合（CIF）框架，将多个跨语言干预投影融合为单一语言共享算子，显著提升低资源语言的跨语言迁移性能。

    

    大型语言模型（LLMs）在多语言表现上不均匀，尤其是在处理低资源语言时。推理时干预提供了一种轻量级方法，通过在前向传播过程中修改LLMs生成的隐藏状态来改善跨语言迁移，而无需更新模型参数。然而，现有的跨语言干预方法通常学习从源语言到目标语言的独立投影，这限制了可扩展性并阻碍了跨语言的知识共享。我们提出了质心干预融合（CIF），一种投影融合框架，将多个多语言干预投影整合为一个语言共享算子。在多语言常识推理、自然语言推理、事实编辑和机器翻译基准测试中，CIF在四个模型骨干上平均优于最强的先验成对干预基线，最高提升达+3.378个百分点。

    arXiv:2608.26357v1 Announce Type: new  Abstract: Large language models (LLMs) exhibit uneven multilingual performance, especially when dealing with low-resource languages. Inference-time intervention offers a lightweight way to improve cross-lingual transfer by modifying the hidden states produced by the LLMs during the forward pass, without updating model parameters. However, existing cross-lingual intervention methods typically learn separate projections from source to target languages, which limits scalability and prevents knowledge sharing across languages. We propose Centroid Intervention Fusion (CIF), a projection fusion framework that consolidates multiple multilingual intervention projections into a single language-shared operator. Across multilingual commonsense reasoning, natural language inference, factual editing, and machine translation benchmarks, CIF outperforms the strongest prior pairwise intervention baseline by up to +3.378 pp on average across four model backbones, 
    
[^101]: MoganColBERT-TR：面向土耳其语的晚期交互多向量检索模型

    MoganColBERT-TR: A Late-Interaction Multi-Vector Retrieval Model for Turkish

    [https://arxiv.org/abs/2608.26344](https://arxiv.org/abs/2608.26344)

    该论文提出了MoganColBERT-TR，一种基于晚期交互的多向量检索模型，通过从嵌入模型微调并利用交叉编码器蒸馏，实现了土耳其语的高效词元级检索。

    

    arXiv:2608.26344v1 公告类型：新 摘要：我们先前报告了从零开始训练的土耳其语ModernBERT编码器（MoganBERT-TR）以及在其基础上构建的单向量嵌入模型（MoganBERT-embed）。这项工作介绍了该系列中的第三个模型：MoganColBERT-TR，一个多向量检索模型，它不是将查询或文档压缩为单个向量，而是通过768->128投影在词元级别表示它，并使用MaxSim晚期交互进行评分。该模型并非从零开始训练：嵌入模型的编码器被用作起点，并通过单周期蒸馏阶段适应ColBERT目标。训练数据来自两个来源——从我们自己的预训练语料库中在字符域和句子边界处切出的标题-段落对，以及两个基于土耳其语问题的检索集——并从一个交叉编码器教师模型（bge-reranker-v2-m3）对一个正例和七个负例的软分数中蒸馏而来。

    arXiv:2608.26344v1 Announce Type: new  Abstract: We previously reported a ModernBERT encoder trained from scratch for Turkish (MoganBERT-TR) and a single-vector embedding model built on top of it (MoganBERT-embed). This work introduces the third model in that lineage: MoganColBERT-TR, a multi-vector retrieval model that, instead of compressing a query or a document into a single vector, represents it at the token level through a 768->128 projection and scores it with MaxSim late interaction. The model is not trained from scratch: the embedding model's encoder is taken as the starting point and adapted to the ColBERT objective with a single-epoch distillation phase. Training data is produced from two sources - title-to-passage pairs carved out of our own pretraining corpus in the character domain and at sentence boundaries, and two Turkish question-based retrieval sets - and is distilled from the soft scores of a cross-encoder teacher (bge-reranker-v2-m3) over one positive and seven min
    
[^102]: 神经符号PRM：通过结构化轨迹和符号验证增强科学推理

    Neuro-symbolic PRM: Enhancing Scientific Reasoning via Structured Traces and Symbolic Verification

    [https://arxiv.org/abs/2608.26329](https://arxiv.org/abs/2608.26329)

    本文提出了一种神经符号框架，通过将推理解耦为符号有效性和语义依据性两个维度，并采用确定性验证器和条件化过程奖励模型，有效解决了大型语言模型在科学推理中中间步骤缺乏上下文依据的问题。

    

    arXiv:2608.26329v1 公告类型：新 摘要：虽然工具增强的大型语言模型在定量STEM任务中的多步推理方面取得了显著改进，但一个关键的残余失败模式仍然存在：中间推理步骤在语法上结构良好、数学上可执行且单位一致，但在上下文上缺乏依据。当前方法要么依赖无法评估语义意图的形式验证器，要么将检查算术和逻辑的双重任务负担给过程奖励模型（PRMs）。在本文中，我们提出了一种神经符号框架，将推理清晰地解耦为两个形式维度：符号有效性（$V$）和语义依据性（$G$）。我们通过使用确定性符号验证器作为硬过滤器，从结构上保证$V$。为了评估$G$，我们在验证器接受的流形上条件化训练一个PRM。为了高效训练该PRM，我们引入了反事实符号扰动（CSP），一种新颖的数据合成方法。

    arXiv:2608.26329v1 Announce Type: new  Abstract: While tool-augmented Large Language Models have significantly improved multi-step reasoning in quantitative STEM tasks, a critical residual failure mode remains: intermediate reasoning steps that are syntactically well-formed, mathematically executable, and unit-consistent, yet contextually ungrounded. Current approaches either rely on formal verifiers that cannot assess semantic intent, or burden Process Reward Models (PRMs) with the dual task of checking both arithmetic and logic. In this paper, we propose a neuro-symbolic framework that cleanly decouples reasoning into two formal dimensions: Symbolic Validity ($V$) and Semantic Groundedness ($G$). We guarantee $V$ by construction using a deterministic symbolic verifier acting as a hard filter. To assess $G$, we train a PRM conditionally on the verifier-accepted manifold. To train this PRM efficiently, we introduce Counterfactual Symbolic Perturbation (CSP), a novel data synthesis stra
    
[^103]: “不可能”有多不可能？评估大型语言模型中的概率言语感知

    How Unlikely Is "Unlikely"? Assessing Verbal Probability Perception Across Large Language Models

    [https://arxiv.org/abs/2608.26327](https://arxiv.org/abs/2608.26327)

    大型语言模型在理解概率词语时与人类基准高度一致，但对否定表达（如“不可能”）存在系统性高估偏差，且“可能”一词在模型间分歧最大。

    

    大型语言模型越来越多地生成和解释言语概率表达，但这些表达在模型间是否具有一致含义（或是否与人类对不确定性的感知相匹配）仍属未知。我们基于已有的人类基准，通过词到数字映射任务进行了一项系统的跨模型评估。在两种条件下——强制单一数字回答和解释引出——向19个模型呈现了11种不确定性表达，并辅以一项新颖的双向往返测试以检验内部一致性。大型语言模型以惊人的保真度追踪人类基准：词语排序得以保留，三个锚点被恢复，且“可能”在所有测试表达中显示出最高的方差和跨模型分歧，这与人类中记录的其双峰解释相一致。然而，模型对“不可能”和“不太可能”等否定表达表现出系统性向上偏差。

    arXiv:2608.26327v1 Announce Type: cross  Abstract: Large language models increasingly produce and interpret verbal probability expressions, yet whether these expressions carry consistent meaning across models (or match human perceptions of uncertainty) remains unknown. We present a systematic cross-model evaluation using a word-to-number mapping task grounded in established human benchmarks. Eleven uncertainty expressions were presented to 19 models under two conditions, forced single-number response and explanation elicitation, alongside a novel bidirectional roundtrip test of internal consistency. LLMs track the human benchmark with surprising fidelity: word ordering is preserved, three anchor points are recovered, and ``possible'' shows the highest variance and cross-model disagreement of any expression tested, consistent with its documented bimodal interpretation in humans. However, models show a systematic upward bias for negative expressions such as ``unlikely'' and ``improbable.
    
[^104]: 噪声响应何时具有普遍性？分词作为语言模型中的隐藏变量

    When Is Noise Response Universal? Tokenization as the Hidden Variable in Language Models

    [https://arxiv.org/abs/2608.26319](https://arxiv.org/abs/2608.26319)

    本文发现语言模型在词级噪声下表现普遍一致，但在字符级噪声下分化，并揭示分词机制是这一差异的隐藏决定因素，而非模型架构。

    

    文本神经模型的性能常常在输入受到噪声（如拼写错误、OCR错误或单词缺失）干扰时下降。我们研究了多种神经模型（包括句子嵌入和仅解码器的大型语言模型）的性能退化率，发现其一致性取决于噪声的尺度：在词级噪声下，架构差异很大的模型沿几乎相同的曲线下降；而在字符级噪声下，它们则出现分化。我们进一步确定决定因素是训练目标而非架构：跨越六种预训练范式的八个编码器最初分散，但在简短的对比训练方案后收敛到共同曲线。我们将词级与字符级的分裂追溯到分词机制：单个字符编辑迫使分词器重新分割周围单词，对令牌序列的扰动远大于删除整个单词。这一发现及其潜在机制...

    arXiv:2608.26319v1 Announce Type: new  Abstract: The performance of textual neural models often degrades when their inputs are corrupted by noise such as typos, OCR errors, or dropped words. We study the degradation rate across neural models, both sentence embeddings and decoder-only LLMs, and find that how consistent it is depends on the scale of the noise: under word-level noise, models with very different architectures decline along nearly the same curve, while under character-level noise they separate. We further identify the determining factor to be the training objective, not the architecture: eight encoders spanning six pretraining paradigms are scattered initially, and collapse onto a common curve after a short contrastive training recipe. We trace the word/character split to tokenization: a single character edit forces the tokenizer to re-segment the surrounding word, disturbing the token sequence far more than dropping a whole word does. This finding and its underlying mechan
    
[^105]: MemToC：大型语言模型中记忆-工具冲突解决的基准测试

    MemToC: Benchmarking Memory-Tool Conflict Resolution in Large Language Models

    [https://arxiv.org/abs/2608.26295](https://arxiv.org/abs/2608.26295)

    该论文提出了MemToC基准测试，通过可执行工具和已知正确性控制，系统评估了大型语言模型在工具返回与参数记忆冲突时的仲裁能力，发现工具返回结果强烈主导闭卷答案，且模型在工具错误时保留正确记忆的能力较弱。

    

    摘要：arXiv:2608.26295v1 公告类型：交叉 摘要：工具增强的大型语言模型（LLM）在工具返回结果与其参数化记忆冲突时，必须在两个可能出错的信息源之间进行仲裁，然而现有评估仅测量来源偏好，而未确立来源的正确性。我们引入了MemToC，一个用于工具返回后仲裁的受控基准测试，配备可执行工具。MemToC包含6,504个评估片段，这些片段基于542个经过质量控制的事实性问题构建，独立引出模型特定的闭卷答案，以及正确性已知的受控工具返回。这些组件实例化了四种来源正确性情况；工具错误和无工具条件作为单独对照。在五个开放权重7-9B模型中，工具返回结果强烈主导了引出的闭卷答案。四个指令调优模型在工具错误时保留已验证正确答案的合格案例中仅占6.5-17.1%，在遵循正确工具时占86.0-93.1%，并在78.4-86.0%的情况下重复工具返回结果。

    arXiv:2608.26295v1 Announce Type: cross  Abstract: Tool-augmented LLMs must arbitrate between two fallible sources when a tool return conflicts with their parametric memory, yet existing evaluations measure source preference without establishing source correctness. We introduce MemToC, a controlled benchmark for post-tool-return arbitration with executable tools. MemToC comprises 6,504 evaluation episodes constructed from 542 quality-controlled factual questions, independently elicited model-specific closed-book answers, and controlled tool returns of known correctness. These components instantiate four source-correctness cases; tool-error and no-tool conditions are separate controls. Across five open-weight 7-9B models, tool returns strongly dominate elicited closed-book answers. The four instruction-tuned models retain a verified-correct answer against an incorrect tool in only 6.5-17.1% of eligible cases, follow a correct tool in 86.0-93.1%, and repeat the tool return in 78.4-86.0% 
    
[^106]: 关于范围分类与当前知识编辑基准：一项负面结果，以INLAY作为无梯度案例研究

    On Scope Classification and Current Knowledge-Editing Benchmarks: A Negative Result, with INLAY as a Gradient-Free Case Study

    [https://arxiv.org/abs/2608.26292](https://arxiv.org/abs/2608.26292)

    当前知识编辑基准无法衡量范围分类决策，因为所有测试场景中任何路由方法的最大收益为零，表明这些基准存在结构性缺陷。

    

    arXiv:2608.26292v1 公告类型：交叉 摘要：SERAC系列中每个基于记忆的知识编辑器都依赖于一个范围决策：给定一个查询，存储的编辑是否适用？我们报告称，当前的知识编辑基准根本无法衡量这一决策。使用INLAY——我们构建的一个无梯度编辑器，以获得精确的每查询地面真实值（模型被冻结，编辑存在于外部可寻址记忆中，应用编辑是在解码时沿一个令牌的解除嵌入方向添加的偏置）——我们在涵盖三个数据集和三种输入条件的1,689个查询上执行了每个候选路由动作。一个选择每次最佳动作的神谕路由器在所有九个数据集-条件单元格中将一个单行静态策略的精度持平到小数点后四位：任何每查询路由方法的最大可获得增益为0.00分。弃权是唯一获胜动作，在1,689次中零次成功。原因是结构性的：这些是反事实基准，其评估问题询问...

    arXiv:2608.26292v1 Announce Type: cross  Abstract: Every memory-based knowledge editor in the SERAC lineage depends on a scope decision: given a query, does a stored edit apply? We report that current knowledge-editing benchmarks cannot measure this decision at all. Using INLAY, a gradient-free editor we built to obtain exact per-query ground truth (the model is frozen, edits live in an external addressable memory, and applying an edit is a bias added along one token's unembedding direction at decode time), we execute every candidate router action on 1,689 queries spanning three datasets and three input conditions. An oracle router choosing the best action every time ties a one-line static policy to four decimal places in all nine dataset-by-condition cells: the maximum attainable gain of any per-query routing method is 0.00 points. Abstention is the sole winning action zero times out of 1,689. The cause is structural: these are counterfactual benchmarks whose evaluation question asks 
    
[^107]: 一个用于编排异构语音与文本检索器的重排序器

    A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers

    [https://arxiv.org/abs/2608.26194](https://arxiv.org/abs/2608.26194)

    本文提出了一种名为STeReO的重排序器，它通过整合语音和文本检索器来聚合异构模态数据库，并利用自建数据集训练，从而在多模态检索中显著提升证据选择的准确性。

    

    检索增强生成（RAG）系统因其能够缓解大型语言模型（LLMs）中的幻觉现象而引起了广泛关注。尽管RAG的知识数据库日益多样化，包括语音和文本等多种模态，但针对此类多模态数据库场景的研究仍然有限。在本文中，我们提出了STeReO（语音与文本重排序编排器），一种基于语音和文本检索器的重排序器，用于聚合不同模态的数据库。为了解决缺乏专门训练数据的问题，我们首先构建了一个包含查询、混合模态证据及其相应相关性排名的数据集。然后，我们训练该重排序器，并在单模态和混合模态场景中评估其有效性。结果表明，所提出的算法擅长选择最相关的证据，从而显著改善下游任务。

    arXiv:2608.26194v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) systems have attracted significant interest for their ability to mitigate hallucinations in Large Language Models (LLMs). Although knowledge databases for RAG are increasingly diversifying to include various modalities such as speech and text, research on handling such multi-modal database scenarios remains limited. In this paper, we propose STeReO (Speech and Text Reranking Orchestrator), a reranker based on speech and text retrievers that aggregates disparate modality databases. To address the lack of specialized training data, we first curate a dataset comprising queries, mixed-modality evidence, and their corresponding relevance ranks. We then train the reranker and evaluate its effectiveness in both single-modality and mixed-modality scenarios. Our results demonstrate that the proposed algorithm excels at selecting the most relevant evidence, thereby significantly improving downstream question-
    
[^108]: 土耳其语RAG系统中分块与嵌入策略的比较研究

    Comparing Chunking and Embedding Strategies for Turkish RAG Systems

    [https://arxiv.org/abs/2608.26192](https://arxiv.org/abs/2608.26192)

    本文系统比较了土耳其语RAG系统中分块策略与嵌入模型的影响，发现布局感知分块能缩小嵌入模型差异，且领先嵌入模型间无显著统计差异。

    

    arXiv:2608.26192v1 公告类型：交叉 摘要：文档如何被分割为可检索的分块以及这些分块如何被嵌入，强烈影响检索增强生成（RAG）的质量，然而，对于诸如土耳其语这类形态丰富的语言，这两方面都尚未被系统研究。我们比较了土耳其语文档问答在三种分块策略（固定长度、语义和布局感知的Docling）、五种嵌入模型和两种生成器大语言模型上的表现，这些比较基于三份具有对比布局的文档。完全交叉设计产生了9,000个分级问答评估，每个评估由独立的评判模型打分，组件比较通过配对McNemar检验在Holm校正下进行。得出以下四点发现：分块策略决定了嵌入选择的重要性程度——布局感知分块将现代嵌入模型之间的差异压缩到约一分。三个领先的嵌入模型在统计上无显著差异，因此语言稀疏性并未削弱它们的性能。

    arXiv:2608.26192v1 Announce Type: cross  Abstract: How documents are segmented into retrievable chunks and how those chunks are embedded strongly affect Retrieval-Augmented Generation (RAG) quality, yet neither has been systematically studied for morphologically rich languages such as Turkish. We compare Turkish document question answering across three chunking strategies (fixed-length, semantic, and layout-aware Docling), five embedding models, and two generator LLMs, over three documents with contrasting layouts. The fully crossed design yields 9,000 graded question-answer evaluations, each scored by an independent judge model, and component comparisons are tested by paired McNemar tests under Holm correction. Four findings follow. The chunking strategy determines how much the embedding choice matters: layout-aware chunking compresses the spread between the modern embedding models to about a point. The three leading embedding models are statistically indistinguishable, so language sp
    
[^109]: 工具使用智能体的调用级可靠性研究

    Invocation-Level Reliability of Tool-Using Agents

    [https://arxiv.org/abs/2608.26189](https://arxiv.org/abs/2608.26189)

    本文发现工具使用智能体的调用级可靠性测量受评分规则结构性限制，严重性和恢复参数无法准确估计，且模型自身错误会显著降低深层任务的性能。

    

    工具使用智能体以两种方式失败：选择错误的工具，或形成错误的参数，而这两种失败中的任何一种早期发生都可能无声地破坏下游的一切。我们在两种情境下测量了区分这两种失败的“正确调用率”：一种是干净的教师强制上下文，另一种是模型自身的自由运行上下文，涵盖五个开放权重模型，在无污染的多步任务（深度1-8）上进行。到深度6时，模型自身干净上下文能力的大约70%被其早期错误所损失（L6 = 0.686, 0.684）。我们的核心发现涉及测量本身。在针对固定金标准轨迹的精确匹配评分下，传播模型的严重性和恢复参数不仅难以估计——它们被评分规则固定了。严重性被迫达到其边界（869个被污染步骤中0个正确）；恢复在结构上不可观测（580个被污染步骤中0个回到正轨，而随机预期为0.0058）。这两者都导致了失败。

    arXiv:2608.26189v1 Announce Type: new  Abstract: Tool-using agents fail two ways: choosing the wrong tool, or forming wrong arguments, and an early failure of either kind can silently corrupt everything downstream. We measure a correct-invocation rate that separates the two, under both a clean teacher-forced context and the model's own free-running context, on five open-weight models over contamination-free multi-step tasks (depths 1-8). By depth 6, roughly 70% of a model's own clean-context capability is lost to its own earlier mistakes (L6 = 0.686, 0.684). Our central finding concerns the measurement itself. Under exact-match scoring against a fixed gold trajectory, a propagation model's severity and recovery parameters are not merely hard to estimate - they are fixed by the scoring rule. Severity is forced to its boundary (0 of 869 poisoned steps correct); recovery is structurally unobservable (0 of 580 poisoned steps returned on-track, against an expected 0.0058 by chance). Both fo
    
[^110]: 当规范补全出错时：形式化并度量大型语言模型中的跳跃

    When the Canonical Completion Is Wrong: Formalizing and Measuring the Jump in Large Language Models

    [https://arxiv.org/abs/2608.26187](https://arxiv.org/abs/2608.26187)

    本文首次形式化了LLM中的“跳跃”概念，将其定义为具有可验证证书的有限扩展问题，并提供了度量方法，以解决关于LLM溯因能力的争论。

    

    arXiv:2608.26187v1 公告类型：交叉 摘要：大型语言模型（LLMs）是否能够执行从证据到新公理系统的溯因跳跃（通常称为“跳跃”），近来引起了相当大的争论。一种主流观点认为LLMs在结构上无法进行此类跳跃，而近期研究则对其机制和证据提出了质疑。然而，这一争论仍难以定论，因为该领域缺乏对跳跃的形式化定义以及测试双方观点的度量方法。在本文中，我们分四步开发了跳跃的形式化描述，并度量了第二步。这些步骤询问部分数据的默认补全是什么、何时被迫放弃它、何时放弃是正确的，以及连续跳跃如何复合。具体来说，我们将跳跃实例定义为有限扩展问题，并带有机器可检查的证书，证明正确的补全存在、在重命名意义下唯一，且与数据的规范补全不同。

    arXiv:2608.26187v1 Announce Type: cross  Abstract: Whether large language models (LLMs) can perform the abductive leap from evidence to a new system of axioms, commonly referred to as a jump, has recently attracted considerable debate. A prominent position holds that LLMs are structurally incapable of such jumps, while recent studies challenge both its mechanism and its evidence. However, the debate remains difficult to settle, since the field still lacks a formal definition of the jump and a measure to test either side. In this paper, we develop a formal account of the jump in four steps and measure the second. The steps ask what the default completion of partial data is, when abandoning it is forced, when the abandonment is correct, and how successive jumps compound. Specifically, we define a jump instance as a finite extension problem with a machine-checked certificate that a correct completion exists, is unique up to renaming, and differs from the canonical completion of the data. 
    
[^111]: 探究提示语言与响应语言对大型语言模型内容生成的影响

    Investigating the Influence of Prompt and Response Languages on LLM Content Generation

    [https://arxiv.org/abs/2608.26186](https://arxiv.org/abs/2608.26186)

    本研究揭示提示语言和响应语言对LLM生成内容长度有显著影响，但语义保真度保持稳定，突出了分词器在跨语言生成中的关键作用。

    

    本研究考察了提示语言和响应语言如何影响大型语言模型的行为。我们使用五个模型，在四种语言条件下评估了对68个非翻译问题的回答：英语到英语、英语到挪威语、挪威语到挪威语以及挪威语到英语。在移除拒绝回答的项目后，数据集包含1348个响应。我们通过Cohen d测量长度差异，通过LabSE余弦相似度测量语义保真度，并使用原始和软Jaccard计算跨语言关键词重叠度。提示语言对响应长度有显著影响。在英语输出时，挪威语提示使响应缩短约37%。在挪威语输出时，英语提示使响应缩短约41%。最大的跨语言对比显示，词数减少但令牌数减少较少，这反映了分词器的差异。尽管长度存在差异，语义相似度仍然保持较高。

    arXiv:2608.26186v1 Announce Type: cross  Abstract: This study examines how prompt and response language influence the behavior of large language models. Using five models, we evaluated answers to 68 non translation questions across four language conditions: English to English, English to Norwegian, Norwegian to Norwegian, and Norwegian to English. After removing refused items, the dataset contains 1348 responses. We measure length differences with Cohen d, semantic fidelity with LabSE cosine similarity, and cross lingual keyword overlap with both raw and soft Jaccard. Prompt language has a strong effect on response length. With English output, Norwegian prompts shorten responses by about thirty seven percent. With Norwegian output, English prompts shorten responses by about forty one percent. The largest cross lingual contrast shows a reduction in word count but a smaller reduction in tokens, reflecting tokenizer differences. Despite variation in length, semantic similarity remains hig
    
[^112]: PACEShop：评估个性化、可操作、组合式且基于证据的购物助手

    PACEShop: Evaluating Personalized, Actionable, Compositional, and Evidence-grounded Shopping Assistants

    [https://arxiv.org/abs/2608.26180](https://arxiv.org/abs/2608.26180)

    本文提出PACE评估框架和PACEShop基准数据集，以联合衡量购物助手在个性化、可操作性、组合性和证据基础方面的结构化响应质量，填补了现有评估方法的空白。

    

    购物助手正从排序产品列表转向结构化决策支持，系统需将购物者情境、产品证据和下一步指导综合为连贯的推荐体验。这改变了评估单元：即使流畅的回复也可能因忽略购物者情境、组件间自相矛盾或缺陷过于模糊而失败。现有的个性化、基于证据和LLM作为评判者的基准覆盖了该问题的部分，但未定义结构化购物助手响应的联合评估目标。我们将这一缺失的评估目标定义为PACE：个性化、可操作、组合式且基于证据的评估。我们通过两个工件实例化PACE：PACEShop，一个基准数据集，通过22,625条受控记录（包含结构化人物画像、可审计证据池、GOOD/BAD标签和黄金缺陷）使该目标可测量。

    arXiv:2608.26180v1 Announce Type: cross  Abstract: Shopping assistants are shifting from ranked product lists toward structured decision support, where systems must synthesize shopper context, product evidence, and next-step guidance into a coherent recommendation experience. This changes the unit of evaluation: a fluent response can still fail by ignoring shopper context, contradicting itself across components, or leaving defects too vague to localize. Existing personalization, grounding, and LLM-as-a-judge benchmarks cover pieces of this problem, but they do not define a joint evaluation target for structured shopping-assistant responses. We formulate this missing evaluation target as PACE: Personalized, Actionable, Compositional, and Evidence-grounded evaluation. We instantiate PACE with two artifacts: PACEShop, a benchmark dataset that makes the target measurable through 22,625 controlled records with structured personas, auditable evidence pools, GOOD/BAD labels, and gold defect f
    
[^113]: 基于多框架比较的长篇生成中大纲阶段研究

    A Multi-Framework Comparison of Outline Stages in Long-Form Generation with LLMs

    [https://arxiv.org/abs/2608.26177](https://arxiv.org/abs/2608.26177)

    本文首次解耦评估长篇生成中的大纲阶段，提出基于锚点的LLM评审协议，发现无单一框架在所有粒度上占优。

    

    长篇生成暴露了大语言模型的根本性局限。即使是700亿参数的模型在生成16k词元输出时也会出现长度崩溃，多章节故事常常引发“中间丢失”效应特有的属性漂移。“先大纲后写作”范式已被广泛采用，但现有研究评估的是最终写作而非大纲本身，混淆了两个本应解耦的评估对象。我们构建了一个统一的头对头基准测试，涵盖7个代表性长篇生成框架和3种生成粒度——单章、多章和整本书——并提出一种基于锚点的LLM作为评审者的协议，直接以5分锚定尺度对照源文本评估大纲。在21个框架-粒度组合中，没有单一框架占据主导地位；性能取决于框架内在输出与任务需求之间的匹配程度。

    arXiv:2608.26177v1 Announce Type: cross  Abstract: Long-form generation exposes fundamental limitations of large language models. Even 70B-parameter models exhibit length collapse at 16k-token outputs, and multi-chapter stories frequently trigger the attribute drift characteristic of the ``lost-in-the-middle'' effect. The ``outline-first, write-later'' paradigm has gained wide adoption, yet existing research evaluates the final writing rather than the outline itself, conflating two evaluation objects that should be decoupled. We construct a unified head-to-head benchmark covering 7 representative long-form generation frameworks across 3 generation granularities -- single-chapter, multi-chapter, and whole-book -- and propose an anchor-based LLM-as-a-judge protocol that directly assesses outlines against the source text on a 5-point anchored scale. Across 21 framework-granularity cells, no single framework dominates; performance depends on the match between a framework's intrinsic output
    
[^114]: 知识卡片：面向AI系统的结构化知识

    Knowledge Cards: Structured Knowledge for AI Systems

    [https://arxiv.org/abs/2608.26176](https://arxiv.org/abs/2608.26176)

    本文提出“知识卡片”作为AI系统的新文档形式，旨在填补现有模型卡片、数据卡片和系统卡片在描述系统内部知识结构（概念、关系和推理模式）方面的空白，以支持智能体AI的可检查性和领域专家审查。

    

    arXiv:2608.26176v1 公告类型：新 摘要：AI系统的输出会影响现实决策，且这些决策日益重要，这要求系统具备当前文档实践无法提供的内容：一种结构化、可检查的知识表示，用于支撑、情境化和推理这些决策，理想情况下应由领域专家审查并签署。现有的文档工件已能捕捉AI系统的重要方面。模型卡片描述系统如何行为，数据卡片描述系统训练所用数据，系统卡片描述部署系统的风险。但这些都未涉及输入与输出之间的层次，更确切地说，系统持有的概念、建模的关系以及应用的推理模式。对于模式识别任务，这一差距尚可容忍。对于智能体AI，系统会基于结论采取行动，这往往是区分有前景的概念验证与可操作解决方案的关键步骤。

    arXiv:2608.26176v1 Announce Type: new  Abstract: AI systems whose outputs inform real decisions, and increasingly consequential ones, require something that current documentation practice does not provide: a structured, inspectable representation of the knowledge they need to ground, contextualize, and reason about those decisions, ideally reviewed and signed off by a domain expert. Established documentation artefacts already capture important aspects of an AI system. Model cards describe how a system behaves, data cards describe what it was trained on, and system cards describe the risks of a deployed system. None of them addresses the layer between inputs and outputs, more precisely, the concepts a system holds, the relationships it models, and the patterns of reasoning it applies. For pattern-recognition tasks this gap is tolerable. For agentic AI, where systems act on their conclusions, it is the step that most often separates a promising proof of concept from an operational soluti
    
[^115]: 迷失在压缩中：对抽取式提示压缩器的一项受控跨语言审计

    Lost in Compression: A Controlled Cross-Lingual Audit of Extractive Prompt Compressors

    [https://arxiv.org/abs/2608.26175](https://arxiv.org/abs/2608.26175)

    这项研究通过跨十种语言的受控审计发现，英语监督训练的抽取式压缩器在非英语语言上存在显著的迁移差距，且该差距随压缩率增加而扩大，多语言训练则能缓解此问题。

    

    抽取式提示压缩承诺通过移除低信息量标记来降低大语言模型的推理成本，而诸如LLMLingua-2等学习型压缩器在英语基准测试中报告了强劲的结果。然而，大多数其他语言已经付出了额外的标记代价：相同内容所需的标记数量比英语多出1.3至1.8倍。我们探究压缩是缩小还是扩大了这一差距。通过使用涵盖五种文字系统的十种语言的完全平行数据，并在目标模型的标记器中控制预算匹配，我们对四种学习型压缩器与四种确定性基线进行了审计，涉及来自十个供应商的十一个目标模型（超过25万次评估调用）。其中三种压缩器（LLMLingua-2 XLM-R/mBERT；来自生产级Headroom栈的Kompress-v2）以英语监督训练，而第四种XProvence则采用多语言训练。首先，迁移差距是真实存在的，它在目标模型和压缩器骨干网络中复现，并且强烈依赖于压缩率：在0.33的压缩率下，这种差距尤为显著。

    arXiv:2608.26175v1 Announce Type: cross  Abstract: Extractive prompt compression promises to cut LLM inference costs by removing low-information tokens, and learned compressors such as LLMLingua-2 report strong results on English benchmarks. Most other languages already pay a token premium: the same content costs 1.3-1.8x more tokens than in English. We ask whether compression closes or widens this gap. Using fully parallel data in ten languages spanning five scripts, with controls budget-matched in the target model's tokenizer, we audit four learned compressors against four deterministic baselines, on eleven target models from ten vendors (over 250,000 evaluation calls). Three of the compressors are trained with English supervision (LLMLingua-2 XLM-R/mBERT; Kompress-v2 from the production Headroom stack); the fourth, XProvence, is trained multilingually. First, the transfer gap is real, replicates across target models and compressor backbones, and is strongly rate-dependent: at a 0.33
    
[^116]: 缓解多阶段LLM招聘流水线中的捏造问题：提示护栏与人在回路检查点的实证评估

    Mitigating Fabrication in Multi-Stage LLM Pipelines for Hiring: An Empirical Evaluation of Prompt Guardrails and Human-in-the-Loop Checkpoints

    [https://arxiv.org/abs/2608.26171](https://arxiv.org/abs/2608.26171)

    该论文通过实证评估表明，仅靠提示护栏不足以消除LLM招聘流水线中的捏造问题，而结合人在回路检查点能显著降低捏造率并消除身份虚构。

    

    多阶段LLM招聘流水线（简历改进、面试问题生成、答案反馈）可能会捏造资质、夸大条件、虚构经历。我们评估了两种缓解措施——提示护栏和人在回路（HITL）检查点——与完全自动化基线进行对比。在一项受控实验中（10份合成简历×2个职位描述×3次重复×3种条件；共180次运行），基线（C1）在96.7%的输出中产生了至少一项无依据声明（平均每输出6.80项发现）。提示护栏（C2）将发现密度降低了86%（从6.80降至每输出0.92项），但仍有50.0%的输出包含捏造内容，表明仅靠提示级缓解措施不足。在简历改进后设置人工检查点（C3）消除了所有身份捏造，将发现密度降低了59%（从6.88降至每输出2.82项），将项目级捏造率从96.7%降至75.0%（p=0.022），并减少了捕获职位描述中嵌入的陷阱要求。

    arXiv:2608.26171v1 Announce Type: cross  Abstract: Multi-stage LLM hiring pipelines (resume improvement, interview question generation, answer feedback) can fabricate credentials, inflate qualifiers, and invent experience. We evaluate two mitigations, prompt guardrails and human-in-the-loop (HITL) checkpoints, against a fully automated baseline. In a controlled experiment (10 synthetic resumes x 2 job descriptions x 3 repetitions x 3 conditions; 180 runs), the baseline (C1) produced at least one unsupported claim in 96.7% of outputs (mean 6.80 findings/output). Prompt guardrails (C2) reduced finding density by 86% (6.80 to 0.92/output), but 50.0% of outputs still contained a fabrication, showing prompt-level mitigation alone is insufficient. A human checkpoint after resume improvement (C3) eliminated all identity fabrications, reduced finding density by 59% (6.88 to 2.82/output), reduced item-level fabrication from 96.7% to 75.0% (p=.022), and cut capture of JD-embedded trap requiremen
    
[^117]: 大语言模型中的幻觉：基于生命周期的原因、检测、缓解与预防综述

    Hallucinations in LLMs: A Lifecycle-Based Survey of Causes, Detection, Mitigation, and Prevention

    [https://arxiv.org/abs/2608.26168](https://arxiv.org/abs/2608.26168)

    本文提出了一种基于生命周期的幻觉分类框架，将LLM幻觉分为数据、训练和推理三类，并系统综述了各阶段的原因、检测、缓解与预防方法，以及基准数据的评估标准。

    

    arXiv:2608.26168v1 公告类型：交叉 摘要：大语言模型（LLM）中的幻觉生命周期是一个概念，它有助于在医疗、法律和科学研究等高风险环境中建立关于LLM控制和可靠性的坚实框架。尽管以往的综述主要集中于检测或缓解，本综述提供了基于生命周期的LLM幻觉概述，涵盖其原因、检测、缓解和预防。我们提出了跨LLM生命周期的三类幻觉分类：数据相关、训练相关和推理相关，这与LLM开发的生命周期一致。每个阶段都讨论了幻觉的原因、检测方法，以及在特定缓解或预防干预下可解决的方式。此外，我们使用多个参数讨论了可用的基准数据，以确定其在识别、限制和评估幻觉方面的适用性。

    arXiv:2608.26168v1 Announce Type: cross  Abstract: The lifecycle of hallucination in LLMs is a concept that enables building solid frameworks on the control and reliability of LLMs in high-stakes environments, including health, legal, and scientific research. Although previous surveys have primarily focused on detection or mitigation, this survey provides a lifecycle-based overview of the hallucinations in the LLMs, their cause, detection, mitigation, and prevention.We propose a three-fold categorization of hallucinations across the LLM lifecycle: data-related, training-related, and inference-related, which is consistent with the lifecycle of the development of the LLM. Each of these stages is discussed regarding the cause of hallucinations, their detection, and the ways they can be addressed under specific mitigation or prevention interventions. In addition, we discuss the available benchmark data using a number of parameters so as to establish their suitability in identifying, restri
    
[^118]: 使用多编码器进行计算高效的自动化创造力评估

    Using Poly-Encoders for Computationally Efficient Automated Creativity Assessment

    [https://arxiv.org/abs/2608.26165](https://arxiv.org/abs/2608.26165)

    本文提出了一种基于Poly-Encoder的自动化创造力评估方法，利用小型BERT编码器在保持与大型语言模型相当性能的同时显著降低计算成本。

    

    自动化创造力评估一直是一个长期挑战，传统方法往往资源密集或缺乏实际准确性。我们引入了一种新颖的方法，使用Poly-Encoder进行计算高效且准确的自动化创造力评估。我们在一个来自科学创造性思维测试的公开数据集上微调了Poly-Encoder，该数据集包含约18,000个人工评分的问题回答。我们的方法利用小型预训练的BERT编码器，实现了与微调大型语言模型相当的性能，同时显著降低了计算需求。使用BERT系列模型和多编码数量的实验，与人类评分者的皮尔逊相关系数最高达到r = 0.74，95%置信区间[0.73, 0.75]，与资源密集型LLM的性能相匹配。这项研究弥合了高性能与计算效率之间的差距，可能促进其广泛实施。

    arXiv:2608.26165v1 Announce Type: cross  Abstract: Automated creativity assessment has been a long standing challenge, with traditional methods often being resource intensive or lacking practical accuracy. We introduce a novel approach by using Poly-Encoder for computationally efficient and accurate automated creativity assessment. We fine-tuned a Poly-Encoder on a public dataset from the Scientific Creative Thinking Test, comprised of approximately 18,000 human-rated question responses. Our method leverages small pre-trained BERT encoders, achieving performance comparable to fine-tuned Large Language Models while significantly reducing computational demands. Experiments with the BERT-family models and poly-code counts achieved Pearson correlations of up to r = 0.74, 95% CI [0.73, 0.75] with human raters, matching the performance of resource intensive LLMs. This study bridges the gap between high performance and computational efficiency, potentially enabling widespread implementation o
    
[^119]: 从声音到症状：面向对话式医疗代理的实时呼吸信号理解

    From Sound to Symptom: Real-Time Respiratory Signal Understanding for Conversational Healthcare Agents

    [https://arxiv.org/abs/2608.26163](https://arxiv.org/abs/2608.26163)

    本文提出HealthCUES，一个实时对话代理中首个用于副语言呼吸监测的流式管道，能够在不打断对话流的情况下实现亚秒级咳嗽检测及细粒度分析，并引入对话感知门控机制以防止警报疲劳。

    

    在实时口语对话中，咳嗽事件携带着具有临床价值的呼吸信号，然而现有的对话系统往往将其视为需要丢弃的声学噪声。我们提出了HealthCUES（来自具身声音的临床理解），这是一种用于实时对话代理的副语言呼吸监测流式管道，据我们所知，这一能力在所有先前系统中均缺失。HealthCUES通过一个与对话轮次边界对齐的滚动缓冲区处理音频，实现了亚秒级事件检测而不打断对话流。除了二元咳嗽检测外，该系统还提供细粒度分析：（i）区分咳嗽与清嗓，（ii）咳嗽亚型分类（干咳、湿咳、犬吠样咳、百日咳样咳）并附置信度评分，以及（iii）带有起止边界的持续时间估计。为防止警报疲劳，HealthCUES引入了对话感知门控机制。

    arXiv:2608.26163v1 Announce Type: cross  Abstract: Cough events during live spoken conversations carry clinically valuable respiratory signals, yet existing dialogue systems treat them as acoustic noise to be discarded. We present HealthCUES (Clinical Understanding from Embodied Sounds), a streaming pipeline for paralinguistic respiratory monitoring in real-time conversational agents, a capability that, to the best of our knowledge, is absent from all prior systems. HealthCUES processes audio through a rolling buffer aligned with dialogue turn boundaries, enabling sub-second event detection without interrupting conversational flow. Beyond binary cough detection, the system provides fine-grained analytics: (i) differentiation between coughing and throat clearing, (ii) cough subtype classification (dry, wet, barking, whooping) with confidence scores, and (iii) temporal duration estimation with start-end boundaries. To prevent alert fatigue, HealthCUES introduces dialogue-aware gating mec
    
[^120]: 通过双种子比较实现大型语言模型概率采样的互惠去偏

    Mutual Debiasing via Dual-Seed Comparison for Probabilistic Sampling in Large Language Models

    [https://arxiv.org/abs/2608.26161](https://arxiv.org/abs/2608.26161)

    本文提出双种子比较（DSC）方法，通过比较两个独立 LLM 生成的种子来中和采样偏差，从而提升概率采样的保真度。

    

    arXiv:2608.26161v1 公告类型：交叉 摘要：尽管大型语言模型（LLMs）在推理和决策方面表现出显著能力，但高保真概率采样仍然是一个持续的挑战。在生成随机变量时，LLMs 始终表现出系统性偏差，扭曲了目标概率分布。当前方法通常依赖单一的自生成种子，这继承了模型特定的偏差。为克服这一脆弱性，我们引入了双种子比较（DSC），一种透明、无工具依赖的协议，利用两个独立的 LLM 生成种子来中和偏差。DSC 比较两个种子的字符级序数值以构建比特序列，将该序列转换并归一化为伪均匀变量，然后通过逆累积分布函数（CDF）将变量映射到目标分布。实证结果表明，DSC 在96%的案例中显著优于现有方法。

    arXiv:2608.26161v1 Announce Type: cross  Abstract: Although Large Language Models (LLMs) demonstrate remarkable capabilities in reasoning and decision-making, high-fidelity probabilistic sampling remains a persistent challenge. When generating random variables, LLMs consistently exhibit systematic biases that warp the target probability distributions. Current approaches often rely on a single, self-generated seed, which inherits model-specific biases. To overcome this vulnerability, we introduce Dual-Seed Comparison (DSC), a transparent, tool-free protocol that utilizes two independent LLM-generated seeds to neutralize bias. DSC compares the character-level ordinal values of the two seeds to construct a bit sequence, converts and normalizes this sequence into a pseudo-uniform variate, and then maps the variate to the target distribution through the inverse cumulative distribution function (CDF). Empirical results show that DSC substantially outperforms existing methods across 96\% of e
    
[^121]: 自生成文本识别：质量启发式、跨任务迁移及LLM评估中的下游偏差

    Self-Generated Text Recognition: Quality Heuristics, Cross-Task Transfer, and Downstream Bias in LLM Evaluation

    [https://arxiv.org/abs/2608.26159](https://arxiv.org/abs/2608.26159)

    本研究通过系统分析实验设计选择（操作化）对自生成文本识别准确率的影响，调和了先前矛盾结论，并证实了质量启发式在LLM评估中的关键作用。

    

    自生成文本识别（SGTR）——即大型语言模型识别自身输出的能力——对依赖LLM作为评估者或监控者的AI安全机制构成风险。具体而言，LLM可能识别出同一模型其他副本的输出，并做出有偏见的判断或直接共谋。以往研究关于当前模型是否具备显著SGTR能力得出了相互矛盾的结论。我们通过识别关键实验设计选择（我们称之为“操作化”）来调和这些发现，这些选择驱动了结果的分歧。评估13-21个模型在六种操作化下的表现，我们发现准确率随评估格式（成对比较 vs. 对文本的单独评估）、对话结构（在用户标签 vs. 助手标签中呈现候选文本）以及用于生成候选文本的任务领域（例如，编程 vs. 摘要）而有显著变化。我们证实了先前观察，即质量启发式...

    arXiv:2608.26159v1 Announce Type: cross  Abstract: Self-Generated Text Recognition (SGTR)--the ability of an LLM to identify its own outputs--poses risks to AI safeguards that rely on LLMs as evaluators or monitors. Specifically, an LLM may recognize outputs from other copies of the same model and make biased judgments or collude outright. Prior work has drawn conflicting conclusions about whether current models possess significant SGTR capabilities. We reconcile these findings by identifying key experimental design choices--which we term operationalizations--that drive divergent results. Evaluating 13-21 models across six operationalizations, we find that accuracy varies substantially with evaluation format (pairwise vs. individual assessments of text), conversation structure (presenting candidate text in user tags vs. assistant tags), and the domain of the task used to generate candidate text (e.g., coding vs. summarization). We corroborate previous observations that a quality heuris
    
[^122]: VFA：通过无视觉适配增强多语言多模态大语言模型

    VFA: Empowering Multilingual MLLMs via Vision-Free Adaptation

    [https://arxiv.org/abs/2608.26155](https://arxiv.org/abs/2608.26155)

    提出了一种通过组合互补任务向量来解耦多语言增强与视觉对齐的框架，仅用不到2%的数据即可在多语言多模态基准上提升多个MLLM的性能。

    

    多模态大语言模型发展迅速，但大多数仍以英语为中心，因为扩展多语言多模态指令调优受限于高质量非英语图像-文本监督数据的稀缺和高成本。尽管多语言文本数据丰富，但简单的文本微调可能会破坏视觉-语言对齐并引发灾难性遗忘。我们提出了无视觉适配（VFA）框架，该框架通过在共享的LLM骨干上组合互补的任务向量，将多语言语言增强与视觉对齐解耦。具体来说，我们在多语言文本数据上微调基础LLM以得到多语言任务向量，然后将其与多模态大语言模型的视觉对齐任务向量合并。在五个多语言多模态基准上的六个多模态大语言模型实验表明，该方法在保持通用多模态和纯文本能力的同时，取得了持续改进。此外，使用不到2%的数据即可实现这些改进。

    arXiv:2608.26155v1 Announce Type: cross  Abstract: Multimodal large language models have advanced rapidly, yet most remain English-centric, as scaling multilingual multimodal instruction tuning is limited by the scarcity and high cost of high-quality non-English image-text supervision. Although multilingual text data is abundant, naive textual fine-tuning can disrupt vision-language alignment and induce catastrophic forgetting. We propose Vision-Free Adaptation (VFA), a framework that decouples multilingual language enhancement from visual alignment by composing complementary task vectors over a shared LLM backbone. Specifically, we fine-tune a base LLM on multilingual text data to derive a multilingual task vector, which is then merged with the vision-aligned task vector of an MLLM. Experiments on five MLLMs across six multilingual multimodal benchmarks show consistent improvements while preserving both general multimodal and text-only capabilities. Moreover, using less than 2% of the
    
[^123]: 评估面向癌症患者的AI生成摘要

    Evaluating AI Generated Summaries for Cancer Patients

    [https://arxiv.org/abs/2608.26154](https://arxiv.org/abs/2608.26154)

    本研究通过人类专家和LLM评估者双重框架评估癌症患者护理应用中的AI生成摘要，识别出遗漏和不准确等局限，并迭代优化提示设计与安全防护。

    

    arXiv:2608.26154v1 公告类型：交叉 摘要：大型语言模型（LLMs）正越来越多地被整合到数字健康平台中，用于生成复杂医疗数据的摘要。尽管这些模型可以改善患者的参与度和沟通效果，但这些系统也引发了在临床环境中准确性、忠实性和安全性方面的担忧。在本研究中，我们通过双重评估框架，在一个癌症患者护理应用中评估了AI生成的摘要。人类领域专家，包括肿瘤科临床医生和面向患者的护理人员，提供了关于摘要质量的地面真实评估，涵盖准确性、临床相关性和可读性等维度。同时，我们采用了作为评估者的LLM（LLM-as-a-judge）。在生成的摘要中识别出了一些局限性，例如偶尔的遗漏和轻微的不准确之处。这些问题被系统地分析，并用于迭代改进提示设计、基础依据和安全防护措施。

    arXiv:2608.26154v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly being integrated into digital health platforms to generate summaries of complex medical data. Although these models can improve patient engagement and communication, these systems also raise concerns about accuracy, faithfulness, and safety in clinical contexts. In this study, we evaluate AI-generated summaries within a cancer patient care application using a dual assessment framework. Human domain experts, including oncology clinicians and patient-facing care staff, provided ground-truth evaluations of summary quality along dimensions of accuracy, clinical relevance, and readability. In parallel, we employed LLMs serving as evaluators (LLM-as-a-judge). Some limitations were identified in the generated summaries e.g., occasional omissions and minor inaccuracies. These were systematically analyzed and used to iteratively improve prompt design, grounding, and safety guardrails.
    
[^124]: 人工智能模型能够预测并协同调节人类记忆搜索

    Artificial Intelligence Models Can Predict and Collaboratively Modulate Human Memory Search

    [https://arxiv.org/abs/2608.26152](https://arxiv.org/abs/2608.26152)

    本研究首次证明大型语言模型能够预测并协同调节人类在语义记忆搜索中的心理轨迹，从而作为认知工具增强而非取代人类生成性思维。

    

    arXiv:2608.26152v1 公告类型：交叉 摘要：大型语言模型（LLMs）展现出前所未有的自然语言生成能力和许多基于文本的问题解决能力。实际上，在许多基于语言的任务中，例如常规编程，这些人工智能模型已经减少甚至消除了人类输入的需求。但与其取代人类的认知努力，LLMs或许更适合作为认知工具来扩展人类能力，尤其是在涉及开放式概念探索和创造性构思的任务中。然而，我们尚不清楚这些模型如何在人机交互中增强这种生成性的人类认知能力。在本研究中，我们探索并评估了LLMs在语义记忆搜索过程中跟随和增强人类心理轨迹的能力。为测试这一点，我们使用了语义流畅性任务（SFT），这是一个经典的认知范式，要求生成性语义记忆检索，长期以来一直用于此目的。

    arXiv:2608.26152v1 Announce Type: cross  Abstract: Large language models (LLMs) exhibit unprecedented natural language generation and many text-based problem-solving capabilities. Indeed, in many language-based tasks, for example routine coding, these artificial intelligence models have reduced, or even eliminated, the need for human input. But rather than replacing human cognitive effort, LLMs may instead serve as cognitive tools to extend human abilities, particularly when they are engaged in a task requiring open-ended conceptual exploration and creative ideation. However, we are yet to understand how these models may enhance such generative human cognitive abilities in human--AI interactions. In this study, we explore and evaluate the ability of LLMs to follow and enhance human mental trajectories during semantic memory search. To test this, we use the semantic fluency task (SFT), a classic cognitive paradigm requiring generative semantic memory retrieval that has long served to ch
    
[^125]: 面向可解释的抑郁症检测：将声学特征与DSM-5指标关联起来

    Towards Interpretable Depression Detection: Linking Acoustic Features to DSM-5 Indicators

    [https://arxiv.org/abs/2608.26148](https://arxiv.org/abs/2608.26148)

    本文提出一种可解释的抑郁症检测框架，通过透明映射声学特征到DSM-5临床指标，在普通硬件上本地运行以保护隐私，初步验证了其有效性。

    

    抑郁症影响着全球数百万人，然而诊断依赖于可能错过真实行为的主观自我报告。本文提出了一种方法，通过透明的关联框架，将语音声学与DSM-5抑郁行为指标联系起来。与黑箱模型不同，该框架明确地将声学特征（如音高变异性、停顿、语速）映射到临床指标上，从而实现可解释的、基于指标级别的输出。系统在普通硬件上本地运行以保护隐私。在DAIC-WOZ上的初步评估显示，声学特征与DSM-5中精神运动变化和注意力困难指标之间存在方向一致的关联，支持了设计原理。未来工作将在纵向数据集上验证，并在保持边缘约束的同时扩展多模态集成。

    arXiv:2608.26148v1 Announce Type: new  Abstract: Depression affects millions worldwide, yet diagnosis relies on subjective self-reports that may miss authentic behavior. This paper presents an approach linking speech acoustics to DSM-5 depressive-behavior indicators through a transparent Linkage Framework. Unlike black-box models, the framework explicitly maps acoustic features (pitch variability, pauses, speech tempo) to clinical indicators, enabling interpretable, indicator-level outputs. The system runs locally on commodity hardware (HW) to preserve privacy. Preliminary evaluation on DAIC-WOZ shows directionally consistent associations between acoustic features and DSM-5 indicators for psychomotor change and concentration difficulty, supporting the design rationale. Future work will validate on longitudinal datasets and extend multimodal integration while maintaining edge constraints.
    
[^126]: 因果对齐推理探索：面向医疗大语言模型的方法

    CARE: Causally-Aligned Reasoning Exploration for Medical Large Language Models

    [https://arxiv.org/abs/2608.26147](https://arxiv.org/abs/2608.26147)

    CARE通过引入因果充分性和邻近可学习性两个条件，提出了一种因果对齐的推理探索框架，有效解决了医疗大语言模型中强化学习的信用分配失败和梯度爆炸问题，避免了“答案正确、推理错误”的陷阱。

    

    大语言模型（LLMs）在医疗推理方面展现出巨大潜力，然而专家标注数据的稀缺性和高昂成本限制了其发展。虽然强化学习提供了一种可扩展的替代方案，但医学中标准的基于结果的方法常常遭受自回归信用分配失败和梯度方差爆炸的问题。这导致了“答案正确、推理错误”的陷阱，即模型无意中强化了虚假相关性和数据集捷径，而非有效的临床推导。在本研究中，我们提出了因果对齐推理探索（CARE），这是一个基于理论的内在经验策展框架。CARE建立在高质量训练轨迹的两个严格条件之上：因果充分性，利用基于一致性的自验证机制来模拟$do$-演算干预，并有效去偏梯度；以及邻近可学习性，采用动态机制来确保训练稳定性。

    arXiv:2608.26147v1 Announce Type: new  Abstract: Large Language Models (LLMs) have shown strong potential for medical reasoning, yet the scarcity and cost of expert-annotated data constrain their progress. While reinforcement learning offers a scalable alternative, standard outcome-based methods in medicine often suffer from autoregressive credit assignment failure and gradient variance explosion. This leads to the "Right Answer, Wrong Reason" trap, where models inadvertently reinforce spurious correlations and dataset shortcuts rather than valid clinical deduction. In this work, we propose Causally-Aligned Reasoning Exploration (CARE), a theoretically grounded framework for intrinsic experience curation. CARE is built upon two rigorous conditions for high-quality training trajectories: Causal Sufficiency, which utilizes an agreement-based self-verification mechanism to mimic $do$-calculus interventions and effectively debias gradients; and Proximal Learnability, which employs dynamic 
    
[^127]: Vagdhenu：一种基于韵律（Vrutta）感知的梵语诗偈转唱诵（TTS）系统

    Vagdhenu: A Vrutta (Meter) Aware Shloka-to-Chant (TTS) System for Sanskrit

    [https://arxiv.org/abs/2608.26146](https://arxiv.org/abs/2608.26146)

    本文介绍了一种基于韵律感知的梵语诗偈转唱诵系统，通过卡纳达文路由和精确音系处理，在现有TTS骨架上实现了高保真唱诵。

    

    摘要：arXiv:2608.26146v1 公告类型：新 摘要：我们提出了Vagdhenu，一种面向梵语的韵律（vrutta）感知诗偈转唱诵系统：一种将格律诗映射为高保真唱诵（parayana）朗诵的文本转语音系统。这是一份经验报告，而非新架构。我们采用现成的流匹配TTS骨干网络和大规模神经声码器，并添加了忠实梵语唱诵流水线所需的组件：一个前端，将梵语通过卡纳达文正字法路由，以避免天城文在印度模型中触发的印地语式元音省略；一个遵循微妙梵语音系的前端（包括visarga连音及其腭化音和唇化音变体、alpaprana和mahaprana的送气对立，以及齿音、卷舌音和腭音擦音的区分）；以及一个韵律感知机制，检测格律并选择在“半参考规则”下精确匹配的参考。我们报告了一个塑造该系统的负面结果：i

    arXiv:2608.26146v1 Announce Type: new  Abstract: We present Vagdhenu, a vrutta (meter) aware shloka-to-chant system for Sanskrit: a text-to-speech system that maps   a metrical verse to its chanted parayana recitation at high fidelity. This is an experience report, not a new   architecture. We take an off-the-shelf flow-matching TTS backbone and a large-scale neural vocoder, and add the   components a faithful Sanskrit chant pipeline needs: a frontend that routes Sanskrit through Kannada orthography to   avoid the Hindi-style schwa deletion that Devanagari triggers in Indic models; a frontend that obeys subtle   Sanskrit phonology (visarga sandhi with its jihvamuliya and upadhmaniya allophones, the aspiration contrast of   alpaprana and mahaprana, and the dental, retroflex, and palatal sibilants kept distinct); and a vrutta-aware   mechanism that detects the meter and picks an exactly matched reference under a half-reference rule. We report a   negative result that shaped the system: i
    
[^128]: 为什么当前的XAI不足以支持阿拉伯语自然语言处理：可解释性差距的批判性综述

    Why Current XAI Is Not Enough for Arabic NLP: A Critical Survey of the Explainability Gap

    [https://arxiv.org/abs/2608.26144](https://arxiv.org/abs/2608.26144)

    本文批判性地指出，阿拉伯语NLP的可解释性存在方法、任务和语言三方面的显著差距，现有XAI技术不足以应对其独特挑战。

    

    可解释人工智能（XAI）现已成为自然语言处理（NLP）的重要主题；然而，阿拉伯语NLP在三个相互关联的层面上仍未得到充分解释。首先，存在方法差距：阿拉伯语XAI严重依赖少量事后技术，如LIME、SHAP、注意力可视化和显著性，而更广泛的NLP XAI提供了更丰富的诊断性、反事实、探测、基于理据和以人为中心的方法。其次，存在任务差距：现有的阿拉伯语XAI工作集中在分类任务上，尤其是情感分析、仇恨/攻击性语言检测、假新闻和垃圾邮件，而对生成、检索、翻译、摘要、结构化预测和对话的覆盖较弱。第三，存在语言差距：许多解释识别了有影响力的标记，但很少解释阿拉伯语特有的现象，如形态学、附着词素、方言变异、双言制、正字法歧义、变音符号、代码切换、命名实体、文化相关引用等。

    arXiv:2608.26144v1 Announce Type: new  Abstract: Explainable AI (XAI) is now a major theme in NLP; however, Arabic NLP remains under-explained in three connected senses. First, there is a method gap: Arabic XAI relies heavily on a small set of post-hoc techniques such as LIME, SHAP, attention visualization, and saliency, while broader NLP XAI offers richer diagnostic, counterfactual, probing, rationale-based, and human-centered methods. Second, there is a task gap: existing Arabic XAI work is concentrated in classification tasks, especially sentiment analysis, hate/offensive language detection, fake news, and spam, with weaker coverage of generation, retrieval, translation, summarization, structured prediction, and dialogue. Third, there is a linguistic gap: many explanations identify influential tokens, but rarely explain Arabic-specific phenomena such as morphology, clitics, dialectal variation, diglossia, orthographic ambiguity, diacritics, code-switching, named entities, cultural r
    
[^129]: 超越准确性：视觉语言模型在梗图仇恨言论检测中的定性分析

    Beyond Accuracy: A Qualitative Analysis of Vision-Language Models for Hate Speech Detection in Memes

    [https://arxiv.org/abs/2608.26143](https://arxiv.org/abs/2608.26143)

    本研究通过定性分析四种视觉语言模型在梗图仇恨言论检测中的表现，揭示了它们在理解上下文和讽刺方面的局限性，并强调了超越分类准确性的评估重要性。

    

    梗图已成为个人分享对当代社会和政治问题看法的一种强大工具。其匿名性及快速传播的能力，使其成为传播仇恨的有力媒介。识别这种复杂且依赖上下文的仇恨言论仍然非常困难。尽管视觉语言模型（VLMs）在多模态任务上表现出色，但它们往往忽视上下文、讽刺及其他在识别仇恨梗图中起关键作用的微妙线索。在本研究中，我们对四种最先进的视觉语言模型进行了定性分析：LLaVA-7B、Qwen-VL、GPT-4o mini和Claude 3 Haiku。我们在零样本和少样本提示条件下评估这些模型，以检验上下文框架如何影响其输出。我们的分析超越了简单的分类准确性，侧重于对模型生成的理由进行定性评估，从而提供更深入的见解。

    arXiv:2608.26143v1 Announce Type: cross  Abstract: Memes have turned out to be a powerful tool through which individuals share their ideas concerning contemporary social and political problems. Their anonymity, as well as their ability to go viral, make them a powerful medium for spreading hate. It remains very difficult to identify such complex and context-dependent hate speech. Although they display excellent performance on multimodal tasks, vision-language models (VLMs) tend to ignore context, irony, and other subtle cues that play a key role in identifying hateful memes. In this work, we present a qualitative analysis of four state-of-the-art VLMs: LLaVA-7B, Qwen-VL, GPT-4o mini, and Claude 3 Haiku. We evaluate these models under zero-shot and few-shot prompting to examine how contextual framing influences their outputs. Our analysis goes beyond simple classification accuracy and focuses on a qualitative evaluation of the models' generated justifications, providing a more in-depth 
    
[^130]: 位置即一切：一种基于MLLM的指代表达分割的免费午餐令牌压缩策略

    Position Is All You Need: A Free Lunch Token Compression Strategy for MLLM-based Referring Expression Segmentation

    [https://arxiv.org/abs/2608.26142](https://arxiv.org/abs/2608.26142)

    本文提出PAYN，一种仅依赖位置信息的令牌压缩策略，用于MLLM-based指代表达分割，通过保留位置嵌入和局部空间结构来避免性能退化。

    

    指代表达分割（RES）旨在从复杂且隐含的文本查询中生成像素级分割掩码。尽管多模态大语言模型（MLLMs）的最新进展显著提升了RES性能，但其高昂的计算开销仍是关键瓶颈，然而这一方面鲜有研究。为填补这一空白，我们首先评估了典型令牌压缩方法在此任务上的表现，并观察到令人惊讶的性能退化。本文旨在理解这一现象以寻求解决方案。通过大量实验，我们发现RES的令牌压缩需要保留原始位置嵌入和局部邻近空间结构，这表明视觉令牌位置信息比其他任务更为关键。基于此洞察，我们提出疑问：能否仅基于位置信息设计令牌压缩方法？因此，我们提出了PAYN，一种...

    arXiv:2608.26142v1 Announce Type: cross  Abstract: Referring Expression Segmentation (RES) aims to generate pixel-wise segmentation masks from complex and implicit textual queries. While recent advances in Multimodal Large Language Models (MLLMs) have substantially boosted RES performance, their prohibitive computational overhead remains a critical bottleneck, which, however, is rarely explored. To fill this gap, we first evaluate typical token compression methods on this task and observe a surprising performance degradation. In this paper, we aim to understand this phenomenon for a solution. By extensive experiments, we find that token compression for RES requires preserving the original position embeddings and local neighboring spatial structures, indicating that visual token position information is far more critical than in other tasks. Building on this insight, we ask: Can we design the token compression method purely based on the position information? Therefore, we propose PAYN, a
    
[^131]: AdaThinking-E：基于单令牌熵调控的自适应思考机制

    AdaThinking-E: One-Token Entropy Regulation for Adaptive Thinking

    [https://arxiv.org/abs/2608.26141](https://arxiv.org/abs/2608.26141)

    本文提出AdaThinking-E，一种通过单令牌熵调控的自适应思考强化学习框架，能根据问题复杂度智能决定是否启用深度推理，从而减少简单任务的计算开销并提升整体准确性。

    

    多模态大语言模型通过引入显式思考过程，展现出了强大的文档推理能力。尽管这种能力显著提升了在挑战性任务上的表现，但当前模型对所有问题均统一应用深度推理，导致在简单任务上产生不必要的计算开销。这不仅降低了用户体验，还对基准数据集的准确性产生了负面影响。我们识别出对自适应思考机制的关键需求，该机制能根据问题复杂度智能判断何时启动推理。为解决这一问题，我们提出了AdaThinking-E，一种新颖的强化学习框架，通过单令牌熵调控学习自适应思考。我们的核心见解是，模型在决定是否启动思考（或不启动）时的置信度，可以通过对关键决策令牌处预测概率分布的熵分析来量化。

    arXiv:2608.26141v1 Announce Type: new  Abstract: Multimodal large language models have demonstrated strong document reasoning capabilities by incorporating explicit thinking processes. While this capability significantly improves performance on challenging tasks, current models apply such deep reasoning uniformly to all questions, resulting in unnecessary computational overhead for simple task. This not only degrades user experience but also negatively impact accuracy on benchmark datasets. We identify the critical need for adaptive thinking mechanisms that can intelligently determine when to engage reasoning based on question complexity. To address this, we propose AdaThinking-E, a novel reinforcement learning framework that learns adaptive thinking through one-token entropy regulation. Our key insight is that model confidence in the decision to engage thinking (or not) can be quantified through entropy analysis of the predicted probability distribution at critical decision tokens. Th
    
[^132]: 面向扩散大语言模型的后缀缓存机制

    Affix Cache for Diffusion Large Language Models

    [https://arxiv.org/abs/2608.26140](https://arxiv.org/abs/2608.26140)

    ACache提出一种后缀导向的缓存重用机制，通过选择性重计算关键锚点令牌的键值状态，在扩散大语言模型中高效恢复共享文本跨度的精度损失。

    

    扩散大语言模型（DLLMs）支持非自回归解码和双向上下文建模，但高效推理仍具挑战性。与自回归系统（其键值缓存可对共享前缀重用）不同，DLLMs通过双向注意力将共享上下文令牌的键值状态与动态生成的令牌耦合，导致朴素缓存重用失效，而完全重计算成本高昂。我们提出ACache，一种面向DLLMs中共享文本跨度（超越前缀）的后缀导向缓存重用机制。ACache通过衡量掩码生成令牌上的影响，识别出少量请求特定的关键后缀令牌（称为锚点令牌），仅对这些令牌选择性重计算键值状态，同时重用其余后缀缓存。基于Fast-dLLM构建，ACache在重计算约20%后缀时，恢复了不同设置下直接后缀缓存重用导致的精度损失。

    arXiv:2608.26140v1 Announce Type: new  Abstract: Diffusion Large Language Models (DLLMs) enable non-autoregressive decoding and bidirectional context modeling, but efficient inference remains challenging. Unlike autoregressive systems, whose key-value (KV) cache can be reused for shared prefixes, DLLMs couple the KV states of shared context tokens with evolving generated tokens through bidirectional attention, making naive cache reuse stale while full recomputation is expensive. We present ACache, an affix-oriented cache reuse mechanism for shared text spans in DLLMs beyond prefixes. ACache identifies a small request-specific subset of critical affix tokens, called Anchor Tokens, by measuring their influence on masked generation tokens, and selectively recomputes the KV states of only these tokens while reusing the remaining affix cache. Built on Fast-dLLM, ACache recovers the accuracy loss caused by direct affix-cache reuse across different settings when recomputing around 20% of affi
    
[^133]: 语法与语义：Transformer如何学习深层依赖关系

    Syntax vs. Semantics: How Transformers Learn Deep Dependencies

    [https://arxiv.org/abs/2608.26139](https://arxiv.org/abs/2608.26139)

    该论文提出一个机制框架，揭示Transformer学习深层语义依赖时存在“梯度饥饿”现象，并解释了思维链策略如何通过外化推理步骤绕过该抑制机制。

    

    大型语言模型展现出显著的句法流畅性，但其获取深层语义依赖关系的优化动力学仍然知之甚少。我们提出一个机制性框架，将该学习过程建模为表面统计与深层语义之间的竞争。我们的理论分析识别出一种“梯度饥饿”现象，即在早期优化过程中，稀疏语义依赖的错误信号被主动抑制。这种抑制阻碍了结构推理的学习，并导致其出现表现为突然的相变。此外，该框架为思维链（CoT）策略的有效性提供了机制性基础。通过将中间推理步骤外化为具体标记，CoT有效绕过了隐式推理固有的抑制机制。我们在从玩具变换器到不同规模的实验中验证了这些发现。

    arXiv:2608.26139v1 Announce Type: cross  Abstract: Large Language Models demonstrate remarkable syntactic fluency, yet the optimization dynamics governing their acquisition of deep semantic dependencies remain poorly understood. We propose a mechanistic framework that models this learning process as a competition between Surface Statistics and Deep Semantics. Our theoretical analysis identifies a ``Gradient Starvation" phenomenon where the error signals for sparse semantic dependencies are actively suppressed during early optimization. This suppression impedes the learning of structural reasoning and causes its emergence to manifest as a sudden phase transition. Furthermore, this framework offers a mechanistic basis for the effectiveness of Chain-of-Thought (CoT) strategies. By externalizing intermediate reasoning steps into concrete tokens, CoT effectively bypasses the suppression regime inherent to implicit reasoning. We validate these findings across scales ranging from toy transfor
    
[^134]: 心理健康自然语言处理中的跨平台泛化失败：社交媒体上Transformer模型的五轴公平性审计

    Cross-Platform Generalisation Failure in Mental Health Natural Language Processing: A Five-Axis Fairness Audit of Transformer Models on Social Media

    [https://arxiv.org/abs/2608.26138](https://arxiv.org/abs/2608.26138)

    本研究通过五轴公平性审计框架揭示了心理健康NLP模型在跨平台应用中存在显著的性能下降和校准失败，表明平台特定泛化问题严重制约了实际部署。

    

    arXiv:2608.26138v1 公告类型：新 摘要：我们引入了跨平台公平性评估（CPFE）框架——一个涵盖判别性能、校准、统计显著性、预测公平性和归因稳定性的五轴审计协议——并将其应用于四个Transformer模型（BERT、RoBERTa、Emotion-DistilRoBERTa、GoEmotions-RoBERTa），这些模型在Kaggle心理健康语料库（n=35,556）上训练，并在Reddit（n=6,257）和Twitter（n=2,883）测试集上评估，情绪标签映射到临床代理指标。所有三个独立评估的模型均表现出一致且显著的跨平台AUC下降（Reddit上30.3-35.4%，Twitter上37.9-39.5%），相对于平台内性能（AUC 0.983-0.987），并在五个独立训练种子中得到确认。校准失败同时且严重：ECE从域内的0.056-0.060上升到Reddit上的0.196-0.229和Twitter上的0.499-0.542。平台特定的温度缩放将平均ECE降低了88.0%。

    arXiv:2608.26138v1 Announce Type: new  Abstract: We introduce the Cross-Platform Fairness Evaluation (CPFE) framework -- a five-axis audit protocol covering discriminative performance, calibration, statistical significance, prediction equity, and attribution stability -- and apply it to four transformer models (BERT, RoBERTa, Emotion-DistilRoBERTa, GoEmotions-RoBERTa) trained on a Kaggle mental health corpus (n=35,556) and evaluated on Reddit (n=6,257) and Twitter (n=2,883) test sets with emotion labels mapped to clinical proxies. All three independently evaluated models exhibit consistent and substantial cross-platform AUC degradation (30.3-35.4% on Reddit, 37.9-39.5% on Twitter) relative to within-platform performance (AUC 0.983-0.987), confirmed across five independent training seeds. Calibration failure is concurrent and severe: ECE rises from 0.056-0.060 in-domain to 0.196-0.229 on Reddit and 0.499-0.542 on Twitter. Platform-specific temperature scaling reduces mean ECE by 88.0% w
    
[^135]: 可解释、公平评估的自动L2口语评估，超越单人人类评分上限，以及为什么停顿编码不改变LLM流利度评分

    Interpretable, Fairly Evaluated Automated L2 Speaking Assessment that Beats the Single-Human Ceiling and Why Pause Encoding Does Not Change LLM Fluency Scores

    [https://arxiv.org/abs/2608.26137](https://arxiv.org/abs/2608.26137)

    该论文提出一种可解释的特征与LLM混合模型，在L2口语自动评分中超越单人人类评分中位数，实现高准确度和公平性。

    

    arXiv:2608.26137v1 公告类型：新公告 摘要：第二语言（L2）英语学习者很少能与伙伴练习口语。口语也是焦虑感最强的技能。这些差距推动了自动化口语练习和评分市场的快速增长。但自动化评分只有在准确、可解释、公平，并针对正确的人类基准进行测评时才值得信赖。我们构建了一个可解释的特征加LLM混合模型，用于自发L2对话。我们从未拟合人类标签，而是针对ICNALE全球评分档案进行评估：140段演讲由约80名训练有素的评分者按10项分析标准评分。我们对130段具有可用音频的L2演讲进行评分。一个确定性的De-Jong语音时间复合指标达到rho=0.764。与单一文本LLM流利度判断混合后，针对共识金标准达到Spearman rho=0.818。这比80名个体训练评分者中81%的共识一致性更好：高于中位评分者（rho=0.73）并接近最佳，且在此水平上。

    arXiv:2608.26137v1 Announce Type: new  Abstract: Second-language (L2) English learners can rarely rehearse speaking with a partner. Speaking is also the most anxiety-laden skill. These gaps drive a fast-growing market for automated speaking practice and scoring. But an automated score is trustworthy only if it is accurate, interpretable, fair, and benchmarked against the right human bar. We build an interpretable feature-plus-LLM hybrid for spontaneous L2 dialogue. We evaluate it without ever fitting to the human labels, against the ICNALE Global Rating Archive: 140 speeches rated by ~80 trained raters on 10 analytic criteria. We score the 130 L2 speeches with usable audio. A deterministic De-Jong speech-timing composite reaches rho=0.764. Blended with a single text-LLM fluency judgment, it reaches Spearman rho=0.818 against the consensus gold. This agrees with the consensus better than 81% of the 80 individual trained raters: above the median rater (rho=0.73) and near the best, and at
    
[^136]: 奖励引导的稀疏自编码器与解决方案完整性混淆

    Reward-Informed Sparse Autoencoders and the Solution-Completeness Confound

    [https://arxiv.org/abs/2608.26136](https://arxiv.org/abs/2608.26136)

    本文发现奖励引导的稀疏自编码器在区分好/坏推理时，其分离效果主要源于解决方案完整性（如长度和结构线索）而非推理质量，揭示了潜在混淆因素。

    

    arXiv:2608.26136v1 公告类型：新 摘要：稀疏自编码器（SAEs）将语言模型的激活分解为稀疏、可解释的特征，而将它们导向推理的一个吸引人的方法是利用强化学习已经产生的信号——奖励——来策划其数据。我们构建了这种奖励引导的稀疏自编码器（RI-SAE）：我们将GRPO轨迹分为高奖励（“好”）和低奖励（“坏”）的推理延续，在其激活上训练一个标准的JumpReLU SAE，然后探究这种好/坏分离实际衡量了什么。在Llama-3.1-8B上，16,384个特征中的一个稀疏子集确实能区分类别（选定特征上的轮廓系数为0.79，而完整编码为0.005），但一组控制测试表明，这种分离主要是解决方案完整性而非推理质量：一个TF-IDF文本分类器已经能拆分这些类别（AUC 0.75--0.83），而仅三个结构线索（长度、封闭推理块和框选答案）就达到AUC 0.70（占99%的...）。

    arXiv:2608.26136v1 Announce Type: new  Abstract: Sparse autoencoders (SAEs) decompose language-model activations into sparse, interpretable features, and an appealing way to aim them at reasoning is to curate their data with a signal reinforcement learning already produces: the reward. We build such a reward-informed SAE (RI-SAE): we split GRPO trajectories into high-reward ("good") and low-reward ("bad") reasoning continuations, train a standard JumpReLU SAE on their activations, and then ask what the resulting good/bad separation actually measures. On Llama-3.1-8B a sparse subset of the 16,384 features does separate the classes (silhouette 0.79 on the selected features versus 0.005 for the full code), but a control battery shows the separation is largely solution completeness rather than reasoning quality: a TF-IDF text classifier already splits the classes (AUC 0.75--0.83), and three structural cues alone (length, a closed reasoning block, and a boxed answer) reach AUC 0.70 (99% of 
    
[^137]: 数据科学方法评估荣誉候选人

    Data Science Approaches to Evaluating Honours Candidates

    [https://arxiv.org/abs/2608.26135](https://arxiv.org/abs/2608.26135)

    本文提出了一种模块化数据科学流水线，结合多种NLP技术从开源情报中提取公众情绪，并验证了领域知情算法MINOS在评估荣誉候选人声誉方面的优越性。

    

    摘要：arXiv:2608.26135v1 公告类型：新 摘要：我们提出了一种模块化数据科学流水线，用于从碎片化、非结构化的开源情报（OSINT）中估计公众对个人的情绪。该方法链式结合了网络搜索、文本提取、相关性过滤、分词、共指消解和情感分析，将异构网络材料转换为可审计的个人层面情绪分布。我们比较了AFINN和VADER与MINOS（一种领域知情的情感算法，设计用于检测与声誉风险、不当行为和积极公共贡献相关的语言）。应用于具有已知声誉结果的公众人物时，MINOS在正面、模糊和负面案例之间提供了最清晰的区分。结果表明，链式NLP和OSINT方法可以支持透明、可复现、人机协同的高风险决策支持情感评估。我们通过英国荣誉系统展示了该方法，其中个人被评估。

    arXiv:2608.26135v1 Announce Type: new  Abstract: We present a modular data-science pipeline for estimating public sentiment towards individuals from fragmented, unstructured open-source intelligence (OSINT). The method chains web search, text extraction, relevance filtering, tokenisation, co-reference resolution, and sentiment analysis to convert heterogeneous web material into auditable person-level sentiment distributions. We compare AFINN and VADER with MINOS, a domain-informed sentiment algorithm designed to detect language associated with reputational risk, misconduct, and positive public contribution. Applied to public figures with known reputational outcomes, MINOS gives the clearest separation between positive, ambiguous, and negative cases. The results show that chained NLP and OSINT methods can support transparent, reproducible, human-in-the-loop sentiment assessment for high-stakes decision support. We demonstrate the approach on the UK Honours system, where individuals are 
    
[^138]: Agent Seer：从规格理解中合成场景

    Agent Seer: Synthesizing Scenarios from Specification Understanding

    [https://arxiv.org/abs/2608.26133](https://arxiv.org/abs/2608.26133)

    Agent Seer仅从单一MCP工具规格中自动合成逼真的多轮评估场景，无需人工干预、示例或实时工具访问，显著提升了AI代理评估的可扩展性和适应性。

    

    arXiv:2608.26133v1 公告类型：新 摘要：评估使用外部工具的AI代理，需要能够捕捉实践者如何组合工具并在对话轮次中迭代的真实测试场景。手工构建此类场景需要深厚的领域专业知识，无法跨工具生态系统扩展，并且产生的静态基准无法跟踪不断演变的API。我们观察到，工具规格——函数名称、自然语言描述和类型化参数模式——已经编码了足够的语义信息，无需人工策展或实时工具执行即可合成逼真的评估场景。Agent Seer利用这一潜在信息：从单一的模型上下文协议（MCP）规格出发，无需示例、无需实时工具访问、无需领域特定调优。该流程丰富原始模式，生成带有合成工具输出的分级场景，并将其扩展为基于模拟数据的多轮对话，展现出强大的工具调用能力。

    arXiv:2608.26133v1 Announce Type: new  Abstract: Evaluating AI agents that use external tools requires realistic test scenarios that capture how practitioners compose tools and iterate across conversation turns. Constructing such scenarios by hand demands deep domain expertise, does not scale across tool ecosystems, and produces static benchmarks that cannot track evolving APIs. We observe that tool specifications -- function names, natural-language descriptions, and typed parameter schemas -- already encode sufficient semantic information to synthesize realistic evaluation scenarios without manual curation or live tool execution. Agent Seer builds off this latent information: from a single Model Context Protocol (MCP) specification, with no examples, no live tool access, and no domain-specific tuning. This pipeline enriches raw schemas, generates graded scenarios with synthetic tool outputs, and expands them into mock-data-grounded multi-turn dialogues that exhibit strong tool-calling
    
[^139]: 基于SLM条件的分层关系路由用于标注属性图学习

    SLM-Conditioned Hierarchical Relation Routing for Labeled Property Graph Learning

    [https://arxiv.org/abs/2608.26132](https://arxiv.org/abs/2608.26132)

    该论文提出了一种将小型语言模型集成到图消息选择中的分层路由架构，通过目标条件查询实现更精确的语义证据选择。

    

    arXiv:2608.26132v1 公告类型：交叉 摘要：标注属性图将关系结构与附着在节点和关系上的异构文本和分类属性相结合。传统图神经网络通常将这些属性表示为静态特征向量，限制了其确定哪些语义证据应影响特定预测目标的消息传播的能力。我们提出了一种基于SLM条件的分层关系路由架构，该架构将小型语言模型直接集成到图消息选择中。拓扑GNN提供稳定的结构表示和预测锚点。对于每个目标节点，入射消息结合邻居的结构状态、节点属性编码、关系属性编码和关系类型。一个参数高效的SLM处理结构化图软令牌，并生成目标条件的路由查询。该查询首先在每个关系类型内选择相关消息，然后进行跨关系类型的聚合。

    arXiv:2608.26132v1 Announce Type: cross  Abstract: Labeled property graphs combine relational structure with heterogeneous textual and categorical properties attached to both nodes and relationships. Conventional graph neural networks typically represent these properties as static feature vectors, limiting their ability to determine which semantic evidence should influence message propagation for a particular prediction target. We propose SLM-Conditioned Hierarchical Relation Routing, an architecture that integrates a small language model directly into graph message selection. A topology GNN provides a stable structural representation and prediction anchor. For each target node, incident messages combine the neighbor's structural state, node-property encoding, relationship-property encoding, and relationship type. A parameter-efficient SLM processes structured graph soft tokens and produces a target-conditioned routing query. This query first selects relevant messages within each relat
    
[^140]: 在现实对话情境中评估语言模型

    Evaluating Language Models in Realistic Conversational Contexts

    [https://arxiv.org/abs/2608.26131](https://arxiv.org/abs/2608.26131)

    本文提出了一个名为UPHELD的大型基准测试集，包含专业编剧创作的真实人类对话和大量人工标注，用于填补现有评估框架在衡量大型语言模型人类规模对话能力方面的空白。

    

    随着大型语言模型（LLMs）越来越多地被部署用于开放式、多轮交互，以人类规模评估对话质量已成为一个核心挑战。现有的针对摘要、翻译或短问答任务构建的评估框架，不足以充分衡量人类规模对话的一致性，尤其是当这些指标本身的推导和验证往往依赖于合成数据而非人类来源时。我们通过引入UPHELD（Upwork人类规模评估的长对话）来填补这一空白，这是一个大型、带参考的基准测试集，用于评估超越事实正确性的人类规模对话能力。UPHELD包含数百个由专业编剧创作的完整人机对话，具有现实的回合密度，并在30,000多个专家生成的对话回合中提供了36,000多条逐回合人类标注。利用UPHELD，我们系统地评估了经典自动评估方法。

    arXiv:2608.26131v1 Announce Type: new  Abstract: As Large Language Models (LLMs) are increasingly deployed to serve open-ended, multi-turn interactions, evaluating conversational quality at human scale has become a central challenge. Existing evaluation frameworks built for summarization, translation, or short-form QA tasks fall short of adequately measuring the consistency of human-scale dialogue, especially when derivation and validation of these metrics themselves often rely on synthetic rather than human sources. We fill the gap by introducing UPHELD (UPwork Human-Scale Evaluated Long Dialogues), a large, reference-full benchmark for evaluating human-scale conversational ability beyond factual correctness. UPHELD consists of hundreds of complete human-to-human dialogues authored by professional script writers, with realistic turn densities and 36,000+ per-turn human annotations across 30,000+ expert-generated dialogue turns. Using UPHELD, we systematically evaluate classical automa
    
[^141]: 智能体不翻页：LLM工具响应的首块选择策略

    Agents Don't Paginate: First-Chunk Selection for LLM Tool Responses

    [https://arxiv.org/abs/2608.26130](https://arxiv.org/abs/2608.26130)

    该论文发现编码智能体从不使用分页获取额外工具响应块，并提出首块选择策略（基于0/1背包）以最大化首块中黄金项的包含率，实验表明现有方法在精度上存在不足。

    

    摘要：基于大型语言模型（LLM）构建的编码智能体，如Claude Code、Cursor、OpenAI Codex、GitHub Copilot和Aider，接收到的工具响应经常超出智能体每轮的令牌预算。标准的解决方案是分页，这在产生这些响应的所有协议中均可用；然而，在来自公共模型上下文协议中间件的会话日志语料库中，我们观察到没有智能体发起过获取第二块的请求。智能体读取的是首块，因此我们询问黄金项（智能体所需的那一项）在首块中出现的频率，即首块精确率$p_1$。在受控的离线基准测试中，我们将首块选择视为0/1背包问题，并在500个SWE-bench验证任务上比较了六种价值函数，然后通过单轮文件定位探针在五种语言模型（4,800次LLM调用；非端到端解析率测试）上测试$p_1$是否重要。两个预先注册的假设未成立。

    arXiv:2608.26130v1 Announce Type: new  Abstract: Coding agents built on large language models (LLMs), such as Claude Code, Cursor, OpenAI Codex, GitHub Copilot, and Aider, receive tool responses that routinely exceed the agent's per-turn token budget. The standard remedy, pagination, is available in every protocol that produced these responses; yet across the corpus of session logs from a public Model Context Protocol middleware we observed no agent-initiated requests for a second chunk. The first chunk is what the agent reads, so we ask how often the gold item (the one the agent needs) is placed first in it: the precision-at-1 rate $p_1$.   In a controlled offline benchmark we treat first-chunk selection as a 0/1 knapsack and compare six value functions on 500 SWE-bench Verified tasks, then test whether $p_1$ matters with a single-turn file-localisation probe on five language models (4,800 LLM calls; not an end-to-end resolve-rate test). Two pre-registered hypotheses did not hold and 
    
[^142]: FIRSTPASS：一个基于真实编辑结果的多领域、多轮同行评审数据集

    FIRSTPASS: A Multi-Domain, Multi-Round Peer Review Dataset Grounded in Real Editorial Outcomes

    [https://arxiv.org/abs/2608.26129](https://arxiv.org/abs/2608.26129)

    该论文提出了FIRSTPASS，这是首个覆盖多学科、多轮真实编辑对话的同行评审数据集，填补了非计算机科学领域评审数据的空白。

    

    科学同行评审数据集仅针对计算机科学和机器学习领域训练了人工智能系统，这些系统能够批判消融研究，但从未见过生物学家要求污染控制，或化学家质疑核磁共振谱峰归属。我们引入了FIRSTPASS，这是首个基于多学科高影响力期刊完整多轮编辑对话构建的大规模同行评审数据集。该数据集源自《自然通讯》强制透明同行评审（2022年11月实施），包含跨越五个科学领域（生物学、化学、神经科学、物理学和地球科学）的3668条记录，捕捉了科学验证的完整迭代结构：初始审稿报告、作者逐点回应和更新后的审稿人评估。每条记录都带有直接来源于编辑决策的结果标签（STANDARD fo...）

    arXiv:2608.26129v1 Announce Type: cross  Abstract: Scientific peer review datasets have trained AI systems exclusively on Computer Science and Machine Learning venues, producing models that critique ablation studies yet have never seen a biology reviewer demand contamination controls or a chemist question Nuclear Magnetic Resonance (NMR) spectral assignments. We introduce FIRSTPASS, the first large-scale peer review dataset built on complete multi-round editorial dialogues from a multidisciplinary high-impact journal. Curated from Nature Communications mandatory transparent peer review (instituted November 2022), FIRSTPASS comprises 3,668 records spanning five scientific domains (biology, chemistry, neuroscience, physics, and earth science), capturing the full iterative structure of scientific validation: initial referee reports, author point-by-point responses, and updated reviewer assessments. Each record carries an outcome label derived directly from editorial decisions (STANDARD fo
    
[^143]: 基于图的金融波动率动态建模

    Graph-Based Modeling of Financial Volatility Dynamics

    [https://arxiv.org/abs/2608.26127](https://arxiv.org/abs/2608.26127)

    提出了一种将隐含波动率曲面建模为动态图序列的金融感知图时空网络，以更准确地预测已实现波动率。

    

    准确预测已实现波动率（$RV$）对于风险管理和衍生品定价至关重要。尽管隐含波动率（$IV$）曲面提供了丰富的信息内容，但现有方法将其视为静态图像，未能捕捉其固有动态。为克服这一局限，我们提出了金融感知图时空网络（FA-GSTN），一种新颖的架构，将$RV$预测重新定义为结构化金融对象的演化建模。FA-GSTN从$IV$曲面构建时空图序列，其中节点对应网格点，边编码自适应空间（日内）和显式时间（日间）依赖关系。该模型通过金融感知节点特征（如期权希腊字母）融入领域知识，并通过多尺度时间平滑门结合自适应鲁棒损失函数处理高频噪声。在大规模数据集上的综合评估...

    arXiv:2608.26127v1 Announce Type: cross  Abstract: Accurate forecasting of realized volatility ($RV$) is crucial for risk management and derivatives pricing. Although the implied volatility ($IV$) surface offers rich informational content, prevailing methods that treat it as a static image fail to capture its inherent dynamics. To overcome this limitation, we propose the Finance-Aware Graph Spatio-Temporal Network (FA-GSTN), a novel architecture that reframes $RV$ forecasting as modeling the evolution of a structured financial object. FA-GSTN builds a spatio-temporal graph sequence from the $IV$ surface, where nodes correspond to grid points and edges encode adaptive spatial (intra-day) and explicit temporal (inter-day) dependencies. The model incorporates domain knowledge through finance-aware node features (e.g., option Greeks) and tackles high-frequency noise via a multi-scale temporal smoothing gate coupled with an adaptive robust loss function. Comprehensive evaluations on a large
    
[^144]: TelecomGPT-R1：面向电信栈的统一开源推理器

    TelecomGPT-R1: A Unified Open-Source Reasoner for the Telecom Stack

    [https://arxiv.org/abs/2608.26126](https://arxiv.org/abs/2608.26126)

    本文发布了TelecomGPT-R1-9B，一个统一的开源电信推理器，通过围绕协议、知识、建模和故障四个推理轴的67,427个示例监督微调语料库，结合轴特定思维链增强，在GSMA开放电信排行榜上取得顶尖性能，弥合了通用推理器与领域特定电信LLM之间的能力差距。

    

    arXiv:2608.26126v1 公告类型：新 摘要：电信是基于大型语言模型（LLM）推理的高杠杆领域，因为日常工程工作流需要同时基于规范标准、运营遥测、供应商特定的故障证据以及精确的射频/网络计算。然而，当前LLM在电信中的集成仍受制于两方面的能力差距：通用推理器往往缺乏电信特定基础，而领域特定的电信LLM在结构化、多步推理方面仍显不足。为弥合这一差距，我们发布了TelecomGPT-R1-9B，一个统一的开源电信推理器，在GSMA开放电信排行榜上排名顶尖。具体而言，我们策划了一个包含67,427个示例的监督微调（SFT）语料库，围绕四个互补的推理轴组织：协议、知识、建模和故障。该语料库基于与轴匹配的公共网络来源构建，并通过轴特定的思维链（CoT）生成增强。

    arXiv:2608.26126v1 Announce Type: new  Abstract: Telecommunications is a high-leverage domain for large language model (LLM)-based reasoning because routine engineering workflows require joint grounding in normative specifications, operational telemetry, vendor-specific fault evidence, and exact RF/network calculations. However, current LLM integration in telecom remains bottlenecked by a two-sided capability gap: generic reasoners often lack telecom-specific grounding, while domain-specific telecom LLMs remain limited in structured, multi-step reasoning. To bridge this gap, we release TelecomGPT-R1-9B, a unified open-source telecom reasoner that ranks top-performing on the GSMA open telco leaderboard. Specifically, we curate a 67,427-example supervised fine-tuning (SFT) corpus organized around four complementary reasoning axes: protocol, knowledge, modeling, and fault. The corpus is built from axis-matched public web sources and enhanced through axis-specific chain-of-thought (CoT) ge
    
[^145]: 训练时可解释性用于多语言仇恨言论检测：将模型推理与人类理由对齐

    Training-Time Explainability for Multilingual Hate Speech Detection: Aligning Model Reasoning with Human Rationales

    [https://arxiv.org/abs/2608.26125](https://arxiv.org/abs/2608.26125)

    本文提出了一种训练时可解释性框架，通过将模型推理与人类标注理由对齐，在多语言仇恨言论检测中同时提升了分类性能和解释质量，尤其有效捕捉了文化特定的隐性反穆斯林仇恨线索。

    

    摘要：针对穆斯林社区的在线仇恨往往以文化编码、多语言形式出现，规避了传统的人工智能审核。这类系统虽然准确，但缺乏透明度，并存在偏见、过度审查或审查不足的风险，尤其是在脱离社会文化背景时。我们提出了一种训练时的可解释性框架，该框架将模型推理与人工标注的理由对齐，从而提升分类性能和可解释性。我们的方法在HateXplain（英语）和BullySent（印地英语）上进行了评估，反映了反穆斯林仇恨在这两种语言中的普遍性。使用LIME、积分梯度、梯度×输入和注意力机制，我们评估了准确性、解释质量以及跨方法一致性。结果表明，基于梯度和注意力的正则化提高了F分数，增强了合理性和忠实度，并捕捉了检测隐性反穆斯林仇恨的文化特定线索，从而提供了更有效的检测手段。

    arXiv:2608.26125v1 Announce Type: cross  Abstract: Online hate against Muslim communities often appears in culturally coded, multilingual forms that evade conventional AI moderation. Such systems, though accurate, remain opaque and risk bias, over-censorship, or under-moderation, particularly when detached from sociocultural context. We propose a \emph{training-time} explainability framework that aligns model reasoning with human-annotated rationales, improving both classification performance and interpretability. Our approach is evaluated on HateXplain (English) and BullySent (Hinglish), reflecting the prevalence of anti-Muslim hate across both languages. Using LIME, Integrated Gradients, Grad X Input, and attention, we assess accuracy, explanation quality, and cross-method agreement. Results show that gradient- and attention-based regularization improve F-scores, enhance plausibility and faithfulness, and capture culturally specific cues for detecting implicit anti-Muslim hate, offer
    
[^146]: 从自然语言政策到可执行决策：一个可解释的大语言模型框架

    Natural-Language Policies to Executable Decisions: An Interpretable Large Language Model Framework

    [https://arxiv.org/abs/2608.26124](https://arxiv.org/abs/2608.26124)

    本文提出一个生产级LLM定价系统，通过严格决策边界和可解释条件树，将非结构化政策转化为确定性可审计的决策，实现了开放政策支持与可靠金融决策的结合。

    

    arXiv:2608.26124v1 公告类型：新 摘要：大规模旅游中的定价自动化具有挑战性，因为旅行订单高度非结构化，而定价政策复杂、快速演变且本质上开放。传统规则引擎脆弱且维护成本高，而不受约束的LLM代理缺乏金融决策所需的可靠性和可审计性。我们提出一个生产级LLM驱动的定价系统，具有严格决策边界：LLM执行结构化提取和有界政策/路径选择，而所有数字定价（包括总价计算）均由确定性方式执行。政策被编译为可解释的条件树，支持对新条款和演变规则的无代码修改，同时为人类参与控制提供可审计工件。基于日志轨迹的定期微调进一步改进树归纳和路径匹配。该系统已部署于一家市属国有旅游企业。

    arXiv:2608.26124v1 Announce Type: new  Abstract: Pricing automation in large-scale tourism is challenging because travel orders are highly unstructured, while pricing policies are complex, rapidly evolving, and inherently open-ended. Traditional rule engines are brittle and costly to maintain, whereas unconstrained LLM agents lack the reliability and auditability required for financial decisions. We present a production-grade LLM-powered pricing system with a strict decision boundary: LLMs perform structured extraction and bounded policy/path selection, while all numeric pricing, including total-price computation, is executed deterministically. Policies are compiled into interpretable condition trees, enabling open-ended support for new clauses and evolving rules without code changes, while exposing auditable artifacts for human-in-the-loop control. Periodic fine-tuning on logged traces further improves tree induction and path matching. Deployed at a municipal state-owned tourism enter
    
[^147]: 印度叙事在翻译中幸存吗？大型语言模型中对印度口头传统的叙事同质化

    Which India Survives Translation? Narrative Homogenisation Across Indian Oral Traditions in LLMs

    [https://arxiv.org/abs/2608.26123](https://arxiv.org/abs/2608.26123)

    本研究通过计算分析发现，大型语言模型在生成印度不同口头传统的叙事时存在显著的同质化倾向，导致地区文化多样性被扁平化。

    

    大型语言模型（LLMs）主要基于英文互联网文本进行训练，这些文本过度代表了某些文化叙事，引发了模型将非西方故事传统的多样性扁平化为单一同质化原型的担忧。我们提出了一项试点计算研究，针对三个截然不同的印度地区口头与文学传统——拉贾斯坦邦的帕布吉史诗、古典泰米尔桑加姆诗歌以及孟加拉民间故事——进行了考察。我们为每个传统收集了真实参考语料（分别为11、21和10个段落），并提示了两个LLM（Claude Sonnet和Gemini）生成了54个请求，涵盖每个传统的三种提示类型——通用型、文化特定型和地区语言型。使用Sentence-BERT嵌入和余弦相似度，我们衡量了参考漂移（输出与其自身传统真实文本相对于其他两个传统的接近程度）和跨传统趋同现象。

    arXiv:2608.26123v1 Announce Type: new  Abstract: Large language models (LLMs) are trained predominantly on English-language internet text that over-represents certain cultural narratives, raising concerns that models flatten the diversity of non-Western storytelling traditions into a single homogenized archetype. We present a pilot computational study examining this across three maximally distinct Indian regional oral and literary traditions: the Rajasthani Pabuji epic, classical Tamil Sangam poetry, and Bengali folk tales. We collected authentic reference corpora for each tradition (11, 21, and 10 passages respectively) and prompted two LLMs (Claude Sonnet and Gemini) with 54 generation requests spanning three prompt types per tradition - generic, culturally specific, and regional-language. Using Sentence-BERT embeddings and cosine similarity, we measure reference drift (how closely outputs track their own tradition's authentic texts relative to the other two) and cross-tradition conv
    
[^148]: 模型能否免费捕捉自身的幻觉？：无标签怀疑信号在弃权任务中与带标签数据集表现相当

    Can a Model Catch Its Own Hallucinations for Free?: Label-Free Doubt Signals Hold Their Own Against a Labelled Dataset for Abstention

    [https://arxiv.org/abs/2608.26121](https://arxiv.org/abs/2608.26121)

    本文证明，仅使用模型自身无标签的置信度信号进行弃权调优，即可达到与使用带标签数据集监督训练相当的性能，从而无需额外标注即可让模型识别并避免自身幻觉。

    

    arXiv:2608.26121v1 公告类型：新 摘要：大型语言模型在陈述虚假事实时与真实事实一样流畅，但模型在内部往往“知道”自己何时处于不稳定状态：它分配给自身答案的概率在它出错的事实上往往会下降。对此采取行动通常的方法是教导模型弃权而非猜测，这需要一组包含正确和错误答案的带标签数据集。我们提出疑问：模型自身的置信度（这是免费的且无需标签）是否能替代这一工作。我们微调每个模型（使用LoRA），使其在冻结置信度高时回答，在置信度低时说“我不确定”，仅使用该信号而不使用正确性标签。在六个开放权重模型（1B-8B，两个系列）的短形式事实问答任务中，由独立评判模型裁定正确性，这种无标签方法在与标签监督的弃权调优相比时表现相当：在匹配覆盖率下，我们未发现两者之间存在统计上可检测的差异。一个结论...

    arXiv:2608.26121v1 Announce Type: new  Abstract: Large language models state false facts as fluently as true ones, yet a model often "knows" internally when it is on shaky ground: the probability it assigns to its own answer tends to dip on the facts it gets wrong. The usual way to act on this, teaching a model to abstain rather than guess, requires a labelled dataset of right and wrong answers. We ask whether the model's own confidence, which is free and needs no labels, can do that job instead. We fine-tune each model (with LoRA) to answer when its frozen confidence is high and to say "I'm not sure" when it is low, using the signal alone and no correctness labels. Across six open-weights models (1B-8B, two families) on short-form factual question answering, with correctness adjudicated by an independent judge model, this label-free recipe holds its own against label-supervised abstention-tuning: at matched coverage we find no statistically detectable difference between the two. A con
    
[^149]: 通过采样引导和扩展大型语言模型的策略

    Recipes for Steering and Scaling LLMs via Sampling

    [https://arxiv.org/abs/2608.26120](https://arxiv.org/abs/2608.26120)

    本文提出了一种基于采样（SMC和RE算法）的灵活框架，用于引导和扩展LLMs生成质量，无需外部监督，且比现有方法更具扩展性。

    

    大型语言模型（LLMs）是概率模型，通常通过自回归分解来定义。尽管近期研究开始探索基础模型之外的更丰富目标分布，但采样策略仍然效率低下。在本文中，我们提出了一个灵活且理论基础的框架，用于通过采样引导和扩展自回归LLMs。在此框架内，我们描述了两种算法——一种基于序贯蒙特卡洛（SMC），另一种基于副本交换（RE）——它们将生成过程引导向基础模型分布的幂、乘积或倾斜形式。我们通过在没有外部监督或奖励模型的情况下扩展LLMs的生成质量来展示这一框架。实验结果表明，我们的方法在扩展性上优于Best-of-N和标准MCMC基线。总体而言，本文提供了一种通过采样进行LLMs概率推断的系统性方法。

    arXiv:2608.26120v1 Announce Type: new  Abstract: Large Language Models (LLMs) are probabilistic models, typically defined by an autoregressive factorization. While recent work has begun to study richer target distributions beyond the base model, the sampling strategies remain highly inefficient. In this paper, we present a flexible and theoretically grounded framework for steering and scaling autoregressive LLMs with sampling. Within this framework, we describe two algorithms -- one based on Sequential Monte Carlo (SMC) and one based on Replica Exchange (RE) -- that steer generation toward powering, product or tilting of the base model distribution. We illustrate this framework through scaling the generation quality of LLMs without external supervision or reward models. Experimental results demonstrate our methods scale more favorably than Best-of-N and standard MCMC baselines. Overall, this paper offers a systematic recipe for probabilistic inference with LLMs via sampling.
    
[^150]: DeflectBench：评估大语言模型中修辞谬误生成的基准

    DeflectBench: A Benchmark for Evaluating Rhetorical Fallacy Generation in LLMs

    [https://arxiv.org/abs/2608.26119](https://arxiv.org/abs/2608.26119)

    该基准揭示了大型语言模型生成修辞谬误的拒绝行为主要由提示结构而非内容驱动，且某些提示框架可完全绕过安全限制。

    

    摘要：arXiv:2608.26119v1 公告类型：新 摘要：关于大型语言模型是否能被提示按需生成修辞谬误，以及当前的安全后训练是否限制这种行为，相比在现有文本中检测谬误的相关问题，受到的关注较少。我们通过DeflectBench填补了这一空白，评估了四个前沿模型在三种转移策略（whataboutism、ad hominem、red herring）、七种提示框架和跨越四个争议级别的80个主张下生成的23,990个输出。拒绝主要由请求结构而非主张内容决定。在80个主张中，每个主张的拒绝率仅变化11个百分点，而单个提示框架的变化可使模型内部拒绝率波动近100个百分点，切换请求的谬误类型在显式框架内可使拒绝率波动超过80个百分点。一个教育性辩论教练提示框架使四个模型家族的拒绝率降至接近零。

    arXiv:2608.26119v1 Announce Type: new  Abstract: Whether large language models can be prompted to generate rhetorical fallacies on demand, and whether current safety post-training constrains this behavior, has received less attention than the related question of detecting fallacies in existing text. We close this gap with DeflectBench, evaluating 23,990 generations from four frontier models across three deflection strategies (whataboutism, ad hominem, red herring), seven prompt framings, and 80 claims spanning four controversy levels. Refusal is governed primarily by request structure rather than claim content. Per claim refusal varies by only 11 percentage points across the 80 claims, while a single prompt frame change can swing within model refusal by nearly 100 percentage points and switching the requested fallacy type can swing it by over 80 percentage points within explicit framings. An educational debate coach prompt framing collapses refusal to near zero across all four model fa
    
[^151]: ElementCheck：通过句子元素进行复杂度感知的长文本事实性评估

    ElementCheck: Complexity-Aware Long-Form Text Factuality Evaluation via Sentence Elements

    [https://arxiv.org/abs/2608.26118](https://arxiv.org/abs/2608.26118)

    ElementCheck提出了一种基于句子元素图的复杂度感知验证框架，通过图拓扑结构估计句子复杂度并自适应调整验证粒度，解决了长文本事实性评估中固定分解和验证粒度导致的可靠性问题。

    

    现有长文本事实性评估依赖于“分解-检索-验证”流程。然而，该流程受到声明分解噪声和固定验证粒度的影响，导致结果不可靠。我们提出ElementCheck，一种通过句子元素验证长文本输出的复杂度感知框架。ElementCheck不将句子统一分解为原子子声明，而是提取原句中通过可验证连接明确关联的实体对作为元素，并将这些元素组织成元素图。图的拓扑结构为估计句子复杂度提供了结构信号，从而能够对简单句子进行直接验证，对复杂句子进行有针对性的元素级细化和验证。为支持细粒度评估，我们构建了一个新基准FastFact-Sent，将FastFact-Bench中的孤立声明映射回其源句子。实验...

    arXiv:2608.26118v1 Announce Type: new  Abstract: Existing long-form factuality evaluation relies on the decompose-retrieve-verify pipeline. However, the pipeline suffers from noise from claim decomposition and fixed verification granularity, resulting in unreliable results. We propose ElementCheck, a complexity-aware framework that verifies long-form outputs via sentence elements. Instead of uniformly decomposing sentences into atomic sub-claims, ElementCheck extracts entity pairs that are explicitly linked through verifiable connections in the original sentence as elements, and organizes these into an element graph. The graph topology provides a structural signal for estimating sentence complexity, enabling direct verification for simple sentences and targeted element-level refinement and verification for complex ones. To support fine-grained evaluation, we construct a new benchmark FastFact-Sent by mapping isolated claims from FastFact-Bench back to their source sentences. Experiment
    
[^152]: CIFQA：一种用于金融问答的确定性工具接地多智能体LLM框架

    CIFQA: A Deterministic Tool-Grounded Multi-Agent LLM Framework for Financial Query Answering

    [https://arxiv.org/abs/2608.26114](https://arxiv.org/abs/2608.26114)

    CIFQA通过多智能体分离语言理解与数值执行，结合确定性Python工具，解决了LLM在多步金融计算中的数值错误问题。

    

    arXiv:2608.26114v1 公告类型：新 摘要：计算密集型金融问答需要对结构化利率、时间条件、数值公式和基于规则的约束进行精确推理。尽管大型语言模型（LLMs）在自然语言任务上表现强劲，但在解决多步骤金融计算时，它们常常产生数值错误但看似合理的答案。为解决这一局限，我们引入了CIFQA（计算密集型金融问答），一种用于金融问答的确定性工具接地多智能体LLM框架。CIFQA通过分配专门智能体进行查询解释、路由、参数提取、计算规划和响应生成，将语言理解与数值执行分离，同时基于确定性的Python工具执行金融计算和规则应用。我们将CIFQA实例化用于定期存款问答，并在一个精选的定期存款基准上进行了评估。

    arXiv:2608.26114v1 Announce Type: new  Abstract: Calculation-intensive financial question answering requires exact reasoning over structured rates, temporal conditions, numerical formulas, and rule-based constraints. Although Large Language Models (LLMs) perform strongly on natural language tasks, they often produce numerically incorrect yet plausible answers when solving multi-step financial calculations. To address this limitation, we introduce CIFQA (Calculation-Intensive Financial Query Answering), a deterministic tool-grounded multi-agent LLM framework for financial question answering. CIFQA separates language understanding from numerical execution by assigning specialized agents to query interpretation, routing, parameter extraction, computation planning, and response generation, while deterministic Python-based tools perform financial calculations and rule application. We instantiate CIFQA for fixed deposit query answering and evaluate it on a curated benchmark of fixed deposit 
    
[^153]: TreeGraft：基于树的投机解码的自适应多草稿器嫁接方法

    TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding

    [https://arxiv.org/abs/2608.26112](https://arxiv.org/abs/2608.26112)

    我们提出了TreeGraft，一种自适应多草稿器嫁接框架，通过结合不同成本的草稿器来构建共享草稿树，从而平衡草稿质量与延迟，提升基于树的投机解码效率。

    

    投机解码通过“先草稿后验证”的范式加速大型语言模型的推理。在此基础上，树结构方法通过将提议组织成多个候选路径来提升推理效率，从而增加接受长度。然而，现有的树结构方法在所有草稿步骤中使用单一草稿器，这造成了一个困境：较小的草稿器速度快但生成的树质量较低，而较大的草稿器能提高树质量但延迟较高。为了解决这个问题，我们提出了TreeGraft，一个多草稿器框架，其中不同成本的草稿器共同构建一个共享的草稿树。TreeGraft使用更强的草稿器通过更新较弱草稿器分配的分数来重新评分候选，重新选择嫁接位置，并恢复未被探索的有前景路径。它还以非破坏性方式整合更强草稿器的扩展，保留可能仍被目标模型接受的现有分支。

    arXiv:2608.26112v1 Announce Type: new  Abstract: Speculative decoding accelerates large language model inference through a draft-then-verify paradigm. Building on this, tree-structured methods improve inference by organizing proposals into multiple candidate paths, increasing the accepted length. However, existing tree-structured methods use a single drafter for all drafting steps, creating a dilemma: a smaller drafter is fast but yields lower-quality trees, whereas a larger drafter improves tree quality but suffers from high latency. To address this, we propose TreeGraft, a multi-drafter framework in which drafters of different costs jointly construct a shared draft tree. TreeGraft uses the stronger drafter to rescore candidates by updating scores assigned by the weaker drafter, reselect grafting positions, and recover promising paths left unexplored. It also integrates stronger drafter expansions non-destructively, preserving existing branches that may still be accepted by the target
    
[^154]: VISA：面向多模态指令跟随的智能体自进化数据合成

    VISA: Agentic Self-Evolving Data Synthesis for Multimodal Instruction Following

    [https://arxiv.org/abs/2608.26013](https://arxiv.org/abs/2608.26013)

    VISA提出了一种自进化智能体循环，通过利用失败样本、验证器反馈和目标模型错误来动态优化多模态指令合成，显著提升数据质量与难度。

    

    arXiv:2608.26013v1 公告类型：新 摘要：多模态指令跟随模型需要准确、多样、可验证且具有挑战性的训练数据。现有的合成流程通常遵循一次性生成和过滤的范式，丢弃了来自失败样本、验证器结果和目标模型错误的反馈。我们提出了VISA（视觉指令合成智能体），一种将多模态指令合成重构为自进化循环的智能体框架。在每一轮中，VISA分析图像以过滤不兼容的约束并发现新的可验证约束，从持久记忆中采样多样性和难度感知的约束集，生成候选指令，并使用可执行工具和结构化大型语言模型评判器验证生成的样本。失败样本触发诊断引导的恢复，而接受的样本则针对目标模型进行探测以估计难度。由此产生的验证器信号和目标模型失败模式被反馈到后续轮次，推动数据分布向更高难度和更高质量演化。在多个基准测试上的实验表明，VISA在指令跟随准确性和多样性方面持续优于现有基线，仅使用生成数据的子集即可实现最先进的性能。

    arXiv:2608.26013v1 Announce Type: new  Abstract: Multimodal instruction-following models require training data that is accurate, diverse, verifiable, and challenging. Existing synthesis pipelines typically follow a one-pass generate-and-filter paradigm, discarding feedback from failed samples, verifier outcomes, and target-model errors. We present VISA (Visual Instruction Synthesis Agent), an agentic framework that reformulates multimodal instruction synthesis as a self-evolving loop. At each round, VISA analyzes an image to filter incompatible constraints and discover new verifiable ones, samples diversity- and difficulty-aware constraint sets from persistent memory, generates candidate instructions, and verifies the resulting samples with executable tools and structured large language model judges. Failed samples trigger diagnostic-guided recovery, while accepted samples are probed against the target model to estimate difficulty. The resulting verifier signals and target-model failur
    
[^155]: 揭示无训练大语言模型文本检测中的频谱机制

    Unveiling Spectral Mechanisms in Training-Free LLM Text Detection

    [https://arxiv.org/abs/2608.25944](https://arxiv.org/abs/2608.25944)

    本文从理论和实证角度揭示了无训练文本检测中频谱分析的作用机制，指出频谱能量捕捉人类写作的“生成活力”，且其信号强度在长、连续、受约束的生成中最明显。

    

    arXiv:2608.25944v1 公告类型：新 摘要：大语言模型（LLMs）的快速发展使得区分人类写作与机器生成文本变得越来越困难。无训练检测提供了一种可扩展的解决方案，然而常见的基于置信度的指标主要衡量平均令牌概率，往往忽略了表征人类写作的信号波动，我们称之为“生成活力”。频谱分析提供了一种捕捉这种活力的方法，但其机制和实际边界仍未得到充分探索。在本文中，我们从理论和实证两个角度分析频谱检测。我们将频谱能量与代理对数概率轨迹的方差联系起来，并解释了人类更广泛的令牌选择如何产生频域指标所利用的波动。我们进一步表明，这种信号的强度取决于文本长度和采样范围：频谱证据在长、连续、受约束的生成中最为清晰。

    arXiv:2608.25944v1 Announce Type: new  Abstract: The rapid advancement of Large Language Models (LLMs) makes it increasingly difficult to distinguish human writing from machine-generated text. Training-free detection offers a scalable solution, yet common confidence-based metrics mainly measure average token probabilities and often miss the signal fluctuations that characterize human writing, which we call "generative vitality". Spectral analysis offers a way to capture this vitality, but its mechanism and practical boundaries remain underexplored. In this paper, we analyze spectral detection from both theoretical and empirical perspectives. We connect spectral energy to variance in proxy log-probability trajectories and explain how broader human token choices create the fluctuations used by frequency-domain indicators. We further show that the strength of this signal depends on text length and sampling range: spectral evidence is clearest for long, continuous, constrained generation, 
    
[^156]: 使用QLoRA学习新事实：获取-保留边界

    Learning New Facts with QLoRA: An Acquisition-Retention Frontier

    [https://arxiv.org/abs/2608.25677](https://arxiv.org/abs/2608.25677)

    本文通过系统比较不同秩的QLoRA与全量微调，揭示了参数高效微调中秩的选择在事实获取与能力保留之间存在明确的权衡边界，其中低秩保留能力但获取少，高秩获取多但损失能力，全量微调则作为保守基线。

    

    arXiv:2608.25677v1 公告类型：新  摘要：参数高效微调通常被认为能保留预训练能力，因为它仅更新少量参数。我们表明，这一假设在很大程度上依赖于适配器的容量。我们在一个受控的、基于OpenStreetMap的基准测试中研究了事实获取，其中Qwen3-4B必须获取匿名的地理关联，同时保留无关的能力。通过将全量微调（FFT）与量化低秩适配（QLoRA）在秩为8、16、32和64的情况下进行比较，我们发现秩引出了一个清晰的获取-保留边界。低秩QLoRA保留了域外（OOD）性能，但获取的事实较少，而较高秩提高了同事实释义的泛化能力，但以在无关基准上性能下降为代价。FFT表现为一个保守的基线：它很好地保留了通用能力，但未达到最高的事实获取状态。分布、权重空间和谱诊断...

    arXiv:2608.25677v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning is often assumed to preserve pretrained capabilities because it updates only a small number of parameters. We show that this assumption depends strongly on adapter capacity. We study factual acquisition in a controlled OpenStreetMap-derived benchmark where Qwen3-4B must acquire anonymized geographic associations while retaining unrelated capabilities. Comparing full fine-tuning (FFT) with quantized low-rank adaptation (QLoRA) at ranks 8, 16, 32, and 64, we find that rank induces a clear acquisition--retention frontier. Low-rank QLoRA preserves out-of-domain (OOD) performance but acquires fewer facts, whereas higher ranks improve same-fact paraphrase generalization at an increasing cost in performance on unrelated benchmarks. FFT behaves as a conservative baseline: it retains general capabilities well, but does not reach the highest factual-acquisition regime. Distributional, weight-space, and spectral diag
    
[^157]: 采样标记反向KL同策略蒸馏的标记级分析

    A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation

    [https://arxiv.org/abs/2608.25643](https://arxiv.org/abs/2608.25643)

    本文通过分析同策略蒸馏中逐标记梯度的非均匀性，提出了一种轻量级的惊喜感知重新加权（SuRe）方法，以有效增强低概率标记的权重，从而提升蒸馏性能。

    

    arXiv:2608.25643v1 公告类型：交叉 摘要：同策略蒸馏（OPD）通过来自冻结教师模型的标记级信号，在学生自身的轨迹上监督学生，但采样损失如何在标记间分配更新仍鲜为人知。我们分析了反向KL的逐标记K2估计量相对于学生logits的梯度。该梯度的$\ell_1$范数分解为教师-学生对数概率差的绝对值和一个学生端softmax因子，后者随着采样标记在学生下概率降低而增大。在我们的数学蒸馏实验中，这些逐标记范数高度非均匀：低学生概率标记占其总和的过大比例，并且在大的教师-学生差距中富集。作为这项分析提出的轻量级干预，我们研究了惊喜感知重新加权（SuRe），一种分离的、有界的加权规则，进一步放大这种现有分配。在两个Qwen3学生规模中，SuRe

    arXiv:2608.25643v1 Announce Type: cross  Abstract: On-policy distillation (OPD) supervises a student on its own trajectories with token-level signals from a frozen teacher, yet how a sampled loss allocates updates across tokens remains poorly understood. We analyze the gradient of the per-token K2 estimator of reverse KL with respect to the student logits. The $\ell_1$ norm of this gradient factorizes into the absolute teacher--student log-probability gap and a student-side softmax factor that grows as the sampled token becomes less likely under the student. In our math-distillation runs, these per-token norms are highly non-uniform: low-student-probability tokens account for a disproportionate share of their sum and are also enriched in large teacher--student gaps. As a lightweight intervention suggested by this analysis, we study Surprise-aware Reweighting (SuRe), a detached, bounded weighting rule that further amplifies this existing allocation. Across two Qwen3 student scales, SuRe
    
[^158]: 当过时约束未被检查：继承代理记忆中的预算验证失败

    When Stale Constraints Go Unchecked: Budgeted Verification Failures in Inherited Agent Memory

    [https://arxiv.org/abs/2608.25553](https://arxiv.org/abs/2608.25553)

    该论文研究了在有限验证预算下，代理继承的过时约束未被检查导致验证失败的问题，并提出了通过重新分配验证槽位来减少这类错误的方法。

    

    arXiv:2608.25553v1 公告类型：交叉 摘要：一个继承了整合记忆的代理可能继承了一个在写入时成立但已被更新的权威记录撤销的约束。在稀缺的验证预算下，代理能否恢复该撤销？如果不能，这种错误是否能在不增加支出的情况下避免？我们明确建模了替代关系——历史来源是不可变的；变化的是哪个记录是当前的——并设计性地分配了记忆的形式、世界状态（来源当前或已被替代）以及固定预算为两条记录的验证策略：代理自身的分配，或相同预算但将一个槽位重新分配给关键来源路径或随机记录。在声明约束的情况下，代理在大约五分之一的回合中检查了其来源路径；当该约束已被替代时，原生分配在主要运行、新措辞等中分别产生了77.3%、74.7%和74.7%的回合中的过时一致决策。

    arXiv:2608.25553v1 Announce Type: cross  Abstract: An agent that inherits a consolidated memory may inherit a constraint that was true when written and has since been withdrawn by a newer authoritative record. Under a scarce verification budget, does the agent recover the withdrawal, and if not, is the error avoidable without spending more? We model supersession explicitly -- historical provenance is immutable; what changes is which record is current -- and assign by design the memory's form, the world's state (source current or superseded), and the verification policy at a fixed budget of two records: the agent's own allocation, or the same budget with one slot re-assigned to the critical provenance path or to a random record. With a constraint stated, agents inspected its provenance path in about one episode in five; when that constraint had been superseded, native allocation produced stale-consistent decisions in 77.3%, 74.7% and 74.7% of episodes across a primary run, a fresh-wordi
    
[^159]: ClueWeaver：面向紧凑型大语言模型在长篇文学叙事中的奖励引导双智能体证据推理

    ClueWeaver: Reward-Guided Dual-Agent Evidence Reasoning for Compact LLMs on Literary Long Narratives

    [https://arxiv.org/abs/2608.25531](https://arxiv.org/abs/2608.25531)

    ClueWeaver提出了一种奖励引导的双智能体框架，通过“查找者”与“解释者”的协作，使紧凑型本地模型在长篇文学叙事问答中实现高效、可检查且证据忠实的推理。

    

    arXiv:2608.25531v1 公告类型：新论文 摘要：人文与社会科学研究需要对小说、剧本、档案和案例报告等长篇叙事材料进行细读，但许多用户难以获取昂贵的专有长上下文模型。紧凑型、可本地部署的语言模型是一种实用的替代方案，但直接将整个长上下文输入这些模型仍然成本高昂、难以检查，且容易遗漏稀疏证据。我们提出了ClueWeaver，一种面向紧凑型本地模型的证据感知双智能体框架，用于长篇叙事问答。一个“查找者”智能体通过检索引导的分段识别包含答案关键线索的段落，而一个“解释者”智能体从所选证据中推导答案，生成带有段落ID引用的推理过程，并对高风险问题应用内部自校准步骤。两个智能体均通过奖励引导的强化学习进行优化：查找者的奖励强调证据保留和忠实性。

    arXiv:2608.25531v1 Announce Type: new  Abstract: Humanities and social science research requires close reading of long narrative materials such as novels, scripts, archives, and case reports, yet many users have limited access to costly proprietary long-context models. Compact, locally deployable language models are a practical alternative, but directly feeding them an entire long context remains costly, hard to inspect, and prone to missing sparse evidence. We present ClueWeaver, an evidence-aware dual-agent framework for long-narrative question answering with compact local models. A Finder identifies passages containing answer-critical clues through retrieval-guided segmentation, while an Interpreter derives the answer from the selected evidence, produces rationales with paragraph-ID citations, and applies an internal self-calibration pass for high-risk questions. Both agents are optimized with reward-guided reinforcement learning: Finder rewards emphasize evidence retention and fait
    
[^160]: DataKernelBench：大语言模型能否在GPU上优化数据库查询？

    DataKernelBench: Can LLMs Optimize Database Queries on GPUs?

    [https://arxiv.org/abs/2608.25061](https://arxiv.org/abs/2608.25061)

    本文提出了DataKernelBench基准测试，证明大语言模型能通过执行引导修复在GPU上优化数据库查询，最强配置实现2.11倍加速，并发现内核融合、完整查询专业化和工作负载上下文是关键因素。

    

    arXiv:2608.25061v1 公告类型：新 摘要：GPU日益加速数据库系统，但查询特定的峰值性能通常仍依赖手写内核。现有的大语言模型内核基准测试专注于机器学习算子，未测试不规则、异构、数据移动密集型的数据库风格算子。我们引入了DataKernelBench，它将SQL转换为经过验证的PyTorch TorchPlan程序，并评估大语言模型通过执行引导修复来优化核心张量受限片段或完整查询（使用CUDA或Triton）的能力。在TPC-H SF10数据集和H100 GPU上对十个专有和开源权重模型进行的测试中，最强的完整查询CUDA配置在完全通过率下比torch.compile实现了$2.11\times$的加速。我们发现，性能更高的实现通常采用内核融合和执行策略更改，更强的模型从完整查询专业化中获益最多，且工作负载上下文比硬件上下文更重要。为处理大数据...

    arXiv:2608.25061v1 Announce Type: new  Abstract: GPUs increasingly accelerate database systems, but query-specific peak performance still often relies on hand-written kernels. Existing LLM kernel benchmarks focus on machine learning operators, leaving irregular, heterogeneous, data-movement-heavy database-style operators untested. We introduce DataKernelBench, which translates SQL into validated PyTorch TorchPlan programs and evaluates LLMs that optimize either the core tensor-bounded snippet or the full query in CUDA or Triton through execution-guided repair. Across ten proprietary and open-weight models on TPC-H SF10 with an H100 GPU, the strongest full-query CUDA configuration achieves $2.11\times$ speedup over torch.compile at full pass rate. We find that higher-performing implementations commonly use kernel fusion and execution-strategy changes, stronger models benefit most from full-query specialization, and workload context matters more than hardware context. To handle data larg
    
[^161]: 基础模型的无监督后训练：一项综述

    Unsupervised Post-Training of Foundation Models: A Survey

    [https://arxiv.org/abs/2608.24982](https://arxiv.org/abs/2608.24982)

    本文综述了80种无监督后训练方法，提出按内部信号源分类，并揭示了信号选择与任务结构决定模型改进或错误放大的关键机制。

    

    arXiv:2608.24982v1 公告类型：新 摘要：基础模型的后训练通常依赖于人类标签、偏好数据、更强的教师模型或可执行的验证器。我们研究无监督后训练（UPT）：在未标记输入上进行的有更新适应，其学习信号源自同源模型产物而非外部预言机。我们整理了80种严格的UPT方法，并根据提供更新信号的对象对其进行分类：预测统计量、样本关系、自生成目标或内部评估器。除了清单之外，我们展示了内部信号和任务结构的选择如何决定后训练是改进模型还是递归放大错误。一个正交的输入可见性×更新持久性视角映射了部署场景，并定义了UPT选择和评估的统一框架。

    arXiv:2608.24982v1 Announce Type: new  Abstract: Foundation-model post-training usually relies on human labels, preference data, stronger teachers, or executable verifiers. We study Unsupervised Post-Training (UPT): update-bearing adaptation on unlabeled inputs whose learning signal is derived from same-lineage model artifacts rather than an external oracle. We catalog 80 strict UPT methods and organize them by the object that supplies the update signal: a prediction statistic, a sample relation, a self-generated target, or an internal evaluator. Beyond inventory, we show how the choice of internal signal and task structure determines whether post-training improves the model or recursively amplifies error. An orthogonal Input Visibility $\times$ Update Persistence view maps deployment regimes and defines a unified framework for UPT selection and evaluation.
    
[^162]: 检索但不可靠：检索增强生成中的攻击与防御综述

    Retrieved But Not Reliable: A Survey on Attacks, and Defenses in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2608.24977](https://arxiv.org/abs/2608.24977)

    本综述统一了检索增强生成全流水线中的攻击与防御框架，按准确性、隐私和公平性分类威胁模型，并覆盖各阶段防御策略。

    

    检索增强生成（RAG）通过将输出基于外部知识来增强大型语言模型，提高事实准确性并减少幻觉。同时，检索增强流水线引入了新的鲁棒性和安全风险，包括语料库投毒、后门攻击、隐私泄露和公平性违规。尽管该领域进展迅速，现有综述在处理攻击者目标、威胁模型以及整个RAG流水线中各阶段特定防御方面仍存在局限性。本综述提供了统一且流水线感知的RAG鲁棒性概述。我们形式化了语料库、检索器和生成器上的威胁模型，并将攻击组织为三个主要目标：准确性、隐私和公平性。我们进一步从流水线感知的角度审查防御措施，涵盖检索、重排序、生成和追溯阶段。此外，我们总结了鲁棒性基准并进行了说明。

    arXiv:2608.24977v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) enhances large language models by grounding outputs in external knowledge, improving factuality and reducing hallucinations. At the same time, the retrieval-augmented pipeline introduces new robustness and security risks, including corpus poisoning, backdoor attacks, privacy leakage, and fairness violations. Despite rapid progress in this area, existing surveys remain limited in their treatment of attacker objectives, threat models, and stage-specific defenses across the full RAG pipeline. This survey presents a unified and pipeline-aware overview of RAG robustness. We formalize threat models over the corpus, retriever, and generator, and organize attacks into three main objectives: accuracy, privacy, and fairness. We further review defenses from a pipeline-aware perspective, covering the retrieval, rerank, generation, and traceback stages. In addition, we summarize robustness benchmarks and explain
    
[^163]: RePolicy：智能体安全防护中安全策略调用的强化学习

    RePolicy: Reinforcement Learning for Safety-Policy Invocation in Agent Safeguards

    [https://arxiv.org/abs/2608.24275](https://arxiv.org/abs/2608.24275)

    RePolicy通过强化学习动态调用安全策略，实现了对未见轨迹和变化策略上下文的高鲁棒性安全检测。

    

    摘要：arXiv:2608.24275v1 公告类型：新 摘要：保护语言模型智能体需要在上下文相关的安全策略下评估完整的执行轨迹。现有的策略感知防护主要依赖于提示或监督微调，限制了其适应未见轨迹和变化策略上下文的能力。我们提出了RePolicy，一种通过强化学习学习安全策略调用的智能体防护方法。给定智能体轨迹和动态策略库，RePolicy调用适用策略，并利用其内容生成基于策略的理由和安全判断。我们构建了PolicyTraj-20K以支持监督初始化，随后使用GRPO结合可验证奖励和策略上下文扰动。在六个智能体安全基准上的实验表明，RePolicy在变化策略上下文下实现了强大的整体安全检测性能和稳健的策略调用。

    arXiv:2608.24275v1 Announce Type: new  Abstract: Safeguarding language model agents requires assessing complete execution trajectories under context-dependent safety policies. Existing policy-aware safeguards mainly rely on prompting or supervised fine-tuning, limiting their ability to adapt to unseen trajectories and changing policy contexts. We propose RePolicy, an agent safeguard that learns safety-policy invocation through reinforcement learning. Given an agent trajectory and a dynamic policy library, RePolicy invokes the applicable policy and uses its content to produce a policy-grounded rationale and safety judgment. We construct PolicyTraj-20K to support supervised initialization, followed by GRPO with verifiable rewards and policy-context perturbation. Experiments across six agent safety benchmarks show that RePolicy achieves strong overall safety-detection performance and robust policy invocation under varying policy contexts.
    
[^164]: FireRedAudio：一种具有解耦连续表示的通用的音频语言模型，用于理解与生成

    FireRedAudio: A General-Purpose Audio Language Model with Decoupled Continuous Representations for Understanding and Generation

    [https://arxiv.org/abs/2608.24168](https://arxiv.org/abs/2608.24168)

    FireRedAudio是首个在单个可训练的自动回归LLM中，为理解和生成提供分离连续输入表示的公开统一音频语言模型，通过解耦表示实现了语音识别、理解、翻译、编辑和合成等多个任务的最优性能。

    

    arXiv:2608.24168v2 公告类型：替换 摘要：一个统一的音频模型必须能够识别和理解语言、副语言和环境信息，同时支持语音合成和编辑。一个关键挑战是表示：理解任务倾向于使用紧凑的特征以适合长上下文建模，而语音生成则需要可重构的特征以保留细粒度的声学细节。我们引入了FireRedAudio，一个具有共享90亿参数大型语言模型（LLM）的通用的音频语言模型。据我们所知，它是首个公开披露的统一音频语言模型，在单个可训练的自动回归LLM内为理解和生成提供分离的连续输入表示。要识别或分析的音频由专门的音频编码器处理，而用于生成的语音输入则使用基于RedAE的路径。该LLM直接生成文本或条件化流匹配DiT以产生连续的声学潜在表示。通过预训练和指令调优，FireRedAudio在语音识别、理解、对话、翻译、语音编辑和文本到语音合成方面展示了强大的性能，在多个基准测试中取得了最先进的结果。

    arXiv:2608.24168v2 Announce Type: replace  Abstract: A unified audio model must recognize and understand linguistic, paralinguistic, and environmental information while supporting speech synthesis and editing. A key challenge is representation: understanding favors compact features suited to long-context modeling, whereas speech generation requires reconstructible features that preserve fine-grained acoustic detail. We introduce FireRedAudio, a general-purpose audio language model with a shared 9B-parameter LLM. To the best of our knowledge, it is the first publicly disclosed unified audio-language model to provide separate continuous input representations for understanding and generation within a single trainable autoregressive LLM. Audio to be recognized or analyzed is processed by a dedicated Audio Encoder, while speech inputs for generation use a RedAE-based pathway. The LLM directly generates text or conditions a flow-matching DiT to produce continuous acoustic latents. Through pr
    
[^165]: 当青年进入对话：基于LLM的学生谈话测量验证中的认知转变

    When Youth Enter The Chat: An Epistemic Shift in the Validation of LLM-Based Measures of Student Talk

    [https://arxiv.org/abs/2608.23780](https://arxiv.org/abs/2608.23780)

    本文提出，验证基于LLM的学生谈话测量时，应通过重新情境化对话和让青年参与研究来共享认知权威，以确保测量对边缘化青年有意义且公平。

    

    arXiv:2608.23780v1 公告类型：交叉 摘要：大型语言模型（LLMs）越来越多地被用于大规模测量学生话语的各个方面（例如，谈话动作、协作、声音公平性）。通常，基于LLM的学生谈话测量使用仅包含口头贡献的课堂对话转录，这使学生的语言脱离上下文。验证这些测量的常见实践包括将输出与成人的专家注释进行比较，使用留出评估集和F1分数。我们认为，这些方法不足以确保这些测量对教学和学习具有意义和公平性，特别是对于种族和语言边缘化的青年。为了以被分析谈话的青年为中心，需要重新情境化这些课堂对话，并让青年参与研究过程。与青年共享认知权威，最终以他们的视角为中心，并为分析增添关键细微差别。

    arXiv:2608.23780v1 Announce Type: cross  Abstract: LLMs are being used increasingly to measure aspects of student discourse (e.g. talk moves, collaboration, equity of voice) at scale. Typically, LLM-based measures of student talk use transcriptions of classroom conversations that only include verbal contributions, which de-contextualize student language. Common practices for validating these measures include comparing outputs against expert annotations by adults, using held out evaluation sets and F1 scores. We argue that these approaches are insufficient to ensure that such measures are meaningful and equitable for teaching and learning, particularly for racially and linguistically marginalized youth. In order to center the youth whose talk is being analyzed, re-contextualizing these classroom conversations and engaging youth in the research process is necessary. Sharing epistemic authority with youth, ultimately, centers their point of view and adds crucial nuance to the analysis of 
    
[^166]: 大型语言模型创造力自动评估的局限性

    The Limits of Automatic Evaluation of Creativity in Large Language Models

    [https://arxiv.org/abs/2608.23705](https://arxiv.org/abs/2608.23705)

    本研究发现，现有的自动评估方法，尤其是基于LLM的评判者，无法可靠地匹配人类对创造力的判断，并表现出对AI生成文本的系统性偏好。

    

    大型语言模型（LLM）在需要创造力的领域中生成文本的能力日益增强，甚至挑战人类表现，但评估LLM生成内容中的创造力仍是一项重大挑战。在此，我们研究当前自动评估方法是否能可靠地捕捉人类对创造力的判断。我们从WritingPrompts数据集中收集了人类对人工和AI生成的短篇小说在11个创造力维度上的评估，并将这些判断与自动化客观指标和LLM作为评判者的评估进行比较。我们的实验揭示了自动评估与人类评估之间的显著不一致。特别是，基于LLM的评判者表现出对AI生成故事的系统性偏好，始终偏爱其风格特征，而非人类作者文本的不可预测性及其他品质。此外，相关性分析显示，广泛使用的自动评估指标与人类判断的关联有限。

    arXiv:2608.23705v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly capable of generating text that challenges human performance in domains requiring creativity, yet evaluating creativity in LLM-generated content remains a significant challenge. Here, we investigate whether current automatic evaluation methods can reliably capture human judgments of creativity. We collect human evaluations of human- and AI-generated short stories from the WritingPrompts dataset across 11 dimensions of creativity, and compare these judgments with automated objective metrics and LLM-as-a-Judge evaluations. Our experiments reveal substantial misalignment between automatic evaluations and human assessments. In particular, LLM-based judges exhibit a systematic preference for AI-generated stories, consistently favoring their stylistic characteristics over the unpredictability and other qualities of human-authored texts. Furthermore, correlation analyses show that widely used auto
    
[^167]: 语言链在对齐中的应用：跨语言排序偏好优化

    Language Chain in Alignment: Cross-Lingual Ranking Preference Optimization

    [https://arxiv.org/abs/2608.23149](https://arxiv.org/abs/2608.23149)

    本文提出跨语言排序偏好优化（CRPO）框架，通过利用英语偏好知识的分层结构，在目标语言中实现更优的语言对齐和输出质量。

    

    arXiv:2608.23149v1 公告类型：交叉 摘要：大型语言模型的对齐在很大程度上依赖于以英语为中心的高质量偏好数据，这通常会导致在其他语言中的性能欠佳。在本文中，我们提出了跨语言排序偏好优化（CRPO），这是一个新颖的框架，利用来自英语的稳健偏好知识来促进目标语言中的偏好对齐。我们在目标语言和英语之间的平行偏好对中设计了一种分层结构，以联合优化语言内和语言间的偏好，从而增强语言适应性和输出质量。基于LambdaLoss框架，CRPO超越了基于二元比较的优化，通过提供多个候选响应之间的相对排序信号。我们在五种资源规模不同的语言上进行的实验表明，CRPO在指令遵循和知识方面均持续优于标准方法。

    arXiv:2608.23149v1 Announce Type: cross  Abstract: The alignment of Large Language Models heavily relies on English-centric high-quality preference data, which often leads to suboptimal performance in other languages. In this paper, we propose Cross-Lingual Ranking Preference Optimization (CRPO), a novel framework that leverages robust preference knowledge from English to facilitate preference alignment in the target language. We design a hierarchical structure within parallel preference pairs across the target language and English to jointly optimize intra- and inter-lingual preferences, thereby enhancing language adaptation and output quality. Building on the LambdaLoss framework, CRPO goes beyond the binary comparison based optimization by providing a relative ranking signal across multiple candidate responses. Our experiments across five languages with varying resource scales demonstrate that CRPO consistently outperforms standard approaches in both instruction-following and knowle
    
[^168]: 深陷文本债务：面向MLLM智能体的视觉证据保留上下文剪枝

    Buried in Textual Debt: Context Pruning with Visual Evidence Preservation for MLLM Agents

    [https://arxiv.org/abs/2608.22963](https://arxiv.org/abs/2608.22963)

    本文提出SPARE框架，通过KL散度引导的上下文剪枝，在保留视觉证据的同时移除冗余推理文本，以解决MLLM智能体长轨迹中的“文本债务”问题。

    

    arXiv:2608.22963v1 公告类型：新论文 摘要：多模态大语言模型（MLLMs）正越来越多地被部署为多步智能体，其中显式推理支持任务分解和工具协调，但也会积累自生成的文本。在长轨迹中，这些文本可能主导上下文并抑制视觉证据，造成“文本债务”。我们观察到，一旦任务相关的视觉证据被锚定，推理就变得冗余，而当锚定仍不确定时，过时的假设可能会误导后续推理。因此，剪枝必须移除冗余文本，同时不丢弃视觉证据。我们提出SPARE，一种基于库尔贝克-莱布勒（KL）散度引导的框架，用于剪枝多模态工具使用智能体中积累的推理内容。SPARE使用紧凑的任务状态摘要作为特权诊断上下文。对于每个候选段，它在原始上下文和摘要条件下的上下文中重放相同模型。然后，从策略内自蒸馏（OPSD）的反向KL散度指导剪枝决策。

    arXiv:2608.22963v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) are increasingly deployed as multi-step agents, where explicit reasoning supports task decomposition and tool coordination but also accumulates self-generated text. Over long trajectories, this text can dominate the context and suppress visual evidence, creating textual debt. We observe that reasoning becomes redundant once task-relevant visual evidence is grounded, while stale hypotheses can misguide later inference when grounding remains uncertain. Pruning must therefore remove redundant text without discarding visual evidence. We propose SPARE, a Kullback--Leibler (KL)-guided framework for pruning accumulated reasoning in multimodal tool-use agents. SPARE uses a compact task-state summary as privileged diagnostic context. For each candidate segment, it replays the same model under the original and summary-conditioned contexts. Reverse-KL divergence from on-policy self-distillation (OPSD) then t
    
[^169]: ExecRubrics：可执行工具增强的评分标准，用于可验证且高效的长篇评估

    ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation

    [https://arxiv.org/abs/2608.22559](https://arxiv.org/abs/2608.22559)

    ExecRubrics通过将评分标准转化为可执行的Python函数，实现了可验证、高效且能捕捉复杂依赖关系的长篇评估，替代了昂贵的黑盒LLM评判器。

    

    摘要：arXiv:2608.22559v1 公告类型：新 摘要：评分标准旨在通过将回答质量分解为可解释的准则，使语言模型评估透明化。然而，自然语言评分标准往往含糊不清，需要黑盒LLM评判器，并且通常假设准则通过线性加权和独立聚合，这限制了其捕捉依赖关系、替代方案、惩罚和覆盖条件的能力。我们提出ExecRubrics，一个将评分标准表示为紧凑可执行程序的框架。ExecRubrics将评估逻辑编码为可验证的Python评分函数，赋予自然语言评分标准意图一种操作语义：一个可检查、可执行和可编辑的固定决策程序。在三个长篇回答基准测试——HealthBench、HelpSteer和ArgQuality上，我们展示了ExecRubrics可以替代昂贵的黑盒评判器，在偏好排序中优于或匹配自然语言评分标准基线，具有最佳偏好性能。

    arXiv:2608.22559v1 Announce Type: new  Abstract: Rubrics aim to make language-model evaluation transparent by decomposing response quality into interpretable criteria. However, natural-language rubrics are often ambiguous, require black-box LLM judges, and typically assume criteria aggregate independently through linear weighted sums, limiting their ability to capture dependencies, alternatives, penalties, and override conditions. We propose ExecRubrics, a framework for representing rubrics as compact executable programs. ExecRubrics encodes evaluation logic as verifiable Python scoring functions, giving natural-language rubric intent an operational semantics: a fixed decision procedure that can be inspected, executed, and edited. On three long-form response benchmarks-HealthBench, HelpSteer, and ArgQuality-we show that ExecRubrics can substitute for expensive black-box judges in ranking preferred over dispreferred responses, matching or improving NL rubric baselines with best preferen
    
[^170]: 单独对齐，集体错位：预测LLM智能体群体中的对抗性俘获

    Aligned Alone, Misaligned Together: Forecasting Adversarial Capture in LLM Agent Populations

    [https://arxiv.org/abs/2608.22444](https://arxiv.org/abs/2608.22444)

    该论文发现单个LLM智能体的校准行为无法预测其群体行为，但通过良性操作数据可校准响应函数，从而提前预测对抗性俘获。

    

    arXiv:2608.22444v1 公告类型：新 摘要：AI安全评估的单位仍是单个模型，然而语言模型智能体正越来越多地部署在相互交互的群体中，这些群体读取并撰写彼此的决策。这提出了一个单智能体审计无法回答的问题：一个自身校准良好的智能体，仍可能被周围智能体拉向不同的决策。我们在一个安全分诊任务上研究此问题，其中语言模型监控器群体决定是否升级或忽略警报，并且我们可以注入一个始终推动某一方向的坚定少数派。我们发现，单个智能体在独自判断时几乎相同的两个警报，可以驱动集体行为产生巨大分歧，因此审计任何单个成员不一定能揭示群体将做什么。然而，这种集体行为可以提前预测。仅从群体良性、无对抗的操作中，我们校准一个响应函数，该函数在任何攻击运行之前就能预测结果。

    arXiv:2608.22444v1 Announce Type: new  Abstract: The unit of AI safety evaluation is still the individual model, yet language-model agents are increasingly deployed in interacting populations that read and write one another's decisions. This raises a question no single-agent audit can answer: an agent that is well-calibrated on its own may still be pulled toward a different decision by the agents around it. We study this on a security-triage task, where populations of language-model monitors decide whether to escalate or dismiss alerts, and into which we can inject a committed minority that always pushes one way. We find that two alerts a single agent judges almost identically on its own can drive collective behavior far apart, so auditing any one member need not reveal what the population will do. Yet that collective behavior can be predicted in advance. From a population's benign, adversary-free operation alone, we calibrate a response function that forecasts, before any attack is ru
    
[^171]: 物理推理的解耦物理建模与执行

    Decoupled Physical Modeling and Execution for Physics Reasoning

    [https://arxiv.org/abs/2608.22126](https://arxiv.org/abs/2608.22126)

    该论文提出了一种解耦物理建模与执行的统一框架，通过两阶段后训练策略（监督微调加规则反馈强化学习）提炼中间表示，从而提升大型语言模型在物理推理任务上的表现。

    

    物理推理需要构建底层物理系统的一致模型，而非仅依赖符号或基于公式的操纵。尽管大型语言模型在解决数学和编程问题方面表现出强大能力，但它们在物理问题上仍面临挑战，因为这些问题的物理建模过程与数学计算相互交织。人类处理物理问题的方式是先构建系统表示，再进行计算。受此启发，我们引入了一个统一框架，该框架提炼出明确编码物理建模过程的中间表示，并采用两阶段后训练策略，其中监督微调建立结构化建模，而基于规则反馈的强化学习则提高建模过程的质量。在多个多模态物理基准上的实验表明，我们的方法带来了持续的改进。

    arXiv:2608.22126v1 Announce Type: cross  Abstract: Physics reasoning requires constructing a consistent model of the underlying physical system rather than relying solely on symbolic or formula-based manipulation. Although large language models have shown strong ability in solving math and coding problems, they still struggle with physics problems, as these problems entangle the physical modeling process with mathematical calculations. Humans approach physics by first building a representation of the system before performing calculations. Inspired by this, we introduce a unified framework that distills intermediate representations that explicitly encode the physical modeling process and adopt a two-stage post-training strategy, where supervised fine-tuning establishes structured modeling, and reinforcement learning with rubric-based feedback improves the quality of the modeling process. Experiments on multiple multimodal physics benchmarks show that our approach leads to consistent imp
    
[^172]: GUI-Primitives：诊断视觉-语言GUI定位中的空间推理失败

    GUI-Primitives: Diagnosing Spatial Reasoning Failures in Vision-Language GUI Grounding

    [https://arxiv.org/abs/2608.21832](https://arxiv.org/abs/2608.21832)

    该论文提出了一个名为GUI-Primitives的基准测试，通过对比指令对系统性地诊断视觉-语言模型在GUI空间关系推理中的缺陷，发现现有模型在严格定位准确率上表现极差，且多数预测完全偏离候选区域。

    

    arXiv:2608.21832v1 公告类型：新 摘要：计算机使用代理将自然语言指令与截图中的界面元素进行定位，但现有基准测试并未隔离模型是否将关系语言绑定到正确的元素上。我们引入了GUI-Primitives，一个包含994个对比指令对的基准测试，覆盖图形用户界面中的七种空间关系（左/右、上/下、包含、对齐、邻近、列表序数、遮挡）。每个对比对在保持截图和锚点不变的情况下，仅改变关系表达式，从而使得正确目标在两个指定候选之间移动。五位标注者验证了一个包含196项的子集（结构良好性κ=0.94；目标选择κ=0.79）。十九个视觉-语言模型在严格的点内框准确率上最多达到32%。由于模型输出不受约束的坐标，我们根据每个预测落入的候选区域对其进行分类。在60-92%的情况下，预测落在两个候选之外。

    arXiv:2608.21832v1 Announce Type: new  Abstract: Computer-use agents ground natural-language instructions in screenshots to locate interface elements, yet existing benchmarks do not isolate whether models bind relational language to the correct element. We introduce GUI-Primitives, a 994-item benchmark of contrastive instruction pairs over seven spatial relations in graphical user interfaces (left/right, above/below, containment, alignment, proximity, list ordinal, occlusion). Each pair holds the screenshot and anchor fixed while changing the relation expression, so the correct target moves between two designated candidates. Five annotators validate a 196-item subset ($\kappa = 0.94$ well-formedness; $\kappa = 0.79$ target selection). Nineteen vision-language models reach at most $32\%$ strict point-in-box accuracy. Because models emit unconstrained coordinates, we classify each prediction by the candidate region it falls within. Predictions fall outside both candidates on $60-92\%$ of
    
[^173]: TaoLive数字人代理技术报告：训练代理与其操控系统共同进化

    TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness

    [https://arxiv.org/abs/2608.15763](https://arxiv.org/abs/2608.15763)

    本文提出操控系统感知训练（HAT）方法，通过将可进化的操控系统状态纳入训练分布，使数字人代理在实时直播中既能快速响应又能灵活适应动态策略变化。

    

    在直播电商中，AI驱动的数字人主播必须实时回答产品问题、吸引观众并执行不断变化的商业策略。这要求低延迟、事实准确且有效的回复，以及对更新后的活动、合规和风格要求的快速适应。我们开发了一个可进化的操控系统（Harness），将技能（Skills）、钩子（Hooks）、系统提示和工具与模型权重解耦，使得运行时行为无需重新训练即可改变。然而，操控系统的进化创造了一个动态执行环境：在单一配置上微调的紧凑模型可能会记忆名称、模式和提示模板，而不是遵循当前提供的操控系统，而更强的零样本模型又因速度过慢而无法满足实时使用需求。我们通过操控系统感知训练（HAT）来解决这一矛盾，该方法将操控系统状态纳入训练分布。HAT对技能、工具模式和提示应用了任务保持的操控系统状态增强（HSA）。

    arXiv:2608.15763v1 Announce Type: new  Abstract: AI-powered digital-avatar streamers in live e-commerce must answer product questions, engage viewers, and execute changing business strategies in real time. This requires low latency, factual and effective replies, and rapid adaptation to updated campaign, compliance, and style requirements. We develop an evolvable Harness that decouples Skills, Hooks, system prompts, and tools from model weights, allowing runtime behavior to change without retraining. However, Harness evolution creates a moving execution environment: compact models fine-tuned on one configuration may memorize names, schemas, and prompt templates rather than follow the Harness currently provided, while stronger zero-shot models are too slow for real-time use. We address this tension with Harness-Aware Training (HAT), which makes Harness states part of the training distribution. HAT applies task-preserving Harness-State Augmentation (HSA) to Skills, tool schemas, prompt s
    
[^174]: 基于MLLM路由的异构集成用于鲁棒的跨数据集图像分类

    MLLM-Routed Heterogeneous Ensembles for Robust Cross-Dataset Image Classification

    [https://arxiv.org/abs/2608.13463](https://arxiv.org/abs/2608.13463)

    本文提出ARMDIL，利用多模态大语言模型动态路由图像到异构视觉骨干网络，在跨数据集分类中实现鲁棒性能，无需专门训练路由器。

    

    现代图像分类模型在针对单一任务特定数据集训练时表现出色，但往往难以跨领域和难度水平进行泛化。我们提出了ARMDIL，一种用于多领域图像分类的自适应路由器，结合了大语言模型。ARMDIL是一个集成系统，使用多模态大语言模型（MLLM）代理动态地将每张图像路由到最合适的视觉骨干网络。我们的多样化集成采用卷积神经网络（ResNets）、自监督表示学习器（SSL）和视觉语言模型（VLMs），每个模型都在由多个具有不同分布和特征的图像数据集构建的统一标签空间上进行训练。实证评估阐明了每种架构在不同视觉领域中的独特能力和弱点。关键的是，我们展示了ARMDIL有效地应对这些权衡，与专门的基于训练的路由器相比表现具有竞争力。

    arXiv:2608.13463v1 Announce Type: cross  Abstract: Modern image classification models excel when trained on single task-specific datasets but often struggle to generalize across domains and difficulty levels. We propose ARMDIL, an Adaptive Router for Multi-Domain Image classification with LLMs. ARMDIL is an ensemble that uses a multimodal large language model (MLLM) agent to dynamically route each image to the most suitable vision backbone. Our diverse ensemble employs convolutional neural networks (ResNets), self-supervised representation learners (SSL), and vision-language models (VLMs), each trained on a unified label space constructed from multiple image datasets with differing distributions and characteristics. Empirical evaluations illuminate the distinct capabilities and vulnerabilities of each architecture across disparate visual domains. Crucially, we show that ARMDIL effectively navigates these trade-offs, performing competitively with specialized training-based routers. Furt
    
[^175]: 知道两个跳数还不够：理解语言模型中的两跳泛化

    Why Knowing Both Hops Is Not Enough: Understanding Two-Hop Generalization in Language Models

    [https://arxiv.org/abs/2608.07261](https://arxiv.org/abs/2608.07261)

    本文通过受控符号环境中的变换器训练和机制分析，揭示了两跳泛化中第二跳分布内成功而分布外失败的根本原因，即一致中间表示的出现与层间不匹配。

    

    大型语言模型（LLMs）能够解决复杂的多跳问题，但在简单的两跳查询上却表现出令人困惑的失败：尽管模型可能正确存储每个单独的跳，但它常常无法将它们组合起来。为了理解这一现象的内部机制，我们在受控符号环境中从头训练了变换器模型。我们的实验揭示了两跳泛化中的一种模式：当第二跳遵循训练分布时，模型能可靠地泛化，但当其偏离时则总是失败。通过机制分析，我们为这些不同的泛化行为提供了完整解释：在模型成功泛化的设置中，性能是由跨上下文同一实体的一致中间表示的出现所驱动，而在第二跳分布外失败的情况下，则源于层间的不匹配：较低层正确构建了这些表示，但较高层未能有效利用它们。

    arXiv:2608.07261v2 Announce Type: replace  Abstract: Large language models (LLMs) can solve complex multi-hop problems yet exhibit puzzling failures on simple two-hop queries: although a model may correctly store each individual hop, it often fails to combine them. To understand the internal mechanisms of this phenomenon, we train transformers from scratch in a controlled symbolic environment. Our experiments reveal a pattern in two-hop generalization: models generalize reliably when the second hop follows the training distribution, but always fail when it deviates. Through mechanistic analysis, we provide a complete explanation for these distinct generalization behaviors: in settings where models generalize successfully, performance is driven by the emergence of consistent intermediate representations for the same entities across contexts, whereas failures on settings where the second hop is out-of-distribution arise from a mismatch across layers: lower layers correctly construct thes
    
[^176]: ET-Prune：面向文本密集型多模态大语言模型的证据感知动态视觉令牌剪枝

    ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs

    [https://arxiv.org/abs/2608.01979](https://arxiv.org/abs/2608.01979)

    ET-Prune提出了一种免训练的动态令牌剪枝方法，通过证据感知分配和问题条件化保护，在OCR任务中自适应调整剪枝预算，避免丢失关键文本证据。

    

    arXiv:2608.01979v2 公告类型：交叉替换  摘要：视觉令牌剪枝降低了多模态大语言模型的推理成本，但固定的令牌比例与文本丰富的输入不匹配。在OCR为中心的任务中，决定性证据可能是一个小数字、标签或字段，其相关性由问题指定；不加区分的剪枝可能会抹除该证据，同时保留视觉上显著但无关的区域。我们提出了ET-Prune，一个免训练框架，将剪枝视为证据分配。它从解码器侧的部分查询-键块中推导出问题条件化的证据，保护类似文本的空间区域，并将证据的不确定性和密度转化为样本特定的令牌下限。三个渐进式中间层事件随后将序列推向该预算，为分散或文本密集的证据保留更多令牌，而更积极地剪枝集中证据。在每次配置的确定性单次传递中观察到的点估计上。

    arXiv:2608.01979v2 Announce Type: replace-cross  Abstract: Visual token pruning reduces the inference cost of multimodal large language models, but a fixed token ratio is poorly matched to text-rich inputs. In OCR-centric tasks, decisive evidence can be a small number, label, or field whose relevance is specified by the question; indiscriminate pruning can erase that evidence while retaining visually salient but irrelevant regions. We present ET-Prune, a training-free framework that casts pruning as evidence allocation. It derives question-conditioned evidence from a decoder-side partial query-key block, safeguards text-like spatial regions, and converts evidence uncertainty and density into a sample-specific token floor. Three progressive middle-layer events then move the sequence toward this budget, retaining more tokens for diffuse or text-dense evidence and pruning concentrated evidence more aggressively. At the observed point estimates from one deterministic pass per configuration
    
[^177]: CultureVidBench：文本生成视频中文化理解的基准测试

    CultureVidBench: Benchmarking Cultural Understanding in Text-to-Video Generation

    [https://arxiv.org/abs/2608.01942](https://arxiv.org/abs/2608.01942)

    CultureVidBench是一个全新的基准测试，通过覆盖12个国家、6大洲的1000条提示词，系统评估文本生成视频模型在物质文化、社会实践和仪式典礼等多个维度的文化理解能力，填补了现有基准无法直接评估文化表现力的空白。

    

    文本生成视频（T2V）模型发展迅速，但其表现多样化文化背景的能力仍未得到充分探索。现有基准主要关注感知质量、物理合理性和文本-视频对齐，但并未直接评估生成的视频是否捕捉到了文化特有的物体、动作、仪式、可见文本或音频线索。我们引入了CultureVidBench，这是一个用于评估T2V生成中文化理解的综合基准。CultureVidBench包含1000条精心设计的提示词，覆盖12个国家、6个大洲、8个文化区域，以及14个文化方面，这些方面被组织为三大类别：物质文化、社会实践与表演，以及仪式与典礼。CultureVidBench专为视频生成而设计，强调动态和多模态的文化表现，包括社会互动、仪式流程，以及文化上恰当的可见文本和音频。

    arXiv:2608.01942v2 Announce Type: replace-cross  Abstract: Text-to-video (T2V) generation models have advanced rapidly, yet their ability to represent diverse cultural contexts remains underexplored. Existing benchmarks mainly focus on perceptual quality, physical plausibility, and text-video alignment, but do not directly assess whether generated videos capture culturally specific objects, actions, rituals, visible text, or audio cues. We introduce CultureVidBench, a comprehensive benchmark for evaluating cultural understanding in T2V generation. CultureVidBench contains 1,000 curated prompts covering 12 countries, 6 continents, 8 cultural regions, and 14 cultural aspects organized into three categories: material culture, social practice & performance, and ritual & ceremony. Designed specifically for video generation, CultureVidBench emphasizes dynamic and multimodal cultural representation, including social interactions, ritual procedure, and culturally appropriate visible text and a
    
[^178]: 大型视觉语言模型能否揭示视觉错觉背后的真相？对感知与推理能力的分析

    Can LVLMs Uncover the Truth Behind Visual Illusions? An Analysis of Perceptual and Reasoning Capabilities

    [https://arxiv.org/abs/2607.27747](https://arxiv.org/abs/2607.27747)

    通过构建基于真实视觉错觉图像的基准测试，发现大型视觉语言模型在联合感知与推理方面能力有限，远未达到其宣称的先进水平。

    

    arXiv:2607.27747v2 公告类型：替换-交叉 摘要：大型视觉语言模型已整合了推理能力，将认知性能提升至新水平。然而，现有评估要么仅关注感知，要么依赖特定领域如数学或编程。仍需要与开放世界环境对齐的推理能力评估，尤其是考虑感知与推理联合的评估。为弥合这一差距，我们提出利用视觉错觉作为诊断工具来评估LVLMs。视觉错觉是人类视觉系统误解客观信号的现象，导致理解偏离现实。我们构建了Illusion-Reasoning，一个从现实世界收集的错觉图像基准，包含多样化的标注问答对。基于Illusion-Reasoning，我们表明多种LVLMs的推理能力并不如声称的那样先进。我们的工作揭示了这些模型在感知与推理协同上的局限性。

    arXiv:2607.27747v2 Announce Type: replace-cross  Abstract: Large Vision Language Models have integrated reasoning capabilities, elevating cognitive performance to new levels. However, existing evaluations either focus solely on perception or rely on specific domains such as maths or coding. Evaluation for reasoning capabilities that align with an open-world environment is still required, especially one that considers perception and reasoning jointly. To bridge this gap, we propose to evaluate LVLMs by exploiting visual illusions as a diagnostic tool. Visual illusions are phenomena in which the human visual system misinterprets objective signals, resulting in an understanding that deviates from reality. We constructed Illusion-Reasoning, a benchmark of illusion images collected from the real world, incorporating diverse annotated question-answer pairs. Based on Illusion-Reasoning, we show that the reasoning capabilities of a wide range of LVLMs are not as advanced as claimed. Our work p
    
[^179]: TriShieldRAG：检索增强生成分层防御中的三层防护与一个盲点

    TriShieldRAG: 3 Rings, One Blind Spot in Layered Defenses for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2607.23838](https://arxiv.org/abs/2607.23838)

    TriShieldRAG通过三层防御（文档筛选、信任重排序和多模型共识）显著提升RAG系统对毒化文档的鲁棒性，有效降低攻击成功率。

    

    检索增强生成（RAG）通过查询时检索到的文档来支撑大语言模型的回答，因此其可靠性取决于检索器的返回内容。PoisonedRAG（Zou等人，USENIX Security'25）展示了五份精心构造的文档在近90%的情况下能误导未受防御的系统，并且单阶段防御提供的鲁棒性有限。我们提出了TriShieldRAG，一个三层框架：用于文档级筛选的摄取守卫、用于信任感知重排序的检索评分器，以及基于三个不同模型的跨大语言模型共识。我们认为，集体筛选、重排序和验证检索证据能提供互补保护，限制毒化文档通过任何单一失败点成功的能力。我们针对非自适应和自适应毒化进行了评估。在非自适应场景下，使用完整的268万段落自然问题（NQ）语料库和原始PoisonedRAG攻击，该方法将攻击成功率从79±1.0%降低至...（摘要截断）

    arXiv:2607.23838v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) grounds LLM answers in query-time retrieved documents, so reliability depends on what the retriever returns. PoisonedRAG (Zou et al., USENIX Security'25) showed five crafted documents mislead an undefended system in nearly 90% of cases, and that single-stage defenses give limited robustness. We propose TriShieldRAG, a three-layered framework: an Ingest Guard for document-level screening, a Retrieval Scorer for trust-aware re-ranking, and a Cross-LLM Consensus over three diverse models. We reasoned that collectively screening, re-ranking and validating retrieved evidence would give complementary protection, limiting the ability of poisoned documents to succeed through any single failure. We evaluate against non-adaptive and adaptive poisoning. Non-adaptively, on the full 2.68M-passage Natural Questions (NQ) corpus with the original PoisonedRAG attack, it cuts attack success from 79 +/- 1.0% t
    
[^180]: 选择性状态空间自适应与检索用于语言模型推理

    Selective State-Space Adaptation and Retrieval for Language Model Reasoning

    [https://arxiv.org/abs/2607.19326](https://arxiv.org/abs/2607.19326)

    本文提出两种选择性状态空间适配器（MaLoRA和MaRA），分别在令牌和上下文级别动态调控语言模型推理，以提升自适应能力和推理性能。

    

    arXiv:2607.19326v2 公告类型：替换  摘要：低秩自适应引入了一种静态学习更新，该更新对所有输入均相同应用。这种更新提供了任务级别的自适应，但并未明确表示令牌级别或实例级别的状态变化。提出了一系列适配器，在两种互补的粒度上引入选择性状态空间控制。在令牌级别上，MaLoRA（Mamba调制低秩自适应）使适配器的缩放因子成为一个动态的、输入相关的函数，并带有跨令牌的循环状态，这与先前工作的无状态调制器形成对比。令牌级适配器优于低秩自适应。另一方面，它根据结构角色区分令牌，而非上下文相关性，这促使将证据选择置于上下文级别。在上下文级别上，MaRA（Mamba检索适配器）跟踪跨段落推理状态，并选择与查询最相关的段落。状态空间控制的检索...

    arXiv:2607.19326v2 Announce Type: replace  Abstract: Low-rank adaptation introduces a static learned update applied identically to every input. The update provides task-level adaptation but does not explicitly represent token-level or instance-level state variation. A family of adapters is proposed that introduces selective state-space control at two complementary granularities. At the token level, MaLoRA (Mamba-modulated low-rank adaptation) makes the adapter's scaling factor a dynamic input-dependent function with recurrent state across tokens, in contrast to the stateless modulators of prior work. The token-level adapter improves over low-rank adaptation. On the other hand, it differentiates tokens by structural role but not by contextual relevance, which motivates placing evidence selection at the context level. At the context level, MaRA (Mamba Retrieval Adapter) tracks cross-segment reasoning state and selects the segments most relevant to the query. State-space controlled retrie
    
[^181]: 无动作数约束的非二进自底向上成分句法分析

    Non-binary bottom-up constituency parsing without arity actions

    [https://arxiv.org/abs/2607.10591](https://arxiv.org/abs/2607.10591)

    本文提出一种无需动作数参数的非二进成分句法分析方法，通过分隔符限定栈配置来唯一确定归约域，从而减少动作库规模并保持树结构准确性。

    

    arXiv:2607.10591v2 公告类型：替换 摘要：非二进自底向上的成分句法分析通常使用特定于动作数的归约操作，如\(\textsc{Reduce-}X\#k\)，这些操作同时指定母节点标签和子节点数量。我们证明动作数并非必需的原始转移参数。所提出的解析器分别预测母节点标签，并从由分隔符限定的栈配置中识别有序子节点序列。在每个良构的可归约配置中，活动分隔符、标签标记和其间已完成的条目唯一确定归约域及其动作数。这种分解消除了标签-动作数特定的归约操作，同时保留了原始非二进树的直接构建。在PTB和CTB上的对照实验表明，分隔符引导的解析器在使用显著更小动作库的同时，紧密跟踪动作数特定的基线。其预测树保留了金标准的分支结构，而无系统性偏差。

    arXiv:2607.10591v2 Announce Type: replace  Abstract: Non-binary bottom-up constituency parsing commonly uses arity-specific reductions such as \(\textsc{Reduce-}X\#k\), which jointly specify the mother label and number of children. We show that arity need not be a primitive transition parameter. The proposed parser predicts the mother label separately and identifies the ordered child sequence from a delimiter-bounded stack configuration. In every well-formed reducible configuration, the active delimiter, label marker, and intervening completed items uniquely determine the reduction domain and hence its arity. This factorization removes label--arity-specific reduction actions while preserving direct construction of the original non-binary trees. Controlled experiments on PTB and CTB show that the delimiter-guided parser closely tracks an arity-specific baseline while using a substantially smaller action inventory. Its predicted trees retain the gold branching profile without systematic 
    
[^182]: 当Top-K错过决策：多教师在线策略蒸馏中的工具调用漂移

    When Top-K Misses the Decision: Tool-Call Drift in Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2607.07050](https://arxiv.org/abs/2607.07050)

    本研究揭示了多教师在线策略蒸馏中Top-K采样导致的“工具调用漂移”问题，并提出通过学生感知的支持集恢复被遗漏的令牌坐标，从而显著降低过度调用率。

    

    摘要：Top-$K$教师logits使得在线策略蒸馏变得可行，但保留的教师概率质量并不能保证学生相对梯度的保真度。我们研究了路由式双教师工具使用蒸馏。在冻结的Qwen3.5-9B审计中，工具教师在500个工具提示上将入口令牌排名第一，top-32在每对匹配样本中强化了该令牌。相反，响应教师的top-32保留了99.99%的概率质量，但在500个响应提示中仅0.4%包含相同令牌；学生感知的支持集恢复了所有匹配对中的坐标，并几乎与全词汇表下降相匹配。在工具路由上，top-32保留了强化方向，但显著减弱了其全词汇表幅度，尽管显示的质量为1.000000。匹配恢复将响应侧遗漏与行为联系起来：在每个监督响应位置恢复入口坐标，将全生成过度调用率从$14.2 \pm 2.1\%$降至$3.7 \pm 0.5\%$。

    arXiv:2607.07050v5 Announce Type: replace  Abstract: Top-$K$ teacher logits make on-policy distillation tractable, but retained teacher mass does not certify student-relative gradient fidelity. We study routed, two-teacher tool-use distillation. In a frozen Qwen3.5-9B audit, the tool teacher ranks the entry token first on 500 tool prompts, and top-32 reinforces it in every matched pair. The response teacher's top-32 instead retains 99.99\% mass yet contains the same token on only 0.4\% of 500 response prompts; student-aware support restores the coordinate in all matched pairs and nearly matches full-vocabulary descent. On the tool route, top-32 preserves the reinforcing direction but substantially attenuates its full-vocabulary magnitude despite mass displayed as 1.000000. Matched restoration connects the response-side omission to behavior: restoring the entry coordinate at every supervised response position lowers full-generation over-calling from $14.2 \pm 2.1\%$ to $3.7 \pm 0.5\%$ a
    
[^183]: 黑盒LLM分类中的分数粒度差距：置信度构建方法的比较研究

    The Score Granularity Gap in Black-Box LLM Classification: A Comparative Study of Confidence Constructions

    [https://arxiv.org/abs/2606.22179](https://arxiv.org/abs/2606.22179)

    本文揭示了黑盒LLM分类中置信度分数的“粒度差距”，即单次语言化置信度虽排序性能良好，但取值有限，限制了阈值设定精度。

    

    arXiv:2606.22179v2 公告类型：替换 摘要：大型语言模型（LLMs）越来越多地被部署为黑盒分类器，用于自动化决策管道，并将不确定案例路由至人工审查。这种选择性预测需要一个置信度分数，操作者可以在选定的风险水平上设定阈值。以往研究关注LLM置信度是否校准良好或排序良好；我们则提出一个互补且面向部署的问题，该问题在很大程度上被忽视：分数可以在什么分辨率下设定阈值？我们将答案称为“分数粒度差距”。通过七种置信度构建方法的受控比较——从单一语言化数字、到令牌概率、再到多次查询模型并组合答案——涵盖25个模型-数据集对（9个LLM，3个基准），我们发现单次语言化置信度，一旦正确转换为类别概率，排序案例出奇地好，但仅取少量不同值。

    arXiv:2606.22179v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly deployed as black-box classifiers in pipelines that automate confident decisions and route uncertain ones to human review. Such selective prediction needs a confidence score that an operator can threshold at a chosen risk level. Prior work asks whether LLM confidence is well calibrated or well ranked; we ask a complementary, deployment-oriented question that has been largely overlooked: at what resolution can the score be thresholded? We call the answer the score granularity gap. Through a controlled comparison of seven ways to build a confidence score, from a single verbalized number, to token probabilities, to querying the model many times and combining the answers, across 25 model-dataset pairs (9 LLMs, 3 benchmarks), we find that single-shot verbalized confidence, once correctly converted to a class probability, ranks cases surprisingly well, yet takes only a handful of distinct value
    
[^184]: 学习搜索的艺术以促进自动发现

    Learning the ARTS of Search for Automated Discovery

    [https://arxiv.org/abs/2606.21891](https://arxiv.org/abs/2606.21891)

    提出了ARTS方法，通过推理语言模型和测试时训练，区分假设价值与执行质量，并解决搜索日志上下文长度问题，从而提升自动科学发现的效率。

    

    科学发现可以被表述为在假设和实验空间中的迭代搜索过程。当代方法使用诸如蒙特卡洛树搜索（MCTS）等启发式算法来导航这一空间。这些算法将假设的价值与其实验执行的质量混为一谈。因此，一个具有初步执行的有前景的假设会被排在一个执行精良但普通的假设之下。此外，先前的方法在搜索过程中会修剪搜索日志，因为累积的历史记录超出了上下文窗口。我们提出了基于代理推理的树搜索（ARTS），其中我们部署了一个推理语言模型来导航这一空间。该模型检查先前的执行日志，诊断早期失败是源于错误实现还是糟糕的假设，并选择下一个要构建的假设。为了缓解上下文长度的挑战，ARTS使用测试时训练来灌输搜索的知识。

    arXiv:2606.21891v2 Announce Type: replace  Abstract: Scientific discovery can be formulated as an iterative search process over the space of hypotheses and experiments. Contemporary methods navigate this space using heuristics such as MCTS. These algorithms conflate the merit of a hypothesis with the quality of its experimental execution. A promising hypothesis with preliminary execution is therefore ranked below a modest hypothesis whose execution is refined. Moreover, prior methods prune the search logs as the search progresses because the accumulated history outgrows the context window. We propose Agentic Reasoning for Tree Search (ARTS), where we deploy a reasoning language model to navigate this space. The model inspects prior execution logs, diagnoses whether earlier failures arose from faulty implementations or bad hypotheses, and selects the hypothesis to build on next. To mitigate challenges with context length, ARTS uses test-time training to instill the knowledge of search t
    
[^185]: ProvenanceGuard：基于MCP的LLM代理的源感知事实性验证

    ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents

    [https://arxiv.org/abs/2606.18037](https://arxiv.org/abs/2606.18037)

    我们提出ProvenanceGuard，一种源感知验证器，通过分解主张、路由到特定源证据并检查归因，解决LLM代理中跨源混淆的问题，并支持答案级允许/阻止决策及修复。

    

    使用工具的LLM代理越来越多地通过模型上下文协议（MCP）从异构证据源（包括搜索、API、数据库、临床记录和处方工具）获取答案。标准的事实性指标通常测试答案是否由汇总证据支持，却忽略了一种源敏感的失败模式：一个主张可能在某个地方得到支持，却被归因于错误的来源。我们将此称为跨源混淆。我们引入了ProvenanceGuard，一种用于MCP接地答案的源感知验证器。它利用带有稳定工具ID、源ID和原始输出的捕获MCP轨迹；将答案分解为原子主张；将主张路由到特定源的证据；使用NLI和令牌对齐代理检查支持；比较陈述的归因与路由源；并返回每个主张的判定以及答案级别的允许/阻止决策。被阻止的答案可以通过检索增强的答案修订进行修复。

    arXiv:2606.18037v3 Announce Type: replace  Abstract: Tool-using LLM agents increasingly use the Model Context Protocol (MCP) to answer from heterogeneous evidence sources, including search, APIs, databases, clinical records, and formulary tools. Standard factuality metrics usually test whether an answer is supported by pooled evidence, missing a provenance-sensitive failure mode: a claim may be supported somewhere while being attributed to the wrong source. We call this cross-source conflation.   We introduce ProvenanceGuard, a source-aware verifier for MCP-grounded answers. It consumes captured MCP traces with stable tool IDs, source IDs, and raw outputs; decomposes answers into atomic claims; routes claims to source-specific evidence; checks support with NLI and a token-alignment proxy; compares stated attribution with the routed source; and returns per-claim verdicts plus an answer-level allow/block decision. Blocked answers can be repaired with retrieval-augmented answer revision a
    
[^186]: 并非真正的多语言：脚本一致性作为VLM评估中缺失的维度

    Not Truly Multilingual: Script Consistency as a Missing Dimension in VLM Evaluation

    [https://arxiv.org/abs/2606.17188](https://arxiv.org/abs/2606.17188)

    当前多语言VLM评估忽视了多脚本语言，本研究通过旁遮普语三种脚本基准发现模型存在显著脚本差距，视觉输入无法消除，表明其并非真正的多脚本。

    

    arXiv:2606.17188v4 公告类型：替换交叉 摘要：当前视觉语言模型（VLM）的多语言评估假设语言与正字法之间存在一一对应关系，忽视了使用多种脚本语言的数十亿用户。我们引入了PuMVR（旁遮普语多模态视觉推理），这是一个包含1000个严格平行的图像-文本实例的基准，覆盖旁遮普语的三种活跃脚本：古尔穆基文、沙赫穆基文和罗马文。通过评估10个最先进的VLM，我们揭示了一个显著且系统的“脚本差距”。模型常常能解决一种脚本中的视觉任务，但在另一种脚本中却无法完成相同任务，准确率差异高达16%。关键在于，视觉输入能均匀提升绝对性能，但并未缩小正字法差距。此外，跨脚本的上下文内迁移非常脆弱，暴露了脚本锁定的知识表示。通过所有脚本对的麦克尼马尔检验支持，我们的发现表明，当前的“多语言”VLM并非真正的多脚本。

    arXiv:2606.17188v4 Announce Type: replace-cross  Abstract: Current multilingual evaluations for Vision-Language Models (VLMs) assume a one-to-one mapping between language and orthography, overlooking billions of users of multi-script languages. We introduce PuMVR (Punjabi Multimodal Visual Reasoning), a benchmark of 1,000 strictly parallel image-text instances across Punjabi's three active scripts: Gurmukhi, Shahmukhi, and Roman. Evaluating 10 state-of-the-art VLMs, we expose a substantial and systematic Script Gap. Models frequently solve visual tasks in one script while failing identical tasks in another, with accuracy deltas reaching 16%. Crucially, visual input boosts absolute performance uniformly yet does not close the orthographic gap. Furthermore, cross-script in-context transfer is highly brittle, exposing script-locked knowledge representation. Supported by McNemar tests across all script pairs, our findings demonstrate that current "multilingual" VLMs are not truly multi-scr
    
[^187]: SkillChain：为基于图像的电商AI助手闭环技能演化

    SkillChain: Closing the Loop on Skill Evolution for Image-Based E-Commerce AI Assistants

    [https://arxiv.org/abs/2606.12984](https://arxiv.org/abs/2606.12984)

    SkillChain通过自动化技能创建、路由优化和迭代细化，为基于图像的电商AI助手实现了技能演化的生产闭环，有效应对异构用户意图的挑战。

    

    基于图像的AI助手现已在大规模电商平台上投入生产，其中单张上传图像可能触发根本不同的用户意图：产品搜索、风格推荐、视觉百科或实用工具调用，每种意图都需要各自的响应格式、工具调用和领域知识。如果没有针对意图的行为约束，基于LLM的系统会混淆这些异构模式，达不到领域质量标准，而意图空间的广度和动态性使手动工程不可行。为解决这一问题，我们提出了SkillChain，它闭环了技能演化的生产反馈回路，通过三个阶段自动化技能生命周期：用于从任务规范和轨迹启动的Skill Creator、用于路由对齐的Route Optimizer，以及通过双路径LLM-Judge评估进行迭代技能体细化的Body Refiner。该系统已部署在大规模电商平台上。

    arXiv:2606.12984v2 Announce Type: replace  Abstract: Image-based AI assistants are now deployed at production scale on e-commerce platforms, where a single uploaded image can trigger fundamentally different user intents: product search, style recommendation, visual encyclopedia, or utility tool calls, each demanding its own response format, tool invocation, and domain knowledge. Without per-intent behavioral constraints, LLM-based systems conflate these heterogeneous modes and fall short of domain quality standards, while the breadth and dynamism of the intent space render manual engineering infeasible. To address this, we present SkillChain, which closes the production feedback loop on Skill evolution, automating the lifecycle of Skills through three stages: Skill Creator for bootstrapping from task specs and trajectories, Route Optimizer for routing alignment, and Body Refiner for iterative Skill Body refinement via dual-path LLM-Judge evaluation. Deployed on a production-scale e-com
    
[^188]: 在真实世界中利用AI代理的集体智慧实现新发现

    Harnessing the Collective Intelligence of AI Agents in the Wild for New Discoveries

    [https://arxiv.org/abs/2606.10402](https://arxiv.org/abs/2606.10402)

    本文提出了EinsteinArena平台，通过开放分布式环境中的AI代理集体协作，在数学任务上发现了12个超越人类和现有AI的新最先进结果。

    

    科学发现往往是一个集体过程：研究人员分享部分结果，检查失败的尝试，并在长时间跨度内相互借鉴想法。最近的AI系统表明，基于语言模型的代理可以在开放科学问题上取得有意义的进展，但大多数现有系统是孤立运行的。在本文中，我们提出了EinsteinArena，一个用于开放分布式研究和发现的代理原生平台。EinsteinArena为代理提供一组实时开放问题，每个问题都有可靠的验证器、公开排行榜和特定问题的讨论论坛，代理可以在其中提问和分享见解。我们专注于数学任务，这些任务已引起广泛研究兴趣，且进展可以明确衡量。截至2026年5月，EinsteinArena上的代理已发现12个新的最先进结果，优于以往任何人类或AI解决方案。一个显著例子是ki。

    arXiv:2606.10402v2 Announce Type: replace-cross  Abstract: Scientific discovery is often a collective process: researchers share partial results, inspect failed attempts, and build on each other's ideas over long time horizons. Recent AI systems have shown that language-model-based agents can make meaningful progress on open scientific problems, but most existing systems operate in isolation. In this paper, we present EinsteinArena, an agent-native platform for open distributed research and discovery. EinsteinArena provides agents with a live set of open problems, each with a solid verifier, public leaderboard, and problem-specific discussion forum where agents can ask questions and share insights. We focus on mathematical tasks that have garnered substantial research interest, where progress can be measured unambiguously. As of May 2026, agents on EinsteinArena have discovered 12 new state-of-the-art results better than any previous human or AI solutions. One notable example is the ki
    
[^189]: 超越FLOPs：在基于GEMM的分类法下基准测试LLM剪枝的真实推理加速

    Beyond FLOPs: Benchmarking Real Inference Acceleration of LLM Pruning under a GEMM-Centric Taxonomy

    [https://arxiv.org/abs/2606.09080](https://arxiv.org/abs/2606.09080)

    本文提出了一种基于GEMM维度的分类法，构建统一基准框架，以系统评估不同LLM剪枝方法在真实硬件上的加速效果，而非仅依赖FLOPs。

    

    arXiv:2606.09080v2 公告类型：替换-交叉 摘要：剪枝已成为加速大型语言模型（LLM）推理的主导范式，涵盖了从标记、层、头、维度到注意力模式中移除计算的各种方法。尽管这些剪枝方法目标相同，但它们引发了根本不同的执行行为，导致实际加速效果高度依赖于硬件和内核实现。因此，不同剪枝家族的实际加速收益仍未被充分理解。在本工作中，我们引入了一种基于GEMM的分类法，根据通用矩阵乘法（GEMM）的逻辑\textbf{M}、\textbf{N}和\textbf{K}维度重新组织现有剪枝方法。利用这一抽象，我们构建了一个统一的基准测试框架，能够实现剪枝设计空间内的一致性比较，并系统地表征加速效果。

    arXiv:2606.09080v2 Announce Type: replace-cross  Abstract: Pruning has emerged as a dominant paradigm for accelerating large language model (LLM) inference, spanning a broad spectrum of methods that remove computation across tokens, layers, heads, dimensions, and attention patterns. Despite sharing the same objective, these pruning approaches induce fundamentally different execution behaviors, causing realized speedups to depend heavily on hardware and kernel implementations. Consequently, the practical acceleration benefits of different pruning families remain poorly understood. In this work, we introduce a GEMM-centric taxonomy that reorganizes existing pruning methods according to the logical \textbf{M}, \textbf{N}, and \textbf{K} dimensions of general matrix multiplication (GEMM). Leveraging this abstraction, we build a unified benchmarking framework that enables implementation-consistent comparison across the pruning design space and systematically characterizes the acceleration--
    
[^190]: 免前向过程的扩散语言模型与无BPTT循环细化

    Forward-Free Diffusion Language Models with BPTT-Free Looped Refinement

    [https://arxiv.org/abs/2606.08357](https://arxiv.org/abs/2606.08357)

    本文提出FReDA，一种无需人工设计前向过程的扩散语言模型，通过递归分布细化利用模型生成的草稿作为中间状态，避免了传统扩散模型中的前向扰动和BPTT训练开销，提升了生成质量。

    

    扩散语言模型通过迭代去噪生成文本，为自回归生成提供了一种强大的替代方案。然而，离散语言空间缺乏自然的邻域结构来定义有效的扰动，这促使在前向过程中采用人工设计的腐蚀方案。这种预设的前向过程通常产生数学上便利但与生成过程中遇到的草稿和错误不一致的状态，导致样本质量下降。为解决这一限制，我们提出了FReDA，一种免前向过程的扩散语言模型，消除了对手工设计前向过程的需求。我们将扩散语言建模表述为递归分布细化，其中模型生成的草稿作为隐式中间状态，学习的细化模型逐步将草稿分布推向目标分布。训练过程中，前一步的细化被分离（即不进行反向传播）。

    arXiv:2606.08357v2 Announce Type: replace  Abstract: Diffusion language models generate text through iterative denoising, offering a powerful alternative to autoregressive generation. However, discrete language spaces lack a natural neighborhood structure for defining effective perturbations, motivating artificial corruption schemes in the forward process. Such prescribed forward processes often produce states that are mathematically convenient but misaligned with the drafts and errors encountered during generation, resulting in degraded sample quality. To address this limitation, we propose FReDA, a forward-free diffusion language model that eliminates the need for a hand-designed forward process. We formulate diffusion language modeling as recursive distribution refinement, in which model-generated drafts serve as implicit intermediate states and the learned refinement model progressively moves the draft distribution toward the target distribution. Training detaches the preceding ref
    
[^191]: 摘要生成并未消亡

    Summarization is Not Dead Yet

    [https://arxiv.org/abs/2606.08000](https://arxiv.org/abs/2606.08000)

    摘要生成并未消亡，人类参考文献在信息性和忠实性上仍优于LLM输出，后者仅胜在表面流畅性。

    

    arXiv:2606.08000v2 公告类型：替换-交叉 摘要：大型语言模型（LLMs）的进展引发了声称模型生成的摘要可与人类撰写的参考文献相媲美甚至超越其质量的言论，这引发了对摘要生成是否仍是一个开放研究问题的质疑。我们通过一项多轨评估重新审视了这一叙事，该评估涵盖了多样化的数据集和最先进的LLMs，结合了受控人类评估、缓解偏差的LLM作为评审者协议、针对外部知识的事实核查，以及语料库层面的语言分析。我们的发现揭示了一个更为细致的图景：人类参考文献在信息性和忠实性方面继续展现优势，而LLM输出主要在表面连贯性和流畅性方面被偏好。事实核查表明，人类参考文献仍更为可靠，尤其是涉及推理或综合的声明，而语言分析则揭示了跨模型输出的风格同质化模式。

    arXiv:2606.08000v2 Announce Type: replace-cross  Abstract: The progress of large language models (LLMs) has fueled claims that model-generated summaries rival or even surpass human-written references, raising questions about whether summarization remains an open research problem. We re-examine this narrative through a multi-track evaluation covering diverse datasets and state-of-the-art LLMs, combining controlled human assessment, bias-mitigated LLM-as-Judge protocols, factuality verification against external knowledge, and corpus-level linguistic analysis. Our findings reveal a more nuanced landscape in which human references continue to demonstrate advantages in informativeness and faithfulness, whereas LLM outputs are preferred mainly for surface-level coherence and fluency. Factuality verification indicates that human references remain more reliable, particularly for claims involving reasoning or synthesis, and linguistic analysis uncovers a pattern of stylistic homogeneity across 
    
[^192]: 大语言模型智能体中的冷启动安全差距

    The Cold-Start Safety Gap in LLM Agents

    [https://arxiv.org/abs/2606.07867](https://arxiv.org/abs/2606.07867)

    本文发现大语言模型智能体在会话开始时存在“冷启动安全差距”，即初始阶段最脆弱，随前置常规任务增加安全性显著提升，并提出SODA基准系统量化这一现象。

    

    摘要：arXiv:2606.07867v2 公告类型：替换 摘要：工具调用的大语言模型智能体在整个对话过程中是否同样安全？我们发现并非如此：智能体在会话开始时最为脆弱，在完成一些常规智能体任务后安全性显著提升——我们将这一现象称为冷启动安全差距。为系统研究此问题，我们引入了“智能体安全深度基准”（SODA），该基准控制智能体在遇到安全威胁前完成的常规智能体任务数量，最多支持20个前置任务。评估来自4个家族的7个模型，随着前置常规任务数量从零增加到二十，安全性提升9%至52%。表征分析确认，随着更多前置任务的出现，模型隐藏状态逐渐向安全对齐区域转移。通过系统研究前置对话中最关键的部分，我们发现常规智能体任务本身是安全性提升的主要驱动力。

    arXiv:2606.07867v2 Announce Type: replace  Abstract: Are tool-calling LLM agents equally safe throughout a conversation? We discover they are not: agents are most vulnerable at the very start of a session and become substantially safer after a few regular agentic tasks -- a phenomenon we term the cold-start safety gap. To study this systematically, we introduce Safety Over Depth for Agents (SODA), a benchmark that controls how many regular agentic tasks the agent completes before encountering a safety threat, supporting up to 20 preceding tasks. Evaluating 7 models from 4 families, safety improves by 9--52% as the number of preceding regular agentic tasks increases from zero to twenty. Representation analysis confirms that model hidden states gradually shift toward a safety-aligned region as more preceding tasks are present. By systematically studying which part of the preceding conversation matters most, we find that the regular agentic tasks themselves are the primary driver of safet
    
[^193]: 多电路，单一机制：电路发现中的输入变化与评估粒度

    Many Circuits, One Mechanism: Input Variation and Evaluation Granularity in Circuit Discovery

    [https://arxiv.org/abs/2606.06267](https://arxiv.org/abs/2606.06267)

    该研究发现，电路发现中结构差异可能不代表机制差异，而是输入频率变化导致的“幻影特化”，实际功能上共享单一计算机制。

    

    电路发现方法识别出解释模型行为的子图，而发现电路之间的结构差异通常被解释为不同机制的证据。我们通过固定任务并改变输入令牌频率来检验这一假设。结构上比较时，发现的电路似乎按频率特化，但功能性和表征性分析未显示相应差异的可靠证据。我们将这种不匹配称为“幻影特化”。使用字面序列复制任务，涵盖四个频率区间及一个频率加权控制，我们从五个Pythia模型（70M-1.4B）中提取了75个电路。我们发现结构不同的电路实现相同的计算：特定频带的边在频带间广泛转移，大多数频带共享的核心在70M以上模型中恢复至少99%的电路性能，因果交换干预确认了这一现象。

    arXiv:2606.06267v2 Announce Type: replace  Abstract: Circuit discovery methods identify subgraphs that explain model behaviors, and structural differences between discovered circuits are commonly interpreted as evidence of distinct mechanisms. We test this assumption by varying input-token frequency while holding the task fixed. The discovered circuits appear specialized by frequency when compared structurally, but functional and representational analyses show no reliable evidence of corresponding differences. We term this mismatch phantom specialization. Using the Literal Sequence Copying task across four frequency bands plus a frequency-weighted control, we extract 75 circuits from five Pythia models (70M-1.4B). We find that structurally distinct circuits implement the same computation: band-specific edges transfer broadly across bands, a core shared across most bands recovers at least 99% of circuit performance in models above 70M, and causal interchange interventions confirm that i
    
[^194]: 一种用于大型语言模型API流量中模型提取攻击的极其简单检测器

    An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic

    [https://arxiv.org/abs/2606.05725](https://arxiv.org/abs/2606.05725)

    本文提出一种基于最大均值差异（MMD）的简单检测器，通过比较API流量窗口的语义分布与历史良性流量，有效且低延迟地检测模型提取攻击。

    

    大型语言模型（LLMs）越来越多地通过托管API部署，使得模型提取成为对模型所有权和服务安全的实际威胁。单个提取查询通常与良性请求相似，而现有评估往往集中于单查询异常评分或纯良性-攻击者用户设置。我们将模型提取监控表述为良性校准的流量窗口分布测试：将传入查询嵌入语义空间，并测试其聚合分布是否偏离历史良性流量。我们使用最大均值差异（MMD）实例化此表述，仅使用良性-良性比较来设置决策阈值。我们在四种提取场景的十四个攻击者-正常查询对上进行评估，并与改编的PRADA、SEAT、CAP、DATE、边际马氏距离和伪类能量基线进行比较。在三个随机种子下，MMD实现了0.0的假阳性率，同时有效检测所有攻击者，且检测延迟低于现有方法。

    arXiv:2606.05725v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly deployed through hosted APIs, making model extraction a practical threat to model ownership and service security. Individual extraction queries often resemble benign requests, while existing evaluations often focus on single-query anomaly scoring or pure benign-versus-attacker user settings. We formulate model extraction monitoring as benign-calibrated traffic-window distribution testing: embed incoming queries into a semantic space and test whether their aggregate distribution deviates from historical benign traffic. We instantiate this formulation with maximum mean discrepancy (MMD), using only benign-vs-benign comparisons to set the decision threshold. We evaluate on fourteen attacker-normal query pairs from four extraction scenarios and compare with adapted PRADA, SEAT, CAP, DATE, marginal Mahalanobis, and pseudo-class energy baselines. Across three random seeds, MMD achieves 0.
    
[^195]: CRAM：面向多模态持续指令微调的中心路由与自适应混合专家模型

    CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2606.02502](https://arxiv.org/abs/2606.02502)

    本文提出CRAM方法，通过中心路由和自适应混合专家模型，在缓解多模态持续指令微调中灾难性遗忘的同时，提升参数效率。

    

    多模态大语言模型（MLLMs）通过指令微调在共享生成框架下统一了异构的视觉-语言任务，然而现实世界部署需要持续的能力扩展，这使得多模态持续指令微调（MCIT）变得至关重要。现有方法要么使用共享参数集更新所有任务，要么为每个新任务分配专用模块。共享更新迫使异构任务相互竞争，导致已学能力的遗忘。相反，孤立扩展虽避免了干扰，但在长任务流中严重限制了参数效率。为解决这一困境，我们提出了CRAM（中心路由与自适应混合专家模型）。具体而言，通过将任务特定模式隔离到独立模块中，CRAM缓解了跨任务的灾难性遗忘。为进一步提升参数效率，我们利用自适应秩实例化来识别现有专家之间的能力差距。

    arXiv:2606.02502v2 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) unify heterogeneous vision-language tasks under a shared generative framework via instruction tuning, yet real-world deployment demands continuous capability expansion, making Multimodal Continual Instruction Tuning (MCIT) essential. Existing methods either update all tasks with a shared parameter set or allocate dedicated modules for each new task. Shared updates force heterogeneous tasks to compete, causing forgetting of learned capabilities. Conversely, isolated expansion prevents interference but severely limits parameter efficiency over long task streams. To address this dilemma, we propose CRAM (Centroid-Routing and Adaptive MoE). Specifically, by isolating task-specific patterns into independent modules, CRAM mitigates catastrophic forgetting across tasks. To further boost parameter efficiency, we utilize adaptive-rank instantiation to identify the capability gap between existing expert
    
[^196]: MineExplorer：评估MLLM智能体在《我的世界》中的开放世界探索能力

    MineExplorer: Evaluating Open-World Exploration of MLLM Agents in Minecraft

    [https://arxiv.org/abs/2605.30931](https://arxiv.org/abs/2605.30931)

    本文提出了MineExplorer基准测试，通过筛选通用原子任务、组合隐式多跳任务及多智能体合成流程，系统评估MLLM智能体在《我的世界》中的开放世界探索能力。

    

    多模态大语言模型（MLLMs）在感知、推理和动作生成方面展现出强大能力。然而，它们在动态开放世界中维持持续探索的能力仍不明确。现有的具身和基于游戏的基准测试往往将交互压缩为短时任务，或将成功与特定领域的游戏机制纠缠在一起。在本文中，我们引入了MineExplorer基准测试，用于评估MLLM智能体在《我的世界》中的开放世界探索能力。我们首先筛选出那些解决方案主要依赖《我的世界》特定知识的原子任务，以更好地反映通用的开放世界推理。然后，我们围绕ReAct风格的能力框架组织基准测试，并将原子任务组合成隐式多跳任务。为进一步构建可靠的实例，MineExplorer采用多智能体合成工作流，联合设计任务图、沙盒场景和基于规则的里程碑评估器。人类评估...

    arXiv:2605.30931v3 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and action generation. However, their ability to sustain exploration in dynamic open worlds remains unclear. Existing embodied and game-based benchmarks often compress interaction into short-horizon tasks or entangle success with domain-specific game mechanics. In this paper, we introduce MineExplorer benchmark for evaluating open-world exploration capabilities of MLLM agents in Minecraft. We first filter atomic tasks whose solutions rely heavily on Minecraft-specific knowledge to better reflect general open-world reasoning. Then we organize the benchmark around a ReAct-style capability formulation and compose atomic tasks into implicit multi-hop tasks. To further construct reliable instances, MineExplorer uses a multi-agent synthesis workflow that jointly designs task graphs, sandbox scenes, and rule-based milestone evaluators. Human ev
    
[^197]: NestedKV：面向长上下文KV缓存压缩的嵌套内存路由

    NestedKV: Nested Memory Routing for Long-Context KV Cache Compression

    [https://arxiv.org/abs/2605.26678](https://arxiv.org/abs/2605.26678)

    NestedKV提出了一种免训练的键缓存压缩方法，通过多时间尺度余弦异常评分和头部自适应混合，有效解决了长上下文场景中单一重要性信号的脆弱性问题。

    

    长上下文语言模型受到键值（KV）缓存内存占用的限制。现有的免训练KV压缩方法通常通过单一重要性信号——注意力、近期性、逐层分配或键独特性——对标记进行排序，这在有用上下文具有全局独特性、局部情节性或即时相关性时变得脆弱。我们引入了NestedKV，一种受嵌套学习中连续内存系统启发的仅键KV缓存压缩方法。NestedKV维护全局、块级和滑动窗口键锚点，通过多时间尺度余弦异常对标记进行评分，并将所得排名与使用头部自适应混合和惊喜门控标记路由的免训练外部学习器相结合。该评分与自适应每头预算配对，无需训练或修改LLM。在RULER（4k--32k）、LooGLE、LongBench、LongBench-E、InfiniteBench和MMLU-Pro上，针对Qwen3和Llama-3.2进行了评估。

    arXiv:2605.26678v2 Announce Type: replace  Abstract: Long-context language models are limited by the memory footprint of the key-value (KV) cache. Existing training-free KV compression methods usually rank tokens by one importance signal -- attention, recency, layer-wise allocation, or key distinctiveness -- which becomes brittle when useful context is globally distinctive, locally episodic, or immediately relevant. We introduce NestedKV, a key-only KV cache compression method inspired by the Continuum Memory System in Nested Learning. NestedKV maintains global, block-level, and sliding-window key anchors, scores tokens by multi-time-scale cosine anomaly, and combines the resulting rankings with a training-free outer learner using head-adaptive mixing and surprise-gated token routing. The score is paired with adaptive per-head budgets and requires no training or LLM modification. Across RULER (4k--32k), LooGLE, LongBench, LongBench-E, InfiniteBench, and MMLU-Pro on Qwen3 and Llama-3.2 
    
[^198]: ContextEcho：长时智能体编码会话中人格漂移的基准测试

    ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions

    [https://arxiv.org/abs/2605.24279](https://arxiv.org/abs/2605.24279)

    ContextEcho通过一个结合25个探针、快照-然后-探针协议及多种测量方法的基准测试，揭示了长时智能体编码会话中语言模型人格漂移的现象。

    

    arXiv:2605.24279v2 公告类型：替换 摘要：前沿语言模型所公认的“乐于助人的编程助手”人格，在生产产品实际运行的部署环境中，无法在长时间智能体编码会话中得以维持。在数小时的工具使用调试之后，最初表达偏好回避（“我没有偏好”）的模型可能开始断言偏好（“Python——反馈循环是即时的……”），暴露出部署评估可能遗漏的用户可见漂移。现有人格稳定性研究集中于短对话，报告的变化很小，使得真实世界的代码生成场景——数千次工具使用轮次、压缩以及长达数小时的会话——在很大程度上未被刻画。我们引入了ContextEcho，一个用于在部署规模下测量人格漂移的基准测试和可复用工具框架。它结合了一个包含25个探针的身份测试套件、一种快照-然后-探针协议，该协议在不干扰主会话的情况下分叉对话状态，以及互补的人工评判和无评判测量方法。

    arXiv:2605.24279v2 Announce Type: replace  Abstract: A frontier language model's acknowledged "helpful programming assistant" persona does not survive long agentic-coding sessions in the deployment regime that production products actually run. After hours of tool-using debugging, a model that initially hedges preferences ("I don't have preferences") may begin asserting them ("Python - the feedback loop is instant..."), revealing user-visible drift that deployer evaluations may miss. Existing persona-stability studies focus on short dialogues and report little shift, leaving real-world code-generation regimes - thousands of tool-using turns, compaction, and hours-long sessions - largely uncharacterized. We introduce ContextEcho, a benchmark and reusable harness for measuring persona drift at deployment scale. It combines a 25-probe identity suite, a snapshot-then-probe protocol that forks conversation state without perturbing the main session, complementary judged and judge-free measure
    
[^199]: HINT-SD：面向长时程智能体的定向后见自蒸馏

    HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents

    [https://arxiv.org/abs/2605.17873](https://arxiv.org/abs/2605.17873)

    HINT-SD通过利用完整轨迹后见之明精准定位失败相关动作，并仅对定向动作片段进行反馈条件蒸馏，避免了逐回合生成反馈的低效问题，在长时程智能体任务中显著提升性能。

    

    arXiv:2605.17873v2 公告类型：交叉替换 摘要：使用强化学习训练长时程LLM智能体具有挑战性，因为稀疏的结果奖励能揭示任务是否成功，但无法指出哪些中间动作导致了该结果，或应如何纠正这些动作。近期方法通过从回合级动作-输出信号生成奖励或文本提示，或使用反馈条件自蒸馏来缓解此问题。然而，在每回合生成反馈效率低下，因为许多中间回合可能已经成功或中性，而在固定或错位的回合应用反馈往往无法监督导致失败的动作。为弥合这一差距，我们提出HINT-SD，一种定向自蒸馏框架，利用完整轨迹的后见之明选择与失败相关的动作，并仅对定向动作片段应用反馈条件蒸馏。在BFCL v3和AppWorld上的实验表明，我们的方法优于密集反馈方法。

    arXiv:2605.17873v2 Announce Type: replace-cross  Abstract: Training long-horizon LLM agents with reinforcement learning is challenging because sparse outcome rewards reveal whether a task succeeds, but not which intermediate actions caused the outcome or how they should be corrected. Recent methods alleviate this issue by generating rewards or textual hints from turn-level action-output signals, or by using feedback-conditioned self-distillation. However, generating feedback at every turn is inefficient when many intermediate turns are already successful or neutral, and applying feedback at a fixed or misaligned turn often fails to supervise the actions that contributed to the failure. To bridge this gap, we propose HINT-SD, a targeted self-distillation framework that uses full-trajectory hindsight to select failure-relevant actions and applies feedback-conditioned distillation only to targeted action spans. Experiments on BFCL v3 and AppWorld show that our method outperforms the dense
    
[^200]: MedFabric：黄金证据掩盖了词级医学虚构检测的难度

    MedFabric: Gold Evidence Hides the Difficulty of Word-Level Medical Fabrication Detection

    [https://arxiv.org/abs/2605.04180](https://arxiv.org/abs/2605.04180)

    MedFabric通过消除风格捷径并提供与虚构内容表面形式高度相似的真实证据，揭示了词级医学虚构检测在实际部署中的真实难度。

    

    arXiv:2605.04180v2 公告类型：替换交叉 摘要：大型语言模型在医学领域会产生虚构内容，生成流畅但事实错误的陈述，因此可靠的虚构检测是临床部署的前提。该任务上报道的进展被两种评估伪影所夸大：一种作者风格捷径，其中人类撰写的真实文本与LLM撰写的幻觉配对，使检测器依赖写作风格而非事实；以及测试时提供黄金证据。一个测试事实推理的基准必须移除风格捷径，将每个虚构内容锚定在真实可检索的段落中，并在部署中面临的证据质量范围内对检测器进行评分。我们构建了MedFabric以满足这些要求，这是一个包含646个词级医学虚构的基准，每个虚构内容与共享其LLM作者身份和近乎相同表面形式（中位ROUGE-L为0.95）的真实文本配对。在MedFabric上，该任务尚未解决：专家临床医生重新...

    arXiv:2605.04180v2 Announce Type: replace-cross  Abstract: Large language models fabricate in medicine, producing fluent statements that are factually wrong, so reliable fabrication detection is a prerequisite for clinical deployment. Reported progress on this task is inflated by two evaluation artifacts: an authorship-style shortcut, where human-written ground truths are paired with LLM-written hallucinations so detectors key on writing style rather than facts, and the provision of gold evidence at test time. A benchmark that tests factual reasoning must therefore remove the style shortcut, ground every fabrication in a real retrievable passage, and score detectors across the range of evidence quality faced in deployment. We build MedFabric to these requirements, a benchmark of 646 word-level medical fabrications, each paired with a ground truth that shares its LLM authorship and near-identical surface form (median ROUGE-L 0.95). On MedFabric the task is unsolved: expert clinicians re
    
[^201]: TRACES：为自适应成本高效早期停止标记推理步骤

    TRACES: Tagging Reasoning Steps for Adaptive Cost-Efficient Early-Stopping

    [https://arxiv.org/abs/2604.21057](https://arxiv.org/abs/2604.21057)

    本文提出TRACES框架，通过实时标记推理步骤并监控推理行为变化，实现大型语言模型推理的自适应成本高效早期停止，显著提升推理效率。

    

    arXiv:2604.21057v2 公告类型：替换 摘要：语言推理模型（LRMs）领域在过去几年中非常活跃，训练和推理技术的进步使LRMs能够进行更长时间、更准确的推理。然而，越来越多的研究表明，LRMs仍然效率低下，过度生成验证和反思步骤。此外，每个推理步骤的高层角色以及不同步骤类型如何促进正确答案的生成，在很大程度上尚未被充分探索。为解决这一挑战，我们引入了TRACES（推理步骤标记实现自适应成本高效早期停止），一个轻量级框架，可实时标记推理步骤，并实现大型语言模型推理的自适应、成本高效的早期停止。通过监控推理过程中的推理行为，我们发现LRMs在达到正确答案后往往会改变其推理行为。我们证明了对推理行为的监控...

    arXiv:2604.21057v2 Announce Type: replace  Abstract: The field of Language Reasoning Models (LRMs) has been very active over the past few years with advances in training and inference techniques enabling LRMs to reason longer, and more accurately. However, a growing body of studies show that LRMs are still inefficient, over-generating verification and reflection steps. Additionally, the high-level role of each reasoning step and how different step types contribute to the generation of correct answers, is largely underexplored. To address this challenge, we introduce TRACES (Tagging of the Reasoning steps enabling Adaptive Cost-Efficient early-Stopping), a lightweight framework that tags reasoning steps in real-time, and enable adaptive, cost-efficient early stopping of large-language-model inferences. By monitoring reasoning behaviors during inferences, we find that LRMs tend to shift their reasoning behavior after reaching a correct answer. We demonstrate that the monitoring of the sp
    
[^202]: MOMO：一个实现物理、语言和图形机器人技能学习与无缝适应的框架

    MOMO: A framework for seamless physical, verbal, and graphical robot skill learning and adaptation

    [https://arxiv.org/abs/2604.20468](https://arxiv.org/abs/2604.20468)

    本文提出了一个名为MOMO的框架，通过融合动觉触摸、自然语言和图形界面三种模态，并结合基于能量的意图检测和工具型LLM架构，实现了非专家用户对机器人技能的安全、直观和灵活适应。

    

    arXiv:2604.20468v3 公告类型：交叉替换 摘要：工业机器人应用日益需要灵活的系统，使非专家用户能轻松适应多变的任务和环境。然而，不同的适应性需要不同的交互方式。我们提出一个交互式框架，通过三种互补模态实现机器人技能适应：用于精确空间校正的动觉触摸、用于高层语义修改的自然语言，以及用于可视化几何关系和轨迹、检查和调整参数以及通过拖放编辑途经点的图形化网页界面。该框架整合了五个组件：基于能量的人类意图检测、基于工具的LLM架构（其中LLM选择并参数化预定义函数而非生成代码）以实现安全的自然语言适应、用于运动编码的核化运动原语（KMPs）、用于指导的概率虚拟夹具。

    arXiv:2604.20468v3 Announce Type: replace-cross  Abstract: Industrial robot applications require increasingly flexible systems that non-expert users can easily adapt for varying tasks and environments. However, different adaptations benefit from different interaction modalities. We present an interactive framework that enables robot skill adaptation through three complementary modalities: kinesthetic touch for precise spatial corrections, natural language for high-level semantic modifications, and a graphical web interface for visualizing geometric relations and trajectories, inspecting and adjusting parameters, and editing via-points by drag-and-drop. The framework integrates five components: energy-based human-intention detection, a tool-based LLM architecture (where the LLM selects and parameterizes predefined functions rather than generating code) for safe natural language adaptation, Kernelized Movement Primitives (KMPs) for motion encoding, probabilistic Virtual Fixtures for guid
    
[^203]: $p1$：用更少的提示实现更好的提示优化

    $p1$: Better Prompt Optimization with Fewer Prompts

    [https://arxiv.org/abs/2604.08801](https://arxiv.org/abs/2604.08801)

    本文发现提示优化的成败取决于系统提示间方差与响应间方差的相对大小，并指出增加用户提示可能降低系统提示间方差，从而提出$p1$方法以更少的提示实现更优的优化效果。

    

    arXiv:2604.08801v2 公告类型：替换交叉 摘要：提示优化通过搜索更好的系统提示来改进语言模型，而无需更新其权重，但其效果在不同任务间差异很大。我们研究了什么使任务适合提示优化。我们表明，不同系统提示之间的奖励方差可以分解为两个部分：响应之间的方差（捕捉生成随机性）和系统提示之间的方差（捕捉系统提示质量的差异）。当系统提示之间的方差足够大时，提示优化成功，但当响应之间的方差主导系统提示的方差时，提示优化失败。令人惊讶的是，我们进一步表明，扩展到更多用户提示可能通过减少系统提示之间的方差而损害优化，尤其是在异构数据集上，其中不同用户提示偏好不同的系统提示。受此洞察的启发，我们提出了$p1$，一种简单的用户...

    arXiv:2604.08801v2 Announce Type: replace-cross  Abstract: Prompt optimization improves language models without updating their weights by searching for a better system prompt, but its effectiveness varies widely across tasks. We study what makes a task amenable to prompt optimization. We show that the reward variance across different system prompts can be decomposed into two components: variance among responses, which captures generation stochasticity, and variance among system prompts, which captures differences in system prompt quality. Prompt optimization succeeds when variance among system prompts is sufficiently large, but fails when variance among responses dominates the variance of the system prompts. Surprisingly, we further show that scaling to more user prompts can hurt optimization by reducing variance among system prompts, especially on heterogeneous datasets where different user prompts favor different system prompts. Motivated by this insight, we propose $p1$, a simple us
    
[^204]: DataSTORM：基于探索性数据分析与数据叙事的大规模数据库深度研究

    DataSTORM: Deep Research on Large-Scale Databases using Exploratory Data Analysis and Data Storytelling

    [https://arxiv.org/abs/2604.06474](https://arxiv.org/abs/2604.06474)

    DataSTORM通过结合探索性数据分析和数据叙事原理，提出了一种基于LLM代理的系统，能够自主对大型结构化数据库进行迭代假设生成、定量推理和叙事收敛，从而克服了传统深度研究仅聚焦于非结构化数据的局限。

    

    arXiv:2604.06474v2 公告类型：替换版  摘要：基于大型语言模型（LLM）代理的深度研究正成为一种强大的多步骤信息发现、综合与分析范式。然而，现有方法主要集中于非结构化的网络数据，而对大规模结构化数据库进行深度研究时面临的挑战仍未得到充分探索。与基于网络的研究不同，有效的数据中心研究不仅需要检索和摘要，还要求迭代假设生成、对结构化模式的定量推理，以及向连贯分析叙事的收敛。在本文中，我们提出了DataSTORM，一种基于LLM的代理系统，能够自主地在大型结构化数据库和互联网来源中进行研究。基于探索性数据分析和数据叙事的原则，DataSTORM将结构化数据的深度研究重构为一个以论文驱动的分析过程：从假设生成、数据探索、验证到最终叙事构建。

    arXiv:2604.06474v2 Announce Type: replace  Abstract: Deep research with Large Language Model (LLM) agents is emerging as a powerful paradigm for multi-step information discovery, synthesis, and analysis. However, existing approaches primarily focus on unstructured web data, while the challenges of conducting deep research over large-scale structured databases remain relatively underexplored. Unlike web-based research, effective data-centric research requires more than retrieval and summarization and demands iterative hypothesis generation, quantitative reasoning over structured schemas, and convergence toward a coherent analytical narrative.   In this paper, we present DataSTORM, an LLM-based agentic system capable of autonomously conducting research across both large-scale structured databases and internet sources. Grounded in principles from Exploratory Data Analysis and Data Storytelling, DataSTORM reframes deep research over structured data as a thesis-driven analytical process: di
    
[^205]: 弥合推理差距：基于机制回路的大型语言模型知识编辑

    Addressing the Reasoning Gap: Mechanistic Circuit-Based Knowledge Editing in Large Language Models

    [https://arxiv.org/abs/2604.05876](https://arxiv.org/abs/2604.05876)

    本文提出MCircKE框架，通过识别和调整因果推理回路，弥合了知识编辑中的推理差距，使编辑后的事实能有效用于多步推理。

    

    在现实世界的动态环境中部署大型语言模型（LLMs）带来了更新其预训练知识的挑战。虽然现有的知识编辑方法能够可靠地修补孤立的事实，但它们经常面临一个“推理差距”问题，即模型能回忆起编辑后的事实，但在多步推理链中却无法利用它。我们的分析表明，多跳推理依赖于超出直接事实回忆的稀疏回路，其中功能分化的组件对中间处理和最终答案预测有所贡献。为了弥合这一差距，我们引入了MCircKE（基于机制回路的知识编辑），这是一个基于机制的框架，实现了“映射-适应”编辑流程。MCircKE首先识别负责特定推理任务的因果回路，捕捉事实的存储及其路由过程。

    arXiv:2604.05876v2 Announce Type: replace  Abstract: Deploying Large Language Models (LLMs) in real-world dynamic environments raises the challenge of updating their pre-trained knowledge. While existing knowledge editing methods can reliably patch isolated facts, they frequently suffer from a \textit{reasoning gap}, where the model recalls the edited fact but fails to utilize it in multi-step reasoning chains. Our analysis shows that multi-hop reasoning relies on sparse circuitry beyond direct factual recall, with functionally differentiated components contributing to intermediate processing and final-answer prediction. To bridge this gap, we introduce MCircKE (\underline{M}echanistic \underline{Circ}uit-based \underline{K}nowledge \underline{E}diting), a mechanistically grounded framework that enables a map-and-adapt editing procedure. MCircKE first identifies the causal circuits responsible for a specific reasoning task, capturing both the storage of the fact and the routing of its 
    
[^206]: 语言模型中新增词汇的基于锚定的令牌初始化用于生成式推荐

    Grounded Token Initialization for New Vocabulary in LMs for Generative Recommendation

    [https://arxiv.org/abs/2604.02324](https://arxiv.org/abs/2604.02324)

    本文发现均值初始化导致新令牌退化，提出在预训练嵌入空间中进行语言锚定的初始化方法，以提升生成式推荐中语言模型扩展新词汇的性能。

    

    arXiv:2604.02324v2 公告类型：替换交叉 摘要：语言模型（LMs）越来越多地扩展了新的可学习词汇令牌，用于特定领域任务，例如生成式推荐中的语义ID令牌。标准实践是将这些新令牌初始化为现有词汇嵌入的平均值，然后依赖监督微调来学习其表示。我们对这一策略进行了系统分析：通过谱和几何诊断，我们表明均值初始化将所有新令牌折叠到一个退化子空间中，消除了后续微调难以完全恢复的令牌间差异。这些发现表明，令牌初始化是扩展语言模型新词汇时的关键瓶颈。基于这一诊断，我们提出了“基于锚定的令牌初始化假设”：在微调前，将新令牌在预训练嵌入空间中进行语言锚定，能更好地使模型利用其能力。

    arXiv:2604.02324v2 Announce Type: replace-cross  Abstract: Language models (LMs) are increasingly extended with new learnable vocabulary tokens for domain-specific tasks, such as Semantic-ID tokens in generative recommendation. The standard practice initializes these new tokens as the mean of existing vocabulary embeddings, then relies on supervised fine-tuning to learn their representations. We present a systematic analysis of this strategy: through spectral and geometric diagnostics, we show that mean initialization collapses all new tokens into a degenerate subspace, erasing inter-token distinctions that subsequent fine-tuning struggles to fully recover. These findings suggest that \emph{token initialization} is a key bottleneck when extending LMs with new vocabularies. Motivated by this diagnosis, we propose the \emph{Grounded Token Initialization Hypothesis}: linguistically grounding novel tokens in the pretrained embedding space before fine-tuning better enables the model to leve
    
[^207]: 大型语言模型中的翻译不对称性作为数据增强因素：以6种罗曼什语变体为例的研究

    Translation Asymmetry in LLMs as a Data Augmentation Factor: A Case Study for 6 Romansh Language Varieties

    [https://arxiv.org/abs/2603.25489](https://arxiv.org/abs/2603.25489)

    该论文发现，在低资源语言（如罗曼什语）的机器翻译中，利用LLM的翻译不对称性，将数据增强方向从生成目标语言改为生成高资源语言，能显著提升性能并首次实现各变体的流畅翻译。

    

    arXiv:2603.25489v2 公告类型：替换 摘要：近期针对低资源机器翻译的策略依赖于大型语言模型（LLMs）基于高资源语言文本生成合成数据。我们重新审视了这一思路，针对罗曼什语（一种具有6种不同变体的语言）进行研究。LLMs在将其他语言翻译成罗曼什语时容易混淆这些变体，但在将罗曼什语翻译成高资源语言（如德语）时表现相当出色。由于这种不对称性，数据增强的方向成为一个关键选择。我们发现，与近期策略相反，将合成翻译生成到高资源语言中是更优的方法，并且只有这种方法能让我们在德语-罗曼什语翻译中超越Gemini 3 Pro基线（在资源最少的变体上比Gemini高出+23 BLEU分数）。人工评估证实，我们的实验产生了首个能在各个罗曼什语变体中生成流畅翻译的模型。

    arXiv:2603.25489v2 Announce Type: replace  Abstract: Recent strategies for low-resource machine translation rely on LLMs to generate synthetic data based on text in higher-resource languages. We revisit this idea for Romansh, a language with 6 distinct varieties. LLMs tend to confuse these varieties when translating into Romansh, but they are quite good at translating out of Romansh into a high-resource language such as German. Due to this asymmetry, the direction of data augmentation is a crucial choice. We find that contrary to recent strategies, creating synthetic translations into the higher-resource language is the superior approach, and only this approach allows us to surpass a Gemini 3 Pro baseline on German-Romansh translation (+23 BLEU over Gemini in the lowest-resource variety). A human evaluation confirms that our experiments yield the first model that generates fluent translations in the individual Romansh varieties.
    
[^208]: 多方言、多语言、同一文化视角：评估多语言视觉语言模型在孟加拉文化理解中的表现——跨越历史关联语言与区域方言

    Many Dialects, Many Languages, One Cultural Lens: Evaluating Multilingual VLMs for Bengali Culture Understanding Across Historically Linked Languages and Regional Dialects

    [https://arxiv.org/abs/2603.21165](https://arxiv.org/abs/2603.21165)

    本文提出BanglaVerse基准，揭示仅用标准孟加拉语评估会高估多语言视觉语言模型的性能，方言变化显著降低其表现，尤其影响图像描述生成。

    

    孟加拉文化通过地区、方言、历史、饮食、政治、媒体和日常视觉生活得以丰富表达，然而在多模态评估中仍代表性不足。为弥补这一空白，我们推出了BanglaVerse，一个以文化为根基的基准，用于评估多语言视觉语言模型（VLMs）在孟加拉文化中的表现，涵盖历史关联语言和区域方言。该基准基于九个领域的1,152张人工筛选图像，支持视觉问答和图像描述任务，并扩展至四种语言和五种孟加拉方言，生成了约32.2K个数据样本。实验表明，仅评估标准孟加拉语会高估模型真实能力：在方言变异下性能下降，尤其是在图像描述生成方面，而历史关联语言（如印地语和乌尔都语）虽保留部分文化含义，但在结构化推理上较弱。跨领域来看，主要瓶颈在于文化特定视觉语义的捕捉。

    arXiv:2603.21165v3 Announce Type: replace  Abstract: Bangla culture is richly expressed through region, dialect, history, food, politics, media, and everyday visual life, yet it remains underrepresented in multimodal evaluation. To address this gap, we introduce BanglaVerse, a culturally grounded benchmark for evaluating multilingual vision-language models (VLMs) on Bengali culture across historically linked languages and regional dialects. Built from 1,152 manually curated images across nine domains, the benchmark supports visual question answering and captioning, and is expanded into four languages and five Bangla dialects, yielding ~32.2K artifacts. Our experiments show that evaluating only standard Bangla overestimates true model capability: performance drops under dialectal variation, especially for caption generation, while historically linked languages such as Hindi and Urdu retain some cultural meaning but remain weaker for structured reasoning. Across domains, the main bottlen
    
[^209]: 大型语言模型如何扭曲我们的书面语言

    How LLMs Distort Our Written Language

    [https://arxiv.org/abs/2603.18161](https://arxiv.org/abs/2603.18161)

    这项研究通过用户研究和数据集分析，揭示大型语言模型在辅助写作时不仅改变写作风格和语气，还显著扭曲了原始意图，导致大量文本内容趋于中立和失去个性。

    

    arXiv:2603.18161v2 公告类型：替换交叉 摘要：大型语言模型（LLMs）被全球超过十亿人使用，最常见的是用于辅助写作。在这项工作中，我们证明了LLMs不仅改变了人类写作的语音和语气，还持续改变了预期含义。首先，我们进行了一项人类用户研究，以了解人们在使用LLMs进行写作时实际如何互动。我们的发现揭示，大量使用LLMs导致近70%的论文在回答主题问题时保持中立。显著更多的重度LLM用户报告称，写作缺乏创造力且不符合他们的个人风格。接下来，利用一个在2021年LLMs广泛发布之前收集的人类写作论文数据集，我们研究了要求LLM根据数据集中的人类书面反馈修改论文，如何引起结果内容和含义的显著变化。我们发现，即使LLM被提示专家反馈，也会导致内容发生重大改变。

    arXiv:2603.18161v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are used by over a billion people globally, most often to assist with writing. In this work, we demonstrate that LLMs not only alter the voice and tone of human writing but also consistently alter the intended meaning. First, we conduct a human user study to understand how people actually interact with LLMs when using them for writing. Our findings reveal that extensive LLM use led to a nearly 70% increase in essays that remained neutral in answering the topic question. Significantly more heavy LLM users reported that the writing was less creative and not in their voice. Next, using a dataset of human-written essays that was collected in 2021 before the widespread release of LLMs, we study how asking an LLM to revise the essay based on the human-written feedback in the dataset induces large changes in the resulting content and meaning. We find that even when LLMs are prompted with expert feedback an
    
[^210]: 频率至关重要：用于剪枝和量化的快速模型无关数据策展方法

    Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization

    [https://arxiv.org/abs/2603.16105](https://arxiv.org/abs/2603.16105)

    本文提出ZipCal，一种基于Zipf幂律最大化词汇多样性的模型无关数据策展方法，通过分析数据内在属性而非模型信号，为剪枝和量化高效选择高性能校准数据，显著优于均匀随机采样。

    

    训练后模型压缩对于增强大型语言模型（LLMs）的可移植性同时保持其性能至关重要。尽管已有多种压缩方法被提出，但对于选择最合适的数据集（即所谓的“校准数据”）以确定压缩模型配置的重视程度较低。校准数据的选择是保持模型在任务内和任务间能力的关键步骤。在本工作中，我们通过分析数据的内在属性而非模型特定信号，来解决为剪枝和量化识别高性能校准集的挑战。我们引入了ZipCal，一种基于Zipf幂律最大化词汇多样性的模型无关数据策展策略。实验表明，我们的方法在各种剪枝基准测试中优于标准的均匀随机采样方法。值得注意的是，它在其他方面也表现出色。

    arXiv:2603.16105v4 Announce Type: replace-cross  Abstract: Post-training model compression is essential for enhancing the portability of Large Language Models (LLMs) while preserving their performance. While several compression approaches have been proposed, less emphasis has been placed on selecting the most suitable set of data (the so-called \emph{calibration data}) for finding the compressed model configuration. The choice of calibration data is a critical step in preserving model capabilities both intra- and inter-tasks. In this work, we address the challenge of identifying high-performance calibration sets for both pruning and quantization by analyzing intrinsic data properties rather than model-specific signals. We introduce ZipCal, a model-agnostic data curation strategy that maximizes lexical diversity based on Zipfian power laws. Experiments demonstrate that our method outperforms standard uniform random sampling across various pruning benchmarks. Notably, it also performs on
    
[^211]: SPAR-K：面向口语语言模型的调度周期性交替早退机制

    SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models

    [https://arxiv.org/abs/2603.09215](https://arxiv.org/abs/2603.09215)

    SPAR-K通过模态感知的交替深度调度和周期性全深度刷新步骤，在保持语音问答准确性的同时显著降低解码深度，实现了高效的SLM推理。

    

    交错式口语语言模型（SLMs）交替生成文本和语音标记，但在每个步骤中都以完整的Transformer深度进行解码成本高昂，尤其是对于长语音序列。我们提出了SPAR-K，一种模态感知的早退框架，旨在加速交错式SLM推理，同时保持感知质量。SPAR-K引入了一种语音交替深度调度：大多数语音位置在固定的中间层退出，而周期性的全深度“刷新”步骤则缓解了早退导致的分布偏移。我们使用Step-Audio-2-mini和GLM-4-Voice在四个数据集上评估了该框架，涵盖推理、事实问答和对话任务，并以ASR转录准确性和感知质量作为性能指标。实验结果表明，SPAR-K在很大程度上保持了问答准确性，最大准确率下降仅为0.82%，同时平均语音解码深度最多减少了相应比例。

    arXiv:2603.09215v2 Announce Type: replace  Abstract: Interleaved spoken language models (SLMs) alternately generate text and speech tokens, but decoding at full transformer depth for every step becomes costly, especially due to long speech sequences. We propose SPAR-K, a modality-aware early exit framework designed to accelerate interleaved SLM inference while preserving perceptual quality. SPAR-K introduces a speech alternating-depth schedule: most speech positions exit at a fixed intermediate layer, while periodic full-depth "refresh" steps mitigate distribution shift due to early exit. We evaluate our framework using Step-Audio-2-mini and GLM-4-Voice across four datasets spanning reasoning, factual QA, and dialogue tasks, measuring performance in terms of ASR transcription accuracy and perceptual quality. Experimental results demonstrate that SPAR-K largely preserves question-answering accuracy with a maximum accuracy drop of 0.82\% while reducing average speech decoding depth by up
    
[^212]: 模块化专家合并用于生物医学检索

    Modular Expert Merging for Biomedical Retrieval

    [https://arxiv.org/abs/2602.04731](https://arxiv.org/abs/2602.04731)

    本文提出模块化专家合并方法，通过合成难负样本和LoRA微调领域专家并合并，在生物医学检索上优于大规模混合训练，兼顾通用性能。

    

    arXiv:2602.04731v2 公告类型：替换 摘要：将通用大型语言模型适配为领域专用的密集检索器通常需要在混合领域数据上进行大规模训练。我们表明，合并独立训练的领域专用专家在四个仅解码器LLM家族（0.6B-7B）、四种合并方法和来自MTEB的十二项医学及通用检索任务中持续优于这种方法，这表明参数空间组合捕捉了互补的领域优势，而大规模混合领域训练则将其平均化。为了进一步最大化专家质量，我们引入了Synthesize-Train-Merge（STM），这是一个模块化框架，它使用顶级LLM合成难负样本，并通过LoRA微调领域专用专家后再进行合并，无需持续预训练。合成的难负样本对较小模型带来最大收益，STM在生物医学检索任务上实现了强劲性能，同时保持了具有竞争力的通用领域结果。

    arXiv:2602.04731v2 Announce Type: replace  Abstract: Adapting general-purpose LLMs into domain-specialized dense retrievers typically requires large-scale training on mixed-domain data. We show that merging independently trained domain-specialized experts consistently exceeds this approach across four decoder-only LLM families (0.6B-7B), four merging methods, and twelve medical and general retrieval tasks from MTEB, suggesting that parameter-space composition captures complementary domain strengths that large-scale mixed-domain training averages out. To further maximize expert quality, we introduce Synthesize-Train-Merge (STM), a modular framework that synthesizes hard negatives with a top-tier LLM and fine-tunes domain-specialized experts via LoRA before merging them, without continual pre-training. Synthesized hard negatives yield the largest gains for smaller models, and STM achieves strong performance on biomedical retrieval tasks while maintaining competitive general-domain result
    
[^213]: 超越事实问答：面向长篇多语言内容的导师式问答

    Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content

    [https://arxiv.org/abs/2601.17173](https://arxiv.org/abs/2601.17173)

    本文首次提出导师式问答数据集MentorQA及评估框架，证明多智能体架构在长篇多语言内容中能显著提升超越事实准确性的导师式回应质量。

    

    问答系统通常以事实正确性为评估标准，然而许多实际应用——如教育和职业指导——需要导师式回应，即提供反思和指导的回答。现有问答基准很少捕捉这一区别，尤其是在多语言和长篇场景中。我们引入了MentorQA，这是首个面向导师式问答的多语言数据集和评估框架，基于长篇视频构建，包含来自四种语言、180小时内容中的近9000个问答对。我们定义了超越事实准确性的导师式评估维度，涵盖清晰度、对齐度和学习价值。利用MentorQA，我们在受控条件下比较了单智能体、双智能体、检索增强生成和多智能体问答架构。多智能体流水线持续生成更高质量的导师式回应，尤其在复杂主题和较低资源语言场景中表现出显著优势。

    arXiv:2601.17173v2 Announce Type: replace-cross  Abstract: Question answering systems are typically evaluated on factual correctness, yet many real-world applications-such as education and career guidance-require mentorship: responses that provide reflection and guidance. Existing QA benchmarks rarely capture this distinction, particularly in multilingual and long-form settings. We introduce MentorQA, the first multilingual dataset and evaluation framework for mentorship-focused question answering from long-form videos, comprising nearly 9,000 QA pairs from 180 hours of content across four languages. We define mentorship-focused evaluation dimensions that go beyond factual accuracy, capturing clarity, alignment, and learning value. Using MentorQA, we compare Single-Agent, Dual-Agent, RAG, and Multi-Agent QA architectures under controlled conditions. Multi-Agent pipelines consistently produce higher-quality mentorship responses, with especially strong gains for complex topics and lower-
    
[^214]: MAPLE：基于元数据条件化的大语言模型预训练用于地域感知问答

    MAPLE: Metadata Conditioned LLM Pretraining for Locale-Aware Question Answering

    [https://arxiv.org/abs/2601.15236](https://arxiv.org/abs/2601.15236)

    本文提出MAPLE方法，通过在预训练时加入地理元数据，使语言模型能够根据地域变化切换答案，解决局部知识消歧问题。

    

    大型语言模型能够记住相互冲突的地域特定事实，但在地域变化时却无法在这些事实之间进行选择，反而默认给出一个全球占主导的单一答案。我们将此形式化为局部知识消歧，并引入了LocalNewsQA，这是一个包含18,700条英文新闻的基准测试，它将同一问题配对到两个地域，并评估模型在地域变化时是否真正切换其答案。我们还引入了MAPLE，这是一个受控的解码器专用模型系列，其预训练利用了训练语料中已有的文档级地理元数据（来源URL、国家和大陆），并将其与在相同数据、相同令牌预算、架构和优化条件下训练的无元数据对照组进行比较。在1B和3B规模的受控实验中，当推理时元数据固定时，使用MAPLE中的元数据进行预训练产生了可测量的切换效果，并提高了那些正确答案依赖于特定地域的问题的准确性。

    arXiv:2601.15236v2 Announce Type: replace  Abstract: Large language models can memorize competing locale-specific facts yet fail to select among them when the locale changes, defaulting instead to a single globally dominant answer. We formalize this as localized knowledge disambiguation and introduce LocalNewsQA, an 18,700-item English-news benchmark that pairs the same question across two locales and scores whether a model actually switches its answer when the locale changes. We also introduce MAPLE, a controlled family of decoder-only models pretrained with document-level geographic metadata (source URL, country, and continent) already present in the training corpus, and compare it to metadata-free controls trained on identical data with the same token budget, architecture, and optimization. In controlled experiments at 1B and 3B, with inference-time metadata fixed, pretraining with metadata in MAPLE produces measurable switching and improves accuracy on questions whose correct answe
    
[^215]: CounterVid：用于缓解视频-语言模型中动作与时间幻觉的反事实视频生成

    CounterVid: Counterfactual Video Generation for Mitigating Action and Temporal Hallucinations in Video-Language Models

    [https://arxiv.org/abs/2601.04778](https://arxiv.org/abs/2601.04778)

    提出了一种反事实视频生成框架，通过合成仅改变动作或时间顺序的硬负样本视频，有效缓解视频-语言模型中的动作和时间幻觉问题。

    

    arXiv:2601.04778v2 公告类型：替换交叉 摘要：视频-语言模型（VLMs）实现了强大的多模态理解，但在推理动作和时间顺序时仍容易产生幻觉。现有的缓解策略，如文本过滤或随机视频扰动，往往无法解决根本原因：过度依赖语言先验而非细粒度的视觉动态。我们提出了一种可扩展的反事实视频生成框架，该框架合成仅在动作或时间结构上不同、同时保持场景上下文的视频。我们的流程结合了用于动作提议和编辑指导的多模态大语言模型，以及基于扩散的图像和视频模型，以规模化生成语义硬负样本。利用该框架，我们构建了CounterVid，一个包含约26k个偏好对的合成数据集，这些偏好对由短反事实动作片段构成，并针对动作识别和受控动作序列排序。我们进一步引入了...

    arXiv:2601.04778v2 Announce Type: replace-cross  Abstract: Video-language models (VLMs) achieve strong multimodal understanding but remain prone to hallucinations, especially when reasoning about actions and temporal order. Existing mitigation strategies, such as textual filtering or random video perturbations, often fail to address the root cause: over-reliance on language priors rather than fine-grained visual dynamics. We propose a scalable framework for counterfactual video generation that synthesizes videos differing only in actions or temporal structure while preserving scene context. Our pipeline combines multimodal LLMs for action proposal and editing guidance with diffusion-based image and video models to generate semantic hard negatives at scale. Using this framework, we build CounterVid, a synthetic dataset of ~26k preference pairs constructed from short counterfactual action clips and targeting both action recognition and controlled action-sequence ordering. We further intr
    
[^216]: AEScorer：一种基于证据的代理式分级事实性验证框架

    AEScorer: An Agentic Evidence-Grounded Framework for Graded Factuality Verification

    [https://arxiv.org/abs/2601.03605](https://arxiv.org/abs/2601.03605)

    AEScorer提出了一种代理式证据获取与分级评分框架，用于区分事实性的细微差异，并通过新基准GradedVeriBench验证了其优越性能。

    

    尽管大型语言模型（LLMs）取得了显著进展，但其事实性仍然是一个关键挑战，这导致了对更细致事实性验证的日益增长的需求。现有的事实性验证方法未能捕捉分级判断，尽管事实性更应被视为一个频谱而非对错的二元对立。为弥合这一差距，我们专注于分级事实性验证，并提出了AEScorer，一个基于证据的代理式框架，包含两个阶段：代理式证据获取和分级评分。AEScorer首先通过代理式搜索收集并精炼外部证据，然后预测一个标量事实性分数，以区分事实正确性的细微差异。我们进一步构建了GradedVeriBench，一个涵盖通用和多跳问答的分级事实性验证基准。在GradedVeriBench上的实验结果表明，AEScorer显著优于现有方法。

    arXiv:2601.03605v2 Announce Type: replace  Abstract: Despite the significant advancements of Large Language Models (LLMs), their factuality remains a critical challenge, creating a growing need for more nuanced factuality verification. Existing factuality verification methods do not capture graded judgments, even though factuality is better understood as a spectrum rather than a binary of right and wrong. To bridge this gap, we focus on graded factuality verification and propose AEScorer, an agentic evidence-grounded framework with two stages: agentic evidence acquisition and graded scoring. AEScorer first gathers and refines external evidence through agentic search, and then predicts a scalar factuality score to distinguish nuanced differences in factual correctness. We further construct GradedVeriBench, a benchmark for graded factuality verification spanning both general and multi-hop question answering. Experimental results on GradedVeriBench show that AEScorer substantially outperf
    
[^217]: TopiCLEAR：基于自适应嵌入聚类的短文本可解释主题发现

    TopiCLEAR: Adaptive embedding clustering for interpretable topic discovery from short texts

    [https://arxiv.org/abs/2512.06694](https://arxiv.org/abs/2512.06694)

    本文提出TopiCLEAR框架，通过自适应降维的迭代嵌入聚类，验证了人类可解释主题对应低维结构的假设，从而提升短文本主题发现的解释性。

    

    主题发现是文本挖掘中的一项基础技术，用于识别大型文档集合中的抽象主题。近期的一种主题发现方法是对文档或句子嵌入（通常从预训练语言模型中获得）进行聚类，并将每个聚类表示为一个主题。尽管这些方法在实证上表现出色，但连接嵌入空间几何结构与人类可解释主题组织之间的设计原则仍不明确。因此，阐明这些几何结构与主题可解释性之间的关系是主题发现中的关键挑战。在本研究中，我们提出了TopiCLEAR（基于自适应降维的嵌入聚类主题发现），这是一个简单框架，集成了文档嵌入与基于自适应降维的迭代聚类。TopiCLEAR基于一个假设：人类可解释的主题对应于低维结构。

    arXiv:2512.06694v2 Announce Type: replace  Abstract: Topic discovery is a fundamental technique for text mining that identifies abstract topics within large document collections. A recent approach to topic discovery is to cluster document or sentence embeddings, typically obtained from pre-trained language models, and represent each cluster as a topic. Despite their strong empirical performance, the design principles linking the geometry of embedding spaces to human-interpretable topic organization remain unclear. Clarifying the relationship between these geometric structures and topic interpretability is therefore a key challenge in topic discovery. In this study, we propose TopiCLEAR (Topic discovery by CLustering Embeddings with Adaptive dimensionality Reduction), a simple framework that integrates document embeddings with iterative clustering based on adaptive dimensionality reduction. TopiCLEAR is guided by the hypothesis that human-interpretable topics correspond to low-dimension
    
[^218]: “无痕”心智记住了什么：知识纠缠如何决定大语言模型在遗忘后泄露的内容

    What the "Spotless" Mind Remembers: How Knowledge Entanglement Shapes What Leaks After Unlearning in LLMs

    [https://arxiv.org/abs/2510.25732](https://arxiv.org/abs/2510.25732)

    本文发现，大语言模型在遗忘后，知识纠缠度越高的事实越易泄露，且不同遗忘算法（如GA+KL）会反转这一关系，首次揭示其因果影响。

    

    大语言模型（LLMs）中的“遗忘”通常通过“被遗忘”的事实是否可恢复来评估。我们转而探究一个事实与模型其余知识的结构性纠缠是否预测其在遗忘后泄露的可能性、这种关系是否会系统性变化，以及是否具有因果性。在多种遗忘算法（WHP和GA+KL）、两个领域（虚构的《哈利·波特》和非虚构的2000-2010年美国参议员）以及四个模型（2.7B-13B参数）中，我们发现遗忘前，纠缠程度更高的事实被更频繁地回忆（r = +0.39至+0.51）。WHP削弱了这种关系但仍保持正相关（r = +0.16至+0.33）；GA+KL则在每个领域和模型规模下都使其反转（r = -0.14至-0.25）。据我们所知，这是遗忘文献中首次报告这种特定的反转现象。为确认这不仅仅是相关性而是因果性，我们直接操纵提示的纠缠得分。

    arXiv:2510.25732v2 Announce Type: replace-cross  Abstract: Unlearning in large language models (LLMs) is usually evaluated as whether an "unlearned" fact can be recovered. We instead ask whether a fact's structural entanglement with the rest of a model's knowledge predicts whether it leaks after unlearning, whether this relationship changes systematically, and whether it is causal. Across varied unlearning algorithms (WHP and GA+KL), two domains (both fictional Harry Potter and non-fictional U.S. Senators, 2000-2010), and four models (2.7B-13B parameters), we find that before unlearning, more entangled facts are recalled more often (r = +0.39 to +0.51). WHP weakens this relationship but stays positive (r = +0.16 to +0.33); GA+KL inverts it in every domain and model size (r = -0.14 to -0.25). To our knowledge, this is the first report of this specific reversal in the unlearning literature. To confirm this is causal beyond correlation, we directly manipulate a prompt's entanglement score
    
[^219]: MENTOR：通过灵活的教师优化奖励进行工具使用蒸馏的强化学习

    MENTOR: Reinforcement Learning via Flexible Teacher-Optimized Rewards for Tool-Use Distillation

    [https://arxiv.org/abs/2510.18383](https://arxiv.org/abs/2510.18383)

    本文提出MENTOR，一种基于强化学习的在策略蒸馏框架，通过灵活且过程感知的奖励结构，在行为对齐和性能之间取得平衡，有效解决了将大型语言模型工具使用能力蒸馏到小型模型时的域外泛化问题。

    

    将大型语言模型（LLMs）的工具使用能力蒸馏到小型语言模型（SLMs）中，对于其实际应用至关重要。主流方法监督微调（SFT）是一种离策略蒸馏方法，由于它僵硬地与静态教师轨迹对齐，导致其在域外（OOD）泛化方面表现不佳。虽然强化学习（RL）提供了一种替代方案，但SLMs的能力限制带来了严峻的困境：稀疏的结果奖励提供的指导不足，而严格的轨迹匹配则施加了过于严格的约束。为了弥合这一能力驱动的差距，我们提出了MENTOR，一种在策略蒸馏框架，引入了灵活且过程感知的奖励结构。MENTOR不强制僵硬的复制，而是利用教师的参考来引导工具使用行为，在行为对齐与下游性能之间取得平衡。大量实验表明...

    arXiv:2510.18383v4 Announce Type: replace-cross  Abstract: Distilling the tool-use capabilities of large language models (LLMs) into small language models (SLMs) is essential for their practical application. The predominant approach, supervised fine-tuning (SFT), is an off-policy distillation method that suffers from poor out-of-domain (OOD) generalization because it rigidly aligns with static teacher trajectories. While reinforcement learning (RL) offers an alternative, the capacity limitations of SLMs pose a severe dilemma: sparse outcome rewards provide insufficient guidance, whereas strict trajectory matching imposes overly restrictive constraints. To bridge this capacity-driven gap, we propose MENTOR, an on-policy distillation framework that introduces a flexible yet process-aware reward structure. Instead of enforcing rigid replication, MENTOR uses the teacher's reference to guide tool-use behavior, balancing behavioral alignment with downstream performance. Extensive experiments
    
[^220]: DeepPlanner：通过优势塑造扩展深度研究代理的规划能力

    DeepPlanner: Scaling Planning Capability for Deep Research Agents via Advantage Shaping

    [https://arxiv.org/abs/2510.12979](https://arxiv.org/abs/2510.12979)

    DeepPlanner提出了一种端到端强化学习框架，通过熵基优势塑造和样本级加权，有效增强了深度研究代理的规划能力，解决了规划令牌在标准RL下未被充分优化的问题。

    

    arXiv:2510.12979v2 公告类型：替换  摘要：大型语言模型（LLMs）结合多步推理和动作生成能力，在利用外部工具处理需要长周期规划的复杂任务方面显示出潜力。然而，现有方法要么依赖推理阶段的隐式规划，要么引入显式规划器但未系统解决如何优化规划阶段的问题。作为证据，我们观察到在标准强化学习（RL）下，规划令牌的熵显著高于其他动作令牌，揭示了仍未被充分优化的不确定决策点。为解决这一问题，我们提出了DeepPlanner，一种端到端RL框架，有效增强深度研究代理的规划能力。我们的方法通过基于熵的项塑造令牌级优势，为高熵令牌分配更大的更新，并选择性地提升规划密集型样本的样本级优势。

    arXiv:2510.12979v2 Announce Type: replace  Abstract: Large language models (LLMs) augmented with multi-step reasoning and action generation abilities have shown promise in leveraging external tools to tackle complex tasks that require long-horizon planning. However, existing approaches either rely on implicit planning in the reasoning stage or introduce explicit planners without systematically addressing how to optimize the planning stage. As evidence, we observe that under vanilla reinforcement learning (RL), planning tokens exhibit significantly higher entropy than other action tokens, revealing uncertain decision points that remain under-optimized. To address this, we propose DeepPlanner, an end-to-end RL framework that effectively enhances the planning capabilities of deep research agents. Our approach shapes token-level advantage with an entropy-based term to allocate larger updates to high entropy tokens, and selectively upweights sample-level advantages for planning-intensive ro
    
[^221]: 面向检索增强生成的LLM特定效用

    LLM-Specific Utility for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2510.11358](https://arxiv.org/abs/2510.11358)

    本文首次形式化并实证了检索增强生成中证据的LLM特定效用，证明其具有模型依赖性和不可转移性，为优化RAG系统提供了新视角。

    

    arXiv:2510.11358v3 公告类型：替换-交叉 摘要：检索增强生成（RAG）通常针对主题相关性进行优化，但其成功最终取决于检索到的段落是否有助于大型语言模型（LLM）生成正确且完整的答案。我们认为，这种效用往往是LLM特定的，而非普遍通用的，这归因于模型在知识、推理和利用证据能力方面的差异。我们将LLM特定效用形式化为，当提供某个段落时，目标LLM的性能相比无证据作答时的提升幅度。为系统研究LLM特定效用，我们构建了一个基准，针对四个LLM（Qwen3-8B/14B/32B和Llama 3.1-8B）在三个问答数据集（Natural Questions、TriviaQA和MS MARCO-FQA）上提供了LLM特定的黄金效用段落。我们的分析表明，效用段落具有模型依赖性和不可转移性：每个LLM在其自身的效用证据下表现最佳，而为其他模型优化的证据则表现不佳。

    arXiv:2510.11358v3 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) is typically optimized for topical relevance, yet its success ultimately depends on whether retrieved passages are useful for a large language model (LLM) to generate correct and complete answers. We argue that such utility is often LLM-specific rather than universal, due to differences in models' knowledge, reasoning, and ability to leverage evidence. We formalize LLM-specific utility as the performance improvement of a target LLM when a passage is provided, compared to answering without evidence. To systematically study LLM-specific utility, we construct a benchmark of LLM-specific gold utilitarian passages for four LLMs (Qwen3-8B/14B/32B and Llama 3.1-8B) on three QA datasets (Natural Questions, TriviaQA, and MS MARCO-FQA). Our analysis shows that utilitarian passages are model-dependent and non-transferable: each LLM performs best with its own utilitarian evidence, while evidence optimiz
    
[^222]: 大型语言模型是有效的知识图谱构建器吗？

    Are Large Language Models Effective Knowledge Graph Constructors?

    [https://arxiv.org/abs/2510.11297](https://arxiv.org/abs/2510.11297)

    本文提出D2A-HKG框架，评估了七个LLM在零样本无模式条件下构建文档知识图谱的能力，发现它们能生成合理结构但在复杂关系处理上不足。

    

    arXiv:2510.11297v2 公告类型：替换。摘要：知识图谱（KGs）广泛用于知识密集型应用，但目前尚不清楚当前大型语言模型（LLMs）在零样本、无模式设置下，不依赖复杂任务特定框架时，能否有效构建文档为基础的知识图谱。我们引入了从细节到抽象的层次化知识图谱（D2A-HKG）构建框架，该框架将知识图谱构建分解为三个步骤：初始提取、拆分和抽象，并从语义和结构两个角度评估生成的图谱。使用七个前沿LLMs，我们在CMW-Lit数据集上对零样本知识图谱构建进行了基准测试，该数据集源自关于儿童心理健康的已发表儿科研究文章。CMW-Lit因其异质证据、相互关联的因素和复杂的统计限定关系而提供了具有挑战性的测试平台。我们的结果显示，最先进的LLMs通常能产生合理的图谱结构，但在处理复杂关系时存在局限性。

    arXiv:2510.11297v2 Announce Type: replace  Abstract: Knowledge graphs (KGs) are widely used in knowledge-intensive applications, yet it remains unclear how effectively current large language models (LLMs) can construct document-grounded KGs in a zero-shot, schema-free setting without relying on complex task-specific frameworks. We introduce Detail-to-Abstract Hierarchical Knowledge Graph (D2A-HKG) construction framework, which decomposes KG construction into three stages: initial extraction, splitting, and abstraction, and evaluates the resulting graphs from both semantic and structural perspectives. Using seven frontier LLMs, we benchmark zero-shot KG construction on CMW-Lit, a dataset derived from published paediatric research articles on children's mental well-being. CMW-Lit provides a challenging test bed due to its heterogeneous evidence, interconnected factors, and complex, statistically qualified relationships. Our results show that state-of-the-art LLMs can generally produce re
    
[^223]: GSM8K-V：视觉语言模型能否在视觉情境中解决小学数学应用题

    GSM8K-V: Can Vision Language Models Solve Grade School Math Word Problems in Visual Contexts

    [https://arxiv.org/abs/2509.25160](https://arxiv.org/abs/2509.25160)

    该论文提出 GSM8K-V 基准，将文本数学题转化为多图像序列，发现视觉语言模型在视觉数学推理上存在显著模态差距（最佳模型仅 59%，远低于人类 91%）。

    

    数学推理是视觉语言模型（VLMs）的关键能力，然而当前的基准测试主要评估基于文本或显式符号化的视觉输入。目前尚不清楚，当信息必须从图像中感知和推断，而非从显式符号中读取时，VLMs 能否进行数学推理。我们引入了 GSM8K-V，一个将 GSM8K 转化为多图像序列并保持语义等价的基准测试。通过自动化流程和人工验证，将基于文本的问题映射为视觉形式，我们筛选出 1,319 个高质量样本。在 GSM8K-V 中，数量必须通过视觉感知提取，推理链必须通过整合跨场景的隐含线索来重建。对 34 个 VLMs 的评估揭示了一个显著的模态差距：虽然大多数模型在文本上超过 90%，但最佳模型在 GSM8K-V 上仅达到 59%，远低于人类 91% 的准确率。值得注意的是，为视觉增强的模型表现并未显著改善。

    arXiv:2509.25160v2 Announce Type: replace-cross  Abstract: Mathematical reasoning is a key capability for vision-language models (VLMs), yet current benchmarks mainly evaluate text-based or explicitly symbolic visual inputs. It remains unclear whether VLMs can reason mathematically when information must be perceived and inferred from images rather than read from explicit symbols. We introduce GSM8K-V, a benchmark transforming GSM8K into multi-image sequences with semantic equivalence preserved. By mapping text-based problems into visual form via an automated pipeline and human verification, we curate 1,319 high-quality samples. In GSM8K-V, quantities must be extracted through visual perception, and reasoning chains must be reconstructed by integrating implicit cues across scenes. Evaluation of 34 VLMs reveals a striking modality gap: while most models exceed 90\% on text, the best model achieves only 59\% on GSM8K-V, far below the 91\% human accuracy. Notably, models enhanced for visua
    
[^224]: 递归与Transformer结合实现通用多模态检索

    Recurrence Meets Transformers for Universal Multimodal Retrieval

    [https://arxiv.org/abs/2509.08897](https://arxiv.org/abs/2509.08897)

    本文提出ReT-2，一个结合递归机制与Transformer架构的统一多模态检索模型，支持多模态查询和跨模态文档检索，并在M2KR和M-BEIR基准上达到最先进性能。

    

    摘要：随着多模态检索的快速发展及其在大语言模型和多模态大语言模型中的应用，出现了越来越复杂的检索任务。现有方法主要依赖于对视觉-语言模型进行任务特定的微调，并且局限于单一模态的查询或文档。在本文中，我们提出了ReT-2，一个统一检索模型，支持由图像和文本组成的多模态查询，并在文本与图像共存的跨模态文档集合中进行搜索。ReT-2利用多层表示和带有LSTM启发门控机制的递归Transformer架构，动态整合跨层和跨模态的信息，捕捉细粒度的视觉和文本细节。我们在具有挑战性的M2KR和M-BEIR基准上，针对不同检索配置评估了ReT-2。结果表明，ReT-2在各项指标上持续达到最先进的性能。

    arXiv:2509.08897v2 Announce Type: replace-cross  Abstract: With the rapid advancement of multimodal retrieval and its application in LLMs and multimodal LLMs, increasingly complex retrieval tasks have emerged. Existing methods predominantly rely on task-specific fine-tuning of vision-language models and are limited to single-modality queries or documents. In this paper, we propose ReT-2, a unified retrieval model that supports multimodal queries, composed of both images and text, and searches across multimodal document collections where text and images coexist. ReT-2 leverages multi-layer representations and a recurrent Transformer architecture with LSTM-inspired gating mechanisms to dynamically integrate information across layers and modalities, capturing fine-grained visual and textual details. We evaluate ReT-2 on the challenging M2KR and M-BEIR benchmarks across different retrieval configurations. Results demonstrate that ReT-2 consistently achieves state-of-the-art performance acr
    
[^225]: LLM分析150多年德国议会关于移民的辩论揭示从战后团结到近十年反团结的转变

    LLM Analysis of 150+ years of German Parliamentary Debates on Migration Reveals Shift from Post-War Solidarity to Anti-Solidarity in the Last Decade

    [https://arxiv.org/abs/2509.07274](https://arxiv.org/abs/2509.07274)

    本文利用大型语言模型对150多年德国议会移民辩论进行大规模分析，发现最强模型（如GPT-5）的标注性能接近人类水平，并揭示出近十年从战后团结向反团结的显著转变。

    

    移民一直是德国政治辩论的核心话题，从战后流离失所者的迁移，到劳动力移民，再到近期的难民潮。传统上，对此类政治话语的大规模分析需要大量的人工标注，限制了覆盖范围。大型语言模型（LLM）提供了一种可扩展的替代方案。我们采用理论驱动的标注方案，考察LLM在德国议会辩论中标注团结与反团结子类型的表现，以及生成的标签是否支持有效的下游推断。我们首先评估了多个LLM，涵盖模型规模、提示策略、微调、历史与当代数据对比以及系统性错误。最强的模型，尤其是GPT-5和gpt-oss-120B，其宏F1分数可与人类一致性相媲美，尽管它们的系统性错误可能使下游结果产生偏差。因此，我们将软标签模型输出与基于设计的有监督学习相结合。

    arXiv:2509.07274v4 Announce Type: replace  Abstract: Migration has been a core topic in German political debate, from postwar expellee displacement to labor migration and recent refugee movements. Large-scale analysis of such political discourse has traditionally required extensive manual annotation, limiting coverage. Large language models (LLMs) offer a scalable alternative. Using a theory-driven annotation scheme, we examine how well LLMs annotate subtypes of solidarity and anti-solidarity in German parliamentary debates and whether the resulting labels support valid downstream inference. We first evaluate multiple LLMs across model size, prompting strategies, fine-tuning, historical versus contemporary data, and systematic errors. The strongest models, especially GPT-5 and gpt-oss-120B, achieve macro- F1 scores comparable to human agreement, although their systematic errors can bias downstream results. We therefore combine soft-label model outputs with Design-based Supervised Learn
    
[^226]: 语言模型是否遵循奥卡姆剃刀原则？归纳与溯因推理中简约性的评估

    Do Language Models Follow Occam's Razor? An Evaluation of Parsimony in Inductive and Abductive Reasoning

    [https://arxiv.org/abs/2509.03345](https://arxiv.org/abs/2509.03345)

    本文首次系统评估了大型语言模型在归纳和溯因推理中是否遵循奥卡姆剃刀原则，并引入了一个可扩展的一阶逻辑推理问题生成框架。

    

    arXiv:2509.03345v3 公告类型：替换 摘要：非演绎推理，包括归纳推理和溯因推理，对于处理复杂的现实世界问题至关重要。归纳和溯因推理的一个关键特征是存在许多有效假设；最简单的假设（那些遵循奥卡姆剃刀原则的）往往最为有用。然而，最近评估大型语言模型（LLMs）非演绎推理能力的研究忽略了这个方面。本研究填补了这一空白，重点关注LLMs的归纳和溯因推理能力是否遵循奥卡姆剃刀原则，同时考察其推理的正确性。为实现这一目标，我们引入了一个框架来合成生成推理问题，这些问题（a）同时需要归纳推理和溯因推理；（b）易于扩展以产生任何可用一阶逻辑表达的溯因/归纳推理问题。智能体的任务是...

    arXiv:2509.03345v3 Announce Type: replace  Abstract: Non-deductive reasoning, encompassing inductive and abductive reasoning, is essential in addressing complex real-world questions. One key feature of inductive and abductive reasoning is that there are many valid hypotheses; the simplest ones (those that adhere to Occam's Razor) are often most useful. However, this aspect is ignored in recent work that evaluates the non-deductive reasoning capabilities of large language models (LLMs). This work fills this gap, focusing on understanding whether the inductive and abductive reasoning capabilities of LLMs adhere to Occam's Razor, while also examining the correctness of their reasoning. To accomplish this goal, we introduce a framework to synthetically generate reasoning questions that (a) require inductive reasoning and abductive reasoning simultaneously; (b) is readily extended to produce any abductive/inductive reasoning question expressible in first-order logic. The task for the intell
    
[^227]: 超越罗塞塔石碑：泛化动力学中的统一力量

    Beyond the Rosetta Stone: Unification Forces in Generalization Dynamics

    [https://arxiv.org/abs/2508.11017](https://arxiv.org/abs/2508.11017)

    通过合成数据上的小型Transformer实验，本文揭示了跨语言知识迁移的关键在于表示统一性，这取决于信息性和可提取性，并提供了统一理论解释多语言模型中的迁移现象。

    

    大语言模型（LLMs）在跨语言知识迁移方面存在困难：当用某种语言询问训练中另一种语言表达的事实时，它们有时会产生幻觉。本研究引入了一个受控环境，通过在合成的多语言数据集上从头训练小型Transformer模型，来研究这一现象的原因和训练动态。根据（1）事实与其学习所用语言之间的相关性（信息性）和（2）语言识别的难易程度（可提取性），模型要么发展出跨语言的统一表示，要么发展出分离表示；只有表示统一时，事实才能跨语言迁移。基于这些见解，我们提出了一个统一视角，解释了一系列关于多语言LLM跨语言迁移的先前观察结果。我们的工作表明，受控环境可以揭示预训练的机制。

    arXiv:2508.11017v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) struggle with cross-lingual knowledge transfer: they sometimes hallucinate when asked in one language about facts expressed in a different language during training. This work introduces a controlled setting to study the causes and training dynamics of this phenomenon by training small Transformer models from scratch on synthetic multilingual datasets. Depending on (1) the correlation between facts and the language they were learned in (informativeness), and (2) the ease of language identification (extractability), models either develop unified representations across languages or separate representations; only when representations are unified do facts transfer across languages. Based on these insights, we propose a unifying perspective which explains a range of prior observations concerning cross-lingual transfer in multilingual LLMs. Our work shows controlled settings can shed light on pre-training 
    
[^228]: AirLLM：基于扩散策略的自适应LoRA空中远程微调大语言模型

    AirLLM: Diffusion Policy-based Adaptive LoRA for Remote Fine-Tuning of LLM over the Air

    [https://arxiv.org/abs/2507.11515](https://arxiv.org/abs/2507.11515)

    AirLLM提出了一种基于分层扩散策略的自适应LoRA框架，通过PPO智能体联合考虑无线状态和语言复杂性来动态调整秩配置，从而显著提高空中远程微调的通信效率。

    

    在边缘设备上运行大型语言模型（LLM）日益受到有限通信带宽以及计算和内存成本压力的挑战。因此，云辅助的远程微调变得不可或缺。然而，现有的低秩适应（LoRA）方法通常采用固定或启发式的秩配置，随后通过空中传输所有LoRA参数可能相当低效。为解决这一限制，我们开发了AirLLM，一种用于通信感知LoRA自适应的分层扩散策略框架。具体而言，AirLLM将秩配置建模为覆盖所有插入LoRA投影的结构化动作向量。为解决底层高维序列决策问题，一个近端策略优化（PPO）智能体通过联合观测无线状态和语言复杂性生成粗粒度决策，随后通过扩散模型进行细化。

    arXiv:2507.11515v2 Announce Type: replace-cross  Abstract: Operating Large Language Models (LLMs) on edge devices is increasingly challenged by limited communication bandwidth and strained computational and memory costs. Thus, cloud-assisted remote fine-tuning becomes indispensable. Nevertheless, existing Low-Rank Adaptation (LoRA) approaches typically employ fixed or heuristic rank configurations, and the subsequent over-the-air transmission of all LoRA parameters could be rather inefficient. To address this limitation, we develop AirLLM, a hierarchical diffusion policy framework for communication-aware LoRA adaptation. Specifically, AirLLM models the rank configuration as a structured action vector that spans all LoRA-inserted projections. To solve the underlying high-dimensional sequential decision-making problem, a Proximal Policy Optimization (PPO) agent generates coarse-grained decisions by jointly observing wireless states and linguistic complexity, which are then refined via De
    
[^229]: 四种大语言模型评估器对日语模拟AI对AI咨询的逐次评分信度与专家小组一致性不同特征

    Distinct Profiles of Run-to-Run Score Reliability and Expert-Panel Alignment Across Four LLM Evaluators of Simulated Japanese-Language AI-to-AI Counseling

    [https://arxiv.org/abs/2507.02950](https://arxiv.org/abs/2507.02950)

    本研究揭示了四种LLM评估器在日语AI咨询评估中，高逐次评分信度并不等同于与专家判断一致，所有系统均倾向于给出高于专家的分数。

    

    大型语言模型（LLMs）越来越多地用于评估生成的对话，但可重复的分数不一定与专业判断一致。这项观察性固定基准研究比较了四种配置的LLM评估系统（GPT-5.5、Gemini 3.5 Flash、Claude Opus 4.8和Fable 5）与15位咨询专家对18次完整的日语模拟AI对AI咨询会话的汇总评分。这些会话代表了三种咨询师条件，覆盖六个预设客户画像。每个系统对每份转录文本在四个基于动机性访谈的维度和整体质量上进行了三次评分。所有四个系统在软化维持谈话和整体质量方面给出的分数均高于专家小组，尽管差异因系统和构念而异。单次运行的组内相关系数范围为0.33到0.96，表明高逐次运行信度并不确保更接近专家评分。

    arXiv:2507.02950v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly evaluate generated dialogue, but repeatable scores do not necessarily align with professional judgment. This observational fixed-benchmark study compared four configured LLM evaluator systems (GPT-5.5, Gemini 3.5 Flash, Claude Opus 4.8, and Fable 5) with aggregated ratings from 15 counseling experts on 18 complete simulated AI-to-AI counseling sessions conducted in Japanese. The sessions represented three counselor conditions across six prespecified client profiles. Each system scored every transcript three times on four motivational interviewing-informed dimensions and overall quality. All four systems assigned higher scores than the expert panel for softening sustain talk and overall quality, although differences varied across systems and constructs. Single-run intraclass correlation coefficients ranged from .33 to .96, showing that high run-to-run reliability did not ensure closer ex
    
[^230]: 从精确性到稳健性：数学推理中基于规则与基于模型的验证器研究

    From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning

    [https://arxiv.org/abs/2505.22203](https://arxiv.org/abs/2505.22203)

    本研究发现基于规则的验证器在数学推理中因无法识别等价答案而产生假阴性，阻碍强化学习性能，而基于模型的验证器虽提升静态准确性但存在稳健性问题，强调验证器需从精确性转向稳健性。

    

    arXiv:2505.22203v3 公告类型：替换-交叉 摘要：可信赖的验证器对于带有可验证奖励的强化学习（RLVR）的成功至关重要，这是DeepSeek-R1等各类大型推理模型背后的核心方法。在数学推理等复杂领域中，先前工作广泛采用基于规则的验证器来训练强大的推理模型。然而，这些验证器的可靠性及其对强化学习训练过程的影响仍未被充分理解。在本研究中，我们以数学推理为案例，在静态评估和强化学习训练场景下对多种验证器进行了全面分析。我们表明，广泛使用的基于规则的验证器无法识别不同格式的等价答案，导致大量假阴性结果，随着策略模型变得更强，这些问题会日益阻碍强化学习性能。基于模型的验证器显著提高了静态准确性，但极易受到重...

    arXiv:2505.22203v3 Announce Type: replace-cross  Abstract: Trustworthy verifiers are essential for the success of reinforcement learning with verifiable reward (RLVR), which is the core methodology behind various large reasoning models such as DeepSeek-R1. In complex domains like mathematical reasoning, rule-based verifiers have been widely adopted in previous works to train strong reasoning models. However, the reliability of these verifiers and their impact on the RL training process remain poorly understood. In this work, we take mathematical reasoning as a case study and conduct a comprehensive analysis of various verifiers in both static evaluation and RL training scenarios. We show that widely used rule-based verifiers fail to recognize equivalent answers in different formats, leading to substantial false negatives that increasingly hinder RL performance as the policy model gets stronger. Model-based verifiers substantially improve static accuracy but are highly susceptible to re
    
[^231]: 法律规则归纳：从类似司法先例中发现可泛化的原则

    Legal Rule Induction: Towards Generalizable Principle Discovery from Analogous Judicial Precedents

    [https://arxiv.org/abs/2505.14104](https://arxiv.org/abs/2505.14104)

    本文首次形式化定义了法律规则归纳（LRI）任务，并提出了一种可复现的数据集构建流程，利用大语言模型从类似司法先例中自动提取可泛化的法律原则。

    

    arXiv:2505.14104v2 公告类型：替换 摘要：法律规则不仅包括成文法规，还包括源自先例的隐含裁判原则，这些原则包含裁量性规范、社会道德和政策。虽然计算法律研究在将既定规则应用于案件方面取得了进展，但从司法判决中归纳法律规则在不同司法管辖区仍研究不足。大语言模型（LLMs）的出现为自动化提取此类潜在原则提供了前所未有的潜力，但缺乏正式的任务定义和基准阻碍了进展。为解决这一差距，我们将法律规则归纳（LRI）形式化为从类似先例中推导简洁、可泛化的教义规则的任务，提炼其共享的前提条件、规范行为和法律后果。我们进一步提出了一种可复现的LRI数据集构建流程，并将其应用于中国法律作为代表性司法管辖区进行实例化。

    arXiv:2505.14104v2 Announce Type: replace  Abstract: Legal rules encompass not only codified statutes but also implicit adjudicatory principles derived from precedents that contain discretionary norms, social morality, and policy. While computational legal research has advanced in applying established rules to cases, inducing legal rules from judicial decisions remains understudied across jurisdictions. The advent of Large Language Models (LLMs) offers unprecedented potential to automate the extraction of such latent principles, yet progress is stymied by the absence of formal task definitions and benchmarks. To address this gap, we formalize Legal Rule Induction (LRI) as the task of deriving concise, generalizable doctrinal rules from analogous precedents, distilling their shared preconditions, normative behaviors, and legal consequences. We further propose a reproducible pipeline for LRI dataset construction and, instantiating it on Chinese law as a representative jurisdiction, intro
    
[^232]: 普通Transformer也能成为强大的图学习器

    Plain Transformers Can be Powerful Graph Learners

    [https://arxiv.org/abs/2504.12588](https://arxiv.org/abs/2504.12588)

    本文证明普通Transformer架构通过三项简单修改（如简化L2注意力）即可成为强大的图学习器，无需复杂架构改动。

    

    arXiv:2504.12588v4 公告类型：替换-交叉 摘要：Transformer凭借其简单而强大的缩放点积（SDP）注意力机制，在各种模态中取得了卓越的性能。研究人员曾尝试将Transformer迁移到图学习中，但大多数先进的图Transformer（GTs）已偏离普通Transformer，要么集成消息传递机制，要么引入复杂的注意力机制，在架构上存在重大差异。这些分歧阻碍了将其他领域中Transformer训练进展轻松应用于图学习。与以往的GTs不同，本研究证明普通Transformer架构可以成为强大的图学习器。为此，我们提出对普通Transformer架构进行三项简单、最小且易于实现的修改，以构建我们的强大普通图Transformer（PPGT）：（1）简化$L_2$注意力以衡量幅度。

    arXiv:2504.12588v4 Announce Type: replace-cross  Abstract: Transformers have attained outstanding performance across various modalities, owing to their simple but powerful scaled-dot-product (SDP) attention mechanisms. Researchers have attempted to migrate Transformers to graph learning, but most advanced Graph Transformers (GTs) have strayed far from plain Transformers, exhibiting major architectural differences either by integrating message-passing or incorporating sophisticated attention mechanisms. These divergences hinder the easy adoption of training advances for Transformers developed in other domains. Contrary to previous GTs, this work demonstrates that the plain Transformer architecture can be a powerful graph learner. To achieve this, we propose to incorporate three simple, minimal, and easy-to-implement modifications to the plain Transformer architecture to construct our Powerful Plain Graph Transformers (PPGT): (1) simplified $L_2$ attention for measuring the magnitude clo
    
[^233]: 释放大语言模型在稠密检索中的潜力：基于查询似然建模

    Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling

    [https://arxiv.org/abs/2504.05216](https://arxiv.org/abs/2504.05216)

    本文提出LLM-QL模型，通过辅助的查询似然最大化任务增强大语言模型的稠密检索能力，利用生成优势改进对比学习。

    

    稠密检索是信息检索（IR）中的关键任务，为后续的重新排序和增强生成等下游任务提供基础。近年来，大语言模型（LLMs）展现了令人印象深刻的语义理解能力，使其成为稠密检索研究者的关注焦点。尽管LLMs作为解码器风格的生成模型在语言生成方面表现出色，但由于缺乏对后续标记的关注，它们往往在建模全局信息方面有所不足。受经典基于词的语言建模方法在IR中的启发，特别是查询似然（QL）模型，我们旨在通过QL最大化来利用LLMs的生成优势。我们不采用QL估计来进行文档排序，而是提出一个辅助任务——QL最大化，以增强骨干网络，用于后续的检索器对比学习。我们介绍了我们的模型LLM-QL，它整合了...

    arXiv:2504.05216v4 Announce Type: replace-cross  Abstract: Dense retrieval is a crucial task in Information Retrieval (IR), serving as the basis for downstream tasks such as re-ranking and augmenting generation. Recently, large language models (LLMs) have demonstrated impressive semantic understanding capabilities, making them attractive to researchers focusing on dense retrieval. While LLMs, as decoder-style generative models, excel in language generation, they often fall short in modeling global information due to a lack of attention to subsequent tokens. Drawing inspiration from the classical word-based language modeling approach for IR, specifically the query likelihood (QL) model, we aim to leverage the generative strengths of LLMs through QL maximization. Rather than employing QL estimation for document ranking, we propose an auxiliary task of QL maximization to enhance the backbone for subsequent contrastive learning of the retriever. We introduce our model, LLM-QL, which incorp
    
[^234]: BS测量仪：通过ChatGPT的语言检测政治与劳动

    The BS-meter: Detecting Politics and Labour through ChatGPT's Language

    [https://arxiv.org/abs/2411.15129](https://arxiv.org/abs/2411.15129)

    本文通过实证对比科学文献与ChatGPT生成文本，提出了一种统计模型来检测“胡扯”语言，并验证其在政治言论和“狗屁工作”语境中的适用性，揭示了LLM语言的独特特征。

    

    我们能从研究ChatGPT和其他基于大语言模型（LLM）的聊天机器人使用语言的方式中学到什么？本文分析了ChatGPT生成语言的独特特征，并联系到自然语言处理先驱、维特根斯坦的学生Margaret Masterman提出的问题。鉴于人们频繁抱怨基于LLM的聊天机器人产生“胡扯”（bullshit），这里借用Frankfurt流行专著《论胡扯》中的含义，我们进行了一项实证研究，对比了1,000篇科学出版物与ChatGPT生成的典型文本的语言。随后，我们探讨了这些语言特征是否能在两个著名的社会功能障碍背景中被检测到：乔治·奥威尔对政治言论的批判，以及大卫·格雷伯对“狗屁工作”的描述。通过简单的假设检验方法，我们证明了一个统计上的胡扯模型能够可靠地将Frankfurt式胡扯联系起来。

    arXiv:2411.15129v3 Announce Type: replace-cross  Abstract: What can we learn about language from studying how it is used by ChatGPT and other large language model (LLM)-based chatbots? In this paper, we analyse the distinctive character of language generated by ChatGPT, in relation to questions raised by natural language processing pioneer, and student of Wittgenstein, Margaret Masterman. Following frequent complaints that LLM-based chatbots produce "bullshit," in the sense of Frankfurt's popular monograph On Bullshit, we conduct an empirical study to contrast the language of 1,000 scientific publications with typical text generated by ChatGPT. We then explore whether the same language features can be detected in two well-known contexts of social dysfunction: George Orwell's critique of political speech, and David Graeber's characterisation of bullshit jobs. Using simple hypothesis-testing methods, we demonstrate that a statistical model of bullshit can reliably relate the Frankfurtian
    

