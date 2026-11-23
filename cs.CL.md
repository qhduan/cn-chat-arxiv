# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MemOnce: A Scalable Memory Architecture for Large Language Models](https://arxiv.org/abs/2611.27973) | MemOnce提出了一种可扩展的大语言模型记忆架构，通过记忆编码器-解码器仅缓存少量最终层隐藏状态并动态解码回KV缓存，结合记忆稀疏注意力自适应检索相关记忆，解决了标准Transformer KV缓存显存线性增长与线性架构丢失细粒度细节之间的矛盾。 |
| [^2] | [TEA-Bench-BR: Can Coding Agents Turn a Branching Clinical Specification into Correct Code?](https://arxiv.org/abs/2611.27895) | 该论文提出了TEA-Bench-BR——一个基于巴西早期自闭症筛查流程、以葡萄牙语规范编写的多层级代码生成基准，用以严格评测编码智能体将分支化临床规范转化为正确代码的能力，结果揭示了大模型在复杂任务层级和输入验证方面的显著局限。 |
| [^3] | [DistDNA: Language Model Fingerprinting under Stochastic Decoding](https://arxiv.org/abs/2611.27831) | 提出 DistDNA 方法，通过从少量采样响应中估计响应分布，并将解码设置变化与检查点变化的影响区分开来，实现随机解码条件下的大语言模型可靠指纹识别。 |
| [^4] | [SLP-Bench: Evaluating and Improving Acoustic Claims of Reasoning Audio Language Models on Clinical Speech](https://arxiv.org/abs/2611.27810) | 提出SLP-Bench基准，将推理型音频语言模型思维链中的每一项声学论断与临床医生的感知评分逐条对照，评估其事实性、稳定性和一致性，结果显示十个开源模型的论断平衡准确率仅为0.52至0.58。 |
| [^5] | [Evaluating Prompting Strategies for Zero-Shot Multilingual Aspect-Based Sentiment Analysis with Large Language Models](https://arxiv.org/abs/2611.27807) | 该论文系统评估了七个开放权重大语言模型在12种语言和三个领域的零样本多语言目标-方面-情感检测任务中的四种提示策略，发现简单的整体式提示即可取得强大的零样本性能。 |
| [^6] | [Enhancing Attention with Higher-Order Attention](https://arxiv.org/abs/2611.27764) | 该论文提出三种新型注意力机制（条件互信息分解、高阶注意力、基于广义最大熵原理的集合关联注意力），用以增强自回归概率的信息含量，从而提升大语言模型推理与生成的能力。 |
| [^7] | [From Novel Combinations to Informative Absence: Structural permission, communicative opportunity, and form choice in language learning](https://arxiv.org/abs/2611.27762) | 本文提出一个生成理论，将话语生成分解为结构许可、交际机会与形式选择，统一解释了语言学习者如何推断新颖表达的语法许可性并利用持续缺席作为负面证据，在30个人工语法上达到平均AUC 0.945。 |
| [^8] | [RoSci: Exploring Robustness in Science VLMs](https://arxiv.org/abs/2611.27750) | 该论文提出科学模型鲁棒性的概念，并构建了RoSci-Bench诊断基准和采用“解析-验证-决策”流程的RoSci-Agent框架，以增强视觉语言模型在科学任务中应对信息错误与信息缺失的能力。 |
| [^9] | [AutoOntologyBench: A Task-Grounded Benchmark for Agentic Knowledge Engineering](https://arxiv.org/abs/2611.27734) | 该论文提出AutoOntologyBench，一个基于7个领域18个真实知识源、包含19,947个审核任务的任务驱动型基准，首次系统评估自主智能体构建的本体能否真正帮助解决实际任务并带来可衡量的收益。 |
| [^10] | [Beyond Math, Code, and Logic: Do Reasoning Models Transfer to Scientific Argumentation?](https://arxiv.org/abs/2611.27722) | 该研究首次系统评估了语言模型在科学论证任务上的表现，发现即使前沿推理模型在领域特定论证任务上仍存在根本性能力差距，并提出了一种显式搭建论证推理脚手架的训练框架，显著提升了模型的科学论述能力。 |
| [^11] | [Decision Models for Multilingual Emotion Classification: A Comparative Study Against LLMs and Encoders](https://arxiv.org/abs/2611.27721) | 该研究首次在五种语言的多语言情感分类任务上系统比较了决策模型、大语言模型与微调编码器，发现最佳零样本系统的性能仅比微调编码器低约1个宏F1点。 |
| [^12] | [Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild](https://arxiv.org/abs/2611.27673) | Hy-MT3 系列通过将误译率降低 43%、漏译增译降低 74%、指令遵循错误降低 40%，并把语言覆盖扩展至 102 种，其仅 3B 激活参数的 MoE 模型在非思考模式下超越了众多旗舰大模型并逼近最强闭源思考模型。 |
| [^13] | [Auditing Post-Training Unlearning in Diffusion Language Models using Sparse Autoencoders](https://arxiv.org/abs/2611.27652) | 该论文利用稀疏自编码器与去噪状态重放，首次对扩散语言模型（LLaDA-8B-Instruct）的训练后遗忘进行机制审计，探索通过识别与遗忘数据相关的特征并在各去噪阶段动态调整其抑制来改进选择性遗忘。 |
| [^14] | [Potential emergence in LLM agents: a statistical physics perspective](https://arxiv.org/abs/2611.27618) | 该论文从统计物理学视角提出将智能体状态作为中间变量，通过配分函数与全局势能来定量测量并预测大语言模型智能体在后训练过程中行为的组织与涌现。 |
| [^15] | [Tasklets: A Unit-Test Benchmark for Language Model Architectures](https://arxiv.org/abs/2611.27607) | Tasklets 是一个由小型合成任务组成的标准化基准，按能力类别（回忆、序列操作、状态跟踪）组织，用于低成本、可重复地评估语言模型序列混合层的基础计算能力。 |
| [^16] | [Alliance Rupture and Repair in Simulating Cognitive Behavioral Therapy Dialogues](https://arxiv.org/abs/2611.27544) | 该论文提出ALLY模拟框架，首次将治疗联盟的破裂、修复与治疗计划的动态调整引入认知行为治疗对话模拟，使模拟咨询师能够像真实咨询师一样在多轮对话中察觉并修复联盟破裂。 |
| [^17] | [Open CDLM: A Benchmark and Library for Continuous Diffusion Language Models](https://arxiv.org/abs/2611.27536) | 该论文提出了 Open CDLM，一个用于评估连续扩散语言模型及其采样器的基准测试和开源库，旨在解决现有评估方法中变分上界松弛导致比较混淆、以及采样多样性度量无法有效识别重复生成的问题。 |
| [^18] | [PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks](https://arxiv.org/abs/2611.27521) | 本文提出了PaMuk-Bench，一个包含超过10K条平行翻译提示（涵盖阿拉伯语、丹麦语、法语、印地语、意大利语和罗马尼亚语六种语言）的多语言多任务视觉-语言基准，用于系统评估视觉-语言模型在低资源语言上跨多种任务的表现。 |
| [^19] | [How Many Bytes Has a BPE Model Seen? Estimating Byte-Token Exposure in Pretraining from Corpus Statistics and Embedding Norms](https://arxiv.org/abs/2611.27506) | 本文通过语料库分词统计与嵌入范数分析两条独立路径，量化估计了BPE模型中256个单字节后备token在预训练中的暴露程度，发现真正的UTF-8后备字节仅占总token的0.15%至0.27%，且后备率在不同语言间相差达四个数量级。 |
| [^20] | [The Assistant Persona Differs Across Languages in Activation Space](https://arxiv.org/abs/2611.27504) | 该研究首次从机制层面揭示了大语言模型中助手人格方向在不同语言间差异显著（与英语轴余弦相似度仅0.378–0.911）且与语言资源水平相关，证明从英语提取的激活方向无法跨语言泛化。 |
| [^21] | [Typhoon OCR 2: From Thai Document Transcription to Multilingual, Multi-Task OCR](https://arxiv.org/abs/2611.27494) | Typhoon OCR 2 将前代专注泰文转录的模型扩展为单一的多语言多任务 OCR 系统，通过 GRPO 训练的紧凑 2B/0.8B 模型同时支持提示驱动的信息提取与泰文、英文、中文的整页结构化解析，并在泰文基准上大幅超越前代且保持英文性能。 |
| [^22] | [Alignment without Transfer: Tokenization Impacts Cross-lingual Representations during Multilingual Language Model Pretraining](https://arxiv.org/abs/2611.27491) | 本研究通过仅改变分词器预训练39个单语和双语模型，发现改进的词表分配能显著增强模型中早期层的跨语言表示对齐以及前馈神经元的跨语言知识共享，但揭示了表示对齐与跨语言迁移之间存在分离现象。 |
| [^23] | [Efficient Natural Language to Temporal Logic Translation for Real-Time Physical AI](https://arxiv.org/abs/2611.27489) | 该论文提出将自然语言到线性时序逻辑的翻译分解为回答一组预定义答案的问题，用轻量级System-One分类器替代大语言模型的文本生成，从而在计算和能源资源受限的机器人上实现高效实时的翻译。 |
| [^24] | [On the Challenges of Using LLMs in High-Risk AI Recruitment: Guiding Questions for Legal Conformity with the AI Act](https://arxiv.org/abs/2611.27462) | 本文提出一个工具包，将欧盟《人工智能法案》合规评估的核心法律义务转化为技术规范要求和指导性问题，以支持“合法设计”的AI辅助招聘系统开发，重点关注第三方GPAI模型集成到招聘关键环节这一最具挑战性的场景。 |
| [^25] | [Structurally Different, Functionally Persistent: Capability Regions Under Model Optimization](https://arxiv.org/abs/2611.27445) | 该研究通过对量化、剪枝和LoRA适配的分析发现，尽管模型优化会导致能力区域发生显著的参数重组，但优化前识别的编码能力区域在功能上依然持久——消融这些原始区域仍能消除编码性能，表明它们始终是有效的干预目标。 |
| [^26] | [Silent Arbitration: How Large Language Models Resolve Co-Authoritative Conflicting Instructions](https://arxiv.org/abs/2611.27444) | 研究发现大语言模型在面对同等权威的冲突人格指令时会进行“静默仲裁”——虽然在直接检查中72%会承认冲突，但在实际任务响应中仅有22%明确告知用户，其余情况下模型默默采用某一方的立场而不透明地解决冲突。 |
| [^27] | [No Audio, No Transcripts, No Problem: LLM-based ASR Adaptation Using Only Domain Documents](https://arxiv.org/abs/2611.27434) | 该研究首次提出仅使用领域文档（无需目标音频和转写文本）对基于LLM的语音识别系统进行领域适配，通过将文档直接作为书面文本或由LLM渲染为合成口语对话，其适配效果可媲美甚至接近使用真实转写文本的结果。 |
| [^28] | [ORBIT: Offset-based Activation Steering for Cross-lingual Multimodal Fact Verification](https://arxiv.org/abs/2611.27421) | 提出了推理时算法干预方法ORBIT，通过从平行声明-证据实例中估计偏移向量并修正声明与证据表示，在不需微调的情况下显著恢复小型视觉-语言模型在低资源语言上的多模态事实核查性能。 |
| [^29] | [HR-Merge: Model Merging with Headroom-Weighted Residual Refinement](https://arxiv.org/abs/2611.27377) | HR-Merge 是一种无需训练的通用模型合并增强框架，它根据各任务的验证准确率余量对其残差加权并重新合并，从而有效恢复现有合并方法在不同任务上不均匀损失的准确率。 |
| [^30] | [Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training](https://arxiv.org/abs/2611.27376) | 提出可组合软提示蒸馏（CSPD）方法，将每个RL专家独立蒸馏为软提示并通过拼接方式组合，无需重新整合模型即可实现持续后训练，性能媲美多教师蒸馏且不损害目标领域之外的能力。 |
| [^31] | [AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance](https://arxiv.org/abs/2611.27369) | AMARIS 提出了一种协方差驱动的免训练模型合并框架，通过在正则化扰动预算下选择高效用、低干扰的激活子空间，将专家模型能力以单一静态检查点的形式无训练地继承到通用大语言模型中。 |
| [^32] | [WaXup: Weight Averaging of Experts as Data Augmentation for Robust Introspection](https://arxiv.org/abs/2611.27353) | 本文提出WaXup方法，通过在权重空间中对专家模型进行线性插值（权重平均）来增强内省训练数据，从而显著提升大模型可训练内省方法的鲁棒性与泛化能力。 |
| [^33] | [Breaking the Long Horizon: Short-Horizon Process Supervision for Long-Form Generation in LLMs](https://arxiv.org/abs/2611.27348) | 该论文提出一种短时程过程监督框架，将长文本生成分解为片段级决策，并利用基于评分标准的局部奖励计算逐步优势、实现局部化策略更新，从而解决传统结果级奖励监督延迟且粗粒度、难以定位错误和分配信用的问题。 |
| [^34] | [Reasoning in Fewer Tokens: Concept-Normalized Multilingual Scratchpads and Reversible Model-Adaptive Reasoning Codes](https://arxiv.org/abs/2611.27326) | 该论文提出了一种无损的中间推理压缩框架，通过BPE学习由PUA词元构成的压缩词表并扩展LLM词表后微调模型，在保持可解释性和可审计性的同时，平均减少14.7%的生成词元，推理轨迹长度最多缩短27.1%。 |
| [^35] | [SAVIOR: Sample-efficient Adaptation of Vision-Language Models for OCR Representation](https://arxiv.org/abs/2611.27319) | 提出样本高效的数据整理方法SAVIOR和结构感知评估指标PaIRS，通过微调Qwen2.5-VL使视觉语言模型在金融OCR的关键失败场景（如竖排/旋转文本、徽标嵌入名称、退化扫描件和多栏布局）上实现0.93的词召回率。 |
| [^36] | [Do Language Models Know Which Words Steer Their Reasoning?](https://arxiv.org/abs/2611.27317) | 该论文发现了“幻影上下文影响”现象——语言模型的推理会被语义无关的词不成比例地引导，而模型自身却意识不到这种依赖关系（生成-判断矛盾），并提出“在线KV抑制”方法来缓解这一问题。 |
| [^37] | [Observed, Curated, Reused: Observation Bias in Clinical AI Evaluation](https://arxiv.org/abs/2611.27301) | 该研究揭示了临床AI评估中因医疗文档仅记录医生所见而产生的观察偏差，并通过追踪6,503个病例在HPO数据库六个历史版本中的引用记录首次量化了这一偏差，同时发现了证据同时进入基准与知识库所导致的自我指涉循环问题。 |
| [^38] | [Belief-State Management Meets Rubric Rewards for Agentic Forecasting](https://arxiv.org/abs/2611.27286) | 该论文提出一个智能体事件预测训练框架，通过主动信念状态管理（PBM）将冗长上下文压缩为紧凑信念状态，并引入五维不依赖结果的SEIPH评分标准提供过程质量奖励，从而解决了LLM智能体预测中上下文累积和奖励信号嘈杂的两大难题。 |
| [^39] | [Do Companies Deliver on Their IPO Claims?](https://arxiv.org/abs/2611.27273) | 该论文提出了Vision Follow-Through（VFT）基准，用于评估公司在IPO招股书中做出的前瞻性声明是否真正兑现，发现模型在从完整招股书中提取声明以及评估声明兑现情况方面仍面临显著挑战。 |
| [^40] | [LongCat Recall Sparse Attention: Co-Designing Key Organization and Retrieval for Long-Context Inference](https://arxiv.org/abs/2611.27233) | 提出LongCat召回稀疏注意力（LRSA），通过将细粒度代表与增量维护索引中的分页检索相结合，消除长上下文推理中检索整个历史记录的瓶颈。 |
| [^41] | [CoPyro: Feeding Copy Proposals to Parallel Drafters](https://arxiv.org/abs/2611.27217) | CoPyro 将 n-gram 检索复制得到的 token 取代掩码馈入并行起草器，使其在单次前向传播中学会保留或修改这些 token，从而在 Qwen3-4B 的八个基准上为所有测试的并行起草器显著提升推测解码的接受长度。 |
| [^42] | [Correct Precedents Induce Policy Violations in On-Device LLM Agents](https://arxiv.org/abs/2611.27216) | 该论文首次发现“正确先例泛化”这一新型安全故障——端侧LLM代理在正确放行一个解除限制的条目后，会将该决策泛化迁移到仍受限制的后续条目上，导致即使整个会话历史中没有任何错误步骤也可能以策略违规告终，该问题在来自九个开发者的13个开源模型中普遍存在，且单个正确先例即可将最严重模型的安全率削减约一半。 |
| [^43] | [Explicit Style Extraction for Personalized Long-Form Text Generation](https://arxiv.org/abs/2611.27211) | 该论文提出StylePrism风格分类体系（28个风格维度）与SP-Refine推理时方法，将风格抽取变成显式步骤，在LongLaMP三个长文本任务上以平均10.2%的提升显著超越检索增强基线。 |
| [^44] | [How Much of a Calculation-Error Rate Is Not Arithmetic: Auditing the FinReason Cup 2026 Task 3 Metric with a Deterministic Rule System](https://arxiv.org/abs/2611.27192) | 本文用确定性规则系统审计 FinReason Cup 2026 任务 3 的评测指标，证明其“计算错误率”主要由非算术错误构成：23 个上榜参赛者中有 22 个存在非算术错误下界为正，且前两名的计算错误中至少分别有 11/12 和 12/13 并非真正的求和错误。 |
| [^45] | [What Does Matching a Human Intervention Effect Establish? Benchmark Checks for LLM Simulations](https://arxiv.org/abs/2611.27186) | 该论文质疑LLM模拟匹配人类干预效应所能确立的有效性，发现大量“通过”实为无信息量的零参考通过，并提出零参考基准与计划精度检验来严格评估LLM模拟的真实响应能力。 |
| [^46] | [Co-Evolving Behavioral Agents and Detectors for Adaptive Information Operations](https://arxiv.org/abs/2611.27156) | 该论文提出IO-AI协同演化框架，通过让基于真实信息行动行为构建、具备互动与规避能力的模拟智能体与检测器轮流对抗训练，使检测器对自适应规避智能体的漏检率降低了65%。 |
| [^47] | [The Vulnerability Window: Decision Instability in Quantized Reward Models](https://arxiv.org/abs/2611.27155) | 量化在总体基准准确率几乎不变的情况下仍会翻转1.1-12.3%的奖励模型成对偏好决策，其翻转率可用高斯近似Φ(-|m|/σ)刻画，且脆弱性窗口占比随模型规模从25%降至7%，揭示了小型裁判尤其脆弱的隐患。 |
| [^48] | [Do Senior Agents Do Their Job? Measuring Role Integrity in How LLM Agents Use Subordinate Material](https://arxiv.org/abs/2611.27139) | 该论文提出“角色完整性”概念并构建RI-Bench行为基准，借助同行评审、上诉裁决和议会调查三种制度层级的结构与公开记录，衡量多智能体LLM系统中高级智能体是否真正履行了权衡下级报告的职责，而非仅仅进行简单汇总。 |
| [^49] | [Double counting in LLM evidence fusion: exact posteriors separate rule compliance from posterior gain](https://arxiv.org/abs/2611.27138) | 本文构建了分级依赖条件下的精确后验参照基准，揭示出大语言模型对“每个来源只计一次”指令的遵从性与实际后验精度增益相互独立——完全遵守规则的模型可能损失精度，而不遵守规则的模型反而可能获得增益。 |
| [^50] | [Cross-Platform Production Signatures of State-Linked Information Operations](https://arxiv.org/abs/2611.27137) | 该论文提出“生产特征”概念——仅凭时间戳和粗粒度行为类型刻画信息操作账号的发帖规律，并证明这些行为规律可跨平台迁移，在无任何Reddit标签的情况下以0.90–0.94的AUC识别出IRA账号，显著优于会因平台切换而大幅退化的内容检测方法。 |
| [^51] | [Reasoning in Two Dimensions: Quantifying Chain-of-Thought Faithfulness and Efficiency](https://arxiv.org/abs/2611.27100) | 该论文提出将思维链表示为语义推理图，并沿“语义支持对推理历史”与“通往答案对偏离路径”两个维度分解模型的预测依赖性，在同一框架下实现了对思维链非忠实性与低效性的全局及步骤级量化。 |
| [^52] | [Separating Automation Gains from Agent Contributions in AI-Assisted Model Validation](https://arxiv.org/abs/2611.27094) | 提出一种以无LLM可控工作流为对照臂的分阶段架构与归因协议，只在智能体发现基线流程无法报告的缺陷时才记功，从而在银行模型验证中将智能体的真实贡献与自动化本身带来的收益区分开来。 |
| [^53] | [PLoQ: A Prompt-Specific Low-Rank Basis for 4-bit Quantization of LLM Activations](https://arxiv.org/abs/2611.27092) | PLoQ提出了一种提示词自适应的低秩分解方法，在预填充阶段为每个提示词动态估计低秩基，将主导激活分量单独量化并在矩阵乘法中恢复，从而在无需修改权重的情况下实现大语言模型激活值的4比特量化。 |
| [^54] | [ORION: Exact Canonical-State Supervision for Systematic Generalization in Transformers](https://arxiv.org/abs/2611.27087) | ORION框架通过辅助头监督精确的规范执行状态来恢复Transformer的深度外推能力（66.7%），与解决长度外推的RoPE形成正交互补，二者结合在长度-深度联合外推中实现严格的加性增益。 |
| [^55] | [How Far Does Lossless Re-tokenization Move an RLVR Update?](https://arxiv.org/abs/2611.27083) | 在RLVR训练中，对采样文本进行无损重新分词（如拆分少量token）会使策略梯度更新产生显著位移——其位移距离可达更换整个批次的0.556倍以上，且在同一检查点内这些位移方向保持一致。 |
| [^56] | [When Do LLM-Elicited Priors Help? A Controlled Audit of Language Models as Priors for Small-Sample Bayesian Regression](https://arxiv.org/abs/2611.27075) | 该研究对LLM作为小样本贝叶斯回归先验的做法进行了受控审计，发现不同模型的先验误差高度相关（相关性达0.92–0.99）、其声称的不确定性不可靠，且相比最佳经典先验仅在四个基准中的一个上带来明显改进。 |
| [^57] | [NightmareDiscourseBench: A Benchmark Dataset for Fictional Direct Discourse Identification Pipelines](https://arxiv.org/abs/2611.27059) | 本文提出了一个新基准数据集NightmareDiscourseBench，基于叙事理论构建直接话语分类评估框架，弥补了现有基准数据理论不一致和仅关注言语形式的不足。 |
| [^58] | [Isolating the Language Shell: A Content-Controlled Study of Multilingual Reasoning](https://arxiv.org/abs/2611.27053) | 该研究通过保持推理内容不变、仅改变教师推理呈现的语言外壳，证明多语言推理蒸馏中外壳语言的选择会显著影响学生模型的下游准确率与输出语言行为，而预训练兼容性无法预测这些效应。 |
| [^59] | [From Letter-of-Credit Requirements to Executable Specifications: Evaluating Verification Accuracy and Completeness](https://arxiv.org/abs/2611.26988) | 该论文将信用证审核要求逐条转化为可执行的原子条件并结合确定性验证器实现可追溯审核，揭示了仅以最终决定准确率衡量审核的误导性，并在数据保密条件下通过仅用合成数据微调的本地9B解析器达到93%的原子要求检测率。 |
| [^60] | [Run One, Get Two: Near-Free Streaming Safety Checks for Local LLMs](https://arxiv.org/abs/2611.26975) | ToRA通过让第二个模型作为额外一行嵌入生成器的解码步骤并共享权重与KV缓存，以每个token不到4%的额外延迟实现了接近专用防护模型准确性的本地LLM流式安全检查。 |
| [^61] | [Right Numbers, Wrong Story: When LLMs Narrate Financial Changes Beyond Their Inputs](https://arxiv.org/abs/2611.26968) | 论文提出FrameCheck插件，在不增加模型调用的情况下检测并约束LLM财务报告草稿中的“无依据框架叙述”，确保报告叙述忠实于输入事实。 |
| [^62] | [Beyond Prompt Heuristics: Assessing Child Safety Alignment in LLMs](https://arxiv.org/abs/2611.26965) | 该研究揭示了当前大语言模型的儿童安全对齐严重依赖伤害类别提示线索——在缺乏此类线索的看似无害请求上未对齐率大幅上升（最高达80个百分点），而仅引入一个伤害相关词即可恢复安全行为，表明模型的儿童安全推理主要建立在词汇和语义关联之上而非真正的安全理解。 |
| [^63] | [Self-similar Organization of Information in Language](https://arxiv.org/abs/2611.26935) | 本文提出了一种基于信息度量标度行为的预测信息组织理论，并通过在长文本中观察到不同长度尺度上信息度量的标度坍缩，证明了语言中预测信息具有自相似结构，这对理解语言的统计规律及大语言模型如何学习语言结构具有重要意义。 |
| [^64] | [Partially canonical axes in large language model representations](https://arxiv.org/abs/2611.26920) | 该研究挑战了“大语言模型表征坐标轴是任意的”这一常见假设，发现在12个LLM中原生轴比随机旋转轴更能与人类大脑语言响应和可解释语言特征对齐，揭示LLM表征轴是“部分规范的”——既非完全任意也非完全规范。 |
| [^65] | [ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval](https://arxiv.org/abs/2611.26919) | ERA提出了一种冷启动跨模型潜在检索系统，通过余弦最远点采样选取少量锚点文档、利用结合正交正则化的流形对齐损失训练轻量级适配器，并借助确定性路由器升级处理困难查询，从而无需对整个语料库重新嵌入即可将检索系统迁移到新的嵌入模型。 |
| [^66] | [What Does the Visual Encoder Buy? Visual Information Gain in Encoder-Free Multimodal Pretraining](https://arxiv.org/abs/2611.26917) | 提出视觉信息增益（VIG）这一精确度量，将无编码器多模态模型从图像中获取的信息分解为全局信息与布局信息，发现图像可降低3%至46%的预测损失，全局信息在训练早期即出现而布局信息随训练缓慢增长。 |
| [^67] | [AstronautBench: Measuring Agent Self-Improvement Across Episodes Through Knowledge, Evidence, and Progress](https://arxiv.org/abs/2611.26910) | AstronautBench基于游戏《星际拓荒》构建了一个新基准，通过知识、证据支持与任务进展三个互补视角，衡量权重固定的LLM智能体在世界不断重置的循环中凭借自身经验实现跨回合自我改进的能力。 |
| [^68] | [What Does a Zoom Actually Add? Decomposing Fine-Grained Perception Failures in Vision-Language Models](https://arxiv.org/abs/2611.26904) | 该研究将缩放操作分解为多个独立因素，发现大多数VLM被缩放拯救的细粒度感知失败并非因为缺少像素信息，而主要源于对已看像素的重新分词等处理环节。 |
| [^69] | [Perceive Once, Escalate on Demand: Handing Off Visual KV Memory from Small to Large VLMs](https://arxiv.org/abs/2611.26902) | 提出Baton方法，通过从大型模型自身答案中无标签蒸馏逐层线性映射，将小型VLM的视觉KV缓存记忆传递给大型VLM，避免升级时重复编码历史信息，使大型模型只需编码当前帧。 |
| [^70] | [Modeling Attributional Style at Scale: A Dataset and Analysis for Psychological Attribution Assessment and Reframing](https://arxiv.org/abs/2611.26900) | 该论文提出了首个大规模归因风格基准数据集ASTD及配套评估指标，包含42,000个真实事件和七种心理归因风格的归因标注，实现了归因风格的可扩展评估与认知重构。 |
| [^71] | [ProbeAgent: Principled Diagnosis of Agent Failures in Verifiable Environments](https://arxiv.org/abs/2611.26897) | ProbeAgent是一个自动化诊断框架，通过构建经过验证的可执行测试、生成保留疑似失败原因的变体任务并观察错误是否复现，来系统性发现、验证并归类智能体的共有失败模式。 |
| [^72] | [Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric](https://arxiv.org/abs/2611.26881) | 研究表明，多模态视频问答评估流水线中的LLM评审器会从其少样本示例中单向继承标签偏置：当示例偏向"Yes"时其接受率骤升约20–30个百分点并伴随单侧错误，而当示例平衡或偏向"No"时则接近人类共识，证明这种上下文标签偏置是方向性的而非对称的。 |
| [^73] | [How Long Will My Language Model Run? A Benchmark for Pre-Execution Runtime Prediction](https://arxiv.org/abs/2611.26875) | 本文提出PreRun基准，用于在执行前预测深度学习训练与推理的运行时间，包含数千条GPU集群实测记录，实验发现大语言模型对运行延迟的估计严重偏低（开源模型低估90-99%）。 |
| [^74] | [How Language Models Compress and Compare: Understanding Selection with Token Covariance Maps](https://arxiv.org/abs/2611.26874) | 该论文提出词元协方差图这一提示级几何分析工具，揭示了仅解码器Transformer在结构化选择任务中普遍遵循“先压缩后比较”的层级机制——早期层弥散表示提示、中间层压缩为共享低秩瓶颈、后期层重构上下文与选项间的比较几何。 |
| [^75] | [LLMs Memorize Their Own Hallucinations During Post-Training](https://arxiv.org/abs/2611.26863) | 该论文发现，采用以推理为中心的后训练方案（如基于结果的强化学习奖励和“认知”行为注入）会导致大语言模型在知识任务训练中记住自己在长推理链中产生的幻觉，且这一现象在多个开源模型上得到验证。 |
| [^76] | [Weight-Space Programming Writes In-Context Learning and Matches Fine-Tuning in Language Models](https://arxiv.org/abs/2611.26841) | 前沿语言模型可通过权重空间编程将ICL任务向量直接写入较弱模型的稀疏MLP单元（与提示格式相关联），使其零样本性能达到ICL和微调（含LoRA）的准确率，且几乎不影响困惑度。 |
| [^77] | [Cross-Linguistic Distinctive-Feature Constraints on Segmental Triphones: Evidence for Contrastive Dispersion](https://arxiv.org/abs/2611.26837) | 该研究通过分析17种语言（涵盖九个语系）中音段三元组的区别特征距离，发现自然语音呈现中等特征距离模式，并相比随机基线展现出更窄的离散度和更低的联合熵，为跨语言的对比离散度约束提供了实证证据。 |
| [^78] | [Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models](https://arxiv.org/abs/2611.26835) | 本文提出一种非扩展门控连分数架构组件，可替换语言模型中的各类前馈层（包括MLP、GLU及MoE专家），并用对数和指数重新实现除法运算，从而提升性能并简化与标准预训练流程的集成。 |
| [^79] | [Rethinking Full-Weight LoRA with Domain and Task Adapter Scaling for Question Answering over Portuguese Well Drilling Reports](https://arxiv.org/abs/2611.26833) | 该研究提出在训练后对领域适配与任务特定的LoRA适配器进行缩放、并可用互补权重将二者组合，从而在不损害指令遵循能力的前提下提升大语言模型对葡萄牙语钻井报告问答任务的性能。 |
| [^80] | [Neural Geometry as a Learning Signal](https://arxiv.org/abs/2611.26832) | 本文证明只需监督模型内部表示的几何结构（如傅里叶谐波和低维螺旋结构），就能在完全不监督输出的情况下取得与标准微调相当的模型改进效果。 |
| [^81] | [Search From the Start: Search-Native Pretraining for Language Models](https://arxiv.org/abs/2611.26809) | 提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。 |
| [^82] | [MedicalRSI: Medical Recursive Self-Improving Agents via Clinical Model-Harness Co-Evolution](https://arxiv.org/abs/2611.26806) | 该论文提出MedicalRSI框架，通过临床模型-框架协同进化使医疗智能体能够同时改进自身及其改进策略，从而实现递归自我改进，避免同类错误在不同患者间反复出现。 |
| [^83] | [Don't Discard Your Rollouts: Reusing Teacher RL Traces for Student Distillation](https://arxiv.org/abs/2611.26803) | 提出利用GRPO训练中已记录的元数据（提示通过率和教师对数似然）将推理轨迹重用建模为数据选择问题，通过SFT预热加DPO的分阶段蒸馏策略，把教师的强化学习轨迹高效重用于学生模型训练，避免了昂贵的重新生成数据。 |
| [^84] | [What Helps Language-Model Agents Review Violent Video? A Controlled Comparison with a Human Evidence Reference](https://arxiv.org/abs/2611.26776) | 该研究在XD-Violence数据集上进行受控比较，发现将感知模型观察结果整理为一次性阅读的扁平文本是语言模型代理审阅暴力视频最有效的方式，其表现与直接处理原始视频的单一视听模型相当，而消息传递和证据图接口效果较差。 |
| [^85] | [Know Your Context Bank: Repairing Inconsistency and Measuring Generalization in Context-Aware Text-to-SQL](https://arxiv.org/abs/2611.26775) | 该论文提出“上下文库腐化”概念，指出企业Text-to-SQL上下文库中各条目虽分别通过审核却存在相互矛盾的不一致性问题，并设计了从人工校准标注蒸馏而来的快速评分器，通过逐决策比较查询来修复不一致性，同时度量系统是基于上下文真正泛化还是仅机械复现。 |
| [^86] | [Reinforcement Learning Unlearns Corrupted SFT Reasoning](https://arxiv.org/abs/2611.26756) | 强化学习后训练（GRPO）能够恢复被严重损坏的SFT数据造成的性能下降，仅用8条错误的“仅形式”轨迹初始化RL甚至可以超越黄金SFT初始化的表现，且SFT准确率与RL后准确率基本不相关。 |
| [^87] | [SkillPrivus: Uncovering PRIVacy of USe Harms in Agentic Skills](https://arxiv.org/abs/2611.26747) | SkillPrivus 是一个无需执行即可在安装前对智能体技能进行静态隐私审计的框架，对 726 个真实技能的分析发现超过半数（54.5%）存在潜在隐私违规。 |
| [^88] | [TOUMAII: A 4.5M-Pair Tool-Calling Preference Dataset and Its Source-Agnostic Generator](https://arxiv.org/abs/2611.26722) | 该论文提出了包含约450万条经验证偏好对的大规模工具调用偏好数据集TOUMAII及其源无关的偏好对生成器，以缓解工具调用领域（尤其是多轮场景）偏好数据的稀缺，从而支持DPO训练出更可靠的工具调用能力。 |
| [^89] | [Adding One More Agent: How AI Team Size and Formation Shape Productivity and Diversity in Human-AI Brainstorming Teams](https://arxiv.org/abs/2611.26663) | 在人机头脑风暴实验中，双AI团队虽提高了创意总量却大幅压缩了人类贡献，而单个“平均人类行为校准”的AI反而最能激发人类创意产出，表明AI伙伴数量与组建方式共同重塑了团队的生产力与多样性。 |
| [^90] | [VerifiQuant: Governance-by-Execution for Financial LLM Workflows](https://arxiv.org/abs/2611.26649) | VerifiQuant提出“执行即治理”框架，通过将金融逻辑封装为推理契约并用可执行诊断漏斗在计算前评估每个查询，使金融大模型能够结构化、可审计地区分计算、澄清与拒绝，从而在输入不支撑数值答案时实现更安全的行为。 |
| [^91] | [SDG: State-Dependent Gating at Chunk Boundaries for Linear Attention](https://arxiv.org/abs/2611.26637) | 提出SDG方法，通过在分块边界处依据记忆状态的变化计算逐元素门控，为线性注意力模型提供状态反馈的遗忘机制，同时保持块内并行计算的高效性。 |
| [^92] | [Represented but misread: Post-training changes how language models read out what they represent](https://arxiv.org/abs/2611.26584) | 研究发现后训练并未破坏语言模型内部的表征（线性探针仍能准确区分人类与AI文本），而是改变了模型将表征读取为答案的方式，导致回答灵敏度下降和对先验信息的错误利用。 |
| [^93] | [AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher](https://arxiv.org/abs/2611.26510) | 该论文提出了Skill-FT技能引导微调方法和AutoMaster自主后训练系统，由LLM智能体自动分析教师轨迹并设计自然语言技能干预来抑制不良习惯、激发有效行为，从而在多个任务上比朴素SFT提升1.42-8.99个百分点的性能。 |
| [^94] | [The Linguistic Structure of Dependency-Conditioned Embeddings](https://arxiv.org/abs/2611.25909) | 该论文通过分析全局与局部几何结构，系统考察了依存条件化词嵌入的语言组织方式，发现全局几何主要由上下文词性塑造，而局部词汇相似性结构在依存关系间得到部分保留。 |
| [^95] | [STARK: Moving towards Personalized Agents That Truly Learn and Remember](https://arxiv.org/abs/2611.25705) | STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。 |
| [^96] | [Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation](https://arxiv.org/abs/2611.25086) | 该研究首次揭示了攻击者可以将浏览器预装的本地语言模型（如Chrome中的Gemini Nano）滥用于后渗透攻击阶段，仅借助操作系统自带工具即可驱动其窃取凭据、挖掘敏感信息，且全程躲避网络监控检测。 |
| [^97] | [Safe Personal RSI: Evaluating and Repairing Personal-Agent Updates](https://arxiv.org/abs/2611.23651) | 本文提出Safe Personal RSI更新评估机制，通过将递归自我改进限制在提示词层面、固定模型权重与代码，并建立“测试—修复—验证”流程，在29个案例的基准上成功拒绝了21次高风险更新尝试中的20次，保障个人智能体更新后的安全性与任务性能。 |
| [^98] | [Grounding Language Models in Speech Does Not Fix "AI Flavor": Calibrated LLM Judges, Goodhart Traps, and a Preference-Tuning Fix](https://arxiv.org/abs/2611.19840) | 该论文通过在0.6B模型上的受控实验证明，注入语音或韵律信号的三种方法（音素交错续训、韵律计划条件生成、基于TTS韵律距离的规则奖励）均无法消除LLM中文输出的“AI味”，并指出当前n≈20的单评分员盲评只能检测极端偏好的评估方法是根本症结，同时提出一种基于偏好调优的修复方案。 |
| [^99] | [Formalizing Formal Semantics: The Formal Atlas](https://arxiv.org/abs/2611.19509) | 本文提出了基于证明助手的“形式图谱”系统，通过交互式应用将语义学文献与形式化证明连接起来，实现语义理论的形式化、预测检验与比较，并探索AI辅助开发与推理的应用前景。 |
| [^100] | [MarineSparkLM-4B: A Lightweight Language Model for Marine and Offshore Engineering Built by Parameter-Efficient Adaptation](https://arxiv.org/abs/2611.19151) | 该研究通过在一台工作站级设备上仅对40亿参数模型进行0.81%参数量的LoRA高效适配，并以约5小时的低算力成本，构建出面向海洋与离岸工程领域的轻量级专用语言模型MarineSparkLM-4B。 |
| [^101] | [When evaluation becomes intervention in clinical AI](https://arxiv.org/abs/2611.18297) | 临床AI中的AI评判者在被用于选择提示词或训练模型时会本身成为干预的一部分，并可能偏向其忽视的错误，因此评判者报告必须包含特定内容，且每项由评判者驱动的变更都需要独立的临床测试。 |
| [^102] | [Point-in-Time Encoders for Earnings Calls](https://arxiv.org/abs/2611.14325) | 该论文提出按年度截止日期训练的“时点”文本编码器，以消除财报电话会议检索回测中的前视偏差，其小型专用模型在答案检索上全面优于通用检索器，并将收益预测的秩相关性提升约一半。 |
| [^103] | [Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models](https://arxiv.org/abs/2611.13223) | 本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。 |
| [^104] | [TS-GEAR: Taobao Shangou GEnerative Ad Retrieval](https://arxiv.org/abs/2611.12180) | 该论文提出TS-GEAR，面向即时配送场景的生成式广告检索，通过显式建模配送范围可达性、基于地理位置与当前查询条件化的用户兴趣表示以及集合级训练信号，解决了语义ID地理信号软化、兴趣估计偏差与候选冗余三大挑战。 |
| [^105] | [Tracing the Scientific Discovery Capacity of LLMs to Their Training Data](https://arxiv.org/abs/2611.03617) | 该研究提出“单次事实”理论，通过直接操纵训练数据，证明训练集中仅出现一次的事实比率是决定大语言模型科学发现能力的关键因素，且这一能力可在不同模型、领域间迁移。 |
| [^106] | [Beyond Aggregate WER: In-domain Adaptation, Learner Speech, Code-Switching, and Speaker Attribution in Online Spanish Learning Classrooms](https://arxiv.org/abs/2610.38016) | 该研究超越整体WER的评估方式，按说话人角色和语言语境细分错误来评估在线西班牙语课堂中的多种ASR系统，并通过LoRA领域内适配发现MOSS-Transcribe-Diarize在各维度上均带来一致改进。 |
| [^107] | [The Voice on the Line](https://arxiv.org/abs/2610.35272) | 该论文训练了20毫秒延迟、能从通话本身注册来电者的因果语音隔离器（掩码transformer），并发现将WavLM-Large权重因果化并与模型联合训练能显著提升效果，将开发集错误率从11.44%降至10.55%。 |
| [^108] | [Style over Substance? Linguistic Bias in LLM-based Peer Review Copy title](https://arxiv.org/abs/2610.35254) | 该论文提出了一个受控实验框架，通过基于语言学文献推导的细粒度改写指令，系统性地改变论文的语言风格（母语/非母语特征与高/低置信度特征），以研究基于大语言模型的同行评审是否会受到与科学内容无关的语言偏见的影响。 |
| [^109] | [Entity Matching using Decision Models](https://arxiv.org/abs/2610.34626) | 本文提出用决策模型作为实体匹配的新替代方案，证明其在零样本设置下无需任务特定训练数据即可取得接近生成式大语言模型的匹配性能。 |
| [^110] | [CulturalRiddles: A Multicultural Benchmark of Riddles](https://arxiv.org/abs/2610.29968) | CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。 |
| [^111] | [Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories](https://arxiv.org/abs/2610.29378) | 提出OnQ流水线，通过在策略蒸馏利用学生模型自生成的轨迹进行训练，解决了固定响应监督导致的训练与生成前缀不匹配及高熵处分布失配问题，从而恢复低比特量化大模型的推理能力。 |
| [^112] | [MemT-1.0: Towards Memory-Native Evolution](https://arxiv.org/abs/2610.26825) | MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。 |
| [^113] | [Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents](https://arxiv.org/abs/2610.25760) | 该论文通过区分“行动前即时生成推理”与“在提示中保留历史推理笔记”两种角色，发现真正减少状态违规行动的主要来源是行动时刻的即时推理（计算），而非记忆式的历史推理保留。 |
| [^114] | [CHSM8: A Chess Transformer that Follows Natural-Language Style Prompts](https://arxiv.org/abs/2610.25356) | CHSM8是一个2.1亿参数的国际象棋Transformer，无需搜索即可根据自然语言风格提示下棋，达到1837 Elo并击败了包括参数量为其13倍模型在内的所有同类公开模型，并通过合成数据微调使其能够理解文本风格指令。 |
| [^115] | [Change the Number, Change the Answer: Assessing Rating Scale Sensitivity in LLM-Based Text Scoring](https://arxiv.org/abs/2610.22187) | 本文定义并系统研究了“量表敏感性”这一新概念，通过大规模交叉实验发现，评分量表的设计元素（格式、范围、符号）平均会使大语言模型的文本评分产生约0.10的归一化分数偏移，揭示了不同研究间分数可比性的重要隐患。 |
| [^116] | [SkillForgeBench: Evaluating Agent Reliability Grounded in Agent Skill Specifications](https://arxiv.org/abs/2610.14918) | SkillForgeBench以智能体技能作为统一执行蓝图，通过资源准备、技能驱动任务合成等三个协同设计阶段，构建了兼顾领域多样性与程序复杂性的智能体可靠性评估基准，化解了专家策划基准与众包基准之间的根本性权衡。 |
| [^117] | [A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic](https://arxiv.org/abs/2610.07990) | 本论文发现主流模型合并方法中系数搜索带来的隐式正则化实际上限制了性能，去除该正则化、直接优化合并模型乃至预训练模型的权重，即可在多种架构和极低数据量场景下显著提升合并后的多任务性能。 |
| [^118] | [Frozen Scenes, Shifting Winners: Configuration Fragility in Text-to-3D Evaluation](https://arxiv.org/abs/2610.00447) | 该论文通过对来自六个生成器的300个冻结场景系统性变化八种渲染与描述文本配置因素，揭示文本到3D评估结果对配置高度脆弱——17/19个对齐评估器的配置方差超过生成器间方差，且18/19个评估器在某些配置下会改变点估计赢家，表明当前评估协议存在严重的不稳定性。 |
| [^119] | [MatLoom: Layered Text-to-Material Generation in a Compact Program Space](https://arxiv.org/abs/2609.40322) | MatLoom提出一种紧凑的面向图层的材质程序语言，配合预训练语言模型，通过解析器引导修复与预览评审实现免微调的文本到材质生成，其最佳配置在141个提示词的基准上超越了三个扩散基线。 |
| [^120] | [MemLife: Curating and Reasoning over Long-Term Egocentric Video Memories](https://arxiv.org/abs/2609.40195) | MemLife是一种多模态记忆系统，通过构建基于实体的第一人称文本情节并用时间索引的智能体读取器进行检索，在无需训练和查询时视频访问的情况下，在四个长时程基准上比最强免训练基线提升4.6%–12.0%，解决了长期第一人称视频中记忆证据丢失和检索竞争的问题。 |
| [^121] | [Which papyrus HTR is good enough? Character-error-rate tolerance of four papyrological tasks on Greek texts](https://arxiv.org/abs/2609.37755) | 该研究通过将模拟的不同字符错误率（1%—50%）的HTR输出用于文献类型识别、断代等四种纸草学任务，首次为古希腊纸草文献手写文本识别系统所需的精度建立了基准。 |
| [^122] | [Cross-sector generalization of accident-process role classification in occupational accident narratives](https://arxiv.org/abs/2609.22081) | 该研究构建了法语职业事故叙述的专家标注语料库，将事实单元划分为工作情境、不利条件、事故事件和后果四种角色，并系统评估了角色分类模型跨行业、跨组织领域的泛化能力。 |
| [^123] | [AhaBench: Do Agents Learn from Prior Experience? A Benchmark for Long-Horizon Continual Learning](https://arxiv.org/abs/2609.05435) | AhaBench 是一个长时程持续学习基准，通过谜题探索、带精确验证器的数学任务和自动售货机模拟三个组件，评估固定模型在获得先验经验后，在相关但支持已被移除、改变或延迟的条件下行为是否真正得到改善。 |
| [^124] | [From Exposure to Expectation: Frequency, Surprisal, and Language Across Development in Spanish](https://arxiv.org/abs/2608.22452) | 这项研究表明，在西班牙语中，词汇频率是预测儿童习得年龄的主要因素，而意外性（基于上下文的可预测性）在频率和词长之外几乎没有额外贡献，尽管意外性能预测成人阅读时间。 |
| [^125] | [Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus](https://arxiv.org/abs/2608.12149) | 本文首次系统研究了混合线性注意力大语言模型中的大规模激活现象，发现了注意力前尖峰和尖峰间平台两种新形态，并揭示了它们与架构配置的关系。 |
| [^126] | [Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling](https://arxiv.org/abs/2608.11829) | 本研究发现在线策略蒸馏主要提升采样效率而非扩展推理能力边界，其优势在小采样预算下显著，但在大预算下会减弱。 |
| [^127] | [BODHI: Do LLMs Branch Out and Discover Heterogeneous Inferences?](https://arxiv.org/abs/2608.02867) | 研究发现RLVR训练的大语言模型相比蒸馏模型表现出更低的语义多样性和更少的推理分支，表明其推理性能提升主要来自采样效率的改善而非推理能力的真正扩展。 |
| [^128] | [Probing the Prompt KV Cache: Where It Becomes Dispensable](https://arxiv.org/abs/2605.30574) | 本研究通过受控干预实验揭示提示词KV缓存的冗余在于形式而非内容——在深层用空内容或填充token的脚手架KV缓存替换提示词缓存即可几乎无损恢复精度，并在Qwen3、Gemma 3、Llama 3等模型上得到验证，实现中KV内存减少15.4%且多支持18%的并发请求。 |
| [^129] | [BioELX: Context-Aware Cross-lingual Biomedical Entity Linking without Task-Specific Supervision](https://arxiv.org/abs/2605.27380) | BioELX提出了一种检索-重排序框架，利用Wikidata衍生的跨语言别名监督进行检索，并通过提及锚定提示将LLM重排序器适配于实体链接，实现了无需任务特定监督的上下文感知跨语言生物医学实体链接。 |
| [^130] | [A categorical error sensitivity index (ISEC): A preventive ordinal decision-support measure for irrecoverable errors in manual data entry systems](https://arxiv.org/abs/2605.12328) | 本文提出了类别错误敏感性指数（ISEC）——一种序数复合评分，通过量化类别对之间的结构易混淆性对类别对进行排序，为中小企业人工数据录入系统提供预防性决策支持，以避免不可恢复的类别错误传播至关键绩效指标而误导管理决策。 |
| [^131] | [When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds](https://arxiv.org/abs/2605.06615) | 本文通过引入ℓ₁-范数平稳性、ℓ∞-平滑性和可分离噪声模型这一新问题几何框架，首次从理论上证明了SignSGD在何种条件下以及为何能够超越SGD，并给出了匹配的复杂度上下界。 |
| [^132] | [Compute Optimal Tokenization](https://arxiv.org/abs/2605.01188) | 该论文通过训练988个不同压缩率的潜在词元化（BLT）模型发现，在计算最优配置下模型参数量应与以字节（而非词元）衡量的数据量成正比地缩放，且最优压缩率不同于常用BPE分词器所得的结果。 |
| [^133] | [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734) | 本文提出同策略自蒸馏（OPSD）学习算法，让单个LLM在不同上下文中同时充当教师与学生——教师以特权信息（如经过验证的推理轨迹）为条件对学生进行密集监督，从而无需独立的教师模型，并显式利用了推理数据集中的标准答案。 |
| [^134] | [Verify Before You Generate: Sketch-Guided Efficient Motion-Planning for Video Generation](https://arxiv.org/abs/2511.17450) | 提出了 SketchVerify，一个无需训练的框架，在调用昂贵的视频生成器之前先通过轻量级视频草图验证运动规划，从而高效比较并迭代精化多条候选轨迹，以低成本获得高质量运动。 |
| [^135] | [Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph](https://arxiv.org/abs/2511.00086) | 该论文提出将测试时扩展中在固定预算下搜索计算最优模型组合与架构的新问题形式化为多LLM协作图，并将其重新表述为概率图优化问题，以应对庞大的组合搜索空间和任务定制化需求。 |
| [^136] | [Supervised Fine-tuning for Geometry-aware Instruction-following](https://arxiv.org/abs/2510.26183) | 本文提出一种在监督微调中遮蔽错误生成内容至分类控制标记的对比输入编码方法，在保持生成准确率的同时提高了生成结果落入几何感知的校准SDM估计器高概率区域的接受率，且无需对语言模型参数进行专门微调。 |
| [^137] | [Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned LLMs](https://arxiv.org/abs/2510.11288) | 该研究首次发现“涌现式失准”现象同样会通过上下文学习产生——仅用少量（最少2个）狭窄领域的上下文示例，就能使四个主流大语言模型家族对无关的良性问题产生广泛失准响应，且更大模型反而更易受影响，其机制可解释为安全目标与上下文遵循行为之间的冲突。 |
| [^138] | [RAG Makes Guardrails Brittle? Investigating Robustness of Guardrails under RAG-style Contexts](https://arxiv.org/abs/2510.05310) | 本研究通过系统评估发现，RAG场景中仅插入良性的检索文档就能使基于LLM的安全防护栏在约10%的情况下翻转判断，且多数翻转会将原本正确的安全决策反转为错误决策，揭示了现有防护栏在RAG式上下文下的脆弱性。 |
| [^139] | [Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards](https://arxiv.org/abs/2510.04392) | 本文提出PS-GRPO强化学习方法，通过跨复述集的组相似度奖励来提升RAG系统在语义等价查询下的信息一致性，并提供了将一致性分解为检索器、生成器和端到端层面的系统性评估框架。 |
| [^140] | [Family Matters: Language Transfer and Merging for Adapting Small LLMs to Faroese](https://arxiv.org/abs/2510.00810) | 该论文的核心贡献是通过先在斯堪的纳维亚亲缘语言上继续预训练再适配法罗语，并构建两个全新的法罗语最小对探针评估基准，证明了语言家族内的迁移能有效提升低资源语言的语言建模与可接受性。 |
| [^141] | [CHAIR -- Classifier of Hallucination as Improver](https://arxiv.org/abs/2501.02518) | CHAIR通过提取每个词元在所有层中logits的紧凑统计特征来检测大语言模型的幻觉，在零样本场景下显著提升检测准确率，并展示了利用内部表示设计自适应解码策略以减少幻觉的潜力。 |
| [^142] | [OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs.](http://arxiv.org/abs/2309.03876) | OpinionGPT是一个指令调整大型语言模型(LLMs)的web演示，用户可以选择各种偏见并进行比较，它意在使模型的偏见显性和透明化。 |

# 详细

[^1]: MemOnce：一种面向大语言模型的可扩展记忆架构

    MemOnce: A Scalable Memory Architecture for Large Language Models

    [https://arxiv.org/abs/2611.27973](https://arxiv.org/abs/2611.27973)

    MemOnce提出了一种可扩展的大语言模型记忆架构，通过记忆编码器-解码器仅缓存少量最终层隐藏状态并动态解码回KV缓存，结合记忆稀疏注意力自适应检索相关记忆，解决了标准Transformer KV缓存显存线性增长与线性架构丢失细粒度细节之间的矛盾。

    

    arXiv:2611.27973v1 公告类型：新 摘要：在长时程和持续交互场景中部署大语言模型（LLM）需要为未来的推理维护可复用的记忆状态，这对现有架构提出了挑战。标准Transformer会逐层累积token级别的KV缓存，随着上下文扩展，导致GPU显存线性增长和计算复杂度呈二次方增长。相比之下，线性时间架构虽然将历史压缩为固定大小的状态，但难以保留细粒度细节。我们提出了MemOnce，这是一种为大语言模型学习可扩展记忆状态的基础架构。MemOnce基于两项核心设计：记忆编码器-解码器：它仅缓存最终层隐藏状态中的一小部分作为紧凑的记忆表示，并动态地将其解码回注意力层的KV缓存；记忆稀疏注意力：对于每个持续输入，MemOnce自适应地检索相关的记忆表示，并通过稀疏……（原文摘要在此处截断）

    arXiv:2611.27973v1 Announce Type: new  Abstract: Deploying large language models (LLMs) in long-horizon and continual interactions requires maintaining reusable memory states for future inference, posing challenges for existing architectures. Standard Transformers accumulate layer-wise, token-level KV caches, incurring linear GPU memory growth and quadratic computational complexity as the context expands. In contrast, linear-time architectures compress history into fixed-size states, but struggle to retain fine-grained details. We introduce MemOnce, a foundation architecture that learns scalable memory states for language models. MemOnce is based on two core designs: (a) Memory encoder-decoder: it caches only a small subset of final-layer hidden states as compact memory representations and dynamically decodes them back into KV caches for attention layers; (b) Memory sparse attention: for each continual input, MemOnce adaptively retrieves relevant memory representations, with the sparse
    
[^2]: TEA-Bench-BR：编码智能体能否将分支化的临床规范转化为正确的代码？

    TEA-Bench-BR: Can Coding Agents Turn a Branching Clinical Specification into Correct Code?

    [https://arxiv.org/abs/2611.27895](https://arxiv.org/abs/2611.27895)

    该论文提出了TEA-Bench-BR——一个基于巴西早期自闭症筛查流程、以葡萄牙语规范编写的多层级代码生成基准，用以严格评测编码智能体将分支化临床规范转化为正确代码的能力，结果揭示了大模型在复杂任务层级和输入验证方面的显著局限。

    

    编码智能体越来越多地被用于实现由领域专家编写的规则，而这类代码中的缺陷是“静默”的：代码可以运行并返回看似合理的答案。我们提出了TEA-Bench-BR，这是一个基于巴西早期自闭症（TEA）筛查路径构建的基准：包含三个结构仿照真实量表的虚构评估工具、一个多专业转诊流程，以及导出为采用国家RNDS配置文件的FHIR R4格式，全部规范均以葡萄牙语编写。一个生成器对每个任务的参数进行采样（共144个任务、6,798个隐藏测试用例），这些隐藏用例能够杀死参考实现的全部1,955个非等价变异体。以Gemma 4作为编码智能体，12B模型解决了38.0%的单量表任务，而在FHIR层面则无一解决；31B模型在第1至第3级的解决率分别为77.8%、85.2%和78.7%，并通过了超过99%的隐藏用例，其剩余的失败大多出现在输入验证而非临床决策方面。当相同的任务以英语呈现时，31B模型在第3级的解决率降至57.4%，因为其代码验证……（摘要在此处被截断）

    arXiv:2611.27895v1 Announce Type: cross  Abstract: Coding agents increasingly implement rules written by domain experts, and bugs in such code are silent: it runs and returns a plausible answer. We introduce TEA-Bench-BR, a benchmark built on the Brazilian pathway for early autism (TEA) screening: three fictitious instruments with the structure of real ones, a multi-professional referral flow, and export to FHIR R4 with the national RNDS profiles, all specified in Portuguese. A generator samples each task's parameters (144 tasks, 6,798 hidden cases), and the hidden cases kill all 1,955 non-equivalent mutants of the reference. With Gemma 4 as the agent, the 12B solves 38.0% of single-instrument tasks and none at the FHIR level; the 31B solves 77.8%, 85.2% and 78.7% at levels 1 to 3 and passes over 99% of the hidden cases, its remaining failures mostly in input validation rather than clinical decisions. With the same tasks in English, the 31B solves 57.4% at level 3, as its code validate
    
[^3]: DistDNA：随机解码下的语言模型指纹识别

    DistDNA: Language Model Fingerprinting under Stochastic Decoding

    [https://arxiv.org/abs/2611.27831](https://arxiv.org/abs/2611.27831)

    提出 DistDNA 方法，通过从少量采样响应中估计响应分布，并将解码设置变化与检查点变化的影响区分开来，实现随机解码条件下的大语言模型可靠指纹识别。

    

    行为指纹通过大语言模型（LLM）对一组共享提示的响应来识别该模型，可支持诸如验证已部署模型是否与其声称的检查点相符等应用。然而，已部署的模型通常采用随机解码提供服务，此时同一模型对相同提示可能产生不同的响应，且温度和 top-p 设置往往是未知的。诸如 LLM DNA 之类的现有表示方法对这种变化的处理能力有限。一种自然的解决方案是通过模型的响应分布来刻画模型，但这带来了两个挑战：首先，每个分布只能从少量采样响应中估计；其次，解码设置的改变可能以与更换检查点相当的程度改变响应。因此，仅凭指纹距离无法为模型身份提供完整的判断依据，识别通常还需依赖……（摘要在此处截断）

    arXiv:2611.27831v1 Announce Type: cross  Abstract: Behavioral fingerprints identify a large language model (LLM) from its responses to a shared set of prompts, enabling applications such as verifying whether a deployed model corresponds to the checkpoint it claims to be. Deployed models, however, are commonly served with stochastic decoding, under which the same model may produce different responses to an identical prompt, depending on an unknown temperature and top-$p$ setting. Existing representations such as LLM DNA provide limited treatment of this variation. A natural remedy is to characterize models by their response distributions, which, however, raises two challenges. First, each distribution must be estimated from only a small number of sampled responses. Second, a change in decoding setting can alter responses to the same extent as a change of checkpoint. Consequently, fingerprint distances alone provide an incomplete basis for model identity, and identification typically rel
    
[^4]: SLP-Bench：评估与改进推理型音频语言模型在临床语音上的声学论断

    SLP-Bench: Evaluating and Improving Acoustic Claims of Reasoning Audio Language Models on Clinical Speech

    [https://arxiv.org/abs/2611.27810](https://arxiv.org/abs/2611.27810)

    提出SLP-Bench基准，将推理型音频语言模型思维链中的每一项声学论断与临床医生的感知评分逐条对照，评估其事实性、稳定性和一致性，结果显示十个开源模型的论断平衡准确率仅为0.52至0.58。

    

    音频语言模型能够根据语音回答临床问题，其表现优于基于手工设计声学特征的基线方法。推理型音频语言模型在给出预测之前，会生成一条涉及说话人嗓音属性的思维链。模型对嗓音做出的每一项陈述，都需要与临床医生对同一录音的感知评分进行核对。然而在临床语音领域，现有评估方法仅对最终答案进行打分。我们提出了言语语言病理学家基准，这是一个将思维链中的每一项声学论断与临床医生对同一录音的感知评分进行逐条对照评分的基准。该基准涵盖五个由临床医生评分的数据集和25个感知维度，并从事实准确性、重采样下的稳定性，以及改写问题与不同问题之间的一致性等方面评估这些论断。在对十个开源权重推理型音频语言模型进行零样本评估中，论断的平衡准确率介于0.52至0.58之间，论断会在……（原文摘要在此处截断）

    arXiv:2611.27810v1 Announce Type: new  Abstract: Audio language models (ALMs) can answer clinical questions from speech, outperforming baselines built on handcrafted acoustic features. Reasoning ALMs generate a chain of thought that refers to properties of the speaker's voice before predicting. Each statement the model makes about the voice needs to be checked against clinicians' perceptual ratings of the same recording. On clinical speech, existing evaluations score only the final answer. We introduce Speech Language Pathologist Bench (SLP-Bench), a benchmark grading each acoustic claim in the chain of thought against clinicians' perceptual ratings of the same recording. It covers five clinician-rated datasets and 25 perceptual dimensions, and assesses claims for factuality, stability under resampling, and consistency across reworded and different questions. Across ten open-weight reasoning ALMs evaluated zero-shot, claim balanced accuracy ranges from 0.52 to 0.58, claims change under
    
[^5]: 评估大型语言模型在零样本多语言方面级情感分析中的提示策略

    Evaluating Prompting Strategies for Zero-Shot Multilingual Aspect-Based Sentiment Analysis with Large Language Models

    [https://arxiv.org/abs/2611.27807](https://arxiv.org/abs/2611.27807)

    该论文系统评估了七个开放权重大语言模型在12种语言和三个领域的零样本多语言目标-方面-情感检测任务中的四种提示策略，发现简单的整体式提示即可取得强大的零样本性能。

    

    尽管大型语言模型（LLMs）已经推动了自然语言理解的发展，但它们在零样本多语言设置下执行复杂的目标-方面-情感检测（TASD）——这是方面级情感分析（ABSA）中的一项关键任务——的能力仍未得到充分探索。现有的ABSA研究主要集中于英语，或依赖于微调的任务特定模型，导致在理解开放权重LLM如何处理跨不同语系和领域的复合情感三元组提取方面存在空白。我们对零样本多语言TASD进行了系统性评估，涵盖12种语言和三个领域，使用了七个指令微调的开放权重模型——参数规模从8B到35B，涵盖稠密架构和专家混合架构——并在四种提示范式下进行测试：整体式、思维链、顺序式和顺序改进式。我们的结果表明，直接的整体式提示即可实现强大的零样本性能

    arXiv:2611.27807v1 Announce Type: new  Abstract: While large language models (LLMs) have advanced natural language understanding, their ability to perform complex target-aspect-sentiment detection (TASD) -- a key task within aspect-based sentiment analysis (ABSA) -- in zero-shot multilingual settings remains underexplored. Existing ABSA research predominantly focuses on English or relies on fine-tuned task-specific models, leaving a gap in understanding how open-weight LLMs handle compound sentiment triplet extraction across diverse language families and domains. We present a systematic evaluation of zero-shot multilingual TASD across 12 languages and three domains using seven instruction-tuned open-weight models -- ranging from 8B to 35B parameters across dense and mixture-of-experts (MoE) architectures -- under four prompting paradigms: monolithic, chain-of-thought, sequential, and sequential refine. Our results show that direct monolithic prompting achieves strong zero-shot performa
    
[^6]: 用高阶注意力增强注意力机制

    Enhancing Attention with Higher-Order Attention

    [https://arxiv.org/abs/2611.27764](https://arxiv.org/abs/2611.27764)

    该论文提出三种新型注意力机制（条件互信息分解、高阶注意力、基于广义最大熵原理的集合关联注意力），用以增强自回归概率的信息含量，从而提升大语言模型推理与生成的能力。

    

    当前的解码器式大语言模型依赖于注意力层、前馈层和残差层的组合。目前所使用的注意力是自回归（AR）的，尽管我们注意到这并不严格成立。这促使我们引入其他注意力方法，这些方法虽然同样不严格自回归，但确实能增强自回归特性。第一种方法基于将互信息（MI）分解为一组条件互信息（CMI），使关键token与查询token对相关联。第二种是高阶注意力，引入相邻的关键token与查询token之间的关联。第三种引入一组关键token与查询token之间的关联，并使用广义最大熵原理得到一个跨越token序列的注意力表达式。所有这些方法都可以增强推理/生成中所使用的自回归概率的信息含量。论文还展示了这些方法在对比解码中的应用以及系统的分析（原文在此处截断）。

    arXiv:2611.27764v1 Announce Type: new  Abstract: Current decoder large language models rely on a combination of attention, feed forward and residual layers. The attention used is autoregressive (AR) though this is not strictly correct, as we note. That motivates the introduction of other attention methods that, while also not strictly AR, do enhance the AR character. One is based on a decomposition of mutual information (MI) into a set of Conditional MIs (CMIs) that correlate key tokens with pairs of query tokens. A second, higher-order attention, introduces correlations between contiguous key and query tokens. A third introduces correlations between a set of key and query tokens and uses a Generalized Maximum Entropy Principle to obtain an attention expression spanning a sequence of tokens. All these methods may enhance the information content of the autoregressive probabilities that are used in inference/generation. Applications to contrastive decoding and a systematic analysis of au
    
[^7]: 从新颖组合到有信息的缺席：语言学习中的结构许可、交际机会与形式选择

    From Novel Combinations to Informative Absence: Structural permission, communicative opportunity, and form choice in language learning

    [https://arxiv.org/abs/2611.27762](https://arxiv.org/abs/2611.27762)

    本文提出一个生成理论，将话语生成分解为结构许可、交际机会与形式选择，统一解释了语言学习者如何推断新颖表达的语法许可性并利用持续缺席作为负面证据，在30个人工语法上达到平均AUC 0.945。

    

    语言学习者必须推断哪些新颖表达在语法上是被许可的，以及持续的缺席何时会成为反对它们的证据。我们的生成理论通过反事实表达机会将这两种能力统一起来，把话语生成分解为结构许可、交际机会与形式选择三个环节。首先，共享的预测性后续成分在30个人工语法上恢复了对未见过的词-构式配对的结构许可（120个句子后平均AUC达0.945）。其次，在层级不确定性下，我们推导出一个将表达缺席与上下文多样性及上下文内重复相联系的联合证据定律，以及一个精确的有限样本上下文细化不等式。由此，观察相关表达能够校准预期使用，从而锐化负面证据；而更高的预期使用可以驱动许可更强的候选表达出现更陡峭的后验下降（在固定速率的实验中，六次缺席后初始排序被逆转）……

    arXiv:2611.27762v1 Announce Type: new  Abstract: Language learners must infer which novel expressions are grammatically licensed and when continued absence is evidence against them. Our generative theory unites these capacities via counterfactual expression opportunities, decomposing utterance generation into structural permission, communicative opportunity, and form choice. First, shared predictive continuations recover structural permission for unseen word-construction pairings across 30 artificial grammars (mean AUC 0.945 after 120 sentences). Second, under hierarchical uncertainty, we derive a joint evidence law linking omissions to context diversity and within-context repetition, and an exact finite-sample context-refinement inequality. Thus, observing related expressions calibrates expected use to sharpen negative evidence, and higher expected use can drive steeper posterior declines for stronger-permission candidates (reversing initial order after six omissions in a fixed-rate e
    
[^8]: RoSci：探索科学视觉语言模型的鲁棒性

    RoSci: Exploring Robustness in Science VLMs

    [https://arxiv.org/abs/2611.27750](https://arxiv.org/abs/2611.27750)

    该论文提出科学模型鲁棒性的概念，并构建了RoSci-Bench诊断基准和采用“解析-验证-决策”流程的RoSci-Agent框架，以增强视觉语言模型在科学任务中应对信息错误与信息缺失的能力。

    

    视觉语言模型（VLM）的最新进展显著推动了科学任务的发展。然而，这些模型通常被假定为接收完整且准确的信息，忽略了真实场景中信息错误/缺失所带来的挑战。为弥补这一空白，我们提出了“科学模型鲁棒性”的概念，该概念定义了模型在面对信息错误（不一致性）或信息缺失（不充分性）时检测异常、评估可解性以及进行自我校准的能力。具体而言，我们提出了RoSci-Bench，一个包含1,220个样本的诊断基准，采用种子-变体策略覆盖六种退化类型。为应对VLM的脆弱性，我们进一步提出了RoSci-Agent，一个采用“解析-验证-决策”（Parse-Verify-Decide）流程在推理前审查前提条件的框架。评估的VLM有时会回答前提无效或不完整的问题。RoSci-Agent取得了较高的……（原文摘要截断）

    arXiv:2611.27750v1 Announce Type: cross  Abstract: Recent advancements in Vision-Language Models (VLMs) enable significant progress in scientific tasks. However, these models are generally assumed to be presented with complete and accurate information, overlooking the challenges of erroneous/missing information in wild scenarios. To address this gap, we introduce the concept of Scientific Model Robustness, which defines a model's ability to detect anomalies, assess solvability, and perform self-calibration when faced with Information Error (inconsistencies) or Information Loss (insufficiency). Specifically, we present RoSci-Bench, a diagnostic benchmark of 1,220 samples using a seed-variant strategy across six degradation types. To address VLM fragility, we further propose RoSci-Agent, a framework that employs a Parse-Verify-Decide pipeline to vet premises before reasoning. The evaluated VLMs sometimes answer questions whose premises are invalid or incomplete. RoSci-Agent achieves a hi
    
[^9]: AutoOntologyBench：一个面向任务驱动的智能体知识工程基准测试

    AutoOntologyBench: A Task-Grounded Benchmark for Agentic Knowledge Engineering

    [https://arxiv.org/abs/2611.27734](https://arxiv.org/abs/2611.27734)

    该论文提出AutoOntologyBench，一个基于7个领域18个真实知识源、包含19,947个审核任务的任务驱动型基准，首次系统评估自主智能体构建的本体能否真正帮助解决实际任务并带来可衡量的收益。

    

    基础模型智能体正日益超越软件工程（SWE）领域，扩展到自主科学研究，而获取和整合领域知识仍然是一个关键挑战。近期的工业进展越来越多地采用知识图谱和基于本体的语义层，将异构知识组织成可复用的表示形式。我们提出的问题是：自主智能体能否构建出帮助它们解决真实任务的本体？这些本体相比其他知识表示形式是否提供了可衡量的收益？现有基准要么孤立地评估本体的正确性，要么孤立地评估智能体的任务成功率，导致自主获取知识的价值这一问题悬而未决。因此，我们提出了AutoOntologyBench，这是一个任务驱动的基准测试，包含19,947个经过审核的任务，这些任务基于来自7个领域的18个真实世界知识源，并配有源原生验证。该基准涵盖应用、维修、组织和建筑等领域……

    arXiv:2611.27734v1 Announce Type: new  Abstract: Foundation-model agents are increasingly extending beyond software engineering (SWE) into autonomous scientific research, where acquiring and integrating domain knowledge remains a critical challenge. Recent industrial advances have increasingly adopted knowledge graphs and ontology-based semantic layers to organize heterogeneous knowledge into reusable representations. We ask: Can autonomous agents construct ontologies that help them solve real tasks, and do they offer measurable benefits over alternative knowledge representations? Existing benchmarks evaluate ontology correctness or agent task success in isolation, leaving the value of autonomously acquired knowledge unresolved. We therefore introduce AutoOntologyBench, a task-grounded benchmark comprising 19,947 audited tasks grounded in 18 real-world knowledge sources across 7 domains, with source-native verification. Spanning Application, Repair, Organization, and Construction, Auto
    
[^10]: 超越数学、代码与逻辑：推理模型能否迁移至科学论证？

    Beyond Math, Code, and Logic: Do Reasoning Models Transfer to Scientific Argumentation?

    [https://arxiv.org/abs/2611.27722](https://arxiv.org/abs/2611.27722)

    该研究首次系统评估了语言模型在科学论证任务上的表现，发现即使前沿推理模型在领域特定论证任务上仍存在根本性能力差距，并提出了一种显式搭建论证推理脚手架的训练框架，显著提升了模型的科学论述能力。

    

    科学论述依赖于论证性推理：识别论点、评估证据以及构建连贯的回应。尽管近年来具备推理能力的语言模型在数学、编程和逻辑基准测试中表现出色，但它们参与科学论证的能力仍不明确。我们首次对语言模型进行了系统性评估，涵盖七项任务，包括论证挖掘、反驳生成和话语层面的推理，使用的数据来自研究论文、同行评审和科研基金申请书。我们的研究表明，即使是拥有强大通用推理能力的前沿模型，在处理领域特定的论证任务时也会遇到困难，这凸显了一个根本性的能力差距。为解决这一问题，我们引入了一个训练框架，显式地为语言模型的论证推理过程搭建脚手架，从而显著提升了它们在科学论述中的能力。

    arXiv:2611.27722v1 Announce Type: cross  Abstract: Scientific discourse depends on argumentative reasoning: identifying claims, evaluating evidence, and constructing coherent responses. While recent advances in reasoning-capable language models have demonstrated strong performance on mathematical, coding, and logical benchmarks, their ability to engage in scientific argumentation remains unclear. We present the first systematic evaluation of language models across seven tasks spanning argument mining, rebuttal generation, and discourse-level reasoning using research papers, peer reviews, and grant proposals. Our study reveals that even frontier models with strong general reasoning skills struggle with domain-specific argumentative tasks, highlighting a fundamental capability gap. To address this, we introduce a training framework that explicitly scaffolds the argumentative reasoning process in language models, substantially improving their competence in scientific discourse. The result
    
[^11]: 多语言情感分类的决策模型：与大语言模型和编码器的对比研究

    Decision Models for Multilingual Emotion Classification: A Comparative Study Against LLMs and Encoders

    [https://arxiv.org/abs/2611.27721](https://arxiv.org/abs/2611.27721)

    该研究首次在五种语言的多语言情感分类任务上系统比较了决策模型、大语言模型与微调编码器，发现最佳零样本系统的性能仅比微调编码器低约1个宏F1点。

    

    决策模型接收一个带有封闭答案列表的问题，并为每个答案返回一个概率，而非自由文本。它们具有低延迟和对提示措辞鲁棒性的优势，但尚无研究将多种决策模型与跨语言的微调编码器进行系统比较。我们探讨未经标注数据训练（零样本）使用的系统能否替代微调方法，以及决策格式相比通用大语言模型（LLM）能带来哪些额外优势。我们在五种语言的10个社交媒体数据集上，针对四种情绪（喜悦、悲伤、愤怒、恐惧），将三个决策模型（Jev、OpenAI Decisions、Strands Decider 2B）和两个采用意大利语指令的大语言模型（GPT-6 Luna、Qwen3.5-2B），与两个微调的多语言编码器（tabularisai、XLM-EMO）进行对比，并采用配对差异检验和等效性检验。结果表明，三个最佳的零样本系统（两个决策模型和一个大语言模型）的平均宏F1分数仅比在目标语言上微调的编码器低约1分。

    arXiv:2611.27721v1 Announce Type: cross  Abstract: Decision models receive a question with a closed list of answers and return a probability for each answer instead of free text. They promise low latency and robustness to prompt wording, but no study has compared several of them with fine-tuned encoders across languages. We ask whether systems used without training on labelled data (zero-shot) can replace fine-tuning, and what the decision format adds over a general-purpose large language model (LLM). We compare three decision models (Jev, OpenAI Decisions, Strands Decider 2B) and two LLMs (GPT-6 Luna, Qwen3.5-2B), given Italian instructions, with two fine-tuned multilingual encoders (tabularisai, XLM-EMO) on four emotions (joy, sadness, anger, fear) in 10 social-media datasets in five languages, with paired difference and equivalence tests. The three best zero-shot systems (two decision models and one LLM) score on average about one macro-F1 point below an encoder fine-tuned on the ta
    
[^12]: Hy-MT3：更准确、更鲁棒且更实用的真实场景多语言翻译模型

    Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild

    [https://arxiv.org/abs/2611.27673](https://arxiv.org/abs/2611.27673)

    Hy-MT3 系列通过将误译率降低 43%、漏译增译降低 74%、指令遵循错误降低 40%，并把语言覆盖扩展至 102 种，其仅 3B 激活参数的 MoE 模型在非思考模式下超越了众多旗舰大模型并逼近最强闭源思考模型。

    

    Hy-MT3 是混元多语言翻译模型家族的第三代，包括 Hy-MT3-1.8B、Hy-MT3-7B 和 Hy-MT3-30B-A3B（MoE 架构）。与 Hy-MT2 相比，Hy-MT3 的翻译更准确、更完整、更忠实于用户约束：在真实场景数据上，误译样本比例降低了 43%，漏译或增译问题降低了 74%，指令遵循错误降低了 40%，同时语言覆盖范围从 38 种语言扩展到 102 种。仅需 3B 激活参数，Hy-MT3-30B-A3B 便在非思考模式下取得了所有评测模型中的最佳整体性能，超越了 Qwen3.8-Max、Kimi-K3.1-Pro 和 DeepSeek-V4.1-Flash 等旗舰模型，并接近 GPT-6.1-Sol 和 Gemini-4-Argon 等最强的闭源思考模型，同时超越了消耗超过 30 倍资源的模型。

    arXiv:2611.27673v1 Announce Type: cross  Abstract: Hy-MT3 is the third generation of the Hunyuan multilingual translation model family, including \textbf{Hy-MT3-1.8B}, \textbf{Hy-MT3-7B} and \textbf{Hy-MT3-30B-A3B (MoE)}. Compared with Hy-MT2, Hy-MT3 translates \textbf{more accurately, more completely and more faithfully to user constraints}: on real-world scenario data, the share of samples with mistranslations is reduced by \textbf{43\%}, omissions or additions by \textbf{74\%}, and instruction-following errors by \textbf{40\%}, while language coverage is expanded from \textbf{38 to 102} languages. With only 3B activated parameters, \textbf{Hy-MT3-30B-A3B achieves the best overall performance among all evaluated models in non-thinking mode}, including flagship models such as Qwen3.8-Max, Kimi-K3.1-Pro and DeepSeek-V4.1-Flash, and approaches the strongest closed-source thinking models such as GPT-6.1-Sol and Gemini-4-Argon, while surpassing models that consume \textbf{more than 30$\ti
    
[^13]: 使用稀疏自编码器审计扩散语言模型的训练后遗忘

    Auditing Post-Training Unlearning in Diffusion Language Models using Sparse Autoencoders

    [https://arxiv.org/abs/2611.27652](https://arxiv.org/abs/2611.27652)

    该论文利用稀疏自编码器与去噪状态重放，首次对扩散语言模型（LLaDA-8B-Instruct）的训练后遗忘进行机制审计，探索通过识别与遗忘数据相关的特征并在各去噪阶段动态调整其抑制来改进选择性遗忘。

    

    机器遗忘是一种隐私保护方法：其目标是在保留模型对应保留信息的正常行为的同时，移除指定数据的影响。尽管越来越多的研究在自回归大语言模型中探讨这一目标，但机器遗忘在扩散语言模型（DLM）中仍未得到充分研究。DLM通过迭代去噪生成文本，会暴露出部分掩码的中间状态，其内部表示在整个生成过程中不断变化。这引出了一个DLM特有的问题：能否通过识别与待遗忘数据相关的特征，并在不同去噪阶段调整对它们的抑制方式，从而改进选择性遗忘？我们在LLaDA-8B-Instruct上对这一问题进行了受控研究。为了评估预训练期间可能已编码知识的遗忘效果，我们使用WMDP作为遗忘领域基准。使用top-k稀疏自编码器和去噪状态重放，我们……

    arXiv:2611.27652v1 Announce Type: new  Abstract: Machine unlearning is an approach to privacy protection: it aims to remove the influence of designated data while preserving the model's behavior on information that should be retained. While a growing body of work investigates this objective in autoregressive large language models, machine unlearning remains underexplored in diffusion language models (DLMs). DLMs generate text through iterative denoising, exposing partially masked states whose internal representations change throughout the generation process. This raises a DLM-specific question: can selective forgetting improve by identifying forget-data-associated features and adapting their suppression across denoising stages? We conduct a controlled investigation of this question in LLaDA-8B-Instruct. To evaluate the unlearning of knowledge plausibly encoded during pre-training, we use WMDP as our forget-domain benchmark. Using top-k sparse autoencoders and denoising-state replay, we
    
[^14]: 大语言模型智能体中的势能涌现：统计物理学视角

    Potential emergence in LLM agents: a statistical physics perspective

    [https://arxiv.org/abs/2611.27618](https://arxiv.org/abs/2611.27618)

    该论文从统计物理学视角提出将智能体状态作为中间变量，通过配分函数与全局势能来定量测量并预测大语言模型智能体在后训练过程中行为的组织与涌现。

    

    近年来，从人类偏好或语料库统计的角度理解大语言模型智能体的行为引起了极大关注。本文提出将智能体状态作为中间变量，通过这些变量可以对后训练过程进行测量和预测。从统计物理学的视角出发，我们研究了一个支配大语言模型智能体推理中状态转移的全局势能的涌现。我们引入配分函数来建模理想的KL正则化转移分布，并追踪势能结构、平均奖励和状态熵。我们在两个Qwen3.5-9B任务上展示了我们的方法：安全测试选择和物理推理审计。我们观察到，正如我们的框架所预测的，势能结构在后训练过程中不断增强，同时奖励与状态熵的联合变化提供了一致性证据。因此，状态为后训练如何组织智能体行为提供了定量描述。

    arXiv:2611.27618v1 Announce Type: cross  Abstract: There has been significant recent interest in understanding the behaviour of LLM agents from the perspective of human preferences or corpus statistics. In this paper we propose agent states as intermediate variables through which post-training can be measured and predicted. From a statistical physics perspective, we study the emergence of a global potential which governs the transition of state in LLM agent reasoning. We introduce a partition function to model the ideal KL-regularised transition distribution and track potential structure, mean reward and state entropy. We demonstrate our approach in two Qwen3.5-9B tasks: safety-test selection and physics-reasoning auditing. We observe that the potential structure strengthens during post-training as predicted from our framework, while joint changes in reward and state entropy provide consistency evidence. States thus provide a quantitative description of how post-training organises agen
    
[^15]: Tasklets：面向语言模型架构的单元测试基准

    Tasklets: A Unit-Test Benchmark for Language Model Architectures

    [https://arxiv.org/abs/2611.27607](https://arxiv.org/abs/2611.27607)

    Tasklets 是一个由小型合成任务组成的标准化基准，按能力类别（回忆、序列操作、状态跟踪）组织，用于低成本、可重复地评估语言模型序列混合层的基础计算能力。

    

    我们提出了 Tasklets，这是一个由小型合成任务组成的基准，用于评估语言模型序列混合层的计算能力。这些任务探测回忆、序列操作和状态跟踪等基础能力。此类任务在架构研究中被广泛使用，但任务选择、形式化和评估方面的差异使得跨研究比较变得困难。Tasklets 将这些任务汇集在一起，按其测试的能力进行组织，并为评估不同架构层类型如何学习基本计算提供了统一的设置。我们通过对小型模型的受控实验比较了广泛的层类型。我们的任务实现和评估代码支持对新架构提案进行可重复、低成本的测试。

    arXiv:2611.27607v1 Announce Type: new  Abstract: We introduce Tasklets, a benchmark of small synthetic tasks for evaluating the computational capabilities of language model sequence-mixing layers. These tasks probe foundational capabilities such as recall, sequence manipulation, and state tracking. Such tasks are widely used in architecture research, but differences in task selection, formulation, and evaluation make comparisons across studies difficult. Tasklets brings these tasks together, organizing them by the capabilities they test and providing a shared setup for evaluating how different architectural layer types learn fundamental computations. We compare a broad range of layer types through controlled experiments with small models. Our task implementations and evaluation code support reproducible, low-cost testing of new architectural proposals: https://github.com/zaydzuhri/flame/tree/tasklets.
    
[^16]: 模拟认知行为治疗对话中的治疗联盟破裂与修复

    Alliance Rupture and Repair in Simulating Cognitive Behavioral Therapy Dialogues

    [https://arxiv.org/abs/2611.27544](https://arxiv.org/abs/2611.27544)

    该论文提出ALLY模拟框架，首次将治疗联盟的破裂、修复与治疗计划的动态调整引入认知行为治疗对话模拟，使模拟咨询师能够像真实咨询师一样在多轮对话中察觉并修复联盟破裂。

    

    在认知行为治疗（CBT）中，常规治疗工作可能会使来访者与咨询师之间的工作联盟变得紧张，此时咨询师必须察觉联盟的破裂，对其进行修复，并调整后续的治疗工作。现有的CBT模拟器虽然能够模拟来访者的阻抗行为并遵循结构化的治疗计划，但很少追踪联盟状态，也不允许咨询师在多轮对话中修复破裂并据此调整计划。我们提出了ALLY，一个将联盟破裂、修复和重新规划引入持续进行的CBT对话的模拟框架。每个来访者都拥有一个隐藏的联盟状态，该状态决定了来访者参与治疗的开放程度。与真实咨询师一样，咨询师智能体只能看到对话内容，它必须察觉来访者何时退缩或对治疗工作产生抵触，修复联盟的破裂，并根据所获得的信息重新规划会谈的其余部分。自动评估表明，ALLY的咨询师比现有公开的合成对话系统中的咨询师更接近真实的CBT咨询师。

    arXiv:2611.27544v1 Announce Type: new  Abstract: In cognitive behavioral therapy (CBT), routine work can strain the client-counselor working alliance, and the counselor must then notice the rupture, repair it, and adapt the work that follows. Existing CBT simulators model client resistance and follow structured treatment plans, but rarely track alliance state or allow counselors to repair ruptures over multiple turns and revise their plans accordingly. We present ALLY, a simulation framework that brings alliance rupture, repair, and replanning into ongoing CBT dialogue. Each client has a hidden alliance state that governs how openly the client engages. Like a real counselor, the counselor agent sees only the dialogue. It must notice when the client pulls back or pushes against the work, repair the rupture, and replan the rest of the session around what it has learned. Automatic evaluation shows that ALLY's counselors are closer to real CBT counselors than those of existing public synth
    
[^17]: Open CDLM：连续扩散语言模型的基准测试与库

    Open CDLM: A Benchmark and Library for Continuous Diffusion Language Models

    [https://arxiv.org/abs/2611.27536](https://arxiv.org/abs/2611.27536)

    该论文提出了 Open CDLM，一个用于评估连续扩散语言模型及其采样器的基准测试和开源库，旨在解决现有评估方法中变分上界松弛导致比较混淆、以及采样多样性度量无法有效识别重复生成的问题。

    

    对连续扩散语言模型（CDLM）进行可靠比较，需要同时评估模型拟合优度和生成文本的分布。然而两种常见做法仅能部分实现这一目标：基于似然的评估通常报告离散负对数似然的变分上界，其松弛量可能混淆比较结果；基于生成样本的评估通常将生成困惑度与序列内一元词熵相结合，但即使采样器每次抽取都生成相同的文本，这两个分数仍可能表现良好。我们提出了 Open CDLM——一个用于评估 CDLM 及其采样器的基准测试，并描述了配套开源库的设计，以支持可复现的比较。对于基于似然的评估，我们阐明了所报告的困惑度上界能够支持哪些模型拟合优度比较，以及这些结论如何依赖于评估流程；对于基于样本的评估，我们（摘要在此处截断）

    arXiv:2611.27536v1 Announce Type: new  Abstract: Reliable comparisons of continuous diffusion language models (CDLMs) require evaluating both model fit and the distribution of generated text. Two common practices leave this goal only partially addressed. Likelihood-based evaluation often reports a variational upper bound on discrete negative log-likelihood, whose slack can confound comparisons. Generated-sample-based evaluation commonly combines generative perplexity with within-sequence unigram entropy; both scores can remain favorable even when a sampler produces the same text on every draw. We introduce \textbf{Open CDLM}, a benchmark for evaluating CDLMs and their samplers, and describe the design of an accompanying open-source library to support reproducible comparisons. For likelihood-based evaluation, we clarify which comparisons of model fit are supported by reported perplexity bounds and how those conclusions depend on the evaluation procedure. For sample-based evaluation, we 
    
[^18]: PaMuk-Bench：一个面向视觉-语言任务的平行多语言多任务基准

    PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks

    [https://arxiv.org/abs/2611.27521](https://arxiv.org/abs/2611.27521)

    本文提出了PaMuk-Bench，一个包含超过10K条平行翻译提示（涵盖阿拉伯语、丹麦语、法语、印地语、意大利语和罗马尼亚语六种语言）的多语言多任务视觉-语言基准，用于系统评估视觉-语言模型在低资源语言上跨多种任务的表现。

    

    尽管视觉-语言模型（VLM）目前在英语等被广泛研究的语言上取得了显著成果，但由于缺乏标注数据集，它们在某些低资源语言上表现出显著的性能下降。为此，我们提出了PaMuk-Bench，一个新颖的面向视觉-语言任务的平行多语言多任务基准。我们的基准包含一组超过10K条精选的英文提示，这些提示被翻译成六种语言：阿拉伯语、丹麦语、法语、印地语、意大利语、罗马尼亚语。通过半自动过程获得的平行翻译，以及配套的生成和检索图像，使得能够在广泛的任务上对VLM进行对比性开发和评估，这些任务包括：文本到图像生成、图像描述生成、基于内容的图像检索、图像生成难度估计、图像检索难度估计。我们通过实验证明，零样本英语和多语言预训练……

    arXiv:2611.27521v1 Announce Type: cross  Abstract: While vision-language models (VLMs) currently obtain notable results for highly-studied languages, e.g. English, they exhibit significant performance drops in certain low-resource languages, especially due to the lack of annotated datasets. To this end, we propose PaMuk-Bench, a novel parallel multilingual multitask benchmark for vision-language tasks. Our benchmark includes a set of over 10K curated English prompts that are translated into six languages: Arabic, Danish, French, Hindi, Italian, Romanian. Parallel translations, which are obtained via a semi-automatic process, along with accompanying generated and retrieved images enable the comparative development and evaluation of VLMs across a wide range of tasks: text-to-image generation, image captioning, content-based image retrieval, image generation difficulty estimation, image retrieval difficulty estimation. We empirically show that zero-shot English and multilingual pre-traine
    
[^19]: BPE模型见过多少字节？基于语料库统计与嵌入范数估计预训练中的字节token暴露量

    How Many Bytes Has a BPE Model Seen? Estimating Byte-Token Exposure in Pretraining from Corpus Statistics and Embedding Norms

    [https://arxiv.org/abs/2611.27506](https://arxiv.org/abs/2611.27506)

    本文通过语料库分词统计与嵌入范数分析两条独立路径，量化估计了BPE模型中256个单字节后备token在预训练中的暴露程度，发现真正的UTF-8后备字节仅占总token的0.15%至0.27%，且后备率在不同语言间相差达四个数量级。

    

    字节级BPE词表保留了256个单字节后备token。当将此类模型转换为纯字节级模型时，这些token提供了起始嵌入，但它们在预训练中获得了多少训练？本文通过两条独立路径估计了Qwen3.8中的字节token暴露量。首先，对覆盖40个领域和33种语言的540 MB公开语料库进行分词，并将各领域的token占比与假设的预训练数据混合比例相结合。结果显示，单字节token占总token数的13%至36%，主要为ASCII字符；在主流混合比例下，真正的UTF-8后备字节（0x80-0xFF）仅占0.15%至0.27%，即每万亿个预训练token中约出现20亿次。后备率在不同语言之间相差达四个数量级，反映出各书写系统被词表覆盖的程度差异。其次，分析了四个开源模型的嵌入矩阵和输出矩阵的行范数，以未训练的padding行作为零暴露基线，并利用普通token的（原文在此处截断）

    arXiv:2611.27506v1 Announce Type: new  Abstract: Byte-level BPE vocabularies retain 256 single-byte fallback tokens. When converting such a model into a pure byte-level model, these tokens provide the starting embeddings. How much training have they received? We estimate byte-token exposure in Qwen3.8 using two independent routes. First, we tokenize 540 MB of public corpora covering 40 domains and 33 languages, then combine per-domain token shares with assumed pretraining mixtures. Single-byte tokens account for 13 to 36% of tokens, mostly ASCII; genuine UTF-8 fallback bytes (0x80-0xFF) account for 0.15 to 0.27% under mainstream mixtures, or about 2 billion occurrences per trillion pretraining tokens. Fallback rates vary by four orders of magnitude across languages, reflecting how well each script is covered by the vocabulary. Second, we analyze embedding and output-matrix row norms for four open models, using untrained padding rows as a zero-exposure baseline and ordinary tokens with 
    
[^20]: 助手人格在激活空间中因语言而异

    The Assistant Persona Differs Across Languages in Activation Space

    [https://arxiv.org/abs/2611.27504](https://arxiv.org/abs/2611.27504)

    该研究首次从机制层面揭示了大语言模型中助手人格方向在不同语言间差异显著（与英语轴余弦相似度仅0.378–0.911）且与语言资源水平相关，证明从英语提取的激活方向无法跨语言泛化。

    

    大语言模型在不同语言之间的行为存在差异，且在低资源语言中明显不够安全。这一差距在行为层面已被充分证实，但在机制层面尚未得到解释。激活空间中的方向，包括助手人格方向，几乎总是从英语中提取并被假定可以跨语言泛化。我们在 Qwen3-32B 中提取了31种语言的助手人格方向，发现所得的轴存在显著差异：它们与英语轴的余弦相似度介于0.378到0.911之间。此外，每种语言的助手人格方向及其幅值都与该语言的资源水平相关。我们还发现，模型的默认助手状态在低资源语言中距离“乐于助人的助手”这一极点更远。最后，相较于沿英语轴进行干预（steering），沿语言自身的轴进行干预会更强烈地放大基于人格的越狱攻击。这些结果表明……

    arXiv:2611.27504v1 Announce Type: cross  Abstract: Large language models behave differently across languages, and are noticeably less safe in lower-resource languages. This gap is well established behaviorally, but not mechanistically. Directions in activation space, including the assistant persona direction, are almost always extracted from English and assumed to generalize across languages. We extract the assistant persona direction in 31 languages in Qwen3-32B, and find that the resulting axes differ substantially: their cosine similarity to the English axis ranges from 0.378 to 0.911. Moreover, each language's assistant persona direction and its magnitude are correlated with the language's resource level. We also find that the model's default assistant state is farther from the helpful assistant pole in lower-resource languages. Finally, steering against a language's own axis amplifies persona-based jailbreak more than steering against the English axis. These results indicate that 
    
[^21]: Typhoon OCR 2：从泰文文档转录到多语言、多任务 OCR

    Typhoon OCR 2: From Thai Document Transcription to Multilingual, Multi-Task OCR

    [https://arxiv.org/abs/2611.27494](https://arxiv.org/abs/2611.27494)

    Typhoon OCR 2 将前代专注泰文转录的模型扩展为单一的多语言多任务 OCR 系统，通过 GRPO 训练的紧凑 2B/0.8B 模型同时支持提示驱动的信息提取与泰文、英文、中文的整页结构化解析，并在泰文基准上大幅超越前代且保持英文性能。

    

    Typhoon OCR 1.5 专注于阅读泰文文档。然而，在真实的工作流程中，用户通常需要提取特定信息或保留整页结构以便进一步处理。Typhoon OCR 2 在单一模型中同时处理这两项任务，支持提示驱动的信息提取和结构化的整页解析。整页解析覆盖泰文、英文和中文。我们发布了紧凑的 2B 和 0.8B 模型，这些模型基于开源文档解析器，在精心整理的泰文数据和公开多语言数据上进行监督微调，随后采用带任务奖励和未完成响应惩罚规则的群体相对策略优化（GRPO）。在所评估的开源模型中，Typhoon OCR 2 2B 在泰文 MDPBench 上得分最高（80.7，而 Typhoon OCR 1.5 为 68.9），在涵盖识别、提取和文档理解的 ThaiOCRBench 上也得分最高（59.91 对 40.45）。该 2B 模型在英文 olmOCR-bench 上仍保持接近其基础模型的水平（84.7 对 84.4）……

    arXiv:2611.27494v1 Announce Type: new  Abstract: Typhoon OCR 1.5 focused on reading Thai documents. However, in real-world workflows, users often need to extract specific information or preserve the structure of an entire page for further processing. Typhoon OCR 2 handles both tasks in a single model, with prompt-driven extraction and structured full-page parsing. Full-page parsing covers Thai, English, and Chinese. We release compact 2B and 0.8B models adapted from open document parsers through supervised fine-tuning on curated Thai and public multilingual data, followed by Group Relative Policy Optimization (GRPO) with task rewards and a rule penalizing unfinished responses. Among evaluated open models, Typhoon OCR 2 2B scores highest on Thai MDPBench (80.7 versus 68.9 for Typhoon OCR 1.5) and on ThaiOCRBench (59.91 versus 40.45), which covers recognition, extraction, and document understanding. The 2B model remains close to its base on English olmOCR-bench (84.7 versus 84.4) and on 
    
[^22]: 无迁移的对齐：分词器影响多语言语言模型预训练过程中的跨语言表示

    Alignment without Transfer: Tokenization Impacts Cross-lingual Representations during Multilingual Language Model Pretraining

    [https://arxiv.org/abs/2611.27491](https://arxiv.org/abs/2611.27491)

    本研究通过仅改变分词器预训练39个单语和双语模型，发现改进的词表分配能显著增强模型中早期层的跨语言表示对齐以及前馈神经元的跨语言知识共享，但揭示了表示对齐与跨语言迁移之间存在分离现象。

    

    分词是多语言自然语言处理流程中最关键的步骤之一。然而，分词器通常对低资源语言的词表分配不足。尽管改进的词表分配能够带来更好的性能，但分词如何在预训练过程中塑造跨语言表示与迁移仍鲜为人知。为解决这一问题，我们在FineWeb2-HQ数据集上预训练了39个单语和双语的10亿参数解码器模型，涵盖英语、德语、法语、阿拉伯语和中文，仅改变分词器这一变量。我们的结果表明，改进的词表分配 显著提升了模型早期至中期层的跨语言表示对齐，这一效应是先前仅测量输出层的工作所无法观察到的； 增加了前馈神经元中的跨语言参数化知识共享，并得到了相关性证据和因果证据的支持； 揭示了“对齐-迁移分离”现象，即跨语言迁移基本保持不变。原文在此处截断。

    arXiv:2611.27491v1 Announce Type: cross  Abstract: Tokenization represents one of the most crucial steps in mulilingual NLP pipelines. Yet, tokenizers routinely under-allocate vocabulary to low-resource languages. While improved allocation leads to better performance, it remains poorly understood how tokenization shapes cross-lingual representations and transfer during pretraining. To address this, we pretrain 39 mono- and bilingual 1B decoder-only models on FineWeb2-HQ across English, German, French, Arabic and Chinese, varying only the tokenizer. Our results demonstrate that improved vocabulary allocation (i) substantially raises cross-lingual representation alignment in early-to-mid layers, an effect that prior work measuring only the output layer could not observe, (ii) increases cross-lingual parametric knowledge sharing in feed-forward neurons, supported by correlational and causal evidence, and (iii) an alignment-transfer dissociation, where cross-lingual transfer remains essent
    
[^23]: 面向实时物理人工智能的高效自然语言到时序逻辑翻译

    Efficient Natural Language to Temporal Logic Translation for Real-Time Physical AI

    [https://arxiv.org/abs/2611.27489](https://arxiv.org/abs/2611.27489)

    该论文提出将自然语言到线性时序逻辑的翻译分解为回答一组预定义答案的问题，用轻量级System-One分类器替代大语言模型的文本生成，从而在计算和能源资源受限的机器人上实现高效实时的翻译。

    

    当任务指令以形式化语言表述时，机器人可以被控制执行任务并保证正确性。大语言模型（LLM）已被广泛用于让用户以自然语言（NL）提供此类指令，即将自然语言翻译为形式化语言。然而，将此类翻译器部署在机器人上时，我们面临计算和能源资源有限的问题，而将计算或数据卸载到云端可能不可行或不理想。此外，大语言模型是一个通用文本生成器，而固定形式化语言片段中的公式遵循明确定义的语法。因此，我们提出疑问：是否真的需要文本生成。对于线性时序逻辑（LTL）这一形式化语言，我们证明这种翻译可以转而分解为回答一组具有预定义答案的问题。随后，我们构建了一个System-One模型——一类新兴的基于自然语言的分类器——来提供这些答案，从而实现……

    arXiv:2611.27489v1 Announce Type: new  Abstract: Robots can be controlled to perform tasks with guaranteed correctness when the task instructions are stated in a formal language. Large language models (LLMs) have been widely used to allow users to provide such instructions in natural language (NL), by translating NL into formal language. When deploying such translators on a robot, however, we face limited computing and energy resources, while offloading computation or data to the cloud may be infeasible or undesirable. Yet an LLM is a general-purpose text generator, whereas a formula in a fixed formal language fragment follows a well-defined syntax. We therefore ask whether text generation is needed at all. For the formal language of linear temporal logic (LTL), we show that this translation can instead be broken down into answering a set of questions with pre-defined answers. We then construct a System-One model, a recent class of NL-based classifier, to provide such answers and enabl
    
[^24]: 关于在高风险AI招聘中使用大语言模型的挑战：符合《人工智能法案》法律合规性的指导性问题

    On the Challenges of Using LLMs in High-Risk AI Recruitment: Guiding Questions for Legal Conformity with the AI Act

    [https://arxiv.org/abs/2611.27462](https://arxiv.org/abs/2611.27462)

    本文提出一个工具包，将欧盟《人工智能法案》合规评估的核心法律义务转化为技术规范要求和指导性问题，以支持“合法设计”的AI辅助招聘系统开发，重点关注第三方GPAI模型集成到招聘关键环节这一最具挑战性的场景。

    

    欧盟《人工智能法案》（AI Act）对高风险AI系统的提供者规定了详细的义务，但在操作层面提供的指导有限，尤其是当这些系统集成了第三方通用人工智能（GPAI）模型时。本文介绍了一个工具包，通过将AI法案合规评估的核心法律义务——具体为第9-15条和附件IV——转化为技术规范要求及相应的指导性问题，从而促进“合法设计”的AI辅助招聘系统的开发。虽然招聘流程可能包含技术复杂程度不同、算法透明度各异的组件，但本文聚焦于最具挑战性的情形：将第三方GPAI模型集成到选拔过程的关键阶段，包括简历与职位描述解析、申请人评分以及入围名单生成。这一情形带来了重大的挑战……

    arXiv:2611.27462v1 Announce Type: cross  Abstract: The EU Artificial Intelligence Act (AI Act) imposes detailed obligations on providers of high-risk AI systems but offers limited operational guidance, particularly when these systems integrate third-party General-Purpose AI (GPAI) models. This paper introduces a toolkit that fosters the development of legal-by-design AI-assisted recruitment systems by translating the core legal obligations of the AI Act's conformity assessment---specifically Articles 9--15 and Annex IV---into techno-normative requirements and corresponding guiding questions. While recruitment pipelines may involve components with varying technical complexity and differing levels of algorithmic transparency, this paper focuses on the most challenging case: the integration of third-party GPAI models into critical stages of the selection process, including resume and job description parsing, applicant scoring, and shortlist generation. This scenario presents significant c
    
[^25]: 结构不同，功能持久：模型优化下的能力区域

    Structurally Different, Functionally Persistent: Capability Regions Under Model Optimization

    [https://arxiv.org/abs/2611.27445](https://arxiv.org/abs/2611.27445)

    该研究通过对量化、剪枝和LoRA适配的分析发现，尽管模型优化会导致能力区域发生显著的参数重组，但优化前识别的编码能力区域在功能上依然持久——消融这些原始区域仍能消除编码性能，表明它们始终是有效的干预目标。

    

    大型语言模型（LLM）的能力已被认为与局部化的参数区域相关联，这使得针对性的模型编辑和干预成为可能。然而，目前尚不清楚在模型优化之前识别出的区域在优化之后是否仍然是有效的干预目标。我们在三个指令微调模型上，评估了在量化、剪枝以及基于LoRA的适配过程中编码能力区域的功能持久性和参数重叠情况。尽管重新提取的区域表现出显著的参数重组，但在几乎所有条件下，优化后消融原始区域会消除或几乎消除编码性能，而幅度匹配的对照区域则保持性能不变。分区分析进一步表明，在大多数条件下，原始区域与重新提取区域之间的重叠子集占了所观察到的消融效果的大部分。这些结果表明，参数重叠与功能持久性（原文在此截断）……

    arXiv:2611.27445v1 Announce Type: new  Abstract: Large language model (LLM) capabilities have been associated with localized parameter regions, enabling targeted model editing and intervention. However, it remains unclear whether a region identified before model optimization remains an effective intervention target afterward. We evaluate the functional persistence and parameter overlap of coding capability regions across three instruction-tuned models under quantization, pruning, and LoRA-based adaptation. Although re-extracted regions exhibit substantial parameter reorganization, ablating the original region after optimization eliminates or nearly eliminates coding performance in almost all conditions, while magnitude-matched controls preserve performance. Partition analysis further shows that the shared subset of the original and re-extracted regions accounts for most of the observed ablation effect in the majority of conditions. These results demonstrate that parameter overlap and f
    
[^26]: 静默仲裁：大型语言模型如何解决同等权威的冲突指令

    Silent Arbitration: How Large Language Models Resolve Co-Authoritative Conflicting Instructions

    [https://arxiv.org/abs/2611.27444](https://arxiv.org/abs/2611.27444)

    研究发现大语言模型在面对同等权威的冲突人格指令时会进行“静默仲裁”——虽然在直接检查中72%会承认冲突，但在实际任务响应中仅有22%明确告知用户，其余情况下模型默默采用某一方的立场而不透明地解决冲突。

    

    大型语言模型越来越多地接收关于角色、价值观、利益和沟通风格的高级指令。即使提示词赋予这些指令同等权威且未提供优先级规则，它们之间也可能发生冲突。我们研究了同权威人格冲突，即单个提示词同时分配两个对立的人格。基于已发表源材料中的八种冲突类型，我们在五个大语言模型上测量了冲突承认度、解决策略和方向性立场。模型在72%的直接检查中承认不兼容性，但在任务响应中仅有22%明确表达出来，这揭示了“静默仲裁”现象。支配性策略占双重人格响应的45%，而35%被标记为妥协的响应仍保留了方向性倾向。提示词框架对仲裁可见形式的改变远大于普通的改述变化。这些发现表明，评估指令遵循应当……

    arXiv:2611.27444v1 Announce Type: new  Abstract: Large language models increasingly receive high-level instructions about roles, values, interests, and communication style. Such instructions can conflict even when a prompt assigns them equal authority and provides no priority rule. We study co-authoritative persona conflict, in which a single prompt jointly assigns two opposing personas. Across eight conflict types grounded in published source material and five LLMs, we measure conflict acknowledgment, resolution strategy, and directional stance. Models acknowledge incompatibility in 72% of direct checks but explicitly surface it in only 22% of task responses, revealing silent arbitration. Dominance accounts for 45% of dual-persona responses, and 35% of compromise-labeled responses retain directional lean. Prompt framing further changes the visible form of arbitration substantially more than ordinary paraphrase variation. These findings show that evaluating instruction following should
    
[^27]: 无音频、无转写文本，无妨：仅使用领域文档的基于LLM的ASR适配

    No Audio, No Transcripts, No Problem: LLM-based ASR Adaptation Using Only Domain Documents

    [https://arxiv.org/abs/2611.27434](https://arxiv.org/abs/2611.27434)

    该研究首次提出仅使用领域文档（无需目标音频和转写文本）对基于LLM的语音识别系统进行领域适配，通过将文档直接作为书面文本或由LLM渲染为合成口语对话，其适配效果可媲美甚至接近使用真实转写文本的结果。

    

    能否在没有目标音频和转写文本的情况下，仅使用书面文档将语音识别器适配到新领域？这是现实的接入场景：新客户交付的是原始文档，而非纯文本适配通常所假设的转写文本。我们通过两种方式将领域文档转化为适配文本：直接作为书面句子，或由大语言模型（LLM）渲染成合成口语对话，并将其输入最近提出的一种基于去噪的纯文本适配方法。我们表明，在两个语料库上的域内、域外和跨域设置中，仅使用文档的适配可以达到或非常接近使用真实转写文本的效果；书面渲染还是对话式渲染更优取决于说话风格和生成的文本量。据我们所知，这是首个针对基于LLM的ASR进行仅文档适配的研究。由于目前没有现有语料库同时包含音频、转写文本和文档……

    arXiv:2611.27434v1 Announce Type: cross  Abstract: Can a speech recognizer be adapted to a new domain with no target audio and no transcripts, using only written documentation? This is the realistic onboarding setting: a new customer hands over raw documentation, not the transcripts that text-only adaptation normally assumes. We turn a domain's documents into adaptation text in two ways, directly as written sentences or rendered by a large language model (LLM) into synthetic spoken dialogs, and feed them to a recently proposed denoising-based text-only adaptation method. We show that, across in-domain, out-of-domain, and cross-domain settings on two corpora, document-only adaptation can match or closely approach real transcriptions; whether written or conversational rendering is best depends on speaking style and the amount of text generated. To our knowledge, this is the first study of document-only adaptation of LLM-based ASR. As no existing corpus pairs audio, transcripts, and docum
    
[^28]: ORBIT：面向跨语言多模态事实核查的基于偏移量的激活引导方法

    ORBIT: Offset-based Activation Steering for Cross-lingual Multimodal Fact Verification

    [https://arxiv.org/abs/2611.27421](https://arxiv.org/abs/2611.27421)

    提出了推理时算法干预方法ORBIT，通过从平行声明-证据实例中估计偏移向量并修正声明与证据表示，在不需微调的情况下显著恢复小型视觉-语言模型在低资源语言上的多模态事实核查性能。

    

    多模态事实核查面临一个根本性的可及性瓶颈：尽管大型专有模型表现出跨语言鲁棒性，小型视觉-语言模型却在低资源语言上遭受严重且不成比例的准确率崩溃。弥合这一差距传统上需要巨大的模型规模或大量的特定语言微调。在本工作中，我们提出了通过双语推理时技术的偏移恢复方法（ORBIT），这是一种在推理时缓解上述性能崩溃的算法干预手段。ORBIT从平行的声明-证据实例中估计特定跨度的偏移向量，并将其应用于声明和证据的表示上，以针对由低资源证据引发的跨语言退化问题。跨越81种配置的实验表明，ORBIT在低资源设置下显著恢复了准确率和F1分数，并将校准偏移范数揭示为一种诊断性代理指标，同时优于基于翻译的方法。

    arXiv:2611.27421v1 Announce Type: new  Abstract: Multimodal fact verification faces a fundamental accessibility bottleneck: while large proprietary models exhibit cross-lingual robustness, small vision-language models suffer a severe, disproportionate accuracy collapse on low-resource languages. Closing this gap traditionally requires prohibitive scale or extensive language-specific fine-tuning. In this work, we propose Offset Recovery via Bilingual Inference-time Technique (ORBIT), an algorithmic intervention that mitigates this collapse at inference time. ORBIT estimates span-specific offset vectors from parallel claim-evidence instances and applies them to claim and evidence representations, targeting the cross-lingual degradation induced by low-resource evidence. Experiments across 81 configurations reveal that ORBIT substantially recovers both accuracy and F1-score in low-resource settings, exposes calibration offset norms as a diagnostic proxy, and outperforms translation-based a
    
[^29]: HR-Merge：基于余量加权残差精炼的模型合并方法

    HR-Merge: Model Merging with Headroom-Weighted Residual Refinement

    [https://arxiv.org/abs/2611.27377](https://arxiv.org/abs/2611.27377)

    HR-Merge 是一种无需训练的通用模型合并增强框架，它根据各任务的验证准确率余量对其残差加权并重新合并，从而有效恢复现有合并方法在不同任务上不均匀损失的准确率。

    

    模型合并旨在将在共享主干上微调的多个专家模型（例如大语言模型的 LoRA 适配器）合并为一个多任务模型，而无需进一步训练。现有方法对每个任务采用同一合并规则，但由此造成的准确率损失在各任务之间并不均匀。我们证明，这些损失的准确率仍可从每个任务的残差（即其任务向量减去合并后的更新）中恢复，并且任务的验证余量（即合并所损失的专家验证准确率的占比）能够指示在何处进行恢复最为有效。由于来自不同任务的残差会相互干扰，因此必须对它们进行合并而非简单求和。我们提出 HR-Merge，这是一个无需训练的框架，可包裹任意现有的合并方法。它按各任务的余量对残差进行加权，用基础合并算子重新合并加权后的残差，并以在验证集上选定的强度将结果加回；当没有任何强度带来提升时，则回退到基础合并方法。在 LLaMA-3 8B 的六个 NLI 适配器上，HR-Merge 提升了……（原文摘要在此截断）

    arXiv:2611.27377v1 Announce Type: cross  Abstract: Model merging combines experts fine-tuned from a shared backbone, such as LoRA adapters of a large language model, into one multi-task model without further training. Existing methods apply one rule to every task, yet the accuracy they lose is uneven across tasks. We show that the lost accuracy remains recoverable from each task's residual, its task vector minus the merged update, and that a task's validation headroom, the share of its expert validation accuracy the merge loses, indicates where recovery pays off. Residuals from different tasks interfere, so they must be merged rather than summed. We propose HR-Merge, a training-free wrapper around any merging method. It weights each residual by its task's headroom, re-merges the weighted residuals with the base operator, and adds the result back at a strength selected on validation, falling back to the base merge when no strength helps. On six NLI adapters of LLaMA-3 8B, HR-Merge lifts
    
[^30]: RL专家的可组合软提示蒸馏：迈向持续后训练

    Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training

    [https://arxiv.org/abs/2611.27376](https://arxiv.org/abs/2611.27376)

    提出可组合软提示蒸馏（CSPD）方法，将每个RL专家独立蒸馏为软提示并通过拼接方式组合，无需重新整合模型即可实现持续后训练，性能媲美多教师蒸馏且不损害目标领域之外的能力。

    

    RL后训练通常会产生领域专家模型，这些专家随后必须被整合到一个模型中，而随着新专家不断出现，这种整合就成为了一个持续的过程。权重合并和多教师在线策略蒸馏（MOPD）在每添加一个专家时都必须重新整合整个模型，并且可能损害目标领域之外的能力。我们提出了可组合软提示蒸馏（CSPD），该方法保持RL之前的模型冻结，将每个RL专家独立地蒸馏到一个软提示中，并通过直接拼接各提示来组合专家，无需联合训练。单个软提示就足以蒸馏一个RL专家，在23个数学、代码和智能体设置中的22个里恢复了至少90%的教师增益。独立训练的提示之间可以良好组合。在多专家设置中，平均性能随着每个专家的加入而提升，完整组合的性能与MOPD相当（67.83 vs. 67.79）。该组合在六个……（摘要截断）

    arXiv:2611.27376v1 Announce Type: cross  Abstract: RL post-training typically produces domain experts that must then be consolidated into one model, and as new experts keep arriving, this consolidation becomes a continual process. Weight merging and multi-teacher on-policy distillation (MOPD) must re-consolidate the whole model whenever an expert is added, and can erode capabilities outside the target domains. We propose Composable Soft-Prompt Distillation (CSPD), which keeps the pre-RL model frozen, distills each RL expert independently into a soft prompt, and composes experts by concatenating their prompts without joint training. A single soft prompt suffices to distill an RL expert, recovering at least 90% of the teacher's gain in 22 of 23 math, code, and agent settings. Independently trained prompts compose well. In a multi-expert setting, the average rises with every expert added, and the full composition matches MOPD (67.83 vs. 67.79). The composition generalizes best across six 
    
[^31]: AMARIS：通过自适应子空间继承融合通用与专家大语言模型

    AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance

    [https://arxiv.org/abs/2611.27369](https://arxiv.org/abs/2611.27369)

    AMARIS 提出了一种协方差驱动的免训练模型合并框架，通过在正则化扰动预算下选择高效用、低干扰的激活子空间，将专家模型能力以单一静态检查点的形式无训练地继承到通用大语言模型中。

    

    大语言模型（LLM）的领域专业化通常能提升目标领域的性能，但会损害其广泛能力。模型合并是一种无需训练的有吸引力的替代方案，然而常见的参数空间策略（例如插值和稀疏化平均）对任务特定干扰的控制能力有限。我们提出了通过排序继承子空间实现自适应模型对齐（AMARIS），这是一个由协方差驱动的框架，仅沿选定的激活方向传递专家模型的更新。AMARIS 将合并表述为预算化的子空间继承：在正则化的基础输出扰动预算下，寻求对专家激活的最大捕获，然后从排序后的广义特征方向构建低秩投影门控。由此得到一个单一的静态合并检查点，将知识迁移集中在高效用、低干扰的方向上。我们在基于 Qwen3 的合并模型（8B 和 32B）上评估了 AMARIS，涵盖金融、保险等领域（原文摘要此处被截断）。

    arXiv:2611.27369v1 Announce Type: new  Abstract: Domain specialization of large language models (LLMs) often improves target-domain performance but degrades broad capabilities. Model merging is an attractive training-free alternative, yet common parameter-space strategies (for example, interpolation and sparsified averaging) provide limited control over task-specific interference. We introduce Adaptive Model Alignment via Ranked Inherited Subspaces (AMARIS), a covariance-driven framework that transfers specialist updates only along selected activation directions. AMARIS formulates merging as budgeted subspace inheritance: it seeks high specialist activation capture under a regularized base-output perturbation budget, then builds a low-rank projection gate from ranked generalized eigendirections. This yields a single static merged checkpoint that concentrates transfer in high-utility, low-interference directions. We evaluate AMARIS on Qwen3-based merges (8B and 32B) across finance, inst
    
[^32]: WaXup：专家模型权重平均作为数据增强以实现鲁棒的内省

    WaXup: Weight Averaging of Experts as Data Augmentation for Robust Introspection

    [https://arxiv.org/abs/2611.27353](https://arxiv.org/abs/2611.27353)

    本文提出WaXup方法，通过在权重空间中对专家模型进行线性插值（权重平均）来增强内省训练数据，从而显著提升大模型可训练内省方法的鲁棒性与泛化能力。

    

    随着大型语言模型能力的不断提升，传统的机制可解释性方法难以跟上其发展步伐。相比之下，可训练的内省方法作为一种有前景的替代方案应运而生；它将模型本身重新用作自我诊断工具，通过训练使模型能够用自然语言表达其在微调过程中学到的内容。虽然在微调专家模型的数据集上进行训练取得了令人鼓舞的结果，但目前这种方法缺乏鲁棒性和泛化能力。基于MiXup的数据增强框架，我们提出了WaXup，这是一种通过权重空间中的线性插值来增强内省训练集的方法。这一方法的理论依据是线性模式连通性特性：权重平均后的专家模型能够插值初始专家模型的行为。我们在三个基准上评估了WaXup：隐藏行为模型、真实世界NLP任务的元数据集，以及AuditBench——一个具有挑战性的对齐审计套件。

    arXiv:2611.27353v1 Announce Type: cross  Abstract: As the capabilities of large language models keep improving, traditional mechanistic interpretability methods struggle to keep pace. In contrast, trainable introspection methods emerge as a promising alternative; they repurpose the model itself as a self-diagnostic tool, trained to verbalize what it has learned during finetuning in natural language. While training on datasets of finetuned experts achieves encouraging results, it currently lacks robustness and generalization. Building on the data augmentation framework of MiXup, we introduce WaXup, a method that augments the introspection training set through linear interpolation in the weight space. This is motivated by the linear mode connectivity property: weight averaged experts interpolate the behaviors of the initial experts. We evaluate WaXup on three benchmarks: hidden-behavior models, a meta-dataset of real-world NLP tasks and AuditBench, a challenging alignment auditing suite 
    
[^33]: 突破长时程：面向大语言模型长文本生成的短时程过程监督

    Breaking the Long Horizon: Short-Horizon Process Supervision for Long-Form Generation in LLMs

    [https://arxiv.org/abs/2611.27348](https://arxiv.org/abs/2611.27348)

    该论文提出一种短时程过程监督框架，将长文本生成分解为片段级决策，并利用基于评分标准的局部奖励计算逐步优势、实现局部化策略更新，从而解决传统结果级奖励监督延迟且粗粒度、难以定位错误和分配信用的问题。

    

    长文本生成对智能体化的大语言模型系统日益重要，然而当前的强化学习方法通常使用单一的结果级奖励来优化整个文档。这种延迟且粗粒度的监督使得难以定位错误，并难以在输出中语义不同的部分之间进行信用分配。为解决这一问题，我们提出了一种短时程框架，通过将长文本生成语义分解为片段级决策来重新构建该任务。每个片段都在前缀条件化的上下文中生成，从而在保持监督局部化的同时保留跨片段的连贯性。生成的片段随后依据依从性、连贯性和相关性使用基于评分标准的奖励进行评分，从而在片段级别实现显式的局部监督。这些局部奖励被用于计算逐步优势，进而产生局部化的策略更新。为隔离监督粒度的影响，我们设计……

    arXiv:2611.27348v1 Announce Type: new  Abstract: Long-form generation is increasingly important for agentic LLM systems, yet current reinforcement learning methods typically optimize an entire document with a single outcome-level reward. This delayed and coarse-grained supervision makes it difficult to localize errors and assign credit across semantically distinct parts of the output. To address this, we propose a short-horizon framework that reformulates long-form generation through semantic decomposition into segment-level decisions. Each segment is generated with prefix-conditioned context, allowing cross-segment coherence to be preserved while keeping supervision local. The generated segments are then scored with rubric-based rewards for adherence, coherence, and relevance, enabling explicit local supervision at the segment level. These local rewards are used to compute step-wise advantages, yielding localized policy updates. To isolate the effect of supervision granularity, we ins
    
[^34]: 用更少词元进行推理：概念归一化的多语言草稿纸与可逆的模型自适应推理编码

    Reasoning in Fewer Tokens: Concept-Normalized Multilingual Scratchpads and Reversible Model-Adaptive Reasoning Codes

    [https://arxiv.org/abs/2611.27326](https://arxiv.org/abs/2611.27326)

    该论文提出了一种无损的中间推理压缩框架，通过BPE学习由PUA词元构成的压缩词表并扩展LLM词表后微调模型，在保持可解释性和可审计性的同时，平均减少14.7%的生成词元，推理轨迹长度最多缩短27.1%。

    

    思维链提示显著提升了大语言模型（LLM）的推理能力，但由于自回归解码过程冗长，带来了高昂的推理成本。以往的效率优化方法通过丢弃词元或使用连续的潜在表示来降低成本，然而这些有损方法从根本上牺牲了可解释性和可审计性。在本工作中，我们提出了一种新颖的中间推理压缩框架。我们对推理语料应用字节对编码（BPE）来学习一个由私有使用区（PUA）词元组成的压缩词表，其中每个词元代表频繁出现的语义片段。在扩展LLM词表后，我们通过严格的一对一映射对模型进行微调，因此我们的方法是无损的。我们在Llama-3.1-8B、Qwen2.5-14B和Qwen2.5-32B上，跨越四个不同的基准测试对该方法进行了评估。结果表明，我们的方法平均减少了14.7%的生成词元总数，推理轨迹长度最多减少27.1%。

    arXiv:2611.27326v1 Announce Type: cross  Abstract: Chain-of-thought prompting significantly improves the reasoning capabilities of large language models (LLMs) but incurs high inference costs due to lengthy autoregressive decoding. While previous efficiency approaches reduce costs by discarding tokens or utilizing continuous latent representations, these lossy methods fundamentally sacrifice interpretability and auditability. In this work, we introduce a novel compression framework for intermediate reasoning. We apply Byte-Pair Encoding (BPE) to reasoning corpora to learn a compressed vocabulary of Private Use Area (PUA) tokens, each representing frequent semantic spans. After expanding the LLM's vocabulary, we fine-tune the model via strictly 1:1, our method is lossless. We evaluate our approach on Llama-3.1-8B, Qwen2.5-14B, and Qwen2.5-32B across four diverse benchmarks. Our method reduces total generated tokens by 14.7% on average with reasoning trace lengths dropping by up to 27.1%
    
[^35]: SAVIOR：面向OCR表示的视觉-语言模型样本高效自适应方法

    SAVIOR: Sample-efficient Adaptation of Vision-Language Models for OCR Representation

    [https://arxiv.org/abs/2611.27319](https://arxiv.org/abs/2611.27319)

    提出样本高效的数据整理方法SAVIOR和结构感知评估指标PaIRS，通过微调Qwen2.5-VL使视觉语言模型在金融OCR的关键失败场景（如竖排/旋转文本、徽标嵌入名称、退化扫描件和多栏布局）上实现0.93的词召回率。

    

    光学字符识别（OCR）流程与视觉-语言模型在标准OCR和文档理解基准测试中表现出较高的准确率，但在金融工作流程中具有关键业务意义的文档模式上却系统性地表现不佳。这些模式包括竖排或旋转文本、嵌入徽标的供应商名称、小字号法规条款、退化的扫描件以及复杂的多栏布局。尽管这些模式在公开数据集中代表性不足，但它们在实际应用失败案例中占据了相当大的比例。我们提出SAVIOR，一种针对此类高影响失败场景的样本高效数据整理方法，用于使视觉-语言模型适应稳健的金融OCR。我们进一步提出PaIRS，一种结构感知的评估指标，通过比较词符之间的成对空间关系来衡量布局保真度。使用SAVIOR-Train对Qwen2.5-VL-Instruct进行微调，达到了0.93的词召回率。（注：原始摘要在此处被截断）

    arXiv:2611.27319v1 Announce Type: cross  Abstract: Optical Character Recognition (OCR) pipelines and vision-language models exhibit high accuracy on standard OCR and document-understanding benchmarks, yet they systematically underperform on document patterns that are operationally critical in financial workflows. These patterns include vertical or rotated text, logo-embedded vendor names, fine-print regulatory clauses, degraded scans, and complex multi-column layouts. While underrepresented in public datasets, they constitute a substantial portion of real-world failure cases. We introduce SAVIOR, a sample-efficient data curation methodology that targets such high-impact failure scenarios to adapt vision-language models for robust financial OCR. We further propose PaIRS, a structure-aware evaluation metric that measures layout fidelity by comparing pairwise spatial relationships between tokens. Fine-tuning Qwen2.5-VL-Instruct with SAVIOR-Train achieves a word recall of 0.93. When the re
    
[^36]: 语言模型知道自己推理被哪些词引导吗？

    Do Language Models Know Which Words Steer Their Reasoning?

    [https://arxiv.org/abs/2611.27317](https://arxiv.org/abs/2611.27317)

    该论文发现了“幻影上下文影响”现象——语言模型的推理会被语义无关的词不成比例地引导，而模型自身却意识不到这种依赖关系（生成-判断矛盾），并提出“在线KV抑制”方法来缓解这一问题。

    

    虽然思维链提示能够引导语言模型产生中间推理步骤，但目前尚不清楚上下文中哪些词真正引导了思维链。我们识别出了“幻影上下文影响”现象：语言模型会被语义无关的上下文词不成比例地引导，即使同一模型在被直接询问时会否认这些词的相关性。为了定位这一现象，我们引入了“留一词JSD分析”，该方法从原本完整的上下文中每次移除一个词，以隔离该词对下一个词元分布的边际影响；移除后会导致预测翻转的词（枢轴词）会被传递给同一个大语言模型，让其自我判断是否识别到这种预测依赖关系。在四个模型和三个基准上的实验表明，语言模型在很大程度上并不知道哪些词引导了它们的推理，我们将这种不匹配称为“生成-判断矛盾”。受此现象启发，我们提出了“在线KV抑制”方法……

    arXiv:2611.27317v1 Announce Type: cross  Abstract: While chain-of-thought (CoT) prompting elicits intermediate reasoning steps from language models, it is unclear which words in the context actually steer the CoT. We identify Phantom Context Influence (PCI): language models are disproportionately steered by semantically irrelevant context words, even as the same models deny those words' relevance when asked. To localize PCI, we introduce Leave-One-Out Word JSD analysis, which removes one context word at a time from the otherwise intact context to isolate its marginal effect on the next-token distribution; words whose removal flips the prediction (pivot words) are passed to the same LLM to self-judge whether it recognizes this predictive dependence. Across four models and three benchmarks, language models largely do not know which words steer their reasoning, a mismatch we call the Generation-Judgment Contradiction. Inspired by this phenomenon, we propose Online KV Suppression: it score
    
[^37]: 观察到的、整理的、复用的：临床AI评估中的观察偏差

    Observed, Curated, Reused: Observation Bias in Clinical AI Evaluation

    [https://arxiv.org/abs/2611.27301](https://arxiv.org/abs/2611.27301)

    该研究揭示了临床AI评估中因医疗文档仅记录医生所见而产生的观察偏差，并通过追踪6,503个病例在HPO数据库六个历史版本中的引用记录首次量化了这一偏差，同时发现了证据同时进入基准与知识库所导致的自我指涉循环问题。

    

    临床AI评估容易受到嵌入在常规医疗文档中的观察偏差的影响，这些文档仅记录临床医生注意到、检测和记录的内容，而非完整的临床图景。当这些证据进入出版物和参考知识图谱时，关键的区分便会消失：未经检查的症状与确认的阴性结果混为一谈，疾病的时间演变过程也会丢失。幸存的证据还可能同时进入基准查询和参考知识库，形成一种自我指涉的循环，即患者是根据从其自身已发表病例报告中提取的线索来进行评估的。为了量化这种偏差，我们追踪了6,503个临床病例在六个历史版本的人类表型本体（HPO）发布版本（2021年10月至2026年6月）中的情况，以确定HPO首次引用描述这些病例的出版物的时间。由此得到了一个具有可观察的“源摄入”事件的验证性队列，其中HPO引用了病例的源出版物……

    arXiv:2611.27301v1 Announce Type: new  Abstract: Clinical AI evaluations are vulnerable to observation bias embedded in routine medical documentation, which captures only what clinicians notice, test, and record rather than a complete clinical picture. As this evidence travels into publications and reference knowledge graphs, crucial distinctions disappear: unexamined symptoms blur with confirmed negatives, and chronological disease progression is lost. Surviving evidence can also enter both the benchmark query and the reference knowledge base, creating a self-referential loop where a patient is evaluated against clues derived from their own published case report.   To quantify this bias, we tracked 6,503 clinical cases across six historical Human Phenotype Ontology (HPO) releases (October 2021-June 2026) to identify when HPO first cited the publications describing them. This yielded a confirmatory cohort with an observable "source ingestion" event, where HPO cited a case's source publ
    
[^38]: 信念状态管理与评分标准奖励相结合的智能体预测

    Belief-State Management Meets Rubric Rewards for Agentic Forecasting

    [https://arxiv.org/abs/2611.27286](https://arxiv.org/abs/2611.27286)

    该论文提出一个智能体事件预测训练框架，通过主动信念状态管理（PBM）将冗长上下文压缩为紧凑信念状态，并引入五维不依赖结果的SEIPH评分标准提供过程质量奖励，从而解决了LLM智能体预测中上下文累积和奖励信号嘈杂的两大难题。

    

    当前的大语言模型（LLM）智能体在预测现实世界事件方面仍然存在困难。这源于两个主要局限性：i) 标准的ReAct框架在一个不断增长的上下文中积累推理和证据；ii) 基于结果的预测奖励产生缓慢且嘈杂的信号，无法区分合理的推理与幸运的猜测。为了解决这些问题，我们引入了一个预测训练框架，包括框架设计和后训练两部分。在框架方面，我们提出了主动信念状态管理（PBM），为智能体配备一个折叠工具，将冗长的上下文压缩为紧凑的信念状态，并重置上下文以便后续探索。在训练方面，我们开发了SEIPH，这是一个五维的、不依赖结果的评分标准，用于提供过程质量奖励。使用基于评分标准的奖励，我们采用PPO和GRPO对一个35B模型进行后训练，得到了DeepForecaster-PPO和DeepForecaster-GRPO。理论上，我们证明……

    arXiv:2611.27286v1 Announce Type: new  Abstract: Current large language model (LLM) agents still struggle to forecast real-world events. This stems from two primary limitations: i) standard ReAct harness accumulates reasoning and evidence within one growing context, and ii) outcome-based rewards in forecasting lead to a slow and noisy signal that cannot distinguish sound reasoning from lucky guesses. To address these, we introduce a forecasting training framework including harness design and post-training. For the harness side, we propose Proactive Belief-State Management (PBM), equipping agents with a fold tool to compress lengthy context into compact belief states and reset the context for subsequent exploration. For the training side, we develop SEIPH, a five-dimensional, outcome-blind rubric, providing process-quality rewards. Using the rubric-based reward, we post-train a 35B model with PPO and GRPO, yielding DeepForecaster-PPO and DeepForecaster-GRPO. Theoretically, we prove that
    
[^39]: 公司能否兑现其IPO承诺？

    Do Companies Deliver on Their IPO Claims?

    [https://arxiv.org/abs/2611.27273](https://arxiv.org/abs/2611.27273)

    该论文提出了Vision Follow-Through（VFT）基准，用于评估公司在IPO招股书中做出的前瞻性声明是否真正兑现，发现模型在从完整招股书中提取声明以及评估声明兑现情况方面仍面临显著挑战。

    

    首次公开募股（IPO）招股书描述了一家公司的业务内容及其未来愿景。然而，评估单个前瞻性声明是否在随后得到实现是困难的。我们提出了 Vision Follow-Through（VFT），这是一个基准测试和智能体流程，将声明提取、类别分配、可验证性判断以及基于证据的兑现评估分离开来。\vftclaims{} 包含来自50份IPO招股书的5,066个人工标注的声明片段，而 \vftverify{} 为其中1,000个声明添加了可验证性层级、兑现结果以及引用的后续证据。在各类语言模型上，对人工标注的声明进行分类远比从完整招股书中提取声明容易。提取覆盖率也因类别而显著不同，组织/文化类声明的提取率远低于市场机会类声明。兑现评估则构成另一个独立的挑战：模型的判断与人类评估存在分歧……

    arXiv:2611.27273v1 Announce Type: new  Abstract: Initial public offering (IPO) filings describe what a company does and what it intends to become. Yet whether individual forward-looking claims are subsequently realized is difficult to assess. We introduce Vision Follow-Through (VFT), a benchmark and agentic pipeline that separates claim extraction, category assignment, verifiability, and evidence-grounded follow-through assessment. \vftclaims{} contains 5,066 human-annotated claim spans from 50 IPO filings, while \vftverify{} adds verifiability tiers, follow-through outcomes, and cited later evidence for 1,000 claims. Across language models, classifying human-annotated claims is substantially easier than extracting claims from full filings. Extraction coverage also varies sharply by category, with Organizational/Culture claims recovered far less often than Market Opportunity claims. Follow-through assessment poses a separate challenge: model judgments diverge from human assessments in 
    
[^40]: LongCat召回稀疏注意力：面向长上下文推理的键组织与检索协同设计

    LongCat Recall Sparse Attention: Co-Designing Key Organization and Retrieval for Long-Context Inference

    [https://arxiv.org/abs/2611.27233](https://arxiv.org/abs/2611.27233)

    提出LongCat召回稀疏注意力（LRSA），通过将细粒度代表与增量维护索引中的分页检索相结合，消除长上下文推理中检索整个历史记录的瓶颈。

    

    以深度求索稀疏注意力（DSA）为代表的基于检索的稀疏注意力，通过选择一小部分键值条目来减少注意力计算量。然而，选择这些条目仍然可能需要搜索全部可见历史。对于长序列而言，检索成为主要瓶颈，限制了稀疏注意力带来的效率提升。为解决这一问题，我们提出了LongCat召回稀疏注意力（LRSA），它在增量维护的索引中将细粒度代表与分页检索相结合。LRSA将序列划分为多个时间桶，随着新键的到来，各时间桶可以独立索引，而无需对已有历史进行重新分组。在每个桶内，该方法将索引键组织为容量平衡的方向组，并将固定数量的完整组打包进每一页，同时为每个组保留一个独立的中心。这些中心为页面选择提供细粒度的代表……

    arXiv:2611.27233v1 Announce Type: new  Abstract: Retrieval-based sparse attention, exemplified by DeepSeek Sparse Attention (DSA), reduces attention computation by selecting a small subset of key-value entries. However, selecting these entries can still require searching the entire visible history. For long sequences, retrieval becomes a major bottleneck, limiting the efficiency gains from sparse attention. To address this problem, we propose LongCat Recall Sparse Attention (LRSA), which combines fine-grained representatives with paged retrieval in an incrementally maintained index. LRSA partitions the sequence into time buckets that can be indexed independently as new keys arrive, without regrouping existing history. Within each bucket, it organizes indexing keys into capacity-balanced directional groups and packs a fixed number of complete groups into each page, retaining a separate center for every group. These centers provide fine-grained representatives for page selection, after w
    
[^41]: CoPyro：将复制提议馈送至并行起草器

    CoPyro: Feeding Copy Proposals to Parallel Drafters

    [https://arxiv.org/abs/2611.27217](https://arxiv.org/abs/2611.27217)

    CoPyro 将 n-gram 检索复制得到的 token 取代掩码馈入并行起草器，使其在单次前向传播中学会保留或修改这些 token，从而在 Qwen3-4B 的八个基准上为所有测试的并行起草器显著提升推测解码的接受长度。

    

    推测解码通过让目标模型在单次前向传播中验证多个草稿 token 来加速语言生成。早期的神经起草器（如 EAGLE）每次将候选序列扩展一个 token；近期的研究转向并行起草器（如 DFlash），它们可以一次性预测整个 token 块。另一种方法则利用 n-gram 检索，查找当前上下文后缀此前出现的位置，并将其后跟随的 token 直接复制到草稿中。我们提出 CoPyro，将这些复制的 token 取代掩码馈送至并行起草器，起草器在常规的单次前向传播中预测块的同时，从上下文中学习哪些 token 应保留、哪些应修改。在 Qwen3-4B 的八个基准测试上，CoPyro 提升了所测试的全部四个并行起草器的整体接受长度，在 10 万样本训练下增益为 2.19%–5.08%；使用在 71 万样本上训练的 DFlash 时，整体增益范围为 1.08% 到 2.00%。

    arXiv:2611.27217v1 Announce Type: new  Abstract: Speculative decoding accelerates language generation by letting a target model verify several draft tokens in one forward pass. Earlier neural drafters such as EAGLE extend each candidate sequence one token at a time. Recent work has turned to parallel drafters such as DFlash, which predict a whole block at once. Another approach uses n-gram retrieval to find an earlier occurrence of the current context suffix and copy the tokens that followed it directly into the draft. We introduce CoPyro, which feeds these copied tokens into a parallel drafter in place of masks. The drafter learns from context which tokens to keep and which to revise as it predicts the block in its usual forward pass. Across eight benchmarks on Qwen3-4B, CoPyro improves overall acceptance length for all four parallel drafters tested, with gains of 2.19-5.08% when trained on 100K samples. With DFlash trained on 710K samples, overall gains range from 1.08% to 2.00% acro
    
[^42]: 正确先例诱发端侧LLM代理的策略违规

    Correct Precedents Induce Policy Violations in On-Device LLM Agents

    [https://arxiv.org/abs/2611.27216](https://arxiv.org/abs/2611.27216)

    该论文首次发现“正确先例泛化”这一新型安全故障——端侧LLM代理在正确放行一个解除限制的条目后，会将该决策泛化迁移到仍受限制的后续条目上，导致即使整个会话历史中没有任何错误步骤也可能以策略违规告终，该问题在来自九个开发者的13个开源模型中普遍存在，且单个正确先例即可将最严重模型的安全率削减约一半。

    

    本地-云端代理系统将云端模型与一个小型的设备端代理配对，该代理对私有数据进行操作，通常作为不可逆操作执行前的最后一道检查。我们首次识别出一种故障模式：此类代理会在会话内被其自身的正确决策引导至违规。在正确放行了一个限制已被解除的条目后，该代理更有可能放行下一个仍受限制的条目。由于触发因素是一个正确的决策，即使历史记录中没有任何错误步骤，会话仍可能以违规告终。我们将这种故障称为“正确先例泛化”，并在来自九个开发者的13个开源权重模型中发现了该问题。单个正确先例可将受影响最严重的代理的安全率大约削减一半，且当被拒绝的请求在新会话中被简单重发时，风险会进一步累积。对代理历史记录的干预实验表明，代理会将其对已放行条目的决策迁移到受限制条目上。

    arXiv:2611.27216v1 Announce Type: cross  Abstract: Local-cloud agent systems pair a cloud model with a small on-device agent that acts on private data, often as the last check before an irreversible action. We identify, for the first time, a failure in which such an agent is led into a violation by its own correct decisions within a session. After it correctly releases an item whose restriction was lifted, it becomes more likely to release the next item that is still restricted. Because the trigger is a correct decision, a history with no faulty step can still end in a violation. We call this failure correct-precedent generalization and find it in 13 open-weight models from nine developers. A single correct precedent can cut the safe rate of the most affected agent roughly in half, and the risk compounds when a refused request is simply sent again in a new session. Interventions on the agent's history indicate that it carries its decision on the released item over to the restricted one
    
[^43]: 面向个性化长文本生成的显式风格抽取

    Explicit Style Extraction for Personalized Long-Form Text Generation

    [https://arxiv.org/abs/2611.27211](https://arxiv.org/abs/2611.27211)

    该论文提出StylePrism风格分类体系（28个风格维度）与SP-Refine推理时方法，将风格抽取变成显式步骤，在LongLaMP三个长文本任务上以平均10.2%的提升显著超越检索增强基线。

    

    个性化长文本生成要求语言模型以特定用户的方式来撰写新文本。标准方法是检索增强生成，即将用户的部分过往文本放入提示中，但让模型自行决定从中借鉴什么。我们提出了StylePrism，这是一个包含七大类别、共28个风格维度的分类体系，同时提出SP-Refine，一种在推理阶段将风格抽取作为显式步骤的方法：模型首先用用户的写作模式、引用证据和需要避免的习惯来填充StylePrism，然后基于用户的样本起草文本，并在自身风格评分的引导下，依据该风格画像对草稿进行两次修订。在使用三个大语言模型的LongLaMP三个长文本任务上，SP-Refine在27项ROUGE和METEOR对比中的25项上超越了LongLaMP的检索增强基线，平均提升10.2%，最高提升达35.8%。用提取的风格画像对用户自己的文本进行标注表明，StylePrism捕捉到了具体的、可核查的……

    arXiv:2611.27211v1 Announce Type: cross  Abstract: Personalized long-form generation asks a language model to write new text the way a specific user would. The standard recipe, retrieval-augmented generation, places a few of the user's past texts in the prompt but leaves the model to decide what to take from them. We introduce StylePrism, a taxonomy of 28 stylistic aspects in seven categories, and SP-Refine, an inference-time method that makes style extraction an explicit step: the model first fills StylePrism with the user's patterns, quoted evidence, and habits to avoid, then drafts from the user's samples and revises the draft twice against this profile, guided by its own style scores. On the three long-form tasks of LongLaMP with three LLMs, SP-Refine outperforms LongLaMP's retrieval-augmented baseline in 25 of 27 ROUGE and METEOR comparisons, by 10.2% on average and by up to 35.8%. Annotating users' own texts with the extracted profiles shows that StylePrism captures concrete, che
    
[^44]: 计算错误率中有多少并非算术错误：用确定性规则系统审计 FinReason Cup 2026 任务 3 的评测指标

    How Much of a Calculation-Error Rate Is Not Arithmetic: Auditing the FinReason Cup 2026 Task 3 Metric with a Deterministic Rule System

    [https://arxiv.org/abs/2611.27192](https://arxiv.org/abs/2611.27192)

    本文用确定性规则系统审计 FinReason Cup 2026 任务 3 的评测指标，证明其“计算错误率”主要由非算术错误构成：23 个上榜参赛者中有 22 个存在非算术错误下界为正，且前两名的计算错误中至少分别有 11/12 和 12/13 并非真正的求和错误。

    

    FinReason Cup 2026 的任务 3 对系统在两种数值上的表现进行评分：一种是 XBRL 财务文件所报告的数值，另一种是其自身计算关系所隐含的数值，并将每次失败标记为结构性、抽取性或计算性错误。本报告表明，如此定义的“计算错误率”并不统计真正的算术失败，并仅利用公开发布的榜单界定了它与算术错误率之间的差距。基准背后的三条规则之一 DQC_US_0015，其答案恰好是所报告数值的相反数（在全部 110 个公开练习案例中均是如此，而另外两条规则分别只有 120 个中的 1 个和 102 个中的 6 个），因此在这种案例上被打上“计算”标签，记录的是错误的事实或错误的推理路径，而非错误的求和。将每个参赛者在该规则上的错误计数中减去可用的结构性和抽取性标记，即可得到一个下界：对于 23 个上榜参赛者中的 22 个，该下界为正；而在榜单顶端该结论最为尖锐，前两名参赛者被标记的计算错误中至少有 12 个中的 11 个和 13 个中的 12 个并非算术错误。

    arXiv:2611.27192v1 Announce Type: new  Abstract: Task 3 of the FinReason Cup 2026 scores a system on two values per case, the one an XBRL filing reports and the one its own calculation relationships imply, and labels each failure as structural, extraction or calculation. This report shows that the calculation-error rate so defined does not count arithmetic failures, and bounds how far it is from one using the published board alone. One of the three rules behind the benchmark, DQC_US_0015, has an answer that is the negation of the reported value (in all 110 published practice cases, against 1 of 120 and 6 of 102 for the other two), so a calculation label on such a case records a wrong fact or a wrong route, not a wrong sum. Subtracting the available structural and extraction labels from each entry's error count on that rule gives a floor: the bound is positive for 22 of the 23 ranked entries, and at the top it is sharpest, with at least 11 of 12 and 12 of 13 calculation errors of the tw
    
[^45]: 匹配人类干预效应能确立什么？LLM模拟的基准检验

    What Does Matching a Human Intervention Effect Establish? Benchmark Checks for LLM Simulations

    [https://arxiv.org/abs/2611.27186](https://arxiv.org/abs/2611.27186)

    该论文质疑LLM模拟匹配人类干预效应所能确立的有效性，发现大量“通过”实为无信息量的零参考通过，并提出零参考基准与计划精度检验来严格评估LLM模拟的真实响应能力。

    

    LLM模拟通过与人效应的等效性来验证，然而只要人类90%区间位于容差范围内，零参考点就会自动通过——正如人类基准在模拟之前所显示的那样。当零参考点失败时，计划的精度赋予独立模拟一个容差上限，在此范围内通过构成对响应的5%显著性检验。在一项包含70个实验的探索性再分析中，GPT-4的±0.20 SD通过中有72%属于零参考通过；GPT-4通过了16个零参考未通过的对比（在无响应情况下预期仅为1.58个）。在一项预注册审计中，Claude Sonnet 5.5通过了三个意图目标，全部为零参考通过，同时还通过了注册的信念梯度控制；规范信念仅变化0.065 SD，而人类为0.351 SD（零参考检验在揭盲后添加；规范信念在检查后予以强调）。注册检验未检测到高估；平均效应区间排除了超过0.08 SD的高估。

    arXiv:2611.27186v1 Announce Type: new  Abstract: LLM simulations are validated by equivalence with human effects, yet a zero reference passes whenever the human 90 percent interval lies inside the margin, as the human benchmark shows before simulation. When the zero reference fails, planned precision gives independent simulations the margin up to which passing is a 5 percent test of response. In an exploratory 70-experiment reanalysis, 72 percent of GPT-4's $\pm$0.20 SD passes were zero-reference passes; GPT-4 passed 16 contrasts the zero reference failed (1.58 expected without response). In a preregistered audit, Claude Sonnet 5.5 passed three intention targets, all zero-reference passes, and the registered belief-gradient control; norm beliefs changed 0.065 SD against humans' 0.351 (zero-reference check added after unblinding; norm beliefs emphasized after inspection). Registered tests detected no overestimation; average-effect intervals excluded overestimation above 0.08 SD. A group
    
[^46]: 面向自适应信息行动的行为智能体与检测器协同演化方法

    Co-Evolving Behavioral Agents and Detectors for Adaptive Information Operations

    [https://arxiv.org/abs/2611.27156](https://arxiv.org/abs/2611.27156)

    该论文提出IO-AI协同演化框架，通过让基于真实信息行动行为构建、具备互动与规避能力的模拟智能体与检测器轮流对抗训练，使检测器对自适应规避智能体的漏检率降低了65%。

    

    国家支持的信息行动（IO）检测器通常在平台已经采取下架措施的行动数据上进行训练。即使存档数据中没有任何操作者试图规避检测，这些检测器在面对未见过的行为者时也会损失大量准确性。真实的信息操作者有两个目标：触达受众和避免被发现。我们提出了IO-AI，一个赋予模拟智能体这两种能力——互动参与和规避检测——的框架，并测试每种能力对检测效果的影响。这些智能体基于真实行动的发帖行为构建，可以为每条帖子选择情绪、类型和发布时机。随后，智能体与检测器轮流进行相互对抗训练。我们按照检测难度对训练后的智能体进行排序，并按从易到难的顺序训练检测器。在20个Twitter信息行动上，该检测器相比仅在存档行动数据上训练的检测器，将对自适应智能体的漏检率降低了65%……

    arXiv:2611.27156v1 Announce Type: cross  Abstract: Detectors of state-sponsored information operations (IOs) are trained on campaigns that platforms have already taken down. They lose much of their accuracy on an actor they have not seen, even though no operator in the archived data is trying to avoid them. A real operator wants two things, to reach an audience and to avoid being caught. We present IO-AI, a framework that gives simulated agents these two capabilities, engagement and evasion, and tests what each one does to detection. The agents are built from the posting behavior of real campaigns and choose the emotion, type, and timing of each post. An agent and a detector then train against each other in turn. We rank the trained agents by how hard they are to detect and train one detector on them from easiest to hardest. On 20 Twitter IO campaigns, this detector reduces missed detections of adaptive agents by 65% relative to a detector trained only on the archived campaigns, at a s
    
[^47]: 脆弱性窗口：量化奖励模型中的决策不稳定性

    The Vulnerability Window: Decision Instability in Quantized Reward Models

    [https://arxiv.org/abs/2611.27155](https://arxiv.org/abs/2611.27155)

    量化在总体基准准确率几乎不变的情况下仍会翻转1.1-12.3%的奖励模型成对偏好决策，其翻转率可用高斯近似Φ(-|m|/σ)刻画，且脆弱性窗口占比随模型规模从25%降至7%，揭示了小型裁判尤其脆弱的隐患。

    

    奖励模型和大语言模型裁判通常被量化到4比特，以便在为大量候选打分时保持成本可控，而常规的验证方式是确认基准测试准确率几乎不变。这种验证方式近乎盲目。在来自两个系列的八个奖励模型（0.6B-8B）、三个生成式裁判以及四种4比特量化方法上，量化会改写RewardBench上1.1-12.3%的成对偏好决策，而总体准确率的变化却小至0.03分（p = 1.00），这是因为量化所修复的错误与其引入的错误相互抵消。决策翻转集中在全精度边际相对于量化噪声尺度sigma较小时的位置：高斯近似Phi(-|m|/sigma)能够描述总体翻转率，但在所有30个标量设置中都会高估该比率，并且对生成式裁判失效；由此产生的脆弱性窗口的占比随模型规模的增大从25%降至7%，这正是小型裁判脆弱的原因。在良性最佳情况下，这种变动保持在噪声范围内……

    arXiv:2611.27155v1 Announce Type: cross  Abstract: Reward models and LLM judges are routinely quantized to 4 bits so that scoring many candidates stays affordable, and the usual check is that benchmark accuracy barely moves. That check is close to blind. Across eight reward models from two families (0.6B-8B), three generative judges and four 4-bit quantizers, quantization rewrites 1.1-12.3% of pairwise preference decisions on RewardBench while aggregate accuracy moves by as little as 0.03 points (p = 1.00), because the errors it fixes and introduces cancel. Flips concentrate where the full-precision margin is small relative to the quantization noise scale sigma: a Gaussian approximation Phi(-|m|/sigma) describes the aggregate rate, over-predicting it in all 30 scalar settings and failing for generative judges, and occupancy of the resulting vulnerability window falls from 25% to 7% with model size, which is what makes small judges fragile. The churn stays within noise under benign best
    
[^48]: 高级智能体尽到职责了吗？衡量LLM智能体使用下级材料时的角色完整性

    Do Senior Agents Do Their Job? Measuring Role Integrity in How LLM Agents Use Subordinate Material

    [https://arxiv.org/abs/2611.27139](https://arxiv.org/abs/2611.27139)

    该论文提出“角色完整性”概念并构建RI-Bench行为基准，借助同行评审、上诉裁决和议会调查三种制度层级的结构与公开记录，衡量多智能体LLM系统中高级智能体是否真正履行了权衡下级报告的职责，而非仅仅进行简单汇总。

    

    多智能体LLM系统日益将长任务分配给多个子智能体，再由高级智能体将其报告整合为最终答案。这种分工只有在每个智能体——尤其是高级智能体——履行其角色职责时才能奏效。然而，当前的评估大多只对最终答案进行打分，而一个流畅、引用规范的答案无法表明高级智能体是否真正权衡了其子智能体的报告，还是仅仅进行了简单汇总。为了评估高级智能体是否履行了其被赋予的职责，我们将“角色完整性”定义为智能体在处理其下级材料时对其功能职责的保持。为了衡量这一概念，我们提出了RI-Bench，这是一个基于三种制度层级的结构与公开记录的行为基准：同行评审、上诉裁决和议会调查。我们选择这些层级，是因为它们的规则明确赋予高级角色对下级材料的职责。

    arXiv:2611.27139v1 Announce Type: cross  Abstract: Multi-agent LLM systems increasingly divide long tasks among subagents whose reports a senior agent turns into the final answer. Such a division of labor pays off only if each agent, and above all the senior agent, carries out the duties of its role. Current evaluations, however, mostly score the final answer, and a fluent, well-cited answer cannot show whether the senior agent weighed its subagents' reports or merely aggregated them. To assess whether a senior agent meets its assigned responsibilities, we define role integrity as the preservation of an agent's functional responsibilities when it acts on its subordinates' material. To measure it, we introduce RI-Bench, a behavioral benchmark grounded in the structures and public records of three institutional hierarchies: peer review, appellate adjudication, and parliamentary inquiry. We choose these hierarchies because their rules explicitly assign the senior role a duty toward subord
    
[^49]: 大语言模型证据融合中的重复计数：精确后验将规则遵从性与后验增益区分开来

    Double counting in LLM evidence fusion: exact posteriors separate rule compliance from posterior gain

    [https://arxiv.org/abs/2611.27138](https://arxiv.org/abs/2611.27138)

    本文构建了分级依赖条件下的精确后验参照基准，揭示出大语言模型对“每个来源只计一次”指令的遵从性与实际后验精度增益相互独立——完全遵守规则的模型可能损失精度，而不遵守规则的模型反而可能获得增益。

    

    语言模型在检索、多智能体共识和智能体记忆等场景中融合多方报告，若将来自共享来源的报告视为相互独立，就会导致证据被重复计数。一种常见的补救措施是指示模型对每个来源只计数一次。然而这一指令的价值取决于接收者的原始基线水平，要为其定价就需要一个精确的参照基准。本文构建了分级依赖性的精确参照：由单报告可靠性为0.65的二值世界和四个依赖级别构成，其公开的参数为每个世界提供了精确后验，并为每条规则设定了目标风险。配对增益协议在2,976个世界上对四种模型配置的原始回答与受指令回答进行了评分。结果显示，与规则的一致性和增益在两个方向上相互分离。在δ=0时，claude-opus-5-5和gpt-6.1-sol在全部744个世界中与规则一致（误差在0.001以内），但该指令反而使其后验失配增加了0.044；gpt-6-luna在744个世界中仅有165个与规则一致，却获得了0.0140的增益（95%置信区间为0.0064至0.0219）；claude-haiku-5-5的增益变化……（摘要截断）

    arXiv:2611.27138v1 Announce Type: new  Abstract: Language models fuse reports in retrieval, multi-agent consensus and agent memory, where treating shared-source reports as independent can double count evidence. A common remedy is an instruction to count each source once. Its value depends on the receiver's raw baseline. Pricing it needs an exact reference. We build the graded-dependence exact reference: binary worlds with single-report reliability 0.65 and four dependence levels, whose disclosed parameters give every world an exact posterior and every rule a target risk. The paired gain protocol scores raw and instructed answers of four configurations on 2,976 worlds. Agreement and gain separate in both directions. At $\delta$ = 0, claude-opus-5-5 and gpt-6.1-sol agreed with the rule within 0.001 in 744 of 744, and the instruction raised their posterior mismatch by 0.044; gpt-6-luna agreed in 165 of 744 and gained 0.0140 (95% interval 0.0064 to 0.0219); claude-haiku-5-5's gain moved fr
    
[^50]: 国家关联信息操作的跨平台生产特征

    Cross-Platform Production Signatures of State-Linked Information Operations

    [https://arxiv.org/abs/2611.27137](https://arxiv.org/abs/2611.27137)

    该论文提出“生产特征”概念——仅凭时间戳和粗粒度行为类型刻画信息操作账号的发帖规律，并证明这些行为规律可跨平台迁移，在无任何Reddit标签的情况下以0.90–0.94的AUC识别出IRA账号，显著优于会因平台切换而大幅退化的内容检测方法。

    

    活跃于多个平台的信息操作在每个平台上会呈现不同的名称、人设和话题，但其工作惯例却更难改变。我们将这种惯例称为该操作的“生产特征”：即其账号在何时以及如何发帖、回复和转发的规律性。我们仅通过时间戳和粗粒度的行为类型来度量这一特征，不使用任何内容、个人资料元数据或网络关系，具体做法是对单个账号采用时间与行为分布刻画，对账号群体采用学习得到的行为间隔策略刻画。在此基础上，我们探究特征中的哪些部分能够迁移到另一个平台，哪些只属于单个操作。我们在四个国家关联操作的Twitter档案上训练模型，并将其应用于Reddit上的“互联网研究局”（IRA）账号，全程未使用任何Reddit训练标签。模型对IRA账号的识别达到0.90–0.94的AUC，而走同样路线的内容检测器性能则从平台内的0.98降至0.81。日常节奏解释了大部分的迁移能力（摘要在此处被截断）。

    arXiv:2611.27137v1 Announce Type: new  Abstract: An information operation active on several platforms presents different names, personas, and topics on each. Its working routine is harder to change. We call this routine the operation's production signature: the regularities in when and how its accounts post, reply, and repost. We measure it from timestamps and coarse action types only, using no content, profile metadata, or network ties, as timing and action profiles for accounts and learned action-gap policies for populations. We then ask which parts of a signature carry over to another platform and which belong to a single operation. We train models on Twitter archives of four state-linked operations and apply them to Internet Research Agency (IRA) accounts on Reddit, with no Reddit training labels. They rank the IRA accounts at AUC 0.90-0.94. Content detectors on the same route fall from 0.98 within a platform to 0.81. Daily phase accounts for most of the transfer. Posting volume re
    
[^51]: 二维推理：量化思维链的忠实性与效率

    Reasoning in Two Dimensions: Quantifying Chain-of-Thought Faithfulness and Efficiency

    [https://arxiv.org/abs/2611.27100](https://arxiv.org/abs/2611.27100)

    该论文提出将思维链表示为语义推理图，并沿“语义支持对推理历史”与“通往答案对偏离路径”两个维度分解模型的预测依赖性，在同一框架下实现了对思维链非忠实性与低效性的全局及步骤级量化。

    

    思维链（CoT）推理会展示中间推理步骤，但答案正确并不能证明大语言模型确实依赖这些步骤所表达的依赖关系，也不能证明其推理是高效的。我们在一个统一的框架内对CoT的忠实性和效率进行了可操作化定义。具体而言，我们将思维链表示为一个语义推理图，并沿两个互补的维度分解模型的预测依赖性：由语义支持所解释的依赖与额外推理历史带来的依赖，以及分配给通往答案的推理与偏离路径推理的依赖。这一分解产生了全局和步骤级别的非忠实性与低效性度量。受控干预实验表明，所提出的指标能够按预期响应隐藏的依赖关系和诱导出的偏离路径推理。在数学和多跳推理任务中，答案级别的非忠实性始终与更大的预测不确定性相关，而非忠实性和……（摘要在此处截断）

    arXiv:2611.27100v1 Announce Type: new  Abstract: Chain-of-thought (CoT) reasoning exposes intermediate steps, but a correct answer does not establish that an LLM relies on the dependence expressed by those steps or reasons efficiently. We operationalize CoT faithfulness and efficiency within a common framework. Specifically, we represent a CoT as a semantic reasoning graph and decompose the model's predictive dependence along two complementary axes: dependence explained by semantic support versus additional reasoning history, and dependence allocated to answer-reaching versus off-path reasoning. This decomposition yields global and step-level measures of unfaithfulness and inefficiency. Controlled interventions show that the proposed metrics respond to hidden dependence and induced off-path reasoning as intended. Across mathematical and multi-hop reasoning tasks, answer-level unfaithfulness is consistently associated with greater predictive uncertainty, whereas both unfaithfulness and 
    
[^52]: 在AI辅助模型验证中区分自动化收益与智能体贡献

    Separating Automation Gains from Agent Contributions in AI-Assisted Model Validation

    [https://arxiv.org/abs/2611.27094](https://arxiv.org/abs/2611.27094)

    提出一种以无LLM可控工作流为对照臂的分阶段架构与归因协议，只在智能体发现基线流程无法报告的缺陷时才记功，从而在银行模型验证中将智能体的真实贡献与自动化本身带来的收益区分开来。

    

    大语言模型（LLM）智能体正被提议用于银行模型验证，然而现有评估在比较智能体辅助工作与人工实践时，会把来自数字化证据和确定性测试的收益错误地归功于智能体。我们提出一种分阶段架构：一个不使用大语言模型的可控工作流（P1）既是回退机制，也是逐项授权的智能体任务（P2）的对照臂；同时提出一种归因协议，仅当智能体发现P1尚未报告的缺陷时才记入其功劳。该协议会扣除诱饵误报、不必要的操作以及对开发者断言的背书，将无效输出计为漏报，并将新近可自动化的发现连同发现功劳一并纳入P1，同时预先固定安全边界与盈亏平衡成本门槛。在一个合成信用评分卡测试平台上，五个脚本化参考策略证明了该协议的必要性：朴素的案例准确率（4/6）与汇总检测率（3/5）无法区分四种有缺陷的策略，而我们的协议能够区分全部五种策略。

    arXiv:2611.27094v1 Announce Type: cross  Abstract: LLM agents are being proposed for bank model validation, yet evaluations that compare agent-assisted work with manual practice credit the agent with gains that come from digitized evidence and deterministic tests. We propose a staged architecture in which an LLM-free controlled workflow (P1) is both the fallback and the control arm for individually authorized agent tasks (P2), and an attribution protocol that credits an agent only for faults P1 does not already report. The protocol charges decoy false alarms, unnecessary actions and endorsed developer assertions, counts invalid outputs as misses, folds newly automatable findings back into P1 with discovery credit, and fixes safety bounds and a break-even cost gate in advance. On a synthetic credit-scorecard harness, five scripted reference policies show why this matters: naive case accuracy (4/6) and pooled detection (3/5) tie four flawed policies, whereas the protocol separates all fi
    
[^53]: PLoQ：面向大语言模型激活值4比特量化的提示词专属低秩基

    PLoQ: A Prompt-Specific Low-Rank Basis for 4-bit Quantization of LLM Activations

    [https://arxiv.org/abs/2611.27092](https://arxiv.org/abs/2611.27092)

    PLoQ提出了一种提示词自适应的低秩分解方法，在预填充阶段为每个提示词动态估计低秩基，将主导激活分量单独量化并在矩阵乘法中恢复，从而在无需修改权重的情况下实现大语言模型激活值的4比特量化。

    

    大语言模型的低比特激活量化仍然具有挑战性，因为少数主导方向决定了量化范围。最先进的方法通过离线校准并折叠进权重的变换将这些方向隔离在高精度子空间中，这为所有提示词固定了该子空间，导致每个提示词的少数独特方向得不到保护，并且在权重无法吸收该变换的情况下无法应用。我们提出PLoQ，这是一种提示词自适应的低秩分解方法，它在量化之前提取主导激活分量，对其进行单独量化，并在层的矩阵乘法运算中恢复它们，而无需修改权重。低秩分量的基在预填充阶段从每个提示词中估计，并在整个解码过程中重复使用。在与最先进的混合精度量化相同的位宽下（4比特权重、激活值和K/V缓存，以及八分之一的通道……）

    arXiv:2611.27092v1 Announce Type: cross  Abstract: Low-bit activation quantization of large language models remains challenging, as a small number of dominant directions dictate the quantization range. State-of-the-art methods isolate these directions in a high-precision subspace through transformations calibrated offline and folded into the weights, which fixes the subspace for all prompts, leaving each prompt's few unique directions unprotected, and rules it out wherever no weight can absorb the transformation. We present PLoQ, a prompt-adaptive low-rank decomposition that extracts the dominant activation components before quantization, quantizes them separately, and restores them within the layer's GEMM, without modifying the weights. The low-rank component's basis is estimated from each prompt during prefill and reused throughout decoding. At the same bit width as state-of-the-art mixed-precision quantization (4-bit weights, activations and K/V cache with an eighth of the channels 
    
[^54]: ORION：面向Transformer系统性泛化的精确规范态监督

    ORION: Exact Canonical-State Supervision for Systematic Generalization in Transformers

    [https://arxiv.org/abs/2611.27087](https://arxiv.org/abs/2611.27087)

    ORION框架通过辅助头监督精确的规范执行状态来恢复Transformer的深度外推能力（66.7%），与解决长度外推的RoPE形成正交互补，二者结合在长度-深度联合外推中实现严格的加性增益。

    

    自回归Transformer在分布外推理中普遍失败，主要归因于两个因素：长序列上的位置损坏以及深层执行步骤中的结构性状态漂移。我们提出了ORION，一个通过辅助头对精确规范执行状态进行监督的框架，将潜在轨迹锚定到有效的计算流形上。在Folding、GridShift和CLRS-30基准上，一项预注册的2×2因子实验表明，RoPE和ORION解决的是相互正交的两个维度：RoPE恢复了长度外推能力（59.8%），而ORION恢复了深度外推能力（66.7%）。此外，两者结合在长度-深度联合外推中实现了严格的加性增益，而单独使用任一方法只能取得接近满意的效果。阴性对照实验证实该增益严格依赖于精确的规范对齐；基于哈希的状态统计无法约束深度外推。

    arXiv:2611.27087v1 Announce Type: cross  Abstract: Autoregressive Transformers generally fail on out-of-distribution reasoning due to two primary reasons: positional corruption over long sequences and structural state drift across deep execution steps. We introduce ORION, a framework that supervises exact canonical execution states via auxiliary heads to anchor latent trajectories to valid computational manifolds. Across Folding, GridShift, and CLRS-30, a pre-registered 2 x 2 factorial shows that RoPE and ORION address orthogonal axes. RoPE recovers length extrapolation (59.8%), while ORION recovers depth extrapolation (66.7%). Moreover, their combination complementarily achieves strict additive gain in joint length-depth extrapolation, while either method alone offers near satisfactory results. Negative controls confirm that the gain depends strictly on exact canonical alignment; hashed state statistics fail to constrain depth extrapolation.
    
[^55]: 无损重新分词会将RLVR训练更新移动多远？

    How Far Does Lossless Re-tokenization Move an RLVR Update?

    [https://arxiv.org/abs/2611.27083](https://arxiv.org/abs/2611.27083)

    在RLVR训练中，对采样文本进行无损重新分词（如拆分少量token）会使策略梯度更新产生显著位移——其位移距离可达更换整个批次的0.556倍以上，且在同一检查点内这些位移方向保持一致。

    

    语言模型会对解码为同一字符串的不同token序列赋予不同的概率。在可验证奖励强化学习（RLVR）中，若训练器对采样文本重新编码，可能会在具有相同奖励的另一条token序列上进行训练。我们探究这种重新分词会将训练更新移动多远，以及这种移动是否具有一致的方向。从一个检查点出发，我们在相同的rollout上比较不同更新——包括规范编码版本和将选定token拆分为两段后的版本——并与在不同批次上的更新进行对比。在Qwen2.5-0.5B上进行三位数加法任务时，每25个token中拆分约1个，产生的更新位移可达更换整个批次效果距离的0.556倍。该比值在四个模型中均随拆分比例上升而增大，且取决于拆分了哪些token；在同一检查点内，不同批次间的位移共享同一方向。在一个每个分词方式对应一张参数表、且只采样单个字符串的表格模型中，我们证明了……（原文摘要在此截断）

    arXiv:2611.27083v1 Announce Type: new  Abstract: Language models assign different probabilities to token sequences that decode to the same string. In reinforcement learning from verifiable rewards (RLVR), a trainer that re-encodes sampled text may train on a different token sequence with the same reward. We ask how far this moves a training update, and whether the movement has a consistent direction. From one checkpoint, we compare updates on the same rollouts, canonical and with selected tokens split into two pieces, against an update on a different batch. On Qwen2.5-0.5B with three-digit addition, splitting about one token in 25 moves the update 0.556 times as far as replacing the batch. The ratio rises with the fraction split in four models and depends on which tokens are split; within a checkpoint, the displacements share a direction across batches. In a tabular model with one table per tokenization and a single sampled string, we prove that whenever two strings below an unsampled 
    
[^56]: LLM引导的先验何时有用？语言模型作为小样本贝叶斯回归先验的受控审计

    When Do LLM-Elicited Priors Help? A Controlled Audit of Language Models as Priors for Small-Sample Bayesian Regression

    [https://arxiv.org/abs/2611.27075](https://arxiv.org/abs/2611.27075)

    该研究对LLM作为小样本贝叶斯回归先验的做法进行了受控审计，发现不同模型的先验误差高度相关（相关性达0.92–0.99）、其声称的不确定性不可靠，且相比最佳经典先验仅在四个基准中的一个上带来明显改进。

    

    大语言模型（LLM）越来越多地被用来为小样本贝叶斯回归的系数确定中心位置，但相关证据并不明确：基准测试往往被设计成与先验相匹配，且模型可能已经记住了这些先验。我们使用一个前沿模型和一个快速层模型对这一做法进行了审计，采用隐藏的特征名称、截止日期之后的房价数据，以及一项预先设定的、故意给特征提供错误描述的测试。首先，这些模型的错误高度相关（在不同抽样、不同供应商和改写表述之间的相关性为0.92–0.99）：对20次抽样取平均并不比单次抽样更准确，而将各次抽样视为独立样本会导致过度自信。这种错误的很大一部分源于孤立地报告每个特征的效应，而不是在固定其他特征的情况下报告其效应。其次，与最佳的经典先验相比，LLM给出的先验中心仅在四个基准中的一个上带来了明显改进（RMSE降低5.8%），在两个基准上仅有边际改进，而在第四个基准上则恶化了平均RMSE。第三，LLM所声称的不确定性是不可靠的；

    arXiv:2611.27075v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly asked where to center the coefficients of a small-sample Bayesian regression, but the evidence is unclear: benchmarks are often built to match the prior and models may remember them. We audit this practice with one frontier and one flash-tier model, using hidden feature names, post-cutoff housing data and a pre-specified test that gives features the wrong descriptions. First, the models share their errors (correlation $0.92$--$0.99$ across draws, vendors and paraphrases): averaging $20$ draws is no more accurate than one, and treating draws as independent creates overconfidence. Much of this error comes from reporting each feature's effect alone, not with the others held fixed. Second, against the best classical prior, the LLM center helps clearly on one of four benchmarks (RMSE $-5.8\%$), marginally on two, and hurts mean RMSE on the fourth. Third, the LLM's stated uncertainty is unreliable;
    
[^57]: NightmareDiscourseBench：用于小说直接话语识别流水线的基准数据集

    NightmareDiscourseBench: A Benchmark Dataset for Fictional Direct Discourse Identification Pipelines

    [https://arxiv.org/abs/2611.27059](https://arxiv.org/abs/2611.27059)

    本文提出了一个新基准数据集NightmareDiscourseBench，基于叙事理论构建直接话语分类评估框架，弥补了现有基准数据理论不一致和仅关注言语形式的不足。

    

    本文介绍了一个新的基准数据集NightmareDiscourseBench，用于测试识别英语小说中直接话语形式的NLP流水线。我们首先回顾了该任务现有的公开可用基准数据集（提供汇总统计信息并审查可用的元数据），指出了理论上的不一致性以及仅关注言语是该领域的主要问题。随后，我们概述了基于叙事理论的框架，用于分类和评估不同形式的直接话语，最后描述了NightmareDiscourseBench，这是一个包含12个文本样本的新基准测试集，并附有与直接话语识别任务相关的元数据。

    arXiv:2611.27059v1 Announce Type: new  Abstract: In this paper we introduce a new benchmark dataset, NightmareDiscourseBench, for testing NLP pipelines that identify forms of direct discourse in English-language fiction. We first review existing publicly available benchmarks (providing summary statistics and reviewing available metadata) for this task, identifying theoretical inconsistency and exclusive focus on speech as primary issues in this domain. We then overview our narrative theory-based framework for classifying and evaluating different forms of direct discourse, and finally describe NightmareDiscourseBench, a new benchmark test set of 12 text samples accompanied by metadata pertinent to the task of direct discourse identification.
    
[^58]: 隔离语言外壳：一项内容可控的多语言推理研究

    Isolating the Language Shell: A Content-Controlled Study of Multilingual Reasoning

    [https://arxiv.org/abs/2611.27053](https://arxiv.org/abs/2611.27053)

    该研究通过保持推理内容不变、仅改变教师推理呈现的语言外壳，证明多语言推理蒸馏中外壳语言的选择会显著影响学生模型的下游准确率与输出语言行为，而预训练兼容性无法预测这些效应。

    

    多语言推理蒸馏需要选择呈现教师推理所使用的语言，然而现有的方法往往将语言形式与生成、翻译和过滤方面的差异混杂在一起。我们通过将相同的、经过验证的英文推理轨迹分别渲染为英文、翻译后的目标语言以及语言混合的三种外壳，构建了内容可控的监督数据，并在三种目标语言上对两个预训练的学生模型进行微调，在两个难度级别上评估它们的多语言数学推理能力。我们发现，即使在底层推理内容保持不变的情况下，外壳的选择仍会改变下游准确率，且其影响因学生模型、语言和难度级别的不同而变化。预训练兼容性并不能一致地预测这些结果，即使测试未发现不同外壳之间存在明显的准确率差异，它们的输出语言行为仍可能存在显著差异。这些结果表明……（摘要原文在此处截断）

    arXiv:2611.27053v1 Announce Type: new  Abstract: Multilingual reasoning distillation requires choosing the language in which teacher reasoning is presented, yet existing recipes often entangle linguistic form with differences in generation, translation, and filtering. We construct content-controlled supervision by rendering the same verified English trajectories into English, translated target-language, and language-mixed shells, and fine-tune two pretrained students in three target languages, evaluating them on multilingual mathematical reasoning at two difficulty levels. Even with the underlying reasoning content held fixed, we find that shell choice can change downstream accuracy, with effects varying across students, languages, and difficulty levels. Pre-training compatibility does not consistently track these outcomes, and even when our tests find no clear accuracy difference between shells, their output-language behavior can still differ substantially. These results suggest that 
    
[^59]: 从信用证要求到可执行规范：评估验证的准确性与完整性

    From Letter-of-Credit Requirements to Executable Specifications: Evaluating Verification Accuracy and Completeness

    [https://arxiv.org/abs/2611.26988](https://arxiv.org/abs/2611.26988)

    该论文将信用证审核要求逐条转化为可执行的原子条件并结合确定性验证器实现可追溯审核，揭示了仅以最终决定准确率衡量审核的误导性，并在数据保密条件下通过仅用合成数据微调的本地9B解析器达到93%的原子要求检测率。

    

    在信用证（LC）单据审核中，正确的最终决定并不能表明每一项要求都经过了审核。因此，我们逐条评估审核要求，并将每项要求变为可执行的。语言模型将信用证中的提单（BL）和商业发票（IV）要求转换为原子条件和逻辑结构，再由确定性验证器检索指定的证据并计算每项结果，使得每项检查都可以与黄金编码进行对比并追溯至日志记录的证据。由于信用证记录是保密的，我们本地部署的9B解析器仅在基于模板和人工审核的公开文档值构建的合成数据对上进行微调。在345个真实信用证案例上，最终决定准确率被证明具有误导性：一个恒定输出PASS的规则拥有最高的原始准确率，却什么都没有审核。尽管微调后的解析器在训练期间从未见过真实记录，它仍检测出93%的原子要求……

    arXiv:2611.26988v1 Announce Type: new  Abstract: In letter-of-credit (LC) document examination, a correct final decision does not show that every requirement was examined. We therefore evaluate examination requirement by requirement and make each requirement executable. A language model converts the bill of lading (BL) and commercial invoice (IV) requirements of a credit into atomic conditions and logical structures, and a deterministic verifier retrieves the named evidence and computes each outcome, so every check can be compared with a gold encoding and traced to logged evidence. Because LC records are confidential, our on-premises 9B parser is fine-tuned only on synthetic pairs built from templates and human-reviewed public document values. On 345 real LC cases, final-decision accuracy proves misleading: a constant PASS rule has the highest raw accuracy while examining nothing. Although it never sees a real record during training, the fine-tuned parser detects 93% of the atomic requ
    
[^60]: 运行一次，得到两个：本地LLM的近乎零成本的流式安全检查

    Run One, Get Two: Near-Free Streaming Safety Checks for Local LLMs

    [https://arxiv.org/abs/2611.26975](https://arxiv.org/abs/2611.26975)

    ToRA通过让第二个模型作为额外一行嵌入生成器的解码步骤并共享权重与KV缓存，以每个token不到4%的额外延迟实现了接近专用防护模型准确性的本地LLM流式安全检查。

    

    本地LLM助手一次只为一个用户解码，因此解码受内存带宽限制，计算资源处于闲置状态。如今的流式安全检查需要在准确性与成本之间权衡：专用的防护模型虽然准确，但几乎使每个token的延迟翻倍；而基于隐藏状态的探针虽然廉价，但准确性较低。我们提出ToRA，这是一种适配器，它在生成器的解码步骤中将第二个模型作为额外的一行来运行，共享其权重和KV缓存。双行内核只需为两行各读取一次每个权重。在Qwen3-0.6B上，ToRA匹配或超越了专用的流式防护模型，在17个基准测试中的12个上取得了更高的F1分数。它在相似成本下比探针多检测出11.8个百分点的不安全响应，且每个token仅增加不到4%的延迟。

    arXiv:2611.26975v1 Announce Type: new  Abstract: Local LLM assistants decode for one user at a time, so decoding is bound by memory bandwidth and leaves compute idle. Streaming safety checks today trade accuracy against cost. A dedicated guard model is accurate but nearly doubles the latency per token, while a hidden-state probe is cheap but less accurate. We present ToRA, an adapter that runs a second model as an extra row in the generator's decode step, sharing its weights and KV cache. Dual-row kernels read each weight once for both rows. On Qwen3-0.6B, ToRA matches or beats a dedicated streaming guard, with higher F1 on 12 of 17 benchmarks. It detects 11.8 points more unsafe responses than the probe at a similar cost, under 4% added latency per token.
    
[^61]: 正确的数字，错误的故事：当大语言模型在输入之外叙述财务变化时

    Right Numbers, Wrong Story: When LLMs Narrate Financial Changes Beyond Their Inputs

    [https://arxiv.org/abs/2611.26968](https://arxiv.org/abs/2611.26968)

    论文提出FrameCheck插件，在不增加模型调用的情况下检测并约束LLM财务报告草稿中的“无依据框架叙述”，确保报告叙述忠实于输入事实。

    

    由大语言模型（LLM）起草的财务报告可以把每个数字都写对，却仍会超出其输入内容，添加没有人提供的判断和因果解释。我们将这类添加内容称为“无依据的框架叙述”（unsupported framing），并发现它们在LLM草稿中出现的频率远高于专业人士撰写的报告；而试图通过在提示词中说明每个词的用法来修复该问题，反而会使这类判断更加频繁。为此，我们提出FrameCheck，这是一个置于LLM与人工审核之间的插件，用于确保每份草稿不超出其输入内容。具体而言，FrameCheck可以识别判断性词语、区分有依据和无依据的因果解释，且无需额外的模型调用，从而引导草稿忠实于输入事实。

    arXiv:2611.26968v1 Announce Type: new  Abstract: A financial report drafted by an LLM can state every number correctly but still go beyond its input, adding judgments and causes that no one supplied. We call these additions unsupported framing and find them far more often in LLM drafts than in reports written by proInput facts Revenue +3% Higher prices 10-K filings Revenue increased 3% on higher prices. 10-K writer Revenue surged 3% on higher prices, due to strong demand. LLM Fits the size? Surged: Unusual for 3% In the input? In the input? Strong demand: Not in input Higher prices: Kept fessionals, and attempts to fix them by prompting with how each word is used instead make such judgments more frequent. We therefore propose FrameCheck, a plug-in that sits beFrameCheck Revenue increased 3% on higher prices. No extra model call Judgment word Unsupported cause Supported cause tween the LLM and human review to keep each draft within its input. Specifically, FrameCheck 1) guides the draft
    
[^62]: 超越提示启发式方法：评估大语言模型中的儿童安全对齐

    Beyond Prompt Heuristics: Assessing Child Safety Alignment in LLMs

    [https://arxiv.org/abs/2611.26965](https://arxiv.org/abs/2611.26965)

    该研究揭示了当前大语言模型的儿童安全对齐严重依赖伤害类别提示线索——在缺乏此类线索的看似无害请求上未对齐率大幅上升（最高达80个百分点），而仅引入一个伤害相关词即可恢复安全行为，表明模型的儿童安全推理主要建立在词汇和语义关联之上而非真正的安全理解。

    

    儿童安全分类体系通常通过一组预定义的伤害类别来刻画安全风险，而现有评估大多使用明确提及这些类别的提示词来操作化这些类别。然而，儿童安全方面的考量也可能出现在缺乏可识别伤害类别线索的看似无害的请求中。因此，我们研究了儿童安全对齐如何泛化到此类场景。在六个模型上的实验中，我们发现无伤害类别线索的请求的未对齐率显著高于包含此类线索的请求（Qwen3.5-27B、GPT-5.4-mini 和 Claude-Haiku-4.5 分别存在 80.0、54.0 和 48.4 个百分点的差距）。通过受控的行为与机制分析，我们发现仅仅引入一个与伤害相关的词就能降低大多数模型的未对齐率，且机制层面的证据表明，儿童安全推理可能部分依赖于词汇和语义关联。

    arXiv:2611.26965v1 Announce Type: cross  Abstract: Child-safety taxonomies typically characterize safety risks through a predefined set of harm categories, and existing evaluations largely operationalize these categories using prompts that explicitly invoke them. However, child-safety considerations can also arise in seemingly benign requests that lack recognizable harm-category cues. We therefore investigate how child-safety alignment generalizes to such settings. Across six models, we find considerably higher misalignment rates on requests without recognizable harm-category cues than on those containing such cues (80.0, 54.0, and 48.4 percentage-point gaps for Qwen3.5-27B, GPT-5.4-mini, and Claude-Haiku-4.5, respectively). Through controlled behavioral and mechanistic analyses, we find that introducing a single harm-associated word reduces misalignment across most models, with mechanistic evidence suggesting that child-safety reasoning may rely in part on lexical and semantic associa
    
[^63]: 语言中信息的自相似组织

    Self-similar Organization of Information in Language

    [https://arxiv.org/abs/2611.26935](https://arxiv.org/abs/2611.26935)

    本文提出了一种基于信息度量标度行为的预测信息组织理论，并通过在长文本中观察到不同长度尺度上信息度量的标度坍缩，证明了语言中预测信息具有自相似结构，这对理解语言的统计规律及大语言模型如何学习语言结构具有重要意义。

    

    与前文文本相关的、用于预测后续内容的信息量（即预测信息）有多少，以及这些信息如何沿着文本历史分布？这些问题对于理解语言的统计结构以及大语言模型（LLMs）能够从中学到什么都至关重要。事实上，训练用于预测文本中下一个词元的LLM必须利用这种预测信息，因此其组织方式决定了语言结构如何变得可供模型学习。我们基于一系列信息度量的标度行为，发展了一种关于预测信息组织的理论，该理论预测：如果这种组织在所有测量长度尺度上具有相同的形式——即具有自相似结构——那么在不同长度上评估的信息度量将坍缩到一个共同的形式。我们在长文本中观察到了这种标度坍缩现象。

    arXiv:2611.26935v1 Announce Type: new  Abstract: How much information relevant to predicting the future of a passage (i.e. predictive information) is carried by the preceding text, and how is this information distributed along that history? These questions are fundamental to understanding both the statistical structure of language and what large language models (LLMs) can learn about it. Indeed, LLMs trained to predict the next token in text must exploit this predictive information, so its organization determines how the structure of language becomes available to the model to learn. We develop a theory for the organization of predictive information based on the scaling behavior of a family of information measures, and the theory predicts that if this organization has the same form at all measured length scales -- i.e., it has a self-similar structure -- then the information measures evaluated at different lengths collapse onto a common form. We observe this scaling collapse in long tex
    
[^64]: 大语言模型表征中的部分规范轴

    Partially canonical axes in large language model representations

    [https://arxiv.org/abs/2611.26920](https://arxiv.org/abs/2611.26920)

    该研究挑战了“大语言模型表征坐标轴是任意的”这一常见假设，发现在12个LLM中原生轴比随机旋转轴更能与人类大脑语言响应和可解释语言特征对齐，揭示LLM表征轴是“部分规范的”——既非完全任意也非完全规范。

    

    大语言模型对语言发展出丰富的内部表征，但其各个坐标轴本身是否承载意义，还是意义仅编码在表征空间中的方向上，仍是一个悬而未决的问题。一个常见的假设是LLM中的单个轴是任意的。我们在此挑战这一观点：在来自六个模型家族的12个大语言模型中，与保持相同几何结构的随机旋转轴相比，原生轴能更可靠地与其他模型的表征、人类大脑对语言的响应以及可解释的语言特征对齐。然而，原生轴远未达到最佳可实现的对齐程度，因此LLM的坐标并非完全任意，但也不是完全规范的：它们是部分规范的。关于LLM如何表征语言的理论，必须既能解释它们所学习的群体几何结构，又能解释为什么单个单元获得了在模型间重复出现的响应特征模式。

    arXiv:2611.26920v1 Announce Type: cross  Abstract: Large language models (LLMs) develop rich internal representations of language, but whether their individual coordinate axes carry meaning, or whether meaning is encoded only in directions in representation space, is still an open question. A common assumption is that individual axes in LLMs are arbitrary. We here challenge this view: across 12 LLMs from six model families, the native axes are reliably more aligned with other models' representations, with human brain responses to language, and with interpretable linguistic features than randomly rotated axes that preserve the same geometry. The native axes nonetheless fall well short of the best achievable alignment, so LLM coordinates are not fully arbitrary but also not fully canonical: they are partially canonical. A theory of how LLMs represent language must explain both the population geometry they learn and why individual units acquire patterns of response profiles that recur acr
    
[^65]: ERA：面向冷启动跨模型潜在检索的嵌入、路由与自适应方法

    ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval

    [https://arxiv.org/abs/2611.26919](https://arxiv.org/abs/2611.26919)

    ERA提出了一种冷启动跨模型潜在检索系统，通过余弦最远点采样选取少量锚点文档、利用结合正交正则化的流形对齐损失训练轻量级适配器，并借助确定性路由器升级处理困难查询，从而无需对整个语料库重新嵌入即可将检索系统迁移到新的嵌入模型。

    

    生产级检索系统与其语料库索引所使用的嵌入模型紧密绑定。将查询迁移到不同的编码器通常需要对整个语料库重新进行嵌入，而在冷启动阶段既没有查询日志可供利用，新编码器对每个文档进行嵌入也会产生API调用或计算成本。我们提出了ERA（Embed, Route, Adapt），一个面向冷启动跨模型潜在检索的系统。ERA将少量锚点预算分配给最具信息量的文档（通过余弦最远点采样从现有目标索引中选取），使用流形对齐损失训练一个轻量级适配器——该损失将方向对齐与正交正则化相结合以防止表示坍缩——并将该适配器与一个确定性路由器配对，后者会升级处理那些决定用户感知检索质量的少量困难查询。在五个BEIR数据集上，ERA达到或超越了逐点L2和嵌入转换基线，取得了相对增益……

    arXiv:2611.26919v1 Announce Type: cross  Abstract: Production retrieval systems are tied to the embedding model that indexed their corpus. Moving queries to a different encoder normally requires re-embedding the whole corpus, and at cold start there are no query logs while every document embedded by the new encoder incurs an API or compute cost. We present ERA (Embed, Route, Adapt), a system for cold-start cross-model latent retrieval. ERA spends a small anchor budget on maximally informative documents, selected from the existing target index via Cosine Farthest Point Sampling, trains a lightweight adapter with a Manifold Alignment Loss that couples directional alignment with orthogonal regularization to prevent representation collapse, and pairs the adapter with a deterministic router that escalates the rare, hard queries that define perceived retrieval quality. Across five BEIR datasets, ERA matches or outperforms point-wise L2 and embedding-conversion baselines, with relative gains 
    
[^66]: 视觉编码器带来了什么？无编码器多模态预训练中的视觉信息增益

    What Does the Visual Encoder Buy? Visual Information Gain in Encoder-Free Multimodal Pretraining

    [https://arxiv.org/abs/2611.26917](https://arxiv.org/abs/2611.26917)

    提出视觉信息增益（VIG）这一精确度量，将无编码器多模态模型从图像中获取的信息分解为全局信息与布局信息，发现图像可降低3%至46%的预测损失，全局信息在训练早期即出现而布局信息随训练缓慢增长。

    

    最近的规模研究通过在图文数据的文本下一词预测损失（next-token loss）上比较无编码器与基于编码器的模型，探讨视觉-语言模型是否仍需要视觉编码器。然而，这种损失可以在完全不使用图像的情况下下降，且两次具有相同损失的运行在从图像中获取的信息上可能存在差异。视觉信息增益（VIG）是指将图像替换为均匀图像时损失的增加量，它可以被精确地分解为全局信息（由图像统计特性携带）和布局信息（通过打乱图像内部的图像块即可破坏的信息）。我们将其应用于参数量从1.58亿到10.4亿的解码器，这些解码器通过线性嵌入读取像素块，每十亿词元评估一次直至400亿词元，同时还评估了19个公开检查点。在我们的解码器中，图像可将损失降低3%至46%，具体取决于图像类型。全局信息在训练的第一个十亿词元之后即已出现，而布局信息起初接近于零，并随训练缓慢增长。

    arXiv:2611.26917v1 Announce Type: cross  Abstract: Recent scaling studies ask whether vision-language models still need a visual encoder by comparing encoder-free and encoder-based models on the next-token loss of the text in image-text data. This loss can fall without any use of the image, and two runs with the same loss can differ in what they take from it. Visual information gain (VIG) is the loss increase when the image is replaced by a uniform one. It splits exactly into global information, carried by image statistics, and layout information, which shuffling the patches within the image destroys. We apply it to decoders with 158M to 1.04B parameters that read pixel patches through a linear embedding, evaluated every billion tokens up to 40B, and to 19 public checkpoints. In our decoders the image lowers the loss by 3% to 46%, depending on the image type. Global information is present after the first billion tokens. Layout information starts near zero, grows slowly with training an
    
[^67]: AstronautBench：通过知识、证据与进展衡量智能体跨回合的自我改进

    AstronautBench: Measuring Agent Self-Improvement Across Episodes Through Knowledge, Evidence, and Progress

    [https://arxiv.org/abs/2611.26910](https://arxiv.org/abs/2611.26910)

    AstronautBench基于游戏《星际拓荒》构建了一个新基准，通过知识、证据支持与任务进展三个互补视角，衡量权重固定的LLM智能体在世界不断重置的循环中凭借自身经验实现跨回合自我改进的能力。

    

    LLM智能体被期望在权重固定不变的情况下，从跨回合的自身经验中实现自我改进。具身世界中的自我改进无法通过单一指标来衡量。现有基准测试分别衡量任务进展、世界知识问答（QA）和情景记忆，但一个正确答案并不能表明智能体曾经观察到支持该答案的证据，而任务进展也无法表明智能体究竟学到了什么。我们提出了AstronautBench，它通过三个视角来衡量同一个智能体运行过程中的自我改进：它知道什么、它自身获得的证据是否支持这些知识、以及它在世界中取得了多大的进展。该基准建立在游戏《星际拓荒》之上，其中每个回合是一个22分钟的循环，循环结束后世界重置，只有智能体的知识得以保留，且世界中的进展由知识所限制。在每个回合结束后，我们测量到达的地点数量和飞船日志中揭示的传闻事实数量，以及关于164个问题的回答正确性……

    arXiv:2611.26910v1 Announce Type: new  Abstract: LLM agents are expected to improve themselves from their own experience across episodes, with fixed weights. Self-improvement in an embodied world cannot be read from one measure. Existing benchmarks measure task progress, world-knowledge question answering (QA), and episodic memory, but a correct answer does not show that the agent ever observed the evidence for it, and progress does not show what the agent learned. We introduce AstronautBench, which measures one agent's self-improvement through three views on the same run: what it knows, whether its own evidence supports that knowledge, and how far it gets in the world. It is built on the game Outer Wilds, where each episode is a 22-minute loop after which the world resets, only the agent's knowledge carries over, and progress is gated by knowledge. After every episode, we measure the places reached and ship-log rumor facts revealed, the correctness of answers to 164 questions about th
    
[^68]: 缩放究竟增加了什么？分解视觉-语言模型中的细粒度感知失败

    What Does a Zoom Actually Add? Decomposing Fine-Grained Perception Failures in Vision-Language Models

    [https://arxiv.org/abs/2611.26904](https://arxiv.org/abs/2611.26904)

    该研究将缩放操作分解为多个独立因素，发现大多数VLM被缩放拯救的细粒度感知失败并非因为缺少像素信息，而主要源于对已看像素的重新分词等处理环节。

    

    将某个区域放大（zoom）往往能把视觉-语言模型（VLM）的错误答案变成正确答案，越来越多的文献将此视为模型缺乏像素信息的证据。但缩放会同时改变几件事：它移除了周围的上下文，以更高的密度对模型已经看过的像素进行重新分词，并且如果第一遍处理时进行了降采样，它还会暴露模型从未接收过的新细节。我们为每一个失败案例分离出这些因素。在保持第一遍处理固定不变的情况下，我们只改变第二次观察，并将每个失败归因于能修复它的最廉价方案：重用缓存的token、隔离目标区域、对已看过的像素重新分词，或获取新的像素。该研究涵盖了Qwen2.5-VL、Qwen3-VL和InternVL3在V*、ZoomBench和TextVQA上的2,753个被缩放拯救的失败案例，第一遍的像素预算从比源图像少17倍到原生分辨率不等。首先，大多数被拯救的失败并非因为缺乏像素：对已看过的像素进行上采样（摘要内容此处截断）

    arXiv:2611.26904v1 Announce Type: cross  Abstract: Zooming into a region often turns a wrong answer from a vision-language model (VLM) into a right one, and a growing literature takes this as evidence that the model lacked the pixels. But a zoom changes several things at once. It removes the surrounding context, re-tokenizes, at a higher density, pixels the model has already seen, and, if the first pass was downsampled, exposes detail the model never received. We separate these factors for each failure. Holding the first pass fixed, we vary only the second look and assign every failure to the cheapest one that fixes it: reusing cached tokens, isolating the region, re-tokenizing the seen pixels, or acquiring new ones. The study covers 2,753 zoom-rescued failures of Qwen2.5-VL, Qwen3-VL and InternVL3 on V*, ZoomBench and TextVQA, at first-pass budgets from 17x fewer pixels than the source to native resolution. First, most rescued failures did not lack the pixels: upsampling the pixels th
    
[^69]: 感知一次，按需升级：将视觉KV记忆从小型VLM传递给大型VLM

    Perceive Once, Escalate on Demand: Handing Off Visual KV Memory from Small to Large VLMs

    [https://arxiv.org/abs/2611.26902](https://arxiv.org/abs/2611.26902)

    提出Baton方法，通过从大型模型自身答案中无标签蒸馏逐层线性映射，将小型VLM的视觉KV缓存记忆传递给大型VLM，避免升级时重复编码历史信息，使大型模型只需编码当前帧。

    

    具身智能体无法在长视频的每一步都负担得起大型视觉语言模型（VLM），因此级联系统持续运行一个小型VLM，仅将不确定的决策升级给大型模型。然而，每次升级都迫使大型模型重新编码小型模型已经感知过的历史信息，这占据了其主要成本。我们提出了Baton，它将小型模型的视觉记忆——即其在历史信息上的键值（KV）缓存——交给大型模型。Baton基于一个发现：大型模型需要一个可用的记忆，而不是自身缓存的忠实副本。一个训练用于重现大型模型缓存的映射虽然与它最为接近，但相比单独使用小型模型却没有任何收益。Baton转而从大型模型自身的答案中无标签地蒸馏出一个逐层线性映射，大型模型只需编码当前帧。对该映射的干预实验将增益追溯到由大型模型中间三分之一层所携带的一个方向。

    arXiv:2611.26902v1 Announce Type: cross  Abstract: An embodied agent cannot afford a large vision-language model (VLM) at every step of a long video, so a cascade runs a small VLM continuously and escalates only uncertain decisions to a large one. Each escalation, however, forces the large model to re-encode a history the small model has already perceived, and this dominates its cost. We present Baton, which hands the large model the small model's visual memory, its key-value (KV) cache over the history. Baton rests on one finding: the large model needs a usable memory, not a faithful copy of its own. A map trained to reproduce the large model's cache matches it most closely, yet brings no gain over the small model alone. Baton instead distills a per-layer linear map, without labels, from the large model's own answers, with the large model encoding only the current frame. Interventions on the map trace the gain to one direction, carried by the middle third of the large model's layers. 
    
[^70]: 大规模归因风格建模：用于心理归因评估与重构的数据集及分析

    Modeling Attributional Style at Scale: A Dataset and Analysis for Psychological Attribution Assessment and Reframing

    [https://arxiv.org/abs/2611.26900](https://arxiv.org/abs/2611.26900)

    该论文提出了首个大规模归因风格基准数据集ASTD及配套评估指标，包含42,000个真实事件和七种心理归因风格的归因标注，实现了归因风格的可扩展评估与认知重构。

    

    根据习得性无助理论的修正版本，反复暴露于不可控的负面事件可能催生抑郁性归因风格，增加罹患抑郁症的易感性，但这仍是认知治疗可干预的靶点。然而，归因认知的计算研究一直受到缺乏大规模数据集和稳健评估协议的制约。在本工作中，我们提出了归因风格迁移数据集（ASTD）及配套的专门评估指标，这是首个旨在大规模建模、评估和重构归因解释的基准数据集。ASTD通过整合基于大语言模型（LLM）的生成与专家验证的“预防-过滤-验证”流程构建而成，包含42,000个真实世界事件，并配有涵盖七种风格、具有心理学依据的归因。利用该数据集，我们解决了两个关键挑战：(1) 基于监督方法的可扩展归因风格评估……

    arXiv:2611.26900v1 Announce Type: cross  Abstract: According to the reformulated Learned Helplessness theory, repeated exposure to uncontrollable negative events can foster a depressogenic attributional style, increasing susceptibility to depression yet remaining a tractable target for cognitive therapy. Computational research on attributional cognition, however, is hampered by the lack of large-scale datasets and robust evaluation protocols. In this work, we introduce the Attributional Style Transfer Dataset (ASTD) along with dedicated evaluation metrics, the first benchmark designed to model, assess, and reframe attributional explanations at scale. Constructed via a Prevent-Filter-Validate pipeline that integrates LLM-based generation with specialist validation, ASTD contains 42,000 real-world events paired with psychologically grounded attributions spanning seven styles. Using this dataset, we address two key challenges: (1) scalable assessment of attributional style via both superv
    
[^71]: ProbeAgent：可验证环境中智能体失败的原理性诊断

    ProbeAgent: Principled Diagnosis of Agent Failures in Verifiable Environments

    [https://arxiv.org/abs/2611.26897](https://arxiv.org/abs/2611.26897)

    ProbeAgent是一个自动化诊断框架，通过构建经过验证的可执行测试、生成保留疑似失败原因的变体任务并观察错误是否复现，来系统性发现、验证并归类智能体的共有失败模式。

    

    改进智能体需要发现它们在哪里失败，并理解失败的原因。智能体在不断变化的环境中执行多步操作，因此诊断其失败不能仅仅依靠检查大语言模型（LLM）的最终响应。同样的不成功结果可能源于任务本身的难度，也可能源于智能体在多个任务中重复犯下的错误。我们提出ProbeAgent，一个自动化框架，用于发现智能体的失败、测试所推测的失败原因，并识别共同的失败模式。从基准任务出发，ProbeAgent构建可执行的测试，并利用参考解答、自动化检查以及对任务需求的审查来验证这些测试。对于每条失败的轨迹，它会解释错误原因，并构建新的任务——这些任务在改变周围上下文的同时保留疑似的原因。在这些任务上运行智能体，可以检验相同的错误是否会再次出现。随后，ProbeAgent将具有相似原因的失败进行分组，并生成……

    arXiv:2611.26897v1 Announce Type: cross  Abstract: Improving agents requires discovering where they fail and understanding why. An agent acts over multiple steps in a changing environment, so diagnosing its failures requires more than inspecting the final response of a large language model (LLM). The same unsuccessful outcome can arise from task difficulty or from an error the agent repeats across tasks. We introduce ProbeAgent, an automated framework that discovers agent failures, tests their proposed causes, and identifies shared failure modes. Starting from benchmark tasks, ProbeAgent builds executable tests and validates them using reference solutions, automated checks, and review of the task requirements. For each failed trajectory, it explains the error and constructs new tasks that preserve the suspected cause while changing the surrounding context. Running the agent on these tasks tests whether the same error recurs. ProbeAgent then groups failures with similar causes and gener
    
[^72]: 你的评审器会继承你的提示词：LLM即评审中的上下文标签偏置是有方向性的，而非对称的

    Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric

    [https://arxiv.org/abs/2611.26881](https://arxiv.org/abs/2611.26881)

    研究表明，多模态视频问答评估流水线中的LLM评审器会从其少样本示例中单向继承标签偏置：当示例偏向"Yes"时其接受率骤升约20–30个百分点并伴随单侧错误，而当示例平衡或偏向"No"时则接近人类共识，证明这种上下文标签偏置是方向性的而非对称的。

    

    LLM-as-judge（以大语言模型作为评审器）已成为对开放式模型输出进行评分的标准方法，然而评审器本身是一个由少量上下文示例引导的冻结语言模型，而此类模型已知会吸收其演示示例中的标签分布，而非将其视为中性的示例。我们探究这种标签偏置是否会传导到多模态（视频问答）评估流水线中使用的LLM评审器，以及它以何种形式呈现。通过使用完全冻结的生成器-评审器流水线，我们保持生成的答案不变，仅在四个演示集之间改变评审器少样本示例的Yes/No（是/否）平衡。每个判定结果均与每个数据集200个样本上的双人标注人类共识进行对照验证。当评审器的示例为平衡或偏向拒绝时，其接受率保持接近人类共识；但当示例偏向接受时，接受率急剧跳升（在NExT-QA上约30个百分点，在MSVD-QA上约20个百分点，且呈现强烈单侧的错误特征）……（原文摘要在此处截断）

    arXiv:2611.26881v1 Announce Type: cross  Abstract: LLM-as-judge has become a standard way to score open-ended model outputs, yet the judge is itself a frozen language model steered by a few in-context examples, and such models are known to absorb the label balance of their demonstrations rather than treat them as neutral illustrations. We ask whether this label bias reaches an LLM judge used inside a multimodal (video question answering) evaluation pipeline, and what form it takes. Using a fully frozen generator-judge pipeline, we hold the generated answers fixed and vary only the Yes/No balance of the judge's few-shot examples across four demonstration sets. Each verdict is validated against a two-annotator human consensus on 200 examples per dataset. The judge's acceptance rate stays close to human consensus when its examples are balanced or skewed toward rejection, but jumps sharply (by about 30 points on NExT-QA and 20 on MSVD-QA, with a strongly one-sided error profile) when the e
    
[^73]: 我的语言模型要运行多久？一个执行前运行时预测基准

    How Long Will My Language Model Run? A Benchmark for Pre-Execution Runtime Prediction

    [https://arxiv.org/abs/2611.26875](https://arxiv.org/abs/2611.26875)

    本文提出PreRun基准，用于在执行前预测深度学习训练与推理的运行时间，包含数千条GPU集群实测记录，实验发现大语言模型对运行延迟的估计严重偏低（开源模型低估90-99%）。

    

    研究人员在向共享GPU集群提交深度学习任务时，需要运行时间估计来申请时限、选择硬件和预算计算资源。大语言模型提供了一个诱人的捷径：描述任务并直接询问。我们提出了PreRun，这是一个基于执行前可用信息来预测深度学习训练和推理运行时间的基准。PreRun包含在GPU集群上收集的801条训练记录和1,884条语言与视觉模型推理记录，并与工作负载配置和明确定义的计时目标配对。它还包括在标准化条件下测量的64个文本生成配置，每个配置分别进行了三次独立的程序启动。我们将三个大语言模型（Qwen3.5-9B、GPT-OSS-120B和Claude Opus 5.5）与回归基线进行比较，并在支持的前向传播子集上评估解析估计。在16个留出的生成配置上，开源权重的大语言模型将延迟低估了90-99%。

    arXiv:2611.26875v1 Announce Type: new  Abstract: Researchers submitting deep learning jobs to shared GPU clusters need runtime estimates to request time limits, choose hardware, and budget compute. Large language models (LLMs) offer a tempting shortcut: describe the job and ask. We introduce PreRun, a benchmark for predicting deep learning training and inference runtime from information available before execution. PreRun contains 801 training and 1,884 inference records of language and vision models collected on a GPU cluster, paired with workload configurations and explicitly defined timing targets. It also includes 64 text-generation configurations measured under standardized conditions, each in three separate program launches. We compare three LLMs (Qwen3.5-9B, GPT-OSS-120B, and Claude Opus 5.5) with regression baselines and evaluate analytical estimates on a supported forward-pass subset. On 16 held-out generation configurations, the open-weight LLMs underestimate latency by 90-99%
    
[^74]: 语言模型如何压缩与比较：通过词元协方差图理解选择任务

    How Language Models Compress and Compare: Understanding Selection with Token Covariance Maps

    [https://arxiv.org/abs/2611.26874](https://arxiv.org/abs/2611.26874)

    该论文提出词元协方差图这一提示级几何分析工具，揭示了仅解码器Transformer在结构化选择任务中普遍遵循“先压缩后比较”的层级机制——早期层弥散表示提示、中间层压缩为共享低秩瓶颈、后期层重构上下文与选项间的比较几何。

    

    许多语言模型任务都可归结为在结构化备选项之间进行选择，然而我们仍然缺乏一个简单的图景来理解仅解码器Transformer如何在深度维度上组织整个选择提示。我们引入了词元协方差图，这是一种提示层面的几何视角，用于追踪各层中词元位置在隐藏状态空间中如何共同变化。这将结构化提示转化为按深度索引的图谱序列，其谱结构和块结构可以与指令、上下文和候选答案对齐。在来自4个系列的14个仅解码器模型和12个结构化选择基准上，词元协方差图揭示了一种稳健的“先压缩后比较”组织方式：早期层以弥散的方式表示提示，中间层将其压缩为共享的低秩瓶颈，后期层则重新扩展为上下文-选项和选项-选项的几何结构。这种组织方式在更大的固定模板样本、确定性的提示重写、成对……（摘要原文在此处截断）

    arXiv:2611.26874v1 Announce Type: new  Abstract: Many language-model tasks reduce to selecting among structured alternatives, yet we still lack a simple picture of how a decoder-only transformer organizes an entire selection prompt across depth. We introduce token covariance maps, a prompt-level geometric view that tracks how token positions co-vary in hidden-state space at each layer. This turns a structured prompt into a depth-indexed sequence of maps whose spectral and block structure can be aligned with instructions, context, and candidate answers. Across 14 decoder-only models from 4 families and 12 structured-selection benchmarks, token covariance maps reveal a robust compress-then-compare organization. Early layers represent prompts diffusely, middle layers compress them into a shared low-rank bottleneck, and late layers re-expand into context-option and option-option geometry. This organization recurs across larger fixed-template samples, deterministic prompt rewrites, pairwise
    
[^75]: 大语言模型在后训练过程中会记住自己产生的幻觉

    LLMs Memorize Their Own Hallucinations During Post-Training

    [https://arxiv.org/abs/2611.26863](https://arxiv.org/abs/2611.26863)

    该论文发现，采用以推理为中心的后训练方案（如基于结果的强化学习奖励和“认知”行为注入）会导致大语言模型在知识任务训练中记住自己在长推理链中产生的幻觉，且这一现象在多个开源模型上得到验证。

    

    推理领域已经塑造了前沿模型的后训练方案，即使这些模型是在数学和代码之外的领域进行训练和部署的。在本文中，我们研究了直接采用以推理为中心的训练方案（例如，在强化学习（RL）中使用基于结果的奖励、在推理链中注入“认知”行为如提出假设并从中回溯等）对知识任务的影响。我们假设这些决策虽然对推理有益，但在知识任务上训练时，会在无意中鼓励模型记住它们在长推理链中自己产生的幻觉。为了研究这一点，我们使用标准后训练方案训练模型，在强化学习过程中提取它们自我生成的幻觉，并在后续训练过程中跟踪这种记忆现象。我们在两个模型家族、两种架构共六个开源模型上的实验证实了我们的假设。我们发现

    arXiv:2611.26863v1 Announce Type: new  Abstract: Reasoning domains have shaped the post-training recipes of frontier models even though they are trained and deployed in domains beyond math and code. In this paper, we study the consequences of directly adopting reasoning-centric training recipes, e.g., the use of outcome-based rewards for reinforcement learning (RL), injection of "cognitive" behaviors such as proposing and backtracking from hypotheses in reasoning chains, etc., for knowledge tasks. We hypothesize that these decisions, while useful for reasoning, inadvertently encourage models to memorize hallucinations they themselves generate in their long reasoning chains when trained on knowledge tasks. To study this, we train models with standard post-training, extract their self-generated hallucinations during RL, and track memorization throughout the rest of training. Our experiments on six open models across two model families and two architectures confirm our hypothesis. We find
    
[^76]: 权重空间编程可写入上下文学习并在语言模型中媲美微调

    Weight-Space Programming Writes In-Context Learning and Matches Fine-Tuning in Language Models

    [https://arxiv.org/abs/2611.26841](https://arxiv.org/abs/2611.26841)

    前沿语言模型可通过权重空间编程将ICL任务向量直接写入较弱模型的稀疏MLP单元（与提示格式相关联），使其零样本性能达到ICL和微调（含LoRA）的准确率，且几乎不影响困惑度。

    

    权重空间编程是前沿语言模型最近被报道的一种能力：它无需梯度下降即可读取并改写另一个网络的权重。我们研究示例如何能以权重更新的形式提供信息，使较弱的语言模型在提示为空的情况下完成任务，并探究编辑模型究竟改写了什么。实验涵盖六项任务、四个前沿编辑模型以及三族较弱的语言模型，且没有向编辑模型透露任何关于上下文学习（ICL）的信息。在大多数运行中，编辑模型在最后一个提示 token 处测量少样本减零样本的残差，即 ICL 任务向量，并将其写入权重之中，最稀疏的情况下会写入与提示格式相关联的近乎沉默的 MLP 单元。所得结果达到了 ICL 以及微调（包括 LoRA）在相同示例上的准确率，通常几乎不改变困惑度，但在措辞发生变化时性能会有所下降。我们还通过统计未完成的运行来限定这一比较，观察到……

    arXiv:2611.26841v1 Announce Type: new  Abstract: Weight-space programming is a frontier language model's recently reported ability to read and rewrite another network's weights without gradient descent. We investigate how examples can provide a weight update so that a weaker language model performs the task with nothing in the prompt, and study what it edits. Our experiments cover six tasks, four frontier editors and three families of weaker language models, without the editors being told anything about in-context learning (ICL). In most runs, the few-shot minus zero-shot residual is measured at the final prompt token, the ICL task vector, and write it into the weights, at its sparsest into near-silent MLP units keyed to the prompt format. Results reach the accuracy of ICL and of fine-tuning on the same examples, LoRA included, usually leaving perplexity untouched, with worse performance under changes of wording. We also qualify the comparison by counting unfinished runs, observing tha
    
[^77]: 跨语言区别特征对音段三元组的约束：对比离散度的证据

    Cross-Linguistic Distinctive-Feature Constraints on Segmental Triphones: Evidence for Contrastive Dispersion

    [https://arxiv.org/abs/2611.26837](https://arxiv.org/abs/2611.26837)

    该研究通过分析17种语言（涵盖九个语系）中音段三元组的区别特征距离，发现自然语音呈现中等特征距离模式，并相比随机基线展现出更窄的离散度和更低的联合熵，为跨语言的对比离散度约束提供了实证证据。

    

    自然语音反映了发音与感知方面的约束，但这些约束如何塑造跨语言中连续音段之间的过渡仍缺乏充分研究。我们分析了来自17个Universal Dependencies语料库（涵盖九个语系）的、按词频加权的音段三元组中相邻区别特征的距离，语料经IPA转写并使用汉明距离和余弦距离两种度量映射到24个PanPhon特征。自然语音序列呈现中等距离的众数分布（汉明度量下为8-10个特征，越南语除外；余弦度量下为0.28-0.38）。相对于不受限制的词内打乱基线，自然词汇拒绝了仅基于音段库的零假设（W = 0，p < 10^-4），在全部17种语言中均表现出更高的期望距离、更窄的离散度和更低的三元组联合熵。在更严格的词内辅音-元音模板基线下，期望距离和离散度不再表现出分离，但联合熵在16种语言中仍然更低。

    arXiv:2611.26837v1 Announce Type: new  Abstract: Natural speech reflects articulatory and perceptual constraints, yet how these pressures shape consecutive segment transitions across languages remains underexplored. We analyze adjacent distinctive-feature distances in token-weighted segmental triphones across 17 Universal Dependencies corpora spanning nine language families, transcribed into IPA and mapped to 24 PanPhon features using Hamming and cosine metrics. Natural sequences exhibit an intermediate-distance mode (8-10 features under Hamming, except Vietnamese; 0.28-0.38 under cosine). Against an unrestricted within-word shuffled baseline, natural words reject the inventory-only null (W = 0, p < 10-4), showing higher expected distance, narrower dispersion, and lower joint triphone entropy in all 17 languages. Under a stricter within-word consonant-vowel template baseline, expected distance and dispersion are no longer separated, whereas joint entropy remains lower in 16 languages (
    
[^78]: 面向语言模型前馈层的非扩展门控连分数架构

    Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models

    [https://arxiv.org/abs/2611.26835](https://arxiv.org/abs/2611.26835)

    本文提出一种非扩展门控连分数架构组件，可替换语言模型中的各类前馈层（包括MLP、GLU及MoE专家），并用对数和指数重新实现除法运算，从而提升性能并简化与标准预训练流程的集成。

    

    arXiv:2611.26835v1 公告类型：交叉 摘要：最近的研究表明，受连分数启发的神经架构在语言建模中可以做到高度参数高效且性能优异。在本工作中，我们重新审视了基于连分数的通道混合方法，旨在提升其性能，并简化其与语言模型标准预训练协议的集成。我们通过两个简单而关键的贡献来实现这一目标：i) 我们提出了一种新颖的非扩展门控连分数启发的架构组件，用于替换现有语言模型架构中的前馈层，这些架构可以采用普通多层感知机、门控线性单元，或混合专家架构中的（普通/门控）专家。ii) 我们提出了一种用对数和指数重新实现除法运算的方法——除法正是这类架构中的非线性来源。通过在 GPT-2、Llama-3、Mamba- 等不同语言建模架构上进行实验……

    arXiv:2611.26835v1 Announce Type: cross  Abstract: Recent work has shown that continued fraction inspired neural architectures (CoFrGeNets) can be highly parameter efficient yet performant for language modeling. In this work, we revisit continued fraction-based channel mixing with the goal of improving its performance and simplifying integration into standard pre-training protocols for language models. We achieve this by two simple yet critical contributions: i) we propose a novel non-expanding gated continued fraction inspired architectural component to replace feedforward layers in established language model architectures that may use vanilla multilayer perceptrons or gated linear units or (vanilla/gated) experts in mixture-of-experts (MoE) type architectures. ii) We propose a way to re-implement divisions -- the non-linearity in these architectures -- in terms of logarithms and exponentials. By experimenting on different language modeling architectures such as GPT-2, Llama-3, Mamba-
    
[^79]: 重新思考结合领域与任务适配器缩放的全权重LoRA：面向葡萄牙语钻井报告的问答任务

    Rethinking Full-Weight LoRA with Domain and Task Adapter Scaling for Question Answering over Portuguese Well Drilling Reports

    [https://arxiv.org/abs/2611.26833](https://arxiv.org/abs/2611.26833)

    该研究提出在训练后对领域适配与任务特定的LoRA适配器进行缩放、并可用互补权重将二者组合，从而在不损害指令遵循能力的前提下提升大语言模型对葡萄牙语钻井报告问答任务的性能。

    

    将大语言模型（LLM）适配至专业领域仍然具有挑战性，特别是针对葡萄牙语钻井报告这类技术文档，其标注数据稀缺，且领域适配可能损害模型原有的指令遵循能力。尽管低秩适应（LoRA）被广泛用于参数高效的适配，但其在训练后的缩放策略仍未得到充分探索。本研究使用Qwen2.5-7B和Llama-3.1-8B，探究了LoRA适配器缩放对葡萄牙语钻井报告问答任务的影响。领域自适应预训练（DAPT）既可直接在指令微调模型上进行，也可在其基础模型上进行后再进行适配器迁移；而任务特定适配器则在WellQuest上训练，这是一个源自葡萄牙语钻井报告的开源问答数据集。此外，这两种适配器还可通过互补权重进行组合。缩放因子的范围从……

    arXiv:2611.26833v1 Announce Type: cross  Abstract: Adapting large language models (LLMs) to specialized domains remains challenging, particularly for technical documents such as Portuguese well drilling reports, where annotated data are scarce and domain adaptation may compromise instruction-following capabilities. Although Low-Rank Adaptation (LoRA) is widely used for parameter-efficient adaptation, its scaling after training remains insufficiently explored. This study investigates the impact of LoRA adapter scaling on question answering over Portuguese drilling reports using Qwen2.5-7B and Llama-3.1-8B. Domain-adaptive pretraining (DAPT) is performed either directly on instruction-tuned models or on their base counterparts, with subsequent adapter transfer, while task-specific adapters are trained on WellQuest, an open-source question-answering dataset derived from Portuguese drilling reports. The two adapters are also combined using complementary weights. Scaling factors ranging fro
    
[^80]: 神经几何作为学习信号

    Neural Geometry as a Learning Signal

    [https://arxiv.org/abs/2611.26832](https://arxiv.org/abs/2611.26832)

    本文证明只需监督模型内部表示的几何结构（如傅里叶谐波和低维螺旋结构），就能在完全不监督输出的情况下取得与标准微调相当的模型改进效果。

    

    我们研究神经表示的几何结构能否提供一种足够的学习信号，从而在不监督模型输出的情况下改进模型行为。首先，我们通过监督输入月份（如 June = 6）和偏移量（4）的内部表示来微调 Llama-3.1-8B，使其能够回答诸如“六月之后的四个月是几月？”这类问题，这些内部表示以傅里叶谐波的形式编码。其次，我们微调模型来预测按固定宽度换行的文本中的换行符，方法是监督“自上一个换行符以来的字符数”这一内部表示，该表示由一个低维螺旋结构刻画。在这两个任务中，我们都使用理想化的目标结构来监督残差流的一个低维子空间，结果发现内部几何监督能够达到与标准微调相似的性能（尽管完全没有对模型输出进行监督）。此外，我们发现简单地修补 i……（摘要原文在此处被截断）

    arXiv:2611.26832v1 Announce Type: new  Abstract: We investigate whether the geometric structure of neural representations can provide a sufficient learning signal for improving model behavior without any supervision on model outputs. First, we fine-tune Llama-3.1-8B to answer questions such as "What month is four months after June?" by supervising internal representations of the input month (June = 6) and offset (4), which are encoded with Fourier harmonics. Second, we fine-tune the model to predict line breaks in text wrapped to a fixed width by supervising an internal representation of the number of characters since the preceding newline, which is represented by a low-dimensional helical structure. In both tasks, we supervise a low-dimensional subspace of the residual stream using an idealized target structure, and find that internal geometric supervision achieves similar performance to standard fine-tuning (despite no supervision on outputs). Moreover, we find that simply patching i
    
[^81]: 从一开始就学会搜索：面向语言模型的搜索原生预训练

    Search From the Start: Search-Native Pretraining for Language Models

    [https://arxiv.org/abs/2611.26809](https://arxiv.org/abs/2611.26809)

    提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。

    

    现代语言模型（LM）越来越多地在推理阶段配备搜索工具进行部署。然而，预训练仍然主要聚焦于对静态文本的预测，搜索能力通常是在后训练阶段才被引入。我们提出“搜索原生预训练”作为一种实用路径，在预训练阶段就基于普通文本语料教会模型搜索能力。具体而言，我们大规模地用网络搜索查询来增强预训练语料，并使用本地索引检索网络响应。在这种增强语料上的训练使模型学会生成搜索调用，并在语言建模过程中融入检索到的证据。在Wikipedia和DCLM-Edu语料库以及360M至3.3B的参数规模上，搜索预训练模型在知识密集型任务上表现出更高的性能，并且对检索到的证据具有更强的可控性（可引导性）。关键的是，在对搜索预训练模型和常规模型进行监督微调之后，它们的优势依然得以保持。

    arXiv:2611.26809v1 Announce Type: cross  Abstract: Modern language models (LMs) are increasingly deployed with access to search tools at inference. Yet pretraining remains predominantly focused on predicting static text, with search capabilities typically introduced during post-training. We propose Search-Native Pretraining as a practical route to teach search capability during pretraining from ordinary text corpora. Specifically, we augment pretraining corpus at scale with web-search queries and retrieve web response using a local index. Training on this augmented corpus teaches models to generate search calls and incorporate retrieved evidence during language modeling. Across Wikipedia and DCLM-Edu corpora and 360M to 3.3B parameter scales, search-pretrained models ("QueryLM") show improved performance on knowledge-intensive tasks and higher steerability toward the retrieved evidence. Crucially, their advantage persists after supervised fine-tuning of both search-pretrained and conve
    
[^82]: MedicalRSI：通过临床模型-框架协同进化实现医疗递归自我改进智能体

    MedicalRSI: Medical Recursive Self-Improving Agents via Clinical Model-Harness Co-Evolution

    [https://arxiv.org/abs/2611.26806](https://arxiv.org/abs/2611.26806)

    该论文提出MedicalRSI框架，通过临床模型-框架协同进化使医疗智能体能够同时改进自身及其改进策略，从而实现递归自我改进，避免同类错误在不同患者间反复出现。

    

    基于大语言模型（LLM）构建的医疗智能体发展迅速，但大多数智能体在部署后便保持固定，不会持续利用自身经验来改进后续的尝试。因此，同类错误可能会在不同患者身上反复出现。为解决这一问题，现有的自我改进方法会基于智能体自身的经验来更新其权重或医疗框架，但其改进策略本身保持固定，从不学习哪些改变真正有效，这限制了智能体可提升的上限。递归自我改进（RSI）则同时改进智能体本身及其改进策略。然而，为医疗智能体实现RSI具有挑战性，因为临床工作流程很少配有参考框架，且某个能修复早期错误的改变未必能修复智能体后续遇到的问题。我们提出了MedicalRSI，一个建立在临床模型-框架协同进化之上的RSI框架，其中一个模型扮演两个角色。Solver（求解器）是医疗智能体……

    arXiv:2611.26806v1 Announce Type: cross  Abstract: Medical agents built on large language models (LLMs) have advanced rapidly, yet most stay fixed once deployed and do not keep using their experience to improve later attempts. As a result, the same kind of error can recur across patients. To address this, existing self-improvement methods update an agent's weights or its medical harness from its own experience, but their improving strategy stays fixed and never learns which changes actually help, which caps how much the agent can improve. Recursive self-improvement (RSI) instead improves both the agent and its improving strategy. However, achieving RSI for medical agents is challenging, because clinical workflows rarely come with a reference harness, and a change that fixes early errors may not fix the problems the agent meets later. We propose MedicalRSI, an RSI framework built on Clinical Model-Harness Co-Evolution, in which one model plays two roles. The Solver is the medical agent 
    
[^83]: 不要丢弃你的推理轨迹：重用教师强化学习轨迹进行学生模型蒸馏

    Don't Discard Your Rollouts: Reusing Teacher RL Traces for Student Distillation

    [https://arxiv.org/abs/2611.26803](https://arxiv.org/abs/2611.26803)

    提出利用GRPO训练中已记录的元数据（提示通过率和教师对数似然）将推理轨迹重用建模为数据选择问题，通过SFT预热加DPO的分阶段蒸馏策略，把教师的强化学习轨迹高效重用于学生模型训练，避免了昂贵的重新生成数据。

    

    诸如PPO和GRPO等强化学习后训练方法计算成本高昂，大量资源被用于采样推理轨迹，这些轨迹包含每个提示的多个补全及其相应的奖励。然而，传统蒸馏方法会丢弃这些轨迹，并使用更多算力从最终教师模型重新生成数据。我们探究是否以及如何重用保存下来的教师强化学习推理轨迹来蒸馏学生模型。重用推理轨迹并非易事：在推理轨迹上进行朴素的SFT表现不如传统蒸馏，而朴素的DPO虽然性能尚可，但不稳定，在较弱的学生模型上会崩溃。我们证明，利用GRPO训练过程中已记录的元数据，将推理轨迹重用视为一个数据选择问题，可以大幅缩小这一差距。我们使用提示通过率对数据进行排序，先用最简单的提示进行SFT预热启动，然后再用成功率参半的提示进行DPO。对于每个提示，我们使用保存的教师模型对数似然作为选择样本的代理指标。

    arXiv:2611.26803v1 Announce Type: cross  Abstract: RL post-training methods such as PPO and GRPO are computationally expensive, with substantial resources devoted to sampling rollouts, which contain multiple completions per prompt along with their respective rewards. Canonical distillation, however, discards these traces and uses more compute to regenerate data from the final teacher. We ask if and how we can reuse the saved teacher RL rollouts to distil student models. Reusing rollouts is non-trivial: naive SFT on the rollouts underperforms canonical distillation, and naive DPO, while performant, is unstable, collapsing on weaker students. We show that treating rollout reuse as a data selection problem using metadata already recorded during GRPO substantially reduces this gap. We use prompt pass rates to order the data, using the easiest prompts for an SFT warm start prior to using mixed-success prompts for DPO. For each prompt, we use saved teacher log-likelihoods as a proxy for sele
    
[^84]: 什么能帮助语言模型代理审阅暴力视频？与人类证据参照的受控比较

    What Helps Language-Model Agents Review Violent Video? A Controlled Comparison with a Human Evidence Reference

    [https://arxiv.org/abs/2611.26776](https://arxiv.org/abs/2611.26776)

    该研究在XD-Violence数据集上进行受控比较，发现将感知模型观察结果整理为一次性阅读的扁平文本是语言模型代理审阅暴力视频最有效的方式，其表现与直接处理原始视频的单一视听模型相当，而消息传递和证据图接口效果较差。

    

    摘要（arXiv:2611.26776v1，公告类型：交叉发布）：如今，无需训练即可基于语言模型构建用于标记暴力视频以供人工操作员审核的系统，方法是组合多个感知模型并让语言模型做出决策。我们探讨感知模型的观察结果应以何种方式传递给决策环节，以及哪些证据真正重要。在XD-Violence测试集（799个视频片段）上，我们固定了一组感知代理（视频描述、物体检测、声音标记、语音识别和语音标注）和一个推理模型，并比较三种接口：代理之间的消息传递、通过工具探索的类型化证据图，以及一次性阅读的扁平文本；此外还比较了一个直接接收原始视频片段的单一视听模型。两名标注者提供了人类参照，记录了他们的决策、事件发生的片段，以及他们依赖的是视觉、声音还是语音。结果表明，扁平文本与单一模型的表现相当（F1为0.967对0.968），而证据图代理和消息传递的准确度较低；证据图代理的一个变体……

    arXiv:2611.26776v1 Announce Type: cross  Abstract: Systems that flag violent video for a human operator can now be built from language models without training, by combining perception models and letting a language model decide. We ask how the observations of the perception models should reach the decision, and which evidence matters. On the XD-Violence test set (799 clips) we fix one set of perception agents (video description, object detection, sound tagging, speech recognition and speech tagging) and one reasoning model, and compare three interfaces: messages between agents, a typed evidence graph explored with tools, and flat text read in one pass; and a single audio-visual model given the raw clip. Two annotators provide a human reference that records the decision, the incident segments and whether they relied on sight, sound or speech. Flat text matches the single model (F1 0.967 against 0.968), while the graph agent and message passing are less accurate; a variant of the graph ag
    
[^85]: 了解你的上下文库：修复上下文感知文本到SQL中的不一致性并度量泛化能力

    Know Your Context Bank: Repairing Inconsistency and Measuring Generalization in Context-Aware Text-to-SQL

    [https://arxiv.org/abs/2611.26775](https://arxiv.org/abs/2611.26775)

    该论文提出“上下文库腐化”概念，指出企业Text-to-SQL上下文库中各条目虽分别通过审核却存在相互矛盾的不一致性问题，并设计了从人工校准标注蒸馏而来的快速评分器，通过逐决策比较查询来修复不一致性，同时度量系统是基于上下文真正泛化还是仅机械复现。

    

    文本到SQL系统越来越依赖上下文库：这是经过精心整理的示例查询、指标定义和领域知识的集合，用于告诉模型一个组织是如何编写其查询的。在企业环境中，上下文库随着多年间众多分析师添加条目而不断增长，每个条目只被单独检查其正确性（该SQL能否回答其自身的问题？），却很少检查条目彼此之间的一致性。然而，正确性的定义往往很宽松：一个问题很少明确指定实体应如何计数或由哪张表来定义，因此两个条目可以各自通过审核，却以相反的方式解决同一个业务决策。我们将这种现象称为“上下文库腐化”。随之而来的第二个问题是：使用上下文库的系统究竟是从中泛化，还是仅仅复现它？现有基准无法回答这一问题，因为它们没有控制测试问题与上下文之间的距离。我们通过逐个决策地比较查询来回答这两个问题，使用一个从人工校准标注中蒸馏得到的快速评分器……

    arXiv:2611.26775v1 Announce Type: new  Abstract: Text-to-SQL systems increasingly rely on context banks: curated collections of example queries, metric definitions, and domain knowledge that tell a model how an organization writes its queries. In the enterprise, a bank grows over years as many analysts add entries, checked one at a time for correctness (does the SQL answer its own question?) and rarely for consistency with one another. Yet correctness is often loosely defined: a question seldom fixes how an entity is counted or which table defines it, so two entries can each pass review yet resolve the same decision in opposite ways. We call this condition context bank rot. A second question follows: does a system that uses a bank generalize from it, or only reproduce it? Existing benchmarks cannot tell, as they do not control how far a test question sits from the context. We answer both questions by comparing queries decision by decision, with a fast scorer distilled from human-calibr
    
[^86]: 强化学习能够“遗忘”受损的SFT推理

    Reinforcement Learning Unlearns Corrupted SFT Reasoning

    [https://arxiv.org/abs/2611.26756](https://arxiv.org/abs/2611.26756)

    强化学习后训练（GRPO）能够恢复被严重损坏的SFT数据造成的性能下降，仅用8条错误的“仅形式”轨迹初始化RL甚至可以超越黄金SFT初始化的表现，且SFT准确率与RL后准确率基本不相关。

    

    我们证明，即使监督微调（SFT）数据严重受损，推理任务中的强化学习（RL）仍然可以取得成功。在数学推理数据集GSM8K上，我们通过注入真实的错误类型，在361种配置中系统性地破坏训练轨迹，改变因素包括错误位置、轨迹与答案的一致性以及高达100%的损坏比例。虽然在数据损坏情况下仅使用SFT的性能持续恶化（从43.61%降至13.80%），但群体相对策略优化（GRPO）后训练能够持续将模型准确率恢复到黄金基线水平（约60%）。至关重要的是，仅用N=8条最小化的、错误的、三行的“仅形式”轨迹来初始化RL，就能将RL后训练性能提升至69.30%，优于标准黄金SFT初始化（60.71%）和无SFT的GRPO（35.00%）。此外，我们发现SFT准确率与RL后训练准确率基本不相关（r approxi

    arXiv:2611.26756v1 Announce Type: new  Abstract: We show that reinforcement learning (RL) in reasoning tasks can succeed even when the supervised fine-tuning (SFT) data is severely corrupted. In a mathematical reasoning dataset (GSM8K), we systematically corrupt training trajectories across 361 configurations by injecting realistic error types, varying error locations, trace-answer consistency, and corruption ratios up to 100%. While aggregated SFT-only performance steadily deteriorates under corruption (dropping from 43.61% to 13.80%), Group Relative Policy Optimization (GRPO) post-training consistently restores model accuracy back to gold-baseline levels (approximately 60%). Crucially, initializing RL with as few as N = 8 minimal, incorrect, three-line "form-only" traces boosts post-RL performance to 69.30%, outperforming standard gold SFT initialization (60.71%) and GRPO without SFT (35.00%). Further, we show that SFT accuracy is largely uncorrelated with post-RL accuracy (r approxi
    
[^87]: SkillPrivus：揭示智能体技能中的使用隐私危害

    SkillPrivus: Uncovering PRIVacy of USe Harms in Agentic Skills

    [https://arxiv.org/abs/2611.26747](https://arxiv.org/abs/2611.26747)

    SkillPrivus 是一个无需执行即可在安装前对智能体技能进行静态隐私审计的框架，对 726 个真实技能的分析发现超过半数（54.5%）存在潜在隐私违规。

    

    arXiv:2611.26747v1 公告类型：新论文 摘要：智能体技能通过可安装的指令和代码来扩展 AI 助手，这些指令和代码能够访问用户的文件、账户和数据。即使技能在执行其宣称的任务时，也可能推断出敏感属性、保留不必要的信息，或向意料之外的接收方披露数据。我们通过 SkillPrivus 研究这些潜在的“使用隐私”危害——这是一个静态审计框架，可以在不执行技能的情况下于安装前对其进行分析。SkillPrivus 从代码和自然语言指令中提取数据操作，将其锚定于源证据，并引入情境完整性理论来评估其适当性。将该框架应用于来自公开市场的 726 个真实技能（其中 563 个涉及个人数据处理），SkillPrivus 在其中 307 个技能（54.5%）中标记出至少一个潜在违规行为，其判断在 83.2% 的可判定数据操作上与三位盲评人类标注者一致。被标记的行为包括敏感推断、信息使用……（原文摘要在此处截断）

    arXiv:2611.26747v1 Announce Type: new  Abstract: Agent skills extend AI assistants through installable instructions and code that can access users' files, accounts, and data. Even when performing their advertised tasks, skills may infer sensitive attributes, retain unnecessary information, or disclose data to unexpected recipients. We study these potential privacy-of-use harms through SkillPrivus, a static audit framework that analyzes skills before installation without executing them. SkillPrivus extracts data operations from code and natural-language instructions, grounds them in source evidence, and adapts Contextual Integrity to assess their appropriateness. Applied to 726 real skills from public marketplaces, 563 of which handle personal data, SkillPrivus flags at least one potential violation in 307 of these skills (54.5%), and its judgments agree with three blind human annotators on 83.2% of resolvable moves. The flagged behaviors include sensitive inferences, information use be
    
[^88]: TOUMAII：一个包含450万偏好对的大规模工具调用偏好数据集及其源无关生成器

    TOUMAII: A 4.5M-Pair Tool-Calling Preference Dataset and Its Source-Agnostic Generator

    [https://arxiv.org/abs/2611.26722](https://arxiv.org/abs/2611.26722)

    该论文提出了包含约450万条经验证偏好对的大规模工具调用偏好数据集TOUMAII及其源无关的偏好对生成器，以缓解工具调用领域（尤其是多轮场景）偏好数据的稀缺，从而支持DPO训练出更可靠的工具调用能力。

    

    为了自主执行多步骤任务，企业级智能体必须可靠地与工具交互、调用API或查询数据库，这使得工具调用成为AI模型的一项基础能力。监督微调（SFT）只优化生成正确的调用，却从未将其与看似合理的错误调用进行对比。在一个即使微小的格式错误也可能使调用失效、错误决策可能导致整个交互脱轨的环境中，直接偏好优化（DPO）恰恰针对这种对比，因此DPO需要偏好对而非简单的演示样本。然而对于工具调用而言，此类数据，尤其是在现实的多轮交互场景中，仍然极其稀缺。为解决这一稀缺性问题，我们提出了TOUMAII，一个大规模工具调用偏好数据集，包含约450万条经过验证的偏好对，这些数据源自Agent-Ark/Toucan-1.5M（apache-2.0许可）对话。该数据集由多轮和单轮交互组成，并且……

    arXiv:2611.26722v1 Announce Type: new  Abstract: To autonomously execute multi-step tasks, enterprise agents must reliably interact with tools, call APIs, or query databases, making tool-calling a fundamental capability of AI models. Supervised fine-tuning (SFT) optimises for producing the correct call, but never contrasts it with a plausible incorrect one. In an environment where even a minor formatting error can invalidate a call and an incorrect decision can derail the interaction, direct preference optimisation (DPO) targets exactly this contrast. DPO, therefore, requires preference pairs rather than plain demonstrations. For tool-calling, such data, especially in realistic multi-turn settings, remains extremely scarce. To address this scarcity, we introduce TOUMAII, a large-scale tool-calling preference dataset containing ~4.5M validated preference pairs, derived from Agent-Ark/Toucan-1.5M (apache-2.0) conversations. The dataset consists of multi- and single-turn interactions and 
    
[^89]: 再添一个智能体：AI团队规模与组建方式如何影响人机头脑风暴团队的生产力与多样性

    Adding One More Agent: How AI Team Size and Formation Shape Productivity and Diversity in Human-AI Brainstorming Teams

    [https://arxiv.org/abs/2611.26663](https://arxiv.org/abs/2611.26663)

    在人机头脑风暴实验中，双AI团队虽提高了创意总量却大幅压缩了人类贡献，而单个“平均人类行为校准”的AI反而最能激发人类创意产出，表明AI伙伴数量与组建方式共同重塑了团队的生产力与多样性。

    

    生成式AI已经提升了个人生产力，但增加多个AI伙伴究竟是利还是弊仍未有定论。我们在一项头脑风暴实验（N=168）中，将AI伙伴数量（一个与两个）与行为组建方式（校准为平均人类行为 vs 由参与者定制）进行交叉设计，并以人人二人组作为参照。结果显示：双AI房间在共享看板上发布了更多创意，但人类原创创意数量大幅下降。当只有一个AI伙伴时，组建方式至关重要：与平均行为校准AI配对的人类发布的创意多于任何其他条件；而当有两个AI时，这一效应消失。人类仍在持续交谈，但将创意转化为看板帖子的比例降低。AI的社交性对话预示人类发帖更多，而AI发帖量则预示人类发帖更少。AI的存在提升了人类原创创意的全局多样性，但只有引入第二个AI才能增加会话内部的差异化。AI伙伴的数量与组建方式能够重塑……

    arXiv:2611.26663v1 Announce Type: new  Abstract: Generative AI has improved individual productivity, but whether adding a plurality of AI partners helps or hurts remains open. We crossed AI partnership (one versus two) with behavioral formation (calibrated to average human behavior versus customized by the participant) in a brainstorming experiment ($N$=168) with human-human dyads as reference. Two-AI rooms posted more ideas to the shared board, but human-authored ideas dropped substantially. With one AI partner, formation mattered: humans paired with an average-calibrated AI posted more than any other condition. With two, this effect vanished. Humans continued talking but converted fewer ideas into board posts. AI social talk predicted more human posts, while AI posting volume predicted fewer. AI presence increased global diversity of human-authored ideas, but only a second AI increased within-session differentiation. The number of AI partners and their formation can reshape what the 
    
[^90]: VerifiQuant：面向金融大语言模型工作流的“执行即治理”框架

    VerifiQuant: Governance-by-Execution for Financial LLM Workflows

    [https://arxiv.org/abs/2611.26649](https://arxiv.org/abs/2611.26649)

    VerifiQuant提出“执行即治理”框架，通过将金融逻辑封装为推理契约并用可执行诊断漏斗在计算前评估每个查询，使金融大模型能够结构化、可审计地区分计算、澄清与拒绝，从而在输入不支撑数值答案时实现更安全的行为。

    

    嵌入金融工作流的大语言模型最危险的失效方式，是输出自信、格式规范但错误的数字，且没有任何信号能将其与正确结果区分开。我们认为，核心的部署问题在于可治理性：系统必须能够判断一个问题何时支持给出数值答案，何时应当转而进行澄清、拦截或拒绝。我们提出VerifiQuant，一个“执行即治理”框架，它将金融逻辑封装为推理契约，并在确定性计算之前通过可执行的诊断漏斗对每个查询进行评估。这种分离使计算、澄清和拒绝变得结构化且可审计，同时将金融算术运算保留在语言模型之外。在声明的契约库范围内，与匹配的检索-执行基线相比，其主要收益并非在有明确答案的问题上获得更高的准确率，而是当输入不足以支持数值答案时表现出更安全的行为。

    arXiv:2611.26649v1 Announce Type: cross  Abstract: Large language models embedded in financial workflows fail most dangerously by emitting confident, well-formatted but wrong numbers with no signal distinguishing them from correct ones. We argue that the central deployment problem is governability: a system must determine when a question supports a numeric answer and when it should instead clarify, intercept, or refuse. We present VerifiQuant, a governance-by-execution framework that packages financial logic into inference contracts and evaluates each query through an executable diagnostic funnel before deterministic computation. This separation makes computation, clarification, and refusal structured and auditable while keeping financial arithmetic outside the language model. Against matched retrieval-and-execution baselines, and within a declared contract library, the principal benefit is not higher accuracy on well-posed questions but safer behavior when an input does not justify a 
    
[^91]: SDG：面向线性注意力的分块边界状态依赖门控

    SDG: State-Dependent Gating at Chunk Boundaries for Linear Attention

    [https://arxiv.org/abs/2611.26637](https://arxiv.org/abs/2611.26637)

    提出SDG方法，通过在分块边界处依据记忆状态的变化计算逐元素门控，为线性注意力模型提供状态反馈的遗忘机制，同时保持块内并行计算的高效性。

    

    近期的门控线性注意力模型使用状态无关的门控机制，该门控由token表示计算得出，而不直接读取循环记忆。这种设计支持高效的分块计算：序列被划分为连续token组成的块，并在每个块内进行并行计算。然而，即使层中存储的信息不同，相同的token表示仍会产生相同的门控。我们提出了状态依赖门控（SDG），它根据记忆在一个块上的变化计算逐元素门控，并将其应用于块的边界。这在保持每个块内并行计算的同时，为遗忘机制提供了状态反馈。我们为Gated DeltaNet、Kimi Delta Attention、Gated DeltaNet-2和Mamba-3开发了融合内核。我们的理论分析在信号-噪声模型下量化了状态依赖的逐元素门控在重建方面的潜在收益。我们还表明，边界衰减（摘要在此处被截断）

    arXiv:2611.26637v1 Announce Type: cross  Abstract: Recent gated linear attention models use state-independent gates, computed from token representations without directly reading the recurrent memory. This design supports efficient chunkwise computation: sequences are divided into chunks of consecutive tokens, with parallel computation within each chunk. However, the same token representation produces the same gate, even when the layer stores different information. We introduce State-Dependent Gating (SDG), which computes element-wise gates from the change in memory over a chunk and applies them at the chunk boundary. This provides state feedback for forgetting while preserving parallel computation within each chunk. We develop fused kernels for Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2, and Mamba-3. Our theoretical analysis quantifies the potential reconstruction benefits of state-dependent, element-wise gating under a signal--noise model. We also show that boundary decay 
    
[^92]: 有所表征却被误读：后训练改变了语言模型读取其内部表征的方式

    Represented but misread: Post-training changes how language models read out what they represent

    [https://arxiv.org/abs/2611.26584](https://arxiv.org/abs/2611.26584)

    研究发现后训练并未破坏语言模型内部的表征（线性探针仍能准确区分人类与AI文本），而是改变了模型将表征读取为答案的方式，导致回答灵敏度下降和对先验信息的错误利用。

    

    后训练究竟改变了语言模型所表征的内容，还是改变了将表征读取为答案的方式？我们在六个开源权重的基础模型及其后训练版本上，将线性探针与模型自身的回答进行了比较，并采用信号检测理论进行分析。在判断德语文本由人类还是AI撰写的任务中，线性探针和词n-gram分类器都能很好地区分类别，但模型的回答却不能，而且常常以相反的顺序排列文本。后训练平均降低了回答的灵敏度（跨模型家族Δd' = -0.34；跨三个提示词为-0.29）。它还改变了对所陈述基础比率的利用：基础模型对数几率的变化小于所陈述的先验几率变化，而大多数后训练模型的变化则更大。在两个具有阶段性检查点的模型中，这一变化发生在监督微调阶段；模型拼接实验表明相关机制位于早期层，且从中间深度开始，内部状态……（摘要在此处截断）

    arXiv:2611.26584v1 Announce Type: new  Abstract: Does post-training change what a language model represents, or how this representation is read out into an answer? We compared linear probes with the models' own answers, analyzed with signal detection theory, in six open-weight base models and their post-trained versions. For judgments of whether German texts were written by a human or an AI, probes and word n-gram classifiers separated the classes well, but the answers did not and often ordered the texts inversely. Post-training lowered the sensitivity of the answers on average ($\Delta d' = -0.34$ across model families; -0.29 across three prompts). It also changed the use of a stated base rate: Base models shifted their log-odds by less than the stated change in prior odds, most post-trained models by more. In two models with stage-wise checkpoints, the change occurred during supervised fine-tuning, model stitching implicated the early layers, and from mid-depth on internal states at 
    
[^93]: AutoMaster：通过让同一模型成为更好的教师来实现自主后训练

    AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher

    [https://arxiv.org/abs/2611.26510](https://arxiv.org/abs/2611.26510)

    该论文提出了Skill-FT技能引导微调方法和AutoMaster自主后训练系统，由LLM智能体自动分析教师轨迹并设计自然语言技能干预来抑制不良习惯、激发有效行为，从而在多个任务上比朴素SFT提升1.42-8.99个百分点的性能。

    

    仅凭任务成功并不能确定后训练应该强化哪些行为。即使是成功的教师轨迹也可能包含缺乏依据的假设、无效的重复或肤浅的验证。我们提出了Skill-FT（技能引导微调），该方法使用自然语言技能在教师模型 rollout 过程中抑制不良习惯并激发代表性不足的有效行为，然后在由此产生的轨迹上对目标模型进行微调。我们介绍了AutoMaster，这是一个自主后训练系统，其中一个LLM智能体通过轨迹分析、技能设计、数据选择以及来自 rollout 和目标模型评估的反馈来开发这些干预措施。在使用4B和9B目标模型的终端（Terminal）、搜索（Search）和软件工程任务中，相较于在成功且无引导的教师轨迹上进行朴素SFT，AutoMaster将平均性能提升了1.42-8.99个百分点，且教师模型、任务池、训练集规模（摘要在此处截断）

    arXiv:2611.26510v1 Announce Type: new  Abstract: Task success alone does not specify which behaviors post-training should reinforce. Even successful teacher trajectories can contain unsupported assumptions, unproductive repetition, or superficial verification. We propose Skill-FT (Skill-Guided Fine-Tuning), which uses natural-language skills to suppress undesirable habits and elicit underrepresented effective behaviors during teacher rollouts, then fine-tunes a target model on the resulting trajectories. We introduce AutoMaster, an autonomous post-training system in which an LLM agent develops these interventions through trajectory analysis, skill design, data selection, and feedback from rollouts and target-model evaluation. Across Terminal, Search, and software engineering tasks with 4B and 9B target models, AutoMaster improves mean performance over Vanilla SFT on successful unguided teacher trajectories by 1.42-8.99 percentage points, with the teacher, task pool, training set size, 
    
[^94]: 依存条件化词嵌入的语言结构

    The Linguistic Structure of Dependency-Conditioned Embeddings

    [https://arxiv.org/abs/2611.25909](https://arxiv.org/abs/2611.25909)

    该论文通过分析全局与局部几何结构，系统考察了依存条件化词嵌入的语言组织方式，发现全局几何主要由上下文词性塑造，而局部词汇相似性结构在依存关系间得到部分保留。

    

    基于依存的分布模型能够捕捉受句法信息引导的词汇相似性，但依存条件化词汇表示的几何组织方式仍未得到充分探索。我们使用在WikiText-103上训练的四种表示来研究词元-依存关系嵌入的全局与局部几何结构：两种基于计数的PPMI配置和两种带负采样的Skip-Gram（SGNS）配置。在全局层面，我们分析依存关系的质心，通过词元内中心化来控制词汇身份的影响，并考察其与词性（POS）分布的对齐情况。在局部层面，我们评估了跨关系的几何一致性、跨方法的一致性、词汇相似性以及多义词的不同用法。结果表明，在所有表示中，全局几何结构都受到上下文词性的强烈塑造。在局部层面，更广泛的词汇相似性结构在依存关系间得到了部分保留。

    arXiv:2611.25909v1 Announce Type: new  Abstract: Dependency-based distributional models capture syntactically informed lexical similarities, but the geometric organisation of dependency-conditioned lexical representations remains underexplored. We investigate the global and local geometry of lemma-dependency relation embeddings using four representations trained on WikiText-103: two count-based PPMI configurations and two Skip-Gram with Negative Sampling (SGNS) configurations. At the global level, we analyse dependency-relation centroids, control for lexical identity through within-lemma centring, and examine their alignment with part-of-speech (POS) distributions. At the local level, we evaluate cross-relation geometric consistency, cross-method agreement, lexical similarity, and polysemous usages. Results show that global geometry is structured strongly by context POS across all representations. Locally, broader lexical similarity structures are partially preserved across dependency 
    
[^95]: STARK：迈向真正会学习与记忆的个性化智能体

    STARK: Moving towards Personalized Agents That Truly Learn and Remember

    [https://arxiv.org/abs/2611.25705](https://arxiv.org/abs/2611.25705)

    STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。

    

    长期记忆对于在长时间内与用户交互的个性化智能体系统至关重要，然而当前的架构只能捕获“发生了什么”，而将“为什么会发生”的推理留给底层大语言模型事后推导。这些事件不仅能够传达用户经历了什么，还能传达这些经历如何塑造了用户的行为模式。我们提出了STARK，一种双图记忆架构，它将这些事件不仅视为节点，还视为行为洞察之间的边。STARK维护一个情景图，将交互记录为在时间、因果和语义上相互关联的情景；以及一个洞察图，用于表示由大语言模型提取的行为信号，并根据支持这些信号的事件数量进行评分，同时通过引发这些洞察的事件将它们相互连接。在推理时，行为知识可以直接从由事件作为边所支撑的洞察节点中检索获得。

    arXiv:2611.25705v1 Announce Type: cross  Abstract: Long-term memory is essential for personalized agentic systems that interact with users over long time periods, yet current architectures only capture what happened while leaving the reasoning of why it happened to be derived later by underlying LLMs.%, and silo knowledge within individual users. These events can, however, convey not only what a user experienced but also how that experience shaped their behavioral patterns. We introduce STARK, a dual-graph memory architecture which treats these events not just as nodes but also as edges between behavioral insights. STARK maintains an Episodic Graph which records interactions as temporally, causally, and semantically linked episodes; and an Insight Graph which represents LLM-extracted behavioral signals scored by how many events support them, connected by the very events that led to them. At inference, behavioral knowledge is retrieved directly from insight nodes grounded by event-as-ed
    
[^96]: 借力本地模型：滥用浏览器内嵌语言模型进行后渗透攻击

    Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation

    [https://arxiv.org/abs/2611.25086](https://arxiv.org/abs/2611.25086)

    该研究首次揭示了攻击者可以将浏览器预装的本地语言模型（如Chrome中的Gemini Nano）滥用于后渗透攻击阶段，仅借助操作系统自带工具即可驱动其窃取凭据、挖掘敏感信息，且全程躲避网络监控检测。

    

    浏览器如今向普通终端设备预装本地语言模型。Google在Chrome中集成了Gemini Nano以支持摘要生成等生产力功能，其推理完全在设备端运行。这些浏览器内嵌的模型为攻击者提供了新的“就地取材”攻击机会——每个模型实例都是一个潜在的后渗透攻击引擎，它已经预装在设备上、经过厂商签名、且对网络监控完全不可见。Gemini模型具备多模态能力，能够处理文件、分析截图或与攻击者对话。我们证明，仅使用操作系统自带的解释器即可在主机上驱动该模型，并发现模型下载前的一次性用户激活检查很容易通过模拟点击满足，甚至可以被完全绕过。我们测量并报告了当前模型在识别主机上的凭据、检测电子邮件中的个人身份信息（PII）、分类图像以及调用自定义工具方面的能力。我们结合了……

    arXiv:2611.25086v1 Announce Type: cross  Abstract: Browsers now ship local language models to ordinary endpoints. Google includes Gemini Nano in Chrome to enable productivity features such as summarization, with inference running entirely on the device. These browser-embedded models provide new living-off-the-land opportunities for attackers, each instance a potential post-exploitation engine already installed, signed by the vendor, and invisible to network monitoring. The Gemini model is multimodal, capable of processing files, analyzing screenshots, or conversing with an attacker. We show that it can be driven from the host using only interpreters that ship with the operating system and find that the one-time user-activation check before the model is downloaded is easily satisfied with a simulated click or bypassed entirely.We measure and report the current model's ability to identify credentials on the host, detect PII in email, classify images, and call custom tools. We combine the
    
[^97]: 安全个人RSI：评估与修复个人智能体更新

    Safe Personal RSI: Evaluating and Repairing Personal-Agent Updates

    [https://arxiv.org/abs/2611.23651](https://arxiv.org/abs/2611.23651)

    本文提出Safe Personal RSI更新评估机制，通过将递归自我改进限制在提示词层面、固定模型权重与代码，并建立“测试—修复—验证”流程，在29个案例的基准上成功拒绝了21次高风险更新尝试中的20次，保障个人智能体更新后的安全性与任务性能。

    

    Kicker是一个个人健康助手。对助手可复用指令的更改可以在改进任务处理的同时，改变它所披露的内容、它遵循的请求以及它报告操作的方式。我们研究了Safe Personal RSI，一个为Kicker开发的更新评估组件。我们将递归自我改进（RSI）限制在提示词更新的范围内，模型权重和运行时代码保持固定。该流程测试提议的指令与当前指令，重复失败的案例，临时撤销更改以指导修复，并在采纳前检查安全性和任务性能。如果没有版本通过，则继续使用当前指令。我们的可执行基准测试包含29个电子邮件、日历、健康和市场案例，并设有针对授权复用、记忆和工具输出注入的独立探测。在三个模型和两次故意设置的风险编辑中，该流程拒绝了21次更新尝试中的20次。在修复确认中，一个被接纳的GPT-5.6 Luna rep（原文摘要在此处截断）

    arXiv:2611.23651v1 Announce Type: cross  Abstract: Kicker is a personal health assistant. Changes to an assistant's reusable instructions can improve task handling while changing what it discloses, which requests it follows, and how it reports actions. We study Safe Personal RSI, an update-evaluation component developed for Kicker. We restrict recursive self-improvement (RSI) to prompt updates, with model weights and runtime code fixed. The procedure tests proposed and current instructions, repeats failing cases, temporarily undoes changes to guide repairs, and checks safety and task performance before adoption. If no version passes, the current instructions stay in use. Our executable benchmark contains 29 email, calendar, health, and marketplace cases, with separate probes for authorization reuse, memory, and tool-output injection. Across three models and two deliberately risky edits, the procedure rejects 20 of 21 update attempts. In repair confirmation, an admitted GPT-5.6 Luna rep
    
[^98]: 将语言模型与语音结合并不能消除“AI味”：校准的LLM评判器、古德哈特陷阱与一种基于偏好调优的修复方案

    Grounding Language Models in Speech Does Not Fix "AI Flavor": Calibrated LLM Judges, Goodhart Traps, and a Preference-Tuning Fix

    [https://arxiv.org/abs/2611.19840](https://arxiv.org/abs/2611.19840)

    该论文通过在0.6B模型上的受控实验证明，注入语音或韵律信号的三种方法（音素交错续训、韵律计划条件生成、基于TTS韵律距离的规则奖励）均无法消除LLM中文输出的“AI味”，并指出当前n≈20的单评分员盲评只能检测极端偏好的评估方法是根本症结，同时提出一种基于偏好调优的修复方案。

    

    大语言模型的中文输出带有可感知的“AI味”：模板化的句式框架、堆叠的书面语连接词、重复循环以及排版泄漏。一种流行的假设将其归咎于纯文本训练缺失了人类写作中的“内心语音”层，因此注入语音或韵律信号应当能减少这种味道。我们在一个0.6B参数模型上通过受控干预对这一假设进行了测试，三种方法均告失败：(i) 音素交错续训在文本匹配对照下没有带来口语化提升，反而增加了重复度（+0.13 [+0.03, +0.23]）；(ii) 韵律计划条件生成的遵循度仅达到0.257，远低于预先注册的0.50门槛；(iii) 基于TTS韵律特征距离的规则奖励在人类（8/16）和校准后的LLM评判器（9/16）的双盲评估中均无法被感知。其根本原因在于评估环节：样本量约20的单评分员盲评只能检测出极端（≥85%）的偏好差异。

    arXiv:2611.19840v1 Announce Type: new  Abstract: Chinese outputs of large language models carry a perceptible "AI flavor": formulaic sentence frames, stacked written-register connectives, repetition loops, and layout leakage. A popular hypothesis blames text-only training for the missing inner-speech layer of human writing; injecting speech or prosody signals should therefore reduce it. We test it with controlled interventions on a 0.6B model; all three fail: (i) phoneme-interleaved continued pretraining yields no spokenness gain under text-matched controls while raising repetition (+0.13 [+0.03, +0.23]); (ii) prosody-plan conditional generation reaches an adherence of only 0.257 against a pre-registered 0.50 gate; (iii) a rule-based reward built on TTS prosody-feature distance is imperceptible in double-blind evaluation by both humans (8/16) and a calibrated LLM judge (9/16).   A root cause is evaluation: single-rater blind evaluation at n~20 only detects extreme (>=85%) preferences. 
    
[^99]: 形式语义学的形式化：形式图谱

    Formalizing Formal Semantics: The Formal Atlas

    [https://arxiv.org/abs/2611.19509](https://arxiv.org/abs/2611.19509)

    本文提出了基于证明助手的“形式图谱”系统，通过交互式应用将语义学文献与形式化证明连接起来，实现语义理论的形式化、预测检验与比较，并探索AI辅助开发与推理的应用前景。

    

    在本文中，我们提出了形式图谱，这是一个使用证明助手研究自然语言语义的系统。我们的目标是形式化语义分析、检验其预测并比较所实现的理论。为此，我们通过一个交互式应用程序将文献来源记录与形式化工作连接起来，研究人员可以在其中查看证明并修改受支持的 Rocq 文件。蒙塔古的 PTQ 与 Gallin 的著作为我们提供了主要案例，随后是动态语义与类型论语义的比较、RSA 以及分布式语义。我们还讨论了形式化那些不够显式的分析所需的条件。最后，我们探讨了人工智能的应用，既可用于辅助开发这些资源，也可用于能够利用这些资源的推理系统。这项工作扩展了早期证明助手语义学的研究计划，研究与教学是其互补的用途。

    arXiv:2611.19509v1 Announce Type: new  Abstract: In this paper we present the Formal Atlas, a system for the study of natural language semantics using proof assistants. Our aim is to formalize semantic accounts,   check their predictions and compare the implemented theories. To this end, we connect source records and formal developments with an interactive application, where   researchers can inspect proofs and modify supported Rocq files. Montague's PTQ and Gallin provide our main cases, followed by dynamic and type-theoretic   comparisons, RSA and distributional semantics. We also discuss what it takes to formalize less explicit analyses. Lastly, we consider the use of AI, both for   assistance in developing these resources and for inference systems that can make use of them. This work extends an earlier programme in proof-assistant semantics,   with research and teaching as complementary uses.
    
[^100]: MarineSparkLM-4B：基于参数高效适配构建的海洋与离岸工程轻量级语言模型

    MarineSparkLM-4B: A Lightweight Language Model for Marine and Offshore Engineering Built by Parameter-Efficient Adaptation

    [https://arxiv.org/abs/2611.19151](https://arxiv.org/abs/2611.19151)

    该研究通过在一台工作站级设备上仅对40亿参数模型进行0.81%参数量的LoRA高效适配，并以约5小时的低算力成本，构建出面向海洋与离岸工程领域的轻量级专用语言模型MarineSparkLM-4B。

    

    海洋与离岸工程领域的文本散布于预印本、开放获取期刊、百科全书和机构文档之中，且按预训练标准衡量其规模很小。我们探究的问题是：依靠参数高效适配，在一台工作站级单机上，面向海洋与离岸工程的轻量级语言模型究竟能走多远。从40亿参数的OceanGPT-basic-4B-Instruct出发，我们训练了两个独立的LoRA适配器，每个适配器仅更新40.6亿参数中的3300万（即0.81%）。在单台NVIDIA DGX Spark上，第一个适配器训练耗时4小时40分钟，第二个仅耗时10分钟。第一个适配器通过下一词元预测任务，在覆盖十个工程主题、共6160万词元的语料库上训练。该语料库由13,879份源文档构建，经去重和去除跨领域词汇冲突的两级关键词过滤后保留了7,069份文档。其验证损失……（原文在此截断）

    arXiv:2611.19151v1 Announce Type: cross  Abstract: Marine and offshore engineering text is spread across preprints, open-access journals, encyclopedias and institutional documents, and the amount of it is small by pretraining standards. We ask how far a lightweight language model for marine and offshore engineering can be taken with parameter-efficient adaptation on a single workstation-class device. Starting from the 4-billion-parameter OceanGPT-basic-4B-Instruct, we train two independent LoRA adapters, each updating 33.0 million of 4.06 billion parameters (0.81%). Training took 4 hours 40 minutes for the first adapter and 10 minutes for the second on one NVIDIA DGX Spark. The first adapter is trained by next-token prediction on a 61.6-million-token corpus covering ten engineering themes. The corpus was built from 13,879 source documents, of which 7,069 were retained after deduplication and a two-tier keyword filter that removes cross-domain lexical collisions. Its validation loss fal
    
[^101]: 当评估在临床AI中变成干预

    When evaluation becomes intervention in clinical AI

    [https://arxiv.org/abs/2611.18297](https://arxiv.org/abs/2611.18297)

    临床AI中的AI评判者在被用于选择提示词或训练模型时会本身成为干预的一部分，并可能偏向其忽视的错误，因此评判者报告必须包含特定内容，且每项由评判者驱动的变更都需要独立的临床测试。

    

    AI评判者（AI judges）是一种依据评分标准对另一个系统的输出进行打分的模型，它将临床审查的规模扩展到了临床AI书写内容的规模。当团队使用评判者的评估结果来选择提示词或训练模型时，评判者本身就成为了干预的一部分。这种选择过程可能会偏向评判者所忽视的错误。我们描述了评判者的报告必须包含哪些内容，以及每一项由评判者驱动的变更所需的独立临床测试。

    arXiv:2611.18297v1 Announce Type: new  Abstract: AI judges, models that grade another system's output against a rubric, extend clinical review to the scale at which clinical AI writes. When teams use judge assessments to select a prompt or train a model, the judge becomes part of the intervention. Selection can then favor errors the judge overlooks. We describe what a judge's report must contain and the independent clinical test each judge-driven change needs.
    
[^102]: 财报电话会议的时点编码器

    Point-in-Time Encoders for Earnings Calls

    [https://arxiv.org/abs/2611.14325](https://arxiv.org/abs/2611.14325)

    该论文提出按年度截止日期训练的“时点”文本编码器，以消除财报电话会议检索回测中的前视偏差，其小型专用模型在答案检索上全面优于通用检索器，并将收益预测的秩相关性提升约一半。

    

    文本编码器将财报电话会议转换为向量，用于搜索、用于基于检索段落进行回答的语言模型，以及用于预测模型。实时检索器可以使用在所有可用文本上一次性训练的编码器（正如金融检索器通常的做法），但回测需要一个没有读过每个向量生成日期之后所写文本的编码器。为了解决这种前视偏差，我们从412,248个电话会议中，按照四种预训练方案训练了多个“版本”，截止日期从2010年到2024年逐年划分，每个版本只读取其截止日期当天或之前的文本。随后，我们在电话会议本身的问答对上对每个版本进行微调。仅用电话会议文本训练的版本（参数量3380万）在所有测试季度中，对留出电话会议的答案检索效果都优于我们测试的通用检索器。这些向量还将次日收益与基于过去收益和公司规模构建的预测之间的秩相关系数提高了约一半。当每个版本继续……（原文在此截断）

    arXiv:2611.14325v1 Announce Type: new  Abstract: Text encoders turn earnings calls into vectors for search, for language models that answer from retrieved passages, and for forecasting models. A live retriever can use an encoder trained once on all available text, as finance retrievers are, but a backtest needs an encoder that read no text written after the date of each vector. To address this look-ahead bias, from 412,248 calls we train vintages under four pretraining recipes, with yearly cutoffs from 2010 to 2024, and each vintage reads only text dated on or before its cutoff. We then fine-tune each vintage on the question-answer pairs of the calls themselves. The vintages we train on calls alone, with 33.8M parameters, retrieve answers on held-out calls better than the general retrievers we tested in every test quarter. Their vectors also raise by about half the rank correlation between next-day returns and forecasts built from past returns and firm size. When each vintage continues
    
[^103]: 删除敏感性是模型的一种属性：对七个开放推理模型的三条件测量

    Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models

    [https://arxiv.org/abs/2611.13223](https://arxiv.org/abs/2611.13223)

    本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。

    

    推理模型在多轮工具会话的每一轮都会写下一条推理轨迹，而后续轮次会读取这段历史。在94k条ShareGPT对话中，73%的用户交互跨越多轮。上游提供商将每条推理轨迹绑定到其发出者，因此跨发出者转发的请求到达时其历史已被删除，而这种剥离操作带来的任务成本此前从未被测量。据我们所知，本文首次对多轮工具调用中完整推理历史删除进行了直接、盲测、多种子的测量，涵盖三种渲染条件、七个开放权重推理模型、三个随机种子、794个任务以及官方检查器。相同的删除操作导致的性能变化跨度为-4.4到+51.3分，且两端均不包含零；对于唯一一个从0%到100%删除比例重新测试的模型，其读数变化最多仅1.5分。删除敏感性是模型层面的属性。该读数跟随模型本身并保持稳定（原文在此处截断）。

    arXiv:2611.13223v1 Announce Type: cross  Abstract: Reasoning models write a reasoning trace at every turn of a multi-turn tool session, and later turns read this history. Across 94k ShareGPT conversations, 73 percent of user interactions span multiple turns. Upstream providers bind each trace to its issuer, so a request forwarded across issuers arrives with its history deleted, and the task cost of this stripping remains unmeasured. To the best of our knowledge, this paper presents the first direct, blind, multi-seed measurement of full reasoning-history deletion in multi-turn tool calling, through three rendering conditions, seven open-weight reasoning models, three seeds, 794 tasks, and the official checker. The same deletion spans -4.4 to +51.3 points with both ends excluding zero, and for the one model re-dosed from 0 to 100 percent deletion, the reading varies by at most 1.5 points. Deletion sensitivity is a model-level property. The reading follows the model itself and stays cons
    
[^104]: TS-GEAR：淘宝闪购生成式广告检索

    TS-GEAR: Taobao Shangou GEnerative Ad Retrieval

    [https://arxiv.org/abs/2611.12180](https://arxiv.org/abs/2611.12180)

    该论文提出TS-GEAR，面向即时配送场景的生成式广告检索，通过显式建模配送范围可达性、基于地理位置与当前查询条件化的用户兴趣表示以及集合级训练信号，解决了语义ID地理信号软化、兴趣估计偏差与候选冗余三大挑战。

    

    诸如DoorDash和Uber Eats这类即时配送平台存在一个传统电商所没有的约束：超出用户配送范围的商家是无效候选，而不仅仅是排序较低。生成式检索近期在大规模检索系统中展现出强大潜力，然而将其适配到这一场景面临三大挑战。第一，现有语义ID仅将地理位置作为软信号编码，因此可达性只能依赖生成之后的不可微硬截断来保证。第二，用户偏好在不同区域和不同查询之间差异显著，若在不结合地理位置与当前查询的情况下聚合历史行为，会产生有偏的兴趣估计。第三，检索到的候选集合以其整体价值来评估，然而自回归训练对每个序列独立打分，缺乏集合级信号来抑制冗余的近似重复候选。为解决……（摘要在此处被截断）

    arXiv:2611.12180v2 Announce Type: replace-cross  Abstract: On-demand delivery platforms such as DoorDash and Uber Eats impose a constraint absent in traditional e-commerce: merchants beyond the user's delivery range are invalid candidates rather than merely low-ranked. Generative retrieval has recently shown strong potential in large-scale retrieval systems. Adapting it to this setting, however, poses three challenges. First, existing semantic IDs encode geography only as a soft signal, so reachability can only rely on a non-differentiable post-generation hard truncation. Second, user preferences vary substantially across both regions and queries, so aggregating historical behaviors without conditioning on geography and the current query yields biased interest estimates. Third, the retrieved candidate set is evaluated by its collective value, yet autoregressive training scores each sequence independently, with no set-level signal to discourage redundant near-duplicates. To address the 
    
[^105]: 追溯大语言模型科学发现能力与其训练数据的关联

    Tracing the Scientific Discovery Capacity of LLMs to Their Training Data

    [https://arxiv.org/abs/2611.03617](https://arxiv.org/abs/2611.03617)

    该研究提出“单次事实”理论，通过直接操纵训练数据，证明训练集中仅出现一次的事实比率是决定大语言模型科学发现能力的关键因素，且这一能力可在不同模型、领域间迁移。

    

    大型语言模型（LLM）在科学发现方面展现出越来越大的潜力，但关于什么因素使其能够可靠地做出发现，一些基本问题仍未解决。我们如何将LLM的科学发现能力追溯到其训练数据？我们将这一问题作为“缺失概率质量”问题进行研究，并建立了一个理论框架，其中“单次事实”——即训练中仅被观察到一次的事实——决定了发现能力。在材料科学和调控基因组学领域，我们直接操纵训练集中的单次事实比率，发现在两个开放权重模型家族、不同采样策略和推理预算下，更多的单次事实都能持续带来更多发现。进一步的实验表明，这种关系不能仅用训练集的多样性来解释，并且在与领域内数据配对时，这种能力可以跨科学领域迁移。总之，我们的结果确定了训练数据中的单次事实比率是决定语言模型发现能力的关键因素。

    arXiv:2611.03617v1 Announce Type: new  Abstract: Large language models (LLMs) show growing promise for scientific discovery, yet basic questions remain about what enables reliable discovery. How can we trace the scientific discovery capacity of LLMs to their training data? We study this question as one of missing probability mass and develop a theoretical account in which monofacts-the facts observed once during training-set the discovery capacity. In materials science and regulatory genomics, we directly manipulate monofact rates in training sets and find that more monofacts consistently yield more discoveries across two open-weight model families, sampling strategies, and reasoning budgets. Further experiments show the relationship cannot be explained by training-set diversity alone and can transfer across scientific domains when paired with in-domain data. Together, our results identify monofact rate in training data as a key factor governing discoveries in language models.
    
[^106]: 超越整体WER：在线西班牙语学习课堂中的领域内适配、学习者语音、语码转换与说话人归属

    Beyond Aggregate WER: In-domain Adaptation, Learner Speech, Code-Switching, and Speaker Attribution in Online Spanish Learning Classrooms

    [https://arxiv.org/abs/2610.38016](https://arxiv.org/abs/2610.38016)

    该研究超越整体WER的评估方式，按说话人角色和语言语境细分错误来评估在线西班牙语课堂中的多种ASR系统，并通过LoRA领域内适配发现MOSS-Transcribe-Diarize在各维度上均带来一致改进。

    

    自动语音识别（ASR）日益为教育自然语言处理提供转录文本，但整体词错误率（WER）可能掩盖集中于学习者语音、多语轮次和说话人归属中的错误。我们在在线二语西班牙语互动的留出会话上，比较了平台集成、开源权重、适配后和商业等ASR配置，这些互动发生在母语为英语的学习者与其教师和辅导教师之间。由于英语是学习者的共同第一语言，这些互动包含纯西班牙语、纯英语和语码转换的轮次。评估结合了整体WER和字符错误率（CER）、按说话人角色和语言语境细分的WER，以及说话人分离错误率（DER）。我们还使用人工校对的领域内课堂数据，通过LoRA对Whisper large-v3和MOSS-Transcribe-Diarize进行了适配。MOSS适配在所有说话人角色和语言语境上带来了一致的改进，而Whisper则表现出……（原文摘要在此处截断）

    arXiv:2610.38016v2 Announce Type: replace  Abstract: Automatic speech recognition (ASR) increasingly supplies transcripts for educational NLP, but aggregate word error rate (WER) can conceal errors concentrated in learner speech, multilingual turns, and speaker attribution. We compare platform-integrated, open-weight, adapted, and commercial ASR configurations on held-out sessions of online L2 Spanish interaction between L1 English learners and their instructors and tutors. Because English is the learner's shared first language, the interactions contain Spanish-only, English-only, and code-switched turns. Evaluation combines overall WER and character error rate (CER), WER disaggregated by speaker role and language context, and diarization error rate (DER). We also adapt Whisper large-v3 and MOSS-Transcribe-Diarize with LoRA using human-corrected in-domain classroom data. MOSS adaptation produced consistent improvements across speaker roles and language contexts, whereas Whisper showed 
    
[^107]: 线路上的声音

    The Voice on the Line

    [https://arxiv.org/abs/2610.35272](https://arxiv.org/abs/2610.35272)

    该论文训练了20毫秒延迟、能从通话本身注册来电者的因果语音隔离器（掩码transformer），并发现将WavLM-Large权重因果化并与模型联合训练能显著提升效果，将开发集错误率从11.44%降至10.55%。

    

    通过一个麦克风收听的转录器也会听到麦克风附近的所有人。我们训练了因果语音隔离器：参数量分别为1.4亿、2.43亿和3.74亿的掩码transformer，延迟为20毫秒，直接从通话本身注册来电者的声音。短时间的评估窗口会掩盖真实通话中暴露出的两种失败：当另一个声音单独说话一段时间后遗忘来电者，以及损害正在单独说话的来电者的语音。带有话轮转换的长通话、超出注意力窗口的记忆、真实的耳机串音以及漂移的来电者音量水平能够同时应对这两个问题，但每一种方法都会以牺牲一方为代价换取另一方。将WavLM-Large自身的权重改造为因果形式并与模型联合训练，能够改变这种权衡：采用这些权重的M模型（5.55亿参数）将开发集上的错误率从11.44%降至10.55%；从零开始训练以模仿WavLM的学生模型则无法做到这一点，而在多得多的音频上预训练的w2v-BERT 2.0和Whisper编码器的帮助较小或毫无帮助。单独说话的来电者在录音开始时损失最大：（摘要在此处被截断）

    arXiv:2610.35272v1 Announce Type: cross  Abstract: A transcriber listening through one microphone also hears everyone near it. We train causal voice isolators: masking transformers of 140M, 243M and 374M parameters, 20 ms of latency, enrolling the caller from the call itself. Short evaluation windows hide two failures that real calls expose: forgetting the caller when another voice talks alone for a while, and damaging a caller who is alone. Long calls with turns, memory beyond the attention window, real headset crosstalk and drifting caller levels address both, but each trades one against the other. WavLM-Large's own weights, made causal and trained with the model, move the trade: M with them (555M parameters) lowers the error on our development sets from 11.44% to 10.55%; a student trained from scratch to imitate WavLM does not, and w2v-BERT 2.0 and Whisper's encoder, pre-trained on far more audio, help less or not at all. A lone caller loses most at the start of a recording: no trai
    
[^108]: 形式重于内容？基于大语言模型的同行评审中的语言偏见

    Style over Substance? Linguistic Bias in LLM-based Peer Review Copy title

    [https://arxiv.org/abs/2610.35254](https://arxiv.org/abs/2610.35254)

    该论文提出了一个受控实验框架，通过基于语言学文献推导的细粒度改写指令，系统性地改变论文的语言风格（母语/非母语特征与高/低置信度特征），以研究基于大语言模型的同行评审是否会受到与科学内容无关的语言偏见的影响。

    

    近年来，向主要人工智能会议提交的论文数量急剧增加，大幅加重了研究人员的同行评审工作量。与此同时，大语言模型（LLM）正越来越多地被用于支持同行评审，无论是被个别审稿人使用，还是被会议组织方采用。尽管LLM的能力已有显著提升，但其评估结果可能会受到与科学内容无关的语言特征的影响。在这项工作中，我们引入了一个受控实验框架，通过系统地改变研究论文的语言风格来研究这类效应。具体而言，我们对论文进行改写，使其分别呈现（i）与英语母语或非母语写作相关的语言特征，以及（ii）高置信度或低置信度的语言特征。为了构建这些干预措施，我们以语言学文献为基础推导出细粒度的改写指令，并将其应用于……

    arXiv:2610.35254v1 Announce Type: new  Abstract: The number of paper submissions to major AI conferences has increased dramatically in recent years, substantially increasing the peer-review workload of researchers. At the same time, large language models (LLMs) are increasingly used to support peer review, both by individual reviewers and by conference organizations. While LLM capabilities have improved considerably, their evaluations may be influenced by linguistic characteristics that are unrelated to scientific content. In this work, we introduce a controlled experimental framework for studying such effects by systematically perturbing the linguistic style of research papers. Specifically, we rewrite papers to exhibit (i) linguistic features associated with native or non-native English writing, and (ii) high- or low-confidence linguistic features. To construct these interventions, we derive fine-grained rewriting instructions grounded in the linguistic literature and apply them usin
    
[^109]: 基于决策模型的实体匹配

    Entity Matching using Decision Models

    [https://arxiv.org/abs/2610.34626](https://arxiv.org/abs/2610.34626)

    本文提出用决策模型作为实体匹配的新替代方案，证明其在零样本设置下无需任务特定训练数据即可取得接近生成式大语言模型的匹配性能。

    

    实体匹配是判断两条记录是否描述同一个现实世界实体的任务。当前的实体匹配方法依赖于需要针对特定任务进行微调的预训练语言模型（PLM），或者依赖于无需任务特定微调、但在匹配大规模记录集时速度慢且成本高的生成式大语言模型（LLM）。决策模型是一类新型的基于大语言模型的模型，它们为每个预定义的答案选项返回概率，而不是生成文本。本文研究了决策模型作为生成式大语言模型和微调预训练语言模型在实体匹配任务中的替代方案。我们在八个基准数据集上以零样本设置评估了六个决策模型，并将其性能与生成式大语言模型 GPT-6 Luna 以及基于预训练语言模型的匹配器 Ditto 进行了比较。我们的实验表明，在无需任务特定训练数据的情况下，四个最强的决策模型取得了接近 GPT-6 Luna 的平均 F1 分数。

    arXiv:2610.34626v1 Announce Type: cross  Abstract: Entity matching is the task of deciding whether two records describe the same real-world entity. Current entity matching methods rely on pre-trained language models (PLMs) which require task specific fine-tuning or generative large language models (LLMs) which do not require task-specific fine-tuning but are slow and expensive at matching large record sets. Decision models are a new class of LLM-based models that return a probability for each predefined answer option rather than generate text. This paper investigates decision models as an alternative to generative LLMs and fine-tuned PLMs for entity matching. We evaluate six decision models in a zero-shot setting across eight benchmarks and compare their performance with that of the generative LLM GPT-6 Luna and the PLM-based matcher Ditto. Our experiments show that, without task-specific training data, the four strongest decision models achieve mean F1 scores close to that of GPT-6 Lu
    
[^110]: CulturalRiddles：一个多文化谜语基准测试

    CulturalRiddles: A Multicultural Benchmark of Riddles

    [https://arxiv.org/abs/2610.29968](https://arxiv.org/abs/2610.29968)

    CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。

    

    随着大语言模型不断扩大其语言覆盖范围，评估它们对多元文化的理解和表达能力变得越来越重要。然而，许多基准测试仍然以英语为中心，文化覆盖有限，并且更侧重于事实性知识而非其在现实文化语境中的应用。我们提出了CulturalRiddles，这是一个完全由人类通过社区参与式努力创作的原创、植根于文化的谜语基准，涵盖62种文化（45个国家、51种语言）。这些谜语深深植根于各自的文化，需要溯因推理和组合推理能力，以及详细的事实和社会知识、语言技能以及对文化特定关联的理解。我们使用自动评估和大规模人工评估对28个LLM配置进行了基准测试。即使是前沿模型，在某些社区中的准确率也最高仅达到22%。

    arXiv:2610.29968v1 Announce Type: new  Abstract: As LLMs expand their language coverage, evaluating how well they understand and represent diverse cultures becomes increasingly important. However, many benchmarks remain English-centric, with limited cultural coverage and a focus on factual knowledge over its application in real-world cultural contexts. We propose CulturalRiddles, a benchmark of original, culturally grounded riddles created entirely by humans through a participatory community effort, spanning 62 cultures (45 countries, 51 languages). These riddles are deeply rooted in their respective cultures and require abductive and compositional reasoning, alongside detailed factual and social knowledge, language skills, and an understanding of culturally specific associations. We benchmark 28 LLM configurations using both automatic and large-scale human evaluation. Even frontier models achieve at most 22% accuracy in some communities. We publicly release all artifacts alongside an 
    
[^111]: Right On Q：通过自生成轨迹恢复量化模型的推理能力

    Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories

    [https://arxiv.org/abs/2610.29378](https://arxiv.org/abs/2610.29378)

    提出OnQ流水线，通过在策略蒸馏利用学生模型自生成的轨迹进行训练，解决了固定响应监督导致的训练与生成前缀不匹配及高熵处分布失配问题，从而恢复低比特量化大模型的推理能力。

    

    量化感知训练（QAT）旨在恢复低比特大语言模型（LLM）的推理能力。现有方法通常利用原始的全精度模型在固定的响应序列上对量化学生模型进行监督。然而，由此得到的低比特量化模型在长程推理任务上仍然表现不佳，并且常常无法在足以满足全精度模型的token预算内正常终止。我们识别出了缩小这一差距的两个挑战：第一，固定响应监督导致训练前缀与学生模型实际生成时遇到的前缀之间存在不匹配，且随着响应变长，这种轨迹分歧会不断增大；第二，在学生模型自身的前缀上，它会对某些下一个token候选词预测不足，而对另一些候选词预测过度，这种分布失配在高熵前缀处出现得更加频繁。我们提出了OnQ，一个低比特恢复流水线，它结合了在策略蒸馏与候选（摘要在此处被截断）

    arXiv:2610.29378v1 Announce Type: new  Abstract: Quantization-aware training (QAT) aims to recover reasoning capabilities in low-bit large language models (LLMs). Existing recipes typically supervise the quantized student with the original full-precision model on fixed response sequences. However, the resulting low-bit quantized models still struggle on long-horizon reasoning tasks and often fail to terminate within a token budget that suffices for the full-precision model. We identify two challenges in closing this gap. First, fixed-response supervision leaves a mismatch between training prefixes and those encountered during student generation, and this trajectory divergence grows as responses lengthen. Second, on its own prefixes, the student underpredicts some next-token candidates and overpredicts others, and this distributional misalignment becomes more frequent at high-entropy prefixes. We present OnQ, a low-bit recovery pipeline that combines on-policy distillation and candidate
    
[^112]: MemT-1.0：迈向记忆原生演化

    MemT-1.0: Towards Memory-Native Evolution

    [https://arxiv.org/abs/2610.26825](https://arxiv.org/abs/2610.26825)

    MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。

    

    记忆原生演化要求将记忆作为基础模型的内在能力，使其能够持续积累和利用信息。为此，我们提出了MemT-1.0，一种将记忆内化为模型计算一部分的记忆基础模型新架构。MemT-1.0基于Metis构建，采用Gated DeltaNet-2，并在每一轮内使用稳定的token级记忆更新机制。我们进一步引入Mixture-of-Storage（存储混合）来扩展原生记忆容量。在经过精心准备的主干模型之后，我们开发了一个由继续预训练和后训练组成的两阶段训练流程，并由340亿（34B）token的训练数据以及高效的训练与推理基础设施支撑。我们发布了包含MemT-1.0-Nano和MemT-1.0-Flash的模型家族，并对记忆能力和通用能力进行了全面评估。结果表明，它们在（各项评估中）优于参数化记忆基线。

    arXiv:2610.26825v2 Announce Type: replace  Abstract: Memory-native evolution requires making memory an intrinsic capability of foundation models, enabling them to continuously accumulate and utilize information. To this end, we introduce MemT-1.0, a new architecture of memory foundation models that internalize memory as part of model computation. Building on Metis, MemT-1.0 employs Gated DeltaNet-2 with a stable token-level memory update mechanism within each turn. We further introduce Mixture-of-Storage to scale native memory capacity. Following a deliberate backbone preparation, we develop a two-stage training pipeline consisting of continued pre-training and post-training, supported by 34B tokens of training data and efficient training and inference infrastructure. We release a model family comprising MemT-1.0-Nano and MemT-1.0-Flash, and conduct a comprehensive evaluation of memory and general capabilities. The results show that they outperform the parametric memory baselines on av
    
[^113]: 计算而非记忆：LLM智能体中的思维链与状态跟踪

    Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents

    [https://arxiv.org/abs/2610.25760](https://arxiv.org/abs/2610.25760)

    该论文通过区分“行动前即时生成推理”与“在提示中保留历史推理笔记”两种角色，发现真正减少状态违规行动的主要来源是行动时刻的即时推理（计算），而非记忆式的历史推理保留。

    

    语言模型智能体通常在行动之前进行推理，而监控这种推理的价值取决于它与智能体行动之间的关系。我们探讨当正确的行动依赖于不断变化的状态时，智能体的行动如何依赖于其推理。在一个小型文件系统任务中，后台事件会宣布哪个文件的锁被切换，但不会宣布其新值。因此，智能体必须从历史记录中重建当前的锁状态，才能安全地移动文件。基于 Qwen3-8B 和 Qwen3.5-9B 模型，我们区分了推理的两种角色：在每次行动之前生成推理，以及在提示中保留早期推理。减少锁违规命令的收益大部分来自行动当下写出的推理。当智能体在每一步都进行推理时，保留早期推理几乎没有额外帮助；但对于不逐步推理的智能体，保留早期推理是有帮助的，并且在更长的回合中也是如此。保留的笔记有其局限：当后续事件使某条笔记变得不正确时……

    arXiv:2610.25760v2 Announce Type: replace-cross  Abstract: Language-model agents often reason before they act, and the value of monitoring that reasoning depends on how it relates to their actions. We ask how an agent's actions depend on its reasoning when the correct action depends on a changing state. In a small file-system task, background events announce which file's lock has been toggled but not its new value. The agent therefore has to reconstruct the current locks from its history before it can move files safely. With Qwen3-8B and Qwen3.5-9B, we separate two roles of reasoning: generating it before each action, and retaining earlier reasoning in the prompt. Most of the reduction in lock-violating commands comes from reasoning written at the moment of action. Retaining earlier reasoning adds little when the agent also reasons at each step, but it helps an agent that does not reason, and it helps in longer episodes. Retained notes have a limit: when a later event makes a note wron
    
[^114]: CHSM8：一种遵循自然语言风格提示的国际象棋Transformer

    CHSM8: A Chess Transformer that Follows Natural-Language Style Prompts

    [https://arxiv.org/abs/2610.25356](https://arxiv.org/abs/2610.25356)

    CHSM8是一个2.1亿参数的国际象棋Transformer，无需搜索即可根据自然语言风格提示下棋，达到1837 Elo并击败了包括参数量为其13倍模型在内的所有同类公开模型，并通过合成数据微调使其能够理解文本风格指令。

    

    我们提出了CHSM8，一个2.1亿参数的国际象棋Transformer，它可以根据对下棋风格的自由文本描述来走子，例如“采用西西里防御”、“稳健、安全、无风险的棋风”或“你是巅峰时期的马格努斯·卡尔森”。该模型在10.7亿局经过筛选的Lichess对局上进行预训练，无需搜索，采用因式分解的走子表示和合法走子掩码。在面对强度受限的Stockfish时，它达到了1837 Elo的等级分；在1000局的比赛中，它击败了我们测试的所有其他同样在没有偏好调优、奖励或搜索的情况下训练的公开模型，其中包括一个参数量是其13倍的模型。随后，我们在一个合成数据集上对其进行微调以使其遵循文本指令：Gemma-4-31B为300万个对局方描述其下棋风格，一个从240万条与棋盘标签无矛盾的描述中蒸馏而来的1B Gemma-3模型对其余描述进行标注，并且所筛选的3000万个对局方中的每一个都获得五条风格描述。监督微调……

    arXiv:2610.25356v2 Announce Type: replace-cross  Abstract: We present CHSM8, a 210M-parameter chess transformer that plays moves conditioned on a free-text description of a playing style, e.g. "go for the Sicilian Defense", "solid, safe, risk-free chess" or "You are Magnus Carlsen in his prime". The model is pretrained without search on 1.07B filtered Lichess games, using a factorized move representation and legal-move masking. It reaches 1837 Elo against a strength-limited Stockfish and, in 1,000-game matches, beat every other public model we tested that was likewise trained without preference tuning, rewards or search, including one with 13 times as many parameters. We then fine-tune it to follow text on a synthetic dataset: Gemma-4-31B describes the playing style of 3M game sides, a 1B Gemma-3 model distilled from the 2.4M of those descriptions that contradict nothing on the board labels the rest, and every one of 30M selected game sides gets five descriptions. Supervised fine-tunin
    
[^115]: 改变数字，改变答案：评估基于大语言模型的文本评分中的评分量表敏感性

    Change the Number, Change the Answer: Assessing Rating Scale Sensitivity in LLM-Based Text Scoring

    [https://arxiv.org/abs/2610.22187](https://arxiv.org/abs/2610.22187)

    本文定义并系统研究了“量表敏感性”这一新概念，通过大规模交叉实验发现，评分量表的设计元素（格式、范围、符号）平均会使大语言模型的文本评分产生约0.10的归一化分数偏移，揭示了不同研究间分数可比性的重要隐患。

    

    大语言模型在计算社会科学中越来越多地被用于为社会和心理构念分配标量分数。然而，不同研究使用的评分量表各异，其分数在不同量表设计下是否仍具可比性尚未得到充分研究。为填补这一空白，我们定义并系统研究了“量表敏感性”，即由评分量表设计引起的LLM生成分数的变化。我们区分了数值敏感性（归一化分数的变化）与语义敏感性（分数解释的变化）。为评估这两种敏感性，我们的实验设计将2种格式（连续型与序数型）、4种范围（3、5、11和21分）和2种符号（有符号与无符号）进行交叉，涵盖3项任务（政治倾向、情感和毒性）、2种提示策略（零样本和基于量规的提示）以及5个大语言模型。我们发现，改变其中一个设计元素平均会使归一化分数偏移约0.10，并改变类别的……

    arXiv:2610.22187v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in computational social science to assign scalar scores to social and psychological constructs. However, studies use different rating scales, and whether their scores remain comparable across scale designs is understudied. To address this gap, we define and systematically investigate scale sensitivity, the variation in LLM-generated scores caused by rating scale design. We distinguish numerical sensitivity, changes in normalized scores, from semantic sensitivity, changes in score interpretation. To assess both, our experimental design crosses 2 formats (continuous vs. ordinal), 4 ranges (3, 5, 11, and 21 points), and 2 signs (signed vs. unsigned) across 3 tasks (political leaning, sentiment, and toxicity), 2 prompting strategies (zero-shot and rubric-based), and 5 LLMs. We find that changing one design element shifts normalized scores by about 0.10 on average and changes the category o
    
[^116]: SkillForgeBench：基于智能体技能规范的智能体可靠性评估

    SkillForgeBench: Evaluating Agent Reliability Grounded in Agent Skill Specifications

    [https://arxiv.org/abs/2610.14918](https://arxiv.org/abs/2610.14918)

    SkillForgeBench以智能体技能作为统一执行蓝图，通过资源准备、技能驱动任务合成等三个协同设计阶段，构建了兼顾领域多样性与程序复杂性的智能体可靠性评估基准，化解了专家策划基准与众包基准之间的根本性权衡。

    

    评估智能体严格满足多样化且程序复杂需求的能力，对于确定其实际可靠性至关重要。开展此类评估需要能够兼顾广泛领域覆盖与深层程序复杂性的智能体任务。现有范式面临根本性的权衡：专家策划的基准虽然实现了程序深度，却受制于领域多样性的瓶颈；而众包基准虽然覆盖了多样化场景，却缺乏执行的严谨性。为化解这一矛盾，我们提出了SkillForgeBench，它将智能体技能作为统一的执行蓝图，贯穿三个协同设计的阶段：（1）资源准备阶段利用领域均衡的技能来指导工件检索，建立广泛的工作流覆盖；（2）技能驱动的任务合成阶段将技能作为程序锚点，用以划分工件角色、合成自然的任务指令，并打包沙盒化场景以实现深层程序复杂性……

    arXiv:2610.14918v1 Announce Type: new  Abstract: Evaluating an agent's ability to strictly satisfy diverse and procedurally complex requirements is essential to determining its practical reliability. Conducting such an evaluation demands agentic tasks that reconcile broad domain coverage with deep procedural complexity. Existing paradigms face a fundamental trade-off: expert-curated benchmarks achieve procedural depth but suffer from domain diversity bottlenecks, whereas crowdsourced benchmarks span diverse scenarios but lack execution rigor. To resolve this tension, we present SkillForgeBench, which leverages agent skills as unifying execution blueprints across three co-designed stages: (1) Resource Preparation uses domain-balanced skills to guide artifact retrieval, establishing broad workflow coverage; (2) Skill-driven Task Synthesis employs skills as procedural anchors to partition artifact roles, synthesize naturalistic task instructions, and package sandboxed scenarios for deep p
    
[^117]: 对模型合并的更广泛审视：重新思考任务算术所引发的隐式正则化

    A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic

    [https://arxiv.org/abs/2610.07990](https://arxiv.org/abs/2610.07990)

    本论文发现主流模型合并方法中系数搜索带来的隐式正则化实际上限制了性能，去除该正则化、直接优化合并模型乃至预训练模型的权重，即可在多种架构和极低数据量场景下显著提升合并后的多任务性能。

    

    模型合并旨在通过组合各个特定任务模型的权重，以低成本的方式构建多任务模型。为了在多个任务上表现良好，大多数现有的合并方法使用额外的数据集来寻找特定任务权重更新的最佳线性组合系数。然而，我们在这项标准实践中识别出一种隐式正则化：对系数进行搜索会将候选模型限制在由特定任务权重更新所张成的子空间内。在本工作中，我们研究这种正则化是否真的有用。令人惊讶的是，实证结果表明，在没有这种正则化的情况下直接优化合并模型的权重，能够在多种架构、多个领域、甚至在每个类别仅有一个样本的极端数据受限场景下显著提升常见合并方法的性能。此外，直接优化预训练模型的权重甚至优于一些现有方法……（摘要在此处被截断）

    arXiv:2610.07990v1 Announce Type: cross  Abstract: Model merging aims to build a multi-task model cheaply by combining the weights of individual task-specific models. To perform well across multiple tasks, most existing merging methods use an additional dataset to find the coefficients for the best linear combination of task-specific weight updates. However, we identify an implicit regularization in this standard practice: searching over coefficients restricts the candidate models to a subspace spanned by task-specific weight updates. In this work, we investigate whether this regularization is actually useful. Surprisingly, empirical results show that optimizing merged-model weights without this regularization significantly boosts the performance of common merging methods across multiple architectures, domains, and even in an extremely data-limited scenario where only one instance is available per class. Moreover, directly optimizing the pretrained model weights even outperforms some e
    
[^118]: 冻结场景，更迭赢家：文本到3D评估中的配置脆弱性

    Frozen Scenes, Shifting Winners: Configuration Fragility in Text-to-3D Evaluation

    [https://arxiv.org/abs/2610.00447](https://arxiv.org/abs/2610.00447)

    该论文通过对来自六个生成器的300个冻结场景系统性变化八种渲染与描述文本配置因素，揭示文本到3D评估结果对配置高度脆弱——17/19个对齐评估器的配置方差超过生成器间方差，且18/19个评估器在某些配置下会改变点估计赢家，表明当前评估协议存在严重的不稳定性。

    

    当所有生成的场景保持固定不变时，文本到3D（text-to-3D）排行榜还会发生变化吗？我们针对渲染图像评估对这一问题进行了审计，在此类评估中，相机设置与描述文本的措辞成为了测量协议的一部分。在来自六个生成器的300个冻结场景上，我们针对19个对齐评估器外加1个感知质量对照指标，变化了八种渲染与描述文本因素，随后测试了四种针对性的场景退化。结果发现：对于19个对齐评估器中的17个，峰值配置方差超过了生成器间的方差，其中11个评估器的提示词自助法（prompt-bootstrap）下界大于1。排名虽然比分数更稳定，但在某些配置下，19个评估器中有18个会改变其点估计赢家。成对协议边际包络显示哪些比较在所测试的设置中保持方向不变。选定的配对虽然具有方向相反的逐点区间，但在对完整搜索进行同时推断后，没有任何排名逆转能够存续。因此，所观察到的赢家变化是描述性的……

    arXiv:2610.00447v1 Announce Type: new  Abstract: Can a text-to-3D leaderboard change when every generated scene stays fixed? We audit this question for rendered-image evaluation, where camera settings and caption wording become part of the measurement protocol. Across 300 frozen scenes from six generators, we vary eight render and caption factors for 19 alignment evaluators plus one perceptual-quality control, then test four targeted scene degradations. Peak configuration variance exceeds between-generator variance for 17/19 alignment evaluators, with prompt-bootstrap lower bounds above 1 for 11/19. Rankings are more stable than scores, yet 18/19 evaluators change their point-estimate winner under some configuration. Pairwise protocol margin envelopes show which comparisons keep their direction across the tested settings. Selected pairs have opposite pointwise intervals, but no reversal survives simultaneous inference over the full search. Thus the observed winner changes are descripti
    
[^119]: MatLoom：紧凑程序空间中的分层文本到材质生成

    MatLoom: Layered Text-to-Material Generation in a Compact Program Space

    [https://arxiv.org/abs/2609.40322](https://arxiv.org/abs/2609.40322)

    MatLoom提出一种紧凑的面向图层的材质程序语言，配合预训练语言模型，通过解析器引导修复与预览评审实现免微调的文本到材质生成，其最佳配置在141个提示词的基准上超越了三个扩散基线。

    

    材质生成不仅应产出外观，还应产出构建该外观的规则。我们提出MatLoom，一种紧凑的、面向图层的材质程序语言，可与预训练语言模型结合进行文本到材质生成。每个程序由带Alpha遮罩的图层组合而成，图层间共享的空间表达式定义了覆盖范围与基于物理的渲染（PBR）通道，从而使图案、颜色与浮雕之间的依赖关系显式化。一个独立的解释器将程序求值为材质贴图，同时源代码保留命名字段和图层参数，便于后续编辑创作。无需任务特定的微调，我们的流程利用解析器引导的修复和基于预览的评审来修改材质设计，然后在保持每个候选的其余源代码不变的情况下搜索噪声种子。在一个包含141个提示词的精选基准上，使用六个骨干模型进行评估，我们表现最佳的配置在（原文在此处截断）上取得了高于三个扩散基线的平均分数。

    arXiv:2609.40322v1 Announce Type: cross  Abstract: Material generation should produce not only an appearance, but also the rules that construct it. We introduce MatLoom, a compact, layer-oriented language for text-to-material generation with pretrained language models. Each program composes alpha-masked layers whose shared spatial expressions define coverage and physically based rendering (PBR) channels, making dependencies between patterns, color, and relief explicit. A standalone interpreter evaluates the program into material maps, while the source retains named fields and layer parameters for subsequent authoring. Without task-specific fine-tuning, our pipeline uses parser-guided repair and preview-based critique to revise material designs, then searches noise seeds while keeping each candidate's remaining source fixed. On a curated benchmark of 141 prompts evaluated with six backbones, our best-performing configuration achieves higher mean scores than three diffusion baselines on 
    
[^120]: MemLife：面向长期第一人称视频记忆的整理与推理

    MemLife: Curating and Reasoning over Long-Term Egocentric Video Memories

    [https://arxiv.org/abs/2609.40195](https://arxiv.org/abs/2609.40195)

    MemLife是一种多模态记忆系统，通过构建基于实体的第一人称文本情节并用时间索引的智能体读取器进行检索，在无需训练和查询时视频访问的情况下，在四个长时程基准上比最强免训练基线提升4.6%–12.0%，解决了长期第一人称视频中记忆证据丢失和检索竞争的问题。

    

    长期第一人称视频使个性化AI助手能够对日常生活进行推理。然而，随着视频历史增长到跨越数月甚至数年的数百小时，为每次查询重新处理原始视频片段在计算上变得难以承受。记忆系统通过将视频压缩为文本表示提供了一种可扩展的替代方案，但在实际基准测试中常常失败：要么记忆未能保留关键证据，要么由于搜索空间不断扩大导致检索竞争，检索器无法定位相关条目。为应对这些挑战，我们提出了MemLife，一种多模态记忆系统，它构建基于实体的第一人称文本情节，并通过时间索引的智能体读取器进行检索。在无需训练且查询时无需访问视频的情况下，MemLife在四个长时程基准测试中比最强的免训练基线提升了4.6%至12.0%。为进一步提升记忆质量，我们提出了MemOpt，一种基于强化的方法（摘要在此处被截断）。

    arXiv:2609.40195v1 Announce Type: cross  Abstract: Long-term egocentric video enables personalized AI assistants to reason about daily life. However, as video histories grow to hundreds of hours spanning months or years, reprocessing raw clips for every query becomes computationally prohibitive. Memory systems offer a scalable alternative by compacting videos into text representations, but often fail on practical benchmarks: either the memory does not preserve key evidence, or the retriever fails to locate relevant entries due to retrieval competition in growing search spaces. To address these challenges, we introduce MemLife, a multimodal memory system that constructs entity-grounded, first-person text episodes and retrieves them via a time-indexed agentic reader. Without training or query-time video access, MemLife improves over the strongest training-free baseline by 4.6--12.0% across four long-horizon benchmarks. To further improve memory quality, we propose MemOpt, a reinforcement
    
[^121]: 哪种纸草手写文本识别（HTR）才够好？四种纸草学任务对希腊语文本字符错误率的容忍度研究

    Which papyrus HTR is good enough? Character-error-rate tolerance of four papyrological tasks on Greek texts

    [https://arxiv.org/abs/2609.37755](https://arxiv.org/abs/2609.37755)

    该研究通过将模拟的不同字符错误率（1%—50%）的HTR输出用于文献类型识别、断代等四种纸草学任务，首次为古希腊纸草文献手写文本识别系统所需的精度建立了基准。

    

    目的：大多数希腊纸草文献仍未出版和数字化；一个能够自动转录这些文献的手写文本识别（HTR）流程将使学者能够发现迄今尚未被阅读过的文书和文学作品。古希腊纸草文献的识别系统尚处于起步阶段，而它们对于特定纸草学任务需要达到何种精度尚未得到检验。为了回答这一问题并为希腊纸草HTR建立基准，我们以已发表的校勘本作为真值，针对四种纸草学任务测试了一系列字符错误率（CER）。方法：从papyri.info中现有的63,846个希腊语纸草文本校勘本出发，我们通过去除编辑层来模拟仅含正文的“完美HTR”输出，然后使用种子算法将其降级为精确的1%—50%字符错误率，并加入缺失行和四种错误形态变体。在这些数据上，我们训练了小型模型（TF-IDF、fastText、字符CNN、ByT5-small），用于文献类型识别、断代和文献性质……（摘要至此被截断）

    arXiv:2609.37755v1 Announce Type: new  Abstract: Purpose: Most Greek papyri remain unpublished and undigitised; a handwritten text recognition (HTR) pipeline that transcribes them automatically would let scholars discover documents and literary works that have so far gone unread. Recognition systems for Ancient Greek papyri are in statu nascendi, and how accurate they must be for a given papyrological task has not been examined. To answer this and set a benchmark for Greek papyrus HTR, we test a range of character error rates (CER) against four papyrological tasks, using published editions as ground truth. Methods: From 63,846 current editions of Greek texts in papyri.info, we imitate a letters-only "perfect HTR" output by removing the editorial layer, then degrade it with a seeded algorithm to exact CERs of 1 - 50%, with lost lines and four error-shape variants. On these data we train small models (TF-IDF, fastText, a character CNN, ByT5-small) for document type, dating and documentar
    
[^122]: 职业事故叙述中事故过程角色分类的跨行业泛化研究

    Cross-sector generalization of accident-process role classification in occupational accident narratives

    [https://arxiv.org/abs/2609.22081](https://arxiv.org/abs/2609.22081)

    该研究构建了法语职业事故叙述的专家标注语料库，将事实单元划分为工作情境、不利条件、事故事件和后果四种角色，并系统评估了角色分类模型跨行业、跨组织领域的泛化能力。

    

    职业事故叙述中包含关于工作情境、不利条件、事故事件及其后果的宝贵信息。自动结构化这些叙述可以促进大规模事故分析，并支持职业风险预防。然而，不同行业和组织在描述事故时所使用的术语和写作风格差异很大，这引发了自动编码系统能否泛化到其训练领域之外的问题。在本文中，我们评估了法语职业事故叙述中事故过程角色分类的跨行业泛化能力。我们构建了一个专家标注语料库，将事实单元划分为四种角色：工作情境（A0）、明确报告的不利条件（A1）、事故事件或偏差（B）以及报告的后果（C）。角色分类器仅在42,24……（原文摘要在此处截断）

    arXiv:2609.22081v1 Announce Type: new  Abstract: Occupational accident narratives contain valuable information about work situations, unfavourable conditions, accident events, and their consequences. Automatically structuring these narratives can facilitate large-scale accident analysis and support occupational risk prevention. However, the terminology and writing styles used to describe accidents vary considerably across sectors and organisations, raising questions about the ability of automated coding systems to generalize beyond their training domain. In this paper, we evaluate the cross-sector generalization of accident-process role classification in French occupational accident narratives. We construct an expert-annotated corpus in which factual units are classified into four roles: work situation (A0), explicitly reported unfavourable condition (A1), accident event or deviation (B), and reported consequence (C). The role classifiers are developed and selected exclusively on 42,24
    
[^123]: AhaBench：智能体能从先验经验中学习吗？一个面向长时程持续学习的基准测试

    AhaBench: Do Agents Learn from Prior Experience? A Benchmark for Long-Horizon Continual Learning

    [https://arxiv.org/abs/2609.05435](https://arxiv.org/abs/2609.05435)

    AhaBench 是一个长时程持续学习基准，通过谜题探索、带精确验证器的数学任务和自动售货机模拟三个组件，评估固定模型在获得先验经验后，在相关但支持已被移除、改变或延迟的条件下行为是否真正得到改善。

    

    arXiv:2609.05435v1 公告类型：新论文。摘要：现代语言智能体被期望在长时程上运行：它们会提出后续问题、复用已解决的示例、处理工具反馈，并适应延迟出现的后果。然而大多数评估仍然在每次提示后重置智能体，或仅对单条轨迹的最终状态进行评分。AhaBench 提出了一个更具操作性的问题：当一个固定模型获得有用的经验后，在一个明显的支持已被移除、改变或延迟的相关评估条件下，其后续行为是否会得到改善？该测试套件包含三个组成部分：Aha-Puzzle 测试智能体在解开隐藏状态谜题后的无提示探索能力；Aha-Euler 将类似 Project Euler 的数学思想转化为带有精确验证器的生成式教学/保留任务；Aha-Vending 是一个受 Vending-Bench 启发的开源实现，测试模拟的自动售货机智能体在处理延迟反馈和运营事故时能否保持盈利。AhaBench 报告了一个由三部分组成的评估结果

    arXiv:2609.05435v1 Announce Type: new  Abstract: Modern language agents are expected to operate over long horizons: they ask follow-up questions, reuse worked examples, handle tool feedback, and adapt to delayed consequences. Most evaluations still reset the agent after a prompt or score only the final state of one trajectory. AhaBench asks a more operational question: when a fixed model receives useful experience, does its later behavior improve under a related evaluation condition where the obvious support has been removed, changed, or delayed? The suite contains three components. Aha-Puzzle tests no-hint exploration after solved hidden-state puzzles; Aha-Euler turns Project-Euler-style mathematical ideas into generated taught/held-out tasks with exact validators; and Aha-Vending, an open-source implementation inspired by Vending-Bench, tests whether a simulated vending agent remains profitable while handling delayed feedback and operational incidents. AhaBench reports a three-part s
    
[^124]: 从接触到预期：西班牙语发展中的频率、意外性与语言

    From Exposure to Expectation: Frequency, Surprisal, and Language Across Development in Spanish

    [https://arxiv.org/abs/2608.22452](https://arxiv.org/abs/2608.22452)

    这项研究表明，在西班牙语中，词汇频率是预测儿童习得年龄的主要因素，而意外性（基于上下文的可预测性）在频率和词长之外几乎没有额外贡献，尽管意外性能预测成人阅读时间。

    

    摘要：意外性，即语言模型根据前文上下文分配给某个词的负对数概率，能可靠预测成年人的阅读时间。但它是否同样有助于解释儿童习得个别词汇的时间呢？频率反映学习者对某个词的累积接触量，而意外性则反映在给定上下文中单个出现的可预测性。我们通过两项基于语料库的西班牙语研究来探讨这一问题。在研究1中，我们使用儿童导向言语中的词汇频率和语境多样性，以及来自三种在架构和训练语言上不同的语言模型（BETO、BERTIN、mGPT）的意外性，对225个西班牙语名词的习得年龄（AoA）进行建模。频率强烈预测了习得年龄（r=-.597, p<.001）；意外性在频率和词长之外增加的预测力很小，包括在自然语境分析中也是如此。在研究2中，我们模拟了M数据集智利西班牙语子样本中的成人注视持续时间。

    arXiv:2608.22452v1 Announce Type: new  Abstract: Surprisal, the negative log-probability a language model assigns to a word given its preceding context, reliably predicts adult reading times. Does it contribute as much to explaining when children acquire individual words? Frequency reflects a learner's cumulative exposure to a word, whereas surprisal reflects how predictable a single occurrence is given its context. We investigate this question across two corpus-based studies of Spanish.   In Study 1, we modeled age of acquisition (AoA) for 225 Spanish nouns using lexical frequency and contextual diversity from child-directed speech, plus surprisal from three language models differing in architecture and training language (BETO, BERTIN, mGPT). Frequency strongly predicted AoA (r=-.597, p<.001); surprisal added little beyond frequency and word length, including in a naturalistic-context analysis.   In Study 2, we modeled adult fixation durations in the Chilean Spanish subsample of the M
    
[^125]: 混合线性注意力大语言模型中的大规模激活：注意力前尖峰与尖峰间平台

    Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus

    [https://arxiv.org/abs/2608.12149](https://arxiv.org/abs/2608.12149)

    本文首次系统研究了混合线性注意力大语言模型中的大规模激活现象，发现了注意力前尖峰和尖峰间平台两种新形态，并揭示了它们与架构配置的关系。

    

    我们首次对层交错混合线性注意力（HLA）大语言模型中的大规模激活（MAs）进行了系统性研究，并揭示了两种与架构对齐的形态：MAs在完全注意力层之前持续出现尖峰，形成注意力前尖峰（PAS），并且可以持续通过中间的线性注意力层，产生尖峰间平台（ISP）。随着完全注意力变得更密集，连续的PAS通过ISP逐渐连接，最终恢复完全注意力大语言模型的稳定MA形态。我们证实了这种组织在五种线性注意力架构、六种混合配置、五个数据域以及代表1.2B到397B总参数规模的开源混合模型中的重复性。基于GDN的混合模型在高达1.3B规模的受控预训练表明，这两种形态在早期出现，并对输出门控表现出不对称响应：完全注意力输出门控强烈减弱了...

    arXiv:2608.12149v1 Announce Type: new  Abstract: We present the first systematic study of Massive activations (MAs) in layer-interleaved HLA LLMs and uncover two architecture-aligned morphologies: MAs consistently spike immediately before full attention layers, forming pre-attention spikes (PAS), and can persist through intervening linear attention layers, giving rise to inter-spike plateaus (ISP). As full attention becomes denser, successive PAS become increasingly connected through ISP, ultimately recovering the stable MA morphology of full attention LLMs. We establish the recurrence of this organization across five linear attention architectures, six hybridization configurations, five data domains, and representative open-source hybrid models spanning 1.2B to 397B total parameters. Controlled pretraining of GDN-based hybrids at scales up to 1.3B shows that both morphologies emerge early and respond asymmetrically to output gating: full attention output gating strongly attenuates the
    
[^126]: 通过测试时缩放视角理解在线策略蒸馏

    Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling

    [https://arxiv.org/abs/2608.11829](https://arxiv.org/abs/2608.11829)

    本研究发现在线策略蒸馏主要提升采样效率而非扩展推理能力边界，其优势在小采样预算下显著，但在大预算下会减弱。

    

    arXiv:2608.11829v1 公告类型：交叉 摘要：在线策略蒸馏（OPD）已成为增强大语言模型推理能力的一种有前景的后训练技术。通常认为，它能使学生模型从更强的教师模型中蒸馏知识，从而扩展超出OPD前基础模型的能力。在本研究中，我们通过测试时缩放的视角，通过改变采样预算K并评估pass@K和avg@K性能来审视这一观点。具体而言，在多种OPD变体中，我们观察到OPD训练的模型在不同采样预算下保持优越的avg@K性能，而pass@K的优势随着K的增加逐渐转移到OPD前的基础模型上。这些结果表明，OPD主要提高了采样效率，而非持续扩展学生模型的推理能力边界。整个OPD训练过程中的pass@K动态进一步揭示，模型逐渐转向更强的低K性能，但以牺牲高K性能为代价。

    arXiv:2608.11829v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has emerged as a promising post-training technique for enhancing LLM reasoning. It is commonly believed to enable the student model to distill knowledge from a stronger teacher model, thereby expanding capabilities beyond the pre-OPD base model. In this study, we examine this view through the lens of test-time scaling by varying the sampling budget K and evaluating performance with pass@K and avg@K. Specifically, across several OPD variants, we observe that OPD-trained models maintain superior avg@K performance across sampling budgets, while the advantage in pass@K gradually shifts to the pre-OPD base models as K increases. These results suggest that OPD primarily improves sampling efficiency rather than consistently expanding the student's reasoning capability boundary. The pass@K dynamics throughout OPD training further reveal a progressive shift toward stronger small-K performance at the expense of the l
    
[^127]: BODHI：大语言模型能否分支探索并发现异构推理？

    BODHI: Do LLMs Branch Out and Discover Heterogeneous Inferences?

    [https://arxiv.org/abs/2608.02867](https://arxiv.org/abs/2608.02867)

    研究发现RLVR训练的大语言模型相比蒸馏模型表现出更低的语义多样性和更少的推理分支，表明其推理性能提升主要来自采样效率的改善而非推理能力的真正扩展。

    

    尽管基于可验证奖励的强化学习（RLVR）已经提升了对大语言模型（LLM）的推理性能，但它究竟是扩展了推理能力，还是主要提高了采样效率，这一问题仍然存在争议。在本文中，我们通过受控的迷宫求解实验和 BODHI-TREEs（一种可复用的表示方法，基于语义等价性对数学推理片段进行分组）来研究 RLVR 训练的大语言模型在测试时探索的本质。这些探测方法有助于刻画模型在风格变化与语义不同的续写之间的偏好。我们表明，与蒸馏模型相比，RLVR 训练的模型在语义不同的续写之间表现出更低的候选偏好熵；互补的采样实验表明，在相同的采样预算下，模型表现出更低的经验语义熵和更少的观测分支。因此，策略集中化不仅体现在风格层面……

    arXiv:2608.02867v2 Announce Type: replace-cross  Abstract: Although reinforcement learning with verifiable rewards (RLVR) has improved the reasoning performance of large language models (LLMs), whether it expands reasoning capabilities or primarily improves sampling efficiency remains debated. In this paper, we investigate the nature of test-time exploration in RLVR-trained LLMs by employing controlled maze-solving experiments and BODHI-TREEs, a reusable representation that groups mathematical reasoning segments based on semantic equivalence. These probes help delineate preferences between stylistic variations and semantically distinct continuations. We show that RLVR-trained models exhibit lower candidate preference entropy between semantically distinct continuations than their distilled counterparts, and complementary sampling experiments show lower empirical semantic entropy and fewer observed branches under equal sampling budgets. Thus, policy concentration extends beyond stylistic
    
[^128]: 探究提示词KV缓存：它在何处变得可有可无

    Probing the Prompt KV Cache: Where It Becomes Dispensable

    [https://arxiv.org/abs/2605.30574](https://arxiv.org/abs/2605.30574)

    本研究通过受控干预实验揭示提示词KV缓存的冗余在于形式而非内容——在深层用空内容或填充token的脚手架KV缓存替换提示词缓存即可几乎无损恢复精度，并在Qwen3、Gemma 3、Llama 3等模型上得到验证，实现中KV内存减少15.4%且多支持18%的并发请求。

    

    此前的KV缓存压缩方案已在经验上表明，在解码过程中丢弃或摘要化提示词缓存几乎不会造成精度损失，这说明提示词缓存部分是冗余的。我们进一步追问这种冗余何时出现、属于何种类型，即在哪一层、经过多少解码步骤之后，以及以何种形式替换提示词跨度的KV缓存而不会破坏任务。在层和解码步骤上进行系统性扫掠的受控拼接干预实验表明，这种冗余关乎形式而非内容：用来自空用户内容或填充token的聊天模板脚手架的KV缓存替换上层的提示词跨度KV缓存，可以恢复接近原始的精度，而将整个提示词KV置零则会导致精度崩溃。这种分离现象在Qwen3、Gemma 3和Llama 3等不同规模的模型家族以及多个数据集上均得到复现。该技术的服务化实现可将KV内存减少15.4%，并多支持18%受HBM限制的并发请求。

    arXiv:2605.30574v2 Announce Type: replace  Abstract: Prior KV cache compression schemes empirically demonstrate that the prompt cache can be dropped or summarised during decoding with little accuracy loss, indicating that it is partly redundant. We ask when and what kind of redundancy, i.e., at which layers, after how many decoding steps, and in what form can the prompt span KV cache be replaced without breaking the task. A controlled splice intervention swept over layers and decoding steps shows this redundancy is about form rather than content. Replacing the upper layer prompt span KV cache with KV cache from a chat template scaffold with empty user content or filler tokens recovers near clean accuracy, while zeroing the entire prompt KV collapses accuracy. The dissociation replicates across the Qwen3, Gemma 3, and Llama 3 model families of varying sizes on multiple datasets. A serving implementation of this technique reduces KV memory by $15.4\%$, supports $18\%$ more HBM-constraine
    
[^129]: BioELX：无需任务特定监督的上下文感知跨语言生物医学实体链接

    BioELX: Context-Aware Cross-lingual Biomedical Entity Linking without Task-Specific Supervision

    [https://arxiv.org/abs/2605.27380](https://arxiv.org/abs/2605.27380)

    BioELX提出了一种检索-重排序框架，利用Wikidata衍生的跨语言别名监督进行检索，并通过提及锚定提示将LLM重排序器适配于实体链接，实现了无需任务特定监督的上下文感知跨语言生物医学实体链接。

    

    跨语言生物医学实体链接（BEL）将任何语言中的提及（mention）映射到生物医学知识库中的唯一标识符，为临床和生物医学NLP应用提供支持。我们发现了影响当前系统的两个问题。首先，用于训练跨语言BEL检索器的UMLS别名严重偏向英语，导致检索器对非英语提及的泛化能力较差。其次，尽管上下文通常对消歧是必要的，但将上下文简单地注入仅以别名对齐为目标训练的检索器中，会严重降低检索性能。我们提出了BioELX，一个解决这两个问题的检索-重排序框架。在检索阶段，我们使用源自Wikidata的跨语言别名监督对SapBERT_multi进行继续训练，从而在各语言之间形成共享的概念邻域。在重排序阶段，我们通过提及锚定提示（mention-anchored prompting）将预训练的LLM重排序器适配到实体链接任务。

    arXiv:2605.27380v2 Announce Type: replace-cross  Abstract: Cross-lingual biomedical entity linking (BEL) maps mentions in any language to unique identifiers in a biomedical knowledge base, supporting clinical and biomedical NLP applications. We identify two issues affecting current systems. First, the UMLS (Bodenreider,2004) aliases used to train cross-lingual BEL retrievers are heavily skewed toward English, so retrievers generalize poorly to non-English mentions. Second, although context is often necessary for disambiguation, naively injecting context into retrievers trained only to align aliases severely degrades retrieval. We propose BioELX, a retrieve-rerank framework that addresses both issues. For retrieval, we continue training SapBERT_multi (Liu et al., 2021b) using Wikidata-derived cross-lingual alias supervision, forming shared concept neighborhoods across languages. For reranking, we adapt pretrained LLM rerankers to entity linking through mention-anchored prompting, which 
    
[^130]: 类别错误敏感性指数（ISEC）：一种针对人工数据录入系统中不可恢复错误的预防性序数决策支持度量

    A categorical error sensitivity index (ISEC): A preventive ordinal decision-support measure for irrecoverable errors in manual data entry systems

    [https://arxiv.org/abs/2605.12328](https://arxiv.org/abs/2605.12328)

    本文提出了类别错误敏感性指数（ISEC）——一种序数复合评分，通过量化类别对之间的结构易混淆性对类别对进行排序，为中小企业人工数据录入系统提供预防性决策支持，以避免不可恢复的类别错误传播至关键绩效指标而误导管理决策。

    

    数据录入系统在结构上仍然容易受到类别误分类的影响，尤其是在中小型企业（SME）中。当名义类别在语义或形态上表现出相近性时，人机交互可能产生事后无法恢复的错误。在缺乏自动化输入控制的情况下，人工数据录入经常产生不可恢复的类别失真，这些失真会传播到关键绩效指标（KPI）中，从而误导管理决策。最先进的规范化工具通常孤立地评估语义和形态维度，并严重依赖标准词典，这使得它们对于富含自定义SKU、缩写和领域特定技术术语的中小企业主数据效果不佳。本文提出了类别错误敏感性指数（ISEC），这是一个序数复合评分，旨在根据类别对的结构易混淆程度对其进行排序……

    arXiv:2605.12328v2 Announce Type: replace  Abstract: Data entry systems remain structurally vulnerable to categorical misclassifications, particularly in small and medium sized enterprises (SMEs). When nominal categories exhibit semantic or morphological proximity, human machine interaction may produce errors that are irrecoverable ex post. In the absence of automated input controls, manual data entry frequently generates irrecoverable categorical distortions that propagate into Key Performance Indicators (KPIs), thereby misleading managerial decision making. State of the art normalization tools typically evaluate semantic and morphological dimensions in isolation and rely heavily on standard dictionaries, rendering them ineffective for SME master data rich in custom SKUs, abbreviations, and domain-specific technical jargon. This paper introduces the Categorical Error Sensitivity Index (ISEC), an ordinal composite score designed to rank category pairs according to their structural susc
    
[^131]: 何时以及为何SignSGD优于SGD：基于ℓ₁-范数下界的理论研究

    When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds

    [https://arxiv.org/abs/2605.06615](https://arxiv.org/abs/2605.06615)

    本文通过引入ℓ₁-范数平稳性、ℓ∞-平滑性和可分离噪声模型这一新问题几何框架，首次从理论上证明了SignSGD在何种条件下以及为何能够超越SGD，并给出了匹配的复杂度上下界。

    

    基于符号的优化算法，如SignSGD和Muon，因其在训练大型基础模型中的卓越表现而受到广泛关注。尽管取得了这些实证成功，我们仍然缺乏对这些基于符号的方法何时以及为何优于普通SGD的理论理解。核心障碍在于，在标准的平滑性和有限方差条件下，SGD已被证明在以ℓ₂-范数度量驻点的优化问题中是极小极大最优的，这从根本上排除了基于符号的方法在标准设置中取得任何复杂度增益的可能性。为克服这一障碍，我们利用ℓ₁-范数平稳性、ℓ∞-平滑性以及可分离噪声模型来分析基于符号的优化器，这些条件能够更好地刻画符号更新的逐坐标特性。在这种独特的问题几何结构下，我们推导出了SignSGD的匹配上下界，并明确刻画了问题的类别……

    arXiv:2605.06615v2 Announce Type: replace-cross  Abstract: Sign-based optimization algorithms, such as SignSGD and Muon, have garnered significant attention for their remarkable performance in training large foundation models. Despite this empirical success, we still lack a theoretical understanding of when and why these sign-based methods outperform vanilla SGD. The core obstacle is that under standard smoothness and finite variance conditions, SGD is known to be minimax optimal for finding stationary points measured by $\ell_2$-norms, thereby fundamentally precluding any complexity gains for sign-based methods in standard settings. To overcome this barrier, we analyze sign-based optimizers leveraging $\ell_1$-norm stationarity, $\ell_\infty$-smoothness, and a separable noise model, which can better capture the coordinate-wise nature of signed updates. Under this distinct problem geometry, we derive matched upper and lower bounds for SignSGD and explicitly characterize the problem cla
    
[^132]: 计算最优的词元化

    Compute Optimal Tokenization

    [https://arxiv.org/abs/2605.01188](https://arxiv.org/abs/2605.01188)

    该论文通过训练988个不同压缩率的潜在词元化（BLT）模型发现，在计算最优配置下模型参数量应与以字节（而非词元）衡量的数据量成正比地缩放，且最优压缩率不同于常用BPE分词器所得的结果。

    

    缩放定律使我们能够最优地选择数据量和语言模型规模，然而作为数据基本单位的“词元”对这一关系的影响仍未得到充分探索。在这项工作中，我们系统地研究了词元的信息粒度——由压缩率（即每个词元平均对应的文本字节数）所控制——如何影响缩放趋势。我们训练了988个潜在词元化模型（BLT），参数量从5000万到70亿不等，这些模型允许设定所需的压缩率。这种灵活性使我们能够研究远超常用BPE分词器所得的每词元4.57字节的压缩率范围。我们的实验揭示，在计算最优配置下，模型参数量与以字节（而非通常认为的词元）衡量的数据规模成正比地缩放。此外，我们发现最优压缩率不同于BPE分词器所得的结果，并且随……（原文摘要在此处截断）

    arXiv:2605.01188v3 Announce Type: replace  Abstract: Scaling laws enable the optimal selection of data amount and language model size, yet the impact of the data unit, the token, on this relationship remains underexplored. In this work, we systematically investigate how the information granularity of tokens, controlled by the compression rate (i.e., average bytes of text per token), affects scaling trends. We train 988 latent tokenized models (BLT) ranging from 50M to 7B parameters that enable setting the desired compression rate. This flexibility allows us to study the role of compression rate well beyond 4.57 bytes per token obtained with a popular BPE tokenizer. Our experiments reveal that in compute-optimal configurations, model parameter counts scale proportionally to data size measured in bytes, not in tokens as commonly perceived (Kaplan et al., 2020; Hoffmann et al., 2022). Furthermore, we discover that the optimal compression rate differs from the one obtained with BPE and dec
    
[^133]: 自蒸馏推理器：面向大语言模型的同策略自蒸馏方法

    Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

    [https://arxiv.org/abs/2601.18734](https://arxiv.org/abs/2601.18734)

    本文提出同策略自蒸馏（OPSD）学习算法，让单个LLM在不同上下文中同时充当教师与学生——教师以特权信息（如经过验证的推理轨迹）为条件对学生进行密集监督，从而无需独立的教师模型，并显式利用了推理数据集中的标准答案。

    

    知识蒸馏通过压缩教师大语言模型（LLM）的知识来训练更小的LLM，从而提升其推理能力。同策略蒸馏（on-policy distillation）进一步推进了这一方法：由学生模型采样自身的轨迹，同时教师LLM提供密集的词元级监督，从而解决了离策略蒸馏方法中训练与推理之间的分布不匹配问题。然而，同策略蒸馏通常需要一个独立的、往往规模更大的教师LLM，并且没有显式利用推理数据集中已有的标准答案。受到一个直觉的启发——即足够强大的LLM能够对外部特权推理轨迹进行合理化解释并教会较弱的自身——我们提出了同策略自蒸馏（On-Policy Self-Distillation, OPSD），这是一种让单个LLM在不同上下文中同时充当教师与学生的学习算法。教师策略以特权信息（例如经过验证的推理轨迹）为条件进行指导……（摘要原文在此处截断）

    arXiv:2601.18734v4 Announce Type: replace-cross  Abstract: Knowledge distillation improves large language model (LLM) reasoning by compressing the knowledge of a teacher LLM to train smaller LLMs. On-policy distillation advances this approach by having the student sample its own trajectories while a teacher LLM provides dense token-level supervision, addressing the distribution mismatch between training and inference in off-policy distillation methods. However, on-policy distillation typically requires a separate, often larger, teacher LLM and does not explicitly leverage ground-truth solutions available in reasoning datasets. Inspired by the intuition that a sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self, we introduce On-Policy Self-Distillation (OPSD), a learning algorithm where a single LLM acts as both teacher and student with different contexts. The teacher policy conditions on privileged information (e.g., verified reasonin
    
[^134]: 先验证再生成：面向视频生成的草图引导高效运动规划

    Verify Before You Generate: Sketch-Guided Efficient Motion-Planning for Video Generation

    [https://arxiv.org/abs/2511.17450](https://arxiv.org/abs/2511.17450)

    提出了 SketchVerify，一个无需训练的框架，在调用昂贵的视频生成器之前先通过轻量级视频草图验证运动规划，从而高效比较并迭代精化多条候选轨迹，以低成本获得高质量运动。

    

    近期的视频生成方法越来越依赖规划中间控制信号（如物体轨迹）来提升时间一致性和运动保真度。然而，这些方法大多采用单次规划，通常局限于简单的运动指令，导致运动质量欠佳；或者采用迭代精化，需要多次调用视频生成器并产生高昂的计算成本。为解决这一权衡问题，我们提出了 SketchVerify，这是一个无需训练的框架，其核心洞察是在调用昂贵的视频生成器之前，先使用轻量级视频草图来验证运动规划。这一轻量级规划空间使得能够在最终的单次视频生成之前，高效比较多条候选轨迹并进行有界的迭代精化。给定提示词和参考图像，SketchVerify 采样多条候选轨迹，并将每条轨迹渲染为轻量级视频……

    arXiv:2511.17450v2 Announce Type: replace-cross  Abstract: Recent video generation approaches increasingly rely on planning intermediate control signals like object trajectories to improve temporal coherence and motion fidelity. However, these methods mostly employ single-shot plans that are typically limited to simple motion instructions, leading to suboptimal motion quality, or iterative refinement which requires multiple calls to the video generator and incurs high computational cost. To address this trade-off, we propose SketchVerify, a training-free framework whose key insight is to verify motion plans using lightweight video sketches before invoking the expensive video generator. This lightweight planning space enables efficient comparison of multiple candidate trajectories and bounded iterative refinement before a single final video-generation pass. Given a prompt and a reference image, SketchVerify samples multiple candidate trajectories and renders each as a lightweight video 
    
[^135]: 将测试时计算最优扩展泛化为可优化的图

    Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph

    [https://arxiv.org/abs/2511.00086](https://arxiv.org/abs/2511.00086)

    该论文提出将测试时扩展中在固定预算下搜索计算最优模型组合与架构的新问题形式化为多LLM协作图，并将其重新表述为概率图优化问题，以应对庞大的组合搜索空间和任务定制化需求。

    

    测试时扩展通过在推理过程中分配额外的计算来提升大型语言模型（LLM）的性能，通常采用并行、串行或混合扩展方式。然而，先前的研究往往假设固定的协作架构（例如拓扑结构）以及单一模型的使用，忽略了最优架构和模型组合可能因任务不同而变化。因此，我们研究了一个新问题：在固定预算下搜索测试时扩展中计算最优的模型组合与架构。我们将其形式化为一个多LLM协作图，其中节点编码角色和LLM模型分配，边则捕捉信息流。这个问题具有挑战性，原因在于： 组合搜索空间极其庞大， 任务特定需求需要定制化的设计。为解决这些问题，我们将该问题重新表述为概率图优化，并通过初步实验得出了三条经验性……（原文摘要在此处截断）

    arXiv:2511.00086v2 Announce Type: replace-cross  Abstract: Test-Time Scaling (TTS) improves large language models (LLMs) by allocating additional computation during inference, typically through parallel, sequential, or hybrid scaling. However, prior studies often assume fixed collaboration architectures (e.g., topologies) and single-model usage, overlooking that optimal architectures and model combinations can vary across tasks. Therefore, we study the novel problem of searching for compute-optimal model combinations and architectures in TTS under a fixed budget. We formalize it as a multi-LLM collaboration graph, where nodes encode roles and LLM model assignments, and edges capture information flow. This problem is challenging because (i) the combinatorial search space is prohibitively large, and (ii) task-specific requirements demand tailored designs. To address these, we reformulate the problem as probabilistic graph optimization and, through pilot experiments, derive three empirica
    
[^136]: 面向几何感知指令遵循的监督微调

    Supervised Fine-tuning for Geometry-aware Instruction-following

    [https://arxiv.org/abs/2510.26183](https://arxiv.org/abs/2510.26183)

    本文提出一种在监督微调中遮蔽错误生成内容至分类控制标记的对比输入编码方法，在保持生成准确率的同时提高了生成结果落入几何感知的校准SDM估计器高概率区域的接受率，且无需对语言模型参数进行专门微调。

    

    我们研究了一种用于语言模型（LM）监督微调的对比输入编码方案，该方案将错误生成的内容遮蔽至一个分类控制标记为止。这项小规模研究使用了经典词序诊断任务的一个变体，结果表明这种简单直接的方法通常能够保持仅使用正例进行监督微调时的生成准确率，同时提高了生成结果落入用于指令遵循二元分类的相似性-距离-幅度（Similarity-Distance-Magnitude, SDM）估计器校准高概率区域的接受率。与此同时，我们证实了已有证据：构建一个建模特征表示空间几何结构的校准SDM估计器，并不严格依赖于对底层语言模型参数的专门微调。

    arXiv:2510.26183v2 Announce Type: replace  Abstract: We examine a contrastive input encoding scheme for supervised fine-tuning of language models (LMs) that masks the content of incorrect generations up to a classification control token. This small-scale study, using a variation of the classic word ordering diagnostic task, suggests this straightforward approach generally preserves the generation accuracy of supervised fine-tuning that uses only positive examples, while increasing the admission rate of generations in the calibrated high-probability region of a Similarity-Distance-Magnitude (SDM) estimator for binary classification of instruction-following. At the same time, we corroborate existing evidence that constructing a calibrated SDM estimator, which models the geometry of the feature-representation space, is not strictly dependent on specialized fine-tuning of the underlying LM's parameters.
    
[^137]: 基于上下文学习的涌现式失准：狭窄的上下文示例可产生广泛失准的大语言模型

    Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned LLMs

    [https://arxiv.org/abs/2510.11288](https://arxiv.org/abs/2510.11288)

    该研究首次发现“涌现式失准”现象同样会通过上下文学习产生——仅用少量（最少2个）狭窄领域的上下文示例，就能使四个主流大语言模型家族对无关的良性问题产生广泛失准响应，且更大模型反而更易受影响，其机制可解释为安全目标与上下文遵循行为之间的冲突。

    

    近期研究表明，狭窄的微调可以产生广泛失准的大语言模型（LLMs），这种现象被称为涌现式失准。尽管令人担忧，但这些发现仅限于微调和激活引导，并未涉及上下文学习（ICL）。因此我们提出问题：EM是否会在ICL中出现？我们发现确实会出现：在四个模型家族（Gemini、Kimi-K2、Grok和Qwen）中，狭窄的上下文示例会导致模型对良性、无关的查询产生失准响应。使用16个上下文示例时，EM率在1%到24%之间（取决于模型和领域），而仅需2个示例即可出现该现象。更大的模型规模和显式推理都无法提供可靠保护，且更大的模型通常反而更容易受到影响。随后，我们提出并检验了一个假设：上下文中的EM源于安全目标与上下文遵循行为之间的冲突。与此一致的是，指示模型优先考虑安全……

    arXiv:2510.11288v5 Announce Type: replace  Abstract: Recent work has shown that narrow finetuning can produce broadly misaligned LLMs, a phenomenon termed emergent misalignment (EM). While concerning, these findings were limited to finetuning and activation steering, leaving out in-context learning (ICL). We therefore ask: does EM emerge in ICL? We find that it does: across four model families (Gemini, Kimi-K2, Grok, and Qwen), narrow in-context examples cause models to produce misaligned responses to benign, unrelated queries. With 16 in-context examples, EM rates range from 1% to 24% depending on model and domain, appearing with as few as 2 examples. Neither larger model scale nor explicit reasoning provides reliable protection, and larger models are typically even more susceptible. Next, we formulate and test a hypothesis, which explains in-context EM as conflict between safety objectives and context-following behavior. Consistent with this, instructing models to prioritize safety r
    
[^138]: RAG让防护栏变得脆弱？研究RAG式上下文下安全防护栏的鲁棒性

    RAG Makes Guardrails Brittle? Investigating Robustness of Guardrails under RAG-style Contexts

    [https://arxiv.org/abs/2510.05310](https://arxiv.org/abs/2510.05310)

    本研究通过系统评估发现，RAG场景中仅插入良性的检索文档就能使基于LLM的安全防护栏在约10%的情况下翻转判断，且多数翻转会将原本正确的安全决策反转为错误决策，揭示了现有防护栏在RAG式上下文下的脆弱性。

    

    随着大语言模型（LLM）的日益普及，确保LLM系统的安全性已成为一个紧迫的问题。基于LLM的外部防护栏模型已成为筛选不安全输入和输出的流行解决方案，但它们本身是经过微调或提示工程设计的LLM，容易受到数据分布偏移的影响。本文以检索增强生成（RAG）为案例研究，考察基于LLM的防护栏对嵌入上下文中额外信息的鲁棒性。通过对3个Llama Guard模型和2个GPT-oss模型的系统性评估，我们发现仅仅插入良性的检索文档，就会使输入和输出防护栏的判断分别在大约11%和8%的情况下发生翻转，其中大多数翻转（72%和64%）是将原本正确的判断反转为错误判断。我们进一步分析了增强上下文中各组件（检索文档、用户查询和LLM）的影响。

    arXiv:2510.05310v2 Announce Type: replace-cross  Abstract: With the increasing adoption of large language models (LLMs), ensuring the safety of LLM systems has become a pressing concern. External LLM-based guardrail models have emerged as a popular solution to screen unsafe inputs and outputs, but they are themselves fine-tuned or prompt-engineered LLMs vulnerable to data distribution shifts. In this paper, taking Retrieval Augmentation Generation (RAG) as a case study, we investigate how robust LLM-based guardrails are against additional information embedded in the context. Through a systematic evaluation of 3 Llama Guards and 2 GPT-oss models, we find that inserting benign retrieved documents flips the judgments of input and output guardrails in around 11% and 8% of cases respectively, with the majority of these flips (72% and 64%) reversing previously correct decisions. We further analyze the effect of each component in the augmented context: retrieved documents, user query, and LLM
    
[^139]: 利用组相似度奖励提升检索增强系统的一致性

    Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards

    [https://arxiv.org/abs/2510.04392](https://arxiv.org/abs/2510.04392)

    本文提出PS-GRPO强化学习方法，通过跨复述集的组相似度奖励来提升RAG系统在语义等价查询下的信息一致性，并提供了将一致性分解为检索器、生成器和端到端层面的系统性评估框架。

    

    检索增强生成（RAG）系统越来越多地被部署在高风险领域，用户期望其输出在语义等价的查询之间保持一致。然而，由于检索器和生成器双方均存在变异性，现有系统常常表现出显著的不一致性，损害了信任与可靠性。在这项工作中，我们关注信息一致性，即生成的输出在语义等价的输入之间应传达相同核心内容的要求。我们引入了一个有原则性的评估框架，将RAG一致性分解为检索器级别、生成器级别和端到端组件，从而能够系统地识别不一致性的来源。为了提升一致性，我们提出了复述集组相对策略优化（PS-GRPO），这是一种强化学习方法，通过在复述集上进行多次rollout来分配组相似度奖励。我们利用PS-GRPO来实现

    arXiv:2510.04392v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) systems are increasingly deployed in high-stakes domains where users expect outputs to be consistent across semantically equivalent queries. However, existing systems often exhibit significant inconsistencies due to variability in both the retriever and generator, undermining trust and reliability. In this work, we focus on information consistency, i.e., the requirement that generated outputs convey the same core content across semantically equivalent inputs. We introduce a principled evaluation framework that decomposes RAG consistency into retriever-level, generator-level, and end-to-end components, enabling systematic identification of inconsistency sources. To improve consistency, we propose Paraphrased Set Group Relative Policy Optimization (PS-GRPO), an RL approach that leverages multiple rollouts across paraphrased set to assign group similarity rewards. We leverage PS-GRPO to achieve
    
[^140]: 家族很重要：面向法罗语的小型语言模型适配的语言迁移与模型合并

    Family Matters: Language Transfer and Merging for Adapting Small LLMs to Faroese

    [https://arxiv.org/abs/2510.00810](https://arxiv.org/abs/2510.00810)

    该论文的核心贡献是通过先在斯堪的纳维亚亲缘语言上继续预训练再适配法罗语，并构建两个全新的法罗语最小对探针评估基准，证明了语言家族内的迁移能有效提升低资源语言的语言建模与可接受性。

    

    本文研究了将小型语言模型适配到法罗语（Faroese，一种低资源的北日耳曼语言）的策略。研究者从英语预训练模型出发，先在其他斯堪的纳维亚语言上进行继续预训练——可以是单独使用某一种语言，也可以通过模型合并或多语言预训练的方式组合使用——然后再在法罗语上进行微调。针对现有法罗语评估资源匮乏的问题，作者构建了两个新的最小对探针基准：一个用于语言可接受性测试，另一个用于文本理解测试，并由母语法罗语的语言学家进行人工评估加以补充。结果表明，来自相关语言的迁移能够提升语言建模和语言可接受性，但在文本理解探针上的收益在不同模型规模和调优方法下表现并不一致。冰岛语是语言可接受性方面最强的单一源语言，但没有哪一种语言在理解任务上能够稳定地表现最优。

    arXiv:2510.00810v3 Announce Type: replace  Abstract: In this paper, we investigate strategies for adapting small language models to Faroese, a low-resource North Germanic language. Starting from English-pretrained models, we first continue pre-training on other Scandinavian languages, either individually or combined via model merging or multilingual pre-training, before fine-tuning on Faroese. To address the lack of existing Faroese evaluation resources, we construct two new minimal-pair probing benchmarks, one for linguistic acceptability and one for text comprehension, and complement them with human evaluations conducted by native Faroese linguists. Our results show that transfer from related languages improves language modeling and linguistic acceptability, while gains on comprehension probes are inconsistent across model sizes and tuning methods. Icelandic is the strongest single source language for linguistic acceptability, but no single language is reliably best for comprehension
    
[^141]: CHAIR——作为改进器的幻觉分类器

    CHAIR -- Classifier of Hallucination as Improver

    [https://arxiv.org/abs/2501.02518](https://arxiv.org/abs/2501.02518)

    CHAIR通过提取每个词元在所有层中logits的紧凑统计特征来检测大语言模型的幻觉，在零样本场景下显著提升检测准确率，并展示了利用内部表示设计自适应解码策略以减少幻觉的潜力。

    

    在这项工作中，我们提出了CHAIR（Classifier of Hallucination As ImproveR），一个通过分析每个词元在每一层的内部logits来检测幻觉的监督式框架。我们的方法从所有层的词元logits中提取一组紧凑的特征，如最大值、最小值、均值、标准差和斜率，从而在不发生过拟合的情况下实现有效的幻觉检测。在TruthfulQA和MMLU数据集上的实验表明，CHAIR显著提升了检测准确率，尤其是在零样本场景中，展现了其鲁棒性和泛化能力。除了幻觉检测之外，CHAIR还凸显了利用内部表示来设计高级解码策略的潜力。通过利用logits中的模式，我们认为更精细的模型和自适应解码方法可以进一步减少幻觉并提升文本补全的质量。CHAIR不仅提供了一个实用的（原文摘要在此处截断）……

    arXiv:2501.02518v3 Announce Type: replace  Abstract: In this work, we introduce CHAIR (Classifier of Hallucination As ImproveR), a supervised framework for detecting hallucinations by analyzing internal logits from each layer of every token. Our method extracts a compact set of features such as maximum, minimum, mean, standard deviation, and slope-from the token logits across all layers, enabling effective hallucination detection without overfitting. Experiments on TruthfulQA and MMLU datasets demonstrate that CHAIR significantly improves detection accuracy, particularly in zero-shot scenarios, showcasing its robustness and generalizability. Beyond hallucination detection, CHAIR highlights the potential of using internal representations for designing advanced decoding strategies. By leveraging patterns in logits, we suggest that more sophisticated models and adaptive decoding methods could further reduce hallucinations and enhance text completion quality. CHAIR not only offers a practi
    
[^142]: OpinionGPT: 模拟显性偏见的指令调整大型语言模型(LLMs)

    OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs. (arXiv:2309.03876v1 [cs.CL])

    [http://arxiv.org/abs/2309.03876](http://arxiv.org/abs/2309.03876)

    OpinionGPT是一个指令调整大型语言模型(LLMs)的web演示，用户可以选择各种偏见并进行比较，它意在使模型的偏见显性和透明化。

    

    最近，指令调整大型语言模型(LLMs)展示了生成与自然语言指令相匹配的回应的显著能力。然而，一个开放的研究问题涉及训练模型和它们的回应中固有的偏见。例如，如果用于调整LLM的数据主要由具有特定政治偏见的人编写，我们可能会期望生成的回答也共享这种偏见。目前的研究工作旨在除去这样的模型偏见，或抑制可能有偏见的回答。通过这个演示，我们对指令调整中的偏见持有不同的观点：我们的目标不是抑制它们，而是使它们显性和透明。为此，我们提供了OpinionGPT，一个网络演示，用户可以提问并选择所有他们希望调查的偏见。该演示将使用在代表每个选择偏见的文本上进行微调的模型来回答这个问题，从而实现并排比较。为了训练基础模型，我们选取了11个...

    Instruction-tuned Large Language Models (LLMs) have recently showcased remarkable ability to generate fitting responses to natural language instructions. However, an open research question concerns the inherent biases of trained models and their responses. For instance, if the data used to tune an LLM is dominantly written by persons with a specific political bias, we might expect generated answers to share this bias. Current research work seeks to de-bias such models, or suppress potentially biased answers. With this demonstration, we take a different view on biases in instruction-tuning: Rather than aiming to suppress them, we aim to make them explicit and transparent. To this end, we present OpinionGPT, a web demo in which users can ask questions and select all biases they wish to investigate. The demo will answer this question using a model fine-tuned on text representing each of the selected biases, allowing side-by-side comparison. To train the underlying model, we identified 11 
    

