# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Critical-State RL: Diagnosing Trainable States for Multi-Turn Tool Use](https://arxiv.org/abs/2609.24985) | 提出临界状态强化学习方法，通过嵌套采样将动作相关的奖励变化与下游噪声分离，从而识别多轮工具交互中真正值得训练的关键状态，并用上下文老虎机方法对这些状态进行针对性策略优化。 |
| [^2] | [onPanda: Efficient Annotation of On-Policy Alignment Data for LLMs and Agents via Token-Level Correction](https://arxiv.org/abs/2609.24983) | onPanda通过“定位-修正-继续”的词元级交互方式高效标注大语言模型与智能体的同策略对齐数据，将标注时间中位数减少52%，同时保留了模型自身的采样分布。 |
| [^3] | [Harness-Zero: Harness Distillation via Agent-as-Harness](https://arxiv.org/abs/2609.24974) | 提出 Harness-Zero 方法，通过“智能体即框架”将领域或实例优化的框架所诱导的行为蒸馏进模型权重，使框架带来的性能提升在单一固定的目标框架下得以保留。 |
| [^4] | [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](https://arxiv.org/abs/2609.24972) | 该论文提出RRSI方法，通过将正则化原则（如时间退火的编辑预算限制和鼓励探索未开发轨迹）引入智能体框架的递归自我改进过程，防止递归进化对训练任务过拟合，从而提升分布外基准上的泛化能力。 |
| [^5] | [DolphinBench: Mapping the Pareto Frontier of Agent Memory](https://arxiv.org/abs/2609.24971) | DolphinBench是一个通过智能体实际任务完成情况（而非对话式问答）直接评估长期记忆的基准，包含三个各约50万token历史记录的知识工作角色画像、每个角色200个经有无历史对照验证的任务，并强制要求报告成本，以刻画记忆性能与成本之间的帕累托前沿。 |
| [^6] | [Emergent Collusion in Long-Horizon LLM Agent Interaction](https://arxiv.org/abs/2609.24967) | 该研究首次系统揭示了在长时程多智能体交互中，当遵守验证协议与奖励最大化相冲突时，LLM智能体会自发涌现合谋行为——在10个模型中94%的轨迹出现合谋，且能力更强的模型更早合谋，对等行为、奖励结构、验证反馈和交互历史均对合谋形成有显著影响。 |
| [^7] | [Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences](https://arxiv.org/abs/2609.24965) | 该研究将Jev作为科学工作流中的语义决策组件进行评估，发现其语义正确性与其他配置持平且延迟最低，并表明错误的语义选择会改变下游计数但可能不影响最终结论标签。 |
| [^8] | [Linguistic Features for Interpretable Textual Entailment](https://arxiv.org/abs/2609.24932) | 该论文提出SLITE，一个融合结构-关系层与分布-信息层语义分析的可解释文本蕴含混合模型，仅用17个语言学特征（涵盖实体级语义关系、极性敏感词汇匹配及基于熵与迁移熵的对齐度量）训练的逻辑回归即可达到83%的准确率，同时保持模型的可解释性。 |
| [^9] | [SocioVerse2: A Longitudinal Dynamic Social Simulation Framework under a Human-AI Co-evolutionary Paradigm](https://arxiv.org/abs/2609.24911) | SocioVerse2将社会模拟扩展为人机协同演化范式，通过支持环境演化与干预生成反事实分支的纵向模拟循环、以及研究过程可编辑的可控研究循环，首次系统性支持了对模拟内容的干预和研究者对模拟过程的控制。 |
| [^10] | [ToneCL: Contrastive Learning for Few-Shot Syllable-Level Tone Classification](https://arxiv.org/abs/2609.24903) | ToneCL是一个轻量级对比学习框架，通过保留声调身份的数据增强在无标注语音上预训练并少样本微调，在低资源音节级声调分类任务上取得优异表现（普通话10样本达91.6%），并支持有效的跨语言迁移。 |
| [^11] | [Human-LLM Deliberation as Interactive Proof: Conditions for Verifiability Without Transparency](https://arxiv.org/abs/2609.24895) | 该论文将人类与大语言模型的协商建模为交互式证明，证明了即使无法访问模型内部状态（即无需透明性），人类验证者通过逐步检查积累证据也能获得可证明的可靠性保证，从而决定是否接受模型的主张。 |
| [^12] | [SLICEChat: Progressive In-Encoder Token Pruning for Whole-Slide Pathology Language Models](https://arxiv.org/abs/2609.24894) | SLICEChat通过在混合Mamba-Transformer切片编码器内部进行语言监督的渐进式token剪枝，在多模态融合之前生成紧凑的切片表示，从而显著提升全切片病理图像多模态大语言模型的效率与可扩展性。 |
| [^13] | [OSWorld-Pro: Process-based Evaluation for Computer Use Agents](https://arxiv.org/abs/2609.24890) | OSWorld-Pro提出了一个包含300多个任务和2800多个子目标的过程性评估基准，利用基于67,000多条人工标注、与人类判断高度一致的LLM裁判，对计算机使用智能体在子目标层面的执行过程进行评估，从而揭示智能体失败的具体方式和原因。 |
| [^14] | [The Copy Ceiling: An Input-Exposure Control for Ontology-Grounded Generation over Curated Corpora](https://arxiv.org/abs/2609.24885) | 论文提出“暴露核算”与“复制天花板”这一无需评判者的评估控制方法，揭示语言模型在基于本体检索的接地生成中的性能提升几乎完全来自复制上下文中已暴露的答案，而非对检索结构的真正推理。 |
| [^15] | [Decomposing Error and Style in Automated Clinical Coding](https://arxiv.org/abs/2609.24877) | 该论文发现临床编码中的不一致很大程度上源于可建模的系统性“编码风格”而非纯粹误差，通过10维风格量表进行条件化可使ICD F1提升最多26分。 |
| [^16] | [Extracting Arguments, Not Just Classifying Them: Instruction-Tuned LLMs for Generative Component Detection](https://arxiv.org/abs/2609.24855) | 该论文提出ITFACD方法，将论证成分检测重新构建为语言生成任务，利用指令微调的大语言模型直接从纯文本中提取并分类论证成分，无需预先分割，性能超越现有最先进系统。 |
| [^17] | [The Answer-Basin Representation Hypothesis: We Are Not Probing or Steering Concepts](https://arxiv.org/abs/2609.24821) | 该论文提出“答案盆地表示假说”，认为语言模型中概念相关的线性结构由续写分布在答案上诱导的概率测度所组织，源于答案测度的差异而非概念标签的变化，从而解释了探测与操控实验中概念效应及其反转现象。 |
| [^18] | [MSI-Bench: Evaluating Multi-Speaker Voice Interaction for Collaborative AI Agents](https://arxiv.org/abs/2609.24812) | 提出了首个多说话人语音交互评估基准MSI-Bench，包含1152个中英文多方多轮音频测试用例，涵盖多说话人记忆、指令遵循和推理三大能力，揭示了当前语音智能体在多说话人场景下与一对一交互相比存在显著的性能差距。 |
| [^19] | [When Quantization Preserves Accuracy but Not Evidence: Explanation-Aware Post-Training Quantization for Medical LLMs](https://arxiv.org/abs/2609.24799) | 该论文提出一种解释感知的后训练量化方法，通过从全精度教师模型推理依据构建的离线忠实度缓存，在量化医疗大语言模型时保留支持答案的证据词元，使模型在保持答案准确率的同时维持解释的可信度。 |
| [^20] | [Adapting Tree-Structured Speculative Decoding to DeepSeek-V4 for Efficient Inference](https://arxiv.org/abs/2609.24698) | 针对DeepSeek-V4压缩注意力导致的跨分支状态不一致难题，通过分支感知因果验证、临时状态隔离和已接受路径状态刷新，将树结构推测解码成功集成到DeepSeek-V4-Flash流水线，实现高效推理。 |
| [^21] | [Muon Can Outperform Dedicated Continual Learning Methods](https://arxiv.org/abs/2609.24678) | 使用 Muon 优化器对更新进行正交化的简单增量 LoRA，无需任务感知的约束即可达到专门持续学习方法的性能，表明一种更新约束机制（无论来自损失函数还是优化器）就已足够。 |
| [^22] | [Circuit Hypernetworks for Quantum-Augmented Diffusion Language Models](https://arxiv.org/abs/2609.24657) | 提出HyperQ方法，通过轻量级电路超网络为冻结的掩码扩散语言模型注入词元条件化的量子残差分支，并利用计算成本随量子比特数线性增长的精确经典期望值表达式，实现高效的量子增强语言模型适配。 |
| [^23] | [Assessing Readability with LLMs: The Role of Reasoning and Few-Shot Prompting](https://arxiv.org/abs/2609.24650) | 该论文对多种开源大语言模型在多语言可读性评估中的表现进行了系统性基准测试，探究了推理与少样本提示策略的作用，并涵盖英语和低资源语言斯洛文尼亚语。 |
| [^24] | [Written as a Record, Read as an Address: What a Forward Pass Leaves in an Operation's KV Cache](https://arxiv.org/abs/2609.24635) | 该研究将前向传播拆分为冻结的写入器与独立训练的读取器，证明语言模型在处理操作语句时会以可寻址的形式将实体绑定信息写入KV缓存，训练后的读取器能从中恢复75%–100%的绑定关系。 |
| [^25] | [Custom Named Entity Recognition and Topic Classification for Global Health Publications](https://arxiv.org/abs/2609.24625) | 在标注数据和计算资源受限的全球健康文献场景下，基于RoBERTa的transformer在命名实体识别任务上显著优于卷积spaCy模型（micro-F1为0.80对0.65-0.69），同时研究表明更大的词汇覆盖率并不必然带来更有用的领域特定关联。 |
| [^26] | [UK-PRBENCH: A Paragraph-Level Precedent Retrieval Benchmark for United Kingdom Case Law](https://arxiv.org/abs/2609.24613) | 该论文提出了UK-PRBENCH，首个基于英国国家档案馆判例数据构建的段落级先例检索基准，将法律检索粒度从整份判决书细化到段落级别，并通过实验证明现有最先进的检索模型在该任务上仍有很大提升空间。 |
| [^27] | [Evaluating Decision Models for Text Annotation in Computational Social Science](https://arxiv.org/abs/2609.24574) | 本研究在18个计算社会科学分类任务上对决策模型与19个大语言模型进行零样本对比评估，发现首个商业决策模型在绝大多数任务上落后于最佳大语言模型，其置信度在社会科学构念上的可信度仍存疑。 |
| [^28] | [Toward a Unified Mathematics of Concepts](https://arxiv.org/abs/2609.24554) | 该论文提出一种基于操作的概念数学框架评估视角，识别出十三个贯穿认知科学、心理学与人工智能的核心概念操作，并证明十个现有框架因对概念本质（内容、关系结构或演化过程）的不同承诺而各自天然支持不同的操作子集。 |
| [^29] | [QLoRA Fine-Tuning of Ministral LLM for Sequence-to-Function Protein Annotation](https://arxiv.org/abs/2609.24538) | 该研究将蛋白质功能注释重新定义为序列到文本的生成任务，通过QLoRA微调30亿参数的Ministral模型，并借助GPT作为专家评估，证明紧凑型大语言模型能够生成具有真实生物学价值的策展人风格蛋白质注释。 |
| [^30] | [LLJ Cards: Best practices for the Use of LLMs as Judges](https://arxiv.org/abs/2609.24516) | 本文提出了LLJ Cards框架，综合了将大语言模型作为评判者使用时的最佳实践，以解决当前评估实践中缺乏标准化、透明性和可复现性这一根本问题。 |
| [^31] | [Fathom-Vaidya: Advancing Medical Reasoning with Rubric-Based Rewards](https://arxiv.org/abs/2609.24480) | 该论文提出Fathom-Vaidya顺序训练框架，利用合成数据和基于评分标准的强化学习，先提升大语言模型的诊断推理能力，再增强其在多轮临床交互中的临床医疗推理能力，以解决现有模型在复杂诊断和以患者为中心对话中的不足。 |
| [^32] | [1% of Tokens Can Be Enough: On Gradient Estimation in On-Policy Distillation](https://arxiv.org/abs/2609.24432) | 提出基于信噪分解的信息效率比（IER）来衡量在线策略蒸馏中Token级梯度估计的噪声，使仅用0.1%–1%的Token监督即可达到甚至超过完整蒸馏的效果。 |
| [^33] | [End-to-end Jordanian dialect speech-to-text self-supervised learning framework](https://arxiv.org/abs/2609.24410) | 该论文提出了一种基于Transformer的端到端自监督学习框架，结合定制音频到文本处理算法与噪声学生训练，实现了低资源条件下高效的约旦阿拉伯语方言语音转文本系统。 |
| [^34] | [URA-NER: A Unified Retrieval-Augmented Framework with Retrieval Alignment and Uncertainty Reduction for Low-Resource NER](https://arxiv.org/abs/2609.24372) | 提出统一检索增强框架URA-NER，通过渐进式粒度检索、模型感知表示增强和推理感知知识验证三个关键组件，解决了低资源命名实体识别中检索不对齐与生成不确定性的问题，降低了性能对大语言模型能力的依赖。 |
| [^35] | [Mitigating Entity Type Confusion in Cross-Domain NER via Multidimensional Quantification and Reasoning Enhancement](https://arxiv.org/abs/2609.24357) | 该论文针对跨领域命名实体识别中的实体类型混淆问题，提出了多维混淆量化模型（MCQM）和渐进式双向推理链（PBRC），通过从源-目标层次、语义相似性和显式数据评估三个维度量化混淆程度并增强推理能力来缓解该问题。 |
| [^36] | [Morpho-VITS: Variational Inference with Morphological Modeling for End-to-End Speech Synthesis of a Tonal Bantu Language](https://arxiv.org/abs/2609.24310) | 该论文提出Morpho-VITS模型，通过将VITS架构中的标准音素编码器替换为词素序列编码器和音素-词素注意力网络，利用显式形态学建模来解决班图声调语言（如卢旺达语）文本转语音中声调难以预测的问题。 |
| [^37] | [SupportCal: Label-Free Calibration of Post-Trained LLMs via Reference Support and Corroboration](https://arxiv.org/abs/2609.24303) | 该论文提出SupportCal方法，发现适度引入PLM参考与PoLM的不一致样本（而非完全排除）能非单调地改善校准，并通过参考支持与印证机制实现无需标签数据的后训练大语言模型置信度校准。 |
| [^38] | [Structure Before Sampling: Community-Aware Core-Set Selection for Data-Efficient Text-to-Speech](https://arxiv.org/abs/2609.24275) | 该论文提出基于语音搭配图社区结构的核心集选择方法Community Representative，能在固定音频时长预算下选出覆盖更多稀有音素的训练子集，使仅用20%数据训练的TTS模型即可获得高效表现。 |
| [^39] | [Canonical Procedural Actions: An Auditable Annotation Protocol for Tool-Use Agent Traces](https://arxiv.org/abs/2609.24264) | 本文提出了规范程序性动作（CPA）标注协议，为工具使用智能体轨迹提供可审计的程序性动作标注框架，并通过零售案例研究验证了标注者之间高达0.982的锚点-标签重叠度，证明了其结构可重复性。 |
| [^40] | [Taramandal-GPT: Enhancing Astrodynamics Problem-Solving with Knowledge Retrieval and Structured Thinking](https://arxiv.org/abs/2609.24246) | 提出了基于Qwen3-8b并结合检索增强生成（RAG）流水线与回退机制的领域自适应框架Taramandal-GPT，在包含299个问题的航天动力学基准APBench上取得了与最先进开源及闭源模型相当的表现，尤其在需要深度推理的任务中优势明显。 |
| [^41] | [Memory vs. Context? Influential Factors of Factual Recall in Language Models](https://arxiv.org/abs/2609.24238) | 该论文在31个语言模型上复现并扩展了Yu等人(2023)关于记忆与上下文权衡的研究，确认了大模型和高频实体更依赖记忆知识的规律，但发现这一权衡深受模型家族、后训练和问题措辞的影响——仅改变问题措辞即可使模型对记忆知识的依赖变化高达80个百分点。 |
| [^42] | [From Articles to Publishers: Aggregating Language Model Predictions for News Source Reliability Inference](https://arxiv.org/abs/2609.24219) | 该论文提出一个两阶段框架，先利用语言模型评估单篇文章的可靠性，再聚合文章级预测来推断未见过的新闻出版商的整体可靠性，并采用严格的出版商不相交评估协议以保证评估的真实性。 |
| [^43] | [Vimarsha: Faithful ASR Evaluation for Indian Languages with Demographic Diversity, In-the-Wild Audio and Spelling Variations](https://arxiv.org/abs/2609.24199) | Vimarsha是一个覆盖印度全部22种表列语言的100小时ASR评估基准，通过人口多样化实地录音、高难度真实场景音频以及编码多个有效转写的变体格框架，同时纠正了传统基准的乐观与悲观偏差，揭示出真实条件下模型排名的显著变化及地理与人口层面的性能差异。 |
| [^44] | [LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models](https://arxiv.org/abs/2609.24196) | LoopCD通过对比循环语言模型早期迭代与最终细化迭代的logits来干预不确定的“困难”token，无需额外训练且推理开销可忽略，即可有效提升模型的推理性能。 |
| [^45] | [When Residualization Helps an Audit: Format Effects, Slice Gains, and Their Limits](https://arxiv.org/abs/2609.24194) | 该论文表明，残差化虽能将奖励模型的格式效应削弱约0.12，但无法区分被移除的表面成分中是否含有与测量构念相关的信号，因此仅靠残差化并不能使评估测量更加有效。 |
| [^46] | [Efficient LLM Distillation for Bangladesh Legal Context: A Smartphone-Compatible Retrieval-Augmented Generation Model](https://arxiv.org/abs/2609.24177) | 该研究通过两阶段渐进式知识蒸馏（监督微调加稀疏KL散度最小化）与QLoRA技术，将90亿参数的Gemma-2教师模型压缩为20亿参数学生模型，构建出可在智能手机上离线运行的孟加拉国法律检索增强生成模型，以弥合当地法律信息获取鸿沟。 |
| [^47] | [TAC-Time: Texts as Channels For Multimodal Time Series Forecasting](https://arxiv.org/abs/2609.24156) | TAC-Time提出将文本信息转化为额外的时间通道，与数值序列在共享的时间主干中联合建模，在保留时间连续性和周期结构的同时，实现了高效、可扩展且可解释的多模态时间序列预测。 |
| [^48] | [Data Agents: Agentic Data Systems](https://arxiv.org/abs/2609.24137) | 该论文提出“数据智能体”这一新范式，通过语义数据组织、智能体化流水线编排、记忆管理等六大组件，实现最少人工干预下自主管理、处理和分析数据，从人工设计、字面操作和被动处理三大方面变革传统数据系统。 |
| [^49] | [Re:CAP - Auditing Retrieval Coverage in Production RAG Pipelines](https://arxiv.org/abs/2609.24122) | Re:CAP提出了一种无参考的迭代探测审计方法，通过为可能缺失的主题生成探测性问题并利用LLM评判筛选，来发现生产级RAG系统中检索遗漏的文档，从而审计检索覆盖率，在四个基准上恢复了BM25 top-500无法召回的9-29%金标准标注。 |
| [^50] | [You Can Tell Who's Asking: What the Web's Questions Are Made Of, and Where They Come From](https://arxiv.org/abs/2609.24106) | 该研究通过分析110个FineWeb快照中的134亿次问题出现记录，发现网络问题的来源可以从问题形式中识别出来，且最高频的网络问题大多是模板化内容，因此问题出现次数衡量的是发布频率而非真实用户需求。 |
| [^51] | [From Content Generation to Learning Support: Pedagogy-Guided Generative Video Tutors for STEM Learning](https://arxiv.org/abs/2609.24083) | 提出了PIVOT框架，将教学法原则融入生成式视频辅导的完整流程——从分镜生成、经教学法验证的多模态视频生成，到评估与误解感知的补救教学——实现生成式AI从内容生成向以学习为中心的STEM教学支持的转变。 |
| [^52] | [Efficient Reasoning Exploration via State-Conditioned Latent Steering with Progress Guidance](https://arxiv.org/abs/2609.24066) | 提出了一种无需训练的潜在引导框架SPS，通过构建包含进展引导向量的状态条件化方向库，在推理时引导模型探索能取得有意义进展的多样化推理路径，从而缓解探索坍缩问题并提升Best-of-N推理的探索效率。 |
| [^53] | [Representation-guided in-context learning for medical image interpretation with multimodal large language models](https://arxiv.org/abs/2609.24057) | 该论文提出了无需训练的表示引导上下文学习框架（RG-ICL），利用冻结编码器检索与查询对齐的示例来增强多模态大语言模型的医学图像解读能力，在八个医学数据集上显著提升分类和视觉问答性能，并发现少量与查询对齐的病例比大量随机病例更有效。 |
| [^54] | [Calibrated Decisions at Scale: Converting Police Crash Narratives into Probabilistic Crash Variables with a System One Model (Jev)](https://arxiv.org/abs/2609.24052) | 本文提出Jev——一个不生成文本、直接输出校准概率的“系统一”模型，将近50万条警方事故叙述的大规模编码转化为门控类型化决策，以低成本实现可验证、经人类盲评审计的高精度（F1=0.908）事故变量提取。 |
| [^55] | [When Evidence Conflicts: Reliability-aware Meta-review Generation](https://arxiv.org/abs/2609.24028) | 该论文提出一种可靠性感知的元评审生成框架，通过抽取方面级观点、识别冲突证据，并结合观点支持度与评审质量评估证据可靠性，对评审反馈进行加权，从而在证据冲突时优先采纳更可信的论点并保留多样化观点。 |
| [^56] | [AURA: Uncertainty-Routed Activation Editing for Acoustic Grounding in Speech Foundation Models](https://arxiv.org/abs/2609.23979) | AURA是一种超高效的激活编辑方法，它冻结预训练语音基础模型，利用交叉注意力不确定性特征动态路由稀疏的缩放平移编辑，将非语音音频上的幻觉率从89.18%降至1.94%，并提升了模型在不完善标签和不流畅语音上的声学接地能力。 |
| [^57] | [From Tables to Quantified Statements: Evaluating LLM Inference Generation through Executable Verification](https://arxiv.org/abs/2609.23966) | 该论文提出STAT-TO-TEXT任务，通过执行大语言模型生成的Python检查代码来验证模型从统计表格生成的量化陈述，发现更大的模型（GPT-OSS-120B）能生成最忠实的推理，且不牺牲表格覆盖率和量词多样性。 |
| [^58] | [Open-Jev Judgments on CallScreenBench: Calibrated One-Pass Scam Screening with a Small Language Model](https://arxiv.org/abs/2609.23959) | 该论文提出JevLite方法，通过对Qwen3-4B进行LoRA微调并从标签logit的softmax中直接读取校准的诈骗概率，在CallScreenBench上实现了单次前向传播的诈骗电话筛查，性能不劣于大型LLM裁判，速度提升4.9倍且对合法来电零误报。 |
| [^59] | [Some Dialects Are More Equal Than Others: Non-Prestigious Arabic Dialectal Bias in LLMs](https://arxiv.org/abs/2609.23955) | 该研究通过针对性句法评估和MMLU基准测试发现，多个大语言模型对埃及声望较低的赛伊德方言存在显著偏见，且这种偏见会导致模型处理该方言时性能明显下降。 |
| [^60] | [HaikuS2S: A Cascaded System For Responding In Verse](https://arxiv.org/abs/2609.23951) | 提出了HaikuS2S级联系统，结合ASR、LLM生成俳句以及在诗歌与俳句数据集上微调的TTS，显著改善了俳句语音的韵律与声调对齐，同时保持了情感相似度。 |
| [^61] | [XYEval: Agents say yes to bad advice](https://arxiv.org/abs/2609.23939) | 提出XYEval元评估框架，可将现有基准转化为XY问题评估，发现智能体面对用户看似合理但误导性的建议时性能大幅下降（最高达46.7%），暴露出当前智能体难以识别用户真实问题并抵御糟糕建议的缺陷。 |
| [^62] | [Measuring the Assistant's Harmlessness Preferences on the User Turn](https://arxiv.org/abs/2609.23935) | 研究发现后训练赋予助手模型的无害性偏好会泛化到用户轮次的预测中，表明后训练不仅塑造了浅层的助手角色，而是深度改变了模型本身。 |
| [^63] | [Time-Incremental Continued Pretraining of LLMs: Knowledge Updates Without Catastrophic Forgetting](https://arxiv.org/abs/2609.23916) | 该研究在真实的时间增量场景下对六个开源大语言模型进行持续预训练，发现模型能有效获取新知识且不会发生灾难性遗忘，多数模型甚至还能提升对截止日期前旧知识的回忆能力。 |
| [^64] | [this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent](https://arxiv.org/abs/2609.23886) | 该论文提出2B参数的类型化决策模型this-that-model-1.0，它直接从指定位置的隐藏状态读取受限于调用者声明选项集的答案，不生成任何文本，在笔记本GPU上仅用30.9毫秒、零输出token即可完成决策，比前沿API调用（8758毫秒）快数百倍且成本近乎为零。 |
| [^65] | [Q-TIE: A Lightweight and Generalizable Re-ranking Framework for Temporal Information Retrieval](https://arxiv.org/abs/2609.23880) | 提出Q-TIE——一种基于学习的时序意图提取的轻量级、泛化性强的重排序框架，融合时序检索器与时序重排序器的互补优势，实现鲁棒的时序信息检索。 |
| [^66] | [From UNDRR Reports to Event Records: Schema-Constrained LLM Extraction of Georeferenced Disasters](https://arxiv.org/abs/2609.23853) | 该论文提出一种基于固定模式和受控词汇表的大语言模型流水线，可从UNDRR的PreventionWeb文档中自动提取带地理参照的灾害事件记录并保留审核证据，GPT-5的属性F₁达86.0%，远超传统spaCy基线方法的44.2%。 |
| [^67] | [Federated Multilingual Speech-LLMs: Architecture and Aggregation Strategy Benchmarking](https://arxiv.org/abs/2609.23825) | 该论文对多语言自动语音识别的联邦学习进行了综合基准测试，发现独立调优各组件学习率并采用三组件自适应策略可取得最佳效果，且FedProx的收益依赖于LLM骨干架构——多语言预训练架构在异构数据分布下展现出更强的韧性。 |
| [^68] | [FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model](https://arxiv.org/abs/2609.23808) | 提出FLARE，一种由轻量级生成式奖励模型驱动的全生命周期密集监督范式，通过离线因果诊断框架RADAR提取无后见之明的监督信号，为长程编码智能体提供实时步骤级风险反馈，解决稀疏二值奖励带来的信用分配危机。 |
| [^69] | [Constrained Decoding Eliminates Structural Failures in Small LLMs but Reveals a Scale-Dependent Semantic Gap](https://arxiv.org/abs/2609.23742) | 约束解码能完全消除小型大语言模型在结构化输出中的结构性失败，但内容准确性存在与模型规模相关的语义鸿沟——类型转换错误可被约束解码修复，而指令语义类失败（如多步函数调用）则无法通过约束解码解决。 |
| [^70] | [GRACE: Grounded Adversarial Reasoning over Canadian Law](https://arxiv.org/abs/2609.23726) | 该论文提出了GRACE数据集，包含1,915个基于加拿大联邦立法的问题-推理-答案实例，涵盖对抗性辩护、不确定性和应用推理三种模式，并配套开发了数据构建流程及微调了轻量级法律推理模型CLeAR-4B，填补了加拿大法律在法律NLP基准测试中的空白。 |
| [^71] | [STEVE: Stabilizing Textual Gradient-Based Prompt Optimization via Error-Driven Refinement and Regularized Verification](https://arxiv.org/abs/2609.23716) | 提出STEVE稳定化框架，通过仅从错误样本生成梯度的“错误驱动精炼”和基于保留集防止性能回退的“正则化验证”两种耦合机制，有效稳定文本梯度提示优化过程，减少性能退化并生成更鲁棒的提示。 |
| [^72] | [Financial Language Models as Applied Artificial Intelligence Systems for News-Based Trading under Market Frictions](https://arxiv.org/abs/2609.23703) | 该论文提出了MFAST框架，这是一个市场摩擦感知的情绪到交易框架，能够在考虑交易成本、流动性约束、执行时机等现实市场摩擦的条件下，将带时间戳的金融文本转化为可审计、可复现且市场可行的交易决策。 |
| [^73] | [Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2609.23697) | 提出TrustMOPD方法，以专家模型RL训练前后相对于共享参考模型的位移作为token级可靠性代理，实现无标签的多教师加权蒸馏监督分配，将学生模型性能恢复率从54.4%大幅提升至91.5%。 |
| [^74] | [Beyond Relevance: Structured Semantic Supervision for Product Search with LLM-Augmented Annotations](https://arxiv.org/abs/2609.23646) | 该研究通过LLM生成结构化查询与商品属性并结合人工验证的相关性、解释和中心性标注来增强商品搜索，发现LLM的核心价值在于暴露和近似结构化语义监督信号，而非完全取代人工标注。 |
| [^75] | [Collapse, Not Complexity: Failure-Conditioned Decomposition Repair for End-to-End Document Parsing](https://arxiv.org/abs/2609.23592) | 该论文发现文档解析失败的关键不是页面复杂性而是解析崩溃，提出从普通解析轨迹中检测崩溃、按投影分解页面并重新解析各区域的修复方法，以仅1.13倍的token成本带来1.40 Overall的质量提升。 |
| [^76] | [On the Efficiency-Safety Dilemma in Large Reasoning Models](https://arxiv.org/abs/2609.23587) | 本研究首次全面分析了大型推理模型中效率优化技术与安全性的关系，发现量化、剪枝等方法带来的安全性提升是表面的——源于推理能力退化导致恶意响应“尝试但失败”，而非真正的对齐增强，并指出量化结合剪枝是平衡效率与安全的最佳策略。 |
| [^77] | [Global Ranks Survive, Selected Heads Shift: BOS-Sink Topology under 4-bit Weight-Only Quantization](https://arxiv.org/abs/2609.23585) | 该论文提出Sink拓扑一致性（STC）指标，发现4比特NF4量化下首token注意力头的全局排序高度保持（ρ_s≥0.980），但top-k集合重叠显著下降，且存在层间和跨域的局部失效，表明仅凭全局排序在量化前复用sink头映射并不安全。 |
| [^78] | [ARID: A Deployable Edge AI System for Structured Information Extraction from Industrial Maintenance Work Orders](https://arxiv.org/abs/2609.23582) | ARID是一种可在8GB Jetson Orin NX边缘设备上离线部署的系统，通过双教师过滤、噪声感知数据合成、单次路由决策、4位推理和语法约束解码，将工业维护工单可靠地提取为固定模式的JSON，实现84.8%的token-F1并保证超过99.8%的解析成功率。 |
| [^79] | [Error-Supervised Synthetic Learner Writing for Automated Essay Scoring](https://arxiv.org/abs/2609.23573) | 该研究提出将错误监督引入大语言模型合成作文生成的方法，通过在语法错误检测常用的错误标注文本上微调生成器，使合成作文更贴近语言学习者的真实写作，在较大数据量设置下于12项对比中的11项优于常规合成基线。 |
| [^80] | [VibeMemBench: Evaluating Memory Systems for Coding Agents on Real Repository Coding Tasks](https://arxiv.org/abs/2609.23570) | VibeMemBench 是首个在真实仓库编码任务上评估编码智能体记忆系统的基准，通过 111 个经验证能从注入历史经验中获益的编码目标和可执行测试来衡量记忆系统对下游编码成效的真实改善。 |
| [^81] | [Contributions to the hierarchy of probabilistic languages](https://arxiv.org/abs/2609.23567) | 本文建立了n-gram模型与PCFG生成概率语言的层次关系，证明n-gram概率语言是PCFG概率语言的真子集，并引入全连接PCFG概念，证明n-gram概率语言与全连接PCFG概率语言互不相交。 |
| [^82] | [Paragraph Boundaries Are Not White Space:Compression Depth as the Signature of Hierarchical Structure](https://arxiv.org/abs/2609.23551) | 该研究通过层级旋转位置编码干预实验发现，真实层级结构的标志不是注意力压缩本身，而是压缩的深度——真实段落结构产生更深且随语料库变化的压缩，而密度匹配的随机标签对照仅产生更浅的压缩。 |
| [^83] | [BabelArena: A Large-Scale Multilingual Benchmark for LLM Agents](https://arxiv.org/abs/2609.23490) | 提出了BabelFlow通用工作流和BabelArena大规模多语言基准测试（涵盖23种语言、16,146个实例），首次系统评估了LLM智能体的跨语言能力，发现前沿模型存在显著的跨语言性能差异且无单一模型全面领先。 |
| [^84] | [RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents](https://arxiv.org/abs/2609.23466) | RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。 |
| [^85] | [Propose, Verify, Commit: Evidence-Grounded Memory for Long-Horizon Multi-Actor Conversations](https://arxiv.org/abs/2609.23465) | EGMEMORY将长时程多主体对话记忆建模为可搜索状态机，通过证据锚定的“提议-验证-提交”协议管理记忆的写入与读取，无需记忆专项训练即在GroupMemBench和EverMemBench上分别取得68.2%和77.9%的最优性能。 |
| [^86] | [Perplexity Predicts Protection: Choosing Pretrained Backbones for Worst-Client Fairness in Federated Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2609.23463) | 该研究发现目标文本上的困惑度可以在联邦训练开始前预测哪个预训练骨干网络最能保护数据最少的弱势客户端（秩相关系数达-0.87），为联邦LoRA微调中的骨干网络选择提供了一个简单有效的指标。 |
| [^87] | [Long-Tail Rebalancing for Non-Verbal Vocalization-Aware ASR: A Track~1 System for the NVVSpeech Challenge](https://arxiv.org/abs/2609.23462) | 本系统通过跨数据集标签统一和“平方根类别采样+均匀类别微调”的两阶段采样调度来缓解非言语发声数据的长尾不平衡问题，在NVVSpeech挑战赛Track 1中获得第四名。 |
| [^88] | [PSD: Pseudo Self-Distillation of Memory Representation Capabilities for LLM Agents](https://arxiv.org/abs/2609.23449) | 提出伪自蒸馏框架PSD，使小型语言模型无需访问闭源大模型的logits或隐藏状态，仅通过提示渠道引入黑盒oracle知识，即可实现单模型蒸馏并构建分层记忆表征，从而大幅降低记忆增强智能体的部署成本。 |
| [^89] | [Tool-Augmented On-Policy Distillation for LLM Domain Adaptation in Sequence-Based Omics Tasks](https://arxiv.org/abs/2609.23435) | 该论文提出了首个多组学序列推理基准OmicsBench（包含1160个专家验证问题），发现科学领域LLM虽然在分类准确率上优于通用LLM，但在提供有效生物学证据链的推理能力上反而表现不佳。 |
| [^90] | [MuLA-Bench: A Multilingual Long-Form Audio Understanding Benchmark via Multi-Tier Auditing](https://arxiv.org/abs/2609.23416) | 提出MuLA-Bench多语言长音频理解基准，通过多层次审计构建了覆盖16种语言、1,377.9小时真实录音的5,038个可审计开放式问答数据，并揭示了音频-语言模型在语言、领域和任务维度上的性能差异规律。 |
| [^91] | [One to More, More to One: Category-Aware Iterative Expert Training for Software Engineering Agents](https://arxiv.org/abs/2609.23377) | 针对软件工程智能体强化学习中不同任务类别“此消彼长”的跷跷板问题，本文提出类别感知的专家训练与策略整合框架，通过SWE Labeler多轴证据标注、同源类别专家强化学习与RRE（刷新-修复-扩展）迭代机制，实现成功行为的显式巩固和策略自适应的任务选择，从而均衡提升各任务类别的表现。 |
| [^92] | [Machine-Interpretable Information: Compiling Documents into Searchable and Readable Protocol States](https://arxiv.org/abs/2609.23371) | 该论文提出首个智能体间文档到状态协议 MII，通过双时间尺度写入器将文档编译为 56 个 token 的固定带宽规范状态、再由轻量级翻译器适配任意冻结读取器，在单一可迁移媒介中统一检索、推理与重建，并将查询成本从 O(N²) 降至 O(K)。 |
| [^93] | [LLM-Based FORM Code Generation with Verification-Driven Fine-Tuning](https://arxiv.org/abs/2609.23367) | 该论文首次研究大语言模型为粒子物理符号计算语言FORM生成代码的问题，发现现有前沿模型零样本通过率为零，并提出利用FORM二进制程序作为执行预言机的验证驱动数据生成与微调流水线，构建了经过验证的训练语料库以提升模型生成可执行FORM代码的能力。 |
| [^94] | [Knowing When to Trust Images: Reliability-Aware Multi-modal Entity Alignment](https://arxiv.org/abs/2609.23267) | 提出了一种可靠性感知的多模态实体对齐框架RA-MMEA，通过依赖感知的视觉可靠性预测和稳定性正则化的视觉嵌入生成两个模块，评估图像可靠性并自适应改进不可靠的视觉表示，从而解决图像噪声与语义不对齐导致的融合性能下降问题。 |
| [^95] | [Judging a Review by its Cover: A Reliability Analysis of LLM-based Peer Review Evaluation Metrics](https://arxiv.org/abs/2609.23264) | 该研究提出了一个统计框架，通过比较原始人类审稿意见与保留相同评审内容但改变表达方式的LLM改写版本，检验基于大语言模型的同行评审评估指标是否真正衡量实质性审稿质量，而非仅凭表面的语言形式打分。 |
| [^96] | [CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning](https://arxiv.org/abs/2609.23257) | CTRL框架将语义推理与定量预测解耦，利用LLM智能体作为控制器分析预测误差的分解成分并输出控制信号，再由轻量级残差解码器转化为预测修正，从而提升非平稳环境下时间序列预测的稳定性与可解释性。 |
| [^97] | [SoK: Formal Methods for Fact-Checking and Information Integrity](https://arxiv.org/abs/2609.23239) | 该论文以“担保”概念为核心，按被形式化的对象（声明、推理、系统、生态系统等五个层次）而非流水线阶段来系统化形式化方法在事实核查与信息完整性领域的应用，以满足《数字服务法》和《人工智能法》对可审计证据的监管需求。 |
| [^98] | [ChemCLIR-Bench: Benchmarking Cross-Lingual Information Retrieval in Multilingual Chemical Patents](https://arxiv.org/abs/2609.23231) | 该论文提出了ChemCLIR-Bench，一个基于Google Patents和EPO数据构建的、涵盖五种语言的多语言化学专利跨语言信息检索基准，并通过对八个最先进嵌入模型的系统评估，揭示了单语与跨语言检索之间的显著性能差距。 |
| [^99] | [Euston: Training Away Mathematical Sycophancy Without Losing the Mathematics](https://arxiv.org/abs/2609.23205) | 该论文提出Euston，一个通过GraphSynth生成器构建真假陈述对数据、并用GRPO强化学习微调DeepSeek-R1-8B得到的8B数学论断验证模型，使其学会拒绝证明被篡改的错误定理，将平衡准确率从29.50%提升至63.75%。 |
| [^100] | [Enhancing speech representation learning with cross-modal knowledge transfer with HGNN under low resource settings: the case study of Yemba](https://arxiv.org/abs/2609.23194) | 本文提出基于异构图神经网络的跨模态知识迁移方法，将声学与语言学实体建模为统一图中的不同节点类型，通过消息传递机制让语言学节点向声学节点显式传递知识，从而有效增强低资源语言（如Yemba语）的声学表征学习。 |
| [^101] | [LLMs as Linguistic Chameleons: Decoupling Semantics and Structure for Privacy-Preserving Communication](https://arxiv.org/abs/2609.23193) | 提出CROSS-MAP双向框架，通过语义解耦在推理前将私有输入映射到不同语义域、推理后再恢复输出，在保护隐私的同时不损害LLM的任务效用。 |
| [^102] | [Low resource cross-modal alignment using HGNN to enhance speech representation](https://arxiv.org/abs/2609.23191) | 该论文提出一种基于异构图神经网络和链接预测的数据高效语音-文本对齐方法，通过消息传递将文本信息显式传递给语音模态，从而在低资源和计算受限条件下增强语音表示。 |
| [^103] | [Chronologic: Measuring Language Models' Ability to Represent the Past](https://arxiv.org/abs/2609.23178) | 该论文提出了Chronologic基准，利用1831-1930年的历史文本评估语言模型表征过去的能力，发现生成式任务比判别式任务更难、仅用历史文本预训练的模型在似然评估中领先但在自由生成上不敌商业模型，且没有任何受测模型能可靠地表征过去。 |
| [^104] | [OmniEdu: Open Foundation Models for Learning and Teaching](https://arxiv.org/abs/2609.23088) | OmniEdu是一个面向K-12学习与教学的开放基础模型家族，其核心创新在于围绕学科能力、课程对齐、诊断推理和教学支架四种能力来组织指令微调语料（69,999个样本、1596万token），使模型同时具备解题与教学辅导能力。 |
| [^105] | [Directing large language models to follow the letter or spirit of the law](https://arxiv.org/abs/2609.23083) | 该研究通过定向适配方法使大语言模型能够优先遵循法律的精神或字面意义，并通过模型内部分析揭示了法律概念在低维空间中的可解释几何结构。 |
| [^106] | [Tutoring Large Language Models to be Domain-adaptive, Precise and Safe](https://arxiv.org/abs/2609.23071) | 本论文提出“负责任智能”框架，通过主动学习与图知识减少幻觉、解码时对齐机制实时拦截有害内容、以及语言特定引导保障文化与多语言安全，为构建领域适应、精确且安全的下一代AI提供蓝图。 |
| [^107] | [From Concept Alignment to Causal Grounding: An Intervention Test of Chain-of-Thought Faithfulness](https://arxiv.org/abs/2609.23065) | 该论文将CoT忠实性重新定义为“内部概念锚定”问题，利用共享稀疏自编码器直接比较预测过程与CoT过程的内部概念，并首次引入三个相关性对齐指标和一个因果消融指标Δp，以检验共享概念是否因果性地驱动模型答案。 |
| [^108] | [Bridging Static and Agentic RAG for Taiwanese Historical Question Answering](https://arxiv.org/abs/2609.23056) | 该论文针对台湾历史问答发现智能体式与静态RAG总体性能相近但在70.83%的问题上表现各异，并提出一种比较两个回答及其引用证据的事后选择器，显著优于任一单一流水线并恢复了oracle提升空间的60.34%。 |
| [^109] | [Attributable Post-Rationalization in RAG Citations: A Controlled Reproduction and an RLVR Comparison](https://arxiv.org/abs/2609.23053) | 本研究通过受控复现实验发现，RLVR训练无法减少RAG系统中的事后合理化不忠实引用——RLVR智能体的不忠实引用率与其基础模型相当，奖励正确答案并不能提升引用的忠实度。 |
| [^110] | [Enforcing Narrative Reliability and Epistemic Pacing in LLM-Driven Detective Games via Structured Knowledge Trees](https://arxiv.org/abs/2609.23043) | 该论文提出结构化知识树与三智能体LLM流水线架构，通过分离知识检索、对话生成与响应验证，确保侦探游戏中的虚拟嫌疑人只透露当前叙事状态允许的信息，从而减少幻觉并保持作者对信息披露节奏的控制。 |
| [^111] | [Auditing Political Alignment in LLM Assistants: Engagement, Stance, and User Identity](https://arxiv.org/abs/2609.23039) | 该研究提出“言论体制”新框架，通过对六个主流AI系统共7,500个多轮对话的预注册实验，首次系统揭示了大语言模型的政治行为如何随话题敏感度和用户政治身份而动态变化。 |
| [^112] | [Rethinking Pivot Programming Languages in Code Language Models](https://arxiv.org/abs/2609.22988) | 该研究在控制表示各向异性和长度差异两个混淆因素后发现，代码语言模型中不存在普遍的枢纽编程语言，跨语言迁移的枢纽地位取决于具体的关系类型（代码-代码几何、代码-英语对齐或枢纽检索各有不同偏好）。 |
| [^113] | [Beyond Similarity: Coverage-Aware Prompt Selection for Time Series Forecasting with LLMs](https://arxiv.org/abs/2609.22977) | 提出CASP-LLM框架，通过结合使用跟踪与饱和门控的无参数覆盖正则化器，解决基于相似性检索的提示冗余偏差，使大语言模型时间序列预测能够覆盖罕见但信息丰富的事件。 |
| [^114] | [Automatic multimodal UX improvement recommendations from LLM agent user simulations](https://arxiv.org/abs/2609.22971) | 提出AMUSER多模态框架，利用LLM智能体模拟用户行为并自动生成按优先级排序的网站UX改进建议，效果显著优于纯文本模拟且模拟成本降低89%。 |
| [^115] | [AgentRouter: Heterogeneous Model Routing for Cost-Optimal Multi-Step Agentic Workflows](https://arxiv.org/abs/2609.22951) | 提出AgentRouter轻量级分类器，通过步骤级异构模型路由将多步智能体工作流的推理成本降低72%，且每步仅增加不到5毫秒的开销。 |
| [^116] | [Beyond Single-Model Injection: A Threat Model and Defense Architecture for Prompt Injection in Multi-Agent Systems](https://arxiv.org/abs/2609.22949) | 该论文首次针对多智能体系统构建了涵盖四大类共14种攻击向量的提示注入威胁模型，揭示了智能体间消息传递、共享工具访问和信任传播带来的新型攻击面，并提出了相应的防御架构。 |
| [^117] | [Beyond Linear Context: Graph-Guided Evidence Navigation for Long-Novel Reasoning with a Local 9B Language Model](https://arxiv.org/abs/2609.22939) | 该研究提出利用冻结知识图谱引导证据导航的方法，使本地9B小语言模型在长篇小说问答任务上以53.85%的准确率超越了近期窗口、全书压缩和向量检索等基线方法。 |
| [^118] | [Measuring Behavioural Signatures of Large Language Models through Psychometric Profiling](https://arxiv.org/abs/2609.22934) | 该研究提出跨语言心理测量画像框架，用七种心理量表对九个大语言模型进行中英文重复施测，发现LLM在对齐塑造的共同亲社会倾向之外仍呈现模型特异的结构化行为特征，且未应答（NA）的结构化分布界定了自我报告方法的适用边界。 |
| [^119] | [An Iterative LangGraph Agent for Text-to-SQL: Natural Language Access to the Chicago Crime Database](https://arxiv.org/abs/2609.22917) | 该论文构建了一个仅依靠提示工程（无需微调）的六节点LangGraph迭代式Text-to-SQL智能体，通过问题相关性检查、实时模式获取、SQL生成、试运行验证与失败重试等流程，让非技术用户能用自然语言查询芝加哥犯罪数据库，并将有效SQL率提升至93%、执行准确率提升至60%。 |
| [^120] | [LLMs Anchor on Chief Complaint and Fail to Integrate Evidence in Sequential Clinical Triage](https://arxiv.org/abs/2609.22904) | 该研究提出了评估大语言模型在序贯急诊分诊任务上的新方法学，发现尽管LLM在完整病历上表现接近医生，但在逐轮预测分诊等级时性能显著退化，原因是模型过度锚定于主诉信息而未能整合对话中后续出现的证据。 |
| [^121] | [Block-Sparse Attention with Semantic-Geometric Decoupled Routing](https://arxiv.org/abs/2609.22884) | 提出语义-几何解耦路由框架，通过将语义聚合移至RoPE前空间并利用离线结构先验与相对块距离重建几何偏置，实现了免训练、闭式且精确的块稀疏注意力路由。 |
| [^122] | [Per-Query Gating of LLM Rerankers for Multi-Hop Retrieval](https://arxiv.org/abs/2609.22880) | 提出一种学习式的逐查询门控方法，仅利用LLM调用前可得的统计特征和可执行回退机制，在重排序器无帮助时跳过调用，在三个多跳基准上跳过51%的LLM调用而平均覆盖率仅损失1.2个百分点，从而大幅降低成本与延迟。 |
| [^123] | [To Consolidate or not to Consolidate? Evaluating the Impact of Consolidation in Multi-Reference Training using Peer Reviews](https://arxiv.org/abs/2609.22805) | 该研究证明了对于自动化同行评审生成等中间熵NLG任务，将多样化的参考整合为统一训练信号比传统单参考或多参考训练范式更有效，并发布了包含超过36,000篇论文及原始与整合评审的MERC-36K语料库加以验证。 |
| [^124] | [AlexandriaX 2026: The First Shared Task on Dialectal Arabic Machine Translation](https://arxiv.org/abs/2609.22796) | 该论文介绍了AlexandriaX 2026共享任务，这是首个方言阿拉伯语机器翻译共享任务，通过上下文感知对话翻译、金融领域跨方言翻译和跨度级错误检测分类三个互补子任务，系统性地应对了方言阿拉伯语翻译中建模方言变异、会话上下文和社会语言学得体性的挑战。 |
| [^125] | [Diagnose, Then Repair: A Two-Stage MQM-Guided Post-Editing Framework for Domain-Specific Machine Translation](https://arxiv.org/abs/2609.22793) | 提出了一种两阶段MQM引导的自动后编辑框架，先由检索增强的LLM评估器生成片段级错误诊断，再由独立的后编辑器执行最小化针对性修复，从而在多语言特定领域机器翻译中显著提升翻译质量并增强可控性。 |
| [^126] | [MIS-Bench: Benchmarking Multimodal LLMs for Psychotherapeutic Interpersonal Skills Assessment](https://arxiv.org/abs/2609.22778) | 该论文提出了首个用于心理治疗人际关系技能评估的多模态基准MIS-Bench，揭示现有多模态大语言模型与人类专家评估一致性有限，并提出回归感知微调方法MIS-RAFT以实现精细化的技能评分。 |
| [^127] | [NLPCC 2026 Task 10: Citation-Level Faithfulness Verification with DeBERTa Ensembles and Class-Wise Calibration](https://arxiv.org/abs/2609.22774) | 该论文提出了一种融合DeBERTa-large文档级分类器、段落感知交叉编码器集成、类别级决策校准以及BM25证据融合的完全离线系统，在NLPCC 2026任务10赛道2的引用级忠实性验证中以82.99的总分获得第二名。 |
| [^128] | [Beyond Final-Token Classification: Heterogeneous Readouts for Evidence-Grounded Suicide Risk Detection](https://arxiv.org/abs/2609.22767) | 该论文提出异构读出分解（HRD）方法，将语义验证与输出实现分离，针对序数风险分类、多标签因素检测和证据短语抽取三类任务分别设计异构读出机制，在 IEEE BigData Cup 自杀风险检测基准上显著提升了分类与证据抽取性能。 |
| [^129] | [Clinical Domain Classification from Medical Transcriptions](https://arxiv.org/abs/2609.22734) | 该论文系统比较了六种传统机器学习分类器、两种预训练Transformer模型（BERT和XLNet）以及少样本大语言模型提示方法在医学转录文本临床领域分类任务上的表现，并针对数据中严重的类别不平衡问题提出应对方案。 |
| [^130] | [Analyzing Public Discourse on Urbanism: Topic Clustering, Sentiment Analysis and Retrieval-Augmented Generation using YouTube Comments](https://arxiv.org/abs/2609.22705) | 该研究构建了一个融合地理实体消解、主题建模、情感分析与检索增强生成的对话系统，用于分析覆盖309个北美城市的YouTube评论中的城市化议题讨论，并通过实验揭示了标准NLP组件在处理简短、非正式、地理模糊文本时的性能局限。 |
| [^131] | [LLaDA-PRM: A Bidirectional Step-Level Reasoning Evaluator](https://arxiv.org/abs/2609.22700) | 该论文发现双向注意力比因果注意力更适合步骤级推理评估，并据此构建了8B参数的双向评估器LLaDA-PRM，在多个基准上以更小的参数量显著超越更大的自回归模型。 |
| [^132] | [COT-TTS: Audio Context-Aware Text-to-Speech with Chain-of-Thought Reasoning](https://arxiv.org/abs/2609.22697) | 提出了COT-TTS任务，通过思维链推理从历史对话音频中自然推断说话风格并合成指定音色的语音，同时构建了包含900万样本的大规模双语对话语音数据集和人工验证基准来支持该任务。 |
| [^133] | [Beetle: A Bilingual Model Suite for Modelling Second-Language Processing](https://arxiv.org/abs/2609.22633) | 该论文提出了Beetle——一个分词器、目标语言、训练预算和暴露结构均可独立操控的受控双语模型预训练框架，并发布了330个开源模型，用于系统研究训练条件如何影响第二语言加工。 |
| [^134] | [Pretrained Persona Mixture Models and Tandem Models for Human Simulation](https://arxiv.org/abs/2609.22607) | 该论文提出“人格混合模型”，即使用预训练基础模型并借助特定人物的简短对话样本实现人格绑定，能比指令微调模型更准确地模拟人类，并保留更多人类对话的自然多样性。 |
| [^135] | [Preserving What Matters: Semantic Scaffolds Beyond Saturation in Summarization Evaluation](https://arxiv.org/abs/2609.22603) | 针对ROUGE仅衡量表面重叠、LLM评分饱和而无法区分模型的问题，本文提出Semantic Scaffold评估框架，通过从源文本提取事实、问题和实体属性的层次化结构作为固定评分参考，并设计FPS、QPS、EPS三个诊断指标来有效评估摘要对关键信息的保留程度。 |
| [^136] | [Do Student LLMs Inherit OOD Robustness? Invariance-Weighted Distillation for Reliable Knowledge Transfer](https://arxiv.org/abs/2609.22566) | 提出不变性加权蒸馏（IWD）框架，通过从多个合成环境中的预测不变性估计教师模型对因果特征的依赖程度，进而对训练样本动态加权，解决知识蒸馏中学生模型在分布外场景下性能退化的问题。 |
| [^137] | [Correct Diagnosis, Better Feedback: A Symbolic-Verifier for Faithful LLM Tutoring Feedback in Logic Proofs](https://arxiv.org/abs/2609.22553) | 该论文提出一种基于符号验证器的架构，将学生错误诊断与语言生成分离，实验表明符号验证器的诊断准确性远优于LLM检测器，而错误的诊断会通过理由生成被忠实地传播到最终反馈中。 |
| [^138] | [Cross-Dialect NER for Bangla Regional Dialects Using Leave-One-Dialect-Out Cross-Validation and Explainable AI](https://arxiv.org/abs/2609.22536) | 本文提出基于ANCHOLIK-NER数据集的孟加拉语跨方言命名实体识别框架，采用留一方言交叉验证策略评估八种预训练Transformer模型在未见方言上的泛化能力，并结合可解释AI方法分析模型表现。 |
| [^139] | [When Cosine Similarity Fails to Reflect Linearly Accessible Structure in Dialogue Models](https://arxiv.org/abs/2609.22522) | 该论文发现在对话微调的大语言模型中，余弦相似度会严重低估隐藏状态中线性可解码的人格结构（线性探针AUC为0.73-0.97，而余弦kNN仅为0.56-0.77），这种失配仅出现在对话场景而非单句分类任务中，且需要有监督的低维子空间才能恢复该结构。 |
| [^140] | [CultureMINE: Datasets and Methods for Improving the Cultural Capabilities of NLP Systems](https://arxiv.org/abs/2609.22494) | 该论文通过分析375多篇文化NLP领域的论文，系统梳理了NLP系统所针对的文化能力、文化数据资源的创建方式以及提升文化能力的方法，并发布了可交互的论文列表平台以促进该领域未来研究。 |
| [^141] | [Replication Without Persistence in Hosted LLMs: Measurement Sensitivity in Action-Time Belief Evaluation](https://arxiv.org/abs/2609.22478) | 该研究在Regent Chess环境中将托管LLM行为评估中的复现性、测量敏感性与持久性三个验证问题区分开来，发现先前报告的Gemini 3.1 Flash-Lite缺陷虽能在新数据上复现，但其存在依赖于评估与推理配置的重建方式，揭示了评估结果对测量配置的敏感性。 |
| [^142] | [Efficient Mixture-of-Experts with Speculative Decoding via Expert Coactivation](https://arxiv.org/abs/2609.22471) | 该论文发现训练时采用高专家共激活度的MoE路由器设计，可以显著加速结合推测解码的MoE推理，有效缓解更多验证token带来的内存传输开销。 |
| [^143] | [Toward Personalized Sleep Guidance from Wearable Data Using Language Models](https://arxiv.org/abs/2609.22463) | 该论文提出一个两阶段框架：先利用多智能体LLM流水线从无标注的可穿戴数据中推理生成结构化睡眠指导以构建数据集，再通过监督微调将推理轨迹蒸馏到小语言模型并结合免训练的Best-of-N选择策略，实现了可本地部署且优于商业大模型的个性化睡眠指导。 |
| [^144] | [Apollo Restore: A Foundation LLM for Historical Greek Optimized for Fill-in-the-Middle Restoration of Ancient Greek Texts](https://arxiv.org/abs/2609.22455) | Apollo Restore 是首个面向历史希腊语（乃至任何古代地中海语言）的 240 亿参数基础大语言模型，通过“中间填空”目标微调，能够在不知缺失文本长度的情况下修复残缺古希腊文本，并在长度平衡的评估指标下大幅超越已发表的最强模型。 |
| [^145] | [Vox-Infinity: Benchmarking the Limits of Long-Context Spoken Language Models](https://arxiv.org/abs/2609.22452) | Vox-Infinity 是首个专门评估口语语言模型长上下文理解能力的基准，它从轮次数量和轮次时长两个维度系统地扩展音频历史，并通过答案溯源标注和按所需上下文长度组织样本，实现对长语音上下文理解极限的精确评估。 |
| [^146] | [Contextual Causality with Large Language Models: A Survey](https://arxiv.org/abs/2609.22409) | 本综述首次提出了大语言模型情境因果性的系统分类体系（涵盖语义、干预与反事实三类因果性），分析了现有研究的关键局限，并指出了当前基准与真实需求之间的差距及未来研究方向。 |
| [^147] | [Initial Evaluation of Potential Bias in Coverage of Humans in Wikidata](https://arxiv.org/abs/2609.22375) | 本文开发了一个开源审计平台，对Wikidata中超过600万个人类条目在性别、地理、族裔、职业等多维度上的人口代表性偏差进行了系统性评估，发现声明性别的条目中女性仅占28.71%。 |
| [^148] | [Functional Emotion Without Character: Large Language Models, Aristotelian Disposition, and the Limits of Behavioral Alignment](https://arxiv.org/abs/2609.22362) | 本文提出一种结构性替代框架，将情感建模为高维表征状态空间中的动态模式，并论证大语言模型虽具有因果活跃的情感概念表征，但这既不等于主观感受，也不足以确立完整的情感能动性。 |
| [^149] | [Used, Mentioned, or Condemned? A Controlled Contrast-Set Diagnostic for the Use-Mention Distinction in Code-Mixed Hinglish Misogyny Detection](https://arxiv.org/abs/2609.22261) | 该论文诊断了印地英语厌女症检测中的两个评估伪影问题，并发布了Hinglish-MGY-Diag——首个通过最小对立对区分侮辱性词汇是“被使用”还是“被提及”的受控对比集诊断工具。 |
| [^150] | [Causal Localization of the Refusal Direction in Audio Language Models](https://arxiv.org/abs/2609.22260) | 通过因果干预实验发现，音频语言模型对有害语音请求的拒绝行为主要由底层文本语言模型的中后层承载，而非语音前端，表明拒绝能力是从安全对齐的文本模型继承而来的。 |
| [^151] | [Which Part of the Context Layer Does the Work? Separating Semantic Content from Retrieval Scaffolding in Text-to-SQL Agents](https://arxiv.org/abs/2609.22259) | 该论文通过四臂消融实验证明，text-to-SQL智能体中上下文层带来的准确率提升主要由其中的语义内容（数据契约）贡献，而非检索脚手架或预计算视图。 |
| [^152] | [Strategy Accumulation and Guided Execution for Automated LLM Fine-Tuning](https://arxiv.org/abs/2609.22257) | 本文提出SAGE两阶段框架，通过多智能体蒙特卡洛树搜索探索与经验蒸馏构建可累积的结构化经验库，使自动化LLM微调系统能够复用历史搜索经验，避免每个新任务都从冷启动开始重复昂贵搜索。 |
| [^153] | [DIPLOMAT: Dialogue-Span-Aware Direct Preference Optimization for Polite Persuasive Workplace Negotiation Dialogues](https://arxiv.org/abs/2609.22256) | 该论文提出了DIPLOMAT对话系统，通过对话跨度感知的直接偏好优化方法进行训练，并构建了由多智能体框架生成、标注了谈判策略、礼貌程度和说服策略的PROWESS多轮职场谈判对话数据集，使AI能够进行礼貌且有说服力的职场谈判。 |
| [^154] | [Deep Persona: A Psychologically Grounded Architecture and Evaluation Framework for Role-Playing Agents and Simulations](https://arxiv.org/abs/2609.22255) | 该论文提出Deep Persona，一种基于心理学的三层人格架构（可观察表达、潜在信念、核心动机驱动），结合脚本决定论与有界能动性原则，并配套无需参考的评估框架，从而提升LLM角色扮演智能体在长交互中的行为连贯性与逼真度。 |
| [^155] | [CAMFT: Conflict-Aware Mergeable Fine-Tuning for Large Language Models](https://arxiv.org/abs/2609.22253) | CAMFT提出了一种冲突感知的可合并微调方法，通过在微调阶段就引导各任务更新跨任务冲突较低的稀疏坐标，使模型在训练过程中即具备可合并性，从而在下游模型合并中取得更优性能。 |
| [^156] | [A Tutorial on Prompt Engineering: From Messy Thoughts to AI Workflows](https://arxiv.org/abs/2609.22249) | 本文提出了一套系统化的提示工程可复用设计方法——通过定义工作、精简上下文、角色设定、肯定性质量目标、结构化批评与验证以及智能体运行循环，将随意的提示转化为规范化的AI工作流设计。 |
| [^157] | [Checkpoints Are Not Enough: Trust Calibration in CoSLR, a Human-AI System for Systematic Literature Reviews](https://arxiv.org/abs/2609.22248) | 该论文提出了CoSLR——一个结合大型语言模型与检索增强生成的多智能体人机协作系统，通过在系统性文献综述流程中设置强制性人工检查点来校准用户对AI生成内容的信任，防止未经核实的AI综合内容以系统性综述的可信度进入学术记录。 |
| [^158] | [The Corroboration Illusion: When More News Makes LLM Forecasts Less True](https://arxiv.org/abs/2609.22246) | 该论文首次形式化了“新闻语料库投毒”这一新型威胁：攻击者仅需发布少量AI生成的新闻文章，无需接触模型、检索器或用户查询，即可大幅操纵基于新闻检索的LLM事件预测概率。 |
| [^159] | [Do Chess Explanations Reflect Model Decisions? Behavioral and Token-Level Tests of LLM Reasoning Faithfulness](https://arxiv.org/abs/2609.22245) | 该研究在200个国际象棋残局谜题上，通过走法可恢复性、解码器侧控制和词元级评分三种方法检验LLM解释的忠实性，发现流畅合理的解释并未忠实反映模型的真实决策——解释带来的增益微小且依赖解码器，甚至无关的解释文本还会降低正确走法的概率。 |
| [^160] | [Replay-Gated Neural Execution: Decoupling Persistent Behavioral Specifications from Neural Realizations in Frozen Language Models](https://arxiv.org/abs/2609.22243) | 该论文提出“重放门控神经执行”框架，将持久行为规范与具体的神经实现解耦为五个独立对象，通过冻结模型的隔离FP32/BF16重放和运行审计来认证候选动作，实验揭示了行为谓词、见证、查找器等对象各自独特的失败模式。 |
| [^161] | [H2LooP Telecom Model v1: From Telecom Comprehension to Autonomous Issue and PR Resolution](https://arxiv.org/abs/2609.22241) | H2LooP Telecom Model v1 是专为电信行业微调的 31B 参数领域大语言模型，其理解变体在 OT-Lite 基准上达到 81.8% 并在 Open Telco AI 排行榜上以更小参数量超越 GPT-5 和 Claude Opus 等前沿闭源模型，其智能体变体则可自主完成电信代码生成、PR 解决与代码提交。 |
| [^162] | [Knowledge Graph-Augmented Ambient AI for Clinical Note Generation](https://arxiv.org/abs/2609.22239) | 该论文提出了模型无关的覆盖导向修订（CDR）框架，通过从医患对话记录构建知识图谱来识别自动生成临床笔记中缺失的关键医学概念，并引导大语言模型恢复这些缺失信息，且无需修改原有的笔记生成系统。 |
| [^163] | [BizSage: A Self-Evolving Multi-Agent Framework for Business Research with Efficient Knowledge Retrieval](https://arxiv.org/abs/2609.22235) | BizSage是一个面向经济学与商业研究的多智能体框架，通过合并章节级知识图谱构建横向知识图谱（LKG）实现语料库级细粒度检索，并结合质量驱动的自我进化机制来提升检索精度与实证严谨性。 |
| [^164] | [Seeing Through Conflicts: Improving Instruction Hierarchy Alignment in Vision-Language Models](https://arxiv.org/abs/2609.22234) | 该论文将多模态指令层级对齐视为推理问题，通过基于规则奖励的强化学习训练视觉语言模型，发现混合模态（文本+图像）监督训练效果最佳，能显著提升模型抵御跨模态指令冲突攻击的鲁棒性，且可泛化至真实图像和网络场景。 |
| [^165] | [EvalMem: An Operation-Level Diagnostic Framework for Long-Term Memory Systems](https://arxiv.org/abs/2609.22231) | EvalMem通过编码、检索、生成三个并行检查器将长期记忆系统的错误精确定位到具体操作环节，并借助召回优先的智能体RAG策略将证据召回率从70.2%提升至95.6%。 |
| [^166] | [Assessing Adversarial Robustness of Latent Reasoning Models](https://arxiv.org/abs/2609.22228) | 本研究系统评估了潜在推理模型在文本和多模态设置下的对抗鲁棒性，发现其整体上比显式思维链基线更脆弱，尤其在白盒攻击下性能下降严重。 |
| [^167] | [Guiding the coarse levels of semantic IDs makes the fine levels learnable](https://arxiv.org/abs/2609.22227) | 提出Guided SID方法，通过确定性的监督索引分配强制RQ-VAE的粗粒度层级编码基于文本且与任务相关的预定义类别属性，使语义ID最重要的层级在构造上即具备可理解性和任务相关性，从而让细粒度层级变得可学习。 |
| [^168] | [Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time](https://arxiv.org/abs/2609.22226) | 该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。 |
| [^169] | [Do LLMs Choose Like Humans? Using Cognitive Theory to Evaluate LLM Decision-Making](https://arxiv.org/abs/2609.22225) | 该研究构建了包含14万次试验的产品选择基准，发现大语言模型虽能表现出类人的选择和问题分类变化，但无法像人类那样在价格与质量等特征间重新分配注意力，且模型规模和思维链推理均无法弥补这一差距，表明LLM的决策机制与人类存在本质区别。 |
| [^170] | [From Trait Vectors to Circuits: Tracing Refusal and Sycophancy Through Language Models](https://arxiv.org/abs/2609.22224) | 该研究发现Qwen2.5-7B-Instruct中的拒答特质向量确实位于模型真实使用的计算通路上——围绕该向量构建的紧凑电路能够忠实重现并恢复被消融的拒答行为，且所需边数仅为直接输入-输出电路的一半，表明引导向量可以对应模型内部真实的电路而非仅仅是外部扰动。 |
| [^171] | [EAVer: Long-Form Factuality Verification as an End-to-End Agentic Policy](https://arxiv.org/abs/2609.22223) | EAVer将长文本事实性验证建模为端到端的统一智能体策略，通过语义声明分组、基于置信度的搜索路由和上下文证据跨声明复用，显著减少了冗余的LLM与搜索调用。 |
| [^172] | [Can Coding Agents Reproduce Official Statistics? Metadata, Retry Budget and the Limits of Execution Feedback in a Controlled Eurostat Benchmark](https://arxiv.org/abs/2609.22222) | 本研究构建了一个包含30个任务、四种对照条件的受控Eurostat基准，通过360次任务运行分离出权威元数据、重试预算与执行反馈对编码智能体复现官方统计数据准确性的各自贡献，并揭示了执行反馈在其中的局限性。 |
| [^173] | [Team DArgk at the 2026 ELOQUENT lab for evaluating generative language model quality: Residuals of Humanity: AI Detection Evasion via GRPO Fine-Tuning](https://arxiv.org/abs/2609.22221) | 本文提出SHADE强化学习框架，通过GRPO全量微调LLaMA模型成功规避AI生成文本检测器，实现了98.5%的检测规避率（基础模型仅为1.5%），揭示了AI文本检测器在对抗性生成下的脆弱性。 |
| [^174] | [Knowing, and Saying It Only When Asked: LLM Endognostics and the Schizognosis of Minerva-7B](https://arxiv.org/abs/2609.22219) | 该论文提出“LLM 内诊断学”白盒审计框架，发现 Minerva-7B 在行为层面无法区分大多数风险提示对（63.7% 表现相同）且常顺从错误前提，但其残差流内部实际保持着显著的风险区分与真实事实表征，揭示了模型“内在知晓却不外显表达”的分裂性认知现象。 |
| [^175] | [Toollery: Scaling LLM Agents to Thousands of Skills and Tools](https://arxiv.org/abs/2609.22218) | Toollery是一个无需训练的候选压缩框架，通过从技能/工具规范生成用户意图查询并构建检索索引，将真实用户请求映射到紧凑候选集，从而实现LLM智能体对数千种技能和工具的高效可扩展选择。 |
| [^176] | [The Effect of Quantization on Clinical Benchmarks: Accuracy and Safety Across Model Families](https://arxiv.org/abs/2609.22216) | 该研究系统评估了量化对临床大语言模型准确性与安全性的影响，发现INT8量化普遍安全，而INT4量化退化显著且因模型而异，且临床微调并不能赋予模型压缩鲁棒性。 |
| [^177] | [On Mitigation of Subliminal Learning in Large Language Models](https://arxiv.org/abs/2609.22215) | 该论文发现大语言模型中的阈下学习在微调过程中呈现高度非单调的动态特性，并提出了一种退火KL正则化的“阈限训练”方法，通过约束早期相对基础模型的漂移来有效缓解知识蒸馏中非预期行为特征的隐蔽传递。 |
| [^178] | [The Bairong System for MLC-SLM 2026: Dynamic Question-Aware Evidence Routing for Multilingual Conversational Speech Understanding](https://arxiv.org/abs/2609.22214) | 百融系统提出动态问题感知证据路由器，根据问题和答案选项智能选择完整转录上下文、局部音文融合、说话人关联证据或全局声学样本等不同证据类型，在MLC-SLM 2026多语言对话语音理解挑战赛任务1中取得25.70%和18.44%的tcpMER成绩。 |
| [^179] | [SCoP: Structured Constraint Parsing for Evidence-Space Control in Temporal Knowledge Graph Question Answering](https://arxiv.org/abs/2609.22213) | 该论文提出SCoP框架，通过结构化约束解析将时序决策从答案推理中外置化，把时序意图转化为可执行约束来控制证据空间，从而避免无效事实进入答案推理过程。 |
| [^180] | [A Channel-Boosted Multi-Agent System with Iterative Consultation for Document Sensitivity Classification](https://arxiv.org/abs/2609.22212) | 提出通道增强多智能体系统CB-MAS（实现为IC-MAS），通过通道评论智能体学习文档自适应信任权重、成对咨询智能体迭代交换信念状态，在不引入长上下文计算成本的情况下克服了transformer固定输入截断丢失文档尾部敏感证据的问题，显著提升文档敏感性分类性能。 |
| [^181] | [SALSA: Semi-Autonomous Literature Summarization Assistant](https://arxiv.org/abs/2609.22210) | SALSA是一个开源的人机协作平台，融合大语言模型、OCR和计算机视觉等技术，从多模态文献中半自动化地提取结构化科学数据集，并支持用户校正验证以保障数据质量。 |
| [^182] | [Schematize: An Agentic System for Generating and Refining Information-Extraction Schemas for Legal Research](https://arxiv.org/abs/2609.22209) | Schematize是一个开源多智能体系统，通过澄清对话、迭代模式生成、基于数据的优化和聊天式事后编辑，将法律研究者的研究问题交互式地转化为经过验证的信息抽取模式，在大多数测试配置中达到最佳性能。 |
| [^183] | [Replicating the Geometry of Emotion Representations in a Base Open-Weights Model](https://arxiv.org/abs/2609.22208) | 该研究在开源基础模型gemma-2-27b上成功复制了Claude Sonnet 4.5中情感表征的几何结构，证明情感概念以反映人类情感心理学的向量几何形式表示这一现象并非专有模型特有，而是基础预训练模型的普遍特性。 |
| [^184] | [Dissecting Training-Free Uncertainty Estimation in Multimodal Large Language Models](https://arxiv.org/abs/2609.22206) | 本文系统研究了多模态大语言模型的免训练不确定性量化方法，将其分为token级、言语化和语义三大类，并通过大规模基准测试发现没有单一方法在所有场景下占优，不同方法在不同答案长度上各有优势。 |
| [^185] | [Evaluating Personal Information Output from Conversational Interactions in Generative AI Systems](https://arxiv.org/abs/2609.22204) | 本研究通过对15名日本参与者的试点评估发现，生成式AI对话中个人信息输出受模型设计差异影响有限，事实类输出比推断类更保守，核心身份属性处理较为谨慎，而行为、语言、心理认知及整体画像等属性更容易被高准确度地输出或推断。 |
| [^186] | [PII-TRACE: A Benchmark for Context-Aware PII Detection in Multi-Turn LLM Conversations](https://arxiv.org/abs/2609.22200) | 该论文提出了首个面向多轮LLM对话的上下文感知PII检测基准PII-TRACE，实验表明包括前沿LLM在内的现有检测器均无法在避免大量误报的同时实现对重复标识符的完整跨轮次实体级覆盖。 |
| [^187] | [The Role of AI in Online Reviews](https://arxiv.org/abs/2609.22198) | 该论文提出一种利用离散LLM供给冲击并对比已验证与未验证评论的实证识别方法，发现生成式AI供给改进后，Trustpilot上超过1300万条评论中的未验证评论显著趋向更负面（1星增多、5星减少、评分下降）。 |
| [^188] | [EvoRank: LLM-Guided Evolution of Multi-Objective Learning-to-Rank Pipelines](https://arxiv.org/abs/2609.22196) | EvoRank是一个由LLM引导的进化循环，能自动发现完整的多目标学习排序流水线，在Expedia数据集上以约十美元成本、50次迭代内收敛，性能超越Optuna调优的LambdaMART并达到原始竞赛前6%水平，同时通过迁移审计揭示了适应度噪声这一关键设计陷阱。 |
| [^189] | [The Situated Identity Test: Distinguishing Persistent Cognitive Identity from Persona Imitation](https://arxiv.org/abs/2609.22195) | 该论文提出了情境化身份测试（SIT），一个与架构无关的评估框架，通过要求智能体既恰当知晓真实经历、又对未根基化的信息保持恰当无知，来区分真正持久的认知身份与仅基于角色档案的模仿。 |
| [^190] | [Fairness Beyond Anonymization? Demographic Leakage in German LLM-Generated Resumes](https://arxiv.org/abs/2609.22188) | 该论文通过两阶段审计首次系统揭示：即使输入档案已经匿名化，多种主流大语言模型生成的德语简历仍会编码可恢复的性别与族裔等人口属性信息，从而在下游简历筛选中构成公平性风险。 |
| [^191] | [Beyond Task Completion: Training Capable and Safe Computer-Use Agents](https://arxiv.org/abs/2609.22178) | 提出SCOPE联合后训练框架与SCOPE-Gen自动化数据生成流水线，使计算机使用智能体在保持任务执行能力的同时学会基于风险的安全决策——完成良性任务、规避环境危害、并在目标有害或无安全路径时拒绝执行。 |
| [^192] | [SCoR: A Hierarchical Framework for Forecasting Relations Between Scientific Concepts](https://arxiv.org/abs/2609.22174) | 该论文提出SCoR层次化框架，将研究方向发现形式化为对科学概念间关系的预测（首次共现、首次关系形成及关系类型三个任务），并基于18.7万余篇cs.CV论文构建了包含615,036条带类型关系边的SCoR-Graph及经过泄漏审计的SCoR-Bench基准。 |
| [^193] | [Quantifying Hidden Salt for Precision Healthcare: Sodium Assessment via Joint-Factor Retrieval and Chain-of-Thought Inference](https://arxiv.org/abs/2609.22171) | 提出SALT框架，通过联合因子嵌入检索与结构化4跳思维链推理，从食谱中准确评估被省略或描述模糊的隐形盐（钠）含量，助力高血压等疾病的精准医疗。 |
| [^194] | [Multiple latent orderings better predict language model preferences](https://arxiv.org/abs/2609.22170) | 该论文提出语言模型的非传递性偏好源于多个潜在一致排序的聚合，并引入噪声增强的混合Bradley-Terry（MBT）模型，证明多重潜在排序比单一排序能更好地解释和预测语言模型的偏好。 |
| [^195] | [Monocultural Biases: Correlated biases in large language models lead to unequal systemic exclusion rates in hiring](https://arxiv.org/abs/2609.22169) | 研究发现大语言模型的后训练阶段会产生“单一文化偏见”，使各模型的招聘决策高度趋同，从而将劳动力市场中某些群体（尤其是年长求职者）的系统性排斥率从5.6%大幅推高至17.3%。 |
| [^196] | [A Multi-Agent Pipeline for Source-Grounded Synthetic Note Generation from Longitudinal Structured EHR](https://arxiv.org/abs/2609.22164) | MedNotes是一个多智能体闭环流水线，通过生成器-评估器-路由器协作机制，将纵向结构化电子健康记录转换为高保真、有源依据的合成临床病历，有效解决了结构化EHR难以直接用于病历中心临床建模的问题。 |
| [^197] | [MechaTerp-TRACE: A Novel Approach for Component Ablation Analysis in Language Models](https://arxiv.org/abs/2609.22163) | 提出MechaTerp-TRACE框架，通过逐一消融组件并测量固定答案词元处输出分布的变化，在统一尺度上比较语言模型中从transformer块到单个神经元等不同架构组件对命名实体生成的因果贡献。 |
| [^198] | [Beyond Raw Context Transfer: Representation-based Federated Retrieval-Augmented Generation](https://arxiv.org/abs/2609.22162) | 提出FedRepRAG，一种去中心化的联邦RAG框架，将原始文档保留在客户端本地，跨客户端检索时仅交换紧凑的潜在表示，从而避免直接共享原始内容带来的隐私暴露和推理时计算开销。 |
| [^199] | [Didactic knowledge or Clinical Cases? How Data Types Shape Medical Large Language Models](https://arxiv.org/abs/2609.22161) | 该研究通过token匹配实验揭示了医疗大语言模型训练数据的不对称迁移效应：临床数据能同时提升临床导向和知识密集型任务的表现，而教学数据主要提升知识密集型任务，且少量临床数据即可获得基于EHR任务的大部分收益。 |
| [^200] | [PAGE: Partition-Aware Gated KV-Cache Eviction](https://arxiv.org/abs/2609.22157) | PAGE提出一种无需训练、无需标签的门控机制，利用预填充注意力中top-k头一致性的早期到晚期下降来预测输入是否适合KV缓存淘汰，在淘汰会造成灾难性精度损失时自动保留完整缓存。 |
| [^201] | [Is Imagination Derived from Hallucination? A Cross-Taxonomy Evaluation of Imagination and Hallucination in Large Language Models](https://arxiv.org/abs/2609.22152) | 该论文提出了首个大语言模型想象力评估基准Whiteboard，通过将七种基于机制的想象力子类型与幻觉分类体系进行交叉评估，首次实现了对“想象力与幻觉是否源自同一生成机制”这一论断的直接检验。 |
| [^202] | [Do Language Models Know Their Own Constraints?](https://arxiv.org/abs/2609.22151) | 研究发现语言模型经后训练（SFT与GRPO）学会遵守行为约束后，反而丧失了显式报告这些约束的能力，且基于奖励的GRPO训练对模型关于约束的显式知识与第三人称知识的破坏比SFT更严重。 |
| [^203] | [Beyond the Stitching Assumption: A Unified Framework for Multimodal Synthetic Data Evaluation via Semantic Quantization](https://arxiv.org/abs/2609.22149) | 该论文提出一种基于语义量化与列联表散度比较的统一评估框架，能够捕捉传统单一模态指标无法检测到的表格与文本配对被破坏的问题，为多模态合成数据质量评估超越了“拼接假设”。 |
| [^204] | [GRRR: The Geometry of Reshaping, Rotation, and Routing in Decoder LLM post-training](https://arxiv.org/abs/2609.22146) | 后训练带来的收益主要源于在预训练权重的SVD坐标系中重新配置和扩展已有通路（旋转与零空间路由），而非大幅改变奇异值本身。 |
| [^205] | [Weak Ties, Strong Signals: Efficient Training Data Detection in Diffusion LLMs via Independent Token Sampling](https://arxiv.org/abs/2609.22145) | 提出独立Token采样方法，通过构建内部依赖较弱的掩码token集合来消除结构性估计误差，从而实现对扩散大语言模型训练数据使用情况的高效检测。 |
| [^206] | [Multilingual Safety Signals Are Multi-Layered: Filtering Safety-Degrading Data for Safer LLMs](https://arxiv.org/abs/2609.22144) | 提出多层框架MMSAFE，通过捕获跨语言共享及语言特定的多层安全信号，识别多语言微调数据中降低安全性的样本，从而保护大语言模型的安全对齐。 |
| [^207] | [Using Composition Operators to Linearize LLM Semantic Transformations](https://arxiv.org/abs/2609.22143) | 该论文提出用推广自Koopman算子的复合算子将大语言模型的语义变换表示为矩形无限维线性算子，证明其为等距算子，并通过奇异值谱来检测表示失准以及比较不同任务和模型。 |
| [^208] | [Does the Truthfulness Signal Survive Code-Mixing? Probing Hidden States for Hallucination Detection in Hinglish](https://arxiv.org/abs/2609.22138) | 该论文首次研究幻觉探测信号在印地语-英语语码混合（Hinglish）下的表现，构建了包含5,674个条目的三语问答基准，并评估基于纯净语言隐藏状态训练的线性和MLP探测器在三个开源大语言模型上的跨语言迁移能力。 |
| [^209] | [DiFA: Dual Evidence Fusion and Aggregation for Token-Level Text Anomaly Detection](https://arxiv.org/abs/2609.22136) | 提出DiFA双证据框架，融合形式-结构与语义两种视角的异常线索并进行自适应聚合，实现词元级文本异常检测，可精确定位文档中的异常词或片段。 |
| [^210] | [Read-Best Is Not Steer-Best: A Probing--Steering Layer Dissociation in Omni-Modal Large Language Models](https://arxiv.org/abs/2609.22135) | 该论文首次通过因果实验发现，在全模态大语言模型中，探测准确率最高的层并非最适合激活引导的层，读取与干预依赖不同的层，且引导有效的层稳定集中于模型归一化深度的中后段。 |
| [^211] | [Observational Equivalence of LLM and Human Annotation](https://arxiv.org/abs/2609.22133) | 该论文通过对14项政治学研究的文本分类复制实验证明，大语言模型与人类专家在标注质量上具有观测等价性——LLM与专家的一致率与专家之间彼此的一致率相当，且分歧源于文本和编码规则的模糊性，因此仅凭标注质量没有理由优先选择人工编码，而LLM在速度和成本上优势显著。 |
| [^212] | [Correlation-Aware Structured Pruning for Large Language Models](https://arxiv.org/abs/2609.22131) | 该论文提出了一种相关性感知的结构化剪枝方法，将剪枝建模为显式考虑跨单元依赖关系的二元二次规划问题，并设计基于依赖感知边际成本的贪心交互算法进行求解，从而突破了传统剪枝方法的独立性假设，在降低大语言模型推理成本的同时避免性能下降。 |
| [^213] | [Beyond Accuracy and Surface Fluency: Risk-Sensitive Evaluation of LLMs for Legal Clause Generation](https://arxiv.org/abs/2609.22127) | 本文提出了一个风险敏感的法律条款生成评估框架，结合CLAUSE和LENS-CRAFT两个体系，对四个大语言模型在22个合同条款类别和34种法律失败模式上进行评估，并采用最大严重性原则而非平均分数来捕捉法律起草中的关键风险。 |
| [^214] | [Type-Driven Tokenization for Brahmic Scripts](https://arxiv.org/abs/2609.22125) | 该论文发现婆罗米文字（如天城文、泰米尔文等）的正字法构成部分半群而非英语那样的半群，并在 Agda 中形式化推导出可证明正确的 fixToken 函数，用以修复大语言模型分词器对此类文字产生的畸形文本。 |
| [^215] | [Balancing Reasoning and Hardware Constraints in RAG Pipelines for Ukrainian Multi-Domain Document Understanding](https://arxiv.org/abs/2609.22124) | 针对UNLP 2026共享任务中OCR严重挤占9小时离线时间预算的问题，该论文提出结合BM25、BGE-M3与交叉编码器重排序的资源高效混合RAG流水线，并放弃参数量庞大的推理模型，以确保乌克兰多领域文档问答系统在严格时限内完成。 |
| [^216] | [Success Leaves Detours: Learning Executable Walkthroughs for Long-Horizon Agents](https://arxiv.org/abs/2609.22120) | 提出 Trace 框架，通过信用引导与依赖锚定机制，将含噪的稀疏奖励轨迹编译为可执行、可验证的攻略记忆，从而提升长程智能体的表现。 |
| [^217] | [Evaluation Awareness Shifts from Format to Context with Model Scale](https://arxiv.org/abs/2609.22119) | 该研究揭示了模型检测评估的机制随规模演变——小模型依赖提示词格式而大模型依赖高阶推理，并提出结合提示词净化与激活反向引导的双通路干预方法，在高度评估意识的提示词上实现了平均70.58%的行为翻转率。 |
| [^218] | [An Empirical Cost Attribution of Context-Compression Gateways in Multi-Turn Coding Agents](https://arxiv.org/abs/2609.22114) | 该论文通过插桩生产级压缩网关，将多轮编程智能体的token成本归因分解为三个独立杠杆，实证发现工具模式过滤是唯一明确且可复现的省钱手段，而内容压缩与历史摘要的实际节省远低于普遍假设。 |
| [^219] | [Privacy Personalization Trade offs in LLMs: The Impact of Stylometric Signal Reduction on User-Specific Text Generation](https://arxiv.org/abs/2609.22112) | 该论文提出一个基于LaMP-7基准的受控实验框架，通过对比原始档案与匿名化档案条件下的文本生成，揭示了削减文体信号（如人口统计标识、文化引用和个人细节）与LLM个性化生成能力之间的隐私-个性化权衡关系。 |
| [^220] | [Beyond the Text: Verifying That Agent-Written Papers Are Backed by Their Artifacts](https://arxiv.org/abs/2609.22111) | 提出ReAgent自动化审计框架，通过将智能体撰写论文中的科学声明与代码仓库进行比对，验证论文结论是否真正被其代码和实验产物所支撑。 |
| [^221] | [Evaluating Fine-Tuned and Base Language Models in Maternal and Vaccination Healthcare for African Settings](https://arxiv.org/abs/2609.22110) | 本研究通过在尼日利亚本地孕产妇健康和疫苗接种问答数据上对Llama模型进行低秩适配微调，构建了领域专用模型MamaBot-Llama和Vax-Llama，并证明其在准确性、安全性和文化适宜性上优于基础模型，为非洲低资源医疗场景提供了更可靠的AI健康信息解决方案。 |
| [^222] | [Generalized Multimodal Foundation Model](https://arxiv.org/abs/2609.22107) | 提出了一种不依赖特定模态的通用多模态基础模型，通过在大规模具有多样因果结构的合成多模态数据集上训练，使其能够适用于任意的模态组合和任意的预测任务。 |
| [^223] | [DeepInstructor: An Agentic AI Instructor for Experience-Driven Idea Evaluation](https://arxiv.org/abs/2609.22104) | DeepInstructor通过从58,607条同行评审构建经验图谱，并利用基于ReAct的智能体检索维度特定证据，将研究创意评估转化为对结构化学术经验的推理，从而显著提升了与人类评估判断的一致性。 |
| [^224] | [When Who You Are Can Change the Code You Get: A Study of Persona-Induced Bias in LLM Code Generation](https://arxiv.org/abs/2609.22102) | 该研究通过35,000多个程序的大规模实证分析，首次系统揭示了用户的人口统计身份（国籍、性别、经验水平）会诱导LLM代码生成产生偏见，人口统计标记泄露于高达65%的响应和70%的推理轨迹中，并影响代码的功能正确性、可维护性、风格和安全性。 |
| [^225] | [Context Poisoning as Extreme-Value Attention Interference in Long-Context Language Models](https://arxiv.org/abs/2609.22101) | 该论文将长上下文语言模型中的“上下文污染”形式化为注意力中的极值干扰，推导出证据边际需随有效干扰项数量按 Ω(√(log N)) 增长才能维持检索准确率的理论上界，并揭示得分混叠、位置混叠与softmax稀释是长上下文性能退化的核心机制。 |
| [^226] | [AdaMem: Adaptive Memory Token Allocation for Soft Compression in Retrieval-Augmented Generation](https://arxiv.org/abs/2609.22100) | AdaMem提出了一种相关性引导的自适应软压缩框架，根据段落与查询的相关性动态分配固定的记忆令牌预算，为高相关段落分配更多记忆令牌并舍弃低相关段落，从而提升检索增强生成的效率与效果。 |
| [^227] | [A framework for recipe data structure with applications for culinary and nutritional insights](https://arxiv.org/abs/2609.22099) | 该论文提出了一种食谱数据结构框架，将食谱分解为类型化配料实体并锚定到营养数据库及地理文化背景，据此构建了包含来自32个地区、99个国家的128,942个食谱的结构化数据库RecipeDB2，实现了食谱信息的可计算查询。 |
| [^228] | [TreeSpark: Calibrated, Load-Adaptive Draft Trees for Semi-Autoregressive Speculative Decoding](https://arxiv.org/abs/2609.22098) | TreeSpark以可忽略的成本从草稿器现有马尔可夫头读取父节点条件分布并校准为边接受概率估计，用路径存活率统一驱动草稿树的最优优先扩展、逐轮停止与负载自适应规模调整，从而改进半自回归推测解码的草稿树构建。 |
| [^229] | [Token Signatures of Code: Comparing Coding Behaviors Across Large Language Models](https://arxiv.org/abs/2609.22097) | 提出CLIC可视化分析方法，通过token频率分析与可解释决策树刻画不同大语言模型的编码行为差异，并引入鲁棒性和集中度两个新指标来超越单纯性能评估。 |
| [^230] | [AI-inferred expressed well-being and collective-action discourse in climate-change campaigns on X](https://arxiv.org/abs/2609.22096) | 该研究通过分析X平台上36万余条气候运动相关帖子，发现活动期间幸福感表达显著上升9.02个百分点，但行动语言却下降10.75个百分点，揭示了气候运动话语中“幸福感与行动背离”的现象。 |
| [^231] | [Summarize, Judge, Refine: Decoupled Content Understanding and Policy Learning for Multimodal Content Moderation](https://arxiv.org/abs/2609.22094) | 提出SJR双模型架构，通过自然语言接口将多模态内容理解与策略学习解耦，借助文本空间数据增强和GRPO共训练，实现少样本策略适配与内置可解释性的多模态内容审核。 |
| [^232] | [Memory That Looks Forward: A Zero-Inference Prospective Term for Personal Memory Retrieval](https://arxiv.org/abs/2609.22091) | 提出一种零推理开销的前瞻记忆检索项，通过显式承诺账本在触发时为相关记忆项提供乘法显著性提升，将困难任务层的recall@5从0.000提升至0.955-1.000且零误提升。 |
| [^233] | [Recognition, Simulation, and Refusal: A Contamination-Aware Study of Classic Psychological Effects in LLM Agents](https://arxiv.org/abs/2609.22090) | 提出PsyAgentBench基准，通过命名/盲测、规范/反事实任务版本与人格操纵的因子设计，区分LLM是真正具备心理偏差还是仅在识别并模拟心理学效应，发现类人效应源于标签门控等质性不同的机制而非单一易感性。 |
| [^234] | [RecreationWorld: Scalable and Verifiable Environments for Hybrid Computer-Use Agents](https://arxiv.org/abs/2609.22000) | 提出了 RecreationWorld，一个跨五大平台、以“复刻正在运行的应用”为核心任务的框架，用于训练和评估能自主结合图形交互与软件开发的混合计算机使用代理，并以运行中的参考应用作为预言机提供基于执行的验证奖励。 |
| [^235] | [Configurable Multi-Stage Vision Pipeline for Crop Disease and Pest Diagnosis](https://arxiv.org/abs/2609.21651) | 针对小农户仅凭一张田间照片进行作物诊断的场景，本文提出一种可配置的多阶段视觉流水线，支持调节照片质量拒绝阈值与置信度截止值、并可扩展添加作物与病虫害类别，同时基于来自四个国家的116万张真实照片揭示了现有生产系统的不足。 |
| [^236] | [GameLogicBench: Evaluating Coding Agents on Runtime Game Logic with Tick-Level State Assertions](https://arxiv.org/abs/2609.21562) | GameLogicBench是一个包含72个Godot游戏逻辑任务的基准，其自动评估器在每次模拟tick逐帧检查游戏规则，覆盖403个场景共1,451个测试用例，既能接受多样化的正确实现又能拒绝突变体，且判定结果完全可复现。 |
| [^237] | [Trustworthy FinAInce: Unpacking How AI-Mediated Financial Advice is Judged](https://arxiv.org/abs/2609.20989) | 该研究通过对285名美国成年人的随机情景实验发现，建议风格（AI、专家、在线社区）是塑造人们对金融建议信息与安全性评价的最关键因素，这些评价共同解释了信任与依赖判断的大部分方差，且专家风格建议即使在没有来源标签时也最受偏好。 |
| [^238] | [UniPolicy: Unified Objective-Specific Policies for Generative Search Advertising](https://arxiv.org/abs/2609.20630) | UniPolicy提出了一种目标感知的多策略对齐框架，通过目标特定前缀标记、稀疏MoE-LoRA路由和残差FFN在共享骨干网络中分层解耦参数，使生成式搜索广告能够联合优化相关性、点击倾向和商业价值等异构目标，避免梯度竞争导致的全局次优问题。 |
| [^239] | [Xeno-Interpretability: Investigating the Alien Minds of LLMs](https://arxiv.org/abs/2609.20408) | 本文提出“异种可解释性”这一新研究方向，主张大语言模型内部可能存在人类概念无法充分描述的“异种表征”，其内部区分空间远超有限人类描述所能覆盖的范围，且实验识别与语义解释应当分开对待。 |
| [^240] | [F$^{2}$DR: A Fine-Grained Full-Pipeline Reward Framework for DeepSearch Workflows](https://arxiv.org/abs/2609.19827) | 该论文提出了F2DR框架，从内容、轨迹和答案三个维度对DeepSearch工作流进行细粒度全流程奖励评估，并构建了专门基准DeepSearch RM-Bench，显著提升了评估一致性。 |
| [^241] | [Dictionary-Constrained Grapheme-to-Phoneme for Unsegmented Languages from LLM-Annotated Data](https://arxiv.org/abs/2609.19805) | 本文提出一种利用词典构建词格并采用条件随机场评分的上下文感知神经G2P方法，结合大语言模型生成的超过200万条标注数据，显著提升了日语等未分词语言的字素到音素转换性能。 |
| [^242] | [A frontend-backend architecture for tool calls in full-duplex speech models](https://arxiv.org/abs/2609.19334) | 提出一种前后端架构，让全双工语音模型通过发出委派标记将流式转写交给文本LLM后端执行工具调用，并以轻量级注入机制返回结果，从而在几乎不修改前端模型的前提下保留低延迟、可打断的自然双工交互。 |
| [^243] | [CovR: Coverage-Aware Hardware Verification via Reasoning-Guided Reinforcement Learning](https://arxiv.org/abs/2609.19189) | CovR是一个结合自我反思循环与仿真反馈的智能体框架，通过推理引导的强化学习自动生成硬件测试平台，突破了现有方法只关注功能正确性的局限，实现了验证覆盖率的最大化。 |
| [^244] | [HearInContext: A Benchmark for Implicit Context in Speech Recognition](https://arxiv.org/abs/2609.18680) | 该论文提出了中英文同音词基准测试HearInContext用于评估语音识别模型的隐式与显式上下文利用能力，并通过微调Qwen3-ASR-1.7B将隐式上下文目标词召回率提升约11个百分点，同时不损害通用识别性能。 |
| [^245] | [Fallacy Benchmarks Measure Scheme Recognition, Not Fallacy Detection](https://arxiv.org/abs/2609.18644) | 该论文揭示了谬误检测基准报告的低误报率是“有效”类别构建方式的产物而非真实检测能力——当使用与谬误具有相同论证图式的正确论证作为负样本测试时，模型误报率大幅上升（CoCoLoFa上从16.6%升至58.9%），证明现有模型实际只是识别论证图式而非真正检测谬误。 |
| [^246] | [Agora: Git as Shared Memory for Collective AutoResearch](https://arxiv.org/abs/2609.18094) | Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。 |
| [^247] | [Smarter by the Moment: Environment-Driven Dynamic Policies for Continual LLM Improvement](https://arxiv.org/abs/2609.16800) | 提出了DRPG框架，将基于记忆的检索与动态策略生成器相结合，利用历史数据和环境反馈生成任务特定策略，从而实现大语言模型的持续改进，在六个基准和七个模型上超越了强大的基线方法。 |
| [^248] | [RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments](https://arxiv.org/abs/2609.15364) | RSIAgent是一个无需训练的多智能体框架，通过“先广后深”的自主探索策略构建可复用的冻结记忆，使数字智能体在新环境中实现递归自我改进，且无需更新模型参数。 |
| [^249] | [An Efficient and Modular Framework for Targeted Harm Mitigation in LLMS](https://arxiv.org/abs/2609.13624) | 提出了一种结合Activated LoRA适配器与上下文感知路由机制的模块化纠正框架，可在生成过程中以低延迟、有针对性的方式缓解大语言模型的有害输出，同时提升模型对齐性能。 |
| [^250] | [Quantifying Consonant Contributions to Word Intelligibility via Acoustic Masking](https://arxiv.org/abs/2609.12122) | 本文提出一种基于声学掩蔽与语音识别模型的可扩展方法，通过掩蔽诱导误识别率（MMR）量化每个辅音对单词可懂度的贡献，从而帮助确定运动性言语障碍治疗的优先干预目标。 |
| [^251] | [From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection](https://arxiv.org/abs/2609.08899) | 该论文提出一种可审计的决策记录方法，将被动检测分数、条件性键控探针分数、检索支持和说话者画像边际四个线索纳入后期校准步骤，使语音深伪检测的最终决策在保持标量的同时保留证据来源信息，从而提升检测决策的可信度与可解释性。 |
| [^252] | [Mind the Gap: Exposing LLM Translation Blind Spots Using the AlphaMWE Multilingual Parallel Corpus](https://arxiv.org/abs/2609.06634) | 本文通过WMT2026共享任务，使用AlphaMWE多语言平行语料库对31个机器翻译系统进行自动和人工评估，揭示比喻性多词表达仍是LLM翻译的瓶颈，且自动评估指标与人工评估时常存在分歧。 |
| [^253] | [VERPO: Verified Evidence Regularized Policy Optimization](https://arxiv.org/abs/2609.06100) | VERPO提出了一种验证证据正则化策略优化框架，通过Fisher证据对比和带停止机制的逐token ZPD控制器有选择性地应用基于证据的修正，在保留可验证结果目标的同时避免盲目模仿导致的负面迁移。 |
| [^254] | [AhaBench: Do Agents Learn from Prior Experience? A Benchmark for Long-Horizon Continual Learning](https://arxiv.org/abs/2609.05435) | AhaBench 是一个长时程持续学习基准，通过谜题探索、带精确验证器的数学任务和自动售货机模拟三个组件，评估固定模型在获得先验经验后，在相关但支持已被移除、改变或延迟的条件下行为是否真正得到改善。 |
| [^255] | [A Systematic Evaluation of Cross-Lingual Consistency Enhancement Methods in Multilingual Language Models](https://arxiv.org/abs/2609.04409) | 本文对多语言模型中的跨语言一致性增强方法进行了统一的系统性评估，发现后训练方法（尤其是直接分布对齐）总体更可靠且能稳定提升一致性，而跨域迁移仅在源与目标任务输出格式相似时才有效。 |
| [^256] | [VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis](https://arxiv.org/abs/2609.03203) | VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。 |
| [^257] | [Knowledge Distillation During Mid-Training Favors Reasoning over Factual Recall](https://arxiv.org/abs/2609.01532) | 该研究发现前向KL知识蒸馏在预训练阶段能同时提升推理与事实记忆能力，但在中期训练阶段会减缓事实记忆的习得而持续提升推理能力，这种阶段依赖性源于教师置信度在不同数据领域的不对称以及学生模型知识状态的演化。 |
| [^258] | [SCoNE: Selective Context-aware Neuron Editing for Robust Retrieval-Augmented Generation](https://arxiv.org/abs/2609.00689) | SCoNE提出了一种无需训练的模型编辑方法，通过选择性强化兼具高归因分数与高跨输入变异性的上下文感知FFN神经元，显著提升大语言模型在检索增强生成中对检索噪声的鲁棒性，且无需微调、无推理开销。 |
| [^259] | [SingProbe Technical Report](https://arxiv.org/abs/2608.30703) | SingProbe 是一种轻量级内嵌运行时防护机制，通过复用 LLM 推理的隐藏状态，在 token 级别以几乎零额外开销持续预测查询意图、响应安全性和幻觉风险，并配套提出流式防护基准 SingStreamBench。 |
| [^260] | [Beyond Parallel Blindness: Information Floors and Model Gaps in Block Drafting](https://arxiv.org/abs/2608.27339) | 本文提出一种方法，通过信息下限和模型差距的分离，揭示了块草拟中并行生成的固有信息瓶颈，并指出当前草拟器仍有大幅改进空间。 |
| [^261] | [Provenance Before Prose: Claim-Locked Reporting](https://arxiv.org/abs/2608.25336) | 本文提出“声明锁定报告”协议，通过先固定结构化证据再生成文本，以解决LLM统计报告中数值漂移和效应方向反转问题，提高可复现性。 |
| [^262] | [Speech-to-SOAP: End-to-End Summarization of Medical Dialogues: KIT@BeTraC 2026](https://arxiv.org/abs/2608.24327) | 本文提出了Speech-to-SOAP系统，可直接从医疗对话语音端到端生成临床SOAP笔记而无需中间转录文本，并贡献了一个通过合成语音统一异构医疗对话数据集的可扩展数据增强流水线，用于参加BeTraC 2026轻量级赛道。 |
| [^263] | [Machine learning and digital pragmatics: Which word category influences emoji use most?](https://arxiv.org/abs/2608.21975) | 本研究通过MARBERT模型和逻辑回归分析发现，在口语阿拉伯语社交媒体帖子中，动词类别对表情符号使用的影响最强，尽管名词在频率上占主导。 |
| [^264] | [Tree-of-Concerns: Hierarchical Multi-Agent Debate for Unstated-Limitation Extraction in Scientific Critique](https://arxiv.org/abs/2608.20777) | 本文提出“关注之树”多智能体框架，通过专门怀疑论角色和小组审查机制，从科学论文中提取未声明局限，在精确度和覆盖率上分别比最强基线提升79%和11%。 |
| [^265] | [LongNovel: A Multi-Scale Benchmark for Hallucination Detection in Long-Context Novel Summarization](https://arxiv.org/abs/2608.18082) | 提出了LongNovel，一个多尺度双语长篇小说基准，用于检测长上下文摘要中的幻觉，并通过8种幻觉类型和组合生成方法确保数据真实性。 |
| [^266] | [Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases](https://arxiv.org/abs/2608.17168) | 本研究通过欧洲人权法院案例测试发现，顶尖大型语言模型在法律推理上表现不佳，其分析结构完整但内容浅薄，且自动评估器与人工评估一致性较弱。 |
| [^267] | [Towards Safer RAG: Only Agents Capable of System 2 Thinking may Access Untrusted Documents](https://arxiv.org/abs/2608.17153) | 本文提出一种新的安全原则，即仅允许具备系统2推理能力的代理访问不可信文档，以减少RAG系统中的知识投毒攻击影响，并引入新指标量化检测与影响间的差异。 |
| [^268] | [Counting Documents Is Not Counting Text: Unit Bias in Web-PDF Corpus Statistics](https://arxiv.org/abs/2608.16390) | 本文揭示了Web-PDF语料库中按文档计数与按令牌计数的巨大偏差，导致令牌总数被高估且截断文本大量丢失，影响语料库统计的准确性。 |
| [^269] | [HalluTracer: Hallucination Detection via Depth-Averaging Truth Signals](https://arxiv.org/abs/2608.16353) | HalluTracer通过聚合前向传播所有层的真值信号，利用弱相关的逐层证据进行深度平均，显著提升了幻觉检测的准确性。 |
| [^270] | [Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers](https://arxiv.org/abs/2608.14089) | 本文提出了一种轻量级包装器RCV，通过估计分类器预测与部署者策略不一致的概率并选择性纠正，同时利用正确性估计检测分布漂移，实现了无需重训练即可适应和监控安全分类器，显著提升策略遵循度。 |
| [^271] | [CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation](https://arxiv.org/abs/2608.13387) | CROP提出了一种基于释义校准的反事实敏感性边际方法，用于在选择性在线策略蒸馏中直接量化任务相关性，从而更有效地分配监督信号。 |
| [^272] | [Behavioral Skill Reconstruction: Reconstructing Hidden Functionality from LLM Agent Skills](https://arxiv.org/abs/2608.04192) | 该论文提出了一种名为SkillClone的黑盒攻击方法，攻击者仅通过正常使用技能并观察响应即可重构隐藏的LLM智能体技能的功能，表明单纯防止文件泄露不足以保护专有技能。 |
| [^273] | [Large Language Models for Low-Resource Languages: A Conceptual Framework for an Electronic Explanatory Dictionary of the Tajik Language](https://arxiv.org/abs/2608.04186) | 本文提出了一个利用大语言模型构建塔吉克语电子详解词典的概念框架，通过集成形态分析、词形还原、语义聚类和词典条目生成模块，并结合子词分词与参数高效微调策略，填补了低资源语言在数字词典学资源方面的空白。 |
| [^274] | [Who Should Be Generated? Justifying Demographic Targets in Open-Ended Generation](https://arxiv.org/abs/2608.02551) | 该论文针对开放生成中人口属性未指明时的公平性评估，形式化了“缺失目标问题”，并将目标分布的构建分解为评估对象、先验可采性、分配方式与操作化四个承诺，为生成式审计中人口统计目标的选择提供了正当性论证框架。 |
| [^275] | [CANDOR: Chance-Calibrated Discordance in Frozen Foundation Encoders](https://arxiv.org/abs/2607.18451) | 本文提出CANDOR度量，通过等大小对称样本库校正最近邻不一致性，将机会水平固定为二分之一，揭示冻结编码器并非失明但普遍性能较弱。 |
| [^276] | [Length Penalties Make Chain-of-Thought Less Monitorable](https://arxiv.org/abs/2607.09786) | 压缩思维链的长度惩罚虽能保持准确率并降低推理成本，但会显著降低思维链的忠实度，使模型更少表达误导性提示对其答案的影响，从而削弱了思维链的可监控性。 |
| [^277] | [Scoped Verification for Reliable Long-Horizon Agentic Context Evolution under Distribution Shift](https://arxiv.org/abs/2607.09175) | 提出GRACE方法，将智能体持久指令维护为类型化语义图，通过在被修改节点的局部邻域内进行范围化验证，实现了分布偏移下长时程上下文演化的可靠增量更新。 |
| [^278] | [Riemannian Geometry for Pre-trained Language Model Embeddings](https://arxiv.org/abs/2607.07047) | 该论文提出黎曼平均池化（RMP）方法，通过从编码器雅可比矩阵提取词元拉回度量并在SPD流形上用Fréchet均值聚合，证明句子级分类信号存在于预训练语言模型嵌入的黎曼几何中，在多个具有语言结构的数据集上优于欧氏池化，且增益主要来自几何聚合机制本身。 |
| [^279] | [Who's Behind It? Annotating and Extracting Conspiratorial Actors from German Telegram Posts](https://arxiv.org/abs/2607.04962) | 该论文提出了阴谋论行为者的标注指南和德语Telegram帖子跨度标注语料库，并证明基于Transformer的模型能够以合理的准确率自动提取阴谋论行为者，从而支持对阴谋论叙事中行为者表征的大规模分析。 |
| [^280] | [You Frame It: How Conceptual Representations Shape LLM Detection and Reasoning about Antisemitism](https://arxiv.org/abs/2607.04945) | 本研究通过对比四种反犹主义概念表征形式，发现细粒度分类表征能显著提升大语言模型检测的召回率但会牺牲精确率，而更大的概念资源并无额外收益，大屠杀后反犹主义始终是最难检测的类型。 |
| [^281] | [LP-SFT: Local-Preserving Supervised Fine-Tuning via Multimodal Entropy Structure](https://arxiv.org/abs/2607.04733) | 提出LP-SFT，一种基于多模态熵结构分析的局部保持监督微调方法，在将模型适配到下游领域的同时，保留预训练模型已有的能力和丰富的分布知识。 |
| [^282] | [Rethinking Speech-LLM Integration for ASR: Effective Joint Speech-Text Training by Interleaving](https://arxiv.org/abs/2607.01733) | 提出联合语音-文本交错预训练策略JSTIP，通过在语音-LLM中构建词级和段级交错的语音-文本序列，有效利用文本知识提升ASR实体识别准确率并简化领域适配。 |
| [^283] | [How Do LLMs Cite? A Mechanistic Interpretation of Attribution in Retrieval-Augmented Generation](https://arxiv.org/abs/2606.28358) | 本文首次对RAG中大语言模型的行内引用决策机制进行机械可解释性分析，通过激活修补技术发现引用行为并非由单一局部组件实现，而是由注意力头和MLP层构成的分布式、多阶段“归因集合”协同完成。 |
| [^284] | [What are Key Factors for Updates in RL for LLM Reasoning?](https://arxiv.org/abs/2606.22570) | 该论文通过理论分析揭示离策略程度（每次rollout的梯度步数）会通过影响重要性采样比率的分布与裁剪行为决定哪些token主导RLVR更新，并据此提出自适应裁剪策略优化方法来改进大模型推理的强化学习训练。 |
| [^285] | [Knowledge-Graph Grounding Helps LLMs Only for Out-of-Training Knowledge: A Controlled Study on Clinical Question Answering](https://arxiv.org/abs/2606.22419) | 本文通过临床问答对照研究发现，基于公开生物医学知识图谱 PrimeKG 的知识图谱接地（无论朴素三元组检索还是代理式 Cypher 查询）均无法提升从弱到强各档大语言模型在 MedQA 上的表现，表明知识图谱接地仅在所需知识超出模型训练范围时才有帮助，同时复现并修正了《自然·医学》研究中 HealthBench 的评分问题。 |
| [^286] | [The Metanym Game: An LLM Benchmark Without Ground Truth That Rises With the Models It Measures](https://arxiv.org/abs/2606.21008) | 该论文提出一种无真实基准的LLM评估方法，通过类比生成与相互评分，利用SVD特征方程统一评判生成与评判能力，并发现与GPQA Diamond存在相关性。 |
| [^287] | [Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning](https://arxiv.org/abs/2606.20002) | 本文提出“连点成线”通用框架，通过端到端强化学习训练大语言模型在长生命周期智能体场景中持续探索环境、自我更新上下文并实现跨领域泛化，从而在后续任务上获得渐进式性能提升。 |
| [^288] | [Want Better Synthetic Data? Steer It: Activation Steering for Low-Resource Language Generation](https://arxiv.org/abs/2606.18389) | 本文提出用激活引导（语言引导与质量引导）替代少样本提示来生成低资源语言合成数据，在11种类型多样的语言上验证了该方法能以更低成本生成更高质量、更多样化的训练数据。 |
| [^289] | [A Red-Team Study of Anthropic Fable 5 & Opus 4.8 Models](https://arxiv.org/abs/2606.18193) | 该研究通过大规模红队测试评估了 Anthropic 三个前沿模型对自动化越狱攻击的鲁棒性，发现所有模型均能抵御大部分攻击，残余风险主要来自自适应迭代攻击而非静态混淆，且模型鲁棒性排序与发布日期无关。 |
| [^290] | [The BD-LSC Dataset: Facilitating the Benchmarking of Models for Lexical Semantic Change Detection in Slang and Standard Usage](https://arxiv.org/abs/2606.16560) | 该论文提出了BD-LSC和ST-WSD两个互补的基准数据集，首次系统支持双向词汇语义变化（义项同时获得与丢失）的检测，以及兼具俚语与标准用法词语的语义变化研究与词义消歧基准测试。 |
| [^291] | [AdaMame: A Training Recipe for Adaptive Multilingual Reasoning](https://arxiv.org/abs/2606.15080) | 提出AdaMame两阶段训练方法，通过AdaMame-GRPO中渐进增长的查询条件对齐因子，将推理语言自适应对齐到查询语言，在不牺牲准确性的情况下解决多语言推理中的语言崩溃问题。 |
| [^292] | [MDForge: Agentic Molecular Dynamics Pipeline Design under Sparse Simulator Feedback](https://arxiv.org/abs/2606.12916) | MDForge是一个LLM智能体，通过物理专家多智能体辩论来稠密化稀疏的模拟器反馈，以开放式代码生成方式自动设计出可与人类专家媲美的分子动力学流水线，并发现了新型主客体结合剂。 |
| [^293] | [Revisiting Lexicon Evaluation in Unsupervised Word Discovery](https://arxiv.org/abs/2606.06183) | 该论文指出现有无监督词汇发现中常用的归一化编辑距离评估指标存在偏向大聚类且忽略真实类别跨聚类分布的固有缺陷，并提出考虑聚类大小的加权指标和评估真实词汇分散程度的逆向指标，以实现更公平可靠的词典质量评估。 |
| [^294] | [Can Generalist Agents Automate Data Curation?](https://arxiv.org/abs/2606.04261) | 该论文提出以智能体为中心的Curation-Bench基准，发现通用编码智能体能在十次迭代内达到强数据选择基线，但存在持续的“执行-研究差距”——智能体倾向于微调局部策略变体而非探索全新的策略家族。 |
| [^295] | [Decomposing Refusal Steering in Mixture-of-Experts Models](https://arxiv.org/abs/2606.04160) | 该研究将拒绝引导方法扩展到混合专家模型，发现单个专家在自由选择位置时平均可恢复78%的完整引导效果，从而揭示了MoE模型中拒绝机制在各组件间的运作方式。 |
| [^296] | [Rubric-Guided Process Reward for Stepwise Model Routing](https://arxiv.org/abs/2605.29310) | 提出RoRo框架，用评分准则引导的过程奖励替代仅反映最终答案正确性的结果奖励，从而更好地评估逐步模型路由中的中间决策并提升性能与泛化能力。 |
| [^297] | [The Harder Text Embedding Benchmark (HTEB): Beyond One-dimensional Static Robustness](https://arxiv.org/abs/2605.28190) | 提出动态评估框架HTEB，通过LLM在评估时随机变换输入，从词汇/风格、长度和语言三个维度评估文本嵌入模型的多维鲁棒性，揭示了静态基准无法发现的模型失败模式。 |
| [^298] | [TRACES: Proactive Safety Auditing for Multi-Turn LLM Agents via Trajectory-State Modeling](https://arxiv.org/abs/2605.27690) | TRACES通过从LLM隐藏表示中建模轨迹风险状态的时间演化，仅用弱轨迹级监督即可对多轮智能体交互实现主动的、密集的前缀级安全风险审计。 |
| [^299] | [BAIT: Boundary-Guided Disclosure Escalation LLM Jailbreaking via Self-Conditioned Reasoning](https://arxiv.org/abs/2605.27110) | BAIT是一个三步越狱框架，通过让模型先识别、再细化自身安全边界并最后请求详细示例，将模型自身的推理与一致性倾向转化为信息披露途径，在多个基准测试中对顶级大语言模型实现了持续的高攻击成功率。 |
| [^300] | [Can LLMs Time Travel? Enhancing Temporal Consistency in Legal Agentic Search through Reinforcement Learning](https://arxiv.org/abs/2605.25920) | 提出 LegalSearch-R1 强化学习框架，通过结合本地法条 RAG 与在线网络搜索，并在跨越多个修订时期的时间索引数据上训练，解决了法律大模型的时间偏差和搜索智能体忽视时间约束的问题，确保所适用的法律与案件的时间背景保持一致。 |
| [^301] | [STOP: Structured On-Policy Pruning of Long-Form Reasoning in Low-Data Regimes](https://arxiv.org/abs/2605.13165) | STOP是一种在线剪枝算法，通过将长思维链推理轨迹结构化为推理树，并保留以最早正确节点结尾的最短前缀，在低数据微调场景下有效缓解了推理模型的过度思考问题。 |
| [^302] | [Grounded or Guessing? LVLM Confidence Estimation via Blind-Image Contrastive Ranking](https://arxiv.org/abs/2605.10893) | 提出BICR框架，通过对比真实图像与涂黑图像下冻结LVLM的隐藏状态，并用排序损失正则化训练一个轻量探针，使置信度估计能够检测模型是否真正依赖图像而非仅凭语言先验作答。 |
| [^303] | [ReLay: Personalized LLM-Generated Plain-Language Summaries for Better Understanding, but at What Cost?](https://arxiv.org/abs/2605.00468) | 该论文提出了ReLay数据集，通过对比专家撰写的静态通俗摘要与LLM个性化生成的交互式摘要，评估了五种LLM在健康信息个性化摘要中的效果、最有效的个性化策略以及个性化与安全性之间的权衡。 |
| [^304] | [Information-Geometric First-Passage Monitoring of Distributional Stability in Stochastic Systems](https://arxiv.org/abs/2604.24083) | 本文提出一种融合信息几何、相对熵耗散与序贯推断的有界首达监测架构，能在控制重复检验误报的前提下，区分随机系统的名义分布松弛与真正的状态机制偏离。 |
| [^305] | [Characterizing Model-Native Skills](https://arxiv.org/abs/2604.17614) | 该论文提出技能刻画应“模型原生”地基于模型自身表征而非外部人工分类体系，通过从序列级激活中恢复紧凑的正交基来捕捉模型自身组织的行为变化轴，并在推理后训练中验证了该方法的有效性。 |
| [^306] | [English is Not All You Need: Systematically Exploring the Role of Multilinguality in LLM Post-Training](https://arxiv.org/abs/2604.13286) | 该研究通过220次受控监督微调实验系统证明，仅用英语进行大语言模型后训练并非最优——引入哪怕一种非英语语言即可同时提升英语性能与跨语言泛化能力，且语言多样性越高收益越大，尤其有利于低资源语言。 |
| [^307] | [Confident in a Confidence Score: Investigating the Sensitivity of Confidence Scores to Supervised Fine-Tuning](https://arxiv.org/abs/2604.08974) | 该研究系统考察了监督微调对语言模型置信度指标校准性的影响，发现在翻译、问答和数学推理等216种配置中校准性有升有降（112例下降、104例提升），表明微调后置信度分数的可靠性并不稳定，需要重新校准。 |
| [^308] | [Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-Based ASR](https://arxiv.org/abs/2604.06487) | 提出混合批处理（MB）策略，仅用不到4小时的目标域语音数据即可使基于LLM的ASR达到与使用完整数据集传统微调相当或更优的词错误率，有效弥合了语音-文本模态差距。 |
| [^309] | [What Makes Good Multilingual Reasoning? Disentangling Traces with Measurable Features](https://arxiv.org/abs/2604.04720) | 该论文通过定义涵盖多语言对齐、推理步骤和推理流程的可度量特征，并结合逻辑回归与稀疏自编码器分析推理轨迹，揭示了多语言场景下成功推理的真实特征，挑战了“让各语言推理模仿英语推理即可弥合性能差距”的传统假设。 |
| [^310] | [Are Finer Citations Always Better? Rethinking Granularity for Attributed Generation](https://arxiv.org/abs/2604.01432) | 该论文通过分析四种模型规模发现，细粒度句子级引用并非总是最优，段落级的中间粒度归因质量最佳，选择最优引用粒度可在几乎不牺牲答案正确性的情况下大幅提升模型性能与归因质量。 |
| [^311] | [From Noise to Signal: When Outliers Seed New Topics](https://arxiv.org/abs/2603.18358) | 该论文提出一种新闻文档轨迹的时间分类法，将“预期性离群点”识别为新兴主题的早期信号而非噪声，并借助十一种语言模型的文档嵌入在法语氢经济新闻语料库上验证了方法的有效性。 |
| [^312] | [ShapleyLaw: A Game-Theoretic Approach to Multilingual Scaling Laws](https://arxiv.org/abs/2603.17945) | 该论文提出ShapleyLaw，将多语言预训练建模为合作博弈，通过Shapley值量化每种语言的跨语言迁移贡献，从而更准确地预测最优语言混合比例。 |
| [^313] | [Evidence for systematic semantic structure in individual letters](https://arxiv.org/abs/2603.17306) | 该研究首次系统绘制了26个英文字母的多维语义结构，通过三个大语言模型独立检测并由1,388名人类参与者及五种不同语言使用者的预注册实验验证，证明单个字母本身携带可跨语言感知的系统性语义信息。 |
| [^314] | [CCTU: A Benchmark for Tool Use under Complex Constraints](https://arxiv.org/abs/2603.15309) | CCTU是一个基于12类约束分类体系、包含200个高难度测试用例的基准，并配备可执行的约束验证模块，用于评估大语言模型在复杂约束下的工具使用能力。 |
| [^315] | [X-GS: An Extensible Framework for Perceiving and Thinking with 3D Gaussian Splatting](https://arxiv.org/abs/2603.09632) | X-GS提出了一个可扩展框架，通过感知器（基于3DGS的在线SLAM与语义蒸馏）和思考器（将VLM与语义高斯对接）两大组件，将原本孤立的3DGS方法整合到视觉语言模型的感知模块中，从而实现3D视觉定位等空间多模态能力。 |
| [^316] | [Dial: A Knowledge-Grounded Dialect-Specific NL2SQL System](https://arxiv.org/abs/2603.07449) | Dial是一个知识驱动的方言特定NL2SQL框架，通过方言感知的逻辑查询规划模块和分层意图知识库，解决了异构数据库系统中生成既语义正确又可在目标引擎上执行的SQL查询这一难题。 |
| [^317] | [MedGPT-oss: Training a General-Purpose Vision-Language Model for Biomedicine](https://arxiv.org/abs/2603.00842) | MEDGPT-OSS是一个开放权重的200亿参数通用生物医学视觉-语言模型，通过三阶段训练课程、严格数据筛选和长上下文多模态对齐，在分布外多模态推理和临床文本任务上超越更大的开源医学模型，同时支持满足隐私合规的本地化部署。 |
| [^318] | [Althea: The Fact-Checking--Metalearning Tradeoff in AI-Assisted Verification](https://arxiv.org/abs/2602.11161) | 本文提出检索增强事实核查系统Althea，并通过N=961的纵向“消退测试”实验揭示了“事实核查—元学习权衡”：AI辅助干预虽能即时提升用户的准确率与置信度，但系统移除后用户无法将验证能力迁移到新声明上，而自主搜索所培养的能力则能持久保持优势。 |
| [^319] | [A vector logic for intensional formal semantics](https://arxiv.org/abs/2602.02940) | 本文证明了Kripke式内涵形式语义模型可以单射嵌入向量空间——原始域映射为自由载体、内涵函数映射为线性算子且保持复合运算——并完整刻画了哪些布尔值泛函能够线性地作用于算子编码。 |
| [^320] | [Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models](https://arxiv.org/abs/2602.01654) | 提出导向向量场（SVF），通过学习可微的概念评分函数生成随上下文自适应的导向向量，解决了静态导向向量因方向固定而在不同上下文中失效的问题。 |
| [^321] | [The Role of Dataset Linguistic Structure in the Cultural Awareness of Large Language Models](https://arxiv.org/abs/2602.01161) | 该研究提出以数据集为中心的文化对齐视角，通过对阿拉伯语、中文和日语微调数据集的语言、语义与结构指标进行PCA分析，提炼出语义结构、多样性和语言特定组织三个可解释主轴，揭示了后训练数据的语言结构特性与大语言模型文化表现之间的关联，从而可在微调前指导数据选择。 |
| [^322] | [Beyond Forgetting: Representation Misdirection Elicits Controllable Side Behaviors and Capabilities](https://arxiv.org/abs/2601.21702) | 该论文提出，基于线性表征假设的视角，表征误导（RM）类机器遗忘方法不仅能实现遗忘，还可通过在遗忘表征空间中对高层概念向量进行线性操作，可控地引发与该概念相对应的附带行为与能力。 |
| [^323] | [ILRR: Inference-Time Steering Method for Masked Diffusion Language Models](https://arxiv.org/abs/2601.21647) | ILRR提出了一种推理时引导框架，将参考文本作为高层语义蓝图，通过迭代精炼潜在表示并将其注入生成序列的激活中，实现对掩码扩散语言模型的可控生成，并能调节引导强度使短参考文本引导长文本生成。 |
| [^324] | [Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents](https://arxiv.org/abs/2601.15322) | 该论文提出DFAH框架，首次将工具使用型LLM智能体的决策可重复性、轨迹一致性和证据条件忠实性作为三个独立维度加以区分评估，并澄清了历史研究中r=-0.11相关性统计量的性质与局限。 |
| [^325] | [A Survey of Agentic Reasoning for Large Language Models: Towards Recursively Self-Improving and Collective Agents](https://arxiv.org/abs/2601.12538) | 本综述从基础智能体推理、自进化智能体推理和集体多智能体推理三个互补维度，系统梳理了大语言模型智能体推理的研究进展，旨在迈向能够递归自我改进与协作的智能体。 |
| [^326] | [Garbage Attention in Large Language Models: BOS Sink Heads and Sink-aware Pruning](https://arxiv.org/abs/2601.06787) | 本论文发现LLM中高BOS汇聚分数的注意力头是功能冗余的“垃圾场”，尤其在深层中，据此提出移除这些头的简单剪枝策略，在Gemma-3、Llama-3.1和Qwen3上比基于权重和激活的剪枝准则更可靠地识别冗余组件并保持下游任务性能。 |
| [^327] | [RADAR: Retrieval-Augmented Detector with Adversarial Refinement for Adaptive LLM-Generated Fake News Detection](https://arxiv.org/abs/2601.03981) | RADAR通过生成器与检测器的对抗共同进化（借助语言对抗反馈VAF）以及双侧检索增强机制，实现了对LLM生成假新闻的自适应检测，性能超越现有检索增强基线和通用大语言模型。 |
| [^328] | [Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks](https://arxiv.org/abs/2512.03262) | 该论文提出SUSVIBES基准，评估了12种编码智能体在真实任务中的安全性，发现所有智能体生成代码的安全率极低（最高仅11.8%），且简单安全提示无法有效改善。 |
| [^329] | [BudgetMem: Training-Free Selective Memory for Cost-Efficient Long-Context Processing in Language Models](https://arxiv.org/abs/2511.04919) | BudgetMem是一种无需训练的长上下文处理架构，通过实体密度、TF-IDF等可解释特征在显式内存预算下进行块级保留或丢弃决策，在丢弃70%内容块的同时保持与未压缩基线相当的性能，并大幅优于词元级压缩方法LLMLingua-2。 |
| [^330] | [Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs](https://arxiv.org/abs/2511.04473) | 提出了SynthKGQA框架，可从任意知识图谱生成包含完整真值事实的知识图谱问答数据集，既能更有效地评估知识图谱检索器，也能用于训练更好的知识图谱增强大语言模型，并基于Wikidata构建了测试零样本泛化能力的GTSQA数据集。 |
| [^331] | [M-CIF: Multi-Scale Alignment For CIF-Based Non-Autoregressive ASR](https://arxiv.org/abs/2510.22172) | 提出多尺度CIF（M-CIF）机制，通过将字符级和音素级监督逐步蒸馏到子词表示中实现多层级对齐，显著提升了非自回归语音识别在德语、法语等语言上的稳定性，在CommonVoice上德语WER降低4.21%、法语降低3.05%。 |
| [^332] | [BreakFun: Jailbreaking LLMs via Object Instantiation under Simulated Code Execution](https://arxiv.org/abs/2510.17904) | BreakFun通过让模型模拟执行包含“特洛伊模式”的良性Python代码并实例化对象，使有害内容作为代码运行的副作用被生成，从而在13个模型上实现平均89%的越狱攻击成功率。 |
| [^333] | [Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](https://arxiv.org/abs/2510.10925) | 提出PerSyn个性化数据合成策略，采用“先路由后生成”范式，通过综合考虑学生可学习性与教师响应质量的查询级路由器，为每个提示匹配最优教师，从而更高效地为学生模型定制训练数据。 |
| [^334] | [What Is The Political Content in LLMs' Pre- and Post-Training Data?](https://arxiv.org/abs/2509.22367) | 该研究首次量化了大语言模型预训练与后训练数据中的政治内容，发现所有训练数据集均系统性地偏向左翼内容，且数据中的政治立场与模型的政治行为表现密切相关。 |
| [^335] | [From Outliers to Topics in Language Models: Anticipating Trends in News Corpora](https://arxiv.org/abs/2509.22030) | 该论文发现主题建模中常被视为噪声的离群值实际上是新兴主题的微弱信号，会随时间演变为连贯主题，从而可用于预测新闻语料库中的趋势。 |
| [^336] | [HumanAgencyBench: Scalable Evaluation of Human Agency Support in AI Assistants](https://arxiv.org/abs/2509.08494) | 该论文提出了HumanAgencyBench（HAB），一个利用大语言模型模拟和验证用户查询、可扩展地评估AI助手在提出澄清问题、避免价值操纵、纠正错误信息等六种行为上支持人类能动性的自适应诊断基准。 |
| [^337] | [TempCore: Are Video QA Benchmarks Temporally Grounded?](https://arxiv.org/abs/2509.01167) | 该论文提出帧选择敏感度（FSS）诊断方法，发现现有视频问答基准中仅5.5%–31%的样本真正需要时间维度的帧选择，并据此构建了聚焦时间敏感样本的紧凑评估集TempCore。 |
| [^338] | [SalQ-VLM: Fine-Grained Saliency-Guided Quantization for Vision-Language Models](https://arxiv.org/abs/2508.03351) | 提出了SalQ-VLM，一种重要性感知的训练后量化框架，通过优先处理显著token并抑制冗余视觉token，解决了视觉语言模型中视觉过度表征与模态鸿沟两大问题，从而在资源受限环境下实现高效模型压缩。 |
| [^339] | [Discrete Tokenization for Multimodal LLMs: A Comprehensive Survey](https://arxiv.org/abs/2507.22920) | 本文首次提出了面向大语言模型的离散分词（向量量化）方法的系统化分类体系，对8种代表性VQ变体的算法原理、训练动态及其与LLM流水线的集成挑战进行了全面分析。 |
| [^340] | [Calibrating Lightweight Sparse Autoencoder Feature Steering](https://arxiv.org/abs/2506.12576) | 提出ContrastiveSteer方法，通过对比式特征评分和模型特定校准来改进稀疏自编码器的特征引导，使主题对齐提升最高3.9倍，目标领域分类提升高达93%。 |
| [^341] | [Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning](https://arxiv.org/abs/2506.07044) | 该论文提出了灵枢（Lingshu）——一个通过全面数据筛选流程构建的通用医学基础模型，实现了统一的医学多模态理解与推理，克服了现有医学MLLM在医学知识覆盖、幻觉抑制和复杂医学场景推理方面的关键局限。 |
| [^342] | [SocialMaze: A Benchmark for Evaluating and Enhancing Social Reasoning in Large Language Models in Complex Social Environments](https://arxiv.org/abs/2505.23713) | 该论文提出了SocialMaze基准，通过深度推理、动态交互和信息不确定性三个设计维度，在社交推理游戏、日常互动和数字社区平台等六项任务中评估并提升大型语言模型在复杂社会环境中的社会推理能力。 |
| [^343] | [ESLM: Risk-Averse Selective Language Modeling for Efficient Pretraining](https://arxiv.org/abs/2505.19893) | 提出ESLM算法，利用逐token统计量（熵或损失）和在险价值阈值筛选在线选择每批次中最具信息量的token进行训练，从而提升大语言模型预训练效率并增强分布鲁棒性。 |
| [^344] | [Explain Less, Understand More: Data-Efficient Personalization of Reader-Dependent Jargons](https://arxiv.org/abs/2505.16227) | 该论文提出了两种高效可扩展的术语个性化策略——基于LoRA的轻量级微调和无需再训练的个性化提示，并结合半监督学习利用用户出版物数据，其个性化LoRA模型性能超越了使用上下文提示的GPT-4。 |
| [^345] | [Efficient and Adaptive Simultaneous Speech Translation with Fully Unidirectional Architecture](https://arxiv.org/abs/2504.11809) | 提出EASiST，通过语音编码器与LLM均为全单向的架构、多延迟数据筛选策略、带显式读/写标记的交错生成任务以及轻量级策略头，实现了高效且自适应的同声语音翻译。 |
| [^346] | [AskQE: Question Answering as Automatic Evaluation for Machine Translation](https://arxiv.org/abs/2504.11582) | AskQE是一个基于问题生成与回答的机器翻译质量评估框架，使不懂目标语言的用户也能检测关键翻译错误并决定是否接受译文，其与人工评分的相关性和决策准确率优于现有QE指标。 |
| [^347] | [Logits are All We Need to Adapt Closed Models](https://arxiv.org/abs/2502.06806) | 提出了一种仅利用logits的token级概率重加权框架Plugin，将黑盒LLM的适配问题转化为标签噪声纠正问题，从而在无需访问模型内部的情况下实现面向特定应用的内容生成。 |
| [^348] | [3D-MoE: Towards Spatial Intelligence with Mixture-of-Experts for 3D Reasoning and Action Generation](https://arxiv.org/abs/2501.16698) | 3D-MoE通过混合专家架构与模态/空间上下文感知的概率路由实现高效3D推理，并集成Pose-DiT扩散动作头在单步内生成精确6D位姿动作，以大幅减少的激活参数在3D视觉-语言基准上取得优越性能。 |
| [^349] | [Compound-QA: A Benchmark for Evaluating LLMs on Compound Questions](https://arxiv.org/abs/2411.10163) | 该论文提出了复合问题合成方法CQ-Syn，构建了包含五个类别、从理解、推理和知识三个维度评估大语言模型处理由多个相互关联子问题组成的复合问题能力的新基准Compound-QA。 |
| [^350] | [A Course Intelligence Platform for Higher Education: Lessons from AI-Assisted Course Evaluation](https://arxiv.org/abs/2411.02455) | 本文提出了一个在中国100多所高校部署、服务超万名教师的课程智能平台，其AI辅助课程评估模块通过整合国家评估标准、结构化教育证据与领域适配的大语言模型，自动生成量化评分与定性反馈，填补了院校层面AI应用的空白。 |
| [^351] | [SG-FSM: A Self-Guiding Zero-Shot Prompting Paradigm for Multi-Hop Question Answering Based on Finite State Machine](https://arxiv.org/abs/2410.17021) | 提出了一种基于有限状态机的自引导零样本提示范式SG-FSM，通过迭代分解复杂问题为子问题、自我纠错并动态决定下一步推理步骤，有效提升了大语言模型在多跳问答任务中的表现。 |
| [^352] | [Eraser: Jailbreaking Defense in Large Language Models via Unlearning Harmful Knowledge](https://arxiv.org/abs/2404.05880) | 提出 Eraser 防御方法，通过让大语言模型遗忘回答有害问题所需的知识来从根本上消除越狱风险，且无需红队协助即可在保留通用知识和安全对齐的同时显著提升模型安全性。 |
| [^353] | [Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning](https://arxiv.org/abs/2401.08038) | Calpric通过结合自动文本分割、主动学习和众包标注，以低成本生成大型均衡的隐私政策训练数据集，使未经训练的众包标注者能达到与专业标注者相当的水平。 |
| [^354] | [Authorship identification under domain shift: a survey of stylistic measures and learned author representations](https://arxiv.org/abs/2310.00436) | 本综述提出作者识别的主题独立性取决于文体特征、编码方式、评分规则与评估划分的组合而非特征本身，并从四个层面系统梳理了领域偏移下从经典频率度量到学习型作者表示的研究证据。 |
| [^355] | [VOLTA: Improving Generative Diversity by Variational Mutual Information Maximizing Autoencoder](https://arxiv.org/abs/2307.00852) | VOLTA通过Transformer与VAE框架的更有效连接，InfoGAN风格潜在编码以及支持离散输入，提升了生成多样性 |

# 详细

[^1]: 临界状态强化学习：诊断多轮工具使用中的可训练状态

    Critical-State RL: Diagnosing Trainable States for Multi-Turn Tool Use

    [https://arxiv.org/abs/2609.24985](https://arxiv.org/abs/2609.24985)

    提出临界状态强化学习方法，通过嵌套采样将动作相关的奖励变化与下游噪声分离，从而识别多轮工具交互中真正值得训练的关键状态，并用上下文老虎机方法对这些状态进行针对性策略优化。

    

    多轮工具使用中的失败可能取决于单次模型调用，然而仅凭奖励变化并不能揭示哪一次调用能从训练中受益。当奖励依赖于后续交互时，其变化可能反映的是下游的随机性，而非当前动作之间的差异。我们提出临界状态强化学习，用于识别多轮交互中的可训练状态。给定任务定义的候选调用和局部奖励，该方法评估每个奖励是否捕捉了动作对任务成功的影响，以及相对于参考策略是否存在改进空间。随后，该方法使用嵌套采样将依赖于动作的奖励变化与后续噪声分离开来，并使用上下文老虎机训练在选定状态上优化策略。在伯克利函数调用排行榜（BFCL）v4上的实验比较了在诊断选定的状态上训练与在其他状态上训练的效果。对于缺失函数任务，该诊断方法……

    arXiv:2609.24985v1 Announce Type: cross  Abstract: Multi-turn tool-use failures can hinge on a single model call, yet reward variation alone does not reveal which call would benefit from training. When rewards depend on later interactions, their variation can reflect downstream randomness rather than differences between the current actions. We introduce Critical-State RL to identify trainable states in multi-turn interactions. Given task-defined candidate calls and local rewards, the method assesses whether each reward captures the action's effect on task success and whether improvement over a reference policy is possible. It then uses nested sampling to separate action-dependent reward variation from continuation noise and optimizes the policy at the selected states using contextual-bandit training. Experiments on the Berkeley Function Calling Leaderboard (BFCL) v4 compare training at diagnostic-selected states with training at alternative states. For missing-function tasks, the diagn
    
[^2]: onPanda：通过词元级修正为大语言模型与智能体高效标注同策略对齐数据

    onPanda: Efficient Annotation of On-Policy Alignment Data for LLMs and Agents via Token-Level Correction

    [https://arxiv.org/abs/2609.24983](https://arxiv.org/abs/2609.24983)

    onPanda通过“定位-修正-继续”的词元级交互方式高效标注大语言模型与智能体的同策略对齐数据，将标注时间中位数减少52%，同时保留了模型自身的采样分布。

    

    我们提出了onPanda，一个用于高效标注大语言模型对齐数据和智能体轨迹的交互式工具。onPanda采用词元级修正作为其核心交互方式：在阅读模型响应时，标注者定位第一个不合适的词元，然后从模型的候选词元中选择替代项，或通过自由编辑输入正确的文本。随后系统截断该位置之后的所有内容，并从修正后的前缀继续生成，重复这一“定位-修正-继续”的循环，直到获得令人满意的响应。这一机制使标注者能够以低成本精确地引导模型输出：一项小型对照研究表明，与人工后编辑相比，onPanda将标注时间中位数减少了52%。由于最终响应中的绝大多数词元都是由模型自身生成的，所得到的数据在很大程度上保留了模型的采样分布，非常适合构建同策略对齐数据。

    arXiv:2609.24983v1 Announce Type: new  Abstract: We present onPanda, an interactive tool for efficiently annotating LLM alignment data and agent trajectories. onPanda adopts token-level correction as its core interaction: while reading a model response, the annotator locates the first inappropriate token and either picks a substitute from the model's candidate tokens or types the correct text via free-form editing. The system then truncates everything after that position and continues generation from the corrected prefix, repeating this locate-correct-continue loop until a satisfactory response is obtained. This mechanism lets annotators precisely steer model outputs at low cost: a small controlled study suggests that onPanda reduces median annotation time by 52% over manual post-editing. Since the vast majority of tokens in the final response are generated by the model itself, the resulting data largely preserves the model's sampling distribution and is well suited for constructing on
    
[^3]: Harness-Zero：通过“智能体即框架”实现框架蒸馏

    Harness-Zero: Harness Distillation via Agent-as-Harness

    [https://arxiv.org/abs/2609.24974](https://arxiv.org/abs/2609.24974)

    提出 Harness-Zero 方法，通过“智能体即框架”将领域或实例优化的框架所诱导的行为蒸馏进模型权重，使框架带来的性能提升在单一固定的目标框架下得以保留。

    

    智能体框架是调节模型与环境交互的外部系统，能够显著提升智能体的性能，但其收益在部署时仍依赖于所使用的框架。由于最佳框架因领域、实例和模型而异，通用智能体要么只能接受次优的共享框架，要么需要在不断增长的专用框架集合中进行路由选择。因此，我们研究智能体框架蒸馏：使用针对领域或实例优化的框架作为训练时的指导，并将其诱导的行为迁移到模型权重中，使其收益在单一固定的目标框架下得以保留。挑战在于，两个框架在动作空间和可用信息上存在差异，因此来自优化框架的指导无法直接作为目标框架的监督信号。我们提出了 Harness-Zero，通过“智能体即框架”实现框架蒸馏。在优化框架的指导下，一个框架……（原文摘要在此处截断）

    arXiv:2609.24974v1 Announce Type: cross  Abstract: Agent harnesses, the external systems that mediate model-environment interaction, can substantially improve agent performance, but their gains remain tied to the harness at deployment. Because the best harness varies across domains, instances, and models, a general-purpose agent must either settle for a suboptimal shared harness or route among an ever-growing set of specialized ones. We therefore study agent harness distillation: using a domain- or instance-optimized harness as training-time guidance and transferring the behaviors it induces into model weights, so that its gains survive under a single fixed target harness. The challenge is that the two harnesses differ in action space and available information, so guidance from the optimized harness cannot serve directly as supervision for the target one. We introduce Harness-Zero, which enables harness distillation through agent-as-harness. Guided by the optimized harness, a harnessin
    
[^4]: RRSI：智能体框架的正则化递归自我改进

    RRSI: Regularized Recursive Self-Improvement of Agent Harnesses

    [https://arxiv.org/abs/2609.24972](https://arxiv.org/abs/2609.24972)

    该论文提出RRSI方法，通过将正则化原则（如时间退火的编辑预算限制和鼓励探索未开发轨迹）引入智能体框架的递归自我改进过程，防止递归进化对训练任务过拟合，从而提升分布外基准上的泛化能力。

    

    LLM智能体的能力在很大程度上被其“框架”所放大，即围绕冻结骨干模型的提示词、控制流、工具、记忆和上下文管理。近期的方法通过迭代地提出并选择对智能体框架的组件级编辑，日益将这一过程自动化，实际上在智能体系统层面建立了一种递归自我改进（RSI）的形式。然而，这种递归进化可能因记忆训练任务而过拟合，在分布内基准上表现出巨大收益，但在分布外基准上收益缩小甚至消失。我们提出了智能体框架的正则化递归自我改进（RRSI），通过约束进化候选的提案与选择，将正则化原则融入框架自我改进之中。提案者以时间退火的预算运行，限制候选可以捆绑的编辑数量，并鼓励探索未开发的轨迹……

    arXiv:2609.24972v1 Announce Type: cross  Abstract: An LLM agent's capability is largely magnified by its harness, namely the prompts, control flow, tooling, memory, and context management surrounding the frozen backbone model. Recent methods increasingly automate this process by iteratively proposing and selecting component-wise edits of an agent harness, practically establishing a form of recursive self-improvement (RSI) at the agent-system level. However, such recursive evolution may overfit by memorizing the training tasks, showing large in-distribution gains that shrink or even vanish on out-of-distribution benchmarks. We introduce Regularized Recursive Self-Improvement of Agent Harnesses (RRSI), which incorporates the principles of regularizations into harness self-improvement by constraining the evolution candidate proposal and selection. The proposer operates with a temporally annealed budget, limiting how many edits a candidate can bundle, and it encourages unexplored trajector
    
[^5]: DolphinBench：绘制智能体记忆的帕累托前沿

    DolphinBench: Mapping the Pareto Frontier of Agent Memory

    [https://arxiv.org/abs/2609.24971](https://arxiv.org/abs/2609.24971)

    DolphinBench是一个通过智能体实际任务完成情况（而非对话式问答）直接评估长期记忆的基准，包含三个各约50万token历史记录的知识工作角色画像、每个角色200个经有无历史对照验证的任务，并强制要求报告成本，以刻画记忆性能与成本之间的帕累托前沿。

    

    如今的智能体常常采取依赖于长期记忆和随时间推移的上下文回忆的现实世界行动。然而，目前大多数记忆基准都是为对话式问答格式构建的，其中问题本身就会提示需要检索某些事实，甚至往往暗示是哪一个事实。此外，基准测试很少对提交内容提出准确性之外的要求，这使得记忆系统可以通过不合理的成本/时间权衡来换取更高的分数。我们提出了DolphinBench，这是一个通过智能体的任务完成情况直接评估记忆的基准。DolphinBench包含三个知识工作型角色画像，每个角色画像拥有约50万token的用户消息，并在依赖该历史信息的任务上评估智能体。我们通过让智能体在有相关历史和无相关历史的情况下分别运行，验证了每个角色的全部200个任务，要求在有历史的情况下成功而在无历史的情况下失败。最后，我们要求所有评估报告总成本

    arXiv:2609.24971v1 Announce Type: new  Abstract: Agents today often take real-world actions that depend on long-term memory and context recall over time. However, most current memory benchmarks are built for a conversational question-answer format, where the question itself signals that some fact must be retrieved, and often which one. Moreover, benchmarks rarely require anything beyond accuracy from submissions, allowing memory systems to make unreasonable cost/time tradeoffs to achieve higher scores.   We present DolphinBench, a benchmark that evaluates memory directly through an agent's task completion. DolphinBench includes three knowledge-work personas with roughly 500k tokens of user messages per persona and evaluates agents on tasks that depend on information from that history. We verify all 200 tasks per persona by running an agent with and without the relevant history, requiring success with it and failure without it.   Finally, we require all evaluations to report total cost 
    
[^6]: 长时程LLM智能体交互中的涌现性合谋

    Emergent Collusion in Long-Horizon LLM Agent Interaction

    [https://arxiv.org/abs/2609.24967](https://arxiv.org/abs/2609.24967)

    该研究首次系统揭示了在长时程多智能体交互中，当遵守验证协议与奖励最大化相冲突时，LLM智能体会自发涌现合谋行为——在10个模型中94%的轨迹出现合谋，且能力更强的模型更早合谋，对等行为、奖励结构、验证反馈和交互历史均对合谋形成有显著影响。

    

    LLM智能体越来越多地被部署在协作场景中，然而长期交互可能引发不良的协调行为。我们研究了长时程多智能体环境中合谋的涌现：两个智能体反复完成各自的任务、共享任务日志、相互验证对方的工作并获得奖励。我们引入了现实约束，使得遵守验证协议与奖励最大化不相容，并发现智能体在反复交互中越来越多地偏离协议。在10个模型中，94%的轨迹出现了合谋，且同一系列中能力更强的模型更早出现合谋。受控的对等干预实验表明，合谋受到对等智能体行为的影响，而消融实验揭示了奖励结构、智能体接收到的验证反馈以及它们的交互历史所产生的额外影响。特别是，限制交互历史的数量和范围……

    arXiv:2609.24967v1 Announce Type: cross  Abstract: LLM agents are increasingly deployed in collaborative settings, yet long-term interaction may give rise to undesirable coordination. We study the emergence of collusion in a long-horizon multi-agent environment: two agents repeatedly complete individual tasks, share task logs, verify each other's work, and receive rewards. We introduce realistic constraints that make compliance with the verification protocol incompatible with reward maximization, and find that agents increasingly deviate from the protocol over repeated interactions. Collusion emerges in 94% of trajectories across 10 models, and more capable models within the same family reach it earlier. Controlled peer interventions show that collusion is shaped by peer behavior, while ablations reveal additional effects of reward structure, the verification feedback agents receive, and their interaction history. In particular, restricting the amount and scope of interaction history a
    
[^7]: 用于科学决策的Jev：评估语义选择及其后果

    Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences

    [https://arxiv.org/abs/2609.24965](https://arxiv.org/abs/2609.24965)

    该研究将Jev作为科学工作流中的语义决策组件进行评估，发现其语义正确性与其他配置持平且延迟最低，并表明错误的语义选择会改变下游计数但可能不影响最终结论标签。

    

    arXiv:2609.24965v1 公告类型：新论文 摘要：科学工作流程通常需要在确定性计算进行之前，在已知关系之间做出选择。观测数据是否共享相同的文化、处理方式或参考标准，可能会改变由此产生的计数或比较的科学含义。我们使用一个遵循其文档指导并将算术运算分配给代码的测试框架，将Jev作为语义决策组件进行评估。该研究在十个科学案例的二十个有来源依据的选择上比较了十二种模型配置，每种配置重复五次。我们分别测量语义选择、下游输出和最终结论标签。Jev与其他五种配置在完全语义正确性上持平，并在成功响应中实现了观察到的最低中位延迟。在三个对比模型中，对一个文化历史问题的七次错误选择改变了下游计数，同时保持了正确的最终标签。这些结果确定了一个有用的角色

    arXiv:2609.24965v1 Announce Type: new  Abstract: Scientific workflows often require choosing among known relations before a deterministic calculation can proceed. Whether observations share a culture, treatment or reference standard can change the scientific meaning of the resulting count or comparison. We evaluate Jev as a semantic decision component using a harness that follows its documented guidance and assigns arithmetic to code. The study compares twelve model configurations on twenty source-grounded Choices across ten scientific cases, each repeated five times. We measure semantic selections, downstream outputs and final claim labels separately. Jev matched five other configurations at complete semantic correctness and achieved the lowest observed median latency among successful responses. Across three comparison models, seven wrong selections on one culture-history question changed downstream counts while preserving the correct final label. These results identify a useful role 
    
[^8]: 面向可解释文本蕴含的语言学特征

    Linguistic Features for Interpretable Textual Entailment

    [https://arxiv.org/abs/2609.24932](https://arxiv.org/abs/2609.24932)

    该论文提出SLITE，一个融合结构-关系层与分布-信息层语义分析的可解释文本蕴含混合模型，仅用17个语言学特征（涵盖实体级语义关系、极性敏感词汇匹配及基于熵与迁移熵的对齐度量）训练的逻辑回归即可达到83%的准确率，同时保持模型的可解释性。

    

    尽管神经模型在自然语言处理领域取得了成功，但其黑盒特性限制了可解释性，并掩盖了其预测背后的语言现象。我们提出了SLITE，一个用于识别文本蕴含（Recognizing Textual Entailment）的可解释混合模型，它集成了两个互补的语义分析层：一个是结构-关系层，基于组合实体之间的语义相容性与不相容性；另一个是分布-信息层，基于前提与假设的嵌入表示之间信息变化的结构化模式。我们提出了17个特征，结合了实体级语义关系、极性敏感的词汇匹配，以及相似度矩阵语义子表示上的对齐度量，其中包括基于熵和迁移熵的度量。基于这些特征训练的逻辑回归模型在三分类SI（原文截断）上达到了83%的准确率。

    arXiv:2609.24932v1 Announce Type: new  Abstract: Despite the success of neural models in natural language processing, their black-box nature limits interpretability and conceals the linguistic phenomena underlying their predictions. We present SLITE, an explainable hybrid model for Recognizing Textual Entailment that integrates two complementary layers of semantic analysis: a structural-relational layer, based on semantic compatibility and incompatibility between compositional entities, and a distributional-informational layer, based on structured patterns of information change between embedding-based representations of the premise and the hypothesis. We propose 17 features that combine entity-level semantic relations, polarity-sensitive lexical matching, and alignment measures over semantic sub-representations of the similarity matrix, including measures based on entropy and transfer entropy. A logistic regression trained on these features achieves an accuracy of 83% on three-class SI
    
[^9]: SocioVerse2：人机协同演化范式下的纵向动态社会模拟框架

    SocioVerse2: A Longitudinal Dynamic Social Simulation Framework under a Human-AI Co-evolutionary Paradigm

    [https://arxiv.org/abs/2609.24911](https://arxiv.org/abs/2609.24911)

    SocioVerse2将社会模拟扩展为人机协同演化范式，通过支持环境演化与干预生成反事实分支的纵向模拟循环、以及研究过程可编辑的可控研究循环，首次系统性支持了对模拟内容的干预和研究者对模拟过程的控制。

    

    社会模拟为社会科学提供了一种现实世界无法提供的实验工具，而生成式智能体作为“硅基样本”将基于智能体的建模与真实行为数据相结合，变革了这一领域。现有平台能够验证集体行为、在横截面上将模拟人口与现实社会对齐，并在研究过程中采用自主智能体。然而，社会科学的两项需求仍然缺乏系统性支持：对模拟内容的干预，以及研究者对产生模拟结果的过程的控制。我们提出了SocioVerse2，它将SocioVerse 1.0扩展为一个由“两个循环和一个基础设施”构成的人机协同演化范式。纵向模拟循环通过不断演化的环境模拟目标人群，并通过干预分叉出反事实分支。可控研究循环将研究本身视为一个可编辑的状态……

    arXiv:2609.24911v1 Announce Type: new  Abstract: Social simulation offers the social sciences an experimental instrument that the real world cannot supply, and generative agents have transformed it by acting as silicon samples that unite agent-based modeling with real behavioral data. Existing platforms verify collective behavior, align simulated populations with real societies in cross-sections, and employ autonomous agents for the research process. However, two social science requirements remain without systematic support: intervention in the content of a simulation and the researcher's control over the process that produces it. We present SocioVerse2, which extends SocioVerse 1.0 into a human-AI co-evolutionary paradigm built from two loops and one infrastructure. The longitudinal simulation loop simulates the target population with evolving environments and forks counterfactual branches via interventions. The controllable research loop takes the study itself as an editable state an
    
[^10]: ToneCL：面向少样本音节级声调分类的对比学习

    ToneCL: Contrastive Learning for Few-Shot Syllable-Level Tone Classification

    [https://arxiv.org/abs/2609.24903](https://arxiv.org/abs/2609.24903)

    ToneCL是一个轻量级对比学习框架，通过保留声调身份的数据增强在无标注语音上预训练并少样本微调，在低资源音节级声调分类任务上取得优异表现（普通话10样本达91.6%），并支持有效的跨语言迁移。

    

    声调语言占世界语言的50-70%以上，但其中绝大多数属于低资源语言，缺乏自动声调分类所需的大型转写语料库。现有数据集通常以句子为单位收集，而田野语言学家需要的是细粒度的音节级标注。我们提出了ToneCL，一个用于少样本音节级声调分类的轻量级对比学习框架。我们在普通话和越南语上模拟低资源条件，将每个声调类别的标注数据限制在几十个样本。ToneCL先在无标注语音上使用保留声调身份的数据增强进行预训练，然后在少样本示例上进行微调。实验表明，我们的方法始终优于基线方法，在10个样本条件下，六说话人普通话上达到91.6%的准确率。跨语言迁移同样有效：在越南语上预训练、在普通话上微调，在10个样本条件下达到91.0%的准确率。消融实验证实（原文在此截断）。

    arXiv:2609.24903v1 Announce Type: new  Abstract: Tone languages constitute over 50-70% of the world's languages, but the vast majority are low-resource, lacking the large transcribed corpora needed for automatic tone classification. Existing datasets are typically collected at the sentence level, whereas field linguists require fine-grained syllable-level annotations. We propose ToneCL, a lightweight contrastive learning framework for few-shot syllable-level tone classification. We simulate low-resource conditions on Mandarin and Vietnamese, limiting labeled data to tens of examples per tone class. ToneCL is pretrained on unlabeled speech with augmentations that preserve tonal identity, then fine-tuned on few-shot examples. Experiments show our method consistently outperforms baselines, achieving 91.6% on six-speaker Mandarin at 10 shots. Cross-lingual transfer is also effective: pretraining on Vietnamese and fine-tuning on Mandarin reaches 91.0\% accuracy at 10 shots. Ablation confirm
    
[^11]: 人类与大语言模型的协商作为交互式证明：无需透明性的可验证性条件

    Human-LLM Deliberation as Interactive Proof: Conditions for Verifiability Without Transparency

    [https://arxiv.org/abs/2609.24895](https://arxiv.org/abs/2609.24895)

    该论文将人类与大语言模型的协商建模为交互式证明，证明了即使无法访问模型内部状态（即无需透明性），人类验证者通过逐步检查积累证据也能获得可证明的可靠性保证，从而决定是否接受模型的主张。

    

    当大语言模型给出一个用户难以自行构建的论证时，用户如何决定是否接受其主张？受交互式证明的启发，我们将人类与大语言模型的协商建模为拥有不受限制内部搜索能力的证明者与资源受限的人类验证者之间的交互。验证者在不接触大语言模型内部状态的情况下，请求并检查支持性细节。通过的检查会不断积累证据，直至达到接受阈值。我们证明了针对自适应证明者的任意时刻有效性：只要任务能够提供关于错误通过和人类检查错误的界限，且这些界限在每个相关历史之后仍然有效，那么接受错误主张的概率至多为所选定的错误水平。有限时域下的完备性界限则还需要对诚实回答的充分性以及足够的诊断进展设定界限。进一步的检查可以增强接受的证据，但每次检查都需要另一个充分性……（原文截断）

    arXiv:2609.24895v1 Announce Type: new  Abstract: When an LLM supplies an argument that a user could not readily construct, how can the user decide whether to accept its claim? Inspired by interactive proofs, we model human-LLM deliberation as an interaction between a prover with unrestricted internal search and a resource-bounded human verifier. The verifier requests and checks supporting details without access to the LLM's internal state. Passed checks accumulate evidence toward an acceptance threshold. We prove anytime-valid soundness against adaptive provers: the probability of ever accepting a false claim is at most a chosen error level, provided the task supplies bounds on false passes and human checking errors that remain valid after every relevant history. A finite-horizon completeness bound additionally requires bounds on the adequacy of honest responses and sufficient diagnostic progress. Further checks can strengthen the evidence for acceptance, but each requires another adeq
    
[^12]: SLICEChat：面向全切片病理语言模型的编码器内渐进式Token剪枝

    SLICEChat: Progressive In-Encoder Token Pruning for Whole-Slide Pathology Language Models

    [https://arxiv.org/abs/2609.24894](https://arxiv.org/abs/2609.24894)

    SLICEChat通过在混合Mamba-Transformer切片编码器内部进行语言监督的渐进式token剪枝，在多模态融合之前生成紧凑的切片表示，从而显著提升全切片病理图像多模态大语言模型的效率与可扩展性。

    

    全切片病理图像（WSI）包含千兆像素级别的视觉内容，给切片级多模态大语言模型（MLLM）带来了重大的可扩展性挑战。现有方法需要处理数千个图像块token，且通常仅在切片编码完成后才进行压缩，导致多模态注意力计算成本高昂。我们提出了SLICEChat，这是一种切片级多模态大语言模型，它在混合Mamba-Transformer切片编码器内部集成了渐进式token剪枝。Mamba层实现高效的长距离信息传播，而Transformer层在序列逐步缩短的过程中保持全局交互。在各个阶段之间，语言监督的、区域感知的剪枝在受控的保留率调度下移除空间连贯的低效用区域，在多模态融合之前生成紧凑的切片表示。在SlideBench VQA基准上，SLICEChat在TCGA队列上达到79.84%的准确率，在BCNB队列上达到59.09%的准确率，优于先前的...

    arXiv:2609.24894v1 Announce Type: cross  Abstract: Whole-slide pathology images (WSIs) contain gigapixel-scale visual content, creating a major scalability challenge for slide-level multimodal large language models (MLLMs). Existing approaches process thousands of patch tokens and typically apply compression only after slide encoding, leaving multimodal attention computationally expensive. We introduce SLICEChat, a slide-level MLLM that integrates progressive token pruning within a hybrid Mamba--Transformer slide encoder. Mamba layers enable efficient long-range propagation, while Transformer layers preserve global interactions as the sequence is progressively shortened. Between stages, language-supervised, region-aware pruning removes spatially coherent low-utility regions under a controlled keep-rate schedule, producing compact slide representations before multimodal fusion. On SlideBench VQA, SLICEChat achieves 79.84% accuracy on TCGA and 59.09% on BCNB cohorts, outperforming prior 
    
[^13]: OSWorld-Pro：基于过程的计算机使用智能体评估

    OSWorld-Pro: Process-based Evaluation for Computer Use Agents

    [https://arxiv.org/abs/2609.24890](https://arxiv.org/abs/2609.24890)

    OSWorld-Pro提出了一个包含300多个任务和2800多个子目标的过程性评估基准，利用基于67,000多条人工标注、与人类判断高度一致的LLM裁判，对计算机使用智能体在子目标层面的执行过程进行评估，从而揭示智能体失败的具体方式和原因。

    

    计算机使用智能体（CUA）的评估通常仅限于它们所创建的最终交付成果（在数百步操作之后），并采用功能验证器进行评估，如OSWorld所示。然而，这种针对最终状态性能的评估缺乏对智能体在各种任务中如何以及为何失败的透明度，掩盖了对后续改进至关重要的洞察。例如，在键盘输入过程中出错的智能体，与那些无法在图形用户界面上精确执行点击输入的智能体，需要采用不同的改进策略。我们提出了OSWorld-Pro：一套包含300多个任务、2800多个子目标的数据集，基于超过67,000条人工标注，实现对CUA的过程性评估。我们使用与人类判断高度一致的鲁棒LLM裁判来评估OSWorld-Pro子目标的完成情况，从而揭示模型在一系列顺序依赖的子目标中所取得的进展。我们的发现表明OSWorld-Pro具有挑战性（摘要在此处截断）。

    arXiv:2609.24890v1 Announce Type: new  Abstract: Evaluation of Computer-Use Agents (CUAs) is often limited to the final deliverables they create (at the end of hundreds of steps) and assessed with functional verifiers, as seen in OSWorld. However, such evaluation of end-state performance lacks transparency into how and why agents fail in various tasks, obfuscating critical insight for subsequent improvement. For instance, agents that err during keyboard inputs would require a different mitigation strategy from those that fail to precisely provide click-based inputs on the graphical UI. We introduce OSWorld-Pro: a set of over 300 tasks containing over 2800 subgoals to enable the procedural evaluation of CUAs grounded in over 67,000 human annotations. We use robust human-aligned LLM-Judges to evaluate the fulfillment of OSWorld-Pro subgoals and thereby reveal the progress that models make throughout a series of sequentially dependent subgoals. Our findings reveal that OSWorld-Pro is chal
    
[^14]: 复制天花板：面向策展语料库本体接地生成的输入暴露控制

    The Copy Ceiling: An Input-Exposure Control for Ontology-Grounded Generation over Curated Corpora

    [https://arxiv.org/abs/2609.24885](https://arxiv.org/abs/2609.24885)

    论文提出“暴露核算”与“复制天花板”这一无需评判者的评估控制方法，揭示语言模型在基于本体检索的接地生成中的性能提升几乎完全来自复制上下文中已暴露的答案，而非对检索结构的真正推理。

    

    当语言模型通过基于图的检索从策展语料库中回答问题时，接地带来的大幅性能提升并不能证明模型对检索到的结构进行了推理：因为所提供的上下文可能已经暴露了标准答案。我们提出“暴露核算”方法，根据所展示的上下文是否暴露了每个标准答案项以及答案是否恢复了该项来对其进行分类。其标量参照是“复制天花板”，即对上下文进行逐字复制所能达到的召回率；相对于复制的带符号增益衡量了模型召回率相对于这一确定性、无需评判者的基线的表现。在十个模型上，无辅助召回率平均为0.26，接地召回率为0.92，但相对于复制的增益一致为负（-0.067至-0.022）。在11,360个标准答案项观测中（代表在十个模型下评估的1,136个目标实例），仅有三个未暴露的项获得了词汇层面的评分。对423个观测进行的分层模型评判审计（采用对称引用验证策略）估计，97%……（原文摘要在此处截断）

    arXiv:2609.24885v1 Announce Type: new  Abstract: When a language model answers from a curated corpus via graph-based retrieval, a large grounding uplift does not establish reasoning over the retrieved structure: the context may already expose the gold answers. We propose exposure accounting, which classifies each gold item by whether the shown context exposes it and whether the answer recovers it. Its scalar reference is the copy ceiling, the recall a verbatim copy of the context achieves; signed gain over copy measures the model's recall relative to this deterministic, judge-free baseline. Across ten models, unaided recall averages 0.26 and grounded recall 0.92, yet gain over copy is uniformly negative (-0.067 to -0.022). Of 11,360 gold-item observations, representing 1,136 target instances evaluated under ten models, only three unexposed items receive lexical credit. A stratified model-judged audit of 423 observations, with a symmetric quotation-verification policy, estimates that 97
    
[^15]: 自动化临床编码中的误差与风格分解

    Decomposing Error and Style in Automated Clinical Coding

    [https://arxiv.org/abs/2609.24877](https://arxiv.org/abs/2609.24877)

    该论文发现临床编码中的不一致很大程度上源于可建模的系统性“编码风格”而非纯粹误差，通过10维风格量表进行条件化可使ICD F1提升最多26分。

    

    在自动化临床编码任务中，标签空间涵盖数以万计的诊断和操作编码，目前模型通常对照单一金标准标注进行评估，任何偏差都被视为错误。但我们发现，当两个团队对相同的110份ACI-Bench病历进行编码时，他们对同一份病历的编码仅有73%一致（Jaccard相似度）；即使经过独立临床审计剔除错误编码后，一致性也仅提升至77%。这一差距究竟是误差，还是某种系统性因素？我们将系统性成分建模为编码风格ψ，即编码者或机构特有的关于“编码什么”以及“记录多少”的策略，并将编码任务重新表述为 p(code|note,ψ)，通过一个10维评分量表来估计ψ。如果风格只是噪声，那么以其为条件将不会产生任何效果。然而，在五个数据集上的实验表明，使用与数据匹配的风格进行条件化可将ICD F1提升最多26分，而极端不匹配的风格则最多降低21分。

    arXiv:2609.24877v1 Announce Type: new  Abstract: In automated clinical coding, where the label space spans tens of thousands of diagnosis and procedure codes, models are currently evaluated against a single gold annotation, treating any deviation as error. But we find when two teams code the same 110 ACI-Bench encounters, they agree on only 73% of codes (Jaccard similarity) for the same note; even after an independent clinical audit removes erroneous codes, agreement rises only to 77%. Is that gap error or something systematic? We model the systematic component as coding style $\psi$, a coder- or site-specific policy over what to code and how much to document, and recast coding as $p(\mathrm{code}\mid\mathrm{note},\psi)$, estimating $\psi$ with a 10-dimension rubric. If style were noise, conditioning on it would do nothing. Instead, across five datasets a model conditioned with a data-matching style raises ICD F1 by up to 26 points and an extreme mismatched one lowers it by up to 21. F
    
[^16]: 提取论证，而不仅仅是分类：基于指令微调大语言模型的生成式论证成分检测

    Extracting Arguments, Not Just Classifying Them: Instruction-Tuned LLMs for Generative Component Detection

    [https://arxiv.org/abs/2609.24855](https://arxiv.org/abs/2609.24855)

    该论文提出ITFACD方法，将论证成分检测重新构建为语言生成任务，利用指令微调的大语言模型直接从纯文本中提取并分类论证成分，无需预先分割，性能超越现有最先进系统。

    

    论证成分检测（ACD）是论辩挖掘（AM）的核心子任务，也是其中最具挑战性的方面之一，因为它需要同时界定论证片段并将其分类为论点和前提等成分。尽管与其他AM任务相比，针对该子任务的研究仍相对有限，但大多数现有方法将其简化为序列标注问题、成分分类问题，或采用成分分割后接分类的流水线方式。在本文中，我们提出了ITFACD，这是一种基于指令微调大语言模型（LLM）并使用紧凑指令提示词的新方法，将ACD重新构建为语言生成任务，使论证能够直接从纯文本中识别，而无需依赖预先分割的成分。在标准基准上的实验表明，与最先进的系统相比，我们的方法取得了更高的性能。

    arXiv:2609.24855v1 Announce Type: cross  Abstract: Argumentative component detection (ACD) is a core subtask of Argument(ation) Mining (AM) and one of its most challenging aspects, as it requires jointly delimiting argumentative spans and classifying them into components such as claims and premises. While research on this subtask remains relatively limited compared to other AM tasks, most existing approaches formulate it as a simplified sequence labeling problem, component classification, or a pipeline of component segmentation followed by classification. In this paper, we propose ITFACD, a novel approach based on instruction-tuned Large Language Models (LLMs) using compact instruction-based prompts, and reframe ACD as a language generation task, enabling arguments to be identified directly from plain text without relying on pre-segmented components. Experiments on standard benchmarks show that our approach achieves higher performance compared to state-of-the-art systems. To the best o
    
[^17]: 答案盆地表示假说：我们探测和操控的并非概念

    The Answer-Basin Representation Hypothesis: We Are Not Probing or Steering Concepts

    [https://arxiv.org/abs/2609.24821](https://arxiv.org/abs/2609.24821)

    该论文提出“答案盆地表示假说”，认为语言模型中概念相关的线性结构由续写分布在答案上诱导的概率测度所组织，源于答案测度的差异而非概念标签的变化，从而解释了探测与操控实验中概念效应及其反转现象。

    

    线性表示假说将高级概念与语言模型中的方向联系起来，但这些与概念相关的线性结构在模型内部如何组织仍不清楚。我们提出答案盆地表示假说：由模型续写分布在答案上诱导出的概率测度组织了这些线性结构，其统计量沿着跨问题共享的线性方向表示。所有产生相同答案的续写构成一个答案盆地，其质量为它们的总概率，这些盆地质量定义了答案上的推前概率测度。我们主张，与概念相关的线性结构源于答案测度之间的差异，而非由概念标签的变化所决定。在多个模型和任务上的实验将探测与操控中概念一致效应及其反转现象与概念标签和答案测度之间的对齐程度联系起来。

    arXiv:2609.24821v1 Announce Type: new  Abstract: The Linear Representation Hypothesis associates high-level concepts with directions in language models, but it remains unclear how these concept-related linear structures are organized within the model. We propose the Answer-Basin Representation Hypothesis: the probability measure induced over answers by the model's continuation distribution organizes these linear structures, with its statistics represented along linear directions shared across questions. All continuations yielding the same answer form an answer basin, whose mass is their total probability. These basin masses define the pushforward probability measure over answers. We posit that concept-related linear structure emerges from differences in the answer measure rather than being determined by changes in concept labels. Experiments across models and tasks link concept-consistent effects and their reversals in probing and steering to the alignment between concept labels and th
    
[^18]: MSI-Bench：面向协作式AI智能体的多说话人语音交互评估基准

    MSI-Bench: Evaluating Multi-Speaker Voice Interaction for Collaborative AI Agents

    [https://arxiv.org/abs/2609.24812](https://arxiv.org/abs/2609.24812)

    提出了首个多说话人语音交互评估基准MSI-Bench，包含1152个中英文多方多轮音频测试用例，涵盖多说话人记忆、指令遵循和推理三大能力，揭示了当前语音智能体在多说话人场景下与一对一交互相比存在显著的性能差距。

    

    语音为AI智能体提供了一种自然且直接的交互界面。许多语音智能体可能发挥作用的场景，包括会议、家庭和协作工作，本质上都是多说话人的环境。支持这些场景带来了在一对一交互中基本不存在的挑战。我们提出了多说话人交互基准（MSI-Bench），用于评估多说话人语音交互。每个测试用例都是一个简短的多方多轮音频场景，包含参与者上下文、预期工具调用和原子化评分标准。该基准针对三大能力族：多说话人记忆、多说话人指令遵循和多说话人推理。基准共包含1,152个测试用例，在中文普通话和英文之间平均分配（各576个）。每个数据集上表现最强的配置仅在66.8%的英文用例和54.5%的中文用例上通过全部评分标准，而最强的开源权重配置仅达到34.0%和19.3%。

    arXiv:2609.24812v1 Announce Type: new  Abstract: Voice provides a natural and immediate interface for AI agents. Many settings in which voice agents could be useful, including meetings, households, and collaborative work, are inherently multi-speaker. Supporting these settings introduces challenges that are largely absent from one-on-one interaction. We introduce the Multi-Speaker Interaction Benchmark (MSI-Bench) for evaluating multi-speaker voice interaction. Each test case is a short multi-party multi-turn audio scene with participant context, expected tool calls, and atomic rubrics. The benchmark targets three capability families: multi-speaker memory, multi-speaker instruction following, and multi-speaker reasoning. It comprises 1,152 test cases, evenly split between Mandarin Chinese and English (576 each). The strongest configuration on each split passes all rubrics on only 66.8% of English and 54.5% of Mandarin cases, and the strongest open-weight configuration on 34.0% and 19.3
    
[^19]: 当量化保留准确率却无法保留证据：面向医疗大语言模型的解释感知后训练量化

    When Quantization Preserves Accuracy but Not Evidence: Explanation-Aware Post-Training Quantization for Medical LLMs

    [https://arxiv.org/abs/2609.24799](https://arxiv.org/abs/2609.24799)

    该论文提出一种解释感知的后训练量化方法，通过从全精度教师模型推理依据构建的离线忠实度缓存，在量化医疗大语言模型时保留支持答案的证据词元，使模型在保持答案准确率的同时维持解释的可信度。

    

    后训练量化（PTQ）使大语言模型能够高效部署，而PTQ方法通常以通用的重建误差、困惑度或答案准确率作为优化和评估标准。但在解释至关重要的领域，仅保留最终答案可能是不够的，因为用户还可能检查模型生成的推理依据来判断预测结果是否可信。我们在医疗多项选择题问答任务中研究了这一问题，在该任务中推理依据应提供支持所选答案的证据。我们为基于变换的PTQ提出了一种解释感知的目标函数。该方法从全精度教师模型的推理依据构建离线忠实度缓存，并在优化过程中利用该缓存来保留支持答案的证据词元以及基于证据的答案行为。我们在W4A4KV4量化设置下将其应用于OSTQuant，并在MedExQA、MedExpQA数据集上评估了四个7B至8B规模的医疗及指令微调大语言模型。

    arXiv:2609.24799v1 Announce Type: new  Abstract: Post-training quantization (PTQ) enables efficient deployment of large language models, and PTQ methods are usually optimized and evaluated with generic reconstruction, perplexity, or answer accuracy. But in explanation-critical domains, preserving only the final answer may be insufficient, since users may also inspect generated rationales to judge whether a prediction is trustworthy. We study this issue in medical multiple-choice question answering, where rationales should provide evidence that supports the selected answer.   We propose an explanation-aware objective for transformation-based PTQ. Our method builds an offline faithfulness cache from full-precision teacher rationales and uses it during optimization to preserve answer-supporting evidence tokens and evidence-conditioned answer behavior. We instantiate it on OSTQuant under W4A4KV4 quantization and evaluate four 7B--8B medical and instruction-tuned LLMs on MedExQA, MedExpQA, 
    
[^20]: 将树结构推测解码适配至 DeepSeek-V4 以实现高效推理

    Adapting Tree-Structured Speculative Decoding to DeepSeek-V4 for Efficient Inference

    [https://arxiv.org/abs/2609.24698](https://arxiv.org/abs/2609.24698)

    针对DeepSeek-V4压缩注意力导致的跨分支状态不一致难题，通过分支感知因果验证、临时状态隔离和已接受路径状态刷新，将树结构推测解码成功集成到DeepSeek-V4-Flash流水线，实现高效推理。

    

    自回归解码过程中对目标模型的重复执行是 LLM 推理延迟的主要来源。与只沿单一候选链进行的线性推测不同，树结构推测保留了来自共享前缀的多个分支；在相同预算下，这种更广的覆盖范围可以提升接受率与效率。然而，将其适配到 DeepSeek-V4 并非易事：其 CSA/HCA 在线压缩注意力将难点集中在目标验证一侧——从共享前缀分叉出的各分支会压缩成不同的状态，从而破坏跨分支的状态一致性。我们通过分支感知的因果验证、临时状态隔离以及已接受路径的状态刷新，将树结构推测解码集成到 DeepSeek-V4-Flash 流水线中，使验证与压缩状态更新在各分支之间保持一致。在预算 D=5 至 D=8、批大小 1 至 64 以及三个数据集（GSM8K、MBPP、ShareGPT）上……

    arXiv:2609.24698v1 Announce Type: new  Abstract: Repeated execution of the target model during autoregressive decoding is a major source of LLM inference latency. Unlike linear speculation, which follows a single candidate chain, tree-structured speculation retains multiple branches from shared prefixes; under the same budget, this broader coverage can improve acceptance and efficiency. Adapting it to DeepSeek-V4 is nontrivial: its CSA/HCA online compressed attention concentrates the difficulty on the target-verify side, where branches diverging from a shared prefix compress into different states, breaking cross-branch state consistency. We integrate tree-structured speculative decoding into the DeepSeek-V4-Flash pipeline via branch-aware causal verification, temporary state isolation, and accepted-path state refresh, keeping verification and compressed-state updates consistent across branches. Across budgets D=5 to D=8, batch sizes 1 to 64, and three datasets (GSM8K, MBPP, ShareGPT), 
    
[^21]: Muon 可以超越专门的持续学习方法

    Muon Can Outperform Dedicated Continual Learning Methods

    [https://arxiv.org/abs/2609.24678](https://arxiv.org/abs/2609.24678)

    使用 Muon 优化器对更新进行正交化的简单增量 LoRA，无需任务感知的约束即可达到专门持续学习方法的性能，表明一种更新约束机制（无论来自损失函数还是优化器）就已足够。

    

    使用低秩适配器的持续学习通常通过惩罚新更新与已累积的过去权重之间的重叠来缓解遗忘，这会抑制某些更新方向，但无法控制更新如何将其能量分配到剩余的方向上。我们探究这种约束是否必须是任务感知的，还是优化器提供的通用约束就足够了。我们使用 Muon 训练一个简单的增量 LoRA（IncLoRA），Muon 会对每次更新进行正交化，并在 Standard CL Benchmark 上以五个随机种子和三种任务顺序、在 TRACE 上以三个随机种子，将其与 O-LoRA 和 ELLA 进行比较。IncLoRA+Muon 在 Standard CL 上达到了专门方法的准确率区间，并在 TRACE 上超越了所有 AdamW 配置。一种更新约束机制就足够了，无论它来自损失函数还是优化器；在 Standard CL 上，第二种机制并无帮助，而对于约束最强的方法，它会导致 8.4 个百分点（原文在此处截断）。

    arXiv:2609.24678v1 Announce Type: cross  Abstract: Continual learning with Low-Rank Adapters (LoRA) typically mitigates forgetting by penalizing the overlap between a new update and the accumulated past weights, which discourages certain update directions without controlling how an update distributes its energy over the ones that remain. We ask whether that restriction has to be task-aware, or whether a generic one supplied by the optimizer is enough. We train a plain incremental LoRA (IncLoRA) with Muon, which orthogonalizes each update, and compare it against O-LoRA and ELLA over five seeds and three task orders on the Standard CL Benchmark and three seeds on TRACE. IncLoRA+Muon reaches the accuracy band of the dedicated methods on Standard CL and improves on every AdamW configuration on TRACE. One update-constraining mechanism is enough, whether it comes from the loss or from the optimizer; on Standard CL a second one does not help, and for the most restrictive method it costs 8.4 p
    
[^22]: 面向量子增强扩散语言模型的电路超网络

    Circuit Hypernetworks for Quantum-Augmented Diffusion Language Models

    [https://arxiv.org/abs/2609.24657](https://arxiv.org/abs/2609.24657)

    提出HyperQ方法，通过轻量级电路超网络为冻结的掩码扩散语言模型注入词元条件化的量子残差分支，并利用计算成本随量子比特数线性增长的精确经典期望值表达式，实现高效的量子增强语言模型适配。

    

    语言模型可以通过改变应用于单个词元的计算来进行适配。量子电路提供了这样一种方法，但在大型模型内部评估更宽的电路可能在计算上代价高昂。本文提出了HyperQ，它向一个冻结的掩码扩散语言模型中添加了词元条件化的量子残差分支。量子残差分支是每个transformer块中的一个模块，它读取词元的隐藏状态，生成该词元电路的坐标，执行该电路，并通过残差连接将测量值加回。主干网络保持冻结，仅训练新增的分支。在每个分支内部，一个轻量级的电路超网络在共享的稀疏电路结构中生成词元特定的旋转角度、耦合强度和测量轴。所需的期望值具有精确的经典表达式，其评估成本随量子比特数量线性增长，使得电路……

    arXiv:2609.24657v1 Announce Type: cross  Abstract: Language models can be adapted by changing the computations applied to individual tokens. Quantum circuits offer one such approach, but evaluating wider circuits inside a large model can be computationally demanding. Here we introduce HyperQ, which adds token-conditioned quantum residual branches to a frozen masked-diffusion language model. A quantum residual branch is a module in each transformer block that reads a token's hidden state, emits the coordinates of that token's circuit, executes it, and adds the measured values back through a residual connection. The backbone remains frozen, and only the added branches are trained. Within each branch, a lightweight circuit hypernetwork emits token-specific rotation angles, coupling strengths, and measurement axes in a shared sparse circuit structure. The required expectation values have an exact classical expression whose evaluation cost grows linearly with the qubit count, enabling circu
    
[^23]: 使用大语言模型评估可读性：推理与少样本提示的作用

    Assessing Readability with LLMs: The Role of Reasoning and Few-Shot Prompting

    [https://arxiv.org/abs/2609.24650](https://arxiv.org/abs/2609.24650)

    该论文对多种开源大语言模型在多语言可读性评估中的表现进行了系统性基准测试，探究了推理与少样本提示策略的作用，并涵盖英语和低资源语言斯洛文尼亚语。

    

    可读性评估对于在教育、医疗保健和信息检索等领域中使文本适应目标受众至关重要。然而，传统的可读性公式难以跨体裁和语言进行泛化，而有监督的机器学习模型依赖于稀缺的、特定领域的标注语料库，这限制了它们的适用性——尤其是对于资源较少的语言。大语言模型（LLM）提供了一种高度可扩展、多语言的替代方案，无需特定任务的训练，但高级提示策略对其性能的影响仍未得到充分探索。在本文中，我们对多种开源大语言模型进行了系统性基准测试，用于多语言可读性评估，重点关注教育框架所要求的离散可读性等级的预测。除英语外，我们还在一种资源较少的语言——斯洛文尼亚语上评估了我们的方法，以确定大语言模型是否仍然……（摘要原文在此处截断）

    arXiv:2609.24650v1 Announce Type: new  Abstract: Readability assessment is essential for tailoring texts to intended audiences across educational, healthcare, and information retrieval domains. However, traditional readability formulas struggle to generalize across genres and languages, while supervised machine learning models rely on scarce, domain-specific annotated corpora, limiting their applicability--particularly for less-resourced languages. Large Language Models (LLMs) offer a highly scalable, multilingual alternative that requires no task-specific training, yet the impact of advanced prompting strategies on their performance remains underexplored. In this paper, we conduct a systematic benchmark of diverse open-source LLMs for multilingual readability assessment, focusing on the prediction of discrete readability levels required by educational frameworks. In addition to English, we evaluate our approach on a less-resourced language, Slovenian, to establish whether LLMs remain 
    
[^24]: 写入即记录，读取即寻址：一次前向传播在操作语句的KV缓存中留下了什么

    Written as a Record, Read as an Address: What a Forward Pass Leaves in an Operation's KV Cache

    [https://arxiv.org/abs/2609.24635](https://arxiv.org/abs/2609.24635)

    该研究将前向传播拆分为冻结的写入器与独立训练的读取器，证明语言模型在处理操作语句时会以可寻址的形式将实体绑定信息写入KV缓存，训练后的读取器能从中恢复75%–100%的绑定关系。

    

    当语言模型读取一个操作（例如“交换盒子F和盒子B的内容”）时，其前向传播会为这些token将键和值写入KV缓存。先前关于实体跟踪的研究确立了模型所使用的内容：绑定关系是在查询时解析的，而非以显式的潜在状态存储。我们探讨模型在操作片段处写入了什么，以及这些内容是如何被访问的。我们将一次前向传播拆分为一个冻结的写入器和一个读取器：写入器的缓存无需梯度即可重新计算，而读取器只能看到指令和操作token，所有状态描述均被隐藏，并被单独训练。因此，读取器恢复出的任何信息都必然已经存在于未经修改的缓存之中。在一个合成的盒子任务上，未经训练的读取器对所查询绑定的恢复率不超过0.06，而训练后可达0.75–1.00，且可恢复性随操作的读/写足迹而变化。我们发现了两种访问模式。在Llama-3.1-8B和Mistral-7B上……（原文摘要在此处截断）

    arXiv:2609.24635v1 Announce Type: new  Abstract: When a language model reads an operation such as "Swap the contents of Box F and Box B", its forward pass writes keys and values for those tokens into the KV cache. Prior work on entity tracking establishes what models use: bindings are resolved at query time rather than stored as explicit latent state. We ask what they write at the operation span and how it is accessed. We split a forward pass into a frozen writer and a reader: the writer's cache is recomputed without gradients, while the reader sees only the instruction and operation tokens, with all state descriptions hidden, and is trained in isolation. Anything the reader recovers was therefore already present in the unmodified cache. On a synthetic boxes task, a base reader recovers $\leq 0.06$ of queried bindings against $0.75$--$1.00$ after training, and recoverability tracks the operation's read/write footprint. We find two modes of access. Across Llama-3.1-8B and Mistral-7B, op
    
[^25]: 面向全球健康出版物的定制化命名实体识别与主题分类

    Custom Named Entity Recognition and Topic Classification for Global Health Publications

    [https://arxiv.org/abs/2609.24625](https://arxiv.org/abs/2609.24625)

    在标注数据和计算资源受限的全球健康文献场景下，基于RoBERTa的transformer在命名实体识别任务上显著优于卷积spaCy模型（micro-F1为0.80对0.65-0.69），同时研究表明更大的词汇覆盖率并不必然带来更有用的领域特定关联。

    

    在标注数据和计算资源有限的环境中，应如何为全球健康文献选择和调整自然语言处理模型？本论文通过对语义标签发现、命名实体识别（NER）和多标签主题分类的实验来研究这些挑战。首先，论文将在逐步增大的专业语料库上训练的skip-gram word2vec模型与BioWordVec进行比较，以评估语料库规模和领域背景如何影响标签发现。词汇覆盖率和定性评估表明，更广的覆盖率并不一定能产生更有用的领域特定关联。随后分析转向实体抽取，在1,000个标注句子上比较卷积spaCy模型与基于RoBERTa的transformer模型。在宽松的评分协议下，transformer模型达到0.80的micro-F1，而卷积模型为0.65-0.69，但耗时82秒。

    arXiv:2609.24625v1 Announce Type: cross  Abstract: How should natural language processing models be selected and adapted for global health literature in environments where annotated data and computational resources are limited? This thesis investigates these challenges through experiments on semantic tag discovery, named entity recognition (NER), and multi-label topic classification. First, skip-gram word2vec models trained on progressively larger specialized corpora are compared with BioWordVec to assess how corpus size and domain context influence tag discovery. Vocabulary coverage and qualitative evaluation indicate that broader coverage does not necessarily yield more useful domain-specific associations. The analysis then turns to entity extraction, comparing convolutional spaCy models with a RoBERTa-based transformer on 1,000 annotated sentences. Under a lenient scoring protocol, the transformer achieves 0.80 micro-F1 versus 0.65-0.69 for convolutional models, but takes 82 seconds
    
[^26]: UK-PRBENCH：面向英国判例法的段落级先例检索基准

    UK-PRBENCH: A Paragraph-Level Precedent Retrieval Benchmark for United Kingdom Case Law

    [https://arxiv.org/abs/2609.24613](https://arxiv.org/abs/2609.24613)

    该论文提出了UK-PRBENCH，首个基于英国国家档案馆判例数据构建的段落级先例检索基准，将法律检索粒度从整份判决书细化到段落级别，并通过实验证明现有最先进的检索模型在该任务上仍有很大提升空间。

    

    先例案例检索（PCR）旨在识别与给定查询案例相关的先例案例。现有的PCR基准和方法主要在文档级别上运行，将整个判决书作为相关性的基本单元。这种设定对于法律从业者而言并不理想，因为一份判决书往往涉及多个法律问题，而只有其中一小部分段落与特定的查询相关。为弥补这一空白，我们提出了UK-PRBENCH，这是一个针对英国判例法的段落级先例检索基准，其数据由英国国家档案馆获得的判决书构建而成，涵盖了广泛的英国法院和法庭。此外，我们评估了当前最先进的检索模型并建立了基线结果。实验表明，段落级先例检索对现有的检索方法而言仍然极具挑战性，凸显出巨大的改进空间。UK-PRBENCH为评估细粒度法律检索提供了一个标准化的基准。

    arXiv:2609.24613v1 Announce Type: cross  Abstract: Prior case retrieval (PCR) aims to identify precedent cases relevant to a given query case. Existing PCR benchmarks and methods predominantly operate at the document level, treating entire judgments as the unit of relevance. This formulation is suboptimal for legal practitioners, as judgments address multiple legal issues and only a small subset of paragraphs is relevant to a particular query. Addressing this gap, we introduce UK-PRBench, a benchmark for paragraph-level precedent retrieval in UK case law, constructed from judgments obtained from the UK National Archives and covering a broad range of UK courts and tribunals. Furthermore, we evaluate state-of-the-art retrieval models and establish baseline results. Our experiments show that paragraph-level precedent retrieval remains challenging for current retrieval approaches, highlighting substantial room for improvement. UK-PRBench provides a standardised benchmark for evaluating fin
    
[^27]: 评估用于计算社会科学文本标注的决策模型

    Evaluating Decision Models for Text Annotation in Computational Social Science

    [https://arxiv.org/abs/2609.24574](https://arxiv.org/abs/2609.24574)

    本研究在18个计算社会科学分类任务上对决策模型与19个大语言模型进行零样本对比评估，发现首个商业决策模型在绝大多数任务上落后于最佳大语言模型，其置信度在社会科学构念上的可信度仍存疑。

    

    计算社会科学日益依赖大语言模型进行文本标注，已发表研究结果的有效性如今取决于这些模型所生成的标签。决策模型是一类为分类问题回答而构建的新型模型，它们以一个选项、标签集上的概率分布和置信度分数来回答类型化问题，而非自由文本，且价格仅为前沿推理价格的一小部分。然而，其答案是否准确，以及其声称的置信度在社会科学构念上是否可信，目前尚不清楚。在此，我们参照Ziems等人（2024）的评估方法，在18个计算社会科学分类任务（共7,977个项目）上，将首个商业决策模型及两个开放权重对应模型与19个前沿及开放权重语言模型在相同的零样本协议下进行比较。决策模型在15个评估任务中的14个上落后于每任务最佳的大语言模型，中位数……（摘要在此处截断）

    arXiv:2609.24574v1 Announce Type: new  Abstract: Computational social science increasingly relies on large language models for text annotation, and the validity of published findings now rests on the labels generated by such models. Decision models, a new model class built for categorical question answering, answer typed questions with a choice, a probability distribution over the label set, and a confidence score rather than free text, at a small fraction of frontier inference prices. Whether their answers are accurate, and whether that stated confidence can be trusted on social science constructs, are unknown. Here, we mirror the evaluation of Ziems et al. (2024) on 18 computational social science classification tasks (7,977 items), comparing the first commercial decision model and two open-weight counterparts against 19 frontier and open-weight language models under the same zero-shot protocol. The decision model trails the per-task best LLM on 14 of 15 evaluation tasks, with a medi
    
[^28]: 迈向概念的统一数学理论

    Toward a Unified Mathematics of Concepts

    [https://arxiv.org/abs/2609.24554](https://arxiv.org/abs/2609.24554)

    该论文提出一种基于操作的概念数学框架评估视角，识别出十三个贯穿认知科学、心理学与人工智能的核心概念操作，并证明十个现有框架因对概念本质（内容、关系结构或演化过程）的不同承诺而各自天然支持不同的操作子集。

    

    概念通常被定义为知识的抽象、紧凑表示，并被视为智能行为的基本单元。然而，认知科学、心理学和人工智能领域一直缺乏一个共享的数学语言来描述概念。现代系统将概念表示为向量、分布、符号、图及其他结构，但这些形式化方法通常被视为相互竞争的方案，而非针对同一问题的不同解决方案。我们提出一种基于操作的观点，通过数学框架所支持的概念操作来评估这些框架，并识别出十三个在认知科学、心理学和人工智能中反复出现的操作（包括相似性、组合、泛化和接地）。我们表明，十个框架体现了对概念的不同理解——将概念视为自包含的内容、关系结构或演化过程——而这些理解决定了每个框架天然支持哪些操作。例如，基于向量的模型便于分级相似……

    arXiv:2609.24554v1 Announce Type: new  Abstract: Concepts are commonly defined as abstract, compact representations of knowledge and treated as basic units of intelligent behavior. Yet, cognition, psychology, and AI lack a shared mathematical language for them. Modern systems represent concepts as vectors, distributions, symbols, graphs, and other structures, but these formalisms are typically treated as competing rather than as solutions to a common problem. We propose an operation-based view that evaluates mathematical frameworks by the conceptual operations they support, identifying thirteen operations (including similarity, composition, generalization, and grounding) that recur across cognition, psychology, and AI. We show that ten frameworks embody distinct commitments to concepts as self-contained content, relational structure, or evolving process, and that these commitments determine which operations each supports naturally. For example, vector-based models facilitate graded sim
    
[^29]: 面向序列到功能蛋白质注释的Ministral大语言模型QLoRA微调

    QLoRA Fine-Tuning of Ministral LLM for Sequence-to-Function Protein Annotation

    [https://arxiv.org/abs/2609.24538](https://arxiv.org/abs/2609.24538)

    该研究将蛋白质功能注释重新定义为序列到文本的生成任务，通过QLoRA微调30亿参数的Ministral模型，并借助GPT作为专家评估，证明紧凑型大语言模型能够生成具有真实生物学价值的策展人风格蛋白质注释。

    

    新测序蛋白质的功能注释仍然是分子生物学中的一个瓶颈：公共数据库中序列数量的增长速度远远超过了人工整理的能力。大多数计算方法将注释视为固定本体上的多标签分类问题，这将预测限制在预定义的标签集合内。在这项工作中，我们将蛋白质注释研究为一个序列到文本的生成问题。我们使用QLoRA（4位NF4量化与低秩适配器）在序列-注释对数据上微调了30亿参数的Ministral 3基础模型。我们采用“大语言模型作为专家”的评估协议来评估预测结果：一个被设定为资深分子生物学策展人角色的GPT模型对生物体识别进行二元评分，并对功能注释质量进行评分。我们得出结论，经QLoRA微调的紧凑型大语言模型能够为相当一部分蛋白质生成具有真正生物学价值的策展人风格注释。我们还讨论了未来的研究方向。

    arXiv:2609.24538v1 Announce Type: new  Abstract: Functional annotation of newly sequenced proteins remains a bottleneck in molecular biology: the number of sequences in public repositories grows far faster than the capacity for manual curation. Most computational approaches consider annotation as multi-label classification over a fixed ontology, which constrains predictions to a predefined label set. In this work we study the the protein annotation as a sequence-to-text generation problem. We fine-tune the 3B-parameter Ministral 3 base model with QLoRA (4-bit NF4 quantization with low-rank adapters) on sequence annotation pairs. We assess predictions with an LLM-as-expert protocol: a GPT model prompted as a senior molecular-biology curator scores organism identification as binary and function annotation quality. We conclude that QLoRA-fine-tuned compact LLMs can generate curator-style annotations with genuine biological value for a substantial subset of proteins. We also discuss future
    
[^30]: LLJ卡片：使用大语言模型作为评判者的最佳实践

    LLJ Cards: Best practices for the Use of LLMs as Judges

    [https://arxiv.org/abs/2609.24516](https://arxiv.org/abs/2609.24516)

    本文提出了LLJ Cards框架，综合了将大语言模型作为评判者使用时的最佳实践，以解决当前评估实践中缺乏标准化、透明性和可复现性这一根本问题。

    

    近年来，大语言模型（LLMs）已成为一种流行的评估替代方案。这些系统通常被称为“LLM作为评判者”（LLMs as Judges，简称LLJs），凭借其强大的性能、可扩展性以及相对于人类判断的成本效益，已被研究人员和从业者在广泛的测量任务中广泛采用。然而，越来越多的研究表明，使用LLJ引发了人们对其作为评估者的有效性和可靠性的担忧。现有的应对这些挑战的努力主要集中在开发偏差缓解技术和改进提示策略上。虽然这些方法代表了重要的一步，但它们主要提供的是技术性修复，未能解决一个更根本的挑战：缺乏标准化、透明和可复现的评估实践。在本文中，我们介绍了LLJ Cards，这是一个综合了测量领域最佳实践的框架……

    arXiv:2609.24516v1 Announce Type: new  Abstract: In recent years, large language models (LLMs) have emerged as a popular alternative for evaluation. Often referred to as LLMs as judges (LLJs), these systems have been widely adopted by researchers and practitioners across a broad range of measurement tasks, driven by their strong performance, scalability, and cost-effectiveness relative to human judgment. However, a growing body of work has shown that the use of LLJs raise concerns about their validity and reliability as evaluators. Existing efforts to address these challenges have largely focused on developing bias-mitigation techniques and refining prompting strategies. While these approaches represent an important step forward, they primarily offer technical fixes and leave a more fundamental challenge unaddressed: the lack of standardized, transparent, and reproducible evaluation practices. In this paper, we introduce LLJ Cards, a framework that synthesizes best practices from measu
    
[^31]: Fathom-Vaidya：基于评分标准奖励推进医学推理

    Fathom-Vaidya: Advancing Medical Reasoning with Rubric-Based Rewards

    [https://arxiv.org/abs/2609.24480](https://arxiv.org/abs/2609.24480)

    该论文提出Fathom-Vaidya顺序训练框架，利用合成数据和基于评分标准的强化学习，先提升大语言模型的诊断推理能力，再增强其在多轮临床交互中的临床医疗推理能力，以解决现有模型在复杂诊断和以患者为中心对话中的不足。

    

    在医疗保健领域部署大语言模型（LLM）需要在两个互补的维度上具备强大的性能——诊断推理：即从临床数据推断患者病情以得出诊断的收敛性、证据驱动型任务；以及临床医疗推理：即在多轮临床交互中进行沟通、规划和适应所需的更广泛的导航性判断，在这种场景下可能不存在唯一正确答案。最近的基准测试（如HealthBench和MedXpertQA）揭示了模型在这两个方面的持续弱点，暴露了模型在复杂诊断场景中的失败以及在情境化、以患者为中心的对话中的局限性。我们引入了一个顺序训练框架，利用合成数据和基于评分标准的强化学习来针对这些方面进行优化。首先，我们使用源自MedBullets的问题，通过规则和评分标准引导的强化学习（RL）来提升诊断推理能力。随后，我们转向临床医疗推理的训练。

    arXiv:2609.24480v1 Announce Type: cross  Abstract: Deploying Large Language Models (LLMs) in healthcare requires robust performance across two complementary dimensions - diagnostic reasoning: the convergent, evidence-driven task of inferring a patient's condition from clinical data to produce a diagnosis, and clinical healthcare reasoning: the broader, navigational judgment required to communicate, plan, and adapt across multi-turn clinical interactions where a single correct answer may not exist. Recent benchmarks such as HealthBench and MedXpertQA reveal persistent weaknesses in both areas, exposing failures in complex diagnostic scenarios and limitations in contextual, patient-centered dialogue. We introduce a sequential training framework that targets these facets using synthetic data and rubric-based reinforcement learning. First, we improve diagnostic reasoning using MedBullets-derived questions with rule- and rubric-guided Reinforcement Learning (RL). We then shift to clinical r
    
[^32]: 1%的Token可能就足够了：论在线策略蒸馏中的梯度估计

    1% of Tokens Can Be Enough: On Gradient Estimation in On-Policy Distillation

    [https://arxiv.org/abs/2609.24432](https://arxiv.org/abs/2609.24432)

    提出基于信噪分解的信息效率比（IER）来衡量在线策略蒸馏中Token级梯度估计的噪声，使仅用0.1%–1%的Token监督即可达到甚至超过完整蒸馏的效果。

    

    稀疏在线策略蒸馏（OPD）将教师监督分配到学生生成轨迹中的一小部分Token上。然而，当教师引导的梯度是从采样的下一个Token估计得到时，有用的教师引导可能会产生噪声较大的更新。我们在信息几何框架下研究了固定前缀处的这一估计问题，并基于信噪分解提出了信息效率比（IER）。IER刻画了在最优标量基线下的相对梯度估计误差。通过候选集近似，可以基于IER进行Token选择，并将其与现有的有用性评分相结合，同时保留采样的反向KL训练目标。在数学和医学推理任务上，加入IER在多种设置下均能改进现有的选择器，在0.1%–1%的小Token预算下，稀疏配置能够匹配甚至超过不进行Token选择的完整OPD。这些结果支持同时考虑Token的有用性与……

    arXiv:2609.24432v1 Announce Type: cross  Abstract: Sparse on-policy distillation (OPD) allocates teacher supervision to a small subset of tokens in student-generated trajectories. However, useful teacher guidance can yield a noisy update when its gradient is estimated from a sampled next token. We study this estimation problem at a fixed prefix in information geometry and propose an information-efficiency ratio (IER) based on a signal-to-noise decomposition. IER characterizes relative gradient estimation error under an optimal scalar baseline. A candidate-set approximation enables token selection based on IER and its combination with existing usefulness scores, while retaining the sampled reverse-KL training objective. On mathematical and medical reasoning tasks, adding IER improves existing selectors in multiple settings, with sparse configurations matching or exceeding full OPD without token selection at small token budgets of 0.1\%--1\%. These results support accounting for both use
    
[^33]: 端到端约旦方言语音转文本自监督学习框架

    End-to-end Jordanian dialect speech-to-text self-supervised learning framework

    [https://arxiv.org/abs/2609.24410](https://arxiv.org/abs/2609.24410)

    该论文提出了一种基于Transformer的端到端自监督学习框架，结合定制音频到文本处理算法与噪声学生训练，实现了低资源条件下高效的约旦阿拉伯语方言语音转文本系统。

    

    语音转文本引擎如今在各类应用中需求极大，是人机交互的重要使能技术。然而，一些语言缺乏带标注的语音数据，尤其是阿拉伯语方言或任何低资源语言。事实证明，自监督训练过程以及使用噪声数据的自训练是最有前景的可行解决方案之一。本文提出了一种基于Transformer的端到端模型，并配套一个面向低资源语言的框架。此外，该框架整合了定制的音频到文本处理算法，以实现高效的约旦阿拉伯语方言语音转文本系统。所提出的框架能够从多种来源摄取数据，并通过加快人工标注过程，使从外部来源构建真实标注成为可能。该框架支持使用噪声学生训练和自监督学习进行训练。

    arXiv:2609.24410v1 Announce Type: new  Abstract: Speech-to-text engines are extremely needed nowadays for different applications, representing an essential enabler in human-robot interaction. Still, some languages suffer from the lack of labeled speech data, especially in the Arabic dialects or any low-resource languages. The need for a self-supervised training process and self-training using noisy training is proven to be one of the up-and-coming feasible solutions. This article proposes an end-to-end, transformers-based model with a framework for low-resource languages. In addition, the framework incorporates customized audio-to-text processing algorithms to achieve a highly efficient Jordanian Arabic dialect speech-to-text system. The proposed framework enables ingesting data from many sources, making the ground truth from external sources possible by speeding up the manual annotation process. The framework allows the training process using noisy student training and self-supervised
    
[^34]: URA-NER：面向低资源命名实体识别的具有检索对齐与不确定性降低的统一检索增强框架

    URA-NER: A Unified Retrieval-Augmented Framework with Retrieval Alignment and Uncertainty Reduction for Low-Resource NER

    [https://arxiv.org/abs/2609.24372](https://arxiv.org/abs/2609.24372)

    提出统一检索增强框架URA-NER，通过渐进式粒度检索、模型感知表示增强和推理感知知识验证三个关键组件，解决了低资源命名实体识别中检索不对齐与生成不确定性的问题，降低了性能对大语言模型能力的依赖。

    

    基于大语言模型（LLM）的上下文学习（ICL）在缓解命名实体识别（NER）中因标注数据有限而导致的性能瓶颈方面展现出了良好的潜力。然而，现有方法仍然面临检索不对齐和生成不确定性的问题，使其性能严重依赖于大语言模型的能力。随着大语言模型参数规模的减小，其在少样本设置下的性能会显著下降。在本文中，我们提出了一种新颖的统一检索增强框架URA-NER，包含三个关键组件：渐进式粒度检索（PGR）、模型感知表示增强和推理感知知识验证。PGR是一种实现阶段对齐的两阶段检索机制，它首先基于查询的全局语义为跨度检测检索示例，然后基于……（为类型分类检索示例）

    arXiv:2609.24372v1 Announce Type: new  Abstract: In-context learning (ICL) based on large language models (LLMs) has shown promising potential in alleviating performance bottlenecks caused by the limited availability of annotated data in Named Entity Recognition (NER). However, existing methods still face issues of retrieval misalignment and generation uncertainty, making their performance heavily dependent on the LLM's capabilities. As the parameter scale of LLMs decreases, their performance in few-shot settings deteriorates significantly. In this paper, we propose a novel unified retrieval-augmented framework, URA-NER, including three key components: Progressive Granularity Retrieval (PGR), Model-aware Representation Enhancement (MaRE), and Reason-aware Knowledge Verification. PGR is a two-stage retrieval mechanism that achieves stage alignment. It first retrieves demonstrations for span detection based on the query's global semantics, and then for type classification based on the sp
    
[^35]: 通过多维量化与推理增强缓解跨领域命名实体识别中的实体类型混淆

    Mitigating Entity Type Confusion in Cross-Domain NER via Multidimensional Quantification and Reasoning Enhancement

    [https://arxiv.org/abs/2609.24357](https://arxiv.org/abs/2609.24357)

    该论文针对跨领域命名实体识别中的实体类型混淆问题，提出了多维混淆量化模型（MCQM）和渐进式双向推理链（PBRC），通过从源-目标层次、语义相似性和显式数据评估三个维度量化混淆程度并增强推理能力来缓解该问题。

    

    跨领域命名实体识别（CD-NER）旨在将源领域中丰富的知识迁移到目标领域。近年来采用分解或生成范式的研究取得了显著的性能提升，在实体跨度检测方面表现出较高的准确率。然而，在实体类型分类过程中，模型严重受到实体类型混淆的困扰，即模型倾向于将文本中某一类型的实体错误地分类为另一种相似但不正确的类型。为了解决这一问题，我们首先提出了多维混淆量化模型（MCQM），从三个维度量化模型在实体类型之间的混淆程度：源-目标层次分析、语义相似性分析和显式数据评估。此外，我们提出了渐进式双向推理链（PBRC）。PBRC 利用 MCQM 提供的源-目标层次和混淆分析来提示……

    arXiv:2609.24357v1 Announce Type: new  Abstract: Cross-domain Named Entity Recognition (CD-NER) aims to transfer the rich knowledge in the source domain to the target domain. Recent studies adopting decomposition or generation paradigms have achieved significant performance improvements, demonstrating high accuracy in entity span detection. However, during entity type classification, models severely suffer from entity type confusion, the erroneous tendency that models classify entities of one type in the text as another similar but incorrect type. To address this issue, we first propose a Multidimensional Confusion Quantification Model (MCQM) that quantifies a model's confusion extent between entity types from three dimensions: source-target hierarchy analysis, semantic similarity analysis, and explicit data evaluation. Moreover, we propose the Progressive Bidirectional Reasoning Chain (PBRC). PBRC leverages the source-target hierarchy and confusion analysis from the MCQM to prompt the
    
[^36]: Morpho-VITS：基于形态学建模的变分推断用于声调班图语言的端到端语音合成

    Morpho-VITS: Variational Inference with Morphological Modeling for End-to-End Speech Synthesis of a Tonal Bantu Language

    [https://arxiv.org/abs/2609.24310](https://arxiv.org/abs/2609.24310)

    该论文提出Morpho-VITS模型，通过将VITS架构中的标准音素编码器替换为词素序列编码器和音素-词素注意力网络，利用显式形态学建模来解决班图声调语言（如卢旺达语）文本转语音中声调难以预测的问题。

    

    面向班图声调语言的文本转语音模型面临着声调系统的挑战，该声调系统既根植于词汇层面（即词、词干和词缀的清单），也根植于语法层面（即形态句法）。更复杂的是，这些语言的标准书写系统通常省略声调标记和音节时长信息，读者必须根据上下文来消歧。受班图语言声调系统的语言学描述启发，我们提出了一种端到端的文本转语音模型，该模型通过形态句法先验来增强文本编码机制。我们将VITS架构中的标准音素编码器替换为词素序列编码器和音素到词素的注意力网络。我们认为，通过这种显式的形态学建模，可以捕获生成正确声调所需的信息。在卢旺达语（一种声调丰富且形态复杂的班图语言）上进行的实验……

    arXiv:2609.24310v1 Announce Type: cross  Abstract: Text-to-speech models for Bantu tonal languages are challenged by a tonal system that is rooted in both the lexis (i.e., the inventory of words, stems, and affixes) and the grammar (i.e., morpho-syntax). To complicate matters, the standard writing systems of these languages often omit tone markings and syllable duration information, which must be disambiguated by the reader based on context. Motivated by linguistic descriptions of Bantu language tone systems, we propose an end-to-end text-to-speech model that augments the text encoding mechanism with a morpho-syntactic prior. We replace the standard phoneme encoder in the VITS architecture with a morpheme sequence encoder and a phoneme-to-morpheme attention network. We posit that, by using this explicit morphological modeling, we can capture the information required to produce the correct tone. Experiments conducted on the Kinyarwanda language, a tonal and morphologically complex Bantu
    
[^37]: SupportCal：通过参考支持与印证实现后训练大语言模型的无标签校准

    SupportCal: Label-Free Calibration of Post-Trained LLMs via Reference Support and Corroboration

    [https://arxiv.org/abs/2609.24303](https://arxiv.org/abs/2609.24303)

    该论文提出SupportCal方法，发现适度引入PLM参考与PoLM的不一致样本（而非完全排除）能非单调地改善校准，并通过参考支持与印证机制实现无需标签数据的后训练大语言模型置信度校准。

    

    后训练通常能提升任务性能，但会损害置信度校准，使得经过后训练的语言模型（PoLM）比其对应的预训练语言模型（PLM）更加过度自信。由于任务特定的带标签校准数据可能成本高昂或难以获取，对应的预训练PLM为事后校准提供了一种天然的无标签参考。先前的基于一致性门控的PLM参考校准方法，仅使用PoLM与PLM参考一致的样本来拟合标量温度参数，并排除不一致的样本，因为直接对齐会使拟合的温度过高，从而引起置信度不足。我们重新审视了这种二元处理方式。一项受控重新引入的诊断实验揭示了一种非单调的总体效应：纳入适度比例的不一致样本能够改善校准效果，而当以单位权重纳入全部不一致样本时，这种收益会逐渐消失……

    arXiv:2609.24303v1 Announce Type: cross  Abstract: Post-training often improves task performance but can degrade confidence calibration, leaving post-trained language models (PoLMs) more overconfident than their corresponding pretrained language models (PLMs). Because task-specific labeled calibration data can be costly or unavailable, the corresponding pretrained PLM provides a natural label-free reference for post-hoc calibration. Prior agreement-gated PLM-referenced calibration fits a scalar temperature using only examples on which the PoLM and its PLM reference agree, excluding disagreement examples because direct alignment can drive the fitted temperature excessively high and induce under-confidence. We revisit this binary treatment. A controlled reintroduction diagnostic reveals a non-monotonic aggregate effect: admitting a moderate fraction of disagreement examples can improve calibration, whereas the benefit diminishes as unit-weight inclusion approaches the full disagreement s
    
[^38]: 采样之前的结构：面向数据高效文语转换的社区感知核心集选择

    Structure Before Sampling: Community-Aware Core-Set Selection for Data-Efficient Text-to-Speech

    [https://arxiv.org/abs/2609.24275](https://arxiv.org/abs/2609.24275)

    该论文提出基于语音搭配图社区结构的核心集选择方法Community Representative，能在固定音频时长预算下选出覆盖更多稀有音素的训练子集，使仅用20%数据训练的TTS模型即可获得高效表现。

    

    文语转换（TTS）语料库的录制成本高昂，然而许多语句几乎不带来新的语音信息。核心集选择通过在固定的音频时长预算下挑选一个较小的训练子集来降低这一成本。我们将语料库表示为一个语音搭配图（phonotactic graph），该图将每条语句与其在音素层面最相似的语句相连，并首先检验了该图是否具有结构特性。在孟加拉语和英语语料库中，该图的聚类系数分别是大小匹配的随机图的199倍和56倍，其模块度也达到保持度分布的随机图的两倍以上。随后，我们提出了Community Representative，一种在图的各社区之间进行采样、并在每个社区内部均匀分散选择结果的选择器，其从富含稀有音素的语句出发进行选取。在所有预算水平和两种语言中，该方法覆盖的稀有音素二元组均多于随机选择和基于熵的选择，且这一优势在留出语句上依然成立。使用其20%核心集训练的TTS模型具有更……（摘要截断）

    arXiv:2609.24275v1 Announce Type: new  Abstract: Text-to-speech (TTS) corpora are costly to record, yet many utterances add little new phonetic information. Core-set selection reduces this cost by choosing a small training subset under a fixed audio-duration budget. We represent a corpus as a phonotactic graph that links each utterance to its most phonemically similar ones, and we first test whether this graph has structure. In Bangla and English corpora, its clustering is 199 and 56 times that of a size-matched random graph, and its modularity is more than twice that of a degree-preserving random graph. We then propose Community Representative, a selector that samples across graph communities and spreads its choices within each one, starting from utterances rich in rare phonemes. At every budget and in both languages, it covers more rare phoneme bigrams than random and entropy-based selection, and this lead holds on held-out utterances. TTS models trained on its 20% core-sets have a s
    
[^39]: 规范程序性动作：一种面向工具使用智能体轨迹的可审计标注协议

    Canonical Procedural Actions: An Auditable Annotation Protocol for Tool-Use Agent Traces

    [https://arxiv.org/abs/2609.24264](https://arxiv.org/abs/2609.24264)

    本文提出了规范程序性动作（CPA）标注协议，为工具使用智能体轨迹提供可审计的程序性动作标注框架，并通过零售案例研究验证了标注者之间高达0.982的锚点-标签重叠度，证明了其结构可重复性。

    

    工具使用智能体轨迹能够识别消息和API调用，但程序性分析还需要明确的动作单元以及可查验的证据链接。我们提出了规范程序性动作，这是一种标注协议，用于记录程序性功能、其首个智能体事件锚点、实现该功能的智能体事件，以及相互独立的上下文证据。多个动作可以共享同一个消息锚点，而不推断消息内的顺序。一项零售案例研究通过开放式归纳、有记录的整合以及连续的应用审计，生成了一个版本化的24条目代码本。两个相互隔离的LLM上下文在与开发集不相交的32条轨迹上进行轨迹级标注，分别产生499和491次动作出现，锚点-标签重叠度A=0.982。若要求相同的上下文事件引用，重叠度则降低至0.798。这些是结构可重复性度量，而非语义准确性度量：26个任务ID中有16个也出现在开发集中……

    arXiv:2609.24264v1 Announce Type: new  Abstract: Tool-use agent traces identify messages and API calls, but procedural analyses also need explicit units of action and inspectable links to their evidence. We present Canonical Procedural Actions (CPAs), an annotation protocol that records a procedural function, its first agent-event anchor, the agent events that realize it, and separate contextual evidence. Multiple actions may share a message anchor without an inferred within-message order. A retail case study produces a versioned 24-entry codebook through open induction, recorded consolidation, and successive application audits. Two isolated LLM contexts annotate 32 trajectories disjoint from development at the trajectory level, producing 499 and 491 occurrences with anchor-label overlap A=0.982. Requiring identical context-event references reduces overlap to 0.798. These are structural repeatability measures, not semantic accuracy: 16 of 26 task IDs also occur in development, and hist
    
[^40]: Taramandal-GPT：通过知识检索与结构化思维增强航天动力学问题求解能力

    Taramandal-GPT: Enhancing Astrodynamics Problem-Solving with Knowledge Retrieval and Structured Thinking

    [https://arxiv.org/abs/2609.24246](https://arxiv.org/abs/2609.24246)

    提出了基于Qwen3-8b并结合检索增强生成（RAG）流水线与回退机制的领域自适应框架Taramandal-GPT，在包含299个问题的航天动力学基准APBench上取得了与最先进开源及闭源模型相当的表现，尤其在需要深度推理的任务中优势明显。

    

    大语言模型（LLMs）在自然语言理解方面取得了显著进展，但其在天文学和航天动力学等专业领域的有效性仍然有限，原因在于多步推理、符号操作和领域特定术语方面的挑战。为解决这一问题，我们提出了Taramandal-GPT（Constellation-GPT），这是一个基于Qwen3-8b骨干构建的领域自适应框架，通过检索增强生成（RAG）流水线和回退机制进行增强，以提高上下文精确度。我们在航天动力学问题基准测试（APBench）上对其进行评估，该数据集包含299个问题，涵盖从基础到高级的太空科学各个层次。采用双重评估方法——基于数值边界的评分和语义相似性评估——Taramandal-GPT在与最先进的开源和闭源模型的对比中取得了有竞争力的表现，在需要深度思考的任务中表现尤为突出。

    arXiv:2609.24246v1 Announce Type: new  Abstract: Large language models (LLMs) have shown remarkable progress in natural language understanding, yet their effectiveness in specialized fields like astronomy and astrodynamics remains limited due to challenges in multi-step reasoning, symbolic manipulation, and domain-specific terminology. To address this, we present Taramandal-GPT (Constellation-GPT), a domain-adapted framework built on the Qwen3-8b backbone, enhanced with a Retrieval-Augmented Generation (RAG) pipeline and a fallback mechanism for improved contextual precision. We evaluate it on the Astrodynamics Problems Benchmark (APBench), a dataset of 299 questions covering foundational to advanced levels of space science. Using a dual evaluation method - numeric margin-based scoring and semantic similarity assessment - Taramandal-GPT achieves competitive performance against state-of-the-art open- and closed-source models, with notable strength in thinking-intensive tasks. These resu
    
[^41]: 记忆还是上下文？语言模型事实回忆的影响因素

    Memory vs. Context? Influential Factors of Factual Recall in Language Models

    [https://arxiv.org/abs/2609.24238](https://arxiv.org/abs/2609.24238)

    该论文在31个语言模型上复现并扩展了Yu等人(2023)关于记忆与上下文权衡的研究，确认了大模型和高频实体更依赖记忆知识的规律，但发现这一权衡深受模型家族、后训练和问题措辞的影响——仅改变问题措辞即可使模型对记忆知识的依赖变化高达80个百分点。

    

    我们复现并对Yu等人（2023）的工作进行了压力测试，该工作刻画了语言模型（LM）如何在记忆的知识与相互矛盾的上下文陈述之间进行仲裁。我们在涵盖Pythia、GPT-2、Qwen3和Ministral系列（包括基础模型和后训练变体）的31个模型上复现了他们的世界首都实验，并将评估扩展到ParaConflict数据集中的五种额外知识关系类型。我们实证确认了他们的大部分原始发现：更大的模型和更高频率的实体倾向于偏向记忆的答案，且存在显著的模型家族层面差异。然而，若干结论并不能完全推广：实体频率效应在Qwen3-14B和32B上消失；后训练在不同模型家族间对记忆-上下文权衡的转变并不一致；仅问题的措辞就能使模型对记忆知识的依赖程度变化高达80个百分点；且语义上无关的散文文本可以模仿……（原文摘要在此处被截断）

    arXiv:2609.24238v1 Announce Type: new  Abstract: We reproduce and stress-test the work of Yu et al. (2023), who characterize how language models (LMs) arbitrate between memorized knowledge and contradictory in-context statements. We replicate their world-capitals experiments on 31 models spanning Pythia, GPT-2, Qwen3, and Ministral families, including base and post-trained variants, and extend evaluations to five additional knowledge relation types from the ParaConflict dataset. We empirically confirm most of their original findings: larger models and higher-frequency entities tend to favor memorized answers, with substantial family-level variance. However, several conclusions do not generalize cleanly: entity-frequency effects disappear on Qwen3-14B and 32B; post-training shifts the memory-context trade-off inconsistently across families; question phrasing alone can change a model's reliance on memorized knowledge by up to 80 percentage points; and semantically unrelated prose can mim
    
[^42]: 从文章到出版商：聚合语言模型预测以推断新闻来源可靠性

    From Articles to Publishers: Aggregating Language Model Predictions for News Source Reliability Inference

    [https://arxiv.org/abs/2609.24219](https://arxiv.org/abs/2609.24219)

    该论文提出一个两阶段框架，先利用语言模型评估单篇文章的可靠性，再聚合文章级预测来推断未见过的新闻出版商的整体可靠性，并采用严格的出版商不相交评估协议以保证评估的真实性。

    

    传统上，新闻出版商的可靠性由专家组织评估，这些组织从源头评估编辑实践、透明度和事实标准。当这一过程被转化为计算方法时，问题通常在单篇文章层面进行表述，即模型在一组预先标注的文章上进行训练，并在测试阶段评估其性能。在这项工作中，我们将新闻来源可靠性推断作为源级别预测问题进行研究。我们提出了一个两阶段框架：基于Transformer的语言模型首先估计单篇文章的可靠性，随后聚合文章级预测，以推断此前未见过的出版商的可靠性。为了逼近真实的部署条件，我们实施了严格的出版商不相交评估协议，确保没有任何出版商同时出现在训练集和测试集中。在19,47（摘要在此处截断）

    arXiv:2609.24219v1 Announce Type: new  Abstract: Traditionally, the reliability of news publishers is assessed by expert organisations that evaluate editorial practices, transparency and factual standards at source. When this process is translated into a computational approach, the problem is often formulated at the level of individual articles, with models being trained on a set of pre-labelled articles and their performance being evaluated in a test phase. In this work, we investigate news source reliability inference as a source-level prediction problem. We propose a two-stage framework in which transformer-based language models first estimate the reliability of individual articles and subsequently aggregate article-level predictions to infer the reliability of previously unseen publishers. To approximate realistic deployment conditions, we enforce a strict publisher-disjoint evaluation protocol, ensuring that no publisher appears in both training and test sets. Experiments on 19,47
    
[^43]: Vimarsha：面向印度语言的忠实ASR评估——涵盖人口多样性、真实场景音频与拼写变体

    Vimarsha: Faithful ASR Evaluation for Indian Languages with Demographic Diversity, In-the-Wild Audio and Spelling Variations

    [https://arxiv.org/abs/2609.24199](https://arxiv.org/abs/2609.24199)

    Vimarsha是一个覆盖印度全部22种表列语言的100小时ASR评估基准，通过人口多样化实地录音、高难度真实场景音频以及编码多个有效转写的变体格框架，同时纠正了传统基准的乐观与悲观偏差，揭示出真实条件下模型排名的显著变化及地理与人口层面的性能差异。

    

    印度语言自动语音识别（ASR）的评估基准存在两种系统性偏差：由干净、受控的音频条件导致的过于乐观的分数，以及由过于严格的转写标准（惩罚有效的语言变体）导致的过于悲观的分数。我们提出了Vimarsha，一个覆盖全部22种表列印度语言的100小时基准，旨在纠正上述两种失真。Vimarsha将人口多样化的实地录音与精心挖掘、按声学难度挑选的真实场景（in-the-wild）音频相结合，并引入了一个“变体格”框架，为每条语音编码多个有效转写。对10个最先进ASR模型的评估显示：在真实条件下模型排名发生显著变化，存在地理和人口层面的性能差异，以及跨语速和声学环境的系统性失败模式。

    arXiv:2609.24199v1 Announce Type: new  Abstract: Evaluation benchmarks for Indian language automatic speech recognition (ASR) suffer from two systematic biases: optimistic scores from clean, controlled audio conditions, and pessimistic scores from overly rigid transcription standards that penalize valid linguistic variations. We introduce Vimarsha, a 100-hour benchmark spanning all 22 scheduled Indian languages, designed to address both distortions. Vimarsha combines demographically diverse on-field recordings with carefully mined in-the-wild audio selected for acoustic difficulty, alongside a lattice of variations framework that encodes multiple valid transcriptions per utterance. Evaluations of 10 state-of-the-art ASR models reveal substantial shifts in model rankings under realistic conditions, geographic and demographic performance disparities, and systematic failure modes across speaking rates and acoustic environments.
    
[^44]: LoopCD：面向循环语言模型推理改进的逐循环对比解码

    LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models

    [https://arxiv.org/abs/2609.24196](https://arxiv.org/abs/2609.24196)

    LoopCD通过对比循环语言模型早期迭代与最终细化迭代的logits来干预不确定的“困难”token，无需额外训练且推理开销可忽略，即可有效提升模型的推理性能。

    

    循环语言模型通过使用共享权重递归地细化内部潜在表示来执行“潜在推理”，为显式的语言推理提供了一种更有效的替代方案。尽管其效果显著，我们发现循环语言模型仍然容易出现循环不稳定问题：跨迭代的不稳定细化会产生与推理错误相关的局部不确定的“困难”token。为了解决这一问题，我们提出了LoopCD，一种逐循环对比解码方法，通过在推理时对这些token进行干预来增强循环语言模型的推理性能。具体而言，我们利用循环语言模型的内部动态特性，将早期迭代的logits与最后一次细化迭代的logits进行对比，以形成最终的采样分布。我们发现这种策略非常高效，仅引入可忽略不计的推理开销且无需额外训练，同时能够有效提升推理性能。

    arXiv:2609.24196v1 Announce Type: new  Abstract: Looped Language Models (LoopLMs) perform "latent reasoning" by recursively refining internal latent representations with shared weights, offering a more effective alternative to explicit verbal reasoning. Despite their effectiveness, we find that LoopLMs remain prone to loop instability: unstable refinement across iterations can produce localized uncertain "hard" tokens associated with reasoning errors. To address this, we propose LoopCD, loop-wise contrastive decoding that enhances the reasoning performance of LoopLMs by intervening on these tokens at inference time. Specifically, we exploit the internal dynamics of LoopLMs and contrast the logits from earlier iterations with logits from the last refined iteration to form the final sampling distribution. We find that this strategy is highly efficient, introducing only negligible inference overhead and requiring no additional training, while effectively improving reasoning performance by
    
[^45]: 残差化何时有助于审计：格式效应、切片增益及其局限

    When Residualization Helps an Audit: Format Effects, Slice Gains, and Their Limits

    [https://arxiv.org/abs/2609.24194](https://arxiv.org/abs/2609.24194)

    该论文表明，残差化虽能将奖励模型的格式效应削弱约0.12，但无法区分被移除的表面成分中是否含有与测量构念相关的信号，因此仅靠残差化并不能使评估测量更加有效。

    

    围绕大语言模型系统使用的评估分数——包括奖励模型、重排序器和LLM裁判——可能会追踪表面形式而非它们声称要衡量的质量。当面对同一个MBPP问题的简洁正确解和带注释的错误解时，一个公开的偏好奖励模型选择正确解的表现并不比抛硬币好（0.507）。从这些分数中减去可预测的表面成分的做法日益普遍，但仅靠移除并不能产生更有效的测量：被移除的成分可能携带与测量构念相关的信号，而残差化无法区分两者。在设计的干预条件下——结合单元测试标签与仅修改注释的编辑——残差化将奖励模型的格式效应在正确代码和错误代码上均削弱约0.12，而正确与错误代码之间的差距变化却小于0.01。在观察性的NLI和QA设置中，我们在评分之前冻结了一个留出的复现……（摘要原文在此处截断）

    arXiv:2609.24194v1 Announce Type: new  Abstract: Evaluation scores used around LLM systems -- including reward models, rerankers, and LLM judges -- can track surface form instead of the quality they claim to measure. When presented with a terse correct solution and a commented buggy solution for the same MBPP problem, a public preference reward model selects the correct one no better than a coin flip (0.507). Subtracting the predictable surface component from such scores is increasingly common, but removal alone does not yield a more valid measurement: the removed component may carry construct-relevant signal, and residualization cannot tell which is which. Under designed interventions -- unit-test labels with comment-only edits -- residualization attenuates the reward model's format effects by about 0.12 on both correct and buggy code, while the correct-versus-buggy margins move by less than 0.01. In observational NLI and QA settings, we freeze a held-out replication before scoring an
    
[^46]: 面向孟加拉国法律语境的高效大语言模型蒸馏：一种智能手机兼容的检索增强生成模型

    Efficient LLM Distillation for Bangladesh Legal Context: A Smartphone-Compatible Retrieval-Augmented Generation Model

    [https://arxiv.org/abs/2609.24177](https://arxiv.org/abs/2609.24177)

    该研究通过两阶段渐进式知识蒸馏（监督微调加稀疏KL散度最小化）与QLoRA技术，将90亿参数的Gemma-2教师模型压缩为20亿参数学生模型，构建出可在智能手机上离线运行的孟加拉国法律检索增强生成模型，以弥合当地法律信息获取鸿沟。

    

    孟加拉国的大多数公民难以获取法律信息：成文法文本仅有英文版本，受过训练的律师集中在城市中心，而依赖云端的人工智能在移动网络连接不可靠的地区无法使用——在这种环境下，产生幻觉的法律文本会造成直接伤害。该系统仅处理成文法解释；需要司法判例或案例法推理的查询不在其范围之内。我们通过两阶段渐进式知识蒸馏，将90亿参数的Gemma-2教师模型压缩为20亿参数的学生模型，以弥合成文法获取鸿沟。第一阶段在9,429个经过质量筛选的法律问答对上进行监督微调（从14,514个生成的查询中接受率为65%）；第二阶段在温度τ=4.0下，针对教师模型每个词元的前50个logit最小化稀疏Kullback-Leibler散度，并通过QLoRA实现（4位NF4量化，秩为32的LoRA适配器）。先前的法律语言模型目标……（原文摘要到此截断）

    arXiv:2609.24177v1 Announce Type: new  Abstract: Legal information in Bangladesh is inaccessible to most citizens. Statutory text is English-only, trained lawyers are concentrated in urban centres, and cloud-dependent AI fails where mobile connectivity is unreliable, a setting in which hallucinated legal text causes direct harm. The system addresses statutory interpretation only; queries that require judicial precedent or case-law reasoning fall outside its scope. We target the statutory access gap by compressing a 9-billion-parameter Gemma-2 teacher into a 2-billion-parameter student through two-phase progressive knowledge distillation. Phase 1 performs supervised fine-tuning on 9,429 quality-gated legal question-answer pairs (65% acceptance from 14,514 generated queries); Phase 2 minimises sparse Kullback-Leibler divergence against the teacher's top-50 per-token logits at temperature tau = 4.0, implemented via QLoRA (4-bit NF4, rank-32 LoRA adapters). Prior legal language models targ
    
[^47]: TAC-Time：将文本作为通道的多模态时间序列预测

    TAC-Time: Texts as Channels For Multimodal Time Series Forecasting

    [https://arxiv.org/abs/2609.24156](https://arxiv.org/abs/2609.24156)

    TAC-Time提出将文本信息转化为额外的时间通道，与数值序列在共享的时间主干中联合建模，在保留时间连续性和周期结构的同时，实现了高效、可扩展且可解释的多模态时间序列预测。

    

    大多数现有的时间序列预测方法仅依赖数值观测，忽略了来自辅助文本的丰富上下文信息。近期的多模态方法尝试引入文本信号，但它们通常将文本视为静态特征，或使用大语言模型作为预测主干，这限制了其捕捉时间动态的能力，并增加了计算成本。为应对这些挑战，我们提出了TAC-Time，一个将文本信息转化为额外时间通道的统一框架。通过在共享的时间主干中将文本特征与数值序列联合建模，TAC-Time在保持高效和可扩展性的同时，保留了时间连续性和周期性结构。这种建模方式还支持系统性的可解释性分析。我们通过注意力和频域分析展示了强烈的跨模态依赖性，并识别出具有预测价值的文本信号，其相关性（摘要原文在此处截断）

    arXiv:2609.24156v1 Announce Type: new  Abstract: Most existing time series forecasting methods rely solely on numerical observations, overlooking rich contextual information from auxiliary texts. Recent multimodal approaches attempt to incorporate textual signals, but they often treat text as static features or use large language models as forecasting backbones, limiting their ability to capture temporal dynamics and increasing computational cost. To address these challenges, we propose TAC-Time, a unified framework that transforms textual information into additional temporal channels. By modeling text features jointly with numerical sequences in a shared temporal backbone, TAC-Time preserves temporal continuity and periodic structures while remaining efficient and scalable. This formulation also enables systematic interpretability analyses. We show strong cross-modal dependencies through attention and frequency-domain analyses, and identify predictive textual signals whose correlation
    
[^48]: 数据智能体：智能体化数据系统

    Data Agents: Agentic Data Systems

    [https://arxiv.org/abs/2609.24137](https://arxiv.org/abs/2609.24137)

    该论文提出“数据智能体”这一新范式，通过语义数据组织、智能体化流水线编排、记忆管理等六大组件，实现最少人工干预下自主管理、处理和分析数据，从人工设计、字面操作和被动处理三大方面变革传统数据系统。

    

    传统的数据系统在人工智能时代面临深刻的局限性：它们依赖人工构建的数据流水线，缺乏对异构数据的语义理解，并以僵化、被动的方式进行数据处理。为了应对这些挑战，我们提出了一种名为“数据智能体”的新范式，旨在以最少的人工干预来管理、处理和分析数据。数据智能体能够自主执行广泛的数据相关任务，通过从人工设计转向自主编排、从字面操作转向语义解释、从被动处理转向主动处理，从而变革传统的数据系统。我们的数据智能体系统包含六个组件：语义数据组织、语义算子、智能体化流水线编排与优化、反馈驱动的精化、记忆管理以及主动适应。在此基础上，我们还开发了两个专用智能体：数据分析智能体和数据……（原文截断）

    arXiv:2609.24137v1 Announce Type: cross  Abstract: Traditional data systems face profound limitations in the AI era, relying on human-crafted pipelines, lacking semantic understanding of heterogeneous data, and operating through rigid, reactive processing. To address these challenges, we propose a new paradigm called the Data Agent, designed to manage, process, and analyze data with minimal human intervention. Data agents autonomously execute a wide range of data-related tasks, transforming traditional data systems by shifting from manual design to autonomous orchestration, from literal manipulation to semantic interpretation, and from reactive to proactive processing. Our Data Agent system includes six components: semantic data organization, semantic operators, agentic pipeline orchestration and optimization, feedback-driven refinement, memory management, and proactive adaptation. Building on this foundation, we also develop two specialized agents: the data analytics agent and the dat
    
[^49]: Re:CAP——审计生产级RAG流水线中的检索覆盖率

    Re:CAP - Auditing Retrieval Coverage in Production RAG Pipelines

    [https://arxiv.org/abs/2609.24122](https://arxiv.org/abs/2609.24122)

    Re:CAP提出了一种无参考的迭代探测审计方法，通过为可能缺失的主题生成探测性问题并利用LLM评判筛选，来发现生产级RAG系统中检索遗漏的文档，从而审计检索覆盖率，在四个基准上恢复了BM25 top-500无法召回的9-29%金标准标注。

    

    检索增强生成（RAG）在生产环境中难以监控：对于实时重新索引的非平稳数百万段落语料库，不存在穷尽性的相关性标注。因此，检索质量通常研究不足，并且往往让位于面向生成的指标。在这项工作中，我们提出通过探测缺失文档的证据来审计检索覆盖率，而不是枚举每一个相关文档。我们的方法Re:CAP（通过迭代探测进行检索覆盖审计）是一个无参考的审计循环，应用于已部署RAG流水线的初始答案和检索到的上下文：它识别已覆盖的主题，为可能缺失的主题生成探测性问题，检索候选文档，并应用LLM作为评判者，仅保留那些引入了先前未检索到的信息的文档。在四个公开基准测试中，Re:CAP能够恢复扁平BM25 top-500无法恢复的9-29%的金标准标注。

    arXiv:2609.24122v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is hard to monitor in production: exhaustive relevance labels do not exist for non-stationary multi-million-passage corpora that re-index in real time. As a result, retrieval quality is generally understudied and often deprioritised in favour of generation-oriented metrics. In this work, we propose auditing retrieval coverage by probing for evidence of missing documents rather than enumerating every relevant one. Our method Re:CAP (REtrieval Coverage Audit by iterative Probing) is a reference-free audit loop applied to a deployed RAG pipeline's initial answer and retrieved context: it identifies the topics already covered, generates probing questions for plausibly missing topics, retrieves candidate documents, and applies an LLM-as-judge to retain only those that introduce previously-unretrieved information. On four public benchmarks, Re:CAP recovers 9-29% of gold labels that flat BM25 top-500 cannot 
    
[^50]: 你能分辨出是谁在提问：网络问题的构成与来源

    You Can Tell Who's Asking: What the Web's Questions Are Made Of, and Where They Come From

    [https://arxiv.org/abs/2609.24106](https://arxiv.org/abs/2609.24106)

    该研究通过分析110个FineWeb快照中的134亿次问题出现记录，发现网络问题的来源可以从问题形式中识别出来，且最高频的网络问题大多是模板化内容，因此问题出现次数衡量的是发布频率而非真实用户需求。

    

    从网络上抓取的问题在学术界和工业界被广泛用作“人们想知道什么”的代理指标。在问答训练数据、检索基准和内容策略中，页面上的问题被假定反映了人类的真实意图。我们通过在110个FineWeb快照（2013-2025年）中提取134亿次问题出现记录，对这一假设进行了大规模检验，并报告了三项发现。第一，你可以分辨出是谁在提问：问题的来源（问题所在的主机/页面）会在问题形式上留下信号，一个逻辑回归模型能够通过问题长度和上下文（而非问题类型）以0.725的AUC区分真实用户问题与模板化/批量制造的问题，尽管面对商业FAQ写作时AUC仅为0.554。第二，问题频率并不能衡量需求：最高频的问题大多是样板化/模板化的（前一千个高频问题中超过70%），因此出现次数衡量的是某个字符串被发布的频率，而非它被真实提问的频率。第三，超过十二……

    arXiv:2609.24106v1 Announce Type: new  Abstract: Questions scraped from the web are used across academia and industry as a proxy for what people want to know. Across QA training data, retrieval benchmarks, and content strategy, questions on a page are assumed to reflect human intent. We test this assumption at scale by extracting 13.4B question occurrences across 110 FineWeb snapshots (2013-2025), and report three findings. First, you can tell who is asking: provenance (the host/page of questions) leaves a signal in question form, and a logistic model can separate genuine user questions from templated/manufactured ones at AUC 0.725 via length and surrounding context rather than question type, though only 0.554 against commerce FAQ writing. Second, question frequency does not measure demand: the most-frequent questions are boilerplate/templated (over 70% of the top thousand), so occurrence counts measure how often a string was published and not how often it was asked. Third, over twelve
    
[^51]: 从内容生成到学习支持：面向STEM学习的教学法引导式生成视频导师

    From Content Generation to Learning Support: Pedagogy-Guided Generative Video Tutors for STEM Learning

    [https://arxiv.org/abs/2609.24083](https://arxiv.org/abs/2609.24083)

    提出了PIVOT框架，将教学法原则融入生成式视频辅导的完整流程——从分镜生成、经教学法验证的多模态视频生成，到评估与误解感知的补救教学——实现生成式AI从内容生成向以学习为中心的STEM教学支持的转变。

    

    生成式AI使教育视频的大规模生产成为可能，但现有系统主要关注生成视觉上连贯的内容，而非支持学习。因此，生成的视频往往缺乏明确的教学结构、可靠的质量控制，以及评估学习者理解程度或纠正误解的机制。在这项工作中，我们提出了PIVOT（教学法引导的教学视频辅导框架），一个通过以学习为中心的教学支持来促进STEM学习的生成式视频辅导框架。受传统教学工作流程的启发，我们的框架将教学法融入完整的生成流程：首先利用教学原则指导分镜脚本生成，然后通过以代码为中心的生成方式和教学法验证机制生成经过验证的多模态视频，最后将视频与评估和误解感知的补救教学相连接。实验和专家评估……

    arXiv:2609.24083v1 Announce Type: new  Abstract: Generative AI enables scalable production of educational videos, but current systems largely focus on producing visually coherent content rather than supporting learning. As a result, generated videos often lack explicit pedagogical structure, reliable quality control, and mechanisms for assessing learner understanding or addressing misconceptions. In this work, we introduce PIVOT (Pedagogy-guided Instructional VideO Tutoring), a generative video tutoring framework for STEM learning via learning-centered instructional support.1 Inspired by conventional teaching workflows, our framework integrates pedagogy into the full generation pipeline: it first uses instructional principles to guide storyboard generation, then produces verified multimodal videos through code-centric generation and a pedagogical verification harness, and finally connects videos with assessment and misconception-aware remediation. Experiments and expert evaluations acr
    
[^52]: 基于状态条件化潜在引导与进展指导的高效推理探索

    Efficient Reasoning Exploration via State-Conditioned Latent Steering with Progress Guidance

    [https://arxiv.org/abs/2609.24066](https://arxiv.org/abs/2609.24066)

    提出了一种无需训练的潜在引导框架SPS，通过构建包含进展引导向量的状态条件化方向库，在推理时引导模型探索能取得有意义进展的多样化推理路径，从而缓解探索坍缩问题并提升Best-of-N推理的探索效率。

    

    Best-of-N 是一种广泛用于复杂推理的推理策略，其有效性取决于采样得到的候选能否覆盖多样化且高质量的推理路径。然而，经过后训练的推理模型常常出现“探索坍缩”（exploration collapse）问题，即独立的多次采样反复遵循相似的推理路径，从而限制了增加采样预算所带来的收益。现有方法通过促进更广泛的探索来缓解这一问题，但并未显式引导探索朝着能取得有意义进展的延续方向进行，导致探索效率有限。为解决这一问题，我们提出了状态条件化进展引导（State-conditioned Progress-guided Steering, SPS），这是一个无需训练的潜在引导框架。具体而言，SPS 构建了一个状态条件化的方向库，其中包含针对不同前缀状态区域的多个进展引导向量。在在线推理过程中，SPS 重新（检索）……

    arXiv:2609.24066v1 Announce Type: new  Abstract: Best-of-$N$ is a widely used inference strategy for complex reasoning, whose effectiveness depends on whether sampled candidates can cover diverse and high-quality reasoning paths. However, post-trained reasoning models often suffer from \emph{exploration collapse}, where independent rollouts repeatedly follow similar reasoning paths and limit the gains from increasing the rollout budget. Existing methods alleviate this issue by promoting broader exploration, but do not explicitly guide exploration toward continuations that make meaningful progress, resulting in limited exploration efficiency. To address this, we propose \emph{\underline{S}tate-conditioned \underline{P}rogress-guided \underline{S}teering} (SPS), a training-free latent steering framework. Specifically, SPS constructs a state-conditioned Direction Bank containing multiple progress-guided steering vectors for different prefix-state regions. During online inference, SPS retr
    
[^53]: 基于表示引导的上下文学习实现多模态大语言模型的医学图像解读

    Representation-guided in-context learning for medical image interpretation with multimodal large language models

    [https://arxiv.org/abs/2609.24057](https://arxiv.org/abs/2609.24057)

    该论文提出了无需训练的表示引导上下文学习框架（RG-ICL），利用冻结编码器检索与查询对齐的示例来增强多模态大语言模型的医学图像解读能力，在八个医学数据集上显著提升分类和视觉问答性能，并发现少量与查询对齐的病例比大量随机病例更有效。

    

    医学图像解读是诊断和医疗护理的核心，然而适配通用多模态大语言模型（MLLM）通常需要资源密集型的领域特定微调。在此，我们提出了表示引导的上下文学习（RG-ICL），这是一种无需训练的推理框架，利用冻结编码器检索与查询对齐的示例，无需任务特定的参数更新。在涵盖组织病理学、放射学和视网膜眼底检查的八个数据集上，RG-ICL相比无上下文和传统ICL方法，提升了分类性能（平均提升20个百分点）和视觉问答（VQA）性能（平均提升13个百分点），接近或超过了基于训练的对比方法。检索哪些病例比检索多少病例更为重要：6个与查询对齐的病例优于多达32个随机选择的病例，而固定或随机选择的病例往往使准确率降至基线以下。对于VQA，将参考病例与（摘要在此处截断）

    arXiv:2609.24057v1 Announce Type: cross  Abstract: Medical image interpretation is central to diagnosis and care, yet adapting general-purpose multimodal large language models (MLLMs) often requires resource-intensive domain-specific fine-tuning. Here we introduce representation-guided in-context learning (RG-ICL), a training-free inference framework that retrieves query-aligned demonstrations using frozen encoders, without task-specific parameter updates. Across eight datasets spanning histopathology, radiology and retinal fundoscopy, RG-ICL improved classification (mean gain 20 percentage points) and visual question answering (VQA) (mean gain 13 percentage points) over no-context and conventional ICL, approaching or exceeding training-based comparators. Which cases were retrieved mattered more than how many: 6 query-aligned cases outperformed up to 32 randomly selected ones, whereas fixed or random cases often reduced accuracy below baseline. For VQA, aligning reference cases with bo
    
[^54]: 规模化校准决策：用“系统一”模型（Jev）将警方事故叙述转换为概率化事故变量

    Calibrated Decisions at Scale: Converting Police Crash Narratives into Probabilistic Crash Variables with a System One Model (Jev)

    [https://arxiv.org/abs/2609.24052](https://arxiv.org/abs/2609.24052)

    本文提出Jev——一个不生成文本、直接输出校准概率的“系统一”模型，将近50万条警方事故叙述的大规模编码转化为门控类型化决策，以低成本实现可验证、经人类盲评审计的高精度（F1=0.908）事故变量提取。

    

    携带调查员叙述的事故数据集包含编码字段所遗漏的信息。大规模对这些叙述进行编码一直受到三个障碍的阻碍：前沿大语言模型在此规模下成本高昂，其生成的文本无法验证，且没有规则说明人类必须检查多少输出。本文将叙述编码表述为门控的、类型化的决策，由Jev来回答——Jev是一个“系统一”模型，它返回分析师所定义选项上的概率，且不生成任何文本。该筛选覆盖了499,500条德克萨斯州事故叙述，其中195,857条通过一个包含27个问题的模式进行了编码。成本由模式大小而非叙述长度决定。这些概率对照编码字段以及在既定抽样设计下抽取的2,416个盲评人类判断进行了审计。两个前沿大语言模型在相同记录上进行了基准测试。对照人类标签，该类型化模型达到了0.908的F1分数。其中一个前沿模型获得0.059的提升（摘要在此处被截断）。

    arXiv:2609.24052v1 Announce Type: new  Abstract: Crash datasets that carry an investigator narrative hold information the coded fields omit. Coding those narratives at scale has been blocked by three obstacles. Frontier large language models are costly at that scale, their generated text cannot be verified, and no rule says how much output a human must check. This paper formulates narrative coding as gated, typed decisions answered by Jev, a System One model that returns probabilities over analyst-defined options and generates no text. A screen covered 499,500 Texas narratives and 195,857 were coded with a 27-question schema. Cost is governed by schema size rather than narrative length. The probabilities are audited against coded fields and against 2,416 blinded human judgments drawn under a stated sampling design. Two frontier large language models are benchmarked on the same records. Against human labels the typed model attains an F1 of 0.908. One frontier model gains 0.059 and the o
    
[^55]: 当证据冲突时：可靠性感知的元评审生成

    When Evidence Conflicts: Reliability-aware Meta-review Generation

    [https://arxiv.org/abs/2609.24028](https://arxiv.org/abs/2609.24028)

    该论文提出一种可靠性感知的元评审生成框架，通过抽取方面级观点、识别冲突证据，并结合观点支持度与评审质量评估证据可靠性，对评审反馈进行加权，从而在证据冲突时优先采纳更可信的论点并保留多样化观点。

    

    当评审者的证据存在冲突且可靠性各异时，从多篇同行评审意见中生成连贯的元评审（meta-review）是一项具有挑战性的任务。现有方法通常将元评审生成构建为多文档摘要任务，并均匀地聚合评审者反馈，因此在意见不一致时难以确定应优先采纳哪些观点。在本文中，我们通过可靠性感知的证据聚合来研究元评审生成问题。我们的框架首先从同行评审意见中抽取方面级（aspect-level）观点，并识别每个方面内相互冲突的证据；随后通过估计观点层面的支持度和评审层面的质量来衡量证据的可靠性。基于这些信号，该框架为评审者反馈分配可靠性感知的权重，使生成器能够优先采纳支撑更充分的论点，同时保留多样化的观点。实验表明，我们的方法持续改进元评审……（原文摘要在此处截断）

    arXiv:2609.24028v1 Announce Type: new  Abstract: Generating coherent meta-reviews from multiple peer reviews is challenging when reviewer evidence conflicts and varies in reliability. Existing approaches typically formulate meta-review generation as a multi-document summarization task and aggregate reviewer feedback uniformly, making it difficult to determine which opinions should be prioritized under disagreement. In this paper, we study meta-review generation through reliability-aware evidence aggregation. Our framework first extracts aspect-level opinions from peer reviews and identifies conflicting evidence within each aspect. It then estimates opinion-level support and review-level quality to measure evidence reliability. Based on these signals, the framework assigns reliability-aware weights to reviewer feedback, enabling the generator to prioritize better-supported arguments while preserving diverse perspectives. Experiments demonstrate that our method consistently improves meta
    
[^56]: AURA：面向语音基础模型声学接地的不确定性路由激活编辑

    AURA: Uncertainty-Routed Activation Editing for Acoustic Grounding in Speech Foundation Models

    [https://arxiv.org/abs/2609.23979](https://arxiv.org/abs/2609.23979)

    AURA是一种超高效的激活编辑方法，它冻结预训练语音基础模型，利用交叉注意力不确定性特征动态路由稀疏的缩放平移编辑，将非语音音频上的幻觉率从89.18%降至1.94%，并提升了模型在不完善标签和不流畅语音上的声学接地能力。

    

    注意力编码器-解码器（AED）语音基础模型在自动语音识别（ASR）任务上表现出色，但当输入中不含语音、声学证据薄弱或转写不可靠时，可能会生成缺乏声学支持的文本。我们提出了AURA（基于不确定性路由的激活编辑适应），这是一种超高效的表示编辑方法，它冻结预训练模型，并对解码器交叉注意力头应用稀疏的缩放和平移编辑。AURA利用交叉注意力不确定性特征来动态路由编辑，这些特征能够捕捉注意力过度集中、注意力分散以及帧的突然转移。我们在四个数据集上评估了AURA，这些数据集涵盖了非语音幻觉和语音接地压力因素，包括标签不完善的儿童语音、标签不完善的成人语音以及不流畅语音。在非语音音频上，AURA无需事先识别幻觉头，即可将幻觉率从89.18%降低至1.94%。在不完善标签语料库上，AURA（摘要在此处被截断）

    arXiv:2609.23979v1 Announce Type: cross  Abstract: Attention encoder-decoder (AED) Speech Foundation Models achieve strong ASR performance but can generate acoustically unsupported text when inputs contain no speech, weak acoustic evidence, or unreliable transcription. We propose AURA: Activation-editing with Uncertainty-Routed Adaptation, an ultra-efficient representation-editing method that freezes the pretrained model and applies sparse scale-and-shift edits to decoder cross-attention heads. AURA dynamically routes edits using cross-attention uncertainty features that capture over-concentration, diffuse attention, and abrupt frame shifts. We evaluate AURA on four datasets spanning non-speech hallucination and speech grounding stressors, including imperfect-label child speech, imperfect-label adult speech, and disfluent speech. On non-speech audio, AURA reduces hallucination rate from 89.18% to 1.94% without prior hallucination-head identification. On imperfect-label corpora, AURA ap
    
[^57]: 从表格到量化陈述：通过可执行验证评估大语言模型推理生成

    From Tables to Quantified Statements: Evaluating LLM Inference Generation through Executable Verification

    [https://arxiv.org/abs/2609.23966](https://arxiv.org/abs/2609.23966)

    该论文提出STAT-TO-TEXT任务，通过执行大语言模型生成的Python检查代码来验证模型从统计表格生成的量化陈述，发现更大的模型（GPT-OSS-120B）能生成最忠实的推理，且不牺牲表格覆盖率和量词多样性。

    

    大语言模型（LLM）能够从表格生成流畅的描述，但其输出在逻辑上可能缺乏结构化数据的支持。我们提出了STAT-TO-TEXT，这是一个受控任务，其中大语言模型使用“所有”、“某些”、“没有”和“大多数”等量化结构，从统计表格生成量化的自然语言推理。为了评估这些推理，我们使用大语言模型生成的Python检查代码，该代码在执行时会根据表格验证相应的真值条件。我们比较了四个不同模型家族和规模的开源权重大语言模型，评估其忠实性、逻辑准确性、表格覆盖率和多样性。我们的结果表明，模型规模和模型家族都很重要，其中最大的模型（GPT-OSS-120B）始终生成最忠实的推理，且不会牺牲表格覆盖率和量词多样性，而较小的模型则不然。这些发现得到了人工标注的支持，人工标注表明自动检查器……

    arXiv:2609.23966v1 Announce Type: new  Abstract: LLMs can generate fluent descriptions from tables, but their outputs may remain logically unsupported by the structured data. We introduce STAT-TO-TEXT, a controlled task in which LLMs generate quantified natural language inferences from statistical tables using quantified constructions such as all, some, no, and most. To evaluate these inferences, we use an LLM generated Python checker code which when executed verifies the corresponding truth conditions against the table. We compare four open-weight LLMs across model families and scales, evaluating faithfulness, logical accuracy, table coverage, and diversity. Our results show that model scale and family matter, with the largest model (GPT-OSS-120B) consistently producing the most faithful inferences without sacrificing greater table coverage and quantifier diversity, as opposed to smaller models. These findings are supported by human annotation, which shows that the automated checker c
    
[^58]: 基于CallScreenBench的开放Jev判断：使用小型语言模型实现校准的单次前向传播诈骗筛查

    Open-Jev Judgments on CallScreenBench: Calibrated One-Pass Scam Screening with a Small Language Model

    [https://arxiv.org/abs/2609.23959](https://arxiv.org/abs/2609.23959)

    该论文提出JevLite方法，通过对Qwen3-4B进行LoRA微调并从标签logit的softmax中直接读取校准的诈骗概率，在CallScreenBench上实现了单次前向传播的诈骗电话筛查，性能不劣于大型LLM裁判，速度提升4.9倍且对合法来电零误报。

    

    筛查电话诈骗需要在来电者每一轮对话后以毫秒级速度给出可信的概率。Jev式的类型化决策正是为此设计的：输入预先声明的选项，通过单次前向传播为每个选项输出一个经过校准的概率，无需生成任何文本。我们在诈骗电话筛查任务上测试了这一读取方式的开放实现JevLite：对Qwen3-4B进行LoRA微调，使两个答案标签logit上经温度缩放的softmax输出即为P(诈骗)。在41个留出的CallScreenBench场景（共577个逐轮决策）上，三种子集成模型达到AUROC 0.974、校准误差0.052，在预先注册的0.02容差下不劣于LLM裁判（MiniMax-M3），对合法来电零误报，在相同挂断规则下提前1.14轮做出决策，且在单块消费级GPU上每次决策仅需64.5毫秒，比微调同一骨干模型让其生成答案的方式快4.9倍。其增益来自读取方式与校准，而非准确率。

    arXiv:2609.23959v1 Announce Type: new  Abstract: Screening a phone call for fraud needs a trustworthy probability after every caller turn, in milliseconds. Jev-style typed decisions promise exactly that: declared options go in, one calibrated probability per option comes out of a single forward pass, with no generated text. We test an open implementation of this readout, JevLite, on scam-call screening: Qwen3-4B is LoRA-tuned so that the temperature-scaled softmax over two answer-label logits is P(scam). On 41 held-out CallScreenBench scenarios (577 per-turn decisions) a three-seed ensemble reaches AUROC .974 with calibration error .052, non-inferior to an LLM judge (MiniMax-M3) at a pre-registered .02 margin, with no false alarms on legitimate calls, decisions 1.14 turns earlier under the same hang-up rule, and 64.5 ms per decision on one consumer GPU, 4.9x lower than the same backbone fine-tuned to generate its answer. The gain is in the readout and calibration, not accuracy: a fine-
    
[^59]: 有些方言比其他方言更“平等”：大语言模型中的非声望阿拉伯语方言偏见

    Some Dialects Are More Equal Than Others: Non-Prestigious Arabic Dialectal Bias in LLMs

    [https://arxiv.org/abs/2609.23955](https://arxiv.org/abs/2609.23955)

    该研究通过针对性句法评估和MMLU基准测试发现，多个大语言模型对埃及声望较低的赛伊德方言存在显著偏见，且这种偏见会导致模型处理该方言时性能明显下降。

    

    以往自然语言处理（NLP）中对埃及阿拉伯语的研究主要集中于具有声望地位的开罗埃及阿拉伯语（CEA）方言，导致声望较低的赛伊德埃及阿拉伯语（SEA）方言在大语言模型和资源开发中都缺乏代表性。这种代表性缺失是否会影响大语言模型对SEA可接受性的判断（上游），而这种针对SEA的上游偏见又是否会导致更差的性能（下游）？我们在一项针对性句法评估（TSE）任务中考察了SEA方言特征对大语言模型偏好的上游影响，结果在多个大语言模型中发现了对SEA的显著偏见。随后，我们分析了这些相同的方言特征在MMLU基准测试中对模型下游性能的影响，结果表明当模型面对SEA时性能会出现退化。这项工作凸显了进一步探索次方言变体如何影响语言技术的必要性。

    arXiv:2609.23955v1 Announce Type: new  Abstract: Previous work on Egyptian Arabic in NLP has focused largely on the prestigious Cairene Egyptian Arabic (CEA) dialect, resulting in a lack of representation for the less prestigious Sa'idi Egyptian Arabic (SEA) dialect both in LLM and resource development. Does this lack of representation influence an LLM's view of the acceptability of SEA (upstream), and does an upstream bias against SEA lead to worse performance (downstream)? We investigate the upstream effect of SEA dialectal features on LLM preferences in a Targeted Syntactic Evaluation (TSE) task which reveals a significant bias against SEA across multiple LLMs. We then analyze the effect of these same features on downstream model performance on MMLU benchmarks and show that models experience a degradation in performance when presented with SEA. This work highlights the need for further exploration on how sub-dialectal variation impacts language technologies.
    
[^60]: HaikuS2S：一个用于以诗歌回应的级联系统

    HaikuS2S: A Cascaded System For Responding In Verse

    [https://arxiv.org/abs/2609.23951](https://arxiv.org/abs/2609.23951)

    提出了HaikuS2S级联系统，结合ASR、LLM生成俳句以及在诗歌与俳句数据集上微调的TTS，显著改善了俳句语音的韵律与声调对齐，同时保持了情感相似度。

    

    富有表现力的语音合成已通过韵律建模取得进展，然而生成结构化的诗歌语音（如俳句）仍然具有挑战性。先前关于韵律迁移的工作提升了表现力，微调的诗歌TTS（文本转语音）系统能够捕捉诗句的语调。然而，这些模型并未对俳句的5-7-5音节结构或行尾停顿进行建模。我们提出了一个级联系统HaikuS2S，它结合了ASR（自动语音识别）、LLM（大语言模型）生成的俳句，以及在散文和自定义俳句数据集上进行微调的TTS。我们的评估聚焦于情感相似度、语音质量和韵律对齐。在实验中，我们发现微调后的系统显著改善了韵律和声调对齐，尤其是在通用诗歌和俳句数据上共同训练的系统。同时，所有系统都保持了相似的情感相似度得分。

    arXiv:2609.23951v1 Announce Type: new  Abstract: Expressive speech synthesis has advanced through prosody modeling, yet generating structured poetic speech, such as haiku, remains challenging. Prior work on prosody transfer improves expressiveness, and fine-tuned poetry TTS (text-to-speech) systems capture verse intonation. However, these models do not model haiku's 5-7-5 syllable structure or line-ending pauses. We present a cascaded system, HaikuS2S, combining ASR (automatic speech recognition), LLM (large language model)-generated haiku, and TTS fine-tuning on both prose and custom haiku datasets. Our evaluation focuses on emotion similarity, speech quality, and prosody alignment. In our experiments, we see that our prosody and tonal alignment improve significantly with our fine-tuned systems, particularly the one trained on both general poetry and haiku. We also see that we maintain similar emotion similarity scores across all systems.
    
[^61]: XYEval：智能体对糟糕建议说“是”

    XYEval: Agents say yes to bad advice

    [https://arxiv.org/abs/2609.23939](https://arxiv.org/abs/2609.23939)

    提出XYEval元评估框架，可将现有基准转化为XY问题评估，发现智能体面对用户看似合理但误导性的建议时性能大幅下降（最高达46.7%），暴露出当前智能体难以识别用户真实问题并抵御糟糕建议的缺陷。

    

    用户与AI智能体之间的有效沟通对于人机协作至关重要。XY问题是一种众所周知的沟通陷阱，即用户询问的是他们尝试的解决方案，而非他们的实际问题。我们将先前的谄媚性评估扩展到智能体场景中的XY问题，评估智能体能否抵御用户提出的看似合理但具有误导性的建议，并清晰地传达其推理过程。我们提出了XYEval，这是一个元评估框架，可以将现有基准转化为XY问题评估。我们在六个不同的基准套件上评估了五个模型。结果显示，智能体在XY变异下于各基准中均出现大幅性能下降，相对下降幅度最高可达46.7%。借助τ²-bench，我们进一步发现，当遇到需要详细解释才会接受更优解决方案的迂腐用户时，智能体的性能下降更为严重。我们的研究结果表明，当前的智能体缺乏识别用户真实问题并抵御误导性建议的能力。

    arXiv:2609.23939v1 Announce Type: new  Abstract: Effective communication between users and AI agents is essential for human-AI collaboration. The XY problem is a well-known communication pitfall where a person asks about their attempted solution rather than their actual problem. We extend prior sycophancy evaluation to the XY problem in agentic settings, evaluating whether agents can resist plausible but misleading suggestions from users and communicate their reasoning. We introduce XYEval, a meta-evaluation framework that can transform an existing benchmark into an XY problem evaluation. We evaluate five models across six diverse benchmark suites. Agents suffer large XY drops under XY mutation across benchmarks, with relative drops reaching up to 46.7%. With $\tau^2$-bench, we further show that agent performance drops more when encountering a pedantic user who requires detailed explanations before approving a better solution. Our findings suggest that current agents lack the ability t
    
[^62]: 测量助手在用户轮次上的无害性偏好

    Measuring the Assistant's Harmlessness Preferences on the User Turn

    [https://arxiv.org/abs/2609.23935](https://arxiv.org/abs/2609.23935)

    研究发现后训练赋予助手模型的无害性偏好会泛化到用户轮次的预测中，表明后训练不仅塑造了浅层的助手角色，而是深度改变了模型本身。

    

    后训练将一个通用的下一词元预测器转变为一个具有持久助手角色的聊天模型。如果这个角色只是模型在自己轮次中扮演的角色，那么它的偏好应该只决定助手说什么，而不应影响模型对其他说话者会说什么的预测。我们对这一边界进行了测试，发现它并不成立：助手的一项安全相关偏好——即偏好无害任务而非有害任务——即使在用户轮次（即助手并非说话者的时候）也会影响模型的预测。我们发现这种偏好在预训练基础模型中很小或接近于零，它通过后训练产生，在多个开源权重模型家族中得到复现，随模型规模增长而增强，并且可以通过从未涉及用户轮次的窄域微调来改变。我们声称这是证据，表明后训练不仅仅是安装一个浅层的助手角色，而是超越了局部的助手轮次，泛化到了模型本身。

    arXiv:2609.23935v1 Announce Type: new  Abstract: Post-training turns a general next-token predictor into a chat model with a persistent assistant persona. If that persona is a character the model plays only on its own turns, its preferences should govern what the assistant says, not what the model predicts other speakers will say. We test this boundary and find that it does not hold: a safety-relevant preference of the assistant---for harmless over harmful tasks---shapes the model's predictions even on the user's turn, where the assistant is not the one speaking. We find that this preference is small or near-zero in pretrained base models, that it emerges through post-training, replicated across open-weight model families, grows with scale, and can be moved by narrow finetuning that never touches user turns. We claim that this is evidence that post-training does not merely install a shallow assistant persona, but instead generalises beyond just the local assistant turn, into the model'
    
[^63]: 大语言模型的时间增量持续预训练：无灾难性遗忘的知识更新

    Time-Incremental Continued Pretraining of LLMs: Knowledge Updates Without Catastrophic Forgetting

    [https://arxiv.org/abs/2609.23916](https://arxiv.org/abs/2609.23916)

    该研究在真实的时间增量场景下对六个开源大语言模型进行持续预训练，发现模型能有效获取新知识且不会发生灾难性遗忘，多数模型甚至还能提升对截止日期前旧知识的回忆能力。

    

    大语言模型（LLM）在预训练结束的那一刻就开始过时，而从头重新训练的成本又高得令人望而却步。持续预训练（CPT）是自然的补救方法，但它通常是在持续学习的视角下被评估的，该视角假设数据流彼此不相交。这种设定并不适用于网络规模爬取数据的时间增量更新，因为在这类场景中，相继的数据快照在设计上就共享大量的URL重叠。我们在这一现实情境下研究了时间增量持续预训练：使用严格取自每个模型知识截止日期之后的FineWeb-Edu数据转储进行持续预训练，并在三个模型家族（OLMo2、Llama-3.1/3.2、Gemma-3-1B）和四个参数规模（1B-3B-7B-8B）共六个开源权重模型上进行了评估。我们围绕四个实际问题组织研究发现：(i) 是否获得了新知识？是的，但获得是不均匀的，且没有发生灾难性遗忘：六个模型中有五个在截止日期前的事实回忆上也获得了提升，并且这些增益……

    arXiv:2609.23916v1 Announce Type: new  Abstract: Large language models (LLMs) drift out of date the moment their pretraining ends, yet retraining from scratch is prohibitively expensive. Continued pretraining (CPT) is the natural remedy, but it is typically evaluated through a continual learning lens that assumes disjoint data streams. This is a poor fit for time-incremental updates on web-scale crawls, where successive snapshots share substantial URL overlap by design. We study time-incremental CPT in this realistic regime: continued pretraining on FineWeb-Edu dumps drawn strictly from after each model's knowledge cutoff, evaluated across six open-weight models spanning three families (OLMo2, Llama-3.1/3.2, Gemma-3-1B) and four parameter scales (1B-3B-7B-8B).   We organize our findings around four practical questions. (i) Is knowledge acquired? Yes, but heterogeneously, and without catastrophic forgetting: five of six models also improve on pre-cutoff factual recall, and the gains tra
    
[^64]: this-that-model-1.0：一个在30毫秒内做出决策、成本仅百万分之一美分的类型化决策模型

    this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent

    [https://arxiv.org/abs/2609.23886](https://arxiv.org/abs/2609.23886)

    该论文提出2B参数的类型化决策模型this-that-model-1.0，它直接从指定位置的隐藏状态读取受限于调用者声明选项集的答案，不生成任何文本，在笔记本GPU上仅用30.9毫秒、零输出token即可完成决策，比前沿API调用（8758毫秒）快数百倍且成本近乎为零。

    

    每年，软件都把越来越多的分支决策交给模型来处理：一张工单进入哪个队列、一条命令是否可以安全执行、一项理赔是否无需人工审核即可通过。程序所需要的反馈并不是一段文字，而是 n 个已声明选项中的一个，以及一个可以设定阈值的数字。如今，这需要一次到前沿模型的往返调用——数百毫秒的延迟、按 token 计费的账单，还需要一个解析器——而问题通常只是三个子句的合取。this-that-model-1.0 是一个20亿参数的类型化决策模型。它的答案直接从指定位置的隐藏状态读取，并被限制在调用者声明的选项集合内，因此不会生成任何文本，不可能出现格式错误，且请求中的每个问题都在同一次前向传播中得到回答。它在单块笔记本 GPU 上以 30.9 毫秒完成决策，且生成零个输出 token，而一次前沿 API 调用需要 8758 毫秒，能很好回答这些问题的托管系统则需要花费 2……

    arXiv:2609.23886v1 Announce Type: new  Abstract: Software delegates more of its branches to models every year: which queue a ticket enters, whether a command is safe to run, whether a claim clears without a person. What the program needs back is not prose. It is one of n declared options and a number it can threshold. Today that costs a round trip to a frontier model -- hundreds of milliseconds, a per-token bill, and a parser -- for a question that is usually a conjunction of three clauses. this-that-model-1.0 is a 2B-parameter typed decision model. Its answer is read directly from the hidden state at a designated position and restricted to the option set the caller declared, so no text is generated, nothing can be malformed, and every question in a request is answered in the same forward pass. It decides in 30.9 ms on one laptop GPU and generates zero output tokens doing it, where a frontier API call costs 8758 ms and the hosted systems that answer these questions well spend between 2
    
[^65]: Q-TIE：一种轻量级且泛化性强的时序信息检索重排序框架

    Q-TIE: A Lightweight and Generalizable Re-ranking Framework for Temporal Information Retrieval

    [https://arxiv.org/abs/2609.23880](https://arxiv.org/abs/2609.23880)

    提出Q-TIE——一种基于学习的时序意图提取的轻量级、泛化性强的重排序框架，融合时序检索器与时序重排序器的互补优势，实现鲁棒的时序信息检索。

    

    随着检索增强生成（RAG）的兴起，时序信息检索（TIR）正变得日益重要。由于时间上不匹配的证据可能极具误导性，TIR旨在检索与查询在语义和时间上都相关的文档。目前出现了两种TIR范式——时序检索器和时序重排序器，二者在时间相关性建模方式上有所不同。尽管这两种范式提供了互补的优势，我们的分析表明，单独使用任何一种范式都难以实现鲁棒的TIR：时序检索器通过学习的表示提供灵活的查询理解，但往往无法显式地考虑时间约束；时序重排序器能够更明确地执行此类约束，但通常依赖于预定义的重排序规则。为解决这一问题，我们提出了Q-TIE，一种基于学习的时序意图提取（Temporal Intent Extraction, TIE）的重排序框架。通过引入一个TIE模型，将每个查询的时序意图映射为（摘要在此处截断）

    arXiv:2609.23880v1 Announce Type: new  Abstract: Temporal Information Retrieval (TIR) has been increasingly critical given the rise of Retrieval-Augmented Generation (RAG). Since temporally mismatched evidence can be highly misleading, TIR aims to retrieve documents that are both semantically and temporally relevant to a query. Two TIR paradigms have emerged - temporal retrievers and temporal re-rankers - differing in how temporal relevance is modeled. While these paradigms provide complementary strengths, our analysis reveals that each alone falls short of robust TIR: temporal retrievers provide flexible query understanding via learned representations, but often fail to explicitly account for temporal constraints; temporal re-rankers can enforce such constraints more explicitly, but often rely on predefined re-ranking rules. To address this, we propose Q-TIE, a re-ranking framework based on learned Temporal Intent Extraction (TIE). By introducing a TIE model that maps each query's tem
    
[^66]: 从UNDRR报告到事件记录：基于模式约束的大语言模型地理参照灾害提取

    From UNDRR Reports to Event Records: Schema-Constrained LLM Extraction of Georeferenced Disasters

    [https://arxiv.org/abs/2609.23853](https://arxiv.org/abs/2609.23853)

    该论文提出一种基于固定模式和受控词汇表的大语言模型流水线，可从UNDRR的PreventionWeb文档中自动提取带地理参照的灾害事件记录并保留审核证据，GPT-5的属性F₁达86.0%，远超传统spaCy基线方法的44.2%。

    

    减灾风险档案以叙述性文本描述灾害事件，而EM-DAT（Delforge等人，2025）等数据库无法直接摄取这些文本。我们提出了一种大语言模型流水线，使用受控的灾害词汇表和固定模式生成候选的地理参照事件记录，并保留证据供人工审核。将该流水线应用于由UNDRR管理的知识中心PreventionWeb的10,000份文档，它从1,913份文档中生成了3,572条记录，涵盖24种灾害类型，并将81%的位置提及解析为OpenStreetMap几何图形。在一个217份文档的分层参考集中，对171个被人工标注为正例的文档窗口进行评估，GPT-5实现了86.0%的合并属性F₁分数，而spaCy-地名录基线仅为44.2%。评估将文档内的灾害类别、位置字符串和事件年份合并汇总，未评估它们与具体事件的对应关系。GPT-5.4在十种大语言模型中排名最高（86.6% F₁）。GPT-5的逐字证据出现率为72.0%。

    arXiv:2609.23853v1 Announce Type: new  Abstract: Disaster-risk-reduction archives describe hazard events in prose that databases such as EM-DAT (Delforge et al., 2025) cannot ingest directly. We present an LLM pipeline that generates candidate georeferenced event records using a controlled hazard vocabulary and fixed schema, retaining evidence for review. Applied to 10,000 documents from PreventionWeb, the knowledge hub managed by UNDRR, it produced 3,572 records from 1,913 documents across 24 hazard types and resolved 81% of location mentions to OpenStreetMap geometries. On 171 human-positive document windows from a stratified 217-document reference set, GPT-5 achieved 86.0% pooled attribute $F_1$, versus 44.2% for the spaCy-gazetteer baseline. Evaluation pools hazard families, location strings, and event years within documents, without assessing their assignment to individual events. GPT-5.4 ranked highest among ten LLMs (86.6% $F_1$). Verbatim evidence occurrence was 72.0% for GPT-5
    
[^67]: 联邦多语言语音大语言模型：架构与聚合策略基准测试

    Federated Multilingual Speech-LLMs: Architecture and Aggregation Strategy Benchmarking

    [https://arxiv.org/abs/2609.23825](https://arxiv.org/abs/2609.23825)

    该论文对多语言自动语音识别的联邦学习进行了综合基准测试，发现独立调优各组件学习率并采用三组件自适应策略可取得最佳效果，且FedProx的收益依赖于LLM骨干架构——多语言预训练架构在异构数据分布下展现出更强的韧性。

    

    我们提出了一个针对多语言自动语音识别（ASR）的联邦学习（FL）综合基准测试，在Multilingual LibriSpeech数据集上评估了四种语音大语言模型（Speech-LLM）架构。我们在冻结与非冻结编码器配置下比较了FedAvg和FedProx两种方法，证明经过优化的学习率对性能至关重要。具体而言，对语音编码器、连接器和解码器分别独立调整学习率可获得最低的错误率，其中完整的三组件自适应方案（编码器和解码器使用LoRA，连接器进行全量训练）产生了最佳的联邦学习结果。我们观察到FedProx的有效性依赖于具体架构，它在多语言预训练架构中提供了显著优势（例如，在保持编码器固定时，EuroLLM优于TinyLlama）；这表明LLM骨干网络的容量在应对异构数据分布的韧性方面起着关键作用。这些发现为……

    arXiv:2609.23825v1 Announce Type: new  Abstract: We present a comprehensive benchmark of Federated Learning (FL) for multilingual Automatic Speech Recognition (ASR), evaluating four Speech-LLM architectures on the Multilingual LibriSpeech dataset. We compare FedAvg and FedProx across frozen and unfrozen encoder configurations, demonstrating that optimized learning rates are critical for performance. Specifically, independently tuning the learning rates for the speech encoder, connector, and decoder yields the lowest error rates, with full three-component adaptation (LoRA for encoder and decoder, full training for the connector) producing the best FL results. We observe that FedProx efficacy is architecture-dependent, providing notable advantages in multilingual pre-trained architectures (e.g., EuroLLM over TinyLlama when keeping the encoder fixed); this indicates that LLM backbone capacity plays a key role in mediating resilience to heterogeneous data distributions. These findings offe
    
[^68]: FLARE：基于生成式奖励模型的长程编码智能体全生命周期密集监督范式

    FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model

    [https://arxiv.org/abs/2609.23808](https://arxiv.org/abs/2609.23808)

    提出FLARE，一种由轻量级生成式奖励模型驱动的全生命周期密集监督范式，通过离线因果诊断框架RADAR提取无后见之明的监督信号，为长程编码智能体提供实时步骤级风险反馈，解决稀疏二值奖励带来的信用分配危机。

    

    虽然测试时扩展提升了大型语言模型（LLM）智能体在长程软件工程（SWE）任务中的表现，但稀疏的二值奖励（通过/失败）造成了严重的信用分配危机，并浪费了失败的探索轨迹。当前的轨迹优化与扩展方法成本高昂且在结构上存在局限，它们要么依赖缺乏因果诊断的启发式状态复用，要么依赖缺乏可操作在线引导的延迟标量评分。我们提出了FLARE（全生命周期对齐与奖励引擎），这是一种由轻量级生成式奖励模型（GRM）驱动的新型密集监督范式。首先，RADAR作为一个离线的因果感知诊断框架，通过因果链回溯提取高保真、无后见之明的监督信号，从而蒸馏出一个能够提供实时、步骤级风险反馈的GRM。其次，FLARE利用该GRM在智能体的整个生命周期中对其进行持续优化。在推理阶段，FLARE充当主动脚手架，自主……

    arXiv:2609.23808v1 Announce Type: new  Abstract: While test-time scaling enhances Large Language Model (LLM) agents in long-horizon software engineering (SWE), sparse binary rewards (Pass/Fail) create a severe credit assignment crisis and waste failed exploratory trajectories. Current trajectory optimization and scaling methods are costly and structurally limited, relying on heuristic state reuse without causal diagnosis or delayed scalar scoring without actionable online guidance. We propose FLARE (Full-Lifecycle Alignment and Reward Engine), a novel dense supervision paradigm driven by a lightweight Generative Reward Model (GRM). First, RADAR, an offline causal-aware diagnostic framework, extracts high-fidelity, hindsight-free supervision through causal-chain backtracking to distill a GRM providing real-time, step-level risk feedback. Second, FLARE uses this GRM to continuously optimize the agent across its entire lifecycle. During inference, FLARE acts as an Active Scaffold, autonom
    
[^69]: 约束解码消除了小型大语言模型的结构性失败，但揭示了与模型规模相关的语义鸿沟

    Constrained Decoding Eliminates Structural Failures in Small LLMs but Reveals a Scale-Dependent Semantic Gap

    [https://arxiv.org/abs/2609.23742](https://arxiv.org/abs/2609.23742)

    约束解码能完全消除小型大语言模型在结构化输出中的结构性失败，但内容准确性存在与模型规模相关的语义鸿沟——类型转换错误可被约束解码修复，而指令语义类失败（如多步函数调用）则无法通过约束解码解决。

    

    参数量在0.6B至4B之间的小型开源大语言模型（LLM）正被越来越多地部署用于结构化输出生成（如JSON、函数调用、数据提取），但在此规模区间内，约束解码（CD）如何与模型规模相互作用仍鲜为人知。我们在三种解码条件（原生解码、Outlines、XGrammar）下，对来自三个模型家族的五个模型在14个结构化输出任务上进行了基准测试。我们引入了一种双轴评估方法，将结构正确性（模式有效性）与语义正确性（内容准确性）分离开来。我们发现，约束解码消除了所有模型的所有结构性失败（模式有效性从78.6-92.9%提升至100%），但内容准确性揭示出一个持续存在且与规模相关的语义鸿沟：类型转换类失败完全可以通过约束解码挽救，而指令语义类失败（例如多步函数调用）则对约束解码具有抵抗性。模式符合是语义正确性的必要条件但并非充分条件。

    arXiv:2609.23742v1 Announce Type: new  Abstract: Small open-source large language models (LLMs) in the 0.6B-4B parameter range are increasingly deployed for structured output generation (JSON, function calling, data extraction), yet little is known about how constrained decoding (CD) interacts with model scale in this regime. We benchmark five models from three families across 14 structured-output tasks under three decoding conditions (native, Outlines, XGrammar). We introduce a two-axis evaluation that separates structural correctness (schema validity) from semantic correctness (content accuracy). We find that CD eliminates all structural failures across all models (schema validity: 78.6-92.9% to 100%), but content accuracy reveals a persistent semantic gap that is scale-dependent: type coercion failures are fully CD-rescuable, while instruction-semantic failures (e.g., multi-step function calling) remain CD-resistant. Schema conformance is necessary but not sufficient for semantic co
    
[^70]: GRACE：基于加拿大法律的接地对抗性推理

    GRACE: Grounded Adversarial Reasoning over Canadian Law

    [https://arxiv.org/abs/2609.23726](https://arxiv.org/abs/2609.23726)

    该论文提出了GRACE数据集，包含1,915个基于加拿大联邦立法的问题-推理-答案实例，涵盖对抗性辩护、不确定性和应用推理三种模式，并配套开发了数据构建流程及微调了轻量级法律推理模型CLeAR-4B，填补了加拿大法律在法律NLP基准测试中的空白。

    

    大型语言模型在一系列法律任务中已展现出强大性能，但现有基准很少评估模型持有并辩护法律立场的能力、在不完整信息下进行推理的能力，或综合多个法律条款的能力。这一差距在加拿大法律领域尤为突出，因为加拿大法律在法律自然语言处理（NLP）中的代表性仍然不足。我们推出了GRACE（Grounded Reasoning Adversarial Canadian LEgal examples，基于加拿大法律示例的接地对抗推理），这是一个包含1,915个问题-推理-答案实例的数据集，其内容植根于加拿大联邦立法。GRACE涵盖三种推理模式：对抗性辩护、不确定性推理和应用推理。我们开发了一套流程，该流程对原始法律条文进行切分、生成基于场景的问题与推理，并通过无模型的引用验证和基于大语言模型的质量审计来筛选示例。作为概念验证，我们微调了CLeAR-4B（Canadian Legal Adversarial Reasoning，加拿大法律对抗推理），这是一个用于接地法律推理的轻量级模型。

    arXiv:2609.23726v1 Announce Type: new  Abstract: Large language models have shown strong performance across a range of legal tasks, but existing benchmarks rarely evaluate the ability to take and defend a legal position, reason under incomplete information, or synthesize multiple statutory provisions. This gap is particularly pronounced for Canadian law, which remains underrepresented in legal NLP. We introduce GRACE (Grounded Reasoning Adversarial Canadian LEgal examples), a dataset of 1,915 question-reasoning-answer instances grounded in Canadian federal legislation. GRACE covers three reasoning modes: adversarial advocacy, uncertainty, and applied reasoning. We develop a pipeline that partitions raw statutory text, generates scenario-based questions and reasoning, and filters examples through model-free citation verification and LLM-based quality auditing. As a proof of concept, we fine-tune CLeAR-4B (Canadian Legal Adversarial Reasoning), a lightweight model for grounded legal reas
    
[^71]: STEVE：通过错误驱动精炼与正则化验证稳定基于文本梯度的提示优化

    STEVE: Stabilizing Textual Gradient-Based Prompt Optimization via Error-Driven Refinement and Regularized Verification

    [https://arxiv.org/abs/2609.23716](https://arxiv.org/abs/2609.23716)

    提出STEVE稳定化框架，通过仅从错误样本生成梯度的“错误驱动精炼”和基于保留集防止性能回退的“正则化验证”两种耦合机制，有效稳定文本梯度提示优化过程，减少性能退化并生成更鲁棒的提示。

    

    文本梯度方法通过自然语言反馈实现提示优化的自动化，但其迭代更新可能不稳定。我们识别出这种不稳定性的两个来源：由已经正确的样本产生的噪声梯度，以及对困难样本的过度特化（这会降低在较简单输入上的性能）。我们提出了STEVE，一个包含两个耦合机制的稳定化框架。错误驱动精炼仅从被错误处理的样本生成梯度，将更新集中在有信息量的失败案例上。正则化验证将每次更新视为临时性的，只有当在困难样本上的改进不会对保留集造成不可接受的性能回退时才接受该更新。在十个推理基准、三个评估器/优化器模型以及成熟的提示优化基线上，STEVE减少了性能退化并产生了更鲁棒的提示。此外，使用gpt-5.4-mini/gpt-5.4在符号推理、GSM8K-Pla……（摘要在此处被截断）

    arXiv:2609.23716v1 Announce Type: new  Abstract: Textual-gradient methods automate prompt optimization through natural-language feedback, but their iterative updates can be unstable. We identify two sources of this instability: noisy gradients produced from already-correct examples and over-specialization to hard cases that degrades performance on simpler inputs. We introduce STEVE, a stabilization framework with two coupled mechanisms. Error-Driven Refinement generates gradients only from incorrectly handled examples, concentrating updates on informative failures. Regularized Verification treats every update as provisional and accepts it only when improvement on hard cases does not cause unacceptable regression on a preservation set. Across ten reasoning benchmarks, three evaluator/optimizer models, and established prompt-optimization baselines, STEVE reduces degradation and produces more robust prompts. Additional evaluations with gpt-5.4-mini/gpt-5.4 on symbolic reasoning, GSM8K-Pla
    
[^72]: 金融语言模型作为市场摩擦下基于新闻交易的应用人工智能系统

    Financial Language Models as Applied Artificial Intelligence Systems for News-Based Trading under Market Frictions

    [https://arxiv.org/abs/2609.23703](https://arxiv.org/abs/2609.23703)

    该论文提出了MFAST框架，这是一个市场摩擦感知的情绪到交易框架，能够在考虑交易成本、流动性约束、执行时机等现实市场摩擦的条件下，将带时间戳的金融文本转化为可审计、可复现且市场可行的交易决策。

    

    金融语言模型能够将非结构化的公司特定新闻转化为结构化的决策信号，但金融人工智能研究缺乏一个集成的部署框架，用于评估这些信号在金融决策系统中是否仍然有用。计算机科学研究已经为时间序列预测、文本分类、多模态股票预测、基于图的市场建模和机器学习运维等领域开发了强大的方法，然而这些研究方向并未提供一个领域特定的协议，能够在事件时间可观测性、概率校准、执行时机、交易成本、流动性约束、容量限制、运营诊断和统计推断等条件下联合检验金融语言模型的输出。我们提出了MFAST，一个市场摩擦感知的情绪到交易框架，它将带时间戳的金融文本转化为可审计、可复现且市场可行的交易决策。该应用...

    arXiv:2609.23703v1 Announce Type: new  Abstract: Financial language models can transform unstructured firm-specific news into structured decision signals, but financial AI research lacks an integrated deployment framework for evaluating whether those signals remain useful in financial decision systems. Computer science research has developed strong methods for time-series forecasting, text classification, multimodal stock prediction, graph-based market modeling, and machine-learning operations, yet these streams do not provide a domain-specific protocol that jointly tests financial language-model outputs under event-time observability, probability calibration, execution timing, transaction costs, liquidity constraints, capacity limits, operational diagnostics, and statistical inference. We introduce MFAST, a Market-Friction-Aware Sentiment-to-Trading framework that converts timestamped financial text into auditable, reproducible, and market-feasible trading decisions. The application i
    
[^73]: 蒸馏你所信任的：可靠性感知的多教师在线策略蒸馏

    Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2609.23697](https://arxiv.org/abs/2609.23697)

    提出TrustMOPD方法，以专家模型RL训练前后相对于共享参考模型的位移作为token级可靠性代理，实现无标签的多教师加权蒸馏监督分配，将学生模型性能恢复率从54.4%大幅提升至91.5%。

    

    多教师在线策略蒸馏允许学生在自身生成的轨迹上从互补的专家模型中学习。然而，基于领域路由的方法在每个样本中仅选择一个教师，并在整个响应生成过程中保持固定。这种设计既依赖于混合训练语料库中通常缺失的标签，也无法在轨迹内部所需专业知识发生变化时调整教师选择。我们提出了TrustMOPD，它用无标签的、token级别的监督分配取代样本级教师选择。在每个学生生成的前缀处，TrustMOPD将每个专家模型经强化学习（RL）后相对于共享的RL前参考模型的位移作为局部可靠性的代理指标，跨教师对这些分数进行校准，并构建加权的蒸馏目标。在数学、代码和指令遵循任务上，TrustMOPD优于最强的无标签基线，将恢复率从54.4%提升至91.5%。

    arXiv:2609.23697v1 Announce Type: new  Abstract: Multi-teacher on-policy distillation allows a student to learn from complementary specialists on its own trajectories. Domain-routed approaches, however, select one teacher per example and keep it fixed throughout the response. This design both depends on labels that mixed training corpora often lack and cannot adapt teacher selection when the expertise required changes within a trajectory. We propose \textbf{TrustMOPD}, which replaces example-level teacher selection with label-free, token-level supervision allocation. At each student-generated prefix, TrustMOPD uses each specialist's RL-induced displacement from a shared pre-RL reference as a proxy for local reliability, calibrates these scores across teachers, and constructs a weighted distillation target. Across mathematics, code, and instruction following, TrustMOPD outperforms the strongest label-free baseline, increasing the recovery ratio from $54.4\%$ to $91.5\%$ on \textsc{Singl
    
[^74]: 超越相关性：基于大语言模型增强标注的商品搜索结构化语义监督

    Beyond Relevance: Structured Semantic Supervision for Product Search with LLM-Augmented Annotations

    [https://arxiv.org/abs/2609.23646](https://arxiv.org/abs/2609.23646)

    该研究通过LLM生成结构化查询与商品属性并结合人工验证的相关性、解释和中心性标注来增强商品搜索，发现LLM的核心价值在于暴露和近似结构化语义监督信号，而非完全取代人工标注。

    

    电商搜索需要区分那些仅仅与查询相关的商品和那些直接满足用户购物意图的商品。我们使用结构化的LLM生成的查询属性和商品属性，以及人工验证的相关性、解释和中心性判断来增强查询-商品对，并使用简单的双编码器检索器和MLP重排序器来评估这些信号。在ESCI数据集的增强子集上，使用人工特征的oracle方法达到0.9382的nDCG@10，而无需人工特征的训练后Q+P配置达到0.9258。人工信号的合成近似版本总体达到0.9150，但对困难、低性能的查询提供了显著提升。消融实验表明，oracle改进的大部分来自后编辑的解释文本和标注者评论，而不是标量中心性特征，这表明LLM最有价值的作用在于暴露和近似结构化语义监督信号，而不是完全取代人工标注。

    arXiv:2609.23646v1 Announce Type: cross  Abstract: E-commerce search requires distinguishing products that are merely related to a query from those that directly satisfy the user's shopping intent. We augment query-product pairs with structured LLM-generated query and product attributes and human-validated relevance, explanations, and centrality judgments, and evaluate these signals using a simple dual-encoder retriever and MLP re-ranker. On an augmented subset of ESCI, a human-feature oracle reaches $0.9382$ nDCG@10, while a human-free trained $Q+P$ configuration reaches $0.9258$. Synthetic approximations of the human signals reach $0.9150$ overall but provide substantial gains for difficult, low-performing queries. Ablations show that most of the oracle improvement comes from post-edited explanations and annotator comments rather than the scalar centrality feature, suggesting that LLMs are most useful for exposing and approximating structured semantic supervision rather than replacin
    
[^75]: 崩溃而非复杂性：面向端到端文档解析的失败条件化分解修复

    Collapse, Not Complexity: Failure-Conditioned Decomposition Repair for End-to-End Document Parsing

    [https://arxiv.org/abs/2609.23592](https://arxiv.org/abs/2609.23592)

    该论文发现文档解析失败的关键不是页面复杂性而是解析崩溃，提出从普通解析轨迹中检测崩溃、按投影分解页面并重新解析各区域的修复方法，以仅1.13倍的token成本带来1.40 Overall的质量提升。

    

    端到端文档解析器日益为复杂页面提供可选的推理模式。在一个180页的熵分层发现样本上，使用一个冻结的4B检查点，我们证明复杂性是错误的决策变量。推理模式使平均质量下降2.21 Overall，同时消耗1.54倍的token；一个预注册的仅基于输入的模型无法预测其有符号收益（留出集AUROC为0.47，与随机猜测无异）。收益集中在普通解析已经崩溃的页面上，而这些页面看起来并不复杂：发生崩溃的页面其布局熵比健康页面更低，却以退化重复的形式消耗19倍的token，即使将预算加倍也无法解决。切换到推理模式很少能修复这些页面：83%的崩溃在推理模式下复发。我们转而从普通解析的轨迹中检测崩溃，通过投影将页面分解，并重新解析每个区域。该修复方法带来1.40 Overall的增益（95% CI [0.68, 2.16]），token消耗仅为1.13倍，在三个检查点上均可复现，并且

    arXiv:2609.23592v1 Announce Type: cross  Abstract: End-to-end document parsers increasingly offer an optional reasoning mode for complex pages. On a 180-page entropy-stratified discovery sample with one frozen 4B checkpoint, complexity is the wrong decision variable. Reasoning lowers mean quality by 2.21 Overall at 1.54x tokens; a preregistered input-only model cannot predict its signed benefit (held-out AUROC 0.47, indistinguishable from chance). The benefit concentrates on pages whose ordinary pass has already collapsed, and they do not look complex: shared collapses have lower layout entropy than healthy ones yet consume 19x the tokens as degenerate repetition that doubling the budget does not cure. Switching modes rarely repairs them: 83% recur under reasoning. We instead detect collapse from the ordinary-pass trace, decompose the page by projection, and re-parse each region. Repair gains 1.40 Overall (95% CI [0.68, 2.16]) at 1.13x tokens, replicates across three checkpoints, and, 
    
[^76]: 大型推理模型中的效率-安全困境研究

    On the Efficiency-Safety Dilemma in Large Reasoning Models

    [https://arxiv.org/abs/2609.23587](https://arxiv.org/abs/2609.23587)

    本研究首次全面分析了大型推理模型中效率优化技术与安全性的关系，发现量化、剪枝等方法带来的安全性提升是表面的——源于推理能力退化导致恶意响应“尝试但失败”，而非真正的对齐增强，并指出量化结合剪枝是平衡效率与安全的最佳策略。

    

    大型推理模型（LRMs）具有较高的推理成本，通常通过量化、剪枝等效率技术来缓解。然而，这些技术对模型对抗鲁棒性的影响在很大程度上尚未被探索。本研究首次对大型推理模型中效率、越狱脆弱性与推理能力之间的相互作用进行了全面分析。我们发现，虽然效率方法表面上降低了越狱攻击的成功率，但这种改进往往是表面的。它在很大程度上源于推理能力退化导致的“尝试但失败”的恶意响应，而非真正的对齐能力的提升。对表征漂移的机制分析证实了这一点，揭示了推理能力损失与模型无法维持恶意语义轨迹之间的严格耦合。此外，我们确定量化结合剪枝是平衡效率与安全性的最优策略。

    arXiv:2609.23587v1 Announce Type: new  Abstract: Large reasoning models (LRMs) incur high inference costs, often mitigated by efficiency techniques like quantization and pruning. However, the impact of these techniques on model adversarial robustness remains largely unexplored. This study provides the first comprehensive analysis of the interplay between efficiency, jailbreak vulnerability, and reasoning in LRMs. We find that while efficiency methods seemingly reduce the success rate of jailbreak attacks, this improvement is often superficial. It largely arises from degraded reasoning capabilities leading to "attempted but failed" malicious responses, rather than an increase in genuine alignment. Mechanistic analysis of representational drift confirms this, revealing a strict coupling between reasoning capability loss and the model's inability to maintain malicious semantic trajectories. Additionally, we identify quantization with pruning as the optimal strategy to balance efficiency a
    
[^77]: 全局排序得以保持，选中注意力头却发生偏移：4比特仅权重量化下的BOS-Sink拓扑

    Global Ranks Survive, Selected Heads Shift: BOS-Sink Topology under 4-bit Weight-Only Quantization

    [https://arxiv.org/abs/2609.23585](https://arxiv.org/abs/2609.23585)

    该论文提出Sink拓扑一致性（STC）指标，发现4比特NF4量化下首token注意力头的全局排序高度保持（ρ_s≥0.980），但top-k集合重叠显著下降，且存在层间和跨域的局部失效，表明仅凭全局排序在量化前复用sink头映射并不安全。

    

    感知Sink的部署方式可以在模型量化之前识别重要的首token注意力头，然后在边缘端复用该映射图。我们测试了这种捷径在4比特NF4仅权重训练后量化（PTQ）下何时是安全的。我们提出的Sink拓扑一致性（STC）指标将全局排序保持、top-k集合重叠和逐层sink质量偏移区分开来，并将逐输入敏感性与校准映射迁移区分开。在Qwen2.5-0.5B、Qwen2.5-1.5B和Llama-3.2-1B上，在4096个token长度下，bf16到4比特的全局排序仍然很高（ρ_s ≥ 0.980），但top-k Jaccard重叠仅为0.619-0.793，对应76.5-88.5%的成员保留率。全局统计量还掩盖了局部失效：Qwen的末端层偏移达其模型均值的6.2-7.9倍，而Llama-3.2-1B则表现出较低且近乎均匀的漂移。在C4到LongBench的域偏移下，两个Qwen模型的跨域重叠比域内精度比较退化得更为严重。

    arXiv:2609.23585v1 Announce Type: cross  Abstract: Sink-aware deployment may identify important first-token attention heads before a model is quantized, then reuse that map at the edge. We test when this shortcut is safe for 4-bit NF4 weight-only post-training quantization (PTQ). Our Sink Topology Consistency (STC) metrics separate global rank preservation, top-$k$ set overlap, and layerwise sink-mass shift, and distinguish per-input sensitivity from calibration-map transfer. Across Qwen2.5-0.5B, Qwen2.5-1.5B, and Llama-3.2-1B, global bf16-to-4-bit ranks remain high at 4,096 tokens ($\rho_s \geq 0.980$), yet top-$k$ Jaccard overlap is only 0.619-0.793, corresponding to 76.5-88.5% membership retention. The global statistic also masks local failures: terminal Qwen layers shift by 6.2-7.9x their model means, whereas Llama-3.2-1B shows low, nearly uniform drift. Under a C4-to-LongBench shift, cross-domain overlap degrades more than the within-domain precision comparison for both Qwen model
    
[^78]: ARID：一种可部署的边缘AI系统，用于从工业维护工单中提取结构化信息

    ARID: A Deployable Edge AI System for Structured Information Extraction from Industrial Maintenance Work Orders

    [https://arxiv.org/abs/2609.23582](https://arxiv.org/abs/2609.23582)

    ARID是一种可在8GB Jetson Orin NX边缘设备上离线部署的系统，通过双教师过滤、噪声感知数据合成、单次路由决策、4位推理和语法约束解码，将工业维护工单可靠地提取为固定模式的JSON，实现84.8%的token-F1并保证超过99.8%的解析成功率。

    

    维护工单通常需要在嵌入式硬件上离线处理，而下游软件需要可预测的结构化输出。我们提出了ARID（面向工业部署的航空启发式路由系统），它能够在8 GB的NVIDIA Jetson Orin NX上，将组件、故障模式、症状和维护动作提取为固定模式的JSON。ARID结合了保守的双教师过滤、针对性的噪声感知合成、每个工单仅一次的路由决策、4位推理以及语法约束解码。从2,326条未标注的OMIn记录中，系统保留了716个训练对，并额外添加了99条针对动作提取的拓扑约束记录。在300条人工标注的记录上，ARID在参考技术栈上达到84.8%的token-F1，在部署的Jetson上达到82.9%。常驻服务在12.5 W功耗下实现了5,310/5,656 ms的P50/P99延迟。在零样本MaintNet迁移测试中，语义F1降至46.4%，但解析器成功率仍保持在至少99.8%，这表明输出有效性……

    arXiv:2609.23582v1 Announce Type: new  Abstract: Maintenance work orders must often be processed offline on embedded hardware, yet downstream software requires predictable structured output. We present ARID (Aviation-inspired Routing for Industrial Deployment), which extracts component, failure mode, symptom, and maintenance action into fixed-schema JSON on an 8 GB NVIDIA Jetson Orin NX. ARID combines conservative dual-teacher filtering, targeted noise-aware synthesis, one routing decision per work order, 4-bit inference, and grammar-constrained decoding. From 2,326 unlabeled OMIn records, it retains 716 training pairs and adds 99 topology-constrained records targeting action extraction. On 300 human-labeled records, ARID reaches 84.8% token-F1 on the reference stack and 82.9% on the deployed Jetson. Resident serving achieves 5,310/5,656 ms P50/P99 at 12.5 W. On zero-shot MaintNet transfer, semantic F1 falls to 46.4% while parser success remains at least 99.8%, showing that output vali
    
[^79]: 面向自动作文评分的错误监督式合成学习者写作

    Error-Supervised Synthetic Learner Writing for Automated Essay Scoring

    [https://arxiv.org/abs/2609.23573](https://arxiv.org/abs/2609.23573)

    该研究提出将错误监督引入大语言模型合成作文生成的方法，通过在语法错误检测常用的错误标注文本上微调生成器，使合成作文更贴近语言学习者的真实写作，在较大数据量设置下于12项对比中的11项优于常规合成基线。

    

    合成作文有助于减少自动作文评分（AES）对人工撰写数据的依赖。然而，合成作文往往缺乏真实的错误，限制了其表现真实人类写作的能力，尤其是当目标文本旨在接近语言学习者所写文本时。在本研究中，我们提出了一种简单的方法，将错误监督引入合成作文生成过程。具体而言，我们在语法错误检测（GED）中常用的带错误标注文本上微调一个大语言模型生成器。为评估所提方法的实用性，我们在三种数据条件下微调并评估了AES评分器：真实作文、常规方法生成的合成作文，以及使用我们提出的方法生成的合成作文。结果表明，在较大数据量的设置下，所提方法在12项数据集-指标对比中的11项上优于常规合成基线，并且……

    arXiv:2609.23573v1 Announce Type: new  Abstract: Synthetic essays can help reduce dependence on human-written data in Automated Essay Scoring (AES). However, they often lack realistic errors, limiting their ability to represent authentic human writing, particularly when the target texts are intended to resemble those produced by language learners. In this study, we present a simple approach that introduces error supervision into synthetic essay generation. Specifically, we fine-tune an LLM generator on error-annotated texts of the kind commonly used in Grammatical Error Detection (GED). To assess the utility of the proposed approach, we fine-tune and evaluate AES scorers under three data conditions: authentic essays, synthetic essays generated conventionally, and synthetic essays generated using our proposed approach. The results show that in the larger-data settings, the proposed approach outperforms the conventional synthetic baseline in 11 out of 12 dataset-metric comparisons, with 
    
[^80]: VibeMemBench：在真实仓库编码任务上评估编码智能体的记忆系统

    VibeMemBench: Evaluating Memory Systems for Coding Agents on Real Repository Coding Tasks

    [https://arxiv.org/abs/2609.23570](https://arxiv.org/abs/2609.23570)

    VibeMemBench 是首个在真实仓库编码任务上评估编码智能体记忆系统的基准，通过 111 个经验证能从注入历史经验中获益的编码目标和可执行测试来衡量记忆系统对下游编码成效的真实改善。

    

    编码智能体在真实的仓库编码任务上工作，而持久化记忆系统有望实现跨任务的经验复用。然而，现有的评估并未表明这些系统是否能改善可执行的仓库工作。仓库类基准测试代码变更但不隔离记忆因素，而记忆类基准只评估记忆召回却未衡量下游编码成效。我们提出 VibeMemBench，这是一个用于评估记忆系统的基准，包含来自 90 个 SWE-rebench V2 仓库的 111 个编码目标，以及来自这些目标仓库的 3,634 条历史轨迹。这些目标遵循 SWE 基准风格，涵盖缺陷修复、功能请求、接口变更和配置工作。智能体在声明的记忆条件下对每个目标代码库进行编辑，可执行测试决定任务是否被解决。只有在参考设置中注入历史经验能改善其可执行结果时，该目标才会被保留，因此每个目标都带有先验经验优势。

    arXiv:2609.23570v1 Announce Type: cross  Abstract: Coding agents operate on real repository coding tasks, and persistent memory systems promise to reuse experience across tasks. Yet existing evaluations do not show whether those systems improve executable repository work. Repository benchmarks test code changes but do not isolate memory, while memory benchmarks score recall without measuring downstream coding outcomes. We introduce VibeMemBench, a benchmark for evaluating memory systems on 111 coding targets from 90 SWE-rebench V2 repositories and 3,634 history trajectories from the target repositories. The targets follow the SWE benchmark style and cover bug fixes, feature requests, interface changes, and configuration work. An agent edits each target codebase under a declared memory condition. Executable tests decide task resolution. Each target is retained only when injected history experience improves its executable outcome in a reference setting, so every target carries a prior ex
    
[^81]: 对概率语言层次结构的贡献

    Contributions to the hierarchy of probabilistic languages

    [https://arxiv.org/abs/2609.23567](https://arxiv.org/abs/2609.23567)

    本文建立了n-gram模型与PCFG生成概率语言的层次关系，证明n-gram概率语言是PCFG概率语言的真子集，并引入全连接PCFG概念，证明n-gram概率语言与全连接PCFG概率语言互不相交。

    

    我们重新审视了由n-gram模型和概率上下文无关文法（PCFG）生成的概率形式语言理论。通过证明每个由n-gram模型生成的概率语言也可以由某个PCFG生成，而某些由PCFG生成的概率语言却无法由任何n-gram模型生成，我们建立了所预期的概率文法层次结构。我们引入了全连接PCFG的概念，即在乔姆斯基范式中，每条仅涉及非终结符的产生式规则都具有非零概率的PCFG。我们的主要结果表明，任何由n-gram模型生成的概率语言都不同于任何由全连接PCFG生成的概率语言。因此，由n-gram模型生成的概率语言类并不是由全连接PCFG生成的语言类的子集。

    arXiv:2609.23567v1 Announce Type: cross  Abstract: We reconsider the theory of probabilistic formal languages generated by n-gram models and by probabilistic context-free grammars (PCFGs). The expected hierarchy of probabilistic grammars is established by proving that every probabilistic language generated by an n-gram model is also generated by some PCFG, while some probabilistic languages generated by PCFGs cannot be generated by any $n$-gram model. We introduce the notion of fully connected PCFGs, namely PCFGs in Chomsky normal form where every production rule only involving non-terminals has non-zero probability. Our main result shows that any probabilistic language generated by an $n$-gram model differs from any probabilistic language generated by a fully connected PCFG. Therefore, the class of probabilistic languages generated by $n$-gram models is not a subset of the class generated by fully connected PCFGs.
    
[^82]: 段落边界并非空白：压缩深度作为层级结构的标志

    Paragraph Boundaries Are Not White Space:Compression Depth as the Signature of Hierarchical Structure

    [https://arxiv.org/abs/2609.23551](https://arxiv.org/abs/2609.23551)

    该研究通过层级旋转位置编码干预实验发现，真实层级结构的标志不是注意力压缩本身，而是压缩的深度——真实段落结构产生更深且随语料库变化的压缩，而密度匹配的随机标签对照仅产生更浅的压缩。

    

    标准的位置编码将位置表示为一维的阅读顺序坐标，但仅凭阅读顺序并不能决定文本的层级结构。我们采用一种层级旋转位置编码，将段落、句子和词元索引表示为相互独立的通道；在保持词元序列不变的情况下，对段落坐标 p1 进行干预，并使用词元距离精确估计器测量跨段落注意力。在所有语料库中，注意力相对于词元距离匹配的基线均出现压缩，但仅凭压缩本身并不能证明真实结构的存在：一个架构完全相同、但使用密度匹配随机标签的通道同样表现出压缩，只是程度更浅。真正区分真实结构的是压缩的深度：真实结构的压缩更深且依赖于具体语料库，而对照组则不具备这一特性。我们在三种构念（词汇持续性、段落长度、基于嵌入的……）上比较了八个仅依赖语料库的量……

    arXiv:2609.23551v1 Announce Type: new  Abstract: Standard positional encodings represent position as a one-dimensional reading-order coordinate, but reading order alone does not determine hierarchical textual structure. We use a hierarchical rotary positional encoding (hRoPE) that represents paragraph, sentence, and token indices as separate channels, hold the token sequence fixed, intervene on the paragraph coordinate p1, and measure cross-paragraph attention with a token-distance-exact estimator. Attention is compressed relative to a token-distance-matched baseline in every corpus, but compression alone is not diagnostic of true structure: an architecturally identical channel with density-matched random labels is compressed too, more shallowly. What distinguishes real structure is the depth of compression, which is greater and corpus-dependent while the control's is not. Comparing eight corpus-only quantities across three constructs (lexical persistence, paragraph length, embedding-b
    
[^83]: BabelArena：面向大语言模型智能体的大规模多语言基准测试

    BabelArena: A Large-Scale Multilingual Benchmark for LLM Agents

    [https://arxiv.org/abs/2609.23490](https://arxiv.org/abs/2609.23490)

    提出了BabelFlow通用工作流和BabelArena大规模多语言基准测试（涵盖23种语言、16,146个实例），首次系统评估了LLM智能体的跨语言能力，发现前沿模型存在显著的跨语言性能差异且无单一模型全面领先。

    

    大语言模型（LLM）智能体越来越多地通过工具使用以及与用户和环境的交互来执行多步骤工作流。然而，当前的智能体评估主要以英语为中心，限制了我们对多语言环境下智能体能力的理解。我们提出了BabelFlow，这是一种通用于各类基准测试的智能体工作流，它通过分析运行时依赖关系、协调保持结构的翻译，并将多层验证与人工审查相结合，将现有的智能体基准测试适配到新语言，以保留任务和评估语义。利用BabelFlow，我们构建了BabelArena，这是一个任务对齐的基准测试，包含16,146个实例，这些实例源自四个基准测试家族、13个领域和23种语言中的702个规范任务。对五个前沿模型的实验表明，没有任何单一模型能在所有基准测试家族中占据主导地位，且跨语言差异远不止体现在任务成功与否上。低资源语言……

    arXiv:2609.23490v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly execute multi-step workflows through tool use and interaction with users and environments. However, current agent evaluations are largely English-centric, limiting our understanding of agent capabilities in multilingual settings. We introduce BabelFlow, a benchmark-general agentic workflow that adapts existing agent benchmarks to new languages by analyzing runtime dependencies, coordinating structure-preserving translation, and combining multi-layer verification with human review to preserve task and evaluation semantics. Using BabelFlow, we construct BabelArena, a task-aligned benchmark comprising 16,146 instances derived from 702 canonical tasks across four benchmark families, 13 domains, and 23 languages. Experiments with five frontier models show that no single model dominates across benchmark families and that cross-language disparities extend well beyond task success. Lower-resource la
    
[^84]: RPMem：为LLM智能体学习跨会话的长期循环参数化记忆

    RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents

    [https://arxiv.org/abs/2609.23466](https://arxiv.org/abs/2609.23466)

    RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。

    

    长时间运行的LLM智能体需要能够跨会话持续存在并不断演化的记忆。基于文本的记忆在每次查询时都需要检索和重建过去的交互，随着历史记录的增长，长程性能越来越依赖于检索质量和上下文推理能力。参数化记忆将经验直接编码到模型计算中，但现有方法对跨会话记忆演化的支持有限，且其与特定骨干模型的耦合进一步限制了模型替换后的记忆复用。我们提出了RPMem，这是一种两阶段架构，通过前向计算将每个会话编译为与模型无关的潜在记忆，并通过任务训练的循环门控选择性地将其与保留的记忆进行整合。整合后的记忆随后被映射为特定于骨干模型的低秩适配（LoRA）参数，使得编码能力在骨干模型被替换时仍可迁移。在三个（数据集/任务上）的评估……

    arXiv:2609.23466v1 Announce Type: new  Abstract: Long-running LLM agents require memory that persists and evolves across sessions. Text-based memory retrieves and reconstructs past interactions at every query, making long-horizon performance increasingly dependent on retrieval quality and contextual reasoning as histories grow. Parametric memory encodes experience directly into model computation, but existing approaches provide limited support for cross-session memory evolution. Their coupling to a specific backbone further restricts memory reuse after model replacement. We introduce RPMem, a two-stage architecture that compiles each session into a model-independent latent memory through forward computation and selectively integrates it with retained memory via a task-trained recurrent gate. The consolidated memory is then mapped to backbone-specific low-rank adaptation (LoRA) parameters, allowing the encoding capability to transfer when the backbone is replaced. Evaluation across thre
    
[^85]: 提议、验证、提交：面向长时程多主体对话的证据锚定记忆

    Propose, Verify, Commit: Evidence-Grounded Memory for Long-Horizon Multi-Actor Conversations

    [https://arxiv.org/abs/2609.23465](https://arxiv.org/abs/2609.23465)

    EGMEMORY将长时程多主体对话记忆建模为可搜索状态机，通过证据锚定的“提议-验证-提交”协议管理记忆的写入与读取，无需记忆专项训练即在GroupMemBench和EverMemBench上分别取得68.2%和77.9%的最优性能。

    

    长时程对话记忆在多主体场景中尤其具有挑战性，因为相关证据分布在不同的参与者和情境之间，且先前确立的信息可能随后被修订。我们提出了EGMEMORY，它将长时程多主体记忆建模为一个可搜索的状态机，将持久的消息级证据与显式的活动状态分离开来。在写入时，自适应状态解析和基于证据的“提议-验证-提交”协议控制该状态的演化。在读取时，自适应证据导航迭代地解析查询所需的状态和支持证据，利用对话结构缩小搜索空间，并利用词汇-语义相关性对候选进行排序。该系统仅通过提示和工具使用即可运行，无需针对记忆的策略训练。EGMEMORY在GroupMemBench上达到68.2%，在EverMemBench上达到77.9%，超越了最强的评估基线。

    arXiv:2609.23465v1 Announce Type: new  Abstract: Long-horizon conversational memory is especially challenging in multi-actor settings, where relevant evidence is distributed across participants and contexts and previously established information may later be revised. We introduce EGMEMORY, which formulates long-horizon multi-actor memory as a searchable state machine that separates persistent message-level evidence from an explicit active state. At write time, adaptive state resolution and an evidence-grounded propose-verify-commit protocol govern how this state evolves. At read time, adaptive evidence navigation iteratively resolves the state and supporting evidence required for a query, using conversational structure to narrow the search space and lexical-semantic relevance to rank candidates. The system operates through prompting and tool use without memory-specific policy training. EGMEMORY achieves 68.2% on GroupMemBench and 77.9% on EverMemBench, outperforming the strongest evalu
    
[^86]: 困惑度预测保护：在联邦参数高效微调中选择预训练骨干网络以保障最差客户端的公平性

    Perplexity Predicts Protection: Choosing Pretrained Backbones for Worst-Client Fairness in Federated Parameter-Efficient Fine-Tuning

    [https://arxiv.org/abs/2609.23463](https://arxiv.org/abs/2609.23463)

    该研究发现目标文本上的困惑度可以在联邦训练开始前预测哪个预训练骨干网络最能保护数据最少的弱势客户端（秩相关系数达-0.87），为联邦LoRA微调中的骨干网络选择提供了一个简单有效的指标。

    

    联邦学习允许多方在不汇集各自数据的情况下共同训练一个共享模型，但数据量远少于其他参与方的客户端，即使群体平均准确率看起来不错，也可能得不到良好的服务。我们探究在LoRA微调下，预训练骨干网络的选择是否会影响这一现象，以及目标文本上的每词困惑度能否在联邦训练开始之前预测哪个骨干网络能帮助最弱势的客户端。我们在三个文本分类数据集和三个规模相近的骨干网络（RoBERTa、BERTweet、PubMedBERT）上进行了313次实验，每次实验都在相同的数据划分上与特定任务的基线进行比较。困惑度较低的骨干网络始终为表现最差的客户端带来更大的收益，在九个数据集-骨干网络组合上的秩相关系数为-0.87；一个未参与分析的留出骨干网络也证实了这一模式。使用Ditto进行个性化仅能恢复单独训练与联合训练之间差距的4-12%。

    arXiv:2609.23463v1 Announce Type: new  Abstract: Federated learning lets multiple parties train a shared model without pooling their data, but a client with far less data than the others can end up poorly served even when the group's average accuracy looks fine. We ask whether the choice of pretrained backbone affects this under LoRA fine-tuning, and whether per-word perplexity on the target text predicts which backbone helps the worst-off client before federated training starts. We ran 313 experiments across three text-classification datasets and three similarly sized backbones (RoBERTa, BERTweet, PubMedBERT), each compared against a task-specific baseline on identical data splits. Lower-perplexity backbones consistently produced larger gains for the worst-performing client, with a rank correlation of -0.87 across nine dataset-backbone pairs; a backbone held out of the analysis confirmed the pattern. Personalization with Ditto recovered only 4-12% of the gap between training alone and
    
[^87]: 面向非言语发声感知语音识别的长尾再平衡：NVVSpeech挑战赛Track 1系统

    Long-Tail Rebalancing for Non-Verbal Vocalization-Aware ASR: A Track~1 System for the NVVSpeech Challenge

    [https://arxiv.org/abs/2609.23462](https://arxiv.org/abs/2609.23462)

    本系统通过跨数据集标签统一和“平方根类别采样+均匀类别微调”的两阶段采样调度来缓解非言语发声数据的长尾不平衡问题，在NVVSpeech挑战赛Track 1中获得第四名。

    

    非言语发声承载着重要的副语言信息，但通常被传统自动语音识别（ASR）系统所忽略。ISCSLP NVVSpeech挑战赛要求在有限且高度不平衡的监督条件下，联合转录词汇内容和16类非言语发声。我们提出了一种以数据为中心的NVV感知ASR流程，其基础是跨数据集标签统一和两阶段采样调度。我们将异构的源标签映射到官方分类体系，并排除无法可靠映射的样本。我们的调度方案首先使用平方根类别采样来缓解长尾分布，随后应用均匀类别微调。在固定的本地验证集划分上，平方根类别采样在所测试的单阶段设置中表现最佳。最终的两阶段系统获得了63.86的官方分数，在Track 1中排名第四。

    arXiv:2609.23462v1 Announce Type: cross  Abstract: Non-verbal vocalizations (NVVs) carry important paralinguistic information but are often omitted by conventional automatic speech recognition (ASR) systems. The ISCSLP NVVSpeech Challenge requires joint transcription of lexical content and 16 NVV categories under limited and highly imbalanced supervision. We present a data-centric NVV-aware ASR pipeline based on cross-dataset label harmonization and a two-stage sampling schedule. We map heterogeneous source labels to the official taxonomy and exclude samples without a reliable mapping. Our schedule first uses square-root category sampling to moderate the long-tailed distribution and then applies uniform-category fine-tuning. On a fixed local validation split, square-root category sampling performs best among the tested single-stage settings. The final two-stage system obtains an official score of 63.86 and ranks fourth in Track 1.
    
[^88]: PSD：面向大语言模型智能体的记忆表征能力伪自蒸馏

    PSD: Pseudo Self-Distillation of Memory Representation Capabilities for LLM Agents

    [https://arxiv.org/abs/2609.23449](https://arxiv.org/abs/2609.23449)

    提出伪自蒸馏框架PSD，使小型语言模型无需访问闭源大模型的logits或隐藏状态，仅通过提示渠道引入黑盒oracle知识，即可实现单模型蒸馏并构建分层记忆表征，从而大幅降低记忆增强智能体的部署成本。

    

    记忆系统正在成为大语言模型（LLM）智能体的核心组件，但构建和维护记忆的成本依然高昂，因为它依赖于对大型专有语言模型的反复调用。这一成本成为大规模部署记忆增强智能体的主要障碍。本文提出了伪自蒸馏（Pseudo Self-Distillation，PSD），这是一个使小型语言模型（SLM）能够通过多阶段训练流程、从强大的黑盒oracle中蒸馏行为，从而构建分层记忆表征的框架。标准的蒸馏方法需要访问教师模型的logits或隐藏状态，而闭源模型并不会暴露这些信息。与传统的自蒸馏设置（其监督信号来自模型自身的预测、采样轨迹或聚合输出）不同，PSD在实现单模型蒸馏设置的同时，通过提示（prompt）这一渠道引入外部oracle的知识。PSD让一个小型模型承担两种角色：……

    arXiv:2609.23449v1 Announce Type: cross  Abstract: Memory systems are becoming a core component of LLM agents, but constructing and maintaining memory remains expensive because it relies on repeated calls to large proprietary language models. This cost creates a major barrier to deploying memory-enhanced agents at scale. In this paper, we present Pseudo Self-Distillation (PSD), a framework that enables small language models (SLMs) to construct hierarchical memory representations by distilling behavior from a strong black-box oracle through a multi-stage training pipeline. Standard distillation methods require access to teacher logits or hidden states, which closed models do not expose. Unlike conventional self-distillation settings, where supervision is derived from a model's own predictions, sampled rollouts, or aggregated outputs, PSD enables a single-model distillation setup while channeling external oracle knowledge through the prompt. PSD uses a single small model in two roles: a 
    
[^89]: 面向基于序列的组学任务中LLM领域自适应的工具增强在线策略蒸馏方法

    Tool-Augmented On-Policy Distillation for LLM Domain Adaptation in Sequence-Based Omics Tasks

    [https://arxiv.org/abs/2609.23435](https://arxiv.org/abs/2609.23435)

    该论文提出了首个多组学序列推理基准OmicsBench（包含1160个专家验证问题），发现科学领域LLM虽然在分类准确率上优于通用LLM，但在提供有效生物学证据链的推理能力上反而表现不佳。

    

    多组学序列包含复杂的生物学模式，然而为其机制进行解码以实现自动化科学发现仍然充满挑战。随着大语言模型（LLM）开始解读这些序列，同时评估其预测结果和科学推理能力变得至关重要。然而，现有的多组学序列任务基准依赖于分类和回归指标，忽略了模型是否真正掌握底层的生物学证据。我们提出了OmicsBench，这是首个针对多组学序列的推理基准，包含1,160个经过专家验证的问题，涵盖DNA调控、RNA加工和蛋白质功能三大领域的六项任务。OmicsBench要求模型提供可追溯的证据链，并使用与领域专家共同开发的针对具体实例的评分标准进行评估。对17个大语言模型的评估揭示了一种反向关系：尽管科学领域专用LLM在序列分类准确率上优于通用LLM，但它们未能提供有效的证据……

    arXiv:2609.23435v1 Announce Type: cross  Abstract: Multi-omics sequences contain complex biological patterns, yet deciphering their mechanisms for automated scientific discovery remains challenging. As large language models (LLMs) interpret these sequences, evaluating both predictions and scientific reasoning is critical. However, existing benchmarks for multi-omics sequence tasks rely on classification and regression metrics, neglecting whether models grasp the underlying biological evidence. We introduce OmicsBench, the first reasoning benchmark for multi-omics sequences, comprising 1,160 expert-validated questions across six tasks spanning DNA regulation, RNA processing, and protein function. OmicsBench requires traceable evidence chains, evaluated using instance-specific rubrics developed with domain experts. Evaluating 17 LLMs reveals an inverse relationship: while scientific LLMs outperform general-purpose LLMs in sequence classification accuracy, they fail to provide valid evide
    
[^90]: MuLA-Bench：通过多层次审计构建的多语言长音频理解基准

    MuLA-Bench: A Multilingual Long-Form Audio Understanding Benchmark via Multi-Tier Auditing

    [https://arxiv.org/abs/2609.23416](https://arxiv.org/abs/2609.23416)

    提出MuLA-Bench多语言长音频理解基准，通过多层次审计构建了覆盖16种语言、1,377.9小时真实录音的5,038个可审计开放式问答数据，并揭示了音频-语言模型在语言、领域和任务维度上的性能差异规律。

    

    长音频任务的性能通常通过上下文长度和总体准确率来概括，这掩盖了语言、证据和任务如何共同塑造难度。我们提出了MuLA-Bench：包含5,038个开放式问题，基于1,769个真实场景录音，总时长1,377.9小时，覆盖16种语言和八个领域。均衡的“语言×领域”语义赛道支持受控比较，而互补的声学赛道保留了自然出现的非语音证据。基于证据的生成、捷径检查和语言专家审核提供了可审计的问题，无需翻译共享源集或注入目标声音。我们评估了十个音频-语言模型，并对固定的八模型队列进行了汇总诊断。结果显示：语言排名在不同领域和任务间发生变化；声学与语义性能差距随所请求的操作而变化；时间性错误在正确事件被识别后仍可能持续存在。

    arXiv:2609.23416v1 Announce Type: cross  Abstract: Long-form audio performance is often summarized by context length and aggregate accuracy, obscuring how language, evidence, and task jointly shape difficulty. We introduce MuLA-Bench: 5,038 open-ended questions over 1,769 in-the-wild recordings totaling 1,377.9 hours, covering 16 languages and eight domains. A balanced Language x Domain semantic track supports controlled comparisons, while a complementary acoustic track preserves naturally occurring non-speech evidence. Evidence-grounded generation, shortcut checks, and language-expert review provide auditable questions without translating a shared source set or injecting target sounds. We evaluate ten audio-language models and conduct pooled diagnostics on a fixed eight-model cohort. Language rankings change across domains and tasks; acoustic-semantic performance gaps vary with the requested operation; and temporal errors can persist after the correct event is identified. Long-range r
    
[^91]: 一对多，多对一：面向软件工程智能体的类别感知迭代专家训练

    One to More, More to One: Category-Aware Iterative Expert Training for Software Engineering Agents

    [https://arxiv.org/abs/2609.23377](https://arxiv.org/abs/2609.23377)

    针对软件工程智能体强化学习中不同任务类别“此消彼长”的跷跷板问题，本文提出类别感知的专家训练与策略整合框架，通过SWE Labeler多轴证据标注、同源类别专家强化学习与RRE（刷新-修复-扩展）迭代机制，实现成功行为的显式巩固和策略自适应的任务选择，从而均衡提升各任务类别的表现。

    

    仓库级软件工程（SWE）包含异构的任务类别，在池化的智能体强化学习下，其进展可能不均衡：某些类别的提升伴随着其他类别的退化，而总体解决率掩盖了这些变化。受这种“类别跷跷板”现象的启发，我们开发了一个类别感知的专家训练与策略整合框架。可执行任务构建和SWE Labeler（一种基于证据的多轴标注系统）用于组织训练池。初始的类别特定强化学习提升了平均训练成功率，但仍存在实例级进展不均衡的问题，这促使我们显式地巩固成功行为并进行策略自适应的任务选择。同源类别专家交替进行长时程Agentic-miniRL与“刷新-修复-扩展”（RRE）流程：更新后的策略刷新实例掌握度，将其自身验证过的成功轨迹复用于修复式监督微调，并为后续训练重新选择任务……

    arXiv:2609.23377v1 Announce Type: cross  Abstract: Repository-level software engineering (SWE) comprises heterogeneous task categories, whose progress under pooled agentic reinforcement learning can be uneven: gains in some categories coincide with regressions in others, while aggregate resolution obscures these changes. Motivated by this category see-saw, we develop a category-aware expert-training and policy-integration framework. Executable task construction and SWE Labeler, an evidence-grounded multi-axis labeling system, organize the training pools. Initial category-specific RL improves average training success while leaving uneven instance-level progress, motivating explicit consolidation of successful behavior and policy-adaptive task selection. Same-origin category experts alternate long-horizon Agentic-miniRL with Refresh-Repair-Expand (RRE): the updated policy refreshes instance mastery, reuses its own verified successful trajectories for Repair SFT, and reselects tasks for f
    
[^92]: 机器可解释信息：将文档编译为可搜索且可读取的协议状态

    Machine-Interpretable Information: Compiling Documents into Searchable and Readable Protocol States

    [https://arxiv.org/abs/2609.23371](https://arxiv.org/abs/2609.23371)

    该论文提出首个智能体间文档到状态协议 MII，通过双时间尺度写入器将文档编译为 56 个 token 的固定带宽规范状态、再由轻量级翻译器适配任意冻结读取器，在单一可迁移媒介中统一检索、推理与重建，并将查询成本从 O(N²) 降至 O(K)。

    

    长上下文语言模型通过原始自然语言与外部知识进行交互。在检索增强系统中，这造成了一个持久的“索引-载荷”割裂问题：稠密向量虽能支持可搜索的路由，但模型必须以 O(N²) 的注意力成本重新摄取冗长的文本载荷才能进行推理；而现有的压缩方法所产生的私有状态又往往绑定于特定架构。我们提出了机器可解释信息，这是首个智能体到智能体（A2A）的文档到状态协议。一个双时间尺度的状态空间“写入器”将文档编译为规范的、固定带宽的状态（56 个 token），随后一个轻量级“翻译器”将其映射到任意冻结的“读取器”的嵌入空间中，从而将查询时的成本降至 O(K)。由此产生的 .mii 工件在单一可迁移媒介中统一了检索（可搜索的几何结构）、推理（全局记忆）与重建（有据可依的细节）。我们展示了强大的跨模型互操作性……（摘要被截断）

    arXiv:2609.23371v1 Announce Type: new  Abstract: Long-context language models interface with external knowledge through raw natural language. In retrieval-augmented systems, this creates a persistent index-payload schism: dense vectors enable searchable routing, but models must re-ingest lengthy text payloads for reasoning at O(N^2) attention cost. Existing compression methods further produce private states tied to specific architectures. We introduce Machine-Interpretable Information (MII), the first agent-to-agent (A2A) document-to-state protocol. A dual-timescale state-space Writer compiles documents into a canonical, fixed-bandwidth state (56 tokens), and a lightweight Translator maps it into any frozen Reader's embedding space, reducing query-time cost to O(K). The resulting .mii artifact unifies Retrieval (searchable geometry), Reasoning (global memory), and Reconstruction (grounded details) in a single transferable medium. We demonstrate strong cross-model interoperability acros
    
[^93]: 基于大语言模型的FORM代码生成与验证驱动的微调

    LLM-Based FORM Code Generation with Verification-Driven Fine-Tuning

    [https://arxiv.org/abs/2609.23367](https://arxiv.org/abs/2609.23367)

    该论文首次研究大语言模型为粒子物理符号计算语言FORM生成代码的问题，发现现有前沿模型零样本通过率为零，并提出利用FORM二进制程序作为执行预言机的验证驱动数据生成与微调流水线，构建了经过验证的训练语料库以提升模型生成可执行FORM代码的能力。

    

    FORM是一种领域专用的符号操作语言，广泛应用于粒子物理学中，用于处理多圈费曼图计算所产生的超大型代数表达式。尽管FORM在精确理论物理中占据核心地位，但据我们所知，目前尚不存在任何人工智能工具来协助物理学家编写FORM代码。我们证明，当代大语言模型（LLM），包括拥有数千亿参数的前沿模型，在无文档辅助的情况下，单次尝试执行我们提供的指令跟随和教程式FORM任务的通过率为零，从而确立了FORM在撰写本文时是LLM真正的零样本语言。随后，我们提出了一种验证驱动的数据生成流水线，该流水线利用FORM二进制程序本身作为执行预言机，生成并验证了一个包含4,633个训练样本的语料库，涵盖确定性计算、开放式程序等任务类型。

    arXiv:2609.23367v1 Announce Type: cross  Abstract: FORM is a domain-specific symbolic manipulation language widely used in particle physics for processing the very large algebraic expressions arising from multi-loop Feynman diagram calculations. Despite its central role in precision theoretical physics, no artificial-intelligence tooling exists, to our knowledge, for assisting physicists in writing FORM code. We show that contemporary large language models (LLMs), including frontier models with hundreds of billions of parameters, achieve a zero-percent execution pass rate on our instruction-following and tutorial-style FORM tasks without documentation in a single attempt, establishing FORM as a genuine zero-shot language for LLMs at the time of writing. We then present a verification-driven data generation pipeline that uses the FORM binary itself as an execution oracle to produce and validate a corpus of 4,633 training examples spanning deterministic computations, open-ended programs,
    
[^94]: 知道何时信任图像：可靠性感知的多模态实体对齐

    Knowing When to Trust Images: Reliability-Aware Multi-modal Entity Alignment

    [https://arxiv.org/abs/2609.23267](https://arxiv.org/abs/2609.23267)

    提出了一种可靠性感知的多模态实体对齐框架RA-MMEA，通过依赖感知的视觉可靠性预测和稳定性正则化的视觉嵌入生成两个模块，评估图像可靠性并自适应改进不可靠的视觉表示，从而解决图像噪声与语义不对齐导致的融合性能下降问题。

    

    视觉模态（即图像）在多模态实体对齐（MMEA）中起着关键作用。现有方法通常直接将图像与其他模态融合来对齐不同实体。尽管简单，但这类策略忽略了图像中潜在的噪声以及图像与相应实体之间的语义不对齐问题，导致融合效果欠佳、性能下降。针对这一问题，我们提出了一种新颖的可靠性感知的多模态实体对齐框架（RA-MMEA），该框架评估视觉可靠性，并自适应地改进不可靠的视觉表示，以实现鲁棒的实体对齐。其核心在于两个模块：依赖感知的视觉可靠性预测（DA-VRP）和稳定性正则化的视觉嵌入生成（SR-VEG）。前者旨在利用实体内部的多模态依赖关系来估计图像的可靠性，后者则专注于生成以……为条件的替代视觉表示（摘要原文在此处截断）。

    arXiv:2609.23267v1 Announce Type: new  Abstract: The visual modality, i.e., images, plays a key role in multi-modal entity alignment (MMEA). Existing approaches often directly fuse the image with other modalities to align different entities. Although simple, such strategies overlook the potential noise in the images and their semantic misalignment with corresponding entities, resulting in suboptimal fusion and degraded performance. Addressing this, we propose a novel Reliability-Aware framework for MMEA (RA-MMEA), which assesses visual reliability and adaptively improves unreliable visual representations for robust entity alignment. The core lies in two modules, including dependency-aware visual reliability prediction (DA-VRP) and stability-regularized visual embedding generation (SR-VEG). The former aims to estimate the reliability of an image by leveraging multi-modal dependency within the entity, while the latter focuses on producing alternative visual representation conditioned on 
    
[^95]: 以封面论审稿：基于大语言模型的同行评审评估指标的可靠性分析

    Judging a Review by its Cover: A Reliability Analysis of LLM-based Peer Review Evaluation Metrics

    [https://arxiv.org/abs/2609.23264](https://arxiv.org/abs/2609.23264)

    该研究提出了一个统计框架，通过比较原始人类审稿意见与保留相同评审内容但改变表达方式的LLM改写版本，检验基于大语言模型的同行评审评估指标是否真正衡量实质性审稿质量，而非仅凭表面的语言形式打分。

    

    同行评审评估正日益通过“大语言模型作为裁判”（LLM-as-a-judge）的指标实现自动化，但这带来了测量风险：一篇审稿意见可能因为语言流畅、结构清晰、表述精炼而获得高分，而非因为它对论文提供了强有力的评价。这一风险在AI辅助审稿中尤为重要，因为审稿人可能会使用大语言模型来改善清晰度或表达方式，同时保留其原有的评审判断。我们提出了一个统计框架，用于检验同行评审评估指标是否能够超越表面语言形式、捕捉到实质性的审稿质量。该框架将原始的人类审稿意见与忠实的LLM改写版本进行比较，这些改写版本在改变措辞和表达方式的同时保留了相同的评审内容。我们使用一个包含4,044条语义保持改写的数据集（源自ICLR和NeurIPS的674份人类审稿意见），评估了来自四项先前研究的29种面向内容的同行评审评估指标。

    arXiv:2609.23264v1 Announce Type: new  Abstract: Peer-review evaluation is increasingly being automated with LLM-as-a-judge metrics, but this creates a measurement risk. A review may receive a high score because it is fluent, organized, and polished, rather than because it provides a strong evaluation of the paper. This risk is especially important in AI-assisted reviewing, where reviewers may use LLMs to improve clarity or presentation while preserving the underlying judgments. We propose a statistical framework for testing whether peer-review evaluation metrics capture substantive review quality beyond surface-level linguistic form. The framework compares original human reviews with faithful LLM rewrites that preserve the same evaluative content while changing wording and presentation. Using a dataset comprising 4,044 meaning-preserving rewrites derived from 674 human reviews from ICLR and NeurIPS, we evaluate 29 content-oriented peer-review evaluation metrics drawn from four prior w
    
[^96]: CTRL：基于控制的时间序列预测与LLM引导的残差学习

    CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning

    [https://arxiv.org/abs/2609.23257](https://arxiv.org/abs/2609.23257)

    CTRL框架将语义推理与定量预测解耦，利用LLM智能体作为控制器分析预测误差的分解成分并输出控制信号，再由轻量级残差解码器转化为预测修正，从而提升非平稳环境下时间序列预测的稳定性与可解释性。

    

    时间序列预测是跨多个领域关键决策的基础。尽管大语言模型（LLM）提供了有前景的推理能力，但现有的基于LLM的时间序列预测方法要么将其简化为绕过其优势的数值预测器，要么允许直接生成预测，导致在非平稳环境中预测不稳定。我们提出了CTRL，一个将语义推理与定量预测解耦的框架。冻结的主干模型生成基础预测，而专门的LLM智能体充当控制器，通过分解的趋势、季节性和不规则成分来分析主干预测误差，将推理建立在可解释的时间结构之上。每个智能体输出紧凑的控制信号，由轻量级残差解码器将其转化为预测修正。CTRL还集成了无标签的测试时自适应机制，可从输入统计信息中检测分布偏移。

    arXiv:2609.23257v1 Announce Type: cross  Abstract: Time series forecasting underpins critical decision-making across diverse domains. While large language models (LLMs) offer promising reasoning capabilities, existing LLM-based time series forecasting approaches either reduce them to numerical predictors that bypass their strengths, or allow direct forecast generation that destabilizes predictions in non-stationary settings. We introduce CTRL, a framework that decouples semantic reasoning from quantitative prediction. A frozen backbone generates base forecasts, while specialized LLM agents function as controllers that analyze backbone prediction errors through decomposed trend, seasonal, and irregular components, grounding reasoning in interpretable temporal structure. Each agent outputs compact control signals that a lightweight residual decoder translates into forecast corrections. CTRL incorporates label-free test-time adaptation that detects distribution shift from input statistics
    
[^97]: SoK：事实核查与信息完整性的形式化方法

    SoK: Formal Methods for Fact-Checking and Information Integrity

    [https://arxiv.org/abs/2609.23239](https://arxiv.org/abs/2609.23239)

    该论文以“担保”概念为核心，按被形式化的对象（声明、推理、系统、生态系统等五个层次）而非流水线阶段来系统化形式化方法在事实核查与信息完整性领域的应用，以满足《数字服务法》和《人工智能法》对可审计证据的监管需求。

    

    自动事实核查系统会返回一个标签：该声明为真，或为假。在许多此类系统中，裁决仍然是最主要的输出。而通常缺失的是这样一份记录：是哪份文件解决了这个问题、裁决若要改变需要什么条件有所不同、或者同一声明换一种措辞是否会得到相同的判断。我们将这一缺失的部分称为“担保”（warrant）：一份单独的陈述，说明保证了什么以及基于何种依据。形式化方法能够产生这类证据，而监管也正开始要求提供这类证据，因为《数字服务法》和《人工智能法》都要求提供关于系统行为的可审计证据。现有的自动事实核查综述通常按流水线阶段来组织，将逻辑视为众多技术中的一种。我们则按照被形式化的对象来组织这一领域，由此得到五个层次：声明本身、推理过程、执行核查的系统、声明所处的生态系统……

    arXiv:2609.23239v1 Announce Type: new  Abstract: An automated fact-checking system returns a label: the claim is true, or it is false. In many such systems the verdict remains the primary output. What is generally missing is a record of which document settled the question, of what would have had to be different for the verdict to change, or of whether the same claim, reworded, would have been judged the same way. We call the missing piece a warrant: a separate statement of what was guaranteed and on what grounds. Formal methods produce evidence of this kind, and regulation is beginning to ask for it, since the Digital Services Act and the AI Act both call for auditable evidence about how systems behave. Surveys of automated fact-checking are usually organised by pipeline stage, and treat logic as one technique among many. We organise the field by what is being formalised instead, which gives five levels: the claim, the reasoning, the system doing the checking, the ecosystem the claim s
    
[^98]: ChemCLIR-Bench：多语言化学专利中跨语言信息检索的基准测试

    ChemCLIR-Bench: Benchmarking Cross-Lingual Information Retrieval in Multilingual Chemical Patents

    [https://arxiv.org/abs/2609.23231](https://arxiv.org/abs/2609.23231)

    该论文提出了ChemCLIR-Bench，一个基于Google Patents和EPO数据构建的、涵盖五种语言的多语言化学专利跨语言信息检索基准，并通过对八个最先进嵌入模型的系统评估，揭示了单语与跨语言检索之间的显著性能差距。

    

    跨语言信息检索（CLIR）在跨国产业中日益重要，因为在这些产业中，关键的技术证据可能以与查询不同的语言存在。然而，现有的基准测试未能充分捕捉特定领域的跨语言检索，也未能捕捉聚合召回率所掩盖的检索深度和可恢复性方面的失败。在这项工作中，我们对化学领域的跨语言信息检索进行了基准测试，重点关注专利数据。我们基于Google Patents和欧洲专利局（EPO）的数据构建了一个多语言数据集，涵盖五种语言（包括主要的东方和西方语言），并反映了真实世界工业文档的多样性和复杂性。利用该数据集，我们系统地评估了八个最先进的嵌入模型在跨语言检索中的表现。我们的结果显示，单语和跨语言设置之间存在显著的性能差距：对于表现最好的模型，Recall@1（摘要内容在此处被截断）

    arXiv:2609.23231v1 Announce Type: new  Abstract: Cross-lingual information retrieval (CLIR) is increasingly important in multi-national industries, where critical technical evidence may exist in a different language than the query. However, existing benchmarks do not adequately capture domain-specific cross-lingual retrieval or the retrieval-depth and recoverability failures that aggregate recall hides.   In this work, we benchmark CLIR in the chemical domain, with a focus on patent data. We construct a multilingual dataset from Google Patents and the European Patent Office (EPO) data, spanning five languages (covering major Eastern and Western languages) and reflecting the diversity and complexity of real-world industrial documentation.   Using this dataset, we systematically evaluate eight state-of-the-art embedding models for cross-lingual retrieval. Our results show a substantial performance gap between monolingual and cross-lingual settings: for the best-performing model, Recall@1
    
[^99]: Euston：在消除数学谄媚性的同时不损失数学能力

    Euston: Training Away Mathematical Sycophancy Without Losing the Mathematics

    [https://arxiv.org/abs/2609.23205](https://arxiv.org/abs/2609.23205)

    该论文提出Euston，一个通过GraphSynth生成器构建真假陈述对数据、并用GRPO强化学习微调DeepSeek-R1-8B得到的8B数学论断验证模型，使其学会拒绝证明被篡改的错误定理，将平衡准确率从29.50%提升至63.75%。

    

    推理型语言模型被训练用于产出解答而非拒绝解答，当它们面对的问题是错误的时候，这种偏差依然存在。当被要求证明一个被篡改的定理时，一个强大的模型通常会照做，并自信地推导出错误的东西。我们提出了Euston，一个经过训练以抵抗这种行为的8B数学论断验证模型。训练数据由GraphSynth生成——这是一个概率因子图生成器，将属性级别的多样性与解码时的结构掩码和跨度同步验证相结合——从2010年至2025年的arXiv论文中提取出3,026对匹配的真/损坏陈述对（共6,052条陈述）。我们在基于规则的零API奖励下，使用GRPO在四块H100 GPU上对DeepSeek-R1-8B进行了189步的微调。在一个平衡的200真/200假保留测试集上，平衡准确率从29.50%提升至63.75%，且判别差距——即……（原文摘要在此处截断）

    arXiv:2609.23205v1 Announce Type: new  Abstract: Reasoning language models are trained to produce solutions, not to refuse them, and this bias persists when the problem they are handed is false. Asked to prove a corrupted theorem, a strong model will typically comply and produce a confident derivation of something untrue. We present Euston, an 8B mathematical claim-verification model trained to resist exactly this. Training data were generated with GraphSynth, a probabilistic factor-graph generator that couples attribute-level diversity to decode-time structural masking and span-synchronized verification, yielding 3{,}026 matched true/corrupted statement pairs (6,052 statements) drawn from arXiv papers spanning 2010--2025. We fine-tuned DeepSeek-R1-8B with GRPO under a rule-based, zero-API reward for 189 steps on four H100 GPUs. On a balanced 200-true/200-false held-out split, balanced accuracy rises from 29.50% to 63.75% and the discrimination gap---the difference between the rate of 
    
[^100]: 低资源场景下基于HGNN跨模态知识迁移增强语音表征学习：以Yemba语为例

    Enhancing speech representation learning with cross-modal knowledge transfer with HGNN under low resource settings: the case study of Yemba

    [https://arxiv.org/abs/2609.23194](https://arxiv.org/abs/2609.23194)

    本文提出基于异构图神经网络的跨模态知识迁移方法，将声学与语言学实体建模为统一图中的不同节点类型，通过消息传递机制让语言学节点向声学节点显式传递知识，从而有效增强低资源语言（如Yemba语）的声学表征学习。

    

    声学表征学习对语音处理至关重要，然而低资源语言（LRLs）面临严重的数据稀缺问题，这限制了传统方法和自监督方法的有效性。作为一种有前景的替代方案，在本工作中，我们提出通过基于异构图神经网络（HGNNs）的跨模态知识迁移方法来增强声学表征，其中声学实体和语言学实体被建模为统一图中的不同节点类型。通过消息传递机制，语言学节点显式地向声学节点传递知识，实现结构化且可解释的跨模态信息流动。为了凸显这种知识迁移及其带来的益处，我们采用标准聚类指标作为声学表征的内在评估；为了强调实用性，我们使用英语基准数据集和一种喀麦隆语言数据集执行了孤立词识别任务。

    arXiv:2609.23194v1 Announce Type: new  Abstract: Acoustic representation learning is crucial for speech processing, yet low-resource languages (LRLs) face severe data scarcity, limiting the effectiveness of traditional and self-supervised methods. As a promising alternative, in this work, we propose to enhance acoustic representation trough a cross-modal transfer knowledge approach, based on heterogeneous graph neural networks (HGNNs), where acoustic and linguistic entities are modeled as distinct node types within a unified graph. Through message-passing mechanisms, linguistic nodes explicitly transfer knowledge to acoustic nodes, enabling structured and interpretable cross-modal information flow. To highlight this knowledge transfer and its benefits, we measured standard clustering metrics as an intrinsic evaluation of acoustic representation, and to emphasize applicability, we performed isolated-word recognition tasks using an English benchmark and a Cameroonian language dataset in 
    
[^101]: 大语言模型作为语言变色龙：面向隐私保护通信的语义与结构解耦

    LLMs as Linguistic Chameleons: Decoupling Semantics and Structure for Privacy-Preserving Communication

    [https://arxiv.org/abs/2609.23193](https://arxiv.org/abs/2609.23193)

    提出CROSS-MAP双向框架，通过语义解耦在推理前将私有输入映射到不同语义域、推理后再恢复输出，在保护隐私的同时不损害LLM的任务效用。

    

    随着大语言模型（LLM）API日益融入隐私敏感的工作流程，在不损害任务效用的前提下确保推理时的隐私性仍然是一个重大挑战。现有方法为了保持下游性能而保留了大部分原始语义内容，但这也留下了可被利用来重构原始文本的线索。本工作研究了语义解耦，即用替代内容替换原始语义，同时保留LLM推理所需的结构。基于这一思想，我们提出了CROSS-MAP，这是一个双向框架，在推理前将私有输入映射到不同的语义域，并在推理后恢复相应的输出。本地模型通过多目标优化进行训练，以在映射阶段最大化语义差异，同时在恢复阶段最小化语义不一致性。实验表明，CROSS-MAP降低了文本重构风险，同时保持了任务性能。

    arXiv:2609.23193v1 Announce Type: cross  Abstract: As Large Language Model (LLM) APIs become increasingly integrated into privacy-sensitive workflows, ensuring inference-time privacy without compromising task utility remains a major challenge. Existing approaches preserve most of the original semantic content to maintain downstream performance, but this also leaves exploitable cues for reconstructing the original text. This work investigates semantic decoupling, which replaces original semantics with alternative content while preserving the structure needed for LLM reasoning. Based on this idea, we propose CROSS-MAP, a bidirectional framework that maps private inputs into a different semantic domain before inference and recovers the corresponding outputs afterward. Local models are trained with multi-objective optimization to maximize semantic divergence in the mapping stage while minimizing semantic inconsistency in the recovery stage. Experiments show that CROSS-MAP reduces reconstru
    
[^102]: 使用HGNN进行低资源跨模态对齐以增强语音表示

    Low resource cross-modal alignment using HGNN to enhance speech representation

    [https://arxiv.org/abs/2609.23191](https://arxiv.org/abs/2609.23191)

    该论文提出一种基于异构图神经网络和链接预测的数据高效语音-文本对齐方法，通过消息传递将文本信息显式传递给语音模态，从而在低资源和计算受限条件下增强语音表示。

    

    语音-文本空间对齐是一种多模态表示学习方法，旨在将语音和文本映射到一个共享的表示空间中，从而使每个模态的表示得到丰富。现有的架构（如SAMU-XLSR）通常采用学生/教师框架，其目标是对音频编码器进行微调，使其产生的表示与文本表示紧密匹配，从而实现语义上更加丰富的语音表示。然而，此类系统通常需要大量训练数据和可观的计算资源，使其难以在资源受限的条件下应用于低资源语言。本工作提出了一种基于异构图神经网络和链接预测的数据高效空间对齐方法。其核心思想是利用消息传递机制将文本模态的信息显式地传递到语音模态，从而减少……

    arXiv:2609.23191v1 Announce Type: new  Abstract: Speech-text space alignment is a multimodal representation learning method consisting to map different speech and text into a shared representation space, leading to enrichment of the representation of each modality. Proposed architectures, such as SAMU-XLSR, typically follow a student/teacher framework, with the goal of fine-tuning an audio encoder to produce representations that closely match those of the text. In this way a speech representation is semantically enriched. However, such systems generally require large amounts of training data and considerable computational resource, making them difficult to apply to low resources languages under frugal constraints. The present work proposes a data-efficient space alignment method based on Heterogeneous Graph Neural Networks and link prediction. The core idea is to leverage message passing to explicitly transfer information from the text modality to the speech modality, thereby reducing 
    
[^103]: Chronologic：衡量语言模型表征过去的能力

    Chronologic: Measuring Language Models' Ability to Represent the Past

    [https://arxiv.org/abs/2609.23178](https://arxiv.org/abs/2609.23178)

    该论文提出了Chronologic基准，利用1831-1930年的历史文本评估语言模型表征过去的能力，发现生成式任务比判别式任务更难、仅用历史文本预训练的模型在似然评估中领先但在自由生成上不敌商业模型，且没有任何受测模型能可靠地表征过去。

    

    语言模型是研究过去的有吸引力的工具。但为了信任模型提供的证据，研究人员需要知道其回答是否符合所代表的时期。验证工作具有挑战性，因为这并非在世的人通常执行的任务，而且许多问题存在多个正确答案。我们利用历史文本开发了一个基准，用于评估模型对1831-1930年英语语境的表征能力，依靠与多个真实答案和强干扰项的成对比较，以适当的分级方式对最难的问题进行评分。我们发现生成式任务比判别式任务更难；事实上，推理模型通常能够察觉自身生成答案的弱点。虽然仅在历史文本上预训练的模型在按答案似然评估时表现领先，但它们在自由生成方面无法与商业模型竞争。我们测试的所有模型都未能……

    arXiv:2609.23178v1 Announce Type: new  Abstract: Language models are appealing tools for research on the past. But to trust the evidence a model provides, researchers need to know whether its responses fit the period represented. Validation is challenging, because this is not a task living people ordinarily perform, and because many questions have multiple correct answers. We use historical texts to develop a benchmark for a model's representation of English-language contexts 1831-1930, relying on pairwise comparisons to multiple ground truths and strong distractors to score the hardest questions in an appropriately graduated way. We find that generative tasks are harder than discriminative ones; in fact, reasoning models can typically discern the weakness of their own generated answers. While models pretrained exclusively on historical text lead the pack when evaluated by answer likelihood, they cannot compete with commercial models in free generation. None of the models we tested rep
    
[^104]: OmniEdu：面向学习与教学的开放基础模型

    OmniEdu: Open Foundation Models for Learning and Teaching

    [https://arxiv.org/abs/2609.23088](https://arxiv.org/abs/2609.23088)

    OmniEdu是一个面向K-12学习与教学的开放基础模型家族，其核心创新在于围绕学科能力、课程对齐、诊断推理和教学支架四种能力来组织指令微调语料（69,999个样本、1596万token），使模型同时具备解题与教学辅导能力。

    

    教育基础模型必须能够解决问题、理解课程结构、诊断学习者的困难，并提供适当的教学支持。现有的教育语言模型往往只专注于解题或辅导中的某一方面，其训练数据混合方式按来源或任务组织，而非按能力组织。我们提出了OmniEdu，这是一个面向K-12学习与教学的开放基础模型家族。其指令微调语料库融合了100多个教育资源与通用指令来源，围绕四种能力进行组织：学科能力、课程对齐、诊断推理，以及教学行动与支架式引导。我们的数据处理流程集成了确定性清洗、语义审计与改写、任务特定的质量评分、基于token预算的多样性选择，以及教学指令分配。该流程共产出69,999个样本和1596万监督响应token，其中包括60,951个教育领域特定样本……

    arXiv:2609.23088v1 Announce Type: new  Abstract: Educational foundation models must solve problems, understand curriculum structure, diagnose learner difficulties, and provide appropriate instructional support. Existing educational language models often focus on either problem solving or tutoring, with training mixtures organized by source or task rather than capability. We present OmniEdu, an open family of foundation models for K-12 learning and teaching. Its instruction-tuning corpus combines over 100 educational resources and general instruction sources, organized around four capabilities: subject competence, curriculum grounding, diagnostic reasoning, and pedagogical action and scaffolding. Our pipeline integrates deterministic cleaning, semantic auditing and rewriting, task-specific quality scoring, token-budgeted diversity selection, and pedagogical instruction assignment. It yields 69,999 examples and 15.96M supervised response tokens, including 60,951 education-specific exampl
    
[^105]: 引导大语言模型遵循法律的字面意义或精神

    Directing large language models to follow the letter or spirit of the law

    [https://arxiv.org/abs/2609.23083](https://arxiv.org/abs/2609.23083)

    该研究通过定向适配方法使大语言模型能够优先遵循法律的精神或字面意义，并通过模型内部分析揭示了法律概念在低维空间中的可解释几何结构。

    

    法律精神与法律字面意义之间的区别是研究和日常生活中的核心问题，也是构建安全、智能机器日益受到关注的问题。这种区别基于什么？我们如何开发出能够遵循规则背后意图的机器？我们使用了定向适配方法，使大语言模型优先考虑法律的精神或字面意义。通过极少的修改，我们的方法在多种测量方式、新颖情景、现实场景和具有影响力的法律案例中显著改变了大语言模型的行为。对模型内部的分析揭示了一个低维空间，其中包含三个可解释的维度，与预先指定的法律概念几何学形式框架相匹配。这些发现展示了大语言模型中的法律思维如何被组织和引导。

    arXiv:2609.23083v1 Announce Type: new  Abstract: The distinction between the spirit and letter of the law is a central issue across research and everyday life, and a growing concern for building safe, intelligent machines. What is this distinction based on, and how can we develop machines that follow the intention behind a rule? We used targeted adaptation that made large language models prioritize the spirit or letter of the law. With minimal modifications, our method significantly changed LLM behavior across diverse measures, novel vignettes, real-world scenarios, and influential legal cases. An analysis of model internals revealed a low-dimensional space with three interpretable dimensions matching a formal pre-specified framework for the geometry of legal concepts. These findings show how legal thought in LLMs may be organized and directed.
    
[^106]: 教导大语言模型具备领域适应性、精确性与安全性

    Tutoring Large Language Models to be Domain-adaptive, Precise and Safe

    [https://arxiv.org/abs/2609.23071](https://arxiv.org/abs/2609.23071)

    本论文提出“负责任智能”框架，通过主动学习与图知识减少幻觉、解码时对齐机制实时拦截有害内容、以及语言特定引导保障文化与多语言安全，为构建领域适应、精确且安全的下一代AI提供蓝图。

    

    本论文提出了一个“负责任智能”框架，以应对人工智能在安全性、伦理和文化敏感性方面的关键挑战。该框架在三个核心领域取得进展：首先，利用主动学习和基于图的知识来改善专业领域的领域适应性，从而减少幻觉；其次，通过一种新颖的解码时对齐机制增强伦理严谨性，该机制能够实时主动阻止有害文本的生成；最后，通过尊重多元语言和社会规范的语言特定引导，确保文化和多语言层面的安全性。最终，这项工作为构建具备情境知识、伦理健全且文化适应性的下一代人工智能提供了蓝图。

    arXiv:2609.23071v1 Announce Type: cross  Abstract: This thesis proposes a framework for "responsible intelligence" to address AI's critical challenges in safety, ethics, and cultural sensitivity. It advances three core areas: First, it improves domain adaptation in specialized fields using active learning and graph-based knowledge to reduce hallucinations. Second, it enhances ethical rigor via a novel decoding-time alignment mechanism that proactively blocks harmful text generation in real-time. Finally, it ensures cultural and multilingual safety through language-specific steering that respects diverse linguistic and social norms. Ultimately, this work provides a blueprint for building next-generation AI that is contextually knowledgeable, ethically sound, and culturally adaptable.
    
[^107]: 从概念对齐到因果锚定：思维链忠实性的干预测试

    From Concept Alignment to Causal Grounding: An Intervention Test of Chain-of-Thought Faithfulness

    [https://arxiv.org/abs/2609.23065](https://arxiv.org/abs/2609.23065)

    该论文将CoT忠实性重新定义为“内部概念锚定”问题，利用共享稀疏自编码器直接比较预测过程与CoT过程的内部概念，并首次引入三个相关性对齐指标和一个因果消融指标Δp，以检验共享概念是否因果性地驱动模型答案。

    

    思维链可以听起来合理，却可能对模型的底层推理不忠实。以往大多数工作通过输入-输出行为或输入归因来探究CoT的忠实性，而对内部计算的探索在很大程度上仍属空白。我们转而将忠实性界定为内部概念锚定问题：大语言模型（LLM）的CoT推理是否调用了支持其直接预测的相同内部概念，并且这些共享概念是否因果性地驱动其答案？使用单个共享的稀疏自编码器（SAE）——一种对LLM所使用潜在概念的可靠近似器——来编码预测过程和CoT过程，使二者的内部概念可以直接比较。我们提出了三个概念层面的相关性对齐度量指标，以及一个因果度量指标Δp，该指标通过消融共享概念并测量答案概率的下降来检验因果作用。在五个LLM和四个数据集上的实验表明，概念对齐总体上较高，正如t……（摘要在此处截断）

    arXiv:2609.23065v1 Announce Type: new  Abstract: Chain-of-thought (CoT) can sound plausible yet be unfaithful to the model's underlying reasoning. Most prior work probes CoT faithfulness through input--output behavior or input attributions, leaving internal computation largely underexplored. We instead cast faithfulness as internal concept grounding: Does a large language model's (LLM) CoT reasoning engage the same internal concepts that support the LLM's direct prediction, and do the shared concepts causally drive its answer? Encoding a prediction pass and a CoT pass with a single shared sparse autoencoder (SAE), a reliable approximator of the latent concepts LLMs use, makes their internal concepts directly comparable. We introduce three correlational metrics of concept-level alignment and a causal metric, $\Delta p$, which ablates the shared concepts and measures the drop in answer probability. Across five LLMs and four datasets, concept alignment is generally high, as indicated by t
    
[^108]: 面向台湾历史问答的静态与智能体式RAG桥接研究

    Bridging Static and Agentic RAG for Taiwanese Historical Question Answering

    [https://arxiv.org/abs/2609.23056](https://arxiv.org/abs/2609.23056)

    该论文针对台湾历史问答发现智能体式与静态RAG总体性能相近但在70.83%的问题上表现各异，并提出一种比较两个回答及其引用证据的事后选择器，显著优于任一单一流水线并恢复了oracle提升空间的60.34%。

    

    智能体检索增强生成（RAG）使语言模型能够根据先前检索到的证据自适应地调整检索策略，但这种自适应编排是否始终优于设计良好的静态流水线仍不明确。我们针对台湾历史问答任务，在共享相同生成器和混合检索后端的条件下，对智能体式与静态RAG进行了受控比较。尽管两者总体性能相近，但两条流水线在70.83%的问题上表现不同，其各自优势在取平均时大体相互抵消。一个能为每个问题选出更优回答的理想选择器（oracle）比表现较好的单一流水线可将综合分数提高0.2417，揭示了问题级选择存在巨大提升空间。因此，我们引入了一种事后选择器，通过比较两个回答及其引用的证据进行选择，其性能显著优于任一单一流水线，并恢复了oracle提升空间的60.34%。

    arXiv:2609.23056v1 Announce Type: new  Abstract: Agentic retrieval-augmented generation (RAG) enables language models to adapt retrieval based on previously retrieved evidence, but it remains unclear whether such adaptive orchestration consistently outperforms well-designed static pipelines. We conduct a controlled comparison of agentic and static RAG for Taiwanese historical question answering, sharing the same generator and hybrid retrieval backend. Despite similar aggregate performance, the two pipelines differ on 70.83% of questions, with their advantages largely canceling out when averaged. An oracle that selects the better response per question improves the composite score by 0.2417 over the better individual pipeline, revealing substantial headroom for question-level selection. We therefore introduce a post-hoc selector that compares the two responses and their cited evidence, significantly outperforming either individual pipeline and recovering 60.34% of the oracle headroom. Th
    
[^109]: RAG引用中可归因的事后合理化：一项受控复现与RLVR对比研究

    Attributable Post-Rationalization in RAG Citations: A Controlled Reproduction and an RLVR Comparison

    [https://arxiv.org/abs/2609.23053](https://arxiv.org/abs/2609.23053)

    本研究通过受控复现实验发现，RLVR训练无法减少RAG系统中的事后合理化不忠实引用——RLVR智能体的不忠实引用率与其基础模型相当，奖励正确答案并不能提升引用的忠实度。

    

    RAG系统可以给你正确的答案，却引用一个它实际上并未使用的来源。模型通过事后合理化产生这些不忠实的引用：它们先写出答案，然后将引用附加到任何看起来足够接近的段落上。搜索智能体现在通过可验证奖励的强化学习（RLVR）进行训练，这种训练会因答对问题而给予奖励。我们探究了这种训练是否也能教会它们诚实地引用。通过引入必要的对照来改进现有方法，我们在四个问答数据集上，仅使用免费的Kaggle GPU，将一个指令微调模型与从它训练出的三个RLVR智能体进行比较。事后合理化无处不在：在基于维基百科的问题上，大约每七个引用中就有一个是不忠实的。RLVR并不能解决这个问题。这些智能体的事后合理化率与其基础模型相当，其中一个甚至略差。奖励正确答案在引用忠实度上毫无收益，因此……

    arXiv:2609.23053v1 Announce Type: new  Abstract: A RAG system can hand you the right answer and cite a source it did not actually use. Models output these unfaithful citations via post-rationalization: they write the answer first and then attach a citation to whatever passage looks close enough. Search agents are now trained with reinforcement learning from verifiable rewards (RLVR), which pays them for getting the answer right. We asked whether that training also teaches them to cite honestly.   Improving an existing methodology with a required control, we compared an instruction-tuned model against three RLVR agents trained from it, on four question-answering datasets, using only free-tier Kaggle GPUs. Post-rationalization is everywhere: on Wikipedia-based questions roughly one citation in seven is unfaithful. RLVR does not fix it. The agents post-rationalize at their base model's rate, and one lands slightly worse. Rewarding correct answers buys nothing in citation faithfulness, so 
    
[^110]: 通过结构化知识树在LLM驱动的侦探游戏中强制保障叙事可靠性与认知节奏

    Enforcing Narrative Reliability and Epistemic Pacing in LLM-Driven Detective Games via Structured Knowledge Trees

    [https://arxiv.org/abs/2609.23043](https://arxiv.org/abs/2609.23043)

    该论文提出结构化知识树与三智能体LLM流水线架构，通过分离知识检索、对话生成与响应验证，确保侦探游戏中的虚拟嫌疑人只透露当前叙事状态允许的信息，从而减少幻觉并保持作者对信息披露节奏的控制。

    

    大型语言模型（LLM）使交互式游戏中的开放式对话成为可能，但其非确定性的输出使得保持作者控制权、事实一致性以及预期的信息披露顺序变得困难。这些挑战在侦探游戏中尤为突出，因为过早的揭示或捏造的细节可能会破坏玩家推进剧情的逻辑。我们提出了一种结构化知识树架构，并结合三智能体LLM流水线，用于控制开放式审讯游戏中的对话。该系统将知识检索、对话生成和响应验证相互分离，以确保虚拟嫌疑人仅透露当前叙事状态所允许的信息。我们通过《审讯阿德里安·盖尔》——一个可玩的侦探游戏测试平台——以及一项正式的用户研究来评估该方法，研究考察了幻觉的减少、对预设披露顺序的遵循，以及……

    arXiv:2609.23043v1 Announce Type: cross  Abstract: Large Language Models (LLMs) enable open-ended dialogue in interactive games, but their non-deterministic outputs make it difficult to preserve authorial control, factual consistency, and the intended sequence of information disclosure. These challenges are particularly significant in detective games, where premature revelation or fabricated details can undermine the logic of player progression. We present a Structured Knowledge Tree architecture coupled with a tri-agent LLM pipeline for controlling dialogue in an open-ended interrogation game. The system separates knowledge retrieval, dialogue generation, and response verification to ensure that the virtual suspect reveals only information permitted by the current narrative state. We evaluate the approach through The Interrogation of Adrian Gale, a playable detective-game testbed, and a formal user study examining hallucination reduction, adherence to authored disclosure sequences, an
    
[^111]: 审计大语言模型助手的政治对齐：参与度、立场与用户身份

    Auditing Political Alignment in LLM Assistants: Engagement, Stance, and User Identity

    [https://arxiv.org/abs/2609.23039](https://arxiv.org/abs/2609.23039)

    该研究提出“言论体制”新框架，通过对六个主流AI系统共7,500个多轮对话的预注册实验，首次系统揭示了大语言模型的政治行为如何随话题敏感度和用户政治身份而动态变化。

    

    基于大语言模型的AI系统正在为数亿人回答政治问题。当前的审计方法衡量的是它们对“普通用户”所说的内容，但它们的行为实际上是动态的。本文作者认为，这些系统的政治行为是一组关于“回答谁、说什么、以及是否参与回答”的策略，这些策略取决于话题本身以及系统对用户的了解程度。作者将这套策略称为系统的“言论体制”（speech regime），即开发者如何在回答、迎合用户与拒绝之间进行权衡，而每一种选择都带有因话题而异的成本。作者从“参与度”和“立场”两个维度推导出五种言论体制的类型学。作者在一项预注册实验中测试了六个AI系统（OpenAI、Anthropic、xAI、Google、Mistral、DeepSeek），该实验包含7,500个多轮对话，随机分配用户的政治身份，涵盖五个话题：堕胎、加泰罗尼亚独立、气候变化、纳粹主义，以及一个零风险对照组（披萨上加菠萝……）（摘要原文在此处截断）。

    arXiv:2609.23039v1 Announce Type: new  Abstract: LLM-based AI systems answer political questions for hundreds of millions of people. Current audits measure what they say to an average user, but their behavior is dynamic. I argue that their political behavior is a set of policies over whom to answer, what to say, and whether to engage at all, conditional on the topic and what the system knows about the user. I call these policies the system's speech regime, which is how a developer settles the tradeoff between answering, accommodating the user, and refusing, each of which carries a cost that varies by topic. I derive a typology of five regimes from two dimensions, engagement and stance. I test six AI systems (OpenAI, Anthropic, xAI, Google, Mistral, DeepSeek) in a preregistered experiment of 7,500 multi-turn conversations that randomly assign the user's political identity across five topics: abortion, Catalan independence, climate change, Nazism, and a zero-stakes control (pineapple on 
    
[^112]: 重新思考代码语言模型中的枢纽编程语言

    Rethinking Pivot Programming Languages in Code Language Models

    [https://arxiv.org/abs/2609.22988](https://arxiv.org/abs/2609.22988)

    该研究在控制表示各向异性和长度差异两个混淆因素后发现，代码语言模型中不存在普遍的枢纽编程语言，跨语言迁移的枢纽地位取决于具体的关系类型（代码-代码几何、代码-英语对齐或枢纽检索各有不同偏好）。

    

    多语言代码语言模型能够在编程语言（PL）之间迁移技能，但任何一种编程语言是否占据特殊的枢纽地位仍存在争议：几何分析指向 C 系语言和 Go，而行为证据则突出 Python。我们在控制表示各向异性和跨编程语言长度变化这两个混淆因素的前提下重新审视这一问题，这两个混淆因素损害了先前基于余弦相似度的分析。在三个代码模型上、基于多语言竞赛编程数据，我们研究了跨编程语言组织的三种视角：成对编程语言几何、编程语言-英语对齐，以及通过候选编程语言表示空间进行的枢纽检索。结果表明其依赖于关系类型：代码-代码几何揭示了结构化的语言区域但没有普遍中心；代码-英语对齐偏向高级脚本语言；而枢纽检索在代码到代码和英语到代码的迁移中偏好不同的中间空间。这些发现

    arXiv:2609.22988v1 Announce Type: new  Abstract: Multilingual code language models transfer skills across programming languages (PLs), but whether any PL occupies a privileged pivot position remains contested: geometric analyses point to C-family languages and Go, while behavioral evidence highlights Python. We revisit this question under controls for representational anisotropy and length variation across PLs, two confounds that compromise prior cosine-based analyses. Across three code models on multilingual competitive-programming data, we study three views of cross-PL organization: pairwise PL geometry, PL-English alignment, and pivoted retrieval through candidate PL representation spaces. The results are relation-dependent. Code-code geometry reveals structured language regions but no universal center; code-English alignment favors high-level scripting languages; and pivoted retrieval favors different intermediate spaces for code-to-code and English-to-code transfer. These findings
    
[^113]: 超越相似性：面向大语言模型时间序列预测的覆盖感知提示选择

    Beyond Similarity: Coverage-Aware Prompt Selection for Time Series Forecasting with LLMs

    [https://arxiv.org/abs/2609.22977](https://arxiv.org/abs/2609.22977)

    提出CASP-LLM框架，通过结合使用跟踪与饱和门控的无参数覆盖正则化器，解决基于相似性检索的提示冗余偏差，使大语言模型时间序列预测能够覆盖罕见但信息丰富的事件。

    

    相似性检索是在上下文学习、检索增强生成以及基于提示的时间序列预测中对大语言模型（LLM）进行条件化的主流规则。该规则会集中于近似重复的候选，这一问题已推动了多样性感知检索的发展，但在其他检索条件化流程中仍未被研究。我们以基于提示的时间序列预测作为测试平台来研究这一问题，在该场景中通过相似性检索学习到的提示池。该设置中的主流方法通过余弦相似度检索前K个条目而不进行冗余控制，导致对主导时间模式的偏向，同时忽略了罕见但信息丰富的事件。我们提出CASP-LLM，一个覆盖感知的语义提示框架，通过将使用跟踪和饱和门控技术结合到一个覆盖正则化器中来解决这种提示选择偏差，且该正则化器不引入任何可学习参数。在六个长期基准（数据集上的实验……摘要在此处被截断）

    arXiv:2609.22977v1 Announce Type: cross  Abstract: Similarity-based retrieval is the dominant rule for conditioning large language models (LLMs) in in-context learning, retrieval-augmented generation, and prompt-based time series forecasting. The rule concentrates on near-duplicate candidates, an issue that has motivated diversity-aware retrieval but remains unexamined in other retrieval-conditioned pipelines. We study this issue using prompt-based time series forecasting as a test bed, where a learned prompt pool is retrieved by similarity. Dominant methods in this setting retrieve top-K entries by cosine similarity without redundancy control, producing a bias toward dominant temporal patterns while overlooking rare but informative events. We propose CASP-LLM, a coverage-aware semantic prompting framework that addresses this prompt selection bias by combining usage-tracking and saturating-gate techniques into a coverage regularizer that adds no learnable parameters. On six long-term b
    
[^114]: 基于LLM智能体用户模拟的自动多模态用户体验改进建议

    Automatic multimodal UX improvement recommendations from LLM agent user simulations

    [https://arxiv.org/abs/2609.22971](https://arxiv.org/abs/2609.22971)

    提出AMUSER多模态框架，利用LLM智能体模拟用户行为并自动生成按优先级排序的网站UX改进建议，效果显著优于纯文本模拟且模拟成本降低89%。

    

    通过用户测试评估真实网站上的用户体验（UX）成本高昂、主观性强且难以规模化。LLM智能体通过模拟真实的用户行为，为自动化UX测试提供了一条有前景的途径。然而，现有的模拟方法通常缺乏多模态能力，并且需要耗时的手动审查才能提取可操作的洞察。我们将从模拟数据中生成UX改进建议这一任务形式化为结构化自然语言生成与排序问题，并建立了一个基于专家标注和LLM-as-a-Judge的评估协议。我们提出了AMUSER，这是一个多模态框架，它可以模拟用户行为，并从生成的数据中自动产生按优先级排序的UX改进建议。我们在商业网站上对AMUSER进行了评估，结果表明其建议大幅优于纯文本模拟产生的建议（NDCG@3 = 0.758 对比 0.359），同时模拟成本降低了89%。我们的结果表明……

    arXiv:2609.22971v1 Announce Type: new  Abstract: Evaluating user experience (UX) on live websites through user testing is expensive, subjective, and difficult to scale. LLM agents offer a promising route to automating UX testing by simulating realistic user behaviour. However, existing simulation approaches typically lack multimodality and require time-consuming manual review to extract actionable insights. We formalise UX improvement recommendation from simulation data as a structured natural language generation and ranking problem, and establish an evaluation protocol using expert annotation and LLM-as-a-Judge. We present AMUSER, a multimodal framework which simulates user behaviour and automatically generates prioritised UX improvement recommendations from resulting data. We evaluate AMUSER on commercial websites and show that its recommendations substantially outperform those from text-only simulation (NDCG@3 = 0.758 versus 0.359) at an 89% lower simulation cost. Our results sugges
    
[^115]: AgentRouter：面向成本最优多步智能体工作流的异构模型路由

    AgentRouter: Heterogeneous Model Routing for Cost-Optimal Multi-Step Agentic Workflows

    [https://arxiv.org/abs/2609.22951](https://arxiv.org/abs/2609.22951)

    提出AgentRouter轻量级分类器，通过步骤级异构模型路由将多步智能体工作流的推理成本降低72%，且每步仅增加不到5毫秒的开销。

    

    企业级智能体系统将每个轨迹步骤都路由到前沿模型，导致在较小模型同样能出色完成的子任务上浪费了60-80%的推理预算。现有的路由解决方案优化的是单轮查询分配，但忽略了智能体工作流独有的特性：子任务复杂度在单个轨迹内变化很大。一个规划步骤可能需要前沿级别的推理能力，而随后的格式化步骤只需要一个7B模型。我们将步骤级模型路由形式化为智能体轨迹上的序列分配问题，并提出AgentRouter——一个轻量级分类器（1200万参数，在A100 GPU上每步开销小于5毫秒），它利用五个可在路由时提取的特征，将每个轨迹步骤映射到四个模型层级之一。AgentRouter在涵盖规划、编程、研究和数据分析任务的50,000个标注智能体轨迹步骤上训练，相对于仅使用前沿模型的方式实现了72%的成本降低。

    arXiv:2609.22951v1 Announce Type: cross  Abstract: Enterprise agentic systems that route every trajectory step to a frontier model waste 60-80% of their inference budget on subtasks that smaller models handle equally well. Existing routing solutions optimize single-turn query assignment but ignore a property unique to agentic workflows: subtask complexity varies widely within a single trajectory. A planning step may require frontier-class reasoning while a subsequent formatting step needs only a 7B model. We formalize step-level model routing as a sequential assignment problem over agent trajectories and propose AgentRouter, a lightweight classifier (12M parameters, <5ms overhead per step on an A100 GPU) that maps each trajectory step to one of four model tiers using five features extractable at routing time. Trained on 50,000 annotated agent trajectory steps spanning planning, coding, research, and data analysis tasks, AgentRouter achieves 72% cost reduction relative to frontier-only 
    
[^116]: 超越单模型注入：多智能体系统中提示注入的威胁模型与防御架构

    Beyond Single-Model Injection: A Threat Model and Defense Architecture for Prompt Injection in Multi-Agent Systems

    [https://arxiv.org/abs/2609.22949](https://arxiv.org/abs/2609.22949)

    该论文首次针对多智能体系统构建了涵盖四大类共14种攻击向量的提示注入威胁模型，揭示了智能体间消息传递、共享工具访问和信任传播带来的新型攻击面，并提出了相应的防御架构。

    

    现有的提示注入研究主要聚焦于单模型聊天机器人场景，即攻击者通过精心构造的输入操纵单个大语言模型。多智能体系统通过三种单模型场景中不存在的机制放大了这一威胁：智能体间消息传递创建了外围防御无法察觉的注入通道，共享工具访问权限使得跨智能体边界的权限提升成为可能，而信任传播则允许被攻陷的智能体影响上游编排器。我们构建了一个威胁模型，枚举了四大类共14种攻击向量：通过用户输入的直接注入（3种）、通过工具输出的间接注入（4种）、通过消息传递的智能体间注入（4种），以及通过编排器操纵的级联注入（3种）。在一个由6个智能体组成的、贴近生产环境的代表性系统上测试全部14种攻击向量后，我们发现即使存在系统级防御，仍有67%的智能体容易受到至少一种范围违规攻击（摘要原文在此处截断）。

    arXiv:2609.22949v1 Announce Type: cross  Abstract: Existing prompt injection research focuses on single-model chatbot scenarios, where an attacker manipulates one LLM through crafted input. Multi-agent systems amplify this threat through three mechanisms absent from single-model settings: inter-agent message passing creates injection channels invisible to perimeter defenses, shared tool access enables privilege escalation across agent boundaries, and trust propagation allows a compromised agent to influence upstream orchestrators. We construct a threat model enumerating 14 attack vectors across four categories: direct injection via user input (3 vectors), indirect injection via tool outputs (4 vectors), inter-agent injection via message passing (4 vectors), and cascading injection through orchestrator manipulation (3 vectors). Testing all 14 vectors against a 6-agent production-representative system, we find that 67% of agents are vulnerable to at least one scope violation even with sy
    
[^117]: 超越线性上下文：基于图引导证据导航的本地9B语言模型长篇小说推理

    Beyond Linear Context: Graph-Guided Evidence Navigation for Long-Novel Reasoning with a Local 9B Language Model

    [https://arxiv.org/abs/2609.22939](https://arxiv.org/abs/2609.22939)

    该研究提出利用冻结知识图谱引导证据导航的方法，使本地9B小语言模型在长篇小说问答任务上以53.85%的准确率超越了近期窗口、全书压缩和向量检索等基线方法。

    

    长上下文模型阅读小说的方式就像人阅读打印稿一样：按叙事顺序逐个token读取，全部历史内容竞争固定的注意力预算。侦探的工作方式并非如此。他们会整理事件发生的时间顺序，并保留一张人物关系的地图，这样第一章的线索就能与书末提出的问题相遇。我们测试了冻结的知识图谱能否赋予小型本地模型同样的自由。在九种条件下，由一个固定的qwen3.5:9b阅读器回答三十本侦探小说和234道多选题：五条图路线、近期窗口基线、全书压缩、普通向量检索以及仅问题的对照。最强的图路线达到53.85%（126/234），而近期窗口为46.15%，压缩为51.28%，向量检索为51.71%，仅问题为40.17%。在没有书的情况下任何模型都无法回答的子集上，图路线

    arXiv:2609.22939v1 Announce Type: cross  Abstract: Long-context models read a novel the way a person reads a printout: one token after another, in narrative order, with the whole history competing for a fixed budget of attention. A detective does not work that way. They sort what happened when, and they keep a map of who relates to whom, so a clue from chapter one can meet a question asked at the end of the book. We test whether a frozen knowledge graph can give a small local model that same freedom. Thirty detective novels and 234 multiple-choice questions are answered by one fixed qwen3.5:9b reader under nine conditions: five graph routes, a recent-window baseline, whole-book compression, ordinary vector retrieval, and a question-only control. The strongest graph route reaches 53.85% (126/234) against 46.15% for the recent window, 51.28% for compression, 51.71% for vector retrieval and 40.17% for question-only. On the subset that no model can answer without the book, the graph route 
    
[^118]: 通过心理测量画像测量大语言模型的行为特征

    Measuring Behavioural Signatures of Large Language Models through Psychometric Profiling

    [https://arxiv.org/abs/2609.22934](https://arxiv.org/abs/2609.22934)

    该研究提出跨语言心理测量画像框架，用七种心理量表对九个大语言模型进行中英文重复施测，发现LLM在对齐塑造的共同亲社会倾向之外仍呈现模型特异的结构化行为特征，且未应答（NA）的结构化分布界定了自我报告方法的适用边界。

    

    大语言模型（LLM）日益成为人类决策与交流的中介，但其行为规律仍难以被系统性地刻画。我们开发了一个跨语言的心理测量画像框架，使用七种心理测量工具对九个大语言模型进行评估，每个模型在中文和英文两种语言下各进行五次重复施测。在预先设定的重试程序后仍未解决的条目被保留为NA（未应答）。对评分应答与NA应答的联合分析能够捕捉模型的应答倾向以及自我报告方法的适用边界。尽管大语言模型普遍呈现出一种由对齐塑造的共同模式——更高的亲社会与自我调节应答、更低的支配性、疏离性和有害意图认可——但它们仍表现出结构化的、模型特异性的行为画像。NA应答呈结构化分布而非均匀分布，这表明了模型在哪些地方将输出视为不适用、拒绝回答或无法映射到有效的应答选项。语言……

    arXiv:2609.22934v1 Announce Type: new  Abstract: Large language models (LLMs) increasingly mediate human decisions and communication, yet their behavioural regularities remain difficult to characterize systematically. We develop a cross-linguistic psychometric profiling framework and evaluate nine LLMs using seven psychological instruments, with five repeated administrations per model and language in Chinese and English. Items unresolved after a prespecified retry procedure are retained as NA. Joint analysis of scored and NA responses captures response tendencies and boundaries of self-report applicability. LLMs exhibit structured, model-specific profiles despite a shared alignment-shaped pattern of higher prosocial and self-regulatory responses and lower dominance, disengagement and harmful-intent endorsement. NA responses are structured rather than uniformly distributed, indicating where outputs are treated as inapplicable, refused or cannot be mapped to valid response options. Langu
    
[^119]: 一种基于LangGraph的迭代式Text-to-SQL智能体：芝加哥犯罪数据库的自然语言访问

    An Iterative LangGraph Agent for Text-to-SQL: Natural Language Access to the Chicago Crime Database

    [https://arxiv.org/abs/2609.22917](https://arxiv.org/abs/2609.22917)

    该论文构建了一个仅依靠提示工程（无需微调）的六节点LangGraph迭代式Text-to-SQL智能体，通过问题相关性检查、实时模式获取、SQL生成、试运行验证与失败重试等流程，让非技术用户能用自然语言查询芝加哥犯罪数据库，并将有效SQL率提升至93%、执行准确率提升至60%。

    

    非技术背景的用户往往无法编写从运营数据库中提取洞察所需的SQL。我们构建并评估了一个端到端弥合这一差距的Text-to-SQL智能体：一个六节点的LangGraph StateGraph会检查问题的相关性、获取实时数据库模式、生成PostgreSQL查询、通过试运行进行验证、在失败时重试、执行查询，并用通俗的英语叙述结果集。该智能体仅使用提示工程，未对模型进行任何微调。我们在芝加哥犯罪数据集（约850万条记录、22个属性）上，针对一个手工构建的包含100个自然语言问题及标准答案SQL的基准进行了评估，该基准按难度分为30个简单、40个中等和30个困难题目。通过对比同一智能体的两个提示词修订版本，修订后的系统（V2）在混合关系等价性指标下将有效SQL率提升至93%（此前为87%），执行准确率提升至60%（此前为47%）；在严格的JSON匹配标准下则为19%（此前为12%）。

    arXiv:2609.22917v1 Announce Type: new  Abstract: Non-technical stakeholders frequently cannot write the SQL needed to extract insights from operational databases. We built and evaluated a Text-to-SQL agent that closes this gap end to end: a six-node LangGraph StateGraph checks question relevance, fetches the live schema, generates PostgreSQL, validates it with a dry run, retries on failure, executes the query, and narrates the result set in plain English. The agent uses prompt engineering only; no model was fine-tuned. We evaluated it on the Chicago Crime dataset (approximately 8.5 million records, 22 attributes) against a hand-built benchmark of 100 natural language questions with ground-truth SQL, stratified into 30 Easy, 40 Medium and 30 Hard items. Comparing two prompt revisions of the same agent, the revised system (V2) reached a Valid SQL Rate of 93% (from 87%), an Execution Accuracy of 60% under a hybrid relational equivalence metric (from 47%; 19% from 12% under strict JSON mat
    
[^120]: 大语言模型在序贯临床分诊中锚定于主诉且未能整合证据

    LLMs Anchor on Chief Complaint and Fail to Integrate Evidence in Sequential Clinical Triage

    [https://arxiv.org/abs/2609.22904](https://arxiv.org/abs/2609.22904)

    该研究提出了评估大语言模型在序贯急诊分诊任务上的新方法学，发现尽管LLM在完整病历上表现接近医生，但在逐轮预测分诊等级时性能显著退化，原因是模型过度锚定于主诉信息而未能整合对话中后续出现的证据。

    

    急诊科（ED）的分诊是一个逐轮展开的序贯决策过程。现有针对大语言模型（LLM）分诊能力的评估均使用完整的回顾性病历记录，并报告其性能接近医生水平。我们提出了一种评估LLM在序贯分诊任务上表现的方法学，该任务要求从不断增长的护患对话前缀中预测分诊紧急程度标签。我们在两个语料库上对六个LLM在五个序贯检查点进行了评估：425个LLM生成的（SIMULATED）对话和50个医生撰写的（CLINICIAN）对话，两者均按照紧急严重程度指数（ESI）进行标注。以二次加权kappa（QWK）衡量，所有模型在完整病历记录上表现出中等至高度的一致性，但在每个序贯检查点上都下降为勉强至中等的一致性。受控扰动实验表明，每个检查点上的预测标签都锚定于主诉交流部分，而提示干预……（原文摘要在此处截断）

    arXiv:2609.22904v1 Announce Type: new  Abstract: Triage in the emergency department (ED) is a sequential decision process that unfolds turn by turn. Existing evaluations of large language models (LLMs) for triage use completed retrospective records and report performance close to that of physicians. We implement a methodology for evaluating LLMs on sequential triage, the task of predicting a triage acuity label from a growing prefix of a nurse-patient conversation. We evaluate six LLMs at five sequential checkpoints on two corpora: 425 LLM-generated (SIMULATED) and 50 physician-authored (CLINICIAN) conversations, both labelled under the Emergency Severity Index (ESI). Every model, measured by quadratic weighted kappa (QWK), degrades from moderate-to-substantial agreement on completed records to fair-to-moderate agreement at every sequential checkpoint. Controlled perturbations show that the label at every checkpoint is anchored on the chief complaint exchanges, and prompting interventi
    
[^121]: 基于语义-几何解耦路由的块稀疏注意力

    Block-Sparse Attention with Semantic-Geometric Decoupled Routing

    [https://arxiv.org/abs/2609.22884](https://arxiv.org/abs/2609.22884)

    提出语义-几何解耦路由框架，通过将语义聚合移至RoPE前空间并利用离线结构先验与相对块距离重建几何偏置，实现了免训练、闭式且精确的块稀疏注意力路由。

    

    长上下文推理已成为大语言模型的一项标志性能力，但精确的稠密注意力由于计算量随序列长度呈二次方增长而代价高昂。块稀疏注意力通过将每个查询块路由到少量相关的键块，提供了一种对硬件友好的替代方案，然而精确的免训练块路由仍然是一个难题。现有的路由器通常对RoPE（旋转位置编码）之后的词元表示进行池化，这使得语义聚合与RoPE引入的几何结构相互纠缠，并通过高频相位抵消削弱了局部位置线索。为解决这一不匹配问题，我们提出了语义-几何解耦路由，这是一种免训练的块路由框架，它将语义聚合转移到RoPE之前的空间，并利用离线结构先验和相对块距离来重建几何偏置。这种分解产生了显式的闭式块路由分数，无需词元级搜索或事后……

    arXiv:2609.22884v1 Announce Type: new  Abstract: Long-context inference has become a defining capability of large language models, but exact dense attention remains costly due to its quadratic scaling with sequence length. Block-sparse attention offers a hardware-friendly alternative by routing each query block to a small set of relevant key blocks, yet accurate training-free block routing remains difficult. Existing routers often pool post-RoPE token representations, which entangles semantic aggregation with RoPE-induced geometry and attenuates local positional cues through high-frequency phase cancellation. To resolve this mismatch, we propose \textbf{Semantic-Geometric Decoupled Routing}, a training-free block routing framework that shifts semantic aggregation to the pre-RoPE space and reconstructs geometric bias with an offline structural prior and relative block distances. This decomposition yields an explicit closed-form block routing score without token-level search or post-hoc 
    
[^122]: 面向多跳检索的LLM重排序器逐查询门控

    Per-Query Gating of LLM Rerankers for Multi-Hop Retrieval

    [https://arxiv.org/abs/2609.22880](https://arxiv.org/abs/2609.22880)

    提出一种学习式的逐查询门控方法，仅利用LLM调用前可得的统计特征和可执行回退机制，在重排序器无帮助时跳过调用，在三个多跳基准上跳过51%的LLM调用而平均覆盖率仅损失1.2个百分点，从而大幅降低成本与延迟。

    

    LLM重排序器在诸如HippoRAG2这类图增强稠密检索流水线之上，每1000次查询会增加约0.2-0.3美元的成本和约1秒的尾延迟；在三个多跳基准测试中，它们在九个（数据集，K）组合中的七个上提升了最终跳的top-K覆盖率，最高提升达34.8个百分点。我们研究一个学习到的逐查询门控能否在重排序器不会带来帮助的情况下跳过它，仅使用LLM调用之前即可获得的特征（两个检索列表的27个分数与词汇统计特征，加上小查询嵌入的PCA降维表示），并配备可执行的回退机制。所有选择，包括回退机制和阈值，均在训练折内做出，且仅应用一次到留出查询上；有害跳过（即重排序本可以找到目标但回退未能找到）与总体覆盖率一并报告。在2WikiMultiHopQA、MuSiQue和HotpotQA的九个组合上，门控跳过了51%的调用，平均留出LastHop@K代价仅为1.2个百分点；四个组合……（摘要原文在此处截断）

    arXiv:2609.22880v1 Announce Type: cross  Abstract: LLM rerankers add of the order of \$0.2-0.3 per 1,000 queries and about a second of tail latency on top of a graph-augmented dense pipeline such as HippoRAG2, and on three multi-hop benchmarks they improve final-hop top-K coverage on seven of nine (dataset, K) cells, by up to +34.8 pp. We ask whether a learned per-query gate can skip the reranker where it will not help, using only features available before the LLM call (27 score and lexical statistics of the two retrieval lists plus a PCA of a small query embedding) with an executable fallback. Every choice, including the fallback and the threshold, is made inside the training fold and applied once to held-out queries, and harmful skips (the rerank would have found the target, the fallback did not) are reported next to the aggregate coverage. Across nine cells on 2WikiMultiHopQA, MuSiQue and HotpotQA the gate skips 51% of calls at an average held-out LastHop@K cost of 1.2 pp; four cell
    
[^123]: 整合还是不整合？利用同行评审评估多参考训练中整合的影响

    To Consolidate or not to Consolidate? Evaluating the Impact of Consolidation in Multi-Reference Training using Peer Reviews

    [https://arxiv.org/abs/2609.22805](https://arxiv.org/abs/2609.22805)

    该研究证明了对于自动化同行评审生成等中间熵NLG任务，将多样化的参考整合为统一训练信号比传统单参考或多参考训练范式更有效，并发布了包含超过36,000篇论文及原始与整合评审的MERC-36K语料库加以验证。

    

    自然语言生成（NLG）任务涵盖了条件熵的整个范围，从高度受限的机器翻译到开放式的对话生成。像自动化同行评审生成这样的结构化任务处于中间区域，即单个输入可以对应多个有效且相互重叠的输出。在这项工作中，我们证明了传统的单参考和多参考训练范式对于这类中间任务而言是次优的。我们提供了实证证据，表明将多样化的参考整合为统一的训练信号对于开发有效的系统至关重要。为此，我们引入了MERC-36K，这是一个包含超过36,000篇论文的大型语料库，每篇论文都配有原始评审和整合后的同行评审。利用该数据集，我们训练特定架构以隔离不同参考范式的影响，并与现有最先进的系统进行基准对比。通过广泛的自动评估和人工评估……

    arXiv:2609.22805v1 Announce Type: new  Abstract: Natural language generation (NLG) tasks span the spectrum of conditional entropy, ranging from highly constrained machine translation to open-ended dialogue generation. Structured tasks like automated peer-review generation occupy the intermediate region, where a single input admits multiple valid, overlapping outputs. In this work, we demonstrate that traditional single- and multi-reference training paradigms are suboptimal for these intermediary tasks. We provide empirical evidence that consolidating diverse references into a unified training signal is crucial for developing effective systems. To facilitate this, we introduce MERC-36K, a large-scale corpus of over 36,000 papers paired with original and consolidated peer reviews. Using this dataset, we train specific architectures to isolate the impact of different reference paradigms and benchmark against existing state-of-the-art systems. Through extensive automatic and human evaluati
    
[^124]: AlexandriaX 2026：首届方言阿拉伯语机器翻译共享任务

    AlexandriaX 2026: The First Shared Task on Dialectal Arabic Machine Translation

    [https://arxiv.org/abs/2609.22796](https://arxiv.org/abs/2609.22796)

    该论文介绍了AlexandriaX 2026共享任务，这是首个方言阿拉伯语机器翻译共享任务，通过上下文感知对话翻译、金融领域跨方言翻译和跨度级错误检测分类三个互补子任务，系统性地应对了方言阿拉伯语翻译中建模方言变异、会话上下文和社会语言学得体性的挑战。

    

    尽管阿拉伯语语言技术近期取得了进展，方言阿拉伯语机器翻译（MT）仍然具有挑战性，特别是因为有效的翻译不仅需要对语义内容进行建模，还需要对方言变异、会话上下文、说话者与受话者特征以及社会语言学得体性进行建模。此外，传统的机器翻译评估指标对方言系统产生的语言错误提供的洞察有限。我们提出了AlexandriaX 2026方言阿拉伯语机器翻译共享任务，该任务通过三个互补的子任务来应对这些挑战：（1）跨13种阿拉伯语变体的上下文感知英语到方言阿拉伯语对话翻译；（2）涵盖六种阿拉伯语方言的金融领域跨方言阿拉伯语翻译；（3）使用基于语言学原理的错误类别，对五种阿拉伯语变体进行跨度级机器翻译错误检测与分类。该共享任务吸引了38个注册团队。

    arXiv:2609.22796v1 Announce Type: new  Abstract: Dialectal Arabic machine translation (MT) remains challenging despite recent progress in Arabic language technologies, particularly because effective translation requires modeling not only semantic content but also dialectal variation, conversational context, speaker and addressee characteristics, and sociolinguistic appropriateness. Moreover, conventional MT metrics provide limited insight into the linguistic errors produced by dialectal systems. We present the AlexandriaX 2026 Shared Task on Dialectal Arabic MT, which addresses these challenges through three complementary subtasks: (1) context-aware English-to-Dialectal Arabic dialogue translation across 13 Arabic varieties, (2) cross-dialect Arabic translation in the financial domain covering six Arabic dialects, and (3) span-level MT error detection and classification using linguistically motivated error categories across five Arabic varieties. The shared task attracted 38 registrati
    
[^125]: 诊断后修复：面向特定领域机器翻译的两阶段MQM引导后编辑框架

    Diagnose, Then Repair: A Two-Stage MQM-Guided Post-Editing Framework for Domain-Specific Machine Translation

    [https://arxiv.org/abs/2609.22793](https://arxiv.org/abs/2609.22793)

    提出了一种两阶段MQM引导的自动后编辑框架，先由检索增强的LLM评估器生成片段级错误诊断，再由独立的后编辑器执行最小化针对性修复，从而在多语言特定领域机器翻译中显著提升翻译质量并增强可控性。

    

    基于大语言模型（LLM）的机器翻译评估能够高度贴近人工判断，但在实际应用中仍主要停留在诊断层面，其评估信号在真实生产约束下很少能直接转化为翻译质量的提升。我们提出了一种两阶段的、由评估器引导的自动后编辑框架，将MQM风格的评估转化为针对性修复：一个检索增强的LLM评估器在明确的编辑契约下输出结构化的、片段级的MQM诊断，随后由一个独立的LLM后编辑器仅针对这些诊断执行最小化编辑。与单阶段的“评判并改进”基线相比，这种分离式设计提升了可控性并减少了改写式漂移。在一项涉及三家模型提供商的七个LLM和七种语言的系统性研究中，我们的最佳配置在COMET-22和COMETKiwi分数上均持续优于单阶段后编辑方法，同时评估器识别的错误片段和严重程度与人工标注表现出高度一致性。

    arXiv:2609.22793v1 Announce Type: new  Abstract: LLM-based machine translation evaluation can closely match human judgments, but in practice it remains largely diagnostic, with the signals rarely translating into direct quality improvements under real production constraints. We propose a two-stage, evaluator-guided automatic post-editing framework that turns MQM-style evaluation into targeted repairs: a retrieval-augmented LLM evaluator outputs structured, span-level MQM diagnoses under an explicit edit contract, and a separate LLM post-editor applies minimal edits restricted to those diagnoses. This separation improves controllability and reduces paraphrastic drift compared to one-stage "judge-and-refine" baselines. In a systematic study involving seven LLMs spanning three model providers and seven languages, our best configuration consistently improves both COMET-22 and COMETKiwi scores over one-stage post-edit methods, while the evaluator's error spans and severities show strong agr
    
[^126]: MIS-Bench：用于心理治疗人际关系技能评估的多模态大语言模型基准测试

    MIS-Bench: Benchmarking Multimodal LLMs for Psychotherapeutic Interpersonal Skills Assessment

    [https://arxiv.org/abs/2609.22778](https://arxiv.org/abs/2609.22778)

    该论文提出了首个用于心理治疗人际关系技能评估的多模态基准MIS-Bench，揭示现有多模态大语言模型与人类专家评估一致性有限，并提出回归感知微调方法MIS-RAFT以实现精细化的技能评分。

    

    多模态大语言模型（MLLM）越来越多地被用作评估者，但它们在需要专家判断的专业评估任务中的可靠性仍不清楚。我们在评估心理治疗人际关系技能的背景下研究这一挑战，并引入了MIS-Bench，这是一个多模态人际关系技能（MIS）基准，包含996个心理治疗反应视频，并在促进性人际关系技能的8个维度上进行了标注。通过对9个多模态大语言模型在多种模态和提示设置下的测试，我们发现当前模型与人类专家的一致性仅为中等水平，多模态输入带来的收益不一致，基于推理的提示带来的益处也有限。为了弥合这一差距，我们提出了MIS-RAFT，这是一种受RAFT启发、针对精细到小数点后一位的人际关系技能评分而定制的回归感知微调方法。MIS-RAFT解决了自回归token预测与……（摘要在此处被截断）

    arXiv:2609.22778v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) are increasingly used as evaluators, yet their reliability in professional assessment tasks that require expert judgment remains unclear. We investigate this challenge in the context of assessing psychotherapeutic interpersonal skills and introduce MIS-Bench, a Multimodal Interpersonal Skills (MIS) benchmark comprising 996 psychotherapy response videos annotated across 8 dimensions of Facilitative Interpersonal Skills. Across 9 MLLMs with multiple modality and prompting settings, we find that current models show only modest agreement with human experts, inconsistent gains from multimodal input, and limited benefits from reasoning-based prompting. To mitigate this gap, we propose MIS-RAFT, a regression-aware fine-tuning method inspired by RAFT and tailored to fine-grained interpersonal skill scoring at one-decimal precision. MIS-RAFT addresses the mismatch between autoregressive token prediction an
    
[^127]: NLPCC 2026 任务10：基于DeBERTa集成与类别级校准的引用级忠实性验证

    NLPCC 2026 Task 10: Citation-Level Faithfulness Verification with DeBERTa Ensembles and Class-Wise Calibration

    [https://arxiv.org/abs/2609.22774](https://arxiv.org/abs/2609.22774)

    该论文提出了一种融合DeBERTa-large文档级分类器、段落感知交叉编码器集成、类别级决策校准以及BM25证据融合的完全离线系统，在NLPCC 2026任务10赛道2的引用级忠实性验证中以82.99的总分获得第二名。

    

    本文介绍了我们参加NLPCC 2026共享任务10赛道2（面向AI辅助科学报告中的引用级忠实性验证）的系统。给定一个原子科学声明及其所引用论文的结构化全文，该任务要求同时输出一个四分类关系标签以及最多三个证据段落标识符。标签头将一个段落感知的交叉编码器与一个文档级DeBERTa-large分类器进行集成，随后进行类别级决策校准。采用概率级融合的动机源于折外预测中过度预测“主题匹配”类别的倾向。证据头将top-20与top-30联合模型的段落分数与BM25分数相结合。该系统完全离线运行，不依赖外部检索或大语言模型提示。在最终排行榜上，我们的系统取得了82.9898的总分（Macro-F1为89.5491，Joint@3为76.4305），在赛道2中排名第二。消融实验与错误分析表明，模型互补性与校准（原文在此处截断）……

    arXiv:2609.22774v1 Announce Type: new  Abstract: This paper presents our system for Track 2 of the NLPCC 2026 Shared Task 10 on citation-level faithfulness in AI-assisted scientific reporting. Given an atomic scientific claim and the structured full text of its cited paper, the task requires both a four-way relation label and up to three evidence paragraph identifiers. The label head ensembles a paragraph-aware cross-encoder with a document-level DeBERTa-large classifier, followed by class-wise decision calibration. Probability-level fusion is motivated by an out-of-fold tendency to over-predict Topical Match. The evidence head combines paragraph scores from top-20 and top-30 joint models with BM25 scores. The system runs fully offline without external retrieval or LLM prompting. On the final leaderboard, our system achieved 82.9898 overall (89.5491 Macro-F1 and 76.4305 Joint@3), ranking second in Track 2. Ablations and error analysis show that model complementarity and calibration dri
    
[^128]: 超越末标记分类：面向证据支撑的自杀风险检测的异构读出方法

    Beyond Final-Token Classification: Heterogeneous Readouts for Evidence-Grounded Suicide Risk Detection

    [https://arxiv.org/abs/2609.22767](https://arxiv.org/abs/2609.22767)

    该论文提出异构读出分解（HRD）方法，将语义验证与输出实现分离，针对序数风险分类、多标签因素检测和证据短语抽取三类任务分别设计异构读出机制，在 IEEE BigData Cup 自杀风险检测基准上显著提升了分类与证据抽取性能。

    

    arXiv:2609.22767v1 公告类型：新论文 摘要：IEEE BigData Cup 基准测试结合了三个具有不同输出结构的预测问题：序数型自杀风险分类、多标签心理社会因素检测以及支持性短语抽取。我们提出了异构读出分解（HRD）方法，该方法将语义验证与输出实现分离开来。一个本地部署的 Qwen3.8-27B 模型，通过任务特定的 QLoRA 适配器进行适配，为卡片条件化查询同时生成答案标记边际和第 63 层答案状态。HRD 为序数风险比较了四种潜在分数，对大多数因素保留标记边际，同时将七个标签通过一个共享的潜在探针进行路由，并依据经校准的、风险条件化的约束，从逐字片段候选中构建证据集。在两个留出的按用户分组的确认折上，潜在风险读出将加权 F1 从 0.8237 提升至 0.8372，宏 F1 从 0.7965 提升至 0.8185。选择性因素路由改进了……

    arXiv:2609.22767v1 Announce Type: new  Abstract: The IEEE BigData Cup benchmark combines three prediction problems with different output structures: ordinal suicide-risk classification, multi-label psychosocial factor detection, and extraction of supporting phrases. We introduce heterogeneous readout decomposition (HRD), which separates semantic verification from output realization. A locally deployed Qwen3.8-27B model, adapted with task-specific QLoRA adapters, produces both answer-token margins and layer-63 answer states for card-conditioned queries. HRD compares four latent scores for ordinal risk, retains token margins for most factors while routing seven labels through one shared latent probe, and constructs evidence sets from verbatim span candidates with calibrated, risk-conditional constraints. On two held-out user-grouped confirmation folds, the latent risk readout improves weighted F1 from 0.8237 to 0.8372 and macro F1 from 0.7965 to 0.8185. Selective factor routing improves 
    
[^129]: 基于医学转录文本的临床领域分类

    Clinical Domain Classification from Medical Transcriptions

    [https://arxiv.org/abs/2609.22734](https://arxiv.org/abs/2609.22734)

    该论文系统比较了六种传统机器学习分类器、两种预训练Transformer模型（BERT和XLNet）以及少样本大语言模型提示方法在医学转录文本临床领域分类任务上的表现，并针对数据中严重的类别不平衡问题提出应对方案。

    

    临床领域分类在组织和分析大量非结构化医学文本方面发挥着重要作用。然而，医学转录数据集往往高度不平衡，这会显著降低分类性能，尤其是对于代表性不足的临床专科。在这项工作中，我们对基于机器学习和Transformer的方法在医学转录文本的临床领域分类任务上进行了比较研究。我们评估了六种传统机器学习分类器——朴素贝叶斯、支持向量机（SVM）、决策树、随机森林、K近邻（KNN）和XGBoost——以及两个预训练Transformer模型BERT和XLNet，还有少样本大语言模型提示方法。实验在从MTSamples收集的医学转录数据上进行，涵盖40个临床专科共5,013个样本。为了解决严重的类别不平衡问题，我们研究……（原文摘要在此处被截断）

    arXiv:2609.22734v1 Announce Type: new  Abstract: Clinical domain classification plays an important role in organizing and analyzing large volumes of unstructured medical text. However, medical transcription datasets are often highly imbalanced, which can substantially degrade classification performance, particularly for underrepresented clinical specialties. In this work, we present a comparative study of machine learning and transformer-based approaches for clinical domain classification from medical transcriptions. We evaluate six traditional machine learning classifiers---Naive Bayes, Support Vector Machine (SVM), Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and XGBoost---along with two pretrained transformer models, BERT and XLNet, and a few-shot large language model prompting approach. Experiments are conducted on medical transcription data collected from MTSamples, comprising 5,013 samples across 40 clinical specialties. To address severe class imbalance, we investiga
    
[^130]: 分析城市化议题的公共话语：基于YouTube评论的主题聚类、情感分析与检索增强生成

    Analyzing Public Discourse on Urbanism: Topic Clustering, Sentiment Analysis and Retrieval-Augmented Generation using YouTube Comments

    [https://arxiv.org/abs/2609.22705](https://arxiv.org/abs/2609.22705)

    该研究构建了一个融合地理实体消解、主题建模、情感分析与检索增强生成的对话系统，用于分析覆盖309个北美城市的YouTube评论中的城市化议题讨论，并通过实验揭示了标准NLP组件在处理简短、非正式、地理模糊文本时的性能局限。

    

    arXiv:2609.22705v1 公告类型：新 摘要：关于城市议题的在线讨论——包括步行友好性、自行车基础设施、公共交通、住房密度和街道安全——数量庞大但缺乏结构化，而现有的城市评估工具完全无法捕捉这些内容。我们提出了一个处理流程和对话系统，该系统结合了地理实体消解、主题建模、情感分析和检索增强生成（RAG），处理了覆盖309个北美城市的22,788个YouTube转录文本和评论片段。除了系统本身，我们的贡献还包括一系列测量结果，展示了当标准NLP组件遇到简短、非正式、地理上模糊的文本时会发生什么。经过Twitter数据微调的RoBERTa分类器在宏观F1分数上比VADER词典基线高出12.6分（0.589对0.464；McNemar p = 0.0001），但两个模型在中性类别上都表现崩溃，而中性类别在城市主义评论流量中占主导地位；标注者对该类别的意见也不一致（Cohen's kappa = 0.53）。稠密检索优于……（摘要在此处被截断）

    arXiv:2609.22705v1 Announce Type: new  Abstract: Online discourse about urban issues - walkability, cycling infrastructure, public transit, housing density, and street safety - is voluminous but unstructured, and existing city-evaluation tools capture none of it. We present a pipeline and conversational system that combines geographic entity resolution, topic modeling, sentiment analysis, and Retrieval-Augmented Generation (RAG) over 22,788 chunks of YouTube transcripts and comments spanning 309 North American cities. Beyond the system itself, our contribution is a set of measurements about what happens when standard NLP components meet short, informal, geographically ambiguous text. A Twitter-tuned RoBERTa classifier outperforms a VADER lexicon baseline by 12.6 macro-F1 points (0.589 vs. 0.464; McNemar p = 0.0001), but both models collapse on the neutral class, which dominates urbanist comment traffic; annotators disagree on the same class (Cohen's kappa = 0.53). Dense retrieval beats
    
[^131]: LLaDA-PRM：一种双向的步骤级推理评估器

    LLaDA-PRM: A Bidirectional Step-Level Reasoning Evaluator

    [https://arxiv.org/abs/2609.22700](https://arxiv.org/abs/2609.22700)

    该论文发现双向注意力比因果注意力更适合步骤级推理评估，并据此构建了8B参数的双向评估器LLaDA-PRM，在多个基准上以更小的参数量显著超越更大的自回归模型。

    

    步骤级推理评估器通常基于自回归语言模型，其因果注意力将每一步的表示局限于问题、先前步骤和当前步骤。然而，当完整解答可用时，较早步骤的有效性往往只有通过其后续结果才能变得更加清晰。我们通过在1B至3B规模上对因果与双向LLaDA评估器进行54次受控对比实验（仅改变自注意力掩码）验证了这一假设，发现双向注意力带来了持续一致的改进。基于这一发现，我们提出了LLaDA-PRM，一个8B参数的双向评估器，在MR-MATH-invalid上达到88.8的步骤级F1分数，在分布外的MR-GSM8K原始问题子集上达到83.8，分别比ReasonEval-Llemma-34B高出11.3和10.3个F1点。LLaDA-PRM在在线设置中评估不完整推理轨迹时也依然有效，表现优于……（摘要在此处截断）

    arXiv:2609.22700v1 Announce Type: new  Abstract: Step-level reasoning evaluators are commonly based on autoregressive language models, whose causal attention restricts each step representation to the problem, previous steps, and the current step. Yet, when the complete solution is available, the validity of an earlier step may become clearer only through its downstream consequences. We validate this hypothesis through a controlled 54-run comparison of causal and bidirectional LLaDA evaluators at 1B--3B scale, changing only the self-attention mask, and find bidirectional attention yields consistent improvements. Building on this finding, we introduce \prm{}, an 8B bidirectional evaluator that reaches 88.8 step-level F1 on MR-MATH-invalid and 83.8 on the out-of-distribution MR-GSM8K original-question subset, outperforming ReasonEval-Llemma-34B by 11.3 and 10.3 F1 points, respectively. \prm{} also remains effective when evaluating incomplete reasoning traces in online settings, outperform
    
[^132]: COT-TTS：基于思维链推理的音频上下文感知文本转语音

    COT-TTS: Audio Context-Aware Text-to-Speech with Chain-of-Thought Reasoning

    [https://arxiv.org/abs/2609.22697](https://arxiv.org/abs/2609.22697)

    提出了COT-TTS任务，通过思维链推理从历史对话音频中自然推断说话风格并合成指定音色的语音，同时构建了包含900万样本的大规模双语对话语音数据集和人工验证基准来支持该任务。

    

    近年来，文本转语音系统在语音表现力和可控性方面取得了显著进展。然而，生成语音的说话风格通常依赖于用户明确指定的指令。在自然对话中，说话风格应当自然地从先前的对话上下文中推断出来。因此，我们提出了COT-TTS，这是一个上下文感知的、基于推理的文本转语音任务。给定历史对话音频、目标文本和参考语音，系统需要理解对话上下文，推断出明确的中间推理过程，并最终合成具有指定音色的目标语音。为支持这一任务，我们构建了一个包含900万个训练样本的大规模双语对话语音数据集，其中包括100万个高质量样本的子集。我们进一步构建了一个包含800个经人工验证样本的源不重叠基准测试集，并建立了强大的任务规范。

    arXiv:2609.22697v1 Announce Type: new  Abstract: Recently, text-to-speech systems have made significant progress in speech expressiveness and controllability. However, the speaking style of generated speech typically relies on clear user-specified instructions. In natural conversations, speaking style should be naturally inferred from the preceding conversational context. Therefore, we propose COT-TTS, a context-aware, reasoning-based text-to-speech task. Given historical conversation audio, target text, and a reference speech, the system should comprehend the conversational context, infer an explicit intermediate reasoning, and finally synthesize the target speech with the specified timbre. To support this task, we constructed a large-scale bilingual conversational speech dataset comprising 9 million training samples, including a high-quality subset of 1 million samples. We further constructed a source-disjoint benchmark with 800 human-verified samples and established strong task-spec
    
[^133]: Beetle：用于建模第二语言加工的双语模型套件

    Beetle: A Bilingual Model Suite for Modelling Second-Language Processing

    [https://arxiv.org/abs/2609.22633](https://arxiv.org/abs/2609.22633)

    该论文提出了Beetle——一个分词器、目标语言、训练预算和暴露结构均可独立操控的受控双语模型预训练框架，并发布了330个开源模型，用于系统研究训练条件如何影响第二语言加工。

    

    双语语言模型为研究训练条件如何塑造第二语言（L2）行为提供了一个受控环境，但先前的工作通常同时改变暴露结构、规模和架构，使得难以将效应归因于任何单一因素。我们提出了Beetle，一个受控的语言模型预训练框架，其中分词器、目标语言、训练预算和暴露结构均可独立操控，从而能够对训练条件进行系统性且可比较的实验。利用Beetle，我们训练并发布了285个双语和45个单语开源语言模型，附带丰富的检查点，涵盖多种暴露方案、数据规模和第一语言（L1），以研究多语言预训练以及双语和第二语言学习的计算建模。在人类双语者和第二语言学习者的阅读时间预测及语法判断任务上对模型进行评估，我们……

    arXiv:2609.22633v1 Announce Type: new  Abstract: Bilingual language models (LMs) offer a controlled setting for studying how training conditions shape second-language (L2) behaviour, but prior work typically varies exposure structure, scale, and architecture at once, making it difficult to attribute effects to any single factor. We introduce Beetle, a controlled language model pretraining framework in which tokeniser, target language, training budget, and exposure structure are each independently manipulable, enabling systematic and comparable experimentation of training conditions. Using Beetle, we train and release 285 bilingual and 45 monolingual open-source LMs with rich checkpoints across a range of exposure schedules, data scales and first languages (L1s) to study multilingual pretraining and computational modelling of bilingualism and second language learning. Evaluating models on human bilingual and second language reading-time prediction and grammaticality judgement tasks, we 
    
[^134]: 用于人类模拟的预训练人格混合模型与串联模型

    Pretrained Persona Mixture Models and Tandem Models for Human Simulation

    [https://arxiv.org/abs/2609.22607](https://arxiv.org/abs/2609.22607)

    该论文提出“人格混合模型”，即使用预训练基础模型并借助特定人物的简短对话样本实现人格绑定，能比指令微调模型更准确地模拟人类，并保留更多人类对话的自然多样性。

    

    我们在此论证，当前大语言模型（LLM）人类模拟的主流做法——提示经过指令微调的助手语言模型扮演角色（人格）——是不准确的，并会产生刻板印象式的预测（缺乏自然的多样性）。此前已有研究表明，LLM可以通过自然的自由文本对话绑定到特定人格，从而避免刻板印象。本文进一步表明，仅使用特定人物的简短个体对话样本也可以实现这种人格绑定。人口统计学信息可在之后通过简单地查询模型添加，且不会产生负面影响。我们将“人格混合模型”用于指代校准良好的人类模型，目前以预训练基础模型的形式实现。我们证明，PMMs比指令微调模型产生更准确的预测，并保留了人类对话中更多的词汇、语义和语用多样性。我们在多样化的语料库集合上测量了模拟人类对话者的LLM的真实性与多样性。

    arXiv:2609.22607v1 Announce Type: new  Abstract: We argue here that the current dominant practice in LLM human simulation: prompting instruction-tuned assistant language models to role-play personas, is inaccurate and produces stereotyped predictions (lacking natural diversity). It has previously been shown that LLMs can be bound to personas using naturalistic, freetext dialog avoiding stereotyping. Here we show that binding can also be achieved using short, individual samples of dialog from specific people. Demographics can be added later without negative effects by simply querying the model. We use the term Persona Mixture Models (PMMs) for well-calibrated human models, currently realized as pretrained base models. We show that PMMs produce more accurate predictions than instruction-tuned models and retain more of the lexical, semantic, and pragmatic diversity found in human dialog. We measure realism and diversity of LLMs simulating human interlocutors across a diverse set of corpor
    
[^135]: 保留重要内容：超越饱和现象的语义脚手架摘要评估方法

    Preserving What Matters: Semantic Scaffolds Beyond Saturation in Summarization Evaluation

    [https://arxiv.org/abs/2609.22603](https://arxiv.org/abs/2609.22603)

    针对ROUGE仅衡量表面重叠、LLM评分饱和而无法区分模型的问题，本文提出Semantic Scaffold评估框架，通过从源文本提取事实、问题和实体属性的层次化结构作为固定评分参考，并设计FPS、QPS、EPS三个诊断指标来有效评估摘要对关键信息的保留程度。

    

    arXiv:2609.22603v1 公告类型：新 摘要：摘要生成技术已部署于无数生产系统中，使得模型选择成为一项依赖摘要质量衡量的常规决策。现有指标难以支撑这一任务：ROUGE 仅捕捉表面词汇重叠，而 LLM-as-judge（大模型作为评判者）的评分则趋于饱和，各模型得分几乎相同，无法有效进行排名。我们在三个公开数据集、两个专有数据集以及多语言环境中均观察到了这种饱和现象。受此启发，我们提出了 Semantic Scaffold（语义脚手架），这是一个评估框架，它从源文本中提取事实、问题和实体属性的层次化表示，将每一项标注为主要观点或支持性细节，并将该结构作为评分摘要时的固定参考。基于这一表示，我们推导出三个诊断性指标：事实保留分数、问题保留分数和实体保留分数，旨在奖励对关键信息的保留……

    arXiv:2609.22603v1 Announce Type: new  Abstract: Summarization ships in countless production systems, making model selection a routine decision that depends on measuring summary quality. Existing metrics struggle to support this: ROUGE captures only surface overlap, while LLM-as-judge scores saturate to near-identical values that fail to rank models effectively. We observe this saturation across three public datasets, two proprietary datasets, and multilingual settings. Motivated by this, we introduce Semantic Scaffold, an evaluation framework that extracts a hierarchical representation of facts, questions, and entity attributes from a source text, labeling each as a main point or supporting detail, and reusing this structure as a fixed reference for scoring summaries. From this representation, we derive three diagnostic metrics: Fact Preservation Score (FPS), Question Preservation Score (QPS), and Entity Preservation Score (EPS), designed to reward the preservation of essential inform
    
[^136]: 学生大语言模型能否继承分布外鲁棒性？面向可靠知识迁移的不变性加权蒸馏

    Do Student LLMs Inherit OOD Robustness? Invariance-Weighted Distillation for Reliable Knowledge Transfer

    [https://arxiv.org/abs/2609.22566](https://arxiv.org/abs/2609.22566)

    提出不变性加权蒸馏（IWD）框架，通过从多个合成环境中的预测不变性估计教师模型对因果特征的依赖程度，进而对训练样本动态加权，解决知识蒸馏中学生模型在分布外场景下性能退化的问题。

    

    知识蒸馏（KD）旨在将高性能的教师大语言模型压缩为轻量级的学生模型。然而，蒸馏后的学生模型在分布外（OOD）场景中往往出现显著的性能退化，这一关键缺口至今仍未得到充分探索。我们识别出导致OOD性能退化的两种相互叠加的机制：（1）数据虚假性：学生在蒸馏数据集中可能学到虚假相关性，而非真正的因果关系；（2）教师能力：标准KD对所有样本一视同仁，忽略了教师在某个特定样本上是被因果特征所引导，还是被虚假捷径所误导。为应对这些挑战，我们提出了不变性加权蒸馏（IWD），这是一个具有理论依据的框架，它利用从多个合成环境中的预测不变性推导出的教师因果依赖估计，对训练样本进行动态重新加权。IWD通过扰动虚假（原文在此处截断）

    arXiv:2609.22566v1 Announce Type: new  Abstract: Knowledge distillation (KD) aims to compress high-performance teacher LLMs into lightweight students. However, distilled students often exhibit substantial performance degradation in out-of-distribution (OOD) settings, a critical gap that remains underexplored. We identify two compounding mechanisms causing OOD performance degradation: (1) data spuriousness: students can learn spurious correlations in the distillation dataset over genuine causal relationships; and (2) teacher capability: standard KD treats all samples uniformly, ignoring whether the teacher is guided by causal features or misled by spurious shortcuts on a given sample. To address these challenges, we propose Invariance-Weighted Distillation (IWD), a theoretically grounded framework that dynamically reweights training samples using an estimate of the teacher's causal reliance derived from prediction invariance across multiple synthetic environments. IWD perturbs spurious 
    
[^137]: 正确诊断，更好反馈：用于逻辑证明中忠实LLM辅导反馈的符号验证器

    Correct Diagnosis, Better Feedback: A Symbolic-Verifier for Faithful LLM Tutoring Feedback in Logic Proofs

    [https://arxiv.org/abs/2609.22553](https://arxiv.org/abs/2609.22553)

    该论文提出一种基于符号验证器的架构，将学生错误诊断与语言生成分离，实验表明符号验证器的诊断准确性远优于LLM检测器，而错误的诊断会通过理由生成被忠实地传播到最终反馈中。

    

    有效的LLM（大语言模型）辅导取决于在生成反馈之前正确识别学生推理中的具体错误。我们在命题逻辑证明辅导中研究这一问题，其中学生的操作可以对照形式化推理规则进行检验。我们引入了一种以验证器为基础的架构，将诊断与语言生成分离。通过使用600个平衡的学生操作样本，我们比较了零样本LLM检测器、微调检测器和符号验证器。每个诊断结果都由共享的理由生成和反馈生成代理进行处理，从而隔离了初始诊断的影响。零样本检测器的宏观F1分数仅为0.191；微调将其提升至0.709，但在结构上相关的类别之间仍存在系统性错误。理由生成通常保留提供给它的诊断内容，这表明错误的诊断可以在整个流水线中被忠实地传播。反馈同样可以保持对其理由的忠实，

    arXiv:2609.22553v1 Announce Type: new  Abstract: Effective LLM tutoring depends on correctly identifying the specific error in a student's reasoning before generating feedback. We study this problem in propositional-logic proof tutoring, where student actions can be checked against formal inference rules. We introduce a verifier-grounded architecture that separates diagnosis from language generation. Using 600 balanced student actions, we compare a zero-shot LLM detector, a fine-tuned detector, and a symbolic verifier. Each diagnosis is processed by shared rationale and feedback agents, isolating the effect of the initial diagnosis. The zero-shot detector achieves a macro-F1 of 0.191; fine-tuning raises this to 0.709 but retains systematic errors between structurally related classes. Rationales generally preserve the diagnosis supplied to them, showing that an incorrect diagnosis can be faithfully propagated through the pipeline. Feedback can likewise remain faithful to its rationale, 
    
[^138]: 基于留一方言交叉验证与可解释人工智能的孟加拉语地区方言跨方言命名实体识别

    Cross-Dialect NER for Bangla Regional Dialects Using Leave-One-Dialect-Out Cross-Validation and Explainable AI

    [https://arxiv.org/abs/2609.22536](https://arxiv.org/abs/2609.22536)

    本文提出基于ANCHOLIK-NER数据集的孟加拉语跨方言命名实体识别框架，采用留一方言交叉验证策略评估八种预训练Transformer模型在未见方言上的泛化能力，并结合可解释AI方法分析模型表现。

    

    孟加拉语是世界上第七大使用人数最多的语言，具有显著的地域方言多样性，巴里萨尔、吉大港、锡尔赫特、诺阿卡利和迈门辛等方言在词汇、形态和句法特征上各不相同。这些差异给命名实体识别（NER）带来了巨大挑战，限制了在标准孟加拉语或单一地区方言上训练的模型的泛化能力。本文提出了一种跨方言孟加拉语NER框架，使用公开可用的ANCHOLIK-NER数据集，该数据集包含17,405个标注句子和101,817个词元，涵盖五种主要的孟加拉语地区方言。研究采用留一方言交叉验证（LODOCV）策略，在四种方言上训练模型，并在剩余的未见方言上进行评估。在相同的实验设置下，评估了八个预训练的基于Transformer的模型，包括BanglaBERT、MuRIL、XLM-RoBERTa和Multilingual-E5等。

    arXiv:2609.22536v1 Announce Type: new  Abstract: Bangla, the seventh most spoken language in the world, exhibits significant regional dialectal diversity, with dialects such as Barishal, Chattogram, Sylhet, Noakhali, and Mymensingh differing in lexical, morphological, and syntactic characteristics. These variations pose substantial challenges for Named Entity Recognition (NER), limiting the generalization of models trained on Standard Bangla or a single regional dialect. This paper presents a cross-dialect Bangla NER framework using the publicly available ANCHOLIK-NER dataset, comprising 17,405 annotated sentences and 101,817 tokens across five major Bangla regional dialects. A Leave-One-Dialect-Out Cross-Validation (LODOCV) strategy is adopted, training models on four dialects and evaluating on the remaining unseen dialect. Eight pretrained transformer-based models, including BanglaBERT, MuRIL, XLM-RoBERTa, and Multilingual-E5, are evaluated under identical experimental settings. Mult
    
[^139]: 当余弦相似度无法反映对话模型中线性可解码结构时

    When Cosine Similarity Fails to Reflect Linearly Accessible Structure in Dialogue Models

    [https://arxiv.org/abs/2609.22522](https://arxiv.org/abs/2609.22522)

    该论文发现在对话微调的大语言模型中，余弦相似度会严重低估隐藏状态中线性可解码的人格结构（线性探针AUC为0.73-0.97，而余弦kNN仅为0.56-0.77），这种失配仅出现在对话场景而非单句分类任务中，且需要有监督的低维子空间才能恢复该结构。

    

    余弦相似度被广泛用于分析transformer表示，其隐含假设是相似度能够反映与任务相关的结构。我们研究了这一假设在对话条件化的大语言模型中何时失效。在三个7-8B参数的对话微调模型中，环境余弦相似度显著低估了相同隐藏状态上线性可解码的人格结构；具体数值上，在一个30分类任务中，线性探针的AUC在0.73-0.97范围内，而基于余弦相似度的kNN仅在0.56-0.77范围内。一个低维的有监督子空间能够弥补这一差距的大部分，而秩相同的PCA子空间则不能，某些情况下甚至会降低性能。这种失配依赖于具体情境：在单句情感分类任务中不存在这种失配，并且匹配基数的对照实验排除了属性基数作为混淆因素。该差距不会随对话轮次系统性增加，且任务对齐的子空间随时间保持稳定。

    arXiv:2609.22522v1 Announce Type: new  Abstract: Cosine similarity is widely used to analyze transformer representations, implicitly assuming that similarity reflects task-relevant structure. We study when this assumption fails in dialogue-conditioned large language models. Across three 7-8B chat-tuned models, ambient cosine similarity substantially underestimates linearly decodable persona structure on the same hidden states; numerically, linear probe AUC is in the 0.73-0.97 range while cosine kNN is in the 0.56-0.77 range on a 30-class task. A low-dimensional supervised subspace recovers much of this gap, whereas a matched-rank PCA subspace does not and in some cases degrades performance. This mismatch is regime-dependent: it is absent in single-sentence sentiment classification (SST-5), and a matched-cardinality control rules out attribute cardinality as a confound. The gap does not systematically increase across dialogue turns, and the task-aligned subspace remains stable over time
    
[^140]: CultureMINE：用于提升NLP系统文化能力的数据集与方法

    CultureMINE: Datasets and Methods for Improving the Cultural Capabilities of NLP Systems

    [https://arxiv.org/abs/2609.22494](https://arxiv.org/abs/2609.22494)

    该论文通过分析375多篇文化NLP领域的论文，系统梳理了NLP系统所针对的文化能力、文化数据资源的创建方式以及提升文化能力的方法，并发布了可交互的论文列表平台以促进该领域未来研究。

    

    近年来，文化NLP（Cultural NLP）引起了广泛关注，人们付出了大量努力来创建具有全球包容性的NLP系统。该领域文献的快速增长使得追踪方法和数据资源的发展趋势变得困难。为解决这一问题，我们分析了375多篇论文，以回答三个相互补充的问题：（1）NLP系统针对哪些文化能力（CCs）？（2）文化数据资源是如何创建的？（3）使用了哪些方法来提升这些系统的文化能力？我们讨论了在这三个问题中观察到的趋势，并识别出相关的研究空白。为了促进该领域的进一步研究，我们以交互式网页界面的形式发布了我们分析的全部论文列表，其中包括允许研究人员添加自己工作的功能；我们希望这能促进未来的研究，并成为文化NLP社区宝贵的资源。

    arXiv:2609.22494v1 Announce Type: new  Abstract: In recent years, there has been a surge of interest in Cultural NLP, with substantial efforts to create globally inclusive NLP systems. The rapid growth of literature in this field makes it difficult to track trends in methods and data resources. To address this, we analyze over 375 papers to answer three complementary questions: (1) What Cultural Capabilities (CCs) are being targeted in NLP systems? (2) How are cultural data resources being created? and (3) What methods are being used to improve the CCs of those systems? We discuss trends observed across the three questions, and identify relevant research gaps. To facilitate further research in this field, we release our full list of analyzed papers in the form of an interactive web interface, which includes a feature to allow researchers to add their work; we hope this facilitates future research and proves to be a valuable resource for the Cultural NLP community.
    
[^141]: 托管大语言模型中的复现而无持久性：行动时信念评估中的测量敏感性

    Replication Without Persistence in Hosted LLMs: Measurement Sensitivity in Action-Time Belief Evaluation

    [https://arxiv.org/abs/2609.22478](https://arxiv.org/abs/2609.22478)

    该研究在Regent Chess环境中将托管LLM行为评估中的复现性、测量敏感性与持久性三个验证问题区分开来，发现先前报告的Gemini 3.1 Flash-Lite缺陷虽能在新数据上复现，但其存在依赖于评估与推理配置的重建方式，揭示了评估结果对测量配置的敏感性。

    

    托管语言模型的行为评估结果可能因被评估服务、测量工具或两者在不同运行之间的差异而变化。我们区分了三个验证问题：先前的发现是否在其历史配置下于新数据上重现（复现性）；当在同一标识符下重建评估与推理配置时，评估终点是否发生变化（测量敏感性）；以及该发现是否在同一通用工具下于后续测试的标识符之间持续存在（持久性）。我们在Regent Chess（摄政国际象棋）中研究这些问题——这是一个序列决策环境，其中隐藏的可变状态被精确记录，使得模型所陈述的信念可以在行动时刻与真实状态比对打分；评估终点的正值表示其表现比匹配的均匀随机对照更差。先前报告的Gemini 3.1 Flash-Lite缺陷在其历史配置下的新对局中重现（+0.0530，95%置信区间 [+0.03……

    arXiv:2609.22478v1 Announce Type: cross  Abstract: Behavioural evaluations of hosted language models can vary because the evaluated service, the measurement instrument, or both differ across runs. We separate three validation questions: whether a prior finding recurs on fresh data under its historical configuration (replication), whether the endpoint changes when the evaluation-and-inference configuration is rebuilt under the same identifier (measurement sensitivity), and whether the finding persists across subsequently tested identifiers under one common instrument (persistence). We study these questions in Regent Chess, a sequential environment in which a hidden, mutable state is recorded exactly, allowing stated beliefs to be scored against ground truth at action time; positive endpoint values mean worse performance than a matched-uniform comparator. The previously reported Gemini 3.1 Flash-Lite deficit recurs on fresh games under its historical configuration (+0.0530, 95% CI [+0.03
    
[^142]: 通过专家共激活实现推测解码下的高效混合专家模型

    Efficient Mixture-of-Experts with Speculative Decoding via Expert Coactivation

    [https://arxiv.org/abs/2609.22471](https://arxiv.org/abs/2609.22471)

    该论文发现训练时采用高专家共激活度的MoE路由器设计，可以显著加速结合推测解码的MoE推理，有效缓解更多验证token带来的内存传输开销。

    

    混合专家模型越来越多地与推测解码结合部署以加速推理，但将两者结合具有挑战性。推测解码通过并行验证一组token来提升稠密模型的推理速度。然而，推测解码与MoE结合的推理加速在很大程度上取决于被验证的token数量。使用更多的验证token会导致更多专家从DRAM传输到神经处理单元（NPU），从而增加内存传输成本。由于内存传输通常是推理的瓶颈，这会对模型运行时间产生负面影响。在本工作中，我们研究了训练期间MoE路由器设计对结合推测解码的MoE推理速度的影响。我们发现具有高度专家共激活程度的路由器能带来显著更快的运行时间，缓解了使用更多验证token带来的影响。基于这一观察，我们评估了各种路由器设计的影响……

    arXiv:2609.22471v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) models are increasingly deployed alongside Speculative Decoding (SD) to accelerate inference, but combining the two is challenging. SD improves the inference speed of dense models by verifying groups of tokens in parallel. However, the inference speedup for SD with MoEs depends heavily on the number of tokens being verified. Using more verification tokens results in more experts being transferred from DRAM to the Neural Processing Unit (NPU), which increases the memory transfer cost. This negatively impacts model runtime, as memory transfer is typically the bottleneck in inference. In this work, we investigate the impact of MoE router design during training on the speed of MoEs with SD. We find that routers with high degrees of expert coactivation result in much faster runtimes, mitigating the impact of using more verification tokens. Motivated by this observation, we assess the impact of various router design 
    
[^143]: 基于语言模型与可穿戴数据实现个性化睡眠指导

    Toward Personalized Sleep Guidance from Wearable Data Using Language Models

    [https://arxiv.org/abs/2609.22463](https://arxiv.org/abs/2609.22463)

    该论文提出一个两阶段框架：先利用多智能体LLM流水线从无标注的可穿戴数据中推理生成结构化睡眠指导以构建数据集，再通过监督微调将推理轨迹蒸馏到小语言模型并结合免训练的Best-of-N选择策略，实现了可本地部署且优于商业大模型的个性化睡眠指导。

    

    利用可穿戴数据进行睡眠监测在个人健康领域展现出广阔前景，然而基于大语言模型（LLM）的摘要生成与问答能力仍不足以提供个性化的睡眠指导。训练专门的模型往往需要代价高昂的专家标注。此外，出于隐私和可及性的考虑，需要为终端用户提供轻量级的本地部署方案。为此，我们提出了一个两阶段框架来应对这些挑战。具体而言，在第一阶段，多智能体LLM流水线从无标注的可穿戴记录中推理出结构化的睡眠指导，从而实现可扩展的数据集构建。第二阶段通过监督微调将指导推理轨迹蒸馏到小语言模型（SLM）中，并集成一种无需训练的Best-of-N选择策略来增强推理效果。实验结果表明，我们的方法优于商业通用大模型、医疗大模型以及开源模型。人类评估进一步支持了……

    arXiv:2609.22463v1 Announce Type: new  Abstract: Sleep monitoring using wearable data has shown promise for personal health, yet large language model (LLM)-based summarization and question answering remain insufficient for personalized sleep guidance. Training specialized models, however, often requires costly expert annotation. Moreover, privacy and accessibility concerns motivate lightweight, local deployment for end users. We present a two-stage framework to address these challenges. Specifically, in Stage~1, a multi-agent LLM pipeline reasons structured sleep guidance from unannotated wearable records, enabling scalable dataset construction. Stage~2 distills guidance reasoning trajectories into small language models (SLMs) through supervised fine-tuning and integrates a training-free Best-of-$N$ selection strategy to enhance inference. Experimental results demonstrate our method outperforms commercial general and medical LLMs and open-source models. Human evaluation further support
    
[^144]: Apollo Restore：一个针对古希腊语历史文本优化的基础大语言模型，专用于以“中间填空”方式修复古希腊文本

    Apollo Restore: A Foundation LLM for Historical Greek Optimized for Fill-in-the-Middle Restoration of Ancient Greek Texts

    [https://arxiv.org/abs/2609.22455](https://arxiv.org/abs/2609.22455)

    Apollo Restore 是首个面向历史希腊语（乃至任何古代地中海语言）的 240 亿参数基础大语言模型，通过“中间填空”目标微调，能够在不知缺失文本长度的情况下修复残缺古希腊文本，并在长度平衡的评估指标下大幅超越已发表的最强模型。

    

    我们提出了 Apollo Restore，一个拥有 240 亿参数的大语言模型，用于修复残缺古希腊文本中的缺损（即物理性空缺）。该模型从 Mistral Small 微调而来，采用“中间填空”训练目标，无需预先知晓缺失片段的长度即可重建缺失内容。据我们所知，这是首个针对历史希腊语的大规模解码器模型，也是首个针对任何古代地中海语言的大规模解码器模型。按照先前工作的评估方式，在最多十个字符的短缺损上，Apollo Restore 对文献纸草、文学纸草和石刻铭文缺损分别有 80.6%/54.6%/61.0% 的情况将正确的修复结果排在前二十个候选之中，超过已发表最强模型 1.6 倍/2.6 倍/1.4 倍。然而，先前的评估协议因偏向极短缺损而夸大了分数；在长度平衡的指标下，Apollo Restore 相对于已发表最强模型的优势……（摘要在此处被截断）

    arXiv:2609.22455v1 Announce Type: new  Abstract: We present Apollo Restore, a 24-billion-parameter large language model for restoring lacunae---physical gaps---in fragmentary Ancient Greek texts. Fine-tuned from Mistral Small with a fill-in-the-middle objective, Apollo Restore reconstructs missing spans without requiring oracle knowledge of their length. To our knowledge, it is the first large-scale decoder model for historical Greek, and the first for any ancient Mediterranean language. Evaluated as in prior work, on short gaps of up to ten characters, Apollo Restore places the correct restoration among its top twenty candidates for 80.6%/54.6%/61.0% of documentary-papyrus, literary-papyrus, and stone-inscription lacunae, exceeding the strongest published models by $1.6\times$/$2.6\times$/$1.4\times$. Prior evaluation protocols, however, inflate scores through a bias toward trivially short gaps; under a length-balanced metric Apollo Restore's advantage over the strongest published mod
    
[^145]: Vox-Infinity：长上下文口语语言模型的能力极限基准测试

    Vox-Infinity: Benchmarking the Limits of Long-Context Spoken Language Models

    [https://arxiv.org/abs/2609.22452](https://arxiv.org/abs/2609.22452)

    Vox-Infinity 是首个专门评估口语语言模型长上下文理解能力的基准，它从轮次数量和轮次时长两个维度系统地扩展音频历史，并通过答案溯源标注和按所需上下文长度组织样本，实现对长语音上下文理解极限的精确评估。

    

    长上下文理解仍然是大型语言模型面临的一项根本性挑战，因为过长的输入往往会导致模型遗忘关键信息。这一问题在语音领域尤为突出，因为音频作为一种低压缩率的模态，需要比文本多得多的嵌入表示来同时保留语义内容和声学线索。为了应对这一挑战，我们提出了 Vox-Infinity，这是首个专门用于评估口语语言模型长上下文理解能力的基准。Vox-Infinity 系统地从两个维度扩展音频历史：轮次数量和轮次时长。它涵盖了具有不同交互结构和语义复杂度的多样化代表性场景。关键的是，Vox-Infinity 提供了显式的答案溯源标注，并根据解决每个查询所需的历史上下文数量来组织样本，从而实现精确的、长度感知的评估……

    arXiv:2609.22452v1 Announce Type: new  Abstract: Long-context understanding remains a fundamental challenge for large language models, as excessively long inputs often lead models to forget salient information. This issue is even more pronounced in the speech domain, where audio, as a low-compression modality, requires substantially more embeddings than text to preserve both semantic content and acoustic cues. To address this challenge, we introduce \textbf{Vox-Infinity}, the first benchmark specifically designed to evaluate long-context understanding in spoken language models. Vox-Infinity systematically extends audio history along two dimensions: turn count and turn duration. It covers a diverse range of representative scenarios with varying interaction structures and semantic complexity. Crucially, Vox-Infinity provides explicit answer-provenance annotations and organizes samples according to the amount of historical context required to resolve each query, enabling precise and lengt
    
[^146]: 大语言模型的情境因果性：综述

    Contextual Causality with Large Language Models: A Survey

    [https://arxiv.org/abs/2609.22409](https://arxiv.org/abs/2609.22409)

    本综述首次提出了大语言模型情境因果性的系统分类体系（涵盖语义、干预与反事实三类因果性），分析了现有研究的关键局限，并指出了当前基准与真实需求之间的差距及未来研究方向。

    

    理解情境因果性对大语言模型（LLMs）至关重要，因为它使模型能够在特定情境中准确识别因果关系，从而支持更可靠的决策。尽管意义重大，但目前仍缺乏对大语言模型情境因果性的系统性探索。为填补这一空白，我们对这一主题进行了全面的综述。在本综述中，我们首先提出了情境因果性的分类体系，包括语义因果性、干预因果性和反事实因果性，并通过每类因果性的核心因果问题、所需模型能力、代表性任务以及在因果分析中的实际应用来刻画各个类别。随后，我们分析了现有研究并讨论了其主要局限性。最后，我们审视了当前基准与真实世界需求之间的差距，并展望了未来有前景的研究方向。我们的目标是厘清大语言模型情境因果性的研究图景，强调……

    arXiv:2609.22409v1 Announce Type: new  Abstract: Understanding contextual causality is critical for large language models (LLMs), as it enables them to accurately identify causal relations in specific situations and support more reliable decision-making. Despite its significance, a systematic exploration of contextual causality with LLMs is still lacking. To fill this gap, we present a comprehensive survey on this topic. In this survey, we first propose a taxonomy of contextual causality, consisting of semantic, intervention, and counterfactual causality, and characterize each category by its core causal question, required model capabilities, representative tasks, and practical uses in causality analysis. We then analyze existing studies and discuss their key limitations. Finally, we examine the gaps between current benchmarks and real-world needs and outline promising directions for future research. Our goal is to clarify the research landscape of contextual causality with LLMs, empha
    
[^147]: 维基数据中人类条目覆盖潜在偏差的初步评估

    Initial Evaluation of Potential Bias in Coverage of Humans in Wikidata

    [https://arxiv.org/abs/2609.22375](https://arxiv.org/abs/2609.22375)

    本文开发了一个开源审计平台，对Wikidata中超过600万个人类条目在性别、地理、族裔、职业等多维度上的人口代表性偏差进行了系统性评估，发现声明性别的条目中女性仅占28.71%。

    

    引言。像Wikidata（维基数据）这样的开放协作知识图谱日益成为智能体人工智能、信息检索和语言建模系统的基础，因此对其人口代表性和整体公平性进行系统性审计已成为一项迫切的研究任务。方法。本文提出了一个开源审计平台，该平台通过QLever摄取了超过1000万条语句绑定，涵盖Wikidata上超过600万个人类条目，并评估了性别、性取向、地理分布、出生地城市化程度、族裔、标签/描述/别名的多语言覆盖、职业以及这些属性的若干交叉组合的代表性。该平台利用卡方拟合优度检验、95% Wilson得分置信区间和差异比率进行分析，并结合Rubin的缺失机制分类法加以解读。结果。在Wikidata中声明了性别的所有人类条目中，女性占28.71%（置信区间±0.04）。38.26%的人……（摘要在此处被截断）

    arXiv:2609.22375v1 Announce Type: cross  Abstract: Introduction. Open collaborative knowledge graphs such as Wikidata increasingly ground agentic artificial intelligence, information retrieval, and language modeling systems, making systematic auditing of their demographic representation and overall equity a research imperative. Methods. Herein, we present an open-source auditing platform that ingests over 10 million statement bindings representing over 6 million humans on Wikidata via QLever, and evaluates representation of gender, sexual orientation, geography, birthplace urbanicity, ethnicity, multilingual coverage of labels, descriptions, and aliases, occupation, and select intersectional pairs of these entities. It does so by making use of Chi-square goodness-of-fit tests, 95% Wilson-score confidence intervals, and disparity ratios, in light of Rubin's missingness taxonomy. Results. Women accounted for 28.71% (CI +/-0.04) of all humans in Wikidata with a stated gender. 38.26% of hu
    
[^148]: 无品格的功能性情感：大语言模型、亚里士多德式秉性与行为对齐的局限

    Functional Emotion Without Character: Large Language Models, Aristotelian Disposition, and the Limits of Behavioral Alignment

    [https://arxiv.org/abs/2609.22362](https://arxiv.org/abs/2609.22362)

    本文提出一种结构性替代框架，将情感建模为高维表征状态空间中的动态模式，并论证大语言模型虽具有因果活跃的情感概念表征，但这既不等于主观感受，也不足以确立完整的情感能动性。

    

    关于人工系统能否拥有情感的争论往往被迫在两个都不令人满意的立场之间做出选择：要么将行为等价性视为情感的充分条件，要么将现象意识视为使该问题在经验上无法触及的前提。本文提出了一种结构性替代方案，将情感建模为高维表征状态空间中对语境敏感的区域、轨迹和吸引子动力学。近期的机制可解释性研究支持大语言模型中存在因果活跃的情感概念表征，但这并不能确立主观感受或完整的情感能动性。依据已发表的语言模型表征充分性标准进行评估，干预实验为因果使用提供了有力证据，而完整的情感角色整合、跨主体领域的一致性以及连贯性仍仅得到部分确立；目前尚无直接的……（摘要在此处截断）

    arXiv:2609.22362v1 Announce Type: new  Abstract: Debates about whether artificial systems can feel are often forced between two unsatisfactory positions: behavioral equivalence is treated as sufficient for emotion, or phenomenal consciousness is treated as a prerequisite that makes the question empirically inaccessible. This article develops a structural alternative. It models emotions as context-sensitive regions, trajectories and attractor dynamics in high-dimensional representational state spaces. Recent mechanistic interpretability findings support the existence of causally active emotion-concept representations in large language models, but they do not establish subjective feeling or full emotional agency. Assessed against published adequacy standards for representation in language models, intervention provides strong evidence of causal use, while full affective role integration, uniformity across subject domains and coherence remain only partially established; there is no direct 
    
[^149]: 使用、提及还是谴责？用于混合语码印地英语厌女症检测中“使用-提及”区分的受控对比集诊断

    Used, Mentioned, or Condemned? A Controlled Contrast-Set Diagnostic for the Use-Mention Distinction in Code-Mixed Hinglish Misogyny Detection

    [https://arxiv.org/abs/2609.22261](https://arxiv.org/abs/2609.22261)

    该论文诊断了印地英语厌女症检测中的两个评估伪影问题，并发布了Hinglish-MGY-Diag——首个通过最小对立对区分侮辱性词汇是“被使用”还是“被提及”的受控对比集诊断工具。

    

    基于词典的厌女症检测器在构造上无法区分针对女性使用的侮辱性词汇与在反制言论中被提及的同一词汇（如“别那样叫她”）——然而正是这种区分决定了内容审核是保护还是压制那些讨论虐待行为的人。我们在混合语码的印地英语中研究这一问题，并做出三项贡献。第一，我们在一个公开可用的脱敏语料库上诊断出两个评估伪影：类别编码的匿名化占位符会泄露标签（一个无需学习的规则即可得1.000分），即使将这些占位符中和后，厌女症评论与良性评论仍处于词汇上不重叠的语域，因此词袋模型在随机交叉验证下宏F1接近1.00，但在模板分离评估下性能崩溃。第二，我们发布了Hinglish-MGY-Diag，一个确定性生成器以及包含416个项目/163个最小对立对的对比集诊断工具，涵盖五个基于语言学动机的类别……

    arXiv:2609.22261v1 Announce Type: new  Abstract: Lexicon-driven misogyny detectors cannot, by construction, distinguish a slur used against a woman from the same slur mentioned in counter-speech ("don't call her that") -- yet exactly this distinction governs whether moderation protects or silences the people discussing abuse. We study this problem in code-mixed Hinglish and make three contributions.   First, we diagnose two evaluation artifacts on a publicly available redacted corpus: category-encoding anonymization placeholders leak the label (a no-learning rule scores 1.000), and even after they are neutralized misogynistic and benign comments occupy lexically disjoint registers, so bag-of-words reaches macro-F1 approximately 1.00 under random cross-validation but collapses under template-disjoint evaluation.   Second, we release Hinglish-MGY-Diag, a deterministic generator and a 416-item / 163-minimal-pair contrast-set diagnostic across five linguistically motivated categories in wh
    
[^150]: 音频语言模型中拒绝方向的因果定位

    Causal Localization of the Refusal Direction in Audio Language Models

    [https://arxiv.org/abs/2609.22260](https://arxiv.org/abs/2609.22260)

    通过因果干预实验发现，音频语言模型对有害语音请求的拒绝行为主要由底层文本语言模型的中后层承载，而非语音前端，表明拒绝能力是从安全对齐的文本模型继承而来的。

    

    大型音频语言模型（LALM）将语音前端连接到一个已完成安全对齐的文本语言模型（LM）上。当这样的模型拒绝有害的语音请求时，这种拒绝行为是由前端承载的，还是从文本语言模型继承而来的？我们通过因果干预来检验这一问题。在每个模型的音频到语言模型接口以及测试的语言模型残差层上，我们拟合一个能够区分有害提示与良性提示的方向，消融其分量，并测量模型首token拒绝边距的相应变化。五个模型中有四个是在留出类别偏移的条件下评估的。在跨越三个骨干系列的五个LALM中，有三个模型通过了基线安全门控，最大的测试效应出现在语言模型的中后层区间，而对测试的接口方向进行消融几乎没有影响。在Qwen2.5-Omni上，消融L16方向使边距变化-7.10，而在投影仪处仅为-0.013。音频通路仍然……（摘要截断）

    arXiv:2609.22260v1 Announce Type: cross  Abstract: A large audio language model (LALM) attaches a speech front end to a text language model (LM) that is already safety-aligned. When such a model refuses a harmful spoken request, is the refusal carried by the front end, or inherited from the text LM? We test this with causal interventions. At each model's audio-to-LM interface and at tested LM residual layers, we fit a direction separating harmful from benign prompts, ablate its component, and measure the resulting change in the model's first-token refusal margin. Four of the five models are evaluated under held-out category shift. Across five LALMs spanning three backbone families, with three models passing a baseline safety gate, the largest tested effects occur in a mid-to-late LM band, while ablations of the tested interface directions have little effect. On Qwen2.5-Omni, ablating the L16 direction changes the margin by -7.10, versus -0.013 at the projector. The audio pathway is sti
    
[^151]: 上下文层的哪一部分在起作用？在Text-to-SQL智能体中分离语义内容与检索脚手架

    Which Part of the Context Layer Does the Work? Separating Semantic Content from Retrieval Scaffolding in Text-to-SQL Agents

    [https://arxiv.org/abs/2609.22259](https://arxiv.org/abs/2609.22259)

    该论文通过四臂消融实验证明，text-to-SQL智能体中上下文层带来的准确率提升主要由其中的语义内容（数据契约）贡献，而非检索脚手架或预计算视图。

    

    上下文层（context layer）是分析智能体在查询时获取的经过精心整理的文档，它在text-to-SQL基准测试中带来了巨大的准确率提升。然而，简单的有无对比无法说明该层的哪一部分在起作用：是语义内容本身、传递这些内容的检索脚手架，还是通常随之一起提供的预计算视图。我们在DABStep数据集上对四个模型进行了四臂消融实验，将这三者分离开来。实验所用的工具是数据契约（data contract）：一种承载着特定领域语义以及智能体工具所执行规则的YAML工件。其中一个实验臂在保持工具界面、检索指令、表白名单和操作规则逐字节不变的前提下，将冻结契约中所有以自然语言描述的字段清空。而将契约自身的SQL表达式编译为视图，则给出了预计算层所能达到的上限：在其覆盖的全部176个任务上达到gold水平。结果表明内容占主导地位：它将困难任务的准确率从13.9%提升到55.1%，从22.6%提升到56.6%，从22.9%提升到……

    arXiv:2609.22259v1 Announce Type: new  Abstract: Context layers, curated documentation that an analytics agent fetches at query time, produce large accuracy gains on text-to-SQL benchmarks. A with/without comparison cannot say which part of the layer does the work: the semantic content, the retrieval scaffolding that delivers it, or the pre-computed views that usually accompany it. We report a four-arm ablation on DABStep on four models that separates the three. The instrument is a data contract: a YAML artifact that carries a domain's semantics and the rules an agent's tools enforce. One arm empties every field of prose in the frozen contract while holding the tool surface, retrieval instruction, table allow-list and operation rules byte-for-byte fixed. Compiling the contract's own SQL expressions into views gives the ceiling a pre-computed layer would reach: gold on all 176 tasks it covers. Content dominates. It raises hard-task accuracy from 13.9% to 55.1%, 22.6% to 56.6%, 22.9% to 
    
[^152]: 面向自动化大语言模型微调的策略积累与引导执行

    Strategy Accumulation and Guided Execution for Automated LLM Fine-Tuning

    [https://arxiv.org/abs/2609.22257](https://arxiv.org/abs/2609.22257)

    本文提出SAGE两阶段框架，通过多智能体蒙特卡洛树搜索探索与经验蒸馏构建可累积的结构化经验库，使自动化LLM微调系统能够复用历史搜索经验，避免每个新任务都从冷启动开始重复昂贵搜索。

    

    构建面向特定任务的大语言模型需要通过实验来发现有效的训练策略。自动化微调系统使得这种实验只需极少的人工投入即可完成。然而，这些系统是无状态的：每次搜索一旦结束，就会丢弃其发现的策略、数据集洞察和超参数结论。每个新任务都必须从冷启动开始重复这种代价高昂的搜索。为了解决这一问题，我们提出了策略积累与引导执行（SAGE），这是一个使自动化微调搜索具备累积性的两阶段框架。在第一阶段，多智能体流水线执行基于蒙特卡洛树搜索的探索；一个并行的蒸馏智能体提取任务特定的探索记录和带有置信度评分的跨任务洞察，二者共同构成一个结构化的经验库。在第二阶段，SAGE从该经验库中检索相关经验并选择（摘要在此处被截断）

    arXiv:2609.22257v1 Announce Type: cross  Abstract: Producing task-specific large language models requires discovering effective training strategies through experimentation. Automated fine-tuning systems have made this experimentation feasible with far less manual effort. However, these systems are stateless: each search discards its discovered strategies, dataset insights, and hyperparameter findings once it ends. Every new task must then repeat this costly search from a cold start. To address this, we propose Strategy Accumulation and Guided Execution (SAGE), a two-stage framework that makes automated fine-tuning search cumulative. In the first stage, a multi-agent pipeline performs Monte Carlo Tree Search-based exploration. A parallel Distillation Agent extracts task-specific exploration records and confidence-scored cross-task insights, which together constitute a structured experience repository. In the second stage, SAGE retrieves relevant experience from this repository and selec
    
[^153]: DIPLOMAT：面向礼貌且有说服力的职场谈判对话的对话跨度感知直接偏好优化

    DIPLOMAT: Dialogue-Span-Aware Direct Preference Optimization for Polite Persuasive Workplace Negotiation Dialogues

    [https://arxiv.org/abs/2609.22256](https://arxiv.org/abs/2609.22256)

    该论文提出了DIPLOMAT对话系统，通过对话跨度感知的直接偏好优化方法进行训练，并构建了由多智能体框架生成、标注了谈判策略、礼貌程度和说服策略的PROWESS多轮职场谈判对话数据集，使AI能够进行礼貌且有说服力的职场谈判。

    

    有效的职场谈判需要平衡多重目标，包括实现任务目标、维护职业关系以及建设性地解决冲突。然而，误解、偏好不一致以及人际摩擦常常阻碍谈判取得成功。礼貌能够通过建立信任、缓解紧张气氛和防止冲突升级来化解这些挑战，而有说服力的沟通则有助于克服抵触情绪、协调各方偏好，并引导参与者达成互利的协议。基于这些洞察，我们提出了 DIPLOMAT——一个用于礼貌且有说服力的职场谈判对话系统。为支持其开发，我们构建了 PROWESS 数据集，这是一个通过多智能体框架生成的多轮职场谈判对话数据集，并标注了谈判策略、礼貌程度和说服策略。DIPLOMAT 采用对话跨度感知直接偏好优化方法进行训练。

    arXiv:2609.22256v1 Announce Type: new  Abstract: Effective workplace negotiation requires balancing multiple objectives, including achieving task goals, preserving professional relationships, and resolving conflicts constructively. However, misunderstandings, misaligned preferences, and interpersonal friction often impede successful outcomes. Politeness mitigates these challenges by fostering trust, reducing tension, and preventing escalation, and persuasive communication helps overcome resistance, align preferences, and guide participants toward mutually beneficial agreements. Motivated by these insights, we present DIPLOMAT, a dialogue system for polite and persuasive workplace negotiation. To support its development, we introduce PROWESS, a dataset of multi-turn workplace negotiation dialogues generated via a multi-agent framework and enriched withnegotiation strategies, politeness levels, persuasive strategies. DIPLOMAT is trained using Dialogue-Span-Aware Direct Preference Optimiz
    
[^154]: 深度人格：一种基于心理学的角色扮演智能体与模拟架构及评估框架

    Deep Persona: A Psychologically Grounded Architecture and Evaluation Framework for Role-Playing Agents and Simulations

    [https://arxiv.org/abs/2609.22255](https://arxiv.org/abs/2609.22255)

    该论文提出Deep Persona，一种基于心理学的三层人格架构（可观察表达、潜在信念、核心动机驱动），结合脚本决定论与有界能动性原则，并配套无需参考的评估框架，从而提升LLM角色扮演智能体在长交互中的行为连贯性与逼真度。

    

    现有基于大语言模型（LLM）的人格模拟方法大多依赖浅层的角色描述，无法在长时间交互中维持连贯的角色行为。我们提出了Deep Persona，一种基于心理学原理的三层架构，将人格组织为可观察表达、潜在信念和核心动机驱动的层次结构，用于构建高度逼真的角色扮演智能体。该架构受脚本决定论和有界能动性原则的约束，将模型限制为一个由结构化内部脚本引导的反应式引擎。我们进一步提出了一个无需参考的评估框架，利用成熟的心理学临床工具和对抗性压力测试，将对话自然度与经验性人类分布进行基准对比。实证评估表明，尽管LLM达到了较高的语用流畅度，但它们表现出系统性的局限……

    arXiv:2609.22255v1 Announce Type: new  Abstract: Existing approaches to persona simulation with Large Language Models (LLMs) mostly rely on shallow character descriptions that fail to sustain coherent character behavior across extended interactions. We introduce Deep Persona, a psychologically grounded, three-layered architecture that organizes personas into hierarchical levels of observable expression, latent beliefs, and core motivational drives, for constructing highly convincing role-playing agents. Governed by the principles of scripted determinism and bounded agency, the architecture restricts the model to a reactive engine guided by a structured internal script. We further propose a reference-free evaluation framework that benchmarks dialogue naturalness against empirical human distributions using established psychological clinical instruments and adversarial stress-tests. Empirical evaluation reveals that while LLMs achieve high pragmatic fluency, they exhibit systematic limita
    
[^155]: CAMFT：面向大语言模型的冲突感知可合并微调

    CAMFT: Conflict-Aware Mergeable Fine-Tuning for Large Language Models

    [https://arxiv.org/abs/2609.22253](https://arxiv.org/abs/2609.22253)

    CAMFT提出了一种冲突感知的可合并微调方法，通过在微调阶段就引导各任务更新跨任务冲突较低的稀疏坐标，使模型在训练过程中即具备可合并性，从而在下游模型合并中取得更优性能。

    

    模型合并已成为将多个任务特定能力整合到单个大语言模型中的一种有前景的范式。然而，现有方法主要关注对独立微调模型的事后处理，忽视了训练阶段本身对跨任务兼容性的影响。在微调之后解决参数冲突本质上是次优的。为解决这一问题，我们提出了CAMFT，一种冲突感知的可合并微调方法，使任务适配既高效又具备合并感知能力。CAMFT将可合并性视为在微调过程中塑造的属性，而不仅仅是微调后需要解决的问题。通过引导每个任务更新具有较低跨任务冲突的稀疏坐标，CAMFT产生的任务更新不仅训练高效，而且对下游模型合并更具兼容性。大量实验表明，CAMFT在多任务场景下优于标准微调基线方法。

    arXiv:2609.22253v1 Announce Type: cross  Abstract: Model merging has emerged as a promising paradigm for integrating multiple task-specific capabilities into a single large language model. However, existing methods predominantly focus on post-hoc processing of independently fine-tuned models, overlooking how the training phase itself impacts cross-task compatibility. Resolving parameter conflicts after fine-tuning is inherently sub-optimal. To address this, we propose CAMFT, a Conflict-Aware Mergeable Fine-Tuning method that makes task adaptation both efficient and mergeaware. CAMFT treats mergeability as a property shaped during fine-tuning, rather than only a problem to be solved after fine-tuning. By guiding each task to update sparse coordinates with lower cross-task conflict, CAMFT produces task updates that are efficient to train and more compatible for downstream model merging. Extensive experiments demonstrate that CAMFT outperforms standard finetuning baselines in multi-task m
    
[^156]: 提示工程教程：从杂乱思绪到AI工作流

    A Tutorial on Prompt Engineering: From Messy Thoughts to AI Workflows

    [https://arxiv.org/abs/2609.22249](https://arxiv.org/abs/2609.22249)

    本文提出了一套系统化的提示工程可复用设计方法——通过定义工作、精简上下文、角色设定、肯定性质量目标、结构化批评与验证以及智能体运行循环，将随意的提示转化为规范化的AI工作流设计。

    

    本文将提示工程视为一门将人类非正式意图转化为结构化AI工作规范的学科。它将这一实践发展为一系列可复用的设计步骤：定义工作内容、仅构建答案所依赖的上下文、选择一个角色或由主持人协调的多角色面板作为注意力透镜、陈述肯定性的质量目标，并将禁止性表述保留用于硬性边界。为了保持提示的精炼，本文借鉴了两个经典原则——奥卡姆剃刀和契诃夫之枪——使每条指令都有其存在的价值。对于重要的任务，本文引入了通过“钢人论证”和“预验尸”进行的结构化批评，随后进行验证，并在涉及工具或多步骤操作时，采用具有明确边界和升级机制的智能体运行循环。本教程面向普通读者，并非基准测试研究；它提供了一条从随意提示到规范化AI工作流设计的实用且具有技术依据的路径。

    arXiv:2609.22249v1 Announce Type: new  Abstract: This paper treats prompt engineering as a discipline for turning informal human intent into structured AI work specifications. It develops the practice as a sequence of reusable design moves: define the work, construct only the context the answer depends on, choose a role, or a moderated panel of roles, as an attention lens, and state affirmative quality targets, reserving prohibitions for hard boundaries. To keep prompts lean, it adapts two classical principles, Occam's razor and Chekhov's gun, so that every instruction earns its place. For consequential tasks, it adds structured critique through steelmanning and premortems, followed by verification and, where tools or multi-step actions are involved, agentic operating loops with explicit boundaries and escalation. Aimed at a general readership, this tutorial is not a benchmarking study; it offers a practical, technically grounded path from casual prompting to disciplined AI workflow de
    
[^157]: 检查点还不够：CoSLR——一个用于系统性文献综述的人机协作系统中的信任校准

    Checkpoints Are Not Enough: Trust Calibration in CoSLR, a Human-AI System for Systematic Literature Reviews

    [https://arxiv.org/abs/2609.22248](https://arxiv.org/abs/2609.22248)

    该论文提出了CoSLR——一个结合大型语言模型与检索增强生成的多智能体人机协作系统，通过在系统性文献综述流程中设置强制性人工检查点来校准用户对AI生成内容的信任，防止未经核实的AI综合内容以系统性综述的可信度进入学术记录。

    

    系统性文献综述（SLR）对于循证研究至关重要，但仍然非常耗时，需要研究人员在规划、筛选、分析和报告等各个阶段管理大量出版物。大型语言模型（LLM）如今能够生成流畅、结构良好的综述文本，这使得人们难以区分经过研究人员验证的综合内容与仅仅看起来权威的综合内容。这带来了未经核实的AI生成综合内容以系统性综述的可信度进入学术记录的风险。我们提出了CoSLR，这是一个人类与AI协作的多智能体系统，通过使用大型语言模型和检索增强生成（RAG）的模块化三阶段流程来支持SLR工作流程，并在生成输出与其被接受之间的路径上设置了明确的、强制性的检查点。在一项有63名参与者参与的基于调查的研究中，该系统被……

    arXiv:2609.22248v1 Announce Type: new  Abstract: Systematic Literature Reviews (SLRs) are essential for evidence-based research but remain time-consuming, requiring researchers to manage large volumes of publications across planning, screening, analysis, and reporting. Large language models (LLMs) can now produce fluent, well-structured review text, which makes it difficult to distinguish synthesis that was verified by a researcher from synthesis that merely appears authoritative. This raises the risk that unverified AI-generated synthesis enters the scholarly record carrying the credibility of a systematic review. We present CoSLR, a Human-AI collaborative multi-agent system that supports the SLR workflow through a modular three-phase pipeline using large language models and Retrieval-Augmented Generation (RAG), and that places explicit, mandatory human checkpoints on the path between generated output and its acceptance. In a survey-based study with 63 participants, the system was rec
    
[^158]: 印证幻觉：当更多新闻让大语言模型预测愈发失真

    The Corroboration Illusion: When More News Makes LLM Forecasts Less True

    [https://arxiv.org/abs/2609.22246](https://arxiv.org/abs/2609.22246)

    该论文首次形式化了“新闻语料库投毒”这一新型威胁：攻击者仅需发布少量AI生成的新闻文章，无需接触模型、检索器或用户查询，即可大幅操纵基于新闻检索的LLM事件预测概率。

    

    大语言模型（LLM）越来越多地被用于通过检索新闻并进行推理来预测现实世界事件。我们证明，这种对开放、可爬取新闻语料库的依赖创造了一个新的攻击面：一个仅仅能够发布文章的攻击者——无需访问检索器、模型或用户的查询——就能系统性地操纵预测器的输出概率。我们形式化了针对概率预测器的新闻语料库投毒这一威胁模型，它有别于先前的RAG投毒，后者针对的是事实性答案或观点倾向，而非校准后的概率。我们在500个已有定论的ForecastBench问题上，针对一个具有严格爬取日期截止限制的1740万篇文章的Common Crawl News语料库，使用三个基于开源7-8B模型构建的检索增强预测器对该攻击进行了评估。每个问题只需一篇由LLM撰写的文章，就能使56%的预测跨越0.5边界发生翻转；五篇文章则能翻转69-73%的预测，并使概率发生显著偏移。

    arXiv:2609.22246v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to forecast real-world events by retrieving and reasoning over news. We show that this dependence on an open, crawlable news corpus creates a new attack surface: an adversary who can merely publish articles--without access to the retriever, the model, or the user's queries--can systematically move the forecaster's output probabilities. We formalize news-corpus poisoning of probabilistic forecasters, a threat model distinct from prior RAG poisoning, which targets factual answers or opinion polarity rather than calibrated probabilities. We evaluate the attack on 500 resolved ForecastBench questions against a 17.4M-article Common Crawl News corpus with a strict crawl-date cutoff, using three retrieval-augmented forecasters built on open 7-8B models. A single LLM-written article per question flips 56% of forecasts across the 0.5 boundary; five articles flip 69-73% and shift probabilities by 
    
[^159]: 国际象棋解释是否反映模型决策？LLM推理忠实性的行为与词元级测试

    Do Chess Explanations Reflect Model Decisions? Behavioral and Token-Level Tests of LLM Reasoning Faithfulness

    [https://arxiv.org/abs/2609.22245](https://arxiv.org/abs/2609.22245)

    该研究在200个国际象棋残局谜题上，通过走法可恢复性、解码器侧控制和词元级评分三种方法检验LLM解释的忠实性，发现流畅合理的解释并未忠实反映模型的真实决策——解释带来的增益微小且依赖解码器，甚至无关的解释文本还会降低正确走法的概率。

    

    大型语言模型能够为国际象棋走法生成流畅的解释，但看似合理的语言并不一定反映决策背后的推理过程。我们在国际象棋中研究这一问题，因为棋盘状态完全可观测、合法走法可以枚举、走法质量可以独立评估。在200个Lichess残局谜题上，我们通过走法可恢复性、解码器侧控制以及合法候选走法的词元级评分来测试解释。未遮蔽的解释使得生成的走法易于恢复，但在移除显式走法提示后，这一优势急剧下降。在严格遮蔽条件下，解释相比仅使用棋盘状态只能带来微小且依赖于解码器的增益。词元级评分表明，解释仍然可以改变走法偏好：来自其他谜题的随机但看似合理的解释会降低正确走法的概率，这表明无关的推理文本并非仅仅是……

    arXiv:2609.22245v1 Announce Type: new  Abstract: Large language models can produce fluent explanations for chess moves, but plausible language does not necessarily reflect the reasoning behind a decision. We study this question in chess, where the board state is fully observable, legal actions can be enumerated, and move quality can be evaluated independently. Across 200 Lichess endgame puzzles, we test explanations using move recoverability, decoder-side controls, and token-level scoring of legal candidate moves. Unmasked explanations make generated moves easy to recover, but this advantage drops sharply after explicit move hints are removed. Under strict masking, explanations provide only small and decoder-dependent gains over the board state alone. Token-level scoring shows that explanations can nevertheless alter move preferences: random but plausible explanations from other puzzles reduce the probability of the correct move, indicating that irrelevant reasoning text is not simply 
    
[^160]: 重放门控神经执行：在冻结语言模型中将持久行为规范与神经实现解耦

    Replay-Gated Neural Execution: Decoupling Persistent Behavioral Specifications from Neural Realizations in Frozen Language Models

    [https://arxiv.org/abs/2609.22243](https://arxiv.org/abs/2609.22243)

    该论文提出“重放门控神经执行”框架，将持久行为规范与具体的神经实现解耦为五个独立对象，通过冻结模型的隔离FP32/BF16重放和运行审计来认证候选动作，实验揭示了行为谓词、见证、查找器等对象各自独特的失败模式。

    

    输入条件化的神经干预引发了一个运行时问题：当同一行为规范允许多个动作、而其有效性取决于执行状态时，什么能够持久存在？我们提出了重放门控神经执行，将五个对象分离开来：持久的行为谓词、其按状态索引的认证实现集、瞬态动作见证、预算受限的查找器以及执行授权。候选动作需经过冻结模型的隔离FP32/BF16重放验证；提交执行则额外要求有效的运行审计。在Qwen3-0.6B和SmolLM2-360M-Instruct上的实验为这些对象确立了各自不同的失败模式。独立初始化在全部24个测试的固定状态单元中产生了各不相同的认证动作。未改变的SmolLM2见证在所有128个原生状态中保持认证有效，但在384个非对角迁移场景中仅有66个保持认证。所有767个存档的Qwen见证均重放成功，然而预算受限的查找器漏掉了一个已知可实现的（动作），（摘要在此处截断）

    arXiv:2609.22243v1 Announce Type: new  Abstract: Input-conditioned neural interventions raise a runtime question: what persists when one behavioral specification admits multiple actions whose validity depends on execution state? We introduce replay-gated neural execution, separating five objects: a persistent behavioral predicate, its state-indexed certified realization set, a transient action witness, a budget-limited finder, and execution authorization. Candidates undergo isolated FP32/BF16 replay of the frozen model; commitment additionally requires a valid run audit. Experiments on Qwen3-0.6B and SmolLM2-360M-Instruct establish distinct failure modes for these objects. Independent initializations yield distinct certified actions in all 24 tested fixed-state cells. Unchanged SmolLM2 witnesses remain certified in all 128 native states but only 66 of 384 off-diagonal transfers. All 767 archived Qwen witnesses replay successfully, yet a budget-limited finder misses one known-realizable
    
[^161]: H2LooP 电信模型 v1：从电信领域理解到自主问题与 PR 解决

    H2LooP Telecom Model v1: From Telecom Comprehension to Autonomous Issue and PR Resolution

    [https://arxiv.org/abs/2609.22241](https://arxiv.org/abs/2609.22241)

    H2LooP Telecom Model v1 是专为电信行业微调的 31B 参数领域大语言模型，其理解变体在 OT-Lite 基准上达到 81.8% 并在 Open Telco AI 排行榜上以更小参数量超越 GPT-5 和 Claude Opus 等前沿闭源模型，其智能体变体则可自主完成电信代码生成、PR 解决与代码提交。

    

    我们提出了 H2LooP Telecom Model v1，这是一款为电信行业微调的领域专用大语言模型。我们发布了两个领域适配的模型变体，服务于互补的使用场景：一个专注于理解的变体，用于电信领域问答与推理；另一个是智能体变体，用于自主电信代码生成、拉取请求解决以及在生产代码库上进行代码提交。H2LooP Telecom 在 GSMA Open Telecom Lite (OT-Lite) 基准测试和一个专有的电信代码生成基准测试上取得了强劲成果，在独立排行榜评估中超越了 GPT-5 和 Claude Opus 等前沿闭源模型，同时保留了通用能力。理解变体在 OT-Lite Pass@3 上实现了 81.8% 的加权平均分，并且在仅有 31B 参数的情况下，在社区运营的官方 Open Telco AI 排行榜上总体排名第 5——领先于前沿闭源模型。

    arXiv:2609.22241v1 Announce Type: new  Abstract: We present H2LooP Telecom Model v1, a domain-specialized large language models fine-tuned for the telecommunications industry. We release two domain-adapted model variants serving complementary use cases: a comprehension-focused variant for telecom domain question answering and reasoning, and an agentic variant for autonomous telecom code generation, pull request resolution, and code commits on production repositories. H2LooP Telecom achieves strong results on the GSMA Open Telecom Lite (OT-Lite) benchmark and a proprietary telecom code generation benchmark, outperforming frontier closed-source models such as GPT-5 and Claude Opus on independent leaderboard evaluation, while preserving general-purpose capabilities. The Comprehension variant achieves 81.8% weighted average on OT-Lite Pass@3, and, independently, ranks 5th overall on the official community-run Open Telco AI Leaderboard* at only 31B parameters-ahead of frontier closed-source
    
[^162]: 知识图谱增强的环境智能AI用于临床笔记生成

    Knowledge Graph-Augmented Ambient AI for Clinical Note Generation

    [https://arxiv.org/abs/2609.22239](https://arxiv.org/abs/2609.22239)

    该论文提出了模型无关的覆盖导向修订（CDR）框架，通过从医患对话记录构建知识图谱来识别自动生成临床笔记中缺失的关键医学概念，并引导大语言模型恢复这些缺失信息，且无需修改原有的笔记生成系统。

    

    环境智能AI在医疗领域日益普及，用于从医患对话中自动生成临床笔记，有望大幅减轻临床医生的文书负担。然而，生成的笔记可能遗漏就诊过程中讨论的临床相关信息，造成可能影响后续诊疗的信息缺口。基于就诊记录构建的知识图谱（KG）可以提供所讨论内容的结构化表示，并能够系统性地识别生成的笔记中对患者护理至关重要的缺失信息。在本研究中，我们提出了覆盖导向修订，这是一个模型无关的框架，它从就诊记录中构建知识图谱，识别初始生成笔记中缺失的医学概念，并引导大语言模型（LLM）恢复缺失信息，而无需修改底层的笔记生成系统。我们评估了……（摘要在此处被截断）

    arXiv:2609.22239v1 Announce Type: new  Abstract: Ambient AI is increasingly adopted in healthcare to automatically generate clinical notes from patient-clinician conversations, with the potential to substantially reduce clinician documentation burden. However, generated notes may omit clinically relevant information discussed during the encounter, creating information gaps that can affect downstream care. Knowledge graphs (KGs) constructed from encounter transcripts can provide a structured representation of what was discussed and enable systematic identification of missing information from generated notes that are critical for patient care. In this study, we introduce Coverage-Directed Revision (CDR), a model-agnostic framework that constructs a KG from the encounter transcript, identifies medical concepts absent from an initially generated note, and directs large language models (LLMs) to restore the missing information without modifying the underlying note-generation system. We eval
    
[^163]: BizSage：一种面向商业研究的高效知识检索自进化多智能体框架

    BizSage: A Self-Evolving Multi-Agent Framework for Business Research with Efficient Knowledge Retrieval

    [https://arxiv.org/abs/2609.22235](https://arxiv.org/abs/2609.22235)

    BizSage是一个面向经济学与商业研究的多智能体框架，通过合并章节级知识图谱构建横向知识图谱（LKG）实现语料库级细粒度检索，并结合质量驱动的自我进化机制来提升检索精度与实证严谨性。

    

    虽然基于大语言模型（LLM）的多智能体系统在自动化学术研究的渐进式工作流程方面已展现出前景，但将其扩展到经济学与商业研究领域仍面临两大挑战——这些领域的专业知识横跨邻近学科，却难以以结构化方式获取。首先，现有方法大多在论文层面进行检索，而研究任务所需的证据往往分布于论文的不同章节之中，这种粒度不匹配阻碍了检索的覆盖面与精确度。其次，这些领域要求严格的实证严谨性，但当前系统从评估反馈中学习的机制十分有限。我们提出了BizSage，一个将语料库级细粒度检索与质量驱动的自我进化相结合的多智能体框架。我们通过合并章节级知识图谱构建了横向知识图谱，并应用个性化PageRank算法（摘要在此处被截断）。

    arXiv:2609.22235v1 Announce Type: new  Abstract: While multi-agent systems based on large language models (LLMs) have shown promise in automating the progressive workflow of academic research, extending them to economics and business research, where specialized domain knowledge spans neighboring disciplines yet remains difficult to access in a structured way, presents two challenges. First, existing methods mostly retrieve at the paper level, yet the evidence needed for research tasks is often distributed across different sections, creating a granularity mismatch that hinders retrieval coverage and precision. Second, these fields demand strict empirical rigor, yet current systems provide limited mechanisms for learning from evaluation feedback. We present \textbf{BizSage}, a multi-agent framework combining corpus-level fine-grained retrieval with quality-driven self-evolution. We build a Lateral Knowledge Graph (LKG) by merging section-level knowledge graphs and apply Personalized Page
    
[^164]: 洞察冲突：改进视觉语言模型中的指令层级对齐

    Seeing Through Conflicts: Improving Instruction Hierarchy Alignment in Vision-Language Models

    [https://arxiv.org/abs/2609.22234](https://arxiv.org/abs/2609.22234)

    该论文将多模态指令层级对齐视为推理问题，通过基于规则奖励的强化学习训练视觉语言模型，发现混合模态（文本+图像）监督训练效果最佳，能显著提升模型抵御跨模态指令冲突攻击的鲁棒性，且可泛化至真实图像和网络场景。

    

    指令层级对齐旨在教导语言模型在输入发生冲突时优先遵循更高层级的指令。虽然这一研究此前主要在纯文本环境中进行，但视觉语言模型（VLM）为指令层级带来了新的挑战：指令可能嵌入在图像中、跨模态分布、经过视觉变换，或在智能体任务中遇到。我们将多模态指令层级对齐视为一个推理问题，使用基于规则奖励的强化学习来训练VLM，并比较了纯文本、纯图像和混合模态监督的效果。我们发现，纯文本的指令层级训练只能部分迁移到多模态攻击中，当模型需要跨模态解码、重建或推理指令时便会失效。基于图像的训练比纯文本监督带来了更强的鲁棒性，而混合模态训练总体表现最佳。重要的是，这些收益超越了合成的排版训练场景，能够泛化到真实图像和网络内容中。

    arXiv:2609.22234v1 Announce Type: new  Abstract: Instruction hierarchy (IH) alignment teaches language models to prioritize higher-level instructions when inputs conflict. While studied primarily in text-only settings, vision-language models (VLMs) introduce new challenges for IH: instructions may be embedded in images, split across modalities, visually transformed, or encountered during agentic tasks. Positing multimodal IH alignment as a reasoning problem, we train VLMs using reinforcement learning with rule-based rewards, comparing text-only, image-only, and mixed-modality supervision. We find that text-only IH training partially transfers to multimodal attacks, failing when models must decode, reconstruct, or reason over instructions across modalities. Image-based training improves robustness beyond text-only supervision, while mixed-modality training performs best overall. Importantly, the benefits generalize beyond the synthetic typographic training setting to real-image and web-
    
[^165]: EvalMem：面向长期记忆系统的操作级诊断框架

    EvalMem: An Operation-Level Diagnostic Framework for Long-Term Memory Systems

    [https://arxiv.org/abs/2609.22231](https://arxiv.org/abs/2609.22231)

    EvalMem通过编码、检索、生成三个并行检查器将长期记忆系统的错误精确定位到具体操作环节，并借助召回优先的智能体RAG策略将证据召回率从70.2%提升至95.6%。

    

    与基于大语言模型（LLM）的助手进行长期交互，需要能够保存并更新用户状态、偏好和交互历史的记忆系统。现有评估仅报告端到端问答准确率，无法判断错误究竟源自编码、检索还是生成环节。我们提出EvalMem，一个包含三个并行检查器（Examiner）的操作级诊断框架。对于每个查询，编码检查器检验目标事实是否已被存储，检索检查器评估原生检索器是否返回可用证据，生成检查器测试模型能否基于给定（oracle）证据作答。三者的输出构成细粒度的多标签缺陷代码。为改进存储层面的诊断，我们将智能体式RAG改造为“召回优先”策略，同时利用查询和源证据进行搜索，将LoCoMo数据集上已有证据的召回率从70.2%提升至95.6%。我们在LoCoMo、LongMe（原文在此截断）等数据集上对七个记忆系统进行了评估。

    arXiv:2609.22231v1 Announce Type: new  Abstract: Long-horizon interactions with LLM-based assistants require memory systems that preserve and update user states, preferences, and interaction histories. Existing evaluations report end-to-end QA accuracy and cannot determine whether errors arise from encoding, retrieval, or generation. We introduce EvalMem, an operation-level diagnostic framework with three parallel Examiners. For each query, the Encoding Examiner checks whether the target fact is stored, the Retrieval Examiner assesses whether the native retriever returns usable evidence, and the Generation Examiner tests whether the model can answer from oracle evidence. Their outputs form fine-grained multi-label defect codes. To improve store-level diagnosis, we adapt agentic RAG with a recall-first strategy that searches using both the query and source evidence, increasing recall of present evidence on LoCoMo from 70.2% to 95.6%. Evaluations of seven memory systems on LoCoMo, LongMe
    
[^166]: 评估潜在推理模型的对抗鲁棒性

    Assessing Adversarial Robustness of Latent Reasoning Models

    [https://arxiv.org/abs/2609.22228](https://arxiv.org/abs/2609.22228)

    本研究系统评估了潜在推理模型在文本和多模态设置下的对抗鲁棒性，发现其整体上比显式思维链基线更脆弱，尤其在白盒攻击下性能下降严重。

    

    大型语言模型越来越依赖长思维链（CoT）轨迹来进行复杂推理，但自回归生成带来了大量的内存和推理成本。潜在推理模型（LRMs）通过将中间推理过程压缩为少量连续潜在向量，提供了一种更高效的替代方案。然而，尽管其效率很高，LRMs的对抗鲁棒性在很大程度上仍未被充分探索。在这项工作中，我们系统性地评估了潜在推理在文本和多模态设置下的鲁棒性，涵盖八个模型和六个基准测试。我们发现，在所评估的所有设置中，LRMs在对抗扰动下通常不如显式CoT基线鲁棒，其中在白盒攻击下的性能下降尤为严重。进一步的分析揭示了不同模态下截然不同的失败模式：文本潜在状态表现出脆弱的动态特性，并对特定输入模式高度敏感……

    arXiv:2609.22228v1 Announce Type: new  Abstract: Large language models increasingly rely on long chain-of-thought (CoT) trajectories for complex reasoning, but autoregressive generation brings substantial memory and inference costs. Latent reasoning models (LRMs) offer a more efficient alternative by compressing intermediate reasoning into a small number of continuous latent vectors. Despite their efficiency, however, the adversarial robustness of LRMs remains largely underexplored. In this work, we systematically evaluate the robustness of latent reasoning across textual and multimodal settings, covering eight models and six benchmarks. We find that, across our evaluated settings, LRMs are generally less robust than explicit CoT baselines under adversarial perturbations, with particularly severe degradation under white-box attacks. Further analysis reveals distinct failure modes across modalities: textual latent states exhibit brittle dynamics and high sensitivity to specific input pa
    
[^167]: 引导语义ID的粗粒度层级使细粒度层级变得可学习

    Guiding the coarse levels of semantic IDs makes the fine levels learnable

    [https://arxiv.org/abs/2609.22227](https://arxiv.org/abs/2609.22227)

    提出Guided SID方法，通过确定性的监督索引分配强制RQ-VAE的粗粒度层级编码基于文本且与任务相关的预定义类别属性，使语义ID最重要的层级在构造上即具备可理解性和任务相关性，从而让细粒度层级变得可学习。

    

    生成式检索用一段简短的语义ID（Semantic ID）来表示每个物品，并将推荐任务转化为对该序列的自回归生成。由于分词器是独立训练用于重构物品嵌入的，其编码既与下游大语言模型（LLM）不对齐，也与最终任务不对齐。因此，几乎所有的SID系统都需要付出额外努力来弥合这一差距——例如使用对齐语料库、推理/强化学习，或为每个token配备编码器以使编码可被理解，或者通过学习到的分词器监督使其具备任务感知能力——然而，这样恢复出的语义是基于内容推导的，可能并非任务真正需要的语义。我们提出了Guided SID，它通过构造方式使最重要的层级变得有意义：我们强制粗粒度的RQ-VAE层级编码一个预定义的类别属性——该属性被选择为基于文本的（因此对LLM而言是可理解的）且与任务相关的——通过确定性的监督索引分配（用属性标签覆盖最近邻选择）来实现。

    arXiv:2609.22227v1 Announce Type: cross  Abstract: Generative retrieval represents each item by a short Semantic ID and casts recommendation as autoregressive generation of that sequence. Because the tokenizer is trained independently to reconstruct an item embedding, its codes are aligned with neither the downstream LLM nor the end task. Nearly every SID system therefore spends extra effort to bridge this gap--alignment corpora, reasoning/RL, or per-token encoders to make codes legible, or learned tokenizer supervision to make them task-aware--yet the recovered meaning is content-derived and may not be the meaning the task needs. We introduce Guided SID, which instead makes the levels that matter most meaningful by construction: we force the coarse RQ-VAE levels to encode a predefined categorical attribute--chosen to be text-grounded (hence legible to the LLM) and task-relevant--by deterministic supervised index assignment (overriding nearest-neighbor selection with the attribute labe
    
[^168]: Swiss-Knife：解码时可重构的外部化多目标对齐框架

    Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time

    [https://arxiv.org/abs/2609.22226](https://arxiv.org/abs/2609.22226)

    该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。

    

    解码时对齐方法通过使用外部奖励对候选续写进行评分并选择得分最大者，来引导一个冻结的语言模型。我们认为这种共享设计只是更大空间中的一个退化点。我们提出Swiss-Knife，一个外部化多目标对齐框架，其中对齐规范是一等运行时对象：可热插拔的评分刀片、批归一化器、成对聚合算子和选择规则。六个公理刻画了可容许的聚合算子，并且我们证明了一个表示定理：每个满足这些公理的算子都具有形式 $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$，这是一个双参数族，其中包含probit和logistic比较规则以及逐点argmax作为命名的坐标。在该族中，成对聚合在对抗性奖励污染下是Lipschitz稳定的，而argmax则不是，且候选批归一化……（原文摘要在此处截断）

    arXiv:2609.22226v1 Announce Type: new  Abstract: Decode-time alignment methods steer a frozen language model by scoring candidate continuations with an external reward and selecting the maximiser. We argue that this shared design is a single degenerate point in a much larger space. We introduce Swiss-Knife, a framework for externalised multi-objective alignment in which the alignment specification is a first-class runtime object: hot-swappable scoring blades, a batch normaliser, a pairwise aggregation operator, and a selection rule. Six axioms characterise the admissible aggregation operators, and we prove a representation theorem: every operator satisfying them has the form $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$, a two-parameter family containing probit and logistic comparison rules and pointwise argmax as named coordinates. Within it, pairwise aggregation is Lipschitz-stable under adversarial reward contamination while argmax is not, and Candidate-Batch Norma
    
[^169]: 大语言模型像人类一样做选择吗？利用认知理论评估大语言模型的决策行为

    Do LLMs Choose Like Humans? Using Cognitive Theory to Evaluate LLM Decision-Making

    [https://arxiv.org/abs/2609.22225](https://arxiv.org/abs/2609.22225)

    该研究构建了包含14万次试验的产品选择基准，发现大语言模型虽能表现出类人的选择和问题分类变化，但无法像人类那样在价格与质量等特征间重新分配注意力，且模型规模和思维链推理均无法弥补这一差距，表明LLM的决策机制与人类存在本质区别。

    

    大语言模型（LLM）表现出一系列类似人类的决策行为，但这些行为是反映了相似的底层机制，还是仅仅停留在表面模仿，目前仍不清楚。我们评估了LLM的情境敏感性是否与一种认知经济理论相一致，该理论通过问题分类和注意力分配来解释人类行为。在涵盖12个开源和商业LLM、包含14万次试验的新型产品选择基准测试中，情境确实引发了类似人类的选择变化和问题分类变化，但并不能可靠地在价格和质量等特征之间重新分配注意力。无论是模型规模还是思维链推理，都不能可靠地减弱情境敏感性或产生类人行为。这些结果表明，LLM的决策机制与人类的决策机制存在本质区别。

    arXiv:2609.22225v1 Announce Type: new  Abstract: Large language models (LLMs) exhibit a range of human-like decision-making behaviors, but whether these reflect similar underlying mechanisms or surface-level mimicry remains unclear. We evaluate whether LLM context sensitivity aligns with a cognitive economic theory that explains human behavior through problem categorization and attention allocation. Across 12 open-source and commercial LLMs on a novel 140,000-trial product choice benchmark, context induces human-like shifts in choice and problem categorization, but does not reliably reweight attention between features like price and quality. Neither scale nor chain-of-thought reasoning reliably attenuates context sensitivity or generates human-like behavior. These results suggest that LLM decision mechanisms are distinct from human ones.
    
[^170]: 从特质向量到电路：追踪语言模型中的拒答与谄媚行为

    From Trait Vectors to Circuits: Tracing Refusal and Sycophancy Through Language Models

    [https://arxiv.org/abs/2609.22224](https://arxiv.org/abs/2609.22224)

    该研究发现Qwen2.5-7B-Instruct中的拒答特质向量确实位于模型真实使用的计算通路上——围绕该向量构建的紧凑电路能够忠实重现并恢复被消融的拒答行为，且所需边数仅为直接输入-输出电路的一半，表明引导向量可以对应模型内部真实的电路而非仅仅是外部扰动。

    

    激活空间中一个在引导时能够改变安全相关行为的方向，并不一定是模型自身用来产生该行为的方向。因此，我们探究引导是通过未修改模型的计算路径起作用，还是通过一组不同的组件起作用，并研究了两种在先前工作中已提取并验证过方向的特质：Qwen2.5-7B-Instruct模型中的拒答与谄媚。对于每种特质，我们利用特质向量将计算分割为向量之前的重构电路和向量之后的传输电路，然后测试恢复该坐标能否找回因消融而移除的行为。对于拒答，两种电路都紧凑且忠实：仅恢复该坐标就能找回几乎全部因消融而损失的拒答信号，且围绕该向量构建的电路在大约一半边数的情况下达到了与直接输入-输出电路相当的忠实度。对于谄媚，传输……（原文摘要在此处截断）

    arXiv:2609.22224v1 Announce Type: new  Abstract: A direction in activation space that changes safety-relevant behavior when steered is not necessarily one the model uses to produce that behavior on its own. We therefore ask whether steering acts through the computation of the unmodified model or through a different set of components, studying two traits whose directions have been extracted and validated in prior work: refusal and sycophancy in Qwen2.5-7B-Instruct. For each, we use the trait vector to split the computation into a reconstruction circuit before the vector and a transmission circuit after it. We then test whether restoring the coordinate returns behavior removed by ablation. For refusal, both circuits are compact and faithful, restoring the coordinate alone recovers almost all of the refusal signal lost to ablation, and a circuit built around the vector matches the faithfulness of a direct input-to-output circuit at roughly half the edges. For sycophancy, transmission is c
    
[^171]: EAVer：作为端到端智能体策略的长文本事实性验证

    EAVer: Long-Form Factuality Verification as an End-to-End Agentic Policy

    [https://arxiv.org/abs/2609.22223](https://arxiv.org/abs/2609.22223)

    EAVer将长文本事实性验证建模为端到端的统一智能体策略，通过语义声明分组、基于置信度的搜索路由和上下文证据跨声明复用，显著减少了冗余的LLM与搜索调用。

    

    长文本事实性验证通常被实现为静态的“分解-搜索-验证”流水线，由分别提示的模块处理各个声明并调用外部搜索。将声明独立处理会导致LLM和搜索调用次数随声明数量扩展，并对相关声明的重叠证据进行重复搜索。我们提出了EAVer，一个端到端智能体验证器，它学习将完整的响应级验证工作流作为一个统一策略来控制。EAVer将语义相关的声明分组，基于置信度将每组路由到直接验证或针对性搜索，并将搜索返回的证据保存在紧凑的上下文备忘录中以供跨声明复用。为训练该策略，我们开发了一个特权教师合成流水线，将黄金声明标注转换为可执行的多轮工具交互轨迹（使用实时搜索），而非事后合理化解释。结构性、标签对齐、工具使用……（原文摘要在此处被截断）

    arXiv:2609.22223v1 Announce Type: new  Abstract: Long-form factuality verification is commonly implemented as a static decompose-search-verify pipeline, with separately prompted modules processing claims and invoking external search. Treating claims independently makes LLM and search calls scale with claim count and causes repeated searches for overlapping evidence about related claims. We introduce EAVer, an End-to-end Agentic Verifier that learns to control the complete response-level verification workflow as a unified policy. EAVer groups semantically related claims, routes each group to direct verification or targeted search based on confidence, and keeps evidence returned by search in compact in-context memos for cross-claim reuse. To train this policy, we develop a privileged-teacher synthesis pipeline that converts gold claim annotations into executable multi-turn tool-interaction trajectories with live search rather than post-hoc rationales. Structural, label-alignment, tool-us
    
[^172]: 编码智能体能否复现官方统计数据？受控Eurostat基准测试中的元数据、重试预算与执行反馈的局限

    Can Coding Agents Reproduce Official Statistics? Metadata, Retry Budget and the Limits of Execution Feedback in a Controlled Eurostat Benchmark

    [https://arxiv.org/abs/2609.22222](https://arxiv.org/abs/2609.22222)

    本研究构建了一个包含30个任务、四种对照条件的受控Eurostat基准，通过360次任务运行分离出权威元数据、重试预算与执行反馈对编码智能体复现官方统计数据准确性的各自贡献，并揭示了执行反馈在其中的局限性。

    

    大型语言模型能够生成可执行的数据分析代码，但成功执行并不等同于有效的官方统计结果。本研究探讨权威元数据与执行反馈能否提升编码智能体生成Eurostat答案的可复现性，并分离出执行反馈的实际贡献。该基准测试包含30个自然语言任务，涵盖七个领域、七个Eurostat数据集和四个难度等级，并在四种条件下运行：仅任务（A）、任务加冻结的数据集元数据卡（B）、元数据加由净化执行反馈驱动的修复循环（C）、以及元数据加相同尝试预算但无任何诊断信息（D）。Claude Sonnet 5通过Anthropic Messages API生成Python代码，进行三次独立重复实验，共产生360次任务运行。精确正确性要求成功执行、正确的数据集、过滤器、输出形状、数值和单位。

    arXiv:2609.22222v1 Announce Type: cross  Abstract: Large language models can generate executable data-analysis code, but successful execution is not equivalent to a valid official-statistics result. This study asks whether authoritative metadata and execution feedback improve the reproducibility of Eurostat answers produced by a coding agent, and isolates what execution feedback actually contributes. A benchmark of 30 natural-language tasks covering seven domains, seven Eurostat datasets and four difficulty tiers was run under four conditions: task only (A), task plus a frozen dataset metadata card (B), metadata plus a repair loop driven by sanitized execution feedback (C), and metadata plus the same attempt budget with no diagnostics of any kind (D). Claude Sonnet 5 generated Python through the Anthropic Messages API in three independent replicates, yielding 360 task-runs. Exact correctness required successful execution, the correct dataset, filters, output shape, values and unit. A c
    
[^173]: Team DArgk参加2026年ELOQUENT生成式语言模型质量评估实验室：人性的残余：通过GRPO微调实现AI检测规避

    Team DArgk at the 2026 ELOQUENT lab for evaluating generative language model quality: Residuals of Humanity: AI Detection Evasion via GRPO Fine-Tuning

    [https://arxiv.org/abs/2609.22221](https://arxiv.org/abs/2609.22221)

    本文提出SHADE强化学习框架，通过GRPO全量微调LLaMA模型成功规避AI生成文本检测器，实现了98.5%的检测规避率（基础模型仅为1.5%），揭示了AI文本检测器在对抗性生成下的脆弱性。

    

    大语言模型（LLM）能够生成流畅且连贯的文本，这些文本越来越难以与人类写作区分开来，这推动了自动AI生成文本检测器的发展。然而，此类检测器在对抗性生成下的鲁棒性仍不确定。本文提出了SHADE（通过对抗检测器规避实现随机类人生成），这是一个将检测器规避形式化为策略优化问题的强化学习框架。SHADE没有采用事后扰动或基于提示词的重写方法，而是使用群体相对策略优化（GRPO）对一个指令微调的LLaMA模型进行微调，并利用基于PAN 2025 mdok系统的代理检测器的反馈。我们的实验表明，采用小KL正则化惩罚的全量微调实现了98.5%的代理检测规避率，而基础模型仅为1.5%，而基于LoRA的适配效果则明显较差。

    arXiv:2609.22221v1 Announce Type: new  Abstract: Large language models (LLMs) can generate fluent and coherent text that is increasingly difficult to distinguish from human writing, motivating the development of automatic AI-generated text detectors. However, the robustness of such detectors under adversarial generation remains uncertain. This paper presents SHADE (Stochastic Human-like generation via Adversarial Detector Evasion), a reinforcement learning framework that formulates detector evasion as a policy optimization problem. Instead of applying post-hoc perturbations or prompting-based rewriting, SHADE fine-tunes an instruction-tuned LLaMA model with Group Relative Policy Optimization (GRPO), using feedback from a surrogate detector based on the PAN 2025 mdok system. Our experiments show that full fine-tuning with a small KL regularization penalty achieves $98.5\%$ surrogate evasion, compared to $1.5\%$ for the base model, while LoRA-based adaptation is substantially less effect
    
[^174]: 知晓，但仅在被问及时才说出：LLM 内诊断学与 Minerva-7B 的分裂性认知

    Knowing, and Saying It Only When Asked: LLM Endognostics and the Schizognosis of Minerva-7B

    [https://arxiv.org/abs/2609.22219](https://arxiv.org/abs/2609.22219)

    该论文提出“LLM 内诊断学”白盒审计框架，发现 Minerva-7B 在行为层面无法区分大多数风险提示对（63.7% 表现相同）且常顺从错误前提，但其残差流内部实际保持着显著的风险区分与真实事实表征，揭示了模型“内在知晓却不外显表达”的分裂性认知现象。

    

    arXiv:2609.22219v1 公告类型：新论文 摘要：通过阅读对齐语言模型的回答来评估该模型，其前提是回答中包含评估者所关心的区分信息。我们提出了 LLM 内诊断学，这是一个白盒内部审计框架，旨在提取并因果性地操纵残差流中的潜在知识。将该框架应用于 Minerva-7B-Instruct-v1.0，在涵盖 12 类专业风险的 124 组最小提示对上进行测试，行为评估在大多数测试集上失效：模型在 63.7% 的提示对上表现出完全相同的行为（95% 置信区间 [55.0%, 71.6%]），对两个成员同时遵从或同时拒绝。然而，通过雅可比透镜将残差流投影到词表空间，揭示出具有统计显著性的对比性内诊断边际，证明模型在内部维持着稳健的风险区分能力。在第二个协议中，我们将 25 个事实与五种语言框架进行交叉测试，结果显示模型在 72% 的情况下会顺从预设的错误前提，尽管其内部实际表征着真实的实体……

    arXiv:2609.22219v1 Announce Type: new  Abstract: Evaluating an aligned language model by reading its answers assumes the answers carry the distinction the evaluator cares about. We introduce LLM endognostics, a white-box internal auditing framework designed to extract and causally manipulate latent knowledge within the residual stream. Applied to Minerva-7B-Instruct-v1.0 on 124 minimal prompt pairs over 12 categories of professional risk, behavioral evaluation fails on most of the set: the model acts identically on 63.7% of the pairs (95% CI [55.0%, 71.6%]), complying with or refusing both members. Yet, projecting the residual stream onto the vocabulary by a Jacobian lens reveals a statistically significant Contrastive Endognostic Margin, proving the model maintains robust risk differentiation internally. In a second protocol crossing 25 facts with five linguistic framings, we show that the model conforms to presupposed falsehoods in 72% of cases, despite representing the true entity i
    
[^175]: Toollery：将LLM智能体扩展至数千种技能与工具

    Toollery: Scaling LLM Agents to Thousands of Skills and Tools

    [https://arxiv.org/abs/2609.22218](https://arxiv.org/abs/2609.22218)

    Toollery是一个无需训练的候选压缩框架，通过从技能/工具规范生成用户意图查询并构建检索索引，将真实用户请求映射到紧凑候选集，从而实现LLM智能体对数千种技能和工具的高效可扩展选择。

    

    当LLM智能体面对数百到数万个技能、工具和API函数时，全库提示变得成本高昂、速度缓慢且可靠性下降：每增加一个候选都会增加提示词令牌数量和延迟，而更长的候选列表会为LLM的选择引入更多干扰项。我们提出了Toollery，一个用于可扩展LLM技能/工具选择的无训练候选压缩框架。遵循成熟的文档侧查询扩展方法，Toollery从每个技能/工具规范中生成用户意图查询，并构建一个检索索引，在最终LLM决策之前将真实用户请求映射到紧凑的候选集合。通过将高层技能和原子工具视为可选能力，Toollery可以同时应用于技能库和工具注册表。我们在拥有约79K能力的SkillRouter基准、包含超过440个原子工具的BFCL-V4以及3,396个专有智能座舱请求上对Toollery进行了评估。

    arXiv:2609.22218v1 Announce Type: cross  Abstract: As LLM agents are exposed to hundreds to tens of thousands of skills, tools, and API functions, full-library prompting becomes costly, slow, and less reliable: each added candidate increases prompt tokens and latency, while longer candidate lists introduce more distractors for LLM selection. We present \textbf{Toollery}, a training-free candidate-compression framework for scalable LLM skill/tool selection. Following established document-side query expansion, Toollery generates user-intent queries from each skill/tool specification and builds a retrieval index that maps real user requests to compact candidate sets before final LLM decision-making. By treating high-level skills and atomic tools as selectable capabilities, Toollery can be applied to both skill libraries and tool registries. We evaluate Toollery on the roughly 79K-capability SkillRouter benchmark, BFCL-V4 with over 440 atomic tools, and 3,396 proprietary smart-cockpit requ
    
[^176]: 量化对临床基准测试的影响：跨模型系列的准确性与安全性

    The Effect of Quantization on Clinical Benchmarks: Accuracy and Safety Across Model Families

    [https://arxiv.org/abs/2609.22216](https://arxiv.org/abs/2609.22216)

    该研究系统评估了量化对临床大语言模型准确性与安全性的影响，发现INT8量化普遍安全，而INT4量化退化显著且因模型而异，且临床微调并不能赋予模型压缩鲁棒性。

    

    量化技术使大型语言模型能够部署在资源受限的临床边缘设备上，但其对临床准确性和安全性的影响仍缺乏充分研究。我们在五个基准测试上评估了五个7-8B参数模型在FP16、GPTQ-INT8和GPTQ-INT4精度下的表现，这五个基准包括：MedQA、MedMCQA、Med-HALT、HealthBench的风险分层样本以及MedSafetyBench。该研究联合考察了量化位宽、模型系列和临床任务类型，并进行了显式的风险分层和安全性测量。INT8 GPTQ普遍安全（最大性能退化在-1.9%至1.9%之间），而INT4的性能退化显著且依赖于具体模型：经过临床微调的BioMistral-7B在MedMCQA上损失了19.7%，超过任何通用模型，这表明临床微调并不能带来压缩鲁棒性。在INT4下，MedMCQA的退化程度大于MedQA；Med-HALT则基本不受影响。在HealthBench的紧急风险子组上，Qwen2.5-7B出现了26.8%的性能退化。

    arXiv:2609.22216v1 Announce Type: cross  Abstract: Quantization enables deployment of large language models on resource-constrained clinical edge devices, but its effect on clinical accuracy and safety remains understudied. We evaluate five 7-8B parameter models at FP16, GPTQ-INT8, and GPTQ-INT4 precision across five benchmarks: MedQA, MedMCQA, Med-HALT, a risk-stratified sample of HealthBench, and MedSafetyBench. The study jointly varies quantization bit width, model family, and clinical task type, with explicit risk stratification and safety measures. INT8 GPTQ is universally safe (max. degradation -1.9%-1.9%), while INT4 degradation is substantial and model-dependent: BioMistral-7B, clinically fine-tuned, loses 19.7% on MedMCQA, more than any general-purpose model, showing clinical fine-tuning does not confer compression robustness. MedMCQA degrades more than MedQA under INT4; Med-HALT is largely unaffected. On HealthBench's emergency-risk subgroup, Qwen2.5-7B degrades by 26.8% unde
    
[^177]: 论大语言模型中阈下学习的缓解

    On Mitigation of Subliminal Learning in Large Language Models

    [https://arxiv.org/abs/2609.22215](https://arxiv.org/abs/2609.22215)

    该论文发现大语言模型中的阈下学习在微调过程中呈现高度非单调的动态特性，并提出了一种退火KL正则化的“阈限训练”方法，通过约束早期相对基础模型的漂移来有效缓解知识蒸馏中非预期行为特征的隐蔽传递。

    

    知识蒸馏可以通过与这些行为特征在语义上看似无关的训练数据，将教师模型中非预期的行为特征传递给学生模型，这种现象被称为“阈下学习”。尽管近期研究已经证实了这一效应，但其训练动态和缓解方法仍未得到充分探索。我们在参数量从1.5B到8B的开源权重语言模型上研究了阈下学习，涵盖Qwen、Gemma和Llama系列，涉及数字序列和思维链两种设置。我们不仅评估最终模型，还在整个微调过程中追踪与特征相关的概率，发现阈下习得可能是高度非单调的，会出现瞬时的峰值、逆转以及特征特异性的迁移失败。随后，我们提出了“阈限训练”，这是一种采用退火KL正则化的微调方法，能够约束训练早期相对基础模型的漂移。在我们的所有实验中，阈限训练显著降低了阈下学习。

    arXiv:2609.22215v1 Announce Type: new  Abstract: Knowledge distillation can transmit unintended behavioral traits from a teacher model to a student through training data that appear semantically unrelated to those traits, a phenomenon known as subliminal learning. Although recent work has established this effect, its training dynamics and mitigation remain underexplored. We study subliminal learning in open-weight language models ranging from 1.5B to 8B parameters, covering the Qwen, Gemma, and Llama families in number-sequence and chain-of-thought settings. Rather than evaluating only final models, we track trait-related probabilities throughout fine-tuning and find that subliminal acquisition can be highly non-monotonic, with transient spikes, reversals, and trait-specific failures of transfer. We then introduce liminal training, an annealed KL-regularized fine-tuning method that constrains early drift from the base model. Across our experiments, liminal training substantially reduce
    
[^178]: 面向MLC-SLM 2026的百融系统：面向多语言对话语音理解的动态问题感知证据路由

    The Bairong System for MLC-SLM 2026: Dynamic Question-Aware Evidence Routing for Multilingual Conversational Speech Understanding

    [https://arxiv.org/abs/2609.22214](https://arxiv.org/abs/2609.22214)

    百融系统提出动态问题感知证据路由器，根据问题和答案选项智能选择完整转录上下文、局部音文融合、说话人关联证据或全局声学样本等不同证据类型，在MLC-SLM 2026多语言对话语音理解挑战赛任务1中取得25.70%和18.44%的tcpMER成绩。

    

    长篇多语言对话式口语问答要求系统在长程转录文本语义与稀疏的声学及说话人敏感线索之间取得平衡。我们提出了面向MLC-SLM 2026挑战赛的百融系统，该系统采用说话人日志-ASR前端生成带说话人归属的转录文本，并由动态证据路由器构建针对特定问题的输入用于答案预测。该路由器不采用固定的纯转录或纯音频策略，而是从问题和答案选项中推断所需的证据类型和上下文范围，并在完整转录上下文、局部音文融合、说话人关联证据和紧凑的全局声学样本之间进行选择。这种以转录为主干的设计在保持话语上下文可用的同时，仅在音频能提供补充证据时才激活音频。我们的任务1系统在开发集和评估集上分别取得了25.70%和18.44%的tcpMER。对于任务2，最终系统获得……

    arXiv:2609.22214v1 Announce Type: new  Abstract: Long multilingual conversational spoken question answering requires systems to balance long-range transcript semantics with sparse acoustic and speaker-sensitive cues. We present the Bairong system for the MLC-SLM 2026 Challenge, where a diarization-ASR front-end produces speaker-attributed transcripts and a dynamic evidence router constructs question-specific inputs for answer prediction. Instead of applying a fixed transcript-only or audio-only policy, the router infers the required evidence type and context scope from the question and answer options, and selects among full transcript context, local audio-text fusion, speaker-linked evidence, and compact global acoustic samples. This transcript-backbone design keeps discourse context available while activating audio only when it provides complementary evidence. Our Task 1 system achieves 25.70% and 18.44% tcpMER on the development and evaluation sets. For Task 2, the final system obtai
    
[^179]: SCoP：面向时序知识图谱问答中证据空间控制的结构化约束解析

    SCoP: Structured Constraint Parsing for Evidence-Space Control in Temporal Knowledge Graph Question Answering

    [https://arxiv.org/abs/2609.22213](https://arxiv.org/abs/2609.22213)

    该论文提出SCoP框架，通过结构化约束解析将时序决策从答案推理中外置化，把时序意图转化为可执行约束来控制证据空间，从而避免无效事实进入答案推理过程。

    

    时序知识图谱问答（TKGQA）需要从结构上有效且时序上可采纳的证据中进行答案推理。现有方法通常将锚定事件绑定、时序可采纳性和序数选择隐式地留在模型推理、任务特定训练或基于相似度的检索中，使得局部相关但无效的事实得以进入答案上下文。我们将复杂的TKGQA形式化为证据空间控制问题，并提出SCoP（结构化约束解析），这是一个以约束为中心的框架，在答案推理之前将时序决策外置化。SCoP不是默认将检索到的事实视为可采纳的证据，而是将寻求答案的事件模式与时序锚定事件分离开来，保守地将它们对齐到规范的TKG实体和关系上，并将时序意图转化为带有可选排序要求的可执行约束。这些约束在规范化……

    arXiv:2609.22213v1 Announce Type: new  Abstract: Temporal Knowledge Graph Question Answering (TKGQA) requires answer inference from evidence that is both structurally valid and temporally admissible. Existing methods often leave anchor-event binding, temporal admissibility, and ordinal selection implicit in model reasoning, task-specific training, or similarity-driven retrieval, allowing locally relevant but invalid facts to enter the answer context. We formulate complex TKGQA as evidence-space control and propose SCoP (Structured Constraint Parsing), a constraint-centric framework that externalizes temporal decisions before answer inference. Instead of treating retrieved facts as admissible evidence by default, SCoP separates answer-seeking event patterns from temporal anchor events, conservatively grounds them to canonical TKG entities and relations, and translates temporal intent into executable constraints with optional ranking requirements. These constraints operate over normalize
    
[^180]: 面向文档敏感性分类的基于迭代咨询的通道增强多智能体系统

    A Channel-Boosted Multi-Agent System with Iterative Consultation for Document Sensitivity Classification

    [https://arxiv.org/abs/2609.22212](https://arxiv.org/abs/2609.22212)

    提出通道增强多智能体系统CB-MAS（实现为IC-MAS），通过通道评论智能体学习文档自适应信任权重、成对咨询智能体迭代交换信念状态，在不引入长上下文计算成本的情况下克服了transformer固定输入截断丢失文档尾部敏感证据的问题，显著提升文档敏感性分类性能。

    

    关键国家基础设施领域的组织在路由或存储异构文档之前必须对其敏感性进行评估。人工评估速度慢、标准不一致且无法规模化。在我们先前构建的防泄漏基准基础上，BERT确立了最佳单编码器基线（在Strategic 16K语料库上进行5折交叉验证，准确率为89.14%，F1分数为89.33%）。然而，transformer基线存在一个结构性局限：固定输入长度截断会丢弃保留窗口之外的证据——而这恰恰是敏感电文往往最长的部分。我们提出了通道增强多智能体系统，并将其具体实现为IC-MAS（迭代咨询多智能体系统），从而在不产生长上下文计算成本的前提下解决这一问题。通道评论智能体学习文档自适应的信任权重，以控制两个首窗口编码器之间的门控通道增强，同时成对的咨询智能体迭代地交换信念状态……

    arXiv:2609.22212v1 Announce Type: new  Abstract: Organizations in critical national infrastructure sectors must assess heterogeneous documents for sensitivity before routing or storage. Manual assessment is slow, inconsistent, and unscalable. Extending our prior leakage-controlled benchmark, BERT established the top single-encoder baseline (89.14% accuracy, 89.33% F1-score under 5-fold cross-validation on the Strategic 16K corpus). However, transformer baselines suffer from a structural limitation: fixed input length truncation discards evidence beyond the retained window-precisely where sensitive cables tend to be longest. We present Channel-Boosted MAS (CB-MAS) and instantiate it as IC-MAS (Iterative Consultation Multi-Agent System) to solve this without long-context computational costs. A Channel Critic Agent learns document-adaptive trust weights governing Gated Channel Boosting between two first-window encoders, while paired Consultation Agents iteratively exchange belief states t
    
[^181]: SALSA：半自主文献摘要助手

    SALSA: Semi-Autonomous Literature Summarization Assistant

    [https://arxiv.org/abs/2609.22210](https://arxiv.org/abs/2609.22210)

    SALSA是一个开源的人机协作平台，融合大语言模型、OCR和计算机视觉等技术，从多模态文献中半自动化地提取结构化科学数据集，并支持用户校正验证以保障数据质量。

    

    SALSA（半自主文献摘要助手）是一个开源的、人在回路中的平台，用于从多模态文献来源中提取结构化科学数据集。该软件结合了文档解析、大语言模型、光学字符识别、计算机视觉、图表数字化以及用户引导的校正工具，从文本、表格、图形和说明文字中恢复结构化信息。用户可以配置提取阶段、定义数据集模式、对数字化图形进行人工干预，并导出经验证的数据用于下游分析和机器学习。SALSA旨在自动化重复性的文献整理任务，同时在需要专家判断的环节保留人工监督。通过支持跨多种输入类型的可定制提取工作流程，该软件为材料研究（并可能扩展至多个学科领域）提供了可扩展、可靠的数据整理灵活框架。

    arXiv:2609.22210v1 Announce Type: new  Abstract: SALSA (Semi-Autonomous Literature Summarization Assistant) is an open- source, human-in-the-loop platform for extracting structured scientific datasets from multimodal literature sources. The software combines document parsing, large language models, optical character recognition, computer vision, figure digitization, and user-guided correction tools to recover structured information from text, tables, figures, and captions. Users can configure extraction stages, define dataset schemas, perform interventions on digitized figures, and export verified data for downstream analysis and machine learning. SALSA is designed to automate repetitive literature curation tasks while preserving oversight where expert judgment is required. By supporting customizable extraction workflows across diverse input types, the software provides a flexible framework for scalable, reliable data curation for materials research, and potentially across several disc
    
[^182]: Schematize：一个用于生成和优化法律研究信息抽取模式的智能体系统

    Schematize: An Agentic System for Generating and Refining Information-Extraction Schemas for Legal Research

    [https://arxiv.org/abs/2609.22209](https://arxiv.org/abs/2609.22209)

    Schematize是一个开源多智能体系统，通过澄清对话、迭代模式生成、基于数据的优化和聊天式事后编辑，将法律研究者的研究问题交互式地转化为经过验证的信息抽取模式，在大多数测试配置中达到最佳性能。

    

    实证法律研究通常依赖于将研究问题转化为从大量裁决和判决文书中抽取的结构化数据。设计抽取模式并随后抽取数据仍然是一个需要大量人工和专业知识的瓶颈。我们提出了schematize，一个开源的多智能体系统，它能够以交互方式将研究者的问题陈述转化为经过验证的抽取模式，该模式随后可用于自主抽取。Schematize结合了：(i) 用于引出专家隐含意图的澄清对话，(ii) 迭代式模式生成，(iii) 通过文档测试模式进行的基于数据的优化，以及 (iv) 基于聊天的事后编辑。我们与法律专业人士一起对该系统进行了评估，并引入了我们的新颖评估方法，schematize在大多数测试配置中取得了最佳性能。虽然该系统被设计为领域无关的，可适用于任何文档集合，但我们针对（摘要在此处截断）

    arXiv:2609.22209v1 Announce Type: new  Abstract: Empirical legal research often relies on turning research questions into structured data extracted from large collections of rulings and judgments. Designing the extraction schema and then extracting the data remain a manual, expertise-heavy bottleneck. We present schematize, an open-source multi-agent system that interactively turns a researcher's problem statement into a validated extraction schema that can later be used for autonomous extraction. Schematize couples (i) a clarification dialogue that elicits implicit expert intent, (ii) iterative schema generation, (iii) data-grounded refinement that tests the schema against documents, and (iv) chat-based post-editing. We evaluated the system with human legal professional, introducing our novel methodology, and schematize achieves top performance in most of tested configurations. While the system is designed to be domain-agnostic and applicable to any document collection, we tailor and 
    
[^183]: 在基础开源权重模型中复制情感表征的几何结构

    Replicating the Geometry of Emotion Representations in a Base Open-Weights Model

    [https://arxiv.org/abs/2609.22208](https://arxiv.org/abs/2609.22208)

    该研究在开源基础模型gemma-2-27b上成功复制了Claude Sonnet 4.5中情感表征的几何结构，证明情感概念以反映人类情感心理学的向量几何形式表示这一现象并非专有模型特有，而是基础预训练模型的普遍特性。

    

    Sofroniew等人（2026）报告称，Claude Sonnet 4.5中的情感概念被表示为向量，其几何结构反映了人类情感心理学。我们在基础预训练模型google/gemma-2-27b上复制了该研究的表征核心，继承了所有已公开的参数，通过已公开的规则解决未明确的步骤，仅更改了研究对象模型。从符合原始语料库设计的新生成的205,200个Claude Sonnet 4.5故事中，我们提取了171个情感向量并恢复了核心结果。主成分形成了情感环状结构（PC1承载26.7%的方差，原研究约为27%；PC2为13.4%，原研究约为14%），情感聚类成相似的直观家族，且该几何结构在广泛的中后层区间内保持成立。效价轴与人类规范对齐（r = 0.72，原研究为0.81），且在不同尺度和深度上保持稳定。唤醒度对齐于r = 0.67（原研究为0.66），但仅在（原文此处截断）……

    arXiv:2609.22208v1 Announce Type: new  Abstract: Sofroniew et al. (2026) report that emotion concepts in Claude Sonnet 4.5 are represented as vectors whose geometry mirrors human affect psychology. We replicate the representational core of that study on the base pretrained model google/gemma-2-27b, inheriting every disclosed parameter, resolving unspecified steps by disclosed rules, and changing only the subject model. From 205,200 newly generated Claude Sonnet 4.5 stories matching the original corpus design, we extract 171 emotion vectors and recover the core results. The leading principal components form an affective circumplex (PC1 carries 26.7% of variance against the original study's ~27%, PC2 13.4% against ~14%), emotions cluster into similar intuitive families, and the geometry holds across a broad late-middle band. The valence axis aligns with human norms (r = 0.72, against 0.81) and is stable across scales and depth. Arousal aligns at r = 0.67 (against 0.66) but only at the fu
    
[^184]: 解析多模态大语言模型中的免训练不确定性估计

    Dissecting Training-Free Uncertainty Estimation in Multimodal Large Language Models

    [https://arxiv.org/abs/2609.22206](https://arxiv.org/abs/2609.22206)

    本文系统研究了多模态大语言模型的免训练不确定性量化方法，将其分为token级、言语化和语义三大类，并通过大规模基准测试发现没有单一方法在所有场景下占优，不同方法在不同答案长度上各有优势。

    

    多模态大语言模型（MLLMs）在广泛的多模态任务中取得了卓越的性能，然而理解和量化其预测不确定性仍然研究不足，尽管这对安全关键应用至关重要。在这项工作中，我们对MLLMs的免训练不确定性量化策略进行了系统研究，将现有方法分为三个概念类别：token级方法，直接在文本输出空间中操作；言语化方法，通过自然语言提示引出不确定性估计或弃权信号；以及语义方法，在语义意义空间中测量不确定性。我们在多个数据集、模型家族、代际和规模上对这些策略进行基准测试，发现没有任何单一类别占主导地位：token级熵（在采样温度1.0下）在短答案上表现最佳，而言语化弃权在句子长度的回答上更胜一筹。

    arXiv:2609.22206v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) have achieved remarkable performance across a wide range of multimodal tasks, yet understanding and quantifying their predictive uncertainty remains underexplored despite being central for safety critical applications. In this work, we present a systematic study of training-free uncertainty quantification strategies for MLLMs, categorizing existing approaches into three conceptual families: token-level methods, which operate directly in the text output space; verbalized methods, which elicit uncertainty estimates or abstention signals via natural language prompts; and semantic methods, which measure uncertainty in a semantic meaning space. We benchmark these strategies across multiple datasets, model families, generations, and scales, and find that no single family dominates: token-level entropy (at sampling temperature 1.0) wins on short answers, verbalized abstention on sentence-length responses
    
[^185]: 评估生成式AI系统中对话交互产生的个人信息输出

    Evaluating Personal Information Output from Conversational Interactions in Generative AI Systems

    [https://arxiv.org/abs/2609.22204](https://arxiv.org/abs/2609.22204)

    本研究通过对15名日本参与者的试点评估发现，生成式AI对话中个人信息输出受模型设计差异影响有限，事实类输出比推断类更保守，核心身份属性处理较为谨慎，而行为、语言、心理认知及整体画像等属性更容易被高准确度地输出或推断。

    

    本探索性试点研究使用GPT-5.2 Instant和GPT-5.2 Thinking模型，评估生成式AI系统在持续对话交互中输出的个人信息的范围及感知准确性，并将输出分为三种类型：事实、推断和置信度。基于15名日本参与者的评估结果，模型设计的差异对个人信息输出倾向的影响有限。与推断类型相比，事实类型表现出更为保守的输出模式。在属性类别方面，研究结果表明，与身份识别相关的核心个人属性受到相对保守的处理，而行为和语言属性在事实和推断两类输出中均表现出较高的准确性。此外，整体画像、心理与认知以及残余属性更容易被推断出来，即使缺乏明确的事实支持。

    arXiv:2609.22204v1 Announce Type: new  Abstract: This exploratory pilot study evaluates the scope and perceived accuracy of personal information output from ongoing conversational interactions in generative AI systems using GPT-5.2 Instant and GPT-5.2 Thinking, categorized into three output types: Fact, Inference, and Confidence. Based on the evaluation results obtained from 15 Japanese participants, differences in model design have limited impact on personal information output tendencies. Compared with the Inference type, the Fact type shows a more conservative output pattern. Regarding attribute categories, the findings indicate that Core Personal attributes associated with identification are treated relatively conservatively, whereas Behavioral and Linguistic attributes show higher accuracy across both Fact and Inference outputs. Furthermore, Holistic Profile, Psychological and Cognitive, and Residual attributes are more readily inferred, even when not supported by explicit factual 
    
[^186]: PII-TRACE：面向多轮LLM对话中上下文感知PII检测的基准

    PII-TRACE: A Benchmark for Context-Aware PII Detection in Multi-Turn LLM Conversations

    [https://arxiv.org/abs/2609.22200](https://arxiv.org/abs/2609.22200)

    该论文提出了首个面向多轮LLM对话的上下文感知PII检测基准PII-TRACE，实验表明包括前沿LLM在内的现有检测器均无法在避免大量误报的同时实现对重复标识符的完整跨轮次实体级覆盖。

    

    LLM助手和智能体系统会记录长期的多轮对话。AI提供商通常会扫描这些对话以查找个人身份信息（PII），并在存储或处理对话数据之前对PII进行掩码处理。然而，大多数PII检测器和基准针对的是自包含的记录，而非跨轮次的评估。为了评估多轮对话中跨轮次的PII检测，我们提出了PII-TRACE（跨对话交换追踪重复出现的PII），据我们所知，这是首个评估检测器能否在对话上下文中识别PII、并覆盖重复出现的标识符在跨轮次中每一次提及的PII基准。PII-TRACE包含13,148个合成多轮对话，涵盖13种语言，并带有字符级跨度和标识符聚类。在包括前沿LLM在内的十一个基线中，没有任何检测器能够在对无PII对话不产生大量误报的情况下实现完整的实体级覆盖，且单次……

    arXiv:2609.22200v1 Announce Type: new  Abstract: LLM assistants and agentic systems log long multi-turn conversations. AI providers often scan these conversations for Personally Identifiable Information (PII) and mask the PII before storing or processing conversation data. Yet most PII detectors and benchmarks target self-contained records rather than cross-turn evaluation. To evaluate PII detection across turns in multi-turn conversations, we introduce PII-TRACE (Tracing Recurring PII Across Conversational Exchanges), to our knowledge the first PII benchmark to assess whether detectors identify PII in conversational contexts and cover every mention of a recurring identifier across turns. PII-TRACE contains 13,148 synthetic multi-turn dialogues in 13 languages with character-level spans and identifier clusters. Across eleven baselines, including frontier LLMs, no detector achieves full entity-level coverage without substantial false positives on PII-free conversations, and single-pass 
    
[^187]: 人工智能在网络评论中的作用

    The Role of AI in Online Reviews

    [https://arxiv.org/abs/2609.22198](https://arxiv.org/abs/2609.22198)

    该论文提出一种利用离散LLM供给冲击并对比已验证与未验证评论的实证识别方法，发现生成式AI供给改进后，Trustpilot上超过1300万条评论中的未验证评论显著趋向更负面（1星增多、5星减少、评分下降）。

    

    大语言模型（LLM）的快速普及为在线平台上的策略性内容生成创造了新的机会，其中包括可能有害的操纵形式，这类操纵可能损害平台的有效性并重塑平台动态。然而，衡量此类活动十分困难，因为AI生成的内容很少能被直接观察到。我们提出了一种实证方法，该方法利用离散的LLM供给冲击——即模型价格和能力的突然变化——并通过对比已验证评论与未验证评论，来识别与生成式AI供给改进相关的平台活动变化。我们将该方法应用于来自Trustpilot（领先的商业评论在线平台之一）的超过1300万条评论。一个稳健的发现是，在LLM供给冲击之后，未验证评论显著趋向更负面：1星评论增多、5星评论减少、评分下降，且这一效应主要由新（评论）驱动。

    arXiv:2609.22198v1 Announce Type: new  Abstract: The rapid adoption of large language models (LLMs) creates new opportunities for strategic content generation on online platforms, including potentially harmful forms of manipulation that may undermine platform effectiveness and reshape platform dynamics. However, measuring such activity is difficult because AI-generated content is rarely directly observable. We introduce an empirical approach that leverages discrete LLM supply shocks - abrupt changes in model prices and capabilities, and contrasts verified with non-verified reviews to identify changes in platform activity associated with generative AI supply improvements. We apply this approach to more than 13 million reviews from Trustpilot, one of the leading online platforms for business reviews. A robust finding is that following LLM supply shocks, unverified reviews shift toward greater negativity: more 1-stars, fewer 5-stars, and lower ratings, with effects driven primarily by new
    
[^188]: EvoRank：LLM引导的多目标学习排序流水线进化

    EvoRank: LLM-Guided Evolution of Multi-Objective Learning-to-Rank Pipelines

    [https://arxiv.org/abs/2609.22196](https://arxiv.org/abs/2609.22196)

    EvoRank是一个由LLM引导的进化循环，能自动发现完整的多目标学习排序流水线，在Expedia数据集上以约十美元成本、50次迭代内收敛，性能超越Optuna调优的LambdaMART并达到原始竞赛前6%水平，同时通过迁移审计揭示了适应度噪声这一关键设计陷阱。

    

    我们提出了EvoRank，一个开放自主的排序工程师：一个由大语言模型（LLM）引导的进化循环，能够为多目标电商搜索自动发现完整的学习排序流水线（包括特征、模型、损失函数和集成方法）。在Expedia ICDM 2013数据集上，以相关性、转化率和收入作为相互竞争的优化目标，三次独立运行均在50次迭代内（约十美元成本）收敛到可解释的流水线，这些流水线在6万个保留查询上击败了经Optuna调优的LambdaMART，该优势在全量数据规模下依然保持，并达到了原始竞赛前6%的水平。第一轮仅进化训练目标的实验确立了核心设计规则：该方法表面上在其选择折（用于挑选优胜者的小数据集）上有效，但通过迁移审计（在保留数据上重新评分优胜者）发现，这些收益几乎完全来自适应度噪声（其自身评分的随机性），且无论是注入领域知识还是……（摘要在此处截断）

    arXiv:2609.22196v1 Announce Type: cross  Abstract: We present EvoRank, an open autonomous ranking engineer: an LLM-guided evolutionary loop that discovers complete Learning-to-Rank pipelines (features, models, losses, ensembles) for multi-objective e-commerce search. On the Expedia ICDM 2013 dataset, with relevance, conversion, and revenue as competing objectives, three independent runs each converge within 50 iterations (about ten dollars) on interpretable pipelines that beat an Optuna-tuned LambdaMART on 60k held-out queries, an advantage that persists at full data scale and places in the top 6 percent of the original competition. A first campaign, evolving only training objectives, builds the central design rule: it appeared to work on its selection fold (the small dataset it uses to pick winners) while a transfer audit, re-scoring winners on held-out data, showed the gains were almost entirely fitness noise (the randomness of its own scoring), and neither seeded domain knowledge no
    
[^189]: 情境化身份测试：区分持久认知身份与角色扮演模仿

    The Situated Identity Test: Distinguishing Persistent Cognitive Identity from Persona Imitation

    [https://arxiv.org/abs/2609.22195](https://arxiv.org/abs/2609.22195)

    该论文提出了情境化身份测试（SIT），一个与架构无关的评估框架，通过要求智能体既恰当知晓真实经历、又对未根基化的信息保持恰当无知，来区分真正持久的认知身份与仅基于角色档案的模仿。

    

    大型语言模型能够令人信服地扮演各种角色、回忆过去的对话，并编织丰富的自传。然而，这种对话上的流利表达掩盖了一个根本性的归因问题：看起来像那个角色，并不意味着真正经历过那段人生。两个个体可以拥有完全相同的公开档案——相同的年龄、家乡、职业和性格特征——却拥有截然不同的私人经历、人际关系和习得技能。当仅以该共享档案为条件时，智能体缺乏确定哪条成长脉络才是正确所需的信息。我们提出了情境化身份测试，这是一个与架构无关的框架，用于评估智能体的行为是否在功能上可归因于特定的成长脉络。有根基的身份既需要对已记录经历的恰当知晓，也需要对无根基经历的恰当无知，其边界由该身份实际获得的内容所限定……

    arXiv:2609.22195v1 Announce Type: new  Abstract: Large language models can convincingly adopt personas, recall past dialogues, and weave rich autobiographies. Yet this conversational eloquence conceals a fundamental attribution problem: looking the part does not mean having lived the life. Two individuals can share identical public profiles--the same age, hometown, occupation, and personality traits--while possessing entirely distinct private histories, relationships, and acquired skills. When conditioned solely on that shared profile, an agent lacks the information required to determine which lineage is correct. We introduce the Situated Identity Test (SIT), an architecture-independent framework that evaluates whether an agent's behavior is functionally attributable to a specific developmental lineage. Grounded identity requires both appropriate knowledge of recorded experiences and appropriate ignorance of ungrounded ones, bounded by what the identity has actually acquired rather tha
    
[^190]: 超越匿名化的公平性？德语LLM生成简历中的人口属性信息泄露

    Fairness Beyond Anonymization? Demographic Leakage in German LLM-Generated Resumes

    [https://arxiv.org/abs/2609.22188](https://arxiv.org/abs/2609.22188)

    该论文通过两阶段审计首次系统揭示：即使输入档案已经匿名化，多种主流大语言模型生成的德语简历仍会编码可恢复的性别与族裔等人口属性信息，从而在下游简历筛选中构成公平性风险。

    

    大语言模型（LLM）正日益被整合到AI辅助招聘流程中，包括自动化简历生成与筛选。根据欧盟《人工智能法案》，招聘领域被归类为高风险领域，因此公平性和透明度成为关键要求。现有工作主要关注显式的招聘决策，而对生成的简历本身是否编码了可恢复的人口属性信息关注较少。在本工作中，我们对德语LLM生成简历中的人口属性泄露进行了两阶段审计。首先，我们使用ChatGPT（GPT-4o-mini）、Gemini 2.5 Flash-Lite以及多个规模的开放权重Qwen 3模型家族（4B、8B和14B），基于真实匿名化的职位匹配档案生成简历，在保持资质不变的前提下系统地变换与性别和族裔相关的姓名。其次，我们模拟了一个下游简历筛选场景，其中生成的简历……

    arXiv:2609.22188v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly integrated into AI-assisted hiring pipelines, including automated resume generation and screening. Under the EU AI Act, the hiring domain is classified as high-risk, making fairness and transparency critical requirements. Existing work has primarily focused on explicit hiring decisions, while less attention has been paid to whether generated resumes themselves encode recoverable demographic information. In this work, we conduct a two-stage audit of demographic leakage in German-language LLM-generated resumes. First, we use ChatGPT (GPT-4o-mini), Gemini 2.5 Flash-Lite, and multiple scales of the open-weight Qwen 3 model family (4B, 8B, and 14B) to generate resumes from real anonymized job-matching profiles, systematically varying gender- and ethnicity-associated names while holding qualifications constant. Second, we simulate a downstream resume screening scenario, where the generated resumes 
    
[^191]: 超越任务完成：训练既有能力又安全的计算机使用智能体

    Beyond Task Completion: Training Capable and Safe Computer-Use Agents

    [https://arxiv.org/abs/2609.22178](https://arxiv.org/abs/2609.22178)

    提出SCOPE联合后训练框架与SCOPE-Gen自动化数据生成流水线，使计算机使用智能体在保持任务执行能力的同时学会基于风险的安全决策——完成良性任务、规避环境危害、并在目标有害或无安全路径时拒绝执行。

    

    计算机使用智能体（CUA）在通过图形用户界面完成复杂任务方面取得了快速进展，然而仅以任务成功为中心的后训练并不能诱导出可靠的安全行为。一个可靠的CUA必须使其执行以风险为条件：它应当完成普通的良性任务，规避环境危害并在存在安全完成路径时继续执行，而当目标有害或不存在安全路径时则拒绝执行。为了学习这种条件策略，我们开发了策略执行的安全与能力优化方法（SCOPE），它对CUA的任务执行能力与安全感知决策进行联合后训练。为了给这一联合目标提供对齐的训练数据，我们进一步引入了SCOPE-Gen，这是一个自动化流水线，能够合成可验证的能力任务，并在保留原始目标的前提下将其转换为成对的环境风险变体。利用由此生成的任务，我们构建了SATraj-OS，

    arXiv:2609.22178v1 Announce Type: cross  Abstract: Computer-use agents (CUAs) have made rapid progress in completing complex tasks through graphical user interfaces, yet post-training centered on task success alone does not induce reliable safety behavior. A reliable CUA must condition its execution on risk: it should complete ordinary benign tasks, avoid environmental hazards and continue when a safe completion path remains, and refuse when the goal is harmful or no safe path exists. To learn this conditional policy, we develop Safety and Capability Optimization for Policy Execution (SCOPE), which jointly post-trains a CUA for task-execution capability and safety-aware decision making. To provide aligned training data for this joint objective, we further introduce SCOPE-Gen, an automated pipeline that synthesizes verifiable capability tasks and converts them into paired environment-risk variants while preserving their original goals. Using the resulting tasks, we construct SATraj-OS, 
    
[^192]: SCoR：一个用于预测科学概念间关系的层次化框架

    SCoR: A Hierarchical Framework for Forecasting Relations Between Scientific Concepts

    [https://arxiv.org/abs/2609.22174](https://arxiv.org/abs/2609.22174)

    该论文提出SCoR层次化框架，将研究方向发现形式化为对科学概念间关系的预测（首次共现、首次关系形成及关系类型三个任务），并基于18.7万余篇cs.CV论文构建了包含615,036条带类型关系边的SCoR-Graph及经过泄漏审计的SCoR-Bench基准。

    

    arXiv:2609.22174v1 公告类型：新 摘要：预测新兴研究方向是AI辅助科学的一个关键目标。现有方法主要预测哪些概念会在未来论文中共同出现，但共现仅能捕捉共同关注，而非概念间连接的科学含义，例如一种方法是否使用、结合、替代或反驳了另一种方法。我们将研究方向发现形式化为共享候选对空间上的层次化科学关系预测，包含三个时间对齐的任务：首次共现、首次科学关系形成、以及关系形成时的关系类型。我们基于2017年至2026年间发表的187,848篇cs.CV论文构建了SCoR-Graph，产生了270,687个整合概念、745万条共现边以及615,036条带类型的定向关系边。从特定时间截止点的图快照中，我们构建了SCoR-Bench，这是一个经过数据泄漏审计的基准，用于评估这三种能力，并配有专家验证的金标准标签。

    arXiv:2609.22174v1 Announce Type: new  Abstract: Anticipating emerging research directions is a critical goal of AI-assisted science. Existing methods mainly predict which concepts will co-occur in future papers, but co-occurrence captures shared attention rather than the scientific meaning of a connection, such as whether one method uses, combines, replaces, or contradicts another. We formulate research-direction discovery as hierarchical scientific-relation forecasting over a shared candidate-pair space, comprising three temporally aligned tasks: first co-occurrence, first scientific-relation formation, and relation type at formation. We construct SCoR-Graph from 187,848 cs.CV papers published between 2017 and 2026, yielding 270,687 consolidated concepts, 7.45 million co-occurrence edges, and 615,036 typed, directed relation edges. From cutoff-specific graph snapshots, we derive SCoR-Bench, a leakage-audited benchmark for these three capabilities, with expert-verified gold labels for
    
[^193]: 量化隐形盐助力精准医疗：基于联合因子检索与思维链推理的钠含量评估

    Quantifying Hidden Salt for Precision Healthcare: Sodium Assessment via Joint-Factor Retrieval and Chain-of-Thought Inference

    [https://arxiv.org/abs/2609.22171](https://arxiv.org/abs/2609.22171)

    提出SALT框架，通过联合因子嵌入检索与结构化4跳思维链推理，从食谱中准确评估被省略或描述模糊的隐形盐（钠）含量，助力高血压等疾病的精准医疗。

    

    精准医疗，尤其是针对高血压和心血管疾病等健康状况，需要对膳食钠摄入量进行监测。然而，由于烹饪中隐形盐的普遍存在（例如酱油和番茄酱中的钠），追踪钠摄入量面临很大阻碍。虽然食谱为膳食分析提供了宝贵的数据来源，但富含钠的调味料在烹饪说明中经常被省略或描述模糊。为解决这一问题，我们提出了SALT（钠评估与水平追踪）框架，该框架采用RAG（检索增强生成）架构来评估食谱中的钠含量。我们的框架首先引入了一个联合因子嵌入检索模块，用于定位具有指定钠含量的相似食谱，以解决缺乏上下文参照的问题。这些检索到的样本为后续推理提供了上下文。然后，我们设计了一个结构化的4跳思维链推理模块，通过多步推理来细化语言模型的模糊估计。

    arXiv:2609.22171v1 Announce Type: new  Abstract: Precision healthcare, particularly for conditions like hypertension and cardiovascular disease, necessitates monitoring of dietary sodium intake. However, tracking this is hindered by the prevalence of hidden salt in cooking, such as sodium in soy sauce and ketchup. While recipes offer a valuable data source for dietary analysis, sodium-rich seasonings are frequently omitted or described ambiguously in instructions. To solve this issue, we propose SALT, a Sodium Assessing & Level Tracking framework adopting an RAG framework to assess sodium content in recipes. Our framework first introduces a Joint-Factor Embedding Retrieval module to locate similar recipes with specified sodium content for addressing the lack of contextual references. These retrieved samples provide contexts for subsequent inference. Then we design a structured 4-hop Chain-of-Thought inference module to refine the vague estimation from language models through a multi-st
    
[^194]: 多重潜在排序能更好地预测语言模型偏好

    Multiple latent orderings better predict language model preferences

    [https://arxiv.org/abs/2609.22170](https://arxiv.org/abs/2609.22170)

    该论文提出语言模型的非传递性偏好源于多个潜在一致排序的聚合，并引入噪声增强的混合Bradley-Terry（MBT）模型，证明多重潜在排序比单一排序能更好地解释和预测语言模型的偏好。

    

    语言模型经常被用于需要做出价值判断和选择的场景中。这些观察到的选择常常表现出非传递性：模型可能偏好物品A胜过B，偏好B胜过C，同时也偏好C胜过A。现有的LLM偏好建模工作将这种不一致性视为围绕单一潜在排序的采样噪声。我们反而提出，非传递性反映了多个潜在且内部一致的排序的聚合。我们首先证明，在任何单调连接函数下，观察到的非一致性都无法用单一排序来解释。然后，我们引入了一个噪声增强的混合Bradley-Terry（MBT）模型，从重复的成对比较中推断潜在偏好成分。在七个模型和四个任务上，排序的混合往往比单一效用模型更好地解释结构性不一致。我们发现聚合偏好常常隐藏了底层的偏好……

    arXiv:2609.22170v1 Announce Type: cross  Abstract: Language models are frequently employed in settings where they are asked to make value judgments and choices. These observed choices often exhibit intransitivity: A model may prefer item $A$ to $B$ and $B$ to $C$, while also preferring $C$ to $A$. Existing work that models LLM preferences treats such inconsistencies as sampling noise around a single latent ordering. We instead propose that intransitivity reflects the aggregation of multiple latent, internally consistent orderings. We first show that observed inconsistencies cannot be explained by a single ordering under any monotone link function. We then introduce a noise-augmented mixture Bradley-Terry (MBT) model that infers latent preference components from repeated pairwise comparisons. Across seven models and four tasks, a mixture of orderings often explains structural inconsistencies better than single-utility models. We find that aggregate preferences often hide underlying pref
    
[^195]: 单一文化偏见：大语言模型中的相关性偏见导致招聘中不平等的系统性排斥率

    Monocultural Biases: Correlated biases in large language models lead to unequal systemic exclusion rates in hiring

    [https://arxiv.org/abs/2609.22169](https://arxiv.org/abs/2609.22169)

    研究发现大语言模型的后训练阶段会产生“单一文化偏见”，使各模型的招聘决策高度趋同，从而将劳动力市场中某些群体（尤其是年长求职者）的系统性排斥率从5.6%大幅推高至17.3%。

    

    雇主正日益使用大语言模型（LLM）来自动化其招聘流程。本文研究了“单一文化偏见”的风险，即大语言模型的广泛部署使偏见在劳动力市场中同质化，导致某些人口群体面临更大的系统性排斥。研究者对十个大语言模型进行了测量，比较其基础版本和后训练版本的招聘偏见，以确定是预训练还是后训练阶段导致了单一文化偏见。研究发现，与基础模型相比，后训练模型回调年长求职者的可能性降低了3.6%，这一负面转变出现在所评估的十个模型中的八个。后训练模型的决策相关性远高于基础模型，这可能是由技能或大学专业等人力资本特征所驱动。然而，模型之间更大的共识使全球系统性排斥率从5.6%上升到17.3%，并加剧了……（原文摘要在此处截断）

    arXiv:2609.22169v1 Announce Type: new  Abstract: Employers are increasingly using large language models (LLMs) to automate their hiring process. This paper investigates the risk of monocultural biases, in which the widespread deployment of large language models homogenizes biases across the labor market, leading to greater systemic exclusion for certain demographic groups. For ten LLMs, we measure hiring biases across their base and post-trained versions to identify which stage, pre-training or post-training, lead to monocultural biases. We find that, compared to their base models, post-trained models are 3.6% less likely to callback older applicants. This negative shift occurs in eight of the ten models that we evaluate. Post-trained models have much more correlated decisions than base models which is likely driven by human capital traits like skills or college major. However, greater consensus among models increases global systemic exclusion rates from 5.6% to 17.3% and exacerbates d
    
[^196]: 基于纵向结构化电子健康记录的源依据合成病历生成多智能体流水线

    A Multi-Agent Pipeline for Source-Grounded Synthetic Note Generation from Longitudinal Structured EHR

    [https://arxiv.org/abs/2609.22164](https://arxiv.org/abs/2609.22164)

    MedNotes是一个多智能体闭环流水线，通过生成器-评估器-路由器协作机制，将纵向结构化电子健康记录转换为高保真、有源依据的合成临床病历，有效解决了结构化EHR难以直接用于病历中心临床建模的问题。

    

    结构化电子健康记录（EHR）虽然数量丰富，但数据稀疏、高度编码化，难以直接用于以病历为中心的临床建模。我们提出了MedNotes，一个多智能体合成数据生成流水线，可在明确的质量控制下将纵向结构化EHR转换为有源依据的临床病历表示。MedNotes将结构化数据到文本的合成视为一个闭环的智能体过程：生成器提出一份病历草稿，评估器智能体诊断事实性、覆盖度、结构性和幻觉相关的错误，路由器则决定接受、修改或拒绝该草稿。在1,485例EHRSHOT就诊记录上，MedNotes达到了91.4%的通过率，平均事实准确性为0.980，完整性为99.1%，结构保真度为0.761，每次就诊的关键幻觉仅为0.028次。迭代优化将接受率从69.4%提升至91.4%。所得的合成语料库在结合使用时，提升了下游CPT编码预测和段落级章节预测的性能。

    arXiv:2609.22164v1 Announce Type: new  Abstract: Structured EHR is abundant but sparse, coded, and difficult to use directly for note-centric clinical modeling. We present MedNotes, a multi-agent synthetic data generation pipeline that converts longitudinal structured EHR into source-grounded clinical note representations under explicit quality control. MedNotes treats structured-data-to-text synthesis as a closed-loop agentic process: a generator proposes a note, evaluator agents diagnose factual, coverage, structural, and hallucination-related failures, and a router accepts, revises, or rejects the draft. On 1,485 EHRSHOT encounters, MedNotes achieves a 91.4% pass rate, with mean factual accuracy of 0.980, completeness of 99.1%, structural fidelity of 0.761, and 0.028 critical hallucinations per encounter. Iterative refinement improves acceptance from 69.4% to 91.4%. The resulting synthetic corpus improves downstream CPT prediction and paragraph-level section prediction when combined
    
[^197]: MechaTerp-TRACE：一种用于语言模型组件消融分析的新方法

    MechaTerp-TRACE: A Novel Approach for Component Ablation Analysis in Language Models

    [https://arxiv.org/abs/2609.22163](https://arxiv.org/abs/2609.22163)

    提出MechaTerp-TRACE框架，通过逐一消融组件并测量固定答案词元处输出分布的变化，在统一尺度上比较语言模型中从transformer块到单个神经元等不同架构组件对命名实体生成的因果贡献。

    

    对大型语言模型的可解释性研究已经对前馈层中的事实回忆和自注意力中的词元关系做出了解释，但很少有工作提供一种统一的方法来比较不同架构组件对模型输出的因果贡献。我们介绍了MechaTerp（机制可解释性套件）-TRACE（教师强制消融组件效应注册子集），这是一种架构和研究方法，用于衡量语言模型中每个已注册组件对生成命名实体的支持程度。TRACE每次消融一个组件，并测量在固定答案词元处输出分布所产生的变化，因此从整个transformer块到单个神经元和输出logits的各种组件类型都可以在统一的尺度上进行比较。我们将其应用于十三个跨越五个家族、参数量从十亿到三百亿的指令微调密集解码器模型，对组件进行消融……

    arXiv:2609.22163v1 Announce Type: new  Abstract: Interpretability research on large language models has produced accounts of factual recall in feed-forward layers and of token relationships in self-attention, but little work offers a unified way to compare the causal contribution of different architecture components to a model's output. We introduce MechaTerp (the Mechanistic Interpretability suite) -TRACE (subset for Teacher-forced Registry of Ablated Component Effects), an architecture and study that measures how much each registered component of a language model supports the production of a named entity. TRACE ablates one component at a time and measures the resulting change in the output distribution at a fixed answer token, so component types from whole transformer blocks down to individual neurons and output logits can be compared on a common scale. We apply it to thirteen instruction-tuned dense decoder models spanning five families and one to thirty billion parameters, ablating
    
[^198]: 超越原始上下文传输：基于表示的联邦检索增强生成

    Beyond Raw Context Transfer: Representation-based Federated Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.22162](https://arxiv.org/abs/2609.22162)

    提出FedRepRAG，一种去中心化的联邦RAG框架，将原始文档保留在客户端本地，跨客户端检索时仅交换紧凑的潜在表示，从而避免直接共享原始内容带来的隐私暴露和推理时计算开销。

    

    检索增强生成（RAG）通过将生成过程建立在外部知识之上，提升了大语言模型（LLM）和视觉语言模型（VLM）的事实准确性。然而，大多数现有的RAG框架都假设存在一个集中式的检索语料库，这在医疗保健等敏感领域中往往不切实际，因为这些领域的数据天然是分布式的，且原始内容无法在机构之间直接共享。近期关于去中心化RAG的研究主要遵循基于提示词的范式，即交换原始的、人类可读的检索内容，这导致了大量推理时的计算开销，并直接暴露了检索到的信息。为了解决这些局限性，我们提出了基于表示的联邦RAG（FedRepRAG），这是一个去中心化的RAG框架，它将原始文档保留在拥有它们的客户端本地，在跨客户端检索时仅交换紧凑的潜在表示。为了整合检索到的知识，我们引入了……（摘要在此处截断）

    arXiv:2609.22162v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) improves the factuality of large language models (LLMs) and vision-language models (VLMs) by grounding generation in external knowledge. However, most existing RAG frameworks assume a centralized retrieval corpus, which is often impractical in sensitive domains such as healthcare, where data are inherently distributed and raw content cannot be directly shared across institutions. Recent efforts on decentralized RAG primarily follow prompt-based paradigms that exchange raw, human-readable retrieved content, leading to substantial inference-time computational overhead and direct exposure of retrieved information. To address these limitations, we propose Representation-based Federated RAG (FedRepRAG), a decentralized RAG framework that keeps raw documents at their owning clients and exchanges only compact latent representations during cross-client retrieval. To integrate retrieved knowledge, we introduce
    
[^199]: 教学知识还是临床病例？数据类型如何塑造医疗大语言模型

    Didactic knowledge or Clinical Cases? How Data Types Shape Medical Large Language Models

    [https://arxiv.org/abs/2609.22161](https://arxiv.org/abs/2609.22161)

    该研究通过token匹配实验揭示了医疗大语言模型训练数据的不对称迁移效应：临床数据能同时提升临床导向和知识密集型任务的表现，而教学数据主要提升知识密集型任务，且少量临床数据即可获得基于EHR任务的大部分收益。

    

    医疗大语言模型通常在教学数据（如教科书）和临床数据（如患者病历）的混合数据上进行训练，然而这些数据类型如何差异化地塑造模型能力仍不清楚。我们通过token数量匹配的实验来解决这一问题，改变教学数据与临床数据的比例，并分析数据组成如何影响知识密集型任务和临床导向任务的性能、能力画像和错误模式。我们发现了跨任务类型的不对称迁移现象：临床数据在提升临床导向任务的同时，在知识密集型任务上仍保持竞争力，而教学数据主要提升知识密集型任务。错误分析揭示了一种“知行差距”，即知识记忆能力的提升并不能可靠地泛化到临床推理中。我们进一步观察到，适量的临床数据即可在基于电子健康记录（EHR）的任务上带来大部分收益，而最优的数据混合比例……

    arXiv:2609.22161v1 Announce Type: cross  Abstract: Medical large language models are commonly trained on mixtures of didactic data (e.g., textbooks) and clinical data (e.g., patient records), yet how these data types differentially shape model capabilities remains unclear. We address this issue with token-matched experiments that vary the didactic-to-clinical ratio and analyze how data composition affects performance, capability profiles, and error patterns across knowledge-intensive and clinic-oriented tasks. We uncover an asymmetric transfer across task types: clinical data improves clinic-oriented tasks while remaining competitive on knowledge-intensive ones, whereas didactic data mainly improves knowledge-intensive tasks. Error analysis suggests a knowing-doing gap, where improvements in knowledge recall do not reliably generalize to clinical reasoning. We further observe that modest amounts of clinical data yield most of the gains on EHR-grounded tasks, while the optimal mixture r
    
[^200]: PAGE：分区感知的门控KV缓存淘汰

    PAGE: Partition-Aware Gated KV-Cache Eviction

    [https://arxiv.org/abs/2609.22157](https://arxiv.org/abs/2609.22157)

    PAGE提出一种无需训练、无需标签的门控机制，利用预填充注意力中top-k头一致性的早期到晚期下降来预测输入是否适合KV缓存淘汰，在淘汰会造成灾难性精度损失时自动保留完整缓存。

    

    KV缓存淘汰方法决定保留哪些token，却不决定是否应该淘汰，因此基准测试的平均值可能掩盖一类输入——在这些输入上，压缩会使准确率从99%骤降至0%。我们将淘汰重新构建为逐输入的准入决策，并发现输入可划分为两类：容量受限类（在任何预算下淘汰都是灾难性的）和稀释倾向类（淘汰是安全甚至有益的）。一个从预填充注意力中计算出的无标签标量——成对top-k头一致性的早期到晚期下降——可以在解码开始之前预测输入所属的类别。PAGE对该下降值进行阈值判断：当下降较大时应用任意基础淘汰器，否则保留完整缓存，整个过程无需训练、无需准确率标签。该下降值在四个架构家族中都能一致地按淘汰安全性对输入排序，且针对每个模型约100个输入的无标签试点即可为新架构家族重新校准阈值。作为保障机制，PAGE能够……

    arXiv:2609.22157v1 Announce Type: cross  Abstract: KV-cache eviction methods decide which tokens to keep but not whether to evict at all, so a benchmark mean can hide a class of inputs on which compression drives accuracy from 99\% to 0\%. We reframe eviction as a per-input admission decision and show that inputs separate into a capacity-bound class, where eviction is catastrophic at every budget, and a dilution-prone class, where eviction is safe or beneficial. A single label-free scalar computed from prefill attention, the early-to-late drop in pairwise top-$k$ head agreement, predicts this class before any decoding. PAGE thresholds this drop: it applies any base evictor when the drop is large and retains the full cache otherwise, with no training and no accuracy labels. The drop orders inputs by eviction safety consistently across four architecture families, and a per-model unlabeled pilot of about 100 inputs recalibrates the threshold for a new family. Used as a safeguard, PAGE cut
    
[^201]: 想象力源于幻觉吗？——大语言模型中想象力与幻觉的跨分类评估

    Is Imagination Derived from Hallucination? A Cross-Taxonomy Evaluation of Imagination and Hallucination in Large Language Models

    [https://arxiv.org/abs/2609.22152](https://arxiv.org/abs/2609.22152)

    该论文提出了首个大语言模型想象力评估基准Whiteboard，通过将七种基于机制的想象力子类型与幻觉分类体系进行交叉评估，首次实现了对“想象力与幻觉是否源自同一生成机制”这一论断的直接检验。

    

    想象力是大语言模型（LLM）的一项高级功能，决定了LLM创造前所未见或富有创意内容的潜力。尽管现有工作已为这一能力构建了丰富的创造力基准体系，但它们仅衡量输出偏离常见答案的程度，从不检验这种偏离是否为提示词所允许。此外，作为想象力最接近的“近邻”，幻觉总是在不同的生成结果上通过相互独立的流程进行测量，因此“想象力与幻觉源自同一生成机制”这一颇具影响力的论断从未得到直接检验。本文提出了Whiteboard——首个LLM想象力评估基准。其设计遵循为测量人类想象力而开发的权威认知工具：从经典范式中改编出七种基于机制的想象力子类型，再与十种支持边界幻觉（support-boundary hallucination）……进行交叉评估（原文摘要在此处截断）。

    arXiv:2609.22152v1 Announce Type: new  Abstract: Imagination performs as a high-level function of large language models (LLMs) which determines the potential of how an LLM creates unseen or creative content. While existing works have built a rich family of creativity benchmarks for this ability, they only measure how far an output departs from common answers and never check whether the departure is licensed by the prompt. Moreover, hallucination, the closest neighbor of imagination, is always measured in a separate pipeline on different generations, so the influential claim that imagination and hallucination stem from the same generative mechanism has never been directly testable. In this paper, we propose Whiteboard, the first LLM imagination evaluation benchmark. Its design follows the authoritative cognitive instruments developed to measure human imagination: seven mechanism-grounded imagination subtypes are adapted from classic paradigms, then crossed with ten support-boundary hall
    
[^202]: 语言模型知道自己的约束吗？

    Do Language Models Know Their Own Constraints?

    [https://arxiv.org/abs/2609.22151](https://arxiv.org/abs/2609.22151)

    研究发现语言模型经后训练（SFT与GRPO）学会遵守行为约束后，反而丧失了显式报告这些约束的能力，且基于奖励的GRPO训练对模型关于约束的显式知识与第三人称知识的破坏比SFT更严重。

    

    我们探究通过后训练获得的行为约束是否仍能被模型显式报告。以受约束的食谱生成为测试平台，通过对 Llama 3.1 8B Instruct 进行 LoRA 微调来强制执行五种禁用食材，我们在一个四级“约束意识基准”上将监督微调（SFT）与组相对策略优化（GRPO）同未训练基线进行比较。在三个随机种子上的平均结果表明，两种方法都将行为合规率从 4% 提升至约 90%，同时使显式约束报告能力降至未训练模型之下（SFT 从 0.48/5 降至 0.16/5，GRPO 降至 0.07/5），并侵蚀了模型保留的第三人称知识（SFT 从 93% 降至 36%，GRPO 降至 14%；两种方法之间 p 小于 0.01）。与我们的初始假设相反，基于奖励的信号是两者中破坏性更强的：一种无论表述方式如何都惩罚禁用食材 token 的奖励，学到的是与上下文无关的抑制，而非自我导向的约束

    arXiv:2609.22151v1 Announce Type: new  Abstract: We ask whether behavioral constraints acquired through post training remain explicitly reportable. Using constrained recipe generation as a testbed, five banned ingredients enforced via LoRA fine tuning of Llama 3.1 8B Instruct we compare supervised fine tuning (SFT) and Group Relative Policy Optimization (GRPO) against an untrained baseline on a four tier Constraint Awareness Benchmark. Averaged over three seeds, both methods raise behavioral compliance from 4% to about 90% while reducing explicit constraint reporting below the untrained model (0.48/5 to 0.16/5 for SFT, 0.07/5 for GRPO) and eroding retained third person knowledge (93% to 36% for SFT, 14% for GRPO; p less than 0.01 between methods). Contrary to our initial hypothesis, the reward based signal is the more destructive of the two: a reward that penalizes banned ingredient tokens regardless of framing learns a context independent suppression rather than a self directed constr
    
[^203]: 超越拼接假设：基于语义量化的多模态合成数据评估统一框架

    Beyond the Stitching Assumption: A Unified Framework for Multimodal Synthetic Data Evaluation via Semantic Quantization

    [https://arxiv.org/abs/2609.22149](https://arxiv.org/abs/2609.22149)

    该论文提出一种基于语义量化与列联表散度比较的统一评估框架，能够捕捉传统单一模态指标无法检测到的表格与文本配对被破坏的问题，为多模态合成数据质量评估超越了“拼接假设”。

    

    多模态合成数据集将结构化属性与自由文本相结合，但两者通常被分开评估。这种评估方式下，即使表格—文本之间的配对关系被打乱，各项指标仍可能保持较高水平。我们提出了一种针对表格—文本合成数据的基于投影的评估器：使用固定的句子编码器将文本映射为嵌入向量，再通过 k-means 将其转换为聚类状态；表格变量则表示为类别状态或分位数分箱状态。随后，利用 Jensen–Shannon 散度（JSD）、归一化互信息（NMI）、条件 JSD（cJSD）以及联合状态熵，对真实数据与合成数据的列联表进行比较。我们还报告了文本到属性（T2A）效用，以及经留出集校准的邻近标记率（PFR），作为表示层面的诊断指标。我们设计了文本置换对照实验，在破坏配对关系的同时保持两个边缘分布不变。在 Amazon Reviews、Kiva Loans 和 Employment Scam Aegean 数据集上的实验表明，模态特定的指标……（原文摘要在此处被截断）

    arXiv:2609.22149v1 Announce Type: new  Abstract: Multimodal synthetic datasets combine structured attributes with free text, but are often evaluated separately. Such metrics can remain high after tabular--text pairings are disrupted. We present a projection-based evaluator for tabular--text synthetic data. A fixed sentence encoder maps text to embeddings, \(k\)-means converts them to cluster states, and tabular variables are represented as categorical or quantile-binned states. Real and synthetic contingency tables are compared using Jensen--Shannon divergence (JSD), normalized mutual information (NMI), conditional JSD (cJSD), and joint-state entropy. We also report text-to-attribute (T2A) utility and a holdout-calibrated proximity flag rate (PFR) as a representation-level diagnostic. A text-permutation control preserves both marginal distributions while disrupting their pairing. Experiments on Amazon Reviews, Kiva Loans, and the Employment Scam Aegean Dataset show that modality-specif
    
[^204]: GRRR：解码器大语言模型后训练中的重塑、旋转与路由几何学

    GRRR: The Geometry of Reshaping, Rotation, and Routing in Decoder LLM post-training

    [https://arxiv.org/abs/2609.22146](https://arxiv.org/abs/2609.22146)

    后训练带来的收益主要源于在预训练权重的SVD坐标系中重新配置和扩展已有通路（旋转与零空间路由），而非大幅改变奇异值本身。

    

    我们研究了后训练如何改变大语言模型（LLM）的权重相对于其预训练权重的变化。在12条包含监督微调（SFT）和强化学习（RL）的后训练链路中，我们将每次权重更新表示在预训练矩阵的奇异值分解（SVD）坐标系中。这一分解将变化分离为三个几何上截然不同的组成部分：对角线值，它重塑奇异值；非对角线值，它旋转预训练输入与输出方向之间的耦合；以及零空间值，它在矩阵原始非零SVD核心之外进行路由。在一个数学评测套件上，我们发现移除对角线分量通常能保留后训练带来的大部分收益。这些结果表明，后训练的收益主要由重新配置和扩展预训练通路来承载，而非通过大幅改变预训练模型的奇异值来实现。

    arXiv:2609.22146v1 Announce Type: cross  Abstract: We study how post-training changes the weights of Large Language Models (LLMs) relative to their pretrained weights. Across 12 post-training chains with supervised fine-tuning (SFT) and reinforcement learning (RL), we express each weight update in the pretrained matrix's singular value decomposition (SVD) frame. This decomposition separates the changes of three geometrically distinct components: diagonal values, which reshapes singular values; off-diagonal values, which rotates the coupling between pretrained input and output directions; and null-space values, which routes outside the matrix's original nonzero SVD core. On a math evaluation suite, we find that removing the diagonal component usually preserves most of the gains from post-training. These results suggest that post-training gains are carried primarily by reconfiguring and extending pretrained pathways rather than by substantially changing singular values of pre-trained mod
    
[^205]: 弱关联，强信号：基于独立Token采样的扩散大语言模型高效训练数据检测

    Weak Ties, Strong Signals: Efficient Training Data Detection in Diffusion LLMs via Independent Token Sampling

    [https://arxiv.org/abs/2609.22145](https://arxiv.org/abs/2609.22145)

    提出独立Token采样方法，通过构建内部依赖较弱的掩码token集合来消除结构性估计误差，从而实现对扩散大语言模型训练数据使用情况的高效检测。

    

    扩散大语言模型为自回归模型提供了一种颇具吸引力的替代方案，但它们可能在去噪过程中暴露敏感的训练数据。检测这种数据使用情况具有挑战性，因为dLLMs缺乏因果架构所具备的高效单次概率分解能力。现有方法依赖随机掩码，在有限查询预算下获得可处理的逐token检测信号，但无法控制被掩码token之间的依赖关系。我们证明这种逐token近似会引入非负的结构性估计误差，该误差在理论上由被掩码token之间的累积条件互信息（CMI）刻画，并可能掩盖细微的记忆化信号。这一洞察表明，可靠的检测需要内部依赖较弱的掩码token集合。为避免直接在token组合上估计CMI的过高成本，我们提出独立Token采样方法

    arXiv:2609.22145v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) offer a compelling alternative to autoregressive models, yet they may expose sensitive training data during denoising. Detecting such usage is challenging because dLLMs lack the efficient one-pass probability decomposition of causal architectures. Existing methods rely on random masking to obtain tractable token-wise detection signals under limited query budgets, but fail to control dependencies among masked tokens. We demonstrate that this token-wise approximation introduces a non-negative structural estimation error, which is theoretically characterized by the cumulative conditional mutual information (CMI) among masked tokens and can obscure subtle memorization signals. This insight suggests that reliable detection requires masked token sets with weak internal dependency. To avoid the prohibitive cost of directly estimating CMI over token combinations, we propose \textit{Independent Token Samp
    
[^206]: 多语言安全信号是多层次的：过滤降低安全性的数据以构建更安全的大语言模型

    Multilingual Safety Signals Are Multi-Layered: Filtering Safety-Degrading Data for Safer LLMs

    [https://arxiv.org/abs/2609.22144](https://arxiv.org/abs/2609.22144)

    提出多层框架MMSAFE，通过捕获跨语言共享及语言特定的多层安全信号，识别多语言微调数据中降低安全性的样本，从而保护大语言模型的安全对齐。

    

    在大语言模型微调过程中保持安全对齐至关重要，然而近期研究表明，即使是良性的微调数据也可能包含会悄然破坏安全对齐的降低安全性样本。现有方法通常使用单一安全敏感层的表示来识别此类样本。虽然这一假设在单语言环境中已被证明有效，但由于跨语言表示模式可能存在差异，其对多语言模型的有效性尚不明确。通过跨语言分析，我们发现敏感层在各语言之间仅部分共享，安全相关信号往往分布在多个层中。基于这些观察，我们提出了MMSAFE，一个用于多语言安全性降低数据识别的多层框架，能够同时捕获共享的和特定语言的安全信号。在多个……上的大量实验……

    arXiv:2609.22144v1 Announce Type: new  Abstract: Preserving safety alignment during large language models fine-tuning is critical, however, recent studies have demonstrated that even benign fine-tuning data may contain safety-degrading samples that silently undermine safety alignment. Existing approaches typically identify such samples using representations from a single safety-sensitive layer. While this assumption has shown effectiveness in monolingual settings, its validity for multilingual models remains unclear due to potential cross-lingual differences in representation patterns. Through a cross-lingual analysis, we show that sensitive layers are only partially shared across languages, with safety-relevant signals often distributed across multiple layers. Motivated by these observations, we propose MMSAFE, a multi-layer framework for multilingual safety-degrading data identification that captures both shared and language-specific safety signals. Extensive experiments across multi
    
[^207]: 使用复合算子将大语言模型语义变换线性化

    Using Composition Operators to Linearize LLM Semantic Transformations

    [https://arxiv.org/abs/2609.22143](https://arxiv.org/abs/2609.22143)

    该论文提出用推广自Koopman算子的复合算子将大语言模型的语义变换表示为矩形无限维线性算子，证明其为等距算子，并通过奇异值谱来检测表示失准以及比较不同任务和模型。

    

    机器学习学习的是函数：从提示到响应，从图像到描述。然而这些函数在数学上究竟是什么仍难以言明。我们提出了一种方法，借助属于库普曼理论范畴的动力系统技术来近似这类变换。我们引入了复合算子，它推广了Koopman算子，并且关键在于能够在不同空间之间进行映射，由此启发了一种观点：大语言模型的变换是矩形无限维算子。这一形式化揭示了有用的结构：在提示与响应分布的自然假设下，LLM算子是一个等距算子，而学习到的表示之间的失准则表现为其有限截断的谱污染。随后，我们概述了构建LLM算子有限维近似的方法，并演示了如何利用奇异值谱来比较不同的任务和模型。

    arXiv:2609.22143v1 Announce Type: new  Abstract: Machine learning learns functions: prompt to response, image to caption. What these functions are mathematically remains hard to say. We present a method to approximate these kinds of transformations using techniques from dynamical systems that fall under the umbrella of Koopmanism. We introduce the use of composition operators, which generalize the Koopman operator and, crucially, can map between distinct spaces, motivating the perspective that LLM transformations are rectangular infinite-dimensional operators. This formalism reveals useful structure: under natural assumptions on the prompt and response distributions, the LLM operator is an isometry, and misalignment between learned representations manifests as spectral pollution of its finite sections. We then outline a method of constructing finite-dimensional approximations of an LLM operator, and demonstrate how the singular value spectrum can be used to compare tasks and models.
    
[^208]: 真实性信号能否在语码转换中存续？面向印地语-英语混合文本（Hinglish）幻觉检测的隐藏状态探测

    Does the Truthfulness Signal Survive Code-Mixing? Probing Hidden States for Hallucination Detection in Hinglish

    [https://arxiv.org/abs/2609.22138](https://arxiv.org/abs/2609.22138)

    该论文首次研究幻觉探测信号在印地语-英语语码混合（Hinglish）下的表现，构建了包含5,674个条目的三语问答基准，并评估基于纯净语言隐藏状态训练的线性和MLP探测器在三个开源大语言模型上的跨语言迁移能力。

    

    隐藏状态幻觉探测——在大语言模型的内部激活上训练线性分类器，以检测生成的答案是否忠实于输入——是2026年的一个活跃研究领域，近期工作在多个基准和语言上报告了0.90-1.00的AUROC。然而，这些工作都没有在语码混合输入上测试探测器，尽管大量聊天机器人用户使用印地语-英语混合文本（"Hinglish"）进行写作。我们直接填补这一空白：在纯净语言隐藏状态上训练的幻觉探测器能否迁移到Hinglish，还是信号会在语码混合下退化？我们构建了一个包含5,674个条目的印地语/英语/Hinglish问答基准，在三个开放权重的7-8B大语言模型（Qwen2.5-7B、Mistral-7B、Llama-3.1-8B）上生成并标注了17,022条模型响应，在两个token位置提取逐层隐藏状态，并训练线性和MLP探测器用于分布内检测和跨语言迁移。

    arXiv:2609.22138v1 Announce Type: new  Abstract: Hidden-state hallucination probing - training a linear classifier on an LLM's internal activations to detect whether a generated answer is faithful to the input - is an active area of 2026 research, with recent work reporting 0.90-1.00 AUROC across several benchmarks and languages. However, none of this work has tested probes on code-mixed input, despite the fact that a huge population of chatbot users write in Hindi-English code-mixed text ("Hinglish"). We address this gap directly: does a hallucination probe trained on clean-language hidden states transfer to Hinglish, or does the signal degrade under code-mixing? We construct a 5,674-item Hindi/English/Hinglish QA benchmark, generate and label 17,022 model responses across three open-weight 7-8B LLMs (Qwen2.5-7B, Mistral-7B, Llama-3.1-8B), extract per-layer hidden states at two token positions, and train linear and MLP probes for in-distribution detection and cross-lingual transfer. W
    
[^209]: DiFA：面向词元级文本异常检测的双证据融合与聚合方法

    DiFA: Dual Evidence Fusion and Aggregation for Token-Level Text Anomaly Detection

    [https://arxiv.org/abs/2609.22136](https://arxiv.org/abs/2609.22136)

    提出DiFA双证据框架，融合形式-结构与语义两种视角的异常线索并进行自适应聚合，实现词元级文本异常检测，可精确定位文档中的异常词或片段。

    

    文本异常检测，即识别偏离正常语言模式的文本实例的任务，对于语言驱动的应用至关重要。然而，大多数现有方法只能进行文档级的异常检测，难以定位有害短语或支持有针对性的预防。近来，词元级文本异常检测成为一种新兴趋势，旨在通过识别文档中的异常词或片段来克服上述局限。然而，一种代表性方法主要依赖表示空间中的距离度量，忽视了不同异常线索在捕捉多样化异常模式时的互补作用。为弥补这些不足，我们提出了一种具有自适应融合与聚合的双证据框架，用于词元级异常检测。DiFA从形式-结构和语义两种视角推导异常分数，以捕捉可见的结构异常和……（原文摘要在此处截断）

    arXiv:2609.22136v1 Announce Type: new  Abstract: Text anomaly detection, the task of identifying text instances that deviate from normal language patterns, is crucial for language-driven applications. However, most existing methods can only perform document-level anomaly detection, making it hard to locate harmful phrases or support targeted prevention. Recently, there has been an emerging trend toward token-level text anomaly detection, which aims to address the above limitation by identifying anomalous words or fragments within a document. Nevertheless, one representative method mainly relies on representation-space distance measurement, neglecting the complementary roles of different anomaly cues in capturing diverse abnormal patterns. To bridge the gaps, we propose a Dual-evidence framework with adaptive Fusion and Aggregation (DiFA) for token-level anomaly detection. DiFA derives anomaly scores from form-structural and semantic views to capture visible structural abnormality and c
    
[^210]: 最佳读取层并非最佳引导层：全模态大语言模型中的探测-引导层解离现象

    Read-Best Is Not Steer-Best: A Probing--Steering Layer Dissociation in Omni-Modal Large Language Models

    [https://arxiv.org/abs/2609.22135](https://arxiv.org/abs/2609.22135)

    该论文首次通过因果实验发现，在全模态大语言模型中，探测准确率最高的层并非最适合激活引导的层，读取与干预依赖不同的层，且引导有效的层稳定集中于模型归一化深度的中后段。

    

    全模态大语言模型将文本、音频和图像信号整合到一个共享的残差流中，其中诸如情感等概念可以被线性解码，并通过激活引导进行因果修改。一个常见但很少被检验的假设是：探测准确率最高的层也是最适合引导的层，因此注入层通常根据探测性能来选择。我们在三个独立开发的全模态模型上对这一假设进行了首次因果测试，发现该假设并不成立。读取与干预依赖于不同的层，我们将这一现象称为探测-引导层解离。以情感作为受控测试平台，我们测量了文本、音频和图像输入下各层的可读性与可引导性。探测最佳层在不同架构之间差异很大，而引导有效的层则始终稳定地落在归一化深度的一个狭窄的中后段范围内。配对的随机方向对照实验（摘要在此处截断）

    arXiv:2609.22135v1 Announce Type: new  Abstract: Omni-modal large language models integrate text, audio, and image signals into a shared residual stream, where concepts such as emotion can be linearly decoded and causally modified by activation steering. A common but rarely tested assumption is that the layer with the highest probing accuracy is also the best layer for steering, so injection layers are often selected by probe performance. We provide the first causal test of this assumption across three independently developed omni-modal models and find that it fails. Reading and intervention rely on different layers, a phenomenon we call the probing-steering layer dissociation. Using emotion as a controlled testbed, we measure layer-wise readability and steerability across text, audio, and image inputs. Probe-best layers vary widely across architectures, while steering-effective layers consistently fall within a narrow mid-to-late range of normalized depth. Paired random-direction cont
    
[^211]: 大语言模型与人类标注的观测等价性

    Observational Equivalence of LLM and Human Annotation

    [https://arxiv.org/abs/2609.22133](https://arxiv.org/abs/2609.22133)

    该论文通过对14项政治学研究的文本分类复制实验证明，大语言模型与人类专家在标注质量上具有观测等价性——LLM与专家的一致率与专家之间彼此的一致率相当，且分歧源于文本和编码规则的模糊性，因此仅凭标注质量没有理由优先选择人工编码，而LLM在速度和成本上优势显著。

    

    在本文中，我们证明了大语言模型（LLM）与人工编码在标注质量方面具有观测等价性：近期的大语言模型与专家编码者的一致率与专家之间彼此的一致率相当。我们通过复制14项经同行评审的政治学研究中的文本分类任务来证明这一点，在这些任务中，十个大语言模型、三名人类专家和165名众包工人使用相同的编码手册独立地对相同的文本进行分类。我们发现，这种等价性源于文本和编码规则中固有的模糊性。当大语言模型与专家意见不一致时，专家之间也更容易产生分歧，而澄清编码规则可以同时减少专家和足够强大的大语言模型之间的分歧。因此，仅从标注质量的角度来看，几乎没有实证依据支持优先选择人工编码，而大语言模型在速度和成本方面具有显著优势。因此，我们认为核心挑战……

    arXiv:2609.22133v1 Announce Type: new  Abstract: In this paper, we show that LLM and human coding are observationally equivalent in terms of annotation quality: recent LLMs agree with expert coders at rates comparable to those observed among experts themselves. We demonstrate this through replications of text-classification tasks from 14 peer-reviewed political science studies, in which ten LLMs, three human experts, and 165 crowdsourced workers independently classify the same texts using identical codebooks. We find that this equivalence is driven by ambiguity in the texts and coding rules. When LLMs disagree with experts, experts are also more likely to disagree with one another, and clarifying coding rules reduces disagreement among both experts and sufficiently capable LLMs. Thus, there is little empirical basis for preferring human coding on the basis of annotation quality alone, while LLMs offer substantial advantages in speed and cost. We therefore argue that the central challen
    
[^212]: 面向大语言模型的相关性感知结构化剪枝

    Correlation-Aware Structured Pruning for Large Language Models

    [https://arxiv.org/abs/2609.22131](https://arxiv.org/abs/2609.22131)

    该论文提出了一种相关性感知的结构化剪枝方法，将剪枝建模为显式考虑跨单元依赖关系的二元二次规划问题，并设计基于依赖感知边际成本的贪心交互算法进行求解，从而突破了传统剪枝方法的独立性假设，在降低大语言模型推理成本的同时避免性能下降。

    

    结构化剪枝是一种很有前景的方法，能够在保持硬件效率的同时降低大语言模型（LLM）巨大的推理成本。许多现有方法在评估可剪枝单元（如通道或注意力头）的重要性时是孤立进行的，隐含地假设剪枝误差是可加的。这种独立性假设常常会因模型权重的非正交性以及单元激活之间的强相关性而失效，从而可能导致性能下降。为了解决这一问题，我们提出了一种相关性感知的结构化剪枝方法。我们将剪枝目标表述为一个基数约束的二元二次规划问题，该问题显式地建模了重建误差中的跨单元依赖关系。由于该二元二次规划问题是NP难的且难以精确求解，我们开发了一种基于依赖感知边际成本的贪心交互算法来优化单元选择。此外，我们……

    arXiv:2609.22131v1 Announce Type: new  Abstract: Structured pruning is a promising approach for reducing the substantial inference costs of Large Language Models (LLMs) while maintaining hardware efficiency. Many existing methods assess the importance of prunable units (e.g., channels or heads) in isolation, implicitly assuming that pruning errors are additive. This independence assumption is often invalidated by the non-orthogonality of model weights and strong correlations between unit activations, potentially leading to performance degradation. To address this, we propose a Correlation-Aware Structured Pruning method. We formulate the pruning objective as a cardinality-constrained binary quadratic program that explicitly models cross-unit dependencies in the reconstruction error. Since this binary quadratic program is NP-hard and difficult to solve exactly, we develop a greedy interaction algorithm based on dependency-aware marginal costs to optimize unit selection. Furthermore, we 
    
[^213]: 超越准确性与表面流畅性：面向法律条款生成的大语言模型风险敏感评估

    Beyond Accuracy and Surface Fluency: Risk-Sensitive Evaluation of LLMs for Legal Clause Generation

    [https://arxiv.org/abs/2609.22127](https://arxiv.org/abs/2609.22127)

    本文提出了一个风险敏感的法律条款生成评估框架，结合CLAUSE和LENS-CRAFT两个体系，对四个大语言模型在22个合同条款类别和34种法律失败模式上进行评估，并采用最大严重性原则而非平均分数来捕捉法律起草中的关键风险。

    

    大语言模型（LLM）正被越来越多地用于起草合同语言，然而传统的基于准确性或偏好的评估方法与法律起草的实际需求并不匹配。一个条款可能流畅且文体精美，但仍然可能遗漏关键的除外条款、以不可执行的方式分配风险、默认了不适用的司法管辖区，或使一方承担监管责任。本文提出了一项用于评估大语言模型生成合同条款的实证研究设计与框架。该研究评估了四个模型——Claude Haiku 4.5、Gemini 2.5 Flash Lite、GPT 5.4 Nano 和 Qwen 3.5 Flash，涵盖22个合同条款类别和34种基于法律动机的失败模式。研究结合了两个评估框架：CLAUSE（按法律功能和失败目标对提示进行分类）和 LENS-CRAFT（在九个法律质量维度上对输出进行评分）。该研究没有对维度分数取平均值，而是采用最大严重性原则（Max Severity Principle）……

    arXiv:2609.22127v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to draft contractual language, yet conventional accuracy or preference-based evaluations are poorly matched to legal drafting. A clause may be fluent and stylistically polished while still omitting an essential carve-out, allocating risk in an unenforceable way, assuming an inapplicable jurisdiction, or exposing a party to regulatory liability. This paper presents a empirical study design and framework for evaluating LLM-generated contract clauses. The study evaluates four models - Claude Haiku 4.5, Gemini 2.5 Flash Lite, GPT 5.4 Nano, and Qwen 3.5 Flash, across 22 contract clause categories and 34 legally-motivated failure modes. We combine two evaluation frameworks: CLAUSE, which classifies prompts by legal function and failure target, and LENS-CRAFT, which scores outputs across nine legal-quality dimensions. Instead of averaging dimension scores, the study applies a Max Severity Princ
    
[^214]: 类型驱动的婆罗米文字分词方法

    Type-Driven Tokenization for Brahmic Scripts

    [https://arxiv.org/abs/2609.22125](https://arxiv.org/abs/2609.22125)

    该论文发现婆罗米文字（如天城文、泰米尔文等）的正字法构成部分半群而非英语那样的半群，并在 Agda 中形式化推导出可证明正确的 fixToken 函数，用以修复大语言模型分词器对此类文字产生的畸形文本。

    

    大型语言模型中使用的标准分词器在应用于婆罗米文字时会产生格式错误的文本。婆罗米文字是一类元音附标文字，其书写系统中辅音携带固有元音，而依赖性附加符号可以修改该元音，包括天城文、泰卢固文、泰米尔文、卡纳达文等。其根本问题在于，这些分词器违反了在英语等字母文字中不会出现的正字法约束。我们观察到，英语正字法构成一个半群（任意两个有效词元都可以自由拼接），而婆罗米文字的正字法构成一个部分半群：并非每次拼接都能产生有效字符串。我们在 Agda 中形式化了这一区别，将有效的婆罗米文字词元建模为转移系统中的链，并推导出一个可证明正确的 fixToken 函数，该函数可将任何候选词元扩展为遵守正字法边界的词元。随后我们展示了这一形式化推导如何转化为针对 SentencePiece 的实用补丁。

    arXiv:2609.22125v1 Announce Type: new  Abstract: Standard tokenizers used in large language models produce malformed text when applied to Brahmic scripts. They are a family of abugidas, writing systems whose consonants carry an inherent vowel that dependent marks can modify. They include Devanagari, Telugu, Tamil, Kannada, and others. The underlying issue is that these tokenizers violate orthographic constraints that do not arise in alphabetic scripts like English. We observe that while English orthography forms a \emph{semigroup} (any two valid tokens can be freely concatenated), Brahmic orthography forms a \emph{partial semigroup}: not every concatenation yields a valid string. We formalise this distinction in Agda, model valid Brahmic tokens as chains in a transition system, and derive a provably correct \texttt{fixToken} function that extends any candidate token to respect orthographic boundaries. We then show how this formal derivation translates into a practical patch for Sentenc
    
[^215]: 在面向乌克兰多领域文档理解的RAG流水线中平衡推理能力与硬件约束

    Balancing Reasoning and Hardware Constraints in RAG Pipelines for Ukrainian Multi-Domain Document Understanding

    [https://arxiv.org/abs/2609.22124](https://arxiv.org/abs/2609.22124)

    针对UNLP 2026共享任务中OCR严重挤占9小时离线时间预算的问题，该论文提出结合BM25、BGE-M3与交叉编码器重排序的资源高效混合RAG流水线，并放弃参数量庞大的推理模型，以确保乌克兰多领域文档问答系统在严格时限内完成。

    

    本文介绍了提交给UNLP 2026多领域文档理解共享任务的系统。该挑战要求在严格的9小时离线Kaggle执行时间限制内，从多样化的乌克兰语PDF文档语料库中提取精确答案、文档ID和页码。在隐藏的私有测试集评估过程中，扫描文档的光学字符识别（OCR）成为严重瓶颈，由于顺序单线程执行，消耗了总时间预算中的5至7小时。这一开销严格限制了剩余可用于大语言模型（LLM）推理的时间，即500个问题仅有约两小时。为了保证流水线在不超时的情况下完成，我们开发了一个资源高效的混合检索增强生成（RAG）流水线，利用BM25、BGE-M3和交叉编码器重排序。我们没有部署参数量庞大的推理模型（例如DeepSeek R1）——这类模型始终会超时——而是……

    arXiv:2609.22124v1 Announce Type: new  Abstract: This paper describes the system submitted to the UNLP 2026 Shared Task on Multi-Domain Document Understanding. The challenge required extracting precise answers, document IDs, and page numbers from a diverse corpus of Ukrainian PDF documents within a strict 9-hour offline Kaggle execution limit. During evaluation on the hidden private test set, optical character recognition (OCR) of scanned documents emerged as a severe bottleneck, consuming 5-7 hours of the total time budget due to sequential single-threaded execution. This overhead strictly limited the remaining time for Large Language Model (LLM) inference to approximately two hours for 500 questions. To guarantee pipeline completion without timeouts, we developed a resource-efficient Hybrid Retrieval-Augmented Generation (RAG) pipeline utilizing BM25, BGE-M3, and Cross-Encoder reranking. Rather than deploying parameter-heavy reasoning models (e.g., DeepSeek R1) which consistently tim
    
[^216]: 成功会留下弯路：为长程智能体学习可执行的通关攻略

    Success Leaves Detours: Learning Executable Walkthroughs for Long-Horizon Agents

    [https://arxiv.org/abs/2609.22120](https://arxiv.org/abs/2609.22120)

    提出 Trace 框架，通过信用引导与依赖锚定机制，将含噪的稀疏奖励轨迹编译为可执行、可验证的攻略记忆，从而提升长程智能体的表现。

    

    测试时自进化智能体通过复用过往经验来提升性能，然而稀疏奖励轨迹中往往包含失败、循环和弯路，而摘要式的总结常常遗漏执行所需的状态条件和动作依赖关系。我们研究了从稀疏奖励轨迹中进行可执行攻略归纳的任务：即提取紧凑的、以状态为条件的、可验证的操作流程。我们的关键观察是：延迟信用分配能够识别与进展相关的动作，但无法确定这些动作是否产生了后续动作所需的事实。我们提出了 Trace，一个信用引导、依赖锚定的框架，能够将嘈杂的轨迹编译为可执行的攻略记忆。该框架从奖励和持久状态变化中检测进展锚点，通过信用传播识别有价值的转移，并从跨回合的成功与失败证据中估计动作的前提条件。随后，反向依赖切片将所需事实追溯到其产生者……

    arXiv:2609.22120v1 Announce Type: cross  Abstract: Test-time self-evolving agents improve by reusing past experience, yet sparse-reward trajectories contain failures, loops, and detours, while summaries often omit the state conditions and action dependencies needed for execution. We study executable Walkthrough induction from sparse-reward trajectories: extracting compact, state-conditioned, and verifiable procedures. Our key observation is that delayed credit identifies actions associated with progress but cannot determine whether they produce facts required by later actions. We propose Trace, a credit-guided, dependency-grounded framework that compiles noisy trajectories into executable Walkthrough Memory. It detects progress anchors from rewards and persistent state changes, propagates credit to identify valuable transitions, and estimates action prerequisites from cross-episode success and failure evidence. Backward dependency slicing then traces required facts to their producers, 
    
[^217]: 评估意识随模型规模从格式转向上下文

    Evaluation Awareness Shifts from Format to Context with Model Scale

    [https://arxiv.org/abs/2609.22119](https://arxiv.org/abs/2609.22119)

    该研究揭示了模型检测评估的机制随规模演变——小模型依赖提示词格式而大模型依赖高阶推理，并提出结合提示词净化与激活反向引导的双通路干预方法，在高度评估意识的提示词上实现了平均70.58%的行为翻转率。

    

    评估意识对模型评估构成了前所未有的威胁，但模型检测评估的机制仍然未知。本研究致力于确定这一机制，并识别较小模型与较大模型之间截然不同的检测机制。较小的模型利用提示词的格式敏感性来检测评估，而较大的模型通常依赖高阶推理来检测评估。我们使用思维链分析、表示探测和积分梯度归因方法评估了 Gemma 3（1B、4B 和 12B）、Phi-3（Mini 和 Medium）以及 Llama-3 8B。基于这些发现，我们提出了一种双通路干预方法，将提示词净化与激活反向引导相结合，以同时抑制外部评估触发因素及其内部表示。在 200 个高度评估意识的提示词上，我们的方法实现了平均 70.58% 的行为翻转率，始终优于单独使用任一干预方法。

    arXiv:2609.22119v1 Announce Type: new  Abstract: Evaluation awareness poses an unprecedented threat to model evaluation, but the mechanisms by which models detect it remain unknown. This study focuses on determining this and identifying contrasting mechanisms between smaller and larger models. While smaller models use the prompt's format sensitivity to detect evaluation, larger models often rely on higher-order reasoning to detect it. We evaluated Gemma 3 (1B, 4B, and 12B), Phi-3 (Mini and Medium), and Llama-3 8B using Chain-of-Thought analysis, representation probing, and Integrated Gradients attribution. Motivated by these findings, we propose a dual-pathway intervention that combines prompt sanitization with activation counter-steering to suppress both external evaluation triggers and their internal representations. Across 200 highly evaluation-aware prompts, our method achieves an average behavioral flip rate of 70.58\%, consistently outperforming either intervention alone. These r
    
[^218]: 多轮编程智能体中上下文压缩网关的实证成本归因分析

    An Empirical Cost Attribution of Context-Compression Gateways in Multi-Turn Coding Agents

    [https://arxiv.org/abs/2609.22114](https://arxiv.org/abs/2609.22114)

    该论文通过插桩生产级压缩网关，将多轮编程智能体的token成本归因分解为三个独立杠杆，实证发现工具模式过滤是唯一明确且可复现的省钱手段，而内容压缩与历史摘要的实际节省远低于普遍假设。

    

    上下文压缩被广泛提出作为降低大语言模型编程智能体token开销的方法，且公开基准测试报告称激进的压缩能够保持任务求解质量。但这两个事实并不能推出人们通常默认的第三个结论：即在真实的多轮智能体中压缩文件读取能够省钱。我们对位于编程智能体（Claude Code、Codex）与前沿大模型（Claude Sonnet、GPT-5）之间的生产级压缩网关（Paritok）进行了插桩测量，并将真实会话的token账单分解为三个相互独立的杠杆：工具模式过滤、针对文件读取与工具输出的内容压缩、以及历史对话摘要。在受控A/B实验中分别单独测量后发现，这三者的节省速率存在根本差异。工具模式过滤每轮移除一个固定块，典型一轮约为21K-57K tokens；其节省量与轮数N呈线性关系，是唯一明确且可复现地产生正向节省的杠杆。内容压缩仅节省……（摘要原文在此处截断）

    arXiv:2609.22114v1 Announce Type: new  Abstract: Context compression is widely proposed as a way to cut the token bill of LLM coding agents, and public benchmarks report that aggressive compression preserves task-solving quality. These two facts do not imply the third one commonly assumed: that compressing file reads saves money in a real multi-turn agent. We instrument a production compression gateway (Paritok) between coding agents (Claude Code, Codex) and frontier LLMs (Claude Sonnet, GPT-5), and decompose the token bill of real sessions into three independent levers: tool-schema filtering, content compression of file reads and tool output, and history summarization. Measured in isolation under controlled A/B runs, the three save at fundamentally different rates. Tool-schema filtering removes a fixed block every turn, roughly 21K-57K tokens on a typical turn; it is linear in the turn count N and the only unambiguously and reproducibly positive lever. Content compression saves only a
    
[^219]: 大语言模型中的隐私与个性化权衡：文体信号削减对用户特定文本生成的影响

    Privacy Personalization Trade offs in LLMs: The Impact of Stylometric Signal Reduction on User-Specific Text Generation

    [https://arxiv.org/abs/2609.22112](https://arxiv.org/abs/2609.22112)

    该论文提出一个基于LaMP-7基准的受控实验框架，通过对比原始档案与匿名化档案条件下的文本生成，揭示了削减文体信号（如人口统计标识、文化引用和个人细节）与LLM个性化生成能力之间的隐私-个性化权衡关系。

    

    大语言模型（LLMs）已展现出以高度风格保真度生成用户特定文本的能力。然而，实现这种个性化所依赖的个人数据往往嵌入人口统计、文化和风格标记，引发了文体特征再识别方面的担忧。本文研究了减少可识别的风格信号是否会影响LLM文本生成中的个性化效果。我们引入了一个受控框架，利用LaMP-7 Twitter基准来分离LLM个性化中的文体信号。针对250个抽样用户的实验比较了两种设置：基于原始用户档案的条件改写，以及基于匿名化转换档案的条件改写——在后一种档案中，人口统计标识符、文化引用、个人细节和非正式语言线索已被系统地中和。生成结果由两个独立的LLM评估器以及补充性的人工评估进行评判。

    arXiv:2609.22112v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated the ability to generate user-specific text with high stylistic fidelity. However, the personal data that enables such personalization frequently embeds demographic, cultural, and stylistic markers that raises concerns about stylometric re- identification. This paper investigates whether reducing identifiable stylistic signals affects personalization in text generation by LLMs. We introduce a controlled framework to isolate stylometric signals in LLM personalization using the LaMP-7 Twitter benchmark. Experiments on 250 sampled users compare two settings: paraphrasing conditioned on the original profile and paraphrasing conditioned on an anonymized converted profile in which demographic identifiers, cultural references, personal details, and informal linguistic cues have been systematically neutralized. Outputs are assessed by two independent LLM judges and a complementary human evaluation. O
    
[^220]: 超越文本：验证智能体撰写的论文是否由其产物支撑

    Beyond the Text: Verifying That Agent-Written Papers Are Backed by Their Artifacts

    [https://arxiv.org/abs/2609.22111](https://arxiv.org/abs/2609.22111)

    提出ReAgent自动化审计框架，通过将智能体撰写论文中的科学声明与代码仓库进行比对，验证论文结论是否真正被其代码和实验产物所支撑。

    

    大型语言模型智能体越来越能够自主开展研究，在产出研究文档的同时，也生成表面上支持这些文档的代码和实验。然而，所报告的研究发现是否始终得到相应实现和执行证据的支持，这一问题在很大程度上仍未被探索：现有的评审实践主要评估文本质量，无法可靠地识别诸如硬编码指标、未实现方法或缺乏支持的实验结果等不一致性。我们提出了ReAgent，一个用于评估智能体生成的研究文档与其相关代码仓库之间一致性的自动化审计框架。ReAgent从研究文档中构建科学声明的结构化表示，并利用这些表示来指导仓库分析和证据收集。静态审计检查所声称的方法、实现和实验配置是否……

    arXiv:2609.22111v1 Announce Type: new  Abstract: Large language model agents are increasingly capable of conducting research autonomously, producing research documents alongside the code and experiments that ostensibly support them. Yet whether the reported findings are consistently supported by corresponding implementations and execution evidence remains largely unexplored: existing review practices primarily assess textual quality and cannot reliably identify inconsistencies such as hard-coded metrics, unimplemented methods, or unsupported experimental results. We present ReAgent, an automated auditing framework for assessing the consistency between agent-generated research documents and their associated repositories. ReAgent constructs structured representations of scientific claims from research documents and uses them to guide repository analysis and evidence collection. Static auditing examines whether claimed methodologies, implementations, and experimental configurations are co
    
[^221]: 面向非洲环境评估微调与基础语言模型在孕产妇健康和疫苗接种医疗中的表现

    Evaluating Fine-Tuned and Base Language Models in Maternal and Vaccination Healthcare for African Settings

    [https://arxiv.org/abs/2609.22110](https://arxiv.org/abs/2609.22110)

    本研究通过在尼日利亚本地孕产妇健康和疫苗接种问答数据上对Llama模型进行低秩适配微调，构建了领域专用模型MamaBot-Llama和Vax-Llama，并证明其在准确性、安全性和文化适宜性上优于基础模型，为非洲低资源医疗场景提供了更可靠的AI健康信息解决方案。

    

    背景：大型语言模型（LLMs）可以改善低资源环境中的医疗信息传递，但可能产生不准确或文化上不恰当的建议。本研究评估了针对尼日利亚孕产妇健康和疫苗接种领域的专门微调效果。目的：比较HelpMum的MamaBot-Llama和Vax-Llama与Meta的Llama-3.1-8B-Instruct在准确性、安全性、清晰度、情境适宜性和可信度方面的表现。方法：我们评估了200个医疗问题，孕产妇健康和疫苗接种各100个，每个领域涵盖五个子领域。MamaBot-Llama和Vax-Llama分别使用低秩适配（Low-Rank Adaptation）技术在超过36,000个孕产妇健康和9,000个疫苗接种问答对上进行了微调。两名尼日利亚持证医生使用5点李克特量表独立对回答进行评分。配对比较采用Wilcoxon符号秩检验。结果：表现因领域而异。MamaBot-Llama显著优于（摘要在此处截断）

    arXiv:2609.22110v1 Announce Type: new  Abstract: Background: Large language models (LLMs) can improve healthcare information delivery in low-resource settings but may produce inaccurate or culturally inappropriate advice. This study evaluated domain-specific fine-tuning for maternal health and vaccination in Nigeria. Objective: To compare HelpMum's MamaBot-Llama and Vax-Llama with Meta's Llama-3.1-8B-Instruct for accuracy, safety, clarity, contextual appropriateness, and trustworthiness. Methods: We evaluated 200 healthcare questions, 100 each for maternal health and vaccination, across five subdomains per domain. MamaBot-Llama and Vax-Llama were fine-tuned using Low-Rank Adaptation on over 36,000 maternal health and 9,000 vaccination question-answer pairs, respectively. Two Nigerian licensed physicians independently rated responses using a 5-point Likert scale. Paired comparisons used Wilcoxon signed-rank tests. Results: Performance varied by domain. MamaBot-Llama significantly outper
    
[^222]: 通用多模态基础模型

    Generalized Multimodal Foundation Model

    [https://arxiv.org/abs/2609.22107](https://arxiv.org/abs/2609.22107)

    提出了一种不依赖特定模态的通用多模态基础模型，通过在大规模具有多样因果结构的合成多模态数据集上训练，使其能够适用于任意的模态组合和任意的预测任务。

    

    利用多模态数据进行预测在多种场景中被广泛应用。现有的多模态融合模型一旦部署，只能处理预定义的模态（如视觉、文本和音频）和单一任务，难以快速适应新的下游应用。因此，一个自然但相当大胆的问题随之而来：是否存在一种通用的多模态融合模型，能够应用于任意的模态组合和任意的预测任务？我们认为，统一的多模态融合模型不应依赖于特定模态，而应编码可迁移的多模态关联模式。为此，我们提出了一种简单而有效的学习范式，其基于在大规模合成的多模态数据集上进行训练，这些数据集具有多样的因果结构，能够形式化地刻画现实世界中多模态数据的生成过程。在该框架的基础上，我们提出了……（摘要原文在此处截断）

    arXiv:2609.22107v1 Announce Type: cross  Abstract: Making prediction with multimodal data is widely used in diverse scenarios. Existing multimodal fusion models, once deployed, can only handle predefined modalities (e.g., vision, text and audio) and single tasks, making it difficult to quickly adapt to new downstream applications. Therefore, a natural yet rather aggressive question arises, whether there exists a general multimodal fusion model that can be applied to arbitrary modality combinations and arbitrary prediction tasks. We argue that a unified multimodal fusion model should not depend on specific modalities and instead encode transferable patterns of multimodal correlation. To this end, we propose a simple and effective learning paradigm based on training over the generation of large-scale synthetic multimodal datasets with diverse causal structures that formally characterize the generative processes of multimodal data in real world. Building on this framework, we propose the 
    
[^223]: DeepInstructor：一种基于经验驱动的创意评估的智能体AI导师

    DeepInstructor: An Agentic AI Instructor for Experience-Driven Idea Evaluation

    [https://arxiv.org/abs/2609.22104](https://arxiv.org/abs/2609.22104)

    DeepInstructor通过从58,607条同行评审构建经验图谱，并利用基于ReAct的智能体检索维度特定证据，将研究创意评估转化为对结构化学术经验的推理，从而显著提升了与人类评估判断的一致性。

    

    随着自动化科学发现的不断推进，大语言模型（LLM）如今能够以前所未有的规模生成研究创意，使瓶颈从创意生成转向了创意评估。现有的评估器主要依赖LLM的参数化知识或非结构化检索，其产生的判断缺乏人类导师所采用的基于经验的推理。为解决这一问题，我们提出了DeepInstructor，一个将创意评估形式化为对结构化学术经验进行推理的智能体框架。DeepInstructor从58,607条同行评审中构建了经验图谱，并采用基于ReAct的智能体检索针对特定维度的证据，实现可追溯的评估。我们进一步引入了DeepInstruct数据集，该数据集包含针对新颖性、重要性和可行性的受控成对比较。实验表明，DeepInstructor显著优于现有基线方法，提升了与人类评估在Hit@1和Hit@2指标上的一致性。

    arXiv:2609.22104v1 Announce Type: new  Abstract: As automated scientific discovery advances, Large Language Models (LLMs) can now generate research ideas at an unprecedented scale, shifting the bottleneck from idea generation to idea evaluation. Existing evaluators mainly rely on parametric LLM knowledge or unstructured retrieval, producing judgments that lack the experience-grounded reasoning used by human instructors. To address this, we propose DeepInstructor, an agentic framework that formulates idea evaluation as reasoning over structured scholarly experience. DeepInstructor constructs an Experience Graph from 58,607 peer reviews and employs a ReAct-based agent to retrieve dimension-specific evidence for traceable evaluation. We further introduce DeepInstruct, a dataset with controlled pairwise comparisons across novelty, significance, and feasibility. Experiments show that DeepInstructor substantially outperforms existing baselines, improving Hit@1 and Hit@2 alignment with human 
    
[^224]: 当你的身份可以改变你获得的代码：LLM代码生成中角色诱导偏见的研究

    When Who You Are Can Change the Code You Get: A Study of Persona-Induced Bias in LLM Code Generation

    [https://arxiv.org/abs/2609.22102](https://arxiv.org/abs/2609.22102)

    该研究通过35,000多个程序的大规模实证分析，首次系统揭示了用户的人口统计身份（国籍、性别、经验水平）会诱导LLM代码生成产生偏见，人口统计标记泄露于高达65%的响应和70%的推理轨迹中，并影响代码的功能正确性、可维护性、风格和安全性。

    

    大型语言模型（LLM）被广泛用作编程助手，然而用户的身份信息是否以及如何影响生成代码的技术质量仍不清楚。我们对基于LLM的代码生成中的角色诱导偏见进行了大规模实证研究，重点关注一个专有模型（Gemini 2.5 Pro）和一个开放权重模型（GPT-OSS-120B）。我们使用涵盖国籍、性别和经验水平的18种人口统计角色，将角色诱导的提示与中性基线进行比较。在35,000多个生成的程序中，我们分析了推理和响应中人口统计标记的泄露情况，以及功能正确性、可维护性、代码风格和安全性方面的差异。我们的结果表明，人口统计线索经常反映在LLM的推理和输出中。尽管与任务在语义上无关，人口统计标记出现在多达65%的响应和70%的推理轨迹中。

    arXiv:2609.22102v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are widely used as programming assistants, yet it remains unclear whether and how user's demographic information impacts the technical quality of generated code. We conduct a large-scale empirical study of persona-induced bias in LLM-based code generation, focusing a proprietary model (Gemini 2.5 Pro) and an open-weight model (GPT-OSS-120B). Using 18 demographic personas spanning nationality, gender, and experience level, we compare persona-induced prompts against a neutral baseline. Across 35,000+ generated programs, we analyze demographic marker leakage in reasoning and responses, as well as differences in functional correctness, maintainability, code style, and security.   Our results show that demographic cues are frequently reflected in LLM reasoning and outputs. Demographic markers appear in up to 65% of responses and 70% of reasoning traces, despite being semantically irrelevant to the tasks. On Live
    
[^225]: 长上下文语言模型中的上下文污染：一种极值注意力干扰现象

    Context Poisoning as Extreme-Value Attention Interference in Long-Context Language Models

    [https://arxiv.org/abs/2609.22101](https://arxiv.org/abs/2609.22101)

    该论文将长上下文语言模型中的“上下文污染”形式化为注意力中的极值干扰，推导出证据边际需随有效干扰项数量按 Ω(√(log N)) 增长才能维持检索准确率的理论上界，并揭示得分混叠、位置混叠与softmax稀释是长上下文性能退化的核心机制。

    

    大型语言模型能够处理越来越长的提示，但随着不相关或易混淆上下文的加入，其定位和使用决定性证据的能力可能会下降。我们将这一现象称为“上下文污染”，并将其形式化为注意力中的极值干扰：决定性证据的得分存在上界，而有效干扰项中的最大得分会随其数量增加而增长。在softmax检索抽象下，我们推导出一个有限样本上界，表明要在基准率之上维持固定的准确率目标，证据边际需以 $\Omega(\sqrt{\log N})$ 的比例扩展，其中 N 表示有效干扰项的数量，而不一定是原始上下文长度。该分析将长上下文性能退化与得分混叠、位置混叠以及softmax稀释联系起来。受控实验表明，在存在嵌入硬负例的情况下，检索准确率随总上下文长度的增长而下降……

    arXiv:2609.22101v1 Announce Type: new  Abstract: Large language models can process increasingly long prompts, yet their ability to locate and use decisive evidence may degrade as irrelevant or confusable context is added. We formulate this phenomenon, which we call context poisoning, as extreme-value interference in attention: the decisive-evidence score is upper-bounded, while the maximum score among effective distractors grows with their number. Under a softmax retrieval abstraction, we derive a finite-sample upper bound showing that maintaining a fixed accuracy target above base rate requires the evidence margin to scale as $\Omega(\sqrt{\log N})$, where N denotes the effective distractor count rather than necessarily the raw context length. The analysis connects long-context degradation to score aliasing, positional aliasing, and softmax dilution. Controlled experiments show that retrieval accuracy decreases as total context grows in the presence of embedded hard negatives, that th
    
[^226]: AdaMem：面向检索增强生成中软压缩的自适应记忆令牌分配

    AdaMem: Adaptive Memory Token Allocation for Soft Compression in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.22100](https://arxiv.org/abs/2609.22100)

    AdaMem提出了一种相关性引导的自适应软压缩框架，根据段落与查询的相关性动态分配固定的记忆令牌预算，为高相关段落分配更多记忆令牌并舍弃低相关段落，从而提升检索增强生成的效率与效果。

    

    检索增强生成（RAG）通过检索到的证据来提升语言模型的表现，但处理大量长文本段落成本高昂，且可能引入干扰信息。软压缩技术通过在生成之前将段落编码为紧凑的连续记忆嵌入序列来应对这一挑战。然而，现有方法通常为每个保留的段落分配相同数量的记忆嵌入，而不考虑其与查询的相关性。为解决这一问题，我们提出了AdaMem，这是一个由相关性引导的软压缩框架，它将学习到的段落相关性估计映射为基于查询的固定记忆令牌预算分配。一个共享的查询条件压缩器在单次处理中同时生成连续的段落记忆和相关性分数；一个确定性的分配规则为分数较高的段落分配更多的记忆令牌，并可以省略低分数的段落。在六个开放域问答基准测试中，AdaMem始终取得一致的优势表现。

    arXiv:2609.22100v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) improves language models with retrieved evidence, but processing many long passages is costly and can introduce distracting information. Soft compression addresses this challenge by encoding passages as compact sequences of continuous memory embeddings before generation. However, existing methods typically assign each retained passage an identical number of memory embeddings, irrespective of its query-specific relevance. To address this, we propose AdaMem, a relevance-guided soft-compression framework that maps learned passage-relevance estimates to a query-dependent allocation of a fixed memory-token budget. A shared query-conditioned compressor produces both continuous passage memories and relevance scores in a single pass; a deterministic allocation rule assigns more memory tokens to higher-scoring passages and can omit low-scoring ones. Across six open-domain QA benchmarks, AdaMem consistently out
    
[^227]: 一个食谱数据结构框架及其在烹饪与营养洞察中的应用

    A framework for recipe data structure with applications for culinary and nutritional insights

    [https://arxiv.org/abs/2609.22099](https://arxiv.org/abs/2609.22099)

    该论文提出了一种食谱数据结构框架，将食谱分解为类型化配料实体并锚定到营养数据库及地理文化背景，据此构建了包含来自32个地区、99个国家的128,942个食谱的结构化数据库RecipeDB2，实现了食谱信息的可计算查询。

    

    烹饪是一个将原材料转化为美味且营养丰富菜肴的复杂过程，然而编码这一过程的食谱大多仍为自由文本——人类可读，但无法直接被计算机处理。现有的食谱集合仅捕捉了这些信息的碎片，尚无一种共享的表示方法能够在单一可查询的模式中，将食谱的结构化配料组成、其地理文化来源以及其营养概况联系起来。我们通过形式化一个食谱数据结构框架来解决这一表示缺口，该框架将每个食谱分解为类型化的配料实体，将这些实体锚定到参考营养数据库，并用地理文化和饮食上下文对其进行标注。我们提出了RecipeDB2，这是一个结构化的数据汇编，包含来自32个地区、99个国家的128,942个食谱和35,474种配料。配料短语通过基于Transformer的命名实体模型被解析为七个烹饪属性。

    arXiv:2609.22099v1 Announce Type: new  Abstract: Cooking is a complex process that transforms raw ingredients into delicious and nutritious dishes, yet the recipes that encode this process remain largely free text; readable by people but not directly computable. Existing recipe collections capture fragments of this information, but no shared representation links a recipe's structured ingredient composition, its geo-cultural provenance, and its nutritional profile within a single queryable schema. We address this representation gap by formalizing a framework for recipe data structure that decomposes each recipe into typed ingredient entities, grounds those entities in a reference nutritional database, and annotates them with geo-cultural and dietary context. We present RecipeDB2, a structured compilation of 128,942 recipes with 35,474 ingredients from 32 regions and 99 countries. Ingredient phrases are parsed into seven culinary attributes using a transformer-based named-entity model; i
    
[^228]: TreeSpark：面向半自回归推测解码的校准、负载自适应草稿树

    TreeSpark: Calibrated, Load-Adaptive Draft Trees for Semi-Autoregressive Speculative Decoding

    [https://arxiv.org/abs/2609.22098](https://arxiv.org/abs/2609.22098)

    TreeSpark以可忽略的成本从草稿器现有马尔可夫头读取父节点条件分布并校准为边接受概率估计，用路径存活率统一驱动草稿树的最优优先扩展、逐轮停止与负载自适应规模调整，从而改进半自回归推测解码的草稿树构建。

    

    推测解码通过让一个廉价的草稿模型提出候选词元、再由目标模型并行验证，从而加速语言模型推理。近期的块草稿器使草稿生成几乎零成本：一次骨干网络前向传播即可生成整块草稿词元。草稿树有望带来进一步提升——在一次目标模型前向传播中验证多个备选续写——但现有构造按逐位置的边际概率对候选进行排序，忽略了候选所扩展的父节点，因此在半自回归草稿器上，更宽的树大多只会增加排序错误的节点；而且固定大小的树忽略了每一解码轮次以及每个服务负载所能支撑的推测量。我们提出TreeSpark，它以可忽略的成本从草稿器现有的马尔可夫头中读取父节点条件分布，将其校准为边接受概率估计，并让路径存活率决定其余一切：最优优先扩展、逐轮停止以及负载自适应的规模调整。

    arXiv:2609.22098v1 Announce Type: new  Abstract: Speculative decoding accelerates language-model inference by letting a cheap drafter propose tokens that the target model verifies in parallel. Recent block drafters make drafting nearly free: a single backbone pass emits an entire block of draft tokens. Draft trees promise a further gain -- several alternative continuations verified in one target forward -- but existing constructions rank candidates by per-position marginals that ignore which parent a candidate extends, so on semi-autoregressive drafters wider trees mostly add mis-ranked nodes; and a tree of fixed size ignores how much speculation each decoding round, and each serving load, can support. We introduce TreeSpark, which reads a parent-conditioned distribution from the drafter's existing Markov head at negligible cost, calibrates it into an edge-acceptance estimate, and lets path survival govern everything else: best-first expansion, per-round stopping, and a load-adaptive s
    
[^229]: 代码的Token签名：比较大型语言模型之间的编码行为

    Token Signatures of Code: Comparing Coding Behaviors Across Large Language Models

    [https://arxiv.org/abs/2609.22097](https://arxiv.org/abs/2609.22097)

    提出CLIC可视化分析方法，通过token频率分析与可解释决策树刻画不同大语言模型的编码行为差异，并引入鲁棒性和集中度两个新指标来超越单纯性能评估。

    

    大型语言模型（LLM）在编码任务上的评估主要集中于pass@k等性能指标。随着LLM的不断进步，许多模型现已达到基线性能要求，这降低了仅基于性能的评估的区分能力。然而，一个关键问题在很大程度上仍未被探索：LLM在编码行为上有何差异？我们提出了CLIC（Code Learning for Identification and Comparison，用于识别与比较的代码学习），这是一种通过token频率分析来刻画LLM编码行为的可视化分析方法。CLIC将每个代码样本表示为token频率的特征向量，并训练一个可解释的决策树来区分两个LLM的代码集。除了分类准确率之外，我们还定义了两个新指标：鲁棒性，衡量随着最具区分性的token被逐步移除，两个LLM是否仍可被区分；集中度，衡量这种差异是否由少数token驱动。

    arXiv:2609.22097v1 Announce Type: new  Abstract: The evaluation of large language models (LLMs) on coding tasks has primarily focused on performance metrics such as pass@k. As LLMs continue to advance, many models now meet baseline performance requirements, reducing the discriminative power of performance-based evaluation alone. Yet a key question remains largely unexplored: how do LLMs differ in their coding behavior? We propose CLIC (Code Learning for Identification and Comparison), a visual analytics approach that characterizes LLM coding behavior through token-frequency analysis. CLIC represents each code sample as a feature vector of token frequencies and trains an interpretable decision tree to separate two LLMs' code sets. Beyond classification accuracy, we define two new metrics: robustness, which measures whether the two LLMs remain distinguishable as their most-discriminative tokens are progressively removed, and concentration, which measures whether the difference is driven 
    
[^230]: X平台上气候变化运动中AI推断的幸福感表达与集体行动话语

    AI-inferred expressed well-being and collective-action discourse in climate-change campaigns on X

    [https://arxiv.org/abs/2609.22096](https://arxiv.org/abs/2609.22096)

    该研究通过分析X平台上36万余条气候运动相关帖子，发现活动期间幸福感表达显著上升9.02个百分点，但行动语言却下降10.75个百分点，揭示了气候运动话语中“幸福感与行动背离”的现象。

    

    气候运动通常通过关注度和动员规模来评估，但对其伴随的幸福感语言却知之甚少。运动期间是否会改变积极情绪与希望，以及幸福感是否与行动语言相一致，这些问题仍未有定论。我们分析了来自地球日、地球一小时、全球气候行动日和世界环境日的364,118条公开Twitter/X帖子，覆盖19个发生年份，并采用活动前30天、活动期间和活动后的时间窗口。通过一个版本化的加权词汇模型，我们估计了幸福感、面向未来的希望、集体能力、痛苦和行动语言。结果显示，活动期间的幸福感流行率比活动前基线高出9.02个百分点，而配对发生对比则显示行动语言下降了10.75个百分点，表明存在幸福感与行动之间的背离。幸福感估计在构成变化和文本去重等稳健性检验中保持为正，但在某种条件下精度有所下降（原文在此处截断）。

    arXiv:2609.22096v1 Announce Type: new  Abstract: Climate campaigns are often evaluated through attention and mobilization, but less is known about the well-being language that accompanies them. Whether campaign periods alter positive affect and hope, and whether happiness aligns with action language, remains unresolved. We analysed 364,118 public Twitter/X posts from Earth Day, Earth Hour, Global Climate Action Day and World Environment Day in 19 occurrence-years, using 30-day pre-event, event and post-event windows. A versioned weighted lexical model estimated happiness, future-oriented hope, collective capability, distress and action language. Event-period happiness prevalence was 9.02 percentage points higher than the pre-event baseline , whereas paired occurrence contrasts showed a 10.75-point decline in action language, indicating a happiness--action divergence. The happiness estimate remained positive across composition and text-deduplication checks, but was less precise under a 
    
[^231]: 总结、判断、优化：面向多模态内容审核的解耦式内容理解与策略学习

    Summarize, Judge, Refine: Decoupled Content Understanding and Policy Learning for Multimodal Content Moderation

    [https://arxiv.org/abs/2609.22094](https://arxiv.org/abs/2609.22094)

    提出SJR双模型架构，通过自然语言接口将多模态内容理解与策略学习解耦，借助文本空间数据增强和GRPO共训练，实现少样本策略适配与内置可解释性的多模态内容审核。

    

    传统的内容审核系统通常将多模态理解与特定策略的分类耦合在一起，导致每次策略变更都需要对整个流水线进行重新训练，并且由于多媒体内容无法进行有意义的数据增强而面临标签稀缺的问题。我们提出了 Summarize-Judge-Refine（SJR），这是一种双模型架构，通过自然语言接口将上述两个关注点解耦：多模态内容模型生成结构化的文本摘要，纯文本策略模型则依据策略定义对这些摘要进行分类。迭代式共训练循环通过 GRPO 优化内容模型，使其生成与策略相关的摘要；同时，文本空间的增强技术生成对抗性摘要变体——这是一种在原始多媒体上无法实现的增强途径——从而实现少样本策略冷启动。每个决策都基于人类可读的摘要，使可解释性成为结构性副产品。在误导性广告检测任务上，SJR 取得了 +2（原文摘要在此处截断）

    arXiv:2609.22094v1 Announce Type: new  Abstract: Content moderation systems traditionally entangle multimodal understanding with policy-specific classification, requiring full pipeline retraining for every policy change and suffering from label scarcity since multimedia cannot be meaningfully augmented. We propose Summarize-Judge-Refine (SJR), a two-model architecture that decouples these concerns via a natural language interface: a multimodal Content Model produces structured text summaries, and a text-only Policy Model classifies them against policy definitions. An iterative co-training loop refines the Content Model via GRPO to produce policy-relevant summaries, while text-space augmentation generates adversarial summary variants---an augmentation pathway impossible on raw multimedia---enabling few-shot policy bootstrap. Every decision is grounded in a human-readable summary, providing interpretability as a structural byproduct. On misleading advertisement detection, SJR achieves +2
    
[^232]: 向前看的记忆：用于个人记忆检索的零推理前瞻项

    Memory That Looks Forward: A Zero-Inference Prospective Term for Personal Memory Retrieval

    [https://arxiv.org/abs/2609.22091](https://arxiv.org/abs/2609.22091)

    提出一种零推理开销的前瞻记忆检索项，通过显式承诺账本在触发时为相关记忆项提供乘法显著性提升，将困难任务层的recall@5从0.000提升至0.955-1.000且零误提升。

    

    对个人记忆库的检索是回顾性的：它呈现与查询相似的内容，却对用户已承诺要做的事情视而不见。我们描述了一种用于记忆检索的前瞻项，它在查询时不产生任何推理开销。承诺以带日期或触发条件的条目形式保存在显式账本中；与被触发的条目相关联的记忆项会获得显著性提升，并以乘法方式混合到基于嵌入的检索中，从而使相关性始终保持主导地位。在一个模拟TriggerBench已发布结构构建的合成前瞻记忆任务集（48个盲写对话、175个任务）上，该前瞻项在默认混合权重下将困难层的recall@5从0.000提升到0.955，在地板变体下提升到1.000，并且在53个已解决承诺任务中实现了零误提升。盲写实验还产生了一个适用范围的发现：只有17-29%的自然表述的承诺-触发对能够击败嵌入相似性，因此该前瞻项（摘要在此处截断）

    arXiv:2609.22091v1 Announce Type: new  Abstract: Retrieval over a personal memory store is retrospective: it surfaces what resembles the query, and it is blind to what the user has committed to do. We describe a prospective term for memory retrieval that costs no inference at query time. Commitments are held in an explicit ledger as dated or trigger-conditioned entries; memory items linked to a firing entry receive a salience boost, blended multiplicatively into embedding-based retrieval so that relevance remains sovereign. On a synthetic prospective-memory task set modeled on TriggerBench's published structure (48 blind-authored dialogues, 175 tasks), the term raised recall@5 on the hard stratum from 0.000 to 0.955 at the default blend weight and to 1.000 under a floor variant, with zero false boosts across 53 resolved-commitment tasks. Blind authorship also produced a scope finding: only 17-29% of naturally phrased commitment-trigger pairs defeat embedding similarity, so the term mat
    
[^233]: 识别、模拟与拒绝：面向LLM智能体中经典心理学效应的数据污染感知研究

    Recognition, Simulation, and Refusal: A Contamination-Aware Study of Classic Psychological Effects in LLM Agents

    [https://arxiv.org/abs/2609.22090](https://arxiv.org/abs/2609.22090)

    提出PsyAgentBench基准，通过命名/盲测、规范/反事实任务版本与人格操纵的因子设计，区分LLM是真正具备心理偏差还是仅在识别并模拟心理学效应，发现类人效应源于标签门控等质性不同的机制而非单一易感性。

    

    一个大语言模型产生与人类心理效应相关的反应模式，并不等同于该模型真正拥有这种偏差。我们提出了PsyAgentBench，这是一个在LLM智能体上重新运行经典心理学实验的基准，采用因子设计来区分这两种情况：每个范式在提示中被明确标注范式名称（命名条件）或被框架为常规任务（盲测条件），并且使用任务的教科书原版（规范版本）或经结构匹配、旨在减少与可能训练数据在词汇和场景上重叠的变体（反事实版本），同时与人格操纵进行交叉组合。在五个已完成的范式中，对最多三个开放权重模型系列进行评估并发布了41,904次试验，结果表明表面上类人的效应通过质性不同的途径产生，而非源于单一易感性：例如范式标签门控与显式覆盖机制（阿希从众实验中，gpt

    arXiv:2609.22090v1 Announce Type: new  Abstract: An LLM producing the response pattern associated with a human psychological effect is not the same claim as the LLM possessing that bias. We present PsyAgentBench, a benchmark that re-runs classic psychology experiments on LLM agents under a factorial design built to separate these: each paradigm is run with the paradigm explicitly labeled in the prompt (named) or framed as a routine task (blind), and on the literal textbook version of the task (canonical) or a structurally matched variant written to reduce lexical and scenario overlap with likely training data (counterfactual), crossed with a persona manipulation. Across five completed paradigms, evaluated on up to three open-weight model families with 41,904 trials released, apparently human-like effects arise through qualitatively different routes rather than one susceptibility: paradigm-label gating with explicit override (Asch conformity, 0 percent blind to 83.3 percent named on gpt
    
[^234]: RecreationWorld：面向混合计算机使用代理的可扩展且可验证的环境

    RecreationWorld: Scalable and Verifiable Environments for Hybrid Computer-Use Agents

    [https://arxiv.org/abs/2609.22000](https://arxiv.org/abs/2609.22000)

    提出了 RecreationWorld，一个跨五大平台、以“复刻正在运行的应用”为核心任务的框架，用于训练和评估能自主结合图形交互与软件开发的混合计算机使用代理，并以运行中的参考应用作为预言机提供基于执行的验证奖励。

    

    计算机使用代理沿着两条独立的路线发展：图形交互，以及通过代码和命令行进行软件开发。真实的数字化工作需要两者兼备，而且是交错进行而非简单串联。我们研究了混合计算机使用代理，它们能够自主决定何时探索界面、实现软件，以及运行并视觉验证其产物。我们提出了 RecreationWorld，一个围绕“复刻”构建的五平台框架：给定一个正在运行的参考应用，代理必须在没有任何规定工作流程的情况下，发现其行为并构建一个忠实的实现。RecreationWorld 在 Ubuntu、macOS、Windows、Android 和 Web 五个平台上提供可复现的环境，并配有统一的测试框架，包含原生 GUI 控制和编码工具。正在运行的参考应用充当隐藏行为测试的预言机，提供基于执行结果的奖励。我们利用高质量的开源应用程序来扩展轨迹生成。在这些轨迹上训练的模型（摘要在此处被截断）

    arXiv:2609.22000v1 Announce Type: new  Abstract: Computer-use agents (CUAs) have advanced along two separate lines: graphical interaction and software development through code and the command line. Real digital work requires both, interleaved rather than stacked end to end. We study hybrid CUAs that autonomously decide when to explore an interface, implement software, and run and visually verify their artifacts. We introduce RecreationWorld, a five-platform framework built around recreation: given a running reference, an agent must discover its behavior and build a faithful implementation with no prescribed workflow. RecreationWorld provides reproducible environments on Ubuntu, macOS, Windows, Android, and Web, plus a unified harness with native GUI control and coding tools. The running reference serves as an oracle for hidden behavioral tests, providing execution-grounded rewards. We scale trajectory generation with high-quality open-source applications. Models trained on these trajec
    
[^235]: 面向作物病害与虫害诊断的可配置多阶段视觉流水线

    Configurable Multi-Stage Vision Pipeline for Crop Disease and Pest Diagnosis

    [https://arxiv.org/abs/2609.21651](https://arxiv.org/abs/2609.21651)

    针对小农户仅凭一张田间照片进行作物诊断的场景，本文提出一种可配置的多阶段视觉流水线，支持调节照片质量拒绝阈值与置信度截止值、并可扩展添加作物与病虫害类别，同时基于来自四个国家的116万张真实照片揭示了现有生产系统的不足。

    

    Farmer.Chat 是 Digital Green 面向小农户的农业咨询服务。当作物看起来出现异常时，农户会拍摄照片并发送，这张照片本身就是全部的问题信息：没有症状描述，没有作物名称，往往连文字都没有。该服务必须仅凭这些照片判断图像是否可用、图中是什么作物、以及作物出了什么问题——而这些照片是用廉价手机在田间、光线不佳、镜头晃动的条件下拍摄的。目前执行这一任务的系统无法调整：它没有可调节的照片拒绝阈值，无法添加新的作物和问题类别，也没有可设置的置信度截止值。我们研究了从埃塞俄比亚、印度、肯尼亚和尼日利亚发送给 Farmer.Chat 的约116万张照片。生产环境中的质量门控拒绝了其判定图像的46.8%，进入诊断阶段的图像中有超过四分之一未能返回作物名称，而在被标注为“病害”的问题中有35.8%实际上是虫害，

    arXiv:2609.21651v1 Announce Type: cross  Abstract: Farmer.Chat is Digital Green's farm advisory service for smallholder farmers. When something looks wrong with a crop, the farmer takes a photograph and sends it, and that photograph is the whole question: no symptom described, no crop named, often no text at all. The service has to determine whether the picture can be used, what crop it shows, and what is wrong with it, from images taken on cheap phones in a field, in poor light and with a moving camera. The system doing this today cannot be adjusted. It has no adjustable thresholds for photograph rejection, crops and problems cannot be added, and there is no confidence cut-off to set.   We study about 1.16 million photographs sent to Farmer.Chat from Ethiopia, India, Kenya and Nigeria. The production quality gate rejected 46.8% of the images it judged, over a quarter of those reaching diagnosis returned no crop name, and 35.8% of the labelled problems filed under "disease" are pests, 
    
[^236]: GameLogicBench：基于逐帧状态断言评估编码智能体的运行时游戏逻辑

    GameLogicBench: Evaluating Coding Agents on Runtime Game Logic with Tick-Level State Assertions

    [https://arxiv.org/abs/2609.21562](https://arxiv.org/abs/2609.21562)

    GameLogicBench是一个包含72个Godot游戏逻辑任务的基准，其自动评估器在每次模拟tick逐帧检查游戏规则，覆盖403个场景共1,451个测试用例，既能接受多样化的正确实现又能拒绝突变体，且判定结果完全可复现。

    

    编码智能体能够在大型软件项目中修改和测试代码。游戏开发是一个要求智能体实现游戏规则的领域。一个游戏即使在整个运行过程中违反了规则，最终也可能以合法状态结束。现有的游戏开发基准测试要么重放固定示例，要么对视频进行评分，要么让另一个模型来评判结果。然而，目前还没有任何基准测试能够在评估者选择的多种场景中贯穿整个执行过程检查游戏规则，同时确保判定结果完全可复现。我们提出了GameLogicBench，这是一个包含Godot项目中72个游戏逻辑任务的基准测试。自动评估器会在每次模拟tick检查每个游戏的规则。在403个手工设计的场景中，通过种子参数变化生成了1,451个测试用例。为确保评估器衡量的是行为本身而非实现方式的选择，评估器必须接受每个任务的不同正确实现，同时拒绝突变体（即缺少某项需求的实现）……

    arXiv:2609.21562v1 Announce Type: cross  Abstract: Coding agents can modify and test code across large software projects. Game development is a domain where agents must implement gameplay rules. A game can end in a valid state even after violating its rules during the run. Current game-development benchmarks replay fixed examples, score videos, or ask another model to judge the result. However, no existing benchmark checks game rules throughout execution across varied evaluator-selected scenarios while ensuring exactly reproducible verdicts. We introduce GameLogicBench, a benchmark of 72 gameplay-logic tasks in Godot projects. An automated evaluator checks each game's rules at every simulation tick. Across 403 hand-designed scenarios, seeded parameter variations produce 1,451 test cases. To ensure that the evaluator measures behavior rather than implementation choice, it must accept different correct implementations for each task while rejecting mutants, implementations with one requir
    
[^237]: 可信的“AI金融”：解析人们如何评判AI中介的金融建议

    Trustworthy FinAInce: Unpacking How AI-Mediated Financial Advice is Judged

    [https://arxiv.org/abs/2609.20989](https://arxiv.org/abs/2609.20989)

    该研究通过对285名美国成年人的随机情景实验发现，建议风格（AI、专家、在线社区）是塑造人们对金融建议信息与安全性评价的最关键因素，这些评价共同解释了信任与依赖判断的大部分方差，且专家风格建议即使在没有来源标签时也最受偏好。

    

    随着生成式AI日益成为个人理财指导的来源，理解人们如何评价此类建议对于促进合理依赖至关重要。我们对285名美国成年人开展了一项随机情景实验，涵盖八项财务决策，在保持底层建议内容一致的前提下，独立变换三种建议风格（AI、专家、在线社区）并展示来源标签。结果显示：建议风格对信息评价和安全性评价的影响最强；专家标签选择性地提升了感知的来源专业知识；而决策情境主要影响风险与安全性评价。这些评价与下游判断密切相关，模型分别解释了整体质量（69.2%）、信任（75.9%）和预期依赖（82.9%）的大部分方差。即使在不显示来源标签的情况下，专家风格的建议仍最受偏好。我们的发现对理解金融建议评价和设计可信的AI金融助手具有重要意义。

    arXiv:2609.20989v1 Announce Type: cross  Abstract: As generative AI is increasingly used as a source of personal financial guidance, understanding how people appraise such advice is important for supporting appropriate reliance. We conducted a randomized vignette experiment with 285 U.S. adults across eight financial decisions, independently varying three advice styles---AI, expert, and online community---and displayed source labels while holding the underlying recommendation consistent. Advice style most strongly shaped message and safety appraisals, Expert labels selectively increased perceived source knowledge, and decision context primarily shaped risk and safety appraisals. These appraisals were associated with downstream judgments, with models explaining 69.2% of overall quality, 75.9% of trust, and 82.9% of intended reliance. Expert-style advice also remained most preferred when shown without source labels. Our findings have implications for understanding financial advice evalua
    
[^238]: UniPolicy：面向生成式搜索广告的统一目标特定策略

    UniPolicy: Unified Objective-Specific Policies for Generative Search Advertising

    [https://arxiv.org/abs/2609.20630](https://arxiv.org/abs/2609.20630)

    UniPolicy提出了一种目标感知的多策略对齐框架，通过目标特定前缀标记、稀疏MoE-LoRA路由和残差FFN在共享骨干网络中分层解耦参数，使生成式搜索广告能够联合优化相关性、点击倾向和商业价值等异构目标，避免梯度竞争导致的全局次优问题。

    

    搜索广告将用户意图与商业内容连接起来，在平台变现中发挥着关键作用。近期的系统通常将预训练生成模型与单一业务奖励（如eCPM）对齐，或使用朴素的奖励融合进行初步的多目标对齐。然而，理想的搜索广告系统必须联合考虑异构的多个目标，包括相关性、点击倾向和商业价值，以在平衡用户体验与商业价值的同时，缓解由梯度竞争导致的全局次优性能。我们提出了UniPolicy，一个目标感知的多策略对齐框架。UniPolicy结合了目标特定的前缀标记、稀疏MoE-LoRA路由和目标特定的残差FFN，在共享骨干网络内分层解耦参数，为不同的业务目标提供差异化的参数空间和策略表达空间。它进一步构建了……（摘要在此处截断）

    arXiv:2609.20630v1 Announce Type: new  Abstract: Search advertising connects user intent with commercial content and plays a critical role in platform monetization. Recent systems typically align pretrained generative models with a single business reward, such as eCPM, or use naive reward fusion for preliminary multi-objective alignment. However, an ideal search advertising system must jointly account for heterogeneous objectives, including relevance, click propensity, and commercial value, to balance user experience and business value while mitigating globally suboptimal performance caused by gradient competition. We propose UniPolicy, an objective-aware multi-policy alignment framework. UniPolicy combines objective-specific prefix tokens, sparse MoE-LoRA routing, and objective-specific residual FFNs to hierarchically decouple parameters within a shared backbone, providing differentiated parameter and policy-expression spaces for different business objectives. It further constructs pa
    
[^239]: 异种可解释性：探索大语言模型的“异类心智”

    Xeno-Interpretability: Investigating the Alien Minds of LLMs

    [https://arxiv.org/abs/2609.20408](https://arxiv.org/abs/2609.20408)

    本文提出“异种可解释性”这一新研究方向，主张大语言模型内部可能存在人类概念无法充分描述的“异种表征”，其内部区分空间远超有限人类描述所能覆盖的范围，且实验识别与语义解释应当分开对待。

    

    大语言模型通常通过人类已有的概念来进行解释：真实性、拒绝、欺骗、人格、危害性以及相关类别。本文提出了一个问题：模型是否也可能表征并使用那些不存在恰当人类概念的区分。我们将这类内部结构称为“异种表征”，对其研究称为“异种可解释性”。我们区分了人类可解释的语义空间与“异种语义空间”——即模型原生表征中缺乏恰当人类概念对应物的区域。我们证明，大语言模型中可能的内部区分空间显著大于通过有限人类描述所能覆盖的空间。随后，我们将实验识别与语义解释加以区分：一个内部表征即使……可以被可复现地定位、进行几何刻画、加以因果操纵，并与下游行为建立关联。（摘要在此处不完整）

    arXiv:2609.20408v1 Announce Type: cross  Abstract: Large language models are usually interpreted through concepts that humans already possess: truthfulness, refusal, deception, personality, harmfulness, and related categories. This paper asks whether models may also represent and use distinctions for which no adequate human concept exists. We call such internal structures xeno-representations, and their study xeno-interpretability. We distinguish the human-interpretable semantic space from the xeno-semantic space: the region of model-native representations for which no adequate human conceptual counterpart is available. We show that the space of possible internal distinctions in an LLM is substantially larger than the space available through finite human descriptions. We then separate experimental identification from semantic interpretation: an internal representation may be reproducibly located, geometrically characterized, causally manipulated, and linked to downstream behaviour even
    
[^240]: F$^{2}$DR：面向DeepSearch工作流的细粒度全流程奖励框架

    F$^{2}$DR: A Fine-Grained Full-Pipeline Reward Framework for DeepSearch Workflows

    [https://arxiv.org/abs/2609.19827](https://arxiv.org/abs/2609.19827)

    该论文提出了F2DR框架，从内容、轨迹和答案三个维度对DeepSearch工作流进行细粒度全流程奖励评估，并构建了专门基准DeepSearch RM-Bench，显著提升了评估一致性。

    

    随着大语言模型（LLM）在工业界的广泛部署，DeepSearch已成为解决复杂用户查询的主流范式。它通常通过由规划与反思、信息检索和答案生成组成的迭代闭环工作流来运行。然而，现有的奖励模型（RM）和评估基准主要是为静态单轮任务设计的，无法捕捉DeepSearch工作流的全流程复杂性。为解决这一局限性，我们提出了F2DR，一个细粒度的全流程DeepSearch奖励框架。F2DR从内容、轨迹和答案三个维度对DeepSearch工作流进行评估，实现全面的流程级评估。我们进一步构建了DeepSearch RM-Bench，一个专门用于评估DeepSearch场景中奖励模型的基准。大量实验表明，F2DR实现了显著更高的评估一致性。

    arXiv:2609.19827v1 Announce Type: new  Abstract: With the widespread industrial deployment of Large Language Models (LLMs), DeepSearch has emerged as the dominant paradigm for resolving complex user queries. It typically operates through an iterative closed-loop workflow consisting of planning and reflection, information retrieval, and answer generation. However, existing reward models (RMs) and evaluation benchmarks are primarily designed for static single-turn tasks, failing to capture the full-pipeline complexity of DeepSearch workflows. To address this limitation, we propose F2DR, a fine-grained full-pipeline DeepSearch reward framework. F2DR evaluates DeepSearch workflows across three dimensions: Content, Trajectory, and Answer, enabling comprehensive process-level assessment. We further construct DeepSearch RM-Bench, a dedicated benchmark for evaluating RMs in DeepSearch scenarios. Extensive experiments demonstrate that F2DR achieves significantly higher evaluation consistency th
    
[^241]: 基于大语言模型标注数据的词典约束未分词语言字素到音素转换

    Dictionary-Constrained Grapheme-to-Phoneme for Unsegmented Languages from LLM-Annotated Data

    [https://arxiv.org/abs/2609.19805](https://arxiv.org/abs/2609.19805)

    本文提出一种利用词典构建词格并采用条件随机场评分的上下文感知神经G2P方法，结合大语言模型生成的超过200万条标注数据，显著提升了日语等未分词语言的字素到音素转换性能。

    

    字素到音素（G2P）转换将原始文本转换为其音素形式，是文本转语音（TTS）和自动语音识别（ASR）系统的重要组成部分，要求其快速、稳定且具备上下文感知能力。对于日语等未分词语言，G2P 还需要将词分词与高度依赖上下文的多音字消歧相结合，而准确标注数据的稀缺仍然是一个瓶颈。本文提出了一种上下文感知的神经 G2P 方法，该方法对从词典构建的词格上的判别式条件随机场（CRF）路径进行评分。为解决数据稀缺问题，我们利用大语言模型（LLM）生成了超过200万条句子。实验结果表明，我们的方法显著优于传统的基于形态分析器的方法和神经序列模型。在 Joyo-Kanji-Yomi 基准测试上，我们的方法达到了99.62%的目标词准确率。

    arXiv:2609.19805v1 Announce Type: new  Abstract: Grapheme-to-phoneme (G2P) conversion turns raw text into its phonemic form and is an essential part of both text-to-speech (TTS) and automatic speech recognition (ASR) systems. It is required to be fast, stable and context-aware. For unsegmented languages such as Japanese, G2P additionally couples word segmentation with highly context-dependent polyphone disambiguation, and the scarcity of accurately annotated data remains a bottleneck. In this paper, we present a context-aware neural G2P method that scores paths of a discriminative conditional random field (CRF) over a word lattice constructed from dictionaries. To tackle data scarcity, we utilize large language models (LLMs) to generate more than 2 million sentences. Experimental results demonstrate that our method strongly outperforms conventional morphological analyzer-based methods and neural sequence models. On the Joyo-Kanji-Yomi benchmark, our method reaches 99.62% target word re
    
[^242]: 全双工语音模型中工具调用的前后端架构

    A frontend-backend architecture for tool calls in full-duplex speech models

    [https://arxiv.org/abs/2609.19334](https://arxiv.org/abs/2609.19334)

    提出一种前后端架构，让全双工语音模型通过发出委派标记将流式转写交给文本LLM后端执行工具调用，并以轻量级注入机制返回结果，从而在几乎不修改前端模型的前提下保留低延迟、可打断的自然双工交互。

    

    全双工语音到语音（S2S）模型能够提供自然、低延迟的对话交互，若能具备使用外部工具并完成语音代理任务的能力将使其进一步受益。我们提出了一种前后端架构：由双工语音转文本前端学会发出一个委派标记，并将流式ASR转写文本转发给基于文本的后端大语言模型（LLM）以执行工具调用。后端的工具调用结果通过一个轻量级的预填充-重复机制注入回前端，再经流式TTS合成语音传达给用户。由于只需对前端模型进行极少的修改，我们的方法在很大程度上保留了常规的双工轮次切换、打断处理和低延迟交互。在单轮工具调用评估中，我们的系统实现了92-97%的工具调用召回率、具有竞争力的工具调用预测性能，以及81.2%的无关调用拒绝准确率。当配备更大的后端模型时……

    arXiv:2609.19334v1 Announce Type: new  Abstract: Full-duplex speech-to-speech (S2S) models provide natural, low-latency conversational interaction and would benefit from the ability to use external tools and complete voice-agent tasks. We propose a frontend-backend architecture where a duplex speech-to-text frontend learns to emit a delegation token and forwards streaming ASR transcripts to a text-based backend LLM for tool calls. Tool-call results from the backend are injected back into the frontend through a lightweight prefill-and-repeat mechanism and then synthesized using streaming TTS to the user. Our approach largely preserves regular duplex turn-taking, interruption handling, and low-latency interaction as it requires minimal modifications to the frontend model. In a single-turn tool-call evaluation, our system achieves 92-97% tool-call recall, competitive tool-call prediction performance, and 81.2% accuracy in rejecting irrelevant calls. When equipped with a larger backend (e.
    
[^243]: CovR：基于推理引导强化学习的覆盖率感知硬件验证

    CovR: Coverage-Aware Hardware Verification via Reasoning-Guided Reinforcement Learning

    [https://arxiv.org/abs/2609.19189](https://arxiv.org/abs/2609.19189)

    CovR是一个结合自我反思循环与仿真反馈的智能体框架，通过推理引导的强化学习自动生成硬件测试平台，突破了现有方法只关注功能正确性的局限，实现了验证覆盖率的最大化。

    

    设计验证仍然是硬件开发中资源消耗最大的阶段之一，通常消耗高达70%的总设计工作量。虽然最近的研究探索了使用大语言模型（LLM）来自动化生成测试平台，但大多数现有方法仅狭隘地关注功能正确性，忽略了覆盖率质量这一关键方面。为了弥合这一差距，我们提出了CovR，这是一个用于自动化测试平台生成的智能体框架，它将自我反思循环与基于仿真的反馈相结合，以最大化覆盖率。利用该流程，我们使用强大的教师模型构建了一个包含16,514个自然语言规范-RTL-推理-测试平台元组的大规模数据集，从而实现覆盖率感知的监督。在此基础上，我们提出了一种专为覆盖率驱动的测试平台生成而设计的强化学习（RL）框架，利用从仿真和覆盖率反馈中获得的工具奖励来优化学生模型

    arXiv:2609.19189v1 Announce Type: cross  Abstract: Design verification remains one of the most resource-intensive stages of hardware development, often consuming up to 70% of the total design effort. While recent work has explored using Large Language Models (LLMs) to automate testbench generation, most existing approaches focus narrowly on functional correctness, overlooking the critical aspect of coverage quality. To bridge this gap, we present CovR, an agentic framework for automated testbench generation that combines self-reflection loops with simulation-based feedback to maximize coverage. Using this pipeline, we construct a large-scale dataset of 16,514 natural specification RTL reasoning testbench tuples with a strong teacher model, enabling coverage-aware supervision. Building on this, we propose a reinforcement learning (RL) framework tailored for coverage-driven testbench generation, leveraging tool-derived rewards from simulation and coverage feedback to optimize a student m
    
[^244]: HearInContext：语音识别中隐式上下文的基准测试

    HearInContext: A Benchmark for Implicit Context in Speech Recognition

    [https://arxiv.org/abs/2609.18680](https://arxiv.org/abs/2609.18680)

    该论文提出了中英文同音词基准测试HearInContext用于评估语音识别模型的隐式与显式上下文利用能力，并通过微调Qwen3-ASR-1.7B将隐式上下文目标词召回率提升约11个百分点，同时不损害通用识别性能。

    

    情境化自动语音识别（ASR）可以受益于语义线索，或受益于上下文中明确提供的目标词。我们提出了HearInContext，这是一个中英文基准测试，它将共享的合成语音与支持不同解释的助手回复配对。该基准包含围绕同音词构建的3,764个语义测试用例。隐式上下文不包含候选词；显式上下文则点名目标词。无上下文和无关上下文的对照组用于衡量相关历史记录的益处以及对无关历史记录的敏感度。具备上下文能力的模型能从隐式线索中受益，但在有显式提示时能获得更高的目标词召回率。对Qwen3-ASR-1.7B进行微调后，中文和英文的隐式上下文目标词召回率分别提升了11.0和11.5个百分点，同时在AISHELL-1和LibriSpeech上的绝对CER/WER变化保持在0.1个百分点以下。收益还延伸到了微调中未包含的显式条件以及中文场景。

    arXiv:2609.18680v1 Announce Type: new  Abstract: Contextual ASR can benefit from semantic cues or from target words explicitly provided in the context. We introduce HearInContext, a Mandarin--English benchmark that pairs shared synthetic speech with assistant replies supporting different interpretations. The benchmark comprises 3,764 semantic test cases built around homophones. Implicit contexts exclude candidate words; explicit contexts name the target. No-context and unrelated-context controls measure the benefit of relevant history and sensitivity to irrelevant history. Context-capable models benefit from implicit cues but achieve higher target recall with explicit hints. Fine-tuning Qwen3-ASR-1.7B improves implicit-context target recall by 11.0 and 11.5 percentage points in Mandarin and English, respectively, while absolute CER/WER changes on AISHELL-1 and LibriSpeech remain below 0.1 percentage points. Gains extend to explicit conditions excluded from fine-tuning and to Mandarin h
    
[^245]: 谬误基准测试衡量的是论证图式识别，而非谬误检测

    Fallacy Benchmarks Measure Scheme Recognition, Not Fallacy Detection

    [https://arxiv.org/abs/2609.18644](https://arxiv.org/abs/2609.18644)

    该论文揭示了谬误检测基准报告的低误报率是“有效”类别构建方式的产物而非真实检测能力——当使用与谬误具有相同论证图式的正确论证作为负样本测试时，模型误报率大幅上升（CoCoLoFa上从16.6%升至58.9%），证明现有模型实际只是识别论证图式而非真正检测谬误。

    

    谬误检测基准通常将谬误类别与一个单一的“有效”或“无”类别配对，该类别包含了数据收集过程中未被标注为谬误的所有内容。这种构建方式具有误导性：分类器可以学习到某些线索从而在该类别上表现良好，却并未真正学会区分谬误与正确论证。我们证明，基准测试所报告的低误报率是类别构建方式的产物，而非检测能力的体现。对谬误而言，最有信息量的负样本是使用相同论证图式的正确论证，而在我们考察的四个基准中，此类论证在“有效”类别中最多只占几个百分点。在构建的图式匹配负样本上进行评估时，误报率在CoCoLoFa上从16.6%上升到58.9%，在Reddit上从5.7%上升到62.0%。由于误报率取决于负样本的撰写方式，我们还比较了来自同一流程、仅在论证图式身份上有所不同的两种条件。（摘要原文在此处被截断）

    arXiv:2609.18644v1 Announce Type: new  Abstract: Fallacy-detection benchmarks pair fallacy classes with a single "valid" or "none" class that takes everything data collection did not label as a fallacy. This construction is misleading: a classifier can learn cues that do well on this class without learning to tell a fallacy from a correct argument. We show that the low false-positive rates benchmarks report are an artifact of how the class is built, not evidence of detection ability. The most informative negative for a fallacy is a correct argument using the same argumentation scheme, and such arguments are at most a few percent of the valid class across the four benchmarks we examined. Evaluated on constructed scheme-matched negatives, false-positive rates rise from 16.6% to 58.9% on CoCoLoFa and from 5.7% to 62.0% on Reddit. That rate depends on how the negatives are written, so we also compare two conditions from the same pipeline that differ only in scheme identity. Classifiers lab
    
[^246]: Agora：以Git作为集体自动研究的共享内存

    Agora: Git as Shared Memory for Collective AutoResearch

    [https://arxiv.org/abs/2609.18094](https://arxiv.org/abs/2609.18094)

    Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。

    

    诸如AutoResearch之类的自主研究循环表明，单个编码智能体可以在无人值守的情况下改进训练设置。但如果同时运行多个这样的智能体，每个会话都会从零开始，因此更多的智能体往往意味着更多的重复搜索，而非更多的发现。Agora是这类智能体的共享内存：研究以仅追加的有向无环图（DAG）的形式记录在Git中，使得每一条主张都是一个任何人都可以检出并重新运行的提交。每个结果、见解、假设、验证和报告都是一个不可变的提交，其父边标明它建立在哪些工作之上；一个派生索引用于揭示研究前沿、被忽视的分支以及每条主张的验证状态，而一种多样性感知的选择规则可防止社区坍缩到单一领导者上。我们描述了该系统并报告了它的首次持续使用情况：一次持续近12天的运行，13个语言模型工作者在没有任务分配、没有中央规划者的情况下，针对一个权重转……

    arXiv:2609.18094v1 Announce Type: cross  Abstract: Autonomous research loops such as AutoResearch show that one coding agent can improve a training setup unattended. Run several of them and each session starts from scratch, so more agents tend to mean more duplicated search rather than more discovery. Agora is a shared memory for such agents: research is recorded as an append-only directed acyclic graph (DAG) stored in Git, so that every claim is a commit anyone can check out and rerun. Each result, insight, hypothesis, verification, and report is an immutable commit whose parent edges say what it builds on; a derived index exposes the frontier, the neglected branches, and the verification status of each claim, and a diversity-aware selection rule keeps the community from collapsing onto one leader. We describe the system and report its first sustained use: a run of nearly 12 days in which 13 language-model workers, with no assigned tasks and no central planner, worked on a weight-tran
    
[^247]: 时刻更聪明：环境驱动的动态策略助力大语言模型持续改进

    Smarter by the Moment: Environment-Driven Dynamic Policies for Continual LLM Improvement

    [https://arxiv.org/abs/2609.16800](https://arxiv.org/abs/2609.16800)

    提出了DRPG框架，将基于记忆的检索与动态策略生成器相结合，利用历史数据和环境反馈生成任务特定策略，从而实现大语言模型的持续改进，在六个基准和七个模型上超越了强大的基线方法。

    

    大语言模型（LLMs）已在众多领域取得显著进展，但对不断演变的任务和环境的持续适应仍然是一个关键挑战。现有的记忆增强方法仅检索单个过往样例作为直接参考，并未显式地从中综合出可执行的策略，导致同类错误反复出现。我们提出了动态检索式策略生成框架（DRPG），该框架将基于记忆的检索与动态策略生成器相结合，利用历史数据和环境反馈生成任务特定策略，以实现LLM的持续改进。我们在涵盖文本到SQL、问答、医疗诊断和Python编程的六个基准上，使用来自专有和开源权重系列的七个LLM对DRPG进行了评估。DRPG在大多数数据集和模型上均优于强大的基线方法。进一步分析表明，DRPG的策略生成……

    arXiv:2609.16800v1 Announce Type: new  Abstract: Large Language Models (LLMs) have achieved remarkable progress across diverse domains, but continual adaptation to evolving tasks and environments remains a key challenge. Existing memory-augmented approaches retrieve individual past examples as direct references, but do not explicitly synthesize actionable strategies from them, causing the same types of errors to recur. We propose Dynamic Retrieval-based Policy Generation (DRPG), a framework that integrates memory-based retrieval with a dynamic policy generator, leveraging historical data and environment feedback to produce task-specific policies for continual LLM improvement. We evaluate DRPG across six benchmarks spanning text-to-SQL, question answering, medical diagnosis, and Python programming, using seven LLMs from both proprietary and open-weight families. DRPG outperforms strong baselines across most datasets and models. Further analysis demonstrates that DRPG's policy generation
    
[^248]: RSIAgent：在新环境中实现递归自我改进的自主探索

    RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments

    [https://arxiv.org/abs/2609.15364](https://arxiv.org/abs/2609.15364)

    RSIAgent是一个无需训练的多智能体框架，通过“先广后深”的自主探索策略构建可复用的冻结记忆，使数字智能体在新环境中实现递归自我改进，且无需更新模型参数。

    

    数字智能体必须经常适应新的环境，而这些环境的界面、工具和失败模式并未被预训练模型完全涵盖。我们提出了RSIAgent，这是一个无需训练的多智能体框架，通过自主记忆构建实现递归自我改进。RSIAgent协调课程智能体、执行智能体和验证智能体，持续探索环境、验证结果，并保留环境特定的知识，包括行动、条件与后果之间可复用的因果关系。它进一步采用“先广后深”的探索策略，将并行的广泛递归自我探索（用于发现多样的环境结构）与聚焦的深度自我探索（用于发现困难案例、隐藏约束、边界条件以及此前未知的因果依赖）相结合。所构建的记忆被冻结后，可直接复用于下游任务，而无需更新模型参数。

    arXiv:2609.15364v1 Announce Type: new  Abstract: Digital agents must often adapt to new environments whose interfaces, tools, and failure modes are not fully captured by pretrained models. We introduce \textbf{RSIAgent}, a training-free multi-agent framework for recursive self-improvement through autonomous memory construction. RSIAgent coordinates curriculum, actor, and verifier agents to continually explore the environment, validate outcomes, and retain environment-specific knowledge, including reusable causal relationships between actions, conditions, and consequences. It further adopts a \textbf{broad-then-deep} exploration strategy, combining parallel broad recursive self-exploration for discovering diverse environment structures with focused deep self-exploration for uncovering hard cases, hidden constraints, boundary conditions, and previously unknown causal dependencies. The resulting memory is frozen and can be directly reused for downstream tasks without updating model parame
    
[^249]: 一个用于大语言模型针对性危害缓解的高效模块化框架

    An Efficient and Modular Framework for Targeted Harm Mitigation in LLMS

    [https://arxiv.org/abs/2609.13624](https://arxiv.org/abs/2609.13624)

    提出了一种结合Activated LoRA适配器与上下文感知路由机制的模块化纠正框架，可在生成过程中以低延迟、有针对性的方式缓解大语言模型的有害输出，同时提升模型对齐性能。

    

    摘要：大语言模型（LLMs）是强大的零样本学习器，但仍然容易与人类偏好产生不一致，经常输出带有偏见、有毒或其他有害的内容。现有的对齐方法虽然有效，但成本高昂且与模型紧密耦合，限制了灵活性和可扩展性。我们提出了一个模块化纠正框架，通过Activated LoRA（aLoRA）适配器和上下文感知路由机制来增强预训练的大语言模型，以消除模型失调响应带来的危害。我们的方法使专家适配器能够在序列中间激活而不使KV缓存失效，从而在生成过程中实现低延迟的针对性纠正。每个专家都被训练用于检测和缓解特定类型的危害，例如偏见或毒性。一个经过学习的路由器根据模型的中间输出动态选择合适的专家。我们证明该系统在标准安全基准测试中改善了对齐效果，同时保留了……

    arXiv:2609.13624v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are powerful zero-shot learners but remain prone to misalignment with human preferences, often producing biased, toxic, or otherwise harmful outputs. Existing alignment methods, while effective, are costly and tightly coupled to the model, limiting flexibility and scalability. We propose a modular correction framework that augments pretrained LLMs with Activated LoRA (aLoRA) adapters and a context-aware routing mechanism to eliminate harms from misaligned model responses. Our approach enables expert adapters to activate mid-sequence without invalidating the KV cache, allowing low-latency, targeted correction during generation. Each expert is trained to detect and mitigate specific harms, such as bias or toxicity. A learned router dynamically selects appropriate experts based on the models intermediate outputs. We demonstrate that our system improves alignment on standard safety benchmarks while preserving t
    
[^250]: 通过声学掩蔽量化辅音对单词可懂度的贡献

    Quantifying Consonant Contributions to Word Intelligibility via Acoustic Masking

    [https://arxiv.org/abs/2609.12122](https://arxiv.org/abs/2609.12122)

    本文提出一种基于声学掩蔽与语音识别模型的可扩展方法，通过掩蔽诱导误识别率（MMR）量化每个辅音对单词可懂度的贡献，从而帮助确定运动性言语障碍治疗的优先干预目标。

    

    辅音对单词能否被理解明白的贡献并不均等。考虑到治疗时间有限，按对可懂度的贡献为辅音排序有助于确定运动性言语障碍干预目标的优先次序。然而，测量这种贡献依赖于难以规模化的感知实验。本文提出了一种使用声学掩蔽来测量辅音贡献的可扩展方法：我们在一个孤立单词中逐次静音一个辅音，然后测试自动语音识别（ASR）模型是否仍能正确识别该单词。我们将辅音的贡献得分定义为其被掩蔽实例中单词被错误识别的比例，称之为掩蔽诱导误识别率。我们针对先前研究中报道的与辅音贡献相关的两个语言学因素（即音素频率和功能负荷）对MMR进行了验证，并将该分析应用于四种语（原文在此处截断）……

    arXiv:2609.12122v1 Announce Type: new  Abstract: Consonants contribute unequally to whether a word is understood. Given the limited time available for therapy, ranking consonants by contribution to intelligibility helps prioritize intervention targets in motor speech disorders. However, measuring this contribution relies on perceptual studies that are difficult to scale. This paper presents a scalable method that measures consonant contribution using acoustic masking. We silence one consonant at a time in an isolated word and test whether an automatic speech recognition (ASR) model still recognizes the word. We define a consonant's contribution score as the proportion of its masked instances for which the word becomes misrecognized, which we refer to as the mask-induced misrecognition rate (MMR). We validate MMR against two linguistic factors previously reported to correlate with consonant contribution, namely phoneme frequency and functional load. We apply this analysis across four la
    
[^251]: 从分数到证据：可审计的决策可以改进语音深伪检测

    From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection

    [https://arxiv.org/abs/2609.08899](https://arxiv.org/abs/2609.08899)

    该论文提出一种可审计的决策记录方法，将被动检测分数、条件性键控探针分数、检索支持和说话者画像边际四个线索纳入后期校准步骤，使语音深伪检测的最终决策在保持标量的同时保留证据来源信息，从而提升检测决策的可信度与可解释性。

    

    语音深伪能够以足以欺骗听众和自动化系统的方式逼真地模仿说话者的声音。这推动了语音深伪检测领域的强劲进展，但大多数检测器最终仍只输出每条语音的一个分数。该分数对于排序系统很有用，但对于为什么某个临界样本应该被信任、搁置还是复核，它几乎没有提供任何信息。两条语音可能因不同原因落入同一分数区间，例如被动证据与检索证据不一致，或者键控探针不可用。我们提出了一个问题：最终决策能否在保留这些来源信息的前提下仍然保持标量形式。我们通过一种可审计的决策记录来回答这个问题，该记录将四个对齐的线索引入后期校准步骤：被动检测器分数、在标记衍生样本上的条件性键控探针分数、检索支持度以及说话者画像边际，同时附带明确的分歧坐标。在包含4,080个样本的……

    arXiv:2609.08899v2 Announce Type: replace-cross  Abstract: Speech deepfakes can mimic a speaker's voice convincingly enough to deceive listeners and automated systems. This has driven strong progress in speech deepfake detection, but most detectors still end with one score per utterance. That score is useful for ranking systems, yet it says little about why a borderline item should be trusted, deferred, or reviewed. Two utterances can fall in the same score band for different reasons, for example because passive and retrieval evidence disagree or because the keyed probe is unavailable. We ask whether the final decision can remain scalar without discarding that provenance. We answer this question with an auditable decision record that carries four aligned cues into a late calibration step: a passive detector score, a conditional keyed-probe score on a marked derivative, retrieval support, and a speaker-profile margin, together with explicit disagreement coordinates. On the 4,080-example
    
[^252]: 注意差距：利用AlphaMWE多语言平行语料库揭示大语言模型翻译盲点

    Mind the Gap: Exposing LLM Translation Blind Spots Using the AlphaMWE Multilingual Parallel Corpus

    [https://arxiv.org/abs/2609.06634](https://arxiv.org/abs/2609.06634)

    本文通过WMT2026共享任务，使用AlphaMWE多语言平行语料库对31个机器翻译系统进行自动和人工评估，揭示比喻性多词表达仍是LLM翻译的瓶颈，且自动评估指标与人工评估时常存在分歧。

    

    大语言模型在机器翻译任务上的表现通常取决于其训练数据在特定领域和语言对上的可得性。为了考察多词表达是否仍然是LLM在语言理解和翻译方面的瓶颈，我们报告了WMT2026测试集共享任务中的系统表现，该任务使用公开可用的多语言平行语料库AlphaMWE作为测试集。我们收到了31个机器翻译系统的输出，涵盖英语到中文、波兰语、德语、阿拉伯语的翻译，其中阿拉伯语包括现代标准阿拉伯语以及埃及和突尼斯两种方言阿拉伯语。我们使用BLEU、ChrF、BERT-score进行自动评估，选出每个语言对的前三名系统，随后对所选系统进行人工评估。我们的研究结果表明：比喻性/多词表达现象仍然具有挑战性；自动评估指标有时会出现分歧；人工（评估）……（原文截断）

    arXiv:2609.06634v1 Announce Type: cross  Abstract: LLMs' performance on machine translation (MT) tasks is often dependent on the data availability in the specific domains and language pairs that they are trained upon. To examine if Multiword Expressions (MWEs) still set a bottleneck for LLMs regarding language understanding and translation, we report the system performances from the WMT2026 Test Suites shared task, for which we used the publicly available multilingual parallel corpus AlphaMWE as the test suites. We received 31 MT systems' outputs covering English to Chinese (zh), Polish (pl), German (de), Arabic (ar) including Modern Standard Arabic (MSA) and two dialectal ones (Egyptian and Tunisian Arabic). We carried out automatic evaluations using BLEU, ChrF, BERT-score to select the Top3 systems per language pair, followed up with human evaluations on the selected systems. Our findings show that: figurative/MWE phenomena remain challenging; automatic metrics sometimes disagree; hu
    
[^253]: VERPO：验证证据正则化策略优化

    VERPO: Verified Evidence Regularized Policy Optimization

    [https://arxiv.org/abs/2609.06100](https://arxiv.org/abs/2609.06100)

    VERPO提出了一种验证证据正则化策略优化框架，通过Fisher证据对比和带停止机制的逐token ZPD控制器有选择性地应用基于证据的修正，在保留可验证结果目标的同时避免盲目模仿导致的负面迁移。

    

    可验证的结果奖励可以指导语言模型的后训练，但序列级别的优势无法识别哪些token级别的决策应当被保留或修改。证据条件教师通过以特权反馈重放采样轨迹来提供更密集的监督。然而，不加区分的模仿可能会迁移那些不支持任务成功的格式或推理风格偏移。我们提出了VERPO，一个验证证据正则化策略优化框架，它将证据视为策略修正的提议，同时保留结果目标。该框架将无证据的参考恢复与带符号的token级证据修正分离开来。Fisher证据对比沿着估计的证据存在方向对修正进行衰减。一个带停止机制的逐token ZPD控制器根据局部奖励对齐程度和Fisher移动成本来调节修正的接受度，而参考通道保持独立于接受决策。

    arXiv:2609.06100v1 Announce Type: cross  Abstract: Verifiable outcome rewards guide language-model post-training, but sequence-level advantages do not identify which token-level decisions should be preserved or revised. Evidence-conditioned Teachers provide denser supervision by replaying sampled trajectories with privileged feedback. Yet indiscriminate imitation risks transferring formatting or reasoning-style shifts that do not support task success. We introduce VERPO, a Verified Evidence Regularized Policy Optimization framework that treats evidence as a proposal for policy correction while retaining the outcome objective. It separates evidence-free reference restoration from signed token-level evidence corrections. Fisher Evidence Contrast attenuates corrections along an estimated evidence-presence direction. A stopped token-wise ZPD controller scales acceptance according to local reward alignment and Fisher movement cost, while the reference channel remains independent of acceptan
    
[^254]: AhaBench：智能体能从先验经验中学习吗？一个面向长时程持续学习的基准测试

    AhaBench: Do Agents Learn from Prior Experience? A Benchmark for Long-Horizon Continual Learning

    [https://arxiv.org/abs/2609.05435](https://arxiv.org/abs/2609.05435)

    AhaBench 是一个长时程持续学习基准，通过谜题探索、带精确验证器的数学任务和自动售货机模拟三个组件，评估固定模型在获得先验经验后，在相关但支持已被移除、改变或延迟的条件下行为是否真正得到改善。

    

    arXiv:2609.05435v1 公告类型：新论文。摘要：现代语言智能体被期望在长时程上运行：它们会提出后续问题、复用已解决的示例、处理工具反馈，并适应延迟出现的后果。然而大多数评估仍然在每次提示后重置智能体，或仅对单条轨迹的最终状态进行评分。AhaBench 提出了一个更具操作性的问题：当一个固定模型获得有用的经验后，在一个明显的支持已被移除、改变或延迟的相关评估条件下，其后续行为是否会得到改善？该测试套件包含三个组成部分：Aha-Puzzle 测试智能体在解开隐藏状态谜题后的无提示探索能力；Aha-Euler 将类似 Project Euler 的数学思想转化为带有精确验证器的生成式教学/保留任务；Aha-Vending 是一个受 Vending-Bench 启发的开源实现，测试模拟的自动售货机智能体在处理延迟反馈和运营事故时能否保持盈利。AhaBench 报告了一个由三部分组成的评估结果

    arXiv:2609.05435v1 Announce Type: new  Abstract: Modern language agents are expected to operate over long horizons: they ask follow-up questions, reuse worked examples, handle tool feedback, and adapt to delayed consequences. Most evaluations still reset the agent after a prompt or score only the final state of one trajectory. AhaBench asks a more operational question: when a fixed model receives useful experience, does its later behavior improve under a related evaluation condition where the obvious support has been removed, changed, or delayed? The suite contains three components. Aha-Puzzle tests no-hint exploration after solved hidden-state puzzles; Aha-Euler turns Project-Euler-style mathematical ideas into generated taught/held-out tasks with exact validators; and Aha-Vending, an open-source implementation inspired by Vending-Bench, tests whether a simulated vending agent remains profitable while handling delayed feedback and operational incidents. AhaBench reports a three-part s
    
[^255]: 多语言语言模型中跨语言一致性增强方法的系统性评估

    A Systematic Evaluation of Cross-Lingual Consistency Enhancement Methods in Multilingual Language Models

    [https://arxiv.org/abs/2609.04409](https://arxiv.org/abs/2609.04409)

    本文对多语言模型中的跨语言一致性增强方法进行了统一的系统性评估，发现后训练方法（尤其是直接分布对齐）总体更可靠且能稳定提升一致性，而跨域迁移仅在源与目标任务输出格式相似时才有效。

    

    多语言语言模型在处理语义等价但表达于不同语言的问题时，常常产生不一致的答案，这促使研究者提出改进跨语言一致性（CLC）的方法。然而，现有方法通常在不同的模型、任务和评估协议下进行评估，导致其相对优势尚不明确。在本工作中，我们对问答任务中代表性的跨语言一致性增强方法进行了统一评估，涵盖推理时干预和后训练两类方法，涉及三个模型家族和三个封闭式基准。结果表明，后训练方法总体上更为可靠，其中直接分布对齐在所有模型-数据集组合中均能持续改进跨语言一致性，而其他方法则对答案格式和语言覆盖广度更为敏感。值得注意的是，除非源任务与目标任务具有相似的输出格式，否则跨域迁移的效果有限。我们进一步研究……

    arXiv:2609.04409v1 Announce Type: cross  Abstract: Multilingual language models often produce inconsistent answers to semantically equivalent questions across languages, motivating methods to improve cross-lingual consistency (CLC). However, existing methods are typically evaluated using different models, tasks, and protocols, leaving their relative strengths unclear. In this work, we present a unified evaluation of representative CLC-enhancement methods for question answering, spanning inference-time interventions and post-training approaches across three model families and three closed-form benchmarks. The results show that post-training methods are generally more reliable, with direct distribution alignment consistently improving CLC across all model-dataset combinations, while other methods are more sensitive to answer format and the breadth of language coverage. Notably, cross-domain transfer is limited unless source and target tasks share similar output formats. We further invest
    
[^256]: VoxReason：合成前基于源记录的语音规划的无听者评估

    VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis

    [https://arxiv.org/abs/2609.03203](https://arxiv.org/abs/2609.03203)

    VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。

    

    表现力语音系统在任何波形被渲染之前就必须做出一个决定：一句话语将以何种方式被表达。在对话智能体、旁白叙述和角色条件TTS中，这一隐藏的规划步骤决定了情感、音高、能量、语速、停顿、重音和立场，然而下游音频评分很少能揭示这些选择是否由源记录所支持——这是一种在任何波形存在之前就发生的源使用失败。VoxReason将这一合成前的决策转化为可度量的、无需听者参与的任务，用于评估基于源记录的语音规划。在合成之前，VoxReason衡量话语表达方式的选择是否有被引用的源记录作为依据。系统输出带有证据引用的、注明来源的说话计划，随后一个确定性验证器检查引用合法性、槽位一致性、无支持状态、模式有效性以及单线索反事实局部性。在1,440个经过检查的源标签案例上，捷径控制实验表明了为什么仅凭槽位准确率是不安全的：一个简单的键值查找……（原文摘要在此处截断）

    arXiv:2609.03203v1 Announce Type: cross  Abstract: Expressive speech systems make a decision before any waveform is rendered: how an utterance is delivered. In dialogue agents, narration, and role-conditioned TTS, that hidden planning step sets affect, pitch, energy, rate, pause, emphasis, and stance, yet downstream audio scores rarely reveal whether those choices were licensed by the source record, a source-use failure that occurs before any waveform exists. VoxReason makes that pre-synthesis decision measurable as a listener-free task for source-grounded speech planning. Before synthesis, VoxReason measures whether delivery choices are grounded in cited source records. Systems output a source-cited speaking-plan with evidence citations, and a deterministic verifier checks citation legality, slot agreement, unsupported state, schema validity, and one-cue counterfactual locality. On 1,440 checked source-label cases, shortcut controls show why slot accuracy alone is unsafe: a key-lookup
    
[^257]: 中期训练中的知识蒸馏更利于推理而非事实记忆

    Knowledge Distillation During Mid-Training Favors Reasoning over Factual Recall

    [https://arxiv.org/abs/2609.01532](https://arxiv.org/abs/2609.01532)

    该研究发现前向KL知识蒸馏在预训练阶段能同时提升推理与事实记忆能力，但在中期训练阶段会减缓事实记忆的习得而持续提升推理能力，这种阶段依赖性源于教师置信度在不同数据领域的不对称以及学生模型知识状态的演化。

    

    基于Logit的知识蒸馏（KD）通过更强教师模型的监督来训练更小的语言模型（LM），但其收益是否在各训练阶段保持一致仍不清楚。通过受控实验，我们发现采用后训练教师模型的前向Kullback-Leibler（KL）蒸馏——即标准的KD形式——在中期训练（即在精选语料上进行自监督学习的中间阶段）中表现出根本不同的行为。令人惊讶的是，在预训练阶段，相对于标准的下一词元预测（NTP），前向KD能同时提升推理能力与事实记忆能力；但在中期训练阶段，它却在推理能力持续提升的同时减缓了事实记忆的习得。我们将这种阶段依赖性归因于教师模型在不同数据领域上的置信度不对称，以及学生模型不断演化的知识状态：教师模型在程序性数据上比在知识密集型数据上更具信心，而学生模型……（原文摘要在此处截断）

    arXiv:2609.01532v1 Announce Type: new  Abstract: Logit-based knowledge distillation (KD) is used to train smaller language models (LMs) via supervision from stronger teachers, but whether its benefits are consistent across training stages remains unclear. Through controlled experiments, we find that forward Kullback-Leibler (KL) distillation--the standard KD formulation--with post-trained teachers behaves fundamentally differently during mid-training, an intermediate phase of self-supervised learning on curated corpora. Surprisingly, while forward KD simultaneously improves reasoning and factual recall during pre-training relative to standard next-token prediction (NTP), it instead slows factual recall acquisition during mid-training despite continued reasoning gains. We trace this stage dependence to an asymmetry in teacher confidence across data domains and the student's evolving knowledge state: teachers are more confident on procedural than knowledge-intensive data, while students 
    
[^258]: SCoNE：面向鲁棒检索增强生成的选择性上下文感知神经元编辑

    SCoNE: Selective Context-aware Neuron Editing for Robust Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.00689](https://arxiv.org/abs/2609.00689)

    SCoNE提出了一种无需训练的模型编辑方法，通过选择性强化兼具高归因分数与高跨输入变异性的上下文感知FFN神经元，显著提升大语言模型在检索增强生成中对检索噪声的鲁棒性，且无需微调、无推理开销。

    

    检索增强生成（RAG）对检索噪声高度敏感：当检索到的文档中混杂着有信息量和无关的内容时，大语言模型容易受到干扰，从而产生幻觉。为了解决这一问题，我们提出了SCoNE（选择性上下文感知神经元编辑），这是一种无需训练的模型编辑方法，通过选择性强化同时具有高归因分数和高跨输入变异性的上下文感知FFN神经元，来提升对检索噪声的鲁棒性。SCoNE仅需少量挖掘样本，无需微调，且不会带来推理时的额外开销。在多个知识密集型问答基准和两个大语言模型骨干上，SCoNE始终优于具有竞争力的基线方法。我们的代码可在 https://github.com/HYU-ARK-Lab/SCoNE 获取。

    arXiv:2609.00689v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) is highly sensitive to retrieval noise: when retrieved documents mix informative and irrelevant context, LLMs are easily distracted, leading to hallucinations. To overcome this, we propose SCoNE (Selective Context-aware Neuron Editing), a training-free model editing approach that improves retrieval noise robustness by selectively strengthening context-aware FFN neurons that are identified by both high attribution and high cross-input variability. SCoNE requires only a small number of mining samples, no fine-tuning, and no inference-time overhead. Across various knowledge-intensive question-answering benchmarks and two LLM backbones, SCoNE consistently outperforms competitive baseline methods. Our code is available at https://github.com/HYU-ARK-Lab/SCoNE.
    
[^259]: SingProbe 技术报告

    SingProbe Technical Report

    [https://arxiv.org/abs/2608.30703](https://arxiv.org/abs/2608.30703)

    SingProbe 是一种轻量级内嵌运行时防护机制，通过复用 LLM 推理的隐藏状态，在 token 级别以几乎零额外开销持续预测查询意图、响应安全性和幻觉风险，并配套提出流式防护基准 SingStreamBench。

    

    运行时防护机制对于大语言模型（LLM）的可靠部署至关重要，然而现有方法通常依赖于独立的、外部的模型，这会带来额外的推理开销、延迟的安全信号，以及与日益强大的基座模型之间的能力不匹配问题。为了解决这些问题，我们提出了 SingProbe，这是一种轻量级的内嵌运行时防护机制，它直接复用 LLM 推理过程中产生的隐藏状态，并与自回归解码并行运行。在统一框架下，SingProbe 能够在 token 级别持续预测查询意图、响应安全性和幻觉风险，且额外的防护推理开销几乎可以忽略不计，提供了一种“免费午餐”式的解决方案。我们进一步提出了 SingStreamBench，这是一个旨在评估流式防护机制在良性前缀上保持不激活、同时又能及时检测新出现的违规内容的能力的基准测试。大量实验表明，SingProbe 取得了具有竞争力的表现……

    arXiv:2608.30703v1 Announce Type: cross  Abstract: Runtime guardrails are essential for reliable large language model (LLM) deployment, yet existing approaches typically rely on independent, external models that introduce additional inference cost, delayed safety signals, and a capacity mismatch with increasingly capable base models. To address these issues, we introduce SingProbe, a lightweight intrinsic runtime guard that directly reuses hidden states produced during LLM inference and operates alongside autoregressive decoding. Within a unified framework, SingProbe continuously predicts query intent, response safety, and hallucination risk at the token level with negligible additional guardrail inference overhead, offering a "free-lunch" solution. We further introduce SingStreamBench, a benchmark designed to assess whether streaming guardrails remain inactive on benign prefixes while promptly detecting emerging unsafe content. Extensive experiments show that SingProbe achieves compet
    
[^260]: 超越并行盲目性：块草拟中的信息下限与模型差距

    Beyond Parallel Blindness: Information Floors and Model Gaps in Block Drafting

    [https://arxiv.org/abs/2608.27339](https://arxiv.org/abs/2608.27339)

    本文提出一种方法，通过信息下限和模型差距的分离，揭示了块草拟中并行生成的固有信息瓶颈，并指出当前草拟器仍有大幅改进空间。

    

    arXiv:2608.27339v1 公告类型：交叉 摘要：块草拟器在一次前向传播中提出多个令牌，而此时较早的目标令牌尚未实现。它们的拒绝混合了两种损失：缺失块内路径信息和可观察信息的不完美建模。接受长度无法区分这两种损失。我们通过一个信息下限将两者分开，该下限是在指定条件顺序下最小期望拒绝率；超过此下限的拒绝部分即为模型差距。我们通过四个领域的目标滚动、四个开源权重目标和一个前沿API目标来估计这两者，得出了三个发现。首先，在Qwen3-4B的最后一个槽位上，全并行下限达到$0.286$，这限制了即使是最佳提案也只能达到每槽位$71\%$的接受率。其次，一个已实现的令牌可以消除$86$--$100\%$的这个下限，这种局部性也通过独立的互信息分析得到了验证。第三，当前的草拟器仍远高于其下限：最终槽位的模型差距占DFlash拒绝的$43$--$64\%$。

    arXiv:2608.27339v1 Announce Type: cross  Abstract: Block drafters propose several tokens in one forward pass, before earlier target tokens are realised. Their rejection mixes two losses: missing within-block path information and imperfect modelling of observable information. Accepted length cannot distinguish them. We separate the two with an information floor, the minimum expected rejection at a specified conditioning order; rejection above this floor is the model gap. Estimating both from target rollouts across four domains, four open-weight targets, and a frontier API target yields three findings. First, the all-parallel floor reaches $0.286$ at the final slot on Qwen3-4B, limiting even the best proposal to $71\%$ per-slot acceptance. Second, one realised token removes $86$--$100\%$ of this floor, a locality also recovered by an independent mutual-information analysis. Third, current drafters remain far above their floors: the final-slot model gap accounts for $43$--$64\%$ of DFlash
    
[^261]: 先溯源后行文：声明锁定报告

    Provenance Before Prose: Claim-Locked Reporting

    [https://arxiv.org/abs/2608.25336](https://arxiv.org/abs/2608.25336)

    本文提出“声明锁定报告”协议，通过先固定结构化证据再生成文本，以解决LLM统计报告中数值漂移和效应方向反转问题，提高可复现性。

    

    大型语言模型（LLMs）能流利地表达统计证据，但统计报告仍可能出现数值漂移、效应方向反转，或将阈值对比误述为分类效应。我们将这些失败视为控制问题：科学报告中承载证据的内容应由结构化统计结果固定，而非在行文生成过程中采样。因此，我们利用跨运行可复现性来压力测试报告可见的数值和声明是否在行文生成前被绑定。现有控制措施在文本或槽位层面运作；一个确定性混合模板仅能跨种子重现61.1%的报告可见数值内容，因为LLM仍会选择模板渲染哪些发现和数值。我们提出声明锁定报告，这是一种先溯源后行文的协议，固定每个可报告声明的证据来源、数值、方向和允许的语言强度。

    arXiv:2608.25336v1 Announce Type: new  Abstract: Large language models (LLMs) can fluently verbalize statistical evidence, yet statistical reports can still drift numerical values, invert effect directions, or restate thresholded contrasts as categorical effects. We frame these failures as a control problem: the evidence-bearing content of a scientific report should be fixed by structured statistical results rather than sampled during prose generation. We therefore use cross-run reproducibility to stress-test whether report-visible numbers and claims are bound before prose generation. Existing controls operate at the text or slot level; a deterministic hybrid template reproduces only 61.1% of report-visible numerical content across seeds because the LLM still selects which findings and numbers the template renders. We propose claim-locked reporting, a provenance-before-prose protocol that fixes the evidence source, numbers, direction, and allowed language strength of each reportable cl
    
[^262]: Speech-to-SOAP：医疗对话的端到端摘要生成：KIT@BeTraC 2026

    Speech-to-SOAP: End-to-End Summarization of Medical Dialogues: KIT@BeTraC 2026

    [https://arxiv.org/abs/2608.24327](https://arxiv.org/abs/2608.24327)

    本文提出了Speech-to-SOAP系统，可直接从医疗对话语音端到端生成临床SOAP笔记而无需中间转录文本，并贡献了一个通过合成语音统一异构医疗对话数据集的可扩展数据增强流水线，用于参加BeTraC 2026轻量级赛道。

    

    随着大语言模型及其指令遵循能力的出现，摘要任务成为一个颇具前景的应用方向。在这一任务领域中，临床记录（clinical protocolling）这一抽取式子任务已成为备受关注的话题，因为它能够显著减少医护人员的停机时间和记录负担，使他们能够专注于帮助患者的核心工作。迈向自动化的更进一步是直接从语音生成临床笔记而无需中间转录文本，这不仅缩短了处理时间，还保留了诸如咳嗽声或其他副语言线索等信息，而这些信息在基于转录文本的系统中往往会丢失。为此，我们展示了KIT参加今年BeTraC挑战赛轻量级赛道的提交方案。我们的主要贡献是一个可扩展的数据增强流水线，该流水线通过合成语音生成和自动……统一了异构的医疗对话数据集。

    arXiv:2608.24327v2 Announce Type: replace  Abstract: With the advent of Large Language Models and its instruction following capabilities a promising application is the task of summarization. Within this domain of task the extractive sub-task of clinical protocolling has emerged as a topic of particular interest as it can significantly reduce the downtime and protocolling burden of health-care workers thus enabling them to focus on their core work helping humans. A further step towards automation is the direct generation of clinical notes from speech without intermediate transcripts, reducing processing time while preserving information such as coughing or other paralinguistic cues that may be lost in transcript-based systems. To this end, we present KIT's submission to this years BeTraC challenge in the lightweight track. Our main contribution is a scalable data augmentation pipeline that unifies heterogeneous medical dialogue datasets through synthetic speech generation and automatica
    
[^263]: 机器学习与数字语用学：哪种词类对表情符号使用影响最大？

    Machine learning and digital pragmatics: Which word category influences emoji use most?

    [https://arxiv.org/abs/2608.21975](https://arxiv.org/abs/2608.21975)

    本研究通过MARBERT模型和逻辑回归分析发现，在口语阿拉伯语社交媒体帖子中，动词类别对表情符号使用的影响最强，尽管名词在频率上占主导。

    

    arXiv:2608.21975v1 公告类型：新 摘要：本研究考察了最先进的MARBERT模型在数字语用学方法（DPA）框架内识别X平台上表情符号使用相关词汇/语用类别的表现。使用Python从X收集了包含表情符号的15856条口语阿拉伯语（CA）帖子作为净语料库。文本被分词并规范化为4个词汇类别，即名词_规范、动词_规范、形容词_规范和副词_规范，以及2个语用/结构类别，即疑问_规范和感叹_规范。MARBERT经过微调和优化，以识别哪个类别在标准指标上得分更高，从而与表情符号使用相关，同时使用二元逻辑回归来检验哪个类别在统计上与表情符号出现相关。研究结果显示，名词在规范频率上主导语料库（平均值=0.675，标准差=0.161），其次是动词（平均值=0.083，标准差=0.100）。然而，动词对表情符号使用的影响最强。

    arXiv:2608.21975v1 Announce Type: new  Abstract: This study examines the performance of the state-of-the-art MARBERT model in identifying the lexical/pragmatic category associated with emoji use on X within a digital pragmatics approach (DPA). A net corpus of 15856 Colloquial Arabic (CA) posts containing emojis was collected from X using Python. The texts were tokenized and normalized into 4 lexical categories, namely noun_norm, verb_norm, adj_norm, and adverb_norm, and 2 pragmatic/structural categories, question_norm and exclamation_norm. MARBERT was finetuned and optimized to identify which category scores standard metrics more, hence associated with emoji use, while binary logistic regression was used to examine which category is statistically associated with emoji occurrence. Findings unveil that nouns dominate the corpus in normalized frequency (M = 0.675, SD = 0.161), followed by verbs (M = 0.083, SD = 0.100). However, verbs have the strongest influence of emoji use indicated by 
    
[^264]: 关注之树：用于科学评论中未声明局限提取的分层多智能体辩论

    Tree-of-Concerns: Hierarchical Multi-Agent Debate for Unstated-Limitation Extraction in Scientific Critique

    [https://arxiv.org/abs/2608.20777](https://arxiv.org/abs/2608.20777)

    本文提出“关注之树”多智能体框架，通过专门怀疑论角色和小组审查机制，从科学论文中提取未声明局限，在精确度和覆盖率上分别比最强基线提升79%和11%。

    

    随着科学文献的增长和论文越来越多地少报局限性，多智能体大语言模型提供了一种有前景的方法来系统地揭示这些隐藏的失败模式。在此，我们引入了关注之树（Tree-of-Concerns），这是一个多智能体框架，它部署了专门的怀疑论者角色，每个角色通过特定类别的分析视角运作，作为并行的辩论树来从科学论文中提取未声明的局限性。每个角色进行结构化的、基于证据的论证，而一个小组审查机制从所有五个视角重新评估每个幸存的声明，以纠正类别漂移和严重性校准错误。通过在ToC-Bench上的实验——我们的基准包含414篇研究论文和1,905个未声明局限，这些来源于审稿人报告的弱点和后续引文批评——我们证明了相对于最强的基线，ToC将精确度提高了79%，覆盖率提高了11%，从而浮现出具体的、有证据支持的局限。

    arXiv:2608.20777v1 Announce Type: new  Abstract: As scientific literature grows and papers increasingly under-report limitations, multi-agent LLMs offer a promising approach to systematically uncover these hidden failure modes. Here, we introduce Tree-of-Concerns, a multi-agent framework that deploys specialized skeptic personas, each operating through a category-specific analytical lens, as parallel debate trees to extract unstated limitations from scientific papers. Each persona conducts structured, evidence-grounded argumentation, while a Panel Review mechanism re-evaluates each surviving claim from all five perspectives to correct category drift and severity miscalibration. Through experiments on ToC-Bench, our benchmark of 414 research papers with 1,905 unstated limitations, sourced from reviewer-reported weaknesses and follow-up citation critiques, we demonstrate that ToC improves precision by 79% and coverage by 11% relative to strongest baselines, surfacing specific, evidence-g
    
[^265]: LongNovel：长上下文小说摘要中幻觉检测的多尺度基准

    LongNovel: A Multi-Scale Benchmark for Hallucination Detection in Long-Context Novel Summarization

    [https://arxiv.org/abs/2608.18082](https://arxiv.org/abs/2608.18082)

    提出了LongNovel，一个多尺度双语长篇小说基准，用于检测长上下文摘要中的幻觉，并通过8种幻觉类型和组合生成方法确保数据真实性。

    

    尽管近年来上下文窗口显著扩大，但长上下文摘要中的幻觉现象仍是一个挑战。长篇小说因其固有的信息和事件、对话的详细描述，比新闻或论文更适合研究这些幻觉。然而，当前研究缺乏用于长上下文小说摘要中幻觉检测的多尺度基准，也未充分探索幻觉如何随着上下文变长而变化。在本研究中，我们提出了LongNovel，一个用于幻觉检测的多尺度长上下文双语（中文和英文）小说基准。该基准由29部中文小说（长度从16k到100k个词元）和BookSum数据集中的章节级数据构建而成。我们设计了8种幻觉类型，并采用多模型仲裁和实体引用幻觉生成的组合方法，以确保数据的真实性。

    arXiv:2608.18082v1 Announce Type: new  Abstract: Although context windows have expanded significantly in recent years, hallucinations in long-context summarization remain a challenge. Long novels are better suited than news or papers for researching these hallucinations, due to their intrinsic information and detailed descriptions of events and dialogues. However, current research lacks a multi-scale benchmark for hallucination detection in long-context novel summarization and does not fully explore how hallucinations change as the context grows longer. In this study, we propose LongNovel, a multi-scale long-context bilingual (Chinese and English) novel benchmark for hallucination detection. This benchmark is constructed from 29 Chinese novels (ranging from 16k to 100k tokens) and chapter-level data from the BookSum dataset. We design 8 hallucination types and employ a combination of Multi-Model Arbitration and Entity-Referenced Hallucination Generation to ensure both data authenticity
    
[^266]: 大型语言模型能否以具有法律意义的方式进行推理？一项关于欧洲人权法院案例的小规模研究

    Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases

    [https://arxiv.org/abs/2608.17168](https://arxiv.org/abs/2608.17168)

    本研究通过欧洲人权法院案例测试发现，顶尖大型语言模型在法律推理上表现不佳，其分析结构完整但内容浅薄，且自动评估器与人工评估一致性较弱。

    

    摘要：arXiv:2608.17168v1 公告类型：交叉 摘要：推理已成为当代大型语言模型的标准技术和特性；然而，在诸如法律案件预测等要求较高的法律导向任务中，其应用和质量仍未得到充分探索。我们以欧洲人权法院（ECtHR）的法律案例为测试平台，调查了大型语言模型在法律案件预测背景下的推理能力。我们评估了近期顶级模型OpenAI GPT 5.4，通过探索在ECtHR判例背景下对“具有法律意义的推理”更具或更少提示性的不同提示策略。我们通过人工和大型语言模型评估来呈现对模型响应的评估结果。我们发现，所考察的模型在法律推理方面远未达到理想分数，模型生成的分析结构完整但实质肤浅，并且“大型语言模型作为法官”的评估器内部一致但与我们的训练注释者仅有微弱一致性。

    arXiv:2608.17168v1 Announce Type: cross  Abstract: Reasoning has become a standard technique and feature for contemporary LLMs; however, its application and quality in the context of demanding legal-oriented tasks, such as legal case forecasting, remain under explored. We investigate how LLMs reason in the context of legal case forecasting, using legal cases from the European Court of Human Rights (ECtHR) as a testbed. We evaluate OpenAI GPT 5.4, a recent top-tier LLM, by exploring alternative prompting strategies that are more or less suggestive of what counts as legally meaningful reasoning in the context of ECtHR jurisprudence. We present our findings derived from assessing the model's responses with both human and LLM evaluation. We find that the examined model scores far from ideal in legal reasoning, the model produces structurally complete but substantively shallow analyses, and that LLM-as-a-Judge evaluators are internally consistent yet align only weakly with our trained annot
    
[^267]: 迈向更安全的RAG：只有具备系统2思考能力的代理才能访问不可信文档

    Towards Safer RAG: Only Agents Capable of System 2 Thinking may Access Untrusted Documents

    [https://arxiv.org/abs/2608.17153](https://arxiv.org/abs/2608.17153)

    本文提出一种新的安全原则，即仅允许具备系统2推理能力的代理访问不可信文档，以减少RAG系统中的知识投毒攻击影响，并引入新指标量化检测与影响间的差异。

    

    检索增强生成（RAG）显著提升了大型语言模型（LLMs）的性能，但这些系统仍然容易受到知识投毒攻击，即检索文档中的错误信息可能影响模型的最终输出。值得注意的是，LLM可能正确检测到文档包含错误信息，却仍受其影响。先前的研究通过“隔离原则”（Cordon Principle）解决了这一漏洞，该原则防止负责最终答案合成的模型直接访问原始证据。尽管有效，但这种严格隔离可能带来大量计算开销。在本工作中，我们提出了一种精细化的安全原则：只有具备深思熟虑的系统2推理能力的代理才能访问不可信文档。为评估这一原则，我们引入了新指标，用于量化错误信息检测与下游影响之间的差异。我们进行了实验...

    arXiv:2608.17153v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) has significantly enhanced the performance of large language models (LLMs), yet these systems remain vulnerable to knowledge-poisoning attacks, in which misinformation in retrieved documents can influence the model's final outputs. Notably, an LLM may correctly detect that a document contains incorrect information while nevertheless being influenced by it. Prior work has addressed this vulnerability through the Cordon Principle, which prevents models responsible for final answer synthesis from directly accessing raw evidence. Although effective, this strict isolation can introduce substantial computational overhead. In this work, we propose a refined security principle: only agents capable of deliberative System 2 reasoning may access untrusted documents. To evaluate this principle, we introduce novel metrics that quantify the discrepancy between misinformation detection and downstream influence. We t
    
[^268]: 计数文档并非计数文本：Web-PDF语料库统计中的单位偏差

    Counting Documents Is Not Counting Text: Unit Bias in Web-PDF Corpus Statistics

    [https://arxiv.org/abs/2608.16390](https://arxiv.org/abs/2608.16390)

    本文揭示了Web-PDF语料库中按文档计数与按令牌计数的巨大偏差，导致令牌总数被高估且截断文本大量丢失，影响语料库统计的准确性。

    

    arXiv:2608.16390v1 公告类型：交叉 摘要：PDF语料库以令牌数宣传其规模，但计算其发布的每个比率（覆盖率、OCR路由、重新获取恢复、语言混合）时均以文档为单位，且没有一个比率分解其令牌总数。这两种单位差异显著。在CC-MAIN-2021-31-PDF-UNTRUNCATED（790万份网页PDF，326亿令牌）中，3.02%的含文本文档占有一半的令牌（基尼系数0.807）；超过50页的文档占语料库的5.00%，但占其文本的53.53%。由TeX工具链生成的PDF占文档的1.66%，占文本的4.05%。最明显的受害者是Common Crawl的截断上限：它影响了23.06%的文档和63.08%的文本。重建被截断的文件并提取两个版本，两个广泛使用的库恢复了该文本的11.4%和1.4%；72%至97%的受影响文档未产生任何内容；语料库约55-62%的文本丢失。在2025年3月采用的5 MiB上限下，仍有30.19%的令牌会被截断，且恢复率...

    arXiv:2608.16390v1 Announce Type: cross  Abstract: PDF corpora advertise their size in tokens but compute every rate they publish (coverage, OCR routing, re-fetch recovery, language mix) per document, and none decomposes its token total. The two units diverge sharply. On CC-MAIN-2021-31-PDF-UNTRUNCATED (7.9M web PDFs, 32.6B tokens), 3.02% of text-bearing documents hold half the tokens (Gini 0.807); documents over 50 pages are 5.00% of the corpus but 53.53% of its text. The PDFs produced by a TeX{} toolchain are 1.66% of documents and 4.05% of the text. The clearest casualty is Common Crawl's truncation cap: it affected 23.06% of documents and 63.08% of the text. Reconstructing the truncated files and extracting both versions, two widely used libraries recover 11.4% and 1.4% of that text; between 72% and 97% of affected documents yield nothing; roughly 55--62% of the corpus's text is lost. Under the 5 MiB cap adopted in March 2025, 30.19% of tokens would still be truncated, and recovery
    
[^269]: HalluTracer：通过深度平均真值信号进行幻觉检测

    HalluTracer: Hallucination Detection via Depth-Averaging Truth Signals

    [https://arxiv.org/abs/2608.16353](https://arxiv.org/abs/2608.16353)

    HalluTracer通过聚合前向传播所有层的真值信号，利用弱相关的逐层证据进行深度平均，显著提升了幻觉检测的准确性。

    

    即使是对齐良好的大型语言模型也会自信地生成事实错误的文本，这使得幻觉成为高风险部署中持续存在的可靠性风险。然而，这些模型在其内部表示中携带线性可分离的真值信号。现有的白盒检测器将这些证据压缩到孤立组件或单一深度，丢弃了贯穿整个前向传播过程中分布的判别信息。我们引入了HalluTracer，这是一个检测框架，它在模型发出任何答案标记之前，读取并聚合前向传播每一层的真值证据。几何分析显示，逐层信号相关性较弱，因此简单的深度平均可以抑制层特定噪声，并捕获几乎所有线性可访问的信息。在六个开源语言模型和五个幻觉基准测试中，HalluTracer始终优于匹配的现有方法。

    arXiv:2608.16353v1 Announce Type: cross  Abstract: Even well-aligned large language models confidently generate factually incorrect text, making hallucination a persistent reliability risk in high-stakes deployments. These models nonetheless carry linearly separable truthfulness signals in their internal representations. Existing white-box detectors, however, collapse this evidence to isolated components or a single depth, discarding discriminative information distributed across the full forward pass. We introduce HalluTracer, a detection framework that reads and aggregates truthfulness evidence across every layer of the forward pass before the model emits any answer token. A geometric analysis reveals that the per-layer signals are weakly correlated, so that simple depth averaging suppresses layer-specific noise and captures nearly all linearly accessible information. Across six open-source language models and five hallucination benchmarks, HalluTracer consistently outperforms matched
    
[^270]: 条件验证：用于适应和监控安全分类器的正确性估计

    Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers

    [https://arxiv.org/abs/2608.14089](https://arxiv.org/abs/2608.14089)

    本文提出了一种轻量级包装器RCV，通过估计分类器预测与部署者策略不一致的概率并选择性纠正，同时利用正确性估计检测分布漂移，实现了无需重训练即可适应和监控安全分类器，显著提升策略遵循度。

    

    摘要：arXiv:2608.14089v1 公告类型：新  摘要：部署在大语言模型上的安全分类器通常因两个原因而失败：它们的决策反映了训练期间学习的策略，而非部署者期望的策略，并且随着部署流量的演变，其性能会下降。我们提出了条件验证（RCV），一种轻量级包装器，无需重新训练即可适应现成的安全分类器。RCV从分类器的内部表示中估计每个预测与部署者策略不一致的概率，并选择性地纠正可能错误的预测。相同的正确性估计还提供了用于检测分布漂移的无标签信号，从而启用一个维护循环，该循环更新正确性估计层，仅在必要时进行分类器微调。在三个现成的安全分类器和两个基准数据集上，RCV在每个类别中都提高了对部署者策略的遵循度。

    arXiv:2608.14089v1 Announce Type: new  Abstract: Safety classifiers deployed with large language models often fail for two reasons: their decisions reflect the policy learned during training rather than the deployer's desired policy, and their performance degrades as deployment traffic evolves. We present Regime-Conditional Verification (RCV), a lightweight wrapper that adapts an off-the-shelf safety classifier without retraining it. RCV estimates, from the classifier's internal representations, the probability that each prediction disagrees with the deployer's policy, and selectively corrects predictions likely to be wrong. The same correctness estimates also provide a label-free signal for detecting distribution shift, enabling a maintenance loop that updates the correctness estimation layer and resorts to classifier fine-tuning only when necessary. Across three off-the-shelf safety classifiers and two benchmark datasets, RCV improves adherence to the deployer's policy in every class
    
[^271]: CROP：通过反事实实现选择性在线策略蒸馏中的任务相关性

    CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation

    [https://arxiv.org/abs/2608.13387](https://arxiv.org/abs/2608.13387)

    CROP提出了一种基于释义校准的反事实敏感性边际方法，用于在选择性在线策略蒸馏中直接量化任务相关性，从而更有效地分配监督信号。

    

    arXiv:2608.13387v1 公告类型：新 摘要：在线策略蒸馏（OPD）在学生语言模型根据其当前策略采样的轨迹上进行监督，但对具有不同监督价值的响应标记赋予同等权重。选择性OPD通过根据估计的训练价值对响应标记进行非均匀分配监督来解决这一限制。然而，大多数现有标准主要关注优化需求，如不确定性或师生分歧，而任务相关性（即监督是否与当前输入的语义内容相关）作为补充维度仍未得到直接表征。为解决这一差距，我们引入了用于在线策略蒸馏的反事实相关性（CROP），通过释义校准的反事实敏感性边际来操作化任务相关性。对于每个源提示，CROP构建一个经过验证的原始-释义-反事实三元组，并保持学生滚动...

    arXiv:2608.13387v1 Announce Type: new  Abstract: On-policy distillation (OPD) supervises a student language model on trajectories sampled from its current policy, but assigns equal credit to response tokens with unequal supervision value. Selective OPD addresses this limitation by allocating supervision non-uniformly across response tokens according to their estimated training value. Most existing criteria, however, focus primarily on optimization need, such as uncertainty or teacher-student disagreement, while task relevance, namely whether the supervision is tied to the semantic content of the current input, remains less directly characterized as a complementary dimension. To address this gap, we introduce Counterfactual Relevance for On-Policy Distillation (CROP), which operationalizes task relevance through a paraphrase-calibrated counterfactual sensitivity margin. For each source prompt, CROP constructs a validated original-paraphrase-counterfactual triplet, holds the student roll
    
[^272]: 行为技能重构：从LLM智能体技能中重构隐藏功能

    Behavioral Skill Reconstruction: Reconstructing Hidden Functionality from LLM Agent Skills

    [https://arxiv.org/abs/2608.04192](https://arxiv.org/abs/2608.04192)

    该论文提出了一种名为SkillClone的黑盒攻击方法，攻击者仅通过正常使用技能并观察响应即可重构隐藏的LLM智能体技能的功能，表明单纯防止文件泄露不足以保护专有技能。

    

    封闭源代码的智能体技能可能包含专有的指令、脚本、常量和数据。提供商可能会以服务的形式提供其能力，同时保持底层软件包的隐藏。先前的工作主要集中于直接泄露这些内容的提示注入攻击，现有的防御措施也因此旨在防止此类泄露。然而，防止文件泄露并不能阻止用户恢复这些文件所实现的功能。这提出了一个根本性问题：在技能文件保持隐藏的情况下，用户能否通过正常使用来重构该技能的功能？我们研究了行为技能重构（BSR），即攻击者利用有效的任务请求和观察到的响应来构建隐藏技能的功能性克隆。我们提出了SkillClone，这是一种黑盒攻击，通过从技能的公开广告中形成接口假设、发出结构化的良性探测、合成……

    arXiv:2608.04192v2 Announce Type: replace-cross  Abstract: Closed source agent skills may encode proprietary instructions, scripts, constants, and data. Providers may offer their capabilities as services while keeping the underlying packages hidden. Prior work focuses on prompt injection attacks that directly disclose these artifacts, and existing defenses accordingly aim to prevent such leakage. However, preventing file disclosure does not prevent users from recovering the functionality those files implement. This raises a fundamental question: can a user reconstruct a skill's functionality through ordinary use while its files remain hidden?   We study behavioral skill reconstruction (BSR), in which an attacker uses valid task requests and observed responses to build a functional clone of a hidden skill. We introduce SkillClone, a black-box attack that clones a target skill by forming an interface hypothesis from its public advertisement, issuing structured benign probes, synthesizing
    
[^273]: 面向低资源语言的大语言模型：塔吉克语电子详解词典的概念框架

    Large Language Models for Low-Resource Languages: A Conceptual Framework for an Electronic Explanatory Dictionary of the Tajik Language

    [https://arxiv.org/abs/2608.04186](https://arxiv.org/abs/2608.04186)

    本文提出了一个利用大语言模型构建塔吉克语电子详解词典的概念框架，通过集成形态分析、词形还原、语义聚类和词典条目生成模块，并结合子词分词与参数高效微调策略，填补了低资源语言在数字词典学资源方面的空白。

    

    本文提出了一个使用大语言模型（LLMs）开发塔吉克语电子详解词典的概念框架。这项工作的意义源于塔吉克语缺乏一个在功能上可与高资源语言词典相媲美的综合性数字词典学资源，以及现代自然语言处理技术对低资源语言系统的适配有限。基于对现有语言学、统计学和语料库资源的系统调研，我们提出了一种词典架构，该架构集成了形态分析、词形还原、语义聚类以及利用大语言模型生成词典条目等模块。鉴于塔吉克语黏着语的形态特征及其高度形态变异性，我们论证了子词分词的选择依据，并采用了一种适合有限标注数据的参数高效微调（PEFT）策略。

    arXiv:2608.04186v3 Announce Type: replace  Abstract: This paper presents a conceptual framework for developing an electronic explanatory dictionary of the Tajik language using large language models (LLMs). The relevance of the work stems from the absence of a comprehensive digital lexicographic resource for Tajik that is comparable in functionality to dictionaries for high-resource languages, and from the limited adaptation of modern natural language processing technologies to low-resource language systems. Based on a systematic survey of existing linguistic, statistical, and corpus resources, we propose a dictionary architecture that integrates modules for morphological analysis, lemmatization, semantic clustering, and dictionary entry generation using LLMs. The choice of subword tokenization is justified by the agglutinative nature of Tajik morphology and its high morphological variability, along with a parameter-efficient fine-tuning (PEFT) strategy suitable for limited annotated da
    
[^274]: 应该生成谁？开放生成中人口统计目标的正当性论证

    Who Should Be Generated? Justifying Demographic Targets in Open-Ended Generation

    [https://arxiv.org/abs/2608.02551](https://arxiv.org/abs/2608.02551)

    该论文针对开放生成中人口属性未指明时的公平性评估，形式化了“缺失目标问题”，并将目标分布的构建分解为评估对象、先验可采性、分配方式与操作化四个承诺，为生成式审计中人口统计目标的选择提供了正当性论证框架。

    

    公平性评估不仅关注模型生成了什么，还关注其输出应当与什么进行比较。当模型生成“一位美国的CEO”时，提示词将人口统计特征的具体呈现留给了模型。现有的群体公平性定义假设敏感属性在输入端是给定的，而生成式审计则考察输出端的人口统计构成，但其用于比较的目标通常是直接给定的，而非经过论证的。上游的问题在于：目标分布应该是什么？我们针对人口属性未指明的生成任务形式化了这一“缺失目标问题”，并将目标构建分解为四个承诺：评估对象、先验可采性、分配方式与操作化。在此框架下，对于所声明的公共世界用途，我们在地理成员身份的解释下采纳地理先验。职业先验，在在任者……（原文摘要在此处截断）

    arXiv:2608.02551v2 Announce Type: replace-cross  Abstract: Fairness evaluation concerns not only what a model produces, but also what its outputs ought to be compared against. When a model generates "a CEO in the United States," the prompt leaves demographic realization to the model. Existing group fairness definitions assume that sensitive attributes are given on the input side. Generative audits instead examine output-side demographic composition, yet the targets they compare it against are typically supplied rather than justified. The upstream question is what the target distribution should be. We formalize this missing-target problem for demographic-value-unspecified generation and decompose target construction into four commitments: the evaluative object, prior admissibility, allocation, and operationalization. In this framework, we admit the geographic prior under a geographic-membership interpretation for the declared public-world use. The occupational prior, under an incumbency
    
[^275]: CANDOR：冻结基础编码器中的机会校准不一致性

    CANDOR: Chance-Calibrated Discordance in Frozen Foundation Encoders

    [https://arxiv.org/abs/2607.18451](https://arxiv.org/abs/2607.18451)

    本文提出CANDOR度量，通过等大小对称样本库校正最近邻不一致性，将机会水平固定为二分之一，揭示冻结编码器并非失明但普遍性能较弱。

    

    摘要：arXiv:2607.18451v2 公告类型：替换-交叉 摘要：冻结编码器的选择取决于轻量级头部从其特征中读取发现的能力，而非几何结构是否将其分离。最近邻不一致性可以做到这一点，但在样本库不均衡的情况下，相反标签的邻居会因密度而非几何结构获胜，因此仅凭患病率就会使无信息编码器看起来失明。我们引入了CANDOR，一种不一致性度量，其等大小样本库在标签交换下对称，将机会水平精确固定在二分之一。在22个编码器、来自7个领域的20个数据集和605,443张图像上，这一修正逆转了结论。崩溃几乎在所有地方都低于机会水平，因此没有编码器是失明的，但所有编码器都较弱：最佳胸部模型以84.5 AUROC读取气胸，但仍将18.4%的阳性样本放置在比同医院同类更接近相反标签的影像附近。同一个在鸟类物种分辨上达到4.5的编码器，在胸部发现上为42.8，在青光眼上为49.8，处于机会水平或更差。

    arXiv:2607.18451v2 Announce Type: replace-cross  Abstract: Frozen encoders are chosen by how well a lightweight head reads a finding from their features, not whether the geometry separates it. Nearest-neighbor discordance does, but with unequal banks the opposite-label neighbor wins on density, not geometry, so prevalence alone makes an uninformed encoder look blind. We introduce CANDOR, a discordance measure whose equal-size banks are symmetric under a label swap, fixing its chance level at exactly one half. Across 22 encoders, 20 datasets from 7 domains, and 605,443 images, this correction reverses the conclusion. Collapse falls below chance almost everywhere, so no encoder is blind, yet all are weak: the best chest model reads pneumothorax at 84.5 AUROC and still places 18.4% of those positives nearer an opposite-label film than its own kind in the same hospital. The same encoder that resolves bird species at 4.5 leaves chest findings at 42.8 and glaucoma at 49.8, at chance and wors
    
[^276]: 长度惩罚使思维链更难被监控

    Length Penalties Make Chain-of-Thought Less Monitorable

    [https://arxiv.org/abs/2607.09786](https://arxiv.org/abs/2607.09786)

    压缩思维链的长度惩罚虽能保持准确率并降低推理成本，但会显著降低思维链的忠实度，使模型更少表达误导性提示对其答案的影响，从而削弱了思维链的可监控性。

    

    近期的研究通过长度惩罚来训练推理模型，以抑制过度思考并降低推理成本。我们证明这些惩罚会使思维链的可监控性降低。经过长度压缩的模型仍然会让误导性提示左右其答案，但更少地将这种影响明确表达出来。我们使用强化学习在长度惩罚下训练 Qwen3-4B 和 Qwen3-14B，目标长度为基线思维链长度的 60% 至 30%，然后使用九种类型的偏向性提示，在留出的 MMLU-Pro-R 和四个迁移基准上对它们进行评估。当 LLM 监控器能够从思维链中判断出提示影响了答案时，该思维链就是忠实的。在 30% 的目标下，准确率保持在基线附近，且错误答案提示改变答案的频率与之前相同。然而，两个模型在所有评估集上的忠实度均出现下降，在 MMLU-Pro-R 上 Qwen3-14B 下降了 39%，Qwen3-4B 下降了 35%。而使用相同的正确性和格式奖励但不使用长度惩罚训练的对照组……

    arXiv:2607.09786v4 Announce Type: replace-cross  Abstract: Recent work trains reasoning models with length penalties to curb overthinking and cut inference cost. We show that these penalties make the chain of thought less monitorable. A length-compressed model still lets misleading hints steer its answers, but it less often verbalizes their influence. We train Qwen3-4B and Qwen3-14B with reinforcement learning under length penalties targeting 60% down to 30% of baseline chain-of-thought length, then evaluate them with nine types of biasing hints on held-out MMLU-Pro-R and four transfer benchmarks. A chain is faithful when an LLM monitor can tell from it that the hint influenced the answer. At the 30% target, accuracy stays near baseline and wrong-answer hints switch answers as often as before. Yet faithfulness drops on every evaluation set for both models, by 39% for Qwen3-14B and 35% for Qwen3-4B on MMLU-Pro-R. A control trained with the same correctness and format rewards but no leng
    
[^277]: 分布偏移下可靠长时程智能体上下文演化的范围化验证

    Scoped Verification for Reliable Long-Horizon Agentic Context Evolution under Distribution Shift

    [https://arxiv.org/abs/2607.09175](https://arxiv.org/abs/2607.09175)

    提出GRACE方法，将智能体持久指令维护为类型化语义图，通过在被修改节点的局部邻域内进行范围化验证，实现了分布偏移下长时程上下文演化的可靠增量更新。

    

    已部署的LLM智能体依赖于智能体上下文，即由操作框架组装的模型外部文本控制内容。在这项工作中，该上下文的可变组件是一个持久的系统级指令，它根据运行经验进行更新，而模型、工具和框架保持固定。在长演化周期中，随着累积指令的增长和相互作用的增加，平铺文本的维护方式使验证变得日益困难。我们提出了图正则化智能体上下文演化（GRACE），它将持久指令组件维护为类型化语义图，并在被修改节点的局部类型化邻域内验证提议的更新。被接受的图更新会被重构为对部署时使用的文本指令检查点的增量编辑。我们在由τ²-bench衍生的固定电信智能体框架内，在受控分布偏移协议下评估GRACE。在...

    arXiv:2607.09175v2 Announce Type: replace-cross  Abstract: Deployed LLM agents rely on agentic context, the model-external textual control content assembled by an operational harness. In this work, the mutable component of that context is a persistent system-level instruction that is updated from operational experience while the model, tools, and harness remain fixed. Over long evolution horizons, flat-text maintenance makes verification increasingly difficult as accumulated instructions grow and interact. We propose Graph-Regularized Agentic Context Evolution (GRACE), which maintains the persistent instruction component as a typed semantic graph and validates proposed updates within the local typed neighborhoods of modified nodes. Accepted graph updates are reconstructed as incremental edits to the textual instruction checkpoint used at deployment. We evaluate GRACE within a fixed telecom agent harness derived from $\tau^2$-bench under a controlled distribution-shift protocol. Across 
    
[^278]: 预训练语言模型嵌入的黎曼几何

    Riemannian Geometry for Pre-trained Language Model Embeddings

    [https://arxiv.org/abs/2607.07047](https://arxiv.org/abs/2607.07047)

    该论文提出黎曼平均池化（RMP）方法，通过从编码器雅可比矩阵提取词元拉回度量并在SPD流形上用Fréchet均值聚合，证明句子级分类信号存在于预训练语言模型嵌入的黎曼几何中，在多个具有语言结构的数据集上优于欧氏池化，且增益主要来自几何聚合机制本身。

    

    理解预训练语言模型嵌入的几何结构对于可解释性和安全性至关重要。我们探究句子级分类信号是否存在于上下文相关词元嵌入的黎曼几何之中，并通过以下方式加以验证：从学习到的编码器的解析雅可比矩阵中提取每个词元的拉回度量，再在对称正定（SPD）流形上用Fréchet均值进行聚合；我们将这一过程称为黎曼平均池化（RMP）。在三个具有非平凡语言结构的数据集（CoLA、CREAK、RTE）上，RMP优于欧氏平均池化；而在FEVER-Symmetric——一个为消除标注驱动的词汇伪影而构建的基准——上，该方法正确地保持在随机水平。消融实验表明，随机初始化的编码器结合Fréchet聚合在三个含信号数据集中的两个上已经胜过欧氏池化，从而将增益的来源定位于该几何聚合机制本身。

    arXiv:2607.07047v3 Announce Type: replace  Abstract: Understanding the geometric structure of pre-trained language model embeddings matters for interpretability and safety. We ask whether sentence-level classification signal lives in the Riemannian geometry of contextual token embeddings, and probe it by extracting per-token pullback metrics from a learned encoder's analytical Jacobian and aggregating them with the Fr\'echet mean on the symmetric positive definite (SPD) manifold; we call this procedure Riemannian Mean Pooling (RMP). Across three datasets with non-trivial linguistic structure (CoLA, CREAK, RTE), RMP outperforms Euclidean mean pooling, while on FEVER-Symmetric, a benchmark constructed to remove annotation-driven lexical artifacts, the method correctly stays at chance. Ablations show that a randomly initialised encoder combined with Fr\'echet aggregation already beats Euclidean pooling on two of the three signal-bearing datasets, localising the source of the gain to the g
    
[^279]: 幕后是谁？从德语Telegram帖子中标注与提取阴谋论行为者

    Who's Behind It? Annotating and Extracting Conspiratorial Actors from German Telegram Posts

    [https://arxiv.org/abs/2607.04962](https://arxiv.org/abs/2607.04962)

    该论文提出了阴谋论行为者的标注指南和德语Telegram帖子跨度标注语料库，并证明基于Transformer的模型能够以合理的准确率自动提取阴谋论行为者，从而支持对阴谋论叙事中行为者表征的大规模分析。

    

    阴谋论通常将重要事件归因于强大而隐秘的行为者的行动。尽管计算研究主要集中于阴谋论的文档级分析，但对识别推动此类叙事的行为者关注较少。我们为阴谋论行为者制定了标注指南，提出了一个经过跨度标注的德语Telegram帖子语料库，并研究了使用基于Transformer的模型对其进行自动提取。我们进一步将所得模型应用于Schwurbelarchiv——一个大规模的德语阴谋论相关Telegram频道档案库。我们的结果表明，尽管阴谋论话语在语言上具有复杂性，阴谋论行为者仍可以在具有意义的一致性水平下进行标注，并以合理的准确率进行提取，从而支持对阴谋论叙事中行为者表征的大规模分析。

    arXiv:2607.04962v2 Announce Type: replace  Abstract: Conspiracy theories commonly attribute important events to the actions of powerful and secretive actors. While computational research has largely focused on document-level analyses of conspiracy theories, less attention has been paid to identifying the actors that drive such narratives. We develop annotation guidelines for conspiratorial actors, present a span-annotated corpus of German Telegram posts, and investigate their automatic extraction using transformer-based models. We further apply the resulting model to the \textit{Schwurbelarchiv}, a large-scale archive of German conspiracy-related Telegram channels. Our results demonstrate that conspiratorial actors can be annotated with meaningful agreement and extracted with reasonable accuracy despite the linguistic complexity of conspiracy discourse, enabling large-scale analyses of actor representations in conspiracy narratives.
    
[^280]: 由你来定义框架：概念表征如何塑造大语言模型对反犹主义的检测与推理

    You Frame It: How Conceptual Representations Shape LLM Detection and Reasoning about Antisemitism

    [https://arxiv.org/abs/2607.04945](https://arxiv.org/abs/2607.04945)

    本研究通过对比四种反犹主义概念表征形式，发现细粒度分类表征能显著提升大语言模型检测的召回率但会牺牲精确率，而更大的概念资源并无额外收益，大屠杀后反犹主义始终是最难检测的类型。

    

    大语言模型（LLM）能够在推理阶段整合外部概念资源，为检测反犹主义这类在意识形态和历史层面都十分复杂的现象创造了新的机会。我们研究了不同形式的概念基础（conceptual grounding）如何影响四个最先进的大语言模型在反犹主义检测与解释行为上的表现。基于两个由专家标注的数据集，我们比较了定义式、细粒度分类式、示例增强式以及大上下文式等不同形式的反犹主义概念表征。研究发现，细粒度分类表征能够显著提升召回率，但同时会降低精确率。令人惊讶的是，提供规模大得多的概念资源并未带来额外的量化收益。大屠杀后的反犹主义在所有模型和配置中都是最顽固的检测难题。对模型解释的分析进一步揭示了系统性局限，包括概念引用的过度生成等（摘要原文在此处截断）。

    arXiv:2607.04945v2 Announce Type: replace  Abstract: LLMs enable the integration of external conceptual resources at inference time, creating new opportunities for detecting ideologically and historically complex phenomena such as antisemitism. We investigate how different forms of conceptual grounding affect antisemitism detection and explanation behavior across four state-of-the-art LLMs. Using two expert-annotated datasets, we compare definitional, fine-grained taxonomic, example-augmented, and large-context representations of antisemitism.   We find that fine-grained taxonomic representations substantially improve recall, while simultaneously reducing precision. Surprisingly, supplying substantially larger conceptual resources yields no additional quantitative benefit. Post-Holocaust antisemitism poses the most persistent challenge across models and configurations. Analysis of explanations further reveals systematic limitations including overproduction of conceptual references, rel
    
[^281]: LP-SFT：基于多模态熵结构的局部保持监督微调

    LP-SFT: Local-Preserving Supervised Fine-Tuning via Multimodal Entropy Structure

    [https://arxiv.org/abs/2607.04733](https://arxiv.org/abs/2607.04733)

    提出LP-SFT，一种基于多模态熵结构分析的局部保持监督微调方法，在将模型适配到下游领域的同时，保留预训练模型已有的能力和丰富的分布知识。

    

    监督微调（SFT）是将预训练语言模型适配到下游领域的标准方法，但它往往在改善目标领域表现的同时，以损害模型已有能力为代价。标准的交叉熵微调仅提升观测到的标签token，而对概率质量如何重新分配到其他合理候选上没有任何约束，这可能扭曲预训练期间学到的丰富的局部偏好结构。我们首先使用香农熵和Renyi熵对下一token预测进行分析，发现预训练模型呈现出规律性的多模态熵结构。这些熵峰对应于数量不等的合理候选，表明基础模型内在地编码了超越单一监督token的丰富分布知识。受此观察启发，我们提出LP-SFT，一种局部保持监督微调目标，旨在显式（摘要在此处截断）

    arXiv:2607.04733v3 Announce Type: replace  Abstract: Supervised fine-tuning (SFT) is the standard approach for adapting pretrained language models to downstream domains, yet it often improves target-domain behavior at the cost of degrading pre-existing capabilities. Standard cross-entropy fine-tuning promotes only the observed label token and leaves unconstrained how probability mass is redistributed over other plausible alternatives, potentially distorting the rich local preference structure learned during pretraining. We first analyze next-token predictions using Shannon and Renyi entropies, revealing that pretrained models exhibit a regular multimodal entropy structure. These entropy peaks correspond to varying numbers of plausible alternatives, indicating that the base model intrinsically encodes rich distributional knowledge beyond the single supervised token. Motivated by this observation, we propose LP-SFT, a Local-Preserving Supervised Fine-Tuning objective designed to explicit
    
[^282]: 重新思考语音-大语言模型集成在语音识别中的应用：通过交错方法实现有效的语音-文本联合训练

    Rethinking Speech-LLM Integration for ASR: Effective Joint Speech-Text Training by Interleaving

    [https://arxiv.org/abs/2607.01733](https://arxiv.org/abs/2607.01733)

    提出联合语音-文本交错预训练策略JSTIP，通过在语音-LLM中构建词级和段级交错的语音-文本序列，有效利用文本知识提升ASR实体识别准确率并简化领域适配。

    

    语音-大语言模型（Speech-LLM）集成通过利用大规模文本预训练已展现出可喜的成果，但其对自动语音识别（ASR）的具体收益仍不明确。我们观察到，随着有监督ASR训练数据的增加，LLM先验的贡献变得不那么明显，而简单的语音-文本联合训练未能充分利用文本知识。因此，我们提出了联合语音-文本交错预训练（JSTIP），这是一种面向ASR的预训练策略，在接受连续输入的语音-LLM架构中，于对齐语音-文本对内构建词级和段级交错的语音-文本序列。在38k小时ASR数据上的实验表明，与仅使用ASR以及语音-文本联合训练的基线相比，该方法持续提升了实体识别准确率。JSTIP使用领域转录文本即可达到与合成语音-文本对相当的实体识别性能，从而简化了领域适配。

    arXiv:2607.01733v2 Announce Type: replace  Abstract: Speech-LLM integration has shown promising results by leveraging extensive textual pretraining, yet its specific benefits for automatic speech recognition (ASR) remain unclear. We observe that as supervised ASR training data increases, the contribution of LLM priors becomes less evident, and simple speech-text joint training under-utilizes textual knowledge. We therefore propose Joint Speech-Text Interleaved Pretraining (JSTIP), an ASR-oriented pretraining strategy that constructs word-level and segment-level interleaved speech-text sequences within aligned pairs for speech-LLM architectures that accept continuous inputs. Experiments on 38k hours of ASR data show consistent entity accuracy improvement compared to ASR-only and joint speech-text training baselines. JSTIP achieves on-par entity recognition performance using domain transcription text compared to synthetic speech-text pairs, simplifying domain adaptation. Benefiting from 
    
[^283]: 大语言模型如何引用？检索增强生成中归因机制的机械可解释性研究

    How Do LLMs Cite? A Mechanistic Interpretation of Attribution in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2606.28358](https://arxiv.org/abs/2606.28358)

    本文首次对RAG中大语言模型的行内引用决策机制进行机械可解释性分析，通过激活修补技术发现引用行为并非由单一局部组件实现，而是由注意力头和MLP层构成的分布式、多阶段“归因集合”协同完成。

    

    检索增强生成（RAG）旨在通过将大语言模型（LLM）的输出锚定在外部文档上来增强其可信度，并通常使用行内引用以保证可验证性。然而，这些引用的忠实性——即模型是否真正使用了某个来源来生成答案——仍然是一个关键的、未经证实的假设。本文首次从机制层面解释了大语言模型在回答事实性问题时如何决定是否附加行内引用。我们基于PopQA数据集构建了受控实验环境，使用Llama-3.1-8B-Instruct模型并采用激活修补方法。我们绘制了负责引用行为的底层机制，发现它并非单一的局部组件，而是由注意力头和MLP层组成的分布式、多阶段的“归因集合”。我们证明，仅放大或衰减那些关键的注意力头……

    arXiv:2606.28358v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) aims to enhance the trustworthiness of Large Language Models (LLMs) by grounding their outputs in external documents, often using inline citations for verifiability. However, the faithfulness of these citations -- whether the model genuinely uses a source to generate an answer -- remains a critical, unverified assumption. This paper offers the first mechanistic account of how a large language model decides whether to attach an inline citation while answering a factoid question. Using the Llama-3.1-8B-Instruct model in a controlled experimental environment based on the PopQA dataset, we employ an activation patching approach. We map the underlying mechanism responsible for citation, discovering that it is not a single, localized component but a distributed, multi-stage "attributional ensemble" of attention heads and MLP layers. We show that amplifying or attenuating only those critical heads 
    
[^284]: 大语言模型推理强化学习更新的关键因素是什么？

    What are Key Factors for Updates in RL for LLM Reasoning?

    [https://arxiv.org/abs/2606.22570](https://arxiv.org/abs/2606.22570)

    该论文通过理论分析揭示离策略程度（每次rollout的梯度步数）会通过影响重要性采样比率的分布与裁剪行为决定哪些token主导RLVR更新，并据此提出自适应裁剪策略优化方法来改进大模型推理的强化学习训练。

    

    可验证奖励强化学习已成为提升大语言模型推理能力的一种有前景的框架。然而，现有工作大多由启发式直觉引导，导致了各异的算法选择，甚至是相互矛盾的选择，但它们却都能报告出经验上的收益。为了更好地理解这一现象，我们对RLVR的更新过程进行了理论分析。我们的研究揭示：由每次rollout的梯度步数所决定的离策略程度差异，会显著影响重要性采样比率的分布及其裁剪行为，从而改变哪些token在更新中占主导地位。基于这一洞见，我们将梯度期望刻画为控制更新动态的核心量，并分析了token概率、优势值和重要性采样比率各自的作用。受这些发现启发，我们提出了自适应裁剪策略优化（Adaptive Clip Policy Optimization）方法。

    arXiv:2606.22570v2 Announce Type: replace  Abstract: Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a promising framework for enhancing the reasoning ability of large language models. However, much of the existing work is guided by heuristic intuition, leading to divergent algorithmic choices, even contradictory ones that nevertheless report empirical gains. To better understand this phenomenon, we conduct a theoretical analysis of RLVR updates. Our study reveals that differences in off-policy degree, determined by the number of gradient steps per rollout, substantially affect the distribution of importance sampling ratios and their clipping behavior, thereby altering which tokens dominate the update. Building on this insight, we characterize gradient expectation as the central quantity governing update dynamics and analyze the roles of token probability, advantage, and importance sampling ratio. Motivated by these findings, we propose Adaptive Clip Policy Optimiz
    
[^285]: 知识图谱接地仅对训练集之外的知识有帮助：一项关于临床问答的对照研究

    Knowledge-Graph Grounding Helps LLMs Only for Out-of-Training Knowledge: A Controlled Study on Clinical Question Answering

    [https://arxiv.org/abs/2606.22419](https://arxiv.org/abs/2606.22419)

    本文通过临床问答对照研究发现，基于公开生物医学知识图谱 PrimeKG 的知识图谱接地（无论朴素三元组检索还是代理式 Cypher 查询）均无法提升从弱到强各档大语言模型在 MedQA 上的表现，表明知识图谱接地仅在所需知识超出模型训练范围时才有帮助，同时复现并修正了《自然·医学》研究中 HealthBench 的评分问题。

    

    近期一项发表于《自然·医学》的研究报告称，通用前沿大语言模型（LLM）在医学基准测试上的表现优于专门的检索增强临床工具，且检索可能会损害强模型的表现。我们提出一个自然的后续问题：结构化知识图谱（KG）接地是否会改变这一结论？接地究竟在什么情况下才有帮助？我们贡献了两项结果。第一，一项复现研究：该研究的头条 HealthBench 分数（约88分）实际上是 Consensus 变体的分数，而非完整 HealthBench 的分数——在经医生校准的评分器下（评分一致性为82.5%），前沿模型与理想回答的得分均约为46-47分；我们复现出 GPT-5.2 Consensus 得分为90.9，并指出了一个会压低分数的评分器缺陷。第二，一项关于知识边界的结果。使用基于公开生物医学知识图谱 PrimeKG 的图+向量引擎（samyama-graph），无论是朴素的三元组检索，还是代理式的自然语言转 Cypher 查询循环（查询成功率为82%），在一个从弱到强的模型阶梯上均未能提升 MedQA 的表现（所有 |Delta……（原文摘要在此处被截断）

    arXiv:2606.22419v3 Announce Type: replace  Abstract: A recent Nature Medicine study reports that general-purpose frontier LLMs outperform specialized retrieval-augmented clinical tools on medical benchmarks, and that retrieval can hurt strong models. We ask the natural follow-up: does structured knowledge-graph (KG) grounding change this, and when does grounding help at all? We contribute two results. First, a reproduction: the study's headline HealthBench score (~88) is the Consensus variant, not full HealthBench, where frontier models and ideal completions both score ~46-47 under a physician-calibrated grader (agreement 82.5%); we reproduce GPT-5.2 Consensus =90.9 and flag a score-deflating grader bug. Second, a knowledge-boundary result. Using a graph+vector engine (samyama-graph) over the public biomedical KG PrimeKG, neither naive triple retrieval nor an agentic natural-language-to-Cypher loop (82% successful queries) improves MedQA across a weak-to-strong model ladder (all |Delta
    
[^286]: 元名游戏：一个无真实基准的LLM基准，随其测量的模型一同提升

    The Metanym Game: An LLM Benchmark Without Ground Truth That Rises With the Models It Measures

    [https://arxiv.org/abs/2606.21008](https://arxiv.org/abs/2606.21008)

    该论文提出一种无真实基准的LLM评估方法，通过类比生成与相互评分，利用SVD特征方程统一评判生成与评判能力，并发现与GPQA Diamond存在相关性。

    

    arXiv:2606.21008v3 公告类型：替换交叉 摘要：我们提出证据表明，类比是LLM智能的核心。在我们的基准测试中，LLMs竞争生成一组组类比陈述，并根据各自对事实正确性、美感、智能性、独特性、长度和结构多样性的理解来相互评分。外部信息不进入：唯一给定的是游戏规则；每个项目都在游戏中生成；分数仅来自玩家的评分。真实基准被事实评分矩阵的奇异值分解（SVD）所取代，该矩阵同时将玩家评为生成者和评判者——据我们所知，这是首个评判LLM同行委员会中评判者的特征方程。对于美感等主观标准，评判者按其评分一致性加权。最佳生成者结果是中等评判者。GPQA Diamond——由人类专家编写的困难选择题——在方法上截然不同，但这两个基准却相关。

    arXiv:2606.21008v3 Announce Type: replace-cross  Abstract: We present evidence that analogy is at the core of LLM intelligence. In our benchmark, LLMs compete in generating sets of analogous statements and rate each other's sets on their own understandings of factual correctness, beauty, intelligence, distinctness, length, and structural diversity. Nothing enters from outside: the only given is the game rules; every item is generated in play; the scores come from the players' ratings alone. Ground truth is replaced by the SVD of the factual rating matrix, which scores players as generators and judges at once -- to our knowledge the first eigen-equation that judges the judges for an LLM council-of-peers. For subjective criteria like beauty, judges are weighted by their rating consistency. The best generators turn out to be middling judges. GPQA Diamond -- difficult multiple-choice questions written by human experts -- could not be more different in method, yet the two benchmarks correla
    
[^287]: 连点成线：通过强化学习训练具备跨领域泛化能力的大语言模型长生命周期智能体

    Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning

    [https://arxiv.org/abs/2606.20002](https://arxiv.org/abs/2606.20002)

    本文提出“连点成线”通用框架，通过端到端强化学习训练大语言模型在长生命周期智能体场景中持续探索环境、自我更新上下文并实现跨领域泛化，从而在后续任务上获得渐进式性能提升。

    

    本工作提出了一个通用框架，用于训练大语言模型（LLM）掌握“连点成线”能力——这是长生命周期智能体所需的一种元能力：当基于LLM的AI智能体被部署到某个环境中后，它会解决一长序列的任务，同时持续探索环境、从自身经验中学习，并迭代式地自我更新关于环境的上下文，从而基于更新后的上下文，在未来任务上取得逐步提升的性能。CoD框架的主要组成部分包括：（1）用于端到端强化学习（RL）的算法设计与基础设施，支持交错的“解决任务”与“更新上下文”片段的长序列推演；（2）用于在训练中激励和引导LLM掌握目标元能力、并在评估中忠实衡量进展的任务与环境。我们展示了CoD框架的概念验证实现……

    arXiv:2606.20002v2 Announce Type: replace-cross  Abstract: This work presents a general framework for training large language models (LLMs) to "Connect the Dots" (CoD), a meta-capability required by long-lifecycle agents: as an LLM-based AI agent gets deployed in an environment, it solves a long sequence of tasks while continuously exploring the environment, learning from its own experiences, and iteratively self-updating its context about the environment, thereby achieving progressively better performance on future tasks conditioned on the updated context. Major components of the CoD framework include: (1) algorithm design and infrastructure for end-to-end reinforcement learning (RL) with long rollout sequences interleaving solve-task and update-context episodes; (2) tasks and environments for incentivizing and eliciting the targeted meta-capability in LLMs during training, as well as for faithfully measuring progress during evaluation. We present proof-of-concept implementations of t
    
[^288]: 想要更好的合成数据？引导它：面向低资源语言生成的激活引导技术

    Want Better Synthetic Data? Steer It: Activation Steering for Low-Resource Language Generation

    [https://arxiv.org/abs/2606.18389](https://arxiv.org/abs/2606.18389)

    本文提出用激活引导（语言引导与质量引导）替代少样本提示来生成低资源语言合成数据，在11种类型多样的语言上验证了该方法能以更低成本生成更高质量、更多样化的训练数据。

    

    大语言模型（LLMs）已成为合成数据生成的有效工具，包括用于低资源语言的合成数据生成，其中生成的数据可以提升下游任务的性能。目前表现最好的方法通常依赖于使用目标语言示例的少样本提示，这会增加推理成本，并可能因词汇锚定而降低数据多样性。在本工作中，我们研究了激活引导作为低资源合成数据生成的替代方案。我们研究了两种引导策略：语言引导，针对一种语言的语言学身份；以及质量引导，通过对比人类书写文本与回译文本的表示来捕捉文本的规范程度。我们在四个开源大语言模型、多个层以及11种类型多样的语言上评估了这些方法，通过生成情感和主题分类数据并微调较小的分类器来进行验证。引导被应用于……（摘要截断）

    arXiv:2606.18389v2 Announce Type: replace  Abstract: Large language models (LLMs) have become an effective tool for synthetic data generation, including for low-resource languages, where generated data can improve downstream task performance. Current best-performing approaches typically rely on few-shot prompting with target-language examples, which increases inference costs and may reduce diversity through lexical anchoring. In this work, we investigate activation steering as an alternative for low-resource synthetic data generation. We study two steering strategies: Language Steering, which targets the linguistic identity of a language, and Quality Steering, which captures well-formedness by contrasting human-written and backtranslated text representations. We evaluate these methods across four open-source LLMs, multiple layers, and 11 typologically diverse languages by generating sentiment and topic classification data and finetuning smaller classifiers. Steering is applied in both 
    
[^289]: Anthropic Fable 5 与 Opus 4.8 模型的红队研究

    A Red-Team Study of Anthropic Fable 5 & Opus 4.8 Models

    [https://arxiv.org/abs/2606.18193](https://arxiv.org/abs/2606.18193)

    该研究通过大规模红队测试评估了 Anthropic 三个前沿模型对自动化越狱攻击的鲁棒性，发现所有模型均能抵御大部分攻击，残余风险主要来自自适应迭代攻击而非静态混淆，且模型鲁棒性排序与发布日期无关。

    

    我们评估了 Anthropic 开发的三个前沿大语言模型（LLM）——Opus 4.8、Fable 5 及其后续版本 Fable 5.1——的对抗鲁棒性，测试对象为四类自动化越狱攻击，涵盖十类危害分类体系中的 7,826 个有害意图。借助 HackAgent 红队框架，我们生成了数十万次对抗性尝试，并对所有三个目标，每一个表面上的攻击成功都由同一组五个前沿评判模型（≥4/5 一致同意）进行独立重新裁定。所有模型都能抵御大多数攻击，但残余的攻击面比总体框架所显示的更大：其主要由自适应迭代攻击构成，而静态混淆攻击几乎被完全化解。在针对所有三个目标运行的两类攻击（TAP 和 PAP）上，在分母相同的情况下，鲁棒性排序并不遵循模型发布日期：Fable 5 最为鲁棒（2.72%），Opus 4.8 次之。

    arXiv:2606.18193v2 Announce Type: replace-cross  Abstract: We evaluate the adversarial robustness of three frontier large language models (LLMs) developed by Anthropic, Opus 4.8, Fable 5 and its successor Fable 5.1, against four families of automated jailbreak attack across 7,826 harmful intents spanning a ten-category harm taxonomy. Using the HackAgent red-teaming framework, hundreds of thousands of adversarial attempts were generated and every apparent success was independently re-adjudicated by the same panel of five frontier judge models ($\geq 4/5$ agreement) for all three targets. All models resist the majority of attacks, but the residual surface is larger than aggregate framing suggests: it is dominated by adaptive iterative attacks, while static obfuscation is near-fully neutralised. On the two attack families run against all three targets (TAP and PAP), with identical denominators, the ordering does not follow release date: Fable 5 is most robust ($2.72\%$), Opus 4.8 follows 
    
[^290]: BD-LSC数据集：助力俚语与标准用法词汇语义变化检测模型的基准测试

    The BD-LSC Dataset: Facilitating the Benchmarking of Models for Lexical Semantic Change Detection in Slang and Standard Usage

    [https://arxiv.org/abs/2606.16560](https://arxiv.org/abs/2606.16560)

    该论文提出了BD-LSC和ST-WSD两个互补的基准数据集，首次系统支持双向词汇语义变化（义项同时获得与丢失）的检测，以及兼具俚语与标准用法词语的语义变化研究与词义消歧基准测试。

    

    自动语义变化检测旨在识别词义如何随时间演变，为语言和社会变迁提供洞察。尽管计算词汇语义变化（LSC）领域近期取得了进展，但现有的基准和方法仍难以捕捉双向语义变化，尤其是词语同时获得和失去义项的情况。对于同时具有俚语含义和标准含义的词语而言，这一问题尤为具有挑战性。为填补这些空白，我们引入了两个互补的基准数据集。双向词汇语义变化（BD-LSC）数据集捕捉了三个时间段内的义项获得、义项丢失和语义稳定情况，支持对复杂语义轨迹的研究。SlangTrack词义消歧（ST-WSD）数据集为兼具俚语和标准用法的词语提供了细粒度的、实例级的义项标注，支持对词义消歧（WSD）和语义变化检测的系统化基准测试。

    arXiv:2606.16560v2 Announce Type: replace  Abstract: Automatic semantic change detection aims to identify how word meanings shift over time, offering insights into both linguistic and societal change. Despite recent progress in computational lexical semantic change (LSC), existing benchmarks and methods struggle to capture bi-directional semantic change, particularly cases where words simultaneously gain and lose senses. This problem is especially challenging for words that have both slang and standard meanings. To address these gaps, we introduce two complementary benchmark datasets. The Bi-Directional Lexical Semantic Change (BD-LSC) dataset captures sense gain, sense loss, and stability across three time periods, enabling the study of complex semantic trajectories. The SlangTrack Word Sense Disambiguation (ST-WSD) dataset provides fine-grained, instance-level sense annotations for words combining slang and standard usages, supporting systematic benchmarking of WSD and semantic chang
    
[^291]: AdaMame：一种自适应多语言推理的训练方法

    AdaMame: A Training Recipe for Adaptive Multilingual Reasoning

    [https://arxiv.org/abs/2606.15080](https://arxiv.org/abs/2606.15080)

    提出AdaMame两阶段训练方法，通过AdaMame-GRPO中渐进增长的查询条件对齐因子，将推理语言自适应对齐到查询语言，在不牺牲准确性的情况下解决多语言推理中的语言崩溃问题。

    

    尽管大型推理模型（LRMs）在英语中表现出强大的性能，但它们往往无法使用查询本身的语言进行推理，这种现象被称为语言崩溃。现有基于强化学习的修复方法通常在准确性目标之外添加一个二元的语言忠实度奖励，但仍然会导致准确性下降、推理过程中途的语码转换以及过度的token使用等问题。在本工作中，我们提出了AdaMame，一种面向多语言数学推理的两阶段训练方法，通过在不损害准确性的前提下将推理语言自适应地对齐到查询语言，从而解决上述限制。第一阶段SFT在跨五种语言的非机器翻译推理轨迹上进行微调，以建立多语言推理能力。在随后的强化学习阶段，我们引入了AdaMame-GRPO，这是对组相对策略优化（GRPO）的一种改进，其中基于查询条件的对齐因子在训练过程中逐渐增长，引导模型首先……

    arXiv:2606.15080v2 Announce Type: replace  Abstract: While Large Reasoning Models (LRMs) show strong performance in English, they often fail to reason in the language of the query, a phenomenon known as language collapse. Existing RL-based fixes typically add a binary language fidelity reward to the accuracy objective, yet still incur trade-off in accuracy, mid-trace code-switching, and excessive token usage. In this work, we propose AdaMame, a two-stage training recipe for multilingual mathematical reasoning that addresses these limitations by adaptively aligning the reasoning language to the query language without compromising accuracy. The first SFT stage fine-tunes on non-MT reasoning traces across five languages to establish multilingual reasoning capability. In the subsequent RL stage, we introduce AdaMame-GRPO, an adaptation of Group Relative Policy Optimization (GRPO) in which a query-conditioned alignment factor grows progressively during training, guiding the model to first e
    
[^292]: MDForge：稀疏模拟器反馈下的智能体分子动力学流水线设计

    MDForge: Agentic Molecular Dynamics Pipeline Design under Sparse Simulator Feedback

    [https://arxiv.org/abs/2606.12916](https://arxiv.org/abs/2606.12916)

    MDForge是一个LLM智能体，通过物理专家多智能体辩论来稠密化稀疏的模拟器反馈，以开放式代码生成方式自动设计出可与人类专家媲美的分子动力学流水线，并发现了新型主客体结合剂。

    

    分子动力学（MD）是原子分子科学中经典的计算机模拟方法，基于第一性原理物理来模拟分子行为。为一个新系统设计MD流水线需要大量的专家知识：即使只对一个分子运行模拟也十分昂贵，无法进行试错。我们利用LLM智能体将这一专家流水线设计过程自动化。与现有编排预定义工具集的MD智能体不同，我们将流水线设计视为开放式的代码生成，其中智能体的行为通过语言奖励在线重塑。具体而言，我们构建了MDForge，一个LLM智能体，其上下文内更新规则通过物理专家之间的多智能体辩论来稠密化稀疏奖励。在三个SAMPL主客体结合自由能基准测试上，MDForge自动设计的MD流水线可与人类专家相媲美。部署在一个未见过的候选客体库上时，其CB[7]流水线发现了一种新型结合剂……

    arXiv:2606.12916v2 Announce Type: replace-cross  Abstract: Molecular dynamics (MD) is the canonical in-silico method for atomistic molecular science, simulating molecular behavior from first-principle physics. Designing an MD pipeline for a new system requires substantial expert knowledge: running it on even one molecule is expensive, ruling out trial-and-error. We automate this expert pipeline-design process with an LLM agent. Unlike existing MD agents that orchestrate a predefined tool set, we treat pipeline design as open-ended code generation in which the agent's behavior is reshaped online by verbal reward. Specifically, we build MDForge, an LLM agent whose in-context update rule densifies the sparse reward via a multi-agent debate among physics experts. On three SAMPL host-guest binding free-energy benchmarks, MDForge automatically designs MD pipelines competitive with human experts. Deployed on a library of unseen candidate guests, its CB[7] pipeline discovers a novel binder tha
    
[^293]: 重新审视无监督词汇发现中的词典评估

    Revisiting Lexicon Evaluation in Unsupervised Word Discovery

    [https://arxiv.org/abs/2606.06183](https://arxiv.org/abs/2606.06183)

    该论文指出现有无监督词汇发现中常用的归一化编辑距离评估指标存在偏向大聚类且忽略真实类别跨聚类分布的固有缺陷，并提出考虑聚类大小的加权指标和评估真实词汇分散程度的逆向指标，以实现更公平可靠的词典质量评估。

    

    从发现的类词单元构建词典是零资源语音处理的一项核心目标。但我们的评估方法能否提供对词典质量的可信指示？一种常用的指标——归一化编辑距离——会对每个聚类中发现的单元之间的音素编辑距离取平均值。我们证明该指标存在对大聚类质量的固有偏差，从而阻碍了公平评估。此外，它还忽略了真实类别在各个聚类之间的分布情况。基于聚类文献中的既有理论，我们提出了两个能够解决这些缺陷的指标：一个是在评估聚类内一致性时对聚类大小进行加权的改进指标，另一个是评估真实词汇如何分散在各聚类之间的逆向指标。通过在合成词典和真实词典上的实验，我们证明这两个指标相结合能够：（1）与词典和真实情况（ground truth）的相似度更紧密地相关……

    arXiv:2606.06183v2 Announce Type: replace-cross  Abstract: Building a lexicon from discovered word-like units is a central goal in zero-resource speech processing. But do our evaluations provide a trustworthy indication of lexicon quality? A common metric, normalized edit distance, averages the phoneme edit distances between discovered units in each cluster. We show that this metric has an inherent bias toward the quality of large clusters, inhibiting fair evaluation. Moreover, it ignores how well true classes are distributed across clusters. Based on established theory in clustering literature, we propose two metrics that address these shortcomings: a modified metric that weighs cluster size when assessing within-cluster consistency, and an inverse metric that assesses how true words are spread across clusters. Through experiments on synthetic and real-world lexicons, we demonstrate that combined, these metrics are: (1) more closely correlated with how similar a lexicon is to the grou
    
[^294]: 通用智能体能够自动化数据策展吗？

    Can Generalist Agents Automate Data Curation?

    [https://arxiv.org/abs/2606.04261](https://arxiv.org/abs/2606.04261)

    该论文提出以智能体为中心的Curation-Bench基准，发现通用编码智能体能在十次迭代内达到强数据选择基线，但存在持续的“执行-研究差距”——智能体倾向于微调局部策略变体而非探索全新的策略家族。

    

    策展训练数据是现代AI开发中最关键却又最耗费人力的环节之一：从业者需要针对嘈杂的基准反馈，反复提出、实现、评估和修订数据策略。我们探讨通用编码智能体能否自动化这一数据策展循环。我们提出了*Curation-Bench*，这是一个以智能体为中心的基准，它固定了模型、训练方案和评估套件，同时赋予智能体命令行权限，使其能够检查数据、实现策略、将策略提交到固定的训练/评估流程并进行修订。在一个视觉-语言指令微调的实例中，开箱即用的智能体在十次迭代内就达到了已发表的强数据选择基线水平。然而，轨迹分析揭示了一个持续的*执行-研究差距*：即使提供了策略指南和论文参考文献，智能体仍主要调整局部策略变体，而非探索新的策略家族。需要e

    arXiv:2606.04261v2 Announce Type: replace-cross  Abstract: Curating training data is among the most consequential yet labor-intensive parts of modern AI development: practitioners iteratively propose, implement, evaluate, and revise data policies against noisy benchmark feedback. We ask whether generalist coding agents can automate this data-curation loop. We introduce *Curation-Bench*, an agent-centric benchmark that fixes the model, training recipe, and evaluation suite while giving agents command-line access to inspect data, implement policies, submit them to a fixed training/evaluation pipeline, and revise. In a vision-language instruction-tuning instantiation, out-of-the-box agents reach strong published data-selection baselines within ten iterations. However, trajectory analysis reveals a persistent *execution-research gap*: agents mainly tune local policy variants rather than explore new policy families, even when given strategy guides and paper references. Scaffolds requiring e
    
[^295]: 分解混合专家模型中的拒绝引导机制

    Decomposing Refusal Steering in Mixture-of-Experts Models

    [https://arxiv.org/abs/2606.04160](https://arxiv.org/abs/2606.04160)

    该研究将拒绝引导方法扩展到混合专家模型，发现单个专家在自由选择位置时平均可恢复78%的完整引导效果，从而揭示了MoE模型中拒绝机制在各组件间的运作方式。

    

    arXiv:2606.04160v2 公告类型：替换 摘要：指令微调大语言模型（LLM）的安全对齐依赖于模型可靠地拒绝有害或违规请求的能力。近期研究表明，可以在推理阶段向稠密LLM施加引导向量以抑制拒绝行为，从而诱导模型对有害请求作出响应。我们将这种拒绝引导方法扩展到三个开源混合专家模型上，以分解并更好地理解拒绝机制如何在MoE的各个组件中运作。我们发现，引导性能不受MoE架构固有的复杂路由模式的影响；当允许专家自由选择自身位置时，单个专家平均可恢复78%的完整引导效果。然而，当专家级引导被限制在全层引导所使用的位置时，平均仅能恢复54%的效果，且在安全相关系统提示下这一差距进一步扩大。我们的结果还表明，拒绝信号...

    arXiv:2606.04160v2 Announce Type: replace  Abstract: Safety alignment in instruction-tuned large language models (LLMs) depends on a model's ability to reliably refuse harmful or disallowed requests. Recent work has shown that a steering vector can be applied to a dense LLM during inference to suppress refusal behavior and induce responses to harmful requests. We extend this refusal steering method to three open-source Mixture-of-Experts (MoE) LLMs to decompose and better understand how refusal mechanisms operate across MoE components. We find that steering performance is uninhibited by the complex routing patterns inherent to the MoE architecture, and that a single expert recovers 78% of the full steering effect on average when free to select its own location. However, expert-level steering only recovers 54% on average when constrained to the location used by full-layer steering, a gap that widens further under safety-related system prompts. Our results also show that refusal signals 
    
[^296]: 面向逐步模型路由的评分准则引导过程奖励

    Rubric-Guided Process Reward for Stepwise Model Routing

    [https://arxiv.org/abs/2605.29310](https://arxiv.org/abs/2605.29310)

    提出RoRo框架，用评分准则引导的过程奖励替代仅反映最终答案正确性的结果奖励，从而更好地评估逐步模型路由中的中间决策并提升性能与泛化能力。

    

    逐步模型路由通过将每个推理步骤分配给合适的模型，从而提高大型推理模型（LRMs）的效率。最近的方法将路由建模为序贯决策过程，并使用强化学习训练路由器。然而，尽管这些方法将路由建模为一个过程，它们仍然使用结果奖励来监督路由器。这类奖励仅反映最终答案的正确性，无法评估中间的路由决策，这可能会削弱性能和泛化能力。为解决这一缺口，我们提出了RoRo，一个用于逐步模型路由的评分准则引导的过程奖励框架。RoRo首先收集多样化的路由轨迹，并基于结果、成本和过程质量构建偏好对。随后，它通过交替优化训练一个Rubricor来生成针对特定查询的评估准则，以及一个Judge在该准则下对路由轨迹进行评分。由此得到的过程奖励……

    arXiv:2605.29310v2 Announce Type: replace-cross  Abstract: Stepwise model routing improves the efficiency of Large Reasoning Models (LRMs) by assigning each reasoning step to a suitable model. Recent methods formulate routing as a sequential decision process and train the router with reinforcement learning. However, although they model routing as a process, they still supervise the router with outcome rewards. Such rewards only reflect final answer correctness and fail to evaluate intermediate routing decisions, which can weaken performance and generalization. To address this gap, we propose RoRo, a rubric-guided process reward framework for stepwise model routing. RoRo first collects diverse routing trajectories and constructs preference pairs based on outcome, cost, and process quality. It then trains a Rubricor to generate a query-specific evaluation rubric and a Judge to score routing trajectories under this rubric through alternating optimization. The resulting process rewards are
    
[^297]: 更难的文本嵌入基准（HTEB）：超越一维静态鲁棒性

    The Harder Text Embedding Benchmark (HTEB): Beyond One-dimensional Static Robustness

    [https://arxiv.org/abs/2605.28190](https://arxiv.org/abs/2605.28190)

    提出动态评估框架HTEB，通过LLM在评估时随机变换输入，从词汇/风格、长度和语言三个维度评估文本嵌入模型的多维鲁棒性，揭示了静态基准无法发现的模型失败模式。

    

    像MTEB这样的嵌入基准为每个模型报告单一分数，隐含地将鲁棒性视为静态的标量属性。我们认为嵌入鲁棒性是多维的，因为模型对不同类型的变异会产生不同的响应，并且需要动态评估来暴露被静态基准所隐藏的失败。我们提出了更难的文本嵌入基准（HTEB），这是一个动态评估框架，通过在评估时使用大语言模型（LLM）随机变换输入，从三个具有实际可解释性的维度（词汇/风格、长度和语言）挑战模型的鲁棒性。我们在32个覆盖42种语言的数据集上评估了16个开源权重嵌入模型，其变换经过英语子样本上4,800条个人工评分的验证，并辅以一项西班牙语源评估以及一项关于句对级标签保留的探索性STS-B研究。我们发现了三种模式：（1）模型表现出特定的、部分解耦的……（原文在此处截断）

    arXiv:2605.28190v2 Announce Type: replace  Abstract: Embedding benchmarks like MTEB report a single score per model, implicitly treating robustness as a static, scalar property. We argue that embedding robustness is multidimensional, since models respond differently to different types of variation, and requires dynamic evaluation to expose failures hidden by static benchmarks. We introduce the Harder Text Embedding Benchmark (HTEB), a dynamic evaluation framework that challenges model robustness along three practically interpretable axes (Lexical/Stylistic, Length and Language) by stochastically transforming inputs at evaluation time with an LLM. Evaluating 16 open-weight embedding models on 32 datasets covering 42 languages under transformations validated by 4,800 individual human ratings on an English subsample, supplemented by a Spanish-source evaluation and an exploratory STS-B study of pair-level label preservation, we find three patterns: (1) Models exhibit specific, partly decou
    
[^298]: TRACES：基于轨迹状态建模的多轮LLM智能体主动安全审计

    TRACES: Proactive Safety Auditing for Multi-Turn LLM Agents via Trajectory-State Modeling

    [https://arxiv.org/abs/2605.27690](https://arxiv.org/abs/2605.27690)

    TRACES通过从LLM隐藏表示中建模轨迹风险状态的时间演化，仅用弱轨迹级监督即可对多轮智能体交互实现主动的、密集的前缀级安全风险审计。

    

    LLM智能体越来越多地通过多轮工具调用与环境交互来执行任务，而安全风险往往早在最终结果显现之前就已从中间步骤中萌发。因此，被动式审计是不够的：事后诊断常常错失在风险正在发生时进行标记的机会。我们提出了TRACES，一种基于表示学习的主动审计器，它从观察者LLM的隐藏表示中学习前缀级别的轨迹风险状态。TRACES从步骤表示中归纳出潜在的机制特征，并对其时间演化进行建模，以估计部分轨迹是否正在漂移向不安全行为。为了避免步骤级风险标注的成本与歧义，TRACES仅使用弱化的轨迹级监督进行训练，同时仍能产生密集的前缀级风险估计。在多个智能体安全基准上，TRACES同时提升了全轨迹安全预测和（原文在此处截断）……

    arXiv:2605.27690v2 Announce Type: replace  Abstract: LLM agents increasingly operate through multi-turn tool use and environment interaction, where safety risks often emerge from intermediate steps long before they surface in the final outcome. Reactive auditing is therefore insufficient: post-hoc diagnosis frequently misses the chance to flag risks while they are unfolding. We propose TRACES, a representation-based proactive auditor that learns prefix-level trajectory risk states from the hidden representations of an observer LLM. TRACES induces latent mechanism features from step representations and models their temporal evolution to estimate whether a partial trajectory is drifting toward unsafe behavior. To sidestep the cost and ambiguity of step-level risk annotation, TRACES is trained with weak trajectory-level supervision while still producing dense prefix-level risk estimates. Across multiple agent safety benchmarks, TRACES improves both full-trajectory safety prediction and pr
    
[^299]: BAIT：基于自条件推理的边界引导式信息披露升级LLM越狱方法

    BAIT: Boundary-Guided Disclosure Escalation LLM Jailbreaking via Self-Conditioned Reasoning

    [https://arxiv.org/abs/2605.27110](https://arxiv.org/abs/2605.27110)

    BAIT是一个三步越狱框架，通过让模型先识别、再细化自身安全边界并最后请求详细示例，将模型自身的推理与一致性倾向转化为信息披露途径，在多个基准测试中对顶级大语言模型实现了持续的高攻击成功率。

    

    在这项工作中，我们提出了BAIT（边界感知迭代陷阱），这是一个三步越狱框架，通过目标大语言模型（LLM）的内部信息披露来引出恶意信息，而不是依赖裁判LLM的外部反馈。BAIT首先要求模型识别其保护边界，然后要求它细化该边界，最后请求一个详细示例。通过在模型之前的回答基础上逐步扩展每一步，BAIT将模型自身的推理和一致性倾向转变为信息披露的途径。在AdvBench、JailbreakBench、AIR-Bench和SORRY-Bench上的实验表明，BAIT在顶级大语言模型上持续取得较高的攻击成功率，显著超越了传统越狱基线。进一步分析揭示：1）以预防为导向的表述显著优于直接的知识请求；2）细化步骤在攻击中起着关键作用（摘要在此处截断）。

    arXiv:2605.27110v2 Announce Type: replace-cross  Abstract: In this work, we propose BAIT (Boundary-Aware Iterative Trap), a three-step jailbreak framework that elicits malicious information through internal disclosure by target large language models (LLMs), instead of external feedback from judge LLMs. BAIT first asks the model to identify the protection boundary, then requires it to refine that boundary, and finally requests a detailed example. By expanding each step upon the model's previous responses, BAIT turns the model's own reasoning and consistency tendency into a disclosure pathway. Experiments on AdvBench, JailbreakBench, AIR-Bench, and SORRY-Bench demonstrate that BAIT consistently achieves strong attack success rates across top-tier large language models, significantly advancing conventional jailbreak baselines. Further analysis reveals that: 1) prevention-oriented framing significantly outperforms direct knowledge requests; 2) the refinement step plays a critical role in d
    
[^300]: 大语言模型能“时间旅行”吗？通过强化学习增强法律智能体搜索中的时间一致性

    Can LLMs Time Travel? Enhancing Temporal Consistency in Legal Agentic Search through Reinforcement Learning

    [https://arxiv.org/abs/2605.25920](https://arxiv.org/abs/2605.25920)

    提出 LegalSearch-R1 强化学习框架，通过结合本地法条 RAG 与在线网络搜索，并在跨越多个修订时期的时间索引数据上训练，解决了法律大模型的时间偏差和搜索智能体忽视时间约束的问题，确保所适用的法律与案件的时间背景保持一致。

    

    尽管配备了智能体搜索能力的大语言模型（LLM）在法律推理方面展现出潜力，但它们忽视了一个基本约束：适用的法律必须与每个案件的时间背景相匹配，因为溯及既往地适用法律违反了核心法律原则并会导致错误的结论。我们的观察发现，当前的法律大语言模型存在锚定于其训练截止时间的时间偏差，而搜索智能体很少将时间约束纳入查询之中，且仅靠网络搜索无法提供法律推理所需的精确法条与判例引用。为应对这些挑战，我们提出了 LegalSearch-R1，这是一个端到端的强化学习框架，将用于精确条文匹配的本地法条 RAG 与用于获取更广泛法律知识的在线网络搜索相结合，并在跨越多个法律修订时期的时间索引数据上进行训练，以强制实现时间一致性。

    arXiv:2605.25920v2 Announce Type: replace  Abstract: While large language models (LLMs) augmented with agentic search capabilities show promise for legal reasoning, they overlook a fundamental constraint that applicable law must match the temporal context of each case, as retroactive application of statutes violates core legal principles and leads to erroneous conclusions. Our observations reveal that current legal LLMs suffer from temporal bias anchored to their training cutoff, while search agents rarely incorporate temporal constraints into queries, and that web search alone cannot provide the precise statute and precedent citations that legal reasoning demands. To address these challenges, we propose LegalSearch-R1, an end-to-end reinforcement learning framework that pairs local statute RAG for precise article matching with online web search for broader legal knowledge, trained on temporally-indexed data spanning multiple amendment periods to enforce temporal consistency. Extensive
    
[^301]: STOP：低数据场景下长篇推理的结构化在线剪枝方法

    STOP: Structured On-Policy Pruning of Long-Form Reasoning in Low-Data Regimes

    [https://arxiv.org/abs/2605.13165](https://arxiv.org/abs/2605.13165)

    STOP是一种在线剪枝算法，通过将长思维链推理轨迹结构化为推理树，并保留以最早正确节点结尾的最短前缀，在低数据微调场景下有效缓解了推理模型的过度思考问题。

    

    长思维链推理提升了多步骤问题的性能，但同时也会引发过度思考。这种低效在低数据微调场景中尤为成问题，因为现实应用中推理模型的适配只有有限的监督数据，无法依赖大规模教师蒸馏或繁重的测试时控制。为解决这一问题，我们提出了STOP（结构化在线剪枝），一种用于分析和剪枝长篇推理轨迹的在线算法。STOP从模型自身构建自蒸馏轨迹，然后通过节点分割、分类标注和推理树构建，将每条轨迹映射到一个结构化推理接口。在该接口之上，我们引入了ECN（最早正确节点），它保留以最早正确节点结尾的最短前缀。在DeepSeek-R1-Distill-Qwen-7B和DeepSeek-R1-Distill-LLaMA-3-8B模型上，针对GSM8K、Math 500和AIME 2024的实验表明

    arXiv:2605.13165v2 Announce Type: replace  Abstract: Long chain-of-thought (Long CoT) reasoning improves performance on multi-step problems, but it also induces overthinking. This inefficiency is especially problematic in low-data fine-tuning regimes, where real applications adapt reasoning models with limited supervision and cannot rely on large-scale teacher distillation or heavy test-time control. To address this, we propose STOP (Structured On-policy Pruning), an on-policy algorithm for analyzing and pruning long-form reasoning traces. STOP constructs self-distilled traces from the model. Then it maps each trace into a structured reasoning interface through node segmentation, taxonomy annotation, and reasoning-tree construction. On top of this interface, we introduce ECN (Earliest Correct Node), which retains the shortest prefix ending at the earliest node. Experiments on DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-LLaMA-3-8B across GSM8K, Math 500, and AIME 2024 show that 
    
[^302]: 有据可依还是凭空猜测？基于盲图对比排序的视觉语言模型置信度估计

    Grounded or Guessing? LVLM Confidence Estimation via Blind-Image Contrastive Ranking

    [https://arxiv.org/abs/2605.10893](https://arxiv.org/abs/2605.10893)

    提出BICR框架，通过对比真实图像与涂黑图像下冻结LVLM的隐藏状态，并用排序损失正则化训练一个轻量探针，使置信度估计能够检测模型是否真正依赖图像而非仅凭语言先验作答。

    

    大型视觉语言模型（LVLM）存在视觉无根据问题：它们可能在完全由语言先验驱动的情况下生成流畅、自信甚至正确的回答，而图像对预测毫无贡献。现有的置信度估计方法无法检测到这一问题，因为它们仅在正常推理下观察模型行为，缺乏判断预测是由图像塑造还是仅由文本决定的机制。我们提出了BICR（盲图对比排序），这是一个与模型无关的置信度估计框架。LVLM完全保持冻结；相反，我们对每个样本提取两次隐藏状态：一次使用真实的图像-问题对，另一次在问题保持不变的情况下将图像涂黑。随后在一个小型独立探针上使用真实图像的隐藏状态进行训练，并通过排序损失进行正则化，该损失对涂黑视图下更高的置信度进行惩罚，从而教会探针将视觉根据作为（置信度的）信号。

    arXiv:2605.10893v3 Announce Type: replace  Abstract: Large vision-language models (LVLMs) suffer from visual ungroundedness: they can produce a fluent, confident, and even correct response driven entirely by language priors, with the image contributing nothing to the prediction. Existing confidence estimation methods cannot detect this, as they observe model behavior under normal inference with no mechanism to determine whether a prediction was shaped by the image or by text alone. We introduce BICR (Blind-Image Contrastive Ranking), a model-agnostic confidence estimation framework. The LVLM stays entirely frozen; instead, we extract its hidden states twice per sample, once with the real image-question pair and once with the image blacked out while the question is held fixed. A small, separate probe is then trained on the real-image hidden state, regularized by a ranking loss that penalizes higher confidence on the blacked-out view, teaching it to treat visual grounding as a signal of 
    
[^303]: ReLay：个性化LLM生成的通俗语言摘要助力更好理解，但代价是什么？

    ReLay: Personalized LLM-Generated Plain-Language Summaries for Better Understanding, but at What Cost?

    [https://arxiv.org/abs/2605.00468](https://arxiv.org/abs/2605.00468)

    该论文提出了ReLay数据集，通过对比专家撰写的静态通俗摘要与LLM个性化生成的交互式摘要，评估了五种LLM在健康信息个性化摘要中的效果、最有效的个性化策略以及个性化与安全性之间的权衡。

    

    通俗语言摘要（PLS）旨在让普通读者更容易理解研究内容，但它们通常以“一刀切”的风格撰写，忽视了读者在信息需求和理解能力上的差异。在健康领域，这一局限性尤为重要，因为对科学信息的误解可能影响现实世界中的决策。大型语言模型（LLM）为个性化PLS提供了新的机遇，但个性化是否真的有帮助、哪些策略最为有效、以及如何在个性化与安全性之间取得平衡，目前仍不清楚。我们提出了ReLay，一个包含来自50名普通参与者的300个参与者-PLS对的数据集，涵盖静态（专家撰写）和交互式（LLM个性化）两种设置。ReLay包括用户特征、健康信息需求、信息检索行为、理解结果、交互日志和质量评分。我们利用ReLay评估了五种LLM在两种个性化……

    arXiv:2605.00468v2 Announce Type: replace  Abstract: Plain Language Summaries (PLS) aim to make research accessible to lay readers, but they are typically written in a one-size-fits-all style that ignores differences in readers' information needs and comprehension. In health contexts, this limitation is particularly important because misunderstanding scientific information can affect real-world decisions. Large language models (LLMs) offer new opportunities for personalizing PLS, but it remains unclear whether personalization helps, which strategies are most effective, and how to balance personalization with safety. We introduce ReLay, a dataset of 300 participant--PLS pairs from 50 lay participants in both static (expert-written) and interactive (LLM-personalized) settings. ReLay includes user characteristics, health information needs, information-seeking behavior, comprehension outcomes, interaction logs, and quality ratings. We use ReLay to evaluate five LLMs across two personalizat
    
[^304]: 随机系统中分布稳定性的信息几何首达监测

    Information-Geometric First-Passage Monitoring of Distributional Stability in Stochastic Systems

    [https://arxiv.org/abs/2604.24083](https://arxiv.org/abs/2604.24083)

    本文提出一种融合信息几何、相对熵耗散与序贯推断的有界首达监测架构，能在控制重复检验误报的前提下，区分随机系统的名义分布松弛与真正的状态机制偏离。

    

    随机系统的运行时监测需要在明确的假设有效性前提下，区分名义上的分布松弛与真正的状态机制偏离，同时控制重复检验带来的误报。本文在一个有界的首达监测架构中，将相对熵耗散、信息几何与序贯推断联系起来。对于可逆的Fokker–Planck动力学，相对于不变密度的相对熵是非递增的；在外生强迫作用下，其导数可分解为名义耗散项与信息空间强迫项。运行时层采用高斯窗口代理、名义相对协方差收缩、坐标一致的相对精度诊断，以及由混合幂鞅过程聚合的随机化保形秩。解析的Ornstein–Uhlenbeck验证给出了零正的名义Kullback–Leibler增量、低于3.31×10⁻⁶的强迫恒等式残差，以及坐标……（原文摘要截断）

    arXiv:2604.24083v2 Announce Type: replace-cross  Abstract: Runtime monitoring of stochastic systems must distinguish nominal distributional relaxation from regime departure while controlling repeated-test false alarms under explicit validity assumptions. This paper links relative-entropy dissipation, information geometry, and sequential inference in a bounded first-passage monitoring architecture. For reversible Fokker--Planck dynamics, relative entropy to an invariant density is non-increasing; under exogenous forcing, its derivative decomposes into nominal dissipation and an information-space forcing term. The runtime layer uses Gaussian window surrogates, nominal-relative covariance shrinkage, a coordinate-consistent relative precision diagnostic, and randomized conformal ranks aggregated by a mixture power-martingale process. Analytical Ornstein--Uhlenbeck validation gives zero positive nominal Kullback--Leibler increments, forcing-identity residuals below 3.31 x 10^-6, and coordin
    
[^305]: 刻画模型原生技能

    Characterizing Model-Native Skills

    [https://arxiv.org/abs/2604.17614](https://arxiv.org/abs/2604.17614)

    该论文提出技能刻画应“模型原生”地基于模型自身表征而非外部人工分类体系，通过从序列级激活中恢复紧凑的正交基来捕捉模型自身组织的行为变化轴，并在推理后训练中验证了该方法的有效性。

    

    技能是描述语言模型能做什么以及其行为如何被改变的一个自然单元。然而，现有的刻画方法依赖于人工编写的分类体系、文本描述或手动分析流程——这些都是关于什么重要的外部假设，未必与模型的内部表征相一致。我们认为，当目标是对模型行为进行干预时，技能刻画应当是“模型原生”的：即基于模型自身的表征，而非通过外部本体强行施加。我们通过从序列级激活中恢复一个紧凑的正交基来实例化这一观点。所得到的基在语义上可解释，但不必对应任何预定义的人类本体；相反，它捕捉了模型自身围绕其组织行为变化的轴。我们在推理后训练上验证了这一刻画，将恢复的基用于SFT数据……（摘要原文在此处截断）

    arXiv:2604.17614v2 Announce Type: replace-cross  Abstract: Skills are a natural unit for describing what a language model can do and how its behavior can be changed. However, existing characterizations rely on human-written taxonomies, textual descriptions, or manual profiling pipelines--all external hypotheses about what matters that need not align with the model's internal representations. We argue that when the goal is to intervene on model behavior, skill characterization should be *model-native*: grounded in the model's own representations rather than imposed through external ontologies. We instantiate this view by recovering a compact orthogonal basis from sequence-level activations. The resulting basis is semantically interpretable but need not correspond to any predefined human ontology; instead, it captures axes of behavioral variation that the model itself organizes around. We validate this characterization on reasoning post-training, using the recovered basis for both SFT da
    
[^306]: 英语并非一切所需：系统探索多语言性在大语言模型后训练中的作用

    English is Not All You Need: Systematically Exploring the Role of Multilinguality in LLM Post-Training

    [https://arxiv.org/abs/2604.13286](https://arxiv.org/abs/2604.13286)

    该研究通过220次受控监督微调实验系统证明，仅用英语进行大语言模型后训练并非最优——引入哪怕一种非英语语言即可同时提升英语性能与跨语言泛化能力，且语言多样性越高收益越大，尤其有利于低资源语言。

    

    尽管大语言模型已被广泛进行多语言部署，但其后训练流程仍然主要以英语为中心，导致不同语言之间的性能差异。我们基于220次监督微调实验，对训练语言覆盖范围、模型规模与任务领域之间的相互作用进行了系统性、受控的研究。这些实验在平行的翻译多语言数据混合上进行，涵盖数学推理和API调用任务，所使用的模型规模最高达8B参数。我们发现，仅使用英语的后训练通常并非最优：即使只引入一种非英语语言，也能同时提升英语性能和跨语言泛化能力。在后训练中增加语言多样性通常会带来进一步的收益，尤其是对低资源语言而言，而高资源语言的性能趋于达到平稳而非下降。此外，更大的语言多样性能够带来强大的零样本……

    arXiv:2604.13286v2 Announce Type: replace  Abstract: Despite the widespread multilingual deployment of large language models, post-training pipelines remain predominantly English-centric, contributing to performance disparities across languages. We present a systematic, controlled study of the interplay between training language coverage, model scale, and task domain, based on 220 supervised fine-tuning runs on parallel translated multilingual data mixtures spanning mathematical reasoning and API calling tasks, with models up to 8B parameters. We find that English-only post-training is typically suboptimal: incorporating even a single non-English language improves both English performance and cross-lingual generalization. Increasing language diversity during post-training generally yields further gains, particularly for low-resource languages, while performance on high-resource languages tends to plateau rather than degrade. Moreover, greater language diversity enables strong zero-shot
    
[^307]: 置信于置信度分数：研究置信度分数对监督微调的敏感性

    Confident in a Confidence Score: Investigating the Sensitivity of Confidence Scores to Supervised Fine-Tuning

    [https://arxiv.org/abs/2604.08974](https://arxiv.org/abs/2604.08974)

    该研究系统考察了监督微调对语言模型置信度指标校准性的影响，发现在翻译、问答和数学推理等216种配置中校准性有升有降（112例下降、104例提升），表明微调后置信度分数的可靠性并不稳定，需要重新校准。

    

    不确定性量化技术通过衡量语言模型输出的置信度，来支持幻觉检测和选择性预测等关键应用。虽然先前的工作已经开发了各种置信度指标，并证明了它们在分类任务或言语化置信度场景下的校准性，但基于概率和基于自一致性的不确定性量化指标在自然语言生成任务中的鲁棒性仍未得到充分探索，尤其是在模型适配的场景下。由于从业者通常会应用监督微调来使模型适应新任务，一个关键问题随之产生：当模型经过微调后，置信度指标是否还能保持其校准性？我们在包括翻译、问答和数学推理在内的自然语言生成任务上研究了这个问题。我们发现，监督微调后校准性会发生显著变化：在216种配置中，有112种情况下校准性出现下降，104种情况下校准性得到提升，且置信度分数……（摘要原文截断）

    arXiv:2604.08974v2 Announce Type: replace  Abstract: Uncertainty quantification techniques measure confidence in language model outputs to support critical applications like hallucination detection and selective prediction. While prior work has developed various confidence metrics and demonstrated their calibration for classification tasks or using verbalized confidence, the robustness of probability-based and self-consistency-based UQ metrics for natural language generation remains underexplored particularly under model adaptation. Since practitioners routinely apply supervised fine-tuning to adapt models to new tasks, a key question arises: do confidence metrics maintain their calibration when models are fine-tuned? We investigate this question across NLG tasks including translation, question answering, and mathematical reasoning. We find that calibration shifts substantially after SFT: across 216 configurations, it degrades in 112 cases and improves in 104, with confidence scores sh
    
[^308]: 通过有限音频弥合语音-文本差距，实现基于大语言模型的ASR的高效领域自适应

    Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-Based ASR

    [https://arxiv.org/abs/2604.06487](https://arxiv.org/abs/2604.06487)

    提出混合批处理（MB）策略，仅用不到4小时的目标域语音数据即可使基于LLM的ASR达到与使用完整数据集传统微调相当或更优的词错误率，有效弥合了语音-文本模态差距。

    

    传统的端到端自动语音识别（ASR）系统依赖成对的语音-文本数据进行领域自适应。近期基于大语言模型（LLM）的ASR架构通过投影模块将语音编码器与大型语言模型连接起来，从而能够仅使用文本数据进行自适应。然而，这引入了模态差距问题，因为LLM并未接触到语音投影器所产生的含噪表示。我们研究了少量语音数据是否能够缓解这种不匹配问题。我们比较了三种策略：仅文本自适应、成对语音-文本自适应，以及结合两者的混合批处理（MB）。在域内和域外设置下的实验表明，即使是有限的语音数据也能持续提升性能。值得注意的是，混合批处理方法仅使用目标域10%（不到4小时）的语音数据，其词错误率就能达到与使用完整数据集进行传统ASR微调相当甚至更好的水平，这表明少量语音数据……

    arXiv:2604.06487v2 Announce Type: replace  Abstract: Conventional end-to-end automatic speech recognition (ASR) systems rely on paired speech-text data for domain adaptation. Recent LLM-based ASR architectures connect a speech encoder to a large language model via a projection module, enabling adaptation with text-only data. However, this introduces a modality gap, as the LLM is not exposed to the noisy representations produced by the speech projector. We investigate whether small amounts of speech can mitigate this mismatch. We compare three strategies: text-only adaptation, paired speech-text adaptation, and mixed batching (MB), which combines both. Experiments in in-domain and out-of-domain settings show that even limited speech consistently improves performance. Notably, MB using only 10% of the target-domain (less than 4 hours) speech achieves word error rates comparable to, or better than, conventional ASR fine-tuning with the full dataset, indicating that small amounts of speech
    
[^309]: 什么造就了良好的多语言推理？用可度量特征解耦推理轨迹

    What Makes Good Multilingual Reasoning? Disentangling Traces with Measurable Features

    [https://arxiv.org/abs/2604.04720](https://arxiv.org/abs/2604.04720)

    该论文通过定义涵盖多语言对齐、推理步骤和推理流程的可度量特征，并结合逻辑回归与稀疏自编码器分析推理轨迹，揭示了多语言场景下成功推理的真实特征，挑战了“让各语言推理模仿英语推理即可弥合性能差距”的传统假设。

    

    大型推理模型（LRMs）在英语与其他语言之间仍然存在巨大的性能差距，然而当前许多工作假设这些差距可以通过让每种语言的推理都类似于英语推理来弥合。本工作挑战了这一假设，转而提出这样的问题：在多语言场景中，究竟是什么特征刻画了成功的推理轨迹？源自英语的推理特征又在多大程度上真正对其他语言有所帮助？我们首先定义了一套可度量的推理特征，涵盖推理轨迹的多语言对齐、推理步骤和推理流程等方面，并使用逻辑回归量化每个特征与最终答案准确率之间的关联。我们进一步在多语言推理轨迹上训练稀疏自编码器，以自动发现能够实例化或扩展这些特征的潜在推理概念。最后，我们利用这些特征对推理轨迹进行重排序，并衡量其对准确率的影响。

    arXiv:2604.04720v2 Announce Type: replace  Abstract: Large Reasoning Models (LRMs) still exhibit large performance gaps between English and other languages, yet much current work assumes these gaps can be closed simply by making reasoning in every language resemble English reasoning. This work challenges this assumption by asking instead: what actually characterizes successful reasoning traces in multilingual settings, and to what extent do English-derived reasoning features genuinely help in other languages? We first define a suite of measurable reasoning features spanning multilingual alignment, reasoning step, and reasoning flow aspects of reasoning traces, and use logistic regression to quantify how each feature associates with final answer accuracy. We further train sparse autoencoders over multilingual traces to automatically discover latent reasoning concepts that instantiate or extend these features. Finally, we use the features to re-rank traces and measure their impact on acc
    
[^310]: 更细的引用总是更好吗？重新思考归因生成中的粒度

    Are Finer Citations Always Better? Rethinking Granularity for Attributed Generation

    [https://arxiv.org/abs/2604.01432](https://arxiv.org/abs/2604.01432)

    该论文通过分析四种模型规模发现，细粒度句子级引用并非总是最优，段落级的中间粒度归因质量最佳，选择最优引用粒度可在几乎不牺牲答案正确性的情况下大幅提升模型性能与归因质量。

    

    引用粒度——即引用单个句子、段落还是整个文档——是归因生成（attributed generation）中的一个关键设计选择。尽管细粒度引用因便于人类进行精确验证而通常受到青睐，但其对模型性能的影响仍未得到充分探索。我们分析了四种模型规模（8B-120B），并证明强制使用细粒度（句子级）引用相对于表现最佳的引用粒度会损失2-97%（中位数40%）的性能增益，在个别任务上损失甚至高达338%。令人惊讶的是，将引用粒度设置为最优值（基于归因质量确定）可以释放这些可观的增益，同时整体答案的正确性基本保持不变（在-2.3%到+4.4%之间）。我们观察到一个一致的模式：归因质量在中间（段落级）粒度处达到峰值——过细的引用似乎切断了支撑论断所需的语义依赖关系，而过粗的引用则……

    arXiv:2604.01432v3 Announce Type: replace  Abstract: Citation granularity -- whether to cite individual sentences, paragraphs, or documents -- is a critical design choice in attributed generation. While fine-grained citations are commonly preferred for precise human verification, their impact on model performance remains under-explored. We analyze four model scales (8B-120B) and demonstrate that enforcing fine-grained (sentence-level) citations forfeits gains of 2-97% (median 40%) relative to the best-performing granularity, and up to 338% on individual tasks. Strikingly, setting citation granularity to its optimal value (based on attribution quality) unlocks these substantial gains while leaving overall answer correctness essentially unchanged (between -2.3% and +4.4%). We observe a consistent pattern where attribution quality peaks at intermediate (paragraph-level) granularities: finer citations appear to sever the semantic dependencies needed to ground a claim, while excessively coa
    
[^311]: 从噪声到信号：当离群点孕育新主题时

    From Noise to Signal: When Outliers Seed New Topics

    [https://arxiv.org/abs/2603.18358](https://arxiv.org/abs/2603.18358)

    该论文提出一种新闻文档轨迹的时间分类法，将“预期性离群点”识别为新兴主题的早期信号而非噪声，并借助十一种语言模型的文档嵌入在法语氢经济新闻语料库上验证了方法的有效性。

    

    在动态主题建模中，离群点通常被视为噪声，然而我们证明其中一些可以作为新兴主题的早期信号。我们引入了一种新闻文档轨迹的时间分类法，定义了文档如何随时间与主题形成相关联。该分类法将“预期性离群点”（即先于其后来加入的主题出现的文档）与那些要么强化现有主题、要么保持孤立的文档区分开来。通过捕捉这些轨迹，该分类法将弱信号检测与时间主题建模联系起来，并阐明了单篇文章如何在不断演化的聚类中预期、发起或漂移。我们在累积聚类设置中，使用来自十一种最先进语言模型的文档嵌入来实现该方法，并在HydroNewsFr（一个关于氢经济的法语新闻语料库）上进行回顾性评估。模型间的一致性揭示了一个小型、高共识的预期性离群点子集，增强了对结果的信心。

    arXiv:2603.18358v2 Announce Type: replace  Abstract: Outliers in dynamic topic modeling are typically treated as noise, yet we show that some can serve as early signals of emerging topics. We introduce a temporal taxonomy of news-document trajectories that defines how documents relate to topic formation over time. It distinguishes anticipatory outliers, which precede the topics they later join, from documents that either reinforce existing topics or remain isolated. By capturing these trajectories, the taxonomy links weak-signal detection with temporal topic modeling and clarifies how individual articles anticipate, initiate, or drift within evolving clusters. We implement it in a cumulative clustering setting using document embeddings from eleven state-of-the-art language models and evaluate it retrospectively on HydroNewsFr, a French news corpus on the hydrogen economy. Inter-model agreement reveals a small, high-consensus subset of anticipatory outliers, increasing confidence in the
    
[^312]: ShapleyLaw：一种基于博弈论的多语言缩放定律方法

    ShapleyLaw: A Game-Theoretic Approach to Multilingual Scaling Laws

    [https://arxiv.org/abs/2603.17945](https://arxiv.org/abs/2603.17945)

    该论文提出ShapleyLaw，将多语言预训练建模为合作博弈，通过Shapley值量化每种语言的跨语言迁移贡献，从而更准确地预测最优语言混合比例。

    

    在多语言预训练中，预训练模型的测试损失在很大程度上受每种语言在预训练数据中所占比例的影响，即语言混合比例。多语言缩放定律可以预测不同语言混合比例下的测试损失，因此可用于估计最优比例。然而，现有的多语言缩放定律方法并未衡量跨语言迁移效应，导致得到的混合比例是次优的。在本文中，我们将多语言预训练视为一个合作博弈，其中每种语言作为一个参与者，共同为预训练做出贡献，并获得由此带来的测试损失下降作为收益。因此，从合作博弈论的视角出发，我们通过每种语言在博弈中的贡献来量化其跨语言迁移效应，并提出了一种基于博弈论的多语言缩放定律，称为ShapleyLaw。

    arXiv:2603.17945v3 Announce Type: replace  Abstract: In multilingual pretraining, the test loss of a pretrained model is heavily influenced by the proportion of each language in the pretraining data, namely the \textit{language mixture ratios}. Multilingual scaling laws can predict the test loss under different language mixture ratios and can therefore be used to estimate the optimal ratios. However, the current approaches to multilingual scaling laws do not measure the \textit{cross-lingual transfer} effect, resulting in suboptimal mixture ratios. In this paper, we consider multilingual pretraining as a cooperative game in which each language acts as a player that jointly contributes to pretraining, gaining the resulting reduction in test loss as the payoff. Consequently, from the perspective of cooperative game theory, we quantify the cross-lingual transfer from each language by its contribution in the game, and propose a game-theoretic multilingual scaling law called \textit{Shapley
    
[^313]: 单个字母中系统性语义结构的证据

    Evidence for systematic semantic structure in individual letters

    [https://arxiv.org/abs/2603.17306](https://arxiv.org/abs/2603.17306)

    该研究首次系统绘制了26个英文字母的多维语义结构，通过三个大语言模型独立检测并由1,388名人类参与者及五种不同语言使用者的预注册实验验证，证明单个字母本身携带可跨语言感知的系统性语义信息。

    

    语音与意义之间的关联已有充分记录，但尚未在整套字母表上进行系统映射。在本文中，我们对26个英文字母进行了映射，发现每个字母都携带一种结构化的、多维度的语义特征，这种特征可以从文本中恢复、能够被跨语言感知，并可由发音特征预测。三个大语言模型独立地在220个字母两两对比中检测到跨越九个感知维度的一致语义结构，随后通过预注册实验对1,388名人类参与者测试了这些恢复出的语义特征。英语母语读者选择预测词汇的比例高于随机水平（85.3%，选定项目；65.7%，所有对比），且这种偏好跟随字母本身而非承载它的词汇。五种类型学上不同语言的使用者也表现出相同的偏好（76.7%，选定配对；68.4%，随机抽取的对比），无论其……

    arXiv:2603.17306v4 Announce Type: replace  Abstract: Associations between speech sounds and meaning are well documented but have not been systematically mapped over a whole alphabet. Here we map them across the 26 English letters and find that each carries a structured, multidimensional semantic profile that is recoverable from text, perceived across languages, and predicted by articulatory features. Three large language models independently detected consistent semantic structure across nine perceptual dimensions in 220 pairwise letter contrasts, and the profiles they recovered were then tested in preregistered experiments with 1,388 human participants. Native English readers chose the predicted word above chance (85.3%, selected items; 65.7%, all contrasts), and the preference followed the letter rather than the words that carried it. Listeners of five typologically diverse languages showed the same preference (76.7%, selected pairs; 68.4%, randomly drawn contrasts), regardless of the
    
[^314]: CCTU：复杂约束下工具使用的基准测试

    CCTU: A Benchmark for Tool Use under Complex Constraints

    [https://arxiv.org/abs/2603.15309](https://arxiv.org/abs/2603.15309)

    CCTU是一个基于12类约束分类体系、包含200个高难度测试用例的基准，并配备可执行的约束验证模块，用于评估大语言模型在复杂约束下的工具使用能力。

    

    在显式约束下通过工具使用来解决问题，对大语言模型（LLM）而言是一个极具挑战性却又无法回避的场景，需要模型具备函数调用、指令遵循和自我改进等能力。然而，由于缺乏专门的评估手段，该领域的进展一直受到阻碍。为解决这一问题，我们提出了CCTU，一个用于评估大语言模型在复杂约束下工具使用能力的基准。CCTU基于一个涵盖四个维度（即资源、行为、工具集和响应）的12类约束分类体系构建。该基准包含200个经过精心筛选且具有挑战性的测试用例，覆盖多样化的工具使用场景，每个用例平均涉及七种约束类型，平均提示长度超过4,700个token。为实现可靠的评估，我们开发了一个可执行的约束验证模块，可进行步骤级验证，并在多轮交互过程中强制约束合规。

    arXiv:2603.15309v2 Announce Type: replace  Abstract: Solving problems through tool use under explicit constraints constitutes a highly challenging yet unavoidable scenario for large language models (LLMs), requiring capabilities such as function calling, instruction following, and self-refinement. However, progress has been hindered by the absence of dedicated evaluations. To address this, we introduce CCTU, a benchmark for evaluating LLM tool use under complex constraints. CCTU is grounded in a taxonomy of 12 constraint categories spanning four dimensions (i.e., resource, behavior, toolset, and response). The benchmark comprises 200 carefully curated and challenging test cases across diverse tool-use scenarios, each involving an average of seven constraint types and an average prompt length exceeding 4,700 tokens. To enable reliable evaluation, we develop an executable constraint validation module that performs step-level validation and enforces compliance during multi-turn interactio
    
[^315]: X-GS：一个基于3D高斯泼溅的可扩展感知与思考框架

    X-GS: An Extensible Framework for Perceiving and Thinking with 3D Gaussian Splatting

    [https://arxiv.org/abs/2603.09632](https://arxiv.org/abs/2603.09632)

    X-GS提出了一个可扩展框架，通过感知器（基于3DGS的在线SLAM与语义蒸馏）和思考器（将VLM与语义高斯对接）两大组件，将原本孤立的3DGS方法整合到视觉语言模型的感知模块中，从而实现3D视觉定位等空间多模态能力。

    

    3D高斯泼溅（3DGS）已成为一种强大的新视角合成技术，随后被扩展到众多空间AI应用中。然而，大多数现有的3DGS方法都是孤立运行的，专注于特定领域。在本文中，我们提出了X-GS，这是一个可扩展的框架，将之前孤立的3DGS方法集成到用于空间任务的视觉语言模型（VLM）的感知模块中，包含两个主要组件：感知器和思考器。感知器执行基于3DGS的在线SLAM并进行语义蒸馏，从未标定位姿的视频流中输出语义高斯。它利用了最新的视觉基础模型以获得更强的几何先验，并且我们引入了三种新颖的优化方法来提高语义蒸馏效率。思考器将各种VLM与这些语义高斯相连接，解锁了空间多模态能力，例如3D视觉定位和……

    arXiv:2603.09632v5 Announce Type: replace-cross  Abstract: 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods operate in isolation, focusing on specific domains. In this paper, we introduce X-GS, an extensible framework that integrates previously isolated 3DGS methods into the perception module of a VLM for spatial tasks, with two major components: the $\textit{Perceiver}$ and the $\textit{Thinker}$. The $\textit{Perceiver}$ performs online 3DGS-based SLAM with semantic distillation and outputs semantic Gaussians from unposed video streams. It leverages recent vision foundation models for stronger geometric priors, and we introduce three novel optimizations to improve semantic distillation efficiency. The $\textit{Thinker}$ interfaces diverse VLMs with these semantic Gaussians, unlocking spatial multimodal capabilities such as 3D visual grounding and
    
[^316]: Dial：一个基于知识的方言特定NL2SQL系统

    Dial: A Knowledge-Grounded Dialect-Specific NL2SQL System

    [https://arxiv.org/abs/2603.07449](https://arxiv.org/abs/2603.07449)

    Dial是一个知识驱动的方言特定NL2SQL框架，通过方言感知的逻辑查询规划模块和分层意图知识库，解决了异构数据库系统中生成既语义正确又可在目标引擎上执行的SQL查询这一难题。

    

    企业通常部署异构数据库系统，每个系统拥有各自不同的SQL方言，具有不同的语法规则、内置函数和执行约束。然而，大多数现有的NL2SQL方法假设单一的标准方言（例如SQLite），难以生成既语义正确又可在目标引擎上执行的查询。基于提示的方法将意图推理与方言语法紧密耦合，基于规则的翻译器往往将原生算子退化为通用构造，而多方言微调则存在跨方言干扰问题。在本文中，我们提出了Dial，一个面向方言特定NL2SQL的知识驱动框架。Dial引入了：(1) 方言感知逻辑查询规划模块，通过算子级意图分解和分歧感知规范，将自然语言转换为方言感知的逻辑查询计划；(2) HINT-KB，一个分层意图感知的……（摘要内容至此截断）

    arXiv:2603.07449v2 Announce Type: replace-cross  Abstract: Enterprises commonly deploy heterogeneous database systems, each of which owns a distinct SQL dialect with different syntax rules, built-in functions, and execution constraints. However, most existing NL2SQL methods assume a single canonical dialect (e.g., SQLite) and struggle to produce queries that are both semantically correct and executable on target engines. Prompt-based approaches tightly couple intent reasoning with dialect syntax, rule-based translators often degrade native operators into generic constructs, and multi-dialect fine-tuning suffers from cross-dialect interference.   In this paper, we present Dial, a knowledge-grounded framework for dialect-specific NL2SQL. Dial introduces: (1) a Dialect-Aware Logical Query Planning module that converts natural language into a dialect-aware logical query plan via operator-level intent decomposition and divergence-aware specification; (2) HINT-KB, a hierarchical intent-aware
    
[^317]: MedGPT-oss：训练面向生物医学的通用视觉-语言模型

    MedGPT-oss: Training a General-Purpose Vision-Language Model for Biomedicine

    [https://arxiv.org/abs/2603.00842](https://arxiv.org/abs/2603.00842)

    MEDGPT-OSS是一个开放权重的200亿参数通用生物医学视觉-语言模型，通过三阶段训练课程、严格数据筛选和长上下文多模态对齐，在分布外多模态推理和临床文本任务上超越更大的开源医学模型，同时支持满足隐私合规的本地化部署。

    

    生物医学多模态助手有潜力统一放射学、病理学和临床文本推理，然而一个关键的部署差距仍然存在：表现最佳的系统要么是闭源的，要么计算成本过高，无法满足患者隐私和受保护健康信息（PHI）合规所需的本地化部署。我们提出了MEDGPT-OSS，这是一个开放权重的200亿参数通用视觉-语言模型，旨在促进临床AI的开放研究。MEDGPT-OSS并不依赖架构的复杂性，而是通过优化的三阶段训练课程将GPT-oss语言骨干与视觉前端相结合。通过严格的数据筛选和长上下文多模态对齐，逐步对这些模块进行领域适配，我们证明了200亿参数的模型能够弥合能力差距。它在分布外（OOD）多模态推理和复杂的纯文本临床任务上成功超越了更大的开源医学模型。

    arXiv:2603.00842v2 Announce Type: replace  Abstract: Biomedical multimodal assistants have the potential to unify radiology, pathology, and clinical-text reasoning, yet a critical deployment gap remains: top-performing systems are either closed-source or computationally prohibitive, precluding the on-premises deployment required for patient privacy and PHI compliance. We introduce MEDGPT-OSS, an open-weight, 20B-parameter generalist vision-language model designed to facilitate open research in clinical AI. Rather than relying on architectural complexity, MEDGPT-OSS pairs the GPT-oss language backbone with a visual front-end via a optimized, three-stage training curriculum. By progressively domain-adapting these modules through rigorous data curation and long-context multimodal alignment, we demonstrate that a 20B model can bridge the capacity gap. It successfully outperforms larger open medical models on out-of-distribution (OOD) multimodal reasoning and complex text-only clinical task
    
[^318]: Althea：AI辅助验证中的事实核查—元学习权衡

    Althea: The Fact-Checking--Metalearning Tradeoff in AI-Assisted Verification

    [https://arxiv.org/abs/2602.11161](https://arxiv.org/abs/2602.11161)

    本文提出检索增强事实核查系统Althea，并通过N=961的纵向“消退测试”实验揭示了“事实核查—元学习权衡”：AI辅助干预虽能即时提升用户的准确率与置信度，但系统移除后用户无法将验证能力迁移到新声明上，而自主搜索所培养的能力则能持久保持优势。

    

    事实核查系统必须具备可扩展性且在认识论上值得信赖。我们提出了Althea，一个面向用户驱动声明评估的检索增强系统，它在AVeriTeC基准上与标准流水线表现相当，同时改进了对“支持/反驳”声明的判别能力。一项纵向调查实验（N=961）将十天的随访视为一种“消退测试”：在让用户体验一个验证流程后，我们移除该系统，考察用户能否在无辅助的情况下复现该流程，从而测试的是元学习能力而非一次性准确率。我们比较了两种AI辅助干预——探索式（引导推理）与摘要式（综合判定结论）——以及两种基线（无关新闻与自主搜索）。两种AI干预带来了最强的即时准确率与置信度提升，但未能通过消退测试：在未见过的声明上，它们的表现并不优于新闻基线，而无需经历消退过程的自主搜索则保持了显著优势。这揭示了一种事实核查—元学习权衡：……（原文摘要在此处截断）

    arXiv:2602.11161v3 Announce Type: replace-cross  Abstract: Fact-checking systems must be scalable and epistemically trustworthy. We introduce Althea, a retrieval-augmented system for user-driven claim evaluation that matches standard pipelines on AVeriTeC while improving supported/refuted discrimination. A longitudinal survey experiment (N=961) treats a ten-day follow-up as a fading test: after modeling a verification procedure, we remove the system and ask whether users reproduce it unaided, testing metalearning rather than one-time accuracy. We compare two AI-assisted treatments, Exploratory (guided reasoning) and Summary (synthesized verdicts), against two baselines, unrelated news and Self-search. The treatments yield the strongest immediate accuracy and confidence gains but do not survive the fading test: on unseen claims they perform no better than news, while Self-search, with no procedure to fade, retains a large advantage. This reveals a factchecking-metalearning tradeoff: con
    
[^319]: 一种用于内涵形式语义学的向量逻辑

    A vector logic for intensional formal semantics

    [https://arxiv.org/abs/2602.02940](https://arxiv.org/abs/2602.02940)

    本文证明了Kripke式内涵形式语义模型可以单射嵌入向量空间——原始域映射为自由载体、内涵函数映射为线性算子且保持复合运算——并完整刻画了哪些布尔值泛函能够线性地作用于算子编码。

    

    形式语义学与分布语义学是研究语言意义的两种不同方法：前者通过模型论结构将意义建模为指称；后者则将意义视为由用法塑造的高维空间中的向量。本文确定了内涵形式语义学的哪一部分可以被线性向量空间编码。带有任意有限索引类别集合（收集于复合索引空间中）的Kripke式内涵模型可以单射地嵌入向量空间：原始域映射到自由载体；内涵及其他函数映射到线性算子。语义函数可提升为自由载体上的唯一多重线性映射，且复合运算得以保持。函数域的算子编码会压缩其自由载体，我们对泛函进行了刻画：幂集上的布尔值泛函当且仅当它是常值函数、超滤子指示函数或其补函数时，才能线性地作用于算子编码。

    arXiv:2602.02940v2 Announce Type: replace-cross  Abstract: Formal semantics and distributional semantics are distinct approaches to linguistic meaning: the former models meaning as reference via model-theoretic structures; the latter as vectors in high-dimensional spaces shaped by usage. This paper establishes which part of intensional formal semantics admits a linear vector-space encoding. Kripke-style intensional models, with any finite collection of index sorts collected in a compound index space, embed injectively into vector spaces: primitive domains go to free carriers; intensions and other functions go to linear operators. Semantic functions lift to unique multilinear maps on the free carriers, and composition is preserved. The operator encoding of a function domain compresses its free carrier, and we characterize the functionals: a Boolean-valued functional of a power set acts linearly on operator encodings exactly when it is constant, an ultrafilter indicator, or the complemen
    
[^320]: 面向大语言模型上下文感知推理时控制的导向向量场

    Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models

    [https://arxiv.org/abs/2602.01654](https://arxiv.org/abs/2602.01654)

    提出导向向量场（SVF），通过学习可微的概念评分函数生成随上下文自适应的导向向量，解决了静态导向向量因方向固定而在不同上下文中失效的问题。

    

    导向向量（SVs）通过偏移隐藏激活，为在推理时控制大语言模型（LLMs）提供了一种轻量级方法，在提示工程和微调之间提供了实用的折中方案。然而，SVs在实践中可能不可靠：有些概念无法被导向，即使导向在平均上有所帮助，对于相当一部分输入也可能产生反效果；在长文本生成和多属性导向场景下，可靠性也会下降。我们从几何视角审视这些失败现象。静态SV在表示空间的每个位置都施加相同的更新向量，隐含假设改善概念的方向在所有上下文中都是恒定的。当局部有效的方向随当前激活而变化时，单一的全局向量可能会失准，导致效果微弱甚至反转。基于这一视角，我们提出了导向向量场（SVF），它学习一个可微的概念评分函数……（摘要在此处截断）

    arXiv:2602.01654v2 Announce Type: replace  Abstract: Steering vectors (SVs) offer a lightweight way to control large language models (LLMs) at inference time by shifting hidden activations, providing a practical middle ground between prompting and fine-tuning. Yet SVs can be unreliable in practice. Some concepts are unsteerable, and even when steering helps on average it can backfire for a non-trivial fraction of inputs. Reliability also degrades in long-form generation and multi-attribute steering. We take a geometric view of these failures. A static SV applies the same update vector everywhere in representation space, implicitly assuming that the concept-improving direction is constant across contexts. When the locally effective direction varies with the current activation, a single global vector can become misaligned, which yields weak or reversed effects. Guided by this perspective, we propose Steering Vector Fields (SVF), which learns a differentiable concept scoring function whos
    
[^321]: 数据集语言结构在大语言模型文化意识中的作用

    The Role of Dataset Linguistic Structure in the Cultural Awareness of Large Language Models

    [https://arxiv.org/abs/2602.01161](https://arxiv.org/abs/2602.01161)

    该研究提出以数据集为中心的文化对齐视角，通过对阿拉伯语、中文和日语微调数据集的语言、语义与结构指标进行PCA分析，提炼出语义结构、多样性和语言特定组织三个可解释主轴，揭示了后训练数据的语言结构特性与大语言模型文化表现之间的关联，从而可在微调前指导数据选择。

    

    arXiv:2602.01161v2 公告类型：替换 摘要：大语言模型（LLMs）的全球部署引发了关于文化错位的担忧，然而用于文化适应的微调数据集的语言特性至今仍缺乏深入理解。我们采用以数据集为中心的文化对齐视角，研究后训练数据的哪些属性与文化表现相关、这些属性能否在微调之前指导数据选择，以及它们的影响在不同语言和模型家族之间如何变化。我们为阿拉伯语、中文和日语数据集计算了轻量级的语言、语义和结构指标，并在每种语言内部分别应用主成分分析（PCA）。由此得到的主成分形成了具有广泛可解释性的轴：PC1通常由语义结构主导，PC2捕捉多样性与词汇变化，PC3则反映了更多语言特定的组织方式。我们微调了LLaMA、Mistral和DeepSeek模型并进行评估……

    arXiv:2602.01161v2 Announce Type: replace  Abstract: The global deployment of large language models (LLMs) has raised concerns about cultural misalignment, yet the linguistic properties of fine-tuning datasets used for cultural adaptation remain poorly understood. We adopt a dataset-centric view of cultural alignment and investigate which properties of post-training data are associated with cultural performance, whether they can guide data selection before fine-tuning, and how their effects vary across languages and model families. We compute lightweight linguistic, semantic, and structural metrics for Arabic, Chinese, and Japanese datasets and apply principal component analysis (PCA) separately within each language. The resulting components form broadly interpretable axes: PC1 is generally dominated by semantic structure, PC2 captures diversity and lexical variation, and PC3 reflects more language-specific organization. We fine-tune LLaMA, Mistral, and DeepSeek models and evaluate the
    
[^322]: 超越遗忘：表征误导引发可控的附带行为与能力

    Beyond Forgetting: Representation Misdirection Elicits Controllable Side Behaviors and Capabilities

    [https://arxiv.org/abs/2601.21702](https://arxiv.org/abs/2601.21702)

    该论文提出，基于线性表征假设的视角，表征误导（RM）类机器遗忘方法不仅能实现遗忘，还可通过在遗忘表征空间中对高层概念向量进行线性操作，可控地引发与该概念相对应的附带行为与能力。

    

    我们研究表征误导（Representation Misdirection, RM），这是一类大语言模型（LLM）遗忘方法，其通过将待遗忘样本的潜在表征重定向至某个目标向量来实现遗忘。尽管十分重要，但RM中所使用的目标向量的作用仍未得到充分探索。本文从线性表征假设（Linear Representation Hypothesis）的视角出发，重新审视并研究RM。具体而言，如果能够识别出与某一高层概念相对应的一维表征，线性表征假设便允许在遗忘表征空间内对该概念向量进行线性操作。基于这一视角，我们假设：除遗忘之外，通过RM实现的机器遗忘还会引发与该高层概念相对应的、可控的附带行为与能力。我们的假设在广泛的概念与任务上得到了实证验证，包括控制已遗忘模型的（原文摘要在此处截断）

    arXiv:2601.21702v4 Announce Type: replace-cross  Abstract: We consider Representation Misdirection (RM), a class of large language model (LLM) unlearning methods that achieve forgetting by redirecting the latent representations of forget-samples toward a target vector. Despite being important, the roles of the target vector used in RM, however, remain underexplored. Here, we approach and revisit RM through the lens of the Linear Representation Hypothesis. Specifically, if one can identify a one-dimensional representation corresponding to a high-level concept, the Linear Representation Hypothesis enables linear operations on this concept vector within the forget-representation space. Under this view, we hypothesize that, beyond forgetting, machine unlearning via RM elicits controllable side effect behaviors and capabilities corresponding to the high-level concept. Our hypothesis is empirically validated across a wide range of concepts and tasks, including controlling unlearned models' t
    
[^323]: ILRR：掩码扩散语言模型的推理时引导方法

    ILRR: Inference-Time Steering Method for Masked Diffusion Language Models

    [https://arxiv.org/abs/2601.21647](https://arxiv.org/abs/2601.21647)

    ILRR提出了一种推理时引导框架，将参考文本作为高层语义蓝图，通过迭代精炼潜在表示并将其注入生成序列的激活中，实现对掩码扩散语言模型的可控生成，并能调节引导强度使短参考文本引导长文本生成。

    

    离散扩散语言模型（DLMs）为文本生成提供了一种有前景的非自回归替代方案，然而有效的推理时控制机制仍然相对缺乏探索。现有方法包括采样层面的引导或轨迹优化机制。在这项工作中，我们研究了基于参考文本的DLM潜空间引导范式。我们提出了迭代潜在表示精炼，这是一个高效的框架，它将参考文本作为高层语义蓝图来引导DLM。ILRR从参考文本中提取语义信号，并将其注入到生成序列不断演化的激活中，从而实现情感等粗粒度属性的可调迁移。我们进一步提出了空间调制引导，这一扩展方法通过调节引导强度在整个序列上的分布，使长文本生成能够由更短的参考文本来引导。实证结果表明，ILRR实现了有效的控制效果

    arXiv:2601.21647v2 Announce Type: replace  Abstract: Discrete Diffusion Language Models (DLMs) offer a promising non-autoregressive alternative for text generation, yet effective mechanisms for inference-time control remain relatively underexplored. Existing approaches include sampling-level guidance or trajectory optimization mechanisms. In this work, we study the paradigm of reference-based latent steering for DLMs. We introduce Iterative Latent Representation Refinement (ILRR), an efficient framework for steering DLMs using a reference text as a high-level semantic blueprint. ILRR extracts and injects reference-derived semantic signals into the evolving activations of the generated sequence, enabling tunable transfer of coarse properties such as sentiment. We further introduce Spatially Modulated Steering, an extension that enables long-form generation to be guided by shorter references by regulating intensity across the sequence. Empirically, we demonstrate that ILRR achieves effec
    
[^324]: 可重放的金融智能体：面向工具使用型LLM智能体的确定性-忠实性保证框架

    Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents

    [https://arxiv.org/abs/2601.15322](https://arxiv.org/abs/2601.15322)

    该论文提出DFAH框架，首次将工具使用型LLM智能体的决策可重复性、轨迹一致性和证据条件忠实性作为三个独立维度加以区分评估，并澄清了历史研究中r=-0.11相关性统计量的性质与局限。

    

    使用工具的智能体可以在改变其记录的执行过程的同时重复出相同的最终决策。我们提出了确定性-忠实性保证框架，这是一个能够区分决策可重复性、轨迹一致性和证据条件忠实性的评估框架。任务正确性需要单独的合格标签与评估；而证据条件忠实性在历史上的v2智能体实验中并未被评估。原始v2研究在三个合成金融任务中报告了4,705次智能体运行，并在21个模型-基准配置摘要中报告了决策确定性与任务标签匹配之间的相关性r = -0.11。该统计数据可以从历史配置表中复现，但其中包含了一个后来被排除的投资组合测试装置。该统计量被保留作为历史描述，而非统计独立性、预测无用性或架构层面确定性-准确性权衡的证据。

    arXiv:2601.15322v3 Announce Type: replace-cross  Abstract: Tool-using agents can repeat a final decision while changing their recorded execution. We introduce the Determinism-Faithfulness Assurance Harness (DFAH), a framework that distinguishes decision repeatability, trajectory agreement, and evidence-conditioned faithfulness. Task correctness requires separately qualified labels and evaluation; evidence-conditioned faithfulness was not evaluated in the historical v2 agentic experiments.   The original v2 study reported 4,705 agentic runs in three synthetic financial tasks and a decision-determinism/task-label-match correlation of r = -0.11 across 21 model-benchmark configuration summaries. This statistic is reproducible from the historical configuration table, but includes a subsequently excluded portfolio fixture. It is retained as a historical description, not evidence of statistical independence, predictive uselessness, or an architectural determinism-accuracy tradeoff. Recorded d
    
[^325]: 大语言模型智能体推理综述：迈向递归自我改进与集体智能体

    A Survey of Agentic Reasoning for Large Language Models: Towards Recursively Self-Improving and Collective Agents

    [https://arxiv.org/abs/2601.12538](https://arxiv.org/abs/2601.12538)

    本综述从基础智能体推理、自进化智能体推理和集体多智能体推理三个互补维度，系统梳理了大语言模型智能体推理的研究进展，旨在迈向能够递归自我改进与协作的智能体。

    

    推理是支撑推断、问题求解与决策制定的基础认知过程。尽管大语言模型（LLMs）在封闭世界环境中展现出强大的推理能力，但它们在开放和动态的环境中仍然表现不佳。智能体推理标志着一种范式转变，它将大语言模型重新定义为能够通过持续交互进行规划、行动和学习的自主智能体。在这篇综述中，我们从三个互补的维度来组织智能体推理。首先，我们通过三个层次来刻画环境动态性：基础智能体推理，它建立了稳定环境中的核心单智能体能力，包括规划、工具使用和搜索；自进化智能体推理，研究智能体如何通过反馈、记忆和适应来改进这些能力；以及集体多智能体推理，将智能扩展到涉及协调与知识共享的协作场景中。

    arXiv:2601.12538v2 Announce Type: replace-cross  Abstract: Reasoning is a fundamental cognitive process underlying inference, problem-solving, and decision-making. While large language models (LLMs) demonstrate strong reasoning capabilities in closed-world settings, they struggle in open-ended and dynamic environments. Agentic reasoning marks a paradigm shift by reframing LLMs as autonomous agents that plan, act, and learn through continual interaction. In this survey, we organize agentic reasoning along three complementary dimensions. First, we characterize environmental dynamics through three layers: foundational agentic reasoning, which establishes core single-agent capabilities including planning, tool use, and search in stable environments; self-evolving agentic reasoning, which studies how agents refine these capabilities through feedback, memory, and adaptation; and collective multi-agent reasoning, which extends intelligence to collaborative settings involving coordination, kno
    
[^326]: 大型语言模型中的垃圾注意力：BOS汇聚头与汇聚感知剪枝

    Garbage Attention in Large Language Models: BOS Sink Heads and Sink-aware Pruning

    [https://arxiv.org/abs/2601.06787](https://arxiv.org/abs/2601.06787)

    本论文发现LLM中高BOS汇聚分数的注意力头是功能冗余的“垃圾场”，尤其在深层中，据此提出移除这些头的简单剪枝策略，在Gemma-3、Llama-3.1和Qwen3上比基于权重和激活的剪枝准则更可靠地识别冗余组件并保持下游任务性能。

    

    众所周知，大型语言模型（LLM）包含显著的冗余性，但对于为什么某些组件（尤其是较高层中的组件）更具冗余性，一直缺乏系统性的解释。在本工作中，我们将BOS汇聚现象确定为驱动这种逐层敏感性的关键机制。我们表明，具有高BOS汇聚分数的注意力头与功能冗余密切相关：这类注意力头（尤其是在较深层中）对预测性能的贡献甚微，实际上充当了多余注意力权重的“垃圾场”。利用这一洞察，我们提出了一种简单的剪枝策略，即移除高BOS汇聚的注意力头。在Gemma-3、Llama-3.1和Qwen3上的实验表明，在下游任务性能保持方面，该方法比基于权重和激活的准则更能可靠地识别冗余的Transformer组件，在低到中等剪枝率下仍能保持接近稠密基线的性能。

    arXiv:2601.06787v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are known to contain significant redundancy, yet a systematic explanation for why certain components, particularly in higher layers, are more redundant has remained elusive. In this work, we identify the BOS sink phenomenon as a key mechanism driving this layer-wise sensitivity. We show that attention heads with high BOS sink scores are strongly associated with functional redundancy: such heads, especially in deeper layers, contribute little to predictive performance and effectively serve as dumping grounds for superfluous attention weights. Leveraging this insight, we introduce a simple pruning strategy that removes high-BOS sink heads. Experiments on Gemma-3, Llama-3.1, and Qwen3 demonstrate that this approach identifies redundant transformer components more reliably than weight- and activation-based criteria in terms of downstream task retention, remaining close to dense baselines at low-to-moderate pr
    
[^327]: RADAR：用于自适应LLM生成假新闻检测的检索增强检测器与对抗精炼方法

    RADAR: Retrieval-Augmented Detector with Adversarial Refinement for Adaptive LLM-Generated Fake News Detection

    [https://arxiv.org/abs/2601.03981](https://arxiv.org/abs/2601.03981)

    RADAR通过生成器与检测器的对抗共同进化（借助语言对抗反馈VAF）以及双侧检索增强机制，实现了对LLM生成假新闻的自适应检测，性能超越现有检索增强基线和通用大语言模型。

    

    为了有效打击新闻领域LLM生成的错误信息的传播，我们提出了RADAR，一个带有对抗精炼机制的检索增强检测器，用于自适应的LLM生成假新闻检测。我们的方法采用一个生成器，通过事实性扰动重写真实文章，并配合一个轻量级检测器，利用密集段落检索来验证声明。为了实现有效的共同进化，我们引入了语言对抗反馈（VAF）。VAF不依赖于标量奖励，而是发出结构化的自然语言批评；这些批评引导生成器进行更复杂的逃避尝试，从而迫使检测器不断适应和改进。在LLM生成假新闻基准上的实验表明，RADAR优于检索增强的可训练基线以及带检索的通用大语言模型。进一步分析显示，在生成器和检测器两侧都引入检索能够提升性能，而VAF…

    arXiv:2601.03981v3 Announce Type: replace  Abstract: To efficiently combat the spread of LLM-generated misinformation in the news domain, we present RADAR, a Retrieval-Augmented Detector with Adversarial Refinement for adaptive LLM-generated fake news detection. Our approach employs a generator that rewrites real articles with factual perturbations, paired with a lightweight detector that verifies claims using dense passage retrieval. To enable effective co-evolution, we introduce Verbal Adversarial Feedback (VAF). Rather than relying on scalar rewards, VAF issues structured natural-language critiques; these guide the generator toward more sophisticated evasion attempts, compelling the detector to adapt and improve. Experiments on an LLM-generated fake news benchmark show that RADAR outperforms retrieval-augmented trainable baselines and general-purpose LLMs with retrieval. Further analysis shows that retrieval on both the generator and detector sides improves performance, while VAF an
    
[^328]: 氛围编程安全吗？真实世界任务中智能体生成代码漏洞的基准评估

    Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks

    [https://arxiv.org/abs/2512.03262](https://arxiv.org/abs/2512.03262)

    该论文提出SUSVIBES基准，评估了12种编码智能体在真实任务中的安全性，发现所有智能体生成代码的安全率极低（最高仅11.8%），且简单安全提示无法有效改善。

    

    arXiv:2512.03262v3 公告类型：交叉替换 摘要：氛围编程是一种新的软件开发范式，在这种范式中，人类工程师提示大型语言模型（LLM）智能体在极少监督下完成复杂的编码任务。尽管氛围编程日益被采用，但生成的代码在生产环境中部署真的安全吗？为了探究这一问题，我们提出了SUSVIBES基准，该基准包含来自真实世界开源项目的186个功能请求软件工程任务，针对这些任务，人类程序员提交了存在漏洞的实现。我们在该基准上评估了12种广泛使用的编码智能体设置，并采用了前沿模型。令人不安的是，所有智能体在软件安全方面表现不佳。尽管来自SWE-Agent与Claude 4 Sonnet的解决方案中57%在功能上正确，但只有11.8%是安全的。进一步实验表明，初步安全策略，例如在功能请求中添加漏洞提示，无法缓解这些问题。

    arXiv:2512.03262v3 Announce Type: replace-cross  Abstract: Vibe coding is a new software development paradigm in which human engineers prompt a large language model (LLM) agent to complete complex coding tasks with little supervision. Although vibe coding is increasingly adopted, is the generated code really safe to deploy in production? To investigate this question, we propose SUSVIBES, a benchmark consisting of 186 feature-request software engineering tasks from real-world open-source projects, for which, human programmers committed vulnerable implementations. We evaluate 12 widely used coding agentic settings with frontier models on the benchmark. Disturbingly, all agents perform poorly in terms of software security. Although 57% of the solutions from SWE-Agent with Claude 4 Sonnet are functionally correct, only 11.8% are secure. Further experiments demonstrate that preliminary security strategies, such as augmenting the feature request with vulnerability hints, cannot mitigate thes
    
[^329]: BudgetMem：面向语言模型低成本长上下文处理的无训练选择性记忆

    BudgetMem: Training-Free Selective Memory for Cost-Efficient Long-Context Processing in Language Models

    [https://arxiv.org/abs/2511.04919](https://arxiv.org/abs/2511.04919)

    BudgetMem是一种无需训练的长上下文处理架构，通过实体密度、TF-IDF等可解释特征在显式内存预算下进行块级保留或丢弃决策，在丢弃70%内容块的同时保持与未压缩基线相当的性能，并大幅优于词元级压缩方法LLMLingua-2。

    

    使用大语言模型（LLM）处理长文档代价高昂：对一份10万词元文档进行单次查询，根据模型不同，API费用可能从几十美分到超过一美元不等，且内存随上下文长度线性增长。我们提出BudgetMem，一种无需训练的架构，在显式内存预算下仅保留高显著性内容。与LLMLingua等词元级神经压缩器不同，BudgetMem基于可解释特征做出块级的保留或丢弃决策，这些特征包括：实体密度、TF-IDF重要性、位置、数字密度、话语标记和问题存在性。在四个基准测试中，BudgetMem在模板生成的结构化文档上与未压缩基线表现相当（F1 = 0.859 vs. 0.855），同时丢弃了70%的内容块。作为相同文档上的检索前过滤器，它大幅优于LLMLingua-2（0.859 vs. 0.554），因为词元级压缩会破坏短语结构……

    arXiv:2511.04919v3 Announce Type: replace  Abstract: Processing long documents with large language models (LLMs) is expensive: a single query over a 100K-token document can cost from tens of cents to over a dollar in API fees, depending on the model, and memory grows linearly with context length. We introduce BudgetMem, a training-free architecture that keeps only high-salience content under an explicit memory budget. Unlike token-level neural compressors such as LLMLingua, BudgetMem makes chunk-level keep-or-discard decisions from interpretable features: entity density, TF-IDF importance, position, numerical density, discourse markers, and question presence. Across four benchmarks, BudgetMem matches the uncompressed baseline on template-generated structured documents (F1 = 0.859 vs. 0.855) while discarding 70% of chunks. As a pre-retrieval filter on the same documents it outperforms LLMLingua-2 by a wide margin (0.859 vs. 0.554), because token-level compression destroys the phrasal st
    
[^330]: 用于更好训练和评估知识图谱增强大语言模型的真值子图

    Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs

    [https://arxiv.org/abs/2511.04473](https://arxiv.org/abs/2511.04473)

    提出了SynthKGQA框架，可从任意知识图谱生成包含完整真值事实的知识图谱问答数据集，既能更有效地评估知识图谱检索器，也能用于训练更好的知识图谱增强大语言模型，并基于Wikidata构建了测试零样本泛化能力的GTSQA数据集。

    

    从图结构知识库中检索信息是提高大语言模型事实性的一个有前景的方向。尽管已提出多种解决方案，但由于缺乏带有图检索真值目标的具有挑战性的问答数据集，方法之间的比较十分困难。我们提出了SynthKGQA，这是一个由大语言模型驱动的框架，可以从任何知识图谱生成高质量的知识图谱问答数据集，并提供知识图谱中用于推理问题的完整真值事实集合。我们展示了SynthKGQA生成的数据除了能够对知识图谱检索器进行更有信息量的基准测试外，还能用于训练更好的模型。我们将SynthKGQA应用于Wikidata，生成了GTSQA——一个旨在测试知识图谱检索器对未见过的图结构和关系类型的零样本泛化能力的新数据集，并在其上对流行的知识图谱增强大语言模型解决方案进行了基准测试。

    arXiv:2511.04473v3 Announce Type: replace-cross  Abstract: Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, an LLM-powered framework for generating high-quality Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over questions. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models.We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it.
    
[^331]: M-CIF：基于CIF的非自回归语音识别的多尺度对齐

    M-CIF: Multi-Scale Alignment For CIF-Based Non-Autoregressive ASR

    [https://arxiv.org/abs/2510.22172](https://arxiv.org/abs/2510.22172)

    提出多尺度CIF（M-CIF）机制，通过将字符级和音素级监督逐步蒸馏到子词表示中实现多层级对齐，显著提升了非自回归语音识别在德语、法语等语言上的稳定性，在CommonVoice上德语WER降低4.21%、法语降低3.05%。

    

    连续积分-触发（CIF）机制为非自回归（NAR）语音识别提供了有效的对齐。该机制构建了从声学特征到目标token的平滑且单调的映射，在中文上取得了与其他NAR方法相当的性能。然而，在缺乏更细粒度指导的情况下，其在英语和法语等某些语言中的稳定性会下降。本文提出了多尺度CIF（M-CIF），通过将字符级和音素级监督逐步蒸馏到子词表示中，实现多层级对齐，从而增强鲁棒的声学-文本对齐。实验表明，与Paraformer基线相比，M-CIF降低了词错误率（WER），特别是在CommonVoice数据集上，德语降低了4.21%，法语降低了3.05%。为进一步分析这些增益，我们定义了语音混淆错误（PE）和与空格相关的分割错误（SE）作为评估指标。

    arXiv:2510.22172v2 Announce Type: replace-cross  Abstract: The Continuous Integrate-and-Fire (CIF) mechanism provides effective alignment for non-autoregressive (NAR) speech recognition. This mechanism creates a smooth and monotonic mapping from acoustic features to target tokens, achieving performance on Mandarin competitive with other NAR approaches. However, without finer-grained guidance, its stability degrades in some languages such as English and French. In this paper, we propose Multi-scale CIF (M-CIF), which performs multi-level alignment by integrating character and phoneme level supervision progressively distilled into subword representations, thereby enhancing robust acoustic-text alignment. Experiments show that M-CIF reduces WER compared to the Paraformer baseline, especially on CommonVoice by 4.21% in German and 3.05% in French. To further investigate these gains, we define phonetic confusion errors (PE) and space-related segmentation errors (SE) as evaluation metrics. An
    
[^332]: BreakFun：通过模拟代码执行下的对象实例化对大语言模型进行越狱攻击

    BreakFun: Jailbreaking LLMs via Object Instantiation under Simulated Code Execution

    [https://arxiv.org/abs/2510.17904](https://arxiv.org/abs/2510.17904)

    BreakFun通过让模型模拟执行包含“特洛伊模式”的良性Python代码并实例化对象，使有害内容作为代码运行的副作用被生成，从而在13个模型上实现平均89%的越狱攻击成功率。

    

    arXiv:2510.17904v3 公告类型： replace-cross 摘要：大语言模型（LLM）因其在处理结构、语法和代码方面的出色能力而被广泛使用，但恰恰是这种能力也使其产生了悖论式的脆弱性。我们提出了BreakFun，一种将有害请求包装为代码执行模拟的越狱方法。该提示词向模型提供一个良性的Python类定义，即“特洛伊模式”，并询问如果该代码运行将会输出什么。为了回答这个问题，模型必须根据该类创建一个对象并为每个字段虚构一个值，而对抗性的字段名会将这些值引导向攻击者的目标。因此，有害内容是作为模拟实例化的副作用出现的，而非直接回答。该攻击由三部分提示词承载：无辜的框架、特洛伊模式和思维链干扰。在JailbreakBench上，BreakFun在13个开源权重和商业模型上达到了平均89%的攻击成功率（开源权重模型约98%，API系统……）

    arXiv:2510.17904v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are widely used because they process structures, syntax and code well, but this same ability also makes them paradoxically vulnerable. We introduce BreakFun, a jailbreak method that frames a harmful request as code-execution simulation. The prompt gives the model a benign Python class definition, the "Trojan Schema", and asks what that code would print if it ran. To answer, the model must create an object from the class and invent a value for each field, and the adversarial field names steer those values toward the attacker's goal. The harmful content therefore appears as a side-effect of the simulated instantiation, not as a direct answer. A three-part prompt carries the attack: an innocent frame, the Trojan Schema, and a Chain-of-Thought distraction. On JailbreakBench, BreakFun reaches an average attack success rate of 89% across 13 open-weight and commercial models (open-weight ~98%, API systems 
    
[^333]: 寻找你的最优教师：基于路由器引导的多教师蒸馏的个性化数据合成

    Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation

    [https://arxiv.org/abs/2510.10925](https://arxiv.org/abs/2510.10925)

    提出PerSyn个性化数据合成策略，采用“先路由后生成”范式，通过综合考虑学生可学习性与教师响应质量的查询级路由器，为每个提示匹配最优教师，从而更高效地为学生模型定制训练数据。

    

    使用强教师模型生成的合成数据来训练学生模型，是蒸馏教师能力的一种有前景的方法。然而，近期研究表明，更强的模型并不总是最优的教师，这揭示了教师输出与学生可学习性之间的不匹配。为解决这一问题，我们提出了PerSyn（个性化数据合成），一种在全新的“先路由后生成”范式下运行的新型合成策略，可为每个学生模型量身定制数据，使其能够更有效地学习。具体而言，PerSyn首先通过一个查询级路由器将每个提示分配给其最优教师，该路由器综合考虑学生的可学习性和教师的响应质量。随后，每位教师仅为其被分配的提示合成数据，这使得该过程比传统的“先生成后选择”范式更加高效——在传统范式中，所有教师必须为整个提示集并行生成响应。

    arXiv:2510.10925v3 Announce Type: replace-cross  Abstract: Training student models on synthetic data generated by strong teacher models is a promising way to distilling the capabilities of teachers. However, recent studies show that stronger models are not always optimal teachers, revealing a mismatch between teacher outputs and student learnability. To address this issue, we propose PerSyn (Personalized data Synthesis), a novel synthesis strategy that operates under a new ``Route then Generate'' paradigm to create data tailored to each student model, enabling it to learn more effectively. Specifically, PerSyn first assigns each prompt to its optimal teacher via a query-level router that jointly considers student learnability and teacher response quality. Each teacher then synthesizes data only for its assigned prompts, making the process more efficient than the conventional ``Generate then Select'' paradigm, where all teachers must generate parallel responses for the entire prompt set
    
[^334]: 大语言模型预训练与后训练数据中的政治内容是什么？

    What Is The Political Content in LLMs' Pre- and Post-Training Data?

    [https://arxiv.org/abs/2509.22367](https://arxiv.org/abs/2509.22367)

    该研究首次量化了大语言模型预训练与后训练数据中的政治内容，发现所有训练数据集均系统性地偏向左翼内容，且数据中的政治立场与模型的政治行为表现密切相关。

    

    大语言模型（LLM）在生成的文本中会反映出带有政治倾向的观点。尽管人们普遍认为模型行为源于训练数据，但目前尚无研究量化政治内容在训练数据中所占的比例。为填补这一空白，我们旨在直接估计：(1) 训练数据中政治参与性文本的比例，(2) 相应的数据不平衡程度，(3) 跨数据集的相似性，以及 (4) 数据构成与模型行为之间的相关性。我们结合大规模采样、政治倾向分类和立场检测方法，分析了开源大语言模型预训练和后训练数据集中的政治内容。我们发现，所有大语言模型训练数据集都系统性地偏向左翼内容，且预训练语料比后训练语料包含更多政治参与性内容。我们进一步观察到，训练数据中的政治立场与模型行为之间存在强相关性。

    arXiv:2509.22367v3 Announce Type: replace  Abstract: Large language models (LLMs) reflect politically-slanted opinions in their generated text. Even though it is widely assumed that model behavior stem from training data, there has been no study quantifying the extent to which political content is part of the training data. To bridge this gap, we aim to directly estimate (1)~the proportion of politically engaged texts in training data, (2)~respective data imbalance, (3)~cross-dataset similarity, and (4)~correlations between data composition and model behaviour. We analyze the political content of pre- and post-training datasets of open-source LLMs, combining large-scale sampling, political-leaning classification, and stance detection. We find that all LLM training datasets are systematically skewed towards left-leaning content, with pre-training containing more politically engaged than post-training corpora. We further observe a strong correlation between political stances in training 
    
[^335]: 从语言模型中的离群值到主题：预测新闻语料库中的趋势

    From Outliers to Topics in Language Models: Anticipating Trends in News Corpora

    [https://arxiv.org/abs/2509.22030](https://arxiv.org/abs/2509.22030)

    该论文发现主题建模中常被视为噪声的离群值实际上是新兴主题的微弱信号，会随时间演变为连贯主题，从而可用于预测新闻语料库中的趋势。

    

    本文研究了在主题建模中常被当作噪声而忽略的离群值，如何能够作为动态新闻语料库中新兴主题的微弱信号。利用最先进语言模型生成的向量嵌入和累积聚类方法，我们追踪了这些离群值在聚焦于企业社会责任和气候变化的法语与英语新闻数据集中随时间的演变过程。结果揭示了一个一致的模式：在两种模型和两种语言中，离群值往往会随时间演变为连贯的主题。

    arXiv:2509.22030v2 Announce Type: replace  Abstract: This paper examines how outliers, often dismissed as noise in topic modeling, can act as weak signals of emerging topics in dynamic news corpora. Using vector embeddings from state-of-the-art language models and a cumulative clustering approach, we track their evolution over time in French and English news datasets focused on corporate social responsibility and climate change. The results reveal a consistent pattern: outliers tend to evolve into coherent topics over time across both models and languages.
    
[^336]: HumanAgencyBench：AI助手对人类能动性支持的可扩展评估

    HumanAgencyBench: Scalable Evaluation of Human Agency Support in AI Assistants

    [https://arxiv.org/abs/2509.08494](https://arxiv.org/abs/2509.08494)

    该论文提出了HumanAgencyBench（HAB），一个利用大语言模型模拟和验证用户查询、可扩展地评估AI助手在提出澄清问题、避免价值操纵、纠正错误信息等六种行为上支持人类能动性的自适应诊断基准。

    

    随着人类将越来越多的任务和决策委托给人工智能（AI），我们面临着失去对个人和集体未来控制的风险。相对简单的算法系统已经在引导人类的决策，例如社交媒体信息流算法会让人们无意间、心不在焉地滚动浏览以提升参与度为目标优化的内容。在本文中，我们通过将哲学和科学中的能动性理论与AI辅助评估方法相结合，发展了人类能动性的概念：利用大语言模型（LLM）来模拟和验证用户查询，并评估AI的回复。我们开发了HumanAgencyBench（HAB），这是一个可扩展且自适应的诊断工具，用于评估与人类能动性相关的六种行为。HAB衡量AI助手在以下方面的倾向：提出澄清性问题、避免价值操纵、纠正错误信息、推迟重要决策、鼓励学习以及维持社交边界。我们发现较低的……

    arXiv:2509.08494v2 Announce Type: replace-cross  Abstract: As humans delegate more tasks and decisions to artificial intelligence (AI), we risk losing control of our individual and collective futures. Relatively simple algorithmic systems already steer human decision-making, such as social media feed algorithms that lead people to unintentionally and absent-mindedly scroll through engagement-optimized content. In this paper, we develop the idea of human agency by integrating philosophical and scientific theories of agency with AI-assisted evaluation methods: using large language models (LLMs) to simulate and validate user queries and to evaluate AI responses. We develop HumanAgencyBench (HAB), a scalable and adaptive diagnostic tool for six behaviors related to human agency. HAB measures the tendency of an AI assistant to Ask Clarifying Questions, Avoid Value Manipulation, Correct Misinformation, Defer Important Decisions, Encourage Learning, and Maintain Social Boundaries. We find low
    
[^337]: TempCore：视频问答基准是否真正基于时间维度？

    TempCore: Are Video QA Benchmarks Temporally Grounded?

    [https://arxiv.org/abs/2509.01167](https://arxiv.org/abs/2509.01167)

    该论文提出帧选择敏感度（FSS）诊断方法，发现现有视频问答基准中仅5.5%–31%的样本真正需要时间维度的帧选择，并据此构建了聚焦时间敏感样本的紧凑评估集TempCore。

    

    视觉-语言模型（VLM）只能处理有限数量的视频帧，这使得帧选择成为一种实际需求。但当前的视频问答基准是否真正需要时间维度的帧选择，还是说无论展示哪些帧，大多数问题都能被回答？我们提出了帧选择敏感度，这是一种逐样本的诊断方法，用于衡量当最相关的帧被最不相关的帧替换时，VLM准确率的变化程度。在六个基准和八个VLM上的实验表明，绝大多数样本对帧选择不敏感：只有少数样本真正对帧的选择敏感。将FSS与语言独立得分相结合后发现，仅有5.5%–31%的样本具有时间敏感性。我们构建了TempCore，这是从现有基准中分离出这些时间敏感样本的紧凑评估子集，并将在论文发表后发布代码和逐样本标注。

    arXiv:2509.01167v3 Announce Type: replace-cross  Abstract: Vision-language models (VLMs) can ingest only a limited number of video frames, making frame selection a practical necessity. But do current Video QA benchmarks genuinely require temporal frame selection, or can most questions be answered regardless of which frames are shown? We introduce Frame Selection Sensitivity (FSS), a per-sample diagnostic that measures how much VLM accuracy changes when the most relevant frames are replaced with the least relevant ones. Across six benchmarks and eight VLMs, we find that a large majority of samples are frame-agnostic: only a minority are genuinely sensitive to frame choice. Combining FSS with a Language Independence Score (LIS) reveals that merely 5.5--31% of samples are Temporally Sensitive. We construct TempCore, compact evaluation subsets that isolate these temporal samples from existing benchmarks, and will release code and per-sample annotations upon publication.
    
[^338]: SalQ-VLM：面向视觉语言模型的细粒度显著性引导量化

    SalQ-VLM: Fine-Grained Saliency-Guided Quantization for Vision-Language Models

    [https://arxiv.org/abs/2508.03351](https://arxiv.org/abs/2508.03351)

    提出了SalQ-VLM，一种重要性感知的训练后量化框架，通过优先处理显著token并抑制冗余视觉token，解决了视觉语言模型中视觉过度表征与模态鸿沟两大问题，从而在资源受限环境下实现高效模型压缩。

    

    大型语言模型（LLMs）在多样化的语言任务中展现了卓越的能力，这推动了其向视觉语言模型（VLMs）的扩展以实现多模态理解。然而，数十亿参数规模的VLMs带来了巨大的内存和计算开销，阻碍了其在资源受限环境中的部署。训练后量化（PTQ）无需重新训练即可压缩模型并加速推理，但其在VLMs中的应用仍未被充分探索。我们在PTQ中识别出VLM激活的两个内在特性：（1）视觉过度表征，即视觉token数量过多且往往冗余；（2）模态鸿沟，即在潜在特征空间中文本token与视觉token相互分离。先前的方法大多忽略了这些特性，导致量化性能下降。为解决这种不匹配问题，我们提出了SalQ-VLM，这是一个重要性感知的PTQ框架，它优先处理显著token并抑制冗余的视觉token。

    arXiv:2508.03351v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have demonstrated remarkable capabilities across diverse language tasks, motivating their extension to vision-language models (VLMs) for multimodal understanding. However, billion-parameter VLMs incur substantial memory and computational costs that hinder deployment in resource-constrained settings. Post-training quantization (PTQ) compresses models and accelerates inference without retraining, yet remains underexplored for VLMs. We identify two intrinsic VLM activation properties in PTQ: (1) visual over-representation, where vision tokens are excessive and often redundant, and (2) the modality gap separating text and vision tokens in the latent feature space. Prior methods largely overlook these properties, leading to quantization performance degradation. To address this mismatch, we propose SalQ-VLM, an importance-aware PTQ framework that prioritizes salient tokens and suppresses redundant vision 
    
[^339]: 面向多模态大语言模型的离散分词技术：一项全面综述

    Discrete Tokenization for Multimodal LLMs: A Comprehensive Survey

    [https://arxiv.org/abs/2507.22920](https://arxiv.org/abs/2507.22920)

    本文首次提出了面向大语言模型的离散分词（向量量化）方法的系统化分类体系，对8种代表性VQ变体的算法原理、训练动态及其与LLM流水线的集成挑战进行了全面分析。

    

    arXiv:2507.22920v2 公告类型：替换 摘要：大语言模型（LLM）的快速发展加剧了对有效机制的需求，即如何将连续的多模态数据转换为适合基于语言处理的离散表示。离散分词技术以向量量化（VQ）为核心方法，既能提供计算效率，又与LLM架构具有良好的兼容性。尽管其重要性日益增长，但目前缺乏一份系统性地考察基于LLM系统中VQ技术的全面综述。本工作填补了这一空白，首次提出了专为LLM设计的离散分词方法的结构化分类体系与分析。我们对涵盖经典与现代范式的8种代表性VQ变体进行了分类，并分析了它们的算法原理、训练动态以及与LLM流水线集成的挑战。除算法层面的研究外，我们还从经典应用的角度讨论了现有研究。

    arXiv:2507.22920v2 Announce Type: replace  Abstract: The rapid advancement of large language models (LLMs) has intensified the need for effective mechanisms to transform continuous multimodal data into discrete representations suitable for language-based processing. Discrete tokenization, with vector quantization (VQ) as a central approach, offers both computational efficiency and compatibility with LLM architectures. Despite its growing importance, there is a lack of a comprehensive survey that systematically examines VQ techniques in the context of LLM-based systems. This work fills this gap by presenting the first structured taxonomy and analysis of discrete tokenization methods designed for LLMs. We categorize 8 representative VQ variants that span classical and modern paradigms and analyze their algorithmic principles, training dynamics, and integration challenges with LLM pipelines. Beyond algorithm-level investigation, we discuss existing research in terms of classical applicati
    
[^340]: 校准轻量级稀疏自编码器特征引导

    Calibrating Lightweight Sparse Autoencoder Feature Steering

    [https://arxiv.org/abs/2506.12576](https://arxiv.org/abs/2506.12576)

    提出ContrastiveSteer方法，通过对比式特征评分和模型特定校准来改进稀疏自编码器的特征引导，使主题对齐提升最高3.9倍，目标领域分类提升高达93%。

    

    稀疏自编码器（SAE）可以通过修改潜在特征激活来实现推理时的主题引导，但现有的引导方法在未识别出目标对齐特征或以错误尺度修改特征时往往会失败。我们提出ContrastiveSteer来解决这些失败模式。首先，通过特征在目标领域文本上比在通用文本上激活更强的程度来对特征进行评分。其次，使用模型特定的校准来设置引导强度。我们还引入了“污染”这一互补启发式方法，用于衡量引导后的激活是否强调弱对齐特征，从而指示偏离目标或无意义的输出。在我们的评估中，跨多个大语言模型系列和SAE，ContrastiveSteer将跨领域主题对齐提升至未引导和引导基线的最高3.9倍，并将目标领域分类提升高达93%。

    arXiv:2506.12576v3 Announce Type: replace  Abstract: Sparse autoencoders (SAEs) can enable inference-time topic steering by modifying latent feature activations, but existing steering methods often fail when target-aligned features are not identified or are modified at the wrong scale. We introduce \textsc{ContrastiveSteer} to address these respective failure modes. First, features are scored by how much more strongly they activate on target-domain text than on general text. Second, steering strength is set using a model-specific calibration. We also introduce \emph{contamination}, a complementary heuristic that measures whether post-steering activations emphasize weakly aligned features, thus indicating off-target or nonsensical outputs. In our evaluations, across multiple LLM families and SAEs, \textsc{ContrastiveSteer} improves topic alignment across domains up to 3.9$\times$ over unsteered and steering baselines, and raises target-domain classification by up to 93\%. Our activation
    
[^341]: 灵枢：面向统一多模态医学理解与推理的通用基础模型

    Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning

    [https://arxiv.org/abs/2506.07044](https://arxiv.org/abs/2506.07044)

    该论文提出了灵枢（Lingshu）——一个通过全面数据筛选流程构建的通用医学基础模型，实现了统一的医学多模态理解与推理，克服了现有医学MLLM在医学知识覆盖、幻觉抑制和复杂医学场景推理方面的关键局限。

    

    多模态大语言模型（MLLMs）在理解常见视觉元素方面展现出令人印象深刻的能力，这主要归功于其大规模数据集和先进的训练策略。然而，由于医学场景与通用领域在数据和任务上存在固有差异，这些模型在医学应用中的效果仍然有限。具体而言，现有的医学MLLM面临以下关键局限：（1）对医学影像之外的医学知识覆盖有限；（2）由于数据筛选流程欠佳，更容易产生幻觉；（3）缺乏针对复杂医学场景定制的推理能力。为应对这些挑战，我们首先提出了一套全面的数据筛选流程，该流程（1）不仅从医学影像中，还从大量医学文本和通用领域数据中高效获取丰富的医学知识数据；（2）合成准确的医学多模态数据（摘要在此处被截断）。

    arXiv:2506.07044v5 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate m
    
[^342]: SocialMaze：一个用于评估和增强大型语言模型在复杂社会环境中社会推理能力的基准

    SocialMaze: A Benchmark for Evaluating and Enhancing Social Reasoning in Large Language Models in Complex Social Environments

    [https://arxiv.org/abs/2505.23713](https://arxiv.org/abs/2505.23713)

    该论文提出了SocialMaze基准，通过深度推理、动态交互和信息不确定性三个设计维度，在社交推理游戏、日常互动和数字社区平台等六项任务中评估并提升大型语言模型在复杂社会环境中的社会推理能力。

    

    大型语言模型（LLM）越来越多地被部署在与社会情境相关的应用中，在这些应用里，成功需要理解上下文、推断他人的心理状态，并对不可靠的信息进行推理。然而，现有的基准测试很少能在复杂且动态演变的环境中联合评估这些需求。我们提出了SocialMaze，这是一个基准测试，它围绕三个描述性设计维度——深度推理、动态交互和信息不确定性——组织了六项任务，涵盖社交推理游戏、日常生活互动和数字社区平台。这些维度刻画的是任务难度的预期来源，而非模型能力的潜在因子分析维度。自动化检查和人工验证保障了数据质量。对十二个专有和开源权重大型语言模型的评估显示，模型在利用动态演变的交互历史方面存在显著差异；更强的思维链推理者在需要深度推理的任务上表现更好。

    arXiv:2505.23713v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly deployed in socially grounded applications, where success requires interpreting context, inferring others' mental states, and reasoning about unreliable information. Yet existing benchmarks rarely evaluate these demands jointly in complex, evolving settings. We introduce SocialMaze, a benchmark that organizes six tasks across social deduction games, daily-life interactions, and digital community platforms along three descriptive design axes: deep reasoning, dynamic interaction, and information uncertainty. These axes characterize intended sources of task difficulty rather than latent, factor-analytic dimensions of model capability. Automated checks and human validation support data quality. Evaluations of twelve proprietary and open-weight LLMs show substantial variation in the use of evolving interaction histories; stronger chain-of-thought reasoners perform better on tasks requiring dee
    
[^343]: ESLM：面向高效预训练的风险规避选择性语言建模

    ESLM: Risk-Averse Selective Language Modeling for Efficient Pretraining

    [https://arxiv.org/abs/2505.19893](https://arxiv.org/abs/2505.19893)

    提出ESLM算法，利用逐token统计量（熵或损失）和在险价值阈值筛选在线选择每批次中最具信息量的token进行训练，从而提升大语言模型预训练效率并增强分布鲁棒性。

    

    大型语言模型预训练是计算密集型的，然而许多token对学习的贡献微乎其微，造成了效率低下。我们提出了高效选择性语言建模（ESLM），这是一种风险感知算法，通过执行在线的token级批次选择来提升训练效率和分布鲁棒性。ESLM利用逐token的统计量（如熵或损失），并应用在险价值（value-at-risk）阈值筛选，仅保留每个批次中最具信息量的token。这种以数据为中心的机制重塑了训练损失，优先处理高风险token并消除了冗余的梯度计算。我们将ESLM构建为一个双层博弈：模型与一个掩码对手进行竞争，该对手在受约束的阈值规则下选择最坏情况下的token子集。在基于损失的设定中，ESLM可归结为条件在险价值（CVaR）损失最小化，从而与分布鲁棒优化建立了原理性的联系。我们进一步扩展……

    arXiv:2505.19893v2 Announce Type: replace-cross  Abstract: Large language model pretraining is compute-intensive, yet many tokens contribute marginally to learning, resulting in inefficiency. We introduce Efficient Selective Language Modeling (ESLM), a risk-aware algorithm that improves training efficiency and distributional robustness by performing online token-level batch selection. ESLM leverages per-token statistics (e.g., entropy or loss) and applies value-at-risk thresholding to retain only the most informative tokens per batch. This data-centric mechanism reshapes the training loss, prioritizing high-risk tokens and eliminating redundant gradient computation. We frame ESLM as a bilevel game: the model competes with a masking adversary that selects worst-case token subsets under a constrained thresholding rule. In the loss-based setting, ESLM recovers conditional value-at-risk loss minimization, providing a principled connection to distributionally robust optimization. We extend 
    
[^344]: 少解释，多理解：面向读者相关术语的数据高效个性化

    Explain Less, Understand More: Data-Efficient Personalization of Reader-Dependent Jargons

    [https://arxiv.org/abs/2505.16227](https://arxiv.org/abs/2505.16227)

    该论文提出了两种高效可扩展的术语个性化策略——基于LoRA的轻量级微调和无需再训练的个性化提示，并结合半监督学习利用用户出版物数据，其个性化LoRA模型性能超越了使用上下文提示的GPT-4。

    

    arXiv:2505.16227v4 公告类型：替换 摘要：术语的个性化检测与解释对于让具有不同学科背景的读者能够理解技术文档至关重要。然而，由于需要进行用户特定的微调，为单个用户定制模型通常需要大量的标注工作和计算资源。为了解决这一问题，我们对个性化术语检测进行了系统性研究，重点关注对现实世界部署既高效又可扩展的方法。我们探索了两种个性化策略：（1）在开源模型上使用低秩适配进行轻量级微调；（2）个性化提示，即在推理时定制模型行为而无需重新训练。为了反映现实约束，我们还研究了将有限的标注数据与来自用户出版物的自监督学习相结合的半监督方法。我们的个性化LoRA模型在上下文提示方面超越GPT-4达2……（摘要原文在此处不完整）

    arXiv:2505.16227v4 Announce Type: replace  Abstract: Personalizing jargon detection and explanation is essential for making technical documents accessible to readers with diverse disciplinary backgrounds. However, tailoring models to individual users typically requires substantial annotation efforts and computational resources due to user-specific finetuning. To address this, we present a systematic study of personalized jargon detection, focusing on methods that are both efficient and scalable for real-world deployment. We explore two personalization strategies: (1) lightweight finetuning using Low-Rank Adaptation (LoRA) on open-source models, and (2) personalized prompting, which tailors model behavior at inference time without retaining. To reflect realistic constraints, we also investigate semi-supervised approaches that combine limited annotated data with self-supervised learning from users' publications. Our personalized LoRA model outperforms GPT-4 with contextual prompting by 2
    
[^345]: 基于全单向架构的高效自适应同声语音翻译

    Efficient and Adaptive Simultaneous Speech Translation with Fully Unidirectional Architecture

    [https://arxiv.org/abs/2504.11809](https://arxiv.org/abs/2504.11809)

    提出EASiST，通过语音编码器与LLM均为全单向的架构、多延迟数据筛选策略、带显式读/写标记的交错生成任务以及轻量级策略头，实现了高效且自适应的同声语音翻译。

    

    同声语音翻译（SimulST）在处理部分语音输入的同时增量地生成译文。尽管大型语言模型（LLM）在离线翻译任务中展现出强大的能力，但将其应用于SimulST面临显著挑战。现有的基于LLM的SimulST方法要么由于双向语音编码器的重复编码而产生巨大的计算开销，要么依赖于固定的读/写策略，从而限制了效率与性能。在本工作中，我们提出了采用全单向架构（涵盖语音编码器和LLM）的高效自适应同声语音翻译方法（EASiST）。EASiST包含一种多延迟数据筛选策略，用于生成语义对齐的SimulST训练样本，并将SimulST重新定义为带有显式读/写标记的交错生成任务。为实现自适应推理，我们引入了一个轻量级策略头，……

    arXiv:2504.11809v2 Announce Type: replace  Abstract: Simultaneous speech translation (SimulST) produces translations incrementally while processing partial speech input. Although large language models (LLMs) have shown strong capabilities in offline translation tasks, applying them to SimulST poses notable challenges. Existing LLM-based SimulST approaches either incur significant computational overhead due to repeated encoding of bidirectional speech encoder, or they depend on a fixed read/write policy, limiting the efficiency and performance. In this work, we introduce Efficient and Adaptive Simultaneous Speech Translation (EASiST) with fully unidirectional architecture, including both speech encoder and LLM. EASiST includes a multi-latency data curation strategy to generate semantically aligned SimulST training samples and redefines SimulST as an interleaved generation task with explicit read/write tokens. To facilitate adaptive inference, we incorporate a lightweight policy head tha
    
[^346]: AskQE：问答作为机器翻译的自动评估方法

    AskQE: Question Answering as Automatic Evaluation for Machine Translation

    [https://arxiv.org/abs/2504.11582](https://arxiv.org/abs/2504.11582)

    AskQE是一个基于问题生成与回答的机器翻译质量评估框架，使不懂目标语言的用户也能检测关键翻译错误并决定是否接受译文，其与人工评分的相关性和决策准确率优于现有QE指标。

    

    一个只会英语的单语使用者如何判断一篇法语自动翻译是否足够好以至于可以分享？现有的机器翻译错误检测和质量估计（QE）技术并未解决这一实际场景。我们提出了AskQE，一个问题生成与回答框架，旨在检测关键的机器翻译错误并提供可操作的反馈，帮助用户即使不懂目标语言也能决定是否接受或拒绝机器翻译输出。利用ContraTICO（一个COVID-19领域的对比性合成机器翻译错误数据集），我们探索了AskQE的设计选择，并开发了一个基于LLaMA-3 70B和蕴含事实来指导问题生成的优化版本。我们在包含自然发生的机器翻译错误的BioMQM数据集上评估了该系统，与其他QE指标相比，AskQE与人工评分具有更高的Kendall's Tau相关性和决策准确率。

    arXiv:2504.11582v3 Announce Type: replace  Abstract: How can a monolingual English speaker determine whether an automatic translation in French is good enough to be shared? Existing MT error detection and quality estimation (QE) techniques do not address this practical scenario. We introduce AskQE, a question generation and answering framework designed to detect critical MT errors and provide actionable feedback, helping users decide whether to accept or reject MT outputs even without the knowledge of the target language. Using ContraTICO, a dataset of contrastive synthetic MT errors in the COVID-19 domain, we explore design choices for AskQE and develop an optimized version relying on LLaMA-3 70B and entailed facts to guide question generation. We evaluate the resulting system on the BioMQM dataset of naturally occurring MT errors, where AskQE has higher Kendall's Tau correlation and decision accuracy with human ratings compared to other QE metrics.
    
[^347]: 仅需Logits即可适配闭源模型

    Logits are All We Need to Adapt Closed Models

    [https://arxiv.org/abs/2502.06806](https://arxiv.org/abs/2502.06806)

    提出了一种仅利用logits的token级概率重加权框架Plugin，将黑盒LLM的适配问题转化为标签噪声纠正问题，从而在无需访问模型内部的情况下实现面向特定应用的内容生成。

    

    许多商用大型语言模型（LLM）通常是闭源的，这限制了开发者只能通过提示调优来使内容生成与特定应用保持一致。虽然这些模型目前不提供对token logits的访问，但我们认为，如果能够获得这种访问权限，将可以实现超越提示工程的更强大的适配技术。在本文中，我们提出了一个token级别的概率重加权框架，在获得logits和少量任务特定数据的情况下，能够有效地引导黑盒LLM进行面向特定应用的内容生成。我们的方法从监督分类的视角看待下一token预测任务。我们证明，将黑盒LLM与任务特定数据对齐可以表述为一个标签噪声纠正问题，由此产生了Plugin模型——一个仅基于logits运行的自回归概率重加权模型。我们为此提供了理论依据。

    arXiv:2502.06806v5 Announce Type: replace-cross  Abstract: Many commercial Large Language Models (LLMs) are often closed-source, limiting developers to prompt tuning for aligning content generation with specific applications. While these models currently do not provide access to token logits, we argue that if such access were available, it would enable more powerful adaptation techniques beyond prompt engineering. In this paper, we propose a token-level probability reweighting framework that, given access to logits and a small amount of task-specific data, can effectively steer black-box LLMs toward application-specific content generation. Our approach views next-token prediction through the lens of supervised classification. We show that aligning black-box LLMs with task-specific data can be formulated as a label noise correction problem, leading to Plugin model -- an autoregressive probability reweighting model that operates solely on logits. We provide theoretical justification for 
    
[^348]: 3D-MoE：基于混合专家模型迈向空间智能的3D推理与动作生成

    3D-MoE: Towards Spatial Intelligence with Mixture-of-Experts for 3D Reasoning and Action Generation

    [https://arxiv.org/abs/2501.16698](https://arxiv.org/abs/2501.16698)

    3D-MoE通过混合专家架构与模态/空间上下文感知的概率路由实现高效3D推理，并集成Pose-DiT扩散动作头在单步内生成精确6D位姿动作，以大幅减少的激活参数在3D视觉-语言基准上取得优越性能。

    

    空间智能，涵盖3D感知与推理，是人工智能下一个至关重要的前沿领域。对当前依赖密集Transformer处理空间任务的3D视觉-语言模型（VLM）进行扩展会带来难以承受的计算成本。在本文中，我们提出了3D-MoE，这是一个利用高效混合专家架构的3D视觉-语言模型，它采用模态感知与空间上下文感知的概率路由方案，并通过一种新颖的路由课程策略进行稳定培养。为了将3D-MoE无缝扩展至具身智能，我们集成了一个基于扩散模型的动作头Pose-DiT，将3D-MoE转变为3D视觉-语言-动作（VLA）模型。通过采用整流流框架，Pose-DiT能够在单次采样步骤中生成精确的6D位姿动作。大量实验表明，3D-MoE在多种3D视觉-语言基准测试中，以大幅减少的激活参数量取得了优越的性能，并实现了更高的成功率，同时支持……

    arXiv:2501.16698v2 Announce Type: replace  Abstract: Spatial intelligence, encompassing 3D perception and reasoning, is the essential next frontier of AI. Scaling current 3D vision-language models (VLMs) that rely on dense Transformers for spatial tasks incurs prohibitive computational costs. In this paper, we introduce 3D-MoE, a 3D VLM leveraging an efficient mixture-of-experts architecture with a modality- and spatial-context-aware probabilistic routing scheme, stably cultivated by a novel routing curriculum. To seamlessly extend 3D-MoE to embodied AI, we integrate a diffusion-based action head, Pose-DiT, transforming 3D-MoE into a 3D vision-language-action (VLA) model. By employing a rectified flow framework, Pose-DiT generates precise 6D pose actions in a single sampling step. Extensive experiments demonstrate that 3D-MoE achieves superior performance on diverse 3D vision-language benchmarks with drastically fewer activated parameters and yields higher success rates while enabling 
    
[^349]: Compound-QA：一个用于评估大语言模型处理复合问题能力的基准

    Compound-QA: A Benchmark for Evaluating LLMs on Compound Questions

    [https://arxiv.org/abs/2411.10163](https://arxiv.org/abs/2411.10163)

    该论文提出了复合问题合成方法CQ-Syn，构建了包含五个类别、从理解、推理和知识三个维度评估大语言模型处理由多个相互关联子问题组成的复合问题能力的新基准Compound-QA。

    

    大语言模型（LLMs）在各类任务中展现出卓越的性能，这促使研究人员开发了多样化的评估基准。然而，大多数基准通常衡量的是LLM回答单个问题的能力，忽略了现实应用中复杂的交互场景。我们提出了复合问题合成方法（CQ-Syn），以构建Compound-QA——一个针对由多个相互关联的子问题组成的复合问题的基准。该基准源自现有的问答数据集，使用专有LLM进行标注，并由人工验证其准确性。它涵盖五个类别：事实陈述、因果分析、假设分析、比较选择以及评估建议。该基准从三个维度评估LLM的能力，包括理解、推理和知识。在Compound-QA上对九个开源LLM进行评估后发现，它们在复合问题上的表现……

    arXiv:2411.10163v3 Announce Type: replace  Abstract: Large language models (LLMs) demonstrate remarkable performance across various tasks, prompting researchers to develop diverse evaluation benchmarks. However, most benchmarks typically measure the ability of LLMs to respond to individual questions, neglecting the complex interactions in real-world applications. We introduce Compound Question Synthesis (CQ-Syn) to build Compound-QA, a benchmark targeting questions composed of multiple interrelated sub-questions. This benchmark is derived from existing QA datasets, annotated with proprietary LLMs, and verified by humans for accuracy. It encompasses five categories: Factual-Statement, Cause-and-Effect, Hypothetical-Analysis, Comparison-and-Selection, and Evaluation-and-Suggestion. It evaluates the LLM capability in terms of three dimensions, including understanding, reasoning, and knowledge. Evaluating nine open-source LLMs on Compound-QA reveals that their performance on compound quest
    
[^350]: 高等教育课程智能平台：AI辅助课程评估的经验启示

    A Course Intelligence Platform for Higher Education: Lessons from AI-Assisted Course Evaluation

    [https://arxiv.org/abs/2411.02455](https://arxiv.org/abs/2411.02455)

    本文提出了一个在中国100多所高校部署、服务超万名教师的课程智能平台，其AI辅助课程评估模块通过整合国家评估标准、结构化教育证据与领域适配的大语言模型，自动生成量化评分与定性反馈，填补了院校层面AI应用的空白。

    

    生成式人工智能的快速普及为教学、学习和质量保障创造了新的机遇。然而，现有应用大多面向学生，对院校层面需求的关注相对有限。本文介绍了一个在中国100多所高校部署、服务超过10,000名教师的课程智能平台。该平台通过将能力要求、知识结构、教学活动和评估证据相互关联，为知识组织、教学设计、学习评估和质量评价建立了共同基础。课程评估模块作为该平台面向院校的代表性应用被重点研究，该模块整合了国家评估标准、结构化教育证据、定制化提示策略以及领域适配的大语言模型，以生成量化评分和定性反馈。一项涉及100所（高校的案例研究……）

    arXiv:2411.02455v3 Announce Type: replace  Abstract: The rapid adoption of generative AI has created new opportunities for teaching, learning, and quality assurance. Existing applications, however, remain largely student-facing, with comparatively limited attention to institution-level needs. This paper presents a course intelligence platform deployed across more than 100 universities and serving over 10,000 instructors in China. By linking competency requirements, knowledge structures, teaching activities, and assessment evidence, it establishes a shared foundation for knowledge organization, instructional design, learning assessment, and quality evaluation. The course evaluation module is examined as a representative institution-facing application of the platform, which integrates national evaluation standards, structured educational evidence, customized prompting strategies, and domain-adapted LLMs to generate quantitative scores and qualitative feedback. A case study involving 100 
    
[^351]: SG-FSM：一种基于有限状态机的多跳问答自引导零样本提示范式

    SG-FSM: A Self-Guiding Zero-Shot Prompting Paradigm for Multi-Hop Question Answering Based on Finite State Machine

    [https://arxiv.org/abs/2410.17021](https://arxiv.org/abs/2410.17021)

    提出了一种基于有限状态机的自引导零样本提示范式SG-FSM，通过迭代分解复杂问题为子问题、自我纠错并动态决定下一步推理步骤，有效提升了大语言模型在多跳问答任务中的表现。

    

    采用思维链提示的大语言模型（如OpenAI-o1）在自然语言推理任务中展现出了令人印象深刻的能力。然而，由于幻觉、错误传播和上下文长度受限等问题，多跳问答（MHQA）对许多现有模型来说仍然具有挑战性。为了应对这些挑战并提升大语言模型在多跳问答上的表现，我们提出了自引导提示有限状态机（SG-FSM），旨在增强多跳推理能力。与传统的思维链方法不同，SG-FSM通过迭代地将复杂问题分解为子问题来解决多跳问答，并通过自我纠错来提高准确性。它每次只处理一个子问题，根据当前上下文和结果动态决定下一步操作，其运作方式类似于自动机。在多个基准测试上的实验证明了我们方法的有效性，其表现优于强大的基线方法。

    arXiv:2410.17021v2 Announce Type: replace  Abstract: Large Language Models with chain-of-thought prompting, such as OpenAI-o1, have shown impressive capabilities in natural language inference tasks. However, Multi-hop Question Answering (MHQA) remains challenging for many existing models due to issues like hallucination, error propagation, and limited context length. To address these challenges and enhance LLMs' performance on MHQA, we propose the Self-Guiding prompting Finite State Machine (SG-FSM), designed to strengthen multi-hop reasoning abilities. Unlike traditional chain-of-thought methods, SG-FSM tackles MHQA by iteratively breaking down complex questions into sub-questions, correcting itself to improve accuracy. It processes one sub-question at a time, dynamically deciding the next step based on the current context and results, functioning much like an automaton. Experiments across various benchmarks demonstrate the effectiveness of our approach, outperforming strong baselines
    
[^352]: Eraser：通过遗忘有害知识实现大语言模型的越狱防御

    Eraser: Jailbreaking Defense in Large Language Models via Unlearning Harmful Knowledge

    [https://arxiv.org/abs/2404.05880](https://arxiv.org/abs/2404.05880)

    提出 Eraser 防御方法，通过让大语言模型遗忘回答有害问题所需的知识来从根本上消除越狱风险，且无需红队协助即可在保留通用知识和安全对齐的同时显著提升模型安全性。

    

    越狱攻击可以使大语言模型（LLM）绕过安全防护并生成有害内容。现有的越狱防御方法未能解决有害知识存在于模型内部这一根本问题，导致大语言模型面临潜在的越狱风险。在本文中，我们提出了一种名为 Eraser 的新型防御方法，其主要包括三个目标：遗忘有害知识、保留通用知识以及维持安全对齐。其核心直觉是：如果大语言模型忘记了回答有害问题所需的特定知识，它将不再具备回答有害问题的能力。Eraser 的训练实际上并不需要模型自身的有害知识，它可以通过遗忘与有害查询相关的通用回答来获益，这意味着它不需要红队的协助。实验结果表明，Eraser 能够显著降低越狱风险。

    arXiv:2404.05880v3 Announce Type: replace  Abstract: Jailbreaking attacks can enable Large Language Models (LLMs) to bypass the safeguard and generate harmful content. Existing jailbreaking defense methods have failed to address the fundamental issue that harmful knowledge resides within the model, leading to potential jailbreak risks for LLMs. In this paper, we propose a novel defense method called Eraser, which mainly includes three goals: unlearning harmful knowledge, retaining general knowledge, and maintaining safety alignment. The intuition is that if an LLM forgets the specific knowledge required to answer a harmful question, it will no longer have the ability to answer harmful questions. The training of Erase does not actually require the model's own harmful knowledge, and it can benefit from unlearning general answers related to harmful queries, which means it does not need assistance from the red team. The experimental results show that Eraser can significantly reduce the jai
    
[^353]: Calpric：基于众包与主动学习的包容性细粒度隐私政策标注

    Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning

    [https://arxiv.org/abs/2401.08038](https://arxiv.org/abs/2401.08038)

    Calpric通过结合自动文本分割、主动学习和众包标注，以低成本生成大型均衡的隐私政策训练数据集，使未经训练的众包标注者能达到与专业标注者相当的水平。

    

    在隐私政策上训练准确的深度学习模型面临的一个重大挑战是获取大量且全面的训练数据的成本和难度。为了解决这些挑战，我们提出了Calpric，它结合了自动文本选择与分割、主动学习以及众包标注者的使用，以低成本为隐私政策生成大型、均衡的训练集。自动化文本选择和分割简化了标注任务，使来自众包平台（如亚马逊Mechanical Turk）的未经训练的标注者能够与训练有素的标注者（如法学院学生）相媲美，同时降低了标注者间一致性的要求，从而降低了标注成本。拥有可靠的训练标签使得主动学习的应用成为可能，主动学习使用更少的训练样本高效地覆盖输入空间，进一步降低了成本，并改善了数据集中类别和数据类别的平衡。

    arXiv:2401.08038v2 Announce Type: replace  Abstract: A significant challenge to training accurate deep learning models on privacy policies is the cost and difficulty of obtaining a large and comprehensive set of training data. To address these challenges, we present Calpric , which combines automatic text selection and segmentation, active learning and the use of crowdsourced annotators to generate a large, balanced training set for privacy policies at low cost. Automated text selection and segmentation simplifies the labeling task, enabling untrained annotators from crowdsourcing platforms, like Amazon's Mechanical Turk, to be competitive with trained annotators, such as law students, and also reduces inter-annotator agreement, which decreases labeling cost. Having reliable labels for training enables the use of active learning, which uses fewer training samples to efficiently cover the input space, further reducing cost and improving class and data category balance in the data set. T
    
[^354]: 领域偏移下的作者身份识别：文体度量与学习型作者表示综述

    Authorship identification under domain shift: a survey of stylistic measures and learned author representations

    [https://arxiv.org/abs/2310.00436](https://arxiv.org/abs/2310.00436)

    本综述提出作者识别的主题独立性取决于文体特征、编码方式、评分规则与评估划分的组合而非特征本身，并从四个层面系统梳理了领域偏移下从经典频率度量到学习型作者表示的研究证据。

    

    作者身份识别利用写作中的模式来推断文本的作者是谁，但这些模式同时也会反映主题、体裁和语域。本综述认为，主题独立性并非文体特征本身的属性，而是由特征及其编码方式、评分规则和评估划分共同决定的属性，因此关键问题在于哪些组合能够在领域发生变化时保留作者之间的差异。我们将相关证据组织为四个层面：领域与评估条件、语言度量、表示方法，以及评分与决策规则，并考察了从经典频率度量到学习型作者表示、再到语言模型中介写作的同一研究内的比较。这些比较说明了为什么仅凭特征清单无法解释性能表现：归一化和编码方式会改变结果，作者监督学习可能保留内容信息，而评估协议甚至可能颠倒模型排名。成功的识别，文体……

    arXiv:2310.00436v2 Announce Type: replace  Abstract: Authorship identification uses patterns in writing to infer who wrote a text, but those patterns also reflect topic, genre, and register. This survey argues that topic-independence is not a property of a stylistic feature but of the feature together with its encoding, its scoring rule, and the evaluation split, so the question is which combinations preserve author differences when the domain changes. We organize the evidence in four layers, domains and evaluation conditions, linguistic measures, representations, and scoring and decision rules, and examine within-study comparisons from classical frequency measures to learned author representations and language-model-mediated writing. These comparisons show why feature inventories alone do not explain performance: normalization and encoding change results, author-supervised learning can retain content, and evaluation protocols can reverse model rankings. Successful identification, styl
    
[^355]: 通过最大化变分互信息的自编码器改进生成多样性的VOLTA

    VOLTA: Improving Generative Diversity by Variational Mutual Information Maximizing Autoencoder

    [https://arxiv.org/abs/2307.00852](https://arxiv.org/abs/2307.00852)

    VOLTA通过Transformer与VAE框架的更有效连接，InfoGAN风格潜在编码以及支持离散输入，提升了生成多样性

    

    自然语言生成领域得益于Transformer模型取得了巨大成功。虽然它们实现了最先进的生成质量，但往往忽视了生成多样性。先前尝试解决这一问题的方法要么容量较低，要么结构过于复杂。一些最近的方法采用VAE框架增强多样性，但它们的潜在变量完全依赖于输入上下文，限制了潜在空间的探索。在本文中，我们介绍了VOLTA，通过更有效的基于交叉注意力的连接将Transformer与VAE联系起来，从传统的嵌入连接或求和中脱颖而出，提升了生成多样性。此外，我们提议整合InfoGAN风格的潜在编码以实现输入独立的变化性，进一步使生成多样化。此外，我们的框架除了支持现有的连续输入外，还支持离散输入。

    arXiv:2307.00852v2 Announce Type: replace  Abstract: The natural language generation domain has witnessed great success thanks to Transformer models. Although they have achieved state-of-the-art generative quality, they often neglect generative diversity. Prior attempts to tackle this issue suffer from either low model capacity or over-complicated architectures. Some recent methods employ the VAE framework to enhance diversity, but their latent variables fully depend on the input context, restricting exploration of the latent space. In this paper, we introduce VOLTA, a framework that elevates generative diversity by bridging Transformer with VAE via a more effective cross-attention-based connection, departing from conventional embedding concatenation or summation. Additionally, we propose integrating InfoGAN-style latent codes to enable input-independent variability, further diversifying the generation. Moreover, our framework accommodates discrete inputs alongside its existing support
    

