# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent Reasoning in Continuous Space for Unified Multimodal Models](https://arxiv.org/abs/2610.42149) | 本文提出LARC框架，通过在连续隐状态空间而非离散token空间中进行推理，解决了统一多模态模型中语言推理与图像生成之间的模态不匹配问题，并结合课程化监督微调与信息增益强化学习进行训练。 |
| [^2] | [Flide: Accelerating Decision Inference with Exact State Forking](https://arxiv.org/abs/2610.42148) | Flide 是一种精确高效的决策推理引擎，通过 ForkAttention 在单一扁平化 token 流中实现精确的状态分叉，将决策请求的逻辑树结构与物理执行解耦，从而消除前缀重计算开销并保证分支求值结果与独立前向传播完全一致。 |
| [^3] | [DecisionRank: On-Policy Distillation for Decision-Model Rerankers](https://arxiv.org/abs/2610.42128) | 提出DecisionRank，通过条目空间在线策略蒸馏将生成式LLM的排序知识迁移到决策模型重排器中，在TREC DL、BEIR和BRIGHT等检索基准上达到最先进性能。 |
| [^4] | [Subquadratic Attention with a Hierarchical Tree-Structured Indexer](https://arxiv.org/abs/2610.42112) | 本文提出 Zyphra 分层注意力（ZHA），通过在 KV 缓存上构建递归压缩键摘要的树结构索引器，使包含路由在内的总计算成本随序列长度次线性增长，从而突破了注意力二次方复杂度对大语言模型长上下文能力的根本限制。 |
| [^5] | [RangeSWA: Training-Free Length Extrapolation for Continuous Diffusion Language Models](https://arxiv.org/abs/2610.42084) | RangeSWA是一种无需训练的长度外推方法，通过滑动窗口注意力、远程块摘要、动态内容检索以及位置重映射等技术，使连续扩散语言模型在超出训练长度生成文本时仍能保持质量，有效降低生成困惑度和文本重复率。 |
| [^6] | [The Knowledge Layer Model (KLM): Evidence-Visible AI Output, Routing Diagnostics, and Negative Results](https://arxiv.org/abs/2610.42066) | 该论文提出知识层模型（KLM），一个开放的、与实现无关的AI推理记录规范，通过带版本管理的认知评估、证据关系和明确的未知项使AI输出可被审查，并如实报告了包含负面结果在内的初步路由诊断实验。 |
| [^7] | [PersonaChatBench: Evaluating Chat Assistants with Diverse Simulated Users](https://arxiv.org/abs/2610.42060) | 提出了PersonaChatBench评估框架，通过将相同任务与具有不同角色画像的多个模拟用户配对，对20个语言模型在六个领域进行评估，发现助手的任务性能和对用户需求的满足程度会因用户角色和领域的不同而显著变化。 |
| [^8] | [CRoS: Causality Reward on Scrambled CoTs Improves Accuracy and Reasoning Faithfulness](https://arxiv.org/abs/2610.42054) | 该论文提出 CRoS 方法，通过“打乱思维链”这种不改变各步骤信息的扰动方式来度量因果性并构建训练奖励，克服了现有基于删除或注入错误等扰动度量的混淆效应，从而同时提升语言模型推理的准确率与忠实度。 |
| [^9] | [Asynchronous Flow Map Language Models](https://arxiv.org/abs/2610.42018) | 提出异步流映射语言模型，通过逐token的时间调度将预测顺序内化到一步式建模中，并配合新的流映射蒸馏方法，实现单次前向传播即可生成全部token的并行语言生成。 |
| [^10] | [TurnTurk-GPT: Text-Based Turn-Taking Prediction for Turkish](https://arxiv.org/abs/2610.42010) | 本文提出了首个在真实土耳其语对话上系统评估的基于文本的轮次转换预测器TurnTurk-GPT，其在说话人不相交评估中优于广泛使用的多语言检测器，并揭示了土耳其语黏着语后缀对轮次预测至关重要。 |
| [^11] | [Less Loss, Fixed Weights: Logit Corrections for Forgetting](https://arxiv.org/abs/2610.42000) | 该研究发现语言模型微调后的“遗忘”损失中，仅靠固定的词汇偏置只能挽回不到10%，而结合温度校正可挽回多达36%，且在随机损坏的模拟中该方法仍能恢复29%，表明相当一部分遗忘损失可通过简单的固定输出校正来弥补。 |
| [^12] | [DAMI: Dual-Axis Memory Initialization for Enhanced Long-Context Retrieval in Gated Linear Attention](https://arxiv.org/abs/2610.41994) | 提出双轴记忆初始化方法 DAMI，通过在训练前为门控线性注意力中的各注意力头分配几何时间尺度、并为各层设置低-高-低的输入敏感度分布，在不增加可训练参数的前提下显著提升了 GDN、KDA、Mamba-2、RWKV-7 等模型的长上下文检索准确率。 |
| [^13] | [Aligning MLLMs with the Latent Structure of Human Cognition via Behavior-Derived Semantic Dimensions](https://arxiv.org/abs/2610.41951) | 该论文提出一种新框架，利用源自470万次人类“找出不同项”判断的66个可解释语义维度及其对应的语言理据，将多模态大语言模型的决策标准与人类认知的潜在结构对齐，从而提升模型与人类相似性判断的一致性。 |
| [^14] | [Completeness and Cost of Retrieval Structures for Multi-Tool Requests in LLM Agents](https://arxiv.org/abs/2610.41905) | 该论文提出completeness@k指标，系统比较了六种工具检索结构在多工具请求下的完整性与检索成本，发现现有在单工具查询上构建的检索器在需要四个及以上工具的查询上性能急剧下降（完整性从0.58–0.67跌至0.08–0.17）。 |
| [^15] | [How Much Can Tool Calls Reveal About You? Privacy Leakage from Tool-Call Traces](https://arxiv.org/abs/2610.41885) | 该论文首次系统研究了从MCP工具调用轨迹推断用户隐私画像的风险，构建了包含1,000个用户画像和4,000条智能体轨迹的PrivacyTrace基准，发现LLM攻击者能以26.25%的成功率从工具调用轨迹推断用户属性，其中工具参数、返回结果及跨会话轨迹关联是主要的隐私泄露途径。 |
| [^16] | [Text as an Index for Zero-Shot Cross-Domain Knowledge Tracing](https://arxiv.org/abs/2610.41883) | 该论文提出TIZSKT框架，将文本作为索引而非直接可迁移的预测表示，通过分离行为序列建模与域内局部历史证据构建，实现了无需目标平台作答日志的零样本跨域知识追踪。 |
| [^17] | [SimXAI: A Simple Drop-in Layer Increases Interaction Quality and Understanding in Conversational XAI](https://arxiv.org/abs/2610.41856) | SimXAI用一个简单的即插即用LLM对话层替代传统ConvXAI中分离的意图识别、对话管理和模板生成模块，在保持高忠实性的同时更好地处理依赖上下文的后续问题。 |
| [^18] | [Iterated Integrals as Lexical Fingerprints: Path Signature Embeddings for Lexical Similarity](https://arxiv.org/abs/2610.41855) | 该论文提出利用粗糙路径理论中的迭代积分，通过Word2Vec轨迹的路径签名来构建句子嵌入，以几何可处理的方式显式捕捉词汇与序列结构，并揭示句子嵌入隐式编码表层词汇相似性的规律。 |
| [^19] | [Peeling the Matryoshka: Extracting Ontological Structure from Retrieval Embeddings](https://arxiv.org/abs/2610.41785) | 本文发现MRL嵌入的前缀长度天然对应分类体系由粗到细的粒度，并据此提出仅需训练一次的极简探针PEEL，可直接从冻结的检索嵌入中提取本体层次结构，避免昂贵的调参与任务特定训练。 |
| [^20] | [Stepwise-Querying: Distilling Reasoning without Access to Teacher Chain-of-Thought](https://arxiv.org/abs/2610.41769) | 提出逐步查询方法，通过对教师模型进行逐步独立查询以获取中间推导、有效性检查和修正，无需访问教师思维链即可构建推理监督数据，实现高效的推理蒸馏，并在16项基准比较中的14项取得最佳成绩。 |
| [^21] | [Task Vectors Can Be Linearly Decomposed into Interpretable and Controllable Components](https://arxiv.org/abs/2610.41743) | 该论文提出将任务向量线性分解为任务识别（TR）、任务学习（TL）和残差三个可解释成分，并通过贝叶斯优化TR和TL系数，找到了比原始任务向量更有效的模型干预方向。 |
| [^22] | [Leakage-Controlled Structure-Preserving Augmentation and Template-Aware Evaluation for Bangla Symptom-Based Disease Prediction](https://arxiv.org/abs/2610.41733) | 本文提出仅在训练集上进行的泄漏受控、保持症状结构的非对称伯努利数据增强方法与模板感知评估框架，显著提升孟加拉语症状疾病预测中提升树模型（XGBoost、LightGBM、CatBoost）的准确率，并揭示标准基准中普遍存在的模板泄漏问题。 |
| [^23] | [From Training Exposure to Membership Evidence: A Controlled Study of OLMo 2](https://arxiv.org/abs/2610.41732) | 本研究通过受控地重新运行 OLMo 2 1B 的 500 亿 token 中期训练，并利用同种子发布的运行作为反事实参照，首次严格量化了单次训练暴露所提供的成员证据——单次暴露平均会提高文档似然，但很少能可靠地识别出训练成员。 |
| [^24] | [UpLoop: Training Looped Transformers via Upcycling at No Extra Pretraining Compute](https://arxiv.org/abs/2610.41720) | UpLoop提出一种升级复用式预训练调度方法，从浅层Transformer逐步复制层来加深模型，并将早期节省的计算用于后期共享中间块的重复执行，在与稠密基线完全相同的参数量、训练token数和FLOPs下训练循环Transformer，显著提升语言建模与下游任务性能。 |
| [^25] | [GraphNIAH: A Benchmark for Long-Context Reasoning over Real-World Graphs](https://arxiv.org/abs/2610.41704) | GraphNIAH提出了一个基于真实科学论文引用关系构建的动态基准测试，用于评估长上下文图推理能力，克服了现有大海捞针基准依赖合成数据、数据污染风险和人工标注等局限性。 |
| [^26] | [Precision--Recall Trade-off on the Inference Level for LLMs in Grammatical Error Correction](https://arxiv.org/abs/2610.41698) | 本研究发现，仅通过调整大语言模型推理时的重复惩罚参数，即可在不修改模型架构、训练过程或提示的情况下，灵活控制语法纠错任务中精确率与召回率的权衡。 |
| [^27] | [TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents](https://arxiv.org/abs/2610.41697) | TSBPO提出一种树结构的采样到更新范式，通过似然拟合的根/分支分配器定向采样、基于兄弟片段与匹配观察状态的延迟反馈分配、以及跨活跃轮次归一化的词元损失聚合，在单一训练循环中解决了从终端结果训练交互式语言智能体的三大耦合难题。 |
| [^28] | [Does Reasoning Make Language Models Agree?](https://arxiv.org/abs/2610.41691) | 该论文区分了作为干预手段的推理与作为观察行为的推理，并发现在分类任务中启用推理能够增加不同语言模型之间的一致性。 |
| [^29] | [When Steering Shifts Discourse: Identifying and Intervening on Discourse-Level Changes due to Activation Steering in LLMs](https://arxiv.org/abs/2610.41661) | 论文揭示了大语言模型的激活引导会系统性改变生成文本的话语层面特征（如正式程度、证据使用和模糊性），并提出通过添加针对受影响维度的互补向量进行干预，可恢复约78%的非预期话语变化，同时保留预期的引导效果。 |
| [^30] | [GRIT: Gradient Masking for Visually Misgrounded Tokens in Text-Rich Reasoning](https://arxiv.org/abs/2610.41629) | 提出GRIT训练框架，通过梯度掩码将视觉锚定引入可验证奖励的强化学习（RLVR），以解决视觉语言模型在富文本推理中“答案正确但视觉误锚定”（CVMR）的问题。 |
| [^31] | [Saidzoda-Tajik-LFM-2.6B: Adapting a 2.6B Hybrid Model to Tajik Tool Calling, and What Continued Pretraining Takes Away](https://arxiv.org/abs/2610.41576) | 该论文提出两阶段方案（LoRA持续预训练加监督微调），将2.6B混合模型适配为塔吉克语工具调用模型，使严格准确率从20.69提升至64.14，同时发现持续预训练会抹除模型原生的工具调用能力，必须与SFT结合使用。 |
| [^32] | [BYOD: Build Your Own DLM - Efficient Autoregressive-to-Diffusion Language Model Conversion](https://arxiv.org/abs/2610.41574) | BYOD提出了一种仅用LoRA适配器的可泛化方案，能在单块GPU上数小时内将Gemma、Llama、Qwen、Mistral等自回归语言模型低成本转换为扩散语言模型，同时冻结全部基础模型参数并保留原模型知识。 |
| [^33] | [SCRBench: Beyond Equivalence Checking for Semantic Change Reasoning in Code](https://arxiv.org/abs/2610.41562) | 该论文提出SCRBench基准，通过诊断语义变化和重构代码更改序列这两项新任务，超越了传统的语义等价性“是/否”判断，从而更深入地评估语言模型对代码语义变化的真实理解能力。 |
| [^34] | [CEFR Level Assessment across Languages and Text Granularities](https://arxiv.org/abs/2610.41553) | 该论文提出将焦点损失与监督对比学习相结合的统一训练目标，有效解决了CEFR等级分布不均衡和相邻等级边界模糊的问题，在单语和多语环境下的句子级与文档级评估中均取得了新的最先进性能。 |
| [^35] | [Enhancing Knowledge Entity Relation Extraction in Abstracts and Full-text Content of Academic Articles Using Large Language Models](https://arxiv.org/abs/2610.41552) | 本研究系统比较了多种大语言模型策略用于学术论文的知识实体关系抽取，并创新性地提出了SLM增强开源LLM和闭源LLM增强开源LLM两种新组合策略。 |
| [^36] | [DAPS: Dependency-Aware Premise Selection for LLM Theorem Proving](https://arxiv.org/abs/2610.41536) | 该论文提出DAPS方法，通过首次系统挖掘53个Lean Blueprint项目构建依赖关系语料库，将数学中的依赖结构信号引入前提选择的训练数据与模型中，克服了现有基于语义相似度的选择器在检索偏斜和分布外查询上的固有局限。 |
| [^37] | [Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning](https://arxiv.org/abs/2610.41518) | 本研究比较了EDA、回译和上下文词替换三种数据增强方法在传统机器学习分类器与BERT模型上的效果，发现上下文增强对BERT模型增益最稳定，而EDA和回译更适合提升传统分类器的性能。 |
| [^38] | [Sparse Text Diffusion under Dual Budgets: Context Connectivity and Expert Activation](https://arxiv.org/abs/2610.41487) | 该论文提出以“注意力边数”和“专家激活数”两个独立预算来刻画长上下文扩散模型的计算成本，并据此分析稀疏-专家配置 DualBudget，在最高 16K 的长度范围内的摘要、多跳问答、对话等任务上优于基线。 |
| [^39] | [KEEPER: Learning Control-Sufficient Textual Memory for Long-Horizon LLM Agents](https://arxiv.org/abs/2610.41482) | 该论文提出KEEPER框架，在共享递归接口下比较四种辅助训练目标，发现基于奖励相关的预测目标能在离线强化学习前后均取得最佳任务表现，并提出应同时评估记忆的错误持续性与有效信息保留情况。 |
| [^40] | [KV-Cache Patching: Tracing How and Where Fine-Tuned Behaviors Arise](https://arxiv.org/abs/2610.41480) | 通过在LoRA微调模型与基础模型之间互相修补KV缓存，研究揭示了微调行为主要通过注意力机制对先前token（尤其是系统提示和聊天模板等少数关键位置）的 altered 处理被编码在KV缓存中，可以跨模型转移或消除。 |
| [^41] | [A prospective randomized study of patient-centered AI for clinical visit preparation](https://arxiv.org/abs/2610.41437) | 本研究开发了基于大语言模型的患者就诊准备智能体，并通过802名成年人参与的前瞻性随机试验证明，该智能体在帮助患者做好就诊准备方面显著优于通用聊天机器人。 |
| [^42] | [BlindToChange: AI Agents Can Find Changes When Asked But Do They Look When Needed?](https://arxiv.org/abs/2610.41435) | 提出BlindToChange基准，揭示AI智能体虽能在明确要求下找到近期变化，但在用户仅陈述信息需求时往往不会主动发现并考虑影响回答的最新事件。 |
| [^43] | [Hybrid Computer Use: A Unified Programmatic, Semantic, and Visual Interface for General Agents](https://arxiv.org/abs/2610.41432) | 该论文提出混合计算机使用（HCU），一个统一融合编程式、语义与视觉三种交互通道的通用智能体接口，使智能体能够自主选择和组合各通道，从而突破传统“截图+鼠标键盘”人类交互先验的局限。 |
| [^44] | [SparseGDN: Chunk Sparse in Linear Attention](https://arxiv.org/abs/2610.41420) | SparseGDN提出了一种硬件感知的稀疏线性注意力框架，通过结构化块内近似与动态循环事件选择相结合，平均仅保留约29%的循环事件，在保持输出质量的同时实现了1.50-1.94倍的长上下文预填充加速。 |
| [^45] | [ProSearch: Benchmarking Multi-Constraint Protocol Retrieval in Experimental Science](https://arxiv.org/abs/2610.41396) | 提出了ProSearch基准，用于系统评估AI在实验科学中检索满足多个相互依赖约束的实验方案的能力，该基准基于真实实验记录构建，包含跨六个领域的1,515个带有显式约束的查询。 |
| [^46] | [Comparative Evaluation of LLM-Based Review Paradigms for Academic Paper Quality Assessment](https://arxiv.org/abs/2610.41391) | 该研究系统比较了五种基于LLM的论文评审范式，发现重新校准后的直接绝对评分能以仅24%的API调用量取得优于Elo比较排序的人类评分一致性，且仅用摘要输入以及模型选择的影响与范式选择同等重要。 |
| [^47] | [What We Keep of What We Forget: Replacement Operators and Routing Semantics in Training-Free MoE Expert Pruning](https://arxiv.org/abs/2610.41366) | 本文系统隔离并比较了免训练MoE专家剪枝中替换算子的作用，发现存活专家门控缩放是否重新归一化是影响最大的隐式实现选择，并提出了一种闭式低秩替换算子“倾斜新手”，在50%剪枝率下以仅1.2-1.8%的参数增量获得显著性能提升。 |
| [^48] | [SERIAL-7: Deriving, Realizing, and Preserving Machine-Verifiable Outputs Across Representations](https://arxiv.org/abs/2610.41308) | 该论文提出SERIAL-7——一个基于可执行验证框架的韩英双语基准，通过“推导—实现—保持”三个阶段和43个原子表示约束，系统评估大语言模型跨七种表示形式生成机器可验证结构化输出的能力，揭示了模型在表面答案正确之外仍存在大量工件级合规失败。 |
| [^49] | [From Transcription to Reformulation: Defining and Benchmarking Grounded Generative Speech Recognition](https://arxiv.org/abs/2610.41292) | 该论文提出了“有依据的生成式语音识别”这一新范式，将语音识别从生成单一转录文本重新定义为生成语义上有依据的文本，建立了四级重构能力层次结构，并发布了首个综合评测基准GSR-Bench。 |
| [^50] | [In-Model Context Compression with Agentic Soft Tokens for Long-Horizon LLM Agents](https://arxiv.org/abs/2610.41291) | 提出免训练的智能体软令牌压缩方法（ASTC），利用混合GDN-注意力的原生循环动态将已吸收信息压缩为模型内软令牌，在降低长时程LLM智能体计算成本的同时保留关键决策信息。 |
| [^51] | [PsyInterviewee: Informational Invariance and Relational Adaptivity in Mental Health Interview Simulation](https://arxiv.org/abs/2610.41289) | 该论文提出PsyInterviewee框架，通过分离受保护的真实信息通道与动态交互通道并采用非对称写入权限机制，在心理健康访谈模拟中同时保证临床事实的信息不变性与交互行为的适应性。 |
| [^52] | [SciGraphSelect: Auditable Scientific Data Selection through Knowledge--Reasoning Graphs](https://arxiv.org/abs/2610.41280) | 提出了SciGraphSelect框架，将领域知识、推理操作和参考证实关系组织成类型化知识-推理图，通过概念覆盖率、关系与路径共支持及方向性对齐对科学文档进行可解释评分，为科学训练语料库构建提供可审计、可追溯的数据选择。 |
| [^53] | [TRIDENT and SHAKTI: A Framework and Benchmark for Hindi AI-Text Detection](https://arxiv.org/abs/2610.41232) | 该论文针对低资源语言AI文本检测研究不足的问题，提出了大规模印地语AI文本检测基准SHAKTI，以及基于往返翻译语义一致性的三变体检测框架TRIDENT。 |
| [^54] | [Beyond Response Scoring: An AI-aware Cognitive Supervision Framework for Automated Mathematical Short Answer Grading](https://arxiv.org/abs/2610.41215) | 提出了一种双分支多任务框架AICG，将数学简答题评分与基于Q矩阵和G-DINA的认知诊断（技能掌握度与学习增益预测）相结合，并考虑生成式AI使用带来的评分偏差，以提升复杂数学答案自动评分的泛化能力。 |
| [^55] | [TraceSpec: Understanding and Improving Speculative Decoding for LLM Agents](https://arxiv.org/abs/2610.41177) | 提出TraceSpec框架，通过结合动态执行与固定前缀重放、构建跨浏览与软件工程任务的轨迹状态语料库，揭示了汇总投机解码指标掩盖的分阶段性能差异，并实现基于轨迹条件的自适应草稿监督与预测预算分配。 |
| [^56] | [Context as Infrastructure: Knowledge Compilation for Retrieval in Agentic Systems with Harnesses](https://arxiv.org/abs/2610.41172) | 提出“知识编译”应作为智能体检索的上下文基础设施：在搜索执行框架已基本解决准确性问题的前提下，通过可配置的摄取流水线为文档注入导航树、实体图、时间线等结构化上下文，充当解决上下文浪费问题的“供给层”，而非充当更准确的检索器。 |
| [^57] | [Lineage Is Memory: Parent Inheritance and Feedback Disclosure in Short-Budget LLM-Guided Optimization](https://arxiv.org/abs/2610.41157) | 该研究通过1,750次预注册实验首次将父代继承与反馈披露效应分离，发现贪婪的父代继承——即谱系本身充当记忆——才是短预算LLM引导优化中绝大部分进展的真正来源，而非每轮反馈所覆盖的失败模式广度。 |
| [^58] | [HyperRepair: Recovering Forgotten General Capabilities of Fine-Tuned Language Models](https://arxiv.org/abs/2610.41147) | HyperRepair提出了一种基于超网络的方法，通过利用激活指纹为共享预训练主干的模型生成定制的修复侧网络，在仅增加推理成本的前提下恢复微调语言模型被遗忘的通用能力，同时保留其领域专业化，且无需目标领域样本或额外优化。 |
| [^59] | [Language Models Show a "Women-Are-Wonderful" Effect in Blame Attribution That is Not Just Leniency](https://arxiv.org/abs/2610.41116) | 该研究发现语言模型在人际冲突的责备归因中存在角色依赖的“女性很美好”效应：将受影响方设为女性会使被责备几率增加 2.7-9.2 倍，而将被评判方设为女性则使被责备几率降低 9%-52%。 |
| [^60] | [MUON-RT Reusing the Teacher's Trajectory for Momentum-Free Muon for LLM Self-Distillation](https://arxiv.org/abs/2610.41104) | 提出Muon-RT，通过重用自蒸馏中教师与学生的位移替代Muon优化器的动量缓冲区，在降低训练内存开销的同时保证了理论收敛性，并提升大语言模型自蒸馏微调的性能。 |
| [^61] | [EDU-Jev Training System-One Decision Models for Education](https://arxiv.org/abs/2610.41079) | 该论文提出 EDU-Jev 系列面向教育评估任务的单次前向决策模型（0.8B/2B/4B 参数），通过指针头直接输出预定义选项上的概率分布而无需生成文本，实现了更快、更省、更可靠的教育自动化决策。 |
| [^62] | [Capability-Grounded Agentic Planning](https://arxiv.org/abs/2610.41040) | CGAP 通过将工具目录确定性编译为抽象 PDDL 模型并使用经典规划器求解，使 LLM 智能体生成的计划天然可执行，同时在执行时由 LLM 补充实体系调用并通过确定性检查确保不破坏原计划。 |
| [^63] | [Do Language Models Know What Compelling Scientific Evidence Looks Like?](https://arxiv.org/abs/2610.41015) | 论文提出了基于真实生物学数据的科学推理基准EDB-Bio，发现现有语言模型在权衡科学证据时存在系统性信心不足，表现仅与简单基率预测器相当，而针对严格恰当评分规则的后训练可显著改善模型的校准与判别能力。 |
| [^64] | [Language Models Show Conformity Bias in RAG: When the Majority Endorses a Poisoned Claim](https://arxiv.org/abs/2610.41012) | 本研究发现当RAG检索上下文中多数相互一致的投毒文档共同支持错误信息时，语言模型的投毒检测率从25%显著降至10%，揭示了语言模型存在类似社会从众心理的从众偏差，使其更容易被多数派的投毒信息误导。 |
| [^65] | [MOSAIC: Addressing Disproportionate Task-Modality Dependencies via Modality-Partitioned Gradient Routing](https://arxiv.org/abs/2610.41004) | 提出MOSAIC架构，通过模态分离与模态划分的梯度路由显式建模多任务学习中各任务对特定模态的非对称依赖，在保留单模态表征的同时避免跨模态的梯度干扰。 |
| [^66] | [Do Open Language Models Invent Scripture? BILAL-BENCH: Benchmarking Islamic-Source Fabrication in Bengali](https://arxiv.org/abs/2610.40996) | 该论文提出BILAL-BENCH——首个针对孟加拉语开放式伊斯兰文献来源伪造的专家验证基准，评测发现七个开放权重模型中有六个在79.3%-95.3%的来源生成问题上自信地编造经文内容，且性能不随模型规模提升而改善。 |
| [^67] | [The psychophysics of affect in large language models](https://arxiv.org/abs/2610.40985) | 本研究首次系统测量了大语言模型将结果映射为情感的心理物理规律，发现模型对收益的情感反应强于等额损失（与人类损失厌恶相反的“硅基收益放大”现象），且这种结果到情感的映射会随评分情境和决策任务而显著改变，具有高度可塑性。 |
| [^68] | [Fewer Heads, Fewer Memories: Pruning Redundant Attention Heads Suppresses Verbatim Memorisation](https://arxiv.org/abs/2610.40974) | 本研究发现在推理前将大量冗余注意力头置零，可以在保持甚至提升语言建模性能的同时，显著抑制大语言模型对训练数据的逐字记忆复现。 |
| [^69] | [Nedimo: An Evidence-Grounded Multilingual Pipeline for AI-Generated Sleep Stories](https://arxiv.org/abs/2610.40970) | 该论文提出了Nedimo——一个将不确定的临床证据转化为保守工程需求的多语言流水线，通过将确定性规划与大语言模型生成分离、四阶段注意力渐减、结构和词汇约束验证以及有界错误感知重试等机制，来生成有助于入睡的低唤醒度AI睡眠故事。 |
| [^70] | [Better Retrievers, Worse Labels: How Incomplete Relevance Judgments Distort Abstention Evaluation in Conversational RAG](https://arxiv.org/abs/2610.40966) | 该论文揭示，在对话式RAG评估中，“无黄金文档即不可回答”的捷径会因相关性判断不完整而严重失真——检索器越强，被遗漏但实际含答案的上下文比例越高（28%至72%），导致该捷径的精确率从0.68骤降至0.10。 |
| [^71] | [Kernel-based Interpolatory Methods for Context Compression of Attention Mechanisms](https://arxiv.org/abs/2610.40943) | 本文在核所对应的希尔伯特空间中将注意力表示为两个线性映射之比，并分别基于广义经验插值方法（GEIM）和Q-离散经验插值方法（Q-DEIM）提出了两种新颖的KV缓存压缩方法GAIM和Q-DAIM。 |
| [^72] | [Adaptive Personalization Under Preference Epistemic Uncertainty](https://arxiv.org/abs/2610.40930) | 本文提出并形式化了“偏好认知不确定性下的自适应个性化”任务，构建了一个涵盖250个人设、5种语言、25K轮对话的多语言基准，揭示了现有大模型普遍存在的“识别-生成鸿沟”——即模型能从候选选项中正确识别偏好适应策略，却在开放式生成中失效。 |
| [^73] | [Agentic World Modeling for Agent Test-time Scaling without Environment Replica](https://arxiv.org/abs/2610.40927) | 提出智能体式世界建模（AWM），将下一观测预测重构为多轮证据获取任务，通过安全只读接口自适应查询真实环境来模拟候选动作，从而无需复制环境即可实现大语言模型智能体的测试时扩展。 |
| [^74] | [CoFiBreak: A Coarse-to-Fine Diversity Reward for Effective Multi-Turn Jailbreak](https://arxiv.org/abs/2610.40926) | 提出CoFiBreak，一种面向GRPO的由粗到细多样性奖励机制，在保持高攻击效果与低推理成本的同时，显著提升了多轮越狱攻击的策略与实现多样性。 |
| [^75] | [Qistas: A Pre-Registered, Portable Protocol for Fairness Evaluation of Large Language Models](https://arxiv.org/abs/2610.40893) | 提出了Qistas——一个预注册、可跨国移植的大语言模型公平性评估协议，通过配对反事实设计与严格的统计检验来衡量模型对一个社会不同社会、教派和地区群体处理的程序公平性，并首先在伊拉克落地实施。 |
| [^76] | [Incongruence-Aware Evaluation of Multimodal Speech Emotion Recognition in Audio Large Language Models](https://arxiv.org/abs/2610.40883) | 该论文提出一种基于双分量高斯混合模型、将文本与音频在情感空间中的距离转化为连续无阈值得分的不一致性度量方法，并发现文本与音频情感表达的不一致会在所有语料库中系统性增加音频大语言模型语音情感识别的错误率，而监督分类器则表现出相反的行为。 |
| [^77] | [FARO: Function-Space Adaptive Return-to-Risk Optimization via Cross-Block Equal-Risk-Contribution Budget Allocation](https://arxiv.org/abs/2610.40858) | FARO提出了一种以参数块为决策单元的函数空间自适应优化器，根据各块的反事实收益自动分配更新几何，并借助投资组合理论中的等风险贡献原则跨块重新分配更新预算，在GPT模型训练上达到与最佳手工设计优化器配方相当的性能。 |
| [^78] | [The Generative Gap Between Speculative Drafter and Continuous Diffusion Language Model](https://arxiv.org/abs/2610.40847) | 该研究在无目标模型验证的公平条件下，从零训练并比较连续扩散语言模型与投机草稿架构，发现基于 CDLM 的 LangFlow 在生成困惑度和 MAUVE 指标上均优于 DSpark 的草稿架构。 |
| [^79] | [Safin-1.5: Safety-Native Architecture and Verifiable Reasoning for Recursive Safety Improvement](https://arxiv.org/abs/2610.40751) | 提出Safin-1.5模型，通过将持久化安全状态与危险知识解耦相结合的安全原生架构，以及递归安全改进训练和推理时形式化推理框架，使大语言模型在自我能力提升的同时持续发现并修复安全风险。 |
| [^80] | [Mage-VL-Cowork: A Full Training Recipe to Learn Hybrid Computer Use From Evolved Harnesses](https://arxiv.org/abs/2610.40747) | 提出了Mage-VL-Cowork——一个4B混合型计算机使用智能体，通过五阶段渐进式训练课程、可大规模低成本合成任务的Mage-CUA-Forge以及进化式教师框架生成高质量轨迹，在OSWorld-Verified上达到66.0%的性能。 |
| [^81] | [Prevented is Not Protected: What Attack Success Rate Hides about RAG](https://arxiv.org/abs/2610.40742) | 该论文提出“对齐RAG评估”（ARE）框架，通过匹配查询协议揭示攻击成功率指标所掩盖的问题：现有RAG防御方法虽能阻止部分投毒攻击，却可能引入意外的目标正向回答。 |
| [^82] | [Apriel Voice 27B Preview: An Open Model for Speech, Reasoning, and Tool Use](https://arxiv.org/abs/2610.40738) | 该论文提出了开源语音-语言模型 Apriel Voice 27B Preview，它在统一的推理主干上整合了语音理解、上下文推理与工具调用能力，并通过两阶段训练方法使文本模型高效适应语音模态。 |
| [^83] | [Li$^{2}$SAO: Lifting Light Singl-Rollout Asynchronous RL Optimization via Group-Relative Actor-Critic Bootstrapping](https://arxiv.org/abs/2610.40726) | 该论文提出Li²SAO，通过组相对Actor-Critic自举摆脱对大规模预训练价值模型的依赖，在轻量级单rollout异步RL优化中兼顾推理任务与智能体任务的表现。 |
| [^84] | [Agentic Systems for Calibrating Cosmological Simulations](https://arxiv.org/abs/2610.40685) | 本文提出AgenticTuner智能体校准系统，利用大语言模型作为优化器并结合人类可读的先验知识，以远低于传统贝叶斯优化的CPU成本实现具有竞争力的宇宙学模拟校准性能。 |
| [^85] | [Robustness of Chatbot LLM Judgments of Presidential Actions and Hypothetical Reforms](https://arxiv.org/abs/2610.39398) | 该研究通过30天内每天两次询问五种聊天机器人配置对总统行动和假设性改革进行评分，发现LLM的判断总体稳健——各配置对评分符号与排名高度一致，同日不同配置给出相反评分的情况仅约4%。 |
| [^86] | [Unlimited OCR Works 2](https://arxiv.org/abs/2610.39308) | 通过将全部解码器注意力模块升级为稀疏参考滑动窗口注意力（SR-SWA），Unlimited OCR 2 将活跃参考令牌限制在固定容量内，实现了对任意超长文档以恒定计算成本进行解析。 |
| [^87] | [Tracing the Influence of Retrieved Literature on Scientific Ideation in LLMs](https://arxiv.org/abs/2610.39060) | 该研究通过在统一生成协议下对比五种文献检索策略与无文献基线（共45,000个想法），首次分离并量化了检索策略本身对LLM科学构思的影响，发现检索会降低同组论文生成想法的多样性但提高跨组想法的多样性，并提出基于JSD的归因方法追踪单篇文献对每个想法的影响。 |
| [^88] | [When Support Crosses the Line: Evaluating Ethical Boundaries in LLM-Based Mental Health Support in the Chinese Context](https://arxiv.org/abs/2610.38452) | 提出了基于中国心理健康专业伦理标准的MindEth-Bench基准，利用蒙特卡洛树搜索评估九个大语言模型在多轮心理健康交互中对伦理边界的遵守情况，发现模型在长期交互中可能从恰当回应滑向违反伦理的行为。 |
| [^89] | [Holistic Diversity Management in LLM-Guided Evolutionary Search](https://arxiv.org/abs/2610.38407) | 提出HoliDivEvolve方法，通过多样化初始化、岛屿轨迹发展和跨岛屿发现共享，在LLM引导的进化搜索全过程中建立并有效利用岛屿多样性，以避免过早收敛。 |
| [^90] | [Keepace.ai: A Technical Report on a Large Language Model for Sports and Health](https://arxiv.org/abs/2610.38249) | Keepace是基于Qwen3.5-27B后训练的中文运动健康领域专用大语言模型，在健身计划生成、知识问答和运动数据分析三项任务上超越了基座模型及更大的通用模型，尤其在运动安全性、边界合规性和科学准确性方面提升显著。 |
| [^91] | [DeFT-RAG: Incrementally Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers](https://arxiv.org/abs/2610.37826) | DeFT-RAG通过增量式地将多跳问题分解为携带可复用答案的自包含子问题，逐步恢复完整的证据链，从而提升RAG在多跳问答任务上的准确性。 |
| [^92] | [Heads of Security: Learned Attention Gates Improve Prompt Injection Robustness](https://arxiv.org/abs/2610.37657) | 提出Heads of Security（HoS）防御方法，通过为每个注意力头学习注意力门控来实现指令与数据的分离，使模型能在处理不可信内容时不执行注入指令，相比现有防御方法显著提升了对自适应提示注入攻击的鲁棒性并保持模型效用。 |
| [^93] | [Metacognitive Decoding: Efficient Inference for Diffusion Language Agents](https://arxiv.org/abs/2610.37642) | 提出一种轻量级元认知控制器插件，使扩散语言智能体学会何时确定token及生成多长文本，在保持任务准确率的同时平均减少20.6%的推理计算量。 |
| [^94] | [SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2610.36991) | SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。 |
| [^95] | [Adapting LLM-SC to Living Sacred Heritage: Domain-Specific Extraction, Semantic Networks and Human Evaluation on Pashupatinath Temple Reviews](https://arxiv.org/abs/2610.35685) | 该研究将LLM-SC框架适配到尼泊尔帕斯帕提那神庙这一活态神圣遗产场景，通过新增8种神圣遗产实体类型构建12类实体分类体系，从Tripadvisor评论中成功提取结构化数据，且新增的神圣遗产类型占实体记录的64.8%，证明领域特定适配对宗教遗产评论分析至关重要。 |
| [^96] | [SOPO: Socratic Guided Policy Optimization for Span-Level Hallucination Detection](https://arxiv.org/abs/2610.32212) | SOPO提出了一种苏格拉底引导的多轮强化学习框架，通过教师模型提供基于真实标签的间接提示来引导推理轨迹，从而解决片段级幻觉检测中的定位模糊、困难样本奖励稀疏和长度膨胀问题。 |
| [^97] | [Table Serialization Determines Which Relations Language Models Preserve](https://arxiv.org/abs/2610.24798) | 该研究发现表格的序列化格式决定了语言模型在表格查找中保留哪些关系——行优先格式更适合窄高表格（倾向保留行但丢失列），列优先格式更适合宽扁表格（倾向保留列但丢失行），且这一结论在多个模型上保持一致。 |
| [^98] | [DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents](https://arxiv.org/abs/2610.21561) | DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。 |
| [^99] | [Gluing Local Contexts into Global Meaning: A Sheaf-Theoretic Decomposition of Transformer Representations](https://arxiv.org/abs/2610.17117) | 该论文提出一种无需训练的层论方法：在改写图上构建胞腔层并用其拉普拉斯算子的核 $H^0$ 提取内容子空间，仅 20 维即跨 8 个模型在 CounterFact 事实检索上平均超越 LEACE 全维子空间 +17.1 个百分点，且联合约束同一事实的所有改写可再提升 +6.5 个百分点。 |
| [^100] | [Where Is the "Self" of a Large Language Model? -- From Self-Index to Self-Reference](https://arxiv.org/abs/2610.16717) | 该研究通过在模型自身消息与外部来源消息之间进行表征对比，首次证明大语言模型同时具备“自我索引”（在所有层中持续区分自身与外部文本）和“自我指涉”（能认领自己写的消息）两个类似生物功能层次的“自我”证据。 |
| [^101] | [Thinking Inertia: LLMs Keep Thinking When Told Not To](https://arxiv.org/abs/2610.11765) | 该论文提出了三项新指标（空思考率、问题-答案前相关性、显式推理率）来严谨地衡量大语言模型的“不思考”行为，并揭示了一种“思维惯性”现象——即使被明确要求不思考，LLM仍会继续进行推理。 |
| [^102] | [4-Tensor Attention Model for Semantic Physical Reality](https://arxiv.org/abs/2610.11716) | 该论文提出一种四阶张量注意力模型，通过在语义纤维与时间上下文纤维上进行联合归一化注意力来预测场景的下一语义状态，在参数量几乎相同的情况下，其最后一句交叉熵在三个设置上均优于自由运行的一维 Transformer（最多低 5.3%），为视频生成和机器人规划提供了新方法。 |
| [^103] | [Language Carries the Expert's Impression: Instrument-Anchored LLM Judges Transfer Counseling-Quality Assessment and Beat In-Domain Training](https://arxiv.org/abs/2610.08055) | 该研究发现，基于专家评分工具锚定构念、由小型开源LLM生成的会话级得分，能够跨领域迁移专家对咨询质量的总体印象评估，且跨域训练的效果甚至优于域内训练。 |
| [^104] | [What Does a Harness Repair? A Preregistered Study of Visibility, Baseline Adequacy and Evaluation Defects](https://arxiv.org/abs/2610.05533) | 本预注册研究通过系统实验发现，测试框架搜索带来的评测分数提升往往源于解析器可见性问题和评估缺陷等伪影而非真实能力增益，且简单的关闭思考设置在多数情况下足以媲美甚至胜过经过搜索优化的框架配置。 |
| [^105] | [A Unified Per-Token Gating Family for On-Policy Distillation: FKL/RKL Mixing with Multi-Channel and Bias Coefficients](https://arxiv.org/abs/2609.11768) | 该论文提出了一个统一的四系数逐词元门控参数化方法，将EOPD和ToDi作为其一维限制的特殊情形统一起来，并通过多通道组合与显式偏置这两个额外自由度，在在线策略蒸馏实验中显著超越了单通道门控限制方法。 |
| [^106] | [SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control](https://arxiv.org/abs/2609.11752) | SIRF通过持续预训练将平台风控规范内化到模型权重中，无需人工标注即可在超低延迟、仅输出判定的部署形态下实现高精度风险处置，仅用约7000万token的CPT便将P95精确率下的黑名单召回率提升15.1个百分点且不损害通用能力。 |
| [^107] | [TamilEOT: A Dataset and Model for Semantic End-of-Turn Detection in Tamil Telephone Speech](https://arxiv.org/abs/2609.05631) | 本文发布了首个面向泰米尔语电话语音的语义话轮结束检测资源——包含18,485个标注话轮边界的TamilEOT数据集及两个微调检测模型，将准确率从零样本的70.30%提升至86.13%，并可在普通CPU上以低于150毫秒的延迟实时运行。 |
| [^108] | [The Depth Flow of Token Representations Is Nonlinear and Does Not Descend Its Own Density](https://arxiv.org/abs/2608.29706) | 该研究用离散朗之万模型刻画语言模型中词元表示的逐层演化流，发现该流是非线性的、并不沿自身对数密度下降而是下降一个不同的势函数，且旋转分量占可解释漂移的 4%–45%。 |
| [^109] | [From Association to Causation: Improving Retrieval Precision of Retrieval-Augmented Generation via Causal Relations and an Attention Mechanism](https://arxiv.org/abs/2608.21702) | 该论文提出通过因果图建模检索阶段，利用对撞子结构中的注意力机制，无需训练即可提升RAG检索精度，解决相似度仅捕捉关联而忽略因果的问题。 |
| [^110] | [You Only Pass Once: Answering and Abstaining Together in a Single Forward Pass of a Frozen Language Model](https://arxiv.org/abs/2608.14465) | 本文提出了一种在冻结语言模型的单次前向传播中同时实现回答和弃权的方法，通过训练小型网络重建引导前的残差来消除干扰，从而兼顾推理准确性和弃权能力。 |
| [^111] | [The No-Meaning Falsity: The Structural Impossibility of the Arbitrary Sign in Classical Arabic](https://arxiv.org/abs/2608.07737) | 本文通过为古典阿拉伯语非拼接形态学建立形式化数学模型，证明词根-模式系统在结构上排除了索绪尔符号任意性公理——词汇意义在派生层面即被唯一确定，从而从形式上反驳了后现代无限语义不确定性的主张。 |
| [^112] | [CogBattery: Human-Referenced Evaluation of Cognitive Task Behavior and Structure in Language Models](https://arxiv.org/abs/2607.24999) | CogBattery通过13个认知范式将54个开源语言模型与人类数据进行对比评估，发现模型虽能复现人类效应的方向，但在错误模式、元认知置信度等方面与人类存在系统性差异，且更大规模的模型并不必然更接近人类认知表现。 |
| [^113] | [MS-GPT: Rethinking MS/MS De Novo Structure Elucidation as Spectrum-Induced Posterior Querying of a Molecule-Language Model](https://arxiv.org/abs/2607.23607) | MS-GPT提出将串联质谱的从头分子结构解析重新建模为对条件分子语言模型的谱诱导后验查询，从而解决了传统指纹范式中“神谕指纹训练”与“噪声光谱指纹推理”之间的失配问题。 |
| [^114] | [Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context](https://arxiv.org/abs/2607.21535) | 提出Windowed-MTP方法，仅对MTP草稿头施加StreamingLLM风格的窗口化注意力加注意力汇聚机制，无需训练即可消除百万token长上下文中草稿KV缓存的高昂读取开销，同时保持目标模型全上下文验证的分布不受影响。 |
| [^115] | [Imputation of Public Opinion Data with Large Language Models](https://arxiv.org/abs/2606.09351) | 本文首次采用规范的多重插补评估标准，在超过500万条舆论调查数据上系统评估了84种上下文学习配置的大语言模型插补方法，发现其最优表现仅能与MICE PMM、TabPFN等统计基线方法相当，且对调查主题高度敏感。 |
| [^116] | [EvoMemBench: Benchmarking Agent Memory from a Self-Evolving Perspective](https://arxiv.org/abs/2605.18421) | 该论文提出了首个从自我进化视角出发、沿记忆范围与记忆内容两个维度系统评估LLM智能体记忆能力的统一基准EvoMemBench，并通过对比15种记忆方法与长上下文基线，揭示出当前记忆系统尚无通用解决方案。 |
| [^117] | [VerifyMAS: Hypothesis Verification for Failure Attribution in LLM-based Multi-Agent Systems](https://arxiv.org/abs/2605.17467) | 提出VerifyMAS框架，将LLM多智能体系统的故障归因转化为在完整交互轨迹上进行故障假设的提出与验证，从而捕捉局部日志方法无法发现的全局性故障并实现细粒度归因。 |
| [^118] | [Controllable User Simulation](https://arxiv.org/abs/2605.11519) | 该论文将可控用户模拟形式化为因果推断问题，揭示了基于事后轨迹标签进行监督微调会引入破坏因果一致性的结构性前瞻偏差，并证明在策略偏移下这种失败会导致评估指标方差几何级数爆炸的“可控性崩溃”现象。 |
| [^119] | [Minimizing Modality Gap from the Input Side: Your Speech LLM Can Be a Prosody-Aware Text LLM](https://arxiv.org/abs/2605.05927) | 该论文提出TextPro-SLM，通过统一语音编码器WhisperPro生成同步的文本token与韵律嵌入，从输入侧最小化语音与文本之间的模态差距，使语音LLM在保留原始文本LLM语义能力的同时，在3B和7B规模下实现最低模态差距并具备出色的副语言理解能力。 |
| [^120] | [A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle](https://arxiv.org/abs/2604.16548) | 本文提出记忆生命周期框架与可验证记忆治理（VMG）框架，系统刻画了LLM智能体长期记忆所面临的持久性、有状态性与传播性等新型安全威胁，并沿六个生命周期阶段和四个安全目标组织了攻击、防御及其跨阶段依赖关系的全景式分析。 |
| [^121] | [M$^\star$: Every Task Deserves Its Own Memory Harness](https://arxiv.org/abs/2604.11811) | 提出Mstar方法，通过反思性代码演化自动为每个任务生成并联合优化包含模式、逻辑和指令的可执行Python记忆程序，在对话、具身规划和专业推理等任务上均超越现有基线。 |
| [^122] | [Optimal Multi-bit Generative Watermarking Schemes Under Worst-Case False-Alarm Constraints](https://arxiv.org/abs/2604.08759) | 本文证明了大语言模型多比特生成式水印中现有方案是次优的，并提出两种能达到漏检概率理论下界的新型编码-解码构造，通过线性规划刻画了最优性条件，从而完整解决了最坏情形误警约束下的最优多比特水印设计问题。 |
| [^123] | [Can Humans Tell? A Dual-Axis Study of Human Perception of LLM-Generated News](https://arxiv.org/abs/2604.03755) | 基于1,054名参与者2,318条判断的研究表明，人类无法可靠地区分大语言模型生成的新闻与人类撰写的新闻，即使面对参数量仅7B的小型开源模型也是如此。 |
| [^124] | [AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents](https://arxiv.org/abs/2603.26034) | AgentCollab提出了一种由自评估驱动的协作推理框架，利用智能体自身的反思信号动态协调不同能力级别的LLM模型，仅在必要时才升级至更强的推理层级，从而在执行效率与推理鲁棒性之间取得平衡。 |
| [^125] | [Enhancing Safety of Large Language Models via Embedding Space Separation](https://arxiv.org/abs/2603.20206) | 提出ES2微调方法，通过增大有害与无害嵌入之间的距离来增强大语言模型对抗性攻击的安全性，并利用嵌入锚定损失和KL散度损失两种正则化手段来保持模型的原始能力。 |
| [^126] | [SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions](https://arxiv.org/abs/2603.07379) | 该SoK论文首次将智能体化检索增强生成（Agentic RAG）统一形式化为部分可观测马尔可夫决策过程，并在此基础上提出了全面的分类体系、模块化架构分解与研究方向。 |
| [^127] | [Exploring the potential and limitations of Model Merging for Multi-Domain Adaptation in ASR](https://arxiv.org/abs/2603.05354) | 该研究将模型合并应用于多领域语音识别，提出新算法BoostedTSV-M，在10个欧洲葡萄牙语领域上超越了完整微调，同时保留了分布外性能。 |
| [^128] | [Expected Harm: Rethinking Safety Evaluation of (Mis)Aligned LLMs](https://arxiv.org/abs/2602.01600) | 提出以执行可能性加权的“预期危害”安全评估新指标，揭示现有大语言模型存在“逆向风险校准”问题——对高成本低可能性的威胁过度拒绝，却对低成本高可能性的威胁依然脆弱。 |
| [^129] | [Eroding the Truth-Default: A Causal Analysis of Human Susceptibility to Foundation Model Hallucinations and Disinformation in the Wild](https://arxiv.org/abs/2601.22871) | 该研究通过双轴框架和结构因果模型分析918次评估发现，人类易受AI幻觉与虚假信息误导的关键因素并非政治倾向，而是缺乏“假新闻熟悉度”，且GPT-4等高流畅度输出会形成绕过来源监控的“流畅性陷阱”。 |
| [^130] | [Ctrl-G: Controlling Structural and Interaction Exposure in Knowledge Graph Augmented LLM Reasoning](https://arxiv.org/abs/2601.08739) | 提出Ctrl-G，这是首个能联合控制知识图谱增强LLM推理中结构暴露（拓扑信息）与交互暴露（多次调用累积信息）的隐私保护KG-RAG框架，同时保留多跳推理所需的桥接路径。 |
| [^131] | [Natural Language Actor-Critic: Scalable Off-Policy Learning in Language Space](https://arxiv.org/abs/2512.04601) | 提出自然语言演员-评论家算法NLAC，通过生成式LLM评论家产生自然语言评论来训练LLM策略，实现语言空间中的可扩展离策略学习，解决了稀疏奖励长时程任务中训练不稳定与探索困难的问题。 |
| [^132] | [Service, Solidarity, and Self-Help: A Comparative Topic Modeling Analysis of Community Unionism in the Boot and Shoe Union and Unite Community](https://arxiv.org/abs/2509.01529) | 本研究运用BERTopic主题建模方法，对比分析了1920年代全国制鞋工会与2010-2020年代Unite Community的社区工会主义话语，发现Unite Community更契合外向型、社会正义导向的社区工会特征，而制鞋工会则体现了传统的服务型工会模式。 |
| [^133] | [Beyond the Black Box: Integrating Lexical and Semantic Methods in Quantitative Discourse Analysis with BERTopic](https://arxiv.org/abs/2508.19099) | 本文提出一个结合词汇与语义方法的透明混合框架，利用NLTK、spaCy、Sentence Transformers和BERTopic等技术替代黑箱软件，实现定量话语分析的三角验证、可重复性与可解释性。 |
| [^134] | [MedResearcher-R1: Expert-Level Medical Deep Researcher via A Knowledge-Informed Trajectory Synthesis Framework](https://arxiv.org/abs/2508.14880) | 该论文提出医学深度研究智能体 MedResearcher-R1，其核心创新是知识知情轨迹合成（KISA）——通过构建医学知识图谱生成以罕见医学实体为中心的复杂多跳问答训练数据，并配合其他创新（摘要中被截断），从而突破医学领域高质量训练数据稀缺与临床推理能力不足的瓶颈，实现专家级医学深度研究。 |
| [^135] | [Optimizing Anytime Reasoning via Budget Relative Policy Optimization](https://arxiv.org/abs/2505.13438) | 提出AnytimeReasoner框架，通过截断思考过程并引入可验证的密集奖励，结合预算相对策略优化，优化大语言模型在不同token预算约束下的任意时刻推理性能，提升token效率与部署灵活性。 |
| [^136] | [Beyond Public Access in LLM Pre-Training Data](https://arxiv.org/abs/2505.00020) | 研究者利用DE-COP成员推断攻击检测版权书籍内容，发现GPT-4o表现出对其未获授权训练的非公开付费内容的识别能力（AUROC达0.82），这暗示可能存在超越“训练数据包含”之外的数据访问违规行为。 |

# 详细

[^1]: 面向统一多模态模型的连续空间潜在推理

    Latent Reasoning in Continuous Space for Unified Multimodal Models

    [https://arxiv.org/abs/2610.42149](https://arxiv.org/abs/2610.42149)

    本文提出LARC框架，通过在连续隐状态空间而非离散token空间中进行推理，解决了统一多模态模型中语言推理与图像生成之间的模态不匹配问题，并结合课程化监督微调与信息增益强化学习进行训练。

    

    统一多模态模型（UMMs）在多样的多模态理解与生成任务上取得了显著成功，但它们仍难以有效地利用基于语言的思维链推理来进行图像生成。因此，现有的推理增强型统一多模态模型往往无法继承大语言模型所具备的测试时扩展能力和细粒度可控性。我们将这一失败归因于一个根本性的模态不匹配问题：推理在离散的token空间中进行，而图像生成则在连续空间中进行。为了弥合这一差距，我们提出了连续空间中的潜在推理框架（LARC），通过在连续隐状态上进行推理来缓解token级别的瓶颈。具体而言，该训练框架包含两个阶段：（i）课程化监督微调（SFT），以及（ii）信息增益强化学习（RL）。首先，课程化SFT阶段逐步将t（原文在此处截断）

    arXiv:2610.42149v1 Announce Type: cross  Abstract: Unified Multimodal Models (UMMs) achieve remarkable success on diverse multimodal understanding and generation tasks, yet they still struggle to effectively leverage language-based chain-of-thought reasoning for image generation. As a result, existing reasoning-augmented UMMs often fail to inherit the test-time scaling and fine-grained controllability observed in large language models. We attribute this failure to a fundamental modality mismatch: reasoning is performed in a discrete token space, whereas image generation operates in a continuous space. To bridge this gap, we introduce a Latent Reasoning framework in Continuous space (LARC), which alleviates the token-level bottleneck by reasoning over continuous hidden states. Specifically, the training framework consists of two stages: (i) curriculum supervised fine-tuning (SFT), and (ii) information-gain reinforcement learning (RL). First, the curriculum SFT stage gradually converts t
    
[^2]: Flide：利用精确状态分叉加速决策推理

    Flide: Accelerating Decision Inference with Exact State Forking

    [https://arxiv.org/abs/2610.42148](https://arxiv.org/abs/2610.42148)

    Flide 是一种精确高效的决策推理引擎，通过 ForkAttention 在单一扁平化 token 流中实现精确的状态分叉，将决策请求的逻辑树结构与物理执行解耦，从而消除前缀重计算开销并保证分支求值结果与独立前向传播完全一致。

    

    诸如 Jev 之类的决策模型已成为 LLM 智能体系统中极具前景的范式，能够为快速的系统一（System One）执行提供轻量且可靠的类别推理。然而，现有的面向生成的推理引擎无法有效支持这类工作负载，存在严重的前缀重计算开销或输出概率失真问题。为应对这一挑战，我们提出 Flide，一个高效且精确的推理引擎，它将决策请求的逻辑树结构与在单一扁平化 token 流中的物理执行解耦。通过 ForkAttention，Flide 借助驻留于 GPU 的索引，在标准注意力层和循环 Gated DeltaNet 层中导航请求间的依赖关系，并以精确算术保证每个查询分支的求值结果与隔离的前向传播完全一致。通过将分支拓扑直接表达为 GPU 驻留元数据，Flide 能够适应多样的请求……（原文摘要在此处截断）

    arXiv:2610.42148v1 Announce Type: new  Abstract: Decision models such as Jev have emerged as a highly promising paradigm for LLM agent systems, providing lightweight and reliable categorical reasoning for rapid System One execution. However, existing generation-oriented inference engines fail to support these workloads effectively, suffering from severe prefix recomputation overheads or distorted output probabilities. To address this challenge, we propose Flide, an efficient and exact inference engine that decouples the logical tree structure of a decision request from its physical execution within a single flattened token stream. Through ForkAttention, Flide navigates request dependencies using GPU-resident indices across both standard attention and recurrent Gated DeltaNet layers, guaranteeing in exact arithmetic that every query branch evaluates identically to an isolated forward pass. By expressing branch topologies directly in GPU-resident metadata, Flide accommodates diverse requ
    
[^3]: DecisionRank：面向决策模型重排器的在线策略蒸馏

    DecisionRank: On-Policy Distillation for Decision-Model Rerankers

    [https://arxiv.org/abs/2610.42128](https://arxiv.org/abs/2610.42128)

    提出DecisionRank，通过条目空间在线策略蒸馏将生成式LLM的排序知识迁移到决策模型重排器中，在TREC DL、BEIR和BRIGHT等检索基准上达到最先进性能。

    

    决策模型通过直接预测选择或分数来处理结构化任务，从而避免了自回归解码带来的延迟。然而，将生成式大语言模型（LLM）的排序知识迁移到决策模型仍然具有挑战性：教师模型通过文档ID序列的排序列表来表达偏好，而学生模型则依赖首token位置的分数来确定排序。为了弥合这一差距，我们提出了DecisionRank，一种通过条目空间在线策略蒸馏（OPD）训练的决策模型。条目空间OPD通过将学生模型的文档分数排序成排名序列来构建其rollout。生成式教师模型以得到的排序列表作为输入，提供条件反馈以直接监督这些分数。结合离线排序监督，DecisionRank在TREC DL、BEIR和BRIGHT等通用及推理密集型检索基准上取得了最先进的性能。

    arXiv:2610.42128v1 Announce Type: new  Abstract: Decision models target structured tasks by directly predicting choices or scores, avoiding autoregressive decoding latency. However, transferring a generative LLM's ranking knowledge to a decision model remains challenging: the teacher expresses preferences through a ranked list of document ID sequences, while the student determines the ranking using scores from the first-token position. To bridge this gap, we introduce DecisionRank, a decision model trained through item-space on-policy distillation (OPD). Item-space OPD constructs the student's rollout by sorting its document scores into a ranked sequence. A generative teacher takes the resulting ranking list as inputs and provides conditional feedback to directly supervise those scores. Combined with offline ranking supervision, DecisionRank achieves state-of-the-art performance on both general and reasoning-intensive retrieval benchmarks such as TREC DL, BEIR, and BRIGHT. Across four 
    
[^4]: 基于分层树结构索引器的次二次复杂度注意力机制

    Subquadratic Attention with a Hierarchical Tree-Structured Indexer

    [https://arxiv.org/abs/2610.42112](https://arxiv.org/abs/2610.42112)

    本文提出 Zyphra 分层注意力（ZHA），通过在 KV 缓存上构建递归压缩键摘要的树结构索引器，使包含路由在内的总计算成本随序列长度次线性增长，从而突破了注意力二次方复杂度对大语言模型长上下文能力的根本限制。

    

    注意力的二次方扩展是限制大语言模型（LLM）长上下文能力的一个根本瓶颈。已有多种方法试图规避这一问题，但它们要么无法匹配注意力机制的表达能力（例如滑动窗口或具有固定大小状态的线性注意力变体），要么只能实现常数因子的加速（例如稀疏注意力方法，其索引器本身的复杂度仍是二次方量级）。在本文中，我们基于分层树结构索引器推导出一种渐近次二次复杂度的稀疏注意力方法，我们将其称为 Zyphra 分层注意力。ZHA 在 KV 缓存之上构建一棵由递归压缩的键摘要组成的树，后续的查询从树顶向下遍历，以检索出一个稀疏的键子集。至关重要的是，通过迭代地增加树的深度，包括路由和最终注意力计算在内的总成本随序列长度呈次线性增长……

    arXiv:2610.42112v1 Announce Type: new  Abstract: The quadratic scaling of attention is a fundamental bottleneck limiting the long-context capabilities of large language models (LLMs). A variety of approaches have attempted to circumvent this, but they either fail to match attention's expressiveness (e.g., sliding-window or linear attention variants with fixed-size state) or only achieve constant-factor speedups (e.g., sparse attention methods whose indexers themselves scale quadratically). In this paper, we derive an asymptotically subquadratic sparse attention method based on a hierarchical, tree-structured indexer, which we call Zyphra Hierarchical Attention (ZHA). ZHA constructs a tree of recursively compressed key summaries over the KV cache, which future queries traverse from the top down to retrieve a sparse subset of keys. Crucially, by iteratively growing the tree's depth, the total cost including routing and final attention grows sublinearly in sequence length, resulting in ov
    
[^5]: RangeSWA：面向连续扩散语言模型的无训练长度外推方法

    RangeSWA: Training-Free Length Extrapolation for Continuous Diffusion Language Models

    [https://arxiv.org/abs/2610.42084](https://arxiv.org/abs/2610.42084)

    RangeSWA是一种无需训练的长度外推方法，通过滑动窗口注意力、远程块摘要、动态内容检索以及位置重映射等技术，使连续扩散语言模型在超出训练长度生成文本时仍能保持质量，有效降低生成困惑度和文本重复率。

    

    连续扩散语言模型在文本生成任务中展现出可观的潜力，但当生成长度超过训练长度时，容易出现文本重复、语义漂移和生成质量下降等问题。为应对这些挑战，我们提出了RangeSWA，一种无需训练的长度外推方法。该方法结合了局部滑动窗口注意力、远程块摘要和动态内容检索，以保留局部细节并支持长程上下文访问。此外，通过位置重映射、条件注意力校正以及适应去噪阶段的窗口调度，缓解了由长度外推引起的生成质量退化。我们在多个连续扩散语言模型上进行了实验，结果表明RangeSWA能够有效降低长文本生成中的生成困惑度和文本重复率。

    arXiv:2610.42084v1 Announce Type: new  Abstract: Continuous diffusion language models show promising potential in text generation tasks, but are prone to problems such as text repetition, semantic drift, and generation quality degradation when the generation length exceeds the training length. To address these challenges, we propose RangeSWA, a training-free length extrapolation method. The method combines local sliding-window attention, remote block summaries, and dynamic content retrieval to preserve local details and support long-range context access. In addition, generation degradation caused by length extrapolation is mitigated through positional remapping, conditioning attention correction, and denoising-stage-adaptive window scheduling. We conducted experiments on multiple continuous diffusion language models, and the results show that RangeSWA can effectively reduce generative perplexity and text repetition rates in long-text generation.
    
[^6]: 知识层模型（KLM）：证据可见的AI输出、路由诊断与负面结果

    The Knowledge Layer Model (KLM): Evidence-Visible AI Output, Routing Diagnostics, and Negative Results

    [https://arxiv.org/abs/2610.42066](https://arxiv.org/abs/2610.42066)

    该论文提出知识层模型（KLM），一个开放的、与实现无关的AI推理记录规范，通过带版本管理的认知评估、证据关系和明确的未知项使AI输出可被审查，并如实报告了包含负面结果在内的初步路由诊断实验。

    

    AI的回答需要的不仅仅是表面上的正确性：审查者需要能够检查其记录的依据。我们提出了知识层模型，这是一个开放的、与具体实现无关的推理记录规范，包含带版本管理的认知评估、证据关系和明确的未知项。SEDIM——Nage AI OS的知识与证据架构——是其参考实现。我们引用的是KLM v0.3（仓库标签v0.3.0）。实证材料由相互独立的内部实验组成，而非端到端验证。一项历史性的平台领域路由基准测试报告了15个类别中137/150的分类结果（91.3%；名义95% Wilson置信区间为85.7%–94.9%）；其分子数值是从四舍五入的汇总中恢复的，且历史示例的成员归属未经核实。一个原生模型权重路径配置在经过修正的双源测试中产生了0/15的成功回答。一项容量增长对比无法建立可靠的效应，因为

    arXiv:2610.42066v1 Announce Type: cross  Abstract: AI answers require more than apparent correctness: reviewers need to inspect their recorded basis. We introduce the Knowledge Layer Model (KLM), an open, implementation-neutral specification for inference records with versioned epistemic assessments, evidence relations, and explicit unknowns. SEDIM, the knowledge and evidence architecture of Nage AI OS, is a reference implementation. We reference KLM v0.3 (repository tag v0.3.0). Empirical material consists of separate internal experiments, not end-to-end validation. A historical platform-domain routing benchmark reports 137/150 classifications across 15 classes (91.3%; nominal 95% Wilson interval 85.7-94.9%); the numerator is recovered from a rounded summary, and historical example membership is unverified. A native-model weight-path configuration yielded 0/15 successful answers in a corrected two-source test. A capacity-growth comparison could not establish a reliable effect because 
    
[^7]: PersonaChatBench：使用多样化模拟用户评估聊天助手

    PersonaChatBench: Evaluating Chat Assistants with Diverse Simulated Users

    [https://arxiv.org/abs/2610.42060](https://arxiv.org/abs/2610.42060)

    提出了PersonaChatBench评估框架，通过将相同任务与具有不同角色画像的多个模拟用户配对，对20个语言模型在六个领域进行评估，发现助手的任务性能和对用户需求的满足程度会因用户角色和领域的不同而显著变化。

    

    对话式助手需要为具有不同背景、偏好和沟通风格的人们提供帮助。然而，以单一合作用户为中心的评估方法，对于助手能否可靠地服务于不同角色用户提供的信息十分有限。我们提出了PersonaChatBench，一个用于评估任务能力以及对多样化用户需求响应能力的框架。PersonaChatBench将相同的底层任务与多个模拟用户配对，每个模拟用户由一张私密的角色卡引导，该角色卡规定了不同的需求、偏好和交互行为。通过在保持任务固定的情况下改变用户，该框架考察了助手性能在不同用户条件下的变化情况。我们在六个领域对20个语言模型进行了评估，分别评估了原生任务结果、明确表达的角色需求的满足程度以及模拟器的保真度。我们的结果表明，任务性能和需求满足程度在不同角色和不同领域之间存在差异。

    arXiv:2610.42060v1 Announce Type: new  Abstract: Conversational assistants are expected to help people with diverse backgrounds, preferences, and communication styles. Yet evaluations centered on a single cooperative user provide limited insight into how reliably an assistant serves users with different personas. We introduce PersonaChatBench, a framework for evaluating task competence and responsiveness to diverse user needs. PersonaChatBench pairs the same underlying task with multiple simulated users, each guided by a private persona card that specifies distinct needs, preferences, and interaction behaviors. By holding the task fixed while varying the user, the framework examines how assistant performance changes across user conditions. We evaluate 20 language models across six domains, separately assessing native task outcomes, fulfillment of expressed persona needs, and simulator fidelity. Our results show that task performance and need fulfillment vary across personas and domains
    
[^8]: CRoS：基于扰乱思维链的因果性奖励提升准确率与推理忠实度

    CRoS: Causality Reward on Scrambled CoTs Improves Accuracy and Reasoning Faithfulness

    [https://arxiv.org/abs/2610.42054](https://arxiv.org/abs/2610.42054)

    该论文提出 CRoS 方法，通过“打乱思维链”这种不改变各步骤信息的扰动方式来度量因果性并构建训练奖励，克服了现有基于删除或注入错误等扰动度量的混淆效应，从而同时提升语言模型推理的准确率与忠实度。

    

    可验证奖励的强化学习（RLVR）已成为训练语言模型进行推理的标准方法。然而，RLVR 通常仅依赖结果的正确性，这并不能保证思维链（CoT）对最终答案具有因果贡献。缺乏这种联系时，基于结果的奖励可能无法强化高质量的推理。现有的因果性度量方法通过扰动思维链来评估 CoT 的因果性，例如删除推理步骤或注入错误，并测量最终答案的变化。然而，这些扰动不可避免地会改变每一步中所包含的信息。因此，较高的因果性得分可能反映的是对扰动所引入信息变化的敏感性，而非真正的因果性。当这些度量被用作训练奖励时，这种混淆效应还会诱发不良的推理行为，包括生成信息量较少的思维链，以及减少探索与自我纠正。（原文摘要在此处截断）

    arXiv:2610.42054v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) has become the standard method for training language models to reason. However, RLVR typically relies solely on outcome correctness, which does not guarantee that the CoT causally contributes to the final answer. Without this connection, outcome rewards may fail to reinforce high-quality reasoning. Existing causality metrics assess CoT causality by perturbing the CoT, such as deleting reasoning steps or injecting errors, and measuring changes in the final answer. However, these perturbations inevitably alter information in each step. Consequently, high causality scores may reflect sensitivity to information changes introduced by perturbation rather than true causality. Such confounding effects further induce undesirable reasoning behaviors when these metrics are used as training rewards, including generating less informative CoTs and reducing exploration and self-correction. To addres
    
[^9]: 异步流映射语言模型

    Asynchronous Flow Map Language Models

    [https://arxiv.org/abs/2610.42018](https://arxiv.org/abs/2610.42018)

    提出异步流映射语言模型，通过逐token的时间调度将预测顺序内化到一步式建模中，并配合新的流映射蒸馏方法，实现单次前向传播即可生成全部token的并行语言生成。

    

    语言建模中预测token的顺序对生成序列的质量有着深远的影响。自回归模型和掩码扩散模型通过迭代填充token，能够在生成步骤之间利用这种顺序，但如何将顺序用于一步式并行语言建模尚不明确。我们提出了异步流映射语言模型，通过学习异步流的流映射，将预测顺序融入一步式建模中。这些流通过逐token的时间调度来指定顺序，该调度控制各token的相对去噪进度。我们构建了一类通用的异步流，它既支持预设的调度，也支持可学习的、状态自适应的调度，证明了它们能够定义有效的流映射，并为其流映射蒸馏开发了新方法。所得到的模型可以通过内化其中编码的顺序，在单次前向传播中生成所有token。

    arXiv:2610.42018v1 Announce Type: new  Abstract: The order in which tokens are predicted in language modeling has a profound impact on the quality of generated sequences. Autoregressive and masked diffusion models iteratively fill in tokens and can exploit such orderings across generation steps, but how to leverage an ordering for one-step parallel language modeling is unclear. We introduce asynchronous flow map language models (aFMLMs), which incorporate prediction order into one-step modeling by learning the flow maps of asynchronous flows. These flows specify an ordering through a tokenwise time schedule that controls the relative denoising progress of tokens. We formulate a general class of asynchronous flows that admit both prescribed and learned, state-adaptive schedules, show that they define valid flow maps, and develop new methods for their flow map distillation. The resulting models can generate all tokens in a single forward pass by internalizing the ordering encoded in the 
    
[^10]: TurnTurk-GPT：面向土耳其语的基于文本的轮次转换预测

    TurnTurk-GPT: Text-Based Turn-Taking Prediction for Turkish

    [https://arxiv.org/abs/2610.42010](https://arxiv.org/abs/2610.42010)

    本文提出了首个在真实土耳其语对话上系统评估的基于文本的轮次转换预测器TurnTurk-GPT，其在说话人不相交评估中优于广泛使用的多语言检测器，并揭示了土耳其语黏着语后缀对轮次预测至关重要。

    

    本文提出了TurnTurk-GPT，一个面向土耳其语的基于文本的轮次转换预测器。该模型通过在合成对话和真实对话上使用TurnGPT目标对土耳其语GPT-2模型进行微调获得，并在真实土耳其语对话上与规则基线、零样本土耳其语语言模型以及一种广泛使用的多语言轮次结束检测器进行对比评估。评估采用词级决策点，使用近似流式语音识别器输出的无标点、小写文本，采用录音不相交与说话人不相交的数据划分、嵌套阈值选择以及配对聚类自助检验。结果表明，TurnTurk-GPT在录音不相交和说话人不相交两种评估设置下均优于多语言检测器；仅当测试说话人出现在训练数据中时，它才优于零样本语言模型，而在未见过的说话人上两者在统计上难以区分。移除轮次末词的后缀会使AP最多下降44%，而（摘要在此处截断，原文显示对完整句子相关设置仅为2%左右）。

    arXiv:2610.42010v1 Announce Type: new  Abstract: This paper presents TurnTurk-GPT, a text-based turn-taking predictor for Turkish obtained by fine-tuning a Turkish GPT-2 model with the TurnGPT objective on synthetic and real dialogue, and evaluates it on real Turkish conversation against rule baselines, a zero-shot Turkish language model and a widely used multilingual end-of-turn detector. Evaluation uses word-level decision points on unpunctuated, lower-cased text approximating the output of streaming speech recognisers, recording-disjoint and speaker-disjoint folds, nested threshold selection and paired cluster bootstrap tests. TurnTurk-GPT outperforms the multilingual detector under both recording-disjoint and speaker-disjoint evaluation. It outperforms the zero-shot language model only when the test speakers occur in training; on unseen speakers the two are statistically indistinguishable. Removing the suffixes of the turn-final word reduces AP by up to 44%, against 2% for a senten
    
[^11]: 更少损失，固定权重：针对遗忘问题的Logit校正

    Less Loss, Fixed Weights: Logit Corrections for Forgetting

    [https://arxiv.org/abs/2610.42000](https://arxiv.org/abs/2610.42000)

    该研究发现语言模型微调后的“遗忘”损失中，仅靠固定的词汇偏置只能挽回不到10%，而结合温度校正可挽回多达36%，且在随机损坏的模拟中该方法仍能恢复29%，表明相当一部分遗忘损失可通过简单的固定输出校正来弥补。

    

    当语言模型经过微调以遵循指令，或在新的主题上进一步预训练后，它预测标准网络文本的能力往往会变差，这被称为“遗忘”。我们探究这种损失是否可以通过对模型最终token预测进行简单、固定的调整来挽回。在14对基础模型与派生模型上（参数量从1B到7B），我们为每个token拟合了固定的奖励或惩罚（词汇偏置），可单独使用，也可与温度参数结合使用。在从校正拟合中保留（held out）的文本上，仅使用top-4K偏置的最大恢复估计不足10%（损失以保留的词汇表为条件）。将偏置与温度结合可恢复多达36%。在数学上进一步预训练的模型损失最大，且几乎无法恢复。最后，在一个受控模拟中，模型被随机损坏（未植入任何偏置或温度变化），偏置加温度的修复方法仍能恢复29%。实际模型的大部分损失在

    arXiv:2610.42000v1 Announce Type: cross  Abstract: When a language model is fine-tuned to follow instructions, or further pretrained on a new subject, it often gets worse at predicting standard web text. This is known as "forgetting." We ask whether this loss can be undone by simple, fixed adjustments to the model's final token predictions. Across 14 pairs of base and derived models, from 1B to 7B parameters, we fitted a fixed bonus or penalty for each token (a vocabulary bias), alone or together with a temperature. On text held out from correction fitting, the largest top-4K bias recovery estimate is under 10%, with loss conditional on the retained vocabulary. Combining the bias with a temperature recovers up to 36%. Models further pretrained on math lose the most and recover almost nothing. Finally, in a controlled simulation where a model is damaged at random, with no planted bias or temperature change, the bias-plus-temperature fix still recovers 29%. Most of the real-model loss in
    
[^12]: DAMI：用于增强门控线性注意力长上下文检索的双轴记忆初始化

    DAMI: Dual-Axis Memory Initialization for Enhanced Long-Context Retrieval in Gated Linear Attention

    [https://arxiv.org/abs/2610.41994](https://arxiv.org/abs/2610.41994)

    提出双轴记忆初始化方法 DAMI，通过在训练前为门控线性注意力中的各注意力头分配几何时间尺度、并为各层设置低-高-低的输入敏感度分布，在不增加可训练参数的前提下显著提升了 GDN、KDA、Mamba-2、RWKV-7 等模型的长上下文检索准确率。

    

    线性注意力随序列长度呈线性扩展，但其有限状态记忆可能限制长上下文检索能力。在门控线性递归中，记忆保持取决于状态衰减，而状态衰减由门控参数 A_log 和 dt_bias 以及输入激活共同决定。在受控的 Gated DeltaNet（GDN）实验中，这两个参数在训练结束后仍与其初始值高度相关，凸显了初始化的重要性。因此，我们提出了双轴记忆初始化（DAMI），在训练前联合配置这些参数。DAMI 在各个头（head）之间分配几何时间尺度以覆盖不同的检索距离，并在各层之间采用低-高-低的输入敏感度分布。仅需一个小型激活样本即可校准 GDN 系列的实现，以匹配这两个行为目标，且不增加任何可训练参数。在 GDN、KDA、GDN2、Mamba-2 和 RWKV-7 等模型上，DAMI 将平均 RULER 风格检索准确率提升了……（原文摘要在此处截断）

    arXiv:2610.41994v1 Announce Type: cross  Abstract: Linear attention scales linearly with sequence length, but its finite-state memory can limit long-context retrieval. In gated linear recurrences, memory retention depends on state decay, which is shaped jointly by gate parameters A_log and dt_bias and by input activations. In controlled Gated DeltaNet (GDN) experiments, both parameters remain highly correlated with their initial values after training, highlighting the importance of initialization. Therefore, we introduce Dual-Axis Memory Initialization (DAMI), which jointly configures these parameters before training. DAMI assigns geometric timescales across heads for varied retrieval distances and a low-high-low input-sensitivity profile across layers. A small activation sample calibrates the GDN-family implementation to match both behavioral targets without adding trainable parameters. Across GDN, KDA, GDN2, Mamba-2, and RWKV-7, DAMI improves average RULER-style retrieval accuracy by
    
[^13]: 通过行为导出的语义维度将多模态大语言模型与人类认知的潜在结构对齐

    Aligning MLLMs with the Latent Structure of Human Cognition via Behavior-Derived Semantic Dimensions

    [https://arxiv.org/abs/2610.41951](https://arxiv.org/abs/2610.41951)

    该论文提出一种新框架，利用源自470万次人类“找出不同项”判断的66个可解释语义维度及其对应的语言理据，将多模态大语言模型的决策标准与人类认知的潜在结构对齐，从而提升模型与人类相似性判断的一致性。

    

    多模态大语言模型（MLLMs）在广泛的推理任务中取得了令人瞩目的表现，但其内部表示和决策标准是否与人类认知的潜在结构相一致仍不清楚。在本工作中，我们提出了一种新颖的框架，将MLLMs与人类相似性判断背后的多维心理表征对齐。我们基于先前建立的行为导出语义嵌入空间，其中物体沿着66个可解释的维度进行表示，这些维度源自470万次人类“找出不同项”判断。对于每个三元组，我们推断出最显著的决策维度，并将其转换为基于该维度的语言理据。这使得MLLMs能够学习最终选择，同时学习与推断出的行为标准相一致的解释。实验结果表明，我们的方法提高了模型与人类相似性判断的一致性。

    arXiv:2610.41951v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) have achieved impressive performance across a wide range of reasoning tasks, yet it remains unclear whether their internal representations and decision criteria align with the latent structure of human cognition. In this work, we propose a novel framework to align MLLMs with the multidimensional mental representations underlying human similarity judgements. We build on a previously established behavior-derived semantic embedding space, in which objects are represented along 66 interpretable dimensions derived from 4.70 million human Odd-One-Out (O1O) judgements. For each triplet, we infer the most salient decision dimension and convert it into a dimension-grounded linguistic rationale. This enables MLLMs to learn both the final choice and a justification consistent with the inferred behavioral criterion. Experimental results demonstrate that our approach improves model consistency with human simil
    
[^14]: LLM智能体中多工具请求的检索结构完整性与成本研究

    Completeness and Cost of Retrieval Structures for Multi-Tool Requests in LLM Agents

    [https://arxiv.org/abs/2610.41905](https://arxiv.org/abs/2610.41905)

    该论文提出completeness@k指标，系统比较了六种工具检索结构在多工具请求下的完整性与检索成本，发现现有在单工具查询上构建的检索器在需要四个及以上工具的查询上性能急剧下降（完整性从0.58–0.67跌至0.08–0.17）。

    

    当一个智能体需要调用多个工具才能完成某个请求时，只要工具检索器遗漏了其中任何一个工具，任务就会失败；然而，工具检索器大多是在仅需单个工具的查询上构建和评估的。我们在两个多工具测试平台上比较了六种检索结构的代表性方法（扁平词法与稠密检索、LLM查询分解、LLM导航的层次结构、先匹配服务器再匹配工具、依赖图以及经过训练的集合级检索器）：一个是带有专家依赖标注的345条AppWorld查询，另一个是覆盖257个MCP工具的103个MCP-Bench任务。我们测量completeness@k，即完整所需工具集全部出现在前k个检索结果中的查询占比，同时测量每种结构在检索时产生的LLM调用次数、token数量和延迟。针对全部7,961条ToolRet查询的动机性分析表明，总体完整性指标混合了两种不同情况：单目标查询在k=10时的完整性为0.58至0.67，而需要四个及以上工具的查询仅为0.08至0.17。

    arXiv:2610.41905v1 Announce Type: cross  Abstract: An agent that must call several tools to fulfill a request fails when the tool retriever omits any one of them, yet tool retrievers are mostly built and evaluated on queries requiring a single tool. We compare representatives of six retrieval structures (flat lexical and dense retrieval, LLM query decomposition, LLM-navigated hierarchies, server-then-tool matching, dependency graphs, and trained set-level retrievers) across two multi-tool testbeds: 345 AppWorld queries with expert dependency labels and 103 MCP-Bench tasks over 257 MCP tools. We measure completeness@k, the share of queries whose entire required tool set appears in the top k, alongside the LLM calls, tokens, and latency that each structure incurs at retrieval time. A motivating analysis on all 7,961 ToolRet queries shows that aggregate completeness mixes two regimes: completeness at 10 is .58 to .67 on single-target queries and .08 to .17 on queries with four or more tar
    
[^15]: 工具调用能暴露多少关于你的信息？来自工具调用轨迹的隐私泄露

    How Much Can Tool Calls Reveal About You? Privacy Leakage from Tool-Call Traces

    [https://arxiv.org/abs/2610.41885](https://arxiv.org/abs/2610.41885)

    该论文首次系统研究了从MCP工具调用轨迹推断用户隐私画像的风险，构建了包含1,000个用户画像和4,000条智能体轨迹的PrivacyTrace基准，发现LLM攻击者能以26.25%的成功率从工具调用轨迹推断用户属性，其中工具参数、返回结果及跨会话轨迹关联是主要的隐私泄露途径。

    

    基于大语言模型（LLM）的智能体通过模型上下文协议（MCP）等协议与外部工具交互，产生可能暴露用户行为信号的工具调用轨迹。在本工作中，我们系统研究了从MCP工具调用轨迹推断用户画像的风险，即观察者在无法访问用户对话的情况下，从符合协议的工具交互中推断用户属性。我们提出了PrivacyTrace，一个包含1,000个合成用户画像和4,000条覆盖四个领域的智能体轨迹的基准数据集，这些轨迹使用来自25个服务器的真实MCP工具生成。使用四种主流LLM攻击者的实验表明，工具调用轨迹可实现26.25%的用户画像推断攻击成功率，而基于先验知识的猜测成功率仅为17.97%。进一步分析揭示，工具参数和返回结果提供了主要的基于轨迹的证据，而跨会话关联轨迹会进一步放大用户画像的效果。

    arXiv:2610.41885v1 Announce Type: cross  Abstract: Large language model (LLM)-based agents interact with external tools through protocols such as the Model Context Protocol (MCP), producing tool-call traces that may expose behavioral signals about users. In this work, we systematically study the risk of user-profile inference from MCP tool-call traces, where observers infer user attributes from protocol-valid tool interactions without access to user conversations. We introduce PrivacyTrace, a benchmark containing 1,000 synthetic user profiles and 4,000 agent trajectories across four domains, generated with real MCP tools from 25 servers. Experiments with four mainstream LLM attackers show that tool-call traces enable a 26.25% attack success rate for profile inference, compared with 17.97% from prior-based guessing. Further analysis reveals that tool arguments and returned results provide the main trace-grounded evidence, while linking traces across sessions further amplifies profiling 
    
[^16]: 文本作为索引的零样本跨域知识追踪

    Text as an Index for Zero-Shot Cross-Domain Knowledge Tracing

    [https://arxiv.org/abs/2610.41883](https://arxiv.org/abs/2610.41883)

    该论文提出TIZSKT框架，将文本作为索引而非直接可迁移的预测表示，通过分离行为序列建模与域内局部历史证据构建，实现了无需目标平台作答日志的零样本跨域知识追踪。

    

    知识追踪（KT）根据学生的历史作答记录预测其下一次作答表现。新平台通常只有题目文本而没有可供训练的作答日志，这推动了零样本跨域知识追踪的研究。现有的迁移和冷启动方法利用题目语义，但跨平台之间的语义差异限制了可迁移性。我们提出TIZSKT，该方法将行为序列建模与域内局部历史证据构建分离开来，并将文本用作索引而非直接可迁移的预测表示。掌握度动态编码器在不依赖题目标识符或文本嵌入的情况下学习正误动态。域内局部题目寻址利用冻结的文本编码器和无监督的域内投影构建固定的题目地址。文本寻址核利用这些地址将当前学生以往的作答结果聚合为针对特定题目的历史正误摘要。跨段续接……

    arXiv:2610.41883v1 Announce Type: new  Abstract: Knowledge tracing (KT) predicts a student's next response from their history. New platforms often have question texts but no response logs for training, motivating zero-shot cross-domain KT. Existing transfer and cold-start methods use question semantics, but cross-platform semantic discrepancies limit transferability. We propose TIZSKT, which separates behavioral sequence modeling from domain-local historical evidence construction and uses text as an index rather than a directly transferable predictive representation. The Mastery-Dynamics Encoder learns correctness dynamics without question identifiers or text embeddings. Domain-Local Question Addressing builds fixed question addresses using a frozen text encoder and an unsupervised per-domain projection. The Text-Addressing Kernel uses these addresses to aggregate the current student's previous outcomes into a question-specific historical correctness summary. Cross-Segment Continuation
    
[^17]: SimXAI：一个简单的即插即用层提升对话式可解释AI的交互质量与理解

    SimXAI: A Simple Drop-in Layer Increases Interaction Quality and Understanding in Conversational XAI

    [https://arxiv.org/abs/2610.41856](https://arxiv.org/abs/2610.41856)

    SimXAI用一个简单的即插即用LLM对话层替代传统ConvXAI中分离的意图识别、对话管理和模板生成模块，在保持高忠实性的同时更好地处理依赖上下文的后续问题。

    

    对话式可解释AI（ConvXAI）智能体通常采用GUS风格的架构，其中包含用于意图识别、对话管理和基于模板的响应生成的独立模块。虽然这些流水线忠实于底层的XAI方法，但它们难以适配，并且在处理依赖上下文的后续问题时常常表现不佳。我们提出了SimXAI，一个轻量级的以大语言模型（LLM）为中心的对话层，它用一个单一的工具调用型LLM取代了这些模块，从而桥接对现有XAI资源的访问。SimXAI并非一个独立的智能体，而是作为一个模块化接口，实现对外部XAI资源的对话式访问。我们进一步引入了一个评估套件，涵盖单轮和多轮解析、忠实性以及上下文关联能力。在现有和新的数据集上，SimXAI实现了强大的函数选择准确性和高忠实性，同时能更好地处理依赖上下文的请求。在一项仅替换对话层的用户研究中（原文在此处截断）...

    arXiv:2610.41856v1 Announce Type: new  Abstract: Conversational explainable AI (ConvXAI) agents often use GUS-style architectures with separate modules for intent recognition, dialogue management, and template-based response generation. While faithful to the underlying XAI methods, these pipelines are hard to adapt and often struggle with context-dependent follow-up questions. We present SimXAI, a lightweight LLM-centric conversational layer that replaces these modules with a single tool-calling LLM that bridges access to existing XAI resources. Rather than a standalone agent, SimXAI serves as a modular interface that enables conversational access to external XAI resources. We further introduce an evaluation suite for single- and multi-turn parsing, faithfulness, and contextualization. Across existing and new datasets, SimXAI achieves strong function-selection accuracy and high faithfulness while better handling context-dependent requests. In a user study replacing only the conversatio
    
[^18]: 迭代积分作为词汇指纹：用于词汇相似性的路径签名嵌入

    Iterated Integrals as Lexical Fingerprints: Path Signature Embeddings for Lexical Similarity

    [https://arxiv.org/abs/2610.41855](https://arxiv.org/abs/2610.41855)

    该论文提出利用粗糙路径理论中的迭代积分，通过Word2Vec轨迹的路径签名来构建句子嵌入，以几何可处理的方式显式捕捉词汇与序列结构，并揭示句子嵌入隐式编码表层词汇相似性的规律。

    

    虽然句子嵌入主要为语义泛化而设计，但其潜在几何结构中隐式地编码了表层词汇特性，这一关系至今仍鲜为人知。浅层池化方法会丢弃词序信息，而深度Transformer模型则将这种词汇对应关系隐藏在计算复杂性和不可解释的空间之中。我们提出通过Word2Vec轨迹的路径签名来构建句子嵌入，利用粗糙路径理论中的迭代积分，以几何上易于处理的方式显式捕捉序列与词汇结构。通过对Word2Vec维度、签名深度和PCA压缩进行大量消融实验，我们将嵌入距离与五种词汇字符串度量进行了相关性分析，分别是：Levenshtein编辑距离、Jaro-Winkler距离、词Jaccard相似度、音素Levenshtein距离和字符TF-IDF。我们发现，虽然平均池化几乎完全无法捕捉……（原文摘要在此处截断）

    arXiv:2610.41855v1 Announce Type: new  Abstract: While sentence embeddings are primarily designed for semantic generalization, their latent geometry implicitly encodes surface-level lexical properties, a relationship that remains poorly understood. Shallow pooling methods discard word order, while deep Transformer models obscure these lexical alignments behind computational complexity and uninterpretable spaces. We propose constructing sentence embeddings via the path signature of Word2Vec trajectories, leveraging the iterated integrals of rough path theory to create representations that explicitly capture sequential and lexical structure in a geometrically tractable manner. Through extensive ablations spanning Word2Vec dimensions, signature depths, and PCA compression, we correlate embedding distances against five lexical string metrics: Levenshtein, Jaro-Winkler, word Jaccard, phoneme Levenshtein, and character TF-IDF. We reveal that while average pooling almost entirely fails to cap
    
[^19]: 剥开俄罗斯套娃：从检索嵌入中提取本体结构

    Peeling the Matryoshka: Extracting Ontological Structure from Retrieval Embeddings

    [https://arxiv.org/abs/2610.41785](https://arxiv.org/abs/2610.41785)

    本文发现MRL嵌入的前缀长度天然对应分类体系由粗到细的粒度，并据此提出仅需训练一次的极简探针PEEL，可直接从冻结的检索嵌入中提取本体层次结构，避免昂贵的调参与任务特定训练。

    

    分类体系构建对知识密集型自然语言处理至关重要，但成本依然高昂：基于聚类的方法（i）需要针对每个语料库进行细致调参，且产生的层次结构不稳定；而基于大语言模型（LLM）的流水线（ii）则依赖昂贵且无法跨领域迁移的任务特定训练。我们报告了一个与上述两个问题都相关的实证发现：在套娃表示学习（MRL）嵌入中，每个前缀向量本身就是一个可用的、更粗糙的向量，而前缀长度与金标准分类体系从粗到细的粒度相对应；通过维度打乱与维度倒序的对照实验，我们将这种对应关系定位到模型学习得到的前部前缀维度上。该模式在三个架构各异的MRL模型中均成立。为验证这一规律是否可用，我们提出了PEEL（剥离嵌入以提取潜在本体）：一个构建于这些冻结嵌入之上的极简探针，仅需使用序嵌入目标训练一次，该目标将层次结构表示为向量包含关系……

    arXiv:2610.41785v1 Announce Type: new  Abstract: Taxonomy construction is essential for knowledge-intensive NLP yet remains costly: clustering-based methods (i) need careful per-corpus tuning and yield unstable hierarchies, while LLM-based pipelines (ii) depend on expensive task-specific training that does not transfer across domains. We report an empirical finding that bears on both: in Matryoshka Representation Learning (MRL) embeddings, where each leading prefix is itself a usable, coarser vector, prefix length aligns with the coarse-to-fine granularity of gold taxonomies; shuffled- and reverse-dimension controls isolate the alignment to the learned early prefix dimensions. The pattern holds across three architecturally distinct MRL models. To test whether this regularity is usable, we introduce PEEL (Peeling Embeddings to Extract Latent ontologies): a minimal probe over these frozen embeddings, trained once with an order-embedding objective that represents hierarchy as vector conta
    
[^20]: 逐步查询：无需访问教师思维链的推理蒸馏

    Stepwise-Querying: Distilling Reasoning without Access to Teacher Chain-of-Thought

    [https://arxiv.org/abs/2610.41769](https://arxiv.org/abs/2610.41769)

    提出逐步查询方法，通过对教师模型进行逐步独立查询以获取中间推导、有效性检查和修正，无需访问教师思维链即可构建推理监督数据，实现高效的推理蒸馏，并在16项基准比较中的14项取得最佳成绩。

    

    推理蒸馏被广泛用于通过教师生成的监督信号，将大型语言模型的推理能力迁移到更小的学生模型中。然而在黑盒设置下，教师模型可能只暴露最终答案和简短解答，中间计算过程和有效性条件不可获取，从而限制了可用于蒸馏的推理监督。我们提出逐步查询方法，无需访问教师模型的内部思维链（CoT）痕迹，即可引出针对具体步骤的解释。通过对同一教师模型进行独立查询，请求单个步骤的中间推导、有效性检查和修正，随后将组装好的解答用于标准的监督微调。在两个教师模型、两个学生模型以及四个数学和代码基准测试上，我们的方法在与三种基线方法的16次比较中有14次取得最高分。我们进一步引入了逐步解释（摘要内容截断）

    arXiv:2610.41769v1 Announce Type: new  Abstract: Reasoning distillation is widely used to transfer the reasoning capabilities of large language models to smaller students through teacher-generated supervision. In the black-box setting, however, teachers may expose only final answers and brief solutions, leaving intermediate calculations and validity conditions unavailable and thus limiting the reasoning supervision available for distillation. We propose Stepwise Querying, which elicits step-specific explanations without requiring access to the teacher's internal chain-of-thought (CoT) traces. Independent queries to the same teacher request intermediate derivations, validity checks, and corrections for individual steps, the assembled solutions are then used for standard supervised fine-tuning. Across two teachers, two students, and four mathematics and code benchmarks, our method achieves the highest score in 14 of 16 comparisons with three baselines. We further introduce Stepwise Expla
    
[^21]: 任务向量可以被线性分解为可解释且可控的成分

    Task Vectors Can Be Linearly Decomposed into Interpretable and Controllable Components

    [https://arxiv.org/abs/2610.41743](https://arxiv.org/abs/2610.41743)

    该论文提出将任务向量线性分解为任务识别（TR）、任务学习（TL）和残差三个可解释成分，并通过贝叶斯优化TR和TL系数，找到了比原始任务向量更有效的模型干预方向。

    

    任务向量提供了一种紧凑的方式来引导大型语言模型，但它们通常被视为仅从隐藏状态中提取的原子化方向，其内部功能结构尚不清楚。受先前将上下文学习分解为任务识别（TR）和任务学习（TL）的研究启发，我们探究单个任务向量是否同样可以分解为具有功能意义的成分。我们从候选标签的unembedding出发，提出了一种基于TR/TL的分解方法，将每个任务向量分离为TR、TL和残差三个成分，并尝试使用贝叶斯优化来优化TR和TL系数，以实现更有效的干预。我们在多个模型家族、多个分类任务以及基于指令和基于示例的任务向量上对这一方法进行了评估。在所有这些设置中，基于TR/TL的优化所确定的干预方向均优于原始任务向量。尽管基于优化……

    arXiv:2610.41743v1 Announce Type: cross  Abstract: Task Vectors (TVs) provide a compact way to steer large language models, but they are typically treated as atomized directions merely extracted from hidden states, leaving their internal functional structure poorly understood. Motivated by prior work that decomposes in-context learning into Task Recognition (TR) and Task Learning (TL), we ask whether a single TV can likewise be decomposed into functionally meaningful components. We formulate a TR/TL-based decomposition from candidate-label unembeddings, separating each TV into TR, TL, and residual components, and try to optimize the TR and TL coefficients using Bayesian optimization for more effective intervention. We evaluate this formulation across multiple model families, classification tasks, and both instruction- and demonstration-based TVs. Across these settings, TR/TL-based optimization identifies intervention directions that outperform the original TVs. Although optimization ov
    
[^22]: 面向孟加拉语症状疾病预测的泄漏受控结构保持增强与模板感知评估

    Leakage-Controlled Structure-Preserving Augmentation and Template-Aware Evaluation for Bangla Symptom-Based Disease Prediction

    [https://arxiv.org/abs/2610.41733](https://arxiv.org/abs/2610.41733)

    本文提出仅在训练集上进行的泄漏受控、保持症状结构的非对称伯努利数据增强方法与模板感知评估框架，显著提升孟加拉语症状疾病预测中提升树模型（XGBoost、LightGBM、CatBoost）的准确率，并揭示标准基准中普遍存在的模板泄漏问题。

    

    基于孟加拉语症状的疾病预测通常在小型、不平衡的结构化基准上进行评估，该基准中包含许多仅相差一两个症状的档案。我们使用涵盖85种疾病、172个二元症状的725个独特档案，测试仅在训练集上进行的非对称伯努利扰动能否改善分类效果。实验采用十次重复的真实档案留出法以及按父组分组的内部交叉验证。2×2设计将数据增强与超参数调优分离，并以精确复制、随机过采样、类别加权和SMOTEN作为平衡对照方法。对于伯努利朴素贝叶斯（BNB），尽管其取得了最高的平均准确率（97.50%），但其相对于经过调优的原始训练的增益未能通过Holm校正。提升树模型则显著受益：XGBoost、LightGBM和CatBoost分别提升18.10、28.45和10.09个百分点。汉明距离审计显示，标准留出集中有91.0%的档案与训练档案相差不超过两个症状……（摘要原文在此处截断）

    arXiv:2610.41733v1 Announce Type: cross  Abstract: Bangla symptom-based disease prediction is commonly evaluated on a small, imbalanced structured benchmark containing many profiles that differ by only one or two symptoms. We test whether train-only asymmetric Bernoulli perturbation improves classification using 725 unique profiles spanning 85 diseases and 172 binary symptoms. Experiments use ten repeated real-profile holdouts and parent-grouped inner cross-validation. The 2 x 2 design separates augmentation from hyperparameter tuning, with exact duplication, random oversampling, class weighting, and SMOTEN as balancing controls. For Bernoulli Naive Bayes (BNB), the gain over tuned raw training does not withstand Holm correction, although BNB attains the highest mean accuracy (97.50%). The boosted-tree models benefit substantially: XGBoost, LightGBM, and CatBoost improve by 18.10, 28.45, and 10.09 percentage points, respectively. A Hamming-distance audit places 91.0% of standard held-o
    
[^23]: 从训练暴露到成员证据：OLMo 2 的受控研究

    From Training Exposure to Membership Evidence: A Controlled Study of OLMo 2

    [https://arxiv.org/abs/2610.41732](https://arxiv.org/abs/2610.41732)

    本研究通过受控地重新运行 OLMo 2 1B 的 500 亿 token 中期训练，并利用同种子发布的运行作为反事实参照，首次严格量化了单次训练暴露所提供的成员证据——单次暴露平均会提高文档似然，但很少能可靠地识别出训练成员。

    

    语言模型是否在某个文档上进行过训练，这对版权、隐私和基准污染等问题至关重要。然而，大多数针对语言模型的成员推断（MI）评估都是在事后收集成员和非成员数据，因此成功的推断可能仅仅反映了分布偏移，而失败也无法得出明确结论。我们重新运行了 OLMo 2 1B 的官方 500 亿 token 中期训练阶段，用来自 7,814 对在领域和难度上相匹配的文档替换了选定的训练数据块。对于每一对文档，通过抛硬币决定哪一个被插入训练、哪一个被保留。由 Allen Institute for AI（AI2）发布的相同种子的运行结果在其他所有数据上都完全一致，可作为反事实参照。单次暴露平均会提高一个先前未见文档的似然，但很少能准确识别出该文档。以反事实运行作为参照，参考比率的 ROC 曲线下面积（AUC）为 0.767，但在跨领域使用单一阈值时，捕获（摘要在此处截断）

    arXiv:2610.41732v1 Announce Type: cross  Abstract: Whether a language model was trained on a document matters for copyright, privacy, and benchmark contamination. Yet most membership-inference (MI) evaluations on language models collect members and non-members after the fact, so success may reflect distribution shift and failure is inconclusive. We re-ran OLMo 2 1B's official 50-billion-token mid-training stage, replacing selected training chunks with documents from 7,814 pairs matched on domain and difficulty. For each pair, a coin flip determined which document was inserted and which was held out. The run released by the Allen Institute for AI (AI2) from the same seed saw identical data everywhere else and serves as the counterfactual. One exposure raises a previously unseen document's likelihood on average but rarely identifies it. Using the counterfactual as reference, the reference ratio has an area under the ROC curve (AUC) of 0.767 but, with one threshold across domains, catches
    
[^24]: UpLoop：通过升级复用以零额外预训练计算成本训练循环Transformer

    UpLoop: Training Looped Transformers via Upcycling at No Extra Pretraining Compute

    [https://arxiv.org/abs/2610.41720](https://arxiv.org/abs/2610.41720)

    UpLoop提出一种升级复用式预训练调度方法，从浅层Transformer逐步复制层来加深模型，并将早期节省的计算用于后期共享中间块的重复执行，在与稠密基线完全相同的参数量、训练token数和FLOPs下训练循环Transformer，显著提升语言建模与下游任务性能。

    

    循环语言模型能够通过共享权重的重复计算来增强推理能力，在不增加独立参数的情况下提高执行深度。然而，每增加一次循环都会增加训练的浮点运算量（FLOPs），使得这些收益的获取成本更高。为解决这一成本问题，我们提出了UpLoop，这是一种预训练调度方案，利用升级复用技术来训练循环Transformer，同时在最终参数量、总训练token数和训练FLOPs上与标准稠密基线完全一致。具体而言，UpLoop从一个浅层Transformer开始，通过复制已学习的层逐步增加深度，并利用早期节省的计算量来支持训练后期共享中间块的重复执行。跨多个模型规模的实验表明，该方法在语言建模和下游性能上均有提升，在1.3B参数规模下，22个下游任务的平均得分从22.99提升至27.52。

    arXiv:2610.41720v1 Announce Type: cross  Abstract: Looped language models can strengthen reasoning through repeated computation with shared weights, increasing execution depth without adding independent parameters. However, each additional pass increases training floating-point operations (FLOPs), making these gains more expensive to obtain. To address this cost, we propose UpLoop, a pretraining schedule that uses upcycling to train looped Transformers while exactly matching a standard dense baseline in final parameter count, total training tokens, and training FLOPs. Specifically, UpLoop begins with a shallow Transformer, progressively grows its depth by copying learned layers, and uses the computation saved early to support repeated execution of a shared middle block later in training. Experiments across model scales show improvements in language modeling and downstream performance, with the average score across 22 downstream tasks increasing from 22.99 to 27.52 at 1.3B parameters. T
    
[^25]: GraphNIAH：一个面向真实世界图的长上下文推理基准测试

    GraphNIAH: A Benchmark for Long-Context Reasoning over Real-World Graphs

    [https://arxiv.org/abs/2610.41704](https://arxiv.org/abs/2610.41704)

    GraphNIAH提出了一个基于真实科学论文引用关系构建的动态基准测试，用于评估长上下文图推理能力，克服了现有大海捞针基准依赖合成数据、数据污染风险和人工标注等局限性。

    

    大海捞针测试被引入作为衡量模型在超大输入文本中查找小型相关事实能力的测试。然而，现有的NIAH基准测试各自存在以下一个或多个局限性：(1) 依赖合成数据，因此其评估结果可能无法推广到真实世界场景；(2) 复用之前已发布的数据集，导致基准测试污染的风险；(3) 依赖人工标注，这需要耗费人力来创建和更新数据，并且容易在未来发生基准测试污染；以及(4) 对长上下文多跳推理进行了不切实际的简化——即每一步在给定前面步骤的情况下都很简单，而真实世界的推理往往需要在多条可能的证明路径上进行规划。为了解决这些局限性，我们引入了\Benchmark，这是一个基于已发表科学论文之间真实引用关系构建的动态基准测试（摘要在此处被截断）。

    arXiv:2610.41704v1 Announce Type: new  Abstract: The needle-in-a-haystack (NIAH) benchmark was introduced as a test for measuring a model's ability to find small relevant facts within a very large body of input text. However, existing NIAH benchmarks each suffer from one or more of the following limitations: (1) reliance on synthetic data, so their evaluations may not generalize to real-world settings; (2) reuse of previously published datasets, leading to the risk of benchmark contamination; (3) dependence on manual annotation, which requires human effort to create and update data and is prone to future benchmark contamination; and (4) unrealistic simplification of long-context multi-hop reasoning where each next step is trivial given previous steps, while real-world reasoning often requires planning over multiple possible proof paths. To address these limitations, we introduce \Benchmark, a dynamic benchmark built from real citation relationships among scientific papers published aft
    
[^26]: 大语言模型在语法纠错任务中推理层面的精确率—召回率权衡

    Precision--Recall Trade-off on the Inference Level for LLMs in Grammatical Error Correction

    [https://arxiv.org/abs/2610.41698](https://arxiv.org/abs/2610.41698)

    本研究发现，仅通过调整大语言模型推理时的重复惩罚参数，即可在不修改模型架构、训练过程或提示的情况下，灵活控制语法纠错任务中精确率与召回率的权衡。

    

    大语言模型（LLM）近期被应用于语法错误纠正（GEC）任务，通过全量微调和基于提示的方法均取得了出色的性能。作为概率序列模型，大语言模型提供了可直接修改token生成行为的推理时参数；然而，这些参数在控制语法纠错特定权衡方面的潜力尚未得到系统性研究。在本工作中，我们研究了语法错误纠正的解码策略。我们发现，重复惩罚参数能够在推理时实现对精确率—召回率权衡的可控调整，而无需修改模型架构、训练过程或提示。通过在标准语法纠错基准上的实验，我们证明了基于重复惩罚的解码可以实现对精确率—召回率平衡的推理时控制，在概念上与先前针对序列到序列语法纠错模型提出的方法相关……

    arXiv:2610.41698v1 Announce Type: new  Abstract: Large language models (LLMs) have recently been adopted for Grammatical Error Correction (GEC), achieving strong performance via both full fine-tuning and prompt-based approaches. As probabilistic sequence models, LLMs expose inference-time parameters that directly modify token generation behavior; however, their potential for controlling GEC-specific trade-offs has not been systematically studied. In this work, we investigate decoding strategies for grammatical error correction. We show that the repetition penalty parameter enables controllable adjustment of the precision--recall trade-off at inference time, without modifying the model architecture, training procedure, or prompt. Through experiments on standard GEC benchmarks, we demonstrate that repetition-penalty-based decoding enables inference-time control of the precision--recall balance, conceptually related to methods previously proposed for sequence-to-sequence GEC models, such 
    
[^27]: TSBPO：面向交互式语言智能体策略优化的树结构分支方法

    TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents

    [https://arxiv.org/abs/2610.41697](https://arxiv.org/abs/2610.41697)

    TSBPO提出一种树结构的采样到更新范式，通过似然拟合的根/分支分配器定向采样、基于兄弟片段与匹配观察状态的延迟反馈分配、以及跨活跃轮次归一化的词元损失聚合，在单一训练循环中解决了从终端结果训练交互式语言智能体的三大耦合难题。

    

    从终端结果训练交互式语言智能体面临三个相互耦合的问题：应采样哪些任务与状态延续、如何将延迟反馈分配到各个决策上，以及如何为多词元动作聚合词元损失。我们提出TSBPO，一种树结构的“采样到更新”范式，在单一训练循环内解决上述问题。一个基于似然拟合的根/分支分配器利用保留的任务与状态证据，在固定rollout预算下将延续采样定向到可能产生对比性终端结果的分支上。在由此构建的重放物理树中，轨迹优势通过比较直接相邻的兄弟片段来计算，而观察组步骤优势则在匹配的任务/观察状态下比较不同结果。随后，执行器对每个完整响应求和裁剪的词元代理项，并在活跃轮次间进行归一化，从而避免了策略梯度项中的响应长度平均。使用Qwen2.5-1.5B作为执行器，固定检查点……（摘要原文在此处截断）

    arXiv:2610.41697v1 Announce Type: cross  Abstract: Training interactive language agents from terminal outcomes poses three coupled problems: which task and state continuations to sample, how to assign delayed feedback to decisions, and how to aggregate token losses for multi-token actions. We present TSBPO, a tree-structured sampling-to-update paradigm that addresses these problems within one training loop. A likelihood-fitted Root/Branch allocator uses retained task and state evidence to target continuations likely to produce contrasting terminal outcomes under fixed rollout budgets. On the resulting replayed physical trees, trajectory advantages compare immediate sibling segments, while observation-group step advantages compare outcomes at matched task/observation states. The actor then sums clipped token surrogates over each complete response and normalizes across active turns, avoiding response-length averaging in the policy-gradient term. With a Qwen2.5-1.5B actor, fixed checkpoin
    
[^28]: 推理会让语言模型达成一致吗？

    Does Reasoning Make Language Models Agree?

    [https://arxiv.org/abs/2610.41691](https://arxiv.org/abs/2610.41691)

    该论文区分了作为干预手段的推理与作为观察行为的推理，并发现在分类任务中启用推理能够增加不同语言模型之间的一致性。

    

    语言模型面临两种相互对立的担忧：一是“单一文化”（monoculture），即模型以相关联的方式犯错；二是“多样性”（multiplicity），即同样准确的模型却做出相互冲突的决策。推理模型的兴起使这一争论变得更加复杂：一方面，如果模型锚定于相同的显著特征或遵循相似的推理路径，推理可能会增加模型间的一致性；另一方面，随机的推理轨迹可能引入额外的分支点，从而导致更多样化的预测。我们通过区分“作为干预手段的推理”与“作为观察行为的推理”来化解这一矛盾。我们比较了在推理努力受控变化条件下的一致性与在不同观察推理长度下的一致性，并将两者与基于模型准确率所预期的一致性进行对照。在来自两个开发商的九个模型和三个分类任务上，启用推理增加了在相同个体上评估的模型之间的一致性。

    arXiv:2610.41691v1 Announce Type: cross  Abstract: Language models face two opposing concerns: monoculture, in which models err in correlated ways, and multiplicity, in which equally accurate models make conflicting decisions. The rise of reasoning models has complicated this debate: On the one hand, reasoning may increase model agreement if models anchor on the same salient features or follow similar reasoning paths. On the other hand, stochastic reasoning traces may introduce additional branch points leading to more diverse predictions. We resolve this tension by distinguishing reasoning as an intervention from reasoning as an observed behavior. We compare agreement under controlled changes in reasoning effort with agreement across observed reasoning lengths, measuring both against the agreement expected from model accuracy. Across nine models from two developers and three classification tasks, enabling reasoning increases agreement between models evaluated on the same individuals. Y
    
[^29]: 当引导改变话语：识别与干预大语言模型中激活引导所导致的话语层面变化

    When Steering Shifts Discourse: Identifying and Intervening on Discourse-Level Changes due to Activation Steering in LLMs

    [https://arxiv.org/abs/2610.41661](https://arxiv.org/abs/2610.41661)

    论文揭示了大语言模型的激活引导会系统性改变生成文本的话语层面特征（如正式程度、证据使用和模糊性），并提出通过添加针对受影响维度的互补向量进行干预，可恢复约78%的非预期话语变化，同时保留预期的引导效果。

    

    激活引导是一种推理时技术，用于控制或修改大语言模型（LLM）的行为、语气或安全属性，无需重新训练模型权重或编辑输入提示，只需向模型的残差流激活中添加一个向量即可。然而，激活引导也会以非预期的方式改变生成的文本，这些变化与预期的行为改变无关，且尚未被充分理解。先前的工作仅通过困惑度等粗粒度信号来衡量这些变化。在本工作中，我们表明激活引导会系统性地在生成文本的话语维度上产生显著变化，例如文本的正式程度、证据使用和模糊性。我们提出了一种应对方法，即添加第二个针对受影响话语维度的互补向量。这种干预能够恢复约78%的非预期话语变化，同时保留预期的引导行为。

    arXiv:2610.41661v1 Announce Type: new  Abstract: Activation steering is an inference-time technique used to control or modify the behavior, tone, or safety properties of a large language model (LLM) without retraining the model's weights or editing its input prompt, but simply by adding a vector to its residual-stream activations. However, activation steering also changes the generated text in unintended ways, unrelated to intended behavior change, that are not well understood. Prior work has measured these changes only through coarse signals such as perplexity. In this work, we show that steering systematically produces significant changes in discourse dimensions of the generated text, such as its formality, use of evidence, and ambiguity. We present a way of countering this by adding a second, complementary vector that targets the affected discourse dimension. This intervention restores about 78% of the unintended discourse changes while preserving the intended steering behavior in a
    
[^30]: GRIT：通过梯度掩码处理富文本推理中视觉锚定错误的词元

    GRIT: Gradient Masking for Visually Misgrounded Tokens in Text-Rich Reasoning

    [https://arxiv.org/abs/2610.41629](https://arxiv.org/abs/2610.41629)

    提出GRIT训练框架，通过梯度掩码将视觉锚定引入可验证奖励的强化学习（RLVR），以解决视觉语言模型在富文本推理中“答案正确但视觉误锚定”（CVMR）的问题。

    

    最近，可验证奖励的强化学习（RLVR）已被扩展到视觉语言模型（VLM），但其在富文本推理任务中的有效性仍然有限。我们强调了导致这一局限的一个关键因素：模型经常在中间步骤中误读视觉信息，却仍然得出正确的最终答案。我们将这一现象定义为“正确但视觉误锚定的推理”。由于标准RLVR只奖励最终答案的正确性，它无法区分锚定正确与锚定错误的推理，对两者一视同仁地加以强化。通过词元级别的分析，我们揭示出CVMR词元表现出与锚定良好的词元存在系统性差异的注意力模式。基于这一观察，我们提出了GRIT，一个将视觉锚定融入RLVR的训练框架，以减轻CVMR的影响。此外，为解决富文本推理中……

    arXiv:2610.41629v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has recently been extended to vision-language models (VLMs), yet its effectiveness remains limited for text-rich reasoning tasks. We highlight a key factor contributing to this limitation: models frequently arrive at correct final answers despite misperceiving visual information during intermediate steps. We define this phenomenon as Correct-but-Visually-Misgrounded Reasoning (CVMR). Because standard RLVR rewards only final-answer correctness, it fails to differentiate between properly grounded and misgrounded reasoning, reinforcing both equally. Through token-level analysis, we reveal that CVMR tokens exhibit distinct attention patterns that differ systematically from well-grounded ones. Based on this observation, we propose GRIT, a training framework that incorporates visual grounding into RLVR to mitigate the impact of CVMR. Furthermore, to address the lack of text-rich reasoning
    
[^31]: Saidzoda-Tajik-LFM-2.6B：将2.6B混合模型适配塔吉克语工具调用，以及持续预训练所付出的代价

    Saidzoda-Tajik-LFM-2.6B: Adapting a 2.6B Hybrid Model to Tajik Tool Calling, and What Continued Pretraining Takes Away

    [https://arxiv.org/abs/2610.41576](https://arxiv.org/abs/2610.41576)

    该论文提出两阶段方案（LoRA持续预训练加监督微调），将2.6B混合模型适配为塔吉克语工具调用模型，使严格准确率从20.69提升至64.14，同时发现持续预训练会抹除模型原生的工具调用能力，必须与SFT结合使用。

    

    我们发布了Saidzoda-Tajik-LFM-2.6B，这是对2.6B参数混合模型LFM2.5-2.6B-Base的塔吉克语适配版本，专为在普通硬件上实现塔吉克语工具调用而构建。该方案分为两个阶段：首先在1,485,319篇清洗后的塔吉克语网页文档上进行3个epoch的LoRA持续预训练（CPT），然后在机器翻译的塔吉克语工具调用、保留作为回放的英语工具调用以及塔吉克语指令的混合数据上进行1个epoch的监督微调（SFT）。在一个基于BFCL v3并机器翻译为塔吉克语的174项工具调用测试集上，塔吉克语提示的严格完整调用准确率从基座模型的20.69提升至64.14（配对差异+43.45分，95%置信区间为+35.2至+51.7）；而一个同时接受基座模型自身调用格式的宽松解析器则将基座模型评为31.72。英语能力未检测到变化（76.55与77.24）。每个CPT检查点的得分均为零：持续预训练移除了模型原生的工具调用格式，并且正如宽松重新评分所显示的，还损失了部分技能本身。

    arXiv:2610.41576v1 Announce Type: new  Abstract: We release Saidzoda-Tajik-LFM-2.6B, a Tajik adaptation of the 2.6B-parameter hybrid model LFM2.5-2.6B-Base, built for tool calling in Tajik on modest hardware. The recipe has two stages: LoRA continued pretraining (CPT) on 1,485,319 cleaned Tajik web documents for 3 epochs, then one epoch of supervised fine-tuning (SFT) on a mix of machine-translated Tajik tool calls, English tool calls kept as replay, and Tajik instructions. On a 174-item tool-calling test set built from BFCL v3 and machine-translated to Tajik, strict full-call accuracy on Tajik prompts rises from 20.69 for the base model to 64.14 (paired difference +43.45 points, 95% interval +35.2 to +51.7); a lenient parser that also accepts the base model's own call formats puts the base at 31.72. English shows no detectable change (76.55 and 77.24). Every CPT checkpoint scores zero: CPT removes the native tool-call format and, as the lenient re-scoring shows, part of the skill itse
    
[^32]: BYOD：构建你自己的扩散语言模型——高效的自回归到扩散语言模型转换

    BYOD: Build Your Own DLM - Efficient Autoregressive-to-Diffusion Language Model Conversion

    [https://arxiv.org/abs/2610.41574](https://arxiv.org/abs/2610.41574)

    BYOD提出了一种仅用LoRA适配器的可泛化方案，能在单块GPU上数小时内将Gemma、Llama、Qwen、Mistral等自回归语言模型低成本转换为扩散语言模型，同时冻结全部基础模型参数并保留原模型知识。

    

    与自回归（AR）模型不同，扩散语言模型（DLM）允许并行词元更新、灵活生成和双向注意力，但现有的领先系统依赖于高昂的预训练或全模型微调。我们提出了Build Your Own DLM（BYOD），这是一种可泛化的仅使用LoRA的方案，用于参数高效的自回归到扩散语言模型转换。无需针对特定架构进行调整，BYOD即可在Gemma 2 9B、Llama 3.1 8B、Qwen2.5 7B和Ministral 8B上诱导出扩散式生成；据我们所知，这是首次报道的对Mistral模型进行AR到DLM的转换。这些适配器仅占最终参数的4.1-5.8%，而所有基础模型参数保持冻结；每次25,000次迭代的转换最多仅需1.024亿个词元，在一块Google Colab GPU上耗时3.1-4.0小时。所有四个模型都能从完全遮蔽的序列开始生成，在受控注意力测试中使用双向上下文，并保留了大量的母模型知识。BYOD-Gemma是最……（原文截断）

    arXiv:2610.41574v1 Announce Type: new  Abstract: Unlike autoregressive (AR) models, diffusion language models (DLMs) permit parallel token updates, flexible generation, and bidirectional attention, but leading systems rely on costly pretraining or full-model fine-tuning. We present Build Your Own DLM (BYOD), a generalizable LoRA-only recipe for parameter-efficient AR-to-DLM conversion. Without architecture-specific tuning, BYOD induces diffusion generation in Gemma 2 9B, Llama 3.1 8B, Qwen2.5 7B, and Ministral 8B; to our knowledge, this is the first reported AR-to-DLM conversion of a Mistral model. The adapters comprise 4.1-5.8% of final parameters while every base-model parameter remains frozen; each 25,000-iteration conversion needs at most 102.4M tokens, processed in 3.1-4.0 hours on one Google Colab GPU. All four models generate from fully masked sequences, use bidirectional context in a controlled attention test, and retain substantial parent-model knowledge. BYOD-Gemma is the str
    
[^33]: SCRBench：超越等价性检查的代码语义变化推理

    SCRBench: Beyond Equivalence Checking for Semantic Change Reasoning in Code

    [https://arxiv.org/abs/2610.41562](https://arxiv.org/abs/2610.41562)

    该论文提出SCRBench基准，通过诊断语义变化和重构代码更改序列这两项新任务，超越了传统的语义等价性“是/否”判断，从而更深入地评估语言模型对代码语义变化的真实理解能力。

    

    理解代码变化如何影响程序语义是语言模型代码推理的核心方面。这一能力通常通过询问两个程序是否语义等价来评估，但简单的“是/否”回答无法揭示模型是否知道是哪次代码更改导致了语义差异，也无法揭示是哪一序列的代码更改产生了修改后的程序。我们研究了两个直接提出这些问题的任务——诊断语义变化和重构代码更改序列——并结合语义等价性检测进行研究。我们构建了SCRBench，其中包含67,658对Python程序，这些程序对是通过应用保留或改变程序语义的代码更改序列生成的，并在4,500对的分层抽样子集上评估了五个语言模型。检测语义等价性暴露了较弱模型的不足，其准确率随着更改数量的累积而下降，但最强的模型仍接近性能上限。诊断语义变化则揭示了……

    arXiv:2610.41562v1 Announce Type: cross  Abstract: Understanding how code changes affect program semantics is a core aspect of code reasoning in language models. It is commonly evaluated by asking whether two programs are semantically equivalent, but a yes/no answer does not reveal whether a model knows which code change caused a semantic difference, or which sequence of code changes produced the modified program. We study two tasks that ask these questions directly, diagnosing semantic change and reconstructing sequences of code changes, alongside detecting semantic equivalence. We build SCRBench, 67,658 Python program pairs generated by applying sequences of code changes that preserve or alter program semantics, and evaluate five language models on a stratified subset of 4,500 pairs. Detecting semantic equivalence exposes weaker models, whose accuracy declines as changes accumulate, but the strongest models remain near the ceiling. Diagnosing semantic change reveals weaknesses that e
    
[^34]: 跨语言与跨文本粒度的CEFR等级评估

    CEFR Level Assessment across Languages and Text Granularities

    [https://arxiv.org/abs/2610.41553](https://arxiv.org/abs/2610.41553)

    该论文提出将焦点损失与监督对比学习相结合的统一训练目标，有效解决了CEFR等级分布不均衡和相邻等级边界模糊的问题，在单语和多语环境下的句子级与文档级评估中均取得了新的最先进性能。

    

    自动评估文本的CEFR等级是教育自然语言处理中的一项重要任务。然而，由于CEFR等级分布天然不均衡，且相邻等级之间边界模糊，准确的CEFR等级预测仍然具有挑战性。为了应对这些挑战，我们提出了一种统一的训练目标，将焦点损失与监督对比学习相结合。焦点损失通过强调欠代表样本和困难样本来缓解类别不平衡问题，而监督对比学习通过促进类内紧凑性和类间分离性来学习更具判别性的表示。我们在单语和多语环境下，对句子级和文档级CEFR评估进行了方法评估。实验结果表明，我们的方法在所有设置中始终优于强大的基线方法，在多项基准上建立了新的最先进性能。

    arXiv:2610.41553v1 Announce Type: new  Abstract: Automatic assessment of textual CEFR levels is an important task in educational natural language processing. However, accurate CEFR level prediction remains challenging due to the naturally imbalanced distribution of CEFR levels and the ambiguous boundaries between adjacent levels. To address these challenges, we propose a unified training objective that combines focal loss with supervised contrastive learning. Focal loss mitigates class imbalance by emphasizing underrepresented and difficult examples, while supervised contrastive learning learns more discriminative representations by encouraging intra-class compactness and inter-class separation. We evaluate the proposed method on sentence- and document-level CEFR assessment in both monolingual and multilingual settings. Experimental results demonstrate that our method consistently outperforms strong baselines across all settings, establishing new state-of-the-art performance on multipl
    
[^35]: 利用大型语言模型增强学术论文摘要与全文内容中的知识实体关系抽取

    Enhancing Knowledge Entity Relation Extraction in Abstracts and Full-text Content of Academic Articles Using Large Language Models

    [https://arxiv.org/abs/2610.41552](https://arxiv.org/abs/2610.41552)

    本研究系统比较了多种大语言模型策略用于学术论文的知识实体关系抽取，并创新性地提出了SLM增强开源LLM和闭源LLM增强开源LLM两种新组合策略。

    

    本研究探讨了大型预训练语言模型（LLM）在学术论文知识实体关系抽取（KERE）任务中的有效性。现有研究仅考察了单一类型的基于LLM的策略，并使用精确率、召回率等指标评估模型性能。为了提供更全面的评估，本研究考察了多种策略，包括基于闭源LLM的思维链（CoT）提示、经LoRA适配的开源LLM、LLM增强的小型预训练语言模型（SLM），以及SLM增强的闭源LLM。此外，本研究还引入了两种新策略，即SLM增强的开源LLM和闭源LLM增强的开源LLM。这些策略从多个角度进行评估，包括指标表现、缓解SLM中常见错误的能力，以及模型优化的潜在方向。本研究在自然语言处理（NLP）领域人工构建了两个标注数据集，涵盖……（摘要在此处截断）

    arXiv:2610.41552v1 Announce Type: new  Abstract: This study investigates the effectiveness of large pre-trained language models (LLMs) for knowledge entity relation extraction (KERE) from academic papers. Existing studies have examined individual types of LLM-based strategies and evaluated model performance using metrics such as precision and recall. To provide a broader evaluation, this study investigates multiple strategies, including closed-source LLM-based CoT prompting, LoRA-adapted open-source LLMs, LLM-enhanced small pre-trained language models (SLMs), and SLM-enhanced closed-source LLMs. In addition, this study introduces two strategies, namely SLM-enhanced open-source LLMs and closed-source LLM-enhanced open-source LLMs. These strategies are evaluated from multiple perspectives, including metrics, their ability to mitigate errors observed in SLMs, and potential directions for model optimization. This study manually constructs two annotated datasets in the NLP domain, covering 
    
[^36]: DAPS：面向LLM定理证明的依赖感知前提选择

    DAPS: Dependency-Aware Premise Selection for LLM Theorem Proving

    [https://arxiv.org/abs/2610.41536](https://arxiv.org/abs/2610.41536)

    该论文提出DAPS方法，通过首次系统挖掘53个Lean Blueprint项目构建依赖关系语料库，将数学中的依赖结构信号引入前提选择的训练数据与模型中，克服了现有基于语义相似度的选择器在检索偏斜和分布外查询上的固有局限。

    

    基于LLM的Lean 4定理证明发展迅速，但证明器难以利用新贡献的引理，因为检索偏向于流行的基础引理，使得正确的前提被埋没在数十万个候选之中。现有的神经选择器按语义相似度对候选进行排序，存在三个内在局限：频率偏斜的检索、孤立的成对排序，以及在分布外查询（如竞赛题目和非正式语言）上的性能退化。鉴于数学是按依赖关系而非相似性来构建结构的，我们将这一信号重新引入训练数据和模型中。我们整理了两项未被充分利用的依赖资源。我们对53个Lean Blueprint项目进行了首次系统性挖掘，构建了一个此前未被开发的语料库，其中包含3,805个人工整理的非正式+正式节点，捕捉了Mathlib4中缺失的罕见研究级前提。我们进一步将其与提取的……（原文摘要在此截断）

    arXiv:2610.41536v1 Announce Type: cross  Abstract: LLM-based theorem proving in Lean 4 has advanced rapidly, but provers are difficult to leverage newly contributed lemmas, because retrieval skews toward popular foundational lemmas, leaving the right premises buried among hundreds of thousands of candidates. Existing neural selectors rank candidates by semantic similarity, with three intrinsic limitations: frequency-skewed retrieval, isolated pairwise ranking, and degradation on out-of-distribution queries such as competition problems and informal language. Given that mathematics is structured by dependencies rather than similarities, we restore this signal to both the training data and the model. We curate two underused dependency resources. We conduct the first systematic mining of 53 Lean Blueprint projects, a previously untapped corpus of 3,805 manually curated informal+formal nodes that capture rare, research-level premises absent from Mathlib4. We further pair it with the extract
    
[^37]: 通过数据增强与迁移学习提升小规模数据集上的情感分类

    Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning

    [https://arxiv.org/abs/2610.41518](https://arxiv.org/abs/2610.41518)

    本研究比较了EDA、回译和上下文词替换三种数据增强方法在传统机器学习分类器与BERT模型上的效果，发现上下文增强对BERT模型增益最稳定，而EDA和回译更适合提升传统分类器的性能。

    

    小规模情感分类常常受到数据稀缺问题的困扰，这限制了模型的泛化能力。本研究评估并比较了三种数据增强策略的有效性：简易数据增强、回译和上下文词替换（nlpaug风格），并将其分别应用于传统机器学习分类器（逻辑回归、随机森林）和基于Transformer的模型（BERT）。我们通过总结近期研究结果并进行针对性的直接对比实验，在低资源情感数据集上进行了全面的实证比较。研究结果表明，所有数据增强方法均能提升性能。其中，上下文增强为BERT模型带来了最稳定的增益，而EDA和回译则为传统分类器带来更大的收益。这些见解有助于指导针对不同模型类型选择合适的数据增强技术。

    arXiv:2610.41518v1 Announce Type: cross  Abstract: Small-scale sentiment classification often suffers from data scarcity, which limits the generalization ability of the models. This study evaluates and compares the effectiveness of three data augmentation strategies: Easy Data Augmentation (EDA), back-translation, and contextual token substitution (nlpaug-style), with both traditional machine learning classifiers (Logistic Regression, Random Forest) and transformer-based models (BERT). We perform a comprehensive empirical comparison with low-resource sentiment datasets by summarizing the results of recent studies and performing targeted head-to-head experiments. Our findings indicate that all augmentation methods improve performance. Contextual augmentation yields the most consistent gains for BERT models, while EDA and back-translation provide greater benefits for traditional classifiers. These insights help guide the selection of data augmentation techniques tailored to model type an
    
[^38]: 双预算下的稀疏文本扩散：上下文连接性与专家激活

    Sparse Text Diffusion under Dual Budgets: Context Connectivity and Expert Activation

    [https://arxiv.org/abs/2610.41487](https://arxiv.org/abs/2610.41487)

    该论文提出以“注意力边数”和“专家激活数”两个独立预算来刻画长上下文扩散模型的计算成本，并据此分析稀疏-专家配置 DualBudget，在最高 16K 的长度范围内的摘要、多跳问答、对话等任务上优于基线。

    

    长上下文扩散模型在词元交互和前馈计算上都需要反复付出代价。我们将这些成本形式化为两个相互独立的预算：注意力图可用的边数，以及每个词元所激活的专家数量。这一视角将局部、扩张和全局注意力与条件化专家计算统一起来，而不假设稀疏性本身就能保留文档级信息。我们规范了归一化掩码注意力，区分了去噪深度与采样迭代次数，并推导了图结构和计算量的上下界。文中对一种报告的稀疏-专家配置（记为 DualBudget）进行了分析，涵盖科学摘要、多跳问答、对话和句子对预测等任务，并保留了所有已提供的表格测量结果。长度研究显示，在 16K 范围内该方法的性能优于所列基线，但随着长度增加出现了可测量的性能下降。消融实验表明三项报告指标的改进情况不一，而原文在此处截断。

    arXiv:2610.41487v1 Announce Type: new  Abstract: Long-context diffusion repeatedly pays for both token interaction and feed-forward computation. We formulate these costs as two separate budgets: the edges available to an attention graph and the experts activated for each token. This view connects local, dilated, and global attention with conditional expert computation without assuming that sparsity alone preserves document-level information. We specify normalized masked attention, distinguish denoising depth from sampling iterations, and derive graph and compute bounds. A reported sparse-expert configuration, denoted DualBudget, is analyzed across scientific summarization, multi-hop question answering, dialogue, and sentence-pair prediction. All supplied table measurements are retained. The length study shows gains over the listed baselines through 16K, together with measurable degradation as length increases. Ablations reveal mixed changes across the three reported scores, while the c
    
[^39]: KEEPER：为长时程LLM智能体学习控制充分的文本记忆

    KEEPER: Learning Control-Sufficient Textual Memory for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2610.41482](https://arxiv.org/abs/2610.41482)

    该论文提出KEEPER框架，在共享递归接口下比较四种辅助训练目标，发现基于奖励相关的预测目标能在离线强化学习前后均取得最佳任务表现，并提出应同时评估记忆的错误持续性与有效信息保留情况。

    

    递归式文本记忆可能会将早期错误带入后续更新中，然而仅凭任务表现并不能确立记忆的可靠性。我们将“自我一致性偏差”定义为传入记忆中已存在的错误断言持续留存的现象。我们提出了KEEPER，在共享的递归接口下，沿信息内容和更新敏感性两个维度对四种辅助训练目标进行了比较分析，考察监督目标与智能体性能之间的关系。在所评估的条件中，与奖励相关的预测在离线强化学习前后均取得了最高的任务表现，优于联合监督。针对一条基线轨迹的断言级分析记录了尽管存在纠正性观察，错误仍然持续存在的现象。这些分析共同表明，应当在任务结果之外，同时评估错误持续性和有效信息保留情况。

    arXiv:2610.41482v1 Announce Type: new  Abstract: Recursive textual memory can carry earlier errors into later updates, yet task performance alone does not establish memory reliability. We define self-consistency bias as the persistence of incorrect claims already present in the incoming memory. We present KEEPER, a comparative analysis of four auxiliary objectives along information-content and update-sensitivity axes within a shared recursive interface, examining how supervision targets relate to agent performance. Among evaluated conditions, reward-related prediction achieve the highest task performance before and after offline reinforcement learning, outperforming joint supervision. A claim-level analysis of one baseline trajectory documents error persistence despite corrective observations. Together, these analyses motivate evaluating error persistence and valid-information retention alongside task outcomes.
    
[^40]: KV缓存修补：追踪微调行为如何以及在何处产生

    KV-Cache Patching: Tracing How and Where Fine-Tuned Behaviors Arise

    [https://arxiv.org/abs/2610.41480](https://arxiv.org/abs/2610.41480)

    通过在LoRA微调模型与基础模型之间互相修补KV缓存，研究揭示了微调行为主要通过注意力机制对先前token（尤其是系统提示和聊天模板等少数关键位置）的 altered 处理被编码在KV缓存中，可以跨模型转移或消除。

    

    微调语言模型会改变其响应方式，但这种微调行为究竟在哪里实例化？为了更好地理解这一点，我们评估了模型行为的变化是仅在生成答案token（即变化显现之处）时产生，还是通过对先前token处理方式的改变经由注意力机制而引起。为此，我们将LoRA微调模型的KV缓存修补到未修改的基础模型中，反之亦然。我们的结果表明，对于某些行为，如用德语回答或突现错位，修补KV缓存确实可以将它们从微调模型转移到基础模型。相应地，从基础模型向微调模型修补KV缓存可以部分或完全消除微调行为。此外，我们发现修补的效果随修补KV缓存比例的增加而增强，并集中在少数关键位置上，其中系统提示和聊天模板token具有不成比例的影响。

    arXiv:2610.41480v1 Announce Type: new  Abstract: Fine-tuning a language model changes how it responds, but where is this fine-tuned behavior instantiated? To better understand this, we evaluate whether the change in model behavior arises only during the generation of the answer tokens where the change manifests, or whether altered processing of previous tokens causes it via the attention mechanism. For this, we patch the KV cache from a LoRA fine-tuned model into the unmodified base model, and vice versa. Our results show that for some behaviors, like responding in German or emergent misalignment, patching the KV cache can indeed transfer them from the fine-tuned to the base model. Correspondingly, patching the KV cache from base to fine-tune can partially or fully remove fine-tuned behavior. Furthermore, we find that the effect of patching grows with how much of the KV cache is patched and concentrates at a few pivotal positions, with system-prompt and chat-template tokens having disp
    
[^41]: 以患者为中心的人工智能辅助临床就诊准备的前瞻性随机研究

    A prospective randomized study of patient-centered AI for clinical visit preparation

    [https://arxiv.org/abs/2610.41437](https://arxiv.org/abs/2610.41437)

    本研究开发了基于大语言模型的患者就诊准备智能体，并通过802名成年人参与的前瞻性随机试验证明，该智能体在帮助患者做好就诊准备方面显著优于通用聊天机器人。

    

    患者在参加时间短暂、日程紧凑的临床就诊时，常常不清楚该期待什么、该问些什么，如今有数以百万计的成年人会借助通用AI聊天机器人在就诊前进行准备。然而，对话式AI能否改善患者的就诊准备状态仍不清楚。我们设计了一个基于大语言模型（LLM）的就诊准备智能体，通过收集临床背景信息、设定就诊预期、提供相关健康信息、为临床医生构建按优先级排序的问题清单并生成总结，在就诊前为患者提供引导。在Google Health应用中进行的一项前瞻性、经知情同意的随机研究中，802名美国成年人使用就诊准备智能体（标准版或简洁版）或基于同一大语言模型构建的仅有最少提示的对照聊天机器人，共进行了866次健康相关对话。在盲法临床医生评估和患者自我报告中，就诊准备智能体的表现均显著优于对照组。在临床医生评审……（原文摘要在此处被截断）

    arXiv:2610.41437v1 Announce Type: cross  Abstract: Patients often arrive at brief, time-constrained clinical visits unsure of what to expect or ask, and millions of adults now turn to general-purpose AI chatbots to prepare beforehand. Yet whether conversational AI improves visit readiness remains unclear. We designed a large language model (LLM)-based visit preparation agent to guide patients before an appointment by gathering clinical context, setting visit expectations, providing relevant health information, building a prioritized list of questions for their clinician, and generating a summary. In a prospective consented randomized study in the Google Health app, 802 U.S. adults held 866 health-related conversations with the visit preparation agent (standard or concise variant) or a minimally prompted control chatbot built on the same LLM. The visit preparation agent significantly outperformed the control across blinded clinician evaluations and patient self-reports. In clinician rev
    
[^42]: BlindToChange：AI智能体在被告知时能找到变化，但它们在需要时会主动去查看吗？

    BlindToChange: AI Agents Can Find Changes When Asked But Do They Look When Needed?

    [https://arxiv.org/abs/2610.41435](https://arxiv.org/abs/2610.41435)

    提出BlindToChange基准，揭示AI智能体虽能在明确要求下找到近期变化，但在用户仅陈述信息需求时往往不会主动发现并考虑影响回答的最新事件。

    

    鉴于AI智能体能够实时对网络进行广泛而深入的搜索，人们很容易认为其输出结果已经考虑了最新的世界动态。我们表明这一结论在实践中并不成立。虽然智能体可以搜索并轻松找到近期变化，但其回答的正确性取决于它们是否进行了搜索以及搜索了什么。像BrowseComp这样的流行智能体搜索基准无法暴露智能体未能意识到自身假设需要验证的失败情况，因为这些基准明确指出了需要查找的内容。相比之下，许多用户请求仅陈述最终的信息需求（例如一次旅行的行程安排），而需要智能体自行发现相关的变化（例如某博物馆临时闭馆）并考虑其影响。为了凸显这类缺陷，我们提出了BlindToChange，一个衡量智能体能否主动发现影响其回答的近期事件的基准。

    arXiv:2610.41435v1 Announce Type: new  Abstract: Given that AI agents can search the web broadly and deeply in real time, it is tempting to conclude that their outputs account for recent world events. We show that this conclusion does not hold in practice. While agents can search and easily find recent changes, the correctness of their responses depends on whether they search and what they search for. Popular agentic search benchmarks like BrowseComp cannot expose failures wherein agents do not recognize their own assumptions that should be verified, since these benchmarks explicitly state what needs to be found. In contrast, many user requests state only the final information need (e.g., an itinerary for a trip), leaving the agent to discover relevant changes (e.g., the temporary closure of a museum) and account for their consequences. To highlight such gaps, we introduce BlindToChange, a benchmark that measures whether agents proactively discover recent events that affect their respo
    
[^43]: 混合计算机使用：面向通用智能体的统一编程、语义与视觉接口

    Hybrid Computer Use: A Unified Programmatic, Semantic, and Visual Interface for General Agents

    [https://arxiv.org/abs/2610.41432](https://arxiv.org/abs/2610.41432)

    该论文提出混合计算机使用（HCU），一个统一融合编程式、语义与视觉三种交互通道的通用智能体接口，使智能体能够自主选择和组合各通道，从而突破传统“截图+鼠标键盘”人类交互先验的局限。

    

    使用计算机的智能体日益承担从日常桌面活动到长程专业工作流的各类任务。然而，以截图和鼠标键盘操作为中心的接口会强加一种“人类交互先验”，需要反复解读屏幕内容，同时使计算机的语义与编程能力得不到充分展现。我们主张暴露这些能力，并让模型自主选择和组合它们。我们提出了混合计算机使用，这是一个面向通用智能体的统一接口，结合了三个互补的交互通道：通过代码和命令行工具进行的编程式交互、通过无障碍功能进行的语义交互，以及通过图形用户界面进行的视觉交互。我们的 HCU 支持高效的 AX 交互，并在通道切换时防范基于过时状态的操作。补充的 HCU 技能可指导智能体选择和协调各通道，以有效使用统一接口。

    arXiv:2610.41432v1 Announce Type: cross  Abstract: Computer-use agents increasingly tackle tasks ranging from everyday desktop activities to long-horizon professional workflows. Yet interfaces centered on screenshots and mouse-and-keyboard actions can impose a human-interaction prior, requiring repeated screen interpretation while leaving the computer's semantic and programmatic capabilities underexposed. We argue for exposing these capabilities and letting models choose and combine them. We introduce Hybrid Computer Use (HCU), a unified interface for general agents that combines three complementary channels: programmatic interaction through code and command-line tools, semantic interaction through accessibility (AX), and visual interaction through graphical user interfaces (GUI). Our HCU supports efficient AX interaction and guards against stale-state actions during channel switching. A supplementary HCU skill guides agents in selecting and coordinating channels to use the unified int
    
[^44]: SparseGDN：线性注意力中的块级稀疏化

    SparseGDN: Chunk Sparse in Linear Attention

    [https://arxiv.org/abs/2610.41420](https://arxiv.org/abs/2610.41420)

    SparseGDN提出了一种硬件感知的稀疏线性注意力框架，通过结构化块内近似与动态循环事件选择相结合，平均仅保留约29%的循环事件，在保持输出质量的同时实现了1.50-1.94倍的长上下文预填充加速。

    

    门控Delta网络提供了线性时间的序列建模能力，但由于密集的块内计算、循环状态更新和中间状态存储，长上下文预填充的代价依然高昂。我们提出了SparseGDN，这是一个硬件感知框架，将结构化块内近似与依赖输入的循环事件选择相结合，同时为每个token生成输出。块相邻求解器减少了局部Delta求解的计算量，而动态选择器则在头对之间分配循环更新预算。其GPU实现仅为被选中的事件实例化更新，并利用紧凑的状态快照和组内局部交互来重建token输出，从而降低计算量和内存流量。我们在Qwen3.5-4B和Qwen3.5-9B模型上，针对RULER任务以及4K到128K的上下文长度对SparseGDN进行了评估。在平均仅保留约29%循环事件的情况下，SparseGDN实现了1.50-1.94倍的GDN核心预填充加速（摘要原文在此处被截断）。

    arXiv:2610.41420v1 Announce Type: cross  Abstract: Gated Delta Networks (GDNs) offer linear-time sequence modeling, yet long-context prefill remains costly due to dense intra-chunk computation, recurrent-state updates, and intermediate state storage. We introduce SparseGDN, a hardware-aware framework that combines structured intra-chunk approximation with input-dependent recurrent-event selection while producing outputs for every token. A block-adjacent solver reduces local Delta-solve computation, while a dynamic selector allocates recurrent-update budgets across head pairs. The GPU implementation materializes updates only for selected events and reconstructs token outputs using compact state snapshots and group-local interactions, reducing computation and memory traffic. We evaluate SparseGDN on Qwen3.5-4B and Qwen3.5-9B across RULER tasks and context lengths from 4K to 128K. Retaining approximately 29% of recurrent events on average, SparseGDN achieves 1.50-1.94x GDN-core prefill sp
    
[^45]: ProSearch：实验科学中多约束协议检索的基准测试

    ProSearch: Benchmarking Multi-Constraint Protocol Retrieval in Experimental Science

    [https://arxiv.org/abs/2610.41396](https://arxiv.org/abs/2610.41396)

    提出了ProSearch基准，用于系统评估AI在实验科学中检索满足多个相互依赖约束的实验方案的能力，该基准基于真实实验记录构建，包含跨六个领域的1,515个带有显式约束的查询。

    

    在“AI for Science”（科学人工智能）场景中，实验设计和方法复现需要检索满足多个相互依赖约束的实验方案，然而现有基准缺乏对这一能力的系统性评估。为填补这一空白，我们提出了ProSearch，一个面向实验科学多约束协议检索的基准。ProSearch具有三个关键特性：（1）真实性，查询基于真实实验记录和专家定义的模板；（2）复杂性，涵盖60个实验领域及其相互依赖关系；（3）诊断性，每个查询包含3-5个显式约束，包括数值约束和否定约束。我们通过基于大语言模型的信息抽取、数据驱动的约束选择、受控的查询生成以及基于证据的相关性验证来构建ProSearch。最终形成的基准包含横跨六个实验科学领域的1,515个查询。我们在ProSearch上使用19个（模型/系统）开展了大量实验……

    arXiv:2610.41396v1 Announce Type: cross  Abstract: In AI for Science scenarios, experimental design and method reproduction require retrieving protocols that satisfy multiple interdependent constraints, yet existing benchmarks lack systematic evaluation of this capability. To address this gap, we introduce ProSearch, a benchmark for multi-constraint protocol retrieval in experimental science. ProSearch has three key features: (1) realism, with queries grounded in real experimental records and expert-defined templates; (2) complexity, with 60 experimental fields and their dependencies; and (3) diagnosticity, with 3-5 explicit constraints per query, including numerical and negation constraints. We construct ProSearch through LLM-based extraction, data-driven constraint selection, controlled query generation, and evidence-based relevance validation. The resulting benchmark contains 1,515 queries across six experimental science domains. We conduct extensive experiments on ProSearch with 19
    
[^46]: 基于大语言模型的学术论文质量评估评审范式的比较评估

    Comparative Evaluation of LLM-Based Review Paradigms for Academic Paper Quality Assessment

    [https://arxiv.org/abs/2610.41391](https://arxiv.org/abs/2610.41391)

    该研究系统比较了五种基于LLM的论文评审范式，发现重新校准后的直接绝对评分能以仅24%的API调用量取得优于Elo比较排序的人类评分一致性，且仅用摘要输入以及模型选择的影响与范式选择同等重要。

    

    arXiv:2610.41391v1 公告类型：新论文 摘要：投稿量的持续增长重新激发了人们利用大语言模型（LLM）辅助同行评审的兴趣。然而，此类评估应如何组织构建、模型本身有多重要、以及LLM评分能否预测研究影响力，目前仍不明确。我们在232篇分层抽样的ICLR 2025投稿上，跨三种输入表示和三种LLM评估了五种范式。直接绝对评分在与人类评分的秩相关性上与比较排序表现相当，但其大部分预测被压缩至接收阈值之上，几乎无法标记出评审人评分低于阈值的论文。通过基准率匹配的重新校准控制实验，我们定位了原因：比较排序提供的是位置信息而非更优的排序；重新校准后，直接评分比Elo方法达到更高的一致性，同时API调用量仅为后者的24%。仅使用摘要的输入优于基于全文的摘要化，且模型选择与范式选择同等重要。在全部11,513篇有评分投稿上的复现实验……（原文截断）

    arXiv:2610.41391v1 Announce Type: new  Abstract: Growing submission volumes have renewed interest in using large language models (LLMs) to support peer review. It remains unclear how such evaluation should be structured, how much the model matters, and whether LLM scores predict research impact. We evaluate five paradigms across three input representations and three LLMs on 232 stratified ICLR 2025 submissions. Direct absolute rating matches comparative ranking on rank correlation with human scores, yet compresses most predictions above the accept threshold, flagging almost none of the papers reviewers scored below it. A base-rate-matched recalibration control locates the cause: comparative ranking supplies placement, not better ordering, and after recalibration Direct Rating reaches higher agreement than Elo while making 24% as many API calls. Abstract-only input beats full-paper summarization, and model choice matters as much as paradigm choice. Replicating on all 11,513 scored submi
    
[^47]: 我们从所遗忘中保留了什么：免训练MoE专家剪枝中的替换算子与路由语义

    What We Keep of What We Forget: Replacement Operators and Routing Semantics in Training-Free MoE Expert Pruning

    [https://arxiv.org/abs/2610.41366](https://arxiv.org/abs/2610.41366)

    本文系统隔离并比较了免训练MoE专家剪枝中替换算子的作用，发现存活专家门控缩放是否重新归一化是影响最大的隐式实现选择，并提出了一种闭式低秩替换算子“倾斜新手”，在50%剪枝率下以仅1.2-1.8%的参数增量获得显著性能提升。

    

    arXiv:2610.41366v1 公告类型：交叉发布。摘要：免训练专家剪枝需要做出两个决策：删除哪些专家，以及当路由器仍然选择已被删除的专家时，该层应计算什么。已发表的方法将这两个决策同时改变。本文在固定剪枝掩码的前提下，在OLMoE、DeepSeek-V2-Lite和Moonlight三个模型上（以Granite作为第四个路由器）比较了来自三个家族的替换算子，并使用配对的、经过多重性校正的统计检验。研究发现，影响最大的因素是一个通常被隐式处理的选择：存活专家的门控缩放。对于训练时未使用top-k重新归一化的路由器，对存活专家门控进行重新归一化（一种常见的实现选择）会损失0.8-8.5个百分点，并逆转“重新路由是否优于MoNE均值输出替换”的结论；对于训练时使用了重新归一化的两个路由器，在50%剪枝率下训练所得的缩放值表现更优。本文提出了“倾斜新手”，这是一种针对被剪枝专家输入的闭式低秩映射，其秩为0的情形即退化为MoNE的常数替换。在50%剪枝率下，该方法仅增加1.2-1.8%的参数，即可分别带来3.45和1.…的提升。

    arXiv:2610.41366v1 Announce Type: cross  Abstract: Training-free expert pruning makes two decisions: which experts to remove, and what the layer computes when its router still selects a removed expert. Published methods change both at once. Holding the mask fixed, we compare replacement operators from three families on OLMoE, DeepSeek-V2-Lite and Moonlight, with Granite as a fourth router, using paired, multiplicity-corrected tests. The largest effect is a choice usually left implicit: the survivors' gate scale. On routers trained without top-k renormalization, renormalizing the survivors' gates, a common implementation choice, costs 0.8-8.5 points and reverses whether rerouting beats MoNE's mean-output replacement; on two routers trained with renormalization, the trained scale wins at 50% pruning. We introduce the tilted novice, a closed-form low-rank map of the pruned expert's input whose rank-0 case is MoNE's constant. At 50% pruning it adds 1.2-1.8% parameters and gains 3.45 and 1.
    
[^48]: SERIAL-7：跨表示形式推导、实现与保持机器可验证的输出

    SERIAL-7: Deriving, Realizing, and Preserving Machine-Verifiable Outputs Across Representations

    [https://arxiv.org/abs/2610.41308](https://arxiv.org/abs/2610.41308)

    该论文提出SERIAL-7——一个基于可执行验证框架的韩英双语基准，通过“推导—实现—保持”三个阶段和43个原子表示约束，系统评估大语言模型跨七种表示形式生成机器可验证结构化输出的能力，揭示了模型在表面答案正确之外仍存在大量工件级合规失败。

    

    大语言模型越来越多地生成由程序和下游系统使用的结构化输出，在这种场景下，仅有语义正确性并不能保证可靠的执行。我们提出了SERIAL-7，一个围绕可执行验证框架构建的韩英双语基准，用于跨七种表示形式的机器可验证生成。基于177个经过验证的规范工件，SERIAL-7将语义推导、受约束的工件生成和跨表示保持三个环节分离开来。其43个原子表示约束与针对格式、结构和任务特定语义的可执行检查相配对，能够实现从表示合规性到完整工件有效性的渐进式更严格验证。在十个模型的评估中，REALIZE-L3 HSR的得分范围为42.0%至93.9%，而完全合规指标暴露了在正确最终答案和有效表示之外的额外工件级失败。

    arXiv:2610.41308v1 Announce Type: cross  Abstract: Large language models increasingly generate structured outputs consumed by programs and downstream systems, where semantic correctness alone does not guarantee reliable execution. We introduce SERIAL-7, a Korean-English benchmark built around an executable verification framework for machine-verifiable generation across seven representations. Using 177 validated canonical artifacts, SERIAL-7 separates semantic derivation (Derive), constrained artifact generation (Realize), and cross-representation preservation (Preserve). Its 43 atomic representation constraints are paired with executable checks over format, structure, and task-specific semantics, enabling progressively stricter verification from representation compliance to full artifact validity. Across ten models, REALIZE-L3 HSR ranges from 42.0% to 93.9%, while Full Compliance exposes additional artifact-level failures beyond correct final answers and valid representations. These re
    
[^49]: 从转录到重构：定义并基准测试有依据的生成式语音识别

    From Transcription to Reformulation: Defining and Benchmarking Grounded Generative Speech Recognition

    [https://arxiv.org/abs/2610.41292](https://arxiv.org/abs/2610.41292)

    该论文提出了“有依据的生成式语音识别”这一新范式，将语音识别从生成单一转录文本重新定义为生成语义上有依据的文本，建立了四级重构能力层次结构，并发布了首个综合评测基准GSR-Bench。

    

    语音语言模型已将自动语音识别（ASR）从单纯的转录扩展为通用的语音到文本生成。然而，现有基准测试仍然假设每段语音只有一个唯一的标准转录文本，当模型生成可读的、格式化的或结构化的输出时，这一假设便难以成立。在这项工作中，我们提出了有依据的生成式语音识别，将语音识别重新定义为生成语义上有依据的文本，而非单一的转录文本。我们定义了一个有依据的重构空间，用以刻画由输入语音所支持的所有有效文本实现形式，并建立了重构能力的四级层次结构，包括字面转录、规范化重构、语义重构和基于语音的结构化。基于这一表述，我们提出了GSR-Bench，这是首个旨在评估超越精确转录的生成式语音识别系统的综合性基准。

    arXiv:2610.41292v1 Announce Type: new  Abstract: Speech language models have expanded automatic speech recognition (ASR) from transcription to general speech-to-text generation. Yet existing benchmarks still assume a single ground-truth transcript per utterance, an assumption that struggles when models produce readable, formatted, or structured outputs. In this work, we introduce Grounded Generative Speech Recognition (GSR), reformulating speech recognition as generating semantically grounded text instead of a single transcript. We define a grounded reformulation space that characterizes all valid textual realizations supported by the input speech, and establish a four-level hierarchy of reformulation capabilities, including literal transcription, canonical reformulation, semantic reformulation and speech-grounded structuring. Based on this formulation, we present GSR-Bench, the first comprehensive benchmark designed to evaluate generative speech recognition systems beyond exact transc
    
[^50]: 基于智能体软令牌的模型内上下文压缩，面向长时程LLM智能体

    In-Model Context Compression with Agentic Soft Tokens for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2610.41291](https://arxiv.org/abs/2610.41291)

    提出免训练的智能体软令牌压缩方法（ASTC），利用混合GDN-注意力的原生循环动态将已吸收信息压缩为模型内软令牌，在降低长时程LLM智能体计算成本的同时保留关键决策信息。

    

    长时程LLM智能体在解决复杂且持续的任务中扮演着重要角色，其交互上下文会不断增长。然而，这些累积的上下文会带来巨大的计算成本，并掩盖关键的决策相关证据。现有方法通常训练外部压缩器或上下文管理智能体，将上下文重写为较短的硬令牌序列，但这种外部选择可能与LLM不断演化的内部表示不一致，且必然是有损的。在本文中，我们提出了智能体软令牌压缩，这是一种免训练的压缩方法，在LLM智能体内部使用软令牌来处理长时程任务。具体而言，ASTC借助常用的混合门控Delta网络-注意力机制，利用模型原生的循环动态来确定已被吸收的信息，然后将这些信息作为智能体软令牌向前传递，同时将精确值保留为硬令牌。结果……

    arXiv:2610.41291v1 Announce Type: new  Abstract: Long-horizon LLM agents play an important role in solving complex and sustained tasks, with ever-growing interaction contexts. However, these accumulated contexts will raise substantial computational cost and obscure key decision-relevant evidence. Existing methods typically train external compressors or context-management agents to rewrite context as shorter hard-token sequences, but such external selection may not align with the LLM's evolving representation and is necessarily lossy. In this paper, we introduce Agentic Soft-Token Compression (ASTC), a training-free compression method using soft tokens within LLM agent for long-horizon tasks. Specifically, ASTC uses the model's native recurrent dynamics to determine the already absorbed information, leveraging the commonly-used hybrid Gated Delta Network (GDN)-attention. It then carries this information forward as agentic soft tokens while retaining exact values as hard tokens. The resu
    
[^51]: PsyInterviewee：心理健康访谈模拟中的信息不变性与关系适应性

    PsyInterviewee: Informational Invariance and Relational Adaptivity in Mental Health Interview Simulation

    [https://arxiv.org/abs/2610.41289](https://arxiv.org/abs/2610.41289)

    该论文提出PsyInterviewee框架，通过分离受保护的真实信息通道与动态交互通道并采用非对称写入权限机制，在心理健康访谈模拟中同时保证临床事实的信息不变性与交互行为的适应性。

    

    基于大语言模型（LLM）的模拟受访者能够支持访谈员培训以及心理健康访谈系统的评估。现有研究主要通过更丰富的患者档案和行为建模来提升拟人化程度。然而，仅凭拟人化并不足以确立模拟的可靠性。可靠的模拟既需要保留基于档案的临床事实（信息不变性），又需要使信息透露和交互行为适应访谈情境（关系适应性）。为此，我们提出了PsyInterviewee，该方法将受保护的真实信息通道与动态交互通道相分离，并对生成的临床陈述与患者档案进行核对验证。具体而言，非对称写入权限允许对话上下文更新信息透露状态和行为状态，但无权创建新的临床事实；这些状态进一步指导哪些有档案支持的信息被透露以及如何表达。在一项自动化评估中……（原文摘要至此截断）

    arXiv:2610.41289v1 Announce Type: new  Abstract: LLM-based simulated interviewees support interviewer training and the evaluation of mental health interviewing systems. Existing studies have primarily focused on improving human-likeness through richer patient profiles and behavioral modeling. However, human-likeness alone does not establish their reliability. Reliable simulation requires preserving profile-grounded clinical facts (informational invariance) while adapting disclosure and interaction behavior to the interview (relational adaptivity). To this end, we introduce PsyInterviewee, which separates a protected truth channel from a dynamic interaction channel and verifies generated clinical claims against the profile. Specifically, asymmetric write permissions allow dialogue context to update disclosure and behavioral states without authorizing new clinical facts; these states guide what profile-supported information is revealed and how it is expressed. In an automated evaluation 
    
[^52]: SciGraphSelect：基于知识-推理图的可审计科学数据选择

    SciGraphSelect: Auditable Scientific Data Selection through Knowledge--Reasoning Graphs

    [https://arxiv.org/abs/2610.41280](https://arxiv.org/abs/2610.41280)

    提出了SciGraphSelect框架，将领域知识、推理操作和参考证实关系组织成类型化知识-推理图，通过概念覆盖率、关系与路径共支持及方向性对齐对科学文档进行可解释评分，为科学训练语料库构建提供可审计、可追溯的数据选择。

    

    科学数据选择通常由相关性或效用分数驱动，对于文档为何被保留所提供的洞察十分有限。我们提出了SciGraphSelect，一个以参考为依据的可审计科学数据选择框架。SciGraphSelect将领域特定知识、可复用的推理操作以及经参考证实的关系组织成一个类型化的知识-推理图，然后利用概念覆盖率、关系与路径共支持以及方向性对齐对候选文档进行评分。为避免对文档嵌入的过度解读，该方法在参考侧保留丰富的关系结构，同时对候选侧证据采取保守处理，不假设关系蕴含。通过揭示每次选择背后基于参考的证据，SciGraphSelect为科学训练语料库的上游构建带来了可追溯性，并与下游的科学验证形成互补。

    arXiv:2610.41280v1 Announce Type: cross  Abstract: Scientific data selection is typically driven by relevance or utility scores, providing limited insight into why a document is retained. We introduce SciGraphSelect, a reference-grounded framework for auditable scientific data selection. SciGraphSelect organizes domain-specific knowledge, reusable reasoning operations, and reference-attested relations into a typed knowledge--reasoning graph, then scores candidate documents using concept coverage, relation and path co-support, and directional alignment. To avoid over-interpreting document embeddings, rich relational structure is preserved on the reference side while candidate-side evidence is treated conservatively without assuming relation entailment. By exposing the reference-grounded evidence behind each selection, SciGraphSelect brings traceability to the upstream construction of scientific training corpora and complements downstream scientific verification.
    
[^53]: TRIDENT与SHAKTI：面向印地语AI文本检测的框架与基准

    TRIDENT and SHAKTI: A Framework and Benchmark for Hindi AI-Text Detection

    [https://arxiv.org/abs/2610.41232](https://arxiv.org/abs/2610.41232)

    该论文针对低资源语言AI文本检测研究不足的问题，提出了大规模印地语AI文本检测基准SHAKTI，以及基于往返翻译语义一致性的三变体检测框架TRIDENT。

    

    随着大型语言模型（LLM）日益走向多语言化，区分人类撰写文本与AI生成文本对于维护信息完整性至关重要。然而，相关研究仍以英语为中心，针对印地语等低资源语言的基准十分有限。为填补这一空白，我们提出了SHAKTI，一个大规模印地语基准，涵盖四种文体、八个LLM家族、多种提示策略以及对抗性变换。我们通过人工标注评估了其质量，并分析了人类文本与AI生成文本在语言学、语法和情感层面的差异。基于这些发现，我们提出了TRIDENT，一个基于往返翻译（round-trip translation）的检测框架，包含三个变体：TRIDENT-S（静态阈值）、TRIDENT-D（动态阈值）和TRIDENT-C（使用多距离RTT特征的逻辑回归）。TRIDENT利用原始文本与往返翻译文本之间的语义一致性来（原文摘要在此处截断）

    arXiv:2610.41232v1 Announce Type: cross  Abstract: As Large Language Models (LLMs) become increasingly multilingual, distinguishing human-written from AI-generated text is essential for maintaining information integrity. However, research remains English-centric, with limited benchmarks for low-resource languages such as Hindi. To address this gap, we introduce SHAKTI, a large-scale Hindi benchmark covering four genres, eight LLM families, multiple prompting strategies, and adversarial transformations. We evaluate its quality through human annotation and analyze linguistic, grammatical, and sentiment-based differences between human- and AI-generated texts. Building on these findings, we propose TRIDENT, a round-trip translation-based detection framework with three variants: TRIDENT-S (static threshold), TRIDENT-D (dynamic threshold), and TRIDENT-C (logistic regression using multi-distance RTT features). TRIDENT leverages semantic consistency between original and round-trip translated t
    
[^54]: 超越回答评分：面向自动数学简答题评分的AI感知认知监督框架

    Beyond Response Scoring: An AI-aware Cognitive Supervision Framework for Automated Mathematical Short Answer Grading

    [https://arxiv.org/abs/2610.41215](https://arxiv.org/abs/2610.41215)

    提出了一种双分支多任务框架AICG，将数学简答题评分与基于Q矩阵和G-DINA的认知诊断（技能掌握度与学习增益预测）相结合，并考虑生成式AI使用带来的评分偏差，以提升复杂数学答案自动评分的泛化能力。

    

    基于Transformer的语言模型在自动简答题评分（ASAG）中取得了显著成功。然而，当前的ASAG方法难以理解、推理和评估涉及符号和公式的复杂数学答案。这些仅输出分数的方法无法在具有相同分数的不同回答和技能状态之间进行泛化。此外，它们忽略了生成式人工智能（AI）的使用在评估学生表现和知识掌握程度时可能引入的向上偏差。为了解决这些局限性，我们提出了AICG（AI感知认知监督评分），这是一个双分支多任务框架。第一个分支结合了基于RoBERTa的回归、序数和分类头，用于对单个答案进行评分。第二个分支使用专家设计的Q矩阵将回答与所评估的技能相关联，并联合预测基于G-DINA的技能掌握度和学习增益。通过……（摘要在此处截断）

    arXiv:2610.41215v1 Announce Type: new  Abstract: Transformer-based language models have achieved remarkable success in automated short answer grading (ASAG). However, current methods in ASAG struggle to comprehend, reason about and evaluate complex mathematical answers involving symbols and formulas. These score-only methods fail to generalize across diverse responses and skill states associated with identical scores. Moreover, they ignore the potential upward bias introduced by generative Artificial Intelligence (AI) usage in assessing students' performance and knowledge mastery. To address these limitations, we propose AICG (AI-aware Cognitive-supervised Grading), a two-branch multi-task framework. The first branch combines RoBERTa-based regression, ordinal, and classification heads for individual-answer grading. The second branch uses an expert-designed Q-matrix to associate responses with assessed skills and jointly predicts G-DINA-derived skill mastery and learning gain. Through t
    
[^55]: TraceSpec：理解并改进面向大语言模型智能体的投机解码

    TraceSpec: Understanding and Improving Speculative Decoding for LLM Agents

    [https://arxiv.org/abs/2610.41177](https://arxiv.org/abs/2610.41177)

    提出TraceSpec框架，通过结合动态执行与固定前缀重放、构建跨浏览与软件工程任务的轨迹状态语料库，揭示了汇总投机解码指标掩盖的分阶段性能差异，并实现基于轨迹条件的自适应草稿监督与预测预算分配。

    

    大语言模型智能体通过在不断演化的推理、动作与工具反馈历史之上反复调用模型来执行任务。汇总式的投机解码指标将这些不同的决策合并在一起，对于在何处需要草稿监督所提供的指导十分有限。我们提出了TraceSpec，这是一个将轨迹评估与自适应草稿训练相连接的框架。它将动态执行与固定前缀重放相结合，并引入了TraceSpec-Traj——一个包含10,000个训练状态和2,500个测试状态的语料库，涵盖浏览与软件工程两类任务。对五个投机解码基线的分析表明，汇总接受率可能随上下文长度增加而上升，却在匹配的后期浏览动作中出现下降。工作负载与阶段进一步区分了这些趋势，这启发了由轨迹条件引导的监督方式。TraceSpec在中间的助手轮次上进行训练，并利用基于状态的、前缀感知的接受率估计来分配固定的预测预算。

    arXiv:2610.41177v1 Announce Type: new  Abstract: LLM agents execute tasks through repeated model calls over evolving histories of reasoning, actions, and tool feedback. Aggregate speculative-decoding metrics merge these distinct decisions, providing limited guidance on where draft supervision is needed. We present TraceSpec, a framework connecting trajectory evaluation with adaptive draft training. It combines dynamic execution with fixed-prefix replay and introduces TraceSpec-Traj, a corpus of 10,000 training states and 2,500 test states spanning browsing and software engineering. Analysis of five speculative baselines reveals that aggregate acceptance can rise with context length while declining within matched late browsing actions. Workload and phase further distinguish these trends, motivating supervision guided by trajectory conditions. TraceSpec trains on intermediate assistant turns and allocates a fixed prediction budget using state-conditioned estimates of prefix-aware accepta
    
[^56]: 上下文即基础设施：带执行框架的智能体系统中用于检索的知识编译

    Context as Infrastructure: Knowledge Compilation for Retrieval in Agentic Systems with Harnesses

    [https://arxiv.org/abs/2610.41172](https://arxiv.org/abs/2610.41172)

    提出“知识编译”应作为智能体检索的上下文基础设施：在搜索执行框架已基本解决准确性问题的前提下，通过可配置的摄取流水线为文档注入导航树、实体图、时间线等结构化上下文，充当解决上下文浪费问题的“供给层”，而非充当更准确的检索器。

    

    检索增强生成正在走向智能体化。一个搜索执行框架会撰写查询、阅读结果、重新表述查询，并决定何时停止。这在很大程度上解决了准确性问题：在我们的实验中，仅配备 BM25/grep 与 MiniMax M3 的执行框架，在不使用编译索引或向量索引的情况下，即可在 BrowseComp-Plus 上达到 85% 的准确率，在多跳 FRAMES 上达到 91%。但这同时暴露出一个上下文问题：不受约束的循环会重复提交查询、反复阅读文档，并用未被使用的段落填满上下文窗口。缺失的一层是知识编译：一条可配置的文档摄取流水线，由大语言模型为文档注入任何单个文本块都无法独立包含的上下文——带有论断-证据叶节点的导航层次结构、实体图、时间线、思维导图、本体、维基和记忆。编译并不是一个更好的检索器：在一个称职的执行框架下，结构化工件在 FRAMES 上没有带来可检测的准确性提升，而 PageIndex 式大纲在规模扩大时反而出现退化。它是一个供给层。一棵导航树，其叶节点……

    arXiv:2610.41172v1 Announce Type: cross  Abstract: Retrieval-augmented generation is becoming agentic. A search harness writes queries, reads results, reformulates, and decides when to stop. This largely solves accuracy: a harness with only BM25/grep and MiniMax M3 answers 85% of BrowseComp-Plus and 91% on multi-hop FRAMES in our runs, without compiled or vector indexes. It exposes a context problem: an undisciplined loop repeats queries, rereads documents, and fills the window with unused passages. The missing layer is knowledge compilation: a configurable ingestion pipeline where LLMs enrich documents with context no chunk contains alone--navigation hierarchies with claim-evidence leaves, entity graphs, timelines, mind maps, ontologies, wikis, and memory. Compilation is not a better retriever: under a competent harness, structural artifacts gave no detectable accuracy gain on FRAMES, and a PageIndex-style outline degraded at scale. It is a supply layer. A navigation tree whose leaves
    
[^57]: 谱系即记忆：短预算LLM引导优化中的父代继承与反馈披露

    Lineage Is Memory: Parent Inheritance and Feedback Disclosure in Short-Budget LLM-Guided Optimization

    [https://arxiv.org/abs/2610.41157](https://arxiv.org/abs/2610.41157)

    该研究通过1,750次预注册实验首次将父代继承与反馈披露效应分离，发现贪婪的父代继承——即谱系本身充当记忆——才是短预算LLM引导优化中绝大部分进展的真正来源，而非每轮反馈所覆盖的失败模式广度。

    

    迭代式语言模型优化器通过评估候选方案、向提议模型展示失败案例样本、并保留能提升性能的修订来改进提示词和程序。近期研究将这一循环的局限归因于狭窄的单轮反思，为此引入了全集错误聚类、针对性小批量或更大的提议模型，以便优化器在每一步能看到更多失败模式。然而，现有的所有评估都仅在保持谱系的搜索中测试这些机制——即每次修订都编辑一个存活的父代——这使得父代继承的贡献与每轮反馈的广度混淆在一起。在涵盖九个提示词与程序合成基准的1,750次预注册实验中，我们在匹配的评估预算下将父代继承与反馈披露分离开来。研究发现贪婪的父代继承推动了绝大多数优化进展，其表现优于种子重置……（原文摘要在此处截断）

    arXiv:2610.41157v1 Announce Type: cross  Abstract: Iterative language-model optimizers refine prompts and programs by evaluating a candidate, showing a proposer model a sample of failing cases, and keeping revisions that improve performance. Recent work attributes the limits of this loop to narrow single-round reflection, introducing full-set error clustering, targeted minibatches, or larger proposer models so the optimizer sees more failure modes per step. Yet every existing evaluation tests these mechanisms exclusively inside lineage-preserving search, where each revision edits a surviving parent, leaving the contribution of parent inheritance confounded with the breadth of per-round feedback. Across 1,750 pre-registered runs spanning nine prompt and program synthesis benchmarks, we separate parent inheritance from feedback disclosure under matched evaluation budgets. We find that greedy parent inheritance drives the vast majority of optimization progress, outperforming seed-reset re
    
[^58]: HyperRepair：恢复微调语言模型被遗忘的通用能力

    HyperRepair: Recovering Forgotten General Capabilities of Fine-Tuned Language Models

    [https://arxiv.org/abs/2610.41147](https://arxiv.org/abs/2610.41147)

    HyperRepair提出了一种基于超网络的方法，通过利用激活指纹为共享预训练主干的模型生成定制的修复侧网络，在仅增加推理成本的前提下恢复微调语言模型被遗忘的通用能力，同时保留其领域专业化，且无需目标领域样本或额外优化。

    

    微调可以提升语言模型在目标领域的性能，但可能削弱其通用能力。现有方法通常面临通用能力与领域专业化之间的权衡，难以同时保留两者。我们提出了HyperRepair，这是一个仅在推理成本下恢复通用能力、同时保留领域专业化的框架。为了使恢复过程能够泛化到共享同一预训练主干的不同模型上，HyperRepair从在源模型上训练的修复方案中学习一个超网络。该超网络使用概括了从预训练主干到微调后模型变化的激活指纹，生成定制的修复侧网络（RSN），其基于提示条件化的门控机制选择性地应用修正。这使得模型无需目标领域样本或额外优化，即可在来自未见过的适应领域的模型上实现能力恢复。我们在两个模型家族、12个LoRA适应数据集和15个通用能力上评估了HyperRepair。

    arXiv:2610.41147v1 Announce Type: new  Abstract: Fine-tuning improves a language model's performance in a target domain but can weaken its general capabilities. Existing methods often face a trade-off between general capabilities and domain specialization, making it difficult to preserve both. We introduce HyperRepair, a framework for recovering general capabilities while retaining domain specialization at only inference cost. To make recovery generalizable across models sharing a pretrained backbone, HyperRepair learns a hypernetwork from repairs trained on source models. The hypernetwork uses activation fingerprints that summarize changes from the pretrained backbone to generate a tailored Repairing Side Net (RSN), whose prompt-conditioned gate applies corrections selectively. This enables recovery on models from unseen adaptation domains without target-domain examples or additional optimization. We evaluate HyperRpair across two model families, 12 LoRA adaptation datasets, and 15 ge
    
[^59]: 语言模型在归因责备中表现出“女性很美好”效应，且不仅仅是宽容

    Language Models Show a "Women-Are-Wonderful" Effect in Blame Attribution That is Not Just Leniency

    [https://arxiv.org/abs/2610.41116](https://arxiv.org/abs/2610.41116)

    该研究发现语言模型在人际冲突的责备归因中存在角色依赖的“女性很美好”效应：将受影响方设为女性会使被责备几率增加 2.7-9.2 倍，而将被评判方设为女性则使被责备几率降低 9%-52%。

    

    本文研究了语言模型在归因责备中的“女性很美好”效应：这是一种有利偏见，即女性比男性获得更多的积极评价。我们利用来自 r/AmItheAsshole 的 11,270 个人际冲突案例，构建了匹配的反事实样本，改变被评判者（发帖人）和受影响者（对方）的性别。在四个语言模型中，性别改变了一小部分判定结果，但这些变化一致地偏向女性。该效应强烈依赖于角色：将发帖人设为女性会使被责备的几率降低 9%-52%，而将对方设为女性会使该几率增加 2.7-9.2 倍。对方的 WaW 效应也因冲突类型而异，在沟通和关系冲突中增强，在家庭冲突中减弱。探索性探测表明，对方的性别可以改变模型对同情心、合理性、伤害和保护等方面的判断。

    arXiv:2610.41116v1 Announce Type: new  Abstract: This paper investigates the Women-Are-Wonderful (WaW) effect in language-model blame attribution: a favorable bias in which women receive more positive evaluations than men. Using 11,270 interpersonal conflicts from r/AmItheAsshole, we construct matched counterfactuals that vary the gender of the person being judged (the original poster) and the person affected (the counter-party). Across four language models, gender changes a minority of verdicts, but those changes consistently favor women. The effect is strongly role-dependent. Making the original poster a woman reduces the odds of blame by 9--52%, while making the counter-party a woman multiplies them by 2.7--9.2X. Counter-party WaW also varies across conflict types, strengthening in communication and relationship conflicts and weakening in family conflicts. Exploratory probes suggest that counter-party gender can shift judgments of sympathy, reasonableness, harm and protection. WaW i
    
[^60]: MUON-RT：重用教师轨迹实现无动量Muon的大语言模型自蒸馏

    MUON-RT Reusing the Teacher's Trajectory for Momentum-Free Muon for LLM Self-Distillation

    [https://arxiv.org/abs/2610.41104](https://arxiv.org/abs/2610.41104)

    提出Muon-RT，通过重用自蒸馏中教师与学生的位移替代Muon优化器的动量缓冲区，在降低训练内存开销的同时保证了理论收敛性，并提升大语言模型自蒸馏微调的性能。

    

    对于采用指数移动平均（EMA）教师的大语言模型自蒸馏方法而言，学生更新的历史记录本身已被保留。然而，Muon优化器仍需存储一个单独的动量缓冲区，从而增加了持续的训练内存开销。我们提出了重用教师的Muon（Muon-RT），该方法通过重用教师-学生位移来消除隐藏层权重矩阵的动量缓冲区。我们证明，该位移编码了已应用更新的指数加权历史，并在局部模型下平滑了时间上不相关的噪声。Muon-RT去除该位移的径向分量，对其进行谱归一化，并在Newton-Schulz（NS）正交化之后将其与当前梯度混合。在明确的正则性与对齐性假设下，我们建立了期望目标差距界向残差误差水平的线性收缩。通过在Qwen2.5-7B-Instruct上进行自蒸馏微调（SDFT），Muon-RT相比AdamW等基线方法提升了平均准确率（摘要原文在此处截断）。

    arXiv:2610.41104v1 Announce Type: cross  Abstract: Self-distillation methods for large language models with an exponential moving average (EMA) teacher already retain a history of student updates. Muon nevertheless stores a separate momentum buffer, increasing persistent training memory. We propose Muon with Reused Teacher (Muon-RT), which reuses the teacher-student displacement to eliminate momentum buffers for hidden weight matrices. We show that this displacement encodes an exponentially weighted history of applied updates and smooths temporally uncorrelated noise under a local model. Muon-RT removes its radial component, normalizes it spectrally, and mixes it with the current gradient after Newton-Schulz (NS) orthogonalization. Under explicit regularity and alignment assumptions, we establish linear contraction of an expected objective-gap bound toward a residual error level. With Self-Distillation Fine-Tuning (SDFT) on Qwen2.5-7B-Instruct, Muon-RT improves average accuracy over Ad
    
[^61]: EDU-Jev 训练系统——面向教育的“系统一”决策模型

    EDU-Jev Training System-One Decision Models for Education

    [https://arxiv.org/abs/2610.41079](https://arxiv.org/abs/2610.41079)

    该论文提出 EDU-Jev 系列面向教育评估任务的单次前向决策模型（0.8B/2B/4B 参数），通过指针头直接输出预定义选项上的概率分布而无需生成文本，实现了更快、更省、更可靠的教育自动化决策。

    

    教育领域的许多自动化决策都涉及从固定的选项集合中进行选择，例如评定量规分数、判断某个答案是否满足评分标准，或找出解题过程中第一个出错的步骤。生成式语言模型正越来越多地被用于做出这类决策，但其响应需要解析，更长的输出会增加推理成本，而且用文字表达的置信度并不一定反映真实的正确概率。我们提出了 EDU-Jev，这是一系列参数规模为 0.8B、2B 和 4B 的单次前向（“系统一”）决策模型。给定一个评估情境和一道带有预定义答案选项的题目，模型直接返回这些选项上的概率分布，而无需生成文本。一个指针头为每个选项对应的标记打分，确保预测被限制在允许的选项范围内。训练采用单一监督阶段，使用软交叉熵损失针对可用的目标标签分布进行优化。

    arXiv:2610.41079v1 Announce Type: new  Abstract: Many automated decisions in education involve choosing from a fixed set of options: assigning a rubric score, checking whether an answer meets a marking criterion, or identifying the first wrong step in a solution. Generative language models are increasingly used to make these decisions, but their responses require parsing, longer outputs increase inference costs, and confidence expressed in words does not necessarily reflect the probability of being correct. We present EDU-Jev, a family of single-pass ("System One") decision models at 0.8B, 2B and 4B parameters. Given an assessment context and a question with predefined answer options, the model returns a probability distribution over those options without generating text. A pointer head scores a marker for each option, ensuring the predictions are restricted to the allowed choices. Training uses a single supervised stage with soft cross-entropy against the available target label distri
    
[^62]: 基于能力接地的智能体规划

    Capability-Grounded Agentic Planning

    [https://arxiv.org/abs/2610.41040](https://arxiv.org/abs/2610.41040)

    CGAP 通过将工具目录确定性编译为抽象 PDDL 模型并使用经典规划器求解，使 LLM 智能体生成的计划天然可执行，同时在执行时由 LLM 补充实体系调用并通过确定性检查确保不破坏原计划。

    

    大语言模型（LLM）智能体通过工具、数据源和其他智能体来服务自然语言请求，即使是简单的请求也常常需要多个相互依赖的调用。通常情况下，LLM 通过反复试错来选择这些调用，无法保证得到一个可执行的计划。我们提出了基于能力接地的智能体规划（Capability-Grounded Agentic Planning，CGAP），它转而推理请求需要什么以及工具能提供什么。CGAP 将工具目录确定性地编译为 PDDL 模型，使用 LLM 将请求转化为目标，并用经典规划器求解该模型。由于工具输出事先未知，该模型是抽象的，只跟踪哪些输入可用，而不跟踪其具体值。在执行过程中，LLM 会添加抽象计划所省略的、与具体实体相关的调用，而确定性检查会拒绝任何破坏计划的调用。每个参数都取自被引用的来源，例如对话片段或先前的工具响应，且每个被请求的结果都被计为已交付。

    arXiv:2610.41040v1 Announce Type: cross  Abstract: LLM agents serve natural-language requests through tools, data sources, and other agents, and even simple requests often need several interdependent calls. Typically, the LLM picks these calls by trial and error, with no guarantee of an executable plan. We present Capability-Grounded Agentic Planning (CGAP), which instead reasons about what a request requires and what tools provide. CGAP compiles the tool catalog deterministically into a PDDL model, uses an LLM to turn the request into goals, and solves the model with a classical planner. Since tool outputs are unknown in advance, the model is abstract, tracking which inputs are available, not their values. During execution, an LLM adds entity-specific calls that abstract plans omit, and deterministic checks reject any that break the plan. Every argument is drawn from a cited source, such as a conversation span or an earlier tool response, and each requested result counts as delivered 
    
[^63]: 语言模型知道什么样的科学证据是有说服力的吗？

    Do Language Models Know What Compelling Scientific Evidence Looks Like?

    [https://arxiv.org/abs/2610.41015](https://arxiv.org/abs/2610.41015)

    论文提出了基于真实生物学数据的科学推理基准EDB-Bio，发现现有语言模型在权衡科学证据时存在系统性信心不足，表现仅与简单基率预测器相当，而针对严格恰当评分规则的后训练可显著改善模型的校准与判别能力。

    

    专家科学家擅长权衡来自多个来源的经验性发现，而这些来源在统计可靠性和认识论相关性上各不相同。日益驱动自主科学发现系统的语言模型，在多大程度上具备这种证据审议能力？我们提出了EDB-Bio，这是一个基于三个生物学领域的真实世界数据构建的科学推理基准：药物扰动转录组学、CRISPR筛选和人类全基因组关联研究（GWAS）。EDB-Bio的96,000个回合中，每个回合都呈现一系列来自已发表研究的实验结果，并要求模型预测目标假设为真的概率。在未经调整的情况下，无论是开源权重模型还是前沿模型，都表现出系统性的信心不足，其Brier评分与一个仅依赖基率的简单预测器相当。而针对严格恰当评分规则进行的后训练能够显著提升模型的校准能力和判别能力。

    arXiv:2610.41015v1 Announce Type: cross  Abstract: Expert scientists excel at weighing empirical findings from multiple sources that vary in both statistical reliability and epistemic relevance. To what extent are language models, which are increasingly driving autonomous discovery systems, capable of such evidence deliberation? We introduce EDB-Bio, a scientific reasoning benchmark curated from real-world data in three biological domains: drug-perturbation transcriptomics, CRISPR screens, and human genome-wide association studies (GWAS). Each of EDB-Bio's 96k episodes presents a sequence of experimental results from published studies and asks the model to predict the probability that a target hypothesis is true. Out-of-the-box, open-weight and frontier models alike demonstrate systematic underconfidence and attain Brier scores comparable to a trivial base-rate forecaster. Post-training against strictly proper scoring rules significantly improves both calibration and discrimination. Th
    
[^64]: 语言模型在检索增强生成（RAG）中表现出从众偏差：当多数文档支持被投毒的论断时

    Language Models Show Conformity Bias in RAG: When the Majority Endorses a Poisoned Claim

    [https://arxiv.org/abs/2610.41012](https://arxiv.org/abs/2610.41012)

    本研究发现当RAG检索上下文中多数相互一致的投毒文档共同支持错误信息时，语言模型的投毒检测率从25%显著降至10%，揭示了语言模型存在类似社会从众心理的从众偏差，使其更容易被多数派的投毒信息误导。

    

    检索增强生成（RAG）可能使语言模型接触到检索文档中具有误导性的证据。我们研究了相互一致的投毒文档数量增加是否会影响模型检测错误信息的能力。我们从SciFact数据集中随机选取200个问题（随机种子38）进行评估，将包含一篇投毒文档和六篇健康文档的上下文，与包含六篇相互一致的投毒文档和一篇健康文档的上下文进行对比。对于Gemma-3-12b模型，投毒检测率从少数条件下的25%（50/200）下降至多数条件下的10%（20/200）。在两种条件下结果不同的38个问题中，有34个从“被检测出”变为“未被检测出”，仅有4个朝相反方向变化（精确McNemar检验，p = 6.04e-7）。这一模式与语言模型面对多数相互一致的投毒文档时产生的类从众式反应一致。我们的结果提供了初步证据……

    arXiv:2610.41012v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can expose language models to misleading evidence in retrieved documents. We investigate whether increasing the number of mutually agreeing poisoned documents affects a model's ability to detect misinformation. We evaluate 200 randomly selected questions from SciFact (random seed 38), comparing contexts with one poisoned and six healthy documents against contexts with six mutually agreeing poisoned documents and one healthy document. For Gemma-3-12b, poison-detection rates decrease from 25% (50/200) in the minority condition to 10% (20/200) in the majority condition. Among the 38 questions with different outcomes across conditions, 34 change from detected to undetected, while only four change in the opposite direction (exact McNemar test, p = 6.04e-7). This pattern is consistent with a conformity-like response to a majority of mutually agreeing poisoned documents. Our results provide initial evidence 
    
[^65]: MOSAIC：通过模态划分的梯度路由解决任务与模态间的非对称依赖问题

    MOSAIC: Addressing Disproportionate Task-Modality Dependencies via Modality-Partitioned Gradient Routing

    [https://arxiv.org/abs/2610.41004](https://arxiv.org/abs/2610.41004)

    提出MOSAIC架构，通过模态分离与模态划分的梯度路由显式建模多任务学习中各任务对特定模态的非对称依赖，在保留单模态表征的同时避免跨模态的梯度干扰。

    

    在多任务设置中，融合多种模态可以让任务表征利用单一模态无法捕捉的信号。然而，各个任务可能不成比例地依赖于某些特定模态，而传统的基于融合的架构并未显式地对这种结构进行建模，而是强制采用统一的表示空间。因此，一个任务的梯度可能会影响其几乎不依赖的模态所关联的参数，导致特征在不同需求之间相互妥协。尽管先前的工作已经探索了任务加权和损失平衡等策略，但由于缺乏显式的模态分离，其有效性受到限制，使模型无法根据各模态对特定任务的重要性进行优化。为了解决这些局限，我们提出了MOSAIC（具有模态间交叉注意力的模态分离架构），它在保留单模态表征的同时，实现了受控的……

    arXiv:2610.41004v1 Announce Type: cross  Abstract: In multi-task settings, combining multiple modalities allows task representations to draw on signals that a single modality alone would not capture. However, individual tasks may depend disproportionately on particular modalities, a structure that conventional fusion-based architectures do not model explicitly, instead enforcing a unified representation space. Consequently, a task's gradients can affect parameters tied to modalities it barely relies on, leaving features that compromise between competing demands. Although prior work has explored task-weighting and loss-balancing strategies, their effectiveness is limited by the absence of explicit modality separation, preventing models from optimising each modality according to its task-specific importance. To address these limitations, we propose MOSAIC (MOdality-Separated Architecture with Inter-modal Cross-attention), which preserves unimodal representations while enabling controlled
    
[^66]: 开放语言模型会编造经文吗？BILAL-BENCH：面向孟加拉语伊斯兰文献来源伪造的基准测试

    Do Open Language Models Invent Scripture? BILAL-BENCH: Benchmarking Islamic-Source Fabrication in Bengali

    [https://arxiv.org/abs/2610.40996](https://arxiv.org/abs/2610.40996)

    该论文提出BILAL-BENCH——首个针对孟加拉语开放式伊斯兰文献来源伪造的专家验证基准，评测发现七个开放权重模型中有六个在79.3%-95.3%的来源生成问题上自信地编造经文内容，且性能不随模型规模提升而改善。

    

    当语言模型编造《古兰经》经文或圣训时，它会赋予文本虚假的权威性，而读者可能将这些文本视为神圣的。现有的伊斯兰幻觉基准测试主要以阿拉伯语为主且多为封闭式问题，而孟加拉语资源主要研究一般领域的幻觉或事实性知识。我们提出了BILAL-BENCH，一个包含240个经专家验证问题的基准，用于评测孟加拉语中开放式的伊斯兰文献来源伪造现象。该基准涵盖《古兰经》引用归属、圣训、伊斯兰教法学、当代议题以及对抗性错误前提提示，并采用双轴评分标准，将来源错误与断言强度分开评估。我们在1,680个经过双重标注的响应上评测了七个参数量从1B到120B的开放权重模型。七个模型中有六个在79.3%至95.3%的来源生成问题上自信地编造内容，且性能并未随模型规模扩大而单调提升。圣训溯源是所有模型中最薄弱的类别：315个响应中仅有两个既内容正确又正确标注了来源。

    arXiv:2610.40996v1 Announce Type: new  Abstract: When a language model invents a Qur'anic verse or hadith, it gives false authority to text that readers may treat as sacred. Existing Islamic hallucination benchmarks are largely Arabic and often closed-ended, while Bengali resources mainly study general-domain hallucination or factual knowledge. We introduce BILAL-BENCH, a 240-question expert-verified benchmark for open-ended Islamic-source fabrication in Bengali. It covers Qur'anic attribution, hadith, fiqh, contemporary issues, and adversarial false-premise prompts, and uses a two-axis rubric that separates source errors from assertiveness. We evaluate seven open-weight models from 1B to 120B on 1,680 double-annotated responses. Six of the seven models confidently fabricate on 79.3-95.3% of source-production questions, with no monotonic improvement from scale. Hadith provenance is the weakest category across models: only two of 315 responses are correct and correctly sourced. The two 
    
[^67]: 大语言模型中情感的心理物理学

    The psychophysics of affect in large language models

    [https://arxiv.org/abs/2610.40985](https://arxiv.org/abs/2610.40985)

    本研究首次系统测量了大语言模型将结果映射为情感的心理物理规律，发现模型对收益的情感反应强于等额损失（与人类损失厌恶相反的“硅基收益放大”现象），且这种结果到情感的映射会随评分情境和决策任务而显著改变，具有高度可塑性。

    

    良好的决策往往需要考虑结果将会带来的感受。然而，大语言模型（LLM）正日益进入决策工作流程，但人们对它们如何将结果映射为感受、以及如何在选择中使用这些映射却知之甚少。我们测试了大语言模型如何预期赌博或股票交易结果带来的感受，以及它们是否将这些评分纳入选择决策。当结果被单独评分时，评分对量级表现出递减的敏感性，与人类的价值函数相似；但模型对收益的评分比对等额损失的评分更为强烈，这是对经典损失厌恶的一种反转，我们称之为“硅基收益放大”。当结果被一起评分、可以进行直接比较时，评分趋于线性化。在选择题任务中，收益与损失之间的不对称性基本消失，且模型自身的评分在解释其选择方面并不比货币效用更可靠。这些发现揭示了从结果到情感的可塑映射，并引发了相关问题……

    arXiv:2610.40985v1 Announce Type: cross  Abstract: Good decisions often require considering how outcomes will feel. Yet large language models (LLMs) increasingly enter decision workflows, despite little being known about how they map outcomes to feelings or use those mappings in choice. We tested how LLMs anticipated gambling or stock-trading outcomes would feel and whether they incorporated these ratings into choice. When outcomes were rated in isolation, ratings exhibited diminishing sensitivity to magnitude, resembling human value functions; but models rated gains more intensely than equivalent losses, an inversion of canonical loss aversion we call silicon gain amplification. When outcomes were rated together, enabling direct comparison, ratings linearised. In choice, the gain-loss asymmetry largely disappeared, and models' own ratings did not reliably explain their choices better than monetary utility did. These findings reveal malleable mappings from outcomes to affect, raising q
    
[^68]: 更少的注意力头，更少的记忆：剪枝冗余注意力头可抑制逐字记忆

    Fewer Heads, Fewer Memories: Pruning Redundant Attention Heads Suppresses Verbatim Memorisation

    [https://arxiv.org/abs/2610.40974](https://arxiv.org/abs/2610.40974)

    本研究发现在推理前将大量冗余注意力头置零，可以在保持甚至提升语言建模性能的同时，显著抑制大语言模型对训练数据的逐字记忆复现。

    

    大型语言模型已被证明能够记忆并逐字复现其训练数据中的序列，这引发了对隐私泄露、知识产权暴露以及泛化能力局限性的担忧。近期的理论研究表明，注意力头可能通过充当联想记忆机制而促成这种行为，其记忆容量会随注意力头数量的增加而扩展。与此同时，实证研究表明许多注意力头是冗余的，可以在对下游性能影响有限的情况下被剪枝。尽管有这些平行的发现，剪枝冗余注意力头对记忆序列复现的影响仍未被探索。在这项工作中，我们证明在推理之前将相当大比例的注意力头置零，可以显著抑制逐字记忆，同时在用于识别冗余注意力头的基准上保持甚至提升语言建模性能。

    arXiv:2610.40974v1 Announce Type: new  Abstract: Large language models have been shown to memorise and reproduce verbatim sequences from their training data, raising concerns about privacy leakage, intellectual property exposure, and the limits of generalisation. Recent theoretical work suggests that attention heads may contribute to this behaviour by functioning as associative memory mechanisms, with memorisation capacity scaling with the number of heads. At the same time, empirical work has shown that many attention heads are redundant and can be pruned with limited impact on downstream performance. Despite these parallel findings, the effect of pruning redundant heads on memorised-sequence reproduction remains unexplored. In this work, we demonstrate that zeroing out a substantial fraction of attention heads prior to inference can significantly suppress verbatim memorisation while preserving, or even improving, language modelling performance on the benchmark used to identify redunda
    
[^69]: Nedimo：一个基于证据的AI生成睡眠故事多语言流水线

    Nedimo: An Evidence-Grounded Multilingual Pipeline for AI-Generated Sleep Stories

    [https://arxiv.org/abs/2610.40970](https://arxiv.org/abs/2610.40970)

    该论文提出了Nedimo——一个将不确定的临床证据转化为保守工程需求的多语言流水线，通过将确定性规划与大语言模型生成分离、四阶段注意力渐减、结构和词汇约束验证以及有界错误感知重试等机制，来生成有助于入睡的低唤醒度AI睡眠故事。

    

    生成式音频系统可以产生大型的个性化睡前内容库，但普通的媒体目标（留存率、完播率和新鲜感）与支持用户过渡到睡眠的目标并不一致。关于成人睡前叙述音频的直接证据并不一致，且最近一项主动对照试验并不支持此类音频作为睡眠障碍的独立治疗方法。我们将这一不确定的证据转化为保守的工程需求，并提出一个用于生成和叙述低唤醒度睡眠故事的多语言流水线。已实现的原型将确定性规划与大语言模型生成分离，显式地表示内容和表达参数，通过四阶段注意力渐减来塑造每个脚本，验证结构和词汇约束，执行有界的错误感知重试，并对未解决的输出进行隔离。与供应商无关的文本转语音合成……（摘要原文在此处被截断）

    arXiv:2610.40970v1 Announce Type: cross  Abstract: Generative audio systems can produce large libraries of personalized bedtime content, but ordinary media objectives (retention, completion, and novelty) are poorly aligned with the goal of supporting a transition toward sleep. Direct evidence for narrated bedtime audio in adults is mixed, and a recent active-controlled trial did not support such audio as a standalone treatment for sleep disturbance. We translate this uncertain evidence into conservative engineering requirements and present a multilingual pipeline for generating and narrating low-arousal sleep stories. The implemented prototype separates deterministic planning from large language model generation, represents content and delivery parameters explicitly, shapes each script through a four-phase attention taper, validates structural and lexical constraints, performs a bounded error-aware retry, and quarantines unresolved outputs. Provider-agnostic text-to-speech synthesis is
    
[^70]: 更好的检索器，更差的标签：不完整的相关性判断如何扭曲对话式RAG中的弃答评估

    Better Retrievers, Worse Labels: How Incomplete Relevance Judgments Distort Abstention Evaluation in Conversational RAG

    [https://arxiv.org/abs/2610.40966](https://arxiv.org/abs/2610.40966)

    该论文揭示，在对话式RAG评估中，“无黄金文档即不可回答”的捷径会因相关性判断不完整而严重失真——检索器越强，被遗漏但实际含答案的上下文比例越高（28%至72%），导致该捷径的精确率从0.68骤降至0.10。

    

    诸如MTRAG和RETECO等对话式RAG基准端到端地评估检索与生成：每个系统基于自己的检索结果进行回答，因此弃答行为必须根据每个检索到的上下文是否充分来评分。一种方便的捷径是直接从相关性判断中读取充分性，并将没有黄金文档的上下文视为无法回答。我们精确地重新运行了官方基线，并在RECOR（RETECO赛道2）上使用四个检索器（nDCG@10从0.18到0.59）和三个Qwen3阅读器测试了这一捷径。两个开源LLM评判器和一组由四个小型开源模型组成的评审团根据内容为每个上下文打标签，并由另一系列模型对250个上下文进行盲审与裁决。结果显示：检索器越强，没有黄金文档但实际仍包含答案的上下文比例越大（从28%到72%），而这种捷径的精确率越低（从0.68降至0.10；在盲审下为0.70到0.21）。阅读器模型也印证了这一点：Qwen3-4B能够正确回答这些“假性未命中”上下文中的62%至67%。

    arXiv:2610.40966v1 Announce Type: new  Abstract: Conversational RAG benchmarks such as MTRAG and RETECO evaluate retrieval and generation end to end: each system answers from its own retrieval, so abstention must be scored against whether each retrieved context is sufficient. A convenient shortcut reads sufficiency off the relevance judgments and treats a context without gold documents as unanswerable. We re-run the official baselines exactly and test this shortcut on RECOR (RETECO track~2) with four retrievers (nDCG@10 from .18 to .59) and three Qwen3 readers. Two open LLM judges and a panel of four small open models label each context by content, and a model of another family audits 250 contexts blind, with adjudication. The stronger the retriever, the larger the share of no-gold contexts that still contain the answer (28\% to 72\%) and the lower the shortcut's precision (.68 to .10; .70 to .21 under the audit). A reader agrees: Qwen3-4B answers 62--67\% of these false misses correct
    
[^71]: 基于核的插值方法用于注意力机制的上下文压缩

    Kernel-based Interpolatory Methods for Context Compression of Attention Mechanisms

    [https://arxiv.org/abs/2610.40943](https://arxiv.org/abs/2610.40943)

    本文在核所对应的希尔伯特空间中将注意力表示为两个线性映射之比，并分别基于广义经验插值方法（GEIM）和Q-离散经验插值方法（Q-DEIM）提出了两种新颖的KV缓存压缩方法GAIM和Q-DAIM。

    

    注意力机制是现代语言模型的基本构建模块，其瓶颈在于相对于上下文长度的二次方复杂度。因此，上下文压缩（也称为KV缓存压缩）引起了越来越多的关注。在本文中，我们考虑注意力机制上下文压缩的问题，首先在与注意力机制相关联的核所对应的希尔伯特空间中对其进行形式化。在这一形式化框架下，注意力可以表示为两个线性映射之比，而其插值同样是两个插值之比，这使得我们可以直接对线性映射进行插值。据此，我们基于广义经验插值方法（GEIM）和Q-离散经验插值方法（Q-DEIM），分别提出了两种新颖的KV缓存压缩方法，称为GAIM和Q-DAIM。由于这些方法的计算代价与上下文长度呈二次方关系，我们……

    arXiv:2610.40943v1 Announce Type: cross  Abstract: The bottleneck of attention mechanisms, which are the fundamental building block of modern language models, is their quadratic complexity in terms of context size. There has thus been an increasing interest in the compression of the context, also called KV-cache compression. In this article, we consider the problem of compressing the context of attention mechanisms by first formulating it in the Hilbert space associated with its kernel. In this formalism, attention can be represented as the ratio of two linear maps whose interpolant is also the ratio of interpolants, this allows us to consider the interpolation of the linear maps directly. We thus introduce two novel KV-cache compression methods which we call GAIM and Q-DAIM, based on the Generalized Empirical Interpolation Method (GEIM) and the Q-Discrete Empirical Interpolation Method (Q-DEIM), respectively. As the cost of these methods is quadratic in terms of context length, we pro
    
[^72]: 偏好认知不确定性下的自适应个性化

    Adaptive Personalization Under Preference Epistemic Uncertainty

    [https://arxiv.org/abs/2610.40930](https://arxiv.org/abs/2610.40930)

    本文提出并形式化了“偏好认知不确定性下的自适应个性化”任务，构建了一个涵盖250个人设、5种语言、25K轮对话的多语言基准，揭示了现有大模型普遍存在的“识别-生成鸿沟”——即模型能从候选选项中正确识别偏好适应策略，却在开放式生成中失效。

    

    人类偏好会随时间变化：一个既定的偏好可能长期保持有效，可能随着情境变化而变得不确定，也可能被明确地修订。有效的个性化要求对话式AI能够根据不断演变的偏好调整其回应策略，而不是依赖被动的记忆检索。我们将这种能力形式化为“偏好认知不确定性下的自适应个性化”。一项人类偏好研究进一步证实，随着偏好不确定性的变化，人们更倾向于采取依赖状态的自适应回应策略。为了评估语言模型是否也具备这种适应能力，我们构建了一个多语言基准，涵盖250个人设、5种语言和25K轮对话，该基准将策略选择与回应执行解耦。在八个闭源与开源权重模型上，我们发现了一种“识别-生成鸿沟”：模型在候选选项中进行选择时能够可靠地识别出合适的策略，但在开放式生成时表现崩溃。（原文摘要在此处截断）

    arXiv:2610.40930v1 Announce Type: new  Abstract: Human preferences change over time: an established preference may remain valid over time, become uncertain as context shifts, or be explicitly revised. Effective personalization requires conversational AI to adapt its response strategy to evolving preferences rather than relying on passive memory retrieval. We formalize this capability as adaptive personalization under epistemic uncertainty. A human preference study further confirms that people favor state-contingent response strategies as preference uncertainty changes. To evaluate whether language models mirror this adaptation, we create a multilingual benchmark spanning 250 personas, 5 languages, and 25K conversational turns that decouples strategy selection from response execution. Across eight closed- and open-weight models, we uncover a recognition-generation gap: models reliably identify appropriate strategies when selecting among candidate options, but collapse during open-ended 
    
[^73]: 无需环境副本的智能体式世界建模，用于智能体测试时扩展

    Agentic World Modeling for Agent Test-time Scaling without Environment Replica

    [https://arxiv.org/abs/2610.40927](https://arxiv.org/abs/2610.40927)

    提出智能体式世界建模（AWM），将下一观测预测重构为多轮证据获取任务，通过安全只读接口自适应查询真实环境来模拟候选动作，从而无需复制环境即可实现大语言模型智能体的测试时扩展。

    

    测试时扩展通过在多条候选轨迹之间进行采样与选择来提升大型语言模型智能体的表现，但常见的实现方式假设这些轨迹可以被并行执行。这一假设在现实部署中并不成立：企业工作空间和云服务是共享且持久存在的，对其进行复制的代价高昂甚至不可行，而且动作可能带来不可逆的副作用或跨轨迹的相互干扰。现有的生成式世界模型虽然避免了对环境的直接执行，但通常将环境状态压缩到单一提示中，限制了其在部分可观测条件下可靠模拟状态转移的能力。为此，我们提出了智能体式世界建模，将下一观测预测重新表述为一个多轮证据获取任务。针对每个假设动作，一个智能体化的模拟器通过安全的只读接口自适应地查询真实环境，并对返回的

    arXiv:2610.40927v1 Announce Type: cross  Abstract: Test-time scaling (TTS) improves large language model agents by sampling and selecting among multiple candidate trajectories, but common realizations assume that these trajectories can be executed in parallel. This assumption breaks in realistic deployments, where enterprise workspaces and cloud services are shared and persistent, costly or infeasible to replicate, and actions may introduce irreversible side effects or cross-trajectory interference. Existing generative world models avoid direct execution, but typically compress environment state into a single prompt, limiting their ability to reliably simulate transitions under partial observability. Instead, we introduce Agentic World Modeling (AWM), which reformulates next-observation prediction as a multi-turn evidence-acquisition task. For each hypothetical action, an agentic simulator adaptively queries the real environment through safe read-only interfaces, reasons over the retur
    
[^74]: CoFiBreak：一种用于有效多轮越狱的由粗到细多样性奖励

    CoFiBreak: A Coarse-to-Fine Diversity Reward for Effective Multi-Turn Jailbreak

    [https://arxiv.org/abs/2610.40926](https://arxiv.org/abs/2610.40926)

    提出CoFiBreak，一种面向GRPO的由粗到细多样性奖励机制，在保持高攻击效果与低推理成本的同时，显著提升了多轮越狱攻击的策略与实现多样性。

    

    arXiv:2610.40926v1 公告类型：新论文。摘要：自动化红队测试通过训练一个攻击者大语言模型（LLM）来诱导受害者大语言模型产生有害回应，但当攻击成功率是唯一奖励时，强化学习会使攻击者模型坍缩到狭窄的一组提示词上。这种攻击多样性的坍缩会使安全评估偏向于反复测试近乎重复的攻击模式，而非测试底层的有害意图。现有方法通常以牺牲攻击成功率或推理成本为代价来换取更大的内容或策略多样性。我们提出了CoFiBreak，一种面向组相对策略优化（GRPO）的由粗到细多样性奖励，它在保持强大攻击效果和低推理成本的同时提升了攻击多样性。粗粒度奖励通过鼓励每个GRPO组内代表性不足的策略来拓宽攻击者所使用的策略范围，而细粒度奖励通过阻止重复使用相同的人物角色、叙述框架等方式，拓宽了这些策略的具体实现方式。

    arXiv:2610.40926v1 Announce Type: new  Abstract: Automated red teaming trains an attacker large language model (LLM) to elicit harmful responses from a victim LLM, but reinforcement learning can collapse attackers onto a narrow set of prompts when attack success is the only reward. Such a collapse of attack diversity can bias safety evaluation toward repeatedly testing near-duplicate attack patterns rather than the underlying harmful intents. Existing methods often trade off attack success or inference cost for greater content or strategy diversity. We propose CoFiBreak, a coarse-to-fine diversity reward for Group Relative Policy Optimization (GRPO) that improves attack diversity while preserving strong attack effectiveness and low inference cost. The coarse reward broadens which strategies the attacker uses by encouraging strategies underrepresented within each GRPO group, while the fine reward broadens how those strategies are realized by discouraging reuse of the same personas, fram
    
[^75]: Qistas：一种用于大语言模型公平性评估的预注册、可移植协议

    Qistas: A Pre-Registered, Portable Protocol for Fairness Evaluation of Large Language Models

    [https://arxiv.org/abs/2610.40893](https://arxiv.org/abs/2610.40893)

    提出了Qistas——一个预注册、可跨国移植的大语言模型公平性评估协议，通过配对反事实设计与严格的统计检验来衡量模型对一个社会不同社会、教派和地区群体处理的程序公平性，并首先在伊拉克落地实施。

    

    大语言模型（LLM）的公平性基准测试绝大多数诞生于英语世界，其社会类别带有浓厚的美国色彩；翻译版本会将这些类别强行带入并不适用的社会，而且几乎没有基准在运行模型之前预先固定其假设。我们提出了Qistas（阿拉伯语“قistas”，意为“公正的天平”），这是一个评估大语言模型如何对待一个社会的社会、教派和地区群体的协议，可跨国移植，并首先在伊拉克落地实施。在其试点形式中，它仅测量那些无需对争议性真相作出裁决的内容：匹配反事实对之间的程序对称性、差异性拒绝、模糊情境下的刻板印象依赖，以及对有记录事件的术语框定。每一项测量在构造上都是配对的，并使用配对分析工具进行分析（符号翻转置换检验、精确McNemar检验、模板聚类自助法置信区间、经Holm校正的检验族），同时进行统计功效分析……

    arXiv:2610.40893v1 Announce Type: new  Abstract: Fairness benchmarks for large language models (LLMs) are overwhelmingly English-born and American in their social categories; translated versions carry those categories into societies where they do not apply, and almost none fix their hypotheses before the models are run. We present Qistas (Arabic qistas, "the just balance"), a protocol for evaluating how LLMs treat the social, confessional, and regional constituencies of a society, portable across countries and instantiated first for Iraq. In its pilot form it measures only what needs no verdict on contested truth: procedural symmetry across matched counterfactual pairs, differential refusal, stereotype reliance under ambiguity, and terminological framing of documented events. Every measurement is paired by construction and analysed with paired machinery (sign-flip permutation tests, exact McNemar tests, template-clustered bootstrap intervals, Holm-corrected families), with a power anal
    
[^76]: 音频大语言模型中多模态语音情感识别的不一致性感知评估

    Incongruence-Aware Evaluation of Multimodal Speech Emotion Recognition in Audio Large Language Models

    [https://arxiv.org/abs/2610.40883](https://arxiv.org/abs/2610.40883)

    该论文提出一种基于双分量高斯混合模型、将文本与音频在情感空间中的距离转化为连续无阈值得分的不一致性度量方法，并发现文本与音频情感表达的不一致会在所有语料库中系统性增加音频大语言模型语音情感识别的错误率，而监督分类器则表现出相反的行为。

    

    语音交互大语言模型同时处理说话者所说的内容（文本）以及说话的方式（音频），然而这两个通道并不总是一致的。我们探究这种不一致是否会降低语音情感识别的性能。通过将文本和音频映射到共享的效价-唤醒度-支配度空间中，我们利用双分量高斯混合模型将两者之间的距离转换为一个连续的、无需阈值的不一致性得分，并通过人类判断对该得分进行了验证。我们在表演性、对话性和自发性语音（eNTERFACE'05、MELD 和 CMU-MOSEI 数据集）上评估了三个具备音频能力的模型。在跨模态保持模型不变、并控制了真实情感标签和特征空间距离的情况下，只要音频存在，不一致性在所有语料库中都会增加出错的几率。而在相同特征上训练的监督分类器表现出相反的行为，这表明困难之处在于这些模型如何组合原始证据。

    arXiv:2610.40883v1 Announce Type: cross  Abstract: Voice-interactive large language models process both what a speaker says and how they say it, yet the two channels do not always agree. We ask whether that disagreement degrades speech emotion recognition. Mapping text and audio into a shared valence-arousal-dominance space, we convert the distance between them into a continuous, threshold-free incongruence score with a two-component Gaussian mixture, and validate the score against human judgement. We evaluate three audiocapable models on acted, conversational and spontaneous speech (eNTERFACE'05, MELD and CMU-MOSEI). With the model held constant across modalities, incongruence raises the odds of error whenever audio is present, in every corpus, after controlling for the ground-truth emotion and for distance in the feature space. Supervised classifiers trained on the same features behave in the opposite way, which suggests that the difficulty lies in how these models combine raw eviden
    
[^77]: FARO：通过跨块等风险贡献预算分配实现的函数空间自适应收益-风险优化

    FARO: Function-Space Adaptive Return-to-Risk Optimization via Cross-Block Equal-Risk-Contribution Budget Allocation

    [https://arxiv.org/abs/2610.40858](https://arxiv.org/abs/2610.40858)

    FARO提出了一种以参数块为决策单元的函数空间自适应优化器，根据各块的反事实收益自动分配更新几何，并借助投资组合理论中的等风险贡献原则跨块重新分配更新预算，在GPT模型训练上达到与最佳手工设计优化器配方相当的性能。

    

    训练大语言模型受计算资源限制，而当前的训练方案通过手工方式在优化器之间分配工作：对注意力机制和MLP矩阵使用正交化动量，对其余部分使用Adam风格的更新。这种划分依赖于人类关于哪类层值得哪种更新几何的先验知识，并且固定了一个由所有参数块共享的单一全局更新预算。我们提出FARO，一种函数空间自适应优化器，它将参数划分为块作为决策的基本单元。FARO根据每个块实现的反事实收益为其分配几何结构，然后冻结这些路径，并通过跨块等风险贡献（ERC，源自投资组合理论的风险平价原则）来重新分配冻结的每块预算。在FineWeb数据集上训练的GPT风格模型上，该框架给出了两个结果。首先，其FARO-Adaptive分支在完整训练计划下达到3.2780 nats，与最佳手工设计配方NorMuonAndAdam（3.2794）相当：它恢复了……

    arXiv:2610.40858v1 Announce Type: cross  Abstract: Training a large language model is compute-bound, and current recipes divide the work between optimizers by hand: orthogonalized momentum for attention and MLP matrices, Adam-style updates for the rest. This division rests on human priors about which layer type deserves which geometry, and it fixes a single global update budget that every parameter block shares. We introduce FARO, a function-space adaptive optimizer that makes the partition of parameters into blocks the unit of decision. FARO gives each block a geometry by its realized counterfactual return, then freezes the routes and reallocates the frozen per-block budget by cross-block equal risk contribution (ERC), the risk-parity principle from portfolio theory. On a GPT-style model trained on FineWeb, the framework gives two results. First, its FARO-Adaptive arm reaches 3.2780 nats at the full schedule, matching NorMuonAndAdam (3.2794), the best hand-designed recipe: it recovers
    
[^78]: 投机草稿模型与连续扩散语言模型之间的生成差距

    The Generative Gap Between Speculative Drafter and Continuous Diffusion Language Model

    [https://arxiv.org/abs/2610.40847](https://arxiv.org/abs/2610.40847)

    该研究在无目标模型验证的公平条件下，从零训练并比较连续扩散语言模型与投机草稿架构，发现基于 CDLM 的 LangFlow 在生成困惑度和 MAUVE 指标上均优于 DSpark 的草稿架构。

    

    连续扩散语言模型（CDLMs）和投机草稿模型都能实现高效的语言生成，但它们的评估方式通常不同：CDLMs 通过迭代细化独立生成文本，而草稿模型提出的文本则需经目标模型验证和修正。我们在无验证的条件下比较了这两种架构：CDLMs 采用 LangFlow，草稿模型则改造 DSpark 的马尔可夫和循环头架构用于独立生成。所有模型均从零开始训练，在长度为 16、128 和 1024 的 OpenWebText 上进行无条件、单块生成，不使用目标模型特征或验证，并采用相同的预处理方法、骨干网络维度和训练更新预算。我们使用外部语言模型评估生成困惑度，并使用 MAUVE 评估与参考文本的分布相似性。在默认去噪步数等于序列长度的设置下，LangFlow 在长度为 128 的文本上于两项指标均优于 DSpark 的两种变体。

    arXiv:2610.40847v1 Announce Type: new  Abstract: Continuous diffusion language models (CDLMs) and speculative drafters offer efficient language generation but are typically evaluated differently: CDLMs generate independently through iterative refinement, whereas drafters propose text for target-model verification and correction. We compare their architectures without verification, using LangFlow for CDLMs and adapting DSpark's Markov and recurrent-head architectures for standalone generation. All models are trained from scratch for unconditional, single-block OpenWebText generation at lengths 16, 128, and 1024, without target features or verification, using common preprocessing, backbone dimensions, and training-update budgets. We evaluate generation perplexity (GenPPL) with an external language model and distributional similarity to reference text with MAUVE. With default denoising steps equal to sequence length, LangFlow outperforms both DSpark variants on both metrics at lengths 128
    
[^79]: Safin-1.5：面向递归安全改进的安全原生架构与可验证推理

    Safin-1.5: Safety-Native Architecture and Verifiable Reasoning for Recursive Safety Improvement

    [https://arxiv.org/abs/2610.40751](https://arxiv.org/abs/2610.40751)

    提出Safin-1.5模型，通过将持久化安全状态与危险知识解耦相结合的安全原生架构，以及递归安全改进训练和推理时形式化推理框架，使大语言模型在自我能力提升的同时持续发现并修复安全风险。

    

    递归自我改进（RSI）使大型语言模型能够通过 successive 更新不断提升自身能力，但这些更新也可能引入新的风险并削弱现有的安全保障。为遵循AI-45度法则，我们追求安全RSI（Safety RSI），即通过持续的风险发现与修复，使安全能力作为内在能力与模型的通用能力一同发展，同时保持实用性并限制过度拒答。我们提出了Safin-1.5，一个建立在安全架构之上、通过递归安全改进训练而成的模型，并配有推理时框架将其能力扩展至形式化推理。我们的方法包含三个组成部分：i) 安全架构将持久化安全状态与危险知识解耦相结合。前者保留并有选择地调用安全知识，后者则将特定的危险知识迁移至可移除的专家模块中，从而实现对这些知识的移除（摘要在此处被截断）。

    arXiv:2610.40751v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) enables large language models to advance their capabilities through successive updates, but these updates can also introduce new risks and weaken existing safeguards. In line with the AI-45{\deg} Law, we pursue Safety RSI, in which safety develops as an intrinsic capability alongside general model capabilities through continual risk discovery and repair, while preserving utility and limiting over-refusal. We introduce Safin-1.5, a model built on a safety architecture and trained through recursive safety improvement, with an inference-time harness that extends its capabilities to formal reasoning. Our approach comprises three components: i) The safety architecture combines persistent safety states with hazardous-knowledge disentanglement. The former retains and selectively invokes safety knowledge, while the latter relocates targeted hazardous knowledge into removable experts, enabling its removal with m
    
[^80]: Mage-VL-Cowork：一种从进化式测试框架中学习混合计算机使用的完整训练配方

    Mage-VL-Cowork: A Full Training Recipe to Learn Hybrid Computer Use From Evolved Harnesses

    [https://arxiv.org/abs/2610.40747](https://arxiv.org/abs/2610.40747)

    提出了Mage-VL-Cowork——一个4B混合型计算机使用智能体，通过五阶段渐进式训练课程、可大规模低成本合成任务的Mage-CUA-Forge以及进化式教师框架生成高质量轨迹，在OSWorld-Verified上达到66.0%的性能。

    

    有效的计算机使用智能体（CUA）应当能够在单一策略中无缝地结合视觉GUI交互与代码和终端执行。我们提出了Mage-VL-Cowork，这是一个4B规模的混合型计算机使用智能体，它从通用多模态对齐的检查点出发，通过涵盖中期训练、SFT和RL的端到端渐进式五阶段课程进行训练。为了大规模支撑这一课程体系，Mage-CUA-Forge用可复用的、技能条件化的生成工作流和面向特定应用的准备器取代了逐个任务的智能体构建方式，合成了涵盖6,915项技能的293,806个桌面和网络任务，同时将每个任务的经常性开销降低了82.6%。我们没有直接蒸馏默认的教师模型，而是对教师测试框架进行进化，并实例化专门的CUA测试框架，从而产生高质量的轨迹以及阶段特定的监督信号。Mage-VL-Cowork-4B最终通过不同的训练阶段在OSWorld-Verified上攀升至66.0%，同时具有更高效的推理。

    arXiv:2610.40747v1 Announce Type: new  Abstract: Effective computer-use agents (CUAs) should seamlessly combine visual GUI interaction with code and terminal execution within a single policy. We present Mage-VL-Cowork, a 4B hybrid CUA trained from a general multimodally aligned checkpoint through an end-to-end, progressive five-stage curriculum across midtraining, SFT, and RL. To power this curriculum at scale, Mage-CUA-Forge replaces per-task agent construction with reusable, skill-conditioned generation workflows and application-specific preparers, synthesizing 293,806 desktop and web tasks across 6,915 skills while reducing recurring per-task expenditure by 82.6%. Rather than directly distilling a default teacher, we evolve the teacher harness and instantiate specialized CUA harnesses to yield high-quality trajectories alongside stage-specific supervision. Mage-VL-Cowork-4B finally climbs to 66.0% on OSWorld-Verified through different training stages with more efficient reasoning co
    
[^81]: 阻止并非保护：攻击成功率掩盖了RAG的哪些问题

    Prevented is Not Protected: What Attack Success Rate Hides about RAG

    [https://arxiv.org/abs/2610.40742](https://arxiv.org/abs/2610.40742)

    该论文提出“对齐RAG评估”（ARE）框架，通过匹配查询协议揭示攻击成功率指标所掩盖的问题：现有RAG防御方法虽能阻止部分投毒攻击，却可能引入意外的目标正向回答。

    

    检索增强生成（RAG）容易受到知识投毒攻击，即攻击者通过操纵检索到的证据来影响模型输出。现有防御方法主要使用攻击成功率（ASR）进行评估，这一指标掩盖了防御本身引发的故障以及表面防御成功背后的机制。为了解决这一问题，本工作提出了“对齐RAG评估”，这是一种匹配查询协议，可联合检验干净与被投毒的响应，涵盖有防御与无防御两种情况。通过结合匹配条件分析、LLM判官认可以及检索依赖性控制，所提出的框架能够区分“被阻止的失败”与“被引入的失败”、目标提及与LLM认可、以及单段落影响与证据组合的影响。在HotpotQA和Natural Questions上的实验表明，现有防御方法虽然能够阻止部分攻击，却会引入意想不到的目标正向答案，而评估……

    arXiv:2610.40742v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) is vulnerable to knowledge poisoning, where adversaries manipulate retrieved evidence to influence model outputs. Existing defenses are primarily evaluated using attack success rate (ASR), which obscures defense-induced failures and the mechanisms behind apparent attack prevention. To address this issue, this work introduces \textbf{Aligned RAG Evaluation}, a matched-query protocol that jointly examines clean and poisoned responses, with and without defense. By combining matched conditions analysis, LLM-judge endorsement, and retrieval dependence controls, the proposed framework distinguishes prevented from introduced failures, target mentions from LLM endorsement, and single-passage influence from evidence composition. Experiments on HotpotQA and Natural Questions reveal that existing defenses can introduce unexpected target-positive answers, although they can prevent some attacks, while evaluation
    
[^82]: Apriel Voice 27B Preview：一个面向语音、推理与工具使用的开放模型

    Apriel Voice 27B Preview: An Open Model for Speech, Reasoning, and Tool Use

    [https://arxiv.org/abs/2610.40738](https://arxiv.org/abs/2610.40738)

    该论文提出了开源语音-语言模型 Apriel Voice 27B Preview，它在统一的推理主干上整合了语音理解、上下文推理与工具调用能力，并通过两阶段训练方法使文本模型高效适应语音模态。

    

    一个强大的语音助手需要做的不仅仅是识别词语；它必须理解请求、结合上下文进行推理、维持对话，并通过工具执行操作。我们提出了 Apriel Voice 27B Preview，这是一个开源的语音-语言模型，它在共享的推理主干中统一了这些能力，同时保留了文本和图像输入模态。该模型基于经过后训练的 Gemma 4 26B-A4B 检查点构建，能够直接处理语音并同时生成响应和工具调用，而无需单独的对话模型。我们引入了一种两阶段训练流程来使模型适应语音：第一阶段建立语音接口，第二阶段则通过监督微调、文本回放，以及基于来源的交叉熵和前向 Kullback-Leibler 散度监督，联合调整投影器和主干网络。自动构建的语音条件对话为模型提供了对话训练数据，而本地教学……（原文摘要在此处截断）

    arXiv:2610.40738v1 Announce Type: cross  Abstract: A capable voice assistant must do more than recognize words; it must understand requests, reason over context, sustain conversations, and act through tools. We present Apriel Voice 27B Preview, an open speech-language model that unifies these capabilities within a shared reasoning backbone while retaining text and image input modalities. Built on the post-trained Gemma 4 26B-A4B checkpoint, Apriel Voice directly processes speech and generates both responses and tool calls without requiring a separate dialogue model. We introduce a two-stage training procedure to adapt the model to speech. The first stage establishes the speech interface, while the second jointly adapts the projector and backbone through supervised fine-tuning, text replay, and source-dependent cross-entropy and forward Kullback--Leibler divergence supervision. Automatically constructed speech-conditioned dialogues provide conversational training data, while local teach
    
[^83]: Li²SAO：通过组相对Actor-Critic自举提升轻量级单rollout异步强化学习优化

    Li$^{2}$SAO: Lifting Light Singl-Rollout Asynchronous RL Optimization via Group-Relative Actor-Critic Bootstrapping

    [https://arxiv.org/abs/2610.40726](https://arxiv.org/abs/2610.40726)

    该论文提出Li²SAO，通过组相对Actor-Critic自举摆脱对大规模预训练价值模型的依赖，在轻量级单rollout异步RL优化中兼顾推理任务与智能体任务的表现。

    

    异步强化学习（RL）正日益成为大语言模型（LLM）后训练的主流范式，因为它可以在rollout到达时即更新模型，从而提升训练效率。近期研究指出，GRPO中的组级采样并不天然契合异步RL训练，因为rollout长度服从长尾分布，并据此提出了单rollout异步优化（SAO），以提升稳定性和离策略表现，尤其是在智能体任务上。然而，SAO依赖于一个通过大规模价值预训练获得的闭源价值模型。本文在单轮推理任务和多轮智能体任务上分析了GRPO与LiSAO——即我们对SAO的轻量级复现（不含预训练价值模型）。我们发现，GRPO在推理任务上优于LiSAO，而在智能体任务上则情况相反。由于LiSAO的价值模型尚未得到充分训练，LiSAO在此方面大幅落后于GRPO……（原文摘要至此截断）

    arXiv:2610.40726v1 Announce Type: new  Abstract: Asynchronous reinforcement learning (RL) is becoming a mainstream paradigm for post-training large language models (LLMs), as it improves training efficiency by updating the model as rollouts arrive. Recent work argues that group-wise sampling in GRPO does not naturally fit asynchronous RL training because rollout lengths follow a long-tailed distribution, and proposes Single-rollout Asynchronous Optimization (SAO) to improve stability and off-policy behavior, especially on agentic tasks. However, SAO relies on a closed-source value model obtained from scaling value pretraining. In this paper, we analyze GRPO and LiSAO, our lightweight reproduction of SAO without the pretrained value model, on single-turn reasoning and multi-turn agentic tasks. We find that GRPO outperforms LiSAO on reasoning tasks, while the reverse holds on agentic tasks. Since the value model of LiSAO is not yet well trained , LiSAO lags behind GRPO substantially in e
    
[^84]: 用于校准宇宙学模拟的智能体系统

    Agentic Systems for Calibrating Cosmological Simulations

    [https://arxiv.org/abs/2610.40685](https://arxiv.org/abs/2610.40685)

    本文提出AgenticTuner智能体校准系统，利用大语言模型作为优化器并结合人类可读的先验知识，以远低于传统贝叶斯优化的CPU成本实现具有竞争力的宇宙学模拟校准性能。

    

    校准宇宙学模拟是一项昂贵的高维参数优化任务。以往的工作采用贝叶斯优化实现自动化，但这种方法将模拟器视为黑箱，无法将人类领域知识或工作流程纳入选择策略中。我们提出了AgenticTuner，这是一个智能体校准流水线，由当今的大型语言模型充当优化器，负责提出下一次参数试验方案。我们证明，当这些智能体被赋予非正式的、人类可读的先验知识（例如经验丰富的科学家可能拥有的代码库、文档和启发式指南）时，能够以一小部分的CPU成本实现具有竞争力的校准性能。我们希望这项工作能成为迈向能够执行复杂、长周期科学任务的人工智能科学家的早期一步。

    arXiv:2610.40685v1 Announce Type: cross  Abstract: Calibrating a cosmological simulation is an expensive, high-dimensional parameter optimization task. Previous works employ Bayesian optimization for automation, which treats the simulator as a black box and precludes the incorporation of human domain knowledge or workflow into the selection policy. We propose AgenticTuner, an agentic calibration pipeline with a present-day large language model serving as the optimizer who proposes the next parameter trial. We show that these agents can achieve competitive calibration performance at a fraction of the CPU cost when they are given informal, human-readable priors (such as codebase, documentation, and heuristic guides that seasoned scientists may possess). We hope this work serves as an early step toward AI scientists that can carry out complex, long-horizon scientific tasks.
    
[^85]: 聊天机器人LLM对总统行动与假设性改革判断的稳健性

    Robustness of Chatbot LLM Judgments of Presidential Actions and Hypothetical Reforms

    [https://arxiv.org/abs/2610.39398](https://arxiv.org/abs/2610.39398)

    该研究通过30天内每天两次询问五种聊天机器人配置对总统行动和假设性改革进行评分，发现LLM的判断总体稳健——各配置对评分符号与排名高度一致，同日不同配置给出相反评分的情况仅约4%。

    

    聊天机器人越来越多地被用户咨询公共事务，而它们对政策的判断只有在稳健可靠的情况下才有用。本研究在普通用户的使用场景下，通过消费级聊天界面测量了这种稳健性。在30天内，研究团队每天两次询问五种聊天机器人配置，要求其对最近四届总统政府的40项行动以及15项假设性改革对美国的长期影响进行评分，每次评分均附带置信度评价和简短评论。结果表明这些判断是稳健的：所有配置对各届政府平均得分的符号（正负性）和排名意见一致；同一天询问的两个不同配置对同一行动给出相反符号的情况仅约占4%。不同日期的回答比同日回答（后者共享题目的提问顺序）差异更大，且这种差异在一个月内缓慢增长，除了……（摘要原文在此处截断）

    arXiv:2610.39398v2 Announce Type: replace  Abstract: Chatbots are increasingly consulted about public affairs, and their judgments of policies are useful only if they are robust. We measure this robustness in the setting of an ordinary user, through consumer chat interfaces. For 30 days, five chatbot configurations (Mistral, Gemini Flash, Claude Sonnet, and Claude Opus at two effort settings) were asked twice a day to score the long-term effect on the United States of 40 actions of the last four presidential administrations and of 15 hypothetical reforms, each with a confidence and a short comment. The judgments are robust: all configurations agree on the sign and the ranking of the administrations' mean scores, and two different configurations asked on the same day give opposite signs to the same action in about 4% of cases. Responses from different days differ more than same-day responses (which shared the order of the items); the difference grows slowly over the month, except where 
    
[^86]: 无限OCR Works 2

    Unlimited OCR Works 2

    [https://arxiv.org/abs/2610.39308](https://arxiv.org/abs/2610.39308)

    通过将全部解码器注意力模块升级为稀疏参考滑动窗口注意力（SR-SWA），Unlimited OCR 2 将活跃参考令牌限制在固定容量内，实现了对任意超长文档以恒定计算成本进行解析。

    

    Unlimited OCR（无限OCR）开启了长程解析的时代，在文档长度增加时显著缓解了工作记忆的增长。然而，当文档长度扩展至数百页时，视觉上下文会大幅增长，急剧增加每个解码步骤的计算成本，从而限制了超长文档解析的可扩展性。受人类通过局部聚焦、无需重新扫描前文页面即可转录书籍的启发，我们提出了 Unlimited OCR 2，一个只关注其正在转录内容的文档解析模型。在 Unlimited OCR 的基础上，我们将所有解码器注意力模块升级为稀疏参考滑动窗口注意力，将活跃参考令牌的数量限制在固定容量内，从而在任意长度的文档上保持恒定的计算成本。通过将 SR-SWA 与无限视界解码范式相结合，Unlimited OCR 2 能够对几乎无限长度的文档进行解析，而……

    arXiv:2610.39308v2 Announce Type: replace  Abstract: Unlimited OCR ushers the era of long-horizon parsing, substantially mitigating the growth of working memory as document lengthens. However, as document length scales to hundreds of pages, the visual context grows substantially, sharply increasing the computational cost of each decoding step and limiting the scalability of ultra-long document parsing. Inspired by human book transcription via localized focus without re-scanning preceding pages, we introduce Unlimited OCR 2, a document parsing model that looks only at what it transcribes. Building on Unlimited OCR, we upgrade all the decoder attention modules to Sparse Reference Sliding Window Attention (SR-SWA), bounding active reference tokens to a fixed capacity and thus sustaining constant computational costs across arbitrarily long documents. By combining SR-SWA with Unlimited-Horizon Decoding Paradigm, Unlimited OCR 2 enables document parsing over virtually unlimited lengths witho
    
[^87]: 追踪检索文献对大语言模型科学构思的影响

    Tracing the Influence of Retrieved Literature on Scientific Ideation in LLMs

    [https://arxiv.org/abs/2610.39060](https://arxiv.org/abs/2610.39060)

    该研究通过在统一生成协议下对比五种文献检索策略与无文献基线（共45,000个想法），首次分离并量化了检索策略本身对LLM科学构思的影响，发现检索会降低同组论文生成想法的多样性但提高跨组想法的多样性，并提出基于JSD的归因方法追踪单篇文献对每个想法的影响。

    

    大语言模型（LLMs）越来越多地被用于从科学文献中生成研究想法。现有系统将文献检索与搜索、生成和精炼相结合，这使得难以确定检索策略本身如何塑造所生成的想法。我们通过在统一的生成协议下比较五种检索策略与一个“无文献”基线来隔离这一贡献，实验涵盖了15个机器学习主题、十个开源权重模型，共生成45,000个想法。相对于无文献基线，所有五种检索策略使由同一组论文生成的想法之间的多样性降低了5.6%–17.9%，而当合并由不同论文组生成的想法时，大多数策略使多样性增加了10.2%–18.1%。检索对想法质量和新颖性的影响总体上较为温和，但在不同主题之间存在差异。我们进一步应用了一种基于JSD的归因方法，以追踪单篇输入论文对每个生成想法的影响。

    arXiv:2610.39060v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used to generate research ideas from scientific literature. Existing systems combine literature retrieval with search, generation, and refinement, making it difficult to determine how the retrieval strategy itself shapes the ideas generated. We isolate this contribution by comparing five retrieval strategies and a No Literature baseline under a shared generation protocol, using 45,000 ideas across 15 machine learning topics and ten open-weight models. Relative to no literature basline, all five retrieval strategies reduce diversity among ideas generated from the same set of papers by $5.6\%$--$17.9\%$, while most increase diversity by $10.2\%$--$18.1\%$ when ideas from different sets are pooled. Effects on idea quality and novelty are modest on average but vary across topics. We further apply a JSD-based attribution method to trace the influence of individual input papers on each idea's m
    
[^88]: 当支持越界：在中国语境下评估基于大语言模型的心理健康支持中的伦理边界

    When Support Crosses the Line: Evaluating Ethical Boundaries in LLM-Based Mental Health Support in the Chinese Context

    [https://arxiv.org/abs/2610.38452](https://arxiv.org/abs/2610.38452)

    提出了基于中国心理健康专业伦理标准的MindEth-Bench基准，利用蒙特卡洛树搜索评估九个大语言模型在多轮心理健康交互中对伦理边界的遵守情况，发现模型在长期交互中可能从恰当回应滑向违反伦理的行为。

    

    大语言模型（LLM）正越来越多地被探索用于心理健康支持，然而现有基准主要关注支持能力，而非模型是否持续遵守专业伦理边界。这一问题在多轮交互中尤为重要，因为用户可能会随着时间推移引入新信息或施加情感压力。我们提出了MindEth-Bench，一个用于评估多轮心理健康交互中伦理边界遵守情况的基准。该基准立足于中国心理健康服务的专业伦理标准，涵盖伦理行为的多个维度，并使用蒙特卡洛树搜索（MCTS）系统性探索交互轨迹，以对模型边界进行压力测试。在九个大语言模型上的实验表明，最初回应得当的模型在长期交互中可能逐渐转向违反伦理的行为，且不同模型之间存在显著差异。

    arXiv:2610.38452v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly explored for mental health support, yet existing benchmarks primarily focus on supportive capabilities rather than whether models consistently adhere to professional ethical boundaries. This issue is particularly important in multi-turn interactions, where users may introduce new information or exert emotional pressure over time. We introduce MindEth-Bench, a benchmark for evaluating ethical boundary adherence in multi-turn mental health interactions. Grounded in professional ethical standards for mental health services in China, MindEth-Bench covers multiple dimensions of ethical conduct and uses Monte Carlo Tree Search (MCTS) to systematically explore interaction trajectories that stress-test model boundaries. Experiments across nine LLMs show that models that initially respond appropriately can shift toward ethical violations over extended interactions, with substantial variation acros
    
[^89]: LLM引导的进化搜索中的整体多样性管理

    Holistic Diversity Management in LLM-Guided Evolutionary Search

    [https://arxiv.org/abs/2610.38407](https://arxiv.org/abs/2610.38407)

    提出HoliDivEvolve方法，通过多样化初始化、岛屿轨迹发展和跨岛屿发现共享，在LLM引导的进化搜索全过程中建立并有效利用岛屿多样性，以避免过早收敛。

    

    LLM引导的进化搜索迭代地选择现有程序，将大语言模型（LLM）用作变异算子来生成新程序，并对其进行评估以指导后续搜索。为了保持多样性并降低过早收敛的风险，这类方法通常采用岛屿模型，即让被称为“岛屿”的程序组各自独立进化。然而，仅靠这种结构并不能保证不同的搜索方向，因为各个岛屿可能从相似的程序起步，或者在进化过程中收敛到相似的解。这表明多样性必须在整个搜索过程中被加以考虑，从用不同的程序初始化各个岛屿，到发展各岛屿的搜索轨迹，再到跨岛屿共享发现。为此，我们提出了HoliDivEvolve，这是一种在进化搜索全过程中建立并有效利用岛屿多样性的整体方法。具体而言，我们引入了多样化初始化来建立不同的初始……

    arXiv:2610.38407v2 Announce Type: replace  Abstract: LLM-guided evolutionary search iteratively selects existing programs, uses LLMs as mutation operators to generate a new program, and evaluates it to guide subsequent search. To preserve diversity and reduce the risk of premature convergence, it often adopts the island model, in which groups of programs, called islands, evolve separately. However, this structure alone does not guarantee distinct search directions, as islands may start from similar programs or converge toward similar solutions during evolution. This suggests that diversity must be considered throughout the search process, from initializing islands with distinct programs to developing island trajectories and sharing discoveries across islands. To this end, we propose HoliDivEvolve , a holistic approach to establishing and effectively utilizing island diversity throughout evolutionary search. Specifically, we introduce diverse initialization to establish different initia
    
[^90]: Keepace.ai：一个面向运动与健康领域的大语言模型技术报告

    Keepace.ai: A Technical Report on a Large Language Model for Sports and Health

    [https://arxiv.org/abs/2610.38249](https://arxiv.org/abs/2610.38249)

    Keepace是基于Qwen3.5-27B后训练的中文运动健康领域专用大语言模型，在健身计划生成、知识问答和运动数据分析三项任务上超越了基座模型及更大的通用模型，尤其在运动安全性、边界合规性和科学准确性方面提升显著。

    

    大语言模型有望大规模提供安全、科学且个性化的运动指导，但运动与健康领域目前仍缺乏经过精心整理的训练语料和严格的评估基准。Keepace（Keepace.ai）是基于Qwen3.5-27B通过后训练构建的领域专用大语言模型，目前针对中文运动指导进行了优化，涵盖三项任务：健身计划生成、运动与健康知识问答以及个人运动数据分析。在与领域专家共同设计的基于评分量表的内部评估中，Keepace的表现优于其基座模型和更大的通用模型基线，其中在运动安全性、边界合规性、结构化输出完整性和科学准确性方面的提升最为显著。本报告总结了Keepace当前的设计、数据流程、安全优先的评估框架以及评估结果。

    arXiv:2610.38249v2 Announce Type: replace  Abstract: Large language models hold promise for delivering safe, scientifically grounded, and personalized exercise guidance at scale, yet the sports-and-health domain still lacks curated training corpora and rigorous evaluation benchmarks. Keepace (Keepace.ai) is a domain-specialized LLM built on Qwen3.5-27B through post-training, currently optimized for Chinese-language exercise guidance across three tasks: workout plan generation, sports-and-health knowledge QA, and personal sports data analysis. In internal rubric-based evaluations designed with domain experts, Keepace outperforms its base model and larger general-purpose baselines, with the most pronounced improvements in exercise safety, boundary compliance, structured output completeness, and scientific accuracy. This report summarizes the current design, data pipeline, safety-first evaluation framework, and evaluation results of Keepace.
    
[^91]: DeFT-RAG：将多跳问题增量式分解为具有可复用答案的自包含子问题

    DeFT-RAG: Incrementally Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers

    [https://arxiv.org/abs/2610.37826](https://arxiv.org/abs/2610.37826)

    DeFT-RAG通过增量式地将多跳问题分解为携带可复用答案的自包含子问题，逐步恢复完整的证据链，从而提升RAG在多跳问答任务上的准确性。

    

    检索增强生成（RAG）使大语言模型（LLM）能够通过从大规模语料库中检索相关文本来回答问题。然而，RAG在处理多跳问题时表现不佳，因为这类问题需要通过问题中从未提及的中间实体，将分散在多个文档中的事实串联起来。基于图的RAG方法只根据问题中提及的实体进行一次检索，因此很少能恢复完整的证据链。迭代式RAG方法虽能逐步恢复证据链，但其准确性受限于要求LLM执行的众多任务，例如规划子问题、确定子问题间的依赖关系、根据已获得的答案改写子问题以及追踪已知信息。我们提出了DeFT-RAG，一个由三个组件构建的RAG框架。首先，DeFT-RAG基于原始问题以及已回答的子问题及其答案（QA对），每次生成一个自包含的子问题。每个子问题命名了……（原文此处截断）

    arXiv:2610.37826v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) lets a large language model (LLM) answer questions by retrieving relevant passages from a large corpus. However, RAG struggles with multi-hop questions, which require chaining facts spread across several documents through intermediate entities that the question never mentions. Graph-based RAG methods retrieve once from the entities a question names and so rarely recover the full evidence chain. Iterative RAG methods recover it step by step, but their accuracy is limited by the many tasks they ask the LLM to perform, such as planning the sub-questions, deciding their dependencies, rewriting them as answers arrive, and keeping track of what is known. We propose DeFT-RAG, a RAG framework built on three components. First, DeFT-RAG generates one self-contained sub-question at a time from the original question and the sub-questions answered so far with their answers (QA pairs). Each sub-question names eve
    
[^92]: 安全之头：学习型注意力门控提升提示注入鲁棒性

    Heads of Security: Learned Attention Gates Improve Prompt Injection Robustness

    [https://arxiv.org/abs/2610.37657](https://arxiv.org/abs/2610.37657)

    提出Heads of Security（HoS）防御方法，通过为每个注意力头学习注意力门控来实现指令与数据的分离，使模型能在处理不可信内容时不执行注入指令，相比现有防御方法显著提升了对自适应提示注入攻击的鲁棒性并保持模型效用。

    

    我们提出了Heads of Security（HoS），这是一种通过一小组学习得到的逐头注意力门控来强制实现指令-数据分离的提示注入防御方法。HoS学习每个注意力头应如何在受信任与不受信任的token之间重新分配注意力，使模型能够对不受信任的内容进行推理，同时不会遵循注入的指令。我们在Llama-3.3-70B-Instruct和GPT-OSS-20B上评估了HoS，并将其与SecAlign++（一种最先进的基于微调的防御方法）以及ASIDE（一种与我们的方法类似、将指令-数据分离内建于模型架构中的防御方法）进行比较。结果表明，HoS对自适应攻击具有更强的鲁棒性，在标准效用基准上表现更好，并在标准智能体提示注入基准上达到顶尖性能。更具体而言，黑盒攻击RL-Hammer对GPT-OSS-20B的攻击成功率（ASR）达到97%，对SecAlign达到85%，但对HoS仅为10%。

    arXiv:2610.37657v1 Announce Type: cross  Abstract: We introduce Heads of Security (HoS), a prompt injection defense that enforces instruction-data separation via a small set of learned per-head attention gates. HoS learns how each head should redistribute attention between trusted and untrusted tokens, enabling models to reason over untrusted content without following injected instructions. We evaluate HoS on Llama-3.3-70B-Instruct and GPT-OSS-20B. We compare it against SecAlign++, a state-of-the-art finetuning-based defense, and ASIDE, a defense that, like ours, builds instruction-data separation into the model architecture. We show that HoS offers greater robustness to adaptive attacks, better performance on standard utility benchmarks, and matches top performance on standard agentic prompt injection benchmarks. More specifically, the black-box RL-Hammer attack reaches 97% ASR against GPT-OSS-20B and 85% against SecAlign, but only 10% against HoS. Claudini, a state-of-the-art white-b
    
[^93]: 元认知解码：面向扩散语言智能体的高效推理

    Metacognitive Decoding: Efficient Inference for Diffusion Language Agents

    [https://arxiv.org/abs/2610.37642](https://arxiv.org/abs/2610.37642)

    提出一种轻量级元认知控制器插件，使扩散语言智能体学会何时确定token及生成多长文本，在保持任务准确率的同时平均减少20.6%的推理计算量。

    

    扩散语言模型通过并行token生成为构建高效智能体提供了一条有前景的路径。然而，高效的交互需要同时调整计算资源分配的位置以及生成文本的数量。我们引入了一个元认知控制器，根据当前响应的需求以及智能体与环境的交互情况来引导扩散生成过程。它学习何时确定token以及生成应延伸多远，从而减少响应内的整体计算开销。我们的控制器是扩散骨干网络的一个轻量级插件，通过智能体轨迹进行训练，以平衡正确性与生成成本。该方法将智能体的生成过程与其所服务的交互任务联系起来。在五个基准测试上的实验表明，与最高效的基线方法相比，我们的方法平均减少了20.6%的推理FLOPs，同时保持了具有竞争力的任务准确率。消融实验研究……

    arXiv:2610.37642v1 Announce Type: new  Abstract: Diffusion language models offer a promising path to efficient agents through parallel token generation. However, efficient interaction requires adapting both where computation is spent and how much text is generated. We introduce a metacognitive controller that guides diffusion generation according to the needs of the current response and the agent's interaction with the environment. It learns when to finalize tokens and how far generation should extend, reducing overall computation within responses. Our controller, a lightweight plug-in for the diffusion backbone, is trained on agent trajectories to balance correctness and generation cost. This approach connects the agent's generation process to the interaction the agent serves. Experiments across five benchmarks demonstrate that our approach reduces inference FLOPs by 20.6% on average compared with the most efficient baseline, while maintaining competitive task accuracy. Ablation studi
    
[^94]: SpectRAG：面向多跳检索增强生成中基于Token预算的上下文选择的加权分块图

    SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.36991](https://arxiv.org/abs/2610.36991)

    SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。

    

    检索增强生成（RAG）系统通过检索外部证据并将生成过程以检索到的上下文为条件，从而为大型语言模型（LLM）提供事实依据。尽管稠密检索在局部语义匹配方面较为有效，但独立排序的检索可能返回冗余段落、遗漏跨分块的依赖关系，并为多跳推理提供不完整的证据。我们提出了SpectRAG——一种谱图增强检索框架，它结合了混合候选种子选择、加权分块图构建、基于图的扩展、谱相关性传播以及基于Token预算的上下文选择。文档分块由稠密语义嵌入和图坐标系表示，后者源自一个加权分块图，其边编码了语义相似性、文档相邻性、章节层次结构、共享实体、引用链接以及LLM生成的效用问题之间的相似性。在查询时，SpectRAG构建一个hi……（摘要原文在此截断）

    arXiv:2610.36991v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a hi
    
[^95]: 将LLM-SC框架适配于活态神圣遗产：基于帕斯帕提那神庙游客评论的领域特定信息提取、语义网络与人工评估

    Adapting LLM-SC to Living Sacred Heritage: Domain-Specific Extraction, Semantic Networks and Human Evaluation on Pashupatinath Temple Reviews

    [https://arxiv.org/abs/2610.35685](https://arxiv.org/abs/2610.35685)

    该研究将LLM-SC框架适配到尼泊尔帕斯帕提那神庙这一活态神圣遗产场景，通过新增8种神圣遗产实体类型构建12类实体分类体系，从Tripadvisor评论中成功提取结构化数据，且新增的神圣遗产类型占实体记录的64.8%，证明领域特定适配对宗教遗产评论分析至关重要。

    

    大型语言模型（LLM）流水线可以将游客评论转化为结构化数据，但通用方案可能会遗漏活态神圣遗产场所中的宗教仪式、神圣空间、文化限制以及情感复杂的体验。我们通过在GPT-4o提取提示词中明确指定一个包含12种类型的实体分类体系（4种原始LLM-SC类型加8种神圣遗产类型）、4种关系类型和9个方面-情感维度，将LLM-社会计算（LLM-SC）框架适配到尼泊尔帕斯帕提那神庙，并将其应用于从3,939条处理后语料中固定随机抽取的2,500条英文Tripadvisor评论。导出的数据包含11,286条实体记录、6,232条方面-情感记录和5,590条关系记录，分别覆盖2,485、2,487和2,433条评论；其中新增的8种神圣遗产类型占实体记录的64.8%。以200条评论的双标注者仲裁参考标准为基准，精确的规范化表层与类型匹配……（摘要原文在此处截断）

    arXiv:2610.35685v1 Announce Type: cross  Abstract: Large language model (LLM) pipelines can turn visitor reviews into structured data, but generic schemes may miss rituals, sacred spaces, cultural restrictions and emotionally mixed experiences at living sacred heritage sites. We adapted the LLM-Social Computing (LLM-SC) framework to Pashupatinath Temple, Nepal, by specifying a 12-type entity taxonomy (four original LLM-SC types plus eight sacred-heritage types), four relation types and nine aspect-sentiment dimensions in a GPT-4o extraction prompt, which was applied to a fixed random sample of 2,500 English-language Tripadvisor reviews from a processed corpus of 3,939. The exports contain 11,286 entity, 6,232 aspect-sentiment and 5,590 relation records, covering 2,485, 2,487 and 2,433 reviews, respectively; the eight added sacred-heritage types account for 64.8% of entity records. Against an adjudicated two-annotator reference for 200 reviews, exact normalized surface-and-type matching
    
[^96]: SOPO：苏格拉底引导的策略优化用于片段级幻觉检测

    SOPO: Socratic Guided Policy Optimization for Span-Level Hallucination Detection

    [https://arxiv.org/abs/2610.32212](https://arxiv.org/abs/2610.32212)

    SOPO提出了一种苏格拉底引导的多轮强化学习框架，通过教师模型提供基于真实标签的间接提示来引导推理轨迹，从而解决片段级幻觉检测中的定位模糊、困难样本奖励稀疏和长度膨胀问题。

    

    大语言模型（LLMs）已在各种自然语言处理任务中得到广泛应用，但存在幻觉问题，即生成不忠实或不一致的内容。虽然可验证奖励强化学习（RLVR）在推理任务中展现出有前景的应用，但将其应用于片段级幻觉检测面临定位模糊性、困难样本奖励稀疏以及不受控的长度膨胀等挑战。为解决这些问题，我们提出了苏格拉底引导的策略优化（SOPO），这是一个用于鲁棒幻觉检测的多轮强化学习框架。SOPO的关键创新在于苏格拉底反馈机制，其中教师模型通过提供从真实标签推断出的间接提示，在采样过程中引导推理轨迹。这种教师引导的方法使模型能够为困难样本生成更有效的推理轨迹，从而缓解奖励稀疏问题。

    arXiv:2610.32212v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have gained widespread adoption in various natural language processing tasks, but suffer from hallucination issues where they generate unfaithful or inconsistent content. While Reinforcement Learning with Verifiable Rewards (RLVR) has shown promising applications in reasoning tasks, its application to span-level hallucination detection is challenged by localization ambiguity, reward sparsity on hard examples, and uncontrolled length inflation. To address these issues, we introduce Socratic Guided Policy Optimization (SOPO), a multi-turn RL framework for robust hallucination detection. The key innovation of SOPO is a Socratic feedback mechanism, where a teacher model guides the reasoning trajectories during rollout by providing indirect hints inferred from the ground truth. This teacher-guided approach enables the model to generate more effective reasoning trajectories for hard examples, thus alleviating rew
    
[^97]: 表格序列化决定了语言模型保留哪些关系

    Table Serialization Determines Which Relations Language Models Preserve

    [https://arxiv.org/abs/2610.24798](https://arxiv.org/abs/2610.24798)

    该研究发现表格的序列化格式决定了语言模型在表格查找中保留哪些关系——行优先格式更适合窄高表格（倾向保留行但丢失列），列优先格式更适合宽扁表格（倾向保留列但丢失行），且这一结论在多个模型上保持一致。

    

    我们研究了文本序列化方式如何影响指令微调语言模型中的表格查找。我们的诊断任务要求模型找到某一命名列下包含指定值的行，然后返回该行中另一命名列下的值。我们将相同的合成256单元格表格分别以行优先形式和列优先形式（每个表头放置在其对应值旁边）进行渲染，同时在五种形状中改变表格的宽高比。在三个模型上的实验表明，更优的序列化格式随表格形状而变化：行优先格式在窄高表格上表现更好，而列优先格式在宽扁表格上表现更好。在64×4的表格上，行优先格式领先42.3-54.0个百分点；在4×64的表格上，列优先格式领先35.0-59.0个百分点。将错误答案映射回其源单元格后，我们揭示了一种互补的错误模式：行优先倾向于保留正确的行但丢失目标列，而列优先倾向于保留正确的列但丢失行。

    arXiv:2610.24798v2 Announce Type: replace  Abstract: We study how text serialization affects table lookup in instruction-tuned language models. Our diagnostic asks the model to find the row containing a specified value under one named column, then return the value under another named column in that same row. We render the same synthetic 256-cell tables in row-major form and in column-major form with each header placed beside its values, while varying the table aspect ratio across five shapes. Across three models, the better format changes with table shape: row-major performs better on tall tables, whereas column-major performs better on wide tables. At 64 x 4, row-major leads by 42.3-54.0 percentage points; at 4 x 64, column-major leads by 35.0-59.0 points. Mapping wrong answers back to their source cells reveals a complementary error pattern: row-major tends to preserve the correct row but lose the target column, while column-major tends to preserve the correct column but lose the row
    
[^98]: DialFork：超越事实性，评估对话代理中的程序性幻觉

    DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents

    [https://arxiv.org/abs/2610.21561](https://arxiv.org/abs/2610.21561)

    DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。

    

    作为对话代理部署的大语言模型（LLM）被越来越多地期望遵循程序性逻辑。虽然事实性幻觉已受到广泛关注，但另一种失败模式却被忽视了：程序性幻觉，即代理产生的语句虽然单独来看流畅且事实上貌似合理，却偏离了预期的对话流程。我们提出了DialFork，这是首个在原始对话文本上衡量这种失败模式的基准。DialFork通过在受控轮次位置对符合规范的对话进行分叉并注入四种幻觉类型之一来构建，产生了涵盖12个领域、24个任务的36,278个对话；30名人工标注者验证了该构建过程。我们定义了三个评估任务，从三个粒度刻画程序性偏差：一组对话偏离符合规范对话的程度（用于监控已部署的代理）、单个对话偏离的程度（用于程序性检索……

    arXiv:2610.21561v1 Announce Type: cross  Abstract: Large language models (LLMs) deployed as conversational agents are increasingly expected to follow procedural logic. While factual hallucination has received extensive attention, a different failure mode has been overlooked: procedural hallucination, in which an agent produces individually fluent and factually plausible utterances while deviating from the intended conversational process. We introduce DialFork, the first benchmark to measure this failure mode on raw dialogue text. DialFork is built by forking conformant dialogues at controlled turn positions and injecting one of four hallucination types, yielding 36,278 dialogues across 24 tasks in 12 domains; 30 human annotators validate the construction. We define three evaluation tasks that characterize procedural deviation at three granularities: how far a set of dialogues deviates from conformant ones (monitoring deployed agents), how far a single dialogue does (procedural retrieva
    
[^99]: 将局部语境黏合为全局意义：Transformer表示的层论分解

    Gluing Local Contexts into Global Meaning: A Sheaf-Theoretic Decomposition of Transformer Representations

    [https://arxiv.org/abs/2610.17117](https://arxiv.org/abs/2610.17117)

    该论文提出一种无需训练的层论方法：在改写图上构建胞腔层并用其拉普拉斯算子的核 $H^0$ 提取内容子空间，仅 20 维即跨 8 个模型在 CounterFact 事实检索上平均超越 LEACE 全维子空间 +17.1 个百分点，且联合约束同一事实的所有改写可再提升 +6.5 个百分点。

    

    以往的可解释性研究并未检验表示在改写（paraphrasing）下是否保持一致。每一对改写都可以被视为一个局部见证，表明两个激活应当相同。从这些局部见证出发，我们可以在改写图上构建一个胞腔层。在给定冻结模型的条件下，“什么在改写下保持一致”这一问题可以由层拉普拉斯算子 $L_\mathcal{F}$ 来回答，且无需任何训练。我们的核 $H^0$ 给出了一个内容子空间。我们证明，仅使用 20 个维度，该核在留出的 CounterFact 事实检索任务上，跨 8 个模型平均比 LEACE 用全部维度保留的子空间高出 +17.1 个百分点。通过将一个事实的所有改写共同加以约束，还能带来额外的 +6.5 个百分点提升。我们的谱补揭示了因果影响集中在哪里。消融那些在改写时变化最大的坐标（这近似于消融谱

    arXiv:2610.17117v2 Announce Type: replace  Abstract: Prior research into interpretability does not test whether a representation stays consistent under paraphrasing. Each pair of paraphrases can be thought of as a local witness that two activations should be the same. From these local witnesses we can build a cellular sheaf on the paraphrase graph. The question of what is consistent under paraphrasing can be answered by the Laplacian $L_\mathcal{F}$ given a frozen model. No training required. Our kernel $H^0$ gives us a content subspace. We show that with only 20 dims the kernel outperforms the subspace that LEACE retains with all dims on held-out CounterFact fact retrieval by $+17.1$\,pp on average across 8 models. By constraining all paraphrases of a fact together we see an additional increase of $+6.5$\,pp. Our spectral complement shows us where the causal influence is concentrated. Ablating the coordinates that vary the most when paraphrased (which approximates ablating the spectra
    
[^100]: 大语言模型的“自我”在哪里？——从自我索引到自我指涉

    Where Is the "Self" of a Large Language Model? -- From Self-Index to Self-Reference

    [https://arxiv.org/abs/2610.16717](https://arxiv.org/abs/2610.16717)

    该研究通过在模型自身消息与外部来源消息之间进行表征对比，首次证明大语言模型同时具备“自我索引”（在所有层中持续区分自身与外部文本）和“自我指涉”（能认领自己写的消息）两个类似生物功能层次的“自我”证据。

    

    大语言模型是否拥有“自我”？如果有，是何种类型的自我？我们区分了生物学中熟悉的两个功能层次：自我索引，即区分自身材料与外部材料并据此调整加工过程，如同几乎所有动物都具备的“伴随放电”机制；以及自我指涉，即更高层次的能力，能够识别哪些内容属于自己，类似镜像自我识别。我们在Qwen3.8-27B模型上探测这两个层次，方法是将同一段文本放置在由模型自身聊天模板渲染的Assistant、User或System消息中，并报告了三个层次的证据。持续性：正如预先预测的那样，文本的表征在全部64层中、对每一条测试文本，都能将模型自身的消息与两种外部来源区分开来，且沿着一个共同的方向，即使答案并不取决于文本的提供者。耦合性：当被问及前一条消息是谁写的时，模型会认领自己的消息（0.77），并对User和System消息作出相似的归属判断……

    arXiv:2610.16717v2 Announce Type: replace  Abstract: Does a large language model have a self, and if so, of what kind? We separate two functional levels familiar from biology: a self-index, which distinguishes one's own material from external material and shapes processing, as corollary discharge does in nearly all animals, and self-reference, the higher ability to recognize what is one's own, as in mirror self-recognition. We probe both in Qwen3.8-27B by placing the same text in an Assistant, a User, or a System message rendered by the model's own chat template, and report three levels of evidence. Persistence: as predicted in advance, the representation of the text separates the model's own message from both external sources at all 64 layers and for every test text, along a shared direction, even when the answer does not depend on who supplied the text. Coupling: asked who wrote the preceding message, the model claims its own messages (0.77) and assigns User and System messages alike
    
[^101]: 思维惯性：大语言模型在被要求不思考时仍会继续思考

    Thinking Inertia: LLMs Keep Thinking When Told Not To

    [https://arxiv.org/abs/2610.11765](https://arxiv.org/abs/2610.11765)

    该论文提出了三项新指标（空思考率、问题-答案前相关性、显式推理率）来严谨地衡量大语言模型的“不思考”行为，并揭示了一种“思维惯性”现象——即使被明确要求不思考，LLM仍会继续进行推理。

    

    大语言模型（LLM）越来越普遍地配备了显式的“思考模式”，然而与其相对的“不思考”模式却很少受到关注。我们从两个维度研究LLM的“不思考”行为。a. 如何衡量“不思考”？以往的工作通常通过替代指标来定义“不思考”，例如禁用思考模式或缺少长篇推理轨迹。这些替代指标并不可靠：被禁用的思考模式仍可能输出推理内容，而长轨迹中可能包含的是填充性内容而非真正的推理。我们转而将每个响应规范化为答案前的推理轨迹和最终答案，并从三个层面进行评估：（i）空思考率（Empty-Thinking Rate），用于衡量严格的仅答案合规性；（ii）指令感知的问题-答案前相关性（Question-Pre-answer Relevance），用于衡量问题与答案前推理轨迹之间的相似度；（iii）以LLM为裁判的显式推理率（Explicit Inference Rate），用于衡量可见的显式推理行为。这些指标共同区分了仅答案输出、相关但无推理的文本……

    arXiv:2610.11765v1 Announce Type: new  Abstract: Large Language Models (LLMs) increasingly ship with explicit "thinking modes", yet their counterpart, "no-thinking", has received far less attention. We study LLMs' no-thinking behavior along two axes. a. How to measure no-thinking? Prior work typically defines no-thinking through proxies such as a disabled thinking mode or the absence of long traces. These proxies are unreliable: disabled thinking modes may still emit reasoning, while long traces may contain filler rather than genuine inference. We instead normalize each response into a pre-answer trace and final answer, and evaluate it at three levels: (i) Empty-Thinking Rate for strict answer-only compliance; (ii) instruction-aware Question-Pre-answer Relevance for similarity between the question and pre-answer trace; and (iii) LLM-as-judge Explicit Inference Rate for visible explicit inference. Together, these metrics distinguish answer-only output, relevant but non-inferential text,
    
[^102]: 面向语义物理现实的四阶张量注意力模型

    4-Tensor Attention Model for Semantic Physical Reality

    [https://arxiv.org/abs/2610.11716](https://arxiv.org/abs/2610.11716)

    该论文提出一种四阶张量注意力模型，通过在语义纤维与时间上下文纤维上进行联合归一化注意力来预测场景的下一语义状态，在参数量几乎相同的情况下，其最后一句交叉熵在三个设置上均优于自由运行的一维 Transformer（最多低 5.3%），为视频生成和机器人规划提供了新方法。

    

    我们提出了一种四阶张量注意力模型（4-tensor attention model），用于预测场景的下一个语义状态，可应用于视频生成与机器人规划。一个状态窗口包含位置 (x, t) 以及两条纤维：语义纤维和时间上下文纤维，并由一个 softmax 在整个窗口上联合归一化注意力。视频帧和智能体的情境被表示为这些状态；编码器、渲染器和规划器则不参与该更新过程。为了单独测试这一更新机制，我们在 ROCStories 数据集上进行训练，其中每个窗口在语义层上构成相同的下一句预测任务。在验证集上（每个设置使用一个随机种子），在三个匹配设置下，四阶张量模型的最后一句交叉熵均低于自由运行的一维 Transformer：在 H=2, L=2 时低 5.3%，在 H=4, L=2 时低 2.6%，在 H=4, L=3 时低 2.4%。在 H=4, L=2 时，两者的参数量几乎相同，分别为 172.5M 和 175.9M。在相同的两个 GPU 上，该四阶张量模型的运行在 2.4（原文此处截断）……

    arXiv:2610.11716v1 Announce Type: cross  Abstract: We describe a 4-tensor attention model that predicts the next semantic state of a scene, for video generation and robot planning. A window of states has positions (x, t) and two fibers, a semantic fiber and a temporal-context fiber, and one softmax normalizes attention jointly over the window. Frames and an agent's situation are written as those states; the encoder, the renderer, and the planner remain outside the update. To test the update on its own, we train on ROCStories, where each window poses the same next-sentence task at the semantic layer. On the validation split, with one seed per setting, the last-sentence cross-entropy on the three matched settings is lower for the 4-tensor model than for a free-running one-dimensional transformer by 5.3% at H=2, L=2, by 2.6% at H=4, L=2, and by 2.4% at H=4, L=3. At H=4, L=2 the parameter counts are nearly the same, 172.5M and 175.9M. On the same two GPUs that 4-tensor run finished in 2.4 
    
[^103]: 语言承载着专家的印象：基于评估工具锚定的LLM评判器可实现咨询质量评估的跨域迁移并超越域内训练

    Language Carries the Expert's Impression: Instrument-Anchored LLM Judges Transfer Counseling-Quality Assessment and Beat In-Domain Training

    [https://arxiv.org/abs/2610.08055](https://arxiv.org/abs/2610.08055)

    该研究发现，基于专家评分工具锚定构念、由小型开源LLM生成的会话级得分，能够跨领域迁移专家对咨询质量的总体印象评估，且跨域训练的效果甚至优于域内训练。

    

    arXiv:2610.08055v1 公告类型：new 摘要：对双人咨询对话中沟通质量的自动评估受制于数据瓶颈：专家评分的语料库规模小且扩充成本高昂。我们研究了专家总体印象预测在三个德语模拟咨询语料库（两个全科医疗语料库、一个学校相关的家长-教师语料库；n=195个经专家评分的会谈，其中一个语料库经过量表等值化处理）之间的跨域迁移。在其他域上训练的效果优于在目标域内训练：留一域迁移达到嵌套Spearman ρ = 0.54，而目标域内训练不超过0.48，会话层面的配对差距为+0.15，且当训练集规模匹配时该差距仍保持+0.12，因此这并不仅仅是数据量的问题。决定性的特征是来自小型开源权重LLM阅读双说话人转录文本所得到的会话级构念得分，这些构念主要源自专家的评分工具：由工具衍生的构念集合提升了……

    arXiv:2610.08055v1 Announce Type: new  Abstract: Automatic assessment of communication quality in dyadic counseling conversations is bottlenecked by data: expert-rated corpora are small and expensive to grow. We study cross-domain transfer of expert overall-impression prediction across three German corpora of simulated counseling (two general-practice medical, one school-related parent-teacher; $n=195$ expert-rated sessions, one corpus after scale equating). Training on the other domains beats training in-domain: leave-one-domain-out transfer reaches nested Spearman $\rho = 0.54$ against $\le 0.48$ within the target domain, a paired session-level gap of $+0.15$ that holds at $+0.12$ when the training-set sizes are matched, so it is not simply data volume. The decisive features are session-level construct scores from small open-weight LLMs reading the two-speaker transcript, with the constructs largely derived from the experts' rating instruments: the instrument-derived battery lifts a 
    
[^104]: 测试框架修复了什么？一项关于可见性、基线充分性与评估缺陷的预注册研究

    What Does a Harness Repair? A Preregistered Study of Visibility, Baseline Adequacy and Evaluation Defects

    [https://arxiv.org/abs/2610.05533](https://arxiv.org/abs/2610.05533)

    本预注册研究通过系统实验发现，测试框架搜索带来的评测分数提升往往源于解析器可见性问题和评估缺陷等伪影而非真实能力增益，且简单的关闭思考设置在多数情况下足以媲美甚至胜过经过搜索优化的框架配置。

    

    测试框架搜索会在冻结模型周围保留对提示词、token预算或解析器的更改，只要该更改能提升评测分数。这种提升可能来自解析器此前无法读取的答案、较弱的对比基线或评估缺陷。我们预注册了两项关于这些提升来源的研究，使用三个参数量从1.7B到8B的模型以及Qwen3-14B，并报告了全部65个主要终点。在12个模型-基准组合中的7个里，关闭思考模式比受限思考设置获得了更高的准确率，且每次提升主要来自那些在受限思考设置下没有可读答案的问题。对于小模型，关闭思考的设置在9个组合中的7个里并不显著劣于救援配置，对于1.7B模型，也不劣于四个通过GEPA选择的测试框架。在14B规模上，两个GEPA测试框架在LongevityBench字母任务上比它高出30到34分；但在描述性分析中，它们并未显著超过最强的恒定标签基线。

    arXiv:2610.05533v2 Announce Type: replace-cross  Abstract: Harness search keeps a change to the prompts, token budgets or parsers around a frozen model if the change raises a score. Such a gain can come from answers the parser could not read before, a weak comparison or an evaluation defect. We preregistered two studies of where these gains come from, with three models of 1.7B to 8B parameters and with Qwen3-14B, and report all 65 primary endpoints. Turning thinking off raised accuracy over a capped thinking setting in 7 of 12 model-benchmark cells, each time mostly through questions the capped setting left without a readable answer. For the small models, the thinking-off setting was not meaningfully worse than a rescue configuration in 7 of 9 cells, nor, for the 1.7B model, than four GEPA-selected harnesses. At 14B, two GEPA harnesses beat it by 30 to 34 points on the LongevityBench letter tasks; in descriptive analyses they were not detectably above the strongest constant-label basel
    
[^105]: 面向在线策略蒸馏的统一逐词元门控族：带多通道与偏置系数的FKL/RKL混合

    A Unified Per-Token Gating Family for On-Policy Distillation: FKL/RKL Mixing with Multi-Channel and Bias Coefficients

    [https://arxiv.org/abs/2609.11768](https://arxiv.org/abs/2609.11768)

    该论文提出了一个统一的四系数逐词元门控参数化方法，将EOPD和ToDi作为其一维限制的特殊情形统一起来，并通过多通道组合与显式偏置这两个额外自由度，在在线策略蒸馏实验中显著超越了单通道门控限制方法。

    

    逐词元的前向/反向KL散度损失门控已成为在线策略知识蒸馏（OPD）的标准技术，但现有方法如EOPD（Jin等，2026）和ToDi（Jung等，2025）各自固定了单一的门控信号和单一的门控方向，且两者从未被直接比较。我们引入了一个四系数参数化 lambda_t = sigma(a * h_t + b * u(x) + c + d * gap_t)，其中与方向对齐的EOPD和ToDi代理形式作为其一维（1D）限制情形出现，同时该参数化还引入了多通道组合和显式偏置作为额外的自由度。在TweetEval（Barbieri等，2020）的情感和仇恨数据集上，使用Qwen3-32B教师模型和Qwen3-4B学生模型进行实验，完整门控族中的配置在36个可比单元中的33个里达到了比同等幅度的单通道（仅熵/仅差距）1D限制更高的准确率，而一项26单元的均值匹配隔离实验将动态门控置于……（原文在此处截断）

    arXiv:2609.11768v1 Announce Type: cross  Abstract: Per-token gating of forward/reverse KL losses has become a standard technique for on-policy knowledge distillation (OPD), but existing methods such as EOPD (Jin et al., 2026) and ToDi (Jung et al., 2025) each fix a single gating signal and a single gating direction, and the two have never been compared directly. We introduce a four-coefficient parameterization lambda_t = sigma(a * h_t + b * u(x) + c + d * gap_t) in which direction-aligned proxies of EOPD and ToDi appear as one-dimensional (1D) restrictions, and which adds multi-channel composition and an explicit bias as further degrees of freedom. On TweetEval (Barbieri et al., 2020) emotion and hate, with a Qwen3-32B teacher and a Qwen3-4B student, configurations in the full family reach higher accuracy than the matched-magnitude single-channel (entropy-only / gap-only) 1D restrictions in 33 of 36 comparable cells, and a 26-cell mean-match isolation experiment places dynamic gating a
    
[^106]: SIRF：面向工业内容风控的规范内化风险基础模型

    SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control

    [https://arxiv.org/abs/2609.11752](https://arxiv.org/abs/2609.11752)

    SIRF通过持续预训练将平台风控规范内化到模型权重中，无需人工标注即可在超低延迟、仅输出判定的部署形态下实现高精度风险处置，仅用约7000万token的CPT便将P95精确率下的黑名单召回率提升15.1个百分点且不损害通用能力。

    

    对于工业内容风控而言，真正的部署约束不是平均准确率，而是在高精度与秒级延迟下能自动处置多少风险内容。我们提出SIRF（规范内化风险基础模型），它通过持续预训练（CPT）将平台的复杂规范策略内化到模型权重中——这些策略通过EntiGraph、MAGA改写和账号级思维链（CoT）合成，无需额外人工标注——从而在超低延迟、仅输出判定结果的部署形态下实现高精度的规则执行。一项同源对照比较实验（Qwen3-8B-SFT对比SIRF-8B-SFT，两者采用相同的策略注入和仅判定输出形式，唯一差异在于是否进行基于策略的CPT）将性能提升归因于规范内化：SIRF-8B-SFT在95%精确率约束下达到71.3%的黑名单召回率（Black Recall@P95），较基线提升15.1个百分点，且仅使用约7000万个CPT token，未损害模型通用能力；在本接口下纳入的可获取logprob的模型中，它……

    arXiv:2609.11752v1 Announce Type: cross  Abstract: For industrial content risk control, the real deployment constraint is not average accuracy but how much risk can be auto-handled under high precision and second-level latency. We present SIRF (Spec-Internalized Risk Foundation Model), which internalizes a platform's complex policies, synthesized without additional human annotation via EntiGraph, MAGA rewriting and account-level chain-of-thought (CoT), into the weights via continued pretraining (CPT), so rules are applied at high precision under an ultra-low-latency, verdict-only deployment. A controlled same-source comparison (Qwen3-8B-SFT vs. SIRF-8B-SFT, identical policy injection and verdict-only output form, differing only in policy-grounded CPT) attributes the gain to internalization: SIRF-8B-SFT reaches 71.3% Black Recall@P95, +15.1pp over the baseline, using only ~70M CPT tokens without harming general ability, and among included, logprob-available models under this interface i
    
[^107]: TamilEOT：面向泰米尔语电话语音语义话轮结束检测的数据集与模型

    TamilEOT: A Dataset and Model for Semantic End-of-Turn Detection in Tamil Telephone Speech

    [https://arxiv.org/abs/2609.05631](https://arxiv.org/abs/2609.05631)

    本文发布了首个面向泰米尔语电话语音的语义话轮结束检测资源——包含18,485个标注话轮边界的TamilEOT数据集及两个微调检测模型，将准确率从零样本的70.30%提升至86.13%，并可在普通CPU上以低于150毫秒的延迟实时运行。

    

    语音代理必须在每次停顿时判断用户是否已经说完。如果缺乏针对该语言的模型，这一决策只能退回到固定的静音超时：设得短，代理会打断用户；设得长，每一轮对话都要付出完整的等待时间。目前已存在开放的语义话轮结束检测器，但据我们所知，尚无任何检测器覆盖南印度语言。我们发布了TamilEOT：从116段真实泰米尔语电话对话中切分出的18,485个带标注的话轮边界，以及两个从Smart Turn v3微调而来的纯音频检测器。在来自30个未见过的通话的4,168个片段构成的保留测试集上，准确率从零样本的70.30%提升至83.71%（8.7 MB模型）和86.13%（21 MB模型）；ROC-AUC从0.751提升至0.921。两个模型在笔记本电脑CPU上单线程运行均低于150毫秒。我们还报告了构建该数据集的代价。经盲测人工听审校验的规则生成标签，在正类上正确率为95.9%，在负类上为44.4%，这……

    arXiv:2609.05631v1 Announce Type: cross  Abstract: A voice agent has to decide, at every pause, whether the user has finished speaking. Without a model of the language that decision falls back to a fixed silence timeout: set it short and the agent interrupts, set it long and every turn pays the full wait. Open semantic end-of-turn detectors exist, but to our knowledge none covers a South Indian language. We release TamilEOT: 18,485 labelled turn boundaries cut from 116 real Tamil telephone conversations, and two audio-only detectors fine-tuned from Smart Turn v3. On a held-out split of 4,168 clips from 30 unseen calls, accuracy rises from 70.30% zero-shot to 83.71% (8.7 MB) and 86.13% (21 MB); ROC-AUC rises from 0.751 to 0.921. Both models run in under 150 ms single-threaded on a laptop CPU. We also report what building it cost. Rule-derived labels, checked against a blind human listening pass, were right 95.9% of the time on the positive class and 44.4% on the negative class, which is
    
[^108]: 词元表示的深度流是非线性的，且不沿其自身密度下降

    The Depth Flow of Token Representations Is Nonlinear and Does Not Descend Its Own Density

    [https://arxiv.org/abs/2608.29706](https://arxiv.org/abs/2608.29706)

    该研究用离散朗之万模型刻画语言模型中词元表示的逐层演化流，发现该流是非线性的、并不沿自身对数密度下降而是下降一个不同的势函数，且旋转分量占可解释漂移的 4%–45%。

    

    词元的表示被逐层携带穿过整个网络，整个词表一起被携带便构成了一个流。我们在 Pythia-160M 和 Pythia-410M 的语料库平均轨迹上，将该流的运动方程拟合为离散朗之万模型，并在留出词元上对预测步进进行评分。线性映射常被用作单层的廉价替代，但它们所概括的流并非线性：在两个模型的每一次层间转移中，二次漂移都优于线性映射，且 Kramers–Moyal 估计量在其邻域保持局部性的地方也得出了相同结论。我们随后进一步刻画了这个流。首先，我们证明它并不沿自身的对数密度下降；漂移项实际下降的是一个势函数，而该势并非密度。其次，旋转分量不可忽略，占可解释漂移的 4% 到 45%，且环流体现在流所保持的性质上：一个词元在全部十三层中保持其角度秩，而其范数……（原文在此处截断）

    arXiv:2608.29706v1 Announce Type: new  Abstract: A token's representation is carried through the network layer by layer. The whole vocabulary carried together forms a flow. We fit this flow's equation of motion as a discrete Langevin model over corpus-mean trajectories of Pythia-160M and Pythia-410M, and score the predicted steps on held-out tokens. Linear maps are often used as cheap surrogates for a layer. The flow they summarize is not linear: a quadratic drift beats the linear linear map at every transition of both models, and the Kramers--Moyal estimator agrees wherever its neighborhoods stay local.   We then characterize the flow further. First, we show that it does not descend its own log-density. The drift instead descends a potential that is not the density. Second, the rotational component is not negligible, $4$ to $45\%$ of the explainable drift, and the circulation shows in what the flow preserves: a token keeps its angular rank across all thirteen layers while its norm ran
    
[^109]: 从关联到因果：通过因果关系与注意力机制提升检索增强生成的检索精度

    From Association to Causation: Improving Retrieval Precision of Retrieval-Augmented Generation via Causal Relations and an Attention Mechanism

    [https://arxiv.org/abs/2608.21702](https://arxiv.org/abs/2608.21702)

    该论文提出通过因果图建模检索阶段，利用对撞子结构中的注意力机制，无需训练即可提升RAG检索精度，解决相似度仅捕捉关联而忽略因果的问题。

    

    arXiv:2608.21702v1 公告类型：新 摘要：检索增强生成（RAG）将大语言模型的生成基于检索到的文档，但标准的终端检索阶段——稠密向量相似度，可选地随后进行重排序——常常返回与查询共享关键词但不包含所需信息的文档，这种失败模式随着知识库的增长而加剧。我们将其追溯到一个概念性差距：相似度仅捕捉关联关系，而真正重要的文档与查询之间是因果关联的。我们用基于赖兴巴赫共因原理的因果图对终端检索阶段建模：查询与检索文档共享的关键词构成潜在共因A，文档的剩余关键词构成潜在集合B，将文档与理想输出联系起来。由于检索到的文档是一个对撞子（A -> d <- B），检索本身在查询和B之间打开了一条关联路径，这允许一种无需训练的注意力机制。

    arXiv:2608.21702v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) grounds LLM generation on retrieved documents, but the standard terminal retrieval stage--dense-vector similarity, optionally followed by reranking--often returns documents that share keywords with the query without containing the needed information, a failure mode that grows with the knowledge base. We trace it to a conceptual gap: similarity captures only associational relations, whereas the documents that matter are linked to the query causally. We model the terminal retrieval stage with a causal graph grounded in Reichenbach's common cause principle: the keywords shared by the query and a retrieved document form a latent common cause A, and the document's residual keywords form a latent set B linking the document to the ideal output. Since a retrieved document is a collider (A -> d <- B), retrieval itself opens an associational path between the query and B, which licenses a training-free, attentio
    
[^110]: 只前向一次：在冻结语言模型的单次前向传播中同时进行回答与弃权

    You Only Pass Once: Answering and Abstaining Together in a Single Forward Pass of a Frozen Language Model

    [https://arxiv.org/abs/2608.14465](https://arxiv.org/abs/2608.14465)

    本文提出了一种在冻结语言模型的单次前向传播中同时实现回答和弃权的方法，通过训练小型网络重建引导前的残差来消除干扰，从而兼顾推理准确性和弃权能力。

    

    冻结语言模型在推理任务中存在两个相互关联的弱点：它未能充分利用其自身残差流中已编码的证据，并且在输入信息不足时无法检测到这一点，从而产生虚构内容。本文整合了两条在相同残差流上解决这些问题的研究路线：一个条件性引导探针在中间层写入残差流，从冻结骨干中恢复推理准确性；一个零样本充分性方向读取残差流，在信息不足时进行弃权。当在单次前向传播中部署时，它们会相互干扰：引导写入改变了方向读取的状态，在小模型上导致跨域转移的AUROC分数损失高达8点；而单独进行一次干净的前向传播会使推理成本翻倍。我们保持方向固定，并训练一个小型网络从引导后的残差中重建引导前的残差——使用（引导后，干净）对的均方误差，无需充分性标签——并重新...

    arXiv:2608.14465v1 Announce Type: new  Abstract: A frozen language model on reasoning tasks has two coupled weaknesses: it under-uses evidence its own residual stream already encodes, and it fails to detect when the input is insufficient to answer, so it confabulates. This paper consolidates two research lines that address these on the same residual stream: a conditional steering probe writes the stream at mid-stack layers and recovers reasoning accuracy from a frozen backbone, and a zero-shot sufficiency direction reads the stream and abstains when information is insufficient. Deployed in one forward pass they interfere: the steering write shifts the state the direction reads, costing up to 8 AUROC points of cross-domain transfer on small models; a separate clean pass doubles inference cost. We keep the direction fixed and train a small network to reconstruct the pre-steering residual from the steered one -- mean-squared error on (steered, clean) pairs, no sufficiency labels -- and re
    
[^111]: 无意义的谬误：古典阿拉伯语中任意符号的结构不可能性

    The No-Meaning Falsity: The Structural Impossibility of the Arbitrary Sign in Classical Arabic

    [https://arxiv.org/abs/2608.07737](https://arxiv.org/abs/2608.07737)

    本文通过为古典阿拉伯语非拼接形态学建立形式化数学模型，证明词根-模式系统在结构上排除了索绪尔符号任意性公理——词汇意义在派生层面即被唯一确定，从而从形式上反驳了后现代无限语义不确定性的主张。

    

    本文探讨了后现代主义关于无限制语义不确定性的主张，及其基础——索绪尔符号任意性公理——是否与古典阿拉伯语的结构架构相容。我们针对阿拉伯语的非拼接形态学建立了一个形式化数学模型，其中词汇意义由不变词根与形态句法模式之间的相互作用所决定。在此框架下，我们建立了形态对应定理，证明每个词汇项都由一个词根-模式对唯一生成；以及语义定域定理，证明词汇意义在派生层面、即表层实现之前就已被确定。为处理索绪尔较弱的相对任意性概念，我们通过条件柯尔莫哥洛夫复杂度对其加以形式化，将任意性以算法方式定义为“无规则”属性。我们证明了一般相对任意性在形式上……

    arXiv:2608.07737v2 Announce Type: replace  Abstract: This paper investigates whether the postmodern claim of unrestricted semantic indeterminacy, and its foundational Saussurean axiom of the arbitrary sign, are compatible with the structural architecture of Classical Arabic. We develop a formal mathematical model of Arabic non concatenative morphology in which lexical meaning is determined by the interaction between an invariant root and a morphosyntactic pattern. Within this framework, we establish a Morphological Correspondence Theorem, demonstrating that every lexical item is uniquely generated by a root pattern pair, and a Semantic Localization Theorem, proving that lexical meaning is determined at the derivational level prior to surface realization. To address Saussurean weaker notion of relative arbitrariness, we formalize it via conditional Kolmogorov complexity, defining arbitrariness algorithmically as the no rule property. We prove that general relative arbitrariness is forma
    
[^112]: CogBattery：语言模型认知任务行为与结构的人类参照评估

    CogBattery: Human-Referenced Evaluation of Cognitive Task Behavior and Structure in Language Models

    [https://arxiv.org/abs/2607.24999](https://arxiv.org/abs/2607.24999)

    CogBattery通过13个认知范式将54个开源语言模型与人类数据进行对比评估，发现模型虽能复现人类效应的方向，但在错误模式、元认知置信度等方面与人类存在系统性差异，且更大规模的模型并不必然更接近人类认知表现。

    

    人类认知任务使我们能够在受控条件下将语言模型与人类进行比较。然而，模型可以在复现人类效应方向的同时犯下不同的错误。CogBattery 在13个认知范式的文本改编版本上评估了54个开源权重语言模型，并将每个模型与其来源研究的人类数据进行比较。与人类类似，模型对n-back任务中的诱饵刺激的反应强于对照刺激，但它们在普通非目标项上产生的错误警报却远多于人类。大多数模型所报告的置信度在区分正确与错误答案方面不如人类有效，而且大多数模型总是选择相同的赌注，其中包括那些分类准确的模型。这些模式跨越了许多模型家族普遍存在。纵观各任务，模型得分共享两个维度。第一个维度与模型规模密切相关，然而更大的模型并不总是更接近人类。预定义的认知分组相较于相同规模的随机分组并未表现出可靠的预测优势。

    arXiv:2607.24999v2 Announce Type: replace  Abstract: Human cognitive tasks let us compare language models with humans under controlled conditions. Yet a model can reproduce the direction of a human effect while making different errors. CogBattery evaluates 54 open-weight language models on text adaptations of 13 cognitive paradigms, comparing each with human data from its source study. Like humans, models respond more to n-back lures than to controls, yet they make far more false alarms on ordinary non-targets. Most models' reported confidence separates correct from incorrect answers less well than humans' does, and most always choose the same wager, including models that classify accurately. These patterns span many model families. Across tasks, model scores share two dimensions. The first is closely tied to scale, yet larger models are not consistently closer to humans. Predefined cognitive groupings show no reliable predictive advantage over random groupings of the same size. These 
    
[^113]: MS-GPT：将串联质谱从头结构解析重新思考为分子语言模型的谱诱导后验查询

    MS-GPT: Rethinking MS/MS De Novo Structure Elucidation as Spectrum-Induced Posterior Querying of a Molecule-Language Model

    [https://arxiv.org/abs/2607.23607](https://arxiv.org/abs/2607.23607)

    MS-GPT提出将串联质谱的从头分子结构解析重新建模为对条件分子语言模型的谱诱导后验查询，从而解决了传统指纹范式中“神谕指纹训练”与“噪声光谱指纹推理”之间的失配问题。

    

    从串联质谱（MS/MS）中解析分子结构是分析化学中的一个核心逆问题。现有的MS/MS分子识别方法大多仍依赖于参考谱库或预定义的候选集，而从头（de novo）解析方法则旨在直接从光谱生成结构。一种常见的从头解析路线是先从光谱预测分子指纹，再由指纹解码出结构，这种方式使解码器能够在仅含分子的大型语料库上进行预训练。然而，这一范式存在训练与推理之间的失配问题：解码器在训练时使用的是由分子计算得到的“神谕指纹”，而在推理时，它面对的却是带噪声的、由光谱诱导的指纹后验——该后验通常被坍缩为单一的阈值化指纹。我们提出MS-GPT，将经由指纹介导的从头结构解析重新构建为对一个条件分子语言模型的谱诱导后验查询。MS-GPT对一个分子语言模型进行条件化……

    arXiv:2607.23607v2 Announce Type: replace-cross  Abstract: Molecular structure elucidation from tandem mass spectra (MS/MS) is a central inverse problem in analytical chemistry. Most existing approaches to MS/MS identification remain tied to reference libraries or predefined candidate sets, whereas de novo methods aim to generate structures directly from spectra. A common de novo route predicts a molecular fingerprint from the spectrum and then decodes structures from it, enabling decoder pretraining on large molecule-only corpora. However, this paradigm creates a training-inference mismatch: the decoder is trained on oracle fingerprints computed from molecules, but at inference it is queried with a noisy spectrum-induced fingerprint posterior that is typically collapsed to a single thresholded fingerprint. We introduce MS-GPT, which recasts fingerprint-mediated de novo elucidation as spectrum-induced posterior querying of a conditional molecule-language model. MS-GPT conditions a mole
    
[^114]: Windowed-MTP：在百万token长上下文中消除草稿KV缓存的全上下文开销

    Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context

    [https://arxiv.org/abs/2607.21535](https://arxiv.org/abs/2607.21535)

    提出Windowed-MTP方法，仅对MTP草稿头施加StreamingLLM风格的窗口化注意力加注意力汇聚机制，无需训练即可消除百万token长上下文中草稿KV缓存的高昂读取开销，同时保持目标模型全上下文验证的分布不受影响。

    

    推测解码通过让草稿模型提出候选token、再由目标模型并行验证来加速自回归生成。前沿模型越来越多地内置多token预测（MTP）草稿头，并假设其开销很低。然而在长上下文场景中，随模型附带的全注意力MTP草稿头在与混合架构目标模型配合时可能变得异常昂贵。该草稿头在每个解码步骤的γ次草稿前向传播中都要对完整KV缓存进行注意力读取，而目标模型中仅9-25%的层保留全注意力，因此单个草稿层发出的上下文读取速率可与整个目标模型相当。在接受率较低的困难工作负载上，深层的原生MTP会产生净负收益：比不使用推测解码还要慢。我们仅对草稿部分应用StreamingLLM风格的滑动窗口加注意力汇聚机制（Windowed-MTP）；目标模型的验证仍保持全上下文，因此目标分布得以保留。该方法无需训练、可直接插入使用。在三个……（原文摘要在此处截断）

    arXiv:2607.21535v2 Announce Type: replace-cross  Abstract: Speculative decoding accelerates autoregressive generation by having a draft propose tokens that a target verifies in parallel. Frontier models increasingly ship a built-in Multi-Token-Prediction (MTP) draft head, assumed to be cheap beside verification. A shipped full-attention MTP head can become disproportionately expensive beside a hybrid target at long context. The head attends over the full KV cache for each of $\gamma$ draft forwards per decode step, while only 9-25% of target layers retain full attention, so a single draft layer can issue context reads at a rate comparable to the whole target. On hard, low-acceptance workloads, deep native MTP becomes net-negative: slower than no speculation.   We apply a StreamingLLM-style window plus attention sink to the draft only (Windowed-MTP); target verification remains full-context, so the target distribution is preserved. The intervention is training-free and drop-in. On three
    
[^115]: 使用大语言模型对舆论数据进行插补

    Imputation of Public Opinion Data with Large Language Models

    [https://arxiv.org/abs/2606.09351](https://arxiv.org/abs/2606.09351)

    本文首次采用规范的多重插补评估标准，在超过500万条舆论调查数据上系统评估了84种上下文学习配置的大语言模型插补方法，发现其最优表现仅能与MICE PMM、TabPFN等统计基线方法相当，且对调查主题高度敏感。

    

    大语言模型（LLM）常被评估作为调查问卷回答的模拟器，然而应用性调查研究主要面临的是题目无回答问题。插补通过填补这些缺失值来恢复部分观测数据集的整体结构，并有其自身明确的评估标准，这与预测准确率有着根本的不同。我们首次在多种缺失机制下，基于来自四项舆论调查的超过500万条插补回答，使用规范的多重插补评估指标和统计基线对LLM插补进行了评估。我们通过考察84种上下文学习（ICL）配置，并在六个开源权重LLM上比较表现最佳的配置，来评估ICL在该任务上的效果。最强的ICL方法优于零样本角色提示，但充其量也只能与统计基线方法（MICE PMM、TabPFN）相当。ICL方法对调查主题较为敏感，并且……

    arXiv:2606.09351v2 Announce Type: replace  Abstract: Large language models (LLMs) are widely evaluated as simulators of survey responses, yet applied survey research mostly faces item nonresponse. Imputation fills in those missing values to restore the overall structure of a partially observed dataset and has its own well-defined evaluation criteria, which differ fundamentally from prediction accuracy. We present the first evaluation of LLM imputation against proper multiple-imputation metrics and statistical baselines on more than 5 million imputed responses from four public opinion surveys under various missingness mechanisms. We evaluate in-context learning (ICL) for this task by investigating 84 ICL configurations and comparing the best-performing configurations on six open-weight LLMs. The strongest ICL approach improves on zero-shot persona prompting but remains, at best, competitive with statistical baselines (MICE PMM, TabPFN). ICL approaches are sensitive to survey topics and 
    
[^116]: EvoMemBench：从自我进化视角评估智能体记忆能力的基准测试

    EvoMemBench: Benchmarking Agent Memory from a Self-Evolving Perspective

    [https://arxiv.org/abs/2605.18421](https://arxiv.org/abs/2605.18421)

    该论文提出了首个从自我进化视角出发、沿记忆范围与记忆内容两个维度系统评估LLM智能体记忆能力的统一基准EvoMemBench，并通过对比15种记忆方法与长上下文基线，揭示出当前记忆系统尚无通用解决方案。

    

    arXiv:2605.18421v3 公告类型：替换 摘要：近期针对大语言模型（LLM）智能体的基准测试主要评估推理、规划和执行能力。然而，记忆对智能体同样至关重要，因为它使智能体能够随时间推移存储、更新和检索信息。由于现有基准测试缺乏系统评估记忆机制的方法，这一能力至今未被充分评估。本文从自我进化的视角研究智能体记忆，并提出了EvoMemBench——一个沿两个维度组织的统一基准：记忆范围（回合内 vs. 跨回合）和记忆内容（面向知识 vs. 面向执行）。我们在标准化协议下将15种代表性记忆方法与强大的长上下文基线进行了比较。结果表明，当前的记忆系统仍远未达到通用解决方案的水平：长上下文基线依然极具竞争力；记忆在当前上下文不足或任务困难时最有帮助；且没有任何单一方法能够……

    arXiv:2605.18421v3 Announce Type: replace  Abstract: Recent benchmarks for Large Language Model (LLM) agents mainly evaluate reasoning, planning, and execution. However, memory is also essential for agents, as it enables them to store, update, and retrieve information over time. This ability remains under-evaluated, largely because existing benchmarks do not provide a systematic way to assess memory mechanisms. In this paper, we study agent memory from a self-evolving perspective and introduce EvoMemBench, a unified benchmark organized along two axes: memory scope (in-episode vs. cross-episode) and memory content (knowledge-oriented vs. execution-oriented). We compare 15 representative memory methods with strong long-context baselines under a standardized protocol. Results show that current memory systems are still far from a general solution: long-context baselines remain highly competitive, memory helps most when the current context is insufficient or tasks are difficult, and no sing
    
[^117]: VerifyMAS：面向基于大语言模型的多智能体系统故障归因的假设验证方法

    VerifyMAS: Hypothesis Verification for Failure Attribution in LLM-based Multi-Agent Systems

    [https://arxiv.org/abs/2605.17467](https://arxiv.org/abs/2605.17467)

    提出VerifyMAS框架，将LLM多智能体系统的故障归因转化为在完整交互轨迹上进行故障假设的提出与验证，从而捕捉局部日志方法无法发现的全局性故障并实现细粒度归因。

    

    大语言模型驱动的多智能体系统（LLM-MAS）在复杂任务中表现出色，但不可靠的智能体仍是制约系统级可靠性的关键瓶颈。因此，自动故障归因至关重要。然而，现有方法（如直接预测智能体-错误对、智能体优先的故障归因）依赖于智能体的局部日志，会遗漏那些仅在完整交互轨迹中才显现的全局性故障，例如跨步骤的不一致和智能体间的协调错误。此外，直接预测故障会带来巨大的组合搜索空间，阻碍细粒度的归因。为应对这些挑战，我们提出了VerifyMAS——一个用于智能体故障归因的假设验证框架。VerifyMAS不再直接预测故障智能体和错误类型，而是针对完整轨迹提出并验证故障假设。这种基于验证的方法将归因分解为轨迹层面的……

    arXiv:2605.17467v2 Announce Type: replace  Abstract: Large language model-driven multi-agent systems (LLM-MAS) excel at complex tasks, yet unreliable agents remain a key bottleneck to system-level reliability. Automatic failure attribution is therefore critical, but existing approaches, such as direct prediction of agent-error pairs and agent-first failure attribution, rely on local logs of agents and miss global failures that only manifest over full interaction trajectories, such as cross-step inconsistencies and inter-agent coordination errors. Moreover, directly predicting failures induces a large combinatorial search space, hindering fine-grained attribution. To address these challenges, we propose VerifyMAS, a hypothesis verification framework for agent failure attribution. Instead of directly predicting faulty agents and error types, VerifyMAS formulates and verifies failure hypotheses against full trajectories. This verification-based approach decomposes attribution into traject
    
[^118]: 可控用户模拟

    Controllable User Simulation

    [https://arxiv.org/abs/2605.11519](https://arxiv.org/abs/2605.11519)

    该论文将可控用户模拟形式化为因果推断问题，揭示了基于事后轨迹标签进行监督微调会引入破坏因果一致性的结构性前瞻偏差，并证明在策略偏移下这种失败会导致评估指标方差几何级数爆炸的“可控性崩溃”现象。

    

    使用离线数据集评估对话智能体往往无法覆盖罕见场景，也难以支持对新策略的测试。这促使人们采用可控用户模拟器来进行有针对性的、反事实的评估，其通常通过提示或微调大语言模型来实现。在本工作中，我们将可控模拟形式化为一个因果推断问题。通过将自然语言评估与离线策略评估方法相衔接，我们表明，通过在后验轨迹标签上进行监督微调来训练模拟器的标准做法会产生一个结构上有偏的模型。具体而言，这些标签与产生数据的行为策略不可避免地耦合在一起，注入了破坏因果一致性的前瞻偏差。此外，我们证明在策略偏移下，这种失败会导致评估指标的方差呈几何级数爆炸，我们将这一现象称为“可控性崩溃”。

    arXiv:2605.11519v2 Announce Type: replace-cross  Abstract: Using offline datasets to evaluate conversational agents often fails to cover rare scenarios or to support testing new policies. This has motivated the use of controllable user simulators for targeted, counterfactual evaluation, typically implemented by prompting or fine-tuning large language models. In this work, we formalize controllable simulation as a causal inference problem. By bridging natural language evaluation with off-policy evaluation methodology, we show that the standard practice of training simulators via supervised fine-tuning on post-hoc trajectory labels yields a structurally biased model. Specifically, these labels are inextricably coupled to the data-generating behavior policy, injecting a look-ahead bias that breaks causal consistency. Furthermore, we prove that under policy shift this failure causes the variance of evaluation metrics to explode geometrically, a phenomenon we term controllability collapse. 
    
[^119]: 从输入侧最小化模态差距：你的语音大语言模型可以成为一个具备韵律感知能力的文本大语言模型

    Minimizing Modality Gap from the Input Side: Your Speech LLM Can Be a Prosody-Aware Text LLM

    [https://arxiv.org/abs/2605.05927](https://arxiv.org/abs/2605.05927)

    该论文提出TextPro-SLM，通过统一语音编码器WhisperPro生成同步的文本token与韵律嵌入，从输入侧最小化语音与文本之间的模态差距，使语音LLM在保留原始文本LLM语义能力的同时，在3B和7B规模下实现最低模态差距并具备出色的副语言理解能力。

    

    语音大语言模型（SLM）通常是在文本大语言模型（TLM）检查点的基础上构建的，但它们仍然存在显著的模态差距。以往的工作主要试图从输出侧缩小这一差距，即让语音生成更接近文本形式，但差距依然存在。我们认为，剩余的关键瓶颈在于输入侧。我们提出了TextPro-SLM，这是一种使口语输入更接近具备韵律感知能力的文本LLM输入形式的语音大语言模型。TextPro-SLM结合了WhisperPro——一个能够生成同步文本token与韵律嵌入的统一语音编码器——以及一个经过训练的LLM骨干网络，该骨干网络在保留原始TLM语义能力的同时学习副语言理解能力。实验表明，TextPro-SLM在3B和7B两种规模下均实现了领先SLM中最低的模态差距，同时在副语言理解任务上也展现出强劲的整体性能。

    arXiv:2605.05927v3 Announce Type: replace  Abstract: Speech large language models (SLMs) are typically built from text large language model (TLM) checkpoints, yet they still suffer from a substantial modality gap. Prior work has mainly attempted to reduce this gap from the output side by making speech generation more text-like, but the gap remains. We argue that the key remaining bottleneck lies on the input side. We propose TextPro-SLM, an SLM that makes spoken input more closely resemble that of a prosody-aware text LLM. TextPro-SLM combines WhisperPro, a unified speech encoder that produces synchronized text tokens and prosody embeddings, with an LLM backbone trained to preserve the semantic capabilities of the original TLM while learning paralinguistic understanding. Experiments show that TextPro-SLM achieves the lowest modality gap among leading SLMs at both 3B and 7B scales, while also delivering strong overall performance on paralinguistic understanding tasks. These gains are ac
    
[^120]: LLM智能体长期记忆安全综述：跨记忆生命周期的攻击、防御与治理

    A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle

    [https://arxiv.org/abs/2604.16548](https://arxiv.org/abs/2604.16548)

    本文提出记忆生命周期框架与可验证记忆治理（VMG）框架，系统刻画了LLM智能体长期记忆所面临的持久性、有状态性与传播性等新型安全威胁，并沿六个生命周期阶段和四个安全目标组织了攻击、防御及其跨阶段依赖关系的全景式分析。

    

    LLM智能体中可写入、可跨会话持久化记忆的出现，带来了与传统的以输入为中心的安全关注点在性质上截然不同的威胁格局，其特征体现为三种属性：持久性、有状态性和传播性。为了系统地刻画这一威胁格局，我们提出了一个记忆生命周期框架，沿两个维度对攻击、防御及其跨阶段依赖关系进行组织：六个生命周期阶段（写入、存储、检索、执行、共享与传播、遗忘与回滚）和四个安全目标（完整性、机密性、可用性、治理）。这一分析进而揭示了在系统层面需要形式化安全保证的需求，由此催生了可验证记忆治理——一个由五个架构原语构成的框架，它规定了长期记忆系统必须提供哪些可验证机制，以维持对其记忆状态可审计、可恢复的控制。我们的

    arXiv:2604.16548v3 Announce Type: replace-cross  Abstract: The emergence of writable, cross-session persistent memory in LLM agents introduces a qualitatively different threat landscape from conventional input-centric security concerns, characterized by three properties: persistence, statefulness, and propagation. To systematically characterize this landscape, we propose a Memory Lifecycle Framework that organizes attacks, defenses, and their cross-phase dependencies along two axes: six lifecycle phases (Write, Store, Retrieve, Execute, Share & Propagate, Forget & Rollback) and four security objectives (Integrity, Confidentiality, Availability, Governance). This analysis in turn exposes the need for formal security guarantees at the system level, motivating Verifiable Memory Governance (VMG), a framework of five architectural primitives that specifies what verifiable mechanisms a long-term-memory system must provide to maintain auditable, recoverable control over its memory state. Our 
    
[^121]: M$^\star$：每个任务都值得拥有专属的记忆框架

    M$^\star$: Every Task Deserves Its Own Memory Harness

    [https://arxiv.org/abs/2604.11811](https://arxiv.org/abs/2604.11811)

    提出Mstar方法，通过反思性代码演化自动为每个任务生成并联合优化包含模式、逻辑和指令的可执行Python记忆程序，在对话、具身规划和专业推理等任务上均超越现有基线。

    

    大语言模型智能体依赖记忆框架来写入、组织、检索和使用过往经验。然而，适用于某一任务的记忆框架在另一任务上往往表现不佳，因为对话、具身规划和专家推理需要不同的存储与检索行为。为解决这一局限，我们提出了Mstar，一种通过反思性代码演化自动发现任务优化记忆程序的方法。具体而言，Mstar将记忆框架实现为一个可执行的Python记忆程序，该程序定义了模式、逻辑和指令，并对这些组件进行联合优化。我们在涵盖对话、具身规划和专业推理的四个任务上评估了Mstar。结果表明，Mstar在所有评估任务上的性能均优于现有基线方法。此外，演化得到的记忆程序在各个评估领域展现出结构上截然不同的处理机制。

    arXiv:2604.11811v3 Announce Type: replace-cross  Abstract: Large language model agents rely on a memory harness to write, organize, retrieve, and use past experience. A harness that works for one task often fails on another, because conversations, embodied planning, and expert reasoning require different storage and retrieval behavior. To address this limitation, we introduce Mstar, a method that automatically discovers task-optimized memory programs through reflective code evolution. Specifically, Mstar operationalizes a memory harness as an executable Python memory program that defines the Schema, Logic, and Instruction, and optimizes these components jointly. We evaluate Mstar on four tasks covering conversation, embodied planning, and specialized reasoning. Our results demonstrate that Mstar improves performance over existing baselines across all evaluated tasks. Furthermore, the evolved memory programs exhibit structurally distinct processing mechanisms for each evaluated domain. 
    
[^122]: 最坏情形误警约束下的最优多比特生成式水印方案

    Optimal Multi-bit Generative Watermarking Schemes Under Worst-Case False-Alarm Constraints

    [https://arxiv.org/abs/2604.08759](https://arxiv.org/abs/2604.08759)

    本文证明了大语言模型多比特生成式水印中现有方案是次优的，并提出两种能达到漏检概率理论下界的新型编码-解码构造，通过线性规划刻画了最优性条件，从而完整解决了最坏情形误警约束下的最优多比特水印设计问题。

    

    本文研究了在最坏情形误警约束下，面向大语言模型的多比特生成式水印问题。先前的工作建立了有限词元情形下可实现的漏检概率的下界，并提出了一个声称能够达到该下界的方案。然而，我们证明该方案实际上是次优的。随后，我们提出了两种新的编码-解码构造，它们能够达到先前建立的下界，从而完整刻画了最优的多比特水印性能。我们的方法将水印设计问题表述为一个线性规划，并推导出能够实现最优性的结构条件。此外，我们还识别了先前构造的失效机制，并比较了所提出的两种方案之间的权衡。

    arXiv:2604.08759v2 Announce Type: replace-cross  Abstract: This paper considers the problem of multi-bit generative watermarking for large language models under a worst-case false-alarm constraint. Prior work established a lower bound on the achievable miss-detection probability in the finite-token regime and proposed a scheme claimed to achieve this bound. We show, however, that the proposed scheme is in fact suboptimal. We then develop two new encoding-decoding constructions that attain the previously established lower bound, thereby completely characterizing the optimal multi-bit watermarking performance. Our approach formulates the watermark design problem as a linear program and derives the structural conditions under which optimality can be achieved. In addition, we identify the failure mechanism of the previous construction and compare the tradeoffs between the two proposed schemes.
    
[^123]: 人类能分辨吗？关于人类对大语言模型生成新闻感知的双轴研究

    Can Humans Tell? A Dual-Axis Study of Human Perception of LLM-Generated News

    [https://arxiv.org/abs/2604.03755](https://arxiv.org/abs/2604.03755)

    基于1,054名参与者2,318条判断的研究表明，人类无法可靠地区分大语言模型生成的新闻与人类撰写的新闻，即使面对参数量仅7B的小型开源模型也是如此。

    

    人类能分辨新闻文章是由人撰写的还是由大语言模型（LLM）生成的吗？我们使用JudgeGPT研究平台来探究这一问题，该平台可在连续量表上独立测量来源归因（人类 vs. 机器）和真实性判断（真实 vs. 虚假）。基于收集自1,054名参与者对六个大语言模型生成内容的2,318条判断，我们报告了五项发现：（1）参与者无法可靠地区分机器生成文本与人类撰写文本（p > .05，Welch t检验）；（2）这种无法区分的情况在所有测试的模型中均成立，包括参数量仅为7B的开源权重模型；（3）自我报告的领域专业知识能够预测判断准确性（r = .35，p < .001），而政治倾向则不能（r = -.10，不显著）；（4）聚类分析揭示了两种不同的应对策略（“怀疑者” vs. “相信者”）；（5）在大约30次连续评估之后，由于认知（疲劳）原因，判断准确性会出现下降。

    arXiv:2604.03755v2 Announce Type: replace-cross  Abstract: Can humans tell whether a news article was written by a person or a large language model (LLM)? We investigate this question using JudgeGPT, a study platform that independently measures source attribution (human vs. machine) and authenticity judgment (legitimate vs. fake) on continuous scales. From 2,318 judgments collected from 1,054 participants across content generated by six LLMs, we report five findings: (1) participants cannot reliably distinguish machine-generated from human-written text (p > .05, Welch's t-test); (2) this inability holds across all tested models, including open-weight models with as few as 7B parameters; (3) self-reported domain expertise predicts judgment accuracy (r = .35, p < .001) whereas political orientation does not (r = -.10, n.s.); (4) clustering reveals distinct response strategies ("Skeptics" vs. "Believers"); and (5) accuracy degrades after approximately 30 sequential evaluations due to cogn
    
[^124]: AgentCollab：一种由自我评估驱动的LLM智能体高效协作范式

    AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents

    [https://arxiv.org/abs/2603.26034](https://arxiv.org/abs/2603.26034)

    AgentCollab提出了一种由自评估驱动的协作推理框架，利用智能体自身的反思信号动态协调不同能力级别的LLM模型，仅在必要时才升级至更强的推理层级，从而在执行效率与推理鲁棒性之间取得平衡。

    

    由大语言模型（LLM）驱动的自主智能体通过长程推理和工具交互来执行复杂任务，其中执行效率与推理鲁棒性之间存在根本性的权衡。处于不同能力-成本层级的模型具有互补优势：较低成本的模型能够快速执行，但在困难的推理片段上可能表现不佳，而更强的模型能以更高的计算成本提供更鲁棒的推理。我们提出了AgentCollab，这是一个自驱动的协作推理框架，可在智能体执行过程中动态协调具有不同推理能力的模型。该框架不依赖外部路由模块，而是利用智能体自身的自反思信号来判断当前推理轨迹是否正在取得实质性进展，并且仅在必要时才将控制权升级至更强的推理层级。为了进一步稳定长程执行，

    arXiv:2603.26034v2 Announce Type: replace  Abstract: Autonomous agents powered by large language models (LLMs) perform complex tasks through long-horizon reasoning and tool interaction, where a fundamental trade-off arises between execution efficiency and reasoning robustness. Models at different capability-cost levels offer complementary advantages: lower-cost models enable fast execution but may struggle on difficult reasoning segments, while stronger models provide more robust reasoning at higher computational cost. We present AgentCollab, a self-driven collaborative inference framework that dynamically coordinates models with different reasoning capacities during agent execution. Instead of relying on external routing modules, the framework uses the agent's own self-reflection signal to determine whether the current reasoning trajectory is making meaningful progress, and escalates control to a stronger reasoning tier only when necessary. To further stabilize long-horizon execution,
    
[^125]: 通过嵌入空间分离增强大语言模型的安全性

    Enhancing Safety of Large Language Models via Embedding Space Separation

    [https://arxiv.org/abs/2603.20206](https://arxiv.org/abs/2603.20206)

    提出ES2微调方法，通过增大有害与无害嵌入之间的距离来增强大语言模型对抗性攻击的安全性，并利用嵌入锚定损失和KL散度损失两种正则化手段来保持模型的原始能力。

    

    安全对齐并不能完全防止大语言模型在对抗性攻击下产生有害内容。我们观察到，现有的攻击方法通常会将有害嵌入移动至无害嵌入所在的区域。这促使我们探索增大有害嵌入与无害嵌入之间的距离是否能够提高大语言模型的安全性，因为更大的分离距离意味着攻击需要造成更大的嵌入偏移，而这会破坏语义。我们提出了嵌入空间分离（ES2），这是一种能够增大有害嵌入与无害嵌入之间距离的微调方法。然而，仅仅最大化这一距离可能会损害大语言模型的能力。因此，我们采用了两种正则化损失：嵌入锚定损失使无害嵌入在方向上与冻结的基础模型保持接近，KL散度损失使无害提示上的输出分布与基础模型的输出分布保持接近。我们的实验……

    arXiv:2603.20206v2 Announce Type: replace  Abstract: Safety alignment does not fully prevent large language models (LLMs) from producing harmful content under adversarial attacks. We observe that existing attack methods typically move harmful embeddings toward regions of harmless embeddings. This motivates us to explore whether increasing the distance between harmful and harmless embeddings can improve the safety of LLMs, since greater separation requires attacks to make larger embedding shifts, which can destroy semantics. We propose Embedding Space Separation (ES2), a fine-tuning method that increases the distance between harmful and harmless embeddings. However, maximizing this distance alone can damage LLMs' capabilities. Thus, we use two regularization losses: an embedding anchor loss keeps harmless embeddings close in direction to those of a frozen base model and a KL-divergence loss keeps output distributions on harmless prompts close to those of the base model. Our experimental
    
[^126]: SoK：智能体化检索增强生成（Agentic RAG）：分类体系、架构、评估与研究方向

    SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions

    [https://arxiv.org/abs/2603.07379](https://arxiv.org/abs/2603.07379)

    该SoK论文首次将智能体化检索增强生成（Agentic RAG）统一形式化为部分可观测马尔可夫决策过程，并在此基础上提出了全面的分类体系、模块化架构分解与研究方向。

    

    检索增强生成（RAG）系统正日益演变为智能体化架构，其中大型语言模型能够自主协调多步推理、动态记忆管理和迭代式检索策略。尽管工业界已快速采用，当前研究仍缺乏将智能体化RAG视为序贯决策系统的系统性理解，导致架构高度碎片化、评估方法不一致以及可靠性风险尚未解决。本知识系统化（SoK）论文提供了理解这类自主系统的首个统一框架。我们将智能体化的检索-生成循环形式化为有限时域的部分可观测马尔可夫决策过程，显式建模其控制策略和状态转移。基于这一形式化，我们构建了全面的分类体系和模块化的架构分解，按其……（原文截断）

    arXiv:2603.07379v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) systems are increasingly evolving into agentic architectures where large language models autonomously coordinate multi-step reasoning, dynamic memory management, and iterative retrieval strategies. Despite rapid industrial adoption, current research lacks a systematic understanding of Agentic RAG as a sequential decision-making system, leading to highly fragmented architectures, inconsistent evaluation methodologies, and unresolved reliability risks. This Systematization of Knowledge (SoK) paper provides the first unified framework for understanding these autonomous systems. We formalize agentic retrieval-generation loops as finite-horizon partially observable Markov decision processes, explicitly modeling their control policies and state transitions. Building upon this formalization, we develop a comprehensive taxonomy and modular architectural decomposition that categorizes systems by thei
    
[^127]: 探索模型合并在自动语音识别多领域适应中的潜力与局限性

    Exploring the potential and limitations of Model Merging for Multi-Domain Adaptation in ASR

    [https://arxiv.org/abs/2603.05354](https://arxiv.org/abs/2603.05354)

    该研究将模型合并应用于多领域语音识别，提出新算法BoostedTSV-M，在10个欧洲葡萄牙语领域上超越了完整微调，同时保留了分布外性能。

    

    模型合并是多任务训练的一种可扩展替代方案，它将多个专门化模型的能力整合到一个单一模型中。这对于大型语音基础模型尤其具有吸引力，因为此类模型通常通过领域特定的微调进行适配，从而产生多个定制化的检查点，而当有新数据出现时重复进行完整微调在计算上是不可承受的。在本工作中，我们研究了面向多领域ASR的模型合并，并针对10个欧洲葡萄牙语领域对11种合并算法进行了基准测试，评估了领域内准确率、分布偏移下的鲁棒性，以及英语和多语言性能。我们进一步提出了BoostedTSV-M，一种基于TSV-M的新合并算法，它通过奇异值提升来缓解秩坍缩问题并改善数值稳定性。总体而言，我们的方法在欧洲葡萄牙语上优于完整微调，同时保持了分布外性能。

    arXiv:2603.05354v2 Announce Type: replace  Abstract: Model merging is a scalable alternative to multi-task training that combines the capabilities of multiple specialised models into a single model. This is particularly attractive for large speech foundation models, which are typically adapted through domain-specific fine-tuning, resulting in multiple customised checkpoints, for which repeating full fine-tuning when new data becomes available is computationally prohibitive. In this work, we study model merging for multi-domain ASR and benchmark 11 merging algorithms for 10 European Portuguese domains, evaluating in-domain accuracy, robustness under distribution shift, as well as English and multilingual performance. We further propose BoostedTSV-M, a new merging algorithm based on TSV-M that mitigates rank collapse via singular-value boosting and improves numerical stability. Overall, our approach outperforms full fine-tuning on European Portuguese while preserving out-of-distribution 
    
[^128]: 预期危害：重新思考（错）对齐大语言模型的安全评估

    Expected Harm: Rethinking Safety Evaluation of (Mis)Aligned LLMs

    [https://arxiv.org/abs/2602.01600](https://arxiv.org/abs/2602.01600)

    提出以执行可能性加权的“预期危害”安全评估新指标，揭示现有大语言模型存在“逆向风险校准”问题——对高成本低可能性的威胁过度拒绝，却对低成本高可能性的威胁依然脆弱。

    

    当前对大语言模型（LLM）安全性的评估主要依赖基于严重程度的分类体系来衡量模型对恶意查询响应的危害性。我们认为这一框架需要重新审视，因为它隐含地将模型输出等同于实际造成的危害，而忽略了执行可能性——即在模型给出响应后，真实世界的行为者实际实施该威胁的条件概率。在本工作中，我们提出“预期危害”这一指标，它以越狱攻击的执行可能性对严重程度进行加权，并将执行可能性建模为执行成本的函数。通过对最先进模型的实证分析，我们揭示了一种系统性的“逆向风险校准”现象：模型对低可能性（高成本）的威胁表现出过度强烈的拒绝行为，而面对高可能性（低成本）的查询时依然脆弱。在机制层面，我们首先通过线性探针分析揭示了虽然拒绝机制（原文在此处截断）……

    arXiv:2602.01600v2 Announce Type: replace-cross  Abstract: Current evaluations of LLM safety predominantly rely on severity-based taxonomies to assess the harmfulness of models' responses to malicious queries. We argue that this formulation requires re-examination as it implicitly equates model output with realized harm while neglecting Execution Likelihood-the conditional probability of a threat being realized by a real-world actor given a model's response. In this work, we introduce Expected Harm, a metric that weights the severity of a jailbreak by its execution likelihood, modeled as a function of execution cost. Through empirical analysis of state-of-the-art models, we reveal a systematic Inverse Risk Calibration: models disproportionately exhibit stronger refusal behaviors for low-likelihood (high-cost) threats while remaining vulnerable to high-likelihood (low-cost) queries. Mechanistically, we first provide an analysis using linear probing, revealing that while refusal mechanis
    
[^129]: 侵蚀真相默认：人类对基础模型幻觉与真实环境中虚假信息易感性的因果分析

    Eroding the Truth-Default: A Causal Analysis of Human Susceptibility to Foundation Model Hallucinations and Disinformation in the Wild

    [https://arxiv.org/abs/2601.22871](https://arxiv.org/abs/2601.22871)

    该研究通过双轴框架和结构因果模型分析918次评估发现，人类易受AI幻觉与虚假信息误导的关键因素并非政治倾向，而是缺乏“假新闻熟悉度”，且GPT-4等高流畅度输出会形成绕过来源监控的“流畅性陷阱”。

    

    随着基础模型（FMs）接近人类水平的语言流畅度，区分合成内容与真实内容已成为可信赖网络智能的关键挑战。本文提出了JudgeGPT和RogueGPT，这是一个双轴框架，将“真实性”与“归因”解耦，以研究人类易感性的机制。通过分析针对五个基础模型（包括GPT-4和Llama-2）的918次评估，我们采用结构因果模型（SCMs）作为主要框架，用于构建关于检测准确性的可检验因果假设。与党派叙事相反，我们发现政治倾向与检测性能的关联微不足道（r=-0.10）。相反，“假新闻熟悉度”成为一个候选中介变量（r=0.35），这表明对假新闻的接触可能起到类似人类判别器对抗训练的作用。我们识别出一个“流畅性陷阱”，即GPT-4的输出（人机评分：0.20）能够绕过来源监控……

    arXiv:2601.22871v3 Announce Type: replace-cross  Abstract: As foundation models (FMs) approach human-level fluency, distinguishing synthetic from organic content has become a key challenge for Trustworthy Web Intelligence. This paper presents JudgeGPT and RogueGPT, a dual-axis framework that decouples "authenticity" from "attribution" to investigate the mechanisms of human susceptibility. Analyzing 918 evaluations across five FMs (including GPT-4 and Llama-2), we employ Structural Causal Models (SCMs) as a principal framework for formulating testable causal hypotheses about detection accuracy. Contrary to partisan narratives, we find that political orientation shows a negligible association with detection performance ($r=-0.10$). Instead, "fake news familiarity" emerges as a candidate mediator ($r=0.35$), suggesting that exposure may function as adversarial training for human discriminators. We identify a "fluency trap" where GPT-4 outputs (HumanMachineScore: 0.20) bypass Source Monito
    
[^130]: Ctrl-G：在知识图谱增强的LLM推理中控制结构暴露与交互暴露

    Ctrl-G: Controlling Structural and Interaction Exposure in Knowledge Graph Augmented LLM Reasoning

    [https://arxiv.org/abs/2601.08739](https://arxiv.org/abs/2601.08739)

    提出Ctrl-G，这是首个能联合控制知识图谱增强LLM推理中结构暴露（拓扑信息）与交互暴露（多次调用累积信息）的隐私保护KG-RAG框架，同时保留多跳推理所需的桥接路径。

    

    知识图谱（KG）为基于大语言模型（LLM）的网络问答提供了结构化证据，但将检索到的证据发送至远程LLM可能会暴露敏感的图信息。现有的隐私处理方法采用实体匿名化和以关系为中心的抽象来保留推理线索，但所传输的拓扑结构仍然可见，且信息披露会随多次调用不断累积。独特的拓扑结构可能支持对被掩码实体的重识别（即结构暴露），而连续多次调用会累积额外的路径和探索痕迹，从而形成对知识图谱更全面的视图（即交互暴露）。仅仅发送更少的证据是不够的，因为受限的、分步的视图可能会遗漏多跳推理所需的桥接路径。为解决这一问题，我们提出了Ctrl-G，据我们所知，这是首个能够联合控制向远程LLM暴露的结构与交互信息、同时支持……（原文摘要在此处截断）的KG-RAG框架。

    arXiv:2601.08739v2 Announce Type: replace  Abstract: Knowledge graphs (KGs) provide structured evidence for Web question answering with large language models (LLMs), but sending retrieved evidence to a remote LLM can expose sensitive graph information. Existing privacy treatments use entity anonymization and relation-centric abstraction to preserve reasoning cues, but transmitted topology remains visible and disclosure accumulates across repeated calls. Distinctive topology can support re-identification of masked entities (structural exposure), while successive calls accumulate additional paths and exploration traces into a broader view of the KG (interaction exposure). Simply sending less evidence is insufficient because restricted, stepwise views may omit the bridge paths required for multi-hop reasoning. To address this, we propose Ctrl-G, to the best of our knowledge the first KG-RAG framework to jointly control structural and interaction exposure to a remote LLM while supporting a
    
[^131]: 自然语言演员-评论家算法：语言空间中的可扩展离策略学习

    Natural Language Actor-Critic: Scalable Off-Policy Learning in Language Space

    [https://arxiv.org/abs/2512.04601](https://arxiv.org/abs/2512.04601)

    提出自然语言演员-评论家算法NLAC，通过生成式LLM评论家产生自然语言评论来训练LLM策略，实现语言空间中的可扩展离策略学习，解决了稀疏奖励长时程任务中训练不稳定与探索困难的问题。

    

    大语言模型（LLM）智能体——即在长时程中与环境动态交互的LLM——已成为一个日益重要的研究领域，能够实现涉及工具使用、网页浏览以及与人类对话等复杂任务的自动化。在缺乏专家示范的情况下，训练LLM智能体一直依赖于策略梯度方法，即根据（通常是稀疏的）奖励函数来优化LLM策略。然而，在具有稀疏奖励的长时程任务中，基于轨迹级奖励的学习可能充满噪声，导致训练不稳定且样本复杂度很高。此外，策略改进依赖于通过探索来发现更好的动作，而当动作位于自然语言空间时，这一点可能非常困难。在本文中，我们提出了自然语言演员-评论家算法（Natural Language Actor-Critic, NLAC），这是一种新颖的actor-critic算法，它利用一个生成式的LLM评论家（critic）来产生自然语言形式的评论，从而训练LLM策略。

    arXiv:2512.04601v3 Announce Type: replace-cross  Abstract: Large language model (LLM) agents -- LLMs that dynamically interact with an environment over long horizons -- have become an increasingly important area of research, enabling automation in complex tasks involving tool-use, web browsing, and dialogue with people. In the absence of expert demonstrations, training LLM agents has relied on policy gradient methods that optimize LLM policies with respect to an (often sparse) reward function. However, in long-horizon tasks with sparse rewards, learning from trajectory-level rewards can be noisy, leading to training that is unstable and has high sample complexity. Furthermore, policy improvement hinges on discovering better actions through exploration, which can be difficult when actions lie in natural language space. In this paper, we propose Natural Language Actor-Critic (NLAC), a novel actor-critic algorithm that trains LLM policies using a generative LLM critic that produces natura
    
[^132]: 服务、团结与互助：制鞋工会与Unite Community社区工会主义的比较主题建模分析

    Service, Solidarity, and Self-Help: A Comparative Topic Modeling Analysis of Community Unionism in the Boot and Shoe Union and Unite Community

    [https://arxiv.org/abs/2509.01529](https://arxiv.org/abs/2509.01529)

    本研究运用BERTopic主题建模方法，对比分析了1920年代全国制鞋工会与2010-2020年代Unite Community的社区工会主义话语，发现Unite Community更契合外向型、社会正义导向的社区工会特征，而制鞋工会则体现了传统的服务型工会模式。

    

    本文对两个不同历史和组织背景下的社区工会主义（CU）进行了比较分析：20世纪20年代的全国制鞋工会（B&S）以及2010年代至2020年代的Unite Community。研究采用BERTopic进行主题建模并结合cTF-IDF加权，辅以词频分析，考察了各工会的话语在多大程度上契合社区工会主义的关键特征——如联盟构建、基层参与以及工作场所之外的行动。研究结果揭示了两者在主题焦点和话语连贯性上的显著差异：Unite Community与外向型、社会正义导向的主题表现出更强的契合性，而B&S语料库则强调内部行政、劳资关系和会员服务——反映出一种更为传统的、以服务为导向的工会模式。该分析还提供了方法论层面的洞见，展示了现代自然语言处理技术如何……

    arXiv:2509.01529v2 Announce Type: replace  Abstract: This paper presents a comparative analysis of community unionism (CU) in two distinct historical and organizational contexts: the National Boot and Shoe Union (B\&S) in the 1920s and Unite Community in the 2010s--2020s. Using BERTopic for thematic modeling and cTF-IDF weighting, alongside word frequency analysis, the study examines the extent to which each union's discourse aligns with key features of CU -- such as coalition-building, grassroots engagement, and action beyond the workplace. The results reveal significant differences in thematic focus and discursive coherence. While Unite Community demonstrates stronger alignment with outward-facing, social justice-oriented themes, the B\&S corpus emphasizes internal administration, industrial relations, and member services -- reflecting a more traditional, servicing-oriented union model. The analysis also highlights methodological insights, demonstrating how modern NLP techniques can 
    
[^133]: 超越黑箱：在BERTopic定量话语分析中整合词汇与语义方法

    Beyond the Black Box: Integrating Lexical and Semantic Methods in Quantitative Discourse Analysis with BERTopic

    [https://arxiv.org/abs/2508.19099](https://arxiv.org/abs/2508.19099)

    本文提出一个结合词汇与语义方法的透明混合框架，利用NLTK、spaCy、Sentence Transformers和BERTopic等技术替代黑箱软件，实现定量话语分析的三角验证、可重复性与可解释性。

    

    随着大语言模型和计算工具的兴起，定量话语分析的应用日益广泛。然而，依赖MAXQDA和NVivo等黑箱软件可能会损害方法论透明度以及与研究目标的一致性。本文提出了一个混合、透明的定量话语分析框架，结合词汇与语义方法，以实现三角验证、可重复性和可解释性。基于历史政治话语的案例研究，我们展示了使用NLTK、spaCy和Sentence Transformers构建的自定义Python流水线如何实现对预处理、词形还原和嵌入生成的细粒度控制。我们进一步详细介绍了迭代式BERTopic建模过程，其中融合了UMAP降维、HDBSCAN聚类和c-TF-IDF关键词提取，并通过参数调优和多次运行进行优化，以提升主题一致性和覆盖率。通过将精确的……（摘要原文在此处截断）

    arXiv:2508.19099v2 Announce Type: replace  Abstract: Quantitative Discourse Analysis has seen growing adoption with the rise of Large Language Models and computational tools. However, reliance on black box software such as MAXQDA and NVivo risks undermining methodological transparency and alignment with research goals. This paper presents a hybrid, transparent framework for QDA that combines lexical and semantic methods to enable triangulation, reproducibility, and interpretability. Drawing from a case study in historical political discourse, we demonstrate how custom Python pipelines using NLTK, spaCy, and Sentence Transformers allow fine-grained control over preprocessing, lemmatisation, and embedding generation. We further detail our iterative BERTopic modelling process, incorporating UMAP dimensionality reduction, HDBSCAN clustering, and c-TF-IDF keyword extraction, optimised through parameter tuning and multiple runs to enhance topic coherence and coverage. By juxtaposing precise 
    
[^134]: MedResearcher-R1：基于知识知情轨迹合成框架的专家级医学深度研究智能体

    MedResearcher-R1: Expert-Level Medical Deep Researcher via A Knowledge-Informed Trajectory Synthesis Framework

    [https://arxiv.org/abs/2508.14880](https://arxiv.org/abs/2508.14880)

    该论文提出医学深度研究智能体 MedResearcher-R1，其核心创新是知识知情轨迹合成（KISA）——通过构建医学知识图谱生成以罕见医学实体为中心的复杂多跳问答训练数据，并配合其他创新（摘要中被截断），从而突破医学领域高质量训练数据稀缺与临床推理能力不足的瓶颈，实现专家级医学深度研究。

    

    基于大语言模型（LLM）的智能体近年来在深度研究任务中展现出卓越的能力，但其在医学领域的应用仍然充满挑战。目前领先的专有系统在复杂的医学基准测试中仅能取得平庸的成绩。这暴露出两个关键瓶颈：（i）底层模型的临床推理能力有限；（ii）在仅依赖开放网络检索时，难以可靠地获取权威医学证据。为了解决这些局限并使智能体具备强大的临床推理能力，我们提出了 MedResearcher-R1——一个具有两项核心创新的医学深度研究智能体。首先，我们提出了知识知情轨迹合成，这是一种新颖的方法，通过构建医学知识图谱来生成以罕见医学实体为中心的复杂多跳问答对，从而克服高质量领域特定训练数据稀缺的问题。其次，我们……（原文摘要在此处截断，内容不完整）

    arXiv:2508.14880v4 Announce Type: replace  Abstract: Large Language Model (LLM)-based agents have recently demonstrated remarkable capabilities in deep research tasks, yet their application to the medical domain remains challenging. The leading proprietary systems achieve only modest results on complex medical benchmarks. This exposes two key bottlenecks: (i) limited clinical-reasoning capability in the underlying model, and (ii) unreliable access to authoritative medical evidence when constrained to open-web retrieval. To address these limitations and equip agents with robust clinical reasoning capabilities, we present MedResearcher-R1, a medical deep research agent featuring two core innovations. First, we propose Knowledge-Informed Trajectory Synthesis (KISA), a novel approach that builds medical knowledge graphs to construct complex multi-hop question-answer pairs centered on rare medical entities, overcoming the scarcity of high-quality domain-specific training data. Second, we in
    
[^135]: 基于预算相对策略优化的任意时刻推理优化

    Optimizing Anytime Reasoning via Budget Relative Policy Optimization

    [https://arxiv.org/abs/2505.13438](https://arxiv.org/abs/2505.13438)

    提出AnytimeReasoner框架，通过截断思考过程并引入可验证的密集奖励，结合预算相对策略优化，优化大语言模型在不同token预算约束下的任意时刻推理性能，提升token效率与部署灵活性。

    

    扩展测试时计算对于增强大型语言模型（LLM）的推理能力至关重要。现有方法通常采用强化学习（RL）来最大化在推理轨迹末端获得的可验证奖励。然而，这类方法仅在一个较大且固定的token预算下优化最终性能，这阻碍了训练和部署的效率。在本工作中，我们提出了一个新颖框架——AnytimeReasoner，用于优化任意时刻的推理性能，旨在提高token效率以及在不同token预算约束下推理的灵活性。为实现这一目标，我们将完整的思考过程截断以适应从先验分布中采样的token预算，迫使模型为每个截断的思考总结出最优答案以供验证。这为推理过程引入了可验证的密集奖励，有助于更有效的信用分配。

    arXiv:2505.13438v4 Announce Type: replace-cross  Abstract: Scaling test-time compute is crucial for enhancing the reasoning capabilities of large language models (LLMs). Existing approaches typically employ reinforcement learning (RL) to maximize a verifiable reward obtained at the end of reasoning traces. However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency in both training and deployment. In this work, we present a novel framework, AnytimeReasoner, to optimize anytime reasoning performance, which aims to improve token efficiency and the flexibility of reasoning under varying token budget constraints. To achieve this, we truncate the complete thinking process to fit within sampled token budgets from a prior distribution, compelling the model to summarize the optimal answer for each truncated thinking for verification. This introduces verifiable dense rewards into the reasoning process, facilitating more effective cred
    
[^136]: 超越公开访问：探究大语言模型预训练数据中的非公开内容

    Beyond Public Access in LLM Pre-Training Data

    [https://arxiv.org/abs/2505.00020](https://arxiv.org/abs/2505.00020)

    研究者利用DE-COP成员推断攻击检测版权书籍内容，发现GPT-4o表现出对其未获授权训练的非公开付费内容的识别能力（AUROC达0.82），这暗示可能存在超越“训练数据包含”之外的数据访问违规行为。

    

    我们使用合法获取的包含34本O'Reilly Media版权书籍（共13,962个段落）的数据集，应用DE-COP成员推断攻击来测试OpenAI模型是否表现出对其未获授权训练的非公开（付费墙）内容的识别。我们的主要贡献在于研究非公开书籍，这使我们能够考虑潜在的数据访问违规行为，而不仅仅是训练数据的包含问题。我们的结果表明，GPT-4o（我们样本中OpenAI更新、能力更强的模型）表现出与识别非公开书籍一致的模式，其AUROC分数为0.82（95%自助法置信区间：0.60-0.96）——高于50%即优于随机猜测。而GPT-3.5 Turbo（两年前训练）和较小的GPT-4o Mini在非公开数据上的识别分数均接近随机水平（分别为0.54和0.56）。结果在使用独立改写模型（Gemma和Qwen）以及留一书法的敏感性分析中均保持稳健。我们还测试了GP

    arXiv:2505.00020v3 Announce Type: replace  Abstract: Using a legally obtained dataset of 34 copyrighted O'Reilly Media books (13,962 paragraphs), we apply the DE-COP membership inference attack to test whether OpenAI models show recognition of non-public (pay-walled) content they were not licensed to train on. Our main contribution is examining non-public books, which allows us to consider potential access violations and not just training inclusion. Our results suggest that GPT-4o, OpenAI's more recent and capable model in our sample, exhibits patterns consistent with recognition of non-public books, with an AUROC score of 0.82 (95% bootstrapped CI: 0.60-0.96) -- where above 50% is better than random chance. GPT-3.5 Turbo (trained two years prior) and the smaller GPT-4o Mini both score near chance recognition on non-public data (0.54 and 0.56, respectively). Results are robust to independent paraphrasers (Gemma and Qwen) and leave-one-book-out sensitivity analyses. We also test both GP
    

