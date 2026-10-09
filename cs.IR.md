# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Compact and Efficient Indexes for Learned Sparse Retrieval](https://arxiv.org/abs/2610.12300) | 本文通过在倒排索引中用文档中心点替代每块摘要、并在正排索引中采用SIMD友好的位打包与4比特量化压缩技术，在不牺牲检索效率的情况下大幅降低了学习型稀疏检索索引的内存占用。 |
| [^2] | [Syn-Omni: Structured Specialization and Progressive Collaboration for Omnimodal Embeddings](https://arxiv.org/abs/2610.12256) | 提出Syn-Omni框架，通过正交模态专家LoRA（OME-LoRA）将通用语义与模态特化表示进行结构化分离，并借助渐进式协同路由（PSR）实现从模态特化到跨模态协同的渐进式专家交互，从而提升全模态嵌入效果。 |
| [^3] | [NativeScope: Relation-Localized Retrieval over Native Topology with a Correct Anchor](https://arxiv.org/abs/2610.12243) | 论文提出 NativeScope，利用已知锚点和关系在数据系统原生结构（章节归属、会话边界、顺序）上先定位范围再排序，相比全范围稠密 RAG 将文档和记忆的原生单元召回率分别提升 42.75 和 22.00 个百分点。 |
| [^4] | [Project Greenhouse: Progress Toward Fully Open and Sovereign Agentic Search](https://arxiv.org/abs/2610.11922) | 温室计划证明，仅用适度计算资源和公开数据集，通过从零预训练加有监督微调的两步简单流程、不依赖第三方骨干模型，即可构建完全开放且自主可控、具有竞争力的智能体搜索重排序模型。 |
| [^5] | [Chaos in the Text: Revealing the Modality Preference in Mixed-Modality Retrievers](https://arxiv.org/abs/2610.11816) | 该研究揭示了混合模态检索器普遍存在“模态偏好”问题：文本表示被系统性地赋予更高相似度分数，导致无关文本比无关图像对检索性能造成更严重的干扰，即“文本中的混沌”现象。 |
| [^6] | [Autoregressive Retriever: Improving Query Understanding from Item Feedback for Universal Multimodal Retrieval](https://arxiv.org/abs/2610.11666) | 提出自回归检索器ARR，通过交替检索物品并利用物品反馈迭代更新查询嵌入，结合监督微调与强化学习优化物品选择，显著提升了通用多模态检索中的查询理解能力。 |
| [^7] | [SkillContrast: Difference-Guided Text Selection for Agent Skill Reranking](https://arxiv.org/abs/2610.11650) | SkillContrast 是一种无需训练的技能文本选择方法，通过比较相似检索技能之间的差异文本（而非共享指令）来提升智能体技能重排序效果，在输入标记减少 51.1-58.8% 的同时，比 TF-IDF 查询选择多获得 54-72 个安全检索的干净命中。 |
| [^8] | [Overview of the NTCIR-19 Automatic Evaluation of LLMs 2 (AEOLLM-2) Task](https://arxiv.org/abs/2610.11598) | NTCIR-19 AEOLLM-2 任务新增了针对大语言模型生成的长篇深度研究报告的自动评估子任务，共收到 10 个团队的 91 份提交，各方法的性能通过与人工标注标签的对比来衡量。 |
| [^9] | [EVIE: Evidence-Vector-Informed Embeddings for Visual Document Retrieval](https://arxiv.org/abs/2610.11553) | EVIE提出证据向量感知的嵌入方法，通过在表示学习和索引构建全过程中保留与查询相关的页面证据，实现了兼顾细粒度页面理解与高效索引的视觉文档检索。 |
| [^10] | [Compactness and Consistency: A Conjoint Framework for Deep Graph Clustering](https://arxiv.org/abs/2610.11506) | 本文提出联合框架CoCo，利用图卷积滤波器从局部和全局两种视角学习鲁棒表示，并将其编码为低秩紧凑形式，从而在深度图聚类中同时捕获节点表示的紧凑性与一致性，克服了GNN局部消息传递难以建模全局关系以及图数据噪声冗余的问题。 |
| [^11] | [Beyond Resolution: Object-to-Image Ratio Mismatch in Instance Retrieval](https://arxiv.org/abs/2610.11489) | 该论文发现视觉实例检索性能下降的主因并非分辨率损失，而是查询与图库图像间物体占画面比例的失配，并据此提出查询侧尺度增强与OWLv2裁剪重排序方法，在无需训练或修改图库的情况下于ILIAS 100M数据集上取得最先进结果。 |
| [^12] | [SAIL: Scientific Agentic Intelligence via a Science-Aware Loop](https://arxiv.org/abs/2610.11451) | SAIL是一个通过“科学感知改进循环”训练的开放科学智能体模型（35B总参数/3B激活参数），由前沿AI智能体自动诊断其任务失败并生成针对性训练任务，使其在文献研究、科学编程和多步骤研究工作流中达到有竞争力的表现。 |
| [^13] | [On-Chain Archaeology of Bitcoin Oracles: Evidence of Use under Limited Observability](https://arxiv.org/abs/2610.11439) | 该研究首次对比特币预言机的链上使用历史进行系统性“考古”，通过投注普查、全链分析与公钥索引搜索发现：早期链上合约虽已留存，但其事件描述大量消失，而现代DLC的公开预言机公告即使合约本身无法被识别也可能长期留存。 |
| [^14] | [RIT-RAG: Navigating Document Corpora with Retrieval-Induced Trees](https://arxiv.org/abs/2610.11370) | RIT-RAG将内容检索与文档结构导航相结合，离线为每篇文档构建目录树，查询时先检索宽泛的文本块并利用其位置诱导出可跨多个文档的、规模可控的子树，再由LLM智能体在子树中导航，从而在保留结构感知优势的同时克服了现有方法无法扩展到大型语料库且选错文档后无法恢复的缺陷。 |
| [^15] | [H2CE: Modeling Geo-Semantic Interactions for POI Reranking with Heterogeneous Two-Stage Cross-Encoders](https://arxiv.org/abs/2610.11277) | H2CE提出了一种异构两阶段交叉编码器，通过将分桶自然语言描述与经MLP处理的精确标量值相结合并在潜空间中融合语义与数值嵌入，建模语义、地理邻近性与数值质量信号之间的非线性交互，从而实现低延迟的POI重排序。 |
| [^16] | [Gated Memory: Admission-Controlled Memory Formation for Conversational AI](https://arxiv.org/abs/2610.11270) | 该论文提出Gated Memory框架，通过在对话与存储之间设置准入控制检查点，在事实提取前基于完整话语上下文评估候选事实，解决了关键上下文信号在提取时不可逆丢失这一制约记忆质量的瓶颈问题。 |
| [^17] | [Learning Multi-Step Query Rewriting via Corpus Feedback for Conversational Search](https://arxiv.org/abs/2610.10955) | 该论文将对话式查询重写重新构建为顺序检索问题，提出一个基于语料库反馈进行多步迭代的重写智能体，通过监督微调与强化学习训练，在无需人工重写标注的情况下显著提升了检索质量。 |
| [^18] | [Language Models for Page-Level Layout Decisions in E-commerce Search](https://arxiv.org/abs/2610.10920) | 该论文研究了利用语言模型离线评估电商搜索页面级布局决策（如在特定位置插入二级堆栈）是否对用户有益，从而减少对昂贵在线A/B测试的依赖。 |
| [^19] | [LIFT: A Lifecycle-aware Interaction Factorization Transformer for Unified Retrieval and Ranking](https://arxiv.org/abs/2610.10556) | LIFT将用户交互分解为请求、物品、上下文、动作四个生命周期状态并建模为因果序列，使检索与排序任务共享历史建模的同时保留各自阶段特定信息，在ML-20M和淘宝数据集上分别超越最强基线4.9%和3.6%。 |
| [^20] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^21] | [RecToolBench: Benchmarking Recommendation-Specific Tool Orchestration under Fuzzy User Intent](https://arxiv.org/abs/2609.30717) | 该论文提出了RecToolBench，一个基于MCP协议、用于评估推荐智能体在模糊用户意图下进行工具编排能力的基准测试，包含超过1,200个可执行任务，覆盖单工具调用、并行调用、顺序工具链和混合编排等多种复杂工具使用模式。 |
| [^22] | [MERGED: Multimodal Entity Resolution via Generated Expert Reasoning Distillation](https://arxiv.org/abs/2609.01913) | MERGED框架让多个大型视觉-语言模型教师为商品对标注并阐述推理，将一致标注用于监督微调、分歧标注经元裁判转化为直接偏好优化的偏好对，从而把结构化推理蒸馏到无需人工标注的7B紧凑学生模型中，实现了生产规模下低成本、低延迟的多模态商品实体解析。 |
| [^23] | [GLM-RAG: Graph Language Models for Graph-Based Retrieval-Augmented Generation](https://arxiv.org/abs/2607.28397) | 本文提出了一种基于图语言模型（GLM）的检索器用于知识图谱检索增强生成，发现微调后的GLM检索器在跨领域泛化能力上优于GNN和向量搜索检索器，并在两个多跳基准上达到SOTA。 |
| [^24] | [Right Family, Wrong Skill: Benchmarking Risk Exposure in Agent Skill Retrieval](https://arxiv.org/abs/2606.10388) | 本文提出了SameCapRisk-Bench基准，专门研究智能体技能检索中“找到正确技能族但暴露错误同族技能”的风险暴露失败，并提供了1,190个技能-风险单元和1,686个评估查询案例。 |
| [^25] | [Generative Spatiotemporal Intent Sequence Recommendation via Implicit Reasoning in Amap](https://arxiv.org/abs/2605.28888) | 高德地图提出GPlan框架，通过渐进式隐式思维链蒸馏将大语言模型的推理能力内化到轻量级模型中，在严格延迟约束下实现逻辑连贯且物理可执行的生成式时空意图序列推荐。 |
| [^26] | [Efficient and Scalable Provenance Tracking for LLM-Generated Code Snippets](https://arxiv.org/abs/2605.28510) | 该论文提出SourceTracker编码器与HybridSourceTracker两阶段混合流水线，先用向量搜索缩小候选范围、再用Winnowing指纹精确重排，从而实现对LLM生成代码在数十亿级训练语料上的高效可扩展溯源追踪。 |
| [^27] | [Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](https://arxiv.org/abs/2603.07233) | 提出了PT-RAG，一个即插即用的两阶段检索增强生成模块，通过GenePT语义检索与可微分Gumbel-Softmax选择器动态获取相关的扰动上下文，从而改进单细胞基因扰动响应的预测。 |
| [^28] | [LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling](https://arxiv.org/abs/2510.18239) | LIME架构通过低秩链接嵌入解耦用户与候选交互、实现注意力权重预计算，并采用线性XOR注意力机制，从根本上降低了推荐系统推理时对候选集大小和用户序列长度的计算成本依赖。 |

# 详细

[^1]: 面向学习型稀疏检索的紧凑高效索引

    Compact and Efficient Indexes for Learned Sparse Retrieval

    [https://arxiv.org/abs/2610.12300](https://arxiv.org/abs/2610.12300)

    本文通过在倒排索引中用文档中心点替代每块摘要、并在正排索引中采用SIMD友好的位打包与4比特量化压缩技术，在不牺牲检索效率的情况下大幅降低了学习型稀疏检索索引的内存占用。

    

    本文研究如何在不牺牲最先进检索数据结构效率的前提下，大幅降低学习型稀疏检索索引的内存占用。基于SEISMIC，我们重新审视了其设计的两个层面：用于筛选候选文档的倒排索引和用于对候选文档评分的正排索引。对于倒排索引，我们用中心点替换代价高昂的每块摘要——即选取现有文档作为块代表——从而将每块元数据从稀疏向量压缩为单个文档标识符。对于正排索引，我们对组件和值都进行了压缩：通过重排词汇表使共同出现的组件彼此靠近，并用DOTPACKING8（一种SIMD友好的位打包方案，将解压缩与点积评估融合为一体）对得到的Δ-间隙进行编码；对于值，则使用根据每个组件分布拟合的紧凑型4比特按组件码本进行量化。

    arXiv:2610.12300v1 Announce Type: new  Abstract: This paper investigates how to substantially reduce the memory footprint of learned sparse retrieval indexes without sacrificing the efficiency of state-of-the-art retrieval data structures. Building on SEISMIC, we revisit both levels of its design: the inverted index used to select candidates and the forward index used to score them. For the inverted index, we replace costly per-block summaries with medoids, namely existing documents elected as block representatives, collapsing the per-block metadata from a sparse vector to a single document identifier. For the forward index, we compress both components and values. We reorder the vocabulary to place co-occurring components closer together and encode the resulting $\Delta$-gaps with DOTPACKING8, a SIMD-friendly bit-packing scheme that fuses decompression with dot-product evaluation; values are quantized with compact per-component 4-bit codebooks fitted to each component's distribution. W
    
[^2]: Syn-Omni：面向全模态嵌入的结构化特化与渐进式协作

    Syn-Omni: Structured Specialization and Progressive Collaboration for Omnimodal Embeddings

    [https://arxiv.org/abs/2610.12256](https://arxiv.org/abs/2610.12256)

    提出Syn-Omni框架，通过正交模态专家LoRA（OME-LoRA）将通用语义与模态特化表示进行结构化分离，并借助渐进式协同路由（PSR）实现从模态特化到跨模态协同的渐进式专家交互，从而提升全模态嵌入效果。

    

    全模态嵌入天然涉及跨异构输入的共享表示与模态特定特征。然而，现有的全模态嵌入方法通常依赖于在混合模态数据上使用单一共享参数空间，这限制了通用表示与模态特定表示之间的结构性分离。为解决这一问题，我们提出了Syn-Omni，一个实现模态特化与受控跨模态协作的结构化全模态适配统一框架。具体而言，我们引入了正交模态专家LoRA（OME-LoRA），将适配分解为用于通用语义的共享LoRA路径与用于模态感知特化的模态专家LoRA路径。此外，渐进式协同路由（PSR）使各模态专家先建立模态特定的先验知识，随后逐渐与其他模态专家交互以实现跨模态协同。在涵盖图像、视频等81个多样化任务的评估中……

    arXiv:2610.12256v1 Announce Type: cross  Abstract: Omnimodal embeddings naturally involve both shared representations and modality-specific features across heterogeneous inputs. However, existing omnimodal embedding methods often rely on a single shared parameter space over mixed-modality data, limiting structural separation between universal and modality-specific representations. To address this, we propose Syn-Omni, a unified framework for structured omnimodal adaptation with modality specialization and controlled cross-modal collaboration. Specifically, we introduce Orthogonal Modality-Expert LoRA (OME-LoRA), which decomposes adaptation into a shared LoRA path for universal semantics and modality-expert LoRA paths for modality-aware specialization. Furthermore, Progressive Synergy Routing (PSR) enables experts to first establish modality-specific priors, then gradually interact with other modality-experts for cross-modal synergy. Evaluated across 81 diverse tasks spanning image, vid
    
[^3]: NativeScope：基于正确锚点在原生拓扑上进行关系定位的检索

    NativeScope: Relation-Localized Retrieval over Native Topology with a Correct Anchor

    [https://arxiv.org/abs/2610.12243](https://arxiv.org/abs/2610.12243)

    论文提出 NativeScope，利用已知锚点和关系在数据系统原生结构（章节归属、会话边界、顺序）上先定位范围再排序，相比全范围稠密 RAG 将文档和记忆的原生单元召回率分别提升 42.75 和 22.00 个百分点。

    

    稠密检索通常根据文本块与问题的语义相似度对其进行排序，这忽略了许多数据系统已经存储的结构信息，包括章节归属、会话边界和原生顺序。我们提出 NativeScope，一种针对具有已知锚点和关系的查询的“先定范围后排序”方法。该方法将查询表示为 q -> (A, r, B)：锚点 A 和关系 r 通过“属于”、“之前”或“之后”算子选择原生单元，目标词 B 仅对与所选范围重叠的文本块进行排序。其内部变体 NS-FullQ 则使用完整问题对相同的候选进行排序。我们在从 QASPER 和 LongMemEval 导出的 200 条受控文档与记忆记录上，在 1,024 个 token 的预算下评估了这两种方法。NativeScope 在文档和记忆上的原生单元召回率分别达到 89.28% 和 72.50%，比实例级全范围稠密 RAG 分别提升 42.75 和 22.00 个百分点。NS-FullQ 达到 87.78% 和……（摘要在此处截断）

    arXiv:2610.12243v1 Announce Type: cross  Abstract: Dense retrieval usually ranks text chunks by their semantic similarity to a question. This ignores structure that many data systems already store, including section membership, session boundaries, and native order. We propose NativeScope, a scope-then-rank method for queries with a known anchor and relation. It represents a query as q -> (A, r, B). The anchor A and relation r select native units through belonging, before, or after operators, and the target term B ranks only chunks that overlap the selected scope. An internal variant, NS-FullQ, ranks the same candidates with the full question. We evaluate both methods on 200 controlled document and memory records derived from QASPER and LongMemEval under a 1,024-token budget. NativeScope attains native-unit recall of 89.28 percent for documents and 72.50 percent for memories, improving over instance-wide Dense RAG by 42.75 and 22.00 percentage points. NS-FullQ reaches 87.78 percent and 
    
[^4]: 温室计划：迈向完全开放与自主可控的智能体搜索

    Project Greenhouse: Progress Toward Fully Open and Sovereign Agentic Search

    [https://arxiv.org/abs/2610.11922](https://arxiv.org/abs/2610.11922)

    温室计划证明，仅用适度计算资源和公开数据集，通过从零预训练加有监督微调的两步简单流程、不依赖第三方骨干模型，即可构建完全开放且自主可控、具有竞争力的智能体搜索重排序模型。

    

    温室计划是我们对一个简单命题的探索：我们相信，仅需适度的计算资源，就有可能构建完全开放且自主可控的智能体搜索模型。作为第一个里程碑，我们描述了如何构建一个具有竞争力的逐点式仅解码器重排序模型，其方法是一个简单的两步流程：从零开始的预训练，随后进行有监督微调，且仅从公开可用的数据集出发。与文献中的主流方法不同，我们不依赖第三方的现成开源权重骨干模型，因此我们对模型训练拥有端到端的完全掌控。我们仅使用少量GPU就完成了大部分实验。本报告阐述了该方法的重要性和优势，并分享了相关成果，使模型训练的各个方面都能够被透明、独立地复现。

    arXiv:2610.11922v1 Announce Type: cross  Abstract: Project Greenhouse represents our exploration of a simple thesis: We believe that it is possible to build fully open and sovereign models for agentic search with only modest computational resources. As a first milestone, we describe how to build a competitive pointwise decoder-only reranker using a simple two-step recipe comprising pre-training from scratch followed by supervised fine-tuning, starting only from commonly available datasets. Contrary to the dominant approach in the literature, we do not rely on existing open-weight backbones from third parties, and thus we are fully in control of model training, from end to end. We were able to accomplish the bulk of our experiments using no more than a handful of GPUs. This report articulates the importance and benefits of our approach, and we share artifacts that enable transparent, independent reproduction of all aspects of model training. Beyond data, code, and configurations that ca
    
[^5]: 文本中的混沌：揭示混合模态检索器中的模态偏好

    Chaos in the Text: Revealing the Modality Preference in Mixed-Modality Retrievers

    [https://arxiv.org/abs/2610.11816](https://arxiv.org/abs/2610.11816)

    该研究揭示了混合模态检索器普遍存在“模态偏好”问题：文本表示被系统性地赋予更高相似度分数，导致无关文本比无关图像对检索性能造成更严重的干扰，即“文本中的混沌”现象。

    

    密集检索器在文本和图像语料库上已取得显著进展，但这些能力能否可靠地扩展到包含文本、图像以及文本-图像融合文档的混合语料库仍不清楚。在本文中，我们系统地研究了跨多种架构的检索器，发现其性能对模态组成高度敏感。当图像文档逐渐被语义对应的文本表示所替代时，检索性能呈现明显的V型曲线——在单模态语料库上表现强劲，但在模态共存时大幅下降。尤其是，无关文本比同等数量的无关图像造成更严重的性能下降，我们将这一现象称为“文本中的混沌”。进一步分析揭示了模态偏好现象：文本表示会系统性地获得更高的相似度分数，使得无关文本能够排在相关图像之前。为了缓解……

    arXiv:2610.11816v1 Announce Type: new  Abstract: Dense retrievers have made significant progress on text and image corpora, but whether these capabilities extend reliably to mixed corpora containing text, image, and fused text-image documents remains unclear. In this paper, we systematically examine retrievers across architectures and find that their performance is highly sensitive to modality composition. As image documents are progressively replaced with semantically corresponding text representations, retrieval performance follows a pronounced V-shaped curve, remaining strong on single-modality corpora but degrading substantially when modalities coexist. In particular, irrelevant text causes more severe degradation than an equal number of irrelevant images, a phenomenon we term Chaos in the Text. Further analysis reveals modality preference, whereby text representations receive systematically higher similarity scores, allowing irrelevant text to outrank relevant images. To mitigate 
    
[^6]: 自回归检索器：利用物品反馈改进通用多模态检索的查询理解

    Autoregressive Retriever: Improving Query Understanding from Item Feedback for Universal Multimodal Retrieval

    [https://arxiv.org/abs/2610.11666](https://arxiv.org/abs/2610.11666)

    提出自回归检索器ARR，通过交替检索物品并利用物品反馈迭代更新查询嵌入，结合监督微调与强化学习优化物品选择，显著提升了通用多模态检索中的查询理解能力。

    

    通用多模态检索通常只对查询进行一次编码，然后通过嵌入相似度对独立索引的物品进行排序。这种设计支持高效搜索，但即使检索到的物品可以帮助澄清信息需求，查询表示也保持不变。我们提出了自回归检索器（ARR），这是一种多模态检索模型，它学习选择信息丰富的物品，并利用其内容来优化后续检索。ARR在检索物品和更新查询嵌入之间交替进行，然后使用最终的嵌入对整个集合进行排序。监督微调通过逐步对比监督教会编码器使用反馈。强化学习将反馈物品视为动作，并利用相关物品的最终倒数排名来优化其选择。查询侧适配器使得可以在固定的物品索引上进行这种优化。ARR展现出强大的检索性能。

    arXiv:2610.11666v1 Announce Type: new  Abstract: Universal multimodal retrieval typically encodes a query once and ranks independently indexed items by embedding similarity. This design supports efficient search, but leaves the query representation unchanged even when retrieved items could help clarify the information need. We introduce the AutoRegressive Retriever (ARR), a multimodal retrieval model that learns both to select informative items and to use their content to refine subsequent retrieval. ARR alternates between retrieving an item and updating the query embedding, then uses the final embedding to rank the collection. Supervised fine-tuning teaches the encoder to use feedback through stepwise contrastive supervision. Reinforcement learning treats feedback items as actions and optimizes their selection using the final reciprocal rank of a relevant item. A query-side adapter enables this optimization against a fixed item index. ARR demonstrates strong retrieval performance on b
    
[^7]: SkillContrast：面向智能体技能重排序的差异引导文本选择

    SkillContrast: Difference-Guided Text Selection for Agent Skill Reranking

    [https://arxiv.org/abs/2610.11650](https://arxiv.org/abs/2610.11650)

    SkillContrast 是一种无需训练的技能文本选择方法，通过比较相似检索技能之间的差异文本（而非共享指令）来提升智能体技能重排序效果，在输入标记减少 51.1-58.8% 的同时，比 TF-IDF 查询选择多获得 54-72 个安全检索的干净命中。

    

    相似的智能体技能可能共享相同的指令，但在使用条件上有所不同。基于查询的文本选择可能会保留共享的指令而遗漏这些差异。我们提出了 SkillContrast，这是一种无需训练的选择器，它通过比较检索到的技能，保留它们之间不同的文本及局部上下文，供预训练重排序器使用。在 SameCapRisk-Bench 的 1,235 个请求上，在相同的每候选输入长度下，该方法在 2 种检索器和 2 种重排序器规模上比 TF-IDF 查询选择多获得 54-72 个“干净命中”（即检索到有用技能的同时不包含其被标记为风险的同类技能）。长度匹配的组件替换实验表明，差异文本是主要设置中的核心贡献因素，而上下文的效果较小且表现不一。相对于完整的技能正文，SkillContrast 使用的模型输入标记减少了 51.1-58.8%，在 0.6B 模型下干净命中数减少 10-18 个，而在 4B 模型下观察到的干净命中数持平或更高。

    arXiv:2610.11650v1 Announce Type: cross  Abstract: Similar agent skills can share instructions but differ in their conditions of use. Query-based text selection may retain shared instructions and omit these distinctions. We introduce SkillContrast, a training-free selector that compares retrieved skills and retains their differing text with local context for a pretrained reranker. On 1,235 requests from SameCapRisk-Bench, it yields 54-72 more clean hits (requests that retrieve a helpful skill without its marked risky sibling) than TF-IDF query selection at identical per-candidate input lengths, across 2 retrievers and 2 reranker sizes. Length-matched component replacements identify differing text as the main contributor in the primary setting, with smaller, mixed context effects. Relative to full skill bodies, SkillContrast uses 51.1-58.8% fewer model-input tokens, with 10-18 fewer clean hits at 0.6B and matching or higher observed clean-hit counts at 4B. Candidate-relative differences
    
[^8]: NTCIR-19 大语言模型自动评估 2（AEOLLM-2）任务概述

    Overview of the NTCIR-19 Automatic Evaluation of LLMs 2 (AEOLLM-2) Task

    [https://arxiv.org/abs/2610.11598](https://arxiv.org/abs/2610.11598)

    NTCIR-19 AEOLLM-2 任务新增了针对大语言模型生成的长篇深度研究报告的自动评估子任务，共收到 10 个团队的 91 份提交，各方法的性能通过与人工标注标签的对比来衡量。

    

    本文概述了 NTCIR-19 大语言模型自动评估 2（AEOLLM-2）任务。在 NTCIR-18 核心任务 AEOLLM 成功举办的基础上，我们为 NTCIR-19 提出了 AEOLLM-2，以进一步研究大语言模型（LLM）的自动评估方法，特别是针对长文本生成场景。在 AEOLLM-2 中，我们引入了一个新的子任务——深度研究评估（Deep Research Evaluation），专注于对 LLM 生成的长篇深度研究报告进行自动评估。参赛者开发了自动评估这些报告质量的方法，每种方法的性能通过将其评分与人工标注的真实标签进行对比来衡量。今年，我们共收到了来自 10 个团队的 91 份提交结果。本文介绍了该任务的背景、数据集构建、评估指标、参赛者的方法以及最终评估结果。

    arXiv:2610.11598v1 Announce Type: new  Abstract: In this paper, we provide an overview of the NTCIR-19 Automatic Evaluation of LLMs 2 (AEOLLM-2) task. Building on the success of the NTCIR-18 core task AEOLLM, we proposed AEOLLM-2 for NTCIR-19 to further investigate automatic evaluation methods for Large Language Models (LLMs), particularly in long-form text generation scenarios. In AEOLLM-2, we introduced a new subtask, Deep Research Evaluation, which focuses on the automatic evaluation of long-form deep research reports generated by LLMs. Participants developed evaluation methods to automatically assess the quality of these reports, and the performance of each method was measured by comparing its scores against human-annotated ground-truth labels. This year, we received 91 runs from 10 teams in total. This paper describes the background of the task, the dataset construction, the evaluation measures, the participants' methods, and the final evaluation results.
    
[^9]: EVIE：面向视觉文档检索的证据向量感知嵌入

    EVIE: Evidence-Vector-Informed Embeddings for Visual Document Retrieval

    [https://arxiv.org/abs/2610.11553](https://arxiv.org/abs/2610.11553)

    EVIE提出证据向量感知的嵌入方法，通过在表示学习和索引构建全过程中保留与查询相关的页面证据，实现了兼顾细粒度页面理解与高效索引的视觉文档检索。

    

    准确且可扩展的视觉文档检索（VDR）同时需要细粒度的页面理解和高效的索引，然而现有方法难以兼顾两者。基于OCR的文本检索会增加预处理延迟，并可能丢失理解复杂页面所需的视觉和结构线索。单向量视觉-语言模型虽然绕过了OCR，但将整个页面压缩为单个向量限制了查询与文档匹配的粒度。采用MaxSim的多向量检索器提供了更细粒度的交互，但需要庞大的索引，且准确率仍有提升空间。我们认为，克服这些限制需要在表示学习和索引构建的全过程中保留与查询相关的页面证据。为此，我们提出了EVIE（证据向量感知嵌入），这是一系列原生视觉文档检索器，集成了三项关键创新：（1）证据判定的数据治理，它……（摘要在此处截断）

    arXiv:2610.11553v1 Announce Type: new  Abstract: Accurate and scalable visual document retrieval (VDR) requires both fine-grained page understanding and efficient indexing, yet existing approaches struggle to achieve both. OCR-based text retrieval adds preprocessing latency and can lose visual and structural cues needed to understand complex pages. Single-vector vision-language models bypass OCR, but compressing an entire page into one vector limits the granularity of query--document matching. Multi-vector retrievers with MaxSim provide finer interactions, yet demand large indexes and still leave room for accuracy improvements. We argue that overcoming these limitations requires preserving query-relevant page evidence throughout representation learning and index construction. To this end, we introduce \textbf{\textit{EVIE}} (Evidence-Vector-Informed Embeddings), a family of native visual document retrievers integrating three key innovations: (1) Evidence-judged data governance, which u
    
[^10]: 紧凑性与一致性：面向深度图聚类的联合框架

    Compactness and Consistency: A Conjoint Framework for Deep Graph Clustering

    [https://arxiv.org/abs/2610.11506](https://arxiv.org/abs/2610.11506)

    本文提出联合框架CoCo，利用图卷积滤波器从局部和全局两种视角学习鲁棒表示，并将其编码为低秩紧凑形式，从而在深度图聚类中同时捕获节点表示的紧凑性与一致性，克服了GNN局部消息传递难以建模全局关系以及图数据噪声冗余的问题。

    

    图聚类是数据分析中的一项基础任务，旨在将图中具有相似特征的节点划分到同一簇中。由于图神经网络（GNN）能够利用节点属性和图拓扑结构来实现有效的簇分配，这一问题得到了广泛研究。然而，通过GNN学习的节点表示通常难以通过局部消息传递机制捕获节点之间的全局关系。此外，图数据中固有的冗余和噪声很容易导致节点表示缺乏紧凑性和鲁棒性。为了解决这些问题，我们提出了一个联合框架CoCo，它为深度图聚类在学习到的节点表示中捕获紧凑性与一致性。在技术上，我们的CoCo利用图卷积滤波器从局部和全局两种视角学习鲁棒的节点表示，然后将其编码为低秩紧凑表示。

    arXiv:2610.11506v1 Announce Type: cross  Abstract: Graph clustering is a fundamental task in data analysis, aiming at grouping nodes with similar characteristics in the graph into clusters. This problem has been widely explored using graph neural networks (GNNs) due to their ability to leverage node attributes and graph topology for effective cluster assignments. However, representations learned through GNNs typically struggle to capture global relationships between nodes via local message-passing mechanisms. Moreover, the redundancy and noise inherently present in graph data may easily result in node representations lacking compactness and robustness. To address these issues, we propose a conjoint framework CoCo, which captures compactness and consistency in the learned node representations for deep graph clustering. Technically, our CoCo leverages graph convolutional filters to learn robust node representations from both local and global views, and then encodes them into low-rank com
    
[^11]: 超越分辨率：实例检索中的对象-图像比例失配问题

    Beyond Resolution: Object-to-Image Ratio Mismatch in Instance Retrieval

    [https://arxiv.org/abs/2610.11489](https://arxiv.org/abs/2610.11489)

    该论文发现视觉实例检索性能下降的主因并非分辨率损失，而是查询与图库图像间物体占画面比例的失配，并据此提出查询侧尺度增强与OWLv2裁剪重排序方法，在无需训练或修改图库的情况下于ILIAS 100M数据集上取得最先进结果。

    

    视觉实例检索在相同物体于查询图像和图库图像中以不同表观大小出现时常常失败。我们证明，主要原因通常不是分辨率损失，而是对象-图像比例失配：即物体在两幅图像中所占的画面比例不同。在一个由3,021个Objaverse物体在五个相机距离下渲染的受控基准测试中，对于12个预训练骨干网络中的9个，超过80%的跨距离性能下降可归因于O2I比例失配而非分辨率；多尺度架构虽将纯分辨率效应降至个位数百分比，却仍同样易受O2I失配影响。这种失败还是不对称的：紧凑的查询图像对宽幅图库图像的检索可靠性高于相反情况。基于这一分析，查询侧尺度增强与OWLv2裁剪重排序器在ILIAS 100M数据集上达到了最先进的性能（重排序前29.2 mAP@1000，重排序后42.0），且无需训练或修改预计算的图库。

    arXiv:2610.11489v1 Announce Type: cross  Abstract: Visual instance retrieval often fails when the same object appears at different apparent sizes in the query and gallery. We show that the dominant cause is usually not resolution loss but object-to-image (O2I) ratio mismatch: the object occupies different fractions of the two images. On a controlled benchmark of 3,021 Objaverse objects rendered at five camera distances, more than 80% of the cross-distance degradation is attributable to O2I mismatch rather than resolution for 9 of 12 pretrained backbones; multi-scale architectures cut the resolution-only effect to single digits yet remain equally susceptible. The failure is also asymmetric: tight queries retrieve more reliably against wide gallery images than the reverse. Guided by this analysis, query-side scale augmentation and an OWLv2 crop reranker reach state of the art on ILIAS 100M (29.2 mAP@1000 before reranking, 42.0 after) without training or modifying the precomputed gallery 
    
[^12]: SAIL：通过科学感知循环实现科学智能体智能

    SAIL: Scientific Agentic Intelligence via a Science-Aware Loop

    [https://arxiv.org/abs/2610.11451](https://arxiv.org/abs/2610.11451)

    SAIL是一个通过“科学感知改进循环”训练的开放科学智能体模型（35B总参数/3B激活参数），由前沿AI智能体自动诊断其任务失败并生成针对性训练任务，使其在文献研究、科学编程和多步骤研究工作流中达到有竞争力的表现。

    

    我们推出了SAIL，一个总参数量为35B、激活参数量为3B的开放模型，面向文献研究、科学编程和多步骤研究工作流。SAIL通过一个科学感知的改进循环开发而成：基于前沿AI模型构建的智能体分析其任务失败案例，并构建针对底层能力差距的训练任务。诊断过程涵盖文献任务中的搜索与证据选择、编程中的科学假设与推理，以及较长周期研究中的规划与修订。这些智能体利用论文集合和科学代码库来构建问题、交互轨迹，以及配备所需环境和工具的可执行任务。我们在多个开发周期中重复这一循环，并通过监督微调、专项训练、多教师同策略蒸馏和智能体强化学习来训练SAIL。SAIL在科学相关任务上取得了具有竞争力的表现。

    arXiv:2610.11451v1 Announce Type: new  Abstract: We introduce SAIL, an open model with 35B total and 3B active parameters for literature research, scientific coding, and multi-step research workflows. SAIL is developed through a science-aware improvement loop: agents built on frontier AI models analyze its task failures and construct training tasks that address the underlying capability gaps. The diagnosis examines search and evidence selection in literature tasks, scientific assumptions and reasoning in coding, and planning and revision in longer investigations. The agents draw on paper collections and scientific code repositories to build problems, interaction trajectories, and executable tasks with the required environments and tools. We repeat this loop over multiple development cycles and train SAIL through supervised fine-tuning, specialist training, multi-teacher on-policy distillation, and agentic reinforcement learning. SAIL achieves competitive performance across scientific r
    
[^13]: 比特币预言机的链上考古：有限可观测性下的使用证据

    On-Chain Archaeology of Bitcoin Oracles: Evidence of Use under Limited Observability

    [https://arxiv.org/abs/2610.11439](https://arxiv.org/abs/2610.11439)

    该研究首次对比特币预言机的链上使用历史进行系统性“考古”，通过投注普查、全链分析与公钥索引搜索发现：早期链上合约虽已留存，但其事件描述大量消失，而现代DLC的公开预言机公告即使合约本身无法被识别也可能长期留存。

    

    在以太坊使“预言机问题”成为广为人知的术语之前，比特币就已经拥有预言机，它们作为数据源、密钥释放服务、联合签名者和仲裁者，在主链上承载着真实价值。本研究追溯了预言机从早期到2026年7月的使用情况及其活动证据的演变。我们结合了对Counterparty投注的完整普查（1,149笔投注）、对比特币全链至第958,628个区块的分析，以及在一个包含8.54亿行的公钥索引中搜索来自Reality Keys、Orisi、Bitrated和Oraclize的已记录密钥。我们还从存档的区块浏览器和实时Nostr中继中恢复了DLC预言机记录。研究得出两个结果。首先，早期合约仍保留在链上，但许多事件描述已经消失，且协议编码和API限制使得到达幸存记录变得困难。其次，对于现代DLC，即使使用它们的合约无法被识别，公开的预言机公告仍可得以留存。

    arXiv:2610.11439v1 Announce Type: cross  Abstract: Before Ethereum made the "oracle problem" a household term, Bitcoin already had oracles serving as feeds, key-release services, federated signers, and arbiters that carried real value on the main chain. This study traces their use and the changing evidence of oracle activity from early days through July 2026. We combine a complete census of Counterparty betting (1,149 bets), analysis of the full Bitcoin chain through block 958,628, and searches for documented keys from Reality Keys, Orisi, Bitrated, and Oraclize in an 854-million-row public-key index. We also recover DLC oracle records from an archived explorer and live Nostr relays. Two results emerge. First, early contracts remain on-chain, but many event descriptions have disappeared, and protocol encoding and API limitations complicate access to the surviving record. However, for modern DLCs, public oracle announcements can survive even when the contracts using them cannot be ident
    
[^14]: RIT-RAG：利用检索诱导树导航文档语料库

    RIT-RAG: Navigating Document Corpora with Retrieval-Induced Trees

    [https://arxiv.org/abs/2610.11370](https://arxiv.org/abs/2610.11370)

    RIT-RAG将内容检索与文档结构导航相结合，离线为每篇文档构建目录树，查询时先检索宽泛的文本块并利用其位置诱导出可跨多个文档的、规模可控的子树，再由LLM智能体在子树中导航，从而在保留结构感知优势的同时克服了现有方法无法扩展到大型语料库且选错文档后无法恢复的缺陷。

    

    检索增强生成（RAG）将语言模型建立在外部语料库之上。智能体式RAG（Agentic RAG）支持迭代式搜索，但使模型只能接触到缺乏文档结构的孤立文本块，难以区分真正相关的证据与仅在表面上与查询相似的文本块。诸如PageIndex之类的结构感知方法虽然能够导航文档结构，但无法扩展到大型语料库的结构，因为这些结构无法容纳于LLM的上下文中。因此，这类方法首先需要通过文档检索器确定单一文档，一旦选择错误便无法恢复。我们提出RIT-RAG（检索诱导树RAG），将内容检索与结构导航相结合。在离线阶段，RIT-RAG根据每个文档的目录或站点地图为其构建一棵树。在查询时，它检索一组宽泛的文本块，并利用这些文本块的位置来诱导出规模可控的子树，这些子树可能跨越多个文档。LLM智能体在这些子树中进行导航，有选择地……（原文摘要在此处截断）

    arXiv:2610.11370v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) grounds language models in external corpora. Agentic RAG enables iterative search, yet exposes the model to isolated chunks without document structure, making it difficult to distinguish relevant evidence from chunks that merely resemble the query. Structure-aware methods such as PageIndex navigate document structure but cannot scale to the structures of large corpora, which do not fit in the LLM context. Hence, they first commit to a single document using a document retriever and cannot recover from a wrong choice. We propose RIT-RAG (Retrieval-Induced Tree RAG), which combines content retrieval with structural navigation. Offline, RIT-RAG builds a tree for each document from its table of contents or sitemap. At query time, it retrieves a broad set of chunks and uses their positions to induce manageable sub-trees, potentially across multiple documents. An LLM agent navigates these sub-trees, selectiv
    
[^15]: H2CE：基于异构两阶段交叉编码器建模地理-语义交互的POI重排序方法

    H2CE: Modeling Geo-Semantic Interactions for POI Reranking with Heterogeneous Two-Stage Cross-Encoders

    [https://arxiv.org/abs/2610.11277](https://arxiv.org/abs/2610.11277)

    H2CE提出了一种异构两阶段交叉编码器，通过将分桶自然语言描述与经MLP处理的精确标量值相结合并在潜空间中融合语义与数值嵌入，建模语义、地理邻近性与数值质量信号之间的非线性交互，从而实现低延迟的POI重排序。

    

    本地搜索中的兴趣点（POI）重排序需要建模查询条件下词法语义、地理空间邻近性以及评分和评论数等数值质量信号之间的权衡，同时还要在实时服务约束下保持实用性。一个距离较近的POI可能只部分满足查询意图，而一个距离较远的POI可能提供更强的语义和质量证据。我们提出了H2CE，一种面向延迟受限POI重排序的异构两阶段交叉编码器。H2CE以两种互补的方式表示数值属性：将分桶后的自然语言描述插入交叉编码器输入以支持语义-数值注意力机制，同时由专用的多层感知机（MLP）处理精确标量值以保留量级信息。所得到的语义嵌入和数值嵌入通过潜空间聚合进行融合，实现了超越标量加权求和的非线性交互。H2CE随后采用两阶段架构……（摘要内容到此截断）

    arXiv:2610.11277v1 Announce Type: new  Abstract: Point-of-Interest (POI) reranking in local search must model query-conditioned tradeoffs among lexical semantics, geospatial proximity, and numerical quality signals such as rating and review count, while remaining practical under real-time serving constraints. A close POI may only partially satisfy the query intent, while a farther one may offer stronger semantic and quality evidence. We present H2CE, a Heterogeneous Two-stage Cross-Encoder for latency-bounded POI reranking. H2CE represents numerical attributes in two complementary ways: bucketized natural-language descriptors are inserted into the cross-encoder input to support semantic--numeric attention, while exact scalar values are processed by dedicated MLPs to preserve magnitude information. The resulting semantic and numerical embeddings are fused through latent-space aggregation, enabling nonlinear interactions beyond scalar weighted sums. H2CE then applies a two-stage architec
    
[^16]: 门控记忆：面向对话式AI的准入控制式记忆形成

    Gated Memory: Admission-Controlled Memory Formation for Conversational AI

    [https://arxiv.org/abs/2610.11270](https://arxiv.org/abs/2610.11270)

    该论文提出Gated Memory框架，通过在对话与存储之间设置准入控制检查点，在事实提取前基于完整话语上下文评估候选事实，解决了关键上下文信号在提取时不可逆丢失这一制约记忆质量的瓶颈问题。

    

    个性化对话式AI依赖于长期记忆系统，该系统从用户话语中提取事实并将其存储在持久化向量库中。尽管在检索、去重和生命周期管理方面已取得进展，但记忆形成阶段——即事实首次写入存储的时刻——几乎未受到任何系统性的关注。我们将此识别为生产系统中记忆质量的关键制约因素。关键上下文信号，例如用户的永久属性与临时情境之间的区分，仅存在于原始话语中，并且在提取过程生成“主语-关系-宾语”三元组的那一刻便不可逆地丢失了，任何下游流程都无法将其恢复。我们提出Gated Memory（门控记忆），一个轻量级、模块化的记忆形成框架，它在对话与存储之间引入两个决策检查点：其中一个是准入门，在提取之前根据完整话语上下文评估每个候选事实……

    arXiv:2610.11270v1 Announce Type: cross  Abstract: Personalized conversational AI relies on long-term memory systems that extract facts from user utterances and store them in persistent vector stores. Despite progress in retrieval, deduplication, and lifecycle management, the formation stage, the moment a fact is first written to storage has received almost no principled attention. We identify this as the binding constraint on memory quality in production systems. Critical contextual signals, such as the distinction between a permanent user attribute and a transient situation, exist only in the original utterance and are irreversibly lost the moment extraction produces a subject-relation-object triple. No downstream process can recover them. We propose Gated Memory, a lightweight, modular formation framework that interposes two decision checkpoints between conversation and storage: an admission gate that evaluates every candidate fact against the full utterance context before extractio
    
[^17]: 通过语料库反馈学习多步查询重写以实现对话式搜索

    Learning Multi-Step Query Rewriting via Corpus Feedback for Conversational Search

    [https://arxiv.org/abs/2610.10955](https://arxiv.org/abs/2610.10955)

    该论文将对话式查询重写重新构建为顺序检索问题，提出一个基于语料库反馈进行多步迭代的重写智能体，通过监督微调与强化学习训练，在无需人工重写标注的情况下显著提升了检索质量。

    

    arXiv:2610.10955v1 公告类型：new 摘要：对话式查询重写（CQR）将与上下文相关的用户发言转化为可供检索器使用的独立查询，大多数方法在检索之前仅根据对话历史一次性完成这一步骤。因此，在没有任何语料库证据可用于纠正其指代消解或词汇选择之前，重写结果就已经被固定下来。我们将CQR重新构建为一个顺序检索问题：智能体重写当前轮次，执行检索，并根据返回的文本来决定下一步的重写。智能体在一个类型化的操作空间中行动，该空间包含三种重写操作——将对话意图解析为独立查询、生成词汇级改写、或合成伪文档以进行文档到文档的匹配，此外还有一个结束回合的停止动作。我们首先使用监督微调训练该策略，随后针对单一的检索质量奖励进行强化学习，且不使用任何人工重写标注。在TopiOCQA和QReCC数据集上（摘要至此处截断）...

    arXiv:2610.10955v1 Announce Type: new  Abstract: Conversational Query Rewriting (CQR) turns a context dependent user turn into a standalone query for a retriever, and most methods do this in a single step from the dialogue history before retrieving once. The rewrite is therefore fixed before any corpus evidence is available to correct its reference resolution or its vocabulary. We recast CQR as a sequential retrieval problem: an agent rewrites the current turn, retrieves, and conditions its next rewrite on the returned passages. The agent acts in a typed space of three rewriting operations, resolving conversational intent into a standalone query, generating lexical reformulations, or synthesizing pseudo-documents for document-to-document matching, together with a stop action that ends the episode. We train the policy with supervised fine-tuning followed by reinforcement learning against a single retrieval-quality reward, using no human rewrite annotations. Across TopiOCQA and QReCC, th
    
[^18]: 电商搜索中用于页面级布局决策的语言模型

    Language Models for Page-Level Layout Decisions in E-commerce Search

    [https://arxiv.org/abs/2610.10920](https://arxiv.org/abs/2610.10920)

    该论文研究了利用语言模型离线评估电商搜索页面级布局决策（如在特定位置插入二级堆栈）是否对用户有益，从而减少对昂贵在线A/B测试的依赖。

    

    电商搜索页面是数百万在线购物者的关键触点。虽然传统搜索引擎返回的是按排名排序的结果列表，但现代电商搜索页面越来越多地整合了推荐系统模块——例如，在特定位置展示替代产品分组的二级堆栈。当引入得当时，二级堆栈可以提升用户参与度；然而，放置位置欠佳可能会干扰浏览流程并降低主要结果的质量。与传统搜索排名不同（后者已有交织比较等成熟的评估技术），在缺乏昂贵的在线A/B测试的情况下，评估页面级布局变化（例如，何时何地插入二级堆栈）仍然具有挑战性。为了解决这一问题，我们研究了离线方法，用于评估给定的布局决策——具体而言，在特定位置包含二级堆栈——是否对用户有益。我们研究……

    arXiv:2610.10920v1 Announce Type: cross  Abstract: E-commerce search pages are critical touchpoints for millions of online shoppers. While traditional search engines return a ranked list of results, modern E-commerce search pages increasingly incorporate recommender system modules -- for example, secondary stacks that surface alternative product groupings at specific positions. When introduced appropriately, secondary stacks can improve user engagement; however, suboptimal placement may disrupt browsing flow and degrade the primary results. Unlike traditional search ranking, where evaluation techniques such as interleaving are well established, evaluating page-level layout changes e.g., when and where to insert a secondary stack remains challenging without costly online A/B testing. To address this, we study offline methods for evaluating whether a given layout decision -- specifically, the inclusion of a secondary stack at a particular position -- is beneficial to users. We investigat
    
[^19]: LIFT：一种用于统一检索与排序的生命周期感知交互分解Transformer

    LIFT: A Lifecycle-aware Interaction Factorization Transformer for Unified Retrieval and Ranking

    [https://arxiv.org/abs/2610.10556](https://arxiv.org/abs/2610.10556)

    LIFT将用户交互分解为请求、物品、上下文、动作四个生命周期状态并建模为因果序列，使检索与排序任务共享历史建模的同时保留各自阶段特定信息，在ML-20M和淘宝数据集上分别超越最强基线4.9%和3.6%。

    

    级联推荐系统在检索和排序阶段使用相同的用户历史，而这两个阶段在交互的不同时间点能够获取不同的信息。我们提出了生命周期感知交互分解Transformer（LIFT），它将每次交互分解为有序的请求、物品、上下文和动作状态，并将它们建模为因果序列。检索读取请求状态，而排序读取上下文状态，使两个任务在共享历史建模的同时保留各自阶段特定的信息。LIFT通过角色条件注意力机制和轻量级的Pre-LN偏置来实现这一表示。在ML-20M和淘宝数据集上，LIFT在所评估的联合模型中取得了最高的联合分数，分别比最强基线提升了4.9%和3.6%。损失权重扫描显示了良好的检索-排序权衡，消融实验和缩放分析进一步检验了生命周期序列建模的有效性。

    arXiv:2610.10556v1 Announce Type: new  Abstract: Cascaded recommender systems use the same user history for retrieval and ranking, while the two stages have access to different information at different points of an interaction. We propose the Lifecycle-aware Interaction Factorization Transformer (LIFT), which decomposes each interaction into ordered Request, Item, Context, and Action states and models them as a causal sequence. Retrieval reads the Request state, while ranking reads the Context state, allowing both tasks to share history modeling while preserving stage-specific information. LIFT instantiates this representation with Role-Conditioned Attention and a lightweight Pre-LN Bias. On ML-20M and Taobao, LIFT achieves the highest Joint Score among the evaluated joint models, improving over the strongest baselines by 4.9% and 3.6%, respectively. Loss-weight sweeps show favorable retrieval--ranking trade-offs, while ablations and scaling analyses further examine lifecycle sequence 
    
[^20]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^21]: RecToolBench：面向模糊用户意图的推荐专用工具编排基准测试

    RecToolBench: Benchmarking Recommendation-Specific Tool Orchestration under Fuzzy User Intent

    [https://arxiv.org/abs/2609.30717](https://arxiv.org/abs/2609.30717)

    该论文提出了RecToolBench，一个基于MCP协议、用于评估推荐智能体在模糊用户意图下进行工具编排能力的基准测试，包含超过1,200个可执行任务，覆盖单工具调用、并行调用、顺序工具链和混合编排等多种复杂工具使用模式。

    

    智能体化推荐系统的最新进展正在推动推荐系统从被动的过滤引擎转变为遵循指令的智能体，这些智能体利用外部工具来解析用户意图。然而，现有的基准测试通常假设用户意图是明确的、工具环境是简化的、或者函数调用是孤立的，这使得面向推荐场景的真实工具编排研究仍未得到充分探索。为了弥补这一空白，我们提出了RecToolBench，这是一个基于模型上下文协议（MCP）的基准测试，用于评估在使用工具的推荐智能体在模糊用户指令下的表现。RecToolBench包含超过1,200个可执行任务，覆盖三个推荐领域、13个MCP服务器和32个工具，涵盖单工具调用、并行工具调用、顺序工具链以及混合工具编排等多种模式。我们通过一个可扩展的“合成—模糊化—评判”流水线构建了RecToolBench，该流水线生成可执行的模糊推荐任务，并使用……（原文摘要在此处截断）评估智能体的执行轨迹。

    arXiv:2609.30717v1 Announce Type: new  Abstract: Recent advances in agentic recommender systems are shifting recommender systems from passive filtering engines to instruction-following agents that use external tools to resolve user intent. However, existing benchmarks often assume explicit user intent, simplified tool environments, or isolated function calls, leaving realistic tool orchestration for recommendation underexplored. To bridge this gap, we propose RecToolBench, a Model Context Protocol (MCP)-based benchmark for evaluating tool-using recommender agents under fuzzy user instructions. RecToolBench contains more than 1,200 executable tasks across three recommendation domains, 13 MCP servers, and 32 tools, spanning single-tool calls, parallel tool calls, sequential tool chains, and hybrid tool orchestration. We construct RecToolBench with a scalable synthesize--fuzzify--judge pipeline that generates executable fuzzy recommendation tasks, and evaluates agent trajectories using ru
    
[^22]: MERGED：通过生成的专家推理蒸馏实现的多模态实体解析

    MERGED: Multimodal Entity Resolution via Generated Expert Reasoning Distillation

    [https://arxiv.org/abs/2609.01913](https://arxiv.org/abs/2609.01913)

    MERGED框架让多个大型视觉-语言模型教师为商品对标注并阐述推理，将一致标注用于监督微调、分歧标注经元裁判转化为直接偏好优化的偏好对，从而把结构化推理蒸馏到无需人工标注的7B紧凑学生模型中，实现了生产规模下低成本、低延迟的多模态商品实体解析。

    

    在商品实体解析中，关系定义会随着业务需求不断演变，而传统上每次适应变化都需要缓慢且昂贵的人工标注，这些标注往往噪声较多且缺乏推理过程。通过零样本提示的大型视觉-语言模型（VLM）可以立即适应新定义，并提供人工标注所缺乏的推理，但其成本和延迟在生产规模下令人难以承受。我们提出了MERGED——一个蒸馏框架，它不仅传递标签，还将大型教师VLM的结构化推理转移到一个紧凑的70亿参数学生模型中，且无需任何人工标注。多个教师模型为每个商品对进行标注，并阐述其决策背后的推理：一致的标注对用于监督微调，而不一致的标注则由元裁判（meta-judge）裁定，转化为用于直接偏好优化（DPO）的偏好对。在多语言电商数据集上对照人工标注的真实标准进行评估……

    arXiv:2609.01913v1 Announce Type: new  Abstract: In product entity resolution, relationship definitions constantly evolve with business needs, yet adapting to each change traditionally requires slow, costly human annotation that is often noisy and carries no reasoning. Large vision-language models (VLMs) prompted zero-shot can adapt to a new definition immediately and supply the reasoning that human labels lack, but their cost and latency are prohibitive at production scale. We present MERGED, a distillation framework that transfers not just labels but structured reasoning from large teacher VLMs into a compact 7B-parameter student, requiring no human annotation. Multiple teachers label each product pair and articulate the reasoning behind their decision: agreement pairs supply supervised fine-tuning, while disagreements are resolved by a meta-judge into preference pairs for Direct Preference Optimization. Evaluated against human-labeled ground truth on a multilingual e-commerce datase
    
[^23]: GLM-RAG：用于基于图的检索增强生成的图语言模型

    GLM-RAG: Graph Language Models for Graph-Based Retrieval-Augmented Generation

    [https://arxiv.org/abs/2607.28397](https://arxiv.org/abs/2607.28397)

    本文提出了一种基于图语言模型（GLM）的检索器用于知识图谱检索增强生成，发现微调后的GLM检索器在跨领域泛化能力上优于GNN和向量搜索检索器，并在两个多跳基准上达到SOTA。

    

    基于知识图谱的检索增强生成（RAG）需要能够有效捕获图结构和语义信息的检索器。近期的方法探索了基于图神经网络（GNN）的检索器，以在多跳推理任务中建模图拓扑结构。与此同时，图语言模型（GLM）作为一种融合图推理能力与语言模型语义能力的新兴范式应运而生。在本工作中，我们提出了一种基于GLM的检索器，并研究了基于GLM的检索器、基于GNN的检索器以及传统基于向量搜索的检索器在单跳和多跳RAG设置中的比较优势，并特别关注其在未见领域上的可迁移性。我们的研究结果表明，经过微调的GLM检索器在领域外具有更好的泛化能力，在两个多跳基准测试上达到了最先进水平（SOTA）。在领域内的多跳问答数据集上，它们仍然与先前的工作相当，并且随着参数（规模增长）展现出可喜的扩展性。

    arXiv:2607.28397v2 Announce Type: replace  Abstract: Retrieval-augmented generation (RAG) over knowledge graphs requires retrievers that can effectively capture both graph structure and semantic information. Recent approaches have explored graph neural network (GNN)-based retrievers to model graph topology in multi-hop reasoning tasks. In parallel, graph language models (GLMs) have emerged as a promising paradigm that integrates graph reasoning and the semantic capabilities of language models. In this work, we introduce a GLM-based retriever and investigate the comparative strengths of GLM-based, GNN-based, and traditional vector-search-based retrievers in single- and multi-hop RAG settings, and with a particular focus on transferability to unseen domains. Our findings suggest that finetuned GLM retrievers generalize better out of domain, achieving SOTA on two multi-hop benchmarks. On in-domain multi-hop QA datasets they remain comparable to prior work, with promising scaling as parame
    
[^24]: 正确的技能族，错误的技能项：智能体技能检索中的风险暴露基准测试

    Right Family, Wrong Skill: Benchmarking Risk Exposure in Agent Skill Retrieval

    [https://arxiv.org/abs/2606.10388](https://arxiv.org/abs/2606.10388)

    本文提出了SameCapRisk-Bench基准，专门研究智能体技能检索中“找到正确技能族但暴露错误同族技能”的风险暴露失败，并提供了1,190个技能-风险单元和1,686个评估查询案例。

    

    arXiv:2606.10388v2 公告类型：替换-交叉 摘要：智能体技能库正成为可路由的软件资产：检索到的技能可以为智能体提供指令、脚本、资源绑定和执行假设。这使得检索失败比广泛的不相关性更具特异性。系统可能找到正确的技能族，却暴露了错误的同族代表性技能。我们将这种失败研究为同族风险暴露检索。每个基准单元将一个有用的技能与一个查询特定的风险孪生技能配对，该孪生技能共享技能族但在执行控制契约（如所需资源、前置条件、过程或工件）上有所不同。我们引入了SameCapRisk-Bench，一个可审计的基准测试，包含1,190个技能-风险单元和1,686个评估查询案例：694个在公共库压力下的标记孪生单元，以及496个硬角色翻转单元，其中相同的两个技能在配对查询中交换有用/风险角色。发布记录包括资格标准、排除规则、查询生成协议和验证程序。

    arXiv:2606.10388v2 Announce Type: replace-cross  Abstract: Agent skill libraries are becoming routable software assets: a retrieved skill can contribute instructions, scripts, resource bindings, and execution assumptions to an agent. This makes retrieval failures more specific than broad irrelevance. A system can find the right capability family yet expose the wrong same-capability representative. We study this failure as same-capability risk-exposure retrieval. Each benchmark unit pairs a helpful skill with a query-specific risky sibling that shares the capability family but differs on an execution-controlling contract, such as the required resource, precondition, procedure, or artifact. We introduce SameCapRisk-Bench, an auditable benchmark with 1,190 skill-risk units and 1,686 evaluation query cases: 694 marked-sibling units under public library pressure and 496 hard role-flip units where the same two skills swap helpful/risky roles across paired queries. The release records admissi
    
[^25]: 高德地图中基于隐式推理的生成式时空意图序列推荐

    Generative Spatiotemporal Intent Sequence Recommendation via Implicit Reasoning in Amap

    [https://arxiv.org/abs/2605.28888](https://arxiv.org/abs/2605.28888)

    高德地图提出GPlan框架，通过渐进式隐式思维链蒸馏将大语言模型的推理能力内化到轻量级模型中，在严格延迟约束下实现逻辑连贯且物理可执行的生成式时空意图序列推荐。

    

    现实世界中的用户行为很少由孤立的动作构成；相反，它往往形成受时空依赖关系支配的意图流。为了提供一体化的服务推荐，我们聚焦于生成式时空意图序列推荐（GSISR）这一任务，其目标是在复杂的时空情境中生成逻辑连贯且物理可执行的意图序列。尽管大语言模型为GSISR提供了强大的推理潜力，但其在工业界的直接部署受到高推理延迟以及规划结果与情境不匹配或物理上不可执行的限制。为应对这些挑战，我们提出了一个生成式框架GPlan，通过两个组件将大语言模型的推理能力内化到轻量级模型中。首先，为了在严格的延迟约束下实现推理，我们引入了渐进式隐式思维链蒸馏（Progressive Implicit CoT Distillation），将显式的推理过程压缩到预留的潜在token中，使小……

    arXiv:2605.28888v2 Announce Type: replace-cross  Abstract: Real-world user behavior rarely consists of isolated actions; instead, it often forms intent flows governed by spatiotemporal dependencies. To provide integrated service recommendations, we focus on the task of Generative Spatiotemporal Intent Sequence Recommendation (GSISR), which aims to generate intent sequences that are logically coherent and physically executable within complex spatiotemporal contexts. While LLMs offer strong reasoning potential for GSISR, direct industrial deployment is limited by high inference latency and context-mismatched or physically infeasible plans. To address these challenges, we propose a generative framework, GPlan, that internalizes LLM reasoning into lightweight models through two components. First, to enable reasoning under strict latency constraints, we introduce Progressive Implicit CoT Distillation, which compresses explicit reasoning processes into reserved latent tokens, allowing small 
    
[^26]: 面向LLM生成代码片段的高效可扩展溯源追踪方法

    Efficient and Scalable Provenance Tracking for LLM-Generated Code Snippets

    [https://arxiv.org/abs/2605.28510](https://arxiv.org/abs/2605.28510)

    该论文提出SourceTracker编码器与HybridSourceTracker两阶段混合流水线，先用向量搜索缩小候选范围、再用Winnowing指纹精确重排，从而实现对LLM生成代码在数十亿级训练语料上的高效可扩展溯源追踪。

    

    大型语言模型（LLM）在代码补全与生成中的应用日益广泛，但它们可能会逐字复现训练样本而不进行作者归属标注，这引发了关于剽窃和许可证合规方面的法律与伦理问题。经典的基于指纹的剽窃检测方法（如Winnowing）仍然非常有效，但其检测过程需要将代码片段与整个训练集进行比较，且其线性时间搜索使得这些方法对于训练现代代码LLM所使用的数十亿级语料库而言并不实用。为了弥合这一差距，我们提出了SourceTracker——一个专为代码检索定制的3亿参数编码器，并构建了一个混合两阶段溯源追踪流水线HybridSourceTracker（HST）。HST首先通过向量搜索缩小候选代码片段的范围，然后使用Winnowing对精确指纹进行重排序。我们在……（训练和评估相关内容在原文中被截断）

    arXiv:2605.28510v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) for code completion and generation are increasingly used in software development, yet they may reproduce training examples verbatim and without authorship attribution, raising legal and ethical concerns around plagiarism and license compliance. Classical fingerprint-based plagiarism detectors, such as Winnowing, remain highly effective, yet the inspection requires comparing fragments of code to the entire training set, and their linear-time search makes them impractical for the billion-scale corpora used to train modern code LLMs. To bridge this gap, we introduce SourceTracker, a 300M-parameter encoder tailored for code retrieval, together with a hybrid two-stage provenance-tracking pipeline HybridSourceTracker (HST). HST first narrows down a small set of candidate snippets via vector search, then re-ranks those candidates using Winnowing on exact fingerprints. We train and evaluate our system on a 
    
[^27]: 用于预测细胞对基因扰动响应的检索增强生成

    Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation

    [https://arxiv.org/abs/2603.07233](https://arxiv.org/abs/2603.07233)

    提出了PT-RAG，一个即插即用的两阶段检索增强生成模块，通过GenePT语义检索与可微分Gumbel-Softmax选择器动态获取相关的扰动上下文，从而改进单细胞基因扰动响应的预测。

    

    预测基因扰动的转录响应是功能基因组学和治疗发现的基础。最近的深度学习模型在单细胞扰动响应预测方面展现出了前景，但它们通常孤立地生成每个响应，而没有显式地利用相关的扰动信息。我们提出了PT-RAG（扰动感知的两阶段检索增强生成），这是一个用于生成式细胞扰动响应的即插即用检索与条件化模块。PT-RAG通过学习访问相关的扰动上下文来增强现有的扰动-响应骨干模型。关键挑战在于，在这种设置下相关性并非固定不变：功能相关的基因在不同细胞类型中可能引发不同的效应。PT-RAG通过两阶段检索机制解决这一问题：首先基于GenePT的语义检索识别K个候选扰动，随后由一个可微分的Gumbel-Softmax选择器…

    arXiv:2603.07233v2 Announce Type: replace  Abstract: Predicting transcriptional responses to genetic perturbations is fundamental to functional genomics and therapeutic discovery. Recent deep learning models have shown promise in single-cell perturbation response prediction, but they typically generate each response in isolation, without explicitly leveraging related perturbations. We introduce PT-RAG (Perturbation-aware Two-stage Retrieval-Augmented Generation), a plug-in retrieval-and-conditioning module for generative cellular perturbation response. PT-RAG augments an existing perturbation-response backbone with learned access to related perturbation contexts. The key challenge is that relevance is not fixed in this setting: functionally related genes may elicit different effects across cell types. PT-RAG addresses this with a two-stage retrieval mechanism: GenePT-based semantic retrieval first identifies K candidate perturbations, after which a differentiable Gumbel-Softmax selecto
    
[^28]: LIME：基于链接的用户-物品交互建模与解耦XOR注意力机制，实现高效的测试时扩展

    LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling

    [https://arxiv.org/abs/2510.18239](https://arxiv.org/abs/2510.18239)

    LIME架构通过低秩链接嵌入解耦用户与候选交互、实现注意力权重预计算，并采用线性XOR注意力机制，从根本上降低了推荐系统推理时对候选集大小和用户序列长度的计算成本依赖。

    

    扩展大型推荐系统需要在三个主要方向上取得进展：处理更长的用户历史、扩大候选集规模以及增加模型容量。尽管前景可观，但Transformer的计算成本随用户序列长度呈二次方增长，并随候选数量呈线性增长。这种权衡使得在推理阶段扩大候选集或增加序列长度的代价极其高昂，尽管这能带来显著的性能提升。我们提出了LIME，一种解决这一权衡的新型架构。通过两项关键创新，LIME从根本上降低了计算复杂度。第一，低秩“链接嵌入”通过解耦用户与候选之间的交互，实现注意力权重的预计算，使推理成本几乎与候选集大小无关。第二，线性注意力机制LIME-XOR降低了相对于用户序列长度的计算复杂度。

    arXiv:2510.18239v4 Announce Type: replace-cross  Abstract: Scaling large recommendation systems requires advancing three major frontiers: processing longer user histories, expanding candidate sets, and increasing model capacity. While promising, transformers' computational cost scales quadratically with the user sequence length and linearly with the number of candidates. This trade-off makes it prohibitively expensive to expand candidate sets or increase sequence length at inference, despite the significant performance improvements. We introduce \textbf{LIME}, a novel architecture that resolves this trade-off. Through two key innovations, LIME fundamentally reduces computational complexity. First, low-rank ``link embeddings" enable pre-computation of attention weights by decoupling user and candidate interactions, making the inference cost nearly independent of candidate set size. Second, a linear attention mechanism, \textbf{LIME-XOR}, reduces the complexity with respect to user seque
    

