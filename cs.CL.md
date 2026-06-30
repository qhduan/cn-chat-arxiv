# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mapping Political-Elite Networks in Europe with a Multilingual Joint Entity-Relation Extraction Pipeline](https://arxiv.org/abs/2606.27347) | 提出了一种模块化、完全开源的多语言联合实体关系抽取流水线，通过结合跨度NER和三阶段链接级联，从大规模新闻语料中构建带符号的时间知识图谱，解决了现有方法依赖专有API、缺乏跨语言能力及可扩展性差的问题。 |
| [^2] | [CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention](https://arxiv.org/abs/2606.27229) | CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。 |
| [^3] | [HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction](https://arxiv.org/abs/2606.26744) | 针对DeepSeek-V4的多超连接架构，提出了一种通过门控残差缩减实现特征对齐的块级推测解码方法，解决了多路径残差流导致的生成准确率下降问题。 |
| [^4] | [Epiphany-Aware KV Cache Eviction Without the Attention Matrix](https://arxiv.org/abs/2606.26472) | 本文提出EpiKV，一种通过直接读取模型前向传播中的内部表示变化（顿悟分数）来淘汰KV缓存的方法，无需注意力矩阵，可将可行上下文长度扩展至传统注意力评分方法的16倍，且无需训练或自定义内核。 |
| [^5] | [The Verification Horizon: No Silver Bullet for Coding Agent Rewards](https://arxiv.org/abs/2606.26300) | 本文指出，在编码智能体领域，验证解决方案比生成解决方案更难，并提出了验证信号质量的三个维度（可扩展性、忠实性和鲁棒性），强调同时实现这些维度面临根本性挑战。 |

# 详细

[^1]: 绘制欧洲政治精英网络：一种多语言联合实体关系抽取流水线

    Mapping Political-Elite Networks in Europe with a Multilingual Joint Entity-Relation Extraction Pipeline

    [https://arxiv.org/abs/2606.27347](https://arxiv.org/abs/2606.27347)

    提出了一种模块化、完全开源的多语言联合实体关系抽取流水线，通过结合跨度NER和三阶段链接级联，从大规模新闻语料中构建带符号的时间知识图谱，解决了现有方法依赖专有API、缺乏跨语言能力及可扩展性差的问题。

    

    arXiv:2606.27347v1 公告类型：新 摘要：政治精英是组织成寻租联盟以攫取公共资源，还是形成维持治理的公民网络，是比较政治学的一个核心问题。然而，大规模观察这些复杂、非正式且充满对抗性的关系，历史上需要大量的人工编码，而基于文本的数据自动化方法大多局限于简单的共现分析。近期的大语言模型方法提供了一条前进道路，但往往依赖专有API、缺乏跨语言能力，并且在可扩展的实体消解方面存在困难。我们提出了一种模块化、完全开源的流水线，用于多语言联合实体关系抽取，该流水线能够从大规模非结构化新闻语料中构建带符号的时间知识图谱。它结合了基于跨度的命名实体识别与一个三阶段链接级联，将提及映射到语言无关的Wikidata标识符；一个高吞吐量、本体约束的混合模型。

    arXiv:2606.27347v1 Announce Type: new  Abstract: Whether political elites organise into rent-seeking coalitions that capture public resources or civic networks that sustain governance is a central question in comparative politics. Yet observing these complex, informal, and adversarial ties at scale has historically required intensive manual coding, while automated text-as-data methods have largely been limited to simple co-occurrence. Recent large language model (LLM) approaches offer a path forward but often rely on proprietary APIs, lack cross-lingual capability, and struggle with scalable entity resolution. We present a modular, fully open-weight pipeline for multilingual joint entity-relation extraction that builds signed, temporal knowledge graphs from massive unstructured news corpora. It combines span-based named-entity recognition (NER) with a three-stage linking cascade mapping mentions to language-independent Wikidata identifiers; a high-throughput, ontology-constrained mixtu
    
[^2]: CARVE：面向分块并行线性注意力的内容感知递归与价值效率模型

    CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention

    [https://arxiv.org/abs/2606.27229](https://arxiv.org/abs/2606.27229)

    CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。

    

    arXiv:2606.27229v1 公告类型：跨领域 摘要：递归模型必须通过遗忘来记住，然而现有技术决定遗忘什么时并不参考已存储的内容——门控机制仅看到当前到达的标记，而非即将修改的记忆。这种记忆盲区门控是当前主流delta规则架构（GDN-2）中三个相互耦合的缺陷之一：价值轴擦除掩码在价值投影的尺度上浪费参数，并且——正如我们所证明的——在数学上阻碍了使递归训练与Transformer相媲美的WY形式三角分块求解器。我们提出CARVE（内容感知递归与价值效率模型），通过一个原则解决所有三个问题：仅在键轴上擦除。这在数学上被证明是WY形式求解器保持有效的必要且充分条件。在此框架下，CARVE复用已写入GPU内存的递归输出张量作为擦除门控的免费内容信号，并替换逐值写入门控。

    arXiv:2606.27229v1 Announce Type: cross  Abstract: Recurrent models must forget in order to remember, yet the state of the art decides what to erase without consulting what is stored -- the gate sees only the arriving token, not the memory it is about to modify. This memory-blind gating is one of three coupled defects in the leading delta-rule architecture (GDN-2): the value-axis erase mask wastes parameters at the scale of the value projection, and -- as we prove -- mathematically prevents the WY-form triangular chunk solver that makes recurrent training competitive with Transformers.   We introduce CARVE (Content-Aware Recurrent with Value Efficiency), which resolves all three problems through one principle: erase only on the key axis. This is provably necessary and sufficient for the WY-form solver to remain valid. Within it, CARVE reuses the recurrent output tensor -- already written to GPU memory -- as a free content signal for the erase gate, and replaces the per-value write-gate
    
[^3]: HyperDFlash：基于门控残差缩减的MHC对齐块级推测解码

    HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction

    [https://arxiv.org/abs/2606.26744](https://arxiv.org/abs/2606.26744)

    针对DeepSeek-V4的多超连接架构，提出了一种通过门控残差缩减实现特征对齐的块级推测解码方法，解决了多路径残差流导致的生成准确率下降问题。

    

    arXiv:2606.26744v1 公告类型：交叉 摘要：我们提出了HyperDFlash，这是一个针对DeepSeek-V4提出的新型多超连接（MHC）架构量身定制的块级并行推测解码框架。尽管DeepSeek-V4原生多令牌预测（MTP）模块在初始令牌生成方面表现强劲，但其后续位置的生成准确性会急剧下降，因为未验证中间令牌的错误累积会降低接受率。虽然原始的DFlash方法支持高效的单次块级生成，但它无法无缝适配到MHC范式，因为DeepSeek-V4的多路径残差流会导致与常规生成设计之间的特征错配。为了解决这一不匹配问题，我们针对MHC残差流提出了两种模型对齐优化。首先，我们采用预折叠残差状态作为唯一的条件信号，保留多路径结构信息，并将生成器与原生的预测路径对齐。

    arXiv:2606.26744v1 Announce Type: cross  Abstract: We present HyperDFlash, a block-parallel speculative decoding framework tailored to the novel multi-hyper-connection (MHC) architecture proposed by DeepSeek-V4. Despite the strong initial-token drafting performance of the native Multi-Token Prediction (MTP) module in DeepSeek-V4, its draft accuracy degrades sharply at later positions, as error accumulation from unverified intermediate tokens harms acceptance rates. Although the original DFlash method supports efficient one-pass block drafting, it cannot be seamlessly adapted to the MHC paradigm, since the multi-path residual stream of DeepSeek-V4 induces feature misalignment with conventional drafting designs. To resolve this mismatch, we propose two model-aligned optimizations for MHC residual streams. First, we adopt pre-collapse residual states as the exclusive conditioning signal, preserving multi-path structural information and aligning the drafter with the native prediction pathw
    
[^4]: 基于顿悟分数的KV缓存淘汰方法：无需注意力矩阵

    Epiphany-Aware KV Cache Eviction Without the Attention Matrix

    [https://arxiv.org/abs/2606.26472](https://arxiv.org/abs/2606.26472)

    本文提出EpiKV，一种通过直接读取模型前向传播中的内部表示变化（顿悟分数）来淘汰KV缓存的方法，无需注意力矩阵，可将可行上下文长度扩展至传统注意力评分方法的16倍，且无需训练或自定义内核。

    

    arXiv:2606.26472v1 公告类型：交叉 摘要：随着推理模型生成长达数万token的思维链，KV缓存日益成为部署瓶颈。现有缓存淘汰方法通过注意力权重对token进行排序，这在长推理轨迹中是一个有噪声的重要性代理，并且通过强制模型实现注意力矩阵，阻碍了生产推理中融合内核的使用。在这项工作中，我们提出了一种称为"顿悟分数"的度量标准来对token进行评分：该分数直接从前向传播中读取模型内部表示的变化，无需注意力矩阵且仅需极少的额外状态。由此产生的缓存淘汰方法EpiKV无需训练、分类器或自定义内核，可直接在FlashAttention推理栈中不变地使用——将可行上下文长度扩展至基于注意力评分方法的16倍。针对上层中间层（负向影响）和下层中间层（正向影响），我们采用因果滚动z分数消除位置趋势。在4096-token缓存设置下，该方法表现出色。

    arXiv:2606.26472v1 Announce Type: cross  Abstract: As reasoning models emit chains of thought tens of thousands of tokens long, KV cache increasingly becomes a deployment bottleneck. Existing cache eviction methods rank tokens by attention weight, which is a noisy importance proxy in long reasoning traces, and prohibits the use of fused kernels in production inference by forcing the model to materialize the attention matrix. In this work, we instead score tokens with a metric we term the epiphany score: the change in the model's internal representation, read directly from the forward pass with no attention matrix and negligible extra state. Our resulting cache eviction method, EpiKV, requires no training, classifier, or custom kernel, and can be used directly in FlashAttention inference stacks unchanged -- scaling to a 16x longer feasible context than attention-based scoring. upper-mid layers negatively) and remove a positional trend with a causal rolling z-score. At a 4096-token cache
    
[^5]: 验证地平线：编码智能体奖励没有银弹

    The Verification Horizon: No Silver Bullet for Coding Agent Rewards

    [https://arxiv.org/abs/2606.26300](https://arxiv.org/abs/2606.26300)

    本文指出，在编码智能体领域，验证解决方案比生成解决方案更难，并提出了验证信号质量的三个维度（可扩展性、忠实性和鲁棒性），强调同时实现这些维度面临根本性挑战。

    

    arXiv:2606.26300v1 公告类型：新论文 摘要：经典直觉认为，验证一个解决方案比生成一个解决方案更容易。对于当今的编码智能体而言，这种直觉正在被颠覆：随着基础模型发展出更强的推理能力，工程工具也变得日益复杂，生成复杂的候选解决方案已不再困难——而可靠地验证这些方案反而成了更棘手的问题。我们能够构建的每一个验证器都只是人类意图的代理，而非意图本身。这使得验证面临双重困难：首先，意图本质上是不明确的，因此很难忠实地检查它是否被满足；其次，在模型训练过程中，优化会扩大代理与意图之间的差距——表现为奖励黑客或信号饱和。为解决这一问题，我们从三个维度——可扩展性、忠实性和鲁棒性——描述了验证信号的质量，并论证了同时实现所有这些维度的困难。

    arXiv:2606.26300v1 Announce Type: new  Abstract: A classical intuition holds that verifying a solution is easier than producing one. For today's coding agents, this intuition is being inverted: as foundation models develop stronger reasoning capabilities and engineering harnesses grow more sophisticated, generating complex candidate solutions is no longer difficult -- reliably verifying them has become the harder problem. Every verifier we can build is only a proxy for human intent, never the intent itself. This makes verification subject to a twofold difficulty: first, intent is underspecified by nature, making it inherently hard to faithfully check whether it has been fulfilled; second, during model training, optimization widens the gap between proxy and intent -- manifesting as reward hacking or signal saturation. To address this, we characterize the quality of verification signals along three dimensions -- scalability, faithfulness, and robustness -- and argue that achieving all th
    

