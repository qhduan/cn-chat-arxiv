# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Good Can Linear Models Be for Time-Series Forecasting?](https://arxiv.org/abs/2606.27282) | 本文挑战了“更大模型容量带来更高精度”的主流观点，通过岭回归证明，精心调整预处理（如上下文长度、归一化）能以极低成本显著缩小与大型模型的性能差距，并揭示了最优回溯长度与预测范围的非单调关系等反直觉模式。 |
| [^2] | [BetXplain: An Explanation-Annotated Dataset for Detecting Manipulative Betting Advertisements on Social Media](https://arxiv.org/abs/2606.27274) | 本文提出了BetXplain数据集，通过人工标注社交媒体上的博彩广告并附带解释，为自动检测操纵性和欺骗性广告提供了可解释的研究基础。 |
| [^3] | [CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention](https://arxiv.org/abs/2606.27229) | CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。 |
| [^4] | [HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction](https://arxiv.org/abs/2606.26744) | 针对DeepSeek-V4的多超连接架构，提出了一种通过门控残差缩减实现特征对齐的块级推测解码方法，解决了多路径残差流导致的生成准确率下降问题。 |
| [^5] | [Epiphany-Aware KV Cache Eviction Without the Attention Matrix](https://arxiv.org/abs/2606.26472) | 本文提出EpiKV，一种通过直接读取模型前向传播中的内部表示变化（顿悟分数）来淘汰KV缓存的方法，无需注意力矩阵，可将可行上下文长度扩展至传统注意力评分方法的16倍，且无需训练或自定义内核。 |
| [^6] | [Finding the Time to Think: Learning Planning Budgets in Real-Time RL](https://arxiv.org/abs/2606.26463) | 提出了一种在实时强化学习中，通过轻量级门控策略动态选择状态依赖的规划预算的方法，有效解决了环境持续运行下的决策延迟问题。 |
| [^7] | [Unbiased Canonical Set-Valued Oracles Via Lattice Theory](https://arxiv.org/abs/2606.26418) | 本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。 |
| [^8] | [The Red Queen G\"odel Machine: Co-Evolving Agents and Their Evaluators](https://arxiv.org/abs/2606.26294) | 本文提出红皇后哥德尔机（RQGM），通过将评估者纳入进化循环，使智能体能在非平稳评估标准下进行递归自我改进，从而突破静态基准的限制。 |
| [^9] | [When to Write and When to Suppress: Route-Specialized Dual Adapters for Memory-Assisted Knowledge Editing](https://arxiv.org/abs/2606.14668) | 本文提出一种路径专用的双适配器方法，通过相关性路由器决定何时应用编辑记忆或保留原始知识，从而在知识编辑中实现精确更新与无关行为的保护。 |
| [^10] | [Closed-Loop CO2 Storage Control With History-Based Reinforcement Learning and Latent Model-Based Adaptation](https://arxiv.org/abs/2605.02405) | 本文提出了一种结合历史强化学习和潜在模型自适应的闭环二氧化碳封存控制方法，通过利用时间井响应信息和自适应机制，有效应对储层不确定性和动态变化。 |
| [^11] | [Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models](https://arxiv.org/abs/2505.12343) | 本文提出了一种名为DCLA的免训练解码方法，通过聚合前层表示构建动态语义参考来纠正语义偏差的层，从而有效缓解大型视觉语言模型中的幻觉问题，并在多个模型和基准上显著提升性能。 |
| [^12] | [Large (and Deep) Factor Models](https://arxiv.org/abs/2402.06635) | 本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。 |
| [^13] | [Multiply Robust Causal Mediation Analysis with Continuous Treatments](https://arxiv.org/abs/2105.09254) | 本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。 |
| [^14] | [Granular ball computing: an efficient, robust, and interpretable adaptive multi-granularity representation and computation method.](http://arxiv.org/abs/2304.11171) | 本文提出了一种基于颗粒球计算的自适应多粒度表示和计算方法，能够提高机器学习的效率、鲁棒性和可解释性。 |

# 详细

[^1]: 线性模型在时间序列预测中能有多好？

    How Good Can Linear Models Be for Time-Series Forecasting?

    [https://arxiv.org/abs/2606.27282](https://arxiv.org/abs/2606.27282)

    本文挑战了“更大模型容量带来更高精度”的主流观点，通过岭回归证明，精心调整预处理（如上下文长度、归一化）能以极低成本显著缩小与大型模型的性能差距，并揭示了最优回溯长度与预测范围的非单调关系等反直觉模式。

    

    arXiv:2606.27282v1 公告类型：新 摘要：时间序列预测研究一直稳步转向更大的架构，从专门的Transformer到通用基础模型，其假设是容量是解锁精度的关键。我们持相反观点：通过调整预处理而非扩大模型规模，可以在成本低得多的前提下缩小大部分差距。我们使用岭回归作为测试平台，因为它具有闭式解和可解释的权重，这可以直接从搜索中读出最优超参数。我们在八个标准基准上对上下文长度、局部归一化、正则化和数据增强进行了搜索，发现了三种模式。(1) 最优回溯长度具有强烈的序列特异性，且通常在预测范围内是非单调的，拟合的幂律指数从ETTm2上的+0.46到Exchange和Traffic上的-0.19，挑战了“更长范围需要更长历史”的惯例。(2) 在一个学习的截断范围内进行归一化处理...

    arXiv:2606.27282v1 Announce Type: new  Abstract: Time-series forecasting research has been moving steadily toward larger architectures, from specialized transformers to general-purpose foundation models, on the assumption that capacity is what unlocks accuracy. We take the opposite position: most of the gap can be closed at far lower cost by tuning preprocessing rather than scaling models. We use Ridge regression as the testbed, since it has a closed-form solution and interpretable weights, which let the optimal hyperparameters be read off the search directly. We search over context length, local normalization, regularization, and augmentation on eight standard benchmarks and find three patterns. (1) Optimal lookback is strongly series-specific and often non-monotonic in forecast horizon, with fitted power-law exponents ranging from $+0.46$ on ETTm2 to $-0.19$ on Exchange and Traffic, challenging the convention that longer horizons need longer history. (2) Normalizing over a learned tr
    
[^2]: BetXplain：用于检测社交媒体上操纵性博彩广告的解释性标注数据集

    BetXplain: An Explanation-Annotated Dataset for Detecting Manipulative Betting Advertisements on Social Media

    [https://arxiv.org/abs/2606.27274](https://arxiv.org/abs/2606.27274)

    本文提出了BetXplain数据集，通过人工标注社交媒体上的博彩广告并附带解释，为自动检测操纵性和欺骗性广告提供了可解释的研究基础。

    

    arXiv:2606.27274v1 公告类型：新 摘要：近年来，社交媒体平台上博彩应用的推广显著增加。许多此类广告使用具有说服力的技术，可能误导用户、鼓励冒险行为，并可能影响用户的心理健康。然而，由于缺乏公开可用的标注数据集，关于自动检测操纵性和欺骗性博彩广告的研究仍然有限。在这项工作中，我们引入了一个新的博彩相关广告数据集，这些广告收集自两个广泛使用的社交媒体平台——Instagram和Reddit。这些广告被人工标注了操纵性和欺骗性广告实践。除了分类标签外，数据集还包括人工提供的解释，描述了每个标注背后的推理过程，从而支持对可解释性方法在检测操纵性广告方面的研究。此外，我们分析了这些广告中使用的策略。

    arXiv:2606.27274v1 Announce Type: new  Abstract: The promotion of betting applications on social media platforms has increased significantly in recent years. Many of these advertisements use persuasive techniques that may mislead users, encourage risky behavior, and potentially influence users' mental well-being. However, research on the automated detection of manipulative and deceptive betting advertisements remains limited due to the lack of publicly available annotated datasets. In this work, we introduce a new dataset of betting-related advertisements collected from two widely used social media platforms, Instagram and Reddit. The advertisements were manually annotated for manipulative and deceptive advertising practices. In addition to classification labels, the dataset includes human-provided explanations that describe the reasoning behind each annotation, enabling research into explainable approaches to detecting manipulative advertising. Furthermore, we analyze the strategies c
    
[^3]: CARVE：面向分块并行线性注意力的内容感知递归与价值效率模型

    CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention

    [https://arxiv.org/abs/2606.27229](https://arxiv.org/abs/2606.27229)

    CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。

    

    arXiv:2606.27229v1 公告类型：跨领域 摘要：递归模型必须通过遗忘来记住，然而现有技术决定遗忘什么时并不参考已存储的内容——门控机制仅看到当前到达的标记，而非即将修改的记忆。这种记忆盲区门控是当前主流delta规则架构（GDN-2）中三个相互耦合的缺陷之一：价值轴擦除掩码在价值投影的尺度上浪费参数，并且——正如我们所证明的——在数学上阻碍了使递归训练与Transformer相媲美的WY形式三角分块求解器。我们提出CARVE（内容感知递归与价值效率模型），通过一个原则解决所有三个问题：仅在键轴上擦除。这在数学上被证明是WY形式求解器保持有效的必要且充分条件。在此框架下，CARVE复用已写入GPU内存的递归输出张量作为擦除门控的免费内容信号，并替换逐值写入门控。

    arXiv:2606.27229v1 Announce Type: cross  Abstract: Recurrent models must forget in order to remember, yet the state of the art decides what to erase without consulting what is stored -- the gate sees only the arriving token, not the memory it is about to modify. This memory-blind gating is one of three coupled defects in the leading delta-rule architecture (GDN-2): the value-axis erase mask wastes parameters at the scale of the value projection, and -- as we prove -- mathematically prevents the WY-form triangular chunk solver that makes recurrent training competitive with Transformers.   We introduce CARVE (Content-Aware Recurrent with Value Efficiency), which resolves all three problems through one principle: erase only on the key axis. This is provably necessary and sufficient for the WY-form solver to remain valid. Within it, CARVE reuses the recurrent output tensor -- already written to GPU memory -- as a free content signal for the erase gate, and replaces the per-value write-gate
    
[^4]: HyperDFlash：基于门控残差缩减的MHC对齐块级推测解码

    HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction

    [https://arxiv.org/abs/2606.26744](https://arxiv.org/abs/2606.26744)

    针对DeepSeek-V4的多超连接架构，提出了一种通过门控残差缩减实现特征对齐的块级推测解码方法，解决了多路径残差流导致的生成准确率下降问题。

    

    arXiv:2606.26744v1 公告类型：交叉 摘要：我们提出了HyperDFlash，这是一个针对DeepSeek-V4提出的新型多超连接（MHC）架构量身定制的块级并行推测解码框架。尽管DeepSeek-V4原生多令牌预测（MTP）模块在初始令牌生成方面表现强劲，但其后续位置的生成准确性会急剧下降，因为未验证中间令牌的错误累积会降低接受率。虽然原始的DFlash方法支持高效的单次块级生成，但它无法无缝适配到MHC范式，因为DeepSeek-V4的多路径残差流会导致与常规生成设计之间的特征错配。为了解决这一不匹配问题，我们针对MHC残差流提出了两种模型对齐优化。首先，我们采用预折叠残差状态作为唯一的条件信号，保留多路径结构信息，并将生成器与原生的预测路径对齐。

    arXiv:2606.26744v1 Announce Type: cross  Abstract: We present HyperDFlash, a block-parallel speculative decoding framework tailored to the novel multi-hyper-connection (MHC) architecture proposed by DeepSeek-V4. Despite the strong initial-token drafting performance of the native Multi-Token Prediction (MTP) module in DeepSeek-V4, its draft accuracy degrades sharply at later positions, as error accumulation from unverified intermediate tokens harms acceptance rates. Although the original DFlash method supports efficient one-pass block drafting, it cannot be seamlessly adapted to the MHC paradigm, since the multi-path residual stream of DeepSeek-V4 induces feature misalignment with conventional drafting designs. To resolve this mismatch, we propose two model-aligned optimizations for MHC residual streams. First, we adopt pre-collapse residual states as the exclusive conditioning signal, preserving multi-path structural information and aligning the drafter with the native prediction pathw
    
[^5]: 基于顿悟分数的KV缓存淘汰方法：无需注意力矩阵

    Epiphany-Aware KV Cache Eviction Without the Attention Matrix

    [https://arxiv.org/abs/2606.26472](https://arxiv.org/abs/2606.26472)

    本文提出EpiKV，一种通过直接读取模型前向传播中的内部表示变化（顿悟分数）来淘汰KV缓存的方法，无需注意力矩阵，可将可行上下文长度扩展至传统注意力评分方法的16倍，且无需训练或自定义内核。

    

    arXiv:2606.26472v1 公告类型：交叉 摘要：随着推理模型生成长达数万token的思维链，KV缓存日益成为部署瓶颈。现有缓存淘汰方法通过注意力权重对token进行排序，这在长推理轨迹中是一个有噪声的重要性代理，并且通过强制模型实现注意力矩阵，阻碍了生产推理中融合内核的使用。在这项工作中，我们提出了一种称为"顿悟分数"的度量标准来对token进行评分：该分数直接从前向传播中读取模型内部表示的变化，无需注意力矩阵且仅需极少的额外状态。由此产生的缓存淘汰方法EpiKV无需训练、分类器或自定义内核，可直接在FlashAttention推理栈中不变地使用——将可行上下文长度扩展至基于注意力评分方法的16倍。针对上层中间层（负向影响）和下层中间层（正向影响），我们采用因果滚动z分数消除位置趋势。在4096-token缓存设置下，该方法表现出色。

    arXiv:2606.26472v1 Announce Type: cross  Abstract: As reasoning models emit chains of thought tens of thousands of tokens long, KV cache increasingly becomes a deployment bottleneck. Existing cache eviction methods rank tokens by attention weight, which is a noisy importance proxy in long reasoning traces, and prohibits the use of fused kernels in production inference by forcing the model to materialize the attention matrix. In this work, we instead score tokens with a metric we term the epiphany score: the change in the model's internal representation, read directly from the forward pass with no attention matrix and negligible extra state. Our resulting cache eviction method, EpiKV, requires no training, classifier, or custom kernel, and can be used directly in FlashAttention inference stacks unchanged -- scaling to a 16x longer feasible context than attention-based scoring. upper-mid layers negatively) and remove a positional trend with a causal rolling z-score. At a 4096-token cache
    
[^6]: 寻找思考的时间：在实时强化学习中学习规划预算

    Finding the Time to Think: Learning Planning Budgets in Real-Time RL

    [https://arxiv.org/abs/2606.26463](https://arxiv.org/abs/2606.26463)

    提出了一种在实时强化学习中，通过轻量级门控策略动态选择状态依赖的规划预算的方法，有效解决了环境持续运行下的决策延迟问题。

    

    深思熟虑需要时间。在实时环境中，这段时间并非免费。标准强化学习（RL）通过让环境无限期等待智能体的决策来回避这一问题。相反，我们研究了实时强化学习环境，在这种环境中，环境在等待智能体行动的同时仍在持续运行。基于先前的实时形式化方法，我们引入了可变延迟实时强化学习，其中智能体在每个决策点自行决定思考多长时间，因为环境在持续演进。对于我们使用的规划智能体而言，正确的延迟是状态依赖的，而单纯地规划“规划多长时间”可能会使智能体陷入瘫痪。我们转而通过在规划器之上训练一个轻量级的门控策略，来选择状态依赖的规划预算。在实时《吃豆人》、《俄罗斯方块》、《贪吃蛇》、《极速六角棋》和《极速围棋》中，我们的门控策略优于固定预算和启发式基线方法，并且能够迁移到环境具有不同实时约束的实时设置中。

    arXiv:2606.26463v1 Announce Type: new  Abstract: Deliberating takes time. In real-time settings, that time is not free. Standard reinforcement learning (RL) sidesteps this as the environment waits indefinitely for the agent's decision. Instead, we study real-time RL environments where the environment progresses while waiting for the agent's action. Building on prior real-time formalizations, we introduce variable-delay real-time RL, where the agent chooses how long to deliberate at each decision point since the environment progresses. For the planning agents we use, the right delay is state-dependent, and naively planning how long to plan can paralyze the agent. We instead approach this setting by training a lightweight gating policy on top of a planner to select state-dependent planning budgets. Across real-time Pac-Man, Tetris, Snake, Speed Hex, and Speed Go, our gating policy outperforms fixed-budget and heuristic baselines, and transfers to a real-time setup where the environment a
    
[^7]: 基于格理论的无偏规范集值预言机

    Unbiased Canonical Set-Valued Oracles Via Lattice Theory

    [https://arxiv.org/abs/2606.26418](https://arxiv.org/abs/2606.26418)

    本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。

    

    非智能体“预言机”AI在估计未来事件概率时面临自指问题：一旦其答案被学习并采取行动，就会改变它被要求报告的概率本身。针对科学家AI计划所倡导的一种回应是只询问反事实问题，并假设答案没有影响进行评估。我们观察到，这类答案一旦被学习就会变得无关紧要，恰恰是因为其前提随后变为假。因此，我们探索了一种自指替代方案：预言机报告的不是单一概率，而是一个同时无偏且与学习后果自洽的credal集。朴素的自洽性要求被太多集合满足（包括无用的答案[0,1]），因此问题在于挑选出一个规范的、非平凡的成员。我们通过闭包完备格上的Knaster–Tarski不动点定理实现了这一点。

    arXiv:2606.26418v1 Announce Type: new  Abstract: A non-agentic "oracle" AI that estimates probabilities of future events faces a self-reference problem: once its answer is learned and acted upon, it can change the very probability it was asked to report. One response, advocated for the Scientist AI programme, is to ask only counterfactual questions, evaluated as if the answer had no influence. We observe that such answers tend to become irrelevant the moment they are learned, precisely because their premise is then false. We therefore explore a self-referential alternative in which the oracle reports not a single probability but a credal set that is simultaneously unbiased and self-consistent with the consequences of being learned. The naive self-consistency requirement is satisfied by too many sets (including the useless answer $[0,1]$), so the problem is to single out a canonical, nontrivial member. We do so with the Knaster--Tarski fixed-point theorem on the complete lattice of clos
    
[^8]: 红皇后哥德尔机：共同进化的智能体及其评估者

    The Red Queen G\"odel Machine: Co-Evolving Agents and Their Evaluators

    [https://arxiv.org/abs/2606.26294](https://arxiv.org/abs/2606.26294)

    本文提出红皇后哥德尔机（RQGM），通过将评估者纳入进化循环，使智能体能在非平稳评估标准下进行递归自我改进，从而突破静态基准的限制。

    

    arXiv:2606.26294v1 公告类型：交叉 摘要：自我改进的智能体在编程基准测试中已达到最先进水平（SOTA），并最近被扩展到通用领域。然而，它们的搜索方法通常假设一个静态的评估标准：一个固定的验证器、基准测试或标注数据集，在智能体改进过程中保持有效。这忽略了进化的一个核心特征：物种随着环境的变化而适应。我们旨在将同样的原则引入递归自我改进，使评估成为改进循环的一部分，并将搜索开放给不断演化的评估者、对抗性目标和可能超越静态基准的动态效用函数。我们引入了红皇后哥德尔机（RQGM），这是一个用于非平稳效用下递归自我改进的演化框架。RQGM通过受控的效用演化实现了这一点：搜索被组织成具有固定期内评估标准的周期，而效用可以跨周期演化。

    arXiv:2606.26294v1 Announce Type: cross  Abstract: Self-improving agents are state-of-the-art (SOTA) on agentic coding benchmarks and have recently been extended to general domains. However, their search methods generally assume a stationary evaluation criterion: a fixed verifier, benchmark, or labeled dataset that remains valid as the agent improves. This ignores a central feature of evolution: species adapt as their environments change with them. We aim to bring the same principle to recursive self-improvement, making evaluation part of the improvement loop and opening search to evolving evaluators, adversarial objectives, and dynamic utilities that may surpass static benchmarks. We introduce the Red Queen Godel Machine (RQGM), an evolutionary framework for recursive self-improvement under non-stationary utilities. The RQGM makes this possible through controlled utility evolution: search is organized into epochs with a fixed within-epoch evaluation criterion, while the utility can be
    
[^9]: 何时写入与何时抑制：用于记忆辅助知识编辑的路径专用双适配器

    When to Write and When to Suppress: Route-Specialized Dual Adapters for Memory-Assisted Knowledge Editing

    [https://arxiv.org/abs/2606.14668](https://arxiv.org/abs/2606.14668)

    本文提出一种路径专用的双适配器方法，通过相关性路由器决定何时应用编辑记忆或保留原始知识，从而在知识编辑中实现精确更新与无关行为的保护。

    

    arXiv:2606.14668v3 公告类型：替换 摘要：知识编辑系统必须更新选定的事实，同时保留邻近但无关的行为。本文在记忆辅助设置下研究这一问题，在该设置中，推理时会检索编辑记忆，并且一个参数高效的适配器会纠正模型的对象偏好。我们认为，核心设计问题不仅在于如何写入编辑，还在于何时抑制它。我们引入了 \method{}，一个路径专用的双适配编辑器。一个相关性路由器首先决定一个提示是否应该接收编辑记忆。被路由的提示使用一个编辑适配器，该适配器经过训练以偏好新对象而非原始对象；未被路由的非直接提示则使用一个单独的位置适配器，该适配器经过训练以保留或恢复原始对象偏好。我们在三种包含1000个案例的协议（\cf{}、\zsre{} 和 \mquake{}）上，在相同的记忆协议和两个7B/8B基础模型下评估了 \method{}。在 Llama-3.1-8B-Instruct 上，\method{} 取得了最佳性能。

    arXiv:2606.14668v3 Announce Type: replace  Abstract: Knowledge editing systems must update selected facts while preserving nearby but irrelevant behavior. This paper studies this problem in a memory-assisted setting where an edit memory is retrieved at inference time and a parameter-efficient adapter corrects the model's object preference. We argue that the central design question is not only how to write an edit, but also when to suppress it. We introduce \method{}, a route-specialized dual-adapter editor. A relevance router first decides whether a prompt should receive an edit memory. Routed prompts use an edit adapter trained to prefer the new object over the original object; unrouted non-direct prompts use a separate locality adapter trained to preserve or restore the original-object preference. We evaluate \method{} on three 1,000-case protocols, \cf{}, \zsre{}, and \mquake{}, under the same memory protocol and two 7B/8B base models. On Llama-3.1-8B-Instruct, \method{} obtains the
    
[^10]: 基于历史强化学习与潜在模型自适应的闭环二氧化碳封存控制

    Closed-Loop CO2 Storage Control With History-Based Reinforcement Learning and Latent Model-Based Adaptation

    [https://arxiv.org/abs/2605.02405](https://arxiv.org/abs/2605.02405)

    本文提出了一种结合历史强化学习和潜在模型自适应的闭环二氧化碳封存控制方法，通过利用时间井响应信息和自适应机制，有效应对储层不确定性和动态变化。

    

    地质二氧化碳封存的闭环管理需要能够适应不确定储层行为的控制策略，同时依赖运营期间实际可获得的观测数据。本文将二氧化碳注入和盐水生产控制建模为一个部分可观测的序贯决策问题，并研究使用高保真储层模拟训练的、可部署的深度强化学习控制器。我们首先比较了特权状态、仅井数据、历史条件、掩蔽课程和非对称师生无模型策略，以量化时间井响应信息和训练时特权模拟器状态的价值。随后，我们评估了一种基于潜在模型的自适应流水线，该流水线重用标称潜在动力学，并在已知注入器故障、泄漏引发的动力学和奖励变化以及分隔储层连通性下重新调整控制器。结果表明，历史条件策略显著优于仅井数据方法，而基于潜在模型的自适应在应对动态变化时表现出鲁棒性。

    arXiv:2605.02405v2 Announce Type: replace  Abstract: Closed-loop management of geological CO2 storage requires control policies that adapt to uncertain reservoir behavior while relying on observations that are realistically available during operation. This work formulates CO2 injection and brine-production control as a partially observable sequential decision problem and studies deployable deep reinforcement-learning controllers trained with high-fidelity reservoir simulation. We first compare privileged-state, well-only, history-conditioned, masking-curriculum, and asymmetric teacher-student model-free policies in order to quantify the value of temporal well-response information and training-time privileged simulator states. We then evaluate a latent model-based adaptation pipeline that reuses nominal latent dynamics and retunes controllers under known injector failure, leakage-induced dynamics and reward shift, and compartmentalized reservoir connectivity. The results show that histo
    
[^11]: 通过层间一致性聚合缓解大型视觉语言模型中的幻觉

    Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models

    [https://arxiv.org/abs/2505.12343](https://arxiv.org/abs/2505.12343)

    本文提出了一种名为DCLA的免训练解码方法，通过聚合前层表示构建动态语义参考来纠正语义偏差的层，从而有效缓解大型视觉语言模型中的幻觉问题，并在多个模型和基准上显著提升性能。

    

    arXiv:2505.12343v2 公告类型：替换交叉 摘要：尽管大型视觉语言模型（LVLMs）能力令人印象深刻，但它们仍然容易产生幻觉，即生成的内容与输入图像不一致。现有的免训练幻觉缓解方法通常存在性能不稳定且对超参数设置高度敏感的问题，这限制了其实用性和更广泛的采用。在本文中，我们提出了一种通过层聚合进行层间一致性解码的方法（DCLA），这是一种免训练的解码机制，无需重新训练、微调或访问外部知识库。具体来说，DCLA通过聚合前几层的表示来构建动态语义参考，并用其纠正语义偏差的层，从而强制实现层间一致性。在七个LVLMs和多个基准上的实验证明了DCLA的通用性：它在MME指标上比标准解码高出28.58分。

    arXiv:2505.12343v2 Announce Type: replace-cross  Abstract: Despite the impressive capabilities of Large Vision-Language Models (LVLMs), they remain susceptible to hallucinations, where generated content is inconsistent with the input image. Existing training-free hallucination mitigation methods often suffer from unstable performance and high sensitivity to hyperparameter settings, which limits their practicality and broader adoption. In this paper, we propose Decoding with Inter-layer Consistency via Layer Aggregation (DCLA), a training-free decoding mechanism that requires no retraining, fine-tuning, or access to external knowledge bases. Specifically, DCLA constructs a dynamic semantic reference by aggregating representations from previous layers and uses it to correct semantically deviated layers, thereby enforcing inter-layer consistency. Experiments across seven LVLMs and multiple benchmarks demonstrate the generality of DCLA: it surpasses standard decoding by 28.58 MME points on
    
[^12]: 大型（和深度）因子模型

    Large (and Deep) Factor Models

    [https://arxiv.org/abs/2402.06635](https://arxiv.org/abs/2402.06635)

    本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。

    

    我们打开了深度学习在投资组合优化中的黑盒子，并证明了一个足够宽而任意深的神经网络(DNN)被训练用来最大化随机贴现因子(SDF)的夏普比率等效于一个大型因子模型(LFM)：一个使用许多非线性特征的线性因子定价模型。这些特征的性质取决于DNN的体系结构，在一种明确可追踪的方式下。这使得首次可以推导出封闭形式的端到端训练的基于DNN的SDF。我们通过实证评估了LFMs，并展示了各种架构选择如何影响SDF的性能。我们证明了深度复杂性的优点：随着足够多的数据，DNN-SDF的外样总体表现会随着神经网络的深度而增加，当隐藏层达到约100层时达到饱和。

    We open up the black box behind Deep Learning for portfolio optimization and prove that a sufficiently wide and arbitrarily deep neural network (DNN) trained to maximize the Sharpe ratio of the Stochastic Discount Factor (SDF) is equivalent to a large factor model (LFM): A linear factor pricing model that uses many non-linear characteristics. The nature of these characteristics depends on the architecture of the DNN in an explicit, tractable fashion. This makes it possible to derive end-to-end trained DNN-based SDFs in closed form for the first time. We evaluate LFMs empirically and show how various architectural choices impact SDF performance. We document the virtue of depth complexity: With enough data, the out-of-sample performance of DNN-SDF is increasing in the NN depth, saturating at huge depths of around 100 hidden layers.
    
[^13]: 在连续治疗下的多重稳健因果中介分析

    Multiply Robust Causal Mediation Analysis with Continuous Treatments

    [https://arxiv.org/abs/2105.09254](https://arxiv.org/abs/2105.09254)

    本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。

    

    在许多应用中，研究人员对治疗或暴露对感兴趣的结果的直接和间接的因果效应。中介分析为鉴定和估计这些因果效应提供了一个严谨的框架。对于二元治疗，Tchetgen Tchetgen和Shpitser (2012)提出了直接和间接效应的高效估计器，基于参数的影响函数。这些估计器具有良好的性质，如多重稳健性和渐近正态性，同时允许对干扰参数进行低于根号n的收敛速度。然而，在涉及连续治疗的情况下，这些基于影响函数的估计器没有准备好应用，除非进行强参数假设。在这项工作中，我们利用核平滑方法提出了一种适用于连续治疗环境的估计器，受到Tchetgen Tchetgen的影响函数估计器的启发。

    In many applications, researchers are interested in the direct and indirect causal effects of a treatment or exposure on an outcome of interest. Mediation analysis offers a rigorous framework for identifying and estimating these causal effects. For binary treatments, efficient estimators for the direct and indirect effects are presented in Tchetgen Tchetgen and Shpitser (2012) based on the influence function of the parameter of interest. These estimators possess desirable properties, such as multiple-robustness and asymptotic normality, while allowing for slower than root-n rates of convergence for the nuisance parameters. However, in settings involving continuous treatments, these influence function-based estimators are not readily applicable without making strong parametric assumptions. In this work, utilizing a kernel-smoothing approach, we propose an estimator suitable for settings with continuous treatments inspired by the influence function-based estimator of Tchetgen Tchetgen an
    
[^14]: 颗粒球计算：一种高效、鲁棒和可解释的自适应多粒度表示和计算方法

    Granular ball computing: an efficient, robust, and interpretable adaptive multi-granularity representation and computation method. (arXiv:2304.11171v1 [cs.LG])

    [http://arxiv.org/abs/2304.11171](http://arxiv.org/abs/2304.11171)

    本文提出了一种基于颗粒球计算的自适应多粒度表示和计算方法，能够提高机器学习的效率、鲁棒性和可解释性。

    

    人类认知具有“先大后小”的认知机制，因此具有自适应的多粒度描述能力。这导致了有效性、鲁棒性和可解释性等计算特性。本文提出了一种新的基于颗粒球计算的自适应多粒度表示和计算方法。他们将这种方法应用于几个机器学习任务，并证明其相对于其他最先进的方法的有效性。

    Human cognition has a ``large-scale first'' cognitive mechanism, therefore possesses adaptive multi-granularity description capabilities. This results in computational characteristics such as efficiency, robustness, and interpretability. Although most existing artificial intelligence learning methods have certain multi-granularity features, they do not fully align with the ``large-scale first'' cognitive mechanism. Multi-granularity granular-ball computing is an important model method developed in recent years. This method can use granular-balls of different sizes to adaptively represent and cover the sample space, and perform learning based on granular-balls. Since the number of coarse-grained "granular-ball" is smaller than the number of sample points, granular-ball computing is more efficient; the coarse-grained characteristics of granular-balls are less likely to be affected by fine-grained sample points, making them more robust; the multi-granularity structure of granular-balls ca
    

