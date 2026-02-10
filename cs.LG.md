# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [KIX: A Metacognitive Generalization Framework](https://arxiv.org/abs/2402.05346) | 人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。 |
| [^2] | [Rates of Convergence in the Central Limit Theorem for Markov Chains, with an Application to TD Learning](https://arxiv.org/abs/2401.15719) | 本研究证明了一个非渐近的中心极限定理，并通过应用于TD学习，展示了其实际应用的可行性。 |
| [^3] | [YaRN: Efficient Context Window Extension of Large Language Models.](http://arxiv.org/abs/2309.00071) | YaRN是一种高效的上下文窗口扩展方法，可以在大型语言模型中有效利用和推断比原始预训练允许的上下文长度更长的上下文，同时超越了之前的最新研究成果。 |
| [^4] | [Estimating the Value of Evidence-Based Decision Making.](http://arxiv.org/abs/2306.13681) | 本文提出了一个实证框架，用于估算证据决策的价值和统计精度投资回报。 |
| [^5] | [UNREAL:Unlabeled Nodes Retrieval and Labeling for Heavily-imbalanced Node Classification.](http://arxiv.org/abs/2303.10371) | 本文提出了一种用于重度不平衡节点分类的迭代过采样方法UNREAL，通过添加未标记节点而不是合成节点，解决了特征和邻域生成的难题，并利用节点嵌入空间中的无监督学习进行几何排名来有效地校准伪标签分配。 |

# 详细

[^1]: KIX: 一种元认知泛化框架

    KIX: A Metacognitive Generalization Framework

    [https://arxiv.org/abs/2402.05346](https://arxiv.org/abs/2402.05346)

    人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。

    

    人类和其他动物能够灵活解决各种任务，并且能够通过重复使用和应用长期积累的高级知识来适应新颖情境，这表现了一种泛化智能行为。但是人工智能代理更多地是专家，缺乏这种通用行为。人工智能代理需要理解和利用关键的结构化知识表示。我们提出了一种元认知泛化框架，称为Knowledge-Interaction-eXecution (KIX)，并且认为通过与对象的交互来利用类型空间可以促进学习可迁移的交互概念和泛化能力。这是将知识融入到强化学习中的一种自然方式，并有望成为人工智能系统中实现自主和通用行为的推广者。

    Humans and other animals aptly exhibit general intelligence behaviors in solving a variety of tasks with flexibility and ability to adapt to novel situations by reusing and applying high level knowledge acquired over time. But artificial agents are more of a specialist, lacking such generalist behaviors. Artificial agents will require understanding and exploiting critical structured knowledge representations. We present a metacognitive generalization framework, Knowledge-Interaction-eXecution (KIX), and argue that interactions with objects leveraging type space facilitate the learning of transferable interaction concepts and generalization. It is a natural way of integrating knowledge into reinforcement learning and promising to act as an enabler for autonomous and generalist behaviors in artificial intelligence systems.
    
[^2]: 关于马尔可夫链中心极限定理的收敛速度，及其在TD学习中的应用

    Rates of Convergence in the Central Limit Theorem for Markov Chains, with an Application to TD Learning

    [https://arxiv.org/abs/2401.15719](https://arxiv.org/abs/2401.15719)

    本研究证明了一个非渐近的中心极限定理，并通过应用于TD学习，展示了其实际应用的可行性。

    

    我们使用Stein方法证明了一个非渐近的、矢量值鞅差的中心极限定理，并利用泊松方程将结果推广到马尔可夫链的函数。然后我们展示了这些结果可以应用于建立基于平均的非渐近的TD学习的中心极限定理。

    We prove a non-asymptotic central limit theorem for vector-valued martingale differences using Stein's method, and use Poisson's equation to extend the result to functions of Markov Chains. We then show that these results can be applied to establish a non-asymptotic central limit theorem for Temporal Difference (TD) learning with averaging.
    
[^3]: YaRN: 大型语言模型的高效上下文窗口扩展方法

    YaRN: Efficient Context Window Extension of Large Language Models. (arXiv:2309.00071v1 [cs.CL])

    [http://arxiv.org/abs/2309.00071](http://arxiv.org/abs/2309.00071)

    YaRN是一种高效的上下文窗口扩展方法，可以在大型语言模型中有效利用和推断比原始预训练允许的上下文长度更长的上下文，同时超越了之前的最新研究成果。

    

    旋转位置嵌入（RoPE）已被证明可以有效地编码transformer-based语言模型中的位置信息。然而，这些模型在超过它们训练的序列长度时无法泛化。我们提出了YaRN（Yet another RoPE extensioN method），一种计算高效的方法来扩展这些模型的上下文窗口，需要的tokens数量和训练步骤少于之前的方法的10倍和2.5倍。使用YaRN，我们展示了LLaMA模型可以有效地利用和推断比原始预训练允许的上下文长度更长的上下文，并且在上下文窗口扩展方面超过了之前的最新研究成果。此外，我们还展示了YaRN具有超越微调数据集有限上下文的能力。我们在https://github.com/jquesnelle/yarn上发布了使用64k和128k上下文窗口进行Fine-tuning的Llama 2 7B/13B的检查点。

    Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length they were trained on. We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models, requiring 10x less tokens and 2.5x less training steps than previous methods. Using YaRN, we show that LLaMA models can effectively utilize and extrapolate to context lengths much longer than their original pre-training would allow, while also surpassing previous the state-of-the-art at context window extension. In addition, we demonstrate that YaRN exhibits the capability to extrapolate beyond the limited context of a fine-tuning dataset. We publish the checkpoints of Llama 2 7B/13B fine-tuned using YaRN with 64k and 128k context windows at https://github.com/jquesnelle/yarn
    
[^4]: 估算基于证据决策的价值

    Estimating the Value of Evidence-Based Decision Making. (arXiv:2306.13681v1 [stat.ME])

    [http://arxiv.org/abs/2306.13681](http://arxiv.org/abs/2306.13681)

    本文提出了一个实证框架，用于估算证据决策的价值和统计精度投资回报。

    

    商业/政策决策通常基于随机实验和观察性研究的证据。本文提出了一个实证框架来估算基于证据的决策（EBDM）的价值和统计精度投资回报。

    Business/policy decisions are often based on evidence from randomized experiments and observational studies. In this article we propose an empirical framework to estimate the value of evidence-based decision making (EBDM) and the return on the investment in statistical precision.
    
[^5]: UNREAL: 用于重度不平衡节点分类的未标记节点检索和标记方法

    UNREAL:Unlabeled Nodes Retrieval and Labeling for Heavily-imbalanced Node Classification. (arXiv:2303.10371v1 [cs.LG])

    [http://arxiv.org/abs/2303.10371](http://arxiv.org/abs/2303.10371)

    本文提出了一种用于重度不平衡节点分类的迭代过采样方法UNREAL，通过添加未标记节点而不是合成节点，解决了特征和邻域生成的难题，并利用节点嵌入空间中的无监督学习进行几何排名来有效地校准伪标签分配。

    

    在现实世界的节点分类任务中，极度倾斜的标签分布很常见。如果不合适地处理，这对少数类别的GNNs性能会有极大的影响。由于其实用性，最近一系列的研究都致力于解决这个难题。现有的过采样方法通过产生“假”的少数节点和合成其特征和局部拓扑来平滑标签分布，这在很大程度上忽略了图上未标记节点的丰富信息。在本文中，我们提出了UNREAL，一种迭代过采样方法。第一个关键区别在于，我们只添加未标记节点而不是合成节点，这消除了特征和邻域生成的挑战。为了选择要添加的未标记节点，我们提出了几何排名来对未标记节点进行排名。几何排名利用节点嵌入空间中的无监督学习来有效地校准伪标签分配。最后，我们确定了问题。

    Extremely skewed label distributions are common in real-world node classification tasks. If not dealt with appropriately, it significantly hurts the performance of GNNs in minority classes. Due to its practical importance, there have been a series of recent research devoted to this challenge. Existing over-sampling techniques smooth the label distribution by generating ``fake'' minority nodes and synthesizing their features and local topology, which largely ignore the rich information of unlabeled nodes on graphs. In this paper, we propose UNREAL, an iterative over-sampling method. The first key difference is that we only add unlabeled nodes instead of synthetic nodes, which eliminates the challenge of feature and neighborhood generation. To select which unlabeled nodes to add, we propose geometric ranking to rank unlabeled nodes. Geometric ranking exploits unsupervised learning in the node embedding space to effectively calibrates pseudo-label assignment. Finally, we identify the issu
    

