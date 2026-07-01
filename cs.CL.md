# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nemotron-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context](https://arxiv.org/abs/2606.26493) | 提出TwoTower架构，通过将上下文表示与去噪解耦为两个独立塔，在保持高质量的同时大幅提升扩散语言模型的生成吞吐量。 |
| [^2] | [The Annotation Scarcity Paradox in Low-Resource NLP Evaluation: A Decade of Acceleration and Emerging Constraints](https://arxiv.org/abs/2605.19066) | 本文揭示了低资源NLP评估中一个关键悖论：技术扩展能力远超真实评估所需的人类基础设施，导致评估瓶颈和结构性不平等。 |

# 详细

[^1]: Nemotron-TwoTower：基于预训练自回归上下文的扩散语言建模

    Nemotron-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context

    [https://arxiv.org/abs/2606.26493](https://arxiv.org/abs/2606.26493)

    提出TwoTower架构，通过将上下文表示与去噪解耦为两个独立塔，在保持高质量的同时大幅提升扩散语言模型的生成吞吐量。

    

    arXiv:2606.26493v1 公告类型：新论文 摘要：扩散语言模型因其并行和迭代生成的潜力，为自回归模型提供了一种有前景的替代方案。然而，现有方法使用单一网络同时处理上下文表示和迭代去噪，迫使一个模型承担两种角色，从而限制了其在任一角色上的能力。我们提出TwoTower，一种逐块自回归扩散模型，它将这两种角色解耦为两个塔：一个冻结的自回归上下文塔，用于因果地处理干净标记；一个可训练的扩散去噪器塔，具有双向块注意力机制，通过交叉注意力与上下文塔交互，对噪声块进行精炼。该模型基于Nemotron-3-Nano-30B-A3B（一个开源的30B混合Mamba-Transformer MoE模型）构建，并在约2.1T个标记上训练。Nemotron-TwoTower在保持自回归基线模型98.7%质量的同时，实现了2.42倍的实际时钟生成吞吐量提升。我们在ht处发布代码和模型权重。

    arXiv:2606.26493v1 Announce Type: new  Abstract: Diffusion language models offer a promising alternative to autoregressive models due to their potential for parallel and iterative generation. However, existing approaches use a single network for both context representation and iterative denoising, forcing one model to serve both roles and limiting its capacity for either role. We propose TwoTower, a block-wise autoregressive diffusion model that decouples these roles into two towers: a frozen AR context tower that causally processes clean tokens, and a trainable diffusion denoiser tower with bidirectional block attention that refines noisy blocks via cross-attention to the context. Built on Nemotron-3-Nano-30B-A3B, an open-weight 30B hybrid Mamba-Transformer MoE model, and trained on approximately 2.1T tokens, Nemotron-TwoTower retains 98.7% of the autoregressive baseline's quality while offering 2.42X higher wall-clock generation throughput. We release the code and model weights at ht
    
[^2]: 低资源自然语言处理评估中的标注稀缺悖论：加速发展的十年与新兴的制约因素

    The Annotation Scarcity Paradox in Low-Resource NLP Evaluation: A Decade of Acceleration and Emerging Constraints

    [https://arxiv.org/abs/2605.19066](https://arxiv.org/abs/2605.19066)

    本文揭示了低资源NLP评估中一个关键悖论：技术扩展能力远超真实评估所需的人类基础设施，导致评估瓶颈和结构性不平等。

    

    arXiv:2605.19066v2 公告类型：替换 摘要：在过去十年中，低资源自然语言处理（NLP）经历了爆炸式增长，这得益于跨语言迁移、大规模多语言模型以及基准测试的迅速普及。然而，这种表面上的进步掩盖了一个关键且未得到充分审视的矛盾：评估日益复杂的生成系统所需的深层社会语言学专业知识严重不足、分布不均，并在结构上被边缘化。我们呈现了一项对低资源NLP评估（2014年至今）的批判性叙事综述，追溯其演变过程，涵盖三个阶段：早期的启发式乐观主义、自上而下的基准扩展幻觉，以及当前生成瓶颈的时代。我们提出了“标注稀缺悖论”这一概念，即当技术能力大规模扩展模型的速度远远超过主权人类基础设施真实评估它们所需的能力时，所产生的结构性摩擦。

    arXiv:2605.19066v2 Announce Type: replace  Abstract: Over the past decade, low-resource natural language processing (NLP) has experienced explosive growth, propelled by cross-lingual transfer, massively multilingual models, and the rapid proliferation of benchmarks. Yet this apparent progress masks a critical, insufficiently examined tension: the deep sociolinguistic expertise required to evaluate increasingly complex generative systems is severely strained, inequitably distributed, and structurally marginalised. We present a critical narrative survey of low-resource NLP evaluation (2014--present), tracing its evolution across three phases: early heuristic optimism, the illusions of top-down benchmark scaling, and the current era of generative bottlenecks. We conceptualise the \emph{Annotation Scarcity Paradox}, the structural friction arising when the technical capacity to scale models vastly outpaces the sovereign human infrastructure required to authentically evaluate them. By exami
    

