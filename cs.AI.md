# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Helmsman of the Masses? Evaluate the Opinion Leadership of Large Language Models in the Werewolf Game](https://arxiv.org/abs/2404.01602) | 本研究通过狼人游戏模拟平台评估了大语言模型的观点领导作用，并开发了两个新的评估指标。 |
| [^2] | [GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM](https://arxiv.org/abs/2403.05527) | GEAR提出了一种高效的KV缓存压缩框架，实现几乎无损的高比率压缩，用于解决大型语言模型推断中因缓存需求增长而导致的记忆绑定问题和性能下降。 |
| [^3] | [FRRI: a novel algorithm for fuzzy-rough rule induction](https://arxiv.org/abs/2403.04447) | 结合模糊与粗糙集理论，提出一种新颖的模糊-粗糙规则归纳算法 FRRI。 |
| [^4] | [Multi-Constraint Safe RL with Objective Suppression for Safety-Critical Applications](https://arxiv.org/abs/2402.15650) | 提出了一种目标抑制的新方法，可以在多约束安全领域中改进安全强化学习任务表现，实验证明此方法结合现有算法能够在减少约束违规的情况下实现与基准线相当的任务奖励水平。 |
| [^5] | [WildfireGPT: Tailored Large Language Model for Wildfire Analysis](https://arxiv.org/abs/2402.07877) | WildfireGPT是一个针对野火分析的定制化大型语言模型，通过提供领域特定的上下文信息和科学准确性，将用户查询转化为关于野火风险的可操作见解。 |
| [^6] | [Exploring the Limitations of Graph Reasoning in Large Language Models](https://arxiv.org/abs/2402.01805) | 本文测试了5种不同的大型语言模型在图推理问题上的推理深度，并发现了LLMs的局限性、偏见和属性。我们发现LLMs对于节点遍历自由度的平均度数呈反向关系，k-shot提示对图推理任务有负面影响，并且LLMs存在积极的回应偏差，无法识别有效解的缺失。我们还提出了一种新的图推理提示技术。 |
| [^7] | [DiffiT: Diffusion Vision Transformers for Image Generation](https://arxiv.org/abs/2312.02139) | DiffiT是一种新的模型，结合了Vision Transformer和扩散模型的优势，在图像生成中表现出色，特别是通过引入细粒度去噪控制和时间依赖的多头自注意力机制，实现了高保真图像的生成。 |
| [^8] | [Systematic AI Approach for AGI: Addressing Alignment, Energy, and AGI Grand Challenges.](http://arxiv.org/abs/2310.15274) | 本论文讨论了面临能源、对齐和从狭义人工智能到AGI的三大挑战的系统化人工智能方法。现有的人工智能方法在能源消耗、系统设计和对齐问题上存在不足，而系统设计在解决对齐、能源和AGI大挑战中是至关重要的。 |
| [^9] | [Interpretable Distribution-Invariant Fairness Measures for Continuous Scores.](http://arxiv.org/abs/2308.11375) | 对于连续评分，我们提出了一种基于Wasserstein距离的分布不变公平性度量方法，能够解释度量结果并适用于比较不同模型、数据集或时间点之间的偏差。 |
| [^10] | [Summaries, Highlights, and Action items: Design, implementation and evaluation of an LLM-powered meeting recap system.](http://arxiv.org/abs/2307.15793) | 这项研究设计、实现和评估了一种基于LLM的会议总结系统，通过减少个人会议负担和增加会议输出的清晰度和一致性，提高了会议体验。 |
| [^11] | [Convergence and sample complexity of natural policy gradient primal-dual methods for constrained MDPs.](http://arxiv.org/abs/2206.02346) | 本文研究了约束马尔可夫决策过程中优化问题的自然策略梯度原始-对偶方法。通过自然策略梯度上升和投影次梯度下降更新变量，我们的方法在全局收敛中实现了次线性速率，而且不受状态-动作空间大小限制。 |

# 详细

[^1]: 大语言模型在狼人游戏中的舵手？评估其观点引领作用

    Helmsman of the Masses? Evaluate the Opinion Leadership of Large Language Models in the Werewolf Game

    [https://arxiv.org/abs/2404.01602](https://arxiv.org/abs/2404.01602)

    本研究通过狼人游戏模拟平台评估了大语言模型的观点领导作用，并开发了两个新的评估指标。

    

    大语言模型（LLMs）在社交推理游戏中展现出令人难忘的战略行为。然而，LLM代理所展示的观点领导力的重要性被忽视了，而这对于多智能体和人工智能交互设置中的实际应用至关重要。在此研究中，我们利用狼人游戏作为模拟平台，评估LLMs的观点引领作用。该游戏中有警长角色，负责总结论据并推荐决策选项，因此可作为观点领袖的可信代理。我们开发了一个整合了警长角色的框架，并设计了两个基于观点领袖关键特征的新指标进行评估。第一个度量标准衡量观点领袖的可靠性，第二个评估...

    arXiv:2404.01602v1 Announce Type: cross  Abstract: Large language models (LLMs) have exhibited memorable strategic behaviors in social deductive games. However, the significance of opinion leadership exhibited by LLM-based agents has been overlooked, which is crucial for practical applications in multi-agent and human-AI interaction settings. Opinion leaders are individuals who have a noticeable impact on the beliefs and behaviors of others within a social group. In this work, we employ the Werewolf game as a simulation platform to assess the opinion leadership of LLMs. The game features the role of the Sheriff, tasked with summarizing arguments and recommending decision options, and therefore serves as a credible proxy for an opinion leader. We develop a framework integrating the Sheriff role and devise two novel metrics for evaluation based on the critical characteristics of opinion leaders. The first metric measures the reliability of the opinion leader, and the second assesses the 
    
[^2]: GEAR: 一种用于几乎无损生成推断大型语言模型的高效KV缓存压缩方案

    GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM

    [https://arxiv.org/abs/2403.05527](https://arxiv.org/abs/2403.05527)

    GEAR提出了一种高效的KV缓存压缩框架，实现几乎无损的高比率压缩，用于解决大型语言模型推断中因缓存需求增长而导致的记忆绑定问题和性能下降。

    

    关键-值（KV）缓存已成为加快大型语言模型（LLMs）推断生成速度的事实标准。然而，随着序列长度增加而增长的缓存需求已将LLM推断转变为一个记忆绑定问题，显著地限制了系统吞吐量。现有方法依赖于丢弃不重要的标记或均匀量化所有条目。然而，这种方法往往会产生较高的近似误差来表示压缩后的矩阵。自回归解码过程进一步增加了每个步骤的误差，导致模型生成中的重大偏差和性能恶化。为了解决这一挑战，我们提出了GEAR，一种高效的KV缓存压缩框架，实现几乎无损的高压缩比。

    arXiv:2403.05527v1 Announce Type: cross  Abstract: Key-value (KV) caching has become the de-facto to accelerate generation speed for large language models (LLMs) inference. However, the growing cache demand with increasing sequence length has transformed LLM inference to be a memory bound problem, significantly constraining the system throughput. Existing methods rely on dropping unimportant tokens or quantizing all entries uniformly. Such methods, however, often incur high approximation errors to represent the compressed matrices. The autoregressive decoding process further compounds the error of each step, resulting in critical deviation in model generation and deterioration of performance. To tackle this challenge, we propose GEAR, an efficient KV cache compression framework that achieves near-lossless high-ratio compression. GEAR first applies quantization to majority of entries of similar magnitudes to ultra-low precision. It then employs a low rank matrix to approximate the quant
    
[^3]: FRRI：一种新颖的模糊-粗糙规则归纳算法

    FRRI: a novel algorithm for fuzzy-rough rule induction

    [https://arxiv.org/abs/2403.04447](https://arxiv.org/abs/2403.04447)

    结合模糊与粗糙集理论，提出一种新颖的模糊-粗糙规则归纳算法 FRRI。

    

    可解释性是机器学习研究的下一个前沿。在寻找白盒模型的过程中-与随机森林或神经网络等黑盒模型相对应，规则归纳算法是一个合乎逻辑且有希望的选择，因为规则可以被人类轻松理解。模糊和粗糙集理论已成功应用于这种原型，几乎总是分开应用。由于规则归纳的两种方法均涉及基于等价类概念的粒计算，将它们结合是自然的选择。QuickRules算法是利用模糊粗糙集理论进行规则归纳的第一次尝试。它基于QuickReduct，这是一个用于构建决策约简的贪婪算法。QuickRules 已经展示了相比其他规则归纳方法的改进。然而，要评估模糊-粗糙规则归纳算法的全部潜力，就需要从基础开始。在本文中，

    arXiv:2403.04447v1 Announce Type: cross  Abstract: Interpretability is the next frontier in machine learning research. In the search for white box models - as opposed to black box models, like random forests or neural networks - rule induction algorithms are a logical and promising option, since the rules can easily be understood by humans. Fuzzy and rough set theory have been successfully applied to this archetype, almost always separately. As both approaches to rule induction involve granular computing based on the concept of equivalence classes, it is natural to combine them. The QuickRules\cite{JensenCornelis2009} algorithm was a first attempt at using fuzzy rough set theory for rule induction. It is based on QuickReduct, a greedy algorithm for building decision reducts. QuickRules already showed an improvement over other rule induction methods. However, to evaluate the full potential of a fuzzy rough rule induction algorithm, one needs to start from the foundations. In this paper,
    
[^4]: 具有目标抑制的多约束安全强化学习用于安全关键应用

    Multi-Constraint Safe RL with Objective Suppression for Safety-Critical Applications

    [https://arxiv.org/abs/2402.15650](https://arxiv.org/abs/2402.15650)

    提出了一种目标抑制的新方法，可以在多约束安全领域中改进安全强化学习任务表现，实验证明此方法结合现有算法能够在减少约束违规的情况下实现与基准线相当的任务奖励水平。

    

    尽管在现实世界中非常常见，但具有多个约束条件的安全强化学习任务仍然是一个具有挑战性的领域。为了解决这一挑战，我们提出了一种新方法，即目标抑制，根据安全评判器自适应地抑制任务奖励最大化目标。我们在两个多约束安全领域中对目标抑制进行了基准测试，包括一个自动驾驶领域，在这个领域中任何错误的行为都可能导致灾难性后果。实证结果表明，我们提出的方法与现有的安全强化学习算法相结合，可以在显著减少约束违规的情况下匹配我们的基准线所达到的任务奖励。

    arXiv:2402.15650v1 Announce Type: cross  Abstract: Safe reinforcement learning tasks with multiple constraints are a challenging domain despite being very common in the real world. To address this challenge, we propose Objective Suppression, a novel method that adaptively suppresses the task reward maximizing objectives according to a safety critic. We benchmark Objective Suppression in two multi-constraint safety domains, including an autonomous driving domain where any incorrect behavior can lead to disastrous consequences. Empirically, we demonstrate that our proposed method, when combined with existing safe RL algorithms, can match the task reward achieved by our baselines with significantly fewer constraint violations.
    
[^5]: WildfireGPT：针对野火分析的定制化大型语言模型

    WildfireGPT: Tailored Large Language Model for Wildfire Analysis

    [https://arxiv.org/abs/2402.07877](https://arxiv.org/abs/2402.07877)

    WildfireGPT是一个针对野火分析的定制化大型语言模型，通过提供领域特定的上下文信息和科学准确性，将用户查询转化为关于野火风险的可操作见解。

    

    大型语言模型（LLMs）的最新进展代表了人工智能（AI）和机器学习（ML）领域的一种变革性能力。然而，LLMs是通用模型，训练于广泛的文本语料库，往往难以提供特定上下文信息，尤其是在需要专业知识的领域，比如野火细节在更广泛的气候变化背景下。对于关注野火弹性和适应性的决策者和政策制定者来说，获取不仅准确而且领域特定的响应至关重要，而不是泛泛而谈。为此，我们开发了WildfireGPT，一个原型LLM代理，旨在将用户查询转化为关于野火风险的可操作见解。我们通过提供气候预测和科学文献等额外上下文信息来丰富WildfireGPT，以确保其信息具有时效性、相关性和科学准确性。这使得WildfireGPT成为一个有效的工具来解决实际问题。

    The recent advancement of large language models (LLMs) represents a transformational capability at the frontier of artificial intelligence (AI) and machine learning (ML). However, LLMs are generalized models, trained on extensive text corpus, and often struggle to provide context-specific information, particularly in areas requiring specialized knowledge such as wildfire details within the broader context of climate change. For decision-makers and policymakers focused on wildfire resilience and adaptation, it is crucial to obtain responses that are not only precise but also domain-specific, rather than generic. To that end, we developed WildfireGPT, a prototype LLM agent designed to transform user queries into actionable insights on wildfire risks. We enrich WildfireGPT by providing additional context such as climate projections and scientific literature to ensure its information is current, relevant, and scientifically accurate. This enables WildfireGPT to be an effective tool for del
    
[^6]: 探索大型语言模型中图推理的局限性

    Exploring the Limitations of Graph Reasoning in Large Language Models

    [https://arxiv.org/abs/2402.01805](https://arxiv.org/abs/2402.01805)

    本文测试了5种不同的大型语言模型在图推理问题上的推理深度，并发现了LLMs的局限性、偏见和属性。我们发现LLMs对于节点遍历自由度的平均度数呈反向关系，k-shot提示对图推理任务有负面影响，并且LLMs存在积极的回应偏差，无法识别有效解的缺失。我们还提出了一种新的图推理提示技术。

    

    预训练的大型语言模型仅通过基于语言的提示就展示了各种类型的推理能力。然而，在本文中，我们通过图推理问题测试了5种不同的大型语言模型（GPT-4，GPT-3.5，Claude-2，Llama-2和Palm-2）的推理深度。特别地，我们设计了10个不同的图遍历问题，每个问题代表着逐步增加的复杂性水平。此外，我们通过对不同图大小以及不同形式的k-shot提示的设置分析了模型的性能。通过这个基准测试过程，我们凸显了LLMs的各种局限性、偏见和属性，比如与每个节点的遍历自由度的平均度数呈反向关系，k-shot提示对图推理任务的整体负面影响，以及积极的回应偏差导致LLMs无法识别有效解的缺失。最后，我们提出一种新的提示技术，专门用于图推理。

    Pretrained Large Language Models have demonstrated various types of reasoning capabilities through language-based prompts alone. However, in this paper, we test the depth of graph reasoning for 5 different LLMs (GPT-4, GPT-3.5, Claude-2, Llama-2 and Palm-2) through the problems of graph reasoning. In particular, we design 10 distinct problems of graph traversal, each representing increasing levels of complexity. Further, we analyze the performance of models across various settings such as varying sizes of graphs as well as different forms of k-shot prompting. We highlight various limitations, biases, and properties of LLMs through this benchmarking process, such as an inverse relation to the average degrees of freedom of traversal per node in graphs, the overall negative impact of k-shot prompting on graph reasoning tasks, and a positive response bias which prevents LLMs from identifying the absence of a valid solution. Finally, we propose a new prompting technique specially designed f
    
[^7]: DiffiT: 用于图像生成的扩散视觉Transformer模型

    DiffiT: Diffusion Vision Transformers for Image Generation

    [https://arxiv.org/abs/2312.02139](https://arxiv.org/abs/2312.02139)

    DiffiT是一种新的模型，结合了Vision Transformer和扩散模型的优势，在图像生成中表现出色，特别是通过引入细粒度去噪控制和时间依赖的多头自注意力机制，实现了高保真图像的生成。

    

    具有强大表现力和高样本质量的扩散模型在生成领域取得了最先进的性能。开创性的视觉Transformer（ViT）展现了强大的建模能力和可扩展性，特别适用于识别任务。本文研究了ViTs在基于扩散的生成学习中的有效性，并提出了一个新模型，称为Diffusion Vision Transformers（DiffiT）。具体地，我们提出了一种用于对去噪过程进行细粒度控制的方法，并引入了时间依赖的多头自注意力（TMSA）机制。DiffiT在生成高保真图像方面非常有效，参数效率也显著提高。我们还提出了基于潜空间和图像空间的DiffiT模型，并在不同分辨率的各种类别条件和非条件综合任务上展现了最先进的性能。潜空间DiffiT模型达到

    arXiv:2312.02139v2 Announce Type: replace-cross  Abstract: Diffusion models with their powerful expressivity and high sample quality have achieved State-Of-The-Art (SOTA) performance in the generative domain. The pioneering Vision Transformer (ViT) has also demonstrated strong modeling capabilities and scalability, especially for recognition tasks. In this paper, we study the effectiveness of ViTs in diffusion-based generative learning and propose a new model denoted as Diffusion Vision Transformers (DiffiT). Specifically, we propose a methodology for finegrained control of the denoising process and introduce the Time-dependant Multihead Self Attention (TMSA) mechanism. DiffiT is surprisingly effective in generating high-fidelity images with significantly better parameter efficiency. We also propose latent and image space DiffiT models and show SOTA performance on a variety of class-conditional and unconditional synthesis tasks at different resolutions. The Latent DiffiT model achieves
    
[^8]: 系统化的人工智能方法用于AGI：解决对齐、能源和AGI大挑战

    Systematic AI Approach for AGI: Addressing Alignment, Energy, and AGI Grand Challenges. (arXiv:2310.15274v1 [cs.AI])

    [http://arxiv.org/abs/2310.15274](http://arxiv.org/abs/2310.15274)

    本论文讨论了面临能源、对齐和从狭义人工智能到AGI的三大挑战的系统化人工智能方法。现有的人工智能方法在能源消耗、系统设计和对齐问题上存在不足，而系统设计在解决对齐、能源和AGI大挑战中是至关重要的。

    

    人工智能面临着三大挑战：能源壁垒、对齐问题和从狭义人工智能到AGI的飞跃。当代人工智能解决方案在模型训练和日常运行过程中消耗着不可持续的能源。更糟糕的是，自2020年以来，每个新的人工智能模型所需的计算量每两个月就翻倍，直接导致能源消耗的增加。从人工智能到AGI的飞跃需要多个功能子系统以平衡的方式运作，这需要一个系统架构。然而，当前的人工智能方法缺乏系统设计；即使系统特征在人脑中扮演着重要角色，从它处理信息的方式到它做出决策的方式。同样，当前的对齐和人工智能伦理方法在很大程度上忽视了系统设计，然而研究表明，大脑的系统架构在健康的道德决策中起着关键作用。在本文中，我们认为系统设计在解决对齐、能源和AGI大挑战中至关重要。

    AI faces a trifecta of grand challenges the Energy Wall, the Alignment Problem and the Leap from Narrow AI to AGI. Contemporary AI solutions consume unsustainable amounts of energy during model training and daily operations.Making things worse, the amount of computation required to train each new AI model has been doubling every 2 months since 2020, directly translating to increases in energy consumption.The leap from AI to AGI requires multiple functional subsystems operating in a balanced manner, which requires a system architecture. However, the current approach to artificial intelligence lacks system design; even though system characteristics play a key role in the human brain from the way it processes information to how it makes decisions. Similarly, current alignment and AI ethics approaches largely ignore system design, yet studies show that the brains system architecture plays a critical role in healthy moral decisions.In this paper, we argue that system design is critically im
    
[^9]: 可解释的分布不变公平性度量方法对于连续评分

    Interpretable Distribution-Invariant Fairness Measures for Continuous Scores. (arXiv:2308.11375v1 [stat.ML])

    [http://arxiv.org/abs/2308.11375](http://arxiv.org/abs/2308.11375)

    对于连续评分，我们提出了一种基于Wasserstein距离的分布不变公平性度量方法，能够解释度量结果并适用于比较不同模型、数据集或时间点之间的偏差。

    

    算法公平性度量通常在二元决策的背景下进行讨论。我们将这种方法扩展到连续评分。到目前为止，基于ROC的度量方法主要用于此目的。其他现有方法主要依赖于评分的分布，不适用于排名任务，或者它们的效果大小不可解释。在这里，我们提出了一种基于Wasserstein距离的连续评分的分布不变公平性度量方法，具有合理的解释。我们的度量方法易于计算，并适用于量化和解释群体差异的强度，以及比较不同模型、数据集或时间点之间的偏差。我们建立了现有评分公平性度量方法的不同族之间的联系，并表明所提出的分布不变公平性度量方法表现更好，因为它们更明确，并且可以量化显著的偏差，而ROC-based不能。

    Measures of algorithmic fairness are usually discussed in the context of binary decisions. We extend the approach to continuous scores. So far, ROC-based measures have mainly been suggested for this purpose. Other existing methods depend heavily on the distribution of scores, are unsuitable for ranking tasks, or their effect sizes are not interpretable. Here, we propose a distributionally invariant version of fairness measures for continuous scores with a reasonable interpretation based on the Wasserstein distance. Our measures are easily computable and well suited for quantifying and interpreting the strength of group disparities as well as for comparing biases across different models, datasets, or time points. We derive a link between the different families of existing fairness measures for scores and show that the proposed distributionally invariant fairness measures outperform ROC-based fairness measures because they are more explicit and can quantify significant biases that ROC-ba
    
[^10]: 概要、亮点和行动项目：设计、实现和评估基于LLM的会议总结系统

    Summaries, Highlights, and Action items: Design, implementation and evaluation of an LLM-powered meeting recap system. (arXiv:2307.15793v1 [cs.HC])

    [http://arxiv.org/abs/2307.15793](http://arxiv.org/abs/2307.15793)

    这项研究设计、实现和评估了一种基于LLM的会议总结系统，通过减少个人会议负担和增加会议输出的清晰度和一致性，提高了会议体验。

    

    会议在工作协调中发挥着关键的基础设施作用。近年来，由于向混合和远程工作的转变，越来越多的会议正在转移到在线计算机媒体空间。这导致了新的问题（例如在更不吸引人的会议上花费更多的时间）和新的机会（例如自动转录/字幕和总结支持）。最近的大型语言模型（LLMs）在对话总结方面取得了进展，通过减少个人的会议负担和增加会议输出的清晰度和一致性，有可能提高会议体验。尽管存在这种潜力，但由于长篇转录和无法根据用户的上下文捕捉到多样的总结需求，它们面临着技术限制。为了填补这些差距，我们设计、实现并在上下文中评估了一种会议总结系统。我们首先构思了两个明显的总结表示方式——重要亮点和结构化的分级会议纪要视图。我们开发了一个系统来实现这些表示方法。

    Meetings play a critical infrastructural role in the coordination of work. In recent years, due to shift to hybrid and remote work, more meetings are moving to online Computer Mediated Spaces. This has led to new problems (e.g. more time spent in less engaging meetings) and new opportunities (e.g. automated transcription/captioning and recap support). Recent advances in large language models (LLMs) for dialog summarization have the potential to improve the experience of meetings by reducing individuals' meeting load and increasing the clarity and alignment of meeting outputs. Despite this potential, they face technological limitation due to long transcripts and inability to capture diverse recap needs based on user's context. To address these gaps, we design, implement and evaluate in-context a meeting recap system. We first conceptualize two salient recap representations -- important highlights, and a structured, hierarchical minutes view. We develop a system to operationalize the rep
    
[^11]: 自然策略梯度原始-对偶方法在约束MDP中的收敛性和样本复杂度研究

    Convergence and sample complexity of natural policy gradient primal-dual methods for constrained MDPs. (arXiv:2206.02346v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2206.02346](http://arxiv.org/abs/2206.02346)

    本文研究了约束马尔可夫决策过程中优化问题的自然策略梯度原始-对偶方法。通过自然策略梯度上升和投影次梯度下降更新变量，我们的方法在全局收敛中实现了次线性速率，而且不受状态-动作空间大小限制。

    

    我们研究了顺序决策问题，旨在最大化预期总奖励，同时满足对预期总效用的约束。我们使用自然策略梯度方法来解决约束马尔可夫决策过程（约束MDP）的折扣无限时序优化控制问题。具体地，我们提出了一种新的自然策略梯度原始-对偶（NPG-PD）方法，该方法通过自然策略梯度上升更新原始变量，通过投影次梯度下降更新对偶变量。尽管底层最大化涉及非凸目标函数和非凸约束集，但在softmax策略参数化下，我们证明了我们的方法在优化间隙和约束违规方面实现全局收敛，并具有次线性速率。此类收敛与状态-动作空间的大小无关，即无维度限制。此外，对于对数线性和一般平滑策略参数化，我们确立了收敛性和样本复杂度界限。

    We study sequential decision making problems aimed at maximizing the expected total reward while satisfying a constraint on the expected total utility. We employ the natural policy gradient method to solve the discounted infinite-horizon optimal control problem for Constrained Markov Decision Processes (constrained MDPs). Specifically, we propose a new Natural Policy Gradient Primal-Dual (NPG-PD) method that updates the primal variable via natural policy gradient ascent and the dual variable via projected sub-gradient descent. Although the underlying maximization involves a nonconcave objective function and a nonconvex constraint set, under the softmax policy parametrization we prove that our method achieves global convergence with sublinear rates regarding both the optimality gap and the constraint violation. Such convergence is independent of the size of the state-action space, i.e., it is~dimension-free. Furthermore, for log-linear and general smooth policy parametrizations, we esta
    

