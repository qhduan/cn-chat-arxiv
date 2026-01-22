# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring and Evaluating Hallucinations in LLM-Powered Code Generation](https://arxiv.org/abs/2404.00971) | 本研究通过主题分析对LLM生成的代码中的幻觉进行了总结和分类，建立了代码中幻觉的全面分类法。 |
| [^2] | [Sora as an AGI World Model? A Complete Survey on Text-to-Video Generation](https://arxiv.org/abs/2403.05131) | 对文本到视频生成技术的发展进行了详细调查, 着重介绍了从传统生成模型到尖端Sora模型的转变，强调了可扩展性和通用性的发展。 |
| [^3] | [CoTBal: Comprehensive Task Balancing for Multi-Task Visual Instruction Tuning](https://arxiv.org/abs/2403.04343) | 提出了一种全面任务平衡算法（CoTBal）用于大型多模态模型的多任务视觉指令调整，首次探索了视觉指令调整中的多任务优化。 |
| [^4] | [GSINA: Improving Subgraph Extraction for Graph Invariant Learning via Graph Sinkhorn Attention](https://arxiv.org/abs/2402.07191) | 本文提出了一种改进的图不变学习方法，通过稀疏性、软性和可微性原则来提取不变子图，从而提高图学习的泛化性能。 |
| [^5] | [Towards an AI Accountability Policy.](http://arxiv.org/abs/2307.13658) | 这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。 |
| [^6] | [Benchmark data to study the influence of pre-training on explanation performance in MR image classification.](http://arxiv.org/abs/2306.12150) | 本研究提出了一个MRI分类任务的基准数据集，用于评估不同模型的解释性能。实验结果表明，XAI方法并不一定比简单模型提供更好的解释，且CNN的解释能力取决于底层数据的复杂性和标签的质量。 |

# 详细

[^1]: 探索和评估LLM驱动的代码生成中的幻觉

    Exploring and Evaluating Hallucinations in LLM-Powered Code Generation

    [https://arxiv.org/abs/2404.00971](https://arxiv.org/abs/2404.00971)

    本研究通过主题分析对LLM生成的代码中的幻觉进行了总结和分类，建立了代码中幻觉的全面分类法。

    

    大型语言模型（LLMs）的崛起已经极大地推动了软件工程任务中许多应用的发展，特别是在代码生成方面。尽管表现出色，LLMs容易产生幻觉，即LLMs可能产生与用户意图偏离、表现出内部不一致或与事实知识不符的输出，使得在广泛应用中部署LLMs可能存在风险。现有研究主要集中在自然语言生成（NLG）领域的幻觉，缺乏对代码生成环境中幻觉类型和程度的理解。为了填补这一空白，我们对LLM生成的代码进行了主题分析，总结和归类其中存在的幻觉。我们的研究建立了LLM生成的代码中幻觉的全面分类法，涵盖了5个主要幻觉类别。

    arXiv:2404.00971v1 Announce Type: cross  Abstract: The rise of Large Language Models (LLMs) has significantly advanced many applications on software engineering tasks, particularly in code generation. Despite the promising performance, LLMs are prone to generate hallucinations, which means LLMs might produce outputs that deviate from users' intent, exhibit internal inconsistencies, or misalign with the factual knowledge, making the deployment of LLMs potentially risky in a wide range of applications. Existing work mainly focuses on investing the hallucination in the domain of natural language generation (NLG), leaving a gap in understanding the types and extent of hallucinations in the context of code generation. To bridge the gap, we conducted a thematic analysis of the LLM-generated code to summarize and categorize the hallucinations present in it. Our study established a comprehensive taxonomy of hallucinations in LLM-generated code, encompassing 5 primary categories of hallucinatio
    
[^2]: Sora作为AGI世界模型？关于文本到视频生成的完整调查

    Sora as an AGI World Model? A Complete Survey on Text-to-Video Generation

    [https://arxiv.org/abs/2403.05131](https://arxiv.org/abs/2403.05131)

    对文本到视频生成技术的发展进行了详细调查, 着重介绍了从传统生成模型到尖端Sora模型的转变，强调了可扩展性和通用性的发展。

    

    arXiv:2403.05131v1 公告类型: 新摘要: 文本到视频生成标志着生成式人工智能不断发展领域中的重要前沿，整合了文本到图像合成、视频字幕和文本引导编辑的进展。本调查对文本到视频技术的发展进行了批判性审视，重点关注传统生成模型向尖端Sora模型转变的过程，突出了可扩展性和通用性的发展。区别于以往作品的分析，我们深入探讨了这些模型的技术框架和演化路径。此外，我们还深入探讨了实际应用，并解决了伦理和技术挑战，如无法执行多实体处理、理解因果关系学习、理解物理互动、感知物体缩放和比例以及对抗物体幻觉，这也是生成模型中长期存在的问题。

    arXiv:2403.05131v1 Announce Type: new  Abstract: Text-to-video generation marks a significant frontier in the rapidly evolving domain of generative AI, integrating advancements in text-to-image synthesis, video captioning, and text-guided editing. This survey critically examines the progression of text-to-video technologies, focusing on the shift from traditional generative models to the cutting-edge Sora model, highlighting developments in scalability and generalizability. Distinguishing our analysis from prior works, we offer an in-depth exploration of the technological frameworks and evolutionary pathways of these models. Additionally, we delve into practical applications and address ethical and technological challenges such as the inability to perform multiple entity handling, comprehend causal-effect learning, understand physical interaction, perceive object scaling and proportioning, and combat object hallucination which is also a long-standing problem in generative models. Our c
    
[^3]: CoTBal: 多任务视觉指令调整的全面任务平衡

    CoTBal: Comprehensive Task Balancing for Multi-Task Visual Instruction Tuning

    [https://arxiv.org/abs/2403.04343](https://arxiv.org/abs/2403.04343)

    提出了一种全面任务平衡算法（CoTBal）用于大型多模态模型的多任务视觉指令调整，首次探索了视觉指令调整中的多任务优化。

    

    arXiv:2403.04343v1 公告类型: 新   摘要: 视觉指令调整是大型多模态模型（LMMs）的关键训练阶段。然而，无差别混合来自各种任务的指令跟随数据的普遍做法可能导致由于任务之间的指令格式和知识领域不同而导致整体性能不佳。为了缓解这个问题，我们提出了一种新颖的全面任务平衡（CoTBal）算法，用于LMMs的多任务视觉指令调整。据我们所知，这是第一项探索视觉指令调整中多任务优化的工作。具体地，我们考虑任务平衡的两个关键维度:（1）任务间贡献，即学习一个任务可能增强其他任务的性能的现象，归因于重叠的知识领域，以及（2）任务内难度，指的是单个任务内的学习难度。通过用基于性能的方法量化这两个维度

    arXiv:2403.04343v1 Announce Type: new  Abstract: Visual instruction tuning is a key training stage of large multimodal models (LMMs). Nevertheless, the common practice of indiscriminately mixing instruction-following data from various tasks may result in suboptimal overall performance due to different instruction formats and knowledge domains across tasks. To mitigate this issue, we propose a novel Comprehensive Task Balancing (CoTBal) algorithm for multi-task visual instruction tuning of LMMs. To our knowledge, this is the first work that explores multi-task optimization in visual instruction tuning. Specifically, we consider two key dimensions for task balancing: (1) Inter-Task Contribution, the phenomenon where learning one task potentially enhances the performance in other tasks, attributable to the overlapping knowledge domains, and (2) Intra-Task Difficulty, which refers to the learning difficulty within a single task. By quantifying these two dimensions with performance-based me
    
[^4]: GSINA: 通过图Sinkhorn Attention改进图不变学习中的子图提取

    GSINA: Improving Subgraph Extraction for Graph Invariant Learning via Graph Sinkhorn Attention

    [https://arxiv.org/abs/2402.07191](https://arxiv.org/abs/2402.07191)

    本文提出了一种改进的图不变学习方法，通过稀疏性、软性和可微性原则来提取不变子图，从而提高图学习的泛化性能。

    

    图不变学习(GIL)是一种有效的方法，用于在不同分布变化下发现图数据与其标签之间的不变关系，以解决各种图学习任务。最近的GIL研究主要集中在从输入图中提取不变子图，作为规则化策略来提高图学习的泛化性能。然而，这些方法在获取不变子图方面也存在各种限制。本文分析了现有工作的缺点，并提出了提取不变子图的相应原则：1）稀疏性，以过滤掉变异特征；2）软性，以获得更广泛的解空间；和3）可微性，以进行端到端优化。为了在一次操作中满足这些原则，我们利用最优传输(OT)理论，并提出了一种新颖的图注意机制，称为图Sinkhorn Attention（G)

    Graph invariant learning (GIL) has been an effective approach to discovering the invariant relationships between graph data and its labels for different graph learning tasks under various distribution shifts. Many recent endeavors of GIL focus on extracting the invariant subgraph from the input graph for prediction as a regularization strategy to improve the generalization performance of graph learning. Despite their success, such methods also have various limitations in obtaining their invariant subgraphs. In this paper, we provide in-depth analyses of the drawbacks of existing works and propose corresponding principles of our invariant subgraph extraction: 1) the sparsity, to filter out the variant features, 2) the softness, for a broader solution space, and 3) the differentiability, for a soundly end-to-end optimization. To meet these principles in one shot, we leverage the Optimal Transport (OT) theory and propose a novel graph attention mechanism called Graph Sinkhorn Attention (G
    
[^5]: 关于AI问责政策的探索

    Towards an AI Accountability Policy. (arXiv:2307.13658v1 [cs.CY])

    [http://arxiv.org/abs/2307.13658](http://arxiv.org/abs/2307.13658)

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。

    

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”作出的回应。在回答相关问题的关键句子末尾，提供了要求评论的问题编号的上标。该白皮书提出了一组相互关联的AI问责政策建议。

    This white paper is a response to the "AI Accountability Policy Request for Comments" by the National Telecommunications and Information Administration of the United States. The question numbers for which comments were requested are provided in superscripts at the end of key sentences answering the respective questions. The white paper offers a set of interconnected recommendations for an AI accountability policy.
    
[^6]: 基于预训练的影响因素研究医学图像分类解释性能的基准数据

    Benchmark data to study the influence of pre-training on explanation performance in MR image classification. (arXiv:2306.12150v1 [cs.CV])

    [http://arxiv.org/abs/2306.12150](http://arxiv.org/abs/2306.12150)

    本研究提出了一个MRI分类任务的基准数据集，用于评估不同模型的解释性能。实验结果表明，XAI方法并不一定比简单模型提供更好的解释，且CNN的解释能力取决于底层数据的复杂性和标签的质量。

    

    卷积神经网络（CNN）常常在医学预测任务中被成功地应用，通常与迁移学习相结合，在训练数据不足时能够提高性能。然而，由于CNN产生的模型高度复杂且通常不提供任何有关其预测机制的信息，这促使了“可解释性”人工智能（XAI）领域的研究。本文提出了一个基准数据集，用于在MRI分类任务中定量评估解释性能。通过这个基准数据集，我们可以了解迁移学习对解释质量的影响。实验结果表明，应用于基于迁移学习的CNN的流行XAI方法并不一定比简单模型提供更好的解释，并且CNN提供有意义解释的能力严重依赖于底层数据的复杂性和标签的质量。

    Convolutional Neural Networks (CNNs) are frequently and successfully used in medical prediction tasks. They are often used in combination with transfer learning, leading to improved performance when training data for the task are scarce. The resulting models are highly complex and typically do not provide any insight into their predictive mechanisms, motivating the field of 'explainable' artificial intelligence (XAI). However, previous studies have rarely quantitatively evaluated the 'explanation performance' of XAI methods against ground-truth data, and transfer learning and its influence on objective measures of explanation performance has not been investigated. Here, we propose a benchmark dataset that allows for quantifying explanation performance in a realistic magnetic resonance imaging (MRI) classification task. We employ this benchmark to understand the influence of transfer learning on the quality of explanations. Experimental results show that popular XAI methods applied to t
    

