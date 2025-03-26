# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistics without Interpretation: A Sober Look at Explainable Machine Learning](https://arxiv.org/abs/2402.02870) | 解释算法往往数学上复杂且难以解释，这导致解释错误。为了向前推进，解释算法需要明确其输出的解释方式，并澄清可以和不能回答的问题。这一论点基于统计学和解释之间的区别，以及可解释机器学习和应用统计学之间的相似性。 |
| [^2] | [Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey](https://arxiv.org/abs/2402.02242) | 本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。 |
| [^3] | [Masking Augmentation for Supervised Learning](https://arxiv.org/abs/2306.11339) | 提出了一种名为MaskSub的新方法，通过使用遮罩子模型和放松的损失函数来强化监督学习中的遮罩增强，提高了性能并加速训练过程。 |
| [^4] | [DistDNAS: Search Efficient Feature Interactions within 2 Hours.](http://arxiv.org/abs/2311.00231) | DistDNAS是一种在推荐系统中高效搜索特征交互的解决方案，通过分布式搜索和选择最佳交互模块，实现了巨大的加速并将搜索时间从2天缩短到2小时。 |
| [^5] | [Multi-Task Knowledge Enhancement for Zero-Shot and Multi-Domain Recommendation in an AI Assistant Application.](http://arxiv.org/abs/2306.06302) | 本文提出了一种利用多领域交互信息和外部知识图来进行新领域预测的方法，并将其应用于一个AI助手应用中，以提高推荐系统的预测准确性。 |
| [^6] | [On Diffusion Modeling for Anomaly Detection.](http://arxiv.org/abs/2305.18593) | 本文研究了扩散建模在无监督和半监督异常检测中的应用，发现去噪扩散概率模型表现很好但计算成本高，因此提出了一种替代方法——扩散时间概率模型，该模型能够通过较大的时间步长上的高后验密度识别异常，并通过深度神经网络提高效率。 |
| [^7] | [Phylo2Vec: a vector representation for binary trees.](http://arxiv.org/abs/2304.12693) | Phylo2Vec是一种新的二叉树简明表示方法，它能够轻松采样二叉树，并以系统性的方法遍历树空间。这种方法用于构建深度神经网络，能够显著提高蛋白质类别预测的性能。 |
| [^8] | [Polysemanticity and Capacity in Neural Networks.](http://arxiv.org/abs/2210.01892) | 该论文通过分析特征容量来理解神经网络中的多义性现象，发现在最优的容量分配下，神经网络倾向于单义地表示重要特征，多义地表示次重要特征，并忽略最不重要的特征。多义性现象在输入具有更高的峰度或稀疏性时更为普遍，并且在某些体系结构中比其他体系结构更为普遍。此外，作者还发现了嵌入空间中的分块半正交结构，不同模型中的分块大小不同，突出了模型体系结构的影响。 |
| [^9] | [Targeted Separation and Convergence with Kernel Discrepancies.](http://arxiv.org/abs/2209.12835) | 通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。 |

# 详细

[^1]: 没有解释的统计学：对可解释机器学习的冷静观察

    Statistics without Interpretation: A Sober Look at Explainable Machine Learning

    [https://arxiv.org/abs/2402.02870](https://arxiv.org/abs/2402.02870)

    解释算法往往数学上复杂且难以解释，这导致解释错误。为了向前推进，解释算法需要明确其输出的解释方式，并澄清可以和不能回答的问题。这一论点基于统计学和解释之间的区别，以及可解释机器学习和应用统计学之间的相似性。

    

    在关于解释算法的快速发展的文献中，这些算法往往不清楚所用于何处及其使用方式。我们认为这是因为解释算法往往在数学上复杂且难以解释。然而，没有清晰解释的复杂统计方法很可能导致解释的错误，这一事实在文献中越来越明显。为了向前推进，关于解释算法的论文应明确解释算法的输出如何解释。他们还应澄清在给出解释的情况下可以回答哪些关于函数的问题，以及哪些问题无法回答。我们的论点基于统计学和它们的解释之间的区别。它还依赖于可解释机器学习和应用统计学之间的相似之处。

    In the rapidly growing literature on explanation algorithms, it often remains unclear what precisely these algorithms are for and how they should be used. We argue that this is because explanation algorithms are often mathematically complex but don't admit a clear interpretation. Unfortunately, complex statistical methods that don't have a clear interpretation are bound to lead to errors in interpretation, a fact that has become increasingly apparent in the literature. In order to move forward, papers on explanation algorithms should make clear how precisely the output of the algorithms should be interpreted. They should also clarify what questions about the function can and cannot be answered given the explanations. Our argument is based on the distinction between statistics and their interpretation. It also relies on parallels between explainable machine learning and applied statistics.
    
[^2]: 面向预训练视觉模型的参数高效微调：一项综述

    Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey

    [https://arxiv.org/abs/2402.02242](https://arxiv.org/abs/2402.02242)

    本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。

    

    大规模预训练的视觉模型（PVMs）展示了在各种下游视觉任务中的适应能力潜力。然而，随着最先进的PVMs达到数十亿甚至数万亿个参数，标准的全面微调范式由于高计算和存储需求变得不可持续。作为响应，研究人员正在探索参数高效微调（PEFT），旨在以最小参数修改超越全面微调的性能。本综述提供了视觉PEFT的全面概述和未来方向，对最新进展进行了系统审查。首先，我们提供了PEFT的正式定义，并讨论了模型预训练方法。然后，我们将现有方法分为三类：基于添加的、基于部分的和基于统一的。最后，我们介绍了常用的数据集和应用，并提出了潜在的未来研究挑战。该综述还提供了丰富的资源收藏。

    Large-scale pre-trained vision models (PVMs) have shown great potential for adaptability across various downstream vision tasks. However, with state-of-the-art PVMs growing to billions or even trillions of parameters, the standard full fine-tuning paradigm is becoming unsustainable due to high computational and storage demands. In response, researchers are exploring parameter-efficient fine-tuning (PEFT), which seeks to exceed the performance of full fine-tuning with minimal parameter modifications. This survey provides a comprehensive overview and future directions for visual PEFT, offering a systematic review of the latest advancements. First, we provide a formal definition of PEFT and discuss model pre-training methods. We then categorize existing methods into three categories: addition-based, partial-based, and unified-based. Finally, we introduce the commonly used datasets and applications and suggest potential future research challenges. A comprehensive collection of resources is
    
[^3]: 遮罩数据增强用于监督学习

    Masking Augmentation for Supervised Learning

    [https://arxiv.org/abs/2306.11339](https://arxiv.org/abs/2306.11339)

    提出了一种名为MaskSub的新方法，通过使用遮罩子模型和放松的损失函数来强化监督学习中的遮罩增强，提高了性能并加速训练过程。

    

    使用随机遮罩进行预训练已经成为训练技术中的新趋势。然而，监督学习在采用遮罩增强方面面临挑战，主要是由于不稳定的训练。本文提出了一种涉及遮罩增强的新方法，称为Masked Sub-model (MaskSub)。MaskSub由主模型和子模型组成；前者享受传统训练方法，而后者利用强大的遮罩增强来训练。MaskSub通过缓解类似于自蒸馏损失的放松损失函数来解决挑战。我们的分析表明，MaskSub提高了性能，训练损失的收敛速度甚至比常规训练更快，这表明我们的方法有助于训练。我们进一步验证了MaskSub在各种训练方法和模型上的有效性，包括DeiT-III，MAE微调，CLIP微调，ResNet和Swin T。

    arXiv:2306.11339v2 Announce Type: replace-cross  Abstract: Pre-training using random masking has emerged as a novel trend in training techniques. However, supervised learning faces a challenge in adopting masking augmentations, primarily due to unstable training. In this paper, we propose a novel way to involve masking augmentations dubbed Masked Sub-model (MaskSub). MaskSub consists of the main-model and sub-model; while the former enjoys conventional training recipes, the latter leverages the benefit of strong masking augmentations in training. MaskSub addresses the challenge by mitigating adverse effects through a relaxed loss function similar to a self-distillation loss. Our analysis shows that MaskSub improves performance, with the training loss converging even faster than regular training, which suggests our method facilitates training. We further validate MaskSub across diverse training recipes and models, including DeiT-III, MAE fine-tuning, CLIP fine-tuning, ResNet, and Swin T
    
[^4]: DistDNAS: 在2小时内高效搜索特征交互

    DistDNAS: Search Efficient Feature Interactions within 2 Hours. (arXiv:2311.00231v1 [cs.IR])

    [http://arxiv.org/abs/2311.00231](http://arxiv.org/abs/2311.00231)

    DistDNAS是一种在推荐系统中高效搜索特征交互的解决方案，通过分布式搜索和选择最佳交互模块，实现了巨大的加速并将搜索时间从2天缩短到2小时。

    

    在推荐系统中，搜索效率和服务效率是构建特征交互和加快模型开发过程的两个主要方面。在大规模基准测试中，由于大量数据上的顺序工作流程，搜索最佳特征交互设计需要付出巨大成本。此外，融合各种来源、顺序和数学运算的交互会引入潜在的冲突和额外的冗余，导致性能和服务成本的次优权衡。本文提出了DistDNAS作为一种简洁的解决方案，可以快速且高效地进行特征交互设计。DistDNAS提出了一个超级网络，将不同顺序和类型的交互模块作为搜索空间进行整合。为了优化搜索效率，DistDNAS在不同的数据日期上分布式搜索并汇总选择最佳的交互模块，实现了超过25倍的加速，将搜索成本从2天减少到2小时。

    Search efficiency and serving efficiency are two major axes in building feature interactions and expediting the model development process in recommender systems. On large-scale benchmarks, searching for the optimal feature interaction design requires extensive cost due to the sequential workflow on the large volume of data. In addition, fusing interactions of various sources, orders, and mathematical operations introduces potential conflicts and additional redundancy toward recommender models, leading to sub-optimal trade-offs in performance and serving cost. In this paper, we present DistDNAS as a neat solution to brew swift and efficient feature interaction design. DistDNAS proposes a supernet to incorporate interaction modules of varying orders and types as a search space. To optimize search efficiency, DistDNAS distributes the search and aggregates the choice of optimal interaction modules on varying data dates, achieving over 25x speed-up and reducing search cost from 2 days to 2 
    
[^5]: 多任务知识增强在AI助手应用中的零样本和多领域推荐中的应用

    Multi-Task Knowledge Enhancement for Zero-Shot and Multi-Domain Recommendation in an AI Assistant Application. (arXiv:2306.06302v1 [cs.IR])

    [http://arxiv.org/abs/2306.06302](http://arxiv.org/abs/2306.06302)

    本文提出了一种利用多领域交互信息和外部知识图来进行新领域预测的方法，并将其应用于一个AI助手应用中，以提高推荐系统的预测准确性。

    

    推荐系统在商业上取得了巨大的成功，但仍然难以将新用户整合进去。由于用户经常在不同领域与内容进行交互，因此可以利用用户在之前的领域中的交互来改善其在新领域中的推荐（多领域推荐）。知识图增强的单一领域推荐（知识图增强）的研究线程独立于此使用外部知识图来提高推荐系统的预测准确性。我们在这项工作中提出将这些方法统一起来：利用其他领域中的交互信息以及外部知识图来进行新领域的推荐。我们将这些想法应用于一个从数百万用户请求的视频、音乐和书籍的数据集中，该数据集用于一个AI助手应用中。

    Recommender systems have found significant commercial success but still struggle with integrating new users. Since users often interact with content in different domains, it is possible to leverage a user's interactions in previous domains to improve that user's recommendations in a new one (multi-domain recommendation). A separate research thread on knowledge graph enhancement uses external knowledge graphs to improve single domain recommendations (knowledge graph enhancement). Both research threads incorporate related information to improve predictions in a new domain. We propose in this work to unify these approaches: Using information from interactions in other domains as well as external knowledge graphs to make predictions in a new domain that would be impossible with either information source alone. We apply these ideas to a dataset derived from millions of users' requests for content across three domains (videos, music, and books) in a live virtual assistant application. We dem
    
[^6]: 关于扩散建模在异常检测中的应用

    On Diffusion Modeling for Anomaly Detection. (arXiv:2305.18593v1 [cs.LG])

    [http://arxiv.org/abs/2305.18593](http://arxiv.org/abs/2305.18593)

    本文研究了扩散建模在无监督和半监督异常检测中的应用，发现去噪扩散概率模型表现很好但计算成本高，因此提出了一种替代方法——扩散时间概率模型，该模型能够通过较大的时间步长上的高后验密度识别异常，并通过深度神经网络提高效率。

    

    扩散模型以其在生成建模中的优异性能而闻名，成为基于密度的异常检测的有吸引力的候选算法。本文研究了各种扩散建模方法在无监督和半监督异常检测中的应用。尤其是发现去噪扩散概率模型（DDPM）在异常检测方面具备很好的表现，但计算成本较高。通过简化DDPM在异常检测中的应用，我们自然地引出另一种称为扩散时间概率模型（DTPM）的替代方法。DTPM估计给定输入的扩散时间的后验分布，能够通过较大的时间步长上的高后验密度识别异常。我们导出了此后验分布的解析形式，并利用深度神经网络提高推理效率。通过在ADBenh基准测试中的实证评估，我们证明了所有基于扩散的异常检测方法的实用性。

    Known for their impressive performance in generative modeling, diffusion models are attractive candidates for density-based anomaly detection. This paper investigates different variations of diffusion modeling for unsupervised and semi-supervised anomaly detection. In particular, we find that Denoising Diffusion Probability Models (DDPM) are performant on anomaly detection benchmarks yet computationally expensive. By simplifying DDPM in application to anomaly detection, we are naturally led to an alternative approach called Diffusion Time Probabilistic Model (DTPM). DTPM estimates the posterior distribution over diffusion time for a given input, enabling the identification of anomalies due to their higher posterior density at larger timesteps. We derive an analytical form for this posterior density and leverage a deep neural network to improve inference efficiency. Through empirical evaluations on the ADBench benchmark, we demonstrate that all diffusion-based anomaly detection methods 
    
[^7]: Phylo2Vec: 一种二叉树的向量表示方法

    Phylo2Vec: a vector representation for binary trees. (arXiv:2304.12693v1 [q-bio.PE])

    [http://arxiv.org/abs/2304.12693](http://arxiv.org/abs/2304.12693)

    Phylo2Vec是一种新的二叉树简明表示方法，它能够轻松采样二叉树，并以系统性的方法遍历树空间。这种方法用于构建深度神经网络，能够显著提高蛋白质类别预测的性能。

    

    从生物数据推断得到的二叉进化树对于理解生物之间共享的进化历史至关重要。根据最大似然等某个最优性准则推断出树中潜在节点的位置是NP-hard问题，这推动了大量启发式方法的发展。然而，这些启发式方法通常缺乏一种系统性的方法来均匀采样随机树或有效地探索指数级增长的树空间，这对于机器学习等优化问题至关重要。因此，我们提出了Phylo2Vec，这是一种新的简明表示方法来表示进化树。Phylo2Vec将任何具有n个叶子的二叉树映射到长度为n的整数向量。我们证明了Phylo2Vec在空间中既是良定的又是双射的。Phylo2Vec的优点是：i）轻松均匀采样二叉树；ii）以非常大或小的步长系统地遍历树空间。作为概念验证，我们使用Phylo2Vec构建了一个深度神经网络，以从氨基酸序列预测蛋白质类别。我们证明了Phylo2Vec显著提高了网络的性能，超过了之前的最优结果。

    Binary phylogenetic trees inferred from biological data are central to understanding the shared evolutionary history of organisms. Inferring the placement of latent nodes in a tree by any optimality criterion (e.g., maximum likelihood) is an NP-hard problem, propelling the development of myriad heuristic approaches. Yet, these heuristics often lack a systematic means of uniformly sampling random trees or effectively exploring a tree space that grows factorially, which are crucial to optimisation problems such as machine learning. Accordingly, we present Phylo2Vec, a new parsimonious representation of a phylogenetic tree. Phylo2Vec maps any binary tree with $n$ leaves to an integer vector of length $n$. We prove that Phylo2Vec is both well-defined and bijective to the space of phylogenetic trees. The advantages of Phylo2Vec are twofold: i) easy uniform sampling of binary trees and ii) systematic ability to traverse tree space in very large or small jumps. As a proof of concept, we use P
    
[^8]: 神经网络中的多义性和容量

    Polysemanticity and Capacity in Neural Networks. (arXiv:2210.01892v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2210.01892](http://arxiv.org/abs/2210.01892)

    该论文通过分析特征容量来理解神经网络中的多义性现象，发现在最优的容量分配下，神经网络倾向于单义地表示重要特征，多义地表示次重要特征，并忽略最不重要的特征。多义性现象在输入具有更高的峰度或稀疏性时更为普遍，并且在某些体系结构中比其他体系结构更为普遍。此外，作者还发现了嵌入空间中的分块半正交结构，不同模型中的分块大小不同，突出了模型体系结构的影响。

    

    神经网络中的单个神经元通常代表无关特征的混合。这种现象称为多义性，使解释神经网络变得更加困难，因此我们的目标是理解其原因。我们提出通过特征容量的视角来理解这一现象，特征容量是每个特征在嵌入空间中占用的分形维度。我们展示了在一个玩具模型中，最优的容量分配倾向于单义地表示最重要的特征，多义地表示次重要特征（与其对损失的影响成比例），并完全忽略最不重要的特征。当输入具有更高的峰度或稀疏性时，多义性更为普遍，并且在某些体系结构中比其他体系结构更为普遍。在得到最优容量分配后，我们进一步研究了嵌入空间的几何结构。我们发现了一个分块半正交的结构，不同模型中的分块大小不同，突出了模型体系结构的影响。

    Individual neurons in neural networks often represent a mixture of unrelated features. This phenomenon, called polysemanticity, can make interpreting neural networks more difficult and so we aim to understand its causes. We propose doing so through the lens of feature \emph{capacity}, which is the fractional dimension each feature consumes in the embedding space. We show that in a toy model the optimal capacity allocation tends to monosemantically represent the most important features, polysemantically represent less important features (in proportion to their impact on the loss), and entirely ignore the least important features. Polysemanticity is more prevalent when the inputs have higher kurtosis or sparsity and more prevalent in some architectures than others. Given an optimal allocation of capacity, we go on to study the geometry of the embedding space. We find a block-semi-orthogonal structure, with differing block sizes in different models, highlighting the impact of model archit
    
[^9]: 通过核差异实现有针对性的分离与收敛

    Targeted Separation and Convergence with Kernel Discrepancies. (arXiv:2209.12835v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.12835](http://arxiv.org/abs/2209.12835)

    通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。

    

    最大均值差异（MMDs）如核Stein差异（KSD）已经成为广泛应用的中心，包括假设检验、采样器选择、分布近似和变分推断。在每个设置中，这些基于核的差异度量需要实现（i）将目标P与其他概率测度分离，甚至（ii）控制对P的弱收敛。在本文中，我们推导了确保（i）和（ii）的新的充分必要条件。对于可分的度量空间上的MMDs，我们描述了分离Bochner可嵌入测度的核，并引入简单的条件来分离所有具有无界核的测度和用有界核来控制收敛。我们利用这些结果在$\mathbb{R}^d$上大大扩展了KSD分离和收敛控制的已知条件，并开发了首个能够精确度量对P的弱收敛的KSDs。在这个过程中，我们强调了我们的结果的影响。

    Maximum mean discrepancies (MMDs) like the kernel Stein discrepancy (KSD) have grown central to a wide range of applications, including hypothesis testing, sampler selection, distribution approximation, and variational inference. In each setting, these kernel-based discrepancy measures are required to (i) separate a target P from other probability measures or even (ii) control weak convergence to P. In this article we derive new sufficient and necessary conditions to ensure (i) and (ii). For MMDs on separable metric spaces, we characterize those kernels that separate Bochner embeddable measures and introduce simple conditions for separating all measures with unbounded kernels and for controlling convergence with bounded kernels. We use these results on $\mathbb{R}^d$ to substantially broaden the known conditions for KSD separation and convergence control and to develop the first KSDs known to exactly metrize weak convergence to P. Along the way, we highlight the implications of our res
    

