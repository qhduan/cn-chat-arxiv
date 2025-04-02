# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Three-Phases SFT Hybrid Model Integrated Strong Prior Module and Data Overlap Estimation in the Eduation Context](https://arxiv.org/abs/2403.15426) | 提出了一种在教育领域中应用的三阶段监督微调模型，通过先验和数据重叠估计实现了教育知识的结构拆卸和增量引导输出。 |
| [^2] | [A Clustering Method with Graph Maximum Decoding Information](https://arxiv.org/abs/2403.13846) | CMDI聚类方法创新性地将二维结构信息理论融入聚类过程中，弥补了基于图的模型聚类方法中忽略的随机游走访问节点和数据中嵌入的结构信息的不确定性。 |
| [^3] | [Large Language Models are In-Context Molecule Learners](https://arxiv.org/abs/2403.04197) | 提出了上下文分子适应（ICMA）范式，允许LLMs通过上下文示例学习分子-文本对齐，解决了在分子-标题翻译任务中对LLMs的挑战。 |
| [^4] | [Explainable Bayesian Optimization.](http://arxiv.org/abs/2401.13334) | 本论文介绍了一种可解释性贝叶斯优化的方法，通过TNTRules生成高质量的解释，填补了贝叶斯优化和可解释人工智能之间的间隙。 |
| [^5] | [An Optimistic-Robust Approach for Dynamic Positioning of Omnichannel Inventories.](http://arxiv.org/abs/2310.12183) | 这篇论文介绍了一种乐观-鲁棒的全渠道库存动态定位方法，通过兼顾库存弹性和改善平均性能，来平衡店铺失去销售和跨渠道电子商务履约成本的权衡。 |
| [^6] | [Towards Enhanced Controllability of Diffusion Models.](http://arxiv.org/abs/2302.14368) | 本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。 |

# 详细

[^1]: 教育环境下集成强先验模块和数据重叠估计的三阶段SFT混合模型

    A Three-Phases SFT Hybrid Model Integrated Strong Prior Module and Data Overlap Estimation in the Eduation Context

    [https://arxiv.org/abs/2403.15426](https://arxiv.org/abs/2403.15426)

    提出了一种在教育领域中应用的三阶段监督微调模型，通过先验和数据重叠估计实现了教育知识的结构拆卸和增量引导输出。

    

    在本文中，我们提出了一种端到端基于先验的三阶段监督微调模型，证明比传统微调方法更有竞争力。具体而言，我们的模型实现了教育知识的结构拆卸和增量引导输出。为此，我们通过采样器和重叠估计神经网络对三种类型的数据进行了健壮的分类，将预处理数据集分三批注入预训练模型进行LORA微调。然后，我们设计了一个先验模块，将系统提示、向量数据库和抽象语法树任务分割相结合。最后，对基于先验的微调模型应用了压缩方法和正则化约束，随后在输出端进行文本过滤以获得增量引导结果。我们的模型代表了真正以丰富的教育知识、分步指导的特点体现导师角色的第一项研究努力。

    arXiv:2403.15426v1 Announce Type: cross  Abstract: In this paper, we propose an end-to-end prior-based three-phases supervised fine-tuned model, which is proved more competitive than traditional fine-tuning method. More specifically, our model realizes the structural disassembly and incremental guided output of educational knowledge. To this end, we robustify data classification of three types via a sampler and overlap estimation neural network, and inject the preprocessing datasets into pre-trained model in three batches for LORA fine-tuning. Then, we design a prior module couples system prompt, vector databases, and abstract syntax tree task segmentation. Finally, the compression method and regularization constraint are applied to the prior-based fine-tuned model, followed by text filter at the output end to obtain incremental guided results. Our model represents the first research effort to truly embody the tutor role with the features of abundant educational knowledge, step-by-step
    
[^2]: 一种具有图最大解码信息的聚类方法

    A Clustering Method with Graph Maximum Decoding Information

    [https://arxiv.org/abs/2403.13846](https://arxiv.org/abs/2403.13846)

    CMDI聚类方法创新性地将二维结构信息理论融入聚类过程中，弥补了基于图的模型聚类方法中忽略的随机游走访问节点和数据中嵌入的结构信息的不确定性。

    

    基于图模型的聚类方法因其在各种知识领域中的广泛适用性而备受关注。其能够与其他相关应用无缝集成的适应性赋予了基于图模型的聚类分析能力，可以强大地从数据集中提取“自然关联”或“图结构”，有助于建模数据点之间的关系。尽管这种方法效果显著，但当前利用基于图的模型的聚类方法忽略了节点之间随机游走访问以及数据中嵌入的结构信息所带来的不确定性。为填补这一空白，我们提出了一种新颖的基于图的模型内最大化解码信息的聚类方法，命名为CMDI。CMDI创新地将二维结构信息理论纳入到聚类过程中，包括两个阶段：图结构提取和图顶点

    arXiv:2403.13846v1 Announce Type: cross  Abstract: The clustering method based on graph models has garnered increased attention for its widespread applicability across various knowledge domains. Its adaptability to integrate seamlessly with other relevant applications endows the graph model-based clustering analysis with the ability to robustly extract "natural associations" or "graph structures" within datasets, facilitating the modelling of relationships between data points. Despite its efficacy, the current clustering method utilizing the graph-based model overlooks the uncertainty associated with random walk access between nodes and the embedded structural information in the data. To address this gap, we present a novel Clustering method for Maximizing Decoding Information within graph-based models, named CMDI. CMDI innovatively incorporates two-dimensional structural information theory into the clustering process, consisting of two phases: graph structure extraction and graph vert
    
[^3]: 大规模语言模型是上下文分子学习器

    Large Language Models are In-Context Molecule Learners

    [https://arxiv.org/abs/2403.04197](https://arxiv.org/abs/2403.04197)

    提出了上下文分子适应（ICMA）范式，允许LLMs通过上下文示例学习分子-文本对齐，解决了在分子-标题翻译任务中对LLMs的挑战。

    

    大型语言模型（LLMs）在生物化学任务中表现出色，尤其是分子标题翻译任务，旨在弥合分子和自然语言文本之间的差距。然而，先前在适应LLMs到分子-标题翻译任务中的方法需要额外的领域特定预训练阶段，存在分子和文本空间之间的弱对齐，或对LLMs的规模有严格要求。为了解决这些挑战，我们提出了上下文分子适应（ICMA），作为一种新的范例，允许LLMs通过上下文示例学习分子-文本对齐，通过上下文分子调整。具体而言，ICMA包括以下三个阶段：跨模态检索、检索后排序和上下文分子调整。

    arXiv:2403.04197v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated exceptional performance in biochemical tasks, especially the molecule caption translation task, which aims to bridge the gap between molecules and natural language texts. However, previous methods in adapting LLMs to the molecule-caption translation task required extra domain-specific pre-training stages, suffered weak alignment between molecular and textual spaces, or imposed stringent demands on the scale of LLMs. To resolve the challenges, we propose In-Context Molecule Adaptation (ICMA), as a new paradigm allowing LLMs to learn the molecule-text alignment from context examples via In-Context Molecule Tuning. Specifically, ICMA incorporates the following three stages: Cross-modal Retrieval, Post-retrieval Re-ranking, and In-context Molecule Tuning. Initially, Cross-modal Retrieval utilizes BM25 Caption Retrieval and Molecule Graph Retrieval to retrieve informative context examples. Addi
    
[^4]: 可解释性贝叶斯优化

    Explainable Bayesian Optimization. (arXiv:2401.13334v1 [cs.LG])

    [http://arxiv.org/abs/2401.13334](http://arxiv.org/abs/2401.13334)

    本论文介绍了一种可解释性贝叶斯优化的方法，通过TNTRules生成高质量的解释，填补了贝叶斯优化和可解释人工智能之间的间隙。

    

    在工业领域，贝叶斯优化（BO）被广泛应用于人工智能协作参数调优的控制系统中。然而，由于近似误差和简化目标，BO的解决方案可能偏离人类专家的真实目标，需要后续调整。BO的黑盒特性限制了协作调优过程，因为专家不信任BO的建议。目前的可解释人工智能（XAI）方法不适用于优化问题，因此无法解决此间隙。为了填补这一间隙，我们提出了TNTRules（TUNE-NOTUNE规则），一种事后基于规则的可解释性方法，通过多目标优化生成高质量的解释。我们对基准优化问题和实际超参数优化任务的评估表明，TNTRules在生成高质量解释方面优于最先进的XAI方法。这项工作对BO和XAI的交叉领域做出了贡献，提供了可解释的优化方法。

    In industry, Bayesian optimization (BO) is widely applied in the human-AI collaborative parameter tuning of cyber-physical systems. However, BO's solutions may deviate from human experts' actual goal due to approximation errors and simplified objectives, requiring subsequent tuning. The black-box nature of BO limits the collaborative tuning process because the expert does not trust the BO recommendations. Current explainable AI (XAI) methods are not tailored for optimization and thus fall short of addressing this gap. To bridge this gap, we propose TNTRules (TUNE-NOTUNE Rules), a post-hoc, rule-based explainability method that produces high quality explanations through multiobjective optimization. Our evaluation of benchmark optimization problems and real-world hyperparameter optimization tasks demonstrates TNTRules' superiority over state-of-the-art XAI methods in generating high quality explanations. This work contributes to the intersection of BO and XAI, providing interpretable opt
    
[^5]: 一种乐观-鲁棒的全渠道库存动态定位方法

    An Optimistic-Robust Approach for Dynamic Positioning of Omnichannel Inventories. (arXiv:2310.12183v1 [math.OC])

    [http://arxiv.org/abs/2310.12183](http://arxiv.org/abs/2310.12183)

    这篇论文介绍了一种乐观-鲁棒的全渠道库存动态定位方法，通过兼顾库存弹性和改善平均性能，来平衡店铺失去销售和跨渠道电子商务履约成本的权衡。

    

    我们介绍了一种新的数据驱动和免分布乐观-鲁棒双模式库存优化策略，以有效分配销售链上的库存，以满足时变的、不确定的全渠道需求。传统的鲁棒优化方法更加注重最坏情况下的对抗性需求，而双模式策略不仅考虑了保持像鲁棒优化一样的弹性，还通过克服内生奇异值的存在而获得了改善平均情况下性能的回报。这种双模式策略在平衡店铺失去销售和跨渠道电子商务履约成本的权衡方面特别有价值，这也是我们库存优化模型的核心所在。由于渠道的异质行为，这些因素是非对称的，前者在失销售成本方面存在偏差，而后者则依赖于网络效应。

    We introduce a new class of data-driven and distribution-free optimistic-robust bimodal inventory optimization (BIO) strategy to effectively allocate inventory across a retail chain to meet time-varying, uncertain omnichannel demand. While prior Robust optimization (RO) methods emphasize the downside, i.e., worst-case adversarial demand, BIO also considers the upside to remain resilient like RO while also reaping the rewards of improved average-case performance by overcoming the presence of endogenous outliers. This bimodal strategy is particularly valuable for balancing the tradeoff between lost sales at the store and the costs of cross-channel e-commerce fulfillment, which is at the core of our inventory optimization model. These factors are asymmetric due to the heterogenous behavior of the channels, with a bias towards the former in terms of lost-sales cost and a dependence on network effects for the latter. We provide structural insights about the BIO solution and how it can be tu
    
[^6]: 实现扩展扩展扩散模型的可控性

    Towards Enhanced Controllability of Diffusion Models. (arXiv:2302.14368v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.14368](http://arxiv.org/abs/2302.14368)

    本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。

    

    去噪扩散模型在生成逼真、高质量和多样化图像方面表现出卓越能力。然而，在生成过程中的可控程度尚未得到充分探讨。受基于GAN潜在空间的图像操纵技术启发，我们训练了一个条件于两个潜在编码、一个空间内容掩码和一个扁平的样式嵌入的扩散模型。我们依赖于扩散模型渐进去噪过程的感性偏置，在空间结构掩码中编码姿势/布局信息，在样式代码中编码语义/样式信息。我们提出了两种通用的采样技术来改善可控性。我们扩展了可组合的扩散模型，允许部分依赖于条件输入，以提高生成质量，同时还提供对每个潜在代码和它们的联合分布量的控制。我们还提出了时间步相关的内容和样式潜在权重调度，进一步提高了控制性。

    Denoising Diffusion models have shown remarkable capabilities in generating realistic, high-quality and diverse images. However, the extent of controllability during generation is underexplored. Inspired by techniques based on GAN latent space for image manipulation, we train a diffusion model conditioned on two latent codes, a spatial content mask and a flattened style embedding. We rely on the inductive bias of the progressive denoising process of diffusion models to encode pose/layout information in the spatial structure mask and semantic/style information in the style code. We propose two generic sampling techniques for improving controllability. We extend composable diffusion models to allow for some dependence between conditional inputs, to improve the quality of generations while also providing control over the amount of guidance from each latent code and their joint distribution. We also propose timestep dependent weight scheduling for content and style latents to further impro
    

