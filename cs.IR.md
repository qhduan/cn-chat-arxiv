# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [One Backpropagation in Two Tower Recommendation Models](https://arxiv.org/abs/2403.18227) | 该论文提出了一种在两塔推荐模型中使用单次反向传播更新策略的方法，挑战了现有算法中平等对待用户和物品的假设。 |
| [^2] | [IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus](https://arxiv.org/abs/2402.14710) | 发布了IEPile，一个包含约0.32B个标记的综合双语IE指令语料库，通过收集和清理33个现有IE数据集并引入基于模式的指令生成，可以提高大型语言模型在信息抽取领域的性能，尤其是零样本泛化。 |
| [^3] | [Unified Hallucination Detection for Multimodal Large Language Models](https://arxiv.org/abs/2402.03190) | 该论文提出了一个新颖的统一的多模态幻觉检测框架UNIHD，并设计了一个评估基准方法MHaluBench来评估幻觉检测方法的进展。这项工作扩展了幻觉检测的研究范围并提供了有效的解决方案。 |
| [^4] | [AI-Generated Images Introduce Invisible Relevance Bias to Text-Image Retrieval.](http://arxiv.org/abs/2311.14084) | 本研究通过构建合适的基准和进行大量实验发现，由AI生成的图像对于文本-图像检索模型引入了一个不可见的相关偏差，即使这些生成的图像在视觉上与查询的相关特征相比并没有更多。这种不可见的相关偏差普遍存在于不同训练数据和架构的检索模型中。 |
| [^5] | [Unveiling the Siren's Song: Towards Reliable Fact-Conflicting Hallucination Detection.](http://arxiv.org/abs/2310.12086) | 该论文介绍了一种为大型语言模型设计的FactCHD事实冲突幻觉检测基准，用于评估LLMs生成文本的事实性。基准包含了多种事实模式，并使用基于事实的证据链进行组合性幻觉的检测。 |
| [^6] | [Challenging the Myth of Graph Collaborative Filtering: a Reasoned and Reproducibility-driven Analysis.](http://arxiv.org/abs/2308.00404) | 本文研究挑战图协同过滤的神话，通过关注结果的可复制性，成功复制了六个流行的图推荐模型在几个常见和新数据集上的结果，并与传统协同过滤模型进行比较。 |
| [^7] | [Interpolating Item and User Fairness in Recommendation Systems.](http://arxiv.org/abs/2306.10050) | 本文研究在推荐系统中平衡项目和用户公平性的框架，并通过低后悔的在线优化算法实现了维持收益同时实现公平推荐的目标。 |

# 详细

[^1]: 两塔推荐模型中的单次反向传播

    One Backpropagation in Two Tower Recommendation Models

    [https://arxiv.org/abs/2403.18227](https://arxiv.org/abs/2403.18227)

    该论文提出了一种在两塔推荐模型中使用单次反向传播更新策略的方法，挑战了现有算法中平等对待用户和物品的假设。

    

    最近几年，已经看到为了减轻信息过载而开发两塔推荐模型的广泛研究。这种模型中可以识别出四个构建模块，分别是用户-物品编码、负采样、损失计算和反向传播更新。据我们所知，现有算法仅研究了前三个模块，却忽略了反向传播模块。他们都采用某种形式的双反向传播策略，基于一个隐含的假设，即在训练阶段平等对待用户和物品。在本文中，我们挑战了这种平等训练假设，并提出了一种新颖的单次反向传播更新策略，这种策略保留了物品编码塔的正常梯度反向传播，但削减了用户编码塔的反向传播。相反，我们提出了一种移动聚合更新策略来更新每个训练周期中的用户编码。

    arXiv:2403.18227v1 Announce Type: new  Abstract: Recent years have witnessed extensive researches on developing two tower recommendation models for relieving information overload. Four building modules can be identified in such models, namely, user-item encoding, negative sampling, loss computing and back-propagation updating. To the best of our knowledge, existing algorithms have researched only on the first three modules, yet neglecting the backpropagation module. They all adopt a kind of two backpropagation strategy, which are based on an implicit assumption of equally treating users and items in the training phase. In this paper, we challenge such an equal training assumption and propose a novel one backpropagation updating strategy, which keeps the normal gradient backpropagation for the item encoding tower, but cuts off the backpropagation for the user encoding tower. Instead, we propose a moving-aggregation updating strategy to update a user encoding in each training epoch. Exce
    
[^2]: IEPile: 挖掘大规模基于模式的信息抽取语料库

    IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus

    [https://arxiv.org/abs/2402.14710](https://arxiv.org/abs/2402.14710)

    发布了IEPile，一个包含约0.32B个标记的综合双语IE指令语料库，通过收集和清理33个现有IE数据集并引入基于模式的指令生成，可以提高大型语言模型在信息抽取领域的性能，尤其是零样本泛化。

    

    大型语言模型（LLMs）在各个领域展现出了显著的潜力；然而，在信息抽取（IE）方面表现出了显著的性能差距。高质量的指令数据是提升LLMs特定能力的关键，而当前的IE数据集往往规模较小、分散且缺乏标准化的模式。因此，我们介绍了IEPile，一个综合的双语（英文和中文）IE指令语料库，包含约0.32B个标记。我们通过收集和清理33个现有IE数据集构建IEPile，并引入基于模式的指令生成来挖掘大规模语料库。在LLaMA和Baichuan上的实验结果表明，使用IEPile可以提高LLMs在IE方面的性能，尤其是零样本泛化。我们开源了资源和预训练模型，希望为自然语言处理社区提供有价值的支持。

    arXiv:2402.14710v1 Announce Type: cross  Abstract: Large Language Models (LLMs) demonstrate remarkable potential across various domains; however, they exhibit a significant performance gap in Information Extraction (IE). Note that high-quality instruction data is the vital key for enhancing the specific capabilities of LLMs, while current IE datasets tend to be small in scale, fragmented, and lack standardized schema. To this end, we introduce IEPile, a comprehensive bilingual (English and Chinese) IE instruction corpus, which contains approximately 0.32B tokens. We construct IEPile by collecting and cleaning 33 existing IE datasets, and introduce schema-based instruction generation to unearth a large-scale corpus. Experimental results on LLaMA and Baichuan demonstrate that using IEPile can enhance the performance of LLMs for IE, especially the zero-shot generalization. We open-source the resource and pre-trained models, hoping to provide valuable support to the NLP community.
    
[^3]: 统一的多模态大型语言模型的幻觉检测

    Unified Hallucination Detection for Multimodal Large Language Models

    [https://arxiv.org/abs/2402.03190](https://arxiv.org/abs/2402.03190)

    该论文提出了一个新颖的统一的多模态幻觉检测框架UNIHD，并设计了一个评估基准方法MHaluBench来评估幻觉检测方法的进展。这项工作扩展了幻觉检测的研究范围并提供了有效的解决方案。

    

    尽管在多模态任务方面取得了重大进展，多模态大型语言模型(MLLMs)仍然存在幻觉的严重问题。因此，可靠地检测MLLMs中的幻觉已成为模型评估和实际应用部署保障的重要方面。之前在这个领域的研究受到了狭窄的任务焦点、不足的幻觉类别涵盖范围以及缺乏详细的细粒度的限制。针对这些挑战，我们的工作扩展了幻觉检测的研究范围。我们提出了一个新颖的元评估基准方法，MHaluBench，精心设计以促进幻觉检测方法的进展评估。此外，我们揭示了一个新颖的统一多模态幻觉检测框架，UNIHD，它利用一套辅助工具来稳健地验证幻觉的发生。我们通过实验证明了UNIHD的有效性。

    Despite significant strides in multimodal tasks, Multimodal Large Language Models (MLLMs) are plagued by the critical issue of hallucination. The reliable detection of such hallucinations in MLLMs has, therefore, become a vital aspect of model evaluation and the safeguarding of practical application deployment. Prior research in this domain has been constrained by a narrow focus on singular tasks, an inadequate range of hallucination categories addressed, and a lack of detailed granularity. In response to these challenges, our work expands the investigative horizons of hallucination detection. We present a novel meta-evaluation benchmark, MHaluBench, meticulously crafted to facilitate the evaluation of advancements in hallucination detection methods. Additionally, we unveil a novel unified multimodal hallucination detection framework, UNIHD, which leverages a suite of auxiliary tools to validate the occurrence of hallucinations robustly. We demonstrate the effectiveness of UNIHD throug
    
[^4]: 由AI生成的图像对文本-图像检索引入了不可见的相关偏差

    AI-Generated Images Introduce Invisible Relevance Bias to Text-Image Retrieval. (arXiv:2311.14084v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2311.14084](http://arxiv.org/abs/2311.14084)

    本研究通过构建合适的基准和进行大量实验发现，由AI生成的图像对于文本-图像检索模型引入了一个不可见的相关偏差，即使这些生成的图像在视觉上与查询的相关特征相比并没有更多。这种不可见的相关偏差普遍存在于不同训练数据和架构的检索模型中。

    

    随着生成模型的进步，由人工智能生成的内容（AIGC）变得更加逼真，涌入互联网。最近的一项研究表明，这种现象导致了网络搜索中的源偏差。特别是，神经检索模型往往将生成的文本排名高于人工编写的文本。本文将这种偏差的研究扩展到跨模态检索。首先，我们成功构建了一个适合探索偏差存在的基准。随后，在这个基准上进行了大量实验，发现AI生成的图像对于文本-图像检索模型引入了一个不可见的相关偏差。具体来说，我们的实验表明，尽管AI生成的图像与查询相比没有更多的视觉相关特征，但文本-图像检索模型往往将AI生成的图像排名高于真实图像。这种不可见的相关偏差在不同训练数据和架构的检索模型中普遍存在。

    With the advancement of generation models, AI-generated content (AIGC) is becoming more realistic, flooding the Internet. A recent study suggests that this phenomenon causes source bias in text retrieval for web search. Specifically, neural retrieval models tend to rank generated texts higher than human-written texts. In this paper, we extend the study of this bias to cross-modal retrieval. Firstly, we successfully construct a suitable benchmark to explore the existence of the bias. Subsequent extensive experiments on this benchmark reveal that AI-generated images introduce an invisible relevance bias to text-image retrieval models. Specifically, our experiments show that text-image retrieval models tend to rank the AI-generated images higher than the real images, even though the AI-generated images do not exhibit more visually relevant features to the query than real images. This invisible relevance bias is prevalent across retrieval models with varying training data and architectures
    
[^5]: 发现塞壬之歌：可靠的事实冲突幻觉检测

    Unveiling the Siren's Song: Towards Reliable Fact-Conflicting Hallucination Detection. (arXiv:2310.12086v1 [cs.CL])

    [http://arxiv.org/abs/2310.12086](http://arxiv.org/abs/2310.12086)

    该论文介绍了一种为大型语言模型设计的FactCHD事实冲突幻觉检测基准，用于评估LLMs生成文本的事实性。基准包含了多种事实模式，并使用基于事实的证据链进行组合性幻觉的检测。

    

    大型语言模型（LLMs），如ChatGPT/GPT-4，因其广泛的实际应用而受到广泛关注，但其在网络平台上存在事实冲突幻觉的问题限制了其采用。对由LLMs产生的文本的事实性评估仍然未被充分探索，不仅涉及对基本事实的判断，还包括对复杂推理任务（如多跳等）中出现的事实错误的评估。为此，我们引入了FactCHD，一种为LLMs精心设计的事实冲突幻觉检测基准。作为在“查询-响应”上下文中评估事实性的关键工具，我们的基准采用了大规模数据集，涵盖了广泛的事实模式，如基本事实，多跳，比较和集合操作模式。我们基准的一个独特特点是其包含基于事实的证据链，从而便于进行组合性幻觉的检测。

    Large Language Models (LLMs), such as ChatGPT/GPT-4, have garnered widespread attention owing to their myriad of practical applications, yet their adoption has been constrained by issues of fact-conflicting hallucinations across web platforms. The assessment of factuality in text, produced by LLMs, remains inadequately explored, extending not only to the judgment of vanilla facts but also encompassing the evaluation of factual errors emerging in complex inferential tasks like multi-hop, and etc. In response, we introduce FactCHD, a fact-conflicting hallucination detection benchmark meticulously designed for LLMs. Functioning as a pivotal tool in evaluating factuality within "Query-Respons" contexts, our benchmark assimilates a large-scale dataset, encapsulating a broad spectrum of factuality patterns, such as vanilla, multi-hops, comparison, and set-operation patterns. A distinctive feature of our benchmark is its incorporation of fact-based chains of evidence, thereby facilitating com
    
[^6]: 挑战图协同过滤的神话：一项基于推理和可复制性的分析

    Challenging the Myth of Graph Collaborative Filtering: a Reasoned and Reproducibility-driven Analysis. (arXiv:2308.00404v1 [cs.IR])

    [http://arxiv.org/abs/2308.00404](http://arxiv.org/abs/2308.00404)

    本文研究挑战图协同过滤的神话，通过关注结果的可复制性，成功复制了六个流行的图推荐模型在几个常见和新数据集上的结果，并与传统协同过滤模型进行比较。

    

    图神经网络模型（GNNs）的成功显著推动了推荐系统的发展，通过将用户和物品有效地建模为一个二分图和无向图。然而，许多原始的基于图的作品通常在未验证其在具体配置下的有效性的情况下采用基线论文的结果。我们的工作解决了这个问题，着重关注结果的可复制性。我们提出了一种成功复制了六个流行且最新的图推荐模型（NGCF、DGCF、LightGCN、SGL、UltraGCN和GFCF）在三个常见基准数据集（Gowalla、Yelp 2018和亚马逊图书）上的结果的代码。此外，我们将这些图模型与在离线评估中表现良好的传统协同过滤模型进行了比较。此外，我们还扩展了对两个缺乏现有文献中已建立设置的新数据集（Allrecipes和BookCrossing）的研究。由于在这些数据集上的性能与以前的基准数据集不同，使得我们对图推荐模型性能的评估结果更加深入和全面。

    The success of graph neural network-based models (GNNs) has significantly advanced recommender systems by effectively modeling users and items as a bipartite, undirected graph. However, many original graph-based works often adopt results from baseline papers without verifying their validity for the specific configuration under analysis. Our work addresses this issue by focusing on the replicability of results. We present a code that successfully replicates results from six popular and recent graph recommendation models (NGCF, DGCF, LightGCN, SGL, UltraGCN, and GFCF) on three common benchmark datasets (Gowalla, Yelp 2018, and Amazon Book). Additionally, we compare these graph models with traditional collaborative filtering models that historically performed well in offline evaluations. Furthermore, we extend our study to two new datasets (Allrecipes and BookCrossing) that lack established setups in existing literature. As the performance on these datasets differs from the previous bench
    
[^7]: 在推荐系统中插值项目和用户公平性

    Interpolating Item and User Fairness in Recommendation Systems. (arXiv:2306.10050v1 [cs.IR])

    [http://arxiv.org/abs/2306.10050](http://arxiv.org/abs/2306.10050)

    本文研究在推荐系统中平衡项目和用户公平性的框架，并通过低后悔的在线优化算法实现了维持收益同时实现公平推荐的目标。

    

    在多边平台中，平台与卖家（项目）和客户（用户）等各种各样的利益相关者互动，每个相关者都有自己的期望结果，寻找合适的平衡点变得非常复杂。在这项工作中，我们研究了“公平成本”，它捕捉了平台在平衡不同利益相关者利益时可能做出的妥协。出于这个目的，我们提出了一个公平推荐框架，其中平台在插值项目和用户公平性约束时最大化其收益。我们在一个更现实但具有挑战性的在线设置中进一步研究了公平推荐问题，在这种情况下，平台缺乏了解用户偏好的知识，只能观察二进制购买决策。为了解决这个问题，我们设计了一种低后悔的在线优化算法，它在维护平台收益的同时管理项目和用户公平性之间的权衡。我们的实验证明了我们提出的框架在实现公平推荐同时保持高收益方面的有效性。

    Online platforms employ recommendation systems to enhance customer engagement and drive revenue. However, in a multi-sided platform where the platform interacts with diverse stakeholders such as sellers (items) and customers (users), each with their own desired outcomes, finding an appropriate middle ground becomes a complex operational challenge. In this work, we investigate the ``price of fairness'', which captures the platform's potential compromises when balancing the interests of different stakeholders. Motivated by this, we propose a fair recommendation framework where the platform maximizes its revenue while interpolating between item and user fairness constraints. We further examine the fair recommendation problem in a more realistic yet challenging online setting, where the platform lacks knowledge of user preferences and can only observe binary purchase decisions. To address this, we design a low-regret online optimization algorithm that preserves the platform's revenue while
    

