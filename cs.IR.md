# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLMRec: Large Language Models with Graph Augmentation for Recommendation.](http://arxiv.org/abs/2311.00423) | LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。 |
| [^2] | [Density-based User Representation through Gaussian Process Regression for Multi-interest Personalized Retrieval.](http://arxiv.org/abs/2310.20091) | 本研究引入了一种基于密度的用户表示(DURs)，利用高斯过程回归实现了有效的多兴趣推荐和检索。该方法不仅能够捕捉用户的兴趣变化，还具备不确定性感知能力，并且适用于大量用户的规模。 |
| [^3] | [On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation.](http://arxiv.org/abs/2307.15053) | 本文批判性审视了(Normalised) Discounted Cumulative Gain作为Top-n推荐离线评估指标的方法，并研究了何时可以期望这些指标逼近在线实验的金标准结果。 |
| [^4] | [GPT4Table: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study.](http://arxiv.org/abs/2305.13062) | 本文设计了一个基准测试来评估大型语言模型（LLMs）对结构化表格数据的理解能力，并发现不同的输入选择会对性能产生影响。在基准测试的基础上，提出了“自我增强”技术以改善理解能力。 |
| [^5] | [Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems.](http://arxiv.org/abs/2305.12102) | 本文介绍了一种名为“特征复用”的框架，它使用单一的表示空间 能够高效有效地学习高质量的特征嵌入，同时区分不同的分类特征。通过在多个公共数据集和新数据集“Web-Available Image Search (WAIS)”上的测试，我们展示了这种方法的优于现有技术的表现。 |
| [^6] | [Editing Language Model-based Knowledge Graph Embeddings.](http://arxiv.org/abs/2301.10405) | 本文提出了一种新的任务——编辑基于语言模型的知识图谱嵌入，旨在实现对KG嵌入的数据高效和快速更新。针对这一任务，提出了一个简单而强大的方案——KGEditor，可以更好地更新特定事实而不影响其余部分的性能。 |

# 详细

[^1]: LLMRec: 使用图增强的大型语言模型用于推荐系统

    LLMRec: Large Language Models with Graph Augmentation for Recommendation. (arXiv:2311.00423v1 [cs.IR])

    [http://arxiv.org/abs/2311.00423](http://arxiv.org/abs/2311.00423)

    LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。

    

    数据稀疏性一直是推荐系统中的一个挑战，之前的研究尝试通过引入附加信息来解决这个问题。然而，这种方法往往会带来噪声、可用性问题和数据质量低下等副作用，从而影响对用户偏好的准确建模，进而对推荐性能产生不利影响。鉴于大型语言模型（LLM）在知识库和推理能力方面的最新进展，我们提出了一个名为LLMRec的新框架，它通过采用三种简单而有效的基于LLM的图增强策略来增强推荐系统。我们的方法利用在线平台（如Netflix，MovieLens）中丰富的内容，在三个方面增强交互图：（i）加强用户-物品交互边，（ii）增强对物品节点属性的理解，（iii）进行用户节点建模，直观地表示用户特征。

    The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuiti
    
[^2]: 基于高斯过程回归的密度用户表示方法用于多兴趣个性化检索

    Density-based User Representation through Gaussian Process Regression for Multi-interest Personalized Retrieval. (arXiv:2310.20091v1 [cs.IR])

    [http://arxiv.org/abs/2310.20091](http://arxiv.org/abs/2310.20091)

    本研究引入了一种基于密度的用户表示(DURs)，利用高斯过程回归实现了有效的多兴趣推荐和检索。该方法不仅能够捕捉用户的兴趣变化，还具备不确定性感知能力，并且适用于大量用户的规模。

    

    在设计个性化推荐系统中，准确建模用户的各种多样化和动态的兴趣仍然是一个重大挑战。现有的用户建模方法，如单点和多点表示，存在准确性、多样性、计算成本和适应性方面的局限性。为了克服这些不足，我们引入了一种新颖的模型——基于密度的用户表示(DURs)，它利用高斯过程回归实现有效的多兴趣推荐和检索。我们的方法GPR4DUR利用DURs来捕捉用户的兴趣变化，无需手动调整，同时具备不确定性感知能力，并且适用于大量用户的规模。使用真实世界的离线数据集进行的实验证实了GPR4DUR的适应性和效率，而使用模拟用户的在线实验则证明了它通过有效利用模型的不确定性，能够解决探索-开发的平衡问题。

    Accurate modeling of the diverse and dynamic interests of users remains a significant challenge in the design of personalized recommender systems. Existing user modeling methods, like single-point and multi-point representations, have limitations w.r.t. accuracy, diversity, computational cost, and adaptability. To overcome these deficiencies, we introduce density-based user representations (DURs), a novel model that leverages Gaussian process regression for effective multi-interest recommendation and retrieval. Our approach, GPR4DUR, exploits DURs to capture user interest variability without manual tuning, incorporates uncertainty-awareness, and scales well to large numbers of users. Experiments using real-world offline datasets confirm the adaptability and efficiency of GPR4DUR, while online experiments with simulated users demonstrate its ability to address the exploration-exploitation trade-off by effectively utilizing model uncertainty.
    
[^3]: 关于(Normalised) Discounted Cumulative Gain作为Top-n推荐的离线评估指标的论文翻译

    On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation. (arXiv:2307.15053v1 [cs.IR])

    [http://arxiv.org/abs/2307.15053](http://arxiv.org/abs/2307.15053)

    本文批判性审视了(Normalised) Discounted Cumulative Gain作为Top-n推荐离线评估指标的方法，并研究了何时可以期望这些指标逼近在线实验的金标准结果。

    

    推荐方法通常通过两种方式进行评估：(1) 通过(模拟)在线实验，通常被视为金标准，或者(2) 通过一些离线评估程序，目标是近似在线实验的结果。文献中采用了几种离线评估指标，受信息检索领域中常见的排名指标的启发。(Normalised) Discounted Cumulative Gain (nDCG)是其中一种广泛采用的度量标准，在很多年里，更高的(n)DCG值被用来展示新方法在Top-n推荐中的最新进展。我们的工作对这种方法进行了批判性的审视，并研究了我们何时可以期望这些指标逼近在线实验的金标准结果。我们从第一原理上正式提出了DCG被认为是在线奖励的无偏估计的假设，并给出了这个指标的推导。

    Approaches to recommendation are typically evaluated in one of two ways: (1) via a (simulated) online experiment, often seen as the gold standard, or (2) via some offline evaluation procedure, where the goal is to approximate the outcome of an online experiment. Several offline evaluation metrics have been adopted in the literature, inspired by ranking metrics prevalent in the field of Information Retrieval. (Normalised) Discounted Cumulative Gain (nDCG) is one such metric that has seen widespread adoption in empirical studies, and higher (n)DCG values have been used to present new methods as the state-of-the-art in top-$n$ recommendation for many years.  Our work takes a critical look at this approach, and investigates when we can expect such metrics to approximate the gold standard outcome of an online experiment. We formally present the assumptions that are necessary to consider DCG an unbiased estimator of online reward and provide a derivation for this metric from first principles
    
[^4]: GPT4Table：大型语言模型能理解结构化表格数据吗？一项基准测试和实证研究

    GPT4Table: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study. (arXiv:2305.13062v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13062](http://arxiv.org/abs/2305.13062)

    本文设计了一个基准测试来评估大型语言模型（LLMs）对结构化表格数据的理解能力，并发现不同的输入选择会对性能产生影响。在基准测试的基础上，提出了“自我增强”技术以改善理解能力。

    

    大型语言模型（LLMs）作为少样本推理器来解决与自然语言相关的任务越来越具吸引力。然而，关于LLMs对结构化数据（例如表格）的理解程度还有很多需要学习的地方。尽管可以使用表格序列化作为LLMs的输入，但目前还缺乏对LLMs是否真正能够理解这类数据的全面研究。本文通过设计一个基准测试来评估LLMs的结构理解能力（SUC）来解决这个问题。我们创建的基准测试包括七个任务，每个任务都有其独特的挑战，例如单元格查找、行检索和大小检测。我们对GPT-3.5和GPT-4进行了一系列评估。我们发现性能因多种输入选择而异，包括表格输入格式、内容顺序、角色提示和分区标记等。根据基准测试评估所得的见解，我们提出了“自我增强”技术以改善性能。

    Large language models (LLMs) are becoming attractive as few-shot reasoners to solve Natural Language (NL)-related tasks. However, there is still much to learn about how well LLMs understand structured data, such as tables. While it is true that tables can be used as inputs to LLMs with serialization, there lack of comprehensive studies examining whether LLMs can truly comprehend such data. In this paper, we try to understand this by designing a benchmark to evaluate the structural understanding capabilities (SUC) of LLMs. The benchmark we create includes seven tasks, each with its own unique challenges, \eg, cell lookup, row retrieval, and size detection. We run a series of evaluations on GPT-3.5 and GPT-4. We discover that the performance varied depending on a number of input choices, including table input format, content order, role prompting, and partition marks. Drawing from the insights gained through the benchmark evaluations, we then propose \textit{self-augmentation} for effect
    
[^5]: 统一嵌入：面向 Web 规模 ML 系统的经过验证的特征表示

    Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems. (arXiv:2305.12102v1 [cs.LG])

    [http://arxiv.org/abs/2305.12102](http://arxiv.org/abs/2305.12102)

    本文介绍了一种名为“特征复用”的框架，它使用单一的表示空间 能够高效有效地学习高质量的特征嵌入，同时区分不同的分类特征。通过在多个公共数据集和新数据集“Web-Available Image Search (WAIS)”上的测试，我们展示了这种方法的优于现有技术的表现。

    

    高效、有效地学习高质量的特征嵌入对于 Web 规模的机器学习系统的性能至关重要。标准方法是将每个特征值表示为一个 d 维嵌入，引入数百亿个参数，而这些特征的基数非常高。这个瓶颈导致了备选嵌入算法的重大进展。本文介绍了一个简单但非常有效的框架，即“特征复用”，在许多不同的分类特征之间使用一个单一的表示空间。我们的理论和实证分析表明，复用的嵌入可以分解为每个组成特征的组件，使得模型可以区分特征。我们展示了复用的嵌入在几个公共数据集上优于现有技术。此外，我们引入了一个名为“Web-Available Image Search (WAIS)”的新数据集，以严格评估 Web 规模下的新嵌入算法。我们邀请社区通过提出可以准确、高效地将数百万张图像嵌入和分类到成千上万个类别的新模型来贡献 WAIS 挑战。

    Learning high-quality feature embeddings efficiently and effectively is critical for the performance of web-scale machine learning systems. A typical model ingests hundreds of features with vocabularies on the order of millions to billions of tokens. The standard approach is to represent each feature value as a d-dimensional embedding, introducing hundreds of billions of parameters for extremely high-cardinality features. This bottleneck has led to substantial progress in alternative embedding algorithms. Many of these methods, however, make the assumption that each feature uses an independent embedding table. This work introduces a simple yet highly effective framework, Feature Multiplexing, where one single representation space is used across many different categorical features. Our theoretical and empirical analysis reveals that multiplexed embeddings can be decomposed into components from each constituent feature, allowing models to distinguish between features. We show that multip
    
[^6]: 基于语言模型的知识图谱嵌入编辑

    Editing Language Model-based Knowledge Graph Embeddings. (arXiv:2301.10405v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.10405](http://arxiv.org/abs/2301.10405)

    本文提出了一种新的任务——编辑基于语言模型的知识图谱嵌入，旨在实现对KG嵌入的数据高效和快速更新。针对这一任务，提出了一个简单而强大的方案——KGEditor，可以更好地更新特定事实而不影响其余部分的性能。

    

    近几十年来，使用语言模型进行知识图谱（KG）嵌入已经取得了实证成功。但是，基于语言模型的KG嵌入通常作为静态工件部署，修改起来具有挑战性，需要重新训练。为了解决这个问题，本文提出了一种新的任务，即编辑基于语言模型的KG嵌入。该任务旨在实现对KG嵌入的数据高效和快速更新，而不影响其余部分的性能。我们构建了四个新数据集：E-FB15k237、A-FB15k237、E-WN18RR 和 A-WN18RR，并评估了几种知识编辑基线，证明了之前的模型处理该任务的能力有限。我们进一步提出了一个简单但强大的基线——KGEditor，它利用超网络的附加参数层来编辑/添加事实。全面的实验结果表明，当更新特定事实而不影响其余部分的性能时，KGEditor 的表现更好。

    Recently decades have witnessed the empirical success of framing Knowledge Graph (KG) embeddings via language models. However, language model-based KG embeddings are usually deployed as static artifacts, which are challenging to modify without re-training after deployment. To address this issue, we propose a new task of editing language model-based KG embeddings in this paper. The proposed task aims to enable data-efficient and fast updates to KG embeddings without damaging the performance of the rest. We build four new datasets: E-FB15k237, A-FB15k237, E-WN18RR, and A-WN18RR, and evaluate several knowledge editing baselines demonstrating the limited ability of previous models to handle the proposed challenging task. We further propose a simple yet strong baseline dubbed KGEditor, which utilizes additional parametric layers of the hyper network to edit/add facts. Comprehensive experimental results demonstrate that KGEditor can perform better when updating specific facts while not affec
    

