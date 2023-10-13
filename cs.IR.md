# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fine-Tuning LLaMA for Multi-Stage Text Retrieval.](http://arxiv.org/abs/2310.08319) | 本研究通过对LLaMA模型进行微调，发现大型语言模型的效果优于较小模型，并且能够全面地表示整个文档，消除了传统的分段和汇集策略的需求。此外，我们的RepLLaMA-RankLLaMA流水线在零-shot情况下展现出强大的有效性。 |
| [^2] | [On Using GUI Interaction Data to Improve Text Retrieval-based Bug Localization.](http://arxiv.org/abs/2310.08083) | 本研究探讨了利用GUI交互数据来改进基于文本检索的错误定位方法。研究发现，将用户界面中的信息与错误报告中的信息相连接，并使用这些信息来检索可能有错误的文件，可以提高错误定位的准确性。 |
| [^3] | [Rethinking Negative Pairs in Code Search.](http://arxiv.org/abs/2310.08069) | 本文提出了一种简单而有效的Soft-InfoNCE损失函数，通过在InfoNCE中插入权重项来解决代码搜索中负样本的问题，包括大型代码库中的虚假负样本和未能区分负样本的潜在相关性。 |
| [^4] | [Rethinking Large-scale Pre-ranking System: Entire-chain Cross-domain Models.](http://arxiv.org/abs/2310.08039) | 本文提出了一种重新思考大规模预排序系统的方法，通过整体链路跨域模型和细粒度神经结构来解决样本选择偏差问题，并改进预排序的准确性。 |
| [^5] | [Continual Learning via Manifold Expansion Replay.](http://arxiv.org/abs/2310.08038) | 该论文提出了一种名为Manifold Expansion Replay（MaER）的新型回放策略，通过扩展知识表示中的隐式流形来减少灾难性遗忘。采用贪心策略增加缓冲区中知识表示的流形直径，并使用Wasserstein距离作为距离度量，以提高模型的鲁棒性和表达能力。 |
| [^6] | [Multi-View Variational Autoencoder for Missing Value Imputation in Untargeted Metabolomics.](http://arxiv.org/abs/2310.07990) | 本文提出了一种新的方法，利用多视图变分自动编码器来填充非目标代谢组学中的缺失值，该方法利用了全基因组测序数据和参考代谢物的信息，可以有效地根据基因组信息填充缺失的代谢组学值。 |
| [^7] | [Refined Mechanism Design for Approximately Structured Priors via Active Regression.](http://arxiv.org/abs/2310.07874) | 本研究提出了一种通过主动学习和机制设计相结合的方法，在拍卖中处理大量商品和策略招标人的问题。通过使用主题模型来近似投标人的先验分布，并设计出相应的机制，以提高机制的稳定性和适应性。 |
| [^8] | [Language Models As Semantic Indexers.](http://arxiv.org/abs/2310.07815) | 本文介绍了一种使用生成性语言模型学习语义ID的自监督框架LMINDEXER。 |
| [^9] | [Non-Stationary Contextual Bandit Learning via Neural Predictive Ensemble Sampling.](http://arxiv.org/abs/2310.07786) | 本文介绍了一种新颖的非稳态情境赌博算法，通过将可扩展的基于深度神经网络的架构与精心设计的探索机制相结合，在非稳态环境中优先收集持久价值信息，从而显著提高了性能。 |

# 详细

[^1]: 对多阶段文本检索中的LLaMA进行微调

    Fine-Tuning LLaMA for Multi-Stage Text Retrieval. (arXiv:2310.08319v1 [cs.IR])

    [http://arxiv.org/abs/2310.08319](http://arxiv.org/abs/2310.08319)

    本研究通过对LLaMA模型进行微调，发现大型语言模型的效果优于较小模型，并且能够全面地表示整个文档，消除了传统的分段和汇集策略的需求。此外，我们的RepLLaMA-RankLLaMA流水线在零-shot情况下展现出强大的有效性。

    

    自从预训练语言模型出现之前，多阶段文本检索的有效性已经得到了充分的证明。然而，大多数现有的研究都使用了过时的模型，没有考虑到最新的大型语言模型（LLMs）的进展。本研究旨在探索最先进的LLMs可能带来的改进。我们对最新的LLaMA模型进行了全面研究，通过使用MS MARCO数据集，将其作为稠密检索器（RepLLaMA）和点对点再排序器（RankLLaMA），用于段落检索和文档检索。我们的研究结果表明，大型语言模型的有效性确实超过了较小模型。此外，由于LLMs可以固有地处理更长的上下文，它们可以全面地表示整个文档，消除了传统的分段和汇集策略的需求。此外，对BEIR的评估表明，我们的RepLLaMA-RankLLaMA流水线展现了强大的零-shot有效性。来自这项研究的模型检查点...

    The effectiveness of multi-stage text retrieval has been solidly demonstrated since before the era of pre-trained language models. However, most existing studies utilize models that predate recent advances in large language models (LLMs). This study seeks to explore potential improvements that state-of-the-art LLMs can bring. We conduct a comprehensive study, fine-tuning the latest LLaMA model both as a dense retriever (RepLLaMA) and as a pointwise reranker (RankLLaMA) for both passage retrieval and document retrieval using the MS MARCO datasets. Our findings demonstrate that the effectiveness of large language models indeed surpasses that of smaller models. Additionally, since LLMs can inherently handle longer contexts, they can represent entire documents holistically, obviating the need for traditional segmenting and pooling strategies. Furthermore, evaluations on BEIR demonstrate that our RepLLaMA-RankLLaMA pipeline exhibits strong zero-shot effectiveness. Model checkpoints from thi
    
[^2]: 利用GUI交互数据来改进基于文本检索的错误定位

    On Using GUI Interaction Data to Improve Text Retrieval-based Bug Localization. (arXiv:2310.08083v1 [cs.SE])

    [http://arxiv.org/abs/2310.08083](http://arxiv.org/abs/2310.08083)

    本研究探讨了利用GUI交互数据来改进基于文本检索的错误定位方法。研究发现，将用户界面中的信息与错误报告中的信息相连接，并使用这些信息来检索可能有错误的文件，可以提高错误定位的准确性。

    

    管理错误报告中最重要的任务之一是定位故障，以便可以应用修复。因此，先前的工作致力于自动化错误定位任务，将其作为信息检索问题进行建模，通过检索潜在的有错误的文件，并根据其与给定错误报告的文本相似性进行排名。然而，错误报告中包含的信息与源代码文件中的标识符或自然语言之间通常存在明显的语义差距。对于面向用户的软件，目前存在一种关键信息来源，可以帮助错误定位，但尚未得到深入研究，即来自GUI的信息。我们研究了以下假设：对于面向最终用户的应用程序，将错误报告中的信息与GUI中的信息连接起来，并利用这些信息来帮助检索可能有错误的文件，可以改进现有的错误定位技术。为了研究这一现象，我们...（摘要未完，省略）

    One of the most important tasks related to managing bug reports is localizing the fault so that a fix can be applied. As such, prior work has aimed to automate this task of bug localization by formulating it as an information retrieval problem, where potentially buggy files are retrieved and ranked according to their textual similarity with a given bug report. However, there is often a notable semantic gap between the information contained in bug reports and identifiers or natural language contained within source code files. For user-facing software, there is currently a key source of information that could aid in bug localization, but has not been thoroughly investigated information from the GUI.  We investigate the hypothesis that, for end user-facing applications, connecting information in a bug report with information from the GUI, and using this to aid in retrieving potentially buggy files, can improve upon existing techniques for bug localization. To examine this phenomenon, we
    
[^3]: 重新思考代码搜索中的负样本对

    Rethinking Negative Pairs in Code Search. (arXiv:2310.08069v1 [cs.SE])

    [http://arxiv.org/abs/2310.08069](http://arxiv.org/abs/2310.08069)

    本文提出了一种简单而有效的Soft-InfoNCE损失函数，通过在InfoNCE中插入权重项来解决代码搜索中负样本的问题，包括大型代码库中的虚假负样本和未能区分负样本的潜在相关性。

    

    最近，对比学习成为细化代码搜索模型以提高软件开发效率和效果的关键组成部分。它将正样本代码片段聚集在一起，同时将与搜索查询不相关的负样本推开。在对比学习中，InfoNCE是最常用的损失函数，因为它具有更好的性能。然而，InfoNCE负样本存在以下问题可能会损害其表示学习的效果：1）由于重复，大型代码库中存在虚假负样本。2）未能明确区分负样本的潜在相关性。例如，对于快速排序算法查询，冒泡排序算法示例要比文件保存函数“更负面”。在本文中，我们通过提出一种简单而有效的Soft-InfoNCE损失来解决上述问题。在我们提出的损失函数中，我们采用了三种方法来估计权重...

    Recently, contrastive learning has become a key component in fine-tuning code search models for software development efficiency and effectiveness. It pulls together positive code snippets while pushing negative samples away given search queries. Among contrastive learning, InfoNCE is the most widely used loss function due to its better performance. However, the following problems in negative samples of InfoNCE may deteriorate its representation learning: 1) The existence of false negative samples in large code corpora due to duplications. 2). The failure to explicitly differentiate between the potential relevance of negative samples. As an example, a bubble sorting algorithm example is less ``negative'' than a file saving function for the quick sorting algorithm query. In this paper, we tackle the above problems by proposing a simple yet effective Soft-InfoNCE loss that inserts weight terms into InfoNCE. In our proposed loss function, we apply three methods to estimate the weights of n
    
[^4]: 重新思考大规模预排序系统：整体链路跨域模型

    Rethinking Large-scale Pre-ranking System: Entire-chain Cross-domain Models. (arXiv:2310.08039v1 [cs.IR])

    [http://arxiv.org/abs/2310.08039](http://arxiv.org/abs/2310.08039)

    本文提出了一种重新思考大规模预排序系统的方法，通过整体链路跨域模型和细粒度神经结构来解决样本选择偏差问题，并改进预排序的准确性。

    

    工业系统，如推荐系统和在线广告，已广泛配备了多阶段架构，包括匹配、预排序、排序和再排序。作为匹配和排序之间的关键桥梁，现有的预排序方法主要忽视了整个链路数据的依赖性，导致子优化性能。在本文中，我们从整体样本空间的角度重新思考预排序系统，并提出了整体链路跨域模型（ECM），利用整个级联阶段的样本来有效减轻样本选择偏差（SSB）问题。此外，我们设计了一种细粒度神经结构，名为ECMM，进一步提高了预排序的准确性。具体来说，我们提出了一个跨域多塔神经网络来综合预测每个阶段的结果，并引入$L0$正则化的子网络路由策略来调整每个阶段的贡献权重。

    Industrial systems such as recommender systems and online advertising, have been widely equipped with multi-stage architectures, which are divided into several cascaded modules, including matching, pre-ranking, ranking and re-ranking. As a critical bridge between matching and ranking, existing pre-ranking approaches mainly endure sample selection bias (SSB) problem owing to ignoring the entire-chain data dependence, resulting in sub-optimal performances. In this paper, we rethink pre-ranking system from the perspective of the entire sample space, and propose Entire-chain Cross-domain Models (ECM), which leverage samples from the whole cascaded stages to effectively alleviate SSB problem. Besides, we design a fine-grained neural structure named ECMM to further improve the pre-ranking accuracy. Specifically, we propose a cross-domain multi-tower neural network to comprehensively predict for each stage result, and introduce the sub-networking routing strategy with $L0$ regularization to r
    
[^5]: 基于流形扩展回放的持续学习

    Continual Learning via Manifold Expansion Replay. (arXiv:2310.08038v1 [cs.LG])

    [http://arxiv.org/abs/2310.08038](http://arxiv.org/abs/2310.08038)

    该论文提出了一种名为Manifold Expansion Replay（MaER）的新型回放策略，通过扩展知识表示中的隐式流形来减少灾难性遗忘。采用贪心策略增加缓冲区中知识表示的流形直径，并使用Wasserstein距离作为距离度量，以提高模型的鲁棒性和表达能力。

    

    在持续学习中，学习者按顺序学习多个任务，每个任务只获取一次数据。灾难性遗忘是持续学习的主要挑战。为了减少遗忘，一些现有的基于回放的方法使用情境记忆来重新播放先前任务的样本。然而，在学习新任务时进行知识整合的过程中，由于旧知识和新知识之间的不平衡，这种策略也会遭受灾难性遗忘。为了解决这个问题，我们提出了一种称为Manifold Expansion Replay（MaER）的新型回放策略。我们认为，在情境记忆中扩展知识表示的隐式流形有助于提高模型的鲁棒性和表达能力。为此，我们提出了一种贪心策略，在内存管理过程中，不断增加由缓冲区中的知识表示的隐式流形的直径。此外，我们将Wasserstein距离引入替代交叉熵作为距离度量。

    In continual learning, the learner learns multiple tasks in sequence, with data being acquired only once for each task. Catastrophic forgetting is a major challenge to continual learning. To reduce forgetting, some existing rehearsal-based methods use episodic memory to replay samples of previous tasks. However, in the process of knowledge integration when learning a new task, this strategy also suffers from catastrophic forgetting due to an imbalance between old and new knowledge. To address this problem, we propose a novel replay strategy called Manifold Expansion Replay (MaER). We argue that expanding the implicit manifold of the knowledge representation in the episodic memory helps to improve the robustness and expressiveness of the model. To this end, we propose a greedy strategy to keep increasing the diameter of the implicit manifold represented by the knowledge in the buffer during memory management. In addition, we introduce Wasserstein distance instead of cross entropy as dis
    
[^6]: 多视图变分自动编码器在非目标代谢组学中缺失值填充中的应用

    Multi-View Variational Autoencoder for Missing Value Imputation in Untargeted Metabolomics. (arXiv:2310.07990v1 [q-bio.GN])

    [http://arxiv.org/abs/2310.07990](http://arxiv.org/abs/2310.07990)

    本文提出了一种新的方法，利用多视图变分自动编码器来填充非目标代谢组学中的缺失值，该方法利用了全基因组测序数据和参考代谢物的信息，可以有效地根据基因组信息填充缺失的代谢组学值。

    

    背景：在基于质谱的代谢组学中，缺失数据是一个常见的挑战，可能导致偏倚和不完整的分析。将全基因组测序（WGS）数据与代谢组学数据整合起来，已经成为增强代谢组学研究中数据填充准确性的一种有前景的方法。方法：在本研究中，我们提出了一种新的方法，利用来自WGS数据和参考代谢物的信息来填充未知代谢物。我们的方法利用多视图变分自动编码器共同对负担评分、多基因风险评分（PGS）和连锁不平衡（LD）删减的单核苷酸多态性（SNPs）进行特征提取和缺失代谢组学数据的填充。通过学习两种组学数据的潜在表示，我们的方法可以根据基因组信息有效地填充缺失的代谢组学值。结果：我们在具有缺失值和不完整数据的实验代谢组学数据集上评估了我们方法的性能。

    Background: Missing data is a common challenge in mass spectrometry-based metabolomics, which can lead to biased and incomplete analyses. The integration of whole-genome sequencing (WGS) data with metabolomics data has emerged as a promising approach to enhance the accuracy of data imputation in metabolomics studies. Method: In this study, we propose a novel method that leverages the information from WGS data and reference metabolites to impute unknown metabolites. Our approach utilizes a multi-view variational autoencoder to jointly model the burden score, polygenetic risk score (PGS), and linkage disequilibrium (LD) pruned single nucleotide polymorphisms (SNPs) for feature extraction and missing metabolomics data imputation. By learning the latent representations of both omics data, our method can effectively impute missing metabolomics values based on genomic information. Results: We evaluate the performance of our method on empirical metabolomics datasets with missing values and de
    
[^7]: 通过主动回归对近似结构的先验进行精细机制设计

    Refined Mechanism Design for Approximately Structured Priors via Active Regression. (arXiv:2310.07874v1 [cs.GT])

    [http://arxiv.org/abs/2310.07874](http://arxiv.org/abs/2310.07874)

    本研究提出了一种通过主动学习和机制设计相结合的方法，在拍卖中处理大量商品和策略招标人的问题。通过使用主题模型来近似投标人的先验分布，并设计出相应的机制，以提高机制的稳定性和适应性。

    

    我们考虑了一个有大量商品m出售给n个策略招标人的最大化收入卖方的问题，他们的估值是从高维未知的先验分布中独立抽取的。众所周知，这种情况下的最优甚至近似最优的机制很难表达或计算，而且即使找到了，通常也具有各种反直觉的特性。在本文中，根据Cai和Daskalakis最近提出的模型，我们考虑投标人的先验分布可以被一个主题模型很好地近似。我们设计了一个负责与投标人进行交互并输出其类型的低维近似的主动学习组件，以及一个负责为低维模型设计机制以适应前一组件的近似类型的机制设计组件。

    We consider the problem of a revenue-maximizing seller with a large number of items $m$ for sale to $n$ strategic bidders, whose valuations are drawn independently from high-dimensional, unknown prior distributions. It is well-known that optimal and even approximately-optimal mechanisms for this setting are notoriously difficult to characterize or compute, and, even when they can be found, are often rife with various counter-intuitive properties. In this paper, following a model introduced recently by Cai and Daskalakis~\cite{cai2022recommender}, we consider the case that bidders' prior distributions can be well-approximated by a topic model. We design an active learning component, responsible for interacting with the bidders and outputting low-dimensional approximations of their types, and a mechanism design component, responsible for robustifying mechanisms for the low-dimensional model to work for the approximate types of the former component. On the active learning front, we cast o
    
[^8]: 语言模型作为语义索引器

    Language Models As Semantic Indexers. (arXiv:2310.07815v1 [cs.IR])

    [http://arxiv.org/abs/2310.07815](http://arxiv.org/abs/2310.07815)

    本文介绍了一种使用生成性语言模型学习语义ID的自监督框架LMINDEXER。

    

    语义标识符（ID）是信息检索中的一个重要概念，旨在保留对象（如文档和项）内部的语义。先前的研究通常采用两阶段流程来学习语义ID，首先使用现成的文本编码器获取嵌入，并根据嵌入来推导ID。然而，每个步骤都会引入潜在的信息损失，并且文本编码器生成的潜在空间内的嵌入分布通常与语义索引所需的预期分布存在固有的不匹配。然而，设计一个既能学习文档的语义表示又能同时学习其分层结构的方法并不容易，因为语义ID是离散和顺序结构的，并且语义监督是不充分的。在本文中，我们引入了LMINDEXER，它是一个自监督框架，用于使用生成性语言模型学习语义ID。

    Semantic identifier (ID) is an important concept in information retrieval that aims to preserve the semantics of objects such as documents and items inside their IDs. Previous studies typically adopt a two-stage pipeline to learn semantic IDs by first procuring embeddings using off-the-shelf text encoders and then deriving IDs based on the embeddings. However, each step introduces potential information loss and there is usually an inherent mismatch between the distribution of embeddings within the latent space produced by text encoders and the anticipated distribution required for semantic indexing. Nevertheless, it is non-trivial to design a method that can learn the document's semantic representations and its hierarchical structure simultaneously, given that semantic IDs are discrete and sequentially structured, and the semantic supervision is deficient. In this paper, we introduce LMINDEXER, a self-supervised framework to learn semantic IDs with a generative language model. We tackl
    
[^9]: 非稳态环境下基于神经预测集成抽样的情境赌博学习

    Non-Stationary Contextual Bandit Learning via Neural Predictive Ensemble Sampling. (arXiv:2310.07786v1 [cs.LG])

    [http://arxiv.org/abs/2310.07786](http://arxiv.org/abs/2310.07786)

    本文介绍了一种新颖的非稳态情境赌博算法，通过将可扩展的基于深度神经网络的架构与精心设计的探索机制相结合，在非稳态环境中优先收集持久价值信息，从而显著提高了性能。

    

    实际世界中的情境赌博应用常常因季节性、偶然性和不断变化的社交趋势而呈非稳态。尽管文献中已提出了许多非稳态情境赌博学习算法，但由于缺乏对持久价值信息的优先考虑，这些算法在探索时过度，或者设计方式难以在具有高维用户特定特征和大规模动作集的现代应用中扩展，或者两者都有。在本文中，我们介绍了一种新颖的非稳态情境赌博算法，它解决了这些问题。它将可扩展的基于深度神经网络的架构与一个精心设计的探索机制相结合，在非稳态环境中战略性地优先收集具有最持久价值的信息。通过在展示明显非稳态的两个实际推荐数据集上进行实证评估，我们证明了我们的方法显著胜过现有的算法。

    Real-world applications of contextual bandits often exhibit non-stationarity due to seasonality, serendipity, and evolving social trends. While a number of non-stationary contextual bandit learning algorithms have been proposed in the literature, they excessively explore due to a lack of prioritization for information of enduring value, or are designed in ways that do not scale in modern applications with high-dimensional user-specific features and large action set, or both. In this paper, we introduce a novel non-stationary contextual bandit algorithm that addresses these concerns. It combines a scalable, deep-neural-network-based architecture with a carefully designed exploration mechanism that strategically prioritizes collecting information with the most lasting value in a non-stationary environment. Through empirical evaluations on two real-world recommendation datasets, which exhibit pronounced non-stationarity, we demonstrate that our approach significantly outperforms the state
    

