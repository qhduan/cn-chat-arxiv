# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Seed-based information retrieval in networks of research publications: Evaluation of direct citations, bibliographic coupling, co-citations and PubMed related article score](https://arxiv.org/abs/2403.09295) | 论文比较了基于种子的研究出版物网络信息检索中的直接引用、文献耦合、共同引用和PubMed相关文章得分等不同方法的表现，发现组合引文方法优于仅使用共同引用。 |
| [^2] | [Learning Metrics that Maximise Power for Accelerated A/B-Tests](https://arxiv.org/abs/2402.03915) | 本论文提出了一种新方法，通过从短期信号中学习指标，直接最大化指标与北极度量标准之间的统计能力，从而减少在线控制实验的成本。 |
| [^3] | [Whole Page Unbiased Learning to Rank](https://arxiv.org/abs/2210.10718) | 本论文提出整页无偏学习排序（WP-ULTR）方法处理整页 SERP 特征引发的偏差，该方法面临适合的用户行为模型的挑战和复杂的模型训练难题。 |
| [^4] | [Language Models As Semantic Indexers.](http://arxiv.org/abs/2310.07815) | 本文介绍了一种使用生成性语言模型学习语义ID的自监督框架LMINDEXER。 |

# 详细

[^1]: 基于种子的研究出版物网络信息检索：对直接引用、文献耦合、共同引用和PubMed相关文章得分的评估

    Seed-based information retrieval in networks of research publications: Evaluation of direct citations, bibliographic coupling, co-citations and PubMed related article score

    [https://arxiv.org/abs/2403.09295](https://arxiv.org/abs/2403.09295)

    论文比较了基于种子的研究出版物网络信息检索中的直接引用、文献耦合、共同引用和PubMed相关文章得分等不同方法的表现，发现组合引文方法优于仅使用共同引用。

    

    在这篇论文中，我们探讨了基于种子的研究出版物网络信息检索。使用系统评审作为基准，结合NIH开放引文收集的出版数据，我们比较了三种基于引文的方法——直接引用、共同引用和文献耦合在召回率和精确率方面的表现。此外，我们还将PubMed相关文章得分以及组合方法纳入比较。我们还对先前使用引文关系进行信息检索的早期研究进行了相当全面的回顾。结果显示共同引用优于文献耦合和直接引用。然而，在研究中，将这三种方法组合起来胜过仅使用共同引用。结果进一步表明，与先前研究一致，将基于引文的方法与文本方法相结合

    arXiv:2403.09295v1 Announce Type: new  Abstract: In this contribution, we deal with seed-based information retrieval in networks of research publications. Using systematic reviews as a baseline, and publication data from the NIH Open Citation Collection, we compare the performance of the three citation-based approaches direct citation, co-citation, and bibliographic coupling with respect to recall and precision measures. In addition, we include the PubMed Related Article score as well as combined approaches in the comparison. We also provide a fairly comprehensive review of earlier research in which citation relations have been used for information retrieval purposes. The results show an advantage for co-citation over bibliographic coupling and direct citation. However, combining the three approaches outperforms the exclusive use of co-citation in the study. The results further indicate, in line with previous research, that combining citation-based approaches with textual approaches en
    
[^2]: 学习最大化加速A/B测试的指标

    Learning Metrics that Maximise Power for Accelerated A/B-Tests

    [https://arxiv.org/abs/2402.03915](https://arxiv.org/abs/2402.03915)

    本论文提出了一种新方法，通过从短期信号中学习指标，直接最大化指标与北极度量标准之间的统计能力，从而减少在线控制实验的成本。

    

    在技术公司中，在线控制实验是一种重要的工具，可以实现自信的决策。定义了一个北极度量标准（如长期收入或用户保留），在A/B测试中，能够在这个指标上有统计显著提升的系统变体可以被认为是优越的。然而，北极度量标准通常具有时延和不敏感性。因此，实验的成本很高：实验需要长时间运行，即使如此，二类错误（即假阴性）仍然普遍存在。为了解决这个问题，我们提出了一种从短期信号中学习指标的方法，这些指标直接最大化它们相对于北极度量标准所具有的统计能力。我们展示了现有方法容易过拟合的问题，即更高的平均度量敏感性并不意味着改进了二类错误，我们建议通过最小化指标在过去实验的$log$上产生的$p$-value来解决。我们从两个社交媒体应用程序中收集了这样的数据集。

    Online controlled experiments are a crucial tool to allow for confident decision-making in technology companies. A North Star metric is defined (such as long-term revenue or user retention), and system variants that statistically significantly improve on this metric in an A/B-test can be considered superior. North Star metrics are typically delayed and insensitive. As a result, the cost of experimentation is high: experiments need to run for a long time, and even then, type-II errors (i.e. false negatives) are prevalent.   We propose to tackle this by learning metrics from short-term signals that directly maximise the statistical power they harness with respect to the North Star. We show that existing approaches are prone to overfitting, in that higher average metric sensitivity does not imply improved type-II errors, and propose to instead minimise the $p$-values a metric would have produced on a log of past experiments. We collect such datasets from two social media applications with
    
[^3]: 整页无偏学习排序

    Whole Page Unbiased Learning to Rank

    [https://arxiv.org/abs/2210.10718](https://arxiv.org/abs/2210.10718)

    本论文提出整页无偏学习排序（WP-ULTR）方法处理整页 SERP 特征引发的偏差，该方法面临适合的用户行为模型的挑战和复杂的模型训练难题。

    

    信息检索系统中页面呈现的偏见，尤其是点击行为方面的偏差，是一个众所周知的挑战，阻碍了使用隐式用户反馈来改进排序模型的性能。因此，提出了无偏学习排序(ULTR)算法，通过偏差点击数据来学习一个无偏的排序模型。然而，大多数现有算法特别设计用于减轻与位置相关的偏差，例如信任偏差，并未考虑到搜索结果页面呈现(SERP)中其他特征引发的偏差，例如由多媒体引发的吸引偏差。不幸的是，这些偏差在工业系统中广泛存在，可能导致不令人满意的搜索体验。因此，我们引入了一个新的问题，即整页无偏学习排序(WP-ULTR)，旨在同时处理整页SERP特征引发的偏差。这带来了巨大的挑战：(1) 很难找到适合的用户行为模型 (用户行为假设)；(2) 复杂的模型训练问题。

    The page presentation biases in the information retrieval system, especially on the click behavior, is a well-known challenge that hinders improving ranking models' performance with implicit user feedback. Unbiased Learning to Rank~(ULTR) algorithms are then proposed to learn an unbiased ranking model with biased click data. However, most existing algorithms are specifically designed to mitigate position-related bias, e.g., trust bias, without considering biases induced by other features in search result page presentation(SERP), e.g. attractive bias induced by the multimedia. Unfortunately, those biases widely exist in industrial systems and may lead to an unsatisfactory search experience. Therefore, we introduce a new problem, i.e., whole-page Unbiased Learning to Rank(WP-ULTR), aiming to handle biases induced by whole-page SERP features simultaneously. It presents tremendous challenges: (1) a suitable user behavior model (user behavior hypothesis) can be hard to find; and (2) complex
    
[^4]: 语言模型作为语义索引器

    Language Models As Semantic Indexers. (arXiv:2310.07815v1 [cs.IR])

    [http://arxiv.org/abs/2310.07815](http://arxiv.org/abs/2310.07815)

    本文介绍了一种使用生成性语言模型学习语义ID的自监督框架LMINDEXER。

    

    语义标识符（ID）是信息检索中的一个重要概念，旨在保留对象（如文档和项）内部的语义。先前的研究通常采用两阶段流程来学习语义ID，首先使用现成的文本编码器获取嵌入，并根据嵌入来推导ID。然而，每个步骤都会引入潜在的信息损失，并且文本编码器生成的潜在空间内的嵌入分布通常与语义索引所需的预期分布存在固有的不匹配。然而，设计一个既能学习文档的语义表示又能同时学习其分层结构的方法并不容易，因为语义ID是离散和顺序结构的，并且语义监督是不充分的。在本文中，我们引入了LMINDEXER，它是一个自监督框架，用于使用生成性语言模型学习语义ID。

    Semantic identifier (ID) is an important concept in information retrieval that aims to preserve the semantics of objects such as documents and items inside their IDs. Previous studies typically adopt a two-stage pipeline to learn semantic IDs by first procuring embeddings using off-the-shelf text encoders and then deriving IDs based on the embeddings. However, each step introduces potential information loss and there is usually an inherent mismatch between the distribution of embeddings within the latent space produced by text encoders and the anticipated distribution required for semantic indexing. Nevertheless, it is non-trivial to design a method that can learn the document's semantic representations and its hierarchical structure simultaneously, given that semantic IDs are discrete and sequentially structured, and the semantic supervision is deficient. In this paper, we introduce LMINDEXER, a self-supervised framework to learn semantic IDs with a generative language model. We tackl
    

