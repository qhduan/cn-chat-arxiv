# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Caseformer: Pre-training for Legal Case Retrieval.](http://arxiv.org/abs/2311.00333) | 本文提出了一种新颖的预训练方法，名为Caseformer，在法律案例检索中解决了标注数据不足的问题，能够更好地理解和捕捉法律语料库中的关键知识和数据结构。 |
| [^2] | [A statistical significance testing approach for measuring term burstiness with applications to domain-specific terminology extraction.](http://arxiv.org/abs/2310.15790) | 我们提出了一种统计显著性测试方法，用于测量专业术语抽取中的术语爆发性。我们的方法基于多项式语言模型，通过启发式公式得到近似测试P值。此外，我们还推导了逆文档频率与逆收集频率之间的关系。 |

# 详细

[^1]: Caseformer: 法律案例检索的预训练

    Caseformer: Pre-training for Legal Case Retrieval. (arXiv:2311.00333v1 [cs.IR])

    [http://arxiv.org/abs/2311.00333](http://arxiv.org/abs/2311.00333)

    本文提出了一种新颖的预训练方法，名为Caseformer，在法律案例检索中解决了标注数据不足的问题，能够更好地理解和捕捉法律语料库中的关键知识和数据结构。

    

    法律案例检索旨在帮助法律工作者找到与他们手头案件相关的案例，这对于保证公平和正义的法律判决非常重要。尽管最近神经检索方法在开放域检索任务（例如网络搜索）方面取得了显著的改进，但是由于对标注数据的渴望，这些方法在法律案例检索中并没有显示出优势。由于需要领域专业知识，对法律领域进行大规模训练数据的标注是困难的，因此传统的基于词汇匹配的搜索技术，如TF-IDF、BM25和查询似然，仍然在法律案例检索系统中盛行。虽然以前的研究已经设计了一些针对开放域任务中IR模型的预训练方法，但是由于无法理解和捕捉法律语料库中的关键知识和数据结构，这些方法在法律案例检索中通常是次优的。为此，我们提出了一种新颖的预训练方法。

    Legal case retrieval aims to help legal workers find relevant cases related to their cases at hand, which is important for the guarantee of fairness and justice in legal judgments. While recent advances in neural retrieval methods have significantly improved the performance of open-domain retrieval tasks (e.g., Web search), their advantages have not been observed in legal case retrieval due to their thirst for annotated data. As annotating large-scale training data in legal domains is prohibitive due to the need for domain expertise, traditional search techniques based on lexical matching such as TF-IDF, BM25, and Query Likelihood are still prevalent in legal case retrieval systems. While previous studies have designed several pre-training methods for IR models in open-domain tasks, these methods are usually suboptimal in legal case retrieval because they cannot understand and capture the key knowledge and data structures in the legal corpus. To this end, we propose a novel pre-trainin
    
[^2]: 一种用于测量专业术语抽取中术语爆发性的统计显著性测试方法

    A statistical significance testing approach for measuring term burstiness with applications to domain-specific terminology extraction. (arXiv:2310.15790v1 [cs.IR])

    [http://arxiv.org/abs/2310.15790](http://arxiv.org/abs/2310.15790)

    我们提出了一种统计显著性测试方法，用于测量专业术语抽取中的术语爆发性。我们的方法基于多项式语言模型，通过启发式公式得到近似测试P值。此外，我们还推导了逆文档频率与逆收集频率之间的关系。

    

    专业术语抽取是文本分析中的重要任务。当语料库中一个术语的出现集中在少数几个文件中时，可称之为“爆发性”。作为内容丰富的术语，爆发性术语非常适合用于主题描述，并且是技术术语的自然候选词。文献中提出了多种术语爆发性的测量方法。然而，在文本分析中，包括与术语爆发性相关的统计显著性测试范式尚未得到充分探索。为了探索这个领域，我们的主要贡献是提出了一种基于多项式语言模型的术语爆发性统计显著性的精确测试方法。由于计算成本过高，我们还提出了一个启发式公式，用于近似测试P值。作为补充的理论贡献，我们推导了一种未经报道的逆文档频率与逆收集频率的关系。

    Domain-specific terminology extraction is an important task in text analysis. A term in a corpus is said to be "bursty" when its occurrences are concentrated in few out of many documents. Being content rich, bursty terms are highly suited for subject matter characterization, and serve as natural candidates for identifying with technical terminology. Multiple measures of term burstiness have been proposed in the literature. However, the statistical significance testing paradigm has remained underexplored in text analysis, including in relation to term burstiness. To test these waters, we propose as our main contribution a multinomial language model-based exact test of statistical significance for term burstiness. Due to its prohibitive computational cost, we advance a heuristic formula designed to serve as a proxy for test P-values. As a complementary theoretical contribution, we derive a previously unreported relationship connecting the inverse document frequency and inverse collection
    

