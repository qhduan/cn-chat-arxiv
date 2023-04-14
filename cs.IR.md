# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dual Contrastive Network for Sequential Recommendation with User and Item-Centric Perspectives.](http://arxiv.org/abs/2209.08446) | 本文提出了一种新颖的双重对比网络（DCN），通过两个组件充分利用了用户和物品两个视角，以生成地面真实的自监督信号，解决了随机屏蔽历史物品带来的序列稀疏性和不可靠信号的问题。实验结果表明，在三个真实数据集上，我们的方法优于当前最先进的纵向推荐方法。 |
| [^2] | [SpaDE: Improving Sparse Representations using a Dual Document Encoder for First-stage Retrieval.](http://arxiv.org/abs/2209.05917) | SpaDE 是一种利用双重编码器学习文档表示的第一阶段检索模型，可以同时改善词汇匹配和扩展额外术语来支持语义匹配，且在实验中表现优异。 |
| [^3] | [Tensor Completion with Provable Consistency and Fairness Guarantees for Recommender Systems.](http://arxiv.org/abs/2204.01815) | 本文介绍了一种新的一致性方法来解决矩阵和张量补全问题，在推荐系统应用中，我们证明了通过保留单位比例和一致性两个约束条件可以实现解的存在性与唯一性。 |

# 详细

[^1]: 纵向推荐中用户和物品视角的双重对比网络

    Dual Contrastive Network for Sequential Recommendation with User and Item-Centric Perspectives. (arXiv:2209.08446v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2209.08446](http://arxiv.org/abs/2209.08446)

    本文提出了一种新颖的双重对比网络（DCN），通过两个组件充分利用了用户和物品两个视角，以生成地面真实的自监督信号，解决了随机屏蔽历史物品带来的序列稀疏性和不可靠信号的问题。实验结果表明，在三个真实数据集上，我们的方法优于当前最先进的纵向推荐方法。

    

    随着流媒体数据的爆发，纵向推荐成为实现时间感知个性化建模的一种有前途的解决方案。本文提出了一种新颖的双重对比网络（DCN），通过两个组件——基于用户的对比学习和基于物品的对比学习，充分利用了用户和物品两个视角，以生成地面真实的自监督信号，解决了随机屏蔽历史物品带来的序列稀疏性和不可靠信号的问题。实验结果表明，在三个真实数据集上，我们的方法优于当前最先进的纵向推荐方法。

    With the outbreak of today's streaming data, the sequential recommendation is a promising solution to achieve time-aware personalized modeling. It aims to infer the next interacted item of a given user based on the history item sequence. Some recent works tend to improve the sequential recommendation via random masking on the history item so as to generate self-supervised signals. But such approaches will indeed result in sparser item sequence and unreliable signals. Besides, the existing sequential recommendation models are only user-centric, i.e., based on the historical items by chronological order to predict the probability of candidate items, which ignores whether the items from a provider can be successfully recommended. Such user-centric recommendation will make it impossible for the provider to expose their new items and result in popular bias.  In this paper, we propose a novel Dual Contrastive Network (DCN) to generate ground-truth self-supervised signals for sequential recom
    
[^2]: SpaDE: 一种利用双重文档编码器改善稀疏表示的第一阶段检索方法

    SpaDE: Improving Sparse Representations using a Dual Document Encoder for First-stage Retrieval. (arXiv:2209.05917v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2209.05917](http://arxiv.org/abs/2209.05917)

    SpaDE 是一种利用双重编码器学习文档表示的第一阶段检索模型，可以同时改善词汇匹配和扩展额外术语来支持语义匹配，且在实验中表现优异。

    

    稀疏的文档表示经常被用来通过精确的词汇匹配来检索相关文档。然而，由于预先计算的倒排索引，会引发词汇不匹配的问题。虽然最近使用预训练语言模型的神经排序模型可以解决这个问题，但它们通常需要昂贵的查询推理成本，这意味着效率和效果之间存在权衡。为了解决这个问题，我们提出了一种新的单编码器排名模型，利用双重编码器学习文档表示，称为 Sparse retriever using a Dual document Encoder (SpaDE)。每个编码器在改善词汇匹配和扩展额外术语来支持语义匹配方面发挥着核心作用。此外，我们的协同训练策略可以有效地训练双重编码器，并避免不必要的干预彼此的训练过程。在几个基准测试中的实验结果表明，SpaDE 超越了现有的检索方法。

    Sparse document representations have been widely used to retrieve relevant documents via exact lexical matching. Owing to the pre-computed inverted index, it supports fast ad-hoc search but incurs the vocabulary mismatch problem. Although recent neural ranking models using pre-trained language models can address this problem, they usually require expensive query inference costs, implying the trade-off between effectiveness and efficiency. Tackling the trade-off, we propose a novel uni-encoder ranking model, Sparse retriever using a Dual document Encoder (SpaDE), learning document representation via the dual encoder. Each encoder plays a central role in (i) adjusting the importance of terms to improve lexical matching and (ii) expanding additional terms to support semantic matching. Furthermore, our co-training strategy trains the dual encoder effectively and avoids unnecessary intervention in training each other. Experimental results on several benchmarks show that SpaDE outperforms ex
    
[^3]: 具有可证明的一致性和公平保证的推荐系统张量补全

    Tensor Completion with Provable Consistency and Fairness Guarantees for Recommender Systems. (arXiv:2204.01815v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2204.01815](http://arxiv.org/abs/2204.01815)

    本文介绍了一种新的一致性方法来解决矩阵和张量补全问题，在推荐系统应用中，我们证明了通过保留单位比例和一致性两个约束条件可以实现解的存在性与唯一性。

    

    我们引入了一种新的基于一致性的方法来定义和解决非负/正矩阵和张量补全问题。该框架的新颖之处在于，我们不是人为地将问题形式化为任意优化问题，例如，最小化一个结构量，如秩或范数，而是展示了一个单一的属性/约束：保留单位比例一致性，保证了解的存在，并在相对较弱的支持假设下保证了解的唯一性。该框架和解算法也直接推广到任意维度的张量中，同时保持了固定维度 d 的问题规模的线性计算复杂性。在推荐系统应用中，我们证明了两个合理的性质，这些性质应该适用于任何 RS 问题的解，足以允许在我们的框架内建立唯一性保证。关键理论贡献是展示了这些约束下解的存在性与唯一性。

    We introduce a new consistency-based approach for defining and solving nonnegative/positive matrix and tensor completion problems. The novelty of the framework is that instead of artificially making the problem well-posed in the form of an application-arbitrary optimization problem, e.g., minimizing a bulk structural measure such as rank or norm, we show that a single property/constraint: preserving unit-scale consistency, guarantees the existence of both a solution and, under relatively weak support assumptions, uniqueness. The framework and solution algorithms also generalize directly to tensors of arbitrary dimensions while maintaining computational complexity that is linear in problem size for fixed dimension d. In the context of recommender system (RS) applications, we prove that two reasonable properties that should be expected to hold for any solution to the RS problem are sufficient to permit uniqueness guarantees to be established within our framework. Key theoretical contribu
    

