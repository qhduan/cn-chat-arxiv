# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model](https://arxiv.org/abs/2606.26373) | 本文提出一种混合隐私保护语义搜索方法，通过SVD截断和秘密正交变换保护文档集合，利用CKKS同态加密保护查询，在受限威胁模型下平衡了安全性与效率。 |
| [^2] | [Towards Recursive Self-Evolving Agentic Literature Retrieval](https://arxiv.org/abs/2605.14306) | 提出了一种递归自进化智能文献检索系统PaSaMaster，通过迭代意图分析、基于证据的无幻觉排序和成本高效的规划-检索分离，在38个学科中实现16.5倍性能提升。 |
| [^3] | [Assortment Planning with Sponsored Products](https://arxiv.org/abs/2402.06158) | 本研究主要关注零售中带有赞助产品的品类规划挑战并将其建模为组合优化任务，以实现在考虑赞助产品的情况下优化预期收入的目的。 |
| [^4] | [Algorithmic neutrality.](http://arxiv.org/abs/2303.05103) | 研究算法中立性以及与算法偏见的关系，以搜索引擎为案例研究，得出搜索中立性是不可能的结论。 |

# 详细

[^1]: 混合隐私感知语义搜索：受限威胁模型下基于SVD截断文档几何与CKKS加密查询重排序

    Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model

    [https://arxiv.org/abs/2606.26373](https://arxiv.org/abs/2606.26373)

    本文提出一种混合隐私保护语义搜索方法，通过SVD截断和秘密正交变换保护文档集合，利用CKKS同态加密保护查询，在受限威胁模型下平衡了安全性与效率。

    

    arXiv:2606.26373v1 公告类型：交叉 摘要：稠密嵌入为语义搜索和检索增强生成提供了强大支持，但嵌入反转攻击可以从向量中重建源文本：当向量数据库泄露时，其背后的文档也会随之泄露。教科书式的防御措施是极端方案——对整个搜索进行同态加密是可靠的，但在百万级文档规模下速度过慢，而隐私噪声在提供保护之前就已严重降低排序质量。我们研究了一条中间路径，利用静态集合与动态查询之间的不对称性。集合通过几何方式保护：每个向量被截断到低维SVD子空间，并通过仅由所有者知道的秘密正交变换进行旋转。查询通过密码学方式保护：在CKKS同态加密下进行重排序，因此诚实但好奇的服务器永远无法看到查询或分数。CKKS参数来自一个小型离线基准测试。我们证明了重构的下界紧致性。

    arXiv:2606.26373v1 Announce Type: cross  Abstract: Dense embeddings power semantic search and retrieval-augmented generation, but embedding-inversion attacks can reconstruct source text from a vector: when a vector database leaks, the documents behind it leak too. The textbook defences are extremes - encrypting the whole search homomorphically is sound but too slow at million-document scale, while privacy noise degrades ranking long before it protects. We study a middle path exploiting the asymmetry between the static collection and the dynamic query. The collection is protected geometrically: each vector is truncated onto a lower-dimensional SVD subspace and rotated by a secret orthogonal transform known only to the owner. The query is protected cryptographically: it is reranked under CKKS homomorphic encryption, so an honest-but-curious server never sees the query or the scores. CKKS parameters come from a small offline benchmark.   We prove a tight lower bound on the reconstruction 
    
[^2]: 迈向递归自进化型智能文献检索系统

    Towards Recursive Self-Evolving Agentic Literature Retrieval

    [https://arxiv.org/abs/2605.14306](https://arxiv.org/abs/2605.14306)

    提出了一种递归自进化智能文献检索系统PaSaMaster，通过迭代意图分析、基于证据的无幻觉排序和成本高效的规划-检索分离，在38个学科中实现16.5倍性能提升。

    

    arXiv:2605.14306v2 公告类型：替换 摘要：科学文献检索必须在理解复杂搜索意图的同时保持源文档的真实性。传统的基于关键词和嵌入的系统能够返回真实来源，但会遗漏细微的意图，而大型语言模型能够捕捉更丰富的意图，但可能编造引用。我们引入了PaSaMaster，一种递归自进化型智能文献检索系统，它迭代地分析意图、检索经过验证的论文，并根据基于证据的相关性分数对其进行排序。PaSaMaster结合了自进化检索（根据随时间排序的证据优化搜索意图）、无幻觉排序（基于经过验证的论文而非生成的引用）以及成本高效的规划-检索分离（将前沿大语言模型保留用于意图理解，同时将检索和评分任务委托给轻量级模型和定制语料库）。在PaSaMaster-Bench涵盖的38个学科中，PaSaMaster实现了16.5倍的性能提升。

    arXiv:2605.14306v2 Announce Type: replace  Abstract: Scientific literature retrieval must understand complex search intents while preserving source authenticity. Traditional keyword and embedding-based systems return authentic sources but miss nuanced intents, whereas large language models capture richer intents but may fabricate citations. We introduce PaSaMaster, a Recursive Self-Evolving agentic literature retrieval system that iteratively analyzes intent, retrieves verified papers and ranks them with evidence-grounded relevance scores. PaSaMaster combines self-evolving retrieval that refines search intent from ranked evidence over time, hallucination-free ranking over verified papers rather than generated citations, and cost-efficient planning--retrieval separation that reserves frontier LLMs for intent understanding while delegating retrieval and scoring to lightweight models and customized corpora. Across 38 disciplines in PaSaMaster-Bench, PaSaMaster achieves a 16.5$\times$ high
    
[^3]: 带有赞助产品的品类规划

    Assortment Planning with Sponsored Products

    [https://arxiv.org/abs/2402.06158](https://arxiv.org/abs/2402.06158)

    本研究主要关注零售中带有赞助产品的品类规划挑战并将其建模为组合优化任务，以实现在考虑赞助产品的情况下优化预期收入的目的。

    

    在零售行业快速发展的背景下，品类规划对于企业的成功起着至关重要的作用。随着赞助产品在在线市场的日益突出地位，零售商在有效管理产品品类方面面临新的挑战。值得注意的是，以前的品类规划研究大多忽视了赞助产品的存在及其对整体推荐效果可能产生的影响。相反，他们通常简化地假设所有产品都是有机产品或非赞助产品。这个研究空白突显了在赞助产品存在的情况下更深入探讨品类规划挑战的必要性。我们将在存在赞助产品的情况下将品类规划问题建模为组合优化任务。最终目标是计算出一种最优的品类规划方案，既能优化预期收入，又能考虑到赞助产品的存在。

    In the rapidly evolving landscape of retail, assortment planning plays a crucial role in determining the success of a business. With the rise of sponsored products and their increasing prominence in online marketplaces, retailers face new challenges in effectively managing their product assortment in the presence of sponsored products. Remarkably, previous research in assortment planning largely overlooks the existence of sponsored products and their potential impact on overall recommendation effectiveness. Instead, they commonly make the simplifying assumption that all products are either organic or non-sponsored. This research gap underscores the necessity for a more thorough investigation of the assortment planning challenge when sponsored products are in play. We formulate the assortment planning problem in the presence of sponsored products as a combinatorial optimization task. The ultimate objective is to compute an assortment plan that optimizes expected revenue while considerin
    
[^4]: 算法中立性

    Algorithmic neutrality. (arXiv:2303.05103v2 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2303.05103](http://arxiv.org/abs/2303.05103)

    研究算法中立性以及与算法偏见的关系，以搜索引擎为案例研究，得出搜索中立性是不可能的结论。

    

    偏见影响着越来越多掌控我们生活的算法。预测性警务系统错误地高估有色人种社区的犯罪率；招聘算法削弱了合格的女性候选人的机会；人脸识别软件难以识别黑皮肤的面部。算法偏见已经受到了重视，相比之下，算法中立性却基本被忽视了。算法中立性是我的研究主题。我提出了三个问题。算法中立性是什么？算法中立性是否可能？当我们考虑算法中立性时，我们可以从算法偏见中学到什么？为了具体回答这些问题，我选择了一个案例研究：搜索引擎。借鉴关于科学中立性的研究，我认为只有当搜索引擎的排名不受某些价值观的影响时，搜索引擎才是中立的，比如政治意识形态或搜索引擎运营商的经济利益。我认为搜索中立性是不可能的。

    Bias infects the algorithms that wield increasing control over our lives. Predictive policing systems overestimate crime in communities of color; hiring algorithms dock qualified female candidates; and facial recognition software struggles to recognize dark-skinned faces. Algorithmic bias has received significant attention. Algorithmic neutrality, in contrast, has been largely neglected. Algorithmic neutrality is my topic. I take up three questions. What is algorithmic neutrality? Is algorithmic neutrality possible? When we have algorithmic neutrality in mind, what can we learn about algorithmic bias? To answer these questions in concrete terms, I work with a case study: search engines. Drawing on work about neutrality in science, I say that a search engine is neutral only if certain values -- like political ideologies or the financial interests of the search engine operator -- play no role in how the search engine ranks pages. Search neutrality, I argue, is impossible. Its impossibili
    

