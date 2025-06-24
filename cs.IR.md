# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [No more optimization rules: LLM-enabled policy-based multi-modal query optimizer (version 1)](https://arxiv.org/abs/2403.13597) | LLM启用了基于策略的多模查询优化器，摆脱了传统的基于规则的优化方法，为查询优化带来全新的可能性。 |
| [^2] | [Leveraging Foundation Models for Content-Based Medical Image Retrieval in Radiology](https://arxiv.org/abs/2403.06567) | 基于内容的医学图像检索中，利用基础模型作为特征提取器，无需微调即可取得与专门模型竞争的性能，尤其在检索病理特征方面具有较大困难。 |

# 详细

[^1]: 不再有优化规则: 基于LLM的基于策略的多模查询优化器（版本1）

    No more optimization rules: LLM-enabled policy-based multi-modal query optimizer (version 1)

    [https://arxiv.org/abs/2403.13597](https://arxiv.org/abs/2403.13597)

    LLM启用了基于策略的多模查询优化器，摆脱了传统的基于规则的优化方法，为查询优化带来全新的可能性。

    

    大语言模型(LLM)在机器学习和深度学习领域标志着一个重要时刻。最近，人们研究了LLM在查询规划中的能力，包括单模和多模查询。然而，对于LLM的查询优化能力还没有相关研究。作为显著影响查询计划执行性能的关键步骤，不应错过这种分析和尝试。另一方面，现有的查询优化器通常是基于规则或基于规则+基于成本的，即它们依赖于人工创建的规则来完成查询计划重写/转换。鉴于现代优化器包括数百至数千条规则，按照类似方式设计一个多模查询优化器将耗费大量时间，因为我们将不得不列举尽可能多的多模优化规则，而这并没有。

    arXiv:2403.13597v1 Announce Type: cross  Abstract: Large language model (LLM) has marked a pivotal moment in the field of machine learning and deep learning. Recently its capability for query planning has been investigated, including both single-modal and multi-modal queries. However, there is no work on the query optimization capability of LLM. As a critical (or could even be the most important) step that significantly impacts the execution performance of the query plan, such analysis and attempts should not be missed. From another aspect, existing query optimizers are usually rule-based or rule-based + cost-based, i.e., they are dependent on manually created rules to complete the query plan rewrite/transformation. Given the fact that modern optimizers include hundreds to thousands of rules, designing a multi-modal query optimizer following a similar way is significantly time-consuming since we will have to enumerate as many multi-modal optimization rules as possible, which has not be
    
[^2]: 利用基础模型进行放射学中基于内容的医学图像检索

    Leveraging Foundation Models for Content-Based Medical Image Retrieval in Radiology

    [https://arxiv.org/abs/2403.06567](https://arxiv.org/abs/2403.06567)

    基于内容的医学图像检索中，利用基础模型作为特征提取器，无需微调即可取得与专门模型竞争的性能，尤其在检索病理特征方面具有较大困难。

    

    Content-based image retrieval（CBIR）有望显著改善放射学中的诊断辅助和医学研究。我们提出利用视觉基础模型作为强大且多功能的现成特征提取器，用于基于内容的医学图像检索。通过在涵盖四种模态和161种病理学的160万张2D放射图像的全面数据集上对这些模型进行基准测试，我们发现弱监督模型表现优异，P@1可达0.594。这种性能不仅与专门化模型竞争，而且无需进行微调。我们的分析进一步探讨了检索病理学与解剖结构的挑战，表明准确检索病理特征更具挑战性。

    arXiv:2403.06567v1 Announce Type: cross  Abstract: Content-based image retrieval (CBIR) has the potential to significantly improve diagnostic aid and medical research in radiology. Current CBIR systems face limitations due to their specialization to certain pathologies, limiting their utility. In response, we propose using vision foundation models as powerful and versatile off-the-shelf feature extractors for content-based medical image retrieval. By benchmarking these models on a comprehensive dataset of 1.6 million 2D radiological images spanning four modalities and 161 pathologies, we identify weakly-supervised models as superior, achieving a P@1 of up to 0.594. This performance not only competes with a specialized model but does so without the need for fine-tuning. Our analysis further explores the challenges in retrieving pathological versus anatomical structures, indicating that accurate retrieval of pathological features presents greater difficulty. Despite these challenges, our
    

