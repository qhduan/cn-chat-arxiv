# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cash non-additive risk measures: horizon risk and generalized entropy.](http://arxiv.org/abs/2401.14443) | 该论文研究了在现金非可加风险度量背景下的时间跨度风险，并引入了一种基于广义Tsallis熵的风险度量，可以动态评估损失的风险性和利率不确定性。这项研究对于量化资本需求有着重要意义。 |
| [^2] | [Linking Representations with Multimodal Contrastive Learning.](http://arxiv.org/abs/2304.03464) | 本文提出了一种名为CLIPPINGS的多模态框架，用于记录链接。该框架利用深度学习和对比学习的方法，通过端到端训练对称的视觉和语言编码器，在度量空间中学习相近或不同类别的表示方法，用于多个应用场景，如构建全面的补充专利注册表和识别不同社交媒体平台上的个人。 |

# 详细

[^1]: 现金非可加风险度量：时间跨度风险和广义熵

    Cash non-additive risk measures: horizon risk and generalized entropy. (arXiv:2401.14443v1 [q-fin.RM])

    [http://arxiv.org/abs/2401.14443](http://arxiv.org/abs/2401.14443)

    该论文研究了在现金非可加风险度量背景下的时间跨度风险，并引入了一种基于广义Tsallis熵的风险度量，可以动态评估损失的风险性和利率不确定性。这项研究对于量化资本需求有着重要意义。

    

    在由BSDEs引起的现金非可加完全动态风险度量的背景下，研究了时间跨度风险。此外，我们引入了一种基于广义Tsallis熵的风险度量，可以动态评估考虑时间跨度风险和利率不确定性的损失的风险性。损失的新的q-熵风险度量可用作资本需求的量化。

    Horizon risk (see arXiv:2301.04971) is studied in the context of cash non-additive fully-dynamic risk measures induced by BSDEs. Furthermore, we introduce a risk measure based on generalized Tsallis entropy which can dynamically evaluate the riskiness of losses considering both horizon risk and interest rate uncertainty. The new q-entropic risk measure on losses can be used as a quantification of capital requirement.
    
[^2]: 用多模态对比学习连接表示

    Linking Representations with Multimodal Contrastive Learning. (arXiv:2304.03464v1 [cs.CV])

    [http://arxiv.org/abs/2304.03464](http://arxiv.org/abs/2304.03464)

    本文提出了一种名为CLIPPINGS的多模态框架，用于记录链接。该框架利用深度学习和对比学习的方法，通过端到端训练对称的视觉和语言编码器，在度量空间中学习相近或不同类别的表示方法，用于多个应用场景，如构建全面的补充专利注册表和识别不同社交媒体平台上的个人。

    

    许多应用需要将包含在各种文档数据集中的实例分组成类。最广泛使用的方法不使用深度学习，也不利用文档固有的多模态性质。值得注意的是，记录链接通常被概念化为字符串匹配问题。本研究开发了 CLIPPINGS，一种用于记录链接的多模态框架。CLIPPINGS 采用端到端训练对称的视觉和语言双编码器，通过对比语言-图像预训练进行对齐，学习一个度量空间，其中给定实例的汇总图像-文本表示靠近同一类中的表示，并远离不同类中的表示。在推理时，可以通过从离线示例嵌入索引中检索它们最近的邻居或聚类它们的表示来链接实例。本研究研究了两个具有挑战性的应用：通过将专利与其对应的监管文件链接来构建全面的补充专利注册表，以及在不同的社交媒体平台上识别个人。

    Many applications require grouping instances contained in diverse document datasets into classes. Most widely used methods do not employ deep learning and do not exploit the inherently multimodal nature of documents. Notably, record linkage is typically conceptualized as a string-matching problem. This study develops CLIPPINGS, (Contrastively Linking Pooled Pre-trained Embeddings), a multimodal framework for record linkage. CLIPPINGS employs end-to-end training of symmetric vision and language bi-encoders, aligned through contrastive language-image pre-training, to learn a metric space where the pooled image-text representation for a given instance is close to representations in the same class and distant from representations in different classes. At inference time, instances can be linked by retrieving their nearest neighbor from an offline exemplar embedding index or by clustering their representations. The study examines two challenging applications: constructing comprehensive suppl
    

