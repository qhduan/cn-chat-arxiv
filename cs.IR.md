# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provider Fairness and Beyond-Accuracy Trade-offs in Recommender Systems.](http://arxiv.org/abs/2309.04250) | 本文研究了推荐系统中的提供商公平性问题以及相应的超越准确性考虑。通过引入后处理重新排序模型，我们在维护用户相关性和推荐质量的同时，优先考虑了提供商公平性。针对多个数据集进行的评估显示该模型在推荐质量各个方面有积极影响。 |
| [^2] | [Offline Recommender System Evaluation under Unobserved Confounding.](http://arxiv.org/abs/2309.04222) | 本论文讨论了在存在潜在混淆因素的情况下进行离线推荐系统评估的问题，并特别关注推荐系统用例。通过对基于策略的估计器进行研究，我们描述了由混淆因素引起的统计偏差。 |
| [^3] | [Receiving an algorithmic recommendation based on documentary filmmaking techniques.](http://arxiv.org/abs/2309.04184) | 本研究分析了基于纪录片制作技术的算法推荐在T{\"e}nk平台上的接受情况，通过构建一组元数据来描述纪录片制作设备的多样性，探讨平台电影爱好者如何理解和接受类似纪录片制作设备的个性化推荐。讨论了这个概念验证的贡献和局限，并提出了提升纪录片仪器化媒介的思考方向。 |
| [^4] | [A Long-Tail Friendly Representation Framework for Artist and Music Similarity.](http://arxiv.org/abs/2309.04182) | 本论文提出了一种长尾友好的表示框架，利用神经网络模型艺术家和音乐之间的相似关系，改善了传统方法在稀疏关系中的表示性能。 |
| [^5] | [PRISTA-Net: Deep Iterative Shrinkage Thresholding Network for Coded Diffraction Patterns Phase Retrieval.](http://arxiv.org/abs/2309.04171) | PRISTA-Net是一个基于深度迭代缩减阈值算法的网络，通过学习非线性变换和注意机制来处理相位恢复问题，并使用快速傅里叶变换和基于对数的损失函数来提高性能。 |
| [^6] | [Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation.](http://arxiv.org/abs/2309.03518) | 本研究提出了一种用于推荐系统的新型紧凑嵌入框架，该框架通过正则化修剪的方式在资源受限的环境中实现了更高的内存效率，从而提供了高准确度的推荐。 |
| [^7] | [TensorBank:Tensor Lakehouse for Foundation Model Training.](http://arxiv.org/abs/2309.02094) | TensorBank是一个基于Tensor的湖仓库，能够以高速从云对象存储流式传输张量到GPU内存，并通过使用分层统计指标进行查询加速。 |
| [^8] | [RecFusion: A Binomial Diffusion Process for 1D Data for Recommendation.](http://arxiv.org/abs/2306.08947) | 本文提出了 RecFusion，一种特定针对1D和/或二进制设置的推荐模型方法，其利用了二项式扩散过程对二元用户-项目交互进行显式建模，并在核心推荐设置和最常见的数据集上接近复杂的VAE基线的表现。 |
| [^9] | [STIXnet: A Novel and Modular Solution for Extracting All STIX Objects in CTI Reports.](http://arxiv.org/abs/2303.09999) | 提出了一种名为STIXnet的解决方案，可以自动提取CTI报告中所有的STIX实体和关系。 |

# 详细

[^1]: 提供商公平性与推荐系统中超越准确性的权衡研究

    Provider Fairness and Beyond-Accuracy Trade-offs in Recommender Systems. (arXiv:2309.04250v1 [cs.IR])

    [http://arxiv.org/abs/2309.04250](http://arxiv.org/abs/2309.04250)

    本文研究了推荐系统中的提供商公平性问题以及相应的超越准确性考虑。通过引入后处理重新排序模型，我们在维护用户相关性和推荐质量的同时，优先考虑了提供商公平性。针对多个数据集进行的评估显示该模型在推荐质量各个方面有积极影响。

    

    推荐系统在改善用户在线体验的同时，也引发了对潜在的提供商公平性问题的关注。这些系统可能会不经意地偏爱热门物品，从而使较不流行的物品边缘化，并妥协了提供商的公平性。虽然以前的研究已经意识到提供商公平性问题，但对这些偏见如何影响推荐系统的超越准确性方面（如多样性、新颖性、覆盖率和偶然性）的调查却不够重视。在本文中，我们通过引入一个简单而有效的后处理重新排序模型，旨在优先考虑提供商公平性的同时，保持用户相关性和推荐质量，来填补这一空白。然后，我们对模型在多个数据集上对推荐质量的各个方面的影响进行深入评估。具体而言，我们将后处理算法应用于四个不同领域数据集上的四个独立的推荐模型，进行评估。

    Recommender systems, while transformative in online user experiences, have raised concerns over potential provider-side fairness issues. These systems may inadvertently favor popular items, thereby marginalizing less popular ones and compromising provider fairness. While previous research has recognized provider-side fairness issues, the investigation into how these biases affect beyond-accuracy aspects of recommendation systems - such as diversity, novelty, coverage, and serendipity - has been less emphasized. In this paper, we address this gap by introducing a simple yet effective post-processing re-ranking model that prioritizes provider fairness, while simultaneously maintaining user relevance and recommendation quality. We then conduct an in-depth evaluation of the model's impact on various aspects of recommendation quality across multiple datasets. Specifically, we apply the post-processing algorithm to four distinct recommendation models across four varied domain datasets, asses
    
[^2]: 未观察到潜在混淆因素下的离线推荐系统评估

    Offline Recommender System Evaluation under Unobserved Confounding. (arXiv:2309.04222v1 [cs.LG])

    [http://arxiv.org/abs/2309.04222](http://arxiv.org/abs/2309.04222)

    本论文讨论了在存在潜在混淆因素的情况下进行离线推荐系统评估的问题，并特别关注推荐系统用例。通过对基于策略的估计器进行研究，我们描述了由混淆因素引起的统计偏差。

    

    离线政策估计方法(OPE)允许我们从记录的数据中学习和评估决策策略，使它们成为离线评估推荐系统的吸引人选择。最近的一些作品报道了成功采用OPE方法的情况。这项工作的一个重要假设是不存在未观察到的混淆因素：在数据收集时影响行动和奖励的随机变量。由于数据收集策略通常在从业者的控制之下，因此很少明确地提及无混淆假设，并且现有文献中很少处理其违规问题。这项工作旨在强调在存在未观察到的混淆因素的情况下进行离线策略估计时出现的问题，特别关注推荐系统的用例。我们专注于基于策略的估计器，其中日志倾向是从记录数据中学习的。我们对由于混淆因素引起的统计偏差进行了描述。

    Off-Policy Estimation (OPE) methods allow us to learn and evaluate decision-making policies from logged data. This makes them an attractive choice for the offline evaluation of recommender systems, and several recent works have reported successful adoption of OPE methods to this end. An important assumption that makes this work is the absence of unobserved confounders: random variables that influence both actions and rewards at data collection time. Because the data collection policy is typically under the practitioner's control, the unconfoundedness assumption is often left implicit, and its violations are rarely dealt with in the existing literature.  This work aims to highlight the problems that arise when performing off-policy estimation in the presence of unobserved confounders, specifically focusing on a recommendation use-case. We focus on policy-based estimators, where the logging propensities are learned from logged data. We characterise the statistical bias that arises due to
    
[^3]: 基于纪录片制作技术的算法推荐的接受研究

    Receiving an algorithmic recommendation based on documentary filmmaking techniques. (arXiv:2309.04184v1 [cs.IR])

    [http://arxiv.org/abs/2309.04184](http://arxiv.org/abs/2309.04184)

    本研究分析了基于纪录片制作技术的算法推荐在T{\"e}nk平台上的接受情况，通过构建一组元数据来描述纪录片制作设备的多样性，探讨平台电影爱好者如何理解和接受类似纪录片制作设备的个性化推荐。讨论了这个概念验证的贡献和局限，并提出了提升纪录片仪器化媒介的思考方向。

    

    本文分析了T{\"e}nk平台电影观众对纪录片的新颖算法推荐的接受情况。为了提出一种替代基于主题分类、导演或制作时期的推荐方法，在这个实验框架内，我们构建了一组元数据，以描述“纪录片制作设备”的丰富多样性。研究的目标是调查平台的电影爱好者在个性化推荐的情况下如何理解和接受有相似纪录片制作设备的4部纪录片。最后，讨论了这个概念验证的贡献和局限，以勾勒出提升纪录片仪器化媒介的思考方向。

    This article analyzes the reception of a novel algorithmic recommendation of documentary films by a panel of moviegoers of the T{\"e}nk platform. In order to propose an alternative to recommendations based on a thematic classification, the director or the production period, a set of metadata has been elaborated within the framework of this experimentation in order to characterize the great variety of ``documentary filmmaking dispositifs'' . The goal is to investigate the different ways in which the platform's film lovers appropriate a personalized recommendation of 4 documentaries with similar or similar filmmaking dispositifs. To conclude, the contributions and limits of this proof of concept are discussed in order to sketch out avenues of reflection for improving the instrumented mediation of documentary films.
    
[^4]: 一种针对艺术家和音乐相似性的长尾友好表示框架

    A Long-Tail Friendly Representation Framework for Artist and Music Similarity. (arXiv:2309.04182v1 [cs.SD])

    [http://arxiv.org/abs/2309.04182](http://arxiv.org/abs/2309.04182)

    本论文提出了一种长尾友好的表示框架，利用神经网络模型艺术家和音乐之间的相似关系，改善了传统方法在稀疏关系中的表示性能。

    

    在音乐检索和推荐中，研究艺术家和音乐之间的相似性是至关重要的，解决长尾现象的挑战日益重要。本文提出了一种利用神经网络建模相似关系的长尾友好表示框架(LTFRF)。我们的方法将音乐、用户、元数据和关系数据整合到一个统一的度量学习框架中，利用元一致性关系作为正则项，引入多关系损失。与图神经网络(GNN)相比，我们提出的框架改善了长尾场景中稀疏的艺术家和音乐之间的关系表示性能。我们在AllMusic数据集上进行了实验和分析，结果表明我们的框架提供了艺术家和音乐表示的有利泛化效果。具体而言，在类似艺术家/音乐推荐任务上，t

    The investigation of the similarity between artists and music is crucial in music retrieval and recommendation, and addressing the challenge of the long-tail phenomenon is increasingly important. This paper proposes a Long-Tail Friendly Representation Framework (LTFRF) that utilizes neural networks to model the similarity relationship. Our approach integrates music, user, metadata, and relationship data into a unified metric learning framework, and employs a meta-consistency relationship as a regular term to introduce the Multi-Relationship Loss. Compared to the Graph Neural Network (GNN), our proposed framework improves the representation performance in long-tail scenarios, which are characterized by sparse relationships between artists and music. We conduct experiments and analysis on the AllMusic dataset, and the results demonstrate that our framework provides a favorable generalization of artist and music representation. Specifically, on similar artist/music recommendation tasks, t
    
[^5]: PRISTA-Net:用于编码衍射图模式相位恢复的深度迭代缩减阈值网络

    PRISTA-Net: Deep Iterative Shrinkage Thresholding Network for Coded Diffraction Patterns Phase Retrieval. (arXiv:2309.04171v1 [cs.CV])

    [http://arxiv.org/abs/2309.04171](http://arxiv.org/abs/2309.04171)

    PRISTA-Net是一个基于深度迭代缩减阈值算法的网络，通过学习非线性变换和注意机制来处理相位恢复问题，并使用快速傅里叶变换和基于对数的损失函数来提高性能。

    

    相位恢复（PR）问题涉及从有限的幅度测量数据中恢复未知图像，是计算成像和图像处理中具有挑战性的非线性逆问题。然而，许多PR方法基于缺乏可解释性和计算复杂性般配和调参需求的黑盒网络模型。为了解决这个问题，我们开发了PRISTA-Net，这是一个基于一阶迭代缩减阈值算法（ISTA）的深度展开网络（DUN）。该网络利用可学习的非线性变换来处理稀疏先验中与近端点映射子问题有关的相位信息，还使用注意机制来聚焦包含图像边缘、纹理和结构的相位信息。此外，还使用快速傅里叶变换（FFT）来学习全局特征以增强局部信息，并通过设计的基于对数的损失函数实现了显著的改善。

    The problem of phase retrieval (PR) involves recovering an unknown image from limited amplitude measurement data and is a challenge nonlinear inverse problem in computational imaging and image processing. However, many of the PR methods are based on black-box network models that lack interpretability and plug-and-play (PnP) frameworks that are computationally complex and require careful parameter tuning. To address this, we have developed PRISTA-Net, a deep unfolding network (DUN) based on the first-order iterative shrinkage thresholding algorithm (ISTA). This network utilizes a learnable nonlinear transformation to address the proximal-point mapping sub-problem associated with the sparse priors, and an attention mechanism to focus on phase information containing image edges, textures, and structures. Additionally, the fast Fourier transform (FFT) is used to learn global features to enhance local information, and the designed logarithmic-based loss function leads to significant improve
    
[^6]: 通过正则化修剪来学习紧凑的组合嵌入以用于推荐

    Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation. (arXiv:2309.03518v1 [cs.IR])

    [http://arxiv.org/abs/2309.03518](http://arxiv.org/abs/2309.03518)

    本研究提出了一种用于推荐系统的新型紧凑嵌入框架，该框架通过正则化修剪的方式在资源受限的环境中实现了更高的内存效率，从而提供了高准确度的推荐。

    

    潜在因素模型是当代推荐系统的主要支柱，由于它们的性能优势，在这些模型中，每个实体（通常是用户/物品）需要用一个固定维度（例如128）的唯一向量嵌入来表示。由于电子商务网站上用户和物品的数量巨大，嵌入表格可以说是推荐系统中最不节省内存的组件。对于任何希望能够有效地按比例扩展到不断增长的用户/物品数量或在资源受限环境中仍然适用的轻量级推荐系统，现有的解决方案要么通过哈希减少所需的嵌入数量，要么通过稀疏化完整的嵌入表格以关闭选定的嵌入维度。然而，由于哈希冲突或嵌入过于稀疏，尤其是在适应更紧凑的内存预算时，这些轻量级推荐器不可避免地会牺牲其准确性。因此，我们提出了一种新颖的紧凑嵌入框架用于推荐系统，称为Compos。

    Latent factor models are the dominant backbones of contemporary recommender systems (RSs) given their performance advantages, where a unique vector embedding with a fixed dimensionality (e.g., 128) is required to represent each entity (commonly a user/item). Due to the large number of users and items on e-commerce sites, the embedding table is arguably the least memory-efficient component of RSs. For any lightweight recommender that aims to efficiently scale with the growing size of users/items or to remain applicable in resource-constrained settings, existing solutions either reduce the number of embeddings needed via hashing, or sparsify the full embedding table to switch off selected embedding dimensions. However, as hash collision arises or embeddings become overly sparse, especially when adapting to a tighter memory budget, those lightweight recommenders inevitably have to compromise their accuracy. To this end, we propose a novel compact embedding framework for RSs, namely Compos
    
[^7]: TensorBank: 基于Tensor的湖仓库用于基础模型训练

    TensorBank:Tensor Lakehouse for Foundation Model Training. (arXiv:2309.02094v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.02094](http://arxiv.org/abs/2309.02094)

    TensorBank是一个基于Tensor的湖仓库，能够以高速从云对象存储流式传输张量到GPU内存，并通过使用分层统计指标进行查询加速。

    

    随着基础模型在自然语言之外的领域的兴起，存储和流式处理高维数据成为基础模型训练的关键需求。在本文中，我们介绍了TensorBank，一个能够基于复杂关系查询从云对象存储（COS）流式传输张量到GPU内存的百亿级张量湖仓库。我们使用分层统计指标（HSI）来加速查询。我们的架构允许使用HTTP范围读取来直接访问块级别的张量。一旦在GPU内存中，数据可以使用PyTorch转换进行转换。我们提供了一个通用的PyTorch数据集类型，配有相应的数据集工厂，用于将关系查询和请求的转换作为一个实例进行翻译。通过使用HSI，可以跳过不相关的块，而无需读取它们，因为这些索引包含不同层次分辨率级别上内容的统计信息。这是一个基于开放标准的有主观观点的架构。

    Storing and streaming high dimensional data for foundation model training became a critical requirement with the rise of foundation models beyond natural language. In this paper we introduce TensorBank, a petabyte scale tensor lakehouse capable of streaming tensors from Cloud Object Store (COS) to GPU memory at wire speed based on complex relational queries. We use Hierarchical Statistical Indices (HSI) for query acceleration. Our architecture allows to directly address tensors on block level using HTTP range reads. Once in GPU memory, data can be transformed using PyTorch transforms. We provide a generic PyTorch dataset type with a corresponding dataset factory translating relational queries and requested transformations as an instance. By making use of the HSI, irrelevant blocks can be skipped without reading them as those indices contain statistics on their content at different hierarchical resolution levels. This is an opinionated architecture powered by open standards and making h
    
[^8]: RecFusion：基于二项式扩散过程的1D数据推荐模型

    RecFusion: A Binomial Diffusion Process for 1D Data for Recommendation. (arXiv:2306.08947v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.08947](http://arxiv.org/abs/2306.08947)

    本文提出了 RecFusion，一种特定针对1D和/或二进制设置的推荐模型方法，其利用了二项式扩散过程对二元用户-项目交互进行显式建模，并在核心推荐设置和最常见的数据集上接近复杂的VAE基线的表现。

    

    本文提出了RecFusion，这是一组用于推荐的扩散模型。不同于包含空间相关性的图像数据，常用于推荐的用户-项目交互矩阵缺乏用户和项目之间的空间关系。我们在一个一维向量上制定了扩散方法，并提出了二项式扩散，这个方法利用了伯努利过程显式地对二元用户-项目交互进行建模。我们展示了RecFusion在核心推荐设置（针对二进制非顺序反馈的前n项推荐）和最常见的数据集（MovieLens和Netflix）上接近于复杂的VAE基线的表现。我们提出的专门针对1D和/或二进制设置的扩散模型的意义超出了推荐系统，例如在医学领域中使用MRI和CT扫描。

    In this paper we propose RecFusion, which comprise a set of diffusion models for recommendation. Unlike image data which contain spatial correlations, a user-item interaction matrix, commonly utilized in recommendation, lacks spatial relationships between users and items. We formulate diffusion on a 1D vector and propose binomial diffusion, which explicitly models binary user-item interactions with a Bernoulli process. We show that RecFusion approaches the performance of complex VAE baselines on the core recommendation setting (top-n recommendation for binary non-sequential feedback) and the most common datasets (MovieLens and Netflix). Our proposed diffusion models that are specialized for 1D and/or binary setups have implications beyond recommendation systems, such as in the medical domain with MRI and CT scans.
    
[^9]: STIXnet: 一种从CTI报告中提取所有STIX对象的新型模块化解决方案

    STIXnet: A Novel and Modular Solution for Extracting All STIX Objects in CTI Reports. (arXiv:2303.09999v1 [cs.IR])

    [http://arxiv.org/abs/2303.09999](http://arxiv.org/abs/2303.09999)

    提出了一种名为STIXnet的解决方案，可以自动提取CTI报告中所有的STIX实体和关系。

    

    从网络威胁情报(CTI)报告中自动提取信息对于风险管理至关重要。本文提出了一种名为STIXnet的解决方案，通过使用自然语言处理（NLP）技术和交互式实体知识库（KB），可以自动提取CTI报告中所有的STIX实体和关系。

    The automatic extraction of information from Cyber Threat Intelligence (CTI) reports is crucial in risk management. The increased frequency of the publications of these reports has led researchers to develop new systems for automatically recovering different types of entities and relations from textual data. Most state-of-the-art models leverage Natural Language Processing (NLP) techniques, which perform greatly in extracting a few types of entities at a time but cannot detect heterogeneous data or their relations. Furthermore, several paradigms, such as STIX, have become de facto standards in the CTI community and dictate a formal categorization of different entities and relations to enable organizations to share data consistently. This paper presents STIXnet, the first solution for the automated extraction of all STIX entities and relationships in CTI reports. Through the use of NLP techniques and an interactive Knowledge Base (KB) of entities, our approach obtains F1 scores comparab
    

