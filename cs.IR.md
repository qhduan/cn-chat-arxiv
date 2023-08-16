# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Investigation Toward The Economic Feasibility of Personalized Medicine For Healthcare Service Providers: The Case of Bladder Cancer.](http://arxiv.org/abs/2308.07924) | 本研究调查了个性化医疗的经济可行性，旨在平衡个体患者需求和经济决策之间的关系。通过将个性化视为一个连续的过程，提供了更大的灵活性。 |
| [^2] | [Synthesizing Political Zero-Shot Relation Classification via Codebook Knowledge, NLI, and ChatGPT.](http://arxiv.org/abs/2308.07876) | 该论文通过利用已建立的注释编码本的知识，探索零样本方法用于政治事件本体关系分类，并介绍一种基于自然语言推理的方法，名为ZSP。ZSP采用了一种树查询框架，提高了解释性、效率和对模式更改的适应性。在细粒度根代码分类上，ZSP的性能明显优于ChatGPT，F1得分提高了40%。 |
| [^3] | [Impression-Aware Recommender Systems.](http://arxiv.org/abs/2308.07857) | 基于印象的推荐系统利用印象数据源提升推荐质量，通过综述分类推荐系统、数据集和评估方法，揭示开放性问题和未来研究方向。 |
| [^4] | [Dynamic Embedding Size Search with Minimum Regret for Streaming Recommender System.](http://arxiv.org/abs/2308.07760) | 本文针对流式推荐系统中嵌入尺寸设置问题，将动态嵌入尺寸搜索建模为一个强盗问题，并从统计学角度分析和量化影响最佳嵌入尺寸的因素。 |
| [^5] | [Self-Supervised Dynamic Hypergraph Recommendation based on Hyper-Relational Knowledge Graph.](http://arxiv.org/abs/2308.07752) | SDK框架提出了一种在超关系知识图上进行自监督动态超图推荐的机制，以解决实体稀疏性、超关系事实建模困难和过度平滑等问题。 |
| [^6] | [SPM: Structured Pretraining and Matching Architectures for Relevance Modeling in Meituan Search.](http://arxiv.org/abs/2308.07711) | 本论文提出了一种用于在Meituan搜索中进行相关性建模的新颖两阶段预训练和匹配架构。 |
| [^7] | [Learning from All Sides: Diversified Positive Augmentation via Self-distillation in Recommendation.](http://arxiv.org/abs/2308.07629) | 本文介绍了一种新的模型无关的多样化自我蒸馏引导的正向增强方法（DivSPA），用于解决个性化推荐中数据稀疏问题。DivSPA通过多种检索策略收集高质量和多样化的正向候选物品，并通过自我蒸馏模块重新排名这些候选者，从而提高了推荐准确性和多样性。 |
| [^8] | [Delphic Costs and Benefits in Web Search: A utilitarian and historical analysis.](http://arxiv.org/abs/2308.07525) | 新框架概念化和操作化了搜索的全面用户体验，其中强调了搜索中的Delphic成本和利益，包括非货币成本和可能受损的利益。 |
| [^9] | [A Survey on Point-of-Interest Recommendations Leveraging Heterogeneous Data.](http://arxiv.org/abs/2308.07426) | 本文针对旅游领域的兴趣点推荐问题进行了调查研究，探讨了利用异构数据解决旅途中兴趣点推荐问题的潜力与挑战。 |
| [^10] | [Improving ICD-based semantic similarity by accounting for varying degrees of comorbidity.](http://arxiv.org/abs/2308.07359) | 本研究通过考虑记录合并疾病的比例项，改进了基于ICD码的语义相似度计算方法。 |
| [^11] | [Large Language Models for Information Retrieval: A Survey.](http://arxiv.org/abs/2308.07107) | 本综述将大型语言模型（LLMs）在信息检索中的发展进行了综述，探讨了其在捕捉上下文信号和语义细微之处方面的优势和挑战，以及与传统检索方法的结合的重要性。 |
| [^12] | [Data augmentation for recommender system: A semi-supervised approach using maximum margin matrix factorization.](http://arxiv.org/abs/2306.13050) | 本研究提出了一种基于最大边际矩阵分解的半监督方法来增广和细化协同过滤算法的评级预测。该方法利用自我训练来评估评分的置信度，并通过系统的数据增广策略来提高算法性能。 |
| [^13] | [Probe: Learning Users' Personalized Projection Bias in Intertemporal Bundle Choices.](http://arxiv.org/abs/2303.06016) | 本文提出了一种新的偏差嵌入式偏好模型——Probe，旨在解决用户在时间跨度的购物选择中的投影偏差和参照点效应，提高决策的有效性和个性化。 |

# 详细

[^1]: 探讨个性化医疗对医疗服务提供者的经济可行性：以膀胱癌为例

    Investigation Toward The Economic Feasibility of Personalized Medicine For Healthcare Service Providers: The Case of Bladder Cancer. (arXiv:2308.07924v1 [cs.IR])

    [http://arxiv.org/abs/2308.07924](http://arxiv.org/abs/2308.07924)

    本研究调查了个性化医疗的经济可行性，旨在平衡个体患者需求和经济决策之间的关系。通过将个性化视为一个连续的过程，提供了更大的灵活性。

    

    在当今复杂的医疗环境中，医疗服务提供者在平衡优化患者护理和复杂经济动态方面面临巨大挑战。个性化医疗基于临床上有前景的个性化药物治疗的出现旨在改变医学。尽管个性化医疗有着巨大的潜力提高治疗效果，但在资源有限的医疗服务提供者中进行整合面临巨大挑战。本研究旨在调查实施个性化医疗的经济可行性。其核心目标是在满足个体患者需求和做出经济可行的决策之间取得平衡。与传统的二元个性化治疗方法不同，我们提出了一个更细致的视角，将个性化视为一个连续的过程。这种方法能够在决策和资源配置中提供更大的灵活性。

    In today's complex healthcare landscape, the pursuit of delivering optimal patient care while navigating intricate economic dynamics poses a significant challenge for healthcare service providers (HSPs). In this already complex dynamics, the emergence of clinically promising personalized medicine based treatment aims to revolutionize medicine. While personalized medicine holds tremendous potential for enhancing therapeutic outcomes, its integration within resource-constrained HSPs presents formidable challenges. In this study, we investigate the economic feasibility of implementing personalized medicine. The central objective is to strike a balance between catering to individual patient needs and making economically viable decisions. Unlike conventional binary approaches to personalized treatment, we propose a more nuanced perspective by treating personalization as a spectrum. This approach allows for greater flexibility in decision-making and resource allocation. To this end, we propo
    
[^2]: 通过编码本知识、自然语言推理和ChatGPT来合成政治零样本关系分类

    Synthesizing Political Zero-Shot Relation Classification via Codebook Knowledge, NLI, and ChatGPT. (arXiv:2308.07876v1 [cs.CL])

    [http://arxiv.org/abs/2308.07876](http://arxiv.org/abs/2308.07876)

    该论文通过利用已建立的注释编码本的知识，探索零样本方法用于政治事件本体关系分类，并介绍一种基于自然语言推理的方法，名为ZSP。ZSP采用了一种树查询框架，提高了解释性、效率和对模式更改的适应性。在细粒度根代码分类上，ZSP的性能明显优于ChatGPT，F1得分提高了40%。

    

    最近的事件编码的监督模型在性能方面远远超过模式匹配方法。然而，它们仅仅依赖于新的注释，忽视了专家数据库中的大量知识，限制了它们在细粒度分类中的适用性。为了解决这些限制，我们通过利用已建立的注释编码本的知识，探索零样本方法用于政治事件本体关系分类。我们的研究涵盖了ChatGPT和一种新颖的基于自然语言推理的方法，名为ZSP。ZSP采用了一种树查询框架，将任务分解为上下文、语态和类别消歧的不同层次。该框架提高了解释性、效率和对模式更改的适应性。通过在我们新策划的数据集上进行大量实验，我们指出了ChatGPT中的不稳定性问题，并突出了ZSP的卓越性能。ZSP在细粒度根代码分类的F1得分上取得了令人印象深刻的提高40%。

    Recent supervised models for event coding vastly outperform pattern-matching methods. However, their reliance solely on new annotations disregards the vast knowledge within expert databases, hindering their applicability to fine-grained classification. To address these limitations, we explore zero-shot approaches for political event ontology relation classification, by leveraging knowledge from established annotation codebooks. Our study encompasses both ChatGPT and a novel natural language inference (NLI) based approach named ZSP. ZSP adopts a tree-query framework that deconstructs the task into context, modality, and class disambiguation levels. This framework improves interpretability, efficiency, and adaptability to schema changes. By conducting extensive experiments on our newly curated datasets, we pinpoint the instability issues within ChatGPT and highlight the superior performance of ZSP. ZSP achieves an impressive 40% improvement in F1 score for fine-grained Rootcode classific
    
[^3]: 基于印象的推荐系统

    Impression-Aware Recommender Systems. (arXiv:2308.07857v1 [cs.IR])

    [http://arxiv.org/abs/2308.07857](http://arxiv.org/abs/2308.07857)

    基于印象的推荐系统利用印象数据源提升推荐质量，通过综述分类推荐系统、数据集和评估方法，揭示开放性问题和未来研究方向。

    

    新型数据源为改进推荐系统的质量带来了新的机遇。印象是一种包含过去推荐（展示的项目）和传统互动的新型数据源。研究人员可以利用印象来优化用户偏好并克服当前推荐系统研究中的限制。印象的相关性和兴趣度逐年增加，因此需要对这类推荐系统中相关工作进行综述。我们提出了一篇关于使用印象的推荐系统的系统文献综述，侧重于研究中的三个基本方面：推荐系统、数据集和评估方法。我们对使用印象的推荐系统的论文进行了三个分类，详细介绍了每篇综述论文，描述了具有印象的数据集，并分析了现有的评估方法。最后，我们提出了值得关注的开放性问题和未来的研究方向，强调了文献中缺失的方面。

    Novel data sources bring new opportunities to improve the quality of recommender systems. Impressions are a novel data source containing past recommendations (shown items) and traditional interactions. Researchers may use impressions to refine user preferences and overcome the current limitations in recommender systems research. The relevance and interest of impressions have increased over the years; hence, the need for a review of relevant work on this type of recommenders. We present a systematic literature review on recommender systems using impressions, focusing on three fundamental angles in research: recommenders, datasets, and evaluation methodologies. We provide three categorizations of papers describing recommenders using impressions, present each reviewed paper in detail, describe datasets with impressions, and analyze the existing evaluation methodologies. Lastly, we present open questions and future directions of interest, highlighting aspects missing in the literature that
    
[^4]: 流式推荐系统的最小后悔动态嵌入尺寸搜索

    Dynamic Embedding Size Search with Minimum Regret for Streaming Recommender System. (arXiv:2308.07760v1 [cs.IR])

    [http://arxiv.org/abs/2308.07760](http://arxiv.org/abs/2308.07760)

    本文针对流式推荐系统中嵌入尺寸设置问题，将动态嵌入尺寸搜索建模为一个强盗问题，并从统计学角度分析和量化影响最佳嵌入尺寸的因素。

    

    随着用户和物品数量的不断增加，传统的静态数据集上训练的推荐系统很难适应不断变化的环境。高吞吐量的数据要求模型及时更新以捕捉用户兴趣动态，这导致了流式推荐系统的出现。由于基于深度学习的推荐系统的普及，嵌入层广泛采用以低维向量表示用户、物品和其他特征的特性。然而，已经证明设置相同和静态的嵌入尺寸在推荐性能和内存成本方面是次优的，特别是对于流式推荐。为解决这个问题，我们首先重新思考了流式模型更新过程，并将动态嵌入尺寸搜索建模为一个强盗问题。然后，我们从统计学角度分析和量化影响最佳嵌入尺寸的因素。

    With the continuous increase of users and items, conventional recommender systems trained on static datasets can hardly adapt to changing environments. The high-throughput data requires the model to be updated in a timely manner for capturing the user interest dynamics, which leads to the emergence of streaming recommender systems. Due to the prevalence of deep learning-based recommender systems, the embedding layer is widely adopted to represent the characteristics of users, items, and other features in low-dimensional vectors. However, it has been proved that setting an identical and static embedding size is sub-optimal in terms of recommendation performance and memory cost, especially for streaming recommendations. To tackle this problem, we first rethink the streaming model update process and model the dynamic embedding size search as a bandit problem. Then, we analyze and quantify the factors that influence the optimal embedding sizes from the statistics perspective. Based on this
    
[^5]: 基于超关系知识图的自监督动态超图推荐

    Self-Supervised Dynamic Hypergraph Recommendation based on Hyper-Relational Knowledge Graph. (arXiv:2308.07752v1 [cs.IR])

    [http://arxiv.org/abs/2308.07752](http://arxiv.org/abs/2308.07752)

    SDK框架提出了一种在超关系知识图上进行自监督动态超图推荐的机制，以解决实体稀疏性、超关系事实建模困难和过度平滑等问题。

    

    知识图谱 (KG) 常被用作辅助信息，以增强协同信号并提高推荐质量。在知识感知推荐 (KGR) 的背景下，图神经网络 (GNN) 已成为对 KG 中的事实和语义信息进行建模的有前途的解决方案。然而，实体的长尾分布导致监督信号的稀疏性，这削弱了使用 KG 增强时的项目表示质量。此外，KG 的二元关系表示简化了超关系事实，使建模复杂的真实世界信息变得困难。此外，过度平滑的现象导致难以区分的表示和信息损失。为了解决这些挑战，我们提出了基于超关系知识图的自监督动态超图推荐 (SDK) 框架。该框架建立了一个跨视图超图自监督学习机制来增强 KG。

    Knowledge graphs (KGs) are commonly used as side information to enhance collaborative signals and improve recommendation quality. In the context of knowledge-aware recommendation (KGR), graph neural networks (GNNs) have emerged as promising solutions for modeling factual and semantic information in KGs. However, the long-tail distribution of entities leads to sparsity in supervision signals, which weakens the quality of item representation when utilizing KG enhancement. Additionally, the binary relation representation of KGs simplifies hyper-relational facts, making it challenging to model complex real-world information. Furthermore, the over-smoothing phenomenon results in indistinguishable representations and information loss. To address these challenges, we propose the SDK (Self-Supervised Dynamic Hypergraph Recommendation based on Hyper-Relational Knowledge Graph) framework. This framework establishes a cross-view hypergraph self-supervised learning mechanism for KG enhancement. Sp
    
[^6]: SPM: Meituan搜索中用于相关性建模的结构化预训练和匹配架构

    SPM: Structured Pretraining and Matching Architectures for Relevance Modeling in Meituan Search. (arXiv:2308.07711v1 [cs.IR])

    [http://arxiv.org/abs/2308.07711](http://arxiv.org/abs/2308.07711)

    本论文提出了一种用于在Meituan搜索中进行相关性建模的新颖两阶段预训练和匹配架构。

    

    在电商搜索中，查询和文档之间的相关性是满足用户体验的基本要求。与传统的电商平台不同，用户在美团等生活服务平台上进行搜索主要是为了产品供应商，这些供应商通常拥有丰富的结构化信息，例如名称、地址、类别、成千上万的产品。使用这些丰富的结构化内容进行搜索相关性建模具有挑战性，主要存在以下问题：（1）不同字段的结构化文档存在语言分布差异，无法直接采用预训练的语言模型方法（如BERT）。（2）不同字段通常具有不同的重要性，且长度差异很大，很难提取对相关性匹配有帮助的文档信息。为了解决这些问题，本文提出了一种新的两阶段预训练和匹配架构，用于丰富结构的相关性匹配。

    In e-commerce search, relevance between query and documents is an essential requirement for satisfying user experience. Different from traditional e-commerce platforms that offer products, users search on life service platforms such as Meituan mainly for product providers, which usually have abundant structured information, e.g. name, address, category, thousands of products. Modeling search relevance with these rich structured contents is challenging due to the following issues: (1) there is language distribution discrepancy among different fields of structured document, making it difficult to directly adopt off-the-shelf pretrained language model based methods like BERT. (2) different fields usually have different importance and their length vary greatly, making it difficult to extract document information helpful for relevance matching.  To tackle these issues, in this paper we propose a novel two-stage pretraining and matching architecture for relevance matching with rich structure
    
[^7]: 从各个方面学习：通过自我蒸馏实现多样化的正向增强在推荐中的应用

    Learning from All Sides: Diversified Positive Augmentation via Self-distillation in Recommendation. (arXiv:2308.07629v1 [cs.IR])

    [http://arxiv.org/abs/2308.07629](http://arxiv.org/abs/2308.07629)

    本文介绍了一种新的模型无关的多样化自我蒸馏引导的正向增强方法（DivSPA），用于解决个性化推荐中数据稀疏问题。DivSPA通过多种检索策略收集高质量和多样化的正向候选物品，并通过自我蒸馏模块重新排名这些候选者，从而提高了推荐准确性和多样性。

    

    个性化推荐依赖于用户的历史行为来提供用户感兴趣的物品，因此严重受到数据稀疏问题的困扰。一个强大的正向物品增强有助于解决稀疏问题，然而很少有工作能够同时考虑这些增强训练标签的准确性和多样性。在这项工作中，我们提出了一种新颖的模型无关的多样化自我蒸馏引导的正向增强（DivSPA），用于准确和多样化的正向物品增强。具体而言，DivSPA首先通过三种检索策略收集与用户的整体兴趣、短期意图和相似用户相对应的高质量和多样化的正向候选物品。接下来，进行自我蒸馏模块来重新检查和重新排名这些候选者作为最终的正向增强。广泛的离线和在线评估验证了我们提出的DivSPA在准确性和多样性方面的有效性。DivSPA简单且有效。

    Personalized recommendation relies on user historical behaviors to provide user-interested items, and thus seriously struggles with the data sparsity issue. A powerful positive item augmentation is beneficial to address the sparsity issue, while few works could jointly consider both the accuracy and diversity of these augmented training labels. In this work, we propose a novel model-agnostic Diversified self-distillation guided positive augmentation (DivSPA) for accurate and diverse positive item augmentations. Specifically, DivSPA first conducts three types of retrieval strategies to collect high-quality and diverse positive item candidates according to users' overall interests, short-term intentions, and similar users. Next, a self-distillation module is conducted to double-check and rerank these candidates as the final positive augmentations. Extensive offline and online evaluations verify the effectiveness of our proposed DivSPA on both accuracy and diversity. DivSPA is simple and 
    
[^8]: Delphic成本与利益在网络搜索中的效用主义和历史分析

    Delphic Costs and Benefits in Web Search: A utilitarian and historical analysis. (arXiv:2308.07525v1 [cs.IR])

    [http://arxiv.org/abs/2308.07525](http://arxiv.org/abs/2308.07525)

    新框架概念化和操作化了搜索的全面用户体验，其中强调了搜索中的Delphic成本和利益，包括非货币成本和可能受损的利益。

    

    本文提出了一个用效用主义视角研究搜索的全面用户体验的新框架。虽然网络搜索引擎被广泛认为是“免费”的，但搜索需要时间和精力，实际上存在许多相互交织的非货币成本（如时间成本、认知成本、互动成本），而利益可能会因为误解和错误信息而受到损害。这种成本和利益的描述似乎与人类在追求某个更大任务中寻找信息的本质联系在一起：大多数成本和损害可以在任何网络搜索引擎的交互、公共图书馆的交互，甚至在与古代神谕的交互中找到。为了强调这种内在联系，我们将这些成本和利益称为Delphic，与明确的财务成本和利益形成对比。

    We present a new framework to conceptualize and operationalize the total user experience of search, by studying the entirety of a search journey from an utilitarian point of view.  Web search engines are widely perceived as "free". But search requires time and effort: in reality there are many intermingled non-monetary costs (e.g. time costs, cognitive costs, interactivity costs) and the benefits may be marred by various impairments, such as misunderstanding and misinformation. This characterization of costs and benefits appears to be inherent to the human search for information within the pursuit of some larger task: most of the costs and impairments can be identified in interactions with any web search engine, interactions with public libraries, and even in interactions with ancient oracles. To emphasize this innate connection, we call these costs and benefits Delphic, in contrast to explicitly financial costs and benefits.  Our main thesis is that the users' satisfaction with a sear
    
[^9]: 利用异构数据进行兴趣点推荐的调查

    A Survey on Point-of-Interest Recommendations Leveraging Heterogeneous Data. (arXiv:2308.07426v1 [cs.IR])

    [http://arxiv.org/abs/2308.07426](http://arxiv.org/abs/2308.07426)

    本文针对旅游领域的兴趣点推荐问题进行了调查研究，探讨了利用异构数据解决旅途中兴趣点推荐问题的潜力与挑战。

    

    旅游是推荐系统的一个重要应用领域。在这个领域中，推荐系统主要负责为交通、住宿、兴趣点或旅游服务提供个性化推荐。在这些任务中，尤其是对个体游客可能感兴趣的兴趣点进行推荐的问题近年来引起了越来越多的关注。然而，在游客“旅途中”提供兴趣点推荐可能会面临特殊挑战，因为用户的上下文变化多样。随着互联网的快速发展和当今各种在线服务的大量数据，各种异构数据源的数据已经变得可用，这些异构数据源为解决旅途中兴趣点推荐问题的挑战提供了巨大潜力。在这项工作中，我们从异构数据的角度提供了2017年至2022年间已发表的兴趣点推荐研究的综述。

    Tourism is an important application domain for recommender systems. In this domain, recommender systems are for example tasked with providing personalized recommendations for transportation, accommodation, points-of-interest (POIs), or tourism services. Among these tasks, in particular the problem of recommending POIs that are of likely interest to individual tourists has gained growing attention in recent years. Providing POI recommendations to tourists \emph{during their trip} can however be especially challenging due to the variability of the users' context. With the rapid development of the Web and today's multitude of online services, vast amounts of data from various sources have become available, and these heterogeneous data sources represent a huge potential to better address the challenges of in-trip POI recommendation problems. In this work, we provide a comprehensive survey of published research on POI recommendation between 2017 and 2022 from the perspective of heterogeneou
    
[^10]: 使用不同程度的合并疾病信息改进基于ICD的语义相似度

    Improving ICD-based semantic similarity by accounting for varying degrees of comorbidity. (arXiv:2308.07359v1 [cs.LG])

    [http://arxiv.org/abs/2308.07359](http://arxiv.org/abs/2308.07359)

    本研究通过考虑记录合并疾病的比例项，改进了基于ICD码的语义相似度计算方法。

    

    在精准医学中，寻找相似的患者是常见的目标，有助于治疗结果评估和临床决策支持。选择广泛可用的患者特征和适当的数学方法来计算相似度是至关重要的。国际疾病和相关健康问题统计分类（ICD）码被全球范围内用于编码疾病，并且几乎适用于所有患者。将其聚合为包含主要和次要诊断的集合，它们可以显示出一定程度的合并疾病并揭示合并疾病模式。可以使用语义相似度算法根据患者的ICD码计算相似度。这些算法通常使用单术语专家评分的数据集进行评估。然而，实际患者数据往往显示出不同程度的记录合并疾病，可能会影响算法的性能。为了解决这个问题，我们提出了一个考虑记录合并疾病的比例项。

    Finding similar patients is a common objective in precision medicine, facilitating treatment outcome assessment and clinical decision support. Choosing widely-available patient features and appropriate mathematical methods for similarity calculations is crucial. International Statistical Classification of Diseases and Related Health Problems (ICD) codes are used worldwide to encode diseases and are available for nearly all patients. Aggregated as sets consisting of primary and secondary diagnoses they can display a degree of comorbidity and reveal comorbidity patterns. It is possible to compute the similarity of patients based on their ICD codes by using semantic similarity algorithms. These algorithms have been traditionally evaluated using a single-term expert rated data set.  However, real-word patient data often display varying degrees of documented comorbidities that might impair algorithm performance. To account for this, we present a scale term that considers documented comorbid
    
[^11]: 信息检索中的大型语言模型：一项综述

    Large Language Models for Information Retrieval: A Survey. (arXiv:2308.07107v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.07107](http://arxiv.org/abs/2308.07107)

    本综述将大型语言模型（LLMs）在信息检索中的发展进行了综述，探讨了其在捕捉上下文信号和语义细微之处方面的优势和挑战，以及与传统检索方法的结合的重要性。

    

    作为信息获取的主要手段，信息检索（IR）系统，如搜索引擎，已经融入到我们的日常生活中。这些系统还作为对话、问答和推荐系统的组成部分。IR的发展轨迹从基于词项的方法起步，逐渐发展成与先进的神经模型相融合。尽管神经模型擅长捕捉复杂的上下文信号和语义细微之处，从而改变了IR的格局，但它们仍然面临着数据稀缺、可解释性以及生成上下文合理但潜在不准确响应的挑战。这种演变需要传统方法（如基于词项的稀疏检索方法与快速响应）和现代神经架构（如具有强大语言理解能力的语言模型）的结合。与此同时，大型语言模型（LLMs），如ChatGPT和GPT-4的出现，引起了一场革命

    As a primary means of information acquisition, information retrieval (IR) systems, such as search engines, have integrated themselves into our daily lives. These systems also serve as components of dialogue, question-answering, and recommender systems. The trajectory of IR has evolved dynamically from its origins in term-based methods to its integration with advanced neural models. While the neural models excel at capturing complex contextual signals and semantic nuances, thereby reshaping the IR landscape, they still face challenges such as data scarcity, interpretability, and the generation of contextually plausible yet potentially inaccurate responses. This evolution requires a combination of both traditional methods (such as term-based sparse retrieval methods with rapid response) and modern neural architectures (such as language models with powerful language understanding capacity). Meanwhile, the emergence of large language models (LLMs), typified by ChatGPT and GPT-4, has revolu
    
[^12]: 推荐系统的数据增广：一种基于最大边际矩阵分解的半监督方法

    Data augmentation for recommender system: A semi-supervised approach using maximum margin matrix factorization. (arXiv:2306.13050v1 [cs.IR])

    [http://arxiv.org/abs/2306.13050](http://arxiv.org/abs/2306.13050)

    本研究提出了一种基于最大边际矩阵分解的半监督方法来增广和细化协同过滤算法的评级预测。该方法利用自我训练来评估评分的置信度，并通过系统的数据增广策略来提高算法性能。

    

    协同过滤已成为推荐系统开发的常用方法，其中，根据用户的过去喜好和其他用户的可用偏好信息预测其对新物品的评分。尽管CF方法很受欢迎，但其性能通常受观察到的条目的稀疏性的极大限制。本研究探讨最大边际矩阵分解（MMMF）的数据增广和细化方面，该方法是广泛接受的用于评级预测的CF技术，之前尚未进行研究。我们利用CF算法的固有特性来评估单个评分的置信度，并提出了一种基于自我训练的半监督评级增强方法。我们假设任何CF算法的预测低置信度是由于训练数据的某些不足，因此，通过采用系统的数据增广策略，可以提高算法的性能。

    Collaborative filtering (CF) has become a popular method for developing recommender systems (RS) where ratings of a user for new items is predicted based on her past preferences and available preference information of other users. Despite the popularity of CF-based methods, their performance is often greatly limited by the sparsity of observed entries. In this study, we explore the data augmentation and refinement aspects of Maximum Margin Matrix Factorization (MMMF), a widely accepted CF technique for the rating predictions, which have not been investigated before. We exploit the inherent characteristics of CF algorithms to assess the confidence level of individual ratings and propose a semi-supervised approach for rating augmentation based on self-training. We hypothesize that any CF algorithm's predictions with low confidence are due to some deficiency in the training data and hence, the performance of the algorithm can be improved by adopting a systematic data augmentation strategy
    
[^13]: Probe：学习用户在时间跨度的捆绑选择中的个性化投影偏差

    Probe: Learning Users' Personalized Projection Bias in Intertemporal Bundle Choices. (arXiv:2303.06016v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.06016](http://arxiv.org/abs/2303.06016)

    本文提出了一种新的偏差嵌入式偏好模型——Probe，旨在解决用户在时间跨度的购物选择中的投影偏差和参照点效应，提高决策的有效性和个性化。

    

    时间跨度的选择需要权衡现在的成本和未来的收益。其中一种具体的选择是决定购买单个物品还是选择包含该物品的捆绑销售方式。以往的研究假设个人对这些选择中涉及的因素有准确的期望。然而，在现实中，用户对这些因素的感知往往存在偏差，导致了非理性和次优的决策。本文重点关注两种常见的偏差：投影偏差和参照点效应，并为此提出了一种新颖的偏差嵌入式偏好模型——Probe。该模型利用加权函数来捕捉用户的投影偏差，利用价值函数来考虑参照点效应，并引入行为经济学中的前景理论来组合加权和价值函数。这使得我们能够确定用户购买捆绑销售的概率，从而提高决策的有效性和个性化。

    Intertemporal choices involve making decisions that require weighing the costs in the present against the benefits in the future. One specific type of intertemporal choice is the decision between purchasing an individual item or opting for a bundle that includes that item. Previous research assumes that individuals have accurate expectations of the factors involved in these choices. However, in reality, users' perceptions of these factors are often biased, leading to irrational and suboptimal decision-making. In this work, we specifically focus on two commonly observed biases: projection bias and the reference-point effect. To address these biases, we propose a novel bias-embedded preference model called Probe. The Probe incorporates a weight function to capture users' projection bias and a value function to account for the reference-point effect, and introduce prospect theory from behavioral economics to combine the weight and value functions. This allows us to determine the probabili
    

