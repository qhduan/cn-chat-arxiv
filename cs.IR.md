# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Parmesan: mathematical concept extraction for education.](http://arxiv.org/abs/2307.06699) | Parmesan是一个原型系统，用于在上下文中搜索和定义数学概念，特别关注范畴论领域。该系统利用自然语言处理组件进行概念提取、关系提取、定义提取和实体链接。通过该系统的开发，可以解决现有技术不能直接应用于范畴论领域的问题，并提供了两个数学语料库以支持系统的使用。 |
| [^2] | [Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations.](http://arxiv.org/abs/2307.06576) | 本文介绍了一种名为GLORY的模型，通过全局图与本地表示相结合，增强了个性化推荐系统。该模型通过构建全局感知历史新闻编码器来融合历史新闻表示，并考虑了用户隐藏的动机和行为。 |
| [^3] | [Assessing the Ability of ChatGPT to Screen Articles for Systematic Reviews.](http://arxiv.org/abs/2307.06464) | 本论文评估了聊天型AI模型ChatGPT在系统性综述（SR）文章筛选中的性能，结果表明ChatGPT是自动化SR过程的可行选择。 |
| [^4] | [Adaptive Graph Contrastive Learning for Recommendation.](http://arxiv.org/abs/2305.10837) | 本文提出了一种自适应图对比学习的推荐框架，通过对比学习的方式改进用户和物品的表示，关注数据中的难以区分的负面例子的信息。 |

# 详细

[^1]: Parmesan：教育中的数学概念提取

    Parmesan: mathematical concept extraction for education. (arXiv:2307.06699v1 [cs.CL])

    [http://arxiv.org/abs/2307.06699](http://arxiv.org/abs/2307.06699)

    Parmesan是一个原型系统，用于在上下文中搜索和定义数学概念，特别关注范畴论领域。该系统利用自然语言处理组件进行概念提取、关系提取、定义提取和实体链接。通过该系统的开发，可以解决现有技术不能直接应用于范畴论领域的问题，并提供了两个数学语料库以支持系统的使用。

    

    数学是一个高度专业化的领域，具有自己独特的挑战，但在自然语言处理领域的研究却有限。然而，数学在许多不同领域的跨学科研究中经常依赖于对数学概念的理解。为了帮助来自其他领域的研究人员，我们开发了一个原型系统，用于在上下文中搜索和定义数学概念，重点关注范畴论领域。这个系统名为Parmesan，依赖于自然语言处理组件，包括概念提取、关系提取、定义提取和实体链接。在开发这个系统的过程中，我们展示了现有技术不能直接应用于范畴论领域，并提出了一种混合技术，这种技术表现良好，但我们预计系统将随着时间的推移而不断演变。我们还提供了两个清理过的数学语料库，用于支持原型系统，这些语料库基于期刊文章。

    Mathematics is a highly specialized domain with its own unique set of challenges that has seen limited study in natural language processing. However, mathematics is used in a wide variety of fields and multidisciplinary research in many different domains often relies on an understanding of mathematical concepts. To aid researchers coming from other fields, we develop a prototype system for searching for and defining mathematical concepts in context, focusing on the field of category theory. This system, Parmesan, depends on natural language processing components including concept extraction, relation extraction, definition extraction, and entity linking. In developing this system, we show that existing techniques cannot be applied directly to the category theory domain, and suggest hybrid techniques that do perform well, though we expect the system to evolve over time. We also provide two cleaned mathematical corpora that power the prototype system, which are based on journal articles 
    
[^2]: 超越本地范围：全球图增强个性化新闻推荐

    Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations. (arXiv:2307.06576v1 [cs.IR])

    [http://arxiv.org/abs/2307.06576](http://arxiv.org/abs/2307.06576)

    本文介绍了一种名为GLORY的模型，通过全局图与本地表示相结合，增强了个性化推荐系统。该模型通过构建全局感知历史新闻编码器来融合历史新闻表示，并考虑了用户隐藏的动机和行为。

    

    精确地向用户推荐候选新闻文章一直是个性化新闻推荐系统的核心挑战。大多数近期的研究主要集中在使用先进的自然语言处理技术从丰富的文本数据中提取语义信息，使用从本地历史新闻派生的基于内容的方法。然而，这种方法缺乏全局视角，未能考虑用户隐藏的动机和行为，超越语义信息。为了解决这个问题，我们提出了一种新颖的模型 GLORY（Global-LOcal news Recommendation sYstem），它结合了从其他用户学到的全局表示和本地表示，来增强个性化推荐系统。我们通过构建一个全局感知历史新闻编码器来实现这一目标，其中包括一个全局新闻图，并使用门控图神经网络来丰富新闻表示，从而通过历史新闻聚合器融合历史新闻表示。

    Precisely recommending candidate news articles to users has always been a core challenge for personalized news recommendation systems. Most recent works primarily focus on using advanced natural language processing techniques to extract semantic information from rich textual data, employing content-based methods derived from local historical news. However, this approach lacks a global perspective, failing to account for users' hidden motivations and behaviors beyond semantic information. To address this challenge, we propose a novel model called GLORY (Global-LOcal news Recommendation sYstem), which combines global representations learned from other users with local representations to enhance personalized recommendation systems. We accomplish this by constructing a Global-aware Historical News Encoder, which includes a global news graph and employs gated graph neural networks to enrich news representations, thereby fusing historical news representations by a historical news aggregator.
    
[^3]: 评估ChatGPT对于系统性综述文章筛选的能力

    Assessing the Ability of ChatGPT to Screen Articles for Systematic Reviews. (arXiv:2307.06464v1 [cs.SE])

    [http://arxiv.org/abs/2307.06464](http://arxiv.org/abs/2307.06464)

    本论文评估了聊天型AI模型ChatGPT在系统性综述（SR）文章筛选中的性能，结果表明ChatGPT是自动化SR过程的可行选择。

    

    通过在研究领域内组织知识，系统性综述（SR）为研究提供了宝贵的线索。有证据表明，SR已成为软件工程中一流的艺术品。然而，SR筛选阶段所需的繁琐手动工作使得这些研究变得昂贵且容易出错。尽管传统上认为筛选不适合自动化，但基于大型语言模型支持的生成式AI驱动的聊天机器人的出现将改变这一情况。在本报告中，我们提出了一种利用这些新技术发展自动化SR筛选的方法。我们评估了ChatGPT在SR文章筛选中的一致性、分类性能和推广能力，并将这些数据与传统用于SR自动化的分类器进行比较。我们的结果表明，ChatGPT是自动化SR过程的可行选择，但开发者在集成时需要仔细考虑。

    By organizing knowledge within a research field, Systematic Reviews (SR) provide valuable leads to steer research. Evidence suggests that SRs have become first-class artifacts in software engineering. However, the tedious manual effort associated with the screening phase of SRs renders these studies a costly and error-prone endeavor. While screening has traditionally been considered not amenable to automation, the advent of generative AI-driven chatbots, backed with large language models is set to disrupt the field. In this report, we propose an approach to leverage these novel technological developments for automating the screening of SRs. We assess the consistency, classification performance, and generalizability of ChatGPT in screening articles for SRs and compare these figures with those of traditional classifiers used in SR automation. Our results indicate that ChatGPT is a viable option to automate the SR processes, but requires careful considerations from developers when integra
    
[^4]: 自适应图对比学习用于推荐系统

    Adaptive Graph Contrastive Learning for Recommendation. (arXiv:2305.10837v1 [cs.IR])

    [http://arxiv.org/abs/2305.10837](http://arxiv.org/abs/2305.10837)

    本文提出了一种自适应图对比学习的推荐框架，通过对比学习的方式改进用户和物品的表示，关注数据中的难以区分的负面例子的信息。

    

    近年来，图神经网络已成功地应用于推荐系统，成为一种有效的协同过滤方法。基于图神经网络的推荐系统的关键思想是沿着用户-物品交互边递归地执行消息传递，以完善编码嵌入，这依赖于充足和高质量的训练数据。由于实际推荐场景中的用户行为数据通常存在噪声并呈现出倾斜分布，一些推荐方法利用自监督学习来改善用户表示，例如SGL和SimGCL。 然而，尽管它们非常有效，但它们通过创建对比视图进行自监督学习，具有数据增强探索，需要进行繁琐的试错选择增强方法。本文提出了一种新的自适应图对比学习（AdaptiveGCL）框架，通过自适应但关注数据中的难以区分的负面例子的信息，用对比学习的方式改进用户和物品的表示。

    Recently, graph neural networks (GNNs) have been successfully applied to recommender systems as an effective collaborative filtering (CF) approach. The key idea of GNN-based recommender system is to recursively perform the message passing along the user-item interaction edge for refining the encoded embeddings, relying on sufficient and high-quality training data. Since user behavior data in practical recommendation scenarios is often noisy and exhibits skewed distribution, some recommendation approaches, e.g., SGL and SimGCL, leverage self-supervised learning to improve user representations against the above issues. Despite their effectiveness, however, they conduct self-supervised learning through creating contrastvie views, depending on the exploration of data augmentations with the problem of tedious trial-and-error selection of augmentation methods. In this paper, we propose a novel Adaptive Graph Contrastive Learning (AdaptiveGCL) framework which conducts graph contrastive learni
    

