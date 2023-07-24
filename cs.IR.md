# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Alleviating the Long-Tail Problem in Conversational Recommender Systems.](http://arxiv.org/abs/2307.11650) | 本文提出了一种名为LOT-CRS的新框架，核心是通过模拟和利用均衡的CRS数据集来改善长尾推荐性能，以解决现有CRS数据集中长尾问题。 |
| [^2] | [Identifying document similarity using a fast estimation of the Levenshtein Distance based on compression and signatures.](http://arxiv.org/abs/2307.11496) | 该论文提出了一种通过压缩和签名快速估计Levenshtein距离的方法，用于识别文档相似性。实验结果表明，在运行时间效率和准确性方面具有很大的潜力。 |
| [^3] | [Analysis of Elephant Movement in Sub-Saharan Africa: Ecological, Climatic, and Conservation Perspectives.](http://arxiv.org/abs/2307.11325) | 本研究分析了萨赫勒以南非洲象移动的模式，重点关注了季节变化和降雨模式等动态驱动因素。研究结果有助于预测生态因素对象迁徙的潜在影响，并为制定保护策略提供了综合的视角。 |
| [^4] | [Jina Embeddings: A Novel Set of High-Performance Sentence Embedding Models.](http://arxiv.org/abs/2307.11224) | Jina Embeddings是一组高性能的句子嵌入模型，能够捕捉文本的语义本质。该论文详细介绍了Jina Embeddings的开发过程，并通过性能评估验证了其优越性能。 |
| [^5] | [RCVaR: an Economic Approach to Estimate Cyberattacks Costs using Data from Industry Reports.](http://arxiv.org/abs/2307.11140) | 本文介绍了一种经济方法RCVaR，利用公开的网络安全报告的现实世界信息来估算网络安全成本。这种方法可以帮助理解由网络攻击导致的财务损失，并确定最重要的网络风险因素。 |
| [^6] | [Towards the Better Ranking Consistency: A Multi-task Learning Framework for Early Stage Ads Ranking.](http://arxiv.org/abs/2307.11096) | 本论文提出了一种多任务学习框架，用于早期广告排序，以解决早期阶段和最终阶段排序之间的一致性问题。 |
| [^7] | [Detecting deceptive reviews using text classification.](http://arxiv.org/abs/2307.10617) | 这篇论文提出了一种使用机器学习模型的方法来识别虚假评论，并通过在餐馆评论的数据集上进行实验验证了其性能。 |
| [^8] | [Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations.](http://arxiv.org/abs/2307.06576) | 本文介绍了一种名为GLORY的模型，通过全局图与本地表示相结合，增强了个性化推荐系统。该模型通过构建全局感知历史新闻编码器来融合历史新闻表示，并考虑了用户隐藏的动机和行为。 |
| [^9] | [Large Language Model Augmented Narrative Driven Recommendations.](http://arxiv.org/abs/2306.02250) | 这个论文研究了如何使用大型语言模型（LLMs）为基于叙事的推荐系统提供数据增强，以解决其缺乏训练数据的问题。 |
| [^10] | [Editable User Profiles for Controllable Text Recommendation.](http://arxiv.org/abs/2304.04250) | 本文提出了一种新的概念值瓶颈模型LACE，用于可控文本推荐。该模型基于用户文档学习个性化的概念表示，并通过多种交互方式为用户提供了控制推荐的机制，验证了在离线和在线实验中该模型的推荐质量和有效性。 |

# 详细

[^1]: 缓解对话式推荐系统中的长尾问题

    Alleviating the Long-Tail Problem in Conversational Recommender Systems. (arXiv:2307.11650v1 [cs.IR])

    [http://arxiv.org/abs/2307.11650](http://arxiv.org/abs/2307.11650)

    本文提出了一种名为LOT-CRS的新框架，核心是通过模拟和利用均衡的CRS数据集来改善长尾推荐性能，以解决现有CRS数据集中长尾问题。

    

    对话式推荐系统（CRS）旨在通过自然语言对话提供推荐服务。为了开发有效的CRS，高质量的CRS数据集非常关键。然而，现有的CRS数据集存在长尾问题，即在对话中很少（甚至从未）提到的项目占较大比例，这些被称为长尾项目。因此，基于这些数据集训练的CRS倾向于推荐频繁出现的项目，并且推荐项目的多样性会大大降低，使用户更容易感到厌倦。为了解决这个问题，本文提出了一种新颖的框架LOT-CRS，该框架专注于模拟和利用平衡的CRS数据集（即均匀涵盖所有项目）来改善CRS的长尾推荐性能。在我们的方法中，我们设计了两个预训练任务，以增强对长尾项目的模拟对话的理解，并采用检索增强的微调与实验室

    Conversational recommender systems (CRS) aim to provide the recommendation service via natural language conversations. To develop an effective CRS, high-quality CRS datasets are very crucial. However, existing CRS datasets suffer from the long-tail issue, \ie a large proportion of items are rarely (or even never) mentioned in the conversations, which are called long-tail items. As a result, the CRSs trained on these datasets tend to recommend frequent items, and the diversity of the recommended items would be largely reduced, making users easier to get bored.  To address this issue, this paper presents \textbf{LOT-CRS}, a novel framework that focuses on simulating and utilizing a balanced CRS dataset (\ie covering all the items evenly) for improving \textbf{LO}ng-\textbf{T}ail recommendation performance of CRSs. In our approach, we design two pre-training tasks to enhance the understanding of simulated conversation for long-tail items, and adopt retrieval-augmented fine-tuning with lab
    
[^2]: 使用基于压缩和签名的Levenshtein距离快速估计方法识别文档相似性

    Identifying document similarity using a fast estimation of the Levenshtein Distance based on compression and signatures. (arXiv:2307.11496v1 [cs.IR])

    [http://arxiv.org/abs/2307.11496](http://arxiv.org/abs/2307.11496)

    该论文提出了一种通过压缩和签名快速估计Levenshtein距离的方法，用于识别文档相似性。实验结果表明，在运行时间效率和准确性方面具有很大的潜力。

    

    识别文档相似性在诸如源代码分析或抄袭检测等应用中有许多应用。然而，识别相似性并非易事，可能具有复杂的时间复杂度。例如，Levenshtein距离是定义两个文档相似性的常见度量标准，但它具有二次运行时间，使其在大文档（以几百千字节开头的大文档）中不适用。在本文中，我们提出了一个新颖的概念，允许估计Levenshtein距离：算法首先使用用户定义的压缩比对文档进行签名（类似于哈希值）压缩。然后可以将签名相互比较（应用一些约束条件），结果就是估计的Levenshtein距离。我们的评估结果显示在运行时间效率和准确性方面都有令人期待的结果。此外，我们引入了一个显著度评分，允许评估人员设定阈值并识别相关文档。

    Identifying document similarity has many applications, e.g., source code analysis or plagiarism detection. However, identifying similarities is not trivial and can be time complex. For instance, the Levenshtein Distance is a common metric to define the similarity between two documents but has quadratic runtime which makes it impractical for large documents where large starts with a few hundred kilobytes. In this paper, we present a novel concept that allows estimating the Levenshtein Distance: the algorithm first compresses documents to signatures (similar to hash values) using a user-defined compression ratio. Signatures can then be compared against each other (some constrains apply) where the outcome is the estimated Levenshtein Distance. Our evaluation shows promising results in terms of runtime efficiency and accuracy. In addition, we introduce a significance score allowing examiners to set a threshold and identify related documents.
    
[^3]: 萨赫勒以南非洲象移动的分析：生态学、气候和保护观点

    Analysis of Elephant Movement in Sub-Saharan Africa: Ecological, Climatic, and Conservation Perspectives. (arXiv:2307.11325v1 [q-bio.PE])

    [http://arxiv.org/abs/2307.11325](http://arxiv.org/abs/2307.11325)

    本研究分析了萨赫勒以南非洲象移动的模式，重点关注了季节变化和降雨模式等动态驱动因素。研究结果有助于预测生态因素对象迁徙的潜在影响，并为制定保护策略提供了综合的视角。

    

    象与环境的相互作用对生态学和保护策略都有深远的影响。本研究提出了一种分析方法来解读萨赫勒以南非洲象移动的复杂模式，重点关注季节变化和降雨模式等关键生态驱动因素。尽管围绕这些具有影响力的因素存在复杂性，我们的分析提供了对非洲动态景观背景下象迁徙行为的全面视角。我们综合的方法使我们能够预测这些生态决定因素对象迁徙的潜在影响，这是建立知情的保护策略的关键一步。考虑到全球气候变化对季节和降雨模式的影响，这种预测尤为重要，因为它未来可能会对象的行动产生显著影响。我们的工作成果旨在不仅推进对移动生态学的理解，同时也为保护实践提供参考。

    The interaction between elephants and their environment has profound implications for both ecology and conservation strategies. This study presents an analytical approach to decipher the intricate patterns of elephant movement in Sub-Saharan Africa, concentrating on key ecological drivers such as seasonal variations and rainfall patterns. Despite the complexities surrounding these influential factors, our analysis provides a holistic view of elephant migratory behavior in the context of the dynamic African landscape. Our comprehensive approach enables us to predict the potential impact of these ecological determinants on elephant migration, a critical step in establishing informed conservation strategies. This projection is particularly crucial given the impacts of global climate change on seasonal and rainfall patterns, which could substantially influence elephant movements in the future. The findings of our work aim to not only advance the understanding of movement ecology but also f
    
[^4]: Jina Embeddings:一种新颖的高性能句子嵌入模型

    Jina Embeddings: A Novel Set of High-Performance Sentence Embedding Models. (arXiv:2307.11224v1 [cs.CL])

    [http://arxiv.org/abs/2307.11224](http://arxiv.org/abs/2307.11224)

    Jina Embeddings是一组高性能的句子嵌入模型，能够捕捉文本的语义本质。该论文详细介绍了Jina Embeddings的开发过程，并通过性能评估验证了其优越性能。

    

    Jina Embeddings由一组高性能的句子嵌入模型组成，能够将各种文本输入转化为数值表示，从而捕捉文本的语义本质。虽然这些模型并非专门设计用于文本生成，但在密集检索和语义文本相似性等应用中表现出色。本文详细介绍了Jina Embeddings的开发过程，从创建高质量的成对和三元数据集开始。它强调了数据清理在数据集准备中的关键作用，并对模型训练过程进行了深入探讨，最后利用Massive Textual Embedding Benchmark（MTEB）进行了全面的性能评估。

    Jina Embeddings constitutes a set of high-performance sentence embedding models adept at translating various textual inputs into numerical representations, thereby capturing the semantic essence of the text. While these models are not exclusively designed for text generation, they excel in applications such as dense retrieval and semantic textual similarity. This paper details the development of Jina Embeddings, starting with the creation of a high-quality pairwise and triplet dataset. It underlines the crucial role of data cleaning in dataset preparation, gives in-depth insights into the model training process, and concludes with a comprehensive performance evaluation using the Massive Textual Embedding Benchmark (MTEB).
    
[^5]: RCVaR:一种利用行业报告数据估算网络攻击成本的经济方法

    RCVaR: an Economic Approach to Estimate Cyberattacks Costs using Data from Industry Reports. (arXiv:2307.11140v1 [cs.CR])

    [http://arxiv.org/abs/2307.11140](http://arxiv.org/abs/2307.11140)

    本文介绍了一种经济方法RCVaR，利用公开的网络安全报告的现实世界信息来估算网络安全成本。这种方法可以帮助理解由网络攻击导致的财务损失，并确定最重要的网络风险因素。

    

    数字化增加了商业机会，也增加了公司成为毁灭性网络攻击受害者的风险。因此，管理风险暴露和网络安全策略对于希望在竞争市场中生存的数字化公司至关重要。然而，理解公司特定的风险并量化其相关成本并不容易。当前的方法无法提供个性化和定量化的网络安全影响货币估计。由于资源有限和技术专长，中小型企业甚至大公司受到影响，并且难以量化其网络攻击风险。因此，必须采用新的方法来支持对网络攻击导致的财务损失的理解。本文介绍了实际网络风险价值 (RCVaR)，这是一种经济方法，利用公开的网络安全报告的现实世界信息来估算网络安全成本。RCVaR从各种来源中确定最重要的网络风险因素

    Digitization increases business opportunities and the risk of companies being victims of devastating cyberattacks. Therefore, managing risk exposure and cybersecurity strategies is essential for digitized companies that want to survive in competitive markets. However, understanding company-specific risks and quantifying their associated costs is not trivial. Current approaches fail to provide individualized and quantitative monetary estimations of cybersecurity impacts. Due to limited resources and technical expertise, SMEs and even large companies are affected and struggle to quantify their cyberattack exposure. Therefore, novel approaches must be placed to support the understanding of the financial loss due to cyberattacks. This article introduces the Real Cyber Value at Risk (RCVaR), an economical approach for estimating cybersecurity costs using real-world information from public cybersecurity reports. RCVaR identifies the most significant cyber risk factors from various sources an
    
[^6]: 为了更好的排序一致性：一种面向早期广告排序的多任务学习框架

    Towards the Better Ranking Consistency: A Multi-task Learning Framework for Early Stage Ads Ranking. (arXiv:2307.11096v1 [cs.IR])

    [http://arxiv.org/abs/2307.11096](http://arxiv.org/abs/2307.11096)

    本论文提出了一种多任务学习框架，用于早期广告排序，以解决早期阶段和最终阶段排序之间的一致性问题。

    

    在大规模广告推荐中，将广告排序系统分为检索、早期和最终阶段是一种常见做法，以平衡效率和准确性。早期阶段的排序通常使用高效模型从一组检索到的广告中生成候选集。然后，将候选集馈送到计算密集且准确的最终阶段排序系统，生成最终的广告推荐。由于系统限制，早期和最终阶段的排序使用不同的特征和模型架构，导致了严重的排序一致性问题，即早期阶段的广告召回率较低，即最终阶段中排名靠前的广告在早期阶段排名较低。为了将更好的广告从早期阶段传递到最终阶段的排名，我们提出了一种面向早期阶段排序的多任务学习框架，以捕获多个最终阶段排序组件（即广告点击和广告质量事件）及其任务关系。

    Dividing ads ranking system into retrieval, early, and final stages is a common practice in large scale ads recommendation to balance the efficiency and accuracy. The early stage ranking often uses efficient models to generate candidates out of a set of retrieved ads. The candidates are then fed into a more computationally intensive but accurate final stage ranking system to produce the final ads recommendation. As the early and final stage ranking use different features and model architectures because of system constraints, a serious ranking consistency issue arises where the early stage has a low ads recall, i.e., top ads in the final stage are ranked low in the early stage. In order to pass better ads from the early to the final stage ranking, we propose a multi-task learning framework for early stage ranking to capture multiple final stage ranking components (i.e. ads clicks and ads quality events) and their task relations. With our multi-task learning framework, we can not only ac
    
[^7]: 使用文本分类检测虚假评论

    Detecting deceptive reviews using text classification. (arXiv:2307.10617v1 [cs.IR])

    [http://arxiv.org/abs/2307.10617](http://arxiv.org/abs/2307.10617)

    这篇论文提出了一种使用机器学习模型的方法来识别虚假评论，并通过在餐馆评论的数据集上进行实验验证了其性能。

    

    近年来，在线评论在推广任何产品或服务方面发挥着重要作用。企业可能会嵌入虚假评论以吸引客户购买他们的产品。他们甚至可能突出强调自己产品的优点或批评竞争对手的产品。市场营销人员、广告商和其他在线商业用户有动机为他们想要推广的产品编写虚假的正面评论，或者为他们真正不喜欢的产品提供虚假的负面评论。因此，识别虚假评论是一个紧迫且持续的研究领域。本研究论文提出了一种机器学习模型方法来识别虚假评论。论文调查了在一个餐馆评论的虚假意见垃圾语料库数据集上进行的多次实验的性能。我们采用了n-gram模型和最大特征来识别虚假评论。

    In recent years, online reviews play a vital role for promoting any kind of product or services. Businesses may embed fake reviews in order to attract customers to purchase their products. They may even highlight the benefits of their own product or criticize the competition's product. Marketers, advertisers, and other online business users have incentive to create fake positive reviews for products which they want to promote or give fake negative reviews for products which they really don't like. So now-a-days writing a deceptive review is inevitable thing for promoting their own business or degrading competitor's reputation. Thus, identifying deceptive reviews is an intense and on-going research area. This research paper proposes machine learning model approach to identify deceptive reviews. The paper investigates the performance of the several experiments done on a Deceptive Opinion Spam Corpus dataset of restaurants reviews. We developed a n-gram model and max features to identify 
    
[^8]: 超越本地范围：全球图增强个性化新闻推荐

    Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations. (arXiv:2307.06576v1 [cs.IR])

    [http://arxiv.org/abs/2307.06576](http://arxiv.org/abs/2307.06576)

    本文介绍了一种名为GLORY的模型，通过全局图与本地表示相结合，增强了个性化推荐系统。该模型通过构建全局感知历史新闻编码器来融合历史新闻表示，并考虑了用户隐藏的动机和行为。

    

    精确地向用户推荐候选新闻文章一直是个性化新闻推荐系统的核心挑战。大多数近期的研究主要集中在使用先进的自然语言处理技术从丰富的文本数据中提取语义信息，使用从本地历史新闻派生的基于内容的方法。然而，这种方法缺乏全局视角，未能考虑用户隐藏的动机和行为，超越语义信息。为了解决这个问题，我们提出了一种新颖的模型 GLORY（Global-LOcal news Recommendation sYstem），它结合了从其他用户学到的全局表示和本地表示，来增强个性化推荐系统。我们通过构建一个全局感知历史新闻编码器来实现这一目标，其中包括一个全局新闻图，并使用门控图神经网络来丰富新闻表示，从而通过历史新闻聚合器融合历史新闻表示。

    Precisely recommending candidate news articles to users has always been a core challenge for personalized news recommendation systems. Most recent works primarily focus on using advanced natural language processing techniques to extract semantic information from rich textual data, employing content-based methods derived from local historical news. However, this approach lacks a global perspective, failing to account for users' hidden motivations and behaviors beyond semantic information. To address this challenge, we propose a novel model called GLORY (Global-LOcal news Recommendation sYstem), which combines global representations learned from other users with local representations to enhance personalized recommendation systems. We accomplish this by constructing a Global-aware Historical News Encoder, which includes a global news graph and employs gated graph neural networks to enrich news representations, thereby fusing historical news representations by a historical news aggregator.
    
[^9]: 大型语言模型增强的基于叙事的推荐系统

    Large Language Model Augmented Narrative Driven Recommendations. (arXiv:2306.02250v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.02250](http://arxiv.org/abs/2306.02250)

    这个论文研究了如何使用大型语言模型（LLMs）为基于叙事的推荐系统提供数据增强，以解决其缺乏训练数据的问题。

    

    基于叙事的推荐系统是一个信息获取问题，用户通过详细描述他们的偏好和背景来请求推荐，比如旅行者在描述他们的喜好、不喜欢和旅行情况时请求景点的推荐。随着自然语言对话界面在搜索和推荐系统中的兴起，这些请求变得越来越重要。然而，基于叙事的推荐系统缺乏丰富的训练数据，并且当前的平台通常不支持这些请求。幸运的是，传统的用户-物品交互数据集包含了丰富的文本数据，例如评论，这些评论经常描述了用户的偏好和背景 - 这些数据可以用来为基于叙事的推荐模型进行训练。在这项工作中，我们探索使用大型语言模型 (LLMs) 来进行数据增强，以训练基于叙事的推荐模型。

    Narrative-driven recommendation (NDR) presents an information access problem where users solicit recommendations with verbose descriptions of their preferences and context, for example, travelers soliciting recommendations for points of interest while describing their likes/dislikes and travel circumstances. These requests are increasingly important with the rise of natural language-based conversational interfaces for search and recommendation systems. However, NDR lacks abundant training data for models, and current platforms commonly do not support these requests. Fortunately, classical user-item interaction datasets contain rich textual data, e.g., reviews, which often describe user preferences and context - this may be used to bootstrap training for NDR models. In this work, we explore using large language models (LLMs) for data augmentation to train NDR models. We use LLMs for authoring synthetic narrative queries from user-item interactions with few-shot prompting and train retri
    
[^10]: 可编辑用户档案的可控文本推荐方法

    Editable User Profiles for Controllable Text Recommendation. (arXiv:2304.04250v1 [cs.IR])

    [http://arxiv.org/abs/2304.04250](http://arxiv.org/abs/2304.04250)

    本文提出了一种新的概念值瓶颈模型LACE，用于可控文本推荐。该模型基于用户文档学习个性化的概念表示，并通过多种交互方式为用户提供了控制推荐的机制，验证了在离线和在线实验中该模型的推荐质量和有效性。

    

    实现高质量推荐的方法通常依赖于从交互数据中学习潜在表示。然而这些方法没有提供给用户控制所接收的推荐的机制。本文提出了LACE，一种新颖的概念值瓶颈模型，用于可控文本推荐。LACE基于用户交互的文档检索，将每个用户表示为简洁的可读的概念集，并基于用户文档学习概念的个性化表示。该基于概念的用户档案被利用来做出推荐。我们的模型设计通过透明的用户档案，提供了控制推荐的多种直观交互方式。我们首先在三个推荐任务（温启动、冷启动和零样本）的六个数据集上进行了离线评估，验证了从LACE获得的推荐质量。接下来，我们在在线实验中验证了LACE的有效性和用户控制能力。

    Methods for making high-quality recommendations often rely on learning latent representations from interaction data. These methods, while performant, do not provide ready mechanisms for users to control the recommendation they receive. Our work tackles this problem by proposing LACE, a novel concept value bottleneck model for controllable text recommendations. LACE represents each user with a succinct set of human-readable concepts through retrieval given user-interacted documents and learns personalized representations of the concepts based on user documents. This concept based user profile is then leveraged to make recommendations. The design of our model affords control over the recommendations through a number of intuitive interactions with a transparent user profile. We first establish the quality of recommendations obtained from LACE in an offline evaluation on three recommendation tasks spanning six datasets in warm-start, cold-start, and zero-shot setups. Next, we validate the 
    

