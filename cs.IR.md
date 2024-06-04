# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal Decoupling](https://arxiv.org/abs/2403.02630) | 该研究提出了FedHCDR框架，通过超图信号解耦的方式解决了联邦跨领域推荐中不同领域数据异质性的问题。 |
| [^2] | [Interpreting Conversational Dense Retrieval by Rewriting-Enhanced Inversion of Session Embedding](https://arxiv.org/abs/2402.12774) | 提出了CONVINV方法，通过增强的重写将不透明的对话式会话嵌入转换为明确可解释的文本，同时保持原始检索性能。 |
| [^3] | [Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search.](http://arxiv.org/abs/2401.04514) | 本论文提出一种扩展的生成增强检索（GAR）框架，通过对代码进行重写来解决代码搜索中存在的风格不匹配问题，实验结果表明该方法显著提高了检索准确性。 |
| [^4] | [TSRankLLM: A Two-Stage Adaptation of LLMs for Text Ranking.](http://arxiv.org/abs/2311.16720) | TSRankLLM提出了一种两阶段适应方法用于文本排序，通过连续预训练和改进的优化策略，实现了更好的性能。 |
| [^5] | [Mixer: Image to Multi-Modal Retrieval Learning for Industrial Application.](http://arxiv.org/abs/2305.03972) | 提出了一种新的可伸缩高效的图像到跨模态检索范式Mixer，解决了领域差距、跨模态数据对齐和融合、繁杂的数据训练标签以及海量查询和及时响应等问题。 |
| [^6] | [RLTP: Reinforcement Learning to Pace for Delayed Impression Modeling in Preloaded Ads.](http://arxiv.org/abs/2302.02592) | RLTP算法是一个强化学习算法，用于解决广告预加载过程中的延迟印象现象。 |

# 详细

[^1]: FedHCDR: 具有超图信号解耦的联邦跨领域推荐

    FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal Decoupling

    [https://arxiv.org/abs/2403.02630](https://arxiv.org/abs/2403.02630)

    该研究提出了FedHCDR框架，通过超图信号解耦的方式解决了联邦跨领域推荐中不同领域数据异质性的问题。

    

    近年来，跨领域推荐（CDR）备受关注，利用来自多个领域的用户数据来增强推荐性能。然而，当前的CDR方法需要跨领域共享用户数据，违反了《通用数据保护条例》（GDPR）。因此，已提出了许多联邦跨领域推荐（FedCDR）方法。然而，不同领域间的数据异质性不可避免地影响了联邦学习的整体性能。在这项研究中，我们提出了FedHCDR，一种具有超图信号解耦的新型联邦跨领域推荐框架。具体地，为了解决不同领域之间的数据异质性，我们引入一种称为超图信号解耦（HSD）的方法，将用户特征解耦为领域独有和领域共享特征。该方法采用高通和低通超图滤波器来进行解耦。

    arXiv:2403.02630v1 Announce Type: new  Abstract: In recent years, Cross-Domain Recommendation (CDR) has drawn significant attention, which utilizes user data from multiple domains to enhance the recommendation performance. However, current CDR methods require sharing user data across domains, thereby violating the General Data Protection Regulation (GDPR). Consequently, numerous approaches have been proposed for Federated Cross-Domain Recommendation (FedCDR). Nevertheless, the data heterogeneity across different domains inevitably influences the overall performance of federated learning. In this study, we propose FedHCDR, a novel Federated Cross-Domain Recommendation framework with Hypergraph signal decoupling. Specifically, to address the data heterogeneity across domains, we introduce an approach called hypergraph signal decoupling (HSD) to decouple the user features into domain-exclusive and domain-shared features. The approach employs high-pass and low-pass hypergraph filters to de
    
[^2]: 通过增强的重写来解释对话式密集检索的会话嵌入

    Interpreting Conversational Dense Retrieval by Rewriting-Enhanced Inversion of Session Embedding

    [https://arxiv.org/abs/2402.12774](https://arxiv.org/abs/2402.12774)

    提出了CONVINV方法，通过增强的重写将不透明的对话式会话嵌入转换为明确可解释的文本，同时保持原始检索性能。

    

    对话式密集检索已被证明在对话式搜索中非常有效。然而，对话式密集检索的一个主要局限性是它们缺乏可解释性，从而阻碍了对模型行为的直观理解以进行有针对性的改进。本文提出了CONVINV，一种简单而有效的方法，可以揭示可解释的对话式密集检索模型。CONVINV将不透明的对话式会话嵌入转换为明确可解释的文本，同时尽可能忠实地保持其原始检索性能。这种转换是通过训练一种基于专门查询编码器的最近提出的Vec2Text模型来实现的，利用了会话和查询嵌入在现有对话式密集检索中共享相同空间的事实。为了进一步增强可解释性，我们建议将外部可解释的查询重写纳入转换过程中。

    arXiv:2402.12774v1 Announce Type: new  Abstract: Conversational dense retrieval has shown to be effective in conversational search. However, a major limitation of conversational dense retrieval is their lack of interpretability, hindering intuitive understanding of model behaviors for targeted improvements. This paper presents CONVINV, a simple yet effective approach to shed light on interpretable conversational dense retrieval models. CONVINV transforms opaque conversational session embeddings into explicitly interpretable text while faithfully maintaining their original retrieval performance as much as possible. Such transformation is achieved by training a recently proposed Vec2Text model based on the ad-hoc query encoder, leveraging the fact that the session and query embeddings share the same space in existing conversational dense retrieval. To further enhance interpretability, we propose to incorporate external interpretable query rewrites into the transformation process. Extensi
    
[^3]: 重写代码：一种用于大型语言模型增强代码搜索的简单方法

    Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search. (arXiv:2401.04514v1 [cs.SE])

    [http://arxiv.org/abs/2401.04514](http://arxiv.org/abs/2401.04514)

    本论文提出一种扩展的生成增强检索（GAR）框架，通过对代码进行重写来解决代码搜索中存在的风格不匹配问题，实验结果表明该方法显著提高了检索准确性。

    

    在代码搜索中，生成增强检索（GAR）框架是一种有前景的策略，通过生成示例代码片段来增强查询，以解决代码片段和自然语言查询之间的主要模态不匹配问题，尤其是在大型语言模型（LLM）展示了代码生成能力的情况下。然而，我们的初步调查发现，LLM增强框架所提供的改进有一定的限制。这种限制可能是因为生成的代码，尽管在功能上准确，但在代码库中与基准代码之间经常显示出明显的风格偏差。在本文中，我们扩展了基础GAR框架，并提出了一种简单而有效的方法，通过对代码库中的代码进行重写（ReCo）来进行风格规范化。实验结果表明，ReCo显著提高了检索准确性。

    In code search, the Generation-Augmented Retrieval (GAR) framework, which generates exemplar code snippets to augment queries, has emerged as a promising strategy to address the principal challenge of modality misalignment between code snippets and natural language queries, particularly with the demonstrated code generation capabilities of Large Language Models (LLMs). Nevertheless, our preliminary investigations indicate that the improvements conferred by such an LLM-augmented framework are somewhat constrained. This limitation could potentially be ascribed to the fact that the generated codes, albeit functionally accurate, frequently display a pronounced stylistic deviation from the ground truth code in the codebase. In this paper, we extend the foundational GAR framework and propose a simple yet effective method that additionally Rewrites the Code (ReCo) within the codebase for style normalization. Experimental results demonstrate that ReCo significantly boosts retrieval accuracy ac
    
[^4]: TSRankLLM: 一种用于文本排序的两阶段LLM适应方法

    TSRankLLM: A Two-Stage Adaptation of LLMs for Text Ranking. (arXiv:2311.16720v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2311.16720](http://arxiv.org/abs/2311.16720)

    TSRankLLM提出了一种两阶段适应方法用于文本排序，通过连续预训练和改进的优化策略，实现了更好的性能。

    

    文本排序是各种信息检索应用中的关键任务，最近预训练语言模型（PLMs），特别是大型语言模型（LLMs）的成功引起了人们对其在文本排序中的应用的兴趣。为了消除PLMs和文本排序之间的不匹配问题，许多学者已经广泛探索了使用有监督排序数据进行微调的方法。然而，以前的研究主要集中在仅编码器和编码器-解码器PLMs上，缺乏对仅解码器LLM的研究。一个例外是RankLLaMA，它建议直接使用有监督的微调（SFT）来全面探索LLaMA。在我们的工作中，我们认为采用两阶段渐进范式会更有益。首先，我们建议使用大规模弱监督语料库对LLMs进行连续预训练（CPT）。其次，我们执行与RankLLaMA一致的SFT，并进一步提出了改进的优化策略。我们在多个基准测试上的实验结果表明我们方法具有卓越的性能。

    Text ranking is a critical task in various information retrieval applications, and the recent success of pre-trained language models (PLMs), especially large language models (LLMs), has sparked interest in their application to text ranking. To eliminate the misalignment between PLMs and text ranking, fine-tuning with supervised ranking data has been widely explored. However, previous studies focus mainly on encoder-only and encoder-decoder PLMs, and decoder-only LLM research is still lacking. An exception to this is RankLLaMA, which suggests direct supervised fine-tuning (SFT) to explore LLaMA fully. In our work, we argue that a two-stage progressive paradigm would be more beneficial. First, we suggest continual pre-training (CPT) on LLMs by using a large-scale weakly-supervised corpus. Second, we perform SFT consistent with RankLLaMA, and propose an improved optimization strategy further. Our experimental results on multiple benchmarks demonstrate the superior performance of our metho
    
[^5]: Mixer: 应用于工业应用的图像到跨模态检索学习

    Mixer: Image to Multi-Modal Retrieval Learning for Industrial Application. (arXiv:2305.03972v1 [cs.IR])

    [http://arxiv.org/abs/2305.03972](http://arxiv.org/abs/2305.03972)

    提出了一种新的可伸缩高效的图像到跨模态检索范式Mixer，解决了领域差距、跨模态数据对齐和融合、繁杂的数据训练标签以及海量查询和及时响应等问题。

    

    跨模态检索一直是电子商务平台和内容分享社交媒体中普遍存在的需求，其中查询是一张图片，文档是具有图片和文本描述的项目。然而，目前这种检索任务仍面临诸多挑战，包括领域差距、跨模态数据对齐和融合、繁杂的数据训练标签以及海量查询和及时响应等问题。为此，我们提出了一种名为Mixer的新型可伸缩和高效的图像查询到跨模态检索学习范式。Mixer通过自适应地整合多模态数据、更高效地挖掘偏斜和嘈杂的数据，并可扩展到高负载量，解决了这些问题。

    Cross-modal retrieval, where the query is an image and the doc is an item with both image and text description, is ubiquitous in e-commerce platforms and content-sharing social media. However, little research attention has been paid to this important application. This type of retrieval task is challenging due to the facts: 1)~domain gap exists between query and doc. 2)~multi-modality alignment and fusion. 3)~skewed training data and noisy labels collected from user behaviors. 4)~huge number of queries and timely responses while the large-scale candidate docs exist. To this end, we propose a novel scalable and efficient image query to multi-modal retrieval learning paradigm called Mixer, which adaptively integrates multi-modality data, mines skewed and noisy data more efficiently and scalable to high traffic. The Mixer consists of three key ingredients: First, for query and doc image, a shared encoder network followed by separate transformation networks are utilized to account for their
    
[^6]: RLTP算法：用于预加载广告中的延迟印象建模的强化学习算法

    RLTP: Reinforcement Learning to Pace for Delayed Impression Modeling in Preloaded Ads. (arXiv:2302.02592v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.02592](http://arxiv.org/abs/2302.02592)

    RLTP算法是一个强化学习算法，用于解决广告预加载过程中的延迟印象现象。

    

    为了增加品牌知名度，许多广告商与广告平台签订合同购买广告流量，然后将广告投放到目标受众中。在整个广告投放期间，广告商通常希望广告获得特定的印象数，并期望广告展示的效果越好越好（如高点击率）。广告平台通过实时调整流量请求的选择概率来满足需求。然而，发布者的策略也会影响广告投放过程，这是广告平台无法控制的。预加载是许多类型广告（如视频广告）的常用策略，以确保在流量请求后显示的响应时间是合理的，这将导致延迟印象现象。传统的配速算法无法很好地处理预加载的特性，因为它们依赖于即时反馈信号。

    To increase brand awareness, many advertisers conclude contracts with advertising platforms to purchase traffic and then deliver advertisements to target audiences. In a whole delivery period, advertisers usually desire a certain impression count for the ads, and they also expect that the delivery performance is as good as possible (e.g., obtaining high click-through rate). Advertising platforms employ pacing algorithms to satisfy the demands via adjusting the selection probabilities to traffic requests in real-time. However, the delivery procedure is also affected by the strategies from publishers, which cannot be controlled by advertising platforms. Preloading is a widely used strategy for many types of ads (e.g., video ads) to make sure that the response time for displaying after a traffic request is legitimate, which results in delayed impression phenomenon. Traditional pacing algorithms cannot handle the preloading nature well because they rely on immediate feedback signals, and m
    

