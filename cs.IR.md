# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An evaluation framework for comparing epidemic intelligence systems.](http://arxiv.org/abs/2303.17431) | 本研究提出了一种新的评估流行病情报系统的框架，通过四个评估目标描述性回顾分析，可以发现面向事件监测系统在流行病监测方面的优缺点。 |
| [^2] | [Methods and advancement of content-based fashion image retrieval: A Review.](http://arxiv.org/abs/2303.17371) | 本综述文章总结了基于内容的时尚图像检索的最新研究，对CBFIR方法进行了分类，并描述了CBFIR研究中常用的数据集和评估措施。 |
| [^3] | [Yes but.. Can ChatGPT Identify Entities in Historical Documents?.](http://arxiv.org/abs/2303.17322) | 本文通过比较ChatGPT和最先进的基于LM的系统，探究了它在历史文献中进行命名实体识别和分类的能力，发现存在多方面的缺陷，包括实体复杂性和提示特定性等。 |
| [^4] | [CSDR-BERT: a pre-trained scientific dataset match model for Chinese Scientific Dataset Retrieval.](http://arxiv.org/abs/2301.12700) | 本文介绍了CSDR-BERT，一种用于汉语科学数据检索的预训练科学数据集匹配模型，采用了预训练和微调范式以及改进的模型结构和优化方法，在公共和自建数据集上均表现出更好的性能。 |
| [^5] | [A Review of Modern Fashion Recommender Systems.](http://arxiv.org/abs/2202.02757) | 本综述综合评估了时尚推荐系统领域的最新研究进展，并分类总结出物品和服装推荐、尺寸推荐和可解释性等方面的研究现状。 |
| [^6] | [MOEF: Modeling Occasion Evolution in Frequency Domain for Promotion-Aware Click-Through Rate Prediction.](http://arxiv.org/abs/2112.13747) | 本文提出了一种新的CTR模型MOEF，它通过在频域中建模场合演变来处理在线分布的不确定性，采用多个专家学习特征表示，取得了在真实世界的电子商务数据集上优于最先进的CTR模型的效果。 |

# 详细

[^1]: 一种比较流行病情报系统的评估框架

    An evaluation framework for comparing epidemic intelligence systems. (arXiv:2303.17431v1 [cs.IR])

    [http://arxiv.org/abs/2303.17431](http://arxiv.org/abs/2303.17431)

    本研究提出了一种新的评估流行病情报系统的框架，通过四个评估目标描述性回顾分析，可以发现面向事件监测系统在流行病监测方面的优缺点。

    

    在流行病情报的背景下，文献中提出了许多面向事件的监测系统，以促进从任何类型的在线信息源中尽早识别和描述潜在健康威胁。每种系统都有其自己的监测定义和优先级，因此选择最适合给定情况的面向事件监测系统对终端用户来说是一个挑战。在本研究中，我们提出了一种新的评估框架来解决这个问题。它首先将原始输入的流行病事件数据转化为一组具有多粒度的归一化事件，然后基于四个评估目标（空间、时间、主题和来源分析）进行描述性回顾分析。我们通过将其应用于一组由不同面向事件监测系统收集的禽流感数据集，并展示了如何利用我们的框架来确定其在流行病监测方面的优缺点。

    In the context of Epidemic Intelligence, many Event-Based Surveillance (EBS) systems have been proposed in the literature to promote the early identification and characterization of potential health threats from online sources of any nature. Each EBS system has its own surveillance definitions and priorities, therefore this makes the task of selecting the most appropriate EBS system for a given situation a challenge for end-users. In this work, we propose a new evaluation framework to address this issue. It first transforms the raw input epidemiological event data into a set of normalized events with multi-granularity, then conducts a descriptive retrospective analysis based on four evaluation objectives: spatial, temporal, thematic and source analysis. We illustrate its relevance by applying it to an Avian Influenza dataset collected by a selection of EBS systems, and show how our framework allows identifying their strengths and drawbacks in terms of epidemic surveillance.
    
[^2]: 基于内容的时尚图像检索方法和进展：综述

    Methods and advancement of content-based fashion image retrieval: A Review. (arXiv:2303.17371v1 [cs.IR])

    [http://arxiv.org/abs/2303.17371](http://arxiv.org/abs/2303.17371)

    本综述文章总结了基于内容的时尚图像检索的最新研究，对CBFIR方法进行了分类，并描述了CBFIR研究中常用的数据集和评估措施。

    

    基于内容的时尚图像检索在我们的日常生活中广泛使用，用于从在线平台上搜索时尚图像或商品。在电子商务购买中，当消费者上传参考图像、带文本的图像、草图或来自日常生活的视觉流时，CBFIR系统可以检索具有相同或可比特征的时尚商品或产品。这降低了CBFIR系统对文本的依赖性，允许更准确直接地搜索所需的时尚产品。考虑到最近的发展，由于多个时尚物品的同时可用性、时尚产品的遮挡和形状变形，CBFIR在现实世界中的视觉搜索仍存在局限性。本文重点介绍了基于图像、带文本的图像、草图和视频的CBFIR方法。因此，我们将CBFIR方法分为四类，即基于图像的CBFIR（增加属性和样式），图像和文本引导、草图引导和视频引导的CBFIR。在本综述文章中，我们总结了CBFIR的最新研究，对CBFIR方法进行了分类，并描述了CBFIR研究中常用的数据集和评估措施。

    Content-based fashion image retrieval (CBFIR) has been widely used in our daily life for searching fashion images or items from online platforms. In e-commerce purchasing, the CBFIR system can retrieve fashion items or products with the same or comparable features when a consumer uploads a reference image, image with text, sketch or visual stream from their daily life. This lowers the CBFIR system reliance on text and allows for a more accurate and direct searching of the desired fashion product. Considering recent developments, CBFIR still has limits when it comes to visual searching in the real world due to the simultaneous availability of multiple fashion items, occlusion of fashion products, and shape deformation. This paper focuses on CBFIR methods with the guidance of images, images with text, sketches, and videos. Accordingly, we categorized CBFIR methods into four main categories, i.e., image-guided CBFIR (with the addition of attributes and styles), image and text-guided, sket
    
[^3]: 能否使用ChatGPT识别历史文献中的实体？

    Yes but.. Can ChatGPT Identify Entities in Historical Documents?. (arXiv:2303.17322v1 [cs.DL])

    [http://arxiv.org/abs/2303.17322](http://arxiv.org/abs/2303.17322)

    本文通过比较ChatGPT和最先进的基于LM的系统，探究了它在历史文献中进行命名实体识别和分类的能力，发现存在多方面的缺陷，包括实体复杂性和提示特定性等。

    

    大型语言模型(LLM)多年来一直被用于现代文档中的实体识别，取得了最先进的性能。最近几个月，会话代理ChatGPT由于其生成听起来合理的答案的能力，在科学界和公众中引起了很多兴趣。本文尝试以零-shot的方式探究ChatGPT在一次性源（例如历史报纸和古典注释）中进行命名实体识别和分类（NERC）任务的能力，并将其与最先进的基于LM的系统进行比较。研究发现，在识别历史文本中的实体方面存在几个缺陷，包括实体注释准则的一致性、实体的复杂性和代码切换以及提示的特定性。此外，由于历史档案对公众不可访问（因此无法在互联网上使用），也影响了ChatGPT的性能。

    Large language models (LLMs) have been leveraged for several years now, obtaining state-of-the-art performance in recognizing entities from modern documents. For the last few months, the conversational agent ChatGPT has "prompted" a lot of interest in the scientific community and public due to its capacity of generating plausible-sounding answers. In this paper, we explore this ability by probing it in the named entity recognition and classification (NERC) task in primary sources (e.g., historical newspapers and classical commentaries) in a zero-shot manner and by comparing it with state-of-the-art LM-based systems. Our findings indicate several shortcomings in identifying entities in historical text that range from the consistency of entity annotation guidelines, entity complexity, and code-switching, to the specificity of prompting. Moreover, as expected, the inaccessibility of historical archives to the public (and thus on the Internet) also impacts its performance.
    
[^4]: CSDR-BERT：一种用于汉语科学数据检索的预训练科学数据集匹配模型

    CSDR-BERT: a pre-trained scientific dataset match model for Chinese Scientific Dataset Retrieval. (arXiv:2301.12700v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.12700](http://arxiv.org/abs/2301.12700)

    本文介绍了CSDR-BERT，一种用于汉语科学数据检索的预训练科学数据集匹配模型，采用了预训练和微调范式以及改进的模型结构和优化方法，在公共和自建数据集上均表现出更好的性能。

    

    随着开放科学运动下开放和共享科学数据集的数量的增加，有效地检索这些数据集是信息检索(IR)研究中的一个关键任务。近年来，大模型的发展，特别是预训练和微调范式，即在大模型上进行预训练并在下游任务上进行微调的范式，为IR匹配任务提供了新的解决方案。在本研究中，我们使用嵌入层中的原始BERT令牌，在模型层中引入SimCSE和K-最近邻方法改进了Sentence-BERT模型结构，使用余弦损失函数在优化阶段优化目标输出。通过比较实验和消融实现，我们的实验结果表明，我们的模型在公共数据集和自建数据集上均优于其他竞争模型。本研究探讨和验证了预训练技术对科学数据集匹配领域的可行性和有效性。

    As the number of open and shared scientific datasets on the Internet increases under the open science movement, efficiently retrieving these datasets is a crucial task in information retrieval (IR) research. In recent years, the development of large models, particularly the pre-training and fine-tuning paradigm, which involves pre-training on large models and fine-tuning on downstream tasks, has provided new solutions for IR match tasks. In this study, we use the original BERT token in the embedding layer, improve the Sentence-BERT model structure in the model layer by introducing the SimCSE and K-Nearest Neighbors method, and use the cosent loss function in the optimization phase to optimize the target output. Our experimental results show that our model outperforms other competing models on both public and self-built datasets through comparative experiments and ablation implementations. This study explores and validates the feasibility and efficiency of pre-training techniques for se
    
[^5]: 现代时尚推荐系统综述

    A Review of Modern Fashion Recommender Systems. (arXiv:2202.02757v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2202.02757](http://arxiv.org/abs/2202.02757)

    本综述综合评估了时尚推荐系统领域的最新研究进展，并分类总结出物品和服装推荐、尺寸推荐和可解释性等方面的研究现状。

    

    近年来，纺织和服装行业蓬勃发展。顾客不再需要亲自去实体店面，排队试穿衣物，因为成千上万的产品现在都可以在在线目录中找到。然而，由于选项太多，一个有效的推荐系统是必不可少的，以便有效地排序、整理并向用户传达相关的产品资料或信息。有效的时尚推荐系统可以显著提高数十亿顾客的购物体验，并增加提供商的销售和收入。本综述的目的是对在服装和时尚产品特定垂直领域运行的推荐系统进行综述。我们确定了时尚推荐系统研究中最紧迫的挑战，并创建了一个分类法，根据它们试图实现的目标（例如，物品或服装推荐、尺寸推荐、可解释性）对文献进行分类。

    The textile and apparel industries have grown tremendously over the last few years. Customers no longer have to visit many stores, stand in long queues, or try on garments in dressing rooms as millions of products are now available in online catalogs. However, given the plethora of options available, an effective recommendation system is necessary to properly sort, order, and communicate relevant product material or information to users. Effective fashion RS can have a noticeable impact on billions of customers' shopping experiences and increase sales and revenues on the provider side.  The goal of this survey is to provide a review of recommender systems that operate in the specific vertical domain of garment and fashion products. We have identified the most pressing challenges in fashion RS research and created a taxonomy that categorizes the literature according to the objective they are trying to accomplish (e.g., item or outfit recommendation, size recommendation, explainability, 
    
[^6]: MOEF:建模频域中的场合演变，实现促销感知的点击率预测

    MOEF: Modeling Occasion Evolution in Frequency Domain for Promotion-Aware Click-Through Rate Prediction. (arXiv:2112.13747v6 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.13747](http://arxiv.org/abs/2112.13747)

    本文提出了一种新的CTR模型MOEF，它通过在频域中建模场合演变来处理在线分布的不确定性，采用多个专家学习特征表示，取得了在真实世界的电子商务数据集上优于最先进的CTR模型的效果。

    

    促销在电子商务中变得越来越重要和普遍，以吸引客户和促进销售，导致场合经常变化，从而驱动用户表现出不同的行为。在这种情况下，由于即将到来的场合分布的不确定性，大多数现有的点击率（CTR）模型无法在在线服务中良好地推广。本文提出了一种新颖的CTR模型，名为MOEF，用于在场合经常变化的情况下进行推荐。首先，我们设计了一个时间序列，其中包括从在线业务场景中生成的场合信号。由于场合信号在频域中更具有区别性，我们对时间窗口应用傅里叶变换，得到一系列频谱，然后通过场合演变层（OEL）进行处理。通过这种方式，可以学习高阶场合表示，以处理在线分布的不确定性。此外，我们采用多个专家来学习特征表示，表示为场合上下文编码（OCE）和模型感知注意力（MAA），以捕捉用户行为和项目特征的不同方面。在真实世界的电子商务数据集上进行的广泛实验证明，MOEF在离线评估和在线服务方面优于最先进的CTR模型。

    Promotions are becoming more important and prevalent in e-commerce to attract customers and boost sales, leading to frequent changes of occasions, which drives users to behave differently. In such situations, most existing Click-Through Rate (CTR) models can't generalize well to online serving due to distribution uncertainty of the upcoming occasion. In this paper, we propose a novel CTR model named MOEF for recommendations under frequent changes of occasions. Firstly, we design a time series that consists of occasion signals generated from the online business scenario. Since occasion signals are more discriminative in the frequency domain, we apply Fourier Transformation to sliding time windows upon the time series, obtaining a sequence of frequency spectrum which is then processed by Occasion Evolution Layer (OEL). In this way, a high-order occasion representation can be learned to handle the online distribution uncertainty. Moreover, we adopt multiple experts to learn feature repres
    

