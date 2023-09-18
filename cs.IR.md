# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [When do Generative Query and Document Expansions Fail? A Comprehensive Study Across Methods, Retrievers, and Datasets.](http://arxiv.org/abs/2309.08541) | 通过对11种扩展技术、12个不同分布变化的数据集和24个检索模型的全面分析，我们发现使用大型语言模型进行查询或文档扩展的效果与检索器性能相关，对于弱模型来说扩展提高了分数，但对于强模型来说扩展通常会损害分数。 |
| [^2] | [SilverRetriever: Advancing Neural Passage Retrieval for Polish Question Answering.](http://arxiv.org/abs/2309.08469) | SilverRetriever是一个特为波兰语问答系统开发的神经检索器，通过训练在多种数据集上取得了显著的改进效果，并且与更大的多语种模型具有竞争力。 |
| [^3] | [Explaining Search Result Stances to Opinionated People.](http://arxiv.org/abs/2309.08460) | 这项研究探讨了向有观点的人解释搜索结果立场的效果，发现立场标签和解释可以帮助用户消费更多不同的搜索结果，但没有发现系统性观点改变的证据。 |
| [^4] | [FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning.](http://arxiv.org/abs/2309.08420) | 提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。 |
| [^5] | [Structural Self-Supervised Objectives for Transformers.](http://arxiv.org/abs/2309.08272) | 本论文提出了三种替代BERT掩码语言模型的预训练目标，包括随机标记置换（RTS）、基于簇的随机标记置换（C-RTS）和交换语言建模（SLM），并且证明这些目标在保持性能的同时，需要更少的预训练时间。此外，本论文还提出了一种结构与下游应用匹配的自监督预训练任务，减少了对标记数据的需求。 |
| [^6] | [AdSEE: Investigating the Impact of Image Style Editing on Advertisement Attractiveness.](http://arxiv.org/abs/2309.08159) | 本文研究了图像样式编辑对广告吸引力的影响。通过引入基于StyleGAN的面部语义编辑和反转，并结合传统的视觉和文本特征，我们提出了AdSEE方法，可用于预测在线广告的点击率。通过对QQ-AD数据集的评估，验证了AdSEE的有效性。 |
| [^7] | [Uncertainty-Aware Multi-View Visual Semantic Embedding.](http://arxiv.org/abs/2309.08154) | 这篇论文提出了一种不确定性感知的多视图视觉语义嵌入框架，在图像-文本检索中有效地利用语义信息进行相似性度量，通过引入不确定性感知损失函数，充分利用二进制标签的不确定性，并将整体匹配分解为多个视图-文本匹配。 |
| [^8] | [iHAS: Instance-wise Hierarchical Architecture Search for Deep Learning Recommendation Models.](http://arxiv.org/abs/2309.07967) | iHAS是一个实例级层次结构搜索的推荐模型，通过自动化神经网络架构搜索，可以在每个实例级别上为不同类型的特征选择最优嵌入维度，并考虑到实例之间的异质性。 |
| [^9] | [Differentiable Retrieval Augmentation via Generative Language Modeling for E-commerce Query Intent Classification.](http://arxiv.org/abs/2308.09308) | 本研究提出了一种可微的检索增强方法，通过生成式语言建模，在电子商务查询意图分类任务中显著提升了性能，解决了检索器和下游模型之间的不可微性问题。 |
| [^10] | [Click-aware Structure Transfer with Sample Weight Assignment for Post-Click Conversion Rate Estimation.](http://arxiv.org/abs/2304.01169) | 本论文提出了一种点击感知的结构转移及样本权重分配方法，用于解决后点击转化率预测中的数据稀疏性问题和知识诅咒问题。 |
| [^11] | [Probe: Learning Users' Personalized Projection Bias in Intertemporal Bundle Choices.](http://arxiv.org/abs/2303.06016) | 本文提出了一种新的偏差嵌入式偏好模型——Probe，旨在解决用户在时间跨度的购物选择中的投影偏差和参照点效应，提高决策的有效性和个性化。 |
| [^12] | [SPEC5G: A Dataset for 5G Cellular Network Protocol Analysis.](http://arxiv.org/abs/2301.09201) | SPEC5G是首个公共5G数据集，用于5G蜂窝网络协议的安全性分析和文本摘要。 |

# 详细

[^1]: 生成式查询和文档扩展何时失败？方法、检索器和数据集的全面研究

    When do Generative Query and Document Expansions Fail? A Comprehensive Study Across Methods, Retrievers, and Datasets. (arXiv:2309.08541v1 [cs.IR])

    [http://arxiv.org/abs/2309.08541](http://arxiv.org/abs/2309.08541)

    通过对11种扩展技术、12个不同分布变化的数据集和24个检索模型的全面分析，我们发现使用大型语言模型进行查询或文档扩展的效果与检索器性能相关，对于弱模型来说扩展提高了分数，但对于强模型来说扩展通常会损害分数。

    

    使用大型语言模型（LM）进行查询或文档扩展可以改善信息检索中的泛化能力。然而，目前尚不清楚这些技术是否普遍有益，还是仅在特定设置下有效，例如对于特定的检索模型、数据集领域或查询类型。为了回答这个问题，我们进行了第一次对基于LM的扩展的全面分析。我们发现，检索器性能与扩展的增益之间存在强烈的负相关关系：扩展改善了较弱模型的分数，但通常会损害较强模型的分数。我们展示了这一趋势在11种扩展技术、12个具有不同分布变化的数据集和24个检索模型的一组实验中成立。通过定性错误分析，我们提出了一个假设，即尽管扩展提供了额外的信息（可能改善了召回率），但它们也增加了噪声，使得很难区分出顶级相关文档（从而引入了错误的正例）

    Using large language models (LMs) for query or document expansion can improve generalization in information retrieval. However, it is unknown whether these techniques are universally beneficial or only effective in specific settings, such as for particular retrieval models, dataset domains, or query types. To answer this, we conduct the first comprehensive analysis of LM-based expansion. We find that there exists a strong negative correlation between retriever performance and gains from expansion: expansion improves scores for weaker models, but generally harms stronger models. We show this trend holds across a set of eleven expansion techniques, twelve datasets with diverse distribution shifts, and twenty-four retrieval models. Through qualitative error analysis, we hypothesize that although expansions provide extra information (potentially improving recall), they add additional noise that makes it difficult to discern between the top relevant documents (thus introducing false positiv
    
[^2]: SilverRetriever：提升波兰问答系统的神经通道检索能力

    SilverRetriever: Advancing Neural Passage Retrieval for Polish Question Answering. (arXiv:2309.08469v1 [cs.CL])

    [http://arxiv.org/abs/2309.08469](http://arxiv.org/abs/2309.08469)

    SilverRetriever是一个特为波兰语问答系统开发的神经检索器，通过训练在多种数据集上取得了显著的改进效果，并且与更大的多语种模型具有竞争力。

    

    现代开放领域的问答系统通常依赖于准确和高效的检索组件来找到包含回答问题所需事实的段落。近年来，由于其出色的性能，神经检索器比词汇替代方式更受欢迎。然而，大部分研究都集中在流行语言如英语或中文上，对于其他语言如波兰语，可用的模型很少。在本文中，我们介绍了SilverRetriever，一个基于多种手动标记或弱标记数据集训练的波兰语神经检索器。SilverRetriever在波兰语模型中取得了比其他模型更好的结果，并与更大的多语种模型具有竞争力。与该模型一起，我们还开源了五个新的段落检索数据集。

    Modern open-domain question answering systems often rely on accurate and efficient retrieval components to find passages containing the facts necessary to answer the question. Recently, neural retrievers have gained popularity over lexical alternatives due to their superior performance. However, most of the work concerns popular languages such as English or Chinese. For others, such as Polish, few models are available. In this work, we present SilverRetriever, a neural retriever for Polish trained on a diverse collection of manually or weakly labeled datasets. SilverRetriever achieves much better results than other Polish models and is competitive with larger multilingual models. Together with the model, we open-source five new passage retrieval datasets.
    
[^3]: 向有观点的人解释搜索结果立场

    Explaining Search Result Stances to Opinionated People. (arXiv:2309.08460v1 [cs.IR])

    [http://arxiv.org/abs/2309.08460](http://arxiv.org/abs/2309.08460)

    这项研究探讨了向有观点的人解释搜索结果立场的效果，发现立场标签和解释可以帮助用户消费更多不同的搜索结果，但没有发现系统性观点改变的证据。

    

    人们在形成观点之前使用网络搜索引擎找到信息，这可能导致具有不同影响水平的实际决策。搜索的认知努力可能使有观点的用户容易受到认知偏见的影响，例如确认偏见。在本文中，我们调查立场标签及其解释是否可以帮助用户消费更多不同的搜索结果。我们自动对三个主题（知识产权、校服和无神论）的搜索结果进行分类和标记，分为反对、中立和支持，并为这些标签生成解释。在一项用户研究中（N =203），我们调查了搜索结果立场偏见（平衡 vs 偏见）和解释水平（纯文本、仅标签、标签和解释）是否会影响被点击的搜索结果的多样性。我们发现立场标签和解释可以导致更多样化的搜索结果消费。然而，我们并没有发现系统性观点改变的证据。

    People use web search engines to find information before forming opinions, which can lead to practical decisions with different levels of impact. The cognitive effort of search can leave opinionated users vulnerable to cognitive biases, e.g., the confirmation bias. In this paper, we investigate whether stance labels and their explanations can help users consume more diverse search results. We automatically classify and label search results on three topics (i.e., intellectual property rights, school uniforms, and atheism) as against, neutral, and in favor, and generate explanations for these labels. In a user study (N =203), we then investigate whether search result stance bias (balanced vs biased) and the level of explanation (plain text, label only, label and explanation) influence the diversity of search results clicked. We find that stance labels and explanations lead to a more diverse search result consumption. However, we do not find evidence for systematic opinion change among us
    
[^4]: FedDCSR: 通过解缠表示学习实现联邦跨领域顺序推荐

    FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning. (arXiv:2309.08420v1 [cs.LG])

    [http://arxiv.org/abs/2309.08420](http://arxiv.org/abs/2309.08420)

    提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。

    

    近年来，利用来自多个领域的用户序列数据的跨领域顺序推荐(CSR)受到了广泛关注。然而，现有的CSR方法需要在领域之间共享原始用户数据，这违反了《通用数据保护条例》(GDPR)。因此，有必要将联邦学习(FL)和CSR相结合，充分利用不同领域的知识，同时保护数据隐私。然而，不同领域之间的序列特征异质性对FL的整体性能有显著影响。在本文中，我们提出了FedDCSR，这是一种通过解缠表示学习的新型联邦跨领域顺序推荐框架。具体而言，为了解决不同领域之间的序列特征异质性，我们引入了一种称为领域内-领域间序列表示解缠(SRD)的方法，将用户序列特征解缠成领域共享和领域专属特征。

    Cross-domain Sequential Recommendation (CSR) which leverages user sequence data from multiple domains has received extensive attention in recent years. However, the existing CSR methods require sharing origin user data across domains, which violates the General Data Protection Regulation (GDPR). Thus, it is necessary to combine federated learning (FL) and CSR to fully utilize knowledge from different domains while preserving data privacy. Nonetheless, the sequence feature heterogeneity across different domains significantly impacts the overall performance of FL. In this paper, we propose FedDCSR, a novel federated cross-domain sequential recommendation framework via disentangled representation learning. Specifically, to address the sequence feature heterogeneity across domains, we introduce an approach called inter-intra domain sequence representation disentanglement (SRD) to disentangle the user sequence features into domain-shared and domain-exclusive features. In addition, we design
    
[^5]: Transformer结构自监督目标的研究

    Structural Self-Supervised Objectives for Transformers. (arXiv:2309.08272v1 [cs.CL])

    [http://arxiv.org/abs/2309.08272](http://arxiv.org/abs/2309.08272)

    本论文提出了三种替代BERT掩码语言模型的预训练目标，包括随机标记置换（RTS）、基于簇的随机标记置换（C-RTS）和交换语言建模（SLM），并且证明这些目标在保持性能的同时，需要更少的预训练时间。此外，本论文还提出了一种结构与下游应用匹配的自监督预训练任务，减少了对标记数据的需求。

    

    本论文旨在通过使用无监督原始数据改进自然语言模型的预训练，使其更高效且与下游应用更加一致。在第一部分中，我们引入了三个替代BERT的掩码语言模型（MLM）的预训练目标，分别是随机标记置换（RTS）、基于簇的随机标记置换（C-RTS）和交换语言建模（SLM）。这些目标涉及到标记的交换而不是屏蔽，其中RTS和C-RTS旨在预测标记的原始性，而SLM则预测原始标记的值。结果显示，RTS和C-RTS需要更少的预训练时间，同时保持与MLM可比较的性能。令人惊讶的是，尽管使用了相同的计算预算，SLM在某些任务上的表现优于MLM。在第二部分中，我们提出了一种结构与下游应用匹配的自监督预训练任务，从而减少了对标记数据的需求。我们使用维基百科和CC-News等大型语料库进行训练。

    This thesis focuses on improving the pre-training of natural language models using unsupervised raw data to make them more efficient and aligned with downstream applications.  In the first part, we introduce three alternative pre-training objectives to BERT's Masked Language Modeling (MLM), namely Random Token Substitution (RTS), Cluster-based Random Token Substitution (C-RTS), and Swapped Language Modeling (SLM). These objectives involve token swapping instead of masking, with RTS and C-RTS aiming to predict token originality and SLM predicting the original token values. Results show that RTS and C-RTS require less pre-training time while maintaining performance comparable to MLM. Surprisingly, SLM outperforms MLM on certain tasks despite using the same computational budget.  In the second part, we proposes self-supervised pre-training tasks that align structurally with downstream applications, reducing the need for labeled data. We use large corpora like Wikipedia and CC-News to trai
    
[^6]: AdSEE: 研究图像样式编辑对广告吸引力的影响

    AdSEE: Investigating the Impact of Image Style Editing on Advertisement Attractiveness. (arXiv:2309.08159v1 [cs.CV])

    [http://arxiv.org/abs/2309.08159](http://arxiv.org/abs/2309.08159)

    本文研究了图像样式编辑对广告吸引力的影响。通过引入基于StyleGAN的面部语义编辑和反转，并结合传统的视觉和文本特征，我们提出了AdSEE方法，可用于预测在线广告的点击率。通过对QQ-AD数据集的评估，验证了AdSEE的有效性。

    

    在电子商务网站、社交媒体平台和搜索引擎中，在线广告是重要的元素。随着移动浏览的日益流行，许多在线广告都通过封面图片以及文本描述来吸引用户的注意力。最近的各种研究致力于通过考虑视觉特征来预测在线广告的点击率，或者通过组合最佳的广告元素来增强可见性。本文提出了广告样式编辑和吸引力增强（AdSEE），探讨了广告图像的语义编辑是否会影响或改变在线广告的受欢迎程度。我们引入了基于StyleGAN的面部语义编辑和反转，对广告图像进行训练，并使用基于GAN的面部潜在表示以及传统的视觉和文本特征来预测点击率。通过一个名为QQ-AD的大型数据集，包含20,527个样本，我们对AdSEE进行了评估。

    Online advertisements are important elements in e-commerce sites, social media platforms, and search engines. With the increasing popularity of mobile browsing, many online ads are displayed with visual information in the form of a cover image in addition to text descriptions to grab the attention of users. Various recent studies have focused on predicting the click rates of online advertisements aware of visual features or composing optimal advertisement elements to enhance visibility. In this paper, we propose Advertisement Style Editing and Attractiveness Enhancement (AdSEE), which explores whether semantic editing to ads images can affect or alter the popularity of online advertisements. We introduce StyleGAN-based facial semantic editing and inversion to ads images and train a click rate predictor attributing GAN-based face latent representations in addition to traditional visual and textual features to click rates. Through a large collected dataset named QQ-AD, containing 20,527 
    
[^7]: 不确定性感知的多视图视觉语义嵌入

    Uncertainty-Aware Multi-View Visual Semantic Embedding. (arXiv:2309.08154v1 [cs.CV])

    [http://arxiv.org/abs/2309.08154](http://arxiv.org/abs/2309.08154)

    这篇论文提出了一种不确定性感知的多视图视觉语义嵌入框架，在图像-文本检索中有效地利用语义信息进行相似性度量，通过引入不确定性感知损失函数，充分利用二进制标签的不确定性，并将整体匹配分解为多个视图-文本匹配。

    

    图像-文本检索的关键挑战是有效地利用语义信息来衡量视觉和语言数据之间的相似性。然而，使用实例级的二进制标签，其中每个图像与一个文本配对，无法捕捉不同语义单元之间的多个对应关系，从而导致多模态语义理解中的不确定性。尽管最近的研究通过更复杂的模型结构或预训练技术捕捉了细粒度信息，但很少有研究直接建模对应关系的不确定性以充分利用二进制标签。为了解决这个问题，我们提出了一种不确定性感知的多视图视觉语义嵌入（UAMVSE）框架，该框架将整体图像-文本匹配分解为多个视图-文本匹配。我们的框架引入了一种不确定性感知损失函数（UALoss），通过自适应地建模每个视图-文本对应关系的不确定性来计算每个视图-文本损失的权重。

    The key challenge in image-text retrieval is effectively leveraging semantic information to measure the similarity between vision and language data. However, using instance-level binary labels, where each image is paired with a single text, fails to capture multiple correspondences between different semantic units, leading to uncertainty in multi-modal semantic understanding. Although recent research has captured fine-grained information through more complex model structures or pre-training techniques, few studies have directly modeled uncertainty of correspondence to fully exploit binary labels. To address this issue, we propose an Uncertainty-Aware Multi-View Visual Semantic Embedding (UAMVSE)} framework that decomposes the overall image-text matching into multiple view-text matchings. Our framework introduce an uncertainty-aware loss function (UALoss) to compute the weighting of each view-text loss by adaptively modeling the uncertainty in each view-text correspondence. Different we
    
[^8]: iHAS: 深度学习推荐模型的实例级层次结构搜索

    iHAS: Instance-wise Hierarchical Architecture Search for Deep Learning Recommendation Models. (arXiv:2309.07967v1 [cs.IR])

    [http://arxiv.org/abs/2309.07967](http://arxiv.org/abs/2309.07967)

    iHAS是一个实例级层次结构搜索的推荐模型，通过自动化神经网络架构搜索，可以在每个实例级别上为不同类型的特征选择最优嵌入维度，并考虑到实例之间的异质性。

    

    当前的推荐系统使用具有统一维度的大型嵌入表，导致过拟合、高计算成本和次优的泛化性能。许多技术旨在通过特征选择或嵌入维度搜索来解决这个问题。然而，这些技术通常为所有实例选择一组固定的特征子集或嵌入维度，并将所有实例都输入到一个推荐模型中，而不考虑物品或用户之间的异质性。本文提出了一种新颖的实例级层次结构搜索框架iHAS，它可以自动在实例级别上进行神经网络架构搜索。具体而言，iHAS包括三个阶段:搜索、聚类和重训练。在搜索阶段，通过精心设计的伯努利门和正则化器，iHAS识别出不同字段特征上的最优实例级嵌入维度。在获得这些维度后，聚类阶段将实例分成不同的类别。

    Current recommender systems employ large-sized embedding tables with uniform dimensions for all features, leading to overfitting, high computational cost, and suboptimal generalizing performance. Many techniques aim to solve this issue by feature selection or embedding dimension search. However, these techniques typically select a fixed subset of features or embedding dimensions for all instances and feed all instances into one recommender model without considering heterogeneity between items or users. This paper proposes a novel instance-wise Hierarchical Architecture Search framework, iHAS, which automates neural architecture search at the instance level. Specifically, iHAS incorporates three stages: searching, clustering, and retraining. The searching stage identifies optimal instance-wise embedding dimensions across different field features via carefully designed Bernoulli gates with stochastic selection and regularizers. After obtaining these dimensions, the clustering stage divid
    
[^9]: 可微检索增强通过生成式语言建模的电子商务查询意图分类

    Differentiable Retrieval Augmentation via Generative Language Modeling for E-commerce Query Intent Classification. (arXiv:2308.09308v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2308.09308](http://arxiv.org/abs/2308.09308)

    本研究提出了一种可微的检索增强方法，通过生成式语言建模，在电子商务查询意图分类任务中显著提升了性能，解决了检索器和下游模型之间的不可微性问题。

    

    检索增强通过使用知识检索器和外部语料库来增强下游模型，而不仅仅是增加模型参数的数量，在许多自然语言处理（NLP）任务中，如文本分类、问题回答等方面已经取得了成功。然而，由于两个部分之间的不可微性，现有方法通常通过分别或异步训练检索器和下游模型来导致性能下降，与端到端联合训练相比。在本文中，我们提出了Differentiable Retrieval Augmentation via Generative lANguage modeling（Dragan），通过一种新颖的可微重构来解决这个问题。我们在电子商务搜索中的一个有挑战性的NLP任务上展示了我们提出的方法的有效性，即查询意图分类。实验结果和消融研究均表明，所提出的方法显著且合理地改进了最先进的基准模型。

    Retrieval augmentation, which enhances downstream models by a knowledge retriever and an external corpus instead of by merely increasing the number of model parameters, has been successfully applied to many natural language processing (NLP) tasks such as text classification, question answering and so on. However, existing methods that separately or asynchronously train the retriever and downstream model mainly due to the non-differentiability between the two parts, usually lead to degraded performance compared to end-to-end joint training. In this paper, we propose Differentiable Retrieval Augmentation via Generative lANguage modeling(Dragan), to address this problem by a novel differentiable reformulation. We demonstrate the effectiveness of our proposed method on a challenging NLP task in e-commerce search, namely query intent classification. Both the experimental results and ablation study show that the proposed method significantly and reasonably improves the state-of-the-art basel
    
[^10]: 点击感知的结构转移及样本权重分配在后点击转化率预测中的应用

    Click-aware Structure Transfer with Sample Weight Assignment for Post-Click Conversion Rate Estimation. (arXiv:2304.01169v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2304.01169](http://arxiv.org/abs/2304.01169)

    本论文提出了一种点击感知的结构转移及样本权重分配方法，用于解决后点击转化率预测中的数据稀疏性问题和知识诅咒问题。

    

    后点击转化率（CVR）预测在推荐和广告等工业应用中起着重要作用。传统的CVR方法通常受到数据稀疏性问题的困扰，因为它们仅依赖于用户点击的样本。为解决这个问题，研究人员引入了多任务学习的方法，利用未点击样本并与CTR任务共享特征表示来进行CVR任务。然而，需要注意的是，CVR和CTR任务在本质上是不同的，甚至可能相互矛盾。因此，引入大量CTR信息而不加区分可能会淹没与CVR相关的有价值信息。本文称此现象为知识诅咒问题。为了解决这个问题，我们认为在引入大量辅助信息和保护与CVR相关的有价值信息之间应该实现一种权衡。

    Post-click Conversion Rate (CVR) prediction task plays an essential role in industrial applications, such as recommendation and advertising. Conventional CVR methods typically suffer from the data sparsity problem as they rely only on samples where the user has clicked. To address this problem, researchers have introduced the method of multi-task learning, which utilizes non-clicked samples and shares feature representations of the Click-Through Rate (CTR) task with the CVR task. However, it should be noted that the CVR and CTR tasks are fundamentally different and may even be contradictory. Therefore, introducing a large amount of CTR information without distinction may drown out valuable information related to CVR. This phenomenon is called the curse of knowledge problem in this paper. To tackle this issue, we argue that a trade-off should be achieved between the introduction of large amounts of auxiliary information and the protection of valuable information related to CVR. Hence, w
    
[^11]: Probe：学习用户在时间跨度的捆绑选择中的个性化投影偏差

    Probe: Learning Users' Personalized Projection Bias in Intertemporal Bundle Choices. (arXiv:2303.06016v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.06016](http://arxiv.org/abs/2303.06016)

    本文提出了一种新的偏差嵌入式偏好模型——Probe，旨在解决用户在时间跨度的购物选择中的投影偏差和参照点效应，提高决策的有效性和个性化。

    

    时间跨度的选择需要权衡现在的成本和未来的收益。其中一种具体的选择是决定购买单个物品还是选择包含该物品的捆绑销售方式。以往的研究假设个人对这些选择中涉及的因素有准确的期望。然而，在现实中，用户对这些因素的感知往往存在偏差，导致了非理性和次优的决策。本文重点关注两种常见的偏差：投影偏差和参照点效应，并为此提出了一种新颖的偏差嵌入式偏好模型——Probe。该模型利用加权函数来捕捉用户的投影偏差，利用价值函数来考虑参照点效应，并引入行为经济学中的前景理论来组合加权和价值函数。这使得我们能够确定用户购买捆绑销售的概率，从而提高决策的有效性和个性化。

    Intertemporal choices involve making decisions that require weighing the costs in the present against the benefits in the future. One specific type of intertemporal choice is the decision between purchasing an individual item or opting for a bundle that includes that item. Previous research assumes that individuals have accurate expectations of the factors involved in these choices. However, in reality, users' perceptions of these factors are often biased, leading to irrational and suboptimal decision-making. In this work, we specifically focus on two commonly observed biases: projection bias and the reference-point effect. To address these biases, we propose a novel bias-embedded preference model called Probe. The Probe incorporates a weight function to capture users' projection bias and a value function to account for the reference-point effect, and introduce prospect theory from behavioral economics to combine the weight and value functions. This allows us to determine the probabili
    
[^12]: SPEC5G：用于5G蜂窝网络协议分析的数据集

    SPEC5G: A Dataset for 5G Cellular Network Protocol Analysis. (arXiv:2301.09201v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.09201](http://arxiv.org/abs/2301.09201)

    SPEC5G是首个公共5G数据集，用于5G蜂窝网络协议的安全性分析和文本摘要。

    

    5G是第五代蜂窝网络协议，是最先进的全球无线标准，能够以提高速度和降低延迟的方式连接几乎所有人和物。因此，其发展、分析和安全性非常重要。然而，目前5G协议的开发和安全分析方法都是完全手动的，比如属性提取、协议摘要和协议规范和实现的语义分析。为了减少这种手动工作，本文提出了SPEC5G，这是首个用于自然语言处理研究的公共5G数据集。该数据集包含来自13094份蜂窝网络规范和13个网站的3,547,586个句子，总计134M个单词。通过利用在自然语言处理任务上取得最先进结果的大规模预训练语言模型，我们使用这个数据集进行与安全相关的文本分类和摘要。安全相关的文本分类可以

    5G is the 5th generation cellular network protocol. It is the state-of-the-art global wireless standard that enables an advanced kind of network designed to connect virtually everyone and everything with increased speed and reduced latency. Therefore, its development, analysis, and security are critical. However, all approaches to the 5G protocol development and security analysis, e.g., property extraction, protocol summarization, and semantic analysis of the protocol specifications and implementations are completely manual. To reduce such manual effort, in this paper, we curate SPEC5G the first-ever public 5G dataset for NLP research. The dataset contains 3,547,586 sentences with 134M words, from 13094 cellular network specifications and 13 online websites. By leveraging large-scale pre-trained language models that have achieved state-of-the-art results on NLP tasks, we use this dataset for security-related text classification and summarization. Security-related text classification ca
    

