# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LightLM: A Lightweight Deep and Narrow Language Model for Generative Recommendation.](http://arxiv.org/abs/2310.17488) | LightLM是一种轻量级的基于Transformer的生成推荐模型，通过引入轻量级深窄Transformer架构来实现直接生成推荐项。 |
| [^2] | [FMMRec: Fairness-aware Multimodal Recommendation.](http://arxiv.org/abs/2310.17373) | 本论文提出了一种名为FMMRec的公平感知多模态推荐方法，通过从模态表示中分离敏感和非敏感信息，实现更公平的表示学习。 |
| [^3] | [Exploring the Potential of Generative AI for the World Wide Web.](http://arxiv.org/abs/2310.17370) | 本文探索了生成型人工智能在万维网领域的潜力，特别关注图像生成。我们开发了WebDiffusion工具来模拟基于稳定扩散的万维网，并评估了生成图像质量。 |
| [^4] | [On Surgical Fine-tuning for Language Encoders.](http://arxiv.org/abs/2310.17041) | 本文表明，对于不同的下游语言任务，只对语言编码器的部分层进行细调即可获得接近甚至优于细调所有层的性能。通过提出一种高效度量方法，我们证明了该方法可以选择性微调导致强大下游性能的层。研究突出表明，任务特定信息通常局部化在少数层内，只调整这些层就足够了。 |
| [^5] | [The Word2vec Graph Model for Author Attribution and Genre Detection in Literary Analysis.](http://arxiv.org/abs/2310.16972) | 提出了一种基于Word2vec图模型的文档建模方法，通过捕捉文档的上下文和风格，实现了准确的作者鉴定和体裁检测任务。 |
| [^6] | [Distributionally Robust Unsupervised Dense Retrieval Training on Web Graphs.](http://arxiv.org/abs/2310.16605) | 本论文提出了一种无监督的密集检索模型Web-DRO，它利用网络结构进行聚类并重新加权，在无监督场景中显著提高了检索效果。群组分布鲁棒优化方法指导模型对高对比损失的群组分配更多权重，在训练过程中更加关注最坏情况。实验结果表明，结合URL信息的网络图训练能达到最佳的聚类性能。 |
| [^7] | [Duplicate Question Retrieval and Confirmation Time Prediction in Software Communities.](http://arxiv.org/abs/2309.05035) | 这项研究旨在解决软件社区问答中的问题重复检索和确认时间预测的挑战，特别是在大型软件系统的CQA中，帮助忙碌的专家管理员更高效地处理重复问题。 |
| [^8] | [DocumentNet: Bridging the Data Gap in Document Pre-Training.](http://arxiv.org/abs/2306.08937) | 这项研究提出了DocumentNet方法，通过从Web上收集大规模和弱标注的数据，弥合了文档预训练中的数据差距，并在各类VDER任务中展现了显著的性能提升。 |
| [^9] | [Multi-grained Hypergraph Interest Modeling for Conversational Recommendation.](http://arxiv.org/abs/2305.04798) | 本文提出了一种多粒度超图兴趣建模方法，通过利用历史对话数据丰富当前对话的上下文，从不同角度捕捉用户兴趣。采用超图结构表示复杂的语义关系，建模用户的历史对话会话，捕捉粗粒度的会话级关系。 |

# 详细

[^1]: LightLM: 一种轻量级的基于Transformer的生成推荐模型

    LightLM: A Lightweight Deep and Narrow Language Model for Generative Recommendation. (arXiv:2310.17488v1 [cs.IR])

    [http://arxiv.org/abs/2310.17488](http://arxiv.org/abs/2310.17488)

    LightLM是一种轻量级的基于Transformer的生成推荐模型，通过引入轻量级深窄Transformer架构来实现直接生成推荐项。

    

    本文介绍了LightLM，一种轻量级的基于Transformer的生成推荐模型。在NLP和视觉等各个人工智能子领域中，基于Transformer的生成建模已经变得越来越重要，而生成推荐由于其对个性化生成建模的独特需求，仍处于初级阶段。现有的生成推荐方法通常使用面向NLP的Transformer架构，如T5、GPT、LLaMA和M6，这些模型比较庞大，且并没有专门针对推荐任务进行设计。LightLM通过引入轻量级深窄Transformer架构来解决这个问题，该架构特别适用于直接生成推荐项。这种结构对于直接的生成推荐非常合适，因为输入主要由适合模型容量的短标记组成，语言模型在这个任务上不需要太宽的结构。我们还...

    This paper presents LightLM, a lightweight Transformer-based language model for generative recommendation. While Transformer-based generative modeling has gained importance in various AI sub-fields such as NLP and vision, generative recommendation is still in its infancy due to its unique demand on personalized generative modeling. Existing works on generative recommendation often use NLP-oriented Transformer architectures such as T5, GPT, LLaMA and M6, which are heavy-weight and are not specifically designed for recommendation tasks. LightLM tackles the issue by introducing a light-weight deep and narrow Transformer architecture, which is specifically tailored for direct generation of recommendation items. This structure is especially apt for straightforward generative recommendation and stems from the observation that language model does not have to be too wide for this task, as the input predominantly consists of short tokens that are well-suited for the model's capacity. We also sh
    
[^2]: FMMRec: 公平感知的多模态推荐

    FMMRec: Fairness-aware Multimodal Recommendation. (arXiv:2310.17373v1 [cs.IR])

    [http://arxiv.org/abs/2310.17373](http://arxiv.org/abs/2310.17373)

    本论文提出了一种名为FMMRec的公平感知多模态推荐方法，通过从模态表示中分离敏感和非敏感信息，实现更公平的表示学习。

    

    最近，多模态推荐因为可以有效解决数据稀疏问题并结合各种模态的表示而受到越来越多的关注。尽管多模态推荐在准确性方面表现出色，但引入不同的模态（例如图像、文本和音频）可能会将更多用户的敏感信息（例如性别和年龄）暴露给推荐系统，从而导致更严重的不公平问题。尽管已经有很多关于公平性的努力，但现有的公平性方法要么与多模态情境不兼容，要么由于忽视多模态内容的敏感信息而导致公平性性能下降。为了在多模态推荐中实现反事实公平性，我们提出了一种新颖的公平感知多模态推荐方法（称为FMMRec），通过从模态表示中分离敏感和非敏感信息，并利用分离后的模态表示来指导更公平的表示学习过程。

    Recently, multimodal recommendations have gained increasing attention for effectively addressing the data sparsity problem by incorporating modality-based representations. Although multimodal recommendations excel in accuracy, the introduction of different modalities (e.g., images, text, and audio) may expose more users' sensitive information (e.g., gender and age) to recommender systems, resulting in potentially more serious unfairness issues. Despite many efforts on fairness, existing fairness-aware methods are either incompatible with multimodal scenarios, or lead to suboptimal fairness performance due to neglecting sensitive information of multimodal content. To achieve counterfactual fairness in multimodal recommendations, we propose a novel fairness-aware multimodal recommendation approach (dubbed as FMMRec) to disentangle the sensitive and non-sensitive information from modal representations and leverage the disentangled modal representations to guide fairer representation learn
    
[^3]: 探索生成型人工智能在万维网领域的潜力

    Exploring the Potential of Generative AI for the World Wide Web. (arXiv:2310.17370v1 [cs.AI])

    [http://arxiv.org/abs/2310.17370](http://arxiv.org/abs/2310.17370)

    本文探索了生成型人工智能在万维网领域的潜力，特别关注图像生成。我们开发了WebDiffusion工具来模拟基于稳定扩散的万维网，并评估了生成图像质量。

    

    生成型人工智能（AI）是一种先进的技术，利用生成模型和用户提示能够生成文本、图像和各种媒体内容。在2022年到2023年期间，生成型人工智能在AI电影到聊天机器人等众多应用领域迅速增长。本文深入探讨了生成型人工智能在万维网领域的潜力，特别关注图像生成。网络开发人员已经利用生成型人工智能来辅助编写文本和图像，而网络浏览器未来可能会使用它来本地生成图像，以修复损坏的网页、节省带宽和增强隐私保护。为了探索这一研究领域，我们开发了一款名为WebDiffusion的工具，可以从客户端和服务器的角度模拟由稳定扩散（一种流行的文本到图像模型）驱动的万维网。WebDiffusion还支持用户意见的众包，我们利用这一功能评估了生成图像质量。

    Generative Artificial Intelligence (AI) is a cutting-edge technology capable of producing text, images, and various media content leveraging generative models and user prompts. Between 2022 and 2023, generative AI surged in popularity with a plethora of applications spanning from AI-powered movies to chatbots. In this paper, we delve into the potential of generative AI within the realm of the World Wide Web, specifically focusing on image generation. Web developers already harness generative AI to help crafting text and images, while Web browsers might use it in the future to locally generate images for tasks like repairing broken webpages, conserving bandwidth, and enhancing privacy. To explore this research area, we have developed WebDiffusion, a tool that allows to simulate a Web powered by stable diffusion, a popular text-to-image model, from both a client and server perspective. WebDiffusion further supports crowdsourcing of user opinions, which we use to evaluate the quality and 
    
[^4]: 关于语言编码器的手术微调

    On Surgical Fine-tuning for Language Encoders. (arXiv:2310.17041v1 [cs.CL])

    [http://arxiv.org/abs/2310.17041](http://arxiv.org/abs/2310.17041)

    本文表明，对于不同的下游语言任务，只对语言编码器的部分层进行细调即可获得接近甚至优于细调所有层的性能。通过提出一种高效度量方法，我们证明了该方法可以选择性微调导致强大下游性能的层。研究突出表明，任务特定信息通常局部化在少数层内，只调整这些层就足够了。

    

    细调预训练的神经语言编码器的所有层（使用所有参数或使用参数高效的方法）往往是将其适应于新任务的默认方法。我们展示了证据，对于不同的下游语言任务，仅细调部分层即可获得接近甚至优于细调语言编码器的所有层的性能。我们提出了一种基于Fisher信息矩阵的对角线（FIM评分）的高效度量方法，用于选择用于选择性微调的候选层。我们在GLUE和SuperGLUE任务以及不同的语言编码器上经验性地展示了，这个度量可以有效选择导致强大下游性能的层。我们的工作突出了与给定的下游任务对应的任务特定信息通常局部化在少数层内，只调整这些层对于强大的性能就足够了。

    Fine-tuning all the layers of a pre-trained neural language encoder (either using all the parameters or using parameter-efficient methods) is often the de-facto way of adapting it to a new task. We show evidence that for different downstream language tasks, fine-tuning only a subset of layers is sufficient to obtain performance that is close to and often better than fine-tuning all the layers in the language encoder. We propose an efficient metric based on the diagonal of the Fisher information matrix (FIM score), to select the candidate layers for selective fine-tuning. We show, empirically on GLUE and SuperGLUE tasks and across distinct language encoders, that this metric can effectively select layers leading to a strong downstream performance. Our work highlights that task-specific information corresponding to a given downstream task is often localized within a few layers, and tuning only those is sufficient for strong performance. Additionally, we demonstrate the robustness of the 
    
[^5]: Word2vec图模型在文学分析中的作者鉴定和体裁检测中的应用

    The Word2vec Graph Model for Author Attribution and Genre Detection in Literary Analysis. (arXiv:2310.16972v1 [cs.IR])

    [http://arxiv.org/abs/2310.16972](http://arxiv.org/abs/2310.16972)

    提出了一种基于Word2vec图模型的文档建模方法，通过捕捉文档的上下文和风格，实现了准确的作者鉴定和体裁检测任务。

    

    分析作者和文章的写作风格对支持各种文学分析，如作者鉴定和体裁检测至关重要。多年来，包括文体学、词袋模型和n-gram等丰富的特征集被广泛用于进行此类分析。然而，这些特征的有效性很大程度上取决于特定语言的语言特征和数据集的特点。因此，基于这些特征集的技术无法在不同领域获得期望的结果。本文提出了一种基于Word2vec图模型的文档建模方法，可以准确捕捉文档的上下文和风格。通过使用这些基于Word2vec图的特征，我们进行分类从而进行作者鉴定和体裁检测任务。我们通过详细的实验研究对广泛的文学作品进行了验证，结果表明与传统基于特征的方法相比，这种方法更为有效。我们的代码和数据是公开可用的。

    Analyzing the writing styles of authors and articles is a key to supporting various literary analyses such as author attribution and genre detection. Over the years, rich sets of features that include stylometry, bag-of-words, n-grams have been widely used to perform such analysis. However, the effectiveness of these features largely depends on the linguistic aspects of a particular language and datasets specific characteristics. Consequently, techniques based on these feature sets cannot give desired results across domains. In this paper, we propose a novel Word2vec graph based modeling of a document that can rightly capture both context and style of the document. By using these Word2vec graph based features, we perform classification to perform author attribution and genre detection tasks. Our detailed experimental study with a comprehensive set of literary writings shows the effectiveness of this method over traditional feature based approaches. Our code and data are publicly availa
    
[^6]: 基于网络图的分布鲁棒无监督密集检索训练

    Distributionally Robust Unsupervised Dense Retrieval Training on Web Graphs. (arXiv:2310.16605v1 [cs.IR])

    [http://arxiv.org/abs/2310.16605](http://arxiv.org/abs/2310.16605)

    本论文提出了一种无监督的密集检索模型Web-DRO，它利用网络结构进行聚类并重新加权，在无监督场景中显著提高了检索效果。群组分布鲁棒优化方法指导模型对高对比损失的群组分配更多权重，在训练过程中更加关注最坏情况。实验结果表明，结合URL信息的网络图训练能达到最佳的聚类性能。

    

    本文介绍了Web-DRO，一种基于网络结构进行聚类并在对比训练期间重新加权的无监督密集检索模型。具体而言，我们首先利用网络图链接并对锚点-文档对进行对比训练，训练一个嵌入模型用于聚类。然后，我们使用群组分布鲁棒优化方法来重新加权不同的锚点-文档对群组，这指导模型将更多权重分配给对比损失更高的群组，并在训练过程中更加关注最坏情况。在MS MARCO和BEIR上的实验表明，我们的模型Web-DRO在无监督场景中显著提高了检索效果。对聚类技术的比较表明，结合URL信息的网络图训练能达到最佳的聚类性能。进一步分析证实了群组权重的稳定性和有效性，表明了一致的模型偏好以及对有价值文档的有效加权。

    This paper introduces Web-DRO, an unsupervised dense retrieval model, which clusters documents based on web structures and reweights the groups during contrastive training. Specifically, we first leverage web graph links and contrastively train an embedding model for clustering anchor-document pairs. Then we use Group Distributional Robust Optimization to reweight different clusters of anchor-document pairs, which guides the model to assign more weights to the group with higher contrastive loss and pay more attention to the worst case during training. Our experiments on MS MARCO and BEIR show that our model, Web-DRO, significantly improves the retrieval effectiveness in unsupervised scenarios. A comparison of clustering techniques shows that training on the web graph combining URL information reaches optimal performance on clustering. Further analysis confirms that group weights are stable and valid, indicating consistent model preferences as well as effective up-weighting of valuable 
    
[^7]: 软件社区中的重复问题检索和确认时间预测

    Duplicate Question Retrieval and Confirmation Time Prediction in Software Communities. (arXiv:2309.05035v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2309.05035](http://arxiv.org/abs/2309.05035)

    这项研究旨在解决软件社区问答中的问题重复检索和确认时间预测的挑战，特别是在大型软件系统的CQA中，帮助忙碌的专家管理员更高效地处理重复问题。

    

    由于存在多个平台和用户之间的大规模共享信息，不同领域的社区问答（CQA）正在快速增长。随着这些在线平台的快速增长，大量的存档数据使得管理员难以检索新问题的可能重复，并在适当的时间识别和确认现有问题对作为重复问题。这个问题在类似askubuntu这样的大型软件系统对应的CQA中尤为重要，管理员需要是专家才能理解问题是否为重复。需要注意的是，在这样的CQA平台上，主要挑战在于管理员本身就是专家，因此通常时间非常宝贵，非常忙碌。为了帮助管理员完成任务，在本研究中，我们解决了askubuntu CQA平台上的两个重大问题：（1）给定一个新问题，检索重复问题；（2）重复问题确认的时间预测。

    Community Question Answering (CQA) in different domains is growing at a large scale because of the availability of several platforms and huge shareable information among users. With the rapid growth of such online platforms, a massive amount of archived data makes it difficult for moderators to retrieve possible duplicates for a new question and identify and confirm existing question pairs as duplicates at the right time. This problem is even more critical in CQAs corresponding to large software systems like askubuntu where moderators need to be experts to comprehend something as a duplicate. Note that the prime challenge in such CQA platforms is that the moderators are themselves experts and are therefore usually extremely busy with their time being extraordinarily expensive. To facilitate the task of the moderators, in this work, we have tackled two significant issues for the askubuntu CQA platform: (1) retrieval of duplicate questions given a new question and (2) duplicate question 
    
[^8]: DocumentNet: 在文档预训练中弥合数据差距

    DocumentNet: Bridging the Data Gap in Document Pre-Training. (arXiv:2306.08937v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.08937](http://arxiv.org/abs/2306.08937)

    这项研究提出了DocumentNet方法，通过从Web上收集大规模和弱标注的数据，弥合了文档预训练中的数据差距，并在各类VDER任务中展现了显著的性能提升。

    

    近年来，文档理解任务，特别是富有视觉元素的文档实体检索（VDER），由于在企业人工智能领域的广泛应用，受到了极大关注。然而，由于严格的隐私约束和高昂的标注成本，这些任务的公开可用数据非常有限。更糟糕的是，来自不同数据集的不重叠实体空间妨碍了文档类型之间的知识转移。在本文中，我们提出了一种从Web收集大规模和弱标注的数据的方法，以利于VDER模型的训练。所收集的数据集名为DocumentNet，不依赖于特定的文档类型或实体集，使其适用于所有的VDER任务。目前的DocumentNet包含了30M个文档，涵盖了近400个文档类型，组织成了一个四级本体结构。在一系列广泛采用的VDER任务上进行的实验表明，当将DocumentNet纳入预训练过程时，取得了显著的改进。

    Document understanding tasks, in particular, Visually-rich Document Entity Retrieval (VDER), have gained significant attention in recent years thanks to their broad applications in enterprise AI. However, publicly available data have been scarce for these tasks due to strict privacy constraints and high annotation costs. To make things worse, the non-overlapping entity spaces from different datasets hinder the knowledge transfer between document types. In this paper, we propose a method to collect massive-scale and weakly labeled data from the web to benefit the training of VDER models. The collected dataset, named DocumentNet, does not depend on specific document types or entity sets, making it universally applicable to all VDER tasks. The current DocumentNet consists of 30M documents spanning nearly 400 document types organized in a four-level ontology. Experiments on a set of broadly adopted VDER tasks show significant improvements when DocumentNet is incorporated into the pre-train
    
[^9]: 多粒度超图兴趣建模对话式推荐算法

    Multi-grained Hypergraph Interest Modeling for Conversational Recommendation. (arXiv:2305.04798v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.04798](http://arxiv.org/abs/2305.04798)

    本文提出了一种多粒度超图兴趣建模方法，通过利用历史对话数据丰富当前对话的上下文，从不同角度捕捉用户兴趣。采用超图结构表示复杂的语义关系，建模用户的历史对话会话，捕捉粗粒度的会话级关系。

    

    对话式推荐系统通过自然语言的多轮对话与用户进行交互，旨在为用户的即时信息需求提供高质量的推荐。尽管已经做出了很多有效的对话式推荐系统，但大多数仍然集中在当前对话的上下文信息上，通常会遇到数据稀缺的问题。因此，我们考虑利用历史对话数据来丰富当前对话的有限上下文。在本文中，我们提出了一种新颖的多粒度超图兴趣建模方法，以从不同的角度捕捉复杂历史数据下的用户兴趣。作为核心思想，我们使用超图来表示历史对话中复杂的语义关系。在我们的方法中，我们首先使用超图结构来建模用户的历史对话会话，并形成一个基于会话的超图，该超图捕捉了粗粒度的会话级关系。

    Conversational recommender system (CRS) interacts with users through multi-turn dialogues in natural language, which aims to provide high-quality recommendations for user's instant information need. Although great efforts have been made to develop effective CRS, most of them still focus on the contextual information from the current dialogue, usually suffering from the data scarcity issue. Therefore, we consider leveraging historical dialogue data to enrich the limited contexts of the current dialogue session.  In this paper, we propose a novel multi-grained hypergraph interest modeling approach to capture user interest beneath intricate historical data from different perspectives. As the core idea, we employ hypergraph to represent complicated semantic relations underlying historical dialogues. In our approach, we first employ the hypergraph structure to model users' historical dialogue sessions and form a session-based hypergraph, which captures coarse-grained, session-level relation
    

