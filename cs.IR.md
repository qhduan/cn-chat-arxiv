# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring Large Language Models for Code Explanation.](http://arxiv.org/abs/2310.16673) | 本论文研究了使用大型语言模型（LLMs）为代码片段生成自然语言摘要的任务，发现代码LLMs优于通用对应模型，在处理具有不相似分布的数据集时，零样本方法可以得到更好的结果。 |
| [^2] | [Distributionally Robust Unsupervised Dense Retrieval Training on Web Graphs.](http://arxiv.org/abs/2310.16605) | 本论文提出了一种无监督的密集检索模型Web-DRO，它利用网络结构进行聚类并重新加权，在无监督场景中显著提高了检索效果。群组分布鲁棒优化方法指导模型对高对比损失的群组分配更多权重，在训练过程中更加关注最坏情况。实验结果表明，结合URL信息的网络图训练能达到最佳的聚类性能。 |
| [^3] | [Model-enhanced Contrastive Reinforcement Learning for Sequential Recommendation.](http://arxiv.org/abs/2310.16566) | 这项研究提出了一种模型增强的对比强化学习方法，用于解决推荐系统中的数据稀疏和过估计问题。 |
| [^4] | [Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph.](http://arxiv.org/abs/2310.16452) | 本文提出了一个名为PEARLM的方法，通过语言建模开展基于路径的知识图谱推荐，解决了现有方法中对预训练知识图谱嵌入的依赖以及未充分利用实体和关系之间相互依赖性的问题，还避免了生成不准确的解释。实验结果表明，与现有方法相比，我们的方法效果显著。 |
| [^5] | [Multiple Key-value Strategy in Recommendation Systems Incorporating Large Language Model.](http://arxiv.org/abs/2310.16409) | 该论文研究了将推荐系统与大型语言模型结合以实现顺序推荐的问题。现有工作主要考虑单键情况，而忽略了多键值数据的重要性。本研究的贡献在于解决了实际应用中多键值数据的推荐问题。 |
| [^6] | [URL-BERT: Training Webpage Representations via Social Media Engagements.](http://arxiv.org/abs/2310.16303) | URL-BERT是一种通过社交媒体互动训练网页表示的方法，通过引入新的预训练目标和对比目标，实现了对URL和网页的更好理解和表示。 |
| [^7] | [Context-aware feature attribution through argumentation.](http://arxiv.org/abs/2310.16157) | 本论文提出了一种基于论证的上下文感知特征归因方法，以解决机器学习和数据分析中特征归因的挑战。该方法利用广义可加模型和梯度方法与替代模型相结合，同时考虑用户的背景信息，从而提高了归因的准确性和解释性。 |
| [^8] | [Clinfo.ai: An Open-Source Retrieval-Augmented Large Language Model System for Answering Medical Questions using Scientific Literature.](http://arxiv.org/abs/2310.16146) | Clinfo.ai是一个开源的系统，使用科学文献回答医学问题。研究人员提出了一个信息检索和抽象概括任务，发布了相应的数据集，并进行了评估。 |
| [^9] | [Context-aware explainable recommendations over knowledge graphs.](http://arxiv.org/abs/2310.16141) | 本文提出了CA-KGCN，一个基于上下文的推荐系统框架，能够将知识图谱中的语义关系纳入建模，提高推荐准确性和可解释性。 |
| [^10] | [TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction.](http://arxiv.org/abs/2310.15556) | TCRA-LLM是通过概述压缩和语义压缩两种方法来减少商业大型语言模型推理成本的方案。 |
| [^11] | [Retrieve Anything To Augment Large Language Models.](http://arxiv.org/abs/2310.07554) | 这项工作提出了一种新的方法，即LLM-Embedder，通过一个统一的嵌入模型全面支持LLMs的多样化检索增强需求。 |
| [^12] | [TELeR: A General Taxonomy of LLM Prompts for Benchmarking Complex Tasks.](http://arxiv.org/abs/2305.11430) | 本文提出了一个通用分类法，可以用来设计具有特定属性的提示来执行各种复杂任务，从而解决了LLM在执行复杂任务方面的性能变异巨大的问题。 |
| [^13] | [Sheaf Neural Networks for Graph-based Recommender Systems.](http://arxiv.org/abs/2304.09097) | 基于Sheaf神经网络的模型提出了一种新的向量空间表示方法，使得其在基准推荐任务上获得最先进的性能表现。 |

# 详细

[^1]: 探索用于代码解释的大型语言模型

    Exploring Large Language Models for Code Explanation. (arXiv:2310.16673v1 [cs.SE])

    [http://arxiv.org/abs/2310.16673](http://arxiv.org/abs/2310.16673)

    本论文研究了使用大型语言模型（LLMs）为代码片段生成自然语言摘要的任务，发现代码LLMs优于通用对应模型，在处理具有不相似分布的数据集时，零样本方法可以得到更好的结果。

    

    自动化代码文档通过解释性文本在代码理解方面可能非常有益。大型语言模型（LLMs）在自然语言处理方面取得了显著的进展，特别是在软件工程任务中，如代码生成和代码摘要。本研究具体研究了使用各种LLMs为代码片段生成自然语言摘要的任务。研究结果表明，代码LLMs优于其通用对应模型，并且当处理具有训练集和测试集之间分布不相似的数据集时，零样本方法产生更好的结果。

    Automating code documentation through explanatory text can prove highly beneficial in code understanding. Large Language Models (LLMs) have made remarkable strides in Natural Language Processing, especially within software engineering tasks such as code generation and code summarization. This study specifically delves into the task of generating natural-language summaries for code snippets, using various LLMs. The findings indicate that Code LLMs outperform their generic counterparts, and zero-shot methods yield superior results when dealing with datasets with dissimilar distributions between training and testing sets.
    
[^2]: 基于网络图的分布鲁棒无监督密集检索训练

    Distributionally Robust Unsupervised Dense Retrieval Training on Web Graphs. (arXiv:2310.16605v1 [cs.IR])

    [http://arxiv.org/abs/2310.16605](http://arxiv.org/abs/2310.16605)

    本论文提出了一种无监督的密集检索模型Web-DRO，它利用网络结构进行聚类并重新加权，在无监督场景中显著提高了检索效果。群组分布鲁棒优化方法指导模型对高对比损失的群组分配更多权重，在训练过程中更加关注最坏情况。实验结果表明，结合URL信息的网络图训练能达到最佳的聚类性能。

    

    本文介绍了Web-DRO，一种基于网络结构进行聚类并在对比训练期间重新加权的无监督密集检索模型。具体而言，我们首先利用网络图链接并对锚点-文档对进行对比训练，训练一个嵌入模型用于聚类。然后，我们使用群组分布鲁棒优化方法来重新加权不同的锚点-文档对群组，这指导模型将更多权重分配给对比损失更高的群组，并在训练过程中更加关注最坏情况。在MS MARCO和BEIR上的实验表明，我们的模型Web-DRO在无监督场景中显著提高了检索效果。对聚类技术的比较表明，结合URL信息的网络图训练能达到最佳的聚类性能。进一步分析证实了群组权重的稳定性和有效性，表明了一致的模型偏好以及对有价值文档的有效加权。

    This paper introduces Web-DRO, an unsupervised dense retrieval model, which clusters documents based on web structures and reweights the groups during contrastive training. Specifically, we first leverage web graph links and contrastively train an embedding model for clustering anchor-document pairs. Then we use Group Distributional Robust Optimization to reweight different clusters of anchor-document pairs, which guides the model to assign more weights to the group with higher contrastive loss and pay more attention to the worst case during training. Our experiments on MS MARCO and BEIR show that our model, Web-DRO, significantly improves the retrieval effectiveness in unsupervised scenarios. A comparison of clustering techniques shows that training on the web graph combining URL information reaches optimal performance on clustering. Further analysis confirms that group weights are stable and valid, indicating consistent model preferences as well as effective up-weighting of valuable 
    
[^3]: 模型增强的对比强化学习用于序列推荐

    Model-enhanced Contrastive Reinforcement Learning for Sequential Recommendation. (arXiv:2310.16566v1 [cs.IR])

    [http://arxiv.org/abs/2310.16566](http://arxiv.org/abs/2310.16566)

    这项研究提出了一种模型增强的对比强化学习方法，用于解决推荐系统中的数据稀疏和过估计问题。

    

    强化学习已经广泛应用于推荐系统，因为其潜力在于优化用户的长期参与度。从强化学习的角度来看，推荐可以被形式化为马尔可夫决策过程(MDP)，其中推荐系统(代理)可以与用户(环境)进行交互，并获得反馈(奖励信号)。然而，出于对用户体验和实现复杂性的考虑，进行在线交互是不切实际的，我们只能使用包含有限奖励信号和状态转换的离线数据集来训练RL推荐者。因此，奖励信号和状态转换的数据稀疏问题非常严重，而这一问题一直被现有的RL推荐系统忽视。更糟糕的是，RL方法通过试错模式来学习，但在隐式反馈推荐任务中无法获得负反馈，进一步加剧了离线RL推荐者的过估计问题。为了解决这些挑战，我们提出了一种模型增强的对比强化学习方法。

    Reinforcement learning (RL) has been widely applied in recommendation systems due to its potential in optimizing the long-term engagement of users. From the perspective of RL, recommendation can be formulated as a Markov decision process (MDP), where recommendation system (agent) can interact with users (environment) and acquire feedback (reward signals).However, it is impractical to conduct online interactions with the concern on user experience and implementation complexity, and we can only train RL recommenders with offline datasets containing limited reward signals and state transitions. Therefore, the data sparsity issue of reward signals and state transitions is very severe, while it has long been overlooked by existing RL recommenders.Worse still, RL methods learn through the trial-and-error mode, but negative feedback cannot be obtained in implicit feedback recommendation tasks, which aggravates the overestimation problem of offline RL recommender. To address these challenges, 
    
[^4]: 可解释的基于路径的知识图推荐中的忠实路径语言建模

    Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph. (arXiv:2310.16452v1 [cs.IR])

    [http://arxiv.org/abs/2310.16452](http://arxiv.org/abs/2310.16452)

    本文提出了一个名为PEARLM的方法，通过语言建模开展基于路径的知识图谱推荐，解决了现有方法中对预训练知识图谱嵌入的依赖以及未充分利用实体和关系之间相互依赖性的问题，还避免了生成不准确的解释。实验结果表明，与现有方法相比，我们的方法效果显著。

    

    针对知识图谱中的路径推理方法在提高推荐系统透明度方面的潜力，本文提出了一种名为PEARLM的新方法，该方法通过语言建模有效捕获用户行为和产品端知识。我们的方法通过语言模型直接从知识图谱上的路径中学习知识图谱嵌入，并将实体和关系统一在同一优化空间中。序列解码的约束保证了路径对知识图谱的忠实性。在两个数据集上的实验证明了我们方法与现有最先进方法的有效性。

    Path reasoning methods over knowledge graphs have gained popularity for their potential to improve transparency in recommender systems. However, the resulting models still rely on pre-trained knowledge graph embeddings, fail to fully exploit the interdependence between entities and relations in the KG for recommendation, and may generate inaccurate explanations. In this paper, we introduce PEARLM, a novel approach that efficiently captures user behaviour and product-side knowledge through language modelling. With our approach, knowledge graph embeddings are directly learned from paths over the KG by the language model, which also unifies entities and relations in the same optimisation space. Constraints on the sequence decoding additionally guarantee path faithfulness with respect to the KG. Experiments on two datasets show the effectiveness of our approach compared to state-of-the-art baselines. Source code and datasets: AVAILABLE AFTER GETTING ACCEPTED.
    
[^5]: 融合大型语言模型的推荐系统中的多键值策略

    Multiple Key-value Strategy in Recommendation Systems Incorporating Large Language Model. (arXiv:2310.16409v1 [cs.IR])

    [http://arxiv.org/abs/2310.16409](http://arxiv.org/abs/2310.16409)

    该论文研究了将推荐系统与大型语言模型结合以实现顺序推荐的问题。现有工作主要考虑单键情况，而忽略了多键值数据的重要性。本研究的贡献在于解决了实际应用中多键值数据的推荐问题。

    

    推荐系统在满足互联网应用中用户信息需求方面发挥着重要作用，通常使用神经网络处理嵌入细节。最近，大型语言模型在计算机视觉和自然语言处理社区取得重大突破。因此，将推荐系统与大型语言模型更好地结合起来成为了新兴的研究方向。尽管一些现有工作对此问题有所贡献，但主要考虑单键情况（如历史交互），特别是在顺序推荐中，多键值数据的情况被简单忽略。然而，多键值数据在实际应用中是主流场景，用户的信息（如年龄、职业等）和物品的信息（如标题、类别等）具有多个键。因此，我们旨在基于多键值数据实现顺序推荐。

    Recommendation system (RS) plays significant roles in matching users information needs for Internet applications, and it usually utilizes the vanilla neural network as the backbone to handle embedding details. Recently, the large language model (LLM) has exhibited emergent abilities and achieved great breakthroughs both in the CV and NLP communities. Thus, it is logical to incorporate RS with LLM better, which has become an emerging research direction. Although some existing works have made their contributions to this issue, they mainly consider the single key situation (e.g. historical interactions), especially in sequential recommendation. The situation of multiple key-value data is simply neglected. This significant scenario is mainstream in real practical applications, where the information of users (e.g. age, occupation, etc) and items (e.g. title, category, etc) has more than one key. Therefore, we aim to implement sequential recommendations based on multiple key-value data by in
    
[^6]: URL-BERT: 通过社交媒体互动训练网页表示

    URL-BERT: Training Webpage Representations via Social Media Engagements. (arXiv:2310.16303v1 [cs.CL])

    [http://arxiv.org/abs/2310.16303](http://arxiv.org/abs/2310.16303)

    URL-BERT是一种通过社交媒体互动训练网页表示的方法，通过引入新的预训练目标和对比目标，实现了对URL和网页的更好理解和表示。

    

    理解和表示网页对于在线社交网络至关重要，用户可以分享和参与URL。常见的语言模型（LM）编码器如BERT可以用于理解和表示网页的文本内容。然而，这些表示可能无法建模网域和URL的主题信息，也无法准确地捕捉它们对社交媒体用户的吸引力。在这项工作中，我们引入了一种新的预训练目标，用于使语言模型适应URL和网页的理解。我们提出的框架包括两个步骤：（1）基于社交媒体上的用户互动学习URL的浅层表示的可扩展图嵌入，以及（2）将LM表示与前述基于图的表示进行对齐的对比目标。我们将这个框架应用到BERT的多语言版本上，得到了模型URL-BERT。我们通过实验证明，我们的持续预训练方法改善了各种任务的网页理解能力。

    Understanding and representing webpages is crucial to online social networks where users may share and engage with URLs. Common language model (LM) encoders such as BERT can be used to understand and represent the textual content of webpages. However, these representations may not model thematic information of web domains and URLs or accurately capture their appeal to social media users. In this work, we introduce a new pre-training objective that can be used to adapt LMs to understand URLs and webpages. Our proposed framework consists of two steps: (1) scalable graph embeddings to learn shallow representations of URLs based on user engagement on social media and (2) a contrastive objective that aligns LM representations with the aforementioned graph-based representation. We apply our framework to the multilingual version of BERT to obtain the model URL-BERT. We experimentally demonstrate that our continued pre-training approach improves webpage understanding on a variety of tasks and 
    
[^7]: 基于论证的上下文感知特征归因

    Context-aware feature attribution through argumentation. (arXiv:2310.16157v1 [cs.LG])

    [http://arxiv.org/abs/2310.16157](http://arxiv.org/abs/2310.16157)

    本论文提出了一种基于论证的上下文感知特征归因方法，以解决机器学习和数据分析中特征归因的挑战。该方法利用广义可加模型和梯度方法与替代模型相结合，同时考虑用户的背景信息，从而提高了归因的准确性和解释性。

    

    特征归因是机器学习和数据分析中的基本任务，涉及确定个别特征或变量对模型输出的贡献。这个过程有助于确定预测结果最重要的特征。特征归因方法的历史可以追溯到广义可加模型 (GAMs)，它通过将因变量和自变量之间的非线性关系纳入模型，扩展了线性回归模型。近年来，基于梯度的方法和替代模型已经被应用于揭示复杂的人工智能 (AI) 系统，但这些方法存在一些局限性。GAMs 往往能够达到较低的准确性，基于梯度的方法很难解释，替代模型通常存在稳定性和保真度问题。此外，大部分现有方法都没有考虑用户的背景，而用户的背景可能会对他们的偏好产生重要影响。为了解决这些限制并推进当前的研究

    Feature attribution is a fundamental task in both machine learning and data analysis, which involves determining the contribution of individual features or variables to a model's output. This process helps identify the most important features for predicting an outcome. The history of feature attribution methods can be traced back to General Additive Models (GAMs), which extend linear regression models by incorporating non-linear relationships between dependent and independent variables. In recent years, gradient-based methods and surrogate models have been applied to unravel complex Artificial Intelligence (AI) systems, but these methods have limitations. GAMs tend to achieve lower accuracy, gradient-based methods can be difficult to interpret, and surrogate models often suffer from stability and fidelity issues. Furthermore, most existing methods do not consider users' contexts, which can significantly influence their preferences. To address these limitations and advance the current s
    
[^8]: Clinfo.ai:用科学文献回答医学问题的开源检索增强型大型语言模型系统

    Clinfo.ai: An Open-Source Retrieval-Augmented Large Language Model System for Answering Medical Questions using Scientific Literature. (arXiv:2310.16146v1 [cs.IR])

    [http://arxiv.org/abs/2310.16146](http://arxiv.org/abs/2310.16146)

    Clinfo.ai是一个开源的系统，使用科学文献回答医学问题。研究人员提出了一个信息检索和抽象概括任务，发布了相应的数据集，并进行了评估。

    

    随着医学文献的快速增长，医生和研究人员很难及时跟上并总结最近的相关发现。虽然现在存在几个基于大型语言模型（LLMs）的闭源摘要工具，但其输出结果缺乏严格和系统的评估。此外，缺乏高质量的数据集和适当的基准任务来评估这些工具。我们通过四个贡献来解决这些问题：我们发布了名为Clinfo.ai的开源WebApp，它基于动态检索的科学文献回答临床问题；我们指定了一个信息检索和抽象概括任务，以评估这种检索增强型LLM系统的性能；我们发布了一个包含200个问题及其对应答案的数据集，我们将其命名为PubMed检索和综述（PubMedRS-200）；并报告了Cli的基准结果。

    The quickly-expanding nature of published medical literature makes it challenging for clinicians and researchers to keep up with and summarize recent, relevant findings in a timely manner. While several closed-source summarization tools based on large language models (LLMs) now exist, rigorous and systematic evaluations of their outputs are lacking. Furthermore, there is a paucity of high-quality datasets and appropriate benchmark tasks with which to evaluate these tools. We address these issues with four contributions: we release Clinfo.ai, an open-source WebApp that answers clinical questions based on dynamically retrieved scientific literature; we specify an information retrieval and abstractive summarization task to evaluate the performance of such retrieval-augmented LLM systems; we release a dataset of 200 questions and corresponding answers derived from published systematic reviews, which we name PubMed Retrieval and Synthesis (PubMedRS-200); and report benchmark results for Cli
    
[^9]: 基于上下文的可解释知识图谱推荐

    Context-aware explainable recommendations over knowledge graphs. (arXiv:2310.16141v1 [cs.IR])

    [http://arxiv.org/abs/2310.16141](http://arxiv.org/abs/2310.16141)

    本文提出了CA-KGCN，一个基于上下文的推荐系统框架，能够将知识图谱中的语义关系纳入建模，提高推荐准确性和可解释性。

    

    知识图谱包含与项目相关的丰富语义关系，将这些语义关系纳入推荐系统有助于探索项目的潜在连接，从而提高预测准确性并增强推荐的可解释性。然而，这种可解释性不适应用户的情境，而情境可以显著影响其偏好。本文提出了CA-KGCN（上下文感知知识图谱卷积网络），这是一个端到端的框架，可以根据用户的情境来建模其偏好，并将与项目相关的丰富语义关系纳入知识图谱中。该框架捕捉用户对不同因素的注意力：上下文和项目特征。具体而言，该框架可以根据特定情境建模用户的偏好并提供适应给定情境的解释。在三个真实世界数据集上的实验证明了我们框架的有效性。

    Knowledge graphs contain rich semantic relationships related to items and incorporating such semantic relationships into recommender systems helps to explore the latent connections of items, thus improving the accuracy of prediction and enhancing the explainability of recommendations. However, such explainability is not adapted to users' contexts, which can significantly influence their preferences. In this work, we propose CA-KGCN (Context-Aware Knowledge Graph Convolutional Network), an end-to-end framework that can model users' preferences adapted to their contexts and can incorporate rich semantic relationships in the knowledge graph related to items. This framework captures users' attention to different factors: contexts and features of items. More specifically, the framework can model users' preferences adapted to their contexts and provide explanations adapted to the given context. Experiments on three real-world datasets show the effectiveness of our framework: modeling users' 
    
[^10]: TCRA-LLM: 用于减少推理成本的令牌压缩检索增强大型语言模型

    TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction. (arXiv:2310.15556v1 [cs.CL])

    [http://arxiv.org/abs/2310.15556](http://arxiv.org/abs/2310.15556)

    TCRA-LLM是通过概述压缩和语义压缩两种方法来减少商业大型语言模型推理成本的方案。

    

    自从ChatGPT发布了API供公众使用以来，构建在商业大型语言模型（LLM）之上的应用程序数量呈指数增长。这种模型的一个流行用法是利用其上下文学习能力并生成响应以回答用户查询，并利用检索增强获得的知识。部署商业检索增强型LLM的一个问题是成本，因为额外检索的上下文大大增加了LLM的输入标记量。为了缓解这个问题，我们提出了一种令牌压缩方案，包括两种方法：概述压缩和语义压缩。第一种方法使用基于T5模型，通过使用包含具有不同长度的样本的自指示数据集进行微调，并通过概述来减少令牌大小。第二种方法通过移除对语义影响较小的词来进一步压缩令牌大小。为了充分评估所提方法的有效性，

    Since ChatGPT released its API for public use, the number of applications built on top of commercial large language models (LLMs) increase exponentially. One popular usage of such models is leveraging its in-context learning ability and generating responses given user queries leveraging knowledge obtained by retrieval augmentation. One problem of deploying commercial retrieval-augmented LLMs is the cost due to the additionally retrieved context that largely increases the input token size of the LLMs. To mitigate this, we propose a token compression scheme that includes two methods: summarization compression and semantic compression. The first method applies a T5-based model that is fine-tuned by datasets generated using self-instruct containing samples with varying lengths and reduce token size by doing summarization. The second method further compresses the token size by removing words with lower impact on the semantic. In order to adequately evaluate the effectiveness of the proposed
    
[^11]: 检索任何内容来增强大型语言模型

    Retrieve Anything To Augment Large Language Models. (arXiv:2310.07554v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.07554](http://arxiv.org/abs/2310.07554)

    这项工作提出了一种新的方法，即LLM-Embedder，通过一个统一的嵌入模型全面支持LLMs的多样化检索增强需求。

    

    大型语言模型(LLMs)面临着由于其在知识、记忆、对齐和行动方面的固有限制而产生的重要挑战。这些挑战不能单靠LLMs自行解决，而应依赖于来自外部世界（如知识库、记忆存储、演示示例和工具）的辅助。检索增强作为LLMs与外部辅助之间的重要机制。然而，传统方法遇到两个紧迫问题。一方面，通用检索器未能适当优化LLMs的检索增强。另一方面，任务特定的检索器缺乏所需的多样性，阻碍其在各种检索增强场景中的性能表现。在这项工作中，我们提出了一种新的方法，即LLM-Embedder，它通过一个统一的嵌入模型全面支持LLMs的多样化检索增强需求。训练这样的统一模型并不容易，由于不同检索增强场景的多样性。

    Large language models (LLMs) face significant challenges stemming from their inherent limitations in knowledge, memory, alignment, and action. These challenges cannot be addressed by LLMs alone, but should rely on assistance from the external world, such as knowledge base, memory store, demonstration examples, and tools. Retrieval augmentation stands as a vital mechanism for bridging the gap between LLMs and the external assistance. However, conventional methods encounter two pressing issues. On the one hand, the general-purpose retrievers are not properly optimized for the retrieval augmentation of LLMs. On the other hand, the task-specific retrievers lack the required versatility, hindering their performance across the diverse retrieval augmentation scenarios.  In this work, we present a novel approach, the LLM-Embedder, which comprehensively supports the diverse retrieval augmentation needs of LLMs with one unified embedding model. Training such a unified model is non-trivial, as va
    
[^12]: TELeR：用于基准测试复杂任务的LLM提示的通用分类法

    TELeR: A General Taxonomy of LLM Prompts for Benchmarking Complex Tasks. (arXiv:2305.11430v1 [cs.AI])

    [http://arxiv.org/abs/2305.11430](http://arxiv.org/abs/2305.11430)

    本文提出了一个通用分类法，可以用来设计具有特定属性的提示来执行各种复杂任务，从而解决了LLM在执行复杂任务方面的性能变异巨大的问题。

    

    尽管LLM在传统对话环境中理解和生成文本时取得了巨大成功，但它们在执行不明确的复杂任务方面的潜力仍然受到很少的研究。本文提出了一种通用分类法，可以用来设计具有特定属性的提示，以执行各种复杂任务，从而解决了使用不同提示类型/风格和提示提供的不同详细程度时LLM性能变化巨大的问题。这个分类法将使未来的基准测试研究能够报告研究中使用的特定提示类别，从而实现跨不同研究的有意义的比较。

    While LLMs have shown great success in understanding and generating text in traditional conversational settings, their potential for performing ill-defined complex tasks is largely under-studied. Indeed, we are yet to conduct comprehensive benchmarking studies with multiple LLMs that are exclusively focused on a complex task. However, conducting such benchmarking studies is challenging because of the large variations in LLMs' performance when different prompt types/styles are used and different degrees of detail are provided in the prompts. To address this issue, the paper proposes a general taxonomy that can be used to design prompts with specific properties in order to perform a wide range of complex tasks. This taxonomy will allow future benchmarking studies to report the specific categories of prompts used as part of the study, enabling meaningful comparisons across different studies. Also, by establishing a common standard through this taxonomy, researchers will be able to draw mo
    
[^13]: 基于Sheaf神经网络的基于图的推荐系统

    Sheaf Neural Networks for Graph-based Recommender Systems. (arXiv:2304.09097v1 [cs.IR])

    [http://arxiv.org/abs/2304.09097](http://arxiv.org/abs/2304.09097)

    基于Sheaf神经网络的模型提出了一种新的向量空间表示方法，使得其在基准推荐任务上获得最先进的性能表现。

    

    近年来，Graph神经网络在许多应用中得到了广泛应用，包括推荐系统。Graph神经网络对其他方法的优越性在于，推荐系统中的许多问题可以自然地建模为图，其中节点可以是用户或项目，边代表偏好关系。 在当前的Graph神经网络方法中，节点用在训练时学习到的静态向量表示。这种静态向量可能只适用于捕捉定义它们的一些用户或项目的微妙差别。为了克服这个限制，我们建议使用最近提出的启发范畴论的模型：Sheaf神经网络。Sheaf神经网络及其连接的拉普拉斯可以通过将每个节点（以及边）与向量空间而不是单个向量相关联来解决上述问题。向量空间表示更丰富，并允许在推理时选择正确的表示。这种方法使我们的模型更具表现力和灵活性，在几个基准推荐任务上实现了最先进的性能。

    Recent progress in Graph Neural Networks has resulted in wide adoption by many applications, including recommendation systems. The reason for Graph Neural Networks' superiority over other approaches is that many problems in recommendation systems can be naturally modeled as graphs, where nodes can be either users or items and edges represent preference relationships. In current Graph Neural Network approaches, nodes are represented with a static vector learned at training time. This static vector might only be suitable to capture some of the nuances of users or items they define. To overcome this limitation, we propose using a recently proposed model inspired by category theory: Sheaf Neural Networks. Sheaf Neural Networks, and its connected Laplacian, can address the previous problem by associating every node (and edge) with a vector space instead than a single vector. The vector space representation is richer and allows picking the proper representation at inference time. This approa
    

