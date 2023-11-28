# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LFG: A Generative Network for Real-Time Recommendation.](http://arxiv.org/abs/2310.20189) | 本文提出了一种名为LFG的生成网络，用于实现实时的推荐系统。该网络通过深度神经网络动态生成用户的潜在因子，无需重新分解或重新训练。实验结果表明，该网络在提高推荐准确性的同时实现了实时推荐的目标。 |
| [^2] | [Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation.](http://arxiv.org/abs/2310.14318) | 本论文提出了一种名为ICSRec的方法，用于建模用户的潜在意图，以提高顺序推荐的性能。该方法通过对比学习和交叉子序列来准确捕捉用户的意图。 |
| [^3] | [Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language.](http://arxiv.org/abs/2310.13540) | 本研究提出了一种新颖的统一预训练语言模型增强顺序推荐方法（UPSR），旨在构建一个统一的预训练推荐模型用于多领域推荐任务。研究者设计了五个关键指标来指导预训练和微调过程中的文本->物品适应和行为序列->文本序列适应。 |
| [^4] | [Prompt Tuning on Graph-augmented Low-resource Text Classification.](http://arxiv.org/abs/2307.10230) | 本论文提出了一种基于图增强的低资源文本分类模型G2P2，通过预训练和提示的方式，利用图结构的语义关系来提升低资源文本分类的性能。 |
| [^5] | [Text2Cohort: Democratizing the NCI Imaging Data Commons with Natural Language Cohort Discovery.](http://arxiv.org/abs/2305.07637) | Text2Cohort是一个基于大语言模型的工具箱，可以将用户输入转化为IDC数据库查询，促进自然语言队列发现，减少研究人员查询IDC数据库的学习曲线，实现了癌症成像数据的民主化。 |

# 详细

[^1]: LFG：一种用于实时推荐的生成网络

    LFG: A Generative Network for Real-Time Recommendation. (arXiv:2310.20189v1 [cs.IR])

    [http://arxiv.org/abs/2310.20189](http://arxiv.org/abs/2310.20189)

    本文提出了一种名为LFG的生成网络，用于实现实时的推荐系统。该网络通过深度神经网络动态生成用户的潜在因子，无需重新分解或重新训练。实验结果表明，该网络在提高推荐准确性的同时实现了实时推荐的目标。

    

    推荐系统是当今重要的信息技术，结合深度学习的推荐算法已成为该领域的研究热点。通过矩阵分解和梯度下降捕捉潜在特征以适应用户偏好的潜在因子模型（LFM）推动了各种改进推荐准确性的推荐算法的出现。然而，基于LFM的协同过滤推荐模型缺乏灵活性，并且在实时推荐方面存在一些缺点，因为当有新用户到达时需要重新进行矩阵分解和重新训练。针对这一问题，本文创新性地提出了一种Latent Factor Generator (LFG)网络，并将电影推荐作为研究主题。LFG通过深度神经网络动态生成用户的潜在因子，无需重新分解或重新训练。实验结果表明，该模型在提高推荐准确性的同时实现了实时推荐的目标。

    Recommender systems are essential information technologies today, and recommendation algorithms combined with deep learning have become a research hotspot in this field. The recommendation model known as LFM (Latent Factor Model), which captures latent features through matrix factorization and gradient descent to fit user preferences, has given rise to various recommendation algorithms that bring new improvements in recommendation accuracy. However, collaborative filtering recommendation models based on LFM lack flexibility and has shortcomings for real-time recommendations, as they need to redo the matrix factorization and retrain using gradient descent when new users arrive. In response to this, this paper innovatively proposes a Latent Factor Generator (LFG) network, and set the movie recommendation as research theme. The LFG dynamically generates user latent factors through deep neural networks without the need for re-factorization or retrain. Experimental results indicate that the
    
[^2]: 用交叉子序列进行意图对比学习的顺序推荐

    Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation. (arXiv:2310.14318v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.14318](http://arxiv.org/abs/2310.14318)

    本论文提出了一种名为ICSRec的方法，用于建模用户的潜在意图，以提高顺序推荐的性能。该方法通过对比学习和交叉子序列来准确捕捉用户的意图。

    

    用户的购买行为主要受到他们的意图影响（例如，购买装饰用的衣服，购买画画用的画笔等）。建模用户的潜在意图可以显著提高推荐的性能。先前的工作通过考虑辅助信息中的预定义标签或引入随机数据增强来建模用户的意图。然而，辅助信息是稀疏的，并且对于推荐系统并不总是可用的，引入随机数据增强可能会引入噪声，从而改变序列中隐藏的意图。因此，利用用户的意图进行顺序推荐可能具有挑战性，因为它们经常变化且不可观察。本文提出了用于顺序推荐的意图对比学习与交叉子序列（ICSRec），用于建模用户的潜在意图。

    The user purchase behaviors are mainly influenced by their intentions (e.g., buying clothes for decoration, buying brushes for painting, etc.). Modeling a user's latent intention can significantly improve the performance of recommendations. Previous works model users' intentions by considering the predefined label in auxiliary information or introducing stochastic data augmentation to learn purposes in the latent space. However, the auxiliary information is sparse and not always available for recommender systems, and introducing stochastic data augmentation may introduce noise and thus change the intentions hidden in the sequence. Therefore, leveraging user intentions for sequential recommendation (SR) can be challenging because they are frequently varied and unobserved. In this paper, Intent contrastive learning with Cross Subsequences for sequential Recommendation (ICSRec) is proposed to model users' latent intentions. Specifically, ICSRec first segments a user's sequential behaviors
    
[^3]: 全面将多领域预训练推荐建模为语言

    Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language. (arXiv:2310.13540v1 [cs.IR])

    [http://arxiv.org/abs/2310.13540](http://arxiv.org/abs/2310.13540)

    本研究提出了一种新颖的统一预训练语言模型增强顺序推荐方法（UPSR），旨在构建一个统一的预训练推荐模型用于多领域推荐任务。研究者设计了五个关键指标来指导预训练和微调过程中的文本->物品适应和行为序列->文本序列适应。

    

    随着预训练语言模型（PLM）在各种自然语言处理任务中的广泛应用，先驱性工作试图探索将PLM中的通用文本信息与用户历史行为序列中的个性化行为信息相结合，以增强顺序推荐（SR）。然而，尽管输入格式和任务目标存在共性，行为和文本信息之间存在巨大差距，这阻碍了将SR作为语言建模完全建模。为了填补这一差距，我们提出了一种新颖的统一预训练语言模型增强顺序推荐（UPSR）方法，旨在构建一个统一的预训练推荐模型用于多领域推荐任务。我们正式设计了自然性、领域一致性、信息性、噪声和模糊性以及文本长度等五个关键指标，分别用于指导预训练和微调过程中的文本->物品适应和行为序列->文本序列适应。

    With the thriving of pre-trained language model (PLM) widely verified in various of NLP tasks, pioneer efforts attempt to explore the possible cooperation of the general textual information in PLM with the personalized behavioral information in user historical behavior sequences to enhance sequential recommendation (SR). However, despite the commonalities of input format and task goal, there are huge gaps between the behavioral and textual information, which obstruct thoroughly modeling SR as language modeling via PLM. To bridge the gap, we propose a novel Unified pre-trained language model enhanced sequential recommendation (UPSR), aiming to build a unified pre-trained recommendation model for multi-domain recommendation tasks. We formally design five key indicators, namely naturalness, domain consistency, informativeness, noise & ambiguity, and text length, to guide the text->item adaptation and behavior sequence->text sequence adaptation differently for pre-training and fine-tuning 
    
[^4]: 基于图增强的低资源文本分类的Prompt调优

    Prompt Tuning on Graph-augmented Low-resource Text Classification. (arXiv:2307.10230v1 [cs.IR])

    [http://arxiv.org/abs/2307.10230](http://arxiv.org/abs/2307.10230)

    本论文提出了一种基于图增强的低资源文本分类模型G2P2，通过预训练和提示的方式，利用图结构的语义关系来提升低资源文本分类的性能。

    

    文本分类是信息检索中的一个基础问题，有许多实际应用，例如预测在线文章的主题和电子商务产品描述的类别。然而，低资源文本分类，即没有或只有很少标注样本的情况，对监督学习构成了严重问题。与此同时，许多文本数据本质上都建立在网络结构上，例如在线文章的超链接/引用网络和电子商务产品的用户-物品购买网络。这些图结构捕捉了丰富的语义关系，有助于增强低资源文本分类。在本文中，我们提出了一种名为Graph-Grounded Pre-training and Prompting (G2P2)的新模型，以两方面方法解决低资源文本分类问题。在预训练阶段，我们提出了三种基于图交互的对比策略，共同预训练图文模型；在下游分类阶段，我们探索了手工设计的提示信息对模型的影响。

    Text classification is a fundamental problem in information retrieval with many real-world applications, such as predicting the topics of online articles and the categories of e-commerce product descriptions. However, low-resource text classification, with no or few labeled samples, presents a serious concern for supervised learning. Meanwhile, many text data are inherently grounded on a network structure, such as a hyperlink/citation network for online articles, and a user-item purchase network for e-commerce products. These graph structures capture rich semantic relationships, which can potentially augment low-resource text classification. In this paper, we propose a novel model called Graph-Grounded Pre-training and Prompting (G2P2) to address low-resource text classification in a two-pronged approach. During pre-training, we propose three graph interaction-based contrastive strategies to jointly pre-train a graph-text model; during downstream classification, we explore handcrafted 
    
[^5]: Text2Cohort: 自然语言队列发现对癌症影像数据共享平台的民主化

    Text2Cohort: Democratizing the NCI Imaging Data Commons with Natural Language Cohort Discovery. (arXiv:2305.07637v1 [cs.LG])

    [http://arxiv.org/abs/2305.07637](http://arxiv.org/abs/2305.07637)

    Text2Cohort是一个基于大语言模型的工具箱，可以将用户输入转化为IDC数据库查询，促进自然语言队列发现，减少研究人员查询IDC数据库的学习曲线，实现了癌症成像数据的民主化。

    

    影像数据共享平台(IDC)是一个基于云的数据库，为研究人员提供开放获取的癌症成像数据和分析工具，旨在促进医学成像研究中的协作。然而，由于其复杂和技术性质，查询IDC数据库以进行队列发现和访问成像数据对研究人员来说具有显著的学习曲线。我们开发了基于大语言模型（LLM）的Text2Cohort工具箱，通过提示工程将用户输入转化为IDC数据库查询，并将查询的响应返回给用户，以促进自然语言队列发现。此外，实现了自动校正以解决查询中的语法和语义错误，通过将错误传回模型进行解释和校正。我们对50个自然语言用户输入进行了Text2Cohort评估，范围从信息提取到队列发现。结果查询和输出由两位计算机科学家进行了确认。

    The Imaging Data Commons (IDC) is a cloud-based database that provides researchers with open access to cancer imaging data and tools for analysis, with the goal of facilitating collaboration in medical imaging research. However, querying the IDC database for cohort discovery and access to imaging data has a significant learning curve for researchers due to its complex and technical nature. We developed Text2Cohort, a large language model (LLM) based toolkit to facilitate natural language cohort discovery by translating user input into IDC database queries through prompt engineering and returning the query's response to the user. Furthermore, autocorrection is implemented to resolve syntax and semantic errors in queries by passing the errors back to the model for interpretation and correction. We evaluate Text2Cohort on 50 natural language user inputs ranging from information extraction to cohort discovery. The resulting queries and outputs were verified by two computer scientists to me
    

