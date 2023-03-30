# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Thistle: A Vector Database in Rust.](http://arxiv.org/abs/2303.16780) | Thistle是一个完全功能的向量数据库，旨在解决回答搜索查询中的潜在知识领域问题，已经在MS MARCO数据集上进行了基准测试，并且有助于推进Rust ML生态系统的发展。 |
| [^2] | [A Novel Patent Similarity Measurement Methodology: Semantic Distance and Technological Distance.](http://arxiv.org/abs/2303.16767) | 该研究提出了一种混合方法，用于自动测量专利之间的相似性，同时考虑语义和技术相似性，并且实验证明该方法优于仅考虑语义相似性的方法。 |
| [^3] | [Computationally Efficient Labeling of Cancer Related Forum Posts by Non-Clinical Text Information Retrieval.](http://arxiv.org/abs/2303.16766) | 本研究基于非临床和免费可用的信息，结合分布式计算、文本检索、聚类和分类方法开发了一个能够检索、聚类和展示关于癌症病程信息的计算有效系统。 |
| [^4] | [Dialogue-to-Video Retrieval.](http://arxiv.org/abs/2303.16761) | 本研究提出了一种基于对话的视频检索系统，使用对话作为搜索描述符，有效地提高了视频检索的准确性。 |
| [^5] | [Exploring celebrity influence on public attitude towards the COVID-19 pandemic: social media shared sentiment analysis.](http://arxiv.org/abs/2303.16759) | 本文研究了公众人物在社交媒体上共享的信息对 COVID-19 疫情中的公众情感和大众意见的影响。通过收集和分析推文，发现公众人物的信息对公众情感和大众意见具有显著的影响。 |
| [^6] | [Judicial Intelligent Assistant System: Extracting Events from Divorce Cases to Detect Disputes for the Judge.](http://arxiv.org/abs/2303.16751) | 本文提出了一种基于两轮标注事件提取技术的离婚案件争议检测方法，实现了司法智能助手（JIA）系统，以自动从离婚案件材料中提取重点事件，通过识别其中的共指来对事件进行对齐，并检测冲突。 |
| [^7] | [A Gold Standard Dataset for the Reviewer Assignment Problem.](http://arxiv.org/abs/2303.16750) | 该论文提出了一个用于审稿人分配问题的新数据集，解决了当前算法难以进行原则比较的问题，并提供了基于此数据集的算法比较结果，为利益相关者在选择算法方面提供了一个基础。 |
| [^8] | [Bi-directional Training for Composed Image Retrieval via Text Prompt Learning.](http://arxiv.org/abs/2303.16604) | 本文提出了一种基于文本提示学习和双向训练的组成图像检索方法，可以应用于现有的体系结构，并且在修改文本存在噪声或歧义的情况下特别有效。 |
| [^9] | [Genetic Analysis of Prostate Cancer with Computer Science Methods.](http://arxiv.org/abs/2303.15851) | 本文应用数据科学、机器学习和拓扑网络分析方法对不同转移部位的前列腺癌肿瘤进行了基因分析，筛选出了与前列腺癌转移相关的13个基因，准确率达到了92%。 |
| [^10] | [Clustering Without Knowing How To: Application and Evaluation.](http://arxiv.org/abs/2209.10267) | 该论文介绍了一个用于图像聚类的众包系统，实验证明只通过众包可以获得有意义的数据聚类，而不需要任何机器学习算法。 |
| [^11] | [Cooperative Retriever and Ranker in Deep Recommenders.](http://arxiv.org/abs/2206.14649) | 本文介绍了深度推荐系统中的检索和排名两阶段工作流程。传统方法中，这两个组件都是独立训练或使用简单的级联管道，效果不佳。最近一些工作提出联合训练检索器和排名器，但仍存在许多限制。因此，还需要探索更有效的协作方法。 |

# 详细

[^1]: Thistle: Rust中的向量数据库

    Thistle: A Vector Database in Rust. (arXiv:2303.16780v1 [cs.IR])

    [http://arxiv.org/abs/2303.16780](http://arxiv.org/abs/2303.16780)

    Thistle是一个完全功能的向量数据库，旨在解决回答搜索查询中的潜在知识领域问题，已经在MS MARCO数据集上进行了基准测试，并且有助于推进Rust ML生态系统的发展。

    

    我们介绍了Thistle，一个完全功能的向量数据库。Thistle是Latent Knowledge Use在回答搜索查询方面的分支，这是初创公司和搜索引擎公司的持续研究课题。我们使用数个著名算法实现Thistle，并在MS MARCO数据集上进行基准测试。结果有助于澄清潜在知识领域以及不断增长的Rust ML生态系统。

    We present Thistle, a fully functional vector database. Thistle is an entry into the domain of latent knowledge use in answering search queries, an ongoing research topic at both start-ups and search engine companies. We implement Thistle with several well-known algorithms, and benchmark results on the MS MARCO dataset. Results help clarify the latent knowledge domain as well as the growing Rust ML ecosystem.
    
[^2]: 一种新的专利相似度测量方法：语义距离和技术距离

    A Novel Patent Similarity Measurement Methodology: Semantic Distance and Technological Distance. (arXiv:2303.16767v1 [cs.IR])

    [http://arxiv.org/abs/2303.16767](http://arxiv.org/abs/2303.16767)

    该研究提出了一种混合方法，用于自动测量专利之间的相似性，同时考虑语义和技术相似性，并且实验证明该方法优于仅考虑语义相似性的方法。

    

    测量专利之间的相似性是确保创新的新颖性的关键步骤。然而，目前大多数专利相似度测量方法仍然依赖于专家手动分类专利。另一方面，一些研究提出了自动化方法；然而，大部分自动化方法只关注专利的语义相似性。为了解决这些问题，我们提出了一种混合方法，用于自动测量专利之间的相似性，同时考虑语义和技术的相似性。我们基于专利文本使用BERT测量语义相似性，使用Jaccard相似性计算专利的技术相似性，并通过分配权重来实现混合。我们的评估结果表明，所提出的方法优于仅考虑语义相似度的基准方法。

    Measuring similarity between patents is an essential step to ensure novelty of innovation. However, a large number of methods of measuring the similarity between patents still rely on manual classification of patents by experts. Another body of research has proposed automated methods; nevertheless, most of it solely focuses on the semantic similarity of patents. In order to tackle these limitations, we propose a hybrid method for automatically measuring the similarity between patents, considering both semantic and technological similarities. We measure the semantic similarity based on patent texts using BERT, calculate the technological similarity with IPC codes using Jaccard similarity, and perform hybridization by assigning weights to the two similarity methods. Our evaluation result demonstrates that the proposed method outperforms the baseline that considers the semantic similarity only.
    
[^3]: 用非临床文本信息检索实现肿瘤相关论坛帖子的计算有效标记

    Computationally Efficient Labeling of Cancer Related Forum Posts by Non-Clinical Text Information Retrieval. (arXiv:2303.16766v1 [cs.IR])

    [http://arxiv.org/abs/2303.16766](http://arxiv.org/abs/2303.16766)

    本研究基于非临床和免费可用的信息，结合分布式计算、文本检索、聚类和分类方法开发了一个能够检索、聚类和展示关于癌症病程信息的计算有效系统。

    

    在线上存在着大量关于癌症的信息，但分类和提取有用信息很困难。几乎所有的医疗保健数据处理研究都涉及正式的临床数据，但非临床数据中也有有价值的信息。本研究将分布式计算、文本检索、聚类和分类方法结合成一个连贯、计算有效的系统，基于非临床和免费可用的信息，可以澄清癌症患者的病程。我们开发了一个完全功能的原型，可以从非临床论坛帖子中检索、聚类和展示关于癌症病程的信息。我们评估了三种聚类算法（MR-DBSCAN、DBSCAN和HDBSCAN），并比较了它们在调整后的兰德指数和总运行时间方面的表现，作为检索的帖子数量和邻域半径函数。聚类结果显示，邻域半径对聚类结果有最显著的影响。

    An abundance of information about cancer exists online, but categorizing and extracting useful information from it is difficult. Almost all research within healthcare data processing is concerned with formal clinical data, but there is valuable information in non-clinical data too. The present study combines methods within distributed computing, text retrieval, clustering, and classification into a coherent and computationally efficient system, that can clarify cancer patient trajectories based on non-clinical and freely available information. We produce a fully-functional prototype that can retrieve, cluster and present information about cancer trajectories from non-clinical forum posts. We evaluate three clustering algorithms (MR-DBSCAN, DBSCAN, and HDBSCAN) and compare them in terms of Adjusted Rand Index and total run time as a function of the number of posts retrieved and the neighborhood radius. Clustering results show that neighborhood radius has the most significant impact on c
    
[^4]: 基于对话的视频检索

    Dialogue-to-Video Retrieval. (arXiv:2303.16761v1 [cs.IR])

    [http://arxiv.org/abs/2303.16761](http://arxiv.org/abs/2303.16761)

    本研究提出了一种基于对话的视频检索系统，使用对话作为搜索描述符，有效地提高了视频检索的准确性。

    

    近年来，在社交媒体等网络平台上，人们进行着越来越多的对话。这启发了基于对话的检索的发展，其中基于对话的视频检索对于推荐系统具有越来越大的兴趣。不同于其他视频检索任务，对话到视频检索使用以用户生成的对话为搜索描述符的结构化查询。本文提出了一个新颖的基于对话的视频检索系统，融合了结构化的对话信息。在AVSD数据集上进行的实验表明，我们提出的使用纯文本查询的方法在R@1上比以前的模型提高了15.8%。此外，我们使用对话作为查询的方法，在R@1、R@5和R@10上分别提高了4.2%、6.2%和8.6%，在R@1、R@5和R@10上分别比基准模型提高了0.7%、3.6%和6.0%。

    Recent years have witnessed an increasing amount of dialogue/conversation on the web especially on social media. That inspires the development of dialogue-based retrieval, in which retrieving videos based on dialogue is of increasing interest for recommendation systems. Different from other video retrieval tasks, dialogue-to-video retrieval uses structured queries in the form of user-generated dialogue as the search descriptor. We present a novel dialogue-to-video retrieval system, incorporating structured conversational information. Experiments conducted on the AVSD dataset show that our proposed approach using plain-text queries improves over the previous counterpart model by 15.8% on R@1. Furthermore, our approach using dialogue as a query, improves retrieval performance by 4.2%, 6.2%, 8.6% on R@1, R@5 and R@10 and outperforms the state-of-the-art model by 0.7%, 3.6% and 6.0% on R@1, R@5 and R@10 respectively.
    
[^5]: 探究名人对公众态度影响的研究：基于社交媒体情感分析的 COVID-19 研究

    Exploring celebrity influence on public attitude towards the COVID-19 pandemic: social media shared sentiment analysis. (arXiv:2303.16759v1 [cs.CL])

    [http://arxiv.org/abs/2303.16759](http://arxiv.org/abs/2303.16759)

    本文研究了公众人物在社交媒体上共享的信息对 COVID-19 疫情中的公众情感和大众意见的影响。通过收集和分析推文，发现公众人物的信息对公众情感和大众意见具有显著的影响。

    

    COVID-19 疫情为健康沟通带来了新机遇，增加了公众使用在线渠道获取与健康相关情绪的机会。人们已经转向社交媒体网络分享与 COVID-19 疫情影响相关的情感。本文研究了公众人物（即运动员、政治家、新闻工作者）共享的社交信息在决定整体公共话语方向中的作用。我们从 2020 年 1 月 1 日到 2022 年 3 月 1 日收集了约 1300 万条推特。使用一个经过调优的 DistilRoBERTa 模型计算了每条推文的情绪，该模型用于比较与公众人物提及同时出现的 COVID-19 疫苗相关推特发布。我们的发现表明，在 COVID-19 疫情的前两年里，与公众人物共享的信息同时出现的情感内容具有一致的模式，影响了公众舆论和大众。

    The COVID-19 pandemic has introduced new opportunities for health communication, including an increase in the public use of online outlets for health-related emotions. People have turned to social media networks to share sentiments related to the impacts of the COVID-19 pandemic. In this paper we examine the role of social messaging shared by Persons in the Public Eye (i.e. athletes, politicians, news personnel) in determining overall public discourse direction. We harvested approximately 13 million tweets ranging from 1 January 2020 to 1 March 2022. The sentiment was calculated for each tweet using a fine-tuned DistilRoBERTa model, which was used to compare COVID-19 vaccine-related Twitter posts (tweets) that co-occurred with mentions of People in the Public Eye. Our findings suggest the presence of consistent patterns of emotional content co-occurring with messaging shared by Persons in the Public Eye for the first two years of the COVID-19 pandemic influenced public opinion and larg
    
[^6]: 司法智能助手系统：从离婚案件中提取事件以检测裁判中的争议

    Judicial Intelligent Assistant System: Extracting Events from Divorce Cases to Detect Disputes for the Judge. (arXiv:2303.16751v1 [cs.CL])

    [http://arxiv.org/abs/2303.16751](http://arxiv.org/abs/2303.16751)

    本文提出了一种基于两轮标注事件提取技术的离婚案件争议检测方法，实现了司法智能助手（JIA）系统，以自动从离婚案件材料中提取重点事件，通过识别其中的共指来对事件进行对齐，并检测冲突。

    

    在民事案件的正式程序中，由不同当事人提供的文本资料描述了案件的发展过程。从这些文本材料中提取案件的关键信息并澄清相关当事人的争议焦点是一项困难而必要的任务。本文提出了一种基于两轮标注事件提取技术的离婚案件争议检测方法。我们按照所提出的方法实现了司法智能助手（JIA）系统，以自动从离婚案件材料中提取重点事件，通过识别其中的共指来对事件进行对齐，并检测冲突。

    In formal procedure of civil cases, the textual materials provided by different parties describe the development process of the cases. It is a difficult but necessary task to extract the key information for the cases from these textual materials and to clarify the dispute focus of related parties. Currently, officers read the materials manually and use methods, such as keyword searching and regular matching, to get the target information. These approaches are time-consuming and heavily depending on prior knowledge and carefulness of the officers. To assist the officers to enhance working efficiency and accuracy, we propose an approach to detect disputes from divorce cases based on a two-round-labeling event extracting technique in this paper. We implement the Judicial Intelligent Assistant (JIA) system according to the proposed approach to 1) automatically extract focus events from divorce case materials, 2) align events by identifying co-reference among them, and 3) detect conflicts a
    
[^7]: 一种用于审稿人分配问题的黄金标准数据集

    A Gold Standard Dataset for the Reviewer Assignment Problem. (arXiv:2303.16750v1 [cs.IR])

    [http://arxiv.org/abs/2303.16750](http://arxiv.org/abs/2303.16750)

    该论文提出了一个用于审稿人分配问题的新数据集，解决了当前算法难以进行原则比较的问题，并提供了基于此数据集的算法比较结果，为利益相关者在选择算法方面提供了一个基础。

    

    许多同行评审期刊或会议正在使用或试图使用算法将投稿分配给审稿人。这些自动化方法的关键是“相似度分数”，即对审稿人在审查论文中的专业水平的数值估计，已经提出了许多算法来计算这些分数。然而，这些算法尚未经过有原则的比较，这使得利益相关者难以以基于证据的方式选择算法。比较现有算法和开发更好算法的关键挑战是缺乏公开可用的黄金标准数据，这些数据将用于进行可重复研究。我们通过收集一组新的相似度得分数据来解决这个问题，并将其发布给研究社区。我们的数据集由58位研究人员提供的477个自我报告的专业水平分数组成，用于评估他们先前阅读的论文的审查经验。我们使用这些数据来比较各种算法，并对标准数据集的设计提出了建议。

    Many peer-review venues are either using or looking to use algorithms to assign submissions to reviewers. The crux of such automated approaches is the notion of the "similarity score"--a numerical estimate of the expertise of a reviewer in reviewing a paper--and many algorithms have been proposed to compute these scores. However, these algorithms have not been subjected to a principled comparison, making it difficult for stakeholders to choose the algorithm in an evidence-based manner. The key challenge in comparing existing algorithms and developing better algorithms is the lack of the publicly available gold-standard data that would be needed to perform reproducible research. We address this challenge by collecting a novel dataset of similarity scores that we release to the research community. Our dataset consists of 477 self-reported expertise scores provided by 58 researchers who evaluated their expertise in reviewing papers they have read previously.  We use this data to compare s
    
[^8]: 基于文本提示学习和双向训练的组成图像检索方法

    Bi-directional Training for Composed Image Retrieval via Text Prompt Learning. (arXiv:2303.16604v1 [cs.CV])

    [http://arxiv.org/abs/2303.16604](http://arxiv.org/abs/2303.16604)

    本文提出了一种基于文本提示学习和双向训练的组成图像检索方法，可以应用于现有的体系结构，并且在修改文本存在噪声或歧义的情况下特别有效。

    

    组成图像检索是根据包含参考图像和描述所需更改的修改文本的多模态用户查询来搜索目标图像的方法。现有的解决这个具有挑战性的任务的方法学习从（参考图像，修改文本）对到图像嵌入的映射，然后将其与大型图像语料库进行匹配。本文提出了一种双向训练方案，利用了这种反向查询，并可应用于现有的组成图像检索体系结构。为了编码双向查询，我们在修改文本前面添加一个可学习的令牌，指定查询的方向，然后微调文本嵌入模块的参数。我们没有对网络架构进行其他更改。在两个标准数据集上的实验表明，双向训练在提高组成图像检索性能方面是有效的，特别是在修改文本存在噪声或歧义的情况下。

    Composed image retrieval searches for a target image based on a multi-modal user query comprised of a reference image and modification text describing the desired changes. Existing approaches to solving this challenging task learn a mapping from the (reference image, modification text)-pair to an image embedding that is then matched against a large image corpus. One area that has not yet been explored is the reverse direction, which asks the question, what reference image when modified as describe by the text would produce the given target image? In this work we propose a bi-directional training scheme that leverages such reversed queries and can be applied to existing composed image retrieval architectures. To encode the bi-directional query we prepend a learnable token to the modification text that designates the direction of the query and then finetune the parameters of the text embedding module. We make no other changes to the network architecture. Experiments on two standard datas
    
[^9]: 计算机科学方法在前列腺癌遗传学中的应用

    Genetic Analysis of Prostate Cancer with Computer Science Methods. (arXiv:2303.15851v1 [cs.IR])

    [http://arxiv.org/abs/2303.15851](http://arxiv.org/abs/2303.15851)

    本文应用数据科学、机器学习和拓扑网络分析方法对不同转移部位的前列腺癌肿瘤进行了基因分析，筛选出了与前列腺癌转移相关的13个基因，准确率达到了92%。

    

    转移性前列腺癌是男性最常见的癌症之一。本文采用数据科学、机器学习和拓扑网络分析方法对不同转移部位的前列腺癌肿瘤进行基因分析。文章提出了一般性的基因表达数据预处理和分析方法来过滤显著基因，并采用机器学习模型和次要肿瘤分类来进一步过滤关键基因。最后，本文对不同类型前列腺癌细胞系样本进行了基因共表达网络分析和社区检测。文章筛选出了与前列腺癌转移相关的13个基因，交叉验证下准确率达到了92%。此外，本文还提供了共表达模式的初步见解。

    Metastatic prostate cancer is one of the most common cancers in men. In the advanced stages of prostate cancer, tumours can metastasise to other tissues in the body, which is fatal. In this thesis, we performed a genetic analysis of prostate cancer tumours at different metastatic sites using data science, machine learning and topological network analysis methods. We presented a general procedure for pre-processing gene expression datasets and pre-filtering significant genes by analytical methods. We then used machine learning models for further key gene filtering and secondary site tumour classification. Finally, we performed gene co-expression network analysis and community detection on samples from different prostate cancer secondary site types. In this work, 13 of the 14,379 genes were selected as the most metastatic prostate cancer related genes, achieving approximately 92% accuracy under cross-validation. In addition, we provide preliminary insights into the co-expression patterns
    
[^10]: 不需要知道如何聚类：应用和评估

    Clustering Without Knowing How To: Application and Evaluation. (arXiv:2209.10267v3 [cs.HC] UPDATED)

    [http://arxiv.org/abs/2209.10267](http://arxiv.org/abs/2209.10267)

    该论文介绍了一个用于图像聚类的众包系统，实验证明只通过众包可以获得有意义的数据聚类，而不需要任何机器学习算法。

    

    众包允许在大规模工人群体上运行简单的人类智能任务，可以解决难以制定算法或在合理时间内训练机器学习模型的问题。其中之一就是按不充分的标准对数据进行聚类，这对人类来说很简单，但对机器来说很困难。在这篇演示论文中，我们构建了一个众包系统用于图像聚类，并在https://github.com/Toloka/crowdclustering 上发布其代码，并供大家免费使用。我们在两个不同的图像数据集上进行了实验，即 Zalando 的 FEIDEGGER裙子和 Toloka 鞋子数据集，确认可以仅通过众包获得有意义的数据聚类，而不需要任何机器学习算法。

    Crowdsourcing allows running simple human intelligence tasks on a large crowd of workers, enabling solving problems for which it is difficult to formulate an algorithm or train a machine learning model in reasonable time. One of such problems is data clustering by an under-specified criterion that is simple for humans, but difficult for machines. In this demonstration paper, we build a crowdsourced system for image clustering and release its code under a free license at https://github.com/Toloka/crowdclustering. Our experiments on two different image datasets, dresses from Zalando's FEIDEGGER and shoes from the Toloka Shoes Dataset, confirm that one can yield meaningful clusters with no machine learning algorithms purely with crowdsourcing.
    
[^11]: 深度推荐系统中的协作检索器和排名器

    Cooperative Retriever and Ranker in Deep Recommenders. (arXiv:2206.14649v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.14649](http://arxiv.org/abs/2206.14649)

    本文介绍了深度推荐系统中的检索和排名两阶段工作流程。传统方法中，这两个组件都是独立训练或使用简单的级联管道，效果不佳。最近一些工作提出联合训练检索器和排名器，但仍存在许多限制。因此，还需要探索更有效的协作方法。

    

    深度推荐系统(DRS)在现代网络服务中被广泛应用。为了处理海量网络内容，DRS采用了两阶段工作流程：检索和排名，以生成其推荐结果。检索器旨在高效地从整个项目中选择一小组相关候选项；而排名器通常更精确但时间消耗更大，应进一步从检索候选项中优化最佳项目。传统上，两个组件要么独立训练，要么在简单的级联管道内训练，这容易产生合作效果差的问题。尽管最近一些工作建议联合训练检索器和排名器，但仍存在许多严重限制：训练和推理中的项分布转移、假阴性和排名顺序不对齐等。因此，探索检索器和排名器之间的有效协作仍然是必要的。

    Deep recommender systems (DRS) are intensively applied in modern web services. To deal with the massive web contents, DRS employs a two-stage workflow: retrieval and ranking, to generate its recommendation results. The retriever aims to select a small set of relevant candidates from the entire items with high efficiency; while the ranker, usually more precise but time-consuming, is supposed to further refine the best items from the retrieved candidates. Traditionally, the two components are trained either independently or within a simple cascading pipeline, which is prone to poor collaboration effect. Though some latest works suggested to train retriever and ranker jointly, there still exist many severe limitations: item distribution shift between training and inference, false negative, and misalignment of ranking order. As such, it remains to explore effective collaborations between retriever and ranker.
    

