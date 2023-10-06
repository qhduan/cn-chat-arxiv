# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.](http://arxiv.org/abs/2310.03714) | DSPy是一个编程模型，将LM流水线抽象为文本转换图，通过声明性模块调用LM实现优化，能够解决复杂的推理问题和数学问题等任务。 |
| [^2] | [FASER: Binary Code Similarity Search through the use of Intermediate Representations.](http://arxiv.org/abs/2310.03605) | 本论文提出了一种名为FASER的方法，通过使用中间表示进行二进制代码相似性搜索。该方法可以跨架构地识别函数，并明确编码函数的语义，以支持各种应用场景。 |
| [^3] | [TPDR: A Novel Two-Step Transformer-based Product and Class Description Match and Retrieval Method.](http://arxiv.org/abs/2310.03491) | TPDR是一种基于Transformer的产品和类描述匹配与检索方法，通过注意机制和对比学习来实现语义对应关系的探索。 |
| [^4] | [Personalized Transformer-based Ranking for e-Commerce at Yandex.](http://arxiv.org/abs/2310.03481) | 本文提出了一个基于个性化Transformer的电子商务排名系统，通过优化排名阶段的特征生成，提高了推荐质量。同时，还引入了一种新颖的技术用于解决偏置上下文的问题。 |
| [^5] | [Amazon Books Rating prediction & Recommendation Model.](http://arxiv.org/abs/2310.03200) | 本文利用亚马逊的数据集构建了一个预测图书评分和推荐图书的模型，提供了处理大数据文件、数据工程和构建模型的流程，并使用了各种PySpark机器学习API、超参数调优和交叉验证进行准确性的分析。 |
| [^6] | [Impedance Leakage Vulnerability and its Utilization in Reverse-engineering Embedded Software.](http://arxiv.org/abs/2310.03175) | 这项研究发现了一种新的安全漏洞——阻抗泄漏，通过利用该漏洞可以从嵌入式设备中提取受保护内存中的软件指令。 |
| [^7] | [Multi-Task Learning For Reduced Popularity Bias In Multi-Territory Video Recommendations.](http://arxiv.org/abs/2310.03148) | 本文提出了一种多任务学习技术和自适应上采样方法，用于减少多领域推荐系统中的流行度偏差。通过实验证明，该框架在多个领域中相对增益高达65.27%。 |
| [^8] | [Context-Based Tweet Engagement Prediction.](http://arxiv.org/abs/2310.03147) | 该论文研究了基于上下文的推文参与度预测问题，使用了Twitter的数据集和评估流程，探讨了仅凭上下文是否可以很好地预测推文的参与度可能性。 |
| [^9] | [A Deep Reinforcement Learning Approach for Interactive Search with Sentence-level Feedback.](http://arxiv.org/abs/2310.03043) | 本研究提出了一种利用深度强化学习的交互式搜索方法，该方法通过整合句级反馈信息来提高搜索准确性。通过适应最新的BERT-based模型进行关键句子选择和项目排序，可以获得更满意的搜索结果。 |
| [^10] | [Graph-enhanced Optimizers for Structure-aware Recommendation Embedding Evolution.](http://arxiv.org/abs/2310.03032) | 本文提出了一种新颖的结构感知嵌入演化(SEvo)机制，能够以较低的计算开销将图结构信息注入到嵌入中，从而在现代推荐系统中实现更高效的性能。 |
| [^11] | [SE-PEF: a Resource for Personalized Expert Finding.](http://arxiv.org/abs/2309.11686) | 该论文介绍了SE-PEF，一个用于个性化专家查找的资源。该资源包括超过25万个查询和56.5万个答案，并使用一套丰富的特征来建模用户之间的社交互动。初步实验结果表明SE-PEF适用于评估和训练有效的专家查找模型。 |
| [^12] | [Exploring Social Choice Mechanisms for Recommendation Fairness in SCRUF.](http://arxiv.org/abs/2309.08621) | 本文通过使用社会选择机制，探索了多个多方面公平应用中的选择机制选项，结果显示不同的选择和分配机制会产生不同但一致的公平性/准确性权衡结果，并且多智能体的构成使得系统能够适应用户人口的动态变化。 |
| [^13] | [SpaDE: Improving Sparse Representations using a Dual Document Encoder for First-stage Retrieval.](http://arxiv.org/abs/2209.05917) | SpaDE 是一种利用双重编码器学习文档表示的第一阶段检索模型，可以同时改善词汇匹配和扩展额外术语来支持语义匹配，且在实验中表现优异。 |
| [^14] | [SR-HetGNN:Session-based Recommendation with Heterogeneous Graph Neural Network.](http://arxiv.org/abs/2108.05641) | 本文提出了一种基于异构图神经网络的会话推荐方法SR-HetGNN，通过学习会话嵌入并捕捉匿名用户的特定偏好，以改进会话推荐系统的效果和准确性。 |

# 详细

[^1]: DSPy: 将声明性语言模型调用编译成自我改进的流水线

    DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. (arXiv:2310.03714v1 [cs.CL])

    [http://arxiv.org/abs/2310.03714](http://arxiv.org/abs/2310.03714)

    DSPy是一个编程模型，将LM流水线抽象为文本转换图，通过声明性模块调用LM实现优化，能够解决复杂的推理问题和数学问题等任务。

    

    ML社区正在快速探索用于提示语言模型(LMs)和将它们堆叠成解决复杂任务的流水线的技术。不幸的是，现有的LM流水线通常使用硬编码的"提示模板"来实现，即通过试错发现的冗长字符串。为了更系统地开发和优化LM流水线，我们引入了DSPy，这是一个以文本转换图的形式抽象LM流水线的编程模型，即通过声明性模块调用LM的命令式计算图。DSPy模块是参数化的，这意味着它们可以通过创建和收集示例来学习如何应用提示、微调、增强和推理技术的组合。我们设计了一个编译器，可以优化任何DSPy流水线以最大化给定的度量标准。我们进行了两个案例研究，显示出简洁的DSPy程序可以表达和优化复杂的推理数学问题、登录日志问题等流水线。

    The ML community is rapidly exploring techniques for prompting language models (LMs) and for stacking them into pipelines that solve complex tasks. Unfortunately, existing LM pipelines are typically implemented using hard-coded "prompt templates", i.e. lengthy strings discovered via trial and error. Toward a more systematic approach for developing and optimizing LM pipelines, we introduce DSPy, a programming model that abstracts LM pipelines as text transformation graphs, i.e. imperative computational graphs where LMs are invoked through declarative modules. DSPy modules are parameterized, meaning they can learn (by creating and collecting demonstrations) how to apply compositions of prompting, finetuning, augmentation, and reasoning techniques. We design a compiler that will optimize any DSPy pipeline to maximize a given metric. We conduct two case studies, showing that succinct DSPy programs can express and optimize sophisticated LM pipelines that reason about math word problems, tac
    
[^2]: FASER: 通过中间表示进行二进制代码相似性搜索

    FASER: Binary Code Similarity Search through the use of Intermediate Representations. (arXiv:2310.03605v1 [cs.CR])

    [http://arxiv.org/abs/2310.03605](http://arxiv.org/abs/2310.03605)

    本论文提出了一种名为FASER的方法，通过使用中间表示进行二进制代码相似性搜索。该方法可以跨架构地识别函数，并明确编码函数的语义，以支持各种应用场景。

    

    能够识别跨架构软件中感兴趣的函数对于分析恶意软件、保护软件供应链或进行漏洞研究都是有用的。跨架构二进制代码相似性搜索已在许多研究中探索，并使用了各种不同的数据来源来实现其目标。通常使用的数据来源包括从二进制文件中提取的常见结构，如函数控制流图或二进制级调用图，反汇编过程的输出或动态分析方法的输出。其中一种受到较少关注的数据来源是二进制中间表示。二进制中间表示具有两个有趣的属性：它们的跨架构性质以及明确编码函数的语义以支持下游使用。在本文中，我们提出了一种名为FASER的函数字符串编码表示方法，它结合了长文档转换技术。

    Being able to identify functions of interest in cross-architecture software is useful whether you are analysing for malware, securing the software supply chain or conducting vulnerability research. Cross-Architecture Binary Code Similarity Search has been explored in numerous studies and has used a wide range of different data sources to achieve its goals. The data sources typically used draw on common structures derived from binaries such as function control flow graphs or binary level call graphs, the output of the disassembly process or the outputs of a dynamic analysis approach. One data source which has received less attention is binary intermediate representations. Binary Intermediate representations possess two interesting properties: they are cross architecture by their very nature and encode the semantics of a function explicitly to support downstream usage. Within this paper we propose Function as a String Encoded Representation (FASER) which combines long document transforme
    
[^3]: TPDR：一种新颖的基于双步骤Transformer的产品和类描述匹配与检索方法

    TPDR: A Novel Two-Step Transformer-based Product and Class Description Match and Retrieval Method. (arXiv:2310.03491v1 [cs.IR])

    [http://arxiv.org/abs/2310.03491](http://arxiv.org/abs/2310.03491)

    TPDR是一种基于Transformer的产品和类描述匹配与检索方法，通过注意机制和对比学习来实现语义对应关系的探索。

    

    有一类公司负责为其他公司中介采购大批量的各种产品，其主要挑战是进行产品描述的标准化，即将客户描述的商品与目录中描述的产品进行匹配。这个问题非常复杂，因为客户的产品描述可能存在以下情况：（1）潜在的噪声；（2）短小且不具备信息（例如，缺少有关型号和尺寸的信息）；（3）跨语言。本文将这个问题形式化为一个排序任务：给定一个初始的客户产品规格（查询），返回最合适的标准化描述（响应）。本文提出了TPDR，一种基于双步骤Transformer的产品和类描述检索方法，该方法能够利用注意机制和对比学习来探索IS和SD之间的语义对应关系。首先，TPDR使用两个编码器的transformers共享嵌入向量空间：

    There is a niche of companies responsible for intermediating the purchase of large batches of varied products for other companies, for which the main challenge is to perform product description standardization, i.e., matching an item described by a client with a product described in a catalog. The problem is complex since the client's product description may be: (1) potentially noisy; (2) short and uninformative (e.g., missing information about model and size); and (3) cross-language. In this paper, we formalize this problem as a ranking task: given an initial client product specification (query), return the most appropriate standardized descriptions (response). In this paper, we propose TPDR, a two-step Transformer-based Product and Class Description Retrieval method that is able to explore the semantic correspondence between IS and SD, by exploiting attention mechanisms and contrastive learning. First, TPDR employs the transformers as two encoders sharing the embedding vector space: 
    
[^4]: 基于个性化Transformer的Yandex电子商务排名系统

    Personalized Transformer-based Ranking for e-Commerce at Yandex. (arXiv:2310.03481v1 [cs.IR])

    [http://arxiv.org/abs/2310.03481](http://arxiv.org/abs/2310.03481)

    本文提出了一个基于个性化Transformer的电子商务排名系统，通过优化排名阶段的特征生成，提高了推荐质量。同时，还引入了一种新颖的技术用于解决偏置上下文的问题。

    

    以用户活动为基础，个性化地提供高质量的推荐对于电子商务平台至关重要，特别是在用户意图不明确的情况下，如主页上。最近，基于嵌入式的个性化系统在电子商务领域的推荐和搜索结果质量方面有了显著的提升。然而，这些工作大多集中在增强检索阶段。在本文中，我们证明了针对电子商务推荐中的排名阶段，检索聚焦的深度学习模型产生的特征是次优的。为了解决这个问题，我们提出了一个两阶段训练过程，通过微调两塔模型来实现最佳的排名性能。我们详细描述了我们专门为电子商务个性化设计的基于Transformer的两塔模型架构。此外，我们还引入了一种新颖的离线模型中去偏置上下文的技术。

    Personalizing the user experience with high-quality recommendations based on user activities is vital for e-commerce platforms. This is particularly important in scenarios where the user's intent is not explicit, such as on the homepage. Recently, personalized embedding-based systems have significantly improved the quality of recommendations and search results in the e-commerce domain. However, most of these works focus on enhancing the retrieval stage.  In this paper, we demonstrate that features produced by retrieval-focused deep learning models are sub-optimal for ranking stage in e-commerce recommendations. To address this issue, we propose a two-stage training process that fine-tunes two-tower models to achieve optimal ranking performance. We provide a detailed description of our transformer-based two-tower model architecture, which is specifically designed for personalization in e-commerce.  Additionally, we introduce a novel technique for debiasing context in offline models and 
    
[^5]: 亚马逊图书评分预测与推荐模型

    Amazon Books Rating prediction & Recommendation Model. (arXiv:2310.03200v1 [cs.IR])

    [http://arxiv.org/abs/2310.03200](http://arxiv.org/abs/2310.03200)

    本文利用亚马逊的数据集构建了一个预测图书评分和推荐图书的模型，提供了处理大数据文件、数据工程和构建模型的流程，并使用了各种PySpark机器学习API、超参数调优和交叉验证进行准确性的分析。

    

    本文利用亚马逊的数据集来预测亚马逊网站上列出的图书评分。作为这个项目的一部分，我们预测了图书的评分，并构建了一个推荐集群。这个推荐集群基于数据集中的列值，比如类别、描述、作者、价格、评论等提供推荐图书。本文提供了处理大数据文件、数据工程、构建模型和提供预测的流程。模型使用了各种PySpark机器学习API来预测图书评分列。此外，我们使用超参数和参数调优。另外，我们还使用了交叉验证和TrainValidationSplit进行泛化。最后，我们比较了二分类和多分类在准确性上的差异。我们将标签从多分类转换为二分类以查看两种分类之间是否有差异。结果表明，我们在二分类中获得了更高的准确性。

    This paper uses the dataset of Amazon to predict the books ratings listed on Amazon website. As part of this project, we predicted the ratings of the books, and also built a recommendation cluster. This recommendation cluster provides the recommended books based on the column's values from dataset, for instance, category, description, author, price, reviews etc. This paper provides a flow of handling big data files, data engineering, building models and providing predictions. The models predict book ratings column using various PySpark Machine Learning APIs. Additionally, we used hyper-parameters and parameters tuning. Also, Cross Validation and TrainValidationSplit were used for generalization. Finally, we performed a comparison between Binary Classification and Multiclass Classification in their accuracies. We converted our label from multiclass to binary to see if we could find any difference between the two classifications. As a result, we found out that we get higher accuracy in b
    
[^6]: 阻抗泄漏脆弱性及其在逆向工程嵌入式软件中的利用

    Impedance Leakage Vulnerability and its Utilization in Reverse-engineering Embedded Software. (arXiv:2310.03175v1 [cs.CR])

    [http://arxiv.org/abs/2310.03175](http://arxiv.org/abs/2310.03175)

    这项研究发现了一种新的安全漏洞——阻抗泄漏，通过利用该漏洞可以从嵌入式设备中提取受保护内存中的软件指令。

    

    发现新的漏洞和实施安全和隐私措施对于保护系统和数据免受物理攻击至关重要。其中一种漏洞是阻抗，一种设备的固有属性，可以通过意外的侧信道泄露信息，从而带来严重的安全和隐私风险。与传统的漏洞不同，阻抗通常被忽视或仅在研究和设计中以特定频率的固定值来处理。此外，阻抗从未被探索过作为信息泄漏的源头。本文证明了嵌入式设备的阻抗并非恒定，并直接与设备上执行的程序相关。我们将此现象定义为阻抗泄漏，并将其作为一种侧信道从受保护的内存中提取软件指令。我们在ATmega328P微控制器和Artix 7 FPGA上的实验表明，阻抗侧信道

    Discovering new vulnerabilities and implementing security and privacy measures are important to protect systems and data against physical attacks. One such vulnerability is impedance, an inherent property of a device that can be exploited to leak information through an unintended side channel, thereby posing significant security and privacy risks. Unlike traditional vulnerabilities, impedance is often overlooked or narrowly explored, as it is typically treated as a fixed value at a specific frequency in research and design endeavors. Moreover, impedance has never been explored as a source of information leakage. This paper demonstrates that the impedance of an embedded device is not constant and directly relates to the programs executed on the device. We define this phenomenon as impedance leakage and use this as a side channel to extract software instructions from protected memory. Our experiment on the ATmega328P microcontroller and the Artix 7 FPGA indicates that the impedance side 
    
[^7]: 多任务学习用于减少多领域视频推荐中的流行度偏差

    Multi-Task Learning For Reduced Popularity Bias In Multi-Territory Video Recommendations. (arXiv:2310.03148v1 [cs.IR])

    [http://arxiv.org/abs/2310.03148](http://arxiv.org/abs/2310.03148)

    本文提出了一种多任务学习技术和自适应上采样方法，用于减少多领域推荐系统中的流行度偏差。通过实验证明，该框架在多个领域中相对增益高达65.27%。

    

    多领域个性化推荐系统中自然产生的各种数据不平衡可能导致全球流行物品的显著项目偏见。局部流行项目可能会被全球流行项目所掩盖。此外，用户的观看模式/统计数据在不同地理位置之间可能发生剧变，这可能表明需要学习特定的用户嵌入。本文提出了一种多任务学习（MTL）技术，以及一种自适应上采样方法，用于减少多领域推荐中的流行偏见。我们的框架通过上采样来丰富含有活跃用户表示的训练样本，并借助MTL来学习基于地理位置的用户嵌入。通过实验证明，与不采用我们提出的技术的基准相比，我们的框架在多个领域的效果显著。值得注意的是，我们在PR-AUC指标上显示出了高达65.27%的相对增益。

    Various data imbalances that naturally arise in a multi-territory personalized recommender system can lead to a significant item bias for globally prevalent items. A locally popular item can be overshadowed by a globally prevalent item. Moreover, users' viewership patterns/statistics can drastically change from one geographic location to another which may suggest to learn specific user embeddings. In this paper, we propose a multi-task learning (MTL) technique, along with an adaptive upsampling method to reduce popularity bias in multi-territory recommendations. Our proposed framework is designed to enrich training examples with active users representation through upsampling, and capable of learning geographic-based user embeddings by leveraging MTL. Through experiments, we demonstrate the effectiveness of our framework in multiple territories compared to a baseline not incorporating our proposed techniques.~Noticeably, we show improved relative gain of up to $65.27\%$ in PR-AUC metric
    
[^8]: 基于上下文的推文参与度预测

    Context-Based Tweet Engagement Prediction. (arXiv:2310.03147v1 [cs.IR])

    [http://arxiv.org/abs/2310.03147](http://arxiv.org/abs/2310.03147)

    该论文研究了基于上下文的推文参与度预测问题，使用了Twitter的数据集和评估流程，探讨了仅凭上下文是否可以很好地预测推文的参与度可能性。

    

    Twitter目前是最大的社交媒体平台之一。其用户可以分享、阅读和参与短推文。在2020年ACM推荐系统会议上，Twitter发布了一个大小约为70GB的数据集，供年度RecSys挑战赛使用。2020年的RecSys挑战赛邀请参与团队创建模型，预测给定用户-推文组合的参与度可能性。提交的模型预测点赞、回复、转发和引用的参与度，并基于两个指标进行评估：精确率-召回率曲线下的面积（PRAUC）和相对交叉熵（RCE）。在这篇学位论文中，我们使用了RecSys 2020挑战赛的数据集和评估流程，研究仅凭上下文能否预测推文参与度的可行性。为此，我们在TU Wien的Little Big Data Cluster上采用Spark引擎创建可扩展的数据预处理、特征工程、特征选择和机器学习流程。我们手动创建。

    Twitter is currently one of the biggest social media platforms. Its users may share, read, and engage with short posts called tweets. For the ACM Recommender Systems Conference 2020, Twitter published a dataset around 70 GB in size for the annual RecSys Challenge. In 2020, the RecSys Challenge invited participating teams to create models that would predict engagement likelihoods for given user-tweet combinations. The submitted models predicting like, reply, retweet, and quote engagements were evaluated based on two metrics: area under the precision-recall curve (PRAUC) and relative cross-entropy (RCE).  In this diploma thesis, we used the RecSys 2020 Challenge dataset and evaluation procedure to investigate how well context alone may be used to predict tweet engagement likelihood. In doing so, we employed the Spark engine on TU Wien's Little Big Data Cluster to create scalable data preprocessing, feature engineering, feature selection, and machine learning pipelines. We manually create
    
[^9]: 一种基于深度强化学习的交互式搜索方法与句级反馈

    A Deep Reinforcement Learning Approach for Interactive Search with Sentence-level Feedback. (arXiv:2310.03043v1 [cs.LG])

    [http://arxiv.org/abs/2310.03043](http://arxiv.org/abs/2310.03043)

    本研究提出了一种利用深度强化学习的交互式搜索方法，该方法通过整合句级反馈信息来提高搜索准确性。通过适应最新的BERT-based模型进行关键句子选择和项目排序，可以获得更满意的搜索结果。

    

    交互式搜索可以通过整合用户的交互反馈来提供更好的搜索体验。这可以显著提高搜索准确性，因为它有助于避免无关信息并捕捉用户的搜索意图。现有的最新系统使用强化学习（RL）模型来整合这些交互，但是忽略了句级反馈中的细粒度信息。然而，这种反馈需要进行广泛的RL行动空间探索和大量的标注数据。本研究通过提出一种新的深度Q学习（DQ）方法DQrank来解决这些挑战。DQrank使用自然语言处理中最新技术BERT-based模型来选择关键句子，并基于用户的参与度对项目进行排序，以获得更满意的回应。我们还提出了两种机制来更好地探索最优行动。DQrank还利用DQ中的经验重现机制来存储反馈句子以增强模型性能。

    Interactive search can provide a better experience by incorporating interaction feedback from the users. This can significantly improve search accuracy as it helps avoid irrelevant information and captures the users' search intents. Existing state-of-the-art (SOTA) systems use reinforcement learning (RL) models to incorporate the interactions but focus on item-level feedback, ignoring the fine-grained information found in sentence-level feedback. Yet such feedback requires extensive RL action space exploration and large amounts of annotated data. This work addresses these challenges by proposing a new deep Q-learning (DQ) approach, DQrank. DQrank adapts BERT-based models, the SOTA in natural language processing, to select crucial sentences based on users' engagement and rank the items to obtain more satisfactory responses. We also propose two mechanisms to better explore optimal actions. DQrank further utilizes the experience replay mechanism in DQ to store the feedback sentences to ob
    
[^10]: 图增强优化器用于结构感知推荐嵌入演化

    Graph-enhanced Optimizers for Structure-aware Recommendation Embedding Evolution. (arXiv:2310.03032v1 [cs.IR])

    [http://arxiv.org/abs/2310.03032](http://arxiv.org/abs/2310.03032)

    本文提出了一种新颖的结构感知嵌入演化(SEvo)机制，能够以较低的计算开销将图结构信息注入到嵌入中，从而在现代推荐系统中实现更高效的性能。

    

    嵌入在现代推荐系统中起着关键作用，因为它们是真实世界实体的虚拟表示，并且是后续决策模型的基础。本文提出了一种新颖的嵌入更新机制，称为结构感知嵌入演化(SEvo)，以鼓励相关节点在每一步中以类似的方式演化。与通常作为中间部分的GNN（图神经网络）不同，SEvo能够直接将图结构信息注入到嵌入中，且在训练过程中计算开销可忽略。本文通过理论分析验证了SEvo的收敛性质及其可能的改进版本，以证明设计的有效性。此外，SEvo可以无缝集成到现有的优化器中，以实现最先进性能。特别是，在矩估计校正的SEvo增强AdamW中，证明了一致的改进效果在多种模型和数据集上，为有效推荐了一种新的技术路线。

    Embedding plays a critical role in modern recommender systems because they are virtual representations of real-world entities and the foundation for subsequent decision models. In this paper, we propose a novel embedding update mechanism, Structure-aware Embedding Evolution (SEvo for short), to encourage related nodes to evolve similarly at each step. Unlike GNN (Graph Neural Network) that typically serves as an intermediate part, SEvo is able to directly inject the graph structure information into embedding with negligible computational overhead in training. The convergence properties of SEvo as well as its possible variants are theoretically analyzed to justify the validity of the designs. Moreover, SEvo can be seamlessly integrated into existing optimizers for state-of-the-art performance. In particular, SEvo-enhanced AdamW with moment estimate correction demonstrates consistent improvements across a spectrum of models and datasets, suggesting a novel technical route to effectively 
    
[^11]: SE-PEF:一个用于个性化专家查找的资源

    SE-PEF: a Resource for Personalized Expert Finding. (arXiv:2309.11686v1 [cs.IR])

    [http://arxiv.org/abs/2309.11686](http://arxiv.org/abs/2309.11686)

    该论文介绍了SE-PEF，一个用于个性化专家查找的资源。该资源包括超过25万个查询和56.5万个答案，并使用一套丰富的特征来建模用户之间的社交互动。初步实验结果表明SE-PEF适用于评估和训练有效的专家查找模型。

    

    个性化信息检索的问题已经被研究了很长时间。与这项任务相关的一个众所周知的问题是缺乏公开可用的数据集，这些数据集可以支持个性化搜索系统的比较评估。为了在这方面做出贡献，本文介绍了SE-PEF（StackExchange-个性化专家查找），这是一个用于设计和评估与专家查找（EF）任务相关的个性化模型的资源。所贡献的数据集包括来自3306个专家的超过25万个查询和56.5万个答案，这些数据集使用了一套丰富的特征来建模热门cQA平台上用户之间的社交互动。初步实验的结果表明，SE-PEF适用于评估和训练有效的EF模型。

    The problem of personalization in Information Retrieval has been under study for a long time. A well-known issue related to this task is the lack of publicly available datasets that can support a comparative evaluation of personalized search systems. To contribute in this respect, this paper introduces SE-PEF (StackExchange - Personalized Expert Finding), a resource useful for designing and evaluating personalized models related to the task of Expert Finding (EF). The contributed dataset includes more than 250k queries and 565k answers from 3 306 experts, which are annotated with a rich set of features modeling the social interactions among the users of a popular cQA platform. The results of the preliminary experiments conducted show the appropriateness of SE-PEF to evaluate and to train effective EF models.
    
[^12]: 在SCRUF中探索推荐公平性的社会选择机制

    Exploring Social Choice Mechanisms for Recommendation Fairness in SCRUF. (arXiv:2309.08621v1 [cs.IR])

    [http://arxiv.org/abs/2309.08621](http://arxiv.org/abs/2309.08621)

    本文通过使用社会选择机制，探索了多个多方面公平应用中的选择机制选项，结果显示不同的选择和分配机制会产生不同但一致的公平性/准确性权衡结果，并且多智能体的构成使得系统能够适应用户人口的动态变化。

    

    推荐系统中的公平性问题往往在实践中具有复杂性，而这一点在简化的研究公式中并没有得到充分的体现。在对公平性问题进行社会选择的框架中，可以在多智能体的公平性关注基础上提供一种灵活且多方面的公平性感知推荐方法。利用社会选择可以增加通用性，并有可能利用经过研究的社会选择算法解决多个竞争的公平性关注之间的紧张关系。本文探讨了在多方面公平应用中选择机制的一系列选项，使用真实数据和合成数据，结果显示不同类别的选择和分配机制在公平性/准确性权衡方面产生了不同但一致的结果。我们还表明，多智能体的构成提供了适应用户人口动态的灵活性。

    Fairness problems in recommender systems often have a complexity in practice that is not adequately captured in simplified research formulations. A social choice formulation of the fairness problem, operating within a multi-agent architecture of fairness concerns, offers a flexible and multi-aspect alternative to fairness-aware recommendation approaches. Leveraging social choice allows for increased generality and the possibility of tapping into well-studied social choice algorithms for resolving the tension between multiple, competing fairness concerns. This paper explores a range of options for choice mechanisms in multi-aspect fairness applications using both real and synthetic data and shows that different classes of choice and allocation mechanisms yield different but consistent fairness / accuracy tradeoffs. We also show that a multi-agent formulation offers flexibility in adapting to user population dynamics.
    
[^13]: SpaDE: 一种利用双重文档编码器改善稀疏表示的第一阶段检索方法

    SpaDE: Improving Sparse Representations using a Dual Document Encoder for First-stage Retrieval. (arXiv:2209.05917v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2209.05917](http://arxiv.org/abs/2209.05917)

    SpaDE 是一种利用双重编码器学习文档表示的第一阶段检索模型，可以同时改善词汇匹配和扩展额外术语来支持语义匹配，且在实验中表现优异。

    

    稀疏的文档表示经常被用来通过精确的词汇匹配来检索相关文档。然而，由于预先计算的倒排索引，会引发词汇不匹配的问题。虽然最近使用预训练语言模型的神经排序模型可以解决这个问题，但它们通常需要昂贵的查询推理成本，这意味着效率和效果之间存在权衡。为了解决这个问题，我们提出了一种新的单编码器排名模型，利用双重编码器学习文档表示，称为 Sparse retriever using a Dual document Encoder (SpaDE)。每个编码器在改善词汇匹配和扩展额外术语来支持语义匹配方面发挥着核心作用。此外，我们的协同训练策略可以有效地训练双重编码器，并避免不必要的干预彼此的训练过程。在几个基准测试中的实验结果表明，SpaDE 超越了现有的检索方法。

    Sparse document representations have been widely used to retrieve relevant documents via exact lexical matching. Owing to the pre-computed inverted index, it supports fast ad-hoc search but incurs the vocabulary mismatch problem. Although recent neural ranking models using pre-trained language models can address this problem, they usually require expensive query inference costs, implying the trade-off between effectiveness and efficiency. Tackling the trade-off, we propose a novel uni-encoder ranking model, Sparse retriever using a Dual document Encoder (SpaDE), learning document representation via the dual encoder. Each encoder plays a central role in (i) adjusting the importance of terms to improve lexical matching and (ii) expanding additional terms to support semantic matching. Furthermore, our co-training strategy trains the dual encoder effectively and avoids unnecessary intervention in training each other. Experimental results on several benchmarks show that SpaDE outperforms ex
    
[^14]: SR-HetGNN:基于异构图神经网络的会话推荐系统

    SR-HetGNN:Session-based Recommendation with Heterogeneous Graph Neural Network. (arXiv:2108.05641v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2108.05641](http://arxiv.org/abs/2108.05641)

    本文提出了一种基于异构图神经网络的会话推荐方法SR-HetGNN，通过学习会话嵌入并捕捉匿名用户的特定偏好，以改进会话推荐系统的效果和准确性。

    

    会话推荐系统的目的是根据先前的会话序列预测用户的下一次点击。目前的研究通常根据用户会话序列中的项目转换来学习用户偏好。然而，会话序列中的其他有效信息，如用户配置文件，往往被忽视，这可能导致模型无法学习用户的具体偏好。在本文中，我们提出了一种基于异构图神经网络的会话推荐方法，命名为SR-HetGNN，它可以通过异构图神经网络（HetGNN）学习会话嵌入，并捕捉匿名用户的特定偏好。具体而言，SR-HetGNN首先根据会话序列构建包含各种类型节点的异构图，可以捕捉项目、用户和会话之间的依赖关系。其次，HetGNN捕捉项目之间的复杂转换并学习包含项目嵌入的特征。

    The purpose of the Session-Based Recommendation System is to predict the user's next click according to the previous session sequence. The current studies generally learn user preferences according to the transitions of items in the user's session sequence. However, other effective information in the session sequence, such as user profiles, are largely ignored which may lead to the model unable to learn the user's specific preferences. In this paper, we propose a heterogeneous graph neural network-based session recommendation method, named SR-HetGNN, which can learn session embeddings by heterogeneous graph neural network (HetGNN), and capture the specific preferences of anonymous users. Specifically, SR-HetGNN first constructs heterogeneous graphs containing various types of nodes according to the session sequence, which can capture the dependencies among items, users, and sessions. Second, HetGNN captures the complex transitions between items and learns the item embeddings containing
    

