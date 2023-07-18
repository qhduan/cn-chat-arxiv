# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data Discovery for the SDGs: A Systematic Rule-based Approach.](http://arxiv.org/abs/2307.07983) | 本研究提出了一种系统的基于规则的方法，用于发现可用于SDG研究的数据类型和来源，并以对SDG 7相关文献的分析为示例。这种方法将手动定性编码与规则应用相结合，计算化地实现了数据实体提取。未来的工作可以通过更先进的自然语言处理和机器学习技术来扩展该方法。 |
| [^2] | [Opinion mining using Double Channel CNN for Recommender System.](http://arxiv.org/abs/2307.07798) | 本文提出了一种使用深度学习模型进行情感分析的方法，通过双通道卷积神经网络进行意见挖掘，并用于推荐产品。通过应用SMOTE算法增加评论数量并对数据进行平衡，使用张量分解算法为聚类分配权重，提高了推荐系统的性能。 |
| [^3] | [Improving Trace Link Recommendation by Using Non-Isotropic Distances and Combinations.](http://arxiv.org/abs/2307.07781) | 本文旨在改进追踪链接推荐，通过使用非等向距离和组合方法。通过研究非线性相似度度量，从几何视角探索语义相似性对于追踪性研究是有帮助的。作者在多个项目数据集上进行了评估，并指出这些发现可以对其他信息检索问题起到基础性作用。 |
| [^4] | [Political Sentiment Analysis of Persian Tweets Using CNN-LSTM Model.](http://arxiv.org/abs/2307.07740) | 本论文使用CNN-LSTM模型对波斯推特的政治情感进行分析，使用ParsBERT进行词汇表示，并比较了机器学习和深度学习模型的效果。实验结果表明，深度学习模型表现更好，其中CNN-LSTM模型在两个数据集上分别达到了89%和71%的分类准确率。 |
| [^5] | [On the Robustness of Epoch-Greedy in Multi-Agent Contextual Bandit Mechanisms.](http://arxiv.org/abs/2307.07675) | 本研究展示了在多Agent上下文赌博机制中，最突出的上下文赌博算法$\epsilon$-greedy可以进行扩展，以解决同时存在的激励因素、上下文和损坏问题 |
| [^6] | [Parmesan: mathematical concept extraction for education.](http://arxiv.org/abs/2307.06699) | Parmesan是一个原型系统，用于在上下文中搜索和定义数学概念，特别关注范畴论领域。该系统利用自然语言处理组件进行概念提取、关系提取、定义提取和实体链接。通过该系统的开发，可以解决现有技术不能直接应用于范畴论领域的问题，并提供了两个数学语料库以支持系统的使用。 |
| [^7] | [Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations.](http://arxiv.org/abs/2307.06576) | 本文介绍了一种名为GLORY的模型，通过全局图与本地表示相结合，增强了个性化推荐系统。该模型通过构建全局感知历史新闻编码器来融合历史新闻表示，并考虑了用户隐藏的动机和行为。 |
| [^8] | [DWT-CompCNN: Deep Image Classification Network for High Throughput JPEG 2000 Compressed Documents.](http://arxiv.org/abs/2306.01359) | 这篇论文提出了一种名为DWT-CompCNN的深度学习模型，它可以直接对使用HTJ2K算法压缩的文档进行分类，从而提高计算效率。 |
| [^9] | [Visualising Personal Data Flows: Insights from a Case Study of Booking.com.](http://arxiv.org/abs/2304.09603) | 本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。 |
| [^10] | [Dual Feedback Attention Framework via Boundary-Aware Auxiliary and Progressive Semantic Optimization for Salient Object Detection in Optical Remote Sensing Imagery.](http://arxiv.org/abs/2303.02867) | 本文提出了一种面向光学遥感图像中显著目标检测的新方法，称为基于边界感知辅助和渐进语义优化的双反馈注意力框架 (DFA-BASO)。通过引入边界保护校准和双特征反馈补充模块，该方法能够减少信息损失、抑制噪声、增强目标的准确性和完整性。 |
| [^11] | [Efficiently Leveraging Multi-level User Intent for Session-based Recommendation via Atten-Mixer Network.](http://arxiv.org/abs/2206.12781) | 本文针对基于会话的推荐任务，通过剖析经典的基于图神经网络的推荐模型，发现一些复杂的图神经网络传播部分是多余的。基于此观察，我们提出了Multi-Level Attention Mixture Network (Atten-Mixer)，它通过移除多余的传播部分，实现了对读出模块的更高效利用。 |

# 详细

[^1]: 数据发现与可持续发展目标：一种系统的基于规则的方法

    Data Discovery for the SDGs: A Systematic Rule-based Approach. (arXiv:2307.07983v1 [cs.IR])

    [http://arxiv.org/abs/2307.07983](http://arxiv.org/abs/2307.07983)

    本研究提出了一种系统的基于规则的方法，用于发现可用于SDG研究的数据类型和来源，并以对SDG 7相关文献的分析为示例。这种方法将手动定性编码与规则应用相结合，计算化地实现了数据实体提取。未来的工作可以通过更先进的自然语言处理和机器学习技术来扩展该方法。

    

    2015年，联合国提出了17个可持续发展目标（SDGs），要在2030年之前实现，在这些目标中，数据被推崇为创新可持续发展和衡量实现SDGs进展的手段。本研究提出了一种系统的方法，用于发现可用于SDG研究的数据类型和来源。所提出的方法将系统的映射方法与手动定性编码相结合，应用规则进行计算化的数据实体提取。本文以对SDG 7相关文献的分析为示例，并提供了相应的结果。本文还对该方法进行讨论，并建议未来通过更先进的自然语言处理和机器学习技术来扩展该方法。

    In 2015, the United Nations put forward 17 Sustainable Development Goals (SDGs) to be achieved by 2030, where data has been promoted as a focus to innovating sustainable development and as a means to measuring progress towards achieving the SDGs. In this study, we propose a systematic approach towards discovering data types and sources that can be used for SDG research. The proposed method integrates a systematic mapping approach using manual qualitative coding over a corpus of SDG-related research literature followed by an automated process that applies rules to perform data entity extraction computationally. This approach is exemplified by an analysis of literature relating to SDG 7, the results of which are also presented in this paper. The paper concludes with a discussion of the approach and suggests future work to extend the method with more advance NLP and machine learning techniques.
    
[^2]: 使用双通道卷积神经网络进行推荐系统的意见挖掘

    Opinion mining using Double Channel CNN for Recommender System. (arXiv:2307.07798v1 [cs.IR])

    [http://arxiv.org/abs/2307.07798](http://arxiv.org/abs/2307.07798)

    本文提出了一种使用深度学习模型进行情感分析的方法，通过双通道卷积神经网络进行意见挖掘，并用于推荐产品。通过应用SMOTE算法增加评论数量并对数据进行平衡，使用张量分解算法为聚类分配权重，提高了推荐系统的性能。

    

    随着互联网和社交媒体的发展，产生了大量的非结构化数据。这些数据中包括用户对在线商店和社交媒体上产品的意见。通过对这些意见进行探索和分类，可以获取有用的信息，包括用户满意度、用户对特定事件的反馈、预测特定产品的销售情况等。在本文中，我们提出了一种使用深度学习模型进行情感分析并用于推荐产品的方法。我们使用了一个具有五层的双通道卷积神经网络模型进行意见挖掘，该模型从数据中提取了重要的特征。我们通过应用SMOTE算法来增加初始数据集的评论数量，并对数据进行平衡。然后我们对这些评论进行了聚类。我们还使用张量分解算法为每个聚类分配了一个权重，从而提高了推荐系统的性能。我们提出的方法已经在实验中取得了很好的结果。

    Much unstructured data has been produced with the growth of the Internet and social media. A significant volume of textual data includes users' opinions about products in online stores and social media. By exploring and categorizing them, helpful information can be acquired, including customer satisfaction, user feedback about a particular event, predicting the sale of a specific product, and other similar cases. In this paper, we present an approach for sentiment analysis with a deep learning model and use it to recommend products. A two-channel convolutional neural network model has been used for opinion mining, which has five layers and extracts essential features from the data. We increased the number of comments by applying the SMOTE algorithm to the initial dataset and balanced the data. Then we proceed to cluster the aspects. We also assign a weight to each cluster using tensor decomposition algorithms that improve the recommender system's performance. Our proposed method has re
    
[^3]: 通过使用非等向距离和组合改进追踪链接推荐

    Improving Trace Link Recommendation by Using Non-Isotropic Distances and Combinations. (arXiv:2307.07781v1 [cs.SE])

    [http://arxiv.org/abs/2307.07781](http://arxiv.org/abs/2307.07781)

    本文旨在改进追踪链接推荐，通过使用非等向距离和组合方法。通过研究非线性相似度度量，从几何视角探索语义相似性对于追踪性研究是有帮助的。作者在多个项目数据集上进行了评估，并指出这些发现可以对其他信息检索问题起到基础性作用。

    

    软件开发生命周期中的构件之间存在追踪链接可以提高软件开发、维护和运营过程中的效率。然而，追踪链接的创建和维护耗时且容易出错。近年来，随着自然语言处理领域强大工具的出现，对自动计算追踪链接进行研究的努力逐渐增加。在本文中，我们报告了在研究用于计算追踪链接的非线性相似度度量时所做的一些观察。我们认为，从几何视角来看待语义相似性可以有助于未来的追踪性研究。我们在四个开源项目和两个工业项目的数据集上评估了我们的观察结果。我们还指出，我们的发现更具普遍性，也可以为其他信息检索问题奠定基础。

    The existence of trace links between artifacts of the software development life cycle can improve the efficiency of many activities during software development, maintenance and operations. Unfortunately, the creation and maintenance of trace links is time-consuming and error-prone. Research efforts have been spent to automatically compute trace links and lately gained momentum, e.g., due to the availability of powerful tools in the area of natural language processing. In this paper, we report on some observations that we made during studying non-linear similarity measures for computing trace links. We argue, that taking a geometric viewpoint on semantic similarity can be helpful for future traceability research. We evaluated our observations on a dataset of four open source projects and two industrial projects. We furthermore point out that our findings are more general and can build the basis for other information retrieval problems as well.
    
[^4]: 使用CNN-LSTM模型对波斯推特的政治情感进行分析

    Political Sentiment Analysis of Persian Tweets Using CNN-LSTM Model. (arXiv:2307.07740v1 [cs.CL])

    [http://arxiv.org/abs/2307.07740](http://arxiv.org/abs/2307.07740)

    本论文使用CNN-LSTM模型对波斯推特的政治情感进行分析，使用ParsBERT进行词汇表示，并比较了机器学习和深度学习模型的效果。实验结果表明，深度学习模型表现更好，其中CNN-LSTM模型在两个数据集上分别达到了89%和71%的分类准确率。

    

    情感分析是识别和分类人们对各种话题的情感或观点的过程。近年来，对Twitter情感的分析成为一个越来越受欢迎的话题。在本文中，我们提出了几种机器学习和深度学习模型，用于分析波斯政治推特的情感。我们使用词袋模型和ParsBERT进行词汇表示的分析。我们应用了高斯朴素贝叶斯、梯度提升、逻辑回归、决策树、随机森林以及CNN和LSTM的组合来分类推特的极性。本研究的结果表明，使用ParsBERT嵌入的深度学习模型比机器学习表现更好。CNN-LSTM模型在第一个有三种类别的数据集上的分类准确率为89％，在第二个有七种类别的数据集上的分类准确率为71％。由于波斯语的复杂性，达到这一效率水平是一项困难的任务。

    Sentiment analysis is the process of identifying and categorizing people's emotions or opinions regarding various topics. The analysis of Twitter sentiment has become an increasingly popular topic in recent years. In this paper, we present several machine learning and a deep learning model to analysis sentiment of Persian political tweets. Our analysis was conducted using Bag of Words and ParsBERT for word representation. We applied Gaussian Naive Bayes, Gradient Boosting, Logistic Regression, Decision Trees, Random Forests, as well as a combination of CNN and LSTM to classify the polarities of tweets. The results of this study indicate that deep learning with ParsBERT embedding performs better than machine learning. The CNN-LSTM model had the highest classification accuracy with 89 percent on the first dataset with three classes and 71 percent on the second dataset with seven classes. Due to the complexity of Persian, it was a difficult task to achieve this level of efficiency.
    
[^5]: 关于多节点上下文赌博机制中Epoch-Greedy的鲁棒性

    On the Robustness of Epoch-Greedy in Multi-Agent Contextual Bandit Mechanisms. (arXiv:2307.07675v1 [cs.LG])

    [http://arxiv.org/abs/2307.07675](http://arxiv.org/abs/2307.07675)

    本研究展示了在多Agent上下文赌博机制中，最突出的上下文赌博算法$\epsilon$-greedy可以进行扩展，以解决同时存在的激励因素、上下文和损坏问题

    

    在像点击付费(Pay-Per-Click)拍卖这样的多臂赌博机制中进行高效学习通常涉及三个挑战：1)引导真实出价行为(激励因素)，2)在用户个性化上下文中使用个性化(上下文)，3)规避点击模式中的操纵(损坏行为)。过去文献中每个挑战都被独立研究过；激励因素已在一系列研究中得到解决，上下文问题已通过上下文赌博算法得到广泛解决，而损坏问题则通过最近的关于具有对抗性损坏的赌博机制工作进行讨论。由于这些挑战同时存在，重要的是了解每种方法在解决其他挑战时的鲁棒性，提供可以同时处理所有挑战的算法，并突出这种组合中的固有局限性。在这项工作中，我们展示了最突出的上下文赌博算法$\epsilon$-greedy可以进行扩展，以解决同时存在的激励因素、上下文和损坏问题。

    Efficient learning in multi-armed bandit mechanisms such as pay-per-click (PPC) auctions typically involves three challenges: 1) inducing truthful bidding behavior (incentives), 2) using personalization in the users (context), and 3) circumventing manipulations in click patterns (corruptions). Each of these challenges has been studied orthogonally in the literature; incentives have been addressed by a line of work on truthful multi-armed bandit mechanisms, context has been extensively tackled by contextual bandit algorithms, while corruptions have been discussed via a recent line of work on bandits with adversarial corruptions. Since these challenges co-exist, it is important to understand the robustness of each of these approaches in addressing the other challenges, provide algorithms that can handle all simultaneously, and highlight inherent limitations in this combination. In this work, we show that the most prominent contextual bandit algorithm, $\epsilon$-greedy can be extended to
    
[^6]: Parmesan：教育中的数学概念提取

    Parmesan: mathematical concept extraction for education. (arXiv:2307.06699v1 [cs.CL])

    [http://arxiv.org/abs/2307.06699](http://arxiv.org/abs/2307.06699)

    Parmesan是一个原型系统，用于在上下文中搜索和定义数学概念，特别关注范畴论领域。该系统利用自然语言处理组件进行概念提取、关系提取、定义提取和实体链接。通过该系统的开发，可以解决现有技术不能直接应用于范畴论领域的问题，并提供了两个数学语料库以支持系统的使用。

    

    数学是一个高度专业化的领域，具有自己独特的挑战，但在自然语言处理领域的研究却有限。然而，数学在许多不同领域的跨学科研究中经常依赖于对数学概念的理解。为了帮助来自其他领域的研究人员，我们开发了一个原型系统，用于在上下文中搜索和定义数学概念，重点关注范畴论领域。这个系统名为Parmesan，依赖于自然语言处理组件，包括概念提取、关系提取、定义提取和实体链接。在开发这个系统的过程中，我们展示了现有技术不能直接应用于范畴论领域，并提出了一种混合技术，这种技术表现良好，但我们预计系统将随着时间的推移而不断演变。我们还提供了两个清理过的数学语料库，用于支持原型系统，这些语料库基于期刊文章。

    Mathematics is a highly specialized domain with its own unique set of challenges that has seen limited study in natural language processing. However, mathematics is used in a wide variety of fields and multidisciplinary research in many different domains often relies on an understanding of mathematical concepts. To aid researchers coming from other fields, we develop a prototype system for searching for and defining mathematical concepts in context, focusing on the field of category theory. This system, Parmesan, depends on natural language processing components including concept extraction, relation extraction, definition extraction, and entity linking. In developing this system, we show that existing techniques cannot be applied directly to the category theory domain, and suggest hybrid techniques that do perform well, though we expect the system to evolve over time. We also provide two cleaned mathematical corpora that power the prototype system, which are based on journal articles 
    
[^7]: 超越本地范围：全球图增强个性化新闻推荐

    Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations. (arXiv:2307.06576v1 [cs.IR])

    [http://arxiv.org/abs/2307.06576](http://arxiv.org/abs/2307.06576)

    本文介绍了一种名为GLORY的模型，通过全局图与本地表示相结合，增强了个性化推荐系统。该模型通过构建全局感知历史新闻编码器来融合历史新闻表示，并考虑了用户隐藏的动机和行为。

    

    精确地向用户推荐候选新闻文章一直是个性化新闻推荐系统的核心挑战。大多数近期的研究主要集中在使用先进的自然语言处理技术从丰富的文本数据中提取语义信息，使用从本地历史新闻派生的基于内容的方法。然而，这种方法缺乏全局视角，未能考虑用户隐藏的动机和行为，超越语义信息。为了解决这个问题，我们提出了一种新颖的模型 GLORY（Global-LOcal news Recommendation sYstem），它结合了从其他用户学到的全局表示和本地表示，来增强个性化推荐系统。我们通过构建一个全局感知历史新闻编码器来实现这一目标，其中包括一个全局新闻图，并使用门控图神经网络来丰富新闻表示，从而通过历史新闻聚合器融合历史新闻表示。

    Precisely recommending candidate news articles to users has always been a core challenge for personalized news recommendation systems. Most recent works primarily focus on using advanced natural language processing techniques to extract semantic information from rich textual data, employing content-based methods derived from local historical news. However, this approach lacks a global perspective, failing to account for users' hidden motivations and behaviors beyond semantic information. To address this challenge, we propose a novel model called GLORY (Global-LOcal news Recommendation sYstem), which combines global representations learned from other users with local representations to enhance personalized recommendation systems. We accomplish this by constructing a Global-aware Historical News Encoder, which includes a global news graph and employs gated graph neural networks to enrich news representations, thereby fusing historical news representations by a historical news aggregator.
    
[^8]: DWT-CompCNN：用于高吞吐量JPEG 2000压缩文档的深度图像分类网络

    DWT-CompCNN: Deep Image Classification Network for High Throughput JPEG 2000 Compressed Documents. (arXiv:2306.01359v1 [cs.CV])

    [http://arxiv.org/abs/2306.01359](http://arxiv.org/abs/2306.01359)

    这篇论文提出了一种名为DWT-CompCNN的深度学习模型，它可以直接对使用HTJ2K算法压缩的文档进行分类，从而提高计算效率。

    

    对于任何包含文档图像的数字应用程序，如检索，文档图像的分类成为必要的阶段。传统上，为了达到这个目的，文档的完整版本，即未压缩的文档图像构成输入数据集，这会因数据量大而带来威胁。因此，如果可以使用文档的压缩表示（在部分解压缩的情况下），直接完成相同的分类任务以使整个过程计算效率更高，那将会是一项创新。本研究提出了一种新颖的深度学习模型DWT-CompCNN，用于使用高吞吐量JPEG 2000（HTJ2K）算法压缩的文档的分类。所提出的DWT-CompCNN包括五个卷积层，卷积核大小分别为16、32、64、128和256用于从提取的小波系数中提高学习能力。

    For any digital application with document images such as retrieval, the classification of document images becomes an essential stage. Conventionally for the purpose, the full versions of the documents, that is the uncompressed document images make the input dataset, which poses a threat due to the big volume required to accommodate the full versions of the documents. Therefore, it would be novel, if the same classification task could be accomplished directly (with some partial decompression) with the compressed representation of documents in order to make the whole process computationally more efficient. In this research work, a novel deep learning model, DWT CompCNN is proposed for classification of documents that are compressed using High Throughput JPEG 2000 (HTJ2K) algorithm. The proposed DWT-CompCNN comprises of five convolutional layers with filter sizes of 16, 32, 64, 128, and 256 consecutively for each increasing layer to improve learning from the wavelet coefficients extracted
    
[^9]: 可视化个人数据流：以Booking.com为例的案例研究

    Visualising Personal Data Flows: Insights from a Case Study of Booking.com. (arXiv:2304.09603v1 [cs.CR])

    [http://arxiv.org/abs/2304.09603](http://arxiv.org/abs/2304.09603)

    本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。

    

    商业机构持有和处理的个人数据量越来越多。政策和法律不断变化，要求这些公司在收集、存储、处理和共享这些数据方面更加透明。本文报告了我们以Booking.com为案例研究，从他们的隐私政策中提取个人数据流的可视化工作。通过展示该公司如何分享其消费者的个人数据，我们提出了问题，并扩展了有关使用隐私政策告知客户个人数据流范围的挑战和限制的讨论。更重要的是，本案例研究可以为未来更以数据流为导向的隐私政策分析和在复杂商业生态系统中建立更全面的个人数据流本体论的研究提供参考。

    Commercial organisations are holding and processing an ever-increasing amount of personal data. Policies and laws are continually changing to require these companies to be more transparent regarding collection, storage, processing and sharing of this data. This paper reports our work of taking Booking.com as a case study to visualise personal data flows extracted from their privacy policy. By showcasing how the company shares its consumers' personal data, we raise questions and extend discussions on the challenges and limitations of using privacy policy to inform customers the true scale and landscape of personal data flows. More importantly, this case study can inform us about future research on more data flow-oriented privacy policy analysis and on the construction of a more comprehensive ontology on personal data flows in complicated business ecosystems.
    
[^10]: 基于边界感知辅助和渐进语义优化的双反馈注意力框架用于光学遥感图像中的显著目标检测

    Dual Feedback Attention Framework via Boundary-Aware Auxiliary and Progressive Semantic Optimization for Salient Object Detection in Optical Remote Sensing Imagery. (arXiv:2303.02867v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.02867](http://arxiv.org/abs/2303.02867)

    本文提出了一种面向光学遥感图像中显著目标检测的新方法，称为基于边界感知辅助和渐进语义优化的双反馈注意力框架 (DFA-BASO)。通过引入边界保护校准和双特征反馈补充模块，该方法能够减少信息损失、抑制噪声、增强目标的准确性和完整性。

    

    光学遥感图像中的显著目标检测逐渐引起了人们的关注，这要归功于深度学习和自然场景图像中的显著目标检测的发展。然而，自然场景图像和光学遥感图像在许多方面是不同的，例如覆盖范围大、背景复杂以及目标类型和尺度的巨大差异。因此，需要一种专门的方法来处理光学遥感图像中的显著目标检测。此外，现有的方法没有充分关注到目标的边界，最终显著性图的完整性仍需要改进。为了解决这些问题，我们提出了一种新的方法，称为基于边界感知辅助和渐进语义优化的双反馈注意力框架（DFA-BASO）。首先，引入了边界保护校准(BPC)模块，用于减少正向传播过程中边界位置信息的丢失，并抑制低级特征中的噪声。其次，引入了双特征反馈补充(DFFC)模块，用于增强正反馈和负反馈之间的相互作用，提高显著性目标的准确性和完整性。

    Salient object detection in optical remote sensing image (ORSI-SOD) has gradually attracted attention thanks to the development of deep learning (DL) and salient object detection in natural scene image (NSI-SOD). However, NSI and ORSI are different in many aspects, such as large coverage, complex background, and large differences in target types and scales. Therefore, a new dedicated method is needed for ORSI-SOD. In addition, existing methods do not pay sufficient attention to the boundary of the object, and the completeness of the final saliency map still needs improvement. To address these issues, we propose a novel method called Dual Feedback Attention Framework via Boundary-Aware Auxiliary and Progressive Semantic Optimization (DFA-BASO). First, Boundary Protection Calibration (BPC) module is proposed to reduce the loss of edge position information during forward propagation and suppress noise in low-level features. Second, a Dual Feature Feedback Complementary (DFFC) module is pr
    
[^11]: 通过Atten-Mixer网络高效利用多级用户意图进行基于会话的推荐

    Efficiently Leveraging Multi-level User Intent for Session-based Recommendation via Atten-Mixer Network. (arXiv:2206.12781v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.12781](http://arxiv.org/abs/2206.12781)

    本文针对基于会话的推荐任务，通过剖析经典的基于图神经网络的推荐模型，发现一些复杂的图神经网络传播部分是多余的。基于此观察，我们提出了Multi-Level Attention Mixture Network (Atten-Mixer)，它通过移除多余的传播部分，实现了对读出模块的更高效利用。

    

    基于会话的推荐旨在根据短暂且动态的会话预测用户的下一个动作。最近，在利用各种精心设计的图神经网络(GNN)捕捉物品之间的成对关系方面引起了越来越多的兴趣，似乎表明设计更复杂的模型是提高实证性能的万灵药。然而，这些模型在模型复杂度呈指数增长的同时，仅取得了相对较小的改进。在本文中，我们剖析了经典的基于GNN的SBR模型，并在经验上发现，一些复杂的GNN传播在给定读出模块在GNN模型中起到重要作用的情况下是多余的。基于这一观察，我们直观地提出了移除GNN传播部分的想法，而读出模块将在模型推理过程中承担更多责任。为此，我们提出了Multi-Level Attention Mixture Network (Atten-Mixer)，它同时利用概念-

    Session-based recommendation (SBR) aims to predict the user's next action based on short and dynamic sessions. Recently, there has been an increasing interest in utilizing various elaborately designed graph neural networks (GNNs) to capture the pair-wise relationships among items, seemingly suggesting the design of more complicated models is the panacea for improving the empirical performance. However, these models achieve relatively marginal improvements with exponential growth in model complexity. In this paper, we dissect the classical GNN-based SBR models and empirically find that some sophisticated GNN propagations are redundant, given the readout module plays a significant role in GNN-based models. Based on this observation, we intuitively propose to remove the GNN propagation part, while the readout module will take on more responsibility in the model reasoning process. To this end, we propose the Multi-Level Attention Mixture Network (Atten-Mixer), which leverages both concept-
    

