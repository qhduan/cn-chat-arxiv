# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Using Quantum Computing to Infer Dynamic Behaviors of Biological and Artificial Neural Networks](https://arxiv.org/abs/2403.18963) | 本研究展示了如何利用Grover和Deutsch-Josza等基础量子算法，通过一组精心构建的条件，推断生物和人工神经网络在一段时间内是否具有继续维持动态活动的潜力。 |
| [^2] | [LLM-Enhanced User-Item Interactions: Leveraging Edge Information for Optimized Recommendations](https://arxiv.org/abs/2402.09617) | 这项研究旨在提高LLM在图数据中的关系挖掘效率和能力，通过整合图神经网络和大型语言模型，以利用边缘信息来理解复杂节点关系，并从图结构中提取有意义洞见。 |
| [^3] | [Boosting for Bounding the Worst-class Error.](http://arxiv.org/abs/2310.14890) | 该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。 |
| [^4] | [TBDetector:Transformer-Based Detector for Advanced Persistent Threats with Provenance Graph.](http://arxiv.org/abs/2304.02838) | 本论文提出了一种采用来源图和Transformer的高级持久性威胁检测方法，利用Transformer的自注意力编码器-解码器提取系统状态的长期上下文特征，并通过来源分析实现对长期运行系统的概括，以检测缓慢攻击。 |
| [^5] | [Real-world Machine Learning Systems: A survey from a Data-Oriented Architecture Perspective.](http://arxiv.org/abs/2302.04810) | 这项调查研究了现实世界中部署机器学习系统的数据导向架构（DOA）的采用情况，发现尽管没有明确提及DOA，但许多论文中的设计决策默默地遵循了DOA的原则。 |

# 详细

[^1]: 使用量子计算推断生物和人工神经网络的动态行为

    Using Quantum Computing to Infer Dynamic Behaviors of Biological and Artificial Neural Networks

    [https://arxiv.org/abs/2403.18963](https://arxiv.org/abs/2403.18963)

    本研究展示了如何利用Grover和Deutsch-Josza等基础量子算法，通过一组精心构建的条件，推断生物和人工神经网络在一段时间内是否具有继续维持动态活动的潜力。

    

    新问题类别的探索是量子计算研究的一个活跃领域。一个基本上完全未被探讨的主题是使用量子算法和计算来探索和询问神经网络的功能动态。这是将量子计算应用于生物和人工神经网络建模和仿真的尚未成熟的主题的一个组成部分。在本研究中，我们展示了如何通过精心构建的一组条件来使用两个基础量子算法，Grover和Deutsch-Josza，以使输出测量具有一种解释，保证我们能够推断一个简单的神经网络表示（适用于生物和人工网络）在一段时间后是否有可能继续维持动态活动。或者这些动态保证会停止，要么是通过'癫痫'动态，要么是静止状态。

    arXiv:2403.18963v1 Announce Type: cross  Abstract: The exploration of new problem classes for quantum computation is an active area of research. An essentially completely unexplored topic is the use of quantum algorithms and computing to explore and ask questions \textit{about} the functional dynamics of neural networks. This is a component of the still-nascent topic of applying quantum computing to the modeling and simulations of biological and artificial neural networks. In this work, we show how a carefully constructed set of conditions can use two foundational quantum algorithms, Grover and Deutsch-Josza, in such a way that the output measurements admit an interpretation that guarantees we can infer if a simple representation of a neural network (which applies to both biological and artificial networks) after some period of time has the potential to continue sustaining dynamic activity. Or whether the dynamics are guaranteed to stop either through 'epileptic' dynamics or quiescence
    
[^2]: 增强LLM用户-物品交互：利用边缘信息进行优化推荐的研究

    LLM-Enhanced User-Item Interactions: Leveraging Edge Information for Optimized Recommendations

    [https://arxiv.org/abs/2402.09617](https://arxiv.org/abs/2402.09617)

    这项研究旨在提高LLM在图数据中的关系挖掘效率和能力，通过整合图神经网络和大型语言模型，以利用边缘信息来理解复杂节点关系，并从图结构中提取有意义洞见。

    

    大型语言模型的出色性能不仅改变了自然语言处理领域的研究格局，还展示了它在各个领域的卓越应用潜力。然而，这些模型在挖掘图数据中的关系方面的潜力仍未得到充分探索。图神经网络作为近年来热门的研究领域，在关系挖掘方面有大量研究。然而，当前图神经网络的尖端研究尚未有效整合大型语言模型，导致在图关系挖掘任务中的效率和能力受限。一个主要的挑战是LLM无法深入利用图中的边缘信息，而这对于理解复杂节点关系至关重要。这种差距限制了LLM从图结构中提取有意义洞见的潜力，限制了它在更复杂的基于图的分析中的适用性。

    arXiv:2402.09617v1 Announce Type: new  Abstract: The extraordinary performance of large language models has not only reshaped the research landscape in the field of NLP but has also demonstrated its exceptional applicative potential in various domains. However, the potential of these models in mining relationships from graph data remains under-explored. Graph neural networks, as a popular research area in recent years, have numerous studies on relationship mining. Yet, current cutting-edge research in graph neural networks has not been effectively integrated with large language models, leading to limited efficiency and capability in graph relationship mining tasks. A primary challenge is the inability of LLMs to deeply exploit the edge information in graphs, which is critical for understanding complex node relationships. This gap limits the potential of LLMs to extract meaningful insights from graph structures, limiting their applicability in more complex graph-based analysis. We focus
    
[^3]: Boosting用于界定最差分类误差

    Boosting for Bounding the Worst-class Error. (arXiv:2310.14890v1 [stat.ML])

    [http://arxiv.org/abs/2310.14890](http://arxiv.org/abs/2310.14890)

    该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。

    

    本文解决了最差类别误差率的问题，而不是针对所有类别的标准误差率的平均。例如，一个三类别分类任务，其中各类别的误差率分别为10％，10％和40％，其最差类别误差率为40％，而在类别平衡条件下的平均误差率为20％。最差类别错误在许多应用中很重要。例如，在医学图像分类任务中，对于恶性肿瘤类别具有40％的错误率而良性和健康类别具有10％的错误率是不能被接受的。我们提出了一种保证最差类别训练误差上界的提升算法，并推导出其泛化界。实验结果表明，该算法降低了最差类别的测试误差率，同时避免了对训练集的过拟合。

    This paper tackles the problem of the worst-class error rate, instead of the standard error rate averaged over all classes. For example, a three-class classification task with class-wise error rates of 10\%, 10\%, and 40\% has a worst-class error rate of 40\%, whereas the average is 20\% under the class-balanced condition. The worst-class error is important in many applications. For example, in a medical image classification task, it would not be acceptable for the malignant tumor class to have a 40\% error rate, while the benign and healthy classes have 10\% error rates.We propose a boosting algorithm that guarantees an upper bound of the worst-class training error and derive its generalization bound. Experimental results show that the algorithm lowers worst-class test error rates while avoiding overfitting to the training set.
    
[^4]: 基于Transformer和来源图的高级持久性威胁检测方法

    TBDetector:Transformer-Based Detector for Advanced Persistent Threats with Provenance Graph. (arXiv:2304.02838v1 [cs.CR])

    [http://arxiv.org/abs/2304.02838](http://arxiv.org/abs/2304.02838)

    本论文提出了一种采用来源图和Transformer的高级持久性威胁检测方法，利用Transformer的自注意力编码器-解码器提取系统状态的长期上下文特征，并通过来源分析实现对长期运行系统的概括，以检测缓慢攻击。

    

    针对高级持久性威胁（APT）攻击的长期潜伏、隐秘多阶段攻击模式，本文提出了一种基于Transformer的APT检测方法，利用来源图提供的历史信息进行APT检测。该方法利用Transformer的自注意力编码器-解码器提取系统状态的长期上下文特征，并通过来源分析实现对长期运行系统的概括，以检测缓慢攻击。此外，作者还引入了异常评分，可评估不同系统状态的异常性。每个状态都有相应的相似度和隔离度分数的异常分数计算。为了评估该方法的有效性

    APT detection is difficult to detect due to the long-term latency, covert and slow multistage attack patterns of Advanced Persistent Threat (APT). To tackle these issues, we propose TBDetector, a transformer-based advanced persistent threat detection method for APT attack detection. Considering that provenance graphs provide rich historical information and have the powerful attacks historic correlation ability to identify anomalous activities, TBDetector employs provenance analysis for APT detection, which summarizes long-running system execution with space efficiency and utilizes transformer with self-attention based encoder-decoder to extract long-term contextual features of system states to detect slow-acting attacks. Furthermore, we further introduce anomaly scores to investigate the anomaly of different system states, where each state is calculated with an anomaly score corresponding to its similarity score and isolation score. To evaluate the effectiveness of the proposed method,
    
[^5]: 现实世界中的机器学习系统：基于数据导向架构的调查

    Real-world Machine Learning Systems: A survey from a Data-Oriented Architecture Perspective. (arXiv:2302.04810v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2302.04810](http://arxiv.org/abs/2302.04810)

    这项调查研究了现实世界中部署机器学习系统的数据导向架构（DOA）的采用情况，发现尽管没有明确提及DOA，但许多论文中的设计决策默默地遵循了DOA的原则。

    

    随着对人工智能的兴趣不断增长，机器学习模型正在作为现实世界系统的一部分部署。这些系统的设计、实现和维护受到现实世界环境的挑战，这些环境产生了更多的异构数据，用户需要更快的响应速度和高效的资源消耗。这些要求将普遍存在的软件架构推向了极限，当部署基于机器学习的系统时。数据导向架构（DOA）是一个新兴的概念，它能更好地为集成机器学习模型的系统提供支持。DOA扩展了当前的架构，创建了数据驱动、松耦合、去中心化和开放的系统。尽管部署的机器学习系统的论文中没有提到DOA，但它们的作者在设计上隐含地遵循了DOA。为什么、如何以及在多大程度上采用DOA在这些系统中尚不清楚。隐含的设计决策限制了从业者对于设计基于机器学习的系统时DOA的认识。

    Machine Learning models are being deployed as parts of real-world systems with the upsurge of interest in artificial intelligence. The design, implementation, and maintenance of such systems are challenged by real-world environments that produce larger amounts of heterogeneous data and users requiring increasingly faster responses with efficient resource consumption. These requirements push prevalent software architectures to the limit when deploying ML-based systems. Data-oriented Architecture (DOA) is an emerging concept that equips systems better for integrating ML models. DOA extends current architectures to create data-driven, loosely coupled, decentralised, open systems. Even though papers on deployed ML-based systems do not mention DOA, their authors made design decisions that implicitly follow DOA. The reasons why, how, and the extent to which DOA is adopted in these systems are unclear. Implicit design decisions limit the practitioners' knowledge of DOA to design ML-based syst
    

