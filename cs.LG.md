# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mixup Barcodes: Quantifying Geometric-Topological Interactions between Point Clouds](https://arxiv.org/abs/2402.15058) | 提出了一种名为混合条形码的新方法，利用标准持久同调与图像持久同调结合，可以量化任意维度两个点集之间的几何-拓扑相互作用，以及引入简单的统计量来量化这种相互作用的复杂性。 |
| [^2] | [The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning](https://arxiv.org/abs/2309.01243) | 本文提出了正态分布不可区分性谱定理 (NDIS Theorem)，旨在利用查询本身的随机性改进随机化机器学习查询的差分隐私机制。 |
| [^3] | [RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization.](http://arxiv.org/abs/2310.07983) | RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。 |
| [^4] | [Causal Influences over Social Learning Networks.](http://arxiv.org/abs/2307.09575) | 本论文研究了社交学习网络中代理之间的因果影响，并提出了一种算法来评估整体影响力和发现高度有影响力的代理。 |
| [^5] | [A General Framework for Learning under Corruption: Label Noise, Attribute Noise, and Beyond.](http://arxiv.org/abs/2307.08643) | 该研究提出了一个通用框架，在分布层面上对不同类型的数据污染模型进行了形式化分析，并通过分析贝叶斯风险的变化展示了这些污染对标准监督学习的影响。这些发现为进一步研究提供了新的方向和基础。 |
| [^6] | [Online Resource Allocation with Convex-set Machine-Learned Advice.](http://arxiv.org/abs/2306.12282) | 该论文提出了一个框架，使用凸集机器学习建议来增强在线资源分配决策。该算法类在一致比率和鲁棒比率之间平衡，并在实验中表现出优异的性能。 |
| [^7] | [Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping.](http://arxiv.org/abs/2305.10721) | 本文证明了线性映射在长期时间序列预测中的重要性，提出了RevIN和CI的方法来提高预测性能，同时发现线性映射可以有效地捕捉时间序列的周期特征。 |
| [^8] | [Smart Learning to Find Dumb Contracts.](http://arxiv.org/abs/2304.10726) | DLVA是一种用于以太坊智能合约的强大深度学习漏洞检测工具，其算法涵盖了源代码到字节码的扩展，并且速度比传统漏洞检测工具提高了10-500倍，并成功地发现了一些Slither误标记的易受攻击的合约。 |

# 详细

[^1]: 混合条形码：量化点云之间的几何-拓扑相互作用

    Mixup Barcodes: Quantifying Geometric-Topological Interactions between Point Clouds

    [https://arxiv.org/abs/2402.15058](https://arxiv.org/abs/2402.15058)

    提出了一种名为混合条形码的新方法，利用标准持久同调与图像持久同调结合，可以量化任意维度两个点集之间的几何-拓扑相互作用，以及引入简单的统计量来量化这种相互作用的复杂性。

    

    我们将标准持久同调与图像持久同调相结合，定义了一种新颖的表征形状和它们之间相互作用的方法。具体而言，我们介绍了：（1）混合条形码，捕捉任意维度两个点集之间的几何-拓扑相互作用（混合）；（2）简单的总混合和总百分比混合统计量，作为一个单一数字来量化相互作用的复杂性；（3）一个用于操作上述工具的软件工具。作为一个概念验证，我们将该工具应用到一个源自机器学习的问题上。具体地，我们研究了不同类别嵌入的可分离性。结果表明，拓扑混合是一种用于表征低维和高维数据交互的有效方法。与持久同调的典型用法相比，这个新工具对于拓扑特征的几何位置更为敏感，这通常是可取的。

    arXiv:2402.15058v1 Announce Type: cross  Abstract: We combine standard persistent homology with image persistent homology to define a novel way of characterizing shapes and interactions between them. In particular, we introduce: (1) a mixup barcode, which captures geometric-topological interactions (mixup) between two point sets in arbitrary dimension; (2) simple summary statistics, total mixup and total percentage mixup, which quantify the complexity of the interactions as a single number; (3) a software tool for playing with the above.   As a proof of concept, we apply this tool to a problem arising from machine learning. In particular, we study the disentanglement in embeddings of different classes. The results suggest that topological mixup is a useful method for characterizing interactions for low and high-dimensional data. Compared to the typical usage of persistent homology, the new tool is sensitive to the geometric locations of the topological features, which is often desirabl
    
[^2]: 正态分布不可区分性谱及其在隐私保护机器学习中的应用

    The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning

    [https://arxiv.org/abs/2309.01243](https://arxiv.org/abs/2309.01243)

    本文提出了正态分布不可区分性谱定理 (NDIS Theorem)，旨在利用查询本身的随机性改进随机化机器学习查询的差分隐私机制。

    

    要实现差分隐私(DP)，通常需要随机化基础查询的输出。在大数据分析中，人们经常使用随机化草图/聚合算法来使处理高维数据变得可行。直观地，这样的机器学习(ML)算法应该提供一些固有的隐私性，但现有的大部分DP机制并没有利用这种固有的随机性，导致潜在的多余噪音。我们工作的动机问题是：(如何)可以通过利用查询本身的随机性来提高随机化ML查询的DP机制的效用？为了给出积极的答案，我们证明了正态分布不可区分性谱定理(简称为NDIS定理)，这是一个具有深远实际影响的理论结果。总的来说，NDIS是一个用于$(\epsilon,\delta)$-不可区分性谱(简称为$

    arXiv:2309.01243v2 Announce Type: replace-cross  Abstract: To achieve differential privacy (DP) one typically randomizes the output of the underlying query. In big data analytics, one often uses randomized sketching/aggregation algorithms to make processing high-dimensional data tractable. Intuitively, such machine learning (ML) algorithms should provide some inherent privacy, yet most if not all existing DP mechanisms do not leverage this inherent randomness, resulting in potentially redundant noising.   The motivating question of our work is:   (How) can we improve the utility of DP mechanisms for randomized ML queries, by leveraging the randomness of the query itself?   Towards a (positive) answer, we prove the Normal Distributions Indistinguishability Spectrum Theorem (in short, NDIS Theorem), a theoretical result with far-reaching practical implications. In a nutshell, NDIS is a closed-form analytic computation for the $(\epsilon,\delta)$-indistinguishability-spectrum (in short, $
    
[^3]: RandCom：去中心化随机通信跳跃方法用于分布式随机优化

    RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization. (arXiv:2310.07983v1 [cs.LG])

    [http://arxiv.org/abs/2310.07983](http://arxiv.org/abs/2310.07983)

    RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。

    

    具有随机通信跳过的分布式优化方法因其在加速通信复杂性方面具有的优势而受到越来越多的关注。然而，现有的研究主要集中在强凸确定性设置的集中式通信协议上。在本研究中，我们提出了一种名为RandCom的分布式优化方法，它采用了概率性的本地更新。我们分析了RandCom在随机非凸、凸和强凸设置中的性能，并证明了它能够通过通信概率来渐近地减少通信开销。此外，我们证明当节点数量增加时，RandCom能够实现线性加速。在随机强凸设置中，我们进一步证明了RandCom可以通过独立于网络的步长实现线性加速。此外，我们将RandCom应用于联邦学习，并提供了关于实现线性加速的潜力的积极结果。

    Distributed optimization methods with random communication skips are gaining increasing attention due to their proven benefits in accelerating communication complexity. Nevertheless, existing research mainly focuses on centralized communication protocols for strongly convex deterministic settings. In this work, we provide a decentralized optimization method called RandCom, which incorporates probabilistic local updates. We analyze the performance of RandCom in stochastic non-convex, convex, and strongly convex settings and demonstrate its ability to asymptotically reduce communication overhead by the probability of communication. Additionally, we prove that RandCom achieves linear speedup as the number of nodes increases. In stochastic strongly convex settings, we further prove that RandCom can achieve linear speedup with network-independent stepsizes. Moreover, we apply RandCom to federated learning and provide positive results concerning the potential for achieving linear speedup and
    
[^4]: 社交学习网络中的因果影响研究

    Causal Influences over Social Learning Networks. (arXiv:2307.09575v1 [cs.SI])

    [http://arxiv.org/abs/2307.09575](http://arxiv.org/abs/2307.09575)

    本论文研究了社交学习网络中代理之间的因果影响，并提出了一种算法来评估整体影响力和发现高度有影响力的代理。

    

    本文研究了相互连接且经过时间交互的代理之间的因果影响。具体而言，本论文考察了社交学习模型和分布式决策协议的动态，并推导出了表明代理之间因果关系并解释网络上影响流动的表达式。结果表明，这些因果关系取决于图的拓扑结构和每个代理对于他们试图解决的推理问题的信息水平。基于这些结论，本文提出了一种算法来评估代理之间的整体影响力，以发现高度有影响力的代理。还提供了一种从原始观测数据中学习必要的模型参数的方法。结果和所提出的算法通过考虑合成数据和真实的Twitter数据加以说明。

    This paper investigates causal influences between agents linked by a social graph and interacting over time. In particular, the work examines the dynamics of social learning models and distributed decision-making protocols, and derives expressions that reveal the causal relations between pairs of agents and explain the flow of influence over the network. The results turn out to be dependent on the graph topology and the level of information that each agent has about the inference problem they are trying to solve. Using these conclusions, the paper proposes an algorithm to rank the overall influence between agents to discover highly influential agents. It also provides a method to learn the necessary model parameters from raw observational data. The results and the proposed algorithm are illustrated by considering both synthetic data and real Twitter data.
    
[^5]: 一个学习受到污染的通用框架：标签噪声、属性噪声等等

    A General Framework for Learning under Corruption: Label Noise, Attribute Noise, and Beyond. (arXiv:2307.08643v1 [cs.LG])

    [http://arxiv.org/abs/2307.08643](http://arxiv.org/abs/2307.08643)

    该研究提出了一个通用框架，在分布层面上对不同类型的数据污染模型进行了形式化分析，并通过分析贝叶斯风险的变化展示了这些污染对标准监督学习的影响。这些发现为进一步研究提供了新的方向和基础。

    

    数据中的污染现象很常见，并且已经在不同的污染模型下进行了广泛研究。尽管如此，对于这些模型之间的关系仍然了解有限，缺乏对污染及其对学习的影响的统一视角。在本研究中，我们通过基于马尔可夫核的一般性和详尽的框架，在分布层面上正式分析了污染模型。我们强调了标签和属性上存在的复杂联合和依赖性污染，这在现有研究中很少触及。此外，我们通过分析贝叶斯风险变化来展示这些污染如何影响标准的监督学习。我们的发现提供了对于“更复杂”污染对学习问题影响的定性洞察，并为未来的定量比较提供了基础。该框架的应用包括污染校正学习，其中包含一个子案例。

    Corruption is frequently observed in collected data and has been extensively studied in machine learning under different corruption models. Despite this, there remains a limited understanding of how these models relate such that a unified view of corruptions and their consequences on learning is still lacking. In this work, we formally analyze corruption models at the distribution level through a general, exhaustive framework based on Markov kernels. We highlight the existence of intricate joint and dependent corruptions on both labels and attributes, which are rarely touched by existing research. Further, we show how these corruptions affect standard supervised learning by analyzing the resulting changes in Bayes Risk. Our findings offer qualitative insights into the consequences of "more complex" corruptions on the learning problem, and provide a foundation for future quantitative comparisons. Applications of the framework include corruption-corrected learning, a subcase of which we 
    
[^6]: 凸集机器学习建议下的在线资源分配

    Online Resource Allocation with Convex-set Machine-Learned Advice. (arXiv:2306.12282v1 [cs.DS])

    [http://arxiv.org/abs/2306.12282](http://arxiv.org/abs/2306.12282)

    该论文提出了一个框架，使用凸集机器学习建议来增强在线资源分配决策。该算法类在一致比率和鲁棒比率之间平衡，并在实验中表现出优异的性能。

    

    在线决策者通常会使用机器学习预测需求，称为建议，该建议可以在资源分配的在线决策过程中潜在地被利用。但是，由于其潜在的不准确性，利用这样的建议会带来挑战。为了解决这个问题，我们提出了一个框架，通过潜在不可靠的机器学习建议增强在线资源分配决策。我们假设该建议由需求向量的一般凸不确定性集表示。我们介绍了一种参数化的 Pareto 最优在线资源分配算法类，该算法在一致比率和鲁棒比率之间平衡。一致比率是指当机器学习建议准确时，算法相对于最优的后见之明解的表现，而鲁棒比率则捕获了建议不准确时对抗性需求过程下的表现。具体而言，我们在 C-Pareto 最优情况下，最大化资源利用，同时在一致比率和鲁棒比率之间实现平衡。我们的实验表明，我们提出的框架在合成和真实数据集上均优于基线。

    Decision-makers often have access to a machine-learned prediction about demand, referred to as advice, which can potentially be utilized in online decision-making processes for resource allocation. However, exploiting such advice poses challenges due to its potential inaccuracy. To address this issue, we propose a framework that enhances online resource allocation decisions with potentially unreliable machine-learned (ML) advice. We assume here that this advice is represented by a general convex uncertainty set for the demand vector.  We introduce a parameterized class of Pareto optimal online resource allocation algorithms that strike a balance between consistent and robust ratios. The consistent ratio measures the algorithm's performance (compared to the optimal hindsight solution) when the ML advice is accurate, while the robust ratio captures performance under an adversarial demand process when the advice is inaccurate. Specifically, in a C-Pareto optimal setting, we maximize the r
    
[^7]: 重新审视长期时间序列预测：线性映射的探究

    Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping. (arXiv:2305.10721v1 [cs.LG])

    [http://arxiv.org/abs/2305.10721](http://arxiv.org/abs/2305.10721)

    本文证明了线性映射在长期时间序列预测中的重要性，提出了RevIN和CI的方法来提高预测性能，同时发现线性映射可以有效地捕捉时间序列的周期特征。

    

    近年来，长期时间序列预测受到了越来越多的关注。虽然有各种专门设计来捕捉时间依赖性的方法，但是先前的研究表明，与其他复杂的架构相比，单个线性层可以实现竞争性的预测性能。本文彻底研究了最近方法的内在有效性，并得出了三个主要结论：1）线性映射对于先前的长期时间序列预测至关重要；2）RevIN（可逆规范化）和CI（通道独立）在提高总体预测性能方面发挥重要作用；3）当增加输入视野时，线性映射能够有效捕捉时间序列的周期特征，并具有对不同通道不同周期的鲁棒性。我们提供了理论和实验解释来支持我们的发现，并讨论了局限性和未来工作。我们框架的代码可在\url{https://git}中获得。

    Long-term time series forecasting has gained significant attention in recent years. While there are various specialized designs for capturing temporal dependency, previous studies have demonstrated that a single linear layer can achieve competitive forecasting performance compared to other complex architectures. In this paper, we thoroughly investigate the intrinsic effectiveness of recent approaches and make three key observations: 1) linear mapping is critical to prior long-term time series forecasting efforts; 2) RevIN (reversible normalization) and CI (Channel Independent) play a vital role in improving overall forecasting performance; and 3) linear mapping can effectively capture periodic features in time series and has robustness for different periods across channels when increasing the input horizon. We provide theoretical and experimental explanations to support our findings and also discuss the limitations and future works. Our framework's code is available at \url{https://git
    
[^8]: 智能学习发现 愚笨合约

    Smart Learning to Find Dumb Contracts. (arXiv:2304.10726v1 [cs.CR])

    [http://arxiv.org/abs/2304.10726](http://arxiv.org/abs/2304.10726)

    DLVA是一种用于以太坊智能合约的强大深度学习漏洞检测工具，其算法涵盖了源代码到字节码的扩展，并且速度比传统漏洞检测工具提高了10-500倍，并成功地发现了一些Slither误标记的易受攻击的合约。

    

    我们引入了基于强大深度学习技术的 Deep Learning Vulnerability Analyzer （DLVA），它是一种针对以字节码为基础的以太坊智能合约的漏洞检测工具。我们在没有手动特征工程、预定义模式或专家规则的情况下，将源代码分析扩展到字节码，训练DLVA判断字节码。DLVA训练算法的鲁棒性也很强：它克服了1.25%误标记合约的错误率，学生超越了老师，并发现了Slither误标记的易受攻击的合约。DLVA比基于形式方法的传统智能合约漏洞检测工具快得多：DLVA检查了29个漏洞所需的时间为0.2秒，速度提高了10-500倍。DLVA有三个关键组成部分：Smart Contract to Vector（SC2Vec）将智能合约转换为深度学习模型的向量表示。Bytecode Tokenizer（BCT）将底层字节码转换为神经网络的有意义的标记，DLVA是神经网络模型，可预测智能合约是否包含漏洞。我们对Etherscan的28,505个经过验证的智能合约数据集进行了DLVA评估，发现它取得了0.964的AUC（真阳率/假阳率曲线下的面积）得分。与基线方法相比，DLVA在F1分数上显示了30.7%的改进，它是精度和召回的调和平均值。

    We introduce Deep Learning Vulnerability Analyzer (DLVA), a vulnerability detection tool for Ethereum smart contracts based on powerful deep learning techniques for sequential data adapted for bytecode. We train DLVA to judge bytecode even though the supervising oracle, Slither, can only judge source code. DLVA's training algorithm is general: we "extend" a source code analysis to bytecode without any manual feature engineering, predefined patterns, or expert rules. DLVA's training algorithm is also robust: it overcame a 1.25% error rate mislabeled contracts, and the student surpassing the teacher; found vulnerable contracts that Slither mislabeled. In addition to extending a source code analyzer to bytecode, DLVA is much faster than conventional tools for smart contract vulnerability detection based on formal methods: DLVA checks contracts for 29 vulnerabilities in 0.2 seconds, a speedup of 10-500x+ compared to traditional tools.  DLVA has three key components. Smart Contract to Vecto
    

