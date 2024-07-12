# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automatic Outlier Rectification via Optimal Transport](https://arxiv.org/abs/2403.14067) | 提出了一种自动异常值矫正机制，通过将矫正和估计集成到联合优化框架中，利用最优输运和凹成本函数来检测和移除异常值，并选择最佳分布来执行估计任务 |
| [^2] | [Adjustment Identification Distance: A gadjid for Causal Structure Learning](https://arxiv.org/abs/2402.08616) | gadjid软件包提供了一种用于因果结构学习的调整识别距离，通过引入框架来计算因果距离，这些距离能够高效评估因果发现算法学习的图形，并且在处理大规模图形时具有较高的性能。 |
| [^3] | [Featurizing Koopman Mode Decomposition](https://arxiv.org/abs/2312.09146) | 本文提出了一种命名为FKMD的先进KMD技术，通过时间嵌入和马氏距离缩放，可以增强对高维动力系统的分析和预测，特别适用于特征未知的情况，并在丙氨酸二肽数据降维和分析Lorenz吸引子和癌症研究中细胞信号问题方面取得了显著改进。 |
| [^4] | [Chunking: Forgetting Matters in Continual Learning even without Changing Tasks.](http://arxiv.org/abs/2310.02206) | 分块是连续学习的重要组成部分，占据实验中离线学习性能下降的约一半。当前的连续学习算法没有解决分块问题，只有在数据分布没有变化时表现与普通SGD训练相仿。 |
| [^5] | [Understanding Transferable Representation Learning and Zero-shot Transfer in CLIP.](http://arxiv.org/abs/2310.00927) | 本文研究了CLIP中的可转移表示学习和零样本传递，提出了一个新的CLIP类型方法，在基准数据集上取得了更好的性能。 |
| [^6] | [Robust, randomized preconditioning for kernel ridge regression.](http://arxiv.org/abs/2304.12465) | 针对核岭回归问题，本文引入了两种强健的随机预处理技术，分别解决了全数据KRR问题和限制版KRR问题，克服了以往预处理器的故障模式。 |

# 详细

[^1]: 通过最优输运的自动异常值矫正

    Automatic Outlier Rectification via Optimal Transport

    [https://arxiv.org/abs/2403.14067](https://arxiv.org/abs/2403.14067)

    提出了一种自动异常值矫正机制，通过将矫正和估计集成到联合优化框架中，利用最优输运和凹成本函数来检测和移除异常值，并选择最佳分布来执行估计任务

    

    在本文中，我们提出了一个新颖的概念框架，使用具有凹成本函数的最优输运来检测异常值。传统的异常值检测方法通常使用两阶段流程：首先检测并移除异常值，然后在清洁数据上执行估计。然而，这种方法并没有将异常值移除与估计任务联系起来，留下了改进的空间。为了解决这一局限性，我们提出了一种自动异常值矫正机制，将矫正和估计集成到一个联合优化框架中。我们首先利用具有凹成本函数的最优输运距离来构建概率分布空间中的矫正集合。然后，我们选择在矫正集合中的最佳分布来执行估计任务。值得注意的是，我们在本文中引入的凹成本函数是使我们的估计器具有关键性的因素。

    arXiv:2403.14067v1 Announce Type: cross  Abstract: In this paper, we propose a novel conceptual framework to detect outliers using optimal transport with a concave cost function. Conventional outlier detection approaches typically use a two-stage procedure: first, outliers are detected and removed, and then estimation is performed on the cleaned data. However, this approach does not inform outlier removal with the estimation task, leaving room for improvement. To address this limitation, we propose an automatic outlier rectification mechanism that integrates rectification and estimation within a joint optimization framework. We take the first step to utilize an optimal transport distance with a concave cost function to construct a rectification set in the space of probability distributions. Then, we select the best distribution within the rectification set to perform the estimation task. Notably, the concave cost function we introduced in this paper is the key to making our estimator e
    
[^2]: Adjustment Identification Distance: 一种用于因果结构学习的调整识别距离

    Adjustment Identification Distance: A gadjid for Causal Structure Learning

    [https://arxiv.org/abs/2402.08616](https://arxiv.org/abs/2402.08616)

    gadjid软件包提供了一种用于因果结构学习的调整识别距离，通过引入框架来计算因果距离，这些距离能够高效评估因果发现算法学习的图形，并且在处理大规模图形时具有较高的性能。

    

    通过因果发现算法学习的图形的评估是困难的：两个图形之间不同的边的数量不能反映出它们在建议因果效应的识别公式方面有何不同。我们引入了一个框架，用于开发图形之间的因果距离，其中包括有向无环图的结构干预距离作为一种特殊情况。我们利用这个框架开发了改进的基于调整的距离，以及对完成的部分有向无环图和因果序列的扩展。我们开发了多项式时间可达性算法来高效计算距离。在我们的gadjid软件包中（在https://github.com/CausalDisco/gadjid上开源），我们提供了我们的距离实现；它们的运行速度比结构干预距离快几个数量级，从而为以前无法扩展的图形尺寸提供了一个因果发现的成功指标。

    Evaluating graphs learned by causal discovery algorithms is difficult: The number of edges that differ between two graphs does not reflect how the graphs differ with respect to the identifying formulas they suggest for causal effects. We introduce a framework for developing causal distances between graphs which includes the structural intervention distance for directed acyclic graphs as a special case. We use this framework to develop improved adjustment-based distances as well as extensions to completed partially directed acyclic graphs and causal orders. We develop polynomial-time reachability algorithms to compute the distances efficiently. In our package gadjid (open source at https://github.com/CausalDisco/gadjid), we provide implementations of our distances; they are orders of magnitude faster than the structural intervention distance and thereby provide a success metric for causal discovery that scales to graph sizes that were previously prohibitive.
    
[^3]: 对Koopman模态分解进行特征化处理

    Featurizing Koopman Mode Decomposition

    [https://arxiv.org/abs/2312.09146](https://arxiv.org/abs/2312.09146)

    本文提出了一种命名为FKMD的先进KMD技术，通过时间嵌入和马氏距离缩放，可以增强对高维动力系统的分析和预测，特别适用于特征未知的情况，并在丙氨酸二肽数据降维和分析Lorenz吸引子和癌症研究中细胞信号问题方面取得了显著改进。

    

    本文介绍了一种先进的Koopman模态分解（KMD）技术：命名为特征化Koopman模态分解（FKMD），该技术利用时间嵌入和马氏距离缩放来增强对高维动力系统的分析和预测。时间嵌入扩展了观测空间，更好地捕捉基础流形结构，而应用于核函数或随机傅里叶特征的马氏距离缩放，则根据系统的动态调整观测值。这有助于在不事先知道良好特征的情况下对KMD进行特征化处理。我们发现，FKMD中的马氏距离缩放可用于对丙氨酸二肽数据进行有效的降维。我们还展示了FKMD如何改善对高维Lorenz吸引子和癌症研究中的细胞信号问题的预测。

    arXiv:2312.09146v3 Announce Type: replace-cross  Abstract: This article introduces an advanced Koopman mode decomposition (KMD) technique -- coined Featurized Koopman Mode Decomposition (FKMD) -- that uses time embedding and Mahalanobis scaling to enhance analysis and prediction of high dimensional dynamical systems. The time embedding expands the observation space to better capture underlying manifold structure, while the Mahalanobis scaling, applied to kernel or random Fourier features, adjusts observations based on the system's dynamics. This aids in featurizing KMD in cases where good features are not a priori known. We find that the Mahalanobis scaling from FKMD can be used for effective dimensionality reduction of alanine dipeptide data. We also show that FKMD improves predictions for a high-dimensional Lorenz attractor and a cell signaling problem from cancer research.
    
[^4]: 分块：即使在不改变任务的情况下在连续学习中遗忘也很重要

    Chunking: Forgetting Matters in Continual Learning even without Changing Tasks. (arXiv:2310.02206v1 [cs.LG])

    [http://arxiv.org/abs/2310.02206](http://arxiv.org/abs/2310.02206)

    分块是连续学习的重要组成部分，占据实验中离线学习性能下降的约一半。当前的连续学习算法没有解决分块问题，只有在数据分布没有变化时表现与普通SGD训练相仿。

    

    在连续学习（CL）的研究中，主要关注动态变化的数据分布所带来的问题。然而，CL可以分解为两个子问题：（a）数据分布的变化，以及（b）处理数据被分成块的事实，因此在任何时间点上只有一部分数据可用于训练。在这项工作中，我们关注后者的子问题--数据的分块--并注意到以前对CL文献中关于分块的分析很少。我们显示出分块是CL的重要组成部分，在我们的实验中占据了离线学习性能下降的约一半。此外，我们的结果显示，当前的CL算法没有解决分块子问题，只有在数据分布没有变化时才能表现出与普通SGD训练一样的水平。我们分析了为什么在数据块上进行学习时性能会下降，并发现遗忘是一个经常被看作是问题的原因。

    Work on continual learning (CL) has largely focused on the problems arising from the dynamically-changing data distribution. However, CL can be decomposed into two sub-problems: (a) shifts in the data distribution, and (b) dealing with the fact that the data is split into chunks and so only a part of the data is available to be trained on at any point in time. In this work, we look at the latter sub-problem -- the chunking of data -- and note that previous analysis of chunking in the CL literature is sparse. We show that chunking is an important part of CL, accounting for around half of the performance drop from offline learning in our experiments. Furthermore, our results reveal that current CL algorithms do not address the chunking sub-problem, only performing as well as plain SGD training when there is no shift in the data distribution. We analyse why performance drops when learning occurs on chunks of data, and find that forgetting, which is often seen to be a problem due to distri
    
[^5]: 理解CLIP中的可转移表示学习和零样本传递

    Understanding Transferable Representation Learning and Zero-shot Transfer in CLIP. (arXiv:2310.00927v1 [cs.LG])

    [http://arxiv.org/abs/2310.00927](http://arxiv.org/abs/2310.00927)

    本文研究了CLIP中的可转移表示学习和零样本传递，提出了一个新的CLIP类型方法，在基准数据集上取得了更好的性能。

    

    多模态学习因其能够利用不同数据源（例如文本和图像）的信息来提高模型性能而日益受到关注。近年来，CLIP作为一种有效的方法，采用视觉-语言对比预训练来学习联合图像和文本表示，并在零样本学习和文本引导的自然图像生成方面表现出非凡的性能。尽管CLIP在实践中取得了巨大的成功，但其理论理解仍然困难。在本文中，我们正式研究了CLIP中的可转移表示学习，并展示了不同模态的特征如何对齐。我们还分析了其在下游任务中的零样本传递性能。受到我们分析的启发，我们提出了一种新的CLIP类型方法，在基准数据集上实现了比CLIP和其他最先进方法更好的性能。

    Multi-modal learning has become increasingly popular due to its ability to leverage information from different data sources (e.g., text and images) to improve the model performance. Recently, CLIP has emerged as an effective approach that employs vision-language contrastive pretraining to learn joint image and text representations and exhibits remarkable performance in zero-shot learning and text-guided natural image generation. Despite the huge practical success of CLIP, its theoretical understanding remains elusive. In this paper, we formally study transferrable representation learning underlying CLIP and demonstrate how features from different modalities get aligned. We also analyze its zero-shot transfer performance on the downstream tasks. Inspired by our analysis, we propose a new CLIP-type approach, which achieves better performance than CLIP and other state-of-the-art methods on benchmark datasets.
    
[^6]: 强健的随机预处理方法解决核岭回归问题

    Robust, randomized preconditioning for kernel ridge regression. (arXiv:2304.12465v1 [math.NA])

    [http://arxiv.org/abs/2304.12465](http://arxiv.org/abs/2304.12465)

    针对核岭回归问题，本文引入了两种强健的随机预处理技术，分别解决了全数据KRR问题和限制版KRR问题，克服了以往预处理器的故障模式。

    

    本论文介绍了两种随机预处理技术，用于强健地解决具有中大规模数据点（$10^4 \leq N \leq 10^7$）的核岭回归（KRR）问题。第一种方法，RPCholesky预处理，能够在假设核矩阵特征值有足够快速的多项式衰减的情况下，以$O（N ^ 2）$算法操作准确地解决全数据KRR问题。第二种方法，KRILL预处理，以$O（（N + k ^ 2）k \ logk）$的代价，为KRR问题的限制版本提供准确的解决方案，该版本涉及$k \ll N$选择的数据中心。所提出的方法解决了广泛的KRR问题，克服了以前的KRR预处理器的故障模式，使它们成为实际应用的理想选择。

    This paper introduces two randomized preconditioning techniques for robustly solving kernel ridge regression (KRR) problems with a medium to large number of data points ($10^4 \leq N \leq 10^7$). The first method, RPCholesky preconditioning, is capable of accurately solving the full-data KRR problem in $O(N^2)$ arithmetic operations, assuming sufficiently rapid polynomial decay of the kernel matrix eigenvalues. The second method, KRILL preconditioning, offers an accurate solution to a restricted version of the KRR problem involving $k \ll N$ selected data centers at a cost of $O((N + k^2) k \log k)$ operations. The proposed methods solve a broad range of KRR problems and overcome the failure modes of previous KRR preconditioners, making them ideal for practical applications.
    

