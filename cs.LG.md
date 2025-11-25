# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Limits of Assumption-free Tests for Algorithm Performance](https://arxiv.org/abs/2402.07388) | 这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。 |
| [^2] | [Adversarial Training for Physics-Informed Neural Networks.](http://arxiv.org/abs/2310.11789) | 这篇论文提出了一种名为AT-PINNs的对抗训练策略，通过对抗样本的微调来增强物理信息神经网络（PINNs）的鲁棒性，并且可以进行具有时间因果关系的推断。 |
| [^3] | [Automatic nodule identification and differentiation in ultrasound videos to facilitate per-nodule examination.](http://arxiv.org/abs/2310.06339) | 本研究针对超声图像中结节异质外观导致的难以进行逐个结节检查的问题，构建了一个基于深度学习的结节重新识别系统，在数百个乳腺超声视频上取得了令人满意的结果。 |
| [^4] | [Bivariate DeepKriging for Large-scale Spatial Interpolation of Wind Fields.](http://arxiv.org/abs/2307.08038) | 本文提出了一种名为双变量深度克里金的方法，它利用空间相关的深度神经网络(DNN)和嵌入层以及基于自助法和集成DNN的无分布不确定性量化方法，用于大规模空间插值风场的预测和估计。 |
| [^5] | [Convergence and concentration properties of constant step-size SGD through Markov chains.](http://arxiv.org/abs/2306.11497) | 本文通过马尔科夫链研究了常步长随机梯度下降的性质，证明了迭代收敛于一个不变分布，并获得了高置信度边界。 |
| [^6] | [When Does Bottom-up Beat Top-down in Hierarchical Community Detection?.](http://arxiv.org/abs/2306.00833) | 本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。 |
| [^7] | [Fairness in Streaming Submodular Maximization over a Matroid Constraint.](http://arxiv.org/abs/2305.15118) | 这篇论文研究了在一个Matroid约束下流式子模最大化中的公平性问题，提供了流式算法和不可能的结果来权衡效率、质量和公平性，并在现实世界应用中进行了实证验证。 |
| [^8] | [VeML: An End-to-End Machine Learning Lifecycle for Large-scale and High-dimensional Data.](http://arxiv.org/abs/2304.13037) | VeML是一种专门用于大规模高维数据的端到端机器学习生命周期的版本管理系统，在解决生命周期高成本问题、数据相似性计算和数据模式分析等关键问题方面表现出色。 |
| [^9] | [High-dimensional multi-view clustering methods.](http://arxiv.org/abs/2303.08582) | 本论文比较了两类高维多视角聚类方法（基于图和基于子空间），重点关注了如何处理高阶相关性，并在基准数据集上进行了实验研究。 |
| [^10] | [Learning a Discrete Set of Optimal Allocation Rules in a Queueing System with Unknown Service Rate.](http://arxiv.org/abs/2202.02419) | 该论文研究了在具有未知到达和服务率的排队系统中的入场控制问题。通过观察到到达时间和系统状态，我们旨在设计一种调度策略，以最大化调度员的长期平均回报。标准的强化学习方法不适用于此问题，因为调度员无法观察到服务时间和离开时间。 |

# 详细

[^1]: 无假设测试算法性能的限制

    The Limits of Assumption-free Tests for Algorithm Performance

    [https://arxiv.org/abs/2402.07388](https://arxiv.org/abs/2402.07388)

    这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。

    

    算法评价和比较是机器学习和统计学中基本的问题，一个算法在给定的建模任务中表现如何，哪个算法表现最佳？许多方法已经开发出来评估算法性能，通常基于交叉验证策略，将感兴趣的算法在不同的数据子集上重新训练，并评估其在留出数据点上的性能。尽管广泛使用这些程序，但对于这些方法的理论性质尚未完全理解。在这项工作中，我们探讨了在有限的数据量下回答这些问题的一些基本限制。特别地，我们区分了两个问题: 算法$A$在大小为$n$的训练集上学习问题有多好，以及在特定大小为$n$的训练数据集上运行$A$所产生的特定拟合模型有多好？我们的主要结果证明，对于任何将算法视为黑盒的测试方法，无法准确地回答这两个问题。

    Algorithm evaluation and comparison are fundamental questions in machine learning and statistics -- how well does an algorithm perform at a given modeling task, and which algorithm performs best? Many methods have been developed to assess algorithm performance, often based around cross-validation type strategies, retraining the algorithm of interest on different subsets of the data and assessing its performance on the held-out data points. Despite the broad use of such procedures, the theoretical properties of these methods are not yet fully understood. In this work, we explore some fundamental limits for answering these questions with limited amounts of data. In particular, we make a distinction between two questions: how good is an algorithm $A$ at the problem of learning from a training set of size $n$, versus, how good is a particular fitted model produced by running $A$ on a particular training data set of size $n$?   Our main results prove that, for any test that treats the algor
    
[^2]: 物理信息神经网络的对抗训练

    Adversarial Training for Physics-Informed Neural Networks. (arXiv:2310.11789v1 [cs.LG])

    [http://arxiv.org/abs/2310.11789](http://arxiv.org/abs/2310.11789)

    这篇论文提出了一种名为AT-PINNs的对抗训练策略，通过对抗样本的微调来增强物理信息神经网络（PINNs）的鲁棒性，并且可以进行具有时间因果关系的推断。

    

    物理信息神经网络在解决偏微分方程问题上显示出巨大的潜力。然而，由于不足的鲁棒性，普通的PINNs在解决涉及多尺度行为或具有尖锐或振荡特征的复杂PDE时经常面临挑战。为了解决这些问题，我们基于投影梯度下降对抗攻击提出了一种对抗训练策略，被称为AT-PINNs。AT-PINNs通过对抗样本的微调来增强PINNs的鲁棒性，可以准确识别模型失效位置并在训练过程中引导模型专注于这些区域。AT-PINNs还可以通过选择围绕时间初始值的初始拟合点来进行因果推断。我们将AT-PINNs应用于具有多尺度系数的椭圆方程、具有多峰解的泊松方程、具有尖锐解的Burgers方程以及Allen-Cahn方程。

    Physics-informed neural networks have shown great promise in solving partial differential equations. However, due to insufficient robustness, vanilla PINNs often face challenges when solving complex PDEs, especially those involving multi-scale behaviors or solutions with sharp or oscillatory characteristics. To address these issues, based on the projected gradient descent adversarial attack, we proposed an adversarial training strategy for PINNs termed by AT-PINNs. AT-PINNs enhance the robustness of PINNs by fine-tuning the model with adversarial samples, which can accurately identify model failure locations and drive the model to focus on those regions during training. AT-PINNs can also perform inference with temporal causality by selecting the initial collocation points around temporal initial values. We implement AT-PINNs to the elliptic equation with multi-scale coefficients, Poisson equation with multi-peak solutions, Burgers equation with sharp solutions and the Allen-Cahn equati
    
[^3]: 自动识别和区分超声视频中的结节，以便进行逐结节检查

    Automatic nodule identification and differentiation in ultrasound videos to facilitate per-nodule examination. (arXiv:2310.06339v1 [eess.IV])

    [http://arxiv.org/abs/2310.06339](http://arxiv.org/abs/2310.06339)

    本研究针对超声图像中结节异质外观导致的难以进行逐个结节检查的问题，构建了一个基于深度学习的结节重新识别系统，在数百个乳腺超声视频上取得了令人满意的结果。

    

    超声是健康筛查中重要的诊断技术，具有无创、经济、无辐射等优点，因此在结节的诊断中被广泛应用。然而，超声图像中，单个结节在不同的切面视图下可能呈现出异质的外观，这使得逐个结节检查变得困难。超声检查通常依赖于超声师的专业知识和临床经验。超声师通常通过检查结节特征和周围结构（如腺体和导管）来区分不同的结节，这是繁琐且耗时的。为了解决这个问题，我们收集了数百个乳腺超声视频，并建立了一个结节重新识别系统，包括基于深度学习模型的提取器，可以从输入视频片段中提取特征向量，以及实时聚类算法，可以自动将特征向量按结节分组。该系统获得了令人满意的结果。

    Ultrasound is a vital diagnostic technique in health screening, with the advantages of non-invasive, cost-effective, and radiation free, and therefore is widely applied in the diagnosis of nodules. However, it relies heavily on the expertise and clinical experience of the sonographer. In ultrasound images, a single nodule might present heterogeneous appearances in different cross-sectional views which makes it hard to perform per-nodule examination. Sonographers usually discriminate different nodules by examining the nodule features and the surrounding structures like gland and duct, which is cumbersome and time-consuming. To address this problem, we collected hundreds of breast ultrasound videos and built a nodule reidentification system that consists of two parts: an extractor based on the deep learning model that can extract feature vectors from the input video clips and a real-time clustering algorithm that automatically groups feature vectors by nodules. The system obtains satisfa
    
[^4]: 大规模空间插值风场的双变量深度克里金方法

    Bivariate DeepKriging for Large-scale Spatial Interpolation of Wind Fields. (arXiv:2307.08038v1 [stat.ML])

    [http://arxiv.org/abs/2307.08038](http://arxiv.org/abs/2307.08038)

    本文提出了一种名为双变量深度克里金的方法，它利用空间相关的深度神经网络(DNN)和嵌入层以及基于自助法和集成DNN的无分布不确定性量化方法，用于大规模空间插值风场的预测和估计。

    

    高空间分辨率的风场数据对于气候、海洋和气象研究中的各种应用至关重要。由于风数据往往具有非高斯分布、高空间变异性和异质性，因此对具有两个维度速度的双变量风场进行大规模空间插值或下缩放是一项具有挑战性的任务。在空间统计学中，常用cokriging来预测双变量空间场。然而，cokriging预测器除了对高斯过程有效外，并不是最优的。此外，对于大型数据集，cokriging计算量巨大。在本文中，我们提出了一种称为双变量深度克里金的方法，它是一个由空间径向基函数构建的空间相关的深度神经网络(DNN)和嵌入层，用于双变量空间数据预测。然后，我们基于自助法和集成DNN开发了一种无分布不确定性量化方法。我们提出的方法优于传统的cokriging方法。

    High spatial resolution wind data are essential for a wide range of applications in climate, oceanographic and meteorological studies. Large-scale spatial interpolation or downscaling of bivariate wind fields having velocity in two dimensions is a challenging task because wind data tend to be non-Gaussian with high spatial variability and heterogeneity. In spatial statistics, cokriging is commonly used for predicting bivariate spatial fields. However, the cokriging predictor is not optimal except for Gaussian processes. Additionally, cokriging is computationally prohibitive for large datasets. In this paper, we propose a method, called bivariate DeepKriging, which is a spatially dependent deep neural network (DNN) with an embedding layer constructed by spatial radial basis functions for bivariate spatial data prediction. We then develop a distribution-free uncertainty quantification method based on bootstrap and ensemble DNN. Our proposed approach outperforms the traditional cokriging 
    
[^5]: 基于马尔科夫链的常步长SGD的收敛和集中性质

    Convergence and concentration properties of constant step-size SGD through Markov chains. (arXiv:2306.11497v1 [stat.ML])

    [http://arxiv.org/abs/2306.11497](http://arxiv.org/abs/2306.11497)

    本文通过马尔科夫链研究了常步长随机梯度下降的性质，证明了迭代收敛于一个不变分布，并获得了高置信度边界。

    

    本文考虑使用常步长随机梯度下降（SGD）优化平滑且强凸的目标，并通过马尔科夫链研究其性质。我们证明，对于具有轻微受控方差的无偏梯度估计，迭代以总变差距离收敛于一个不变分布。我们还在与以前工作相比梯度噪声分布的放宽假设下，在Wasserstein-2距离下建立了这种收敛性。由于极限分布的不变性质，我们的分析表明，当这些对于梯度成立时，后者继承了亚高斯或亚指数浓度特性。这允许推导出对于最终估计的高置信度边界。最后，在这种条件下，在线性情况下，对于Polyak-Ruppert序列的尾部，我们获得了一个无维度偏差限制。所有结果均为非渐近性质，并讨论了其后果。

    We consider the optimization of a smooth and strongly convex objective using constant step-size stochastic gradient descent (SGD) and study its properties through the prism of Markov chains. We show that, for unbiased gradient estimates with mildly controlled variance, the iteration converges to an invariant distribution in total variation distance. We also establish this convergence in Wasserstein-2 distance under a relaxed assumption on the gradient noise distribution compared to previous work. Thanks to the invariance property of the limit distribution, our analysis shows that the latter inherits sub-Gaussian or sub-exponential concentration properties when these hold true for the gradient. This allows the derivation of high-confidence bounds for the final estimate. Finally, under such conditions in the linear case, we obtain a dimension-free deviation bound for the Polyak-Ruppert average of a tail sequence. All our results are non-asymptotic and their consequences are discussed thr
    
[^6]: 自下而上何时击败自上而下进行分层社区检测？

    When Does Bottom-up Beat Top-down in Hierarchical Community Detection?. (arXiv:2306.00833v1 [cs.SI])

    [http://arxiv.org/abs/2306.00833](http://arxiv.org/abs/2306.00833)

    本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。

    

    网络的分层聚类是指查找一组社区的树形结构，其中层次结构的较低级别显示更细粒度的社区结构。解决这一问题的算法有两个主要类别：自上而下的算法和自下而上的算法。本文研究了使用自下而上算法恢复分层随机块模型的树形结构和社区结构的理论保证。我们还确定了这种自下而上算法在层次结构的中间层次上达到了确切恢复信息理论阈值。值得注意的是，这些恢复条件相对于现有的自上而下算法的条件来说，限制更少。

    Hierarchical clustering of networks consists in finding a tree of communities, such that lower levels of the hierarchy reveal finer-grained community structures. There are two main classes of algorithms tackling this problem. Divisive ($\textit{top-down}$) algorithms recursively partition the nodes into two communities, until a stopping rule indicates that no further split is needed. In contrast, agglomerative ($\textit{bottom-up}$) algorithms first identify the smallest community structure and then repeatedly merge the communities using a $\textit{linkage}$ method. In this article, we establish theoretical guarantees for the recovery of the hierarchical tree and community structure of a Hierarchical Stochastic Block Model by a bottom-up algorithm. We also establish that this bottom-up algorithm attains the information-theoretic threshold for exact recovery at intermediate levels of the hierarchy. Notably, these recovery conditions are less restrictive compared to those existing for to
    
[^7]: 在一个Matroid约束下流式子模最大化中的公平性

    Fairness in Streaming Submodular Maximization over a Matroid Constraint. (arXiv:2305.15118v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.15118](http://arxiv.org/abs/2305.15118)

    这篇论文研究了在一个Matroid约束下流式子模最大化中的公平性问题，提供了流式算法和不可能的结果来权衡效率、质量和公平性，并在现实世界应用中进行了实证验证。

    

    流式子模最大化是从一个大规模数据集中选择一个代表性子集的自然模型。如果数据点具有敏感属性，如性别或种族，强制公平性以避免偏见和歧视变得重要。这引起了对开发公平机器学习算法的极大兴趣。最近，这样的算法已经被开发用于基于基数约束的单调子模最大化。在本文中，我们研究了这个问题的自然推广到一个Matroid约束。我们提供了流式算法以及不可能的结果，这些结果在效率、质量和公平性之间提供了权衡。我们在一系列知名的现实世界应用中对我们的发现进行了经验证实：基于示例的聚类、电影推荐和社交网络中的最大覆盖。

    Streaming submodular maximization is a natural model for the task of selecting a representative subset from a large-scale dataset. If datapoints have sensitive attributes such as gender or race, it becomes important to enforce fairness to avoid bias and discrimination. This has spurred significant interest in developing fair machine learning algorithms. Recently, such algorithms have been developed for monotone submodular maximization under a cardinality constraint.  In this paper, we study the natural generalization of this problem to a matroid constraint. We give streaming algorithms as well as impossibility results that provide trade-offs between efficiency, quality and fairness. We validate our findings empirically on a range of well-known real-world applications: exemplar-based clustering, movie recommendation, and maximum coverage in social networks.
    
[^8]: VeML：大规模高维数据的端到端机器学习生命周期

    VeML: An End-to-End Machine Learning Lifecycle for Large-scale and High-dimensional Data. (arXiv:2304.13037v1 [cs.LG])

    [http://arxiv.org/abs/2304.13037](http://arxiv.org/abs/2304.13037)

    VeML是一种专门用于大规模高维数据的端到端机器学习生命周期的版本管理系统，在解决生命周期高成本问题、数据相似性计算和数据模式分析等关键问题方面表现出色。

    

    端到端的机器学习生命周期包含许多迭代过程，从数据准备和机器学习模型设计到模型训练，再到部署训练好的模型用于推理。当构建一个机器学习问题的端到端生命周期时，必须设计和执行许多机器学习管道，这会产生大量的生命周期版本。因此，本文介绍了VeML，一种专门用于端到端机器学习生命周期的版本管理系统。我们的系统解决了其他系统没有解决的几个关键问题。首先，我们解决了构建机器学习生命周期的高成本问题，特别是针对大规模和高维数据集。我们通过提议将在我们系统中管理的类似数据集的生命周期转移到新的训练数据来解决这个问题。我们设计了一种基于核心集的算法，可以有效地计算大规模高维数据的相似性。另一个关键问题是由于训练数据和测试数据的差异而导致模型准确性下降。我们开发了一种数据模式分析方法来检测先前使用的数据和新数据之间的差异。我们的系统使用户可以自定义机器学习生命周期工作流，并将生命周期的各个阶段与其API连接起来，作为用户运行自定义代码的桥梁。 VeML已应用于处理多个真实世界的机器学习问题，结果证明了我们的系统的有效性。

    An end-to-end machine learning (ML) lifecycle consists of many iterative processes, from data preparation and ML model design to model training and then deploying the trained model for inference. When building an end-to-end lifecycle for an ML problem, many ML pipelines must be designed and executed that produce a huge number of lifecycle versions. Therefore, this paper introduces VeML, a Version management system dedicated to end-to-end ML Lifecycle. Our system tackles several crucial problems that other systems have not solved. First, we address the high cost of building an ML lifecycle, especially for large-scale and high-dimensional dataset. We solve this problem by proposing to transfer the lifecycle of similar datasets managed in our system to the new training data. We design an algorithm based on the core set to compute similarity for large-scale, high-dimensional data efficiently. Another critical issue is the model accuracy degradation by the difference between training data a
    
[^9]: 高维多视角聚类方法

    High-dimensional multi-view clustering methods. (arXiv:2303.08582v1 [cs.LG])

    [http://arxiv.org/abs/2303.08582](http://arxiv.org/abs/2303.08582)

    本论文比较了两类高维多视角聚类方法（基于图和基于子空间），重点关注了如何处理高阶相关性，并在基准数据集上进行了实验研究。

    

    最近几年，相比于单视角聚类，多视角聚类被广泛应用于数据分析中。它可以提供更多的数据信息，但也带来了一些挑战，如如何组合这些视角或特征。最近的研究主要集中在张量表示上，而不是将数据视为简单的矩阵。这种方法可以处理数据之间的高阶相关性，而基于矩阵的方法则难以捕捉这种相关性。因此，我们将研究和比较这些方法，特别是基于图的聚类和子空间聚类，以及在基准数据集上的实验结果。

    Multi-view clustering has been widely used in recent years in comparison to single-view clustering, for clear reasons, as it offers more insights into the data, which has brought with it some challenges, such as how to combine these views or features. Most of recent work in this field focuses mainly on tensor representation instead of treating the data as simple matrices. This permits to deal with the high-order correlation between the data which the based matrix approach struggles to capture. Accordingly, we will examine and compare these approaches, particularly in two categories, namely graph-based clustering and subspace-based clustering. We will conduct and report experiments of the main clustering methods over a benchmark datasets.
    
[^10]: 学习具有未知服务率的排队系统中一组离散的最优分配规则

    Learning a Discrete Set of Optimal Allocation Rules in a Queueing System with Unknown Service Rate. (arXiv:2202.02419v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2202.02419](http://arxiv.org/abs/2202.02419)

    该论文研究了在具有未知到达和服务率的排队系统中的入场控制问题。通过观察到到达时间和系统状态，我们旨在设计一种调度策略，以最大化调度员的长期平均回报。标准的强化学习方法不适用于此问题，因为调度员无法观察到服务时间和离开时间。

    

    在广泛应用于通信网络、呼叫中心以及设计生产系统、消息系统和基于应用的停车系统等现代应用领域之外的Erlang-B阻塞模型中，考虑到到达和服务率未知的情况下对该系统的入场控制进行研究。在我们的模型中，在每个作业到达时，调度员决定将作业分配给一个可用的服务器或者阻塞它。每个已服务的作业为调度员带来了固定的回报，但也导致了每单位服务时间的成本。我们的目标是设计一种调度策略，基于仅观察到到达时间和每次到达时系统状态的情况，从而最大化调度员的长期平均回报，这反映了对这种系统的现实采样。关键是，调度员既不观察服务时间也不观察离开时间，因此不能应用使用奖励信号的标准强化学习方法。因此，我们发展了我们的学习基于...

    Motivated by the wide range of modern applications of the Erlang-B blocking model beyond communication networks and call centers to sizing and pricing in design production systems, messaging systems, and app-based parking systems, we study admission control for such a system but with unknown arrival and service rates. In our model, at every job arrival, a dispatcher decides to assign the job to an available server or block it. Every served job yields a fixed reward for the dispatcher, but it also results in a cost per unit time of service. Our goal is to design a dispatching policy that maximizes the long-term average reward for the dispatcher based on observing only the arrival times and the state of the system at each arrival that reflects a realistic sampling of such systems. Critically, the dispatcher observes neither the service times nor departure times so that standard reinforcement learning-based approaches that use reward signals do not apply. Hence, we develop our learning-ba
    

