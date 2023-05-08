# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Verifiable Learning for Robust Tree Ensembles.](http://arxiv.org/abs/2305.03626) | 本论文提出了一种可验证学习的方法，即训练易于验证的限制模型类来解决决策树集成的 NP-hard 问题，并成功设计出一种新的训练算法，使得在多项式时间内可以进行安全验证，而且仍保持着该领域最好的鲁棒性能。 |
| [^2] | [Optimizing Hyperparameters with Conformal Quantile Regression.](http://arxiv.org/abs/2305.03623) | 该论文提出了利用合服量化回归优化超参数，相比高斯过程，该方法对观测噪声做出最少的假设，更真实鲁棒。此外，作者还提出了在多保真度设置中聚合结果的方法，在实际任务中优于传统方法。 |
| [^3] | [Differentially Private Topological Data Analysis.](http://arxiv.org/abs/2305.03609) | 本文尝试使用差分隐私实现拓扑数据分析并生成接近最优的私有持久图，提出使用 $L^1$-距离计算持久图并采用指数机制保护隐私，成功实现在隐私保护和数据分析之间的平衡。 |
| [^4] | [Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient.](http://arxiv.org/abs/2305.03571) | 本论文利用随机策略梯度（SPG）强化学习，成功设计了一种无需通道模型的语义通信系统，能够传输意义而非精确版本，达到了信息速率节省的目的。 |
| [^5] | [The geometry of financial institutions -- Wasserstein clustering of financial data.](http://arxiv.org/abs/2305.03565) | 本文提出了一种新的算法，Wasserstein聚类，用于处理金融机构的复杂数据，有效地解决了缺失值和基于特定特征识别聚类所面临的挑战。该算法可用于监管者的监管工作，并在其领域取得了良好的效果。 |
| [^6] | [Contrastive Graph Clustering in Curvature Spaces.](http://arxiv.org/abs/2305.03555) | 该论文提出了一种基于异构曲率空间的对比图聚类方法CONGREGATE，该方法在各种基准数据集上都取得了最先进的性能，展示了其有效性。 |
| [^7] | [Random Smoothing Regularization in Kernel Gradient Descent Learning.](http://arxiv.org/abs/2305.03531) | 本文提出了一种随机平滑正则化的框架，能够自适应地、有效地学习属于经典Sobolev空间范围内的各种真实函数，通过引入噪声避免过拟合，该方法可以在较快的速度下实现最优收敛率。 |
| [^8] | [Sparsifying Bayesian neural networks with latent binary variables and normalizing flows.](http://arxiv.org/abs/2305.03395) | 本论文介绍了一种新的方法来稀疏化贝叶斯神经网络，使用潜在二进制变量和归一化流，实现了网络在测试时的自动稀疏化，而且结果表明这个方法在准确性上能够与现有的稀疏化方法相媲美。 |
| [^9] | [Decentralized diffusion-based learning under non-parametric limited prior knowledge.](http://arxiv.org/abs/2305.03295) | 本文提出在非参数情况下，仅通过相邻节点之间的信息传播，避免数据交换的分散扩散学习算法。 |
| [^10] | [Demystifying Softmax Gating in Gaussian Mixture of Experts.](http://arxiv.org/abs/2305.03288) | 本文提出了新的参数Vononoi损失函数并建立了MLE的收敛速度来解决高斯混合专家模型中的Softmax门控问题，研究表明该门控与高斯分布中的专家函数通过偏微分方程相互作用，是一个复杂依赖关系。 |
| [^11] | [A Bootstrap Algorithm for Fast Supervised Learning.](http://arxiv.org/abs/2305.03099) | 本文探讨了一种Bootstrap算法，该算法可以通过自助法、重抽样和线性回归来更新隐藏层的加权连接，从而达到更快的收敛速度。 |
| [^12] | [Transferablility of coVariance Neural Networks and Application to Interpretable Brain Age Prediction using Anatomical Features.](http://arxiv.org/abs/2305.01807) | 本研究首次从理论上研究了基于协方差神经网络的可转移性，证明了当数据集的协方差矩阵收敛到一个极限对象时，VNN能够展现出性能可转移性。多尺度神经影像数据集可以在多个尺度上研究脑部，并且可以验证VNN的可转移性。 |
| [^13] | [Enhancing Robustness of Gradient-Boosted Decision Trees through One-Hot Encoding and Regularization.](http://arxiv.org/abs/2304.13761) | 通过独热编码和正则化提高梯度提升决策树的鲁棒性，研究表明对带有$L_1$或$L_2$正则化的线性回归形式进行拟合可提高GBDT模型的鲁棒性。 |
| [^14] | [Sparse Cholesky Factorization for Solving Nonlinear PDEs via Gaussian Processes.](http://arxiv.org/abs/2304.01294) | 本文提出了一种稀疏Cholesky分解算法，用于高斯过程求解非线性偏微分方程，能够有效处理高维和畸形域的问题。 |
| [^15] | [Toward Large Kernel Models.](http://arxiv.org/abs/2302.02605) | 本文提出了一种构建大规模通用核模型的方法，这解决了传统核机器中模型大小与数据大小相互耦合的问题，使其能够在大数据集上进行训练。 |
| [^16] | [Posterior Regularization on Bayesian Hierarchical Mixture Clustering.](http://arxiv.org/abs/2105.06903) | 本文提出了一种后验正则化方法来改进贝叶斯分层混合聚类模型，在每个层级对节点实施最大间隔约束以增强集群的分离。 |
| [^17] | [Learning Node Representations against Perturbations.](http://arxiv.org/abs/2008.11416) | 本文讨论如何在GNN中针对扰动学习节点表示，并提出了稳定-可识别GNN反对扰动 (SIGNNAP) 模型以无监督形式学习可靠的节点表示。 |
| [^18] | [Uncertainty Quantification for Bayesian Optimization.](http://arxiv.org/abs/2002.01569) | 本文提出了一种评估贝叶斯优化算法输出不确定性的新方法，通过构建置信区间实现。这一理论提供了所有现有的顺序取样策略和停止准则的统一不确定性量化框架。 |
| [^19] | [Finding Outliers in Gaussian Model-Based Clustering.](http://arxiv.org/abs/1907.01136) | 研究提出了一种修剪异常值的算法，该算法删除最不可能出现的数据点，然后用符合参考分布的对数似然度进行修剪，从而固有估计异常值的数量。 |
| [^20] | [Fast and Robust Rank Aggregation against Model Misspecification.](http://arxiv.org/abs/1905.12341) | 本文提出了CoarsenRank，其具有鲁棒性，适用于模型错误特化。它采用由粗到精的方案来处理用户收集的信息，并利用排名空间中的几何结构来更好地模拟聚合过程。在实验中验证了CoarsenRank的有效性。 |

# 详细

[^1]: 鲁棒决策树集成的可验证学习

    Verifiable Learning for Robust Tree Ensembles. (arXiv:2305.03626v1 [cs.LG])

    [http://arxiv.org/abs/2305.03626](http://arxiv.org/abs/2305.03626)

    本论文提出了一种可验证学习的方法，即训练易于验证的限制模型类来解决决策树集成的 NP-hard 问题，并成功设计出一种新的训练算法，使得在多项式时间内可以进行安全验证，而且仍保持着该领域最好的鲁棒性能。

    

    在测试时间内验证机器学习模型对抗攻击的鲁棒性是一个重要的研究问题。不幸的是，先前的研究确定，对于决策树集成，这个问题是 NP-hard ，因此对于特定的输入来说是不可解的。在本文中，我们确定了一类受限决策树集成，称为 large-spread 集成，其允许在多项式时间内运行安全验证算法。然后，我们提出了一种新方法，称为可验证学习，该方法倡导训练这种易于验证的受限模型类。我们通过设计一种新的训练算法，从标记数据中自动学习 large-spread 决策树集成来展示这种方法的益处，从而使其能够在多项式时间内进行安全验证。公开可用数据集上的实验结果证实，使用我们的算法训练的 large-spread 集成可以在几秒钟内使用标准半定编程求解器进行验证，同时对抗当前最先进的攻击具有竞争力的性能。

    Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on publicly available datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using stand
    
[^2]: 利用合服量化回归优化超参数

    Optimizing Hyperparameters with Conformal Quantile Regression. (arXiv:2305.03623v1 [cs.LG])

    [http://arxiv.org/abs/2305.03623](http://arxiv.org/abs/2305.03623)

    该论文提出了利用合服量化回归优化超参数，相比高斯过程，该方法对观测噪声做出最少的假设，更真实鲁棒。此外，作者还提出了在多保真度设置中聚合结果的方法，在实际任务中优于传统方法。

    

    很多现有的超参数优化算法依赖于基于模型的优化工具，它们可以学习代理模型来指导搜索。高斯过程是默认的代理模型，因为它们可以捕捉不确定性，但是它们对观测噪声做出强烈的假设，这在实践中可能是不合理的。在这项工作中，我们提出利用合服量化回归，该方法对观测噪声做出最少的假设，因此更真实和鲁棒地建模目标函数，并在实证基准上快速实现超参数优化收敛。为了在多保真度设置中应用我们的方法，我们提出了一种简单而有效的技术，通过聚合不同资源水平上观察到的结果，在许多实际任务中优于传统方法。

    Many state-of-the-art hyperparameter optimization (HPO) algorithms rely on model-based optimizers that learn surrogate models of the target function to guide the search. Gaussian processes are the de facto surrogate model due to their ability to capture uncertainty but they make strong assumptions about the observation noise, which might not be warranted in practice. In this work, we propose to leverage conformalized quantile regression which makes minimal assumptions about the observation noise and, as a result, models the target function in a more realistic and robust fashion which translates to quicker HPO convergence on empirical benchmarks. To apply our method in a multi-fidelity setting, we propose a simple, yet effective, technique that aggregates observed results across different resource levels and outperforms conventional methods across many empirical tasks.
    
[^3]: 差分隐私在拓扑数据分析中的应用

    Differentially Private Topological Data Analysis. (arXiv:2305.03609v1 [stat.ML])

    [http://arxiv.org/abs/2305.03609](http://arxiv.org/abs/2305.03609)

    本文尝试使用差分隐私实现拓扑数据分析并生成接近最优的私有持久图，提出使用 $L^1$-距离计算持久图并采用指数机制保护隐私，成功实现在隐私保护和数据分析之间的平衡。

    

    本文是首篇尝试使用差分隐私实现拓扑数据分析并生成接近最优的私有持久图。我们通过瓶颈距离分析持久图的灵敏度，发现常用的 \v{C}ech 复形的灵敏度并不会随着样本量 $n$ 的增加而降低，这使得 \v{C}ech 复形持久图难以隐私化。作为替代方法，我们提出使用 $L^1$-距离来计算持久图，发现其灵敏度为 $O(1/n)$。基于灵敏度分析，我们提出采用指数机制，其效用函数定义为 $L^1$-DTM 持久图的瓶颈距离。同时，我们还推导出了我们隐私机制的精度上下界；得到的界限表明我们的机制隐私误差接近最优。我们展示了我们的私有持久图方法的性能。

    This paper is the first to attempt differentially private (DP) topological data analysis (TDA), producing near-optimal private persistence diagrams. We analyze the sensitivity of persistence diagrams in terms of the bottleneck distance, and we show that the commonly used \v{C}ech complex has sensitivity that does not decrease as the sample size $n$ increases. This makes it challenging for the persistence diagrams of \v{C}ech complexes to be privatized. As an alternative, we show that the persistence diagram obtained by the $L^1$-distance to measure (DTM) has sensitivity $O(1/n)$. Based on the sensitivity analysis, we propose using the exponential mechanism whose utility function is defined in terms of the bottleneck distance of the $L^1$-DTM persistence diagrams. We also derive upper and lower bounds of the accuracy of our privacy mechanism; the obtained bounds indicate that the privacy error of our mechanism is near-optimal. We demonstrate the performance of our privatized persistence
    
[^4]: 基于随机策略梯度的模型无关语义通信强化学习

    Model-free Reinforcement Learning of Semantic Communication by Stochastic Policy Gradient. (arXiv:2305.03571v1 [eess.SP])

    [http://arxiv.org/abs/2305.03571](http://arxiv.org/abs/2305.03571)

    本论文利用随机策略梯度（SPG）强化学习，成功设计了一种无需通道模型的语义通信系统，能够传输意义而非精确版本，达到了信息速率节省的目的。

    

    受机器学习工具在无线通信方面的成功启发，韦弗（Weaver）于1949年提出的语义通信概念引起了人们的关注。它打破了香农经典的设计范例，旨在传输消息的意义，即语义，而不是精确版本，从而实现信息速率节省。在这项工作中，我们应用了随机策略梯度（SPG）来设计一种基于强化学习的语义通信系统，不需要已知或可微分通道模型，这是实际部署的关键步骤。此外，我们从最大化接收和目标变量之间的互信息出发，激发了将SPG用于经典和语义通信的动机。数值结果表明，我们的方法达到了与基于重新参数化技巧的模型感知方法相当的性能，尽管收敛速度有所降低。

    Motivated by the recent success of Machine Learning tools in wireless communications, the idea of semantic communication by Weaver from 1949 has gained attention. It breaks with Shannon's classic design paradigm by aiming to transmit the meaning, i.e., semantics, of a message instead of its exact version, allowing for information rate savings. In this work, we apply the Stochastic Policy Gradient (SPG) to design a semantic communication system by reinforcement learning, not requiring a known or differentiable channel model a crucial step towards deployment in practice. Further, we motivate the use of SPG for both classic and semantic communication from the maximization of the mutual information between received and target variables. Numerical results show that our approach achieves comparable performance to a model-aware approach based on the reparametrization trick, albeit with a decreased convergence rate.
    
[^5]: 金融机构的几何形态--金融数据的Wasserstein聚类

    The geometry of financial institutions -- Wasserstein clustering of financial data. (arXiv:2305.03565v1 [stat.ML])

    [http://arxiv.org/abs/2305.03565](http://arxiv.org/abs/2305.03565)

    本文提出了一种新的算法，Wasserstein聚类，用于处理金融机构的复杂数据，有效地解决了缺失值和基于特定特征识别聚类所面临的挑战。该算法可用于监管者的监管工作，并在其领域取得了良好的效果。

    

    不断增加的各种有趣对象的细节和大数据的可用性使得有必要开发将这些信息压缩成代表性和可理解的地图的方法。金融监管是一个展示这种需求的领域，因为监管机构需要从金融机构获取多样化的数据，有时是高度细粒度的，以监督和评估他们的活动。然而，处理和分析这样的数据可能是一项艰巨的任务，尤其是考虑到处理缺失值和基于特定特征识别聚类所面临的挑战。为了解决这些挑战，我们提出了一种适用于概率分布的Lloyd算法变体，并使用广义Wasserstein重心构建表示不同对象上的给定数据的度量空间，从而应对金融监管背景下监管者面临的具体挑战。我们相信这种方法在金融监管领域具有实用价值。

    The increasing availability of granular and big data on various objects of interest has made it necessary to develop methods for condensing this information into a representative and intelligible map. Financial regulation is a field that exemplifies this need, as regulators require diverse and often highly granular data from financial institutions to monitor and assess their activities. However, processing and analyzing such data can be a daunting task, especially given the challenges of dealing with missing values and identifying clusters based on specific features.  To address these challenges, we propose a variant of Lloyd's algorithm that applies to probability distributions and uses generalized Wasserstein barycenters to construct a metric space which represents given data on various objects in condensed form. By applying our method to the financial regulation context, we demonstrate its usefulness in dealing with the specific challenges faced by regulators in this domain. We beli
    
[^6]: 曲率空间中对比图聚类

    Contrastive Graph Clustering in Curvature Spaces. (arXiv:2305.03555v1 [cs.LG])

    [http://arxiv.org/abs/2305.03555](http://arxiv.org/abs/2305.03555)

    该论文提出了一种基于异构曲率空间的对比图聚类方法CONGREGATE，该方法在各种基准数据集上都取得了最先进的性能，展示了其有效性。

    

    图聚类一直是一个长期研究的话题，在近年来的深度学习方法中取得了显着的成功。尽管如此，我们观察到仍然存在一些重要问题尚未得到解决。一方面，从几何角度进行图聚类具有吸引力，但很少涉及到它的几何聚类空间。另一方面，对比学习可以提高深度图聚类的效果，但通常会在图增强或难例挖掘方面面临困难。为了填补这一空白，我们重新思考图聚类问题，并尝试首次引入异构曲率空间到图聚类问题中。相应地，我们提出了一个名为CONGREGATE的新颖的端到端对比图聚类模型，用Ricci曲率解决几何图聚类。为了支持几何聚类，我们构建了一个理论上支撑的异构曲率空间框架，可以捕捉图的各种曲率特征。我们提出的方法在几个基准数据集上实现了最先进的性能，展示了我们基于曲率的几何图聚类方法的有效性。

    Graph clustering is a longstanding research topic, and has achieved remarkable success with the deep learning methods in recent years. Nevertheless, we observe that several important issues largely remain open. On the one hand, graph clustering from the geometric perspective is appealing but has rarely been touched before, as it lacks a promising space for geometric clustering. On the other hand, contrastive learning boosts the deep graph clustering but usually struggles in either graph augmentation or hard sample mining. To bridge this gap, we rethink the problem of graph clustering from geometric perspective and, to the best of our knowledge, make the first attempt to introduce a heterogeneous curvature space to graph clustering problem. Correspondingly, we present a novel end-to-end contrastive graph clustering model named CONGREGATE, addressing geometric graph clustering with Ricci curvatures. To support geometric clustering, we construct a theoretically grounded Heterogeneous Curv
    
[^7]: 核梯度下降学习中的随机平滑正则化

    Random Smoothing Regularization in Kernel Gradient Descent Learning. (arXiv:2305.03531v1 [stat.ML])

    [http://arxiv.org/abs/2305.03531](http://arxiv.org/abs/2305.03531)

    本文提出了一种随机平滑正则化的框架，能够自适应地、有效地学习属于经典Sobolev空间范围内的各种真实函数，通过引入噪声避免过拟合，该方法可以在较快的速度下实现最优收敛率。

    

    随机平滑数据增强是一种独特的正则化形式，可以通过向输入数据引入噪声来防止过拟合，鼓励模型学习更广泛的特征。尽管在各种应用中都取得了成功，但随机平滑的正则化能力缺乏系统的研究。在本文中，我们旨在通过提出一个随机平滑正则化的框架，能够自适应地、有效地学习属于经典 Sobolev 空间范围内的各种真实函数。具体而言，我们研究了两种基础的函数空间：低固有维度的 Sobolev 空间，其中包括 $D$ 维欧几里德空间或低维子流形作为特例，以及具有张量结构的混合平滑 Sobolev 空间。通过使用随机平滑正则化作为新型卷积平滑核，我们可以在这些情况下实现最优收敛率。

    Random smoothing data augmentation is a unique form of regularization that can prevent overfitting by introducing noise to the input data, encouraging the model to learn more generalized features. Despite its success in various applications, there has been a lack of systematic study on the regularization ability of random smoothing. In this paper, we aim to bridge this gap by presenting a framework for random smoothing regularization that can adaptively and effectively learn a wide range of ground truth functions belonging to the classical Sobolev spaces. Specifically, we investigate two underlying function spaces: the Sobolev space of low intrinsic dimension, which includes the Sobolev space in $D$-dimensional Euclidean space or low-dimensional sub-manifolds as special cases, and the mixed smooth Sobolev space with a tensor structure. By using random smoothing regularization as novel convolution-based smoothing kernels, we can attain optimal convergence rates in these cases using a ke
    
[^8]: 用潜在二进制变量和归一化流来稀疏化贝叶斯神经网络

    Sparsifying Bayesian neural networks with latent binary variables and normalizing flows. (arXiv:2305.03395v1 [stat.ML])

    [http://arxiv.org/abs/2305.03395](http://arxiv.org/abs/2305.03395)

    本论文介绍了一种新的方法来稀疏化贝叶斯神经网络，使用潜在二进制变量和归一化流，实现了网络在测试时的自动稀疏化，而且结果表明这个方法在准确性上能够与现有的稀疏化方法相媲美。

    

    人工神经网络（ANN）是现代许多应用中强大的机器学习方法，如面部识别、机器翻译和癌症诊断。ANN的一个常见问题是它们通常具有数百万或数十亿个可训练参数，并且因此倾向于过度拟合训练数据。这在需要可靠的不确定性估计的应用中特别有问题。贝叶斯神经网络（BNN）可以改善这一问题，因为它们包含参数不确定性。此外，潜在二进制贝叶斯神经网络（LBBNN）通过允许将权重打开或关闭，从而在权重和结构的联合空间中启用推断，也考虑了结构不确定性。本文将考虑LBBNN方法的两个扩展：首先，通过使用局部重参数化技巧（LRT）直接采样隐藏单元，我们得到了更加计算有效的算法。更重要的是，通过使用归一化流，我们可以近似潜在二进制变量的后验分布，从而在测试时实现网络的稀疏化。我们实验证明，我们提出的方法与现有的稀疏化技术相比，能够获得竞争性的结果，同时保持类似的准确性。

    Artificial neural networks (ANNs) are powerful machine learning methods used in many modern applications such as facial recognition, machine translation, and cancer diagnostics. A common issue with ANNs is that they usually have millions or billions of trainable parameters, and therefore tend to overfit to the training data. This is especially problematic in applications where it is important to have reliable uncertainty estimates. Bayesian neural networks (BNN) can improve on this, since they incorporate parameter uncertainty. In addition, latent binary Bayesian neural networks (LBBNN) also take into account structural uncertainty by allowing the weights to be turned on or off, enabling inference in the joint space of weights and structures. In this paper, we will consider two extensions to the LBBNN method: Firstly, by using the local reparametrization trick (LRT) to sample the hidden units directly, we get a more computationally efficient algorithm. More importantly, by using normal
    
[^9]: 非参数有限先验知识下的分散扩散学习

    Decentralized diffusion-based learning under non-parametric limited prior knowledge. (arXiv:2305.03295v1 [stat.ML])

    [http://arxiv.org/abs/2305.03295](http://arxiv.org/abs/2305.03295)

    本文提出在非参数情况下，仅通过相邻节点之间的信息传播，避免数据交换的分散扩散学习算法。

    

    我们研究在噪声环境中，从局部代理的测量结果中学习非线性现象 m 的扩散网络学习问题。对于分散的网络，仅在直接相邻节点之间传播信息，我们提出了一种非参数学习算法，避免了原始数据交换，仅需要对 m 有轻微的先验知识。对所提出的方法进行了非渐近估计误差界的导出，并通过模拟实验说明了它的潜在应用。

    We study the problem of diffusion-based network learning of a nonlinear phenomenon, $m$, from local agents' measurements collected in a noisy environment. For a decentralized network and information spreading merely between directly neighboring nodes, we propose a non-parametric learning algorithm, that avoids raw data exchange and requires only mild \textit{a priori} knowledge about $m$. Non-asymptotic estimation error bounds are derived for the proposed method. Its potential applications are illustrated through simulation experiments.
    
[^10]: 解密高斯混合专家模型中的Softmax门控问题

    Demystifying Softmax Gating in Gaussian Mixture of Experts. (arXiv:2305.03288v1 [stat.ML])

    [http://arxiv.org/abs/2305.03288](http://arxiv.org/abs/2305.03288)

    本文提出了新的参数Vononoi损失函数并建立了MLE的收敛速度来解决高斯混合专家模型中的Softmax门控问题，研究表明该门控与高斯分布中的专家函数通过偏微分方程相互作用，是一个复杂依赖关系。

    

    理解Softmax门控高斯混合专家模型的参数估计一直是文献中长期未解决的问题。这主要是由于三个基本理论挑战与Softmax门控相关：（i）只能识别参数的平移；（ii）Softmax门控和高斯分布中专家函数之间通过偏微分方程的内在相互作用；（iii）Softmax门控高斯混合专家模型的条件密度的分子和分母之间的复杂依赖关系。我们通过提出新的参数Vononoi损失函数并建立MLE的收敛速度来解决这些挑战，用于解决这些模型的参数估计。当专家数量未知且超额指定时，我们的发现表明MLE的速率与一组多项式方程的可解性问题有关。

    Understanding parameter estimation of softmax gating Gaussian mixture of experts has remained a long-standing open problem in the literature. It is mainly due to three fundamental theoretical challenges associated with the softmax gating: (i) the identifiability only up to the translation of the parameters; (ii) the intrinsic interaction via partial differential equation between the softmax gating and the expert functions in Gaussian distribution; (iii) the complex dependence between the numerator and denominator of the conditional density of softmax gating Gaussian mixture of experts. We resolve these challenges by proposing novel Vononoi loss functions among parameters and establishing the convergence rates of the maximum likelihood estimator (MLE) for solving parameter estimation in these models. When the number of experts is unknown and over-specified, our findings show a connection between the rate of MLE and a solvability problem of a system of polynomial equations.
    
[^11]: 一种用于快速有监督学习的Bootstrap算法

    A Bootstrap Algorithm for Fast Supervised Learning. (arXiv:2305.03099v1 [cs.LG])

    [http://arxiv.org/abs/2305.03099](http://arxiv.org/abs/2305.03099)

    本文探讨了一种Bootstrap算法，该算法可以通过自助法、重抽样和线性回归来更新隐藏层的加权连接，从而达到更快的收敛速度。

    

    训练神经网络（NN）通常依赖某种类型的曲线跟随方法，例如梯度下降（GD）（和随机梯度下降（SGD）），ADADELTA，ADAM或有限内存算法。这些算法的收敛通常依赖于访问大量的观测值以实现高精度，并且对于某些函数类，这些算法可能需要多个epoch的数据点才能进行。本文探讨了一种不同的技术，可以实现更快的收敛速度，尤其是对于浅层的网络而言。它不是曲线跟随，而是依赖于“分离”隐藏层并通过自助法、重抽样和线性回归来更新它们的加权连接。通过利用重抽样的观测值，本方法的收敛被实证地显示出快速和需要更少的数据点：特别是，我们的实验表明，我们只需要少量的数据点即可。

    Training a neural network (NN) typically relies on some type of curve-following method, such as gradient descent (GD) (and stochastic gradient descent (SGD)), ADADELTA, ADAM or limited memory algorithms. Convergence for these algorithms usually relies on having access to a large quantity of observations in order to achieve a high level of accuracy and, with certain classes of functions, these algorithms could take multiple epochs of data points to catch on. Herein, a different technique with the potential of achieving dramatically better speeds of convergence, especially for shallow networks, is explored: it does not curve-follow but rather relies on 'decoupling' hidden layers and on updating their weighted connections through bootstrapping, resampling and linear regression. By utilizing resampled observations, the convergence of this process is empirically shown to be remarkably fast and to require a lower amount of data points: in particular, our experiments show that one needs a fra
    
[^12]: 基于协方差神经网络的可转移学习和应用于解释性脑龄预测

    Transferablility of coVariance Neural Networks and Application to Interpretable Brain Age Prediction using Anatomical Features. (arXiv:2305.01807v1 [cs.LG])

    [http://arxiv.org/abs/2305.01807](http://arxiv.org/abs/2305.01807)

    本研究首次从理论上研究了基于协方差神经网络的可转移性，证明了当数据集的协方差矩阵收敛到一个极限对象时，VNN能够展现出性能可转移性。多尺度神经影像数据集可以在多个尺度上研究脑部，并且可以验证VNN的可转移性。

    

    图卷积网络（GCN）利用基于拓扑图的卷积操作来组合图上的信息进行推理任务。我们最近的工作中，通过使用协方差矩阵作为图来设计了一种类似于传统PCA数据分析方法的协方差神经网络（VNN），并具有显著的优势。本文首先从理论上研究了VNN的可转移性。可转移性的概念是从学习模型可以在“兼容”的数据集上泛化的直观期望中产生的。我们展示了VNN从GCN继承的无标度数据处理架构，并证明当数据集的协方差矩阵收敛到一个极限对象时，VNN能够展现出性能可转移性。多尺度神经影像数据集可以在多个尺度上研究脑部，并且可以验证VNN的可转移性。

    Graph convolutional networks (GCN) leverage topology-driven graph convolutional operations to combine information across the graph for inference tasks. In our recent work, we have studied GCNs with covariance matrices as graphs in the form of coVariance neural networks (VNNs) that draw similarities with traditional PCA-driven data analysis approaches while offering significant advantages over them. In this paper, we first focus on theoretically characterizing the transferability of VNNs. The notion of transferability is motivated from the intuitive expectation that learning models could generalize to "compatible" datasets (possibly of different dimensionalities) with minimal effort. VNNs inherit the scale-free data processing architecture from GCNs and here, we show that VNNs exhibit transferability of performance over datasets whose covariance matrices converge to a limit object. Multi-scale neuroimaging datasets enable the study of the brain at multiple scales and hence, can validate
    
[^13]: 通过独热编码和正则化提高梯度提升决策树的鲁棒性

    Enhancing Robustness of Gradient-Boosted Decision Trees through One-Hot Encoding and Regularization. (arXiv:2304.13761v1 [stat.ML])

    [http://arxiv.org/abs/2304.13761](http://arxiv.org/abs/2304.13761)

    通过独热编码和正则化提高梯度提升决策树的鲁棒性，研究表明对带有$L_1$或$L_2$正则化的线性回归形式进行拟合可提高GBDT模型的鲁棒性。

    

    梯度提升决策树(GBDT)是一种广泛应用的高效机器学习方法，用于表格数据建模。然而，它们复杂的结构可能导致模型对未见数据中的小协变量扰动的鲁棒性较低。本研究应用独热编码将GBDT模型转换为线性框架，通过将每个树叶编码为一个虚拟变量。这允许使用线性回归技术，以及一种新颖的风险分解方法来评估GBDT模型对协变量扰动的鲁棒性。我们建议通过重新拟合其带有$L_1$或$L_2$正则化的线性回归形式，提高GBDT模型的鲁棒性。理论结果表明了正则化对模型性能和鲁棒性的影响。在数值实验中，证明了所提出的正则化方法可以提高独热编码GBDT模型的鲁棒性。

    Gradient-boosted decision trees (GBDT) are widely used and highly effective machine learning approach for tabular data modeling. However, their complex structure may lead to low robustness against small covariate perturbation in unseen data. In this study, we apply one-hot encoding to convert a GBDT model into a linear framework, through encoding of each tree leaf to one dummy variable. This allows for the use of linear regression techniques, plus a novel risk decomposition for assessing the robustness of a GBDT model against covariate perturbations. We propose to enhance the robustness of GBDT models by refitting their linear regression forms with $L_1$ or $L_2$ regularization. Theoretical results are obtained about the effect of regularization on the model performance and robustness. It is demonstrated through numerical experiments that the proposed regularization approach can enhance the robustness of the one-hot-encoded GBDT models.
    
[^14]: 通过高斯过程求解非线性偏微分方程的稀疏Cholesky分解方法

    Sparse Cholesky Factorization for Solving Nonlinear PDEs via Gaussian Processes. (arXiv:2304.01294v1 [math.NA])

    [http://arxiv.org/abs/2304.01294](http://arxiv.org/abs/2304.01294)

    本文提出了一种稀疏Cholesky分解算法，用于高斯过程求解非线性偏微分方程，能够有效处理高维和畸形域的问题。

    

    本文研究了一个高斯过程框架求解一般非线性偏微分方程的计算可伸缩性。这个框架把求解PDE转化为解非线性约束下的二次优化问题。其复杂度的瓶颈在于利用高斯过程的协方差核及其在拟合点的偏导数进行点对点计算所得到的密集协方差矩阵的计算。我们提出了一种基于Diracs和导数测量的新排列顺序的稀疏Cholesky分解算法用于计算此类协方差矩阵。我们严格地确定了该Cholesky分解的稀疏模式，并量化了相应Vecchia近似的指数收敛精度，在Kullback-Leibler距离度量下达到最优。这使我们能够以$O(N\log^d(N/\epsilon))$的空间复杂度和$O(N\log^{d+2}(N/\epsilon))$的时间复杂度计算$\epsilon$-近似的逆Cholesky因子。其中，$N$表示拟合点的数量，$d$为物理域的维数。我们在几个高维（最高可达到$d=50$）和畸形域的基准问题上展示了这种方法的有效性。

    We study the computational scalability of a Gaussian process (GP) framework for solving general nonlinear partial differential equations (PDEs). This framework transforms solving PDEs to solving quadratic optimization problem with nonlinear constraints. Its complexity bottleneck lies in computing with dense kernel matrices obtained from pointwise evaluations of the covariance kernel of the GP and its partial derivatives at collocation points.  We present a sparse Cholesky factorization algorithm for such kernel matrices based on the near-sparsity of the Cholesky factor under a new ordering of Diracs and derivative measurements. We rigorously identify the sparsity pattern and quantify the exponentially convergent accuracy of the corresponding Vecchia approximation of the GP, which is optimal in the Kullback-Leibler divergence. This enables us to compute $\epsilon$-approximate inverse Cholesky factors of the kernel matrices with complexity $O(N\log^d(N/\epsilon))$ in space and $O(N\log^{
    
[^15]: 向大核模型迈进

    Toward Large Kernel Models. (arXiv:2302.02605v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02605](http://arxiv.org/abs/2302.02605)

    本文提出了一种构建大规模通用核模型的方法，这解决了传统核机器中模型大小与数据大小相互耦合的问题，使其能够在大数据集上进行训练。

    

    最近的研究表明，与深度神经网络（DNN）相比，核机器在小数据集上的表现通常可以达到或超过DNN。核机器的兴趣受到其在某些情况下等效于宽神经网络的发现的推动。然而，DNN的一个关键特征是它们能够独立地扩展模型大小和训练数据量，而在传统的核机器中，模型大小与数据大小是相互耦合的。由于这种耦合，将核机器扩展到大数据是计算上具有挑战性的。在本文中，我们提供了一种构建大规模通用核模型的方法，这是核机器的一般化，通过解耦模型和数据，允许在大数据集上进行训练。具体地，我们引入了基于投影双重预处理SGD的EigenPro 3.0算法，并展示了使用现有核方法不可能实现的模型和数据规模的扩展。

    Recent studies indicate that kernel machines can often perform similarly or better than deep neural networks (DNNs) on small datasets. The interest in kernel machines has been additionally bolstered by the discovery of their equivalence to wide neural networks in certain regimes. However, a key feature of DNNs is their ability to scale the model size and training data size independently, whereas in traditional kernel machines model size is tied to data size. Because of this coupling, scaling kernel machines to large data has been computationally challenging. In this paper, we provide a way forward for constructing large-scale general kernel models, which are a generalization of kernel machines that decouples the model and data, allowing training on large datasets. Specifically, we introduce EigenPro 3.0, an algorithm based on projected dual preconditioned SGD and show scaling to model and data sizes which have not been possible with existing kernel methods.
    
[^16]: 贝叶斯分层混合聚类中的后验正则化

    Posterior Regularization on Bayesian Hierarchical Mixture Clustering. (arXiv:2105.06903v7 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2105.06903](http://arxiv.org/abs/2105.06903)

    本文提出了一种后验正则化方法来改进贝叶斯分层混合聚类模型，在每个层级对节点实施最大间隔约束以增强集群的分离。

    

    贝叶斯分层混合聚类通过在生成过程中用层级狄利克雷过程混合模型(HDPMM)替换传统的高斯-高斯核来实现从父节点到子节点的扩散，从而改进了传统的贝叶斯分层聚类。然而，BHMC可能会产生具有高节点方差的树，表明在较高层级之间的节点之间存在较弱的分离。为了解决这个问题，我们采用了后验正则化(Posterior Regularization)，它对每个层级的节点实施最大间隔约束以增强集群的分离。我们阐述了如何将PR应用于BHMC，并证明了它在改进BHMC模型方面的有效性。

    Bayesian hierarchical mixture clustering (BHMC) improves traditionalBayesian hierarchical clustering by replacing conventional Gaussian-to-Gaussian kernels with a Hierarchical Dirichlet Process Mixture Model(HDPMM) for parent-to-child diffusion in the generative process. However,BHMC may produce trees with high nodal variance, indicating weak separation between nodes at higher levels. To address this issue, we employ Posterior Regularization, which imposes max-margin constraints on nodes at every level to enhance cluster separation. We illustrate how to apply PR toBHMC and demonstrate its effectiveness in improving the BHMC model.
    
[^17]: 针对扰动学习节点表示

    Learning Node Representations against Perturbations. (arXiv:2008.11416v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2008.11416](http://arxiv.org/abs/2008.11416)

    本文讨论如何在GNN中针对扰动学习节点表示，并提出了稳定-可识别GNN反对扰动 (SIGNNAP) 模型以无监督形式学习可靠的节点表示。

    

    最近的图神经网络 (GNN) 在节点表示学习方面取得了显著的表现。GNN的成功关键因素之一是节点表示上的“平滑”属性。尽管如此，大多数GNN模型对图输入的扰动很脆弱，可能会学习到不可靠的节点表示。本文研究如何在GNN中针对扰动学习节点表示。具体而言，我们认为节点表示应在输入略微扰动时保持稳定，并且应能够识别不同结构的节点表示，这两者分别被称为节点表示的“稳定性”和“可识别性”。为此，我们提出了一种名为稳定-可识别GNN反对扰动 (SIGNNAP) 的新模型，该模型以无监督的方式学习可靠的节点表示。SIGNNAP通过对比目标来形式化“稳定性”和“可识别性”，并保留了...

    Recent graph neural networks (GNN) has achieved remarkable performance in node representation learning. One key factor of GNN's success is the \emph{smoothness} property on node representations. Despite this, most GNN models are fragile to the perturbations on graph inputs and could learn unreliable node representations. In this paper, we study how to learn node representations against perturbations in GNN. Specifically, we consider that a node representation should remain stable under slight perturbations on the input, and node representations from different structures should be identifiable, which two are termed as the \emph{stability} and \emph{identifiability} on node representations, respectively. To this end, we propose a novel model called Stability-Identifiability GNN Against Perturbations (SIGNNAP) that learns reliable node representations in an unsupervised manner. SIGNNAP formalizes the \emph{stability} and \emph{identifiability} by a contrastive objective and preserves the 
    
[^18]: 贝叶斯优化中的不确定性量化

    Uncertainty Quantification for Bayesian Optimization. (arXiv:2002.01569v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2002.01569](http://arxiv.org/abs/2002.01569)

    本文提出了一种评估贝叶斯优化算法输出不确定性的新方法，通过构建置信区间实现。这一理论提供了所有现有的顺序取样策略和停止准则的统一不确定性量化框架。

    

    贝叶斯优化是一种全局优化技术。在贝叶斯优化中，目标函数被建模为高斯过程的实现。尽管高斯过程的假设意味着贝叶斯优化输出的随机分布，但对于此不确定性的量化在文献中很少有研究。在本文中，我们提出了一种新方法来评估贝叶斯优化算法的输出不确定性，该方法通过构建目标函数最大点（或值）的置信区间来实现。这些区间可以高效地计算，其置信水平由本研究中新开发的顺序高斯过程回归的统一误差界保证。我们的理论为所有现有的顺序取样策略和停止准则提供了统一的不确定性量化框架。

    Bayesian optimization is a class of global optimization techniques. In Bayesian optimization, the underlying objective function is modeled as a realization of a Gaussian process. Although the Gaussian process assumption implies a random distribution of the Bayesian optimization outputs, quantification of this uncertainty is rarely studied in the literature. In this work, we propose a novel approach to assess the output uncertainty of Bayesian optimization algorithms, which proceeds by constructing confidence regions of the maximum point (or value) of the objective function. These regions can be computed efficiently, and their confidence levels are guaranteed by the uniform error bounds for sequential Gaussian process regression newly developed in the present work. Our theory provides a unified uncertainty quantification framework for all existing sequential sampling policies and stopping criteria.
    
[^19]: 基于高斯模型的聚类中异常值的发现

    Finding Outliers in Gaussian Model-Based Clustering. (arXiv:1907.01136v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/1907.01136](http://arxiv.org/abs/1907.01136)

    研究提出了一种修剪异常值的算法，该算法删除最不可能出现的数据点，然后用符合参考分布的对数似然度进行修剪，从而固有估计异常值的数量。

    

    无监督分类或聚类常常受到异常值的影响。然而，在无监督分类中处理异常值的研究较少。目前，异常值算法可分为两大类：异常点包含方法和修剪方法，这些方法通常需要预先指定要删除的数据点的数量。本文利用样本马氏距离的贝塔分布导出了一个近似分布，用于有限高斯混合模型子集的对数似然度。提出了一种算法，该算法删除最不可能出现的数据点，即判定为异常值，直到对数似然度符合参考分布。这导致了一种固有估计异常值数量的修剪方法。

    Unsupervised classification, or clustering, is a problem often plagued by outliers, yet there is a paucity of work on handling outliers in unsupervised classification. Outlier algorithms tend to fall into two broad categories: outlier inclusion methods and trimming methods, which often require pre-specification of the number of points to remove. The fact that sample Mahalanobis distance is beta-distributed is used to derive an approximate distribution for the log-likelihoods of subset finite Gaussian mixture models. An algorithm is proposed that removes the least likely points, which are deemed outliers, until the log-likelihoods adhere to the reference distribution. This results in a trimming method which inherently estimates the number of outliers present.
    
[^20]: 快速、鲁棒的排名聚合算法在模型错误特化方面的应用

    Fast and Robust Rank Aggregation against Model Misspecification. (arXiv:1905.12341v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1905.12341](http://arxiv.org/abs/1905.12341)

    本文提出了CoarsenRank，其具有鲁棒性，适用于模型错误特化。它采用由粗到精的方案来处理用户收集的信息，并利用排名空间中的几何结构来更好地模拟聚合过程。在实验中验证了CoarsenRank的有效性。

    

    排名聚合算法（RA）用于将来自不同用户的偏好总结成一个总排序。在假设用户同质化的情况下，这项工作基本无误。但在复杂的实际情况下，由于同质假设无法验证，RA的模型错误特化出现了。现有的健壮RA通常采用排名模型扩充来解释额外的噪声，其中收集到的偏好可以被视为单个附加到理想偏好上的扰动。由于健壮闯闯亮RAs大多依赖于某些扰动假设，因此它们不能很好地推广到真实世界中对不确定噪声的偏好。在本文中，我们提出了CoarsenRank，它对模型的错误特化具有鲁棒性。具体而言，我们的CoarsenRank具有以下特性：（1）CoarsenRank是针对轻微的模型错误特化而设计的，假设与模型假设一致的理想偏好位于收集到的偏好的附近。（2）CoarsenRank采用由粗到精的方案来捕捉收集到的来自不同用户的细微差异， 并利用聚类技术。（3）CoarsenRank利用排名空间的几何结构来更好地模拟聚合过程，并进一步增强其鲁棒性。我们进行了广泛的实验，涵盖合成和真实世界的数据集，验证了CoarsenRank的有效性。

    In rank aggregation (RA), a collection of preferences from different users are summarized into a total order under the assumption of homogeneity of users. Model misspecification in RA arises since the homogeneity assumption fails to be satisfied in the complex real-world situation. Existing robust RAs usually resort to an augmentation of the ranking model to account for additional noises, where the collected preferences can be treated as a noisy perturbation of idealized preferences. Since the majority of robust RAs rely on certain perturbation assumptions, they cannot generalize well to agnostic noise-corrupted preferences in the real world. In this paper, we propose CoarsenRank, which possesses robustness against model misspecification. Specifically, the properties of our CoarsenRank are summarized as follows: (1) CoarsenRank is designed for mild model misspecification, which assumes there exist the ideal preferences (consistent with model assumption) that locates in a neighborhood o
    

