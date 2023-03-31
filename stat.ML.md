# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient distributed representations beyond negative sampling.](http://arxiv.org/abs/2303.17475) | 本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。 |
| [^2] | [Fast inference of latent space dynamics in huge relational event networks.](http://arxiv.org/abs/2303.17460) | 本研究提出了一种适用于巨型关系事件网络的基于可能性的算法，可以快速推理出潜在空间动态，并实现分层推断网络社区动态。 |
| [^3] | [The Graphical Nadaraya-Watson Estimator on Latent Position Models.](http://arxiv.org/abs/2303.17229) | 研究了潜在位置模型上的图形Nadaraya-Watson估计器的性质，对于更复杂的方法有理论指导意义。 |
| [^4] | [Sublinear Convergence Rates of Extragradient-Type Methods: A Survey on Classical and Recent Developments.](http://arxiv.org/abs/2303.17192) | 本文调查了外推梯度方法和其变种的最新进展，并提供了次线性最佳迭代和最后迭代的收敛速率。 |
| [^5] | [Deep Single Image Camera Calibration by Heatmap Regression to Recover Fisheye Images Under ManhattanWorld AssumptionWithout Ambiguity.](http://arxiv.org/abs/2303.17166) | 本文提出一种基于学习的标定方法，使用热度图回归来消除曼哈顿世界假设下鱼眼图片中横向角度歧义，同时恢复旋转和消除鱼眼失真。该方法使用优化的对角线点缓解图像中缺乏消失点的情况，并在实验证明其性能优于现有技术。 |
| [^6] | [Contextual Combinatorial Bandits with Probabilistically Triggered Arms.](http://arxiv.org/abs/2303.17110) | 本文研究了带有概率触发臂的情境组合赌博机，在不同条件下设计了C$^2$-UCB-T算法和VAC$^2$-UCB算法，并分别导出了对应的遗憾值上限，为相关应用提供了理论支持。 |
| [^7] | [Efficient Sampling of Stochastic Differential Equations with Positive Semi-Definite Models.](http://arxiv.org/abs/2303.17109) | 本文提出了一个从正半定随机微分方程中高效采样的方法，可以利用正半定-PSD模型在精度$\varepsilon$下生成iid样本。算法复杂度为$O(T d \log(1/\varepsilon) m^2 + d m^{\beta+1} \log(T)/\varepsilon^2)$，其中$T$是时间步数，$\beta$是Fokker-Planck解的正则性。 |
| [^8] | [Training Neural Networks is NP-Hard in Fixed Dimension.](http://arxiv.org/abs/2303.17045) | 研究了训练具有ReLU和线性阈值激活函数的两层神经网络的固定维度下的NP难度。 回答了两个问题，证明了这两个问题在二维情况下是NP难的，此外在ReLU案例中证明了固定参数问题的参数化固定复杂度维数和ReLU数量的组合参数。 |
| [^9] | [Federated Stochastic Bandit Learning with Unobserved Context.](http://arxiv.org/abs/2303.17043) | 本文提出了一种联邦随机多臂上下文赌博算法以最大化累积奖励，针对未知上下文的情况通过执行特征向量转换解决问题。 |
| [^10] | [Sparse joint shift in multinomial classification.](http://arxiv.org/abs/2303.16971) | 该论文提出了一种稀疏联合偏移模型，用于解决整体数据集偏移问题，提供了传递SJS、修正类后验概率、SJS的可辨认性、SJS与协变量转移关系等新结果。 |
| [^11] | [Leveraging joint sparsity in hierarchical Bayesian learning.](http://arxiv.org/abs/2303.16954) | 本文提出了一种分层贝叶斯学习方法，用于从多个测量向量中推断联合稀疏的参数向量，该方法使用共同的伽马分布超参数来强制联合稀疏性，并在实验中进行了验证。 |
| [^12] | [Are Neural Architecture Search Benchmarks Well Designed? A Deeper Look Into Operation Importance.](http://arxiv.org/abs/2303.16938) | 本论文对当前广泛使用的NAS基准测试进行了经验研究，发现只需一小部分的操作即可生成接近最高性能的架构，同时这些基准测试存在缺点可能影响公平比较并提供不可靠结果。 |
| [^13] | [Non-Asymptotic Lower Bounds For Training Data Reconstruction.](http://arxiv.org/abs/2303.16372) | 本文通过研究差分隐私和度量隐私学习器在对抗者重构错误方面的鲁棒性，得出了非渐进性下界，覆盖了高维情况，且扩展了深度学习算法的隐私分析 |
| [^14] | [Operator learning with PCA-Net: upper and lower complexity bounds.](http://arxiv.org/abs/2303.16317) | 本文发展了PCA-Net的近似理论，得出了通用逼近结果，并识别出了使用PCA-Net进行高效操作学习的潜在障碍：输出分布的复杂性和算子空间的内在复杂性。 |
| [^15] | [Lifting uniform learners via distributional decomposition.](http://arxiv.org/abs/2303.16208) | 本文介绍了一种方法，可以将任何在均匀分布下有效的PAC学习算法转换成一个在任意未知分布下有效的算法，而且对于单调分布，只需要用$\mathcal{D}$中的样本。算法的核心是通过一个算法将$\mathcal{D}$逼近成由子立方体混合而成的混合均匀分布。 |
| [^16] | [Validation of uncertainty quantification metrics: a primer based on the consistency and adaptivity concepts.](http://arxiv.org/abs/2303.07170) | 本文介绍了一种基于一致性和适应性概念的UQ度量验证方法，通过重新审视已有的方法，提高了对UQ度量能力的理解。 |
| [^17] | [Physics-informed Information Field Theory for Modeling Physical Systems with Uncertainty Quantification.](http://arxiv.org/abs/2301.07609) | 该论文扩展了信息场理论(IFT)到物理信息场理论(PIFT)，将描述场的物理定律的信息编码为函数先验。从这个PIFT得出的后验与任何数值方案无关，并且可以捕捉多种模式。 |
| [^18] | [Sliced Optimal Partial Transport.](http://arxiv.org/abs/2212.08049) | 本文提出了一种适用于一维非负测度之间最优偏转运输问题的高效算法，并通过切片的方式定义了切片最优偏转运输距离。 |
| [^19] | [Packed-Ensembles for Efficient Uncertainty Estimation.](http://arxiv.org/abs/2210.09184) | Packed-Ensembles是一种能够在标准神经网络内运行的轻量级结构化集合，它通过精心调节编码空间的维度来设计。该方法在不损失效果的情况下提高了训练和推理速度。 |
| [^20] | [Clustered Graph Matching for Label Recovery and Graph Classification.](http://arxiv.org/abs/2205.03486) | 本论文提出一种利用顶点对齐的平均图，聚类平均图和混淆网络匹配的策略，比起传统的全局平均图策略，可以更有效地提高匹配性能和分类精度。 |
| [^21] | [Random Manifold Sampling and Joint Sparse Regularization for Multi-label Feature Selection.](http://arxiv.org/abs/2204.06445) | 本文提出了一种基于联合约束优化问题的 $\ell_{2,1}$ 和 $\ell_{F}$ 正则化方法来获得最相关的几个特征，并在流形正则化中实现了基于随机游走策略的高度稳健的邻域图。该方法在真实数据集上的比较实验中表现优异。 |
| [^22] | [Approximation bounds for norm constrained neural networks with applications to regression and GANs.](http://arxiv.org/abs/2201.09418) | 本文研究了具范数约束的ReLU神经网络的逼近能力，并证明了对于平滑函数类，这些网络的逼近误差有上下界。此外，应用结果分析了回归和GAN分布估计问题的收敛性，最终证明了当GAN的判别器选择合适的具范数约束的神经网络时，可以实现学习概率分布的最优速率。 |
| [^23] | [Statistically Meaningful Approximation: a Case Study on Approximating Turing Machines with Transformers.](http://arxiv.org/abs/2107.13163) | 本文提出了统计上意义的近似的正式定义，研究了过度参数化的前馈神经网络和变换器的SM近似在布尔电路和图灵机中的应用，重点在于探索近似网络应该具有良好的统计可学性的概念，达到更有意义的近似效果。 |
| [^24] | [Out-of-sample error estimate for robust M-estimators with convex penalty.](http://arxiv.org/abs/2008.11840) | 该论文提出了一种通用的样外误差估计方法，适用于正则化具有凸惩罚的鲁棒$M$-估计，该方法仅通过固定的观测数据依赖于特定量，其中在高维渐近区域中，该估计具有相对误差，具有广泛的适用性。 |
| [^25] | [Variational Wasserstein Barycenters for Geometric Clustering.](http://arxiv.org/abs/2002.10543) | 该论文提出了利用变分Wasserstein质心解决几何聚类问题的方法，特别是Monge WBs与K-means聚类和共同聚类相关，同时还提出了两个新问题——正则化K-means和Wasserstein质心压缩，并演示了VWBs在解决这些聚类相关问题的有效性。 |
| [^26] | [Optimal Experimental Design for Staggered Rollouts.](http://arxiv.org/abs/1911.03764) | 本文研究了隔开式试验的最优设计问题。对于非自适应实验，提出了一个近似最优解；对于自适应实验，提出了一种新算法——精度导向的自适应实验（PGAE）算法，它使用贝叶斯决策理论来最大化估计治疗效果的预期精度。 |

# 详细

[^1]: 超越负采样的高效分布式表示方法

    Efficient distributed representations beyond negative sampling. (arXiv:2303.17475v1 [cs.LG])

    [http://arxiv.org/abs/2303.17475](http://arxiv.org/abs/2303.17475)

    本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。

    

    本文介绍了一种高效的学习分布式表示（也称为嵌入）的方法。该方法通过最小化一个类似于Word2Vec算法中引入并在多个工作中采用的目标函数来实现。优化计算的瓶颈是softmax归一化常数的计算，这需要与样本大小呈二次比例的操作数。这种复杂度不适用于大型数据集，所以负采样是一个常见的解决方法，可以在与样本大小线性相关的时间内获得分布式表示。然而，负采样会改变损失函数，因此解决的是与最初提出的不同的优化问题。我们的贡献在于展示如何通过线性时间估计softmax归一化常数，从而设计了一种有效的优化策略来学习分布式表示。我们使用不同的数据集进行测试，并展示了我们的方法在嵌入质量和训练时间方面优于负采样。

    This article describes an efficient method to learn distributed representations, also known as embeddings. This is accomplished minimizing an objective function similar to the one introduced in the Word2Vec algorithm and later adopted in several works. The optimization computational bottleneck is the calculation of the softmax normalization constants for which a number of operations scaling quadratically with the sample size is required. This complexity is unsuited for large datasets and negative sampling is a popular workaround, allowing one to obtain distributed representations in linear time with respect to the sample size. Negative sampling consists, however, in a change of the loss function and hence solves a different optimization problem from the one originally proposed. Our contribution is to show that the sotfmax normalization constants can be estimated in linear time, allowing us to design an efficient optimization strategy to learn distributed representations. We test our ap
    
[^2]: 巨型关系事件网络中潜在空间动态的快速推断

    Fast inference of latent space dynamics in huge relational event networks. (arXiv:2303.17460v1 [cs.SI])

    [http://arxiv.org/abs/2303.17460](http://arxiv.org/abs/2303.17460)

    本研究提出了一种适用于巨型关系事件网络的基于可能性的算法，可以快速推理出潜在空间动态，并实现分层推断网络社区动态。

    

    关系事件是社交互动的一种类型，有时被称为动态网络。它的动态通常取决于新兴的模式，即内生变量，或外部力量，即外生变量。然而，对于网络中的行为者，尤其是对于大型网络，全面的信息是罕见的。网络分析中的潜在空间方法是解释驱动网络配置的未测量协变量的流行方式。贝叶斯和EM类型的算法已被提出用于推断潜在空间，但是许多社交网络应用程序的规模以及过程（因此是潜在空间）的动态特性使得计算变得极其昂贵。在本项工作中，我们提出了一种基于可能性的算法，可以处理巨型关系事件网络。我们提出了一种嵌入到可解释的潜在空间中的推断网络社区动态的分层策略。节点动态是通过向后概率推断的贝叶斯统计模型实现的。

    Relational events are a type of social interactions, that sometimes are referred to as dynamic networks. Its dynamics typically depends on emerging patterns, so-called endogenous variables, or external forces, referred to as exogenous variables. Comprehensive information on the actors in the network, especially for huge networks, is rare, however. A latent space approach in network analysis has been a popular way to account for unmeasured covariates that are driving network configurations. Bayesian and EM-type algorithms have been proposed for inferring the latent space, but both the sheer size many social network applications as well as the dynamic nature of the process, and therefore the latent space, make computations prohibitively expensive. In this work we propose a likelihood-based algorithm that can deal with huge relational event networks. We propose a hierarchical strategy for inferring network community dynamics embedded into an interpretable latent space. Node dynamics are d
    
[^3]: 潜在位置模型上的图形Nadaraya-Watson估计器

    The Graphical Nadaraya-Watson Estimator on Latent Position Models. (arXiv:2303.17229v1 [stat.ML])

    [http://arxiv.org/abs/2303.17229](http://arxiv.org/abs/2303.17229)

    研究了潜在位置模型上的图形Nadaraya-Watson估计器的性质，对于更复杂的方法有理论指导意义。

    

    鉴于有标记节点的图形，我们对估计器的质量感兴趣，该估计器针对未标记节点预测其标记邻居的观测值的平均值。我们在这个背景下严格研究了浓度属性、方差界和风险界。虽然估计器本身非常简单，数据生成过程对于实际应用过于理想，但我们相信我们的小步骤将有助于更复杂方法（如图形神经网络）的理论理解。

    Given a graph with a subset of labeled nodes, we are interested in the quality of the averaging estimator which for an unlabeled node predicts the average of the observations of its labeled neighbours. We rigorously study concentration properties, variance bounds and risk bounds in this context. While the estimator itself is very simple and the data generating process is too idealistic for practical applications, we believe that our small steps will contribute towards the theoretical understanding of more sophisticated methods such as Graph Neural Networks.
    
[^4]: Extragradient类型方法的次线性收敛速率：对经典和最新进展的调查

    Sublinear Convergence Rates of Extragradient-Type Methods: A Survey on Classical and Recent Developments. (arXiv:2303.17192v1 [math.OC])

    [http://arxiv.org/abs/2303.17192](http://arxiv.org/abs/2303.17192)

    本文调查了外推梯度方法和其变种的最新进展，并提供了次线性最佳迭代和最后迭代的收敛速率。

    

    其中，外推梯度（EG）是G. M. Korpelevich在1976年引入的一种广泛应用于近似解决最小斜率问题和其扩展的方法，如变分不等式和单调包含。多年来，文献中提出并研究了EG的各种变体。最近，由于在机器学习和鲁棒优化中的新应用，这些方法变得越来越流行。本文概述了EG方法及其变种的最新进展，用于近似求解非线性方程和包含，重点关注单调性和协同单调性设置。我们为不同类别的算法提供了统一的收敛分析，重点是次线性最佳迭代和最后迭代的收敛速率。我们还讨论了基于Halpern固定点迭代和Nesterov加速技术的最近加速变体的EG。我们使用简单的论证和基本的数学工具来使我们的方法易于理解和应用。

    The extragradient (EG), introduced by G. M. Korpelevich in 1976, is a well-known method to approximate solutions of saddle-point problems and their extensions such as variational inequalities and monotone inclusions. Over the years, numerous variants of EG have been proposed and studied in the literature. Recently, these methods have gained popularity due to new applications in machine learning and robust optimization. In this work, we survey the latest developments in the EG method and its variants for approximating solutions of nonlinear equations and inclusions, with a focus on the monotonicity and co-hypomonotonicity settings. We provide a unified convergence analysis for different classes of algorithms, with an emphasis on sublinear best-iterate and last-iterate convergence rates. We also discuss recent accelerated variants of EG based on both Halpern fixed-point iteration and Nesterov's accelerated techniques. Our approach uses simple arguments and basic mathematical tools to mak
    
[^5]: 使用热度图回归进行深度单张图片摄像机标定，在曼哈顿世界假设下在不模糊的情况下还原鱼眼图片

    Deep Single Image Camera Calibration by Heatmap Regression to Recover Fisheye Images Under ManhattanWorld AssumptionWithout Ambiguity. (arXiv:2303.17166v1 [cs.CV])

    [http://arxiv.org/abs/2303.17166](http://arxiv.org/abs/2303.17166)

    本文提出一种基于学习的标定方法，使用热度图回归来消除曼哈顿世界假设下鱼眼图片中横向角度歧义，同时恢复旋转和消除鱼眼失真。该方法使用优化的对角线点缓解图像中缺乏消失点的情况，并在实验证明其性能优于现有技术。

    

    在正交世界坐标系中，曼哈顿世界沿着长方体建筑物广泛用于各种计算机视觉任务。然而，曼哈顿世界需要改进，因为图像中的横向角度的原点是任意的，即具有四倍轮换对称的横向角度的歧义。为了解决这个问题，我们提出了一个基于摄像机和行驶方向的道路方向的平角定义。我们提出了一种基于学习的标定方法，它使用热度图回归来消除歧义，类似于姿态估计关键点。与此同时，我们的两个分支网络恢复旋转并从一般场景图像中消除鱼眼失真。为了缓解图像中缺乏消失点的情况，我们引入了具有空间均匀性最佳的对角线点。大量实验证明，我们的方法在曼哈顿世界假设下对鱼眼图像的深度单张图片摄像机标定优于现有技术，没有歧义。

    In orthogonal world coordinates, a Manhattan world lying along cuboid buildings is widely useful for various computer vision tasks. However, the Manhattan world has much room for improvement because the origin of pan angles from an image is arbitrary, that is, four-fold rotational symmetric ambiguity of pan angles. To address this problem, we propose a definition for the pan-angle origin based on the directions of the roads with respect to a camera and the direction of travel. We propose a learning-based calibration method that uses heatmap regression to remove the ambiguity by each direction of labeled image coordinates, similar to pose estimation keypoints. Simultaneously, our two-branched network recovers the rotation and removes fisheye distortion from a general scene image. To alleviate the lack of vanishing points in images, we introduce auxiliary diagonal points that have the optimal 3D arrangement of spatial uniformity. Extensive experiments demonstrated that our method outperf
    
[^6]: 带有概率触发臂的情境组合赌博机

    Contextual Combinatorial Bandits with Probabilistically Triggered Arms. (arXiv:2303.17110v1 [cs.LG])

    [http://arxiv.org/abs/2303.17110](http://arxiv.org/abs/2303.17110)

    本文研究了带有概率触发臂的情境组合赌博机，在不同条件下设计了C$^2$-UCB-T算法和VAC$^2$-UCB算法，并分别导出了对应的遗憾值上限，为相关应用提供了理论支持。

    

    本研究探讨了在捕捉广泛应用范围的一系列平滑条件下的带有概率触发臂的情境组合赌博机(C$^2$MAB-T)，例如情境级联赌博机和情境最大化赌博机。在模拟触发概率(TPM)的条件下，我们设计了C$^2$-UCB-T算法，并提出了一种新的分析方法，实现了一个$\tilde{O}(d\sqrt{KT})$的遗憾值上限，消除了一个可能指数级增长的因子$O(1/p_{\min})$，其中$d$是情境的维数，$p_{\min}$是能被触发的任何臂的最小正概率，批大小$K$是每轮能被触发的臂的最大数量。在方差调制(VM)或触发概率和方差调制(TPVM)条件下，我们提出了一种新的方差自适应算法VAC$^2$-UCB，并导出了一个$\tilde{O}(d\sqrt{T})$的遗憾值上限，该上限与批大小$K$无关。作为一个有价值的副产品，我们发现我们的一个...

    We study contextual combinatorial bandits with probabilistically triggered arms (C$^2$MAB-T) under a variety of smoothness conditions that capture a wide range of applications, such as contextual cascading bandits and contextual influence maximization bandits. Under the triggering probability modulated (TPM) condition, we devise the C$^2$-UCB-T algorithm and propose a novel analysis that achieves an $\tilde{O}(d\sqrt{KT})$ regret bound, removing a potentially exponentially large factor $O(1/p_{\min})$, where $d$ is the dimension of contexts, $p_{\min}$ is the minimum positive probability that any arm can be triggered, and batch-size $K$ is the maximum number of arms that can be triggered per round. Under the variance modulated (VM) or triggering probability and variance modulated (TPVM) conditions, we propose a new variance-adaptive algorithm VAC$^2$-UCB and derive a regret bound $\tilde{O}(d\sqrt{T})$, which is independent of the batch-size $K$. As a valuable by-product, we find our a
    
[^7]: 正半定随机微分方程的高效采样

    Efficient Sampling of Stochastic Differential Equations with Positive Semi-Definite Models. (arXiv:2303.17109v1 [stat.ML])

    [http://arxiv.org/abs/2303.17109](http://arxiv.org/abs/2303.17109)

    本文提出了一个从正半定随机微分方程中高效采样的方法，可以利用正半定-PSD模型在精度$\varepsilon$下生成iid样本。算法复杂度为$O(T d \log(1/\varepsilon) m^2 + d m^{\beta+1} \log(T)/\varepsilon^2)$，其中$T$是时间步数，$\beta$是Fokker-Planck解的正则性。

    

    本文解决了在已知漂移函数和扩散矩阵的情况下，从随机微分方程中高效采样的问题。所提出的方法利用了一个最近的概率模型（正半定-PSD模型）\citep{rudi2021psd}，从中可以获得精度为$\varepsilon$的独立同分布（iid）样本，其成本为$m^2 d \log(1/\varepsilon)$，其中$m$是模型的维度，$d$是空间的维度。所提出的方法包括：首先计算满足与SDE相关联的Fokker-Planck方程（或其分数变体）的PSD模型，误差为$\varepsilon$，然后从生成的PSD模型中采样。假设Fokker-Planck解具有一定的正则性（即$\beta$阶可微性以及其零点的一些几何条件），我们得到一个算法：（a）在准备阶段，获得具有L2距离$\varepsilon$的PSD模型作为真实概率密度函数的估计；（b）在采样阶段，以精度$\varepsilon$生成SDE解的iid样本。所得到的复杂度为$O(T d \log(1/\varepsilon) m^2 + d m^{\beta+1} \log(T)/\varepsilon^2)$，其中$T$是SDE的时间步数，$\beta$是Fokker-Planck解的正则性。

    This paper deals with the problem of efficient sampling from a stochastic differential equation, given the drift function and the diffusion matrix. The proposed approach leverages a recent model for probabilities \citep{rudi2021psd} (the positive semi-definite -- PSD model) from which it is possible to obtain independent and identically distributed (i.i.d.) samples at precision $\varepsilon$ with a cost that is $m^2 d \log(1/\varepsilon)$ where $m$ is the dimension of the model, $d$ the dimension of the space. The proposed approach consists in: first, computing the PSD model that satisfies the Fokker-Planck equation (or its fractional variant) associated with the SDE, up to error $\varepsilon$, and then sampling from the resulting PSD model. Assuming some regularity of the Fokker-Planck solution (i.e. $\beta$-times differentiability plus some geometric condition on its zeros) We obtain an algorithm that: (a) in the preparatory phase obtains a PSD model with L2 distance $\varepsilon$ fr
    
[^8]: 训练神经网络在固定维度上是NP难的

    Training Neural Networks is NP-Hard in Fixed Dimension. (arXiv:2303.17045v1 [cs.CC])

    [http://arxiv.org/abs/2303.17045](http://arxiv.org/abs/2303.17045)

    研究了训练具有ReLU和线性阈值激活函数的两层神经网络的固定维度下的NP难度。 回答了两个问题，证明了这两个问题在二维情况下是NP难的，此外在ReLU案例中证明了固定参数问题的参数化固定复杂度维数和ReLU数量的组合参数。

    

    我们研究了在输入数据维度和隐藏神经元数量方面对两层神经网络进行参数化复杂性的研究，考虑ReLU和线性阈值激活函数。尽管这些问题的计算复杂性近年来已经被多次研究，但仍有几个问题尚未解决。我们回答了Arora et al. [ICLR '18]和Khalife和Basu [IPCO '22]的问题，显示两个问题在二维情况下都是NP难的，这排除了任何常数维度的多项式时间算法。我们还回答了Froese等人[JAIR '22]的问题，证明了具有零培训误差的四个ReLU(或两个线性阈值神经元)的W [1]-hardness。最后，在ReLU案例中，我们展示了参数化固定复杂度维数和ReLU数量的组合参数，如果网络被假定为计算凸映射，则可用于固定参数问题。我们的结果几乎完全解决了这些参数的复杂性状况。

    We study the parameterized complexity of training two-layer neural networks with respect to the dimension of the input data and the number of hidden neurons, considering ReLU and linear threshold activation functions. Albeit the computational complexity of these problems has been studied numerous times in recent years, several questions are still open. We answer questions by Arora et al. [ICLR '18] and Khalife and Basu [IPCO '22] showing that both problems are NP-hard for two dimensions, which excludes any polynomial-time algorithm for constant dimension. We also answer a question by Froese et al. [JAIR '22] proving W[1]-hardness for four ReLUs (or two linear threshold neurons) with zero training error. Finally, in the ReLU case, we show fixed-parameter tractability for the combined parameter number of dimensions and number of ReLUs if the network is assumed to compute a convex map. Our results settle the complexity status regarding these parameters almost completely.
    
[^9]: 无观测上下文的联邦随机赌博学习

    Federated Stochastic Bandit Learning with Unobserved Context. (arXiv:2303.17043v1 [cs.LG])

    [http://arxiv.org/abs/2303.17043](http://arxiv.org/abs/2303.17043)

    本文提出了一种联邦随机多臂上下文赌博算法以最大化累积奖励，针对未知上下文的情况通过执行特征向量转换解决问题。

    

    本文研究了具有未知上下文的联邦随机多臂上下文赌博问题，其中M个代理面临不同的赌博机并协作学习。通信模型由中央服务器组成，并且代理会定期与中央服务器共享其估计结果，以便选择最优动作以最小化总后悔。我们假设精确的上下文不可观察，代理仅观测上下文的分布。例如，当上下文本身是噪声测量或基于预测机制时，就会出现这种情况。我们的目标是开发一种分布式联邦算法，促进代理之间的协作学习，选择一系列最优动作以最大化累积奖励。通过执行特征向量转换，我们提出了一种基于消除的算法，并证明了线性参数化奖励函数的后悔界。最后，我们验证了算法的性能。

    We study the problem of federated stochastic multi-arm contextual bandits with unknown contexts, in which M agents are faced with different bandits and collaborate to learn. The communication model consists of a central server and the agents share their estimates with the central server periodically to learn to choose optimal actions in order to minimize the total regret. We assume that the exact contexts are not observable and the agents observe only a distribution of the contexts. Such a situation arises, for instance, when the context itself is a noisy measurement or based on a prediction mechanism. Our goal is to develop a distributed and federated algorithm that facilitates collaborative learning among the agents to select a sequence of optimal actions so as to maximize the cumulative reward. By performing a feature vector transformation, we propose an elimination-based algorithm and prove the regret bound for linearly parametrized reward functions. Finally, we validated the perfo
    
[^10]: 多项式分类中的稀疏联合偏移

    Sparse joint shift in multinomial classification. (arXiv:2303.16971v1 [stat.ML])

    [http://arxiv.org/abs/2303.16971](http://arxiv.org/abs/2303.16971)

    该论文提出了一种稀疏联合偏移模型，用于解决整体数据集偏移问题，提供了传递SJS、修正类后验概率、SJS的可辨认性、SJS与协变量转移关系等新结果。

    

    稀疏联合偏移（SJS）是一种针对数据集整体偏移的可处理模型，可能会导致特征和标签的边际分布以及后验概率和类条件特征分布的变化。在没有标签观测的情况下，为目标数据集拟合SJS可能会产生标签的有效预测和类先验概率的估计。我们在特征集之间传递SJS方面提供了新的结果，提出了一个基于目标分布的类后验概率的条件修正公式，确定性SJS的可辨认性以及SJS和协变量转移之间的关系。此外，我们指出了用于估计SJS特征的算法中的不一致性，因为它们可能会妨碍寻找最优解。

    Sparse joint shift (SJS) was recently proposed as a tractable model for general dataset shift which may cause changes to the marginal distributions of features and labels as well as the posterior probabilities and the class-conditional feature distributions. Fitting SJS for a target dataset without label observations may produce valid predictions of labels and estimates of class prior probabilities. We present new results on the transmission of SJS from sets of features to larger sets of features, a conditional correction formula for the class posterior probabilities under the target distribution, identifiability of SJS, and the relationship between SJS and covariate shift. In addition, we point out inconsistencies in the algorithms which were proposed for estimating the characteristics of SJS, as they could hamper the search for optimal solutions.
    
[^11]: 利用联合稀疏性的分层贝叶斯学习方法

    Leveraging joint sparsity in hierarchical Bayesian learning. (arXiv:2303.16954v1 [stat.ML])

    [http://arxiv.org/abs/2303.16954](http://arxiv.org/abs/2303.16954)

    本文提出了一种分层贝叶斯学习方法，用于从多个测量向量中推断联合稀疏的参数向量，该方法使用共同的伽马分布超参数来强制联合稀疏性，并在实验中进行了验证。

    

    我们提出了一种分层贝叶斯学习方法，从多个测量向量中推断联合稀疏的参数向量。我们的模型为每个参数向量使用单独的条件高斯先验，并使用共同的伽马分布超参数来强制联合稀疏性。得到的联合稀疏性先验与现有的贝叶斯推断方法相结合，形成了一系列新算法。我们的数值实验，包括多线圈磁共振成像应用，证明了我们的新方法始终优于常用的分层贝叶斯方法。

    We present a hierarchical Bayesian learning approach to infer jointly sparse parameter vectors from multiple measurement vectors. Our model uses separate conditionally Gaussian priors for each parameter vector and common gamma-distributed hyper-parameters to enforce joint sparsity. The resulting joint-sparsity-promoting priors are combined with existing Bayesian inference methods to generate a new family of algorithms. Our numerical experiments, which include a multi-coil magnetic resonance imaging application, demonstrate that our new approach consistently outperforms commonly used hierarchical Bayesian methods.
    
[^12]: 神经架构搜索基准测试是否设计良好？对操作重要性的深入研究

    Are Neural Architecture Search Benchmarks Well Designed? A Deeper Look Into Operation Importance. (arXiv:2303.16938v1 [cs.LG])

    [http://arxiv.org/abs/2303.16938](http://arxiv.org/abs/2303.16938)

    本论文对当前广泛使用的NAS基准测试进行了经验研究，发现只需一小部分的操作即可生成接近最高性能的架构，同时这些基准测试存在缺点可能影响公平比较并提供不可靠结果。

    

    神经架构搜索（NAS）基准测试显著提高了开发和比较NAS方法的能力，同时通过提供关于数千个训练过的神经网络的元信息，大幅减少了计算开销。然而，表格基准测试具有几个缺点，可能会阻碍公平比较并提供不可靠的结果。在这项工作中，我们对广泛使用的NAS-Bench-101、NAS-Bench-201和TransNAS-Bench-101基准测试进行了经验性分析，重点关注它们的通用性以及不同操作如何影响所生成架构的性能。我们发现，仅需要操作池的一部分即可生成接近最高性能范围的架构。此外，性能分布具有负偏斜。

    Neural Architecture Search (NAS) benchmarks significantly improved the capability of developing and comparing NAS methods while at the same time drastically reduced the computational overhead by providing meta-information about thousands of trained neural networks. However, tabular benchmarks have several drawbacks that can hinder fair comparisons and provide unreliable results. These usually focus on providing a small pool of operations in heavily constrained search spaces -- usually cell-based neural networks with pre-defined outer-skeletons. In this work, we conducted an empirical analysis of the widely used NAS-Bench-101, NAS-Bench-201 and TransNAS-Bench-101 benchmarks in terms of their generability and how different operations influence the performance of the generated architectures. We found that only a subset of the operation pool is required to generate architectures close to the upper-bound of the performance range. Also, the performance distribution is negatively skewed, havi
    
[^13]: 训练数据重构的非渐进性下界

    Non-Asymptotic Lower Bounds For Training Data Reconstruction. (arXiv:2303.16372v1 [cs.LG])

    [http://arxiv.org/abs/2303.16372](http://arxiv.org/abs/2303.16372)

    本文通过研究差分隐私和度量隐私学习器在对抗者重构错误方面的鲁棒性，得出了非渐进性下界，覆盖了高维情况，且扩展了深度学习算法的隐私分析

    

    本文研究了专业对手进行训练数据重构攻击时私有学习算法的语义保证强度。我们通过导出非渐进量级下界来研究了满足差分隐私（DP）和度量隐私（mDP）的学习器对抗者重构错误的鲁棒性。此外，我们还证明了我们对mDP的分析覆盖了高维情况。本文进一步对流行的深度学习算法，如DP-SGD和Projected Noisy SGD进行了度量差分隐私的扩展隐私分析。

    We investigate semantic guarantees of private learning algorithms for their resilience to training Data Reconstruction Attacks (DRAs) by informed adversaries. To this end, we derive non-asymptotic minimax lower bounds on the adversary's reconstruction error against learners that satisfy differential privacy (DP) and metric differential privacy (mDP). Furthermore, we demonstrate that our lower bound analysis for the latter also covers the high dimensional regime, wherein, the input data dimensionality may be larger than the adversary's query budget. Motivated by the theoretical improvements conferred by metric DP, we extend the privacy analysis of popular deep learning algorithms such as DP-SGD and Projected Noisy SGD to cover the broader notion of metric differential privacy.
    
[^14]: PCA-Net：操作学习的复杂性上下界

    Operator learning with PCA-Net: upper and lower complexity bounds. (arXiv:2303.16317v1 [cs.LG])

    [http://arxiv.org/abs/2303.16317](http://arxiv.org/abs/2303.16317)

    本文发展了PCA-Net的近似理论，得出了通用逼近结果，并识别出了使用PCA-Net进行高效操作学习的潜在障碍：输出分布的复杂性和算子空间的内在复杂性。

    

    神经算子在计算科学和工程中备受关注。PCA-Net是一种最近提出的神经算子架构，它将主成分分析(PCA)与神经网络相结合，以逼近潜在的算子。本文对这种方法进行了近似理论的发展，改进并显着扩展了此方向的以前的工作。在定性界限方面，本文得出了新颖的通用逼近结果，在对潜在算子和数据生成分布的最小假设的前提下。在定量限制方面，本文识别了使用PCA-Net进行高效操作学习的两个潜在障碍，通过导出下界进行了严格证明，第一个障碍与输出分布的复杂性有关，由PCA特征值的缓慢衰减来衡量；另一个障碍涉及无限维输入和输出空间之间的算子空间的内在复杂性。

    Neural operators are gaining attention in computational science and engineering. PCA-Net is a recently proposed neural operator architecture which combines principal component analysis (PCA) with neural networks to approximate an underlying operator. The present work develops approximation theory for this approach, improving and significantly extending previous work in this direction. In terms of qualitative bounds, this paper derives a novel universal approximation result, under minimal assumptions on the underlying operator and the data-generating distribution. In terms of quantitative bounds, two potential obstacles to efficient operator learning with PCA-Net are identified, and made rigorous through the derivation of lower complexity bounds; the first relates to the complexity of the output distribution, measured by a slow decay of the PCA eigenvalues. The other obstacle relates the inherent complexity of the space of operators between infinite-dimensional input and output spaces, 
    
[^15]: 利用分布分解提高均匀学习算法的性能

    Lifting uniform learners via distributional decomposition. (arXiv:2303.16208v1 [stat.ML])

    [http://arxiv.org/abs/2303.16208](http://arxiv.org/abs/2303.16208)

    本文介绍了一种方法，可以将任何在均匀分布下有效的PAC学习算法转换成一个在任意未知分布下有效的算法，而且对于单调分布，只需要用$\mathcal{D}$中的样本。算法的核心是通过一个算法将$\mathcal{D}$逼近成由子立方体混合而成的混合均匀分布。

    

    我们展示了如何将任何在均匀分布下有效的PAC学习算法转换成一个在任意未知分布$\mathcal{D}$下有效的算法。我们的转换效率随$\mathcal{D}$的固有复杂性而变化，对于在$\{\pm 1\}^n$上的分布，其pmf由深度为$d$的决策树计算，则时间复杂度为$\mathrm{poly}(n, (md)^d)$，其中$m$是原始算法的样本复杂度。对于单调分布，我们的转换仅使用$\mathcal{D}$中的样本，而对于一般分布，我们使用子立方体条件样本。其中一个关键技术是一个算法，它在给出$\mathcal{D}$的访问权限的情况下，产生了一个最优决策树分解$\mathcal{D}$：一个逼近了$\mathcal{D}$的混合均匀分布的分离子立方体。通过这个分解，我们在每个子立方体上运行均匀分布学习器，并将结果合并起来。

    We show how any PAC learning algorithm that works under the uniform distribution can be transformed, in a blackbox fashion, into one that works under an arbitrary and unknown distribution $\mathcal{D}$. The efficiency of our transformation scales with the inherent complexity of $\mathcal{D}$, running in $\mathrm{poly}(n, (md)^d)$ time for distributions over $\{\pm 1\}^n$ whose pmfs are computed by depth-$d$ decision trees, where $m$ is the sample complexity of the original algorithm. For monotone distributions our transformation uses only samples from $\mathcal{D}$, and for general ones it uses subcube conditioning samples.  A key technical ingredient is an algorithm which, given the aforementioned access to $\mathcal{D}$, produces an optimal decision tree decomposition of $\mathcal{D}$: an approximation of $\mathcal{D}$ as a mixture of uniform distributions over disjoint subcubes. With this decomposition in hand, we run the uniform-distribution learner on each subcube and combine the 
    
[^16]: 基于一致性和适应性概念的不确定性量化度量的验证：一个入门指南。

    Validation of uncertainty quantification metrics: a primer based on the consistency and adaptivity concepts. (arXiv:2303.07170v2 [physics.chem-ph] UPDATED)

    [http://arxiv.org/abs/2303.07170](http://arxiv.org/abs/2303.07170)

    本文介绍了一种基于一致性和适应性概念的UQ度量验证方法，通过重新审视已有的方法，提高了对UQ度量能力的理解。

    

    不确定性量化（UQ）验证的实践，尤其是在物理化学科学的机器学习中，依赖于几种图形方法（散点图、校准曲线、可靠性图和置信曲线），这些图形方法探索校准的互补方面，但有些方面并没有得到很好的探索。例如，这些方法中没有一种涉及到UQ度量在输入特征范围内的可靠性（适应性）。基于一致性和适应性的互补概念，重新审视了基于方差和间隔的UQ度量的常见验证方法，旨在提供更好的理解它们的能力。本研究旨在介绍UQ验证，并从几个基本规则中导出所有方法。这些方法在合成数据集和从最近的物理化学机器学习UQ文献中提取的代表性示例上进行了说明和测试。

    The practice of uncertainty quantification (UQ) validation, notably in machine learning for the physico-chemical sciences, rests on several graphical methods (scattering plots, calibration curves, reliability diagrams and confidence curves) which explore complementary aspects of calibration, without covering all the desirable ones. For instance, none of these methods deals with the reliability of UQ metrics across the range of input features (adaptivity). Based on the complementary concepts of consistency and adaptivity, the toolbox of common validation methods for variance- and intervals- based UQ metrics is revisited with the aim to provide a better grasp on their capabilities. This study is conceived as an introduction to UQ validation, and all methods are derived from a few basic rules. The methods are illustrated and tested on synthetic datasets and representative examples extracted from the recent physico-chemical machine learning UQ literature.
    
[^17]: 物理学知识作为不确定性量化模型的信息场理论

    Physics-informed Information Field Theory for Modeling Physical Systems with Uncertainty Quantification. (arXiv:2301.07609v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.07609](http://arxiv.org/abs/2301.07609)

    该论文扩展了信息场理论(IFT)到物理信息场理论(PIFT)，将描述场的物理定律的信息编码为函数先验。从这个PIFT得出的后验与任何数值方案无关，并且可以捕捉多种模式。

    

    数据驱动的方法结合物理学知识是建模系统的强有力技术。此类模型的目标是通过将测量结果与已知物理定律相结合，高效地求解基本场。由于许多系统包含未知元素，如缺失参数、嘈杂数据或不完整的物理定律，因此这通常被视为一种不确定性量化问题。处理所有变量的常见技术通常取决于用于近似后验的数值方案，并且希望有一种不依赖于任何离散化的方法。信息场理论（IFT）提供了对不一定是高斯场的场进行统计学的工具。我们通过将描述场的物理定律的信息编码为函数先验来扩展IFT到物理信息场理论（PIFT）。从这个PIFT得出的后验与任何数值方案无关，并且可以捕捉多种模式。

    Data-driven approaches coupled with physical knowledge are powerful techniques to model systems. The goal of such models is to efficiently solve for the underlying field by combining measurements with known physical laws. As many systems contain unknown elements, such as missing parameters, noisy data, or incomplete physical laws, this is widely approached as an uncertainty quantification problem. The common techniques to handle all the variables typically depend on the numerical scheme used to approximate the posterior, and it is desirable to have a method which is independent of any such discretization. Information field theory (IFT) provides the tools necessary to perform statistics over fields that are not necessarily Gaussian. We extend IFT to physics-informed IFT (PIFT) by encoding the functional priors with information about the physical laws which describe the field. The posteriors derived from this PIFT remain independent of any numerical scheme and can capture multiple modes,
    
[^18]: 切片最优偏转运输

    Sliced Optimal Partial Transport. (arXiv:2212.08049v5 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.08049](http://arxiv.org/abs/2212.08049)

    本文提出了一种适用于一维非负测度之间最优偏转运输问题的高效算法，并通过切片的方式定义了切片最优偏转运输距离。

    

    最优传输（OT）已经在机器学习、数据科学和计算机视觉中变得极其流行。OT问题的核心假设是源和目标测度的总质量相等，这限制了它的应用。最优偏转运输（OPT）是最近提出的解决这个限制的方法。与OT问题类似，OPT的计算依赖于解决线性规划问题（通常在高维度中），这可能会变得计算上困难。在本文中，我们提出了一种计算一维非负测度之间OPT问题的有效算法。接下来，遵循切片OT距离的思想，我们利用切片定义了切片OPT距离。最后，我们展示了切片OPT-based方法在各种数值实验中的计算和精度优势。特别是，我们展示了我们提出的Sliced-OPT在噪声点云配准中的应用。

    Optimal transport (OT) has become exceedingly popular in machine learning, data science, and computer vision. The core assumption in the OT problem is the equal total amount of mass in source and target measures, which limits its application. Optimal Partial Transport (OPT) is a recently proposed solution to this limitation. Similar to the OT problem, the computation of OPT relies on solving a linear programming problem (often in high dimensions), which can become computationally prohibitive. In this paper, we propose an efficient algorithm for calculating the OPT problem between two non-negative measures in one dimension. Next, following the idea of sliced OT distances, we utilize slicing to define the sliced OPT distance. Finally, we demonstrate the computational and accuracy benefits of the sliced OPT-based method in various numerical experiments. In particular, we show an application of our proposed Sliced-OPT in noisy point cloud registration.
    
[^19]: 紧凑集成用于高效的不确定性估计

    Packed-Ensembles for Efficient Uncertainty Estimation. (arXiv:2210.09184v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.09184](http://arxiv.org/abs/2210.09184)

    Packed-Ensembles是一种能够在标准神经网络内运行的轻量级结构化集合，它通过精心调节编码空间的维度来设计。该方法在不损失效果的情况下提高了训练和推理速度。

    

    深度集成是实现关键指标（如准确性、校准、不确定性估计和超出分布检测）卓越性能的突出方法。但是，现实系统的硬件限制限制了更小的集合和较低容量的网络，严重损害了它们的性能和属性。我们引入了一种称为Packed-Ensembles（PE）的策略，通过精心调节其编码空间的维度来设计和训练轻量级结构化集合。我们利用组卷积将集合并行化为单个共享骨干，并进行前向传递以提高训练和推理速度。PE旨在在标准神经网络的内存限制内运行。

    Deep Ensembles (DE) are a prominent approach for achieving excellent performance on key metrics such as accuracy, calibration, uncertainty estimation, and out-of-distribution detection. However, hardware limitations of real-world systems constrain to smaller ensembles and lower-capacity networks, significantly deteriorating their performance and properties. We introduce Packed-Ensembles (PE), a strategy to design and train lightweight structured ensembles by carefully modulating the dimension of their encoding space. We leverage grouped convolutions to parallelize the ensemble into a single shared backbone and forward pass to improve training and inference speeds. PE is designed to operate within the memory limits of a standard neural network. Our extensive research indicates that PE accurately preserves the properties of DE, such as diversity, and performs equally well in terms of accuracy, calibration, out-of-distribution detection, and robustness to distribution shift. We make our c
    
[^20]: 聚类图匹配用于标签恢复和图分类

    Clustered Graph Matching for Label Recovery and Graph Classification. (arXiv:2205.03486v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.03486](http://arxiv.org/abs/2205.03486)

    本论文提出一种利用顶点对齐的平均图，聚类平均图和混淆网络匹配的策略，比起传统的全局平均图策略，可以更有效地提高匹配性能和分类精度。

    

    针对一组顶点对齐网络和一个额外的标签混淆网络，我们提出了一种利用顶点对齐集合中的信号来恢复混淆网络标签的方法。我们考虑将混淆网络与不同粒度下的顶点对齐集合中的平均网络进行匹配。我们证明并证实，在网络来自不同网络类的情况下，将网络聚类到类中，然后将新网络匹配到聚类平均值，可以比将其匹配到全局平均图产生更高的匹配性能。此外，通过最小化相对于每个聚类平均值的图匹配目标函数，这种方法同时对混淆图进行了分类和顶点标签恢复。这些理论研究通过一个有启示意义的真实数据实验匹配人类连接体来得到更多巩固。

    Given a collection of vertex-aligned networks and an additional label-shuffled network, we propose procedures for leveraging the signal in the vertex-aligned collection to recover the labels of the shuffled network. We consider matching the shuffled network to averages of the networks in the vertex-aligned collection at different levels of granularity. We demonstrate both in theory and practice that if the graphs come from different network classes, then clustering the networks into classes followed by matching the new graph to cluster-averages can yield higher fidelity matching performance than matching to the global average graph. Moreover, by minimizing the graph matching objective function with respect to each cluster average, this approach simultaneously classifies and recovers the vertex labels for the shuffled graph. These theoretical developments are further reinforced via an illuminating real data experiment matching human connectomes.
    
[^21]: 随机流形采样和联合稀疏正则化的多标签特征选择

    Random Manifold Sampling and Joint Sparse Regularization for Multi-label Feature Selection. (arXiv:2204.06445v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2204.06445](http://arxiv.org/abs/2204.06445)

    本文提出了一种基于联合约束优化问题的 $\ell_{2,1}$ 和 $\ell_{F}$ 正则化方法来获得最相关的几个特征，并在流形正则化中实现了基于随机游走策略的高度稳健的邻域图。该方法在真实数据集上的比较实验中表现优异。

    

    多标签学习通常用于挖掘特征和标签之间的相关性，特征选择可以通过少量特征保留尽可能多的信息。 $\ell_{2,1}$ 正则化可以获得稀疏系数矩阵，但不能有效地解决多重共线性问题。本文提出的模型通过解决 $\ell_{2,1}$ 和 $\ell_{F}$ 正则化的联合约束优化问题来获取最相关的几个特征。在流形正则化中，我们根据联合信息矩阵实现了基于随机游走策略的高度稳健的邻域图。此外，我们还给出了解决该模型的算法并证明了其收敛性。在真实数据集上的比较实验表明，所提出的方法优于其他方法。

    Multi-label learning is usually used to mine the correlation between features and labels, and feature selection can retain as much information as possible through a small number of features. $\ell_{2,1}$ regularization method can get sparse coefficient matrix, but it can not solve multicollinearity problem effectively. The model proposed in this paper can obtain the most relevant few features by solving the joint constrained optimization problems of $\ell_{2,1}$ and $\ell_{F}$ regularization.In manifold regularization, we implement random walk strategy based on joint information matrix, and get a highly robust neighborhood graph.In addition, we given the algorithm for solving the model and proved its convergence.Comparative experiments on real-world data sets show that the proposed method outperforms other methods.
    
[^22]: 具范数约束的神经网络的逼近误差界与应用研究

    Approximation bounds for norm constrained neural networks with applications to regression and GANs. (arXiv:2201.09418v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.09418](http://arxiv.org/abs/2201.09418)

    本文研究了具范数约束的ReLU神经网络的逼近能力，并证明了对于平滑函数类，这些网络的逼近误差有上下界。此外，应用结果分析了回归和GAN分布估计问题的收敛性，最终证明了当GAN的判别器选择合适的具范数约束的神经网络时，可以实现学习概率分布的最优速率。

    

    本文研究带权重范数约束的ReLU神经网络的逼近能力，对于平滑的函数类，我们证明了这些网络的逼近误差上下界。通过神经网络的Rademacher复杂度导出下界证明，这可能具有独立的研究价值。我们应用这些逼近误差界限来分析使用具范数约束的神经网络进行回归和GAN分布估计的收敛性。特别的，我们得到了过参数神经网络的收敛速率。同时，我们还证明了当判别器选择合适的具范数约束的神经网络时，GAN可以实现学习概率分布的最优速率。

    This paper studies the approximation capacity of ReLU neural networks with norm constraint on the weights. We prove upper and lower bounds on the approximation error of these networks for smooth function classes. The lower bound is derived through the Rademacher complexity of neural networks, which may be of independent interest. We apply these approximation bounds to analyze the convergences of regression using norm constrained neural networks and distribution estimation by GANs. In particular, we obtain convergence rates for over-parameterized neural networks. It is also shown that GANs can achieve optimal rate of learning probability distributions, when the discriminator is a properly chosen norm constrained neural network.
    
[^23]: 统计上意义的近似：一种在变换器中近似图灵机的案例研究

    Statistically Meaningful Approximation: a Case Study on Approximating Turing Machines with Transformers. (arXiv:2107.13163v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2107.13163](http://arxiv.org/abs/2107.13163)

    本文提出了统计上意义的近似的正式定义，研究了过度参数化的前馈神经网络和变换器的SM近似在布尔电路和图灵机中的应用，重点在于探索近似网络应该具有良好的统计可学性的概念，达到更有意义的近似效果。

    

    理论上研究神经网络结构的常用方法是分析它们可以近似的函数。然而，近似理论中的构造可能是不现实的，因此意义不太明确。为了解决这些问题，本文提出了统计上意义的（SM）近似的正式定义，要求近似网络具有良好的统计可学性。我们研究了两种函数类别的SM近似：布尔电路和图灵机。我们表明，过度参数化的前馈神经网络可以SM近似布尔电路，采样复杂度仅取决于电路大小，而不是网络大小。此外，我们还表明，变换器可以SM近似计算时间受$T$限制的图灵机，采样复杂度多项式地取决于字母大小、状态空间大小和$\log (T)$。我们还在...

    A common lens to theoretically study neural net architectures is to analyze the functions they can approximate. However, constructions from approximation theory may be unrealistic and therefore less meaningful. For example, a common unrealistic trick is to encode target function values using infinite precision. To address these issues, this work proposes a formal definition of statistically meaningful (SM) approximation which requires the approximating network to exhibit good statistical learnability. We study SM approximation for two function classes: boolean circuits and Turing machines. We show that overparameterized feedforward neural nets can SM approximate boolean circuits with sample complexity depending only polynomially on the circuit size, not the size of the network. In addition, we show that transformers can SM approximate Turing machines with computation time bounded by $T$ with sample complexity polynomial in the alphabet size, state space size, and $\log (T)$. We also in
    
[^24]: 针对具有凸惩罚的鲁棒M-估计的样外误差估计

    Out-of-sample error estimate for robust M-estimators with convex penalty. (arXiv:2008.11840v5 [math.ST] UPDATED)

    [http://arxiv.org/abs/2008.11840](http://arxiv.org/abs/2008.11840)

    该论文提出了一种通用的样外误差估计方法，适用于正则化具有凸惩罚的鲁棒$M$-估计，该方法仅通过固定的观测数据依赖于特定量，其中在高维渐近区域中，该估计具有相对误差，具有广泛的适用性。

    

    在观测到$(X,y)$且$p,n$同阶的高维线性回归中，提出了一种用于正则化具有凸惩罚的鲁棒$M$-估计的通用样外误差估计。如果$\psi$是鲁棒数据拟合损失函数$\rho$的导数，则该估计仅通过$\hat\psi = \psi(y-X\hat\beta)$、$X^\top \hat\psi$以及$X$固定时的导数$(\partial/\partial y)\hat\psi$和$(\partial/\partial y)X\hat\beta$依赖于观测数据。在具有高斯协变量和独立噪声的线性模型中，这种样外误差估计在$n^{-1/2}$阶具有相对误差，无论是在$p/n\le \gamma$的非渐近情况下还是在高维渐近区域$p/n\to\gamma'\in(0,\infty)$中均成立。只要$\psi=\rho'$是1-Lipschitz的，即使通用可微损失函数$\rho$也是被允许的。当惩罚参数进行适当缩放时，样外误差估计的有效性在满足强凸性假设下或$\ell_1$惩罚的Huber损失中均成立。

    A generic out-of-sample error estimate is proposed for robust $M$-estimators regularized with a convex penalty in high-dimensional linear regression where $(X,y)$ is observed and $p,n$ are of the same order. If $\psi$ is the derivative of the robust data-fitting loss $\rho$, the estimate depends on the observed data only through the quantities $\hat\psi = \psi(y-X\hat\beta)$, $X^\top \hat\psi$ and the derivatives $(\partial/\partial y) \hat\psi$ and $(\partial/\partial y) X\hat\beta$ for fixed $X$.  The out-of-sample error estimate enjoys a relative error of order $n^{-1/2}$ in a linear model with Gaussian covariates and independent noise, either non-asymptotically when $p/n\le \gamma$ or asymptotically in the high-dimensional asymptotic regime $p/n\to\gamma'\in(0,\infty)$. General differentiable loss functions $\rho$ are allowed provided that $\psi=\rho'$ is 1-Lipschitz. The validity of the out-of-sample error estimate holds either under a strong convexity assumption, or for the $\ell
    
[^25]: 变分Wasserstein质心用于几何聚类

    Variational Wasserstein Barycenters for Geometric Clustering. (arXiv:2002.10543v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2002.10543](http://arxiv.org/abs/2002.10543)

    该论文提出了利用变分Wasserstein质心解决几何聚类问题的方法，特别是Monge WBs与K-means聚类和共同聚类相关，同时还提出了两个新问题——正则化K-means和Wasserstein质心压缩，并演示了VWBs在解决这些聚类相关问题的有效性。

    

    我们提出通过解决具有变分原理的Monge映射来计算Wasserstein质心(WBs)。我们讨论了WBs的度量特性，并探索它们的联系，特别是Monge WBs与K-means聚类和共同聚类的联系。我们还讨论了Monge WBs在非平衡度量和球形域上的可行性。我们提出了两个新问题——正则化K-means和Wasserstein质心压缩。我们演示了使用VWBs解决这些聚类相关问题的方法。

    We propose to compute Wasserstein barycenters (WBs) by solving for Monge maps with variational principle. We discuss the metric properties of WBs and explore their connections, especially the connections of Monge WBs, to K-means clustering and co-clustering. We also discuss the feasibility of Monge WBs on unbalanced measures and spherical domains. We propose two new problems -regularized K-means and Wasserstein barycenter compression. We demonstrate the use of VWBs in solving these clustering-related problems.
    
[^26]: 隔开式试验的最优设计

    Optimal Experimental Design for Staggered Rollouts. (arXiv:1911.03764v5 [econ.EM] UPDATED)

    [http://arxiv.org/abs/1911.03764](http://arxiv.org/abs/1911.03764)

    本文研究了隔开式试验的最优设计问题。对于非自适应实验，提出了一个近似最优解；对于自适应实验，提出了一种新算法——精度导向的自适应实验（PGAE）算法，它使用贝叶斯决策理论来最大化估计治疗效果的预期精度。

    

    本文研究了在不同时期内某组数据单元的治疗开始时间存在差异时，对实验进行设计和分析的问题。设计问题涉及选择每个数据单元的初始治疗时间以便最精确地估计治疗的瞬时效应和累积效应。我们首先考虑非自适应实验，其中所有的治疗分配决策都在实验开始之前做出。针对这种情况，我们证明了优化问题通常是NP难的，并提出了一种近似最优解。在该解决方案下，每个时期进入治疗的分数最初较低，然后变高，最后再次降低。接下来，我们研究了自适应实验设计问题，其中在收集每个时期的数据后更新继续实验和治疗分配决策。对于自适应情况，我们提出了一种新算法——精度导向的自适应实验（PGAE）算法，它使用贝叶斯决策理论来最大化估计治疗效果的预期精度。我们证明了PGAE算法达到了悔恨的下限，悔恨定义为期望累计平方标准误差和任意治疗分配策略所能实现的最佳误差之间的差异。

    In this paper, we study the design and analysis of experiments conducted on a set of units over multiple time periods where the starting time of the treatment may vary by unit. The design problem involves selecting an initial treatment time for each unit in order to most precisely estimate both the instantaneous and cumulative effects of the treatment. We first consider non-adaptive experiments, where all treatment assignment decisions are made prior to the start of the experiment. For this case, we show that the optimization problem is generally NP-hard, and we propose a near-optimal solution. Under this solution, the fraction entering treatment each period is initially low, then high, and finally low again. Next, we study an adaptive experimental design problem, where both the decision to continue the experiment and treatment assignment decisions are updated after each period's data is collected. For the adaptive case, we propose a new algorithm, the Precision-Guided Adaptive Experim
    

