# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Implicit Bias and Fast Convergence Rates for Self-attention](https://arxiv.org/abs/2402.05738) | 该论文研究了在自注意力网络中使用梯度下降训练的隐性偏差以及其收敛速率，通过证明在特定数据设置下收敛性是全局的，并提供了W_t到W_mm的有限时间收敛率。 |
| [^2] | [Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity](https://arxiv.org/abs/2402.03167) | 本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。 |
| [^3] | [Sample, estimate, aggregate: A recipe for causal discovery foundation models](https://arxiv.org/abs/2402.01929) | 本文提出一种因果发现框架，通过深度学习模型预训练与经典发现算法的结合，实现了快速、准确地推断因果结构，并在实验中展示了与现有方法相比更好的表现和推理速度。 |
| [^4] | [A deep implicit-explicit minimizing movement method for option pricing in jump-diffusion models.](http://arxiv.org/abs/2401.06740) | 这份论文介绍了一种用于定价跳跃扩散模型下欧式篮式期权的深度学习方法，采用了隐式-显式最小移动方法以及残差型人工神经网络逼近，并通过稀疏网格高斯-埃尔米特逼近和基于ANN的高维专用求积规则来离散化积分运算符。 |
| [^5] | [Conditional Density Estimations from Privacy-Protected Data.](http://arxiv.org/abs/2310.12781) | 本文提出了一种从隐私保护数据中进行条件密度估计的方法，使用神经条件密度估计器来近似模型参数的后验分布，从而解决了在统计分析过程中只能访问私有化数据导致的计算复杂度增加的问题。 |
| [^6] | [Optimal vintage factor analysis with deflation varimax.](http://arxiv.org/abs/2310.10545) | 本文提出了一种采用通货紧缩变量旋转的拟合因子分析方法，在每一行上逐步求解正交矩阵，相比于传统方法具有更好的计算性能和灵活性，并且在更广泛的背景下提供了理论保证。 |
| [^7] | [Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences.](http://arxiv.org/abs/2309.03791) | 本论文介绍了一种新的方法ARMOR_D来加强深度学习模型的对抗鲁棒性，该方法基于最优传输正则化差异，通过在分布的邻域上进行最大化期望损失来实现。实验证明，ARMOR_D方法在恶意软件检测和图像识别应用中能够优于现有方法，在对抗攻击下的鲁棒性方面具有较好的效果。 |
| [^8] | [Wasserstein multivariate auto-regressive models for modeling distributional time series and its application in graph learning.](http://arxiv.org/abs/2207.05442) | 本文提出了一种新的自回归模型，用于分析多元分布时间序列。并且在Wasserstein空间中建模了随机对象，提供了该模型的解的存在性和一致估计器。此方法可以应用于年龄分布和自行车共享网络的观察数据。 |

# 详细

[^1]: 隐性偏差与自注意力的快速收敛速率

    Implicit Bias and Fast Convergence Rates for Self-attention

    [https://arxiv.org/abs/2402.05738](https://arxiv.org/abs/2402.05738)

    该论文研究了在自注意力网络中使用梯度下降训练的隐性偏差以及其收敛速率，通过证明在特定数据设置下收敛性是全局的，并提供了W_t到W_mm的有限时间收敛率。

    

    自注意力是transformer的核心机制，它使其与传统神经网络有所区别，并驱动其出色的性能。为了开发自注意力的基本优化原则，我们研究了用梯度下降（GD）训练具有固定线性解码器的自注意力层在二元分类中的隐性偏差。受到在可分离数据上线性逻辑回归中GD的研究启发，最近的工作表明，随着迭代次数t无限接近于无穷大，键-查询矩阵W_t在局部上（相对于初始化方向）收敛到一个硬边界支持向量机解W_mm。我们的工作在四个方面增强了这个结果。首先，我们确定了非平凡的数据设置，对于这些设置，收敛性是全局的，并揭示了优化空间的特性。其次，我们首次提供了W_t到W_mm的有限时间收敛率，并量化了稀疏化的速率。

    Self-attention, the core mechanism of transformers, distinguishes them from traditional neural networks and drives their outstanding performance. Towards developing the fundamental optimization principles of self-attention, we investigate the implicit bias of gradient descent (GD) in training a self-attention layer with fixed linear decoder in binary classification. Drawing inspiration from the study of GD in linear logistic regression over separable data, recent work demonstrates that as the number of iterations $t$ approaches infinity, the key-query matrix $W_t$ converges locally (with respect to the initialization direction) to a hard-margin SVM solution $W_{mm}$. Our work enhances this result in four aspects. Firstly, we identify non-trivial data settings for which convergence is provably global, thus shedding light on the optimization landscape. Secondly, we provide the first finite-time convergence rate for $W_t$ to $W_{mm}$, along with quantifying the rate of sparsification in t
    
[^2]: 图上的去中心化双级优化: 无环算法更新和瞬态迭代复杂性

    Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity

    [https://arxiv.org/abs/2402.03167](https://arxiv.org/abs/2402.03167)

    本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。

    

    随机双级优化（SBO）在处理嵌套结构方面的多样性使其在机器学习中变得越来越重要。为了解决大规模SBO，去中心化方法作为有效的范例出现，其中节点与直接相邻节点进行通信，无需中央服务器，从而提高通信效率和增强算法的稳健性。然而，当前的去中心化SBO算法面临挑战，包括昂贵的内部循环更新和对网络拓扑、数据异构性和嵌套双级算法结构的影响不明确。在本文中，我们引入了一种单循环的去中心化SBO（D-SOBA）算法，并建立了其瞬态迭代复杂性，首次澄清了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA实现了最先进的渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性。

    Stochastic bilevel optimization (SBO) is becoming increasingly essential in machine learning due to its versatility in handling nested structures. To address large-scale SBO, decentralized approaches have emerged as effective paradigms in which nodes communicate with immediate neighbors without a central server, thereby improving communication efficiency and enhancing algorithmic robustness. However, current decentralized SBO algorithms face challenges, including expensive inner-loop updates and unclear understanding of the influence of network topology, data heterogeneity, and the nested bilevel algorithmic structures. In this paper, we introduce a single-loop decentralized SBO (D-SOBA) algorithm and establish its transient iteration complexity, which, for the first time, clarifies the joint influence of network topology and data heterogeneity on decentralized bilevel algorithms. D-SOBA achieves the state-of-the-art asymptotic rate, asymptotic gradient/Hessian complexity, and transien
    
[^3]: 样本、估计、聚合：因果发现基础模型的一种方法

    Sample, estimate, aggregate: A recipe for causal discovery foundation models

    [https://arxiv.org/abs/2402.01929](https://arxiv.org/abs/2402.01929)

    本文提出一种因果发现框架，通过深度学习模型预训练与经典发现算法的结合，实现了快速、准确地推断因果结构，并在实验中展示了与现有方法相比更好的表现和推理速度。

    

    因果发现是从数据中推断因果结构的任务，它可以加速科学研究、指导决策等。然而，现有因果发现算法的每个数据集的特性使它们变得缓慢、需要大量数据并且脆弱。受基础模型的启发，我们提出了一种因果发现框架，其中深度学习模型预训练用于处理在较小的变量子集上运行的经典发现算法的预测。这种方法可以利用以下观察结果：经典算法的输出在小问题上计算速度快，对（边际）数据结构具有信息量，且它们的输出结构作为对象在数据集之间可以进行比较。我们的方法在合成和实际数据集上实现了最先进的性能，可以推广到训练期间未见过的数据生成机制，并且提供比现有模型快几个数量级的推理速度。

    Causal discovery, the task of inferring causal structure from data, promises to accelerate scientific research, inform policy making, and more. However, the per-dataset nature of existing causal discovery algorithms renders them slow, data hungry, and brittle. Inspired by foundation models, we propose a causal discovery framework where a deep learning model is pretrained to resolve predictions from classical discovery algorithms run over smaller subsets of variables. This method is enabled by the observations that the outputs from classical algorithms are fast to compute for small problems, informative of (marginal) data structure, and their structure outputs as objects remain comparable across datasets. Our method achieves state-of-the-art performance on synthetic and realistic datasets, generalizes to data generating mechanisms not seen during training, and offers inference speeds that are orders of magnitude faster than existing models.
    
[^4]: 一种用于跳跃扩散模型期权定价的深度隐式-显式最小移动方法

    A deep implicit-explicit minimizing movement method for option pricing in jump-diffusion models. (arXiv:2401.06740v1 [q-fin.CP])

    [http://arxiv.org/abs/2401.06740](http://arxiv.org/abs/2401.06740)

    这份论文介绍了一种用于定价跳跃扩散模型下欧式篮式期权的深度学习方法，采用了隐式-显式最小移动方法以及残差型人工神经网络逼近，并通过稀疏网格高斯-埃尔米特逼近和基于ANN的高维专用求积规则来离散化积分运算符。

    

    我们提出了一种新颖的深度学习方法，用于定价跳跃扩散动态下的欧式篮式期权。将期权定价问题表述为一个偏积分微分方程，并通过一种新的隐式-显式最小移动时间步法进行近似，该方法使用深度残差型人工神经网络（ANNs）逐步逼近。积分运算符通过两种不同的方法离散化：a）通过稀疏网格高斯-埃尔米特逼近，采用奇异值分解产生的局部坐标轴，并且b）通过基于ANN的高维专用求积规则。关键是，所提出的ANN的构造确保了解决方案在标的资产较大值时的渐近行为，并且与解决方案先验已知的定性特性相一致输出。对方法维度的性能和鲁棒性进行了评估。

    We develop a novel deep learning approach for pricing European basket options written on assets that follow jump-diffusion dynamics. The option pricing problem is formulated as a partial integro-differential equation, which is approximated via a new implicit-explicit minimizing movement time-stepping approach, involving approximation by deep, residual-type Artificial Neural Networks (ANNs) for each time step. The integral operator is discretized via two different approaches: a) a sparse-grid Gauss--Hermite approximation following localised coordinate axes arising from singular value decompositions, and b) an ANN-based high-dimensional special-purpose quadrature rule. Crucially, the proposed ANN is constructed to ensure the asymptotic behavior of the solution for large values of the underlyings and also leads to consistent outputs with respect to a priori known qualitative properties of the solution. The performance and robustness with respect to the dimension of the methods are assesse
    
[^5]: 从隐私保护数据中进行条件密度估计

    Conditional Density Estimations from Privacy-Protected Data. (arXiv:2310.12781v1 [stat.ML])

    [http://arxiv.org/abs/2310.12781](http://arxiv.org/abs/2310.12781)

    本文提出了一种从隐私保护数据中进行条件密度估计的方法，使用神经条件密度估计器来近似模型参数的后验分布，从而解决了在统计分析过程中只能访问私有化数据导致的计算复杂度增加的问题。

    

    许多现代统计分析和机器学习应用需要在敏感用户数据上进行模型训练。差分隐私提供了一种正式的保证，即个体用户信息不会泄露。在这个框架下，随机算法向保密数据注入校准的噪声，从而产生隐私保护的数据集或查询。然而，在统计分析过程中只能访问私有化数据会导致计算复杂度增加，难以对基础机密数据的参数进行有效的推理。在本工作中，我们提出了基于隐私保护数据集的基于模拟的推理方法。具体而言，我们使用神经条件密度估计器作为一组灵活的分布来近似给定观测到的私有查询结果的模型参数的后验分布。我们在传染病模型下的离散时间序列数据以及普通线性回归模型上说明了我们的方法。

    Many modern statistical analysis and machine learning applications require training models on sensitive user data. Differential privacy provides a formal guarantee that individual-level information about users does not leak. In this framework, randomized algorithms inject calibrated noise into the confidential data, resulting in privacy-protected datasets or queries. However, restricting access to only the privatized data during statistical analysis makes it computationally challenging to perform valid inferences on parameters underlying the confidential data. In this work, we propose simulation-based inference methods from privacy-protected datasets. Specifically, we use neural conditional density estimators as a flexible family of distributions to approximate the posterior distribution of model parameters given the observed private query results. We illustrate our methods on discrete time-series data under an infectious disease model and on ordinary linear regression models. Illustra
    
[^6]: 优化拟合因子分析与通货紧缩变量旋转

    Optimal vintage factor analysis with deflation varimax. (arXiv:2310.10545v1 [stat.ML])

    [http://arxiv.org/abs/2310.10545](http://arxiv.org/abs/2310.10545)

    本文提出了一种采用通货紧缩变量旋转的拟合因子分析方法，在每一行上逐步求解正交矩阵，相比于传统方法具有更好的计算性能和灵活性，并且在更广泛的背景下提供了理论保证。

    

    通货紧缩变量旋转是一种重要的因子分析方法，旨在首先找到原始数据的低维表示，然后寻求旋转，使旋转后的低维表示具有科学意义。尽管Principal Component Analysis (PCA) followed by the varimax rotation被广泛应用于拟合因子分析，但由于varimax rotation需要在正交矩阵集合上解非凸优化问题，因此很难提供理论保证。本文提出了一种逐行求解正交矩阵的通货紧缩变量旋转过程。除了在计算上的优势和灵活性之外，我们还能在广泛的背景下对所提出的过程进行完全的理论保证。在PCA之后采用这种新的varimax方法作为第二步，我们进一步分析了这个两步过程在一个更一般的因子模型的情况下。

    Vintage factor analysis is one important type of factor analysis that aims to first find a low-dimensional representation of the original data, and then to seek a rotation such that the rotated low-dimensional representation is scientifically meaningful. Perhaps the most widely used vintage factor analysis is the Principal Component Analysis (PCA) followed by the varimax rotation. Despite its popularity, little theoretical guarantee can be provided mainly because varimax rotation requires to solve a non-convex optimization over the set of orthogonal matrices.  In this paper, we propose a deflation varimax procedure that solves each row of an orthogonal matrix sequentially. In addition to its net computational gain and flexibility, we are able to fully establish theoretical guarantees for the proposed procedure in a broad context.  Adopting this new varimax approach as the second step after PCA, we further analyze this two step procedure under a general class of factor models. Our resul
    
[^7]: 使用最优传输正则化差异来提高对抗性鲁棒深度学习

    Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences. (arXiv:2309.03791v1 [cs.LG])

    [http://arxiv.org/abs/2309.03791](http://arxiv.org/abs/2309.03791)

    本论文介绍了一种新的方法ARMOR_D来加强深度学习模型的对抗鲁棒性，该方法基于最优传输正则化差异，通过在分布的邻域上进行最大化期望损失来实现。实验证明，ARMOR_D方法在恶意软件检测和图像识别应用中能够优于现有方法，在对抗攻击下的鲁棒性方面具有较好的效果。

    

    我们引入了ARMOR_D方法作为增强深度学习模型对抗性鲁棒性的创新方法。这些方法基于一种新的最优传输正则化差异类，通过信息差异和最优传输成本之间的infimal卷积构建。我们使用这些方法来增强对抗性鲁棒性，通过在分布的邻域上最大化期望损失，这被称为分布鲁棒优化技术。作为构建对抗样本的工具，我们的方法允许样本根据最优传输成本进行传输，并根据信息差异进行重新加权。我们在恶意软件检测和图像识别应用上证明了我们方法的有效性，并发现在增强对抗攻击鲁棒性方面，据我们所知，它优于现有方法。ARMOR_D在FGSM攻击下的robustified准确率达到98.29%，在其他攻击下达到98.18%。

    We introduce the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These methods are based on a new class of optimal-transport-regularized divergences, constructed via an infimal convolution between an information divergence and an optimal-transport (OT) cost. We use these as tools to enhance adversarial robustness by maximizing the expected loss over a neighborhood of distributions, a technique known as distributionally robust optimization. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence. We demonstrate the effectiveness of our method on malware detection and image recognition applications and find that, to our knowledge, it outperforms existing methods at enhancing the robustness against adversarial attacks. $ARMOR_D$ yields the robustified accuracy of $98.29\%$ against $FGSM$ and $98.18\%$ aga
    
[^8]: Wasserstein多元自回归模型用于建模分布时间序列及其在图形学习中的应用

    Wasserstein multivariate auto-regressive models for modeling distributional time series and its application in graph learning. (arXiv:2207.05442v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2207.05442](http://arxiv.org/abs/2207.05442)

    本文提出了一种新的自回归模型，用于分析多元分布时间序列。并且在Wasserstein空间中建模了随机对象，提供了该模型的解的存在性和一致估计器。此方法可以应用于年龄分布和自行车共享网络的观察数据。

    

    我们提出了一种新的自回归模型，用于统计分析多元分布时间序列。感兴趣的数据包括一组在实线有界间隔上支持的概率测度的多个系列，并且被不同时间瞬间所索引。概率测度被建模为Wasserstein空间中的随机对象。我们通过在Lebesgue测度的切空间中建立自回归模型，首先对所有原始测度进行居中处理，以便它们的Fréchet平均值成为Lebesgue测度。利用迭代随机函数系统的理论，提供了这样一个模型的解的存在性、唯一性和平稳性的结果。我们还提出了模型系数的一致估计器。除了对模拟数据的分析，我们还使用两个实际数据集进行了模型演示：一个是不同国家年龄分布的观察数据集，另一个是巴黎自行车共享网络的观察数据集。

    We propose a new auto-regressive model for the statistical analysis of multivariate distributional time series. The data of interest consist of a collection of multiple series of probability measures supported over a bounded interval of the real line, and that are indexed by distinct time instants. The probability measures are modelled as random objects in the Wasserstein space. We establish the auto-regressive model in the tangent space at the Lebesgue measure by first centering all the raw measures so that their Fr\'echet means turn to be the Lebesgue measure. Using the theory of iterated random function systems, results on the existence, uniqueness and stationarity of the solution of such a model are provided. We also propose a consistent estimator for the model coefficient. In addition to the analysis of simulated data, the proposed model is illustrated with two real data sets made of observations from age distribution in different countries and bike sharing network in Paris. Final
    

