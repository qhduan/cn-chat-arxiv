# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Shallow ReLU neural networks and finite elements](https://arxiv.org/abs/2403.05809) | 在凸多面体网格上，提出了用两个隐藏层的ReLU神经网络来弱表示分段线性函数，并根据网格中的多面体和超平面的数量准确确定了所需的神经元数，建立了浅层ReLU神经网络和有限元函数之间的联系。 |
| [^2] | [Geometry-induced Implicit Regularization in Deep ReLU Neural Networks](https://arxiv.org/abs/2402.08269) | 通过研究参数变化时输出集合的几何特征，我们发现在深度ReLU神经网络的优化过程中存在几何引导的隐式正则化现象。 |
| [^3] | [On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit](https://arxiv.org/abs/2402.06388) | 该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。 |
| [^4] | [Beyond Expectations: Learning with Stochastic Dominance Made Practical](https://arxiv.org/abs/2402.02698) | 这项工作首次尝试建立了一个随机优势学习的通用框架，并推广了随机优势的概念以使其能够在任意两个随机变量之间进行比较。同时，我们还开发了一种有效的计算方法来处理连续性评估的问题。 |
| [^5] | [Sample Path Regularity of Gaussian Processes from the Covariance Kernel](https://arxiv.org/abs/2312.14886) | 本文提供了关于高斯过程样本路径正则性的新颖和紧凑的特征描述，通过协方差核对应的GP样本路径达到一定正则性的充分必要条件，对常用于机器学习应用中的GPs的样本路径正则性进行了探讨。 |
| [^6] | [MFAI: A Scalable Bayesian Matrix Factorization Approach to Leveraging Auxiliary Information](https://arxiv.org/abs/2303.02566) | MFAI是一种可扩展的贝叶斯矩阵分解方法，通过利用辅助信息来克服由于数据质量差导致的挑战，具有灵活建模非线性关系和对辅助信息的鲁棒性。 |
| [^7] | [A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model.](http://arxiv.org/abs/2310.11143) | 本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。 |
| [^8] | [ETDock: A Novel Equivariant Transformer for Protein-Ligand Docking.](http://arxiv.org/abs/2310.08061) | 提出了一种新颖的等变Transformer神经网络用于蛋白质-配体对接，通过融合配体的图层特征，并使用TAMformer模块学习配体和蛋白质的表示，实现了对配体位姿的准确预测，并通过迭代优化方法生成精炼的配体位姿。 |
| [^9] | [On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective.](http://arxiv.org/abs/2304.13836) | 本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。 |

# 详细

[^1]: 浅层ReLU神经网络和有限元

    Shallow ReLU neural networks and finite elements

    [https://arxiv.org/abs/2403.05809](https://arxiv.org/abs/2403.05809)

    在凸多面体网格上，提出了用两个隐藏层的ReLU神经网络来弱表示分段线性函数，并根据网格中的多面体和超平面的数量准确确定了所需的神经元数，建立了浅层ReLU神经网络和有限元函数之间的联系。

    

    我们指出在凸多面体网格上，可以用两个隐藏层的ReLU神经网络在弱意义下表示（连续或不连续的）分段线性函数。此外，基于涉及到的多面体和超平面的数量，准确给出了弱表示所需的两个隐藏层的神经元数。这些结果自然地适用于常数和线性有限元函数。这种弱表示建立了浅层ReLU神经网络和有限元函数之间的桥梁，并为通过有限元函数分析ReLU神经网络在$L^p$范数中的逼近能力提供了视角。此外，我们还讨论了最近张量神经网络对张量有限元函数的严格表示。

    arXiv:2403.05809v1 Announce Type: cross  Abstract: We point out that (continuous or discontinuous) piecewise linear functions on a convex polytope mesh can be represented by two-hidden-layer ReLU neural networks in a weak sense. In addition, the numbers of neurons of the two hidden layers required to weakly represent are accurately given based on the numbers of polytopes and hyperplanes involved in this mesh. The results naturally hold for constant and linear finite element functions. Such weak representation establishes a bridge between shallow ReLU neural networks and finite element functions, and leads to a perspective for analyzing approximation capability of ReLU neural networks in $L^p$ norm via finite element functions. Moreover, we discuss the strict representation for tensor finite element functions via the recent tensor neural networks.
    
[^2]: 深度ReLU神经网络中的几何引导隐式正则化

    Geometry-induced Implicit Regularization in Deep ReLU Neural Networks

    [https://arxiv.org/abs/2402.08269](https://arxiv.org/abs/2402.08269)

    通过研究参数变化时输出集合的几何特征，我们发现在深度ReLU神经网络的优化过程中存在几何引导的隐式正则化现象。

    

    众所周知，具有比训练样本更多参数的神经网络不会过拟合。隐式正则化现象在优化过程中出现，对“好”的网络有利。因此，如果我们不考虑所有可能的网络，而只考虑“好”的网络，参数数量就不是一个足够衡量复杂性的指标。为了更好地理解在优化过程中哪些网络受到青睐，我们研究了参数变化时输出集合的几何特征。当输入固定时，我们证明了这个集合的维度会发生变化，并且局部维度，即批次功能维度，几乎总是由隐藏层中的激活模式决定。我们证明了批次功能维度对网络参数化的对称性（神经元排列和正向缩放）是不变的。实证上，我们证实了在优化过程中批次功能维度会下降。因此，优化过程具有隐式正则化的效果。

    It is well known that neural networks with many more parameters than training examples do not overfit. Implicit regularization phenomena, which are still not well understood, occur during optimization and 'good' networks are favored. Thus the number of parameters is not an adequate measure of complexity if we do not consider all possible networks but only the 'good' ones. To better understand which networks are favored during optimization, we study the geometry of the output set as parameters vary. When the inputs are fixed, we prove that the dimension of this set changes and that the local dimension, called batch functional dimension, is almost surely determined by the activation patterns in the hidden layers. We prove that the batch functional dimension is invariant to the symmetries of the network parameterization: neuron permutations and positive rescalings. Empirically, we establish that the batch functional dimension decreases during optimization. As a consequence, optimization l
    
[^3]: 关于随机梯度下降（SGD）的收敛速度及其在修改的多臂赌博机上的策略梯度应用

    On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit

    [https://arxiv.org/abs/2402.06388](https://arxiv.org/abs/2402.06388)

    该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。

    

    我们提出了一个自包含的证明，证明了当学习速率遵循逆时间衰减规则时，随机梯度下降（SGD）的收敛速度；接下来，我们将这些结果应用于带有L2正则化的修改的策略梯度多臂赌博机（MAB）的收敛性分析。

    We present a self-contained proof of the convergence rate of the Stochastic Gradient Descent (SGD) when the learning rate follows an inverse time decays schedule; we next apply the results to the convergence of a modified form of policy gradient Multi-Armed Bandit (MAB) with $L2$ regularization.
    
[^4]: 超越期望: 现实中实现随机优势学习

    Beyond Expectations: Learning with Stochastic Dominance Made Practical

    [https://arxiv.org/abs/2402.02698](https://arxiv.org/abs/2402.02698)

    这项工作首次尝试建立了一个随机优势学习的通用框架，并推广了随机优势的概念以使其能够在任意两个随机变量之间进行比较。同时，我们还开发了一种有效的计算方法来处理连续性评估的问题。

    

    随机优势模型对决策时具有风险厌恶偏好的不确定结果进行建模，相比于仅仅依赖期望值，自然地捕捉了底层不确定性的内在结构。尽管在理论上具有吸引力，但随机优势在机器学习中的应用却很少，主要是由于以下挑战：$\textbf{i)}$ 随机优势的原始概念仅提供了$\textit{部分序}$，因此不能作为最优性准则；和 $\textbf{ii)}$ 由于评估随机优势的连续性本质，目前还缺乏高效的计算方法。在这项工作中，我们首次尝试建立一个与随机优势学习相关的通用框架。我们首先将随机优势概念推广，使得任意两个随机变量之间的比较成为可能。接下来我们开发了一个有效的计算方法，以解决评估随机优势的连续性问题。

    Stochastic dominance models risk-averse preferences for decision making with uncertain outcomes, which naturally captures the intrinsic structure of the underlying uncertainty, in contrast to simply resorting to the expectations. Despite theoretically appealing, the application of stochastic dominance in machine learning has been scarce, due to the following challenges: $\textbf{i)}$, the original concept of stochastic dominance only provides a $\textit{partial order}$, therefore, is not amenable to serve as an optimality criterion; and $\textbf{ii)}$, an efficient computational recipe remains lacking due to the continuum nature of evaluating stochastic dominance.%, which barriers its application for machine learning.   In this work, we make the first attempt towards establishing a general framework of learning with stochastic dominance. We first generalize the stochastic dominance concept to enable feasible comparisons between any arbitrary pair of random variables. We next develop a 
    
[^5]: 来自协方差核的高斯过程样本路径正则性

    Sample Path Regularity of Gaussian Processes from the Covariance Kernel

    [https://arxiv.org/abs/2312.14886](https://arxiv.org/abs/2312.14886)

    本文提供了关于高斯过程样本路径正则性的新颖和紧凑的特征描述，通过协方差核对应的GP样本路径达到一定正则性的充分必要条件，对常用于机器学习应用中的GPs的样本路径正则性进行了探讨。

    

    高斯过程（GPs）是定义函数空间上的概率分布的最常见形式主义。尽管GPs的应用广泛，但对于GP样本路径的全面理解，即它们定义概率测度的函数空间，尚缺乏。在实践中，GPs不是通过概率测度构建的，而是通过均值函数和协方差核构建的。本文针对协方差核提供了GP样本路径达到给定正则性所需的充分必要条件。我们使用H\"older正则性框架，因为它提供了特别简单的条件，在平稳和各向同性GPs的情况下进一步简化。然后，我们证明我们的结果允许对机器学习应用中常用的GPs的样本路径正则性进行新颖且异常紧凑的表征。

    arXiv:2312.14886v2 Announce Type: replace  Abstract: Gaussian processes (GPs) are the most common formalism for defining probability distributions over spaces of functions. While applications of GPs are myriad, a comprehensive understanding of GP sample paths, i.e. the function spaces over which they define a probability measure, is lacking. In practice, GPs are not constructed through a probability measure, but instead through a mean function and a covariance kernel. In this paper we provide necessary and sufficient conditions on the covariance kernel for the sample paths of the corresponding GP to attain a given regularity. We use the framework of H\"older regularity as it grants particularly straightforward conditions, which simplify further in the cases of stationary and isotropic GPs. We then demonstrate that our results allow for novel and unusually tight characterisations of the sample path regularities of the GPs commonly used in machine learning applications, such as the Mat\'
    
[^6]: MFAI:一种可扩展的贝叶斯矩阵分解方法来利用辅助信息

    MFAI: A Scalable Bayesian Matrix Factorization Approach to Leveraging Auxiliary Information

    [https://arxiv.org/abs/2303.02566](https://arxiv.org/abs/2303.02566)

    MFAI是一种可扩展的贝叶斯矩阵分解方法，通过利用辅助信息来克服由于数据质量差导致的挑战，具有灵活建模非线性关系和对辅助信息的鲁棒性。

    

    在各种实际情况下，矩阵分解方法在数据质量差的情况下往往表现不佳，例如数据稀疏性高和信噪比低。在这里，我们考虑利用辅助信息的矩阵分解问题，辅助信息在实际应用中是大量可用的，以克服由于数据质量差引起的挑战。与现有方法主要依赖于简单线性模型将辅助信息与主数据矩阵结合不同，我们提出将梯度增强树集成到概率矩阵分解框架中以有效地利用辅助信息(MFAI)。因此，MFAI自然地继承了梯度增强树的几个显著特点，如灵活建模非线性关系、对辅助信息中的不相关特征和缺失值具有鲁棒性。MFAI中的参数可以在经验贝叶斯框架下自动确定，使其适应于利用辅助信息。

    In various practical situations, matrix factorization methods suffer from poor data quality, such as high data sparsity and low signal-to-noise ratio (SNR). Here, we consider a matrix factorization problem by utilizing auxiliary information, which is massively available in real-world applications, to overcome the challenges caused by poor data quality. Unlike existing methods that mainly rely on simple linear models to combine auxiliary information with the main data matrix, we propose to integrate gradient boosted trees in the probabilistic matrix factorization framework to effectively leverage auxiliary information (MFAI). Thus, MFAI naturally inherits several salient features of gradient boosted trees, such as the capability of flexibly modeling nonlinear relationships and robustness to irrelevant features and missing values in auxiliary information. The parameters in MFAI can be automatically determined under the empirical Bayes framework, making it adaptive to the utilization of a
    
[^7]: 一种基于机器学习的概率暴露模型的德国高分辨率室内氡气地图

    A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model. (arXiv:2310.11143v1 [stat.ML])

    [http://arxiv.org/abs/2310.11143](http://arxiv.org/abs/2310.11143)

    本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。

    

    室内氡气是一种致癌的放射性气体，可以在室内积累。通常情况下，全国范围内的室内氡暴露是基于广泛的测量活动估计得来的。然而，样本的特征往往与人口特征不同，这是由于许多相关因素，如地质源氡气的可用性或楼层水平。此外，样本大小通常不允许以高空间分辨率进行暴露估计。我们提出了一种基于模型的方法，可以比纯数据方法更加现实地估计室内氡分布，并具有更高的空间分辨率。我们采用了两阶段建模方法：1）应用分位数回归森林，使用环境和建筑数据作为预测因子，估计了德国每个住宅楼的每个楼层的室内氡概率分布函数；2）使用概率蒙特卡罗抽样技术使它们组合和。

    Radon is a carcinogenic, radioactive gas that can accumulate indoors. Indoor radon exposure at the national scale is usually estimated on the basis of extensive measurement campaigns. However, characteristics of the sample often differ from the characteristics of the population due to the large number of relevant factors such as the availability of geogenic radon or floor level. Furthermore, the sample size usually does not allow exposure estimation with high spatial resolution. We propose a model-based approach that allows a more realistic estimation of indoor radon distribution with a higher spatial resolution than a purely data-based approach. We applied a two-stage modelling approach: 1) a quantile regression forest using environmental and building data as predictors was applied to estimate the probability distribution function of indoor radon for each floor level of each residential building in Germany; (2) a probabilistic Monte Carlo sampling technique enabled the combination and
    
[^8]: ETDock: 一种新颖的等变Transformer用于蛋白质-配体对接

    ETDock: A Novel Equivariant Transformer for Protein-Ligand Docking. (arXiv:2310.08061v1 [q-bio.BM])

    [http://arxiv.org/abs/2310.08061](http://arxiv.org/abs/2310.08061)

    提出了一种新颖的等变Transformer神经网络用于蛋白质-配体对接，通过融合配体的图层特征，并使用TAMformer模块学习配体和蛋白质的表示，实现了对配体位姿的准确预测，并通过迭代优化方法生成精炼的配体位姿。

    

    预测蛋白质和配体之间的对接是药物开发中关键且具有挑战性的任务。然而，传统的对接方法主要依赖于评分函数，而基于深度学习的对接方法通常忽略了蛋白质和配体的3D空间信息以及配体的图层特征，限制了它们的性能。为了解决这些限制，我们提出了一种蛋白质-配体对接位姿预测的等变Transformer神经网络。我们的方法通过特征处理来融合配体的图层特征，然后使用我们提出的TAMformer模块学习配体和蛋白质的表示。此外，我们采用基于预测的距离矩阵的迭代优化方法来生成精炼的配体位姿。实验结果表明，我们的模型可以达到最先进的性能水平。

    Predicting the docking between proteins and ligands is a crucial and challenging task for drug discovery. However, traditional docking methods mainly rely on scoring functions, and deep learning-based docking approaches usually neglect the 3D spatial information of proteins and ligands, as well as the graph-level features of ligands, which limits their performance. To address these limitations, we propose an equivariant transformer neural network for protein-ligand docking pose prediction. Our approach involves the fusion of ligand graph-level features by feature processing, followed by the learning of ligand and protein representations using our proposed TAMformer module. Additionally, we employ an iterative optimization approach based on the predicted distance matrix to generate refined ligand poses. The experimental results on real datasets show that our model can achieve state-of-the-art performance.
    
[^9]: 论RemOve-And-Retrain的陷阱：数据处理不等式的视角

    On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective. (arXiv:2304.13836v1 [cs.LG])

    [http://arxiv.org/abs/2304.13836](http://arxiv.org/abs/2304.13836)

    本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。

    

    本文评估了RemOve-And-Retrain（ROAR）协议的可靠性，该协议用于测量特征重要性估计的性能。我们从理论背景和实证实验中发现，具有较少有关决策功能的信息的属性在ROAR基准测试中表现更好，与ROAR的原始目的相矛盾。这种现象也出现在最近提出的变体RemOve-And-Debias（ROAD）中，我们提出了ROAR归因度量中毛糙度偏差的一致趋势。我们的结果提醒人们不要盲目依赖ROAR的性能评估指标。

    This paper assesses the reliability of the RemOve-And-Retrain (ROAR) protocol, which is used to measure the performance of feature importance estimates. Our findings from the theoretical background and empirical experiments indicate that attributions that possess less information about the decision function can perform better in ROAR benchmarks, conflicting with the original purpose of ROAR. This phenomenon is also observed in the recently proposed variant RemOve-And-Debias (ROAD), and we propose a consistent trend of blurriness bias in ROAR attribution metrics. Our results caution against uncritical reliance on ROAR metrics.
    

