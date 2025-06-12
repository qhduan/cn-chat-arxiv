# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Plug-and-Play image restoration with Stochastic deNOising REgularization](https://arxiv.org/abs/2402.01779) | 本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。 |
| [^2] | [Feature Normalization Prevents Collapse of Non-contrastive Learning Dynamics.](http://arxiv.org/abs/2309.16109) | 本论文研究了非对比学习中的动力崩溃问题，发现特征归一化可以防止此问题的出现，为解决自监督表示学习的计算效率提供了新的思路。 |
| [^3] | [Simulation-based inference using surjective sequential neural likelihood estimation.](http://arxiv.org/abs/2308.01054) | 我们提出了一种使用全射序列神经似然估计（SSNL）进行基于仿真的推断的新方法，在模型中无法计算似然函数并且只能使用模拟器生成数据的情况下，SSNL通过拟合降维的全射归一化流模型，并将其作为替代似然函数，解决了先前基于似然方法在高维数据集中遇到的问题，并在各种实验中展示了其优越性能。 |
| [^4] | [Optimal Estimation in Mixed-Membership Stochastic Block Models.](http://arxiv.org/abs/2307.14530) | 本论文研究了重叠社区检测问题，在混合成员随机块模型的基础上提出了一个新的估计器，并建立了估计误差的极小下界。 |
| [^5] | [Randomized Block-Coordinate Optimistic Gradient Algorithms for Root-Finding Problems.](http://arxiv.org/abs/2301.03113) | 本文提出了两种新的随机乐观梯度算法来解决大规模情况下的根查找问题。第一种算法在底层算子满足一定条件时可以达到较好的收敛速度，第二种算法是一种加速算法，可以更快地收敛。算法的收敛性和解的存在性也得到了证明。 |

# 详细

[^1]: 带有随机去噪正则化的即插即用图像恢复

    Plug-and-Play image restoration with Stochastic deNOising REgularization

    [https://arxiv.org/abs/2402.01779](https://arxiv.org/abs/2402.01779)

    本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。

    

    即插即用（PnP）算法是一类迭代算法，通过结合物理模型和深度神经网络进行正则化来解决图像反演问题。尽管这些算法能够产生令人印象深刻的图像恢复结果，但它们依赖于在迭代过程中越来越少噪音的图像上的一种非标准的去噪器使用方法，这与基于扩散模型（DM）的最新算法相矛盾，在这些算法中，去噪器仅应用于重新加噪的图像上。我们提出了一种新的PnP框架，称为随机去噪正则化（SNORE），它仅在噪声水平适当的图像上应用去噪器。它基于显式的随机正则化，从而导致了一种解决病态逆问题的随机梯度下降算法。我们提供了该算法及其退火扩展的收敛分析。在实验上，我们证明SNORE在去模糊和修复任务上与最先进的方法相竞争。

    Plug-and-Play (PnP) algorithms are a class of iterative algorithms that address image inverse problems by combining a physical model and a deep neural network for regularization. Even if they produce impressive image restoration results, these algorithms rely on a non-standard use of a denoiser on images that are less and less noisy along the iterations, which contrasts with recent algorithms based on Diffusion Models (DM), where the denoiser is applied only on re-noised images. We propose a new PnP framework, called Stochastic deNOising REgularization (SNORE), which applies the denoiser only on images with noise of the adequate level. It is based on an explicit stochastic regularization, which leads to a stochastic gradient descent algorithm to solve ill-posed inverse problems. A convergence analysis of this algorithm and its annealing extension is provided. Experimentally, we prove that SNORE is competitive with respect to state-of-the-art methods on deblurring and inpainting tasks, 
    
[^2]: 特征归一化防止非对比学习动力的崩溃

    Feature Normalization Prevents Collapse of Non-contrastive Learning Dynamics. (arXiv:2309.16109v1 [cs.LG])

    [http://arxiv.org/abs/2309.16109](http://arxiv.org/abs/2309.16109)

    本论文研究了非对比学习中的动力崩溃问题，发现特征归一化可以防止此问题的出现，为解决自监督表示学习的计算效率提供了新的思路。

    

    对比学习是一种自监督表示学习框架，通过数据增强生成的两个正视图在数据表示空间中通过吸引力使它们相似，而通过排斥力使它们远离负样本。非对比学习通过BYOL和SimSiam等手段去除了负样本，并提高了计算效率。虽然由于缺乏排斥力，学到的表示可能会崩溃成一个单点，但田等人（2021）通过学习动力分析揭示，如果数据增强足够强于正则化，则表示可以避免崩溃。然而，他们的分析没有考虑常用的特征归一化，即在衡量表示相似性之前进行的归一化操作，因此过强的正则化可能会导致动力崩溃，这在特征归一化存在的情况下是不自然的行为。

    Contrastive learning is a self-supervised representation learning framework, where two positive views generated through data augmentation are made similar by an attraction force in a data representation space, while a repulsive force makes them far from negative examples. Non-contrastive learning, represented by BYOL and SimSiam, further gets rid of negative examples and improves computational efficiency. While learned representations may collapse into a single point due to the lack of the repulsive force at first sight, Tian et al. (2021) revealed through the learning dynamics analysis that the representations can avoid collapse if data augmentation is sufficiently stronger than regularization. However, their analysis does not take into account commonly-used feature normalization, a normalizer before measuring the similarity of representations, and hence excessively strong regularization may collapse the dynamics, which is an unnatural behavior under the presence of feature normalizat
    
[^3]: 使用全射序列神经似然估计进行基于仿真的推断

    Simulation-based inference using surjective sequential neural likelihood estimation. (arXiv:2308.01054v1 [stat.ML])

    [http://arxiv.org/abs/2308.01054](http://arxiv.org/abs/2308.01054)

    我们提出了一种使用全射序列神经似然估计（SSNL）进行基于仿真的推断的新方法，在模型中无法计算似然函数并且只能使用模拟器生成数据的情况下，SSNL通过拟合降维的全射归一化流模型，并将其作为替代似然函数，解决了先前基于似然方法在高维数据集中遇到的问题，并在各种实验中展示了其优越性能。

    

    我们提出了全射序列神经似然（SSNL）估计方法，这是一种在模型中无法计算似然函数并且只能使用可以生成合成数据的模拟器时进行基于仿真的推断的新方法。SSNL拟合一个降维的全射归一化流模型，并将其用作替代似然函数，从而可以使用传统的贝叶斯推断方法，包括马尔科夫链蒙特卡罗方法或变分推断。通过将数据嵌入到低维空间中，SSNL解决了先前基于似然方法在应用于高维数据集时遇到的几个问题，例如包含无信息数据维度或位于较低维流形上的数据。我们对SSNL在各种实验中进行了评估，并表明它通常优于在基于仿真推断中使用的现代方法，例如在一项来自天体物理学的具有挑战性的真实世界例子上对磁场模型的建模。

    We present Surjective Sequential Neural Likelihood (SSNL) estimation, a novel method for simulation-based inference in models where the evaluation of the likelihood function is not tractable and only a simulator that can generate synthetic data is available. SSNL fits a dimensionality-reducing surjective normalizing flow model and uses it as a surrogate likelihood function which allows for conventional Bayesian inference using either Markov chain Monte Carlo methods or variational inference. By embedding the data in a low-dimensional space, SSNL solves several issues previous likelihood-based methods had when applied to high-dimensional data sets that, for instance, contain non-informative data dimensions or lie along a lower-dimensional manifold. We evaluate SSNL on a wide variety of experiments and show that it generally outperforms contemporary methods used in simulation-based inference, for instance, on a challenging real-world example from astrophysics which models the magnetic fi
    
[^4]: 混合成员随机块模型中的最优估计

    Optimal Estimation in Mixed-Membership Stochastic Block Models. (arXiv:2307.14530v1 [stat.ML])

    [http://arxiv.org/abs/2307.14530](http://arxiv.org/abs/2307.14530)

    本论文研究了重叠社区检测问题，在混合成员随机块模型的基础上提出了一个新的估计器，并建立了估计误差的极小下界。

    

    社区检测是现代网络科学中最关键的问题之一。其应用可以在各个领域找到，从蛋白质建模到社交网络分析。最近，出现了许多论文研究重叠社区检测问题，即网络中的每个节点可能属于多个社区。在本文中，我们考虑了由Airoldi等人（2008）首次提出的混合成员随机块模型（MMSB）。MMSB在图中对重叠社区结构提供了相当一般的设置。本文的核心问题是在观察到的网络中重建社区之间的关系。我们比较了不同的方法，并建立了估计误差的极小下界。然后，我们提出了一个与这个下界匹配的新估计器。理论结果在对所考虑的模型的相当普遍条件下得到证明。最后，我们通过一系列实验来说明这个理论。

    Community detection is one of the most critical problems in modern network science. Its applications can be found in various fields, from protein modeling to social network analysis. Recently, many papers appeared studying the problem of overlapping community detection, where each node of a network may belong to several communities. In this work, we consider Mixed-Membership Stochastic Block Model (MMSB) first proposed by Airoldi et al. (2008). MMSB provides quite a general setting for modeling overlapping community structure in graphs. The central question of this paper is to reconstruct relations between communities given an observed network. We compare different approaches and establish the minimax lower bound on the estimation error. Then, we propose a new estimator that matches this lower bound. Theoretical results are proved under fairly general conditions on the considered model. Finally, we illustrate the theory in a series of experiments.
    
[^5]: 随机块坐标乐观梯度算法解决根查找问题

    Randomized Block-Coordinate Optimistic Gradient Algorithms for Root-Finding Problems. (arXiv:2301.03113v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2301.03113](http://arxiv.org/abs/2301.03113)

    本文提出了两种新的随机乐观梯度算法来解决大规模情况下的根查找问题。第一种算法在底层算子满足一定条件时可以达到较好的收敛速度，第二种算法是一种加速算法，可以更快地收敛。算法的收敛性和解的存在性也得到了证明。

    

    本文提出了两种新的随机块坐标乐观梯度算法，用于在大规模情况下近似求解非线性方程，也称为根查找问题。第一种算法使用恒定的步长，非加速算法，在底层算子G满足Lipschitz连续性和弱Minty解条件时，它在数学期望E[||Gx^k||^2]上达到O(1/k)的最优迭代收敛速度，其中E[·]表示期望，k为迭代计数器。第二种方法是一种新的加速随机块坐标乐观梯度算法，在G的夹逼性条件下，该算法在E[||Gx^k||^2]和E[||x^{k+1} x^{k}||^2]上分别达到O(1/k^2)和o(1/k^2)的迭代收敛速度。此外，我们证明迭代序列{x^k}几乎必然收敛到一个解，以及在此解处Gx^k的模的平方在...

    In this paper, we develop two new randomized block-coordinate optimistic gradient algorithms to approximate a solution of nonlinear equations in large-scale settings, which are called root-finding problems. Our first algorithm is non-accelerated with constant stepsizes, and achieves $\mathcal{O}(1/k)$ best-iterate convergence rate on $\mathbb{E}[ \Vert Gx^k\Vert^2]$ when the underlying operator $G$ is Lipschitz continuous and satisfies a weak Minty solution condition, where $\mathbb{E}[\cdot]$ is the expectation and $k$ is the iteration counter. Our second method is a new accelerated randomized block-coordinate optimistic gradient algorithm. We establish both $\mathcal{O}(1/k^2)$ and $o(1/k^2)$ last-iterate convergence rates on both $\mathbb{E}[ \Vert Gx^k\Vert^2]$ and $\mathbb{E}[ \Vert x^{k+1} x^{k}\Vert^2]$ for this algorithm under the co-coerciveness of $G$. In addition, we prove that the iterate sequence $\{x^k\}$ converges to a solution almost surely, and $\Vert Gx^k\Vert^2$ at
    

