# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Structure-Preserving Kernel Method for Learning Hamiltonian Systems](https://arxiv.org/abs/2403.10070) | 提出了一种保结构的核岭回归方法，可以从噪声观测数据中恢复哈密顿函数，拓展了核回归方法，并具有出色的数值性能和收敛速度。 |
| [^2] | [Structured Matrix Learning under Arbitrary Entrywise Dependence and Estimation of Markov Transition Kernel.](http://arxiv.org/abs/2401.02520) | 本论文提出了在任意元素间依赖下进行结构化矩阵估计的通用框架，并证明了提出的最小二乘估计器在各种噪声分布下的紧致性。此外，论文还提出了一个新颖的结果，论述了无关低秩矩阵的结构特点。最后，论文还展示了该框架在结构化马尔可夫转移核估计问题中的应用。 |
| [^3] | [Asymptotically Efficient Online Learning for Censored Regression Models Under Non-I.I.D Data.](http://arxiv.org/abs/2309.09454) | 这项研究提出了一种渐近高效的在线学习方法，应用于随机被审查回归模型，并在一般情况下达到了最好的性能。 |
| [^4] | [Incremental Outlier Detection Modelling Using Streaming Analytics in Finance & Health Care.](http://arxiv.org/abs/2305.09907) | 本文利用在线异常检测算法建立了流式环境下的增量学习模型，在金融和医疗保健领域取得实际应用。 |
| [^5] | [Orthogonal Non-negative Matrix Factorization: a Maximum-Entropy-Principle Approach.](http://arxiv.org/abs/2210.02672) | 本文提出了一种新的解决正交非负矩阵分解问题的方法，该方法使用了基于最大熵原则的解决方案，并保证了矩阵的正交性和稀疏性以及非负性。该方法在不影响近似质量的情况下具有较好的性能速度和优于文献中类似方法的稀疏性、正交性。 |

# 详细

[^1]: 用于学习哈密顿系统的保结构核方法

    A Structure-Preserving Kernel Method for Learning Hamiltonian Systems

    [https://arxiv.org/abs/2403.10070](https://arxiv.org/abs/2403.10070)

    提出了一种保结构的核岭回归方法，可以从噪声观测数据中恢复哈密顿函数，拓展了核回归方法，并具有出色的数值性能和收敛速度。

    

    提出了一种保结构的核岭回归方法，允许从包含哈密顿向量场的噪声观测数据集中恢复潜在的高维非线性哈密顿函数。该方法提出了一个闭式解，在这一设置中表现出优秀的数值性能，超越了文献中提出的其他技术。从方法论的角度看，该论文扩展了核回归方法，解决需要包含梯度线性函数的损失函数的问题，特别地，在这一背景下证明了微分再现属性和表示定理。分析了保结构核估计器和高斯后验均值估计器之间的关系。进行了完整的误差分析，提供使用固定和自适应正则化参数的收敛速度。所提出方法的优良性能得到了确认。

    arXiv:2403.10070v1 Announce Type: cross  Abstract: A structure-preserving kernel ridge regression method is presented that allows the recovery of potentially high-dimensional and nonlinear Hamiltonian functions out of datasets made of noisy observations of Hamiltonian vector fields. The method proposes a closed-form solution that yields excellent numerical performances that surpass other techniques proposed in the literature in this setup. From the methodological point of view, the paper extends kernel regression methods to problems in which loss functions involving linear functions of gradients are required and, in particular, a differential reproducing property and a Representer Theorem are proved in this context. The relation between the structure-preserving kernel estimator and the Gaussian posterior mean estimator is analyzed. A full error analysis is conducted that provides convergence rates using fixed and adaptive regularization parameters. The good performance of the proposed 
    
[^2]: 在任意元素间依赖下的结构化矩阵学习与马尔可夫转移核估计

    Structured Matrix Learning under Arbitrary Entrywise Dependence and Estimation of Markov Transition Kernel. (arXiv:2401.02520v1 [stat.ML])

    [http://arxiv.org/abs/2401.02520](http://arxiv.org/abs/2401.02520)

    本论文提出了在任意元素间依赖下进行结构化矩阵估计的通用框架，并证明了提出的最小二乘估计器在各种噪声分布下的紧致性。此外，论文还提出了一个新颖的结果，论述了无关低秩矩阵的结构特点。最后，论文还展示了该框架在结构化马尔可夫转移核估计问题中的应用。

    

    结构化矩阵估计问题通常在强噪声依赖假设下进行研究。本文考虑噪声低秩加稀疏矩阵恢复的一般框架，其中噪声矩阵可以来自任意具有元素间任意依赖的联合分布。我们提出了一个无关相位约束的最小二乘估计器，并且证明了它在各种噪声分布下都是紧致的，既满足确定性下界又匹配最小化风险。为了实现这一点，我们建立了一个新颖的结果，断言两个任意的低秩无关矩阵之间的差异必须在其元素上扩散能量，换句话说不能太稀疏，这揭示了无关低秩矩阵的结构，可能引起独立兴趣。然后，我们展示了我们框架在几个重要的统计机器学习问题中的应用。在估计结构化马尔可夫转移核的问题中，采用了这种方法。

    The problem of structured matrix estimation has been studied mostly under strong noise dependence assumptions. This paper considers a general framework of noisy low-rank-plus-sparse matrix recovery, where the noise matrix may come from any joint distribution with arbitrary dependence across entries. We propose an incoherent-constrained least-square estimator and prove its tightness both in the sense of deterministic lower bound and matching minimax risks under various noise distributions. To attain this, we establish a novel result asserting that the difference between two arbitrary low-rank incoherent matrices must spread energy out across its entries, in other words cannot be too sparse, which sheds light on the structure of incoherent low-rank matrices and may be of independent interest. We then showcase the applications of our framework to several important statistical machine learning problems. In the problem of estimating a structured Markov transition kernel, the proposed method
    
[^3]: 非独立同分布数据条件下渐近高效的在线学习方法在被审查回归模型中的应用

    Asymptotically Efficient Online Learning for Censored Regression Models Under Non-I.I.D Data. (arXiv:2309.09454v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.09454](http://arxiv.org/abs/2309.09454)

    这项研究提出了一种渐近高效的在线学习方法，应用于随机被审查回归模型，并在一般情况下达到了最好的性能。

    

    本文研究了渐近高效的在线学习方法在随机被审查回归模型中的应用，该模型涉及到学习和统计学的各个领域，但迄今为止仍缺乏关于学习算法效率的全面理论研究。为此，我们提出了一种两步在线算法，第一步专注于实现算法收敛性，第二步用于改善估计性能。在数据的一般激励条件下，我们通过应用随机李雅普诺夫函数方法和对鞅的极限理论，证明了我们的算法是强一致的和渐近正态的。此外，我们还证明了估计值的协方差在渐近上可以达到克拉美洛界，这意味着所提出的算法的性能在一般情况下是可以期望的最好的。与大多数现有的研究不同，我们的结果是不依赖传统方法而得出的。

    The asymptotically efficient online learning problem is investigated for stochastic censored regression models, which arise from various fields of learning and statistics but up to now still lacks comprehensive theoretical studies on the efficiency of the learning algorithms. For this, we propose a two-step online algorithm, where the first step focuses on achieving algorithm convergence, and the second step is dedicated to improving the estimation performance. Under a general excitation condition on the data, we show that our algorithm is strongly consistent and asymptotically normal by employing the stochastic Lyapunov function method and limit theories for martingales. Moreover, we show that the covariances of the estimates can achieve the Cramer-Rao (C-R) bound asymptotically, indicating that the performance of the proposed algorithm is the best possible that one can expect in general. Unlike most of the existing works, our results are obtained without resorting to the traditionall
    
[^4]: 利用流式分析在金融和医疗保健中进行增量异常检测建模

    Incremental Outlier Detection Modelling Using Streaming Analytics in Finance & Health Care. (arXiv:2305.09907v1 [cs.LG])

    [http://arxiv.org/abs/2305.09907](http://arxiv.org/abs/2305.09907)

    本文利用在线异常检测算法建立了流式环境下的增量学习模型，在金融和医疗保健领域取得实际应用。

    

    本文构建了在线模型，该模型通过使用流环境下的在线异常检测算法进行增量构建。我们认识到应当使用流式模型来处理流式数据的高度必要性。本项目的目标是研究和分析适用于现实环境的流式模型的重要性。本文实现了各种异常检测算法，如One class支持向量机（OC-SVM）、孤立森林自适应滑动窗口方法（IForest ASD）、Exact Storm、基于角度的异常检测（ABOD）、局部异常因子（LOF）、KitNet、KNN ASD方法。并验证了上述构建模型在各种金融问题上的有效性和正确性，例如信用卡欺诈检测、流失预测、以太坊欺诈预测。此外，我们还分析了模型在健康预测问题上的表现，如心脏中风预测、糖尿病预测等。

    In this paper, we had built the online model which are built incrementally by using online outlier detection algorithms under the streaming environment. We identified that there is highly necessity to have the streaming models to tackle the streaming data. The objective of this project is to study and analyze the importance of streaming models which is applicable in the real-world environment. In this work, we built various Outlier Detection (OD) algorithms viz., One class Support Vector Machine (OC-SVM), Isolation Forest Adaptive Sliding window approach (IForest ASD), Exact Storm, Angle based outlier detection (ABOD), Local outlier factor (LOF), KitNet, KNN ASD methods. The effectiveness and validity of the above-built models on various finance problems such as credit card fraud detection, churn prediction, ethereum fraud prediction. Further, we also analyzed the performance of the models on the health care prediction problems such as heart stroke prediction, diabetes prediction and h
    
[^5]: 正交非负矩阵分解:最大熵原则方法

    Orthogonal Non-negative Matrix Factorization: a Maximum-Entropy-Principle Approach. (arXiv:2210.02672v2 [cs.DS] UPDATED)

    [http://arxiv.org/abs/2210.02672](http://arxiv.org/abs/2210.02672)

    本文提出了一种新的解决正交非负矩阵分解问题的方法，该方法使用了基于最大熵原则的解决方案，并保证了矩阵的正交性和稀疏性以及非负性。该方法在不影响近似质量的情况下具有较好的性能速度和优于文献中类似方法的稀疏性、正交性。

    

    本文提出了一种解决正交非负矩阵分解（ONMF）问题的新方法，该问题的目标是通过两个非负矩阵（特征矩阵和混合矩阵）的乘积来近似输入数据矩阵，其中一个矩阵是正交的。我们展示了如何将ONMF解释为特定的设施定位问题，并针对ONMF问题采用基于最大熵原则的FLP解决方案进行了调整。所提出的方法保证了特征矩阵或混合矩阵的正交性和稀疏性，同时确保了两者的非负性。此外，我们的方法还开发了一个定量的“真实”潜在特征数量的特征-超参数用于ONMF。针对合成数据集以及标准的基因芯片数组数据集进行的评估表明，该方法在不影响近似质量的情况下具有较好的稀疏性、正交性和性能速度，相对于文献中类似方法有显著的改善。

    In this paper, we introduce a new methodology to solve the orthogonal nonnegative matrix factorization (ONMF) problem, where the objective is to approximate an input data matrix by a product of two nonnegative matrices, the features matrix and the mixing matrix, where one of them is orthogonal. We show how the ONMF can be interpreted as a specific facility-location problem (FLP), and adapt a maximum-entropy-principle based solution for FLP to the ONMF problem. The proposed approach guarantees orthogonality and sparsity of the features or the mixing matrix, while ensuring nonnegativity of both. Additionally, our methodology develops a quantitative characterization of ``true" number of underlying features - a hyperparameter required for the ONMF. An evaluation of the proposed method conducted on synthetic datasets, as well as a standard genetic microarray dataset indicates significantly better sparsity, orthogonality, and performance speed compared to similar methods in the literature, w
    

