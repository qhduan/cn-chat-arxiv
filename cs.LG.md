# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Critic-Actor for Average Reward MDPs with Function Approximation: A Finite-Time Analysis](https://rss.arxiv.org/abs/2402.01371) | 本论文提出了一个评论家-演员算法，解决了长期平均奖励设置中的函数逼近问题，并进行了有限时间分析。实验结果表明，我们的算法能够在评论家的均方误差上界为$\epsilon$的情况下，获得样本复杂度为$\mathcal{\tilde{O}}(\epsilon^{-2.08})$，优于演员-评论家算法的结果。 |
| [^2] | [Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks](https://arxiv.org/abs/2403.14488) | 该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。 |
| [^3] | [Semisupervised score based matching algorithm to evaluate the effect of public health interventions](https://arxiv.org/abs/2403.12367) | 提出了一种基于二次评分函数的一对一匹配算法，通过设计权重最小化配对训练单元之间的得分差异，同时最大化未配对训练单元之间的得分差异 |
| [^4] | [TorchCP: A Library for Conformal Prediction based on PyTorch](https://arxiv.org/abs/2402.12683) | TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output. |
| [^5] | [Survey of Privacy Threats and Countermeasures in Federated Learning](https://arxiv.org/abs/2402.00342) | 本文调查了联邦学习中的隐私威胁和对策，并对水平联邦学习、垂直联邦学习和迁移联邦学习的典型类型进行了分类和描述。 |
| [^6] | [Guaranteed Nonconvex Factorization Approach for Tensor Train Recovery.](http://arxiv.org/abs/2401.02592) | 本研究提出了一种保证非凸分解方法用于张量列车恢复的新方法。该方法通过优化左正交TT格式来实现正交结构，并使用黎曼梯度下降算法来优化因子。我们证明了该方法具有局部线性收敛性，并且在满足受限等谱性质的条件下能够以线性速率收敛到真实张量。 |
| [^7] | [Finite Time Analysis of Constrained Actor Critic and Constrained Natural Actor Critic Algorithms.](http://arxiv.org/abs/2310.16363) | 本文研究了受约束的Actor Critic和受约束的Natural Actor Critic算法的有限时间分析，证明了这些算法能找到性能函数的一阶稳定点，并且具有较低的样本复杂度。 |
| [^8] | [Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation.](http://arxiv.org/abs/2309.08289) | 本研究利用几何深度学习和去噪扩散概率模型优化大肠的分割结果，并结合先进的表面重构模型，实现对大肠3D形状的精化恢复。 |
| [^9] | [Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model.](http://arxiv.org/abs/2306.12968) | 本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。 |
| [^10] | [Label Embedding by Johnson-Lindenstrauss Matrices.](http://arxiv.org/abs/2305.19470) | 这篇论文提出了基于JLMs的标签嵌入方法，将多元分类问题转化为有限回归问题，具有较高的计算效率和预测准确性。 |

# 详细

[^1]: Critic-Actor算法在平均奖励MDPs中的函数逼近问题：有限时间分析

    Critic-Actor for Average Reward MDPs with Function Approximation: A Finite-Time Analysis

    [https://rss.arxiv.org/abs/2402.01371](https://rss.arxiv.org/abs/2402.01371)

    本论文提出了一个评论家-演员算法，解决了长期平均奖励设置中的函数逼近问题，并进行了有限时间分析。实验结果表明，我们的算法能够在评论家的均方误差上界为$\epsilon$的情况下，获得样本复杂度为$\mathcal{\tilde{O}}(\epsilon^{-2.08})$，优于演员-评论家算法的结果。

    

    最近，关于两个时间尺度演员-评论家算法的渐近和非渐近收敛分析的研究工作非常活跃，其中演员的更新速度比评论家慢。在最近的一项工作中，提出了一个评论家-演员算法，用于无限时域折扣成本设置中的查找表情况，其中演员和评论家的时间尺度相反，并给出了渐近收敛分析。在我们的工作中，我们首次提出了一个具有函数逼近的评论家-演员算法，并在长期平均奖励设置中进行了首次有限时间（非渐近）分析。我们得到了最优的学习速率，并证明了我们的算法从评论家的均方误差上界为$\epsilon$，其样本复杂度为$\mathcal{\tilde{O}}(\epsilon^{-2.08})$，此结果比演员-评论家算法获得的结果要好。

    In recent years, there has been a lot of research work activity focused on carrying out asymptotic and non-asymptotic convergence analyses for two-timescale actor critic algorithms where the actor updates are performed on a timescale that is slower than that of the critic. In a recent work, the critic-actor algorithm has been presented for the infinite horizon discounted cost setting in the look-up table case where the timescales of the actor and the critic are reversed and asymptotic convergence analysis has been presented. In our work, we present the first critic-actor algorithm with function approximation and in the long-run average reward setting and present the first finite-time (non-asymptotic) analysis of such a scheme. We obtain optimal learning rates and prove that our algorithm achieves a sample complexity of $\mathcal{\tilde{O}}(\epsilon^{-2.08})$ for the mean squared error of the critic to be upper bounded by $\epsilon$ which is better than the one obtained for actor-critic
    
[^2]: 基于物理学因果推理的机器人操作任务中安全稳健的下一最佳动作选择

    Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks

    [https://arxiv.org/abs/2403.14488](https://arxiv.org/abs/2403.14488)

    该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。

    

    安全高效的物体操作是许多真实世界机器人应用的关键推手。然而，这种挑战在于机器人操作必须对一系列传感器和执行器的不确定性具有稳健性。本文提出了一个基于物理知识和因果推理的框架，用于让机器人在部分可观察的环境中对候选动作进行概率推理，以完成一个积木堆叠任务。我们将刚体系统动力学的基于物理学的仿真与因果贝叶斯网络（CBN）结合起来，定义了机器人决策过程的因果生成概率模型。通过基于仿真的蒙特卡洛实验，我们展示了我们的框架成功地能够：(1) 高准确度地预测积木塔的稳定性（预测准确率：88.6%）；和，(2) 为积木堆叠任务选择一个近似的下一最佳动作，供整合的机器人系统执行，实现94.2%的任务成功率。

    arXiv:2403.14488v1 Announce Type: cross  Abstract: Safe and efficient object manipulation is a key enabler of many real-world robot applications. However, this is challenging because robot operation must be robust to a range of sensor and actuator uncertainties. In this paper, we present a physics-informed causal-inference-based framework for a robot to probabilistically reason about candidate actions in a block stacking task in a partially observable setting. We integrate a physics-based simulation of the rigid-body system dynamics with a causal Bayesian network (CBN) formulation to define a causal generative probabilistic model of the robot decision-making process. Using simulation-based Monte Carlo experiments, we demonstrate our framework's ability to successfully: (1) predict block tower stability with high accuracy (Pred Acc: 88.6%); and, (2) select an approximate next-best action for the block stacking task, for execution by an integrated robot system, achieving 94.2% task succe
    
[^3]: 半监督计分匹配算法评估公共卫生干预效果

    Semisupervised score based matching algorithm to evaluate the effect of public health interventions

    [https://arxiv.org/abs/2403.12367](https://arxiv.org/abs/2403.12367)

    提出了一种基于二次评分函数的一对一匹配算法，通过设计权重最小化配对训练单元之间的得分差异，同时最大化未配对训练单元之间的得分差异

    

    多元匹配算法在观察性研究中“配对”相似的研究单元，以消除由于缺乏随机性而引起的潜在偏倚和混杂效应。我们提出了一种基于二次评分函数的新型一对一匹配算法，权重$\beta$被设计为最小化配对训练单元之间的得分差异，同时最大化未配对训练单元之间的得分差异。

    arXiv:2403.12367v1 Announce Type: cross  Abstract: Multivariate matching algorithms "pair" similar study units in an observational study to remove potential bias and confounding effects caused by the absence of randomizations. In one-to-one multivariate matching algorithms, a large number of "pairs" to be matched could mean both the information from a large sample and a large number of tasks, and therefore, to best match the pairs, such a matching algorithm with efficiency and comparatively limited auxiliary matching knowledge provided through a "training" set of paired units by domain experts, is practically intriguing.   We proposed a novel one-to-one matching algorithm based on a quadratic score function $S_{\beta}(x_i,x_j)= \beta^T (x_i-x_j)(x_i-x_j)^T \beta$. The weights $\beta$, which can be interpreted as a variable importance measure, are designed to minimize the score difference between paired training units while maximizing the score difference between unpaired training units
    
[^4]: TorchCP：基于PyTorch的一种适用于合拟常规预测的库

    TorchCP: A Library for Conformal Prediction based on PyTorch

    [https://arxiv.org/abs/2402.12683](https://arxiv.org/abs/2402.12683)

    TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output.

    

    TorchCP是一个用于深度学习模型上的合拟常规预测研究的Python工具包。它包含了用于后验和训练方法的各种实现，用于分类和回归任务（包括多维输出）。TorchCP建立在PyTorch之上，并利用矩阵计算的优势，提供简洁高效的推理实现。该代码采用LGPL许可证，并在$\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$开源。

    arXiv:2402.12683v1 Announce Type: new  Abstract: TorchCP is a Python toolbox for conformal prediction research on deep learning models. It contains various implementations for posthoc and training methods for classification and regression tasks (including multi-dimension output). TorchCP is built on PyTorch (Paszke et al., 2019) and leverages the advantages of matrix computation to provide concise and efficient inference implementations. The code is licensed under the LGPL license and is open-sourced at $\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$.
    
[^5]: 联邦学习中隐私威胁和对策的调查

    Survey of Privacy Threats and Countermeasures in Federated Learning

    [https://arxiv.org/abs/2402.00342](https://arxiv.org/abs/2402.00342)

    本文调查了联邦学习中的隐私威胁和对策，并对水平联邦学习、垂直联邦学习和迁移联邦学习的典型类型进行了分类和描述。

    

    联邦学习被广泛认为是一种注重隐私的学习方法，因为客户端之间没有直接交换训练数据。然而，联邦学习中存在隐私威胁，并且已经对隐私对策进行了研究。然而，我们注意到常见和独特的隐私威胁在典型类型的联邦学习中尚未以全面和具体的方式进行分类和描述。在本文中，我们描述了水平联邦学习、垂直联邦学习和迁移联邦学习的隐私威胁和对策。

    Federated learning is widely considered to be as a privacy-aware learning method because no training data is exchanged directly between clients. Nevertheless, there are threats to privacy in federated learning, and privacy countermeasures have been studied. However, we note that common and unique privacy threats among typical types of federated learning have not been categorized and described in a comprehensive and specific way. In this paper, we describe privacy threats and countermeasures for the typical types of federated learning; horizontal federated learning, vertical federated learning, and transfer federated learning.
    
[^6]: 保证非凸分解方法用于张量列车恢复的研究

    Guaranteed Nonconvex Factorization Approach for Tensor Train Recovery. (arXiv:2401.02592v1 [stat.ML])

    [http://arxiv.org/abs/2401.02592](http://arxiv.org/abs/2401.02592)

    本研究提出了一种保证非凸分解方法用于张量列车恢复的新方法。该方法通过优化左正交TT格式来实现正交结构，并使用黎曼梯度下降算法来优化因子。我们证明了该方法具有局部线性收敛性，并且在满足受限等谱性质的条件下能够以线性速率收敛到真实张量。

    

    在本文中，我们首次提供了对于分解方法的收敛性保证。具体而言，为了避免尺度歧义并便于理论分析，我们优化所谓的左正交TT格式，强制使大部分因子彼此正交。为了确保正交结构，我们利用黎曼梯度下降（RGD）来优化Stiefel流形上的这些因子。我们首先深入研究TT分解问题，并建立了RGD的局部线性收敛性。值得注意的是，随着张量阶数的增加，收敛速率仅经历线性下降。然后，我们研究了感知问题，即从线性测量中恢复TT格式张量。假设感知算子满足受限等谱性质（RIP），我们证明在适当的初始化下，通过谱初始化获得，RGD也会以线性速率收敛到真实张量。此外，我们扩展了我们的研究。

    In this paper, we provide the first convergence guarantee for the factorization approach. Specifically, to avoid the scaling ambiguity and to facilitate theoretical analysis, we optimize over the so-called left-orthogonal TT format which enforces orthonormality among most of the factors. To ensure the orthonormal structure, we utilize the Riemannian gradient descent (RGD) for optimizing those factors over the Stiefel manifold. We first delve into the TT factorization problem and establish the local linear convergence of RGD. Notably, the rate of convergence only experiences a linear decline as the tensor order increases. We then study the sensing problem that aims to recover a TT format tensor from linear measurements. Assuming the sensing operator satisfies the restricted isometry property (RIP), we show that with a proper initialization, which could be obtained through spectral initialization, RGD also converges to the ground-truth tensor at a linear rate. Furthermore, we expand our 
    
[^7]: 受约束的Actor Critic和受约束的Natural Actor Critic算法的有限时间分析

    Finite Time Analysis of Constrained Actor Critic and Constrained Natural Actor Critic Algorithms. (arXiv:2310.16363v1 [cs.LG])

    [http://arxiv.org/abs/2310.16363](http://arxiv.org/abs/2310.16363)

    本文研究了受约束的Actor Critic和受约束的Natural Actor Critic算法的有限时间分析，证明了这些算法能找到性能函数的一阶稳定点，并且具有较低的样本复杂度。

    

    Actor Critic方法在广泛的强化学习任务中找到了巨大的应用，特别是当状态-动作空间很大的时候。本文考虑使用函数逼近的actor critic和natural actor critic算法来处理涉及不等式约束的马尔可夫决策过程（C-MDP），并在非 i.i.d（马尔可夫）环境中进行了非渐近分析。我们考虑长期平均成本准则，其中目标和约束函数都是某些规定成本函数的适当策略依赖的长期平均。我们使用拉格朗日乘子法处理不等式约束。我们证明这些算法保证能找到性能（拉格朗日）函数$L(\theta,\gamma)$的一阶稳定点（即$\Vert \nabla L(\theta,\gamma)\Vert_2^2 \leq \epsilon$），并且其样本复杂度为$\mathcal{\tilde{O}}(\epsilon^{-2.5})$。

    Actor Critic methods have found immense applications on a wide range of Reinforcement Learning tasks especially when the state-action space is large. In this paper, we consider actor critic and natural actor critic algorithms with function approximation for constrained Markov decision processes (C-MDP) involving inequality constraints and carry out a non-asymptotic analysis for both of these algorithms in a non-i.i.d (Markovian) setting. We consider the long-run average cost criterion where both the objective and the constraint functions are suitable policy-dependent long-run averages of certain prescribed cost functions. We handle the inequality constraints using the Lagrange multiplier method. We prove that these algorithms are guaranteed to find a first-order stationary point (i.e., $\Vert \nabla L(\theta,\gamma)\Vert_2^2 \leq \epsilon$) of the performance (Lagrange) function $L(\theta,\gamma)$, with a sample complexity of $\mathcal{\tilde{O}}(\epsilon^{-2.5})$ in the case of both C
    
[^8]: 利用点扩散模型对大肠的3D形状进行精化以生成数字幻影

    Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation. (arXiv:2309.08289v1 [cs.CV])

    [http://arxiv.org/abs/2309.08289](http://arxiv.org/abs/2309.08289)

    本研究利用几何深度学习和去噪扩散概率模型优化大肠的分割结果，并结合先进的表面重构模型，实现对大肠3D形状的精化恢复。

    

    准确建模人体器官在构建虚拟成像试验的计算仿真中起着至关重要的作用。然而，从计算机断层扫描中生成解剖学上可信的器官表面重建仍然对人体结构中的许多器官来说是个挑战。在处理大肠时，这个挑战尤为明显。在这项研究中，我们利用几何深度学习和去噪扩散概率模型的最新进展来优化大肠分割结果。首先，我们将器官表示为从3D分割掩模表面采样得到的点云。随后，我们使用分层变分自编码器获得器官形状的全局和局部潜在表示。我们在分层潜在空间中训练两个条件去噪扩散模型来进行形状精化。为了进一步提高我们的方法，我们还结合了一种先进的表面重构模型，从而实现形状的更好恢复。

    Accurate 3D modeling of human organs plays a crucial role in building computational phantoms for virtual imaging trials. However, generating anatomically plausible reconstructions of organ surfaces from computed tomography scans remains challenging for many structures in the human body. This challenge is particularly evident when dealing with the large intestine. In this study, we leverage recent advancements in geometric deep learning and denoising diffusion probabilistic models to refine the segmentation results of the large intestine. We begin by representing the organ as point clouds sampled from the surface of the 3D segmentation mask. Subsequently, we employ a hierarchical variational autoencoder to obtain global and local latent representations of the organ's shape. We train two conditional denoising diffusion models in the hierarchical latent space to perform shape refinement. To further enhance our method, we incorporate a state-of-the-art surface reconstruction model, allowin
    
[^9]: 标记随机块模型中的最优簇恢复问题

    Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model. (arXiv:2306.12968v1 [cs.SI])

    [http://arxiv.org/abs/2306.12968](http://arxiv.org/abs/2306.12968)

    本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。

    

    本文考虑在有限数量的簇的情况下，用标记随机块模型（LSBM）恢复隐藏的社群，其中簇大小随着物品总数$n$的增长而线性增长。在LSBM中，为每对物品（独立地）观测到一个标签。我们的目标是设计一种有效的算法，利用观测到的标签来恢复簇。为此，我们重新审视了关于期望被任何聚类算法误分类的物品数量的实例特定下界。我们提出了实例自适应聚类（IAC），这是第一个在期望和高概率下都能匹配这些下界表现的算法。IAC由一次谱聚类算法和一个迭代的基于似然的簇分配改进组成。这种方法基于实例特定的下界，不需要任何模型参数，包括簇的数量。通过仅执行一次谱聚类，IAC在计算和存储方面都是高效的。

    We consider the problem of recovering hidden communities in the Labeled Stochastic Block Model (LSBM) with a finite number of clusters, where cluster sizes grow linearly with the total number $n$ of items. In the LSBM, a label is (independently) observed for each pair of items. Our objective is to devise an efficient algorithm that recovers clusters using the observed labels. To this end, we revisit instance-specific lower bounds on the expected number of misclassified items satisfied by any clustering algorithm. We present Instance-Adaptive Clustering (IAC), the first algorithm whose performance matches these lower bounds both in expectation and with high probability. IAC consists of a one-time spectral clustering algorithm followed by an iterative likelihood-based cluster assignment improvement. This approach is based on the instance-specific lower bound and does not require any model parameters, including the number of clusters. By performing the spectral clustering only once, IAC m
    
[^10]: 用Johnson-Lindenstrauss矩阵进行标签嵌入

    Label Embedding by Johnson-Lindenstrauss Matrices. (arXiv:2305.19470v1 [cs.LG])

    [http://arxiv.org/abs/2305.19470](http://arxiv.org/abs/2305.19470)

    这篇论文提出了基于JLMs的标签嵌入方法，将多元分类问题转化为有限回归问题，具有较高的计算效率和预测准确性。

    

    我们提出了一个基于Johnson-Lindenstrauss矩阵（JLMs）的简单且可扩展的极端多元分类框架。利用JLM的列来嵌入标签，将一个C类分类问题转化为具有$\cO(\log C)$输出维度的回归问题。我们得出了一个超量风险限制，阐明了计算效率和预测准确性之间的权衡，并进一步表明，在Massart噪声条件下，降维的惩罚会消失。我们的方法易于并行化，并且实验结果展示了在大规模应用中其有效性和可扩展性。

    We present a simple and scalable framework for extreme multiclass classification based on Johnson-Lindenstrauss matrices (JLMs). Using the columns of a JLM to embed the labels, a $C$-class classification problem is transformed into a regression problem with $\cO(\log C)$ output dimension. We derive an excess risk bound, revealing a tradeoff between computational efficiency and prediction accuracy, and further show that under the Massart noise condition, the penalty for dimension reduction vanishes. Our approach is easily parallelizable, and experimental results demonstrate its effectiveness and scalability in large-scale applications.
    

