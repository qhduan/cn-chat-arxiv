# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Contrastive Approach to Prior Free Positive Unlabeled Learning](https://arxiv.org/abs/2402.06038) | 该论文提出了一种免先验正无标学习的对比方法，通过预训练不变表示学习特征空间并利用嵌入的浓度特性对未标记样本进行伪标签处理，相比现有方法，在多个标准数据集上表现优异，同时不需要先验知识或类先验的估计。 |
| [^2] | [Debiased Sample Selection for Combating Noisy Labels.](http://arxiv.org/abs/2401.13360) | 本文提出了一个无噪声专家模型（ITEM）来解决样本选择中的训练偏差和数据偏差问题。通过设计一个鲁棒的网络架构来集成多个专家，可以减少选择集不平衡和累积错误，并在使用更少参数的情况下实现更好的选择和预测性能。 |
| [^3] | [DCSI -- An improved measure of cluster separability based on separation and connectedness.](http://arxiv.org/abs/2310.12806) | 这篇论文提出了一种改进的聚类可分离性度量方法，旨在量化类间分离和类内连通性，对于密度聚类具有较好的性能表现。 |
| [^4] | [A Theory of Non-Linear Feature Learning with One Gradient Step in Two-Layer Neural Networks.](http://arxiv.org/abs/2310.07891) | 这篇论文提出了一种关于两层神经网络中非线性特征学习的理论。通过一步梯度下降训练的过程中引入不同的多项式特征，该方法能够学习到目标函数的非线性组件，而更新的神经网络的性能则由这些特征所决定。 |
| [^5] | [Learning force laws in many-body systems.](http://arxiv.org/abs/2310.05273) | 在这篇论文中，作者展示了一种结合了物理直觉的机器学习方法，用于推断尘埃等离子体实验中的力学规律。通过对粒子轨迹的训练，该模型考虑了对称性和非相同粒子之间的非互逆力，并提取了每个粒子的质量和电荷。模型的准确性指示出尘埃等离子体中存在超出当前理论分辨率的新物理，并展示了机器学习在引导多体系统科学发现方面的潜力。 |
| [^6] | [Structure and Gradient Dynamics Near Global Minima of Two-layer Neural Networks.](http://arxiv.org/abs/2309.00508) | 本论文通过分析两层神经网络在全局最小值附近的结构和梯度动力学，揭示了其泛化能力较强的原因。 |
| [^7] | [Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge.](http://arxiv.org/abs/2307.08813) | 本研究评估了不同大型语言模型在提取分子相互作用和通路知识方面的有效性，并讨论了未来机遇和挑战。 |
| [^8] | [Deep Generative Models for Physiological Signals: A Systematic Literature Review.](http://arxiv.org/abs/2307.06162) | 本文是对深度生成模型在生理信号研究领域的系统综述，总结了最新最先进的研究进展，有助于了解这些模型在生理信号中的应用和挑战，同时提供了评估和基准测试的指导。 |
| [^9] | [Absorbing Phase Transitions in Artificial Deep Neural Networks.](http://arxiv.org/abs/2307.02284) | 本文研究了在适当初始化的有限神经网络中的吸收相变及其普适性，证明了即使在有限网络中仍然存在着从有序状态到混沌状态的过渡，并且不同的网络架构会反映在过渡的普适类上。 |
| [^10] | [Cooperation Is All You Need.](http://arxiv.org/abs/2305.10449) | 引入了一种基于“本地处理器民主”的算法Cooperator，该算法在强化学习中表现比Transformer算法更好。 |
| [^11] | [Policy Gradient Converges to the Globally Optimal Policy for Nearly Linear-Quadratic Regulators.](http://arxiv.org/abs/2303.08431) | 本论文研究了强化学习方法在几乎线性二次型调节器系统中找到最优策略的问题，提出了一个策略梯度算法，可以以线性速率收敛于全局最优解。 |
| [^12] | [Bandit Social Learning: Exploration under Myopic Behavior.](http://arxiv.org/abs/2302.07425) | 该论文研究了自私行为下的劫匪社交学习问题，发现存在一种探索激励权衡，即武器探索和社交探索之间的权衡，受到代理的短视行为的限制会加剧这种权衡，并导致遗憾率与代理数量成线性关系。 |
| [^13] | [Designing Universal Causal Deep Learning Models: The Case of Infinite-Dimensional Dynamical Systems from Stochastic Analysis.](http://arxiv.org/abs/2210.13300) | 设计了一个DL模型框架，名为因果神经算子（CNO），以逼近因果算子（CO），并证明了CNO模型可以在紧致集上一致逼近Hölder或平滑迹类算子。 |

# 详细

[^1]: 免先验正无标（Positive Unlabeled）学习的对比方法

    Contrastive Approach to Prior Free Positive Unlabeled Learning

    [https://arxiv.org/abs/2402.06038](https://arxiv.org/abs/2402.06038)

    该论文提出了一种免先验正无标学习的对比方法，通过预训练不变表示学习特征空间并利用嵌入的浓度特性对未标记样本进行伪标签处理，相比现有方法，在多个标准数据集上表现优异，同时不需要先验知识或类先验的估计。

    

    正无标（Positive Unlabeled）学习是指在给定少量标记的正样本和一组未标记样本（可能是正例或负例）的情况下学习一个二分类器的任务。在本文中，我们提出了一种新颖的正无标学习框架，通过保证不变表示学习学习特征空间，并利用嵌入的浓度特性对未标记样本进行伪标签处理。总体而言，我们提出的方法在多个标准正无标基准数据集上轻松超越了现有的正无标学习方法，而不需要先验知识或类先验的估计。值得注意的是，我们的方法在标记数据稀缺的情况下仍然有效，而大多数正无标学习算法则失败。我们还提供了简单的理论分析来推动我们提出的算法，并为我们的方法建立了一般化保证。

    Positive Unlabeled (PU) learning refers to the task of learning a binary classifier given a few labeled positive samples, and a set of unlabeled samples (which could be positive or negative). In this paper, we propose a novel PU learning framework, that starts by learning a feature space through pretext-invariant representation learning and then applies pseudo-labeling to the unlabeled examples, leveraging the concentration property of the embeddings. Overall, our proposed approach handily outperforms state-of-the-art PU learning methods across several standard PU benchmark datasets, while not requiring a-priori knowledge or estimate of class prior. Remarkably, our method remains effective even when labeled data is scant, where most PU learning algorithms falter. We also provide simple theoretical analysis motivating our proposed algorithms and establish generalization guarantee for our approach.
    
[^2]: 对抗噪声标签的无偏样本选择

    Debiased Sample Selection for Combating Noisy Labels. (arXiv:2401.13360v1 [cs.LG])

    [http://arxiv.org/abs/2401.13360](http://arxiv.org/abs/2401.13360)

    本文提出了一个无噪声专家模型（ITEM）来解决样本选择中的训练偏差和数据偏差问题。通过设计一个鲁棒的网络架构来集成多个专家，可以减少选择集不平衡和累积错误，并在使用更少参数的情况下实现更好的选择和预测性能。

    

    学习使用噪声标签旨在确保模型在标签错误的训练集上具有泛化能力。样本选择策略通过选择可靠的标签子集来实现有希望的性能。本文实证表明，现有的样本选择方法在实践中存在数据和训练偏差，分别表示为选择集不平衡和累积错误。然而，先前的研究只处理了训练偏差。为了解决这个局限性，我们提出了一个适用于样本选择的无噪声专家模型（ITEM）。具体来说，为了减轻训练偏差，我们设计了一个鲁棒的网络架构，与多个专家集成。与目前的双分支网络相比，我们的网络在训练更少参数的情况下，通过集成这些专家来实现更好的选择和预测性能。同时，为了减轻数据偏差，我们提出了一种混合采样策略。

    Learning with noisy labels aims to ensure model generalization given a label-corrupted training set. The sample selection strategy achieves promising performance by selecting a label-reliable subset for model training. In this paper, we empirically reveal that existing sample selection methods suffer from both data and training bias that are represented as imbalanced selected sets and accumulation errors in practice, respectively. However, only the training bias was handled in previous studies. To address this limitation, we propose a noIse-Tolerant Expert Model (ITEM) for debiased learning in sample selection. Specifically, to mitigate the training bias, we design a robust network architecture that integrates with multiple experts. Compared with the prevailing double-branch network, our network exhibits better performance of selection and prediction by ensembling these experts while training with fewer parameters. Meanwhile, to mitigate the data bias, we propose a mixed sampling strat
    
[^3]: DCSI -- 基于分离和连通性的改进的聚类可分离性度量

    DCSI -- An improved measure of cluster separability based on separation and connectedness. (arXiv:2310.12806v1 [stat.ML])

    [http://arxiv.org/abs/2310.12806](http://arxiv.org/abs/2310.12806)

    这篇论文提出了一种改进的聚类可分离性度量方法，旨在量化类间分离和类内连通性，对于密度聚类具有较好的性能表现。

    

    确定给定数据集中的类别标签是否对应于有意义的聚类对于使用真实数据集评估聚类算法至关重要。这个特性可以通过可分离性度量来量化。现有文献的综述显示，既有的基于分类的复杂性度量方法和聚类有效性指标 (CVIs) 都没有充分融入基于密度的聚类的核心特征：类间分离和类内连通性。一种新开发的度量方法 (密度聚类可分离性指数, DCSI) 旨在量化这两个特征，并且也可用作 CVI。对合成数据的广泛实验表明，DCSI 与通过调整兰德指数 (ARI) 测量的DBSCAN的性能之间有很强的相关性，但在对多类数据集进行密度聚类不适当的重叠类别时缺乏鲁棒性。对经常使用的真实数据集进行详细评估显示，DCSI 能够更好地区分密度聚类的可分离性。

    Whether class labels in a given data set correspond to meaningful clusters is crucial for the evaluation of clustering algorithms using real-world data sets. This property can be quantified by separability measures. A review of the existing literature shows that neither classification-based complexity measures nor cluster validity indices (CVIs) adequately incorporate the central aspects of separability for density-based clustering: between-class separation and within-class connectedness. A newly developed measure (density cluster separability index, DCSI) aims to quantify these two characteristics and can also be used as a CVI. Extensive experiments on synthetic data indicate that DCSI correlates strongly with the performance of DBSCAN measured via the adjusted rand index (ARI) but lacks robustness when it comes to multi-class data sets with overlapping classes that are ill-suited for density-based hard clustering. Detailed evaluation on frequently used real-world data sets shows that
    
[^4]: 两层神经网络中一次梯度下降的非线性特征学习理论

    A Theory of Non-Linear Feature Learning with One Gradient Step in Two-Layer Neural Networks. (arXiv:2310.07891v1 [stat.ML])

    [http://arxiv.org/abs/2310.07891](http://arxiv.org/abs/2310.07891)

    这篇论文提出了一种关于两层神经网络中非线性特征学习的理论。通过一步梯度下降训练的过程中引入不同的多项式特征，该方法能够学习到目标函数的非线性组件，而更新的神经网络的性能则由这些特征所决定。

    

    特征学习被认为是深度神经网络成功的基本原因之一。在特定条件下已经严格证明，在两层全连接神经网络中，第一层进行一步梯度下降，然后在第二层进行岭回归可以导致特征学习；特征矩阵的谱中会出现分离的一维组件，称为“spike”。然而，使用固定梯度下降步长时，这个“spike”仅提供了目标函数的线性组件的信息，因此学习非线性组件是不可能的。我们展示了当学习率随样本大小增长时，这样的训练实际上引入了多个一维组件，每个组件对应一个特定的多项式特征。我们进一步证明了更新的神经网络的极限大维度和大样本训练和测试误差完全由这些“spike”所决定。

    Feature learning is thought to be one of the fundamental reasons for the success of deep neural networks. It is rigorously known that in two-layer fully-connected neural networks under certain conditions, one step of gradient descent on the first layer followed by ridge regression on the second layer can lead to feature learning; characterized by the appearance of a separated rank-one component -- spike -- in the spectrum of the feature matrix. However, with a constant gradient descent step size, this spike only carries information from the linear component of the target function and therefore learning non-linear components is impossible. We show that with a learning rate that grows with the sample size, such training in fact introduces multiple rank-one components, each corresponding to a specific polynomial feature. We further prove that the limiting large-dimensional and large sample training and test errors of the updated neural networks are fully characterized by these spikes. By 
    
[^5]: 在多体系统中学习力学规律

    Learning force laws in many-body systems. (arXiv:2310.05273v1 [physics.plasm-ph] CROSS LISTED)

    [http://arxiv.org/abs/2310.05273](http://arxiv.org/abs/2310.05273)

    在这篇论文中，作者展示了一种结合了物理直觉的机器学习方法，用于推断尘埃等离子体实验中的力学规律。通过对粒子轨迹的训练，该模型考虑了对称性和非相同粒子之间的非互逆力，并提取了每个粒子的质量和电荷。模型的准确性指示出尘埃等离子体中存在超出当前理论分辨率的新物理，并展示了机器学习在引导多体系统科学发现方面的潜力。

    

    描述自然系统的科学规律可能比我们的直觉更复杂，因此我们发现规律的方法必须改变。机器学习（ML）模型可以分析大量数据，但其结构应该符合基本的物理约束条件以提供有用的见解。在这里，我们展示了一种结合了物理直觉的ML方法，以推断尘埃等离子体实验中的力学法则。通过对3D粒子轨迹进行训练，该模型考虑了固有的对称性和非相同粒子之间的有效非互逆力，并提取出每个粒子的质量和电荷。模型的准确性（R^2 > 0.99）指示出尘埃等离子体中超出当前理论分辨率的新物理，并展示了机器学习驱动的方法如何引导多体系统中的科学发现的新途径。

    Scientific laws describing natural systems may be more complex than our intuition can handle, and thus how we discover laws must change. Machine learning (ML) models can analyze large quantities of data, but their structure should match the underlying physical constraints to provide useful insight. Here we demonstrate a ML approach that incorporates such physical intuition to infer force laws in dusty plasma experiments. Trained on 3D particle trajectories, the model accounts for inherent symmetries and non-identical particles, accurately learns the effective non-reciprocal forces between particles, and extracts each particle's mass and charge. The model's accuracy (R^2 > 0.99) points to new physics in dusty plasma beyond the resolution of current theories and demonstrates how ML-powered approaches can guide new routes of scientific discovery in many-body systems.
    
[^6]: 两层神经网络全局最小值附近的结构和梯度动力学

    Structure and Gradient Dynamics Near Global Minima of Two-layer Neural Networks. (arXiv:2309.00508v1 [cs.LG])

    [http://arxiv.org/abs/2309.00508](http://arxiv.org/abs/2309.00508)

    本论文通过分析两层神经网络在全局最小值附近的结构和梯度动力学，揭示了其泛化能力较强的原因。

    

    在温和的假设下，我们研究了两层神经网络在全局最小值附近的损失函数表面的结构，确定了能够实现完美泛化的参数集，并完整描述了其周围的梯度流动态。通过新颖的技术，我们揭示了复杂的损失函数表面的一些简单方面，并揭示了模型、目标函数、样本和初始化对训练动力学的不同影响。基于这些结果，我们还解释了为什么（过度参数化的）神经网络可以很好地泛化。

    Under mild assumptions, we investigate the structure of loss landscape of two-layer neural networks near global minima, determine the set of parameters which give perfect generalization, and fully characterize the gradient flows around it. With novel techniques, our work uncovers some simple aspects of the complicated loss landscape and reveals how model, target function, samples and initialization affect the training dynamics differently. Based on these results, we also explain why (overparametrized) neural networks could generalize well.
    
[^7]: 大型语言模型在提取分子相互作用和通路知识方面的比较性能评估

    Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge. (arXiv:2307.08813v1 [cs.CL])

    [http://arxiv.org/abs/2307.08813](http://arxiv.org/abs/2307.08813)

    本研究评估了不同大型语言模型在提取分子相互作用和通路知识方面的有效性，并讨论了未来机遇和挑战。

    

    理解蛋白质相互作用和通路知识对于揭示生物系统的复杂性和研究生物功能和复杂疾病的基本机制至关重要。尽管现有的数据库提供了来自文献和其他源的策划生物数据，但它们往往不完整且维护工作繁重，因此需要替代方法。在本研究中，我们提出利用大型语言模型的能力，通过自动从相关科学文献中提取这些知识来解决这些问题。为了实现这个目标，在这项工作中，我们调查了不同大型语言模型在识别蛋白质相互作用、通路和基因调控关系等任务中的有效性。我们对不同模型的性能进行了彻底评估，突出了重要的发现，并讨论了这种方法所面临的未来机遇和挑战。代码和数据集链接可在论文中找到。

    Understanding protein interactions and pathway knowledge is crucial for unraveling the complexities of living systems and investigating the underlying mechanisms of biological functions and complex diseases. While existing databases provide curated biological data from literature and other sources, they are often incomplete and their maintenance is labor-intensive, necessitating alternative approaches. In this study, we propose to harness the capabilities of large language models to address these issues by automatically extracting such knowledge from the relevant scientific literature. Toward this goal, in this work, we investigate the effectiveness of different large language models in tasks that involve recognizing protein interactions, pathways, and gene regulatory relations. We thoroughly evaluate the performance of various models, highlight the significant findings, and discuss both the future opportunities and the remaining challenges associated with this approach. The code and d
    
[^8]: 深度生成模型对生理信号的系统文献综述

    Deep Generative Models for Physiological Signals: A Systematic Literature Review. (arXiv:2307.06162v1 [cs.LG])

    [http://arxiv.org/abs/2307.06162](http://arxiv.org/abs/2307.06162)

    本文是对深度生成模型在生理信号研究领域的系统综述，总结了最新最先进的研究进展，有助于了解这些模型在生理信号中的应用和挑战，同时提供了评估和基准测试的指导。

    

    本文对深度生成模型在生理信号，特别是心电图、脑电图、光电容抗图和肌电图领域的文献进行了系统综述。与已有的综述文章相比，本文是第一篇总结最新最先进的深度生成模型的综述。通过分析与深度生成模型相关的最新研究，以及这些模型的主要应用和挑战，本综述为对这些模型应用于生理信号的整体理解做出了贡献。此外，通过强调采用的评估协议和最常用的生理数据库，本综述有助于对深度生成模型进行评估和基准测试。

    In this paper, we present a systematic literature review on deep generative models for physiological signals, particularly electrocardiogram, electroencephalogram, photoplethysmogram and electromyogram. Compared to the existing review papers, we present the first review that summarizes the recent state-of-the-art deep generative models. By analysing the state-of-the-art research related to deep generative models along with their main applications and challenges, this review contributes to the overall understanding of these models applied to physiological signals. Additionally, by highlighting the employed evaluation protocol and the most used physiological databases, this review facilitates the assessment and benchmarking of deep generative models.
    
[^9]: 人工深度神经网络中的吸收相变

    Absorbing Phase Transitions in Artificial Deep Neural Networks. (arXiv:2307.02284v1 [stat.ML])

    [http://arxiv.org/abs/2307.02284](http://arxiv.org/abs/2307.02284)

    本文研究了在适当初始化的有限神经网络中的吸收相变及其普适性，证明了即使在有限网络中仍然存在着从有序状态到混沌状态的过渡，并且不同的网络架构会反映在过渡的普适类上。

    

    由于著名的平均场理论，对于各种体系的无限宽度神经网络的行为的理论理解已经迅速发展。然而，对于更实际和现实重要性更强的有限网络，缺乏清晰直观的框架来延伸我们的理解。在本文中，我们展示了适当初始化的神经网络的行为可以用吸收相变中的普遍临界现象来理解。具体而言，我们研究了全连接前馈神经网络和卷积神经网络中从有序状态到混沌状态的相变，并强调了体系架构的差异与相变的普适类之间的关系。值得注意的是，我们还成功地应用了有限尺度扩展的方法，这表明了直观的现象学。

    Theoretical understanding of the behavior of infinitely-wide neural networks has been rapidly developed for various architectures due to the celebrated mean-field theory. However, there is a lack of a clear, intuitive framework for extending our understanding to finite networks that are of more practical and realistic importance. In the present contribution, we demonstrate that the behavior of properly initialized neural networks can be understood in terms of universal critical phenomena in absorbing phase transitions. More specifically, we study the order-to-chaos transition in the fully-connected feedforward neural networks and the convolutional ones to show that (i) there is a well-defined transition from the ordered state to the chaotics state even for the finite networks, and (ii) difference in architecture is reflected in that of the universality class of the transition. Remarkably, the finite-size scaling can also be successfully applied, indicating that intuitive phenomenologic
    
[^10]: 合作是你所需要的。 （arXiv:2305.10449v1 [cs.LG]）

    Cooperation Is All You Need. (arXiv:2305.10449v1 [cs.LG])

    [http://arxiv.org/abs/2305.10449](http://arxiv.org/abs/2305.10449)

    引入了一种基于“本地处理器民主”的算法Cooperator，该算法在强化学习中表现比Transformer算法更好。

    

    在超越“树突民主”之上，我们引入了一个名为Cooperator的“本地处理器民主”。在这里，我们将它们与基于Transformers的机器学习算法（例如ChatGPT）在置换不变神经网络强化学习（RL）中的功能进行比较。 Transformers基于长期以来的“积分-发射”“点”神经元的概念，而Cooperator则受到最近神经生物学突破的启示，这些突破表明，精神生活的细胞基础取决于新皮层中具有两个功能上不同点的上皮神经元。我们表明，当用于RL时，基于Cooperator的算法学习速度比基于Transformer的算法快得多，即使它们具有相同数量的参数。

    Going beyond 'dendritic democracy', we introduce a 'democracy of local processors', termed Cooperator. Here we compare their capabilities when used in permutation-invariant neural networks for reinforcement learning (RL), with machine learning algorithms based on Transformers, such as ChatGPT. Transformers are based on the long-standing conception of integrate-and-fire 'point' neurons, whereas Cooperator is inspired by recent neurobiological breakthroughs suggesting that the cellular foundations of mental life depend on context-sensitive pyramidal neurons in the neocortex which have two functionally distinct points. We show that when used for RL, an algorithm based on Cooperator learns far quicker than that based on Transformer, even while having the same number of parameters.
    
[^11]: 政策梯度算法收敛于几乎线性二次型调节器的全局最优策略

    Policy Gradient Converges to the Globally Optimal Policy for Nearly Linear-Quadratic Regulators. (arXiv:2303.08431v1 [cs.LG])

    [http://arxiv.org/abs/2303.08431](http://arxiv.org/abs/2303.08431)

    本论文研究了强化学习方法在几乎线性二次型调节器系统中找到最优策略的问题，提出了一个策略梯度算法，可以以线性速率收敛于全局最优解。

    

    决策者只获得了非完整信息的非线性控制系统在各种应用中普遍存在。本研究探索了强化学习方法，以找到几乎线性二次型调节器系统中最优策略。我们考虑一个动态系统，结合线性和非线性组成部分，并由相同结构的策略进行管理。在假设非线性组成部分包含具有小型Lipschitz系数的内核的情况下，我们对成本函数的优化进行了表征。虽然成本函数通常是非凸的，但我们确立了全局最优解附近局部的强凸性和光滑性。此外，我们提出了一种初始化机制，以利用这些属性。在此基础上，我们设计了一个策略梯度算法，可以保证以线性速率收敛于全局最优解。

    Nonlinear control systems with partial information to the decision maker are prevalent in a variety of applications. As a step toward studying such nonlinear systems, this work explores reinforcement learning methods for finding the optimal policy in the nearly linear-quadratic regulator systems. In particular, we consider a dynamic system that combines linear and nonlinear components, and is governed by a policy with the same structure. Assuming that the nonlinear component comprises kernels with small Lipschitz coefficients, we characterize the optimization landscape of the cost function. Although the cost function is nonconvex in general, we establish the local strong convexity and smoothness in the vicinity of the global optimizer. Additionally, we propose an initialization mechanism to leverage these properties. Building on the developments, we design a policy gradient algorithm that is guaranteed to converge to the globally optimal policy with a linear rate.
    
[^12]: 自私行为下的劫匪社交学习

    Bandit Social Learning: Exploration under Myopic Behavior. (arXiv:2302.07425v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2302.07425](http://arxiv.org/abs/2302.07425)

    该论文研究了自私行为下的劫匪社交学习问题，发现存在一种探索激励权衡，即武器探索和社交探索之间的权衡，受到代理的短视行为的限制会加剧这种权衡，并导致遗憾率与代理数量成线性关系。

    

    我们研究了一种社交学习动态，其中代理按照简单的多臂劫匪协议共同行动。代理以顺序方式到达，选择武器并接收相关奖励。每个代理观察先前代理的完整历史记录（武器和奖励），不存在私有信号。尽管代理共同面临开发和利用的探索折衷，但每个代理人都是一见钟情的，无需考虑探索。我们允许一系列与（参数化）置信区间一致的自私行为，包括“无偏”行为和各种行为偏差。虽然这些行为的极端版本对应于众所周知的劫匪算法，但我们证明了更温和的版本会导致明显的探索失败，因此遗憾率与代理数量成线性关系。我们通过分析“温和乐观”的代理提供匹配的遗憾上界。因此，我们建立了两种类型的探索激励之间的基本权衡：武器探索是固有于劫匪问题的，只受当前代理的行动影响，而社交探索是由先前代理行为驱动的，因此有利于未来代理。由于代理的短视行为限制了社交探索，这种权衡被加剧。

    We study social learning dynamics where the agents collectively follow a simple multi-armed bandit protocol. Agents arrive sequentially, choose arms and receive associated rewards. Each agent observes the full history (arms and rewards) of the previous agents, and there are no private signals. While collectively the agents face exploration-exploitation tradeoff, each agent acts myopically, without regards to exploration. Motivating scenarios concern reviews and ratings on online platforms.  We allow a wide range of myopic behaviors that are consistent with (parameterized) confidence intervals, including the "unbiased" behavior as well as various behaviorial biases. While extreme versions of these behaviors correspond to well-known bandit algorithms, we prove that more moderate versions lead to stark exploration failures, and consequently to regret rates that are linear in the number of agents. We provide matching upper bounds on regret by analyzing "moderately optimistic" agents.  As a
    
[^13]: 设计通用因果深度学习模型：以随机分析中的无限维动态系统为例

    Designing Universal Causal Deep Learning Models: The Case of Infinite-Dimensional Dynamical Systems from Stochastic Analysis. (arXiv:2210.13300v2 [math.DS] UPDATED)

    [http://arxiv.org/abs/2210.13300](http://arxiv.org/abs/2210.13300)

    设计了一个DL模型框架，名为因果神经算子（CNO），以逼近因果算子（CO），并证明了CNO模型可以在紧致集上一致逼近Hölder或平滑迹类算子。

    

    因果算子（CO）在当代随机分析中扮演着重要角色，例如各种随机微分方程的解算子。然而，目前还没有一个能够逼近CO的深度学习（DL）模型的规范框架。本文通过引入一个DL模型设计框架来提出一个“几何感知”的解决方案，该框架以合适的无限维线性度量空间为输入，并返回适应这些线性几何的通用连续序列DL模型。我们称这些模型为因果神经算子（CNO）。我们的主要结果表明，我们的框架所产生的模型可以在紧致集上和跨任意有限时间视野上一致逼近Hölder或平滑迹类算子，这些算子因果地映射给定线性度量空间之间的序列。我们的分析揭示了关于CNO的潜在状态空间维度的新定量关系，甚至对于（经典的）有限维DL模型也有新的影响。

    Causal operators (CO), such as various solution operators to stochastic differential equations, play a central role in contemporary stochastic analysis; however, there is still no canonical framework for designing Deep Learning (DL) models capable of approximating COs. This paper proposes a "geometry-aware'" solution to this open problem by introducing a DL model-design framework that takes suitable infinite-dimensional linear metric spaces as inputs and returns a universal sequential DL model adapted to these linear geometries. We call these models Causal Neural Operators (CNOs). Our main result states that the models produced by our framework can uniformly approximate on compact sets and across arbitrarily finite-time horizons H\"older or smooth trace class operators, which causally map sequences between given linear metric spaces. Our analysis uncovers new quantitative relationships on the latent state-space dimension of CNOs which even have new implications for (classical) finite-d
    

