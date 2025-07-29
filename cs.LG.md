# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the rates of convergence for learning with convolutional neural networks](https://arxiv.org/abs/2403.16459) | 该研究提出了对具有一定权重约束的CNNs的新逼近上界，以及对前馈神经网络的覆盖数做了新的分析，为基于CNNs的学习问题推导了收敛速率，并在学习平滑函数和二元分类方面取得了极小最优的结果。 |
| [^2] | [GLC++: Source-Free Universal Domain Adaptation through Global-Local Clustering and Contrastive Affinity Learning](https://arxiv.org/abs/2403.14410) | 该论文提出了GLC++方法，通过全局和局部聚类以及对比关联学习实现了无源通用域自适应，能够准确分类已知数据并将其从未知数据中分离。 |
| [^3] | [The Effect of Data Poisoning on Counterfactual Explanations](https://arxiv.org/abs/2402.08290) | 本研究研究了反事实解释在数据污染方面的脆弱性，发现最先进的反事实生成方法和工具包容易受到数据污染的影响。 |
| [^4] | [Uncertainty-Aware Testing-Time Optimization for 3D Human Pose Estimation](https://arxiv.org/abs/2402.02339) | 本文提出了一种不确定性感知的测试时间优化（UAO）框架，通过量化关节点的不确定性来缓解过拟合问题，提高3D人体姿势估计的性能。 |
| [^5] | [REDS: Resource-Efficient Deep Subnetworks for Dynamic Resource Constraints](https://arxiv.org/abs/2311.13349) | 本论文提出了一种名为REDS的资源高效深度子网络，通过利用神经元的排列不变性和新颖的迭代背包优化器来实现模型在不同资源约束下的自适应性，并通过优化计算块和重新安排操作顺序等方法提高计算效率。 |
| [^6] | [Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task.](http://arxiv.org/abs/2310.09336) | 组合能力以乘法方式出现：研究了条件扩散模型在合成任务中的组合泛化能力，结果显示这种能力受到底层数据生成过程的结构影响，且模型在学习到更高级的组合时存在困难。 |
| [^7] | [Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization.](http://arxiv.org/abs/2309.10370) | 本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。 |
| [^8] | [Learning High-Dimensional Nonparametric Differential Equations via Multivariate Occupation Kernel Functions.](http://arxiv.org/abs/2306.10189) | 本论文提出了一种线性方法，通过多元占位核函数在高维状态空间中学习非参数ODE系统，可以解决显式公式按二次方缩放的问题。这种方法在高度非线性的数据和图像数据中都具有通用性。 |
| [^9] | [Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning.](http://arxiv.org/abs/2305.15612) | 该论文提出了一种基于密度比估计和半监督学习的贝叶斯优化方法，通过使用监督分类器代替密度比来估计全局最优解的两组数据的类别概率，避免了分类器对全局解决方案过于自信的问题。 |

# 详细

[^1]: 关于使用卷积神经网络进行学习收敛速率的研究

    On the rates of convergence for learning with convolutional neural networks

    [https://arxiv.org/abs/2403.16459](https://arxiv.org/abs/2403.16459)

    该研究提出了对具有一定权重约束的CNNs的新逼近上界，以及对前馈神经网络的覆盖数做了新的分析，为基于CNNs的学习问题推导了收敛速率，并在学习平滑函数和二元分类方面取得了极小最优的结果。

    

    我们研究了卷积神经网络（CNNs）的逼近和学习能力。第一个结果证明了在权重上有一定约束条件下CNNs的新逼近上界。第二个结果给出了对前馈神经网络的覆盖数的新分析，其中CNNs是其特例。该分析详细考虑了权重的大小，在某些情况下给出了比现有文献更好的上界。利用这两个结果，我们能够推导基于CNNs的估计器在许多学习问题中的收敛速率。特别地，我们在非参数回归设置中为基于CNNs的最小二乘学习平滑函数建立了极小最优的收敛速率。对于二元分类，我们推导了具有铰链损失和逻辑损失的CNN分类器的收敛速度。同时还表明所得到的速率在几种情况下是极小最优的。

    arXiv:2403.16459v1 Announce Type: new  Abstract: We study the approximation and learning capacities of convolutional neural networks (CNNs). Our first result proves a new approximation bound for CNNs with certain constraint on the weights. Our second result gives a new analysis on the covering number of feed-forward neural networks, which include CNNs as special cases. The analysis carefully takes into account the size of the weights and hence gives better bounds than existing literature in some situations. Using these two results, we are able to derive rates of convergence for estimators based on CNNs in many learning problems. In particular, we establish minimax optimal convergence rates of the least squares based on CNNs for learning smooth functions in the nonparametric regression setting. For binary classification, we derive convergence rates for CNN classifiers with hinge loss and logistic loss. It is also shown that the obtained rates are minimax optimal in several settings.
    
[^2]: GLC++: 全局局部聚类和对比关联学习的无源通用域自适应

    GLC++: Source-Free Universal Domain Adaptation through Global-Local Clustering and Contrastive Affinity Learning

    [https://arxiv.org/abs/2403.14410](https://arxiv.org/abs/2403.14410)

    该论文提出了GLC++方法，通过全局和局部聚类以及对比关联学习实现了无源通用域自适应，能够准确分类已知数据并将其从未知数据中分离。

    

    深度神经网络经常在协变量和类别转移下表现出次优性能。无源域自适应（SFDA）为这一困境提供了一个有希望的解决方案，然而大多数SFDA方法局限于封闭集场景。在本文中，我们探讨了旨在准确分类属于常见类别的“已知”数据并将其与目标专有“未知”数据隔离开来的无源通用域自适应（SF-UniDA）。我们提出了一种新颖的全球和局部聚类（GLC）技术，其中包括自适应的一对全局聚类算法来区分目标类别，辅以本地k-NN聚类策略以减轻负面转移。尽管有效，但固有的封闭源架构导致对“未知”数据的统一处理，阻碍了对不同“未知”类别的识别。为了解决这个问题，我们将GLC发展到GLC++，整合了对比亲和性。

    arXiv:2403.14410v1 Announce Type: cross  Abstract: Deep neural networks often exhibit sub-optimal performance under covariate and category shifts. Source-Free Domain Adaptation (SFDA) presents a promising solution to this dilemma, yet most SFDA approaches are restricted to closed-set scenarios. In this paper, we explore Source-Free Universal Domain Adaptation (SF-UniDA) aiming to accurately classify "known" data belonging to common categories and segregate them from target-private "unknown" data. We propose a novel Global and Local Clustering (GLC) technique, which comprises an adaptive one-vs-all global clustering algorithm to discern between target classes, complemented by a local k-NN clustering strategy to mitigate negative transfer. Despite the effectiveness, the inherent closed-set source architecture leads to uniform treatment of "unknown" data, impeding the identification of distinct "unknown" categories. To address this, we evolve GLC to GLC++, integrating a contrastive affini
    
[^3]: 数据污染对反事实解释的影响

    The Effect of Data Poisoning on Counterfactual Explanations

    [https://arxiv.org/abs/2402.08290](https://arxiv.org/abs/2402.08290)

    本研究研究了反事实解释在数据污染方面的脆弱性，发现最先进的反事实生成方法和工具包容易受到数据污染的影响。

    

    反事实解释是分析黑盒系统预测的一种流行方法，它们提供了根据不同情况建议改变输入以获得不同（更有利）系统输出的计算补救机会。然而，最近的研究突显了它们对不同类型操纵的脆弱性。本研究研究了反事实解释对数据污染的脆弱性。我们在增加三个不同层次的补救成本方面，形式化地研究了反事实解释在单个实例、某个子组或所有实例上的数据污染。我们证明了最先进的反事实生成方法和工具包对此类数据污染是脆弱的。

    Counterfactual explanations provide a popular method for analyzing the predictions of black-box systems, and they can offer the opportunity for computational recourse by suggesting actionable changes on how to change the input to obtain a different (i.e. more favorable) system output. However, recent work highlighted their vulnerability to different types of manipulations. This work studies the vulnerability of counterfactual explanations to data poisoning. We formalize data poisoning in the context of counterfactual explanations for increasing the cost of recourse on three different levels: locally for a single instance, or a sub-group of instances, or globally for all instances. We demonstrate that state-of-the-art counterfactual generation methods \& toolboxes are vulnerable to such data poisoning.
    
[^4]: 不确定性感知的3D人体姿势估计测试时间优化

    Uncertainty-Aware Testing-Time Optimization for 3D Human Pose Estimation

    [https://arxiv.org/abs/2402.02339](https://arxiv.org/abs/2402.02339)

    本文提出了一种不确定性感知的测试时间优化（UAO）框架，通过量化关节点的不确定性来缓解过拟合问题，提高3D人体姿势估计的性能。

    

    尽管数据驱动方法在3D人体姿势估计方面取得了成功，但它们常常受到域间差异的限制，表现出有限的泛化能力。相比之下，基于优化的方法在特定情况下进行微调方面表现优秀，但整体表现通常不如数据驱动方法。我们观察到先前的基于优化的方法通常依赖于投影约束，这仅仅确保了在2D空间中的对齐，可能导致过拟合问题。为了解决这个问题，我们提出了一种不确定性感知的测试时间优化 (UAO) 框架，它保留了预训练模型的先验信息，并利用关节点的不确定性来缓解过拟合问题。具体而言，在训练阶段，我们设计了一个有效的2D到3D网络，用于估计相应的3D姿势，并量化每个3D关节点的不确定性。对于测试时的优化，所提出的优化框架冻结预训练模型，并仅优化少量关键参数，以提高性能。

    Although data-driven methods have achieved success in 3D human pose estimation, they often suffer from domain gaps and exhibit limited generalization. In contrast, optimization-based methods excel in fine-tuning for specific cases but are generally inferior to data-driven methods in overall performance. We observe that previous optimization-based methods commonly rely on projection constraint, which only ensures alignment in 2D space, potentially leading to the overfitting problem. To address this, we propose an Uncertainty-Aware testing-time Optimization (UAO) framework, which keeps the prior information of pre-trained model and alleviates the overfitting problem using the uncertainty of joints. Specifically, during the training phase, we design an effective 2D-to-3D network for estimating the corresponding 3D pose while quantifying the uncertainty of each 3D joint. For optimization during testing, the proposed optimization framework freezes the pre-trained model and optimizes only a 
    
[^5]: REDS: 资源高效的深度子网络用于动态资源约束

    REDS: Resource-Efficient Deep Subnetworks for Dynamic Resource Constraints

    [https://arxiv.org/abs/2311.13349](https://arxiv.org/abs/2311.13349)

    本论文提出了一种名为REDS的资源高效深度子网络，通过利用神经元的排列不变性和新颖的迭代背包优化器来实现模型在不同资源约束下的自适应性，并通过优化计算块和重新安排操作顺序等方法提高计算效率。

    

    部署在边缘设备上的深度模型经常遇到资源变化，这源于能量水平波动、时间约束或系统中其他关键任务的优先级。目前的机器学习流水线生成的是资源不可知的模型，并不能在运行时进行调整。在这项工作中，我们引入了Resource-Efficient Deep Subnetworks (REDS)来应对可变资源下的模型适应性。与最先进技术相比，REDS利用结构化稀疏性，通过利用神经元的排列不变性，从而允许硬件特定的优化。具体来说，REDS通过（1）跳过由新颖的迭代背包优化器识别的顺序计算块，以及（2）利用简单的数学重新安排REDS计算图中操作的顺序，以利用数据缓存而实现计算效率。REDS支持传统的深度网络频率。

    arXiv:2311.13349v2 Announce Type: replace  Abstract: Deep models deployed on edge devices frequently encounter resource variability, which arises from fluctuating energy levels, timing constraints, or prioritization of other critical tasks within the system. State-of-the-art machine learning pipelines generate resource-agnostic models, not capable to adapt at runtime. In this work we introduce Resource-Efficient Deep Subnetworks (REDS) to tackle model adaptation to variable resources. In contrast to the state-of-the-art, REDS use structured sparsity constructively by exploiting permutation invariance of neurons, which allows for hardware-specific optimizations. Specifically, REDS achieve computational efficiency by (1) skipping sequential computational blocks identified by a novel iterative knapsack optimizer, and (2) leveraging simple math to re-arrange the order of operations in REDS computational graph to take advantage of the data cache. REDS support conventional deep networks freq
    
[^6]: 组合能力以乘法方式出现：在合成任务中探索扩散模型

    Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task. (arXiv:2310.09336v1 [cs.LG])

    [http://arxiv.org/abs/2310.09336](http://arxiv.org/abs/2310.09336)

    组合能力以乘法方式出现：研究了条件扩散模型在合成任务中的组合泛化能力，结果显示这种能力受到底层数据生成过程的结构影响，且模型在学习到更高级的组合时存在困难。

    

    现代生成模型展示出了产生极为逼真数据的前所未有的能力。然而，考虑到现实世界的自然组合性，这些模型在实际应用中可靠使用需要展示出能够组合新的概念集合以生成训练数据集中未见的输出的能力。先前的研究表明，最近的扩散模型确实表现出了有趣的组合泛化能力，但它们也会出现无法预测的失败。受此启发，我们在合成环境中进行了有控制性的研究，以了解条件扩散模型的组合泛化能力，我们变化了训练数据的不同属性并测量了模型生成越界样本的能力。我们的结果显示：（i）从一个概念生成样本的能力和将它们组合起来的能力的出现顺序受到了底层数据生成过程的结构的影响；（ii）在组合任务上的表现表明模型在学习到更高级的组合时存在困难。

    Modern generative models exhibit unprecedented capabilities to generate extremely realistic data. However, given the inherent compositionality of the real world, reliable use of these models in practical applications requires that they exhibit the capability to compose a novel set of concepts to generate outputs not seen in the training data set. Prior work demonstrates that recent diffusion models do exhibit intriguing compositional generalization abilities, but also fail unpredictably. Motivated by this, we perform a controlled study for understanding compositional generalization in conditional diffusion models in a synthetic setting, varying different attributes of the training data and measuring the model's ability to generate samples out-of-distribution. Our results show: (i) the order in which the ability to generate samples from a concept and compose them emerges is governed by the structure of the underlying data-generating process; (ii) performance on compositional tasks exhib
    
[^7]: 浅层神经网络的几何结构和基于${\mathcal L}^2$代价最小化的构造方法

    Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization. (arXiv:2309.10370v1 [cs.LG])

    [http://arxiv.org/abs/2309.10370](http://arxiv.org/abs/2309.10370)

    本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。

    

    本文给出了一个几何解释：浅层神经网络的结构由一个隐藏层、一个斜坡激活函数、一个${\mathcal L}^2$谱范类（或者Hilbert-Schmidt）的代价函数、输入空间${\mathbb R}^M$、输出空间${\mathbb R}^Q$（其中$Q\leq M$），以及训练输入样本数量$N>QM$所特征。我们证明了代价函数的最小值具有$O(\delta_P)$的上界，其中$\delta_P$衡量了训练输入的信噪比。我们使用适应于属于同一输出向量$y_j$的训练输入向量$\overline{x_{0,j}}$的投影来获得近似的优化器，其中$j=1,\dots,Q$。在特殊情况$M=Q$下，我们明确确定了代价函数的一个确切退化局部最小值；这个尖锐的值与对于$Q\leq M$所获得的上界之间有一个相对误差$O(\delta_P^2)$。上界证明的方法提供了一个构造性训练的网络；我们证明它测度了$Q$维空间中的给定输出。

    In this paper, we provide a geometric interpretation of the structure of shallow neural networks characterized by one hidden layer, a ramp activation function, an ${\mathcal L}^2$ Schatten class (or Hilbert-Schmidt) cost function, input space ${\mathbb R}^M$, output space ${\mathbb R}^Q$ with $Q\leq M$, and training input sample size $N>QM$. We prove an upper bound on the minimum of the cost function of order $O(\delta_P$ where $\delta_P$ measures the signal to noise ratio of training inputs. We obtain an approximate optimizer using projections adapted to the averages $\overline{x_{0,j}}$ of training input vectors belonging to the same output vector $y_j$, $j=1,\dots,Q$. In the special case $M=Q$, we explicitly determine an exact degenerate local minimum of the cost function; the sharp value differs from the upper bound obtained for $Q\leq M$ by a relative error $O(\delta_P^2)$. The proof of the upper bound yields a constructively trained network; we show that it metrizes the $Q$-dimen
    
[^8]: 通过多元占位核函数学习高维非参数微分方程

    Learning High-Dimensional Nonparametric Differential Equations via Multivariate Occupation Kernel Functions. (arXiv:2306.10189v1 [stat.ML])

    [http://arxiv.org/abs/2306.10189](http://arxiv.org/abs/2306.10189)

    本论文提出了一种线性方法，通过多元占位核函数在高维状态空间中学习非参数ODE系统，可以解决显式公式按二次方缩放的问题。这种方法在高度非线性的数据和图像数据中都具有通用性。

    

    从$d$维状态空间中$n$个轨迹快照中学习非参数的常微分方程（ODE）系统需要学习$d$个函数。除非具有额外的系统属性知识，例如稀疏性和对称性，否则显式的公式按二次方缩放。在这项工作中，我们提出了一种使用向量值再生核希尔伯特空间提供的隐式公式学习的线性方法。通过将ODE重写为更弱的积分形式，我们随后进行最小化并推导出我们的学习算法。最小化问题的解向量场依赖于与解轨迹相关的多元占位核函数。我们通过对高度非线性的模拟和真实数据进行实验证实了我们的方法，其中$d$可能超过100。我们进一步通过从图像数据学习非参数一阶拟线性偏微分方程展示了所提出的方法的多样性。

    Learning a nonparametric system of ordinary differential equations (ODEs) from $n$ trajectory snapshots in a $d$-dimensional state space requires learning $d$ functions of $d$ variables. Explicit formulations scale quadratically in $d$ unless additional knowledge about system properties, such as sparsity and symmetries, is available. In this work, we propose a linear approach to learning using the implicit formulation provided by vector-valued Reproducing Kernel Hilbert Spaces. By rewriting the ODEs in a weaker integral form, which we subsequently minimize, we derive our learning algorithm. The minimization problem's solution for the vector field relies on multivariate occupation kernel functions associated with the solution trajectories. We validate our approach through experiments on highly nonlinear simulated and real data, where $d$ may exceed 100. We further demonstrate the versatility of the proposed method by learning a nonparametric first order quasilinear partial differential 
    
[^9]: 基于密度比估计的半监督学习贝叶斯优化

    Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning. (arXiv:2305.15612v1 [cs.LG])

    [http://arxiv.org/abs/2305.15612](http://arxiv.org/abs/2305.15612)

    该论文提出了一种基于密度比估计和半监督学习的贝叶斯优化方法，通过使用监督分类器代替密度比来估计全局最优解的两组数据的类别概率，避免了分类器对全局解决方案过于自信的问题。

    

    贝叶斯优化在科学与工程的多个领域受到了广泛关注，因为它能高效地找到昂贵黑盒函数的全局最优解。通常，一个概率回归模型，如高斯过程、随机森林和贝叶斯神经网络，被广泛用作替代函数，用于模拟在给定输入和训练数据集的情况下函数评估的显式分布。除了基于概率回归的贝叶斯优化，基于密度比估计的贝叶斯优化已被提出来估计相对于全局最优解相对接近和相对远离的两组密度比。为了进一步发展这一研究，可以使用监督分类器来估计这两组的类别概率，而不是密度比。然而，此策略中使用的监督分类器倾向于对全局解决方案过于自信。

    Bayesian optimization has attracted huge attention from diverse research areas in science and engineering, since it is capable of finding a global optimum of an expensive-to-evaluate black-box function efficiently. In general, a probabilistic regression model, e.g., Gaussian processes, random forests, and Bayesian neural networks, is widely used as a surrogate function to model an explicit distribution over function evaluations given an input to estimate and a training dataset. Beyond the probabilistic regression-based Bayesian optimization, density ratio estimation-based Bayesian optimization has been suggested in order to estimate a density ratio of the groups relatively close and relatively far to a global optimum. Developing this line of research further, a supervised classifier can be employed to estimate a class probability for the two groups instead of a density ratio. However, the supervised classifiers used in this strategy tend to be overconfident for a global solution candid
    

