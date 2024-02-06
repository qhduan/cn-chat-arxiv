# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization](https://rss.arxiv.org/abs/2402.01401) | 通过Lipschitz正则化实现零样本机器遗忘，可以及时忘记私人或受版权保护的信息，同时保持模型性能。 |
| [^2] | [Characterizing Overfitting in Kernel Ridgeless Regression Through the Eigenspectrum](https://rss.arxiv.org/abs/2402.01297) | 我们通过推导核矩阵的特征数界限，增强了核岭回归的测试误差界限。对于多项式谱衰减的核，我们恢复了先前的结果；对于指数谱衰减，我们提出了新的非平凡的界限。我们的研究表明，特征谱衰减多项式的核回归器具有良好的泛化能力，而特征谱指数衰减的核回归器则具有灾难性的过拟合。 |
| [^3] | [Ginger: An Efficient Curvature Approximation with Linear Complexity for General Neural Networks](https://arxiv.org/abs/2402.03295) | Ginger是一种用于通用神经网络的高效曲率近似方法，具有线性复杂度。它通过特征分解来逆向计算广义高斯牛顿矩阵，避免了传统方法中的高内存和高时间复杂度问题。 |
| [^4] | [Flora: Low-Rank Adapters Are Secretly Gradient Compressors](https://arxiv.org/abs/2402.03293) | 本文研究了低秩适配器的动力学，并提出了一种基于随机投影的方法Flora，通过重新采样投影矩阵实现高秩更新，同时减少优化状态的空间复杂度。 |
| [^5] | [A Framework for Partially Observed Reward-States in RLHF](https://arxiv.org/abs/2402.03282) | 这篇论文提出了一个针对RLHF的框架，在其中考虑了部分观察到的奖励状态，并通过将基数反馈和决斗反馈缩减为PORRL形式进行了建模和算法开发。 |
| [^6] | [Learning Best-in-Class Policies for the Predict-then-Optimize Framework](https://arxiv.org/abs/2402.03256) | 我们提出了一种新颖的决策感知替代损失函数家族，用于predict-then-optimize框架，并且通过数值证据证实了其在误设置下的优越性。 |
| [^7] | [Minimum Description Length and Generalization Guarantees for Representation Learning](https://arxiv.org/abs/2402.03254) | 本文提出了一个可压缩性框架，通过计算表示学习算法的泛化误差的上界，改进了现有启发式方法，并提供了关于理论泛化保证的新见解。 |
| [^8] | [The Benefits of Reusing Batches for Gradient Descent in Two-Layer Networks: Breaking the Curse of Information and Leap Exponents](https://arxiv.org/abs/2402.03220) | 该论文研究了在两层神经网络中学习多指数目标函数时，重复使用批次的梯度下降（GD）的训练动态。研究发现，与单次GD相比，多次GD能够克服目标函数的限制，仅需两个时间步骤就能实现网络与目标子空间的重叠，展示了在有限时间内有效学习的广泛函数类。这些结果基于动力平均场理论（DMFT）的分析。 |
| [^9] | [A Random Matrix Approach to Low-Multilinear-Rank Tensor Approximation](https://arxiv.org/abs/2402.03169) | 该研究采用随机矩阵方法，在低多线性秩张量逼近中展示了对种植的低秩信号的估计，并根据大维谱行为和信噪比准确预测了重建性能，并给出了HOOI收敛的充分条件。 |
| [^10] | [Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity](https://arxiv.org/abs/2402.03167) | 本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。 |
| [^11] | [A Multi-step Loss Function for Robust Learning of the Dynamics in Model-based Reinforcement Learning](https://arxiv.org/abs/2402.03146) | 本文提出了一种用于稳健学习模型预测的多步损失函数，通过加权平均均方误差损失在不同未来时间点上来训练单步模型。这种损失函数在存在噪声的情况下尤为有效，并在各种任务中取得了显著的预测改善。 |
| [^12] | [How Free is Parameter-Free Stochastic Optimization?](https://arxiv.org/abs/2402.03126) | 这个论文研究了无参随机优化的问题，提出了一种完全无参的方法，通过简单的超参数搜索技术在非凸和凸设置下都能取得优于先进算法的性能。同时，论文还建立了一个下界，指出完全无参的方法在某些情况下无法实现。 |
| [^13] | [High-dimensional Bayesian Optimization via Covariance Matrix Adaptation Strategy](https://arxiv.org/abs/2402.03104) | 本文提出了一种高维贝叶斯优化方法，通过协方差矩阵适应策略定义局部搜索区域，能够解决将贝叶斯优化应用于高维优化问题的挑战。 |
| [^14] | [Diffusive Gibbs Sampling](https://arxiv.org/abs/2402.03008) | 扩散吉布斯采样是一种创新的采样方法，通过集成扩散模型并应用吉布斯采样，有效地从具有远程和断开模态特征的分布中采样，表现出比其他方法更好的混合性能，并在多种任务中取得显著改进的结果。 |
| [^15] | [On the development of a practical Bayesian optimisation algorithm for expensive experiments and simulations with changing environmental conditions](https://arxiv.org/abs/2402.03006) | 本文在受环境条件变化影响的昂贵实验和模拟中，将贝叶斯优化方法推广到包含可控和不可控参数的系统优化中，通过在所有变量上拟合全局代理模型，但只在对不可控变量的测量条件下优化可控参数。 |
| [^16] | [Careful with that Scalpel: Improving Gradient Surgery with an EMA](https://arxiv.org/abs/2402.02998) | 通过将训练损失梯度和辅助梯度在训练梯度方向上的正交投影结合起来，使用EMA（指数移动平均）可以改进梯度手术，提高深度学习估计管道的性能。 |
| [^17] | [Boosting, Voting Classifiers and Randomized Sample Compression Schemes](https://arxiv.org/abs/2402.02976) | 本研究提出了一种随机提升算法来解决传统提升算法的性能问题，并通过构建一个通用框架将样本压缩方法扩展到支持随机学习算法，实现了在样本大小上具有单对数依赖的泛化错误。 |
| [^18] | [Towards Understanding the Word Sensitivity of Attention Layers: A Study via Random Features](https://arxiv.org/abs/2402.02969) | 通过研究随机特征，我们发现注意力层具有较高的词敏感性，这对于理解transformers的成功以及自然语言处理任务中的上下文含义非常重要。 |
| [^19] | [On Least Squares Estimation in Softmax Gating Mixture of Experts](https://arxiv.org/abs/2402.02952) | 本研究探讨了在确定性MoE模型下使用最小二乘估计器的性能，并建立了强可识别性条件来描述不同类型专家函数的收敛行为。 |
| [^20] | [Dynamic Byzantine-Robust Learning: Adapting to Switching Byzantine Workers](https://arxiv.org/abs/2402.02951) | $\textsf{DynaBRO}$是一种动态拜占庭-强鲁棒学习的方法，能够适应切换拜占庭工作机制，并且在渐近收敛速率上与静态情况相匹配。通过多级蒙特卡洛渐变估计技术、强鲁棒工作机制更新的聚合和故障安全过滤器的引入，我们的方法能够经受住$\mathcal{O}(\sqrt{T})$轮拜占庭身份的改变。另外，通过使用自适应学习率，我们的方法消除了对百分比的需求。 |
| [^21] | [Bayesian Federated Inference for regression models with heterogeneous multi-center populations](https://arxiv.org/abs/2402.02898) | 这项研究提出了一种利用贝叶斯联合推断方法，在不同中心分别分析本地数据，并将统计推断结果组合起来，以解决样本量不足的问题，并准确估计回归模型的参数。 |
| [^22] | [Graph Neural Machine: A New Model for Learning with Tabular Data](https://arxiv.org/abs/2402.02862) | 本论文提出了一种新的机器学习模型，图神经机器（GNM），用于处理表格数据。GNM使用同步消息传递方案，并用几乎完全图代替了多层感知机（MLP）的有向无环图。实验结果表明，在多个数据集上，GNM模型的性能优于MLP架构。 |
| [^23] | [Leveraging Noisy Observations in Zero-Sum Games](https://arxiv.org/abs/2402.02861) | 本文研究了在零和游戏中利用嘈杂观察的优势。具体而言，通过对给定概率测度进行采样，领导者承诺对手选择行动，而追随者根据其当前信息选择行动。证明了带有嘈杂行动可观测性的博弈总是存在均衡，且鉴定了其唯一性的必要条件。此外，嘈杂观察对追随者的最佳响应集合的基数有重要影响。 |
| [^24] | [Importance sampling for online variational learning](https://arxiv.org/abs/2402.02859) | 本文提出了一个在状态空间模型中用于在线学习平滑分布的高效算法，并在离线和真正的在线设置中展示了性能。 |
| [^25] | [Deep autoregressive density nets vs neural ensembles for model-based offline reinforcement learning](https://arxiv.org/abs/2402.02858) | 本文对比了在基于模型的离线强化学习中，使用深度自回归密度网络和神经集合的方法。通过在D4RL基准测试上展示，我们质疑了使用神经集合的普遍观点，并发现单个良好校准的自回归模型可以获得更好的性能。同时，我们还分析了模型学习的静态指标，并得出了关于代理最终性能的重要模型特性。 |
| [^26] | [Non-asymptotic Analysis of Biased Adaptive Stochastic Approximation](https://arxiv.org/abs/2402.02857) | 本文对于具有偏态梯度和自适应步长的SGD进行了全面的非渐进分析，证明了Adagrad和RMSProp算法在收敛速度上与无偏情况相似，并通过实验结果验证了收敛结果，展示了如何降低偏差的影响。 |
| [^27] | [Enhancing Compositional Generalization via Compositional Feature Alignment](https://arxiv.org/abs/2402.02851) | 通过组合特征对齐，增强了模型的组合通用性，使其能够推广到未见过的领域-类别组合。 |
| [^28] | [Bayes-Optimal Fair Classification with Linear Disparity Constraints via Pre-, In-, and Post-processing](https://arxiv.org/abs/2402.02817) | 本文提出了一种基于贝叶斯最优的公平分类方法，通过先处理、中处理和后处理来最小化分类错误，并在给定群体公平性约束的情况下进行优化。该方法引入了线性和双线性差异度量的概念，并找到了贝叶斯最优公平分类器的形式。本方法能够处理多个公平性约束和常见情况。 |
| [^29] | [Glocal Hypergradient Estimation with Koopman Operator](https://arxiv.org/abs/2402.02741) | 本文提出了一种具有Koopman算子的全局超梯度估计方法，通过使用局部超梯度的轨迹来高效地近似全局超梯度，实现了超参数的贪婪优化，兼具可靠性和效率。 |
| [^30] | [InVA: Integrative Variational Autoencoder for Harmonization of Multi-modal Neuroimaging Data](https://arxiv.org/abs/2402.02734) | InVA是一种综合变分自编码器方法，利用多模态神经影像数据中不同来源的多个图像来进行预测推理，相较于传统的VAE方法具有更好的效果。 |
| [^31] | [Discounted Adaptive Online Prediction](https://arxiv.org/abs/2402.02720) | 本论文提出了一种折扣自适应在线预测算法，该算法适应于复杂的损失序列和比较器，并改进了非自适应算法。算法具有无需结构性假设的理论保证，并且在超参数调整方面具有鲁棒性。通过在线符合预测任务的实验证明了算法的好处。 |
| [^32] | [Understanding What Affects Generalization Gap in Visual Reinforcement Learning: Theory and Empirical Evidence](https://arxiv.org/abs/2402.02701) | 本文通过理论和实证研究，揭示了在测试环境具有干扰因素时影响视觉强化学习中泛化差距的关键因素。结果表明，最小化训练和测试环境之间的表示距离是减少泛化差距最关键的因素。 |
| [^33] | [Sample Complexity Characterization for Linear Contextual MDPs](https://arxiv.org/abs/2402.02700) | 本文研究了线性上下文马尔可夫决策过程（CMDPs）的样本复杂性表征，并提出了两种模型的新颖算法，证明它们具有所需的多项式样本复杂性。其中，对于第一个模型，通过去除可达性假设，改进了现有结果。 |
| [^34] | [Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional Gaussian Mixtures](https://arxiv.org/abs/2402.02697) | 本文通过对深度均衡模型和显式神经网络模型进行理论分析和实验证明，在高维高斯混合数据下，可以通过设计浅显式网络来实现与给定深度均衡模型相同的特征光谱行为。 |
| [^35] | [Statistical Guarantees for Link Prediction using Graph Neural Networks](https://arxiv.org/abs/2402.02692) | 本文提出了一种线性图神经网络（LG-GNN）架构，通过计算边缘概率来预测图中的链接，并推导了其在链接预测任务中的性能统计保证。这种架构对于稀疏和稠密图都适用，并在真实和合成数据集上验证了其优势。 |
| [^36] | [Counterfactual Fairness Is Not Demographic Parity, and Other Observations](https://arxiv.org/abs/2402.02663) | 这里是中文总结出的一句话要点：文章探讨了因果概念与纯粹概率概念之间的等价性，并发现计算上的公正并不等同于人口统计数据的平等。同时还纠正了一些有关计算上的公正的误解。 |
| [^37] | [Variational DAG Estimation via State Augmentation With Stochastic Permutations](https://arxiv.org/abs/2402.02644) | 使用状态扩展和随机排列进行变分DAG估计的方法可以超越竞争的贝叶斯和非贝叶斯基准方法，从而在估计贝叶斯网络结构方面取得更好的性能。 |
| [^38] | [$C^*$-Algebraic Machine Learning: Moving in a New Direction](https://arxiv.org/abs/2402.02637) | $C^*$-代数机器学习是将$C^*$-代数与机器学习结合的新研究方向，它通过统一现有的学习策略，并构建更多元化和信息丰富的数据模型的新框架，为机器学习提供了一种新的方法。 |
| [^39] | [A new approach for imprecise probabilities](https://arxiv.org/abs/2402.02556) | 本论文引入了一种新的区间概率测度的概念，用于表示不确定性概率。通过特征化一类广泛的区间概率测度，建立了更新规则，并提出了随机优势的定义。此外，还给出了凯恩斯-兰姆齐争论的正式解决方案。 |
| [^40] | [A Fast Method for Lasso and Logistic Lasso](https://arxiv.org/abs/2402.02463) | 本论文提出了一种快速解决Lasso和Logistic Lasso问题的方法，通过采用主动集方法和适当的求解器，成功实现了加速。在压缩感知、Lasso回归和Logistic Lasso回归实验中，与传统方法相比，我们的方法平均能提高约30倍的速度。 |
| [^41] | [On Minimum Trace Factor Analysis - An Old Song Sung to a New Tune](https://arxiv.org/abs/2402.02459) | 本文提出了最小化迹因子分析（MTFA）的放松版本，该方法能够有效降低因异方差噪声造成的过拟合问题，并解决了在因子分析和谱方法中常见的异常情况和病态诅咒问题。 |
| [^42] | [FreDF: Learning to Forecast in Frequency Domain](https://arxiv.org/abs/2402.02399) | FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。 |
| [^43] | [Stereographic Spherical Sliced Wasserstein Distances](https://arxiv.org/abs/2402.02345) | 本文提出了一种快速且高度并行的用于比较球形测度的距离，使用了立体投影和广义Radon变换，称之为立体投影球面切片瓦瑟斯坦（S3W）距离。通过仔细处理立体投影引起的距离畸变，并进行了理论分析，证明了该方法在速度和效果上的优势。 |
| [^44] | [A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding](https://arxiv.org/abs/2402.02306) | 本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。 |
| [^45] | [Goodness-of-Fit and Clustering of Spherical Data: the QuadratiK package in R and Python](https://arxiv.org/abs/2402.02290) | QuadratiK软件包是一个在R和Python中实现的数据分析工具，它提供了一套全面的拟合度测试和基于核方法的聚类技术，特别适用于处理球形数据。 |
| [^46] | [Future Directions in Foundations of Graph Machine Learning](https://arxiv.org/abs/2402.02287) | 图机器学习领域的未来方向应该是发展一个更加均衡的理论，从更完整的角度探究图神经网络的表达能力、泛化和优化之间的相互关系。 |
| [^47] | [Causal Bayesian Optimization via Exogenous Distribution Learning](https://arxiv.org/abs/2402.02277) | 本文引入了一种新的方法，通过学习外源变量的分布，提高了结构化因果模型的近似精度，并将因果贝叶斯优化扩展到更一般的因果方案。 |
| [^48] | [Characterization of the Distortion-Perception Tradeoff for Finite Channels with Arbitrary Metrics](https://arxiv.org/abs/2402.02265) | 本研究中，我们对于具有任意度量的有限通道中的失真感知权衡进行了研究，发现计算失真感知函数和最优重建相当于求解一组线性规划问题，并提供了失真感知权衡的结构特征化和二进制源的闭式表达式。 |
| [^49] | [Distributional Reduction: Unifying Dimensionality Reduction and Clustering with Gromov-Wasserstein Projection](https://arxiv.org/abs/2402.02239) | 本文提出了一种新的分布约简方法，利用格罗莫夫-瓦瑟斯坦投影统一了降维和聚类，通过优化问题同时解决降维和聚类，实验证明了该方法在多个领域表现出卓越性能。 |
| [^50] | [Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems](https://arxiv.org/abs/2402.02190) | 本研究提出了连续张量放松方法(CTRA)，用于在组合优化问题中寻找多样化的解决方案。CTRA通过对离散决策变量进行连续放松，解决了寻找多样化解决方案的挑战。 |
| [^51] | [Off-Policy Evaluation of Slate Bandit Policies via Optimizing Abstraction](https://arxiv.org/abs/2402.02171) | 我们提出了一种名为潜在IPS（LIPS）的新的Slate Bandit OPE估计器，通过在低维度的Slate抽象空间中定义重要性权重，并通过数据驱动的方式优化Slate抽象来减小偏差和方差。 |
| [^52] | [A Bayesian cluster validity index](https://arxiv.org/abs/2402.02162) | 该论文提出了一个基于贝叶斯方法的聚类有效性指数，该指数根据现有的基础指数定义，并用于检测次优聚类数，通过与其他指数进行比较，验证了其有效性。 |
| [^53] | [Position Paper: Why the Shooting in the Dark Method Dominates Recommender Systems Practice; A Call to Abandon Anti-Utopian Thinking](https://arxiv.org/abs/2402.02152) | 这篇论文质疑了推荐系统实践中目前常用的“摸着石头过河”方法，呼吁摒弃反乌托邦思维。论文提出了使用深度学习堆栈的非标准用法，以解锁奖励优化的推荐系统的潜力。 |
| [^54] | [Accelerating Look-ahead in Bayesian Optimization: Multilevel Monte Carlo is All you Need](https://arxiv.org/abs/2402.02111) | 本文利用多层蒙特卡洛方法加速贝叶斯优化中的前瞻过程，并证明在涉及嵌套期望和最大化的问题中具有优势。 |
| [^55] | [Self-attention Networks Localize When QK-eigenspectrum Concentrates](https://arxiv.org/abs/2402.02098) | 本文研究了自注意网络中的注意力定位问题，通过QK特征值谱的集中定位现象来解决不同观点之间的矛盾。 |
| [^56] | [$\alpha$-Divergence Loss Function for Neural Density Ratio Estimation](https://arxiv.org/abs/2402.02041) | 本文提出了一种应用于神经密度比估计的$\alpha$-散度损失函数($\alpha$-Div)，通过简洁实现和稳定优化解决了现有方法中存在的优化问题。实验证明了这种损失函数的稳定性，并提出了对DRE任务的估计准确性的研究，同时给出了样本要求的解决方案。 |
| [^57] | [GenFormer: A Deep-Learning-Based Approach for Generating Multivariate Stochastic Processes](https://arxiv.org/abs/2402.02010) | GenFormer是一种基于深度学习的方法，用于生成多元随机过程。它能保留目标统计特性，包括边际分布，并能在具有挑战性的应用中近似捕捉到其他期望的统计特性。应用于风速数据模拟的实验中，GenFormer模型用于计算风险管理的超越概率。 |
| [^58] | [Combining T-learning and DR-learning: a framework for oracle-efficient estimation of causal contrasts](https://arxiv.org/abs/2402.01972) | 这篇论文介绍了高效插件学习的框架，能够有效估计异质因果对比，并解决了其他学习策略的一些缺点。该框架构建了人口风险函数的高效插件估计器，具有稳定性和鲁棒性。 |
| [^59] | [Distributional Off-policy Evaluation with Bellman Residual Minimization](https://arxiv.org/abs/2402.01900) | 这篇论文研究了使用Bellman残差最小化的方法来解决分布式离线策略评估问题，并提出了一种称为能量Bellman残差最小化（EBRM）的方法来估计返回分布。在可实现性假设下，建立了EBRM估计器的有限样本误差界。 |
| [^60] | [On f-Divergence Principled Domain Adaptation: An Improved Framework](https://arxiv.org/abs/2402.01887) | 本文改进了基于f-散度的无监督领域自适应（UDA）框架，引入了f-领域差异度量指标，并通过去除绝对值函数和引入缩放参数，提出了新的目标误差和样本复杂度界限，从而使得我们能够恢复以前的KL结果，将算法和理论之间的差距缩小，并通过定位技术开发了快速率的泛化界限。实验结果证明了基于f-DD的领域学习算法在流行的UDA基准测试中表现出了卓越的性能。 |
| [^61] | [Challenges in Training PINNs: A Loss Landscape Perspective](https://arxiv.org/abs/2402.01868) | 本文探讨了训练PINNs的挑战，强调了损失函数空间在训练过程中的作用，引入了新颖的二阶优化器NNCG并优化了PINN性能，为训练PINNs提供了有价值的洞见和更强大的优化策略。 |
| [^62] | [What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement](https://arxiv.org/abs/2402.01865) | 本文研究了语言模型更新中的遗忘现象，提出了一种预测上游实例遗忘的方法，以改进重播过程的可控性和解释性。根据预训练实例的预-softmax对数几率分数变化与在线学习实例的相似性，提出了一种部分可解释的预测模型，在BART模型上表现良好但在T5模型上失败。此外，还展示了基于内积的黑盒分类器。 |
| [^63] | [SPDE priors for uncertainty quantification of end-to-end neural data assimilation schemes](https://arxiv.org/abs/2402.01855) | SPDE先验在最优插值中的应用及其与神经网络的联合学习问题，为大规模地球物理数据集的时空插值提供了一种新的方法。 |
| [^64] | [Multi-Armed Bandits with Interference](https://arxiv.org/abs/2402.01845) | 这篇论文研究了在在线平台中与干扰进行的实验。在多臂赌博机问题中，学习者分配不同的臂给每个实验单元，根据单元之间的空间距离和对手选择的匹配函数来决定每个单元在每轮的回报。研究发现，转换政策能够实现最佳的预期遗憾，但任何转换政策都会遭受一定的遗憾现象。 |
| [^65] | [Misspecification uncertainties in near-deterministic regression](https://arxiv.org/abs/2402.01810) | 该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。 |
| [^66] | [DoubleMLDeep: Estimation of Causal Effects with Multimodal Data](https://arxiv.org/abs/2402.01785) | 本文提出了一个利用文本和图像在因果推断和治疗效应估计中的双机器学习框架，并提出了一种生成半合成数据集的方法用于评估因果效应估计的性能。这些方法和架构在半合成数据集上进行了评估，并与标准方法进行了比较，显示了直接使用文本和图像进行因果研究的潜在好处。 |
| [^67] | [Plug-and-Play image restoration with Stochastic deNOising REgularization](https://arxiv.org/abs/2402.01779) | 本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。 |
| [^68] | [Not All Learnable Distribution Classes are Privately Learnable](https://arxiv.org/abs/2402.00267) | 这篇论文证明了一类分布虽然可以在有限样本下以总变差距离进行学习，但却无法在（ε，δ）-差分隐私下学习。 |
| [^69] | [An extended asymmetric sigmoid with Perceptron (SIGTRON) for imbalanced linear classification](https://arxiv.org/abs/2312.16043) | 本文提出了一个新的多项式参数化sigmoid函数(SIGTRON)，并且介绍了其伴随的SIC模型。相比传统的成本敏感学习模型，在给定的训练数据集接近良好平衡的条件下，所提出的SIC模型对于数据集的变化更加适应，并通过创建倾斜的超平面方程来实现。 |
| [^70] | [Analyzing Sharpness-aware Minimization under Overparameterization](https://arxiv.org/abs/2311.17539) | 本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。 |
| [^71] | [Metric Space Magnitude for Evaluating the Diversity of Latent Representations](https://arxiv.org/abs/2311.16054) | 基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。 |
| [^72] | [One Pass Streaming Algorithm for Super Long Token Attention Approximation in Sublinear Space](https://arxiv.org/abs/2311.14652) | 本文研究了在超长上下文下内存效率的问题，提出一种用于超长Token注意力近似的单次流算法，通过构建矩阵$U_1, U_2$加速注意力计算，解决了部署大型语言模型时的计算资源问题。 |
| [^73] | [Assumption-lean and Data-adaptive Post-Prediction Inference](https://arxiv.org/abs/2311.14220) | 这项工作介绍了一种假设简化和数据自适应的后预测推断（POP-Inf）过程，可以有效且有力地基于机器学习预测结果进行统计推断。 |
| [^74] | [Learning Causal Representations from General Environments: Identifiability and Intrinsic Ambiguity](https://arxiv.org/abs/2311.12267) | 该论文研究了从一般环境中学习因果表示的问题，提供了基于这种环境生成的数据的可辨识性结果，并指出了受到围绕节点歧义的限制。同时提出了一个算法可以恢复出地面真实模型 |
| [^75] | [Surprisal Driven $k$-NN for Robust and Interpretable Nonparametric Learning](https://arxiv.org/abs/2311.10246) | 本论文提出了一种基于惊喜性驱动的稳健可解释的k-NN算法，通过使用信息论的角度对传统算法进行新的阐释，实现了在非参数学习中的分类、回归、密度估计和异常检测等任务。 |
| [^76] | [Distributional GFlowNets with Quantile Flows](https://arxiv.org/abs/2302.05793) | 本文提出了一种带分布式量化流的GFlowNets模型，通过将流函数转化为分布，在训练过程中提供更多信息的学习信号。通过量化函数参数化每个边流，我们提出的算法可以学习风险敏感的策略，实现对风险不确定性场景的处理，并在现有基准上取得了显著改进。 |
| [^77] | [Realizable Learning is All You Need](https://arxiv.org/abs/2111.04746) | 可实现学习与无偏学习的等价性是学习理论中的基本现象，我们提出了第一个独立于模型的框架来解释这个等价性，它可以适用于各种设置，并拓展了我们对各种学习情况的理解。 |
| [^78] | [Post-Regularization Confidence Bands for Ordinary Differential Equations](https://arxiv.org/abs/2110.12510) | 本文提出了一种新的方法，通过后正则化置信带来推断未知函数和有噪声数据观测下的普通微分方程的个体调控函数，该方法结合了局部核学习和新的去偏方法，有效解决了建立置信带的挑战性问题。 |
| [^79] | [Multiply Robust Causal Mediation Analysis with Continuous Treatments](https://arxiv.org/abs/2105.09254) | 本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。 |
| [^80] | [A rigorous introduction to linear models](https://arxiv.org/abs/2105.04240) | 本书旨在向读者提供对线性模型及其理论的严格介绍，并总结了线性模型在回归问题中的重要性和应用。 |
| [^81] | [Optimal Clustering from Noisy Binary Feedback](https://arxiv.org/abs/1910.06002) | 本论文研究了通过二进制用户反馈进行聚类的问题，并提出了一种算法来最小化聚类恢复错误率。 |
| [^82] | [Agnostic Sample Compression Schemes for Regression](https://arxiv.org/abs/1810.01864) | 本文在绝对值损失函数为 $\ell_p$ 的不确定回归设置中构建了一种通用的逼近样本压缩方案，对于线性回归可以实现线性维度大小的压缩，对于 $\ell_1$ 和 $\ell_\infty$ 损失函数可以实现线性维度大小的有效完全样本压缩方案；同时，证明了其他 $\ell_p$ 损失函数不存在有限尺寸的完全不可知压缩方案的结果，并提出了开放问题。 |
| [^83] | [Ricci flow-guided autoencoders in learning time-dependent dynamics.](http://arxiv.org/abs/2401.14591) | 利用Ricci流引导的自编码器方法能够学习非线性动力学，尤其是偏微分方程。该方法通过在训练中学习流形，并使用Ricci流使流形潜空间逐步适应动力学的变化，从而获得更好的表示能力。在实验中，我们展示了该方法在具有周期性和随机性的PDE上的应用，并评估了在分布内和外推场景中的误差。 |
| [^84] | [Low-Tubal-Rank Tensor Recovery via Factorized Gradient Descent.](http://arxiv.org/abs/2401.11940) | 本文提出了一种通过分解梯度下降方法解决低胞状秩张量恢复问题的高效方法，该方法通过将大张量分解为两个较小的因子张量，在减少计算成本和存储需求的同时，确保了收敛性。 |
| [^85] | [A general theory for robust clustering via trimmed mean.](http://arxiv.org/abs/2401.05574) | 本文提出了一种通过使用修剪均值类型的中心点估计的混合聚类技术，用于在存在次高斯误差的中心点周围分布的弱初始化条件下产生最优错误标记保证，并且在存在敌对异常值的情况下仍然有效。 |
| [^86] | [SASSL: Enhancing Self-Supervised Learning via Neural Style Transfer.](http://arxiv.org/abs/2312.01187) | SASSL提出了一种基于神经风格迁移的增强技术，通过解耦语义和风格属性，在自监督学习中生成多样化的增强样本，从而提升了图像分类性能。 |
| [^87] | [Unsupervised Federated Learning: A Federated Gradient EM Algorithm for Heterogeneous Mixture Models with Robustness against Adversarial Attacks.](http://arxiv.org/abs/2310.15330) | 本文介绍了一种针对带有异构混合比例的混合模型的无监督学习的新型联邦梯度EM算法，在适用于普通混合模型的全面有限样本理论基础上，对高斯混合模型（GMM）和混合回归（MoRs）进行了具体的估计误差分析。该算法具有适应未知任务相似性、抵抗对少部分数据源的对抗攻击、保护本地数据隐私以及计算和通信效率等关键优势。 |
| [^88] | [A Theory of Non-Linear Feature Learning with One Gradient Step in Two-Layer Neural Networks.](http://arxiv.org/abs/2310.07891) | 这篇论文提出了一种关于两层神经网络中非线性特征学习的理论。通过一步梯度下降训练的过程中引入不同的多项式特征，该方法能够学习到目标函数的非线性组件，而更新的神经网络的性能则由这些特征所决定。 |
| [^89] | [Theoretical Analysis of Robust Overfitting for Wide DNNs: An NTK Approach.](http://arxiv.org/abs/2310.06112) | 本文理论分析了宽深度神经网络的鲁棒过拟合现象，并提出了一种名为Adv-NTK的AT算法来增强神经网络的鲁棒性。 |
| [^90] | [Entropy-MCMC: Sampling from Flat Basins with Ease.](http://arxiv.org/abs/2310.05401) | 本文提出了一种Entropy-MCMC的方法，通过引入一个辅助的引导变量来在平坦盆地中进行采样，以解决深度神经网络后验分布的多模态问题，并证明了该方法的收敛性。 |
| [^91] | [Uncovering hidden geometry in Transformers via disentangling position and context.](http://arxiv.org/abs/2310.04861) | 本文通过分解transformer的隐藏状态，揭示了其在语义理解中的隐含几何结构。 |
| [^92] | [Learning to Scale Logits for Temperature-Conditional GFlowNets.](http://arxiv.org/abs/2310.02823) | 这项研究提出了一种名为LSL-GFN的新型架构设计，可以大大加速温度条件下GFlowNets的训练，从而提高GFlowNets的探索和利用能力。 |
| [^93] | [Controlling Continuous Relaxation for Combinatorial Optimization.](http://arxiv.org/abs/2309.16965) | 本文研究了在相对密集的图上组合优化问题中物理启发的图神经网络（PI-GNN）求解器的表现。通过数值实验，我们发现PI-GNN求解器在学习早期可能陷入所有变量为零的局部解。为了解决这个问题，我们通过控制连续性和离散性提出了一种改进方法。 |
| [^94] | [Unsupervised Contrast-Consistent Ranking with Language Models.](http://arxiv.org/abs/2309.06991) | 无监督的对比一致排序与语言模型，通过训练一个受逻辑约束引导的探测模型，实现在多个语句中始终映射到对比的真-假极点的排序任务。 |
| [^95] | [Everything, Everywhere All in One Evaluation: Using Multiverse Analysis to Evaluate the Influence of Model Design Decisions on Algorithmic Fairness.](http://arxiv.org/abs/2308.16681) | 通过多元宇宙分析评估模型设计决策对算法公平性的影响，可以揭示算法决策系统中设计决策的关键作用。 |
| [^96] | [Extending Path-Dependent NJ-ODEs to Noisy Observations and a Dependent Observation Framework.](http://arxiv.org/abs/2307.13147) | 该论文研究了将路径相关的NJ-ODE方法扩展到具有噪声观测和相关观测框架的问题。研究提出了两种扩展方法，并提供了理论保证和实证示例。 |
| [^97] | [Big Data - Supply Chain Management Framework for Forecasting: Data Preprocessing and Machine Learning Techniques.](http://arxiv.org/abs/2307.12971) | 本文介绍了一种新的大数据-供应链管理框架，通过数据预处理和机器学习技术实现供应链预测，优化操作管理、透明度，并讨论了幻影库存对预测的不利影响。 |
| [^98] | [The Connection Between R-Learning and Inverse-Variance Weighting for Estimation of Heterogeneous Treatment Effects.](http://arxiv.org/abs/2307.09700) | R-Learning在估计条件平均治疗效果时采用了逆变数加权的形式来稳定回归，并简化了偏差项。 |
| [^99] | [Fast Empirical Scenarios.](http://arxiv.org/abs/2307.03927) | 该论文提出了两种快速的经验场景提取算法，一种识别之前未观察到的场景并提供场景的协方差矩阵表示，另一种从已实现的世界状态中选择重要的数据点，并与高阶样本矩一致，这些算法计算效率高且适用于一致的基于场景的建模和高维数值积分。 |
| [^100] | [Optimizing protein fitness using Gibbs sampling with Graph-based Smoothing.](http://arxiv.org/abs/2307.00494) | 使用基于图形平滑的Gibbs采样方法（GGS）优化蛋白质适应性，消除了突变距离的限制，同时提高了搜索效率。该方法在发现高适应性蛋白质方面达到了最先进水平。 |
| [^101] | [Differentially Private Domain Adaptation with Theoretical Guarantees.](http://arxiv.org/abs/2306.08838) | 该论文提出了两种具有差分隐私保障的自适应算法，用于在受隐私约束且有限标记数据条件下，从公开源领域到目标领域进行监督域自适应。该算法能够解决一般的优化问题，并具有有利的理论学习保证。 |
| [^102] | [Analysis and Approximate Inference of Large and Dense Random Kronecker Graphs.](http://arxiv.org/abs/2306.08489) | 本文对大规模密集随机Kronecker图进行了分析和近似推断，提出了“去噪声和求解”元算法，用于近似推断图参数，并具有较低的计算复杂度和性能保证。 |
| [^103] | [Exploiting Observation Bias to Improve Matrix Completion.](http://arxiv.org/abs/2306.04775) | 本研究利用观测偏差来改进矩阵补全问题，提出一个简单的两阶段算法，实现了与对未观测协变量的监督学习性能相当的结果。 |
| [^104] | [Solving NP-hard Min-max Routing Problems as Sequential Generation with Equity Context.](http://arxiv.org/abs/2306.02689) | 本文提出了一个新的深度学习框架Equity-Transformer来解决大规模的最小最大路径问题。该模型利用可扩展的深度学习模型进行顺序决策，并生成考虑公平工作负载的顺序动作。研究显示，Equity-Transformer在两个代表性最小最大路径问题中具有卓越的性能。 |
| [^105] | [On Size-Independent Sample Complexity of ReLU Networks.](http://arxiv.org/abs/2306.01992) | 本文研究了ReLU神经网络的样本复杂度，给出了一个现有方法精细化的结果，实现了无深度依赖性的上界。 |
| [^106] | [Why Clean Generalization and Robust Overfitting Both Happen in Adversarial Training.](http://arxiv.org/abs/2306.01271) | 对抗训练是训练深度神经网络抗击对抗扰动的标准方法, 其学习机制导致干净泛化和强健过拟合现象同时发生。 |
| [^107] | [Laplace-Approximated Neural Additive Models: Improving Interpretability with Bayesian Inference.](http://arxiv.org/abs/2305.16905) | 本文提出了拉普拉斯逼近神经加性模型，该模型从贝叶斯角度考虑加性结构，在恢复的特征交互中提供可信区间，提供可处理的边缘似然估计，可用于执行隐式特征选择并对特征对进行排名。 |
| [^108] | [Neural incomplete factorization: learning preconditioners for the conjugate gradient method.](http://arxiv.org/abs/2305.16368) | 本文提出了一种名为神经不完全分解的新方法，利用自监督训练的图神经网络生成适用于特定问题域的有效预处理器。其通过替换传统手工预处理器显着提高了收敛和计算效率，在合成和真实问题上进行的实验均表现出竞争力。 |
| [^109] | [Relabel Minimal Training Subset to Flip a Prediction.](http://arxiv.org/abs/2305.12809) | 本文利用扩展影响函数提出了一种有效的识别和重新标记最小训练子集的方法，并证明其始终能够成功翻转测试结果，同时还提供了挑战模型预测、评估模型鲁棒性和洞察训练集偏差等多重作用。 |
| [^110] | [Q-malizing flow and infinitesimal density ratio estimation.](http://arxiv.org/abs/2305.11857) | 研究提出了一种可以从一个数据分布P传输到任意访问通过有限样本的Q的流模型。这个模型通过神经ODE模型进行，可以进行无穷小DRE。 |
| [^111] | [Controlling Posterior Collapse by an Inverse Lipschitz Constraint on the Decoder Network.](http://arxiv.org/abs/2304.12770) | 本文提出了一种基于反Lipschitz约束的解码器网络，可以简单明了地控制广泛的VAE模型的后验坍塌程度，并带有具体的理论保证。 |
| [^112] | [On the strong stability of ergodic iterations.](http://arxiv.org/abs/2304.04657) | 本论文研究了迭代随机函数生成的过程的强稳定性，证明了适用于递归映射的温和条件下的强稳定性，并且提供了多个应用及相关领域的新结果。 |

# 详细

[^1]: 通过Lipschitz正则化在规模上实现零样本机器遗忘

    Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization

    [https://rss.arxiv.org/abs/2402.01401](https://rss.arxiv.org/abs/2402.01401)

    通过Lipschitz正则化实现零样本机器遗忘，可以及时忘记私人或受版权保护的信息，同时保持模型性能。

    

    为了遵守人工智能和数据规定，从训练得到的机器学习模型中遗忘私人或受版权保护的信息的需求变得越来越重要。遗忘的关键挑战是及时忘记必要的数据，同时保持模型性能。在这项工作中，我们解决了零样本遗忘的场景，即只有一个经过训练的模型和要遗忘的数据，遗忘算法必须能够移除数据。根据这样定义，现有的最先进的方法是不够的。基于Lipschitz连续性的概念，我们提出了一种方法，通过对样本扰动的输出进行平滑处理来诱导遗忘。我们展示了这种平滑性成功地实现了遗忘，同时保持了总体模型性能。我们对我们的方法进行了广泛的经验评估，包括一系列当代基准测试，验证了我们的方法在严格的零样本约束下达到了最先进的性能。

    To comply with AI and data regulations, the need to forget private or copyrighted information from trained machine learning models is increasingly important. The key challenge in unlearning is forgetting the necessary data in a timely manner, while preserving model performance. In this work, we address the zero-shot unlearning scenario, whereby an unlearning algorithm must be able to remove data given only a trained model and the data to be forgotten. Under such a definition, existing state-of-the-art methods are insufficient. Building on the concepts of Lipschitz continuity, we present a method that induces smoothing of the forget sample's output, with respect to perturbations of that sample. We show this smoothing successfully results in forgetting while preserving general model performance. We perform extensive empirical evaluation of our method over a range of contemporary benchmarks, verifying that our method achieves state-of-the-art performance under the strict constraints of ze
    
[^2]: 通过特征谱表征核岭回归的过拟合

    Characterizing Overfitting in Kernel Ridgeless Regression Through the Eigenspectrum

    [https://rss.arxiv.org/abs/2402.01297](https://rss.arxiv.org/abs/2402.01297)

    我们通过推导核矩阵的特征数界限，增强了核岭回归的测试误差界限。对于多项式谱衰减的核，我们恢复了先前的结果；对于指数谱衰减，我们提出了新的非平凡的界限。我们的研究表明，特征谱衰减多项式的核回归器具有良好的泛化能力，而特征谱指数衰减的核回归器则具有灾难性的过拟合。

    

    我们推导了核矩阵的条件数的新界限，然后利用这些界限增强了在固定输入维度的过参数化区域中核岭回归的现有非渐近测试误差界限。对于具有多项式谱衰减的核，我们恢复了先前工作的界限；对于指数衰减，我们的界限是非平凡和新颖的。我们对过拟合的结论是双重的：(i) 谱衰减多项式的核回归器必须在存在噪声标记的训练数据的情况下得到很好的泛化；这些模型表现出所谓的温和过拟合；(ii) 如果任何核岭回归器的特征谱指数衰减，则其泛化差，即表现出灾难性过拟合。这增加了核岭回归器表现出良性过拟合的可用特征谱衰减次多项式的极端情况的表征。我们的分析结合了新的随机矩阵理论(RMT)。

    We derive new bounds for the condition number of kernel matrices, which we then use to enhance existing non-asymptotic test error bounds for kernel ridgeless regression in the over-parameterized regime for a fixed input dimension. For kernels with polynomial spectral decay, we recover the bound from previous work; for exponential decay, our bound is non-trivial and novel.   Our conclusion on overfitting is two-fold: (i) kernel regressors whose eigenspectrum decays polynomially must generalize well, even in the presence of noisy labeled training data; these models exhibit so-called tempered overfitting; (ii) if the eigenspectrum of any kernel ridge regressor decays exponentially, then it generalizes poorly, i.e., it exhibits catastrophic overfitting. This adds to the available characterization of kernel ridge regressors exhibiting benign overfitting as the extremal case where the eigenspectrum of the kernel decays sub-polynomially. Our analysis combines new random matrix theory (RMT) te
    
[^3]: Ginger: 一种用于通用神经网络的线性复杂度高效曲率近似方法

    Ginger: An Efficient Curvature Approximation with Linear Complexity for General Neural Networks

    [https://arxiv.org/abs/2402.03295](https://arxiv.org/abs/2402.03295)

    Ginger是一种用于通用神经网络的高效曲率近似方法，具有线性复杂度。它通过特征分解来逆向计算广义高斯牛顿矩阵，避免了传统方法中的高内存和高时间复杂度问题。

    

    二阶优化方法，如广义高斯牛顿法，由于利用了目标函数的曲率信息和预处理矩阵，被认为更加强大。尽管在理论上具有诱人的优势，但它们不易应用于现代深度学习。主要原因是计算矩阵的逆所需的二次内存和三次时间复杂度是不可行的，即使使用先进的硬件也不行。在这项工作中，我们提出了Ginger，一种用于广义高斯牛顿矩阵逆的特征分解方法。我们的方法在每次迭代中具有高效的线性内存和时间复杂度。我们直接维护条件矩阵的逆，以使近似更加准确，而不是近似条件矩阵。我们提供了Ginger在非凸目标上的收敛结果。我们在不同任务和不同模型架构上的实验证实了我们方法的有效性。我们的代码...

    Second-order optimization approaches like the generalized Gauss-Newton method are considered more powerful as they utilize the curvature information of the objective function with preconditioning matrices. Albeit offering tempting theoretical benefits, they are not easily applicable to modern deep learning. The major reason is due to the quadratic memory and cubic time complexity to compute the inverse of the matrix. These requirements are infeasible even with state-of-the-art hardware. In this work, we propose Ginger, an eigendecomposition for the inverse of the generalized Gauss-Newton matrix. Our method enjoys efficient linear memory and time complexity for each iteration. Instead of approximating the conditioning matrix, we directly maintain its inverse to make the approximation more accurate. We provide the convergence result of Ginger for non-convex objectives. Our experiments on different tasks with different model architectures verify the effectiveness of our method. Our code i
    
[^4]: Flora: 低秩适配器是悄悄的梯度压缩器

    Flora: Low-Rank Adapters Are Secretly Gradient Compressors

    [https://arxiv.org/abs/2402.03293](https://arxiv.org/abs/2402.03293)

    本文研究了低秩适配器的动力学，并提出了一种基于随机投影的方法Flora，通过重新采样投影矩阵实现高秩更新，同时减少优化状态的空间复杂度。

    

    尽管大型神经网络展示了完成不同任务的显着能力，但它们需要过多的内存使用来存储训练的优化状态。为了缓解这个问题，提出低秩适配（LoRA）来通过训练更少的参数来减少优化状态。然而，LoRA将整体权重更新矩阵限制为低秩，限制了模型的性能。在这项工作中，我们研究了LoRA的动力学，并确定它可以近似为随机投影。基于这一观察，我们提出了Flora，它能够通过重新采样投影矩阵实现高秩更新，同时享受优化状态的次线性空间复杂度。我们在不同任务和模型架构上进行实验证实了我们方法的有效性。

    Despite large neural networks demonstrating remarkable abilities to complete different tasks, they require excessive memory usage to store the optimization states for training. To alleviate this, the low-rank adaptation (LoRA) is proposed to reduce the optimization states by training fewer parameters. However, LoRA restricts overall weight update matrices to be low-rank, limiting the model performance. In this work, we investigate the dynamics of LoRA and identify that it can be approximated by a random projection. Based on this observation, we propose Flora, which is able to achieve high-rank updates by resampling the projection matrices while enjoying the sublinear space complexity of optimization states. We conduct experiments across different tasks and model architectures to verify the effectiveness of our approach.
    
[^5]: 一个部分观察到的奖励状态在RLHF中的框架

    A Framework for Partially Observed Reward-States in RLHF

    [https://arxiv.org/abs/2402.03282](https://arxiv.org/abs/2402.03282)

    这篇论文提出了一个针对RLHF的框架，在其中考虑了部分观察到的奖励状态，并通过将基数反馈和决斗反馈缩减为PORRL形式进行了建模和算法开发。

    

    最近几年来，强化学习从人类反馈（RLHF）的研究因其在LLMs的发展中起到的作用而变得重要。神经科学研究表明，人类对刺激的反应已知依赖于部分观察到的“内部状态”。不幸的是，当前的RLHF模型没有考虑到这一点。此外，大多数RLHF模型没有考虑到中间反馈，在实证研究中变得越来越重要，可以帮助提高样本复杂性和对齐性。为了解决这些局限性，我们将RLHF建模为部分观察到的奖励状态的强化学习（PORRL）。我们展示了从RLHF中两种主要形式的人类反馈 - 基数反馈和决斗反馈到PORRL的缩减。对于基数反馈，我们开发了通用的统计高效算法，并将它们实例化为POR-UCRL和POR-UCBVI。对于决斗反馈，我们表明，简单的基数反馈缩减不能达到亚线性的决斗回归。

    The study of reinforcement learning from human feedback (RLHF) has gained prominence in recent years due to its role in the development of LLMs. Neuroscience research shows that human responses to stimuli are known to depend on partially-observed "internal states." Unfortunately current models of RLHF do not take take this into consideration. Moreover most RLHF models do not account for intermediate feedback, which is gaining importance in empirical work and can help improve both sample complexity and alignment. To address these limitations, we model RLHF as reinforcement learning with partially observed reward-states (PORRL). We show reductions from the the two dominant forms of human feedback in RLHF - cardinal and dueling feedback to PORRL. For cardinal feedback, we develop generic statistically efficient algorithms and instantiate them to present POR-UCRL and POR-UCBVI. For dueling feedback, we show that a naive reduction to cardinal feedback fails to achieve sublinear dueling regr
    
[^6]: 学习Predict-then-Optimize框架中的最优策略

    Learning Best-in-Class Policies for the Predict-then-Optimize Framework

    [https://arxiv.org/abs/2402.03256](https://arxiv.org/abs/2402.03256)

    我们提出了一种新颖的决策感知替代损失函数家族，用于predict-then-optimize框架，并且通过数值证据证实了其在误设置下的优越性。

    

    我们提出了一种新颖的决策感知替代损失函数家族，称为Perturbation Gradient（PG）损失，用于predict-then-optimize框架。这些损失直接近似了下游决策损失，并可以使用现成的基于梯度的方法进行优化。重要的是，与现有的替代损失不同，我们的PG损失的近似误差随着样本数量的增加而消失。这意味着优化我们的替代损失可以在渐近意义下得到最佳策略，即使在误设置下也是如此。这是第一个在误设置下的这样的结果，我们提供了数值证据证实了当基础模型误设置且噪声不是中心对称时，我们的PG损失在实践中显著优于现有的提案。鉴于在实践中误设置很常见--特别是当我们可能更喜欢一个更简单、更可解释的模型时--PG损失提供了一种新颖的、理论上有依据的、可计算的决策感知方法。

    We propose a novel family of decision-aware surrogate losses, called Perturbation Gradient (PG) losses, for the predict-then-optimize framework. These losses directly approximate the downstream decision loss and can be optimized using off-the-shelf gradient-based methods. Importantly, unlike existing surrogate losses, the approximation error of our PG losses vanishes as the number of samples grows. This implies that optimizing our surrogate loss yields a best-in-class policy asymptotically, even in misspecified settings. This is the first such result in misspecified settings and we provide numerical evidence confirming our PG losses substantively outperform existing proposals when the underlying model is misspecified and the noise is not centrally symmetric. Insofar as misspecification is commonplace in practice -- especially when we might prefer a simpler, more interpretable model -- PG losses offer a novel, theoretically justified, method for computationally tractable decision-aware 
    
[^7]: 表示学习的最小描述长度和泛化保证

    Minimum Description Length and Generalization Guarantees for Representation Learning

    [https://arxiv.org/abs/2402.03254](https://arxiv.org/abs/2402.03254)

    本文提出了一个可压缩性框架，通过计算表示学习算法的泛化误差的上界，改进了现有启发式方法，并提供了关于理论泛化保证的新见解。

    

    设计高效的统计有监督学习算法的一个主要挑战是找到不仅在可用训练样本上表现良好而且在未见数据上也表现良好的表示形式。尽管表示学习的研究引发了许多兴趣，但大多数现有方法都是启发式的；对于理论上的泛化保证几乎没有什么了解。在本文中，我们建立了一个可压缩性框架，使我们能够通过标签或潜在变量（表示形式）的"最小描述长度"（MDL）来推导表示学习算法的泛化误差的上界。与通常被认为反映算法泛化能力的编码器输入和表示之间的互信息相比，我们的新界限涉及表示（或标签）分布之间的"多字母"相对熵，在相关文献中对算法的泛化能力的反映还不足。

    A major challenge in designing efficient statistical supervised learning algorithms is finding representations that perform well not only on available training samples but also on unseen data. While the study of representation learning has spurred much interest, most existing such approaches are heuristic; and very little is known about theoretical generalization guarantees.   In this paper, we establish a compressibility framework that allows us to derive upper bounds on the generalization error of a representation learning algorithm in terms of the "Minimum Description Length" (MDL) of the labels or the latent variables (representations). Rather than the mutual information between the encoder's input and the representation, which is often believed to reflect the algorithm's generalization capability in the related literature but in fact, falls short of doing so, our new bounds involve the "multi-letter" relative entropy between the distribution of the representations (or labels) of t
    
[^8]: 重复使用批次在两层网络的梯度下降中的好处：打破信息和跳跃指数的诅咒

    The Benefits of Reusing Batches for Gradient Descent in Two-Layer Networks: Breaking the Curse of Information and Leap Exponents

    [https://arxiv.org/abs/2402.03220](https://arxiv.org/abs/2402.03220)

    该论文研究了在两层神经网络中学习多指数目标函数时，重复使用批次的梯度下降（GD）的训练动态。研究发现，与单次GD相比，多次GD能够克服目标函数的限制，仅需两个时间步骤就能实现网络与目标子空间的重叠，展示了在有限时间内有效学习的广泛函数类。这些结果基于动力平均场理论（DMFT）的分析。

    

    本研究探讨了学习多指数目标函数时，两层神经网络的训练动态。我们关注重复多次使用批次的多次梯度下降（GD），并展示它与单次梯度下降相比，显著改变了对于哪些函数是可学习的的结论。具体而言，我们发现具有有限步长的多次GD能够克服目标函数的信息指数（Ben Arous等人，2021）和跳跃指数（Abbe等人，2023）所给出的梯度流和单次GD的限制。我们发现，通过重复使用批次，网络仅需两个时间步骤就能与目标子空间达成重叠，即使函数不满足阶梯性质（Abbe等人，2021）。我们对能够在有限时间内有效学习的（广泛的）函数类进行了表征。我们的结果证明基于动力平均场理论（DMFT）的分析。我们进一步提供了动态的闭式描述。

    We investigate the training dynamics of two-layer neural networks when learning multi-index target functions. We focus on multi-pass gradient descent (GD) that reuses the batches multiple times and show that it significantly changes the conclusion about which functions are learnable compared to single-pass gradient descent. In particular, multi-pass GD with finite stepsize is found to overcome the limitations of gradient flow and single-pass GD given by the information exponent (Ben Arous et al., 2021) and leap exponent (Abbe et al., 2023) of the target function. We show that upon re-using batches, the network achieves in just two time steps an overlap with the target subspace even for functions not satisfying the staircase property (Abbe et al., 2021). We characterize the (broad) class of functions efficiently learned in finite time. The proof of our results is based on the analysis of the Dynamical Mean-Field Theory (DMFT). We further provide a closed-form description of the dynamica
    
[^9]: 低多线性秩张量逼近的随机矩阵方法

    A Random Matrix Approach to Low-Multilinear-Rank Tensor Approximation

    [https://arxiv.org/abs/2402.03169](https://arxiv.org/abs/2402.03169)

    该研究采用随机矩阵方法，在低多线性秩张量逼近中展示了对种植的低秩信号的估计，并根据大维谱行为和信噪比准确预测了重建性能，并给出了HOOI收敛的充分条件。

    

    本研究从计算阈值附近的一般尖峰张量模型，对种植的低秩信号估计进行了全面的认识。依靠大型随机矩阵理论的标准工具，我们表征了数据张量的展开的大维谱行为，并展示了决定主要信号方向可检测性的相关信噪比。这些结果可以准确地预测在非平凡区域的截断多线性奇异值分解(MLSVD)的重建性能。这一点尤其重要，因为它作为更高阶正交迭代(HOOI)方案的初始化，其收敛到最佳低多线性秩逼近完全取决于其初始化。我们给出了HOOI收敛的充分条件，并证明在大维极限下收敛前的迭代次数趋于1。

    This work presents a comprehensive understanding of the estimation of a planted low-rank signal from a general spiked tensor model near the computational threshold. Relying on standard tools from the theory of large random matrices, we characterize the large-dimensional spectral behavior of the unfoldings of the data tensor and exhibit relevant signal-to-noise ratios governing the detectability of the principal directions of the signal. These results allow to accurately predict the reconstruction performance of truncated multilinear SVD (MLSVD) in the non-trivial regime. This is particularly important since it serves as an initialization of the higher-order orthogonal iteration (HOOI) scheme, whose convergence to the best low-multilinear-rank approximation depends entirely on its initialization. We give a sufficient condition for the convergence of HOOI and show that the number of iterations before convergence tends to $1$ in the large-dimensional limit.
    
[^10]: 图上的去中心化双级优化: 无环算法更新和瞬态迭代复杂性

    Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity

    [https://arxiv.org/abs/2402.03167](https://arxiv.org/abs/2402.03167)

    本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。

    

    随机双级优化（SBO）在处理嵌套结构方面的多样性使其在机器学习中变得越来越重要。为了解决大规模SBO，去中心化方法作为有效的范例出现，其中节点与直接相邻节点进行通信，无需中央服务器，从而提高通信效率和增强算法的稳健性。然而，当前的去中心化SBO算法面临挑战，包括昂贵的内部循环更新和对网络拓扑、数据异构性和嵌套双级算法结构的影响不明确。在本文中，我们引入了一种单循环的去中心化SBO（D-SOBA）算法，并建立了其瞬态迭代复杂性，首次澄清了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA实现了最先进的渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性。

    Stochastic bilevel optimization (SBO) is becoming increasingly essential in machine learning due to its versatility in handling nested structures. To address large-scale SBO, decentralized approaches have emerged as effective paradigms in which nodes communicate with immediate neighbors without a central server, thereby improving communication efficiency and enhancing algorithmic robustness. However, current decentralized SBO algorithms face challenges, including expensive inner-loop updates and unclear understanding of the influence of network topology, data heterogeneity, and the nested bilevel algorithmic structures. In this paper, we introduce a single-loop decentralized SBO (D-SOBA) algorithm and establish its transient iteration complexity, which, for the first time, clarifies the joint influence of network topology and data heterogeneity on decentralized bilevel algorithms. D-SOBA achieves the state-of-the-art asymptotic rate, asymptotic gradient/Hessian complexity, and transien
    
[^11]: 用于稳健学习模型预测的多步损失函数

    A Multi-step Loss Function for Robust Learning of the Dynamics in Model-based Reinforcement Learning

    [https://arxiv.org/abs/2402.03146](https://arxiv.org/abs/2402.03146)

    本文提出了一种用于稳健学习模型预测的多步损失函数，通过加权平均均方误差损失在不同未来时间点上来训练单步模型。这种损失函数在存在噪声的情况下尤为有效，并在各种任务中取得了显著的预测改善。

    

    在基于模型的强化学习中，大多数算法依赖于从数据学到的单步动力学模型模拟轨迹。这种方法的一个关键挑战是随着轨迹长度的增加，单步预测误差的累积问题。本文通过使用多步目标来训练单步模型来解决这个问题。我们的目标是在不同的未来时间点上加权平均的均方误差(MSE)损失函数。我们发现，这种新的损失函数在数据存在噪声的情况下特别有用（观测上的加性高斯噪声），而这种情况在现实环境中经常出现。为了支持多步损失函数，首先我们在两种可处理的情况下研究了它的性质：i）一维线性系统，和ii）两参数非线性系统。其次，我们在各种任务（环境或数据集）中展示了使用这种损失函数训练的模型在未来预测中的平均R2得分方面取得了显著的改善。

    In model-based reinforcement learning, most algorithms rely on simulating trajectories from one-step models of the dynamics learned on data. A critical challenge of this approach is the compounding of one-step prediction errors as the length of the trajectory grows. In this paper we tackle this issue by using a multi-step objective to train one-step models. Our objective is a weighted sum of the mean squared error (MSE) loss at various future horizons. We find that this new loss is particularly useful when the data is noisy (additive Gaussian noise in the observations), which is often the case in real-life environments. To support the multi-step loss, first we study its properties in two tractable cases: i) uni-dimensional linear system, and ii) two-parameter non-linear system. Second, we show in a variety of tasks (environments or datasets) that the models learned with this loss achieve a significant improvement in terms of the averaged R2-score on future prediction horizons. Finally,
    
[^12]: 无参随机优化的自由度有多高？

    How Free is Parameter-Free Stochastic Optimization?

    [https://arxiv.org/abs/2402.03126](https://arxiv.org/abs/2402.03126)

    这个论文研究了无参随机优化的问题，提出了一种完全无参的方法，通过简单的超参数搜索技术在非凸和凸设置下都能取得优于先进算法的性能。同时，论文还建立了一个下界，指出完全无参的方法在某些情况下无法实现。

    

    我们研究了无参随机优化的问题，探讨了在什么条件下可以存在完全无参的方法：这些方法可以达到与最优调参方法相竞争的收敛速度，而不需要对真实问题参数有很多知识。现有的无参方法只能被视为“部分”无参，因为它们需要对真实问题参数有一些非平凡的知识，比如随机梯度范数的上界、到最小值的距离的上界等。在非凸设置中，我们证明了一个简单的超参数搜索技术可以得到一个完全无参的方法，在性能上超过了更复杂的先进算法。在具有噪声函数值的凸设置下，在较小的噪声假设下，我们也提供了类似的结果。最后，假设只能访问随机梯度，我们建立了一个下界，使得完全无参的方法无法实现。

    We study the problem of parameter-free stochastic optimization, inquiring whether, and under what conditions, do fully parameter-free methods exist: these are methods that achieve convergence rates competitive with optimally tuned methods, without requiring significant knowledge of the true problem parameters. Existing parameter-free methods can only be considered ``partially'' parameter-free, as they require some non-trivial knowledge of the true problem parameters, such as a bound on the stochastic gradient norms, a bound on the distance to a minimizer, etc. In the non-convex setting, we demonstrate that a simple hyperparameter search technique results in a fully parameter-free method that outperforms more sophisticated state-of-the-art algorithms. We also provide a similar result in the convex setting with access to noisy function values under mild noise assumptions. Finally, assuming only access to stochastic gradients, we establish a lower bound that renders fully parameter-free s
    
[^13]: 高维贝叶斯优化通过协方差矩阵适应策略

    High-dimensional Bayesian Optimization via Covariance Matrix Adaptation Strategy

    [https://arxiv.org/abs/2402.03104](https://arxiv.org/abs/2402.03104)

    本文提出了一种高维贝叶斯优化方法，通过协方差矩阵适应策略定义局部搜索区域，能够解决将贝叶斯优化应用于高维优化问题的挑战。

    

    贝叶斯优化（BO）是一种寻找昂贵黑盒函数全局最优解的有效方法。然而，众所周知，将BO应用于高维优化问题具有挑战性。为了解决这个问题，一个有希望的解决方案是使用局部搜索策略将搜索域划分成包含全局最优解可能性较高的局部区域，然后在这些区域内使用BO优化目标函数。在本文中，我们提出了一种使用协方差矩阵适应（CMA）策略定义局部区域的新技术。具体来说，我们使用CMA来学习一个能够估计数据点作为目标函数全局最优解概率的搜索分布。基于这个搜索分布，我们定义由高概率数据点组成的局部区域。我们的方法作为一个元算法，可以整合现有的黑盒BO优化方法。

    Bayesian Optimization (BO) is an effective method for finding the global optimum of expensive black-box functions. However, it is well known that applying BO to high-dimensional optimization problems is challenging. To address this issue, a promising solution is to use a local search strategy that partitions the search domain into local regions with high likelihood of containing the global optimum, and then use BO to optimize the objective function within these regions. In this paper, we propose a novel technique for defining the local regions using the Covariance Matrix Adaptation (CMA) strategy. Specifically, we use CMA to learn a search distribution that can estimate the probabilities of data points being the global optimum of the objective function. Based on this search distribution, we then define the local regions consisting of data points with high probabilities of being the global optimum. Our approach serves as a meta-algorithm as it can incorporate existing black-box BO optim
    
[^14]: 扩散吉布斯采样

    Diffusive Gibbs Sampling

    [https://arxiv.org/abs/2402.03008](https://arxiv.org/abs/2402.03008)

    扩散吉布斯采样是一种创新的采样方法，通过集成扩散模型并应用吉布斯采样，有效地从具有远程和断开模态特征的分布中采样，表现出比其他方法更好的混合性能，并在多种任务中取得显著改进的结果。

    

    传统马尔可夫链蒙特卡洛（MCMC）方法在多模态分布的混合不足方面存在着挑战，特别是在贝叶斯推断和分子动力学等实际应用中。针对这个问题，我们提出了一种创新的采样方法——扩散吉布斯采样（DiGS），用于有效采样具有远程和断开模态特征的分布。DiGS集成了扩散模型的最新发展，利用高斯卷积创建一个辅助噪声分布，以在原始空间中连接孤立的模态，并应用吉布斯采样从两个空间中交替抽取样本。我们的方法在采样多模态分布方面表现出比并行温度法等最先进方法更好的混合性能。我们证明我们的采样器在各种任务中取得了显著改进的结果，包括高斯混合模型、贝叶斯神经网络和分子动力学。

    The inadequate mixing of conventional Markov Chain Monte Carlo (MCMC) methods for multi-modal distributions presents a significant challenge in practical applications such as Bayesian inference and molecular dynamics. Addressing this, we propose Diffusive Gibbs Sampling (DiGS), an innovative family of sampling methods designed for effective sampling from distributions characterized by distant and disconnected modes. DiGS integrates recent developments in diffusion models, leveraging Gaussian convolution to create an auxiliary noisy distribution that bridges isolated modes in the original space and applying Gibbs sampling to alternately draw samples from both spaces. Our approach exhibits a better mixing property for sampling multi-modal distributions than state-of-the-art methods such as parallel tempering. We demonstrate that our sampler attains substantially improved results across various tasks, including mixtures of Gaussians, Bayesian neural networks and molecular dynamics.
    
[^15]: 开发一种适用于受环境条件变化影响的昂贵实验和模拟的实用贝叶斯优化算法

    On the development of a practical Bayesian optimisation algorithm for expensive experiments and simulations with changing environmental conditions

    [https://arxiv.org/abs/2402.03006](https://arxiv.org/abs/2402.03006)

    本文在受环境条件变化影响的昂贵实验和模拟中，将贝叶斯优化方法推广到包含可控和不可控参数的系统优化中，通过在所有变量上拟合全局代理模型，但只在对不可控变量的测量条件下优化可控参数。

    

    工程实验通常在受控环境中进行，可以将参数设置为任何所需值。然而，在真实环境中，通常假设相同条件不成立，因为许多实验受不可控制的环境条件（如温度、湿度和风速）的影响。在优化这些实验时，应该重点关注在给定不可控变量条件下找到最优值。本文将贝叶斯优化方法推广到在包含可控和不可控参数的变化环境中进行系统优化。该推广通过在所有可控和环境变量上拟合全局代理模型，但只在对不可控变量的测量条件下优化可控参数。该方法在两个合成测试函数上进行了验证，研究了噪声水平、环境参数数量和参数波动的影响。

    Experiments in engineering are typically conducted in controlled environments where parameters can be set to any desired value. This assumes that the same applies in a real-world setting -- an assumption that is often incorrect as many experiments are influenced by uncontrollable environmental conditions such as temperature, humidity and wind speed. When optimising such experiments, the focus should lie on finding optimal values conditionally on these uncontrollable variables. This article extends Bayesian optimisation to the optimisation of systems in changing environments that include controllable and uncontrollable parameters. The extension fits a global surrogate model over all controllable and environmental variables but optimises only the controllable parameters conditional on measurements of the uncontrollable variables. The method is validated on two synthetic test functions and the effects of the noise level, the number of the environmental parameters, the parameter fluctuatio
    
[^16]: 小心使用手术刀：使用EMA改进梯度手术

    Careful with that Scalpel: Improving Gradient Surgery with an EMA

    [https://arxiv.org/abs/2402.02998](https://arxiv.org/abs/2402.02998)

    通过将训练损失梯度和辅助梯度在训练梯度方向上的正交投影结合起来，使用EMA（指数移动平均）可以改进梯度手术，提高深度学习估计管道的性能。

    

    在深度学习估计管道中，除了最小化单一的训练损失外，还依赖于辅助目标来量化和鼓励模型的可取属性（例如在另一个数据集上的表现，鲁棒性，与先前的一致性）。虽然将辅助损失与训练损失相加作为正则化的最简单方法，但最近的研究表明，通过混合梯度而不仅仅是简单相加，可以提高性能；这被称为梯度手术。我们将这个问题看作是一个约束最小化问题，其中辅助目标在训练损失的最小化集合中被最小化。为了解决这个双层问题，我们采用了一个参数更新方向，它将训练损失梯度和辅助梯度在训练梯度方向上的正交投影结合起来。在梯度来自小批次的情况下，我们解释了如何使用训练损失梯度的移动平均来维护。

    Beyond minimizing a single training loss, many deep learning estimation pipelines rely on an auxiliary objective to quantify and encourage desirable properties of the model (e.g. performance on another dataset, robustness, agreement with a prior). Although the simplest approach to incorporating an auxiliary loss is to sum it with the training loss as a regularizer, recent works have shown that one can improve performance by blending the gradients beyond a simple sum; this is known as gradient surgery. We cast the problem as a constrained minimization problem where the auxiliary objective is minimized among the set of minimizers of the training loss. To solve this bilevel problem, we follow a parameter update direction that combines the training loss gradient and the orthogonal projection of the auxiliary gradient to the training gradient. In a setting where gradients come from mini-batches, we explain how, using a moving average of the training loss gradients, we can carefully maintain
    
[^17]: 提升，投票分类器和随机采样压缩方案

    Boosting, Voting Classifiers and Randomized Sample Compression Schemes

    [https://arxiv.org/abs/2402.02976](https://arxiv.org/abs/2402.02976)

    本研究提出了一种随机提升算法来解决传统提升算法的性能问题，并通过构建一个通用框架将样本压缩方法扩展到支持随机学习算法，实现了在样本大小上具有单对数依赖的泛化错误。

    

    在提升中，我们旨在利用多个弱学习器来产生一个强学习器。这个范式的核心是将强学习器建模为一个投票分类器，它输出弱学习器的加权多数投票。尽管许多成功的提升算法，如标志性的AdaBoost，产生投票分类器，但它们的理论性能长期以来一直不够优化：迄今为止，已知的使投票分类器达到给定准确性所需的训练样本数的最佳界限总是至少包含至多两个对数因子，而这已经超过了一般的弱到强学习器所能实现的范围。在这项工作中，我们通过提出一种随机提升算法打破这一障碍，该算法输出的投票分类器在样本大小上包含单对数依赖的泛化错误。我们通过构建一个通用框架将样本压缩方法扩展到支持随机学习算法来获得这个结果。

    In boosting, we aim to leverage multiple weak learners to produce a strong learner. At the center of this paradigm lies the concept of building the strong learner as a voting classifier, which outputs a weighted majority vote of the weak learners. While many successful boosting algorithms, such as the iconic AdaBoost, produce voting classifiers, their theoretical performance has long remained sub-optimal: the best known bounds on the number of training examples necessary for a voting classifier to obtain a given accuracy has so far always contained at least two logarithmic factors above what is known to be achievable by general weak-to-strong learners. In this work, we break this barrier by proposing a randomized boosting algorithm that outputs voting classifiers whose generalization error contains a single logarithmic dependency on the sample size. We obtain this result by building a general framework that extends sample compression methods to support randomized learning algorithms ba
    
[^18]: 关于注意力层的词敏感性的理解：通过随机特征的研究

    Towards Understanding the Word Sensitivity of Attention Layers: A Study via Random Features

    [https://arxiv.org/abs/2402.02969](https://arxiv.org/abs/2402.02969)

    通过研究随机特征，我们发现注意力层具有较高的词敏感性，这对于理解transformers的成功以及自然语言处理任务中的上下文含义非常重要。

    

    揭示transformers异常成功背后原因需要更好地理解为什么注意力层适用于自然语言处理任务。特别是，这些任务要求预测模型捕捉上下文含义，即使句子很长，这往往取决于一个或几个词。我们的工作在随机特征的典型设置中研究了这一关键属性，称为词敏感性（WS）。我们展示了注意力层具有较高的WS，即在嵌入空间中存在一个向量，能够大幅扰动随机注意力特征映射。这个论点关键地利用了注意力层中softmax的作用，突显了它相对于其他激活函数（如ReLU）的优势。相反，标准随机特征的WS是$1/\sqrt{n}$阶的，$n$是文本样本中的单词数，因此它随上下文的长度而衰减。然后，我们将这些关于词敏感性的结果转化为泛化界：由于...

    Unveiling the reasons behind the exceptional success of transformers requires a better understanding of why attention layers are suitable for NLP tasks. In particular, such tasks require predictive models to capture contextual meaning which often depends on one or few words, even if the sentence is long. Our work studies this key property, dubbed word sensitivity (WS), in the prototypical setting of random features. We show that attention layers enjoy high WS, namely, there exists a vector in the space of embeddings that largely perturbs the random attention features map. The argument critically exploits the role of the softmax in the attention layer, highlighting its benefit compared to other activations (e.g., ReLU). In contrast, the WS of standard random features is of order $1/\sqrt{n}$, $n$ being the number of words in the textual sample, and thus it decays with the length of the context. We then translate these results on the word sensitivity into generalization bounds: due to th
    
[^19]: 关于Softmax Gating混合专家模型中最小二乘估计的研究

    On Least Squares Estimation in Softmax Gating Mixture of Experts

    [https://arxiv.org/abs/2402.02952](https://arxiv.org/abs/2402.02952)

    本研究探讨了在确定性MoE模型下使用最小二乘估计器的性能，并建立了强可识别性条件来描述不同类型专家函数的收敛行为。

    

    专家模型是一种统计机器学习设计，使用Softmax Gating函数聚合多个专家网络，以形成一个更复杂和表达力更强的模型。尽管由于可扩展性而在多个应用领域中广泛使用，但MoE模型的数学和统计性质复杂且难以分析。因此，以前的理论工作主要集中在概率MoE模型上，这些模型假设数据是由高斯MoE模型生成的，这在实践中是不切实际的。在这项工作中，我们研究了在确定性MoE模型下最小二乘估计器（LSE）的性能，在该模型中，数据根据回归模型进行采样，这是一个尚未被充分探索的设置。我们建立了一个称为强可识别性的条件，以表征不同类型专家函数的收敛行为。我们证明了对于强可识别专家的估计速度，即

    Mixture of experts (MoE) model is a statistical machine learning design that aggregates multiple expert networks using a softmax gating function in order to form a more intricate and expressive model. Despite being commonly used in several applications owing to their scalability, the mathematical and statistical properties of MoE models are complex and difficult to analyze. As a result, previous theoretical works have primarily focused on probabilistic MoE models by imposing the impractical assumption that the data are generated from a Gaussian MoE model. In this work, we investigate the performance of the least squares estimators (LSE) under a deterministic MoE model where the data are sampled according to a regression model, a setting that has remained largely unexplored. We establish a condition called strong identifiability to characterize the convergence behavior of various types of expert functions. We demonstrate that the rates for estimating strongly identifiable experts, namel
    
[^20]: 动态拜占庭-强鲁棒学习：适应切换拜占庭工作机制

    Dynamic Byzantine-Robust Learning: Adapting to Switching Byzantine Workers

    [https://arxiv.org/abs/2402.02951](https://arxiv.org/abs/2402.02951)

    $\textsf{DynaBRO}$是一种动态拜占庭-强鲁棒学习的方法，能够适应切换拜占庭工作机制，并且在渐近收敛速率上与静态情况相匹配。通过多级蒙特卡洛渐变估计技术、强鲁棒工作机制更新的聚合和故障安全过滤器的引入，我们的方法能够经受住$\mathcal{O}(\sqrt{T})$轮拜占庭身份的改变。另外，通过使用自适应学习率，我们的方法消除了对百分比的需求。

    

    拜占庭-强鲁棒学习作为一种突出的容错分布式机器学习框架已经出现。然而，大多数技术考虑的是静态情况，其中在学习过程中拜占庭机器的身份保持不变。这种假设不能捕捉到现实世界中的动态拜占庭行为，可能包括短暂故障或有针对性的时间攻击。为了解决这个限制，我们提出了一种新的方法$\textsf{DynaBRO}$，它能够经受住$\mathcal{O}(\sqrt{T})$轮拜占庭身份的改变（其中$T$是总训练轮数），同时与静态情况下的渐近收敛速率相匹配。我们的方法将多级蒙特卡洛（MLMC）渐变估计技术与工作机制更新的强鲁棒聚合相结合，并引入了一个故障安全过滤器来限制动态拜占庭策略的偏差。此外，通过利用自适应学习率，我们的方法消除了对百分比的需求。

    Byzantine-robust learning has emerged as a prominent fault-tolerant distributed machine learning framework. However, most techniques consider the static setting, wherein the identity of Byzantine machines remains fixed during the learning process. This assumption does not capture real-world dynamic Byzantine behaviors, which may include transient malfunctions or targeted temporal attacks. Addressing this limitation, we propose $\textsf{DynaBRO}$ -- a new method capable of withstanding $\mathcal{O}(\sqrt{T})$ rounds of Byzantine identity alterations (where $T$ is the total number of training rounds), while matching the asymptotic convergence rate of the static setting. Our method combines a multi-level Monte Carlo (MLMC) gradient estimation technique with robust aggregation of worker updates and incorporates a fail-safe filter to limit bias from dynamic Byzantine strategies. Additionally, by leveraging an adaptive learning rate, our approach eliminates the need for knowing the percentag
    
[^21]: 具有异质多中心人群的回归模型的贝叶斯联合推断

    Bayesian Federated Inference for regression models with heterogeneous multi-center populations

    [https://arxiv.org/abs/2402.02898](https://arxiv.org/abs/2402.02898)

    这项研究提出了一种利用贝叶斯联合推断方法，在不同中心分别分析本地数据，并将统计推断结果组合起来，以解决样本量不足的问题，并准确估计回归模型的参数。

    

    为了准确估计回归模型的参数，样本量必须相对于可能的预测变量个数足够大。在实际应用中，通常缺乏足够的数据，这可能导致模型过拟合，并因此无法对新患者的结果进行可靠预测。合并来自不同（医疗）中心收集的数据可以缓解这个问题，但通常由于隐私法规或物流问题而不可行。另一种方法是分析各个中心的本地数据，然后使用贝叶斯联合推断（BFI）方法将统计推断结果进行组合。这种方法的目标是从各个中心的推断结果中计算出如果对组合数据进行了统计分析后会得到什么结果。我们解释了同质和异质中心人群下的方法，并给出了真实的示例。

    To estimate accurately the parameters of a regression model, the sample size must be large enough relative to the number of possible predictors for the model. In practice, sufficient data is often lacking, which can lead to overfitting of the model and, as a consequence, unreliable predictions of the outcome of new patients. Pooling data from different data sets collected in different (medical) centers would alleviate this problem, but is often not feasible due to privacy regulation or logistic problems. An alternative route would be to analyze the local data in the centers separately and combine the statistical inference results with the Bayesian Federated Inference (BFI) methodology. The aim of this approach is to compute from the inference results in separate centers what would have been found if the statistical analysis was performed on the combined data. We explain the methodology under homogeneity and heterogeneity across the populations in the separate centers, and give real lif
    
[^22]: 图神经机器：一种处理表格数据的新模型

    Graph Neural Machine: A New Model for Learning with Tabular Data

    [https://arxiv.org/abs/2402.02862](https://arxiv.org/abs/2402.02862)

    本论文提出了一种新的机器学习模型，图神经机器（GNM），用于处理表格数据。GNM使用同步消息传递方案，并用几乎完全图代替了多层感知机（MLP）的有向无环图。实验结果表明，在多个数据集上，GNM模型的性能优于MLP架构。

    

    近年来，人们对将不同领域的数据映射到图结构的方法越来越感兴趣。神经网络模型如多层感知机（MLP）可以被建模为图。事实上，MLP可以表示为有向无环图。图神经网络（GNN）最近已成为在图上执行机器学习任务的标准工具。在这项工作中，我们展示了MLP等价于一个基于异步消息传递的GNN模型，该模型在MLP的图表示上操作。然后，我们提出了一种新的处理表格数据的机器学习模型，称为图神经机器（GNM），它用一个几乎完全图取代了MLP的有向无环图，并采用同步消息传递方案。我们表明单个GNM模型可以模拟多个MLP模型。我们在多个分类和回归数据集上评估了所提出的模型。在大多数情况下，GNM模型优于MLP架构。

    In recent years, there has been a growing interest in mapping data from different domains to graph structures. Among others, neural network models such as the multi-layer perceptron (MLP) can be modeled as graphs. In fact, MLPs can be represented as directed acyclic graphs. Graph neural networks (GNNs) have recently become the standard tool for performing machine learning tasks on graphs. In this work, we show that an MLP is equivalent to an asynchronous message passing GNN model which operates on the MLP's graph representation. We then propose a new machine learning model for tabular data, the so-called Graph Neural Machine (GNM), which replaces the MLP's directed acyclic graph with a nearly complete graph and which employs a synchronous message passing scheme. We show that a single GNM model can simulate multiple MLP models. We evaluate the proposed model in several classification and regression datasets. In most cases, the GNM model outperforms the MLP architecture.
    
[^23]: 利用嘈杂观察在零和游戏中的优势

    Leveraging Noisy Observations in Zero-Sum Games

    [https://arxiv.org/abs/2402.02861](https://arxiv.org/abs/2402.02861)

    本文研究了在零和游戏中利用嘈杂观察的优势。具体而言，通过对给定概率测度进行采样，领导者承诺对手选择行动，而追随者根据其当前信息选择行动。证明了带有嘈杂行动可观测性的博弈总是存在均衡，且鉴定了其唯一性的必要条件。此外，嘈杂观察对追随者的最佳响应集合的基数有重要影响。

    

    本文研究了一类零和游戏的实例，其中一位玩家（领导者）承诺对手（追随者）通过对给定概率测度（策略）进行采样来选择行动。领导者的行动由追随者作为任意信道的输出观察到。作为回应，追随者根据其当前信息选择行动，即领导者的承诺和相应行动的嘈杂观察。在这种情况下，证明了带有嘈杂行动可观测性的博弈总是存在均衡，并且确定了其唯一性的必要条件。有趣的是，嘈杂观察对追随者的最佳响应集合的基数有重要影响。在特定条件下，这样的最佳响应集合几乎必然是单例集。所提出的模型可以捕捉到与勒贝格测度相对的任何信道噪声。作为例子，考虑了信道噪声与Lebesgue测度的密度有关的情况。

    This paper studies an instance of zero-sum games in which one player (the leader) commits to its opponent (the follower) to choose its actions by sampling a given probability measure (strategy). The actions of the leader are observed by the follower as the output of an arbitrary channel. In response to that, the follower chooses its action based on its current information, that is, the leader's commitment and the corresponding noisy observation of its action. Within this context, the equilibrium of the game with noisy action observability is shown to always exist and the necessary conditions for its uniqueness are identified. Interestingly, the noisy observations have important impact on the cardinality of the follower's set of best responses. Under particular conditions, such a set of best responses is proved to be a singleton almost surely. The proposed model captures any channel noise with a density with respect to the Lebesgue measure. As an example, the case in which the channel i
    
[^24]: 在线变分学习中的重要性采样

    Importance sampling for online variational learning

    [https://arxiv.org/abs/2402.02859](https://arxiv.org/abs/2402.02859)

    本文提出了一个在状态空间模型中用于在线学习平滑分布的高效算法，并在离线和真正的在线设置中展示了性能。

    

    本文研究了状态空间模型中的在线变分估计问题。我们专注于学习平滑分布，即给定观测的潜在状态的联合分布，采用变分方法和蒙特卡罗重要性采样。我们提出了一种在流数据情况下计算证据下界（ELBO）梯度的高效算法，其中观测值按顺序到达。我们的贡献包括一个计算效率高的在线ELBO估计器，在离线和真正的在线设置中展示了性能，并适用于计算关于联合平滑分布的一般期望。

    This article addresses online variational estimation in state-space models. We focus on learning the smoothing distribution, i.e. the joint distribution of the latent states given the observations, using a variational approach together with Monte Carlo importance sampling.  We propose an efficient algorithm for computing the gradient of the evidence lower bound (ELBO) in the context of streaming data, where observations arrive sequentially.  Our contributions include a computationally efficient online ELBO estimator, demonstrated performance in offline and true online settings, and adaptability for computing general expectations under joint smoothing distributions.
    
[^25]: 深度自回归密度网络与神经集合在基于模型的离线强化学习中的对比

    Deep autoregressive density nets vs neural ensembles for model-based offline reinforcement learning

    [https://arxiv.org/abs/2402.02858](https://arxiv.org/abs/2402.02858)

    本文对比了在基于模型的离线强化学习中，使用深度自回归密度网络和神经集合的方法。通过在D4RL基准测试上展示，我们质疑了使用神经集合的普遍观点，并发现单个良好校准的自回归模型可以获得更好的性能。同时，我们还分析了模型学习的静态指标，并得出了关于代理最终性能的重要模型特性。

    

    我们考虑仅有系统转换集合可用于策略优化的离线强化学习问题。在最近的研究进展中，我们考虑了一种基于模型的强化学习算法，该算法从可用数据中推断系统动态，并在模型推演上进行策略优化。这种方法容易受到模型误差的影响，可能会导致在真实系统上的灾难性失败。标准解决方案是依靠集合进行不确定性启发式，并避免在模型不确定性太大时利用模型。通过展示在D4RL基准测试上使用单个良好校准的自回归模型可以获得更好的性能，我们质疑了必须使用集合的普遍观点。我们还分析了与模型学习有关的静态指标，并得出了关于代理的最终性能的重要模型特性的结论。

    We consider the problem of offline reinforcement learning where only a set of system transitions is made available for policy optimization. Following recent advances in the field, we consider a model-based reinforcement learning algorithm that infers the system dynamics from the available data and performs policy optimization on imaginary model rollouts. This approach is vulnerable to exploiting model errors which can lead to catastrophic failures on the real system. The standard solution is to rely on ensembles for uncertainty heuristics and to avoid exploiting the model where it is too uncertain. We challenge the popular belief that we must resort to ensembles by showing that better performance can be obtained with a single well-calibrated autoregressive model on the D4RL benchmark. We also analyze static metrics of model-learning and conclude on the important model properties for the final performance of the agent.
    
[^26]: 偏态自适应随机逼近的非渐进分析

    Non-asymptotic Analysis of Biased Adaptive Stochastic Approximation

    [https://arxiv.org/abs/2402.02857](https://arxiv.org/abs/2402.02857)

    本文对于具有偏态梯度和自适应步长的SGD进行了全面的非渐进分析，证明了Adagrad和RMSProp算法在收敛速度上与无偏情况相似，并通过实验结果验证了收敛结果，展示了如何降低偏差的影响。

    

    自适应步长随机梯度下降（SGD）现在广泛用于训练深度神经网络。大多数理论结果假设可以获得无偏的梯度估计器，然而在一些最近的深度学习和强化学习应用中，使用了蒙特卡洛方法，却无法满足这一假设。本文对具有偏态梯度和自适应步长的SGD进行了全面的非渐进性分析，针对凸和非凸平滑函数。我们的研究包括时变偏差，并强调控制偏差和均方误差（MSE）梯度估计的重要性。特别地，我们证明了使用偏态梯度的Adagrad和RMSProp算法对于非凸平滑函数的收敛速度与文献中无偏情况下的结果相似。最后，我们提供了使用变分自动编码器（VAE）的实验结果，证明了我们的收敛结果，并展示了如何通过适当的方法降低偏差的影响。

    Stochastic Gradient Descent (SGD) with adaptive steps is now widely used for training deep neural networks. Most theoretical results assume access to unbiased gradient estimators, which is not the case in several recent deep learning and reinforcement learning applications that use Monte Carlo methods. This paper provides a comprehensive non-asymptotic analysis of SGD with biased gradients and adaptive steps for convex and non-convex smooth functions. Our study incorporates time-dependent bias and emphasizes the importance of controlling the bias and Mean Squared Error (MSE) of the gradient estimator. In particular, we establish that Adagrad and RMSProp with biased gradients converge to critical points for smooth non-convex functions at a rate similar to existing results in the literature for the unbiased case. Finally, we provide experimental results using Variational Autoenconders (VAE) that illustrate our convergence results and show how the effect of bias can be reduced by appropri
    
[^27]: 通过组合特征对齐增强组合通用性

    Enhancing Compositional Generalization via Compositional Feature Alignment

    [https://arxiv.org/abs/2402.02851](https://arxiv.org/abs/2402.02851)

    通过组合特征对齐，增强了模型的组合通用性，使其能够推广到未见过的领域-类别组合。

    

    机器学习模型在现实世界的应用中经常面临数据分布偏移的问题，即训练数据和测试数据分布之间存在差异。在常见的多领域多类别设置中，随着类别和领域数量的增加，很难为每个领域-类别组合收集训练数据。这个挑战自然地引发了对具备组合通用性（CG）能力的模型的探索，即模型可以推广到未见过的领域-类别组合。为了深入研究CG挑战，我们开发了CG-Bench，这是一套从现有实际图像数据集派生的CG基准测试，并观察到目前在基础模型（如CLIP和DINOv2）上流行的预训练-微调范式在这个挑战中存在困难。为了解决这个挑战，我们提出了组合特征对齐（CFA），这是一种简单的两阶段微调技术，它通过在预训练的编码器上学习两个正交线性头部来对齐类别和领域的标签。

    Real-world applications of machine learning models often confront data distribution shifts, wherein discrepancies exist between the training and test data distributions. In the common multi-domain multi-class setup, as the number of classes and domains scales up, it becomes infeasible to gather training data for every domain-class combination. This challenge naturally leads the quest for models with Compositional Generalization (CG) ability, where models can generalize to unseen domain-class combinations. To delve into the CG challenge, we develop CG-Bench, a suite of CG benchmarks derived from existing real-world image datasets, and observe that the prevalent pretraining-finetuning paradigm on foundational models, such as CLIP and DINOv2, struggles with the challenge. To address this challenge, we propose Compositional Feature Alignment (CFA), a simple two-stage finetuning technique that i) learns two orthogonal linear heads on a pretrained encoder with respect to class and domain lab
    
[^28]: 基于先处理、中处理和后处理的线性差异约束下的贝叶斯最优公平分类

    Bayes-Optimal Fair Classification with Linear Disparity Constraints via Pre-, In-, and Post-processing

    [https://arxiv.org/abs/2402.02817](https://arxiv.org/abs/2402.02817)

    本文提出了一种基于贝叶斯最优的公平分类方法，通过先处理、中处理和后处理来最小化分类错误，并在给定群体公平性约束的情况下进行优化。该方法引入了线性和双线性差异度量的概念，并找到了贝叶斯最优公平分类器的形式。本方法能够处理多个公平性约束和常见情况。

    

    机器学习算法可能对受保护的群体产生不公平的影响。为解决这个问题，我们开发了基于贝叶斯最优的公平分类方法，旨在在给定群体公平性约束的情况下最小化分类错误。我们引入了线性差异度量的概念，它们是概率分类器的线性函数；以及双线性差异度量，它们在群体回归函数方面也是线性的。我们证明了几种常见的差异度量（如人口平等、机会平等和预测平等）都是双线性的。我们通过揭示与Neyman-Pearson引理的连接，找到了在单一线性差异度量下的贝叶斯最优公平分类器的形式。对于双线性差异度量，贝叶斯最优公平分类器变成了群体阈值规则。我们的方法还可以处理多个公平性约束（如平等的几率）和受保护属性常见的情况。

    Machine learning algorithms may have disparate impacts on protected groups. To address this, we develop methods for Bayes-optimal fair classification, aiming to minimize classification error subject to given group fairness constraints. We introduce the notion of \emph{linear disparity measures}, which are linear functions of a probabilistic classifier; and \emph{bilinear disparity measures}, which are also linear in the group-wise regression functions. We show that several popular disparity measures -- the deviations from demographic parity, equality of opportunity, and predictive equality -- are bilinear.   We find the form of Bayes-optimal fair classifiers under a single linear disparity measure, by uncovering a connection with the Neyman-Pearson lemma. For bilinear disparity measures, Bayes-optimal fair classifiers become group-wise thresholding rules. Our approach can also handle multiple fairness constraints (such as equalized odds), and the common scenario when the protected attr
    
[^29]: 具有Koopman算子的全局超梯度估计

    Glocal Hypergradient Estimation with Koopman Operator

    [https://arxiv.org/abs/2402.02741](https://arxiv.org/abs/2402.02741)

    本文提出了一种具有Koopman算子的全局超梯度估计方法，通过使用局部超梯度的轨迹来高效地近似全局超梯度，实现了超参数的贪婪优化，兼具可靠性和效率。

    

    基于梯度的超参数优化方法使用超梯度来更新超参数，即元标准的梯度与超参数的关系。先前的研究使用两种不同的更新策略：一种是使用模型训练完成后得到的全局超梯度来优化超参数，另一种是使用每个模型更新之后得到的局部超梯度。虽然全局超梯度具有可靠性，但计算成本显著；相反，局部超梯度速度快但常常不是最优的。在本文中，我们提出了glocal超梯度估计，将“全局”的质量与“局部”的效率结合起来。为此，我们使用Koopman算子理论来线性化超梯度的动态，以便可以仅通过使用局部超梯度的轨迹来高效地近似全局超梯度。因此，我们可以使用估计的全局超梯度贪婪地优化超参数，同时实现可靠性和效率。

    Gradient-based hyperparameter optimization methods update hyperparameters using hypergradients, gradients of a meta criterion with respect to hyperparameters. Previous research used two distinct update strategies: optimizing hyperparameters using global hypergradients obtained after completing model training or local hypergradients derived after every few model updates. While global hypergradients offer reliability, their computational cost is significant; conversely, local hypergradients provide speed but are often suboptimal. In this paper, we propose glocal hypergradient estimation, blending "global" quality with "local" efficiency. To this end, we use the Koopman operator theory to linearize the dynamics of hypergradients so that the global hypergradients can be efficiently approximated only by using a trajectory of local hypergradients. Consequently, we can optimize hyperparameters greedily using estimated global hypergradients, achieving both reliability and efficiency simultaneo
    
[^30]: InVA: 综合变分自编码器用于多模态神经影像数据的协调

    InVA: Integrative Variational Autoencoder for Harmonization of Multi-modal Neuroimaging Data

    [https://arxiv.org/abs/2402.02734](https://arxiv.org/abs/2402.02734)

    InVA是一种综合变分自编码器方法，利用多模态神经影像数据中不同来源的多个图像来进行预测推理，相较于传统的VAE方法具有更好的效果。

    

    在探索多个来自不同成像模式的图像之间的非线性关联方面具有重要意义。尽管有越来越多的文献研究基于多个图像来推断图像的预测推理，但现有方法在有效借用多个成像模式之间的信息来预测图像方面存在局限。本文建立在变分自编码器（VAEs）的文献基础上，提出了一种新颖的方法，称为综合变分自编码器（InVA）方法，它从不同来源获得的多个图像中借用信息来绘制图像的预测推理。所提出的方法捕捉了结果图像与输入图像之间的复杂非线性关联，并允许快速计算。数值结果表明，InVA相对于通常不允许借用输入图像之间信息的VAE具有明显的优势。

    There is a significant interest in exploring non-linear associations among multiple images derived from diverse imaging modalities. While there is a growing literature on image-on-image regression to delineate predictive inference of an image based on multiple images, existing approaches have limitations in efficiently borrowing information between multiple imaging modalities in the prediction of an image. Building on the literature of Variational Auto Encoders (VAEs), this article proposes a novel approach, referred to as Integrative Variational Autoencoder (\texttt{InVA}) method, which borrows information from multiple images obtained from different sources to draw predictive inference of an image. The proposed approach captures complex non-linear association between the outcome image and input images, while allowing rapid computation. Numerical results demonstrate substantial advantages of \texttt{InVA} over VAEs, which typically do not allow borrowing information between input imag
    
[^31]: 折扣自适应在线预测

    Discounted Adaptive Online Prediction

    [https://arxiv.org/abs/2402.02720](https://arxiv.org/abs/2402.02720)

    本论文提出了一种折扣自适应在线预测算法，该算法适应于复杂的损失序列和比较器，并改进了非自适应算法。算法具有无需结构性假设的理论保证，并且在超参数调整方面具有鲁棒性。通过在线符合预测任务的实验证明了算法的好处。

    

    在线学习并不总是要记住一切。由于未来在统计上可能与过去有很大的不同，一个关键的挑战是在新数据到来时优雅地忘记历史。为了形式化这种直觉，我们运用最近发展的自适应在线学习技术重新思考了经典的折扣遗憾概念。我们的主要结果是一个新的算法，它适应于损失序列和比较器的复杂性，改进了广泛使用的非自适应算法-梯度下降算法，且具有恒定的学习率。特别地，我们的理论保证不需要任何结构性假设，只要求凸性，并且该算法经过证明对次优的超参数调整具有鲁棒性。我们进一步通过在线符合预测来展示这些好处，而在线符合预测是一个带有集合成员决策的下游在线学习任务。

    Online learning is not always about memorizing everything. Since the future can be statistically very different from the past, a critical challenge is to gracefully forget the history while new data comes in. To formalize this intuition, we revisit the classical notion of discounted regret using recently developed techniques in adaptive online learning. Our main result is a new algorithm that adapts to the complexity of both the loss sequence and the comparator, improving the widespread non-adaptive algorithm - gradient descent with a constant learning rate. In particular, our theoretical guarantee does not require any structural assumption beyond convexity, and the algorithm is provably robust to suboptimal hyperparameter tuning. We further demonstrate such benefits through online conformal prediction, a downstream online learning task with set-membership decisions.
    
[^32]: 理解影响视觉强化学习中泛化差距的因素：理论和实证证据

    Understanding What Affects Generalization Gap in Visual Reinforcement Learning: Theory and Empirical Evidence

    [https://arxiv.org/abs/2402.02701](https://arxiv.org/abs/2402.02701)

    本文通过理论和实证研究，揭示了在测试环境具有干扰因素时影响视觉强化学习中泛化差距的关键因素。结果表明，最小化训练和测试环境之间的表示距离是减少泛化差距最关键的因素。

    

    最近，有许多努力致力于在视觉强化学习中学习对连续控制有用的策略。在这种场景下，学习一个具有泛化能力的策略非常重要，因为测试环境可能与训练环境不同，例如在部署过程中存在干扰因素。许多实际算法被提出来解决这个问题。然而，据我们所知，它们中没有一种算法能够从理论上解释泛化差距的影响因素以及为什么他们的方法有效。在本文中，我们通过在测试环境具有干扰因素时理论上回答影响泛化差距的关键因素来解决这个问题。我们的理论表明，最小化训练和测试环境之间的表示距离（与人类直觉一致）对于减少泛化差距的效益至关重要。我们的理论结果得到了DM数据的实证证据的支持。

    Recently, there are many efforts attempting to learn useful policies for continuous control in visual reinforcement learning (RL). In this scenario, it is important to learn a generalizable policy, as the testing environment may differ from the training environment, e.g., there exist distractors during deployment. Many practical algorithms are proposed to handle this problem. However, to the best of our knowledge, none of them provide a theoretical understanding of what affects the generalization gap and why their proposed methods work. In this paper, we bridge this issue by theoretically answering the key factors that contribute to the generalization gap when the testing environment has distractors. Our theories indicate that minimizing the representation distance between training and testing environments, which aligns with human intuition, is the most critical for the benefit of reducing the generalization gap. Our theoretical results are supported by the empirical evidence in the DM
    
[^33]: 线性上下文马尔可夫决策过程的样本复杂性表征

    Sample Complexity Characterization for Linear Contextual MDPs

    [https://arxiv.org/abs/2402.02700](https://arxiv.org/abs/2402.02700)

    本文研究了线性上下文马尔可夫决策过程（CMDPs）的样本复杂性表征，并提出了两种模型的新颖算法，证明它们具有所需的多项式样本复杂性。其中，对于第一个模型，通过去除可达性假设，改进了现有结果。

    

    上下文马尔可夫决策过程（CMDPs）描述了一类强化学习问题，其中转移内核和奖励函数可以随时间变化，并由一个上下文变量索引的不同MDPs。虽然CMDPs作为一个重要的框架，可以模拟具有时变环境的许多实际应用，但它们在理论上很少有研究。在本文中，我们研究了两个线性函数逼近模型下的CMDPs：模型I具有上下文变化表示和所有上下文公共线性权重；以及模型II具有所有上下文的公共表示和上下文变化的线性权重。对于这两个模型，我们提出了新颖的基于模型的算法，并证明它们具有所需多项式样本复杂性的保证的ε-次优间隙。特别是，将我们对第一个模型的结果实例化为表格CMDP，通过去除可达性假设，改进了现有结果。我们对第二个模型的结果。

    Contextual Markov decision processes (CMDPs) describe a class of reinforcement learning problems in which the transition kernels and reward functions can change over time with different MDPs indexed by a context variable. While CMDPs serve as an important framework to model many real-world applications with time-varying environments, they are largely unexplored from theoretical perspective. In this paper, we study CMDPs under two linear function approximation models: Model I with context-varying representations and common linear weights for all contexts; and Model II with common representations for all contexts and context-varying linear weights. For both models, we propose novel model-based algorithms and show that they enjoy guaranteed $\epsilon$-suboptimality gap with desired polynomial sample complexity. In particular, instantiating our result for the first model to the tabular CMDP improves the existing result by removing the reachability assumption. Our result for the second mode
    
[^34]: 深度均衡模型与高维高斯混合模型中不太深的显式模型几乎等价

    Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High-dimensional Gaussian Mixtures

    [https://arxiv.org/abs/2402.02697](https://arxiv.org/abs/2402.02697)

    本文通过对深度均衡模型和显式神经网络模型进行理论分析和实验证明，在高维高斯混合数据下，可以通过设计浅显式网络来实现与给定深度均衡模型相同的特征光谱行为。

    

    深度均衡模型（DEQs）作为典型的隐式神经网络，在各种任务上取得了显着的成功。然而，我们对隐式DEQ和显式神经网络模型之间的连接和差异缺乏理论上的理解。在本文中，我们借鉴最近在随机矩阵理论方面的进展，对高维高斯混合模型输入数据下，隐式DEQ的共轭核（CK）和神经切向核（NTK）矩阵的特征光谱进行了深入分析。我们在这个设置中证明了这些隐式-CKs和NTKs的光谱行为取决于DEQ激活函数和初始权重方差，但仅通过一组四个非线性方程。作为这一理论结果的直接影响，我们证明可以精心设计一个浅显式网络来产生与给定DEQ相同的CK或NTK。尽管这里是针对高斯混合数据推导的，经验结果表明

    Deep equilibrium models (DEQs), as a typical implicit neural network, have demonstrated remarkable success on various tasks. There is, however, a lack of theoretical understanding of the connections and differences between implicit DEQs and explicit neural network models. In this paper, leveraging recent advances in random matrix theory (RMT), we perform an in-depth analysis on the eigenspectra of the conjugate kernel (CK) and neural tangent kernel (NTK) matrices for implicit DEQs, when the input data are drawn from a high-dimensional Gaussian mixture. We prove, in this setting, that the spectral behavior of these Implicit-CKs and NTKs depend on the DEQ activation function and initial weight variances, but only via a system of four nonlinear equations. As a direct consequence of this theoretical result, we demonstrate that a shallow explicit network can be carefully designed to produce the same CK or NTK as a given DEQ. Despite derived here for Gaussian mixture data, empirical results 
    
[^35]: 使用图神经网络的链接预测的统计保证

    Statistical Guarantees for Link Prediction using Graph Neural Networks

    [https://arxiv.org/abs/2402.02692](https://arxiv.org/abs/2402.02692)

    本文提出了一种线性图神经网络（LG-GNN）架构，通过计算边缘概率来预测图中的链接，并推导了其在链接预测任务中的性能统计保证。这种架构对于稀疏和稠密图都适用，并在真实和合成数据集上验证了其优势。

    

    本文针对由图上生成的图网络中的链接预测任务，推导了图神经网络（GNN）性能的统计保证。我们提出了一个线性GNN架构（LG-GNN），可以产生对潜在边缘概率的一致估计。我们对均方误差进行了界定，并对LG-GNN在检测高概率边缘的能力给出了保证。我们的保证适用于稀疏和稠密图。最后，我们展示了经典GCN架构的一些缺点，并在真实和合成数据集上验证了我们的结果。

    This paper derives statistical guarantees for the performance of Graph Neural Networks (GNNs) in link prediction tasks on graphs generated by a graphon. We propose a linear GNN architecture (LG-GNN) that produces consistent estimators for the underlying edge probabilities. We establish a bound on the mean squared error and give guarantees on the ability of LG-GNN to detect high-probability edges. Our guarantees hold for both sparse and dense graphs. Finally, we demonstrate some of the shortcomings of the classical GCN architecture, as well as verify our results on real and synthetic datasets.
    
[^36]: 计算上的公正并非人口统计数据的平等，以及其他观察

    Counterfactual Fairness Is Not Demographic Parity, and Other Observations

    [https://arxiv.org/abs/2402.02663](https://arxiv.org/abs/2402.02663)

    这里是中文总结出的一句话要点：文章探讨了因果概念与纯粹概率概念之间的等价性，并发现计算上的公正并不等同于人口统计数据的平等。同时还纠正了一些有关计算上的公正的误解。

    

    需要谨慎对待在因果概念与纯粹概率概念之间进行等价性的断言。在本简短的文章中，我对最近一个声称计算上的公正等同于人口统计数据的平等的主张进行了审查。仔细研究后发现该主张不成立。我将借此机会解决一些关于计算上的公正的更广泛误解。

    Blanket statements of equivalence between causal concepts and purely probabilistic concepts should be approached with care. In this short note, I examine a recent claim that counterfactual fairness is equivalent to demographic parity. The claim fails to hold up upon closer examination. I will take the opportunity to address some broader misunderstandings about counterfactual fairness.
    
[^37]: 通过状态扩展和随机排列的方法进行变分DAG估计

    Variational DAG Estimation via State Augmentation With Stochastic Permutations

    [https://arxiv.org/abs/2402.02644](https://arxiv.org/abs/2402.02644)

    使用状态扩展和随机排列进行变分DAG估计的方法可以超越竞争的贝叶斯和非贝叶斯基准方法，从而在估计贝叶斯网络结构方面取得更好的性能。

    

    从观测数据中估计贝叶斯网络的结构，即有向无环图（DAG），是一个在统计和计算上都很困难的问题，在因果发现等领域有着重要应用。贝叶斯方法在解决这个任务方面是一个有希望的方向，因为它们允许进行不确定性量化，并处理众所周知的可识别性问题。从概率推断的角度来看，主要的挑战是（i）表示满足DAG约束的图的分布和（ii）估计底层组合空间的后验概率。我们提出了一种方法，通过在DAG和排列的扩展空间上构建联合分布来解决这些挑战。我们通过变分推断进行后验估计，在其中利用了离散分布的连续松弛。我们展示了我们的方法在一系列合成和实际数据上能够超越竞争的贝叶斯和非贝叶斯基准方法。

    Estimating the structure of a Bayesian network, in the form of a directed acyclic graph (DAG), from observational data is a statistically and computationally hard problem with essential applications in areas such as causal discovery. Bayesian approaches are a promising direction for solving this task, as they allow for uncertainty quantification and deal with well-known identifiability issues. From a probabilistic inference perspective, the main challenges are (i) representing distributions over graphs that satisfy the DAG constraint and (ii) estimating a posterior over the underlying combinatorial space. We propose an approach that addresses these challenges by formulating a joint distribution on an augmented space of DAGs and permutations. We carry out posterior estimation via variational inference, where we exploit continuous relaxations of discrete distributions. We show that our approach can outperform competitive Bayesian and non-Bayesian benchmarks on a range of synthetic and re
    
[^38]: $C^*$-代数机器学习：迈向新的方向

    $C^*$-Algebraic Machine Learning: Moving in a New Direction

    [https://arxiv.org/abs/2402.02637](https://arxiv.org/abs/2402.02637)

    $C^*$-代数机器学习是将$C^*$-代数与机器学习结合的新研究方向，它通过统一现有的学习策略，并构建更多元化和信息丰富的数据模型的新框架，为机器学习提供了一种新的方法。

    

    机器学习与数学的几个领域（如统计学、概率论和线性代数）有着长期的合作传统。我们提出了机器学习研究的一个新方向：$C^*$-代数机器学习，这是$C^*$-代数和机器学习之间的交流和相互滋养。$C^*$-代数是复数空间的自然推广的数学概念，它使我们能够统一现有的学习策略，并构建一个更多元化和信息丰富的数据模型的新框架。我们解释了在机器学习中使用$C^*$-代数的原因和方法，并提供了在核方法和神经网络背景下设计$C^*$-代数学习模型的技术考虑。此外，我们讨论了$C^*$-代数机器学习中的开放问题和挑战，并提出了我们对未来发展和应用的思考。

    Machine learning has a long collaborative tradition with several fields of mathematics, such as statistics, probability and linear algebra. We propose a new direction for machine learning research: $C^*$-algebraic ML $-$ a cross-fertilization between $C^*$-algebra and machine learning. The mathematical concept of $C^*$-algebra is a natural generalization of the space of complex numbers. It enables us to unify existing learning strategies, and construct a new framework for more diverse and information-rich data models. We explain why and how to use $C^*$-algebras in machine learning, and provide technical considerations that go into the design of $C^*$-algebraic learning models in the contexts of kernel methods and neural networks. Furthermore, we discuss open questions and challenges in $C^*$-algebraic ML and give our thoughts for future development and applications.
    
[^39]: 一种新的处理不确定性概率的方法

    A new approach for imprecise probabilities

    [https://arxiv.org/abs/2402.02556](https://arxiv.org/abs/2402.02556)

    本论文引入了一种新的区间概率测度的概念，用于表示不确定性概率。通过特征化一类广泛的区间概率测度，建立了更新规则，并提出了随机优势的定义。此外，还给出了凯恩斯-兰姆齐争论的正式解决方案。

    

    该论文引入了一种新的区间概率测度的概念，以一种自然和连贯的方式表示不确定性或不精确概率。在一个集合的代数中，我们引入了一个弱补充的概念，记为$\psi$。事件$H$的区间概率测度定义为与不确定性事件集合$(\psi(H))^c$相关的标准补集$H^c$。我们对一类广泛的区间概率测度进行了特征化，并定义了它们的特性。此外，我们建立了一个更新规则，考虑了统计独立性和依赖性的概念。我们还给出了随机变量的区间分布的公式，引入了两个随机变量之间的随机优势的相应定义。作为副产品，我们提出了一个对凯恩斯-兰姆齐争论的正式解决方案。

    This paper introduces a novel concept of interval probability measures that enables the representation of imprecise probabilities, or uncertainty, in a natural and coherent manner. Within an algebra of sets, we introduce a notion of weak complementation denoted as $\psi$. The interval probability measure of an event $H$ is defined with respect to the set of indecisive eventualities $(\psi(H))^c$, which is included in the standard complement $H^c$.   We characterize a broad class of interval probability measures and define their properties. Additionally, we establish an updating rule with respect to $H$, incorporating concepts of statistical independence and dependence. The interval distribution of a random variable is formulated, and a corresponding definition of stochastic dominance between two random variables is introduced. As a byproduct, a formal solution to the century-old Keynes-Ramsey controversy is presented.
    
[^40]: 一种快速的Lasso和Logistic Lasso方法

    A Fast Method for Lasso and Logistic Lasso

    [https://arxiv.org/abs/2402.02463](https://arxiv.org/abs/2402.02463)

    本论文提出了一种快速解决Lasso和Logistic Lasso问题的方法，通过采用主动集方法和适当的求解器，成功实现了加速。在压缩感知、Lasso回归和Logistic Lasso回归实验中，与传统方法相比，我们的方法平均能提高约30倍的速度。

    

    我们提出了一种快速解决压缩感知、Lasso回归和Logistic Lasso回归问题的方法，该方法使用主动集方法迭代运行适当的求解器。我们设计了一种更新主动集的策略，相比于单次调用多个求解器（包括Matlab的lassoglm和glmnet以及用于稀疏重构的梯度投影算法GPSR），能够实现大幅加速。对于压缩感知，我们的方法与GPSR的混合平均速度提高了31.41倍（对于高斯系列）和25.64倍（对于二进制系列）。在Lasso回归实验中，我们的方法与GPSR的混合平均速度提高了30.67倍。在Logistic Lasso回归的实验中，我们的方法与lassoglm的混合平均速度提高了11.95倍，与glmnet的混合平均速度提高了1.40倍。

    We propose a fast method for solving compressed sensing, Lasso regression, and Logistic Lasso regression problems that iteratively runs an appropriate solver using an active set approach. We design a strategy to update the active set that achieves a large speedup over a single call of several solvers, including gradient projection for sparse reconstruction (GPSR), lassoglm of Matlab, and glmnet. For compressed sensing, the hybrid of our method and GPSR is 31.41 times faster than GPSR on average for Gaussian ensembles and 25.64 faster on average for binary ensembles. For Lasso regression, the hybrid of our method and GPSR achieves a 30.67-fold average speedup in our experiments. In our experiments on Logistic Lasso regression, the hybrid of our method and lassoglm gives an 11.95-fold average speedup, and the hybrid of our method and glmnet gives a 1.40-fold average speedup.
    
[^41]: 在最小化迹因子分析中 - 这首老歌以新的方式演唱

    On Minimum Trace Factor Analysis - An Old Song Sung to a New Tune

    [https://arxiv.org/abs/2402.02459](https://arxiv.org/abs/2402.02459)

    本文提出了最小化迹因子分析（MTFA）的放松版本，该方法能够有效降低因异方差噪声造成的过拟合问题，并解决了在因子分析和谱方法中常见的异常情况和病态诅咒问题。

    

    维度降低方法，例如主成分分析（PCA）和因子分析，在数据科学中是很常用的。然而，对于具有显著异方差噪声的数据，寻找稳健的低维逼近存在明显且被广泛接受的挑战。本文介绍了最小化迹因子分析（MTFA）的放松版本，这是一种凸优化方法，其根源可以追溯到1940年Ledermann的工作。这种放松方法在不过度拟合异方差扰动方面特别有效，解决了因素分析中经常被提到的Heywood案例和最近发现的现有谱方法中"病态诅咒"问题。我们在所得低秩子空间的精确度和所提算法的收敛速度上提供了理论保证。我们发现了与现有方法（包括HeteroPCA，Lasso和Soft-Impute）的一些有趣联系，以填补一些空白。

    Dimensionality reduction methods, such as principal component analysis (PCA) and factor analysis, are central to many problems in data science. There are, however, serious and well-understood challenges to finding robust low dimensional approximations for data with significant heteroskedastic noise. This paper introduces a relaxed version of Minimum Trace Factor Analysis (MTFA), a convex optimization method with roots dating back to the work of Ledermann in 1940. This relaxation is particularly effective at not overfitting to heteroskedastic perturbations and addresses the commonly cited Heywood cases in factor analysis and the recently identified "curse of ill-conditioning" for existing spectral methods. We provide theoretical guarantees on the accuracy of the resulting low rank subspace and the convergence rate of the proposed algorithm to compute that matrix. We develop a number of interesting connections to existing methods, including HeteroPCA, Lasso, and Soft-Impute, to fill an i
    
[^42]: FreDF: 在频域中学习预测

    FreDF: Learning to Forecast in Frequency Domain

    [https://arxiv.org/abs/2402.02399](https://arxiv.org/abs/2402.02399)

    FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。

    

    时间序列建模在历史序列和标签序列中都面临自相关的挑战。当前的研究主要集中在处理历史序列中的自相关问题，但往往忽视了标签序列中的自相关存在。具体来说，新兴的预测模型主要遵循直接预测（DF）范式，在标签序列中假设条件独立性下生成多步预测。这种假设忽视了标签序列中固有的自相关性，从而限制了基于DF的模型的性能。针对这一问题，我们引入了频域增强直接预测（FreDF），通过在频域中学习预测来避免标签自相关的复杂性。我们的实验证明，FreDF在性能上大大超过了包括iTransformer在内的现有最先进方法，并且与各种预测模型兼容。

    Time series modeling is uniquely challenged by the presence of autocorrelation in both historical and label sequences. Current research predominantly focuses on handling autocorrelation within the historical sequence but often neglects its presence in the label sequence. Specifically, emerging forecast models mainly conform to the direct forecast (DF) paradigm, generating multi-step forecasts under the assumption of conditional independence within the label sequence. This assumption disregards the inherent autocorrelation in the label sequence, thereby limiting the performance of DF-based models. In response to this gap, we introduce the Frequency-enhanced Direct Forecast (FreDF), which bypasses the complexity of label autocorrelation by learning to forecast in the frequency domain. Our experiments demonstrate that FreDF substantially outperforms existing state-of-the-art methods including iTransformer and is compatible with a variety of forecast models.
    
[^43]: Stereographic Spherical Sliced Wasserstein Distances - 应用于球形概率分布比较的立体投影球面切片瓦瑟斯坦距离

    Stereographic Spherical Sliced Wasserstein Distances

    [https://arxiv.org/abs/2402.02345](https://arxiv.org/abs/2402.02345)

    本文提出了一种快速且高度并行的用于比较球形测度的距离，使用了立体投影和广义Radon变换，称之为立体投影球面切片瓦瑟斯坦（S3W）距离。通过仔细处理立体投影引起的距离畸变，并进行了理论分析，证明了该方法在速度和效果上的优势。

    

    在地质学、医学领域、计算机视觉和深度表示学习等各个领域，比较球形概率分布是非常重要的。基于最优传输的距离，比如瓦瑟斯坦距离，对于比较概率测度已经引发了活跃的研究，以开发计算效率高的球形概率测度的变体。本文介绍了一种高速且高度并行化的用于比较球形测度的距离，使用了立体投影和广义Radon变换，我们称之为立体投影球面切片瓦瑟斯坦（S3W）距离。我们仔细处理了立体投影引起的距离畸变，并对我们提出的度量及其具有旋转不变性的变体进行了广泛的理论分析。最后，我们评估了所提出的度量的性能，并将其与最近的基线进行了比较，从遥感和处理效率两个方面进行了评估。

    Comparing spherical probability distributions is of great interest in various fields, including geology, medical domains, computer vision, and deep representation learning. The utility of optimal transport-based distances, such as the Wasserstein distance, for comparing probability measures has spurred active research in developing computationally efficient variations of these distances for spherical probability measures. This paper introduces a high-speed and highly parallelizable distance for comparing spherical measures using the stereographic projection and the generalized Radon transform, which we refer to as the Stereographic Spherical Sliced Wasserstein (S3W) distance. We carefully address the distance distortion caused by the stereographic projection and provide an extensive theoretical analysis of our proposed metric and its rotationally invariant variation. Finally, we evaluate the performance of the proposed metrics and compare them with recent baselines in terms of both spe
    
[^44]: 弹性贝叶斯g形式在具有时变混杂的因果生存分析中的应用

    A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding

    [https://arxiv.org/abs/2402.02306](https://arxiv.org/abs/2402.02306)

    本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。

    

    在具有时间至事件结果的纵向观察性研究中，因果分析的常见目标是在研究群体中估计在假设干预情景下的因果生存曲线。g形式是这种分析的一个特别有用的工具。为了增强传统的参数化g形式方法，我们开发了一种更灵活的贝叶斯g形式估计器。该估计器同时支持纵向预测和因果推断。它在模拟时变生成组件的建模中引入了贝叶斯附加回归树，旨在减轻由于模型错误规范造成的偏差。具体而言，我们引入了一类更通用的离散生存数据g形式。这些公式可以引入纵向平衡分数，这在处理越来越多的时变混杂因素时是一种有效的降维方法。

    In longitudinal observational studies with a time-to-event outcome, a common objective in causal analysis is to estimate the causal survival curve under hypothetical intervention scenarios within the study cohort. The g-formula is a particularly useful tool for this analysis. To enhance the traditional parametric g-formula approach, we developed a more adaptable Bayesian g-formula estimator. This estimator facilitates both longitudinal predictive and causal inference. It incorporates Bayesian additive regression trees in the modeling of the time-evolving generative components, aiming to mitigate bias due to model misspecification. Specifically, we introduce a more general class of g-formulas for discrete survival data. These formulas can incorporate the longitudinal balancing scores, which serve as an effective method for dimension reduction and are vital when dealing with an expanding array of time-varying confounders. The minimum sufficient formulation of these longitudinal balancing
    
[^45]: 球形数据的拟合度和聚类：R和Python中的QuadratiK软件包

    Goodness-of-Fit and Clustering of Spherical Data: the QuadratiK package in R and Python

    [https://arxiv.org/abs/2402.02290](https://arxiv.org/abs/2402.02290)

    QuadratiK软件包是一个在R和Python中实现的数据分析工具，它提供了一套全面的拟合度测试和基于核方法的聚类技术，特别适用于处理球形数据。

    

    我们介绍了QuadratiK软件包，该软件包包含了创新的数据分析方法。该软件包在R和Python中实现，提供了一套全面的适应度拟合测试和基于核方法的二次距离的聚类技术，从而弥合了统计学和机器学习文献之间的差距。我们的软件实现了单样本、双样本和k样本适应度拟合测试，提供了一种高效且数学上合理的方法来评估概率分布的拟合度。我们的软件扩展了功能，包括基于泊松核密度的$d$维球上均匀性测试，以及从泊松核密度中生成随机样本的算法。特别值得注意的是，我们的软件还包括一种针对球形数据而特别量身定制的独特聚类算法，该算法利用了球面上基于泊松核密度的混合模型。同时，我们的软件还包括其他图形功能。

    We introduce the QuadratiK package that incorporates innovative data analysis methodologies. The presented software, implemented in both R and Python, offers a comprehensive set of goodness-of-fit tests and clustering techniques using kernel-based quadratic distances, thereby bridging the gap between the statistical and machine learning literatures. Our software implements one, two and k-sample tests for goodness of fit, providing an efficient and mathematically sound way to assess the fit of probability distributions. Expanded capabilities of our software include supporting tests for uniformity on the $d$-dimensional Sphere based on Poisson kernel densities, and algorithms for generating random samples from Poisson kernel densities. Particularly noteworthy is the incorporation of a unique clustering algorithm specifically tailored for spherical data that leverages a mixture of Poisson-kernel-based densities on the sphere. Alongside this, our software includes additional graphical func
    
[^46]: 图机器学习基础的未来方向

    Future Directions in Foundations of Graph Machine Learning

    [https://arxiv.org/abs/2402.02287](https://arxiv.org/abs/2402.02287)

    图机器学习领域的未来方向应该是发展一个更加均衡的理论，从更完整的角度探究图神经网络的表达能力、泛化和优化之间的相互关系。

    

    随着图数据在不同学科（从生命科学到社会科学和工程科学）上的广泛应用，图机器学习，尤其是使用图神经网络（GNNs），引起了人们浓厚的兴趣。尽管在实际应用中取得了成功，但我们对GNNs性质的理论理解仍然非常不完整。最近的理论发展主要集中在阐明GNNs粗粒度表达能力方面，主要采用组合技巧。然而，这些研究与实践并不完全一致，特别是在使用随机一阶优化技术训练GNNs时，对GNNs的泛化行为的理解。在这篇定位论文中，我们认为图机器学习领域需要将注意力转移到发展一个更加均衡的图机器学习理论上来，重点关注表达能力、泛化和优化的相互关系的更全面的理解。

    Machine learning on graphs, especially using graph neural networks (GNNs), has seen a surge in interest due to the wide availability of graph data across a broad spectrum of disciplines, from life to social and engineering sciences. Despite their practical success, our theoretical understanding of the properties of GNNs remains highly incomplete. Recent theoretical advancements primarily focus on elucidating the coarse-grained expressive power of GNNs, predominantly employing combinatorial techniques. However, these studies do not perfectly align with practice, particularly in understanding the generalization behavior of GNNs when trained with stochastic first-order optimization techniques. In this position paper, we argue that the graph machine learning community needs to shift its attention to developing a more balanced theory of graph machine learning, focusing on a more thorough understanding of the interplay of expressive power, generalization, and optimization.
    
[^47]: 因果贝叶斯优化通过外源分布学习

    Causal Bayesian Optimization via Exogenous Distribution Learning

    [https://arxiv.org/abs/2402.02277](https://arxiv.org/abs/2402.02277)

    本文引入了一种新的方法，通过学习外源变量的分布，提高了结构化因果模型的近似精度，并将因果贝叶斯优化扩展到更一般的因果方案。

    

    在结构化因果模型中，将目标变量最大化作为操作目标是一个重要的问题。现有的因果贝叶斯优化（CBO）方法要么依赖于改变因果结构以最大化奖励的硬干预，要么引入动作节点到内生变量中，以调整数据生成机制以实现目标。本文引入了一种新的方法来学习外源变量的分布，这在现有方法中通常被忽略或通过期望进行边缘化。外源分布学习提高了通常通过有限观测数据训练的代理模型中的结构化因果模型的近似精度。此外，学习到的外源分布将现有的CBO扩展到超出加性噪声模型（ANM）的一般因果方案。恢复外源变量使我们能够为噪声或未观测到的隐藏变量使用更灵活的先验。引入了一种新的CBO方法。

    Maximizing a target variable as an operational objective in a structured causal model is an important problem. Existing Causal Bayesian Optimization (CBO) methods either rely on hard interventions that alter the causal structure to maximize the reward; or introduce action nodes to endogenous variables so that the data generation mechanisms are adjusted to achieve the objective. In this paper, a novel method is introduced to learn the distribution of exogenous variables, which is typically ignored or marginalized through expectation by existing methods.   Exogenous distribution learning improves the approximation accuracy of structured causal models in a surrogate model that is usually trained with limited observational data. Moreover, the learned exogenous distribution extends existing CBO to general causal schemes beyond Additive Noise Models (ANM). The recovery of exogenous variables allows us to use a more flexible prior for noise or unobserved hidden variables. A new CBO method is 
    
[^48]: 对于具有任意度量的有限通道，失真感知权衡的特征化研究

    Characterization of the Distortion-Perception Tradeoff for Finite Channels with Arbitrary Metrics

    [https://arxiv.org/abs/2402.02265](https://arxiv.org/abs/2402.02265)

    本研究中，我们对于具有任意度量的有限通道中的失真感知权衡进行了研究，发现计算失真感知函数和最优重建相当于求解一组线性规划问题，并提供了失真感知权衡的结构特征化和二进制源的闭式表达式。

    

    在人类检查时，重建信号与真实信号不应有区别。通常情况下，这种高感知质量的实现需要付出高重建误差的代价，反之亦然。我们对于具有任意度量的有限字母表通道中的失真感知权衡进行研究，将感知指数定义为沃瑟斯坦距离-$1$，将失真矩阵定义为任意。在这个设定下，我们证明了计算失真感知函数和最优重建等价于求解一组线性规划问题。我们进一步对失真感知权衡进行了结构特征化，其中失真感知函数在感知指数上是分段线性的。对于二进制源，我们还导出了闭式表达式。

    Whenever inspected by humans, reconstructed signals should not be distinguished from real ones. Typically, such a high perceptual quality comes at the price of high reconstruction error, and vice versa. We study this distortion-perception (DP) tradeoff over finite-alphabet channels, for the Wasserstein-$1$ distance induced by a general metric as the perception index, and an arbitrary distortion matrix. Under this setting, we show that computing the DP function and the optimal reconstructions is equivalent to solving a set of linear programming problems. We provide a structural characterization of the DP tradeoff, where the DP function is piecewise linear in the perception index. We further derive a closed-form expression for the case of binary sources.
    
[^49]: 分布约简：用格罗莫夫-瓦瑟斯坦投影统一降维和聚类

    Distributional Reduction: Unifying Dimensionality Reduction and Clustering with Gromov-Wasserstein Projection

    [https://arxiv.org/abs/2402.02239](https://arxiv.org/abs/2402.02239)

    本文提出了一种新的分布约简方法，利用格罗莫夫-瓦瑟斯坦投影统一了降维和聚类，通过优化问题同时解决降维和聚类，实验证明了该方法在多个领域表现出卓越性能。

    

    无监督学习旨在捕捉潜在的大规模和高维数据集的结构。传统上，这涉及使用降维方法将数据投影到可解释的空间上，或将数据点组织成有意义的聚类。在实践中，这些方法通常是按顺序使用的，而不能保证聚类与降维相一致。在这项工作中，我们提出了一个新的观点：使用分布。通过利用最优输运的工具，特别是格罗莫夫-瓦瑟斯坦距离，我们将聚类和降维统一为一个称为分布约简的单一框架。这使我们能够通过单个优化问题同时解决聚类和降维。通过全面的实验证明了我们方法的多功能性和解释性，并表明它在各种图像和基因组数据集上优于现有方法。

    Unsupervised learning aims to capture the underlying structure of potentially large and high-dimensional datasets. Traditionally, this involves using dimensionality reduction methods to project data onto interpretable spaces or organizing points into meaningful clusters. In practice, these methods are used sequentially, without guaranteeing that the clustering aligns well with the conducted dimensionality reduction. In this work, we offer a fresh perspective: that of distributions. Leveraging tools from optimal transport, particularly the Gromov-Wasserstein distance, we unify clustering and dimensionality reduction into a single framework called distributional reduction. This allows us to jointly address clustering and dimensionality reduction with a single optimization problem. Through comprehensive experiments, we highlight the versatility and interpretability of our method and show that it outperforms existing approaches across a variety of image and genomics datasets.
    
[^50]: 在组合优化问题中寻找多样化解决方案的连续张量放松方法

    Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems

    [https://arxiv.org/abs/2402.02190](https://arxiv.org/abs/2402.02190)

    本研究提出了连续张量放松方法(CTRA)，用于在组合优化问题中寻找多样化的解决方案。CTRA通过对离散决策变量进行连续放松，解决了寻找多样化解决方案的挑战。

    

    在组合优化问题中，寻找最佳解是最常见的目标。然而，在实际场景中，单一解决方案可能不适用，因为目标函数和约束条件只是原始现实世界情况的近似值。为了解决这个问题，寻找具有不同特征的多样化解决方案和约束严重性的变化成为自然的方向。这种策略提供了在后处理过程中选择合适解决方案的灵活性。然而，发现这些多样化解决方案比确定单一解决方案更具挑战性。为了克服这一挑战，本研究引入了连续张量松弛退火 (CTRA) 方法，用于基于无监督学习的组合优化求解器。CTRA通过扩展连续松弛方法，将离散决策变量转换为连续张量，同时解决了多个问题。该方法找到了不同特征的多样化解决方案和约束严重性的变化。

    Finding the best solution is the most common objective in combinatorial optimization (CO) problems. However, a single solution may not be suitable in practical scenarios, as the objective functions and constraints are only approximations of original real-world situations. To tackle this, finding (i) "heterogeneous solutions", diverse solutions with distinct characteristics, and (ii) "penalty-diversified solutions", variations in constraint severity, are natural directions. This strategy provides the flexibility to select a suitable solution during post-processing. However, discovering these diverse solutions is more challenging than identifying a single solution. To overcome this challenge, this study introduces Continual Tensor Relaxation Annealing (CTRA) for unsupervised-learning-based CO solvers. CTRA addresses various problems simultaneously by extending the continual relaxation approach, which transforms discrete decision variables into continual tensors. This method finds heterog
    
[^51]: 通过优化抽象的方式进行Slate Bandit策略的离策略评估

    Off-Policy Evaluation of Slate Bandit Policies via Optimizing Abstraction

    [https://arxiv.org/abs/2402.02171](https://arxiv.org/abs/2402.02171)

    我们提出了一种名为潜在IPS（LIPS）的新的Slate Bandit OPE估计器，通过在低维度的Slate抽象空间中定义重要性权重，并通过数据驱动的方式优化Slate抽象来减小偏差和方差。

    

    我们研究了Slate上下文强盗问题中的离策略评估（OPE），其中一个策略选择称为slates的多维动作。这个问题在推荐系统、搜索引擎、营销以及医疗应用中广泛存在，然而，由于动作空间大，典型的逆倾向评分（IPS）估计器存在较大的方差，使得有效的OPE成为一个重大挑战。伪逆（PI）估计器已被引入以减小方差问题，通过假设奖励函数线性，但这可能导致显著的偏差，因为这个假设在观测数据中很难验证并且经常会被实质性违反。为了解决之前估计器的局限性，我们开发了一种新的Slate Bandit OPE估计器，称为潜在IPS（LIPS），它在低维度的Slate抽象空间中定义了重要性权重，我们通过数据驱动的方式优化Slate抽象来最小化LIPS的偏差和方差。

    We study off-policy evaluation (OPE) in the problem of slate contextual bandits where a policy selects multi-dimensional actions known as slates. This problem is widespread in recommender systems, search engines, marketing, to medical applications, however, the typical Inverse Propensity Scoring (IPS) estimator suffers from substantial variance due to large action spaces, making effective OPE a significant challenge. The PseudoInverse (PI) estimator has been introduced to mitigate the variance issue by assuming linearity in the reward function, but this can result in significant bias as this assumption is hard-to-verify from observed data and is often substantially violated. To address the limitations of previous estimators, we develop a novel estimator for OPE of slate bandits, called Latent IPS (LIPS), which defines importance weights in a low-dimensional slate abstraction space where we optimize slate abstractions to minimize the bias and variance of LIPS in a data-driven way. By do
    
[^52]: 一个贝叶斯聚类有效性指数

    A Bayesian cluster validity index

    [https://arxiv.org/abs/2402.02162](https://arxiv.org/abs/2402.02162)

    该论文提出了一个基于贝叶斯方法的聚类有效性指数，该指数根据现有的基础指数定义，并用于检测次优聚类数，通过与其他指数进行比较，验证了其有效性。

    

    在应用聚类算法时，选择聚类数是关键步骤之一。为了完成这个任务，引入了各种聚类有效性指数（CVIs）。大多数聚类有效性指数都被定义为检测数据集中隐藏的最优聚类数。然而，用户有时并不期望获得最优聚类数，而是更适合他们应用的次优聚类数。这促使我们引入了一种基于现有基础指数的贝叶斯聚类有效性指数（BCVI）。该指数基于狄利克雷或广义狄利克雷先验定义，得到相同的后验分布。然后我们基于Wiroonsri指数（WI）和Wiroonsri-Preedasawakul指数（WP）作为硬聚类和软聚类的基础指数来测试我们的BCVI。我们将它们的结果与原始的基础指数以及一些其他存在的CVIs（包括Davies and Bouldin (DB)，Starczewski (STR)）进行比较。

    Selecting the number of clusters is one of the key processes when applying clustering algorithms. To fulfill this task, various cluster validity indices (CVIs) have been introduced. Most of the cluster validity indices are defined to detect the optimal number of clusters hidden in a dataset. However, users sometimes do not expect to get the optimal number of groups but a secondary one which is more reasonable for their applications. This has motivated us to introduce a Bayesian cluster validity index (BCVI) based on existing underlying indices. This index is defined based on either Dirichlet or Generalized Dirichlet priors which result in the same posterior distribution. Our BCVI is then tested based on the Wiroonsri index (WI), and the Wiroonsri-Preedasawakul index (WP) as underlying indices for hard and soft clustering, respectively. We compare their outcomes with the original underlying indices, as well as a few more existing CVIs including Davies and Bouldin (DB), Starczewski (STR)
    
[^53]: 论文题目：为什么“摸着石头过河”方法主导推荐系统实践；呼吁摒弃反乌托邦思维

    Position Paper: Why the Shooting in the Dark Method Dominates Recommender Systems Practice; A Call to Abandon Anti-Utopian Thinking

    [https://arxiv.org/abs/2402.02152](https://arxiv.org/abs/2402.02152)

    这篇论文质疑了推荐系统实践中目前常用的“摸着石头过河”方法，呼吁摒弃反乌托邦思维。论文提出了使用深度学习堆栈的非标准用法，以解锁奖励优化的推荐系统的潜力。

    

    应用推荐系统研究处于一种奇特的境地。尽管在通过A/B测试来衡量性能方面有一个非常严格的协议，但找到要测试的“B”的最佳方法并没有明确地针对性能，而是针对一个代理指标。因此，一个A/B测试的成功或失败完全取决于所提出的代理指标是否与性能相关性更好。没有原则可以在离线情况下确定一个代理指标是否比另一个更好，这使得从业者们摸不着头脑。本论文的目的是质疑这种反乌托邦思维，并主张深度学习堆栈的非标准用法实际上有潜力解锁优化奖励的推荐系统。

    Applied recommender systems research is in a curious position. While there is a very rigorous protocol for measuring performance by A/B testing, best practice for finding a `B' to test does not explicitly target performance but rather targets a proxy measure. The success or failure of a given A/B test then depends entirely on if the proposed proxy is better correlated to performance than the previous proxy. No principle exists to identify if one proxy is better than another offline, leaving the practitioners shooting in the dark. The purpose of this position paper is to question this anti-Utopian thinking and argue that a non-standard use of the deep learning stacks actually has the potential to unlock reward optimizing recommendation.
    
[^54]: 加速贝叶斯优化中的前瞻：多层蒙特卡洛就够了

    Accelerating Look-ahead in Bayesian Optimization: Multilevel Monte Carlo is All you Need

    [https://arxiv.org/abs/2402.02111](https://arxiv.org/abs/2402.02111)

    本文利用多层蒙特卡洛方法加速贝叶斯优化中的前瞻过程，并证明在涉及嵌套期望和最大化的问题中具有优势。

    

    我们利用多层蒙特卡洛(MLMC)来提高涉及嵌套期望和最大化的多步前瞻贝叶斯优化(BO)方法的性能。普通蒙特卡洛的复杂度在嵌套操作中会降低，而MLMC能够以规范蒙特卡洛收敛速度解决这类问题，而且不依赖于维度和平滑性假设。我们的理论研究主要关注一步和两步前瞻采集函数的近似改进，但正如我们所讨论的，这种方法在多种方面是可推广的，包括超越BO的背景。我们通过数值验证了我们的发现，并在几个基准示例中展示了MLMC在BO中的优势。代码在这里获取：https://github.com/Shangda-Yang/MLMCBO。

    We leverage multilevel Monte Carlo (MLMC) to improve the performance of multi-step look-ahead Bayesian optimization (BO) methods that involve nested expectations and maximizations. The complexity rate of naive Monte Carlo degrades for nested operations, whereas MLMC is capable of achieving the canonical Monte Carlo convergence rate for this type of problem, independently of dimension and without any smoothness assumptions. Our theoretical study focuses on the approximation improvements for one- and two-step look-ahead acquisition functions, but, as we discuss, the approach is generalizable in various ways, including beyond the context of BO. Findings are verified numerically and the benefits of MLMC for BO are illustrated on several benchmark examples. Code is available here https://github.com/Shangda-Yang/MLMCBO.
    
[^55]: 自注意网络在QK特征值谱集中时进行定位

    Self-attention Networks Localize When QK-eigenspectrum Concentrates

    [https://arxiv.org/abs/2402.02098](https://arxiv.org/abs/2402.02098)

    本文研究了自注意网络中的注意力定位问题，通过QK特征值谱的集中定位现象来解决不同观点之间的矛盾。

    

    自注意机制在现代机器学习中非常流行。它具有适应性选择输入序列中的标记，并通过调节注意力定位的程度来实现，这被很多研究人员认为是强大模型性能的基础，但也复杂化了学习动力学的基本机制。近年来，主要有两种观点将注意力定位与模型性能联系起来。一种观点是秩坍缩，即自注意块嵌入的标记在不同的标记之间变得非常相似，导致网络表达能力降低。另一种观点是熵坍缩，即注意概率接近非均匀且熵低，使得学习动力学更容易陷入平台期。这两种失效模式似乎相互矛盾，因为秩和熵坍缩分别与均匀和非均匀注意力相关。为此，我们对QK特征值谱的集中定位进行了表征。

    The self-attention mechanism prevails in modern machine learning. It has an interesting functionality of adaptively selecting tokens from an input sequence by modulating the degree of attention localization, which many researchers speculate is the basis of the powerful model performance but complicates the underlying mechanism of the learning dynamics. In recent years, mainly two arguments have connected attention localization to the model performances. One is the rank collapse, where the embedded tokens by a self-attention block become very similar across different tokens, leading to a less expressive network. The other is the entropy collapse, where the attention probability approaches non-uniform and entails low entropy, making the learning dynamics more likely to be trapped in plateaus. These two failure modes may apparently contradict each other because the rank and entropy collapses are relevant to uniform and non-uniform attention, respectively. To this end, we characterize the 
    
[^56]: 用于神经密度比估计的$\alpha$-散度损失函数

    $\alpha$-Divergence Loss Function for Neural Density Ratio Estimation

    [https://arxiv.org/abs/2402.02041](https://arxiv.org/abs/2402.02041)

    本文提出了一种应用于神经密度比估计的$\alpha$-散度损失函数($\alpha$-Div)，通过简洁实现和稳定优化解决了现有方法中存在的优化问题。实验证明了这种损失函数的稳定性，并提出了对DRE任务的估计准确性的研究，同时给出了样本要求的解决方案。

    

    最近，神经网络在机器学习中的基础技术密度比估计(DRE)方面取得了最先进的结果。然而，现有方法因DRE的损失函数而出现了优化问题：KL散度需要大样本，训练损失梯度消失，损失函数梯度有偏。因此，本文提出了一种提供简洁实现和稳定优化的$\alpha$-散度损失函数($\alpha$-Div)。此外，还给出了对所提出的损失函数的技术验证。实验证明了所提出的损失函数的稳定性，并研究了DRE任务的估计准确性。此外，本研究还提出了使用所提出的损失函数进行DRE的样本要求，以$L_1$误差的上界联系起来，该上界将高维度DRE任务中的维度诅咒作为一个共同问题。

    Recently, neural networks have produced state-of-the-art results for density-ratio estimation (DRE), a fundamental technique in machine learning. However, existing methods bear optimization issues that arise from the loss functions of DRE: a large sample requirement of Kullback--Leibler (KL)-divergence, vanishing of train loss gradients, and biased gradients of the loss functions. Thus, an $\alpha$-divergence loss function ($\alpha$-Div) that offers concise implementation and stable optimization is proposed in this paper. Furthermore, technical justifications for the proposed loss function are presented. The stability of the proposed loss function is empirically demonstrated and the estimation accuracy of DRE tasks is investigated. Additionally, this study presents a sample requirement for DRE using the proposed loss function in terms of the upper bound of $L_1$ error, which connects a curse of dimensionality as a common problem in high-dimensional DRE tasks.
    
[^57]: GenFormer: 一种基于深度学习的生成多元随机过程的方法

    GenFormer: A Deep-Learning-Based Approach for Generating Multivariate Stochastic Processes

    [https://arxiv.org/abs/2402.02010](https://arxiv.org/abs/2402.02010)

    GenFormer是一种基于深度学习的方法，用于生成多元随机过程。它能保留目标统计特性，包括边际分布，并能在具有挑战性的应用中近似捕捉到其他期望的统计特性。应用于风速数据模拟的实验中，GenFormer模型用于计算风险管理的超越概率。

    

    随机生成器对于生成保持目标统计特性的合成实现非常重要。我们提出了GenFormer，一个用于时空多元随机过程的随机生成器。它采用基于Transformer的深度学习模型构建，学习了一个将马尔可夫状态序列映射到时间序列值的映射关系。GenFormer模型生成的合成数据保留了目标边际分布，并在涉及大量空间位置和长时间模拟的挑战性应用中近似捕捉到其他期望的统计特性。我们将GenFormer模型应用于在佛罗里达州的各个站点模拟合成风速数据，以计算风险管理的超越概率。

    Stochastic generators are essential to produce synthetic realizations that preserve target statistical properties. We propose GenFormer, a stochastic generator for spatio-temporal multivariate stochastic processes. It is constructed using a Transformer-based deep learning model that learns a mapping between a Markov state sequence and time series values. The synthetic data generated by the GenFormer model preserves the target marginal distributions and approximately captures other desired statistical properties even in challenging applications involving a large number of spatial locations and a long simulation horizon. The GenFormer model is applied to simulate synthetic wind speed data at various stations in Florida to calculate exceedance probabilities for risk management.
    
[^58]: 组合T-learning和DR-learning：一个用于高效估计因果对比的框架

    Combining T-learning and DR-learning: a framework for oracle-efficient estimation of causal contrasts

    [https://arxiv.org/abs/2402.01972](https://arxiv.org/abs/2402.01972)

    这篇论文介绍了高效插件学习的框架，能够有效估计异质因果对比，并解决了其他学习策略的一些缺点。该框架构建了人口风险函数的高效插件估计器，具有稳定性和鲁棒性。

    

    我们引入了高效插件（EP）学习，这是一种用于估计异质因果对比的新框架，例如条件平均处理效应和条件相对风险。 EP学习框架享有与Neyman正交学习策略（如DR-learning和R-learning）相同的oracle效率，同时解决了它们的一些主要缺点，包括（i）实际适用性可能受到损失函数非凸性的阻碍； （ii）它们可能因违反界限的倒数概率加权和伪结果而导致性能和稳定性差。为了避免这些缺点，EP学习者构建了因果对比的人口风险函数的高效插件估计器，从而继承了T-learning等插件估计策略的稳定性和鲁棒性特性。在合理条件下，基于经验风险最小化的EP学习者具有oracle效率，表现出渐近等价的性质。

    We introduce efficient plug-in (EP) learning, a novel framework for the estimation of heterogeneous causal contrasts, such as the conditional average treatment effect and conditional relative risk. The EP-learning framework enjoys the same oracle-efficiency as Neyman-orthogonal learning strategies, such as DR-learning and R-learning, while addressing some of their primary drawbacks, including that (i) their practical applicability can be hindered by loss function non-convexity; and (ii) they may suffer from poor performance and instability due to inverse probability weighting and pseudo-outcomes that violate bounds. To avoid these drawbacks, EP-learner constructs an efficient plug-in estimator of the population risk function for the causal contrast, thereby inheriting the stability and robustness properties of plug-in estimation strategies like T-learning. Under reasonable conditions, EP-learners based on empirical risk minimization are oracle-efficient, exhibiting asymptotic equivalen
    
[^59]: 使用Bellman残差最小化的分布式离线策略评估

    Distributional Off-policy Evaluation with Bellman Residual Minimization

    [https://arxiv.org/abs/2402.01900](https://arxiv.org/abs/2402.01900)

    这篇论文研究了使用Bellman残差最小化的方法来解决分布式离线策略评估问题，并提出了一种称为能量Bellman残差最小化（EBRM）的方法来估计返回分布。在可实现性假设下，建立了EBRM估计器的有限样本误差界。

    

    我们考虑分布式离线策略评估的问题，它是许多分布式强化学习（DRL）算法的基础。与大多数现有的方法（依赖于最大值-扩展的统计距离，如最大值Wasserstein距离）不同，我们研究用于量化分布式Bellman残差的期望-扩展的统计距离，并且证明它可以上界估计返回分布的期望误差。基于这个有吸引力的性质，通过将Bellman残差最小化框架推广到DRL，我们提出了一种称为能量Bellman残差最小化（EBRM）的方法来估计返回分布。我们在可实现性假设下建立了EBRM估计器的有限样本误差界。此外，我们引入了一种基于多步引导过程的方法的变体，以实现多步扩展。通过选择适当的步长，我们获得了更好的误差界。

    We consider the problem of distributional off-policy evaluation which serves as the foundation of many distributional reinforcement learning (DRL) algorithms. In contrast to most existing works (that rely on supremum-extended statistical distances such as supremum-Wasserstein distance), we study the expectation-extended statistical distance for quantifying the distributional Bellman residuals and show that it can upper bound the expected error of estimating the return distribution. Based on this appealing property, by extending the framework of Bellman residual minimization to DRL, we propose a method called Energy Bellman Residual Minimizer (EBRM) to estimate the return distribution. We establish a finite-sample error bound for the EBRM estimator under the realizability assumption. Furthermore, we introduce a variant of our method based on a multi-step bootstrapping procedure to enable multi-step extension. By selecting an appropriate step level, we obtain a better error bound for thi
    
[^60]: 基于f-散度原理的领域自适应：一个改进的框架

    On f-Divergence Principled Domain Adaptation: An Improved Framework

    [https://arxiv.org/abs/2402.01887](https://arxiv.org/abs/2402.01887)

    本文改进了基于f-散度的无监督领域自适应（UDA）框架，引入了f-领域差异度量指标，并通过去除绝对值函数和引入缩放参数，提出了新的目标误差和样本复杂度界限，从而使得我们能够恢复以前的KL结果，将算法和理论之间的差距缩小，并通过定位技术开发了快速率的泛化界限。实验结果证明了基于f-DD的领域学习算法在流行的UDA基准测试中表现出了卓越的性能。

    

    无监督领域自适应（UDA）在解决机器学习中的分布偏移问题中起着至关重要的作用。在本文中，我们通过改进Acuna等人（2021年）提出的UDA的理论基础，对其基于f-散度的差异度进行了改进，并引入了一个新的度量指标，即f-领域差异（f-DD）。通过去除绝对值函数并引入一个缩放参数，f-DD产生了新的目标误差和样本复杂度界限，使我们能够恢复以前基于KL的结果，并弥合了Acuna等人（2021年）中提出的算法和理论之间的差距。利用定位技术，我们还开发了一种快速率的泛化界限。实证结果表明，在流行的UDA基准测试中，基于f-DD的领域学习算法表现出优越性能。

    Unsupervised domain adaptation (UDA) plays a crucial role in addressing distribution shifts in machine learning. In this work, we improve the theoretical foundations of UDA proposed by Acuna et al. (2021) by refining their f-divergence-based discrepancy and additionally introducing a new measure, f-domain discrepancy (f-DD). By removing the absolute value function and incorporating a scaling parameter, f-DD yields novel target error and sample complexity bounds, allowing us to recover previous KL-based results and bridging the gap between algorithms and theory presented in Acuna et al. (2021). Leveraging a localization technique, we also develop a fast-rate generalization bound. Empirical results demonstrate the superior performance of f-DD-based domain learning algorithms over previous works in popular UDA benchmarks.
    
[^61]: 训练PINNs的挑战：从损失函数空间角度探究

    Challenges in Training PINNs: A Loss Landscape Perspective

    [https://arxiv.org/abs/2402.01868](https://arxiv.org/abs/2402.01868)

    本文探讨了训练PINNs的挑战，强调了损失函数空间在训练过程中的作用，引入了新颖的二阶优化器NNCG并优化了PINN性能，为训练PINNs提供了有价值的洞见和更强大的优化策略。

    

    本文通过研究物理信息神经网络（PINNs）的训练挑战，强调了损失函数空间在训练过程中的作用。我们分析了在最小化PINN损失函数方面的困难，特别是由于残差项中的微分算子引起的病态条件。我们比较了基于梯度的优化器Adam、L-BFGS以及它们的组合Adam+L-BFGS的性能，表明Adam+L-BFGS更优，并介绍了一种新颖的二阶优化器NysNewton-CG（NNCG），显著提高了PINN的性能。从理论上，我们阐明了病态微分算子与PINN损失中的病态条件之间的联系，并展示了结合一阶和二阶优化方法的好处。我们的工作为训练PINNs提供了有价值的洞见和更强大的优化策略，可以提高PINNs在解决困难的偏微分方程中的实用性。

    This paper explores challenges in training Physics-Informed Neural Networks (PINNs), emphasizing the role of the loss landscape in the training process. We examine difficulties in minimizing the PINN loss function, particularly due to ill-conditioning caused by differential operators in the residual term. We compare gradient-based optimizers Adam, L-BFGS, and their combination Adam+L-BFGS, showing the superiority of Adam+L-BFGS, and introduce a novel second-order optimizer, NysNewton-CG (NNCG), which significantly improves PINN performance. Theoretically, our work elucidates the connection between ill-conditioned differential operators and ill-conditioning in the PINN loss and shows the benefits of combining first- and second-order optimization methods. Our work presents valuable insights and more powerful optimization strategies for training PINNs, which could improve the utility of PINNs for solving difficult partial differential equations.
    
[^62]: 我的模型会忘记什么？语言模型改进中的被遗忘实例预测

    What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement

    [https://arxiv.org/abs/2402.01865](https://arxiv.org/abs/2402.01865)

    本文研究了语言模型更新中的遗忘现象，提出了一种预测上游实例遗忘的方法，以改进重播过程的可控性和解释性。根据预训练实例的预-softmax对数几率分数变化与在线学习实例的相似性，提出了一种部分可解释的预测模型，在BART模型上表现良好但在T5模型上失败。此外，还展示了基于内积的黑盒分类器。

    

    在实际应用中，语言模型会出现错误。然而，仅仅通过将模型更新为纠正错误实例，会导致灾难性的遗忘，更新后的模型在指导微调或上游训练阶段中学到的实例上出现错误。随机重播上游数据的效果不令人满意，往往伴随着较高的方差和较差的可控性。为了改善重播过程的可控性和解释性，我们试图预测由于模型更新而遗忘的上游实例。我们根据一组在线学习的实例和相应被遗忘的上游预训练实例训练预测模型。我们提出了一种部分可解释的预测模型，该模型基于这样的观察结果：预训练实例的预-softmax对数几率分数的变化类似于在线学习实例的变化，这在BART模型上表现出不错的效果，但在T5模型上失败。我们进一步展示了基于内积的黑盒分类器

    Language models deployed in the wild make errors. However, simply updating the model with the corrected error instances causes catastrophic forgetting -- the updated model makes errors on instances learned during the instruction tuning or upstream training phase. Randomly replaying upstream data yields unsatisfactory performance and often comes with high variance and poor controllability. To this end, we try to forecast upstream examples that will be forgotten due to a model update for improved controllability of the replay process and interpretability. We train forecasting models given a collection of online learned examples and corresponding forgotten upstream pre-training examples. We propose a partially interpretable forecasting model based on the observation that changes in pre-softmax logit scores of pretraining examples resemble that of online learned examples, which performs decently on BART but fails on T5 models. We further show a black-box classifier based on inner products 
    
[^63]: SPDE先验在端到端神经数据同化方案的不确定性量化中的应用

    SPDE priors for uncertainty quantification of end-to-end neural data assimilation schemes

    [https://arxiv.org/abs/2402.01855](https://arxiv.org/abs/2402.01855)

    SPDE先验在最优插值中的应用及其与神经网络的联合学习问题，为大规模地球物理数据集的时空插值提供了一种新的方法。

    

    大规模地球物理数据集的时空插值通常通过最优插值(Optimal Interpolation，OI)和更复杂的基于模型或数据驱动的数据同化技术来处理。在过去的十年中，随机偏微分方程(Spatio-temporal Partial Differential Equations，SPDE)和高斯马尔科夫随机场(Gaussian Markov Random Fields，GMRF)之间的联系开辟了一条新的途径，用于处理最优插值中的大数据集和物理诱导协方差矩阵。深度学习社区的最新进展也使得可以将这个问题视为嵌入数据同化变分框架的神经网络体系结构的联合学习问题。重建任务被视为一个包含在变分内部成本中的先验学习问题和后者的基于梯度的最小化：先验模型和求解器都被表示为具有自动微分的神经网络，可以通过最小化损失函数来训练，该损失函数通常被表示为一些真实值和重建值之间的均方误差。

    The spatio-temporal interpolation of large geophysical datasets has historically been adressed by Optimal Interpolation (OI) and more sophisticated model-based or data-driven DA techniques. In the last ten years, the link established between Stochastic Partial Differential Equations (SPDE) and Gaussian Markov Random Fields (GMRF) opened a new way of handling both large datasets and physically-induced covariance matrix in Optimal Interpolation. Recent advances in the deep learning community also enables to adress this problem as neural architecture embedding data assimilation variational framework. The reconstruction task is seen as a joint learning problem of the prior involved in the variational inner cost and the gradient-based minimization of the latter: both prior models and solvers are stated as neural networks with automatic differentiation which can be trained by minimizing a loss function, typically stated as the mean squared error between some ground truth and the reconstructi
    
[^64]: 具有干扰的多臂赌博机问题

    Multi-Armed Bandits with Interference

    [https://arxiv.org/abs/2402.01845](https://arxiv.org/abs/2402.01845)

    这篇论文研究了在在线平台中与干扰进行的实验。在多臂赌博机问题中，学习者分配不同的臂给每个实验单元，根据单元之间的空间距离和对手选择的匹配函数来决定每个单元在每轮的回报。研究发现，转换政策能够实现最佳的预期遗憾，但任何转换政策都会遭受一定的遗憾现象。

    

    在当代在线平台上，与干扰进行实验是一个重大挑战。以往有关干扰实验的研究集中在政策的最终输出上，而对于累计性能则了解不足。为了填补这一空白，我们引入了“具有干扰的多臂赌博机”（MABI）问题，在时间段为T轮的情况下，学习者为N个实验单元中的每个分配一个臂。每个单元在每一轮的回报取决于“所有”单元的治疗方式，而单元之间的空间距离会导致单元的影响力逐渐衰减。此外，我们使用了一个通用设置，其中回报函数由对手选择，并且在轮次和单元之间可以任意变化。我们首先证明了转换政策能够对最佳固定臂政策实现最优的“预期”遗憾，遗憾值为$O(\sqrt T)$。然而，任何一个转换政策的遗憾（作为一个随机变量）都会遭受一定的遗憾现象。

    Experimentation with interference poses a significant challenge in contemporary online platforms. Prior research on experimentation with interference has concentrated on the final output of a policy. The cumulative performance, while equally crucial, is less well understood. To address this gap, we introduce the problem of {\em Multi-armed Bandits with Interference} (MABI), where the learner assigns an arm to each of $N$ experimental units over a time horizon of $T$ rounds. The reward of each unit in each round depends on the treatments of {\em all} units, where the influence of a unit decays in the spatial distance between units. Furthermore, we employ a general setup wherein the reward functions are chosen by an adversary and may vary arbitrarily across rounds and units. We first show that switchback policies achieve an optimal {\em expected} regret $\tilde O(\sqrt T)$ against the best fixed-arm policy. Nonetheless, the regret (as a random variable) for any switchback policy suffers 
    
[^65]: 近确定性回归中的错误规范化不确定性

    Misspecification uncertainties in near-deterministic regression

    [https://arxiv.org/abs/2402.01810](https://arxiv.org/abs/2402.01810)

    该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。

    

    期望损失是模型泛化误差的上界，可用于学习的鲁棒PAC-Bayes边界。然而，损失最小化被认为忽略了错误规范化，即模型不能完全复制观测结果。这导致大数据或欠参数化极限下对参数不确定性的显著低估。我们分析近确定性、错误规范化和欠参数化替代模型的泛化误差，这是科学和工程中广泛相关的一个领域。我们证明后验分布必须覆盖每个训练点，以避免发散的泛化误差，并导出一个符合这个约束的组合模型。对于线性模型，这种高效的方法产生的额外开销最小。这种高效方法在模型问题上进行了演示，然后应用于原子尺度机器学习中的高维数据集。

    The expected loss is an upper bound to the model generalization error which admits robust PAC-Bayes bounds for learning. However, loss minimization is known to ignore misspecification, where models cannot exactly reproduce observations. This leads to significant underestimates of parameter uncertainties in the large data, or underparameterized, limit. We analyze the generalization error of near-deterministic, misspecified and underparametrized surrogate models, a regime of broad relevance in science and engineering. We show posterior distributions must cover every training point to avoid a divergent generalization error and derive an ensemble {ansatz} that respects this constraint, which for linear models incurs minimal overhead. The efficient approach is demonstrated on model problems before application to high dimensional datasets in atomistic machine learning. Parameter uncertainties from misspecification survive in the underparametrized limit, giving accurate prediction and boundin
    
[^66]: DoubleMLDeep: 利用多模态数据对因果效应进行估计

    DoubleMLDeep: Estimation of Causal Effects with Multimodal Data

    [https://arxiv.org/abs/2402.01785](https://arxiv.org/abs/2402.01785)

    本文提出了一个利用文本和图像在因果推断和治疗效应估计中的双机器学习框架，并提出了一种生成半合成数据集的方法用于评估因果效应估计的性能。这些方法和架构在半合成数据集上进行了评估，并与标准方法进行了比较，显示了直接使用文本和图像进行因果研究的潜在好处。

    

    本文探讨了在因果推断和治疗效应估计中使用非结构化的多模态数据，即文本和图像。我们提出了一种适应于双机器学习（DML）框架，特别是部分线性模型的神经网络架构。我们论文的另一个贡献是提出了一种生成半合成数据集的新方法，该方法可用于评估在文本和图像作为混淆因素的情况下因果效应估计的性能。我们在半合成数据集上评估并与标准方法进行比较，突出了直接在因果研究中使用文本和图像的潜在好处。我们的研究结果对经济学、市场营销、金融、医学和数据科学等领域的研究人员和从业者具有重要意义，他们希望使用非传统数据估计因果数量。

    This paper explores the use of unstructured, multimodal data, namely text and images, in causal inference and treatment effect estimation. We propose a neural network architecture that is adapted to the double machine learning (DML) framework, specifically the partially linear model. An additional contribution of our paper is a new method to generate a semi-synthetic dataset which can be used to evaluate the performance of causal effect estimation in the presence of text and images as confounders. The proposed methods and architectures are evaluated on the semi-synthetic dataset and compared to standard approaches, highlighting the potential benefit of using text and images directly in causal studies. Our findings have implications for researchers and practitioners in economics, marketing, finance, medicine and data science in general who are interested in estimating causal quantities using non-traditional data.
    
[^67]: 带有随机去噪正则化的即插即用图像恢复

    Plug-and-Play image restoration with Stochastic deNOising REgularization

    [https://arxiv.org/abs/2402.01779](https://arxiv.org/abs/2402.01779)

    本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。

    

    即插即用（PnP）算法是一类迭代算法，通过结合物理模型和深度神经网络进行正则化来解决图像反演问题。尽管这些算法能够产生令人印象深刻的图像恢复结果，但它们依赖于在迭代过程中越来越少噪音的图像上的一种非标准的去噪器使用方法，这与基于扩散模型（DM）的最新算法相矛盾，在这些算法中，去噪器仅应用于重新加噪的图像上。我们提出了一种新的PnP框架，称为随机去噪正则化（SNORE），它仅在噪声水平适当的图像上应用去噪器。它基于显式的随机正则化，从而导致了一种解决病态逆问题的随机梯度下降算法。我们提供了该算法及其退火扩展的收敛分析。在实验上，我们证明SNORE在去模糊和修复任务上与最先进的方法相竞争。

    Plug-and-Play (PnP) algorithms are a class of iterative algorithms that address image inverse problems by combining a physical model and a deep neural network for regularization. Even if they produce impressive image restoration results, these algorithms rely on a non-standard use of a denoiser on images that are less and less noisy along the iterations, which contrasts with recent algorithms based on Diffusion Models (DM), where the denoiser is applied only on re-noised images. We propose a new PnP framework, called Stochastic deNOising REgularization (SNORE), which applies the denoiser only on images with noise of the adequate level. It is based on an explicit stochastic regularization, which leads to a stochastic gradient descent algorithm to solve ill-posed inverse problems. A convergence analysis of this algorithm and its annealing extension is provided. Experimentally, we prove that SNORE is competitive with respect to state-of-the-art methods on deblurring and inpainting tasks, 
    
[^68]: 并非所有可学习的分布类都能在差分隐私下进行学习

    Not All Learnable Distribution Classes are Privately Learnable

    [https://arxiv.org/abs/2402.00267](https://arxiv.org/abs/2402.00267)

    这篇论文证明了一类分布虽然可以在有限样本下以总变差距离进行学习，但却无法在（ε，δ）-差分隐私下学习。

    

    我们给出了一个示例，展示了一类分布在有限样本下可以以总变差距离进行学习，但在（ε，δ）-差分隐私下无法学习。这推翻了Ashtiani的一个猜想。

    We give an example of a class of distributions that is learnable in total variation distance with a finite number of samples, but not learnable under $(\varepsilon, \delta)$-differential privacy. This refutes a conjecture of Ashtiani.
    
[^69]: 一种针对不平衡线性分类的扩展非对称sigmoid和感知机(SIGTRON)

    An extended asymmetric sigmoid with Perceptron (SIGTRON) for imbalanced linear classification

    [https://arxiv.org/abs/2312.16043](https://arxiv.org/abs/2312.16043)

    本文提出了一个新的多项式参数化sigmoid函数(SIGTRON)，并且介绍了其伴随的SIC模型。相比传统的成本敏感学习模型，在给定的训练数据集接近良好平衡的条件下，所提出的SIC模型对于数据集的变化更加适应，并通过创建倾斜的超平面方程来实现。

    

    本文提出了一种新的多项式参数化sigmoid函数，称为SIGTRON，它是一种扩展的非对称sigmoid函数和感知机的结合，以及它的伴随凸模型SIGTRON-不平衡分类(SIC)模型，该模型使用了虚拟SIGTRON产生的凸损失函数。与传统的$\pi$-加权成本敏感学习模型相比，SIC模型在损失函数上没有外部的$\pi$-权重，而是在虚拟的SIGTRON产生的损失函数中有内部参数。因此，当给定的训练数据集接近良好平衡的条件时，我们展示了所提出的SIC模型对数据集的变化更加适应，比如训练集和测试集之间比例不平衡的不一致性。这种适应是通过创建一个倾斜的超平面方程来实现的。另外，我们提出了一个基于拟牛顿优化(L-BFGS)框架的虚拟凸损失，通过开发一个基于区间的二分线性搜索算法来实现。

    This article presents a new polynomial parameterized sigmoid called SIGTRON, which is an extended asymmetric sigmoid with Perceptron, and its companion convex model called SIGTRON-imbalanced classification (SIC) model that employs a virtual SIGTRON-induced convex loss function. In contrast to the conventional $\pi$-weighted cost-sensitive learning model, the SIC model does not have an external $\pi$-weight on the loss function but has internal parameters in the virtual SIGTRON-induced loss function. As a consequence, when the given training dataset is close to the well-balanced condition, we show that the proposed SIC model is more adaptive to variations of the dataset, such as the inconsistency of the scale-class-imbalance ratio between the training and test datasets. This adaptation is achieved by creating a skewed hyperplane equation. Additionally, we present a quasi-Newton optimization(L-BFGS) framework for the virtual convex loss by developing an interval-based bisection line sear
    
[^70]: 在过参数化下分析锐度感知最小化

    Analyzing Sharpness-aware Minimization under Overparameterization

    [https://arxiv.org/abs/2311.17539](https://arxiv.org/abs/2311.17539)

    本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。

    

    在训练过参数化的神经网络时，尽管训练损失相同，但可以得到具有不同泛化能力的极小值。有证据表明，极小值的锐度与其泛化误差之间存在相关性，因此已经做出了更多努力开发一种优化方法，以显式地找到扁平极小值作为更具有泛化能力的解。然而，至今为止，关于过参数化对锐度感知最小化（SAM）策略的影响的研究还不多。在这项工作中，我们分析了在不同程度的过参数化下的SAM，并提出了实证和理论结果，表明过参数化对SAM具有重要影响。具体而言，我们进行了广泛的数值实验，涵盖了各个领域，并表明存在一种一致的趋势，即SAM在过参数化增加的情况下仍然受益。我们还发现了一些令人信服的案例，说明了过参数化的影响。

    Training an overparameterized neural network can yield minimizers of different generalization capabilities despite the same level of training loss. With evidence that suggests a correlation between sharpness of minima and their generalization errors, increasing efforts have been made to develop an optimization method to explicitly find flat minima as more generalizable solutions. However, this sharpness-aware minimization (SAM) strategy has not been studied much yet as to whether and how it is affected by overparameterization.   In this work, we analyze SAM under overparameterization of varying degrees and present both empirical and theoretical results that indicate a critical influence of overparameterization on SAM. Specifically, we conduct extensive numerical experiments across various domains, and show that there exists a consistent trend that SAM continues to benefit from increasing overparameterization. We also discover compelling cases where the effect of overparameterization is
    
[^71]: 用于评估潜在表示多样性的度量空间大小

    Metric Space Magnitude for Evaluating the Diversity of Latent Representations

    [https://arxiv.org/abs/2311.16054](https://arxiv.org/abs/2311.16054)

    基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。

    

    度量空间的大小是一种近期建立的不变性，能够在多个尺度上提供空间的“有效大小”的衡量，并捕捉到许多几何属性。我们发展了一系列基于大小的潜在表示内在多样性度量，形式化了有限度量空间大小函数之间的新颖不相似性概念。我们的度量在数据扰动下保证稳定，可以高效计算，并且能够对潜在表示进行严格的多尺度比较。我们展示了我们的度量在实验套件中的实用性和卓越性能，包括不同领域和任务的多样性评估、模式崩溃检测以及用于文本、图像和图形数据的生成模型评估。

    The magnitude of a metric space is a recently-established invariant, providing a measure of the 'effective size' of a space across multiple scales while also capturing numerous geometrical properties. We develop a family of magnitude-based measures of the intrinsic diversity of latent representations, formalising a novel notion of dissimilarity between magnitude functions of finite metric spaces. Our measures are provably stable under perturbations of the data, can be efficiently calculated, and enable a rigorous multi-scale comparison of latent representations. We show the utility and superior performance of our measures in an experimental suite that comprises different domains and tasks, including the evaluation of diversity, the detection of mode collapse, and the evaluation of generative models for text, image, and graph data.
    
[^72]: 一种超长Token注意力近似的单次流算法

    One Pass Streaming Algorithm for Super Long Token Attention Approximation in Sublinear Space

    [https://arxiv.org/abs/2311.14652](https://arxiv.org/abs/2311.14652)

    本文研究了在超长上下文下内存效率的问题，提出一种用于超长Token注意力近似的单次流算法，通过构建矩阵$U_1, U_2$加速注意力计算，解决了部署大型语言模型时的计算资源问题。

    

    注意力计算同时具有$O(n^2)$的时间复杂度和$O(n^2)$的空间复杂度，这使得在需要大量计算资源的流应用中部署大型语言模型(Large Language Models，LLMs)变得困难。在最近的OpenAI DevDay（2023年11月6日），OpenAI发布了一种能够支持128K长文档的新模型，在我们的论文中，我们关注的是当上下文长度$n$远大于128K ($n \gg 2^d$)时的内存有效问题。考虑到具有 Query、Key 和 Value 矩阵$Q, K, V \in \mathbb{R}^{n \times d}$的单层自注意力，多项式方法近似了注意力输出$T \in \mathbb{R}^{n \times d}$。它通过构建$U_1, U_2 \in \mathbb{R}^{n \times t}$在$n^{1+o(1)}$次时间执行内加速注意力计算${\sf Attn}(Q, K, V)$。尽管如此，计算近似的注意力矩阵$U_1U_2^\top \in \mathbb{R}^{n \times n}$仍需要$O(n^2)$的空间。

    Attention computation takes both the time complexity of $O(n^2)$ and the space complexity of $O(n^2)$ simultaneously, which makes deploying Large Language Models (LLMs) in streaming applications that involve long contexts requiring substantial computational resources. In recent OpenAI DevDay (Nov 6, 2023), OpenAI released a new model that is able to support a 128K-long document, in our paper, we focus on the memory-efficient issue when context length $n$ is much greater than 128K ($n \gg 2^d$). Considering a single-layer self-attention with Query, Key, and Value matrices $Q, K, V \in \mathbb{R}^{n \times d}$, the polynomial method approximates the attention output $T \in \mathbb{R}^{n \times d}$. It accomplishes this by constructing $U_1, U_2 \in \mathbb{R}^{n \times t}$ to expedite attention ${\sf Attn}(Q, K, V)$ computation within $n^{1+o(1)}$ time executions. Despite this, computing the approximated attention matrix $U_1U_2^\top \in \mathbb{R}^{n \times n}$ still necessitates $O(n^2
    
[^73]: 假设简化和数据自适应的后预测推断

    Assumption-lean and Data-adaptive Post-Prediction Inference

    [https://arxiv.org/abs/2311.14220](https://arxiv.org/abs/2311.14220)

    这项工作介绍了一种假设简化和数据自适应的后预测推断（POP-Inf）过程，可以有效且有力地基于机器学习预测结果进行统计推断。

    

    现代科学研究面临的主要挑战是黄金标准数据的有限可用性，而获取这些数据既耗费时间又费力。随着机器学习（ML）的快速发展，科学家们依赖于ML算法使用易得的协变量来预测这些黄金标准结果。然而，这些预测结果常常直接用于后续的统计分析中，忽略了预测过程引入的不精确性和异质性。这可能导致虚假的正面结果和无效的科学结论。在这项工作中，我们介绍了一种假设简化和数据自适应的后预测推断（POP-Inf）过程，它允许基于ML预测结果进行有效和有力的推断。它的“假设简化”属性保证在广泛的统计量上不基于ML预测做出可靠的统计推断。它的“数据自适应”特性保证了相较于现有方法的效率提高。

    A primary challenge facing modern scientific research is the limited availability of gold-standard data which can be both costly and labor-intensive to obtain. With the rapid development of machine learning (ML), scientists have relied on ML algorithms to predict these gold-standard outcomes with easily obtained covariates. However, these predicted outcomes are often used directly in subsequent statistical analyses, ignoring imprecision and heterogeneity introduced by the prediction procedure. This will likely result in false positive findings and invalid scientific conclusions. In this work, we introduce an assumption-lean and data-adaptive Post-Prediction Inference (POP-Inf) procedure that allows valid and powerful inference based on ML-predicted outcomes. Its "assumption-lean" property guarantees reliable statistical inference without assumptions on the ML-prediction, for a wide range of statistical quantities. Its "data-adaptive'" feature guarantees an efficiency gain over existing
    
[^74]: 从一般环境中学习因果表示：可辨识性和内在歧义

    Learning Causal Representations from General Environments: Identifiability and Intrinsic Ambiguity

    [https://arxiv.org/abs/2311.12267](https://arxiv.org/abs/2311.12267)

    该论文研究了从一般环境中学习因果表示的问题，提供了基于这种环境生成的数据的可辨识性结果，并指出了受到围绕节点歧义的限制。同时提出了一个算法可以恢复出地面真实模型

    

    我们研究因果表示学习，即从低级观测数据（如文本和图像）中恢复高级潜在变量及其因果关系的任务，假设可以访问从多个环境生成的观察结果。之前关于因果表示可辨识性的结果通常假设可以访问单节点干预，但实际上这是不切实际的，因为潜在变量本身就未知。在本研究中，我们提供了基于来自一般环境的数据的第一个可辨识性结果。我们展示了对于线性因果模型，虽然可以完全恢复因果图，但潜在变量只能被识别到受到围绕节点歧义（SNA）的程度上。我们提供了我们保证的对应对，证明了在我们的设置中SNA基本上是不可避免的。我们还提出了一个算法LiNGCReL，可以被证明可以恢复出地面真实模型

    We study causal representation learning, the task of recovering high-level latent variables and their causal relationships in the form of a causal graph from low-level observed data (such as text and images), assuming access to observations generated from multiple environments. Prior results on the identifiability of causal representations typically assume access to single-node interventions which is rather unrealistic in practice, since the latent variables are unknown in the first place. In this work, we provide the first identifiability results based on data that stem from general environments. We show that for linear causal models, while the causal graph can be fully recovered, the latent variables are only identified up to the surrounded-node ambiguity (SNA) \citep{varici2023score}. We provide a counterpart of our guarantee, showing that SNA is basically unavoidable in our setting. We also propose an algorithm, \texttt{LiNGCReL} which provably recovers the ground-truth model up to
    
[^75]: 基于惊喜性驱动的稳健可解释的非参数学习中的k-NN算法

    Surprisal Driven $k$-NN for Robust and Interpretable Nonparametric Learning

    [https://arxiv.org/abs/2311.10246](https://arxiv.org/abs/2311.10246)

    本论文提出了一种基于惊喜性驱动的稳健可解释的k-NN算法，通过使用信息论的角度对传统算法进行新的阐释，实现了在非参数学习中的分类、回归、密度估计和异常检测等任务。

    

    非参数学习是机器学习中的一个基本概念，旨在捕捉数据中的复杂模式和关系，而不对潜在的数据分布做出强烈的假设。在这一范式下，最为著名的算法之一是k最近邻（k-NN）算法。在这项工作中，我们通过使用机器学习在安全关键应用中的应用，从信息论的角度对传统的最近邻算法进行了新的阐释，并提出了一种稳健可解释的框架，用于分类、回归、密度估计和异常检测等任务。我们可以通过计算增加特征时的条件熵来确定数据点的权重和特征的贡献，而无需进行显式的模型训练。这使我们能够通过提供详细的数据点影响权重来计算特征的贡献。

    Nonparametric learning is a fundamental concept in machine learning that aims to capture complex patterns and relationships in data without making strong assumptions about the underlying data distribution. Owing to simplicity and familiarity, one of the most well-known algorithms under this paradigm is the $k$-nearest neighbors ($k$-NN) algorithm. Driven by the usage of machine learning in safety-critical applications, in this work, we shed new light on the traditional nearest neighbors algorithm from the perspective of information theory and propose a robust and interpretable framework for tasks such as classification, regression, density estimation, and anomaly detection using a single model. We can determine data point weights as well as feature contributions by calculating the conditional entropy for adding a feature without the need for explicit model training. This allows us to compute feature contributions by providing detailed data point influence weights with perfect attributi
    
[^76]: 带有分布式量化流的GFlowNets

    Distributional GFlowNets with Quantile Flows

    [https://arxiv.org/abs/2302.05793](https://arxiv.org/abs/2302.05793)

    本文提出了一种带分布式量化流的GFlowNets模型，通过将流函数转化为分布，在训练过程中提供更多信息的学习信号。通过量化函数参数化每个边流，我们提出的算法可以学习风险敏感的策略，实现对风险不确定性场景的处理，并在现有基准上取得了显著改进。

    

    生成式流网络（GFlowNets）是一种新的概率采样器系列，其中代理通过一系列决策步骤学习生成复杂组合结构的随机策略。尽管受强化学习启发，当前的GFlowNet框架在适用性上相对有限，无法处理奖励函数中的随机性。在这项工作中，我们采用分布式范式来处理GFlowNets，将每个流函数转化为一个分布，从而在训练过程中提供更多信息的学习信号。通过通过量化函数对每个边流进行参数化，我们提出的“量化匹配” GFlowNet学习算法能够学习风险敏感的策略，这是处理风险不确定性场景的基本组成部分。此外，我们发现与之前的方法相比，分布式方法由于我们增强的训练算法，可以在现有基准上实现显着改进。

    Generative Flow Networks (GFlowNets) are a new family of probabilistic samplers where an agent learns a stochastic policy for generating complex combinatorial structure through a series of decision-making steps. Despite being inspired from reinforcement learning, the current GFlowNet framework is relatively limited in its applicability and cannot handle stochasticity in the reward function. In this work, we adopt a distributional paradigm for GFlowNets, turning each flow function into a distribution, thus providing more informative learning signals during training. By parameterizing each edge flow through their quantile functions, our proposed \textit{quantile matching} GFlowNet learning algorithm is able to learn a risk-sensitive policy, an essential component for handling scenarios with risk uncertainty. Moreover, we find that the distributional approach can achieve substantial improvement on existing benchmarks compared to prior methods due to our enhanced training algorithm, even i
    
[^77]: 可实现学习就是你所需要的一切

    Realizable Learning is All You Need

    [https://arxiv.org/abs/2111.04746](https://arxiv.org/abs/2111.04746)

    可实现学习与无偏学习的等价性是学习理论中的基本现象，我们提出了第一个独立于模型的框架来解释这个等价性，它可以适用于各种设置，并拓展了我们对各种学习情况的理解。

    

    可实现学习与无偏学习的等价性是学习理论中的基本现象。从经典的PAC学习和回归到最近的趋势如对抗鲁棒学习，令人惊讶的是我们仍然缺乏一个统一的理论；传统的等价性证明往往是零散的，并且依赖于强的模型特定假设，如均匀收敛和样本压缩。在这项工作中，我们提供了第一个独立于模型的框架，解释了可实现学习与无偏学习的等价性：一个三行代码的黑盒简化，统一和拓展了我们对各种设置的理解。这包括了没有已知可学习性描述的模型，如具有任意分布假设和更一般的损失函数的学习，以及一系列其他流行的设置，如鲁棒学习、部分学习、公平学习和统计查询模型。

    The equivalence of realizable and agnostic learnability is a fundamental phenomenon in learning theory. With variants ranging from classical settings like PAC learning and regression to recent trends such as adversarially robust learning, it's surprising that we still lack a unified theory; traditional proofs of the equivalence tend to be disparate, and rely on strong model-specific assumptions like uniform convergence and sample compression.   In this work, we give the first model-independent framework explaining the equivalence of realizable and agnostic learnability: a three-line blackbox reduction that simplifies, unifies, and extends our understanding across a wide variety of settings. This includes models with no known characterization of learnability such as learning with arbitrary distributional assumptions and more general loss functions, as well as a host of other popular settings such as robust learning, partial learning, fair learning, and the statistical query model.   Mor
    
[^78]: 后正则化置信带在普通微分方程中的应用

    Post-Regularization Confidence Bands for Ordinary Differential Equations

    [https://arxiv.org/abs/2110.12510](https://arxiv.org/abs/2110.12510)

    本文提出了一种新的方法，通过后正则化置信带来推断未知函数和有噪声数据观测下的普通微分方程的个体调控函数，该方法结合了局部核学习和新的去偏方法，有效解决了建立置信带的挑战性问题。

    

    普通微分方程（ODE）是研究生物和物理过程系统动态的重要工具。ODE建模的一个核心问题是推断一个信号变量对另一个信号变量的个体调控作用的显著性。然而，对于具有未知调控关系的ODE建立置信带是具有挑战性的，并且仍然是一个开放的问题。在本文中，我们构建了针对具有未知函数和有噪声数据观测的ODE中个体调控函数的后正则化置信带。我们的提议是第一种这样的方法，并且建立在两个新颖的要素上。第一个要素是将再生核学习与局部泰勒展开相结合的新型局部化核学习方法，第二个要素是解决无穷维度函数和附加的测量误差的新的去偏方法。我们证明了所构建的置信带具有期望的渐近覆盖概率，并且能够恢复调控关系。

    Ordinary differential equation (ODE) is an important tool to study the dynamics of a system of biological and physical processes. A central question in ODE modeling is to infer the significance of individual regulatory effect of one signal variable on another. However, building confidence band for ODE with unknown regulatory relations is challenging, and it remains largely an open question. In this article, we construct post-regularization confidence band for individual regulatory function in ODE with unknown functionals and noisy data observations. Our proposal is the first of its kind, and is built on two novel ingredients. The first is a new localized kernel learning approach that combines reproducing kernel learning with local Taylor approximation, and the second is a new de-biasing method that tackles infinite-dimensional functionals and additional measurement errors. We show that the constructed confidence band has the desired asymptotic coverage probability, and the recovered re
    
[^79]: 在连续治疗下的多重稳健因果中介分析

    Multiply Robust Causal Mediation Analysis with Continuous Treatments

    [https://arxiv.org/abs/2105.09254](https://arxiv.org/abs/2105.09254)

    本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。

    

    在许多应用中，研究人员对治疗或暴露对感兴趣的结果的直接和间接的因果效应。中介分析为鉴定和估计这些因果效应提供了一个严谨的框架。对于二元治疗，Tchetgen Tchetgen和Shpitser (2012)提出了直接和间接效应的高效估计器，基于参数的影响函数。这些估计器具有良好的性质，如多重稳健性和渐近正态性，同时允许对干扰参数进行低于根号n的收敛速度。然而，在涉及连续治疗的情况下，这些基于影响函数的估计器没有准备好应用，除非进行强参数假设。在这项工作中，我们利用核平滑方法提出了一种适用于连续治疗环境的估计器，受到Tchetgen Tchetgen的影响函数估计器的启发。

    In many applications, researchers are interested in the direct and indirect causal effects of a treatment or exposure on an outcome of interest. Mediation analysis offers a rigorous framework for identifying and estimating these causal effects. For binary treatments, efficient estimators for the direct and indirect effects are presented in Tchetgen Tchetgen and Shpitser (2012) based on the influence function of the parameter of interest. These estimators possess desirable properties, such as multiple-robustness and asymptotic normality, while allowing for slower than root-n rates of convergence for the nuisance parameters. However, in settings involving continuous treatments, these influence function-based estimators are not readily applicable without making strong parametric assumptions. In this work, utilizing a kernel-smoothing approach, we propose an estimator suitable for settings with continuous treatments inspired by the influence function-based estimator of Tchetgen Tchetgen an
    
[^80]: 一个对线性模型进行严格介绍的书籍

    A rigorous introduction to linear models

    [https://arxiv.org/abs/2105.04240](https://arxiv.org/abs/2105.04240)

    本书旨在向读者提供对线性模型及其理论的严格介绍，并总结了线性模型在回归问题中的重要性和应用。

    

    本书旨在向读者介绍线性模型及其背后的理论。我们的目标是为读者提供一个严谨的介绍，前提是读者具有普通最小二乘法的先前经验。在机器学习中，输出通常是输入的非线性函数。深度学习甚至旨在找到具有许多层的非线性依赖关系，这需要大量计算。然而，大多数算法都是基于简单的线性模型构建的。我们从不同的角度描述线性模型，找到模型背后的性质和理论。线性模型是回归问题中的主要技术，最主要的工具是最小二乘逼近，它最小化了平方误差的和。当我们有兴趣找到最小化相应的期望平方误差的回归函数时，这是一个自然的选择。本书主要总结了线性模型背后的目的和重要理论的意义，例如概率分布、推导和估计方法等等。

    This book is meant to provide an introduction to linear models and the theories behind them. Our goal is to give a rigorous introduction to the readers with prior exposure to ordinary least squares. In machine learning, the output is usually a nonlinear function of the input. Deep learning even aims to find a nonlinear dependence with many layers, which require a large amount of computation. However, most of these algorithms build upon simple linear models. We then describe linear models from different perspectives and find the properties and theories behind the models. The linear model is the main technique in regression problems, and the primary tool for it is the least squares approximation, which minimizes a sum of squared errors. This is a natural choice when we're interested in finding the regression function which minimizes the corresponding expected squared error. This book is primarily a summary of purpose, significance of important theories behind linear models, e.g., distrib
    
[^81]: 来自噪声二进制反馈的最优聚类

    Optimal Clustering from Noisy Binary Feedback

    [https://arxiv.org/abs/1910.06002](https://arxiv.org/abs/1910.06002)

    本论文研究了通过二进制用户反馈进行聚类的问题，并提出了一种算法来最小化聚类恢复错误率。

    

    我们研究了通过二进制用户反馈来进行聚类的问题。这样的问题在大规模标记任务中以最小的用户工作量解决的众包平台上出现。例如，在一些最近的reCAPTCHA系统中，用户的点击（二进制答案）可以用来有效地标记图像。在我们的推理问题中，项目被分成最初未知的不重叠的聚类。为了恢复这些聚类，学习者按顺序向用户呈现一系列项目，每个项目都附有一个从固定有限集合中选择的具有二进制答案的问题。对于这些项目中的每一个，用户提供的是一个由项目聚类、问题和一个描述对项目进行分类的难度的项目特定参数决定期望的噪声答案。目标是设计一种算法，具有最小的聚类恢复错误率。我们得到了任何算法满足的问题特定的信息理论下界，用于错误率。

    We study the problem of clustering a set of items from binary user feedback. Such a problem arises in crowdsourcing platforms solving large-scale labeling tasks with minimal effort put on the users. For example, in some of the recent reCAPTCHA systems, users clicks (binary answers) can be used to efficiently label images. In our inference problem, items are grouped into initially unknown non-overlapping clusters. To recover these clusters, the learner sequentially presents to users a finite list of items together with a question with a binary answer selected from a fixed finite set. For each of these items, the user provides a noisy answer whose expectation is determined by the item cluster and the question and by an item-specific parameter characterizing the {\it hardness} of classifying the item. The objective is to devise an algorithm with a minimal cluster recovery error rate. We derive problem-specific information-theoretical lower bounds on the error rate satisfied by any algorit
    
[^82]: 无知回归问题中的不可知样本压缩方案

    Agnostic Sample Compression Schemes for Regression

    [https://arxiv.org/abs/1810.01864](https://arxiv.org/abs/1810.01864)

    本文在绝对值损失函数为 $\ell_p$ 的不确定回归设置中构建了一种通用的逼近样本压缩方案，对于线性回归可以实现线性维度大小的压缩，对于 $\ell_1$ 和 $\ell_\infty$ 损失函数可以实现线性维度大小的有效完全样本压缩方案；同时，证明了其他 $\ell_p$ 损失函数不存在有限尺寸的完全不可知压缩方案的结果，并提出了开放问题。

    

    我们在绝对值损失函数为 $\ell_p$ 的不确定回归设置中获得了第一个有限样本压缩的积极结果，其中 $p \in [1, \infty]$。我们构建了一种通用的逼近样本压缩方案，适用于展示了指数级大小的fat-shattering维度但与样本数量无关的实值函数类。值得注意的是，在线性回归中，我们构造了一个线性维度大小的逼近压缩。此外，在$\ell_1$和$\ell_\infty$损失函数中，我们甚至可以展示出一个线性维度大小的有效完全样本压缩方案。我们进一步证明了对于其他每一个 $\ell_p$ 损失函数，其中 $p \in (1,\infty)$，不存在有限尺寸的完全不可知压缩方案。这进一步改进和推广了David、Moran和Yehudayoff对于$\ell_2$损失的负面结果。我们最后提出了一般性的开放问题：对于 $\ell_1$ 损失的不可知回归问题，是否每个函数类都存在尺寸为...的完全压缩方案？

    We obtain the first positive results for bounded sample compression in the agnostic regression setting with the $\ell_p$ loss, where $p\in [1,\infty]$. We construct a generic approximate sample compression scheme for real-valued function classes exhibiting exponential size in the fat-shattering dimension but independent of the sample size. Notably, for linear regression, an approximate compression of size linear in the dimension is constructed. Moreover, for $\ell_1$ and $\ell_\infty$ losses, we can even exhibit an efficient exact sample compression scheme of size linear in the dimension. We further show that for every other $\ell_p$ loss, $p\in (1,\infty)$, there does not exist an exact agnostic compression scheme of bounded size. This refines and generalizes a negative result of David, Moran, and Yehudayoff for the $\ell_2$ loss. We close by posing general open questions: for agnostic regression with $\ell_1$ loss, does every function class admits an exact compression scheme of size 
    
[^83]: 利用Ricci流引导的自编码器学习时变动力学

    Ricci flow-guided autoencoders in learning time-dependent dynamics. (arXiv:2401.14591v1 [cs.LG])

    [http://arxiv.org/abs/2401.14591](http://arxiv.org/abs/2401.14591)

    利用Ricci流引导的自编码器方法能够学习非线性动力学，尤其是偏微分方程。该方法通过在训练中学习流形，并使用Ricci流使流形潜空间逐步适应动力学的变化，从而获得更好的表示能力。在实验中，我们展示了该方法在具有周期性和随机性的PDE上的应用，并评估了在分布内和外推场景中的误差。

    

    我们提出了一种基于流形的自编码器方法，用于学习时间上的非线性动力学，尤其是偏微分方程（PDE），其中流形潜空间根据Ricci流发展。这可以通过在物理信息设置中模拟Ricci流来实现，并且可以匹配流形量，以便实现Ricci流。使用我们的方法，流形是作为训练过程的一部分学习的，因此可以识别出理想的几何形状，同时演变也能在静态方法上引起更宽容的潜在表示。我们在一系列数值实验中展示了我们的方法，包括具有周期性和随机性等理想特征的PDE，并在分布内和外推场景中进行误差评估。

    We present a manifold-based autoencoder method for learning nonlinear dynamics in time, notably partial differential equations (PDEs), in which the manifold latent space evolves according to Ricci flow. This can be accomplished by simulating Ricci flow in a physics-informed setting, and manifold quantities can be matched so that Ricci flow is empirically achieved. With our methodology, the manifold is learned as part of the training procedure, so ideal geometries may be discerned, while the evolution simultaneously induces a more accommodating latent representation over static methods. We present our method on a range of numerical experiments consisting of PDEs that encompass desirable characteristics such as periodicity and randomness, remarking error on in-distribution and extrapolation scenarios.
    
[^84]: 通过分解梯度下降实现低胞状秩张量恢复

    Low-Tubal-Rank Tensor Recovery via Factorized Gradient Descent. (arXiv:2401.11940v1 [cs.LG])

    [http://arxiv.org/abs/2401.11940](http://arxiv.org/abs/2401.11940)

    本文提出了一种通过分解梯度下降方法解决低胞状秩张量恢复问题的高效方法，该方法通过将大张量分解为两个较小的因子张量，在减少计算成本和存储需求的同时，确保了收敛性。

    

    本文研究了从少量被破坏的线性测量中恢复具有低胞状秩结构的张量的问题。传统方法需要计算张量奇异值分解（t-SVD），这是一种计算密集的过程，使它们难以处理大规模张量。为了解决这个挑战，我们提出了一种基于类似于Burer-Monteiro（BM）方法的分解过程的高效低胞状秩张量恢复方法。具体而言，我们的基本方法涉及将一个大张量分解为两个较小的因子张量，然后通过分解梯度下降（FGD）来解决问题。该策略消除了t-SVD计算的需要，从而减少了计算成本和存储需求。我们提供了严格的理论分析，以保证FGD在无噪声和有噪声情况下的收敛性。

    This paper considers the problem of recovering a tensor with an underlying low-tubal-rank structure from a small number of corrupted linear measurements. Traditional approaches tackling such a problem require the computation of tensor Singular Value Decomposition (t-SVD), that is a computationally intensive process, rendering them impractical for dealing with large-scale tensors. Aim to address this challenge, we propose an efficient and effective low-tubal-rank tensor recovery method based on a factorization procedure akin to the Burer-Monteiro (BM) method. Precisely, our fundamental approach involves decomposing a large tensor into two smaller factor tensors, followed by solving the problem through factorized gradient descent (FGD). This strategy eliminates the need for t-SVD computation, thereby reducing computational costs and storage requirements. We provide rigorous theoretical analysis to ensure the convergence of FGD under both noise-free and noisy situations. Additionally, it 
    
[^85]: 通过修剪均值的鲁棒聚类的一般理论

    A general theory for robust clustering via trimmed mean. (arXiv:2401.05574v1 [math.ST])

    [http://arxiv.org/abs/2401.05574](http://arxiv.org/abs/2401.05574)

    本文提出了一种通过使用修剪均值类型的中心点估计的混合聚类技术，用于在存在次高斯误差的中心点周围分布的弱初始化条件下产生最优错误标记保证，并且在存在敌对异常值的情况下仍然有效。

    

    在存在异质数据的统计机器学习中，聚类是一种基本工具。许多最近的结果主要关注在数据围绕带有次高斯误差的中心点分布时的最优错误标记保证。然而，限制性的次高斯模型在实践中常常无效，因为各种实际应用展示了围绕中心点的重尾分布或受到可能的敌对攻击，需要具有鲁棒数据驱动初始化的鲁棒聚类。在本文中，我们引入一种混合聚类技术，利用一种新颖的多变量修剪均值类型的中心点估计，在中心点周围的误差分布的弱初始化条件下产生错误标记保证。我们还给出了一个相匹配的下界，上界依赖于聚类的数量。此外，我们的方法即使在存在敌对异常值的情况下也能产生最优错误标记。我们的结果简化为亚高斯模型的情况。

    Clustering is a fundamental tool in statistical machine learning in the presence of heterogeneous data. Many recent results focus primarily on optimal mislabeling guarantees, when data are distributed around centroids with sub-Gaussian errors. Yet, the restrictive sub-Gaussian model is often invalid in practice, since various real-world applications exhibit heavy tail distributions around the centroids or suffer from possible adversarial attacks that call for robust clustering with a robust data-driven initialization. In this paper, we introduce a hybrid clustering technique with a novel multivariate trimmed mean type centroid estimate to produce mislabeling guarantees under a weak initialization condition for general error distributions around the centroids. A matching lower bound is derived, up to factors depending on the number of clusters. In addition, our approach also produces the optimal mislabeling even in the presence of adversarial outliers. Our results reduce to the sub-Gaus
    
[^86]: SASSL:通过神经风格迁移增强自监督学习

    SASSL: Enhancing Self-Supervised Learning via Neural Style Transfer. (arXiv:2312.01187v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.01187](http://arxiv.org/abs/2312.01187)

    SASSL提出了一种基于神经风格迁移的增强技术，通过解耦语义和风格属性，在自监督学习中生成多样化的增强样本，从而提升了图像分类性能。

    

    自监督学习依赖于数据增强来从无标签图像中提取有意义的表征。现有的最先进的增强流水线包括了各种原始的转换，但通常忽略了自然图像的结构。因此，增强样本可能显示出退化的语义信息和低风格多样性，从而影响到自监督表征的下游性能。为了克服这个问题，我们提出了一种名为SASSL的新型增强技术，它基于神经风格迁移。该方法将图像中的语义和风格属性解耦，并仅对风格应用转换，保持内容，生成多样化的增强样本，更好地保留它们的语义属性。实验结果显示，与广为接受的MoCo v2相比，我们的技术在ImageNet上的top-1分类性能提升超过2%。

    Self-supervised learning relies heavily on data augmentation to extract meaningful representations from unlabeled images. While existing state-of-the-art augmentation pipelines incorporate a wide range of primitive transformations, these often disregard natural image structure. Thus, augmented samples can exhibit degraded semantic information and low stylistic diversity, affecting downstream performance of self-supervised representations. To overcome this, we propose SASSL: Style Augmentations for Self Supervised Learning, a novel augmentation technique based on Neural Style Transfer. The method decouples semantic and stylistic attributes in images and applies transformations exclusively to the style while preserving content, generating diverse augmented samples that better retain their semantic properties. Experimental results show our technique achieves a top-1 classification performance improvement of more than 2% on ImageNet compared to the well-established MoCo v2. We also measure
    
[^87]: 无监督联邦学习：具有对抗攻击鲁棒性的异构混合模型的联邦梯度EM算法

    Unsupervised Federated Learning: A Federated Gradient EM Algorithm for Heterogeneous Mixture Models with Robustness against Adversarial Attacks. (arXiv:2310.15330v1 [stat.ML])

    [http://arxiv.org/abs/2310.15330](http://arxiv.org/abs/2310.15330)

    本文介绍了一种针对带有异构混合比例的混合模型的无监督学习的新型联邦梯度EM算法，在适用于普通混合模型的全面有限样本理论基础上，对高斯混合模型（GMM）和混合回归（MoRs）进行了具体的估计误差分析。该算法具有适应未知任务相似性、抵抗对少部分数据源的对抗攻击、保护本地数据隐私以及计算和通信效率等关键优势。

    

    尽管有监督的联邦学习方法取得了显著的成功，但无监督的联邦学习领域相对较少探索。在本文中，我们介绍了一种针对带有异构混合比例的混合模型的无监督学习的新型联邦梯度EM算法。我们首先提出了适用于普通混合模型的全面有限样本理论，然后将这一通用理论应用于高斯混合模型（GMM）和混合回归（MoRs）以描述模型参数和混合比例的显式估计误差。我们提出的联邦梯度EM算法具有以下几个关键优势：适应未知任务相似性、对少部分数据源的对抗攻击具有弹性、保护本地数据隐私以及计算和通信效率。

    While supervised federated learning approaches have enjoyed significant success, the domain of unsupervised federated learning remains relatively underexplored. In this paper, we introduce a novel federated gradient EM algorithm designed for the unsupervised learning of mixture models with heterogeneous mixture proportions across tasks. We begin with a comprehensive finite-sample theory that holds for general mixture models, then apply this general theory on Gaussian Mixture Models (GMMs) and Mixture of Regressions (MoRs) to characterize the explicit estimation error of model parameters and mixture proportions. Our proposed federated gradient EM algorithm demonstrates several key advantages: adaptability to unknown task similarity, resilience against adversarial attacks on a small fraction of data sources, protection of local data privacy, and computational and communication efficiency.
    
[^88]: 两层神经网络中一次梯度下降的非线性特征学习理论

    A Theory of Non-Linear Feature Learning with One Gradient Step in Two-Layer Neural Networks. (arXiv:2310.07891v1 [stat.ML])

    [http://arxiv.org/abs/2310.07891](http://arxiv.org/abs/2310.07891)

    这篇论文提出了一种关于两层神经网络中非线性特征学习的理论。通过一步梯度下降训练的过程中引入不同的多项式特征，该方法能够学习到目标函数的非线性组件，而更新的神经网络的性能则由这些特征所决定。

    

    特征学习被认为是深度神经网络成功的基本原因之一。在特定条件下已经严格证明，在两层全连接神经网络中，第一层进行一步梯度下降，然后在第二层进行岭回归可以导致特征学习；特征矩阵的谱中会出现分离的一维组件，称为“spike”。然而，使用固定梯度下降步长时，这个“spike”仅提供了目标函数的线性组件的信息，因此学习非线性组件是不可能的。我们展示了当学习率随样本大小增长时，这样的训练实际上引入了多个一维组件，每个组件对应一个特定的多项式特征。我们进一步证明了更新的神经网络的极限大维度和大样本训练和测试误差完全由这些“spike”所决定。

    Feature learning is thought to be one of the fundamental reasons for the success of deep neural networks. It is rigorously known that in two-layer fully-connected neural networks under certain conditions, one step of gradient descent on the first layer followed by ridge regression on the second layer can lead to feature learning; characterized by the appearance of a separated rank-one component -- spike -- in the spectrum of the feature matrix. However, with a constant gradient descent step size, this spike only carries information from the linear component of the target function and therefore learning non-linear components is impossible. We show that with a learning rate that grows with the sample size, such training in fact introduces multiple rank-one components, each corresponding to a specific polynomial feature. We further prove that the limiting large-dimensional and large sample training and test errors of the updated neural networks are fully characterized by these spikes. By 
    
[^89]: 宽深度神经网络的鲁棒过拟合的理论分析：一种NTK方法

    Theoretical Analysis of Robust Overfitting for Wide DNNs: An NTK Approach. (arXiv:2310.06112v1 [cs.LG])

    [http://arxiv.org/abs/2310.06112](http://arxiv.org/abs/2310.06112)

    本文理论分析了宽深度神经网络的鲁棒过拟合现象，并提出了一种名为Adv-NTK的AT算法来增强神经网络的鲁棒性。

    

    对抗训练(AT)是增强深度神经网络(DNNs)鲁棒性的经典方法。然而，最近的研究实验证明它存在鲁棒过拟合现象，即长时间的AT可能对DNNs的鲁棒性产生不利影响。本文对DNNs的鲁棒过拟合提出了一个理论解释。具体而言，我们将神经切向核(NTK)理论非平凡地扩展到AT，并证明了通过AT训练的宽DNN可以很好地近似为一个线性化的DNN。此外，对于平方损失，可以推导出线性化DNN的闭式AT动力学，揭示了一种新的AT退化现象：长期的AT将导致宽DNN退化为没有AT的DNN，从而引起鲁棒过拟合。根据我们的理论结果，我们进一步设计了一种名为Adv-NTK的方法，这是第一种针对无限宽的DNNs的AT算法。在实际数据集上的实验结果表明，Adv-NTK可以帮助无限宽的DNNs提升鲁棒性。

    Adversarial training (AT) is a canonical method for enhancing the robustness of deep neural networks (DNNs). However, recent studies empirically demonstrated that it suffers from robust overfitting, i.e., a long time AT can be detrimental to the robustness of DNNs. This paper presents a theoretical explanation of robust overfitting for DNNs. Specifically, we non-trivially extend the neural tangent kernel (NTK) theory to AT and prove that an adversarially trained wide DNN can be well approximated by a linearized DNN. Moreover, for squared loss, closed-form AT dynamics for the linearized DNN can be derived, which reveals a new AT degeneration phenomenon: a long-term AT will result in a wide DNN degenerates to that obtained without AT and thus cause robust overfitting. Based on our theoretical results, we further design a method namely Adv-NTK, the first AT algorithm for infinite-width DNNs. Experiments on real-world datasets show that Adv-NTK can help infinite-width DNNs enhance comparab
    
[^90]: Entropy-MCMC: 轻松从平坦盆地进行采样

    Entropy-MCMC: Sampling from Flat Basins with Ease. (arXiv:2310.05401v1 [cs.LG] CROSS LISTED)

    [http://arxiv.org/abs/2310.05401](http://arxiv.org/abs/2310.05401)

    本文提出了一种Entropy-MCMC的方法，通过引入一个辅助的引导变量来在平坦盆地中进行采样，以解决深度神经网络后验分布的多模态问题，并证明了该方法的收敛性。

    

    贝叶斯深度学习依赖于对后验分布的质量估计。然而，深度神经网络的后验分布在性质上是高度多模态的，局部模式表现出不同的泛化性能。在有限的计算资源下，从原始后验分布中进行采样可能会导致次优性能，因为一些样本可能会陷入“坏”模式并出现过拟合。基于观察到低泛化误差的“好”模式通常存在于能量景观的平坦盆地中，我们提出通过偏置采样朝向这些平坦区域的后验。具体而言，我们引入了一个辅助引导变量，其稳态分布类似于平滑后验分布，并且没有尖锐的模态，以引导MCMC采样器在平坦的盆地中采样。通过将此引导变量与模型参数相结合，我们创建了一个简单的联合分布，可以在最小计算开销下实现高效采样。我们证明了我们的元算法的收敛性。

    Bayesian deep learning counts on the quality of posterior distribution estimation. However, the posterior of deep neural networks is highly multi-modal in nature, with local modes exhibiting varying generalization performance. Given a practical budget, sampling from the original posterior can lead to suboptimal performance, as some samples may become trapped in "bad" modes and suffer from overfitting. Leveraging the observation that "good" modes with low generalization error often reside in flat basins of the energy landscape, we propose to bias sampling on the posterior toward these flat regions. Specifically, we introduce an auxiliary guiding variable, the stationary distribution of which resembles a smoothed posterior free from sharp modes, to lead the MCMC sampler to flat basins. By integrating this guiding variable with the model parameter, we create a simple joint distribution that enables efficient sampling with minimal computational overhead. We prove the convergence of our met
    
[^91]: 通过区分位置和上下文来揭示Transformers中的隐藏几何

    Uncovering hidden geometry in Transformers via disentangling position and context. (arXiv:2310.04861v1 [cs.LG])

    [http://arxiv.org/abs/2310.04861](http://arxiv.org/abs/2310.04861)

    本文通过分解transformer的隐藏状态，揭示了其在语义理解中的隐含几何结构。

    

    Transformers广泛用于从输入令牌中提取复杂的语义意义，然而它们通常作为黑盒模型运行。本文提出了一种简单而信息丰富的方法，将训练好的transformer的隐藏状态（或嵌入）分解为可解释的组件。对于任何层，输入序列样本的嵌入向量由一个张量表示 $\boldsymbol{h} \in \mathbb{R}^{C \times T \times d}$。给定在序列（或上下文） $c \le C$ 的位置 $t \le T$ 处的嵌入向量 $\boldsymbol{h}_{c,t} \in \mathbb{R}^d$，提取均值效果得到分解形式 \[ \boldsymbol{h}_{c,t} = \boldsymbol{\mu} + \mathbf{pos}_t + \mathbf{ctx}_c + \mathbf{resid}_{c,t} \] 其中 $\boldsymbol{\mu}$ 是全局均值向量，$\mathbf{pos}_t$ 和 $\mathbf{ctx}_c$ 分别是跨上下文和跨位置的均值向量，$\mathbf{resid}_{c,t}$ 是残余向量。针对流行的transformer架构和多样的文本数据集，经验结果表明...

    Transformers are widely used to extract complex semantic meanings from input tokens, yet they usually operate as black-box models. In this paper, we present a simple yet informative decomposition of hidden states (or embeddings) of trained transformers into interpretable components. For any layer, embedding vectors of input sequence samples are represented by a tensor $\boldsymbol{h} \in \mathbb{R}^{C \times T \times d}$. Given embedding vector $\boldsymbol{h}_{c,t} \in \mathbb{R}^d$ at sequence position $t \le T$ in a sequence (or context) $c \le C$, extracting the mean effects yields the decomposition \[ \boldsymbol{h}_{c,t} = \boldsymbol{\mu} + \mathbf{pos}_t + \mathbf{ctx}_c + \mathbf{resid}_{c,t} \] where $\boldsymbol{\mu}$ is the global mean vector, $\mathbf{pos}_t$ and $\mathbf{ctx}_c$ are the mean vectors across contexts and across positions respectively, and $\mathbf{resid}_{c,t}$ is the residual vector. For popular transformer architectures and diverse text datasets, empirica
    
[^92]: 学习温度条件下尺度标量化的GFlowNets

    Learning to Scale Logits for Temperature-Conditional GFlowNets. (arXiv:2310.02823v1 [cs.LG])

    [http://arxiv.org/abs/2310.02823](http://arxiv.org/abs/2310.02823)

    这项研究提出了一种名为LSL-GFN的新型架构设计，可以大大加速温度条件下GFlowNets的训练，从而提高GFlowNets的探索和利用能力。

    

    GFlowNets是一种概率模型，通过学习随机策略来顺序生成组合结构，例如分子图。它们的训练目标是按比例采样具有相应温度调节的对象的奖励。在GFlowNets中，温度条件下的GFlowNets代表了一系列由温度索引的策略，每个策略与相应的温度调节奖励函数相关联。温度条件下的GFlowNets的主要优势在于通过调整温度来控制对GFlowNets的探索和利用。我们提出了一种名为学习温度条件下尺度标量化的GFlowNets（LSL-GFN）的新型架构设计，它极大地加速了温度条件下GFlowNets的训练。它基于一个思想，即之前提出的温度条件方法在深度网络的训练中引入了数值挑战，因为不同的温度可能导致非常不同的情况。

    GFlowNets are probabilistic models that learn a stochastic policy that sequentially generates compositional structures, such as molecular graphs. They are trained with the objective of sampling such objects with probability proportional to the object's reward. Among GFlowNets, the temperature-conditional GFlowNets represent a family of policies indexed by temperature, and each is associated with the correspondingly tempered reward function. The major benefit of temperature-conditional GFlowNets is the controllability of GFlowNets' exploration and exploitation through adjusting temperature. We propose Learning to Scale Logits for temperature-conditional GFlowNets (LSL-GFN), a novel architectural design that greatly accelerates the training of temperature-conditional GFlowNets. It is based on the idea that previously proposed temperature-conditioning approaches introduced numerical challenges in the training of the deep network because different temperatures may give rise to very differe
    
[^93]: 控制组合优化的连续放松

    Controlling Continuous Relaxation for Combinatorial Optimization. (arXiv:2309.16965v1 [stat.ML])

    [http://arxiv.org/abs/2309.16965](http://arxiv.org/abs/2309.16965)

    本文研究了在相对密集的图上组合优化问题中物理启发的图神经网络（PI-GNN）求解器的表现。通过数值实验，我们发现PI-GNN求解器在学习早期可能陷入所有变量为零的局部解。为了解决这个问题，我们通过控制连续性和离散性提出了一种改进方法。

    

    最近在组合优化（CO）问题中，图神经网络（GNNs）显示出巨大潜力。通过无监督学习找到近似解的受物理启发的GNN（PI-GNN）求解器在大规模CO问题上引起了极大关注。然而，对于相对密集图上的CO问题，贪婪算法的性能恶化，但对于PI-GNN求解器的性能却没有太多讨论。此外，由于PI-GNN求解器采用了放松策略，学习后需要从连续空间人工转换回原始离散空间，可能会破坏解的鲁棒性。本文通过数值实验证明了PI-GNN求解器在密集图上的CO问题的学习早期可能陷入局部解的情况，其中所有变量都为零。然后，我们通过控制连续性和离散性来解决这些问题。

    Recent advancements in combinatorial optimization (CO) problems emphasize the potential of graph neural networks (GNNs). The physics-inspired GNN (PI-GNN) solver, which finds approximate solutions through unsupervised learning, has attracted significant attention for large-scale CO problems. Nevertheless, there has been limited discussion on the performance of the PI-GNN solver for CO problems on relatively dense graphs where the performance of greedy algorithms worsens. In addition, since the PI-GNN solver employs a relaxation strategy, an artificial transformation from the continuous space back to the original discrete space is necessary after learning, potentially undermining the robustness of the solutions. This paper numerically demonstrates that the PI-GNN solver can be trapped in a local solution, where all variables are zero, in the early stage of learning for CO problems on the dense graphs. Then, we address these problems by controlling the continuity and discreteness of rela
    
[^94]: 无监督的对比一致排序与语言模型

    Unsupervised Contrast-Consistent Ranking with Language Models. (arXiv:2309.06991v1 [cs.LG])

    [http://arxiv.org/abs/2309.06991](http://arxiv.org/abs/2309.06991)

    无监督的对比一致排序与语言模型，通过训练一个受逻辑约束引导的探测模型，实现在多个语句中始终映射到对比的真-假极点的排序任务。

    

    语言模型包含基于排序的知识，并且是处理上下文排名任务的强大解决者。最近的研究关注于配对、点对和列表提示技术，以揭示语言模型的排序知识。然而，我们发现，即使在仔细校准和限制解码的情况下，基于提示的技术在产生的排序中也不总是自洽的。这促使我们探索一种受无监督探测方法Contrast-Consistent Search（CCS）启发的替代方法。这个想法是训练一个受逻辑约束引导的探测模型：模型对一个语句及其否定的表示必须在多个语句中始终映射到对比的真-假极点。我们假设类似的约束适用于所有项通过一致性对相关排序任务。

    Language models contain ranking-based knowledge and are powerful solvers of in-context ranking tasks. For instance, they may have parametric knowledge about the ordering of countries by size or may be able to rank reviews by sentiment. Recent work focuses on pairwise, pointwise, and listwise prompting techniques to elicit a language model's ranking knowledge. However, we find that even with careful calibration and constrained decoding, prompting-based techniques may not always be self-consistent in the rankings they produce. This motivates us to explore an alternative approach that is inspired by an unsupervised probing method called Contrast-Consistent Search (CCS). The idea is to train a probing model guided by a logical constraint: a model's representation of a statement and its negation must be mapped to contrastive true-false poles consistently across multiple statements. We hypothesize that similar constraints apply to ranking tasks where all items are related via consistent pair
    
[^95]: 通过多元宇宙分析评估模型设计决策对算法公平性的影响：一切，无处不在，全方位评估

    Everything, Everywhere All in One Evaluation: Using Multiverse Analysis to Evaluate the Influence of Model Design Decisions on Algorithmic Fairness. (arXiv:2308.16681v1 [stat.ML])

    [http://arxiv.org/abs/2308.16681](http://arxiv.org/abs/2308.16681)

    通过多元宇宙分析评估模型设计决策对算法公平性的影响，可以揭示算法决策系统中设计决策的关键作用。

    

    全球范围内的许多系统都利用算法决策来（部分）自动化以前由人类进行的决策。当设计良好时，这些系统承诺更客观的决策，同时节省大量资源，节约人力。然而，当算法决策系统设计不良时，可能会导致对社会群体进行歧视的不公平决策。算法决策系统的下游效应在很大程度上取决于系统设计和实施过程中的决策，因为数据中的偏见可能会在建模过程中缓解或加强。许多这些设计决策是隐含进行的，不知道它们确切地如何影响最终系统。因此，明确算法决策系统设计中的决策并了解这些决策如何影响结果系统的公平性非常重要。为了研究这个问题，我们借鉴了心理学领域的见解，并引入了多元宇宙分析方法。

    A vast number of systems across the world use algorithmic decision making (ADM) to (partially) automate decisions that have previously been made by humans. When designed well, these systems promise more objective decisions while saving large amounts of resources and freeing up human time. However, when ADM systems are not designed well, they can lead to unfair decisions which discriminate against societal groups. The downstream effects of ADMs critically depend on the decisions made during the systems' design and implementation, as biases in data can be mitigated or reinforced along the modeling pipeline. Many of these design decisions are made implicitly, without knowing exactly how they will influence the final system. It is therefore important to make explicit the decisions made during the design of ADM systems and understand how these decisions affect the fairness of the resulting system.  To study this issue, we draw on insights from the field of psychology and introduce the metho
    
[^96]: 将路径相关的NJ-ODE扩展到有噪声的观测和相关观测框架

    Extending Path-Dependent NJ-ODEs to Noisy Observations and a Dependent Observation Framework. (arXiv:2307.13147v1 [stat.ML])

    [http://arxiv.org/abs/2307.13147](http://arxiv.org/abs/2307.13147)

    该论文研究了将路径相关的NJ-ODE方法扩展到具有噪声观测和相关观测框架的问题。研究提出了两种扩展方法，并提供了理论保证和实证示例。

    

    路径相关的神经跳跃ODE (PD-NJ-ODE) 是一种用于预测具有不规则和不完整观测的连续时间随机过程的模型。具体而言，该方法通过学习给定不规则采样的不完整过去观测的最优预测。迄今为止，假设过程本身和坐标分别观测时间是独立的，并且假设观测是无噪声的。在这项工作中，我们讨论了两种扩展来解除这些限制，并提供了理论保证以及它们的实证示例。

    The Path-Dependent Neural Jump ODE (PD-NJ-ODE) is a model for predicting continuous-time stochastic processes with irregular and incomplete observations. In particular, the method learns optimal forecasts given irregularly sampled time series of incomplete past observations. So far the process itself and the coordinate-wise observation times were assumed to be independent and observations were assumed to be noiseless. In this work we discuss two extensions to lift these restrictions and provide theoretical guarantees as well as empirical examples for them.
    
[^97]: 大数据-供应链管理框架的预测：数据预处理和机器学习技术

    Big Data - Supply Chain Management Framework for Forecasting: Data Preprocessing and Machine Learning Techniques. (arXiv:2307.12971v1 [cs.LG])

    [http://arxiv.org/abs/2307.12971](http://arxiv.org/abs/2307.12971)

    本文介绍了一种新的大数据-供应链管理框架，通过数据预处理和机器学习技术实现供应链预测，优化操作管理、透明度，并讨论了幻影库存对预测的不利影响。

    

    本文旨在系统地识别和比较分析最先进的供应链预测策略和技术。提出了一个新的框架，将大数据分析应用于供应链管理中，包括问题识别、数据来源、探索性数据分析、机器学习模型训练、超参数调优、性能评估和优化，以及预测对人力、库存和整个供应链的影响。首先讨论了根据供应链策略收集数据的需求以及如何收集数据。文章讨论了根据周期或供应链目标需要不同类型的预测。推荐使用供应链绩效指标和误差测量系统来优化表现最佳的模型。还讨论了幻影库存对预测的不利影响以及管理决策依赖供应链绩效指标来确定模型性能参数和改进运营管理、透明度的问题。

    This article intends to systematically identify and comparatively analyze state-of-the-art supply chain (SC) forecasting strategies and technologies. A novel framework has been proposed incorporating Big Data Analytics in SC Management (problem identification, data sources, exploratory data analysis, machine-learning model training, hyperparameter tuning, performance evaluation, and optimization), forecasting effects on human-workforce, inventory, and overall SC. Initially, the need to collect data according to SC strategy and how to collect them has been discussed. The article discusses the need for different types of forecasting according to the period or SC objective. The SC KPIs and the error-measurement systems have been recommended to optimize the top-performing model. The adverse effects of phantom inventory on forecasting and the dependence of managerial decisions on the SC KPIs for determining model performance parameters and improving operations management, transparency, and 
    
[^98]: R-Learning与异质性治疗效果估计中的逆变数加权的连接

    The Connection Between R-Learning and Inverse-Variance Weighting for Estimation of Heterogeneous Treatment Effects. (arXiv:2307.09700v1 [stat.ME])

    [http://arxiv.org/abs/2307.09700](http://arxiv.org/abs/2307.09700)

    R-Learning在估计条件平均治疗效果时采用了逆变数加权的形式来稳定回归，并简化了偏差项。

    

    我们的动机是为了探讨广泛流行的“R-Learner”的性能。像其他估计条件平均治疗效果（CATEs）的方法一样，R-Learning可以表示为加权伪结果回归（POR）。先前对POR技术的比较已经仔细注意了伪结果转换的选择。然而，我们认为性能的主要驱动因素实际上是权重的选择。具体地说，我们认为R-Learning隐式地执行了加权形式的POR，其中权重稳定了回归，并允许对偏差项进行方便的简化。

    Our motivation is to shed light the performance of the widely popular "R-Learner." Like many other methods for estimating conditional average treatment effects (CATEs), R-Learning can be expressed as a weighted pseudo-outcome regression (POR). Previous comparisons of POR techniques have paid careful attention to the choice of pseudo-outcome transformation. However, we argue that the dominant driver of performance is actually the choice of weights. Specifically, we argue that R-Learning implicitly performs an inverse-variance weighted form of POR. These weights stabilize the regression and allow for convenient simplifications of bias terms.
    
[^99]: 快速经验场景

    Fast Empirical Scenarios. (arXiv:2307.03927v1 [stat.ML])

    [http://arxiv.org/abs/2307.03927](http://arxiv.org/abs/2307.03927)

    该论文提出了两种快速的经验场景提取算法，一种识别之前未观察到的场景并提供场景的协方差矩阵表示，另一种从已实现的世界状态中选择重要的数据点，并与高阶样本矩一致，这些算法计算效率高且适用于一致的基于场景的建模和高维数值积分。

    

    我们希望从大型和高维面板数据中提取一小部分与样本矩一致的代表性场景。在两种新算法中，第一种算法识别之前未观察到的场景，并提供了一种基于场景的协方差矩阵表示。第二种算法从已实现的世界状态中选择重要的数据点，并与高阶样本矩信息一致。这两种算法计算效率高，并可用于一致的基于场景的建模和高维数值积分。广泛的数值基准测试研究和在投资组合优化中的应用支持所提出的算法。

    We seek to extract a small number of representative scenarios from large and high-dimensional panel data that are consistent with sample moments. Among two novel algorithms, the first identifies scenarios that have not been observed before, and comes with a scenario-based representation of covariance matrices. The second proposal picks important data points from states of the world that have already realized, and are consistent with higher-order sample moment information. Both algorithms are efficient to compute, and lend themselves to consistent scenario-based modeling and high-dimensional numerical integration. Extensive numerical benchmarking studies and an application in portfolio optimization favor the proposed algorithms.
    
[^100]: 使用基于图形平滑的Gibbs采样优化蛋白质适应性。

    Optimizing protein fitness using Gibbs sampling with Graph-based Smoothing. (arXiv:2307.00494v1 [q-bio.BM])

    [http://arxiv.org/abs/2307.00494](http://arxiv.org/abs/2307.00494)

    使用基于图形平滑的Gibbs采样方法（GGS）优化蛋白质适应性，消除了突变距离的限制，同时提高了搜索效率。该方法在发现高适应性蛋白质方面达到了最先进水平。

    

    能够设计出在给定任务上具有更高适应性的新型蛋白质对许多医学领域来说都是革命性的。然而，通过穷举搜索海量序列空间是不可行的。以前的方法将搜索限制在从参考序列的小突变半径范围内，但这样的启发式方法极大地限制了设计空间。我们的工作旨在消除突变距离的限制，同时实现高效的探索。我们提出了基于图形平滑的Gibbs采样（GGS），它通过迭代应用带有梯度的Gibbs来提出有利的突变，并使用基于图形平滑的方法去除导致假阳性的噪声梯度。我们的方法在训练集中发现了高适应性蛋白质，最多具有8个突变。我们通过研究GFP和AAV设计问题、消融试验和基准模型来阐明结果。

    The ability to design novel proteins with higher fitness on a given task would be revolutionary for many fields of medicine. However, brute-force search through the combinatorially large space of sequences is infeasible. Prior methods constrain search to a small mutational radius from a reference sequence, but such heuristics drastically limit the design space. Our work seeks to remove the restriction on mutational distance while enabling efficient exploration. We propose Gibbs sampling with Graph-based Smoothing (GGS) which iteratively applies Gibbs with gradients to propose advantageous mutations using graph-based smoothing to remove noisy gradients that lead to false positives. Our method is state-of-the-art in discovering high-fitness proteins with up to 8 mutations from the training set. We study the GFP and AAV design problems, ablations, and baselines to elucidate the results. Code: https://github.com/kirjner/GGS
    
[^101]: 具有理论保障的差分隐私域自适应算法

    Differentially Private Domain Adaptation with Theoretical Guarantees. (arXiv:2306.08838v1 [cs.LG])

    [http://arxiv.org/abs/2306.08838](http://arxiv.org/abs/2306.08838)

    该论文提出了两种具有差分隐私保障的自适应算法，用于在受隐私约束且有限标记数据条件下，从公开源领域到目标领域进行监督域自适应。该算法能够解决一般的优化问题，并具有有利的理论学习保证。

    

    在许多应用中，学习者可用的标记数据受到隐私约束并相对有限。为了为目标领域导出更准确的预测器，通常有利于利用来自与目标领域相近的另一领域的公开标记数据。这是从公共源领域到私有目标领域的现代监督域自适应问题。我们提出了两种 $(\epsilon, \delta)$-差分隐私自适应算法，用于监督性自适应。对于其我们使用了一般的优化问题，该优化问题最近被证明具有有利的理论学习保证。我们的第一个算法是为具有线性预测器的回归设计的，并显示为解决凸优化问题。我们的第二个算法是一种更一般的解决方案，用于可能是非凸但Lipschitz和平滑的损失函数。虽然我们的主要目标是进行理论分析，但我们也报告了几个实验的结果。

    In many applications, the labeled data at the learner's disposal is subject to privacy constraints and is relatively limited. To derive a more accurate predictor for the target domain, it is often beneficial to leverage publicly available labeled data from an alternative domain, somewhat close to the target domain. This is the modern problem of supervised domain adaptation from a public source to a private target domain. We present two $(\epsilon, \delta)$-differentially private adaptation algorithms for supervised adaptation, for which we make use of a general optimization problem, recently shown to benefit from favorable theoretical learning guarantees. Our first algorithm is designed for regression with linear predictors and shown to solve a convex optimization problem. Our second algorithm is a more general solution for loss functions that may be non-convex but Lipschitz and smooth. While our main objective is a theoretical analysis, we also report the results of several experiment
    
[^102]: 大规模密集随机Kronecker图的分析和近似推断

    Analysis and Approximate Inference of Large and Dense Random Kronecker Graphs. (arXiv:2306.08489v1 [stat.ML])

    [http://arxiv.org/abs/2306.08489](http://arxiv.org/abs/2306.08489)

    本文对大规模密集随机Kronecker图进行了分析和近似推断，提出了“去噪声和求解”元算法，用于近似推断图参数，并具有较低的计算复杂度和性能保证。

    

    随机图模型在科学和工业中发挥着越来越重要的作用，并在各种领域中得到应用，包括社交和交通网络、推荐系统和分子遗传学。本文对\cite{leskovec2010kronecker}中提出的随机Kronecker图模型进行了深入的分析，当图顶点数量$N$很大时。基于最近在随机矩阵理论方面的进展，我们证明，在密集的情况下，随机Kronecker图邻接矩阵近似遵循一个信号加噪声模型，其中信号矩阵的秩很小（最多为$\log N$阶），在图参数中是线性的，而随机的噪声矩阵具有四分之一圆形奇异值分布。这个观察允许我们提出了一种“去噪声和求解”元算法来近似推断图参数，具有较低的计算复杂度和（渐近的）性能保证。通过图i的数值实验进行了验证。

    Random graph models are playing an increasingly important role in science and industry, and finds their applications in a variety of fields ranging from social and traffic networks, to recommendation systems and molecular genetics. In this paper, we perform an in-depth analysis of the random Kronecker graph model proposed in \cite{leskovec2010kronecker}, when the number of graph vertices $N$ is large. Built upon recent advances in random matrix theory, we show, in the dense regime, that the random Kronecker graph adjacency matrix follows approximately a signal-plus-noise model, with a small-rank (of order at most $\log N$) signal matrix that is linear in the graph parameters and a random noise matrix having a quarter-circle-form singular value distribution. This observation allows us to propose a ``denoise-and-solve'' meta algorithm to approximately infer the graph parameters, with reduced computational complexity and (asymptotic) performance guarantee. Numerical experiments of graph i
    
[^103]: 利用观测偏差提高矩阵补全的方法研究

    Exploiting Observation Bias to Improve Matrix Completion. (arXiv:2306.04775v1 [cs.LG])

    [http://arxiv.org/abs/2306.04775](http://arxiv.org/abs/2306.04775)

    本研究利用观测偏差来改进矩阵补全问题，提出一个简单的两阶段算法，实现了与对未观测协变量的监督学习性能相当的结果。

    

    我们考虑了一种变形的矩阵补全问题，其中输入数据以偏差的方式呈现，类似于Ma和Chen所引入的模型。我们的目标是利用偏差与感兴趣的结果之间的共享信息来改进预测。为此，我们提出了一个简单的两阶段算法：（i）将观测模式解释为完全观测的噪声矩阵，我们对观测模式应用传统的矩阵补全方法来估计潜在因素之间的距离； (ii)我们对恢复的特征应用监督学习来填补缺失观察。我们建立了有限样本误差率，这些误差率与相应的监督学习参数率相竞争，这表明我们的学习性能与使用未观测协变量相当。实证评估使用真实世界数据集反映了类似的表现。

    We consider a variant of matrix completion where entries are revealed in a biased manner, adopting a model akin to that introduced by Ma and Chen. Instead of treating this observation bias as a disadvantage, as is typically the case, our goal is to exploit the shared information between the bias and the outcome of interest to improve predictions. Towards this, we propose a simple two-stage algorithm: (i) interpreting the observation pattern as a fully observed noisy matrix, we apply traditional matrix completion methods to the observation pattern to estimate the distances between the latent factors; (ii) we apply supervised learning on the recovered features to impute missing observations. We establish finite-sample error rates that are competitive with the corresponding supervised learning parametric rates, suggesting that our learning performance is comparable to having access to the unobserved covariates. Empirical evaluation using a real-world dataset reflects similar performance g
    
[^104]: 将NP困难的最小最大路径问题作为具有公平背景的顺序生成来解决

    Solving NP-hard Min-max Routing Problems as Sequential Generation with Equity Context. (arXiv:2306.02689v1 [cs.LG])

    [http://arxiv.org/abs/2306.02689](http://arxiv.org/abs/2306.02689)

    本文提出了一个新的深度学习框架Equity-Transformer来解决大规模的最小最大路径问题。该模型利用可扩展的深度学习模型进行顺序决策，并生成考虑公平工作负载的顺序动作。研究显示，Equity-Transformer在两个代表性最小最大路径问题中具有卓越的性能。

    

    最小最大路径问题旨在最小化所有代理商协同访问所有城市的最大旅游长度，即完成时间。这些问题包括有影响力的实际应用，但被认为是NP困难的。现有方法面临挑战，特别是在需要协调众多代理商覆盖数千个城市的大规模问题中。本文提出了一个新的深度学习框架来解决大规模的最小最大路径问题。我们将多个代理商的同时决策建模为顺序生成过程，允许利用可扩展的深度学习模型进行顺序决策。在顺序近似问题中，我们提出了一个可扩展的上下文Transformer模型Equity-Transformer，它生成考虑其他代理商之间公平工作负载的顺序动作。Equity-Transformer的有效性通过其在两个代表性最小最大路径问题中具有卓越的性能得到证明。

    Min-max routing problems aim to minimize the maximum tour length among agents as they collaboratively visit all cities, i.e., the completion time. These problems include impactful real-world applications but are known as NP-hard. Existing methods are facing challenges, particularly in large-scale problems that require the coordination of numerous agents to cover thousands of cities. This paper proposes a new deep-learning framework to solve large-scale min-max routing problems. We model the simultaneous decision-making of multiple agents as a sequential generation process, allowing the utilization of scalable deep-learning models for sequential decision-making. In the sequentially approximated problem, we propose a scalable contextual Transformer model, Equity-Transformer, which generates sequential actions considering an equitable workload among other agents. The effectiveness of Equity-Transformer is demonstrated through its superior performance in two representative min-max routing 
    
[^105]: 关于ReLU网络的大小无关样本复杂度

    On Size-Independent Sample Complexity of ReLU Networks. (arXiv:2306.01992v1 [cs.LG])

    [http://arxiv.org/abs/2306.01992](http://arxiv.org/abs/2306.01992)

    本文研究了ReLU神经网络的样本复杂度，给出了一个现有方法精细化的结果，实现了无深度依赖性的上界。

    

    我们从泛化的角度研究了学习ReLU神经网络的样本复杂度。在权重矩阵上给定范数约束的情况下，一个常见的方法是估计相关函数类的Rademacher复杂度。之前Golowich-Rakhlin-Shamir (2020)获得了一个不依赖于网络大小的（与Frobenius范数的乘积成比例）上界，除了一个平方根深度的因子。我们给出了一个精细化的结果，通常根本没有明显的深度依赖性。

    We study the sample complexity of learning ReLU neural networks from the point of view of generalization. Given norm constraints on the weight matrices, a common approach is to estimate the Rademacher complexity of the associated function class. Previously Golowich-Rakhlin-Shamir (2020) obtained a bound independent of the network size (scaling with a product of Frobenius norms) except for a factor of the square-root depth. We give a refinement which often has no explicit depth-dependence at all.
    
[^106]: 为什么在对抗训练中会同时出现干净泛化和强健过拟合现象？

    Why Clean Generalization and Robust Overfitting Both Happen in Adversarial Training. (arXiv:2306.01271v1 [cs.LG])

    [http://arxiv.org/abs/2306.01271](http://arxiv.org/abs/2306.01271)

    对抗训练是训练深度神经网络抗击对抗扰动的标准方法, 其学习机制导致干净泛化和强健过拟合现象同时发生。

    

    对抗训练是训练深度神经网络抗击对抗扰动的标准方法。与在标准深度学习环境中出现惊人的干净泛化能力类似，通过对抗训练训练的神经网络也能很好地泛化到未见过的干净数据。然而，与干净泛化不同的是，尽管对抗训练能够实现低鲁棒训练误差，仍存在显著的鲁棒泛化距离，这促使我们探索在学习过程中导致干净泛化和强健过拟合现象同时发生的机制。本文提供了对抗训练中这种现象的理论理解。首先，我们提出了对抗训练的理论框架，分析了特征学习过程，解释了对抗训练如何导致网络学习者进入到干净泛化和强健过拟合状态。具体来说，我们证明了，通过迫使学习器成为强预测网络，对抗训练将导致干净泛化和鲁棒过拟合现象同时发生。

    Adversarial training is a standard method to train deep neural networks to be robust to adversarial perturbation. Similar to surprising $\textit{clean generalization}$ ability in the standard deep learning setting, neural networks trained by adversarial training also generalize well for $\textit{unseen clean data}$. However, in constrast with clean generalization, while adversarial training method is able to achieve low $\textit{robust training error}$, there still exists a significant $\textit{robust generalization gap}$, which promotes us exploring what mechanism leads to both $\textit{clean generalization and robust overfitting (CGRO)}$ during learning process. In this paper, we provide a theoretical understanding of this CGRO phenomenon in adversarial training. First, we propose a theoretical framework of adversarial training, where we analyze $\textit{feature learning process}$ to explain how adversarial training leads network learner to CGRO regime. Specifically, we prove that, u
    
[^107]: 拉普拉斯逼近神经加性模型：贝叶斯推理提高解释性

    Laplace-Approximated Neural Additive Models: Improving Interpretability with Bayesian Inference. (arXiv:2305.16905v1 [stat.ML])

    [http://arxiv.org/abs/2305.16905](http://arxiv.org/abs/2305.16905)

    本文提出了拉普拉斯逼近神经加性模型，该模型从贝叶斯角度考虑加性结构，在恢复的特征交互中提供可信区间，提供可处理的边缘似然估计，可用于执行隐式特征选择并对特征对进行排名。

    

    深度神经网络（DNN）在许多领域取得了成功应用，但它们的黑盒性质阻碍了解释性。神经加性模型（NAM）解决了这个问题，将网络分为加性子网络，从而使输入特征和预测之间的交互变得明显。在本文中，我们从贝叶斯角度考虑加性结构，并开发了一个实用的拉普拉斯逼近方法。这种方法在以下三个方面提高了可解释性：a）它通过估计子网络的函数空间不确定性为恢复的特征交互提供可信区间；b）它提供可处理的边缘似然估计，可用于通过经验贝叶斯过程执行特征的隐式选择；c）它可用于对特征对进行排名，作为精细调整的交互模型候选。我们在几个基准数据集上实证表明，我们提出的拉普拉斯逼近神经加性模型（LA-NAM）提高了NAM模型的可解释性，并进一步揭示了学习到的子网络的交互结构。

    Deep neural networks (DNNs) have found successful applications in many fields, but their black-box nature hinders interpretability. This is addressed by the neural additive model (NAM), in which the network is divided into additive sub-networks, thus making apparent the interaction between input features and predictions. In this paper, we approach the additive structure from a Bayesian perspective and develop a practical Laplace approximation. This enhances interpretability in three primary ways: a) It provides credible intervals for the recovered feature interactions by estimating function-space uncertainty of the sub-networks; b) it yields a tractable estimate of the marginal likelihood, which can be used to perform an implicit selection of features through an empirical Bayes procedure; and c) it can be used to rank feature pairs as candidates for second-order interactions in fine-tuned interaction models. We show empirically that our proposed Laplace-approximated NAM (LA-NAM) improv
    
[^108]: 神经不完全分解：学习共轭梯度法的预处理器

    Neural incomplete factorization: learning preconditioners for the conjugate gradient method. (arXiv:2305.16368v1 [math.OC])

    [http://arxiv.org/abs/2305.16368](http://arxiv.org/abs/2305.16368)

    本文提出了一种名为神经不完全分解的新方法，利用自监督训练的图神经网络生成适用于特定问题域的有效预处理器。其通过替换传统手工预处理器显着提高了收敛和计算效率，在合成和真实问题上进行的实验均表现出竞争力。

    

    本文提出了一种新型的数据驱动方法，用于加速科学计算和优化中遇到的大规模线性方程组求解。我们的方法利用自监督训练图神经网络，生成适用于特定问题域的有效预处理器。通过替换与共轭梯度法一起使用的传统手工预处理器，我们的方法（称为神经不完全分解）显着加速了收敛和计算效率。我们的方法的核心是一种受稀疏矩阵理论启发的新型消息传递块，它与寻找矩阵的稀疏分解的目标相一致。我们在合成问题和来自科学计算的真实问题上评估了我们的方法。我们的结果表明，神经不完全分解始终优于最常见的通用预处理器，包括不完全的Cholesky方法，在收敛速度和计算效率方面表现出竞争力。

    In this paper, we develop a novel data-driven approach to accelerate solving large-scale linear equation systems encountered in scientific computing and optimization. Our method utilizes self-supervised training of a graph neural network to generate an effective preconditioner tailored to the specific problem domain. By replacing conventional hand-crafted preconditioners used with the conjugate gradient method, our approach, named neural incomplete factorization (NeuralIF), significantly speeds-up convergence and computational efficiency. At the core of our method is a novel message-passing block, inspired by sparse matrix theory, that aligns with the objective to find a sparse factorization of the matrix. We evaluate our proposed method on both a synthetic and a real-world problem arising from scientific computing. Our results demonstrate that NeuralIF consistently outperforms the most common general-purpose preconditioners, including the incomplete Cholesky method, achieving competit
    
[^109]: 通过重新标记最小训练子集来翻转预测

    Relabel Minimal Training Subset to Flip a Prediction. (arXiv:2305.12809v1 [cs.LG])

    [http://arxiv.org/abs/2305.12809](http://arxiv.org/abs/2305.12809)

    本文利用扩展影响函数提出了一种有效的识别和重新标记最小训练子集的方法，并证明其始终能够成功翻转测试结果，同时还提供了挑战模型预测、评估模型鲁棒性和洞察训练集偏差等多重作用。

    

    Yang等人发现，仅删除1%的训练数据就可能导致预测结果翻转。鉴于机器学习模型中存在噪声数据的普遍性，本文提出了一个问题：在模型训练之前通过重新标记一个小的训练数据子集可否导致测试结果翻转？本文利用扩展影响函数提出了一种有效的识别和重新标记这种子集的方法，并证明了其始终能够产生成功的结果。这种机制有多重作用：（1）提供了一种补充方法，可以通过恢复可能错误标记的训练数据来挑战模型预测；（2）评估模型的鲁棒性，因为本文发现子集的大小与训练集中噪声数据的比例之间存在显著关系；（3）提供了洞察训练集偏差的见解。据我们所知，这项工作代表了对识别最小训练子集问题的第一次研究。

    Yang et al. (2023) discovered that removing a mere 1% of training points can often lead to the flipping of a prediction. Given the prevalence of noisy data in machine learning models, we pose the question: can we also result in the flipping of a test prediction by relabeling a small subset of the training data before the model is trained? In this paper, utilizing the extended influence function, we propose an efficient procedure for identifying and relabeling such a subset, demonstrating consistent success. This mechanism serves multiple purposes: (1) providing a complementary approach to challenge model predictions by recovering potentially mislabeled training points; (2) evaluating model resilience, as our research uncovers a significant relationship between the subset's size and the ratio of noisy data in the training set; and (3) offering insights into bias within the training set. To the best of our knowledge, this work represents the first investigation into the problem of identi
    
[^110]: Q-malizing流和无穷小密度比估计

    Q-malizing flow and infinitesimal density ratio estimation. (arXiv:2305.11857v1 [stat.ML])

    [http://arxiv.org/abs/2305.11857](http://arxiv.org/abs/2305.11857)

    研究提出了一种可以从一个数据分布P传输到任意访问通过有限样本的Q的流模型。这个模型通过神经ODE模型进行，可以进行无穷小DRE。

    

    连续的正则化流在生成任务中被广泛使用，其中流网络从数据分布P传输到正态分布。一种能够从P传输到任意Q的流模型，其中P和Q都可通过有限样本访问，将在各种应用兴趣中使用，特别是在最近开发的望远镜密度比估计中（DRE），它需要构建中间密度以在P和Q之间建立桥梁。在这项工作中，我们提出了这样的“Q-malizing流”，通过神经ODE模型进行，该模型通过经验样本的可逆传输从P到Q（反之亦然），并通过最小化传输成本进行正则化。训练好的流模型使我们能够沿与时间参数化的log密度进行无穷小DRE，通过训练附加的连续时间流网络使用分类损失来估计log密度的时间偏导数。通过积分时间得分网络

    Continuous normalizing flows are widely used in generative tasks, where a flow network transports from a data distribution $P$ to a normal distribution. A flow model that can transport from $P$ to an arbitrary $Q$, where both $P$ and $Q$ are accessible via finite samples, would be of various application interests, particularly in the recently developed telescoping density ratio estimation (DRE) which calls for the construction of intermediate densities to bridge between $P$ and $Q$. In this work, we propose such a ``Q-malizing flow'' by a neural-ODE model which is trained to transport invertibly from $P$ to $Q$ (and vice versa) from empirical samples and is regularized by minimizing the transport cost. The trained flow model allows us to perform infinitesimal DRE along the time-parametrized $\log$-density by training an additional continuous-time flow network using classification loss, which estimates the time-partial derivative of the $\log$-density. Integrating the time-score network
    
[^111]: 基于反Lipschitz约束的解码器网络控制后验坍塌

    Controlling Posterior Collapse by an Inverse Lipschitz Constraint on the Decoder Network. (arXiv:2304.12770v1 [cs.LG])

    [http://arxiv.org/abs/2304.12770](http://arxiv.org/abs/2304.12770)

    本文提出了一种基于反Lipschitz约束的解码器网络，可以简单明了地控制广泛的VAE模型的后验坍塌程度，并带有具体的理论保证。

    

    变分自编码器（VAE）是深度生成模型中取得巨大成功的一种。然而，在实践中，它们存在一个称为后验坍塌的问题，当编码器与没有考虑输入数据的潜在结构的先验重合或坍塌时就会发生。本文介绍了一种基于反Lipschitz神经网络的解码器，基于这个架构，提供了一种新方法，可以简单明了地控制广泛的VAE模型的后验坍塌程度，并带有具体的理论保证。我们还通过几个数值实验证明了我们方法的有效性。

    Variational autoencoders (VAEs) are one of the deep generative models that have experienced enormous success over the past decades. However, in practice, they suffer from a problem called posterior collapse, which occurs when the encoder coincides, or collapses, with the prior taking no information from the latent structure of the input data into consideration. In this work, we introduce an inverse Lipschitz neural network into the decoder and, based on this architecture, provide a new method that can control in a simple and clear manner the degree of posterior collapse for a wide range of VAE models equipped with a concrete theoretical guarantee. We also illustrate the effectiveness of our method through several numerical experiments.
    
[^112]: 论随机遍历的强稳定性

    On the strong stability of ergodic iterations. (arXiv:2304.04657v1 [math.PR])

    [http://arxiv.org/abs/2304.04657](http://arxiv.org/abs/2304.04657)

    本论文研究了迭代随机函数生成的过程的强稳定性，证明了适用于递归映射的温和条件下的强稳定性，并且提供了多个应用及相关领域的新结果。

    

    我们重新审视了由随机函数迭代生成的过程，这些函数由一个平稳且符合遍历条件的序列驱动。如果存在一个随机初始化使得该过程是稳定和遍历的，并且对于任何其他初始化，两个过程之间的差异几乎肯定收敛于零，那么这样的过程被称为强稳定。在对应递归映射上施加一些温和的条件，而不在驱动序列上施加任何条件下，我们展示了迭代的强稳定性。多个应用被研究，如随机逼近和排队。此外，我们推导出了具有依赖噪声的 Langevin 型迭代和多型分支过程的新结果。

    We revisit processes generated by iterated random functions driven by a stationary and ergodic sequence. Such a process is called strongly stable if a random initialization exists, for which the process is stationary and ergodic, and for any other initialization, the difference of the two processes converges to zero almost surely. Under some mild conditions on the corresponding recursive map, without any condition on the driving sequence, we show the strong stability of iterations. Several applications are surveyed such as stochastic approximation and queuing. Furthermore, new results are deduced for Langevin-type iterations with dependent noise and for multitype branching processes.
    

