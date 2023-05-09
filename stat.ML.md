# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gaussian process deconvolution.](http://arxiv.org/abs/2305.04871) | 本文提出了一种基于高斯过程的贝叶斯非参数方法，可以解决连续时间信号的去卷积问题，适用于观测值中可能存在缺失数据且信号滤波器未知的情况。 |
| [^2] | [Local Optimization Achieves Global Optimality in Multi-Agent Reinforcement Learning.](http://arxiv.org/abs/2305.04819) | 本文提出了一种多智能体PPO算法，利用局部优化达到全局最优，具有统计保证，这是首个能在协作马尔可夫博弈中证明收敛的算法。 |
| [^3] | [Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods.](http://arxiv.org/abs/2305.04634) | 研究提出了使用CNN学习空间过程的似然函数。即使在没有确切似然函数的情况下，通过分类任务进行的神经网络的训练，可以隐式地学习似然函数。使用Platt缩放可以提高神经似然面的准确性。 |
| [^4] | [The Signature Kernel.](http://arxiv.org/abs/2305.04625) | Signature核函数是一种正定的顺序数据核函数，具有高效的计算算法和强大的实证性能。 |
| [^5] | [Building Neural Networks on Matrix Manifolds: A Gyrovector Space Approach.](http://arxiv.org/abs/2305.04560) | 本文通过将陀螺矢量空间中的一些概念推广到SPD和Grassmann流形，提出了在这些流形上构建神经网络的新模型和新层，并在人类动作识别和知识图谱完成两个应用中展示了其有效性。 |
| [^6] | [Q&A Label Learning.](http://arxiv.org/abs/2305.04539) | 本文提出了一种新的标注方法，称为Q&A标注。该方法涉及到一个问题生成器和一个回答者，通过一系列问题将相应的标签分配给实例，并通过导出损失函数，对使用Q&A标签的实例的监督式机器学习的分类风险进行了评估。 |
| [^7] | [Axiomatization of Interventional Probability Distributions.](http://arxiv.org/abs/2305.04479) | 本论文提供了一种简单和清晰的因果理论，它不需要使用任何建模假设，包括大多数具有潜在变量和因果循环的情况，并且不假定存在底层真正的因果图-事实上，它是因果图的副产品。 |
| [^8] | [New metrics and search algorithms for weighted causal DAGs.](http://arxiv.org/abs/2305.04445) | 本研究提供了针对加权因果 DAGs的新度量和搜索算法，发现了用于自适应干预的因果图，提供了一个新的基准来捕捉搜索算法的最坏干预成本，并提供自适应搜索算法实现对数逼近。 |
| [^9] | [A Variational Perspective on Solving Inverse Problems with Diffusion Models.](http://arxiv.org/abs/2305.04391) | 该论文提出了一种通过去噪扩散过程自然地导致正则化的变分方法（RED-Diff），可用于解决不同反问题。加权机制可以衡量不同时间步长的去噪器的贡献。 |
| [^10] | [A Generalized Framework for Predictive Clustering and Optimization.](http://arxiv.org/abs/2305.04364) | 本文提出了一种通用的预测性聚类优化框架，该框架允许不同的聚类定义和回归、分类目标，并提供了高度可扩展的算法以解决大型数据集的优化问题。 |
| [^11] | [Fast parameter estimation of Generalized Extreme Value distribution using Neural Networks.](http://arxiv.org/abs/2305.04341) | 本论文提出了一种利用神经网络进行计算有效的广义极值分布参数估计的方法，与传统的极大似然方法相比，具有相似的准确性和显著的计算加速度。 |
| [^12] | [Classification Tree Pruning Under Covariate Shift.](http://arxiv.org/abs/2305.04335) | 本文提出了一种基于协变量转移的分类树剪枝方法，可以访问大部分来自分布 $P_{X，Y}$ 的数据，但是只能获得来自拥有不同 $X$-边缘的目标分布 $Q_{X，Y}$ 的少量数据。使用的优化标准是一个关于分布 $P_{X} \to Q_{X}$ 的 \emph{平均差异}，该标准可以显著放宽最近提出的 \emph{转移指数}，最终可以得到最优的剪枝结果。 |
| [^13] | [Learning Mixtures of Gaussians with Censored Data.](http://arxiv.org/abs/2305.04127) | 本文提出了一种学习高斯混合模型的算法，该算法仅需要很少的样本且能够对权重和均值进行准确估计。 |
| [^14] | [The Fundamental Limits of Structure-Agnostic Functional Estimation.](http://arxiv.org/abs/2305.04116) | 一阶去偏方法在最小二乘意义下在干扰函数生存在特定函数空间时被证明是次优的，这促进了“高阶”去偏方法的发展。 |
| [^15] | [Efficient Learning for Selecting Top-m Context-Dependent Designs.](http://arxiv.org/abs/2305.04086) | 该论文采用贝叶斯框架下的随机动态规划方法，开发一种顺序抽样策略，提高了选择上下文相关设计的前m个的效率。 |
| [^16] | [Adam-family Methods for Nonsmooth Optimization with Convergence Guarantees.](http://arxiv.org/abs/2305.03938) | 本文提出了一种新的双时间尺度框架，证明了其在温和条件下收敛性，该框架包括了各种流行的Adam家族算法，用于训练无平滑神经网络和应对重尾噪声的需求，并通过实验表明了其效率和鲁棒性。 |
| [^17] | [Trajectory-oriented optimization of stochastic epidemiological models.](http://arxiv.org/abs/2305.03926) | 本论文提出了一个基于轨迹的优化方法来处理随机流行病学模型，可以找到与实际观测值接近的实际轨迹，而不是仅使平均模拟结果与实测数据相符。 |
| [^18] | [Twin support vector quantile regression.](http://arxiv.org/abs/2305.03894) | TSVQR能够捕捉现代数据中的异质和不对称信息，并有效地描述了所有数据点的异质分布信息。通过构造两个较小的二次规划问题，TSVQR生成两个非平行平面，测量每个分位数水平下限和上限之间的分布不对称性。在多个实验中，TSVQR优于以前的分位数回归方法。 |
| [^19] | [On High-dimensional and Low-rank Tensor Bandits.](http://arxiv.org/abs/2305.03884) | 本研究提出了一个通用的张量赌博机模型，其中行动和系统参数由张量表示，着重于未知系统张量为低秩的情况。所开发的 TOFU 算法首先利用灵活的张量回归技术估计与系统张量相关联的低维子空间，然后将原始问题转换成一个具有系统参数范数约束的新问题，并采用范数约束赌博子例程解决。 |
| [^20] | [Learning Stochastic Dynamical System via Flow Map Operator.](http://arxiv.org/abs/2305.03874) | 该论文提出了一种通过测量数据学习未知随机动力学系统的数值框架随机流映射学习（sFML），在不同类型的随机系统上进行的全面实验证明了 sFML 的有效性。 |
| [^21] | [No-Regret Constrained Bayesian Optimization of Noisy and Expensive Hybrid Models using Differentiable Quantile Function Approximations.](http://arxiv.org/abs/2305.03824) | 本文提出了一种新颖的算法，CUQB，来解决复合函数（混合模型）的高效约束全局优化问题，并取得了良好的效果，在合成和真实的应用程序中均得到了验证，包括进行了最优控制的流体流量和拓扑结构优化，后者比当前最先进的设计强2倍。 |
| [^22] | [Calibration Assessment and Boldness-Recalibration for Binary Events.](http://arxiv.org/abs/2305.03780) | 本研究提出了一种假设检验和贝叶斯模型选择方法来评估校准，并提供一种大胆再校准策略，使实践者能够在满足所需的校准水平的情况下负责任地增强预测。 |
| [^23] | [Majorizing Measures, Codes, and Information.](http://arxiv.org/abs/2305.02960) | 本文介绍了一种基于信息论的趋势测度定理视角，该视角将随机过程的有限性与索引度量空间元素的有效可变长度编码的存在性相关联。 |
| [^24] | [Shotgun crystal structure prediction using machine-learned formation energies.](http://arxiv.org/abs/2305.02158) | 本研究使用机器学习方法在多个结构预测标准测试中精确识别含有100个以上原子的许多材料的全局最小结构，并以单次能量评估为基础，取代了重复的第一原理能量计算过程。 |
| [^25] | [Commentary on explainable artificial intelligence methods: SHAP and LIME.](http://arxiv.org/abs/2305.02012) | 这篇评论对可解释人工智能方法 SHAP 和 LIME 进行了评述和比较，提出了一个框架且突出了它们的优缺点。 |
| [^26] | [Performative Prediction with Bandit Feedback: Learning through Reparameterization.](http://arxiv.org/abs/2305.01094) | 本文提出一种新的在线反馈的实现式预测框架，解决了在模型部署自身改变数据分布的情况下优化准确性的问题。 |
| [^27] | [Mixtures of Gaussian process experts based on kernel stick-breaking processes.](http://arxiv.org/abs/2304.13833) | 提出了一种新的基于核棍棒过程的高斯过程专家混合模型，能够维持直观吸引力并提高模型性能，具有实用性。 |
| [^28] | [A mean-field games laboratory for generative modeling.](http://arxiv.org/abs/2304.13534) | 本文提出了使用均场博弈作为实验室对生成模型进行设计和分析的方法，并建立了这种方法与主要流动和扩散型生成模型之间的关联。通过研究每个生成模型与它们相关的 MFG 的最优条件，本文提出了一个基于双人 MFG 的新的生成模型，该模型在提高样本多样性和逼真度的同时改善了解缠结和公平性。 |
| [^29] | [Repeated Principal-Agent Games with Unobserved Agent Rewards and Perfect-Knowledge Agents.](http://arxiv.org/abs/2304.07407) | 本文提出了一个利用多臂老虎机框架结构来处理未知代理奖励的策略，并证明了其性能是渐进最优的。 |
| [^30] | [Selecting Robust Features for Machine Learning Applications using Multidata Causal Discovery.](http://arxiv.org/abs/2304.05294) | 本文提出了一种多数据因果特征选择方法，它可以同时处理一组时间序列数据集，生成一个单一的因果驱动集，并且可以过滤掉因果虚假链接，最终输入到机器学习模型中预测目标。 |
| [^31] | [Deep Momentum Multi-Marginal Schr\"odinger Bridge.](http://arxiv.org/abs/2303.01751) | 该论文提出了一种新的计算框架DMSB，它可以学习满足时间上位置边际约束的随机系统的平滑度量值样条，用于解决高维多边际轨迹推断任务，并在实验中表现出显著的性能优势。同时，该框架还为解决具有各种类型的边际约束的随机轨迹重建任务提供了一个通用框架。 |
| [^32] | [Approximately Bayes-Optimal Pseudo Label Selection.](http://arxiv.org/abs/2302.08883) | 本文介绍了BPLS，一种用于PLS的贝叶斯框架，通过解析逼近选择标签实例的标准，以避免由过度自信但错误预测的实例选择而导致的确认偏差问题。 |
| [^33] | [Robust online active learning.](http://arxiv.org/abs/2302.00422) | 本文提出了一种自适应方法，用于鲁棒的在线主动学习，并在受污染的数据流中证明了其性能表现优异，同时确保了稳定性并减少异常值的负面影响。 |
| [^34] | [Combinatorial Causal Bandits without Graph Skeleton.](http://arxiv.org/abs/2301.13392) | 本文研究了在二值一般因果模型和BGLMs上不考虑图骨架的组合因果赌博机问题，提出了可在BGLMs上实现的无需图骨架的遗憾最小化算法，达到了与依赖于图结构的最先进算法相同的渐进遗憾率$O(\sqrt{T}\ln T)$。 |
| [^35] | [A mixed-categorical correlation kernel for Gaussian process.](http://arxiv.org/abs/2211.08262) | 提出一种新的混合类别相关核的高斯过程代理，相较于其他现有模型在分析和工程问题上表现更好。 |
| [^36] | [Learning-Augmented Private Algorithms for Multiple Quantile Release.](http://arxiv.org/abs/2210.11222) | 本文提出一种新的隐私保护方法：使用学习增强算法框架，为多分位数发布任务提供可扩展的预测质量误差保证。 |
| [^37] | [Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models.](http://arxiv.org/abs/2208.09399) | 本文提出了一种新的状态空间框架(SSSD)来插补时间序列数据中的缺失值和进行预测，在各种数据集和不同的缺失情况下，SSSD都表现出更好的性能，并可以有效处理黑屏缺失情况。 |
| [^38] | [Non-Asymptotic Analysis of Ensemble Kalman Updates: Effective Dimension and Localization.](http://arxiv.org/abs/2208.03246) | 本文开发了集合卡尔曼更新的非渐近分析，解释了为什么在先前的协方差具有中等的有效维度、快速谱衰减或近似稀疏的情况下，小的集合大小就足够了。 |
| [^39] | [Wasserstein multivariate auto-regressive models for modeling distributional time series and its application in graph learning.](http://arxiv.org/abs/2207.05442) | 本文提出了一种新的自回归模型，用于分析多元分布时间序列。并且在Wasserstein空间中建模了随机对象，提供了该模型的解的存在性和一致估计器。此方法可以应用于年龄分布和自行车共享网络的观察数据。 |
| [^40] | [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second.](http://arxiv.org/abs/2207.01848) | TabPFN是一种可以在不到一秒钟内完成小型表格数据集的监督分类的Transformer，无需超参数调整，并且具有竞争力。它使用先验适应网络（PFN）逼近基于先验的贝叶斯推断，先验融合了因果推理的思想。 |
| [^41] | [Learning Group Importance using the Differentiable Hypergeometric Distribution.](http://arxiv.org/abs/2203.01629) | 本文提出了不同iable hypergeometric distribution，使用重要性学习方法来解决在许多应用程序中将一组元素划分为先验未知大小的子集的问题，并在弱监督学习和聚类方面优于以前的方法。 |
| [^42] | [Fairness Implications of Encoding Protected Categorical Attributes.](http://arxiv.org/abs/2201.11358) | 该研究比较了两种常用的编码方法-“one-hot编码”和“target编码”，并探讨了其对机器学习模型性能和公平性的影响。 |
| [^43] | [CausalSim: A Causal Framework for Unbiased Trace-Driven Simulation.](http://arxiv.org/abs/2201.01811) | CausalSim提出了一种因果框架，通过学习系统动态和潜在因素的因果模型，消除追踪数据中的偏差，解决了当前追踪驱动仿真器的缺陷。 |
| [^44] | [A Unified and Constructive Framework for the Universality of Neural Networks.](http://arxiv.org/abs/2112.14877) | 本论文提出了神经网络普适性的建构框架，任何nAI激活函数都是普适的，该框架具有统一、构造性和新视角的优势。 |
| [^45] | [Learning Safety Filters for Unknown Discrete-Time Linear Systems.](http://arxiv.org/abs/2111.00631) | 本论文提出了一种基于学习的安全滤波器，针对带有未知模型和未知协方差的高斯噪声的离散时间线性时不变系统，通过收紧安全约束和构建鲁棒优化问题，以最小程度地修改名义控制动作，以高概率确保安全性。 |
| [^46] | [Weighting-Based Treatment Effect Estimation via Distribution Learning.](http://arxiv.org/abs/2012.13805) | 本文提出了一种基于分布学习的加权方法，通过学习协变量在治疗组和对照组的分布并利用比率作为权重来估计治疗效果，以缓解现有加权方法中模型错误设置的问题。 |
| [^47] | [Survival Modeling of Suicide Risk with Rare and Uncertain Diagnoses.](http://arxiv.org/abs/2009.02597) | 该研究针对罕见和不确定诊断的自杀风险进行生存建模，采用医疗索赔数据研究了自杀未遂的患者随后的自杀未遂风险，通过开发一种综合的Cox淘汰模型来完成生存回归。 |
| [^48] | [Sequential Gaussian Processes for Online Learning of Nonstationary Functions.](http://arxiv.org/abs/1905.10003) | 本文提出了一种基于顺序蒙特卡罗算法的连续高斯过程模型，以解决高斯过程模型的计算复杂度高，难以在线顺序更新的问题，同时允许拟合具有非平稳性质的函数。方法优于现有最先进方法的性能。 |

# 详细

[^1]: 高斯过程去卷积问题的贝叶斯非参数方法

    Gaussian process deconvolution. (arXiv:2305.04871v1 [stat.ML])

    [http://arxiv.org/abs/2305.04871](http://arxiv.org/abs/2305.04871)

    本文提出了一种基于高斯过程的贝叶斯非参数方法，可以解决连续时间信号的去卷积问题，适用于观测值中可能存在缺失数据且信号滤波器未知的情况。

    

    本文考虑去卷积问题，即从卷积处理的观测值 $\mathbf{y}$ 中恢复潜在信号 $x(\cdot)$，其中观测值 $\mathbf{y}$ 可能对应于 $y$ 的一部分缺失，滤波器 $h$ 可能未知且噪声 $\eta$ 可加性。当 $x$ 是连续时间信号时，我们采用高斯过程（GP）先验分布来解决这一问题，提出了一种闭合的贝叶斯非参数去卷积策略。我们首先分析了直接模型，以建立其良好定义的条件。然后，我们转向逆问题，研究了：（i）一些必要条件，使得贝叶斯去卷积计算有可能成立，以及（ii）在哪种程度上可以从数据中学习到滤波器 $h$，以及在盲去卷积情况下可以近似滤波器 $h$ 的程度。所提出的方法被称为高斯过程去卷积（GPDC），并与其他去卷积方法进行了比较。

    Let us consider the deconvolution problem, that is, to recover a latent source $x(\cdot)$ from the observations $\y = [y_1,\ldots,y_N]$ of a convolution process $y = x\star h + \eta$, where $\eta$ is an additive noise, the observations in $\y$ might have missing parts with respect to $y$, and the filter $h$ could be unknown. We propose a novel strategy to address this task when $x$ is a continuous-time signal: we adopt a Gaussian process (GP) prior on the source $x$, which allows for closed-form Bayesian nonparametric deconvolution. We first analyse the direct model to establish the conditions under which the model is well defined. Then, we turn to the inverse problem, where we study i) some necessary conditions under which Bayesian deconvolution is feasible, and ii) to which extent the filter $h$ can be learnt from data or approximated for the blind deconvolution case. The proposed approach, termed Gaussian process deconvolution (GPDC) is compared to other deconvolution methods concep
    
[^2]: 多智能体强化学习中的局部优化达到全局最优

    Local Optimization Achieves Global Optimality in Multi-Agent Reinforcement Learning. (arXiv:2305.04819v1 [cs.LG])

    [http://arxiv.org/abs/2305.04819](http://arxiv.org/abs/2305.04819)

    本文提出了一种多智能体PPO算法，利用局部优化达到全局最优，具有统计保证，这是首个能在协作马尔可夫博弈中证明收敛的算法。

    

    带函数逼近的策略优化方法在多智能体强化学习中被广泛使用，但如何设计具有统计保证的算法仍然难以捉摸。利用多智能体表现差异引理，该引理表征了多智能体策略优化的潜在空间，我们发现本地化的动作价值函数对于每个局部策略都可以作为理想的下降方向。根据这一观察结果，我们提出了一个多智能体PPO算法，其中每个智能体的本地策略更新类似于vanilla PPO。我们证明，对于马尔可夫博弈和问题相关量的标准正则性条件，我们的算法以亚线性速度收敛于全局最优策略。我们将算法扩展到离线策略设置，并引入悲观主义来评估策略，这与实验结果相符。据我们所知，这是首个能在协作马尔可夫博弈中证明收敛的多智能体PPO算法。

    Policy optimization methods with function approximation are widely used in multi-agent reinforcement learning. However, it remains elusive how to design such algorithms with statistical guarantees. Leveraging a multi-agent performance difference lemma that characterizes the landscape of multi-agent policy optimization, we find that the localized action value function serves as an ideal descent direction for each local policy. Motivated by the observation, we present a multi-agent PPO algorithm in which the local policy of each agent is updated similarly to vanilla PPO. We prove that with standard regularity conditions on the Markov game and problem-dependent quantities, our algorithm converges to the globally optimal policy at a sublinear rate. We extend our algorithm to the off-policy setting and introduce pessimism to policy evaluation, which aligns with experiments. To our knowledge, this is the first provably convergent multi-agent PPO algorithm in cooperative Markov games.
    
[^3]: 空间过程的神经似然面

    Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods. (arXiv:2305.04634v1 [stat.ME])

    [http://arxiv.org/abs/2305.04634](http://arxiv.org/abs/2305.04634)

    研究提出了使用CNN学习空间过程的似然函数。即使在没有确切似然函数的情况下，通过分类任务进行的神经网络的训练，可以隐式地学习似然函数。使用Platt缩放可以提高神经似然面的准确性。

    

    在空间统计中，当拟合空间过程到真实世界的数据时，快速准确的参数估计和可靠的不确定性量化手段可能是一项具有挑战性的任务，因为似然函数可能评估缓慢或难以处理。 在本研究中，我们提出使用卷积神经网络（CNN）学习空间过程的似然函数。通过特定设计的分类任务，我们的神经网络隐式地学习似然函数，即使在没有显式可用的确切似然函数的情况下也可以实现。一旦在分类任务上进行了训练，我们的神经网络使用Platt缩放进行校准，从而提高了神经似然面的准确性。为了展示我们的方法，我们比较了来自神经似然面的最大似然估计和近似置信区间与两个不同空间过程（高斯过程和对数高斯Cox过程）的相应精确或近似的似然函数构成的等效物。

    In spatial statistics, fast and accurate parameter estimation coupled with a reliable means of uncertainty quantification can be a challenging task when fitting a spatial process to real-world data because the likelihood function might be slow to evaluate or intractable. In this work, we propose using convolutional neural networks (CNNs) to learn the likelihood function of a spatial process. Through a specifically designed classification task, our neural network implicitly learns the likelihood function, even in situations where the exact likelihood is not explicitly available. Once trained on the classification task, our neural network is calibrated using Platt scaling which improves the accuracy of the neural likelihood surfaces. To demonstrate our approach, we compare maximum likelihood estimates and approximate confidence regions constructed from the neural likelihood surface with the equivalent for exact or approximate likelihood for two different spatial processes: a Gaussian Pro
    
[^4]: 《Signature Kernel》

    The Signature Kernel. (arXiv:2305.04625v1 [math.PR])

    [http://arxiv.org/abs/2305.04625](http://arxiv.org/abs/2305.04625)

    Signature核函数是一种正定的顺序数据核函数，具有高效的计算算法和强大的实证性能。

    

    Signature核函数是用于顺序数据的正定核函数。它继承了随机分析的理论保证，具有高效的计算算法，并且表现出强大的实证性能。在即将出版的Springer手册的这篇简短调查论文中，我们介绍了Signature核函数并强调了这些理论和计算性质。

    The signature kernel is a positive definite kernel for sequential data. It inherits theoretical guarantees from stochastic analysis, has efficient algorithms for computation, and shows strong empirical performance. In this short survey paper for a forthcoming Springer handbook, we give an elementary introduction to the signature kernel and highlight these theoretical and computational properties.
    
[^5]: 基于矩阵流形的神经网络构建：陀螺矢量空间方法

    Building Neural Networks on Matrix Manifolds: A Gyrovector Space Approach. (arXiv:2305.04560v1 [stat.ML])

    [http://arxiv.org/abs/2305.04560](http://arxiv.org/abs/2305.04560)

    本文通过将陀螺矢量空间中的一些概念推广到SPD和Grassmann流形，提出了在这些流形上构建神经网络的新模型和新层，并在人类动作识别和知识图谱完成两个应用中展示了其有效性。

    

    矩阵流形，如对称正定（SPD）矩阵和Grassmann流形，出现在许多应用中。最近，通过应用陀螺群和陀螺矢量空间的理论——这是一个研究双曲几何的强大框架——一些工作尝试在矩阵流形上构建欧几里德神经网络的原则性推广。然而，由于缺乏考虑流形的内积和陀螺角等概念的陀螺矢量空间，相比于用于研究双曲几何的那些概念，这些工作提供的技术和数学工具仍然有限。在本文中，我们将陀螺矢量空间中的一些概念推广到SPD和Grassmann流形，并提出了在这些流形上构建神经网络的新模型和新层。我们展示了我们的方法在人类动作识别和知识图谱完成两个应用中的有效性。

    Matrix manifolds, such as manifolds of Symmetric Positive Definite (SPD) matrices and Grassmann manifolds, appear in many applications. Recently, by applying the theory of gyrogroups and gyrovector spaces that is a powerful framework for studying hyperbolic geometry, some works have attempted to build principled generalizations of Euclidean neural networks on matrix manifolds. However, due to the lack of many concepts in gyrovector spaces for the considered manifolds, e.g., the inner product and gyroangles, techniques and mathematical tools provided by these works are still limited compared to those developed for studying hyperbolic geometry. In this paper, we generalize some notions in gyrovector spaces for SPD and Grassmann manifolds, and propose new models and layers for building neural networks on these manifolds. We show the effectiveness of our approach in two applications, i.e., human action recognition and knowledge graph completion.
    
[^6]: Q&A标签学习

    Q&A Label Learning. (arXiv:2305.04539v1 [cs.LG])

    [http://arxiv.org/abs/2305.04539](http://arxiv.org/abs/2305.04539)

    本文提出了一种新的标注方法，称为Q&A标注。该方法涉及到一个问题生成器和一个回答者，通过一系列问题将相应的标签分配给实例，并通过导出损失函数，对使用Q&A标签的实例的监督式机器学习的分类风险进行了评估。

    

    为了进行监督式机器学习，将标签分配给实例是至关重要的。在本文中，我们提出了一种新的注释方法，称为Q&A标注。它涉及到一个问题生成器，用于询问有关要分配的实例的标签的问题，以及一个回答问题并将对应标签分配给实例的标注者。我们得出了两种不同Q&A标注过程中分配标签的标签生成模型，并证明了在两种过程中，所得到的模型部分与先前的研究结果一致。本研究与以往研究的主要区别在于，本研究的标签生成模型并未被假定，而是基于特定注释方法Q&A标注而推导出来的。我们还导出了一个损失函数，用于评估使用分配了Q&A标签的实例的普通监督式机器学习的分类风险，并评估了

    Assigning labels to instances is crucial for supervised machine learning. In this paper, we proposed a novel annotation method called Q&A labeling, which involves a question generator that asks questions about the labels of the instances to be assigned, and an annotator who answers the questions and assigns the corresponding labels to the instances. We derived a generative model of labels assigned according to two different Q&A labeling procedures that differ in the way questions are asked and answered. We showed that, in both procedures, the derived model is partially consistent with that assumed in previous studies. The main distinction of this study from previous studies lies in the fact that the label generative model was not assumed, but rather derived based on the definition of a specific annotation method, Q&A labeling. We also derived a loss function to evaluate the classification risk of ordinary supervised machine learning using instances assigned Q&A labels and evaluated the
    
[^7]: 干预概率分布的公理化

    Axiomatization of Interventional Probability Distributions. (arXiv:2305.04479v1 [math.ST])

    [http://arxiv.org/abs/2305.04479](http://arxiv.org/abs/2305.04479)

    本论文提供了一种简单和清晰的因果理论，它不需要使用任何建模假设，包括大多数具有潜在变量和因果循环的情况，并且不假定存在底层真正的因果图-事实上，它是因果图的副产品。

    

    因果干预是因果推断中的基本工具。在结构因果模型的情况下，它被公理化为do-演算规则。我们提供了一种简单的公理化方法，用于区分不同类型的干预分布的概率分布族。我们的公理化方法整洁地导致了一种简单和清晰的因果理论，具有几个优点：它不需要使用任何建模假设，例如结构性因果模型所强加的假设；它只依赖于单个变量的干预；它包括大多数具有潜在变量和因果循环的情况；更重要的是，它不假定存在底层真正的因果图--事实上，因果图是我们理论的副产品。我们展示了，在我们的公理化方法下，干预分布对于定义的干预因果图是马尔可夫的，并且观察到的联合概率分布对于获得的因果图是马尔可夫的；这些结果是一致的。

    Causal intervention is an essential tool in causal inference. It is axiomatized under the rules of do-calculus in the case of structure causal models. We provide simple axiomatizations for families of probability distributions to be different types of interventional distributions. Our axiomatizations neatly lead to a simple and clear theory of causality that has several advantages: it does not need to make use of any modeling assumptions such as those imposed by structural causal models; it only relies on interventions on single variables; it includes most cases with latent variables and causal cycles; and more importantly, it does not assume the existence of an underlying true causal graph--in fact, a causal graph is a by-product of our theory. We show that, under our axiomatizations, the intervened distributions are Markovian to the defined intervened causal graphs, and an observed joint probability distribution is Markovian to the obtained causal graph; these results are consistent 
    
[^8]: 用于加权因果 DAG 的新度量和搜索算法

    New metrics and search algorithms for weighted causal DAGs. (arXiv:2305.04445v1 [cs.LG])

    [http://arxiv.org/abs/2305.04445](http://arxiv.org/abs/2305.04445)

    本研究提供了针对加权因果 DAGs的新度量和搜索算法，发现了用于自适应干预的因果图，提供了一个新的基准来捕捉搜索算法的最坏干预成本，并提供自适应搜索算法实现对数逼近。

    

    从数据中恢复因果关系是一个重要的问题。在使用观测数据时，只能恢复到一个马尔科夫等价类的因果图，并且需要额外的假设或干预数据来完成恢复。本文在一些标准假设下，通过节点相关干预成本的自适应干预，研究因果图发现。对于这种情况，我们证明没有算法能够比验证次数的顺序更好地实现渐近保证，验证次数是自适应搜索算法的一个成熟基准。在这个负面结果的基础上，我们定义了一个捕捉任何搜索算法最坏干预成本的新基准。此外，针对这个新基准，我们提供了自适应搜索算法，在各种设置下都能实现对数逼近：原子、有界大小的干预和广义成本。

    Recovering causal relationships from data is an important problem. Using observational data, one can typically only recover causal graphs up to a Markov equivalence class and additional assumptions or interventional data are needed for complete recovery. In this work, under some standard assumptions, we study causal graph discovery via adaptive interventions with node-dependent interventional costs. For this setting, we show that no algorithm can achieve an approximation guarantee that is asymptotically better than linear in the number of vertices with respect to the verification number; a well-established benchmark for adaptive search algorithms. Motivated by this negative result, we define a new benchmark that captures the worst-case interventional cost for any search algorithm. Furthermore, with respect to this new benchmark, we provide adaptive search algorithms that achieve logarithmic approximations under various settings: atomic, bounded size interventions and generalized cost o
    
[^9]: 用扩散模型解决反问题的变分视角

    A Variational Perspective on Solving Inverse Problems with Diffusion Models. (arXiv:2305.04391v1 [cs.LG])

    [http://arxiv.org/abs/2305.04391](http://arxiv.org/abs/2305.04391)

    该论文提出了一种通过去噪扩散过程自然地导致正则化的变分方法（RED-Diff），可用于解决不同反问题。加权机制可以衡量不同时间步长的去噪器的贡献。

    

    扩散模型已成为视觉领域基础模型的关键支柱之一。其中一个关键应用是通过单个扩散先验普遍解决不同的反问题，而无需为每个任务重新训练。然而，由于扩散过程的非线性和迭代性质使得后验难以处理，因此我们提出了一种变分方法，旨在近似真实后验分布。我们展示了我们的方法通过去噪扩散过程（RED-Diff）自然地导致正则化，其中来自不同时间步长的去噪器同时对图像施加不同的结构约束。为了衡量不同时间步长的去噪器的贡献，我们提出了一种基于信号-t的加权机制。

    Diffusion models have emerged as a key pillar of foundation models in visual domains. One of their critical applications is to universally solve different downstream inverse tasks via a single diffusion prior without re-training for each task. Most inverse tasks can be formulated as inferring a posterior distribution over data (e.g., a full image) given a measurement (e.g., a masked image). This is however challenging in diffusion models since the nonlinear and iterative nature of the diffusion process renders the posterior intractable. To cope with this challenge, we propose a variational approach that by design seeks to approximate the true posterior distribution. We show that our approach naturally leads to regularization by denoising diffusion process (RED-Diff) where denoisers at different timesteps concurrently impose different structural constraints over the image. To gauge the contribution of denoisers from different timesteps, we propose a weighting mechanism based on signal-t
    
[^10]: 预测性聚类与优化的通用框架

    A Generalized Framework for Predictive Clustering and Optimization. (arXiv:2305.04364v1 [cs.LG])

    [http://arxiv.org/abs/2305.04364](http://arxiv.org/abs/2305.04364)

    本文提出了一种通用的预测性聚类优化框架，该框架允许不同的聚类定义和回归、分类目标，并提供了高度可扩展的算法以解决大型数据集的优化问题。

    

    聚类是一种强大而广泛使用的数据科学工具。虽然聚类通常被认为是无监督学习技术，但也存在着像Spath的聚类回归这样试图找到产生监督目标低回归误差的数据聚类的监督变体。在本文中，我们为预测性聚类定义了一个通用的优化框架，该框架可以接受不同的聚类定义（任意点分配、最接近的中心和边界框）和回归、分类目标。然后，我们提出了一种联合优化策略，在这个通用框架中利用混合整数线性规划（MILP）进行全局优化。为了减轻大型数据集的可扩展性问题，我们还提供了受主导最小化（MM）启发的高度可扩展的贪婪算法。

    Clustering is a powerful and extensively used data science tool. While clustering is generally thought of as an unsupervised learning technique, there are also supervised variations such as Spath's clusterwise regression that attempt to find clusters of data that yield low regression error on a supervised target. We believe that clusterwise regression is just a single vertex of a largely unexplored design space of supervised clustering models. In this article, we define a generalized optimization framework for predictive clustering that admits different cluster definitions (arbitrary point assignment, closest center, and bounding box) and both regression and classification objectives. We then present a joint optimization strategy that exploits mixed-integer linear programming (MILP) for global optimization in this generalized framework. To alleviate scalability concerns for large datasets, we also provide highly scalable greedy algorithms inspired by the Majorization-Minimization (MM) 
    
[^11]: 利用神经网络快速估计广义极值分布的参数

    Fast parameter estimation of Generalized Extreme Value distribution using Neural Networks. (arXiv:2305.04341v1 [stat.ML])

    [http://arxiv.org/abs/2305.04341](http://arxiv.org/abs/2305.04341)

    本论文提出了一种利用神经网络进行计算有效的广义极值分布参数估计的方法，与传统的极大似然方法相比，具有相似的准确性和显著的计算加速度。

    

    广义极值分布的重尾性使其成为建模洪水、干旱、热浪、野火等极端事件的流行选择。然而，使用传统的极大似然方法估计分布的参数，即使对于中等大小的数据集也可能具有计算密集度。为了克服这个限制，我们提出了一种使用神经网络的计算有效的不需要似然的估计方法。通过广泛的仿真研究，我们证明了所提出的基于神经网络的方法提供具有可比较准确性的广义极值分布参数估计，但具有显着的计算加速度。为了考虑估计不确定度，我们利用训练好的网络中固有的参数自助法。最后，我们将这种方法应用于来自社区气候系统模型的1000年年最大温度数据。

    The heavy-tailed behavior of the generalized extreme-value distribution makes it a popular choice for modeling extreme events such as floods, droughts, heatwaves, wildfires, etc. However, estimating the distribution's parameters using conventional maximum likelihood methods can be computationally intensive, even for moderate-sized datasets. To overcome this limitation, we propose a computationally efficient, likelihood-free estimation method utilizing a neural network. Through an extensive simulation study, we demonstrate that the proposed neural network-based method provides Generalized Extreme Value (GEV) distribution parameter estimates with comparable accuracy to the conventional maximum likelihood method but with a significant computational speedup. To account for estimation uncertainty, we utilize parametric bootstrapping, which is inherent in the trained network. Finally, we apply this method to 1000-year annual maximum temperature data from the Community Climate System Model ve
    
[^12]: 基于协变量转移的分类树剪枝

    Classification Tree Pruning Under Covariate Shift. (arXiv:2305.04335v1 [stat.ML])

    [http://arxiv.org/abs/2305.04335](http://arxiv.org/abs/2305.04335)

    本文提出了一种基于协变量转移的分类树剪枝方法，可以访问大部分来自分布 $P_{X，Y}$ 的数据，但是只能获得来自拥有不同 $X$-边缘的目标分布 $Q_{X，Y}$ 的少量数据。使用的优化标准是一个关于分布 $P_{X} \to Q_{X}$ 的 \emph{平均差异}，该标准可以显著放宽最近提出的 \emph{转移指数}，最终可以得到最优的剪枝结果。

    

    本文考虑在训练数据不均匀的情况下，选择适当的子树以平衡偏差和方差的分类树剪枝问题。我们提出了一种针对这种情况的最优剪枝的高效程序，该程序可以访问大部分来自分布 $P_{X，Y}$ 的数据，但是只能获得来自拥有不同 $X$-边缘的目标分布 $Q_{X，Y}$ 的少量数据。在基本交叉验证和其他进行惩罚的变体，如基于信息度量的剪枝方法非常不理想的情况下，我们提供了一种最优剪枝的方法。使用的优化标准是一个关于分布 $P_{X} \to Q_{X}$ 的 \emph{平均差异}（在 $X$ 空间上平均），该标准可以显著放宽最近提出的 \emph{转移指数} 这一统计学概念，该概念被证明能够紧密地捕捉这种分布转移情况下分类的极限限制。我们放宽的标准可以被看作是分布之间的\emph{相对维度}度量，因为它涉及到信息的现有度量概念，例如闵可夫斯基和Rényi维度。

    We consider the problem of \emph{pruning} a classification tree, that is, selecting a suitable subtree that balances bias and variance, in common situations with inhomogeneous training data. Namely, assuming access to mostly data from a distribution $P_{X, Y}$, but little data from a desired distribution $Q_{X, Y}$ with different $X$-marginals, we present the first efficient procedure for optimal pruning in such situations, when cross-validation and other penalized variants are grossly inadequate. Optimality is derived with respect to a notion of \emph{average discrepancy} $P_{X} \to Q_{X}$ (averaged over $X$ space) which significantly relaxes a recent notion -- termed \emph{transfer-exponent} -- shown to tightly capture the limits of classification under such a distribution shift. Our relaxed notion can be viewed as a measure of \emph{relative dimension} between distributions, as it relates to existing notions of information such as the Minkowski and Renyi dimensions.
    
[^13]: 使用截断数据学习高斯混合模型

    Learning Mixtures of Gaussians with Censored Data. (arXiv:2305.04127v1 [cs.LG])

    [http://arxiv.org/abs/2305.04127](http://arxiv.org/abs/2305.04127)

    本文提出了一种学习高斯混合模型的算法，该算法仅需要很少的样本且能够对权重和均值进行准确估计。

    

    本文研究了在具有截断数据的情况下，学习高斯混合模型的问题。即从一个混合单变量高斯分布$\sum_{i=1}^k w_i \mathcal{N}(\mu_i,\sigma^2)$中观测到的样本只有当其位于$S$集合内时才会被观察到。我们提出了一种算法，仅需要$\frac{1}{\varepsilon^{O(k)}}$个样本即可在$\varepsilon$误差内估计权重$w_i$和均值$\mu_i$。

    We study the problem of learning mixtures of Gaussians with censored data. Statistical learning with censored data is a classical problem, with numerous practical applications, however, finite-sample guarantees for even simple latent variable models such as Gaussian mixtures are missing. Formally, we are given censored data from a mixture of univariate Gaussians $$\sum_{i=1}^k w_i \mathcal{N}(\mu_i,\sigma^2),$$ i.e. the sample is observed only if it lies inside a set $S$. The goal is to learn the weights $w_i$ and the means $\mu_i$. We propose an algorithm that takes only $\frac{1}{\varepsilon^{O(k)}}$ samples to estimate the weights $w_i$ and the means $\mu_i$ within $\varepsilon$ error.
    
[^14]: 结构无关函数估计的基本限制

    The Fundamental Limits of Structure-Agnostic Functional Estimation. (arXiv:2305.04116v1 [math.ST])

    [http://arxiv.org/abs/2305.04116](http://arxiv.org/abs/2305.04116)

    一阶去偏方法在最小二乘意义下在干扰函数生存在特定函数空间时被证明是次优的，这促进了“高阶”去偏方法的发展。

    

    近年来，许多因果推断和函数估计问题的发展都源于这样一个事实：在非常弱的条件下，经典的一步（一阶）去偏方法或它们较新的样本分割双机器学习方法可以比插补估计更好地工作。这些一阶校正以黑盒子方式改善插补估计值，因此经常与强大的现成估计方法一起使用。然而，当干扰函数生存在Holder型函数空间中时，这些一阶方法在最小二乘意义下被证明是次优的。这种一阶去偏的次优性促进了“高阶”去偏方法的发展。由此产生的估计量在某些情况下被证明是在Holder类型空间上最小化的，并且它们的分析与基础函数空间的性质密切相关。

    Many recent developments in causal inference, and functional estimation problems more generally, have been motivated by the fact that classical one-step (first-order) debiasing methods, or their more recent sample-split double machine-learning avatars, can outperform plugin estimators under surprisingly weak conditions. These first-order corrections improve on plugin estimators in a black-box fashion, and consequently are often used in conjunction with powerful off-the-shelf estimation methods. These first-order methods are however provably suboptimal in a minimax sense for functional estimation when the nuisance functions live in Holder-type function spaces. This suboptimality of first-order debiasing has motivated the development of "higher-order" debiasing methods. The resulting estimators are, in some cases, provably optimal over Holder-type spaces, but both the estimators which are minimax-optimal and their analyses are crucially tied to properties of the underlying function space
    
[^15]: 选择前m个上下文相关设计的高效学习

    Efficient Learning for Selecting Top-m Context-Dependent Designs. (arXiv:2305.04086v1 [stat.ML])

    [http://arxiv.org/abs/2305.04086](http://arxiv.org/abs/2305.04086)

    该论文采用贝叶斯框架下的随机动态规划方法，开发一种顺序抽样策略，提高了选择上下文相关设计的前m个的效率。

    

    我们考虑一个针对上下文相关决策的模拟优化问题，旨在确定所有上下文情境下的前m个设计。在贝叶斯框架下，我们将最优动态抽样决策制定为随机动态规划问题，并开发一种顺序抽样策略，以高效地学习每个上下文情境下每个设计的性能。导出了渐进最优抽样比例以实现选择误报概率的最坏情况的最优大偏差率。证明了所提出的抽样策略是一致的，并且其渐近抽样比率是渐近最优的。数字实验表明，所提出的方法改善了选择上下文相关设计的前m个的效率。

    We consider a simulation optimization problem for a context-dependent decision-making, which aims to determine the top-m designs for all contexts. Under a Bayesian framework, we formulate the optimal dynamic sampling decision as a stochastic dynamic programming problem, and develop a sequential sampling policy to efficiently learn the performance of each design under each context. The asymptotically optimal sampling ratios are derived to attain the optimal large deviations rate of the worst-case of probability of false selection. The proposed sampling policy is proved to be consistent and its asymptotic sampling ratios are asymptotically optimal. Numerical experiments demonstrate that the proposed method improves the efficiency for selection of top-m context-dependent designs.
    
[^16]: Adam家族算法在无平滑优化中的收敛性保证研究

    Adam-family Methods for Nonsmooth Optimization with Convergence Guarantees. (arXiv:2305.03938v1 [math.OC])

    [http://arxiv.org/abs/2305.03938](http://arxiv.org/abs/2305.03938)

    本文提出了一种新的双时间尺度框架，证明了其在温和条件下收敛性，该框架包括了各种流行的Adam家族算法，用于训练无平滑神经网络和应对重尾噪声的需求，并通过实验表明了其效率和鲁棒性。

    

    本文对Adam家族算法在无平滑优化中的收敛性进行了全面研究，特别是在无平滑神经网络的训练中。我们提出了一种新的双时间尺度框架，采用双时间尺度更新方案，证明了其在温和条件下的收敛性。我们的框架包括了各种流行的Adam家族算法，在训练无平滑神经网络中提供了收敛性保证。此外，我们还开发了随机次梯度方法，结合梯度裁剪技术，用于训练具有重尾噪声的无平滑神经网络。通过我们的框架，我们展示了我们提出的方法甚至在仅假定评估噪声可积的情况下也会收敛。广泛的数值实验证明了我们提出的方法的高效性和稳健性。

    In this paper, we present a comprehensive study on the convergence properties of Adam-family methods for nonsmooth optimization, especially in the training of nonsmooth neural networks. We introduce a novel two-timescale framework that adopts a two-timescale updating scheme, and prove its convergence properties under mild assumptions. Our proposed framework encompasses various popular Adam-family methods, providing convergence guarantees for these methods in training nonsmooth neural networks. Furthermore, we develop stochastic subgradient methods that incorporate gradient clipping techniques for training nonsmooth neural networks with heavy-tailed noise. Through our framework, we show that our proposed methods converge even when the evaluation noises are only assumed to be integrable. Extensive numerical experiments demonstrate the high efficiency and robustness of our proposed methods.
    
[^17]: 基于轨迹的随机流行病学模型优化

    Trajectory-oriented optimization of stochastic epidemiological models. (arXiv:2305.03926v1 [stat.AP])

    [http://arxiv.org/abs/2305.03926](http://arxiv.org/abs/2305.03926)

    本论文提出了一个基于轨迹的优化方法来处理随机流行病学模型，可以找到与实际观测值接近的实际轨迹，而不是仅使平均模拟结果与实测数据相符。

    

    针对随机模型，为了进行预测和运行模拟，需要进行地面实测标定。由于输出结果通常是通过集成或分布来描述，因此需要对每个成员进行标定。本文提出了一种基于高斯过程代理和Thompson采样的优化策略来寻找与事实相一致的输入参数设置和随机数种子，该Trajectory Oriented Optimization（TOO）方法可以产生与实际观测值接近的实际轨迹，而不是仅虽然模拟的平均行为与事实相符。

    Epidemiological models must be calibrated to ground truth for downstream tasks such as producing forward projections or running what-if scenarios. The meaning of calibration changes in case of a stochastic model since output from such a model is generally described via an ensemble or a distribution. Each member of the ensemble is usually mapped to a random number seed (explicitly or implicitly). With the goal of finding not only the input parameter settings but also the random seeds that are consistent with the ground truth, we propose a class of Gaussian process (GP) surrogates along with an optimization strategy based on Thompson sampling. This Trajectory Oriented Optimization (TOO) approach produces actual trajectories close to the empirical observations instead of a set of parameter settings where only the mean simulation behavior matches with the ground truth.
    
[^18]: 双支持向量分位数回归

    Twin support vector quantile regression. (arXiv:2305.03894v1 [stat.ML])

    [http://arxiv.org/abs/2305.03894](http://arxiv.org/abs/2305.03894)

    TSVQR能够捕捉现代数据中的异质和不对称信息，并有效地描述了所有数据点的异质分布信息。通过构造两个较小的二次规划问题，TSVQR生成两个非平行平面，测量每个分位数水平下限和上限之间的分布不对称性。在多个实验中，TSVQR优于以前的分位数回归方法。

    

    我们提出了一种双支持向量分位数回归(TSVQR)，用于捕捉现代数据中异质和不对称信息。TSVQR利用分位数参数有效地描述了所有数据点的异质分布信息。相应地，TSVQR构造了两个较小的二次规划问题(QPPs)，生成两个非平行平面，以测量每个分位数水平下限和上限之间的分布不对称性。TSVQR中的QPP比以前的分位数回归方法更小且更易于解决。此外，TSVQR的双重坐标下降算法也加速了训练速度。在六个人造数据集、五个基准数据集、两个大规模数据集、两个时间序列数据集和两个不平衡数据集上的实验结果表明，TSVQR在完全捕获异质性方面的效果优于以前的分位数回归方法。

    We propose a twin support vector quantile regression (TSVQR) to capture the heterogeneous and asymmetric information in modern data. Using a quantile parameter, TSVQR effectively depicts the heterogeneous distribution information with respect to all portions of data points. Correspondingly, TSVQR constructs two smaller sized quadratic programming problems (QPPs) to generate two nonparallel planes to measure the distributional asymmetry between the lower and upper bounds at each quantile level. The QPPs in TSVQR are smaller and easier to solve than those in previous quantile regression methods. Moreover, the dual coordinate descent algorithm for TSVQR also accelerates the training speed. Experimental results on six artiffcial data sets, ffve benchmark data sets, two large scale data sets, two time-series data sets, and two imbalanced data sets indicate that the TSVQR outperforms previous quantile regression methods in terms of the effectiveness of completely capturing the heterogeneous 
    
[^19]: 高维低秩张量赌博机研究

    On High-dimensional and Low-rank Tensor Bandits. (arXiv:2305.03884v1 [stat.ML])

    [http://arxiv.org/abs/2305.03884](http://arxiv.org/abs/2305.03884)

    本研究提出了一个通用的张量赌博机模型，其中行动和系统参数由张量表示，着重于未知系统张量为低秩的情况。所开发的 TOFU 算法首先利用灵活的张量回归技术估计与系统张量相关联的低维子空间，然后将原始问题转换成一个具有系统参数范数约束的新问题，并采用范数约束赌博子例程解决。

    

    大多数线性赌博机的研究都侧重于整个系统的一维特征。虽然代表性很强，但这种方法可能不能模拟高维但有优势结构的应用，例如用于推荐系统的低秩张量表示。为了解决这个限制，本研究研究了一个通用的张量赌博机模型，其中行动和系统参数由张量表示，而我们特别关注未知系统张量为低秩的情况。发展了一种新的赌博机算法TOFU（不确定性中的张量乐观），该算法首先利用灵活的张量回归技术估计与系统张量相关联的低维子空间。然后利用这些估计将原始问题转换为一个具有其系统参数范数约束的新问题。最后，TOFU采用范数约束赌博子例程，利用这些约束来实现问题的解决。

    Most existing studies on linear bandits focus on the one-dimensional characterization of the overall system. While being representative, this formulation may fail to model applications with high-dimensional but favorable structures, such as the low-rank tensor representation for recommender systems. To address this limitation, this work studies a general tensor bandits model, where actions and system parameters are represented by tensors as opposed to vectors, and we particularly focus on the case that the unknown system tensor is low-rank. A novel bandit algorithm, coined TOFU (Tensor Optimism in the Face of Uncertainty), is developed. TOFU first leverages flexible tensor regression techniques to estimate low-dimensional subspaces associated with the system tensor. These estimates are then utilized to convert the original problem to a new one with norm constraints on its system parameters. Lastly, a norm-constrained bandit subroutine is adopted by TOFU, which utilizes these constraint
    
[^20]: 通过流映射算子学习随机动力学系统

    Learning Stochastic Dynamical System via Flow Map Operator. (arXiv:2305.03874v1 [cs.LG])

    [http://arxiv.org/abs/2305.03874](http://arxiv.org/abs/2305.03874)

    该论文提出了一种通过测量数据学习未知随机动力学系统的数值框架随机流映射学习（sFML），在不同类型的随机系统上进行的全面实验证明了 sFML 的有效性。

    

    我们提出了一种通过测量数据学习未知随机动力学系统的数值框架。称为随机流映射学习（sFML），这个新框架是流映射学习（FML）的扩展，后者是为了学习确定性动力学系统而开发的。对于学习随机系统，我们定义了一个随机流映射，它是两个子流映射的叠加：一个确定性子映射和一个随机子映射。随机训练数据首先用于构建确定性子映射，然后是随机子映射。确定性子映射采用残差网络（ResNet）形式，类似于FML对于确定性系统的工作。对于随机子映射，我们采用生成模型，尤其是生成对抗网络（GANs）在本文中应用。最终构建的随机流映射定义了一个随机演化模型，它在分布方面是未知随机系统的弱近似。在不同类型的随机系统上进行的全面实验证明了sFML揭示未知随机系统各种类型的非线性、噪声协方差结构和时间相关特性的有效性。

    We present a numerical framework for learning unknown stochastic dynamical systems using measurement data. Termed stochastic flow map learning (sFML), the new framework is an extension of flow map learning (FML) that was developed for learning deterministic dynamical systems. For learning stochastic systems, we define a stochastic flow map that is a superposition of two sub-flow maps: a deterministic sub-map and a stochastic sub-map. The stochastic training data are used to construct the deterministic sub-map first, followed by the stochastic sub-map. The deterministic sub-map takes the form of residual network (ResNet), similar to the work of FML for deterministic systems. For the stochastic sub-map, we employ a generative model, particularly generative adversarial networks (GANs) in this paper. The final constructed stochastic flow map then defines a stochastic evolution model that is a weak approximation, in term of distribution, of the unknown stochastic system. A comprehensive set
    
[^21]: 无遗憾的约束贝叶斯优化方法用于带有噪声和昂贵混合模型的差分分位数函数逼近

    No-Regret Constrained Bayesian Optimization of Noisy and Expensive Hybrid Models using Differentiable Quantile Function Approximations. (arXiv:2305.03824v1 [stat.ML])

    [http://arxiv.org/abs/2305.03824](http://arxiv.org/abs/2305.03824)

    本文提出了一种新颖的算法，CUQB，来解决复合函数（混合模型）的高效约束全局优化问题，并取得了良好的效果，在合成和真实的应用程序中均得到了验证，包括进行了最优控制的流体流量和拓扑结构优化，后者比当前最先进的设计强2倍。

    

    本文研究了复合函数（混合模型）的高效约束全局优化问题，该模型的输入是具有矢量值输出和有噪声观测的昂贵黑盒函数，这在实际的科学、工程、制造和控制应用中经常出现。我们提出了一种新颖的算法Constrained Upper Quantile Bound（CUQB），用于解决这种问题，直接利用了我们展示的目标和约束函数的复合结构，从而大大提高了采样效率。CUQB的概念简单，避免了先前方法所使用的约束逼近。虽然CUQB的收购函数不在封闭形式中，但我们提出了一种新颖的可微随机逼近，使其能够有效地最大化。我们进一步得出了对于累积遗憾和约束违规的界限。由于在某些规则假设下这些界限对迭代次数具有次线性依赖性，因此我们的算法在渐近意义下无遗憾并满足约束条件。我们在几个合成和真实的应用程序中展示了CUQB的功效，包括桥架拓扑 - 在其中，我们发现的结构比当前最先进的设计强2倍 - 以及流体流量的最优控制，其中我们使用的方法比以前的方法少了3倍的模拟。

    This paper investigates the problem of efficient constrained global optimization of composite functions (hybrid models) whose input is an expensive black-box function with vector-valued outputs and noisy observations, which often arises in real-world science, engineering, manufacturing, and control applications. We propose a novel algorithm, Constrained Upper Quantile Bound (CUQB), to solve such problems that directly exploits the composite structure of the objective and constraint functions that we show leads substantially improved sampling efficiency. CUQB is conceptually simple and avoids the constraint approximations used by previous methods. Although the CUQB acquisition function is not available in closed form, we propose a novel differentiable stochastic approximation that enables it to be efficiently maximized. We further derive bounds on the cumulative regret and constraint violation. Since these bounds depend sublinearly on the number of iterations under some regularity assum
    
[^22]: 二元事件的校准评估和大胆再校准

    Calibration Assessment and Boldness-Recalibration for Binary Events. (arXiv:2305.03780v1 [stat.ME])

    [http://arxiv.org/abs/2305.03780](http://arxiv.org/abs/2305.03780)

    本研究提出了一种假设检验和贝叶斯模型选择方法来评估校准，并提供一种大胆再校准策略，使实践者能够在满足所需的校准水平的情况下负责任地增强预测。

    

    概率预测对于医学、经济、图像分类、体育分析、娱乐等许多领域中的决策制定至关重要。理想情况下，概率预测应该 (i) 校准良好 (ii) 准确 (iii) 大胆，即远离事件的基础频率。满足这三个条件的预测对于决策制定是有信息量的。然而，校准和大胆之间存在基本的紧张关系，因为当预测过于谨慎时(即非大胆)校准度量可以很高。本文的目的是开发一种假设检验和贝叶斯模型选择方法来评估校准，并提供一种大胆再校准策略，使实践者能够在满足所需的校准水平的情况下负责任地增强预测。具体而言，我们允许用户预先指定他们所需的后验校准概率，然后在此约束下最大化增强预测。我们通过模拟研究和实际数据应用验证了我们方法的性能。

    Probability predictions are essential to inform decision making in medicine, economics, image classification, sports analytics, entertainment, and many other fields. Ideally, probability predictions are (i) well calibrated, (ii) accurate, and (iii) bold, i.e., far from the base rate of the event. Predictions that satisfy these three criteria are informative for decision making. However, there is a fundamental tension between calibration and boldness, since calibration metrics can be high when predictions are overly cautious, i.e., non-bold. The purpose of this work is to develop a hypothesis test and Bayesian model selection approach to assess calibration, and a strategy for boldness-recalibration that enables practitioners to responsibly embolden predictions subject to their required level of calibration. Specifically, we allow the user to pre-specify their desired posterior probability of calibration, then maximally embolden predictions subject to this constraint. We verify the perfo
    
[^23]: Majorizing Measures, Codes, and Information（测度主导、码和信息）

    Majorizing Measures, Codes, and Information. (arXiv:2305.02960v1 [cs.IT])

    [http://arxiv.org/abs/2305.02960](http://arxiv.org/abs/2305.02960)

    本文介绍了一种基于信息论的趋势测度定理视角，该视角将随机过程的有限性与索引度量空间元素的有效可变长度编码的存在性相关联。

    

    Fernique和Talagrand的趋势测度定理是随机过程理论中的一个基本结果。它将度量空间中元素索引的随机过程的有限性与来自某些多尺度组合结构（如填充和覆盖树）的复杂性度量相关联。本文在Andreas Maurer的一份鲜为人知的预印本中首次概述的思路上构建一种基于信息论的趋势测度定理视角，根据该视角，随机过程的有限性是用索引度量空间元素的有效可变长度编码的存在性来表述的。

    The majorizing measure theorem of Fernique and Talagrand is a fundamental result in the theory of random processes. It relates the boundedness of random processes indexed by elements of a metric space to complexity measures arising from certain multiscale combinatorial structures, such as packing and covering trees. This paper builds on the ideas first outlined in a little-noticed preprint of Andreas Maurer to present an information-theoretic perspective on the majorizing measure theorem, according to which the boundedness of random processes is phrased in terms of the existence of efficient variable-length codes for the elements of the indexing metric space.
    
[^24]: 使用机器学习的形成能量预测方法进行猎枪晶体结构预测

    Shotgun crystal structure prediction using machine-learned formation energies. (arXiv:2305.02158v1 [physics.comp-ph])

    [http://arxiv.org/abs/2305.02158](http://arxiv.org/abs/2305.02158)

    本研究使用机器学习方法在多个结构预测标准测试中精确识别含有100个以上原子的许多材料的全局最小结构，并以单次能量评估为基础，取代了重复的第一原理能量计算过程。

    

    可以通过找到原子构型能量曲面的全局或局部极小值来预测组装原子的稳定或亚稳定晶体结构。通常，这需要重复的第一原理能量计算，这在包含30个以上原子的大型系统中是不实际的。本研究使用简单但功能强大的机器学习工作流，使用机器学习辅助第一原理能量计算，对大量虚拟创建的晶体结构进行非迭代式单次筛选，从而在解决晶体结构预测问题方面取得了重大进展。

    Stable or metastable crystal structures of assembled atoms can be predicted by finding the global or local minima of the energy surface with respect to the atomic configurations. Generally, this requires repeated first-principles energy calculations that are impractical for large systems, such as those containing more than 30 atoms in the unit cell. Here, we have made significant progress in solving the crystal structure prediction problem with a simple but powerful machine-learning workflow; using a machine-learning surrogate for first-principles energy calculations, we performed non-iterative, single-shot screening using a large library of virtually created crystal structures. The present method relies on two key technical components: transfer learning, which enables a highly accurate energy prediction of pre-relaxed crystalline states given only a small set of training samples from first-principles calculations, and generative models to create promising and diverse crystal structure
    
[^25]: 可解释人工智能方法评述：SHAP 和 LIME

    Commentary on explainable artificial intelligence methods: SHAP and LIME. (arXiv:2305.02012v1 [stat.ML])

    [http://arxiv.org/abs/2305.02012](http://arxiv.org/abs/2305.02012)

    这篇评论对可解释人工智能方法 SHAP 和 LIME 进行了评述和比较，提出了一个框架且突出了它们的优缺点。

    

    可解释人工智能（XAI）方法已经发展出来，将机器学习模型的黑匣子转化为更易理解的形式。这些方法有助于传达模型的工作原理，旨在使机器学习模型更透明，并增加最终用户对其输出的信任。 SHapley Additive exPlanations（SHAP）和Local Interpretable Model Agnostic Explanation（LIME）是两种在表格数据中广泛使用的XAI方法。在这篇评论中，我们讨论了两种方法的可解释性度量是如何生成的，并提出了一个解释它们输出的框架，突出了它们的优缺点。

    eXplainable artificial intelligence (XAI) methods have emerged to convert the black box of machine learning models into a more digestible form. These methods help to communicate how the model works with the aim of making machine learning models more transparent and increasing the trust of end-users into their output. SHapley Additive exPlanations (SHAP) and Local Interpretable Model Agnostic Explanation (LIME) are two widely used XAI methods particularly with tabular data. In this commentary piece, we discuss the way the explainability metrics of these two methods are generated and propose a framework for interpretation of their outputs, highlighting their weaknesses and strengths.
    
[^26]: 通过重新参数化学习实现在线反馈的实现式预测

    Performative Prediction with Bandit Feedback: Learning through Reparameterization. (arXiv:2305.01094v1 [cs.LG])

    [http://arxiv.org/abs/2305.01094](http://arxiv.org/abs/2305.01094)

    本文提出一种新的在线反馈的实现式预测框架，解决了在模型部署自身改变数据分布的情况下优化准确性的问题。

    

    本文提出了在数据分布由模型部署自身改变的情形下预测的一个框架——实现式预测。现有研究的重点在于优化准确性，但是其假设往往难以在实践中得到满足。本文针对这类问题，提出了一种两层零阶优化算法，通过重新参数化实现式预测目标，从而将非凸的目标转化为凸的目标。

    Performative prediction, as introduced by Perdomo et al. (2020), is a framework for studying social prediction in which the data distribution itself changes in response to the deployment of a model. Existing work on optimizing accuracy in this setting hinges on two assumptions that are easily violated in practice: that the performative risk is convex over the deployed model, and that the mapping from the model to the data distribution is known to the model designer in advance. In this paper, we initiate the study of tractable performative prediction problems that do not require these assumptions. To tackle this more challenging setting, we develop a two-level zeroth-order optimization algorithm, where one level aims to compute the distribution map, and the other level reparameterizes the performative prediction objective as a function of the induced data distribution. Under mild conditions, this reparameterization allows us to transform the non-convex objective into a convex one and ac
    
[^27]: 基于核棍棒过程的高斯过程专家混合模型

    Mixtures of Gaussian process experts based on kernel stick-breaking processes. (arXiv:2304.13833v1 [stat.ML])

    [http://arxiv.org/abs/2304.13833](http://arxiv.org/abs/2304.13833)

    提出了一种新的基于核棍棒过程的高斯过程专家混合模型，能够维持直观吸引力并提高模型性能，具有实用性。

    

    高斯过程专家混合模型是一类能同时解决标准高斯过程中存在的两个关键限制：可扩展性和预测性能的模型。使用狄利克雷过程作为门函数的模型能够直观地解释和自动选择混合物中专家的数量。虽然现有模型在感知非平稳性、多模性和异方差性方面表现良好，但其门函数的简单性可能会限制在应用于复杂数据生成过程时的预测性能。我们利用最近在相关狄利克雷过程文献中的进展，提出了一种基于核棍棒过程的新型高斯过程专家混合模型。我们的模型保持直观吸引力，同时提高现有模型的性能。为了使其实用性，我们设计了一个后验计算的切片抽样采样器。

    Mixtures of Gaussian process experts is a class of models that can simultaneously address two of the key limitations inherent in standard Gaussian processes: scalability and predictive performance. In particular, models that use Dirichlet processes as gating functions permit straightforward interpretation and automatic selection of the number of experts in a mixture. While the existing models are intuitive and capable of capturing non-stationarity, multi-modality and heteroskedasticity, the simplicity of their gating functions may limit the predictive performance when applied to complex data-generating processes. Capitalising on the recent advancement in the dependent Dirichlet processes literature, we propose a new mixture model of Gaussian process experts based on kernel stick-breaking processes. Our model maintains the intuitive appeal yet improve the performance of the existing models. To make it practical, we design a sampler for posterior computation based on the slice sampling. 
    
[^28]: 用均场博弈为生成模型搭建实验室

    A mean-field games laboratory for generative modeling. (arXiv:2304.13534v1 [stat.ML])

    [http://arxiv.org/abs/2304.13534](http://arxiv.org/abs/2304.13534)

    本文提出了使用均场博弈作为实验室对生成模型进行设计和分析的方法，并建立了这种方法与主要流动和扩散型生成模型之间的关联。通过研究每个生成模型与它们相关的 MFG 的最优条件，本文提出了一个基于双人 MFG 的新的生成模型，该模型在提高样本多样性和逼真度的同时改善了解缠结和公平性。

    

    本文展示了均场博弈 (MFGs) 作为一种数学框架用于解释、增强和设计生成模型的多功能性。我们建立了 MFGs 与主要流动和扩散型生成模型之间关联，并通过不同的粒子动力学和代价函数推导了这三个类别的生成模型。此外，我们通过研究它们相关的 MFG 的最优条件——一组耦合的非线性偏微分方程，来研究每个生成模型的数学结构和特性。本文还提出了一个新的基于双人 MFG 的生成模型，其中一个代理合成样本，另一个代理对样本进行识别，理论和实验结果表明，该模型生成的样本多样且逼真，同时与基准模型相比，改善了解缠结和公平性。总之，本文突显了 MFGs 作为设计和分析生成模型的实验室的潜力。

    In this paper, we demonstrate the versatility of mean-field games (MFGs) as a mathematical framework for explaining, enhancing, and designing generative models. There is a pervasive sense in the generative modeling community that the various flow and diffusion-based generative models have some foundational common structure and interrelationships. We establish connections between MFGs and major classes of flow and diffusion-based generative models including continuous-time normalizing flows, score-based models, and Wasserstein gradient flows. We derive these three classes of generative models through different choices of particle dynamics and cost functions. Furthermore, we study the mathematical structure and properties of each generative model by studying their associated MFG's optimality condition, which is a set of coupled nonlinear partial differential equations (PDEs). The theory of MFGs, therefore, enables the study of generative models through the theory of nonlinear PDEs. Throu
    
[^29]: 未观测到代理奖励的重复负责人代理博弈问题研究

    Repeated Principal-Agent Games with Unobserved Agent Rewards and Perfect-Knowledge Agents. (arXiv:2304.07407v1 [cs.LG])

    [http://arxiv.org/abs/2304.07407](http://arxiv.org/abs/2304.07407)

    本文提出了一个利用多臂老虎机框架结构来处理未知代理奖励的策略，并证明了其性能是渐进最优的。

    

    本文研究了一个多臂老虎机框架中的重复负责人代理博弈场景，其中代理选择一种老虎机后会获得奖励和激励，但负责人只能观察到代理选择了哪个老虎机以及代理相应的激励，而想要设计一种合适的策略却充满了挑战性。本文提出了一种利用多臂老虎机框架结构来处理未知代理奖励的策略，并证明了其性能是渐进最优的。

    Motivated by a number of real-world applications from domains like healthcare and sustainable transportation, in this paper we study a scenario of repeated principal-agent games within a multi-armed bandit (MAB) framework, where: the principal gives a different incentive for each bandit arm, the agent picks a bandit arm to maximize its own expected reward plus incentive, and the principal observes which arm is chosen and receives a reward (different than that of the agent) for the chosen arm. Designing policies for the principal is challenging because the principal cannot directly observe the reward that the agent receives for their chosen actions, and so the principal cannot directly learn the expected reward using existing estimation techniques. As a result, the problem of designing policies for this scenario, as well as similar ones, remains mostly unexplored. In this paper, we construct a policy that achieves a low regret (i.e., square-root regret up to a log factor) in this scenar
    
[^30]: 使用多数据因果推断选择机器学习应用的强健特征

    Selecting Robust Features for Machine Learning Applications using Multidata Causal Discovery. (arXiv:2304.05294v1 [stat.ML])

    [http://arxiv.org/abs/2304.05294](http://arxiv.org/abs/2304.05294)

    本文提出了一种多数据因果特征选择方法，它可以同时处理一组时间序列数据集，生成一个单一的因果驱动集，并且可以过滤掉因果虚假链接，最终输入到机器学习模型中预测目标。

    

    强健的特征选择对于创建可靠和可解释的机器学习（ML）模型至关重要。在领域知识有限、潜在交互未知的情况下设计统计预测模型时，选择最优特征集通常很困难。为了解决这个问题，我们引入了一种多数据（M）因果特征选择方法，它同时处理一组时间序列数据集，并生成一个单一的因果驱动集。该方法使用Tigramite Python包中实现的因果发现算法PC1或PCMCI。这些算法利用条件独立性测试推断因果图的部分。我们的因果特征选择方法在将剩余因果特征作为输入传递给ML模型（多元线性回归，随机森林）预测目标之前，过滤掉因果虚假链接。我们将该框架应用于预测西太平洋热带地区的地震强度。

    Robust feature selection is vital for creating reliable and interpretable Machine Learning (ML) models. When designing statistical prediction models in cases where domain knowledge is limited and underlying interactions are unknown, choosing the optimal set of features is often difficult. To mitigate this issue, we introduce a Multidata (M) causal feature selection approach that simultaneously processes an ensemble of time series datasets and produces a single set of causal drivers. This approach uses the causal discovery algorithms PC1 or PCMCI that are implemented in the Tigramite Python package. These algorithms utilize conditional independence tests to infer parts of the causal graph. Our causal feature selection approach filters out causally-spurious links before passing the remaining causal features as inputs to ML models (Multiple linear regression, Random Forest) that predict the targets. We apply our framework to the statistical intensity prediction of Western Pacific Tropical
    
[^31]: 深动量多重边际Schr\"odinger桥模型

    Deep Momentum Multi-Marginal Schr\"odinger Bridge. (arXiv:2303.01751v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.01751](http://arxiv.org/abs/2303.01751)

    该论文提出了一种新的计算框架DMSB，它可以学习满足时间上位置边际约束的随机系统的平滑度量值样条，用于解决高维多边际轨迹推断任务，并在实验中表现出显著的性能优势。同时，该框架还为解决具有各种类型的边际约束的随机轨迹重建任务提供了一个通用框架。

    

    在粗略时间间隔下，使用未标记样本从分布中重建人口动态是一个关键的挑战。最近的方法如流模型或Schr\"odinger桥模型表现出诱人的性能，但是推断出的样本轨迹未能解释潜在的随机性，或者是DMSB，一种新颖的计算框架，它能够学习满足时间上位置边际约束的随机系统的平滑度量值样条。通过调整著名的Bregman迭代和将比例拟合迭代扩展到相空间，我们成功地高效处理了高维多边际轨迹推断任务。我们的算法在合成数据集和真实的单细胞RNA序列数据集实验中显著优于基线。此外，所提出的DMSB框架为解决具有各种类型的边际约束的随机轨迹重建任务提供了一个通用框架。

    It is a crucial challenge to reconstruct population dynamics using unlabeled samples from distributions at coarse time intervals. Recent approaches such as flow-based models or Schr\"odinger Bridge (SB) models have demonstrated appealing performance, yet the inferred sample trajectories either fail to account for the underlying stochasticity or are $\underline{D}$eep $\underline{M}$omentum Multi-Marginal $\underline{S}$chr\"odinger $\underline{B}$ridge(DMSB), a novel computational framework that learns the smooth measure-valued spline for stochastic systems that satisfy position marginal constraints across time. By tailoring the celebrated Bregman Iteration and extending the Iteration Proportional Fitting to phase space, we manage to handle high-dimensional multi-marginal trajectory inference tasks efficiently. Our algorithm outperforms baselines significantly, as evidenced by experiments for synthetic datasets and a real-world single-cell RNA sequence dataset. Additionally, the propos
    
[^32]: 近乎贝叶斯最优的伪标签选择

    Approximately Bayes-Optimal Pseudo Label Selection. (arXiv:2302.08883v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08883](http://arxiv.org/abs/2302.08883)

    本文介绍了BPLS，一种用于PLS的贝叶斯框架，通过解析逼近选择标签实例的标准，以避免由过度自信但错误预测的实例选择而导致的确认偏差问题。

    

    自训练的半监督学习严重依赖于伪标签选择（PLS）。选择通常取决于初始模型拟合标记数据的程度。过早的过拟合可能通过选择具有过度自信但错误的预测的实例（通常称为确认偏差）而传播到最终模型。本文介绍了BPLS，这是一种用于PLS的贝叶斯框架，旨在减轻这个问题。其核心是选择标签实例的标准：伪样本的后验预测的分析近似。我们通过证明伪样本的后验预测的贝叶斯最优性获得了这种选择标准。我们进一步通过解析逼近克服计算难题。它与边际似然的关系使我们能够提出基于拉普拉斯方法和高斯积分的逼近。我们针对参数广义线性和非参数广义加性模型对BPLS进行了实证评估。

    Semi-supervised learning by self-training heavily relies on pseudo-label selection (PLS). The selection often depends on the initial model fit on labeled data. Early overfitting might thus be propagated to the final model by selecting instances with overconfident but erroneous predictions, often referred to as confirmation bias. This paper introduces BPLS, a Bayesian framework for PLS that aims to mitigate this issue. At its core lies a criterion for selecting instances to label: an analytical approximation of the posterior predictive of pseudo-samples. We derive this selection criterion by proving Bayes optimality of the posterior predictive of pseudo-samples. We further overcome computational hurdles by approximating the criterion analytically. Its relation to the marginal likelihood allows us to come up with an approximation based on Laplace's method and the Gaussian integral. We empirically assess BPLS for parametric generalized linear and non-parametric generalized additive models
    
[^33]: 鲁棒的在线主动学习策略

    Robust online active learning. (arXiv:2302.00422v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.00422](http://arxiv.org/abs/2302.00422)

    本文提出了一种自适应方法，用于鲁棒的在线主动学习，并在受污染的数据流中证明了其性能表现优异，同时确保了稳定性并减少异常值的负面影响。

    

    在许多工业应用中，获得标记的观测数据并不简单，通常需要人工专家干预或使用昂贵的测试设备。在这种情况下，主动学习可以大大提高拟合模型时最信息数据点的建议。减少模型开发所需的观测数据数量可以减轻训练所需的计算负担和标记相关的操作支出。特别是在线主动学习，在需要在极短时间内决定是否获取数据点标记的高容量生产过程中非常有用。然而，尽管最近致力于开发在线主动学习策略，但在存在异常值的情况下这些方法的行为仍未得到彻底研究。在这项工作中，我们调查了在线主动线性回归在受污染的数据流中的性能，并提出了一种自适应方法，用于鲁棒的在线主动学习，同时保证稳定性并减少异常值的负面影响。

    In many industrial applications, obtaining labeled observations is not straightforward as it often requires the intervention of human experts or the use of expensive testing equipment. In these circumstances, active learning can be highly beneficial in suggesting the most informative data points to be used when fitting a model. Reducing the number of observations needed for model development alleviates both the computational burden required for training and the operational expenses related to labeling. Online active learning, in particular, is useful in high-volume production processes where the decision about the acquisition of the label for a data point needs to be taken within an extremely short time frame. However, despite the recent efforts to develop online active learning strategies, the behavior of these methods in the presence of outliers has not been thoroughly examined. In this work, we investigate the performance of online active linear regression in contaminated data strea
    
[^34]: 不考虑图骨架的组合因果赌博机

    Combinatorial Causal Bandits without Graph Skeleton. (arXiv:2301.13392v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.13392](http://arxiv.org/abs/2301.13392)

    本文研究了在二值一般因果模型和BGLMs上不考虑图骨架的组合因果赌博机问题，提出了可在BGLMs上实现的无需图骨架的遗憾最小化算法，达到了与依赖于图结构的最先进算法相同的渐进遗憾率$O(\sqrt{T}\ln T)$。

    

    在组合因果赌博机问题中，学习代理在每一轮选择一组变量进行干预，收集观测变量的反馈以最小化期望遗憾或样本复杂度。先前的工作研究了一般因果模型和二值广义线性模型（BGLMs）中的问题。但是，它们都需要先验知识来构建因果关系图。本文研究了在二值一般因果模型和BGLMs上不考虑图骨架的组合因果赌博机问题。我们首先在一般因果模型上提供了累积遗憾的指数下限。然后，我们设计了一种无需图骨架来实现BGLMs的遗憾最小化算法，表明它仍然达到$O(\sqrt{T}\ln T)$的期望遗憾。这个渐进的遗憾率与依赖于图结构的最先进算法相同。

    In combinatorial causal bandits (CCB), the learning agent chooses a subset of variables in each round to intervene and collects feedback from the observed variables to minimize expected regret or sample complexity. Previous works study this problem in both general causal models and binary generalized linear models (BGLMs). However, all of them require prior knowledge of causal graph structure. This paper studies the CCB problem without the graph structure on binary general causal models and BGLMs. We first provide an exponential lower bound of cumulative regrets for the CCB problem on general causal models. To overcome the exponentially large space of parameters, we then consider the CCB problem on BGLMs. We design a regret minimization algorithm for BGLMs even without the graph skeleton and show that it still achieves $O(\sqrt{T}\ln T)$ expected regret. This asymptotic regret is the same as the state-of-art algorithms relying on the graph structure. Moreover, we sacrifice the regret t
    
[^35]: 一种混合类别相关核的高斯过程

    A mixed-categorical correlation kernel for Gaussian process. (arXiv:2211.08262v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2211.08262](http://arxiv.org/abs/2211.08262)

    提出一种新的混合类别相关核的高斯过程代理，相较于其他现有模型在分析和工程问题上表现更好。

    

    近年来，基于高斯过程代理的混合类别元模型引起了越来越多的关注。在这种情况下，一些现有的方法使用不同的策略，通过使用连续核（例如，连续松弛和Gower距离基于高斯过程）或通过直接估计相关矩阵。在本文中，我们提出了一种基于核的方法，将连续指数核扩展为处理混合类别变量。所提出的核引导到了一个新的高斯代理，它概括了连续松弛和Gower距离基于高斯过程模型。我们在分析和工程问题上证明了，我们的提出的高斯过程模型比其他基于核的现有模型具有更高的可能性和更小的残差误差。我们的方法可使用开源软件SMT。

    Recently, there has been a growing interest for mixed-categorical meta-models based on Gaussian process (GP) surrogates. In this setting, several existing approaches use different strategies either by using continuous kernels (e.g., continuous relaxation and Gower distance based GP) or by using a direct estimation of the correlation matrix. In this paper, we present a kernel-based approach that extends continuous exponential kernels to handle mixed-categorical variables. The proposed kernel leads to a new GP surrogate that generalizes both the continuous relaxation and the Gower distance based GP models. We demonstrate, on both analytical and engineering problems, that our proposed GP model gives a higher likelihood and a smaller residual error than the other kernel-based state-of-the-art models. Our method is available in the open-source software SMT.
    
[^36]: 多分位数发布的学习增强私有算法

    Learning-Augmented Private Algorithms for Multiple Quantile Release. (arXiv:2210.11222v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2210.11222](http://arxiv.org/abs/2210.11222)

    本文提出一种新的隐私保护方法：使用学习增强算法框架，为多分位数发布任务提供可扩展的预测质量误差保证。

    

    当应用差分隐私于敏感数据时，我们常常可以利用额外的信息例如其他敏感数据、公众数据或人类信息先验来提升性能。本文提出了使用学习增强算法（或具有预测能力的算法）框架，这个框架通常使用于优化时间复杂度或竞争比率。该框架为设计和分析保护隐私的方法提供了一种强有力的方法，并能够利用这些额外信息以提高效用。该想法体现在重要的多分位数发布任务中，在此我们得出了随着自然质量预测的错误保证，同时（几乎）恢复了最先进的预测独立的保证。我们的分析具有几个优点，包括对数据的最小假设，一种自然的增强鲁棒性的方法，以及为两个从其他数据中学习预测的新颖“元”算法提供有用的替代损失。

    When applying differential privacy to sensitive data, we can often improve performance using external information such as other sensitive data, public data, or human priors. We propose to use the learning-augmented algorithms (or algorithms with predictions) framework -- previously applied largely to improve time complexity or competitive ratios -- as a powerful way of designing and analyzing privacy-preserving methods that can take advantage of such external information to improve utility. This idea is instantiated on the important task of multiple quantile release, for which we derive error guarantees that scale with a natural measure of prediction quality while (almost) recovering state-of-the-art prediction-independent guarantees. Our analysis enjoys several advantages, including minimal assumptions about the data, a natural way of adding robustness, and the provision of useful surrogate losses for two novel ``meta" algorithms that learn predictions from other (potentially sensitiv
    
[^37]: 带约束状态空间模型的扩散时间序列插补和预测

    Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models. (arXiv:2208.09399v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.09399](http://arxiv.org/abs/2208.09399)

    本文提出了一种新的状态空间框架(SSSD)来插补时间序列数据中的缺失值和进行预测，在各种数据集和不同的缺失情况下，SSSD都表现出更好的性能，并可以有效处理黑屏缺失情况。

    

    缺失值的插补对于许多现实世界的数据分析管道来说是一个重要的障碍。本文聚焦于时间序列数据，并提出了SSSD，这是一种依赖于两个新兴技术的插补模型，分别是（条件）扩散模型作为最先进的生成模型以及带约束的状态空间模型作为内部模型架构，其特别适用于捕捉时间序列数据中的长期依赖关系。我们证明，SSSD在各种数据集和不同的缺失情况下，包括挑战性的黑屏缺失情况下，均可达到或甚至超过最先进的概率插补和预测性能，而先前的方法则无法提供有意义的结果。

    The imputation of missing values represents a significant obstacle for many real-world data analysis pipelines. Here, we focus on time series data and put forward SSSD, an imputation model that relies on two emerging technologies, (conditional) diffusion models as state-of-the-art generative models and structured state space models as internal model architecture, which are particularly suited to capture long-term dependencies in time series data. We demonstrate that SSSD matches or even exceeds state-of-the-art probabilistic imputation and forecasting performance on a broad range of data sets and different missingness scenarios, including the challenging blackout-missing scenarios, where prior approaches failed to provide meaningful results.
    
[^38]: 集合卡尔曼更新的非渐近分析: 有效维度和本地化

    Non-Asymptotic Analysis of Ensemble Kalman Updates: Effective Dimension and Localization. (arXiv:2208.03246v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2208.03246](http://arxiv.org/abs/2208.03246)

    本文开发了集合卡尔曼更新的非渐近分析，解释了为什么在先前的协方差具有中等的有效维度、快速谱衰减或近似稀疏的情况下，小的集合大小就足够了。

    

    许多用于反问题和数据同化的现代算法依赖于集合卡尔曼更新来将先前的预测结果与观测数据融合。集合卡尔曼方法通常在小集合大小时表现良好，这在生成每个粒子很昂贵的应用中是必要的。本文开发了集合卡尔曼更新的非渐近分析，从理论上严格说明了为什么如果先前的协方差具有中等的有效维度，快速谱衰减或近似稀疏，则小的集合大小就足够了。我们在统一框架下提出了我们的理论，比较了使用扰动观测、平方根滤波和本地化的集合卡尔曼更新的几种实现。作为我们分析的一部分，我们开发了适用于近似稀疏矩阵的无维度协方差估计界限，这可能是独立感兴趣的内容。

    Many modern algorithms for inverse problems and data assimilation rely on ensemble Kalman updates to blend prior predictions with observed data. Ensemble Kalman methods often perform well with a small ensemble size, which is essential in applications where generating each particle is costly. This paper develops a non-asymptotic analysis of ensemble Kalman updates that rigorously explains why a small ensemble size suffices if the prior covariance has moderate effective dimension due to fast spectrum decay or approximate sparsity. We present our theory in a unified framework, comparing several implementations of ensemble Kalman updates that use perturbed observations, square root filtering, and localization. As part of our analysis, we develop new dimension-free covariance estimation bounds for approximately sparse matrices that may be of independent interest.
    
[^39]: Wasserstein多元自回归模型用于建模分布时间序列及其在图形学习中的应用

    Wasserstein multivariate auto-regressive models for modeling distributional time series and its application in graph learning. (arXiv:2207.05442v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2207.05442](http://arxiv.org/abs/2207.05442)

    本文提出了一种新的自回归模型，用于分析多元分布时间序列。并且在Wasserstein空间中建模了随机对象，提供了该模型的解的存在性和一致估计器。此方法可以应用于年龄分布和自行车共享网络的观察数据。

    

    我们提出了一种新的自回归模型，用于统计分析多元分布时间序列。感兴趣的数据包括一组在实线有界间隔上支持的概率测度的多个系列，并且被不同时间瞬间所索引。概率测度被建模为Wasserstein空间中的随机对象。我们通过在Lebesgue测度的切空间中建立自回归模型，首先对所有原始测度进行居中处理，以便它们的Fréchet平均值成为Lebesgue测度。利用迭代随机函数系统的理论，提供了这样一个模型的解的存在性、唯一性和平稳性的结果。我们还提出了模型系数的一致估计器。除了对模拟数据的分析，我们还使用两个实际数据集进行了模型演示：一个是不同国家年龄分布的观察数据集，另一个是巴黎自行车共享网络的观察数据集。

    We propose a new auto-regressive model for the statistical analysis of multivariate distributional time series. The data of interest consist of a collection of multiple series of probability measures supported over a bounded interval of the real line, and that are indexed by distinct time instants. The probability measures are modelled as random objects in the Wasserstein space. We establish the auto-regressive model in the tangent space at the Lebesgue measure by first centering all the raw measures so that their Fr\'echet means turn to be the Lebesgue measure. Using the theory of iterated random function systems, results on the existence, uniqueness and stationarity of the solution of such a model are provided. We also propose a consistent estimator for the model coefficient. In addition to the analysis of simulated data, the proposed model is illustrated with two real data sets made of observations from age distribution in different countries and bike sharing network in Paris. Final
    
[^40]: TabPFN：在一秒内解决小型表格分类问题的Transformer

    TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. (arXiv:2207.01848v5 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.01848](http://arxiv.org/abs/2207.01848)

    TabPFN是一种可以在不到一秒钟内完成小型表格数据集的监督分类的Transformer，无需超参数调整，并且具有竞争力。它使用先验适应网络（PFN）逼近基于先验的贝叶斯推断，先验融合了因果推理的思想。

    

    本文提出了TabPFN，一种经过训练的Transformer，可以在不到一秒钟的时间内完成小型表格数据集的监督分类，无需超参数调整，并且在分类方法的最新状态下具有竞争力。TabPFN完全包含在我们网络的权重中，接受训练和测试样本作为设置值输入，并在单个前向传递中为整个测试集提供预测。TabPFN是一种先验适应网络（PFN），只需要线下训练一次，即可逼近基于我们的先验的合成数据集上的贝叶斯推断。这个先验融合了因果推理的思想：它包括一个大的结构因果模型空间，偏好于简单结构。在OpenML-CC18套件的18个包含最多1000个训练数据点、最多100个纯数值特征且无缺失值、最多10个类别的数据集中，我们展示了我们的方法明显优于提升树，与复杂的最新AutoM方法表现相当。

    We present TabPFN, a trained Transformer that can do supervised classification for small tabular datasets in less than a second, needs no hyperparameter tuning and is competitive with state-of-the-art classification methods. TabPFN is fully entailed in the weights of our network, which accepts training and test samples as a set-valued input and yields predictions for the entire test set in a single forward pass. TabPFN is a Prior-Data Fitted Network (PFN) and is trained offline once, to approximate Bayesian inference on synthetic datasets drawn from our prior. This prior incorporates ideas from causal reasoning: It entails a large space of structural causal models with a preference for simple structures. On the 18 datasets in the OpenML-CC18 suite that contain up to 1 000 training data points, up to 100 purely numerical features without missing values, and up to 10 classes, we show that our method clearly outperforms boosted trees and performs on par with complex state-of-the-art AutoM
    
[^41]: 使用可微分超几何分布进行小组重要性学习

    Learning Group Importance using the Differentiable Hypergeometric Distribution. (arXiv:2203.01629v5 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.01629](http://arxiv.org/abs/2203.01629)

    本文提出了不同iable hypergeometric distribution，使用重要性学习方法来解决在许多应用程序中将一组元素划分为先验未知大小的子集的问题，并在弱监督学习和聚类方面优于以前的方法。

    

    在许多应用程序中，将一组元素划分为先验未知大小的子集是必要的。这些子集大小很少明确学习 - 无论是聚类应用程序中的簇大小还是弱监督学习中的共享与独立生成潜在因素的数量。由于硬性约束条件，正确子集大小的概率分布是不可微分的，这禁止了基于梯度的优化。在这项工作中，我们提出了可微分超几何分布。超几何分布基于它们的相对重要性模拟不同组大小的概率。我们引入可重参数化梯度来学习小组之间的重要性，并强调在两个典型应用程序中显式学习子集大小的优点：弱监督学习和聚类。在这两个应用程序中，我们优于依赖于次优启发式模拟未知大小的先前方法。

    Partitioning a set of elements into subsets of a priori unknown sizes is essential in many applications. These subset sizes are rarely explicitly learned - be it the cluster sizes in clustering applications or the number of shared versus independent generative latent factors in weakly-supervised learning. Probability distributions over correct combinations of subset sizes are non-differentiable due to hard constraints, which prohibit gradient-based optimization. In this work, we propose the differentiable hypergeometric distribution. The hypergeometric distribution models the probability of different group sizes based on their relative importance. We introduce reparameterizable gradients to learn the importance between groups and highlight the advantage of explicitly learning the size of subsets in two typical applications: weakly-supervised learning and clustering. In both applications, we outperform previous approaches, which rely on suboptimal heuristics to model the unknown size of
    
[^42]: 编码保护分类属性的公平性影响。

    Fairness Implications of Encoding Protected Categorical Attributes. (arXiv:2201.11358v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.11358](http://arxiv.org/abs/2201.11358)

    该研究比较了两种常用的编码方法-“one-hot编码”和“target编码”，并探讨了其对机器学习模型性能和公平性的影响。

    

    过去的研究表明，在机器学习中明确使用保护属性可以同时提高性能和公平性。但是，许多机器学习算法无法直接处理分类属性，例如出生国家或种族。由于保护属性经常是分类的，因此必须将其编码为可以输入所选择的机器学习算法的特征，例如支持向量机、梯度提升决策树或线性模型。编码方法影响机器学习算法将学习如何和什么，影响模型的性能和公平性。该研究比较了两种最著名的编码方法——“one-hot编码”和“target编码”的准确性和公平性影响。我们区分了这些编码方法可能产生的两种诱导偏差类型，这可能导致不公平的模型。第一种类型是无法消除的偏差，由于直接组别类别歧视而导致。

    Past research has demonstrated that the explicit use of protected attributes in machine learning can improve both performance and fairness. Many machine learning algorithms, however, cannot directly process categorical attributes, such as country of birth or ethnicity. Because protected attributes frequently are categorical, they must be encoded as features that can be input to a chosen machine learning algorithm, e.g.\ support vector machines, gradient boosting decision trees or linear models. Thereby, encoding methods influence how and what the machine learning algorithm will learn, affecting model performance and fairness. This work compares the accuracy and fairness implications of the two most well-known encoding methods: \emph{one-hot encoding} and \emph{target encoding}. We distinguish between two types of induced bias that may arise from these encoding methods and may lead to unfair models. The first type, \textit{irreducible bias}, is due to direct group category discriminatio
    
[^43]: CausalSim: 一种用于无偏差追踪驱动仿真的因果框架

    CausalSim: A Causal Framework for Unbiased Trace-Driven Simulation. (arXiv:2201.01811v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.01811](http://arxiv.org/abs/2201.01811)

    CausalSim提出了一种因果框架，通过学习系统动态和潜在因素的因果模型，消除追踪数据中的偏差，解决了当前追踪驱动仿真器的缺陷。

    

    我们提出了CausalSim，一种用于无偏差追踪驱动仿真的因果框架。当前的追踪驱动仿真器假设进行仿真的干预（例如，新算法）不会影响追踪的有效性。然而，现实世界中的追踪常常会受到算法在追踪收集期间进行选择的影响，因此，在干预下重演追踪可能会导致不正确的结果。CausalSim通过学习系统动态和捕获追踪收集期间基础系统条件的潜在因素的因果模型来解决这个挑战。它使用固定算法集下的初始随机对照试验（RCT）来学习这些模型，然后在模拟新算法时应用它们来消除追踪数据中的偏差。

    We present CausalSim, a causal framework for unbiased trace-driven simulation. Current trace-driven simulators assume that the interventions being simulated (e.g., a new algorithm) would not affect the validity of the traces. However, real-world traces are often biased by the choices algorithms make during trace collection, and hence replaying traces under an intervention may lead to incorrect results. CausalSim addresses this challenge by learning a causal model of the system dynamics and latent factors capturing the underlying system conditions during trace collection. It learns these models using an initial randomized control trial (RCT) under a fixed set of algorithms, and then applies them to remove biases from trace data when simulating new algorithms.  Key to CausalSim is mapping unbiased trace-driven simulation to a tensor completion problem with extremely sparse observations. By exploiting a basic distributional invariance property present in RCT data, CausalSim enables a nove
    
[^44]: 神经网络普适性的统一建构框架

    A Unified and Constructive Framework for the Universality of Neural Networks. (arXiv:2112.14877v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.14877](http://arxiv.org/abs/2112.14877)

    本论文提出了神经网络普适性的建构框架，任何nAI激活函数都是普适的，该框架具有统一、构造性和新视角的优势。

    

    神经网络之所以能够复制复杂的任务或函数之一是因为它们的普适性。虽然过去几十年来神经网络理论取得了巨大的进展，但尚未提供单一的建构框架来解释神经网络的普适性。本文是第一个为大多数已有激活函数提供统一的建构框架以解释它们的普适性的尝试。在框架的核心是神经网络近似恒等（nAI）的概念。主要的结果是：\emph{任何nAI激活函数都是普适的}。事实证明，大多数激活函数都是nAI，因此在紧致空间连续函数空间内是普适的。该框架比现有的对应物具有\textbf{几个优势}。首先，它是建立在从功能分析、概率论和数值分析的基本手段之上的构造性框架。其次，它是第一个适用于包括大多数已有激活函数在内的大类激活函数的统一框架。第三，它提出了神经网络普适性的新视角。

    One of the reasons why many neural networks are capable of replicating complicated tasks or functions is their universal property. Though the past few decades have seen tremendous advances in theories of neural networks, a single constructive framework for neural network universality remains unavailable. This paper is the first effort to provide a unified and constructive framework for the universality of a large class of activation functions including most of existing ones. At the heart of the framework is the concept of neural network approximate identity (nAI). The main result is: {\em any nAI activation function is universal}. It turns out that most of existing activation functions are nAI, and thus universal in the space of continuous functions on compacta. The framework induces {\bf several advantages} over the contemporary counterparts. First, it is constructive with elementary means from functional analysis, probability theory, and numerical analysis. Second, it is the first un
    
[^45]: 学习未知离散时间线性系统的安全滤波器

    Learning Safety Filters for Unknown Discrete-Time Linear Systems. (arXiv:2111.00631v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2111.00631](http://arxiv.org/abs/2111.00631)

    本论文提出了一种基于学习的安全滤波器，针对带有未知模型和未知协方差的高斯噪声的离散时间线性时不变系统，通过收紧安全约束和构建鲁棒优化问题，以最小程度地修改名义控制动作，以高概率确保安全性。

    

    本文针对带有未知模型和未知协方差的高斯噪声的离散时间线性时不变系统开发了一种基于学习的安全滤波器。安全性通过对状态和控制输入施加多面体约束来描述。经验性地学习模型和过程噪声协方差及其置信区间，构建了一个鲁棒优化问题，以最小程度地修改名义控制动作，以高概率确保安全性。优化问题依赖于收紧原始的安全性约束。由于最初缺乏可靠模型构建所需信息，因此收紧的幅度较大，但随着更多数据的可用性，其逐渐缩小。

    A learning-based safety filter is developed for discrete-time linear time-invariant systems with unknown models subject to Gaussian noises with unknown covariance. Safety is characterized using polytopic constraints on the states and control inputs. The empirically learned model and process noise covariance with their confidence bounds are used to construct a robust optimization problem for minimally modifying nominal control actions to ensure safety with high probability. The optimization problem relies on tightening the original safety constraints. The magnitude of the tightening is larger at the beginning since there is little information to construct reliable models, but shrinks with time as more data becomes available.
    
[^46]: 基于分布学习的加权治疗效果估计

    Weighting-Based Treatment Effect Estimation via Distribution Learning. (arXiv:2012.13805v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2012.13805](http://arxiv.org/abs/2012.13805)

    本文提出了一种基于分布学习的加权方法，通过学习协变量在治疗组和对照组的分布并利用比率作为权重来估计治疗效果，以缓解现有加权方法中模型错误设置的问题。

    

    现有的治疗效果估计加权方法通常建立在倾向得分或协变量平衡的基础上。它们通常对治疗分配或结果模型施加强制性假设，以获得无偏估计，如线性或特定的函数形式，这很容易导致模型错误设置的主要缺点。本文旨在通过开发基于分布学习的加权方法来缓解这些问题。我们首先学习以治疗分配为条件的协变量真实潜在分布，然后利用治疗组协变量密度与对照组协变量密度之比作为估计治疗效果的权重。具体而言，我们提出通过变量间的可逆变换来近似治疗组和对照组中的协变量分布。为了证明我们方法的优越性、鲁棒性和泛化性，我们进行了大量实验。

    Existing weighting methods for treatment effect estimation are often built upon the idea of propensity scores or covariate balance. They usually impose strong assumptions on treatment assignment or outcome model to obtain unbiased estimation, such as linearity or specific functional forms, which easily leads to the major drawback of model mis-specification. In this paper, we aim to alleviate these issues by developing a distribution learning-based weighting method. We first learn the true underlying distribution of covariates conditioned on treatment assignment, then leverage the ratio of covariates' density in the treatment group to that of the control group as the weight for estimating treatment effects. Specifically, we propose to approximate the distribution of covariates in both treatment and control groups through invertible transformations via change of variables. To demonstrate the superiority, robustness, and generalizability of our method, we conduct extensive experiments usi
    
[^47]: 自杀风险的罕见和不确定诊断的生存建模研究

    Survival Modeling of Suicide Risk with Rare and Uncertain Diagnoses. (arXiv:2009.02597v2 [stat.AP] UPDATED)

    [http://arxiv.org/abs/2009.02597](http://arxiv.org/abs/2009.02597)

    该研究针对罕见和不确定诊断的自杀风险进行生存建模，采用医疗索赔数据研究了自杀未遂的患者随后的自杀未遂风险，通过开发一种综合的Cox淘汰模型来完成生存回归。

    

    在通过改善行为保健实现自杀预防的迫切需求的驱使下，我们使用医疗索赔数据研究了因自杀未遂而住院并后来出院的患者随后的自杀未遂风险。了解这些患者在升高的自杀风险下的风险行为是实现“零自杀”的目标的重要一步。一个即时且非常规的挑战是，从医疗索赔中识别自杀未遂包含重大的不确定性：几乎有20％的“疑似”自杀未遂是从表明外部原因所致损伤和中毒的诊断编码中确定的。因此，了解这些未确定事件中哪些更有可能是实际的自杀未遂事件以及如何在严重的截尾生存分析中正确地利用它们是非常有趣的。为了解决这些相关问题，我们开发了一种具有正则化的综合Cox淘汰模型来执行生存回归。

    Motivated by the pressing need for suicide prevention through improving behavioral healthcare, we use medical claims data to study the risk of subsequent suicide attempts for patients who were hospitalized due to suicide attempts and later discharged. Understanding the risk behaviors of such patients at elevated suicide risk is an important step toward the goal of "Zero Suicide." An immediate and unconventional challenge is that the identification of suicide attempts from medical claims contains substantial uncertainty: almost 20% of "suspected" suicide attempts are identified from diagnosis codes indicating external causes of injury and poisoning with undermined intent. It is thus of great interest to learn which of these undetermined events are more likely actual suicide attempts and how to properly utilize them in survival analysis with severe censoring. To tackle these interrelated problems, we develop an integrative Cox cure model with regularization to perform survival regression
    
[^48]: 用于在线学习非平稳函数的连续高斯过程

    Sequential Gaussian Processes for Online Learning of Nonstationary Functions. (arXiv:1905.10003v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/1905.10003](http://arxiv.org/abs/1905.10003)

    本文提出了一种基于顺序蒙特卡罗算法的连续高斯过程模型，以解决高斯过程模型的计算复杂度高，难以在线顺序更新的问题，同时允许拟合具有非平稳性质的函数。方法优于现有最先进方法的性能。

    

    许多机器学习问题可以在估计函数的上下文中得到解决，通常这些函数是时间相关的函数，并且是实时地随着观测的到来而估计的。高斯过程是建模实值非线性函数的一个有吸引力的选择，由于其灵活性和不确定性量化。然而，典型的高斯过程回归模型存在若干不足：1）传统高斯过程推断的复杂度$O(N^{3})$随着观测值的个数N成增长；2）逐步更新高斯过程模型不容易；3）协方差核通常对函数施加平稳性约束，而具有非平稳协方差核的高斯过程通常难以在实践中使用。为了克服这些问题，我们提出了一个顺序蒙特卡罗算法来拟合无限混合高斯过程，以捕捉非平稳行为，同时允许在线、分布推断。我们的方法在实验中优于现有最先进方法的性能。

    Many machine learning problems can be framed in the context of estimating functions, and often these are time-dependent functions that are estimated in real-time as observations arrive. Gaussian processes (GPs) are an attractive choice for modeling real-valued nonlinear functions due to their flexibility and uncertainty quantification. However, the typical GP regression model suffers from several drawbacks: 1) Conventional GP inference scales $O(N^{3})$ with respect to the number of observations; 2) Updating a GP model sequentially is not trivial; and 3) Covariance kernels typically enforce stationarity constraints on the function, while GPs with non-stationary covariance kernels are often intractable to use in practice. To overcome these issues, we propose a sequential Monte Carlo algorithm to fit infinite mixtures of GPs that capture non-stationary behavior while allowing for online, distributed inference. Our approach empirically improves performance over state-of-the-art methods fo
    

