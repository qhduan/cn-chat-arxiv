# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spectral and Subspace Dynamics of LoRA in Supervised Fine-Tuning and Reinforcement Learning](https://arxiv.org/abs/2611.08414) | 该研究通过追踪LoRA更新的奇异谱与子空间，发现强化学习相比监督微调表现出谱集中和更快的子空间旋转，并揭示了组均值基线减除是造成这两种差异的根本机制。 |
| [^2] | [Conformal Prediction for Time-Dependent PDEs](https://arxiv.org/abs/2611.08409) | 本文证明了时间依赖偏微分方程的共形预测覆盖保证在本质上难以实现——函数空间中相近时刻的分布可相互奇异，离散化中全变差距离随网格分辨率指数饱和——并给出了为确保覆盖而对空间离散化施加的必要限制。 |
| [^3] | [Unifying Central Flow, Rod Flow, and the Edge Coupling](https://arxiv.org/abs/2611.08358) | 本文提出“边缘耦合”这一泛函，证明中心流与杆流均可由它推导得出，二者分别对应不同时间尺度上的时间粗粒化，从而统一了刻画梯度下降稳定性边缘行为的两种连续时间模型。 |
| [^4] | [Series-wise Cross-validation for Global Time Series Forecasting](https://arxiv.org/abs/2611.08252) | 该论文为基于深度学习的全局时间序列预测模型形式化了按序列划分的交叉验证方法，突破传统仅按时间划分的范式，并在11个数据集和10种神经网络架构上系统评估了K折、自助法等多种序列级交叉验证策略。 |
| [^5] | [BUMP: Brief Unexpected Mid-Interpolation Peak](https://arxiv.org/abs/2611.08182) | 本文发现了神经网络插值中的 BUMP 现象（测试精度在插值端点附近短暂提升后向中点下降），并基于此提出了将权重混合与预测空间混合相结合的新型集成方法。 |
| [^6] | [Patch Attention Wisely: Understanding Patch Length in Attention-Based Forecasting](https://arxiv.org/abs/2611.08160) | 该论文提出了一个理论框架，解释了基于注意力的时间序列预测中补丁长度为何影响预测精度，并基于线性状态空间模型的精确分解，推导出在时间相关性下平衡表示误差与估计误差的补丁选择准则。 |
| [^7] | [Identifying Neural Dynamics Using Nonlinear Interventional State Space Models](https://arxiv.org/abs/2611.08113) | 该研究将干预状态空间模型的可识别性理论从线性动力学推广到一般非线性潜在动力学与单射发射函数，为识别神经回路动力学提供了更广泛的理论保证。 |
| [^8] | [Doubly-robust causal inference for multi-valued treatment regimes with recurrent neural networks](https://arxiv.org/abs/2611.08077) | 该论文将循环神经网络融入扩展至多值治疗的纵向目标最小损失估计（LTMLE）框架，实现了对多值治疗方案因果效应的双重稳健估计，模拟结果显示该方法近似无偏且覆盖率接近名义水平。 |
| [^9] | [Joint Modeling of Conditional Mean and Variability in Multilevel Functional Data: Application to Postprandial Glucose Trajectories in the AEGIS Study](https://arxiv.org/abs/2611.07937) | 该论文提出了一种异方差函数混合效应回归框架，联合建模多水平重复函数轨迹的条件均值与方差，并通过参与者层面的自助法量化参数不确定性，应用于AEGIS研究揭示了各因素对餐后血糖均值与变异性的时依赖差异化影响。 |
| [^10] | [Network Effect Decomposition for Causal Effect Estimation under Network Interference](https://arxiv.org/abs/2611.07879) | 提出NEDNet网络，将网络干扰下的结果响应分解为查询无关的处理前基线以及相互独立的处理响应通路和暴露响应通路，从而在观测网络数据上实现更准确的因果效应估计。 |
| [^11] | [Optimal Learning in Games with Arbitrary Delays: Best-of-Both-Worlds for Self-Play and Adversarial Opponents](https://arxiv.org/abs/2611.07773) | 提出单一算法MAO-OMD，在具有任意未知反馈延迟的零和博弈中，同时实现了自博弈时的最优快速遗憾速率 O(σ_max+1) 和对抗任意对手时极小极大最优的 O(√(T+d_tot)) 保证。 |
| [^12] | [Identification and Bounding of Joint Expectation over Potential Outcomes](https://arxiv.org/abs/2611.07536) | 本文提出同单调性与反单调性两种新假设，为潜在结果函数联合期望的识别提供了适用于离散、连续和混合结果变量的统一框架，并在无这些假设时推导出尖锐界限，为衡量因果效应异质性提供了新的估计工具。 |
| [^13] | [Changepoint localization via nested exchangeability tests: from point estimates to confidence intervals](https://arxiv.org/abs/2611.07399) | 该论文提出CLONE方法，通过在以变点估计值为中心的嵌套窗口中进行可交换性检验，为任意黑盒变点估计器构建具有有限样本覆盖保证的无分布置信区间，且仅需O(log n)次置换检验。 |
| [^14] | [Last-Iterate Convergence of Stochastic Gradient Methods for Unbounded Problems](https://arxiv.org/abs/2611.07334) | 本文在去除了有界定义域和有界方差假设的条件下，证明了随机梯度方法的末次迭代具有接近最优的 $\widetilde{\mathcal{O}}(1/\sqrt{T})$ 收敛率，并在迭代总次数已知时达到了最优的 $\mathcal{O}(1/\sqrt{T})$ 阶，与强假设下的已知结果相匹配。 |
| [^15] | [Neural Realized Measures: Forecast-Supervised Intraday Aggregation for Volatility Forecasting](https://arxiv.org/abs/2611.07287) | 该论文提出神经实现测度（NRM），一种以波动率预测为监督、通过可学习的正的上下文相关权重来聚合日内观测的框架，将经典实现测度嵌套为特例，从而改进波动率预测。 |
| [^16] | [Satisficing as competitive Bayesian weighting: a minimal model of goal-directed choice under an unattained target](https://arxiv.org/abs/2611.07270) | 该论文证明“风险敏感满足化”这一多臂老虎机学习规则可等价改写为类贝叶斯估计器“注意力竞争贝叶斯（ACB）”——未达成的抱负水平充当所有动作的先验均值，有限试验预算决定证据覆盖先验的强度，从而给出了以未实现目标为参照的目标导向选择的最小形式化模型。 |
| [^17] | [Double Machine Learning for Panel Data: Identification, Specification, and a Bank Regulation Effect](https://arxiv.org/abs/2611.07249) | 本文将双重/去偏机器学习拓展至面板数据，刻画了共同处理时点下“组内”与“一阶差分”估计量的识别条件，并以2018年美国银行监管改革为例，揭示了模型设定选择与函数形式稳健性对因果识别和估计结果的影响。 |
| [^18] | [Task-Calibrated Random Features via Influence-Based Error Cancellation](https://arxiv.org/abs/2611.07209) | 该论文提出任务校准随机特征（TCRF），通过基于任务特定影响构建的可重用校准空间来平衡频率矩并消除对下游任务有影响的采样误差，从而改进核岭回归和核Stein差异等核方法的近似质量。 |
| [^19] | [ShapleyOG: Fast Adaptive Shapley Value Estimation with Odd Gaussian Processes](https://arxiv.org/abs/2611.07173) | ShapleyOG利用Shapley值仅依赖于奇数分量的特性，提出了一种基于奇数高斯过程的快速自适应估计方法，克服了ShaplEIG等现有自适应方法因每轮迭代重复拟合高斯过程代理模型而带来的计算开销问题。 |
| [^20] | [Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference](https://arxiv.org/abs/2611.07129) | 本文提出了贝叶斯张量链核机器（BTT-KM），通过在TT核心上施加分层ARD先验并结合平均场变分推断，实现了从数据中自动推断TT秩与特征维度的闭式贝叶斯学习，且计算复杂度与确定性方法相同。 |
| [^21] | [ToroFit: Fitting Wedge Sums of Tori to High-Dimensional Data](https://arxiv.org/abs/2611.06990) | 本文提出ToroFit框架，利用持续上同调与上积识别高维点云数据中各环面结构分量，并通过将低能量多元三角多项式拟合归结为广义岭回归，实现对环面楔和型非线性数据的高效检测与参数化。 |
| [^22] | [Sampling Design and Confidence Tubes for Integral Curves on Riemannian Manifolds](https://arxiv.org/abs/2611.00771) | 本文在紧黎曼流形上提出了估计未知向量场积分曲线的被动与主动两种抽样方法，证明了被动核估计的曲线误差阶为 $n^{-2/(d+3)}$ 并给出由线性随机微分方程刻画的函数型中心极限定理，同时证明基于先导信息的主动抽样可将期望最大曲线误差提升至 $N^{-1/2}$ 阶，从而为构建置信管奠定基础。 |
| [^23] | [Scalable Regularized Vector Multiplicative Error Models for Positive-valued Financial Time Series](https://arxiv.org/abs/2610.08443) | 本文提出一种基于分层滞后结构与分块坐标下降算法的正则化对数向量乘性误差模型，实现了高维正值金融时间序列的高效估计与预测。 |
| [^24] | [High-dimensional Gaussian Graphical Model Testing for Long-Memory Time Series](https://arxiv.org/abs/2609.30565) | 提出了一种适用于长记忆时间序列的数据自适应高斯图模型条件独立性检验方法，建立了有限样本高斯逼近界并通过块自助法保证超高维情形下的有效性，检验在水平和功效上均具渐近一致性，可应用于fMRI功能连接性分析。 |
| [^25] | [Counterexamples and Sufficient Conditions: Comments on "Optimally-Transported Generalized Method of Moments"](https://arxiv.org/abs/2609.20260) | 本文对Schennach & Starck提出的最优传输广义矩方法（OTGMM）估计量的核心定理给出反例，证明其定理2-6在原假设条件下不成立，并探讨使这些结论成立的充分条件。 |
| [^26] | [Speed Limit for Information Acquisition in Stochastic Learning Dynamics](https://arxiv.org/abs/2609.08219) | 本研究将随机梯度下降建模为马尔可夫随机过程，推导出费希尔信息流的速度极限，定量约束了可训练参数获取数据中潜在变量信息的速率，并将信息流分解为漂移（确定性学习力）与噪声（SGD涨落）两部分贡献。 |
| [^27] | [Deep Barycentric Regression for Optimal Transport Map Estimation and its Statistical Optimality](https://arxiv.org/abs/2609.06598) | 该论文提出了一种简单的两步方法BROT：先计算未正则化的最优传输方案，再用深度神经网络对所得重心目标进行最小二乘回归，从而实现了兼具统计最优保证与实用可扩展性的最优传输映射估计。 |
| [^28] | [Geometry-Constrained Kolmogorov-Arnold Networks: Learning Edge Geometry via Banach Duality](https://arxiv.org/abs/2608.25807) | 本文提出了一种通过每条边学习标量指数 $p$ 来动态调整激活函数几何结构的KAN变体，在符号回归任务中优于固定基函数方法。 |
| [^29] | [Inference and Uncertainty Quantification for Streaming $r$-PCA](https://arxiv.org/abs/2608.18374) | 本文通过Oja算法解决了流式PCA的尖锐收敛速率和分布推断问题，首次在一般秩下实现无余项的收敛分析，并建立匹配的极小极大下界。 |
| [^30] | [Designing Reinforcement Learning for Diffusion Models: A Unified Path-Space View](https://arxiv.org/abs/2608.14430) | 本文通过统一路径空间原则，揭示了扩散模型RL中反向轨迹与前向匹配方法的共同起源，并推导出方差缩减的等效形式，弥合了方法间的经验差距。 |
| [^31] | [Integrating Background Knowledge for Scalable Causal Discovery](https://arxiv.org/abs/2607.10456) | 该论文提出了一个在因果发现过程中（而非仅作为事后后处理）整合专家背景知识的框架，并针对仅恢复部分图结构的大规模可扩展因果发现方法进行了多算法实现与实验验证。 |
| [^32] | [Diffusion-Driven State Space Models](https://arxiv.org/abs/2606.21036) | 本文提出扩散驱动的状态空间模型（DDSSM），以扩散模型替代传统高斯转移分布，解决了在序列数据上联合训练自编码器与扩散模型的开放问题，并在多峰转移时间序列的拟合与预测上超越现有最先进的深度状态空间模型。 |
| [^33] | [Recursively Trained Diffusion Models: Limiting Collapse Distribution and Spectral Characterization](https://arxiv.org/abs/2606.13796) | 该论文证明，即使在评分估计与采样完全精确的理想条件下，反向扩散的提前停止也会使递归自训练的扩散模型以几何速率收敛到一个唯一的极限分布（即数据分布的 increasingly 高斯平滑版本的无限混合），且该过程如同低通滤波器般滤除高频信息，从而完全刻画了模型坍缩的极限分布与收敛速度。 |
| [^34] | [Querying Counterfactuals on Tissue Graphs with Supervised Disentanglement](https://arxiv.org/abs/2606.08493) | 该论文提出Cellina框架，通过监督解耦将细胞的内在状态与空间上下文分离，并首次将组织图反事实形式化为边扰动和节点扰动两类空间干预，在覆盖超过250万个细胞的基准测试中超越了现有空间与非空间方法。 |
| [^35] | [Fast Rates for Offline Contextual Bandits with Forward-KL Regularization under Single-Policy Concentrability](https://arxiv.org/abs/2605.09214) | 本文首次在单策略集中性条件下，为前向KL正则化的离线上下文老虎机给出了表格型与一般函数逼近设置下的 Õ(ε^{-1}) 快速收敛率上界，通过新颖地运用悲观原理的凸分析框架填补了前向KL正则化统计分析的空白。 |
| [^36] | [Tight Generalization Bounds for Noiseless Inverse Optimization](https://arxiv.org/abs/2605.08866) | 本文为无噪声逆向优化给出了紧致的高概率 O(d/T) 泛化界，在动作唯一性条件下将其与老虎机最优臂识别结果对齐，并证明瞬时与累积遗憾的下界与对抗性设置的上界相匹配，从而表明该速率不可再改进。 |
| [^37] | [Empirical Bayes Rebiasing](https://arxiv.org/abs/2605.08069) | 提出经验贝叶斯再偏置策略，通过估计未知偏差分布从数据中学习应重新引入的偏差量，从而显著缩短校准置信区间，并在预测驱动推断（如大语言模型胜率评估）和家系GWAS直接遗传效应推断中取得显著精度提升。 |
| [^38] | [Almost Sure Convergence Rates of Stochastic Approximation and Reinforcement Learning via a Poisson-Moreau Drift](https://arxiv.org/abs/2605.07104) | 本文通过泊松-莫罗漂移技术，为马尔可夫噪声下期望更新具有压缩性的随机逼近算法（如Q学习和线性时序差分学习）建立了几乎必然收敛速率，在幂律学习率下获得了可任意接近o(n^(1-2η))的速率，在调和学习率下当参数满足条件时平方误差指数可任意接近1。 |
| [^39] | [Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes](https://arxiv.org/abs/2605.06152) | 本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。 |
| [^40] | [Hyper Input Convex Neural Networks for Shape Constrained Learning and Optimal Transport](https://arxiv.org/abs/2604.26942) | HyCNN 结合 Maxout 网络与输入凸神经网络，以指数级更少的参数实现高效凸函数学习，在凸回归、插值以及高维最优传输任务中显著优于现有方法。 |
| [^41] | [Learning to Think from Multiple Thinkers: How Chain-of-Thought Heterogeneity Affects Efficient Learnability](https://arxiv.org/abs/2604.24737) | 该论文首次从学习理论视角研究异构思维链监督问题，提出多种由不同来源（最终答案相同但中间步骤各异）提供CoT数据的异构监督模型，并探讨在单一同构来源可被高效学习的前提下，何时能够从异构监督数据中实现高效学习。 |
| [^42] | [Symplectic Inductive Bias for Data-Driven Target Reachability in Hamiltonian Systems](https://arxiv.org/abs/2604.17213) | 该论文提出利用哈密顿系统固有的辛几何结构与能量水平集上的递归特性作为归纳偏置，从而以数据高效的方式解决非线性控制中的目标可达性问题。 |
| [^43] | [One Method Does Not Fit All: A Large-Scale Benchmark of Imputation Methods for Single-Cell RNA Sequencing Data](https://arxiv.org/abs/2603.24626) | 本文提出了首个大规模标准化基准测试框架，在10种实验方案、6个下游任务和30个数据集上系统评估了16种单细胞RNA测序数据填补方法，结果表明没有一种方法能够适用于所有场景。 |
| [^44] | [The Stability of Online Algorithms in Performative Prediction](https://arxiv.org/abs/2602.24207) | 本文证明部署在表演性预测环境中的任何无悔在线算法都会无条件收敛到（混合）表演性稳定均衡，且无需对人群如何响应预测做任何假设。 |
| [^45] | [Oracle-Robust Online Alignment for Large Language Models](https://arxiv.org/abs/2602.20457) | 本文针对偏好反馈预言机可能存在偏差的大语言模型在线对齐问题，提出最坏情况鲁棒优化目标，并证明该鲁棒目标可精确分解为原始损失加显式敏感度惩罚项，进而设计了投影随机复合更新算法进行求解。 |
| [^46] | [Online Conformal Prediction via Universal Portfolio Algorithms](https://arxiv.org/abs/2602.03168) | 本文识别出“线性化遗憾”这一关键概念并建立了通用的“遗憾到覆盖”理论，据此提出无需任何参数调整的UP-OCP方法，通过将问题归约为两资产投资组合选择问题并利用通用投资组合算法，实现了参数无关的在线保形预测覆盖率保证。 |
| [^47] | [Greedy Alignment Principle for Optimizer Hyperparameter Selection](https://arxiv.org/abs/2512.06370) | 该论文提出了贪心对齐原理，将优化器选择形式化为最大化损失期望下降速率（等价于优化器滤波器与梯度自相关的内积），证明了贪心最优解的存在性与稳定性，并据此为 SGD+Momentum 和 Adam/AdamW 导出了简单的动态动量选择规则。 |
| [^48] | [Variational Estimators for Node Popularity Models](https://arxiv.org/abs/2511.17783) | 本文为双向节点流行度模型（TNPM）开发了一种计算高效且有理论保证的变分期望最大化（VEM）估计框架，在二部网络中证明了社区划分的标签一致性，并实现了优于现有方法的估计精度。 |
| [^49] | [Beyond Maximum Likelihood: Variational Inequality Estimation for Generalized Linear Models](https://arxiv.org/abs/2511.03087) | 本文提出用基于变分不等式的算子均衡条件取代最大似然来估计广义线性模型，证明了其有限样本误差界与渐近正态性以及相应算法的收敛保证，并能处理非标准、非光滑或饱和的连接函数。 |
| [^50] | [Spectral Thresholds in Correlated Spiked Models and Fundamental Limits of Partial Least Squares](https://arxiv.org/abs/2510.17561) | 本文利用随机矩阵理论证明了部分对齐的尖峰互协方差模型中样本奇异值会经历BBP型相变，首次精确刻画了偏最小二乘法（PLS）的信号恢复能力，并揭示了PLS与贝叶斯最优估计器之间的根本性能差距，指出在某些信噪比和相关性区间内PLS会完全无法恢复信号。 |
| [^51] | [On the Rate of Gaussian Approximation for Online Linear Regression Problems](https://arxiv.org/abs/2509.14039) | 本文针对恒定步长下的在线线性回归问题推导了高斯近似速率，显式刻画了其对问题维度和设计矩阵相关量的依赖，并证明当样本量足够大时可获得 $\sqrt{\log{n}/n}$ 阶的正态近似速率。 |
| [^52] | [Online Budget Allocation with Censored Semi-Bandit Feedback](https://arxiv.org/abs/2508.05844) | 本文研究了删失半老虎机反馈下的在线预算分配问题，提出了一种基于乐观原则的算法，并证明在收益递减情形下其遗憾随时间范围呈多对数级增长。 |
| [^53] | [Implicit Regularization via Spectral Neural Networks and Non-linear Matrix Sensing](https://arxiv.org/abs/2402.17595) | 本文在更现实的神经网络背景下探讨了隐式正则化现象，通过研究非线性激活函数的一般类别，严格证明了在矩阵感知问题设置中这些网络的隐式正则化现象，同时提供了严格的速率保证，确保梯度的指数级快速收敛。 |

# 详细

[^1]: 监督微调与强化学习中LoRA的谱与子空间动力学

    Spectral and Subspace Dynamics of LoRA in Supervised Fine-Tuning and Reinforcement Learning

    [https://arxiv.org/abs/2611.08414](https://arxiv.org/abs/2611.08414)

    该研究通过追踪LoRA更新的奇异谱与子空间，发现强化学习相比监督微调表现出谱集中和更快的子空间旋转，并揭示了组均值基线减除是造成这两种差异的根本机制。

    

    低秩适应（LoRA）通常同时用于监督微调（SFT）和强化学习（RL）两个阶段，且往往采用相似的优化器设置。然而，这两个阶段的不同动力学如何在LoRA所优化的低秩增量中体现，目前仍不清楚。通过在数学推理任务上追踪LoRA更新的奇异谱和子空间，我们发现：在所有学习率下，RL都会使谱趋于集中，而SFT在中等学习率下则会扩展谱；此外，在小学习率下，RL旋转奇异子空间的速度远快于SFT。通过在SFT和RL之间逐个成分进行插值，我们确定了组均值基线减除是造成这两种差异的根源。通过LoRA更新的二次多指标模型揭示了其背后的机制：组均值基线减除使两种相互竞争的力量——一致性拟合与梯度噪声——之间的平衡发生倾斜，从而趋向谱集中和更快的（子空间旋转）……

    arXiv:2611.08414v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) is commonly used in both supervised fine-tuning (SFT) and reinforcement learning (RL), often under similar optimizer settings. Yet, how the different dynamics of the two stages manifest in the low-rank increment that LoRA optimizes remains unclear. Tracking the singular spectrum and subspaces of LoRA updates on mathematical reasoning tasks, we find that RL concentrates the spectrum at every learning rate, while SFT expands it at moderate learning rates; moreover, RL rotates the singular subspaces much faster than SFT at small learning rates. Interpolating between SFT and RL one ingredient at a time, we pinpoint group-mean baseline subtraction as the source of both contrasts. A quadratic multi-index model updated via LoRA unveils the underlying mechanism: group-mean baseline subtraction tilts the balance between two competing forces, coherent fitting and gradient noise, toward spectral concentration and faster s
    
[^2]: 时间依赖偏微分方程的共形预测

    Conformal Prediction for Time-Dependent PDEs

    [https://arxiv.org/abs/2611.08409](https://arxiv.org/abs/2611.08409)

    本文证明了时间依赖偏微分方程的共形预测覆盖保证在本质上难以实现——函数空间中相近时刻的分布可相互奇异，离散化中全变差距离随网格分辨率指数饱和——并给出了为确保覆盖而对空间离散化施加的必要限制。

    

    不确定性量化在科学机器学习中至关重要，因为模型的输出会为洪水预报、金融风险管理以及机器热控制等安全关键任务提供依据。共形预测能够提供无分布假设的覆盖保证，但在物理和工程中常见的时间依赖场景下，这些保证可能失效，导致系统性的覆盖不足。我们在由偏微分方程描述的时间依赖物理系统的代理模型背景下研究这一问题。我们证明，在函数空间设定下，任意接近的两个时刻的分布可以相互奇异，使得精确的覆盖保证不可能实现。我们进一步表明，在离散化设定下，分布之间的全变差距离往往随网格分辨率的提高而呈指数级饱和，这要求对空间离散化施加限制才能确保覆盖。最后，对于某些……

    arXiv:2611.08409v1 Announce Type: new  Abstract: Uncertainty quantification is crucial in scientific machine learning, where models inform safety-critical tasks such as flood forecasting, financial risk management, and thermal control in machines. Conformal prediction provides distribution-free coverage guarantees, but in time-dependent settings common to physics and engineering, these guarantees can break down, leading to systematic undercoverage. We study this problem in the context of surrogate models for time-dependent physical systems described by partial differential equations. We prove that in a function space setting, distributions at arbitrarily close times can be mutually singular, making exact coverage guarantees impossible. We further show that in discretized settings, the total variation distance between distributions often saturates exponentially fast with the grid resolution, mandating restrictions on the spatial discretization to ensure coverage. Finally, for certain se
    
[^3]: 统一中心流、杆流与边缘耦合

    Unifying Central Flow, Rod Flow, and the Edge Coupling

    [https://arxiv.org/abs/2611.08358](https://arxiv.org/abs/2611.08358)

    本文提出“边缘耦合”这一泛函，证明中心流与杆流均可由它推导得出，二者分别对应不同时间尺度上的时间粗粒化，从而统一了刻画梯度下降稳定性边缘行为的两种连续时间模型。

    

    在稳定性边缘，梯度下降的迭代点会沿着Hessian矩阵的尖锐方向发生振荡。这对连续时间模型构成了重大挑战，因为通常的连续时间近似——梯度流——并不允许振荡。已有若干连续时间模型被提出来刻画稳定性边缘处的行为，其中包括中心流和杆流。尽管这两种模型都可以被解释为在时间上执行粗粒化，但它们之间的联系仍然知之甚少。本文证明，这两种流都可以从第三个对象——边缘耦合——推导出来：边缘耦合是定义在相邻迭代点上的一个泛函，其临界点正是梯度下降的不动点和周期二轨道。研究揭示，这两种流对应于不同时间尺度上的粗粒化：杆流仅在最快的时间尺度上进行粗粒化，而中心流则在快……

    arXiv:2611.08358v1 Announce Type: cross  Abstract: At the edge of stability, the iterates of gradient descent oscillate along the sharp directions of the Hessian. This poses a significant challenge for continuous-time models as the usual continuous-time approximation -- the gradient flow -- admits no oscillations. Several continuous-time models have been proposed to capture the behavior at the edge of stability, among them central flow and rod flow. Although both models can be interpreted as performing a coarse-graining in time, the connection between them remains poorly understood. Here we show that both flows can be derived from a third object, the edge coupling: a functional on consecutive iterates whose critical points are the fixed points and period-two orbits of gradient descent. It is revealed that the two flows correspond to coarse-grainings at different time scales. Rod flow coarse-grains over only the fastest time scale, while central flow coarse-grains over both the fast and
    
[^4]: 面向全局时间序列预测的序列级交叉验证

    Series-wise Cross-validation for Global Time Series Forecasting

    [https://arxiv.org/abs/2611.08252](https://arxiv.org/abs/2611.08252)

    该论文为基于深度学习的全局时间序列预测模型形式化了按序列划分的交叉验证方法，突破传统仅按时间划分的范式，并在11个数据集和10种神经网络架构上系统评估了K折、自助法等多种序列级交叉验证策略。

    

    我们研究时间序列预测中的交叉验证（CV）问题，重点关注基于深度学习的全局预测模型。预测领域主流的交叉验证策略是按时间划分数据，将每个序列的最后部分用于测试。然而，对于在时间序列集合上训练的全局模型而言，一种根本不同的方法成为可能：按序列和时间同时进行划分。我们为全局预测模型形式化了序列级交叉验证，并在11个数据集和10种神经网络架构上对多种实现方式进行了基准测试，包括K折交叉验证、自助法和蒙特卡洛交叉验证。我们从五个维度评估交叉验证程序：性能估计、过度乐观率、模型选择准确性、遗憾值以及下游预测性能。我们从训练数据中随机抽取一部分序列予以保留，使其仅出现在测试集中。该协议模拟了新出现序列尚未……（摘要在此处截断）

    arXiv:2611.08252v1 Announce Type: cross  Abstract: We consider the problem of cross-validation (CV) in time series forecasting, with a focus on global forecasting models based on deep learning. The prevailing CV strategy in forecasting is to partition data across time with the last part of each series used for testing. With global models trained on collections of time series, however, a fundamentally different approach becomes possible: partitioning across series and time. We formalize series-wise CV for global forecasting models and benchmark several instantiations, including K-fold CV, bootstrapping, and Monte Carlo CV, on 11 datasets with 10 neural network architectures. We evaluate CV procedures along five dimensions: performance estimation, overoptimism rate, model selection accuracy, regret, and downstream forecasting performance. We withhold a random fraction of series from training so that they appear only in the test set. This protocol simulates the emergence of new series not
    
[^5]: BUMP：短暂意外的插值中段峰值

    BUMP: Brief Unexpected Mid-Interpolation Peak

    [https://arxiv.org/abs/2611.08182](https://arxiv.org/abs/2611.08182)

    本文发现了神经网络插值中的 BUMP 现象（测试精度在插值端点附近短暂提升后向中点下降），并基于此提出了将权重混合与预测空间混合相结合的新型集成方法。

    

    我们在两个置换对齐的神经网络之间进行线性插值时发现了一个有趣的现象：测试精度在插值端点附近会短暂提升，随后向中点方向下降，我们将这一现象称为 BUMP（Brief Unexpected Mid-interpolation Peak，短暂意外的插值中段峰值）。我们观察到，BUMP 在不同数据集、不同架构和不同训练方案下均一致出现，并且在早期检查点和极端剪枝等剧烈扰动下依然保持稳健。此外，受控消融实验表明，该现象无法归因于简单的正则化效应。尽管类似的近端点增益在以往的插值曲线图中有所显现，但此前并未被明确记录，也未得到系统性、深入的分析。利用这一效应，我们提出了一种简单而新颖的集成方案：权重-预测混合集成，它将权重空间的混合与预测空间的混合相结合。在我们的全部实验中，所得的集成方法始终取得一致的表现。

    arXiv:2611.08182v1 Announce Type: new  Abstract: We identify an intriguing phenomenon when linearly interpolating between permutation-aligned neural networks: test accuracy briefly improves near the interpolation endpoints before dropping toward the midpoint, which we term BUMP (Brief Unexpected Mid-interpolation Peak). We observe that BUMP appears consistently across datasets, architectures, training regimes, and remains robust under substantial perturbations such as early checkpoints and extreme pruning. Moreover, controlled ablations suggest that it cannot be attributed to trivial regularization effects. While similar near-endpoint gains are visible in prior interpolation plots, they have not been explicitly documented or systematically and thoroughly analyzed. Leveraging this effect, we propose a simple, novel ensembling scheme: weight-prediction mixed ensembling that combines weight mixing with prediction-space mixing. Across our experiments, the resulting ensembling method consis
    
[^6]: 明智的补丁注意力：理解基于注意力的预测中的补丁长度

    Patch Attention Wisely: Understanding Patch Length in Attention-Based Forecasting

    [https://arxiv.org/abs/2611.08160](https://arxiv.org/abs/2611.08160)

    该论文提出了一个理论框架，解释了基于注意力的时间序列预测中补丁长度为何影响预测精度，并基于线性状态空间模型的精确分解，推导出在时间相关性下平衡表示误差与估计误差的补丁选择准则。

    

    基于注意力机制的时间序列预测模型通常将连续的观测值分组为补丁。补丁长度对预测性能有显著影响，但其选择在很大程度上仍依赖经验。我们开发了一个理论框架，解释了补丁长度为何对预测精度至关重要，并为其选择提供了有原则的依据。对于线性状态空间模型，我们推导出了预测系数矩阵的精确分解，将预测动态与滤波动态分离开来。我们证明，在长预测时域下单一预测分量占主导地位通常并不意味着秩一表示。这解释了为什么仅凭较长的预测时域并不一定会降低表示复杂度。我们将该分解与估计误差界相结合，开发了一种补丁选择准则，能够在时间相关性存在的情况下平衡表示误差与估计误差。我们的分析建立了在这些条件下……

    arXiv:2611.08160v1 Announce Type: cross  Abstract: Attention-based time series forecasters commonly group consecutive observations into patches. Patch length is known to have a substantial impact on forecasting performance, yet its selection remains largely empirical. We develop a theoretical framework that explains why patch length matters for forecasting accuracy and provides a principled basis for its selection. For linear state-space models, we derive an exact factorization of the forecasting coefficient matrix that separates forecast dynamics from filtering dynamics. We show that dominance of a single forecast component at long horizons does not generally imply a rank-one representation. This explains why longer horizons alone need not reduce representational complexity. We combine the factorization with estimation bounds to develop a patch-selection criterion that balances representation and estimation errors under temporal dependence. Our analysis establishes conditions under wh
    
[^7]: 使用非线性干预状态空间模型识别神经动力学

    Identifying Neural Dynamics Using Nonlinear Interventional State Space Models

    [https://arxiv.org/abs/2611.08113](https://arxiv.org/abs/2611.08113)

    该研究将干预状态空间模型的可识别性理论从线性动力学推广到一般非线性潜在动力学与单射发射函数，为识别神经回路动力学提供了更广泛的理论保证。

    

    状态空间模型（SSMs）能够从高维观测数据中恢复低维潜在动力学，但仅凭观测数据通常无法识别这些潜在变量。干预状态空间模型利用扰动打破观测对称性，从而在允许置换、缩放和平移的等价意义下识别模型参数。然而，先前的工作将可识别性限制在线性潜在动力学和分段线性发射函数，排除了神经回路中许多典型的行为特征。我们将iSSMs扩展到更广泛的模型类别：我们证明（i）在一般的单射发射函数和高斯潜在转移下，仿射可识别性成立；（ii）在有干预的情况下，在一般的非线性潜在动力学下，直到置换、缩放和平移的可识别性仍然保持。我们在模拟的FitzHugh-Nagumo动力学上验证了该理论，随后展示了两个应用：在相互竞争的模型之间进行裁决……（原文摘要在此处截断）

    arXiv:2611.08113v1 Announce Type: cross  Abstract: State-space models (SSMs) recover low-dimensional latent dynamics from high-dimensional observations, but latents are generally not identifiable from observational data alone. Interventional state-space models (iSSMs) use perturbations to break observational symmetries and identify parameters up to permutation, scaling, and shift. However, prior work restricts identifiability to linear latent dynamics and piecewise-linear emissions, excluding many behaviors characteristic of neural circuits. We extend iSSMs to a substantially broader model class: we prove that (i) affine identifiability holds under general injective emission functions and Gaussian latent transitions, and (ii) with interventions, identifiability up to permutation, scaling, and shift is preserved under general nonlinear latent dynamics. We validate our theory on simulated FitzHugh-Nagumo dynamics, and then demonstrate two applications: adjudicating between competing mode
    
[^8]: 基于循环神经网络的多值治疗方案双重稳健因果推断

    Doubly-robust causal inference for multi-valued treatment regimes with recurrent neural networks

    [https://arxiv.org/abs/2611.08077](https://arxiv.org/abs/2611.08077)

    该论文将循环神经网络融入扩展至多值治疗的纵向目标最小损失估计（LTMLE）框架，实现了对多值治疗方案因果效应的双重稳健估计，模拟结果显示该方法近似无偏且覆盖率接近名义水平。

    

    我们提出了一种双重稳健方法，用于估计多值治疗方案对患者结局的因果效应。多值治疗方案根据患者的病史从一组相互竞争的选项中为其分配个性化治疗。现有的最优治疗方法大多局限于二分类或连续型治疗。我们通过使用多项模型对治疗分配进行建模，将纵向目标最小损失估计器（LTMLE）扩展以适应多值治疗。我们进一步将循环神经网络（RNN）整合到LTMLE框架中，作为结局模型、治疗模型和删失模型的学习器，并将该方法与集成机器学习进行比较。模拟研究表明，无论采用集成机器学习还是RNN，LTMLE均近似无偏且覆盖率接近名义水平，而使用集成机器学习的顺序g-计算则存在显著偏差。

    arXiv:2611.08077v1 Announce Type: cross  Abstract: We propose a doubly-robust method to estimate the causal effects of multi-valued treatment regimes on patient outcomes. A multi-valued treatment regime assigns personalized treatments from a set of competing options based on a patient's history. Existing methods for optimal treatment regimes are largely limited to binary or continuous treatments. We extend the longitudinal targeted minimum loss-based estimator (LTMLE) to accommodate multi-valued treatments by modeling treatment assignments with a multinomial model. We further integrate recurrent neural networks (RNNs) into the LTMLE framework as learners for the outcome, treatment, and censoring models, and compare this approach to ensemble machine learning. Simulation studies show that LTMLE with either ensemble machine learning or RNNs is approximately unbiased, with close to nominal coverage, whereas sequential g-computation with ensemble machine learning is substantially biased und
    
[^9]: 多水平函数数据中条件均值与变异性的联合建模：在AEGIS研究餐后血糖轨迹中的应用

    Joint Modeling of Conditional Mean and Variability in Multilevel Functional Data: Application to Postprandial Glucose Trajectories in the AEGIS Study

    [https://arxiv.org/abs/2611.07937](https://arxiv.org/abs/2611.07937)

    该论文提出了一种异方差函数混合效应回归框架，联合建模多水平重复函数轨迹的条件均值与方差，并通过参与者层面的自助法量化参数不确定性，应用于AEGIS研究揭示了各因素对餐后血糖均值与变异性的时依赖差异化影响。

    

    函数数据分析为研究协变量与高频生物信号之间的时变关联提供了框架。在数字健康领域，参与者通常会贡献重复的函数轨迹，然而现有的多水平回归方法主要关注条件均值，忽视了变异性及其在生物异质性中的作用。我们提出了一种异方差函数混合效应回归框架，联合建模重复轨迹的条件均值与方差，并在两个成分中都引入了参与者特定的随机效应。我们还开发了一种参与者层面的自助法（bootstrap）来量化估计的函数参数的不确定性。我们将该方法应用于AEGIS研究中的餐后血糖轨迹数据。结果显示，餐前血糖、碳水化合物摄入量以及餐食类型与血糖变异性呈现出明显不同的时依赖关联，而HbA1c则与条件均值关联更强。

    arXiv:2611.07937v1 Announce Type: cross  Abstract: Functional data analysis provides a framework for studying time-varying associations between covariates and high-frequency biological signals. In digital health, participants often contribute repeated functional trajectories, yet existing multilevel regression methods largely focus on the conditional mean, overlooking variability and its role in biological heterogeneity. We propose a heteroskedastic functional mixed-effects regression framework that jointly models the conditional mean and variance of repeated trajectories, with participant-specific random effects in both components. We also develop a participant-level bootstrap to quantify uncertainty in estimated functional parameters. We apply the method to postprandial glucose trajectories from the AEGIS study. Premeal glucose, carbohydrate intake, and meal type show distinct time-dependent associations with glucose variability, whereas HbA1c is more strongly associated with the con
    
[^10]: 网络效应分解：用于网络干扰下的因果效应估计

    Network Effect Decomposition for Causal Effect Estimation under Network Interference

    [https://arxiv.org/abs/2611.07879](https://arxiv.org/abs/2611.07879)

    提出NEDNet网络，将网络干扰下的结果响应分解为查询无关的处理前基线以及相互独立的处理响应通路和暴露响应通路，从而在观测网络数据上实现更准确的因果效应估计。

    

    从观测网络数据中估计因果效应，需要同时考虑个体的处理和邻域暴露。现有的结果模型通常将这些干预变量合并到一个联合预测器中，并通过预测对比来估计因果效应，而相应的响应成分则是隐式的。我们提出了网络效应分解网络（NEDNet），它将条件结果响应参数化为一个与查询无关的处理前基线，加上相互独立的处理响应通路和暴露响应通路。路径特定的邻域上下文化机制为这两个通路提供局部的处理前信息。我们通过一个唯一的、参考归一化的分解来刻画拟合的响应，该分解等价于一个零混合处理-暴露对比。在基于BlogCatalog等数据集构建的半合成基准上的实验（原文在此处截断）……

    arXiv:2611.07879v1 Announce Type: cross  Abstract: Estimating causal effects from observational network data requires accounting for both individual treatment and neighborhood exposure. Existing outcome models commonly incorporate these intervention variables into a joint predictor and estimate causal effects through prediction contrasts, leaving the corresponding response components implicit. We propose the $\underline{N}$etwork $\underline{E}$ffect $\underline{D}$ecomposition $\underline{Net}$work (NEDNet), which parameterizes the conditional outcome response as a query-invariant pre-treatment baseline plus separate treatment- and exposure-response pathways. Path-specific neighborhood contextualization supplies local pre-treatment information to the two pathways. We characterize the fitted response through a unique reference-normalized decomposition, which is equivalent to a zero mixed treatment--exposure contrast. Experiments on semi-synthetic benchmarks derived from BlogCatalog and
    
[^11]: 具有任意延迟的博弈中的最优学习：自博弈与对抗对手场景的两全其美

    Optimal Learning in Games with Arbitrary Delays: Best-of-Both-Worlds for Self-Play and Adversarial Opponents

    [https://arxiv.org/abs/2611.07773](https://arxiv.org/abs/2611.07773)

    提出单一算法MAO-OMD，在具有任意未知反馈延迟的零和博弈中，同时实现了自博弈时的最优快速遗憾速率 O(σ_max+1) 和对抗任意对手时极小极大最优的 O(√(T+d_tot)) 保证。

    

    我们研究了具有延迟反馈的零和博弈中的学习问题。我们提出了多锚点乐观在线镜像下降算法，这是一种单一算法，能够在光滑博弈的自博弈中获得最优的快速遗憾速率，同时保持对任意对手的极小极大最优保证。特别地，在具有任意且未知反馈延迟的光滑零和博弈中，当双方参与者都使用MAO-OMD时，该算法获得 O(σ_max+1) 的社会遗憾和个体遗憾，其中 σ_max 表示未完成梯度的最大数量；同时对任意对手保证极小极大最优的 O(√(T+d_tot)) 个体遗憾，其中 d_tot 为总延迟。此外，我们证明了通过将问题标准黑盒归约到无延迟设定，无法同时获得这些最优速率。我们还将结果扩展到多人博弈，以及具有参与者相关延迟的博弈。

    arXiv:2611.07773v1 Announce Type: new  Abstract: We study learning in zero-sum games with delayed feedback. We introduce Multi-Anchor Optimistic Online Mirror Descent (MAO-OMD), a single algorithm which obtains optimal fast rates in self-play on smooth games, while simultaneously maintaining minimax optimal guarantees against arbitrary opponents. In particular, in smooth zero-sum games with arbitrary and unknown feedback delays, MAO-OMD obtains $O(\sigma_{\max}+1)$ social and individual regrets when used by both players, where $\sigma_{\max}$ denotes the maximum number of outstanding gradients, while also guaranteeing minimax optimal $O(\sqrt{T+d_{\mathrm{tot}}})$ individual regret against arbitrary opponents, where $d_{\mathrm{tot}}$ is the total delay. Moreover, we show that these optimal rates cannot be obtained simultaneously using the standard black-box reduction to the undelayed setting. We also extend our results to multi-player games, and to games with player-dependent delay se
    
[^12]: 潜在结果联合期望的识别与界定

    Identification and Bounding of Joint Expectation over Potential Outcomes

    [https://arxiv.org/abs/2611.07536](https://arxiv.org/abs/2611.07536)

    本文提出同单调性与反单调性两种新假设，为潜在结果函数联合期望的识别提供了适用于离散、连续和混合结果变量的统一框架，并在无这些假设时推导出尖锐界限，为衡量因果效应异质性提供了新的估计工具。

    

    潜在结果 $(Y_1,Y_0)$ 的函数的联合期望是因果推断中的一个基本量，特别是在研究因果效应的变化或异质性时。本文为其识别提出了新颖的假设：同单调性和反单调性。这些假设为离散、连续以及混合结果变量的识别提供了统一的框架。在这些假设不成立的情况下，我们推导了两类广泛函数的尖锐界限，为个体因果效应矩的界定给出了新结果，而个体因果效应对于衡量效应异质性至关重要。我们提出了相应的估计方法，并通过模拟研究和真实数据集进行了展示。

    arXiv:2611.07536v1 Announce Type: cross  Abstract: The joint expectation of functions of potential outcomes $(Y_1,Y_0)$ is a fundamental quantity in causal inference, particularly when investigating the variation or heterogeneity of causal effects. This paper provides novel assumptions for their identification: comonotonicity and countercomonotonicity. These assumptions offer a unified framework for their identification across discrete, continuous, and mixed outcome variables. In the absence of these assumptions, we derive sharp bounds for two broad classes of functions, yielding new results for bounding moments of individual causal effects, which are central to measuring effect heterogeneity. We present corresponding estimation methods and illustrate them in simulation studies and real-world datasets.
    
[^13]: 基于嵌套可交换性检验的变点定位：从点估计到置信区间

    Changepoint localization via nested exchangeability tests: from point estimates to confidence intervals

    [https://arxiv.org/abs/2611.07399](https://arxiv.org/abs/2611.07399)

    该论文提出CLONE方法，通过在以变点估计值为中心的嵌套窗口中进行可交换性检验，为任意黑盒变点估计器构建具有有限样本覆盖保证的无分布置信区间，且仅需O(log n)次置换检验。

    

    我们研究了离线环境下变点估计的无分布不确定性量化问题。我们观察到一个具有单一变点的序列，并已获得其位置的估计值。变化前后的数据段相互独立，且各自由来自任意未知分布的独立同分布（或更一般地，可交换）观测值组成。我们提出了CLONE（Changepoint LOcalization via Nested Exchangeability testing，基于嵌套可交换性检验的变点定位），这是一种新颖的方法，通过在以参考估计值为中心的嵌套窗口中进行可交换性检验，围绕该估计值构建无分布置信区间。在这些假设下，CLONE能够为任何黑盒参考估计器提供有限样本覆盖保证。借助几何递减的半径网格，CLONE仅需O(log n)次置换检验。我们刻画了该框架下的最优得分，并给出了实用建议，从而获得紧凑的置信区间。

    arXiv:2611.07399v1 Announce Type: cross  Abstract: We study distribution-free uncertainty quantification for a changepoint estimate in an offline setting. We observe a sequence with a single changepoint and are given an estimate of its location. The pre- and post-change segments are independent and each consists of i.i.d. (or, more generally, exchangeable) observations from arbitrary and unknown distributions. We introduce Changepoint LOcalization via Nested Exchangeability testing (CLONE), a novel method for constructing distribution-free confidence intervals around that estimate by testing exchangeability in nested windows centered at the reference estimate. Under these assumptions, CLONE provides finite-sample coverage for any black-box reference estimator. Using a geometrically shrinking radius grid, CLONE requires only $O(log n)$ permutation tests. We characterize optimal scores for the framework and provide practical recommendations that result in sharp confidence intervals. Unde
    
[^14]: 无界问题下随机梯度方法的末次迭代收敛性

    Last-Iterate Convergence of Stochastic Gradient Methods for Unbounded Problems

    [https://arxiv.org/abs/2611.07334](https://arxiv.org/abs/2611.07334)

    本文在去除了有界定义域和有界方差假设的条件下，证明了随机梯度方法的末次迭代具有接近最优的 $\widetilde{\mathcal{O}}(1/\sqrt{T})$ 收敛率，并在迭代总次数已知时达到了最优的 $\mathcal{O}(1/\sqrt{T})$ 阶，与强假设下的已知结果相匹配。

    

    随机梯度方法的末次迭代收敛保证近来受到广泛关注，因为在实际应用中算法通常返回最后一次迭代的结果，而非所有迭代的平均值。本文研究了两种分析近端（proximal）和投影（projected）随机梯度方法末次迭代收敛性的途径，且无需有界定义域或有界方差假设，其中黑箱变分推断作为贯穿全文的应用实例。在两个相关的、放宽有界方差要求的假设条件下，我们证明了末次迭代具有接近最优的任意时刻收敛率 $\widetilde{\mathcal{O}}(1/\sqrt{T})$。此外，当总迭代次数 $T$ 已知时，我们进一步去除了对数因子，得到了最优阶 $\mathcal{O}(1/\sqrt{T})$，与有界方差或有界定义域假设下已知的收敛阶相匹配。

    arXiv:2611.07334v1 Announce Type: cross  Abstract: Last-iterate convergence guarantees for stochastic gradient methods have attracted significant attention as practical implementations typically return the last iterate instead of an average of the iterates. In this paper, we study two approaches for analyzing the last-iterate convergence of proximal and projected stochastic gradient methods without bounded domain or bounded variance assumptions, where black-box variational inference serves as a running example. Under two related assumptions relaxing the bounded variance, we show near-optimal anytime rate $\widetilde{\mathcal{O}}(1/\sqrt{T})$ for the last iterate. Moreover, when the horizon $T$ is known, we remove the logarithmic factor and obtain the optimal order $\mathcal{O}(1/\sqrt{T})$, matching the order known under bounded variance or bounded domain assumptions.
    
[^15]: 神经实现测度：面向波动率预测的预测监督型日内聚合方法

    Neural Realized Measures: Forecast-Supervised Intraday Aggregation for Volatility Forecasting

    [https://arxiv.org/abs/2611.07287](https://arxiv.org/abs/2611.07287)

    该论文提出神经实现测度（NRM），一种以波动率预测为监督、通过可学习的正的上下文相关权重来聚合日内观测的框架，将经典实现测度嵌套为特例，从而改进波动率预测。

    

    高频波动率预测通常将测量与预测两个环节分离：日内观测首先通过固定的聚合规则被压缩为日度实现测度，随后所得统计量再被传入预测模型。因此，这种传统流程将测度构建视为与预测无关的预处理步骤。我们提出这样的问题：聚合阶段本身能否基于下游预测目标进行学习，同时保留既有实现测度的统计结构。我们提出神经实现测度（NRM），这是一个保持原语、以预测为监督的聚合框架，它保留了原始的测度特定日内贡献，但以正的、均值归一化且依赖上下文的权重取代了均匀聚合。当使用均匀权重时，可以精确恢复相应的原始实现测度（RRM），因此经典构造被嵌套为该框架的特例。

    arXiv:2611.07287v1 Announce Type: new  Abstract: High-frequency volatility forecasting typically separates measurement from prediction: intraday observations are first compressed into a daily realized measure using a fixed aggregation rule, and the resulting statistic is then passed to a forecasting model. This conventional pipeline therefore treats measure construction as forecast-agnostic preprocessing. We ask whether the aggregation stage itself can instead be learned from the downstream forecasting objective while preserving the statistical structure of established realized measures. We propose Neural Realized Measures (NRM), a primitive-preserving, forecast-supervised aggregation framework that retains the original measure-specific intraday contributions but replaces uniform aggregation with positive, mean-normalized context-dependent weights. Uniform weights exactly recover the corresponding raw realized measure (RRM), so the classical construction is nested as a special case. Be
    
[^16]: 满足化作为竞争性贝叶斯加权：未达成目标下目标导向选择的最小模型

    Satisficing as competitive Bayesian weighting: a minimal model of goal-directed choice under an unattained target

    [https://arxiv.org/abs/2611.07270](https://arxiv.org/abs/2611.07270)

    该论文证明“风险敏感满足化”这一多臂老虎机学习规则可等价改写为类贝叶斯估计器“注意力竞争贝叶斯（ACB）”——未达成的抱负水平充当所有动作的先验均值，有限试验预算决定证据覆盖先验的强度，从而给出了以未实现目标为参照的目标导向选择的最小形式化模型。

    

    动物和人类往往并非进行最大化决策；它们相对于尚未实现的目标（抱负水平）而行动。一个尚未实现的目标如何组织选择行为？Deacon（2012）将这种以“缺失”为参照的组织方式称为“意向性的”（ententional），而 García-Valdecasas 与 Deacon（2024）则主张，目的最好被理解为约束而非特定状态。我们在一个最小化的形式化环境中考察这一思想：风险敏感满足化（RS），一种多臂老虎机学习规则，它以每个动作过去被试行的份额乘以其相对于抱负水平的盈余或缺口来评估该动作的价值。我们证明 RS 可以被改写为一个类贝叶斯估计器——注意力竞争贝叶斯（ACB），其中抱负水平充当每个动作的先验均值，而有限的试验预算决定了证据能在多大程度上覆盖这一先验。由此得出两个结论。第一，不再被采样的动作会被部分覆盖……（原文摘要至此截断）

    arXiv:2611.07270v1 Announce Type: cross  Abstract: Animals and humans often do not maximize; they act relative to goals (aspiration levels) that are frequently not yet met. How can a goal that is not (yet) realized organize choice? Deacon (2012) calls such absence-referenced organization \emph{ententional}, and Garc\'{i}a-Valdecasas and Deacon (2024) argue that ends are best understood as constraints rather than as particular states. We examine this idea in a minimal formal setting: Risk-sensitive Satisficing (RS), a bandit learning rule that values each action by its share of past trials multiplied by its surplus or shortfall relative to an aspiration level. We show that RS can be rewritten as a Bayesian-like estimator, Attention Competitive Bayes (ACB), in which the aspiration level is the prior mean of every action and the finite budget of trials sets how strongly evidence can override it. Two consequences follow. First, actions that are no longer sampled are partially overridden by
    
[^17]: 面板数据的双重机器学习：识别、模型设定与银行监管效应

    Double Machine Learning for Panel Data: Identification, Specification, and a Bank Regulation Effect

    [https://arxiv.org/abs/2611.07249](https://arxiv.org/abs/2611.07249)

    本文将双重/去偏机器学习拓展至面板数据，刻画了共同处理时点下“组内”与“一阶差分”估计量的识别条件，并以2018年美国银行监管改革为例，揭示了模型设定选择与函数形式稳健性对因果识别和估计结果的影响。

    

    本文利用双重/去偏机器学习考察面板估计的稳健性与因果解释。原型处理设定为2018年美国银行监管改革。在共同处理时点下，本文刻画了识别与冗余参数限制条件，并推导出“组内”与“一阶差分”得分在银行层面的精确表示。模拟实验评估了面板模型变换引起的设定偏差以及在较弱重叠条件下的推断表现。实证结果表明，估计值随资产规模分类与前期增长情况而变化；对前期增长进行调整会降低四种机器学习方法下的重叠加权估计值，而双重稳健估计依赖于拟合的倾向得分和集中的权重。由于两类银行均获得了监管放松，本文进一步估计了差异化的暴露效应。研究发现突出了模型设定问题以及函数形式稳健性与因果识别之间的关系。

    arXiv:2611.07249v1 Announce Type: cross  Abstract: This paper examines robustness and causal interpretation of panel estimates using double/debiased machine learning. The prototype treatment setup is the 2018 US banking regulation. Under common treatment timing, we characterize identification and nuisance restrictions and derive exact bank-level representations of specified "within" and "first-difference" scores. Simulations assess panel model transformation-induced specification bias and inference under weaker overlap. Empirically, estimates vary with asset-size classification and prior growth. Prior-growth adjustment reduces overlap-weighted estimates across four Machine Learners (ML); doubly robust estimates depend on the fitted propensity and concentrated weights. Both bank groups received relief, so we estimate differential exposure effects. The findings highlight specification issues and the relation between functional-form robustness and causal identification.
    
[^18]: 基于影响性误差消除的任务校准随机特征

    Task-Calibrated Random Features via Influence-Based Error Cancellation

    [https://arxiv.org/abs/2611.07209](https://arxiv.org/abs/2611.07209)

    该论文提出任务校准随机特征（TCRF），通过基于任务特定影响构建的可重用校准空间来平衡频率矩并消除对下游任务有影响的采样误差，从而改进核岭回归和核Stein差异等核方法的近似质量。

    

    随机特征通过蒙特卡洛方法近似核积分，然而只有特定的频率采样误差会影响下游任务。我们提出了任务校准随机特征（TCRF），它利用在核条目之间共享的权重，来平衡由任务特定影响构建的频率矩。闭式解权重与经典的最小距离校准方法相一致；其贡献在于提供了一个可重用的、由任务导出的校准空间。我们证明了第一阶方差等于投影到该空间后的残差方差。对于核岭回归，完全消除会产生非高斯的二阶极限。对于核Stein差异，子集恒等式和不交块方差分解消除了试点噪声的线性分量。正的指数倾斜权重保持了半正定的特征表示，并支持更高的可行校准秩。配对实验显示了改进的近似

    arXiv:2611.07209v1 Announce Type: cross  Abstract: Random features approximate kernel integrals by Monte Carlo, yet only selected frequency-sampling errors affect a downstream task. We propose Task-Calibrated Random Features (TCRF), which balance frequency moments constructed from task-specific influence using weights shared across kernel entries. Closed-form weights coincide with classical minimum-distance calibration; the contribution is a reusable, task-derived calibration space. We show that first-order variance equals the residual variance after projection onto this space. For kernel ridge regression, complete cancellation yields a non-Gaussian second-order limit. For kernel Stein discrepancy, a subset identity and disjoint-block variance decomposition eliminate the linear component of pilot noise. Positive exponential-tilting weights preserve the positive-semidefinite feature representation and support higher feasible calibration ranks. Paired experiments show improved approximat
    
[^19]: ShapleyOG：基于奇数高斯过程的快速自适应Shapley值估计

    ShapleyOG: Fast Adaptive Shapley Value Estimation with Odd Gaussian Processes

    [https://arxiv.org/abs/2611.07173](https://arxiv.org/abs/2611.07173)

    ShapleyOG利用Shapley值仅依赖于奇数分量的特性，提出了一种基于奇数高斯过程的快速自适应估计方法，克服了ShaplEIG等现有自适应方法因每轮迭代重复拟合高斯过程代理模型而带来的计算开销问题。

    

    Shapley值是机器学习中进行归因分析的核心工具。由于精确计算Shapley值需要在指数级数量的玩家联盟上评估被解释的合作博弈，因此通常基于有限次数的评估来进行估计。最近的研究（ShaplEIG）表明，自适应地选择要评估的联盟能够显著提高相对于经典估计器的样本效率。然而，这种优势伴随着一个顺序化流程的代价：在每次迭代中，都需要重新拟合博弈的高斯过程代理模型，并选择对Shapley值具有最大信息增益期望的联盟。由此产生的计算开销使其仅适用于评估预算较低的高成本博弈。我们提出了基于奇数高斯过程的快速自适应Shapley值估计方法（ShapleyOG）以克服这一限制。ShapleyOG利用了Shapley值仅依赖于奇数分量这一事实（摘要在此处截断）。

    arXiv:2611.07173v1 Announce Type: cross  Abstract: Shapley values are a central tool for attribution in machine learning. Since their exact computation requires evaluating the cooperative game to be explained on exponentially many player coalitions, they are typically estimated from a limited number of evaluations. Recently, ShaplEIG has shown that adaptively selecting the coalitions to evaluate substantially improves sample efficiency over classical estimators. This, however, comes at the cost of a sequential procedure that, in every iteration, refits a Gaussian process surrogate of the game and selects the coalition with the largest expected information gain about the Shapley values. The resulting computational overhead limits its applicability to costly games with low evaluation budgets. We present fast adaptive Shapley value estimation with Odd Gaussian Processes (ShapleyOG) to overcome this limitation. ShapleyOG leverages the fact that Shapley values depend only on the odd compone
    
[^20]: 具有自动模型复杂度推断的贝叶斯张量链核机器

    Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference

    [https://arxiv.org/abs/2611.07129](https://arxiv.org/abs/2611.07129)

    本文提出了贝叶斯张量链核机器（BTT-KM），通过在TT核心上施加分层ARD先验并结合平均场变分推断，实现了从数据中自动推断TT秩与特征维度的闭式贝叶斯学习，且计算复杂度与确定性方法相同。

    

    张量网络（TN）核机器通过使用低秩张量网络来降低计算量和内存需求，从而加速监督学习。然而，大多数基于张量网络的方法是确定性的，其秩和特征维度需要手动调整，而现有的贝叶斯张量网络核机器仅限于CANDECOMP/PARAFAC分解（CPD），其单一秩由所有模式共享。我们将该框架扩展到张量链（TT）分解，提出了贝叶斯张量链核机器（BTT-KM），其在TT核心上的分层ARD类型先验可以从数据中自动推断TT秩和特征维度。由于每个TT核心仅通过其相邻核心与链上其他核心耦合，其闭式后验更新需要沿链传播所有其他核心的贡献。平均场变分推断产生了一个贝叶斯交替最小二乘算法，其计算复杂度与确定性对应算法相同，并具有闭式预测分布。

    arXiv:2611.07129v1 Announce Type: cross  Abstract: Tensor Network (TN) kernel machines accelerate supervised learning by using low-rank TNs to reduce computation and memory. However, most TN-based methods are deterministic with manually tuned ranks and feature dimensions, while the existing Bayesian TN kernel machine is restricted to the CANDECOMP/PARAFAC decomposition (CPD), whose single rank is shared by all modes. We extend this framework to the tensor train (TT) and propose Bayesian Tensor Train Kernel Machines (BTT-KM), whose hierarchical ARD-type priors on the TT cores infer the TT ranks and feature dimensions from data. Since each TT core is coupled to the others only through its neighbors, its closed-form posterior update requires propagating the contributions of all other cores along the chain. Mean-field variational inference yields a Bayesian alternating least squares algorithm with the same computational complexity as its deterministic counterpart and a closed-form predicti
    
[^21]: ToroFit：用环面的楔和拟合高维数据

    ToroFit: Fitting Wedge Sums of Tori to High-Dimensional Data

    [https://arxiv.org/abs/2611.06990](https://arxiv.org/abs/2611.06990)

    本文提出ToroFit框架，利用持续上同调与上积识别高维点云数据中各环面结构分量，并通过将低能量多元三角多项式拟合归结为广义岭回归，实现对环面楔和型非线性数据的高效检测与参数化。

    

    科学应用中产生的数据往往呈现出复杂的非线性结构。识别并参数化这些非线性结构可以改进建模，增强预测分析、数据插值、去噪和异常检测能力。在本文中，我们开发了理论与计算方法，用于检测并参数化位于环面楔和附近的欧氏点云数据。具体而言，我们利用持续上同调和上积来识别不同的环面分量，并在每个分量上通过优化低能量多元三角多项式来获得最佳数据拟合。由此产生的优化问题可归结为广义岭回归，可借助标准技术高效求解。我们在合成数据集和真实数据集上展示了所提框架的实用性，包括低帧率循环视频的插值、准周期动力系统中吸引子的恢复等。

    arXiv:2611.06990v1 Announce Type: cross  Abstract: Data arising in scientific applications often exhibits complex nonlinear structure. Identifying and parameterizing said nonlinearities can enable better modeling, enhance predictive analysis, data interpolation, denoising, and anomaly detection. In this paper, we develop theoretical and computational methods to detect and parametrize Euclidean point cloud data lying near a wedge sum of tori. Specifically, we use persistent cohomology and cup products to identify the distinct toroidal components, and on each component, we optimize over low-energy multivariate trigonometric polynomials for best data fit. The ensuing optimization problem reduces to generalized ridge regression, which can be solved efficiently via standard techniques. We demonstrate the utility of the proposed framework in both synthetic and real datasets, including interpolation of low frame-rate recurrent videos, attractor recovery in quasiperiodic dynamical systems, and
    
[^22]: 黎曼流形上积分曲线的抽样设计与置信管

    Sampling Design and Confidence Tubes for Integral Curves on Riemannian Manifolds

    [https://arxiv.org/abs/2611.00771](https://arxiv.org/abs/2611.00771)

    本文在紧黎曼流形上提出了估计未知向量场积分曲线的被动与主动两种抽样方法，证明了被动核估计的曲线误差阶为 $n^{-2/(d+3)}$ 并给出由线性随机微分方程刻画的函数型中心极限定理，同时证明基于先导信息的主动抽样可将期望最大曲线误差提升至 $N^{-1/2}$ 阶，从而为构建置信管奠定基础。

    

    我们在已知紧黎曼流形上，利用带噪声的切向量场来估计未知向量场的积分曲线。我们考虑两种抽样设置：一种是在已收集的随机位置上的独立观测，另一种是在固定响应预算下，允许利用先导信息来选择后续观测位置。在适当的正则性假设下，被动核估计量的曲线误差以概率达到 $n^{-2/(d+3)}$ 阶，其中 $n$ 为样本量，$d$ 为流形维数。其函数型中心极限定理的极限由一个线性随机微分方程刻画。对于主动抽样，一个独立的先导样本将新的一批测量引导至目标曲线的邻域。在局部化、光滑性和设计条件下，这种集中性使得期望最大曲线误差达到 $N^{-1/2}$ 阶，其中先导样本计入总响应预算 $N$。两种方法均……（摘要原文在此处截断）

    arXiv:2611.00771v2 Announce Type: replace-cross  Abstract: We estimate the integral curve of an unknown vector field from noisy tangent vector field on a known compact Riemannian manifold. We consider two sampling settings: independent observations at random locations that have already been collected, and a fixed response budget that allows subsequent locations to be chosen using pilot information. Under suitable regularity assumptions, the passive kernel estimator has curve error of order $n^{-2/(d+3)}$ in probability, with sample size $n$ and manifold dimension $d$. Its functional central limit theorem has a limit characterized by a linear stochastic differential equation. For active sampling, an independent pilot directs one new batch of measurements to neighborhoods of the target curve. Under localization, smoothness and design conditions, this concentration yields expected maximum curve error of order $N^{-1/2}$, including the pilot in the total response budget $N$. Both methods p
    
[^23]: 面向正值金融时间序列的可扩展正则化向量乘性误差模型

    Scalable Regularized Vector Multiplicative Error Models for Positive-valued Financial Time Series

    [https://arxiv.org/abs/2610.08443](https://arxiv.org/abs/2610.08443)

    本文提出一种基于分层滞后结构与分块坐标下降算法的正则化对数向量乘性误差模型，实现了高维正值金融时间序列的高效估计与预测。

    

    对数乘性误差模型在对数vMEM形式下已被广泛应用于多元正值金融时间序列的建模与预测。然而，随着系统维度和滞后阶数的增加，参数数量迅速增长，使得高维情形下的模型估计在计算上极具挑战性。本文针对采用Tsionas（2004）多元伽马误差分布的对数vMEM模型，提出了基于分层滞后结构（Nicholson et al., 2020）的正则化估计方法。参数估计采用带有Gauss-Seidel式更新方案（Wright, 2015）的分块坐标下降算法，相比传统的惩罚极大似然方法，该策略实现了更高效的计算。此外，论文通过组合三种分层滞后结构（分量级、元素级、自身-其他）与四种惩罚项（组lasso、自适应组lasso、组……），对各类竞争模型进行了比较评估。

    arXiv:2610.08443v1 Announce Type: cross  Abstract: The logarithmic multiplicative error model (log-vMEM) has been useful in modeling and forecasting multivariate positive-valued financial time series. The number of parameters grow rapidly with the dimension of the system and the lag order, making estimation computationally demanding in high-dimensional settings. This paper describes regularized estimation via hierarchical lag structures (Nicholson et al., 2020) for log-vMEM models with multivariate gamma error distribution of Tsionas (2004). The parameter estimation is performed using a blockwise coordinate descent algorithm with a Gauss-Seidel-style update scheme (Wright, 2015). This enables an efficient computation strategy compared to traditional penalized maximum likelihood approaches. The competing models are juxtaposed against each other by combining three hierarchical lag structures (componentwise, elementwise, own-other) and four penalties(group-lasso, adaptive group-lasso, gro
    
[^24]: 面向长记忆时间序列的高维高斯图模型检验

    High-dimensional Gaussian Graphical Model Testing for Long-Memory Time Series

    [https://arxiv.org/abs/2609.30565](https://arxiv.org/abs/2609.30565)

    提出了一种适用于长记忆时间序列的数据自适应高斯图模型条件独立性检验方法，建立了有限样本高斯逼近界并通过块自助法保证超高维情形下的有效性，检验在水平和功效上均具渐近一致性，可应用于fMRI功能连接性分析。

    

    许多现实世界中的高维时间序列表现出长记忆性，但针对这一情形的高斯图模型检验研究仍然不足。我们开发了一种直接的、数据自适应的检验统计量，用于评估平稳高斯时间序列图结构中的条件独立性。我们为该统计量建立了有限样本的Berry--Esseen型高斯逼近界，该结果同时适用于短记忆和长记忆时间序列。该检验程序通过块自助法实现完全数据自适应，我们为其提供了包括超高维情形在内的有限样本有效性结果，且该方法可以扩展到两样本检验中以比较不同的图结构。我们还对该统计量开发了一种一致性增强的修正方法，并证明此类检验在检验水平和功效两方面均达到渐近一致性。我们将所提出的方法应用于真实的fMRI数据，以理解功能连接性……

    arXiv:2609.30565v1 Announce Type: cross  Abstract: Many real-world high-dimensional time series exhibit long-memory, but Gaussian graphical model testing in this regime remains understudied. We develop a direct, data-adaptive test statistic for assessing conditional independence in the graph structure of stationary Gaussian time series. We establish a finite-sample, Berry--Esseen type Gaussian approximation bound for the statistic, which applies to both short-memory and long-memory time series. The testing procedure is fully data-adaptive using block bootstrap method, on which we provide a finite-sample validity result including in the ultra-high-dimensional scenario, and can be extended to comparing graphical structures in two-sample tests. We also develop a consistency-empowered correction to the statistic and show that such tests attain asymptotic consistency in both size and power. Our proposed method is applied to a real-world fMRI data to understand functional connectivities with
    
[^25]: 反例与充分条件：对《最优传输广义矩方法》一文的评论

    Counterexamples and Sufficient Conditions: Comments on "Optimally-Transported Generalized Method of Moments"

    [https://arxiv.org/abs/2609.20260](https://arxiv.org/abs/2609.20260)

    本文对Schennach & Starck提出的最优传输广义矩方法（OTGMM）估计量的核心定理给出反例，证明其定理2-6在原假设条件下不成立，并探讨使这些结论成立的充分条件。

    

    我们针对Schennach & Starck（2026a）提出的最优传输广义矩方法（OTGMM）估计量进行评论，并在其所述假设条件下给出针对定理2-6的反例。首先，小误差分析中所使用的假设不足以保证定理2的一致性和定理3的渐近正态性。其次，我们考察大误差分析，其中定理4声称OTGMM估计量等价于带修正矩的GMM估计量。我们证明，在标量模型中，定理4所选择的值不同于唯一的OTGMM最小化点，且违反了OTGMM样本矩约束。在一个满足定理5和定理6假设的过度识别模型中，拉格朗日乘子的第一个分量在OTGMM估计量和带修正矩的GMM估计量下具有不同的概率极限。在模型误设的情形下，OTGMM所选择的总体值取决于……（原文摘要至此截断）

    arXiv:2609.20260v1 Announce Type: cross  Abstract: We comment on the optimally-transported generalized method of moments (OTGMM) estimator proposed by Schennach & Starck (2026a) and give counterexamples to Theorems 2-6 under their stated assumptions. First, the assumptions used in the small-error analysis are insufficient for consistency in Theorem 2 and asymptotic normality in Theorem 3. Next, we consider the large-error analysis, in which Theorem 4 states that the OTGMM estimator is equivalent to a GMM estimator with modified moments. We show that in a scalar model, Theorem 4 selects a value that differs from the unique OTGMM minimizer and violates the OTGMM sample moment restriction. In an overidentified model satisfying the assumptions used in Theorems 5 and 6, the first component of the Lagrange multiplier has different probability limits under the OTGMM estimator and the GMM estimator with modified moments. Under misspecification, the population value selected by OTGMM depends on
    
[^26]: 随机学习动力学中信息获取的速度极限

    Speed Limit for Information Acquisition in Stochastic Learning Dynamics

    [https://arxiv.org/abs/2609.08219](https://arxiv.org/abs/2609.08219)

    本研究将随机梯度下降建模为马尔可夫随机过程，推导出费希尔信息流的速度极限，定量约束了可训练参数获取数据中潜在变量信息的速率，并将信息流分解为漂移（确定性学习力）与噪声（SGD涨落）两部分贡献。

    

    神经网络通过学习获取内部表示。在本工作中，我们将随机梯度下降（SGD）表述为一个马尔可夫随机过程，并推导出一个费希尔信息流速度极限，该极限约束了可训练参数从数据生成过程中的潜在变量获取信息的速率。所得不等式将信息流分解为漂移与噪声两部分贡献，从而从信息论视角量化了确定性学习力和SGD诱导的涨落所起的作用。我们在可解析求解的基函数线性回归中验证了该界限，其中由该界限预测的信息预算再现了不同潜在变量在习得参数中被编码的先后次序及特征时间尺度。这些结果将费希尔信息速度极限确立为一个定量框架，可用于诊断扩散动力学何时以及如何……

    arXiv:2609.08219v1 Announce Type: cross  Abstract: Neural networks acquire internal representations through learning. In this work, we formulate stochastic gradient descent (SGD) as a Markovian stochastic process and derive a Fisher-information flow speed limit that bounds the rate at which trainable parameters can acquire information about latent variables in the data-generating process. The resulting inequality decomposes the information flow into drift and noise contributions, thereby quantifying the roles of deterministic learning forces and SGD-induced fluctuations from an information-theoretic perspective. We verify the bound in analytically tractable basis-function linear regression, where the information budget predicted by the bound reproduces the ordering and characteristic time scales with which different latent variables are encoded in the learned parameters. These results establish Fisher-information speed limits as a quantitative framework for diagnosing when and how diff
    
[^27]: 面向最优传输映射估计及其统计最优性的深度重心回归

    Deep Barycentric Regression for Optimal Transport Map Estimation and its Statistical Optimality

    [https://arxiv.org/abs/2609.06598](https://arxiv.org/abs/2609.06598)

    该论文提出了一种简单的两步方法BROT：先计算未正则化的最优传输方案，再用深度神经网络对所得重心目标进行最小二乘回归，从而实现了兼具统计最优保证与实用可扩展性的最优传输映射估计。

    

    最优传输（OT）映射提供了一种用于对齐概率分布的几何变换，并已成为机器学习中的实用工具。然而，现有的OT映射估计器在严格的统计保证与基于稳定训练目标的实用参数化估计之间仍存在差距。理论性估计器能够达到极小化极大（minimax）最优收敛速度，但它们通常是非参数化的，可能需要苛刻的实现设计或高昂的推断成本。实用估计器则是参数化且可扩展的，但其统计保证仍未被充分研究，且其极小极大式的类对抗训练目标可能对优化算法较为敏感。我们提出了BROT（Barycentric Regression for OT，最优传输重心回归），这是一种简单的两步方法：首先计算未正则化的OT方案，然后通过最小二乘回归将深度神经网络（DNN）拟合到由此产生的重心目标上。在标准正则（原文在此处截断）……

    arXiv:2609.06598v1 Announce Type: cross  Abstract: The optimal transport (OT) map provides a geometric transformation for aligning probability distributions and has become a useful tool in machine learning. However, existing estimators of the OT map still exhibit a gap between sharp statistical guarantees and practical parametric estimation based on stable training objectives. Theoretical estimators achieve minimax optimal convergence rates, but they are typically nonparametric and can incur demanding implementation design or inference costs. Practical estimators are parametric and scalable, but their statistical guarantees remain underexplored, and their min-max, adversarial-like training objectives can be sensitive to optimization algorithms. We propose BROT (Barycentric Regression for OT), a simple two-step method that first computes the unregularized OT plan and then fits a deep neural network (DNN) to the induced barycentric targets by least-squares regression. Under standard regu
    
[^28]: 几何约束的科尔莫戈罗夫-阿诺德网络：通过巴拿赫对偶学习边几何

    Geometry-Constrained Kolmogorov-Arnold Networks: Learning Edge Geometry via Banach Duality

    [https://arxiv.org/abs/2608.25807](https://arxiv.org/abs/2608.25807)

    本文提出了一种通过每条边学习标量指数 $p$ 来动态调整激活函数几何结构的KAN变体，在符号回归任务中优于固定基函数方法。

    

    科尔莫戈罗夫-阿诺德网络（KANs）用可学习的单变量边函数替代深度架构中的固定激活函数，使得边参数化的选择成为核心问题。现有变体依赖固定基函数，如样条、多项式或傅里叶特征，这些基函数在数据观测前就施加了函数空间的几何结构。我们引入了几何约束的KANs，这是一类基于巴拿赫对偶映射导出的边激活函数家族，其中几何本身通过每条边上的标量指数 $p > 1$ 进行学习。该指数控制定性响应：亚欧几里得值产生尖锐的阈值状行为，类似于 $\ell_1$（LASSO）几何；$p = 2$ 恢复线性区域；更大的值在原点附近产生更平坦的响应。在50个符号回归目标上（40个来自AI Feynman基准，加上10个合成压力测试），几何约束的KANs在中等性能上匹配或超越了所有固定基线的基准。

    arXiv:2608.25807v1 Announce Type: new  Abstract: Kolmogorov-Arnold Networks (KANs) replace fixed activations in deep architectures with learnable univariate edge functions, making the choice of edge parametrisation central. Existing variants rely on fixed bases such as splines, polynomials, or Fourier features, which impose a function-space geometry before data are observed. We introduce geometry-constrained KANs, a family of edge activations derived from Banach duality maps in which the geometry itself is learned through a scalar exponent $p > 1$ per edge. This exponent controls the qualitative response: sub-Euclidean values produce sharp, threshold-like behaviour reminiscent of the $\ell_1$ (LASSO) geometry, $p = 2$ recovers the linear regime, and larger values produce flatter responses near the origin. Across 50 symbolic-regression targets ($40$ from the AI Feynman benchmark plus $10$ synthetic stress tests), geometry-constrained KANs match or beat every fixed-basis baseline on medi
    
[^29]: 流式$r$-PCA的推断与不确定性量化

    Inference and Uncertainty Quantification for Streaming $r$-PCA

    [https://arxiv.org/abs/2608.18374](https://arxiv.org/abs/2608.18374)

    本文通过Oja算法解决了流式PCA的尖锐收敛速率和分布推断问题，首次在一般秩下实现无余项的收敛分析，并建立匹配的极小极大下界。

    

    arXiv:2608.18374v1 公告类型：交叉 摘要：我们通过Oja算法解决了流式PCA中的两个开放问题：在次高斯数据下一般秩的尖锐算子范数收敛，以及由此产生的子空间估计量的分布推断。现有的收敛性分析，即使在秩为一的情况下，要么假设数据有界，要么留下不消失的余项，从而阻碍了对多项式衰减尾部谱的适应性，而现有的分布结果仅限于秩为一的情况。我们的收敛理论去除了这些余项，并得到了一个尖锐的速率。在密集尾部尖峰协方差体制下，该速率在对数因子内达到极小极大速率。更一般地，我们在温和的非退化条件下，证明了在密集尾部和稀疏尾部体制下，匹配的下界在对数因子内成立。该分析导出了Oja迭代的线性化，进而实现了对基因的高维高斯近似。

    arXiv:2608.18374v1 Announce Type: cross  Abstract: We address two open questions in streaming PCA via Oja's algorithm: sharp operator-norm convergence for general rank under sub-Gaussian data, and distributional inference for the resulting subspace estimator. Existing convergence analyses, even in the rank-one case, either assume bounded data or leave non-vanishing remainder terms that prevent adaptation to a polynomially vanishing tail spectrum, while existing distributional results are confined to the rank-one case. Our convergence theory removes these remainder terms and yields a sharp rate. In the dense-tail spiked covariance regime, this rate matches the minimax rate up to logarithmic factors. More generally, we prove a matching lower bound, up to logarithmic factors, across both dense-tail and sparse-tail regimes under a mild nondegeneracy condition. The analysis yields a linearization of Oja's iterates, which in turn enables a high-dimensional Gaussian approximation for the gene
    
[^30]: 设计扩散模型的强化学习：一个统一的路径空间视角

    Designing Reinforcement Learning for Diffusion Models: A Unified Path-Space View

    [https://arxiv.org/abs/2608.14430](https://arxiv.org/abs/2608.14430)

    本文通过统一路径空间原则，揭示了扩散模型RL中反向轨迹与前向匹配方法的共同起源，并推导出方差缩减的等效形式，弥合了方法间的经验差距。

    

    强化学习（RL）后训练提供了一种直接的方法，使扩散模型与人类偏好和任务特定奖励对齐。然而，当前用于扩散模型的RL算法仍然碎片化：反向轨迹方法依赖于离散化的似然比，而前向匹配方法则对奖励标注的噪声版本回放样本进行训练。本文表明，这些看似不同的损失函数源于一个单一的路径空间原则。从正则化扩散RL目标出发，我们使用采样SDE之间的重要性采样，获得轨迹空间上的显式策略梯度估计器。该估计器包含Flow-GRPO型更新背后的随机伊藤积分；我们推导出一个等效的方差缩减值梯度形式，它恢复了AWM和DiffusionNFT的前向匹配结构。这确定了这些方法族之间的经验差距作为方差缩减。

    arXiv:2608.14430v1 Announce Type: new  Abstract: Reinforcement learning (RL) post-training provides a direct way to align diffusion models with human preferences and task-specific rewards. However, current RL algorithms for diffusion models remain fragmented: reverse-trajectory methods rely on discretized likelihood ratios, whereas forward-matching methods train on reward-labeled noising versions of the rollout samples. This paper shows that these seemingly different losses arise from a single path-space principle. Starting from the regularized diffusion-RL objective, we use importance sampling between sampling SDEs to obtain an explicit policy-gradient estimator on trajectory space. The estimator contains the stochastic It\^o integral underlying Flow-GRPO-type updates; we derive an equivalent variance-reduced value-gradient form that recovers the forward-matching structure of AWM and DiffusionNFT. This identifies the empirical gap between these method families as a variance-reduction 
    
[^31]: 整合背景知识以实现可扩展的因果发现

    Integrating Background Knowledge for Scalable Causal Discovery

    [https://arxiv.org/abs/2607.10456](https://arxiv.org/abs/2607.10456)

    该论文提出了一个在因果发现过程中（而非仅作为事后后处理）整合专家背景知识的框架，并针对仅恢复部分图结构的大规模可扩展因果发现方法进行了多算法实现与实验验证。

    

    在因果发现的实际应用中，专家背景知识往往是可获得的。这类对真实因果图的约束能够在因果效应的可识别性和学习结构的准确性方面帮助因果发现，同时还能缩小候选因果图的搜索空间。由于在变量数量众多时因果发现的计算开销可能变得非常昂贵，因此在因果发现过程中有效利用背景知识至关重要。然而，目前大多数方法仅在因果发现完成后的后处理步骤中使用背景知识来精炼学习到的图。在这项工作中，我们开发了一个在因果发现过程中利用背景知识的框架，尤其关注仅恢复整个图的一部分子集的可扩展因果发现方法。我们为多种算法实现了该框架，并通过实验证明，利用背景知识（摘要在此处被截断）……

    arXiv:2607.10456v2 Announce Type: replace-cross  Abstract: Expert background knowledge is often available in practical applications of causal discovery. Such constraints on the true causal graph can help causal discovery in terms of identifiability of causal effects and accuracy of the learned structure, but also in reducing the space of candidate causal graphs. As causal discovery can become computationally expensive for large number of variables, it is crucial to utilize background knowledge effectively during the causal discovery process. However, most current methods only use background knowledge in a postprocessing step after causal discovery to refine the learned graph. In this work, we develop a framework for utilizing background knowledge during the causal discovery process, focusing especially on scalable causal discovery methods that recover only a subset of the whole graph. We implement our framework for multiple algorithms and empirically show that utilizing background know
    
[^32]: 扩散驱动的状态空间模型

    Diffusion-Driven State Space Models

    [https://arxiv.org/abs/2606.21036](https://arxiv.org/abs/2606.21036)

    本文提出扩散驱动的状态空间模型（DDSSM），以扩散模型替代传统高斯转移分布，解决了在序列数据上联合训练自编码器与扩散模型的开放问题，并在多峰转移时间序列的拟合与预测上超越现有最先进的深度状态空间模型。

    

    在许多领域中，从业者希望模型既能产生准确的预测，又能忠实地捕捉潜在的系统动态。现有方法通常会牺牲其中一个目标：深度状态空间模型往往假设潜变量转移服从高斯分布，从而限制了拟合与预测能力，而扩散模型虽然具有很强的表达能力，却缺乏对底层动态的原则性推断。为了结合两者的优势，我们提出了扩散驱动的状态空间模型（DDSSM），它用扩散模型替代了传统的高斯转移分布。我们的DDSSM解决了如何在序列数据上联合训练自编码器和扩散模型这一开放性问题，从而拓展了面向时间序列的潜扩散模型相关文献。此外，我们发现DDSSM在拟合与预测具有多峰转移的模拟时间序列时，实证表现优于最先进的深度状态空间模型。

    arXiv:2606.21036v2 Announce Type: replace-cross  Abstract: In many domains, practitioners seek models that produce accurate forecasts while faithfully capturing latent system dynamics. Existing approaches typically sacrifice one of these goals: deep state space models often assume Gaussian latent transitions, limiting fit and forecasting, while diffusion models are highly expressive but lack principled inference for the underlying dynamics. To combine the strengths of both, we introduce the Diffusion-Driven State Space Model (DDSSM), which replaces the conventional Gaussian transition distribution with a diffusion model. Our DDSSM resolves the open problem of how to jointly train an autoencoder and a diffusion model on sequential data, thereby extending the literature on latent diffusion models for time series. Moreover, we find that the DDSSM empirically outperforms a state-of-the-art deep SSM at fitting and forecasting a simulated time series with multimodal transitions.
    
[^33]: 递归训练的扩散模型：极限坍缩分布与谱特征刻画

    Recursively Trained Diffusion Models: Limiting Collapse Distribution and Spectral Characterization

    [https://arxiv.org/abs/2606.13796](https://arxiv.org/abs/2606.13796)

    该论文证明，即使在评分估计与采样完全精确的理想条件下，反向扩散的提前停止也会使递归自训练的扩散模型以几何速率收敛到一个唯一的极限分布（即数据分布的 increasingly 高斯平滑版本的无限混合），且该过程如同低通滤波器般滤除高频信息，从而完全刻画了模型坍缩的极限分布与收敛速度。

    

    在生成模型自身的输出上进行递归训练可能导致模型坍缩（model collapse），即不断累积地偏离真实数据分布。现有理论工作针对扩散模型给出了有限轮次误差累积的界，但仍有两个问题悬而未决：递归最终会收敛到什么分布？收敛速度有多快？我们对这两个问题均给出了解答，并分离出一种不同于“学习不完美”的机制：即使评分估计完全精确、采样也完全精确，反向扩散过程的提前停止（数值稳定性所必需的）也会驱动分布逐渐偏离真实数据分布。我们证明该递归以几何速率收敛到一个唯一的极限分布，该分布具有闭式刻画——它是数据分布经过越来越强的高斯平滑后的版本的无限混合。对该极限分布的 Hermite 谱分解揭示出，递归训练起到了低通滤波器的作用：高频……（原文摘要在此处截断）

    arXiv:2606.13796v2 Announce Type: replace-cross  Abstract: Recursive training of generative models on their own outputs can lead to model collapse, a compounding drift away from the true data distribution. Existing theoretical works bound finite-round error accumulation in the context of diffusion models, but two questions remain open:~what distribution does the recursion converge to, and how fast? We answer both, isolating a mechanism distinct from imperfect learning: even with perfect score estimation and exact sampling, the early stopping of the reverse diffusion (required for numerical stability) drives a progressive drift away from the data distribution. We prove that this recursion converges geometrically to a unique limiting distribution, which admits a closed-form characterization as an infinite mixture of increasingly Gaussian-smoothed versions of the data distribution. A Hermite spectral decomposition of this limit reveals that recursive training acts as a low-pass filter: hi
    
[^34]: 基于监督解耦的组织图反事实查询

    Querying Counterfactuals on Tissue Graphs with Supervised Disentanglement

    [https://arxiv.org/abs/2606.08493](https://arxiv.org/abs/2606.08493)

    该论文提出Cellina框架，通过监督解耦将细胞的内在状态与空间上下文分离，并首次将组织图反事实形式化为边扰动和节点扰动两类空间干预，在覆盖超过250万个细胞的基准测试中超越了现有空间与非空间方法。

    

    组织图反事实旨在回答这样一个问题：如果一个细胞的空间邻域发生改变，它会表达什么？这是预测组织中细胞行为的核心问题。然而，现有方法只能呈现此类查询的特定实例。在这项工作中，我们首先将组织图反事实形式化为一类空间干预，这些干预要么重新连接细胞之间的连接（边扰动），要么修改其邻居的表达（节点扰动）。随后，我们提出了Cellina框架，该框架利用监督解耦将细胞的内在状态从其空间环境中分解出来，并将空间环境作为计算机内扰动预测的条件输入。在涵盖超过250万个具有空间分辨率的观察性与干预性筛选细胞的基准测试中，Cellina在计算机内扰动预测方面优于空间感知和非空间的竞争方法，同时训练速度与非空间方法相当。

    arXiv:2606.08493v4 Announce Type: replace-cross  Abstract: Tissue graph counterfactuals ask what a cell would express if its spatial neighborhood were altered, a question central to predicting cell behavior in tissues. Existing methods, however, present only specific instances of such queries. In this work, we first formalize tissue graph counterfactuals as a class of spatial interventions that either rewire connections between cells (edge perturbation) or modify the expression of their neighbors (node perturbation). We then introduce Cellina, a framework that uses supervised disentanglement to decompose a cell's intrinsic state from its spatial context, using the latter as a conditioning input for in-silico perturbation predictions. Across benchmarks spanning over 2.5 million spatially-resolved cells in observational and interventional screens, Cellina outperforms both spatially-informed and non-spatial competitors in in-silico perturbations, while training as fast as non-spatial meth
    
[^35]: 单策略集中性下具有前向KL正则化的离线上下文老虎机的快速收敛率

    Fast Rates for Offline Contextual Bandits with Forward-KL Regularization under Single-Policy Concentrability

    [https://arxiv.org/abs/2605.09214](https://arxiv.org/abs/2605.09214)

    本文首次在单策略集中性条件下，为前向KL正则化的离线上下文老虎机给出了表格型与一般函数逼近设置下的 Õ(ε^{-1}) 快速收敛率上界，通过新颖地运用悲观原理的凸分析框架填补了前向KL正则化统计分析的空白。

    

    KL（Kullback-Leibler）散度正则化以反向KL或前向KL的形式广泛存在于强化学习算法中。最近的研究表明，在反向KL正则化下的决策问题可以达到 ε^{-1} 型的快速收敛率，相比之下标准的样本复杂度为 ε^{-2} 型。然而，对于前向KL正则化的目标函数，现有的统计分析要么不适用，要么只能得到 Õ(ε^{-2}) 的慢收敛率。我们通过对前向KL正则化离线上下文老虎机的精简分析，迈出了解决这一问题的第一步。我们在表格型设置和一般函数逼近设置下，均在单策略集中性的概念下，首次给出了 Õ(ε^{-1}) 的上界。特别地，我们的凸分析流程以一种新颖的方式利用悲观原理统一了这些设置，并完全绕过了证明中的……（原文摘要在此处截断）

    arXiv:2605.09214v2 Announce Type: replace-cross  Abstract: \emph{Kullback-Leibler} (KL) regularization is ubiquitous in reinforcement learning algorithms in the form of \emph{reverse} or \emph{forward} KL. Recent studies have demonstrated $\epsilon^{-1}$-type fast rates for decision making under reverse KL regularization, in contrast to the standard $\epsilon^{-2}$-type sample complexity. However, for forward-KL-regularized objectives, existing statistical analyses are either not applicable or result in $\tilde{O}(\epsilon^{-2})$ slow rates. We take the first step towards addressing this problem via a streamlined analysis of forward-KL-regularized offline CBs. We give the first $\tilde{O}(\epsilon^{-1})$ upper bounds in tabular and general function approximation settings, both under notions of \emph{single-policy concentrability}. In particular, our convex-analytical pipeline unifies these settings by exploiting the pessimism principle in a novel way and completely bypasses the proof r
    
[^36]: 无噪声逆向优化的紧致泛化界

    Tight Generalization Bounds for Noiseless Inverse Optimization

    [https://arxiv.org/abs/2605.08866](https://arxiv.org/abs/2605.08866)

    本文为无噪声逆向优化给出了紧致的高概率 O(d/T) 泛化界，在动作唯一性条件下将其与老虎机最优臂识别结果对齐，并证明瞬时与累积遗憾的下界与对抗性设置的上界相匹配，从而表明该速率不可再改进。

    

    逆向优化（IO）旨在从观测到的上下文—动作数据中推断决策者目标函数的参数。我们研究无噪声的逆向优化，即演示数据由真实目标函数生成。我们为诱导动作集提供了高概率的 O(d/T) 泛化界，其中 d 是未知参数的数量，T 是训练数据集的规模。在保证所选动作唯一性的附加条件下，我们进一步加强了这一保证，使我们的 IO 保证与老虎机文献中最优臂识别的结果相一致。我们进一步证明，O(d/T) 的速率对于本文所考虑的所有一致估计量都是紧的，并将该结果扩展到瞬时遗憾和累积遗憾两种情形。值得注意的是，由此得到的遗憾下界与对抗性设置中相应的上界相匹配，这表明随机 IO 设置实际上在本质上等同于……

    arXiv:2605.08866v2 Announce Type: replace-cross  Abstract: Inverse optimization (IO) seeks to infer the parameters of a decision-maker's objective from observed context--action data. We study noiseless IO, where demonstrations are generated by a ground-truth objective. We provide a high-probability ${O}(\frac{d}{T})$ generalization bound for the induced action set, where $d$ is the number of unknown parameters and $T$ is the size of the training dataset. We strengthen this guarantee under additional conditions that ensure uniqueness of the chosen action, bringing our IO guarantees in line with best-arm identification results in the bandit literature. We further show that the ${O}(\frac{d}{T})$ rate is tight over all consistent estimators considered here, and extend the result to both instantaneous and cumulative regret. Notably, the resulting regret lower bound matches the corresponding upper bounds in the adversarial setting, indicating that the stochastic IO setting is effectively ad
    
[^37]: 经验贝叶斯再偏置

    Empirical Bayes Rebiasing

    [https://arxiv.org/abs/2605.08069](https://arxiv.org/abs/2605.08069)

    提出经验贝叶斯再偏置策略，通过估计未知偏差分布从数据中学习应重新引入的偏差量，从而显著缩短校准置信区间，并在预测驱动推断（如大语言模型胜率评估）和家系GWAS直接遗传效应推断中取得显著精度提升。

    

    我们研究了对大量含噪且有偏的估计进行同时分析的方法，其中每个估计都配有一个比其本身噪声更大的偏差估计。分析者的目标是为每个参数构建短且校准良好的置信区间。标准的去偏方法是将偏差估计从每个有偏估计中减去，这会增大方差并导致区间变长。在本文中，我们提出了一种经验贝叶斯再偏置策略，该策略从完全去偏的估计出发，通过估计未知的偏差分布，从数据中学习应该重新引入多少偏差。当偏差分布采用非参数极大似然方法估计时，我们提供了区间覆盖率的收敛速度。此外，我们在预测驱动推断中展示了显著的精度提升，包括大语言模型的两两胜率评估，以及基于家系的全基因组关联分析（GWAS）中直接遗传效应的推断。

    arXiv:2605.08069v2 Announce Type: replace-cross  Abstract: We study methods for simultaneous analysis of many noisy and biased estimates, each paired with an even noisier estimate of its own bias. The analyst's goal is to construct short calibrated intervals for each parameter. The standard debiasing approach, which subtracts the bias estimate from each biased estimate, inflates variance and yields long intervals. In this paper, we propose an empirical Bayes rebiasing strategy that starts from the fully debiased estimates and learns from data how much bias to reintroduce by estimating the unknown bias distribution. We provide convergence rates for the coverage of our intervals when the bias distribution is estimated using nonparametric maximum likelihood. Furthermore, we demonstrate substantial precision gains in prediction-powered inference, including pairwise LLM win-rate evaluations, as well as for inference of direct genetic effects in family-based GWAS.
    
[^38]: 基于泊松-莫罗漂移的随机逼近与强化学习的几乎必然收敛速率

    Almost Sure Convergence Rates of Stochastic Approximation and Reinforcement Learning via a Poisson-Moreau Drift

    [https://arxiv.org/abs/2605.07104](https://arxiv.org/abs/2605.07104)

    本文通过泊松-莫罗漂移技术，为马尔可夫噪声下期望更新具有压缩性的随机逼近算法（如Q学习和线性时序差分学习）建立了几乎必然收敛速率，在幂律学习率下获得了可任意接近o(n^(1-2η))的速率，在调和学习率下当参数满足条件时平方误差指数可任意接近1。

    

    在马尔可夫噪声下建立随机逼近和强化学习的几乎必然收敛速率是一个根本性的理论挑战。我们为一类期望更新具有压缩性质的随机逼近算法在这一挑战上取得了进展，该设置出现在许多强化学习算法中，例如Q学习和线性时序差分学习。具体而言，对于幂律学习率O(n^(-η))（其中η ∈ (1/2, 1)），我们获得了可任意接近o(n^(1-2η))的几乎必然收敛速率。对于调和学习率α_n=α/(n+1)，对于每个ζ<min{1,2(1-κ)α}，几乎必然平方误差速率为o(n^(-ζ))，其中κ为压缩因子。因此，当2(1-κ)α≥1时，可获得任意接近1的指数。这接近了经典迭代对数定律（原文截断）……的n^(-1)多项式指数。

    arXiv:2605.07104v3 Announce Type: replace  Abstract: Establishing almost sure convergence rates for stochastic approximation and reinforcement learning under Markovian noise is a fundamental theoretical challenge. We make progress towards this challenge for a class of stochastic approximation algorithms whose expected updates are contractive, a setting that arises in many reinforcement learning algorithms such as $Q$-learning and linear temporal difference learning. Specifically, for a power-law learning rate $O(n^{-\eta})$ with $\eta \in (1/2, 1)$, we obtain an almost sure convergence rate arbitrarily close to $o(n^{1 - 2\eta})$. For the harmonic learning rate $\alpha_n=\alpha/(n+1)$, the almost sure squared-error rate is $o(n^{-\zeta})$ for every $\zeta<\min\{1,2(1-\kappa)\alpha\}$, where $\kappa$ is the contraction factor. Thus exponents arbitrarily close to one are available when $2(1-\kappa)\alpha\geq1$. This approaches the $n^{-1}$ polynomial exponent of classical law-of-the-iter
    
[^39]: 顿悟还是故障？低精度如何驱动“弹弓机制”式损失尖峰

    Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes

    [https://arxiv.org/abs/2605.06152](https://arxiv.org/abs/2605.06152)

    本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。

    

    深度神经网络在无正则化的长期训练过程中会表现出周期性的损失尖峰，这一现象被称为“弹弓机制”。现有工作通常将其归因于内在的优化动力学，但其触发机制仍不清楚。本文证明该现象是浮点算术精度极限的结果：当训练进入高置信度阶段后，正确类别 logit 与其他 logit 之间的差值可能超过吸收误差阈值。于是在反向传播过程中，正确类别的梯度被精确舍入为零，而错误类别的梯度仍保持非零。这打破了跨类别梯度的零和约束，并在分类器层的参数更新中引入了系统性漂移。我们证明该漂移与特征之间形成了正反馈回路，导致全局分类器均值与全局特征（摘要在此处截断）。

    arXiv:2605.06152v4 Announce Type: replace-cross  Abstract: Deep neural networks exhibit periodic loss spikes during unregularized long-term training, a phenomenon known as the "Slingshot Mechanism." Existing work usually attributes this to intrinsic optimization dynamics, but its triggering mechanism remains unclear. This paper proves that this phenomenon is a result of floating-point arithmetic precision limits. As training enters a high-confidence stage, the difference between the correct-class logit and the other logits may exceed the absorption-error threshold. Then during backpropagation, the gradient of the correct class is rounded exactly to zero, while the gradients of the incorrect classes remain nonzero. This breaks the zero-sum constraint of gradients across classes and introduces a systematic drift in the parameter update of the classifier layer. We prove that this drift forms a positive feedback loop with the feature, causing the global classifier mean and the global featu
    
[^40]: 用于形状约束学习与最优传输的超输入凸神经网络

    Hyper Input Convex Neural Networks for Shape Constrained Learning and Optimal Transport

    [https://arxiv.org/abs/2604.26942](https://arxiv.org/abs/2604.26942)

    HyCNN 结合 Maxout 网络与输入凸神经网络，以指数级更少的参数实现高效凸函数学习，在凸回归、插值以及高维最优传输任务中显著优于现有方法。

    

    我们提出了超输入凸神经网络，这是一种为学习凸函数而设计的新型神经网络架构。HyCNN 将 Maxout 网络的原理与输入凸神经网络（ICNN）相结合，构建出一种在输入上始终保持凸性的神经网络，理论上能够利用网络深度，并且与 ICNN 相比在大规模训练时表现更加可靠。具体而言，我们证明了在给定精度下逼近二次函数时，HyCNN 所需的参数数量比 ICNN 呈指数级减少。通过一系列合成实验，我们证明了 HyCNN 在凸回归和插值任务的预测性能上优于现有的 ICNN 和多层感知机（MLP）。我们进一步将 HyCNN 应用于学习合成示例和单细胞数据的高维最优传输映射，在这些任务中，它们通常优于基于 ICNN 的神经最优传输方法以及其他基线方法。

    arXiv:2604.26942v2 Announce Type: replace  Abstract: We introduce Hyper Input Convex Neural Networks (HyCNNs), a novel neural network architecture designed for learning convex functions. HyCNNs combine the principles of Maxout networks with input convex neural networks (ICNNs) to create a neural network that is always convex in the input, theoretically capable of leveraging depth, and performs reliably when trained at scale compared to ICNNs. Concretely, we prove that HyCNNs require exponentially fewer parameters than ICNNs to approximate quadratic functions up to a given precision. Throughout a series of synthetic experiments, we demonstrate that HyCNNs outperform existing ICNNs and MLPs in terms of predictive performance for convex regression and interpolation tasks. We further apply HyCNNs to learn high-dimensional optimal transport maps for synthetic examples and for single-cell data, where they oftentimes outperform ICNN-based neural optimal transport methods and other baselines a
    
[^41]: 从多个思考者那里学习思考：思维链异质性如何影响高效可学习性

    Learning to Think from Multiple Thinkers: How Chain-of-Thought Heterogeneity Affects Efficient Learnability

    [https://arxiv.org/abs/2604.24737](https://arxiv.org/abs/2604.24737)

    该论文首次从学习理论视角研究异构思维链监督问题，提出多种由不同来源（最终答案相同但中间步骤各异）提供CoT数据的异构监督模型，并探讨在单一同构来源可被高效学习的前提下，何时能够从异构监督数据中实现高效学习。

    

    思维链监督通过暴露易于通过下一词预测学习的中间步骤，使计算上困难的学习问题变得可解。然而在实践中，CoT数据集来自具有不同推理风格的异构来源，并且需要投入大量精力来策划适合下一词预测训练的监督数据。这种数据策划在使CoT监督的计算优势得以实现方面扮演着什么角色？在这项工作中，我们从学习理论的角度迈出了理解这一问题的第一步。我们引入了来自多个来源的异构CoT监督的几种模型，其中每个来源都提供具有相同最终答案但中间步骤不同的正确推理轨迹。假设将下一词预测器拟合到来自任何单一来源的同构轨迹在计算上是可处理的，我们探讨这在何种条件下足以支持从异构监督中进行高效学习。

    arXiv:2604.24737v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) supervision makes computationally hard learning problems tractable by exposing intermediate steps that are easy to learn via next-token prediction. However, in practice, CoT datasets draw on heterogeneous sources with different reasoning styles, and substantial effort goes into curating supervision suitable for next-token training. What role does this curation play in enabling the computational advantage of CoT supervision? In this work, we take a first step toward understanding this question from a learning-theoretic perspective. We introduce several models of $heterogeneous$ $CoT$ $supervision$ from multiple sources, each providing correct traces with the same final answer but different intermediate steps. Assuming that fitting a next-token predictor to $homogeneous$ traces from any single source is computationally tractable, we ask when this suffices for efficient learning from heterogeneous supervisio
    
[^42]: 哈密顿系统中面向数据驱动目标可达性的辛归纳偏置

    Symplectic Inductive Bias for Data-Driven Target Reachability in Hamiltonian Systems

    [https://arxiv.org/abs/2604.17213](https://arxiv.org/abs/2604.17213)

    该论文提出利用哈密顿系统固有的辛几何结构与能量水平集上的递归特性作为归纳偏置，从而以数据高效的方式解决非线性控制中的目标可达性问题。

    

    归纳偏置是指对假设类的限制，使学习方法能够从有限数据中有效泛化。控制领域中的一个经典例子是线性性，它为稳定性和最优控制提供了低样本复杂度的保证。相比之下，对于一般的非线性动力学，理论保证通常依赖于光滑性假设（如Lipschitz连续性），这类假设在与覆盖论证结合时，可能导致数据需求随环境维度呈指数级增长。在本文中，我们认为实现数据高效的非线性控制需要利用自然本身所蕴含的归纳偏置，即物理定律所施加的结构。聚焦于哈密顿系统，我们利用辛几何以及能量水平集上的内在递归特性来解决目标可达性问题。我们的方法将递归性质与最近提出的一类称为“链式……”（chain po...，原文此处截断）的策略相结合。

    arXiv:2604.17213v2 Announce Type: replace-cross  Abstract: Inductive bias refers to restrictions on the hypothesis class that enable a learning method to generalize effectively from limited data. A canonical example in control is linearity, which underpins low sample-complexity guarantees for stabilization and optimal control. For general nonlinear dynamics, by contrast, guarantees often rely on smoothness assumptions (e.g., Lipschitz continuity) which, when combined with covering arguments, can lead to data requirements that grow exponentially with the ambient dimension. In this paper we argue that data-efficient nonlinear control demands exploiting inductive bias embedded in nature itself, namely, structure imposed by physical laws. Focusing on Hamiltonian systems, we leverage symplectic geometry and intrinsic recurrence on energy level sets to solve target reachability problems. Our approach combines the recurrence property with a recently proposed class of policies, called chain po
    
[^43]: 一种方法并非适用于所有情况：单细胞RNA测序数据填补方法的大规模基准测试

    One Method Does Not Fit All: A Large-Scale Benchmark of Imputation Methods for Single-Cell RNA Sequencing Data

    [https://arxiv.org/abs/2603.24626](https://arxiv.org/abs/2603.24626)

    本文提出了首个大规模标准化基准测试框架，在10种实验方案、6个下游任务和30个数据集上系统评估了16种单细胞RNA测序数据填补方法，结果表明没有一种方法能够适用于所有场景。

    

    单细胞RNA测序技术能够在细胞分辨率下进行基因表达谱分析，但其固有地受到由dropout事件导致的稀疏性影响，即由于技术限制，已表达的基因被记录为零。这些dropout事件会扭曲基因表达分布，并损害下游分析的效果。为应对这一挑战，众多传统方法和基于深度学习（DL）的填补方法已被开发出来，用于检测dropout事件并填补其真实表达值。然而，对这些方法的评估大多局限于特定的下游任务和有限的数据集，缺乏在标准化基准测试条件下的大规模系统性比较。为填补这一空白，我们提出了一个全面的评估框架，在10种实验方案、6个下游任务和30个基准数据集上对16种填补方法进行了评估。通过大量实验，我们系统地……（原文摘要到此截断）

    arXiv:2603.24626v3 Announce Type: replace-cross  Abstract: Single-cell RNA sequencing (scRNA-seq) enables gene expression profiling at cellular resolution but is inherently affected by sparsity caused by dropout events, where expressed genes are recorded as zeros due to technical limitations. These dropout events distort gene expression distributions and compromise downstream analyses. To address this challenge, numerous traditional and deep learning (DL)-based imputation methods have been developed to detect dropout events and impute their true expression values. However, evaluations of these methods have largely been restricted to specific downstream tasks and limited datasets, and large-scale systematic comparisons under standardized benchmarking conditions are lacking. To fill this gap, we present a comprehensive framework that evaluates 16 imputation methods across 10 experimental protocols, 6 downstream tasks, and 30 benchmark datasets. Through extensive experimentation, we syste
    
[^44]: 表演性预测中在线算法的稳定性

    The Stability of Online Algorithms in Performative Prediction

    [https://arxiv.org/abs/2602.24207](https://arxiv.org/abs/2602.24207)

    本文证明部署在表演性预测环境中的任何无悔在线算法都会无条件收敛到（混合）表演性稳定均衡，且无需对人群如何响应预测做任何假设。

    

    在决策中使用算法预测会引发一个反馈循环：我们所部署的模型会主动影响所观测到的数据分布，而这些数据随后又被用于重新训练。Perdomo 等人于 2020 年在其关于表演性预测（performative prediction）的工作中对这一动态进行了形式化刻画。我们的主要结果是一个无条件归约，表明部署在表演性环境中的任何无悔算法都会收敛到一个（混合）表演性稳定均衡：在这种解中，模型主动塑造数据分布，使得其自身的预测在事后看来是最优的。在本工作之前，该领域所有的正面结果都对模型如何影响分布施加了严格的限制。通过采用鞅论证并允许随机化，我们避免了对人群如何响应预测的任何假设，并规避了近期的困难性结果——这些结果表明确定性的稳定模型在一般情况下是 PPAD 难以计算的。最后，关于……（原文摘要在此处截断）

    arXiv:2602.24207v3 Announce Type: replace  Abstract: The use of algorithmic predictions in decision-making leads to a feedback loop where the models we deploy actively influence the data distributions we see, and later use to retrain on. This dynamic was formalized by Perdomo et al. 2020 in their work on performative prediction. Our main result is an unconditional reduction showing that any no-regret algorithm deployed in performative settings converges to a (mixed) performatively stable equilibrium: a solution in which models actively shape data distributions in ways that their own predictions look optimal in hindsight. Prior to our work, all positive results in this area imposed strong restrictions on how models influenced distributions. By using a martingale argument and allowing randomization, we avoid any assumption on how populations respond to predictions and sidestep recent hardness results showing that deterministic stable models are in general PPAD-hard to compute. Lastly, on
    
[^45]: 大语言模型的预言机鲁棒在线对齐

    Oracle-Robust Online Alignment for Large Language Models

    [https://arxiv.org/abs/2602.20457](https://arxiv.org/abs/2602.20457)

    本文针对偏好反馈预言机可能存在偏差的大语言模型在线对齐问题，提出最坏情况鲁棒优化目标，并证明该鲁棒目标可精确分解为原始损失加显式敏感度惩罚项，进而设计了投影随机复合更新算法进行求解。

    

    我们研究了在偏好反馈被错误设定情形下的大语言模型在线对齐问题，其中观测到的偏好预言机与理想但未知的真实预言机之间存在偏差。由于数据收集与策略更新之间的耦合，在线大语言模型对齐问题本质上是一个双层强化学习问题。近期，SAIL（自我提升的高效在线对齐）框架已将该问题约化为一个易于处理的单层目标。本文在该问题中引入了逐点预言机不确定性集合，并将预言机鲁棒的在线对齐目标表述为一个最坏情况优化问题。对于对数线性策略，我们证明该鲁棒目标可以精确地闭式分解为原始损失函数加上一个显式的敏感度惩罚项。在有限响应的对数线性设定下，我们针对所得的弱凸目标提出了投影随机复合更新方法……

    arXiv:2602.20457v2 Announce Type: replace  Abstract: We study online alignment of large language models under misspecified preference feedback, where the observed preference oracle deviates from an ideal but unknown ground-truth oracle. The online LLM alignment problem is a bi-level reinforcement problem due to the coupling between data collection and policy updates. Recently, the problem has been reduced to a tractable single-level objective in the SAIL (Self-Improving Efficient Online Alignment) framework. In this paper, we introduce a pointwise oracle uncertainty set in this problem and formulate an oracle-robust online alignment objective as a worst-case optimization problem. For log-linear policies, we show that this robust objective admits an exact closed-form decomposition into the original loss function plus an explicit sensitivity penalty. In the finite-response log-linear setting, we develop projected stochastic composite updates for the resulting weakly convex objective and 
    
[^46]: 基于通用投资组合算法的在线保形预测

    Online Conformal Prediction via Universal Portfolio Algorithms

    [https://arxiv.org/abs/2602.03168](https://arxiv.org/abs/2602.03168)

    本文识别出“线性化遗憾”这一关键概念并建立了通用的“遗憾到覆盖”理论，据此提出无需任何参数调整的UP-OCP方法，通过将问题归约为两资产投资组合选择问题并利用通用投资组合算法，实现了参数无关的在线保形预测覆盖率保证。

    

    在线保形预测（OCP）旨在为任意（可能是对抗性的）数据流提供能够实现长期 1-α 覆盖率的预测区间，同时尽可能保持区间信息量。现有的OCP方法通常需要手动调整学习率才能良好工作，并且可能还需要针对特定算法的分析。在本文中，我们基于 (1-α)-pinball 损失，为区间值OCP发展了一套通用的“遗憾到覆盖”理论。我们的第一个贡献是识别出“线性化遗憾”这一关键概念，表明控制该量即可为任何在线算法导出覆盖界。这一结果依赖于一个黑盒归约，其仅依赖于线性化遗憾上界的Fenchel共轭。基于这一理论，我们通过将OCP问题归约为两资产投资组合选择问题，并利用通用投资组合算法，提出了UP-OCP——一种无需调参的OCP方法。我们展示了在……上的强有限时间界。

    arXiv:2602.03168v2 Announce Type: replace-cross  Abstract: Online conformal prediction (OCP) seeks prediction intervals that achieve long-run $1-\alpha$ coverage for arbitrary (possibly adversarial) data streams, while remaining as informative as possible. Existing OCP methods often require manual learning-rate tuning to work well, and may also require algorithm-specific analyses. Here, we develop a general regret-to-coverage theory for interval-valued OCP based on the $(1-\alpha)$-pinball loss. Our first contribution is to identify \emph{linearized regret} as a key notion, showing that controlling it implies coverage bounds for any online algorithm. This relies on a black-box reduction that depends only on the Fenchel conjugate of an upper bound on the linearized regret. Building on this theory, we propose UP-OCP, a parameter-free method for OCP, via a reduction to a two-asset portfolio selection problem, leveraging universal portfolio algorithms. We show strong finite-time bounds on 
    
[^47]: 面向优化器超参数选择的贪心对齐原理

    Greedy Alignment Principle for Optimizer Hyperparameter Selection

    [https://arxiv.org/abs/2512.06370](https://arxiv.org/abs/2512.06370)

    该论文提出了贪心对齐原理，将优化器选择形式化为最大化损失期望下降速率（等价于优化器滤波器与梯度自相关的内积），证明了贪心最优解的存在性与稳定性，并据此为 SGD+Momentum 和 Adam/AdamW 导出了简单的动态动量选择规则。

    

    近期的研究表明，梯度-更新对齐是调节优化器更新的一个强大信号，通常能够带来更快的训练速度。我们将这一逐更新的启发式方法提升为一个具有严格数学依据的原理，用于选择和调整优化器超参数。通过将梯度和更新视为信号，并将优化器视为在两者之间进行映射的因果滤波器，我们将优化器选择问题形式化为在一组给定的优化器族上最大化损失的期望下降速率。我们证明该目标恰好等于优化器滤波器与梯度自相关函数之间的内积，并证明了贪心最优解的存在性，且在估计的梯度统计量受到扰动时具有稳定性界。针对基于动量的优化器，该理论为 SGD+Momentum 和 Adam/AdamW 都推导出了简单的动态动量选择规则。在图像分类、语言模型微调等多项任务上的实验验证了该方法的有效性。

    arXiv:2512.06370v4 Announce Type: replace  Abstract: Recent works have shown that gradient-update alignment is a powerful signal for modulating optimizer updates, often leading to faster training. We promote this update-wise heuristic as a mathematically grounded principle for selecting and tuning optimizer hyperparameters. By treating gradients and updates as signals and an optimizer as a causal filter that maps between them, we formulate optimizer selection as maximizing the expected drop rate in loss over a prescribed family of optimizers. We show that this objective is exactly the inner product between the optimizer filter and the gradient autocorrelation, and prove that a greedy optimum exists and has a stability bound under perturbations of the estimated gradient statistics. Specializing in momentum-based optimizers, the theory yields simple dynamic momentum selection rules for both SGD+Momentum and Adam/AdamW. Experiments across image classification, language model fine-tuning, 
    
[^48]: 节点流行度模型的变分估计器

    Variational Estimators for Node Popularity Models

    [https://arxiv.org/abs/2511.17783](https://arxiv.org/abs/2511.17783)

    本文为双向节点流行度模型（TNPM）开发了一种计算高效且有理论保证的变分期望最大化（VEM）估计框架，在二部网络中证明了社区划分的标签一致性，并实现了优于现有方法的估计精度。

    

    节点流行度被认为是建模现实世界网络的一个关键因素，它刻画了不同社区之间连接的异质性。这一概念在二部网络中同样重要，其中不同分部中的节点可能表现出不同的流行度模式，由此催生了双向节点流行度模型（TNPM）等模型。现有方法，如两阶段分割余弦（TSDC）算法，提供了可扩展的估计方法，但在估计精度或对不同类型网络的适用性方面可能存在局限。在本文中，我们为TNPM开发了一个计算高效且有理论支撑的变分期望最大化（VEM）框架。我们在二部网络中为所提出的变分估计器产生的社区划分估计建立了标签一致性。通过大量的仿真研究，我们证明了我们的方法能够实现更优的估计精度。

    arXiv:2511.17783v2 Announce Type: replace-cross  Abstract: Node popularity is recognized as a key factor in modeling real-world networks, capturing heterogeneity in connectivity across communities. This concept is equally important in bipartite networks, where nodes in different partitions may exhibit varying popularity patterns, motivating models such as the Two-Way Node Popularity Model (TNPM). Existing methods, such as the Two-Stage Divided Cosine (TSDC) algorithm, provide a scalable estimation approach but may have limitations in terms of accuracy or applicability across different types of networks. In this paper, we develop a computationally efficient and theoretically justified variational expectation-maximization (VEM) framework for the TNPM. We establish label consistency for the estimated community assignments produced by the proposed variational estimator in bipartite networks. Through extensive simulation studies, we show that our method achieves superior estimation accuracy
    
[^49]: 超越最大似然：广义线性模型的变分不等式估计

    Beyond Maximum Likelihood: Variational Inequality Estimation for Generalized Linear Models

    [https://arxiv.org/abs/2511.03087](https://arxiv.org/abs/2511.03087)

    本文提出用基于变分不等式的算子均衡条件取代最大似然来估计广义线性模型，证明了其有限样本误差界与渐近正态性以及相应算法的收敛保证，并能处理非标准、非光滑或饱和的连接函数。

    

    广义线性模型（GLMs）是统计建模的基础工具，最大似然估计（MLE）是参数推断的经典方法。虽然MLE在标准（canonical）GLMs中表现良好，但在更一般的情形下——例如具有非标准、非光滑或饱和连接函数的设置——它可能在计算上变得困难，因为由此产生的优化问题可能是病态的、非凸的或不可微的。本文研究了一种基于变分不等式（VIs）的替代估计框架，该框架通过基于算子的均衡条件而非似然最小化来构建GLM估计。我们从统计角度分析了VI估计量，建立了有限样本误差界和渐近正态性，并给出了不动点算法和随机逼近算法的收敛性保证。该框架适用于一大类连接函数。

    arXiv:2511.03087v3 Announce Type: replace-cross  Abstract: Generalized linear models (GLMs) are fundamental tools for statistical modeling, with maximum likelihood estimation (MLE) serving as the classical approach for parameter inference. While MLE performs well for canonical GLMs, it can become computationally challenging in more general settings with non-canonical, non-smooth, or saturating link functions, where the resulting optimization landscape may be ill-conditioned, non-convex, or non-differentiable. In this paper, we study an alternative estimation framework based on variational inequalities (VIs), which formulates GLM estimation through an operator-based equilibrium condition rather than likelihood minimization. We analyze the VI estimator from a statistical perspective and establish finite-sample error bounds and asymptotic normality, together with convergence guarantees for fixed-point and stochastic approximation algorithms. The framework accommodates a broad class of lin
    
[^50]: 相关尖峰模型中的谱阈值与偏最小二乘法的基本极限

    Spectral Thresholds in Correlated Spiked Models and Fundamental Limits of Partial Least Squares

    [https://arxiv.org/abs/2510.17561](https://arxiv.org/abs/2510.17561)

    本文利用随机矩阵理论证明了部分对齐的尖峰互协方差模型中样本奇异值会经历BBP型相变，首次精确刻画了偏最小二乘法（PLS）的信号恢复能力，并揭示了PLS与贝叶斯最优估计器之间的根本性能差距，指出在某些信噪比和相关性区间内PLS会完全无法恢复信号。

    

    我们对尖峰互协方差模型提供了严格的随机矩阵理论分析，该模型中两个高维数据通道之间的信号是部分对齐的。这些模型的动机来自多模态学习，并构成了偏最小二乘法（PLS）——一种被广泛使用但理论研究不足的方法——背后的标准生成设定。我们证明了样本互协方差矩阵的领先奇异值会经历Baik-Ben Arous-Péché（BBP）型相变，并刻画了信息性成分出现的精确阈值。我们的结果首次对PLS在此设定下的信号恢复能力给出了精确的渐近描述，揭示了PLS与贝叶斯最优估计器之间的根本性能差距。特别地，我们识别出信噪比（SNR）和相关性所在的区间：尽管信号在原则上是可以被检测到的，但PLS在这些区间内却无法恢复任何信号。

    arXiv:2510.17561v2 Announce Type: replace-cross  Abstract: We provide a rigorous random matrix theory analysis of spiked cross-covariance models where the signals across two high-dimensional data channels are partially aligned. These models are motivated by multi-modal learning and form the standard generative setting underlying Partial Least Squares (PLS), a widely used yet theoretically underdeveloped method. We show that the leading singular values of the sample cross-covariance matrix undergo a Baik-Ben Arous-Peche (BBP)-type phase transition, and we characterize the precise thresholds for the emergence of informative components. Our results yield the first sharp asymptotic description of the signal recovery capabilities of PLS in this setting, revealing a fundamental performance gap between PLS and the Bayes-optimal estimator. In particular, we identify the SNR and correlation regimes where PLS fails to recover any signal, despite detectability being possible in principle. These f
    
[^51]: 关于在线线性回归问题的高斯近似速率研究

    On the Rate of Gaussian Approximation for Online Linear Regression Problems

    [https://arxiv.org/abs/2509.14039](https://arxiv.org/abs/2509.14039)

    本文针对恒定步长下的在线线性回归问题推导了高斯近似速率，显式刻画了其对问题维度和设计矩阵相关量的依赖，并证明当样本量足够大时可获得 $\sqrt{\log{n}/n}$ 阶的正态近似速率。

    

    本文研究了在线线性回归任务中的高斯近似问题。我们在恒定步长的设定下推导了相应的收敛速率，并研究了该收敛速率对问题维度 $d$ 以及与设计矩阵相关量的显式依赖关系。当迭代次数 $n$ 事先已知时，只要样本量 $n$ 足够大，我们的结果给出了阶为 $\sqrt{\log{n}/n}$ 的正态近似速率。

    arXiv:2509.14039v2 Announce Type: replace-cross  Abstract: In this paper, we consider the problem of Gaussian approximation for the online linear regression task. We derive the corresponding rates for the setting of a constant stepsize and study the explicit dependence of the convergence rate on the problem dimension $d$ and quantities related to the design matrix. When the number of iterations $n$ is known in advance, our results yield a normal approximation rate of order $\sqrt{\log{n}/n}$, provided that the sample size $n$ is large enough.
    
[^52]: 基于删失半老虎机反馈的在线预算分配

    Online Budget Allocation with Censored Semi-Bandit Feedback

    [https://arxiv.org/abs/2508.05844](https://arxiv.org/abs/2508.05844)

    本文研究了删失半老虎机反馈下的在线预算分配问题，提出了一种基于乐观原则的算法，并证明在收益递减情形下其遗憾随时间范围呈多对数级增长。

    

    我们研究了跨 K 个任务的随机预算分配问题。在每一轮 t，学习者选择一个分配方案 X_t ∈ Δ_K。任务 k 以概率 F_k(X_{t,k}) 成功，其中 F_1,...,F_K 是非递减的预算-成功曲线，任务成功时会产生一个均值为未知数 μ_k 的随机奖励。学习者可以观察到哪些任务成功了，且只有当任务成功时才能观察到该任务的奖励（删失半老虎机反馈）。该模型可以刻画诸如将支付分配给众包工作者或将出价分配给同时进行的多个拍卖等场景，并且涵盖了随机多臂老虎机和半老虎机问题。我们设计了一种基于乐观原则的算法，该算法能够在删失半老虎机反馈下运行，并假设可以使用一个精确的最大化 oracle。我们的主要结果是一个依赖于实例的界，表明在收益递减的情形下，该算法的遗憾随时间范围呈多对数级增长。

    arXiv:2508.05844v3 Announce Type: replace-cross  Abstract: We study a stochastic budget-allocation problem over $K$ tasks. At each round $t$, the learner chooses an allocation $X_t \in \Delta_K$. Task $k$ succeeds with probability $F_k(X_{t,k})$, where $F_1,\dots,F_K$ are nondecreasing budget-to-success curves, and upon success yields a random reward with unknown mean $\mu_k$. The learner observes which tasks succeed, and observes a task's reward only upon success (censored semi-bandit feedback). This model captures, for instance, splitting payments across crowdsourcing workers or distributing bids across simultaneous auctions, and subsumes stochastic multi-armed bandits and semi-bandits. We design an optimism-based algorithm that operates under censored semi-bandit feedback and assumes access to an exact maximization oracle. Our main result is an instance-dependent bound showing that in diminishing-returns regimes, the regret of this algorithm scales polylogarithmically with the horiz
    
[^53]: 通过谱神经网络和非线性矩阵感知实现隐式正则化

    Implicit Regularization via Spectral Neural Networks and Non-linear Matrix Sensing

    [https://arxiv.org/abs/2402.17595](https://arxiv.org/abs/2402.17595)

    本文在更现实的神经网络背景下探讨了隐式正则化现象，通过研究非线性激活函数的一般类别，严格证明了在矩阵感知问题设置中这些网络的隐式正则化现象，同时提供了严格的速率保证，确保梯度的指数级快速收敛。

    

    隐式正则化现象近年来引起了人们的兴趣，作为神经网络出色泛化能力的一个基本方面。简而言之，它意味着在许多神经网络中，即使损失函数中没有任何显式正则化器，梯度下降动态也会收敛到一个正则化学习问题的解。然而，已知的试图从理论上解释这一现象的结果主要集中在线性神经网络的设置上，线性结构的简单性对现有论据特别关键。本文在更现实的神经网络和一般非线性激活函数的背景下探讨了这一问题，并在矩阵感知问题的设置中严格证明了这些网络的隐式正则化现象，同时提供了严格的速率保证，确保梯度的指数级快速收敛。

    arXiv:2402.17595v1 Announce Type: cross  Abstract: The phenomenon of implicit regularization has attracted interest in recent years as a fundamental aspect of the remarkable generalizing ability of neural networks. In a nutshell, it entails that gradient descent dynamics in many neural nets, even without any explicit regularizer in the loss function, converges to the solution of a regularized learning problem. However, known results attempting to theoretically explain this phenomenon focus overwhelmingly on the setting of linear neural nets, and the simplicity of the linear structure is particularly crucial to existing arguments. In this paper, we explore this problem in the context of more realistic neural networks with a general class of non-linear activation functions, and rigorously demonstrate the implicit regularization phenomenon for such networks in the setting of matrix sensing problems, together with rigorous rate guarantees that ensure exponentially fast convergence of gradi
    

