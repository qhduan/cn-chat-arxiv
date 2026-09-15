# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quenched Ensemble Sampling](https://arxiv.org/abs/2609.15894) | 该论文提出淬火系综采样方法，通过将嵌套采样中的硬能量约束推广为能量边界处的一族排斥势，在保持能量单调递减路径的同时实现可扩展的基于梯度的采样，从而能够跨越一阶相变准确估计边际似然并抽取后验样本。 |
| [^2] | [Sharp Rates and a One-Line Correction for Spectral Representation Learning](https://arxiv.org/abs/2609.15825) | 该论文推翻了支撑谱表示学习的各向同性假设，指出迁移风险仅取决于任务协方差向依赖算子主导奇异方向的压缩是否与算子谱一致排序，并据此给出了匹配的锐利收敛速率和一行修正方法。 |
| [^3] | [Learning under Target Shift: Optimal Density Ratio Estimation and Importance-Weighted Regression](https://arxiv.org/abs/2609.15785) | 针对连续输出的目标偏移学习问题，提出RKHS中的谱正则化密度比估计方法，首次给出高概率有限样本收敛保证，并借此实现最优的重要性加权回归。 |
| [^4] | [Predictive Likelihood Ratios for Language Model Watermark Detection](https://arxiv.org/abs/2609.15657) | 该论文提出基于预测似然比的语言模型水印检测方法，通过对不确定的亏缺和尾部分布进行先验平均实现跨信号设定的稳健检测功效，并证明所得贝叶斯因子同时是检验鞅，使第一类错误控制在备择假设误设及可选停止下依然有效。 |
| [^5] | [On Detecting Multiple Simultaneous Change-points in High Dimensional Non-Stationary Time Series](https://arxiv.org/abs/2609.15479) | 本文提出基于标准与自适应融合组lasso方法来检测高维非平稳经济金融时间序列中的多重同时变点，证明了该方法的L_2一致性与L_0一致性，并量化了实现一致性检测所需的条件，最后通过美国过去50年的经济金融数据验证了其有效性。 |
| [^6] | [Graph Matching Relaxations and Amortization for Supervised Graph Prediction](https://arxiv.org/abs/2609.15437) | 该论文证明了Gromov-Wasserstein目标是监督图预测中最合适的图匹配松弛形式，并提出基于可微Sinkhorn算法的参数化匹配器来摊销图匹配问题，实现图预测模块与匹配器的联合学习。 |
| [^7] | [ReLU Neural Network Approximation to Smooth Functional Operator: Dimensional Decay and Error Analysis](https://arxiv.org/abs/2609.15355) | 本文通过结合坐标截断、各向异性分割与局部泰勒逼近的构造性分析，证明在广义指数坐标衰减条件下，深度ReLU神经网络对无穷维希尔伯特空间上光滑泛函的一致逼近误差上界与基于伪维数的下界相匹配。 |
| [^8] | [Conformal Individual Treatment Effect Estimation under Networked Interference](https://arxiv.org/abs/2609.15254) | 本文提出一种干扰调整的加权共形预测方法，通过构建共形p值的可观测上界来处理网络干扰，为反事实结果和个体处理效应提供有限样本边际覆盖保证。 |
| [^9] | [Bandits with Probing: Optimal Regret and the Limits of Winner Feedback](https://arxiv.org/abs/2609.15248) | 本文针对带探测的多臂老虎机问题首次确定了两个极小极大最优遗憾定律：在赢家反馈下，独立随机奖励情形的遗憾阶为 Φ_{n,k}(T)，而一般联合独立同分布奖励及固定序列情形的遗憾阶为 R_{n,k}(T)，并给出了匹配的任意时刻上界。 |
| [^10] | [Structured Features Overfit Where Random Features Grok](https://arxiv.org/abs/2609.15047) | 本文揭示了“顿悟”现象的边界条件——随机高斯特征映射上的岭回归会呈现先记忆后泛化的延迟，而结构化的带限傅里叶特征映射则不会出现这种延迟，其性能退化由活跃支撑集的尖锐边界决定而非插值效应。 |
| [^11] | [Data Attribution at Scale via Influence Matrix Estimation](https://arxiv.org/abs/2609.15044) | 该论文将预算受限的数据归因问题转化为从少量测量中估计大型影响矩阵的问题，并据此提出了MAGE和SPEL两种算法，大幅降低了大规模数据归因的计算成本。 |
| [^12] | [Distributed Fast Fixed-Point Algorithms for Composite Monotone Inclusions over Networks](https://arxiv.org/abs/2609.14953) | 本文提出了两种将 Nesterov 型加速与原始-对偶技术相结合的去中心化快速不动点算法（ND-DFFP 和 NI-DFFP），用于在网络中各智能体算子信息私有的条件下求解复合单调包含问题，并首次在该问题类上给出了原始空间中的精确收敛速率保证。 |
| [^13] | [Steady-State Convergence of Stochastic Approximation](https://arxiv.org/abs/2609.14922) | 本文为马尔可夫乘性噪声驱动的固定步长收缩型随机逼近建立了统一的稳态收敛理论，通过多步普适性框架突破了对独立同分布/加性噪声和全局可微性的限制，并以最优速率获得了缩放稳态的高斯近似。 |
| [^14] | [Shapley Value Estimation for Multi-Site Data with Blockwise-Missing Features](https://arxiv.org/abs/2609.14902) | 该论文揭示了在分块缺失的多站点数据中直接插补再计算Shapley值会引入依赖联盟的系统性偏差，并提出FUSHAP方法，利用部分观测的辅助站点数据来降低估计方差、实现更可靠的特征归因。 |
| [^15] | [A Functional SVD Framework for Regularized Multivariate Functional PCA with Dual Penalization](https://arxiv.org/abs/2609.14815) | 本文提出了一种基于函数奇异值分解的正则化多元函数主成分分析新框架，通过对函数主成分及主成分得分（含稀疏性惩罚）进行双重正则化，突破了现有方法仅惩罚函数主成分的局限，显著增强了结果的可解释性。 |
| [^16] | [From matrix inversion to constraints: provably tighter confidence regions for importance weights in label shift](https://arxiv.org/abs/2609.14802) | 提出从矩阵求逆转向直接矩阵约束的新框架，为标签偏移下重要性权重的估计提供可证明更紧的置信区域，同时保持精确的有限样本有效性。 |
| [^17] | [PU classification under Non-SCAR: clustering-assisted logistic model with oversampling enhancement](https://arxiv.org/abs/2609.14675) | 本论文的主要创新是在SCAR假设不成立的PU分类场景中，将SMOTE过采样技术与基于聚类辅助的逻辑回归方法（含严格和非严格Lasso正则化）相结合，以缓解类别不平衡并显著提升分类性能。 |
| [^18] | [Direct Conditional Transition Sampling for Diffusion Inverse Problems](https://arxiv.org/abs/2609.14596) | 本文提出DCTS方法，通过沿短内路径估计测量条件下的干净均值并将高斯源噪声直接传输到下一含噪状态，替代昂贵的概率流ODE积分与MCMC，在四个逆问题上实现竞争性重建质量且速度提升高达16.8倍。 |
| [^19] | [Evaluation of optimisation and Bayesian inference methods for reaction rates in atmospheric chemical mechanisms](https://arxiv.org/abs/2609.14569) | 本研究使用具有已知真实值的合成数据，比较了ODE约束神经网络优化与MCMC贝叶斯推断两种方法在大气自氧化机制反应速率系数估计中的性能，评估了它们在不同噪声水平和观测类型（直接浓度观测与质谱观测）下的表现。 |
| [^20] | [Multi-source conformal prediction: leveraging heterogeneity via localization](https://arxiv.org/abs/2609.14531) | 该论文提出多源随机局部化共形预测（MS-RLCP），通过数据自适应的来源选择和可解释的包络分布概念，在异质多源数据与分布偏移场景下实现了有限样本覆盖保证。 |
| [^21] | [Nonparametric Variance-Penalized Actor-Critic: Statistical Inference for Risk-Sensitive Reinforcement Learning](https://arxiv.org/abs/2609.14327) | 提出了一种非参数方差惩罚行动者-评论家框架（VPAC），用基于自助法和随机缩放的统计在线估计器替代第二个评论家网络来估计回报方差，无需辅助网络且方差惩罚有界，并证明了方差惩罚Q学习和双时间尺度行动者-评论家算法的几乎必然收敛性。 |
| [^22] | [On Large-Scale Multiple Testing Over Networks: A Non-Asymptotic Approach](https://arxiv.org/abs/2609.14170) | 该论文发现分布式多重检验中有限样本FDR失控源于赢者诅咒偏差，并提出交叉拟合贪心聚合算法（CFGA），通过数据分割实现了网络上分布式多重检验在有限样本下的严格FDR控制。 |
| [^23] | [Riemannian ascent--descent for nonconvex nonconcave minimax landscapes: convergence to basin saddle points and applications to distributionally robust optimization](https://arxiv.org/abs/2609.14141) | 该论文提出“盆地鞍点”的新概念，并证明黎曼梯度上升多步下降迭代在局部 Łojasiewicz 型增长条件下收敛到此类鞍点，为分布鲁棒优化中非凸非凹的极小极大问题提供了收敛性理论框架。 |
| [^24] | [Exact Finite Attention Responses From RoPE Derivatives](https://arxiv.org/abs/2609.14127) | 该论文提出通过对RoPE导数进行积分并保留键值交互项，无需线性化即可精确预测注意力干预（位置编辑）的效果，符号预测准确率达95%以上，边际MAE相比现有方法最多降低九倍以上。 |
| [^25] | [Just add noise: Debiasing tree-based variable importance in mixed data](https://arxiv.org/abs/2609.14083) | 提出通过向分类变量添加少量噪声这一简单方法，消除基于树的变量重要性度量（如随机森林）对连续变量的偏好偏差，从而实现对混合数据的变量选择并控制错误发现率。 |
| [^26] | [Linear Ensemble Sampling with Smaller Ensembles](https://arxiv.org/abs/2609.13954) | 提出一种仅在正则化Gram矩阵显著变化时才刷新集成的线性集成采样算法，将充分集成规模从Θ(d log T)降低至Θ(d log d + d log log T)，同时在任意有界动作集上保持最先进的Õ(d^{3/2}√T)遗憾保证。 |
| [^27] | [Equilibrium bias and convergence in augmented primal--dual dynamics with sampled constraints](https://arxiv.org/abs/2609.13925) | 本文揭示了采样约束引起的增广乘子信号偏倚会导致原始-对偶动力学均衡点偏移，给出了保持KKT均衡的充要条件，并通过递归约束估计与联合能量分析证明了光滑凸锥问题上原始-对偶迭代几乎必然收敛到KKT点。 |
| [^28] | [Resolution-Independent Analysis of Encoder--Decoder Operator Learning via Limiting Kernels](https://arxiv.org/abs/2609.13798) | 本文通过证明编码器-解码器架构中诱导的算子值核随分辨率提高收敛到极限核，建立了算子学习的分辨率无关理论，并为正则化随机梯度下降给出了分离编码、正则化与优化误差项的收敛上界。 |
| [^29] | [On the Equivalence of Stochastic Control and Path Space Formulations for Schr\"odinger Bridges over Compact Connected Lie Groups](https://arxiv.org/abs/2609.13758) | 本文证明了紧致连通李群上薛定谔桥问题的随机最优控制表述与路径空间相对熵最小化表述的等价性，从而在解析、概率和计算三个层面为薛定谔桥的研究提供了新的理论基础。 |
| [^30] | [Online Bayesian Node Classification on Inductive Graphs under Distribution Shift](https://arxiv.org/abs/2609.13655) | 本文提出变分贝叶斯最后一层（VBLL）方法，通过联合训练GNN编码器与最后一层的近似后验，并在测试时进行在线更新，实现了分布偏移下归纳图上节点分类的校准不确定性量化。 |
| [^31] | [Curvature-Independent Regret Bounds for Distributed Online Optimization on Hadamard Manifolds](https://arxiv.org/abs/2609.13646) | 该论文针对Hadamard流形上的分布式在线优化，提出了结合局部h-次梯度更新与隐式Fréchet均值共识的D-ROGD算法，首次建立了与曲率无关的遗憾界，对h-凸和强h-凸目标分别达到O(√T)和O(log T)的遗憾率。 |
| [^32] | [Assumption-Lean Inference for Spectral Differential Network Analysis of High-Dimensional Time Series](https://arxiv.org/abs/2609.13609) | 本文提出了一个针对高维时间序列谱差分网络分析的少假设推断框架，通过直接估计两个高维逆谱密度的差异，并开发新的高斯近似误差界来实现统计推断，可应用于诸如大脑连接网络在刺激前后变化的研究。 |
| [^33] | [Synthetic Nearest Neighbors: Extending Synthetic Controls for Matrix Completion with Missing Not at Random Data](https://arxiv.org/abs/2609.13586) | 该论文提出合成最近邻（SNN）方法，将计量经济学中的合成控制思想引入非随机缺失（MNAR）数据的矩阵补全，放宽了正性和独立性假设，支持灵活异质的观测模式，并建立了有限样本误差界、渐近正态性及可行的逐项推断。 |
| [^34] | [A pullback-corrected scalar auxiliary variable optimizer with momentum and adaptive mobility](https://arxiv.org/abs/2609.13569) | 该论文提出了一种融合动量与自适应迁移率的拉回校正标量辅助变量优化器，通过单次隐式求解同时校正梯度与动量，证明了Loewner序非递增的迁移率可保证精确的修正能量律，并给出了驻点局部稳定性的充要条件。 |
| [^35] | [Toward Optimal Switching Regret for Multi-Armed Bandits with Oblivious Adversary](https://arxiv.org/abs/2609.13547) | 本文提出单一算法，在非自适应对手下对任意未知的切换次数 $S$ 均能达到最优的 $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ 期望切换遗憾，解决了 Auer 等人提出的开放问题。 |
| [^36] | [Certifiably Interpretable Training of ReLU-MLPs for Boolean Tasks with Guaranteed Truth-Table Generalization](https://arxiv.org/abs/2609.13439) | MACCHIATO算法通过迭代地将布尔函数残差投影到低维{AND, OR, XOR}电路类并将其精确编译为ReLU-MLP，实现了从部分真值表观测出发、可认证且可解释地训练ReLU-MLP，同时保证真值表的泛化。 |
| [^37] | [Converge Then Diversify: Decoupling Convergence and Diversity in Multi-Objective Bayesian Optimisation](https://arxiv.org/abs/2609.13396) | 本文提出将多目标贝叶斯优化中的收敛性与多样性两个目标解耦，采用“先收敛后多样化”的策略，从而在有限的搜索预算下更有效地逼近帕累托前沿。 |
| [^38] | [Beyond Point Forecasts: A Survey on Probabilistic Forecasting for Time Series and Spatiotemporal Data](https://arxiv.org/abs/2609.13345) | 这篇综述通过按不确定性在预测流程中引入的位置和方式建立统一分类框架，整合了时间序列与时空概率预测中分散的统计建模、机器学习和深度生成建模方法，并涵盖了集成校准、贝叶斯建模、分布回归以及新兴的时间序列基础模型等范式。 |
| [^39] | [Stochastic Gradient Descent over P2](https://arxiv.org/abs/2609.13343) | 该论文将经典欧氏空间中SGD的扩散（高斯）近似理论首次推广到Wasserstein空间P2上的优化问题，通过Lions可微性将问题提升至线性希尔伯特空间，并构造了与随机梯度矩信息相匹配的高斯随机场近似。 |
| [^40] | [Adaptive Conformal Redistribution for Inter-class Transitional Uncertainty in Medical Image Classification](https://arxiv.org/abs/2609.13303) | 本文提出AdaConRed，一种无需标签的保形后决策规则，通过视觉-语言生成式增强、基础模型嵌入和熵调制的自适应预测集，将医学图像分类中过渡类别的模糊预测集重新分配为精确的单一类别判定。 |
| [^41] | [Harnessing human expertise for high-precision robotic assembly in industrialized construction: A sample-efficient installer-in-the-loop interactive reinforcement learning framework](https://arxiv.org/abs/2609.13234) | 该论文提出一种安装人员在环的交互式强化学习框架，通过遥操作演示、事件驱动接管和验收对齐奖励，将安装人员的隐性专业知识高效转化为工业化建筑中模块化组件高精度机器人装配的自主能力。 |
| [^42] | [Chemical and geometric representation fidelity improves drug--target affinity prediction](https://arxiv.org/abs/2609.13230) | 该论文提出表征保持框架ReGeoDTA，通过在分子表征中保留化学异质性、在蛋白质结构中保留连续几何关系，解决了表征阶段的信息丢失这一上游瓶颈，从而在三个基准数据集上持续提升药物-靶点亲和力预测性能。 |
| [^43] | [A Hilbert-Valued Functional Decomposition Framework for Explaining Time-Dependent Outputs](https://arxiv.org/abs/2609.11295) | 本文提出一个希尔伯特值泛函分解框架，将基于特征的解释方法从标量输出推广到时间相关的函数型输出，并通过基于核的输出表示实现多时间粒度的时间依赖感知解释。 |
| [^44] | [Flow Duality and Source Geometry for Categorical Generation](https://arxiv.org/abs/2609.10863) | 本文揭示了连续与离散流匹配之间的对偶性——通过逐位置argmax投影可将连续凸插值路径转化为离散凸插值路径，并证明连续源分布的几何设计（高斯、有界均匀、中心负指数等）会显著影响分类生成中的转移时机和词表规模依赖性。 |
| [^45] | [Semiparametric Inference for Conditional Shapley Feature Importance](https://arxiv.org/abs/2609.10313) | 本文针对条件Shapley特征重要性提出了一种带K折交叉拟合和U统计量修正的半参数一步估计器，消除了蒙特卡洛偏差，在双重稳健速率条件下实现√n一致性与渐近正态性，并提供覆盖率有保证的Wald置信区间。 |
| [^46] | [Distillation of Synthetic Data for Time Series Foundation Models](https://arxiv.org/abs/2609.09586) | 本文提出合成数据蒸馏（SDD）方法，通过将时间序列基础模型的输出与每条轨迹的条件预测分布而非已实现的未来值进行比较来构建预训练损失目标，该方法在理论上可证明降低随机梯度协方差，并在实证中使400万至25亿参数规模的模型验证损失收敛更快。 |
| [^47] | [Optimal Transport for Network Comparison: A Review with Machine Learning Applications](https://arxiv.org/abs/2608.27500) | 本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。 |
| [^48] | [The geometry of AI validation: Exact certification limits for iid best-of-N search](https://arxiv.org/abs/2608.21496) | 本文通过核几何方法，精确推导出独立同分布最佳N次搜索中验证的模糊宽度公式，并揭示其主导尺度为$m^2/N$，为AI验证提供了理论极限。 |
| [^49] | [The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning](https://arxiv.org/abs/2607.23711) | 该论文提出了一个无需拟合参数的逐层谱定律 $s^\ast=\bar\theta/(\gamma\sigma_1(BA))$，可仅从权重矩阵的测量谱预测LoRA微调中入侵维度出现的临界更新强度，在18个适配器、9,840次层扫描的验证中，82%的层上阈值预测误差在两倍以内。 |
| [^50] | [Operator-Informed Gaussian Processes for Complex Helmholtz Wavefields: From Synthetic Benchmarks to In Vivo Brain Elastography](https://arxiv.org/abs/2607.14193) | 本文提出将算子信息引导的高斯过程回归扩展到复值亥姆霍兹波场问题，通过将复算子实化为等价的耦合实块以实现标准实值高斯过程条件化推断，并提供了从对角先验到多尺度变体的一族先验，在合成基准问题上验证了其竞争力并有望应用于在体脑弹性成像。 |
| [^51] | [Markov Chain Monte Carlo with Diffusion Paths](https://arxiv.org/abs/2607.11631) | 该论文提出沿扩散路径而非传统温度调节方法构建中间分布进行MCMC采样，能够保留多峰分布中各模态的相对权重并获得更好的混合性质。 |
| [^52] | [Differential Privacy of Gaussian Process Posterior Sampling](https://arxiv.org/abs/2606.17995) | 该论文首次证明高斯过程后验采样的内在随机性本身即可提供差分隐私保证，并通过分离后验均值与数据依赖协方差两条泄露通道，确定有效岭正则化和协方差尺度为控制隐私的关键量。 |
| [^53] | [SAILS: Surrogate-based Analysis of Interactions via Local Effect Smooths](https://arxiv.org/abs/2606.09404) | SAILS是一个模型无关的框架，通过广义可加模型代理在局部效应层面检测特征交互，将交互形式分类为线性、乘积可分离和非乘积可分离类型，并为每种类型提供可解释的可视化。 |
| [^54] | [InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimate](https://arxiv.org/abs/2606.00241) | InfoAtlas是一个基础模型式的神经互信息估计器，通过在大规模合成依赖模式数据上预训练，实现单次前向传播即可直接估计互信息，在保持最先进精度的同时获得100倍加速，并以单一统一模型灵活处理不同维度、样本量及真实世界场景。 |
| [^55] | [Proper Calibeating](https://arxiv.org/abs/2605.26703) | 本文将经典的“校准”与“校准击败”概念从二次评分规则推广到所有有界合适评分规则，证明校准蕴含合适校准但校准击败不必然蕴含合适校准击败，并提出总是合适的“完全校准击败”方法。 |
| [^56] | [Autoencoder-Based Parameter Estimation for Superposed Multi-Component Damped Sinusoidal Signals](https://arxiv.org/abs/2604.03985) | 提出一种基于自编码器的方法，利用潜在空间在含噪的叠加多分量阻尼正弦信号中高精度地估计每个分量的频率、相位、衰减时间和幅度，即使在信号快速衰减和存在次要分量等困难情况下依然有效。 |
| [^57] | [Optimal Learning Rate Schedules under Functional Scaling Laws: Power Decay and Warmup-Stable-Decay](https://arxiv.org/abs/2602.06797) | 本研究证明了在功能缩放律框架下最优学习率调度存在尖锐相变：简单任务应从训练伊始就采用幂衰减调度，困难任务则应采用预热-稳定-衰减（WSD）式调度，且两种情形下的衰减指数均由模型容量唯一决定。 |
| [^58] | [Exact Recovery by Neighborhood Smoothing in Directed Stochastic Block Models](https://arxiv.org/abs/2601.16427) | 本文提出一种基于邻域平滑的方法，通过聚类顶点的出边连接概率轮廓在稀疏有向随机块模型中实现精确社区恢复，并建立了非对称平滑估计量的有限样本逐行误差界以及精确恢复的分离条件。 |
| [^59] | [Basic Inequalities for First-Order Optimization with Applications to Statistical Risk Analysis](https://arxiv.org/abs/2512.24999) | 本文提出了一阶迭代优化算法的“基本不等式”统一框架，通过算法固有几何下的距离刻画目标函数值与任意参考点的差距，从而将隐式与显式正则化联系起来，并可广泛应用于统计风险分析。 |
| [^60] | [Sublinear Sketches for Approximate Nearest Neighbor and Kernel Density Estimation](https://arxiv.org/abs/2510.23039) | 本文提出了适用于动态数据流的新型草图算法，使近似最近邻搜索和近似核密度估计同时获得次线性空间与次线性查询时间的保证。 |
| [^61] | [Uncertainty-Aware Calibrated Clinical Text Classification with Large Language Models](https://arxiv.org/abs/2509.19375) | 该论文将闭集临床文本分类创新性地建模为无似然后验推断问题，把提示条件化的大语言模型视为类条件随机模拟器，并利用序贯蒙特卡罗近似贝叶斯计算获得诊断后验分布，从而同时实现准确预测与良好校准的不确定性估计，为临床决策提供可融合先验医学知识的可靠置信度。 |
| [^62] | [Are Targeted Data Poisoning Attacks as Effective as We Think?](https://arxiv.org/abs/2509.06896) | 本文提出仅利用干净模型信息识别数据集中最容易和最难被投毒的样本，主张应评估目标性投毒攻击在最坏情况下的真实效果而非平均成功率，并据此指导针对性的防御策略。 |
| [^63] | [N$^2$: A Unified Python Package and Test Bench for Nearest Neighbor-Based Matrix Completion](https://arxiv.org/abs/2506.04166) | 本文发布了N$^2$——一个模块化、可扩展的基于近邻方法的矩阵补全统一Python包与测试平台，并提出了一种在多个场景下达到最先进水平的新近邻变体，同时提供了覆盖医疗保健、推荐系统、因果推断和LLM评估等领域的真实数据集基准套件。 |
| [^64] | [Noise-Adaptive Conformal Classification with Marginal Coverage](https://arxiv.org/abs/2501.18060) | 本文提出了一种噪声自适应保形推断方法，能够有效处理随机标签噪声导致的可交换性偏差，在低质量标签场景下仍能生成具有严格边际覆盖保证的信息丰富预测集合。 |
| [^65] | [From Linear to Linearizable Optimization: A Novel Framework with Applications to Stationary and Non-stationary DR-submodular Optimization](https://arxiv.org/abs/2405.00065) | 本文提出上可线性化/可二次化函数的新框架及通用元算法，将线性/二次最大化算法统一转换为凹优化和DR-次模优化算法，并在多种反馈设置下获得了动态和自适应遗憾保证，改进了现有最优结果。 |
| [^66] | [Prelimit Coupling and Steady-State Convergence of Constant-stepsize Nonsmooth Contractive SA](https://arxiv.org/abs/2404.06023) | 该论文提出预极限耦合技术，证明了常数步长非光滑压缩随机逼近（含Q-learning）的稳态收敛性，并发现其渐近偏差与步长的平方根成正比，从而可利用Richardson-Romberg外推法有效减少偏差。 |
| [^67] | [Approximation of RKHS Functionals by Neural Networks](https://arxiv.org/abs/2403.12187) | 本文研究了使用神经网络逼近再生核希尔伯特空间（RKHS）上的函数型，并建立了逼近的普适性，推导了逆多重二次、高斯和Sobolev核引起的误差界限，证明神经网络可以准确逼近广义函数线性模型中的回归映射。 |
| [^68] | [Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces](https://arxiv.org/abs/2402.04691) | 本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。 |
| [^69] | [Unfair Utilities and First Steps Towards Improving Them.](http://arxiv.org/abs/2306.00636) | 该论文提出了一个新的公平框架——考虑政策优化哪个效用，定义了信息价值公平，提出不应使用不满足这一标准的实用程序，并探讨了修改实用程序以满足此公平标准可能对最优政策产生的影响。 |

# 详细

[^1]: 淬火系综采样

    Quenched Ensemble Sampling

    [https://arxiv.org/abs/2609.15894](https://arxiv.org/abs/2609.15894)

    该论文提出淬火系综采样方法，通过将嵌套采样中的硬能量约束推广为能量边界处的一族排斥势，在保持能量单调递减路径的同时实现可扩展的基于梯度的采样，从而能够跨越一阶相变准确估计边际似然并抽取后验样本。

    

    从物理系统能量函数中进行采样的一些最严峻的挑战出现在相变处，此时态密度发生突变，许多采样算法会陷入停滞。嵌套采样是一种在硬能量约束下遍历态密度的粒子方法，已知其对这类相变具有鲁棒性，但其在高维中的应用受到了在该约束下进行采样的困难所限制。在本工作中，我们提出了淬火系综采样，该方法将硬约束推广为能量边界处的一族排斥势。这在保持能量单调递减的淬火路径的同时，使受约束的目标分布能够适用于可扩展的基于梯度的采样核。我们在相变的合成模型上证明了，我们的方法能够在一阶相变过程中估计边际似然并抽取后验样本，而流行的替代方法在此情形下则会失效。

    arXiv:2609.15894v1 Announce Type: cross  Abstract: Some of the sharpest challenges in sampling from the energy functions of physical systems arise at phase transitions, where the density of states changes abruptly and many sampling algorithms stall. Nested sampling is a particle method that traverses the density of states under a hard energy constraint and is known to be robust to such transitions, but its application in high dimension is limited by the difficulty of sampling under that constraint. In this work we introduce Quenched Ensemble Sampling, which generalises the hard constraint to a family of repulsive potentials at the energy boundary. This preserves the quenched path of monotonically decreasing energy while making the constrained target amenable to scalable gradient-based kernels. We demonstrate on synthetic models of phase transitions that our method estimates the marginal likelihood and draws posterior samples across a first-order transition where popular alternatives su
    
[^2]: 谱表示学习的锐利速率与一行修正

    Sharp Rates and a One-Line Correction for Spectral Representation Learning

    [https://arxiv.org/abs/2609.15825](https://arxiv.org/abs/2609.15825)

    该论文推翻了支撑谱表示学习的各向同性假设，指出迁移风险仅取决于任务协方差向依赖算子主导奇异方向的压缩是否与算子谱一致排序，并据此给出了匹配的锐利收敛速率和一行修正方法。

    

    一个自监督编码器只需训练一次、冻结，之后便可通过轻量级探针在训练时未被指定的各种任务上重复使用；实践者关心的问题是：现成的特征何时足够好，何时需要修正。典型相关分析（CCA）、HGR最大相关以及谱对比损失的总体最优解都会返回交叉视图依赖算子的前 $k$ 个奇异子空间，其理论依据是各向同性：如果任务先验没有方向偏好，那么该子空间是普遍最优的。我们证明各向同性是一个错误的假设。任务先验仅通过任务协方差 $\Lambda=\mathbb{E}[\Delta\Delta^\top]$ 进入迁移风险，而且仅通过其向算子主导奇异方向的压缩进入；关键不在于 $\Lambda$ 是否各向同性，而在于其偏好方向是否与算子的谱一致地排序。我们证明了匹配的双边收敛速率——

    arXiv:2609.15825v1 Announce Type: new  Abstract: A self-supervised encoder is trained once, frozen, and reused through lightweight probes on tasks nobody named at training time; the practitioner's question is when the off-the-shelf features are good enough and when they need fixing. Canonical correlation analysis, HGR maximal correlation, and the population optimum of the spectral contrastive loss all return the top-$k$ singular subspace of a cross-view dependence operator, justified by isotropy: if the task prior has no directional preference, that subspace is universally optimal. We show isotropy is the wrong hypothesis. The prior enters the transfer risk only through the task covariance $\Lambda=\mathbb{E}[\Delta\Delta^\top]$, and only through its compression onto the operator's leading singular directions; what matters is not whether $\Lambda$ is isotropic but whether its preferred directions are ordered consistently with the operator's spectrum. We prove matching two-sided rates--
    
[^3]: 目标偏移下的学习：最优密度比估计与重要性加权回归

    Learning under Target Shift: Optimal Density Ratio Estimation and Importance-Weighted Regression

    [https://arxiv.org/abs/2609.15785](https://arxiv.org/abs/2609.15785)

    针对连续输出的目标偏移学习问题，提出RKHS中的谱正则化密度比估计方法，首次给出高概率有限样本收敛保证，并借此实现最优的重要性加权回归。

    

    我们研究了连续输出情形下目标偏移的密度比估计与重要性加权回归问题。在目标偏移下，给定输出时输入的条件分布在训练分布与测试分布之间保持不变，而输出的边际分布可能发生变化。尽管该问题在离散输出情形下已被广泛研究，但连续输出的设定在很大程度上仍未被充分理解：重要性权重由一个未知的密度比函数决定，而现有的估计方法缺乏明确的有限样本收敛速率。我们提出了一种在再生核希尔伯特空间（RKHS）中的谱正则化方法，利用带标签的训练样本和无标签的测试输入来估计连续密度比。在正则性参数 $\iota>0$ 的源条件下，我们建立了高概率有限样本保证，并证明该估计器达到了最优……

    arXiv:2609.15785v1 Announce Type: cross  Abstract: We study density ratio estimation and importance-weighted regression under target shift with continuous outputs. Under target shift, the conditional distribution of the inputs given the outputs remains invariant across the training and test distributions, while the output marginal distribution may change. Although this problem has been extensively studied for discrete outputs, the continuous setting is substantially less understood: the importance weights are determined by an unknown density ratio function, for which existing estimation methods lack explicit finite-sample convergence rates. We propose a spectral regularization method in a reproducing kernel Hilbert space (RKHS) for estimating the continuous density ratio from labeled training samples and unlabeled test inputs. Under a source condition with regularity parameter $\iota>0$, we establish high-probability finite-sample guarantees and show that the estimator achieves the cap
    
[^4]: 语言模型水印检测的预测似然比

    Predictive Likelihood Ratios for Language Model Watermark Detection

    [https://arxiv.org/abs/2609.15657](https://arxiv.org/abs/2609.15657)

    该论文提出基于预测似然比的语言模型水印检测方法，通过对不确定的亏缺和尾部分布进行先验平均实现跨信号设定的稳健检测功效，并证明所得贝叶斯因子同时是检验鞅，使第一类错误控制在备择假设误设及可选停止下依然有效。

    

    密钥水印检测通过检验观测到的token与由密钥重建的伪随机变量之间的依赖关系来工作。基于Li等人（2025）的主枢轴（pivotal）框架，我们构建了预测似然比，对不确定的概率亏缺（probability deficits）和残差尾部分布进行平均。其目标是在不同的规格设定下获得稳健的检测功效，而无需针对单一信号强度进行调优。混合先验将尾部形状与有效宽度相结合；分层扩展则允许亏缺或宽度在同一文档内部发生变化。该检验在固定显著性水平下最大化先验平均功效，但通常并非一致最优（uniformly most powerful）或极小极大最优。在精确条件枢轴原假设下，在每次观测前选定的归一化预测备择假设会产生一个贝叶斯因子，该因子同时也是一个检验鞅：第一类错误的控制不受备择假设误设的影响，并且在可选停止条件下依然保持有效。

    arXiv:2609.15657v1 Announce Type: cross  Abstract: Keyed watermark detection tests dependence between observed tokens and pseudorandom variables reconstructed from a secret key. Building on the pivotal framework of Li et al. (2025), we construct predictive likelihood ratios that average over uncertain probability deficits and residual-tail distributions. The aim is robust detection power across alternative specifications without requiring a single signal-strength tuning. A mixture prior combines tail shape and effective width; hierarchical extensions allow within-document variation in deficit or width. The test maximizes prior-averaged power at a fixed size, but is not generally uniformly most powerful or minimax. Under the exact conditional pivot null, normalized predictive alternatives selected before each observation yield a Bayes factor that is also a test martingale: Type I error control is unaffected by alternative misspecification and remains valid under optional stopping. This 
    
[^5]: 高维非平稳时间序列中多重同时变点的检测研究

    On Detecting Multiple Simultaneous Change-points in High Dimensional Non-Stationary Time Series

    [https://arxiv.org/abs/2609.15479](https://arxiv.org/abs/2609.15479)

    本文提出基于标准与自适应融合组lasso方法来检测高维非平稳经济金融时间序列中的多重同时变点，证明了该方法的L_2一致性与L_0一致性，并量化了实现一致性检测所需的条件，最后通过美国过去50年的经济金融数据验证了其有效性。

    

    本文研究高维非平稳经济与金融时间序列数据中多重同时（系统性）变点的检测问题。所采用的分析框架基于标准及自适应融合组lasso方法，其中混合L_{2,1}惩罚项采用统一权重或由数据依赖的权重进行重新加权。本文证明，在适当条件下，该方法具有L_2一致性，并且通过采用数据依赖的权重，能够以趋于1的概率正确选择变点（L_0一致性）。文章还量化了结构变化的最小平均幅度、变点数量与观测数量之间的相互作用条件，以实现变点的一致性发现。该方法的有效性通过分析过去50年美国经济与金融时间序列的大型面板数据得到展示。

    arXiv:2609.15479v1 Announce Type: cross  Abstract: This paper studies the detection of multiple simultaneous (systematic) change points for high-dimensional nonstantionary economic and financial time series data. The analytic framework used is based on the standard and adaptive fused group lasso method, where the mixed L_{2,1} penalty is either uniform or re-weighted by data-dependent weights. This paper shows that, under appropriate conditions, this approach is L_2 consistent and, by adopting the data-dependent weights, could correctly select the change points with probability approaching unity (L_0 consis- tency). It quantifies the conditions on the interplay among the averaged minimum magnitude of structural changes, the number of change points and the number of observations for consistently discovering the change points. The performance of this approach is illustrated via an analysis of a large panel of U.S. economic and financial time series data over the past 50 years.
    
[^6]: 监督图预测的图匹配松弛与摊销

    Graph Matching Relaxations and Amortization for Supervised Graph Prediction

    [https://arxiv.org/abs/2609.15437](https://arxiv.org/abs/2609.15437)

    该论文证明了Gromov-Wasserstein目标是监督图预测中最合适的图匹配松弛形式，并提出基于可微Sinkhorn算法的参数化匹配器来摊销图匹配问题，实现图预测模块与匹配器的联合学习。

    

    监督图预测（SGP）的端到端训练需要一个置换不变的损失函数来比较具有任意节点排序的预测图和目标图。这类损失函数通常涉及一个代价高昂的图匹配问题。我们首先研究了该问题的三种最优传输（Optimal Transport）松弛形式，并从理论和实证上表明，Gromov-Wasserstein（GW）目标最适合于监督图预测。随后，为了避免为每个训练样本求解由此产生的内层优化问题，我们提出对图匹配（节点对齐）问题进行摊销。对于每个训练样本，损失函数利用由参数化匹配器提供的传输计划，该匹配器基于应用于经验节点分布的可微Sinkhorn算法构建。图预测模块和匹配器被联合学习。我们在复杂度递增的玩具和现实世界监督图预测问题上展示了该方法的有效性，其中包括一个新颖的质谱到分子骨架（Mass-spectra to Scaffold）预测任务。

    arXiv:2609.15437v1 Announce Type: cross  Abstract: End-to-end Supervised Graph Prediction (SGP) requires a permutation-invariant loss to compare predicted and target graphs with arbitrary node orderings. Such losses typically involve a costly graph-matching problem. We first study three Optimal Transport relaxations of this problem and show, theoretically and empirically, that the Gromov-Wasserstein (GW) objective is the most suitable for SGP. Then, to avoid solving the resulting inner optimization for every training example, we propose to amortize the graph matching (node alignment) problem. For each training sample, the loss function leverages a transport plan provided by a parametric matcher based on the differentiable Sinkhorn algorithm applied on empirical node distributions. The graph prediction module and the matcher are jointly learned. We showcase the efficiency of this approach on toy and real world SGP problems of increasing complexity including a novel Mass-spectra to Scaff
    
[^7]: ReLU神经网络对光滑泛函算子的逼近：维度衰减与误差分析

    ReLU Neural Network Approximation to Smooth Functional Operator: Dimensional Decay and Error Analysis

    [https://arxiv.org/abs/2609.15355](https://arxiv.org/abs/2609.15355)

    本文通过结合坐标截断、各向异性分割与局部泰勒逼近的构造性分析，证明在广义指数坐标衰减条件下，深度ReLU神经网络对无穷维希尔伯特空间上光滑泛函的一致逼近误差上界与基于伪维数的下界相匹配。

    

    我们研究了深度ReLU神经网络对无穷维可分离希尔伯特空间上光滑标量值泛函的一致逼近。将泛函输入写为 $X(t)=\sum_{d\geq1}\xi_d\nu_d(t)$，我们通过 $w_ds_d$ 来量化坐标 $d$ 的重要性，其中 $s_d$ 界定相应基函数得分的幅值，$w_d$ 控制目标泛函的方向性Fréchet敏感度。我们的构造性分析结合了坐标截断、各向异性分割、局部泰勒逼近和ReLU网络实现，同时允许保留坐标之间的无限制交互作用。我们建立了一致逼近误差的一般非渐近上界，以及最坏情形逼近误差的基于伪维数的补充下界。在广义指数坐标衰减 $w_ds_d\asymp\exp(-cd^\rho)$（其中 $\rho>0$）的条件下，上界与下界相匹配……

    arXiv:2609.15355v1 Announce Type: cross  Abstract: We study the uniform approximation of smooth scalar-valued functionals on an infinite-dimensional separable Hilbert space by deep ReLU neural networks. Writing the functional input as $X(t)=\sum_{d\geq1}\xi_d\nu_d(t)$, we quantify the importance of coordinate $d$ through $w_ds_d$, where $s_d$ bounds the magnitude of the corresponding basis score and $w_d$ controls the directional Fr\'echet sensitivity of the target functional. Our constructive analysis combines coordinate truncation, anisotropic partitioning, local Taylor approximation, and ReLU network realization, while allowing unrestricted interactions among the retained coordinates. We establish a general nonasymptotic upper bound for the uniform approximation error and a complementary pseudo-dimension-based lower bound for the worst-case approximation error. Under generalized exponential coordinate decay $w_ds_d\asymp\exp(-cd^\rho)$, with $\rho>0$, the upper and lower bounds matc
    
[^8]: 网络干扰下的共形个体处理效应估计

    Conformal Individual Treatment Effect Estimation under Networked Interference

    [https://arxiv.org/abs/2609.15254](https://arxiv.org/abs/2609.15254)

    本文提出一种干扰调整的加权共形预测方法，通过构建共形p值的可观测上界来处理网络干扰，为反事实结果和个体处理效应提供有限样本边际覆盖保证。

    

    共形反事实预测在无干扰假设下为反事实结果和个体处理效应构建具有有限样本覆盖保证的预测集。在本工作中，我们放宽了这一假设，允许每个单元的潜在结果依赖于其他单元的处理和协变量。在这种设置下，倾向得分重加权无法恢复加权可交换性，现有方法可能无法实现有效覆盖。为了解决这一问题，我们开发了干扰调整的加权共形预测方法，通过在目标干预下构建理想的、未观测的共形p值的可观测上界来考虑干扰。所得到的预测集在直推式和归纳式两种设置中，均为反事实结果和个体处理效应提供了有限样本边际覆盖保证。当（摘要在此处截断）我们还推导出了更精确的构造。

    arXiv:2609.15254v1 Announce Type: cross  Abstract: Conformal counterfactual prediction constructs prediction sets with finite-sample coverage guarantees for counterfactual outcomes and individual treatment effects under the no-interference assumption. In this work, we relax this assumption by allowing each unit's potential outcomes to depend on other units' treatments and covariates. In this setting, propensity-score reweighting does not restore weighted exchangeability, and existing methods may fail to achieve valid coverage. To address this issue, we develop interference-adjusted weighted conformal prediction that accounts for interference by constructing an observable upper bound on the ideal and unobserved conformal $p$-value under the target intervention. The resulting prediction sets provide finite-sample marginal coverage guarantees for counterfactual outcomes and individual treatment effects in both transductive and inductive settings. We also derive a sharper construction when
    
[^9]: 带探测的多臂老虎机：最优遗憾与赢家反馈的极限

    Bandits with Probing: Optimal Regret and the Limits of Winner Feedback

    [https://arxiv.org/abs/2609.15248](https://arxiv.org/abs/2609.15248)

    本文针对带探测的多臂老虎机问题首次确定了两个极小极大最优遗憾定律：在赢家反馈下，独立随机奖励情形的遗憾阶为 Φ_{n,k}(T)，而一般联合独立同分布奖励及固定序列情形的遗憾阶为 R_{n,k}(T)，并给出了匹配的任意时刻上界。

    

    学习者在每轮最多探测 n 个臂中的 k 个，接收它们在 [0,1] 区间内奖励的最大值，并与最优固定臂进行竞争。那么，探测的优势何时能为学习“买单”？我们确定了两个极小极大定律。在具有赢家反馈（最大奖励值及获胜臂标签）的独立随机奖励下，或者在给定块最大值之间单个带符号对比的任意固定序列下，极小极大遗憾的阶为 Φ_{n,k}(T)=min{(n-k)T/n, (n-k)/k}，其中 2≤k<n。在赢家反馈下，任意联合独立同分布奖励和固定序列的极小极大遗憾阶均为 R_{n,k}(T)=(n-k)/n · min{T, (n+T)/k, √(nT/k)}。这两个定律均包含普适常数和任意时刻有效上界。第一个定律将遗憾归结为纯粹的覆盖成本：同轮对比吸收了稳定性成本，而独立性允许精确重采样，其收益用于推动样本的进一步获取。第二个定律则额外引入了一个学习成本……

    arXiv:2609.15248v1 Announce Type: new  Abstract: A learner probes at most $k$ of $n$ arms each round, receives the maximum of their rewards in $[0,1]$, and competes with the best fixed arm. When does the probing advantage pay for learning? We determine two minimax laws. Under independent stochastic rewards with winner feedback (the maximum and a winning label), or on arbitrary fixed sequences given a single signed contrast between block maxima, the minimax regret has order $\Phi_{n,k}(T)=\min\{\frac{n-k}{n}T,\frac{n-k}{k}\}$, $2\le k<n$. Under winner feedback, both arbitrary joint i.i.d. rewards and fixed sequences have minimax regret of order $R_{n,k}(T)=\frac{n-k}{n}\min\{T,\frac{n+T}{k},\sqrt{\frac{nT}{k}}\}$. Both laws have universal constants and anytime upper bounds. The first reduces regret to a pure coverage cost: same-round contrasts absorb the stability cost, and independence permits exact resampling whose gains fund sample advancement. The second adds a learning cost that be
    
[^10]: 结构化特征过拟合之处，随机特征却出现顿悟

    Structured Features Overfit Where Random Features Grok

    [https://arxiv.org/abs/2609.15047](https://arxiv.org/abs/2609.15047)

    本文揭示了“顿悟”现象的边界条件——随机高斯特征映射上的岭回归会呈现先记忆后泛化的延迟，而结构化的带限傅里叶特征映射则不会出现这种延迟，其性能退化由活跃支撑集的尖锐边界决定而非插值效应。

    

    Xu、Vardi和Safran（ICML 2026）证明了在非结构化随机高斯特征映射上的过参数化岭回归会出现“顿悟”（grokking）现象，其中记忆与泛化之间的延迟随权重衰减λ以1/λ的速度增长。我们表明，在结构化特征映射上同样的延迟并不会出现。对于$\mathbb{Z}_p^2$上的带限傅里叶特征映射，当单字符目标位于可表达类之内时，在固定正权重衰减下扩大带宽会使峰值保留集精度从1.00单调下降至0.07，在整个扫描过程中不存在任何“先记忆后泛化”的阶段。这种退化并非插值效应：它发生在容量比q/n = 0.638处，远低于插值阈值，且其成因与超过阈值后出现的精确零空间有所不同。真正具有尖锐边界的是活跃支撑集。在保持名义维度固定并将频带遮蔽回108（摘要在此处截断）

    arXiv:2609.15047v1 Announce Type: new  Abstract: Xu, Vardi and Safran (ICML 2026) prove that over-parameterized ridge regression over an unstructured random Gaussian feature map groks, with the delay between memorization and generalization growing as $1/\lambda$ in the weight decay. We show that on a structured feature map the same delay does not appear. For a band-limited Fourier feature map over $\mathbb{Z}_p^2$ carrying a single-character target that lies inside the expressible class, enlarging the band at fixed positive weight decay drives peak held-out accuracy monotonically from $1.00$ to $0.07$, with no memorize-then-generalize regime anywhere along the sweep. The degradation is not an interpolation effect. It sets in at capacity ratio $q/n = 0.638$, far below the interpolation threshold, on separate grounds from the exact null space that appears above it. What does have a sharp boundary is the active support. Holding the nominal dimension fixed and masking the band back to $108
    
[^11]: 通过影响矩阵估计实现大规模数据归因

    Data Attribution at Scale via Influence Matrix Estimation

    [https://arxiv.org/abs/2609.15044](https://arxiv.org/abs/2609.15044)

    该论文将预算受限的数据归因问题转化为从少量测量中估计大型影响矩阵的问题，并据此提出了MAGE和SPEL两种算法，大幅降低了大规模数据归因的计算成本。

    

    数据归因旨在量化单个训练样本如何塑造模型的预测，并支撑包括数据估值、机器遗忘和模型可解释性在内的一系列问题。尽管已有大量研究工作，但由于神经网络的非凸性质，计算上可扩展的方法往往难以准确预测移除训练数据所产生的影响。为克服这一挑战，基于元梯度的方法（如MAGIC (Ilyas and Engstrom, 2025)）通过整个训练过程对每个预测进行微分，并计算其相对于训练数据的精确影响，但需要对每个预测单独运行一次完整的训练。为降低这一成本，我们将预算受限的数据归因问题转化为从少量测量中估计一个大型影响矩阵。我们证明，最适合恢复该矩阵的测量方式与最适合归因本身的测量方式有所不同。随后，我们提出了两种算法，MAGE和SPEL。

    arXiv:2609.15044v1 Announce Type: cross  Abstract: Data attribution seeks to quantify how individual training examples shape a model's predictions and underpins problems including data valuation, machine unlearning, and model interpretability. Despite having a long line of work, computationally scalable methods often struggle to predict the effect of removing training data in neural networks due to their non-convex nature. To overcome this challenge, metagradient-based methods such as MAGIC (Ilyas and Engstrom, 2025) differentiate each prediction through the entire training run and compute its exact influence with respect to the training data, but require a separate run for every prediction. To reduce this cost, we cast budgeted attribution as estimating a large influence matrix from a small number of measurements. We show that the measurements most appropriate for recovering this matrix differ from those best suited for attribution itself. We then present two algorithms, MAGE and SPEL
    
[^12]: 网络上复合单调包含问题的分布式快速不动点算法

    Distributed Fast Fixed-Point Algorithms for Composite Monotone Inclusions over Networks

    [https://arxiv.org/abs/2609.14953](https://arxiv.org/abs/2609.14953)

    本文提出了两种将 Nesterov 型加速与原始-对偶技术相结合的去中心化快速不动点算法（ND-DFFP 和 NI-DFFP），用于在网络中各智能体算子信息私有的条件下求解复合单调包含问题，并首次在该问题类上给出了原始空间中的精确收敛速率保证。

    

    本文旨在开发新的高效分布式算法，用于在一个由 $n$ 个智能体组成的连通网络上求解一类单调包含问题 $0 \in \sum_{i=1}^n (G_ix + T_ix)$，其中单值算子 $G_i$ 和可能为多值的算子 $T_i$ 对智能体 $i$ 保持私有。针对该问题类，现有的分布式算法主要属于非加速类型，且其在原始（primal）空间中的精确收敛速率在很大程度上尚未被探索。为弥补这一空白，我们提出了两种去中心化快速不动点算法 \texttt{ND-DFFP} 和 \texttt{NI-DFFP}，它们在两种典型设定下将 Nesterov 型加速与原始-对偶技术相结合：(i) $G_i$ 的 Lipschitz 连续性以及 $G_i+T_i$ 的极大单调性；(ii) $G_i$ 的余强制性（co-coercivity）以及 $T_i$ 的极大单调性。其中 \texttt{ND-DFFP} 采用同质的依赖网络的步长，而 \texttt{NI-DFFP} 则……（摘要在此处被截断）

    arXiv:2609.14953v1 Announce Type: cross  Abstract: This paper aims to develop new and efficient distributed algorithms for solving a class of monotone inclusions, $0 \in \sum_{i=1}^n (G_ix + T_ix)$, over a connected network of $n$ agents, where the single-valued operator $G_i$ and the possibly multivalued operator $T_i$ remain private to agent $i$. Existing distributed algorithms for this problem class are primarily non-accelerated, and their exact convergence rates in the original primal space are largely unexplored. To bridge this gap, we propose two Decentralized Fast Fixed-Point-based algorithms, \texttt{ND-DFFP} and \texttt{NI-DFFP}, which integrate Nesterov-type acceleration with primal-dual techniques under two prominent settings: (i) \textit{Lipschitz continuity of $G_i$ and maximal monotonicity of $G_i+T_i$}; and (ii) \textit{co-coercivity of $G_i$ and maximal monotonicity of $T_i$}. While \texttt{ND-DFFP} utilizes a homogeneous network-dependent stepsize, \texttt{NI-DFFP} ref
    
[^13]: 随机逼近的稳态收敛性

    Steady-State Convergence of Stochastic Approximation

    [https://arxiv.org/abs/2609.14922](https://arxiv.org/abs/2609.14922)

    本文为马尔可夫乘性噪声驱动的固定步长收缩型随机逼近建立了统一的稳态收敛理论，通过多步普适性框架突破了对独立同分布/加性噪声和全局可微性的限制，并以最优速率获得了缩放稳态的高斯近似。

    

    对于固定步长的随机逼近（SA），迭代点列在分布上收敛于一个依赖于步长 α 的平稳分布。稳态收敛（SSC）研究当 α ↓ 0 时缩放后平稳分布的极限。现有的 SSC 理论要求噪声为独立同分布或加性噪声，且要求均值算子全局可微，所得到的收敛速率也是次优的。本文为马尔可夫乘性噪声驱动的固定步长收缩型随机逼近建立了一个统一的 SSC 理论，同时涵盖了局部可微和局部不可微的均值算子。一个关键的方法学贡献是多步普适性框架，该框架在保持稳态极限的同时，逐步将原始随机递归约化为易于处理的辅助动力学。在不动点处局部二次线性化的条件下，我们以最优速率 O(√α…) 获得了缩放后稳态的高斯近似。

    arXiv:2609.14922v1 Announce Type: cross  Abstract: For constant-stepsize stochastic approximation (SA), the iterates converge in distribution to a stationary law that depends on the stepsize $\alpha.$ Steady-state convergence (SSC) concerns the limit of the scaled stationary distribution as $\alpha \downarrow 0.$ Existing SSC theory requires i.i.d. or additive noise and global differentiability of the mean operator, and yields suboptimal rates. We develop a unified SSC theory for constant-stepsize contractive SA driven by Markovian, multiplicative noise, covering both locally differentiable and locally nondifferentiable mean operators. A key methodological contribution is a multi-step universality framework that progressively reduces the original stochastic recursion to tractable auxiliary dynamics while preserving its steady-state limit. Under local quadratic linearization at the fixed point, we obtain a Gaussian approximation of the scaled steady state at the optimal rate $O(\sqrt{\a
    
[^14]: 面向具有分块缺失特征的多站点数据的Shapley值估计

    Shapley Value Estimation for Multi-Site Data with Blockwise-Missing Features

    [https://arxiv.org/abs/2609.14902](https://arxiv.org/abs/2609.14902)

    该论文揭示了在分块缺失的多站点数据中直接插补再计算Shapley值会引入依赖联盟的系统性偏差，并提出FUSHAP方法，利用部分观测的辅助站点数据来降低估计方差、实现更可靠的特征归因。

    

    基于Shapley值（SV）的方法是机器学习中特征归因的主流框架，然而现有的人群级Shapley估计量通常假设用于评估联盟博弈的观测数据在同一特征空间下被完整观测。这一假设在生物医学、社会科学和环境监测等领域的多站点研究中经常被打破，因为不同机构在不同协议下记录不同的特征，从而在各数据源之间产生了系统性的分块缺失。我们首先证明，在计算Shapley值之前对缺失特征进行插补这一标准处理方式，会给最终的特征归因引入系统性的、依赖于联盟的偏差。随后我们提出了FUSHAP（基于部分观测数据的融合Shapley归因，Fusion Shapley Attribution from Partially-observed data），该方法利用部分观测的辅助站点数据来降低初步单一（数据源）估计的方差……

    arXiv:2609.14902v1 Announce Type: cross  Abstract: Shapley value (SV)-based methods are the prevailing framework for feature attribution in machine learning, yet existing population-level Shapley estimators generally assume that observations used to evaluate the coalitional game are fully observed under a common feature space. This assumption is routinely violated in multi-site studies across biomedicine, social science, and environmental monitoring, where institutions record different features under different protocols, producing systematic blockwise missingness across sources. We first show that the standard remedy of imputing missing features before computing Shapley values introduces systematic, coalition-dependent bias into the resulting attributions. We then propose \textbf{FUSHAP} (\textbf{Fu}sion \textbf{Sh}apley \textbf{A}ttribution from \textbf{P}artially-observed data), a method that leverages partially-observed auxiliary sites to reduce the variance of a preliminary single-
    
[^15]: 一种用于双重惩罚正则化多元函数主成分分析的函数奇异值分解框架

    A Functional SVD Framework for Regularized Multivariate Functional PCA with Dual Penalization

    [https://arxiv.org/abs/2609.14815](https://arxiv.org/abs/2609.14815)

    本文提出了一种基于函数奇异值分解的正则化多元函数主成分分析新框架，通过对函数主成分及主成分得分（含稀疏性惩罚）进行双重正则化，突破了现有方法仅惩罚函数主成分的局限，显著增强了结果的可解释性。

    

    本文介绍了一种通过函数奇异值分解（SVD）实现正则化多元函数主成分分析（ReMFPCA）的新框架。所提出的方法通过在希尔伯特空间框架内引入广义函数SVD，扩展了现有的多元函数主成分分析（MFPCA）方法，能够同时对函数主成分（PC）及其相关的主成分得分进行正则化。该框架的一个关键创新是在主成分得分上引入稀疏性惩罚，通过滤除无关的个体特异性变异来增强可解释性。这种双重惩罚策略相较于现有的仅对函数主成分进行惩罚的基于协方差的特征分解方法，是一个重大进步。文章提出了两种幂算法实现方式——顺序式和联合式，并提出了基于迭代回归的交叉验证方法，用于选择最优平滑参数。

    arXiv:2609.14815v1 Announce Type: cross  Abstract: This paper introduces a novel framework for Regularized Multivariate Functional Principal Component Analysis (ReMFPCA) via Functional Singular Value Decomposition (SVD). The proposed method extends existing MFPCA approaches by incorporating a generalized functional SVD within a Hilbert space framework, enabling simultaneous regularization of both functional principal components (PCs) and their associated PC scores. A key innovation of this framework is the inclusion of a sparsity penalty on the PC scores, which enhances interpretability by filtering out irrelevant subject-specific variations. This dual-penalization strategy represents a significant advancement beyond existing covariance-based eigen decomposition methods, which penalize only the functional PCs. Two power algorithm implementations, sequential and joint, are proposed, together with a cross-validation approach based on iterative regression for optimal smoothing parameter s
    
[^16]: 从矩阵求逆到约束：标签偏移下重要性权重的可证明更紧的置信区域

    From matrix inversion to constraints: provably tighter confidence regions for importance weights in label shift

    [https://arxiv.org/abs/2609.14802](https://arxiv.org/abs/2609.14802)

    提出从矩阵求逆转向直接矩阵约束的新框架，为标签偏移下重要性权重的估计提供可证明更紧的置信区域，同时保持精确的有限样本有效性。

    

    重要性权重在标签偏移下的领域自适应中至关重要，但其效用常常受到与估计相关的有限样本不确定性的削弱。现有方法通常通过对区间值线性系统进行高斯消元来分析这种不确定性，这会导致过于保守的置信区域和低效的下游应用。我们提出了一种从基于求逆的推断到直接矩阵约束框架的范式转变。我们利用该框架定义联合置信区域，并通过线性规划提取边际区间，为重要性权重推导出可证明更紧的边界，同时保持精确的有限样本有效性。此外，我们分析了置信区域的几何结构，并提供了其直径边界的理论结果。我们在文本、图像、多模态基准测试上进行了评估，包括AGNews、MNIST、CIFAR-10、N24News以及真实世界的自动驾驶……（原文摘要在此处截断）

    arXiv:2609.14802v1 Announce Type: cross  Abstract: Importance weights are essential in domain adaptation under label shift, yet their utility is often undermined by the finite sample uncertainty associated with their estimation. Existing methods typically analyze this uncertainty through Gaussian elimination on interval-valued linear systems, which leads to overly conservative confidence regions and inefficient downstream applications. We propose a paradigm shift from inversion-based inference to a direct matrix constraint framework. We use this framework to define a joint confidence region and extract marginal intervals via linear programming, deriving provably tighter bounds for importance weights while maintaining exact finite-sample validity. Furthermore, we analyze the confidence region's geometry and provide the theoretical results for its diameter bounds. Evaluated across text, image, multimodal benchmarks, including AGNews, MNIST, CIFAR-10, N24News, and a real-world autonomous 
    
[^17]: 非SCAR假设下的PU分类：基于聚类辅助的过采样增强逻辑模型

    PU classification under Non-SCAR: clustering-assisted logistic model with oversampling enhancement

    [https://arxiv.org/abs/2609.14675](https://arxiv.org/abs/2609.14675)

    本论文的主要创新是在SCAR假设不成立的PU分类场景中，将SMOTE过采样技术与基于聚类辅助的逻辑回归方法（含严格和非严格Lasso正则化）相结合，以缓解类别不平衡并显著提升分类性能。

    

    本研究解决了在SCAR假设不成立情况下的正例-未标注（PU）分类问题。我们研究了基于逻辑回归的方法，即聚类方法及其采用严格与非严格Lasso正则化的扩展形式。本研究的主要贡献是将SMOTE技术融入其中以缓解类别不平衡问题，并系统评估其对所考虑算法性能的影响。SMOTE首先被应用于重新平衡训练数据集；接着，通过2-means（二均值）聚类获得清洗后的标签；然后在清洗后的数据上训练逻辑回归模型，其中被识别出的正例实例用额外的真实正例进行增强，其余观测则被当作负例处理。实验评估在13个真实基准数据集和1个合成数据集上进行，并与朴素方法和Spy-EM方法进行了对比。结果表明，融入（摘要在此处截断）

    arXiv:2609.14675v1 Announce Type: cross  Abstract: This study addresses the PU classification problem under violations of the SCAR assumption. We investigate logistic regression-based approaches, namely the cluster method and its extensions with strict and non-strict Lasso regularization. The primary contribution of this work is the integration of the SMOTE technique to alleviate class imbalance and systematically assess its impact on the performance of the considered algorithms. SMOTE is first applied to rebalance the training dataset. Next, cleaning labels are derived via 2-means clustering. Logistic regression is then trained on the cleaned data, where identified positive instances are augmented with additional true positives and the remaining observations are treated as negative. The experimental evaluation is conducted on 13 real benchmark datasets and one synthetic dataset. For comparison, we include the naive approach and the Spy-EM method. The results demonstrate that incorpora
    
[^18]: 面向扩散逆问题的直接条件转移采样

    Direct Conditional Transition Sampling for Diffusion Inverse Problems

    [https://arxiv.org/abs/2609.14596](https://arxiv.org/abs/2609.14596)

    本文提出DCTS方法，通过沿短内路径估计测量条件下的干净均值并将高斯源噪声直接传输到下一含噪状态，替代昂贵的概率流ODE积分与MCMC，在四个逆问题上实现竞争性重建质量且速度提升高达16.8倍。

    

    免训练的扩散逆问题求解器通常需要在局部测量引导与代价高昂的干净空间后验更新之间做出选择。独立后验刷新可以通过采样一个以测量为条件的干净样本并对其重新加噪来改善全局校正，但其实际实现需要概率流ODE积分和干净空间的马尔可夫链蒙特卡罗（MCMC）。我们提出了直接条件转移采样（DCTS），这是对同一理想刷新目标的一种直接随机流近似。DCTS不是显式地抽取干净样本，而是沿着一条短的内路径估计以测量为条件的干净均值，并将高斯源噪声直接传输到下一个含噪状态。去噪器兼容的充分统计量和协方差缩放的算子更新使得这种条件均值估计成为可能。在四个逆问题上的实验表明，DCTS在实现具有竞争力的重建质量的同时，速度提升高达16.8倍。

    arXiv:2609.14596v1 Announce Type: cross  Abstract: Training-free diffusion inverse solvers typically choose between local measurement guidance and costly clean-space posterior updates. Independent posterior refresh can improve global correction by sampling a clean conditional and re-noising it, but its practical realization requires probability-flow ODE integration and clean-space Markov chain Monte Carlo (MCMC). We propose Direct Conditional Transition Sampling (DCTS), a direct stochastic-flow approximation to the same ideal refresh target. Rather than explicitly drawing a clean sample, DCTS estimates the measurement-conditioned clean mean along a short inner path and transports Gaussian source noise directly to the next noisy state. A denoiser-compatible sufficient statistic and a covariance-scaled operator update enable this conditional-mean estimation. Experiments on four inverse problems demonstrate that DCTS achieves competitive reconstruction quality with up to $16.8\times$ spee
    
[^19]: 大气化学机制中反应速率的优化与贝叶斯推断方法评估

    Evaluation of optimisation and Bayesian inference methods for reaction rates in atmospheric chemical mechanisms

    [https://arxiv.org/abs/2609.14569](https://arxiv.org/abs/2609.14569)

    本研究使用具有已知真实值的合成数据，比较了ODE约束神经网络优化与MCMC贝叶斯推断两种方法在大气自氧化机制反应速率系数估计中的性能，评估了它们在不同噪声水平和观测类型（直接浓度观测与质谱观测）下的表现。

    

    约束反应速率系数是显式大气化学机制开发中的核心挑战，特别是对于自氧化系统，其中许多反应路径只能通过高分辨率质谱间接观测。在本研究中，我们使用具有已知真实值的合成数据，对一个玩具案例自氧化机制的速率系数优化方法进行了评估。研究比较了两种互补的方法：ODE约束的神经网络优化方法，它能够为不确定的速率系数提供高效的点估计；以及马尔可夫链蒙特卡罗（MCMC）方法，它对速率系数的后验分布进行采样并量化参数不确定性。这些方法在不同噪声水平下，分别使用直接浓度观测和质谱观测进行了测试。对于未受扰动和低噪声的合成观测，两种方法均收敛于已知速率……

    arXiv:2609.14569v1 Announce Type: cross  Abstract: Constraining reaction rate coefficients is a central challenge in the development of explicit atmospheric chemical mechanisms, particularly for autoxidation systems where many reaction pathways are only indirectly observed through high-resolution mass spectrometry. In this study, we evaluate rate-coefficient optimisation methods for a toy-case autoxidation mechanism using synthetic data with known ground truth. Two complementary approaches are compared: ODE-constrained neural-network optimisation, which provides efficient point estimates of uncertain rate coefficients, and the Markov Chain Monte Carlo (MCMC) approach, which samples the posterior distribution of rate coefficients and quantifies parameter uncertainty. The methods are tested using direct concentration observations and mass-spectral observations under different noise levels. For unperturbed and low-noise synthetic observations, both methods converged towards the known rate
    
[^20]: 多源共形预测：通过局部化利用异质性

    Multi-source conformal prediction: leveraging heterogeneity via localization

    [https://arxiv.org/abs/2609.14531](https://arxiv.org/abs/2609.14531)

    该论文提出多源随机局部化共形预测（MS-RLCP），通过数据自适应的来源选择和可解释的包络分布概念，在异质多源数据与分布偏移场景下实现了有限样本覆盖保证。

    

    许多现代预测任务涉及来自多个异质来源的数据，而测试分布可能与任何单个来源存在显著差异。尽管异质性带来了挑战，但它也提供了机会：不同的来源可能提供互补信息，特征空间的某些区域在一个来源中可能比在另一个来源中得到更好的表示。我们提出了多源随机局部化共形预测（MS-RLCP），它建立在随机局部化共形预测（RLCP）（Hore 和 Barber，2025）的局部覆盖性质之上，并通过数据自适应的来源选择将其扩展到多源场景。在广泛采用的假设下，即所有来源与测试总体在给定特征条件下的响应分布相同，我们利用一种可解释的“包络分布”概念建立了有限样本覆盖界限，该概念刻画了各来源在特征空间中的总体表示能力。我们的分析表明……

    arXiv:2609.14531v1 Announce Type: cross  Abstract: Many modern prediction tasks involve data from multiple heterogeneous sources, while the test distribution may differ substantially from any individual source. Although heterogeneity poses challenges, it also offers an opportunity: different sources may provide complementary information, with some regions of the feature space better represented in one source than another. We propose Multi-Source Randomly Localized Conformal Prediction (MS-RLCP), which builds on the local coverage properties of randomly localized conformal prediction (RLCP) (Hore and Barber, 2025) and extends it to multiple sources through data-adaptive source selection. Under the widely adopted assumption of a shared response distribution conditional on the features across sources and the test population, we establish finite-sample coverage bounds using an interpretable notion of envelope distribution that captures their aggregate feature-space representation. Our anal
    
[^21]: 非参数方差惩罚行动者-评论家算法：面向风险敏感强化学习的统计推断

    Nonparametric Variance-Penalized Actor-Critic: Statistical Inference for Risk-Sensitive Reinforcement Learning

    [https://arxiv.org/abs/2609.14327](https://arxiv.org/abs/2609.14327)

    提出了一种非参数方差惩罚行动者-评论家框架（VPAC），用基于自助法和随机缩放的统计在线估计器替代第二个评论家网络来估计回报方差，无需辅助网络且方差惩罚有界，并证明了方差惩罚Q学习和双时间尺度行动者-评论家算法的几乎必然收敛性。

    

    方差惩罚是风险敏感强化学习（RL）的一种原则性方法，它显式地在期望回报与策略稳定性之间进行权衡。现有方法需要一个专门的第二评论家网络来在线估计回报方差，这增加了架构复杂性，并在学习过程中加剧了估计误差的累积。我们提出了一种非参数方差惩罚行动者-评论家（VPAC）框架，该框架用基于自助法和随机缩放的具有统计学依据的在线估计器取代了方差评论家网络，这些技术源自随机逼近领域的统计推断文献。这些估计器不需要辅助网络，保持单一评论家架构，并且产生的方差惩罚在构造上是有界的，从而实现了简洁的收敛性分析。我们通过常微分方程方法为方差惩罚Q学习算法和双时间尺度行动者-评论家变体建立了几乎必然收敛性。

    arXiv:2609.14327v1 Announce Type: new  Abstract: Variance penalization is a principled approach to risk-sensitive reinforcement learning (RL) that explicitly trades expected return for policy stability. Existing methods require a dedicated second critic to estimate return variance online, adding architectural complexity and compounding estimation error during learning. We propose a nonparametric variance-penalized actor-critic (VPAC) framework that replaces the variance critic with statistically grounded online estimators based on bootstrapping and random scaling, techniques drawn from the statistical inference literature for stochastic approximation. These estimators require no auxiliary network, maintain a single-critic architecture, and produce variance penalties that are bounded by construction, enabling clean convergence analysis. We establish almost-sure convergence for both a variance-penalized Q-learning algorithm and a two-timescale actor-critic variant via the ordinary differ
    
[^22]: 关于网络上大规模多重检验的非渐近方法

    On Large-Scale Multiple Testing Over Networks: A Non-Asymptotic Approach

    [https://arxiv.org/abs/2609.14170](https://arxiv.org/abs/2609.14170)

    该论文发现分布式多重检验中有限样本FDR失控源于赢者诅咒偏差，并提出交叉拟合贪心聚合算法（CFGA），通过数据分割实现了网络上分布式多重检验在有限样本下的严格FDR控制。

    

    分布式多重检验要求网络中的N个站点在严格的通信预算下控制全局错误发现率（FDR）。Pournaderi和Xiang（2024）提出的贪心区间聚合算法渐近地解决了这一问题，但在有限样本下可能违反FDR≤α的约束。我们将这一违反追溯到被选密度统计量中的“赢者诅咒”偏差，在标准带宽ε≍m⁻¹/²下其精确阶为Θ(m⁻¹/⁴√(log m))，其中m是网络中p值的总数。交叉拟合贪心聚合算法（CFGA）通过在每个节点数据的一半上选择嵌套拒绝族，并在另一半上进行评分，从而消除这一偏差，在每节点零假设比例已知的情况下实现有限样本FDR≤α；一个膨胀变体在η=1/m的可忽略松弛下覆盖了插入法（plug-in）设定。BONuS-GA则通过计数型knockoff进行校准，掩盖一组合成的均匀零假设，使得每个……（原文摘要在此处截断）

    arXiv:2609.14170v1 Announce Type: cross  Abstract: Distributed multiple testing asks $N$ sites to control a global false discovery rate (FDR) under a tight communication budget. The greedy interval-aggregation algorithm of Pournaderi and Xiang (2024) solves this asymptotically but can violate $\mathrm{FDR}\le\alpha$ at finite samples. We trace the violation to a winner's-curse bias in the selected density statistics, of exact order $\Theta(m^{-1/4}\sqrt{\log m})$ at the standard bandwidth $\varepsilon\asymp m^{-1/2}$, with $m$ the total number of p-values in the network. Cross-Fit Greedy Aggregation (CFGA) eliminates the curse by selecting the nested rejection family on one half of each node's data and scoring it on the other, achieving finite-sample $\mathrm{FDR}\le\alpha$ when per-node null rates are known; an inflated variant covers the plug-in setting at a vanishing $\eta=1/m$ slack. BONuS-GA instead masks a bag of synthetic uniform nulls calibrated by counting knockoffs, so every 
    
[^23]: 非凸非凹极小极大景观的黎曼上升-下降方法：向盆地鞍点的收敛及其在分布鲁棒优化中的应用

    Riemannian ascent--descent for nonconvex nonconcave minimax landscapes: convergence to basin saddle points and applications to distributionally robust optimization

    [https://arxiv.org/abs/2609.14141](https://arxiv.org/abs/2609.14141)

    该论文提出“盆地鞍点”的新概念，并证明黎曼梯度上升多步下降迭代在局部 Łojasiewicz 型增长条件下收敛到此类鞍点，为分布鲁棒优化中非凸非凹的极小极大问题提供了收敛性理论框架。

    

    我们研究一类针对统计风险问题的分布鲁棒优化（DRO）问题，该问题被表述为欧几里得空间与黎曼流形乘积上的极小极大问题。由于所得到的极小极大景观通常是非凸非凹的，目前尚无已知可用的全局收敛一阶方法。我们转而引入“盆地鞍点”的概念，即局部定义在笛卡尔积上的纳什均衡，该笛卡尔积由局部极小临界集某连通分量周围的 δ 盆地与测度流形上的测地球构成。在局部极小临界集连通分量周围的 δ 盆地内，我们在指数 β ∈ (1,2] 的局部 Łojasiewicz 型增长条件下，为黎曼梯度上升多步下降迭代发展了一个收敛到盆地鞍点的抽象收敛框架。在临界集满足 Lipschitz 正则性的条件下，我们建立了……

    arXiv:2609.14141v1 Announce Type: cross  Abstract: We study a class of distributionally robust optimization (DRO) problems for the statistical risk problem, formulated as minimax problems over the product of a Euclidean space and a Riemannian manifold. Because the resulting minimax landscape is nonconvex nonconcave in general, no globally convergent first order method is known to be available. We instead introduce the notion of a \emph{basin saddle point}, a Nash equilibrium defined locally on the Cartesian product of a $\delta$ basin around a connected component of the local minima critical set and a geodesic ball on the measure manifold. We develop an abstract convergence framework for a Riemannian gradient ascent multistep descent iteration to a basin saddle point under a local \L{}ojasiewicz type growth condition, with exponent $\beta \in (1,2]$, in the $\delta$ basin around connected components of the local minima critical sets. Under Lipschitz regularity of critical sets we estab
    
[^24]: 基于RoPE导数的精确有限注意力响应

    Exact Finite Attention Responses From RoPE Derivatives

    [https://arxiv.org/abs/2609.14127](https://arxiv.org/abs/2609.14127)

    该论文提出通过对RoPE导数进行积分并保留键值交互项，无需线性化即可精确预测注意力干预（位置编辑）的效果，符号预测准确率达95%以上，边际MAE相比现有方法最多降低九倍以上。

    

    我们推导了注意力干预的精确局部响应，使得候选编辑能够仅通过一个缓存的基线和一次反向传播来进行评分。出发点是RoPE导数 $\partial_p z(p) = A z(p)$：其积分给出了有限的位置位移，我们将其贯穿整个softmax进行计算，而不对旋转或归一化进行任何线性化。所得预测在768个保留提示集上的92,160次实际位置编辑中实现了95.36–96.52%的符号准确率，与位置雅可比方法相比将答案边际MAE降低了73.6–82.5%，与零基线相比降低了36.2–50.9%。对于同时进行的键和值编辑，同样的差分演算方法分离出了交互项 $C_{KV} = \sum_j (p'_j - p_j)\,\varepsilon_j$，而简单地将各自的归因相加会忽略这一项。保留该项使得在跨两个Qwen尺寸、两个任务的5,120次干预扫描的所有设置中，下游边际MAE均降低了超过九倍（摘要在此处被截断）。

    arXiv:2609.14127v1 Announce Type: cross  Abstract: We derive exact local responses for attention interventions, allowing candidate edits to be scored from a cached baseline and one backward pass. The starting point is the RoPE derivative $\partial_p z(p) = A z(p)$: its integral gives the finite positional displacement, which we carry through the softmax without linearising either rotation or normalisation. The resulting predictions achieve 95.36--96.52% sign accuracy across 92,160 executed positional edits on 768 held-out prompt sets, reducing answer-margin MAE by 73.6--82.5% against the positional Jacobian and by 36.2--50.9% against zero. For simultaneous key and value edits, the same divided-difference calculus isolates the interaction term $C_{KV} = \sum_j (p'_j - p_j)\,\varepsilon_j$, which is omitted by adding separate attributions. Retaining it reduces downstream margin MAE by more than a factor of nine in every setting of a 5,120-intervention sweep across two Qwen sizes, two tas
    
[^25]: 只需添加噪声：消除混合数据中基于树的变量重要性偏差

    Just add noise: Debiasing tree-based variable importance in mixed data

    [https://arxiv.org/abs/2609.14083](https://arxiv.org/abs/2609.14083)

    提出通过向分类变量添加少量噪声这一简单方法，消除基于树的变量重要性度量（如随机森林）对连续变量的偏好偏差，从而实现对混合数据的变量选择并控制错误发现率。

    

    arXiv:2609.14083v1 公告类型：交叉 摘要：来自随机森林等基于树的方法的变量重要性得分往往偏向连续型预测变量而非分类型预测变量。我们对这种偏差进行了理论分析，并提出了一种简单的补救方法：向每个分类型预测变量添加少量噪声。该校正方法在多种模拟数据集和真实世界数据集上得到了验证，并与整合路径稳定性选择相结合，实现了对混合数据进行变量选择，同时控制错误发现率。

    arXiv:2609.14083v1 Announce Type: cross  Abstract: Variable importance scores from tree-based methods such as random forests favor continuous predictors over categorical ones. We present a theoretical analysis of this bias and propose a simple remedy: add a small amount of noise to each categorical predictor. The correction is demonstrated on a variety of simulated and real-world datasets and combined with integrated path stability selection to perform variable selection with false discovery control for mixed data.
    
[^26]: 更小集成规模的线性集成采样

    Linear Ensemble Sampling with Smaller Ensembles

    [https://arxiv.org/abs/2609.13954](https://arxiv.org/abs/2609.13954)

    提出一种仅在正则化Gram矩阵显著变化时才刷新集成的线性集成采样算法，将充分集成规模从Θ(d log T)降低至Θ(d log d + d log log T)，同时在任意有界动作集上保持最先进的Õ(d^{3/2}√T)遗憾保证。

    

    集成采样通过维护一组模型集合，为随机化探索提供了一种实用的方法，但在保持强遗憾保证的前提下，集成规模可以小到什么程度仍是一个悬而未决的问题。具体而言，现有的保证使用 Θ(d log T) 的集成规模，相对于内在的 Ω(d) 集成规模下界，在时间范围 T 上留下了对数级的差距。我们旨在缩小这一差距，提出了一种仅在正则化 Gram 矩阵发生显著变化时才刷新集成的集成采样算法。这一机制将扰动分析局部化到 Gram 矩阵漂移可控的时间段内，从而将充分的集成规模降低到 Θ(d log d + d log log T)，同时在任意有界动作集上仍保持集成采样最先进的 Õ(d^{3/2}√T) 遗憾界。我们进一步证明，当动作集为基数为 K 的有限集合时，所提出的算法能够实现……（摘要在此处被截断）

    arXiv:2609.13954v1 Announce Type: new  Abstract: Ensemble sampling offers a practical approach to randomized exploration by maintaining a collection of models, but how small an ensemble can be while retaining strong regret guarantees remains unresolved. In particular, the existing guarantees use an ensemble size of $\Theta(d\log T)$, leaving a logarithmic gap in the horizon $T$ relative to the intrinsic $\Omega(d)$ ensemble-size barrier. We aim to narrow this gap by proposing an ensemble sampling algorithm that refreshes the ensemble only when the regularized Gram matrix changes substantially. This mechanism localizes the perturbation analysis to epochs with controlled Gram-matrix drift and reduces the sufficient ensemble size to $\Theta(d\log d+d\log\log T)$, while preserving the state-of-the-art $\tilde O(d^{3/2}\sqrt T)$ regret for ensemble sampling with arbitrary bounded arm sets. We further show that, when the arm set is finite of cardinality $K$, the proposed algorithm achieves t
    
[^27]: 含采样约束的增广原始-对偶动力学中的均衡偏倚与收敛性

    Equilibrium bias and convergence in augmented primal--dual dynamics with sampled constraints

    [https://arxiv.org/abs/2609.13925](https://arxiv.org/abs/2609.13925)

    本文揭示了采样约束引起的增广乘子信号偏倚会导致原始-对偶动力学均衡点偏移，给出了保持KKT均衡的充要条件，并通过递归约束估计与联合能量分析证明了光滑凸锥问题上原始-对偶迭代几乎必然收敛到KKT点。

    

    本研究探讨了当约束值由样本估计时，增广原始-对偶动力学的稳定性与收敛性。无偏的约束观测可能产生有偏的增广乘子信号，从而使平均动力学的均衡点发生偏移。对于分量不等式，我们给出了保持Karush-Kuhn-Tucker（KKT）均衡点的充分必要条件，并构造了一个凸优化例子，其局部指数稳定的均衡点违反互补松弛条件。为解决这一偏倚问题，在形成增广乘子信号之前对约束值进行递归估计。对于光滑凸锥问题，通过联合能量分析证明了原始状态、对偶状态与估计状态的有界性、估计误差的消失性，以及在全球正则性条件和有界条件二阶矩假设下，原始-对偶迭代序列几乎必然收敛到单个KKT点。该结果允许解不唯一的情形。

    arXiv:2609.13925v1 Announce Type: cross  Abstract: This work studies the stability and convergence of augmented primal-dual dynamics when constraint values are estimated from samples. Unbiased constraint observations can produce a biased augmented multiplier signal, shifting the equilibria of the mean dynamics. For componentwise inequalities, we give a necessary and sufficient condition for preserving the Karush-Kuhn-Tucker (KKT) equilibria and construct a convex example with a locally exponentially stable equilibrium that violates complementarity. To address this bias, constraint values are estimated recursively before forming the augmented multiplier signal. For smooth convex conic problems, a joint energy analysis establishes boundedness of the primal, dual, and estimation states, vanishing estimation error, and almost sure convergence of the primal-dual iterates to a single KKT point under global regularity and bounded conditional second moments. The result allows nonunique solutio
    
[^28]: 基于极限核的编码器-解码器算子学习的分辨率无关分析

    Resolution-Independent Analysis of Encoder--Decoder Operator Learning via Limiting Kernels

    [https://arxiv.org/abs/2609.13798](https://arxiv.org/abs/2609.13798)

    本文通过证明编码器-解码器架构中诱导的算子值核随分辨率提高收敛到极限核，建立了算子学习的分辨率无关理论，并为正则化随机梯度下降给出了分离编码、正则化与优化误差项的收敛上界。

    

    算子学习是在函数空间上定义的，但训练数据通常只能通过有限维表示获得。在编码器-解码器架构中，编码空间上的矩阵值核会在原始函数空间上诱导出一个算子值核，且相应的再生核希尔伯特空间是等距同构的。随着输入和输出分辨率的提高，诱导核收敛到一个极限核（就其关联的积分算子的算子范数收敛意义而言），这使得正则性假设可以独立于编码分辨率来表述。对于正则化随机梯度下降，我们针对递减步长和固定步长建立了误差上界，将编码与正则化项分别与阶为 \(t^{-\theta}\) 和 \(T^{-\theta'}\)（其中 \(\theta,\theta'\in(0,1)\)）的优化项分离。我们进一步证明了下界……

    arXiv:2609.13798v1 Announce Type: cross  Abstract: Operator learning is formulated on function spaces, but training data are typically available only through finite-dimensional representations. In encoder--decoder architectures, a matrix-valued kernel on the encoded space induces an operator-valued kernel on the original function spaces, and the corresponding reproducing kernel Hilbert spaces are isometrically isomorphic. As the input and output resolutions increase, the induced kernels converge to a limiting kernel, in the sense of operator-norm convergence of their associated integral operators, allowing regularity assumptions to be stated independently of the encoding resolution. For regularized stochastic gradient descent, we establish upper bounds for decreasing and fixed step sizes, separating the encoding and regularization terms from optimization terms of order \(t^{-\theta}\) and \(T^{-\theta'}\), respectively, for any \(\theta,\theta'\in(0,1)\). We further prove lower bounds 
    
[^29]: 论紧致连通李群上薛定谔桥的随机控制表述与路径空间表述的等价性

    On the Equivalence of Stochastic Control and Path Space Formulations for Schr\"odinger Bridges over Compact Connected Lie Groups

    [https://arxiv.org/abs/2609.13758](https://arxiv.org/abs/2609.13758)

    本文证明了紧致连通李群上薛定谔桥问题的随机最优控制表述与路径空间相对熵最小化表述的等价性，从而在解析、概率和计算三个层面为薛定谔桥的研究提供了新的理论基础。

    

    针对紧致连通李群上的运动学方程，我们建立了薛定谔桥问题的随机最优控制表述与路径空间表述之间的等价性。利用水平提升与随机反发展这两个几何概念，我们推导出了一个Girsanov型的测度变换结果，并证明期望控制能量等于受控路径律相对于参考维纳测度的相对熵。因此，薛定谔桥问题等价于一个在给定端点边缘分布约束下的路径空间相对熵最小化问题。我们的结果具有三个有用的推论。从解析的角度看，所证明的等价性有助于证明薛定谔桥的存在性与唯一性。从概率的角度看，它有助于将薛定谔桥解释为满足端点约束的无控制随机动力学最可能的偏离。从计算的角度看，它……

    arXiv:2609.13758v1 Announce Type: cross  Abstract: We establish the equivalence between the stochastic optimal control and path space formulations of the Schr\"odinger bridge problem (SBP) for the kinematic equation on a compact connected Lie group. Using the geometric concepts of horizontal lift and stochastic anti-development, we derive a Girsanov-type change-of-measure result, and show that the expected control energy equals the relative entropy of the controlled path law with respect to the reference Wiener measure. Thus, the SBP is equivalently a path space relative entropy minimization problem subject to prescribed endpoint marginals.Our result has three useful implications. From an analytic viewpoint, the shown equivalence helps prove the existence and uniqueness of the SB. From a probabilistic viewpoint, it helps interpret the SB as the most probable deviation of the uncontrolled stochastic dynamics consistent with the endpoint constraints. From a computational viewpoint, it al
    
[^30]: 分布偏移下归纳图上的在线贝叶斯节点分类

    Online Bayesian Node Classification on Inductive Graphs under Distribution Shift

    [https://arxiv.org/abs/2609.13655](https://arxiv.org/abs/2609.13655)

    本文提出变分贝叶斯最后一层（VBLL）方法，通过联合训练GNN编码器与最后一层的近似后验，并在测试时进行在线更新，实现了分布偏移下归纳图上节点分类的校准不确定性量化。

    

    在不断演化的图上，节点分类器必须满足两个关键要求：在分布偏移下对新到节点的归纳泛化能力，以及面向安全敏感应用的校准不确定性。标准的图神经网络（GNN）通常只训练一次，无法满足上述任一要求。我们通过在确定性GNN编码器之上放置随机的最后一层参数，对贝叶斯最后一层（BLL）模型进行适配，以实现不确定性量化。分类所需的类别softmax似然破坏了高斯共轭性，因此无论是训练后验还是测试时的流式更新都没有闭式解。为应对这两个挑战，我们提出了一种变分贝叶斯最后一层（VBLL）目标，通过最大化证据下界（ELBO）并结合蒙特卡洛期望对数似然，联合训练编码器与近似的最后一层后验。在测试阶段，我们冻结编码器并应用在线L（摘要在此处被截断）

    arXiv:2609.13655v1 Announce Type: new  Abstract: On evolving graphs, node classifiers must satisfy two key requirements: inductive generalization to newly arriving nodes under distribution shift and calibrated uncertainty for safety-sensitive applications. Standard graph neural networks (GNNs) are typically trained once and address neither requirement. We adapt the Bayesian last-layer (BLL) model by placing random last-layer parameters on top of a deterministic GNN encoder for uncertainty quantification. The categorical softmax likelihood required for classification breaks Gaussian conjugacy, so neither the training posterior nor the test-time streaming update has a closed-form solution. To address both challenges, we introduce a variational Bayesian last-layer (VBLL) objective that jointly trains the encoder and an approximate last-layer posterior by maximizing an evidence lower bound with a Monte Carlo expected log-likelihood. At test time, we freeze the encoder and apply an online L
    
[^31]: Hadamard流形上分布式在线优化的曲率无关遗憾界

    Curvature-Independent Regret Bounds for Distributed Online Optimization on Hadamard Manifolds

    [https://arxiv.org/abs/2609.13646](https://arxiv.org/abs/2609.13646)

    该论文针对Hadamard流形上的分布式在线优化，提出了结合局部h-次梯度更新与隐式Fréchet均值共识的D-ROGD算法，首次建立了与曲率无关的遗憾界，对h-凸和强h-凸目标分别达到O(√T)和O(log T)的遗憾率。

    

    本工作研究Hadamard流形上的去中心化在线黎曼优化问题。先前在测地凸性（g-凸性）条件下的工作在优化分析中可能需要曲率信息，通常通过截面曲率的有限下界来引入。曲率还可能出现在切空间黎曼共识方案的步长或收缩因子中。在本工作中，我们针对一类更窄的球面凸（h-凸）函数，放松了对曲率的依赖。我们研究了分布式黎曼在线梯度下降（D-ROGD）算法，该算法将局部黎曼h-次梯度更新与隐式Fréchet均值共识相结合。对于h-凸和强h-凸的局部目标函数，我们分别建立了O(√T)和O(log T)的静态遗憾界，在T的意义上匹配了相应的欧几里得速率，且对网络的依赖仅由谱间隙决定。据我们所知，这些是首个与曲率无关的遗憾界。

    arXiv:2609.13646v1 Announce Type: new  Abstract: This work addresses decentralized online Riemannian optimization on Hadamard manifolds. Prior work under geodesic convexity (g-convexity) may require curvature information in the optimization analysis, typically through a finite lower bound on the sectional curvature. Curvature may also enter the step size or contraction factor of tangent-space Riemannian consensus schemes. In this work, we relax the curvature dependence for a narrower class of horospherical convex (h-convex) functions. We study Distributed Riemannian Online Gradient Descent (D-ROGD), which combines local Riemannian h-subgradient updates with an implicit Fr\'echet-mean consensus. For h-convex and strongly h-convex local objectives, we establish $O(\sqrt{T})$ and $O(\log T)$ static regret, respectively, matching the corresponding Euclidean rates with respect to $T$, with network dependence governed solely by the spectral gap. To our knowledge, these are the first curvatur
    
[^32]: 高维时间序列谱差分网络分析的少假设推断

    Assumption-Lean Inference for Spectral Differential Network Analysis of High-Dimensional Time Series

    [https://arxiv.org/abs/2609.13609](https://arxiv.org/abs/2609.13609)

    本文提出了一个针对高维时间序列谱差分网络分析的少假设推断框架，通过直接估计两个高维逆谱密度的差异，并开发新的高斯近似误差界来实现统计推断，可应用于诸如大脑连接网络在刺激前后变化的研究。

    

    多变量时间序列的网络分析在从神经科学到地震学的许多领域都很流行。逆谱密度是时间序列网络分析的常见选择，因为它表示了在去除所有其他变量的最佳线性预测之后，两个变量之间的频域相关性。在许多应用中，研究目标是探索这些网络在不同条件下如何变化。例如，在神经科学中，人们可能感兴趣的是大脑连接网络在刺激前后如何变化。为实现这一目标，我们开发了一个基于两个高维逆谱密度差异直接估计的推断框架。我们为任意去偏D-轨迹估计方法开发了一个新的高斯近似误差界，并利用该误差界来确定谱密度Welch估计器的最优窗口大小，以及建立渐近正态性。

    arXiv:2609.13609v1 Announce Type: cross  Abstract: Network analysis for multivariate time series is popular in many fields, from neuroscience to seismology. The inverse spectral density is a common choice for time series network analysis due to its representation of the frequency domain correlation between two variables after removing the best linear predictor of all other variables. In many applications, the goal is to study how these networks change across different conditions. For example, in neuroscience, one might be interested in how the brain connectivity network changes before and after stimulation. Towards this goal, we develop an inference framework based on a direct estimate of the difference in two high-dimensional inverse spectral densities. We develop a new Gaussian approximation error bound for any de-biased D-trace estimation procedure which is then leveraged to both inform optimal window sizes of Welch's estimators of the spectral density and establish asymptotic norma
    
[^33]: 合成最近邻：将合成控制方法扩展至非随机缺失数据的矩阵补全

    Synthetic Nearest Neighbors: Extending Synthetic Controls for Matrix Completion with Missing Not at Random Data

    [https://arxiv.org/abs/2609.13586](https://arxiv.org/abs/2609.13586)

    该论文提出合成最近邻（SNN）方法，将计量经济学中的合成控制思想引入非随机缺失（MNAR）数据的矩阵补全，放宽了正性和独立性假设，支持灵活异质的观测模式，并建立了有限样本误差界、渐近正态性及可行的逐项推断。

    

    我们为非随机缺失（MNAR）数据下的矩阵补全开发了一个因果框架。借鉴计量经济学面板数据文献中的合成控制方法，我们的方法放宽了MNAR矩阵补全中常见的两个假设：观测指标的正性和独立性假设。与传统的面板数据模型（通常要求预设的块稀疏几何结构）不同，我们的框架通过目标特定的局部信息结构，能够适应灵活且异质的观测模式。我们提出了合成最近邻（SNN），一种受局部合成控制思想启发的估计量，并在适当条件下建立了有限样本逐项误差界和均值恢复的一致性。我们进一步在异方差噪声下推导了渐近正态性，并发展了可行的逐项统计推断。为估计逐项噪声方差，我们将同样的局部原则应用于结果的平方，从而获得一致性的估计。

    arXiv:2609.13586v1 Announce Type: cross  Abstract: We develop a causal framework for matrix completion under missing not at random (MNAR) data. Drawing on synthetic controls from the econometric panel data literature, our approach relaxes two assumptions common in MNAR matrix completion: positivity and independence of observation indicators. Unlike traditional panel data models, which often require prescribed block-sparse geometries, our framework accommodates flexible, heterogeneous observation patterns through target-specific local information structures. We propose synthetic nearest neighbors (SNN), a local synthetic-controls-inspired estimator, and establish finite-sample entrywise error bounds and consistency for mean recovery under suitable conditions. We further derive asymptotic normality under heteroskedastic noise and develop feasible entrywise inference. To estimate entry-specific noise variances, we apply the same local principle to squared outcomes, obtaining consistency u
    
[^34]: 带动量与自适应迁移率的拉回校正标量辅助变量优化器

    A pullback-corrected scalar auxiliary variable optimizer with momentum and adaptive mobility

    [https://arxiv.org/abs/2609.13569](https://arxiv.org/abs/2609.13569)

    该论文提出了一种融合动量与自适应迁移率的拉回校正标量辅助变量优化器，通过单次隐式求解同时校正梯度与动量，证明了Loewner序非递增的迁移率可保证精确的修正能量律，并给出了驻点局部稳定性的充要条件。

    

    科学机器学习中的目标函数通常被规定为若干项之和，例如物理信息神经网络的残差损失、边界损失、初始条件损失与数据损失。在拉回校正标量辅助变量（PB--SAV）方法中，用一个标量来追踪偏移后的目标函数，而各分量梯度则构建一个秩至多等于分量个数的半正定曲率校正。我们将该校正引入一种同时具有动量和自适应迁移率的优化器中，并在单次隐式求解中将其同时作用于梯度与存储的动量。在Loewner序意义下非递增的迁移率可以导出一个精确的修正能量律，该结果涵盖了欧氏型以及AMSGrad型的迁移率选择；而对于在求解之后追加的动量，其对应的恒等式中则包含一个符号不定的交叉项。对于固定迁移率的情形，我们给出了驻点处局部稳定性的一个充要条件，该条件依赖于……（摘要原文在此处截断）

    arXiv:2609.13569v1 Announce Type: cross  Abstract: Objectives in scientific machine learning are often prescribed as a sum of several terms, such as the residual, boundary, initial, and data losses of a physics-informed neural network. In the pullback-corrected scalar auxiliary variable (PB--SAV) method, one scalar tracks the shifted objective while the component gradients build a positive semidefinite curvature correction of rank at most the number of components. We carry that correction into an optimizer with momentum and an adaptive mobility, applying it to the gradient and the stored momentum in a single implicit solve. A mobility that is nonincreasing in the Loewner order yields an exact modified energy law, covering Euclidean and AMSGrad-type choices; the corresponding identity for momentum appended after the solve carries a cross term of indefinite sign. For a fixed mobility we give a necessary and sufficient condition for local stability at a stationary point, depending on the 
    
[^35]: 面向具有非自适应对手的多臂老虎机的最优切换遗憾

    Toward Optimal Switching Regret for Multi-Armed Bandits with Oblivious Adversary

    [https://arxiv.org/abs/2609.13547](https://arxiv.org/abs/2609.13547)

    本文提出单一算法，在非自适应对手下对任意未知的切换次数 $S$ 均能达到最优的 $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ 期望切换遗憾，解决了 Auer 等人提出的开放问题。

    

    我们研究对抗性多臂老虎机中的切换遗憾问题，其中学习者与一个至多变化 $S$ 次的动作序列竞争。当 $S$ 已知时，可以获得 $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ 的最优期望遗憾 [Auer et al., 2002]。然而，当 $S$ 未知时，Marinov 和 Zimmert [2021] 证明在自适应对手下这一保证是不可能实现的。在本文中，我们证明单个算法在非自适应对手下对每一个 $S$ 都能达到 $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ 的期望遗憾，从而解决了 Auer 等人 [2019b] 提出的一个开放问题。我们的算法结合了以小学习率初始化的固定分享学习器，以及利用随机化学习率和隐式探索来搜索局部改进的二分区间子程序。重要的是，非均匀先验偏向于跟随主学习器，使得维护众多子程序的代价保持在较小水平。当子程序……

    arXiv:2609.13547v1 Announce Type: new  Abstract: We study switching regret in adversarial multi-armed bandits, where the learner competes with an arm sequence that changes at most $S$ times. When $S$ is known, an optimal expected regret of $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ is obtainable [Auer et al., 2002]. However, when $S$ is unknown, Marinov and Zimmert [2021] show that this guarantee is impossible under an adaptive adversary. In this paper, we show that a single algorithm achieves $\widetilde{\mathcal{O}}(\sqrt{(S+1)KT})$ expected regret for every $S$ against an oblivious adversary, resolving an open problem of Auer et al. [2019b]. Our algorithm combines a fixed-share learner initialized with a small learning rate and dyadic-interval subroutines that search for local improvements using randomized learning rates and implicit exploration. Importantly, a non-uniform prior favors following the main learner, keeping the cost of maintaining many subroutines small. When the subrou
    
[^36]: 面向布尔任务的ReLU-MLP可认证可解释训练与真值表泛化保证

    Certifiably Interpretable Training of ReLU-MLPs for Boolean Tasks with Guaranteed Truth-Table Generalization

    [https://arxiv.org/abs/2609.13439](https://arxiv.org/abs/2609.13439)

    MACCHIATO算法通过迭代地将布尔函数残差投影到低维{AND, OR, XOR}电路类并将其精确编译为ReLU-MLP，实现了从部分真值表观测出发、可认证且可解释地训练ReLU-MLP，同时保证真值表的泛化。

    

    随着计算规模的扩大、模型的演进以及训练算法的进步，我们解释由此催生的日益强大的人工智能系统的能力正在不断被削弱。为了帮助保障可解释性，我们引入了一种专门的训练算法（MACCHIATO），该算法联合构建：(i) 一个从部分真值表观测中显式构建的ReLU多层感知机（ReLU-MLP），以及(ii) 一个在带符号文字上、由{AND, OR, XOR}门构成的显式布尔电路，该电路可以认证其子网络计算的内容以及它们如何组合。直观地说，我们迭代地将布尔函数的残差投影到低维{AND, OR, XOR}电路类上，并将所得电路精确编译为ReLU-MLP；我们结合了ReLU-MLP电路编译、ESPRESSO逻辑最小化以及基于影响力的变量选择。大致来说，

    arXiv:2609.13439v1 Announce Type: cross  Abstract: As compute scales, models evolve, and training algorithms advance, our ability to explain the increasingly powerful AI systems they enable is eroding. To help safeguard interpretability, we introduce a specialized training algorithm (MACCHIATO) that jointly constructs (i) an explicitly structured $\operatorname{ReLU}$-MLP from partial truth-table observations and (ii) an explicit Boolean circuit over signed literals with $\{\operatorname{AND},\operatorname{OR},\operatorname{XOR}\}$ gates certifying what its subnetworks compute and how they compose. Intuitively, we iteratively project the residuals of a Boolean function onto low-dimensional $\{\operatorname{AND},\operatorname{OR},\operatorname{XOR}\}$-circuit classes and exactly compile the resulting circuit into a $\operatorname{ReLU}$-MLP; we combine $\operatorname{ReLU}$-MLP circuit compilation, ESPRESSO logic minimization, and influence-based variable selection.   Roughly speaking, 
    
[^37]: 先收敛后多样化：多目标贝叶斯优化中收敛性与多样性的解耦

    Converge Then Diversify: Decoupling Convergence and Diversity in Multi-Objective Bayesian Optimisation

    [https://arxiv.org/abs/2609.13396](https://arxiv.org/abs/2609.13396)

    本文提出将多目标贝叶斯优化中的收敛性与多样性两个目标解耦，采用“先收敛后多样化”的策略，从而在有限的搜索预算下更有效地逼近帕累托前沿。

    

    多目标贝叶斯优化（MOBO）是一种样本高效的方法，用于优化具有多个目标的昂贵黑箱函数。在MOBO中，目标是充分逼近帕累托前沿，即获得一个高质量的解集，该解集具备1）良好的收敛性（接近帕累托前沿）和2）良好的多样性（在帕累托前沿上分布广泛）。现有的MOBO方法通常旨在同时完成这两项任务，即在保持非支配解多样性的同时，将搜索推向帕累托前沿，从而理想情况下解能够逐渐逼近整个前沿。当拥有充足的搜索预算时，这种方法是有效的。然而，在整个搜索过程中同时兼顾收敛性和多样性并不容易，需要精心的设计。在非常紧张的预算下，可能无法生成足够多的解来同时逼近整个帕累托前沿……（摘要原文在此处截断）

    arXiv:2609.13396v1 Announce Type: new  Abstract: Multi-objective Bayesian optimisation (MOBO) is a sample-efficient approach for optimising expensive black-box functions with multiple objectives. In MOBO, the goal is to adequately approximate the Pareto front; that is, to obtain a high-quality solution set with 1) good convergence (closeness to the Pareto front) and 2) good diversity (spread across the Pareto front). Existing MOBO methods typically aim to accomplish these two tasks simultaneously, i.e., driving the search towards the Pareto front while maintaining a diverse set of nondominated solutions, such that the solutions, ideally, can gradually approach the entire front. When sufficient search budgets are available, this approach is effective. However, considering both convergence and diversity throughout the search is not easy and requires careful design. Under very tight budgets, there may not be enough solutions generated to be able to simultaneously approach the entire Paret
    
[^38]: 超越点预测：时间序列与时空数据概率预测综述

    Beyond Point Forecasts: A Survey on Probabilistic Forecasting for Time Series and Spatiotemporal Data

    [https://arxiv.org/abs/2609.13345](https://arxiv.org/abs/2609.13345)

    这篇综述通过按不确定性在预测流程中引入的位置和方式建立统一分类框架，整合了时间序列与时空概率预测中分散的统计建模、机器学习和深度生成建模方法，并涵盖了集成校准、贝叶斯建模、分布回归以及新兴的时间序列基础模型等范式。

    

    概率预测是不确定性下决策的核心，但其方法论版图在时间序列预测与时空预测、统计建模、机器学习以及深度生成建模等领域之间日益碎片化。本综述通过根据不确定性在预测流程中引入的位置和方式来组织概率预测方法，构建了一个统一的视角。我们的分类体系将模型无关方法（包括集成方法和无分布校准）与模型内在方法（涵盖贝叶斯建模、参数化预测分布、分布回归以及现代生成模型）联系起来，并进一步审视了时间序列基础模型的新兴角色。除方法论的综合梳理之外，我们还识别了不同范式所依赖的假设、计算需求以及所刻画的不确定性形式，并将这些差异转化为……（摘要内容在此处截断）

    arXiv:2609.13345v1 Announce Type: cross  Abstract: Probabilistic forecasting is central to decision-making under uncertainty, yet its methodological landscape has become increasingly fragmented across temporal and spatiotemporal forecasting, statistical modeling, machine learning, and deep generative modeling. This survey develops a unified perspective by organizing probabilistic forecasting methods according to where and how uncertainty is introduced into the forecasting pipeline. Our taxonomy connects model-agnostic approaches including ensembles and distribution-free calibration, with model-intrinsic approaches spanning Bayesian modeling, parametric predictive distributions, distributional regression, and modern generative models, and further examines the emerging role of time series foundation models. Beyond methodological synthesis, we identify the assumptions, computational demands, and forms of uncertainty represented by different paradigms, and translate these distinctions into
    
[^39]: P2空间（Wasserstein空间）上的随机梯度下降

    Stochastic Gradient Descent over P2

    [https://arxiv.org/abs/2609.13343](https://arxiv.org/abs/2609.13343)

    该论文将经典欧氏空间中SGD的扩散（高斯）近似理论首次推广到Wasserstein空间P2上的优化问题，通过Lions可微性将问题提升至线性希尔伯特空间，并构造了与随机梯度矩信息相匹配的高斯随机场近似。

    

    随机梯度下降（SGD）存在扩散近似方法，即用高斯噪声替代随机梯度中复杂的随机性，这为理解其动力学和长时间行为提供了强有力的工具。我们研究了类似的近似原理是否适用于概率测度空间上的优化问题，其目标函数是定义在Wasserstein空间P2上的泛函。P2的非线性几何结构和无穷维特性阻碍了经典欧几里得理论的直接推广。利用Lions可微性，我们将该问题提升到一个线性希尔伯特空间，从而可以进行高阶微分演算。随后，我们构造了一个高斯随机场近似，其速度场与原始随机梯度的均值和协方差相匹配。通过在高阶泰勒展开中利用这种矩匹配，我们证明了高斯近似能够捕捉原始动力学……（摘要原文在此处截断）

    arXiv:2609.13343v1 Announce Type: cross  Abstract: Stochastic gradient descent (SGD) admits diffusion approximations that replace the complicated randomness of stochastic gradients by Gaussian noise, providing a powerful tool for understanding its dynamics and long-time behavior. We investigate whether an analogous approximation principle holds for optimization over probability measures, where the objective is a functional defined on the Wasserstein space P2. The nonlinear geometry and infinite-dimensional nature of P2 prevent a direct extension of the classical Euclidean theory. Using Lions differentiability, we lift the problem to a linear Hilbert space, where higher-order differential calculus becomes available. We then construct a Gaussian random-field approximation whose velocity field matches the mean and covariance of the original stochastic gradient. By exploiting this moment matching through higher-order Taylor expansions, we show that the Gaussian approximation captures the S
    
[^40]: 医学图像分类中类间过渡不确定性的自适应保形重分配

    Adaptive Conformal Redistribution for Inter-class Transitional Uncertainty in Medical Image Classification

    [https://arxiv.org/abs/2609.13303](https://arxiv.org/abs/2609.13303)

    本文提出AdaConRed，一种无需标签的保形后决策规则，通过视觉-语言生成式增强、基础模型嵌入和熵调制的自适应预测集，将医学图像分类中过渡类别的模糊预测集重新分配为精确的单一类别判定。

    

    医学图像分类经常受到过渡类别的干扰，这些类别的特征分布与相邻类别重叠，从而产生模糊的决策边界。保形预测能够返回具有不确定性感知的预测集，但在需要做出单一决策的临床筛查场景中，这些预测集并不直接可操作。本工作提出了自适应保形重分配，这是一种无需标签的保形后决策规则，可将模糊的预测集转换为精细的类别分配。该工作开发了一个五阶段流水线：利用视觉-语言生成式增强解决少数类样本稀缺问题；采用冻结的DermFoundation编码器提供特征嵌入；使用轻量级多层感知机执行分类；通过熵调制、边界感知的非一致性分数构建自适应预测集；将被预测为过渡类别且具有多标签预测集的样本重新分配到最可能的替代类别。

    arXiv:2609.13303v1 Announce Type: cross  Abstract: Medical image classification is frequently complicated by transitional categories whose feature distributions overlap those of adjacent classes, producing ambiguous decision boundaries. Conformal prediction returns uncertainty-aware prediction sets, but these are not directly actionable in clinical screening, where a single decision is required. This work proposes adaptive conformal redistribution (AdaConRed), a label-free post-conformal decision rule that converts ambiguous prediction sets into refined class assignments. A five-stage pipeline is developed. Vision-language generative augmentation addresses minority-class scarcity; a frozen DermFoundation encoder provides embeddings; a lightweight multi-layer perceptron performs classification; an entropy-modulated, margin-aware nonconformity score constructs adaptive prediction sets; samples predicted as transitional with multi-label sets are reassigned to the most probable alternative
    
[^41]: 在工业化建筑中利用人类专业知识实现高精度机器人装配：一种样本高效的安装人员在环交互式强化学习框架

    Harnessing human expertise for high-precision robotic assembly in industrialized construction: A sample-efficient installer-in-the-loop interactive reinforcement learning framework

    [https://arxiv.org/abs/2609.13234](https://arxiv.org/abs/2609.13234)

    该论文提出一种安装人员在环的交互式强化学习框架，通过遥操作演示、事件驱动接管和验收对齐奖励，将安装人员的隐性专业知识高效转化为工业化建筑中模块化组件高精度机器人装配的自主能力。

    

    工业化建筑对预制窗单元等模块化组件的机器人装配提出了严格的精度要求。在公差敏感的操作中，核心瓶颈不仅在于机械间隙，还在于如何在稀疏的验收反馈、接触多变性和毫米级约束条件下，将隐性安装人员专业知识转化为数据高效的自主能力。我们提出了一种安装人员在环的交互式强化学习框架，该框架通过离线遥操作演示、接触失败边界处的稀疏事件驱动二元接管以及与验收标准对齐的终端奖励来获取专业知识，并在统一模式下进行记录，以实现可追溯的离线到在线适应。基于Q分块与流Q学习构建的时间抽象动作序列策略，可在稀疏终端奖励下捕获多模态恢复操作，同时非更新热启动阶段稳定了离线……（摘要原文在此处截断）

    arXiv:2609.13234v1 Announce Type: cross  Abstract: Industrialized construction imposes stringent precision requirements on robotic assembly of modular components such as prefabricated window units. In tolerance-critical operations, the central bottleneck is not only mechanical clearance but also converting tacit installer expertise into data-efficient autonomy under sparse acceptance feedback, contact variability, and millimeter-scale constraints. We present an installer-in-the-loop interactive reinforcement learning framework that acquires expertise through offline teleoperated demonstrations, sparse event-driven binary takeovers at contact-failure boundaries, and acceptance-aligned terminal rewards, logged under a unified schema for traceable offline-to-online adaptation. A temporally abstract action-sequence policy built on Q-chunking with Flow Q-Learning captures multimodal recovery maneuvers under sparse terminal rewards, while a non-updating warm-start phase stabilizes the offlin
    
[^42]: 化学与几何表征保真度提升药物-靶点亲和力预测

    Chemical and geometric representation fidelity improves drug--target affinity prediction

    [https://arxiv.org/abs/2609.13230](https://arxiv.org/abs/2609.13230)

    该论文提出表征保持框架ReGeoDTA，通过在分子表征中保留化学异质性、在蛋白质结构中保留连续几何关系，解决了表征阶段的信息丢失这一上游瓶颈，从而在三个基准数据集上持续提升药物-靶点亲和力预测性能。

    

    预测药物-靶点结合亲和力（DTA）要求模型能够区分分子识别背后细微的化学与结构决定因素。尽管近期的方法越来越多地纳入更丰富的药物和蛋白质信息，但这些信息在表征构建过程中可能被压缩、同质化或离散化，导致与亲和力相关的差异在交互建模之前就已丢失。我们假设这种表征阶段的信息丢失构成了一个上游瓶颈，无法通过日益复杂的交互预测器来可靠克服。为验证这一假设，我们开发了ReGeoDTA，这是一个表征保持框架，能够在分子表征中保持与亲和力相关的化学异质性，并在蛋白质结构中保持连续的几何关系。在三个基准数据集上，ReGeoDTA持续提升了亲和力预测性能，且所提出的表征……（原文摘要在此处截断）

    arXiv:2609.13230v1 Announce Type: cross  Abstract: Predicting drug--target binding affinity (DTA) requires models to distinguish subtle chemical and structural determinants underlying molecular recognition. Although recent approaches increasingly incorporate richer drug and protein information, such information may be compressed, homogenized or discretized during representation construction, causing affinity-relevant distinctions to be lost before interaction modelling. We hypothesized that this representation-stage information loss constitutes an upstream bottleneck that cannot be reliably overcome by increasingly complex interaction predictors. To test this hypothesis, we developed ReGeoDTA, a representation-preserving framework that maintains affinity-relevant chemical heterogeneity in molecular representations and continuous geometric relationships in protein structures. Across three benchmark datasets, ReGeoDTA consistently improved affinity prediction, and the proposed representa
    
[^43]: 用于解释时间相关输出的希尔伯特值泛函分解框架

    A Hilbert-Valued Functional Decomposition Framework for Explaining Time-Dependent Outputs

    [https://arxiv.org/abs/2609.11295](https://arxiv.org/abs/2609.11295)

    本文提出一个希尔伯特值泛函分解框架，将基于特征的解释方法从标量输出推广到时间相关的函数型输出，并通过基于核的输出表示实现多时间粒度的时间依赖感知解释。

    

    基于特征的解释方法量化特征对模型预测的影响，但这类方法主要是为标量输出设计的。然而在许多应用中，输出是函数型或多变量的，例如需求预测中随时间变化的轨迹。因此，现有方法通常独立地解释每个输出位置，忽略了输出各分量之间的依赖关系。我们通过开发一个针对时间相关输出的统一基于特征解释框架来解决这一局限。具体而言，我们将泛函分解推广到希尔伯特值的预测函数，并将现有的基于特征解释框架扩展到该设置中。我们的框架引入了基于核的输出表示，能够在多个时间粒度层次上实现时间依赖感知的解释，包括特定时间、时间分辨和时间聚合三个层次，同时提供了一个统一视角，使现有方法成为该框架下的特例。

    arXiv:2609.11295v1 Announce Type: cross  Abstract: Feature-based explanations quantify features' influence on model predictions, but are primarily designed for scalar outputs. In many applications, however, outputs are functional or multivariate, such as time-dependent trajectories in demand forecasting. Consequently, existing approaches typically explain each output location independently, ignoring dependencies across the output components. We address this limitation by developing a unified framework for feature-based explanations of time-dependent outputs. Specifically, we generalize functional decomposition to Hilbert-valued prediction functions and extend an existing feature-based explanation framework to this setting. Our framework introduces kernel-based output representations that enable time-dependency-aware explanations at multiple levels of temporal granularity, including time-specific, time-resolved, and time-aggregated, while providing a unified view in which existing metho
    
[^44]: 流对偶性与分类生成中的源分布几何

    Flow Duality and Source Geometry for Categorical Generation

    [https://arxiv.org/abs/2609.10863](https://arxiv.org/abs/2609.10863)

    本文揭示了连续与离散流匹配之间的对偶性——通过逐位置argmax投影可将连续凸插值路径转化为离散凸插值路径，并证明连续源分布的几何设计（高斯、有界均匀、中心负指数等）会显著影响分类生成中的转移时机和词表规模依赖性。

    

    连续流匹配与离散流匹配通常被视为两种相互独立的构造。本文识别出二者之间的一种对偶性：将带有独热目标的连续凸插值路径通过逐位置 argmax 投影，即可得到离散凸插值路径。该结果要求源分布具备适当的坐标对称性和边界正则性，从而使连续源分布成为分类生成中一个显式的设计选择。我们推导了高斯源、有界均匀源和中心负指数源所诱导的离散插值行为，结果表明不同的源几何会导致定性上不同的转移时机以及对词表规模的依赖性。小规模的视觉诊断实验和一个简短的语言建模试点实验显示，这些源设计带来的影响同样会体现在学得的传输映射和早期生成质量之中。

    arXiv:2609.10863v1 Announce Type: new  Abstract: Continuous and discrete flow matching are usually treated as separate constructions. This paper identifies a duality between them: projecting continuous convex-interpolant paths with one-hot targets through a position-wise argmax yields discrete convex-interpolant paths. The result requires source laws with appropriate coordinate symmetry and boundary regularity, and it makes the continuous source distribution an explicit design choice for categorical generation. We derive the induced discrete interpolation behavior for Gaussian, bounded-uniform, and centered negative-exponential sources, showing that different source geometries lead to qualitatively different transition timing and vocabulary-size dependence. Small visual diagnostics and a short language-modeling pilot suggest that these source-design effects can also appear in learned transports and early generative quality.
    
[^45]: 条件Shapley特征重要性的半参数推断

    Semiparametric Inference for Conditional Shapley Feature Importance

    [https://arxiv.org/abs/2609.10313](https://arxiv.org/abs/2609.10313)

    本文针对条件Shapley特征重要性提出了一种带K折交叉拟合和U统计量修正的半参数一步估计器，消除了蒙特卡洛偏差，在双重稳健速率条件下实现√n一致性与渐近正态性，并提供覆盖率有保证的Wald置信区间。

    

    Shapley值被广泛用于事后特征归因，但大多数估计器仅返回点估计量而无法量化不确定性，且流行的实现方法从边际分布中采样联盟外特征，这在特征相关时会导致重要性归因错误。本文研究了条件形式化方法，即联盟外特征在其真实条件分布下被积分出去。目标是一个全局的、基于损失的重要性度量，它将条件价值函数与SAGE风格的损失聚合相结合。我们提出了一种结合K折交叉拟合的一步估计器，并对平方损失进行U统计量修正，从而消除了朴素插入估计器的蒙特卡洛偏差；该估计器在双重稳健速率条件下具有√n一致性和渐近正态性，由此构造的Wald置信区间达到了名义覆盖率。此外，还给出了Pinsker型界以量化工作条件分布被误设时产生的偏差。

    arXiv:2609.10313v1 Announce Type: cross  Abstract: Shapley values are widely used for post-hoc feature attribution, but most estimators return point quantities and do not quantify uncertainty, and popular implementations sample out-of-coalition features from their marginal distribution, which misattributes importance when features are dependent. This paper studies the conditional formulation, in which out-of-coalition features are integrated out under their true conditional distribution. The target is a global, loss-based importance that pairs a conditional value function with a SAGE-style loss aggregation. We propose a one-step estimator with K-fold cross-fitting and a U-statistic correction of the squared loss that removes the Monte Carlo bias of the naive plug-in; it is $\sqrt{n}$-consistent and asymptotically normal under double-robust rate conditions, and the resulting Wald interval attains nominal coverage. A Pinsker-type bound quantifies the bias from misspecifying the working c
    
[^46]: 时间序列基础模型的合成数据蒸馏

    Distillation of Synthetic Data for Time Series Foundation Models

    [https://arxiv.org/abs/2609.09586](https://arxiv.org/abs/2609.09586)

    本文提出合成数据蒸馏（SDD）方法，通过将时间序列基础模型的输出与每条轨迹的条件预测分布而非已实现的未来值进行比较来构建预训练损失目标，该方法在理论上可证明降低随机梯度协方差，并在实证中使400万至25亿参数规模的模型验证损失收敛更快。

    

    时间序列基础模型（TSFMs）越来越多地在合成生成的时间序列轨迹上进行预训练，其中数据生成过程是已知的。当前的预训练方案基于将TSFM输出与每条轨迹的已实现未来值进行比较的损失目标。我们转而提出将TSFM输出与每条轨迹的条件预测分布进行比较的损失目标，我们将这一过程称为合成数据蒸馏（SDD）。SDD对应于训练目标的Rao-Blackwell化，即它在保持随机梯度期望不变的同时，在Loewner偏序意义下可证明地降低随机梯度的协方差。我们在参数规模从400万到25亿的TSFM模型家族上对SDD进行了实证验证，并观察到在每个模型规模上验证损失都收敛得更快：在高斯过程数据上，SDD达到或超越了现状损失方法的表现。

    arXiv:2609.09586v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) are increasingly pre-trained on synthetically generated time series trajectories, where the data generating process is known. Current pre-training recipes are based on loss objectives which compare TSFM outputs to realized future values of each trajectory. We instead propose loss objectives which compare TSFM outputs to the conditional forecast distribution of each trajectory, a procedure we call synthetic data distillation (SDD). SDD corresponds to a Rao-Blackwellization of the training objective, in that it leaves the expectation of stochastic gradients unchanged while provably reducing the covariance of the stochastic gradient under the Loewner partial ordering. We empirically validate SDD on a TSFM model family of sizes from $4$M to $2.5$B parameters, and observe faster convergence of validation loss at every model size: on Gaussian Process data, SDD attains or improves upon the Status Quo loss w
    
[^47]: 用于网络比较的最优传输：综述及其机器学习应用

    Optimal Transport for Network Comparison: A Review with Machine Learning Applications

    [https://arxiv.org/abs/2608.27500](https://arxiv.org/abs/2608.27500)

    本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。

    

    运用最优传输进行网络比较是网络科学中一个不断发展的研究领域。与标准的图度量不同，最优传输不仅计算网络间的相异性，还提供一个传输方案来解释一张图如何演变为另一张图。本文综述了如何利用三种主要距离——Wasserstein距离、Gromov-Wasserstein距离和Bures-Wasserstein距离——来比较无向无权图。我们考察了通过节点特征概率分布在一维情形下Wasserstein距离的闭式解，并展示了Wasserstein距离和Gromov-Wasserstein距离的传输方案如何捕捉图扰动后具体哪些节点影响了距离。对于Bures-Wasserstein距离，我们利用拉普拉斯谱推导出上界，从而避免了完整的谱分解。最后，我们使用合成网络数据集评估这些距离在聚类任务中的表现，并应用于真实世界的时间序列网络数据。

    arXiv:2608.27500v1 Announce Type: cross  Abstract: Network comparison using optimal transport is a growing area of research in network science. Unlike standard graph metrics, optimal transport computes both network dissimilarity and a transport plan that explains how one graph morphs into another. In this paper, we review how optimal transport compares undirected, unweighted graphs using three primary distances: the Wasserstein, Gromov-Wasserstein, and Bures-Wasserstein distances. We examine the closed form of the Wasserstein distance in one dimension via node feature probability distributions, and show how the transport plans of the Wasserstein and Gromov-Wasserstein distances capture which specific nodes influence the distance after graph perturbation. For the Bures-Wasserstein distance, we derive bounds using Laplacian spectra to bypass full spectral decompositions. Finally, we evaluate these distances using a synthetic network dataset for clustering and a real-world time series net
    
[^48]: AI验证的几何学：独立同分布最佳N次搜索的精确认证极限

    The geometry of AI validation: Exact certification limits for iid best-of-N search

    [https://arxiv.org/abs/2608.21496](https://arxiv.org/abs/2608.21496)

    本文通过核几何方法，精确推导出独立同分布最佳N次搜索中验证的模糊宽度公式，并揭示其主导尺度为$m^2/N$，为AI验证提供了理论极限。

    

    摘要：人工智能系统越来越多地生成备选方案、检查证据并部署选定的输出。因此，验证是相对于目标的：证据仅在被产生它的干预所解决的方向上认证部署。我们将验证和部署规则表示为可靠性表面上的核。它们的跨度几何将复制（减少采样噪声）与新的干预方向（减少结构性盲区）区分开来。我们使这一原理在独立同分布的最佳N次搜索中精确成立。在标量排序、随机平局、最大选择、有界二元真值以及稳定的排序-真值关系下，通过$n=m$了解最佳$n$可靠性，留下精确的模糊宽度$B_{m,N}=1+2\sum_{r=1}^{m}(-1)^r\cos^{2N}{r\pi/[2(m+1)]}$。显式有界世界达到整个区间，完整前缀在限于$n\le m$的可靠性均值审计中是信息最大化的。主导尺度是$m^2/N$：这……

    arXiv:2608.21496v1 Announce Type: cross  Abstract: AI systems increasingly generate alternatives, inspect evidence, and deploy a selected output. Validation is therefore target-relative: evidence certifies deployment only in directions resolved by the interventions that produced it. We represent validation and deployment rules as kernels over a reliability surface. Their span geometry separates replication, which reduces sampling noise, from new intervention directions, which reduce structural blindness. We make this principle exact for iid best-of-$N$ search. Under scalar ranking, randomized ties, maximum selection, bounded binary truth, and a stable rank-truth relation, knowing best-of-$n$ reliability through $n=m$ leaves exact ambiguity width $B_{m,N}=1+2\sum_{r=1}^{m}(-1)^r\cos^{2N}{r\pi/[2(m+1)]}$. Explicit bounded worlds attain the entire interval, and the complete prefix is information-maximal among reliability-mean audits confined to $n\le m$. The governing scale is $m^2/N$: wh
    
[^49]: 入侵者阈值：LoRA微调的谱定律

    The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning

    [https://arxiv.org/abs/2607.23711](https://arxiv.org/abs/2607.23711)

    该论文提出了一个无需拟合参数的逐层谱定律 $s^\ast=\bar\theta/(\gamma\sigma_1(BA))$，可仅从权重矩阵的测量谱预测LoRA微调中入侵维度出现的临界更新强度，在18个适配器、9,840次层扫描的验证中，82%的层上阈值预测误差在两倍以内。

    

    LoRA微调会产生“入侵维度”：即更新后权重矩阵 $W+BA$ 中新的主导奇异向量，它们与所有预训练奇异向量几乎正交，并会导致灾难性遗忘。自其被发现以来，尚无理论能够基于测量的谱逐层预测它们何时出现。我们推导出了每层的临界更新强度 $s^\ast=\bar\theta/(\gamma\sigma_1(BA))$，该强度仅通过矩形尖峰形变变换从 $W$ 的测量谱计算得出，同时对更新后的谱给出了精确的久期方程刻画，且无需任何拟合参数。在一项预先设定的研究中，涵盖四个稠密Transformer家族、一个状态空间模型、一个混合专家模型和一个编码器-解码器模型（共18个适配器、9,840次层扫描），该定律在82%的层上将经验阈值定位在两倍因子范围内，并能在部署时区分含有入侵维度与不含入侵维度的层……

    arXiv:2607.23711v2 Announce Type: replace  Abstract: LoRA fine-tuning can create intruder dimensions: new leading singular vectors of the updated weight matrix $W+BA$ that are nearly orthogonal to all pretrained singular vectors and that drive catastrophic forgetting. Since their discovery, no theory has predicted, layer by layer on measured spectra, when they appear. We derive a per-layer critical update strength $s^\ast=\bar\theta/(\gamma\sigma_1(BA))$, computed from the measured spectrum of $W$ alone through the rectangular spiked-deformation transform, together with an exact secular-equation characterization of the updated spectrum, with no fitted parameters. In a pre-specified study spanning four dense Transformer families, a state-space model, a mixture-of-experts model, and an encoder-decoder (18 adapters, 9{,}840 layer scans), the law localizes the empirical threshold within a factor of two on $82\%$ of layers, separates intruder-bearing from intruder-free layers at deployment 
    
[^50]: 面向复值亥姆霍兹波场的算子信息引导高斯过程：从合成基准到在体脑弹性成像

    Operator-Informed Gaussian Processes for Complex Helmholtz Wavefields: From Synthetic Benchmarks to In Vivo Brain Elastography

    [https://arxiv.org/abs/2607.14193](https://arxiv.org/abs/2607.14193)

    本文提出将算子信息引导的高斯过程回归扩展到复值亥姆霍兹波场问题，通过将复算子实化为等价的耦合实块以实现标准实值高斯过程条件化推断，并提供了从对角先验到多尺度变体的一族先验，在合成基准问题上验证了其竞争力并有望应用于在体脑弹性成像。

    

    亥姆霍兹方程描述时间简谐波传播，在耗散介质中，复模量使其平方波数 κ² 成为复数。从稀疏、含噪数据中推断此类场，需要既能求解又能量化自身不确定性的求解器。物理信息引导的高斯过程（GP）回归通过返回解的后验分布提供了这种能力，然而算子条件化的方法几乎只针对实值场发展。我们通过将复算子实化为等价的耦合实块，将算子信息引导的高斯过程回归扩展到复值亥姆霍兹问题，从而可以使用标准的实值高斯过程条件化进行推断。该构造允许一族先验，从适当的对角先验到协同区域化和多尺度变体，并以偏微分方程残差和边界迹作为条件。在一维到三维的基准问题上，该求解器具有竞争力……

    arXiv:2607.14193v3 Announce Type: replace-cross  Abstract: The Helmholtz equation governs time-harmonic wave propagation, and in dissipative media a complex modulus renders its squared wavenumber $\kappa^2$ complex. Inferring such fields from sparse, noisy data calls for solvers that also quantify their own uncertainty. Physics-informed Gaussian-process (GP) regression supplies this by returning a posterior over the solution, yet operator-conditioned formulations have been developed almost exclusively for real-valued fields. We extend operator-informed GP regression to complex-valued Helmholtz problems by realifying the complex operator into an equivalent coupled real block, which enables inference with standard real-valued GP conditioning. The construction admits a family of priors, from a proper diagonal prior to coregionalized and multiscale variants, and conditions on PDE residuals and boundary traces. On benchmark problems in one to three dimensions, the solver is competitive with
    
[^51]: 基于扩散路径的马尔可夫链蒙特卡洛方法

    Markov Chain Monte Carlo with Diffusion Paths

    [https://arxiv.org/abs/2607.11631](https://arxiv.org/abs/2607.11631)

    该论文提出沿扩散路径而非传统温度调节方法构建中间分布进行MCMC采样，能够保留多峰分布中各模态的相对权重并获得更好的混合性质。

    

    从多峰分布中进行采样是经典局部马尔可夫链蒙特卡洛（MCMC）方法长期面临的挑战。一种流行的解决方案是引入一系列中间分布，在目标分布和更简单的参考分布之间进行插值。经典的选择是温度调节（tempering）方法，它将密度提升到某个幂次，但这会扭曲非对称模态之间的相对权重，并可能导致混合性能不佳。我们转而提出沿扩散路径进行插值，即一个向目标分布逐渐添加噪声使其趋向高斯分布的扩散过程的边缘分布。这条路径保留了各模态之间的相对权重，并具有良好的混合性质，我们通过对应理想转移核的谱隙分析使这一点更加明确。沿路径进行采样需要路径的中间分数，可以通过变分方法从非归一化目标分布中进行估计，从而得到一个近似采样器。为了消除（摘要在此处截断）...

    arXiv:2607.11631v2 Announce Type: replace-cross  Abstract: Sampling from multimodal distributions is a longstanding challenge for classical local Markov chain Monte Carlo (MCMC) methods. A popular remedy is to introduce a sequence of intermediate distributions that interpolate between the target and a simpler reference. The classical choice, tempering, raises the density to a power, but distorts the relative weights of asymmetric modes and can lead to poor mixing. We instead propose interpolating along the diffusion path, the marginals of a noising diffusion process that carries the target toward a Gaussian. This path preserves the relative weights of the modes and enjoys favorable mixing properties, which we make precise through a spectral-gap analysis of the corresponding ideal transition kernel. Sampling along the path requires its intermediate scores, which can be estimated from the unnormalized target through variational approaches, yielding only an approximate sampler. To remove 
    
[^52]: 高斯过程后验采样的差分隐私

    Differential Privacy of Gaussian Process Posterior Sampling

    [https://arxiv.org/abs/2606.17995](https://arxiv.org/abs/2606.17995)

    该论文首次证明高斯过程后验采样的内在随机性本身即可提供差分隐私保证，并通过分离后验均值与数据依赖协方差两条泄露通道，确定有效岭正则化和协方差尺度为控制隐私的关键量。

    

    我们研究了当整个训练集（包括协变量和响应）均为隐私数据时，发布高斯过程（GP）函数型后验样本路径的隐私问题。与注入外部噪声的标准差分隐私（DP）机制不同，后验采样本身具有内在随机性，我们证明了这种随机性能够提供有用的隐私保证。我们推导了Rényi-DP保证，将通过后验均值的隐私泄露与由数据依赖的后验协方差所导致的独立泄露通道分离开来。该分析确定了有效岭正则化和协方差尺度是控制隐私的主要量，并在若干实际感兴趣的情形下给出了更锐利的保证，同时扩展到了重复发布和自适应发布的场景。成员推断攻击验证了所预测的对正则化、协方差尺度和发布路径数量的依赖关系。下游任务上的效用实验……（摘要截断）

    arXiv:2606.17995v2 Announce Type: replace-cross  Abstract: We study the privacy of releasing functional posterior sample paths from a Gaussian process (GP) when the entire training set including covariates and responses is private. Unlike standard differential-privacy (DP) mechanisms that inject external noise, posterior sampling is intrinsically random and we show that this randomness provides useful privacy guarantees. We derive R\'enyi-DP guarantees separating privacy leakage through the posterior mean from a distinct channel induced by the data-dependent posterior covariance. The analysis identifies effective ridge regularisation and covariance scale as the principal privacy-controlling quantities and yields sharper guarantees in several regimes of practical interest as well as extensions to repeated and adaptive releases. Membership inference attacks confirm the predicted dependence on regularisation, covariance scale and the number of released paths. Utility experiments on downst
    
[^53]: SAILS：基于代理模型并通过局部效应平滑的交互分析

    SAILS: Surrogate-based Analysis of Interactions via Local Effect Smooths

    [https://arxiv.org/abs/2606.09404](https://arxiv.org/abs/2606.09404)

    SAILS是一个模型无关的框架，通过广义可加模型代理在局部效应层面检测特征交互，将交互形式分类为线性、乘积可分离和非乘积可分离类型，并为每种类型提供可解释的可视化。

    

    特征交互是机器学习模型预测能力的重要来源，然而现有的解释方法只能检测和量化交互，无法揭示其函数形式，或者只能可视化受限的交互类型。我们提出了SAILS（基于代理模型并通过局部效应平滑的交互分析），这是一个模型无关的框架，通过将广义可加模型（GAM）代理拟合到黑盒模型的局部效应上来分析成对交互。对于感兴趣特征的每个区间，代理的平滑项在导数层面分离出交互成分，从而实现：(i) 通过基于平滑项显著性检验得出的启发式方法进行交互检测；(ii) 将交互形式分类为线性、乘积可分离和非乘积可分离三种类型；(iii) 为每种交互类型提供定制化的可解释可视化。我们对所提出的框架进行了实证验证。

    arXiv:2606.09404v2 Announce Type: replace-cross  Abstract: Feature interactions drive much of the predictive power of machine learning models, yet existing explanation methods only detect and quantify interactions without revealing their functional form, or visualize only restricted interaction types. We propose Surrogate-based Analysis of Interactions via Local Effect Smooths (SAILS), a model-agnostic framework that analyzes pairwise interactions through generalized additive model (GAM) surrogates fitted to the local effects of a black-box model. For each interval of a feature of interest, the surrogate smooth terms isolate the interaction components on derivative level, enabling (i) interaction detection through a heuristic derived from significance tests on smooth terms, (ii) interaction form categorization into linear, product-separable, and non-product-separable types, and (iii) tailored, interpretable visualizations for each interaction type. We empirically validate the framework
    
[^54]: InfoAtlas：一个用于零样本统计依赖估计的基础模型

    InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimate

    [https://arxiv.org/abs/2606.00241](https://arxiv.org/abs/2606.00241)

    InfoAtlas是一个基础模型式的神经互信息估计器，通过在大规模合成依赖模式数据上预训练，实现单次前向传播即可直接估计互信息，在保持最先进精度的同时获得100倍加速，并以单一统一模型灵活处理不同维度、样本量及真实世界场景。

    

    衡量高维随机变量之间的统计依赖性是数据科学和机器学习中的一项基础任务。神经互信息（MI）估计器为此提供了一条有前景的途径，但它们通常需要针对每个新数据集进行代价高昂的迭代优化，使其难以适用于实时应用场景。我们提出了InfoAtlas，这是一种类似基础模型的架构，通过在单次前向传播中直接推断互信息来消除这一瓶颈。InfoAtlas在大规模合成数据上进行了预训练，这些数据包含丰富的依赖模式，使其能够识别多样化的依赖结构并直接从数据集中预测互信息。全面的实验表明，InfoAtlas在精度上可与最先进的神经估计器相媲美，同时实现了100倍的加速，并能通过单一统一模型灵活处理不同的维度和样本量，还能有效泛化到复杂的真实世界场景。

    arXiv:2606.00241v2 Announce Type: replace-cross  Abstract: Measuring statistical dependency between high-dimensional random variables is a fundamental task in data science and machine learning. Neural mutual information (MI) estimators offer a promising avenue, but they typically require costly iterative optimization for each new dataset, making them impractical for real-time applications. We present InfoAtlas, a foundation model-like architecture that eliminates this bottleneck by directly inferring MI in a single forward pass. Pretrained on large-scale synthetic data with rich dependence patterns, InfoAtlas learns to identify diverse dependence structures and predict MI directly from the dataset. Comprehensive experiments demonstrate that InfoAtlas matches state-of-the-art neural estimators in accuracy while achieving $100\times$ speedup, can flexibly handle varying dimensions and sample sizes through a single unified model, and generalizes effectively to complex, real-world scenario
    
[^55]: 合适的校准击败（Proper Calibeating）

    Proper Calibeating

    [https://arxiv.org/abs/2605.26703](https://arxiv.org/abs/2605.26703)

    本文将经典的“校准”与“校准击败”概念从二次评分规则推广到所有有界合适评分规则，证明校准蕴含合适校准但校准击败不必然蕴含合适校准击败，并提出总是合适的“完全校准击败”方法。

    

    经典的“校准预测”概念及其近期改进的“校准击败”概念，均是相对于标准二次评分规则定义的。我们将这些概念扩展到合适评分规则的类别（即真实分布为最优预测的评分规则），并通过要求相应的保证在所有有界合适评分规则上一致成立，定义了“合适校准”与“合适校准击败”。我们首先证明了校准总是蕴含合适校准，而校准击败却不一定蕴含合适校准击败。其次，我们展示了如何保证合适校准击败和合适多重校准击败；特别地，“完全校准击败”——一种在校准击败联合分箱意义上更强形式的校准击败——总是合适的。最后，我们考虑了不确定性下的决策问题，即决策者对预测做出最优应对。我们证明了（合适）校准等价于……

    arXiv:2605.26703v3 Announce Type: replace-cross  Abstract: The classic concept of "calibrated forecasts" and its more recent refinement, "calibeating," are defined with respect to the standard quadratic scoring rule. We extend these notions to the class of proper scoring rules (for which the true distribution is an optimal forecast) and define \textit{proper calibration} and \textit{proper calibeating} by requiring the corresponding guarantees to hold uniformly over all bounded proper scoring rules. We first establish that calibration always implies proper calibration, whereas calibeating need not imply proper calibeating. Second, we show how to guarantee proper calibeating and proper multicalibeating; in particular, \textit{complete calibeating}---a strong form of calibeating that calibeats the joint binning---is always proper. Finally, we consider \textit{decision-making under uncertainty}, where one best replies to the forecasts. We establish that (proper) calibration is equivalent 
    
[^56]: 基于自编码器的叠加多分量阻尼正弦信号参数估计

    Autoencoder-Based Parameter Estimation for Superposed Multi-Component Damped Sinusoidal Signals

    [https://arxiv.org/abs/2604.03985](https://arxiv.org/abs/2604.03985)

    提出一种基于自编码器的方法，利用潜在空间在含噪的叠加多分量阻尼正弦信号中高精度地估计每个分量的频率、相位、衰减时间和幅度，即使在信号快速衰减和存在次要分量等困难情况下依然有效。

    

    阻尼正弦振荡广泛存在于许多物理系统中，对其分析可以获得潜在的物理特性。然而，当信号快速衰减、多个分量叠加且存在观测噪声时，参数估计变得困难。在本研究中，我们开发了一种基于自编码器的方法，利用潜在空间来估计含噪多分量阻尼正弦信号中每个分量的频率、相位、衰减时间和幅度。我们在高斯分布训练下研究了多分量情况，并通过高斯训练与均匀训练的对比进一步检验了训练数据分布的影响。性能通过波形重建和参数估计精度进行评估。我们发现，即使在具有挑战性的设置下（例如涉及次要分量的情形），所提出的方法仍能以高精度估计参数。

    arXiv:2604.03985v2 Announce Type: replace  Abstract: Damped sinusoidal oscillations are widely observed in many physical systems, and their analysis provides access to underlying physical properties. However, parameter estimation becomes difficult when the signal decays rapidly, multiple components are superposed, and observational noise is present. In this study, we develop an autoencoder-based method that uses the latent space to estimate the frequency, phase, decay time, and amplitude of each component in noisy multi-component damped sinusoidal signals. We investigate multi-component cases under Gaussian-distribution training and further examine the effect of the training-data distribution through comparisons between Gaussian and uniform training. The performance is evaluated through waveform reconstruction and parameter-estimation accuracy. We find that the proposed method can estimate the parameters with high accuracy even in challenging setups, such as those involving a subdomina
    
[^57]: 功能缩放律下的最优学习率调度：幂衰减与预热-稳定-衰减

    Optimal Learning Rate Schedules under Functional Scaling Laws: Power Decay and Warmup-Stable-Decay

    [https://arxiv.org/abs/2602.06797](https://arxiv.org/abs/2602.06797)

    本研究证明了在功能缩放律框架下最优学习率调度存在尖锐相变：简单任务应从训练伊始就采用幂衰减调度，困难任务则应采用预热-稳定-衰减（WSD）式调度，且两种情形下的衰减指数均由模型容量唯一决定。

    

    我们在功能缩放律（FSL）框架（Li et al., 2025）下研究最优学习率（LR）调度，该框架将训练动态分解为信号学习和噪声遗忘两个部分。在幂律核回归中，这两个组成部分分别由源指数 $s>0$ 和容量指数 $q>1$ 控制，其中较小的 $s$ 对应更难的任务。对于固定的训练长度 $N$，我们在稳定性约束下刻画了使最后一步损失最小化的调度，并揭示了一个尖锐的相变现象。在简单任务区间 $s>1-1/q$ 中，最优调度从训练一开始就遵循幂衰减；在困难任务区间 $s<1-1/q$ 中，最优调度呈现为预热-稳定-衰减（WSD）形式（Hu et al., 2024），即在训练的大部分时间保持在最大可容许学习率，最后再进行衰减。在两种区间中，衰减指数均为 $2q-1$：任务难度决定何时开始衰减，而模型容量决定衰减的速率。

    arXiv:2602.06797v3 Announce Type: replace-cross  Abstract: We study optimal learning rate (LR) schedules under the functional scaling law (FSL) framework (Li et al., 2025), which decomposes training dynamics into signal learning and noise forgetting. In power-law kernel regression, these two components are governed by a source exponent $s>0$ and a capacity exponent $q>1$, respectively, with smaller $s$ corresponding to harder tasks. For a fixed training horizon $N$, we characterize the schedules that minimize the final-step loss under a stability constraint and reveal a sharp phase transition. In the easy-task regime $s>1-1/q$, the optimal schedule follows power decay from the beginning of training; in the hard-task regime $s<1-1/q$, it becomes warmup-stable-decay (WSD)-like (Hu et al., 2024), staying at the largest admissible LR for most of training before a final decay. In both regimes, the decay exponent is $2q-1$: task difficulty determines when to decay, while model capacity deter
    
[^58]: 有向随机块模型中基于邻域平滑的精确恢复

    Exact Recovery by Neighborhood Smoothing in Directed Stochastic Block Models

    [https://arxiv.org/abs/2601.16427](https://arxiv.org/abs/2601.16427)

    本文提出一种基于邻域平滑的方法，通过聚类顶点的出边连接概率轮廓在稀疏有向随机块模型中实现精确社区恢复，并建立了非对称平滑估计量的有限样本逐行误差界以及精确恢复的分离条件。

    

    我们研究了稀疏有向随机块模型中的精确社区恢复问题，采用对连接概率轮廓进行邻域平滑的方法。所提出的方法根据顶点估计出的出边连接概率轮廓对顶点进行聚类。对于每个顶点，其完整的出边轮廓是通过对经验上相似的顶点的邻接行取平均来估计的，随后对估计出的轮廓应用 K-均值聚类。通过将相同的构造应用于转置邻接矩阵，可以得到基于入边连接概率轮廓的类似过程。我们为该非对称平滑估计量建立了有限样本的一致逐行误差界，并推导出其在归一化二到无穷范数下的相合性。我们进一步证明，当不同总体轮廓之间的最小间隔大于逐行估计误差时，即可实现精确恢复。该结果允许稀疏度趋于零……

    arXiv:2601.16427v3 Announce Type: replace-cross  Abstract: We study exact community recovery in sparse directed stochastic block models using neighborhood smoothing of connection-probability profiles. The proposed method clusters vertices according to their estimated outgoing connection-probability profiles. For each vertex, its complete outgoing profile is estimated by averaging the adjacency rows of empirically similar vertices, after which \(K\)-means is applied to the estimated profiles. An analogous procedure based on incoming connection-probability profiles is obtained by applying the same construction to the transposed adjacency matrix.   We establish a finite-sample uniform row-wise error bound for the asymmetric smoothed estimator and derive consistency in the normalized two-to-infinity norm. We then show that exact recovery follows when the minimum separation between distinct population profiles dominates the row-wise estimation error. The result permits a vanishing sparsity 
    
[^59]: 一阶优化的基本不等式及其在统计风险分析中的应用

    Basic Inequalities for First-Order Optimization with Applications to Statistical Risk Analysis

    [https://arxiv.org/abs/2512.24999](https://arxiv.org/abs/2512.24999)

    本文提出了一阶迭代优化算法的“基本不等式”统一框架，通过算法固有几何下的距离刻画目标函数值与任意参考点的差距，从而将隐式与显式正则化联系起来，并可广泛应用于统计风险分析。

    

    在这项工作中，我们为一阶迭代优化算法引入了“基本不等式”，构建了一个简单而多功能的框架，将隐式正则化与显式正则化联系起来。基于文献中已有的关于优化迭代的比较不等式，我们对这些论证进行了扩展和统一，形成了一个通用框架，可作为统计分析的工具。更具体地说，设 $f$ 表示待优化的目标函数。给定一个在 $\theta_0$ 处初始化、当前迭代点为 $\theta_T$ 的一阶迭代算法，基本不等式利用累积步长以及 $\theta_0$、$\theta_T$ 和 $z$ 之间的距离，对任意参考点 $z$ 给出 $f(\theta_T) - f(z)$ 的上界。这些距离是在优化算法固有的几何结构下度量的，这进而转化为一种贯穿其中的正则化概念……

    arXiv:2512.24999v2 Announce Type: replace-cross  Abstract: In this work, we introduce $\textit{basic inequalities}$ for first-order iterative optimization algorithms, forming a simple yet versatile framework which connects implicit and explicit regularization. Building on related comparison inequalities for optimization iterates that already exist in the literature, we extend and unify these arguments to produce a general framework, which can be used as a tool for statistical analysis. In more detail, let $f$ denote the objective function to be optimized. Given a first-order iterative algorithm initialized at $\theta_0$, with current iterate $\theta_T$, the basic inequality upper bounds $f(\theta_T) - f(z)$ for any reference point $z$ in terms of the accumulated step sizes, and the distances between $\theta_0$, $\theta_T$, and $z$. These distances are measured in a geometry inherent to the optimization algorithm, which then translates into a notion of regularization being applied acros
    
[^60]: 面向近似最近邻搜索与核密度估计的次线性草图

    Sublinear Sketches for Approximate Nearest Neighbor and Kernel Density Estimation

    [https://arxiv.org/abs/2510.23039](https://arxiv.org/abs/2510.23039)

    本文提出了适用于动态数据流的新型草图算法，使近似最近邻搜索和近似核密度估计同时获得次线性空间与次线性查询时间的保证。

    

    arXiv:2510.23039v2 公告类型：替换 摘要：近似最近邻（ANN）搜索和近似核密度估计（A-KDE）是现代机器学习核心的基础性问题，在数据分析、信息系统和大规模决策中具有广泛的应用。在海量且动态变化的数据流场景中，一个核心挑战是设计紧凑的草图，使其既能保留数据的关键结构特性，又能支持高效的查询。在本工作中，我们开发了新的草图算法，针对动态数据流中的 ANN 和 A-KDE 两个问题，同时实现了次线性的空间和查询时间保证。对于流模型中的 ANN 问题，在自然假设下，我们设计了一种次线性草图，通过仅存储总输入中次线性比例（n^(-η)）的数据，仅需 O(n^((1-η)(1+ρ))) 的内存开销，其中 ρ 是局部敏感哈希（LSH）族的一个参数，且 0<η<1。我们的方法支持次线性查询时间、批量查询，并可将该技术扩展到……（原文摘要在此处截断）

    arXiv:2510.23039v2 Announce Type: replace  Abstract: Approximate Nearest Neighbor (ANN) search and Approximate Kernel Density Estimation (A-KDE) are fundamental problems at the core of modern machine learning, with broad applications in data analysis, information systems, and large-scale decision making. In massive and dynamic data streams, a central challenge is to design compact sketches that preserve essential structural properties of the data while enabling efficient queries.   In this work, we develop new sketching algorithms that achieve sublinear space and query time guarantees for both ANN and A-KDE for a dynamic stream of data. For ANN in the streaming model, under natural assumptions, we design a sublinear sketch that requires only $\mathcal{O}(n^{(1-\eta)(1+\rho)})$ memory by storing only a sublinear ($n^{-\eta}$) fraction of the total inputs, where $\rho$ is a parameter of the LSH family, and $0<\eta<1$. Our method supports sublinear query time, batch queries, and extends t
    
[^61]: 基于大语言模型的不确定性感知校准临床文本分类

    Uncertainty-Aware Calibrated Clinical Text Classification with Large Language Models

    [https://arxiv.org/abs/2509.19375](https://arxiv.org/abs/2509.19375)

    该论文将闭集临床文本分类创新性地建模为无似然后验推断问题，把提示条件化的大语言模型视为类条件随机模拟器，并利用序贯蒙特卡罗近似贝叶斯计算获得诊断后验分布，从而同时实现准确预测与良好校准的不确定性估计，为临床决策提供可融合先验医学知识的可靠置信度。

    

    大语言模型正被越来越多地用于临床文本分类，在这种场景下，过度自信的错误分类可能直接影响患者护理。现有的黑盒不确定性方法使用softmax概率、言语化置信度、提示一致性或生成一致性，为固定的大语言模型预测附加一个置信度分数。这些信号通常校准不佳，且缺乏将模型证据与先验临床信念相结合的机制。我们转而将闭集临床分类问题表述为针对诊断假设的无似然后验推断。一个以提示为条件的大语言模型被视为类条件随机模拟器：对于每个候选诊断，它生成合成的临床描述，并在嵌入摘要空间中将其与观察到的病例进行比较。随后，序贯蒙特卡罗近似贝叶斯计算得出诊断上的后验分布，从中可以同时获得预测和（此处的摘要内容不完整）

    arXiv:2509.19375v2 Announce Type: replace-cross  Abstract: Large language models are increasingly used for clinical text classification, where overconfident misclassifications can directly affect patient care. Existing black-box uncertainty methods attach a confidence score to a fixed LLM prediction using softmax probabilities, verbalised confidence, prompt agreement, or generation consistency. These signals are often poorly calibrated and offer no mechanism for combining model evidence with prior clinical belief. We instead formulate closed-set clinical classification as likelihood-free posterior inference over diagnostic hypotheses. A prompt-conditioned LLM is treated as a class-conditional stochastic simulator: for each candidate diagnosis it generates synthetic clinical descriptions, which are compared with the observed case in an embedding summary space. Sequential Monte Carlo Approximate Bayesian Computation then yields a posterior over diagnoses from which both the prediction an
    
[^62]: 目标性数据投毒攻击是否如我们想象的那样有效？

    Are Targeted Data Poisoning Attacks as Effective as We Think?

    [https://arxiv.org/abs/2509.06896](https://arxiv.org/abs/2509.06896)

    本文提出仅利用干净模型信息识别数据集中最容易和最难被投毒的样本，主张应评估目标性投毒攻击在最坏情况下的真实效果而非平均成功率，并据此指导针对性的防御策略。

    

    目标性数据投毒攻击通过在训练过程中注入恶意数据来操纵模型对特定测试样本的预测。然而，现有的评估方法报告的是随机选择目标的平均攻击成功率，掩盖了真实的最坏情况下的攻击有效性。我们认为正确的评估应当聚焦于最难被投毒的样本。同样的推理也适用于防御：由于目标性攻击在数据分布层面不留任何痕迹，防御者应当主动识别最脆弱的样本并采取针对性的对抗措施。给定一个测试数据集，本文仅基于干净模型信息就能识别出其中最容易和最难被投毒的样本。具体而言，我们利用干净训练动态进行粗略评估，并使用投毒距离和投毒预算对投毒类别进行细粒度分类。我们的实验表明，这些指标能够可靠地按照投毒脆弱性对样本进行分层，从而支持严格的……

    arXiv:2509.06896v3 Announce Type: replace  Abstract: Targeted data poisoning attacks manipulate model predictions on specific test samples by injecting malicious data into training. Yet existing evaluations report average attack success rates over randomly selected targets, obscuring true worst-case effectiveness. We argue that the right evaluation focuses on the hardest samples to poison. The same reasoning applies to defense: since targeted attacks leave no footprint at the distribution level, defenders should proactively identify the most vulnerable samples and apply targeted countermeasures. Given a test dataset, this paper identifies both the easiest and hardest to poison examples based on only clean model information. Specifically, we offer coarse evaluations using clean training dynamics, and fine-grained classification on poison class using poison distances and budgets. Our experiments show these metrics reliably stratify samples by poisoning vulnerability, enabling both rigoro
    
[^63]: N$^2$：一个用于基于近邻方法的矩阵补全的统一Python包与测试平台

    N$^2$: A Unified Python Package and Test Bench for Nearest Neighbor-Based Matrix Completion

    [https://arxiv.org/abs/2506.04166](https://arxiv.org/abs/2506.04166)

    本文发布了N$^2$——一个模块化、可扩展的基于近邻方法的矩阵补全统一Python包与测试平台，并提出了一种在多个场景下达到最先进水平的新近邻变体，同时提供了覆盖医疗保健、推荐系统、因果推断和LLM评估等领域的真实数据集基准套件。

    

    近邻（NN）方法已重新成为矩阵补全任务中极具竞争力的工具，展现出强大的实证性能，并获得了近期的理论保证，包括逐元素误差界、置信区间以及极小极大最优性。尽管方法本身简单，近期研究表明近邻方法对多种缺失模式具有鲁棒性，并在各类应用中均表现有效。本文介绍了N$^2$，一个统一的Python包和测试平台，它通过模块化、可扩展的接口整合了一大类基于近邻的方法。N$^2$面向研究人员和从业者构建，支持快速实验与基准测试。基于该框架，我们提出了一种新的近邻方法变体，在多个场景下取得了最先进的结果。我们还发布了一套真实世界数据集的基准套件，涵盖医疗保健、推荐系统、因果推断和LLM评估等领域，旨在对矩阵补全方法进行严格测试。

    arXiv:2506.04166v3 Announce Type: replace  Abstract: Nearest neighbor (NN) methods have re-emerged as competitive tools for matrix completion, offering strong empirical performance and recent theoretical guarantees, including entry-wise error bounds, confidence intervals, and minimax optimality. Despite their simplicity, recent work has shown that NN approaches are robust to a range of missingness patterns and effective across diverse applications. This paper introduces N$^2$, a unified Python package and testbed that consolidates a broad class of NN-based methods through a modular, extensible interface. Built for both researchers and practitioners, N$^2$ supports rapid experimentation and benchmarking. Using this framework, we introduce a new NN variant that achieves state-of-the-art results in several settings. We also release a benchmark suite of real-world datasets, from healthcare and recommender systems to causal inference and LLM evaluation, designed to stress-test matrix comple
    
[^64]: 具有边际覆盖保证的噪声自适应保形分类

    Noise-Adaptive Conformal Classification with Marginal Coverage

    [https://arxiv.org/abs/2501.18060](https://arxiv.org/abs/2501.18060)

    本文提出了一种噪声自适应保形推断方法，能够有效处理随机标签噪声导致的可交换性偏差，在低质量标签场景下仍能生成具有严格边际覆盖保证的信息丰富预测集合。

    

    保形推断为机器学习中的不确定性量化提供了一个严格的统计框架，能够为任何分类模型生成校准良好的预测集合，并具有精确的覆盖保证。然而，该方法依赖于完美数据可交换性这一理想化假设，这限制了其在面对现实世界复杂情况时的有效性，例如低质量标签——这是现代大规模数据集中普遍存在的问题。本工作通过引入一种自适应保形推断方法来解决这一开放性问题，该方法能够有效处理由随机标签噪声引起的可交换性偏差，即使在那些具有挑战性的场景下也能生成信息丰富的预测集合，并具有严格的边际覆盖保证。我们通过大量数值实验验证了该方法的有效性，实验在合成数据集和真实数据集（包括BigEarthNet和CIFAR-10H）上进行。

    arXiv:2501.18060v2 Announce Type: replace-cross  Abstract: Conformal inference provides a rigorous statistical framework for uncertainty quantification in machine learning, enabling well-calibrated prediction sets with precise coverage guarantees for any classification model. However, its reliance on the idealized assumption of perfect data exchangeability limits its effectiveness in the presence of real-world complications, such as low-quality labels---a widespread issue in modern large-scale data sets. This work tackles this open problem by introducing an adaptive conformal inference method capable of efficiently handling deviations from exchangeability caused by random label noise, leading to informative prediction sets with tight marginal coverage guarantees even in those challenging scenarios. We validate our method through extensive numerical experiments demonstrating its effectiveness on synthetic and real data sets, including BigEarthNet and CIFAR-10H.
    
[^65]: 从线性到可线性化优化：一个新颖框架及其在平稳与非平稳DR-次模优化中的应用

    From Linear to Linearizable Optimization: A Novel Framework with Applications to Stationary and Non-stationary DR-submodular Optimization

    [https://arxiv.org/abs/2405.00065](https://arxiv.org/abs/2405.00065)

    本文提出上可线性化/可二次化函数的新框架及通用元算法，将线性/二次最大化算法统一转换为凹优化和DR-次模优化算法，并在多种反馈设置下获得了动态和自适应遗憾保证，改进了现有最优结果。

    

    本文提出了上可线性化/可二次化函数的概念，这是一类在包括单调与非单调情形在内的多种设置下扩展了凹性与DR-次模性的函数。文章设计了一种通用的元算法，能够将线性/二次最大化算法转换为优化上可线性化/可二次化函数的算法，从而为解决凹优化和DR-次模优化问题提供了统一的方法。论文进一步将这些结果扩展到多种反馈设置，实现了半强盗/一阶反馈与强盗/零阶反馈之间的转换，以及一阶/零阶反馈与半强盗/强盗反馈之间的相互转换。借助该框架，以现有凸优化算法作为基础算法推导出了新的算法，在多种情形下改进了最先进的结果，并为DR-次模最大化问题获得了动态遗憾和自适应遗憾保证。

    arXiv:2405.00065v4 Announce Type: replace-cross  Abstract: This paper introduces the notion of upper-linearizable/quadratizable functions, a class that extends concavity and DR-submodularity in various settings, including monotone and non-monotone cases. A general meta-algorithm is devised to convert algorithms for linear/quadratic maximization into ones that optimize upper-linearizable/quadratizable functions, offering a unified approach to tackling concave and DR-submodular optimization problems. The paper extends these results to multiple feedback settings, facilitating conversions between semi-bandit/first-order feedback and bandit/zeroth-order feedback, as well as between first/zeroth-order feedback and semi-bandit/bandit feedback. Leveraging this framework, new algorithms are derived using existing results as base algorithms for convex optimization, improving upon state-of-the-art results in various cases. Dynamic and adaptive regret guarantees are obtained for DR-submodular maxi
    
[^66]: 常步长非光滑压缩随机逼近的预极限耦合与稳态收敛

    Prelimit Coupling and Steady-State Convergence of Constant-stepsize Nonsmooth Contractive SA

    [https://arxiv.org/abs/2404.06023](https://arxiv.org/abs/2404.06023)

    该论文提出预极限耦合技术，证明了常数步长非光滑压缩随机逼近（含Q-learning）的稳态收敛性，并发现其渐近偏差与步长的平方根成正比，从而可利用Richardson-Romberg外推法有效减少偏差。

    

    受Q-learning启发，我们研究了具有常数步长的非光滑压缩随机逼近（SA）。我们关注两类重要的动力学：1）带加性噪声的非光滑压缩SA，以及2）同步和异步Q-learning，其同时具有加性和乘性噪声。对于这两类动力学，我们在Wasserstein距离下建立了迭代序列到平稳极限分布的弱收敛性。此外，我们提出了一种预极限耦合技术来建立稳态收敛，并刻画了当步长趋于零时平稳分布的极限。利用这一结果，我们推导出非光滑SA的渐近偏差与步长的平方根成正比，这与光滑SA形成鲜明对比。这种偏差刻画使得可以在非光滑SA中使用Richardson-Romberg外推法来减少偏差。

    arXiv:2404.06023v3 Announce Type: replace-cross  Abstract: Motivated by Q-learning, we study nonsmooth contractive stochastic approximation (SA) with constant stepsize. We focus on two important classes of dynamics: 1) nonsmooth contractive SA with additive noise, and 2) synchronous and asynchronous Q-learning, which features both additive and multiplicative noise. For both dynamics, we establish weak convergence of the iterates to a stationary limit distribution in Wasserstein distance. Furthermore, we propose a prelimit coupling technique for establishing steady-state convergence and characterize the limit of the stationary distribution as the stepsize goes to zero. Using this result, we derive that the asymptotic bias of nonsmooth SA is proportional to the square root of the stepsize, which stands in sharp contrast to smooth SA. This bias characterization allows for the use of Richardson-Romberg extrapolation for bias reduction in nonsmooth SA.
    
[^67]: 神经网络逼近RKHS函数型

    Approximation of RKHS Functionals by Neural Networks

    [https://arxiv.org/abs/2403.12187](https://arxiv.org/abs/2403.12187)

    本文研究了使用神经网络逼近再生核希尔伯特空间（RKHS）上的函数型，并建立了逼近的普适性，推导了逆多重二次、高斯和Sobolev核引起的误差界限，证明神经网络可以准确逼近广义函数线性模型中的回归映射。

    

    受到时间序列和图像等丰富功能性数据的启发，人们越来越感兴趣将这些数据整合到神经网络中，并从函数空间到R（即函数型）学习映射。本文研究了使用神经网络逼近再生核希尔伯特空间（RKHS）上的函数型。我们建立了对RKHS上函数型逼近的普适性。具体来说，我们推导了通过逆多重二次、高斯和Sobolev核引起的明确误差界限。此外，我们将我们的研究应用于函数回归，证明了神经网络可以准确逼近广义函数线性模型中的回归映射。现有的功能性学习作品需要积分型基函数展开与一组预定义的基函数。通过在RKHS中利用插值正交投影，我们提出的网络是...

    arXiv:2403.12187v1 Announce Type: cross  Abstract: Motivated by the abundance of functional data such as time series and images, there has been a growing interest in integrating such data into neural networks and learning maps from function spaces to R (i.e., functionals). In this paper, we study the approximation of functionals on reproducing kernel Hilbert spaces (RKHS's) using neural networks. We establish the universality of the approximation of functionals on the RKHS's. Specifically, we derive explicit error bounds for those induced by inverse multiquadric, Gaussian, and Sobolev kernels. Moreover, we apply our findings to functional regression, proving that neural networks can accurately approximate the regression maps in generalized functional linear models. Existing works on functional learning require integration-type basis function expansions with a set of pre-specified basis functions. By leveraging the interpolating orthogonal projections in RKHS's, our proposed network is 
    
[^68]: 在一般希尔伯特空间中使用随机梯度下降学习算子

    Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces

    [https://arxiv.org/abs/2402.04691](https://arxiv.org/abs/2402.04691)

    本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。

    

    本研究探讨了利用随机梯度下降（SGD）在一般希尔伯特空间中学习算子的方法。我们提出了针对目标算子的弱和强规则条件，以描述其内在结构和复杂性。在这些条件下，我们建立了SGD算法的收敛速度的上界，并进行了极小值下界分析，进一步说明我们的收敛分析和规则条件定量地刻画了使用SGD算法解决算子学习问题的可行性。值得强调的是，我们的收敛分析对于非线性算子学习仍然有效。我们证明了SGD估计器将收敛于非线性目标算子的最佳线性近似。此外，将我们的分析应用于基于矢量值和实值再生核希尔伯特空间的算子学习问题，产生了新的收敛结果，从而完善了现有文献的结论。

    This study investigates leveraging stochastic gradient descent (SGD) to learn operators between general Hilbert spaces. We propose weak and strong regularity conditions for the target operator to depict its intrinsic structure and complexity. Under these conditions, we establish upper bounds for convergence rates of the SGD algorithm and conduct a minimax lower bound analysis, further illustrating that our convergence analysis and regularity conditions quantitatively characterize the tractability of solving operator learning problems using the SGD algorithm. It is crucial to highlight that our convergence analysis is still valid for nonlinear operator learning. We show that the SGD estimator will converge to the best linear approximation of the nonlinear target operator. Moreover, applying our analysis to operator learning problems based on vector-valued and real-valued reproducing kernel Hilbert spaces yields new convergence results, thereby refining the conclusions of existing litera
    
[^69]: 不公平的实用程序及其改进的第一步

    Unfair Utilities and First Steps Towards Improving Them. (arXiv:2306.00636v1 [stat.ML])

    [http://arxiv.org/abs/2306.00636](http://arxiv.org/abs/2306.00636)

    该论文提出了一个新的公平框架——考虑政策优化哪个效用，定义了信息价值公平，提出不应使用不满足这一标准的实用程序，并探讨了修改实用程序以满足此公平标准可能对最优政策产生的影响。

    

    许多公平标准对政策或预测器的选择进行限制。在这项工作中，我们提出了一个不同的思考公平的框架：我们考虑政策正在优化哪个效用，而不是限制政策或预测器的选择。我们定义了信息价值公平，并建议不使用不满足此标准的实用程序。我们描述了如何修改实用程序以满足这种公平标准，并讨论了这可能对相应的最优政策产生的影响。

    Many fairness criteria constrain the policy or choice of predictors. In this work, we propose a different framework for thinking about fairness: Instead of constraining the policy or choice of predictors, we consider which utility a policy is optimizing for. We define value of information fairness and propose to not use utilities that do not satisfy this criterion. We describe how to modify a utility to satisfy this fairness criterion and discuss the consequences this might have on the corresponding optimal policies.
    

