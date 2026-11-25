# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [U-Statistics-Powered LLM Reasoning](https://arxiv.org/abs/2611.33064) | 该论文利用 GRPO 策略梯度的 U 统计量结构，将组大小选择转化为方差比估计这一经典统计问题，并提出考虑分母估计不确定性的悲观比率估计量，从而以系统且稳定的方式确定这一关键超参数。 |
| [^2] | [Towards Realistic Evaluation of Uncertainty Quantification for LLM-Based Recommender Systems](https://arxiv.org/abs/2611.33036) | 该论文提出了一个固定排序基准，在统一的输出契约下将八种异构的不确定性量化信号作为记录级监控器进行评估，从而首次实现了对基于LLM的推荐系统中不确定性量化方法的公平、现实的可比性评估。 |
| [^3] | [Online V-Learning with Adaptive Longitudinal Data](https://arxiv.org/abs/2611.32947) | 本文针对移动健康中策略在线批量更新所导致的自适应收集、患者轨迹相依的数据，在可变确定性批次大小设定下建立了V学习的统计理论并证明了一致性。 |
| [^4] | [PHM-ensemble: Scalable and Stable Probabilistic Multiresolution Clustering via Ensemble Density Estimation](https://arxiv.org/abs/2611.32945) | 该论文提出PHM-ensemble框架，通过集成密度估计汇集多个混合模型并对成分进行层次化合并，实现了同时兼顾保真度、可解释性、稳定性和可扩展性的概率多分辨率聚类。 |
| [^5] | [SGD with Large Learning Rates: General Analysis & Improved Rates](https://arxiv.org/abs/2611.32762) | 本文证明了大学习率的SGD在随机梯度噪声存在下仍能加速收敛，对线性预测器的交叉熵损失和两层神经网络建立了加速的O(T^-2)收敛速率，并在稳定区域内给出了更紧致的收敛保证。 |
| [^6] | [Latent Class Mixed Model Variational Autoencoders for Scalable Phenotyping of Health Trajectories](https://arxiv.org/abs/2611.32391) | 提出LCMM-VAE，一种以类别特定线性混合模型的混合作为潜变量先验的深度生成模型，通过闭式变分EM算法，实现了对高维、不规则采样纵向健康轨迹的可扩展且临床可解释的聚类表型分析。 |
| [^7] | [From the ELBO to Maximum Likelihood: Convergence of Variational Autoencoders for Nonlinear Mixed-Effects Modeling](https://arxiv.org/abs/2611.32372) | 本文对用于非线性混合效应建模的VAE-NLME算法进行了收敛性分析，证明了其基于ELBO的版本无需精确训练编码器即可几乎必然收敛到ELBO的稳定点，且最终算法通过重要性加权充分统计量收敛到极大似然意义下的解。 |
| [^8] | [Seldon TFM family: Seldon-Mini, Seldon and Seldon+](https://arxiv.org/abs/2611.32257) | Seldon 表格基础模型家族仅用合成数据训练，在多项表格基准测试中名列前茅，其开源模型 Seldon-Mini 和 Seldon 同时支持分类与回归，并在准确性与推理速度的权衡上处于帕累托前沿。 |
| [^9] | [EnSurv: Deep Generative Survival Modeling with Application to Kidney Graft Outcomes](https://arxiv.org/abs/2611.32127) | 该研究提出了深度生成式生存分析框架EnSurv，利用结合Kaplan-Meier尾部补全的能量评分处理右删失数据，以捕捉非线性临床协变量交互作用，从而准确预测肾移植后移植物失效时间，服务于器官分配优化与个体化免疫抑制治疗。 |
| [^10] | [Metric-Dependent Scaling Laws for Sketched Multiclass Softmax Regression](https://arxiv.org/abs/2611.31995) | 该论文在高斯师生模型下证明，草图化多分类Softmax回归中交叉熵超额风险与0-1超额风险共享由分辨率项加“有效学习维度除以样本量”构成的共同谱尺度，并据此给出率最优的草图规模与训练时程选择。 |
| [^11] | [A Minimax Optimal Leave One Out Bound for Bounded Losses](https://arxiv.org/abs/2611.31972) | 本文针对有界损失下的留一预测问题给出了极小极大最优损失界，证明超额损失阶为 $\min\{B, \sqrt{Br\log M/n} + B\log M/n\}$，且单个确定性规则即可对所有 $r$ 达到该速率，并同时给出了匹配的下界。 |
| [^12] | [The Generalized Volterra Signature](https://arxiv.org/abs/2611.31967) | 本文提出广义Volterra签名及其新的代数编码结构与广义Chen恒等式，并首次建立了几何Volterra签名的shuffle积，从而导出其万能逼近定理。 |
| [^13] | [DUST: Directional Uncertainty-aware and Scale-invariant Transfer Learning](https://arxiv.org/abs/2611.31948) | 提出了一种基于源回归参数尺度不变方向分布的概率迁移学习框架DUST，能够同时纳入源参数的不确定性并实现尺度不变且稳健的迁移学习。 |
| [^14] | [Meta-Amortized Cognitive Modeling](https://arxiv.org/abs/2611.31914) | 该论文提出CogFormerV2元摊销框架，用单个网络一次性将贝叶斯推断摊销至整个认知模型空间，仅需在单GPU上训练约一天，即可对涵盖二元、多选项和连续决策的超过146亿个模型配置快速给出准确且校准良好的后验分布，从而免去每次模型修改后重新模拟与训练的成本。 |
| [^15] | [A General Framework for E-process Stopping Rules in Best-Arm Identification](https://arxiv.org/abs/2611.31898) | 该论文提出了最优臂识别中e-过程停止规则的通用框架，通过e-过程分解将最优性条件分为三个部分并证明了渐近最优性的充分条件，进而推导出广义线性老虎机问题的渐近最优算法。 |
| [^16] | [Generating health variables for a synthetic population of Santa Catarina, Brazil, via ensemble learning and a calibrated linear program](https://arxiv.org/abs/2611.31726) | 本文提出一种将逻辑回归、随机森林与支持向量机的集成学习与校准线性规划相结合的方法，为巴西圣卡塔琳娜州的合成人口生成吸烟、饮酒、锻炼、自评健康和慢性病等健康变量，并证明在温和条件下可省去惩罚项从而精确再现调查比例，在PNS数据交叉验证中取得最佳效果。 |
| [^17] | [Online Statistical Inference for Derivative-Free Constrained Stochastic Optimization](https://arxiv.org/abs/2611.31687) | 本文提出一种基于随机投影的免导数随机序列二次规划方法（SDF-SSQP），在无导数信息的等式约束随机优化问题中，将每轮迭代成本降至 $O((d+m)^2)$ 的同时实现有效的在线统计推断。 |
| [^18] | [A Kernel Framework for Linear Operator Equations](https://arxiv.org/abs/2611.31626) | 该论文提出了一个统一的再生核希尔伯特空间框架，将带非局部项或异质约束的线性算子方程的求解归结为单个线性系统，并借助Nyström近似降低大规模配置点问题的计算成本。 |
| [^19] | [Bayesian Data Integration and Model Updating in Probabilistic Digital Twins](https://arxiv.org/abs/2611.31620) | 本文提出了一种用于概率数字孪生的序贯贝叶斯更新框架，通过区分属性数据与行为数据并引入基于主动学习的代理模型策略，实现了多源数据的高效融合与计算成本的显著降低。 |
| [^20] | [Instance-Dependent Bandit Convex Optimization in One Dimension](https://arxiv.org/abs/2611.31608) | 本文在一维赌博机凸优化中首次刻画了实例依赖样本复杂度，引入新量 Δ_ε(f) 证明定位极小值点问题可获得依赖该量的加速，同时惊人地发现简单遗憾设定下不存在任何实例依赖的改进。 |
| [^21] | [ExitLUT: Boolean Early Exits for Efficient FPGA-Based Neural Network Inference](https://arxiv.org/abs/2611.31540) | ExitLUT用直接映射到FPGA LUT的可学习布尔函数替代softmax置信度退出决策，将节省的退出路径资源重新投入主数据通路，在保持精度不变的前提下实现高达4.08倍的吞吐量提升。 |
| [^22] | [Diagnosing Epistemic Uncertainty in Neural Operators: Effects of Initialization, Discretization, and Function Sampling](https://arxiv.org/abs/2611.31526) | 该论文系统研究了神经算子中认知不确定性的三大来源（训练函数采样、输出离散化、模型初始化），通过训练约46,000个模型并构建集成，定量刻画了各来源对算子学习性能差异的独立贡献。 |
| [^23] | [SOTS: Signature-based Optimal Transport Selection](https://arxiv.org/abs/2611.31525) | SOTS提出了一种无监督核心集选择框架，通过将不规则多变量时间序列映射为截断路径签名并贪心最小化子集与全量数据之间的切片Wasserstein距离来选择训练数据，同时提供了随机核心集的收敛性理论保证。 |
| [^24] | [Transformers with Signatures: Towards Universal In-Context Learning in Path Space](https://arxiv.org/abs/2611.31439) | 该论文利用路径签名特征建立了面向路径空间的上下文学习理论框架，显式构造了可模拟上下文Lasso的每层双注意力头因果softmax Transformer，并给出超越IID数据设定的有限样本预测误差界。 |
| [^25] | [Local subspace encoding for spectral prediction: from TabPFN to operator-adaptive calibration](https://arxiv.org/abs/2611.31245) | 该研究证实局部振动子空间编码（LVSE）表示方法在nirs4all生态系统的光谱数据集上能显著降低TabPFN的RMSE并提升分类加权F1，且其价值可扩展至经典定标方法。 |
| [^26] | [$K$-Midranks: A robust clustering algorithm for heavy-tailed data with applications to spectroscopy and finance](https://arxiv.org/abs/2611.30953) | 本文提出了一种名为 $K$-中位秩的鲁棒聚类算法，通过将观测值替换为逐坐标秩来降低重尾数据中极端值的影响，在柯西型重尾数据上优于传统聚类方法。 |
| [^27] | [Sparse Hodge Preference Optimization: Fine-Tuning Language Models toward Alignment beyond Transitive Preferences](https://arxiv.org/abs/2611.30135) | 本文提出稀疏霍奇偏好优化（SHPO），借助组合霍奇分解理论为DPO引入稀疏反对称旋度残差，使其能够捕捉人类偏好数据中的非传递性与循环结构，并证明其严格扩大了DPO的假设类。 |
| [^28] | [How Do DiTs Denoise? An Aggregator-Predictor View of Attention and MLPs](https://arxiv.org/abs/2610.29854) | 论文提出将扩散Transformer视为“聚合器-预测器”架构的新解释框架——AdaLN调制的注意力充当依赖输入与噪声的上下文聚合器，MLP充当可学习键-值表示的patch预测器，并从理论上证明基于局部上下文的聚合能以指数衰减的误差逼近全局贝叶斯去噪器。 |
| [^29] | [Announced Breaks Separate Conformal Reliability from Frequency Calibration](https://arxiv.org/abs/2610.23917) | 本文证明预先公告的结构性变点使“下一区间的保形可靠性”与“封闭窗口的频率校准”这两种重置审计目标可以分离，并提出在公告时点重启经验分位数与自适应递归的方法，前者达到1/√n的minimax可靠性速率、后者达到1/n的频率差距，且实验中覆盖率误差显著优于自适应基线（0.035对0.052）。 |
| [^30] | [BridgeTwist: Twisting Schr\"odinger Bridges for Training-Free Conditional Sampling](https://arxiv.org/abs/2610.18658) | 提出了BridgeTwist，一种基于带温度调节扭转序贯蒙特卡洛方案的免训练条件采样器，可复用预训练薛定谔桥生成模型，利用由速度场与得分场构造的闭式即插即用扭转代理实现条件后验采样。 |
| [^31] | [Beyond Point Forecasts: A Survey on Probabilistic Forecasting for Time Series and Spatiotemporal Data](https://arxiv.org/abs/2609.13345) | 这篇综述通过按不确定性在预测流程中引入的位置和方式建立统一分类框架，整合了时间序列与时空概率预测中分散的统计建模、机器学习和深度生成建模方法，并涵盖了集成校准、贝叶斯建模、分布回归以及新兴的时间序列基础模型等范式。 |
| [^32] | [qshap: Fast Shapley Decomposition of $R^2$ for Gradient-Boosted Trees](https://arxiv.org/abs/2608.24104) | qshap是一种新工具，通过Shapley分解梯度提升树中的$R^2$值，实现特征和观测对模型性能贡献的全局量化。 |
| [^33] | [Identifiability Limits of Physics-Informed Inference for Spatial Stochastic Dynamics from Static Snapshots](https://arxiv.org/abs/2607.01749) | 本研究通过结构性可辨识性分析揭示了利用物理信息机器学习从分子静态空间快照推断空间随机动力学的根本极限：分布式源本质上不可辨识，而点源（如转录位点）可恢复可辨识性，且这些极限还受边界条件、空间正则性等建模选择的制约。 |
| [^34] | [A Unified Heat-Ball Framework for Fokker--Planck Equations: Mean-Value Identities, Exact Sampling, and Error Analysis](https://arxiv.org/abs/2606.27954) | 本文提出了一个统一的非齐次热球框架，为具有一般非线性、时变漂移的 Fokker–Planck 方程推导出密度、对数密度、熵密度与得分函数的局部均值恒等式，从而为扩散生成模型中漂移结构的分析、局部化采样以及低密度区域的处理提供了新工具。 |
| [^35] | [Jacobian-Velocity Bounds for Deployment Risk Under Covariate Drift](https://arxiv.org/abs/2605.04932) | 该论文通过时域庞加莱不等式与雅可比-速度定理建立了协变量漂移下冻结预测器长期部署的风险上界，并据此提出漂移对齐切向正则化（DTR）及配套监控指标来有效控制部署风险。 |
| [^36] | [Flexible Matrix Estimation with Side Information](https://arxiv.org/abs/2603.24833) | 该论文提出将高维矩阵分解为行与列特征的非线性交互项、行特征驱动成分、列特征驱动成分及残余低秩结构四个互补成分的灵活估计框架，结合筛投影与核范数惩罚分别估计各成分，克服了归纳矩阵补全等现有方法的限制性假设。 |
| [^37] | [Unbiased Approximate Vector-Jacobian Products for Efficient Backpropagation](https://arxiv.org/abs/2602.14701) | 本文提出在反向传播中用随机化的无偏近似替代精确的向量-雅可比积，从理论上分析了精度与成本之间的权衡并证明了稀疏约束下最小方差的最优性，实验验证了该方法能有效降低深度学习的训练成本。 |
| [^38] | [On Approximate Computation of Critical Points](https://arxiv.org/abs/2601.21917) | 该论文证明了即使对系数极简单的非凸三次多项式，临界点的粗略近似计算也是计算难解的——除非P=NP，否则不存在多项式时间算法能输出近似临界点。 |
| [^39] | [Generalized Guarantees for Variational Inference in the Presence of Even and Elliptical Symmetry](https://arxiv.org/abs/2511.01064) | 该论文将反向KL散度变分推断在偶对称下恢复均值、椭圆对称下恢复相关矩阵的保证，从对数凹目标分布推广到更一般的单峰（含重尾）目标分布，并扩展至仅部分对称的情形。 |
| [^40] | [Efficient Prediction of pass@k Scaling in Large Language Models](https://arxiv.org/abs/2510.05197) | 本文提出了一种稳健的估计框架，能够在远小于大规模尝试次数的采样预算下高效准确地预测大语言模型的pass@k缩放行为，从而帮助模型提供商和监管机构评估模型的能力与风险。 |
| [^41] | [Perfect Clustering in Extremely Sparse Multilayer Networks with Groupwise Subspace Structure](https://arxiv.org/abs/2507.19423) | 本文提出一个极其广义的多层网络聚类模型——各层遵循广义随机点积图模型且可按潜在子空间结构分组，并证明了即使各层网络极度稀疏、逐层分析失效时，仍能完美地恢复具有不同子空间结构的层分组。 |
| [^42] | [Sequential Monte Carlo approximations of Wasserstein--Fisher--Rao gradient flows](https://arxiv.org/abs/2506.05905) | 该论文提出了一种基于Wasserstein-Fisher-Rao几何、利用重要性采样与序贯蒙特卡洛的新算法来近似KL散度梯度流以实现分布采样，并通过广泛的实证研究揭示了该算法优于其他流行蒙特卡洛算法的适用场景。 |
| [^43] | [Causal pieces: analysing and improving spiking neural networks piece by piece](https://arxiv.org/abs/2504.14015) | 提出“因果片段”新概念来分析脉冲神经网络，证明片段内输出脉冲时间对输入和参数局部Lipschitz连续，且因果片段数量可作为衡量SNN逼近能力的有效指标。 |
| [^44] | [Fast Sketching of Polynomial Kernels of Polynomial Degree](https://arxiv.org/abs/2108.09420) | 本文提出了一种新的无意识草图方法，消除了运行时间对多项式核次数的依赖，大幅改进了神经正切核、反余弦核等慢增长核的近似运行速度。 |
| [^45] | [Individual Treatment Effects in Extreme Regimes.](http://arxiv.org/abs/2306.11697) | 本文提出了一种新的框架，通过测量潜在结果在存在或缺乏治疗的情况下的尾部衰减率变化，来估计极端环境下的个体治疗效果（ITE$_2$）。 |

# 详细

[^1]: 基于 U 统计量的大语言模型推理

    U-Statistics-Powered LLM Reasoning

    [https://arxiv.org/abs/2611.33064](https://arxiv.org/abs/2611.33064)

    该论文利用 GRPO 策略梯度的 U 统计量结构，将组大小选择转化为方差比估计这一经典统计问题，并提出考虑分母估计不确定性的悲观比率估计量，从而以系统且稳定的方式确定这一关键超参数。

    

    自 DeepSeek 提出组相对策略优化（GRPO）以来，该方法已被广泛用于训练大型推理模型和智能体，并催生了众多变体。然而，组大小作为这类 GRPO 型算法中的一个关键超参数，目前仍在很大程度上依靠反复试错来选择。本文利用 GRPO 策略梯度的 U 统计量结构，将组大小的选择重新表述为一个经典的统计问题——方差比估计。当分母较小时，该问题尤其具有挑战性，导致朴素的代入估计量不稳定。为应对这一挑战，我们提出了一种考虑分母估计不确定性的悲观比率估计量，并刻画了该估计量在三种样本量情形下的统计性质。实证方面，我们在来自 Reasoning Gym 的合成任务和真实世界基准上采用了基于 jackknife 的悲观规则（原文在此处截断）。

    arXiv:2611.33064v1 Announce Type: new  Abstract: Since its introduction by DeepSeek, group relative policy optimization (GRPO) has been widely adopted for training large reasoning models and agents, spurring a proliferation of its variants. Yet the group size, a crucial hyperparameter in these GRPO-type algorithms, remains largely chosen by trial and error. Exploiting the U-statistic structure of GRPO's policy gradient, we recast the selection of the group size as a classical statistical problem of variance ratio estimation. This problem is particularly challenging when the denominator of the ratio is small, rendering naive plug-in estimators unstable. To address this challenge, we propose a pessimistic ratio estimator that accounts for the denominator's estimation uncertainty. We characterise the statistical properties of this estimator across three sample-size regimes. Empirically, using %a jackknife-based pessimistic rule on synthetic tasks from Reasoning Gym and real-world benchmar
    
[^2]: 迈向基于大语言模型的推荐系统中不确定性量化的现实评估

    Towards Realistic Evaluation of Uncertainty Quantification for LLM-Based Recommender Systems

    [https://arxiv.org/abs/2611.33036](https://arxiv.org/abs/2611.33036)

    该论文提出了一个固定排序基准，在统一的输出契约下将八种异构的不确定性量化信号作为记录级监控器进行评估，从而首次实现了对基于LLM的推荐系统中不确定性量化方法的公平、现实的可比性评估。

    

    大语言模型（LLM）为推荐系统提供了灵活的接口。然而，其给出的排序结果需要可靠的实例级不确定性监控。在这种场景下，比较不确定性量化（UQ）方法变得具有挑战性，因为LLM排序器缺乏标准的物品分类头。其打分可能依赖于选项标签、提示词、生成过程以及可用的用户和物品证据。现有方法从不同的来源提取不确定性，并将其应用于推荐过程的不同阶段。一些方法评估已生成的排序结果，而另一些方法则利用不确定性来改变排序本身，这使得直接比较变得困难。本文引入了一个固定排序基准，在共享的输出契约下，将八种异构的UQ信号作为记录级监控器进行评估。我们在四个推荐数据集和两个LLM骨干网络上，在共享的…（原文在此处截断）

    arXiv:2611.33036v1 Announce Type: cross  Abstract: Large language models (LLMs) provide a flexible interface for recommendation systems. However, their issued rankings require reliable instance-level uncertainty monitoring. In this scenario, comparing uncertainty quantification (UQ) methods becomes challenging because LLM rankers lack a standard item-classification head. Their scores may depend on option labels, prompts, generation procedures, and the available user and item evidence. Existing methods derive uncertainty from different sources and apply it at different stages of the recommendation process. Some methods evaluate an issued ranking, whereas others use uncertainty to change the ranking itself, making direct comparison difficult. Here we introduce a fixed-ranking benchmark which evaluates eight heterogeneous UQ signals as record-level monitors under a shared output contract. We evaluate these signals across four recommendation datasets and two LLM backbones under a shared fi
    
[^3]: 自适应纵向数据下的在线V学习

    Online V-Learning with Adaptive Longitudinal Data

    [https://arxiv.org/abs/2611.32947](https://arxiv.org/abs/2611.32947)

    本文针对移动健康中策略在线批量更新所导致的自适应收集、患者轨迹相依的数据，在可变确定性批次大小设定下建立了V学习的统计理论并证明了一致性。

    

    用于估计最优动态治疗方案的强化学习方法正日益应用于移动健康领域，在这类场景中治疗决策需要持续不断地做出。Luckett等人（2020）提出了V学习，该方法可从离线轨迹中估计最优策略。在实际应用中，移动健康策略是在线更新的。批量更新——即在每个新数据块到达后重新估计策略——是介于离线学习与每次观测后更新之间的一种可行折中方案。然而，由于每个批次的处理策略是基于早期批次中跨患者汇集的数据估计得到的，由此产生的数据是自适应收集的，且患者轨迹之间相互依赖，这一情形是现有离线理论未能涵盖的。我们在批次大小可变且为确定性的设定下，为V学习建立了统计理论。对于每个固定批次，随着患者数量的增长，我们证明了估计量的一致性（原文摘要在此处截断）。

    arXiv:2611.32947v1 Announce Type: cross  Abstract: Reinforcement learning methods for estimating optimal dynamic treatment regimes are increasingly applied in mobile health (mHealth), where treatment decisions are made continually. Luckett et al. (2020) proposed V-learning, which estimates an optimal policy from offline trajectories. In practice, mHealth policies are updated online. Batch updating, in which the policy is re-estimated after each block of new data, is a tractable middle ground between offline learning and updating after every observation. However, because each batch's treatment policy is estimated from data pooled across patients in earlier batches, the resulting data are adaptively collected and patient trajectories are dependent, a setting that offline theory does not address. We develop statistical theory for V-learning in this setting with varying, deterministic batch sizes. For each fixed batch, as the number of patients grows, we establish uniform consistency of th
    
[^4]: PHM-ensemble：基于集成密度估计的可扩展且稳定的概率多分辨率聚类

    PHM-ensemble: Scalable and Stable Probabilistic Multiresolution Clustering via Ensemble Density Estimation

    [https://arxiv.org/abs/2611.32945](https://arxiv.org/abs/2611.32945)

    该论文提出PHM-ensemble框架，通过集成密度估计汇集多个混合模型并对成分进行层次化合并，实现了同时兼顾保真度、可解释性、稳定性和可扩展性的概率多分辨率聚类。

    

    聚类是跨科学领域探索性数据分析的基础工具。我们认为聚类方法应从四个维度进行评估：对数据底层结构的保真度、所恢复群组的可解释性、在数据扰动和算法随机性下的稳定性，以及对现代科学数据集的可扩展性。然而，现有方法很少能同时满足这四项标准。为弥补这一空白，我们提出了PHM-ensemble，这是一个用于多分辨率聚类的通用概率框架，它通过汇集来自集成拟合的混合模型来估计数据的底层密度，随后通过对混合成分的层次化合并构建聚类层次结构。在模拟数据集和MNIST基准测试中，PHM-ensemble与真实标签的一致性达到或超过广泛使用的聚类算法，在多次运行中产生更稳定的结果，并能揭示复杂的底层……（原文摘要在此处截断）

    arXiv:2611.32945v1 Announce Type: cross  Abstract: Clustering is a foundational tool for exploratory data analysis across the sciences. We argue that clustering methods should be evaluated along four dimensions: fidelity to the data's underlying structure, interpretability of the recovered groups, stability under data perturbations and algorithmic randomness, and scalability to modern scientific datasets. Yet existing methods rarely satisfy all four criteria. To address this gap, we introduce PHM-ensemble, a general probabilistic framework for multiresolution clustering that estimates the underlying data density by pooling mixture models from an ensemble of fits and subsequently constructs a clustering hierarchy through hierarchical merging of the mixture components. On simulated datasets and the MNIST benchmark, PHM-ensemble matches or outperforms widely used clustering algorithms in agreement with ground truth, produces more stable results across runs, and reveals complex underlying 
    
[^5]: 大学习率下的随机梯度下降：一般性分析与改进的收敛速率

    SGD with Large Learning Rates: General Analysis & Improved Rates

    [https://arxiv.org/abs/2611.32762](https://arxiv.org/abs/2611.32762)

    本文证明了大学习率的SGD在随机梯度噪声存在下仍能加速收敛，对线性预测器的交叉熵损失和两层神经网络建立了加速的O(T^-2)收敛速率，并在稳定区域内给出了更紧致的收敛保证。

    

    随机梯度下降（SGD）是现代机器学习的核心。然而，其经典分析通常将步长限制在低于一个依赖于曲率的稳定性阈值以下。确定性优化领域的最新进展挑战了这一范式，表明适当选择的大步长不仅可以保持收敛性，还能加速梯度下降（GD）。这一现象在存在随机梯度噪声的情况下是否仍然成立，一直是一个悬而未决的问题（Wu et al., 2024）。在本工作中，我们对带有线性预测器的交叉熵损失以及两层神经网络肯定地回答了这一问题，建立了加速的 $\mathcal{O}(T^{-2})$ 收敛速率。我们分析的核心是对随机大步长动力学的细粒度刻画。这种对轨迹更精确的控制进一步在稳定区域内给出了紧致的收敛保证，改进了……

    arXiv:2611.32762v1 Announce Type: cross  Abstract: Stochastic Gradient Descent lies at the core of modern machine learning. However its classical analysis typically restricts the stepsize to be below a curvature-dependent stability threshold. Recent developments in deterministic optimization have challenged this paradigm, showing that suitably chosen large stepsizes can not only preserve convergence, but also accelerate Gradient Descent (GD). Whether this phenomenon persists in the presence of stochastic gradient noise has remained an open question (Wu et al., 2024). In this work, we answer this question affirmatively for the cross-entropy loss with linear predictors and two-layer neural networks, establishing an accelerated $\mathcal{O}(T^{-2})$ convergence rate. At the core of our analysis is a fine-grained characterization of the stochastic large-stepsize dynamics. This sharper control of the trajectory further yields tight convergence guarantees within the stable regime, improving 
    
[^6]: 用于健康轨迹可扩展表型分析的潜类别混合模型变分自编码器

    Latent Class Mixed Model Variational Autoencoders for Scalable Phenotyping of Health Trajectories

    [https://arxiv.org/abs/2611.32391](https://arxiv.org/abs/2611.32391)

    提出LCMM-VAE，一种以类别特定线性混合模型的混合作为潜变量先验的深度生成模型，通过闭式变分EM算法，实现了对高维、不规则采样纵向健康轨迹的可扩展且临床可解释的聚类表型分析。

    

    arXiv:2611.32391v1 公告类型：交叉 摘要：对纵向健康轨迹进行聚类需要处理高维测量数据、不规则的就诊时间以及受试者层面的异质性，同时还要保留临床可解释的结构。现有方法只能部分解决这一问题：潜类别混合模型能够产生可解释的、由协变量驱动的划分，但无法扩展到超过少数几个标志物的情形；而深度时间序列聚类虽然具备可扩展性，却牺牲了这种结构，并且通常要求统一的时间网格。我们提出了潜类别混合模型变分自编码器（LCMM-VAE），这是一种深度生成模型，其潜变量先验是类别特定的线性混合模型的混合，因此每个受试者的潜轨迹可以分解为其所属类别的均值轨迹和个体特异性偏差。我们推导了一种变分EM算法，其中的类别归属责任、随机效应和混合模型参数均以闭式解形式更新，并提供了可辨识性保证……

    arXiv:2611.32391v1 Announce Type: cross  Abstract: Clustering longitudinal health trajectories requires handling high-dimensional measurements, irregular visit times, and subject-level heterogeneity while retaining clinically interpretable structure. Existing methods address this only partially: latent-class mixed models yield an interpretable, covariate-driven partition but do not scale past a handful of markers, while deep time-series clustering scales at the cost of that structure and generally requires a common time grid. We introduce the Latent Class Mixed Model Variational Autoencoder (LCMM-VAE), a deep generative model whose latent prior is a mixture of class-specific linear mixed models, so that each subject's latent trajectory decomposes into its class mean trajectory and a subject-specific deviation. We derive a variational EM algorithm whose class responsibilities, random effects and mixed-model parameters are updated in closed form, and provide identifiability guarantees un
    
[^7]: 从ELBO到极大似然：非线性混合效应建模中变分自编码器的收敛性分析

    From the ELBO to Maximum Likelihood: Convergence of Variational Autoencoders for Nonlinear Mixed-Effects Modeling

    [https://arxiv.org/abs/2611.32372](https://arxiv.org/abs/2611.32372)

    本文对用于非线性混合效应建模的VAE-NLME算法进行了收敛性分析，证明了其基于ELBO的版本无需精确训练编码器即可几乎必然收敛到ELBO的稳定点，且最终算法通过重要性加权充分统计量收敛到极大似然意义下的解。

    

    非线性混合效应模型是药物计量学中的核心统计模型，在药物研发中为剂量选择、个体化给药以及监管决策提供支持。从统计学角度看，NLME建模是一个边缘化似然问题。变分自编码器通过神经网络编码器对难以求解的后验分布进行摊销式近似来解决此类问题，并使得AI组件能够集成到NLME建模之中。本文对VAE-NLME算法进行了收敛性分析。首先，证明了其基于证据下界（ELBO）的版本在几乎必然意义下收敛到ELBO的稳定点集合，且无需假设编码器被精确训练。其次，最终的VAE-NLME算法通过基于梯度的优化方法在ELBO上训练编码器，并将其作为重要性加权充分统计量的建议分布。该算法几乎必然收敛到……（原文在此处截断）

    arXiv:2611.32372v1 Announce Type: new  Abstract: Nonlinear mixed-effects (NLME) models are the central statistical models of pharmacometrics, where they support dose selection, individualized dosing and regulatory decisions in drug development. Statistically, NLME modeling is a marginalized likelihood problem. Variational autoencoders (VAEs) solve such problems with a neural network encoder that amortizes the approximation of the intractable posterior distribution, and allow the integration of AI components into NLME modeling. This paper provides the convergence analysis of the VAE-NLME algorithm. First, its version for the evidence lower bound (ELBO) is proven to converge almost surely to the set of stationary points of the ELBO, without assuming that the encoder is trained exactly. Second, the final VAE-NLME algorithm trains the encoder on the ELBO by gradient-based optimization and uses it as proposal for importance-weighted sufficient statistics. It converges almost surely to the s
    
[^8]: Seldon 表格基础模型家族：Seldon-Mini、Seldon 与 Seldon+

    Seldon TFM family: Seldon-Mini, Seldon and Seldon+

    [https://arxiv.org/abs/2611.32257](https://arxiv.org/abs/2611.32257)

    Seldon 表格基础模型家族仅用合成数据训练，在多项表格基准测试中名列前茅，其开源模型 Seldon-Mini 和 Seldon 同时支持分类与回归，并在准确性与推理速度的权衡上处于帕累托前沿。

    

    我们介绍了 Seldon，一个表格基础模型家族。我们的分类器 Seldon+ 在 TabBench（我们构建的包含 124 个分类数据集的基准测试）及其含缺失值的变体上，在 15 个模型中排名第一。在我们自行运行的公开基准测试中，它还在 BeyondArena IID 分类和 TabBench-Bio 上排名第一，在 TabArena 分类任务上排名第二。开源权重模型 Seldon-Mini（2700 万参数）和 Seldon（2.13 亿参数）均位于 TabArena 上 Elo 得分与推理时间的帕累托前沿：没有任何模型能够同时做到更准确且更快速。在 TabArena 的 109 个排行榜条目中，Seldon 排名第四，Seldon-Mini 排名第七；在 BeyondArena 的 IID 赛道上，Seldon 在 42 个条目中排名第二。Seldon-Mini 和 Seldon 同时支持分类与回归任务，采用与 Kumo-Tabular、TabFM 和 TabPFN-3.5 类似的架构，并且仅使用合成表格数据进行训练。

    arXiv:2611.32257v1 Announce Type: new  Abstract: We introduce Seldon, a family of tabular foundation models. Our classifier Seldon+ ranks first among 15 models on TabBench, our benchmark of 124 classification datasets, and on its variant with missing values. In our own runs of public benchmarks, it also ranks first on BeyondArena IID classification and TabBench-Bio, and second on TabArena classification. The open-weight models Seldon-Mini (27M parameters) and Seldon (213M parameters) both lie on the TabArena Pareto front of Elo against inference time: no entry is both more accurate and faster. On TabArena, Seldon ranks fourth and Seldon-Mini seventh among 109 leaderboard entries, and on the IID track of BeyondArena, Seldon ranks second among 42 entries. Seldon-Mini and Seldon handle both classification and regression, use an architecture similar to those of Kumo-Tabular, TabFM and TabPFN-3.5, and are trained on synthetic tables only.
    
[^9]: EnSurv：深度生成式生存建模及其在肾脏移植物结局中的应用

    EnSurv: Deep Generative Survival Modeling with Application to Kidney Graft Outcomes

    [https://arxiv.org/abs/2611.32127](https://arxiv.org/abs/2611.32127)

    该研究提出了深度生成式生存分析框架EnSurv，利用结合Kaplan-Meier尾部补全的能量评分处理右删失数据，以捕捉非线性临床协变量交互作用，从而准确预测肾移植后移植物失效时间，服务于器官分配优化与个体化免疫抑制治疗。

    

    肾移植是危及生命的终末期肾病的最佳治疗方式，与透析相比具有显著的生存优势。鉴于供体器官的严重短缺，准确预测移植物过早失效的时间是优化器官分配和制定个体化免疫抑制治疗方案的一项重要临床任务。本研究受一个全面的器官共享联合网络（UNOS）数据集的启发，该数据集包含243,443条肾移植记录，具有复杂的多区域结构和较高的右删失率。我们提出了EnSurv（生存engression），这是一种深度生成式生存分析框架，旨在捕捉复杂的非线性临床协变量交互作用，以建模移植物失效时间。在前加性噪声模型家族中，EnSurv通过优化结合了合并Kaplan-Meier尾部补全的能量评分来处理右删失问题。我们识别了合并c……（原文在此处截断）

    arXiv:2611.32127v1 Announce Type: cross  Abstract: Kidney transplantation is the optimal treatment for life-threatening end-stage renal disease, providing significant survival advantages over dialysis. Given the severe shortage of donor organs, accurately predicting the time to premature graft failure is an essential clinical task for optimizing organ allocation and tailoring individualized immunosuppressive therapies. This study is motivated by a comprehensive United Network of Organ Sharing (UNOS) dataset containing 243,443 kidney transplant records with complex multi-region structures and high right-censoring rates. We propose EnSurv (survival engression), a deep generative survival framework designed to capture complex, non-linear clinical covariate interactions for modeling time to graft failure. Within a pre-additive noise model (pre-ANM) family, EnSurv optimizes an energy score augmented with pooled Kaplan-Meier tail completion to handle right-censoring. We identify the pooled-c
    
[^10]: 草图化多分类Softmax回归中度量依赖的标度律

    Metric-Dependent Scaling Laws for Sketched Multiclass Softmax Regression

    [https://arxiv.org/abs/2611.31995](https://arxiv.org/abs/2611.31995)

    该论文在高斯师生模型下证明，草图化多分类Softmax回归中交叉熵超额风险与0-1超额风险共享由分辨率项加“有效学习维度除以样本量”构成的共同谱尺度，并据此给出率最优的草图规模与训练时程选择。

    

    我们研究了草图化多分类Softmax回归中，采用一般稳定确定性学习率的全批量梯度下降的标度律。在一个具有幂律特征谱与教师谱的高斯师生模型中，学生观测到一个随机特征草图，并在整个优化过程中重复使用有限的带标签样本。草图化通常使学生模型处于误设状态，因此分析必须同时考虑近似误差、非线性有限时间学习，以及由数据重复使用引入的固定经验扰动。在一个以指数概率出现的草图事件上，我们证明交叉熵超额风险与超额0-1风险共享一个共同的谱尺度，该尺度由一个分辨率项加上有效学习维度除以样本量构成；师生决策不一致程度则遵循该尺度的平方根。由此得到的平衡给出了率最优的草图规模与累积训练时程选择，合成实验支持了所预测的（标度律）……

    arXiv:2611.31995v1 Announce Type: new  Abstract: We study scaling laws for full-batch gradient descent with general stable deterministic learning rates in sketched multiclass softmax regression. In a Gaussian teacher student model with power law feature and teacher spectra, the student observes a random feature sketch and reuses a finite labeled sample throughout optimization. Sketching generally makes the student misspecified, so the analysis must jointly account for approximation, nonlinear finite time learning, and the fixed empirical perturbation induced by data reuse. On an exponentially likely sketch event, we show that cross entropy excess and excess zero one risk share a common spectral scale, consisting of a resolution term and an effective learned dimension divided by sample size; teacher student decision disagreement follows the square root of that scale. The resulting balance gives rate optimal sketch and cumulative horizon choices, and synthetic experiments support the pre
    
[^11]: 有界损失下留一法的极小极大最优界

    A Minimax Optimal Leave One Out Bound for Bounded Losses

    [https://arxiv.org/abs/2611.31972](https://arxiv.org/abs/2611.31972)

    本文针对有界损失下的留一预测问题给出了极小极大最优损失界，证明超额损失阶为 $\min\{B, \sqrt{Br\log M/n} + B\log M/n\}$，且单个确定性规则即可对所有 $r$ 达到该速率，并同时给出了匹配的下界。

    

    我们研究了在 $n$ 个固定输入点上、由 $M$ 个实值专家进行的适当留一预测问题。每个标签依次被隐藏，学习器利用其余标签在留出的输入点上选择某位专家的预测。我们确定了这类规则能够多好地匹配最佳单个专家在整个样本上的平均损失。对于任何有界损失——即关于绝对误差连续、非递减且在零点取值为零的函数——设其最大值为 $B$。在专家数量为 $M$、最佳专家平均损失至多为 $r$ 的样本类上，极小极大超额损失的阶为 $\min\{B, \sqrt{\frac{Br\log M}{n}}+\frac{B\log M}{n}\}$。对于每个专家类和每种损失，存在一个确定性规则对所有 $r\in[0,B]$ 都能达到该速率。相应的匹配下界即使在给定 $r$ 的随机适当规则下依然成立。

    arXiv:2611.31972v1 Announce Type: cross  Abstract: We study proper leave-one-out prediction with $M$ real-valued experts on $n$ fixed inputs. Each label is hidden in turn, and the learner uses the remaining labels to choose one of the experts' predictions at the held-out input. We determine how closely such a rule can match the average loss of the best single expert on the full sample. For any bounded loss that is a continuous nondecreasing function of absolute error and vanishes at zero, let $B$ denote its maximum value. Over classes of size $M$ and samples on which the best expert has average loss at most $r$, the minimax excess loss is of order $\min\{B, \sqrt{\frac{Br\log M}{n}}+\frac{B\log M}{n}\}$. For each class and loss, one deterministic rule attains this rate for every $r\in[0,B]$. The matching lower bound holds even for randomized proper rules given $r$.
    
[^12]: 广义Volterra签名

    The Generalized Volterra Signature

    [https://arxiv.org/abs/2611.31967](https://arxiv.org/abs/2611.31967)

    本文提出广义Volterra签名及其新的代数编码结构与广义Chen恒等式，并首次建立了几何Volterra签名的shuffle积，从而导出其万能逼近定理。

    

    在本文中，我们引入了路径 \(\bX\) 的广义Volterra签名 \(\gvsig(\bX)\)，它扩展了 [Hager, Harang, Pelizzari & Tindel, arXiv:2603.04525, 2026] 中引入的经典Volterra签名，而后者将Lyons的路径签名推广到了Volterra情形。我们建立了一种新的代数结构来编码广义Volterra签名，并建立了一个广义的Chen恒等式。通过将微分树限制为梯形树，我们引入了几何Volterra签名 \(\geovsig(\bX)\)，并建立了其签名分量之间的shuffle积（洗牌积）。据我们所知，这是首个针对Volterra签名的shuffle积，并使我们能够为几何Volterra签名提出一个万能逼近定理。对于一类特殊的核函数，我们证明了扩展路径 \(\Bar{\bX}\) 的存在性，并将其经典签名与几何Volterra签名联系起来。

    arXiv:2611.31967v1 Announce Type: cross  Abstract: In this paper, we introduce the generalized Volterra signature \(\gvsig(\bX)\) of a path \(\bX\), extending the classical Volterra signature introduced in [Hager, Harang, Pelizzari \& Tindel, arXiv:2603.04525, 2026], which extends Lyons' path signature to the Volterra setting. We establish a new algebraic structure to encode the generalized Volterra signature. We also establish a generalized Chen's identity. By restricting the differential trees to ladder trees, we introduce the geometric Volterra signature \(\geovsig(\bX)\) and establish the shuffle product of its signature components. To the best of our knowledge, the shuffle product for the Volterra signature is the first of its kind and allows us to formulate a universal approximation theorem for geometric Volterra signatures. For a special class of kernel functions, we prove the existence of an extended path \(\Bar{\bX}\) and relate its classical signature to the geometric Volterr
    
[^13]: DUST：方向不确定性感知与尺度不变的迁移学习

    DUST: Directional Uncertainty-aware and Scale-invariant Transfer Learning

    [https://arxiv.org/abs/2611.31948](https://arxiv.org/abs/2611.31948)

    提出了一种基于源回归参数尺度不变方向分布的概率迁移学习框架DUST，能够同时纳入源参数的不确定性并实现尺度不变且稳健的迁移学习。

    

    迁移学习旨在通过借鉴相关外部源数据集的信息来提高目标回归任务的效率。现有方法通过强制目标回归参数与源回归参数之间的欧氏距离接近来实现迁移，或者最近通过基于角度的惩罚来鼓励二者的方向对齐。虽然这类基于角度的惩罚减轻了迁移学习方法对源和目标回归系数之间尺度差异的敏感性，但它们仍然依赖于源回归参数的点估计，因此没有提供直接机制来纳入源信息中的不确定性。为此，我们提出了一个用于迁移学习的概率框架，该框架在源回归参数估计的尺度不变方向分布层面进行操作，而非依赖其点估计，从而实现稳健的、尺度不变的且不确定性感知的迁移学习。

    arXiv:2611.31948v1 Announce Type: cross  Abstract: Transfer learning seeks to improve efficiency in a target regression task by borrowing information from related external source dataset(s). Existing approaches achieve this by enforcing Euclidean proximity between the target and source regression parameters or, more recently, by encouraging their directional alignment through angle-based penalties. While such angle-based penalties mitigate the sensitivity of the transfer learning methods to the scale differences between source and target regression coefficients, they still rely on a point estimate of the source regression parameter and thus provide no direct mechanism to incorporate uncertainty in the source information. To that end, we propose a probabilistic framework for transfer learning that operates at the level of a \emph{scale-invariant directional distribution} of the source regression parameter estimate, rather than its point estimate, to enable robust, scale-invariant and un
    
[^14]: 元摊销认知建模

    Meta-Amortized Cognitive Modeling

    [https://arxiv.org/abs/2611.31914](https://arxiv.org/abs/2611.31914)

    该论文提出CogFormerV2元摊销框架，用单个网络一次性将贝叶斯推断摊销至整个认知模型空间，仅需在单GPU上训练约一天，即可对涵盖二元、多选项和连续决策的超过146亿个模型配置快速给出准确且校准良好的后验分布，从而免去每次模型修改后重新模拟与训练的成本。

    

    认知建模是一个高度迭代的过程。随着理论的发展与数据的积累，研究者需要反复修改模型结构、参数化方式、协变量以及实验设计。基于神经网络的摊销贝叶斯推断能够为单个模型提供快速估计，但每一次模型修改都需要重新进行模拟和训练。在此，我们提出了 CogFormerV2——一个元摊销框架，使研究者能够用单个网络来指定、修改和拟合认知模型，将摊销能力一次性覆盖整个模型空间。该网络仅需在单块GPU上训练约一天，即可在超过146亿个模型配置上给出准确且校准良好的后验分布，这些配置涵盖二元、多选项和连续决策任务，并具有不同的参数集合、协变量结构、试次数量和污染水平。它可以拟合来自不同行为任务的真实经验数据，并可通过……扩展到新模型。

    arXiv:2611.31914v1 Announce Type: new  Abstract: Cognitive modeling is highly iterative. Researchers repeatedly revise model structures, parameterizations, covariates, and experimental designs as theories evolve and data accumulates. Amortized Bayesian inference with neural networks provides rapid estimation for individual models, but every revision requires new simulations and training. Here, we introduce CogFormerV2, a meta-amortized framework that enables researchers to specify, revise, and fit cognitive models with a single network, carrying amortization across an entire space of models all at once. Trained once in about a day on a single GPU, it delivers accurate, well-calibrated posteriors across more than 14.6 billion model configurations spanning binary, multi-alternative, and continuous decisions, with varying parameter sets, covariate structures, trial counts, and contamination levels. It can fit empirical data from distinct behavioral tasks and extends to new models through 
    
[^15]: 最优臂识别中e-过程停止规则的通用框架

    A General Framework for E-process Stopping Rules in Best-Arm Identification

    [https://arxiv.org/abs/2611.31898](https://arxiv.org/abs/2611.31898)

    该论文提出了最优臂识别中e-过程停止规则的通用框架，通过e-过程分解将最优性条件分为三个部分并证明了渐近最优性的充分条件，进而推导出广义线性老虎机问题的渐近最优算法。

    

    我们在最优臂识别问题中提出了一种e-过程停止规则。e-过程停止规则与Chernoff停止规则的不同之处在于其对停止决策的假设结构。e-过程停止规则的主要优势在于停止规则设计的灵活性，事实上，它在多个老虎机问题上可以恢复先前提出的Chernoff停止规则。此外，我们通过e-过程的分解分析了δ-PAC算法的渐近最优性。该分解将最优性条件分离为三个部分：采样策略分配的收敛性、e-过程设计的效率以及老虎机问题的内在随机行为。随后，我们证明了老虎机算法渐近最优性的一个充分条件。作为e-过程停止规则的应用，我们推导了针对广义线性老虎机问题的渐近最优算法。

    arXiv:2611.31898v1 Announce Type: cross  Abstract: We propose an e-process stopping rule in the best-arm identification problem. The e-process stopping rule is different from the Chernoff stopping rule by its hypothesis structure on the stopping decision. The major benefit of an e-process stopping rule is the flexibility in stopping rule design and in fact, it recovers previously suggested Chernoff stopping rule on several bandit problems. Moreover, we analyze the asymptotic optimality of the $\delta$-PAC algorithm with the decomposition of e-processes. The decomposition separates optimality conditions into three parts; the sampling policy allocation convergence, the efficiency of e-process design, and the bandit problem's inherent stochastic behavior. Subsequently, we prove a sufficient condition for asymptotic optimality in bandit algorithms. For the application of an e-process stopping rule, we derived an asymptotically optimal algorithm for the generalized linear bandit problem.
    
[^16]: 通过集成学习与校准线性规划为巴西圣卡塔琳娜州的合成人口生成健康变量

    Generating health variables for a synthetic population of Santa Catarina, Brazil, via ensemble learning and a calibrated linear program

    [https://arxiv.org/abs/2611.31726](https://arxiv.org/abs/2611.31726)

    本文提出一种将逻辑回归、随机森林与支持向量机的集成学习与校准线性规划相结合的方法，为巴西圣卡塔琳娜州的合成人口生成吸烟、饮酒、锻炼、自评健康和慢性病等健康变量，并证明在温和条件下可省去惩罚项从而精确再现调查比例，在PNS数据交叉验证中取得最佳效果。

    

    流行病学中的基于主体的模型需要带有健康相关属性的合成人口，而在巴西，这些属性只能通过诸如全国健康调查（Pesquisa Nacional de Saúde, PNS）之类的抽样调查获得。我们为圣卡塔琳娜州的地理参考合成人口添加了健康变量，包括吸烟、饮酒、体育锻炼、自评健康和慢性病。类别概率由逻辑回归、随机森林和支持向量机的集成方法进行估计，个体则由一个能够再现从调查中估计的类别比例的线性规划分配到相应类别。我们证明，在集成方法满足的一个温和条件下，该规划的惩罚项不再需要，且比例可以被精确再现。在基于PNS数据的交叉验证中，该方法取得了最佳结果。

    arXiv:2611.31726v1 Announce Type: cross  Abstract: Agent-based models in epidemiology require synthetic populations with health-related attributes, which in Brazil are available only from sample surveys such as the Pesquisa Nacional de Sa\'ude (PNS). We add health variables, including smoking, alcohol consumption, physical exercise, self-rated health and chronic conditions, to a georeferenced synthetic population of the state of Santa Catarina. Class probabilities are estimated by an ensemble of logistic regression, random forest and support vector machine, and individuals are assigned to classes by a linear program that reproduces the class proportions estimated from the survey. We show that, under a mild condition satisfied by our ensemble, the penalty term of this program is no longer needed, and the proportions are reproduced exactly. In a cross-validation on PNS data, the method attains the best results.
    
[^17]: 无导数约束随机优化的在线统计推断

    Online Statistical Inference for Derivative-Free Constrained Stochastic Optimization

    [https://arxiv.org/abs/2611.31687](https://arxiv.org/abs/2611.31687)

    本文提出一种基于随机投影的免导数随机序列二次规划方法（SDF-SSQP），在无导数信息的等式约束随机优化问题中，将每轮迭代成本降至 $O((d+m)^2)$ 的同时实现有效的在线统计推断。

    

    我们研究在导数信息不可用或获取代价高昂的情形下，非线性等式约束随机优化问题的统计推断。尽管随机序列二次规划（SSQP）的最新进展已实现了高效的在线优化，但现有方法要么依赖梯度信息，要么计算成本过高。为解决这些局限，我们开发了一种基于随机投影的免导数SSQP（SDF-SSQP）方法，可同时实现计算效率和有效的统计推断。通过利用随机投影技术，所提方法将每次迭代的计算成本降低至 $O((d+m)^2)$，其中 $d$ 为决策变量的维数，$m$ 为约束的数量，使其非常适合大规模问题。为克服随机投影和零阶近似给统计推断带来的挑战，我们采用……（原文摘要在此处截断）

    arXiv:2611.31687v1 Announce Type: cross  Abstract: We study statistical inference for nonlinear equality-constrained stochastic optimization problems in settings where derivative information is unavailable or expensive to obtain. While recent advances in stochastic sequential quadratic programming (SSQP) have enabled efficient online optimization, existing methods either rely on gradient information or suffer from high computational costs. To address these limitations, we develop a sketched derivative-free SSQP (SDF-SSQP) method that simultaneously achieves computational efficiency and valid statistical inference. By leveraging randomized sketching, the proposed method reduces the per-iteration computational cost to $O((d+m)^2)$, where $d$ is the dimension of the decision variable and $m$ is the number of constraints, making it well suited for large-scale problems. To overcome the challenges that both sketching and zeroth-order approximations introduce for statistical inference, we ado
    
[^18]: 线性算子方程的核框架

    A Kernel Framework for Linear Operator Equations

    [https://arxiv.org/abs/2611.31626](https://arxiv.org/abs/2611.31626)

    该论文提出了一个统一的再生核希尔伯特空间框架，将带非局部项或异质约束的线性算子方程的求解归结为单个线性系统，并借助Nyström近似降低大规模配置点问题的计算成本。

    

    求解具有非局部项或异质约束的线性算子方程通常需要特定的数值格式。我们开发了一种基于将此类方程嵌入再生核希尔伯特空间（RKHS）的统一方法。微分算子、边界算子和积分算子在RKHS中由其Riesz表示元表示。方程在有限多个配置点上以正则化最小二乘意义求解。所得解是算子诱导表示元的有限线性组合，并通过求解单个线性系统获得。其敏感性（如金融中的希腊字母）可通过直接微分得到。为降低大规模配置点集的计算成本，我们引入了一种基于对算子诱导表示元进行子采样的Nyström近似。在正则化经验范数意义下，该方法在缩减的试探空间内给出该解的最佳逼近。数值算例涵盖常（微分方程等）

    arXiv:2611.31626v1 Announce Type: cross  Abstract: Solving linear operator equations with nonlocal terms or heterogeneous constraints often requires specific numerical schemes. We develop a unified approach based on embedding such equations into a reproducing kernel Hilbert space (RKHS). Differential, boundary, and integral operators are represented by their Riesz representers in the RKHS. The equations are solved in the regularized least-squares sense at finitely many collocation points. The solution is a finite linear combination of operator-induced representers and is obtained from a single linear system. Its sensitivities, such as financial Greeks, follow by direct differentiation. To reduce the computational cost for large collocation sets, we introduce a Nystr\"om approximation based on subsampling the operator-induced representers. It yields the best approximation of this solution within the reduced trial space, in a regularized empirical norm. Numerical examples cover ordinary,
    
[^19]: 概率数字孪生中的贝叶斯数据集成与模型更新

    Bayesian Data Integration and Model Updating in Probabilistic Digital Twins

    [https://arxiv.org/abs/2611.31620](https://arxiv.org/abs/2611.31620)

    本文提出了一种用于概率数字孪生的序贯贝叶斯更新框架，通过区分属性数据与行为数据并引入基于主动学习的代理模型策略，实现了多源数据的高效融合与计算成本的显著降低。

    

    本文提出了一个在概率数字孪生中进行序贯贝叶斯更新的框架，该框架利用随时间从多个来源收集的数据。该框架根据观测数据与数字状态的关系，区分了属性数据和行为数据。属性数据直接为不确定模型参数的子集提供信息，因此可以使用高效的专用贝叶斯更新方法进行整合。行为数据通过预测模型提供间接信息，需要求解贝叶斯逆问题。为了降低相关的计算成本，本文引入了一种基于主动学习的代理模型策略，该策略随数字状态演化，并在与后验推断相关的区域自适应地细化预测模型。该框架通过一个理想化的悬臂梁算例以及一个承受变化水位及相关荷载的岩土工程运河墙壁案例研究进行了评估。

    arXiv:2611.31620v1 Announce Type: cross  Abstract: This paper proposes a framework for sequential Bayesian updating in probabilistic digital twins using data from multiple sources collected over time. The framework distinguishes between property data and behavior data, according to how observations relate to the digital state. Property data directly inform a subset of the uncertain model parameters and can therefore be incorporated using efficient dedicated Bayesian updating methods. Behavior data provide indirect information through the prediction model and require the solution of a Bayesian inverse problem. To reduce the associated computational cost, an active-learning-based surrogate strategy is introduced that evolves with the digital state and adaptively refines the prediction model in regions relevant to posterior inference. The framework is evaluated using an idealized cantilever beam example and a geotechnical case study of a canal wall subject to varying water levels and corr
    
[^20]: 一维实例依赖的赌博机凸优化

    Instance-Dependent Bandit Convex Optimization in One Dimension

    [https://arxiv.org/abs/2611.31608](https://arxiv.org/abs/2611.31608)

    本文在一维赌博机凸优化中首次刻画了实例依赖样本复杂度，引入新量 Δ_ε(f) 证明定位极小值点问题可获得依赖该量的加速，同时惊人地发现简单遗憾设定下不存在任何实例依赖的改进。

    

    我们研究了一维随机赌博机凸优化在固定置信度设定下的实例依赖样本复杂度。首先，我们考虑极小值点定位问题，即学习者试图识别一个与唯一极小值点距离不超过 ε 的点。我们引入了一个新的实例依赖量 Δ_ε(f)，它对应于 f−min f 的下水平集长度为 2ε 的阈值。我们证明了量级为 Δ_ε(f)^{-2} 的下界，并给出一种二分类式的算法，该算法在对数因子范围内匹配这一速率。随后，我们重新审视简单遗憾设定，其目标是识别一个 ε-最优点。令人惊讶的是，在此设定下不可能获得可比的实例依赖改进：每个正确的算法在每个实例上的样本复杂度均为 ε^{-2} 量级，在常数因子内匹配极小极大速率。这些结果共同区分了这两种设定。

    arXiv:2611.31608v1 Announce Type: cross  Abstract: We study the instance-dependent sample complexity of one-dimensional stochastic bandit convex optimization in the fixed-confidence setting. First, we consider minimizer localization, where the learner seeks to identify a point within distance $\varepsilon$ of the unique minimizer. We introduce a new instance-dependent quantity $\Delta_\varepsilon(f)$, corresponding to the threshold for which the sublevel set of $f-\min f$ has length $2\varepsilon$. We prove a lower bound of order $\Delta_\varepsilon(f)^{-2}$ and give a bisection-style algorithm that matches this rate up to logarithmic factors. We then revisit the simple-regret setting, where the goal is to identify an $\varepsilon$-optimal point. Surprisingly, no comparable instance-dependent improvement is possible: every correct algorithm has sample complexity of order $\varepsilon^{-2}$ on every instance, matching the minimax rate up to constants. Together, these results separate th
    
[^21]: ExitLUT：面向高效FPGA神经网络推理的布尔早期退出机制

    ExitLUT: Boolean Early Exits for Efficient FPGA-Based Neural Network Inference

    [https://arxiv.org/abs/2611.31540](https://arxiv.org/abs/2611.31540)

    ExitLUT用直接映射到FPGA LUT的可学习布尔函数替代softmax置信度退出决策，将节省的退出路径资源重新投入主数据通路，在保持精度不变的前提下实现高达4.08倍的吞吐量提升。

    

    早期退出神经网络通过在达到足够置信度时终止推理来降低平均推理成本。在FPGA上，传统的基于置信度的退出机制会引入与LUT结构不匹配的算术控制路径，消耗了本可用于主数据通路的资源。我们提出ExitLUT，用直接映射到FPGA LUT中的可学习布尔函数来替代softmax风格的退出决策。在CNN评估中，fpgaConvNet是静态空间数据流基线，而ATHEENA是其基于softmax的早期退出扩展。ExitLUT将ATHEENA的退出决策LUT使用量相比softmax路径降低2.58倍，并消除了决策路径的DSP使用，同时保持精度和退出行为不变。重新运行设计空间探索可将这些节省的资源再投入到数据通路中，吞吐量相比fpgaConvNet提升高达4.08倍，相比ATHEENA提升1.82倍。对于基于LUT的神经网络，嵌入式ExitLUT退出实现了1.41–1.89倍的性能提升。

    arXiv:2611.31540v1 Announce Type: cross  Abstract: Early exit neural networks reduce average inference cost by terminating once sufficient confidence is reached. On FPGAs, conventional confidence-based exits add arithmetic control paths poorly matched to LUT fabrics, consuming resources that could otherwise support the main datapath. We present ExitLUT, which replaces softmax-style exit decisions with learned Boolean functions mapped directly into FPGA LUTs. In our CNN evaluation, fpgaConvNet is the static spatial-dataflow baseline, while ATHEENA is its softmax-based early exit extension. ExitLUT reduces ATHEENA's exit-decision LUT usage by 2.58$\times$ over the softmax path and eliminates decision-path DSP usage, while preserving accuracy and exit behavior. Rerunning design-space exploration reinvests these savings into the datapath, improving throughput by up to 4.08$\times$ over fpgaConvNet and 1.82$\times$ over ATHEENA. For LUT-based NNs, embedded ExitLUT exits achieve 1.41-1.89$\t
    
[^22]: 诊断神经算子中的认知不确定性：初始化、离散化与函数采样的影响

    Diagnosing Epistemic Uncertainty in Neural Operators: Effects of Initialization, Discretization, and Function Sampling

    [https://arxiv.org/abs/2611.31526](https://arxiv.org/abs/2611.31526)

    该论文系统研究了神经算子中认知不确定性的三大来源（训练函数采样、输出离散化、模型初始化），通过训练约46,000个模型并构建集成，定量刻画了各来源对算子学习性能差异的独立贡献。

    

    神经算子为函数空间之间的映射提供了高效的代理模型，但其可靠性无法通过单一训练模型的性能来刻画。现有的神经算子不确定性量化方法通常只刻画预测不确定性，而未能分别识别认知不确定性各来源的独立贡献。即使底层物理问题是确定性的，训练函数空间的有限采样、输出离散化和模型初始化也会在所学到的算子之间产生显著差异。我们在多样化的算子学习架构集合上系统地研究了这些认知不确定性来源。在实验中，我们训练了约46,000个神经算子模型，并在训练数据集、模型初始化和输出离散化上构建集成，以刻画相对L2误差的分布特征。

    arXiv:2611.31526v1 Announce Type: new  Abstract: Neural operators provide efficient surrogates for mappings between function spaces, but their reliability cannot be characterized by the performance of a single trained model. Existing neural operator uncertainty quantification approaches often characterize predictive uncertainty without separately identifying the contributions of individual sources of epistemic uncertainty. Finite sampling of the training function space, output discretization, and model initialization can produce substantial variation among learned operators even when the underlying physical problem is deterministic. We systematically investigate these sources of epistemic uncertainty across a diverse set of operator learning architectures. Across our experiments, we train approximately 46,000 neural operator models and construct ensembles over training datasets, model initializations, and output discretizations to characterize distributions of relative $L_2$ error rath
    
[^23]: SOTS：基于签名的最优传输选择

    SOTS: Signature-based Optimal Transport Selection

    [https://arxiv.org/abs/2611.31525](https://arxiv.org/abs/2611.31525)

    SOTS提出了一种无监督核心集选择框架，通过将不规则多变量时间序列映射为截断路径签名并贪心最小化子集与全量数据之间的切片Wasserstein距离来选择训练数据，同时提供了随机核心集的收敛性理论保证。

    

    在工业应用中，由于专家标注成本高且耗时，机器学习模型的训练数据选择成为一个关键瓶颈。现有的时间序列核心集选择方法要么完全忽略时间结构，要么无法处理变长、不规则采样、多通道信号——而这正是工业监测与测试中最常见的场景。我们提出了SOTS（Signature-based Optimal Transport Selection，基于签名的最优传输选择），一个面向不规则多变量时间序列（IMTS）分类的无监督核心集选择框架。每个样本被映射到一个截断路径签名，以捕捉其各通道的联合几何结构；随后通过贪心地最小化候选子集与完整数据集的经验签名分布之间的切片Wasserstein距离来选择核心集，我们为该准则建立了随机核心集的收敛性保证。我们在一个re……

    arXiv:2611.31525v1 Announce Type: new  Abstract: Training data selection for machine learning models is a critical bottleneck in industrial applications where expert annotation is costly and time-consuming. Existing coreset selection methods for time series either ignore temporal structure entirely or cannot handle variable-length, irregularly sampled, multi-channel signals, the setting most common in industrial monitoring and testing. We introduce SOTS (Signature-based Optimal Transport Selection), an unsupervised coreset selection framework for irregular multivariate time series (IMTS) classification. Each sample is mapped to a truncated path signature, capturing the joint geometric structure of its channels, and the coreset is selected by greedily minimizing the Sliced Wasserstein distance between the empirical signature distributions of the candidate subset and the full dataset, a criterion for which we establish a convergence guarantee for random coresets. We validate SOTS on a re
    
[^24]: 带签名的Transformer：迈向路径空间中的通用上下文学习

    Transformers with Signatures: Towards Universal In-Context Learning in Path Space

    [https://arxiv.org/abs/2611.31439](https://arxiv.org/abs/2611.31439)

    该论文利用路径签名特征建立了面向路径空间的上下文学习理论框架，显式构造了可模拟上下文Lasso的每层双注意力头因果softmax Transformer，并给出超越IID数据设定的有限样本预测误差界。

    

    上下文学习（ICL）使Transformer能够通过演示样例适应新任务，而无需更新其参数。然而，针对Transformer的理论保证在一般情况下仍然有限，尤其是当目标响应非线性地依赖于观测路径的演化时，更是难以获得。我们利用路径签名特征，建立了一个面向路径空间的上下文学习框架。基于签名的通用线性逼近性质，我们显式构造了一个每层具有两个注意力头的因果softmax Transformer，用以模拟上下文Lasso。在稀疏签名近似以及适当的矩和集中性条件下，我们建立了有限样本预测误差界。我们的分析超越了现有上下文学习理论中的IID数据设定，允许演示样例与查询之间存在依赖关系。在上下文稀疏恢复、非线性路径回归、执行成本预测和金融（数据）上的实验……

    arXiv:2611.31439v1 Announce Type: new  Abstract: In-context learning (ICL) enables transformers to adapt to new tasks from demonstrations without updating their parameters. However, theoretical guarantees for transformers remain limited in general, and elusive in particular when target responses depend nonlinearly on the evolution of observed paths. We establish an ICL framework for path space, using path signature features. Building on the universal linear approximation property of signatures, we construct explicitly a causal softmax transformer with two attention heads per layer that emulates in-context Lasso. Under sparse signature approximation and suitable moment and concentration conditions, we establish finite-sample prediction error bounds. Our analysis goes beyond the IID data setting in the existing ICL theory, allowing dependence among demonstrations and the query.   Experiments on in-context sparse recovery, nonlinear path regression, execution-cost prediction, and financia
    
[^25]: 用于光谱预测的局部子空间编码：从TabPFN到算子自适应定标

    Local subspace encoding for spectral prediction: from TabPFN to operator-adaptive calibration

    [https://arxiv.org/abs/2611.31245](https://arxiv.org/abs/2611.31245)

    该研究证实局部振动子空间编码（LVSE）表示方法在nirs4all生态系统的光谱数据集上能显著降低TabPFN的RMSE并提升分类加权F1，且其价值可扩展至经典定标方法。

    

    TabPFN已经在近红外光谱上展现出了有用的预测性能。随后，RamanPFN提出了两种用于表格推理的光谱表示方法：全局成分解混和局部振动子空间编码。我们评估了这些表示方法是否也能在使用nirs4all生态系统构建的数据集上发挥作用，以及它们的价值是否能延伸到经典定标方法。我们重新分析了来自公共光谱数据集目录的已完成实验。在61个匹配的数值预测任务的事后子集上（剔除了十个类别代码、元数据或语义模糊的目标变量后），LVSE相对于原始光谱TabPFN，在检查点2.5、3和3.5处平均分别将RMSE降低了10.22%、12.80%和14.52%，中位数降幅分别为4.99%、4.74%和7.82%。在76个分类数据集上，加权F1分别提高了2.22、2.56和3.81个百分点。（注：原摘要似乎在末尾被截断）

    arXiv:2611.31245v1 Announce Type: new  Abstract: TabPFN has already shown useful predictive performance on near-infrared spectra. RamanPFN subsequently proposed two spectral representations for tabular inference: Global Compositional Unmixing (GCU) and Local Vibrational Subspace Encoding (LVSE). We assess whether these representations also help on datasets assembled through the nirs4all ecosystem, and whether their value extends to classical calibration. We reanalyse completed experiments from a public spectral dataset catalogue. On a post-hoc subset of 61 matched numerical prediction tasks, after excluding ten class-code, metadata or semantically ambiguous targets, LVSE reduces RMSE relative to raw-spectrum TabPFN by 10.22%, 12.80% and 14.52% on average for checkpoints 2.5, 3 and 3.5, respectively, with median reductions of 4.99%, 4.74% and 7.82%. Across 76 classification datasets, weighted F1 increases by 2.22, 2.56 and 3.81 percentage points. The 2.5 classification procedure uses ou
    
[^26]: $K$-中位秩：一种针对重尾数据的鲁棒聚类算法及其在光谱学和金融中的应用

    $K$-Midranks: A robust clustering algorithm for heavy-tailed data with applications to spectroscopy and finance

    [https://arxiv.org/abs/2611.30953](https://arxiv.org/abs/2611.30953)

    本文提出了一种名为 $K$-中位秩的鲁棒聚类算法，通过将观测值替换为逐坐标秩来降低重尾数据中极端值的影响，在柯西型重尾数据上优于传统聚类方法。

    

    对于依赖原始观测值之间距离或特定分布假设的方法而言，对包含极端观测值和重尾分布的数据进行聚类可能具有挑战性。我们提出了 $K$-中位秩（$K$-midranks），这是一种聚类方法，它将观测值替换为逐坐标的秩，并使用类似于 $K$-均值的程序对变换后的数据进行划分。这种变换降低了极端值的影响，同时保留了每个变量内观测值的排序信息。我们通过模拟研究将 $K$-中位秩与 $K$-均值、$K$-中心点以及基于高斯分布、学生t分布、柯西分布和广义双曲分布的混合模型进行了比较，并提供了一些理论性质。$K$-中位秩在柯西平移数据上通常优于竞争方法。相比之下，$K$-均值和基于模型的方法在高斯数据以及三自由度的学生t数据上表现更好。我们展示了该方法在光谱学和金融领域的应用。

    arXiv:2611.30953v1 Announce Type: cross  Abstract: Clustering data with extreme observations and heavy tails can be challenging for methods that rely on distances between raw observations or specific distributional assumptions. We introduce $K$-midranks, a clustering method that replaces observations with coordinatewise ranks and partitions the transformed data using a procedure analogous to $K$-means. This transformation reduces the influence of extreme values while preserving the ordering of observations within each variable. We evaluate $K$-midranks against $K$-means, $K$-medoids, and mixture models based on Gaussian, Student's t, Cauchy, and generalized hyperbolic distributions through simulation studies, and provide some theoretical properties. $K$-midranks generally outperforms the competing methods on Cauchy-shifted data. In contrast, $K$-means and model-based methods perform better on Gaussian data and Student's t data with three degrees of freedom. We demonstrate the method's 
    
[^27]: 稀疏霍奇偏好优化：微调语言模型以实现对超越传递性偏好的对齐

    Sparse Hodge Preference Optimization: Fine-Tuning Language Models toward Alignment beyond Transitive Preferences

    [https://arxiv.org/abs/2611.30135](https://arxiv.org/abs/2611.30135)

    本文提出稀疏霍奇偏好优化（SHPO），借助组合霍奇分解理论为DPO引入稀疏反对称旋度残差，使其能够捕捉人类偏好数据中的非传递性与循环结构，并证明其严格扩大了DPO的假设类。

    

    直接偏好优化的广泛应用，反映了其在利用成对偏好数据微调大型语言模型以符合人类期望方面的巨大成功。它直接从人类偏好中优化策略，无需显式的奖励建模，也避免了基于强化学习方法所带来的复杂性。然而，经典的DPO依赖于Bradley-Terry采样模型，无法有效捕捉聚合的人类偏好数据中的非传递性或循环结构。在本文中，我们提出了稀疏霍奇偏好优化，这是一种偏好优化算法，通过受组合霍奇分解启发的稀疏反对称旋度残差来增强DPO。我们证明了在响应表示满足一些温和正则性条件的情形下，SHPO在包含至少一个三角形的比较图上严格扩大了DPO的假设类。我们进一步……（原文摘要在此处截断）

    arXiv:2611.30135v1 Announce Type: cross  Abstract: The growing adoption of Direct Preference Optimization (DPO), which leverages pairwise preference data, reflects its great success in fine-tuning large language models to align with human expectations. It directly optimizes the policy from human preferences without explicit reward modeling or the complexities of reinforcement learning-based approaches. However, the classic DPO relies on the Bradley--Terry sampling model and cannot effectively capture intransitivity or cyclic structures in aggregated human preference data. In this article, we propose Sparse Hodge Preference Optimization (SHPO), a preference optimization algorithm that augments DPO with a sparse antisymmetric curl residual motivated by the combinatorial Hodge decomposition. We prove that SHPO strictly enlarges DPO's hypothesis class on comparison graphs containing at least one triangle, under some mild regularity conditions on the response representations. We further est
    
[^28]: DiT如何去噪？从聚合器-预测器视角理解注意力与MLP

    How Do DiTs Denoise? An Aggregator-Predictor View of Attention and MLPs

    [https://arxiv.org/abs/2610.29854](https://arxiv.org/abs/2610.29854)

    论文提出将扩散Transformer视为“聚合器-预测器”架构的新解释框架——AdaLN调制的注意力充当依赖输入与噪声的上下文聚合器，MLP充当可学习键-值表示的patch预测器，并从理论上证明基于局部上下文的聚合能以指数衰减的误差逼近全局贝叶斯去噪器。

    

    扩散Transformer（DiT）如何表示图像内容，并学习复用这些内容的规则以实现泛化？我们提出了一种“聚合器-预测器”视角，将DiT视为输入自适应的局部去噪器，其中上下文聚合与patch预测为理解注意力和MLP提供了功能性角色。我们首先证明，对于高斯马尔可夫随机场（GMRF）先验，全局贝叶斯去噪器可以仅使用上下文patch的一个子集来近似，且误差随上下文半径呈指数衰减。我们进一步表明，常被用于解释扩散模型泛化能力的逐patch线性去噪器，在聚合信息足够充分时能够恢复经验局部去噪器。在这一视角下，经AdaLN调制的注意力自然地参数化了一个依赖输入和噪声的聚合器，而MLP则提供了一个可学习的键-值表示，用于逐token的patch预测。我们的实验支持这一观点。在真实数据上训练的DiT……

    arXiv:2610.29854v1 Announce Type: cross  Abstract: How do diffusion transformers (DiTs) represent image content and learn rules for reusing it to generalize? We develop an aggregator--predictor view of DiTs as input-adaptive local denoisers, in which context aggregation and patch prediction provide functional roles for interpreting attention and MLPs. We first establish that, for Gaussian Markov random field (GMRF) priors, the global Bayes denoiser can be approximated using only a subset of context patches, with error decaying exponentially in context radius. We further show that patchwise linear denoisers, often invoked to explain diffusion generalization, recover the empirical local denoiser with sufficiently informative aggregation. Within this view, AdaLN-modulated attention naturally parameterizes an input- and noise-dependent aggregator, while the MLP admits a learned key--value representation for tokenwise patch prediction. Our experiments support this view. DiTs trained on real
    
[^29]: 预先公告的变点使保形可靠性与频率校准相分离

    Announced Breaks Separate Conformal Reliability from Frequency Calibration

    [https://arxiv.org/abs/2610.23917](https://arxiv.org/abs/2610.23917)

    本文证明预先公告的结构性变点使“下一区间的保形可靠性”与“封闭窗口的频率校准”这两种重置审计目标可以分离，并提出在公告时点重启经验分位数与自适应递归的方法，前者达到1/√n的minimax可靠性速率、后者达到1/n的频率差距，且实验中覆盖率误差显著优于自适应基线（0.035对0.052）。

    

    许多结构性变点会提前公告，然而自适应保形方法只能从自身的预测误差中获知它们。在公告日期进行重置只能服务于两种审计之一：下一区间的保形可靠性，或一个封闭窗口的频率校准，而这两者可能不一致。对于独立得分，在变点边界处重启经验分位数可以达到极小化极大（minimax）可靠性速率，其阶为断裂后样本数的平方根分之一（1/√n）。在得分有界且递归下限累计增量有界的条件下，重启自适应递归可以达到阶为断裂后样本数分之一（1/n）的频率差距，但该差距对下一区间没有任何约束作用。在一次较大的合成变点之后，重启分位数在第50次预测时的覆盖率误差为0.035，而五个自适应基线中最好者为0.052。在每次合成变点之后的前50次预测中，重启递归都具有最小的频率差距。真实变点大多是……（原文在此截断）

    arXiv:2610.23917v2 Announce Type: replace-cross  Abstract: Many structural breaks are announced in advance, yet adaptive conformal methods learn of them only from their own errors. A reset at the announced date serves one of two audits, conformal reliability of the next interval or frequency calibration of a closed window, and the two can disagree. For independent scores, restarting the empirical quantile at the boundary attains the minimax reliability rate, of order one over the square root of the post-break count. Restarting an adaptive recursion attains a frequency gap of order one over that count on bounded scores when its floor adds a bounded total. That gap bounds nothing about the next interval. After a large synthetic break, the restarted quantile's coverage error at the 50th forecast is 0.035, against 0.052 for the best of five adaptive baselines. The restarted recursion has the smallest frequency gap over the first 50 forecasts after every synthetic break. Real breaks are mos
    
[^30]: BridgeTwist：扭转薛定谔桥以实现免训练条件采样

    BridgeTwist: Twisting Schr\"odinger Bridges for Training-Free Conditional Sampling

    [https://arxiv.org/abs/2610.18658](https://arxiv.org/abs/2610.18658)

    提出了BridgeTwist，一种基于带温度调节扭转序贯蒙特卡洛方案的免训练条件采样器，可复用预训练薛定谔桥生成模型，利用由速度场与得分场构造的闭式即插即用扭转代理实现条件后验采样。

    

    生成模型通常需要服务于多种下游条件任务。在免训练的设置下，预训练的无条件模型被复用为先验，并与任务特定的似然函数相结合，从而从相应的后验分布中采样。尽管针对扩散模型和流模型已开发出此类免训练采样器，但作为在一般源分布与目标分布之间进行灵活随机输运的模型，薛定谔桥（SB）正日益受到关注，而其对应的免训练采样器却尚未被探索。我们提出了BridgeTwist，一种面向预训练薛定谔桥生成模型的免训练条件采样器，其被表述为一种带温度调节的扭转序贯蒙特卡洛方案。其中的扭转量是一个闭式的即插即用代理，通过将SB预训练所暴露的速度场与得分场组合为一个终端预测器，并在该终端处评估似然函数而得到。由于该代理在靠近源分布处可靠性较低，我们对其进行了温度调节……

    arXiv:2610.18658v1 Announce Type: cross  Abstract: Generative models are often required to serve many downstream conditional tasks. In the training-free setting, a pretrained unconditional model is reused as a prior and combined with a task-specific likelihood to sample from the corresponding posterior. While such training-free samplers have been developed for diffusion and flow models, their counterpart for Schr\"odinger bridge (SB) models remains unexplored, despite the growing appeal of SBs as flexible stochastic transport models between general source and target distributions. We propose BridgeTwist, a training-free conditional sampler for pretrained SB generative models, formulated as a tempered twisted sequential Monte Carlo scheme. Its twist is a closed-form plug-in surrogate, obtained by combining the velocity and score fields exposed by SB pretraining into an endpoint predictor and evaluating the likelihood there. Since this surrogate is less reliable near the source, we tempe
    
[^31]: 超越点预测：时间序列与时空数据概率预测综述

    Beyond Point Forecasts: A Survey on Probabilistic Forecasting for Time Series and Spatiotemporal Data

    [https://arxiv.org/abs/2609.13345](https://arxiv.org/abs/2609.13345)

    这篇综述通过按不确定性在预测流程中引入的位置和方式建立统一分类框架，整合了时间序列与时空概率预测中分散的统计建模、机器学习和深度生成建模方法，并涵盖了集成校准、贝叶斯建模、分布回归以及新兴的时间序列基础模型等范式。

    

    概率预测是不确定性下决策的核心，但其方法论版图在时间序列预测与时空预测、统计建模、机器学习以及深度生成建模等领域之间日益碎片化。本综述通过根据不确定性在预测流程中引入的位置和方式来组织概率预测方法，构建了一个统一的视角。我们的分类体系将模型无关方法（包括集成方法和无分布校准）与模型内在方法（涵盖贝叶斯建模、参数化预测分布、分布回归以及现代生成模型）联系起来，并进一步审视了时间序列基础模型的新兴角色。除方法论的综合梳理之外，我们还识别了不同范式所依赖的假设、计算需求以及所刻画的不确定性形式，并将这些差异转化为……（摘要内容在此处截断）

    arXiv:2609.13345v1 Announce Type: cross  Abstract: Probabilistic forecasting is central to decision-making under uncertainty, yet its methodological landscape has become increasingly fragmented across temporal and spatiotemporal forecasting, statistical modeling, machine learning, and deep generative modeling. This survey develops a unified perspective by organizing probabilistic forecasting methods according to where and how uncertainty is introduced into the forecasting pipeline. Our taxonomy connects model-agnostic approaches including ensembles and distribution-free calibration, with model-intrinsic approaches spanning Bayesian modeling, parametric predictive distributions, distributional regression, and modern generative models, and further examines the emerging role of time series foundation models. Beyond methodological synthesis, we identify the assumptions, computational demands, and forms of uncertainty represented by different paradigms, and translate these distinctions into
    
[^32]: qshap：梯度提升树的$R^2$快速Shapley分解

    qshap: Fast Shapley Decomposition of $R^2$ for Gradient-Boosted Trees

    [https://arxiv.org/abs/2608.24104](https://arxiv.org/abs/2608.24104)

    qshap是一种新工具，通过Shapley分解梯度提升树中的$R^2$值，实现特征和观测对模型性能贡献的全局量化。

    

    arXiv:2608.24104v1 公告类型：交叉 摘要：已有大量方法用于量化树集成模型中单个预测的特征归因。然而，许多应用需要特征对整体模型性能贡献的全局度量。尽管局部归因分数可以聚合以表征特征重要性，但此类汇总并不能直接分解预测性能的度量，如$R^2$。本文介绍了qshap，该工具同时提供R和Python版本，它实现了对梯度提升决策树（GBDTs）中$R^2$值的Shapley分解，以量化特征对模型性能的特定贡献。通过分解单个观测的二次损失，qshap提供了灵活的工具来探索单个特征和观测的重要性。qshap目前支持广泛使用的GBDT实现，包括xgboost、lightgbm和catboost，通过统一的树表示和高效的C++后端实现。

    arXiv:2608.24104v1 Announce Type: cross  Abstract: Numerous methods have been developed to quantify feature attributions in individual predictions for tree ensembles. However, many applications require global measures of feature contributions to overall model performance. Although local attribution scores can be aggregated to characterize feature importance, such summaries do not directly decompose measures of predictive performance, such as $R^2$. This article introduces qshap, available in both R and Python, which provides Shapley decomposition of $R^2$ values for gradient-boosted decision trees (GBDTs) to quantify feature-specific contributions to model performance. By decomposing the quadratic loss of individual observations, qshap provides flexible tools to explore the importance of individual features and observations. qshap currently supports widely used GBDT implementations, including xgboost, lightgbm, and catboost, through a unified tree representation and efficient C++ backe
    
[^33]: 基于静态快照的空间随机动力学物理信息推断的可辨识性极限

    Identifiability Limits of Physics-Informed Inference for Spatial Stochastic Dynamics from Static Snapshots

    [https://arxiv.org/abs/2607.01749](https://arxiv.org/abs/2607.01749)

    本研究通过结构性可辨识性分析揭示了利用物理信息机器学习从分子静态空间快照推断空间随机动力学的根本极限：分布式源本质上不可辨识，而点源（如转录位点）可恢复可辨识性，且这些极限还受边界条件、空间正则性等建模选择的制约。

    

    尽管测量规模和分辨率不断提升，许多生物学测量仅能提供分子位置的静态快照，而非产生这些快照的动力学过程。通过将灵活的表示方法与机制约束相结合，物理信息机器学习为从静态快照推断这些动力学过程提供了一条有前景的途径。受基因表达亚细胞成像的启发，我们探讨了分子的静态空间模式何时能够辨识空间变化的扩散率、生成、销毁以及边界交换过程，并研究了不同的推断方案在该任务上的表现。结构性可辨识性分析表明，分布式源是不可辨识的，而诸如转录位点这样的点源则可以恢复可辨识性。这些极限还进一步受到看似无关紧要的建模选择的影响：边界条件、底层动力学的空间正则性，甚至随机性……

    arXiv:2607.01749v2 Announce Type: replace-cross  Abstract: Despite increasing scale and resolution, many biological measurements provide only a static snapshot of molecular positions rather than the dynamics that produced them. By combining flexible representations with mechanistic constraints, physics-informed machine learning offers a promising route to inferring these dynamics from static snapshots. Motivated by subcellular imaging of gene expression, we ask when a static spatial pattern of molecules can identify spatially varying diffusivity, creation, destruction, and boundary exchange, and how different inference schemes perform on the task. A structural identifiability analysis shows that distributed sources are non-identifiable, whereas a point source such as a transcription site can restore identifiability. These limits are further shaped by seemingly innocuous modeling choices: the boundary conditions, the spatial regularity of the underlying dynamics, and even the stochastic
    
[^34]: Fokker–Planck方程的统一热球框架：均值恒等式、精确采样与误差分析

    A Unified Heat-Ball Framework for Fokker--Planck Equations: Mean-Value Identities, Exact Sampling, and Error Analysis

    [https://arxiv.org/abs/2606.27954](https://arxiv.org/abs/2606.27954)

    本文提出了一个统一的非齐次热球框架，为具有一般非线性、时变漂移的 Fokker–Planck 方程推导出密度、对数密度、熵密度与得分函数的局部均值恒等式，从而为扩散生成模型中漂移结构的分析、局部化采样以及低密度区域的处理提供了新工具。

    

    具有非线性或学习得到的前向漂移的扩散生成模型提出了一些在高斯前向过程中被掩盖的问题：漂移的无散度部分与梯度部分如何塑造演化中的密度，采样能否利用局部而非全局信息，以及如何处理低密度区域——在这些区域中 $u$ 携带微弱信号，而 $\log u$ 发散，熵密度 $u\log u$ 则介于两者之间。这些问题促使本研究探讨具有一般性、可能为非线性且时变的漂移的 Fokker–Planck 方程的局部表示。我们推导了密度 $u$、对数密度 $\log u$、熵密度 $u\log u$ 以及得分函数 $\nabla\log u$ 的非齐次热球恒等式，并保留了由漂移产生的源项；其未缩放的残差在热球平方半径的最低阶意义上恢复出相应的微分残差。（摘要在此处被截断）

    arXiv:2606.27954v2 Announce Type: replace  Abstract: Diffusion-based generative models with nonlinear or learned forward drifts raise questions that stay hidden for Gaussian forward processes: how the divergence-free and gradient parts of the drift shape the evolving density, whether sampling can draw on local rather than global information, and how to treat low-density regions, where $u$ carries a weak signal while $\log u$ diverges and the entropy density $u\log u$ falls between the two. These questions motivate the present study of local representations for Fokker--Planck equations with general, possibly nonlinear and time-dependent drift. We derive inhomogeneous heat-ball identities for the density $u$, the log-density $\log u$, the entropy density $u\log u$, and the score $\nabla\log u$, keeping the source terms generated by the drift; their unscaled residuals recover the corresponding differential residuals at leading order in the squared heat-ball radius. The weights appearing i
    
[^35]: 协变量漂移下部署风险的雅可比-速度界

    Jacobian-Velocity Bounds for Deployment Risk Under Covariate Drift

    [https://arxiv.org/abs/2605.04932](https://arxiv.org/abs/2605.04932)

    该论文通过时域庞加莱不等式与雅可比-速度定理建立了协变量漂移下冻结预测器长期部署的风险上界，并据此提出漂移对齐切向正则化（DTR）及配套监控指标来有效控制部署风险。

    

    我们研究了冻结预测器在动态协变量偏移下的长期部署问题。首先，一个时域庞加莱（Poincaré）不等式将时间上的风险波动性归约为导数能量，随后由雅可比-速度定理提供相应的逐路径控制。通过余项扩展，我们将经由模型介导的可控项与未通过得分雅可比传递的条件风险变化分离开来。在低秩漂移条件下，该界将可控项分解为平行和正交于漂移子空间的切向能量。这一结果启发了漂移对齐切向正则化（DTR）、其带有较弱正交护栏的各向异性扩展，以及与之相匹配的监控代理指标。受控实验验证了该时间不等式、方向性平滑的高效性以及余项的必要性。主要的实地研究采用一个128维气体传感器阵列在十个按时间顺序排列的批次上进行。一个秩为4的基估计……（摘要截断）

    arXiv:2605.04932v3 Announce Type: replace-cross  Abstract: We study long-horizon deployment of a frozen predictor under dynamic covariate shift. A time-domain Poincar\'e inequality first reduces temporal risk volatility to derivative energy, and a Jacobian-velocity theorem supplies the corresponding pathwise control. A remainder extension separates the model-mediated, controllable term from conditional-risk variation not transmitted through the score Jacobian. Under low-rank drift, the bound decomposes the controllable term into tangent energy parallel and orthogonal to the drift subspace. This motivates drift-aligned tangent regularization (DTR), its anisotropic extension with a weaker orthogonal guardrail, and a matched monitoring proxy. Controlled experiments verify the temporal inequality, the efficiency of directional smoothing, and the necessity of the remainder term. The main field study uses a 128-dimensional gas-sensor array over ten chronological batches. A rank-4 basis estim
    
[^36]: 基于辅助信息的灵活矩阵估计

    Flexible Matrix Estimation with Side Information

    [https://arxiv.org/abs/2603.24833](https://arxiv.org/abs/2603.24833)

    该论文提出将高维矩阵分解为行与列特征的非线性交互项、行特征驱动成分、列特征驱动成分及残余低秩结构四个互补成分的灵活估计框架，结合筛投影与核范数惩罚分别估计各成分，克服了归纳矩阵补全等现有方法的限制性假设。

    

    我们提出了一个灵活的高维矩阵估计框架，能够将行和列两方面的辅助信息纳入估计过程。现有方法（如归纳矩阵补全）往往施加了限制性结构——例如，要求精确的低秩协变量交互项、线性的协变量效应，并且难以利用仅由单侧（行或列）或双侧均无法解释的成分——而且常常省略显式的噪声成分。为解决这些局限性，我们提出将潜在矩阵分解为四个互补成分之和：（可能非线性的）行特征与列特征之间的交互项；由行特征驱动的成分；由列特征驱动的成分；以及无法由观测特征解释的残余低秩结构。通过结合基于筛（sieve）的投影与核范数惩罚，每个成分可以被分别估计……

    arXiv:2603.24833v2 Announce Type: replace-cross  Abstract: We introduce a flexible framework for high-dimensional matrix estimation to incorporate side information for both rows and columns. Existing approaches, such as inductive matrix completion, often impose restrictive structure---for example, an exact low-rank covariate interaction term, linear covariate effects, and limited ability to exploit components explained only by one side (row or column) or by neither---and frequently omit an explicit noise component. To address these limitations, we propose to decompose the underlying matrix as the sum of four complementary components: (possibly nonlinear) interaction between row and column characteristics; row characteristic-driven component, column characteristic-driven component, and residual low-rank structure unexplained by observed characteristics. By combining sieve-based projection with nuclear-norm penalization, each component can be estimated separately and these estimated comp
    
[^37]: 用于高效反向传播的无偏近似向量-雅可比积

    Unbiased Approximate Vector-Jacobian Products for Efficient Backpropagation

    [https://arxiv.org/abs/2602.14701](https://arxiv.org/abs/2602.14701)

    本文提出在反向传播中用随机化的无偏近似替代精确的向量-雅可比积，从理论上分析了精度与成本之间的权衡并证明了稀疏约束下最小方差的最优性，实验验证了该方法能有效降低深度学习的训练成本。

    

    在本工作中，我们提出了降低深度神经网络训练的计算与内存成本的方法。我们的方法是在反向传播过程中，用随机化的无偏近似来替代精确的向量-雅可比积。我们对达到目标精度所需的训练轮数与每轮成本降低之间的权衡进行了理论分析。随后，我们确定了向量-雅可比积的若干特定无偏估计，并证明了它们在稀疏性约束下具有最小方差的最优性质。最后，我们在多层感知机、BagNets 和视觉 Transformer 等架构上进行了深入实验。这些实验验证了我们的理论结果，并证实了所提出的无偏随机化反向传播方法在降低深度学习成本方面的潜力。

    arXiv:2602.14701v2 Announce Type: replace  Abstract: In this work we introduce methods to reduce the computational and memory costs of training deep neural networks. Our approach consists in replacing exact vector-jacobian products by randomized, unbiased approximations thereof during backpropagation. We provide a theoretical analysis of the trade-off between the number of epochs needed to achieve a target precision and the cost reduction for each epoch. We then identify specific unbiased estimates of vector-jacobian products for which we establish desirable optimality properties of minimal variance under sparsity constraints. Finally we provide in-depth experiments on multi-layer perceptrons, BagNets and Visual Transfomers architectures. These validate our theoretical results, and confirm the potential of our proposed unbiased randomized backpropagation approach for reducing the cost of deep learning.
    
[^38]: 关于临界点的近似计算

    On Approximate Computation of Critical Points

    [https://arxiv.org/abs/2601.21917](https://arxiv.org/abs/2601.21917)

    该论文证明了即使对系数极简单的非凸三次多项式，临界点的粗略近似计算也是计算难解的——除非P=NP，否则不存在多项式时间算法能输出近似临界点。

    

    我们证明了，即使对于一些简单的非凸函数类，计算临界点的粗略近似也是计算上难解的。我们首先提出了一族系数属于{-1,0,1}的n元三次多项式，它们具有精确的临界点，但没有任何多项式时间算法能够以标准表示形式写出一个√n/4-近似临界点。随后我们进一步证明，即使精确临界点具有简短的表示形式，计算困难性依然存在。更具体地说，我们证明：对于每个固定的η>0，除非P=NP，否则不存在多项式时间（甚至伪多项式时间）算法，能够接受一个系数为有理数{c_α}的n元三次多项式，并在该多项式在{-1,0,1}^n中存在临界点的情况下，返回一个梯度范数至多为(max_α |c_α|)·n^(1/2-η)的点。该算法在所有其他输入上可以返回任意点。

    arXiv:2601.21917v2 Announce Type: replace-cross  Abstract: We show that computing even coarse approximations of critical points is intractable for simple classes of nonconvex functions. We first present a family of $n$-variate cubic polynomials with coefficients in $\{-1,0,1\}$ that have exact critical points but for which no polynomial-time algorithm can write down a $\sqrt{n}/4$-approximate critical point in its standard representation. We then show that computational hardness persists when an exact critical point has a short representation. More specifically, we prove that for every fixed $\eta>0$, unless P $=$ NP, no polynomial-time (or even pseudo-polynomial-time) algorithm can take an $n$-variate cubic polynomial with rational coefficients ${c_\alpha}$ and return a point with gradient norm at most $(\max_\alpha |c_\alpha|)n^{1/2-\eta}$ whenever the polynomial has a critical point in $\{-1,0,1\}^n$. The algorithm may return an arbitrary point on all other inputs. Both results are 
    
[^39]: 偶对称与椭圆对称条件下变分推断的广义保证

    Generalized Guarantees for Variational Inference in the Presence of Even and Elliptical Symmetry

    [https://arxiv.org/abs/2511.01064](https://arxiv.org/abs/2511.01064)

    该论文将反向KL散度变分推断在偶对称下恢复均值、椭圆对称下恢复相关矩阵的保证，从对数凹目标分布推广到更一般的单峰（含重尾）目标分布，并扩展至仅部分对称的情形。

    

    变分推断（VI）通过在一族易于处理的分布 $\mathcal Q$ 中寻找最佳匹配 $q$ 来近似目标密度 $p$。变分近似通过最小化某种散度求得，最常用的是反向 Kullback-Leibler 散度 $\text{KL}(q||p)$，不过也可以考虑其他散度。在实践中，$p \notin \mathcal Q$，因此最优变分近似存在误设（misspecification）。尽管存在这种误设，最近的研究表明，对于对数凹的目标分布 $p$，$\text{KL}(q||p)$ 的最小化点在偶对称条件下保证能够恢复均值，在椭圆对称条件下同样保证能够恢复相关矩阵。我们在若干方向上对这一结果进行了扩展。首先，我们放宽了对数凹性假设，仅要求 $p$ 是单峰的，从而允许目标分布具有重尾特性。其次，我们针对 $p$ 仅在某些（方向上）表现出对称性的情形获得了相应保证……

    arXiv:2511.01064v4 Announce Type: replace-cross  Abstract: Variational inference (VI) approximates a target density $p$ by the best match $q$ in a family $\mathcal Q$ of tractable distributions. The variational approximation is found by minimizing a divergence, most commonly the reverse Kullback-Leibler divergence $\text{KL}(q||p)$, although other divergences can be considered. In practice, $p \notin \mathcal Q$ and so the optimal variational approximation is misspecified. Despite this misspecification, it was recently shown that the minimizer of $\text{KL}(q||p)$ is guaranteed to recover the mean in the presence of even symmetry and likewise the correlation matrix in the presence of elliptical symmetry for a log-concave target $p$. We extend this result in several directions. First, we relax the assumption of log-concavity and instead merely require $p$ to be unimodal, allowing for heavy-tailed targets. Second, we obtain guarantees for cases where $p$ exhibits symmetry along some but 
    
[^40]: 大语言模型pass@k缩放规律的高效预测

    Efficient Prediction of pass@k Scaling in Large Language Models

    [https://arxiv.org/abs/2510.05197](https://arxiv.org/abs/2510.05197)

    本文提出了一种稳健的估计框架，能够在远小于大规模尝试次数的采样预算下高效准确地预测大语言模型的pass@k缩放行为，从而帮助模型提供商和监管机构评估模型的能力与风险。

    

    评估前沿AI系统的能力与风险是一个重要的研究领域；最近的研究表明，重复采样可以显著地同时提升这两者。例如，重复采样使模型能够解决困难的数学和编程问题，但也可能增加其被越狱攻击的风险。这些结果引出了一个关键问题：在采样预算远小于实际尝试次数的情况下，如何准确预测模型在扩展到海量尝试次数时的行为？这个问题与每天为数亿用户提供服务的模型提供商，以及寻求防止危害的政府监管机构直接相关。为回答这一问题，我们做出了三项贡献。首先，我们发现用于拟合这些规律的标准方法存在统计上的缺陷，阻碍了预测的准确性，尤其是在数据有限的场景中。其次，我们通过引入一个稳健的估计框架来弥补这些缺陷

    arXiv:2510.05197v2 Announce Type: replace  Abstract: Assessing the capabilities and risks of frontier AI systems is a critical area of research; recent work has shown that repeated sampling can dramatically increase both. For instance, repeated sampling allows models to solve difficult math and coding problems, but it may also increase their risk of being jailbroken. Such results raise a crucial question: how can one accurately predict a model's behavior when scaled to a massive number of attempts, given a vastly smaller sampling budget? This question is directly relevant to model providers, who serve hundreds of millions of users daily, and to governmental regulators, who seek to prevent harms. To answer this question, we make three contributions. First, we find that standard methods for fitting these laws suffer from statistical shortcomings that hinder predictions, especially in data-limited scenarios. Second, we remedy these shortcomings by introducing a robust estimation framework
    
[^41]: 具有分组子空间结构的极稀疏多层网络中的完美聚类

    Perfect Clustering in Extremely Sparse Multilayer Networks with Groupwise Subspace Structure

    [https://arxiv.org/abs/2507.19423](https://arxiv.org/abs/2507.19423)

    本文提出一个极其广义的多层网络聚类模型——各层遵循广义随机点积图模型且可按潜在子空间结构分组，并证明了即使各层网络极度稀疏、逐层分析失效时，仍能完美地恢复具有不同子空间结构的层分组。

    

    本文研究了建立在同一节点集合上的多层网络，其中每一层都遵循广义随机点积图模型。此外，所有层可以被划分为若干组，每组拥有其独特的潜在子空间结构，而除此之外各层的连接概率矩阵可以各不相同。这一设置极其广泛，将大多数多层网络模型都作为特例包含在内。同时，我们关注层网络可能极度稀疏的情形，在这种情况下逐层分析是不可靠的。该问题也可以被视为经典问题的推广：从一整个矩阵集合中恢复共享结构，扩展到结构仅在特定矩阵组内共享的情形。此设置下的关键任务是恢复具有唯一子空间结构的层分组。

    arXiv:2507.19423v2 Announce Type: replace-cross  Abstract: The paper studies a multilayer network on the same collection of nodes, where each layer follows the Generalized Random Dot Product Graph Model. Furthermore, all layers can be partitioned into groups, each group with its own unique latent subspace structure, but otherwise the matrices of connection probabilities can all be different. This setting is extremely general and includes most multilayer network models as special cases. In addition, we are interested in the situation where the layer networks may be extremely sparse and, therefore, the layer-by-layer analysis is unreliable. This problem can also be seen as a generalization of the classic setting in which one recovers a shared structure across an entire collection of matrices, extending it to the case where such structures are shared only within specific groups of matrices. The key task in this setting is to recover the groups of layers with unique subspace structures, si
    
[^42]: Wasserstein-Fisher-Rao梯度流的序贯蒙特卡洛近似

    Sequential Monte Carlo approximations of Wasserstein--Fisher--Rao gradient flows

    [https://arxiv.org/abs/2506.05905](https://arxiv.org/abs/2506.05905)

    该论文提出了一种基于Wasserstein-Fisher-Rao几何、利用重要性采样与序贯蒙特卡洛的新算法来近似KL散度梯度流以实现分布采样，并通过广泛的实证研究揭示了该算法优于其他流行蒙特卡洛算法的适用场景。

    

    我们考虑从一个概率分布 $\pi$ 中采样的问题。众所周知，该问题可以表述为概率分布空间上的一个优化问题，其目标是最小化与 $\pi$ 之间的Kullback-Leibler散度（KL散度）。我们考虑了若干偏微分方程（PDE），其解是使KL散度最小化的分布，并将这些方程与著名的蒙特卡洛算法联系起来。我们特别关注通过在概率分布空间上引入Wasserstein-Fisher-Rao几何所得到的PDE，并证明这些方程可以自然地通过重要性采样和序贯蒙特卡洛方法来实现。我们提出了一种新颖的算法来近似KL散度的Wasserstein-Fisher-Rao流，并进行了广泛的实证研究，以确定该算法在何种情况下优于其他流行的蒙特卡洛算法。

    arXiv:2506.05905v3 Announce Type: replace-cross  Abstract: We consider the problem of sampling from a probability distribution $\pi$. It is well known that this can be written as an optimisation problem over the space of probability distribution in which we aim to minimise the Kullback--Leibler divergence from $\pi$. We consider several partial differential equations (PDEs) whose solution is a minimiser of the Kullback--Leibler divergence from $\pi$ and connect them to well-known Monte Carlo algorithms. We focus in particular on PDEs obtained by considering the Wasserstein--Fisher--Rao geometry over the space of probabilities and show that these lead to a natural implementation using importance sampling and sequential Monte Carlo. We propose a novel algorithm to approximate the Wasserstein--Fisher--Rao flow of the Kullback--Leibler divergence and conduct an extensive empirical study to identify when this algorithm outperforms other popular Monte Carlo algorithms.
    
[^43]: 因果片段：逐片分析和改进脉冲神经网络

    Causal pieces: analysing and improving spiking neural networks piece by piece

    [https://arxiv.org/abs/2504.14015](https://arxiv.org/abs/2504.14015)

    提出“因果片段”新概念来分析脉冲神经网络，证明片段内输出脉冲时间对输入和参数局部Lipschitz连续，且因果片段数量可作为衡量SNN逼近能力的有效指标。

    

    我们提出了“因果片段”，这是一个用于分析脉冲神经网络（SNN）的新概念，其灵感来源于人工神经网络（ANN）中用于研究表达能力和可训练性的“线性片段”。因果片段将具有单脉冲编码的前馈SNN的输入和参数空间划分为不同的区域，在每个区域中，相同的子网络导致输出脉冲的产生。对于由具有大膜时间常数的基于电流的泄漏积分发放（LIF）神经元构成的网络，我们证明了在每个因果片段内，输出脉冲时间相对于输入和网络参数是局部Lipschitz连续的。我们进一步证明了依赖于因果片段数量的逼近误差下界。因此，因果片段的数量是衡量SNN逼近能力的一个度量指标，该指标在脉冲时间不连续的情况下仍然有效，并且适用于同时具有兴奋性和抑制性突触的网络。

    arXiv:2504.14015v2 Announce Type: replace-cross  Abstract: We introduce "causal pieces", a novel concept for analysing spiking neural networks (SNNs), inspired by "linear pieces" used to study expressivity and trainability in artificial neural networks (ANNs). Causal pieces partition the input and parameter space of a feedforward SNN with single-spike coding into distinct regions where the same subnetwork causes the output spikes. For networks of current-based leaky integrate-and-fire (LIF) neurons with large membrane time constants, we show that within each causal piece, output spike times are locally Lipschitz continuous with respect to inputs and network parameters. We further prove a lower bound on the approximation error that depends on the number of causal pieces. Thus, the number of causal pieces is a measure of the approximation capabilities of SNNs, which is valid despite spike-time discontinuities and applies to networks with both excitatory and inhibitory synapses. Empirical
    
[^44]: 多项式次数多项式核的快速草图化

    Fast Sketching of Polynomial Kernels of Polynomial Degree

    [https://arxiv.org/abs/2108.09420](https://arxiv.org/abs/2108.09420)

    本文提出了一种新的无意识草图方法，消除了运行时间对多项式核次数的依赖，大幅改进了神经正切核、反余弦核等慢增长核的近似运行速度。

    

    核方法是机器学习中的基础方法，更快的核近似算法可以直接加速机器学习中的许多核心任务。多项式核尤其重要，因为其他核通常可以通过泰勒级数展开用多项式核来近似。最近的无意识草图技术将运行时间对多项式核次数 q 的依赖从指数级降低到多项式级，这对于高斯核非常有用，因为在数据半径有界且岭参数 λ ≥ 1/poly(n) 的情况下，q 可以选择为多对数级。然而，对于增长较慢的核，例如神经正切核和反余弦核，q 需要取多项式级，而先前的工作在运行时间上会产生多项式因子的减速。我们提出了一种新的无意识草图方法，通过消除这种依赖性，大幅改进了运行时间。

    arXiv:2108.09420v2 Announce Type: replace-cross  Abstract: Kernel methods are fundamental in machine learning, and faster algorithms for kernel approximation provide direct speedups for many core tasks in machine learning. The polynomial kernel is especially important as other kernels can often be approximated by the polynomial kernel via a Taylor series expansion. Recent techniques in oblivious sketching reduce the dependence in the running time on the degree $q$ of the polynomial kernel from exponential to polynomial, which is useful for the Gaussian kernel, for which $q$ can be chosen to be polylogarithmic (for data of bounded radius and ridge parameter $\lambda\geq 1/\mathrm{poly}(n)$). However, for more slowly growing kernels, such as the neural tangent and arc-cosine kernels, $q$ needs to be polynomial, and previous work incurs a polynomial factor slowdown in the running time. We give a new oblivious sketch which greatly improves upon this running time, by removing the dependence
    
[^45]: 极端环境下的个体治疗效果

    Individual Treatment Effects in Extreme Regimes. (arXiv:2306.11697v1 [stat.ME])

    [http://arxiv.org/abs/2306.11697](http://arxiv.org/abs/2306.11697)

    本文提出了一种新的框架，通过测量潜在结果在存在或缺乏治疗的情况下的尾部衰减率变化，来估计极端环境下的个体治疗效果（ITE$_2$）。

    

    了解极端环境下的个体治疗效果对于描述不同干预策略的风险至关重要。但极端环境数据很难收集，因为它在实践中很少被观察到。为了解决这个问题，我们提出了一个新的框架来估计极端环境下的个体治疗效果（ITE$_2$）。具体而言，我们通过测量潜在结果在存在或缺乏治疗的情况下的尾部衰减率变化来量化这种效果。随后，我们建立了 ITE$_2$ 的计算条件，并开发了计算算法。我们在各种合成和半合成数据集上演示了我们提出的方法的功效。

    Understanding individual treatment effects in extreme regimes is important for characterizing risks associated with different interventions. This is hindered by the fact that extreme regime data may be hard to collect, as it is scarcely observed in practice. In addressing this issue, we propose a new framework for estimating the individual treatment effect in extreme regimes (ITE$_2$). Specifically, we quantify this effect by the changes in the tail decay rates of potential outcomes in the presence or absence of the treatment. Subsequently, we establish conditions under which ITE$_2$ may be calculated and develop algorithms for its computation. We demonstrate the efficacy of our proposed method on various synthetic and semi-synthetic datasets.
    

