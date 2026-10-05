# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simulation-Free Learning of Population Dynamics with Wasserstein Lagrangian Residuals](https://arxiv.org/abs/2610.03679) | 提出免模拟方法Double-Stitch，通过惩罚学习到的种群路径上运动方程的残差来学习Wasserstein空间中的拉格朗日力学，从而以低训练成本建模保守性和周期性的种群动力学。 |
| [^2] | [Planning to Learn](https://arxiv.org/abs/2610.03667) | 该论文揭示了精确策略梯度输给交叉熵的根本原因在于其短视性——只看重即时收益而忽视每次更新对未来学习的奠基作用，并将交叉熵重新诠释为“耐心的准确率”（考虑未来学习收益的长时域误差总量），从而通过在剩余学习量处截断该总量来改进优化方法。 |
| [^3] | [Amortized Structured Stochastic Variational Inference for Gaussian Process Latent Variable Models](https://arxiv.org/abs/2610.03647) | 本文将摊销结构化随机变分推断应用于高斯过程潜变量模型，通过让潜空间的变分后验条件依赖于诱导点的取值，突破了平均场变分近似的局限，从而在数据流形重建的多项指标上取得了改进。 |
| [^4] | [When May a Bandit Leave Its Anchor? E-Process-Authorized Thompson Sampling under Non-stationarity](https://arxiv.org/abs/2610.03646) | 提出e-过程授权的汤普森采样（e-ATS），通过随时有效的e-过程决定非平稳环境中每只手臂何时从全历史锚点切换到折扣遗忘状态，保证平稳条件下偏离乐观汤普森采样的概率不超过设定的 $\alpha_E$，实验表明证据控制的是适应何时开始而非其是否总有益。 |
| [^5] | [Broken scale symmetries in undercomplete linear autoencoders](https://arxiv.org/abs/2610.03640) | 该论文发现，在欠完备线性自编码器中，有限步长的SGD会以有向的方式破坏尺度对称性，在PCA解流形上系统性地偏向放大解码器权重，直至动力学触及有限步长的稳定性边界。 |
| [^6] | [Below what training size do deep tabular generators stop beating trivial baselines? A preregistered benchmark on a size ladder of clinical and standard datasets](https://arxiv.org/abs/2610.03500) | 这项预注册的规模阶梯基准测试（共2,220次运行）发现，在临床等小型数据集上，几乎所有深度表格生成模型（如CTGAN、TVAE、TabDDPM）在任何测试的训练规模下都无法以超过随机噪声的幅度击败简单基线方法，挑战了深度生成模型在小型表格数据上的价值假设。 |
| [^7] | [AREX: Affine-Residual Exponential Integrator for Few-Step Sampling in Flow Matching](https://arxiv.org/abs/2610.03483) | AREX是一种无需训练的流匹配模型少步采样器，它将采样动力学分解为由目标均值和协方差决定的仿射分量（用显式矩阵值传播子积分）与神经残差项，在无需重训练的情况下持续提升少步采样的样本保真度。 |
| [^8] | [When Is Accuracy Evidence? A Unified Theory of Generalisation, Validation, and Information Fusion](https://arxiv.org/abs/2610.03465) | 该论文提出一个统一的指数框架，将交叉验证准确率转化为真实风险的保守上界，并通过有效折数Keff证明：当各折数据强相关时，单纯增大交叉验证折数K并不能增强统计证据。 |
| [^9] | [Iterating Consistency Models: Stability, Error Bounds and Noise Schedules](https://arxiv.org/abs/2610.03414) | 该论文将多步一致性模型采样分析为加噪与近似去噪算子的复合，在可验证的稳定性假设下推导出非渐近误差界，揭示了噪声调度中早期大噪声驱动误差收缩、后期小噪声控制残余偏差的作用机制，为CM采样器设计提供了理论指导。 |
| [^10] | [DAWIS: Data Assimilation with Windowed Inverse Sampling via Multitask Interpolants](https://arxiv.org/abs/2610.03314) | DAWIS提出了一种统一的数据同化框架，通过用覆盖连续状态窗口的多任务随机插值子替代单一流时间先验，使滤波、固定滞后平滑和分块平滑得以在同一框架内实现，从而能在新观测到达时修正过去状态并避免误差累积。 |
| [^11] | [SDECast: Probabilistic Weather Forecasting in Continuous Time with Neural SDEs](https://arxiv.org/abs/2610.03313) | SDECast提出了一个基于神经随机微分方程的连续时间概率天气预报框架，无需训练时重复SDE模拟即可直接在物理空间学习随机动力学，并成功扩展到小时级分辨率的全球天气预报。 |
| [^12] | [Near-Optimal Convex Optimization with Lazy Second-Order Oracles](https://arxiv.org/abs/2610.03222) | 本文针对惰性二阶预言机凸优化问题，通过新的分块零链下界构造和匹配的算法设计，将复杂度界改进至 $\tilde{\mathcal{O}}(m+ m^{1/7} \epsilon^{-2/7})$ 并在对数因子内紧致，显著优于先前结果。 |
| [^13] | [Predictively Oriented Gaussian Process Posteriors](https://arxiv.org/abs/2610.03201) | 提出预测导向高斯过程（PrO-GPs），将预测不确定性作为主要推断目标，在模型错误设定下比标准高斯过程产生校准更好的预测分布。 |
| [^14] | [Invariance of Clustering Operations in Causal Effect Identification](https://arxiv.org/abs/2610.03101) | 本文提出了一类基于原图c-分量条件的“识别不变”聚类操作，可同时保持因果效应的可识别性与不可识别性，从而安全地简化因果图并加速因果效应识别。 |
| [^15] | [GTDD: Generative Test-Driven Development for AI Coding Agents with Adversarial Testing](https://arxiv.org/abs/2610.02952) | 提出生成式测试驱动开发（GTDD），由独立的测试智能体在每轮候选实现后基于人类指定的行为契约对抗性地生成新测试输入并不断反馈反例，以解决AI编码智能体过拟合固定测试集而导致预期行为遗漏的问题。 |
| [^16] | [Cross-Fitting Under Nonregularity: Normality and Inference via Locality](https://arxiv.org/abs/2610.02944) | 本文证明在非正则条件下，一大类交叉拟合估计量仍满足中心极限定理，但需针对交叉折相关性修正渐近方差，并据此提出了可达到渐近名义覆盖率的新置信区间构造方法。 |
| [^17] | [A Residual Tree Gaussian Process Modeling Framework for High-Dimensional Data](https://arxiv.org/abs/2610.02893) | 该论文提出 ResTGP，一种贝叶斯残差树高斯过程框架，通过沿二叉树在多分辨率层级上迭代分解预测过程与残差过程，实现了对高维异质大空间数据的灵活多尺度协方差建模与分而治之的高效计算。 |
| [^18] | [Muon Learns Facts Better: Understanding the Role of Spectral Orthogonalization](https://arxiv.org/abs/2610.02798) | 本文通过可解析的事实回忆模型和线性 Transformer 分析了 Muon 优化器中谱正交化的作用机制，揭示其对特征学习动力学的改变，并说明这使 Muon 比梯度下降和 Adam 更擅长学习“主体-关系到答案”的事实映射。 |
| [^19] | [Hold-Out Scoring for Efficient Gaussian DAG Learning](https://arxiv.org/abs/2610.02785) | 该论文提出HOST算法，以逐节点留出评分与凸回归取代子集搜索，仅凭对得分误差的单侧控制即可恢复正确的节点排序，从而在无需入度上界的情况下实现高效的高斯DAG学习。 |
| [^20] | [Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback](https://arxiv.org/abs/2610.02771) | 本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。 |
| [^21] | [Differential Privacy of Gradient Descent on Perturbed Objectives](https://arxiv.org/abs/2610.02716) | 该论文证明了在强凸光滑目标上，对扰动目标运行梯度下降的有限次迭代是高斯噪声的 $C^1$ 微分同胚（并给出雅可比最小奇异值的定量下界），从而可直接用换元法分析有限迭代的差分隐私，且对广义线性模型而言，迭代条件成立时隐私界不显式依赖环境维度。 |
| [^22] | [Generalization Properties of Score-matching Diffusion Models for Intrinsically Low-dimensional Data](https://arxiv.org/abs/2610.02663) | 该论文为流匹配模型在具有内在低维结构的数据上提供了统计泛化理论保证，推导出依赖于数据内在维度的 Wasserstein-p 有限样本误差界，克服了以往分析中限制性假设和忽略低维结构的不足。 |
| [^23] | [High-Dimensional Asymptotics and Dataset Selection for Private Transfer Learning](https://arxiv.org/abs/2610.02578) | 本文针对隐私保护迁移学习中的数据集选择问题，提出了一种仅基于汇总统计量、采用加权岭估计器的多元异构源高维回归方法，在 ρ-零集中差分隐私保证下研究其高维渐近性质，以判断额外数据是否值得购买或纳入协同学习。 |
| [^24] | [ENCORE: Exact Non-equilibrium COntrol with Replica Exchange for Diffusion Generation](https://arxiv.org/abs/2610.02538) | 该论文提出了首个精确的并行推理时控制方法ENCORE，通过让每个副本保存生成轨迹使向上移动成为截断操作，从而避免模拟难以处理的时间反转，实现了无偏的副本交换控制。 |
| [^25] | [Learning Style, Forgetting Semantics: A Case Study of SFT and RFT on Classification Tasks](https://arxiv.org/abs/2610.02437) | 本文通过将策略更新精确分解为语义与风格两个成分，揭示了SFT比RFT遗忘更多的原因——SFT会沿教师风格偏好产生离轴风格漂移从而破坏语义记忆，而RFT能保持类内风格对称性。 |
| [^26] | [Conformal Prediction for Time Series with Deep Sequence Models](https://arxiv.org/abs/2610.02357) | 本文首次系统性地研究了深度序列模型在时间序列共形预测中的应用，通过条件分位数回归等三种方法，解决了传统共形预测所依赖的数据可交换性假设在时间序列中不成立的问题。 |
| [^27] | [Expected Utility Regret Rule: Minimax and Bayes Optimal Portfolio Choice](https://arxiv.org/abs/2610.02290) | 提出期望效用遗憾（EUR）规则，该规则无需先验分布即可同时达到极小极大与贝叶斯最优下界，并将均值-方差组合和风险平价组合统一为该框架的特例。 |
| [^28] | [TRACE: A Reproducible Benchmark for Electricity Price Forecasting with Official Operational Text](https://arxiv.org/abs/2610.02256) | TRACE是一个将电力价格与预测截止时点官方运行文本配对、并严格防止信息泄露的可复现电力价格预测基准，它证明了文本上下文的预测价值——使时间序列基础模型的上尾pinball损失中位数降低7.4%。 |
| [^29] | [Counterfactual Predictions in Scientific Emulators Without Controlled Experiments](https://arxiv.org/abs/2610.02252) | 提出 ReRoute 框架，无需受控实验或模拟器数据，仅通过将查询输入固定为参考值并沿已知机制路径重新引入其变化，结合事实数据微调，即可让科学模拟器准确回答“如果条件不同会怎样”的反事实预测问题。 |
| [^30] | [Nearest-neighbour baselines for fingerprint prediction from MS/MS spectra under different assumptions](https://arxiv.org/abs/2610.02249) | 该论文系统比较了在不同推理信息假设下的多种最近邻检索变体用于从MS/MS谱图预测分子指纹，旨在建立更严格的基线以实现更严谨的基准测试并更好地衡量领域进展。 |
| [^31] | [Feature tracking in physics-informed neural networks via joint optimization of nonlinear deformation manifolds: application to shocks](https://arxiv.org/abs/2610.02230) | 提出特征追踪PINN（FT-PINN），通过联合优化解网络与参数化非线性流形上的微分同胚变形映射，使配点自动集中于弯曲、倾斜、合并等任意几何形状的激波特征处，无需先验位置信息即可提升含激波守恒律问题的求解精度。 |
| [^32] | [Pragmatic DML with AI-Learned Representations](https://arxiv.org/abs/2610.01935) | 该论文揭示了AI学习表示的误差会以结果回归误差与平衡权重误差的乘积形式影响因果参数估计，并证明了交叉拟合DML可为依赖表示的目标提供有效推断，且与按折表示学习/微调兼容，为基于AI表示的因果推断提供了实用框架。 |
| [^33] | [Platonic Task Arithmetic](https://arxiv.org/abs/2610.00929) | 本文提出“柏拉图任务向量”概念，并引入形状与模型架构和嵌入维度无关的“通用任务描述符”矩阵，使任务算术（如任务加法与取反）首次能够跨越不同模型架构进行迁移与应用。 |
| [^34] | [Learning to Price Electricity for Optimal Demand Response](https://arxiv.org/abs/2610.00755) | 本文提出一种基于神经网络的上下文电价定价算法，将定价建模为Stackelberg博弈并学习从上下文特征到可行电价的受限映射，通过模拟美国多个城市电网验证了该方法能显著提升需求响应计划的价值。 |
| [^35] | [Copula Active Subspaces I: A Score-Covariance Method for Reduced-Order Non-Gaussian Density Estimation](https://arxiv.org/abs/2609.36142) | 提出 Copula 活跃子空间（CAS）方法，利用 copula 得分协方差的主特征向量识别非高斯噪声分布中依赖结构的变化方向，从而实现贝叶斯推断中非高斯噪声密度的降阶表示与估计。 |
| [^36] | [Learning from the Gap Between Pass@K and Pass@1](https://arxiv.org/abs/2609.35793) | 提出 GapFT 方法，通过在 Pass@K 与 Pass@1 的差距（即单样本失败但 K 个样本内可解决的问题）上进行微调，将测试时搜索带来的能力吸收进模型，从而提升单样本解码的性能。 |
| [^37] | [AECSF: Adaptive Ensemble Conditional Score Filtering for High-Dimensional Nonlinear Data Assimilation](https://arxiv.org/abs/2609.32411) | 提出了一种免训练的自适应集成条件得分滤波器AECSF，利用条件Tweedie恒等式构建解析可处理的得分估计器，从而在高维非线性数据同化中同时避免粒子权重退化并捕捉非高斯后验结构。 |
| [^38] | [DeepGOF-1: A Pretrained Convolutional Goodness-of-Fit Test for Logistic Regression with a Computable Consistency Certificate](https://arxiv.org/abs/2609.29575) | 提出了一种统计量为预训练冻结卷积网络的逻辑回归拟合优度检验，p值通过分析者自身的bootstrap校准来保证检验水平的精确性，并首次提供可通过单次前向传播计算得出的一致性证书。 |
| [^39] | [Penalized Nonreversible Langevin for Constrained Sampling](https://arxiv.org/abs/2609.25381) | 提出了将平方距离惩罚与非可逆斜对称扰动相结合的朗之万算法以实现紧凸集上的约束采样，并在对数索博列夫不等式与漂移收缩条件下给出了非渐近的总变差和 2-Wasserstein 误差界。 |
| [^40] | [EDGE: a closed-form directed test for the calibration of probabilistic binary classifiers](https://arxiv.org/abs/2608.20511) | 本文提出EDGE，一种针对逻辑回归的闭合形式校准检验，通过将分箱残差投影到平滑扭曲基上，提供具有零分布的统计检验，克服了传统可靠性图无法区分真实校准误差与噪声的局限性。 |
| [^41] | [Fine-Tuning Generative Models for Extreme Events via CVaR-Penalized Wasserstein Gradient Flows](https://arxiv.org/abs/2608.11544) | 提出了一种基于CVaR惩罚的Wasserstein梯度流方法，无需先验知识即可微调生成模型以捕捉重尾分布和极端事件，克服了标准生成器在尾部欠采样时速度消失的局限。 |
| [^42] | [Deep learning-based prediction of time-resolved adhesive forces in viscoelastic Hertzian contacts](https://arxiv.org/abs/2607.19060) | 本文提出一种标量条件化的有状态序列到序列深度学习模型，结合固定测量步长（FMS）表示方法，能够从位移历史快速预测粘弹性赫兹接触中的完整时间分辨粘附力演化，克服了传统数值模拟计算成本高、无法用于实时应用和设计优化的局限。 |
| [^43] | [DAGR: State-Conditioned Goal Representations via Difference-Aware Goal Cross-Attention](https://arxiv.org/abs/2607.13731) | 提出DAGR方法，通过多尺度门控交叉注意力和差异感知的注意力规则，将目标条件强化学习中静态的目标嵌入精炼为状态条件化表示，使策略能直接感知目标中尚未完成的部分，并从理论上揭示了后归一化放置方式对门控结构保证条件的破坏。 |
| [^44] | [Decision-Aware Training for Sample-Based Generative Models](https://arxiv.org/abs/2607.01171) | 提出决策感知训练方法，通过可微分优化层计算决策损失并将其与能量分数结合，使样本生成模型的训练能够直接惩罚下游决策成本，从而在高风险决策场景中生成更具实用价值的概率预测。 |
| [^45] | [Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference](https://arxiv.org/abs/2606.27090) | 本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。 |
| [^46] | [Diffusion Flow Matching: Dimension-Improved KL Bounds and Wasserstein Guarantees](https://arxiv.org/abs/2606.16610) | 本文为基于布朗运动的扩散流匹配提供了在KL散度和2-Wasserstein距离下具有更优维度依赖性的离散化误差收敛保证，在温和条件下达到了最先进的收敛标度。 |
| [^47] | [Operator Calculus for Population-Based Optimization: Modular Convergence and Finite-Population Guarantees](https://arxiv.org/abs/2606.14289) | 本文提出一种面向基于种群优化的算子微积分框架，使经过独立验证的更新规则效应可以模块化地组合，为收敛性分析提供可复用的构建模块，并给出有限评估预算下的收敛保证。 |
| [^48] | [Reliability of Probabilistic Emulation of Physical Systems](https://arxiv.org/abs/2606.12997) | 本研究开发了一个评估框架，在匹配的模型规模和计算预算下系统比较了生成式模型与CRPS训练的确定性模型集合在物理系统概率预报中的表现，发现CRPS训练的模型集合在预测区间的经验覆盖率上通常具有更可靠的不确定性。 |
| [^49] | [On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance](https://arxiv.org/abs/2606.00467) | 提出“定义特定熟悉度”（DSF）指标，证明大语言模型内化先验与任务定义的对齐程度能显著预测其标注性能，且提示中的额外信息难以纠正模型零样本的“决策粘性”错误。 |
| [^50] | [Escaping the Capacity Ceiling: Routing on the Stiefel Manifold for Bilinear SPD Layers](https://arxiv.org/abs/2605.31043) | 提出SCAP层，通过交叉注意力将K个Stiefel专家滤波器动态组合为样本特定的双线性映射，从而突破SPD网络中单滤波器的容量上限，解决堆叠BiMap层无法提升容量的问题。 |
| [^51] | [An Elastic Shape Variational Autoencoder for Skeleton Pose Trajectories](https://arxiv.org/abs/2605.09231) | 提出了一种基于Kendall形状流形上TSRVF表示的几何感知生成模型ES-VAE，通过固有地消除平移、旋转、缩放和执行速度等干扰因素，专注于骨架姿态轨迹内在形状动态的建模。 |
| [^52] | [Classical and Quantum Speedups for Non-Convex Optimization via Energy Conserving Descent](https://arxiv.org/abs/2604.13022) | 本文首次对能量守恒下降（ECD）进行了理论分析，证明其随机版本和量子版本在非凸优化中相比随机梯度下降和量子隧穿游走基线均能实现指数级的命中时间加速，且量子版本在高势垒问题上具有进一步的加速优势。 |
| [^53] | [Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning](https://arxiv.org/abs/2602.10273) | Power-SMC是一种免训练的低延迟序列级幂采样方法，以接近标准解码的速度实现分布锐化，从而提升大语言模型的推理能力。 |
| [^54] | [When Does Pooling Pay? Credibility and Resolution under Forgetting in Intermittent-Demand Forecasting](https://arxiv.org/abs/2511.12749) | 该论文证明，在分层经验贝叶斯间歇性需求预测模型中，“遗忘序列自身历史”与“跨序列信息共享”可统一为同一个决策，并提出拟合窗口诊断、可信度界、分辨条件及事前筛选准则，用以判断何时池化信息才有价值。 |
| [^55] | [High-Dimensional Asymptotics of Differentially Private PCA](https://arxiv.org/abs/2511.07270) | 该论文针对差分隐私主成分分析，通过分析指数机制，在高维设置下给出了隐私损失随数据集变化的精确渐近刻画，弥补了传统一致上界在特定数据集上过于保守的不足。 |
| [^56] | [Bifidelity Karhunen-Lo\`eve Expansion Surrogate with Active Learning for Random Fields](https://arxiv.org/abs/2511.03756) | 提出了一种将Karhunen-Loève展开与多项式混沌展开相结合的双保真度代理模型，并利用基于交叉验证和高斯过程回归的主动学习策略自适应选择高保真度采样点，从而在有限计算成本下实现随机场的高精度建模。 |
| [^57] | [Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station](https://arxiv.org/abs/2510.23463) | 该论文研究了基于多天线基站的多址接入衰落信道上的空中联邦学习，巧妙地将信道噪声从性能损害转化为差分隐私保护的天然随机性来源，突破了现有工作在信道模型和损失函数假设上的限制，实现了隐私保护与训练性能的协同优化。 |
| [^58] | [A fast non-reversible sampler for Bayesian mixture models](https://arxiv.org/abs/2510.03226) | 该论文提出了一种适用于贝叶斯混合模型的新型非可逆采样方案，理论上保证其渐近方差不会比标准可逆采样器差超过四倍，并在实际场景（尤其是大样本和组分重叠情形）中显著加快收敛速度。 |
| [^59] | [Error Propagation in Dynamic Programming: From Stochastic Control to American Option Pricing](https://arxiv.org/abs/2509.20239) | 本文为离散时间随机最优控制建立了结合再生核希尔伯特空间回归与蒙特卡洛抽样的动态规划近似框架，提出自然的误差分解并严格分析了误差从到期日向初始时刻反向传播的规律，可应用于美式期权定价。 |
| [^60] | [Estimating prevalence with precision and accuracy](https://arxiv.org/abs/2507.06061) | 本文提出了一种贝叶斯聚合量化器PQ，它在保证足够覆盖率的同时生成更窄的预测区间，从而比现有方法更精确地估计流行率并更有效地量化估计的不确定性。 |
| [^61] | [Asymptotic Performance of Time-Varying Bayesian Optimization](https://arxiv.org/abs/2505.13012) | 本文首次为时变贝叶斯优化（TVBO）算法的累积遗憾提供了上界和与算法无关的下界，推导出算法具有无悔性质的充分条件，且其分析首次覆盖了实践中使用的所有主要类别的平稳核函数。 |
| [^62] | [Flexible Nonparametric Inference for Causal Effects under the Front-Door Model](https://arxiv.org/abs/2312.10234) | 本文在前门准则下针对平均处理效应和处理组平均处理效应提出了多种新颖的一步估计器和目标最小损失估计器，这些估计器兼容灵活的机器学习冗余参数估计，并建立了根号n一致性和渐近线性的理论条件。 |
| [^63] | [Score diffusion models without early stopping: finite Fisher information is all you need.](http://arxiv.org/abs/2308.12240) | 无早停的分数扩散模型不需要得分函数的Lipschitz均匀条件，只需要有限的费舍尔信息。 |

# 详细

[^1]: 基于Wasserstein拉格朗日残差的免模拟种群动力学学习

    Simulation-Free Learning of Population Dynamics with Wasserstein Lagrangian Residuals

    [https://arxiv.org/abs/2610.03679](https://arxiv.org/abs/2610.03679)

    提出免模拟方法Double-Stitch，通过惩罚学习到的种群路径上运动方程的残差来学习Wasserstein空间中的拉格朗日力学，从而以低训练成本建模保守性和周期性的种群动力学。

    

    细胞、生物体和流体的动力学通常被建模为随时间演化的概率分布。从不成对的快照中重建和外推这种演化需要对潜在过程做出假设。Wasserstein梯度流是一种常见的选择，但它无法描述保守性或周期性动力学。Wasserstein空间中的拉格朗日力学则可以同时涵盖两者，但现有的学习方法都是基于模拟的：它们在每次训练步骤中都需要运行数值求解器，这使得训练成本高昂。我们提出Double-Stitch，这是一种免模拟的方法，通过沿着学习到的种群路径惩罚运动方程的残差来学习这些力学。我们从不需要梯度速度的Clebsch变分原理推导出该方程，并证明当方程成立时残差恰好为零。我们在合成数据集、单细胞数据集和海洋涡流数据集上测试了Double-Stitch。

    arXiv:2610.03679v1 Announce Type: new  Abstract: The dynamics of cells, organisms, and fluids are often modeled as probability distributions evolving over time. Reconstructing and extrapolating this evolution from unpaired snapshots requires assumptions about the underlying process. Wasserstein gradient flows are a common choice, but they cannot describe conservative or periodic dynamics. Lagrangian mechanics in Wasserstein space covers both, but existing methods for learning it are simulation-based: they run a numerical solver at every training step, which makes training expensive. We propose Double-Stitch, a simulation-free method that learns these mechanics by penalizing the residual of the equation of motion along a learned population path. We derive this equation from a Clebsch variational principle that does not require gradient velocities, and show that the residual vanishes exactly when the equation holds. We test Double-Stitch on synthetic, single-cell and ocean vortex dataset
    
[^2]: 规划以学习

    Planning to Learn

    [https://arxiv.org/abs/2610.03667](https://arxiv.org/abs/2610.03667)

    该论文揭示了精确策略梯度输给交叉熵的根本原因在于其短视性——只看重即时收益而忽视每次更新对未来学习的奠基作用，并将交叉熵重新诠释为“耐心的准确率”（考虑未来学习收益的长时域误差总量），从而通过在剩余学习量处截断该总量来改进优化方法。

    

    策略梯度方法是现代强化学习的核心，也广泛应用于大语言模型的后训练。当它们表现不佳时，人们通常将其归咎于探索、信用分配和动作采样噪声。而分类问题则完全不存在这些问题。分类器可以看作一个策略，其期望奖励（即期望准确率）就是它分配给正确标签的概率；由于正确标签是已知的，策略梯度是精确且平滑的。然而，即使以期望准确率来衡量，精确策略梯度仍然输给交叉熵。其原因在于精确梯度是短视的：它仅根据更新当前所能带来的收益来评估其价值，但每一次更新同时也决定了下一次更新的起点，因此一次更新的真正价值取决于还剩下多少学习空间。从这个视角来看，交叉熵是一种“耐心的准确率”——即若一个样本的对数几率以单位速度永久增长，它所对应支付的总误差；而精确策略梯度则是这一总量的零时域极限。将这一总量在剩余学习量处截断，便得到了……（原文摘要在此处截断）

    arXiv:2610.03667v1 Announce Type: new  Abstract: Policy-gradient methods are central to modern reinforcement learning, including LLM post-training. When they struggle, the usual suspects are exploration, credit assignment and action-sampling noise. Classification has none of them. A classifier is a policy whose expected reward, its \emph{expected accuracy}, is the probability it assigns to the correct label, and because that label is known, the policy gradient is exact and smooth. Yet exact policy gradient loses to cross-entropy, even on expected accuracy. The exact gradient is myopic: it values an update only by what it buys now, but each update also sets where the next one starts, so an update's value depends on how much learning remains. Viewed this way, cross-entropy is patient accuracy, the total error an example would pay if its log-odds rose at unit speed forever, while exact policy gradient is the zero-horizon limit. Truncating this total at the learning that remains yields the
    
[^3]: 面向高斯过程潜变量模型的摊销结构化随机变分推断

    Amortized Structured Stochastic Variational Inference for Gaussian Process Latent Variable Models

    [https://arxiv.org/abs/2610.03647](https://arxiv.org/abs/2610.03647)

    本文将摊销结构化随机变分推断应用于高斯过程潜变量模型，通过让潜空间的变分后验条件依赖于诱导点的取值，突破了平均场变分近似的局限，从而在数据流形重建的多项指标上取得了改进。

    

    许多机器学习方法旨在逼近数据所在的低维流形。这类方法的一个理想特性是能够捕捉所学习流形的认知不确定性。高斯过程潜变量模型就是实现这一目标的模型之一，其中从潜空间出发的高斯过程（GP）映射提供了对流形不确定性的估计。然而，这种不确定性估计的有效性受到GP诱导点与潜变量之间平均场变分近似的限制。在本工作中，我们应用摊销结构化随机变分推断，使潜空间的变分后验可以有条件地依赖于诱导点的取值。我们证明了这种更灵活的变分后验在与数据流形上点的重建相关的多项指标上均有所提升。

    arXiv:2610.03647v1 Announce Type: cross  Abstract: Many machine learning methods aim to approximate the lower-dimensional manifold on which the data lives. A desirable feature of such methods is that they should capture the epistemic uncertainty of this learned manifold. One model that achieves this is the Gaussian Process Latent Variable Model, in which a Gaussian Process (GP) mapping from the latent space provides an estimate of the uncertainty of the manifold. However, the effectiveness of this uncertainty estimation is limited by the mean-field variational approximation between the GP inducing points and the latent variables. In this work, we apply Amortized Structured Stochastic Variational Inference to allow the variational posterior for the latent space to be conditionally dependent on the value of the inducing points. We demonstrate that this more flexible variational posterior improves several metrics relating to the reconstruction of points on the data manifold.
    
[^4]: 老虎机何时可以离开它的锚点？非平稳性下由E-过程授权的汤普森采样

    When May a Bandit Leave Its Anchor? E-Process-Authorized Thompson Sampling under Non-stationarity

    [https://arxiv.org/abs/2610.03646](https://arxiv.org/abs/2610.03646)

    提出e-过程授权的汤普森采样（e-ATS），通过随时有效的e-过程决定非平稳环境中每只手臂何时从全历史锚点切换到折扣遗忘状态，保证平稳条件下偏离乐观汤普森采样的概率不超过设定的 $\alpha_E$，实验表明证据控制的是适应何时开始而非其是否总有益。

    

    平稳性会奖励记忆，但在发生变化之后，同样的历史可能会产生误导。我们探讨何时应当允许遗忘。E-过程授权的汤普森采样（e-ATS）为每只手臂同时维护全历史与折扣的Beta状态。一个随时有效的e-过程首先对折扣状态进行授权，随后一个可逆的相关性分数控制其影响程度。在获得授权之前，e-ATS完全遵循乐观汤普森采样（OTS）。在Beta-伯努利先验预测的平稳模型下，e-ATS偏离OTS的概率至多为所设定的 $\alpha_E$，无需拟合任何阈值。相对于e-ATS，移除授权机制使注册实验套件上的平均归一化动态伪遗憾增加了38.4%，但在文献衍生的重放套件上却降低了7.5%。因此，证据控制的是适应何时开始，而不是它是否总是有益的。

    arXiv:2610.03646v1 Announce Type: new  Abstract: Stationarity rewards memory, but after a change the same history can mislead. We ask when forgetting should be permitted. E-process-authorized Thompson sampling (e-ATS) gives each arm full-history and discounted Beta states. An anytime-valid e-process first authorizes the discounted state, then a reversible relevance score controls its influence. Before authorization, e-ATS exactly follows optimistic Thompson sampling (OTS). Under a Beta-Bernoulli prior-predictive stationary model, e-ATS's probability of ever departing from OTS is at most the chosen $\alpha_E$, without fitted thresholds. Relative to e-ATS, removing authorization increased mean normalized dynamic pseudo-regret by $38.4\%$ on the registered suite but reduced it by $7.5\%$ on the literature-derived replay suite. Therefore, evidence controls when adaptation begins, not whether it always helps.
    
[^5]: 欠完备线性自编码器中被破坏的尺度对称性

    Broken scale symmetries in undercomplete linear autoencoders

    [https://arxiv.org/abs/2610.03640](https://arxiv.org/abs/2610.03640)

    该论文发现，在欠完备线性自编码器中，有限步长的SGD会以有向的方式破坏尺度对称性，在PCA解流形上系统性地偏向放大解码器权重，直至动力学触及有限步长的稳定性边界。

    

    神经网络的损失景观具有许多对称性，这些对称性在梯度流下得以保持，但在有限步长的随机梯度下降（SGD）中会被破坏。这类对称性的一个典型例子是同质网络中的尺度对称性：可以将某一层的参数放大，同时将下一层的参数缩小，而不会改变网络的输出。先前的工作记录了SGD为平衡梯度噪声或最小化波动而破坏这种对称性的若干案例。在本研究中，我们证明欠完备线性自编码器的解几何结构反而为尺度漂移选择了一个优先的方向：在PCA解流形上，SGD偏向较大的解码器权重。这种有向的尺度漂移发生在缓慢的时间尺度上，其动力学可以用解析可处理的有效描述来刻画。然而，这种漂移无法无限持续下去：尺度的不断增大最终会将动力学驱向有限步长的稳定性边界。由此产生的解是最尖…（摘要在此处被截断）

    arXiv:2610.03640v1 Announce Type: new  Abstract: Neural network loss landscapes have many symmetries, which are preserved by gradient flow but broken by finite-stepsize stochastic gradient descent (SGD). A canonical example of such a symmetry is scale in homogeneous networks: one can scale up the parameters in one layer and down in the next without changing the network output. Previous work has documented cases in which SGD breaks this symmetry in favor of balancing gradient noise or minimizing fluctuations. Here, we show that the solution geometry of undercomplete linear autoencoders instead selects a preferred sign for scale drift: on the PCA solution manifold, SGD favors large decoder weights. This directed scale drift occurs on a slow timescale, and its dynamics admit an analytically-tractable effective description. However, it cannot continue indefinitely: increasing scale eventually drives the dynamics towards a finite-stepsize stability boundary. The resulting solutions are shar
    
[^6]: 在多大的训练规模以下，深度表格数据生成器不再胜过简单基线？一项关于临床和标准数据集规模阶梯的预注册基准测试

    Below what training size do deep tabular generators stop beating trivial baselines? A preregistered benchmark on a size ladder of clinical and standard datasets

    [https://arxiv.org/abs/2610.03500](https://arxiv.org/abs/2610.03500)

    这项预注册的规模阶梯基准测试（共2,220次运行）发现，在临床等小型数据集上，几乎所有深度表格生成模型（如CTGAN、TVAE、TabDDPM）在任何测试的训练规模下都无法以超过随机噪声的幅度击败简单基线方法，挑战了深度生成模型在小型表格数据上的价值假设。

    

    深度表格生成模型通常在拥有数万行数据的数据集上进行基准测试；而临床数据集往往只有数百行。我们预注册并运行了一项规模阶梯基准测试，以找出这两种情形的分界点：将8个公开数据集从200到20,000个训练行进行子采样，使用七个生成器（独立边缘分布、高斯Copula、SMOTE、无条件SMOTE、CTGAN、TVAE、TabDDPM），采用固定的20次试验调优预算和5个评估种子，外加4个真实规模的原生小型临床数据集，总计进行了2,220次已承诺的运行。主要评估指标是在合成数据上训练、在真实数据上测试的固定分类器的AUROC。在我们测量的所有训练规模下，24个（数据集，深度模型）配对中有23个，没有任何深度模型能以超过种子噪声的幅度击败最佳简单基线。在49个（数据集，规模）组合中，最佳基线在40个中获胜。我们预注册的预测——即深度模型的排名在小规模数据下会不稳定——被证伪：平均Kend……（摘要在此处截断）

    arXiv:2610.03500v1 Announce Type: new  Abstract: Deep tabular generative models are benchmarked on datasets with tens of thousands of rows; clinical datasets have hundreds. We preregistered and ran a size-ladder benchmark to find where the two regimes diverge: 8 public datasets subsampled from 200 to 20,000 training rows, seven generators (independent marginals, Gaussian copula, SMOTE, unconditional SMOTE, CTGAN, TVAE, TabDDPM) with a fixed 20-trial tuning budget and 5 evaluation seeds, plus 4 natively small clinical datasets at true size, for 2,220 committed runs in total. The primary metric is the AUROC of fixed classifiers trained on synthetic and tested on real data. In 23 of 24 (dataset, deep model) pairs no deep model ever beats the best trivial baseline by more than seed noise, at any training size we measured. The best baseline wins 40 of 49 (dataset, size) cells. Our preregistered prediction that the deep models' ranking would be unstable at small sizes is falsified: mean Kend
    
[^7]: AREX：用于流匹配少步采样的仿射-残差指数积分器

    AREX: Affine-Residual Exponential Integrator for Few-Step Sampling in Flow Matching

    [https://arxiv.org/abs/2610.03483](https://arxiv.org/abs/2610.03483)

    AREX是一种无需训练的流匹配模型少步采样器，它将采样动力学分解为由目标均值和协方差决定的仿射分量（用显式矩阵值传播子积分）与神经残差项，在无需重训练的情况下持续提升少步采样的样本保真度。

    

    我们提出了AREX，一种面向预训练流匹配模型的无需训练的采样器，它利用目标均值和协方差来捕获采样动态中可解析处理的部分。我们证明了矩匹配高斯目标的速度场是边缘速度场的 $L^2$ 最优仿射近似。这促使我们将学习到的动力学分解为覆盖整个采样路径的仿射分量（由目标的前两阶矩决定）以及一个神经残差项。AREX保留仿射分量，并使用显式矩阵值传播子对其进行积分；相应地，我们只需对残差项进行积分。这不同于标量指数积分器，后者只能解析地处理各向同性的线性动力学。在图像和文本生成图像任务中，AREX在少步采样机制下持续提升样本保真度，而无需重新训练底层模型。

    arXiv:2610.03483v1 Announce Type: cross  Abstract: We introduce AREX, a training-free sampler for pretrained flow matching models that uses the target mean and covariance to capture an analytically tractable part of the sampling dynamics. We show that the velocity field of the moment-matched Gaussian target is the $L^2$-optimal affine approximation to the marginal velocity field. This motivates decomposition of the learned dynamics into an affine component over the whole sampling path, determined by the first two target moments, and a neural residual term. AREX keeps the affine component and integrates it using an explicit matrix-valued propagator. In turn, we only require to integrate over the residual term. This differs from scalar exponential integrators, which analytically handle only isotropic linear dynamics. Across image and text-to-image generation tasks, AREX consistently improves sample fidelity in the few-step sampling regime without retraining the underlying model.
    
[^8]: 何时准确率才是证据？泛化、验证与信息融合的统一理论

    When Is Accuracy Evidence? A Unified Theory of Generalisation, Validation, and Information Fusion

    [https://arxiv.org/abs/2610.03465](https://arxiv.org/abs/2610.03465)

    该论文提出一个统一的指数框架，将交叉验证准确率转化为真实风险的保守上界，并通过有效折数Keff证明：当各折数据强相关时，单纯增大交叉验证折数K并不能增强统计证据。

    

    K折交叉验证（CV）被广泛用作样本外性能的证据，然而在异构数据下，各折既非独立实验，也非信息量均等。交叉上界验证（CUBV）用真实风险的保守上界取代逐点的CV准确率。本文通过单一的指数框架对CUBV进行了推广，其中泛化间隙的矩生成函数由累积量包络gamma(lambda)所控制。由此得到一族风险界，涵盖Hoeffding界、Bernstein界、依赖感知、PAC-Bayesian以及异构数据源融合等多种情形。对于K折CV，各折间隙之间的相关性通过联合次高斯代理矩阵来建模。在等相关条件下，得到有效折数Keff = K/[1+(K-1)rho]，表明当各折之间存在强相关性时，增大K并不一定能增加统计证据。该框架还被用于……（原文摘要在此处截断）

    arXiv:2610.03465v1 Announce Type: cross  Abstract: K-fold cross-validation (CV) is widely used as evidence of out-of-sample performance, although folds are neither independent experiments nor equally informative under heterogeneous data. Cross Upper-Bound Validation (CUBV) replaces point-wise CV accuracy by conservative upper bounds on true risk. Here we generalise CUBV through a single exponential framework in which the moment-generating function of the generalisation gap is controlled by a cumulant envelope gamma(lambda). This yields a family of risk bounds covering Hoeffding-, Bernstein-, dependency-aware, PAC-Bayesian, and heterogeneous source-fusion settings. For K-fold CV, dependence between fold-wise gaps is modelled through a joint sub-Gaussian proxy matrix. Under equicorrelation, this gives an effective number of folds, Keff = K/[1+(K-1)rho], showing that increasing K does not necessarily increase statistical evidence when folds are strongly dependent. The framework is also ex
    
[^9]: 迭代一致性模型：稳定性、误差界与噪声调度

    Iterating Consistency Models: Stability, Error Bounds and Noise Schedules

    [https://arxiv.org/abs/2610.03414](https://arxiv.org/abs/2610.03414)

    该论文将多步一致性模型采样分析为加噪与近似去噪算子的复合，在可验证的稳定性假设下推导出非渐近误差界，揭示了噪声调度中早期大噪声驱动误差收缩、后期小噪声控制残余偏差的作用机制，为CM采样器设计提供了理论指导。

    

    一致性模型（Consistency Models, CMs）已成为用少量步骤生成高质量样本的主流方法。然而，增加采样步骤既可能提升也可能降低样本质量，且这种现象对噪声调度高度敏感，而现有理论无法完全解释。为了提供精度保证并指导CM采样器设计，我们将多步CM采样分析为加噪算子与近似去噪算子的复合。在显式且可验证的稳定性假设下，我们推导出一个非渐近误差界，该误差界将初始化误差的收缩与近似误差的累积分离开来。该误差界为噪声调度分配了不同的角色：较大的早期噪声水平驱动误差收缩，而较小的后期噪声水平控制残余偏差。作为推论，我们为强对数凹和半对数凹目标分布得到了显式常数。我们进一步建立了一个互补的理论保证，其假设条件、一步精度和……（摘要在此处被截断）

    arXiv:2610.03414v1 Announce Type: cross  Abstract: Consistency models (CMs) have become a leading approach for generating high-quality samples in few steps. However, adding steps can improve or degrade sample quality in ways that are highly sensitive to the schedule and that existing theory does not fully explain. To provide accuracy guarantees and guide CM sampler design, we analyze multistep CM sampling as a composition of noising and approximate denoising operators. Under explicit, verifiable stability assumptions, we derive a non-asymptotic error bound that separates contraction of the initialization error from accumulation of approximation error. The bound assigns distinct roles to the schedule: large early noise levels drive contraction, while small late noise levels control the residual bias. As a corollary, we obtain explicit constants for strongly log-concave and semi-log-concave targets. We further establish a complementary guarantee whose assumptions, one-step accuracy and s
    
[^10]: DAWIS：基于多任务插值子的窗口化逆采样数据同化

    DAWIS: Data Assimilation with Windowed Inverse Sampling via Multitask Interpolants

    [https://arxiv.org/abs/2610.03314](https://arxiv.org/abs/2610.03314)

    DAWIS提出了一种统一的数据同化框架，通过用覆盖连续状态窗口的多任务随机插值子替代单一流时间先验，使滤波、固定滞后平滑和分块平滑得以在同一框架内实现，从而能在新观测到达时修正过去状态并避免误差累积。

    

    基于流和扩散的生成模型最近已成为动力系统中灵活且高效的预测模型。当与推理时引导相结合时，它们为高维非高斯数据同化（DA）提供了一条有前景的途径——数据同化是将预测与观测相结合以估计潜在系统状态的问题。然而，现有的滤波器以固定的历史为条件，仅同化最新的观测结果，因此无法在新观测到达时修正过去的状态。这导致估计结果被束缚在与后续观测可能相矛盾的历史上，误差在同化运行过程中不断累积。为此，我们提出了DAWIS，这是一种统一的数据同化方法，在单一框架内涵盖了滤波、固定滞后平滑和分块平滑。DAWIS用跨越连续状态窗口的多任务随机插值子替代状态级先验的单一流时间……（摘要未完）

    arXiv:2610.03314v1 Announce Type: cross  Abstract: Flow- and diffusion-based generative models have recently emerged as flexible and highly efficient forecasting models for dynamical systems. When combined with inference-time guidance, they offer a promising route to high-dimensional non-Gaussian data assimilation (DA), the problem of combining forecasts with observations to estimate latent system states. Existing filters, however, condition on a fixed history and assimilate only the most recent observation, leaving them unable to revise past states when new observations arrive. Estimates then stay tethered to a history that later observations may contradict, and errors accumulate over the assimilation run. To this end, we introduce **DAWIS**, a unified DA method covering filtering, fixed-lag smoothing, and block smoothing within a single framework. DAWIS replaces the single flow time of a state-level prior with a multitask stochastic interpolant over a window of consecutive states, as
    
[^11]: SDECast：基于神经随机微分方程的连续时间概率天气预报

    SDECast: Probabilistic Weather Forecasting in Continuous Time with Neural SDEs

    [https://arxiv.org/abs/2610.03313](https://arxiv.org/abs/2610.03313)

    SDECast提出了一个基于神经随机微分方程的连续时间概率天气预报框架，无需训练时重复SDE模拟即可直接在物理空间学习随机动力学，并成功扩展到小时级分辨率的全球天气预报。

    

    现有的机器学习天气预报模型通常通过固定时间分辨率的自回归滚动方式生成预报。虽然这种方式对于长期预测非常高效，但在使用较短时间步长时可能出现严重的误差累积，并且没有显式地编码大气动力学的局部性与时间连续性。为了解决这些局限，我们提出了SDECast，一个用于连续时间概率天气预报的神经随机微分方程（Neural SDE）框架。SDECast扩展了SDE Matching方法，直接在物理空间中学习随机动力学，而无需在训练过程中进行重复的SDE模拟。在一个模拟的地球物理流动实验中，我们展示了SDECast能够恢复有意义的漂移动力学，并忠实地再现了底层的连续时间行为。随后，我们证明了该方法在小时分辨率全球天气预报上的可扩展性。

    arXiv:2610.03313v1 Announce Type: cross  Abstract: Existing machine learning weather forecasting models typically generate forecasts through autoregressive rollouts at a fixed temporal resolution. While highly efficient for long-range prediction, this formulation can suffer from severe error accumulation when used with shorter time steps and does not explicitly encode the locality and temporal continuity of atmospheric dynamics. To address these limitations, we introduce **SDECast**, a Neural Stochastic Differential Equation (SDE) framework for continuous-time probabilistic weather forecasting. SDECast extends SDE Matching to learn stochastic dynamics directly in physical space, without requiring repeated SDE simulation during training. On a simulated geophysical flow, we show that SDECast recovers meaningful drift dynamics and faithfully reproduces the underlying continuous-time behavior. We then demonstrate its scalability to global weather forecasting at hourly resolution, where SDE
    
[^12]: 基于惰性二阶预言机的近最优凸优化

    Near-Optimal Convex Optimization with Lazy Second-Order Oracles

    [https://arxiv.org/abs/2610.03222](https://arxiv.org/abs/2610.03222)

    本文针对惰性二阶预言机凸优化问题，通过新的分块零链下界构造和匹配的算法设计，将复杂度界改进至 $\tilde{\mathcal{O}}(m+ m^{1/7} \epsilon^{-2/7})$ 并在对数因子内紧致，显著优于先前结果。

    

    本文研究了使用惰性二阶预言机（Doikov、Chayti 和 Jaggi，ICML 2023）的凸优化复杂度问题，其中算法每次迭代查询梯度，每 $m$ 次迭代查询一次 Hessian 矩阵。在该设置下，我们通过一种新颖的分块零链构造，证明了找到 $\epsilon$-解所需总迭代次数的下界为 $\Omega(m+ m^{1/7} \epsilon^{-2/7})$。随后，我们提出了一种新方法，达到了新的上界 $\tilde{\mathcal{O}}(m+ m^{1/7} \epsilon^{-2/7})$，该结果显著改进了先前（Chen、Liu、Luo 和 Zhang，COLT 2026）的 $\tilde{\mathcal{O}}(m+ m^{13/21} \epsilon^{-2/7})$ 上界，并在仅相差对数因子的意义下是紧致的。

    arXiv:2610.03222v1 Announce Type: cross  Abstract: This paper studies the complexity of convex optimization using lazy second-order oracles (Doikov, Chayti, and Jaggi, ICML 2023), where an algorithm queries gradients every iteration and Hessians once per $m$ iterations. Under this setting, we show a lower bound of $\Omega(m+ m^{1/7} \epsilon^{-2/7})$ on the number of total iterations to find an $\epsilon$-solution using a novel block zero-chain construction. Then we propose a novel method that achieves a new upper bound of $\tilde{\mathcal{O}}(m+ m^{1/7} \epsilon^{-2/7})$, which significantly improves the prior one (Chen, Liu, Luo, and Zhang, COLT 2026) of $\tilde{\mathcal{O}}(m+ m^{13/21} \epsilon^{-2/7})$ and is tight up to logarithmic factors.
    
[^13]: 预测导向的高斯过程后验

    Predictively Oriented Gaussian Process Posteriors

    [https://arxiv.org/abs/2610.03201](https://arxiv.org/abs/2610.03201)

    提出预测导向高斯过程（PrO-GPs），将预测不确定性作为主要推断目标，在模型错误设定下比标准高斯过程产生校准更好的预测分布。

    

    高斯过程（GP）是建模和量化函数关系中不确定性的强大工具。然而，它们要求使用者做出许多设计决策，例如核函数和观测模型的选择。次优的选择可能产生错误设定的模型，无法捕捉潜在的数据生成过程。我们提出了预测导向高斯过程（PrO-GPs），它将预测不确定性作为主要的推断目标，为标准高斯过程提供了一种稳健的替代方案。尽管对非参数模型直接计算PrO后验是不可行的，我们推导了一种简化形式和实用的采样方案以实现高效计算。通过合成数据和真实数据的实验，我们表明与标准高斯过程方法相比，PrO-GPs在模型错误设定的情况下能够产生更好校准的预测分布。

    arXiv:2610.03201v1 Announce Type: cross  Abstract: Gaussian Processes (GPs) are a powerful tool for modelling and quantifying uncertainty in functional relationships. However, they require practitioners to make a number of design decisions, such as the choice of the kernel and the observation model. Suboptimal choices can produce misspecified models that do not capture the underlying data generating process. We introduce Predictively Oriented Gaussian Processes (PrO-GPs), which treat predictive uncertainty as the primary inferential target and provide a robust alternative to standard GPs. Although direct computation of a PrO posterior for nonparametric models is intractable, we derive a reduced formulation and practical sampling scheme for efficient computation. Through synthetic and real data experiments, we show that PrO-GPs produce better calibrated predictive distributions under model misspecification compared to standard GP approaches.
    
[^14]: 因果效应识别中聚类操作的不变性

    Invariance of Clustering Operations in Causal Effect Identification

    [https://arxiv.org/abs/2610.03101](https://arxiv.org/abs/2610.03101)

    本文提出了一类基于原图c-分量条件的“识别不变”聚类操作，可同时保持因果效应的可识别性与不可识别性，从而安全地简化因果图并加速因果效应识别。

    

    在因果图中对变量进行聚类可以减小图的规模并简化因果推断。然而，任意的聚类可能会改变变量之间关键的因果关系，从而导致错误的结论。虽然在温和条件下，聚类图中因果效应的可识别性意味着原图中的可识别性，但在缺乏进一步假设的情况下，聚类图中的不可识别性并不能推出原图中的不可识别性。当可识别性与不可识别性均被保持时，该聚类操作被称为“识别不变的”。我们基于与原图c-分量相关的条件，提出了一大类识别不变的聚类操作。最后，我们展示了这些结果在实际场景中的应用。

    arXiv:2610.03101v1 Announce Type: cross  Abstract: Clustering variables in causal graphs reduces the size of the graph and simplifies causal inference. However, arbitrary clustering can alter crucial causal relations among variables and lead to erroneous conclusions. While the identifiability of a causal effect in the clustered graph implies the identifiability in the original graph under mild conditions, nonidentifiability in clustered graph does not imply nonidentifiability in the original graph without further assumptions. When both identifiability and nonidentifiability are preserved, the clustering operation is called identification invariant. We present a broad class of clustering operations that are identification invariant based on conditions related to the c-components of the original graph. Finally, we demonstrate use of the results in practical settings.
    
[^15]: GTDD：面向AI编码智能体的生成式测试驱动开发与对抗测试

    GTDD: Generative Test-Driven Development for AI Coding Agents with Adversarial Testing

    [https://arxiv.org/abs/2610.02952](https://arxiv.org/abs/2610.02952)

    提出生成式测试驱动开发（GTDD），由独立的测试智能体在每轮候选实现后基于人类指定的行为契约对抗性地生成新测试输入并不断反馈反例，以解决AI编码智能体过拟合固定测试集而导致预期行为遗漏的问题。

    

    测试驱动开发为AI编码智能体提供了实现软件的可执行需求。由于这些智能体能够使其实现适应所观察到的示例，通过一组预先确定的测试可能会导致预期行为的相当大一部分未被实现。我们提出了生成式测试驱动开发（GTDD），这是测试驱动开发的一种形式化表述，其中在每次候选实现被固定后，一个独立的测试智能体会根据人类指定的行为契约和之前轮次的反馈生成新的输入。一个可信的评估器会检查这些输入，将简化的反例返回给编码智能体，并将其保存用于回归测试，从而使开发过程持续面对超出初始示例的失败情况。我们通过对自适应候选选择下错误接受的有限总体分析，刻画了该过程所提供的证据。由此得到的界……

    arXiv:2610.02952v1 Announce Type: cross  Abstract: Test-driven development gives AI coding agents executable requirements for implementing software. Because these agents can adapt their implementations to the examples they observe, passing a predetermined collection of tests can leave substantial parts of the intended behavior unimplemented. We propose Generative Test-Driven Development (GTDD), a formulation of test-driven development in which a separate testing agent generates new inputs after each candidate implementation is fixed, using a human-specified behavioral contract and the feedback from earlier rounds. A trusted evaluator checks these inputs, returns reduced counterexamples to the coding agent, and saves them for regression testing, so development continually confronts failures beyond the initial examples. We characterize the evidence that this process provides through a finite-population analysis of false acceptance under adaptive candidate selection. The resulting bounds 
    
[^16]: 非正则条件下的交叉拟合：基于局部性的正态性与统计推断

    Cross-Fitting Under Nonregularity: Normality and Inference via Locality

    [https://arxiv.org/abs/2610.02944](https://arxiv.org/abs/2610.02944)

    本文证明在非正则条件下，一大类交叉拟合估计量仍满足中心极限定理，但需针对交叉折相关性修正渐近方差，并据此提出了可达到渐近名义覆盖率的新置信区间构造方法。

    

    交叉拟合在大量应用研究中已是常规操作。虽然忽略交叉折之间依赖性的传统置信区间在若干情境下是渐近有效的，但在许多具有共同非正则形式的应用中，这些置信区间的实际覆盖率不足：从检验一个拟合模型是否优于另一个模型的经典交叉验证问题，到利用机器学习检验异质性处理效应，再到估计可能非唯一的最优治疗方案的价值。利用一个新的局部性条件，本文证明了一大类交叉拟合估计量尽管存在非正则性，仍然满足中心极限定理，但其渐近方差必须针对交叉折之间的相关性进行修正。随后，本文提出了一种估计这种相关性的方法，并构造了能够达到渐近名义覆盖率的新置信区间。最后，本文表明所提出的置信区间在……方面达到近似（摘要原文在此处被截断）。

    arXiv:2610.02944v1 Announce Type: new  Abstract: Cross-fitting is routine in much of applied research. While conventional confidence intervals that ignore cross-fold dependence are asymptotically valid in several settings, they undercover in many applications that share a common form of nonregularity: from the classic cross-validation problem of testing whether a fitted model outperforms another, to testing for heterogeneous treatment effects with machine learning, to estimating the value of a potentially non-unique optimal treatment regime. Exploiting a new locality condition, I show that a large class of cross-fitting estimators still satisfies a central limit theorem despite the nonregularity, but with an asymptotic variance that must be adjusted for the cross-fold correlation. Then, I propose a method for estimating this correlation and construct new confidence intervals that attain asymptotically nominal coverage. Finally, I show that the proposed confidence intervals attain appro
    
[^17]: 面向高维数据的残差树高斯过程建模框架

    A Residual Tree Gaussian Process Modeling Framework for High-Dimensional Data

    [https://arxiv.org/abs/2610.02893](https://arxiv.org/abs/2610.02893)

    该论文提出 ResTGP，一种贝叶斯残差树高斯过程框架，通过沿二叉树在多分辨率层级上迭代分解预测过程与残差过程，实现了对高维异质大空间数据的灵活多尺度协方差建模与分而治之的高效计算。

    

    随着测量技术的进步和计算能力的不断提升，具有异质结构的大型空间数据常常在高维域上被收集。现有的高斯过程（GP）模型和计算策略往往不足以分析多维域上的此类数据集。为了应对这些挑战，我们提出了一种名为 ResTGP 的贝叶斯残差树高斯过程方法，用于分析多维域上可能具有异质结构的大型空间数据。其核心思想是通过迭代计算预测过程和残差过程，沿着一棵二叉树在级联的分辨率层级上对高斯过程进行分解，使得每个树节点（无论是内部节点还是叶节点）上的残差过程对于刻画该节点内更精细层级的依赖关系都是充分的。这使得模型能够以灵活的多尺度方式刻画潜在的协方差结构，同时实现分而治之的（计算策略）……

    arXiv:2610.02893v1 Announce Type: cross  Abstract: With the advance of measurement technologies and increasing computing power, large spatial data with heterogeneous structures are often collected over high-dimensional domains. Existing Gaussian process (GP) models and computational strategies are often inadequate for analyzing such datasets in multi-dimensional domains. To address these challenges, we develop a Bayesian residual tree GP methodology called ResTGP for large spatial data with potentially heterogeneous structures in multi-dimensional domains. The key idea is to decompose a Gaussian process at a cascade of resolutions along a dyadic tree through iteratively computing predictive and residual processes so that the residual process on each tree node, both interior and leaf, becomes sufficient for the finer-level dependency within that node. This allows characterization of the underlying covariance structure in a flexible, multi-scale manner while achieving divide-and-conquer 
    
[^18]: Muon 更擅长学习事实：理解谱正交化的作用

    Muon Learns Facts Better: Understanding the Role of Spectral Orthogonalization

    [https://arxiv.org/abs/2610.02798](https://arxiv.org/abs/2610.02798)

    本文通过可解析的事实回忆模型和线性 Transformer 分析了 Muon 优化器中谱正交化的作用机制，揭示其对特征学习动力学的改变，并说明这使 Muon 比梯度下降和 Adam 更擅长学习“主体-关系到答案”的事实映射。

    

    arXiv:2610.02798v1 通告类型：新 摘要：Muon 优化器对矩阵形式的更新应用谱正交化，并已在大规模神经网络训练中展现出优异的性能，然而这一变换在特征学习中的作用机制仍鲜为人知。在本工作中，我们通过一个可解析的事实回忆模型来研究这一问题：其中每条事实将每个“主体-关系”对映射到一个答案，而一个线性 Transformer 学习恢复该映射所需的、依赖于主体和依赖于关系的信息。该 Transformer 分别采用梯度流（GF）、谱 GF 或符号 GF 进行优化，它们分别是梯度下降、Muon 和 Adam 的连续时间极限。先前的研究（Nichani et al., 2025）已表明，当主体数量超过关系数量时，GF 会先学习依赖于关系的信息，后学习依赖于主体的信息，从而在训练过程中产生一个特征分离阶段。我们对该分离阶段进行了刻画……（原文摘要在此处截断）

    arXiv:2610.02798v1 Announce Type: new  Abstract: The Muon optimizer applies spectral orthogonalization to matrix-valued updates and has shown strong performance in large-scale neural network training, yet the mechanisms of this transformation in feature learning remain poorly understood. In this work, we investigate this question through a tractable factual-recall model, where a fact maps each subject-relation pair to an answer, and a linear transformer learns the subject- and relation-dependent information required to recover this mapping. The transformer is optimized with gradient flow (GF), spectral GF, or Sign GF, which are continuous-time limits of gradient descent, Muon, and Adam, respectively. Prior studies (Nichani et al., 2025) have shown that when the number of subjects exceeds the number of relations, GF learns relation-dependent information before subject-dependent information, producing a feature-separation phase during training. We characterize this separation with the le
    
[^19]: 用于高效高斯DAG学习的留出评分法

    Hold-Out Scoring for Efficient Gaussian DAG Learning

    [https://arxiv.org/abs/2610.02785](https://arxiv.org/abs/2610.02785)

    该论文提出HOST算法，以逐节点留出评分与凸回归取代子集搜索，仅凭对得分误差的单侧控制即可恢复正确的节点排序，从而在无需入度上界的情况下实现高效的高斯DAG学习。

    

    高维高斯DAG（有向无环图）学习面临着统计与计算之间的鸿沟：具有精细样本复杂度的方法依赖于计算代价高昂的子集搜索以及需要预先给定的入度上界，而多项式时间的替代方法则具有较差的样本复杂度。我们提出了HOST，一种高效的DAG学习算法，它用逐节点的留出评分和凸回归取代子集搜索，且无需预先给定入度上界。我们的关键洞察是：恢复正确的节点排序并不需要在排序得分上具有一致小的估计误差，而只需要对这些误差实施单侧控制。在排序步骤中，HOST利用了如下事实：使用留出样本进行的得分估计在期望意义上会抬高排序得分，这对于那些尚不应被选中的候选节点而言恰好是有利的误差方向。在给定排序之后，HOST通过递归地从两个节点之间的总效应中剔除间接效应来恢复父节点（摘要在此处截断）。

    arXiv:2610.02785v1 Announce Type: cross  Abstract: High-dimensional Gaussian DAG learning faces a statistical-computational gap: methods with sharp sample complexity rely on computationally expensive subset search and a supplied indegree bound, whereas polynomial-time alternatives have less favorable sample complexity. We introduce HOST, an efficient DAG learning algorithm that replaces subset search with nodewise hold-out scoring and convex regression, without requiring a supplied indegree bound. Our key insight is that recovering a correct ordering does not require uniformly small estimation errors in ordering scores but only one-sided control of those errors. In the ordering step, HOST exploits the fact that score estimation using hold-out samples inflates ordering scores in expectation, which is the favorable direction for candidates that should not yet be selected. Given the ordering, HOST recovers parents by recursively removing indirect effects from total effects between two nod
    
[^20]: 具有1比特反馈的近最优固定置信度最优臂识别

    Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback

    [https://arxiv.org/abs/2610.02771](https://arxiv.org/abs/2610.02771)

    本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。

    

    我们研究在严格1比特反馈约束下的固定置信度最优臂识别问题。在每一轮中，学习者选择一个臂和一个查询集合，并且仅接收一个比特，该比特指示采样的奖励是否属于该集合。我们考虑一种具有逐臂定位的无分布有限方差设置，在这种设置下，直接的经验均值估计不再可用，截断处理变得不可避免。我们首先基于随机化阈值查询和截断尾积分恒等式，构建了一个时间一致的1比特均值估计基元。随后，我们将该基元嵌入到候选-挑战者式的最优臂识别算法中。固定截断算法提供了简单的任意时刻（ε,δ)-PAC保证，而分阶段自适应截断算法则将截断水平与当前分辨率相匹配，从而产生了间隙自适应的样本复杂度。我们还证明了一个K臂最坏情况的信息论下界（摘要原文在此处截断）。

    arXiv:2610.02771v1 Announce Type: cross  Abstract: We study fixed-confidence best-arm identification under strict 1-bit feedback constraints. At each round, the learner selects an arm and a query set, and receives only a single bit indicating whether the sampled reward belongs to that set. We consider a distribution-free finite-variance setting with arm-wise localization, where direct empirical mean estimation is no longer available and clipping becomes unavoidable. We first formulate a time-uniform 1-bit mean-estimation primitive based on randomized threshold queries and a clipped tail-integral identity. We then embed this primitive into candidate-challenger best-arm identification algorithms. A fixed-clipping algorithm gives a simple anytime $(\epsilon,\delta)$-PAC guarantee, while a phased adaptive-clipping algorithm matches the clipping level to the current resolution and yields a gap-adaptive sample complexity. We also prove a $K$-arm worst-case information-theoretic lower bound s
    
[^21]: 扰动目标上梯度下降的差分隐私

    Differential Privacy of Gradient Descent on Perturbed Objectives

    [https://arxiv.org/abs/2610.02716](https://arxiv.org/abs/2610.02716)

    该论文证明了在强凸光滑目标上，对扰动目标运行梯度下降的有限次迭代是高斯噪声的 $C^1$ 微分同胚（并给出雅可比最小奇异值的定量下界），从而可直接用换元法分析有限迭代的差分隐私，且对广义线性模型而言，迭代条件成立时隐私界不显式依赖环境维度。

    

    目标扰动方法在正则化经验风险上加入一个随机线性项，并精确释放扰动后的极小化点。我们研究有限计算情形下的隐私性，即释放确定性梯度下降在 $w\mapsto F(w;S)+\langle z,w\rangle$ 上的第 $N$ 次迭代，其中噪声 $z\sim\mathcal N(0,\sigma^2I_d)$ 是在优化开始前一次性抽取的。对于具有 Lipschitz Hessian 的强凸光滑目标，我们证明了一个显式条件，在该条件下映射 $z\mapsto w_N$ 在隐私论证所用的有界区域上是 $C^1$ 微分同胚，并给出了其雅可比矩阵最小奇异值的定量下界。这使得可以对有限次迭代直接进行换元分析。对于广义线性模型，一旦迭代条件成立，所得到的隐私分布界不再包含显式的环境维度因子，且其有限次迭代的修正项以几何速度衰减。通过令自由截断参数……（原文摘要在此处截断）

    arXiv:2610.02716v1 Announce Type: new  Abstract: Objective perturbation adds a random linear term to a regularized empirical risk and releases the exact perturbed minimizer. We study the finite computation obtained by releasing the $N$-th iterate of deterministic gradient descent on $w\mapsto F(w;S)+\langle z,w\rangle$, where $z\sim\mathcal N(0,\sigma^2I_d)$ is drawn once before optimization. For strongly convex and smooth objectives with Lipschitz Hessian, we prove an explicit condition under which the map $z\mapsto w_N$ is a $C^1$-diffeomorphism on the bounded domains used in the privacy argument, with a quantitative lower bound on the smallest singular value of its Jacobian. This permits a direct change-of-variables analysis of the finite iterate. For generalized linear models, the resulting privacy-profile bound has no explicit ambient-dimension factor once the iteration condition holds, and its finite-iteration correction decreases geometrically. By letting the free truncation par
    
[^22]: 面向内在低维数据的得分匹配扩散模型的泛化性质

    Generalization Properties of Score-matching Diffusion Models for Intrinsically Low-dimensional Data

    [https://arxiv.org/abs/2610.02663](https://arxiv.org/abs/2610.02663)

    该论文为流匹配模型在具有内在低维结构的数据上提供了统计泛化理论保证，推导出依赖于数据内在维度的 Wasserstein-p 有限样本误差界，克服了以往分析中限制性假设和忽略低维结构的不足。

    

    尽管流匹配模型在实证应用中取得了显著成功，但其统计泛化保证的理论研究仍然不完善。现有分析通常对估计的速度场施加限制性假设，且得到的收敛速率无法反映真实数据（如自然图像和分子几何结构）中普遍存在的内在低维结构。在本工作中，我们研究了流匹配模型从有限样本中学习未知分布 P_data 的统计泛化性能。我们对学习到的生成分布，在 Wasserstein-p 距离度量下（对所有 p≥1），推导出了有限样本误差界。具体而言，给定来自 P_data 的 n 个独立同分布样本，我们证明：对于每一个 d>d_p*(P_data)，只要恰当选择网络架构和超参数，学习到的分布 P̂^FM 就满足相应的误差界。

    arXiv:2610.02663v1 Announce Type: cross  Abstract: Despite the remarkable empirical success of flow-matching models, their statistical generalization guarantees remain underdeveloped. Existing analyses often impose restrictive assumptions on the estimated velocity field and yield convergence rates that fail to reflect the intrinsic low-dimensional structure common in real data, such as natural images and molecular geometries. In this work, we study the statistical generalization of flow-matching models for learning an unknown distribution $P_{\mathrm{data}}$ from finitely many samples. We derive finite-sample error bounds on the learned generative distribution, measured in the Wasserstein-$p$ distance, for all $p\geq 1$. Specifically, given $n$ i.i.d. samples from $P_{\mathrm{data}}$, we show that, for every $d>d_p^\ast(P_{\mathrm{data}})$ and appropriately chosen network architectures and hyperparameters, the learned distribution $\widehat{P}^{\mathrm{FM}}$ satisfies $ \mathbb{W}_p(\w
    
[^23]: 高维渐近理论与隐私保护迁移学习中的数据集选择

    High-Dimensional Asymptotics and Dataset Selection for Private Transfer Learning

    [https://arxiv.org/abs/2610.02578](https://arxiv.org/abs/2610.02578)

    本文针对隐私保护迁移学习中的数据集选择问题，提出了一种仅基于汇总统计量、采用加权岭估计器的多元异构源高维回归方法，在 ρ-零集中差分隐私保证下研究其高维渐近性质，以判断额外数据是否值得购买或纳入协同学习。

    

    无论是购买外部数据还是参与协同学习，决策者都必须判断额外数据能否充分提升预测性能，从而证明其成本是合理的。这带来了几个挑战：(i) 决策通常只能依赖公开可得的聚合统计信息，而非个体层面的数据；(ii) 协变量偏移和模型偏移可能导致负迁移，使额外数据反而降低而非提升性能；(iii) 如果数据是敏感的，其隐私化处理需要注入噪声，这也可能抵消更大样本量带来的收益。本文通过具有多个异构数据源的高维回归以及加权岭估计器来建模数据集选择问题。我们的方法仅使用汇总统计信息，并能在 ρ-零集中差分隐私框架下，提供仅针对标签或同时针对特征与标签的隐私保证。

    arXiv:2610.02578v1 Announce Type: cross  Abstract: To commit to buying external data or participate in collaborative learning, one must decide whether the additional data will improve prediction enough to justify the cost. This comes with several challenges: (i) the decision often relies only on aggregated statistics available publicly, rather than individual-level data; (ii) covariate and model shifts can induce negative transfer, so the additional data deteriorates rather than improves performance; (iii) if the data is sensitive, its privatization requires the injection of noise, which can also offset the benefit of a larger sample size. In this paper, we model the problem of dataset selection through high-dimensional regression with multiple heterogeneous sources and a weighted ridge estimator. Our approach uses only summary statistics and it gives privacy guarantees either on labels only or jointly on features and labels, in terms of $\rho$-zero-concentrated differential privacy. T
    
[^24]: ENCORE：基于副本交换的扩散生成精确非平衡控制

    ENCORE: Exact Non-equilibrium COntrol with Replica Exchange for Diffusion Generation

    [https://arxiv.org/abs/2610.02538](https://arxiv.org/abs/2610.02538)

    该论文提出了首个精确的并行推理时控制方法ENCORE，通过让每个副本保存生成轨迹使向上移动成为截断操作，从而避免模拟难以处理的时间反转，实现了无偏的副本交换控制。

    

    推理时控制能够在无需重新训练的情况下，将预训练的生成模型引导至目标分布。我们研究了倾斜目标分布 $\pi_0\propto G_0\,p_0$，其中 $p_0$ 是采样器的输出分布，$G_0$ 是可评估的重加权函数。现有方法依赖于基于序贯蒙特卡罗（SMC）的序贯退火，或基于副本交换（RE）的并行退火。序贯控制是精确的，但需要庞大的粒子群体；然而目前尚不存在精确的并行控制方法：现有的 RE 校正方法近似模拟了一个难以处理的时间反转过程，因而存在偏差。我们提出了基于副本交换的精确非平衡控制方法（ENCORE），这是首个精确的并行控制方法。每个副本保存其生成轨迹，因此向上移动仅是一种截断操作，从而无需模拟难以处理的时间反转过程。我们证明了目标分布的不变性，并表明所得动力学即为非平衡副本交换的动力学（摘要在此处截断）

    arXiv:2610.02538v1 Announce Type: cross  Abstract: Inference-time control steers a pretrained generative model towards a target distribution without retraining. We study tilted targets $\pi_0\propto G_0\,p_0$, where $p_0$ is the sampler output distribution and $G_0$ is an evaluable reweighting function. Existing approaches rely on sequential annealing with sequential Monte Carlo (SMC) or parallel annealing with replica exchange (RE). Sequential control is exact but needs large particle populations, whereas no exact parallel control method exists: existing RE corrections approximate an intractable time reversal and are biased. We propose Exact Non-equilibrium COntrol with Replica Exchange (ENCORE), the first exact parallel control method. Each replica stores its generation trajectory, so the upward move is a truncation and the intractable time reversal is never simulated. We prove target invariance and show that the resulting dynamics are those of non-equilibrium replica exchange with t
    
[^25]: 学习风格，遗忘语义：SFT与RFT在分类任务上的案例研究

    Learning Style, Forgetting Semantics: A Case Study of SFT and RFT on Classification Tasks

    [https://arxiv.org/abs/2610.02437](https://arxiv.org/abs/2610.02437)

    本文通过将策略更新精确分解为语义与风格两个成分，揭示了SFT比RFT遗忘更多的原因——SFT会沿教师风格偏好产生离轴风格漂移从而破坏语义记忆，而RFT能保持类内风格对称性。

    

    为什么即使所有教师演示在语义上都是正确的，监督微调（SFT）仍比强化微调（RFT）导致更多的遗忘？我们在分类任务上研究这个问题，其中每个语义类别内的标记以不同风格表达相同的语义答案。这些任务共享潜在的语义规则，但在提示分布和教师的风格偏好上有所不同。利用一个易于处理的线性softmax策略，我们推导出策略更新在语义成分和风格成分上的精确分解。我们证明，在相同策略和提示下，SFT和RFT具有平行的语义更新，但风格动态不同。从一个没有任何类内风格偏好的策略出发，采用精确策略梯度的RFT能保持这种对称性，而使用非均匀教师的SFT在群体更新下会沿着非零任务均值产生离轴风格漂移。我们利用这种漂移建立了一个……

    arXiv:2610.02437v1 Announce Type: cross  Abstract: Why does supervised fine-tuning (SFT) lead to more forgetting than reinforcement fine-tuning (RFT), even when all teacher demonstrations are semantically correct? We study this question on classification tasks where tokens within each semantic class express the same semantic answer in different styles. The tasks share an underlying semantic rule but differ in their prompt distributions and teachers' stylistic preferences. Using a tractable linear-softmax policy, we derive an exact decomposition of the updates into semantic and style components. We show that, at a common policy and prompt, SFT and RFT have parallel semantic updates but differ in their style dynamics. Starting from a policy with no within-class style preference, RFT with exact policy gradients preserves this symmetry, whereas SFT with a nonuniform teacher develops off-axis style drift along a nonzero task mean under population updates. We use this drift to establish a se
    
[^26]: 基于深度序列模型的时间序列共形预测

    Conformal Prediction for Time Series with Deep Sequence Models

    [https://arxiv.org/abs/2610.02357](https://arxiv.org/abs/2610.02357)

    本文首次系统性地研究了深度序列模型在时间序列共形预测中的应用，通过条件分位数回归等三种方法，解决了传统共形预测所依赖的数据可交换性假设在时间序列中不成立的问题。

    

    深度学习在时间序列预测方面的最新进展放大了对可靠不确定性量化的需求。共形预测作为一种无分布的框架，因能够构建具有覆盖率保证的预测区间而受到关注。然而，其覆盖率保证依赖于数据可交换性这一假设，而该假设在时间序列数据中通常不成立。目前已有大量研究致力于开发能够克服这一局限的时间序列共形预测方法。尽管循环神经网络和Transformer等深度序列模型经常被用于时间序列的共形预测中，但关于如何系统性地将深度序列模型应用于时间序列共形预测的研究仍然有限。在这项工作中，我们通过三种方法系统地研究了深度序列模型在时间序列共形预测中的应用：条件分位数回归、条件分位数（摘要在此处截断）

    arXiv:2610.02357v1 Announce Type: cross  Abstract: Recent advances in deep learning for time series prediction have amplified the need for reliable uncertainty quantification. Conformal prediction has gained attention as a distribution-free framework for constructing prediction intervals with coverage guarantees. However, its coverage guarantees rely on data exchangeability, an assumption generally violated in time series. Active research has focused on developing conformal prediction methods for time series that overcome this limitation. While deep sequence models, such as recurrent neural networks and Transformers, have often been used in conformal prediction for time series, limited work has systematically studied how deep sequence models can be utilized in conformal prediction for time series. In this work, we systematically investigate the use of deep sequence models in conformal prediction for time series through three approaches: conditional quantile regression, conditional quan
    
[^27]: 期望效用遗憾规则：极小极大与贝叶斯最优投资组合选择

    Expected Utility Regret Rule: Minimax and Bayes Optimal Portfolio Choice

    [https://arxiv.org/abs/2610.02290](https://arxiv.org/abs/2610.02290)

    提出期望效用遗憾（EUR）规则，该规则无需先验分布即可同时达到极小极大与贝叶斯最优下界，并将均值-方差组合和风险平价组合统一为该框架的特例。

    

    本研究考虑投资组合选择问题，即为投资者推荐一个投资组合，以最大化其财富的期望效用。我们的目标是构建一个在期望效用遗憾（即“先知”投资者的期望效用与从数据中选择的投资组合所实现的期望效用之差）意义上渐近最优的投资组合选择规则。我们提出了期望效用遗憾（EUR）规则，该规则联合选择投资组合类别并估计其权重。在正则参数化收益模型中，单一的EUR规则在不使用定义贝叶斯准则的先验分布的情况下，同时达到了极小极大下界和贝叶斯下界，包括它们的首项常数。随后，我们将均值-方差组合和风险平价组合推导为该框架的特例。在光滑、递增且凹的效用函数下，EUR规则与样本均值-方差组合在……（摘要在此处截断）

    arXiv:2610.02290v1 Announce Type: cross  Abstract: This study considers the problem of portfolio choice, where we recommend a portfolio to an investor to maximize the expected utility of their wealth. Our goal is to construct an asymptotically optimal portfolio choice rule in terms of expected utility regret, the difference between the expected utility of an oracle investor and that achieved by a portfolio chosen from data. We propose the Expected Utility Regret (EUR) rule, which jointly selects a portfolio class and estimates its weights. In a regular parametric return model, a single EUR rule attains both the minimax and the Bayes lower bounds, including their leading constants, without using the prior distribution that defines the Bayes criterion. We then derive the mean--variance and risk-parity portfolios as special cases of this framework. Under smooth increasing and concave utility, the EUR rule and the sample mean--variance portfolio attain the same leading expected regret when
    
[^28]: TRACE：一个包含官方运行文本的可复现电力价格预测基准

    TRACE: A Reproducible Benchmark for Electricity Price Forecasting with Official Operational Text

    [https://arxiv.org/abs/2610.02256](https://arxiv.org/abs/2610.02256)

    TRACE是一个将电力价格与预测截止时点官方运行文本配对、并严格防止信息泄露的可复现电力价格预测基准，它证明了文本上下文的预测价值——使时间序列基础模型的上尾pinball损失中位数降低7.4%。

    

    电力价格预测（EPF）为电力市场中的调度、竞价和风险管理提供支持，然而现有基准主要关注数值输入，对预测时点文本上下文的预测价值评估不足。我们提出了TRACE，这是一个包含7,300个区域-日样本的可复现基准，将美国某主要电力市场五个区域的价格与预测截止时点可获得的官方运行文本相配对。TRACE在每个截止时点重建官方运行文本，从而防止截止时点之后的信息泄露。我们从语义对齐和预测价值两方面对TRACE进行评估。语义评估结果与真实价格的中心走势及两种尾部风险保持一致，其中与价格上尾风险的一致性最强。预测价值体现在：各时间序列基础模型的上尾pinball损失中位数降低了7.4%。受控的跨日文本错配消融实验逆转了上述增益……

    arXiv:2610.02256v1 Announce Type: new  Abstract: Electricity price forecasting (EPF) supports scheduling, bidding, and risk management in electricity markets, yet existing benchmarks focus mainly on numerical inputs, leaving the forecasting value of forecast-time textual context insufficiently evaluated. We introduce TRACE, a reproducible benchmark of 7,300 zone--day instances pairing prices from five zones in a major U.S. market with official operational text available at the forecast cutoff. TRACE reconstructs official operational text at each cutoff, preventing post-cutoff information leakage. We evaluate TRACE for semantic alignment and forecasting value. Semantic assessments align with central movement and both tail risks in ground-truth prices, most consistently for upper-tail price risk. Forecasting value is reflected in a median 7.4\% reduction in upper-tail pinball loss across time-series foundation models. A controlled cross-day text-mismatch ablation reverses the gains, fall
    
[^29]: 无需受控实验的科学模拟器反事实预测

    Counterfactual Predictions in Scientific Emulators Without Controlled Experiments

    [https://arxiv.org/abs/2610.02252](https://arxiv.org/abs/2610.02252)

    提出 ReRoute 框架，无需受控实验或模拟器数据，仅通过将查询输入固定为参考值并沿已知机制路径重新引入其变化，结合事实数据微调，即可让科学模拟器准确回答“如果条件不同会怎样”的反事实预测问题。

    

    许多科学问题需要对从未观测到的情况进行推理：如果条件、干预或历史有所不同会怎样？模型可以在已观测数据上做出准确预测，但当相互关联的输入被独立改变时，模型在这类“假设性”查询上往往会失效。一种常见的补救方法是加入受控仿真数据，使这些相关因素被显式解耦，但这需要访问模拟器、计算开销可能很高，并且会继承模拟器自身的建模假设。我们提出了 ReRoute，一个面向目标性科学“假设性”预测的框架，它将事实数据与部分机制知识相结合，无需受控干预数据即可完成适配。ReRoute 将预训练骨干网络中被查询的输入固定到一个参考值，通过已知的机制路径重新引入其变化，并在原始事实数据上进行微调，同时将下游效应留给学习到的动力学模型。

    arXiv:2610.02252v1 Announce Type: cross  Abstract: Many scientific questions require reasoning about what was never observed: What if the conditions, interventions, or history had been different? Models can predict accurately on observed data yet fail on such what-if queries when correlated inputs are varied independently. A common remedy is to add controlled simulation data in which these factors are explicitly disentangled, but this requires access to a simulator, can be computationally expensive, and inherits the simulator's modeling assumptions. We introduce ReRoute, a framework for targeted scientific what-if prediction that combines factual data with partial mechanistic knowledge, without requiring controlled intervention data for adaptation. ReRoute fixes the queried input of a pretrained backbone to a reference value, reintroduces its variation through a known mechanistic pathway, and fine-tunes on the original factual data, while leaving downstream effects to the learned dynam
    
[^30]: 不同假设下基于最近邻方法从MS/MS谱图预测分子指纹的基线研究

    Nearest-neighbour baselines for fingerprint prediction from MS/MS spectra under different assumptions

    [https://arxiv.org/abs/2610.02249](https://arxiv.org/abs/2610.02249)

    该论文系统比较了在不同推理信息假设下的多种最近邻检索变体用于从MS/MS谱图预测分子指纹，旨在建立更严格的基线以实现更严谨的基准测试并更好地衡量领域进展。

    

    最近的研究表明，最近邻检索为从MS/MS谱图预测分子指纹提供了一个强有力的基线方法，其若干变体能够匹敌甚至超越当前的深度学习模型（Khoo and Barzilay, 2026; Liu et al., 2026; Gupta et al., 2026）。值得注意的是，“最近邻”涵盖了一系列检索方法，这些方法在推理阶段对可用信息的假设有所不同。在本报告中，我们系统地比较了几种最近邻变体，并展示了这些不同的假设如何影响性能。我们的目标是建立更严格的基线，从而实现更严谨的基准测试，并更好地衡量该领域的研究进展。

    arXiv:2610.02249v1 Announce Type: new  Abstract: It has recently been shown that nearest-neighbour retrieval provides a strong baseline for molecular fingerprint prediction from MS/MS spectra, with several variants matching or outperforming current deep learning models (Khoo and Barzilay, 2026; Liu et al., 2026; Gupta et al., 2026). Importantly, "nearest neighbour" encompasses a family of retrieval methods that differ in the information assumed to be available at inference. In this report, we systematically compare several nearest-neighbour variants and show how these differing assumptions affect performance. Our goal is to establish stricter baselines that enable more rigorous benchmarking and better measure progress in this area.
    
[^31]: 通过非线性变形流形的联合优化实现物理信息神经网络中的特征追踪：在激波问题中的应用

    Feature tracking in physics-informed neural networks via joint optimization of nonlinear deformation manifolds: application to shocks

    [https://arxiv.org/abs/2610.02230](https://arxiv.org/abs/2610.02230)

    提出特征追踪PINN（FT-PINN），通过联合优化解网络与参数化非线性流形上的微分同胚变形映射，使配点自动集中于弯曲、倾斜、合并等任意几何形状的激波特征处，无需先验位置信息即可提升含激波守恒律问题的求解精度。

    

    物理信息神经网络（PINNs）在求解含激波的守恒律问题时常常收敛到不准确的解，这是因为均匀分布的配点对局部特征采样不足，使得残差被已经良好解析的区域所主导。我们提出了一种特征追踪PINN（FT-PINN），其中解网络定义在固定参考域上，并与来自参数化非线性流形的微分同胚变形映射进行复合。通过最小化拉回（pulled-back）的守恒律残差，变形参数与解网络参数被联合训练。这使得配点能够沿本质上任意几何形状的特征集中，包括弯曲、倾斜和合并的激波，而无需事先知道它们的位置。该框架不依赖于特定的参数化方式。边界保持通过位移的切向投影被精确保证，且折叠……（摘要在此处被截断）

    arXiv:2610.02230v1 Announce Type: cross  Abstract: Physics-informed neural networks (PINNs) often converge to inaccurate solutions for conservation laws with shocks, because uniformly distributed collocation points undersample localized features and let the residual be dominated by regions that are already well resolved. We propose a feature-tracking PINN (FT-PINN) in which the solution network is defined on a fixed reference domain and composed with a diffeomorphic deformation map from a parameterized nonlinear manifold. The deformation and solution-network parameters are trained jointly by minimizing the pulled-back conservation-law residual. This lets collocation points concentrate along features of essentially arbitrary geometry, including curved, oblique, and merging shocks, without prior knowledge of their locations. The framework is agnostic to the choice of parameterization. Boundary preservation is enforced exactly through a tangential projection of the displacement, and foldi
    
[^32]: 实用的双机器学习方法与AI学习表示

    Pragmatic DML with AI-Learned Representations

    [https://arxiv.org/abs/2610.01935](https://arxiv.org/abs/2610.01935)

    该论文揭示了AI学习表示的误差会以结果回归误差与平衡权重误差的乘积形式影响因果参数估计，并证明了交叉拟合DML可为依赖表示的目标提供有效推断，且与按折表示学习/微调兼容，为基于AI表示的因果推断提供了实用框架。

    

    文本、图像及其他丰富的协变量正日益被压缩为AI学习得到的表示，并被用作因果分析中的控制变量。我们研究了这一方法在何种情况下是有效的，并针对基于学习表示的因果推断开发了一个实用框架。对于广泛一类估计对象，不完美的表示会通过两个表示误差的乘积来扭曲目标因果参数：一个来自结果回归，另一个来自平衡权重（或Riesz表示元）。这带来了三项建设性成果。第一，交叉拟合的双机器学习（DML）为依赖表示的目标提供了有效的Wald推断。当表示误差较小时，同一置信区间能够覆盖因果参数，甚至可以达到半参数有效界。第二，按折进行的表示学习（或微调）与针对因果参数的DML推断是兼容的。为此，我们开发了凸-（原文在此处截断）

    arXiv:2610.01935v1 Announce Type: new  Abstract: Text, images, and other rich covariates are increasingly compressed into AI-learned representations and then used as controls in causal analysis. We study when this approach is valid and develop a practical framework for causal inference with learned representations. For a broad class of estimands, an imperfect representation distorts the target causal parameter by the product of two representation errors: one in the outcome regression and one in the balancing weight (or Riesz representer). This yields three constructive results. First, cross-fitted double machine learning (DML) provides valid Wald inference for the representation-dependent target. When representation errors are small, the same interval covers the causal parameter, and it can even attain the semiparametric efficiency bound. Second, fold-wise representation learning (or fine-tuning) is compatible with DML inference for the causal parameter. To this end, we develop convex-
    
[^33]: 柏拉图式任务算术

    Platonic Task Arithmetic

    [https://arxiv.org/abs/2610.00929](https://arxiv.org/abs/2610.00929)

    本文提出“柏拉图任务向量”概念，并引入形状与模型架构和嵌入维度无关的“通用任务描述符”矩阵，使任务算术（如任务加法与取反）首次能够跨越不同模型架构进行迁移与应用。

    

    针对同一任务进行专门化训练的模型会收敛到相似的行为，然而产生这种行为的参数更新却缺乏共同的坐标系，因此权重空间中的任务算术仍然局限于单一模型，在没有结构对应关系的情况下无法跨越不同架构。借鉴柏拉图的洞穴寓言，我们假设这些特定于模型的更新是某个共享的、与模型无关的对象的投影，我们将其称为“柏拉图任务向量”。为了使这一概念对将图像或音频编码器与文本编码器配对的模型具有可操作性，我们引入了通用任务描述符：一种形状独立于架构和嵌入维度的矩阵，它记录任务的功能效果，并支持将加法和取反作为矩阵运算。将描述符迁移到目标模型中意味着对目标模型进行编辑，直到它能在任务的无标签探测图像和类别名称提示上重现该描述符，无需逐图像标注。

    arXiv:2610.00929v1 Announce Type: cross  Abstract: Models specialized for the same task converge to similar behavior, yet the parameter updates that produce it share no common coordinate system, so weight-space task arithmetic stays confined to a single model and cannot cross architectures without a structural correspondence. Drawing on Plato's allegory of the cave, we hypothesize that these model-specific updates are shadows of one shared, model-agnostic object, which we call the platonic task vector. To make it operational for models that pair an image or audio encoder with a text encoder, we introduce Universal Task Descriptors: matrices whose shape is independent of architecture and embedding dimension, which record a task's functional effect and support addition and negation as matrix operations. Transferring a descriptor into a target means editing the target until it reproduces the descriptor on the task's unlabeled probe images and class-name prompts, requiring no per-image lab
    
[^34]: 学习电价定价以实现最优需求响应

    Learning to Price Electricity for Optimal Demand Response

    [https://arxiv.org/abs/2610.00755](https://arxiv.org/abs/2610.00755)

    本文提出一种基于神经网络的上下文电价定价算法，将定价建模为Stackelberg博弈并学习从上下文特征到可行电价的受限映射，通过模拟美国多个城市电网验证了该方法能显著提升需求响应计划的价值。

    

    利用随时间变化的电价来引导消费者需求响应，并更好地使能源需求与可再生能源生产相匹配，这一点引起了广泛关注。然而，最优电价通常会随时间变化，以响应诸如天气预报、日出/日落时间和星期规律等复杂信号；而现有方法无法有效利用如此丰富的上下文信息。在此，我们提出了一种基于神经网络的上下文能量定价算法，将定价问题建模为Stackelberg博弈，并利用了Mehrabi等人（2024）提出的均场解表示方法。该方法学习从上下文特征到可行价格信号的受限映射。我们通过模拟美国多个城市的电网验证了我们的方法，结果表明，融入上下文信息可以显著提升需求响应计划的价值。

    arXiv:2610.00755v1 Announce Type: new  Abstract: There is considerable interest in using time-varying electricity prices to shape consumer demand response, and better align energy demand with renewable production. However, optimal prices generally vary over time in response to complex signals such as weather forecasts, sunrise/sunset times, and day-of-week patterns; and existing methods are not able to make efficient use of such rich contextual information. Here, we propose a neural-network-based algorithm for contextual energy pricing, modeling pricing as a Stackelberg game and leveraging a mean-field solution representation from Mehrabi et al.~(2024). The approach learns constrained mappings from contextual features to feasible price signals. We validate our approach by simulating the energy grid in several US cities, and show that incorporating contextual information can considerably increase the value of the demand response programs.
    
[^35]: Copula (连接函数) 活跃子空间 I：一种用于降阶非高斯密度估计的得分协方差方法

    Copula Active Subspaces I: A Score-Covariance Method for Reduced-Order Non-Gaussian Density Estimation

    [https://arxiv.org/abs/2609.36142](https://arxiv.org/abs/2609.36142)

    提出 Copula 活跃子空间（CAS）方法，利用 copula 得分协方差的主特征向量识别非高斯噪声分布中依赖结构的变化方向，从而实现贝叶斯推断中非高斯噪声密度的降阶表示与估计。

    

    在具有非高斯观测噪声的贝叶斯推断问题中，后验分布的准确性完全取决于噪声密度的准确性，而基于梯度的采样器需要该密度及其梯度可以逐点求值——无论是通过显式表达式还是通过代码，且不需要内部求解。我们提出 Copula 活跃子空间（CAS）来表示这种噪声密度。通过逐分量的秩变换将噪声分布的依赖结构隔离在其 copula 中，再通过秩为 r 的降阶仅保留依赖结构发生变化的方向。这些方向是 copula 得分协方差 C := Cov_{π_Z}(∇log c^Z) 的主特征向量，这正是使该降阶成为 copula 活跃子空间的原因。由于当各坐标相互独立时 C 为零，这些方向即为依赖方向，而数据的协方差未必能识别出这些方向。（摘要在此处被截断）

    arXiv:2609.36142v1 Announce Type: cross  Abstract: In Bayesian inference problems with non-Gaussian observation noise, the posterior is only as accurate as the noise density, and gradient-based samplers need that density and its gradient evaluable pointwise, whether from an explicit expression or from code, and without an inner solve. We propose Copula Active Subspaces (CAS) to represent this noise density. A componentwise rank transform isolates the noise law's dependence in its copula, and a rank-$r$ reduction keeps only the directions along which that dependence varies. These directions are the leading eigenvectors of the copula score covariance $\boldsymbol{C} := \mathrm{Cov}_{\pi_{\boldsymbol{Z}}}(\nabla\log c^{Z})$, which is what makes the reduction a copula active subspace. Because $\boldsymbol{C}$ vanishes when the coordinates are independent, these are directions of dependence, which the covariance of the data need not identify. From this construction follow a Gaussian-referen
    
[^36]: 从 Pass@K 与 Pass@1 之间的差距中学习

    Learning from the Gap Between Pass@K and Pass@1

    [https://arxiv.org/abs/2609.35793](https://arxiv.org/abs/2609.35793)

    提出 GapFT 方法，通过在 Pass@K 与 Pass@1 的差距（即单样本失败但 K 个样本内可解决的问题）上进行微调，将测试时搜索带来的能力吸收进模型，从而提升单样本解码的性能。

    

    大语言模型越来越多地采用基于可验证奖励的强化学习（RLVR）进行训练。精确的验证器还可以通过从多个样本中挑选出一个通过的响应来支持测试时扩展，而其他部署方式则使用束搜索、自适应采样或工具。我们研究单样本解码——即每个查询只获得一个响应而不进行搜索——以探究搜索中暴露出的行为能否被吸收进模型之中。现有的基于验证响应的后训练方法通常不会区分那些在首次解码时就已经解决的问题与在 K 个样本内才得以恢复的失败问题。在固定预算下，这可能导致训练样例被浪费在重复部署策略已经具备的行为上。我们提出 GapFT，它根据源检查点的单样本结果来选择训练证据，并在 Pass@K 与 Pass@1 之间的差距上进行微调：即策略在单样本上失败但在 K 个样本内能够解决的问题。我们匹配训练样例，

    arXiv:2609.35793v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly trained with reinforcement learning from verifiable rewards (RLVR). An exact verifier can also support test-time scaling by selecting a passing response from multiple samples, while other deployments use beam search, adaptive sampling, or tools. We study single-sample decoding, where each query receives one response without search, to ask whether search-exposed behavior can be absorbed into the model. Existing verified-response post-training recipes do not generally distinguish problems already solved on the first decode from failures recovered within K samples. Under a fixed budget, this can spend examples repeating behavior the deployed policy already has. We introduce GapFT, which selects training evidence by the source checkpoint's single-sample outcome and fine-tunes on the Pass@K-Pass@1 gap: problems the policy fails on one sample but solves within K samples. We match training examples,
    
[^37]: AECSF：用于高维非线性数据同化的自适应集成条件得分滤波

    AECSF: Adaptive Ensemble Conditional Score Filtering for High-Dimensional Nonlinear Data Assimilation

    [https://arxiv.org/abs/2609.32411](https://arxiv.org/abs/2609.32411)

    提出了一种免训练的自适应集成条件得分滤波器AECSF，利用条件Tweedie恒等式构建解析可处理的得分估计器，从而在高维非线性数据同化中同时避免粒子权重退化并捕捉非高斯后验结构。

    

    高维非线性动力系统的贝叶斯状态估计面临着统计精度与计算可行性之间的根本性矛盾：粒子权重可能发生退化坍缩，而高斯集成更新则可能遗漏非高斯的后验结构。基于得分的扩散滤波器提供了一种基于采样的替代方案，但现有的免训练得分滤波器通常依赖启发式的似然修正，由于忽略了与每个含噪反向粒子相关联的系统状态的不确定性，这可能会损害后验估计的精度。为解决这些问题，我们提出了AECSF——一种免训练的自适应集成条件得分滤波器。AECSF基于条件Tweedie恒等式构建了一个解析可处理的得分估计器，将含噪后验得分估计问题转化为在给定含噪反向粒子与观测条件下估计系统状态的条件均值。为估计这些条件……（摘要原文在此处截断）

    arXiv:2609.32411v2 Announce Type: replace-cross  Abstract: Bayesian state estimation for high-dimensional nonlinear dynamical systems entails a fundamental tension between statistical fidelity and computational tractability, as particle weights can collapse, while Gaussian ensemble updates can miss non-Gaussian posterior structure. Score-based diffusion filters offer a sampling-based alternative, but existing training-free score filters often rely on heuristic likelihood corrections, which can compromise posterior accuracy by neglecting uncertainty about the system state associated with each noisy reverse particle. To address these issues, we propose AECSF, a training-free adaptive ensemble conditional score filter. AECSF constructs an analytically tractable score estimator from the conditional Tweedie identity, which recasts noisy posterior score estimation as estimating the conditional mean of the system state given a noisy reverse particle and the observation. To estimate these cond
    
[^38]: DeepGOF-1：一种用于逻辑回归的预训练卷积拟合优度检验及其可计算的一致性证书

    DeepGOF-1: A Pretrained Convolutional Goodness-of-Fit Test for Logistic Regression with a Computable Consistency Certificate

    [https://arxiv.org/abs/2609.29575](https://arxiv.org/abs/2609.29575)

    提出了一种统计量为预训练冻结卷积网络的逻辑回归拟合优度检验，p值通过分析者自身的bootstrap校准来保证检验水平的精确性，并首次提供可通过单次前向传播计算得出的一致性证书。

    

    逻辑回归的拟合优度检验在最需要它们的场合反而最不可靠：在小样本条件下，其检验水平会偏离名义水平，而将多个检验组合起来会加剧这种偏离。我们提出了一种新的检验方法，其统计量是一个卷积网络，该网络只需在模拟的偏离数据上训练一次，便能将失拟“读取”为一幅图像：以协变量秩为坐标的标准ized残差网格。分析者无需进行任何训练。网络以冻结状态发布，p值是观测得分在分析者自身自助法（bootstrap）样本中的秩，因此检验水平是校准过程的属性，而非网络所学内容的属性。我们证明了在枢轴性条件下的精确性、无该条件时的渐近精确性，以及一个一致性定理——其关键条件可通过冻结权重在单次前向传播中计算得出，从而为每种备择假设提供一份证书；我们还测量了该检验的“盲锥”范围。在一个预先声明的六十单元网格上，部署后的检验水平在五十八个单元中保持在名义区间内。

    arXiv:2609.29575v1 Announce Type: cross  Abstract: Goodness-of-fit tests for logistic regression are least reliable where they are most needed: at small samples their levels drift from the nominal one, and combining them worsens the drift. We propose a test whose statistic is a convolutional network, trained once on simulated departures, that reads misfit as a picture: a grid of standardized residuals over covariate ranks. The analyst never trains. The network ships frozen, and the p-value is the rank of the observed score within the analyst's own bootstrap, so the level is a property of the calibration rather than of what the network learned. We prove exactness under pivotality, asymptotic exactness without it, and a consistency theorem whose key condition is computable from the frozen weights in one forward pass, giving a per-alternative certificate; we also measure the test's blind cone. On a pre-declared sixty-cell grid the deployed level stays in the nominal band in fifty-eight ce
    
[^39]: 用于约束采样的惩罚性非可逆朗之万算法

    Penalized Nonreversible Langevin for Constrained Sampling

    [https://arxiv.org/abs/2609.25381](https://arxiv.org/abs/2609.25381)

    提出了将平方距离惩罚与非可逆斜对称扰动相结合的朗之万算法以实现紧凸集上的约束采样，并在对数索博列夫不等式与漂移收缩条件下给出了非渐近的总变差和 2-Wasserstein 误差界。

    

    我们提出了用于从 $\pi(x)\propto e^{-f(x)}\mathbf 1_{\mathcal C}(x)$ 中采样的惩罚性非可逆朗之万算法，其中 $\mathcal C\subset\mathbb R^d$ 是一个紧凸集。这些算法将平方距离惩罚与能够保持惩罚吉布斯分布的常数型或相容的状态依赖斜对称扰动相结合。对于光滑且可能非凸的 $f$，我们在对数索博列夫不等式条件下推导了全梯度算法的非渐近总变差界。当可获得无偏随机梯度时，我们在适应性二次度量下，基于全漂移项的全局收缩性和利普希茨条件建立了 2-Wasserstein 界。对于固定的惩罚参数，相对于惩罚吉布斯分布的误差以指数速度衰减到一个 $\mathcal{O}(\sqrt{\eta})$ 邻域，其中 $\eta$ 为步长。我们还界定了惩罚吉布斯分布与目标分布之间的差异（摘要在此处被截断）。

    arXiv:2609.25381v1 Announce Type: cross  Abstract: We propose penalized nonreversible Langevin algorithms for sampling from $\pi(x)\propto e^{-f(x)}\mathbf 1_{\mathcal C}(x)$, where $\mathcal C\subset\mathbb R^d$ is a compact convex set. The algorithms combine a squared distance penalty with constant or compatible state dependent skew symmetric perturbations that preserve the penalized Gibbs distribution. For smooth, possibly nonconvex $f$, we derive nonasymptotic total variation bounds for the full gradient algorithm under a log Sobolev inequality. When unbiased stochastic gradients are available, we establish $2$-Wasserstein bounds under global contraction and Lipschitz conditions on the full drift in an adapted quadratic metric. For a fixed penalty parameter, the error relative to the penalized Gibbs distribution decays exponentially to an $\mathcal{O}(\sqrt{\eta})$ neighborhood, where $\eta$ is the stepsize. We also bound the discrepancy between the penalized Gibbs distribution and
    
[^40]: EDGE：一种用于概率二元分类器校准的闭合形式定向检验

    EDGE: a closed-form directed test for the calibration of probabilistic binary classifiers

    [https://arxiv.org/abs/2608.20511](https://arxiv.org/abs/2608.20511)

    本文提出EDGE，一种针对逻辑回归的闭合形式校准检验，通过将分箱残差投影到平滑扭曲基上，提供具有零分布的统计检验，克服了传统可靠性图无法区分真实校准误差与噪声的局限性。

    

    概率二元分类器几乎处处以判别能力来评判——准确率、ROC曲线及其曲线下面积。这些标准都对预测概率的单调扭曲保持不变，因此分类器可以完美排序，但返回的概率可能严重错误。校准是决策所需的性质，而该领域的工具——带可靠性图的分箱期望校准误差——是描述性的：它没有零分布，因此无法判断其显示的校准误差是真实还是噪声，且依赖于分箱方式。我们提出EDGE，一种针对典型概率分类器——逻辑回归的校准检验。EDGE读取与可靠性图相同的分箱预测-观测表，并将其标准化分箱残差投影到一组预定义的平滑校准扭曲形状的小基上。其零分布是加权和。

    arXiv:2608.20511v1 Announce Type: cross  Abstract: A probabilistic binary classifier is judged almost everywhere by discrimination - accuracy, the ROC curve, the area under it. Every such criterion is invariant to a monotone distortion of the predicted probabilities, so a classifier can rank perfectly and still return probabilities that are badly wrong. Calibration is the property decisions need, and the field's instrument for it, the binned expected calibration error with its reliability diagram, is descriptive: it has no null distribution, so it cannot say whether the miscalibration it displays is real or noise, and it depends on the binning. We propose EDGE, a calibration test for the canonical probabilistic classifier, logistic regression. EDGE reads the same binned predicted-versus-observed table a reliability diagram plots, and projects its standardized bin residuals onto a small pre-specified basis of smooth calibration-distortion shapes. Its null distribution is a weighted sum 
    
[^41]: 针对极端事件的生成模型微调：基于CVaR惩罚的Wasserstein梯度流

    Fine-Tuning Generative Models for Extreme Events via CVaR-Penalized Wasserstein Gradient Flows

    [https://arxiv.org/abs/2608.11544](https://arxiv.org/abs/2608.11544)

    提出了一种基于CVaR惩罚的Wasserstein梯度流方法，无需先验知识即可微调生成模型以捕捉重尾分布和极端事件，克服了标准生成器在尾部欠采样时速度消失的局限。

    

    arXiv:2608.11544v1 公告类型：交叉 摘要：我们提出了CVaR惩罚生成粒子算法（CVaR-GPA），这是一种鲁棒、尾部无关的算法，用于微调生成模型以学习重尾分布并捕捉极端事件，无需对目标的尾部特征有任何先验知识或估计。该方法是将Lipschitz正则化的Kullback-Leibler（KL）散度与条件风险价值（CVaR）差异项惩罚相结合的Wasserstein梯度流：Lipschitz正则化的KL散度在目标分布的最小假设下实现鲁棒学习，而CVaR惩罚恢复了在欠采样尾部中过早消失的速度。该惩罚流具有有界但非Lipschitz的速度场，这不同于标准生成器的Lipschitz传输映射（后者保留轻尾源的尾部行为），从而能够向更重尾的目标传输。为了定义这一f...

    arXiv:2608.11544v1 Announce Type: cross  Abstract: We propose CVaR-penalized Generative Particle Algorithm (CVaR-GPA), a robust, tail-agnostic algorithm for fine-tuning generative models to learn heavy-tailed distributions and capture extreme events, requiring no prior knowledge or estimation of the target's tail characteristics. The method is the Wasserstein gradient flow of the Lipschitz-regularized Kullback-Leibler (KL) divergence penalized by a Conditional Value-at-Risk (CVaR) discrepancy term: the Lipschitz-regularized KL divergence enables robust learning under minimal assumptions on the target distribution, while the CVaR penalty restores the velocity that otherwise vanishes prematurely in the under-sampled tails. The penalized flow admits a bounded but non-Lipschitz velocity field. This departs from the Lipschitz transport maps of standard generators, which preserve the tail behavior of a light-tailed source, and enables transport toward heavier-tailed targets. To define this f
    
[^42]: 基于深度学习的粘弹性赫兹接触中时间分辨粘附力预测

    Deep learning-based prediction of time-resolved adhesive forces in viscoelastic Hertzian contacts

    [https://arxiv.org/abs/2607.19060](https://arxiv.org/abs/2607.19060)

    本文提出一种标量条件化的有状态序列到序列深度学习模型，结合固定测量步长（FMS）表示方法，能够从位移历史快速预测粘弹性赫兹接触中的完整时间分辨粘附力演化，克服了传统数值模拟计算成本高、无法用于实时应用和设计优化的局限。

    

    快速预测粘附性软质粘弹性接触的响应是当前软体机器人技术以及抓取和操控任务中的一项挑战。确定完整的时间分辨力轨迹需要完整的数值模拟，其计算成本强烈依赖于参数，使其在实时应用或设计优化循环中并不实用。在这项工作中，我们通过训练一个标量条件化的、有状态的序列到序列深度学习模型来克服这一限制，该模型能够根据规定的位移历史预测完整的力演化，适用于短程和长程粘附两种情形。数据集涵盖四个数量级的加载和卸载速率，并包含不同的停留时间，Tabor参数范围为0.2至3.2。为了实现跨这些异构时间尺度的学习，我们引入了一种固定测量步长（FMS）表示方法，将可变的……

    arXiv:2607.19060v2 Announce Type: replace-cross  Abstract: Fast prediction of the response of adhesive soft viscoelastic contacts represents a current challenge in soft robotics and for gripping and manipulation tasks. Determining the complete time-resolved force trajectory requires full numerical simulations, whose computational cost is strongly parameter-dependent, making them impractical for real-time application or design-optimization loops. In this work, we overcome this limitation by training a scalar-conditioned, stateful, sequence-to-sequence deep learning model to predict the full force evolution from a prescribed displacement history for both short- and long-range adhesion regimes. The data set spans four orders of magnitude in loading and unloading rates and includes varied dwell times, with the Tabor parameter ranging from $0.2$ to $3.2$. To enable learning across these heterogeneous time scales, we introduce a fixed-measurement-step (FMS) representation that converts varia
    
[^43]: DAGR：通过差异感知目标交叉注意力实现状态条件化的目标表示

    DAGR: State-Conditioned Goal Representations via Difference-Aware Goal Cross-Attention

    [https://arxiv.org/abs/2607.13731](https://arxiv.org/abs/2607.13731)

    提出DAGR方法，通过多尺度门控交叉注意力和差异感知的注意力规则，将目标条件强化学习中静态的目标嵌入精炼为状态条件化表示，使策略能直接感知目标中尚未完成的部分，并从理论上揭示了后归一化放置方式对门控结构保证条件的破坏。

    

    目标条件强化学习的关键在于目标如何被编码。对比式、度量式、时间距离式和信息论式的编码器在优化目标上各持己见，但它们在一件事上是一致的：这些编码器都不关注当前状态，因此目标嵌入无法标记出目标中哪一部分仍需要采取动作，策略必须通过同时反演两个编码器来恢复这一线索。我们提出DAGR，它通过多尺度门控交叉注意力，将任何后期融合编码器的静态目标嵌入精炼为状态条件化的嵌入。一个门控残差机制使精炼结果保持在基础嵌入附近，而差异感知的注意力规则则根据每个token上状态与目标之间的失配程度对注意力分数进行偏置。我们提出一个单一条件来刻画这种精炼结构所能保证的性质，即该模块在门完全关闭时是否返回其原始输入。我们证明了常见的后归一化（post-norm）放置方式违反了这一条件，并在冻结的检查点上测量了由此造成的后果，同时通过恢复……来弥补部分损失。

    arXiv:2607.13731v2 Announce Type: replace  Abstract: Goal-conditioned reinforcement learning hinges on how the goal is encoded. Contrastive, metric, temporal-distance and information-theoretic encoders disagree on the objective. They agree on one thing. None of them sees the current state, so the embedding cannot mark which part of the goal still needs action, and the policy must recover that cue by inverting both encoders. We propose DAGR, which refines the static embedding of any late-fusion encoder into a state-conditioned one through multi-scale gated cross-attention. A gated residual holds the refinement near the base, and a difference-aware attention rule biases the scores by a per-token state-goal mismatch. A single condition decides what such a refinement can guarantee, namely whether the block returns its input at closed gates. We prove that the usual post-norm placement violates it, measure the consequence on frozen checkpoints, and recover part of the resulting loss by resto
    
[^44]: 面向样本生成模型的决策感知训练

    Decision-Aware Training for Sample-Based Generative Models

    [https://arxiv.org/abs/2607.01171](https://arxiv.org/abs/2607.01171)

    提出决策感知训练方法，通过可微分优化层计算决策损失并将其与能量分数结合，使样本生成模型的训练能够直接惩罚下游决策成本，从而在高风险决策场景中生成更具实用价值的概率预测。

    

    样本生成模型越来越多地被用于高风险决策场景下的概率预测，然而它们的训练目标对决策者的成本结构并不敏感。这些模型通常使用严格适当的评分规则进行训练，例如能量分数，这类规则按照数据密度成比例地分配训练信号，而完全没有意识到预测误差在哪些地方对下游决策的代价最高。因此，我们提出了针对样本生成模型的决策感知训练方法，在能量分数目标的基础上增加一个可微分的决策损失，直接惩罚基于模型预测采取行动所产生的成本。这一组合损失在理论上有充分依据，因为决策损失本身就是一个适当的评分规则。我们通过一个可微分优化层来计算该决策损失，其梯度集中于输出空间中对成本敏感的区域，从而使该方法的效果……

    arXiv:2607.01171v2 Announce Type: replace  Abstract: Sample-based generative models are increasingly used for probabilistic forecasting in high-stakes decision settings, yet their training objectives are blind to the decision maker's cost structure. These models are commonly trained with strictly proper scoring rules, such as the energy score, which allocate their training signal in proportion to data density, with no awareness of where forecast errors are most costly for downstream decisions. We therefore propose decision-aware training for sample-based generative models, augmenting the energy score objective with a differentiable decision loss that directly penalises the cost incurred by acting on the model's forecast. This combined loss is theoretically grounded, as the decision loss is itself a proper scoring rule. We compute the decision loss via a differentiable optimisation layer. Its gradient concentrates in cost-sensitive regions of the output space, making the method's effect
    
[^45]: 超越全局分歧：贝叶斯推理中的局部质量视角

    Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference

    [https://arxiv.org/abs/2606.27090](https://arxiv.org/abs/2606.27090)

    本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。

    

    摘要：arXiv:2606.27090v1 公告类型：交叉 摘要：全局目标函数，如KL散度和ELBO，在贝叶斯推理中被广泛用于度量分布差异。本文研究这些目标函数未能直接捕捉的局部质量行为。我们引入并使用了两种数学工具：（1）质量指数，用于记录局部质量的多项式和对数衰减尺度；（2）正则化扩展KL（RE-KL），一种在存在奇异成分时可公式化的局部化散度。质量指数有助于刻画贝叶斯更新如何改变局部质量：（1）幂对数似然因子显式地改变它；（2）参数依赖的支持域或其平滑软化，可能通过参数值附近剩余的质量量来改变局部尺度。利用局部RE-KL，我们证明了在两种KL方向下比较局部小球质量的绝对、相对和方向性不等式。这些结果共同为局部质量行为提供了理论依据。

    arXiv:2606.27090v1 Announce Type: cross  Abstract: Global objectives, such as KL divergence and ELBO, are widely used in Bayesian inference for measuring distributional discrepancy. This paper studies their local-mass behaviour that is not directly captured by such objectives. We introduce and use two mathematical tools: (1) Mass Index for recording the polynomial and logarithmic decay scales of local mass, and (2) regularised extended KL (RE-KL), a set-localised divergence that can be formulated in the presence of singular components. Mass Indices help characterise how Bayesian updating changes local mass: (1) power-log likelihood factors shift it explicitly, and (2) parameter-dependent supports, or their smooth softenings, may change the local scale through the amount of mass that remains near the parameter value. Using local RE-KL, we prove absolute, relative, and directional inequalities for comparing local small-ball masses under the two KL directions. Together, these results prov
    
[^46]: 扩散流匹配：维度改进的KL界与Wasserstein保证

    Diffusion Flow Matching: Dimension-Improved KL Bounds and Wasserstein Guarantees

    [https://arxiv.org/abs/2606.16610](https://arxiv.org/abs/2606.16610)

    本文为基于布朗运动的扩散流匹配提供了在KL散度和2-Wasserstein距离下具有更优维度依赖性的离散化误差收敛保证，在温和条件下达到了最先进的收敛标度。

    

    扩散流匹配（DFM）近来已成为一种用途广泛的生成建模框架，但其理论收敛性质仍未被完全理解。在本工作中，我们为基于布朗运动的DFM提供了精细且新颖的收敛保证，重点关注离散化误差。我们的分析在Kullback-Leibler（KL）散度和2-Wasserstein距离下进行。在有限矩条件和温和的得分（score）可积性假设下，我们推导出了相比先前工作具有更优维度依赖性的KL收敛界，据我们所知，在最少的条件下达到了最先进的收敛标度。我们进一步将分析扩展到2-Wasserstein距离：在额外的一阶得分可积性假设和弱对数凹性条件下，我们获得了与KL情形维度依赖性一致的收敛保证。

    arXiv:2606.16610v2 Announce Type: replace-cross  Abstract: Diffusion Flow Matching (DFM) has recently emerged as a versatile framework for generative modeling, yet its theoretical convergence properties remain only partially understood. In this work, we provide refined and novel convergence guarantees for Brownian motion based DFMs, focusing on the discretization error. Our analysis is conducted under the Kullback-Leibler (KL) divergence and the 2-Wasserstein distance. Under finite-moment conditions and a mild score integrability assumption, we derive KL convergence bounds with improved dimensional dependence compared to prior work, achieving, up to our knowledge, state-of-the-art scaling under minimal conditions. We further extend the analysis to the 2-Wasserstein distance: under an additional first-order score integrability assumption and a weak log-concavity condition, we obtain convergence guarantees with dimensional dependence consistent with the KL case.
    
[^47]: 面向基于种群优化的算子微积分：模块化收敛性与有限种群保证

    Operator Calculus for Population-Based Optimization: Modular Convergence and Finite-Population Guarantees

    [https://arxiv.org/abs/2606.14289](https://arxiv.org/abs/2606.14289)

    本文提出一种面向基于种群优化的算子微积分框架，使经过独立验证的更新规则效应可以模块化地组合，为收敛性分析提供可复用的构建模块，并给出有限评估预算下的收敛保证。

    

    基于种群的优化器将变异、选择和重组等更新规则组合在一起。当其中某条规则发生变化时，通常不清楚哪些收敛保证仍然成立，以及应如何评估新的组合。我们发展了一种算子微积分：算子即种群更新规则，而该微积分规定了如何将各自经过独立验证的效应进行组合。在明确的正则性和小步长条件下，由更新引起的主要变化可以相加，从而为收敛分析提供可复用的构建模块。该框架区分了找到并保留好的解、降低种群平均目标值以及使候选解集中于最优解附近这三类目标，并指出了获得有限评估预算保证所需的额外逼近条件。应用包括分布自适应、重组式演化和共识动力学，并验证了非凸情形。在……上进行的受控实验……

    arXiv:2606.14289v2 Announce Type: replace-cross  Abstract: Population-based optimizers combine update rules such as mutation, selection, and recombination. When one rule changes, it is often unclear which convergence guarantees survive or how the new combination should be assessed. We develop an operator calculus: an operator is a population-update rule, and the calculus specifies how separately checked effects can be combined. Under explicit regularity and small-step conditions, the leading changes caused by the updates add, yielding reusable building blocks for convergence analysis. The framework distinguishes finding and retaining a good solution, reducing the population's mean objective, and concentrating candidates near an optimizer, and identifies the extra approximation conditions needed for finite evaluation-budget guarantees. Applications include distribution adaptation, recombinative evolution, and consensus dynamics, with verified nonconvex cases. Controlled experiments on a
    
[^48]: 物理系统概率仿真的可靠性

    Reliability of Probabilistic Emulation of Physical Systems

    [https://arxiv.org/abs/2606.12997](https://arxiv.org/abs/2606.12997)

    本研究开发了一个评估框架，在匹配的模型规模和计算预算下系统比较了生成式模型与CRPS训练的确定性模型集合在物理系统概率预报中的表现，发现CRPS训练的模型集合在预测区间的经验覆盖率上通常具有更可靠的不确定性。

    

    生成物理系统概率预报的两种主流方法已经出现：一种是生成式模型（如扩散模型或流匹配），另一种是注入随机性的确定性模型集合，后者使用连续排序概率评分（CRPS）损失进行训练。虽然这两种方法都表现出强大的预测精度，但其不确定性的可靠性尚未得到系统评估。我们通过开发一个评估框架来填补这一空白，该框架在匹配的模型规模和计算预算下，在多种二维时空物理系统上对这两种方法进行评估。我们通过检查预测区间的经验覆盖率来评估概率仿真的可靠性，同时还考虑了精度和计算效率指标。经过CRPS训练的模型集合通常在单步预测和自回归滚动预测中都能获得更可靠的不确定性，展现出更好的覆盖率。

    arXiv:2606.12997v2 Announce Type: replace  Abstract: Two dominant approaches have emerged for generating probabilistic forecasts of physical systems: generative models, such as diffusion or flow matching; and ensembles of deterministic models with stochasticity injected, trained using the continuous ranked probability score (CRPS) loss. While both approaches have demonstrated strong predictive accuracy, the reliability of their uncertainties has not been systematically assessed. We address this gap by developing a framework to evaluate both approaches across diverse 2D spatiotemporal physical systems, under matched model size and computational budget. We assess the reliability of probabilistic emulation by inspecting the empirical coverage of predictive intervals, while also considering accuracy and computational efficiency metrics. CRPS-trained ensembles typically achieve more reliable uncertainties on both single-step prediction and autoregressive rollouts, demonstrating better cover
    
[^49]: 论大语言模型适应性的局限：模型内化先验对标注任务性能的影响

    On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance

    [https://arxiv.org/abs/2606.00467](https://arxiv.org/abs/2606.00467)

    提出“定义特定熟悉度”（DSF）指标，证明大语言模型内化先验与任务定义的对齐程度能显著预测其标注性能，且提示中的额外信息难以纠正模型零样本的“决策粘性”错误。

    

    大语言模型（LLM）越来越多地被用于零样本标注和“LLM作为评判者”任务，但其可靠性取决于模型内化的先验与用户所提供指令之间的交互方式。我们从三个维度研究了这种交互：(1) LLM对数据和任务定义的熟悉程度与其性能之间的关系；(2) 提示中的额外信息能否纠正零样本错误（即“决策粘性”）；(3) 模型对不一致任务定义的易感性。我们提出了“定义特定熟悉度”（DSF）这一概念，用于衡量模型所引出的概念与目标定义之间的对齐程度。在九个大语言模型和六个毒性数据集（五个主要数据集加一个额外的鲁棒性数据集）上的实验表明，在控制数据集身份后，DSF能够预测标注性能（偏相关系数 r=+0.41）。这种关联在所有测试的提示条件下均保持为正。相比之下……（原文摘要在此处截断）

    arXiv:2606.00467v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly used for zero-shot annotation and LLM-as-a-judge tasks, yet their reliability hinges on how model-internalized priors interact with user-provided instructions. We investigate three dimensions of this interaction: (1) how an LLM's familiarity with data and task definitions relates to performance, (2) whether additional information in prompts can correct zero-shot errors ("decision stickiness"), and (3) model susceptibility to misaligned task definitions. We introduce Definition-Specific Familiarity (DSF), which measures alignment between a model's elicited concept and the target definition. Across nine LLMs and six toxicity datasets (five primary datasets plus an additional robustness dataset), DSF predicts annotation performance after controlling for dataset identity (partial $r=+0.41$). This association remains positive across all prompting conditions tested. In contrast, three com
    
[^50]: 突破容量上限：在Stiefel流形上进行路由以构建双线性SPD层

    Escaping the Capacity Ceiling: Routing on the Stiefel Manifold for Bilinear SPD Layers

    [https://arxiv.org/abs/2605.31043](https://arxiv.org/abs/2605.31043)

    提出SCAP层，通过交叉注意力将K个Stiefel专家滤波器动态组合为样本特定的双线性映射，从而突破SPD网络中单滤波器的容量上限，解决堆叠BiMap层无法提升容量的问题。

    

    在对称正定（SPD）流形上的深度网络通过将数据几何编码为归纳偏置，有望实现富有表现力的表示，但将BiMap层与标准的ReEig非线性堆叠往往不会增加模型容量：在真实的经预处理的脑电（EEG）数据上，ReEig很少被激活，因此无论堆叠多少层，网络的表现都如同单层。在最坏情况下，当各个域之间不共享判别方向时，我们证明了单个滤波器存在容量上限，因而无法同时完全对齐所有域。为了克服这一限制，我们提出了SCAP（Stiefel交叉注意力池化），该层通过交叉注意力将K个专家组合成样本特定的双线性映射，从而实现一族Stiefel滤波器。我们证明，当各域的最优滤波器在共享切空间基点附近仅跨越少数几个方向时，该层可以用少于域数量的专家在低阶意义上匹配每个域独立的滤波器组；在最坏情况下，其对齐经验……

    arXiv:2605.31043v2 Announce Type: replace-cross  Abstract: Deep networks on the symmetric positive-definite (SPD) manifold promise expressive representations by encoding data geometry as an inductive bias, but stacking BiMap layers with the standard ReEig nonlinearity often adds no capacity: on real, preconditioned EEG data, ReEig rarely activates, so the stack behaves as a single layer at any depth. In the worst case, when domains share no discriminative directions, we prove a single filter has a capacity ceiling, so it cannot fully align every domain at once. To overcome that, we propose SCAP (Stiefel Cross-Attention Pool), a layer implementing a family of Stiefel filters by combining a pool of $K$ experts into a sample-specific bilinear map via cross-attention. We show that it matches a per-domain filter bank to first order with fewer experts than domains when domain-optimal filters span few directions near a shared tangent-space basepoint; in the worst case, its alignment empirical
    
[^51]: 面向骨架姿态轨迹的弹性形状变分自编码器

    An Elastic Shape Variational Autoencoder for Skeleton Pose Trajectories

    [https://arxiv.org/abs/2605.09231](https://arxiv.org/abs/2605.09231)

    提出了一种基于Kendall形状流形上TSRVF表示的几何感知生成模型ES-VAE，通过固有地消除平移、旋转、缩放和执行速度等干扰因素，专注于骨架姿态轨迹内在形状动态的建模。

    

    深度生成模型为建模图像、视频、3D物体和文本等复杂的结构化数据提供了灵活的框架。然而，当应用于人体骨架序列时，标准变分自编码器（VAE）往往将大量模型容量分配给干扰因素——例如相机朝向、主体尺度、视角和执行速度——而非形状及其运动的内在几何结构。我们提出了弹性形状变分自编码器（Elastic Shape - Variational Autoencoder，ES-VAE），这是一种针对骨架轨迹的几何感知生成模型，它利用了Kendall形状流形上的传输平方根速度场（TSRVF）表示。这种表示能够固有地消除形状的刚性平移、旋转和全局缩放，以及序列的时间速率变化，从而分离出潜在的形状动态。ES-VAE编码器将骨架序列映射到一个融合了R……

    arXiv:2605.09231v4 Announce Type: replace-cross  Abstract: Deep generative models provide flexible frameworks for modeling complex, structured data such as images, videos, 3D objects, and texts. However, when applied to sequences of human skeletons, standard variational autoencoders (VAEs) often allocate substantial capacity to nuisance factors-such as camera orientation, subject scale, viewpoint, and execution speed-rather than the intrinsic geometry of shapes and their motion. We propose the Elastic Shape - Variational Autoencoder (ES-VAE), a geometry-aware generative model for skeletal trajectories that leverages the transported square-root velocity field (TSRVF) representation on Kendall's shape manifold. This representation inherently removes rigid translations, rotations, and global scaling of shapes, and temporal rate variability of sequences, isolating the underlying shape dynamics. The ES-VAE encoder maps skeletal sequences to a low-dimensional latent space incorporating the R
    
[^52]: 通过能量守恒下降实现非凸优化的经典与量子加速

    Classical and Quantum Speedups for Non-Convex Optimization via Energy Conserving Descent

    [https://arxiv.org/abs/2604.13022](https://arxiv.org/abs/2604.13022)

    本文首次对能量守恒下降（ECD）进行了理论分析，证明其随机版本和量子版本在非凸优化中相比随机梯度下降和量子隧穿游走基线均能实现指数级的命中时间加速，且量子版本在高势垒问题上具有进一步的加速优势。

    

    我们提出了对能量守恒下降（ECD）的首个解析研究，作为第一部分聚焦于一维情形。我们形式化了带有能量保持噪声的随机ECD动力学，以及ECD哈密顿量的量子类比（qECD），为在一个可处理的、可显式计算势垒穿越机制的模型中通过哈密顿量模拟构建量子算法奠定了基础。对于欠猜测机制下的一维双井目标函数，我们计算了从局部极小值到全局极小值的期望动力学命中时间。我们证明sECD和qECD相对于它们各自的基于梯度的基线——随机梯度下降（SGD）和量子隧穿游走（QTW）——在连续命中时间上均表现出指数级改进。对于具有高势垒的目标函数，qECD相比sECD具有进一步的命中时间改进。从机制上讲，ECD规避了指数代价

    arXiv:2604.13022v2 Announce Type: replace-cross  Abstract: We present the first analytical study of ECD, focusing on the one-dimensional setting for this first installment. We formalize a stochastic ECD dynamics (sECD) with energy-preserving noise, as well as a quantum analog of the ECD Hamiltonian (qECD), providing the foundation for a quantum algorithm through Hamiltonian simulation in a tractable model where the barrier-crossing mechanism can be computed explicitly. For one-dimensional double-well objectives in the under-guessing regime, we compute the expected dynamical hitting times from a local minimum to the global minimum. We prove that both sECD and qECD exhibit exponential improvements in continuous hitting time relative to their respective gradient-based baselines, stochastic gradient descent (SGD) and quantum tunneling walk (QTW). For objectives with tall barriers, qECD admits a further hitting time improvement over sECD. Mechanistically, ECD sidesteps the exponential cost 
    
[^53]: Power-SMC：用于免训练大语言模型推理的低延迟序列级幂采样方法

    Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning

    [https://arxiv.org/abs/2602.10273](https://arxiv.org/abs/2602.10273)

    Power-SMC是一种免训练的低延迟序列级幂采样方法，以接近标准解码的速度实现分布锐化，从而提升大语言模型的推理能力。

    

    大语言模型的推理能力通常被归因于“分布锐化”，即将输出概率集中在高似然序列上。最近的研究表明，这种锐化效应可以在推理阶段获得，而无需修改模型参数，并且能够激发强大的推理性能。一个自然的数学形式化是“序列级幂分布”，它与模型概率的α次方（α>1）成正比。先前的研究利用Metropolis-Hastings（MH）采样从该分布中抽取样本并取得了出色的效果，然而代价是数量级级别的推理减速。我们提出了Power-SMC，这是一种免训练的采样方法，它以接近标准解码的延迟来针对相同的幂分布。Power-SMC并行维护多个候选序列，每个候选序列被赋予一个分数，即……

    arXiv:2602.10273v3 Announce Type: replace-cross  Abstract: Reasoning ability in large language models is often attributed to \emph{distribution sharpening}: concentrating output probability on high-likelihood sequences. Recent works show that this sharpening effect can be obtained at inference time, without modifying model parameters, and can elicit strong reasoning performance. A natural formalization is the \emph{sequence-level power distribution}, which is proportional to the model's probability raised to an exponent $\alpha>1$. Prior work leveraged Metropolis--Hastings (MH) sampling to draw samples from this distribution and achieves strong results, however, at order-of-magnitude inference slowdowns. We introduce \textbf{Power-SMC}, a \textit{`training-free'} sampling method that targets the same power distribution yielding close to standard decoding latency. Power-SMC maintains multiple candidate sequences in parallel. Each candidate sequence is assigned a score, namely the \emph{
    
[^54]: 何时池化才有回报？间歇性需求预测中遗忘机制下的可信度与分辨率

    When Does Pooling Pay? Credibility and Resolution under Forgetting in Intermittent-Demand Forecasting

    [https://arxiv.org/abs/2511.12749](https://arxiv.org/abs/2511.12749)

    该论文证明，在分层经验贝叶斯间歇性需求预测模型中，“遗忘序列自身历史”与“跨序列信息共享”可统一为同一个决策，并提出拟合窗口诊断、可信度界、分辨条件及事前筛选准则，用以判断何时池化信息才有价值。

    

    对大量稀疏时间序列进行预测需要做出两个选择：每个序列自身的历史应保留多少，以及应从其他序列借鉴多少信息。我们证明，当同一个指数近期性算子同时作用于分层经验贝叶斯 hurdle 模型中的项目级与组级统计量时，这两个选择便合而为一：遗忘既保持了共享先验的杠杆作用，又抬高了噪声底线，使得精细的跨序列结构更难被分辨。我们将这种双向效应形式化为拟合窗口诊断方法、可信度界以及候选池的分辨条件，并据此推导出一个事前筛选准则，用于识别精细化无法带来收益的情形。在由超过 19,000 条序列组成的五个公开间歇性需求数据集上——即项目级证据最稀疏的场景——该诊断在两个方向上均判断正确。在截断会降低可信度的情形下，共享先验是值得的：关闭它将带来 5% 和 12% 的成本损失（原文摘要此处截断）。

    arXiv:2511.12749v3 Announce Type: replace-cross  Abstract: Forecasting many sparse series requires two choices: how much of each series' own past to retain, and how much to borrow from other series. We show that when one exponential recency operator is applied to item- and group-level statistics alike in a hierarchical empirical-Bayes hurdle model, the two become one decision: forgetting preserves the leverage of a shared prior while raising the noise floor against which fine cross-series structure must be resolved. We formalize this two-sided effect as a fitting-window diagnostic, a credibility bound and a resolution condition for candidate pools, and derive from it an ex ante screen for regimes where refinement cannot pay. On five public intermittent-demand panels comprising over 19,000 series, where item-level evidence is sparsest, the diagnostic is right in both directions. Where truncation lowers credibility the shared prior pays: switching it off costs $5\%$ and $12\%$ at the sho
    
[^55]: 差分隐私主成分分析的高维渐近性

    High-Dimensional Asymptotics of Differentially Private PCA

    [https://arxiv.org/abs/2511.07270](https://arxiv.org/abs/2511.07270)

    该论文针对差分隐私主成分分析，通过分析指数机制，在高维设置下给出了隐私损失随数据集变化的精确渐近刻画，弥补了传统一致上界在特定数据集上过于保守的不足。

    

    在差分隐私中，通过引入随机噪声对敏感数据集的汇总统计量进行私有化处理后再发布。噪声水平决定了隐私损失，即量化攻击者利用已发布的统计量检测某一目标个体是否存在于数据集中的难易程度。大多数隐私分析给出的是对所有数据集一致成立的非渐近隐私损失上界。有时，这些上界在特定数据集上可能过于悲观。在这种情况下，用精确的隐私刻画来补充这些隐私上界会很有用，因为这种刻画能够量化某个机制在给定数据集上的确切隐私损失。基于这一目标，我们研究了差分隐私主成分分析（PCA），其目标是对包含 $n$ 个样本和 $p$ 个特征的数据集的主成分进行私有化。我们分析了指数机制，并为其隐私损失提供了精确的渐近刻画。

    arXiv:2511.07270v4 Announce Type: replace-cross  Abstract: In differential privacy, random noise is introduced to privatize summary statistics of a sensitive dataset before releasing them. The noise level determines the privacy loss, which quantifies how easily an adversary can detect a target individual's presence in the dataset using the published statistic. Most privacy analyses provide non-asymptotic upper bounds on the privacy loss which hold uniformly across all datasets. Sometimes, these bounds can be pessimistic on a given dataset. In such cases, it can be useful to complement these privacy bounds with sharp privacy characterizations that quantify a mechanism's exact privacy loss on a given dataset. With this goal, we study differentially private principal component analysis (PCA), where the goal is to privatize the leading principal components of a dataset with $n$ samples and $p$ features. We analyze the exponential mechanism and provide sharp asymptotic characterizations of 
    
[^56]: 基于主动学习的随机场双保真度Karhunen-Loève展开代理模型

    Bifidelity Karhunen-Lo\`eve Expansion Surrogate with Active Learning for Random Fields

    [https://arxiv.org/abs/2511.03756](https://arxiv.org/abs/2511.03756)

    提出了一种将Karhunen-Loève展开与多项式混沌展开相结合的双保真度代理模型，并利用基于交叉验证和高斯过程回归的主动学习策略自适应选择高保真度采样点，从而在有限计算成本下实现随机场的高精度建模。

    

    我们提出了一种用于不确定输入下场值感兴趣量的双保真度Karhunen-Loève展开（KLE）代理模型。本文考虑的感兴趣量为标量场。该方法将KLE的谱效率与多项式混沌展开（PCEs）相结合，以保持输入不确定性与输出场之间的显式映射关系。通过耦合能够捕捉主导响应趋势的低成本低保真度（LF）仿真与数量有限、用于校正系统偏差的高保真度（HF）仿真，所提出的方法能够构建精确且计算成本可控的代理模型。为了进一步提高代理模型精度，我们开发了一种主动学习策略，该策略基于代理模型的泛化误差自适应地选择新的HF评估点，泛化误差通过交叉验证进行估计，并采用高斯过程回归进行建模。随后通过最大化某种准则来获取新的HF样本。

    arXiv:2511.03756v2 Announce Type: replace-cross  Abstract: We present a bifidelity Karhunen--Lo\`{e}ve expansion (KLE) surrogate model for field-valued quantities of interest (QoIs) under uncertain inputs. The QoIs considered here are scalar fields. The approach combines the spectral efficiency of the KLE with polynomial chaos expansions (PCEs) to preserve an explicit mapping between input uncertainties and output fields. By coupling inexpensive low-fidelity (LF) simulations that capture dominant response trends with a limited number of high-fidelity (HF) simulations that correct for systematic bias, the proposed method can enable accurate and computationally affordable surrogate construction. To further improve surrogate accuracy, we develop an active learning strategy that adaptively selects new HF evaluations based on the surrogate's generalization error, estimated via cross-validation and modeled using Gaussian process regression. New HF samples are then acquired by maximizing an e
    
[^57]: 差分隐私作为额外收益：基于多天线基站的多址接入衰落信道上的联邦学习

    Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station

    [https://arxiv.org/abs/2510.23463](https://arxiv.org/abs/2510.23463)

    该论文研究了基于多天线基站的多址接入衰落信道上的空中联邦学习，巧妙地将信道噪声从性能损害转化为差分隐私保护的天然随机性来源，突破了现有工作在信道模型和损失函数假设上的限制，实现了隐私保护与训练性能的协同优化。

    

    联邦学习（FL）是一种分布式学习范式，它通过在训练过程中无需交换原始数据来保护隐私。在其典型的边缘部署实例中，无线传输由模拟空中计算（AirComp）实现，称为空中联邦学习（AirFL），其中固有的信道噪声扮演着一种独特的“亦敌亦友”的角色：一方面，它因带噪的全局聚合而降低训练性能；另一方面，它为隐私保护机制提供了天然的随机性来源，而这类隐私保护可以通过差分隐私（DP）进行形式化量化。然而，有效利用这种信道损伤仍然具有挑战性，因为现有工作大多在简单信道模型或受限损失函数类型的假设下，仅考虑单轮或非收敛隐私损失界限下的（本地）差分隐私增强。在本文中，我们研究了多址接入衰落信道上的空中联邦学习

    arXiv:2510.23463v4 Announce Type: replace  Abstract: Federated Learning (FL) is a distributed learning paradigm that preserves privacy by eliminating the need to exchange raw data during training. In its prototypical edge instantiation with underlying wireless transmissions enabled by analog over-the-air computing (AirComp), referred to as \emph{over-the-air FL (AirFL)}, the inherent channel noise plays a unique role of \emph{frenemy} in the sense that it degrades training due to noisy global aggregation while providing a natural source of randomness for privacy-preserving mechanisms, formally quantified by \emph{differential privacy (DP)}. It remains, nevertheless, challenging to effectively harness such channel impairments, as prior arts, under assumptions of either simple channel models or restricted types of loss functions, mostly considering (local) DP enhancement with a single-round or non-convergent bound on privacy loss. In this paper, we study AirFL over multiple-access fading
    
[^58]: 贝叶斯混合模型的快速非可逆采样器

    A fast non-reversible sampler for Bayesian mixture models

    [https://arxiv.org/abs/2510.03226](https://arxiv.org/abs/2510.03226)

    该论文提出了一种适用于贝叶斯混合模型的新型非可逆采样方案，理论上保证其渐近方差不会比标准可逆采样器差超过四倍，并在实际场景（尤其是大样本和组分重叠情形）中显著加快收敛速度。

    

    混合模型是贝叶斯建模的基石，众所周知，从所得的后验分布中进行采样可能是一项艰巨的任务。特别是，当观测数量 $n$ 较大时，流行的可逆马尔可夫链蒙特卡罗方法通常收敛缓慢。本文为贝叶斯混合模型（组分数量固定或可变）引入了一种新颖而简单的非可逆采样方案，该方案在许多感兴趣的场景中被证明能大幅优于经典采样器，尤其是在收敛阶段以及混合模型中各组分存在不可忽略的重叠时。在理论层面，我们证明了所提出的非可逆方案在渐近方差方面的性能不会比标准方案差超过四倍；并且我们提供了缩放极限分析，表明该非可逆采样器能够缩短收敛时间。

    arXiv:2510.03226v2 Announce Type: replace-cross  Abstract: Mixtures models are a cornerstone of Bayesian modelling, and it is well-known that sampling from the resulting posterior distribution can be a hard task. In particular, popular reversible Markov chain Monte Carlo schemes are often slow to converge when the number of observations $n$ is large. In this paper we introduce a novel and simple non-reversible sampling scheme for Bayesian mixture models (with fixed or varying number of components), which is shown to drastically outperform classical samplers in many scenarios of interest, especially during convergence phase and when components in the mixture have non-negligible overlap. At the theoretical level, we show that the performance of the proposed non-reversible scheme cannot be worse than the standard one, in terms of asymptotic variance, by more than a factor of four; and we provide a scaling limit analysis suggesting that the non-reversible sampler can reduce the convergence
    
[^59]: 动态规划中的误差传播：从随机控制到美式期权定价

    Error Propagation in Dynamic Programming: From Stochastic Control to American Option Pricing

    [https://arxiv.org/abs/2509.20239](https://arxiv.org/abs/2509.20239)

    本文为离散时间随机最优控制建立了结合再生核希尔伯特空间回归与蒙特卡洛抽样的动态规划近似框架，提出自然的误差分解并严格分析了误差从到期日向初始时刻反向传播的规律，可应用于美式期权定价。

    

    本文研究离散时间随机最优控制（SOC）的理论与方法基础。我们首先在一个一般的动态规划框架下表述控制问题，并引入进行详细收敛性分析所需的数学结构。相关的价值函数通过结合非参数回归方法与蒙特卡洛子抽样的序列近似来估计。回归步骤在再生核希尔伯特空间（RKHS）中进行，利用经典的核岭回归（KRR）算法，同时引入蒙特卡洛抽样方法来估计续值（continuation value）。为评估价值函数估计器的精度，我们提出了一种自然的误差分解方法，并严格控制在每个时间步产生的误差项。随后我们分析了该误差如何随时间反向传播——从到期日到初始时刻——这是一个相对较少被探索的方面。

    arXiv:2509.20239v2 Announce Type: replace-cross  Abstract: This paper investigates theoretical and methodological foundations for stochastic optimal control (SOC) in discrete time. We start formulating the control problem in a general dynamic programming framework, introducing the mathematical structure needed for a detailed convergence analysis. The associate value function is estimated through a sequence of approximations combining nonparametric regression methods and Monte Carlo subsampling. The regression step is performed within reproducing kernel Hilbert spaces (RKHSs), exploiting the classical KRR algorithm, while Monte Carlo sampling methods are introduced to estimate the continuation value. To assess the accuracy of our value function estimator, we propose a natural error decomposition and rigorously control the resulting error terms at each time step. We then analyze how this error propagates backward in time-from maturity to the initial stage-a relatively underexplored aspec
    
[^60]: 精确且准确地估计流行率

    Estimating prevalence with precision and accuracy

    [https://arxiv.org/abs/2507.06061](https://arxiv.org/abs/2507.06061)

    本文提出了一种贝叶斯聚合量化器PQ，它在保证足够覆盖率的同时生成更窄的预测区间，从而比现有方法更精确地估计流行率并更有效地量化估计的不确定性。

    

    与分类（其目标是估计每个数据点的类别）不同，量化（或称流行率估计）旨在估计数据集中各类别的分布情况。流行率估计中的一项重要任务是对流行率估计的不确定性进行量化。在本文中，我们提出了精确量化器，这是一种贝叶斯聚合量化器，能够在保证足够覆盖率（即预测区间包含真实流行率的比例足够高）的同时，获得狭窄的预测区间。我们发现，随着底层分类器判别能力的增强以及验证集与测试集大小比例的提高，PQ能够产生比现有方法更精确的流行率估计。这些实证结果表明，与现有方法相比，PQ能更有效地利用验证信息来量化流行率估计中的不确定性。

    arXiv:2507.06061v2 Announce Type: replace-cross  Abstract: Unlike classification, whose goal is to estimate the class of each data point, quantification (or prevalence estimation) aims to estimate the distribution of classes in a dataset. An important task in prevalence estimation is to quantify the uncertainty in prevalence estimates. In this paper, we introduce Precise Quantifier (PQ), a Bayesian aggregative quantifier that achieves narrow prediction intervals with sufficient coverage (i.e., sufficient proportion of intervals containing the true prevalence). We find that PQ produces more precise prevalence estimates than existing methods as the discriminative power of the underlying classifier increases and as the validation-to-test size ratio increases. These empirical results suggest that PQ uses validation information more effectively to quantify uncertainty in prevalence estimates than existing approaches.
    
[^61]: 时变贝叶斯优化的渐近性能

    Asymptotic Performance of Time-Varying Bayesian Optimization

    [https://arxiv.org/abs/2505.13012](https://arxiv.org/abs/2505.13012)

    本文首次为时变贝叶斯优化（TVBO）算法的累积遗憾提供了上界和与算法无关的下界，推导出算法具有无悔性质的充分条件，且其分析首次覆盖了实践中使用的所有主要类别的平稳核函数。

    

    时变贝叶斯优化（TVBO）是优化可能带有噪声且评估代价高昂的时变黑盒目标函数的首选框架，但其卓越的实证性能至今仍缺乏理论上的解释。TVBO算法的瞬时遗憾是否有可能渐近消失？如果可以，何时会消失？我们通过为TVBO算法的累积遗憾提供上界和与算法无关的下界来回答这一重要问题。在此过程中，我们对TVBO框架提供了重要见解，并推导出TVBO算法具有无悔性质的充分条件。据我们所知，我们的分析是首个覆盖实践中使用的所有主要平稳核函数类别的研究。

    arXiv:2505.13012v3 Announce Type: replace-cross  Abstract: Time-Varying Bayesian Optimization (TVBO) is the go-to framework for optimizing a time-varying black-box objective function that may be noisy and expensive to evaluate, but its excellent empirical performance remains to be understood theoretically. Is it possible for the instantaneous regret of a TVBO algorithm to vanish asymptotically, and if so, when? We answer this question of great importance by providing upper bounds and algorithm-independent lower bounds for the cumulative regret of TVBO algorithms. In doing so, we provide important insights about the TVBO framework and derive sufficient conditions for a TVBO algorithm to have the no-regret property. To the best of our knowledge, our analysis is the first to cover all major classes of stationary kernel functions used in practice.
    
[^62]: 前门模型下因果效应的灵活非参数推断

    Flexible Nonparametric Inference for Causal Effects under the Front-Door Model

    [https://arxiv.org/abs/2312.10234](https://arxiv.org/abs/2312.10234)

    本文在前门准则下针对平均处理效应和处理组平均处理效应提出了多种新颖的一步估计器和目标最小损失估计器，这些估计器兼容灵活的机器学习冗余参数估计，并建立了根号n一致性和渐近线性的理论条件。

    

    在观察性研究中评估因果处理效应需要解决混杂问题。虽然后门准则可以通过对观测协变量进行调整来实现识别，但在存在未测量混杂因素时会失效。前门准则提供了一种替代方法，它利用完全中介处理效应且不受处理-结果对之间未测量混杂因素影响的变量。我们在前门假设下，针对平均处理效应和处理组平均处理效应开发了新颖的一步估计器和基于目标最小损失的估计器。我们的估计器建立在观测数据分布的多种参数化之上，其中包括完全避免对中介变量密度建模的方法，并且与灵活的、基于机器学习的冗余参数估计相兼容。我们通过推导建立了根号n一致性和渐近线性的条件。

    arXiv:2312.10234v4 Announce Type: replace-cross  Abstract: Evaluating causal treatment effects in observational studies requires addressing confounding. While the back-door criterion enables identification through adjustment for observed covariates, it fails in the presence of unmeasured confounding. The front-door criterion offers an alternative by leveraging variables that fully mediate the treatment effect and are unaffected by unmeasured confounders of the treatment-outcome pair. We develop novel one-step and targeted minimum loss-based estimators for both the average treatment effect and the average treatment effect on the treated under front-door assumptions. Our estimators are built on multiple parameterizations of the observed data distribution, including approaches that avoid modeling the mediator density entirely, and are compatible with flexible, machine learning-based nuisance estimation. We establish conditions for root-n consistency and asymptotic linearity by deriving se
    
[^63]: 无早停的分数扩散模型：有限费舍尔信息就足够了

    Score diffusion models without early stopping: finite Fisher information is all you need. (arXiv:2308.12240v1 [math.ST])

    [http://arxiv.org/abs/2308.12240](http://arxiv.org/abs/2308.12240)

    无早停的分数扩散模型不需要得分函数的Lipschitz均匀条件，只需要有限的费舍尔信息。

    

    分数扩散模型是一种围绕着与随机微分方程相关的得分函数估计的生成模型。在获得近似的得分函数之后，利用它来模拟相应的时间逆过程，最终实现近似数据样本的生成。尽管这些模型具有显著的实际意义，但在涉及非常规得分和估计器的情况下，仍存在一个显著的挑战，即缺乏全面的定量结果。在几乎所有的Kullback Leibler散度的相关结果中，都假设得分函数或其近似在时间上是Lipschitz均匀的。然而，在实践中，这个条件非常严格，或者很难建立。为了解决这个问题，先前的研究主要是关注分数扩散模型的早停版本在KL散度上的收敛界限，并且...

    Diffusion models are a new class of generative models that revolve around the estimation of the score function associated with a stochastic differential equation. Subsequent to its acquisition, the approximated score function is then harnessed to simulate the corresponding time-reversal process, ultimately enabling the generation of approximate data samples. Despite their evident practical significance these models carry, a notable challenge persists in the form of a lack of comprehensive quantitative results, especially in scenarios involving non-regular scores and estimators. In almost all reported bounds in Kullback Leibler (KL) divergence, it is assumed that either the score function or its approximation is Lipschitz uniformly in time. However, this condition is very restrictive in practice or appears to be difficult to establish.  To circumvent this issue, previous works mainly focused on establishing convergence bounds in KL for an early stopped version of the diffusion model and
    

