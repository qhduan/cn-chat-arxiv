# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Autoregressive Chain-of-Thought Learning via Bagging](https://arxiv.org/abs/2611.21597) | 本文提出一种基于bagging的自回归思维链学习算法，通过在自助采样数据上训练CoT一致的生成器并进行多数投票集成，实现了与思维链长度无关的最优样本复杂度，且该集成可被蒸馏为单个生成器。 |
| [^2] | [ONAM: Fitting Orthogonal Neural Additive Models for Post-Hoc Interpretability of Black-Box Predictions](https://arxiv.org/abs/2611.21550) | 提出了R包ONAM，通过正交神经加性模型对黑盒模型的预测进行事后功能分解，将其拆分为与各输入特征相关的组件，并配备可视化和定量评估工具，实现模型无关的可解释性。 |
| [^3] | [The cost of enforcing long-run coverage: a phase transition in online conformal prediction](https://arxiv.org/abs/2611.21542) | 本文揭示了在线保形预测中存在一种相变现象：在对抗性场景下强制对长期经验覆盖率施加严格约束，会以瞬时覆盖出现不稳定的振荡为代价，从而刻画了长期覆盖与瞬时稳定之间的根本性权衡。 |
| [^4] | [Almost Controlled: The FDR of Benjamini-Hochberg on Weighted Conformal p-Values](https://arxiv.org/abs/2611.21509) | 本文通过经精确算术认证的反例证明Benjamini-Hochberg程序在有限样本中无法总是控制加权保形p值的错误发现率，并量化了FDR相对超出目标水平的上界（阶约为√(Bm/(nq))）。 |
| [^5] | [Differentiable Exact Learning](https://arxiv.org/abs/2611.21486) | 该论文提出可微精确学习框架与可微有限状态转换器（DFST），仅需极小的训练数据集即可在算术和通用模拟任务上实现极端长度泛化，并将成功经验总结为学习公式“状态跟踪控制器 + 无界外部环境 + 策略轨迹观测 = 极端长度泛化”。 |
| [^6] | [Generative Data Augmentation Without External Data: An Empirical and Theoretical Study on Deep Networks](https://arxiv.org/abs/2611.21467) | 本研究首次系统研究了不依赖外部数据的生成式数据增强，通过理论证明其收益由生成分布与真实分布间的总变差距离带来的代价与训练集扩大带来的收益之间的权衡决定，并通过多骨干、多数据集（含医疗数据）的实验验证了生成样本何时能提升或降低分类器的测试准确率。 |
| [^7] | [Comments on "Deep Fr\'echet Regression"](https://arxiv.org/abs/2611.21454) | 本文针对“深度Fréchet回归”提出基于条件流匹配的替代视角，直接在响应空间中学习协变量依赖的传输映射，无需单独的流形学习与逆向映射，为建模完整条件分布提供了更灵活的框架。 |
| [^8] | [Nonparametric Instrumental Kernel Regression: The Well- and Misspecified Setting Under General Link Conditions](https://arxiv.org/abs/2611.21391) | 本文将核工具变量回归推广到取值为希尔伯特空间元素（如曲线、密度、核均值嵌入）的结果变量，并证明其两阶段谱正则化估计量是在核积分算子生成的希尔伯特尺度上的正则化方法，从而在良定与误定情形下刻画了估计量收敛的目标解。 |
| [^9] | [High-probability Uniform Mixing via Dobrushin Contraction and Accelerated Hamiltonian Monte Carlo](https://arxiv.org/abs/2611.21305) | 本文针对临界阻尼Metropolis调整动能朗之万方法，同时在水条件数与维度两方面证明了混合加速性：在强凸、梯度与Hessian Lipschitz条件下，仅需 𝒪((κ^{1/2}+L_H^{1/2}m^{-3/4})d^{1/4}) 次梯度评估即可在总变差意义下混合到 ε 精度，且结果对高概率区域任意起点一致成立并适用于温启动。 |
| [^10] | [Compressed Simulation-Based Inference: Neural Posterior Estimation for Massive Data via Low-Dimensional Sketches](https://arxiv.org/abs/2611.21286) | 该论文提出压缩式基于模拟的推断方法，通过仅用 $2p+1$ 个随机低维草图即可实现后验一致性，从而避免手工设计摘要统计量的困难和学习摘要的高昂计算成本，实现海量数据下的神经后验估计。 |
| [^11] | [In-Context Estimation and Prediction for Parametric ODEs](https://arxiv.org/abs/2611.21268) | 该论文证明Transformer可以通过“粗搜索初始化+局部强凸区域梯度下降”的两阶段近似最大似然程序，从含噪轨迹观测中对一般非凸的参数化常微分方程实现上下文参数估计与轨迹预测。 |
| [^12] | [Variance-adaptive confidence regions for fixed-benchmark binomial mixtures, with application to LLM evaluation](https://arxiv.org/abs/2611.21266) | 该论文提出方差自适应（VA）置信区域方法，基于固定基准上的二项计数为经验分布线性泛函构造对任意有界泛函同时具有有限样本覆盖保证的置信区间，其可通过线性规划计算并在一阶最坏情形意义下达到最优宽度，可直接应用于大语言模型评估。 |
| [^13] | [Dimension-free spectral laws and the risk of spectral filters](https://arxiv.org/abs/2611.21254) | 该论文突破了经典随机矩阵理论中比例渐近性的限制，在无穷维希尔伯特空间设定下建立了与维数无关的谱定律，给出了样本 Gram 矩阵经验谱定律的确定性等价物及显式非渐近误差界，为函数型数据分析和核方法中的谱滤波器风险分析提供了理论工具。 |
| [^14] | [The Price of Choosing Examples: Context Overfitting in Adaptive In-Context Learning](https://arxiv.org/abs/2611.21251) | 该论文首次将提示构建形式化为自适应数据分析，提出“上下文容量”概念，证明自适应选择示例的上下文学习存在 √(κ/k) 阶的泛化过拟合代价，并给出匹配下界和 PAC-贝叶斯检索规则。 |
| [^15] | [Anytime-Valid PAC-Bayes Certificates for Adaptive Test-Time Scaling](https://arxiv.org/abs/2611.21241) | 本文证明了适用于自适应选择的任意时刻有效PAC-Bayes证书，使测试时扩展系统在自适应确定采样量、提示词和验证器之后仍能获得统计有效的泛化保证，且同一保证覆盖部署前的全部方案搜索过程。 |
| [^16] | [Summary Sufficiency in Simulation-Based Inference via Mutual Information](https://arxiv.org/abs/2611.21158) | 该论文针对仿真推断中摘要统计量充分性的互信息估计歧义问题，提出一种可处理的方法，以区分较小的BA下界究竟源于摘要统计量不充分还是密度估计器不准确。 |
| [^17] | [The Limits of Closed-Form Couplings for Few-Step Generation](https://arxiv.org/abs/2611.21119) | 研究发现高维情况下扩散过程的闭式解析解几乎退化为线性变换，因此无需预训练、直接利用闭式耦合进行少步生成的方法在实践中不可行。 |
| [^18] | [CONGA: Compiling Gaussian Fiber Geometry for Structured Bayesian Inference](https://arxiv.org/abs/2611.20995) | 该论文提出CONGA编译器，通过识别结构化贝叶斯模型中的条件高斯潜变量块，并利用精确仿射传输映射（Brenier和Knothe-Rosenblatt映射）将其与非高斯核心分离，从而将KL与相对Fisher推断目标约化到低维核心上以提升后验推断效率。 |
| [^19] | [FARCH: A Functional ARCH Model for Calibrated Prediction Bands](https://arxiv.org/abs/2611.20972) | 提出FARCH模型，通过在曲线及其平方残差上复用同一个基于小波半度量的核加权方案，联合非参数地预测函数型数据的条件均值与波动率，从而构建出逐点和一致（上确界范数）两种校准预测带。 |
| [^20] | [A Convergence Analysis of the Random Feature Method for Linear PDEs](https://arxiv.org/abs/2611.20951) | 本文在核方法框架下对随机特征方法求解线性偏微分方程建立了系统的收敛性理论，将有限随机特征逼近与RKHS估计问题归结为核岭回归，并基于积分算子特征值给出了误差的上下界。 |
| [^21] | [Enhancing Mixed Oil Length Prediction through Graph Representation Learning](https://arxiv.org/abs/2611.20660) | 该论文提出SDI-GRL模型，通过将混油样本和管道特征构建为图节点并利用消息传递机制，捕捉样本间的统计依赖性，从而显著提升成品油管道运输中混油段长度的预测精度。 |
| [^22] | [Calibrated Functional Prediction from Overlapping Nonlinear Observations](https://arxiv.org/abs/2611.20567) | 该论文提出基于Schur有理算子与局部高斯投影的共享稀疏核方法，结合半正定特征表示、正矩阵积分与排序校准扰动界，实现了从重叠非线性观测出发、具有完整有限池钉球风险保证的校准泛函预测。 |
| [^23] | [Decorrelation Before Variable Selection: Left and Right Transforms and Their Trade-Off](https://arxiv.org/abs/2611.20420) | 该论文揭示了变量选择前去相关的左变换（保持系数、改变噪声）与右变换（保持噪声、改变系数）之间的权衡，指出当真实变量排前时Cholesky白化能保持噪声变量不活跃，而同时保留系数与噪声的去相关则会损失信息。 |
| [^24] | [High-dimensional transfer learning with unaligned ordinal responses](https://arxiv.org/abs/2611.20411) | 该论文针对源域与目标域有序响应类别不对齐的高维迁移学习问题，提出了样本分割和全目标两种算法，实现了参数估计与超额风险的极小化极大最优速率，并在医学数据上验证了有效性。 |
| [^25] | [Generalized Prediction Set with Bandit Feedback](https://arxiv.org/abs/2611.20387) | 提出BanditGPS，首次在赌博机反馈的在线学习设置下实现结合OOD检测的集合值预测，通过对真实类别标签的估计，生成小而信息丰富的预测集并提升OOD检测性能。 |
| [^26] | [Measuring Coherence in Predictive Models](https://arxiv.org/abs/2611.20325) | 本文提出了基于MMD的“不连贯性比率”来衡量模型更新过程偏离连贯性（即当前预测与更新后预期预测的一致性）的程度，并刻画了不连贯性何时会影响诸如主动学习和贝叶斯实验设计等顺序决策。 |
| [^27] | [The Manifold-Index Model: provably and efficiently avoiding the statistical and computational curse of dimensionality](https://arxiv.org/abs/2611.20275) | 提出流形指数模型及MANOR算法，通过对相似响应的预测变量进行多尺度PCA估计切平面，使预测误差达到仅依赖流形内在维度 $m$ 的最优非参数速率 $n^{-4/(m+4)}$，可证明地克服了环境维度带来的维数灾难。 |
| [^28] | [Integrating Latent Block Models into Variational Autoencoders for Disentangled Representation Learning](https://arxiv.org/abs/2611.20253) | LBM-VAE通过将潜在块模型集成到变分自编码器中，利用块结构的潜在表示实现观测数据与潜在维度的联合聚类，并允许变化因素跨潜在维度组编码，从而有效支持解耦表示学习。 |
| [^29] | [Does Learning Order Explain Forgetting? The Lie Bracket as an Early-Warning Signal for Catastrophic Forgetting Without Old Data](https://arxiv.org/abs/2611.20087) | 本文实证验证了两个任务梯度场的李括号决定了训练顺序导致的权重差异并与灾难性遗忘相关，且仅凭旧任务训练结束时存储的梯度和前20个Hessian特征对，就能在无需旧数据的条件下早期预测遗忘，效果优于EWC惩罚项。 |
| [^30] | [Weighted conformal prediction under covariate shift: process limits, estimated weights and efficiency](https://arxiv.org/abs/2611.20052) | 本文建立了协变量偏移下加权保形预测中自归一化加权经验过程的核心极限定理，证明收敛当且仅当 $\chi^2$ 散度有限（而非 $\sup v$），由此导出跨水平覆盖率的联合分布、一致性 bootstrap、渐近精确的膨胀系数与 Berry-Esseen 速率，并揭示用校准协变量估计的权重可能比 oracle 权重更高效。 |
| [^31] | [Certified Algorithms for Gaussian Location Mixture NPMLEs via Discretized Gradient Flows](https://arxiv.org/abs/2611.19882) | 该论文提出了一种基于Fisher-Rao梯度流熵邻近离散化的认证算法，能够在与样本量近线性、与 $1/\varepsilon$ 多项式相关的时间内计算高斯位置混合NPMLE的 $\varepsilon$-近似解，且所得估计具有精确有限维表示并附带似然差距保证。 |
| [^32] | [Sharp Gaussian Approximation for Q-Learning](https://arxiv.org/abs/2611.19850) | 本文为带Polyak-Ruppert平均的异步Q-学习在Wasserstein距离下建立了$\widetilde{O}(n^{-1/4})$的精确非渐近高斯逼近速率及匹配下界，并揭示了马尔可夫噪声偏差与有限样本协方差偏差之间的权衡如何决定收敛速率。 |
| [^33] | [Conditional Asset Pricing Transformer](https://arxiv.org/abs/2611.19783) | 本文提出条件资产定价Transformer（CAP-T），通过两阶段注意力框架将超额收益分解为条件错误定价与因子风险补偿等成分，实现对收益更准确的预测与理论保证。 |
| [^34] | [Tight Sequential Confidence Intervals for Online Kernel Regression](https://arxiv.org/abs/2611.19772) | 本文在核的最坏情形下刻画了在线核回归置信区间的尖锐极小极大宽度，从而解决了Vakili等人（2021）提出的开放问题。 |
| [^35] | [Learning Autoregressive Neural Operators on Solver-Defined Physical Constraint Manifolds](https://arxiv.org/abs/2611.19753) | 提出PCM-NO方法，通过“预测增量—切向投影—度量回缩”的投影残差机制，保证自回归神经算子每一步预测都严格落在满足不可压缩性与边界条件的求解器可行集上，实现物理约束流形上的封闭状态转移。 |
| [^36] | [Joint Learning of Factual and Counterfactual Worlds for Causal Prior-data Fitted Networks](https://arxiv.org/abs/2611.19710) | 该论文提出一种新的因果先验数据拟合网络，通过在合成环境中联合学习每个单元的事实与反事实两个世界的结果并利用二者间的依赖关系，突破了现有因果PFN只能学习单一世界结果的限制，从而提升处理效应的估计效果。 |
| [^37] | [Minimax Estimation of Discrete Distributions under Hockey-stick Divergence](https://arxiv.org/abs/2611.19702) | 本文首次研究了基于曲棍球棒散度的离散分布估计问题，给出了以字母表大小、样本数和散度参数表示的极小极大期望损失的非渐近上下界，并揭示了多项式与指数两种量级机制之间的相变现象。 |
| [^38] | [A Geometric Understanding of Denoising Autoencoder with Applications to Image Denoising](https://arxiv.org/abs/2611.19653) | 本文在流形假设下首次系统建立了去噪自编码器的泛化理论，通过局部坐标卡与神经网络逼近技术在三种几何场景（单流形、分离与相交流形并集）下推导出期望重构误差的上界，为其在图像去噪中的优异表现提供了几何解释。 |
| [^39] | [Bellman Residual Bracketing: Resolving the Sample Complexity of Discounted and Average-Reward RL](https://arxiv.org/abs/2611.19566) | 本文提出贝尔曼残差括号技术，建立了折扣型与平均奖励强化学习中插件方法的紧致样本复杂度上界，并结合下界在常数因子内解决了两类问题的极小极大样本复杂度。 |
| [^40] | [Steering Bayesian Inference and Decisions through Data Poisoning](https://arxiv.org/abs/2611.19557) | 该论文提出一种数据投毒攻击方法：攻击者只需通过删除或复制少量精心选择的观测数据，即可操纵贝叶斯后验分布中的特定矩或下游决策，而无需完全指定目标后验分布。 |
| [^41] | [Geometric Stability and Sample Complexity for Compositional Function Priors](https://arxiv.org/abs/2611.19523) | 该论文证明了复合函数类 f(x)=g(Ax+Δh(x)) 的覆盖数界，表明其统计复杂度可在低维内蕴参数化速率与全维非参数速率之间平滑过渡，且过渡程度由非线性修正项的大小控制。 |
| [^42] | [Over-the-Air Semantic Alignment via Dual Stacked Intelligent Metasurfaces](https://arxiv.org/abs/2611.19515) | 本文提出双堆叠智能超表面架构，通过在波域中联合利用链路两端的可编程超表面与无线信道，无需数字处理即可实现异构潜在空间之间的端到端语义对齐。 |
| [^43] | [Interpreting Random Forests: Comparing Surrogate Models and Rule Extraction Methods](https://arxiv.org/abs/2611.19471) | 本文系统比较了12种解释随机森林的方法，发现没有一种方法在所有目标上均最优，其中SLIM在回归任务中综合表现最强，为研究者根据具体研究问题选择合适的解释方法提供了指导。 |
| [^44] | [Efficient and Accurate Causal Validation of Training Data Attribution](https://arxiv.org/abs/2611.19451) | 该论文提出从训练子集数量、重训练种子数和归因集成模型数三个维度对LDS估计进行去偏差，从而大幅降低训练数据归因因果验证所需的重新训练计算量，同时保持估计精度。 |
| [^45] | [Generalization Bounds for Neural Networks with Sparse Connectivity](https://arxiv.org/abs/2611.19382) | 本文针对层间权重矩阵受ℓ_p拟范数约束的稀疏连接深度神经网络，推导出了非渐近泛化间隙上界，从理论上解释了稀疏性如何影响神经网络的统计性能。 |
| [^46] | [Learning the Average Dose-Response Function with a Right-Censored Treatment](https://arxiv.org/abs/2611.19204) | 本文提出了一种基于增广局部估计方程的多重稳健估计量，用于在处理时长右删失与结局缺失下估计平均剂量-反应函数，并通过去偏技术实现了无需欠平滑的渐近有效统计推断。 |
| [^47] | [Mean-field Token Steering](https://arxiv.org/abs/2611.19163) | 该论文将 Transformer 深层中的随机词元引导建模为单位球面上相互作用粒子系统的平均场最小努力控制问题，证明其受控动力学等价于适配球面几何的平均场薛定谔桥，并提出动态 Sinkhorn 递推的新推广方法来计算一致的词元演化。 |
| [^48] | [Log-optimal two-sample testing by betting](https://arxiv.org/abs/2611.19062) | 本文针对标准Borel观测空间构造了一个统一的、任意时刻有效的打赌式双样本检验方法，对所有未知分布对及任意样本量比率均渐近对数最优，其最优对数财富增长率由Jensen–Shannon散度刻画。 |
| [^49] | [High-dimensional limit theorems for adaptive variants of SGD I: Ballistic limits](https://arxiv.org/abs/2611.19059) | 本文通过平均化方法将ADAM、MUON等自适应动量SGD变体中由指数移动平均和预条件化引起的非马尔可夫性转化为马尔可夫有效动力学，建立了高维极限定理，从而实现对不同优化算法的精确定量比较。 |
| [^50] | [HeteroWear: Sensor-Aware Self-Supervised Learning for Heterogeneous Wearable Activity Recognition](https://arxiv.org/abs/2611.19027) | 提出了传感器感知的自监督框架HeteroWear，通过在包含八个数据集、173名受试者的异构可穿戴语料库上预训练，将传感器内时间建模与跨传感器关系建模分离，实现了可跨异构传感器配置迁移的活动识别表征。 |
| [^51] | [Taking Band-Limited Maximum Likelihood to High Dimensions: Linear Memory, the Parametric Rate, and a Self-Tuning Cut-off](https://arxiv.org/abs/2611.19011) | 该论文证明分箱BLML方程是严格凸目标函数的驻点条件，并提出无需构造核矩阵的块坐标下降算法，实现了线性内存扩展与单卡GPU在八维下拟合百万级数据点，同时证明了参数速率 n⁻¹ 与自调节截断在高维中依然有效。 |
| [^52] | [Trajectory-Based Bayesian Discovery of a Host-Pathogen Model from Quasi-Steady-State Observations](https://arxiv.org/abs/2611.18815) | 提出了一种结合稀疏贝叶斯学习、ℓ0迭代硬阈值与基于贝叶斯证据的两步交换爬山搜索的方法，能够从准稳态观测数据中成功恢复宿主-病原体动力学模型的全部真实机制，克服了经典稀疏辨识方法在此情形下的失效问题。 |
| [^53] | [Setting Optimal Reserve Prices using Fine-Tuned Language Models](https://arxiv.org/abs/2611.18785) | 该论文以拍卖出价的结构似然函数为训练目标微调预训练语言模型，从非结构化商品描述中学习估值分布表示，从而在无需预先指定产品属性或建模潜在竞拍者数量的情况下，为差异化商品设定最优保留价格。 |
| [^54] | [Stagewise Adaptive Boosting](https://arxiv.org/abs/2611.18662) | 本文提出分阶段自适应提升这一通用框架，通过沿单调路径选择符合结构偏好的更新来促进稀疏性等解的性质，并可扩展至组选择、多结果模型相似性以及预测变量优先关系等场景。 |
| [^55] | [Domino: Joint Unmasking via First-order Markov Chain for Few-Step Masked Diffusion](https://arxiv.org/abs/2611.18645) | 提出 Domino 框架，用一阶马尔可夫链建模同一步骤中揭示的 token 之间的条件依赖，仅以 2.6% 的参数开销实现少步掩码扩散的联合去掩码生成。 |
| [^56] | [Ensemble Bounds for Pairwise Dissimilar Decomposition of Deterministic Datasets](https://arxiv.org/abs/2611.18625) | 本文利用冗余图与二项随机图，并结合Lovász局部引理，首次为确定性数据集的两两不相似批次分解的最小规模给出了上界与下界的集成界限。 |
| [^57] | [Temporal Regularized Learning: Self-Supervised Learning Local in Space and Time](https://arxiv.org/abs/2611.17019) | 提出TeReL方法，通过逐样本的纯局部更新（Hebbian前馈与反Hebbian侧向连接）实现空间和时间上均局部化的VICReg式自监督学习，本质上是一种软化的慢特征分析。 |
| [^58] | [On the Difficulty of Sequence-Level Supervised Learning for Autoregressive Language Models](https://arxiv.org/abs/2611.16736) | 本文证明了将推理时解码过程纳入训练的两种序列级监督损失函数（基于贪婪解码和自回归采样）都存在无下界的根本性缺陷，并通过自回归Gumbel技巧揭示了二者之间的内在联系。 |
| [^59] | [Improved Elastic Weight Consolidation as an Optimization Constraint for Continual Learning](https://arxiv.org/abs/2611.16039) | 本文将持续学习重新表述为一种优化约束（CLOC），证明EWC的经验Fisher惩罚是该约束的一阶形式，由此揭示EWC对已拟合良好的样本权重过低的缺陷，并提出改进的EWC方法。 |
| [^60] | [Announced Breaks Separate Conformal Reliability from Frequency Calibration](https://arxiv.org/abs/2610.23917) | 本文证明预先公告的结构性变点使“下一区间的保形可靠性”与“封闭窗口的频率校准”这两种重置审计目标可以分离，并提出在公告时点重启经验分位数与自适应递归的方法，前者达到1/√n的minimax可靠性速率、后者达到1/n的频率差距，且实验中覆盖率误差显著优于自适应基线（0.035对0.052）。 |
| [^61] | [Latent Annealed Posterior Sampling with a Variance-Preserving Bridge for Sparsely Observed Random Fields](https://arxiv.org/abs/2610.20659) | 提出LAPS方法，利用流匹配学习潜在先验并在潜在空间中进行退火后验采样，结合方差保持桥控制协方差膨胀，从而高效解决稀疏观测随机场后验推断中空间不确定性刻画与计算成本增长的难题。 |
| [^62] | [Measuring Variable Importance Using Proportional Inclusive Marginal Effects](https://arxiv.org/abs/2610.17053) | 提出了PRIME（比例包含边际效应）这一新的变量重要性度量方法，通过无模型变量选择技术仅纳入活跃变量进行计算，克服了PME仅适用于确定性模型且计算成本高的局限，能从含噪声数据中准确可靠地量化变量重要性。 |
| [^63] | [Kernelized Parameter Sensitivity Atlases for Amortized Conditional Interpretability](https://arxiv.org/abs/2610.12737) | 该论文提出 KPSA，通过经验神经正切核的分块分解构建参数组敏感性的核化图谱，用轻量级核函数摊销式地为任意新输入估计敏感性，从而以远低于逐样本反向传播的成本实现输入条件化的模型可解释性。 |
| [^64] | [Explaining the Saliency Map Sparsity of Adversarially-Trained Neural Networks](https://arxiv.org/abs/2610.10666) | 本文首次为对抗训练神经网络梯度显著性图的稀疏性现象提供了理论解释，证明随着数据点和神经元数量增长，网络的最小化解收敛到具有最小梯度和Barron范数的贝叶斯分类器，从而自然产生了稀疏性。 |
| [^65] | [Inference for stochastic differential equations driven by weighted sub-fractional Brownian motion using neural networks and the Euler approximation](https://arxiv.org/abs/2610.00793) | 本文提出将神经网络与Euler近似相结合，对由加权亚分数布朗运动这类高斯过程驱动的随机微分方程进行统计推断，从离散观测数据中估计漂移系数、扩散系数与噪声协方差。 |
| [^66] | [Global Convergence of Third-Order Langevin Dynamics for Non-Convex Optimization via Simulated Annealing](https://arxiv.org/abs/2609.28611) | 该论文证明了在模拟退火框架下，采用固定摩擦与递减噪声的三阶朗之万动力学在非凸优化中可依概率收敛到全局最小值，并给出了离散化格式保持该收敛速率的充分步长条件。 |
| [^67] | [Rank and computation of the pathlifting Jacobian of a DAG ReLU network](https://arxiv.org/abs/2609.18682) | 本文通过对骨架矩阵进行初等归纳证明了DAG ReLU网络路径提升雅可比矩阵的秩，并提出了一种无需反向传播、计算成本更低的雅可比矩阵计算方法。 |
| [^68] | [A Deep Latent Variable Framework for Jointly Modeling Missingness, Measurement Error, and Heterogeneity](https://arxiv.org/abs/2608.30040) | 提出了一种基于深度潜变量表示的统一概率框架，通过层次树路由变分自编码器联合处理缺失数据、测量误差和群体异质性三类问题，并借助重新汇聚路由机制在子群体间共享参数以提升统计效率。 |
| [^69] | [K\"ahler landscapes for complex neural network descents and guarantees including a search and destroy of the Calabi-Yau manifold](https://arxiv.org/abs/2608.19584) | 本文提出了一种在复参数神经网络中使用Kähler信息度量和自然梯度下降的新方法，并针对Calabi-Yau流形上的不良曲率条件提供了理论保证，通过几何定义的全局势实现了搜索与消灭策略。 |
| [^70] | [Approximate Message Passing with Random Initialization for Phase Retrieval](https://arxiv.org/abs/2608.01654) | 本文对随机初始化的近似消息传递（AMP）算法在无噪声相位恢复中的轨迹给出了非渐近刻画，证明其可同时达到信息论弱恢复阈值和强恢复阈值，性能与谱初始化AMP相当。 |
| [^71] | [Deep Adaptive Bayesian Screening](https://arxiv.org/abs/2607.16927) | DABS通过离线学习策略网络在严格实验预算下自适应地执行因子筛选，并创新性地在部署时整合后验推断，从而同时给出因子活性概率和效应大小的可信区间。 |
| [^72] | [From Jumps to Signatures: A Generative Method for Temporal Point Processes](https://arxiv.org/abs/2607.06652) | 该论文提出到达间隔嵌入方法，将签名理论从连续路径扩展到时间点过程的离散事件序列，并据此构建了首个基于签名的生成模型sigTPP，通过路径级损失对完整轨迹进行训练。 |
| [^73] | [Dynamic Regret for Non-Stationary Linear Bandits via Misspecification Reductions](https://arxiv.org/abs/2607.02891) | 该论文提出一种分块规约方法，将非平稳线性老虎机中的参数漂移转化为上下文线性老虎机中的模型误设定问题，从而为一般性时变决策集建立了动态遗憾保证。 |
| [^74] | [Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference](https://arxiv.org/abs/2606.27090) | 本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。 |
| [^75] | [Human-AI Teaming Through the Lens of Calibration](https://arxiv.org/abs/2606.10906) | 本文从统计校准视角分析人机协作团队框架，发现组合人类与模型预测的现有方法会破坏人类的校准性，而委托方法虽能保持校准，却将负担转移到需要精细校准且难以达到要求的拒绝元模型上。 |
| [^76] | [Tree-Guided Identify-Then-Exploit: A Unified Framework of Best Arm Identification and Regret Minimization for Dueling Bandits](https://arxiv.org/abs/2606.01799) | 提出首个统一框架TG-ITE，通过树引导识别方法在无需传递性假设下以O(N)次比较找到高置信度候选臂，并针对最优臂识别、弱遗憾和强遗憾三种目标设计相应利用策略，首次以赢家停留式算法实现O(N)弱遗憾及BAI与弱遗憾的联合优化。 |
| [^77] | [Parameter-Free and Group Conditional Online Conformal Prediction](https://arxiv.org/abs/2606.00419) | 本文提出了一种免参数（无需学习率调优）的群组条件在线保形预测算法，能够在数据分布随时间变化的在线环境中为多个群组提供强大的覆盖保证，从而解决公平性问题。 |
| [^78] | [Beyond Differences: Estimating and Tuning Ratio-Based Treatment Effects](https://arxiv.org/abs/2605.26288) | 本文提出Q-Learner方法，将二元结果的比率CATE估计转化为两个倾向性分类任务，仅需标准的log-loss超参数调优即可可靠地估计处理效应的水平与分布，尤其适合工业实践场景。 |
| [^79] | [Improving the Efficiency of Subgroup Analysis in Randomized Controlled Trials with TMLE](https://arxiv.org/abs/2605.15483) | 该论文提出两种利用同一试验中亚组外参与者信息的目标最大似然估计量（TMLE-PR与A-TMLE），在不依赖外部真实世界数据的前提下，提高随机对照试验中亚组分析的统计效率。 |
| [^80] | [Nonparametric estimation of time-varying network connections by multi-stage smoothing](https://arxiv.org/abs/2605.06862) | 本文提出一种多阶段平滑估计方法，通过结合时间局部平滑与数据驱动的节点域平滑来非参数估计时变网络的边概率，能够同时捕捉网络连接的平滑时间演化与结构模式。 |
| [^81] | [Training-Time Batch Normalization Reshapes Local Partition Geometry in Piecewise-Affine Networks](https://arxiv.org/abs/2605.04946) | 该论文证明了训练时BN层的切换超平面是批次质心超平面的平移，其偏移仅由标准化量与批次散度决定（与偏置和批次位置无关），并据此给出CPA网络局部线性区域数量的精确公式和下界，揭示BN从根本上重塑了网络的局部划分几何。 |
| [^82] | [PAIR-CI: Calibrated Conditional Independence Testing for Causal Discovery with Incomplete Data](https://arxiv.org/abs/2605.04838) | 该论文提出PAIR-CI，一种通过配对置换设计将多重插补直接融入推断过程、使插补误差在损失差异中相互抵消的非参数条件独立性检验，从而解决了不完整数据因果发现中“先插补后检验”范式校准失准的问题。 |
| [^83] | [Layer-wise Geometric Approximation Rates for Deep Networks](https://arxiv.org/abs/2604.20219) | 本文构造了一个宽度为 $3dN+2d+4$、各隐藏层激活模式完全相同的单一深度网络，其逐层中间读出以由目标函数 $L^p$ 平移模（Hölder 条件下为 $O(N^{-\alpha\ell})$）控制的速率在逐级精细的尺度上逼近目标，从而为网络深度赋予了尺度依赖的全新解释。 |
| [^84] | [Structural interpretability in SVMs with truncated orthogonal polynomial kernels](https://arxiv.org/abs/2604.15285) | 该论文提出了正交表示贡献分析（ORCA）方法，利用截断正交多项式核SVM的有限标准正交多项式基展开，通过正交核贡献（OKC）指数精确量化各输入变量、交互阶数及多项式次数对决策函数的贡献，从而实现对支持向量机的结构可解释性分析。 |
| [^85] | [Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression](https://arxiv.org/abs/2604.13410) | 该论文提出一种两阶段核岭回归方法，先学习响应模型再构建校正分布偏移的伪结果，从而在存在混淆偏差的情况下估计连续处理效应，并利用平均效应函数更简单的结构特点达到了最优学习界限。 |
| [^86] | [Overcoming the Incentive Collapse Paradox](https://arxiv.org/abs/2603.27049) | 该论文证明了激励崩塌源于任何仅基于观测任务准确性的支付规则，并提出一种哨兵审计支付机制，能以有限成本维持严格为正且可控的人类努力水平，不受AI准确性的影响。 |
| [^87] | [A Noise Sensitivity Exponent Controls Large Statistical-to-Computational Gaps in Single- and Multi-Index Models](https://arxiv.org/abs/2603.17896) | 该论文提出由激活函数决定的噪声敏感度指数（NSE），证明这一简单量完全刻画并控制了单指标与多指标模型中大范围统计-计算差距的存在与大小。 |
| [^88] | [RL unknotter, hard unknots and unknotting number](https://arxiv.org/abs/2603.07955) | 本文开发了基于强化学习的纽结图简化流水线，智能体通过学习Reidemeister移动策略成功解开“非常困难”的平凡纽结图，并通过自改进的工作簿驱动扩展系统性地改进了素纽结解结数的上界。 |
| [^89] | [Generalized Bayes for Causal Inference](https://arxiv.org/abs/2603.03035) | 提出了一种无需显式似然建模的广义贝叶斯因果推断框架，通过直接对因果估计目标设置先验并用识别驱动的损失函数进行更新，使现有基于损失的因果估计器具备完整的不确定性量化能力。 |
| [^90] | [A Comparative Study of UMAP and Other Dimensionality Reduction Methods](https://arxiv.org/abs/2603.02275) | 本文系统比较了UMAP、有监督UMAP与PCA、SIR、t-SNE等多种降维方法的性能，发现有监督UMAP在分类任务中表现优异，但在回归场景下的应用存在明显局限。 |
| [^91] | [Sharp Convergence Rates for Masked Diffusion Models](https://arxiv.org/abs/2602.22505) | 本文首次为掩码扩散模型中的欧拉采样方法建立了直接基于总变差 的精确收敛速率分析，克服了现有KL散度分析参数依赖松散、假设过强以及无法覆盖首达采样器（FHS）的局限。 |
| [^92] | [Concept Influence: Leveraging Interpretability to Improve Performance and Efficiency in Training Data Attribution](https://arxiv.org/abs/2602.14869) | 该论文提出“概念影响”方法，将模型行为归因于语义方向而非单个测试样本，并证明简单的探针归因方法是其一阶近似，从而提升了训练数据归因的性能与效率。 |
| [^93] | [Constrained and Composite Sampling via Proximal Sampler](https://arxiv.org/abs/2602.14478) | 该论文提出基于近端采样器的框架，通过上图提升、割平面方法与拒绝采样，实现了凸集上约束采样及复合密度 $e^{-f-h}$ 的采样，且仅需分离、求值和次梯度预言机，摆脱了对投影和障碍函数的依赖。 |
| [^94] | [Distribution-free certificate of similarity between distributions with blurred total variation distance](https://arxiv.org/abs/2602.05862) | 论文证明对连续分布而言TV距离的无分布置信上界必然平凡，从而提出用其松弛形式——模糊TV距离——在不做任何分布假设的情况下以有限样本置信度认证分布相似性。 |
| [^95] | [Optimal scaling laws in learning hierarchical multi-index models](https://arxiv.org/abs/2602.05846) | 该论文为两层神经网络学习层次化多指标目标建立了精确的信息论标度律理论，证明层次特征通过级联相变被依次恢复且简单的谱估计器即可达到最优速率，从而统一解释了浅层神经网络的标度律、平台现象与谱结构。 |
| [^96] | [Multi-layer Cross-attention is Provably Optimal for Multi-modal In-context Learning](https://arxiv.org/abs/2602.04872) | 本文首次建立了多模态上下文学习的理论分析框架，证明单层自注意力无法在多模态任务上一致地达到贝叶斯最优，并提出一种新颖的线性化多层交叉注意力机制来可证明地恢复最优预测器。 |
| [^97] | [PCA of probability measures: Sparse and Dense sampling regimes](https://arxiv.org/abs/2602.02190) | 本文首次研究了基于 $n$ 个概率测度（每个通过 $m$ 个样本观测）的双重渐近机制下的PCA，推导出经验协方差算子与PCA超额风险 $n^{-1/2} + m^{-\alpha}$ 的收敛速度，刻画了从稀疏到稠密采样机制的转变，并证明稠密机制下的收敛速度是极小极大最优的。 |
| [^98] | [Function-on-function nonlinear autoregressive model (F2NARX) for probabilistic surrogate modeling of stochastic dynamical systems](https://arxiv.org/abs/2602.01929) | 本文提出 F2NARX 模型，将 F-NARX 重新表述为函数对函数回归问题，并结合主成分分析、高斯过程回归与无迹变换，在保持高精度的同时大幅提升预测效率，实现对随机动力系统响应的高效概率预测。 |
| [^99] | [A Statistical Theory of Gated Attention through the Lens of Hierarchical Mixture of Experts](https://arxiv.org/abs/2602.01468) | 该论文首次为门控注意力建立了统计理论，证明门控注意力矩阵的每个元素可写成分层专家混合的形式，并由此从理论上证明门控注意力比多头自注意力更具样本效率。 |
| [^100] | [A Unified Kantorovich Duality for Multimarginal Optimal Transport](https://arxiv.org/abs/2601.17171) | 该论文建立了多边际最优传输的统一Kantorovich对偶理论，在紧与非紧（波兰空间）情形下均证明了对偶最优解的存在性，并刻画了最优对偶势的结构。 |
| [^101] | [Lost in Aggregation: The Causal Interpretation of the Instrumental Variable Estimand](https://arxiv.org/abs/2601.12120) | 本文揭示了一个在工具变量有效性论证中被普遍忽视的“聚合处理变量”问题，并证明在线性设定下聚合处理的因果效应通常是模糊的，因为它依赖于对聚合变量干预的具体实施方式。 |
| [^102] | [Physics-Informed Neural Networks for Joint Source and Parameter Estimation in Advection-Diffusion Equations](https://arxiv.org/abs/2512.07755) | 提出了一种基于多网络神经正切核的加权自适应PINN方法，以对流扩散方程为耦合约束，实现对源函数与PDE参数的联合估计，有效缓解稀疏测量下源反演的严重病态性问题。 |
| [^103] | [Incomplete U-Statistics of Equireplicate Designs: Berry-Esseen Bound and Efficient Construction](https://arxiv.org/abs/2510.20755) | 本文通过将不完全U统计量的依赖图视为其设计超图的线图，利用Stein方法首次在完全退化情形下为确定性设计建立了Berry–Esseen界，并推广至学生化统计量，从而在不依赖重抽样的前提下实现了高效构造与高斯极限推断。 |
| [^104] | [Prediction-Augmented Trees for Reliable Statistical Inference](https://arxiv.org/abs/2510.16937) | 该论文提出了预测增强残差树（PART）估计器，通过利用额外的计算从固定的标注数据、未标注数据和预训练预测器中挖掘更多推断价值，从而构建更精确的置信区间，提升统计推断的可靠性。 |
| [^105] | [On accuracy of density estimation in generative models based on stochastic optimal control](https://arxiv.org/abs/2506.03043) | 本文为基于薛定谔桥的生成模型建立了密度估计的非渐近统计保证，证明在光滑性假设下精确经验风险最小化可达到 $\widetilde{O}(n^{-2s/(2s+d)})$ 的KL收敛速率，达到经典非参数极小极大最优率（至多相差对数因子）。 |
| [^106] | [Benign Overfitting of Two-Layer ReLU Networks in Fixed Dimension through Localized Initialization](https://arxiv.org/abs/2505.11621) | 该论文通过一种结合高斯权重与仅在小球形帽上激活的局部化神经元的初始化方法，首次在固定维度下证明了有限宽度两层ReLU网络经梯度流训练后可实现对含噪数据的良性过拟合，并给出了样本量、宽度与训练时间的明确界限。 |
| [^107] | [Forests for Differences: Beyond Parametric Difference-in-Differences](https://arxiv.org/abs/2505.09706) | 本文提出双重差分贝叶斯因果森林（DiD-BCF）这一非参数模型，统一估计ATE、GATE和CATE，并将条件平行趋势与无预期假设直接嵌入贝叶斯非参数框架，在非线性、选择偏差和效应异质性场景下显著优于现有基准方法。 |
| [^108] | [Rectifying Conformity Scores for Better Conditional Coverage](https://arxiv.org/abs/2502.16336) | 提出一种通过对符合性分数进行可训练变换（基于条件分位数估计）的新方法，在保证精确边际覆盖的同时显著改善条件覆盖，尤其适用于多输出问题中自适应置信集的构建。 |
| [^109] | [The Revenue Cost of Inference in Online Assortment Optimization](https://arxiv.org/abs/2501.19277) | 该论文刻画了在线组合实验中收入损失与估计方差之间的理论最优权衡前沿，证明当收入最优组合在所有目标梯度方向均提供信息时零成本学习可行，并提出保留参考报价的试运行设计使实际方差准则收敛到该oracle最优值。 |
| [^110] | [Projected gradient methods for nonconvex and stochastic smooth optimization: new complexities and auto-conditioned stepsizes](https://arxiv.org/abs/2412.14291) | 该论文提出了一类自适应条件的投影梯度方法，无需预知梯度Lipschitz常数或线搜索即可在凸与非凸优化问题上实现统一的迭代复杂度，并进一步推广至随机优化设置。 |
| [^111] | [Transfer Learning for High-dimensional Quantile Regression with Distribution Shift](https://arxiv.org/abs/2411.19933) | 该论文针对高维分位数回归，提出了一种新的可迁移集合与迁移学习框架，能够同时应对参数偏移、协变量偏移和残差偏移三种分布偏移问题，并通过非渐近误差界、源检测一致性理论以及正交去偏推断方法证明了其有效性和优越性。 |
| [^112] | [RECOVAR: Representation Covariances on Deep Latent Spaces for Seismic Event Detection](https://arxiv.org/abs/2407.18402) | 提出了RECOVAR，一种无需事件/噪声标签、直接从原始波形训练的无监督地震事件检测方法，通过卷积自编码器集成与互协方差度量实现事件检测，性能可比监督检测器并在跨数据集和不平衡数据场景下保持稳健。 |
| [^113] | [Convergence of Continuous Normalizing Flows for Learning Probability Distributions](https://arxiv.org/abs/2404.00551) | 本文研究了具有线性插值的CNFs在从有限随机样本中学习概率分布时的理论性质，建立了非渐近误差界，并提供了收敛分析框架。 |
| [^114] | [A Nearly-Linear Time Algorithm for Structured Support Vector Machines.](http://arxiv.org/abs/2307.07735) | 这篇论文提出了针对结构化支持向量机的接近线性时间算法，解决了二次规划输入规模和解决时间的问题。 |
| [^115] | [Penalized Likelihood Inference with Survey Data.](http://arxiv.org/abs/2304.07855) | 本文将三种Lasso推断方法应用于调查数据的推断，证明在具有调查权重和/或异方差的广义线性模型中的推断程序渐近有效性， 并将这些方法推广到对非线性参数函数进行推断，例如在调查logit模型中的平均边际效应 |

# 详细

[^1]: 通过Bagging实现最优的自回归思维链学习

    Optimal Autoregressive Chain-of-Thought Learning via Bagging

    [https://arxiv.org/abs/2611.21597](https://arxiv.org/abs/2611.21597)

    本文提出一种基于bagging的自回归思维链学习算法，通过在自助采样数据上训练CoT一致的生成器并进行多数投票集成，实现了与思维链长度无关的最优样本复杂度，且该集成可被蒸馏为单个生成器。

    

    我们研究如何在给定由 $f$ 生成的长度为 $T$ 的思维链（CoT）演示数据的情况下，从函数类 $\mathcal{F}$ 中学习一个自回归的下一词元生成器 $f\colon\Sigma^\star\to\Sigma$。我们提出了一种基于 bagging 的简单算法，该算法输出多个生成器的多数投票结果，这些生成器在训练数据的自助采样（bootstrap）子集上被训练为与 CoT 保持一致。对于二元词元集合，该算法达到了最优样本复杂度 $O((d_{\mathrm{VC}}+\log(1/\delta))/\epsilon)$，且与思维链长度 $T$ 无关，其中 $d_{\mathrm{VC}}$ 是 $\mathcal{F}$ 的 VC 维。对于有限词元集合 $\Sigma$，VC 维被其多分类类比——Graph 维 $d_{\mathrm{G}}$ 所取代，样本复杂度在 $\log|\Sigma|$ 因子范围内是最优的。多数投票集成中生成器的数量为 $O(\log(T/\delta))$，并且我们证明了该集成可以被蒸馏为 $\mathcal{F}$ 中的单个生成器……

    arXiv:2611.21597v1 Announce Type: new  Abstract: We study how to learn an autoregressive next-token generator $f\colon\Sigma^\star\to\Sigma$ from a class $\mathcal{F}$, given chain-of-thought (CoT) demonstrations of length $T$ generated by $f$. We present a simple algorithm based on bagging, which outputs a majority vote of generators that are trained to be CoT-consistent on bootstrap samples of the training data. For binary token sets, it achieves the optimal sample complexity of $O((d_{\mathrm{VC}}+\log(1/\delta))/\epsilon)$, independent of the CoT length $T$, where $d_{\mathrm{VC}}$ is the VC dimension of $\mathcal{F}$. For finite token sets $\Sigma$, the VC dimension is replaced by a multiclass analogue, the Graph dimension $d_{\mathrm{G}}$, and the sample complexity is optimal up to a $\log|\Sigma|$ factor. The number of generators in the majority-vote ensemble is $O(\log(T/\delta))$, and we show that the ensemble can be distilled into a single generator in $\mathcal{F}$ at the ex
    
[^2]: ONAM：拟合正交神经加性模型以实现黑盒预测的事后可解释性

    ONAM: Fitting Orthogonal Neural Additive Models for Post-Hoc Interpretability of Black-Box Predictions

    [https://arxiv.org/abs/2611.21550](https://arxiv.org/abs/2611.21550)

    提出了R包ONAM，通过正交神经加性模型对黑盒模型的预测进行事后功能分解，将其拆分为与各输入特征相关的组件，并配备可视化和定量评估工具，实现模型无关的可解释性。

    

    现代机器学习模型（如人工神经网络）通过日益复杂的黑盒架构实现了高预测性能，但这限制了它们在需要透明性、可解释性和公平性的场景中的应用。在此我们介绍R包ONAM，它提供了一种新颖且实用的函数分解实现方法——这是一种通过结构化效应表示来促进黑盒模型可解释性的技术。ONAM实现了一个算法框架，可将预测事后分解为与不同输入特征相关联的组件。该软件包设计注重可用性和效率，包含可视化功能以及评估可解释性的定量指标。作为一种模型无关的方法，它可以应用于广泛的预测模型。文中描述了其数学基础、算法设计和软件架构。

    arXiv:2611.21550v1 Announce Type: cross  Abstract: Modern machine learning models, such as artificial neural networks, achieve high predictive performance through increasingly complex black-box architectures, limiting their use in settings where transparency, interpretability, and fairness are required. Here we present the R package ONAM, which provides a novel and practical implementation of functional decomposition, a technique to facilitate the interpretability of black-box models through structured effect representations. ONAM implements an algorithmic framework for the post-hoc decomposition of predictions into components associated with different input features. The package is designed for usability and efficiency, including functionalities for visualization as well as quantitative measures to assess interpretability. As a model-agnostic method, it can be applied to a wide range of prediction models. We describe the mathematical foundations, algorithmic design, and software archi
    
[^3]: 强制长期覆盖率的代价：在线保形预测中的相变现象

    The cost of enforcing long-run coverage: a phase transition in online conformal prediction

    [https://arxiv.org/abs/2611.21542](https://arxiv.org/abs/2611.21542)

    本文揭示了在线保形预测中存在一种相变现象：在对抗性场景下强制对长期经验覆盖率施加严格约束，会以瞬时覆盖出现不稳定的振荡为代价，从而刻画了长期覆盖与瞬时稳定之间的根本性权衡。

    

    考虑一个在线预测场景：在每个时刻，我们会收到一个新的特征向量，并被要求提供一个预测区间或预测集合，使其很可能包含对应的响应值。在这种场景下，如果数据点是独立同分布的（或更一般地，是可交换的），保形预测提供了一种机制，可以在任何已训练模型的输出周围构建有效的预测集合，在每个时刻 t 提供覆盖率 ≥ 1-α 的保证。然而在实践中，诸如分布漂移、变点或其他挑战可能导致独立同分布假设失效。在这种对抗性场景中，在线保形预测方法仍然可以对长期经验覆盖率提供保证。但是，这两类目标之间存在权衡：为了防范对抗性场景而对长期经验覆盖率施加严格约束，可能会导致瞬时覆盖出现不稳定的振荡（原摘要在此处被截断）。

    arXiv:2611.21542v1 Announce Type: cross  Abstract: Consider an online prediction setting, where at each time, we are given a new feature vector and asked to provide a prediction interval or prediction set that will likely contain the corresponding response. In this setting, if the data points are i.i.d. (or more generally, exchangeable), conformal prediction provides a mechanism for building valid prediction sets around the output of any trained model, offering coverage probability $\geq 1-\alpha$ at each time $t$. In practice, however, issues such as distribution drift, changepoints, or other challenges may cause violations of the i.i.d. assumption. In this adversarial setting, online conformal prediction methods can still offer guarantees on long-run empirical coverage. However, there is a trade-off between these two types of goals: enforcing tight constraints on long-run empirical coverage to protect against the adversarial setting can lead to unstable oscillations in the instantane
    
[^4]: 近乎受控：Benjamini-Hochberg方法在加权保形p值上的错误发现率

    Almost Controlled: The FDR of Benjamini-Hochberg on Weighted Conformal p-Values

    [https://arxiv.org/abs/2611.21509](https://arxiv.org/abs/2611.21509)

    本文通过经精确算术认证的反例证明Benjamini-Hochberg程序在有限样本中无法总是控制加权保形p值的错误发现率，并量化了FDR相对超出目标水平的上界（阶约为√(Bm/(nq))）。

    

    加权保形p值将保形推断推广到了存在协变量偏移的情形。使用它们的一种方式是将其输入Benjamini-Hochberg（BH）程序，用于决定选择哪些测试单元或检测异常值。但加权保形p值可能不满足正相关性，因此我们想知道：在有限样本中，BH程序是否仍然能控制错误发现率（FDR）？基于实验，它似乎可以，但我们证明了它并不能。我们给出了两个反例，并通过精确的有理数算术加以认证，其中FDR超过了其目标水平q。我们还确定了FDR与q之间的差异可以有多大。当有n个校准点、m个测试单元和权重比B时，相对超额FDR/q-1被一个关于n、m、B和q的函数所界定，当Bm/(nq)较小时，该函数的阶为√(Bm/(nq))。该界对B≤n的每一种配置以及每一个n都成立。这一阶是可以达到的

    arXiv:2611.21509v1 Announce Type: cross  Abstract: Weighted conformal p-values generalize conformal inference to cases with covariate shift. One way to use them is to pass them to the Benjamini-Hochberg (BH) procedure, either to decide which test units to select or to detect outliers. But weighted conformal p-values may not be positively dependent, so we want to know: does BH still control the false discovery rate (FDR) in finite samples? Based on experiments, it appears to, but we prove it doesn't. We give two counterexamples, certified in exact rational arithmetic, where the FDR exceeds its target level $q$. We also identify how large the difference between the FDR and $q$ can be. With $n$ calibration points, $m$ test units and weight ratio $B$, the relative excess $\mathrm{FDR}/q-1$ is bounded by a function of $n$, $m$, $B$ and $q$ that is of order $\sqrt{Bm/(nq)}$ when $Bm/(nq)$ is small. The bound holds for every configuration with $B\le n$ and every $n$. This order can be reached
    
[^5]: 可微精确学习

    Differentiable Exact Learning

    [https://arxiv.org/abs/2611.21486](https://arxiv.org/abs/2611.21486)

    该论文提出可微精确学习框架与可微有限状态转换器（DFST），仅需极小的训练数据集即可在算术和通用模拟任务上实现极端长度泛化，并将成功经验总结为学习公式“状态跟踪控制器 + 无界外部环境 + 策略轨迹观测 = 极端长度泛化”。

    

    我们提出了一个可微精确学习框架，该框架将无界外部环境与状态跟踪神经控制器相耦合，并在策略轨迹观测（PTOs）上对其进行训练——PTO是一种专家行为轨迹，它暴露观测和动作，同时隐藏内部状态。除学习框架外，我们还提出了可微有限状态转换器（DFST），它使用多线性更新，支持对数并行训练，并能模拟任意有限状态控制器，从而通过环境交互实现计算通用性。基于微小的训练数据集，紧凑的DFST在算术和通用模拟任务上实现了极端长度泛化。我们在RNN和GRU上复现了DFST的成功，从而支持一个更广泛的学习公式：状态跟踪控制器 + 无界外部环境 + 策略轨迹观测 = 极端长度泛化。除长度泛化之外，我们的度量提取……（摘要在此处截断）

    arXiv:2611.21486v1 Announce Type: cross  Abstract: We introduce a framework for differentiable exact learning that couples an unbounded external environment to a state-tracking neural controller, which we train on policy-trajectory observations (PTOs): expert behavior traces that expose observations and actions while hiding internal states. In addition to the learning framework, we introduce Differentiable Finite-State Transducers (DFSTs) which use multilinear updates, support log-parallel training, and emulate arbitrary finite-state controllers, yielding computational universality through environmental interaction. From tiny training datasets, compact DFSTs achieve extreme length generalization on arithmetic and universal-simulation tasks. We replicate the success of DFSTs on RNNs and GRUs, thereby supporting a broader learning recipe: State-Tracking Controller + Unbounded External Environment + PTOs = Extreme Length Generalization. Beyond length generalization, our metric-extraction 
    
[^6]: 无需外部数据的生成式数据增强：对深度网络的实证与理论研究

    Generative Data Augmentation Without External Data: An Empirical and Theoretical Study on Deep Networks

    [https://arxiv.org/abs/2611.21467](https://arxiv.org/abs/2611.21467)

    本研究首次系统研究了不依赖外部数据的生成式数据增强，通过理论证明其收益由生成分布与真实分布间的总变差距离带来的代价与训练集扩大带来的收益之间的权衡决定，并通过多骨干、多数据集（含医疗数据）的实验验证了生成样本何时能提升或降低分类器的测试准确率。

    

    生成式数据增强将来自生成模型的样本添加到分类器的训练集中。许多有影响力的研究使用在大型外部数据集上预训练的生成器，从而将合成数据的价值与通过预训练迁移的知识混杂在一起。我们研究无外部数据的情形，即每个生成器仅从分类器自身的训练集学习，并探讨何时以及为何向该训练集添加生成的样本会提高或降低分类器的测试准确率。我们在六个数据集（其中两个为医疗数据集）上使用四种骨干网络训练分类器，并添加来自四个生成器的样本。我们对使用合成数据训练的分类器的超额风险进行了界定，即其总体风险可能高出其假设类中最小风险的程度。该界定因真实分布与生成分布之间的总变差距离而对合成数据进行惩罚，同时因训练集规模的扩大而给予合成数据相应的收益。该...

    arXiv:2611.21467v1 Announce Type: cross  Abstract: Generative data augmentation adds samples from generative models to a classifier's training set. Many influential studies use generators pre-trained on large external datasets, and so mix the value of synthetic data with knowledge transferred through pretraining. We study the case without external data, where each generator learns only from the classifier's own training set, and ask when and why adding generated samples to that training set raises or lowers the classifier's test accuracy. We train classifiers with four backbones on six datasets, two of them medical, adding samples from four generators. We bound the excess risk of a classifier trained with synthetic data, that is, how far its population risk can lie above the minimum risk within its hypothesis class. The bound charges synthetic data for the total variation distance between the real and generated distributions and credits synthetic data for a larger training set. The gai
    
[^7]: 对“深度Fréchet回归”的评论

    Comments on "Deep Fr\'echet Regression"

    [https://arxiv.org/abs/2611.21454](https://arxiv.org/abs/2611.21454)

    本文针对“深度Fréchet回归”提出基于条件流匹配的替代视角，直接在响应空间中学习协变量依赖的传输映射，无需单独的流形学习与逆向映射，为建模完整条件分布提供了更灵活的框架。

    

    我们讨论了《美国统计学会杂志》（JASA）的讨论论文“深度Fréchet回归”，并探索了基于条件流匹配的另一种视角。我们提出直接在响应空间中学习依赖于协变量的传输映射，从而避免了单独的流形学习和逆向映射。这一视角为引入几何结构和建模完整的条件分布提供了一个潜在更为灵活的框架。

    arXiv:2611.21454v1 Announce Type: cross  Abstract: We discuss the Journal of the American Statistical Association (JASA) discussion paper "Deep Fr\'echet Regression" and explore an alternative perspective based on conditional flow matching. We propose learning covariate-dependent transport directly within the response space, avoiding separate manifold learning and reverse mapping. This perspective offers a potentially more flexible framework for incorporating geometric structure and modeling full conditional distributions.
    
[^8]: 非参数工具变量核回归：一般链接条件下的良定与误定情形

    Nonparametric Instrumental Kernel Regression: The Well- and Misspecified Setting Under General Link Conditions

    [https://arxiv.org/abs/2611.21391](https://arxiv.org/abs/2611.21391)

    本文将核工具变量回归推广到取值为希尔伯特空间元素（如曲线、密度、核均值嵌入）的结果变量，并证明其两阶段谱正则化估计量是在核积分算子生成的希尔伯特尺度上的正则化方法，从而在良定与误定情形下刻画了估计量收敛的目标解。

    

    我们研究结果变量取值于可分希尔伯特空间 $\mathbb{Y}$ 的非参数工具变量回归问题，结果变量例如可以是曲线、密度或分布的核均值嵌入。结构函数求解一个不适定的算子方程，其条件期望算子不必是单射的，其解也不必属于假设空间。我们分析了一种两阶段核估计量，该估计量将核工具变量回归扩展到取值为 $\mathbb{Y}$ 的结果变量，并允许在两个阶段中使用谱正则化方法，包括第二阶段中的 Landweber 迭代和 Showalter 方法。我们证明第二阶段是在由核积分算子的逆所生成的希尔伯特尺度中的一种正则化方法。这一视角明确了估计量收敛的目标：在良定情形下，该目标是再生核希尔伯特空间中的最小范数解（原文摘要在此处不完整）。

    arXiv:2611.21391v1 Announce Type: cross  Abstract: We study nonparametric instrumental variable regression when the outcome takes values in a separable Hilbert space $\mathbb{Y}$, for instance a curve, a density or the kernel mean embedding of a distribution. The structural function solves an ill-posed operator equation whose conditional expectation operator need not be injective, and whose solutions need not belong to the hypothesis space. We analyse a two-stage kernel estimator that extends kernel instrumental variable regression to $\mathbb{Y}$-valued outcomes and allows spectral regularisation in both stages, including Landweber iteration and Showalter's method in the second stage. We show that the second stage is a regularisation method in the Hilbert scale generated by the inverse of the kernel integral operator. This viewpoint identifies the solution towards which the estimator converges: the minimum-norm solution in the reproducing kernel Hilbert space in the well-specified cas
    
[^9]: 基于Dobrushin收缩与加速哈密顿蒙特卡洛的高概率一致混合

    High-probability Uniform Mixing via Dobrushin Contraction and Accelerated Hamiltonian Monte Carlo

    [https://arxiv.org/abs/2611.21305](https://arxiv.org/abs/2611.21305)

    本文针对临界阻尼Metropolis调整动能朗之万方法，同时在水条件数与维度两方面证明了混合加速性：在强凸、梯度与Hessian Lipschitz条件下，仅需 𝒪((κ^{1/2}+L_H^{1/2}m^{-3/4})d^{1/4}) 次梯度评估即可在总变差意义下混合到 ε 精度，且结果对高概率区域任意起点一致成立并适用于温启动。

    

    自哈密顿蒙特卡洛（HMC）问世并得到广泛应用以来，其动因在于相比扩散型方法，它能够在条件数依赖和维度依赖两个方面加速向平衡态的收敛。在本工作中，我们针对带有持续均匀刷新的临界阻尼Metropolis调整动能朗之万方法，同时在这两个方面建立了这种加速性。更精确地说，对于目标测度 μ(dx) ∝ e^{-U(x)}dx，其中势函数满足 m-强凸、L-梯度Lipschitz 以及 L_H-Frobenius Hessian Lipschitz 条件，我们证明该方法在总变差意义下混合到 ε-精度，所需梯度评估次数为 𝒪((κ^{1/2} + L_H^{1/2} m^{-3/4}) d^{1/4})（相差对数因子），且该结果既对处于高平衡概率区域内的所有起始点一致成立，也对温启动成立。证明过程融合了 Bou-Rabee 等人先前研究的……

    arXiv:2611.21305v1 Announce Type: cross  Abstract: Hamiltonian Monte Carlo, since its invention and subsequent rise to widespread use, has been motivated by its ability to accelerate convergence to equilibrium compared to diffusive methods, both in terms of dependence on condition number and dimension. In the present work, we establish this acceleration simultaneously in both aspects for critically damped Metropolis-adjusted kinetic Langevin with persistent uniforms. More precisely, for target measures $\mu(dx)\propto e^{-U(x)}dx$ with $m$-strongly convex, $L$-gradient Lipschitz, and $L_H$-Frobenius Hessian Lipschitz potentials, we show that the method mixes to $\varepsilon$-accuracy in total variation using $\mathcal O\bigl((\kappa^{1/2}+L_H^{1/2}m^{-3/4})\,d^{1/4}\bigr)$ gradient evaluations up to logarithmic factors, both uniformly over starting points in domains of high equilibrium probability and from warm starts. The proof incorporates ideas from previous studies by Bou-Rabee and
    
[^10]: 压缩式基于模拟的推断：利用低维草图实现海量数据的神经后验估计

    Compressed Simulation-Based Inference: Neural Posterior Estimation for Massive Data via Low-Dimensional Sketches

    [https://arxiv.org/abs/2611.21286](https://arxiv.org/abs/2611.21286)

    该论文提出压缩式基于模拟的推断方法，通过仅用 $2p+1$ 个随机低维草图即可实现后验一致性，从而避免手工设计摘要统计量的困难和学习摘要的高昂计算成本，实现海量数据下的神经后验估计。

    

    神经后验估计（NPE）是一种流行的贝叶斯基于模拟的推断（SBI）方法，它使用在模拟数据-参数对上训练的神经网络，在给定用户选择的先验条件下近似后验分布。对于高维观测数据，由于维数灾难，NPE方法通常近似参数在用户选择或学习得到的摘要统计量条件下的条件密度，而非整个数据集。然而，这两种方法都面临挑战：为复杂模型手工设计良好的摘要统计量需要科学家付出大量努力，而从海量数据集中学习摘要统计量的计算代价可能过于高昂。为解决这些挑战，我们提出近似参数在数据低维草图条件下的条件密度。我们证明，对于一个具有 $p$ 维参数的模型，仅需 $2p+1$ 个随机草图即可保证后验一致性。我们比较……（摘要此处截断）

    arXiv:2611.21286v1 Announce Type: cross  Abstract: Neural posterior estimation (NPE) is a popular Bayesian simulation-based inference (SBI) approach that uses neural networks trained on simulated data--parameter pairs to approximate the posterior given a user-chosen prior. For high-dimensional observations, NPE methods typically approximate the conditional density of the parameter given a user-chosen or learned summary statistic, rather than the entire dataset, due to the curse of dimensionality. However, both approaches face challenges: hand-crafting good summaries for complex models requires considerable effort from scientists, while learning them from massive datasets can be prohibitively computationally expensive. To address these challenges, we propose approximating the conditional density of a parameter given low-dimensional sketches of the data. We show that, for a model with a $p$-dimensional parameter, only $2p+1$ random sketches are needed for posterior consistency. We compar
    
[^11]: 参数化常微分方程的上下文估计与预测

    In-Context Estimation and Prediction for Parametric ODEs

    [https://arxiv.org/abs/2611.21268](https://arxiv.org/abs/2611.21268)

    该论文证明Transformer可以通过“粗搜索初始化+局部强凸区域梯度下降”的两阶段近似最大似然程序，从含噪轨迹观测中对一般非凸的参数化常微分方程实现上下文参数估计与轨迹预测。

    

    Transformer已经展示了执行上下文学习的能力，即无需更新自身参数，仅凭少量示例就能适应新任务。现有理论主要集中于模拟基于梯度的算法来解决凸统计问题，或建立一般性的函数逼近保证，而未能利用问题背后的统计结构。我们研究参数化常微分方程（ODE）的上下文参数估计与轨迹预测问题，其中参数估计通常是非凸的。在适当的可辨识性和正则性条件下，我们构建了一个Transformer，通过两阶段近似最大似然过程从含噪轨迹观测中估计未知系统参数：首先由粗搜索提供处于局部强凸区域内的初始化，随后执行具有局部线性收缩性的梯度下降。对于预测任务，我们将该估计器组合到后续过程中（摘要在此处截断）。

    arXiv:2611.21268v1 Announce Type: cross  Abstract: Transformers have demonstrated the ability to perform in-context learning, that is, to adapt to new tasks from a few examples without updating their parameters. Existing theory has largely focused on emulating gradient-based algorithms for convex statistical problems or establishing general function approximation guarantees without exploiting the underlying statistical structure. We study in-context parameter estimation and trajectory prediction for parametric ordinary differential equations (ODEs), where estimation is generally nonconvex. Under suitable identifiability and regularity conditions, we construct a transformer that estimates unknown system parameters from noisy trajectory observations through a two-stage approximate maximum likelihood procedure: a coarse search provides an initialization within a locally strongly convex region, followed by gradient descent with local linear contraction. For prediction, we compose this esti
    
[^12]: 固定基准下二项混合分布的方差自适应置信区域及其在大语言模型评估中的应用

    Variance-adaptive confidence regions for fixed-benchmark binomial mixtures, with application to LLM evaluation

    [https://arxiv.org/abs/2611.21266](https://arxiv.org/abs/2611.21266)

    该论文提出方差自适应（VA）置信区域方法，基于固定基准上的二项计数为经验分布线性泛函构造对任意有界泛函同时具有有限样本覆盖保证的置信区间，其可通过线性规划计算并在一阶最坏情形意义下达到最优宽度，可直接应用于大语言模型评估。

    

    我们为成功概率 $p_1,\dots,p_m$ 的经验分布 $F_m$ 的线性泛函 $\theta(F_m)=\int h\,dF_m$ 构造置信区间，数据来自独立计数 $c_i\sim\mathrm{Bin}(n,p_i)$，例如语言模型在每个问题上运行 $n$ 次的情形。对于任意的 $n$、$m$ 和固定的 $(p_i)$，覆盖率至少为 $1-\alpha$，且对所有有界的 $h$ 同时成立。方差自适应（VA）区域用它所蕴含的尾部计数的条件方差来检验每个候选混合分布。该方差关于分布是线性的，因此该区域可表示为一个线性规划，而胞腔包络使其在网格之外仍然有效。为独立同分布抽样校准的区域针对的是超总体分布，对于 $\theta(F_m)$ 而言是保守的。在利用全部 $n$ 个尾部计数、固定 $n$、连续 $h$ 且网格随 $m\to\infty$ 不断加密的条件下，VA 宽度的识别部分在一阶最坏情形意义上是最优的，与具有相同指标集的任何局部化方法相当。在预先固定的规则下……（原文摘要在此处截断）

    arXiv:2611.21266v1 Announce Type: cross  Abstract: We construct confidence intervals for linear functionals $\theta(F_m)=\int h\,dF_m$ of the empirical law $F_m$ of success probabilities $p_1,\dots,p_m$, from independent counts $c_i\sim\mathrm{Bin}(n,p_i)$, as for a language model run $n$ times per problem. Coverage is at least $1-\alpha$ for every $n$, $m$ and fixed $(p_i)$, simultaneously for all bounded $h$. The variance-adaptive (VA) region tests each candidate mixing law with the conditional variance of the tail counts it implies. This variance is linear in the law, so the region is a linear program, and cell envelopes keep it valid off-grid. Regions calibrated for iid sampling target a super-population law and are conservative for $\theta(F_m)$. With all $n$ tail counts, fixed $n$, continuous $h$ and a grid refined as $m\to\infty$, the identification part of VA's width is first-order worst-case optimal, as for any localization with the same index set. Under a rule fixed before th
    
[^13]: 无维度谱定律与谱滤波器的风险

    Dimension-free spectral laws and the risk of spectral filters

    [https://arxiv.org/abs/2611.21254](https://arxiv.org/abs/2611.21254)

    该论文突破了经典随机矩阵理论中比例渐近性的限制，在无穷维希尔伯特空间设定下建立了与维数无关的谱定律，给出了样本 Gram 矩阵经验谱定律的确定性等价物及显式非渐近误差界，为函数型数据分析和核方法中的谱滤波器风险分析提供了理论工具。

    

    样本协方差的谱性质是高维数据现代机器学习的核心。随机矩阵理论（RMT）为此提供了强大的工具；然而，大多数经典结果都假设比例渐近性，即 $d/n$ 趋于一个有限极限。在函数型数据分析和核方法中，固定的无穷维希尔伯特空间往往是观测数据更自然的设定，这超出了比例渐近性的适用范围。因此，需要为希尔伯特空间 $\mathcal{H}$ 中具有迹类总体协方差 $\Sigma$ 的独立同分布观测 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_n$ 建立与维数无关的谱理论。基于 Cheng 和 Montanari（2024）的与维数无关的岭回归分析，我们建立了两个具有显式非渐近误差界的确定性等价物。第一个用确定性的谱定律 $\mu_n 来近似样本 Gram 矩阵的经验谱定律 $\widehat{\mu}_n$…

    arXiv:2611.21254v1 Announce Type: cross  Abstract: Spectral properties of the sample covariance are key to modern machine learning with high-dimensional data. Random matrix theory (RMT) supplies powerful tools; however, most classical results assume proportional asymptotics, where $d/n$ tends to a finite limit. In functional data analysis and kernel methods, a fixed infinite-dimensional Hilbert space is often the more natural setting for the observations, beyond the scope of proportional asymptotics. This calls for a dimension-free spectral theory for i.i.d. observations $\boldsymbol{x}_1, \dots, \boldsymbol{x}_n$ in a Hilbert space $\mathcal{H}$ with trace-class population covariance $\Sigma$. Building on the dimension-free ridge regression analysis of Cheng and Montanari (2024), we establish two deterministic equivalents with explicit nonasymptotic error bounds. The first approximates the empirical spectral law $\widehat{\mu}_n$ of the sample Gram matrix by a deterministic law $\mu_n
    
[^14]: 选择示例的代价：自适应上下文学习中的上下文过拟合

    The Price of Choosing Examples: Context Overfitting in Adaptive In-Context Learning

    [https://arxiv.org/abs/2611.21251](https://arxiv.org/abs/2611.21251)

    该论文首次将提示构建形式化为自适应数据分析，提出“上下文容量”概念，证明自适应选择示例的上下文学习存在 √(κ/k) 阶的泛化过拟合代价，并给出匹配下界和 PAC-贝叶斯检索规则。

    

    上下文学习通常被分析为提示中的示例在学习器被选定之前就已采样的情形。但在实际部署的系统中并非如此：示例是在看到查询和数据存储之后才被检索、过滤、重排序、编辑并反复测试的。我们通过条件随机核演算将提示构建视为自适应数据分析，从而形式化了这一差距。我们的核心量是上下文容量，即数据存储与最终提示记录之间的互信息或近似最大信息。对于任何冻结的上下文学习器和任何容量为 κ 的自适应上下文流水线，我们证明了阶为 √(κ/k) 的有限样本上下文泛化上界，其中 k 是所选示例的数量，其损失构成所报告的分数。我们证明了匹配的下界，当 log k ≪ κ ≪ k 时其常数是精确的，并推导出两种 PAC-贝叶斯检索规则：一种先（摘要在此处被截断）……

    arXiv:2611.21251v1 Announce Type: cross  Abstract: In-context learning is usually analyzed as if the examples in the prompt were sampled before the learner is chosen. In deployed systems they are not: examples are retrieved, filtered, re-ranked, edited, and repeatedly tested after seeing the query and the data store. We formalize this gap by treating prompt construction as adaptive data analysis through a conditional stochastic-kernel calculus. Our main quantity is the context capacity, the mutual information or approximate max-information between the data store and the final prompt transcript. For any frozen in-context learner and any adaptive context pipeline with capacity $\kappa$, we prove a finite-sample context generalization bound of order $\sqrt{\kappa/k}$, where $k$ is the number of selected examples whose losses form the reported score. We prove a matching lower bound, whose constant is exact when $\log k\ll\kappa\ll k$, and derive two PAC-Bayesian retrieval rules: a first-or
    
[^15]: 自适应测试时扩展的任意时刻有效PAC-Bayes证书

    Anytime-Valid PAC-Bayes Certificates for Adaptive Test-Time Scaling

    [https://arxiv.org/abs/2611.21241](https://arxiv.org/abs/2611.21241)

    本文证明了适用于自适应选择的任意时刻有效PAC-Bayes证书，使测试时扩展系统在自适应确定采样量、提示词和验证器之后仍能获得统计有效的泛化保证，且同一保证覆盖部署前的全部方案搜索过程。

    

    测试时扩展系统通常在反复检查校准集之后，才决定采样多少、使用哪个提示词，以及信任哪个验证器或聚合规则。在这种自适应选择之后，标准的验证误差界便不再有效。我们将测试时扩展方案建模为具有随机成本的随机化预测器，并证明了一个时间一致的PAC-Bayes证书，该证书同时对所有校准时刻、所有关于方案的后验混合分布，以及所有依赖数据的停止规则成立。因此，同一事件既覆盖了部署之前进行的方案搜索，也覆盖了最终报告的固定方案。对于预算和验证器的有限网格，惩罚项是熟悉的 $\log M/n$，在最坏情况下，任何在选择之后有效的证书都无法避免该惩罚项；而非均匀先验和局部化的后验混合分布可以为结构化搜索提供更紧的证书。我们将该结果专门化到自洽性方法并展示

    arXiv:2611.21241v1 Announce Type: cross  Abstract: Test-time scaling systems often choose how much to sample, which prompt to use, and which verifier or aggregation rule to trust after repeatedly inspecting a calibration set. Standard validation error bars are not valid after this adaptive selection. We model a test-time scaling recipe as a randomized predictor with a random cost and prove a time-uniform PAC-Bayes certificate that holds simultaneously over all calibration times, all posterior mixtures over recipes, and all data-dependent stopping rules. Thus the same event covers the recipe search that precedes deployment, not merely the fixed recipe finally reported. For a finite grid of budgets and verifiers, the penalty is the familiar $\log M/n$, which in the worst case no certificate valid after selection can avoid, while non-uniform priors and localized posterior mixtures can give sharper certificates for structured searches. We specialize the result to self-consistency and show 
    
[^16]: 基于互信息的仿真推断中摘要统计量充分性研究

    Summary Sufficiency in Simulation-Based Inference via Mutual Information

    [https://arxiv.org/abs/2611.21158](https://arxiv.org/abs/2611.21158)

    该论文针对仿真推断中摘要统计量充分性的互信息估计歧义问题，提出一种可处理的方法，以区分较小的BA下界究竟源于摘要统计量不充分还是密度估计器不准确。

    

    仿真推断使得在似然函数不可处理但可以从模型中模拟数据的情况下进行后验估计成为可能。对于高维或可变长度的数据，仿真推断方法通常依赖低维摘要统计量作为后验估计的条件输入，这使得摘要统计量的充分性对准确推断至关重要。互信息为充分性提供了一种自然的信息论刻画，用于量化摘要统计量保留了多少参数信息。然而，在无似然设置中，这些互信息量通常是不可计算的。Sui等人（2026）使用Barber–Agakov（BA）下界从训练好的神经后验中估计互信息，同时讨论了但并未计算Fisher信息上界。这使得估计结果存在歧义：较小的下界可能意味着摘要统计量不充分，也可能意味着密度估计器不准确。我们通过一种可处理的（trac……此处摘要截断）方法来解决这一局限。

    arXiv:2611.21158v1 Announce Type: cross  Abstract: Simulation-based inference (SBI) enables posterior estimation when the likelihood is intractable but data can be simulated from the model. For high-dimensional or variable-length data, SBI methods typically rely on low-dimensional summary statistics as conditioning inputs for posterior estimation, making their sufficiency crucial for accurate inference. Mutual information (MI) provides a natural information-theoretic characterization of sufficiency, quantifying how much parameter information a summary retains. However, these MI quantities are generally intractable in likelihood-free settings. Sui et al. (2026) estimate MI from a trained neural posterior using the Barber--Agakov (BA) lower bound, while discussing, but not computing, a Fisher-information upper bound. This makes the estimate ambiguous: a small lower bound may indicate either an insufficient summary or an inaccurate density estimator. We address this limitation with a trac
    
[^17]: 少步生成中闭式耦合的局限性

    The Limits of Closed-Form Couplings for Few-Step Generation

    [https://arxiv.org/abs/2611.21119](https://arxiv.org/abs/2611.21119)

    研究发现高维情况下扩散过程的闭式解析解几乎退化为线性变换，因此无需预训练、直接利用闭式耦合进行少步生成的方法在实践中不可行。

    

    基于流模型的少步生成器需要复杂的技术方案，例如预训练的流模型或一致性损失。尽管整流流因其在少步采样中的成功而广受欢迎，但其代价是需要一个预训练模型，并且需要为每个数据集采样大量的耦合。然而，对于有限数据集和高斯概率路径，经验分布的概率流向量场存在闭式解，因此人们可以将该场作为教师，完全跳过预训练。我们证明，令人惊讶的是，这种方法对于具有实际意义的问题是不可能实现的。特别地，高维空间中扩散过程的解析解几乎变成一个线性变换。积分的结果返回的是初始化时抽取的噪声向量，加上一个在经验上表现为数据点微弱副本的漂移项，且该漂移几乎与噪声无关。因此该耦合……

    arXiv:2611.21119v1 Announce Type: cross  Abstract: Few-step generators via flow models require complicated technical solutions such as a pretrained flow model or a consistency loss. While rectified flows have gained popularity for their success in few-step sampling, they pay a cost of needing a pre-trained model and sampling a large number of couplings per dataset. Yet for a finite dataset and a Gaussian probability path, the probability-flow vector field of the empirical distribution is available in closed form, so one could use this field as the teacher and skip pretraining altogether. We demonstrate that, surprisingly, this approach is impossible for problems of practical relevance. In particular, the analytic solution of the diffusion process in high dimensions becomes almost a linear transformation. The integration returns the noise vector drawn at initialisation, plus a drift that is empirically a faint copy of the data point, almost independent of the noise. The coupling therefo
    
[^18]: CONGA：为结构化贝叶斯推断编译高斯纤维几何

    CONGA: Compiling Gaussian Fiber Geometry for Structured Bayesian Inference

    [https://arxiv.org/abs/2611.20995](https://arxiv.org/abs/2611.20995)

    该论文提出CONGA编译器，通过识别结构化贝叶斯模型中的条件高斯潜变量块，并利用精确仿射传输映射（Brenier和Knothe-Rosenblatt映射）将其与非高斯核心分离，从而将KL与相对Fisher推断目标约化到低维核心上以提升后验推断效率。

    

    在结构化贝叶斯模型中，一个较大的潜变量块在给定较小参数集的条件下通常服从高斯分布，然而通用的推断方法往往需要在预热阶段重新学习这一几何结构。我们研究如何利用这一结构来构建用于后验推断的传输映射。条件均值和精度矩阵确定了精确的仿射映射，包括对称Brenier映射和三角形Knothe-Rosenblatt映射，它们将一个标准正态块与其余的非高斯核心分离开来。KL散度和相对Fisher目标随之约化为该核心上的目标函数。我们提出了CONGA——一个编译器，它识别候选的高斯块，并根据其结构与计算成本选择时间型、谱型或稀疏型实现方式。边缘化需要从模型本身建立恒等式；仅有数值证据时，只允许进行保持目标分布的变量替换，而检查失败时模型保持不变。在Stock-Watson宏观数据（摘要原文在此处截断）

    arXiv:2611.20995v1 Announce Type: cross  Abstract: In structured Bayesian models, a large latent block is often Gaussian conditional on a smaller set of parameters, yet generic inference methods relearn this geometry during warmup. We study how this structure can be used to construct transports for posterior inference. The conditional mean and precision determine exact affine maps, including symmetric Brenier and triangular Knothe-Rosenblatt maps, that separate a standard normal block from the remaining non-Gaussian core. KL and relative Fisher objectives then reduce to objectives on this core. We introduce CONGA, a compiler that identifies candidate Gaussian blocks and selects a temporal, spectral, or sparse implementation according to their structure and computational cost. Marginalization requires an identity established from the model; numerical evidence alone permits only a target-preserving change of variables, and a failed check leaves the model unchanged. On the Stock-Watson in
    
[^19]: FARCH：用于校准预测带的函数型ARCH模型

    FARCH: A Functional ARCH Model for Calibrated Prediction Bands

    [https://arxiv.org/abs/2611.20972](https://arxiv.org/abs/2611.20972)

    提出FARCH模型，通过在曲线及其平方残差上复用同一个基于小波半度量的核加权方案，联合非参数地预测函数型数据的条件均值与波动率，从而构建出逐点和一致（上确界范数）两种校准预测带。

    

    预测一条完整的曲线，而非其少数几个标量摘要，需要一种本身即为函数型的不确定性概念。我们提出FARCH——一种具有条件异方差的函数型非线性自回归模型，其中下一条曲线的条件均值和条件波动率通过重复使用同一个核加权方案进行联合非参数预测：该方案建立在曲线之间的小波半度量之上，先应用于曲线本身，再应用于其平方残差。基于这一对估计量，我们推导出两种预测带的构造：一种是逐点预测带，由边际情景分位数获得；另一种是关于上确界范数的一致预测带，为涉及整条预测路径而非任何单一时刻的决策进行校准。该方法应用于1996至2009年法国全国半小时电力需求的日前预测任务，并在完全相同的实验协议下与四种现有方法进行了比较……

    arXiv:2611.20972v1 Announce Type: cross  Abstract: Forecasting a curve, rather than a handful of scalar summaries of it, requires a notion of uncertainty that is itself functional. We propose FARCH, a functional nonlinear autoregressive model with conditional heteroscedasticity, in which the conditional mean and the conditional volatility of the next curve are forecasted jointly and nonparametrically, by reusing a single kernel weighting scheme --- built on a wavelet semi-metric between curves --- first on the curves themselves and then on their squared residuals. From this pair of estimators we derive two constructions of prediction bands: a pointwise band, obtained from marginal scenario quantiles, and a uniform, sup-norm band, calibrated for decisions that concern the entire forecasted path rather than any single instant. Applied to day-ahead forecasting of half-hourly French national electricity demand (1996--2009), and compared, under an identical protocol, against four existing p
    
[^20]: 线性偏微分方程随机特征方法的收敛性分析

    A Convergence Analysis of the Random Feature Method for Linear PDEs

    [https://arxiv.org/abs/2611.20951](https://arxiv.org/abs/2611.20951)

    本文在核方法框架下对随机特征方法求解线性偏微分方程建立了系统的收敛性理论，将有限随机特征逼近与RKHS估计问题归结为核岭回归，并基于积分算子特征值给出了误差的上下界。

    

    arXiv:2611.20951v1 公告类型： cross 摘要：随机特征方法（RFM）凭借其卓越的精度和计算效率，在求解偏微分方程（PDE）的机器学习方法中脱颖而出。然而，RFM的理论基础尚未得到完全阐明。本文在核方法框架下对RFM进行了全面分析，重点关注两个主要方面：（i）利用有限个随机特征在Sobolev范数和PDE残差半范数意义下的逼近；（ii）基于有限个无噪声配置点观测，在诱导出的再生核希尔伯特空间（RKHS）上进行估计。我们将这两个问题与核岭回归（KRR）联系起来，并在相应假设下依据相关积分算子的特征值建立了上下界。结合逼近下界与估计下界，我们在随机特征数量与配置点数量同时受限的条件下建立了一个下界。

    arXiv:2611.20951v1 Announce Type: cross  Abstract: The Random Feature Method (RFM) distinguishes itself among machine learning approaches to partial differential equations (PDEs) through its superior accuracy and computational efficiency. Nevertheless, the theoretical underpinnings of RFM remain incompletely elucidated. This paper provides a comprehensive analysis of RFM within the framework of kernel methods, focusing on two principal aspects: (i) approximation by finitely many random features in Sobolev norms and the PDE residual seminorm, and (ii) estimation over the induced reproducing kernel Hilbert space (RKHS) from finitely many noiseless collocation observations. We relate both problems to kernel ridge regression (KRR) and establish upper and lower bounds in terms of the eigenvalues of the associated integral operators under the respective assumptions. Combining the approximation and estimation lower bounds, we establish a lower bound under simultaneous restrictions on the numb
    
[^21]: 基于图表示学习的混油段长度预测增强方法

    Enhancing Mixed Oil Length Prediction through Graph Representation Learning

    [https://arxiv.org/abs/2611.20660](https://arxiv.org/abs/2611.20660)

    该论文提出SDI-GRL模型，通过将混油样本和管道特征构建为图节点并利用消息传递机制，捕捉样本间的统计依赖性，从而显著提升成品油管道运输中混油段长度的预测精度。

    

    在成品油管道运输过程中，不可避免形成的混油段给管道网络带来了巨大的经济挑战。因此，准确预测这些混油段的长度至关重要。尽管先前的方法通过结合机器学习和深度学习算法已实现相对准确的预测，但它们忽略了现实世界中混油样本之间依然存在的统计依赖性，从而降低了模型精度。为解决这一问题，我们提出了一种新颖的样本依赖性融合图表示学习（SDI-GRL）模型。通过将样本和管道特征表示为图中的节点，SDI-GRL通过共享特征连接相互依赖的样本，使消息传递能够在相关样本之间传播信息。这种基于图的框架使我们的模型能够捕捉到先前工作所忽略的样本间统计依赖性。

    arXiv:2611.20660v1 Announce Type: new  Abstract: During the pipeline transportation of refined oil products, the inevitable formation of mixed oil segments poses considerable economic challenges to pipeline networks. Therefore, it is important to accurately predict the length of these mixed oil segments. Although previous methods have achieved relatively accurate predictions by integrating machine and deep learning algorithms, they ignore the statistical dependencies that remain among real-world mixed oil samples, which degrades model accuracy. To address this issue, we propose a novel Sample Dependencies Incorporated Graph Representation Learning (SDI-GRL) model. By representing samples and pipeline features as nodes of a graph, SDI-GRL connects dependent samples through their shared features, allowing message passing to propagate information across related samples. This graph-based framework allows our model to capture the statistical dependencies among samples absent in prior works.
    
[^22]: 从重叠非线性观测中进行校准泛函预测

    Calibrated Functional Prediction from Overlapping Nonlinear Observations

    [https://arxiv.org/abs/2611.20567](https://arxiv.org/abs/2611.20567)

    该论文提出基于Schur有理算子与局部高斯投影的共享稀疏核方法，结合半正定特征表示、正矩阵积分与排序校准扰动界，实现了从重叠非线性观测出发、具有完整有限池钉球风险保证的校准泛函预测。

    

    我们研究从共享隐藏表状态的标量观测中预测非线性审计泛函。对于Schur有理算子，局部高斯投影为重叠任务、重复行标识以及跨行依赖产生一个共同的稀疏核。共享状态残差样本通过半正定特征表示扩展了这一核。正矩阵积分和样本值区间将原始输入与实际保存的混合预测连接起来。随后，排序校准角色扰动界为其完整的有限池钉球风险提供了证明。独立的完全损失选择确定岭参数、分数尺度和残差权重，而任务秩校准则提供一个上端点。评估在每个数据集上使用二十个新参考上下文，涵盖两个任务规模，在十个基准总体上共给出400个条件。平均数据集完全损失比相对于二次核为0.9977……（原文摘要在此处截断）

    arXiv:2611.20567v1 Announce Type: new  Abstract: We study prediction of nonlinear audit functionals from scalar observations of a shared hidden table state. For Schur-rational operators, local Gaussian projections yield a common sparse kernel for overlapping tasks, repeated row identities, and cross-row dependence. Shared-state residual samples extend this kernel through a positive-semidefinite feature representation. Positive matrix integration and sample-value enclosures connect the original inputs to the actual saved hybrid prediction. A sorted calibration-role perturbation bound then certifies its complete finite-pool pinball risk. Independent complete-loss selection chooses the ridge, score scale, and residual weight, while task-rank calibration supplies an upper endpoint. Evaluation uses twenty new reference contexts per dataset at two task sizes, giving 400 conditions across ten benchmark populations. The mean dataset complete-loss ratios are 0.9977 to the degree-two kernel, 0.9
    
[^23]: 变量选择前的去相关：左变换与右变换及其权衡

    Decorrelation Before Variable Selection: Left and Right Transforms and Their Trade-Off

    [https://arxiv.org/abs/2611.20420](https://arxiv.org/abs/2611.20420)

    该论文揭示了变量选择前去相关的左变换（保持系数、改变噪声）与右变换（保持噪声、改变系数）之间的权衡，指出当真实变量排前时Cholesky白化能保持噪声变量不活跃，而同时保留系数与噪声的去相关则会损失信息。

    

    去相关存在两种形式。左变换保持系数不变而改变噪声；右变换保持噪声不变而改变系数。当真实变量排在前面时，仅使用Cholesky白化即可使噪声变量保持不活跃；而同时保留系数与噪声的去相关则会损失信息。

    arXiv:2611.20420v1 Announce Type: cross  Abstract: Decorrelation has two forms. A left transform keeps coefficients and changes noise; a right transform keeps noise and changes coefficients. Cholesky whitening alone keeps noise variables inactive when true ones come first; decorrelation preserving coefficients and noise loses information.
    
[^24]: 非对齐有序响应的高维迁移学习

    High-dimensional transfer learning with unaligned ordinal responses

    [https://arxiv.org/abs/2611.20411](https://arxiv.org/abs/2611.20411)

    该论文针对源域与目标域有序响应类别不对齐的高维迁移学习问题，提出了样本分割和全目标两种算法，实现了参数估计与超额风险的极小化极大最优速率，并在医学数据上验证了有效性。

    

    在这项工作中，我们研究了具有非对齐有序响应的高维迁移学习问题，允许源域和目标域拥有不同的回归系数、累积连接和阈值。我们开发了样本分割和全目标两种算法，它们先合并源域和目标域的似然，再估计目标域校正项。在适当条件下，我们的算法在参数估计和超额风险方面均达到极小化极大最优速率。仿真结果表明，在有限样本估计和预测任务中，全目标算法优于样本分割算法。在一个医学数据集上的应用进一步表明，当从五分类源数据向二分类住院死亡率迁移信息时，我们的全目标算法相比仅使用目标数据和简单合并数据的基线方法均取得了更好的预测效果。

    arXiv:2611.20411v1 Announce Type: cross  Abstract: In this work, we study high-dimensional transfer learning with unaligned ordinal responses, allowing different domain-specific regression coefficients, cumulative links and thresholds. We develop the sample-split and full-target algorithms that pool source and target likelihoods and then estimate a target correction. Under suitable conditions, our algorithms achieve minimax-optimal rates for parameter estimation and excess risk. Simulations favor the full-target algorithm over the sample-split algorithm for finite-sample estimation and prediction. An application to a medical dataset further demonstrates the predictive gains of our full-target algorithm over the target-only and pooled baselines when transferring information from the five-category source data to binary hospital mortality.
    
[^25]: 基于赌博机反馈的广义预测集

    Generalized Prediction Set with Bandit Feedback

    [https://arxiv.org/abs/2611.20387](https://arxiv.org/abs/2611.20387)

    提出BanditGPS，首次在赌博机反馈的在线学习设置下实现结合OOD检测的集合值预测，通过对真实类别标签的估计，生成小而信息丰富的预测集并提升OOD检测性能。

    

    在充满不确定性的高风险环境中，集合值预测通过为每个测试实例给出多个潜在标签作为预测，提供了一种谨慎且稳健的机制，以降低预测错误带来的潜在风险。然而，在诸如带赌博机反馈的在线学习等场景下，将这一范式与分布外（OOD）检测相结合的研究仍然很少。赌博机反馈机制只告知学习者所选择的臂/动作是否正确，而不是显式的真实标签，因此当采取错误动作时，真实的类别标签是未知的。为应对这一挑战，我们提出了BanditGPS，它在赌博机反馈设置下进行带OOD检测的集合值预测，并利用对类别标签真实值的估计。BanditGPS实现了三个目标：给出小而富有信息量的预测集、提升OOD检测性能，以及

    arXiv:2611.20387v1 Announce Type: cross  Abstract: In high-stakes environments where uncertainties abound, set-valued prediction offers a cautious and robust mechanism by presenting multiple potential labels as the prediction for each test instance to mitigate the potential risk associated with prediction errors. Yet, integrating this paradigm with out-of-distribution (OOD) detection remains scarcely explored in such settings as online learning with bandit feedback. The bandit feedback mechanism informs the learner about the correctness of the pulled arm/action instead of the explicit ground truth label, leaving the true class label unknown when an incorrect action is taken. To address this challenge, we introduce BanditGPS which conducts set-valued prediction with OOD detection in the bandit feedback setting, using an estimation to the ground truth of class labels. BanditGPS achieves three objectives: render small/informative prediction sets, enhance the OOD detection performance, and
    
[^26]: 衡量预测模型中的连贯性

    Measuring Coherence in Predictive Models

    [https://arxiv.org/abs/2611.20325](https://arxiv.org/abs/2611.20325)

    本文提出了基于MMD的“不连贯性比率”来衡量模型更新过程偏离连贯性（即当前预测与更新后预期预测的一致性）的程度，并刻画了不连贯性何时会影响诸如主动学习和贝叶斯实验设计等顺序决策。

    

    许多顺序决策过程（如主动学习）隐含地假设模型的当前预测与它在根据新数据进行更新后预期做出的预测相一致，我们将这一性质称为连贯性。大多数实际使用的模型并不通过精确的条件化进行更新，因此不保证具有连贯性，这可能导致更差的决策。然而，目前尚无通用的方法来衡量不连贯性，也无法判断它何时会产生影响。我们开发了衡量不连贯性的工具，并刻画了它何时会影响决策。我们提出了“不连贯性比率”，这是一种基于MMD的度量方法，用于衡量任何更新过程偏离连贯性的程度，且仅从该过程自身的预测和更新中计算得出，同时我们还提供了一个随时有效的连贯性检验方法。在贝叶斯实验设计中，对于连贯模型而言等价的采集目标，在我们研究的每一个近似模型下都会选择不同的数据，而不连贯性比率能够追踪这种……

    arXiv:2611.20325v1 Announce Type: cross  Abstract: Many sequential decision-making procedures, such as active learning, implicitly assume that a model's current predictions agree with those it expects to make after updating on new data, a property we call coherence. Most models in use do not update by exact conditioning and are not guaranteed to be coherent, which can lead to worse decisions. Yet there is currently no general way to measure incoherence or to tell when it matters. We develop tools to measure incoherence and characterise when it affects decisions. We introduce the incoherence ratio, an MMD-based measure of how far any updating procedure departs from coherence, computed only from the procedure's own predictions and updates, and accompany it with an anytime-valid test of coherence. In Bayesian experimental design, acquisition objectives that are equivalent for coherent models choose different data under every approximate model we study, and the incoherence ratio tracks how
    
[^27]: 流形指数模型：可证明且高效地避免统计与计算上的维数灾难

    The Manifold-Index Model: provably and efficiently avoiding the statistical and computational curse of dimensionality

    [https://arxiv.org/abs/2611.20275](https://arxiv.org/abs/2611.20275)

    提出流形指数模型及MANOR算法，通过对相似响应的预测变量进行多尺度PCA估计切平面，使预测误差达到仅依赖流形内在维度 $m$ 的最优非参数速率 $n^{-4/(m+4)}$，可证明地克服了环境维度带来的维数灾难。

    

    我们研究预测变量分布在 $\mathbb{R}^D$ 中未知 $m$ 维流形 $\mathcal{M}$ 周围管状区域内的向量值回归问题。在流形指数模型中，平均响应为 $F=f\circ\Pi_{\mathcal{M}}$，其中 $\Pi_{\mathcal{M}}$ 是到流形 $\mathcal{M}$ 的最近点投影，$f:\mathcal{M}\to\mathbb{R}^p$ 是一个可能非单射的连接函数。多尺度各向异性局部回归（MANOR）通过对具有相似响应的预测变量进行多尺度主成分分析（PCA）来估计切平面，然后通过汇集局部斜率的外积对其进行精细化，使邻域沿切向收缩的同时保留完整的法向纤维。我们假设连接函数的微分满足 Lipschitz 条件且响应噪声有界。当 $p\ge m$ 时，在我们的几何与采样条件下，$n$ 个独立观测可给出 $n^{-2/(m+4)}$ 阶的平均切空间误差，以及达到 $m$ 维非参数速率 $n^{-4/(m+4)}$ 的积分平方预测误差。

    arXiv:2611.20275v1 Announce Type: cross  Abstract: We study vector-valued regression with predictors filling a tube around an unknown $m$-dimensional manifold $\mathcal{M}$ in $\mathbb{R}^D$. In the Manifold-Index Model, the mean response is $F=f\circ\Pi_{\mathcal{M}}$, where $\Pi_{\mathcal{M}}$ is the nearest-point projection onto $\mathcal{M}$ and $f:\mathcal{M}\to\mathbb{R}^p$ is a possibly non-injective link function. Multiscale ANisotropic lOcal Regression (MANOR) estimates tangent planes by multiscale PCA of predictors with similar responses, then refines them by pooling outer products of local slopes. The neighborhoods narrow tangentially while retaining complete normal fibers. We assume a Lipschitz link differential and bounded response noise. For $p\ge m$, under our geometric and sampling conditions, $n$ independent observations give mean tangent-space error of order $n^{-2/(m+4)}$ and integrated squared prediction error at the $m$-dimensional nonparametric rate $n^{-4/(m+4)}$
    
[^28]: 将潜在块模型集成到变分自编码器中以实现解耦表示学习

    Integrating Latent Block Models into Variational Autoencoders for Disentangled Representation Learning

    [https://arxiv.org/abs/2611.20253](https://arxiv.org/abs/2611.20253)

    LBM-VAE通过将潜在块模型集成到变分自编码器中，利用块结构的潜在表示实现观测数据与潜在维度的联合聚类，并允许变化因素跨潜在维度组编码，从而有效支持解耦表示学习。

    

    我们提出了LBM-VAE，这是一个将潜在块模型集成到变分自编码器中以联合对观测数据和潜在维度进行聚类的新型框架。除了聚类能力之外，LBM-VAE还提供了一种块结构的潜在表示，天然支持解耦表示学习。与传统方法将每个变化因素与单个潜在维度相关联不同，我们的框架允许变化因素在潜在维度组上进行编码。实验结果证明了LBM-VAE具有出色的聚类性能，同时在解耦表示学习方面取得了具有竞争力的结果。LBM-VAE凸显了将协同聚类与变分推断相结合以学习结构化和可解释潜在表示的潜力。

    arXiv:2611.20253v1 Announce Type: cross  Abstract: We introduce LBM-VAE, a novel framework that integrates Latent Block Models (LBMs) into Variational Autoencoders (VAEs) to jointly cluster observations and latent dimensions. Beyond its clustering capabilities, LBM-VAE provides a block-structured latent representation that naturally supports disentangled representation learning. In contrast to conventional approaches that associate each factor of variation with an individual latent dimension, our framework allows factors of variation to be encoded across groups of latent dimensions. Experimental results demonstrate the promising clustering performance of LBM-VAE, while achieving competitive results in disentangled representation learning. LBM-VAE highlights the potential of combining co-clustering and variational inference to learn structured and interpretable latent representations.
    
[^29]: 学习顺序能否解释遗忘？李括号作为无需旧数据的灾难性遗忘早期预警信号

    Does Learning Order Explain Forgetting? The Lie Bracket as an Early-Warning Signal for Catastrophic Forgetting Without Old Data

    [https://arxiv.org/abs/2611.20087](https://arxiv.org/abs/2611.20087)

    本文实证验证了两个任务梯度场的李括号决定了训练顺序导致的权重差异并与灾难性遗忘相关，且仅凭旧任务训练结束时存储的梯度和前20个Hessian特征对，就能在无需旧数据的条件下早期预测遗忘，效果优于EWC惩罚项。

    

    先在任务A上训练、再在任务B上训练的网络，其最终权重与先训练B再训练A的网络不同。在主导阶上，这一差距由两个任务梯度场的李括号（Lie bracket）决定，Dherin (2023) 猜想非零的李括号可能与灾难性遗忘相关。本文对这一猜想进行了实证检验。在Split MNIST、Split Fashion-MNIST 和 Split CIFAR-10 上，任务A的遗忘几乎恰好等于两种训练顺序下A的损失之差（Spearman rho = 0.96-1.00）。随后，作者进一步探究李括号能否在任务B训练早期、且不使用A的任何数据的情况下预测遗忘——仅利用A训练结束时存储的小型摘要信息：其梯度及其前20个Hessian特征对。当旧任务已被充分学习且遗忘较慢（任务增量式学习）时，这个存储的李括号对遗忘的排序相关系数达到 rho = 0.70-0.81，而 Elastic Weight Consolidation（EWC）惩罚项仅为 0.33-0.58，并且匹配或超过（原文在此处截断）。

    arXiv:2611.20087v1 Announce Type: new  Abstract: A network trained on task A and then task B ends at different weights than one trained on B and then A. To leading order this gap is set by the Lie bracket of the two tasks' gradient fields, and Dherin (2023) conjectured that a non-vanishing bracket may be linked to catastrophic forgetting. I test this conjecture empirically. Across Split MNIST, Split Fashion-MNIST and Split CIFAR-10, the forgetting of A is almost exactly the difference in A's loss between the two training orders (Spearman rho = 0.96-1.00). I then ask whether the bracket can predict that forgetting early in training on B and without any of A's data, using only a small summary stored when A finishes: its gradient and its top 20 Hessian eigenpairs. When the old task is well learned and forgetting is slow (task-incremental), this stored bracket ranks forgetting with rho = 0.70-0.81, against 0.33-0.58 for the penalty of Elastic Weight Consolidation, and matches or exceeds re
    
[^30]: 协变量偏移下的加权保形预测：过程极限、估计权重与效率

    Weighted conformal prediction under covariate shift: process limits, estimated weights and efficiency

    [https://arxiv.org/abs/2611.20052](https://arxiv.org/abs/2611.20052)

    本文建立了协变量偏移下加权保形预测中自归一化加权经验过程的核心极限定理，证明收敛当且仅当 $\chi^2$ 散度有限（而非 $\sup v$），由此导出跨水平覆盖率的联合分布、一致性 bootstrap、渐近精确的膨胀系数与 Berry-Esseen 速率，并揭示用校准协变量估计的权重可能比 oracle 权重更高效。

    

    加权分裂保形预测通过用似然比 $v=dQ_X/dP_X$ 对校准分数重新加权来处理协变量偏移；其训练条件覆盖率是自归一化加权经验分布函数 $\hat F_m$ 的泛函，本文研究了过程 $\sqrt m(\hat F_m-F_Q)$。当且仅当 $\chi^2(Q\|P)<\infty$ 时（对非退化分数而言），该过程在 $\ell^\infty$ 中收敛到高斯过程 $W(G(t))-F_Q(t)W(K^2)$，其中 $G(t)=\mathbb{E}_Q[v\mathbf{1}\{S\le t\}]$ 且 $K^2=1+\chi^2(Q\|P)$；而 $\sup v$ 不起任何作用。该极限给出了跨置信水平的覆盖率的联合分布、一致的 bootstrap 方法、在水平上一致渐近精确的膨胀系数，以及 Berry-Esseen 型收敛速率。在使用估计比率的情况下，只要估计使用了校准协变量，一致展开仍然有效；估计误差通过一个协方差项进入，该协方差项可使方差降低到低于 oracle 权重的方差水平，此外还有成对的 boot……（摘要截断）

    arXiv:2611.20052v1 Announce Type: cross  Abstract: Weighted split conformal prediction handles covariate shift by reweighting calibration scores with the likelihood ratio $v=dQ_X/dP_X$; its training-conditional coverage is a functional of the self-normalized weighted empirical distribution function $\hat F_m$, and we study the process $\sqrt m(\hat F_m-F_Q)$. It converges in $\ell^\infty$ to the Gaussian process $W(G(t))-F_Q(t)W(K^2)$, with $G(t)=\mathbb{E}_Q[v\mathbf{1}\{S\le t\}]$ and $K^2=1+\chi^2(Q\|P)$, if and only if $\chi^2(Q\|P)<\infty$ (for non-degenerate scores); $\sup v$ plays no role. The limit gives the joint law of the coverage across levels, consistent bootstraps, inflations asymptotically exact uniformly in the level, and Berry-Esseen-type rates. With estimated ratios, a uniform expansion remains valid when the estimate uses the calibration covariates; the estimation error enters through a covariance term that can reduce the variance below the oracle's, and a pairs boot
    
[^31]: 基于离散化梯度流的高斯位置混合NPMLE认证算法

    Certified Algorithms for Gaussian Location Mixture NPMLEs via Discretized Gradient Flows

    [https://arxiv.org/abs/2611.19882](https://arxiv.org/abs/2611.19882)

    该论文提出了一种基于Fisher-Rao梯度流熵邻近离散化的认证算法，能够在与样本量近线性、与 $1/\varepsilon$ 多项式相关的时间内计算高斯位置混合NPMLE的 $\varepsilon$-近似解，且所得估计具有精确有限维表示并附带似然差距保证。

    

    我们通过Fisher-Rao梯度流研究任意维度 $d\ge 1$ 下高斯位置混合的非参数极大似然估计（NPMLE）的近似计算问题。我们提出了一种基于该流熵邻近离散化的认证算法，在给定足够小的 $\varepsilon > 0$ 时，该算法返回一个具有精确有限维表示且满足 $\hat L_n-L_n(f_{\tilde\mu})\le\varepsilon$ 的绝对连续混合测度 $\tilde\mu$。对于几乎每个满足 $n>2(d+1)(m+1)^2$ 的数据集 $X_1,\cdots, X_n \in \mathbb{R}^d$（其中 $m$ 为NPMLE的支撑点数量），该算法的运行时间为 $O(n\varepsilon^{-1}\log^{d+2}(1/\varepsilon))$。在此过程中，我们还建立了关于连续时间Fisher-Rao流的适定性与收敛性质的结果，特别是将流初始化为包含数据（原文此处截断）……

    arXiv:2611.19882v1 Announce Type: cross  Abstract: We study the computation of approximate nonparametric maximum likelihood estimators (NPMLEs) for Gaussian location mixtures in arbitrary dimension $d\ge 1$ via the Fisher-Rao gradient flow. We provide a certified algorithm based on an entropic proximal discretization of the flow which, given a sufficiently small $\varepsilon > 0$, returns an absolutely continuous mixing measure $\tilde\mu$ admitting an exact finite-dimensional representation and satisfying $\hat L_n-L_n(f_{\tilde\mu})\le\varepsilon$. For almost every dataset $X_1,\cdots, X_n \in \mathbb{R}^d$ satisfying $n>2(d+1)(m+1)^2$, where $m$ is the number of support points of the NPMLE, the algorithm requires $O(n\varepsilon^{-1}\log^{d+2}(1/\varepsilon))$ time. En route, we also establish results on the well-posedness and convergence properties of the continuous-time Fisher--Rao flow. In particular, initializing the flow at the uniform probability measure on a ball containing t
    
[^32]: Q-学习的精确高斯逼近

    Sharp Gaussian Approximation for Q-Learning

    [https://arxiv.org/abs/2611.19850](https://arxiv.org/abs/2611.19850)

    本文为带Polyak-Ruppert平均的异步Q-学习在Wasserstein距离下建立了$\widetilde{O}(n^{-1/4})$的精确非渐近高斯逼近速率及匹配下界，并揭示了马尔可夫噪声偏差与有限样本协方差偏差之间的权衡如何决定收敛速率。

    

    高斯逼近为强化学习中的在线统计推断提供了基础。我们为采用Polyak-Ruppert平均的异步Q-学习建立了Wasserstein距离下的精确非渐近高斯逼近速率。我们证明了$\widetilde{O}(n^{-1/4})$的上界，其中$n$为迭代次数，并同时给出了在对数因子意义下与之匹配的下界。由于Bellman算子的非线性，Q-学习比线性随机逼近更难分析；而异步更新则因马尔可夫噪声带来了进一步的挑战。我们的分析将Bellman非线性、马尔可夫噪声以及有限样本协方差偏差这三类误差分离开来。特别地，我们证明即使Bellman算子是线性的，马尔可夫噪声也会引入不可避免的偏差。该偏差与协方差偏差之间的权衡决定了收敛速率。（摘要在此处截断）

    arXiv:2611.19850v1 Announce Type: cross  Abstract: Gaussian approximation provides a foundation for online statistical inference in reinforcement learning. We establish sharp non-asymptotic Gaussian approximation rates in Wasserstein distance for asynchronous Q-learning with Polyak-Ruppert averaging. We prove an upper bound of $\widetilde{O}(n^{-1/4})$, where $n$ is the number of iterations, together with a matching lower bound up to logarithmic factors. Q-learning is more challenging to analyze than linear stochastic approximation due to the nonlinearity of the Bellman operator. Asynchronous updates introduce further challenges due to Markovian noise. Our analysis separates the errors from Bellman nonlinearity, Markovian noise, and the finite-sample covariance discrepancy. In particular, we show that Markovian noise induces an unavoidable bias even when the Bellman operator is linear. The trade-off between this bias and the covariance discrepancy determines the convergence rate. We fu
    
[^33]: 条件资产定价Transformer

    Conditional Asset Pricing Transformer

    [https://arxiv.org/abs/2611.19783](https://arxiv.org/abs/2611.19783)

    本文提出条件资产定价Transformer（CAP-T），通过两阶段注意力框架将超额收益分解为条件错误定价与因子风险补偿等成分，实现对收益更准确的预测与理论保证。

    

    在本文中，我们提出了一种条件资产定价Transformer（CAP-T），它通过注意力机制捕捉企业之间的非线性交互作用，以预测超额收益。在第一阶段，一个Transformer估计条件期望收益，并对其残差进行核加权主成分分析以估计局部因子载荷空间。将条件均值投影到该空间上，可以将因子风险补偿与条件错误定价分离开来。随后，在第二阶段，使用独立的Transformer分别学习这两个组成部分，其预测结果相结合用于预测收益。由此，所提出的框架将收益分解为条件错误定价、因子风险补偿、中心化因子成分和不可预测的新息项。我们建立了近似界和泛化界，并刻画了第一阶段的预测误差如何通过局部因子估计和成分学习进行传播，从而提供了统计……（摘要在此处截断）

    arXiv:2611.19783v1 Announce Type: cross  Abstract: In this paper, we propose a conditional asset pricing transformer (CAP-T) that captures nonlinear interactions among firms through attention to forecast excess returns. In the first stage, a transformer estimates conditional expected returns, and kernel-weighted principal component analysis of its residuals estimates the local factor loading space. Projecting the conditional mean onto this space separates factor risk compensation from conditional mispricing. Then, in the second stage, separate transformers learn these two components, whose predictions are combined to forecast returns. As a result, the proposed framework decomposes returns into conditional mispricing, factor risk compensation, a centered factor component, and an unpredictable innovation. We establish approximation and generalization bounds and characterize how first-stage prediction errors propagate through local factor estimation and component learning, providing stati
    
[^34]: 在线核回归的紧致序贯置信区间

    Tight Sequential Confidence Intervals for Online Kernel Regression

    [https://arxiv.org/abs/2611.19772](https://arxiv.org/abs/2611.19772)

    本文在核的最坏情形下刻画了在线核回归置信区间的尖锐极小极大宽度，从而解决了Vakili等人（2021）提出的开放问题。

    

    我们在核的最坏情形下，刻画了再生核希尔伯特空间（RKHS）中元素的在线置信区间的尖锐极小极大宽度。这解决了Vakili等人（2021）提出的原始离散时间形式的开放问题。学习者在自适应选择的点上观测到 $y_i=f(x_i)+\epsilon_i$，其中 $\|f\|_{\mathcal H_k}\le B$ 且噪声为独立同分布的 $R$-次高斯噪声，并报告区间 $|f(x)-\mu_n(x)|\le\rho_n(\delta)\sigma_n(x)$，其中 $\mu_n$ 和 $\sigma_n$ 是正则化参数为 $\lambda$ 的代理高斯过程模型的后验均值和标准差。记 $\rho_n^*(\delta)$ 为固定核下的极小极大置信宽度，我们证明：对于线性核与周期Sobolev核之和，对所有足够大的 $n$ 以及所有 $\delta\in(0,1/4]$，均有 $\rho_n^*(\delta)\asymp B+(R/\lambda)\sqrt{\gamma_n+\log(1/\delta)}$，其中 $\gamma_n$ 为最大信息增益。

    arXiv:2611.19772v1 Announce Type: cross  Abstract: We characterize, in the worst case over kernels, the sharp minimax width of online confidence intervals for elements of a reproducing kernel Hilbert space (RKHS). This resolves the open problem posed by Vakili et al. (2021) in its original discrete-time form. A learner observes $y_i=f(x_i)+\epsilon_i$ at adaptively chosen points, with $\|f\|_{\mathcal H_k}\le B$ and i.i.d. $R$-sub-Gaussian noise, and reports the interval $|f(x)-\mu_n(x)|\le\rho_n(\delta)\sigma_n(x)$, where $\mu_n$ and $\sigma_n$ are the posterior mean and standard deviation of a surrogate Gaussian process model with regularization parameter $\lambda$. Writing $\rho_n^*(\delta)$ for the minimax confidence width for a fixed kernel, we show that, for the sum of a linear kernel and a periodic Sobolev kernel, all sufficiently large $n$, and all $\delta\in(0,1/4]$, $\rho_n^*(\delta)\asymp B+(R/\lambda)\sqrt{\gamma_n+\log(1/\delta)}$, where $\gamma_n$ is the maximal informati
    
[^35]: 在求解器定义的物理约束流形上学习自回归神经算子

    Learning Autoregressive Neural Operators on Solver-Defined Physical Constraint Manifolds

    [https://arxiv.org/abs/2611.19753](https://arxiv.org/abs/2611.19753)

    提出PCM-NO方法，通过“预测增量—切向投影—度量回缩”的投影残差机制，保证自回归神经算子每一步预测都严格落在满足不可压缩性与边界条件的求解器可行集上，实现物理约束流形上的封闭状态转移。

    

    自回归神经算子通过反复组合学习到的转移算子来近似时间相关的偏微分方程（PDE）求解器，但由此产生的状态在数值问题的离散约束下未必保持可行。我们针对不可压缩流研究了这一问题，其中每个速度状态必须同时满足求解器层面的不可压缩性约束和给定的边界值。我们将自回归算子学习表述为求解器定义的仿射可行集之间的映射，并提出了 PCM-NO——这一可行转移原理的投影残差实现。其神经骨干网络预测候选状态增量，固定的切向投影提取出可采纳分量，度量回缩则将推进后的状态映射到指定的可行集上。通过精确的仿射约束求解，所得的转移在可行集上是封闭的，同时一个仿射恒等式将可采纳更新参数化与稳……（原文摘要在此处截断）

    arXiv:2611.19753v1 Announce Type: cross  Abstract: Autoregressive neural operators approximate time-dependent PDE solvers by repeatedly composing a learned transition, although the resulting states need not remain feasible under the discrete constraints of the numerical problem. We study this issue for incompressible flow, where each velocity state must satisfy solver-level incompressibility together with prescribed boundary values. We formulate autoregressive operator learning as a map between solver-defined affine feasible sets and introduce PCM-NO, a projected-residual realization of this feasible-transition principle. The neural backbone predicts a candidate state increment, a fixed tangent projection extracts its admissible component, and a metric retraction maps the advanced state to the prescribed feasible set. With exact affine constraint solves, the resulting transition is closed on the feasible set, while an affine identity separates admissible-update parameterization from st
    
[^36]: 面向因果先验数据拟合网络的事实世界与反事实世界联合学习

    Joint Learning of Factual and Counterfactual Worlds for Causal Prior-data Fitted Networks

    [https://arxiv.org/abs/2611.19710](https://arxiv.org/abs/2611.19710)

    该论文提出一种新的因果先验数据拟合网络，通过在合成环境中联合学习每个单元的事实与反事实两个世界的结果并利用二者间的依赖关系，突破了现有因果PFN只能学习单一世界结果的限制，从而提升处理效应的估计效果。

    

    估计处理效应是许多学科科学研究的核心，需要对替代世界进行反事实推理。然而在现实世界中，每个单元只会在事实世界下揭示其结果，因此任何基于真实数据训练的模型都无法联合学习事实结果与反事实结果。因此，该问题只有在可以同时生成两个世界的合成环境中才可求解。先验数据拟合网络（PFNs）的最新进展表明，完全在合成数据上预训练的模型可以迁移到现实世界的预测任务中，这一范式随后也被应用于因果效应估计。尽管处于这种完全合成的设置中，现有的因果PFN仍将现实世界的约束带入其训练目标，把每个单元的学习限制在单一世界的结果上，从而丢失了两个世界之间的依赖关系，处理效应的密度默认依赖于独立性假设。

    arXiv:2611.19710v1 Announce Type: cross  Abstract: Estimating treatment effects is central to science across many disciplines and requires counterfactual reasoning about alternative worlds. In the real world, however, each unit reveals its outcome only under the factual world. Thus, the factual and counterfactual outcomes can never be jointly learned by any model trained on real data. The problem is therefore tractable only in synthetic settings, where both worlds can be generated. Recent progress on prior-data fitted networks (PFNs) has shown that models pre-trained purely on synthetic data can transfer to real-world prediction. This paradigm has since carried over to causal effect estimation. Despite this fully synthetic setup, existing causal PFNs carry the real-world constraint into their training objective, confining learning to a single world's outcome per unit. The dependence between the two worlds is thereby lost, and the density of the treatment effect relies by default on ind
    
[^37]: 曲棍球棒散度下离散分布的极小极大估计

    Minimax Estimation of Discrete Distributions under Hockey-stick Divergence

    [https://arxiv.org/abs/2611.19702](https://arxiv.org/abs/2611.19702)

    本文首次研究了基于曲棍球棒散度的离散分布估计问题，给出了以字母表大小、样本数和散度参数表示的极小极大期望损失的非渐近上下界，并揭示了多项式与指数两种量级机制之间的相变现象。

    

    我们开创了基于独立样本、以曲棍球棒散度为损失度量来估计离散分布的研究。我们给出了极小极大期望损失的非渐近上界和下界，这些界以字母表大小、样本数量和散度参数表示。上界通过加常数估计器实现，下界则通过贝叶斯风险论证推导得出。上界和下界均揭示了随着问题参数的变化，损失在多项式量级与指数量级之间的相变现象。

    arXiv:2611.19702v1 Announce Type: cross  Abstract: We initiate the study of estimating a discrete distribution with respect to the hockey-stick divergence from independent samples. We provide non-asymptotic upper and lower bounds on the minimax expected loss in terms of the alphabet size, the number of samples, and the divergence parameter. The upper bounds are achieved by add-constant estimators, while the lower bound is derived via a Bayes risk argument. Both upper bounds and lower bounds reveal a transition between polynomial and exponential regimes as the parameters of the problem vary.
    
[^38]: 去噪自编码器的几何理解及其在图像去噪中的应用

    A Geometric Understanding of Denoising Autoencoder with Applications to Image Denoising

    [https://arxiv.org/abs/2611.19653](https://arxiv.org/abs/2611.19653)

    本文在流形假设下首次系统建立了去噪自编码器的泛化理论，通过局部坐标卡与神经网络逼近技术在三种几何场景（单流形、分离与相交流形并集）下推导出期望重构误差的上界，为其在图像去噪中的优异表现提供了几何解释。

    

    去噪自编码器在从含噪观测中恢复干净数据方面已展现出强大的经验性能，但其背后的统计行为仍未得到充分探索。本文在流形假设下研究去噪自编码器，假设干净数据支撑在高维环境空间中嵌入的低维光滑流形上，而含噪观测由各向同性高斯噪声生成。我们在三种几何场景下建立了泛化保证：单个流形、成对分离的有限流形并集，以及相交流形的有限并集。利用局部坐标卡表示、单位分解和神经网络逼近，我们推导出 去 噪自编码器期望平方重构误差的上界。在单流形和分离并集场景中，误差由样本相关的统计项（阶为 n^{-2/(d+8)} 量级）和一个不依赖于样本的逼近项组成；在相交流形并集的更复杂场景中，我们刻画了误差在相交邻域附近的退化行为。理论结果为去噪自编码器在图像去噪任务中的成功提供了几何解释。

    arXiv:2611.19653v1 Announce Type: cross  Abstract: Denoising autoencoders have shown strong empirical performance in recovering clean data from their noisy observations, yet their underlying statistical behavior remains underexplored. In this paper, we study denoising autoencoders under the manifold hypothesis, assuming that clean data are supported on low-dimensional smooth manifolds embedded in a high-dimensional ambient space, while noisy observations are generated by isotropic Gaussian noise. We establish generalization guarantees in three geometric scenarios: a single manifold, a pairwise separated finite union of manifolds, and a finite union of intersecting manifolds. Using local chart representations, partitions of unity, and neural network approximation, we derive upper bounds for the expected squared reconstruction error for denoising autoencoder. In the single-manifold and separated-union scenarios, the error consists of a sample-dependent statistical term of order $n^{-2/(d
    
[^39]: 贝尔曼残差括号法：解决折扣型与平均奖励强化学习的样本复杂度问题

    Bellman Residual Bracketing: Resolving the Sample Complexity of Discounted and Average-Reward RL

    [https://arxiv.org/abs/2611.19566](https://arxiv.org/abs/2611.19566)

    本文提出贝尔曼残差括号技术，建立了折扣型与平均奖励强化学习中插件方法的紧致样本复杂度上界，并结合下界在常数因子内解决了两类问题的极小极大样本复杂度。

    

    我们研究了基于生成模型的强化学习（RL）中经验规划或插件（plug-in）方法的样本复杂度。我们证明折扣型插件方法能够以 $O(SA (1-\gamma)^{-3} \varepsilon^{-2} \log(SA/\delta))$ 的样本量找到 $\varepsilon$-最优策略，其中 $S$、$A$、$\gamma$、$\delta$ 分别表示状态数、动作数、折扣因子和失败概率。在平均奖励设定下，我们证明折扣型插件和直接（无折扣）插件方法分别达到 $O(SAH\varepsilon^{-2}\log(SA/\delta))$ 和 $O(SAD\varepsilon^{-2}\log(SA/\delta))$ 的复杂度，其中 $H$ 是最优偏差跨度的上界，$D$ 是直径。结合我们针对非自适应生成模型给出的下界，这些结果在常数因子范围内解决了折扣型RL和平均奖励RL的极小极大（minimax）样本复杂度问题——无论是基于跨度的类别还是基于直径的类别。我们的主要技术贡献……

    arXiv:2611.19566v1 Announce Type: new  Abstract: We study the sample complexity of the empirical planning or plug-in approach for reinforcement learning (RL) with a generative model. We show the discounted plug-in approach finds an $\varepsilon$-optimal policy with $O(SA (1-\gamma)^{-3} \varepsilon^{-2} \log(SA/\delta))$ samples, where $S, A, \gamma, \delta$ are the state and action sizes, discount factor, and failure probability, respectively. In the average-reward setting, we show that discounted plug-in and direct (undiscounted) plug-in achieve $O(SAH\varepsilon^{-2}\log(SA/\delta))$ and $O(SAD\varepsilon^{-2}\log(SA/\delta))$ complexities, respectively, where $H$ upper-bounds the optimal bias span and $D$ is the diameter. Combining with our lower bounds for nonadaptive generative models, these results resolve the minimax sample complexity of discounted RL and average-reward RL --- in both the span- and diameter-based classes --- up to constant factors. Our main technical contributi
    
[^40]: 通过数据投毒引导贝叶斯推断与决策

    Steering Bayesian Inference and Decisions through Data Poisoning

    [https://arxiv.org/abs/2611.19557](https://arxiv.org/abs/2611.19557)

    该论文提出一种数据投毒攻击方法：攻击者只需通过删除或复制少量精心选择的观测数据，即可操纵贝叶斯后验分布中的特定矩或下游决策，而无需完全指定目标后验分布。

    

    对抗性机器学习已经揭示，统计模型可以通过被投毒的数据进行操纵，贝叶斯推断也不例外。先前的研究表明，微妙的投毒攻击可以将后验分布推向攻击者选定的分布。这种后验吸引攻击需要一个完全指定的目标后验分布；然而，攻击者通常只关心少数几个后验量，或者更关键的是，只关心下游决策。在本文中，我们研究攻击者如何通过删除或复制少量观测数据来改变贝叶斯推断和决策，这些观测数据的选择旨在最小化所得后验分布与目标分布之间的最大均值差异（MMD）。合适的核函数使攻击者能够操纵选定的后验矩，而无需指定目标的其余部分。由于贝叶斯行动仅通过期望效用依赖于后验分布，而期望效用本身就是后验矩，因此同样的方法……

    arXiv:2611.19557v1 Announce Type: cross  Abstract: Adversarial machine learning has revealed that statistical models can be manipulated through poisoned data, and Bayesian inference is no exception. Earlier work has shown that subtle poisoning attacks can drive the posterior toward a distribution of the attacker's choosing. Such posterior attraction attacks require a fully specified target posterior; however, attackers typically care about only a few posterior quantities or, more consequentially, about a downstream decision. Herein, we study how an attacker can alter Bayesian inference and decisions by deleting or replicating a few observations, chosen to minimize the maximum mean discrepancy between the resulting posterior and a target distribution. A suitable kernel lets the attacker steer selected posterior moments without specifying the rest of the target. Because Bayes actions depend on the posterior only through expected utilities, which are themselves posterior moments, the same
    
[^41]: 复合函数先验的几何稳定性与样本复杂度

    Geometric Stability and Sample Complexity for Compositional Function Priors

    [https://arxiv.org/abs/2611.19523](https://arxiv.org/abs/2611.19523)

    该论文证明了复合函数类 f(x)=g(Ax+Δh(x)) 的覆盖数界，表明其统计复杂度可在低维内蕴参数化速率与全维非参数速率之间平滑过渡，且过渡程度由非线性修正项的大小控制。

    

    我们研究复合结构何时能在监督学习中产生降维效果。我们将学习问题建模为带有函数空间先验的经验风险最小化：通过一个正则化泛函选取具有良好逼近性质和度量熵的函数子集。所考虑的模型为复合形式 $f(x)=g(Ax+\Delta h(x))$，其中 $A\in\mathbb{R}^{k\times d}$ 是线性基映射，$\Delta h$ 是非线性修正项，$g$ 是外层函数；也就是说，$f$ 是输入的一个 $k$ 维表示（除修正项外为线性）的任意光滑函数。我们证明了覆盖数上界，表明该函数类的统计复杂度介于两种机制之间：一种是内蕴机制——对线性映射的 $kd$ 个元素呈参数化形式，仅在表示维度 $k$ 上呈非参数化形式；另一种是整个 $d$ 维空间的环境非参数速率；二者之间的插值程度由修正项大小 $\varepsilon=\|\Delta h\...$ 控制。

    arXiv:2611.19523v1 Announce Type: cross  Abstract: We study when compositional structure yields dimension reduction in supervised learning. We frame learning as empirical risk minimization with a function-space prior: a regularization functional selects a subset of functions with favorable approximation and metric entropy. The models are compositions $f(x)=g(Ax+\Delta h(x))$, where $A\in\mathbb{R}^{k\times d}$ is a linear base map, $\Delta h$ is a nonlinear correction, and $g$ is an outer function; that is, $f$ is an arbitrary smooth function of a $k$-dimensional representation of the input that is linear up to a correction. We prove covering-number bounds showing that the statistical complexity of this class interpolates between an intrinsic regime, parametric in the $kd$ entries of the linear map and nonparametric only in the representation dimension $k$, and an ambient $d$-dimensional nonparametric rate; the interpolation is controlled by the correction size $\varepsilon=\|\Delta h\
    
[^42]: 基于双堆叠智能超表面的空口语义对齐

    Over-the-Air Semantic Alignment via Dual Stacked Intelligent Metasurfaces

    [https://arxiv.org/abs/2611.19515](https://arxiv.org/abs/2611.19515)

    本文提出双堆叠智能超表面架构，通过在波域中联合利用链路两端的可编程超表面与无线信道，无需数字处理即可实现异构潜在空间之间的端到端语义对齐。

    

    语义通信能够实现面向任务的信息交换，但当发送端和接收端采用异构机器学习模型时，会面临潜在空间不匹配的问题。现有的语义对齐方法依赖数字处理，会引入计算开销和时延开销。本文提出了一种双堆叠智能超表面架构，可在波域中完全实现语义对齐。通过联合利用链路两端的可编程超表面以及无线信道，所提出的架构实现了异构潜在空间之间的端到端线性变换。我们证明了仅相位调控的多层SIM能够同时逼近监督式线性对齐器和零样本Parseval框架均衡器。我们研究了单SIM、非对称与对称的分离式双SIM以及联合优化的双SIM等多种配置，并考虑了信道均衡与建设性信道利用。

    arXiv:2611.19515v1 Announce Type: cross  Abstract: Semantic communications enable task-oriented information exchange, but suffer from latent-space mismatch when transmitter and receiver employ heterogeneous machine learning models. Existing semantic alignment methods rely on digital processing, introducing computational and latency overhead. This paper proposes a dual stacked intelligent metasurfaces (dual-SIM) architecture that performs semantic alignment entirely in the wave domain. By jointly exploiting programmable metasurfaces at both ends of a link and the wireless channel, the proposed architecture realizes an end-to-end linear transformation between heterogeneous latent spaces. We show phase-only multilayer SIM can approximate both supervised linear aligners and zero-shot Parseval-frame equalizers. We investigate single-SIM, disjoint asymmetric and symmetric dual-SIM, and jointly optimized dual-SIM configurations, considering channel equalization and constructive channel exploi
    
[^43]: 解读随机森林：代理模型与规则提取方法的比较

    Interpreting Random Forests: Comparing Surrogate Models and Rule Extraction Methods

    [https://arxiv.org/abs/2611.19471](https://arxiv.org/abs/2611.19471)

    本文系统比较了12种解释随机森林的方法，发现没有一种方法在所有目标上均最优，其中SLIM在回归任务中综合表现最强，为研究者根据具体研究问题选择合适的解释方法提供了指导。

    

    随机森林（RF）通常具有很高的预测性能，但其集成结构难以解释。代理模型和规则提取方法可以提供可解释的近似，但针对特定研究问题如何选择合适的方法，相关指导仍然有限。我们在11个模拟设置和67个用于回归与分类任务的基准数据集上，比较了12种解释随机森林的方法，并以决策树作为面向预测的参考。评估内容包括与随机森林的预测及结构相似性、模型复杂度、预测性能和运行时间。模拟实验还利用已知的效应变量和噪声变量额外评估了可解释性。结果显示，没有一种方法能在所有目标上表现最佳。对于回归任务，SLIM在预测相似性或测试数据性能方面提供了最强的综合表现，而线性模型则为线性效应提供了更简单的全局解释。

    arXiv:2611.19471v1 Announce Type: cross  Abstract: Random forests (RFs) often achieve high prediction performance, but their ensemble structure can be difficult to interpret. Surrogate models and rule extraction methods offer interpretable approximations, yet guidance on selecting an appropriate method for a specific research question remains limited. We compared 12 methods for interpreting RFs, with a decision tree as a prediction-oriented reference, across 11 simulation settings and 67 benchmark datasets for regression and classification. Evaluation addressed prediction and structural similarity to the RF, model complexity, prediction performance, and runtime. Simulations additionally assessed explainability using known effect and noise variables. No method performed best across all objectives. For regression, SLIM provided the strongest general performance for prediction similarity or performance on test data, while a linear model offered a simpler global interpretation of linear ef
    
[^44]: 训练数据归因的高效且精确的因果验证

    Efficient and Accurate Causal Validation of Training Data Attribution

    [https://arxiv.org/abs/2611.19451](https://arxiv.org/abs/2611.19451)

    该论文提出从训练子集数量、重训练种子数和归因集成模型数三个维度对LDS估计进行去偏差，从而大幅降低训练数据归因因果验证所需的重新训练计算量，同时保持估计精度。

    

    训练数据归因（TDA）方法的因果验证代价高昂，因为评估过程需要在原始训练集的许多不同子集上重新训练模型。例如，线性数据建模分数（LDS）已成为评估TDA方法的标准指标，它衡量的是TDA预测值与预期的重训练模型输出之间的相关性。在本文中，我们从三个维度减少了估计LDS所需的计算量：采样的训练子集数量、每个子集的重训练随机种子数量，以及归因集成中的模型数量。上述每个因素都会在LDS估计中引入某种形式的偏差，而消除这些偏差来源可以显著减少在LDS估计中达到相同精度所需的重训练次数。在ResNet-9/CIFAR-10上，有限子集去偏差方法仅使用4个子集就实现了3.2%的相对RMSE，而32个子集即可匹配原有方法的精度。

    arXiv:2611.19451v1 Announce Type: new  Abstract: Causal validation of training data attribution (TDA) methods is costly, since evaluation involves retraining a model on many different subsets of the original training set. For example, the Linear Datamodeling Score (LDS), which has become a standard metric for evaluating TDA methods, measures the correlation between TDA predictions and expected retrained model outputs. In this paper, we reduce the amount of computation needed to estimate the LDS along three axes: the number of sampled training subsets, the number of retraining seeds per subset, and the number of models in the attribution ensemble. Each of these factors induces a form of bias in the LDS estimation, and removing these sources of bias significantly reduces the number of retrainings needed to match the same accuracy in LDS estimation. On ResNet-9/CIFAR-10, finite-subset debiasing achieves $3.2\%$ relative RMSE with as few as $4$ subsets, and $32$ subsets match the accuracy 
    
[^45]: 稀疏连接神经网络的泛化边界

    Generalization Bounds for Neural Networks with Sparse Connectivity

    [https://arxiv.org/abs/2611.19382](https://arxiv.org/abs/2611.19382)

    本文针对层间权重矩阵受ℓ_p拟范数约束的稀疏连接深度神经网络，推导出了非渐近泛化间隙上界，从理论上解释了稀疏性如何影响神经网络的统计性能。

    

    实证证据表明，大型神经网络很少有效利用其全部可用参数，学习到的解往往呈现出显著的稀疏性。尽管这种现象普遍存在，现有的泛化理论仅能部分解释这种稀疏性如何影响统计性能。在本文中，我们为层间权重矩阵受 $\ell_p$ 拟范数约束的深度神经网络推导了非渐近泛化间隙边界。

    arXiv:2611.19382v1 Announce Type: cross  Abstract: Empirical evidence suggests that large neural networks rarely make effective use of all available parameters, with learned solutions exhibiting pronounced sparsity. Despite this ubiquity, existing generalization theory only partially explains how such sparsity influences statistical performance. In this paper, we derive non-asymptotic generalization gap bounds for deep neural networks whose layer-wise weight matrices are restricted by $\ell_p$ quasi-norm ($0
    
[^46]: 基于右删失处理的平均剂量-反应函数学习

    Learning the Average Dose-Response Function with a Right-Censored Treatment

    [https://arxiv.org/abs/2611.19204](https://arxiv.org/abs/2611.19204)

    本文提出了一种基于增广局部估计方程的多重稳健估计量，用于在处理时长右删失与结局缺失下估计平均剂量-反应函数，并通过去偏技术实现了无需欠平滑的渐近有效统计推断。

    

    本文研究了在处理时长右删失以及删失单元结局缺失情况下，平均剂量-反应函数的估计与推断问题。利用删失记录所提供的时长下界信息，我们通过增广局部估计方程构建了一种新颖的多重稳健估计量。在适当的正则条件下，当结局回归、处理密度和删失生存函数三者中任意两个被一致估计时，该估计量仍保持一致性。我们在交叉拟合框架下，基于对干扰参数估计的适当条件，建立了该估计量的收敛速度和渐近分布。为解决可能扭曲置信区间覆盖率的局部平滑偏差问题，我们进一步开发了一种去偏估计量，无需欠平滑即可实现渐近有效的逐点推断。在附加的一致性条件下，我们利用高斯近似构建了同步置信带。

    arXiv:2611.19204v1 Announce Type: cross  Abstract: This paper considers estimation and inference for the average dose-response function under right-censored treatment duration and missing outcomes for censored units. Using the duration lower bounds provided by censored records, we construct a novel multiply robust estimator through augmented local estimating equations. Under suitable regularity conditions, the estimator remains consistent when any two of the outcome regression, treatment density, and censoring survival function are consistently estimated. We establish its convergence rate and asymptotic distribution within a cross-fitting framework under appropriate conditions on nuisance estimation. To address local smoothing bias that can distort confidence interval coverage, we further develop a debiased estimator that supports asymptotically valid pointwise inference without undersmoothing. Under additional uniform conditions, we construct simultaneous confidence bands using a Gaus
    
[^47]: 平均场词元引导

    Mean-field Token Steering

    [https://arxiv.org/abs/2611.19163](https://arxiv.org/abs/2611.19163)

    该论文将 Transformer 深层中的随机词元引导建模为单位球面上相互作用粒子系统的平均场最小努力控制问题，证明其受控动力学等价于适配球面几何的平均场薛定谔桥，并提出动态 Sinkhorn 递推的新推广方法来计算一致的词元演化。

    

    我们将通过深层 Transformer 层堆栈的随机词元引导视为单位球面上相互作用粒子系统的平均场控制问题，其中相互作用源于注意力机制。在平均场极限下，词元群体通过一个受控的积分-偏微分方程演化，其中反馈控制作为校正向量场，引导词元群体趋向表示空间中的词元聚合构型。我们推导了在固定时间范围（被解释为归一化层深度）内完成该任务所需的最小努力控制。我们证明所得的词元群体受控动力学可被理解为适配球面几何的平均场薛定谔桥。在此基础上，我们提出了一种动态 Sinkhorn 递推的新颖推广方法，用于计算一致的词元演化。数值实验证实了所提出的方法。

    arXiv:2611.19163v1 Announce Type: cross  Abstract: We consider stochastic token steering through a deep stack of transformer layers as a mean-field control problem for an interacting particle system on the unit sphere, where interactions arise from the attention mechanism. In the mean-field limit, the token population evolves via a controlled integro-partial differential equation where the feedback control acts as a corrective vector field guiding the population toward an aggregate configuration of tokens in the representation space. We deduce the minimum effort control to accomplish this task over a fixed time horizon interpreted as normalized layer depth. We show that the resulting controlled dynamics for the token population can be understood as a mean-field Schr\"odinger bridge adapted to the sphere geometry. Building on this, we propose a novel generalization of the dynamic Sinkhorn recursion to compute the consistent token evolution. Numerical experiments confirm that the propose
    
[^48]: 基于打赌的对数最优双样本检验

    Log-optimal two-sample testing by betting

    [https://arxiv.org/abs/2611.19062](https://arxiv.org/abs/2611.19062)

    本文针对标准Borel观测空间构造了一个统一的、任意时刻有效的打赌式双样本检验方法，对所有未知分布对及任意样本量比率均渐近对数最优，其最优对数财富增长率由Jensen–Shannon散度刻画。

    

    对于每个标准Borel观测空间，我们构建了一个统一的任意时刻有效双样本检验程序，该程序对于每一对未知概率分布以及任意发散样本量的极限比率（包括零和无穷）都是渐近对数最优的。对于独立且平衡的样本对，最优对数财富增长率为 $2\JS(P,Q)$，其中 $\JS$ 为Jensen–Shannon散度。当样本量不等时，该增长率变为加权Jensen–Shannon信息；当 $n_1/n_2\to\infty$ 时，得到单样本极限 $\log W(n_1,n_2)/n_2\to\KL(Q\|P)$。通过细化有限划分以及使用可测特征核的对数损失学习，我们提供了普适性的构造方法。我们的主要贡献在于该构造的普适性及其与经典批量效率概念之间的联系。实际实现允许对分类器进行持续微调，并在有限字母表、高斯数据和手写数字上进行了演示。

    arXiv:2611.19062v1 Announce Type: cross  Abstract: For each standard Borel observation space, we construct a single anytime-valid two-sample procedure that is asymptotically log-optimal for every pair of unknown probability laws and every limiting ratio of diverging sample sizes, including zero and infinity. For independent balanced pairs, the optimal log-wealth rate is $2\JS(P,Q)$, where $\JS$ is the Jensen--Shannon divergence. Unequal sizes yield weighted Jensen--Shannon information and the one-sample limit $\log W(n_1,n_2)/n_2\to\KL(Q\|P)$ when $n_1/n_2\to\infty$. Refining finite partitions and logarithmic-loss learning with measurable characteristic kernels provide universal constructions. Our main contribution is the generality of our construction, and its connections to classical batch efficiency notions. Practical implementations allow continuous classifier fine-tuning, as demonstrated on finite alphabets, Gaussian data, and handwritten digits.
    
[^49]: 随机梯度下降（SGD）自适应变体的高维极限定理 I：弹道极限

    High-dimensional limit theorems for adaptive variants of SGD I: Ballistic limits

    [https://arxiv.org/abs/2611.19059](https://arxiv.org/abs/2611.19059)

    本文通过平均化方法将ADAM、MUON等自适应动量SGD变体中由指数移动平均和预条件化引起的非马尔可夫性转化为马尔可夫有效动力学，建立了高维极限定理，从而实现对不同优化算法的精确定量比较。

    

    我们为基于动量的自适应随机梯度下降（SGD）变体建立了高维有效动力学。这些有效动力学由一个马尔可夫过程给出。关键观察是，算法中由指数移动平均和预条件化所导致的非马尔可夫性可以通过平均化方法来处理。我们的方法适用于一大类算法，包括自适应动量估计（ADAM）、基于牛顿-舒尔茨正交化的动量方法（MUON）及其变体。该方法还适用于广泛的学习任务，例如张量PCA、高斯投影模型、师生网络，以及使用实践中常见的MUON与ADAM混合方法在XOR问题上训练的多层神经网络。我们展示了该技术如何在算法之间进行精确的定量比较。特别地，我们发现了有趣的现象，例如交叉（cross-over）现象。

    arXiv:2611.19059v1 Announce Type: cross  Abstract: We develop the high-dimensional effective dynamics for adaptive, momentum-based variants of Stochastic Gradient Descent (SGD). These effective dynamics are given by a Markov process. They key observation is that the non-Markovianity in the algorithm due to the exponential moving average and preconditioning can be treated using the method of averaging. Our approach applies to a broad class of algorithms including to Adaptive momentum estimation (ADAM), Momentum orthogonalized by Newton-Schultz (MUON), and its variants. It also applies to a broad class of learning tasks, such at tensor PCA, Gaussian projective models, teacher-student networks, and a multi-layer neural network trained on the XOR problem using a hybrid of MUON and ADAM as commonly done in practice. We demonstrate how to this technique can give precise, quantitative comparisons between algorithms. In particular, we find interesting phenomena, such as the cross-over phenomen
    
[^50]: HeteroWear：面向异构可穿戴活动识别的传感器感知自监督学习

    HeteroWear: Sensor-Aware Self-Supervised Learning for Heterogeneous Wearable Activity Recognition

    [https://arxiv.org/abs/2611.19027](https://arxiv.org/abs/2611.19027)

    提出了传感器感知的自监督框架HeteroWear，通过在包含八个数据集、173名受试者的异构可穿戴语料库上预训练，将传感器内时间建模与跨传感器关系建模分离，实现了可跨异构传感器配置迁移的活动识别表征。

    

    面向可穿戴传感的自监督学习通常局限于单一数据集或设备系列，这留下了一个悬而未决的问题：学习到的表征能否跨异构传感器配置进行迁移，以及性能提升究竟来自预训练本身，而非架构设计或更丰富的传感器输入。我们提出了HeteroWear，这是一个传感器感知的自监督框架，在一个涵盖八个数据集的可穿戴语料库（173名受试者，358,261个时间窗口）上进行预训练，该语料库横跨不同身体位置、设备和采样率的惯性信号与生理信号。HeteroWear利用传感器上下文嵌入、时间Transformer和跨传感器Transformer，以及潜在空间的跨注意力融合，将传感器内的时间建模与跨传感器的关系建模分离开来。训练过程结合了结构化传感器丢弃、跨配置一致性学习以及逐轴掩码传感器建模；对于缺失的传感器，直接省略而非重建。在留一数据集的预训练和受试者不重叠的评估设置下……

    arXiv:2611.19027v1 Announce Type: cross  Abstract: Self-supervised learning for wearable sensing is usually confined to a single dataset or device family, leaving open whether representations transfer across heterogeneous sensor configurations and whether gains come from pretraining rather than architecture or richer sensor input. We present HeteroWear, a sensor-aware self-supervised framework pretrained on an eight-dataset wearable corpus (173 subjects, 358,261 windows) spanning inertial and physiological signals across body locations, devices, and sampling rates. HeteroWear separates within-sensor temporal modeling from cross-sensor relational modeling using sensor-context embeddings, temporal and cross-sensor Transformers, and latent cross-attention fusion. Training combines structured sensor dropout, cross-configuration consistency, and per-axis masked sensor modeling; missing sensors are omitted rather than reconstructed. Under leave-one-dataset-out pretraining and subject-disjoin
    
[^51]: 将带宽受限最大似然推向高维：线性内存、参数速率与自调节截断

    Taking Band-Limited Maximum Likelihood to High Dimensions: Linear Memory, the Parametric Rate, and a Self-Tuning Cut-off

    [https://arxiv.org/abs/2611.19011](https://arxiv.org/abs/2611.19011)

    该论文证明分箱BLML方程是严格凸目标函数的驻点条件，并提出无需构造核矩阵的块坐标下降算法，实现了线性内存扩展与单卡GPU在八维下拟合百万级数据点，同时证明了参数速率 n⁻¹ 与自调节截断在高维中依然有效。

    

    带宽受限最大似然（BLML）将密度估计为一个带宽受限函数的平方。该估计是非负的，并且当真实密度的平方根为带宽受限时，其误差以参数速率 n^{-1} 下降。其算法通过数据分箱实现，但这仅在一维或二维下快速：从三维起，每个点都有自己独立的箱，核系统变为稠密，内存按 n² 增长——在 10⁵ 个点时达到 40 GB。该速率在高维（即核估计器变慢的场景）下是否仍然成立，此前从未被验证。我们研究的是稠密数据，即密度填满整个空间，而非接近低维流形的数据。我们证明分箱方程是一个严格凸目标函数的驻点条件，并通过块坐标下降法将其最小化，而无需构造核矩阵：内存关于 n 呈线性，一次扫描的代价为 O(n²d)，一次拟合约需十分钟，一块 A100 GPU 可在八维下拟合 10⁶ 个点……

    arXiv:2611.19011v1 Announce Type: cross  Abstract: Band-limited maximum likelihood (BLML) estimates a density as the square of a band-limited function. The estimate is non-negative, and when the square root of the true density is band-limited its error falls at the parametric rate $n^{-1}$. Its algorithm bins the data, which is fast only in one or two dimensions: from three on every point has its own bin, the kernel system is dense, and memory grows as $n^2$, 40 GB at $10^5$ points. Whether the rate survives in higher dimension, where kernel estimators slow down, had not been tested. We study dense data, densities that fill their space, not data near a low-dimensional manifold. We show that the binned equations are the stationarity condition of a strictly convex objective and minimise it by block coordinate descent without forming the kernel matrix: memory is linear in $n$, a sweep costs $O(n^2 d)$ and a fit takes about ten, and one A100 GPU fits $10^6$ points in eight dimensions in ab
    
[^52]: 基于轨迹的贝叶斯方法从准稳态观测中发现宿主-病原体模型

    Trajectory-Based Bayesian Discovery of a Host-Pathogen Model from Quasi-Steady-State Observations

    [https://arxiv.org/abs/2611.18815](https://arxiv.org/abs/2611.18815)

    提出了一种结合稀疏贝叶斯学习、ℓ0迭代硬阈值与基于贝叶斯证据的两步交换爬山搜索的方法，能够从准稳态观测数据中成功恢复宿主-病原体动力学模型的全部真实机制，克服了经典稀疏辨识方法在此情形下的失效问题。

    

    当测量数据来自接近稳定的状态时，非线性动力系统的系统辨识变得极具挑战性。我们在一个带有三次项的双稳态宿主-病原体相互作用模型中研究了这一情形，该模型的七个真实机制隐藏在二十个候选项之中。数据来自准稳态观测窗口上的一条无噪声轨迹以及初始状态。在这种情况下，经典的稀疏辨识方法无法识别出正确的机制，因为候选函数库沿着准稳态轨迹会变得病态。为解决这一问题，我们提出了一种替代方法：先用稀疏贝叶斯学习和基于追踪的ℓ0迭代硬阈值方法提供一个良好的起点，然后通过对候选常微分方程解的贝叶斯证据进行两步交换爬山搜索来恢复真实模型。我们的结果表明成功实现了模型恢复，即正确识别了模型的支撑集。

    arXiv:2611.18815v1 Announce Type: cross  Abstract: System identification of nonlinear dynamical systems becomes challenging when the measured data come from a nearly settled state. We study this situation in the case of a bistable host-pathogen interaction model with a cubic term, whose seven true mechanisms are hidden among twenty candidate terms. The data are from one noise-free trajectory on a quasi-steady-state observed window and the initial state. In this situation, classical sparse identification approaches fail to identify the correct mechanisms as the candidate library becomes ill-conditioned along the quasi-steady-state trajectory. To address this, we propose an alternative approach in which sparse Bayesian learning and $\ell_{0}$ iterative hard thresholding with pursuit provide a good start, and a two-swap climb on the Bayesian evidence of the candidate ODE solutions recovers the true model. Our results show successful model recovery, i.e., correct identification of its supp
    
[^53]: 使用微调语言模型设定最优保留价

    Setting Optimal Reserve Prices using Fine-Tuned Language Models

    [https://arxiv.org/abs/2611.18785](https://arxiv.org/abs/2611.18785)

    该论文以拍卖出价的结构似然函数为训练目标微调预训练语言模型，从非结构化商品描述中学习估值分布表示，从而在无需预先指定产品属性或建模潜在竞拍者数量的情况下，为差异化商品设定最优保留价格。

    

    在线拍卖平台必须为高度差异化的产品设定保留价格，而这些产品的非结构化描述中蕴含着有价值的需求信息。现有方法通常要求管理者预先指定产品属性。我们探讨语言模型能否在竞拍开始之前，通过学习商品刊登描述和历史出价来为具体商品设定保留价格。我们以拍卖出价的结构似然函数作为训练目标，对预训练的文本嵌入语言模型进行微调，从而将表示学习引导至决定保留价决策的估值分布上。我们证明，连续的最高出价能够识别设定最优保留价所需的尾部信息，而无需观察或建模潜在竞拍者的数量。一种分样本估计器在一半数据上学习表示，并在该表示的条件下进行结构估计与推断。

    arXiv:2611.18785v1 Announce Type: new  Abstract: Online auction houses must set reserve prices for highly differentiated products whose unstructured descriptions contain valuable demand information. Existing approaches generally require managers to specify product attributes in advance. We ask whether a language model can instead learn from listing descriptions and historical bids to set item-specific reserve prices before bidding begins. We fine-tune a pretrained text embedding language model using the structural likelihood of auction bids as its training objective. This directs representation learning toward the valuation distribution that determines the reserve decision. We show that consecutive upper bids identify the tail information needed to set an optimal reserve without observing or modeling the number of potential bidders. A split-sample estimator learns the representation on one half of the data and conducts structural estimation and inference conditional on that representat
    
[^54]: 分阶段自适应提升算法

    Stagewise Adaptive Boosting

    [https://arxiv.org/abs/2611.18662](https://arxiv.org/abs/2611.18662)

    本文提出分阶段自适应提升这一通用框架，通过沿单调路径选择符合结构偏好的更新来促进稀疏性等解的性质，并可扩展至组选择、多结果模型相似性以及预测变量优先关系等场景。

    

    分阶段算法和诱导稀疏性的正则化是高维建模中两种有影响力的方法。然而，现有的分阶段理论主要集中于凸正则化，而更强的稀疏性以及其他结构偏好通常通过固定水平的非凸优化来追求，而非生成一条完整的自适应解路径。我们提出了分阶段自适应提升，这是一个通用框架，通过选择与结构偏好一致的更新来构建单调路径。该框架能够在路径本身上促进稀疏性，同时还允许进行扩展以促进解的其他性质，包括组选择、多个结果模型之间的相似性，以及预测变量之间已知的优先级或先决关系。我们研究了其理论性质，并通过模拟和真实数据分析研究了其有限样本性能。在这些研究中，所提出的路径（摘要在此处截断）

    arXiv:2611.18662v1 Announce Type: new  Abstract: Stagewise algorithms and sparsity-inducing regularization are two influential approaches to high-dimensional modeling. Existing stagewise theory, however, largely centers on convex regularization, whereas stronger sparsity and other structural preferences are often pursued through fixed-level nonconvex optimization rather than generating an entire adaptive solution path. We develop stagewise adaptive boosting, a general framework that builds a monotone path by selecting the update consistent with structural preference. The framework can encourage sparsity along the path itself and also admits extensions that promote other properties of the solution, including group selection, similarity across models for multiple outcomes, and known priorities or prerequisite relations among predictors. We study its theoretical property as well as finite sample performance using simulations and real-data analyses. Across these studies, the proposed pathw
    
[^55]: Domino：基于一阶马尔可夫链的少步掩码扩散联合去掩码方法

    Domino: Joint Unmasking via First-order Markov Chain for Few-Step Masked Diffusion

    [https://arxiv.org/abs/2611.18645](https://arxiv.org/abs/2611.18645)

    提出 Domino 框架，用一阶马尔可夫链建模同一步骤中揭示的 token 之间的条件依赖，仅以 2.6% 的参数开销实现少步掩码扩散的联合去掩码生成。

    

    掩码扩散模型通过在多个去噪步骤中逐步揭示被掩码的位置来生成序列。少步采样器必须一次揭示许多位置，但标准的反向核对它们是独立抽取的，主要忽略了在同一步骤中揭示的token之间的条件依赖关系。我们提出了一个名为 Domino 的新颖框架，它使用一阶马尔可夫链在给定当前状态的情况下对揭示位置之间的这种依赖关系进行建模。第一个被揭示的位置保留预训练掩码扩散模型（MDM）的后验分布，而每个后续位置则根据其前驱token遵循给定的转移分布。值得注意的是，由此产生的步骤在构造上是归一化的，将 MDM 中的因子化步骤作为特例包含在内，并具有闭式似然。我们通过在预训练 MDM 去噪器的隐藏状态上添加一个轻量级模块来实现 Domino 转移，参数开销仅为2.6%，……

    arXiv:2611.18645v1 Announce Type: cross  Abstract: Masked diffusion models generate a sequence by revealing masked positions over many denoising steps. A few-step sampler has to reveal many positions at once, but the standard reverse kernel draws them independently and primarily ignores the conditional dependence between tokens revealed in the same step. We propose a novel framework called Domino, which models this dependence with a first-order Markov chain over the revealed positions, given the current state. The first revealed position keeps the posterior of a pre-trained masked diffusion model (MDM), and each subsequent position follows a transition given the token of its predecessor. Notably, the resulting step is normalized by construction, contains the factorized step in MDMs as a special case, and has a likelihood in closed form. We implement the Domino transitions via a lightweight module on the hidden states of a pre-trained MDM denoiser with only 2.6% parameter overhead and t
    
[^56]: 确定性数据集两两不相似分解的集成界限

    Ensemble Bounds for Pairwise Dissimilar Decomposition of Deterministic Datasets

    [https://arxiv.org/abs/2611.18625](https://arxiv.org/abs/2611.18625)

    本文利用冗余图与二项随机图，并结合Lovász局部引理，首次为确定性数据集的两两不相似批次分解的最小规模给出了上界与下界的集成界限。

    

    在本文中，我们利用冗余图研究确定性数据集的降低相似性分解，并获得了此类约束分解最小规模的集成界限。我们考虑跨批次降低相似性的批次分解，并利用Lovász局部引理，得到了在给定两两不相似性条件下分解最小规模的界限。随后，我们考虑带有给定松弛参数的唯一批次分解，并得到其最小规模的上界与下界集成界限。在整个过程中，我们通过以相似数据点对的数量为条件，使用二项随机图来获得所需的集成界限。

    arXiv:2611.18625v1 Announce Type: cross  Abstract: In this paper, we study reduced similarity decompositions of \emph{deterministic} datasets using redundancy graphs and obtain \emph{ensemble} bounds for the minimum size of such constrained decomposition. We consider batch decompositions that reduce similarity \emph{across} batches and use the Lov\'asz local lemma to obtain bounds for the minimum size of a decomposition with a given pairwise dissimilarity. We then consider unique batch decomposition with a given relaxation parameter and obtain upper and lower ensemble bounds for the minimum size. Throughout, we use Binomial random graphs to obtain the desired ensemble bounds, by conditioning on the number of similar \emph{pairs} of data points.
    
[^57]: 时间正则化学习：空间与时间上均局部化的自监督学习

    Temporal Regularized Learning: Self-Supervised Learning Local in Space and Time

    [https://arxiv.org/abs/2611.17019](https://arxiv.org/abs/2611.17019)

    提出TeReL方法，通过逐样本的纯局部更新（Hebbian前馈与反Hebbian侧向连接）实现空间和时间上均局部化的VICReg式自监督学习，本质上是一种软化的慢特征分析。

    

    诸如VICReg等自监督目标通过使相关输入相似来学习表示，同时利用方差和协方差惩罚来防止表示坍塌。这些惩罚是对一批样本的统计量，因此层局部变体通常仅在空间上局部，而在时间上并不局部。我们提出了时间正则化学习，它以逐样本的方式优化VICReg风格的目标，并采用纯局部更新：每次更新仅使用当前层和前一层的活动以及一个固定大小的运行状态。时间上相邻的输入作为正样本对，这使得该目标成为慢特征分析的一种软形式。每个神经元均值和方差的运行估计，加上每层一个侧向连接矩阵，取代了批次统计量；在求导过程中将其保持固定，使梯度转化为Hebbian前馈更新和反Hebbian侧向更新。我们证明了两层网络……

    arXiv:2611.17019v2 Announce Type: replace  Abstract: Self-supervised objectives such as VICReg learn representations by making related inputs similar, while variance and covariance penalties prevent collapse. These penalties are statistics over a batch of samples, so layer-local variants, too, are typically local in space but not in time. We introduce Temporal Regularized Learning (TeReL), which optimizes a VICReg-style objective one sample at a time with purely local updates: each update uses only the current activity of its own and the preceding layer and a fixed-size running state. Temporally adjacent inputs serve as positive pairs, which makes the objective a soft form of Slow Feature Analysis (SFA). Running estimates of each neuron's mean and variance, together with one lateral matrix per layer, replace the batch statistics; held fixed during differentiation, they turn the gradient into a Hebbian feedforward update and an anti-Hebbian lateral update. We show that a two-layer netwo
    
[^58]: 论自回归语言模型序列级监督学习的困难性

    On the Difficulty of Sequence-Level Supervised Learning for Autoregressive Language Models

    [https://arxiv.org/abs/2611.16736](https://arxiv.org/abs/2611.16736)

    本文证明了将推理时解码过程纳入训练的两种序列级监督损失函数（基于贪婪解码和自回归采样）都存在无下界的根本性缺陷，并通过自回归Gumbel技巧揭示了二者之间的内在联系。

    

    在监督学习阶段，自回归语言模型通常通过最小化训练数据的负对数似然来进行训练，该目标通过教师强制分解为令牌级交叉熵之和。这一目标简单且高效，但可能存在所谓的“曝光偏差”问题：即模型训练方式与推理时的使用方式之间存在不匹配。那么，我们能否设计出将推理时解码过程（如贪婪解码或自回归采样）纳入其中的序列级损失函数呢？我们证明，两种自然的尝试会导致存在根本性缺陷的损失函数：它们是无下界的。我们进一步证明，这两个有缺陷的损失函数之间存在着紧密联系：利用自回归Gumbel技巧，基于采样的损失是Gumbel扰动下贪婪损失的期望值。由于这种病态源于解码后的序列对t（摘要在此处截断）

    arXiv:2611.16736v2 Announce Type: replace  Abstract: During the supervised learning phase, autoregressive language models are typically trained by minimizing the negative log-likelihood of the training data, which decomposes as a sum of token-level cross-entropies via teacher forcing. This objective is simple and efficient but potentially suffers from so-called exposure bias: a mismatch between the way the model is trained and the way it is used at inference time. Can we instead design sequence-level loss functions that incorporate inference-time decoding procedures, such as greedy decoding or autoregressive sampling? We show that two natural attempts lead to loss functions that are fundamentally flawed: unbounded below. We further show that these two flawed loss functions are intimately connected: using autoregressive Gumbel tricks, the sampling based loss is the expected greedy loss under Gumbel perturbations. Since the pathology stems from the dependence of the decoded sequence on t
    
[^59]: 改进的弹性权重巩固：作为持续学习的优化约束

    Improved Elastic Weight Consolidation as an Optimization Constraint for Continual Learning

    [https://arxiv.org/abs/2611.16039](https://arxiv.org/abs/2611.16039)

    本文将持续学习重新表述为一种优化约束（CLOC），证明EWC的经验Fisher惩罚是该约束的一阶形式，由此揭示EWC对已拟合良好的样本权重过低的缺陷，并提出改进的EWC方法。

    

    基于正则化的持续学习通过对改变先前任务中重要参数的行为施加惩罚来缓解灾难性遗忘。弹性权重巩固是其中的标准方法，其二次惩罚项以经验Fisher信息矩阵加权，这一做法通常被解释为对损失曲率的近似。我们则从“在新任务上训练应当保留什么”这一角度出发来推导该惩罚项。在这一表述中——持续学习作为一种优化约束——新任务的训练必须使已学习行为的选定泛函保持不变，而该约束的局部近似即为所存储的惩罚项。我们证明：当被保留的泛函为每个旧样本的损失时，EWC的经验Fisher恰好是该约束的一阶形式。这一推导揭示出一个缺陷：EWC按照每个样本损失梯度的幅值对其进行加权，因此模型已拟合得很好的样本几乎不产生贡献，尽管后续训练仍可能遗忘这些样本。改进的EW（摘要在此处截断）

    arXiv:2611.16039v2 Announce Type: replace  Abstract: Regularization-based continual learning reduces catastrophic forgetting by penalizing changes to parameters that mattered for earlier tasks. Elastic Weight Consolidation (EWC), a standard instance, weights its quadratic penalty by the empirical Fisher, usually justified as approximating loss curvature. We instead derive the penalty from what training on a new task should preserve. In this formulation, Continual Learning as an Optimization Constraint (CLOC), new training must keep chosen functionals of learned behavior unchanged, and the local approximation of this constraint is the stored penalty. We show that EWC's empirical Fisher is exactly the first-order form of this constraint when the preserved functional is each old sample's loss. The derivation exposes a flaw: EWC weights each sample by the scale of its loss gradient, so samples the model fits well barely contribute, although later training can still forget them. Improved EW
    
[^60]: 预先公告的变点使保形可靠性与频率校准相分离

    Announced Breaks Separate Conformal Reliability from Frequency Calibration

    [https://arxiv.org/abs/2610.23917](https://arxiv.org/abs/2610.23917)

    本文证明预先公告的结构性变点使“下一区间的保形可靠性”与“封闭窗口的频率校准”这两种重置审计目标可以分离，并提出在公告时点重启经验分位数与自适应递归的方法，前者达到1/√n的minimax可靠性速率、后者达到1/n的频率差距，且实验中覆盖率误差显著优于自适应基线（0.035对0.052）。

    

    许多结构性变点会提前公告，然而自适应保形方法只能从自身的预测误差中获知它们。在公告日期进行重置只能服务于两种审计之一：下一区间的保形可靠性，或一个封闭窗口的频率校准，而这两者可能不一致。对于独立得分，在变点边界处重启经验分位数可以达到极小化极大（minimax）可靠性速率，其阶为断裂后样本数的平方根分之一（1/√n）。在得分有界且递归下限累计增量有界的条件下，重启自适应递归可以达到阶为断裂后样本数分之一（1/n）的频率差距，但该差距对下一区间没有任何约束作用。在一次较大的合成变点之后，重启分位数在第50次预测时的覆盖率误差为0.035，而五个自适应基线中最好者为0.052。在每次合成变点之后的前50次预测中，重启递归都具有最小的频率差距。真实变点大多是……（原文在此截断）

    arXiv:2610.23917v2 Announce Type: replace-cross  Abstract: Many structural breaks are announced in advance, yet adaptive conformal methods learn of them only from their own errors. A reset at the announced date serves one of two audits, conformal reliability of the next interval or frequency calibration of a closed window, and the two can disagree. For independent scores, restarting the empirical quantile at the boundary attains the minimax reliability rate, of order one over the square root of the post-break count. Restarting an adaptive recursion attains a frequency gap of order one over that count on bounded scores when its floor adds a bounded total. That gap bounds nothing about the next interval. After a large synthetic break, the restarted quantile's coverage error at the 50th forecast is 0.035, against 0.052 for the best of five adaptive baselines. The restarted recursion has the smallest frequency gap over the first 50 forecasts after every synthetic break. Real breaks are mos
    
[^61]: 基于方差保持桥的稀疏观测随机场潜在退火后验采样

    Latent Annealed Posterior Sampling with a Variance-Preserving Bridge for Sparsely Observed Random Fields

    [https://arxiv.org/abs/2610.20659](https://arxiv.org/abs/2610.20659)

    提出LAPS方法，利用流匹配学习潜在先验并在潜在空间中进行退火后验采样，结合方差保持桥控制协方差膨胀，从而高效解决稀疏观测随机场后验推断中空间不确定性刻画与计算成本增长的难题。

    

    从稀疏观测中对随机场进行后验推断面临两大挑战：如何准确刻画复杂的空间不确定性，以及如何避免条件采样成本随空间网格加密而大幅增长。为解决这些挑战，我们提出了潜在退火后验采样（LAPS）。LAPS首先将随机场映射到降维后的潜在空间，同时利用由Branch网络和Trunk网络组成的坐标解码器，实现对任意空间位置的场值评估。随后采用流匹配学习潜在先验分布。基于该学习到的先验，后验采样完全在潜在空间中进行，观测似然通过退火过程逐步融入采样。为控制局部协方差膨胀，我们引入了一种方差保持（VP）桥，在传输的潜在状态与目标状态之间分配协方差预算。

    arXiv:2610.20659v1 Announce Type: cross  Abstract: Posterior inference for random fields from sparse observations presents two major challenges: accurately characterizing complex spatial uncertainty and preventing conditional sampling costs from growing substantially as the spatial grid is refined. To address these challenges, we propose latent annealed posterior sampling (LAPS). LAPS first maps the random field into a latent space with reduced dimensionality, while a coordinate decoder composed of Branch and Trunk networks enables field evaluation at arbitrary spatial locations. Flow matching is then used to learn the latent prior distribution. Based on this learned prior, posterior sampling is performed entirely in the latent space, with the observation likelihood incorporated progressively through an annealing procedure. To control local covariance inflation, we introduce a variance-preserving (VP) bridge that allocates a covariance budget between the transported latent state and a 
    
[^62]: 基于比例包含边际效应的变量重要性度量

    Measuring Variable Importance Using Proportional Inclusive Marginal Effects

    [https://arxiv.org/abs/2610.17053](https://arxiv.org/abs/2610.17053)

    提出了PRIME（比例包含边际效应）这一新的变量重要性度量方法，通过无模型变量选择技术仅纳入活跃变量进行计算，克服了PME仅适用于确定性模型且计算成本高的局限，能从含噪声数据中准确可靠地量化变量重要性。

    

    基于Shapley的度量方法在统计学和机器学习中被广泛用于衡量变量重要性。然而，这些度量方法存在一个严重问题——当一个非活跃变量与活跃变量相关时，该非活跃变量可能获得正的重要性度量值。Herin等人最近提出了一种称为比例边际效应（PME）的度量方法来缓解这种虚假效应。然而，PME是为确定性模型开发的，且计算成本很高。本文提出了一种称为比例包含边际效应（PRIME）的新度量方法，克服了PME的这两个局限性。PRIME的关键思想是使用一种新颖的无模型变量选择技术来识别活跃变量，并仅将这些变量纳入重要性度量的计算中。对测试函数和真实数据集的分析表明，PRIME能够从含噪声的输入-输出数据中准确、可靠地量化变量重要性。

    arXiv:2610.17053v1 Announce Type: new  Abstract: Shapley-based measures are widely used for measuring variable importance in statistics and machine learning. However, these measures have a serious issue--a nonactive variable can get a positive importance measure when it is correlated with an active variable. Herin et al. recently proposed a measure called proportional marginal effects (PME) that mitigates this spurious effect. However, PME is developed for deterministic models and has high computational cost. In this article, we propose a new measure called proportional inclusive marginal effects (PRIME) that overcomes these two limitations of PME. The key idea in PRIME is to use a novel model-free variable selection technique to identify the active variables and include only them in the computation of the importance measure. Analysis of test functions and real datasets show that PRIME can accurately and reliably quantify variable importance from noisy input-output data.
    
[^63]: 面向摊销式条件可解释性的核化参数敏感性图谱

    Kernelized Parameter Sensitivity Atlases for Amortized Conditional Interpretability

    [https://arxiv.org/abs/2610.12737](https://arxiv.org/abs/2610.12737)

    该论文提出 KPSA，通过经验神经正切核的分块分解构建参数组敏感性的核化图谱，用轻量级核函数摊销式地为任意新输入估计敏感性，从而以远低于逐样本反向传播的成本实现输入条件化的模型可解释性。

    

    基于梯度的归因方法能够提供关于模型参数的输入特定信息，但通过反向传播为大量样本计算归因代价高昂，而静态重要性得分则丢弃了对输入的依赖性。我们从经验神经正切核的分块分解出发推导出 KPSA（核化参数敏感性图谱）。它度量归一化的参数组敏感性，将这些敏感性来源嵌入一个显式的查询空间，并使用轻量级的核函数为未见过的输入估计敏感性。我们推导了区分有限来源误差与不可约表示误差的理论界，并给出了基于间隔（margin）的 top-k 恢复保证。本文研究了两个可解释性应用。其一，KPSA 在视觉、时序预测和蛋白质建模等多种设置中识别出对扰动高度敏感的参数组。在参数预算跨越五倍范围（0.02%–0.1%）的情况下，其选定的参数组可达到扰动效果的 39%–85%（原文在此处截断）。

    arXiv:2610.12737v1 Announce Type: new  Abstract: Gradient-based attribution provides input-specific information about model parameters, but computing it via backpropagation for many samples is expensive, while static importance scores discard input dependence. We derive KPSA (Kernelized Parameter Sensitivity Atlas) from a block decomposition of the empirical neural tangent kernel. It measures normalized parameter-group sensitivity, embeds those sources in an explicit query space, and uses a lightweight kernel to estimate sensitivities for unseen inputs. We derive bounds that separate finite-source from irreducible representation error and give margin-based top-$k$ recovery. This paper studies two interpretability applications. Firstly, KPSA identifies parameter groups with high susceptibility to perturbations across vision, forecasting, and protein-modeling setups. Across a fivefold range of parameter budgets ($0.02$-$0.1\%$), its selected groups attain $39$-$85\%$ of the perturbation 
    
[^64]: 解释对抗训练神经网络的显著性图稀疏性

    Explaining the Saliency Map Sparsity of Adversarially-Trained Neural Networks

    [https://arxiv.org/abs/2610.10666](https://arxiv.org/abs/2610.10666)

    本文首次为对抗训练神经网络梯度显著性图的稀疏性现象提供了理论解释，证明随着数据点和神经元数量增长，网络的最小化解收敛到具有最小梯度和Barron范数的贝叶斯分类器，从而自然产生了稀疏性。

    

    理解深度神经网络为何做出某个特定预测，对于其安全部署至关重要。在计算机视觉领域，显著性图通过突出显示对预测最具影响力的图像区域，仍然是一种被广泛使用的解释形式。一个经验性观察是，对抗训练神经网络的梯度显著性图呈现出明显的稀疏性。本文针对两层ReLU网络对这一现象提出了理论解释。我们建立在已确立的等价性之上——即对抗训练等价于在经验风险最小化中加入权重衰减惩罚项以及一个额外的对抗全变差项（该等价性对某些损失函数成立）。随着数据点和神经元数量的增长，且正则化参数以适当的速率趋于零，我们证明了最小化解收敛到具有最小梯度和Barron范数的贝叶斯分类器。稀疏性的出现是因为，对于对抗训练……

    arXiv:2610.10666v1 Announce Type: new  Abstract: Understanding why deep neural networks make a given prediction is of great importance for their safe deployment. In computer vision, saliency maps, which highlight the image region most influential for a prediction, remain a widely-used form of explanation. An empirical observation is the apparent sparsity of gradient saliency maps of adversarially-trained neural networks. In this paper, we propose a theoretical explanation of this phenomenon for two-layer ReLU networks. We build on the established equivalence of adversarial training to the minimization of the empirical risk with weight-decay penalization and an added adversarial total variation term -- valid for certain loss functions. As the number of data points and neurons grows and the regularization parameters are sent to zero at appropriate rates, we prove that minimizers converge to a Bayes classifier with minimal gradient and Barron norm. Sparsity appears since for adversarial t
    
[^65]: 基于神经网络与Euler近似的加权亚分数布朗运动驱动的随机微分方程的统计推断

    Inference for stochastic differential equations driven by weighted sub-fractional Brownian motion using neural networks and the Euler approximation

    [https://arxiv.org/abs/2610.00793](https://arxiv.org/abs/2610.00793)

    本文提出将神经网络与Euler近似相结合，对由加权亚分数布朗运动这类高斯过程驱动的随机微分方程进行统计推断，从离散观测数据中估计漂移系数、扩散系数与噪声协方差。

    

    我们考虑从高斯过程驱动的随机微分方程的离散观测中对漂移系数、扩散系数和噪声协方差进行估计。对于固定的观测时间区间 T>0 和已知的初始状态 x₀∈ℝ，我们研究方程 dX_t = a(X_t)dt + σ(X_t)dZ_t^{β,f}，其中 X₀ = x₀，0 ≤ t ≤ T。这里 a:ℝ→ℝ 是漂移系数，σ:ℝ→(0,∞) 是扩散系数，而 Z^{β,f} 是属于加权亚分数布朗运动族的中心化高斯过程，其协方差为 Cov(Z_s^{β,f}, Z_t^{β,f}) = ∫₀^{s∧t} f(r)q_β(s−r, t−r)dr，其中 0 ≤ s, t ≤ T，s∧t = min{s,t}。时间权重 f:[0,T]→[0,∞) 是可测、有界且几乎处处为正的函数，β∈(0,…（原文摘要在此处截断）

    arXiv:2610.00793v1 Announce Type: new  Abstract: We consider the estimation of drift, diffusion, and noise covariance from discrete observations of stochastic differential equations driven by Gaussian processes. For a fixed observation horizon $T>0$ and a known initial state $x_0\in\mathbb R$, we study \begin{equation*}   dX_t=a(X_t)\,dt+\sigma(X_t)\,dZ_t^{\beta,f},   \qquad X_0=x_0,\quad 0\leq t\leq T. \end{equation*} \smallskip\noindent Here $a:\mathbb R\to\mathbb R$ is the drift coefficient, $\sigma:\mathbb R\to(0,\infty)$ is the diffusion coefficient, and $Z^{\beta,f}$ is a centered Gaussian process from the weighted sub-fractional Brownian family, with covariance \begin{equation*}   \operatorname{Cov}(Z_s^{\beta,f},Z_t^{\beta,f})   =\int_0^{s\wedge t} f(r)q_\beta(s-r,t-r)\,dr,   \qquad 0\leq s,t\leq T. \end{equation*} \smallskip\noindent Here $s\wedge t=\min\{s,t\}$. The temporal weight $f:[0,T]\to[0,\infty)$ is measurable, bounded, and positive almost everywhere, and $\beta\in(0,
    
[^66]: 基于模拟退火的三阶朗之万动力学在非凸优化中的全局收敛性

    Global Convergence of Third-Order Langevin Dynamics for Non-Convex Optimization via Simulated Annealing

    [https://arxiv.org/abs/2609.28611](https://arxiv.org/abs/2609.28611)

    该论文证明了在模拟退火框架下，采用固定摩擦与递减噪声的三阶朗之万动力学在非凸优化中可依概率收敛到全局最小值，并给出了离散化格式保持该收敛速率的充分步长条件。

    

    我们研究了在固定摩擦力与递减噪声的模拟退火框架下，三阶朗之万动力学用于非凸优化的全局收敛性保证。一个显式的三块扭曲熵结构将耗散从含噪的辅助变量传递到整个状态空间。在耗散性、正则性以及低温泛函不等式假设下，对数冷却调度使目标值以势垒控制的动力学速率依概率收敛到全局最小值。对于精确力积分和中点三阶段离散化格式，多项式递减的步长可在物理时间尺度上保持该收敛速率。三次局部端点估计给出了比现有冻结力动力学结果更宽松的充分步长条件。与单梯度UBU积分器的比较表明，在相同的强耦合分析框架下，其中心化随机局部误差会带来更小的充分（步长条件要求）。

    arXiv:2609.28611v1 Announce Type: cross  Abstract: We study global convergence guarantees of third-order Langevin dynamics for non-convex optimization via simulated annealing with fixed friction and decreasing noise. An explicit three-block distorted entropy transfers dissipation from the noisy auxiliary variable to the full state. Under dissipativity, regularity, and low-temperature functional-inequality assumptions, logarithmic cooling drives the objective values to the global minimum in probability at the barrier-controlled kinetic rate. For the exact-force-integral and midpoint three-stage discretizations, polynomially decreasing steps preserve this rate on the physical time scale. The cubic local endpoint estimate gives a less restrictive sufficient step-size condition than the available frozen-force kinetic result. A comparison with the one-gradient UBU integrator shows how its centered stochastic local error leads, under the same strong-coupling analysis, to a smaller sufficient
    
[^67]: DAG ReLU网络路径提升雅可比矩阵的秩与计算

    Rank and computation of the pathlifting Jacobian of a DAG ReLU network

    [https://arxiv.org/abs/2609.18682](https://arxiv.org/abs/2609.18682)

    本文通过对骨架矩阵进行初等归纳证明了DAG ReLU网络路径提升雅可比矩阵的秩，并提出了一种无需反向传播、计算成本更低的雅可比矩阵计算方法。

    

    本文通过对网络的隐藏节点数量进行归纳，为DAG ReLU网络的路径提升雅可比矩阵的秩提供了一个自包含的证明。实际上，这种归纳是初等的，关键方法在于考虑网络的骨架矩阵（一个编码网络路径的稀疏矩阵），并将其中的一个隐藏神经元的表示转换为输出节点。该证明依赖于一些中间命题，这些命题将路径提升、其雅可比矩阵、网络参数及其骨架矩阵联系起来，除了能够得出路径提升雅可比矩阵秩的结论外，还提供了一种无需反向传播即可计算该矩阵的方法，其计算成本在实践中比常规的反向传播高效得多。本文附带一个Python模块，该模块实现了论文中针对前馈网络的各个命题，并用于实验性地量化计算……

    arXiv:2609.18682v1 Announce Type: cross  Abstract: This paper provides a self-contained proof of the rank of the pathlifting Jacobian of a DAG ReLU network by performing an induction on the network's number of hidden nodes. In fact, the induction is elementary, and the key recipe is to consider the skeleton matrix of the network, a sparse matrix encoding the network paths, and transform the representation of one of its hidden neurons into an output node. The proof relies on intermediate propositions which link the pathlifting, its Jacobian, the network parameters, and its skeleton matrix, which, on top of permitting to conclude on the rank of the pathlifting Jacobian, also provide a way to compute it without backpropagation and whose computation cost is super efficient in practice compare to usual backpropagation. The paper is provided with a Python module that implements the different propositions of the paper for feed forward networks and is used to experimentally quantifies the comp
    
[^68]: 一种联合建模缺失性、测量误差与异质性的深度潜变量框架

    A Deep Latent Variable Framework for Jointly Modeling Missingness, Measurement Error, and Heterogeneity

    [https://arxiv.org/abs/2608.30040](https://arxiv.org/abs/2608.30040)

    提出了一种基于深度潜变量表示的统一概率框架，通过层次树路由变分自编码器联合处理缺失数据、测量误差和群体异质性三类问题，并借助重新汇聚路由机制在子群体间共享参数以提升统计效率。

    

    缺失数据、测量误差和群体异质性是现代观察性研究与机器学习应用数据分析中普遍存在的挑战。尽管这些问题经常共存并相互作用，但现有工作往往将它们分开处理。我们提出了一个统一的概率框架，利用深度潜变量表示来联合应对这些问题。所提出的方法将一种新颖的层次树路由变分自编码器与模式感知的潜变量表示以及基于校准的去噪机制相结合。该框架兼容包括MCAR、MAR和MNAR在内的多种缺失数据机制，同时能够学习子群体特有和全局共享的潜变量结构。所引入的重新汇聚路由机制使选择性参数能够在相关子群体之间共享，从而提供了灵活性并提升了统计效率。仿真……

    arXiv:2608.30040v1 Announce Type: cross  Abstract: Missing data, measurement error, and population heterogeneity are pervasive challenges in analyzing data arising from modern observational studies and machine learning applications. Although these problems frequently coexist and interact, they are often treated separately in existing works. We propose a unified probabilistic framework that jointly addresses these issues utilizing deep latent variable representation. The proposed method integrates a novel hierarchical tree-routed variational autoencoder with pattern-aware latent representations and calibration-based denoising. The framework accommodates missing data mechanisms, including MCAR, MAR, and MNAR, while simultaneously learning subgroup-specific and globally shared latent structure. The introduced reconvergent routing mechanism enables selective parameters to be shared across related subpopulations, which offers flexibility as well as improved statistical efficiency. Simulatio
    
[^69]: 复杂神经网络下降的Kähler景观与包括Calabi-Yau流形搜索与消灭的保证

    K\"ahler landscapes for complex neural network descents and guarantees including a search and destroy of the Calabi-Yau manifold

    [https://arxiv.org/abs/2608.19584](https://arxiv.org/abs/2608.19584)

    本文提出了一种在复参数神经网络中使用Kähler信息度量和自然梯度下降的新方法，并针对Calabi-Yau流形上的不良曲率条件提供了理论保证，通过几何定义的全局势实现了搜索与消灭策略。

    

    我们研究复参数化网络的景观。我们的方法受参数的信息论流形视角以及经典优化保证的启发，尽管涉及复杂几何变体，如通过Dolbeault渐近。下降路径在交叉熵下承认Kähler信息度量，通过Wirtinger Hessian作用于对数似然势。我们关注一种下降更新规则，采用自然梯度下降，通过逆度量缩放的微分损失，使下降路径保持在全纯切丛中。我们强调Calabi-Yau信息流形，这些流形通过不良曲率条件的景观提供了理论保证。在Calabi-Yau度量下，特别是在非紧致设置中，具有全局势而非调用Calabi猜想的拓扑要求，一个楔入无处消失的全纯...

    arXiv:2608.19584v1 Announce Type: new  Abstract: We study landscapes for complex-parameterized networks. Our approach is motivated with an information-theoretic manifold perspective of the parameter and via classical optimization guarantees although of complex geometric variety such as through Dolbeault asymptotics. The descent path admits a K\"ahler information metric under a cross-entropy via the Wirtinger Hessian on the log-likelihood potential. We restrict attention to a descent update rule with natural gradient descent via a differentiated loss scaled by the inverse metric, so the descent path remains in the holomorphic tangent bundle. We emphasize Calabi-Yau information manifolds which profane theoretical guarantees via an ill-curvature-conditioned landscape. Under a Calabi-Yau metric, specifically in a non-compact setting with a global potential so defined geometrically rather than invoking the topological requirements of the Calabi conjecture, a wedged nowhere-vanishing holomor
    
[^70]: 面向相位恢复的随机初始化近似消息传递算法

    Approximate Message Passing with Random Initialization for Phase Retrieval

    [https://arxiv.org/abs/2608.01654](https://arxiv.org/abs/2608.01654)

    本文对随机初始化的近似消息传递（AMP）算法在无噪声相位恢复中的轨迹给出了非渐近刻画，证明其可同时达到信息论弱恢复阈值和强恢复阈值，性能与谱初始化AMP相当。

    

    随机初始化的一阶迭代算法简单、应用广泛，并在许多高维推断问题中被证明经验上有效，但与那些从更精细的、针对特定问题的初始化（如谱初始化）出发的算法相比，其理论行为仍然远未得到充分理解。本文通过研究针对无噪声相位恢复（一个典型的非凸推断问题）的随机初始化近似消息传递（AMP）算法，为缩小这一差距迈出了一步。我们给出了从随机初始化出发的AMP轨迹的非渐近、增长视野刻画，并证明它同时达到了信息论的弱恢复阈值 δ_weak=1/2 和 AMP 的强恢复阈值 δ_str≈1.13，与已针对谱初始化AMP所建立的恢复阈值相匹配。关键的技术要素有两方面：（i）一个非渐近……

    arXiv:2608.01654v2 Announce Type: replace-cross  Abstract: Randomly initialized first-order iterative algorithms are simple, widely used, and empirically effective across many high-dimensional inference problems, yet their theoretical behavior remains far less understood than that of algorithms started from more elaborate, problem-specific initializations such as spectral initialization. This paper takes a step toward closing this gap by studying randomly initialized approximate message passing (AMP) for noiseless phase retrieval, a canonical nonconvex inference problem. We give a non-asymptotic, growing-horizon characterization of the AMP trajectory from random initialization, and show that it achieves both the information-theoretic weak recovery threshold $\delta_{\rm weak}=1/2$ and the AMP strong recovery threshold $\delta_{\rm str}\approx1.13$, matching the recovery thresholds established for spectrally initialized AMP. The key technical ingredients are twofold: (i) a non-asymptoti
    
[^71]: 深度自适应贝叶斯筛选

    Deep Adaptive Bayesian Screening

    [https://arxiv.org/abs/2607.16927](https://arxiv.org/abs/2607.16927)

    DABS通过离线学习策略网络在严格实验预算下自适应地执行因子筛选，并创新性地在部署时整合后验推断，从而同时给出因子活性概率和效应大小的可信区间。

    

    我们提出了深度自适应贝叶斯筛选，这是一种在严格的实验预算下，于离散设计空间中执行自适应因子筛选的方法。DABS离线学习一个策略网络来顺序选择信息量大的实验，从而摊销贝叶斯最优实验设计。它能够处理二元设计，并通过带有强遗传性的尖峰-平板先验引入稀疏性和交互作用。该模型使用关于因子活性的信息对比下界进行训练，其中干扰效应大小和噪声方差被解析地积分掉。与先前的摊销贝叶斯设计方法不同，DABS还在部署阶段集成了后验推断，产生因子活性的后验概率以及效应大小的可信区间。我们在根据已发表的实验基准校准的模拟筛选问题上评估了DABS，结果表明其性能与最强的非自适应方法相当或更优。

    arXiv:2607.16927v2 Announce Type: replace-cross  Abstract: We introduce Deep Adaptive Bayesian Screening (DABS), a method for performing adaptive factorial screening in discrete design spaces under tight experimental budgets. DABS learns a policy network offline to sequentially select informative experiments, amortizing Bayesian Optimal Experimental Design. It handles binary designs, incorporates sparsity and interactions via a spike-and-slab prior with strong heredity. The model is trained using a contrastive lower bound on information about factor activity with nuisance effect sizes and noise variance analytically integrated out. Unlike prior amortized Bayesian design approaches, DABS also integrates posterior inference at deployment, yielding posterior probabilities of factor activity and credible intervals on effect sizes. We evaluate DABS on simulated screening problems calibrated to published experimental benchmarks, and show that it matches or outperforms the strongest non-adapt
    
[^72]: 从跳跃到签名：时间点过程的一种生成式方法

    From Jumps to Signatures: A Generative Method for Temporal Point Processes

    [https://arxiv.org/abs/2607.06652](https://arxiv.org/abs/2607.06652)

    该论文提出到达间隔嵌入方法，将签名理论从连续路径扩展到时间点过程的离散事件序列，并据此构建了首个基于签名的生成模型sigTPP，通过路径级损失对完整轨迹进行训练。

    

    粗路径签名是连续路径的通用特征映射，并通过期望签名来刻画路径分布。然而，这些理论保证并不能直接推广到时间点过程（TPP）的càdlàg路径上，这限制了签名方法在事件序列中的应用。此外，神经时间点过程模型（包括近期的生成式方法）优化的都是逐事件目标，缺乏全局的序列级损失函数，同时对变长事件序列的评估也缺乏分布差异度量方法。本文提出了一个统一的路径层面框架来解决这些局限。我们引入了到达间隔嵌入，这是一种从跳跃路径到有界变差连续路径的稳定、单射提升方法，将签名方法扩展到了离散事件序列。我们的理论贡献催生了sigTPP，这是首个基于签名的时间点过程生成模型，使用完整轨迹上的路径级损失进行训练。

    arXiv:2607.06652v2 Announce Type: replace  Abstract: Rough path signatures are a universal feature map for continuous paths and, via the expected signature, characterise path distributions. These guarantees do not directly extend to cadlag paths of Temporal Point Processes (TPPs), limiting the use of signature methods for event sequences. Furthermore, neural TPP models, including recent generative approaches, optimise per-event objectives with no global sequence-level loss, while evaluation of variable-length event sequences lacks distributional discrepancy measures. This paper proposes a common pathwise framework for addressing these limitations. We introduce the interarrival embedding, a stable, injective lift from jump paths to continuous paths of bounded variation, extending signature methods to discrete event sequences. Our theoretical contributions give rise to sigTPP, the first signature-based generative model for TPPs, trained using a path-level loss on complete trajectories. W
    
[^73]: 基于误设定规约的非平稳线性老虎机动态遗憾研究

    Dynamic Regret for Non-Stationary Linear Bandits via Misspecification Reductions

    [https://arxiv.org/abs/2607.02891](https://arxiv.org/abs/2607.02891)

    该论文提出一种分块规约方法，将非平稳线性老虎机中的参数漂移转化为上下文线性老虎机中的模型误设定问题，从而为一般性时变决策集建立了动态遗憾保证。

    

    许多在线决策问题同时涉及特定轮次的可行动作和漂移的奖励模型：符合条件的广告展示、可行的价格以及可用的治疗方案会随时间变化，而用户偏好、需求曲线和患者反应也可能不断演变。受这些应用的启发，我们研究了具有特定轮次可行决策集的非平稳线性老虎机问题。我们的关键贡献是一种分块规约方法，它在保留特定轮次决策基准的同时，将非平稳线性老虎机中的参数漂移与上下文线性老虎机中的模型误设定联系起来。这一联系使我们能够利用误设定自适应学习方法，为一般性的时变决策集获得动态遗憾保证。对于已知的路径长度预算 $P_T$，我们在遗忘型参数路径下为一般紧致、自适应且非预测性的决策集建立了这些保证，并针对 $K$ 臂上下文问题（摘要内容在此处截断）……

    arXiv:2607.02891v2 Announce Type: replace  Abstract: Many online decision-making problems involve both round-specific feasible actions and drifting reward models: eligible ad impressions, feasible prices, and available treatments can change over time, while user preferences, demand curves, and patient responses may evolve. Motivated by these applications, we study non-stationary linear bandits with round-specific feasible decision sets. Our key contribution is a blockwise reduction that connects parameter drift in non-stationary linear bandits to model misspecification in contextual linear bandits while preserving the round-specific decision benchmark. This connection allows us to leverage misspecification-adaptive learning to obtain dynamic-regret guarantees for general time-varying decision sets. For a known path-length budget $P_T$, we establish these guarantees for general compact, adaptive non-anticipating decision sets under an oblivious parameter path, and for $K$-armed contextu
    
[^74]: 超越全局分歧：贝叶斯推理中的局部质量视角

    Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference

    [https://arxiv.org/abs/2606.27090](https://arxiv.org/abs/2606.27090)

    本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。

    

    摘要：arXiv:2606.27090v1 公告类型：交叉 摘要：全局目标函数，如KL散度和ELBO，在贝叶斯推理中被广泛用于度量分布差异。本文研究这些目标函数未能直接捕捉的局部质量行为。我们引入并使用了两种数学工具：（1）质量指数，用于记录局部质量的多项式和对数衰减尺度；（2）正则化扩展KL（RE-KL），一种在存在奇异成分时可公式化的局部化散度。质量指数有助于刻画贝叶斯更新如何改变局部质量：（1）幂对数似然因子显式地改变它；（2）参数依赖的支持域或其平滑软化，可能通过参数值附近剩余的质量量来改变局部尺度。利用局部RE-KL，我们证明了在两种KL方向下比较局部小球质量的绝对、相对和方向性不等式。这些结果共同为局部质量行为提供了理论依据。

    arXiv:2606.27090v1 Announce Type: cross  Abstract: Global objectives, such as KL divergence and ELBO, are widely used in Bayesian inference for measuring distributional discrepancy. This paper studies their local-mass behaviour that is not directly captured by such objectives. We introduce and use two mathematical tools: (1) Mass Index for recording the polynomial and logarithmic decay scales of local mass, and (2) regularised extended KL (RE-KL), a set-localised divergence that can be formulated in the presence of singular components. Mass Indices help characterise how Bayesian updating changes local mass: (1) power-log likelihood factors shift it explicitly, and (2) parameter-dependent supports, or their smooth softenings, may change the local scale through the amount of mass that remains near the parameter value. Using local RE-KL, we prove absolute, relative, and directional inequalities for comparing local small-ball masses under the two KL directions. Together, these results prov
    
[^75]: 通过校准视角审视人机协作团队

    Human-AI Teaming Through the Lens of Calibration

    [https://arxiv.org/abs/2606.10906](https://arxiv.org/abs/2606.10906)

    本文从统计校准视角分析人机协作团队框架，发现组合人类与模型预测的现有方法会破坏人类的校准性，而委托方法虽能保持校准，却将负担转移到需要精细校准且难以达到要求的拒绝元模型上。

    

    我们通过统计校准的视角研究人机协作团队的模型。我们假设团队由一个AI模型和一个人类组成——两者均相对于特征空间的某种划分是经过校准的——并揭示校准假设如何传播到协作框架中。特别地，我们考虑两类框架：(i) 组合人类和模型的预测，或 (ii) 将预测责任委托给人类或模型。我们通过理论和实证结果表明，现有的组合方法无法保持人类的校准程度。而委托方法（正是由于委托这一行为本身）能够保持下游预测器的校准，但会将负担转移到决定由谁进行预测的拒绝元模型上。该拒绝模型必须经过足够精细的校准，才能定位每个团队成员更具优势的区域，而这一要求会随着人类专业水平的提升而增长，最终变得难以实现。

    arXiv:2606.10906v2 Announce Type: replace-cross  Abstract: We study models for human-AI teaming through the lens of statistical calibration. We assume the team consists of an AI model and human---both of which are calibrated with respect to some partitioning of the feature space---and expose how the calibration assumptions propagate into the teaming framework. In particular, we consider frameworks that either (i) combine human and model predictions or (ii) delegate prediction responsibility to either a human or model. We show via theoretical and empirical results that existing methods for combination do not preserve the human's degree of calibration. Methods for delegation (by the very act of delegation) preserve calibration of the downstream predictors but shift the burden onto the rejector meta-model that decides who predicts. The rejector must be calibrated finely enough to locate where each member is superior, a demand that grows with the human's expertise and becomes unattainable 
    
[^76]: 树引导的先识别后利用：对决老虎机中最优臂识别与遗憾最小化的统一框架

    Tree-Guided Identify-Then-Exploit: A Unified Framework of Best Arm Identification and Regret Minimization for Dueling Bandits

    [https://arxiv.org/abs/2606.01799](https://arxiv.org/abs/2606.01799)

    提出首个统一框架TG-ITE，通过树引导识别方法在无需传递性假设下以O(N)次比较找到高置信度候选臂，并针对最优臂识别、弱遗憾和强遗憾三种目标设计相应利用策略，首次以赢家停留式算法实现O(N)弱遗憾及BAI与弱遗憾的联合优化。

    

    我们研究了在孔多塞赢家假设且全局间隙为正的条件下，$N$ 臂随机对决老虎机问题，考虑了三种广泛采用的目标：最优臂识别（BAI）、弱遗憾和强遗憾。我们提出了树引导的先识别后利用框架（TG-ITE），据我们所知，这是首个能够统一处理所有这些目标的框架。在无需传递性假设的情况下，我们提出了一种共享的树引导识别方法，能够在 $O(N)$ 次比较内找到高置信度的当前最优臂。我们进一步提出了多种利用策略，利用这一热启动阶段来优化当前的具体目标。对于固定的全局间隙 $\Delta$，该方法使我们的方案能够：（1）构建首个赢家停留式算法以实现 $O(N)$ 的弱遗憾；（2）实现最优臂识别与弱遗憾的联合优化，两者均具有 $O(N)$ 的保证，消除了次优的 $O(\log N)$ 因子……

    arXiv:2606.01799v2 Announce Type: replace  Abstract: We study $N$-armed stochastic dueling bandits under the Condorcet-winner assumption with a positive global gap, where three widely adopted objectives are considered: best-arm identification (BAI), weak regret, and strong regret. We propose Tree-Guided Identify-Then-Exploit (TG-ITE), the first unified framework to tackle all these objectives to our knowledge. Without requiring transitivity assumptions, we propose a shared tree-guided identification approach to find a high-confidence incumbent within $O(N)$ comparisons. We further propose varied exploitation strategies to utilize this warm-start stage to optimize the specific objectives at hand. For a fixed global gap $\Delta$, this methodology enables our approach to (1) build the first winner-stays-style algorithm to achieve $O(N)$ weak regret; (2) realize the joint optimization of BAI and weak regret with $O(N)$ guarantees for both, eliminating the sub-optimal $O(\log N)$ factors of
    
[^77]: 免参数与群组条件在线保形预测

    Parameter-Free and Group Conditional Online Conformal Prediction

    [https://arxiv.org/abs/2606.00419](https://arxiv.org/abs/2606.00419)

    本文提出了一种免参数（无需学习率调优）的群组条件在线保形预测算法，能够在数据分布随时间变化的在线环境中为多个群组提供强大的覆盖保证，从而解决公平性问题。

    

    在数据特征和标签随时间变化的在线环境中，量化机器学习模型预测的不确定性是一项挑战。在线保形预测（OCP）方法通过为每个观测值生成预测区间来应对这一问题，并保证其中一定比例的区间包含真实标签，这一性质被称为边际覆盖率。然而，OCP方法可能会系统性地对属于某些群组（如由年龄或种族等属性定义的群组）的观测值产生错误覆盖，从而引发公平性方面的担忧。群组条件OCP（G-OCP）方法通过生成能够为多个群组提供覆盖保证的预测区间来解决这一问题。现有方法依赖于学习率来控制其对数据流变化的反应方式，但学习率在在线环境中难以调优。借助免学习率方法的最新进展，我们开发了一种免学习率的G-OCP算法，该算法具有强大的群组覆盖保证。

    arXiv:2606.00419v5 Announce Type: replace-cross  Abstract: Quantifying the uncertainty in predictions made by machine learning models is challenging in online settings where the characteristics of the data and labels change over time. Online conformal prediction (OCP) methods tackle this problem by producing intervals for each observation and guaranteeing that a proportion of them contain their labels, a property known as marginal coverage. OCP methods, however, may systematically miscover observations belonging to groups defined by attributes like age or race, raising fairness concerns. Group-conditional OCP (G-OCP) methods address this by producing intervals that provide coverage for multiple groups. Existing approaches rely on learning rates, which control how they react to changes in the data stream, but can be hard to tune in online settings. Leveraging advancements in learning-rate-free methods, we develop a learning-rate-free G-OCP algorithm with strong group-coverage guarantees
    
[^78]: 超越差异：基于比率的处理效应估计与调优

    Beyond Differences: Estimating and Tuning Ratio-Based Treatment Effects

    [https://arxiv.org/abs/2605.26288](https://arxiv.org/abs/2605.26288)

    本文提出Q-Learner方法，将二元结果的比率CATE估计转化为两个倾向性分类任务，仅需标准的log-loss超参数调优即可可靠地估计处理效应的水平与分布，尤其适合工业实践场景。

    

    当处理效应自然地表示为比率时——如医学、定价和营销等领域——基于比率的CATE τ(x) = E[Y|W=1,X=x] / E[Y|W=0,X=x] 是合适的目标量。然而常见的估计器要么除以估计的基线，要么施加参数形式，要么对结果进行回归建模，在这些情况下，更好的拟合并不一定意味着更好的效应估计。我们提出了Q-Learner，它将τ(x)表示为两个几率（odds）之比，从而将二元结果的比率CATE估计简化为两个倾向性分类任务。在合成数据和随机化真实世界数据的基准测试中，Q-Learner被证明特别适合工业实践。仅需对其log-loss进行标准的超参数调优，就足以使其在排序上表现良好，并比其他方法更可靠地把握效应的水平和分布；相比之下，例如常见的S-Learner需要昂贵的、针对效应的专门调优才能在排序上表现良好。

    arXiv:2605.26288v2 Announce Type: replace-cross  Abstract: When treatment effects are naturally expressed as ratios -- as in medicine, pricing, and marketing -- the ratio-based CATE $\tau(x) = \mathbb{E}[Y|W=1,X=x] / \mathbb{E}[Y|W=0,X=x]$ is the appropriate estimand. Yet the common estimators divide by an estimated baseline, impose a parametric form, or regress the outcome, where a better fit need not mean a better effect estimate. We introduce the Q-Learner, which writes $\tau(x)$ as a ratio of two odds, reducing ratio-CATE estimation for binary outcomes to two propensity classification tasks. In benchmarks on synthetic and randomized real-world data, the Q-Learner proves especially suited to industry practice. Standard hyperparameter tuning on its log-loss suffices to make it rank well, get the level of the effects right and their spread more reliably than the alternatives, whereas, for instance, the common S-Learner needs a costly, effect-specific tuning to rank well, often best, b
    
[^79]: 使用TMLE提高随机对照试验中亚组分析的效率

    Improving the Efficiency of Subgroup Analysis in Randomized Controlled Trials with TMLE

    [https://arxiv.org/abs/2605.15483](https://arxiv.org/abs/2605.15483)

    该论文提出两种利用同一试验中亚组外参与者信息的目标最大似然估计量（TMLE-PR与A-TMLE），在不依赖外部真实世界数据的前提下，提高随机对照试验中亚组分析的统计效率。

    

    随机对照试验中的亚组分析常常因样本量有限而导致统计效能不足。我们通过利用目标亚组之外的试验参与者来增强亚组内的估计，从而应对这一挑战。具体而言，我们研究了两种从同一试验中非亚组参与者借入信息的目标最大似然估计量（TMLE）：采用合并回归的TMLE（TMLE-PR）和自适应目标最大似然估计量（A-TMLE）。这两种估计量均能在不依赖任何外部真实世界数据的情况下实现信息共享，从而充分发挥试验本身的关键优势：最重要的是随机化治疗所带来的抗偏倚保护，此外还包括统一协调的数据收集以及一致的治疗与结局定义。本文提出的一般性策略直接推进了包括美国食品药品监督管理局（FDA）在内的关键监管机构的优先事项，通过改善……（原文摘要在此处截断）

    arXiv:2605.15483v2 Announce Type: replace-cross  Abstract: Subgroup analyses within randomized controlled trials are often underpowered due to limited sample sizes. We address this challenge by leveraging trial participants outside the subgroup of interest to augment estimation within the subgroup. Specifically, we study two Targeted Maximum Likelihood Estimators (TMLEs) that borrow information from non-subgroup participants within the same trial: a TMLE with pooled regression (TMLE-PR) and an Adaptive Targeted Maximum Likelihood Estimator (A-TMLE). Both estimators enable information sharing without relying on any external real-world data, thereby capitalizing on key strengths of the trial: most importantly, the protection against bias afforded by the randomized treatment, but also harmonized data collection, and consistent treatment and outcome definitions. The general strategy proposed here directly advances the priorities of key regulatory agencies, including the FDA, by improving t
    
[^80]: 基于多阶段平滑的时变网络连接非参数估计

    Nonparametric estimation of time-varying network connections by multi-stage smoothing

    [https://arxiv.org/abs/2605.06862](https://arxiv.org/abs/2605.06862)

    本文提出一种多阶段平滑估计方法，通过结合时间局部平滑与数据驱动的节点域平滑来非参数估计时变网络的边概率，能够同时捕捉网络连接的平滑时间演化与结构模式。

    

    我们考虑估计在多个时间点上观测的时变网络的潜在边概率问题。该概率结构由一个时变图函数表示，其在时间维度上满足Hölder光滑性条件，并在潜变量上满足分段Lipschitz条件。我们提出了一种多阶段平滑估计方法，首先对每条边应用时间局部平滑，然后使用从该方法改编的数据驱动邻域构造进行节点域平滑。当需要在整个时间域上保持一致精度时，还引入了额外的时间平滑步骤作为可选的改进。模拟研究展示了在不同生成模型下结合时间平滑与节点域平滑的优势。我们还将该方法应用于真实的时变网络数据集，结果表明该方法能够同时捕捉连接中的平滑时间演化特征和结构模式。

    arXiv:2605.06862v2 Announce Type: replace-cross  Abstract: We consider the problem of estimating the underlying edge probabilities of a time-varying network observed at multiple time points. The probability structure is represented by a time-varying graphon that satisfies temporal H\"older smoothness and piecewise Lipschitz conditions in the latent variables. We propose a multi-stage smoothing estimator that first applies temporal local smoothing to each edge and then performs node-domain smoothing using a data-driven neighborhood construction adapted from the method. An additional temporal smoothing step is introduced as an optional refinement when uniform accuracy over the entire time domain is required. Simulation studies demonstrate the benefits of combining temporal and node-domain smoothing under different generative models. We also apply the method to a real time-varying network dataset and show that it captures both smooth temporal evolution and structural patterns in the conne
    
[^81]: 训练时批归一化重塑分段仿射网络中的局部划分几何

    Training-Time Batch Normalization Reshapes Local Partition Geometry in Piecewise-Affine Networks

    [https://arxiv.org/abs/2605.04946](https://arxiv.org/abs/2605.04946)

    该论文证明了训练时BN层的切换超平面是批次质心超平面的平移，其偏移仅由标准化量与批次散度决定（与偏置和批次位置无关），并据此给出CPA网络局部线性区域数量的精确公式和下界，揭示BN从根本上重塑了网络的局部划分几何。

    

    批归一化（BN）对训练期间所实现函数的影响，相较于其对优化的影响，人们的理解仍然较少。我们通过在数据周围局部 $\ell_\infty$ 窗口中所诱导的仿射区域，研究连续分段仿射（CPA）网络中的训练时BN。以小批次为条件，具有可学习参数和一般CPA激活函数的BN层的每个切换超平面，都是经过批次质心的某个超平面的平移，其偏移量由一个标准化量乘以批次散度给出，与原始偏置无关。我们证明了在一般位置条件下，用窗口切割数和内部交叉点数表示窗口中区域数量的精确公式。对于BN层，该公式给出了用标准化偏移量表示的局部区域计数下界；该下界仅通过批次散度依赖于批次，而与偏置或批次位置无关。在没有BN的情况下，所能得到的最好此类下界是……

    arXiv:2605.04946v3 Announce Type: replace  Abstract: The effect of batch normalization (BN) on the function realized during training is less understood than its effect on optimization. We study training-time BN in continuous piecewise-affine (CPA) networks through the affine regions they induce in local $\ell_\infty$ windows around the data. Conditioned on a mini-batch, every switching hyperplane of a BN layer with learned parameters and a general CPA activation is a translate of a hyperplane through the batch centroid, at an offset given by a standardized quantity times the batch spread, independent of the raw bias. We prove an exact formula, valid under general position, for the number of regions in a window in terms of window cuts and interior intersections. For a BN layer it gives a lower bound on the local count in terms of the standardized offsets; the bound depends on the batch only through its spread and not on the biases or the batch location. Without BN the best such bound is
    
[^82]: PAIR-CI：面向不完整数据因果发现的校准条件独立性检验

    PAIR-CI: Calibrated Conditional Independence Testing for Causal Discovery with Incomplete Data

    [https://arxiv.org/abs/2605.04838](https://arxiv.org/abs/2605.04838)

    该论文提出PAIR-CI，一种通过配对置换设计将多重插补直接融入推断过程、使插补误差在损失差异中相互抵消的非参数条件独立性检验，从而解决了不完整数据因果发现中“先插补后检验”范式校准失准的问题。

    

    针对不完整数据的因果发现，标准的基于约束的范式——“先插补、后检验”——常常出现校准失准：当插补误差诱导出伪条件相关时，任何一致的条件独立性（CI）检验拒绝真原假设的概率都会趋近于1。我们提出了PAIR-CI，这是一种非参数条件独立性检验方法，通过配对置换设计将多重插补直接整合到推断过程中，从而恢复校准性。PAIR-CI比较包含与排除候选变量的交叉验证模型，同时两者接收相同的插补后条件集，使得插补误差在两者的损失差异中相互抵消，而不是污染检验统计量。一个方差估计量将一个可证明一致的交叉验证估计量嵌套于Rubin规则之内，由此得到的推断在多次插补极限下渐近有效，并且按照推导出的（符号……）保持保守。

    arXiv:2605.04838v2 Announce Type: replace-cross  Abstract: The standard constraint-based paradigm for causal discovery with incomplete data -- impute first, test second -- is frequently miscalibrated: any consistent conditional independence (CI) test rejects a true null with probability approaching 1 when imputation error induces spurious conditional dependence. We introduce PAIR-CI, a nonparametric CI test that restores calibration by integrating multiple imputation directly into the inferential procedure via a paired permutation design. PAIR-CI compares cross-validated models that include and exclude the candidate variable while receiving the same imputed conditioning set, so that imputation error cancels in their loss difference rather than contaminating the test statistic. A variance estimator that nests a provably consistent cross-validation estimator inside Rubin's rules yields inference that is asymptotically valid in the many-imputation limit and conservative by a derived, sign
    
[^83]: 深度网络的逐层几何逼近速率

    Layer-wise Geometric Approximation Rates for Deep Networks

    [https://arxiv.org/abs/2604.20219](https://arxiv.org/abs/2604.20219)

    本文构造了一个宽度为 $3dN+2d+4$、各隐藏层激活模式完全相同的单一深度网络，其逐层中间读出以由目标函数 $L^p$ 平移模（Hölder 条件下为 $O(N^{-\alpha\ell})$）控制的速率在逐级精细的尺度上逼近目标，从而为网络深度赋予了尺度依赖的全新解释。

    

    arXiv:2604.20219v3 公告类型：替换 摘要：深度被广泛认为是深度神经网络取得成功的核心因素，然而标准的逼近结果通常只给出网络最终输出的误差界。受经典多尺度展开与多级深度学习的启发，我们通过构造一个单一网络——其各层中间读出在逐级精细的尺度上逼近目标函数——赋予深度一个依赖于尺度的解释。对于任意 $f\in L^p([0,1]^d)$（$1\le p<\infty$）和任意整数 $N\ge2$，我们构造了一个宽度为 $3dN+2d+4$ 的单一网络，其每个隐藏层具有相同的激活模式，即前两个神经元使用正弦激活、其余使用 $\mathtt{ReLU}$ 激活。对每个 $\ell\ge0$，第 $\ell+4$ 层读出 $\Phi_\ell$ 的 $L^p$ 逼近误差被 $(2d+1)\omega_{f,p}(N^{-\ell})$ 所界定，其中 $\omega_{f,p}$ 表示 $f$ 的 $L^p$ 平移模。在 Hölder 条件 $\omega_{f,p}(t)\le Ct^\alpha$ 下……

    arXiv:2604.20219v3 Announce Type: replace  Abstract: Depth is widely viewed as a central contributor to the success of deep neural networks, yet standard approximation results typically give error bounds only for the final output of a network. Inspired by classical multiscale expansions and multigrade deep learning, we give depth a scale-dependent interpretation by constructing a single network whose intermediate readouts approximate the target at successively finer scales. For any $f\in L^p([0,1]^d)$ with $1\le p <\infty$ and any integer $N\ge2$, we construct a single network of width $3dN+2d+4$ with the same activation pattern in every hidden layer, using sine in the first two neurons and $\mathtt{ReLU}$ in the rest. For every $\ell\ge0$, the $L^p$ approximation error of the readout $\Phi_\ell$ at layer $\ell+4$ is bounded by $(2d+1)\omega_{f,p}(N^{-\ell})$, where $\omega_{f,p}$ denotes the $L^p$ translation modulus of $f$. Under the H\"older condition $\omega_{f,p}(t)\le Ct^\alpha$ 
    
[^84]: 基于截断正交多项式核的支持向量机的结构可解释性

    Structural interpretability in SVMs with truncated orthogonal polynomial kernels

    [https://arxiv.org/abs/2604.15285](https://arxiv.org/abs/2604.15285)

    该论文提出了正交表示贡献分析（ORCA）方法，利用截断正交多项式核SVM的有限标准正交多项式基展开，通过正交核贡献（OKC）指数精确量化各输入变量、交互阶数及多项式次数对决策函数的贡献，从而实现对支持向量机的结构可解释性分析。

    

    支持向量机（SVM）可以产生预测性能相似但内部结构不同的分类器，尤其是在各个解释变量的贡献及其交互作用方面存在差异。我们提出了正交表示贡献分析，这是一种针对由截断Christoffel–Darboux核的张量积构建的支持向量机的训练后分析方法。相应的再生核希尔伯特空间（RKHS）具有有限的标准正交多项式基，拟合决策函数的正则化部分在该基下具有精确的展开形式。展开系数的平方诱导出其RKHS范数平方的加性分解。基于这一表示，归一化的正交核贡献（OKC）指数量化了与总多项式次数、交互阶数以及输入变量贡献（包括单独贡献和通过交互作用产生的贡献）相关的RKHS范数平方的比例。

    arXiv:2604.15285v2 Announce Type: replace-cross  Abstract: Support vector machines (SVMs) can yield classifiers with similar predictive performance but different internal structures, particularly in the contributions of individual explanatory variables and their interactions. We introduce Orthogonal Representation Contribution Analysis (ORCA), a post-training methodology for SVMs constructed from tensor products of truncated Christoffel--Darboux kernels. The associated reproducing kernel Hilbert space (RKHS) admits a finite orthonormal polynomial basis in which the regularized component of the fitted decision function has an exact expansion. Squared expansion coefficients induce an additive decomposition of its squared RKHS norm. Based on this representation, normalized Orthogonal Kernel Contribution (OKC) indices quantify the fractions of the squared RKHS norm associated with total polynomial degrees, interaction orders, and input-variable contributions, both individually and through 
    
[^85]: 基于两阶段核岭回归的连续处理效应估计

    Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression

    [https://arxiv.org/abs/2604.13410](https://arxiv.org/abs/2604.13410)

    该论文提出一种两阶段核岭回归方法，先学习响应模型再构建校正分布偏移的伪结果，从而在存在混淆偏差的情况下估计连续处理效应，并利用平均效应函数更简单的结构特点达到了最优学习界限。

    

    我们研究估计连续处理的效应函数问题，该函数将每个处理值映射到总体平均结果。这一设定中的核心挑战是混淆：处理分配通常依赖于协变量，从而产生选择偏差，使得直接将响应对处理进行回归的方法不可靠。为解决这一问题，我们提出了一种两阶段核岭回归方法。在第一阶段，我们学习一个将响应表示为处理和协变量函数的模型；在第二阶段，我们利用该模型构建校正分布偏移的伪结果，然后拟合第二个模型来估计处理效应。尽管响应同时随处理和协变量变化，但通过对协变量取平均得到的效应函数通常要简单得多，而我们的估计器能够适应这种结构。我们的最优学习界限是在没有……（原文在此处截断）

    arXiv:2604.13410v3 Announce Type: replace-cross  Abstract: We study the problem of estimating the effect function for a continuous treatment, which maps each treatment value to a population-averaged outcome. A central challenge in this setting is confounding: treatment assignment often depends on covariates, creating selection bias that makes direct regression of the response on treatment unreliable. To address this issue, we propose a two-stage kernel ridge regression method. In the first stage, we learn a model for the response as a function of both treatment and covariates; in the second stage, we use this model to construct pseudo-outcomes that correct for distribution shift, and then fit a second model to estimate the treatment effect. Although the response varies with both treatment and covariates, the induced effect function obtained by averaging over covariates is typically much simpler, and our estimator adapts to this structure. Our optimal learning bounds are achieved withou
    
[^86]: 克服激励崩塌悖论

    Overcoming the Incentive Collapse Paradox

    [https://arxiv.org/abs/2603.27049](https://arxiv.org/abs/2603.27049)

    该论文证明了激励崩塌源于任何仅基于观测任务准确性的支付规则，并提出一种哨兵审计支付机制，能以有限成本维持严格为正且可控的人类努力水平，不受AI准确性的影响。

    

    AI辅助的任务委派日益普遍，然而在这类系统中，人类的努力成本高昂且通常无法被观测。Bastani和Cachon（2025）以及Sambasivan等人（2021）的最新研究表明，基于准确性的支付方案存在激励崩塌问题：随着AI准确性的提高，维持积极的人类努力需要无上限的支付。我们在一个预算受限的委托-代理框架中研究这一现象，其中策略性行动的人类代理的输出准确性取决于无法观测的努力程度。我们的第一个贡献是一个一般性的不可能性结果，表明激励崩塌不仅仅是简单线性支付的局限性，而是产生于任何仅基于观测任务准确性的支付规则。为了克服这一障碍，我们提出了一种哨兵审计支付机制，该机制能够以有限的成本强制执行严格为正且可控的人类努力水平，且不受AI准确性的影响。基于这种激励鲁棒的

    arXiv:2603.27049v3 Announce Type: replace-cross  Abstract: AI-assisted task delegation is increasingly common, yet human effort in such systems is costly and typically unobserved. Recent work by Bastani and Cachon (2025); Sambasivan et al. (2021) shows that accuracy-based payment schemes suffer from incentive collapse: as AI accuracy improves, sustaining positive human effort requires unbounded payments. We study this phenomenon in a budget-constrained principal-agent framework with strategic human agents whose output accuracy depends on unobserved effort. Our first contribution is a general impossibility result showing that incentive collapse is not merely a limitation of simple linear payments, but arises for any payment rule based only on observed task accuracy.To overcome this barrier, we propose a sentinel-auditing payment mechanism that enforces a strictly positive and controllable level of human effort at finite cost, independent of AI accuracy. Building on this incentive-robust
    
[^87]: 噪声敏感度指数控制单指标与多指标模型中的显著统计-计算差距

    A Noise Sensitivity Exponent Controls Large Statistical-to-Computational Gaps in Single- and Multi-Index Models

    [https://arxiv.org/abs/2603.17896](https://arxiv.org/abs/2603.17896)

    该论文提出由激活函数决定的噪声敏感度指数（NSE），证明这一简单量完全刻画并控制了单指标与多指标模型中大范围统计-计算差距的存在与大小。

    

    理解学习在统计上何时可行但在计算上却很困难，是高维统计中的一个核心挑战。在本工作中，我们在单指标和多指标模型的背景下研究这一问题，这两类函数被广泛研究，是探测机器学习方法在高维数据中发现特征能力的基准。我们的主要贡献是证明噪声敏感度指数（NSE）——一个由激活函数决定的简单量——在这些模型的广泛区域内支配统计-计算差距的存在性与大小。我们首先证明，在具有大加性噪声的单指标模型中，计算瓶颈的出现完全由NSE刻画。随后我们证明，同一指数控制着大型可分离多指标模型专业化转变中的统计-计算差距，其中个体……

    arXiv:2603.17896v2 Announce Type: replace-cross  Abstract: Understanding when learning is statistically possible yet computationally hard is a central challenge in high-dimensional statistics. In this work, we investigate this question in the context of single- and multi-index models, classes of functions widely studied as benchmarks to probe the ability of machine learning methods to discover features in high-dimensional data. Our main contribution is to show that a Noise Sensitivity Exponent (NSE) - a simple quantity determined by the activation function - governs the existence and magnitude of statistical-to-computational gaps within a broad regime of these models. We first establish that, in single-index models with large additive noise, the onset of a computational bottleneck is fully characterized by the NSE. We then demonstrate that the same exponent controls a statistical-computational gap in the specialization transition of large separable multi-index models, where individual 
    
[^88]: 强化学习解结器、困难平凡纽结与解结数

    RL unknotter, hard unknots and unknotting number

    [https://arxiv.org/abs/2603.07955](https://arxiv.org/abs/2603.07955)

    本文开发了基于强化学习的纽结图简化流水线，智能体通过学习Reidemeister移动策略成功解开“非常困难”的平凡纽结图，并通过自改进的工作簿驱动扩展系统性地改进了素纽结解结数的上界。

    

    我们开发了一个用于简化纽结图的强化学习流水线。经过训练的智能体学习移动建议和价值启发式方法，以执行Reidemeister移动。该流水线适用于任意纽结和链环；我们在“非常困难”的平凡纽结图上对其进行测试，并利用图膨胀方法在 $4_1\#9_{10}$ 上进行测试，其中我们研究了最近确立的令人惊讶的解结数上界3。此外，我们阐述了一种基于工作簿驱动的自改进流水线扩展，该扩展系统地改进了素纽结的解结数上界。

    arXiv:2603.07955v4 Announce Type: replace-cross  Abstract: We develop a reinforcement learning pipeline for simplifying knot diagrams. A trained agent learns move proposals and a value heuristic for navigating Reidemeister moves. The pipeline applies to arbitrary knots and links; we test it on ``very hard'' unknot diagrams and, using diagram inflation, on $4_1\#9_{10}$ where we investigate the recently established and surprising upper bound of three for the unknotting number. In addition, we explain a self-improving workbook-driven extension of the pipeline that systematically improves unknotting number upper bounds on the prime knots.
    
[^89]: 因果推断的广义贝叶斯方法

    Generalized Bayes for Causal Inference

    [https://arxiv.org/abs/2603.03035](https://arxiv.org/abs/2603.03035)

    提出了一种无需显式似然建模的广义贝叶斯因果推断框架，通过直接对因果估计目标设置先验并用识别驱动的损失函数进行更新，使现有基于损失的因果估计器具备完整的不确定性量化能力。

    

    不确定性量化是因果机器学习许多应用的核心，然而对因果效应进行有原则的贝叶斯推断仍然充满挑战。标准贝叶斯方法通常需要为数据生成过程指定概率模型，其中包括倾向得分和结局回归等高维讨厌参数。因此，标准后验容易受到强建模选择（包括复杂的先验设定）的影响。在本文中，我们提出了一个用于因果推断的广义贝叶斯框架。我们的框架避免了显式的似然建模；相反，我们直接在因果估计目标（causal estimands）上放置先验，并使用由识别驱动的损失函数对其进行更新，从而得到因果效应的广义后验。由此，我们的框架将现有的基于损失的因果估计器转化为具有完整不确定性量化的估计器。我们的框架灵活且……

    arXiv:2603.03035v2 Announce Type: replace-cross  Abstract: Uncertainty quantification is central to many applications of causal machine learning, yet principled Bayesian inference for causal effects remains challenging. Standard Bayesian approaches typically require specifying a probabilistic model for the data-generating process, including high-dimensional nuisance components such as propensity scores and outcome regressions. Standard posteriors are thus vulnerable to strong modeling choices, including complex prior elicitation. In this paper, we propose a generalized Bayesian framework for causal inference. Our framework avoids explicit likelihood modeling; instead, we place priors directly on the causal estimands and update these using an identification-driven loss function, which yields generalized posteriors for causal effects. As a result, our framework turns existing loss-based causal estimators into estimators with full uncertainty quantification. Our framework is flexible and 
    
[^90]: UMAP与其他降维方法的比较研究

    A Comparative Study of UMAP and Other Dimensionality Reduction Methods

    [https://arxiv.org/abs/2603.02275](https://arxiv.org/abs/2603.02275)

    本文系统比较了UMAP、有监督UMAP与PCA、SIR、t-SNE等多种降维方法的性能，发现有监督UMAP在分类任务中表现优异，但在回归场景下的应用存在明显局限。

    

    均匀流形近似与投影（UMAP）是一种被广泛使用的流形学习降维技术。本文通过全面的对比分析，研究了UMAP、有监督UMAP以及多种竞争性降维方法，包括主成分分析（PCA）、核PCA、切片逆回归（SIR）、核SIR以及t分布随机邻域嵌入（t-SNE）。尽管UMAP因能够保留局部和全局结构而受到广泛关注，但其有监督扩展形式，尤其是在回归场景下的应用，仍然探索不足。我们使用模拟数据和真实数据集，对有监督UMAP在回归和分类任务中的表现进行了系统评估，并通过低维嵌入上的预测精度来衡量其性能。结果表明，有监督UMAP在分类任务中表现良好，但在有效结合回归目标方面存在局限性。

    arXiv:2603.02275v3 Announce Type: replace  Abstract: Uniform Manifold Approximation and Projection (UMAP) is a widely used manifold learning technique for dimensionality reduction. This paper studies UMAP, supervised UMAP, and several competing dimensionality reduction methods, including Principal Component Analysis (PCA), Kernel PCA, Sliced Inverse Regression (SIR), Kernel SIR, and t-distributed Stochastic Neighbor Embedding, through a comprehensive comparative analysis. Although UMAP has attracted substantial attention for preserving local and global structures, its supervised extensions, particularly for regression settings, remain rather underexplored. We provide a systematic evaluation of supervised UMAP for both regression and classification using simulated and real datasets, with performance assessed via predictive accuracy on low-dimensional embeddings. Our results show that supervised UMAP performs well for classification but exhibits limitations in effectively incorporating r
    
[^91]: 掩码扩散模型的精确收敛速率

    Sharp Convergence Rates for Masked Diffusion Models

    [https://arxiv.org/abs/2602.22505](https://arxiv.org/abs/2602.22505)

    本文首次为掩码扩散模型中的欧拉采样方法建立了直接基于总变差 的精确收敛速率分析，克服了现有KL散度分析参数依赖松散、假设过强以及无法覆盖首达采样器（FHS）的局限。

    

    离散扩散模型在文本及其他符号领域取得了出色的实证表现，其中掩码（吸收率）变体已成为自回归模型的有力竞争替代方案。在现有采样器中，欧拉方法仍是许多应用中的标准选择，而最近，首达采样器（FHS）在掩码扩散模型中展现出可观的潜力。尽管这些采样器在实践中取得了成功，但对其理论理解仍不完善。许多现有分析基于Kullback-Leibler (KL) 散度进行，这通常导致参数依赖关系较为松散，并且需要对分数估计施加较强的假设。此外，这些理论保证未能覆盖最近开发的高性能采样器FHS。在这项工作中，我们首次为欧拉方法开发了直接基于总变差的 分析，克服了上述局限。我们的结果放松了对分数估计的假设要求。

    arXiv:2602.22505v2 Announce Type: replace  Abstract: Discrete diffusion models have achieved strong empirical performance in text and other symbolic domains, with masked (absorbing-rate) variants emerging as competitive alternatives to autoregressive models. Among existing samplers, the Euler method remains the standard choice in many applications, and more recently, the First-Hitting Sampler (FHS) has shown considerable promise for masked diffusion models. Despite their practical success, the theoretical understanding of these samplers remains incomplete. Many existing analyses are conducted in Kullback-Leibler (KL) divergence, which often yields loose parameter dependencies and requires strong assumptions on score estimation. Moreover, these guarantees do not cover recently developed high-performance sampler of FHS. In this work, we first develop a direct total-variation (TV)-based analysis for the Euler method that overcomes these limitations. Our results relax assumptions on score 
    
[^92]: 概念影响：利用可解释性提升训练数据归因的性能与效率

    Concept Influence: Leveraging Interpretability to Improve Performance and Efficiency in Training Data Attribution

    [https://arxiv.org/abs/2602.14869](https://arxiv.org/abs/2602.14869)

    该论文提出“概念影响”方法，将模型行为归因于语义方向而非单个测试样本，并证明简单的探针归因方法是其一阶近似，从而提升了训练数据归因的性能与效率。

    

    随着大型语言模型被越来越多地训练和微调，从业者需要方法来识别哪些训练数据驱动了特定的模型行为，尤其是非预期的行为。训练数据归因方法通过估计数据点的影响来应对这一问题。现有方法（如影响函数）不仅计算代价高昂，而且基于单个测试样本进行归因，这会使结果偏向句法相似性而非语义相似性。为了解决可扩展性以及面向抽象行为的归因问题，我们在归因过程中利用了模型内部的可解释结构。首先，我们提出了概念影响，它将模型行为归因于语义方向（如线性探针或稀疏自编码器特征），而不是单个测试样本。其次，我们证明了简单的基于探针的归因方法是概念影响的一阶近似，能够取得相当的性能。

    arXiv:2602.14869v2 Announce Type: replace  Abstract: As large language models are increasingly trained and fine-tuned, practitioners need methods to identify which training data drive specific behaviors, particularly unintended ones. Training Data Attribution (TDA) methods address this by estimating datapoint influence. Existing approaches like influence functions are both computationally expensive and attribute based on single test examples, which can bias results toward syntactic rather than semantic similarity. To address these issues of scalability and influence to abstract behavior, we leverage interpretable structures within the model during the attribution. First, we introduce Concept Influence which attribute model behavior to semantic directions (such as linear probes or sparse autoencoder features) rather than individual test examples. Second, we show that simple probe-based attribution methods are first-order approximations of Concept Influence that achieve comparable perfor
    
[^93]: 通过近端采样器实现约束采样与复合采样

    Constrained and Composite Sampling via Proximal Sampler

    [https://arxiv.org/abs/2602.14478](https://arxiv.org/abs/2602.14478)

    该论文提出基于近端采样器的框架，通过上图提升、割平面方法与拒绝采样，实现了凸集上约束采样及复合密度 $e^{-f-h}$ 的采样，且仅需分离、求值和次梯度预言机，摆脱了对投影和障碍函数的依赖。

    

    我们研究两类对数凹采样问题：约束采样和复合采样。首先，我们考虑在凸集 $K\subset\mathbb R^d$ 上从正比于 $e^{-f}$ 的分布中采样，其中 $f$ 是凸函数。通过上图提升技术，我们将该问题归约为在 $\mathbb R^{d+1}$ 中的凸集上带一维指数倾斜的均匀采样问题。随后，我们将近端采样器应用于提升后的分布。我们利用割平面方法和拒绝采样，开发了其受限高斯预言机的无偏实现，该实现仅需 $K$ 的分离预言机以及 $f$ 的求值和次梯度预言机，而无需依赖投影或障碍函数。其次，我们研究从 $e^{-f-h}$ 中进行复合采样，其中 $f$ 和 $h$ 是闭凸函数。这种结构在贝叶斯推断中自然出现，其中 $f$ 表示数据保真度，$h$ 编码先验信息。通过应用……（摘要原文在此处截断）

    arXiv:2602.14478v2 Announce Type: replace-cross  Abstract: We study two log-concave sampling problems: constrained sampling and composite sampling. First, we consider sampling from a distribution proportional to $e^{-f}$ on a convex set $K\subset\mathbb R^d$, where $f$ is convex. Using an epigraph lifting, we reduce this problem to uniform sampling up to a one-dimensional exponential tilt over a convex set in $\mathbb R^{d+1}$. We then apply the proximal sampler to the lifted distribution. We develop an unbiased implementation of its restricted Gaussian oracle by using the cutting-plane method and rejection sampling, requiring only a separation oracle for $K$ and evaluation and subgradient oracles for $f$, without relying on projections or barrier functions.   Second, we study composite sampling from $e^{-f-h}$, where $f$ and $h$ are closed and convex. This structure arises naturally in Bayesian inference, where $f$ represents data fidelity and $h$ encodes prior information. By applyin
    
[^94]: 基于模糊全变差距离的分布相似性的无分布认证

    Distribution-free certificate of similarity between distributions with blurred total variation distance

    [https://arxiv.org/abs/2602.05862](https://arxiv.org/abs/2602.05862)

    论文证明对连续分布而言TV距离的无分布置信上界必然平凡，从而提出用其松弛形式——模糊TV距离——在不做任何分布假设的情况下以有限样本置信度认证分布相似性。

    

    arXiv:2602.05862v3 公告类型：replace-cross 摘要：经典的双样本检验旨在根据从两个分布中抽取的样本来判断这两个分布是否相等。即使在无分布（distribution-free）的设定下，诸如置换检验等著名方法也能提供两个分布不同的统计证据。然而，未能拒绝“相等”这一假设并不一定意味着两个分布确实相似。在这项工作中，我们研究与之互补的问题：在不对待测分布施加任何假设的前提下，以有限样本置信度认证分布之间的相似性。然而，我们证明，对于连续分布而言，全变差（TV）距离的任何无分布置信上界必然以高概率是平凡的，即使两个分布完全相同也是如此。因此，在实践中不可能以无分布保证的方式认证TV距离意义下的相似性。为此，我们转而研究模糊TV距离（blurred TV distance）——TV距离的一种松弛形式，它使得……

    arXiv:2602.05862v3 Announce Type: replace-cross  Abstract: Classical two-sample tests aim to determine whether two distributions are equal or not based on samples from each one. Even in an distribution-free setting, well-known methods such as the permutation test can provide statistical evidence that the two distributions differ. However, failure to reject equality does not necessarily establish that they are similar. In this work, we study the complementary problem of certifying distributional similarity with finite-sample confidence, without imposing assumptions on the underlying distributions. We show, however, that for continuous distributions, any distribution-free upper confidence bound on the total variation (TV) distance must be trivial with high probability, even if the distributions are identical. Thus, certifying similarity in TV with a distribution-free guarantee is practically impossible. We therefore examine the blurred TV distance, a relaxation of TV distance that enable
    
[^95]: 学习层次化多指标模型中的最优标度律

    Optimal scaling laws in learning hierarchical multi-index models

    [https://arxiv.org/abs/2602.05846](https://arxiv.org/abs/2602.05846)

    该论文为两层神经网络学习层次化多指标目标建立了精确的信息论标度律理论，证明层次特征通过级联相变被依次恢复且简单的谱估计器即可达到最优速率，从而统一解释了浅层神经网络的标度律、平台现象与谱结构。

    

    在本工作中，我们为一类在层次化多指标目标上训练的两层神经网络，在真正受表示能力限制的情形下，提供了一个精确的标度律理论。我们推导出了子空间恢复和预测误差的精确信息论标度律，揭示了目标的层次化特征如何通过一系列相变的级联被依次学习。我们进一步证明，这些最优速率可以由一个简单的、与目标无关的谱估计器实现，该估计器可被解释为对第一层权重进行梯度下降时小学习率极限下的结果。一旦识别出合适的表示，读出层就可以通过高效的过程以统计最优的方式被学习。由此，我们为在此类层次化目标上训练的浅层神经网络中的标度律、平台现象和谱结构提供了一个统一而严格的解释。

    arXiv:2602.05846v2 Announce Type: replace-cross  Abstract: In this work, we provide a sharp theory of scaling laws for two-layer neural networks trained on a class of hierarchical multi-index targets, in a genuinely representation-limited regime. We derive exact information-theoretic scaling laws for subspace recovery and prediction error, revealing how the hierarchical features of the target are sequentially learned through a cascade of phase transitions. We further show that these optimal rates are achieved by a simple, target-agnostic spectral estimator, which can be interpreted as the small learning-rate limit of gradient descent on the first-layer weights. Once an adapted representation is identified, the readout can be learned statistically optimally, using an efficient procedure. As a consequence, we provide a unified and rigorous explanation of scaling laws, plateau phenomena, and spectral structure in shallow neural networks trained on such hierarchical targets.
    
[^96]: 多层交叉注意力在多模态上下文学习中被证明是最优的

    Multi-layer Cross-attention is Provably Optimal for Multi-modal In-context Learning

    [https://arxiv.org/abs/2602.04872](https://arxiv.org/abs/2602.04872)

    本文首次建立了多模态上下文学习的理论分析框架，证明单层自注意力无法在多模态任务上一致地达到贝叶斯最优，并提出一种新颖的线性化多层交叉注意力机制来可证明地恢复最优预测器。

    

    近期的进展迅速加深了我们对现代基于注意力的神经网络中上下文学习机制的理解。然而，现有结果仅局限于单模态数据；相比之下，多模态数据上下文学习的理论基础仍然知之甚少。我们引入了一个数学上易于处理的框架来研究多模态学习，并探索类Transformer架构何时能够在上下文中达到贝叶斯最优性能。为了对多模态问题进行建模，我们假设观测数据来自一个潜在因子模型。我们的第一个结果是对表达能力的一个负面结论：我们证明单层自注意力模型无法在任务分布上一致地恢复贝叶斯最优预测器。为了解决这一局限，我们引入了一种新颖的线性化交叉注意力机制，并在交叉注[摘要在此处截断]

    arXiv:2602.04872v5 Announce Type: replace-cross  Abstract: Recent progress has rapidly advanced our understanding of the mechanisms underlying in-context learning in modern attention-based neural networks. However, existing results focus exclusively on unimodal data; in contrast, the theoretical underpinnings of in-context learning for multi-modal data remain poorly understood. We introduce a mathematically tractable framework for studying multi-modal learning and explore when transformer-like architectures can recover Bayes-optimal performance in-context. To model multi-modal problems, we assume the observed data arises from a latent factor model. Our first result comprises a negative take on expressibility: we prove that single-layer self-attention models fail to recover the Bayes-optimal predictor uniformly over the task distribution. To address this limitation, we introduce a novel, linearized cross-attention mechanism, which we study in the regime where both the number of cross-at
    
[^97]: 概率测度的主成分分析：稀疏与稠密采样机制

    PCA of probability measures: Sparse and Dense sampling regimes

    [https://arxiv.org/abs/2602.02190](https://arxiv.org/abs/2602.02190)

    本文首次研究了基于 $n$ 个概率测度（每个通过 $m$ 个样本观测）的双重渐近机制下的PCA，推导出经验协方差算子与PCA超额风险 $n^{-1/2} + m^{-\alpha}$ 的收敛速度，刻画了从稀疏到稠密采样机制的转变，并证明稠密机制下的收敛速度是极小极大最优的。

    

    对概率测度进行主成分分析（PCA）的一种常用方法是将其嵌入到希尔伯特空间中，从而可以应用标准的函数型PCA技术。虽然基于 $m$ 个样本估计单个测度嵌入的收敛速度已被充分理解，但现有文献尚未涉及涉及多个测度的情形。本文研究了双重渐近机制下的PCA，即观测 $n$ 个概率测度，每个测度通过 $m$ 个样本进行观测。我们为经验协方差算子和PCA超额风险推导出了形如 $n^{-1/2} + m^{-\alpha}$ 的收敛速度，其中 $\alpha>0$ 取决于所选的嵌入方式。这刻画了测度数量 $n$ 与每个测度样本数量 $m$ 之间的关系，揭示了收敛行为中从稀疏（小 $m$）到稠密（大 $m$）机制的转变。此外，我们证明了稠密机制下的收敛速度对于经验协方差误差而言是极小极大最优的。

    arXiv:2602.02190v3 Announce Type: replace-cross  Abstract: A common approach to perform PCA on probability measures is to embed them into a Hilbert space where standard functional PCA techniques apply. While convergence rates for estimating the embedding of a single measure from $m$ samples are well understood, the literature has not addressed the setting involving multiple measures. In this paper, we study PCA in a double asymptotic regime where $n$ probability measures are observed, each through $m$ samples. We derive convergence rates of the form $n^{-1/2} + m^{-\alpha}$ for the empirical covariance operator and the PCA excess risk, where $\alpha>0$ depends on the chosen embedding. This characterizes the relationship between the number $n$ of measures and the number $m$ of samples per measure, revealing a sparse (small $m$) to dense (large $m$) transition in the convergence behavior. Moreover, we prove that the dense-regime rate is minimax optimal for the empirical covariance error.
    
[^98]: 用于随机动力系统概率代理建模的函数对函数非线性自回归模型（F2NARX）

    Function-on-function nonlinear autoregressive model (F2NARX) for probabilistic surrogate modeling of stochastic dynamical systems

    [https://arxiv.org/abs/2602.01929](https://arxiv.org/abs/2602.01929)

    本文提出 F2NARX 模型，将 F-NARX 重新表述为函数对函数回归问题，并结合主成分分析、高斯过程回归与无迹变换，在保持高精度的同时大幅提升预测效率，实现对随机动力系统响应的高效概率预测。

    

    在许多工程领域，构建准确且计算高效的代理模型来预测动力系统响应至关重要，但由于从外部激励和系统参数到系统响应的映射具有强非线性与高维特性，这一任务依然充满挑战。本工作提出了一种新颖的函数对函数非线性自回归模型（Function-on-Function Nonlinear AutoRegressive model with eXogenous inputs，简称 F2NARX），该模型从函数对函数回归的视角对近期提出的 F-NARX 方法进行了重新构建。所提出的框架在保持高精度的同时显著提升了预测效率。通过将主成分分析与高斯过程回归相结合，F2NARX 进一步能够以自回归的方式、借助无迹变换（unscented transform）实现对动力系统响应的概率预测。该方法的有效性通过不同复杂度的案例研究得到了验证。

    arXiv:2602.01929v3 Announce Type: replace-cross  Abstract: Constructing accurate and computationally efficient surrogate models for predicting dynamical system responses is critical in many engineering domains, yet remains challenging due to the strongly nonlinear and high-dimensional mapping from external excitations and system parameters to system responses. This work introduces a novel Function-on-Function Nonlinear AutoRegressive model with eXogenous inputs (F2NARX), which reformulates the recently proposed $\mathcal{F}$-NARX method from a function-on-function regression perspective. The proposed framework substantially improves predictive efficiency while maintaining high accuracy. By combining principal component analysis with Gaussian process regression, F2NARX further enables probabilistic predictions of dynamical responses via the unscented transform in an autoregressive manner. The effectiveness of the method is demonstrated through case studies of varying complexity. Results
    
[^99]: 从分层专家混合视角看门控注意力的统计理论

    A Statistical Theory of Gated Attention through the Lens of Hierarchical Mixture of Experts

    [https://arxiv.org/abs/2602.01468](https://arxiv.org/abs/2602.01468)

    该论文首次为门控注意力建立了统计理论，证明门控注意力矩阵的每个元素可写成分层专家混合的形式，并由此从理论上证明门控注意力比多头自注意力更具样本效率。

    

    自注意力机制通过使模型能够从具有长程依赖关系的数据中学习，为广泛使用的Transformer架构的成功做出了巨大贡献。为了提升性能，最近有人提出了一种在多头自注意力中引入门控机制的门控注意力模型，作为一种有前景的替代方案。门控注意力已在实证上被证明能够增强标准注意力中低秩映射的表达能力，甚至能够消除注意力汇聚现象。尽管效果显著，但文献中仍缺乏对门控注意力优势的清晰理论理解。为填补这一空白，我们严格地证明了门控注意力矩阵或多头自注意力矩阵中的每个元素都可以写成分层专家混合的形式。通过将学习重新表述为专家估计问题，我们证明了门控注意力比多头自注意力更具样本效率。

    arXiv:2602.01468v2 Announce Type: replace  Abstract: Self-attention has greatly contributed to the success of the widely used Transformer architecture by enabling learning from data with long-range dependencies. In an effort to improve performance, a gated attention model that leverages a gating mechanism within the multi-head self-attention has recently been proposed as a promising alternative. Gated attention has been empirically demonstrated to increase the expressiveness of low-rank mapping in standard attention and even to eliminate the attention sink phenomenon. Despite its efficacy, a clear theoretical understanding of gated attention's benefits remains lacking in the literature. To close this gap, we rigorously show that each entry in a gated attention matrix or a multi-head self-attention matrix can be written as a hierarchical mixture of experts. By recasting learning as an expert estimation problem, we demonstrate that gated attention is more sample-efficient than multi-head
    
[^100]: 多边际最优传输的统一Kantorovich对偶理论

    A Unified Kantorovich Duality for Multimarginal Optimal Transport

    [https://arxiv.org/abs/2601.17171](https://arxiv.org/abs/2601.17171)

    该论文建立了多边际最优传输的统一Kantorovich对偶理论，在紧与非紧（波兰空间）情形下均证明了对偶最优解的存在性，并刻画了最优对偶势的结构。

    

    我们研究具有有界连续成本函数的多边际最优传输（MOT）的Kantorovich对偶。主要关注点不仅在于原始问题与对偶问题取值的相等性，还在于最优对偶势的结构。对于紧度量空间，我们证明对偶问题在相互 $c$-共轭族类中存在最优解。证明结合了Fenchel--Rockafellar对偶论证、等度连续性估计、对偶势的规范化处理以及Arzelà--Ascoli定理。随后我们考虑非紧情形，即边际空间为波兰空间的情况。我们首先通过截断与紧性论证恢复了Kantorovich对偶恒等式，其中利用了成本函数的有界性和Prokhorov紧性。对于对偶最优解的存在性，我们依赖于最优支撑集的几何性质。在自然的支撑分裂条件下，我们在规范类中获得了一个有界的Borel可测对偶最优解。

    arXiv:2601.17171v2 Announce Type: replace-cross  Abstract: We study Kantorovich duality for multimarginal optimal transport (MOT) with bounded continuous cost functions. The main focus is not only the equality between the primal and dual values, but also the structure of optimal dual potentials. For compact metric spaces, we prove that the dual problem admits an optimizer in the class of mutually $c$-conjugate families. The proof combines a Fenchel--Rockafellar duality argument with equicontinuity estimates, a normalization of the dual potentials, and the Arzel\`a--Ascoli theorem. We then consider the non-compact case, where the marginal spaces are Polish. We first recover the Kantorovich duality identity by a truncation and tightness argument, using the boundedness of the cost and Prokhorov compactness. For dual attainment, we rely on the geometry of optimal supports. Under a natural support-splitting condition, we obtain a bounded Borel measurable dual optimizer in the canonical clas
    
[^101]: 迷失于聚合中：工具变量估计对象的因果解释

    Lost in Aggregation: The Causal Interpretation of the Instrumental Variable Estimand

    [https://arxiv.org/abs/2601.12120](https://arxiv.org/abs/2601.12120)

    本文揭示了一个在工具变量有效性论证中被普遍忽视的“聚合处理变量”问题，并证明在线性设定下聚合处理的因果效应通常是模糊的，因为它依赖于对聚合变量干预的具体实施方式。

    

    工具变量估计已成为社会科学和流行病学中减轻混杂偏差的标准方法，因为在这些领域开展随机实验的成本可能过于高昂或不可行。然而，证明工具变量的有效性往往具有挑战性。我们强调了在论证工具变量有效性时普遍被忽视的一个问题：“聚合处理变量”的存在，即处理变量（如教育、GDP、热量摄入）由更细粒度的、未被观测的成分组成，而每种成分可能对结果产生不同的影响。虽然聚合问题本身具有普遍性，但我们的重点放在线性设定下的工具变量估计上，这是大多数应用工具变量实践的基础。我们证明，聚合处理的因果效应通常是模糊的，因为它取决于对聚合变量的干预是如何被具体实施的。

    arXiv:2601.12120v3 Announce Type: replace-cross  Abstract: Instrumental variable estimation has emerged as a standard approach to mitigating confounding bias in the social sciences and epidemiology, where conducting randomized experiments can be too costly or infeasible. However, justifying the validity of the instrument is frequently challenging. We highlight a problem generally neglected in arguments for instrumental variable validity: the presence of an "aggregate treatment variable", where the treatment (e.g., education, GDP, caloric intake) is composed of finer-grained, unobserved components that each may have a different effect on the outcome. While the aggregation problem itself is general, our focus is on instrumental variable estimation in a linear setting, the regime underlying much of applied instrumental variable practice. We show that the causal effect of an aggregate treatment is generally ambiguous, as it depends on how an intervention on the aggregate is instantiated at
    
[^102]: 用于对流扩散方程中源项与参数联合估计的物理信息神经网络

    Physics-Informed Neural Networks for Joint Source and Parameter Estimation in Advection-Diffusion Equations

    [https://arxiv.org/abs/2512.07755](https://arxiv.org/abs/2512.07755)

    提出了一种基于多网络神经正切核的加权自适应PINN方法，以对流扩散方程为耦合约束，实现对源函数与PDE参数的联合估计，有效缓解稀疏测量下源反演的严重病态性问题。

    

    在稀疏测量条件下，抛物型偏微分方程（PDE）的源反演问题由于严重的病态性以及所涉及的多个未知量（包括源函数和PDE参数），使用物理信息神经网络（PINNs）求解尤为困难。尽管PINNs的神经正切核（NTK）已广泛应用于仅涉及单个神经网络的正问题，但其在涉及多个神经网络的逆问题中的扩展仍较少被探索。在本工作中，我们提出了一种基于PINNs神经正切核的加权自适应方法，其中包含表示解、未知源和PDE参数的多个神经网络。该方法的核心思想是同时求解这些未知量的联合恢复，将底层PDE作为耦合多个未知泛函参数的约束条件，从而实现更高效的参数利用。

    arXiv:2512.07755v4 Announce Type: replace-cross  Abstract: Source inversion problems under sparse measurements for parabolic partial differential equations (PDEs) are particularly challenging to solve using physics-informed neural networks (PINNs), due to their severe ill-posedness and the multiple unknowns involved including the source function and the PDE parameters. Although the neural tangent kernel (NTK) of PINNs has been widely used in forward problems involving a single neural network, its extension to inverse problems involving multiple neural networks remains less explored. In this work, we propose a weighted adaptive approach based on the NTK of PINNs including multiple networks representing the solution, the unknown source, and the PDE parameters. The key idea behind our methodology is to simultaneously solve the joint recovery of these unknowns thereby using the underlying PDE as a constraint that couples multiple unknowns functional parameters, leading to more efficient us
    
[^103]: 等重复设计的不完全U统计量：Berry–Esseen界与高效构造

    Incomplete U-Statistics of Equireplicate Designs: Berry-Esseen Bound and Efficient Construction

    [https://arxiv.org/abs/2510.20755](https://arxiv.org/abs/2510.20755)

    本文通过将不完全U统计量的依赖图视为其设计超图的线图，利用Stein方法首次在完全退化情形下为确定性设计建立了Berry–Esseen界，并推广至学生化统计量，从而在不依赖重抽样的前提下实现了高效构造与高斯极限推断。

    

    U统计量是一类基本的估计量，它推广了样本均值的概念，并支撑着非参数统计的大部分内容。尽管在统计学和概率论中已被广泛研究，但其仍面临关键挑战：高昂的计算成本——不完全U统计量部分地解决了这一问题——以及其在退化情形下非标准的渐近行为，这通常需要借助重抽样方法来进行假设检验。本文基于超图理论和组合设计，为U统计量提供了一个全新的视角。我们将不完全U统计量的依赖图与设计超图的线图等同起来，并利用Stein方法在完全退化情形下为确定性设计推导出Berry–Esseen界。我们进一步将这些结果推广到学生化统计量，使用了一个专门针对完全退化情形设计的方差估计量。所得到的界为……建立了高斯极限（摘要原文在此处截断）。

    arXiv:2510.20755v5 Announce Type: replace-cross  Abstract: U-statistics are a fundamental class of estimators that generalize the sample mean and underpin much of nonparametric statistics. Although extensively studied in both statistics and probability, key challenges remain: their high computational cost - addressed partly through incomplete U-statistics - and their non-standard asymptotic behavior in the degenerate case, which typically requires resampling methods for hypothesis testing. This paper presents a novel perspective on U-statistics, grounded in hypergraph theory and combinatorial designs. We identify the dependency graph of an incomplete U-statistic with the line graph of its design hypergraph and use Stein's method to derive Berry-Esseen bounds for deterministic designs under full degeneracy. We further extend these results to studentized statistics using a variance estimator tailored to the fully degenerate setting. The resulting bounds establish Gaussian limits for supe
    
[^104]: 用于可靠统计推断的预测增强树

    Prediction-Augmented Trees for Reliable Statistical Inference

    [https://arxiv.org/abs/2510.16937](https://arxiv.org/abs/2510.16937)

    该论文提出了预测增强残差树（PART）估计器，通过利用额外的计算从固定的标注数据、未标注数据和预训练预测器中挖掘更多推断价值，从而构建更精确的置信区间，提升统计推断的可靠性。

    

    机器学习（ML）在科学发现中正扮演着越来越核心的角色，但要将机器学习的预测转化为严谨的科学结论，仍然需要基于金标准标注数据进行统计推断。科学领域中已出现一种常见的研究设置：分析者拥有一个较小的标注样本、一个大得多的未标注样本以及一个预训练的预测器，目标是针对感兴趣的量构建置信区间，这一定义方式由 Angelopoulos 等人（2023）正式提出。在这一机器学习驱动的科学发现文献中，标注数据通常是精度的约束瓶颈，而计算资源相对不受限制，这引出了一个设计问题：如何利用额外的计算，从固定的标注数据集、固定的未标注数据集和固定的预测器中提取更多的推断价值。为回答这一问题，我们提出了预测增强残差树（PART）估计器。PART 接受相同的输入……（原文摘要在此处截断）

    arXiv:2510.16937v2 Announce Type: replace-cross  Abstract: Machine learning (ML) is increasingly central to scientific discovery, but translating ML predictions into rigorous scientific claims still requires statistical inference on gold-standard labeled data. A setting that has emerged across the sciences provides the analyst with a small labeled sample, a much larger unlabeled sample, and a pre-trained predictor, with the goal of constructing a confidence interval for a quantity of interest, as was formalized in Angelopoulos et al. (2023). In this ML-driven scientific discovery literature, labeled data is typically the binding constraint on precision while computation is comparatively unconstrained, motivating the design question of how additional computation can extract more inferential value from a fixed labeled set, a fixed unlabeled set, and a fixed predictor.   To address this question, we propose the Prediction-Augmented Residual Tree (PART) estimator. PART takes the same input
    
[^105]: 基于随机最优控制的生成模型中密度估计精度研究

    On accuracy of density estimation in generative models based on stochastic optimal control

    [https://arxiv.org/abs/2506.03043](https://arxiv.org/abs/2506.03043)

    本文为基于薛定谔桥的生成模型建立了密度估计的非渐近统计保证，证明在光滑性假设下精确经验风险最小化可达到 $\widetilde{O}(n^{-2s/(2s+d)})$ 的KL收敛速率，达到经典非参数极小极大最优率（至多相差对数因子）。

    

    我们研究了基于Ornstein-Uhlenbeck参考过程的薛定谔桥模型的统计精度。给定一个已知的紧支撑初始分布以及来自未知目标分布的独立样本，我们通过在指定函数类上最小化经验Kullback-Leibler（KL）风险来估计右薛定谔对数势。对于有界的亚高斯目标密度（其支撑可以是无界的），我们为目标分布与估计的终端边缘分布之间的KL散度建立了非渐近的高概率oracle不等式。在目标对数密度满足额外的双侧高斯界和光滑性假设下，我们推导了稀疏ReLU网络的逼近保证，并为精确经验风险最小化建立了 $\widetilde{O}(n^{-2s/(2s+d)})$ 的KL收敛速率。该速率在对数因子范围内与经典非参数KL风险收敛速率相匹配。

    arXiv:2506.03043v2 Announce Type: replace  Abstract: We study the statistical accuracy of Schr\"odinger bridge models based on an Ornstein-Uhlenbeck reference process. Given a known, compactly supported initial distribution and independent samples from an unknown target distribution, we estimate a right Schr\"odinger log-potential by minimizing empirical Kullback-Leibler (KL) risk over a prescribed function class. For bounded sub-Gaussian target densities, potentially with unbounded support, we establish a non-asymptotic high-probability oracle inequality for the KL-divergence between the target and estimated terminal marginals. Under additional two-sided Gaussian bounds and smoothness assumptions on the target log-density, we derive approximation guarantees for sparse ReLU networks and establish a KL convergence rate of $\widetilde{O}(n^{-2s / (2s + d)})$ for exact empirical risk minimization. This matches, up to logarithmic factors, the classical nonparametric KL-risk convergence exp
    
[^106]: 通过局部化初始化实现固定维度下两层ReLU网络的良性过拟合

    Benign Overfitting of Two-Layer ReLU Networks in Fixed Dimension through Localized Initialization

    [https://arxiv.org/abs/2505.11621](https://arxiv.org/abs/2505.11621)

    该论文通过一种结合高斯权重与仅在小球形帽上激活的局部化神经元的初始化方法，首次在固定维度下证明了有限宽度两层ReLU网络经梯度流训练后可实现对含噪数据的良性过拟合，并给出了样本量、宽度与训练时间的明确界限。

    

    良性过拟合是指模型能够完全拟合含噪训练数据，同时仍保持良好的泛化能力。我们在固定维度下，针对采用标准ReLU激活函数、通过梯度流训练的有限宽度两层网络确立了这一现象。我们提出的数据无关初始化方法将高斯权重与初始仅在收缩球形帽上激活的神经元相结合。对于均匀球面输入和任意固定的有界条件响应规律，训练风险和超额风险在所有足够晚的时间上均一致地依概率收敛于零。参数也以概率趋于一的方式收敛到一个插值极限，该极限的超额风险依概率趋于零。我们使用仅依赖于样本量和维度的单一调度方案。我们的证明将网络与一个核流联系起来，该核流的极限将岭回归与拟合训练残差的局部化尖峰相结合。我们给出了明确的样本量、宽度和训练时间界限。

    arXiv:2505.11621v2 Announce Type: replace-cross  Abstract: Benign overfitting occurs when a model fits noisy training data exactly and still generalizes well. We establish this behavior for finite-width two-layer networks with the standard ReLU activation, trained by gradient flow in fixed dimension. Our data-independent initialization combines Gaussian weights with neurons initially active only on shrinking spherical caps. For uniform spherical inputs and any fixed bounded conditional response law, training and excess risks vanish in probability uniformly over all sufficiently late times. The parameters also converge, with probability tending to one, to an interpolating limit whose excess risk vanishes in probability. We use one schedule depending only on sample size and dimension. Our proof connects the network to a kernel flow whose limit combines ridge regression with localized spikes that fit the training residuals. We give explicit sample-size, width and training-time bounds.
    
[^107]: 森林用于差异：超越参数化双重差分方法

    Forests for Differences: Beyond Parametric Difference-in-Differences

    [https://arxiv.org/abs/2505.09706](https://arxiv.org/abs/2505.09706)

    本文提出双重差分贝叶斯因果森林（DiD-BCF）这一非参数模型，统一估计ATE、GATE和CATE，并将条件平行趋势与无预期假设直接嵌入贝叶斯非参数框架，在非线性、选择偏差和效应异质性场景下显著优于现有基准方法。

    

    本文提出了双重差分贝叶斯因果森林（DiD-BCF），这是一种新颖的非参数模型，旨在解决双重差分估计中的关键挑战，例如错峰采用和异质性处理效应。DiD-BCF提供了一个统一框架，用于估计平均处理效应（ATE）、组平均处理效应（GATE）和条件平均处理效应（CATE）。通过为特定队列的水平、共同的未处理趋势和处理效应分别构建回归森林，该模型将条件平行趋势和无预期限制直接嵌入到贝叶斯非参数方法中。我们建立了精确处理效应后验分布的非参数识别性和综合处理效应一致性。大量仿真实验表明，DiD-BCF相比现有基准方法具有更优越的性能，尤其是在存在非线性、选择偏差和效应异质性的情况下。该方法被应用于美国最低工资政策（评估）……（原文摘要在此截断）

    arXiv:2505.09706v3 Announce Type: replace-cross  Abstract: This paper introduces the Difference-in-Differences Bayesian Causal Forest (DiD-BCF), a novel non-parametric model addressing key challenges in DiD estimation, such as staggered adoption and heterogeneous treatment effects. DiD-BCF provides a unified framework for estimating Average (ATE), Group-Average (GATE), and Conditional Average Treatment Effects (CATE). By structuring separate regression forests for cohort-specific levels, common untreated trends and treatment effects, the model directly embeds conditional parallel trends and no-anticipation restrictions into Bayesian nonparametrics. We establish non-parametric identification and integrated treatment effect consistency of the exact treatment effect posterior distribution. Extensive simulations demonstrate DiD-BCF's superior performance over established benchmarks, particularly under non-linearity, selection biases, and effect heterogeneity. Applied to U.S. minimum wage p
    
[^108]: 校正符合性分数以获得更好的条件覆盖

    Rectifying Conformity Scores for Better Conditional Coverage

    [https://arxiv.org/abs/2502.16336](https://arxiv.org/abs/2502.16336)

    提出一种通过对符合性分数进行可训练变换（基于条件分位数估计）的新方法，在保证精确边际覆盖的同时显著改善条件覆盖，尤其适用于多输出问题中自适应置信集的构建。

    

    我们提出了一种在分裂符合性预测框架内生成置信集的新方法。该方法对任意给定的符合性分数执行可训练的变换，以改善条件覆盖，同时确保精确的边际覆盖。该变换基于对符合性分数条件分位数的估计。所得方法在构建多输出问题中的自适应置信集时尤为有益，而标准的符合性分位数回归方法在这类问题中适用性有限。我们推导出一个理论界，刻画了分位数估计精度对近似条件有效性的影响，这与仅提供边际覆盖的经典符合性预测理论界不同。实验表明，我们的方法对局部数据结构具有高度自适应性，在条件覆盖方面优于现有方法，并带来了提升。

    arXiv:2502.16336v3 Announce Type: replace-cross  Abstract: We present a new method for generating confidence sets within the split conformal prediction framework. Our method performs a trainable transformation of any given conformity score to improve conditional coverage while ensuring exact marginal coverage. The transformation is based on an estimate of the conditional quantile of conformity scores. The resulting method is particularly beneficial for constructing adaptive confidence sets in multi-output problems where standard conformal quantile regression approaches have limited applicability. We develop a theoretical bound that captures the influence of the accuracy of the quantile estimate on the approximate conditional validity, unlike classical bounds for conformal prediction methods that only offer marginal coverage. We experimentally show that our method is highly adaptive to the local data structure and outperforms existing methods in terms of conditional coverage, improving 
    
[^109]: 在线组合优化中统计推断的收入成本

    The Revenue Cost of Inference in Online Assortment Optimization

    [https://arxiv.org/abs/2501.19277](https://arxiv.org/abs/2501.19277)

    该论文刻画了在线组合实验中收入损失与估计方差之间的理论最优权衡前沿，证明当收入最优组合在所有目标梯度方向均提供信息时零成本学习可行，并提出保留参考报价的试运行设计使实际方差准则收敛到该oracle最优值。

    

    运行组合（选品）实验的卖家必须在赚取收入的同时收集数据，以便比较未来的报价方案。对于一个正则的有限维参数化选择模型和一个有限的组合族，我们刻画了预期收入损失与预设收入比较下最大局部一阶方差常数之间的“oracle”（理论最优）前沿。当且仅当收入最大化的组合在每个目标梯度方向上都能提供信息时，该oracle基准才能在零成本下保持有限。当成本高昂的报价能够补足缺失的信息时，我们推导出逆预算方差定律的精确首项系数以及极限意义下的最优支出份额。我们提出一种基于试运行的程序，通过保留参考报价来维持信息量，并收紧估计的预算约束以考虑不确定性。我们给出了使所实施设计的方差准则收敛到稳定化oracle值的条件……

    arXiv:2501.19277v5 Announce Type: replace-cross  Abstract: A seller running an assortment experiment must earn revenue while collecting data to compare future offers. For a regular finite-dimensional parametric choice model and a finite assortment family, we characterize the oracle frontier between expected revenue loss and the largest local first-order variance constant across prespecified revenue comparisons. The oracle benchmark remains finite at zero cost exactly when revenue-maximizing assortments provide information in every target-gradient direction. When costly offers can supply the missing information, we derive the exact leading coefficient of the inverse-budget variance law and the limiting optimal spending shares. A pilot-based procedure reserves reference offers to preserve information and tightens the estimated budget constraint to account for uncertainty. We give conditions for the implemented design's variance criterion to converge to the stabilized oracle value at a fi
    
[^110]: 用于非凸和随机光滑优化的投影梯度方法：新的复杂度界与自适应条件步长

    Projected gradient methods for nonconvex and stochastic smooth optimization: new complexities and auto-conditioned stepsizes

    [https://arxiv.org/abs/2412.14291](https://arxiv.org/abs/2412.14291)

    该论文提出了一类自适应条件的投影梯度方法，无需预知梯度Lipschitz常数或线搜索即可在凸与非凸优化问题上实现统一的迭代复杂度，并进一步推广至随机优化设置。

    

    我们提出了一类新颖的投影梯度（PG）方法，用于在凸紧集上最小化光滑但不一定凸的函数。我们首先对常数步长PG方法提供了新颖的分析，为凸和非凸问题寻找近似驻点实现了统一的迭代复杂度。随后，我们开发了一种“自适应条件”投影梯度（AC-PG）变体，它在无需输入梯度Lipschitz常数或任何线搜索程序的情况下即可达到相同的迭代复杂度。其关键思想是利用从先前迭代中收集的一阶信息来估计Lipschitz常数，并证明低估Lipschitz常数所引起的误差可以得到适当控制。接着，我们将PG方法推广到随机设置，提出了随机投影梯度（SPG）方法和方差缩减随机梯度方法（摘要在此处截断）。

    arXiv:2412.14291v3 Announce Type: replace-cross  Abstract: We present a novel class of projected gradient (PG) methods for minimizing a smooth but not necessarily convex function over a convex compact set. We first provide a novel analysis of the constant-stepsize PG method, achieving a unified iteration complexity for finding an approximate stationary point for convex and nonconvex problems. We then develop an "auto-conditioned" projected gradient (AC-PG) variant that achieves the same iteration complexity without requiring the input of the Lipschitz constant of the gradient or any line search procedure. The key idea is to estimate the Lipschitz constant using first-order information gathered from the previous iterations, and to show that the error caused by underestimating the Lipschitz constant can be properly controlled. We then generalize the PG methods to the stochastic setting, by proposing a stochastic projected gradient (SPG) method and a variance-reduced stochastic gradient (
    
[^111]: 分布偏移下高维分位数回归的迁移学习

    Transfer Learning for High-dimensional Quantile Regression with Distribution Shift

    [https://arxiv.org/abs/2411.19933](https://arxiv.org/abs/2411.19933)

    该论文针对高维分位数回归，提出了一种新的可迁移集合与迁移学习框架，能够同时应对参数偏移、协变量偏移和残差偏移三种分布偏移问题，并通过非渐近误差界、源检测一致性理论以及正交去偏推断方法证明了其有效性和优越性。

    

    来自相关源研究的信息通常能够增强目标研究的发现。然而，目标研究与源研究之间的分布偏移会严重影响知识迁移的效率。在高维回归的设定下，现有的迁移方法主要关注参数偏移。本文关注知识迁移下的高维分位数回归在三种类型分布偏移中的表现：参数偏移、协变量偏移和残差偏移。我们提出了一种新颖的可迁移集合和一种新的迁移框架来应对上述三种差异。我们建立了非渐近估计误差界和源检测一致性，以验证我们的方法在存在分布偏移时的可用性与优越性。此外，我们还提出了一种正交去偏方法用于带知识迁移的统计推断，从而获得更精细的渐近性质。

    arXiv:2411.19933v3 Announce Type: replace-cross  Abstract: Information from related source studies can often enhance the findings of a target study. However, the distribution shift between target and source studies can severely impact the efficiency of knowledge transfer. In the high-dimensional regression setting, existing transfer approaches mainly focus on the parameter shift. In this paper, we focus on the high-dimensional quantile regression with knowledge transfer under three types of distribution shift: parameter shift, covariate shift, and residual shift. We propose a novel transferable set and a new transfer framework to address the above three discrepancies. Non-asymptotic estimation error bounds and source detection consistency are established to validate the availability and superiority of our method in the presence of distribution shift. Additionally, an orthogonal debiased approach is proposed for statistical inference with knowledge transfer, leading to sharper asymptoti
    
[^112]: RECOVAR：基于深度潜在空间表示协方差的地震事件检测方法

    RECOVAR: Representation Covariances on Deep Latent Spaces for Seismic Event Detection

    [https://arxiv.org/abs/2407.18402](https://arxiv.org/abs/2407.18402)

    提出了RECOVAR，一种无需事件/噪声标签、直接从原始波形训练的无监督地震事件检测方法，通过卷积自编码器集成与互协方差度量实现事件检测，性能可比监督检测器并在跨数据集和不平衡数据场景下保持稳健。

    

    现代深度学习方法提升了地震检测的效果，但目前应用最广泛的方法通常依赖于从大量带标注的波形数据集中进行监督学习。构建此类数据集既耗时，又会引入来自现有地震目录、检测流程和人工审核的偏差，这可能限制跨数据集的泛化能力以及对代表性不足事件的敏感性。我们提出了RECOVAR，这是一种无监督的地震事件检测方法，可直接从原始波形进行训练，无需使用事件/噪声标签。该方法采用卷积自编码器集成，并通过基于互协方差的度量来衡量其压缩潜在表示中的时间对齐性，从而检测事件。结果表明，RECOVAR在公开基准数据集上可达到监督检测器的性能水平，在跨数据集场景下保持稳健，并且在类别不平衡的连续波形数据上依然有效。

    arXiv:2407.18402v3 Announce Type: replace  Abstract: Modern deep learning methods have improved earthquake detection, but the most widely used approaches typically rely on supervised learning from large labeled waveform datasets. Constructing such datasets is time-consuming and introduces biases from existing catalogs, detection pipelines, and human review, which can limit cross-dataset generalization and sensitivity to underrepresented events. We present RECOVAR, an unsupervised method for seismic event detection that can be trained directly from raw waveforms without using event/noise labels. The method uses an ensemble of convolutional autoencoders and detects events by measuring temporal alignment in their compressed latent representations through a cross-covariance-based metric. The results suggest that RECOVAR can reach the performance of supervised detectors on public benchmark datasets, remain robust under cross-dataset scenarios, and stay effective on imbalanced continuous wav
    
[^113]: 连续正规化流在学习概率分布中的收敛性

    Convergence of Continuous Normalizing Flows for Learning Probability Distributions

    [https://arxiv.org/abs/2404.00551](https://arxiv.org/abs/2404.00551)

    本文研究了具有线性插值的CNFs在从有限随机样本中学习概率分布时的理论性质，建立了非渐近误差界，并提供了收敛分析框架。

    

    连续正规化流（CNFs）是一种基于常微分方程的学习概率分布的生成方法。这种方法在各种应用中表现出显著的经验成功，包括大规模图像合成、蛋白质结构预测和分子生成。本文研究了具有线性插值的CNFs在从有限随机样本中学习概率分布时的理论性质，使用了流匹配目标函数。我们建立了基于CNFs的分布估计器的非渐近误差界，以Wasserstein-2距离表示。我们分析的关键假设是目标分布满足以下三个条件之一：要么具有有界支持，要么是强对数凹的，要么是有限或无限混合的高斯分布。我们提出了一个包含了误差收敛分析框架

    arXiv:2404.00551v1 Announce Type: cross  Abstract: Continuous normalizing flows (CNFs) are a generative method for learning probability distributions, which is based on ordinary differential equations. This method has shown remarkable empirical success across various applications, including large-scale image synthesis, protein structure prediction, and molecule generation. In this work, we study the theoretical properties of CNFs with linear interpolation in learning probability distributions from a finite random sample, using a flow matching objective function. We establish non-asymptotic error bounds for the distribution estimator based on CNFs, in terms of the Wasserstein-2 distance. The key assumption in our analysis is that the target distribution satisfies one of the following three conditions: it either has a bounded support, is strongly log-concave, or is a finite or infinite mixture of Gaussian distributions. We present a convergence analysis framework that encompasses the err
    
[^114]: 结构化支持向量机的接近线性时间算法

    A Nearly-Linear Time Algorithm for Structured Support Vector Machines. (arXiv:2307.07735v1 [math.OC])

    [http://arxiv.org/abs/2307.07735](http://arxiv.org/abs/2307.07735)

    这篇论文提出了针对结构化支持向量机的接近线性时间算法，解决了二次规划输入规模和解决时间的问题。

    

    二次规划是凸优化领域中的基本问题。许多实际任务可以表示为二次规划，例如支持向量机（SVM）。在深度学习方法盛行之前，线性SVM是过去三十年来最流行的机器学习工具之一。一般来说，一个二次规划的输入规模为Θ(n^2)（其中n是变量的数量），因此解决该问题需要Ω(n^2)的时间。然而，SVM产生的二次规划的输入规模为O(n)，这使得设计接近线性时间算法成为可能。两个重要的SVM类别是具有低秩核因式分解和低树宽规模的程序。低树宽凸优化在过去几年中引起了越来越多的关注（例如线性规划[Dong, Lee and Ye 2021]和半定规划[Gu and Song 2022]）。因此，一个重要的开放问题是是否存在接近线性时间算法。

    Quadratic programming is a fundamental problem in the field of convex optimization. Many practical tasks can be formulated as quadratic programming, for example, the support vector machine (SVM). Linear SVM is one of the most popular tools over the last three decades in machine learning before deep learning method dominating.  In general, a quadratic program has input size $\Theta(n^2)$ (where $n$ is the number of variables), thus takes $\Omega(n^2)$ time to solve. Nevertheless, quadratic programs coming from SVMs has input size $O(n)$, allowing the possibility of designing nearly-linear time algorithms. Two important classes of SVMs are programs admitting low-rank kernel factorizations and low-treewidth programs. Low-treewidth convex optimization has gained increasing interest in the past few years (e.g.~linear programming [Dong, Lee and Ye 2021] and semidefinite programming [Gu and Song 2022]). Therefore, an important open question is whether there exist nearly-linear time algorithms
    
[^115]: 调查数据的惩罚似然推断方法

    Penalized Likelihood Inference with Survey Data. (arXiv:2304.07855v1 [econ.EM])

    [http://arxiv.org/abs/2304.07855](http://arxiv.org/abs/2304.07855)

    本文将三种Lasso推断方法应用于调查数据的推断，证明在具有调查权重和/或异方差的广义线性模型中的推断程序渐近有效性， 并将这些方法推广到对非线性参数函数进行推断，例如在调查logit模型中的平均边际效应

    

    本文将三种Lasso推断方法，Debiased Lasso、$C(\alpha)$和Selective Inference推广到调查数据的环境中。我们证明了在具有调查权重和/或异方差的广义线性模型中推断程序的渐近有效性。此外，我们将这些方法推广到对非线性参数函数进行推断，例如在调查logit模型中的平均边际效应。我们通过模拟数据和加拿大2020年互联网使用调查数据来说明该方法的有效性。

    This paper extends three Lasso inferential methods, Debiased Lasso, $C(\alpha)$ and Selective Inference to a survey environment. We establish the asymptotic validity of the inference procedures in generalized linear models with survey weights and/or heteroskedasticity. Moreover, we generalize the methods to inference on nonlinear parameter functions e.g. the average marginal effect in survey logit models. We illustrate the effectiveness of the approach in simulated data and Canadian Internet Use Survey 2020 data.
    

