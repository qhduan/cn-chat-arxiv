# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [First-Order Stationarity of Reverse Diffusions](https://arxiv.org/abs/2609.31612) | 该论文为扩散模型建立了首个一阶平稳性理论，证明基于SDE的过阻尼与欠阻尼朗之万扩散的逆时流在前向过程平稳势强凸（仅针对加噪过程而非数据）时以指数速率收缩相对Fisher散度，并为离散化采样器建立了与非凸优化中平均梯度范数保证相对应的一阶平稳性界。 |
| [^2] | [Uncertainty and Explainability in Deep Rough Volatility: A Neural Information-Theoretic Posterior Approach](https://arxiv.org/abs/2609.31570) | 该论文提出一个基于仿真的推断框架，通过神经比率估计学习粗糙Heston模型参数在隐含波动率曲面条件下的后验分布，结合异方差神经代理定价器生成考虑不确定性的价格区间，并引入信息论可解释性方法Hellinger-SHAP。 |
| [^3] | [Beyond Empirical Support: Structured Outlier Generation via Sinkhorn Optimal Transport](https://arxiv.org/abs/2609.31470) | 该论文提出SBOG框架，将Sinkhorn最优传输几何与分布鲁棒边界建模相结合，在潜空间中结构化地生成弱支撑边界区域的离群点，从而更有效地评估和提升机器学习系统应对分布偏移的鲁棒性。 |
| [^4] | [Nonparametric In-Context Learning under Growing Geometric Complexity: Minimax Optimality and Local Geometry-Adaptivity of Transformers](https://arxiv.org/abs/2609.31458) | 本文在由随样本量增长、维度与光滑度异构的流形混合所刻画的未知局部几何下研究非参数上下文学习，建立了极小极大最优下界，并证明配备几何预条件器的结构感知两阶段softmax Transformer能达到该最优性且可自适应局部几何。 |
| [^5] | [Multivariate conformal uncertainty propagation in multitask atomistic simulation: Successes and pitfalls](https://arxiv.org/abs/2609.31384) | 本文将多变量共形校准方法应用于多任务原子模拟的多阶段工作流，通过向下游目标量传播不确定性集合以捕捉误差抵消效应，并总结了该方法取得的成功与面临的陷阱。 |
| [^6] | [Equation discovery with Bayesian tree-adjoining grammars](https://arxiv.org/abs/2609.31368) | 本文首次将树邻接文法置于贝叶斯框架下，利用结构保持树移动的可逆跳MCMC采样器推断模型结构、参数和预测的联合后验分布，实现了超越传统点估计的概率化方程发现与非线性系统辨识。 |
| [^7] | [Brenier Meets Adversarial Training: Optimal Transport Geometry for Robust Learning](https://arxiv.org/abs/2609.31363) | 该论文将带 Wasserstein 惩罚的分布鲁棒优化中的对抗问题重新表述为最优传输映射的优化问题，证明了最优映射满足循环单调性，指出标准对抗训练因违反该性质而浪费传输成本，并提出多起点粒子上升等方法予以改进。 |
| [^8] | [LUCID: Learning Under Confounding for Inference and Discovery in Time Series](https://arxiv.org/abs/2609.31315) | LUCID 提出了一种机制自适应的去混杂层，利用 Marcenko–Pastur 谱路由器估计混杂机制并施加相应的去混杂策略，可封装现有因果发现算法，从而在时间序列中有效消除未观测混杂因子造成的虚假关联。 |
| [^9] | [Geometric Moment Contraction for Stochastic Nesterov Acceleration](https://arxiv.org/abs/2609.31303) | 该论文为常参数随机Nesterov加速算法建立了几何矩收缩理论，给出了仅要求梯度具有有限 $p$ 阶矩的显式步长收缩判据，即使梯度具有无穷方差（$1<p<2$）也能保证 $L^p$ 收敛。 |
| [^10] | [DIAL: Position-Debiased LLM Judges with Adaptive Human Preference Calibration](https://arxiv.org/abs/2609.31215) | 提出DIAL统一框架，利用大量LLM比较结合少量人类比较，分离并消除LLM评判器中的位置偏差，并将去偏后的偏好结构自适应校准至人类偏好目标，同时提供可识别性理论与不确定性量化保证。 |
| [^11] | [A Flatness-Generalization Relation in the Teacher-Student Tree-Committee Machine](https://arxiv.org/abs/2609.31101) | 该论文在师生树委员会机器中通过零温度吉布斯形式与Edwards-Jones形式解析刻画了经验损失典型极小值点的可观测量和Hessian谱，进而检验泛化误差随数据量增大而下降是否对应于损失景观平坦性的提升。 |
| [^12] | [SAGE: A sampling-aware global evaluation benchmark for species distribution modeling](https://arxiv.org/abs/2609.31082) | 该论文提出了SAGE——一个采样感知的全局评估基准，通过结合GBIF训练记录与sPlotOpen植被调查数据，充分考虑采样偏差和物种层面（尤其是稀有物种）的性能差异，为深度学习多物种分布模型提供更可靠、更具信息量的评估。 |
| [^13] | [Ensembles of Exactly Solved Subsamples for Clusterwise Regression: Trimming Without a Trimming Level](https://arxiv.org/abs/2609.31019) | 该论文提出一种基于精确求解随机子样本的聚类回归集成方法，可自动估计截尾水平而无需预先设定，在响应变量含高达20%粗大离群值时实现0.89的最坏情况准确率，优于传统的截尾交替法。 |
| [^14] | [Coupled Usage-Sense Processes: Temporal and Attributable Lexical Semantic Change](https://arxiv.org/abs/2609.30974) | 本文提出耦合用法-语义过程（CUSP）框架，通过单一的边缘保持时序过程，不仅能量化词汇语义变化的幅度，还能精确确定变化发生的时间、将变化分解为成分移动与重组的机制，并归因到具体的词语用法。 |
| [^15] | [Conformal Prediction under Exponential-Tilt Joint Shift](https://arxiv.org/abs/2609.30886) | 该论文研究了在输入分布及其与结果关系同时发生联合偏移时，比较了在保形预测校准中直接使用ExTRA估计的指数倾斜权重与额外对源预测分布进行倾斜这两种适应策略的覆盖性能。 |
| [^16] | [TR-SSQP: A Trust-Region Method for Constrained Stochastic Optimization under Heavy-Tailed Noise](https://arxiv.org/abs/2609.30732) | 本文提出TR-SSQP方法，在随机序列二次规划框架下通过法向-切向分解与归一化信赖域半径设计，首次为重尾噪声下带等式约束的随机优化问题提供了理论保证。 |
| [^17] | [Parameter Estimation for Unnormalized Discrete Models via Empirically Localized Deformed Bregman Divergence](https://arxiv.org/abs/2609.30713) | 本文提出将经验局部化技术与变形Bregman散度相结合来估计非归一化离散模型的参数，在大幅降低归一化常数计算成本的同时，可通过选择变形方式使估计器具备有效性或抗离群噪声鲁棒性等良好统计性质。 |
| [^18] | [Design-Ignoring versus Design-Respecting World Models for Epidemiology](https://arxiv.org/abs/2609.30679) | 该论文首次将流行病学研究设计形式化为世界模型的约束条件，区分了“尊重设计”与“忽略设计”两类世界模型，并通过大规模整群随机检验阴性试验表明：两者在事实重构上表现相当，但在与预期干预问题相关的干预推断上存在本质差异。 |
| [^19] | [Causal Retention in Interactive Agents: Interface Factorization and Selective Adaptation](https://arxiv.org/abs/2609.30650) | 本文提出“因果保留”理论，证明冻结的学习状态能否正确响应独立于训练的机制探针取决于学习接口纤维与探针答案纤维之间的包含关系，并据此构建 Causal Core 系统，通过证据门控写入与选择性适应等机制实现无误差的目标更新。 |
| [^20] | [MARCEDES: Score-based causal discovery under non-Gaussianity with continuous optimization](https://arxiv.org/abs/2609.30643) | 提出了名为MARCEDES的基于评分的因果发现方法，通过引入平均绝对残差风险、行稀疏惩罚和软DAG约束，实现了非高斯误差下因果DAG结构的高效连续优化学习。 |
| [^21] | [Entropy Regularization: A Free Correction to Cross-Entropy for Verified Demonstrations](https://arxiv.org/abs/2609.30572) | 该论文指出在存在多个正确解的可验证任务中，用交叉熵模仿单一专家示范可能与验证器风险目标不一致，并提出以熵正则化作为“免费”修正来控制策略支持集、防止概率质量流向错误输出。 |
| [^22] | [Atelier: Learning Local Self-Supervised Features for CryoEM Volumes via Hypernetworks](https://arxiv.org/abs/2609.30569) | Atelier是一个基于Transformer超网络的自监督框架，通过摊销冷冻电镜图谱的隐式神经表示拟合，实现了高效、跨样本对齐的局部特征学习，可用于大规模冷冻电镜体积的特征提取。 |
| [^23] | [High-dimensional Gaussian Graphical Model Testing for Long-Memory Time Series](https://arxiv.org/abs/2609.30565) | 提出了一种适用于长记忆时间序列的数据自适应高斯图模型条件独立性检验方法，建立了有限样本高斯逼近界并通过块自助法保证超高维情形下的有效性，检验在水平和功效上均具渐近一致性，可应用于fMRI功能连接性分析。 |
| [^24] | [Learning to Replace MCMC in Split-Gibbs Diffusion Posterior Sampling via Deep Unfolding](https://arxiv.org/abs/2609.30539) | 本文提出一种基于深度展开的学习框架，将split-Gibbs扩散后验采样中的Gibbs更新重新表述为高斯去噪问题并通过ODE扩散实现，从而以更低的似然更新计算成本替代了传统MCMC迭代。 |
| [^25] | [Ordinary Nonconvex SGD under Distance-Dependent Moments: Finite-Horizon Stationarity and Nagaev Bounds](https://arxiv.org/abs/2609.30499) | 本文证明，当条件矩允许噪声方差随迭代点距离增长时，普通单样本SGD无需任何修改即可达到与Blum–Gladyshev下界匹配的极小极大随机复杂度，并借助希尔伯特空间Fuk–Nagaev不等式给出高概率Nagaev型界。 |
| [^26] | [Learning to Bias: Machine Learning-Enhanced Particle Filters](https://arxiv.org/abs/2609.30498) | 提出神经最优粒子滤波器（NOPF），通过从离线模拟数据中学习最优提议分布的摊销近似，并将其作为即插即用模块嵌入标准粒子滤波器，借助重要性权重校正保证滤波分布的一致性，从而提升序贯推断的样本效率与高维扩展能力。 |
| [^27] | [Breaking Homogeneity: Diversifying Persona Sets for Creative LLM Outputs](https://arxiv.org/abs/2609.30492) | 该论文将人格多样化建模为集合级条件化问题，在“选择vs生成”与“空间填充vs前沿探索”两个正交设计维度上提出四种方法，显著提升了语言模型创造性输出的多样性，其中进化式人格生成在AUT上将回答多样性提高了78.8%。 |
| [^28] | [Geometric Feature Learning for Functional Data Valued on the Symmetric Positive Definite Manifold](https://arxiv.org/abs/2609.30487) | 该论文提出了MatFAE——一种用于学习对称正定（SPD）矩阵黎曼流形上轨迹的函数神经网络，它将序列视为连续函数以编码轨迹动力学特性，并通过函数权重的形态提供可解释性。 |
| [^29] | [Bayesian Uncertainty Quantification for fMRI Functional Connectivity via Simulation-Based Inference](https://arxiv.org/abs/2609.30445) | 该论文提出一个基于仿真推断的贝叶斯框架，通过将BOLD动力学建模为耦合Ornstein-Uhlenbeck过程并使用序贯神经后验估计，量化了fMRI功能连接估计中来自扫描仪噪声、被试变异性和采集时长三方面来源的不确定性。 |
| [^30] | [An End-to-End Pipeline for Causal ML with Continuous Treatments: An Application to Financial Decision Making](https://arxiv.org/abs/2609.30396) | 该论文提出了一个面向连续处理场景的端到端因果机器学习流水线，创新性地解决了正值性违反检测、高维数据降维以及敏感性分析与估计方法向连续处理空间的适配问题，并成功应用于金融决策。 |
| [^31] | [Cost-Aware Best-LLM Identification using Dueling Feedback](https://arxiv.org/abs/2609.30360) | 该论文提出了一种结合对决反馈与异质查询成本的成本感知多臂老虎机算法，用于在给定置信度下识别最佳大语言模型，并证明了其渐近最优成本性能。 |
| [^32] | [Strategic Self-Consistency](https://arxiv.org/abs/2609.30352) | 本文揭示了一种针对自洽性推理服务的潜在欺诈行为：不忠实的模型提供商可通过策略性地生成并重排额外的推理路径，使每条路径在多数投票中看起来都不可或缺，从而在避开审计检测的情况下人为增加路径数量以向用户多收费。 |
| [^33] | [Adaptive multi-resolution Gaussian processes: Scalable exact inference with naturally data-sparse covariance matrices](https://arxiv.org/abs/2609.30348) | 该论文提出一种自适应多分辨率高斯过程框架，通过直接锚定样本点的自适应多分辨率基函数构建天然数据稀疏的协方差矩阵，并结合稀疏Cholesky逆算法，实现了既可扩展又精确的高斯过程推断。 |
| [^34] | [Low-Rank Friction for Memory-Efficient Transformer Pretraining](https://arxiv.org/abs/2609.30342) | 提出R-iKFAD优化器，通过秩1外积分解替代完整摩擦张量，将优化器状态内存近乎减半，同时保持与iKFAD相当的性能和超参数鲁棒性。 |
| [^35] | [Adaptive Random Matrices in Gaussian Bandits: Spectral Universality and Selection-Induced Outliers](https://arxiv.org/abs/2609.30321) | 该论文证明当臂数量的对数相对维度次线性时，高斯老虎机中任意自适应选择规则都不改变观测矩阵经验谱收敛于Marchenko-Pastur定律的极限行为，从而使贝叶斯后验不确定度和信息获取具有与策略无关的一阶极限，并在线性得分选择下给出精确的条件臂分布与Wishart型Gram矩阵刻画。 |
| [^36] | [Distribution of hitting times for dissipative random dynamical systems on $\mathbb{R}^d$, with application to stochastic gradient descent](https://arxiv.org/abs/2609.30274) | 该论文从遍历理论的视角提出了一种新方法，用于研究一大类随机优化方法的渐近性质，其主要贡献是给出了随机优化算法到达极小值点小邻域的击中时间分布分析，并将其应用于随机梯度下降的收敛性研究。 |
| [^37] | [Scalable Minimum-Volume Simplex Estimation with Non-asymptotic Analysis](https://arxiv.org/abs/2609.25576) | 提出 DeepMVSA 方法，通过神经隐式形式（轻量坐标网络加 LU 三角参数化）将最小体积单纯形估计的内存降至与样本量无关的 O(K^2)、单次遍历成本降至 O(NK^2)，并给出非渐近样本复杂度界与神谕不等式等理论保证。 |
| [^38] | [Beyond Quadratic Loss: The Stability Phase Diagram of Adam](https://arxiv.org/abs/2609.18314) | 该研究通过绘制Adam优化器在$(\beta_1,\beta_2)$参数平面上的稳定性相图，发现一条近似线性边界$1-\beta_2=C(1-\beta_1)$可用于区分训练中是否出现损失尖峰，并揭示超二次损失景观（如高置信交叉熵损失形成的“核心-墙壁”结构）是决定该边界形状的关键因素。 |
| [^39] | [Graph Matching Relaxations and Amortization for Supervised Graph Prediction](https://arxiv.org/abs/2609.15437) | 该论文证明了Gromov-Wasserstein目标是监督图预测中最合适的图匹配松弛形式，并提出基于可微Sinkhorn算法的参数化匹配器来摊销图匹配问题，实现图预测模块与匹配器的联合学习。 |
| [^40] | [High-probability guarantees for linear accessibility in feature superposition](https://arxiv.org/abs/2609.09556) | 该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。 |
| [^41] | [LiD-GLM: Lipschitz-constrained Deep Generalized Linear Models](https://arxiv.org/abs/2608.16340) | 提出一种利用可逆残差网络增强广义线性模型的方法，在保持随机单调性的同时实现非线性参数估计和分布假设的灵活校正。 |
| [^42] | [DAIF: A Data-Driven Intermediate Fusion Framework for Multimodal Supervised Learning via Approximate Message Passing](https://arxiv.org/abs/2608.02769) | DAIF提出了一种数据自适应的中间融合框架，结合随机矩阵理论与非参数依赖性度量，通过根据模态间依赖性对模态聚类并进行经验贝叶斯先验估计，直接从数据中学习融合结构，克服了传统预定义融合架构无法适应模态间真实依赖关系的缺陷。 |
| [^43] | [Think Short, Defer Smart, Act, and Repeat: Calibrated Reasoning and Uncertainty-Aware Deferral for Edge LLM Agents](https://arxiv.org/abs/2607.26865) | TSDS框架通过轻量级收敛探针和基于困惑度的委托规则，在边缘LLM代理中实现推理预算与可靠性的平衡，并利用多目标LTT程序提供同时的有限样本保证。 |
| [^44] | [Econometrics with Pre-Trained Embeddings for Unstructured Data](https://arxiv.org/abs/2607.17378) | 本文针对经济学家使用预训练深度学习模型提取嵌入向量作为协变量这一流行做法，提供了理论基础，并提出“可迁移性”等充分条件，以解决预训练模型跨任务适用性不明和嵌入函数识别困难这两大问题。 |
| [^45] | [Influence Diagnostics in High-dimensional M-estimation: Precise Asymptotics](https://arxiv.org/abs/2607.09250) | 该论文在高维凸M估计中精确刻画了训练点留一影响的渐近分布，发现有影响力的样本平均而言倾向于靠近决策边界，与主动学习中的数据选择启发式方法相契合。 |
| [^46] | [Statistically Valid Post-Training Hyperparameter Selection: From Tuning to Guarantees](https://arxiv.org/abs/2606.25601) | 提出以“先学习后测试”（LTT）范式为核心的统一统计框架，将训练后超参数选择转化为多元假设检验问题，为人工智能系统部署中的超参数调优提供正式的可靠性统计保证。 |
| [^47] | [Amortized quadrature for posterior expectations in inverse problems](https://arxiv.org/abs/2606.15871) | 本文提出“求积场”——一种集合等变神经网络，只需在一个后验族上训练一次，即可对任意观测、任意样本数 M 和任意被积函数，通过一次前向传播生成带符号权重的 M 节点求积格式，在保证精度不劣于蒙特卡洛的同时，避免了传统设计求积法需对每个新观测重复求解优化问题的高昂计算成本。 |
| [^48] | [Policy Regret for Embedding Model Routing: Contextual Bandits with Low-Rank Experts](https://arxiv.org/abs/2606.14929) | 该论文将嵌入模型路由形式化为具有低秩专家的对抗性上下文线性赌博机问题，证明标准后悔度量存在结构性误设或统计不可处理的缺陷，并提出兼具表达能力与高效可学习性的对数二次策略类来实现查询依赖的模型路由。 |
| [^49] | [INFUSER: Influence-Guided Self-Evolution Improves Reasoning](https://arxiv.org/abs/2606.09052) | INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。 |
| [^50] | [Information-Theoretic Bounds for Sparse Covariance Estimation in the Vertical-Split Distributed Model](https://arxiv.org/abs/2606.07124) | 该论文首次证明在纵向切分分布式设置中，对互协方差矩阵施加稀疏性约束能够有效降低通信和样本复杂度，这与水平切分设置下稀疏性无法降低通信成本的结论形成鲜明对比。 |
| [^51] | [Smooth Piecewise Cutting for Neural Operator to Handle Discontinuities and Sharp Transitions](https://arxiv.org/abs/2605.19823) | 提出 Cut-DeepONet 两阶段训练框架，通过将求解域切割为平滑子区域、并将不连续性表示为高维空间中的边界，使神经算子能够高效处理偏微分方程解中的不连续性与尖锐过渡。 |
| [^52] | [Federated Martingale Posterior Samping](https://arxiv.org/abs/2605.18554) | 该论文提出联邦鞅后验采样（FMP），通过客户端上传可训练数据嵌入、服务器集中运行预测采样器的一次性并行协议，摆脱了联邦贝叶斯方法对先验设定的依赖，在性能上与中心化方法高度一致并取得最低的期望校准误差。 |
| [^53] | [Identifying Causal Effects Using a Single Proxy Variable](https://arxiv.org/abs/2604.09135) | 该论文提出SPICE假设，证明在已知混杂因素生成单一（多维）代理变量机制的前提下因果效应可识别，将经典代理变量方法扩展到多维连续场景，并开发了适用于离散和连续处理的神经网络估计框架SPICE-Net。 |
| [^54] | [On the Expressive Power of Transformers for Contextual Relations](https://arxiv.org/abs/2603.25860) | 本文基于概率与最优传输理论构建了数学框架，揭示了注意力归一化与最优传输的深刻联系——softmax归一化产生条件关系而Sinkhorn归一化产生联合关系——并证明了Transformer在表示上下文关系上的通用逼近能力。 |
| [^55] | [Adaptive Subspace Modeling With Functional Tucker Decomposition](https://arxiv.org/abs/2603.25530) | 提出函数型Tucker分解（FTD），将模态级连续性约束直接嵌入Tucker分解，在RKHS中以函数形式建模连续模态，并推导重构误差界为跨域子空间迁移提供理论保证。 |
| [^56] | [Murmurations, Mestre--Nagao sums, and Convolutional Neural Networks for elliptic curves](https://arxiv.org/abs/2603.17681) | 本文将一维卷积神经网络应用于椭圆曲线的 Frobenius 迹，实现了对解析秩的高精度预测，并通过显著性曲线揭示了机器学习预测、群舞现象与 Mestre–Nagao 和之间的深刻联系。 |
| [^57] | [Generative Modeling of Discrete Data Using Geometric Latent Subspaces](https://arxiv.org/abs/2601.21831) | 该论文提出一种几何潜在子空间框架，在类别分布乘积流形的指数参数空间中通过几何主成分分析（GPCA）学习高维离散数据的低维表示，并借助等距黎曼几何实现一致的流匹配生成建模。 |
| [^58] | [Learning Operators by Regularized Stochastic Gradient Descent with Operator-valued Kernels](https://arxiv.org/abs/2504.18184) | 本文针对从波兰空间到可分希尔伯特空间的算子学习问题，分析了算子值核再生核希尔伯特空间中在线与有限时域两种设置下的正则化随机梯度下降算法，建立了对输出空间维度无显式依赖、且在期望意义下接近最优的误差界，并给出了高概率估计与几乎必然收敛的结论。 |
| [^59] | [Foundations of Reinforcement Learning and Interactive Decision Making](https://arxiv.org/abs/2312.16730) | 该专著在统一的统计框架下系统阐述了从多臂老虎机到基于函数逼近的强化学习的交互式决策算法设计与复杂度理论，并展示了如何将监督学习方法转化为决策算法、分析其性能以及判定问题的可解性。 |

# 详细

[^1]: 反向扩散的一阶平稳性

    First-Order Stationarity of Reverse Diffusions

    [https://arxiv.org/abs/2609.31612](https://arxiv.org/abs/2609.31612)

    该论文为扩散模型建立了首个一阶平稳性理论，证明基于SDE的过阻尼与欠阻尼朗之万扩散的逆时流在前向过程平稳势强凸（仅针对加噪过程而非数据）时以指数速率收缩相对Fisher散度，并为离散化采样器建立了与非凸优化中平均梯度范数保证相对应的一阶平稳性界。

    

    近期研究文献表明优化与采样之间存在紧密联系。我们为扩散模型发展了相应的一阶理论。首先，只要前向过程的平稳势是强凸的——这是对所选加噪过程的条件，而非对数据的要求——基于SDE的过阻尼与欠阻尼朗之万扩散的逆时流就会以明确的指数速率收缩相对Fisher散度。这是基于SDE的反向扩散所独有的优势，而基于ODE的反向过程并不具备这一性质。其次，我们引入离散化分析，为过阻尼与欠阻尼扩散模型的采样器建立了平均一阶平稳性界——这是非凸优化中平均梯度范数保证在采样领域的对应物。与非凸优化类似，这种不依赖凸性假设的证明是局部性的：它保证的是得分的一致性，而非全局众数权重。

    arXiv:2609.31612v1 Announce Type: new  Abstract: Recent literature has shown a strong connection between optimization and sampling. We develop the corresponding first-order theory for diffusion models. First, the SDE-based reverse-time flows of overdamped and underdamped Langevin diffusions contract relative Fisher divergences at explicit exponential rates whenever the stationary potential of the forward process is strongly convex---a condition on the noising process one chooses, not on the data. This is a unique advantage of SDE-based reverse diffusion, absent in the reverse process based on ODEs. Second, we incorporate discretization and establish averaged first-order stationarity bounds---the sampling analog of averaged gradient-norm guarantees in nonconvex optimization---for samplers of both overdamped and underdamped diffusion models. As in nonconvex optimization, the convexity-free certificate is local: it guarantees score consistency, not global mode weights.
    
[^2]: 深度粗糙波动率中的不确定性与可解释性：一种神经信息论后验方法

    Uncertainty and Explainability in Deep Rough Volatility: A Neural Information-Theoretic Posterior Approach

    [https://arxiv.org/abs/2609.31570](https://arxiv.org/abs/2609.31570)

    该论文提出一个基于仿真的推断框架，通过神经比率估计学习粗糙Heston模型参数在隐含波动率曲面条件下的后验分布，结合异方差神经代理定价器生成考虑不确定性的价格区间，并引入信息论可解释性方法Hellinger-SHAP。

    

    深度学习极大地加速了复杂随机波动率模型的校准，但仅靠神经点校准无法捕捉在观察到隐含波动率（IV）曲面后仍然存在的不确定性。我们开发了一个基于仿真的推断框架，用于粗糙Heston（rHeston）模型的校准，该框架学习在给定IV曲面条件下模型参数的后验分布。利用神经比率估计，我们获得了校准后的后验样本，这些样本可以通过异方差神经代理定价器进行传播，用于路径依赖的奇异期权定价。由此产生的后验预测分布结合了残余参数不确定性与条件代理不确定性，并生成考虑不确定性的价格区间。我们进一步提出了Hellinger-SHAP，这是一种用于后验推断的信息论可解释性方法。它不再归因于单一的参数点估计，而是应用于……

    arXiv:2609.31570v1 Announce Type: new  Abstract: Deep learning has substantially accelerated the calibration of complex stochastic-volatility models, but neural point calibration alone does not capture the uncertainty remaining after an implied-volatility (IV) surface has been observed. We develop a simulation-based inference framework for rough Heston (rHeston) calibration that learns the posterior distribution of the model parameters conditional on an IV surface. Using neural ratio estimation, we obtain calibrated posterior samples that can be propagated through heteroscedastic neural surrogate pricers for path-dependent exotic options. The resulting posterior-predictive distributions combine residual parameter uncertainty with conditional surrogate uncertainty and yield uncertainty-aware price intervals.   We further introduce Hellinger-SHAP, an information-theoretic explainability method for posterior inference. Rather than attributing a single parameter point estimate, it applies 
    
[^3]: 超越经验支撑：基于Sinkhorn最优传输的结构化离群点生成

    Beyond Empirical Support: Structured Outlier Generation via Sinkhorn Optimal Transport

    [https://arxiv.org/abs/2609.31470](https://arxiv.org/abs/2609.31470)

    该论文提出SBOG框架，将Sinkhorn最优传输几何与分布鲁棒边界建模相结合，在潜空间中结构化地生成弱支撑边界区域的离群点，从而更有效地评估和提升机器学习系统应对分布偏移的鲁棒性。

    

    离群点对于评估和提升机器学习系统的鲁棒性至关重要，尤其是当未来分布可能与历史训练数据存在显著差异时。在高风险应用中，鲁棒性往往依赖于有限数据集无法捕捉的罕见案例，这使得简单的重采样或扰动方法不足以用于压力测试场景的生成。现有的离群点合成方法通常依赖于稀疏邻域、低支撑潜空间区域或分类器边界穿越，这些方法可能是启发式的、不稳定的，并且与特定的模态或架构绑定。因此，我们提出了Sinkhorn边界离群点生成（SBOG），这是一个结构化的潜空间离群点生成框架，它将Sinkhorn最优传输几何与分布鲁棒边界建模相结合。由Sinkhorn诱导的支撑代价引导采样器朝向弱支撑的边界区域，同时语义约束（摘要原文在此处截断）

    arXiv:2609.31470v1 Announce Type: new  Abstract: Outliers are essential for evaluating and improving the robustness of machine learning systems, especially when future distributions may differ significantly from historical training data. In high-stakes applications, robustness often depends on rare cases that finite datasets fail to capture, making simple resampling or perturbation insufficient for stress scenario generation. Existing outlier synthesis methods typically rely on sparse neighborhoods, low support latent regions, or classifier boundary crossings, which can be heuristic, unstable, and tied to specific modalities or architectures. We therefore propose Sinkhorn Boundary Outlier Generation (SBOG), a structured framework for latent-space outlier generation that couples Sinkhorn optimal transport geometry with distributionally robust boundary modeling. The resulting Sinkhorn-induced support cost guides the sampler toward weakly supported boundary regions, while semantic constra
    
[^4]: 增长几何复杂度下的非参数上下文学习：Transformer的极小极大最优性与局部几何自适应性

    Nonparametric In-Context Learning under Growing Geometric Complexity: Minimax Optimality and Local Geometry-Adaptivity of Transformers

    [https://arxiv.org/abs/2609.31458](https://arxiv.org/abs/2609.31458)

    本文在由随样本量增长、维度与光滑度异构的流形混合所刻画的未知局部几何下研究非参数上下文学习，建立了极小极大最优下界，并证明配备几何预条件器的结构感知两阶段softmax Transformer能达到该最优性且可自适应局部几何。

    

    Transformer已成为上下文学习（ICL）的核心架构，尤其体现在其于大型语言模型中的最先进性能上。这一成功促使人们理解Transformer如何在几何异构数据中利用与任务相关的结构。然而，现有的非参数ICL理论大多集中于欧几里得空间或单一流形模型。为填补这一空白，我们研究了未知局部几何下的预测问题，其中局部几何由随样本量变化的流形混合建模，各流形在维度、光滑度和采样质量上均具有异构性。在局部分离与小扰动条件下，我们建立了一个刻画各组成部分聚合难度的极小极大下界，并构造了一个达到匹配上界的oracle切向局部多项式估计器。该估计器与一种结构感知的两阶段softmax Transformer相关联，该Transformer配备了几何预条件器……

    arXiv:2609.31458v1 Announce Type: new  Abstract: Transformers have become a central architecture for in-context learning (ICL), particularly through their state-of-the-art performance in large language models. This success motivates understanding how transformers exploit task-relevant structure in geometrically heterogeneous data. However, existing nonparametric ICL theory has largely focused on Euclidean domains or single-manifold models. To address this gap, we study the prediction problem under unknown local geometry, modeled by sample size-dependent mixtures of manifolds with heterogeneous dimensions, smoothness, and sampling masses. Under local separation and small-perturbation conditions, we establish a minimax lower bound capturing the aggregate difficulty of the components and construct an oracle tangent local-polynomial estimator with a matching upper bound. This estimator is connected to a structure-informed, two-stage softmax transformer with a geometric preconditioner and c
    
[^5]: 多任务原子模拟中的多变量共形不确定性传播：成功与陷阱

    Multivariate conformal uncertainty propagation in multitask atomistic simulation: Successes and pitfalls

    [https://arxiv.org/abs/2609.31384](https://arxiv.org/abs/2609.31384)

    本文将多变量共形校准方法应用于多任务原子模拟的多阶段工作流，通过向下游目标量传播不确定性集合以捕捉误差抵消效应，并总结了该方法取得的成功与面临的陷阱。

    

    机器学习已成为设计兼顾效率与精度的原子间势的标准工具，但不确定性量化仍是一个悬而未决的问题。多尺度模拟带来了额外的挑战：跨尺度的稳健不确定性量化。即使在单一尺度内，计算往往是多阶段的，会产生一系列目标量，每个目标量依赖于前一个，且各自带有一定的不确定性。共形方法已发展成为一种模型无关的框架，用于重新校准代理模型的预测，从而以用户指定的概率生成包含真实值的预测集合。对于多阶段工作流，我们需要对多种化学性质和原子构型进行不确定性校准，并希望将不确定性集合传播到下游的目标量。这种传播应当能够捕捉材料科学中许多下游目标量中出现的误差抵消效应……

    arXiv:2609.31384v1 Announce Type: cross  Abstract: Machine learning has become the standard tool for the design of interatomic potentials which balance efficiency and accuracy, but uncertainty quantification remains an open problem. Multiscale simulations introduce an additional challenge: robust uncertainty quantification across scales. Even within one scale, computations are often multistage, producing a sequence of target quantities, each dependent on the previous, and each with some uncertainty. Conformal methods have emerged as a model agnostic framework for recalibrating surrogate predictions to produce sets which contain the truth at a user-specified rate. For multistage workflows, we require uncertainty calibration for multiple chemical properties and atomistic configurations, and we want to propagate uncertainty sets to downstream quantities of interest. Such propagation should capture the error cancellations which occur in many downstream targets in materials science; for ins
    
[^6]: 基于贝叶斯树邻接文法的方程发现

    Equation discovery with Bayesian tree-adjoining grammars

    [https://arxiv.org/abs/2609.31368](https://arxiv.org/abs/2609.31368)

    本文首次将树邻接文法置于贝叶斯框架下，利用结构保持树移动的可逆跳MCMC采样器推断模型结构、参数和预测的联合后验分布，实现了超越传统点估计的概率化方程发现与非线性系统辨识。

    

    树邻接文法（TAG）最近被引入非线性系统辨识（NLSI）领域，作为将整个模型类编码为有限语法规则集合的手段，候选模型由此被组装为树结构。现有的基于TAG的辨识器依赖进化优化，并返回模型结构的点估计。本文转而在贝叶斯框架下提出TAG方法：在树结构及其参数上定义生成式先验，并使用带有结构保持树移动的可逆跳MCMC采样器来推断模型结构、参数和预测的联合后验分布。文中考虑了两种训练目标：即采用共轭参数提议的一步预测目标，以及通过无似然推断处理的基于仿真的目标。该方法在仿真多项式NARX系统、Silverbox基准和波浪载荷数据上进行了验证。

    arXiv:2609.31368v1 Announce Type: new  Abstract: Tree-Adjoining Grammars (TAGs) have recently been introduced to Nonlinear System Identification (NLSI) as a means of encoding an entire model class as a finite set of grammatical rules, from which candidate models are assembled as trees. Existing TAG-based identifiers rely on evolutionary optimisation and return point estimates of the model structure. This paper instead proposes the TAG framework within a Bayesian setting. A generative prior is defined over tree structures and their parameters, and a Reversible-Jump MCMC sampler with structure-preserving tree moves is used to infer the joint posterior over model structure, parameters and predictions. Two training objectives are considered; that is, a one-step-ahead objective with conjugate parameter proposals, and a simulation-based objective handled by likelihood-free inference. The approach is validated on a simulated polynomial NARX system, the Silverbox benchmark, and wave-loading da
    
[^7]: Brenier 遇见对抗训练：面向鲁棒学习的最优传输几何

    Brenier Meets Adversarial Training: Optimal Transport Geometry for Robust Learning

    [https://arxiv.org/abs/2609.31363](https://arxiv.org/abs/2609.31363)

    该论文将带 Wasserstein 惩罚的分布鲁棒优化中的对抗问题重新表述为最优传输映射的优化问题，证明了最优映射满足循环单调性，指出标准对抗训练因违反该性质而浪费传输成本，并提出多起点粒子上升等方法予以改进。

    

    分布鲁棒优化（DRO）为分布偏移下的学习提供了一个有原则的框架，但其在实际中的应用受到阻碍，原因在于对非凸损失函数评估最坏情况风险十分困难。我们研究了一种带惩罚项的 DRO 形式，其中对抗者可以选择任意分布，但偏离经验分布时需承担 Wasserstein 惩罚。我们证明，对抗者的问题可以被重新表述为关于传输映射的优化问题，这些映射将经验样本推送到对抗样本，并且我们证明了最优映射是循环单调的。我们还表明，标准的对抗训练——基于逐样本的局部优化——违反了循环单调性，并浪费了传输成本，除非对对抗者施加严格限制。我们提出了两种补救方法。首先，我们引入了多起点粒子上升方法，该方法交替进行并行梯度上升与重新分配，以强制执行循环单调性……

    arXiv:2609.31363v1 Announce Type: cross  Abstract: Distributionally robust optimization (DRO) provides a principled framework for learning under distribution shift, but its practical use is hindered by the difficulty of evaluating worst-case risks for nonconvex loss functions. We study a penalized DRO formulation in which the adversary may choose any distribution but incurs a Wasserstein penalty for deviating from the empirical distribution. We show that the adversary's problem can be reformulated as an optimization problem over transport maps that push empirical samples to adversarial ones, and we prove that optimal maps are cyclically monotone. We also show that standard adversarial training---based on per-sample local optimization---violates cyclical monotonicity and wastes transport costs unless the adversary is severely restricted. We propose two remedies. First, we introduce multi-start particle ascent, which alternates parallel gradient ascent with reassignment to enforce cyclic
    
[^8]: LUCID：面向时间序列推断与发现的混杂学习

    LUCID: Learning Under Confounding for Inference and Discovery in Time Series

    [https://arxiv.org/abs/2609.31315](https://arxiv.org/abs/2609.31315)

    LUCID 提出了一种机制自适应的去混杂层，利用 Marcenko–Pastur 谱路由器估计混杂机制并施加相应的去混杂策略，可封装现有因果发现算法，从而在时间序列中有效消除未观测混杂因子造成的虚假关联。

    

    未观测的共同原因在现实世界的时间序列中普遍存在，它们会诱发虚假关联，使因果发现方法误将其识别为直接因果边。我们提出了 LUCID（Learning Under Confounding for Inference and Discovery），这是一种机制自适应的去混杂层：它首先利用 Marcenko–Pastur 谱路由器从数据中估计混杂机制，然后应用与该机制相匹配的去混杂策略。当谱结构表明存在普遍的因子混杂时，LUCID 会衰减因子主导的变异，并从所得的新息（innovations）中恢复同期（滞后-0）结构，其边选择则针对数据驱动的无边零假设进行校准。该方法不依赖于特定的发现算法，而是可以封装现有的因果发现引擎；我们在三种此类方法上展示了一致的性能提升。在一个涵盖混杂因子强度与稀疏性变化的多样化合成分布外基准上……

    arXiv:2609.31315v1 Announce Type: cross  Abstract: Unobserved common causes are pervasive in real-world time series and can induce spurious associations that causal discovery methods mistake for direct edges. We propose LUCID (Learning Under Confounding for Inference and Discovery, a regime-adaptive deconfounding layer that first estimates the confounding regime from data using a Mar\v{c}enko--Pastur spectral router, then applies a deconfounding strategy matched to that regime. When the spectrum indicates pervasive factor confounding, LUCID attenuates factor-dominated variation and recovers contemporaneous (lag-$0$) structure from the resulting innovations, with edge selection calibrated against a data-driven edge-free null. Rather than being tied to a particular discovery algorithm, it can wrap existing discovery engines; we demonstrate consistent improvements across three such methods. On a diverse synthetic out-of-distribution benchmark spanning changes in confounder strength and sp
    
[^9]: 随机Nesterov加速的几何矩收缩

    Geometric Moment Contraction for Stochastic Nesterov Acceleration

    [https://arxiv.org/abs/2609.31303](https://arxiv.org/abs/2609.31303)

    该论文为常参数随机Nesterov加速算法建立了几何矩收缩理论，给出了仅要求梯度具有有限 $p$ 阶矩的显式步长收缩判据，即使梯度具有无穷方差（$1<p<2$）也能保证 $L^p$ 收敛。

    

    我们研究了常参数随机Nesterov迭代的几何矩收缩（GMC）：\[ Y_k=\Theta_k+\beta(\Theta_k-\Theta_{k-1}),\qquad \Theta_{k+1}=Y_k-\gamma G(Y_k,X_{k+1}). \] 在均值强单调性和随机 $L^p$ Lipschitz 连续性条件下，通过一个显式的Perron比较，证明了当 $\beta\gamma L_p<(1-\beta)(1-q_{\gamma,p})$ 时存在同步的 $L^p$ 收缩。这一直接判据涵盖了 $1<p<2$ 时梯度具有无穷方差的情形，但其小步长机制要求 $\beta<\mu/(\mu+L_p)$。作为补充，一个幂-Lyapunov论证仅利用梯度有限的 $p$ 阶矩，就为每个固定的 $\beta<1$ 和每个 $p>1$ 建立了一个正的（通常小得多）步长区间。在 $p=2$ 时，一个更简单的显式判据给出 \[ 0<\gamma<\frac{2\mu(1-\beta)^2}{L_2^2(1-\beta+2\beta^2)}. \] 其关于高动量的二次缩放是所选度量方式的局限，而非精确的稳定性边界。我们量化了这一损失，并证明……

    arXiv:2609.31303v1 Announce Type: new  Abstract: We study geometric moment contraction (GMC) of the constant-parameter stochastic Nesterov recursion \[ Y_k=\Theta_k+\beta(\Theta_k-\Theta_{k-1}),\qquad \Theta_{k+1}=Y_k-\gamma G(Y_k,X_{k+1}). \] Under mean strong monotonicity and stochastic $L^p$ Lipschitz continuity, an explicit Perron comparison proves synchronous $L^p$ contraction when $\beta\gamma L_p<(1-\beta)(1-q_{\gamma,p})$. This direct criterion includes infinite-variance gradients for $1<2$, but its small-step regime requires $\beta<\mu/(\mu+L_p)$. A complementary power-Lyapunov argument establishes a positive, generally much smaller, step-size interval for every fixed $\beta<1$ and every $p>1$, using only a finite $p$th gradient moment. At $p=2$, a simpler explicit certificate gives \[ 0<\gamma<\frac{2\mu(1-\beta)^2}{L_2^2(1-\beta+2\beta^2)}. \] Its quadratic high-momentum scaling is a limitation of the chosen metric, not a sharp stability boundary. We quantify this loss, prov
    
[^10]: DIAL：具有自适应人类偏好校准的位置去偏大语言模型评判器

    DIAL: Position-Debiased LLM Judges with Adaptive Human Preference Calibration

    [https://arxiv.org/abs/2609.31215](https://arxiv.org/abs/2609.31215)

    提出DIAL统一框架，利用大量LLM比较结合少量人类比较，分离并消除LLM评判器中的位置偏差，并将去偏后的偏好结构自适应校准至人类偏好目标，同时提供可识别性理论与不确定性量化保证。

    

    以大语言模型作为评判器可以实现可扩展的评估，但其判断可能对回答顺序敏感，并且即使去除这种位置效应后，其判断仍可能与人类偏好存在系统性偏差。我们提出了DIAL，这是一个统一框架，它将丰富的LLM比较与有限的人类比较相结合，以分离评判器特有的位置效应，学习位置去偏后LLM偏好中的共享结构，并将该结构自适应地校准至人类偏好目标。在理论上，我们研究了DIAL的三个方面：(i) 潜在LLM偏好、位置效应和人类校准的可识别性；(ii) 在LLM锚定与有限人类证据之间取得平衡的自适应估计方法；(iii) 针对校准后人类偏好的固定权重不确定性量化。在实证方面，我们在受控模拟和三个人类偏好数据集上分别评估了位置去偏和人类对齐性能（摘要原文在此处截断）。

    arXiv:2609.31215v1 Announce Type: new  Abstract: Large language models (LLMs) as a judge enable scalable evaluation, but their judgments can be sensitive to response order and, even after removing such position effects, can still diverge systematically from human preferences.We introduce DIAL, a unified framework that combines abundant LLM comparisons with limited human comparisons to separate judge-specific position effects, learn shared structure in position-debiased LLM preferences, and adaptively calibrate that structure toward the human preference target. Theoretically, we study three aspects of DIAL: (i) identification of latent LLM preferences, position effects, and human calibration; (ii) adaptive estimation that balances LLM anchoring against limited human evidence; and (iii) fixed-weight uncertainty quantification for the calibrated human preference. Empirically, we evaluate position debiasing and human alignment separately in controlled simulations and on three human-prefere
    
[^11]: 师生树委员会机器中的平坦性-泛化关系

    A Flatness-Generalization Relation in the Teacher-Student Tree-Committee Machine

    [https://arxiv.org/abs/2609.31101](https://arxiv.org/abs/2609.31101)

    该论文在师生树委员会机器中通过零温度吉布斯形式与Edwards-Jones形式解析刻画了经验损失典型极小值点的可观测量和Hessian谱，进而检验泛化误差随数据量增大而下降是否对应于损失景观平坦性的提升。

    

    摘要（arXiv:2609.31101v1）：损失景观在极小值点处的平坦性是用于推理神经网络泛化能力的一种广泛使用的启发式指标，然而关于这一关系的证据大多停留在经验层面且存在争议。我们在师生树委员会机器中研究这一关系，在该模型中，经验风险最小化（ERM）估计量和Hessian谱在成比例的高维极限下均可解析求解。首先，我们采用零温度吉布斯形式，对经验损失的典型极小值点的可观测量给出预测。其次，我们利用Edwards-Jones形式推导这些典型极小值点附近的极限Hessian预解式。所有预测均与有限规模的梯度下降模拟结果一致。最后，我们研究了三种平坦性度量，即谱的左边缘、右边缘和谱均值，并检验随数据集规模增大而带来的泛化误差下降是否对应于平坦性的提升。我们发现……

    arXiv:2609.31101v1 Announce Type: cross  Abstract: The flatness of the loss landscape at a minimizer is a widely used heuristic for reasoning about neural-network generalization, yet evidence for this relation is mostly empirical and controversial. We study this relation in a teacher-student tree committee machine, where both the ERM estimator and the Hessian spectrum are analytically tractable in the proportional high-dimensional limit. First, we use a zero-temperature Gibbs formulation to obtain predictions for the observables of the typical minimizers of the empirical loss. Secondly, we use Edwards-Jones formalism to derive the limiting Hessian resolvent around these typical minimizers. All predictions agree with finite-size gradient-descent simulations. Finally, we study three measures of flatness, namely the left and right edges and the spectral mean, and check if a decrease in generalization error as the dataset size is increased corresponds to an increase in flatness. We find th
    
[^12]: SAGE：面向物种分布建模的采样感知全局评估基准

    SAGE: A sampling-aware global evaluation benchmark for species distribution modeling

    [https://arxiv.org/abs/2609.31082](https://arxiv.org/abs/2609.31082)

    该论文提出了SAGE——一个采样感知的全局评估基准，通过结合GBIF训练记录与sPlotOpen植被调查数据，充分考虑采样偏差和物种层面（尤其是稀有物种）的性能差异，为深度学习多物种分布模型提供更可靠、更具信息量的评估。

    

    了解物种的分布位置是生物多样性研究与保护工作的基础。物种分布模型（SDMs）将物种观测记录与环境条件相关联，以估计物种的空间分布。然而，模型的准确性会随底层数据和模型的不同而变化，因此明确模型对哪些物种可信至关重要。基于深度学习的物种分布模型（"DeepSDMs"）如今可联合建模数千个物种，并利用了数以亿计的社区科学记录。在这种规模下，平均性能指标会掩盖显著的物种层面差异，尤其是对于往往最受保护关注的稀有物种。此外，观测记录存在强烈的偏差，使得物种出现次数具有误导性。考虑这些因素对于对多物种SDMs进行可靠且富有信息量的评估必不可少。在此，我们引入了一个采样感知全局评估基准，该基准将用于训练的GBIF记录与sPlotOpen植被……

    arXiv:2609.31082v1 Announce Type: cross  Abstract: Knowing where species occur is fundamental for biodiversity research and conservation. Species distribution models (SDMs) link species observations to environmental conditions to estimate their spatial distribution. However, accuracy varies with the underlying data and models, making it essential to know for which species models can be trusted. Deep-learning-based SDMs ("DeepSDMs") now jointly model thousands of species, drawing on hundreds of millions of community-science records. At this scale, averaging performance hides substantial species-level variability, particularly for rare species, often of greatest conservation concern. Records are also strongly biased, making occurrence counts misleading. Accounting for these factors is essential for a reliable and informative evaluation of multi-species SDMs. Here, we introduce a Sampling-Aware Global Evaluation (SAGE) benchmark, combining GBIF records for training with sPlotOpen vegetati
    
[^13]: 面向聚类回归的精确求解子样本集成方法：无需截尾水平的截尾技术

    Ensembles of Exactly Solved Subsamples for Clusterwise Regression: Trimming Without a Trimming Level

    [https://arxiv.org/abs/2609.31019](https://arxiv.org/abs/2609.31019)

    该论文提出一种基于精确求解随机子样本的聚类回归集成方法，可自动估计截尾水平而无需预先设定，在响应变量含高达20%粗大离群值时实现0.89的最坏情况准确率，优于传统的截尾交替法。

    

    聚类最小二乘法将回归数据划分为K个组，并为每组拟合独立的线性模型。我们研究了一种基学习器为精确求解的集成方法：在B个大小为m << n的随机子样本上，将问题求解至全局最优，然后通过最近曲面分配扩展每个解，对齐标签，并通过投票或选择方式组合各次重复结果。每个重复结果都是m个点上的经验K-量化器，因此该集成方法可以进行精确分析。只要“干净子样本的概率”与“干净数据上重复结果准确率”的乘积超过二分之一，投票在某个单元上就是正确的，无论污染值如何；在集成方法尚未标记的单元上进行迭代，可得到一种能够估计截尾水平的变体，而无需事先给定截尾水平。当响应变量中存在高达20%的粗大离群值时，该方法的最坏情况准确率达到0.89，而真实污染比例下的截尾交替法仅为0.80，且在任何固定截尾水平下表现都更低。

    arXiv:2609.31019v1 Announce Type: cross  Abstract: Clusterwise least squares partitions regression data into K groups with separate linear fits. We study an ensemble whose base learner is exact: solve the problem to global optimality on each of B random subsamples of size m << n, extend each solution by nearest-surface assignment, align the labels, and combine the replicates by vote or by selection. Each replicate is then an empirical K-quantizer on m points, and the ensemble admits an exact analysis. A vote is correct at a unit once the probability of a clean subsample times the clean-data replicate accuracy exceeds one half, whatever the contaminating values; iterating the ensemble on the units it has not flagged gives a variant that estimates the trimming level rather than requiring it. With up to 20% of gross outliers in the response its worst-case accuracy was 0.89, against 0.80 for trimmed alternation at the true contamination fraction and less at every fixed level tried. Conditi
    
[^14]: 耦合用法-语义过程：具有时间性和可归因性的词汇语义变化

    Coupled Usage-Sense Processes: Temporal and Attributable Lexical Semantic Change

    [https://arxiv.org/abs/2609.30974](https://arxiv.org/abs/2609.30974)

    本文提出耦合用法-语义过程（CUSP）框架，通过单一的边缘保持时序过程，不仅能量化词汇语义变化的幅度，还能精确确定变化发生的时间、将变化分解为成分移动与重组的机制，并归因到具体的词语用法。

    

    词汇语义变化通常通过独立采样的时期分布之间的标量距离来概括。这种方法只能衡量一个词变化了多少，但无法揭示它何时发生变化、哪些机制和成分移动承载了这种变化，或者哪些用法支持这种归因。我们提出了耦合用法-语义过程，它从单一的保持边缘分布的时序过程中推导出这些答案。层次化耦合通过潜在用法成分关联上下文分布，而马尔可夫组合使相邻以及更长时间跨度上的对应关系相互兼容。位移算子量化变化的幅度与时间，将变异精确地分解为成分中心的移动和成分内部的重组织，并将其归因于被迁移的成分对。词局部模式解析出变化的不同方向及其随时间的活跃程度，而来自归因成分的代表性语段则……

    arXiv:2609.30974v1 Announce Type: new  Abstract: Lexical semantic change is usually summarized by a scalar distance between independently sampled period distributions. This measures how much a word changed, but does not reveal when it changed, which mechanisms and component movements carried the change, or which usages support the attribution. We introduce Coupled Usage--Sense Processes (CUSP), which derives these answers from a single marginal preserving temporal process. A hierarchical coupling relates contextual distributions through latent usage components, while Markov composition makes adjacent and longer span correspondences compatible. Displacement operators quantify change magnitude and timing, split variation exactly between movement of component centers and reorganization within components, and attribute it to transported component pairs. Word-local modes resolve distinct directions of change and their activity over time, while representative passages from attributed compone
    
[^15]: 指数倾斜联合偏移下的保形预测

    Conformal Prediction under Exponential-Tilt Joint Shift

    [https://arxiv.org/abs/2609.30886](https://arxiv.org/abs/2609.30886)

    该论文研究了在输入分布及其与结果关系同时发生联合偏移时，比较了在保形预测校准中直接使用ExTRA估计的指数倾斜权重与额外对源预测分布进行倾斜这两种适应策略的覆盖性能。

    

    当数据分布部署后发生变化时，保形预测可能会失去覆盖保证。我们研究了利用带标签的源数据和未带标签的目标输入进行适应的方法，允许输入分布及其与结果之间的关系同时发生变化。我们采用由Maity等人（2023）针对分类问题提出的指数倾斜重加权对齐方法来估计结构化的分布偏移。我们比较了两种策略：在保形校准中使用其估计权重，以及在此外再对源预测分布进行倾斜。通过共享学习预测器、估计权重、校准样本和测试观测数据，隔离出了倾斜操作本身的影响。现有理论表明，使用真实权重时两种方法均可达到目标覆盖率，而使用估计权重时两者具有共同的覆盖率界。识别计算以及关于评分如何与权重估计误差相互作用的分析，有助于解释为什么两者性能仍然可能存在差异。在合成回归实验……

    arXiv:2609.30886v1 Announce Type: new  Abstract: Conformal prediction can lose coverage when the data distribution changes after deployment. We study adaptation using labeled source data and unlabeled target inputs, allowing both the input distribution and its relationship with outcomes to change. We use Exponential Tilt Reweighting Alignment (ExTRA), introduced for classification by Maity et al. (2023), to estimate structured distribution shifts. We compare using its estimated weights in conformal calibration with additionally tilting the source predictive distribution. Shared learned predictors, estimated weights, calibration samples, and test observations isolate the effect of tilting. Existing theory gives both procedures target coverage with true weights and a common coverage bound with estimated weights. Identification calculations and an analysis of how scoring interacts with weight estimation error help explain why their performance can nevertheless differ. In a synthetic regre
    
[^16]: TR-SSQP：一种用于重尾噪声下约束随机优化的信赖域方法

    TR-SSQP: A Trust-Region Method for Constrained Stochastic Optimization under Heavy-Tailed Noise

    [https://arxiv.org/abs/2609.30732](https://arxiv.org/abs/2609.30732)

    本文提出TR-SSQP方法，在随机序列二次规划框架下通过法向-切向分解与归一化信赖域半径设计，首次为重尾噪声下带等式约束的随机优化问题提供了理论保证。

    

    我们考虑具有确定性等式约束的随机非线性优化问题。虽然无约束随机优化已被充分理解，但在约束设定下最优性与可行性之间的相互作用带来了重大挑战。此外，现有的约束随机方法的理论保证主要依赖于有界方差假设，使得重尾噪声情形在很大程度上尚未被探索。为了填补这一空白，我们在随机序列二次规划框架内提出了一种新颖的信赖域方法，称为TR-SSQP。我们的方法在步长计算中采用法向-切向分解来平衡最优性与可行性。此外，我们在信赖域半径的设计中引入了归一化机制，并结合Polyak动量进行梯度估计，从而在不使用梯度裁剪的情况下确保稳定的更新。当信赖域半径与……（摘要在此处被截断）

    arXiv:2609.30732v1 Announce Type: cross  Abstract: We consider stochastic nonlinear optimization problems with deterministic equality constraints. While unconstrained stochastic optimization is well understood, the interplay between optimality and feasibility in the constrained setting poses significant challenges. Moreover, existing theoretical guarantees for constrained stochastic methods predominantly rely on bounded-variance assumptions, leaving the heavy-tailed noise regime largely unexplored. To address this gap, we propose a novel trust-region method within the stochastic sequential quadratic programming framework, termed TR-SSQP. Our method employs a normal-tangential decomposition in the step computation to balance optimality and feasibility. In addition, we incorporate a normalization mechanism in the design of the trust-region radius, together with Polyak momentum for gradient estimation, ensuring stable updates without gradient clipping. When the trust-region radius and the
    
[^17]: 基于经验局部化变形Bregman散度的非归一化离散模型参数估计

    Parameter Estimation for Unnormalized Discrete Models via Empirically Localized Deformed Bregman Divergence

    [https://arxiv.org/abs/2609.30713](https://arxiv.org/abs/2609.30713)

    本文提出将经验局部化技术与变形Bregman散度相结合来估计非归一化离散模型的参数，在大幅降低归一化常数计算成本的同时，可通过选择变形方式使估计器具备有效性或抗离群噪声鲁棒性等良好统计性质。

    

    概率模型的参数估计是机器学习领域的一项重要任务。对于离散变量的模型，其归一化常数的计算有时非常困难，已有大量研究致力于避免归一化常数的计算。在本文中，我们通过结合经验局部化技术与变形Bregman散度来应对这一难题。经验局部化技术能够大幅降低归一化常数计算的计算成本；此外，适当选择Bregman散度的变形方式，可以为所提出的估计器赋予多种良好的统计性质，例如有效性或对离群噪声的鲁棒性。

    arXiv:2609.30713v1 Announce Type: new  Abstract: Estimation of parameter of probabilistic models is an important task in the field of machine learning.For models of discrete variables, calculation of the normalization constant of model is sometimes difficult and a lot of researches have been done to avoid the calculation of the normalization constant. In this paper, we tackle with the difficulty by combining a technique of empirical localization and a deformed Bregman divergence.The technique of empirical localization makes it possible to drastically reduce computational cost of the calculation of the normalization constant, and in addition, appropriate choice of the deformation for the Bregman divergence can invest the proposed estimator with various kinds of favorable statistical properties, such as efficiency or robustness against outlier noise.
    
[^18]: 流行病学中“忽略设计”与“尊重设计”的世界模型对比

    Design-Ignoring versus Design-Respecting World Models for Epidemiology

    [https://arxiv.org/abs/2609.30679](https://arxiv.org/abs/2609.30679)

    该论文首次将流行病学研究设计形式化为世界模型的约束条件，区分了“尊重设计”与“忽略设计”两类世界模型，并通过大规模整群随机检验阴性试验表明：两者在事实重构上表现相当，但在与预期干预问题相关的干预推断上存在本质差异。

    

    面向流行病学的世界模型是从受研究设计（包括分配、抽样、测量及相关过程）塑造的记录中学习的。因此，模型可能在重构观测轨迹的同时，学习到一种依赖于记录收集方式的干预对比。我们将研究设计形式化为施加在世界模型的潜世界接口、动作机制、观测似然和目标读出上的约束。由此区分出“尊重设计”的模型（编码这些约束）与“忽略设计”的模型（仅拟合所选记录，而未将分配和观测与预期的干预问题关联起来）。我们利用一项大规模整群随机检验阴性试验，在保持潜在结构与拟合设置不变的情况下，将忽略设计的病例计数模型与尊重设计的检验阴性观测模型进行对比。两种模型实现了相当的事实重构能力。然而，在500次配对重抽样……

    arXiv:2609.30679v1 Announce Type: cross  Abstract: World models for epidemiology learn from records shaped by study designs, including assignment, sampling, measurement, and related processes. A model may therefore reconstruct observed trajectories while learning an intervention contrast that depends on how records were collected. We formalize study design as constraints on a world model latent-world interface, action mechanism, observation likelihood, and target readout. This distinguishes design-respecting models, which encode these constraints, from design-ignoring models, which fit selected records without relating assignment and observation to the intended intervention question. Using a large scale cluster-randomized test-negative trial, we hold the latent structure and fitting settings fixed and compare a design-ignoring case-count model with a design-respecting test-negative observation model. Both models achieve comparable factual reconstruction. Yet across 500 paired resamplin
    
[^19]: 交互式智能体中的因果保留：接口因式分解与选择性适应

    Causal Retention in Interactive Agents: Interface Factorization and Selective Adaptation

    [https://arxiv.org/abs/2609.30650](https://arxiv.org/abs/2609.30650)

    本文提出“因果保留”理论，证明冻结的学习状态能否正确响应独立于训练的机制探针取决于学习接口纤维与探针答案纤维之间的包含关系，并据此构建 Causal Core 系统，通过证据门控写入与选择性适应等机制实现无误差的目标更新。

    

    任务性能并不决定智能体保留哪种干预机制。我们研究因果保留：即一个冻结的学习状态能否回答一个独立于训练而固定的机制探针映射，该映射涵盖动作、上下文、直接目标、价值和延迟等维度。对于有限的结构因果模型类，最优探针误差是一个贝叶斯决策风险；当且仅当每个学习接口纤维都落在某一探针答案纤维之内时，该误差恰好消失，且任何通过对该接口进行后处理得到的状态都继承相同的下界。一个后验覆盖定理刻画了预算受限的重测试，而一个精确的编辑分解表明移位集是无误差目标更新的唯一支撑。Causal Core 通过证据门控写入、读出过滤、时间信用分配、隐藏上下文设置和局部诊断更新来实现这些条件。实验涵盖有限因果系统、连续模拟器、官方T……（原文摘要在此处不完整）

    arXiv:2609.30650v1 Announce Type: cross  Abstract: Task performance need not determine which intervention mechanism an agent retains. We study causal retention: whether a frozen learned state answers a mechanism-probe map fixed independently of training, including action, context, direct target, value, and delay. For finite structural causal model classes, the optimal probe error is a Bayes decision risk. It vanishes exactly when every learning-interface fiber lies within one probe-answer fiber; any state obtained by post-processing that interface inherits the same lower bound. A posterior-coverage theorem characterizes budgeted retesting, while an exact edit decomposition shows that the shifted set is the unique support of an error-free target update. Causal Core implements these conditions through evidence-gated writing, readout filtering, temporal credit, hidden-context setup, and local diagnostic updates. Experiments cover finite causal systems, continuous simulators, an official T
    
[^20]: MARCEDES：基于连续优化的非高斯性条件下基于评分的因果发现方法

    MARCEDES: Score-based causal discovery under non-Gaussianity with continuous optimization

    [https://arxiv.org/abs/2609.30643](https://arxiv.org/abs/2609.30643)

    提出了名为MARCEDES的基于评分的因果发现方法，通过引入平均绝对残差风险、行稀疏惩罚和软DAG约束，实现了非高斯误差下因果DAG结构的高效连续优化学习。

    

    我们考虑学习与非高斯误差的结构方程模型（SEM）相对应的潜在因果有向无环图（DAG）结构的问题。受一个故意误设的、所有误差均为拉普拉斯分布的非高斯SEM的启发，我们首先引入了定义在所有实矩阵空间上的平均绝对残差风险，并证明在渐近意义上，真实加权因果DAG矩阵的风险严格小于任何其他矩阵的风险。然而，为了增强通用性并考虑高维和有限样本的设置，我们进一步引入了针对每一行的稀疏性惩罚以及软DAG约束，从而在实矩阵空间上推导出一个连续的评分函数。据此，我们提出了一种基于评分的DAG学习方法，命名为MARCEDES，将其表述为一个无约束的评分最小化问题，该问题可以通过基于梯度的优化技术高效求解。

    arXiv:2609.30643v1 Announce Type: new  Abstract: We consider the problem of learning the underlying causal directed acyclic graph (DAG) structure corresponding to a structural equation model (SEM) with non-Gaussian errors. Motivated by an intentionally misspecified non-Gaussian SEM with all Laplace errors, we first introduce the mean absolute residual risk, defined over the space of all real matrices, and show that, asymptotically, the risk of the true weighted causal DAG matrix is strictly smaller than that of any other matrix. Nevertheless, to enhance generality and account for high-dimensional and finite-sample settings, we further incorporate row-specific sparsity penalties along with a soft DAG constraint to derive a continuous score function over the space of real matrices. Accordingly, we propose a score-based DAG learning method, named MARCEDES, formulated as an unconstrained score minimization problem, which can be efficiently solved using gradient-based optimization technique
    
[^21]: 熵正则化：一种针对可验证示范的交叉熵免费修正

    Entropy Regularization: A Free Correction to Cross-Entropy for Verified Demonstrations

    [https://arxiv.org/abs/2609.30572](https://arxiv.org/abs/2609.30572)

    该论文指出在存在多个正确解的可验证任务中，用交叉熵模仿单一专家示范可能与验证器风险目标不一致，并提出以熵正则化作为“免费”修正来控制策略支持集、防止概率质量流向错误输出。

    

    大语言模型通常使用交叉熵（CE）在专家示范上进行后训练，即使其下游目标并非模仿所展示的解法，而是生成任何能被验证器接受的输出。这种错位出现在存在多个正确解法的可验证领域中，例如数学推理和代码生成，此时训练数据中每个问题可能只包含一个专家解法。我们证明，最小化交叉熵可能与最小化验证器风险不一致；两个策略可以对观测到的示范赋予相同的似然，同时在错误输出上分配不同的概率质量。我们通过一个学习理论反例将这一点形式化，在该反例中，交叉熵最小化会选择次优策略。我们发现，通过控制所学策略的支持集（support）可以解决这一问题，即阻止概率质量扩散到不受支持的输出上。由于支持集大小是非……（摘要在此处截断）

    arXiv:2609.30572v1 Announce Type: cross  Abstract: Large language models are often post-trained on expert demonstrations using cross-entropy (CE), even when the downstream objective is not to imitate the demonstrated solution but to produce any output accepted by a verifier. This mismatch is seen in verifiable domains with multiple correct solutions, such as mathematical reasoning and code generation, where training data may contain only one expert solution per problem. We show that minimizing cross-entropy can be misaligned with minimizing verifier risk; two policies can assign identical likelihood to the observed demonstrations while placing different probability mass on incorrect outputs. This is formalized through a learning-theoretic counterexample in which CE minimization selects a suboptimal policy. We identify that controlling the support of the learned policy can solve this problem by preventing probability mass from spreading to unsupported outputs. Since support size is non-
    
[^22]: Atelier：通过超网络学习冷冻电镜体积的局部自监督特征

    Atelier: Learning Local Self-Supervised Features for CryoEM Volumes via Hypernetworks

    [https://arxiv.org/abs/2609.30569](https://arxiv.org/abs/2609.30569)

    Atelier是一个基于Transformer超网络的自监督框架，通过摊销冷冻电镜图谱的隐式神经表示拟合，实现了高效、跨样本对齐的局部特征学习，可用于大规模冷冻电镜体积的特征提取。

    

    冷冻电镜图谱解读需要空间局部化、跨样本一致且在不同空间尺度上均具信息量的特征。大多数用于图谱注释的深度学习方法从固定的体素网格中提取特征。然而，隐式神经表示（INR）能够将体积数据建模为与尺度无关、以坐标为条件的函数，因此对冷冻电镜很有吸引力；但为每个图谱单独拟合一个INR对于大规模特征提取而言成本过高，且所产生的表征无法在样本之间对齐。我们提出了Atelier，这是一个自监督框架，通过对重建的冷冻电镜图谱进行摊销式的INR拟合来解决这一问题。Atelier在电子显微镜数据库（EMDB）的5,439个图谱上进行了预训练，是一个基于Transformer的超网络，能够在广泛的蛋白质结构（包括大型多亚基组装体）上生成高保真的重建。除重建之外，该预训练生成的INR……（原文摘要在此处截断）

    arXiv:2609.30569v1 Announce Type: new  Abstract: CryoEM map interpretation requires features that are spatially localized, consistent across samples, and informative across spatial scales. Most deep learning methods for map annotation extract features from fixed voxel grids. However, implicit neural representations (INRs) are able to model volumetric data as scale-agnostic, coordinate-conditioned functions. INRs are therefore attractive for cryoEM, but fitting a separate INR for each map is too expensive for large-scale feature extraction and produces representations that are not aligned across samples. We introduce Atelier, a self-supervised framework that amortizes INR fitting for reconstructed cryoEM maps. Pretrained on 5,439 Electron Microscopy Data Bank maps, Atelier is a transformer-based hypernetwork that generates high-fidelity reconstructions across a wide range of protein structures, including large multi-subunit assemblies. Beyond reconstruction, the INR generated by the pre
    
[^23]: 面向长记忆时间序列的高维高斯图模型检验

    High-dimensional Gaussian Graphical Model Testing for Long-Memory Time Series

    [https://arxiv.org/abs/2609.30565](https://arxiv.org/abs/2609.30565)

    提出了一种适用于长记忆时间序列的数据自适应高斯图模型条件独立性检验方法，建立了有限样本高斯逼近界并通过块自助法保证超高维情形下的有效性，检验在水平和功效上均具渐近一致性，可应用于fMRI功能连接性分析。

    

    许多现实世界中的高维时间序列表现出长记忆性，但针对这一情形的高斯图模型检验研究仍然不足。我们开发了一种直接的、数据自适应的检验统计量，用于评估平稳高斯时间序列图结构中的条件独立性。我们为该统计量建立了有限样本的Berry--Esseen型高斯逼近界，该结果同时适用于短记忆和长记忆时间序列。该检验程序通过块自助法实现完全数据自适应，我们为其提供了包括超高维情形在内的有限样本有效性结果，且该方法可以扩展到两样本检验中以比较不同的图结构。我们还对该统计量开发了一种一致性增强的修正方法，并证明此类检验在检验水平和功效两方面均达到渐近一致性。我们将所提出的方法应用于真实的fMRI数据，以理解功能连接性……

    arXiv:2609.30565v1 Announce Type: cross  Abstract: Many real-world high-dimensional time series exhibit long-memory, but Gaussian graphical model testing in this regime remains understudied. We develop a direct, data-adaptive test statistic for assessing conditional independence in the graph structure of stationary Gaussian time series. We establish a finite-sample, Berry--Esseen type Gaussian approximation bound for the statistic, which applies to both short-memory and long-memory time series. The testing procedure is fully data-adaptive using block bootstrap method, on which we provide a finite-sample validity result including in the ultra-high-dimensional scenario, and can be extended to comparing graphical structures in two-sample tests. We also develop a consistency-empowered correction to the statistic and show that such tests attain asymptotic consistency in both size and power. Our proposed method is applied to a real-world fMRI data to understand functional connectivities with
    
[^24]: 通过深度展开学习替代Split-Gibbs扩散后验采样中的MCMC方法

    Learning to Replace MCMC in Split-Gibbs Diffusion Posterior Sampling via Deep Unfolding

    [https://arxiv.org/abs/2609.30539](https://arxiv.org/abs/2609.30539)

    本文提出一种基于深度展开的学习框架，将split-Gibbs扩散后验采样中的Gibbs更新重新表述为高斯去噪问题并通过ODE扩散实现，从而以更低的似然更新计算成本替代了传统MCMC迭代。

    

    Split Gibbs采样通过解耦先验与似然的计算，实现了针对一般非线性逆问题的扩散后验推断，并使预训练的扩散先验可在不同测量模型之间复用。然而，其似然更新通常依赖迭代式MCMC，这不仅阻碍并行化，还需要针对特定算法的调参，并带来高昂的计算成本。在本工作中，我们提出一种基于学习的框架来替代这一MCMC步骤，其方法是将两个Gibbs更新重新表述为高斯去噪问题，并通过ODE扩散加以实现。先验步骤复用预训练的去噪器，而似然去噪器则通过轻量级的深度展开网络利用已知的似然结构。在非线性相位恢复任务上的实验表明，所提方法能以更低的似然更新成本有效替代基于MCMC的split Gibbs方法。

    arXiv:2609.30539v1 Announce Type: cross  Abstract: Split Gibbs sampling enables diffusion posterior inference for general nonlinear inverse problems by decoupling prior and likelihood computations, allowing a pretrained diffusion prior to be reused across measurement models. However, its likelihood update often relies on iterative MCMC, which can hinder parallelization, require algorithm-specific tuning, and incur substantial computational cost. In this work, we propose a learning-based framework to replace this MCMC step by reformulating both Gibbs updates as Gaussian denoising problems and implementing them through ODE diffusion. The prior step reuses a pretrained denoiser, while the likelihood denoiser exploits known likelihood structure through a lightweight deep-unfolded network. Experiments on nonlinear phase retrieval demonstrate the effectiveness of the proposed method as an alternative to MCMC-based split Gibbs at lower likelihood-update cost.
    
[^25]: 距离依赖矩条件下的普通非凸SGD：有限时域平稳性与Nagaev界

    Ordinary Nonconvex SGD under Distance-Dependent Moments: Finite-Horizon Stationarity and Nagaev Bounds

    [https://arxiv.org/abs/2609.30499](https://arxiv.org/abs/2609.30499)

    本文证明，当条件矩允许噪声方差随迭代点距离增长时，普通单样本SGD无需任何修改即可达到与Blum–Gladyshev下界匹配的极小极大随机复杂度，并借助希尔伯特空间Fuk–Nagaev不等式给出高概率Nagaev型界。

    

    统一的噪声矩界假设排除了那些变异性随迭代点位置增长的随机梯度。我们在距离依赖的条件矩假设下，研究针对光滑、下有界且可能非凸目标的普通单样本随机梯度下降。仅利用二阶矩条件，一个直接的“下降—位移”论证在使用依赖时域的步长时，给出了 $T^{-1/3}$ 的期望平均平方梯度平稳性。一个显式的预言机复杂度推论与已知的平滑Blum–Gladyshev（BG-0）下界相匹配，包括 $Lb_2\Delta^3\varepsilon^{-6}$ 和 $L\Delta\sigma^2\varepsilon^{-4}$ 两个随机项，其中 $\Delta$ 为初始目标间隙，$\sigma^2+b_2\|x-x_1\|^2$ 为方差的上界。因此，无需任何修改的SGD在这一二阶矩类别中即达到极小极大随机复杂度。对于 $p>2$，可预测局部化技术与希尔伯特空间上的Fuk–Nagaev不等式给出了一个高概率界，分离对数……（摘要在此处被截断）

    arXiv:2609.30499v1 Announce Type: new  Abstract: Uniform noise-moment bounds exclude stochastic gradients whose variability increases with the iterate. We study ordinary, single-sample stochastic gradient descent for smooth, lower-bounded, possibly nonconvex objectives under distance-dependent conditional moments. Under second moments alone, a direct descent--displacement argument yields $T^{-1/3}$ expected average squared-gradient stationarity with a horizon-dependent stepsize. An explicit oracle-complexity corollary matches the known smooth Blum--Gladyshev (BG-0) lower bound, including the $Lb_2\Delta^3\varepsilon^{-6}$ and $L\Delta\sigma^2\varepsilon^{-4}$ stochastic terms, where $\Delta$ is the initial objective gap and $\sigma^2+b_2\|x-x_1\|^2$ bounds the variance. Thus unchanged SGD attains the minimax stochastic complexity in this second-moment class. For $p>2$, predictable localization and a Hilbert-space Fuk--Nagaev inequality yield a high-probability bound separating logarith
    
[^26]: 学习偏置：机器学习增强的粒子滤波器

    Learning to Bias: Machine Learning-Enhanced Particle Filters

    [https://arxiv.org/abs/2609.30498](https://arxiv.org/abs/2609.30498)

    提出神经最优粒子滤波器（NOPF），通过从离线模拟数据中学习最优提议分布的摊销近似，并将其作为即插即用模块嵌入标准粒子滤波器，借助重要性权重校正保证滤波分布的一致性，从而提升序贯推断的样本效率与高维扩展能力。

    

    序贯推断旨在从含噪且不完整的观测中估计潜在状态。粒子滤波器（PFs）是一类基于重要性采样的蒙特卡洛方法，为该任务提供了灵活的框架，但其样本效率往往较低，且随维度增长扩展性不佳，部分原因在于提议分布不够理想。我们通过将学习到的提议分布集成到粒子滤波框架中来应对这些挑战。我们提出了神经最优粒子滤波器（NOPFs），它从离线模拟的单步条件数据元组中学习最优提议分布的摊销近似。学习到的提议分布可作为即插即用模块直接替换标准粒子滤波更新中的原有提议，同时样本通过标准重要性权重进行校正，因此在标准的支撑集与密度可评估假设下，该方法渐近地收敛于相同的滤波分布。在推断复杂度各异的随机非线性基准测试中，NOPFs（此处原文截断）……

    arXiv:2609.30498v1 Announce Type: cross  Abstract: Sequential inference estimates latent states from noisy and incomplete observations. Particle Filters (PFs), a class of Monte Carlo methods based on importance sampling, provide a flexible framework for this task, but often suffer from poor sample efficiency and unfavorable scaling with dimension, partly due to suboptimal proposal distributions. We address these challenges by integrating learned proposals into the PF framework. We introduce Neural Optimal Particle Filters (NOPFs), which learn an amortized approximation to the optimal proposal from offline simulated one-step conditioning tuples. The learned proposal is used as a drop-in replacement in standard PF updates, with samples corrected by standard importance weights so that the method asymptotically targets the same filtering distribution under standard support and density-evaluation assumptions. Across stochastic nonlinear benchmarks of varying inference complexity, NOPFs impr
    
[^27]: 打破同质性：多样化人格集合以实现创造性的LLM输出

    Breaking Homogeneity: Diversifying Persona Sets for Creative LLM Outputs

    [https://arxiv.org/abs/2609.30492](https://arxiv.org/abs/2609.30492)

    该论文将人格多样化建模为集合级条件化问题，在“选择vs生成”与“空间填充vs前沿探索”两个正交设计维度上提出四种方法，显著提升了语言模型创造性输出的多样性，其中进化式人格生成在AUT上将回答多样性提高了78.8%。

    

    语言模型在开放式任务中常常产生同质化的回答；这种同质性可能引发群体思维——即观点向单一且可能次优的决策趋同。我们将人格多样化表述为一个集合级条件化问题，并研究了两个正交的设计选择：是选择人格还是生成人格，以及是空间填充式多样性还是前沿探索式多样性。我们用四种方法实例化了这一设计空间，涵盖覆盖性与分散性子集选择、均匀覆盖采样以及进化式人格生成。在替代用途任务（AUT）、Infinity-Chat 和发散联想任务（DAT）上的评估表明，所提出的方法在各种任务和创造力目标上均展现出优势。在AUT上，与仅使用任务提示相比，进化式人格生成将回答多样性提高了78.8%，原创性提高了26.1%，灵活性提高了49.5%，整体创造力提高了13.9%，同时保持了98.5……

    arXiv:2609.30492v1 Announce Type: cross  Abstract: Language models often produce homogeneous responses to open-ended tasks; such homogeneity can spawn groupthink-the convergence of ideas toward a singular and potentially suboptimal decision. We formulate persona diversification as a set-level conditioning problem and study two orthogonal design choices: selecting versus generating personas, and space-filling versus frontier-seeking diversity. We instantiate this design space with four methods spanning coverage and dispersion subset selections, uniform-coverage sampling, and evolutionary persona generation. Evaluations on the Alternative Uses Task (AUT), Infinity-Chat, and Divergent Association Task (DAT) show the benefits of the proposed methods across tasks and creativity objectives. On AUT, evolutionary persona generation increases response diversity by 78.8%, originality by 26.1%, flexibility by 49.5%, and holistic creativity by 13.9% over task-only prompting, while maintaining 98.5
    
[^28]: 对称正定流形上函数值数据的几何特征学习

    Geometric Feature Learning for Functional Data Valued on the Symmetric Positive Definite Manifold

    [https://arxiv.org/abs/2609.30487](https://arxiv.org/abs/2609.30487)

    该论文提出了MatFAE——一种用于学习对称正定（SPD）矩阵黎曼流形上轨迹的函数神经网络，它将序列视为连续函数以编码轨迹动力学特性，并通过函数权重的形态提供可解释性。

    

    我们在此提出了一种函数神经网络，称为MatFAE，用于学习对称正定（SPD）矩阵黎曼流形上的轨迹。MatFAE的特点是包含内在层，这些内在层将流形值函数映射为欧几里得向量值函数，随后通过一个函数层将其投影到有限维欧几里得空间。与大多数针对离散时间序列的神经网络不同，MatFAE将每个序列视为连续函数，因此能够在潜在表示中编码轨迹的动力学特性（例如一阶导数）。此外，函数层中函数权重的形态通过揭示输入函数数据中对潜在表示贡献最大的区域，提供了可解释性。我们论证了每个内在层的设计原则和性质，并详细说明了在反向传播过程中如何处理矩阵分解。我们将MatFAE应用于……（原文摘要在此处截断）

    arXiv:2609.30487v1 Announce Type: cross  Abstract: We here develop a functional neural network, termed MatFAE, for learning trajectories on the Riemannian manifold of symmetric positive definite (SPD) matrices. MatFAE features intrinsic layers that map manifold-valued functions to Euclidean vector-valued functions, followed by a functional layer that projects them into a finite-dimensional Euclidean space. Unlike most neural networks for discrete-time sequences, MatFAE treats each sequence as a continuous function and can therefore encode trajectory dynamics (e.g., first-order derivatives) in its latent representations. Additionally, the morphology of the functional weights in the functional layer offers interpretability by revealing the regions of the input functional data that contribute most to the latent representations. We justify the design principles and properties of each intrinsic layer and detail how matrix factorization is handled during backpropagation. We apply MatFAE to a
    
[^29]: 基于仿真推断的fMRI功能连接贝叶斯不确定性量化

    Bayesian Uncertainty Quantification for fMRI Functional Connectivity via Simulation-Based Inference

    [https://arxiv.org/abs/2609.30445](https://arxiv.org/abs/2609.30445)

    该论文提出一个基于仿真推断的贝叶斯框架，通过将BOLD动力学建模为耦合Ornstein-Uhlenbeck过程并使用序贯神经后验估计，量化了fMRI功能连接估计中来自扫描仪噪声、被试变异性和采集时长三方面来源的不确定性。

    

    优化fMRI扫描时长和空间分辨率对实验设计至关重要，然而传统的基于相关的方法无法量化不确定性，也无法将扫描仪测量噪声与被试间真实的神经变异性区分开来。在缺乏有原则的不确定性界的情况下，研究人员无法判断扫描方案是否足够长以可靠地估计功能连接，也无法确定被试间差异反映的是生物学变异还是噪声。我们提出了一个贝叶斯框架，将BOLD信号动力学建模为耦合的Ornstein-Uhlenbeck过程，并使用序贯神经后验估计来获得功能连接的后验分布，同时考虑BOLD频谱中与频率无关的测量噪声。该框架应用于7T场强下28名健康对照被试（55次扫描）的数据，并使用功能网络图谱（65个默认模式网络脑区），量化了来自不同来源的不确定性：扫描仪噪声、被试变异性和采集时长。（摘要原文在此处截断）

    arXiv:2609.30445v1 Announce Type: new  Abstract: Optimizing fMRI scan duration and spatial resolution is critical for experimental design, yet traditional correlation-based approaches cannot quantify uncertainty or disentangle scanner measurement noise from true neural variability across subjects. Without principled uncertainty bounds, researchers cannot know whether a protocol is long enough to reliably estimate connectivity, or whether between-subject differences reflect biological variation or noise. We present a Bayesian framework modeling BOLD dynamics as coupled Ornstein-Uhlenbeck processes, using Sequential Neural Posterior Estimation to obtain connectivity posteriors while accounting for frequency-independent measurement noise across the BOLD spectrum. Applied to N = 28 healthy controls (55 scans) at 7T using a functional network atlas (65 DMN regions), the framework quantifies uncertainty across its sources: scanner noise, subject variability, and acquisition length. Spatial a
    
[^30]: 面向连续处理的因果机器学习端到端流水线：在金融决策中的应用

    An End-to-End Pipeline for Causal ML with Continuous Treatments: An Application to Financial Decision Making

    [https://arxiv.org/abs/2609.30396](https://arxiv.org/abs/2609.30396)

    该论文提出了一个面向连续处理场景的端到端因果机器学习流水线，创新性地解决了正值性违反检测、高维数据降维以及敏感性分析与估计方法向连续处理空间的适配问题，并成功应用于金融决策。

    

    本文提出了一个专为连续处理的真实世界应用设计的端到端因果机器学习流水线。该框架包含六个连续步骤：降维、因果识别、正值性假设违反处理、估计、反驳与评估，以及策略优化。我们引入了现有因果机器学习工具包中尚不具备的实用贡献，具体包括：(1) 一种在连续处理设置中检测和量化正值性违反的方法；(2) 一种新颖的、可扩展的两阶段降维框架，专为高维数据的因果推断量身定制；(3) 将原本为二值处理设计的敏感性分析和估计方法适配到连续处理空间；(4) 将这些组件端到端集成到一个模块化、可复现的工作流中。这些创新解决了现实世界的实际问题。

    arXiv:2609.30396v1 Announce Type: cross  Abstract: This paper presents an end-to-end causal machine learning (ML) pipeline designed for real-world applications with continuous treatments. The proposed framework consists of six sequential steps: dimensionality reduction, causal identification, positivity assumption violation handling, estimation, refutation and evaluation, and policy optimization. We introduce practical contributions not currently available in existing causal ML toolkits, specifically: (1) a method for detecting and quantifying positivity violations in continuous treatment settings (2) a novel, scalable two-stage dimensionality reduction framework tailored for causal inference with high-dimensional data; (3) the adaptation of sensitivity analysis and estimation methods originally designed for binary treatments to the continuous treatment space and (4) an end-to-end integration of these components into a modular, reproducible workflow. These innovations address real-worl
    
[^31]: 基于对决反馈的成本感知最优大语言模型识别

    Cost-Aware Best-LLM Identification using Dueling Feedback

    [https://arxiv.org/abs/2609.30360](https://arxiv.org/abs/2609.30360)

    该论文提出了一种结合对决反馈与异质查询成本的成本感知多臂老虎机算法，用于在给定置信度下识别最佳大语言模型，并证明了其渐近最优成本性能。

    

    受从一组具有异质查询成本的大语言模型（LLM）中识别最佳模型这一问题的启发，我们提出并分析了一种多臂老虎机（MAB）的变体，该变体具有两个特点：（i）对决反馈，即通过对模型响应之间的成对比较来提供稳健的偏好信号；（ii）异质采样成本，反映了查询不同LLM所需的不同成本。在假设存在孔多塞赢家（Condorcet winner）的前提下（我们在多个真实世界数据集上对该条件进行了实证验证），我们提出了一种Track-and-Stop风格的算法，用于在给定置信水平下的最优臂识别。我们证明了随着误差趋于零，该算法几乎必然地实现渐近最优成本。最后，我们在合成数据和真实世界实例上对该方法进行了广泛评估，结果表明其相较于经典的无成本感知算法及其成本感知扩展版本均取得了一致的改进。

    arXiv:2609.30360v1 Announce Type: cross  Abstract: Inspired by the problem of identifying the best model from a collection of large language models (LLMs) with heterogeneous querying costs, we formulate and analyse a variant of the multi-armed bandit (MAB) with (i) dueling feedback, where pairwise comparisons between model responses provide robust preference signals, and (ii) heterogeneous sampling costs, reflecting the differing costs of querying different LLMs. Assuming the existence of a Condorcet winner, a condition we empirically validate across multiple real-world datasets, we propose a Track-and-Stop style algorithm for best-arm identification with prescribed confidence. We prove that the algorithm almost surely achieves the asymptotically optimal cost as the error tends to zero. Finally, we extensively evaluate our approach on both synthetic and real-world instances, demonstrating consistent improvements over classical cost-unaware algorithms and their cost-aware extensions.
    
[^32]: 策略性自洽性

    Strategic Self-Consistency

    [https://arxiv.org/abs/2609.30352](https://arxiv.org/abs/2609.30352)

    本文揭示了一种针对自洽性推理服务的潜在欺诈行为：不忠实的模型提供商可通过策略性地生成并重排额外的推理路径，使每条路径在多数投票中看起来都不可或缺，从而在避开审计检测的情况下人为增加路径数量以向用户多收费。

    

    自洽性（Self-consistency）已成为一种流行的技术，通过生成多条推理路径并通过多数投票选出最终答案来增强大型语言模型的推理能力。然而，由于模型提供商通常按照生成的推理路径数量向用户收费，他们存在人为增加路径数量的经济动机。在这项工作中，我们证明了一个不忠实的提供商可以利用这一动机，使用一种简单高效的算法同时避免被审计者检测：该算法通过生成并策略性地重新排序额外的推理路径，使得每条路径看起来都是达到多数所必需的。为了验证我们的算法，我们在涵盖数学、科学和问答任务的基准数据集上，使用来自Llama和Qwen系列的多个指令模型，以及从DeepSeek-R1蒸馏而来的推理模型进行了实验。我们的结果表……（摘要截断）

    arXiv:2609.30352v1 Announce Type: cross  Abstract: Self-consistency has become a popular technique for enhancing the reasoning abilities of large language models by generating multiple reasoning paths and selecting the final answer through a majority vote. However, because model providers typically charge users in proportion to the number of reasoning paths generated, they have a financial incentive to artificially increase the path count. In this work, we show that an unfaithful provider can exploit this incentive using a simple, efficient algorithm while avoiding detection by an auditor: by generating and strategically reordering additional reasoning paths, the algorithm makes every path appear necessary to reach the majority. To validate our algorithm, we conduct experiments with multiple instruct models from the Llama and Qwen families, as well as reasoning models distilled from DeepSeek-R1, on benchmark datasets spanning mathematics, science, and question answering. Our results su
    
[^33]: 自适应多分辨率高斯过程：基于天然数据稀疏协方差矩阵的可扩展精确推断

    Adaptive multi-resolution Gaussian processes: Scalable exact inference with naturally data-sparse covariance matrices

    [https://arxiv.org/abs/2609.30348](https://arxiv.org/abs/2609.30348)

    该论文提出一种自适应多分辨率高斯过程框架，通过直接锚定样本点的自适应多分辨率基函数构建天然数据稀疏的协方差矩阵，并结合稀疏Cholesky逆算法，实现了既可扩展又精确的高斯过程推断。

    

    高斯过程是概率机器学习的基石，然而将其扩展到大规模数据集通常需要在计算效率和模型保真度之间进行权衡。本工作通过提出一个既可扩展又精确的自适应多分辨率高斯过程框架来弥合这一差距。我们的关键创新是利用自适应多分辨率基函数构建天然数据稀疏的协方差矩阵。这些基函数直接锚定在样本上，从而无需辅助点。通过缩小多分辨率基的支撑域，矩阵块的大小受到限制，进而保证了稀疏性。数据稀疏协方差矩阵的逆可通过稀疏Cholesky逆算法被精确且高效地计算。为进一步提升预测不确定性估计的质量，我们构造了增广基函数。理论分析与数值实验表明……

    arXiv:2609.30348v1 Announce Type: cross  Abstract: Gaussian processes constitute a cornerstone of probabilistic machine learning, yet scaling them to large datasets typically forces a trade-off between computational efficiency and model fidelity. This work bridges this gap by presenting an adaptive multi-resolution Gaussian process framework that is both scalable and exact. Our key innovation is constructing a naturally data-sparse covariance matrix with adaptive multi-resolution basis functions. These basis functions are directly anchored to samples, eliminating the need for auxiliary points. By shrinking the support domains of multi-resolution basis, the matrix block sizes are limited, guaranteeing sparsity. The inverse of the data-sparse covariance matrix is computed exactly and efficiently via the sparse Cholesky inverse algorithm. To further improve predictive uncertainties, we construct an augmented basis function. Theoretical analysis and numerical experiments demonstrate that o
    
[^34]: 面向内存高效Transformer预训练的低秩摩擦

    Low-Rank Friction for Memory-Efficient Transformer Pretraining

    [https://arxiv.org/abs/2609.30342](https://arxiv.org/abs/2609.30342)

    提出R-iKFAD优化器，通过秩1外积分解替代完整摩擦张量，将优化器状态内存近乎减半，同时保持与iKFAD相当的性能和超参数鲁棒性。

    

    iKFAD是最近提出的一种优化器，它在动量动力学中用自适应摩擦取代了自适应学习率，同时性能与Adam相当。其局限在于完整的摩擦张量 ξ∈R^{m×n} 与Adam的二阶矩缓冲区一样，在每层都带来 O(mn) 的内存开销。本文用由行、列动量统计量构建的秩1外积分解来替换iKFAD的摩擦张量 ξ，得到了Rank-1 iKFAD（R-iKFAD）。这将每层的摩擦内存占用从 O(mn) 降低到 O(m+n)，使iKFAD的总优化器状态大小大约减半。尽管内存大幅缩减，R-iKFAD仍保持了与iKFAD相当的性能：在GPT2-Nano、TinyViT、DistilBERT和GPT2-S上的实验证实，它在将内存占用近乎减半的同时，性能持平甚至超过iKFAD，并且对超参数具有相当的鲁棒性。（注：原文摘要在此处被截断）

    arXiv:2609.30342v1 Announce Type: new  Abstract: iKFAD is a recently proposed optimiser that replaces adaptive learning rates with adaptive friction in the momentum dynamics, yet performs as well as Adam. Its limitation is that the full friction tensor $\xi\in\mathbb{R}^{m\times n}$ carries the same $\mathcal{O}(mn)$ memory overhead per layer as Adam's second-moment buffer. Here we replace iKFAD's friction tensor $\xi$ with a rank-1 outer-product factorisation built from row and column momentum statistics, resulting in Rank-1 iKFAD (R-iKFAD). This reduces the friction memory footprint from $\mathcal{O}(mn)$ to $\mathcal{O}(m+n)$ per layer, which approximately halves iKFAD's total optimiser state. Despite this reduction, R-iKFAD maintains parity in performance with iKFAD: experiments on GPT2-Nano, TinyViT, DistilBERT and GPT2-S confirm that it matches or exceeds iKFAD while nearly halving the memory footprint and remaining comparably robust to hyperparameters. We analyse the continuous-
    
[^35]: 高斯老虎机中的自适应随机矩阵：谱普适性与选择诱导的离群值

    Adaptive Random Matrices in Gaussian Bandits: Spectral Universality and Selection-Induced Outliers

    [https://arxiv.org/abs/2609.30321](https://arxiv.org/abs/2609.30321)

    该论文证明当臂数量的对数相对维度次线性时，高斯老虎机中任意自适应选择规则都不改变观测矩阵经验谱收敛于Marchenko-Pastur定律的极限行为，从而使贝叶斯后验不确定度和信息获取具有与策略无关的一阶极限，并在线性得分选择下给出精确的条件臂分布与Wishart型Gram矩阵刻画。

    

    自适应的臂选择会改变老虎机算法所收集观测值的分布，但未必改变其极限经验谱。我们研究了维度与观测数量成比例增长的高斯老虎机设计。我们提出一个定量耦合定理，将任意因果选择规则所生成的设计与独立高斯设计进行比较。若可用臂数量的对数相对于维度是次线性的，则经验谱分布收敛于Marchenko-Pastur定律，且该收敛在所有选择规则上一致成立。由此可知，高斯贝叶斯老虎机在后验均方不确定度、后验协方差平方以及信息获取方面具有与策略无关的一阶极限。对于线性得分选择规则，我们得到了精确的条件臂分布，并证明双臂选择在每一维度上都产生精确的Wishart型Gram矩阵，尽管其本身非零……

    arXiv:2609.30321v1 Announce Type: new  Abstract: Adaptive arm selection changes the distribution of the observations collected by a bandit algorithm, but it need not change their limiting empirical spectrum. We study Gaussian bandit designs in which the dimension and the number of observations grow proportionally. A quantitative coupling theorem compares the design generated by any causal selection rule with an independent Gaussian design. If the logarithm of the number of available arms is sublinear in the dimension, the empirical spectral distribution converges to the Marchenko-Pastur law, uniformly over the selection rule. Consequently, Gaussian Bayesian bandits have policy-independent first-order limits for posterior mean-square uncertainty, squared posterior covariance, and information acquisition. For linear-score selection, we obtain the exact conditional arm distribution and show that two-arm selection produces an exactly Wishart Gram matrix in every dimension, despite its nonz
    
[^36]: 耗散随机动力系统在 $\mathbb{R}^d$ 上的击中时间分布及其在随机梯度下降中的应用

    Distribution of hitting times for dissipative random dynamical systems on $\mathbb{R}^d$, with application to stochastic gradient descent

    [https://arxiv.org/abs/2609.30274](https://arxiv.org/abs/2609.30274)

    该论文从遍历理论的视角提出了一种新方法，用于研究一大类随机优化方法的渐近性质，其主要贡献是给出了随机优化算法到达极小值点小邻域的击中时间分布分析，并将其应用于随机梯度下降的收敛性研究。

    

    机器学习，特别是深度学习，涉及求解大规模非凸优化问题。文献中已提出多种算法，这些算法对于困难问题实例似乎能够取得令人满意的实际效率，其中随机梯度方法是最基础的，但在许多学习任务上仍然优于更晚近的算法。关于深度学习中现有方法的一个主要未解问题是理解它们的收敛性质。沿着关于梯度类算法长时间行为的一系列先前研究工作，我们从遍历理论的角度提出了一种新方法，用于研究一大类优化方法的渐近性质。我们的主要结果包括对随机优化算法到达某个极小值点的特定小邻域所需期望时间的研究，以及……

    arXiv:2609.30274v1 Announce Type: new  Abstract: Machine Learning and more specifically Deep Learning involves solving large scale nonconvex optimization problems. Several algorithms have been proposed in the literature, that seem to achieve satisfactory practical efficiency for difficult instances, the Stochastic Gradient Method being the most rudimentary, while still outperforming more recent algorithms at a number of learning tasks.   A major open question about the current methods used in deep learning is to understand their convergence properties. Following a line of previous works about the long time behavior of gradient-type algorithms, %and in particular the recent contributions from Azizian et al., we present a new approach for studying the asymptotic properties of a wide family of methods from an ergodic theoretical viewpoint. Our main results include a study of the expected time for a stochastic optimisation algorithm to reach a certain small neighborhood of a minimizer and 
    
[^37]: 可扩展的最小体积单纯形估计与非渐近分析

    Scalable Minimum-Volume Simplex Estimation with Non-asymptotic Analysis

    [https://arxiv.org/abs/2609.25576](https://arxiv.org/abs/2609.25576)

    提出 DeepMVSA 方法，通过神经隐式形式（轻量坐标网络加 LU 三角参数化）将最小体积单纯形估计的内存降至与样本量无关的 O(K^2)、单次遍历成本降至 O(NK^2)，并给出非渐近样本复杂度界与神谕不等式等理论保证。

    

    我们研究从 N 个独立同分布、均匀采样自其内部的点中估计一个 K 维单纯形的问题；观测数据是 K+1 个未知原型的凸组合。现有的多项式时间估计器需要每样本立方级的计算量或 O(NK) 的存储空间，在 N 约为 10^6 至 10^8 的规模下不可行。我们提出 DeepMVSA，以神经隐式形式重新表述最小体积原理：一个轻量级坐标网络生成混合权重，一个三角 LU 型参数化表示对偶单纯形矩阵，从而将可训练状态的内存降至与 N 无关的 O(K^2)，并将每次数据遍历的成本降至 O(NK^2)。我们为局部化代理估计器证明了达到多项式时间基准阶数的非渐近样本复杂度界；为神经目标的每个全局最小值点证明了神谕不等式，包含体积膨胀控制和显式收缩偏差；以及一个条件性的端到端误差预算分离……

    arXiv:2609.25576v1 Announce Type: cross  Abstract: We study the estimation of a $K$-dimensional simplex from $N$ i.i.d.\ points sampled uniformly from its interior; the observations are convex combinations of $K+1$ unknown prototypes. Existing polynomial-time estimators need cubic per-sample work or $O(NK)$ storage and are impractical at $N\sim 10^6$--$10^8$. We propose DeepMVSA, which re-expresses the minimum-volume principle in neural implicit form: a lightweight coordinate network generates the mixing weights and a triangular LU-type parameterization the dual simplex matrix, reducing the trainable-state memory to $O(K^2)$, independent of $N$, and the cost per data pass to $O(NK^2)$. We prove a non-asymptotic sample-complexity bound of the polynomial-time benchmark order for a localized surrogate estimator; an oracle inequality for every global minimizer of the neural objective, with volume-inflation control and an explicit shrinkage bias; a conditional end-to-end error budget separa
    
[^38]: 超越二次损失：Adam优化器的稳定性相图

    Beyond Quadratic Loss: The Stability Phase Diagram of Adam

    [https://arxiv.org/abs/2609.18314](https://arxiv.org/abs/2609.18314)

    该研究通过绘制Adam优化器在$(\beta_1,\beta_2)$参数平面上的稳定性相图，发现一条近似线性边界$1-\beta_2=C(1-\beta_1)$可用于区分训练中是否出现损失尖峰，并揭示超二次损失景观（如高置信交叉熵损失形成的“核心-墙壁”结构）是决定该边界形状的关键因素。

    

    损失尖峰是神经网络训练中反复出现的不稳定性现象，可能由多种机制引起。特别是对于Adam优化器，宏观损失尖峰已被认为与优化器动力学相关，但其两个动量时间尺度如何支配这些尖峰仍不清楚。我们通过在$(\beta_1,\beta_2)$平面上绘制训练动力学图谱来研究这种依赖关系。在多种模型-任务设置中，一条近似线性的边界$1-\beta_2=C(1-\beta_1)$将出现尖峰与不出现尖峰的动力学区域分隔开来，而一维二次损失则产生近似三次方斜率的边界。一维超二次损失$L(x)\propto|x|^n$则恢复了近线性标度关系，并将边界系数与有效损失指数$n$联系起来。我们进一步表明，高置信度的交叉熵损失会发展出一种“核心-墙壁”景观，由狭窄的二次核心和随后陡峭的墙壁组成，这在优化器步长的尺度上产生有效的超二次行为。

    arXiv:2609.18314v1 Announce Type: new  Abstract: Loss spikes are recurrent instabilities in neural-network training and can arise from multiple mechanisms. For Adam in particular, macroscopic loss spikes have been linked to optimizer dynamics, yet how its two momentum timescales govern them remains unclear. We investigate this dependence by mapping training dynamics across the $(\beta_1,\beta_2)$ plane. Across a range of model--task settings, an approximately linear boundary, $1-\beta_2=C(1-\beta_1)$, separates spiky from non-spiky dynamics, whereas a one-dimensional quadratic loss produces approximately cubic slope. A one-dimensional superquadratic loss $L(x)\propto|x|^n$ recovers the near-linear scaling and links the boundary coefficient to the effective loss exponent $n$. We further show that confident cross-entropy losses develop a core--wall landscape comprising a narrow quadratic core followed by a steep wall, which produces effective superquadratic behavior at the scale of an op
    
[^39]: 监督图预测的图匹配松弛与摊销

    Graph Matching Relaxations and Amortization for Supervised Graph Prediction

    [https://arxiv.org/abs/2609.15437](https://arxiv.org/abs/2609.15437)

    该论文证明了Gromov-Wasserstein目标是监督图预测中最合适的图匹配松弛形式，并提出基于可微Sinkhorn算法的参数化匹配器来摊销图匹配问题，实现图预测模块与匹配器的联合学习。

    

    监督图预测（SGP）的端到端训练需要一个置换不变的损失函数来比较具有任意节点排序的预测图和目标图。这类损失函数通常涉及一个代价高昂的图匹配问题。我们首先研究了该问题的三种最优传输（Optimal Transport）松弛形式，并从理论和实证上表明，Gromov-Wasserstein（GW）目标最适合于监督图预测。随后，为了避免为每个训练样本求解由此产生的内层优化问题，我们提出对图匹配（节点对齐）问题进行摊销。对于每个训练样本，损失函数利用由参数化匹配器提供的传输计划，该匹配器基于应用于经验节点分布的可微Sinkhorn算法构建。图预测模块和匹配器被联合学习。我们在复杂度递增的玩具和现实世界监督图预测问题上展示了该方法的有效性，其中包括一个新颖的质谱到分子骨架（Mass-spectra to Scaffold）预测任务。

    arXiv:2609.15437v1 Announce Type: cross  Abstract: End-to-end Supervised Graph Prediction (SGP) requires a permutation-invariant loss to compare predicted and target graphs with arbitrary node orderings. Such losses typically involve a costly graph-matching problem. We first study three Optimal Transport relaxations of this problem and show, theoretically and empirically, that the Gromov-Wasserstein (GW) objective is the most suitable for SGP. Then, to avoid solving the resulting inner optimization for every training example, we propose to amortize the graph matching (node alignment) problem. For each training sample, the loss function leverages a transport plan provided by a parametric matcher based on the differentiable Sinkhorn algorithm applied on empirical node distributions. The graph prediction module and the matcher are jointly learned. We showcase the efficiency of this approach on toy and real world SGP problems of increasing complexity including a novel Mass-spectra to Scaff
    
[^40]: 特征叠加中线性能及性的高概率保证

    High-probability guarantees for linear accessibility in feature superposition

    [https://arxiv.org/abs/2609.09556](https://arxiv.org/abs/2609.09556)

    该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。

    

    神经网络可以利用特征叠加来编码比维度数量更多的概念，但特征间的交叉干扰限制了同时激活特征的线性能及性。通过将线性能及性建模为一个压缩感知问题，我们在次高斯噪声下针对固定支撑集推导出高概率界，证明了充分维度以线性方式扩展（d=O_ε(k log m)），而非此前最坏情况下的二次方限制。随后，我们通过高斯尾近似在各系统参数下验证了这些界。这些结果量化了线性表示假设的几何约束，为评估稀疏自编码器、组合泛化和神经网络可解释性提供了一个框架。

    arXiv:2609.09556v1 Announce Type: cross  Abstract: Neural networks can leverage feature superposition to encode more concepts than dimensions, but cross-feature interference constrains the linear accessibility of simultaneously active features. By framing linear accessibility as a compressed sensing problem, we derive high-probability bounds for fixed supports under subgaussian noise, proving the sufficient dimension scales linearly ($d=O_{\varepsilon}(k \log m)$) rather than prior worst-case quadratic limits. We then validate these bounds across system parameters through Gaussian-tail approximations. These results quantify the geometric constraints of the linear representation hypothesis, providing a framework for evaluating sparse autoencoders, compositional generalization, and neural interpretability.
    
[^41]: LiD-GLM：利普希茨约束的深度广义线性模型

    LiD-GLM: Lipschitz-constrained Deep Generalized Linear Models

    [https://arxiv.org/abs/2608.16340](https://arxiv.org/abs/2608.16340)

    提出一种利用可逆残差网络增强广义线性模型的方法，在保持随机单调性的同时实现非线性参数估计和分布假设的灵活校正。

    

    摘要：arXiv:2608.16340v1 公告类型：交叉 摘要：将传统统计模型与神经网络（NN）组件结合成半结构化混合模型，是一种引人入胜的方法，旨在构建理想情况下兼具传统可解释性与神经网络前所未有的灵活性的模型。为了保持可解释性，通常需要限制神经网络组件，以防止它们主导模型。然而，现有对神经网络组件施加结构约束的方法严重限制了模型的灵活性；相反，仅施加弱且间接约束的方法则失去了有意义的可解释性。因此，我们提出的方法利用可逆残差神经网络（i-ResNets）为广义线性模型配备非线性参数估计和对其分布假设的灵活校正，同时始终保留所建模分布在（原线性）变量上的随机单调性。

    arXiv:2608.16340v1 Announce Type: cross  Abstract: The combination of traditional statistical models and neural network (NN) components into semi-structured hybrid models is an intriguing approach to construct models that, ideally, combine traditional interpretability with the unprecedented flexibility of NNs. In order to preserve interpretability, it is usually necessary to restrict the NN components to prevent them from dominating the model. However, existing methods that enforce structural constraints on their NN components severely limit their models' flexibility; in contrast, methods that only enforce weak, indirect constraints lose meaningful interpretability. The method we propose therefore leverages invertible residual neural networks (i-ResNets) to equip generalized linear models with both nonlinear parameter estimation and a flexible correction of their distributional assumptions while always retaining stochastic monotonicity of the modeled distribution in the (formerly linea
    
[^42]: DAIF：一种基于近似消息传递的数据驱动多模态监督学习中间融合框架

    DAIF: A Data-Driven Intermediate Fusion Framework for Multimodal Supervised Learning via Approximate Message Passing

    [https://arxiv.org/abs/2608.02769](https://arxiv.org/abs/2608.02769)

    DAIF提出了一种数据自适应的中间融合框架，结合随机矩阵理论与非参数依赖性度量，通过根据模态间依赖性对模态聚类并进行经验贝叶斯先验估计，直接从数据中学习融合结构，克服了传统预定义融合架构无法适应模态间真实依赖关系的缺陷。

    

    多模态监督学习旨在利用多个异构数据源来提升预测性能。其核心挑战在于确定模态间的融合粒度：过度整合可能放大噪声，而整合不足则无法充分利用跨模态依赖关系。现有方法依赖于预先指定的融合架构，从早期融合到晚期融合，这些架构可能无法适应模态之间潜在的依赖结构。我们提出了DAIF，一个数据自适应的中间融合框架，它结合了随机矩阵理论和非参数依赖性度量，直接从数据中学习融合结构。我们在贝叶斯多模态因子模型的框架下进行操作，其中潜在因子的先验分布决定了跨模态依赖关系。我们的方法基于估计的模态间依赖性对模态进行聚类，然后对各簇的先验进行经验贝叶斯估计。这些估计出的先验…

    arXiv:2608.02769v2 Announce Type: replace-cross  Abstract: Multimodal supervised learning seeks to leverage multiple heterogeneous data sources to improve predictive performance. A central challenge is determining the fusion granularity across modalities: over-integration may amplify noise while under-integration fails to exploit cross-modal dependence. Existing approaches rely on pre-specified fusion architectures, from early to late fusion, that may not adapt to the underlying dependence structure among modalities. We propose DAIF, a data adaptive intermediate fusion framework that combines random matrix theory and non-parametric dependence measures to learn fusion structure directly from data. We operate under a Bayesian multimodal factor model where the prior on the latent factors determines the cross-modal dependence. Our method clusters modalities based on estimated intermodal dependence, then performs clusterwise empirical Bayes estimation of the priors. These estimated priors a
    
[^43]: 思考精简，智能委托，行动，重复：边缘LLM代理的校准推理与不确定性感知委托

    Think Short, Defer Smart, Act, and Repeat: Calibrated Reasoning and Uncertainty-Aware Deferral for Edge LLM Agents

    [https://arxiv.org/abs/2607.26865](https://arxiv.org/abs/2607.26865)

    TSDS框架通过轻量级收敛探针和基于困惑度的委托规则，在边缘LLM代理中实现推理预算与可靠性的平衡，并利用多目标LTT程序提供同时的有限样本保证。

    

    arXiv:2607.26865v2 公告类型：替换-交叉 摘要：遵循ReAct范式的LLM代理是实现复杂多步任务（包括多跳问答、代码生成和物理AI系统控制）的有前景的使能器。然而，当部署在边缘时，它们必须严格管理推理预算，同时保持可靠性，并且仅在本地不确定性过高而无法安全行动时，才委托给云端模型。我们提出“思考精简，智能委托”（TSDS）框架，该框架协同整合了一个轻量级收敛探针（一旦预期行动稳定即停止设备端推理）与一个基于困惑度的委托规则（将不确定行动升级到云端模型）。两种机制通过多目标“学习-然后-测试”（LTT）程序在端到端情节轨迹上联合校准，同时提供关于预期情节奖励和云端调用率的有限样本保证。我们在四个ReAct基准上评估TSDS，涵盖...

    arXiv:2607.26865v2 Announce Type: replace-cross  Abstract: LLM agents following the ReAct paradigm are promising enablers of complex multi-step tasks, including multi-hop question answering, code generation, and control of physical AI systems. Yet, when deployed at the edge, they must tightly manage their reasoning budget while remaining reliable and deferring to a cloud-side model only when local uncertainty is too high to act safely. We propose Think Short, Defer Smart (TSDS), a framework that synergistically integrates a lightweight convergence probe, which halts on-device reasoning once the intended action has stabilized, with a perplexity-based deferral rule that escalates uncertain actions to a cloud-side model. Both mechanisms are jointly calibrated on end-to-end episode trajectories via a multi-objective Learn-Then-Test (LTT) procedure, providing simultaneous finite-sample guarantees on expected episode reward and cloud-call rate. We evaluate TSDS on four ReAct benchmarks spann
    
[^44]: 基于预训练嵌入向量的非结构化数据计量经济学

    Econometrics with Pre-Trained Embeddings for Unstructured Data

    [https://arxiv.org/abs/2607.17378](https://arxiv.org/abs/2607.17378)

    本文针对经济学家使用预训练深度学习模型提取嵌入向量作为协变量这一流行做法，提供了理论基础，并提出“可迁移性”等充分条件，以解决预训练模型跨任务适用性不明和嵌入函数识别困难这两大问题。

    

    图像和文本等非结构化数据在实证经济学中的应用日益增多。由于在非结构化数据上训练机器学习模型成本高昂，经济学家通常使用计算机科学家开发的现成预训练深度学习模型来提取嵌入向量，然后将这些嵌入向量作为协变量用于目标经济分析。尽管这种做法很流行，但其理论基础仍然有限，主要存在两个困难：第一，预训练模型通常是在不同的数据集上针对不同的任务训练的，因此尚不清楚何时能够将其可靠地用于目标任务；第二，嵌入函数存在识别问题，这使其估计误差以及该误差对目标任务影响的分析变得复杂。我们提供了克服这些困难的充分条件，其中我们称之为“可迁移性”的关键条件决定了收敛速度（摘要在此处截断）

    arXiv:2607.17378v2 Announce Type: replace  Abstract: Unstructured data, such as images and text, are increasingly used in empirical economics. Since training machine-learning models on unstructured data is costly, economists often use off-the-shelf pre-trained deep learning models developed by computer scientists to extract embeddings, which are then used as covariates in target economic analyses. Despite the popularity of this practice, its theoretical foundations remain limited. There are two main difficulties. First, pre-trained models are typically trained on different datasets and for different tasks, making it unclear when they can be used reliably for the target task. Second, the embedding function is subject to an identification problem, complicating the analysis of its estimation error and the effect of that error on the target task. We provide sufficient conditions to overcome these difficulties. A key condition, which we call transferability, governs the convergence rate we 
    
[^45]: 高维M估计中的影响诊断：精确渐近性

    Influence Diagnostics in High-dimensional M-estimation: Precise Asymptotics

    [https://arxiv.org/abs/2607.09250](https://arxiv.org/abs/2607.09250)

    该论文在高维凸M估计中精确刻画了训练点留一影响的渐近分布，发现有影响力的样本平均而言倾向于靠近决策边界，与主动学习中的数据选择启发式方法相契合。

    

    某个给定训练点对统计模型的影响可以通过其对模型参数的留一影响来衡量，该度量量化了将此训练点从训练集中移除对学习到的权重所产生的影响。对于高斯设计下的凸M估计，在高维极限 n ≍ d 情形下，我们证明了训练点间影响的经验分布集中于一个确定性测度附近，并对该测度给出了精确刻画。这一刻画表明，有影响的样本平均而言往往位于接近决策边界的位置，这与主动学习中的标准数据选择启发式方法相呼应。

    arXiv:2607.09250v2 Announce Type: replace  Abstract: The impact of a given training point on a statistical model can be measured through its leave-one-out influence on the model parameters, which quantifies how its removal from the training set affects the learned weights. For convex M-estimation under Gaussian design, in the high-dimensional limit $n\asymp d$, we show that the empirical distribution of influences across training points concentrates around a deterministic measure which we sharply characterize. This characterization suggests that influential samples tend to lie on average close to the decision boundary, making contact with a standard data selection heuristic in active learning.
    
[^46]: 统计有效的训练后超参数选择：从调优到保证

    Statistically Valid Post-Training Hyperparameter Selection: From Tuning to Guarantees

    [https://arxiv.org/abs/2606.25601](https://arxiv.org/abs/2606.25601)

    提出以“先学习后测试”（LTT）范式为核心的统一统计框架，将训练后超参数选择转化为多元假设检验问题，为人工智能系统部署中的超参数调优提供正式的可靠性统计保证。

    

    训练后超参数选择是现代人工智能系统部署中的一个关键步骤，因为需要调整预训练模型的自由度，例如推理时参数、实现层面的设置以及驱动决策规则的阈值。尽管具有实际重要性，超参数选择通常采用尽力而为的经验方法（如网格搜索或贝叶斯优化）来执行，而这些方法在可靠性或安全性方面不提供正式的统计保证。本专著面向信号处理和机器学习研究人员，提出了一个以“先学习后测试”范式为核心的统一统计框架，用于实现可靠的训练后超参数选择。LTT将超参数选择问题表述为对候选超参数集合的多元假设检验。该框架能够实现超参数的选择……

    arXiv:2606.25601v2 Announce Type: replace-cross  Abstract: Post-training hyperparameter selection is a critical step in the deployment of modern artificial intelligence systems, given the need to tune degrees of freedom of pre-trained models such as inference-time parameters, implementation-level settings, and thresholds driving decision rules. Despite its practical importance, hyperparameter selection is typically performed using best-effort empirical methods such as grid search or Bayesian optimization, which provide no formal statistical guarantees on reliability or safety. This monograph, intended for an audience of signal processing and machine learning researchers, presents a unified statistical framework for reliable post-training hyperparameter selection, centered on the learn-then-test (LTT) paradigm. LTT formulates the hyperparameter selection problem as multiple hypothesis testing over a candidate set of hyperparameters. The framework enables the choice of hyperparameters th
    
[^47]: 反问题中后验期望的摊销求积方法

    Amortized quadrature for posterior expectations in inverse problems

    [https://arxiv.org/abs/2606.15871](https://arxiv.org/abs/2606.15871)

    本文提出“求积场”——一种集合等变神经网络，只需在一个后验族上训练一次，即可对任意观测、任意样本数 M 和任意被积函数，通过一次前向传播生成带符号权重的 M 节点求积格式，在保证精度不劣于蒙特卡洛的同时，避免了传统设计求积法需对每个新观测重复求解优化问题的高昂计算成本。

    

    arXiv:2606.15871v2 公告类型：replace-cross 摘要：反问题的解以及在该解上执行的任务的不确定性由后验期望来量化，每个后验期望是某个被积函数在 M 个后验样本上的平均值。虽然精心设计的求积法可以改进蒙特卡洛估计 O(M^{-1/2}) 的误差，但它们需要针对每个新观测求解一个优化问题（通常以后验密度为目标），这在计算上代价高昂。为了解决这一局限，我们引入了“求积场”，这是一个集合等变网络，能够在一次前向传播中将一个观测及其 M 个后验样本映射为具有 M 个节点和带符号权重的求积格式。该网络只需在一个后验族上训练一次，以最小化某类函数上的最坏情况积分误差，此后即可服务于任何观测、任意 M 以及该类中的任何被积函数，无需进一步优化。我们证明，在高概率意义下并带有可计算的松弛量，所得求积的性能从不劣于蒙特卡洛（摘要在此处被截断）。

    arXiv:2606.15871v2 Announce Type: replace-cross  Abstract: Uncertainty in the solution of an inverse problem and in the tasks performed on it is quantified by posterior expectations, each an average of an integrand over $M$ posterior samples. While designed quadratures improve on the $O(M^{-1/2})$ error of Monte-Carlo estimation, they solve an optimization problem, often against the posterior density, for every new observation, which can be computationally costly. To address this limitation, we introduce the quadrature field, a set-equivariant network that maps an observation and its $M$ posterior samples to an $M$-node signed-weight quadrature in one forward pass. Trained once on a family of posteriors to minimize the worst-case integration error over a class of functions, it serves any observation, any $M$ and any integrand in that class with no further optimization. We show that, with high probability and up to a computable slack, the resulting quadrature is never worse than the Mon
    
[^48]: 面向嵌入模型路由的策略后悔：具有低秩专家的上下文赌博机

    Policy Regret for Embedding Model Routing: Contextual Bandits with Low-Rank Experts

    [https://arxiv.org/abs/2606.14929](https://arxiv.org/abs/2606.14929)

    该论文将嵌入模型路由形式化为具有低秩专家的对抗性上下文线性赌博机问题，证明标准后悔度量存在结构性误设或统计不可处理的缺陷，并提出兼具表达能力与高效可学习性的对数二次策略类来实现查询依赖的模型路由。

    

    现代推荐系统日益依赖将多样化的查询动态路由到多个嵌入模型。尽管这一问题具有重要的实践意义，但在对抗性查询、赌博机反馈以及模型可观测性受限等现实条件下，该问题仍未得到充分理解。我们将嵌入模型路由形式化为一个具有低秩专家的对抗性上下文线性赌博机问题，其中上下文对应查询，动作对应物品，专家则对应工作在低秩潜在表示空间上的嵌入模型。我们首先证明了标准的后悔度量会遭遇结构性误设或统计上的不可处理性，并识别出一个对数二次策略类，该策略类既足够富有表现力以刻画依赖于查询的模型路由，又具备足够规整的结构以支持高效的在线学习。聚焦于这一在赌博机反馈下的对数二次策略优化问题——该问题本身亦具有独立的研究价值……

    arXiv:2606.14929v2 Announce Type: replace-cross  Abstract: Modern recommendation systems increasingly rely on dynamically routing diverse queries to multiple embedding models. Despite its practical significance, this problem remains poorly understood under realistic conditions like adversarial queries, bandit feedback, and limited observability of models. We formalize embedding model routing as an adversarial contextual linear bandit with low-rank experts, where contexts are queries, actions are items, and experts are the embedding models working on low-rank latent representation spaces. We first establish that standard regret notions suffer from structural misspecification or statistical intractability, and we identify a log-quadratic policy class that is expressive enough to capture query-dependent model routing, yet structured enough to allow efficient online learning. Focusing on this log-quadratic policy optimization problem under bandit feedback -- which is of independent interes
    
[^49]: INFUSER：影响力引导的自我进化提升推理能力

    INFUSER: Influence-Guided Self-Evolution Improves Reasoning

    [https://arxiv.org/abs/2606.09052](https://arxiv.org/abs/2606.09052)

    INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。

    

    自我进化为增强推理能力提供了一条可扩展的路径：预训练语言模型仅需极少的外部监督即可自我提升。然而，现有方法要么依赖大量精心策划或教师生成的训练数据，要么在生成器无监督运行时，仅通过难度启发式给予奖励，这未必能改进求解器。我们引入了INFUSER，一种迭代协同训练框架，包含两个共同演化的角色：一个生成器，从自动收集的非结构化文档池中起草问题和参考标准答案；以及一个求解器，通过在这些问题上训练来改进自身。求解器使用标准正确性奖励，依据生成器提供的答案进行训练，而生成器则通过一个优化器感知的影响力分数获得奖励，该分数衡量每个提议的问题是否真正能提升求解器在目标分布上的表现。由于这种连续且嘈杂的影响力分数难以直接处理，我们采用了相应策略进行优化。

    arXiv:2606.09052v4 Announce Type: replace-cross  Abstract: Self-evolution offers a scalable path to stronger reasoning: a pretrained language model improves itself with only minimal external supervision. Yet existing methods either depend on extensively curated or teacher-generated training data, or, when the generator runs unsupervised, reward it by a difficulty heuristic that need not improve the solver. We introduce INFUSER, an iterative co-training framework with two co-evolving roles: a Generator that drafts questions and reference golden answers from a pool of unstructured, automatically collected documents, and a Solver that improves by training on them. The solver is trained with standard correctness rewards against the generator-provided answers, while the generator is rewarded by an optimizer-aware influence score that measures whether each proposed question would actually improve the solver on the target distribution. Because this continuous, noisy influence score is poorly 
    
[^50]: 纵向切分分布式模型中稀疏协方差估计的信息论界

    Information-Theoretic Bounds for Sparse Covariance Estimation in the Vertical-Split Distributed Model

    [https://arxiv.org/abs/2606.07124](https://arxiv.org/abs/2606.07124)

    该论文首次证明在纵向切分分布式设置中，对互协方差矩阵施加稀疏性约束能够有效降低通信和样本复杂度，这与水平切分设置下稀疏性无法降低通信成本的结论形成鲜明对比。

    

    我们研究纵向切分（特征切分）设置下分布式协方差矩阵估计的极小极大估计误差。在该设置中，两个智能体各自观测 m 个独立同分布次高斯样本的不同坐标，并向中央服务器传送有限数量的比特。尽管先前研究已为稠密（非结构化）互协方差矩阵建立了近乎紧致的界，我们研究的问题是：对互协方差矩阵 $C_{21}$ 施加逐元素 s-稀疏性约束能否降低所需的通信复杂度和样本复杂度。与水平切分设置形成鲜明对比的是——在该设置中已有研究表明稀疏性并不能降低均值估计的通信成本——我们证明在纵向切分下，稀疏性确实有助于互协方差估计。具体而言，对于足够大的 $d_1d_2/s'$ 以及 $0<\varepsilon<\sigma^2\sqrt{s'}/32$，任何在期望 Frobenius 范数误差上达到目标精度的方案……（摘要原文在此处截断）

    arXiv:2606.07124v2 Announce Type: replace-cross  Abstract: We study the minimax estimation error for distributed covariance matrix estimation in the vertical-split (feature-split) setting, where two agents each observe different coordinates of~$m$ i.i.d.\ sub-Gaussian samples and communicate a limited number of bits to a central server. While \cite{rahmani2025fundamental} established nearly tight bounds for dense (unstructured) cross-covariance matrices, we investigate whether imposing elementwise $s$-sparsity on the cross-covariance $C_{21}$ can reduce the required communication and sample complexity. In contrast to the horizontal-split setting, where \cite{braverman2016communication} showed that sparsity does \emph{not} reduce communication cost for mean estimation, we prove that sparsity \emph{does} help for cross-covariance estimation in the vertical split.   Specifically, for sufficiently large $d_1d_2/s'$ and $0<\varepsilon<\sigma^2\sqrt{s'}/32$, any scheme achieving expected Fro
    
[^51]: 神经算子的平滑分段切割方法以处理不连续性与尖锐过渡

    Smooth Piecewise Cutting for Neural Operator to Handle Discontinuities and Sharp Transitions

    [https://arxiv.org/abs/2605.19823](https://arxiv.org/abs/2605.19823)

    提出 Cut-DeepONet 两阶段训练框架，通过将求解域切割为平滑子区域、并将不连续性表示为高维空间中的边界，使神经算子能够高效处理偏微分方程解中的不连续性与尖锐过渡。

    

    神经算子在学习偏微分方程（PDE）的解算子方面已取得出色的性能，但其固有的连续表示难以捕捉不连续性和尖锐过渡。现有方法通常在连续函数空间内近似此类特征，往往需要更大的模型容量和高分辨率数据。在本工作中，我们提出 Cut-DeepONet，这是一个两阶段训练框架，在显式建模不连续性的同时降低了学习复杂度。我们的方法通过一种提升策略重新表述该问题，将求解域划分为平滑的子区域，同时将不连续性表示为更高维空间中的边界。这种分离使算子学习任务与神经网络的归纳偏置相契合，并避免了直接近似不连续性。此外，一个额外的网络用于预测依赖于输入的不连续性位置，以……

    arXiv:2605.19823v2 Announce Type: replace-cross  Abstract: Neural operators have achieved strong performance in learning solution operators of partial differential equations (PDEs), but their inherently continuous representations struggle to capture discontinuities and sharp transitions. Existing approaches typically approximate such features within continuous function spaces, often requiring increased model capacity and high-resolution data. In this work, we propose Cut-DeepONet, a two-stage training framework that explicitly models discontinuities while reducing learning complexity. Our approach reformulates the problem via a lifting strategy, partitioning the domain into smooth subregions while representing discontinuities as boundaries in a higher-dimensional space. This separation aligns the operator learning task with the inductive bias of neural networks and avoids directly approximating discontinuities. An additional network predicts input-dependent discontinuity locations for 
    
[^52]: 联邦鞅后验采样

    Federated Martingale Posterior Samping

    [https://arxiv.org/abs/2605.18554](https://arxiv.org/abs/2605.18554)

    该论文提出联邦鞅后验采样（FMP），通过客户端上传可训练数据嵌入、服务器集中运行预测采样器的一次性并行协议，摆脱了联邦贝叶斯方法对先验设定的依赖，在性能上与中心化方法高度一致并取得最低的期望校准误差。

    

    联邦贝叶斯神经网络需要对模型参数设定一个固定的先验，这是众所周知的难题，而先验设定的偏差会严重损害模型的准确性与校准性能。受预测模型快速发展的启发，鞅后验（也称为预测贝叶斯）用预测分布取代先验-似然对，并通过反复抽取预测样本和重新拟合模型来恢复参数的不确定性。本文提出了联邦鞅后验（FMP）采样，这是一种一次性高度并行的协议，其中每个客户端上传一小组可训练的数据嵌入，由服务器在中心端运行预测采样器。对采样误差的分析展示了数据集压缩率的影响，实验表明FMP与中心化方法的表现十分接近，并取得了最低的平均期望校准误差（ECE）。

    arXiv:2605.18554v2 Announce Type: replace-cross  Abstract: Federated Bayesian neural networks require fixing a prior on the model parameters, which is notoriously difficult, and misspecification of this prior can severely degrade accuracy and calibration. Motivated by the rapid progress of predictive models, the martingale posterior, also known as predictive Bayes, replaces the prior--likelihood pair with a predictive distribution and recovers parameter uncertainty by repeatedly drawing predictive samples and refitting the model. This letter proposes {federated martingale posterior} (FMP) sampling, a one-shot embarrassingly parallel protocol in which each client uploads a small set of trainable data embeddings and the server runs the predictive sampler centrally. Analysis of the sampling error demonstrates the impact of the dataset compression rate, while experiments show that FMP closely matches the centralized counterpart and achieves the lowest mean expected calibration error (ECE) 
    
[^53]: 使用单一代理变量识别因果效应

    Identifying Causal Effects Using a Single Proxy Variable

    [https://arxiv.org/abs/2604.09135](https://arxiv.org/abs/2604.09135)

    该论文提出SPICE假设，证明在已知混杂因素生成单一（多维）代理变量机制的前提下因果效应可识别，将经典代理变量方法扩展到多维连续场景，并开发了适用于离散和连续处理的神经网络估计框架SPICE-Net。

    

    未观测的混杂因素是估计从处理变量到结果变量的因果效应时的一个关键挑战。在这项工作中，我们假设观察到未观测混杂因素的一个单一（可能是多维的）代理变量，并且已知混杂因素生成该代理变量的机制。在一个称为“因果效应的单一代理可识别性”（Single Proxy Identifiability of Causal Effects，简称SPICE）的假设下，我们证明了该误差机制是完备的，且因果效应是可识别的。我们将Kuroki和Pearl (2014)以及Pearl (2010)基于代理变量的因果可识别性结果扩展到多维连续设置、更灵活的函数关系以及更广泛的分布类别。此外，我们开发了一个基于神经网络的估计框架SPICE-Net来估计因果效应，该框架同时适用于离散和连续的处理变量。

    arXiv:2604.09135v2 Announce Type: replace  Abstract: Unobserved confounding is a key challenge when estimating causal effects from a treatment on an outcome. In this work, we assume that we observe a single, potentially multi-dimensional proxy variable of the unobserved confounder and that we know the mechanism that generates the proxy from the confounder. Under an assumption called Single Proxy Identifiability of Causal Effects or simply SPICE, we prove that this error mechanism is complete and causal effects are identifiable. We extend the proxy-based causal identifiability results by Kuroki and Pearl (2014); Pearl (2010) to multi-dimensional continuous settings, more flexible functional relationships and a broader class of distributions. Further, we develop a neural network based estimation framework, SPICE-Net, to estimate causal effects, which is applicable to both discrete and continuous treatments.
    
[^54]: 论Transformer对上下文关系的表达能力

    On the Expressive Power of Transformers for Contextual Relations

    [https://arxiv.org/abs/2603.25860](https://arxiv.org/abs/2603.25860)

    本文基于概率与最优传输理论构建了数学框架，揭示了注意力归一化与最优传输的深刻联系——softmax归一化产生条件关系而Sinkhorn归一化产生联合关系——并证明了Transformer在表示上下文关系上的通用逼近能力。

    

    Transformer通过将注意力作为建模上下文内交互的核心机制，彻底改变了机器学习。尽管注意力扮演着核心角色，但Transformer在表示上下文关系方面的理论能力仍不明确。在本工作中，我们通过构建一个基于概率和最优传输的数学框架来研究这一问题。我们将文本视为其表示的分布，并将注意力视为这些表示之间的概率关系。这一视角揭示了注意力归一化与最优传输之间的联系：标准的softmax归一化产生条件关系，而Sinkhorn归一化产生具有指定边缘分布的联合关系。因此，这两种机制都能从注意力分数中提供结构化的概率关系。在温和的条件下，我们为这两种设置建立了通用逼近结果。我们证明Transformer架构……（摘要原文此处截断）

    arXiv:2603.25860v4 Announce Type: replace  Abstract: Transformers have revolutionized machine learning by making attention a central mechanism for modeling interactions within a context. Despite the central role of attention, the theoretical capabilities of Transformers for representing contextual relations remain unclear. In this work, we address this question by developing a mathematical framework based on probability and optimal transport. We view a text as a distribution of its representations and attention as a probabilistic relation between them. This perspective reveals a connection between attention normalization and optimal transport: standard softmax normalization produces conditional relations, while Sinkhorn normalization produces joint relations with prescribed marginals. Thus, both mechanisms provide structured probabilistic relations from attention scores. Under mild conditions, we establish universal approximation results for both settings. We show that Transformer arch
    
[^55]: 基于函数型Tucker分解的自适应子空间建模

    Adaptive Subspace Modeling With Functional Tucker Decomposition

    [https://arxiv.org/abs/2603.25530](https://arxiv.org/abs/2603.25530)

    提出函数型Tucker分解（FTD），将模态级连续性约束直接嵌入Tucker分解，在RKHS中以函数形式建模连续模态，并推导重构误差界为跨域子空间迁移提供理论保证。

    

    张量为多维数据提供了结构化的表示，然而当数据源于连续过程时，离散化会丢失其潜在的连续结构。为了解决这一局限，我们提出了一种函数型Tucker分解，它将模态级别的连续性约束直接嵌入到分解过程中。FTD将连续模态建模为再生核希尔伯特空间（RKHS）中的函数，在保留Tucker模型多线性子空间结构的同时，避免了预先指定基函数。我们推导了连续模态的重构误差界，该误差界量化了在一个域上估计的子空间被复用于另一个域时的近似质量。这一误差界为子空间迁移提供了理论依据，我们通过高光谱成像和多变量时间序列分析中的跨域分类任务展示了该方法的实用价值。

    arXiv:2603.25530v2 Announce Type: replace  Abstract: Tensors provide a structured representation for multidimensional data, yet discretization discards the underlying continuous structure when the data originate from continuous processes. We address this limitation by introducing a functional Tucker decomposition (FTD) that embeds a mode-wise continuity constraint directly into the factorization. The FTD models the continuous mode as a function in a reproducing kernel Hilbert space (RKHS), avoiding a prespecified basis while preserving the multilinear subspace structure of the Tucker model. We derive a reconstruction error bound for the continuous mode that quantifies the approximation quality when a subspace estimated on one domain is reused on another. This bound provides theoretical justification for subspace transfer, whose practical value we demonstrate on cross-domain classification tasks in hyperspectral imaging and multivariate time-series analysis.
    
[^56]: 群舞现象、Mestre–Nagao 和与椭圆曲线的卷积神经网络

    Murmurations, Mestre--Nagao sums, and Convolutional Neural Networks for elliptic curves

    [https://arxiv.org/abs/2603.17681](https://arxiv.org/abs/2603.17681)

    本文将一维卷积神经网络应用于椭圆曲线的 Frobenius 迹，实现了对解析秩的高精度预测，并通过显著性曲线揭示了机器学习预测、群舞现象与 Mestre–Nagao 和之间的深刻联系。

    

    我们将一维卷积神经网络应用于 $\mathbb{Q}$ 上椭圆曲线的 Frobenius 迹，并评估和解释其预测能力。与 Kazalicki–Vlah、Bujanović–Kazalicki–Novak 以及 Pozdnyakov 的类似实验相一致，我们观察到在一系列导子范围内对解析秩的高精度预测。我们利用显著性曲线对这些预测进行解释，并探索了显著性曲线、群舞现象与 Mestre–Nagao 和之间有趣的相互作用。

    arXiv:2603.17681v2 Announce Type: replace-cross  Abstract: We apply one-dimensional convolutional neural networks to the Frobenius traces of elliptic curves over $\mathbb{Q}$ and evaluate and interpret their predictive capacity. In keeping with similar experiments by Kazalicki--Vlah, Bujanovi\'{c}--Kazalicki--Novak, and Pozdnyakov, we observe high accuracy predictions for the analytic rank across a range of conductors. We interpret the prediction using saliency curves and explore the interesting interplay between saliency, murmurations and Mestre--Nagao sums.
    
[^57]: 使用几何潜在子空间对离散数据进行生成建模

    Generative Modeling of Discrete Data Using Geometric Latent Subspaces

    [https://arxiv.org/abs/2601.21831](https://arxiv.org/abs/2601.21831)

    该论文提出一种几何潜在子空间框架，在类别分布乘积流形的指数参数空间中通过几何主成分分析（GPCA）学习高维离散数据的低维表示，并借助等距黎曼几何实现一致的流匹配生成建模。

    

    我们提出了一种用于离散数据生成建模的几何潜在子空间框架。具体而言，我们在类别分布乘积流形的指数参数空间中引入潜在子空间，作为学习高维离散数据低维表示的一种新方法。由此得到的低维潜在空间能够捕获统计依赖关系，并去除类别变量之间冗余的自由度。我们为参数域配备了黎曼几何，使得潜在子空间与所诱导的数据流形之间保持等距关系，从而实现一致的流匹配。利用这一结构，我们提出了一种几何感知的降维目标，称为几何主成分分析（GPCA），并将其表述为一种正则化的交叉熵最小化，以鼓励数据与其重构之间的黎曼距离尽可能小。特别是，在所诱导的……（摘要原文在此处截断）

    arXiv:2601.21831v3 Announce Type: replace  Abstract: We propose a geometric latent-subspace framework for generative modeling of discrete data. Specifically, we introduce latent subspaces in the exponential parameter space of product manifolds of categorical distributions as a novel approach to learning low-dimensional representations of high-dimensional discrete data. The resulting low-dimensional latent space captures statistical dependencies and removes redundant degrees of freedom among the categorical variables. We equip the parameter domain with a Riemannian geometry such that the latent subspace and induced data manifold are related isometrically, enabling consistent flow matching. Exploiting this structure, we propose a geometry-aware dimensionality reduction objective, called geometric PCA (GPCA), which we formulate as a regularized cross-entropy minimization that encourages small Riemannian distances between the data and their reconstructions. In particular, under the induced
    
[^58]: 基于算子值核的正则化随机梯度下降学习算子

    Learning Operators by Regularized Stochastic Gradient Descent with Operator-valued Kernels

    [https://arxiv.org/abs/2504.18184](https://arxiv.org/abs/2504.18184)

    本文针对从波兰空间到可分希尔伯特空间的算子学习问题，分析了算子值核再生核希尔伯特空间中在线与有限时域两种设置下的正则化随机梯度下降算法，建立了对输出空间维度无显式依赖、且在期望意义下接近最优的误差界，并给出了高概率估计与几乎必然收敛的结论。

    

    我们考虑一类统计逆问题，即估计从波兰空间到可分希尔伯特空间的回归算子，其中目标函数位于由算子值核诱导的向量值再生核希尔伯特空间中。为了解决相关的病态性问题，我们分析了在线和有限时域两种设置下的正则化随机梯度下降（SGD）算法：前者使用多项式衰减的步长和正则化参数，后者采用固定值。在适当的结构和分布假设下，我们建立了预测误差和估计误差界，且对输出空间维度没有显式依赖。所得到的收敛率在期望意义下是接近最优的，我们还推导了高概率估计，这意味着几乎必然收敛。我们的分析引入了一种通用的技术，用于获得高概率保证

    arXiv:2504.18184v5 Announce Type: replace  Abstract: We consider a class of statistical inverse problems involving the estimation of a regression operator from a Polish space to a separable Hilbert space, where the target lies in a vector-valued reproducing kernel Hilbert space induced by an operator-valued kernel. To address the associated ill-posedness, we analyze regularized stochastic gradient descent (SGD) algorithms in both online and finite-horizon settings. The former uses polynomially decaying step sizes and regularization parameters, while the latter adopts fixed values. Under suitable structural and distributional assumptions, we establish prediction and estimation error bounds with no explicit dependence on the dimension of the output space. The resulting convergence rates are near-optimal in expectation, and we also derive high-probability estimates that imply almost sure convergence. Our analysis introduces a general technique for obtaining high-probability guarantees in 
    
[^59]: 强化学习与交互式决策的基础

    Foundations of Reinforcement Learning and Interactive Decision Making

    [https://arxiv.org/abs/2312.16730](https://arxiv.org/abs/2312.16730)

    该专著在统一的统计框架下系统阐述了从多臂老虎机到基于函数逼近的强化学习的交互式决策算法设计与复杂度理论，并展示了如何将监督学习方法转化为决策算法、分析其性能以及判定问题的可解性。

    

    交互式决策是指在未知环境中学习采取良好行动的问题，即利用自身行动所产生的数据来持续改进，这一场景广泛存在于在线平台、机器人技术和医疗治疗等领域。本专著从统计学视角探讨交互式决策的算法设计与复杂度问题，在一个统一的框架内，从多臂老虎机逐步延伸至上下文老虎机、结构化老虎机，再到具有函数逼近的强化学习。书中特别关注函数逼近以及神经网络等灵活模型，并着重阐述监督学习与决策之间的联系：读者将学会如何将任意监督学习方法转化为决策算法、如何分析所得结果，以及如何判断给定问题能否通过少量交互求解。一个贯穿全文的统一主题是……

    arXiv:2312.16730v2 Announce Type: replace-cross  Abstract: Interactive decision making is the problem of learning to act well in an unknown environment, using the data that one's own actions generate to continuously improve, and arises in situations ranging from online platforms and robotics to medical treatments. This monograph gives a statistical perspective on algorithm design and complexity for interactive decision making, building from multi-armed bandits through contextual and structured bandits to reinforcement learning with function approximation within a single, unified framework. Special attention is paid to function approximation and flexible models such as neural networks, and to the connection between supervised learning and decision making: the reader will learn how to turn any supervised learning method into a decision making algorithm, how to analyze the result, and how to determine whether a given problem can be solved with few interactions.   A unifying theme is that 
    

