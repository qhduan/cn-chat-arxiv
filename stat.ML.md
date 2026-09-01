# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sharp Approximation Rates for Neural Networks with Affine Latent Parameterizations](https://arxiv.org/abs/2608.31157) | 本文针对仿射潜在参数化神经网络建立了紧的逼近速率，刻画了潜在维度与网络参数预算之间的最优权衡，并将超网络、低维参数化、参数高效适配和模型压缩统一在同一理论框架之下。 |
| [^2] | [Implementing neural network mixed-effects models in Template Model Builder (TMB)](https://arxiv.org/abs/2608.31133) | 本文提出了一个基于TMB的通用框架，利用自动微分和拉普拉斯近似实现神经网络混合效应模型，用户只需指定负联合对数似然，框架即可自动积分随机效应并计算精确梯度，从而避免了手动推导和简化近似。 |
| [^3] | [Overcoming critical slowing down in frustrated spin systems by learned multiscale sampling](https://arxiv.org/abs/2608.31114) | 提出利用小波条件重整化群（WCRG）方法学习而非构造簇，实现了对阻挫自旋系统从粗到细的多尺度采样，成功克服了传统簇算法在阻挫系统中失效导致的临界慢化问题。 |
| [^4] | [When Can We Work in Embedding Space? What Text Embeddings Preserve](https://arxiv.org/abs/2608.31059) | 该论文在潜在主题混合生成模型下精确刻画了文本嵌入可替代文本使用的条件，证明嵌入聚类对应于主题混合相似的文档、控制嵌入等价于控制主题混合物，并通过美国363个都市区的应用证明基于LLM生成经济描述的嵌入聚类能恢复可解释的经济原型并更清晰地分离当地就业动态。 |
| [^5] | [Learning the Geometry of Admissible Hypotheses through Inductive Bias in Training Distributions](https://arxiv.org/abs/2608.31028) | 该论文提出将稀疏性、逻辑依赖、常见PDE族和物理可容许性等科学归纳偏置直接嵌入训练分布，利用门控变分自编码器学习可容许偏微分方程的连续潜在流形表示，从而为混合变量和组合假设空间提供概率化表示的新框架。 |
| [^6] | [Selection-Aware Stress Testing for Interactive Agents](https://arxiv.org/abs/2608.30916) | 该论文提出SASST方法，通过在发现任务上学习任务重加权并在独立的确认任务上结合联合统计界限进行验证，解决了智能体评估中从同一数据同时得出工作流选择和压力测试结论所导致的虚假优势问题。 |
| [^7] | [Marginal Coordinate Test for Fr\'echet Regression with Random Objects](https://arxiv.org/abs/2608.30644) | 该论文提出了一种针对随机对象Fréchet回归的边际坐标检验方法，通过半监督设计构建无需响应残差的核条件均值依赖性（KCMD）U统计量，并建立了加权中心化卡方零极限、野自助法有效性、一致性和局部功效等完整理论性质。 |
| [^8] | [Informative Label Missingness in Multiclass Classification Information Geometry and Excess Risk](https://arxiv.org/abs/2608.30561) | 本文为多分类中的信息性标签缺失建立了基于似然的通用理论，通过有效信息分解与超额风险的二次展开，提出分类加权广义特征值准则，证明当缺失标签模式携带信息时，部分标注分类器可以获得比完全标注分类器更小的渐近分类风险。 |
| [^9] | [When the Martingale Never Stops Firing: Anytime-Valid Gating on Real Forecast Streams](https://arxiv.org/abs/2608.30502) | 该论文通过预设的案例研究，实测了当监测器在真实、非可交换的预测流上运行并与被纠正的学习器构成反馈回路时，保形检验鞅基于Ville不等式的任意时点有效误报保证是否依然成立。 |
| [^10] | [Confounding Masquerading as Improvement: A Systematic Evaluation of Offline Reinforcement Learning for Stroke Antithrombotic Treatment in a 129,000-Patient Registry](https://arxiv.org/abs/2608.30442) | 本研究通过在12.9万例卒中患者登记数据上系统评估多种离线强化学习算法与奖励设计，发现表观的策略改进实际源于奖励中内嵌的基线严重程度与预后混杂，去混杂后改进不再统计显著，警示临床离线RL评估中混杂偏倚可能被误判为策略优于医生。 |
| [^11] | [Learning PDE Time-Stepping with Neural Cellular Automata](https://arxiv.org/abs/2608.30328) | 本文提出基于神经细胞自动机（NCA）的PDE替代模型，通过学习在每个网格单元重复应用的局部同质更新规则来模拟微分算子的局部性，在五个经典PDE上于超出训练时间域两倍的时间范围内取得了最优的长时间预测精度。 |
| [^12] | [Estimating Population-Risk Curves Along Nonconvex Gradient Flows from the Training Sample](https://arxiv.org/abs/2608.30261) | 该论文提出Flow-ALO方法，从训练样本估计非凸梯度流的条件总体风险曲线，并在温和条件下给出显式的 $(n-1)^{-2}$ 误差界，且无需Hessian可逆。 |
| [^13] | [Fairness in multi-class multi-group classification problems via contextial coherent risk measures](https://arxiv.org/abs/2608.30223) | 该论文提出了一种基于一致性风险度量理论的公平分类器设计方法，能够处理向量值敏感属性导致的多类别、多重叠群体分类中的公平性问题，同时保护个体权利，并提供了可扩展且对数据损坏和稀缺具有鲁棒性的专门数值求解方法。 |
| [^14] | [Robust K-means Clustering using the Density Power Divergence Measure](https://arxiv.org/abs/2608.30093) | 提出了一种结合密度幂散度与马氏距离的鲁棒K-means聚类方法MK-means DPD及其保证有限步收敛的变体DC-MK-means DPD，并设计了两个抗异常值的聚类评估指标。 |
| [^15] | [Learning Representations through Token Prediction: Geometry, Approximation, and Downstream Guarantees](https://arxiv.org/abs/2608.30072) | 该论文建立了一个统计理论框架，证明词元预测会依据上下文分布间的Hellinger距离相似性来组织词元嵌入，并由此给出表示几何、编码器近似与下游任务性能之间的理论保证。 |
| [^16] | [A Deep Latent Variable Framework for Jointly Modeling Missingness, Measurement Error, and Heterogeneity](https://arxiv.org/abs/2608.30040) | 提出了一种基于深度潜变量表示的统一概率框架，通过层次树路由变分自编码器联合处理缺失数据、测量误差和群体异质性三类问题，并借助重新汇聚路由机制在子群体间共享参数以提升统计效率。 |
| [^17] | [Neural ODE enhanced linear mixed effect models for estimating complex association patterns of time-varying covariates with the marker trajectory](https://arxiv.org/abs/2608.29714) | 该论文提出Neural ODE-LMM模型，将神经常微分方程嵌入线性混合效应框架，利用学习到的向量场把协变量轨迹编码为连续时间潜在状态来驱动固定效应和随机效应设计，从而在保留经典似然推断的同时，灵活地捕捉时变协变量与结局标志物轨迹之间的复杂累积性关联模式。 |
| [^18] | [Which LLM for Which Work? Budgeted Model Allocation under Uncertain Evaluation](https://arxiv.org/abs/2608.29560) | 该论文研究了在模型质量评估存在不确定性时，如何为拥有固定AI预算的公司在各项重复性工作上分配大语言模型，指出质量表估计的两大失效来源——模型很少在相同工作上被比较、记录的分数只是代理指标——并证明购买更多评估也无法消除后者的不确定性。 |
| [^19] | [Online Gate-Driven Flow Control in Resin Transfer Moulding Using a Neural-Network Surrogate](https://arxiv.org/abs/2608.29521) | 该论文提出了一种结合迭代扩展卡尔曼滤波与神经网络代理模型的实时估计与控制策略，利用压力传感器数据在线估计边缘渗流强度并优化辅助闸口压力，从而防止树脂传递模塑过程中干斑的形成。 |
| [^20] | [Deciding When to Decide: Testing Operational Suboptimality Under Distributional Shift](https://arxiv.org/abs/2608.29465) | 本文提出RADAR框架，通过逆优化推断潜在偏好并以决策遗憾（最优性差距）为检验目标，判断分布偏移下已部署的决策是否已实质性次优而需要重新优化，从而忽略与决策无关的变化。 |
| [^21] | [SS-ESOAP: Self-Scaled Adaptive Preconditioning for Physics-Informed Learning](https://arxiv.org/abs/2608.29448) | SS-ESOAP通过在SOAP预条件方法上引入标量割线能量校正与自适应基更新及方差状态缩减，显著提升了物理信息神经网络在多数PDE基准上的训练精度与显存效率。 |
| [^22] | [Content Exploration Beyond the Feed: Creator Supply and the Shared Corpus](https://arxiv.org/abs/2608.29430) | 该论文通过某大型短视频平台的四项实验首次揭示了内容探索的双重价值——生产侧探索可使创作者发帖量提升8.55%，观众侧探索虽增加观看次数但减少观看时长，且探索引发的创作者供给与自然采纳会补充共享内容库，突破传统仅衡量观众侧效果的评估局限。 |
| [^23] | [Signed random Fourier features for fast density estimation with indefinite kernels](https://arxiv.org/abs/2608.29265) | 本文提出带符号随机傅里叶特征（SRFF）技术，将随机傅里叶特征方法从正定核推广至不定核，从而实现对大规模数据集的高效核密度估计。 |
| [^24] | [Uniform Statistical Convergence of Empirical Sinkhorn Potentials with Exponential and Polynomial Dependence on the Regularization Parameter](https://arxiv.org/abs/2608.29152) | 该论文证明了经验Sinkhorn势在商范数下具有 $n^{-1/2}$ 的非渐近收敛速率，并通过总体Sinkhorn映射的多项式残差稳定性等几何条件，将收敛界对正则化参数的依赖从指数级改进为多项式级。 |
| [^25] | [PathGuide: Dynamic Classifier-Free Guidance via On-Policy Transport Alignment](https://arxiv.org/abs/2608.29107) | 提出PathGuide框架，将无分类器引导的强度选择从静态参数调整为在线策略传输问题，利用连续性方程的弱形式证明了当引导场与精确条件场弱等价时采样路径与目标条件分布一致，从而实现动态的引导优化。 |
| [^26] | [Sharp Restricted Isometry Thresholds for Global Minima of Rank-Restricted Matrix LASSO](https://arxiv.org/abs/2608.29018) | 本文确定了秩约束矩阵LASSO在全局极小值处实现精确恢复的锐利限制等距阈值 $\delta<\delta_{\mathrm{sharp}}(k/r_{\star})$，并证明该阈值无法进一步改进。 |
| [^27] | [Jigsaw-CRL: Recovering Global Latent Causal Order from Fragmented Multi-Client Interventions](https://arxiv.org/abs/2608.28991) | 提出Jigsaw-CRL框架，首次在碎片化多客户端干预设置下（每个客户端仅能访问和干预部分潜在变量），通过组装客户端特定的结构片段来恢复全局潜在因果序。 |
| [^28] | [The information geometry of product-reference discrete diffusion: Interaction growth complexity and optimal scheduling](https://arxiv.org/abs/2608.28949) | 该论文提出了“交互增长复杂度（IGC）”这一基于信息几何的路径度量，可精确刻画积参考离散扩散采样器的KL离散化误差与迭代复杂度，并据此设计出复杂度更低的最优步长调度方案。 |
| [^29] | [Representation Learning with Quantum Signal Processing](https://arxiv.org/abs/2608.28828) | 该论文将量子信号处理（QSP）确立为表示学习的可解量子模型，精确计算了其量子神经正切核的统计特性，并证明了稀疏数据下完整非线性梯度流收敛于可积标量流以及普遍成立的有限深度速度极限。 |
| [^30] | [Quantitative Target Convergence and Uniform-in-Time Propagation of Chaos for Langevin-Regularized SVGD](https://arxiv.org/abs/2608.28827) | 该论文首次为朗之万正则化的斯坦变分梯度下降（SVGD）建立了定量目标收敛率与时间一致的混沌传播理论，证明在目标分布满足对数Sobolev不等式时算法可获得指数级最后迭代收敛。 |
| [^31] | [Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation](https://arxiv.org/abs/2608.26638) | 本文提出预测驱动评估框架，结合少量人工判断与大规模自动评分实现无偏且高效的系统比较，并引入PPSR元指标来衡量自动指标节省人工标注的程度，优于现有元指标。 |
| [^32] | [DTD-VAE: Disentangled Temporal Dependencies VAE for Credit Risk Prediction](https://arxiv.org/abs/2608.26473) | 本文提出DTD-VAE模型，通过解纠缠时间依赖并区分信用风险特征与客户偏好，提升了信用风险预测的准确性。 |
| [^33] | [ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing](https://arxiv.org/abs/2608.26083) | ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。 |
| [^34] | [On the Identifiability of Masked Prediction: Mode Blindness and Mask Schedules](https://arxiv.org/abs/2608.01383) | 本文发现掩码预测的可识别性完全由掩码调度决定，大上下文主导的调度对全局模式权重具有不可消除的盲性，并引入ε-可识别性模量来量化这一现象。 |
| [^35] | [The Value of Depth in Message Passing on Sparse Graphs: A Kesten-Stigum Dichotomy](https://arxiv.org/abs/2607.16676) | 该论文证明了稀疏图上消息传递的深度价值由单一的Kesten-Stigum比率κ=γ²Δ决定，呈现二分性：当κ<1时误差以几何速率收敛，深度超过O(log(1/ε))的额外层对误差的改善小于ε。 |
| [^36] | [Seq2Synth: Benchmarking Temporal Fidelity in Synthetic Sequential Tabular Data](https://arxiv.org/abs/2607.15606) | 该论文提出了Seq2Synth，一个评估合成序列表格数据时间保真度的统一基准，揭示出静态指标接近完美的生成模型仍会违反基本时间约束，且静态与时间感知评估的模型排名存在显著差异。 |
| [^37] | [The Dual Nature of LLM Persona: Aggregated Tendencies and Frame-Dependent Geometry](https://arxiv.org/abs/2607.02368) | 本论文发现LLM人格表达包含聚合倾向与框架依赖几何两个可分离成分，后者并非固有属性，而是编码聚合无法捕捉信息的协调模式。 |
| [^38] | [Self-Organized Conformal Prediction: Reducing Regional Coverage Gaps with Unsupervised Group Discovery](https://arxiv.org/abs/2606.29403) | 提出自组织保形预测（SOCP），利用无需校准标签的无监督自组织映射（SOM）发现输入空间分组并从中提取校准缓冲，在预测器和非一致性分数保持不变的前提下实现精确的条件覆盖有效性，从而减少特征空间异质区域的覆盖差距。 |
| [^39] | [In LLM Reasoning, there is Irrationality on top of Value Misalignment](https://arxiv.org/abs/2606.20624) | 该论文提出“理性价值风险”这一新概念，将“即使经过良好价值对齐的LLM在推理时仍无法最大化对齐价值”这一差距进行数学形式化，并通过覆盖多类主流模型和基准的大量实验证明该风险普遍存在，且价值对齐只能减少而无法消除它。 |
| [^40] | [PEAR: Permutation-Equivariant Adaptive Routing Multi-Agent Debate](https://arxiv.org/abs/2606.20621) | PEAR是一种推理时免训练的多智能体辩论协议，通过在辩论轮次间动态切换智能体角色分配和稀疏拓扑结构，消除了固定拓扑带来的位置偏差和角色敏感性，使影响力分布更均匀，从而提升大语言模型推理的准确性与泛化能力。 |
| [^41] | [Mamba-Assisted Non-Markovian Closure for Reduced-Order Modeling](https://arxiv.org/abs/2606.05371) | 该论文提出Mamba辅助闭合框架，将非马尔可夫闭合建模转化为序列建模问题，利用Mamba模型高效预测闭合项并与降阶控制方程耦合，从而实现对高维动力系统的高效降阶建模。 |
| [^42] | [Simultaneous Monitoring of Shape and Surface Color via 4D Point Clouds: A Registration-free Approach](https://arxiv.org/abs/2605.08753) | 提出了一种基于4D点云的免配准框架SMAC，利用Laplace-Beltrami算子谱特性同时监测形状变形与颜色异常，并通过空间感知诊断程序定位异常来源。 |
| [^43] | [Sub-Gaussian Concentration and Entropic Normality of the Maximum Likelihood Estimator](https://arxiv.org/abs/2605.07107) | 本文强化了极大似然估计的经典渐近正态性结果，建立了归一化MLE的次高斯尾界、全矩收敛及熵中心极限定理，并在附加正则性条件下进一步证明了MLE本身的熵正态性。 |
| [^44] | [Augmented transfer regression learning for completely missing covariates](https://arxiv.org/abs/2605.04469) | 针对目标人群中协变量完全缺失的跨人群数据问题，提出一种增强迁移回归学习方法，在子人群漂移假设下结合重要性加权估计方程与矩插补，实现了双重稳健的参数估计。 |
| [^45] | [First-Order Efficiency for Probabilistic Value Estimation via A Statistical Viewpoint](https://arxiv.org/abs/2605.02827) | 该论文从统计视角揭示了看似不同的各类概率价值（如Shapley值）蒙特卡洛估计器实际上共享一个由采样定律和工作代理函数决定的一阶展开结构，从而推导出主均方误差的显式表达式并建立了一阶有效性理论。 |
| [^46] | [Inverting Foundation Models of Brain Function with Simulation-Based Inference](https://arxiv.org/abs/2604.23865) | 该研究将脑活动基础模型与大型语言模型结合，利用基于仿真的推断方法实现了从合成脑活动中反向恢复刺激的语言参数（效价、唤醒度、支配度），证明了脑模拟器的神经编码完整保留了刺激维度信息，且LLMs可作为可控刺激生成器。 |
| [^47] | [Contrast-Space Projection for Network Meta-Analysis: An Exact and Invariant Study-Based Decomposition of Direct and Indirect Contributions](https://arxiv.org/abs/2604.21994) | 本文提出了网络元分析的对比空间投影方法，通过唯一且不变的研究级分解精确量化直接与间接证据的贡献，既能精确重构NMA估计量，又统一了广义Cochran Q的异质性与不一致性分解。 |
| [^48] | [Elements of Conformal Prediction](https://arxiv.org/abs/2603.23923) | 本文系统阐述了共形预测这一无分布且模型无关的预测推断框架的核心思想：仅需可交换性等弱假设，即可为任意黑箱学习算法提供精确的有限样本保证。 |
| [^49] | [Accelerate Vector Diffusion Maps by Landmarks](https://arxiv.org/abs/2603.21247) | 提出 LA-VDM 算法，通过地标约束扩散和两阶段归一化加速向量扩散映射，能够从点云精确恢复平行移动并渐近收敛于连接拉普拉斯算子。 |
| [^50] | [Model Selection and Parameter Estimation for Multidimensional Gaussian Mixture Models with a Common Covariance Matrix](https://arxiv.org/abs/2603.19657) | 该论文针对已知共同协方差矩阵的多维高斯混合模型，提出基于傅里叶协方差矩阵的谱方法进行模型阶数选择与分量均值估计，证明了区分 $k$ 分量与 $(k-1)$ 分量混合模型需要 $\Omega(\Delta^{-(4k-4)})$ 个样本的极小化极大下界，并给出了样本量为 $\Delta^{-(8k-8)}$ 阶的谱阈值oracle估计器、实用的奇异值比率估计器以及基于MUSIC型投影目标的分数初始化梯度下降均值估计方法。 |
| [^51] | [Prediction-Powered Conditional Inference](https://arxiv.org/abs/2603.05575) | 该论文提出一种将RKHS局部化与机器学习预测校正相结合的预测驱动条件推断框架，在标注数据稀缺、未标注数据充足的场景下，对条件均值等条件泛函构造出方差更低且始终有效的估计量与置信区间。 |
| [^52] | [AI-Generated Measurements for Identification and Inference with Missing Data: A Weak Shadow Variable Approach](https://arxiv.org/abs/2602.16061) | 该论文提出一个弱假设的部分识别框架，将AI（如大语言模型）从非结构化数据中生成的测量值用作弱影子变量，从而在非随机缺失（MNAR）数据下实现总体量的识别与推断。 |
| [^53] | [Universal Redundancies in Time Series Foundation Models](https://arxiv.org/abs/2602.01605) | 该论文发现领先的时间序列基础模型中间层存在普遍冗余，模型对整层消融具有鲁棒性，并通过将Transformer框架化为核回归器的理论框架，提出了基于注意力头投影稳定秩的纯内在消融策略。 |
| [^54] | [Soft Fitted Q-Iteration without Bellman Completeness: Occupancy Reweighting and Temperature Annealing](https://arxiv.org/abs/2512.23927) | 该论文提出无需贝尔曼完备性假设的软拟合Q迭代方法，通过占用度重加权和温度退火，利用折扣占用度范数下的压缩性来保证离线强化学习的稳定性。 |
| [^55] | [Fitted Q-Evaluation without Bellman Completeness via Occupancy Weighting](https://arxiv.org/abs/2512.23805) | 该论文提出占用度加权FQE，通过用目标策略折扣占用度比率改变回归权重，使投影范数与目标策略动力学对齐并恢复Bellman收缩性，从而在无需Bellman完备性假设的情况下获得有限样本评估保证。 |
| [^56] | [Diffusion Models in Simulation-Based Inference: A Tutorial Review](https://arxiv.org/abs/2512.20685) | 本综述系统梳理了扩散模型在基于模拟推断（SBI）中的最新进展，涵盖训练、推断与评估的设计选择，并讨论了引导、流匹配、一致性模型等概念以及噪声调度、参数化和采样器对效率与统计精度的影响。 |
| [^57] | [Autotune: fast, accurate, and automatic tuning parameter selection for Lasso](https://arxiv.org/abs/2512.11139) | 该论文提出autotune方法，通过在回归系数与噪声标准差之间交替优化带惩罚的高斯对数似然，实现Lasso调优参数的全自动选择，在低信噪比情形下比现有方法更快且具有更优的泛化性能和模型选择效果。 |
| [^58] | [All Emulators are Wrong, Many are Useful, and Some are More Useful Than Others: A Reproducible Comparison of Computer Model Surrogates](https://arxiv.org/abs/2512.09060) | 该论文对29种代理模型在60个经典测试函数和40个真实数据集上进行了大规模、完全可复现的统一比较，并发布了R包duqling以支持公平一致的代理模型基准测试。 |
| [^59] | [Latency-Response Theory Model: Evaluating Large Language Models via Response Accuracy and Chain-of-Thought Length](https://arxiv.org/abs/2512.07019) | 提出潜时-反应理论（LaRT）模型，通过引入潜在能力与潜在速度之间的相关参数，联合建模LLM的响应准确率与思维链长度，并配套高效随机逼近EM算法，为LLM评估提供了更全面的框架。 |
| [^60] | [SHAKE-GNN: Scalable Hierarchical Kirchhoff-Forest Graph Neural Network](https://arxiv.org/abs/2509.22100) | SHAKE-GNN是一种基于Kirchhoff森林层次结构的新型可扩展图级图神经网络框架，通过随机多分辨率分解生成多尺度表示，在大规模图分类任务中实现了性能与可扩展性之间的灵活权衡。 |
| [^61] | [Online selective conformal inference: adaptive scores, convergence rates and optimality](https://arxiv.org/abs/2508.10336) | 该论文提出OnlineSCI算法，将ACI在线共形推断扩展至用户可自主选择推断时机的选择性设定，能够控制被选时间点上的错误覆盖比例及条件瞬时错误率，并给出收敛速率与最优性保证。 |
| [^62] | [Integrating attention into explanation frameworks for language and vision transformers](https://arxiv.org/abs/2508.08966) | 该论文提出两种新颖的解释方法，将注意力权重整合进Shapley值分解等可解释AI框架中，为自然语言处理和计算机视觉任务中的Transformer模型提供有意义的解释。 |
| [^63] | [Test of partial effects for Frechet regression on Bures-Wasserstein manifolds](https://arxiv.org/abs/2506.23487) | 本文提出了一种针对Bures-Wasserstein流形上Fréchet回归偏效应的新型检验方法，证明了其渐近有效性和一致性，并将其应用于单细胞数据中年龄对基因共表达结构影响的研究。 |
| [^64] | [Optimal Estimation of Watermark Proportions in Hybrid AI-Human Texts](https://arxiv.org/abs/2506.22343) | 本文将混合来源文本中水印比例的估计转化为基于枢轴统计量的混合模型参数估计问题，证明了该参数在某些水印方案下不可辨识，而对采用连续枢轴统计量的水印方法在温和条件下可辨识，并提出了高效的最优估计器。 |
| [^65] | [Stein's method for marginals on large graphical models](https://arxiv.org/abs/2410.11771) | 该论文受斯坦因方法启发提出新颖的 δ-局部性条件，为大型图模型中近似分布的边缘分布建立了与维度无关的一致误差界，并据此发展了局部化采样方法。 |
| [^66] | [Adaptive teachers for amortized samplers](https://arxiv.org/abs/2410.01432) | 提出了一种教师-学生框架，利用自适应的辅助“教师”模型采样“学生”摊销采样器的高损失区域，从而提供高效的训练课程，增强模式覆盖和探索效率。 |
| [^67] | [Understanding Deep Learning via Notions of Rank](https://arxiv.org/abs/2408.02111) | 本论文以“秩”为核心概念构建深度学习理论，证明了梯度训练会对多种神经网络产生隐式低秩正则化从而可能解释对自然数据的泛化，并借助神经网络与张量分解的联系，用秩的概念刻画了图神经网络建模交互的能力。 |
| [^68] | [Autoencoders in Function Space](https://arxiv.org/abs/2408.01362) | 本文提出并分析了函数空间中的自编码器（FAE）与变分自编码器（FVAE），解决了函数空间中VAE目标函数的良定义性难题，使算法能够在不同分辨率之间平滑运作。 |
| [^69] | [Model Selection and Parameter Estimation of One-Dimensional Gaussian Mixture Models](https://arxiv.org/abs/2404.12613) | 本文针对一维高斯混合模型证明了阶数估计的样本复杂度基本下界，并提出一种基于傅里叶测量的估计算法，其样本复杂度与该下界匹配，从而以最优复杂度同时实现模型阶数和混合分布的估计。 |
| [^70] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^71] | [Investigating Statistical Inference and Covariate Effects in Shallow Neural Networks](https://arxiv.org/abs/2311.08139) | 该论文研究了如何在惩罚化浅层前馈神经网络中应用经典统计推断方法（如协变量级Wald检验和协变量效应可视化），使其从黑盒预测工具转变为具有可解释性的统计模型。 |
| [^72] | [GFlowNets and variational inference](https://arxiv.org/abs/2210.00580) | 本文证明了变分推断与生成流网络在特定条件下学习目标的期望梯度等价，并指出 GFlowNets 得益于无需重要性采样的离策略训练能力，在捕捉多峰目标分布的多样性方面更具优势。 |
| [^73] | [PhyloGFN: Phylogenetic inference with generative flow networks.](http://arxiv.org/abs/2310.08774) | PhyloGFN是一种基于生成流网络的系统发育推断方法，通过采样复杂的组合结构，能够产生多样且高质量的进化假设，并在边缘似然估计方面具有竞争力。 |
| [^74] | [Delta-AI: Local objectives for amortized inference in sparse graphical models.](http://arxiv.org/abs/2310.02423) | Delta-AI算法提出了一种基于稀疏图模型的摊还推理方法，通过局部信用分配和离策略训练加快了训练速度。 |
| [^75] | [Potential Energy Advantage of Quantum Economy.](http://arxiv.org/abs/2308.08025) | 量子计算在能源效率方面具有优势，并且能够在盈利和能源效率上超越经典计算。这使得量子计算成为计算行业更可持续的选择。 |
| [^76] | [Deep graph kernel point processes.](http://arxiv.org/abs/2306.11313) | 本文提出了一种基于潜在图拓扑的图点过程方法，并开发了一种新颖的深度图核来描述事件之间的触发和抑制效应，该方法在合成和实际数据集上具有优越性。 |

# 详细

[^1]: 具有仿射潜在参数化的神经网络的紧逼近速率

    Sharp Approximation Rates for Neural Networks with Affine Latent Parameterizations

    [https://arxiv.org/abs/2608.31157](https://arxiv.org/abs/2608.31157)

    本文针对仿射潜在参数化神经网络建立了紧的逼近速率，刻画了潜在维度与网络参数预算之间的最优权衡，并将超网络、低维参数化、参数高效适配和模型压缩统一在同一理论框架之下。

    

    许多参数高效方法通过低维潜在表示来生成大型神经网络的参数。给定一个具有 $P_\Phi$ 个参数槽位的架构 $\Phi$，我们记 $\boldsymbol{\theta}_f=\mathcal{G}(\boldsymbol{\xi}_f)$，其中 $\mathcal{G}\colon\mathbb{R}^M\to\mathbb{R}^{P_\Phi}$ 是参数生成器，$\boldsymbol{\xi}_f\in\mathbb{R}^M$ 是目标函数 $f$ 的潜在表示。架构 $\Phi$ 和生成器 $\mathcal{G}$ 在整个目标类中共享，而每个目标函数 $f$ 由其自身的潜在向量 $\boldsymbol{\xi}_f$ 表示，并利用 $\Phi_{\mathcal{G}(\boldsymbol{\xi}_f)}$ 来逼近 $f$。该框架涵盖了超网络、低维参数化、参数高效适配以及模型压缩。因此，理解潜在维度 $M$ 与网络预算 $P$ 之间的权衡对于刻画……至关重要（摘要在此处被截断）。

    arXiv:2608.31157v1 Announce Type: new  Abstract: Many parameter-efficient methods generate the parameters of a large neural network from a low-dimensional latent representation. Given an architecture $\Phi$ with $P_\Phi$ parameter slots, we write $\boldsymbol{\theta}_f=\mathcal{G}(\boldsymbol{\xi}_f)$, where $\mathcal{G}\colon\mathbb{R}^M\to\mathbb{R}^{P_\Phi}$ is a parameter generator and $\boldsymbol{\xi}_f\in\mathbb{R}^M$ is a latent representation of the target function $f$. The architecture $\Phi$ and the generator $\mathcal{G}$ are shared across the entire target class, while each target $f$ is represented by its own latent vector $\boldsymbol{\xi}_f$, with $\Phi_{\mathcal{G}(\boldsymbol{\xi}_f)}$ approximating $f$. This framework encompasses hypernetworks, low-dimensional parameterizations, parameter-efficient adaptation, and model compression. Understanding the tradeoff between the latent dimension $M$ and the network budget $P$ is therefore fundamental to characterizing the ex
    
[^2]: 在Template Model Builder (TMB)中实现神经网络混合效应模型

    Implementing neural network mixed-effects models in Template Model Builder (TMB)

    [https://arxiv.org/abs/2608.31133](https://arxiv.org/abs/2608.31133)

    本文提出了一个基于TMB的通用框架，利用自动微分和拉普拉斯近似实现神经网络混合效应模型，用户只需指定负联合对数似然，框架即可自动积分随机效应并计算精确梯度，从而避免了手动推导和简化近似。

    

    神经网络混合效应模型（NMMs）通过将人工神经网络强大的表示能力与预测能力，同混合效应模型捕捉复杂相关结构的能力相结合，获得了越来越多的关注。然而，现有的估计方法严重依赖于对目标函数和梯度的手动推导，这在本质上迫使研究者采用简化的近似方法，并严重限制了NMMs的复杂度和准确性。在这项工作中，我们介绍了一个使用Template Model Builder (TMB)实现NMMs的通用框架。通过利用自动微分和拉普拉斯近似，TMB只要求用户指定负联合对数似然以及任意正则化项。该框架自动对随机效应进行积分，并计算边际目标函数及其精确梯度，从而消除了手动推导或临时近似的需求。我们演示了……（原文摘要在此处截断）

    arXiv:2608.31133v1 Announce Type: cross  Abstract: Neural network mixed-effects models (NMMs) have gained traction by combining the strong representation and predictive power of artificial neural networks with the capacity of mixed-effects modeling to capture complex correlation structures. However, existing estimation approaches rely heavily on manual derivations of objective functions and gradients, which inherently forces simplifying approximations and severely constrains the complexity and accuracy of NMMs. In this work, we introduce a general framework for implementing NMMs using Template Model Builder (TMB). By leveraging automatic differentiation and Laplace approximation, TMB requires users to specify only the negative joint log-likelihood and any regularization terms. The framework automatically integrates out random effects and evaluates the marginal objective function alongside its exact gradients, eliminating the need for manual derivations or ad hoc approximations. We demo
    
[^3]: 通过学习的多尺度采样克服阻挫自旋系统中的临界慢化

    Overcoming critical slowing down in frustrated spin systems by learned multiscale sampling

    [https://arxiv.org/abs/2608.31114](https://arxiv.org/abs/2608.31114)

    提出利用小波条件重整化群（WCRG）方法学习而非构造簇，实现了对阻挫自旋系统从粗到细的多尺度采样，成功克服了传统簇算法在阻挫系统中失效导致的临界慢化问题。

    

    诸如 Swendsen-Wang 和 Wolff 方法等簇算法是缓解统计系统中临界慢化现象的最成功的 MCMC 方法之一。然而，这些构造性的簇算法在存在哪怕极其微弱的阻挫时也会失效。在这里，我们通过学习而非构造相关簇来规避这一根本性限制。具体而言，我们使用小波条件重整化群（WCRG）采样方法来学习一个阻挫二维软自旋模型中集体涨落的概率分布，然后通过采样条件小波分布，从粗尺度到细尺度递归地生成系统构型。WCRG 方法再现了系统在不同相中的主要统计特性，包括局域场分布和结构因子。在类伊辛临界点处，条件动力学保持去相关（原文在此处截断）。

    arXiv:2608.31114v1 Announce Type: cross  Abstract: Cluster algorithms, such as the Swendsen--Wang and Wolff methods, are among the most successful MCMC methods for mitigating critical slowing down in statistical systems. These constructive cluster algorithms, however, fail in the presence of even extremely weak frustration. Here, we sidestep this fundamental limitation by learning rather than constructing the relevant clusters. Specifically, we use the wavelet conditional renormalization group (WCRG) sampling method to learn the probability distribution of collective fluctuations of a frustrated two-dimensional soft-spin model. Configurations are then generated recursively from coarse to fine scales by sampling conditional wavelet distributions. The WCRG method reproduces the main statistical properties of the system across different phases, including the local-field distribution and the structure factor. At an Ising-like critical point, the conditional dynamics remains decorrelated wi
    
[^4]: 我们何时可以在嵌入空间中工作？文本嵌入保留了什么

    When Can We Work in Embedding Space? What Text Embeddings Preserve

    [https://arxiv.org/abs/2608.31059](https://arxiv.org/abs/2608.31059)

    该论文在潜在主题混合生成模型下精确刻画了文本嵌入可替代文本使用的条件，证明嵌入聚类对应于主题混合相似的文档、控制嵌入等价于控制主题混合物，并通过美国363个都市区的应用证明基于LLM生成经济描述的嵌入聚类能恢复可解释的经济原型并更清晰地分离当地就业动态。

    

    文本嵌入何时可以作为实证分析的输入？它们的使用基于一个假设：我们可以用文本的低维嵌入来替代文本本身，且在此过程中损失很小。我在一个文档为潜在主题混合物的生成模型下使这一假设变得精确。我研究了两种用途——在嵌入空间中对单元进行聚类，以及对高维文本进行控制。嵌入的聚类是一组具有相似主题混合物的文档；对嵌入进行控制等价于对主题混合物进行控制，因此有效性归结为该混合物是否捕捉到了混杂因素。在对美国363个都市区的应用中，基于LLM生成的经济描述的嵌入聚类恢复出可解释的经济原型，并且比基于模型残差的聚类或基于精心挑选的行业与人口统计协变量集合的聚类更能清晰地分离当地就业动态。

    arXiv:2608.31059v1 Announce Type: cross  Abstract: When do text embeddings work as inputs to empirical analysis? Their use rests on an assumption: that we can trade text for its low-dimensional embedding, and lose little in doing so. I make that assumption precise under a generative model in which documents are mixtures of latent topics. I study two uses---clustering units in embedding space and controlling for high-dimensional text. A cluster of embeddings is a set of documents with similar topic mixtures; controlling for the embedding is equivalent to controlling for the topic mixture, so validity reduces to whether that mixture captures the confounding. In an application to 363 U.S. metropolitan areas, embedding-based clusters of LLM-generated economic descriptions recover interpretable economic archetypes and separate local employment dynamics more sharply than clustering on model residuals, or on a curated set of industry and demographic covariates.
    
[^5]: 通过训练分布中的归纳偏置学习可容许假设的几何结构

    Learning the Geometry of Admissible Hypotheses through Inductive Bias in Training Distributions

    [https://arxiv.org/abs/2608.31028](https://arxiv.org/abs/2608.31028)

    该论文提出将稀疏性、逻辑依赖、常见PDE族和物理可容许性等科学归纳偏置直接嵌入训练分布，利用门控变分自编码器学习可容许偏微分方程的连续潜在流形表示，从而为混合变量和组合假设空间提供概率化表示的新框架。

    

    科学发现通常需要在符合实验观测的多个相互竞争的假设之间进行推理。然而，对于混合变量和组合型的假设空间，构建概率表示仍然具有挑战性，因为活跃的模型组件及其相关参数都是未知的。在这项工作中，我们提出了一个框架，通过将科学归纳偏置直接嵌入到训练分布中，来学习可容许偏微分方程（PDE）的连续潜在表示。逐步丰富的结构原理（例如稀疏性、逻辑依赖关系、常见PDE族以及物理可容许性）被用来生成一个结构化的假设分布，门控变分自编码器从中学习到一个连续的潜在流形。实验结果表明，所得到的11维表示能够准确地重建大量的……

    arXiv:2608.31028v1 Announce Type: cross  Abstract: Scientific discovery often requires reasoning over competing hypotheses that are consistent with experimental observations. For mixed-variable and combinatorial hypothesis spaces, however, constructing probabilistic representations remains challenging because both the active model components and their associated parameters are unknown. In this work, we present a framework for learning continuous latent representations of admissible partial differential equations (PDEs) by embedding a scientific inductive bias directly into the training distribution. Progressively richer structural principles (e.g., sparsity, logical dependencies, common PDE families, and physical admissibility) are used to generate a structured distribution of hypotheses from which a gated variational autoencoder learns a continuous latent manifold. Experimental results show that the resulting 11-dimensional representation accurately reconstructs a broad collection of 
    
[^6]: 面向交互式智能体的选择感知压力测试

    Selection-Aware Stress Testing for Interactive Agents

    [https://arxiv.org/abs/2608.30916](https://arxiv.org/abs/2608.30916)

    该论文提出SASST方法，通过在发现任务上学习任务重加权并在独立的确认任务上结合联合统计界限进行验证，解决了智能体评估中从同一数据同时得出工作流选择和压力测试结论所导致的虚假优势问题。

    

    智能体评估通常使用同一个基准来选择工作流，然后再寻找该工作流优势减弱的任务类型，因此这两个结论都是从同一数据中选出的。我们提出了选择感知语义压力测试（SASST），该方法从发现任务上的执行前特征中学习任务重加权，并在独立的确认任务上评估相同的配对比较。该协议检查支持度与稳定性，对所有计划的声明使用联合统计界限，并且可以返回“无声明”的结果。我们在给定的聚类假设下证明了条件渐近有效性。一项四十聚类的审计发现了高斯方法的覆盖不足问题以及Bonferroni t界限的保守性。在一项480回合的τ-bench研究中，发现阶段获得的3.75点优势在确认阶段完全消失。在另一项第二模型的研究中，同样既未确认工作流的收益，也未确认稳定的压力规则。

    arXiv:2608.30916v1 Announce Type: new  Abstract: Agent evaluations often use one benchmark to choose a workflow and then search for task types where its advantage weakens, so both conclusions are selected from the same data. We introduce Selection-Aware Semantic Stress Testing (\SASST{}), which learns a task reweighting from pre-execution features on discovery tasks and evaluates the same paired comparison on separate confirmation tasks. The protocol checks support and stability, uses joint bounds for all planned claims, and can return no claim. We prove conditional asymptotic validity under stated cluster assumptions. A forty-cluster audit finds Gaussian undercoverage and conservative Bonferroni $t$ bounds. In one 480-episode $\tau$-bench study, a $3.75$ point discovery gain vanished on confirmation. A second-model study likewise confirmed neither a workflow benefit nor a stable stress rule.
    
[^7]: 随机对象Fréchet回归的边际坐标检验

    Marginal Coordinate Test for Fr\'echet Regression with Random Objects

    [https://arxiv.org/abs/2608.30644](https://arxiv.org/abs/2608.30644)

    该论文提出了一种针对随机对象Fréchet回归的边际坐标检验方法，通过半监督设计构建无需响应残差的核条件均值依赖性（KCMD）U统计量，并建立了加权中心化卡方零极限、野自助法有效性、一致性和局部功效等完整理论性质。

    

    我们针对具有欧氏预测变量和可分度量空间中随机对象响应的回归，开发了一种边际坐标检验方法。其目标是检验某个预测变量在给定其余预测变量的条件下，是否为响应提供额外信息。在半监督设计中，未标记样本用于估计预测变量的条件均值，而独立的已标记样本则保留用于统计推断。所得残差与乘积空间核相结合，构成一个无需响应残差的核条件均值依赖性（KCMD）U统计量。基于恒等映射的主检验针对必要的条件均值限制，而多重变换扩展则探测更广泛的备择假设。我们建立了加权中心化卡方零假设极限分布、野自助法（wild bootstrap）的有效性、对固定可检测备择假设的一致性，以及在均值元素备择假设下的局部功效。对于同时性（后续内容被截断）……

    arXiv:2608.30644v1 Announce Type: cross  Abstract: We develop a marginal coordinate test for regression with Euclidean predictors and a random-object response in a separable metric space. The goal is to test whether a predictor provides additional information about the response conditional on the remaining predictors. In a semi-supervised design, an unlabeled sample is used to estimate predictor conditional means, while an independent labeled sample is reserved for inference. The resulting residuals are combined with a product-space kernel to form a kernel conditional mean dependence (KCMD) U-statistic without requiring a response residual. The primary identity-based test targets a necessary conditional mean restriction, while a multiple-transformation extension probes broader alternatives. We establish a weighted centered chi-square null limit, wild bootstrap validity, consistency against fixed detectable alternatives, and local power under mean-element alternatives. For simultaneous 
    
[^8]: 多分类中的信息性标签缺失：信息几何与超额风险

    Informative Label Missingness in Multiclass Classification Information Geometry and Excess Risk

    [https://arxiv.org/abs/2608.30561](https://arxiv.org/abs/2608.30561)

    本文为多分类中的信息性标签缺失建立了基于似然的通用理论，通过有效信息分解与超额风险的二次展开，提出分类加权广义特征值准则，证明当缺失标签模式携带信息时，部分标注分类器可以获得比完全标注分类器更小的渐近分类风险。

    

    信息性标签缺失可能会改变完全标注分类器与部分标注分类器之间通常的效率排序，因为缺失标签的模式本身可能携带关于分类模型的信息。我们在参数化多分类框架下为这一现象建立了一套基于似然的通用理论。通过一个有效信息分解，将因类成员信息不可得而损失的信息与缺失标签机制所贡献的信息分离开来。随后，我们推导了插入式超额风险在多分类贝叶斯边界的活跃成对面上的一类二次展开，表明分类效率取决于信息增益与损失如何与扰动决策边界的方向相一致。由此得到一个分类加权的广义特征值准则，在该准则下，信息性的部分标注分类即使不能全局优于完全标注分类器，也可能具有更小的渐近分类风险。

    arXiv:2608.30561v1 Announce Type: cross  Abstract: Informative label missingness can change the usual efficiency ordering between completely and partially labelled classifiers because the pattern of missing labels may itself carry information about the classification model. We develop a general likelihood-based theory for this phenomenon in parametric multiclass classification. An efficient-information decomposition separates information lost through unavailable class memberships from information contributed by the missing-label mechanism. We then derive a quadratic expansion of plug-in excess risk over the active pairwise faces of the multiclass Bayes boundary, showing that classification efficiency depends on how information gains and losses align with directions that perturb the decision boundary. This yields a classification-weighted generalized-eigenvalue criterion under which informative partial classification may have smaller asymptotic classification risk without globally domin
    
[^9]: 当鞅永不停止触发：真实预测流上的任意时点有效门控

    When the Martingale Never Stops Firing: Anytime-Valid Gating on Real Forecast Streams

    [https://arxiv.org/abs/2608.30502](https://arxiv.org/abs/2608.30502)

    该论文通过预设的案例研究，实测了当监测器在真实、非可交换的预测流上运行并与被纠正的学习器构成反馈回路时，保形检验鞅基于Ville不等式的任意时点有效误报保证是否依然成立。

    

    机器学习系统正越来越多地在运行过程中被实时纠正，而何时进行干预的决策也越来越多地被交给统计监测器来完成。任意时点有效推断承诺提供可在任何时刻据以行动的证据，这正是该场景所需要的保证，并且它正从理论走向实际部署的监测系统。保形检验鞅是实现变化检测的工具，而Ville不等式在可交换数据上为其误报概率设定了上限。然而，这一保证是有条件的：只有当被监测的数据流满足可交换性时，部署系统才能继承该保证。这一前提恰恰在这些监测器最有用武之地的场景中最难满足——即在存在相关性的数据上，以及当监测器处于会修改其读取分数的学习器的反馈回路之中。而且，这一前提在实践中很少被测量。我们在一个预先设定的案例研究中对其进行了测量：在该研究中，这样的监测器对一个卡尔曼适配器的在线更新进行门控，该适配器用于在五个（预测数据集上）纠正冻结的时间序列基础模型……

    arXiv:2608.30502v1 Announce Type: new  Abstract: Machine learning systems are increasingly corrected while they run, and the decision of when to intervene is increasingly delegated to statistical monitors. Anytime-valid inference promises evidence that can be acted on at any moment, exactly the guarantee this setting needs, and it is moving from theory into deployed monitoring. Conformal test martingales are the change-detection instrument, and Ville's inequality caps their false-alarm probability on exchangeable data. The guarantee is conditional. A deployment inherits it only if the stream it monitors behaves exchangeably. The premise is hardest to satisfy where these monitors are most useful, on dependent data and inside loops where the monitor modifies the learner whose scores it reads. It is also rarely measured. We measure it in a pre-specified case study, where such a monitor gates the online updates of a Kalman adapter correcting frozen time-series foundation models on five for
    
[^10]: 混杂伪装成改进：基于12.9万患者登记数据的卒中抗栓治疗离线强化学习系统性评估

    Confounding Masquerading as Improvement: A Systematic Evaluation of Offline Reinforcement Learning for Stroke Antithrombotic Treatment in a 129,000-Patient Registry

    [https://arxiv.org/abs/2608.30442](https://arxiv.org/abs/2608.30442)

    本研究通过在12.9万例卒中患者登记数据上系统评估多种离线强化学习算法与奖励设计，发现表观的策略改进实际源于奖励中内嵌的基线严重程度与预后混杂，去混杂后改进不再统计显著，警示临床离线RL评估中混杂偏倚可能被误判为策略优于医生。

    

    近期的离线强化学习（RL）研究报告了在临床结局上优于医生决策的策略。我们对来自全国性登记数据库（N = 129,033）中44,894名2018年后急性缺血性卒中患者，系统性地对五种离线RL算法族和14种奖励设计进行了部分交叉评估。标准拟合Q评估（FQE）得出表观的策略改进估计值为+0.0069；加入早期神经功能恶化惩罚后增至+0.0101。我们识别出“奖励内嵌混杂”现象，即代理终末奖励不仅编码治疗疗效，还编码了基线严重程度和预后。2×2析因分析发现，终末奖励混杂占观察到的信号变化的218.6%，因此去除该混杂后估计值会越过零值（过度校正）。经过受双重机器学习（DML）启发的GBM奖励残差化处理后，FQE估计值衰减至+0.0033（p = 0.132），完全去混杂后为+0.0025（p = 0.291）。

    arXiv:2608.30442v1 Announce Type: new  Abstract: Recent offline reinforcement learning (RL) studies report policies that outperform physician decisions on clinical outcomes. We conduct a systematic, partially crossed evaluation of five offline RL algorithm families and 14 reward designs in 44,894 post-2018 acute ischemic stroke patients from a nationwide registry (N = 129,033).   Standard Fitted Q-Evaluation (FQE) yields an apparent policy-improvement estimate of +0.0069; adding an Early Neurological Deterioration penalty increases it to +0.0101. We identify reward-embedded confounding, in which a proxy terminal reward encodes baseline severity and prognosis as well as treatment efficacy. A 2 x 2 factorial analysis finds that terminal reward confounding accounts for 218.6% of the observed signal change, so its removal overshoots the null.   After DML-inspired GBM reward residualization, the FQE estimate attenuates to +0.0033 (p = 0.132), and full deconfounding yields +0.0025 (p = 0.291
    
[^11]: 基于神经细胞自动机的偏微分方程时间步进学习

    Learning PDE Time-Stepping with Neural Cellular Automata

    [https://arxiv.org/abs/2608.30328](https://arxiv.org/abs/2608.30328)

    本文提出基于神经细胞自动机（NCA）的PDE替代模型，通过学习在每个网格单元重复应用的局部同质更新规则来模拟微分算子的局部性，在五个经典PDE上于超出训练时间域两倍的时间范围内取得了最优的长时间预测精度。

    

    经典的偏微分方程（PDE）数值求解器在不同初始条件下反复求解时计算代价高昂，这促使人们需要学习型替代模型。本文提出了一种可训练的基于神经细胞自动机（NCA）的替代模型，用于学习PDE的长时间动力学。与一次性将整个初始场映射到完整轨迹的做法不同，我们提出的模型学习一个小型、局部、同质的更新规则，该规则在每个网格单元上以相同方式被重复应用，这与微分算子的局部性相呼应。我们在五个经典偏微分方程（热传导方程、对流方程、Burgers方程、Allen-Cahn方程和Fisher-KPP方程）上，将该框架与三个基线方法进行对比：PDE-Net、改进的物理信息神经网络（PINN）以及傅里叶神经算子（FNO），评估时间域为训练时间域的两倍。所提出的模型实现了最低的长时间相对误差。

    arXiv:2608.30328v1 Announce Type: new  Abstract: Classical numerical solvers for partial differential equations (PDEs) are computationally expensive to solve repeatedly across varying initial conditions, motivating the need for learned surrogates. In this paper, we propose a trainable Neural Cellular Automata (NCA) based surrogate model for learning long time PDE dynamics. Rather than mapping an entire initial field to a full trajectory in one shot, our proposed model learns a small, local, homogeneous update rule that is applied identically and repeatedly at every grid cell, mirroring the locality of differential operators. We benchmark this framework against three baselines: PDE - Net, a modified physics-informed neural network (PINN), and a Fourier Neural Operator (FNO), on five canonical PDEs (heat, advection, Burgers, Allen - Cahn, and Fisher - KPP), evaluated at temporal domain two times beyond the training temporal domain. The proposed model achieves the lowest long-horizon rela
    
[^12]: 从训练样本估计沿非凸梯度流的总体风险曲线

    Estimating Population-Risk Curves Along Nonconvex Gradient Flows from the Training Sample

    [https://arxiv.org/abs/2608.30261](https://arxiv.org/abs/2608.30261)

    该论文提出Flow-ALO方法，从训练样本估计非凸梯度流的条件总体风险曲线，并在温和条件下给出显式的 $(n-1)^{-2}$ 误差界，且无需Hessian可逆。

    

    我们从训练样本出发，估计一个已实现的光滑非凸梯度流的条件总体风险曲线。流近似留一法（Flow-ALO）传播删除响应，并在近似删除路径上评估被省略的观测点。风险曲线误差被分解为响应近似误差、精确留一波动以及删除到全量风险的传递三部分。在每个固定的有限时间区间上，有界的中心化训练损失梯度、单侧Hessian下界、局部Lipschitz的Hessian以及严格的管道闭合条件，共同给出了删除响应误差的显式 $(n-1)^{-2}$ 界。有界的评估损失梯度将该删除响应界传递到得分上，且无需Hessian可逆。直接的一阶折刀相消与精确留一集中性分别控制删除到全量风险的传递与波动部分，从而完成了条件总体风险曲线的完整恢复。

    arXiv:2608.30261v1 Announce Type: cross  Abstract: We estimate the conditional population-risk curve of a realized smooth nonconvex gradient flow from the training sample. Flow approximate leave-one-out (Flow-ALO) propagates a deletion response and evaluates omitted observations at approximate deleted paths. The risk-curve error decomposes into response approximation, exact-LOO fluctuation, and deletion-to-full risk transfer. On each fixed finite horizon, bounded centered training-loss gradients, a one-sided Hessian lower bound, locally Lipschitz Hessians, and a strict tube-closure condition yield an explicit $(n-1)^{-2}$ bound for the deletion-response error. Bounded evaluation-loss gradients transfer the deletion-response bound to the score without requiring the Hessian to be invertible. Direct first-order jackknife cancellation and exact-LOO concentration control deletion-to-full risk transfer and fluctuation, respectively, completing recovery of the conditional population-risk curv
    
[^13]: 基于情境一致性风险度量的多类别多群体分类问题中的公平性研究

    Fairness in multi-class multi-group classification problems via contextial coherent risk measures

    [https://arxiv.org/abs/2608.30223](https://arxiv.org/abs/2608.30223)

    该论文提出了一种基于一致性风险度量理论的公平分类器设计方法，能够处理向量值敏感属性导致的多类别、多重叠群体分类中的公平性问题，同时保护个体权利，并提供了可扩展且对数据损坏和稀缺具有鲁棒性的专门数值求解方法。

    

    我们提出了一种针对多类别分类问题的新型公平分类器设计，该问题涉及向量值的敏感属性。在这种情形下，每个敏感属性具有多个取值，并形成了与公平性考量相关的多个群体。这些群体自然是相互重叠的，因此还应当分析各因素之间的交互作用。此外，借助该分类进行决策的决策者不应以牺牲个体权利为代价来满足群体层面的公平性指标。我们提出了一种利用一致性风险度量理论与方法来解决公平性挑战的途径。进一步地，我们提出了一种专门的数值方法来求解由此产生的优化问题。该方法在观测数量增加时具有良好的可扩展性。此外，我们注意到所得到的分类器对于数据损坏或数据稀缺的情况具有鲁棒性。

    arXiv:2608.30223v1 Announce Type: cross  Abstract: We propose a new design of fair classifiers for multi-class classification problems in the presence of vector-valued sensitive attributes. In that scenario each sensitive attribute has multiple values and forms several groups relevant to the fairness consideration. Naturally those groups are overlapping and one should also analyze the interaction of factors. Additionally, the decision makers aided by the classification should not violate individual rights at the expense of satisfying fairness metrics at the group level. We propose an approach using the theory and methods of coherent measures of risk aiming at resolving the fairness challenges. Further, we propose a specialized numerical method for solving the resulting optimization problem. The method scales well with the increase of the number of observations. Additionally, we note that the obtained classifier is robust with respect to corrupted data or to situation when data is scarc
    
[^14]: 基于密度幂散度度量的鲁棒K-means聚类

    Robust K-means Clustering using the Density Power Divergence Measure

    [https://arxiv.org/abs/2608.30093](https://arxiv.org/abs/2608.30093)

    提出了一种结合密度幂散度与马氏距离的鲁棒K-means聚类方法MK-means DPD及其保证有限步收敛的变体DC-MK-means DPD，并设计了两个抗异常值的聚类评估指标。

    

    我们提出了一种鲁棒聚类方法 MK-means DPD，该方法利用密度幂散度（DPD）度量结合马氏距离来估计聚类中心和协方差矩阵，使其能够抵抗异常值并适应异质的椭圆形聚类，这与经典的K-means算法不同。由于基于马氏距离的K-means缺乏一般性的收敛保证，我们进一步引入了一个收敛变体——密度一致 MK-means DPD（DC-MK-means DPD），该方法从逐点DPD损失的角度重新定义了聚类分配步骤。我们证明了一个形式化定理，表明所得算法可在有限步数内收敛。此外，我们还提出了两个新的鲁棒内部评估指标——中位数Davies-Bouldin指数和截尾Calinski-Harabasz指数，以确保性能比较本身不会被异常值扭曲。所提方法的有效性已在模拟（数据上得到验证，原文此处截断）。

    arXiv:2608.30093v1 Announce Type: cross  Abstract: We introduce a robust clustering method, MK-means DPD, that estimates cluster centers and covariance matrices using density power divergence (DPD) measures combined with Mahalanobis distance, making it resistant to outliers and adaptable to heterogeneous, elliptical clusters, unlike the classical K-means algorithm. Since Mahalanobis distance-based K-means lacks a general convergence guarantee, we further introduce a convergent variant, Density-Consistent MK-means DPD (DC-MK-means DPD), which redefines the cluster assignment step in terms of a pointwise DPD loss. We prove a formal theorem establishing that the resulting algorithm converges in a finite number of steps. We also propose two new robust internal evaluation indices, a Median Davies-Bouldin Index and a Trimmed Calinski-Harabasz Index, to ensure that performance comparisons are not themselves distorted by outliers. The efficacy of the proposed methods is demonstrated on simulat
    
[^15]: 通过词元预测学习表示：几何、近似与下游保证

    Learning Representations through Token Prediction: Geometry, Approximation, and Downstream Guarantees

    [https://arxiv.org/abs/2608.30072](https://arxiv.org/abs/2608.30072)

    该论文建立了一个统计理论框架，证明词元预测会依据上下文分布间的Hellinger距离相似性来组织词元嵌入，并由此给出表示几何、编码器近似与下游任务性能之间的理论保证。

    

    词元预测是现代语言模型的核心预训练目标。尽管其在实践中取得了巨大成功，但为什么词元预测能够学习到广泛有用的表示仍缺乏完整的理论理解。我们开发了一个统计框架，将词元预测与表示几何、编码器近似以及下游性能联系起来。在softmax预测头下，我们证明了准确的词元预测会依据不同词元类型所出现的上下文分布之间的相似性（以Hellinger距离衡量）来组织词元嵌入，其显式误差由预测精度和词元频率决定。同时，上下文表示为目标词元相对于这些嵌入的条件分布提供了低维坐标。我们进一步引入了一个自洽性原则，表明共享表示块的重复应用可以渐进地……（原文摘要在此处被截断）

    arXiv:2608.30072v1 Announce Type: cross  Abstract: Token prediction is a central pre-training objective for modern language models. Despite its empirical success, why token prediction learns broadly useful representations remains incompletely understood. We develop a statistical framework connecting token prediction with representation geometry, encoder approximation, and downstream performance. Under a softmax prediction head, we show that accurate token prediction organizes token embeddings according to similarities between the distributions of contexts in which different token types appear, as measured by Hellinger distance, with explicit errors governed by prediction accuracy and token frequency. Meanwhile, the contextual representation provides a low-dimensional coordinate for the conditional distribution of the target token relative to these embeddings. We further introduce a self-consistency principle showing that repeated applications of a shared representation block can progre
    
[^16]: 一种联合建模缺失性、测量误差与异质性的深度潜变量框架

    A Deep Latent Variable Framework for Jointly Modeling Missingness, Measurement Error, and Heterogeneity

    [https://arxiv.org/abs/2608.30040](https://arxiv.org/abs/2608.30040)

    提出了一种基于深度潜变量表示的统一概率框架，通过层次树路由变分自编码器联合处理缺失数据、测量误差和群体异质性三类问题，并借助重新汇聚路由机制在子群体间共享参数以提升统计效率。

    

    缺失数据、测量误差和群体异质性是现代观察性研究与机器学习应用数据分析中普遍存在的挑战。尽管这些问题经常共存并相互作用，但现有工作往往将它们分开处理。我们提出了一个统一的概率框架，利用深度潜变量表示来联合应对这些问题。所提出的方法将一种新颖的层次树路由变分自编码器与模式感知的潜变量表示以及基于校准的去噪机制相结合。该框架兼容包括MCAR、MAR和MNAR在内的多种缺失数据机制，同时能够学习子群体特有和全局共享的潜变量结构。所引入的重新汇聚路由机制使选择性参数能够在相关子群体之间共享，从而提供了灵活性并提升了统计效率。仿真……

    arXiv:2608.30040v1 Announce Type: cross  Abstract: Missing data, measurement error, and population heterogeneity are pervasive challenges in analyzing data arising from modern observational studies and machine learning applications. Although these problems frequently coexist and interact, they are often treated separately in existing works. We propose a unified probabilistic framework that jointly addresses these issues utilizing deep latent variable representation. The proposed method integrates a novel hierarchical tree-routed variational autoencoder with pattern-aware latent representations and calibration-based denoising. The framework accommodates missing data mechanisms, including MCAR, MAR, and MNAR, while simultaneously learning subgroup-specific and globally shared latent structure. The introduced reconvergent routing mechanism enables selective parameters to be shared across related subpopulations, which offers flexibility as well as improved statistical efficiency. Simulatio
    
[^17]: 神经常微分方程增强的线性混合效应模型：用于估计时变协变量与标志物轨迹之间的复杂关联模式

    Neural ODE enhanced linear mixed effect models for estimating complex association patterns of time-varying covariates with the marker trajectory

    [https://arxiv.org/abs/2608.29714](https://arxiv.org/abs/2608.29714)

    该论文提出Neural ODE-LMM模型，将神经常微分方程嵌入线性混合效应框架，利用学习到的向量场把协变量轨迹编码为连续时间潜在状态来驱动固定效应和随机效应设计，从而在保留经典似然推断的同时，灵活地捕捉时变协变量与结局标志物轨迹之间的复杂累积性关联模式。

    

    纵向队列研究产生的重复测量数据使我们能够评估暴露因素与健康结局之间随时间变化的关联模式。经典的线性混合效应模型（LMMs）能够适应多种多样的关联模式，同时可以处理不规则间隔、部分观测的测量数据。但是，这类模型要求分析者预先指定暴露历史与结局之间关联的函数形式。我们提出了Neural ODE-LMM，该方法将神经常微分方程嵌入到线性混合效应框架之中：通过一个学习得到的向量场，将协变量轨迹编码为连续时间的潜在状态，该潜在状态同时驱动固定效应和随机效应的设计，同时保留了标准的LMM观测模型。这使得模型在保留经典的基于似然的统计推断的同时，能够灵活地学习复杂的、可能具有累积效应的协变量影响。所有参数均通过最大化p……（原文摘要在此处截断）

    arXiv:2608.29714v1 Announce Type: cross  Abstract: Longitudinal cohort studies produce repeated data that enable the assessment of time-varying association patterns between exposures and health outcomes. Classical linear mixed-effects models (LMMs) can accommodate a large variety of association patterns while accounting for the irregularly spaced, partially observed measurement. But they require the analyst to pre-specify the functional form linking the exposure history to the outcome. We propose the Neural ODE-LMM, which embeds a Neural Ordinary Differential Equation (Neural ODE) within the linear mixed-effects framework: a learned vector field encodes covariate trajectories into a continuous-time latent state that drives both the fixed- and random-effect design, while preserving the standard LMM observation model. This retains classical likelihood-based inference while learning complex, potentially cumulative, covariate effects flexibly. All parameters are estimated by maximising a p
    
[^18]: 哪种大语言模型适合哪项工作？不确定评估下的预算模型分配

    Which LLM for Which Work? Budgeted Model Allocation under Uncertain Evaluation

    [https://arxiv.org/abs/2608.29560](https://arxiv.org/abs/2608.29560)

    该论文研究了在模型质量评估存在不确定性时，如何为拥有固定AI预算的公司在各项重复性工作上分配大语言模型，指出质量表估计的两大失效来源——模型很少在相同工作上被比较、记录的分数只是代理指标——并证明购买更多评估也无法消除后者的不确定性。

    

    一家拥有固定人工智能（AI）预算的公司必须决定由哪个大语言模型（LLM）来处理每项重复性工作。它所缺乏的是质量表，即每个模型在每项工作上的表现如何。给定该质量表，决策就是一个多选择背包问题，求解是常规操作，因此难点在于估计该质量表，而这种估计会在两个方面失效：模型很少在相同的工作上进行比较，且记录的分数通常是代理指标，而非公司真正看重的结果。因果方法和离策略方法可以修复第一个问题，但依赖于第二个问题；而评估器验证方法可以估计第二个问题，但止步于决策之前。更糟糕的是，购买更多的重新评估也无法解决第二个问题：随机化控制的是哪些请求被评分，而不是分数如何产生，因此无论购买多少评估，质量表仍保持不确定。然而，即使当……（原文截断），部署决策可能仍然是确定的。

    arXiv:2608.29560v1 Announce Type: new  Abstract: A company with a fixed artificial intelligence (AI) budget must decide which large language model (LLM) handles each recurring workload. What it lacks is the quality table, how well each model performs on each workload. Given that table, the decision is a multiple-choice knapsack problem and is routine to solve, so estimating it is the difficulty, and that estimation fails in two ways. Models are rarely compared on the same work, and the recorded score is usually a proxy rather than the outcome the company values. Causal and off-policy methods repair the first but condition on the second, while evaluator-validation methods estimate the second but stop short of the decision. Worse, buying more re-evaluation cannot settle the second: randomization governs which requests are scored, not how a score is produced, so the table stays uncertain however much evaluation is purchased. Yet the deployment decision may still be determined even when th
    
[^19]: 基于神经网络代理模型的树脂传递模塑在线闸口驱动流动控制

    Online Gate-Driven Flow Control in Resin Transfer Moulding Using a Neural-Network Surrogate

    [https://arxiv.org/abs/2608.29521](https://arxiv.org/abs/2608.29521)

    该论文提出了一种结合迭代扩展卡尔曼滤波与神经网络代理模型的实时估计与控制策略，利用压力传感器数据在线估计边缘渗流强度并优化辅助闸口压力，从而防止树脂传递模塑过程中干斑的形成。

    

    在树脂传递模塑工艺中，在树脂前沿到达出口通气口之前，纤维预制件必须完全饱和，以防止干斑的形成。在实际操作中，由于边缘渗流效应的影响，流动前沿很少能够均匀推进。我们提出了一种将估计与控制相结合的策略来解决这一问题。我们利用充填过程中采集的压力传感器数据，通过迭代扩展卡尔曼滤波器来估计未知的边缘渗流强度，并同时优化辅助闸口压力，以防止树脂在完全饱和之前到达通气口。为了使该方法能够实时运行，我们用神经网络代理模型替代了计算代价高昂的有限元-控制体积模型，并采用贝叶斯近似误差框架来刻画代理模型与原始模型之间的差异。为了使代理模型在时间依赖控制输入下的训练可行……

    arXiv:2608.29521v1 Announce Type: cross  Abstract: In resin transfer moulding, complete saturation of the fibre preform is necessary before the resin front reaches the outlet vent(s), to prevent dry-spot formation. In practice, the flow front rarely advances uniformly due to race-tracking effects. We propose a combined estimation and control strategy to address this issue. We use pressure-sensor data collected during filling to estimate the unknown race-tracking strengths via an iterated extended Kalman filter, and to simultaneously optimise auxiliary gate pressures to prevent the resin from arriving at the vent before complete saturation occurs. To make this approach feasible in real time, we replace the expensive finite element-control volume model with neural network surrogate models. The Bayesian approximation error framework is used to account for the discrepancy between the surrogates and the original model. To make the training of the surrogates feasible with time-dependent cont
    
[^20]: 决定何时重新决策：分布偏移下的运行次优性检验

    Deciding When to Decide: Testing Operational Suboptimality Under Distributional Shift

    [https://arxiv.org/abs/2608.29465](https://arxiv.org/abs/2608.29465)

    本文提出RADAR框架，通过逆优化推断潜在偏好并以决策遗憾（最优性差距）为检验目标，判断分布偏移下已部署的决策是否已实质性次优而需要重新优化，从而忽略与决策无关的变化。

    

    已部署的决策通常只优化一次并被长期保留，因为更新会带来运营、监管或切换成本。随着运行条件的变化，此类决策何时应当被重新优化？我们针对随机优化问题研究这一问题，其中目标函数的具体形式已知，但决策者的权衡偏好由一个未知的偏好参数所编码。标准的分布偏移检验与这一目标并不契合：它们可能标记出可检测但与决策无关的变化，却无法判断现有决策是否已变得实质性地次优。我们提出了RADAR（基于遗憾的决策充分性与风险评估框架），这是一个以决策为中心的框架，利用逆优化来推断潜在偏好，并检验已部署决策在当前分布下的最优性差距。通过以遗憾为检验目标，RADAR能够忽略与决策无关的分布偏移，同时检测出值得重新优化的变化。

    arXiv:2608.29465v1 Announce Type: cross  Abstract: Deployed decisions are often optimized once and retained because updates impose operational, regulatory, or switching costs. As operating conditions change, when should such decisions be re-optimized? We study this question for stochastic optimization when the objective's functional form is known but the decision maker's trade-offs are encoded by an unknown preference parameter. Standard distribution-shift tests are poorly aligned with this goal: they can flag detectable yet decision-irrelevant changes without determining whether the incumbent decision has become materially suboptimal. We propose \texttt{RADAR} (Regret-based Assessment of Decision Adequacy and Risk), a decision-focused framework that uses inverse optimization to infer latent preferences and tests the deployed decision's optimality gap under the current distribution. By targeting regret, \texttt{RADAR} ignores decision-irrelevant shifts while detecting changes that warr
    
[^21]: SS-ESOAP：面向物理信息学习的自缩放自适应预条件方法

    SS-ESOAP: Self-Scaled Adaptive Preconditioning for Physics-Informed Learning

    [https://arxiv.org/abs/2608.29448](https://arxiv.org/abs/2608.29448)

    SS-ESOAP通过在SOAP预条件方法上引入标量割线能量校正与自适应基更新及方差状态缩减，显著提升了物理信息神经网络在多数PDE基准上的训练精度与显存效率。

    

    物理信息神经网络（PINNs）常常面临病态的目标函数，这限制了高精度训练。稠密拟牛顿方法可以改善局部条件数，但需要昂贵的优化器状态，而诸如SOAP之类的Kronecker分解方法虽然可以扩展到更大的网络，但依赖于周期性的基更新。我们提出了SS-ESOAP，该方法在SOAP风格的预条件方法基础上，增加了适应Kronecker几何的标量割线能量校正，以及自适应基更新和随后的方差状态缩减。我们刻画了标量校正所诱导的方向性割线匹配，并给出了基变换之间方差状态失配的界。在八个PDE基准测试中，SS-ESOAP在其中六个基准（包括Burgers和Boussinesq）上取得了最低的最终残差，而SOAP系列基线方法在Gray-Scott和Ginzburg-Landau问题上表现更好。在Boussinesq问题上，SS-ESOAP在4.1小时内以9.2 GB的峰值显存达到了10^-5的残差……

    arXiv:2608.29448v1 Announce Type: cross  Abstract: Physics-informed neural networks (PINNs) often face ill-conditioned objectives that limit high-accuracy training. Dense quasi-Newton methods improve local conditioning but require expensive optimizer state, while Kronecker-factored methods such as SOAP scale to larger networks but rely on periodic basis updates. We introduce \method, which augments SOAP-style preconditioning with a scalar secant-energy correction adapted to Kronecker geometry and an adaptive basis update followed by variance-state downscaling. We characterize the directional secant matching induced by the scalar correction and give a bound on variance-state mismatch across basis changes. Across eight PDE benchmarks, \method attains the lowest final residual on six, including Burgers and Boussinesq, while SOAP-family baselines perform better on Gray-Scott and Ginzburg-Landau. On Boussinesq, \method reaches a residual of $10^{-5}$ in 4.1 hours with 9.2 GB peak VRAM, whil
    
[^22]: 超越信息流的内容探索：创作者供给与共享内容库

    Content Exploration Beyond the Feed: Creator Supply and the Shared Corpus

    [https://arxiv.org/abs/2608.29430](https://arxiv.org/abs/2608.29430)

    该论文通过某大型短视频平台的四项实验首次揭示了内容探索的双重价值——生产侧探索可使创作者发帖量提升8.55%，观众侧探索虽增加观看次数但减少观看时长，且探索引发的创作者供给与自然采纳会补充共享内容库，突破传统仅衡量观众侧效果的评估局限。

    

    工业级推荐系统通过有预算的探索为新内容提供初始曝光，然后依据早期表现决定后续分发。在许多短视频平台上，探索是新视频触达观众的主要途径。观众侧的测试衡量内容消费，而我们综述的已发表预算目标均忽略了创作者的反应。我们分析了某大型短视频平台上的四项实验。一项为期八个月的创作者侧消融实验发现，相对于最低基线，生产侧探索使每位创作者发布的视频数量提升8.55%，至少发布一次视频的创作者数量提升7.10%。一项预算匹配的重新分配实验提高了创作者参与度，且短期内未检测到观众侧的明显变化。一项为期一年的观众侧消融实验发现，视频观看次数增加1.74%，但观看时长减少2.13%。一次投放的观看既能创造即时的信息流价值，也可能引发有机的自然采纳，还能激励创作者供给。自然采纳与创作者供给会持续补充共享内容库，由此产生两个测量上的局限。

    arXiv:2608.29430v1 Announce Type: cross  Abstract: Industrial recommenders give new content initial views through budgeted exploration, then use early performance to decide further delivery. On many short-video platforms, exploration is the primary way new videos reach viewers. Viewer-side tests measure consumption; the published budget objectives we review omit creator response. We analyze four experiments on a major short-video platform. An eight-month creator ablation finds production exploration raises videos posted per creator by 8.55% and creators posting at least once by 7.10% relative to a minimal floor. A budget-matched reallocation raises creator participation with no detectable short-run viewer-side change. A year-long viewer ablation finds 1.74% more video views but 2.13% less view time. A delivered view creates immediate feed value, can trigger organic take-up, and can induce creator supply. Take-up and supply replenish a shared corpus, creating two measurement limits. Vie
    
[^23]: 面向不定核快速密度估计的带符号随机傅里叶特征方法

    Signed random Fourier features for fast density estimation with indefinite kernels

    [https://arxiv.org/abs/2608.29265](https://arxiv.org/abs/2608.29265)

    本文提出带符号随机傅里叶特征（SRFF）技术，将随机傅里叶特征方法从正定核推广至不定核，从而实现对大规模数据集的高效核密度估计。

    

    arXiv:2608.29265v1 公告类型：cross 摘要：核密度估计（KDE）是密度函数最基础的统计估计量之一。其在包含 $N$ 个数据点的数据集上的直接实现会产生 $\mathcal{O}(N^{2})$ 的计算成本，这对于大规模数据集而言是难以承受的。核近似技术可以将计算成本降低至 $\mathcal{O}(N)$。基于从核函数谱密度中采样的随机傅里叶特征（RFF）技术，已成为加速机器学习中核估计器的流行方法。遗憾的是，该技术仅限于正定核，而核密度估计中常用的多数核函数（如抛物线核）并不满足正定性。为克服这一限制，本文提出了带符号随机傅里叶特征（SRFF）技术。它是随机傅里叶特征的推广形式，适用于逆傅里叶变换绝对可积的不定核函数。

    arXiv:2608.29265v1 Announce Type: cross  Abstract: Kernel density estimation (KDE) is one of the most fundamental statistical estimators of density functions. Its direct implementation on a dataset of $N$ points incurs an $\mathcal{O}(N^{2})$ computational cost, which is prohibitive for large-scale datasets. Kernel approximation techniques can be applied to bring the computational cost down to $\mathcal{O}(N)$. The random Fourier features (RFF) technique, based on sampling from the spectral density of the kernel function, has become popular to speed up kernel estimators for machine learning applications. Unfortunately, it is restricted to positive definite kernels, while the majority of kernel functions popular in KDE, such as the parabolic kernel, do not satisfy this property. To overcome this limitation, this article introduces the signed random Fourier features (SRFF) technique. It is a generalization of RFF compatible with indefinite kernels whose inverse Fourier transform is absol
    
[^24]: 经验Sinkhorn势的均匀统计收敛性：对正则化参数的指数与多项式依赖

    Uniform Statistical Convergence of Empirical Sinkhorn Potentials with Exponential and Polynomial Dependence on the Regularization Parameter

    [https://arxiv.org/abs/2608.29152](https://arxiv.org/abs/2608.29152)

    该论文证明了经验Sinkhorn势在商范数下具有 $n^{-1/2}$ 的非渐近收敛速率，并通过总体Sinkhorn映射的多项式残差稳定性等几何条件，将收敛界对正则化参数的依赖从指数级改进为多项式级。

    

    我们研究了在均匀损失下熵正则最优传输势的经验Sinkhorn估计量。由于势仅在相差一个加性常数的意义下唯一，我们使用商上确界范数来度量误差，其定义为 $d_\infty([u],[v]) = \inf_{a\in\mathbb{R}}\|u-v-a\|_\infty$。对于固定的正则化参数 $\varepsilon>0$，我们建立了非渐近的 $n^{-1/2}$ 统计收敛速率。这一结果通过将Birkhoff-Hopf压缩定理与归一化核截面的熵界相结合而实现。然而，该界中的常数随 $1/\varepsilon$ 呈指数增长。为改进此结果，我们分离出了若干几何条件，在这些条件下经验估计量在保持 $n^{-1/2}$ 速率的同时，对 $1/\varepsilon$ 仅有多项式依赖。其关键要求是对总体Sinkhorn映射的多项式残差稳定性估计。我们为此提供了充分判据，包括多项式压缩条件（摘要在此处被截断）。

    arXiv:2608.29152v1 Announce Type: cross  Abstract: We study the empirical Sinkhorn estimator of the entropic optimal transport potentials under the uniform loss. Since the potentials are only unique up to additive constants, we measure the error using the quotient supremum norm, defined as $d_\infty([u],[v]) = \inf_{a\in\mathbb{R}}\|u-v-a\|_\infty$. For a fixed regularization parameter $\varepsilon>0$, we establish a non-asymptotic statistical rate of $n^{-1/2}$. This is achieved by combining the Birkhoff-Hopf contraction theorem with entropy bounds on normalized kernel sections. However, the constant in this bound grows exponentially with $1/\epsilon$. To improve this, we isolate geometric conditions under which the empirical estimator maintains the $n^{-1/2}$ rate but features polynomial dependence on $1/\varepsilon$. The key requirement is a polynomial residual-stability estimate for the population Sinkhorn map. We provide sufficient criteria for this, including a polynomial contrac
    
[^25]: PathGuide：基于在线策略传输对齐的动态无分类器引导

    PathGuide: Dynamic Classifier-Free Guidance via On-Policy Transport Alignment

    [https://arxiv.org/abs/2608.29107](https://arxiv.org/abs/2608.29107)

    提出PathGuide框架，将无分类器引导的强度选择从静态参数调整为在线策略传输问题，利用连续性方程的弱形式证明了当引导场与精确条件场弱等价时采样路径与目标条件分布一致，从而实现动态的引导优化。

    

    尽管现代生成模型在建模复杂数据方面表现出色，但在条件生成中实现精确的推理时控制仍然是一个关键挑战。无分类器引导（CFG）是实现此类控制的主要机制，然而它通常被视为一个静态的调节参数。然而，在基于流的模型中，引导强度从根本上决定了速度场以及由此产生的概率路径，这使得引导选择成为一个动态的路径优化问题。我们提出了PathGuide，一个将标量CFG选择重新表述为在线策略传输问题的框架。利用连续性方程的弱形式，我们推导出了一个具有直接路径正确性解释的选择准则：我们证明，如果引导场沿着生成的轨迹与精确的条件场弱等价，那么采样器的路径将与目标条件分布律重合。对于标量CFG，该准则产生了一个严格二次的……（原文摘要在此处被截断）

    arXiv:2608.29107v1 Announce Type: new  Abstract: While modern generative models excel at modeling complex data, precise inference-time control in conditional generation remains a critical challenge. Classifier-free guidance (CFG) is a primary mechanism for such control, yet it is typically treated as a static tuning parameter. In flow-based models, however, the guidance scale fundamentally dictates the velocity field and the resulting probability path, making guidance selection a dynamic path-optimization problem. We introduce PathGuide, a framework that reformulates scalar CFG selection as an on-policy transport problem. Leveraging the weak form of the continuity equation, we derive a selection criterion with a direct path-correctness interpretation: we prove that if the guided field is weakly equivalent to the exact conditional field along the generated rollout, the sampler's path coincides with the target conditional law. For scalar CFG, this criterion yields a strictly quadratic lo
    
[^26]: 秩约束矩阵LASSO全局极小值的精确限制等距阈值

    Sharp Restricted Isometry Thresholds for Global Minima of Rank-Restricted Matrix LASSO

    [https://arxiv.org/abs/2608.29018](https://arxiv.org/abs/2608.29018)

    本文确定了秩约束矩阵LASSO在全局极小值处实现精确恢复的锐利限制等距阈值 $\delta<\delta_{\mathrm{sharp}}(k/r_{\star})$，并证明该阈值无法进一步改进。

    

    我们确定了秩约束矩阵LASSO在全局极小值处实现恢复的精确限制等距阈值。对于目标秩 $r_{\star}$，若秩-$k$ RIP常数满足 $\delta<\delta_{\mathrm{sharp}}(k/r_{\star})$，其中当 $t<4/3$ 时 $\delta_{\mathrm{sharp}}(t)=t/(4-t)$，当 $t\ge4/3$ 时 $\delta_{\mathrm{sharp}}(t)=\sqrt{(t-1)/t}$，则对于所有 $\lambda\gtrsim\|\mathcal{A}^{*}(\xi)\|_{\mathrm{op}}$ 以及每个搜索秩 $r\ge r_{\star}$，每个全局极小值点的Frobenius误差均满足 $\lesssim\sqrt{r_{\star}}\lambda$。这些常数仅依赖于RIP常数和 $t=k/r_{\star}$，特别地，与搜索秩无关。当秩约束不起作用时，该结果可特化为普通的凸矩阵LASSO。我们还针对稀疏性约束向量LASSO获得了类似的结果。反过来，我们证明阈值 $\delta<\delta_{\mathrm{sharp}}(k/r_{\star})$ 无法改进，原因是……

    arXiv:2608.29018v1 Announce Type: cross  Abstract: We determine the sharp restricted isometry threshold for recovery at global minima of the rank-restricted matrix LASSO. For target rank $r_{\star}$, if the rank-$k$ RIP constant satisfies $\delta<\delta_{\mathrm{sharp}}(k/r_{\star})$, where $\delta_{\mathrm{sharp}}(t)=t/(4-t)$ for $0<4/3$ and $\delta_{\mathrm{sharp}}(t)=\sqrt{(t-1)/t}$ for $t\ge4/3$, then every global minimizer has Frobenius error $\lesssim\sqrt{r_{\star}}\lambda$ for all $\lambda\gtrsim\|\mathcal{A}^{*}(\xi)\|_{\mathrm{op}}$ and at every search rank $r\ge r_{\star}$. The constants depend only on the RIP constant and $t=k/r_{\star}$, and in particular are independent of the search rank. When the rank restriction is inactive, the result specializes to the ordinary convex matrix LASSO. We also obtain the analogous results for sparsity-restricted vector LASSO. Conversely, we show that the threshold $\delta<\delta_{\mathrm{sharp}}(k/r_{\star})$ cannot be improved, due to t
    
[^27]: Jigsaw-CRL：从碎片化的多客户端干预中恢复全局潜在因果序

    Jigsaw-CRL: Recovering Global Latent Causal Order from Fragmented Multi-Client Interventions

    [https://arxiv.org/abs/2608.28991](https://arxiv.org/abs/2608.28991)

    提出Jigsaw-CRL框架，首次在碎片化多客户端干预设置下（每个客户端仅能访问和干预部分潜在变量），通过组装客户端特定的结构片段来恢复全局潜在因果序。

    

    因果表征学习（CRL）旨在从高维观测中恢复潜在因果变量及其结构关系。现有的CRL方法通常假设所有环境都定义在相同的潜在变量之上，或至少共享一个共同的潜在表征空间。我们研究了一种碎片化的多客户端设置，其中多个客户端与同一个全局潜在因果系统交互，但每个客户端只能访问和干预潜在变量的一个子集。在这种情形下，对未被使用的潜在变量进行边缘化会产生双向边，因此单个客户端不再能拥有节点级的潜在因果图，必须通过组装客户端特定的结构片段来恢复全局潜在因果序。我们提出了Jigsaw-CRL，一个从这种碎片化干预中恢复全局潜在因果序的框架。在软干预下，精度矩阵之间的差异……

    arXiv:2608.28991v1 Announce Type: cross  Abstract: Causal representation learning (CRL) aims to recover latent causal variables and their structural relations from high-dimensional observations. Existing CRL methods typically assume that all environments are defined over the same latent variables, or at least share a common latent representation space. We study a fragmented multi-client setting, where multiple clients interact with the same global latent causal system but each client only accesses and intervenes on a subset of the latent variables. In this regime, marginalizing unused latent variables induces bidirected edges, so a single client no longer admits a node-wise latent causal graph, and the global latent causal order must be recovered by assembling client-specific structural fragments. We propose \textbf{Jigsaw-CRL}, a framework for recovering global latent causal order from such fragmented interventions. Under soft interventions, differences between precision matrices acro
    
[^28]: 积参考离散扩散的信息几何：交互增长复杂度与最优调度

    The information geometry of product-reference discrete diffusion: Interaction growth complexity and optimal scheduling

    [https://arxiv.org/abs/2608.28949](https://arxiv.org/abs/2608.28949)

    该论文提出了“交互增长复杂度（IGC）”这一基于信息几何的路径度量，可精确刻画积参考离散扩散采样器的KL离散化误差与迭代复杂度，并据此设计出复杂度更低的最优步长调度方案。

    

    我们研究了一类用于从离散分布中采样的积参考扩散算法。我们证明，其采样性能可以通过一种基于路径的数据几何度量来刻画，我们将其称为交互增长复杂度（Interaction Growth Complexity, IGC）。我们证明，一个二元IGC核可以精确表示KL离散化误差以及一个简单的一步上界。更简单的一元IGC密度则可用于研究步长选择对获得KL散度下ε-精确样本所需迭代复杂度的影响。以对数平方可靠性赔率等距步长遍历路径的采样器，其性能取决于IGC的总质量；而经过精细设计的步长选择则具有更低的复杂度，该复杂度取决于一个平方根泛函。在细网格极限下，这两种刻画都变得精确紧致。我们还允许使用一般的积参考分布。

    arXiv:2608.28949v1 Announce Type: cross  Abstract: We study a class of product-reference diffusion algorithms for sampling from a discrete distribution. We show that their sampling performance can be characterized using a path-based measure of data geometry that we call the interaction growth complexity (IGC). We show that a bivariate IGC kernel gives an exact representation of both the KL discretization error and a simple one-step upper bound. The simpler univariate IGC density can be used to study the effect of stepsize choices on the iteration complexity required to obtain $\epsilon$-accurate samples in KL divergence. Samplers that traverse the path with equi-spaced steps in log-squared-reliability-odds have performance that depends on the aggregate IGC mass, whereas refined choices of stepsizes have a lower complexity depending on a square-root functional. In the fine-grid limit, both of these characterizations become sharp. We also allow general product reference distributions and
    
[^29]: 基于量子信号处理的表示学习

    Representation Learning with Quantum Signal Processing

    [https://arxiv.org/abs/2608.28828](https://arxiv.org/abs/2608.28828)

    该论文将量子信号处理（QSP）确立为表示学习的可解量子模型，精确计算了其量子神经正切核的统计特性，并证明了稀疏数据下完整非线性梯度流收敛于可积标量流以及普遍成立的有限深度速度极限。

    

    表示学习始于训练改变了定义数据间相似性的特征之时，而冻结核模型只能对固定几何进行重新加权。我们建立了量子信号处理（QSP）作为表示学习机制的可解量子模型。在任意深度下，我们计算了其量子神经正切核的精确均值与方差，揭示了一种依赖于输入的角几何结构，即使在底层幺正算子趋近Haar随机性时，其对角线仍保持非自平均特性。我们还为完整的非线性梯度流证明了稀疏数据保证，无需冻结或对核进行系综平均：实际动力学收敛于一个具有时间依赖核闭包的可积标量流，并给出显式收敛时间。有限深度速度极限对每个数据集和轨迹均成立。在更高的数据密度下，数值结果表明存在超越标量和冻结核描述的耦合演化。

    arXiv:2608.28828v1 Announce Type: cross  Abstract: Representation learning begins when training changes the features that define similarity between data. A frozen-kernel model only reweights a fixed geometry. We establish quantum signal processing (QSP) as a solvable quantum model of the representation-learning regime. At arbitrary depth, we compute the exact mean and variance of its quantum neural tangent kernel, revealing an input-dependent angular geometry whose diagonal remains non-self-averaging even when the underlying unitary approaches Haar randomness. We also prove a sparse-data guarantee for the full nonlinear gradient flow without freezing or ensemble-averaging the kernel: the realized dynamics converges to an integrable scalar flow with a time-dependent kernel closure and explicit convergence times. A finite-depth speed limit holds for every data set and trajectory. At higher data density, numerical results show coupled evolution beyond both the scalar and frozen-kernel des
    
[^30]: 朗之万正则化SVGD的定量目标收敛与时间一致的混沌传播

    Quantitative Target Convergence and Uniform-in-Time Propagation of Chaos for Langevin-Regularized SVGD

    [https://arxiv.org/abs/2608.28827](https://arxiv.org/abs/2608.28827)

    该论文首次为朗之万正则化的斯坦变分梯度下降（SVGD）建立了定量目标收敛率与时间一致的混沌传播理论，证明在目标分布满足对数Sobolev不等式时算法可获得指数级最后迭代收敛。

    

    我们为朗之万正则化的斯坦变分梯度下降建立了到目标分布的定量收敛性以及时间一致的混沌传播。其中斯坦相互作用相对于受限的朗之万漂移无需很小，且通常并不产生收缩的粒子耦合。在平均场层面，斯坦分量与朗之万分量分别在核诱导的斯坦几何与2-Wasserstein几何中耗散相同的相对熵，由此产生平方核斯坦差异和相对Fisher信息。在目标分布满足对数Sobolev不等式的条件下，这给出了指数级的最后迭代收敛。我们还推导了相对于乘积目标分布的有限粒子熵恒等式，从而得到经验测度随时间的指数收敛（直至多项式量级的采样误差）。对于混沌传播，我们发展了两种互补的有限时间方法：同步耦合结合指数矩估计……

    arXiv:2608.28827v1 Announce Type: cross  Abstract: We establish quantitative convergence to the target and uniform-in-time propagation of chaos for Langevin-regularized Stein variational gradient descent. The Stein interaction need not be small relative to the confining Langevin drift and does not generally yield a contractive particle coupling. At the mean-field level, the Stein and Langevin components dissipate the same relative entropy in the kernel-induced Stein and $2$-Wasserstein geometries, producing the squared kernel Stein discrepancy and relative Fisher information. Under a log-Sobolev inequality for the target, this yields exponential last-iterate convergence. We also derive a finite-particle entropy identity relative to the product target, giving exponential-in-time convergence of the empirical measure up to polynomial sampling errors.   For propagation of chaos, we develop two complementary finite-time approaches. A synchronous coupling, combined with exponential moment es
    
[^31]: 哪些指标能节省最多的人工标注？预测驱动评估与元评估

    Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation

    [https://arxiv.org/abs/2608.26638](https://arxiv.org/abs/2608.26638)

    本文提出预测驱动评估框架，结合少量人工判断与大规模自动评分实现无偏且高效的系统比较，并引入PPSR元指标来衡量自动指标节省人工标注的程度，优于现有元指标。

    

    arXiv:2608.26638v1 公告类型：新 摘要：在各种不可验证的任务中，人工评估可靠但昂贵，而自动指标更具可扩展性但往往存在偏差。基于预测驱动推断（PPI），我们提出了预测驱动评估框架，该框架将有限的人工判断与大规模自动评分相结合，以获得数据高效且可证明无偏的系统比较。我们开发了参数化和非参数化程序，分析了配对与非配对设计之间的效率权衡，并在六个WMT数据集上验证了该框架。我们进一步引入了预测驱动节省率（PPSR），这是一种元指标，用于衡量在预测驱动评估中使用自动指标时可以节省多少人工标注。PPSR直接针对预测驱动评估的指标效用，并比现有系统级元指标产生更具区分性和稳定性的指标排名。总体而言，我们的新范式重新定义了...

    arXiv:2608.26638v1 Announce Type: new  Abstract: Across various non-verifiable tasks, human evaluation is reliable but expensive, while automatic metrics are more scalable but often biased. Building on prediction-powered inference (PPI), we propose prediction-powered evaluation, a framework that combines limited human judgments with large-scale automatic scores to obtain data-efficient system comparisons that are provably unbiased. We develop parametric and non-parametric procedures, analyze the efficiency trade-off between paired and unpaired designs, and validate the framework on six WMT datasets. We further introduce the Prediction-Powered Saving Ratio (PPSR), a meta-metric that measures how much human annotation an automatic metric can save when used within prediction-powered evaluation. PPSR directly targets metric utility for prediction-powered evaluation and yields more discriminative and stable metric rankings than existing system-level meta-metrics. Overall, our new paradigm r
    
[^32]: DTD-VAE：用于信用风险预测的解纠缠时间依赖变分自编码器

    DTD-VAE: Disentangled Temporal Dependencies VAE for Credit Risk Prediction

    [https://arxiv.org/abs/2608.26473](https://arxiv.org/abs/2608.26473)

    本文提出DTD-VAE模型，通过解纠缠时间依赖并区分信用风险特征与客户偏好，提升了信用风险预测的准确性。

    

    评估客户信用worthiness对零售银行业务至关重要，因为它影响营销策略、客户关系管理和信用风险控制。传统方法往往难以捕捉复杂的时间依赖关系，并从客户数据中提取相关信息，这对准确的风险评估至关重要。具体而言，它们无法区分指示信用风险的时间模式与反映一般客户行为或偏好的模式，导致风险预测不佳。在本研究中，我们引入了解纠缠时间依赖变分自编码器（DTD-VAE），这是对传统VAE的一种改进，旨在解纠缠时间依赖，并将信用风险相关特征与过去的客户偏好区分开来。DTD-VAE的特征推断模块包含一个自回归时间依赖学习机制，能够熟练捕捉时间依赖关系。

    arXiv:2608.26473v1 Announce Type: new  Abstract: Evaluating customer creditworthiness is crucial for retail banking operations, as it impacts marketing strategies, customer relationship management, and credit risk control. Traditional methods often struggle to capture complex temporal dependencies and extract pertinent information from customer data, crucial for accurate risk assessment. Specifically, they fail to differentiate between temporal patterns indicative of credit risk and those reflecting general customer behavior or preferences, leading to suboptimal risk predictions. In this study, we introduce the Disentangled Temporal Dependencies Variational Autoencoder (DTD-VAE), an advancement over conventional VAE, designed to disentangle temporal dependencies and distinguish credit risk-related features from past customer preferences. The feature inference module of the DTD-VAE incorporates an autoregressive temporal dependency learning mechanism that adeptly captures the temporal d
    
[^33]: ICON分解：用于模型审计的深度表示多变量概念级解释

    ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing

    [https://arxiv.org/abs/2608.26083](https://arxiv.org/abs/2608.26083)

    ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。

    

    arXiv:2608.26083v1 公告类型：新 摘要：深度神经网络经常利用训练数据中的虚假关联，这种失败被称为捷径学习。基于概念的可解释性方法通过测试诸如患者性别或扫描仪设置等概念是否能从网络层中解码来筛选捷径。由于每个概念是单独评估的，这些方法可能会将概念之间的相关性误认为是模型使用它们的证据。我们引入了ICON分解，它转而量化每个概念在考虑所有其他概念和结果后所解释的层方差的比例。在具有已知真实标签的合成数据上，ICON比七种替代基线方法更准确地恢复了概念重要性。在皮肤病变和脑成像模型中，它隔离了模型真正依赖的概念，量化了任何提供的概念未解释的表示部分，并产生了我们验证过的稀疏解释。

    arXiv:2608.26083v1 Announce Type: new  Abstract: Deep neural networks often exploit spurious associations in their training data, a failure known as shortcut learning. Concept-based explainability methods screen for shortcuts by testing whether concepts such as a patient's sex or scanner settings can be decoded from a network layer. Because each concept is evaluated in isolation, these methods can mistake correlations between concepts as evidence that the model uses them. We introduce ICON decomposition, which instead quantifies how much of a layer's variance each concept explains after accounting for all other concepts and the outcome. On synthetic data with known ground truth, ICON recovers concept importance more accurately than seven alternative baseline methods. On skin-lesion and brain-imaging models, it isolates the concepts on which a model genuinely relies, quantifies the representation unexplained by any of the supplied concepts, and yields sparse explanations that we validat
    
[^34]: 关于掩码预测的可识别性：模式盲性与掩码调度

    On the Identifiability of Masked Prediction: Mode Blindness and Mask Schedules

    [https://arxiv.org/abs/2608.01383](https://arxiv.org/abs/2608.01383)

    本文发现掩码预测的可识别性完全由掩码调度决定，大上下文主导的调度对全局模式权重具有不可消除的盲性，并引入ε-可识别性模量来量化这一现象。

    

    掩码预测通过拟合一个由调度加权的条件分布族来学习表示，但尚不清楚何时接近最优的条件预测能够确定底层联合分布。我们针对具有两个良好分离的全局模式的数据研究这一问题，这些数据超出了快速混合恢复保证的适用范围，并表明答案仅由掩码调度决定。在大上下文模式固定下，对两个模式进行重新加权可以在总变差距离上使联合分布移动一个常数，而掩码目标在可见上下文大小上仅产生指数级小的扰动：由大上下文主导的掩码调度被证明对全局模式权重是盲的。为了量化这一点，我们引入了一个ε-可识别性模量，即与给定超额风险一致的最大分布误差，并证明它在超额风险指数级小时仍保持宏观尺度。一个精确的信息分解揭示了...

    arXiv:2608.01383v2 Announce Type: cross  Abstract: Masked prediction learns representations by fitting a schedule-weighted family of conditional laws, but it remains unclear when near-optimal conditional prediction pins down the underlying joint law. We study this question for data with two well-separated global modes, outside the reach of rapid-mixing recovery guarantees, and show that the answer is decided by the mask schedule alone. Under large-context mode pinning, reweighting the two modes can move the joint law by a constant in total variation while perturbing the masked objective exponentially little in the visible-context size: mask schedules dominated by large contexts are provably blind to the global mode weights. To quantify this, we introduce an $\varepsilon$-identifiability modulus, the largest distributional error consistent with a given excess risk, and prove that it remains macroscopic at an excess risk that is exponentially small. An exact information decomposition pin
    
[^35]: 稀疏图上消息传递中深度的价值：一个Kesten-Stigum二分性

    The Value of Depth in Message Passing on Sparse Graphs: A Kesten-Stigum Dichotomy

    [https://arxiv.org/abs/2607.16676](https://arxiv.org/abs/2607.16676)

    该论文证明了稀疏图上消息传递的深度价值由单一的Kesten-Stigum比率κ=γ²Δ决定，呈现二分性：当κ<1时误差以几何速率收敛，深度超过O(log(1/ε))的额外层对误差的改善小于ε。

    

    图神经网络在稀疏图上需要多深？我们研究其最纯粹的统计形式：在平均度 Δ=O(1) 的稀疏上下文随机块模型（CSBM）上进行节点分类，其局部弱极限是一棵带广播标记的泊松 Galton-Watson 树。已有工作推导出一个消息传递分类器 h_ℓ，它从每个距离 k≤ℓ 的顶点聚合衰减的证据 2 artanh(γ^k t(X_v))，其中 γ 是边信号，t 是特征的有界似然比变换。我们证明深度的价值由一个单一的数值决定，即 Kesten-Stigum 比率 κ=γ²Δ。在阈值以下（κ<1），误差序列以几何速率成为柯西序列，对所有 ℓ'>ℓ 有 |𝓔(ℓ)−𝓔(ℓ')|≤Cκ^((ℓ+1)/3)，因此深度超过 O(log(1/ε)) 的所有层对误差的改变都小于 ε；卷（摘要在此处被截断）

    arXiv:2607.16676v2 Announce Type: replace-cross  Abstract: How deep does a graph neural network need to be on a sparse graph? We study its purest statistical form: node classification on the sparse contextual stochastic block model (CSBM) with average degree $\Delta=O(1)$, whose local weak limit is a broadcast-labelled Poisson Galton-Watson tree. Prior work derived a message-passing classifier $h_\ell$ that aggregates from each vertex at distance $k\le\ell$ the attenuated evidence $2\operatorname{artanh}(\gamma^k t(X_v))$, with $\gamma$ the edge signal and $t$ a bounded likelihood-ratio transform of the feature. We prove that the value of depth is governed by a single number, the Kesten-Stigum ratio $\kappa=\gamma^2\Delta$. Below the threshold ($\kappa<1$), the error sequence is Cauchy at a geometric rate, $|\mathcal{E}(\ell)-\mathcal{E}(\ell')|\le C\kappa^{(\ell+1)/3}$ for all $\ell'>\ell$, so all layers beyond depth $O(\log(1/\epsilon))$ change the error by less than $\epsilon$; conv
    
[^36]: Seq2Synth：合成序列表格数据中时间保真度的基准测试

    Seq2Synth: Benchmarking Temporal Fidelity in Synthetic Sequential Tabular Data

    [https://arxiv.org/abs/2607.15606](https://arxiv.org/abs/2607.15606)

    该论文提出了Seq2Synth，一个评估合成序列表格数据时间保真度的统一基准，揭示出静态指标接近完美的生成模型仍会违反基本时间约束，且静态与时间感知评估的模型排名存在显著差异。

    

    合成序列表格数据日益被用于隐私保护的数据共享与研究，然而传统的表格数据评估指标往往忽视了时间结构。现有的单表和关系型评估协议大多将记录简化为静态分布，导致关键的时间属性得不到充分评估。我们提出了Seq2Synth，一个用于评估这些属性的统一基准。其分类体系通过刻画时间和模式属性来确定适用的评估方法，涵盖时间戳、横截面、纵向和结构保真度，以及轨迹感知的效用性与隐私性。在一个包含13个数据集的基准中的七个核心数据集和八个生成器上的实验表明，静态保真度接近完美的模型仍然违反基本的时间约束，产生重复的时间戳、不规则的间隔和不完整的观测网格。此外，静态评估与时间感知评估得出的模型排名存在实质性差异。

    arXiv:2607.15606v3 Announce Type: replace  Abstract: Synthetic sequential tabular data are increasingly used for privacy-preserving data sharing and research, yet conventional tabular metrics often overlook temporal structure. Existing single-table and relational evaluation protocols largely collapse records into static distributions, leaving key temporal properties insufficiently evaluated. We introduce Seq2Synth, a unified benchmark for assessing these properties. Its taxonomy characterizes temporal and schema properties to determine applicable evaluations, covering timestamp, cross-sectional, longitudinal, and structural fidelity, alongside trajectory-aware utility and privacy. Across seven core datasets from a 13-dataset benchmark and eight generators, models with near-perfect static fidelity still violate basic temporal constraints, producing duplicate timestamps, irregular intervals, and incomplete observation grids. Moreover, static and temporal-aware rankings diverge substantia
    
[^37]: LLM人格的双重本质：聚合倾向与框架依赖的几何结构

    The Dual Nature of LLM Persona: Aggregated Tendencies and Frame-Dependent Geometry

    [https://arxiv.org/abs/2607.02368](https://arxiv.org/abs/2607.02368)

    本论文发现LLM人格表达包含聚合倾向与框架依赖几何两个可分离成分，后者并非固有属性，而是编码聚合无法捕捉信息的协调模式。

    

    通过心理测量问卷对LLM人格的评估通常依赖于聚合分数，忽略了实例内部的关联结构。我们测试了这种几何结构是固有的还是依赖于框架的。我们使用IPIP-50响应构建实例内部相关矩阵，在GPT-4o模拟美国和华裔美国人角色时，通过操控问题顺序来分析SPD流形上的几何结构。我们发现人格表达包含两个可分离的组成部分：聚合特征（大五人格分数）在随机化下下降（21%），但对框架具有鲁棒性；几何特征（SPD流形）在框架错位下崩溃（下降42%），但在共享框架下显著恢复（至84%），超过了聚合特征（76%）。这种崩溃-恢复模式表明，人格几何并非固有属性，而是一种依赖于框架的协调模式，编码了聚合方法无法捕捉的信息。

    arXiv:2607.02368v2 Announce Type: replace-cross  Abstract: Evaluations of LLM personas via psychometric questionnaires typically rely on aggregate scores, discarding within-instance correlation structure. We test whether this geometric structure is intrinsic or frame-dependent. Constructing within-instance correlation matrices from IPIP-50 responses, we analyze geometry on SPD manifolds under manipulated question orderings in GPT-4o simulating American and Chinese-American personas. We find that persona expression comprises two dissociable components: aggregated features (Big Five scores) degrade under randomization (21% drop) but are frame-robust; geometric features (SPD manifold) collapse under frame misalignment (42% drop) but recover substantially (to 84%) under shared frames, surpassing aggregated features (76%). This collapse-recovery pattern reveals that persona geometry is not intrinsic but a frame-dependent coordination pattern encoding information invisible to aggregation. Ou
    
[^38]: 自组织保形预测：通过无监督群体发现减少区域覆盖差距

    Self-Organized Conformal Prediction: Reducing Regional Coverage Gaps with Unsupervised Group Discovery

    [https://arxiv.org/abs/2606.29403](https://arxiv.org/abs/2606.29403)

    提出自组织保形预测（SOCP），利用无需校准标签的无监督自组织映射（SOM）发现输入空间分组并从中提取校准缓冲，在预测器和非一致性分数保持不变的前提下实现精确的条件覆盖有效性，从而减少特征空间异质区域的覆盖差距。

    

    保形预测能够保证边际覆盖率，但汇总的校准分位数可能会掩盖特征空间中异质区域的系统性覆盖不足问题。我们提出自组织保形预测（SOCP），这是一种校准方案，利用无需校准标签训练的无监督自组织映射（SOM）来发现输入空间的分组。在预测时，查询的最佳匹配单元（BMU）从单个单元、固定网格邻域或基于原型的扩展中提取校准缓冲。当固定邻域过于稀疏时，模式3（Regime 3）根据原型距离添加单元，其全局预算在观察到任何校准分数之前，就从训练单元的占用情况和计划校准规模中选定。预测器和非一致性分数保持不变。仅单元检索具有精确的单元条件有效性，且每个固定的单元并集都具有精确的检索集有效性。（摘要在此处截断）

    arXiv:2606.29403v2 Announce Type: replace-cross  Abstract: Conformal prediction guarantees marginal coverage, but a pooled calibration quantile can hide systematic undercoverage across heterogeneous regions of the feature space. We introduce Self-Organized Conformal Prediction (SOCP), a calibration scheme that discovers input-space groups with an unsupervised Self-Organizing Map (SOM) trained without calibration labels. At prediction time, the query's best-matching unit (BMU) draws a calibration buffer from one cell, a fixed grid neighborhood, or a prototype-based enlargement. When fixed neighborhoods are too sparse, Regime 3 adds cells by prototype distance, using a global budget selected from training-cell occupancies and the planned calibration size before any calibration score is observed. The predictor and nonconformity score remain unchanged. Cell-only retrieval has exact cell-conditional validity, and each fixed union of cells has exact retrieved-set validity. Interpreting a nei
    
[^39]: 在大语言模型推理中，价值错位之上还存在非理性

    In LLM Reasoning, there is Irrationality on top of Value Misalignment

    [https://arxiv.org/abs/2606.20624](https://arxiv.org/abs/2606.20624)

    该论文提出“理性价值风险”这一新概念，将“即使经过良好价值对齐的LLM在推理时仍无法最大化对齐价值”这一差距进行数学形式化，并通过覆盖多类主流模型和基准的大量实验证明该风险普遍存在，且价值对齐只能减少而无法消除它。

    

    在将大语言模型（LLM）与目标价值函数对齐方面已经取得了显著进展。我们认为，即使一个LLM在（后）训练中已经得到了很好的对齐，它在推理时仍可能无法最大化已对齐的价值。我们将这一差距在数学上形式化为“理性价值风险”：即模型实际部署的推理策略与其理性对应策略（其响应以最陡方向最大化效用）之间的效用差异。理性价值风险的估计误差进一步被分解为来自受限提示、受限响应和不完美验证器的三个组成部分。我们开展了大量实验，涵盖Llama-3.1、Qwen-2.5、Tülu-3系列模型（7B-72B）、GPT-5.2、GPT-5.5和DeepSeek-V4，以及UltraFeedback、AlpacaEval、GSM8K、MATH、HumanEval和MathArena等基准测试。结果验证了：(1) 理性价值风险普遍存在；(2) 价值对齐可以减少但无法避免该风险；(3) 自一致性……（摘要原文在此处截断）

    arXiv:2606.20624v2 Announce Type: replace  Abstract: Significant progress has been made in aligning LLMs with target value functions. We argue that, even when an LLM has been well aligned in (post-)training, it may still fail to maximise the aligned value in reasoning. We mathematically formalise this gap as rational value risk: the utility discrepancy between a model's deployed reasoning strategy and its rational counterpart whose responses maximise utility in the steepest direction. The estimation error of rational value risk is further decomposed into three components from bounded prompts, bounded responses, and imperfect verifiers. Extensive experiments are conducted, covering models Llama-3.1, Qwen-2.5, T\"ulu-3 families (7B-72B), GPT-5.2, GPT-5.5, and DeepSeek-V4, and benchmarks UltraFeedback, AlpacaEval, GSM8K, MATH, HumanEval, and MathArena. The results validate that (1) rational value risk is widespread; (2) value alignment can reduce, but cannot avoid, it; (3) self-consistenc
    
[^40]: PEAR：置换等变自适应路由多智能体辩论

    PEAR: Permutation-Equivariant Adaptive Routing Multi-Agent Debate

    [https://arxiv.org/abs/2606.20621](https://arxiv.org/abs/2606.20621)

    PEAR是一种推理时免训练的多智能体辩论协议，通过在辩论轮次间动态切换智能体角色分配和稀疏拓扑结构，消除了固定拓扑带来的位置偏差和角色敏感性，使影响力分布更均匀，从而提升大语言模型推理的准确性与泛化能力。

    

    多智能体辩论通过迭代式的同伴互评提高了大语言模型（LLM）的可靠性。然而，固定的拓扑结构常常会引入持续存在的位置偏差、放大不可靠智能体的影响，并导致对角色分配的高度敏感。我们提出了置换等变自适应路由多智能体辩论（PEAR），这是一种推理阶段的免训练协议，能够在连续的辩论轮次之间动态重新配置通信角色和稀疏拓扑。通过根据不断演变的智能体状态策略性地切换智能体与角色的分配，PEAR防止任何智能体永久占据特权网络位置，并将影响力更均匀地分布于整个辩论中。我们从理论上将PEAR刻画为一个等变稀疏路由器：它在智能体重标记下保持准确性，同时降低路由复杂度并提升泛化能力。在四个推理任务上的全面实证评估表明...

    arXiv:2606.20621v2 Announce Type: replace  Abstract: Multi-agent debate improves the reliability of large language models (LLMs) through iterative peer critiques. However, fixed topologies often introduce persistent positional biases, amplify unreliable agents, and cause high sensitivity to role assignments. We introduce \textit{Permutation-Equivariant Adaptive Routing Multi-Agent Debate (PEAR)}, an inference-time train-free protocol that dynamically reconfigures communication roles and sparse topologies across consecutive debate rounds. By strategically switching agent-to-role assignments based on evolving agent states, PEAR prevents any agent from permanently occupying a privileged network position or distributes influence more evenly across the debate. We theoretically characterize PEAR as an equivariant sparse router: it preserves accuracy under agent relabeling while reducing routing complexity and improving generalization. Comprehensive empirical evaluations across four reasoning
    
[^41]: 基于Mamba辅助的非马尔可夫闭合降阶建模方法

    Mamba-Assisted Non-Markovian Closure for Reduced-Order Modeling

    [https://arxiv.org/abs/2606.05371](https://arxiv.org/abs/2606.05371)

    该论文提出Mamba辅助闭合框架，将非马尔可夫闭合建模转化为序列建模问题，利用Mamba模型高效预测闭合项并与降阶控制方程耦合，从而实现对高维动力系统的高效降阶建模。

    

    arXiv:2606.05371v2 公告类型：替换 摘要：高维动力系统的降阶建模常常受到由未解析变量引起的闭合效应的阻碍，这些效应会在已解析动力学中引入非马尔可夫依赖性。受Mori-Zwanzig形式体系中出现的依赖历史的记忆项启发，我们将非马尔可夫闭合建模重新表述为一个序列建模问题，并提出了Mamba辅助闭合框架。MAC采用基于Mamba的序列模型，从已解析轨迹预测闭合项，并通过数值积分器将学习到的闭合项与降阶控制方程耦合，从而在时间上推进已解析变量。在训练过程中，Mamba中的选择性扫描机制能够实现高效的并行序列处理，计算量随序列长度线性扩展；而在自回归推理阶段，则通过循环状态更新以基本恒定的每步成本进行。我们在四个基准……（原文在此处截断）

    arXiv:2606.05371v2 Announce Type: replace  Abstract: Reduced-order modeling of high-dimensional dynamical systems is often hindered by closure effects arising from unresolved variables, which can introduce non-Markovian dependence into the resolved dynamics. Motivated by the history-dependent memory term arising in the Mori--Zwanzig formalism, we recast non-Markovian closure modeling as a sequence modeling problem and propose the Mamba-Assisted Closure (MAC) framework. MAC employs a Mamba-based sequence model to predict the closure from the resolved trajectory and couples the learned closure with the reduced-order governing equations through a numerical integrator to advance the resolved variables in time. During training, the selective scan mechanism in Mamba enables efficient parallel sequence processing with linear scaling in sequence length, while autoregressive inference proceeds through recurrent state updates at essentially constant per-step cost. We evaluate MAC on four benchma
    
[^42]: 基于4D点云的形状与表面颜色同时监测：一种免配准方法

    Simultaneous Monitoring of Shape and Surface Color via 4D Point Clouds: A Registration-free Approach

    [https://arxiv.org/abs/2605.08753](https://arxiv.org/abs/2605.08753)

    提出了一种基于4D点云的免配准框架SMAC，利用Laplace-Beltrami算子谱特性同时监测形状变形与颜色异常，并通过空间感知诊断程序定位异常来源。

    

    先进制造技术能够生产具有高形状复杂度和空间变化材料组成的复杂零件。点云与颜色属性的数据融合形成了4D点云，这是一种紧凑且信息丰富的表示形式，同时编码了形状和材料信息。本文提出了一种通过4D点云同时监测形状与颜色（SMAC）的免配准框架。所提出的框架利用Laplace-Beltrami算子的谱特性来捕获和监测几何特征以及形状与表面颜色之间的关系。提出了一种组合监测方案以有效检测形状变形和颜色异常，并配合一种空间感知的信号后诊断程序来确定变化来源并定位颜色异常。重要的是，两个组件均不依赖于配准或网格重建。

    arXiv:2605.08753v2 Announce Type: replace-cross  Abstract: Advanced manufacturing technologies allow for the production of intricate parts featuring high shape complexity and spatially-varying material composition. Data fusion of point clouds with chromatic attributes provides 4D point clouds, a compact and informative representation that encodes both shape and material information. In this paper, we present a registration-free framework for Simultaneous Monitoring of shApe and Color (SMAC) via 4D point clouds. The proposed framework leverages Laplace-Beltrami operator spectral properties to capture and monitor geometric features and the relationship between shape and surface color. A combined monitoring scheme is proposed to effectively detect shape deformations and color anomalies, along with a spatially-aware post-signal diagnostic procedure to determine the source of change and localize color anomalies. Importantly, neither component relies on registration or mesh reconstruction, e
    
[^43]: 极大似然估计的次高斯集中性与熵正态性

    Sub-Gaussian Concentration and Entropic Normality of the Maximum Likelihood Estimator

    [https://arxiv.org/abs/2605.07107](https://arxiv.org/abs/2605.07107)

    本文强化了极大似然估计的经典渐近正态性结果，建立了归一化MLE的次高斯尾界、全矩收敛及熵中心极限定理，并在附加正则性条件下进一步证明了MLE本身的熵正态性。

    

    众所周知，在标准正则性条件下，极大似然估计（MLE）满足中心极限定理，且随着样本量的增长依分布收敛于高斯随机变量。本文通过为归一化的MLE建立几种更强的渐近正态性形式，强化了这一经典结果。在对得分函数附加假设的条件下，我们首先为归一化估计误差建立了次高斯尾界以及所有矩的收敛性。随后，我们证明了该估计量平滑版本的熵中心极限定理，表明其以相对熵收敛于极限高斯分布。当归一化估计的Fisher信息有界，或其密度具有有界的一阶导数时，我们进一步证明可以去除平滑处理，从而得到MLE本身的熵正态性。证明过程中发展的一些辅助工具可能具有独立的研究价值。

    arXiv:2605.07107v4 Announce Type: replace-cross  Abstract: It is well known that, under standard regularity conditions, the maximum likelihood estimator (MLE) satisfies a central limit theorem and converges in distribution to a Gaussian random variable as the sample size grows. This paper strengthens this classical result by developing several stronger forms of asymptotic normality for the normalized MLE. With additional assumptions on the score, we first establish sub-Gaussian tail bounds and convergence of all moments for the normalized estimation error. We then prove an entropic central limit theorem for a smoothed version of the estimator, showing convergence in relative entropy to the limiting Gaussian law. When the Fisher information of the normalized estimate is bounded, or its density has bounded first derivative, we further show that the smoothing can be removed, yielding entropic normality of the MLE itself. The proofs develop auxiliary tools that may be of independent intere
    
[^44]: 针对完全缺失协变量的增强迁移回归学习方法

    Augmented transfer regression learning for completely missing covariates

    [https://arxiv.org/abs/2605.04469](https://arxiv.org/abs/2605.04469)

    针对目标人群中协变量完全缺失的跨人群数据问题，提出一种增强迁移回归学习方法，在子人群漂移假设下结合重要性加权估计方程与矩插补，实现了双重稳健的参数估计。

    

    大规模人群级数据集（如英国生物样本库UK Biobank和“All of Us”研究计划）常常缺少特定分析所需的协变量（例如基因或生活方式指标），而相关研究则测量了这些变量。这就产生了一种跨人群的缺失数据问题：目标人群中的协变量完全未被观测，而非单一数据集内的部分缺失。针对这一情形，我们提出了一种增强迁移回归学习方法。关键的识别条件是一个子人群漂移假设：结局变量与已观测协变量的联合分布在源人群和目标人群之间可以不同，但缺失协变量在给定已观测变量条件下的条件分布保持不变。我们将重要性加权的估计方程与缺失协变量一阶矩和二阶矩的插补项相结合，所得估计量具有双重稳健性。

    arXiv:2605.04469v2 Announce Type: replace-cross  Abstract: Large-scale population-level datasets, such as the UK Biobank and the All of Us Research Program, often lack covariates needed for a specific analysis, such as genetic or lifestyle measures, while related studies measure them. This creates a cross-population missing data problem in which covariates are completely unobserved in the target population, rather than partially missing within one dataset. We propose an augmented transfer regression learning method for this setting. The key identifying condition is a sub-population shift assumption: the joint distribution of the outcome and observed covariates may differ across source and target populations, but the conditional distribution of the missing covariates given observed variables is invariant. We combine importance-weighted estimating equations with imputation terms for first- and second-order moments of the missing covariates. The resulting estimator is doubly robust and re
    
[^45]: 基于统计视角的概率价值估计的一阶有效性

    First-Order Efficiency for Probabilistic Value Estimation via A Statistical Viewpoint

    [https://arxiv.org/abs/2605.02827](https://arxiv.org/abs/2605.02827)

    该论文从统计视角揭示了看似不同的各类概率价值（如Shapley值）蒙特卡洛估计器实际上共享一个由采样定律和工作代理函数决定的一阶展开结构，从而推导出主均方误差的显式表达式并建立了一阶有效性理论。

    

    概率价值，包括Shapley值和半值，提供了一个与模型无关的框架，用于将黑盒模型的行为归因于数据点或特征，在可解释人工智能和数据估值等领域有广泛的应用。然而，其精确计算需要对指数级数量的联盟进行效用评估，这使得蒙特卡洛近似在现代机器学习应用中变得不可或缺。现有的估计器通常通过不同的表示策略构建，包括加权平均、自归一化加权、回归调整和加权最小二乘法。我们的关键观察是，这些看似不同的构造共享一个共同的一阶展开式，其主项由采样定律和一个工作代理函数所决定。这一阶表示给出了主均方误差（MSE）的显式表达式。

    arXiv:2605.02827v2 Announce Type: replace  Abstract: Probabilistic values, including Shapley values and semivalues, provide a model-agnostic framework to attribute the behavior of a black-box model to data points or features, with a wide range of applications including explainable artificial intelligence and data valuation. However, their exact computation requires utility evaluations over exponentially many coalitions, making Monte Carlo approximation essential in modern machine learning applications. Existing estimators are often developed through different representation strategies, including weighted averages, self-normalized weighting, regression adjustment, and weighted least squares. Our key observation is that these seemingly distinct constructions share a common first-order expansion, in which the leading term is determined by the sampling law and a working surrogate function. This first-order representation yields an explicit expression for the leading mean squared error (MSE
    
[^46]: 基于仿真推断的脑功能基础模型反演

    Inverting Foundation Models of Brain Function with Simulation-Based Inference

    [https://arxiv.org/abs/2604.23865](https://arxiv.org/abs/2604.23865)

    该研究将脑活动基础模型与大型语言模型结合，利用基于仿真的推断方法实现了从合成脑活动中反向恢复刺激的语言参数（效价、唤醒度、支配度），证明了脑模拟器的神经编码完整保留了刺激维度信息，且LLMs可作为可控刺激生成器。

    

    arXiv:2604.23865v3 公告类型： replace-cross 摘要：脑活动基础模型有望为计算神经科学开辟新前沿，通过模拟神经对复杂刺激的响应，实现跨任务和跨模态的建模。一个自然的后续问题是：这些模型能否被反向使用？我们能否从合成脑活动中恢复出刺激本身或其属性？我们在一个概念验证场景中使用TRIBEv2来研究这个问题。我们将脑模拟器与大型语言模型（LLMs）配对，后者根据效价、唤醒度和支配度等语言参数生成新闻标题。随后，我们利用基于仿真的推断方法学习从脑图到潜在刺激参数的概率映射。结果表明，这些参数可以从预测的脑图中被恢复出来，证明该模拟器的合成神经编码保留了关于受控刺激维度的信息。研究结果还表明，大型语言模型可以作为可控的刺激生成器。

    arXiv:2604.23865v3 Announce Type: replace-cross  Abstract: Foundation models of brain activity promise a new frontier for in silico neuroscience by emulating neural responses to complex stimuli across tasks and modalities. A natural next step is to ask whether these models can also be used in reverse. Can we recover a stimulus or its properties from synthetic brain activity? We study this question in a proof-of-concept setting using TRIBEv2. We pair the brain emulator with large language models (LLMs) that generate news headlines from linguistic parameters such as valence, arousal, and dominance. We then use simulation-based inference to learn a probabilistic mapping from brain maps to latent stimulus parameters. Our results show that these parameters can be recovered from predicted brain maps, demonstrating that the emulator's synthetic neural encodings preserve information about the controlled stimulus dimensions. They also show that LLMs can serve as controllable stimulus generators
    
[^47]: 网络元分析的对比空间投影：一种精确且不变的基于研究的直接与间接贡献分解

    Contrast-Space Projection for Network Meta-Analysis: An Exact and Invariant Study-Based Decomposition of Direct and Indirect Contributions

    [https://arxiv.org/abs/2604.21994](https://arxiv.org/abs/2604.21994)

    本文提出了网络元分析的对比空间投影方法，通过唯一且不变的研究级分解精确量化直接与间接证据的贡献，既能精确重构NMA估计量，又统一了广义Cochran Q的异质性与不一致性分解。

    

    网络元分析（NMA）整合了治疗网络中的直接比较与间接比较，但能够精确重现NMA估计值的贡献分解方法仍然缺失，尤其是对于包含相关对比的多臂试验。我们提出了NMA的一种对比空间投影表述，将估计量表示为观测到的两两对比到一致性约束对比空间的线性映射。基于这一表示，我们通过一种规范的研究内约简来定义直接证据与间接证据，该约简消除了代数冗余，从而产生唯一且不变的研究级分解。由此得到的协方差感知权重能够精确重构NMA估计量，并可进一步分解为间接路径层面的成分。在固定效应模型下，同一投影还表示了广义Cochran Q统计量向设计内异质性与设计间不一致性的分解。

    arXiv:2604.21994v2 Announce Type: replace-cross  Abstract: Network meta-analysis (NMA) combines direct and indirect comparisons across a treatment network, but exact contribution decompositions that reproduce NMA estimates are lacking, especially for multi-arm trials with correlated contrasts. We develop a contrast-space projection formulation of NMA that expresses the estimator as a linear mapping of observed pairwise contrasts onto the consistency-constrained contrast space. Building on this representation, we define direct and indirect evidence through a canonical within-study reduction that removes algebraic redundancy and yields a unique, invariant study-level decomposition. The resulting covariance-aware weights exactly reconstruct the NMA estimator and can be further resolved into indirect path-level components. Under fixed effects, the same projection also represents the generalized Cochran Q decomposition into within-design heterogeneity and between-design inconsistency. The f
    
[^48]: 共形预测要素

    Elements of Conformal Prediction

    [https://arxiv.org/abs/2603.23923](https://arxiv.org/abs/2603.23923)

    本文系统阐述了共形预测这一无分布且模型无关的预测推断框架的核心思想：仅需可交换性等弱假设，即可为任意黑箱学习算法提供精确的有限样本保证。

    

    预测推断是统计学中的一项基本任务，传统上通过关于数据分布的参数化假设以及对模型如何从数据中学习的详细分析来解决。近年来，共形预测作为一种替代框架兴起，非常适合涉及高维数据和复杂机器学习模型的现代应用。其吸引力在于它既是无分布的——主要依赖可交换性等对称性假设——又是模型无关的，将学习算法视为黑箱。即使在这些有限的假设下，共形预测也能提供精确的有限样本保证，尽管这些保证通常是边际的，需要仔细解读。本文阐述了共形预测的核心思想，并综述了部分精选方法。本文并非详尽无遗的综述，而是旨在提供一个清晰的概念入门点和具有教学性质的介绍。

    arXiv:2603.23923v2 Announce Type: replace-cross  Abstract: Predictive inference is a fundamental task in statistics, traditionally addressed using parametric assumptions about the data distribution and detailed analyses of how models learn from data. In recent years, conformal prediction has emerged as an alternative framework that is well suited to modern applications involving high-dimensional data and complex machine learning models. Its appeal stems from being both distribution-free---relying mainly on symmetry assumptions such as exchangeability---and model-agnostic, treating the learning algorithm as a black box. Even under such limited assumptions, conformal prediction provides exact finite-sample guarantees, although these are typically marginal and require careful interpretation. This paper explains the core ideas of conformal prediction and reviews selected methods. Rather than offering an exhaustive survey, it aims to provide a clear conceptual entry point and a pedagogical 
    
[^49]: 基于地标加速向量扩散映射

    Accelerate Vector Diffusion Maps by Landmarks

    [https://arxiv.org/abs/2603.21247](https://arxiv.org/abs/2603.21247)

    提出 LA-VDM 算法，通过地标约束扩散和两阶段归一化加速向量扩散映射，能够从点云精确恢复平行移动并渐近收敛于连接拉普拉斯算子。

    

    我们提出了一种地标约束算法 LA-VDM（Landmark Accelerated Vector Diffusion Maps，地标加速向量扩散映射），用于加速建立在图连接拉普拉斯算子（GCL）之上的向量扩散映射（VDM）框架，该框架能够捕捉复杂数据集中的成对连接关系。LA-VDM 引入了一种新颖的两阶段归一化方法，有效解决了数据集和地标集合中采样密度不均匀的问题。在具有标架丛结构的流形模型下，我们证明了可以通过地标约束扩散从点云中精确恢复平行移动，因此 LA-VDM 渐近收敛于连接拉普拉斯算子。通过在模拟数据集上的实验以及非局部图像去噪的应用，验证了 LA-VDM 的性能与准确性。

    arXiv:2603.21247v2 Announce Type: replace-cross  Abstract: We propose a landmark-constrained algorithm, LA-VDM (Landmark Accelerated Vector Diffusion Maps), to accelerate the Vector Diffusion Maps (VDM) framework built upon the Graph Connection Laplacian (GCL), which captures pairwise connection relationships within complex datasets. LA-VDM introduces a novel two-stage normalization that effectively address nonuniform sampling densities in both the data and the landmark sets. Under a manifold model with the frame bundle structure, we show that we can accurately recover the parallel transport with landmark-constrained diffusion from a point cloud, and hence asymptotically LA-VDM converges to the connection Laplacian. The performance and accuracy of LA-VDM are demonstrated through experiments on simulated datasets and an application to nonlocal image denoising.
    
[^50]: 具有共同协方差矩阵的多维高斯混合模型的模型选择与参数估计

    Model Selection and Parameter Estimation for Multidimensional Gaussian Mixture Models with a Common Covariance Matrix

    [https://arxiv.org/abs/2603.19657](https://arxiv.org/abs/2603.19657)

    该论文针对已知共同协方差矩阵的多维高斯混合模型，提出基于傅里叶协方差矩阵的谱方法进行模型阶数选择与分量均值估计，证明了区分 $k$ 分量与 $(k-1)$ 分量混合模型需要 $\Omega(\Delta^{-(4k-4)})$ 个样本的极小化极大下界，并给出了样本量为 $\Delta^{-(8k-8)}$ 阶的谱阈值oracle估计器、实用的奇异值比率估计器以及基于MUSIC型投影目标的分数初始化梯度下降均值估计方法。

    

    我们研究了具有已知共同协方差矩阵的多维高斯混合模型的模型阶数选择与分量均值估计问题。利用经验特征函数测量，我们构造了傅里叶协方差矩阵，其总体对应矩阵的秩等于混合分量的数量。我们建立了一个极小化极大下界，表明将分离的 $k$ 分量混合模型与 $(k-1)$ 分量混合模型类别区分开来需要 $\Omega(\Delta^{-(4k-4)})$ 个样本。随后，我们开发了一个谱阈值oracle估计器，对于固定的 $k$，其充分样本量为 $\Delta^{-(8k-8)}$ 阶，并同时提出了一个实用的奇异值比率估计器。在给定模型阶数的情况下，我们通过在MUSIC型投影目标上采用分数初始化的梯度下降来估计分量均值。在显式的样本量条件下，合格的样本初始化位于经认证的吸引区域内。

    arXiv:2603.19657v2 Announce Type: replace-cross  Abstract: We study model-order selection and component-mean estimation for multidimensional Gaussian mixture models with a known common covariance matrix. Using empirical characteristic-function measurements, we construct Fourier covariance matrices whose population counterparts have rank equal to the number of mixture components. We establish a minimax lower bound showing that distinguishing a separated $k$-component mixture from the class of $(k-1)$-component mixtures requires $\Omega(\Delta^{-(4k-4)})$ samples. We then develop an oracle spectral-thresholding estimator with a sufficient sample size of order $\Delta^{-(8k-8)}$ for fixed $k$, together with a practical singular-value-ratio estimator. Given the model order, we estimate the component means by score-initialized gradient descent on a MUSIC-type projection objective. Under an explicit sample-size condition, a qualifying sample initialization lies in a certified attraction regi
    
[^51]: 基于预测的条件推断

    Prediction-Powered Conditional Inference

    [https://arxiv.org/abs/2603.05575](https://arxiv.org/abs/2603.05575)

    该论文提出一种将RKHS局部化与机器学习预测校正相结合的预测驱动条件推断框架，在标注数据稀缺、未标注数据充足的场景下，对条件均值等条件泛函构造出方差更低且始终有效的估计量与置信区间。

    

    我们研究了预测驱动条件推断问题，其场景为：标注数据稀缺、未标注协变量充足，并且有一个黑箱机器学习预测器可用。目标是对在固定目标点处求值的条件泛函（例如条件均值）进行统计推断，且不对条件关系施加参数模型假设。我们的方法将局部化技术与基于预测的方差缩减相结合。首先，我们提出一种RKHS（再生核希尔伯特空间）局部化方法，从协变量中学习数据自适应的权重，并将目标点处的条件矩问题重新表述为加权后的无条件矩问题。其次，我们通过基于校正的分解将机器学习预测融入该局部化矩，从而得到预测驱动的估计量和置信区间：当预测器具有较强信息量时可以降低方差，同时无论预测器质量如何都能保持推断的有效性。

    arXiv:2603.05575v2 Announce Type: replace-cross  Abstract: We study prediction-powered conditional inference in the setting where labeled data are scarce, unlabeled covariates are abundant, and a black-box machine-learning predictor is available. The goal is to perform statistical inference on conditional functionals evaluated at a fixed target point, such as conditional means, without imposing a parametric model for the conditional relationship. Our approach combines localization with prediction-based variance reduction. First, we introduce an RKHS localization method that learns a data-adaptive weight from covariates and reformulates the target conditional moment at the target point as a weighted unconditional moment. Second, we incorporate machine-learning predictions through a correction-based decomposition of this localized moment, yielding a prediction-powered estimator and confidence interval that reduce variance when the predictor is informative while preserving validity regard
    
[^52]: 用于缺失数据识别与推断的AI生成测量值：一种弱影子变量方法

    AI-Generated Measurements for Identification and Inference with Missing Data: A Weak Shadow Variable Approach

    [https://arxiv.org/abs/2602.16061](https://arxiv.org/abs/2602.16061)

    该论文提出一个弱假设的部分识别框架，将AI（如大语言模型）从非结构化数据中生成的测量值用作弱影子变量，从而在非随机缺失（MNAR）数据下实现总体量的识别与推断。

    

    在商业和社会科学应用中，结果变量常常以依赖于未观测结果本身的方式缺失。例如在服务系统中，客户是否提交评分取决于他们本会给出的评分。这种非随机缺失（MNAR）机制使得在不强加关于观测过程的强假设的情况下，总体量难以识别。与此同时，丰富的非结构化数据（如客户交互历史）日益可得，并可以利用大语言模型（LLM）等工具将其构建为结构化测量值。在本工作中，我们开发了一个弱假设的部分识别框架，将此类测量值用作弱影子变量，其定义为：在给定真实结果和观测协变量的条件下，与缺失机制条件独立的、含有结果信息的代理变量。重要的是，这些测量值无需准确预测缺失的结果或……

    arXiv:2602.16061v3 Announce Type: replace-cross  Abstract: Across business and social science applications, outcomes are often missing in ways that depend on the unobserved outcomes themselves. In service systems, for example, whether a customer submits a rating depends on the rating they would have provided. Such missing-not-at-random (MNAR) mechanisms make population quantities difficult to identify without strong assumptions on the observation process. Meanwhile, rich unstructured data, such as customer interaction histories, are increasingly available and can be used to construct structured measurements using tools such as large language models (LLMs). In this work, we develop an assumption-lean partial identification framework that uses such measurements as weak shadow variables, defined as outcome-informative proxies that are conditionally independent of missingness given the true outcome and observed covariates. Importantly, they need not accurately predict missing outcomes or s
    
[^53]: 时间序列基础模型中的普遍冗余性

    Universal Redundancies in Time Series Foundation Models

    [https://arxiv.org/abs/2602.01605](https://arxiv.org/abs/2602.01605)

    该论文发现领先的时间序列基础模型中间层存在普遍冗余，模型对整层消融具有鲁棒性，并通过将Transformer框架化为核回归器的理论框架，提出了基于注意力头投影稳定秩的纯内在消融策略。

    

    时间序列基础模型利用大规模预训练，能够在推理阶段准确预测未见过的时序数据，而无需针对特定任务进行微调。通过在标准基准上进行大规模评估，我们发现领先的基于Transformer的TSFM在其中间层中存在冗余组件。我们引入了一套用于TSFM机制可解释性的工具，包括对特定组件的消融实验以及残差流上的直接logit归因。我们的发现在多个具有不同架构的领先TSFM上，以及在多样化的真实世界和合成时序数据集上均保持一致。我们发现研究中所有模型对整层消融都具有鲁棒性。此外，我们开发了一个将Transformer框架化为核回归器的理论框架，并由此提出了一种基于每个注意力头投影矩阵稳定秩的、纯内在的注意力头消融策略。

    arXiv:2602.01605v2 Announce Type: replace  Abstract: Time Series Foundation Models (TSFMs) leverage extensive pretraining to accurately predict unseen time series during inference, without the need for task-specific fine-tuning. Through large-scale evaluations on standard benchmarks, we find that leading transformer-based TSFMs exhibit redundant components in their intermediate layers. We introduce a set of tools for mechanistic interpretability of TSFMs, including ablations of specific components and direct logit attribution on the residual stream. Our findings are consistent across several leading TSFMs with diverse architectures, and across a diverse set of real-world and synthetic time-series datasets. We discover that all models in our study are robust to ablations of entire layers. Furthermore, we develop a theoretical framework framing transformers as kernel regressors, motivating a purely intrinsic strategy for ablating heads based on the stable rank of the per-head projection 
    
[^54]: 无需贝尔曼完备性的软拟合Q迭代：占用度重加权与温度退火

    Soft Fitted Q-Iteration without Bellman Completeness: Occupancy Reweighting and Temperature Annealing

    [https://arxiv.org/abs/2512.23927](https://arxiv.org/abs/2512.23927)

    该论文提出无需贝尔曼完备性假设的软拟合Q迭代方法，通过占用度重加权和温度退火，利用折扣占用度范数下的压缩性来保证离线强化学习的稳定性。

    

    拟合Q迭代（FQI）是离线强化学习中一种标准的基于回归的最优控制方法，但其在函数逼近下的稳定性通常依赖于贝尔曼完备性假设，即要求拟合类的贝尔曼映像仍保留在该函数类之内。我们在不依赖该假设的情况下，研究相对于固定参考策略的Kullback–Leibler（KL）正则化（即“软”）FQI。我们的关键洞察是：软控制（soft control）在折扣占用度范数下局部继承了策略评估的压缩性质。在软最优不动点处，软贝尔曼算子的线性化恰好就是软最优策略的贝尔曼算子，而该算子在其折扣占用度范数下是压缩的；在同一范数下进行投影可以保持这种压缩性。标准的软FQI则是在离线状态-动作分布下进行投影，因而无法保证这一性质。受此观察启发，我们提出了占用度重加权（occupancy reweighting）与温度退火方法……

    arXiv:2512.23927v3 Announce Type: replace-cross  Abstract: Fitted \(Q\)-iteration (FQI) is a standard regression-based method for optimal control in offline reinforcement learning, but its stability under function approximation often relies on Bellman completeness, which requires Bellman images of the fitted class to remain in the class. We study Kullback--Leibler (KL)-regularized, or soft, FQI relative to a fixed reference policy without this assumption. Our key insight is that soft control locally inherits the contraction of policy evaluation in a discounted-occupancy norm. At the soft-optimal fixed point, the linearization of the soft Bellman operator is exactly the Bellman operator for the soft-optimal policy, which contracts in its discounted-occupancy norm; projection in the same norm preserves this contraction. Standard soft FQI instead projects under the offline state-action distribution and need not preserve this property. Motivated by this observation, we propose \emph{occupa
    
[^55]: 基于占用度加权的无需Bellman完备性的拟合Q评估

    Fitted Q-Evaluation without Bellman Completeness via Occupancy Weighting

    [https://arxiv.org/abs/2512.23805](https://arxiv.org/abs/2512.23805)

    该论文提出占用度加权FQE，通过用目标策略折扣占用度比率改变回归权重，使投影范数与目标策略动力学对齐并恢复Bellman收缩性，从而在无需Bellman完备性假设的情况下获得有限样本评估保证。

    

    拟合Q评估（FQE）是一种标准的基于回归的离线策略评估方法，但在分布偏移的情况下，仅有值函数可实现性并不能保证收敛，且现有分析通常需要Bellman完备性假设。我们将这种不稳定性追溯到一个几何失配问题：标准FQE在由离线分布诱导的范数下投影Bellman目标，而这不一定能保持Bellman收缩性。因此，我们研究了占用度加权FQE（occupancy-weighted FQE），该方法仅改变回归权重。通过目标策略的折扣占用度比率进行加权，可以使投影范数与目标策略的动力学对齐，从而恢复总体投影Bellman算子的收缩性。我们在使用估计占用度比率以及函数类失配的情形下推导了有限样本保证，并将有限迭代误差、统计误差、近似误差和比率估计误差分离开来。精确的占用度加权消除了对Bellman完备性的需求。

    arXiv:2512.23805v4 Announce Type: replace-cross  Abstract: Fitted \(Q\)-evaluation (FQE) is a standard regression-based method for off-policy evaluation, but under distribution shift, value-function realizability alone does not ensure convergence, and existing analyses often require Bellman completeness. We trace this instability to a geometric mismatch: standard FQE projects Bellman targets in the norm induced by the offline distribution, which need not preserve Bellman contraction. We therefore study \emph{occupancy-weighted FQE}, which changes only the regression weights. Weighting by a target-policy discounted occupancy ratio aligns the projection norm with the target-policy dynamics and restores contraction of the population projected Bellman operator. We derive finite-sample guarantees with estimated occupancy ratios and function-class misspecification, separating finite-iteration, statistical, approximation, and ratio-estimation errors. Exact occupancy weighting removes the need
    
[^56]: 基于模拟推断中的扩散模型：教程综述

    Diffusion Models in Simulation-Based Inference: A Tutorial Review

    [https://arxiv.org/abs/2512.20685](https://arxiv.org/abs/2512.20685)

    本综述系统梳理了扩散模型在基于模拟推断（SBI）中的最新进展，涵盖训练、推断与评估的设计选择，并讨论了引导、流匹配、一致性模型等概念以及噪声调度、参数化和采样器对效率与统计精度的影响。

    

    扩散模型近来已成为基于模拟推断中强大的学习工具，能够从模拟数据和真实数据中快速且准确地估计潜在参数。其基于分数的公式化为学习参数与观测之间的条件分布或联合分布提供了灵活的方式，从而为各种建模问题提供了通用的解决方案。在本教程综述中，我们综合了扩散模型在SBI领域的最新进展，涵盖训练、推断和评估的设计选择。我们重点介绍了引导、分数组合、流匹配、一致性模型和联合建模等概念所带来的机遇。此外，我们讨论了噪声调度、参数化方法和采样器如何影响效率与统计准确性。最后，我们通过涵盖不同参数维度、模拟预算和模型类型的案例研究来阐释这些概念。

    arXiv:2512.20685v4 Announce Type: replace-cross  Abstract: Diffusion models have recently emerged as powerful learners for simulation-based inference (SBI), enabling fast and accurate estimation of latent parameters from simulated and real data. Their score-based formulation offers a flexible way to learn conditional or joint distributions over parameters and observations, thereby providing a versatile solution to various modeling problems. In this tutorial review, we synthesize recent developments on diffusion models for SBI, covering design choices for training, inference, and evaluation. We highlight opportunities created by various concepts such as guidance, score composition, flow matching, consistency models, and joint modeling. Furthermore, we discuss how efficiency and statistical accuracy are affected by noise schedules, parameterizations, and samplers. Finally, we illustrate these concepts with case studies across parameter dimensionalities, simulation budgets, and model type
    
[^57]: Autotune：面向Lasso的快速、准确且自动化的调优参数选择方法

    Autotune: fast, accurate, and automatic tuning parameter selection for Lasso

    [https://arxiv.org/abs/2512.11139](https://arxiv.org/abs/2512.11139)

    该论文提出autotune方法，通过在回归系数与噪声标准差之间交替优化带惩罚的高斯对数似然，实现Lasso调优参数的全自动选择，在低信噪比情形下比现有方法更快且具有更优的泛化性能和模型选择效果。

    

    最小绝对收缩与选择算子（Lasso）是一种流行的高维回归方法，目前已被广泛用于估计诸如向量自回归（VAR）等高维时间序列模型。尽管已有大量可选方法，如何高效且准确地选择其调优参数仍然是一个挑战。我们提出了 $\mathsf{autotune}$，这是一种让Lasso自动完成调优的策略，它通过在回归系数和噪声标准差之间交替优化带惩罚的高斯对数似然来实现。通过在回归模型和VAR模型上开展的大量模拟实验，我们表明在低信噪比环境下，$\mathsf{autotune}$ 比现有方法更快，并具有更好的泛化能力和模型选择性能。在此过程中，$\mathsf{autotune}$ 还提供了一种可用于高维统计推断的新的噪声标准差估计量，以及一种新的可视化……

    arXiv:2512.11139v3 Announce Type: replace-cross  Abstract: Least absolute shrinkage and selection operator (Lasso), a popular method for high-dimensional regression, is now used widely for estimating high-dimensional time series models such as the vector autoregression (VAR). Selecting its tuning parameter efficiently and accurately remains a challenge, despite the abundance of available methods for doing so. We propose $\mathsf{autotune}$, a strategy for Lasso to automatically tune itself by optimizing a penalized Gaussian log-likelihood alternately over regression coefficients and noise standard deviation. Using extensive simulation experiments on regression and VAR models, we show that $\mathsf{autotune}$ is faster, and provides better generalization and model selection than established alternatives in low signal-to-noise regimes. In the process, $\mathsf{autotune}$ provides a new estimator of noise standard deviation that can be used for high-dimensional inference, and a new visual
    
[^58]: 所有代理模型都是错的，许多是有用的，而有些比其他的更有用：计算机模型代理方法的可复现比较

    All Emulators are Wrong, Many are Useful, and Some are More Useful Than Others: A Reproducible Comparison of Computer Model Surrogates

    [https://arxiv.org/abs/2512.09060](https://arxiv.org/abs/2512.09060)

    该论文对29种代理模型在60个经典测试函数和40个真实数据集上进行了大规模、完全可复现的统一比较，并发布了R包duqling以支持公平一致的代理模型基准测试。

    

    准确且高效的代理建模对现代计算科学至关重要，而可供选择的代理建模方法数量惊人。随着新方法不断涌现，由于基准测试实践不一致以及（有时）可复现性和透明度有限，比较不同方法的相对优缺点仍然是一项挑战。在这项工作中，我们对29种不同的代理模型在60个经典测试函数和40个真实代理建模数据集上进行了大规模、完全可复现的比较。为了促进严格、公平的对等比较，我们引入了R包duqling，它通过一致、简单的语法和输入的自动内部缩放，简化了可复现的模拟研究。该框架使研究人员能够在统一的环境中比较代理模型，并使得以最小的努力复现或扩展先前的研究成为可能。

    arXiv:2512.09060v3 Announce Type: replace-cross  Abstract: Accurate and efficient surrogate modeling is essential for modern computational science, and there are a staggering number of emulation methods to choose from. With new methods being developed all the time, comparing the relative strengths and weaknesses of different methods remains a challenge due to inconsistent benchmarking practices and (sometimes) limited reproducibility and transparency. In this work, we present a large-scale, fully reproducible comparison of $29$ distinct emulators across $60$ canonical test functions and $40$ real emulation datasets. To facilitate rigorous, apples-to-apples comparisons, we introduce the R package \texttt{duqling}, which streamlines reproducible simulation studies using a consistent, simple syntax, and automatic internal scaling of inputs. This framework allows researchers to compare emulators in a unified environment and makes it possible to replicate or extend previous studies with min
    
[^59]: 潜时-反应理论模型：通过响应准确率与思维链长度评估大型语言模型

    Latency-Response Theory Model: Evaluating Large Language Models via Response Accuracy and Chain-of-Thought Length

    [https://arxiv.org/abs/2512.07019](https://arxiv.org/abs/2512.07019)

    提出潜时-反应理论（LaRT）模型，通过引入潜在能力与潜在速度之间的相关参数，联合建模LLM的响应准确率与思维链长度，并配套高效随机逼近EM算法，为LLM评估提供了更全面的框架。

    

    大型语言模型（LLM）的迅速发展 necessitates 有效的评估方法，以便为下游应用和可操作的后续改进提供指导。结合计算机化自适应测试的题目反应理论（IRT）模型最近成为一种有前景的框架，可通过响应准确率来评估LLM。除了简单的响应准确率之外，LLM的思维链长度也是衡量其推理能力的重要指标。为了利用思维链长度信息来辅助LLM评估，我们提出了潜时-反应理论（LaRT）模型，该模型通过引入潜在能力与潜在速度之间的关键相关参数，对响应准确率和思维链长度进行联合建模。我们推导了一种高效的随机逼近期望最大化（EM）算法用于参数估计，并为潜在参数建立了严格的可辨识性结果。

    arXiv:2512.07019v4 Announce Type: replace-cross  Abstract: The proliferation of Large Language Models (LLMs) necessitates valid evaluation methods to provide guidance for both downstream applications and actionable future improvements. The Item Response Theory (IRT) model with Computerized Adaptive Testing has recently emerged as a promising framework for evaluating LLMs via their response accuracy. Beyond simple response accuracy, LLMs' chain of thought (CoT) lengths serve as a vital indicator of their reasoning ability. To leverage the CoT length information to assist LLM evaluation, we propose the \textbf{La}tency-\textbf{R}esponse \textbf{T}heory (LaRT) model, which jointly models both the response accuracy and CoT length by introducing a key correlation parameter between the latent ability and the latent speed. We derive an efficient stochastic approximation Expectation-Maximization algorithm for parameter estimation. We establish rigorous identifiability results for the latent ab
    
[^60]: SHAKE-GNN：可扩展的层次化Kirchhoff森林图神经网络

    SHAKE-GNN: Scalable Hierarchical Kirchhoff-Forest Graph Neural Network

    [https://arxiv.org/abs/2509.22100](https://arxiv.org/abs/2509.22100)

    SHAKE-GNN是一种基于Kirchhoff森林层次结构的新型可扩展图级图神经网络框架，通过随机多分辨率分解生成多尺度表示，在大规模图分类任务中实现了性能与可扩展性之间的灵活权衡。

    

    图神经网络（GNN）在一系列学习任务中取得了显著的成功。然而，将GNN扩展到大规模图仍然是一个重大挑战，尤其是对于图级任务。在本工作中，我们提出了SHAKE-GNN，这是一种基于Kirchhoff森林层次结构的新型可扩展图级GNN框架。Kirchhoff森林是一类随机生成森林，用于构建图的随机多分辨率分解。SHAKE-GNN能够产生多尺度表示，从而在效率和性能之间实现灵活的权衡。我们引入了一种改进的、数据驱动的权衡参数选择策略，并分析了SHAKE-GNN的时间复杂度。在多个大规模图分类基准上的实验结果表明，SHAKE-GNN在提供更好可扩展性的同时，取得了有竞争力的性能。

    arXiv:2509.22100v2 Announce Type: replace  Abstract: Graph Neural Networks (GNNs) have achieved remarkable success across a range of learning tasks. However, scaling GNNs to large graphs remains a significant challenge, especially for graph-level tasks. In this work, we introduce SHAKE-GNN, a novel scalable graph-level GNN framework based on a hierarchy of Kirchhoff Forests, a class of random spanning forests used to construct stochastic multi-resolution decompositions of graphs. SHAKE-GNN produces multi-scale representations, enabling flexible trade-offs between efficiency and performance. We introduce an improved, data-driven strategy for selecting the trade-off parameter and analyse the time-complexity of SHAKE-GNN. Experimental results on multiple large-scale graph classification benchmarks demonstrate that SHAKE-GNN achieves competitive performance while offering improved scalability.
    
[^61]: 在线选择性共形推断：自适应分数、收敛速率与最优性

    Online selective conformal inference: adaptive scores, convergence rates and optimality

    [https://arxiv.org/abs/2508.10336](https://arxiv.org/abs/2508.10336)

    该论文提出OnlineSCI算法，将ACI在线共形推断扩展至用户可自主选择推断时机的选择性设定，能够控制被选时间点上的错误覆盖比例及条件瞬时错误率，并给出收敛速率与最优性保证。

    

    在监督式在线设定中，量化不确定性的方法由Gibbs和Candès（2021）的开创性工作提出。对于任意给定的点预测算法，其方法（ACI）能够生成共形预测集，使其在长时间跨度内的平均错误覆盖率接近预先指定的水平α。我们引入了该算法的一个扩展版本，称为OnlineSCI，允许用户额外选择在哪些时间点进行此类推断。OnlineSCI涵盖了多种重要的在线选择性任务，例如为极端结果构建预测区间、带弃权的分类以及在线检验。对于任意序列，OnlineSCI通过路径界控制被选择时间点上的错误覆盖比例；在随机假设下，它还能以一个非渐近余项为精度，控制在给定选择条件下的瞬时错误率。重要的是，我们的理论涵盖了……（摘要在此处截断）

    arXiv:2508.10336v3 Announce Type: replace-cross  Abstract: In a supervised online setting, quantifying uncertainty has been proposed in the seminal work of Gibbs and Cand\`es (2021). For any given point-prediction algorithm, their method (ACI) produces a conformal prediction set with an average miscoverage getting close to a prespecified level $\alpha$ for a long time horizon. We introduce an extended version of this algorithm, called OnlineSCI, allowing the user to additionally select times where such an inference should be made. OnlineSCI encompasses several prominent online selective tasks, such as building prediction intervals for extreme outcomes, classification with abstention, and online testing. OnlineSCI controls the false coverage proportion among selected times via a pathwise bound for arbitrary sequences, as well as the instantaneous error rate conditional on selection, up to a non-asymptotic remainder term, under stochastic assumptions. Importantly, our theory covers the c
    
[^62]: 将注意力机制融入语言与视觉Transformer的解释框架

    Integrating attention into explanation frameworks for language and vision transformers

    [https://arxiv.org/abs/2508.08966](https://arxiv.org/abs/2508.08966)

    该论文提出两种新颖的解释方法，将注意力权重整合进Shapley值分解等可解释AI框架中，为自然语言处理和计算机视觉任务中的Transformer模型提供有意义的解释。

    

    注意力机制位于Transformer架构的核心，提供了一种可解释的模型内部信号，这激发了人们对基于注意力的模型解释日益增长的兴趣。尽管注意力权重并不直接决定模型输出，但它们反映了词元间影响的模式，这些模式可以为既有的可解释性技术提供信息并加以补充。本研究探讨了利用注意力权重中编码的信息来提供有意义模型解释的潜力，将其整合到针对模型行为根本不同方面的可解释人工智能（XAI）框架中。为此，我们开发了两种新颖的解释方法，同时适用于自然语言处理和计算机视觉任务。第一种方法通过依据注意力权重所刻画的词元两两交互来重新定义特征函数，将注意力权重整合到Shapley值分解中，从而……

    arXiv:2508.08966v2 Announce Type: replace  Abstract: The attention mechanism lies at the core of the transformer architecture, providing an interpretable model-internal signal that has motivated a growing interest in attention-based model explanations. Although attention weights do not directly determine model outputs, they reflect patterns of token influence that can inform and complement established explainability techniques. This work studies the potential of utilising the information encoded in attention weights to provide meaningful model explanations by integrating them into explainable AI (XAI) frameworks that target fundamentally different aspects of model behaviour. To this end, we develop two novel explanation methods applicable to both natural language processing and computer vision tasks. The first integrates attention weights into the Shapley value decomposition by redefining the characteristic function in terms of pairwise token interactions via attention weights, thus ad
    
[^63]: Bures-Wasserstein流形上Fréchet回归的偏效应检验

    Test of partial effects for Frechet regression on Bures-Wasserstein manifolds

    [https://arxiv.org/abs/2506.23487](https://arxiv.org/abs/2506.23487)

    本文提出了一种针对Bures-Wasserstein流形上Fréchet回归偏效应的新型检验方法，证明了其渐近有效性和一致性，并将其应用于单细胞数据中年龄对基因共表达结构影响的研究。

    

    我们提出了一种新的检验方法，用于评估响应变量位于Bures-Wasserstein流形上的Fréchet回归中的偏效应。在原假设下，我们证明该统计量可以由一个退化V-统计量近似，其极限分布为卡方随机变量的加权混合，权重由与再生核希尔伯特空间（RKHS）核相关的积分算子的特征值确定。我们建立了所提检验的渐近有效性和一致性，并通过模拟研究考察了其有限样本性能。我们将所提出的检验应用于研究年龄（在控制其他协变量的情况下）对单细胞数据中基因共表达结构的影响。

    arXiv:2506.23487v2 Announce Type: replace-cross  Abstract: We propose a novel test for assessing partial effects in Fr\'echet regression with responses lying on the Bures-Wasserstein manifold. Under the null hypothesis, we show that the statistic admits a degenerate V-statistic approximation whose limiting distribution is a weighted mixture of chi-squared random variables, with weights determined by the eigenvalues of an integral operator associated with a reproducing kernel Hilbert space (RKHS) kernel. We establish the asymptotic validity and consistency of the proposed test. Its finite-sample performance is examined through simulation studies. We apply the proposed test to study the effect of age, while controlling for other covariates, on gene co-expression structure in single-cell data.
    
[^64]: 混合AI-人类文本中水印比例的最优估计

    Optimal Estimation of Watermark Proportions in Hybrid AI-Human Texts

    [https://arxiv.org/abs/2506.22343](https://arxiv.org/abs/2506.22343)

    本文将混合来源文本中水印比例的估计转化为基于枢轴统计量的混合模型参数估计问题，证明了该参数在某些水印方案下不可辨识，而对采用连续枢轴统计量的水印方法在温和条件下可辨识，并提出了高效的最优估计器。

    

    大语言模型（LLM）中的文本水印正日益成为检测合成文本以及区分人类撰写内容与LLM生成内容的重要工具。尽管现有研究大多关注判断整篇文本是否带有水印，但许多现实场景涉及混合来源的文本，即人类撰写内容与带水印内容的混合。本文研究混合来源文本中水印比例的最优估计问题。我们将该问题转化为基于枢轴统计量的混合模型中比例参数的估计问题。首先，我们证明在某些水印方案中该参数甚至是不可辨识的，更谈不上一致可估计。与之形成鲜明对比的是，对于采用连续枢轴统计量进行检测的水印方法，我们证明了在温和条件下比例参数是可辨识的。我们提出了高效的估计器……

    arXiv:2506.22343v2 Announce Type: replace-cross  Abstract: Text watermarks in large language models (LLMs) are an increasingly important tool for detecting synthetic text and distinguishing human-written content from LLM-generated text. While most existing studies focus on determining whether entire texts are watermarked, many real-world scenarios involve mixed-source texts, which blend human-written and watermarked content. In this paper, we address the problem of optimally estimating the watermark proportion in mixed-source texts. We cast this problem as estimating the proportion parameter in a mixture model based on \emph{pivotal statistics}. First, we show that this parameter is not even identifiable in certain watermarking schemes, let alone consistently estimable. In stark contrast, for watermarking methods that employ continuous pivotal statistics for detection, we demonstrate that the proportion parameter is identifiable under mild conditions. We propose efficient estimators fo
    
[^65]: 大型图模型边缘分布的斯坦因方法

    Stein's method for marginals on large graphical models

    [https://arxiv.org/abs/2410.11771](https://arxiv.org/abs/2410.11771)

    该论文受斯坦因方法启发提出新颖的 δ-局部性条件，为大型图模型中近似分布的边缘分布建立了与维度无关的一致误差界，并据此发展了局部化采样方法。

    

    许多空间模型表现出局部性结构，这有效降低了其内在维度，使得高维分布的高效近似与采样成为可能。然而，现有的近似技术主要关注联合分布，无法对低维边缘分布提供精确的精度控制，而边缘分布恰恰是许多实际场景中的主要关注对象。通过利用局部性结构，我们为近似分布的边缘分布建立了与维度无关的一致误差界。受斯坦因方法的启发，我们引入了一种新颖的 δ-局部性条件来量化分布中的局部性，并将其与稀疏图模型等结构假设联系起来。这一理论保证推动了现有采样方法的局部化，我们通过局部化似然信息子空间方法和局部化得分匹配方法进行了说明。我们展示了……

    arXiv:2410.11771v4 Announce Type: replace  Abstract: Many spatial models exhibit locality structures that effectively reduce their intrinsic dimensionality, enabling efficient approximation and sampling of high-dimensional distributions. However, existing approximation techniques primarily focus on joint distributions and do not provide precise accuracy control for low-dimensional marginals, which are of primary interest in many practical scenarios. By leveraging the locality structures, we establish a dimension independent uniform error bound for the marginals of approximate distributions. Inspired by the Stein's method, we introduce a novel $\delta$-locality condition that quantifies the locality in distributions, and link it to the structural assumptions such as the sparse graphical models. The theoretical guarantee motivates the localization of existing sampling methods, as we illustrate through the localized likelihood-informed subspace method and localized score matching. We show
    
[^66]: 摊销采样器的自适应教师

    Adaptive teachers for amortized samplers

    [https://arxiv.org/abs/2410.01432](https://arxiv.org/abs/2410.01432)

    提出了一种教师-学生框架，利用自适应的辅助“教师”模型采样“学生”摊销采样器的高损失区域，从而提供高效的训练课程，增强模式覆盖和探索效率。

    

    摊销推理是训练参数化模型（如神经网络）来近似给定非归一化密度分布的任务，其中精确采样是难以实现的。当采样被实现为序列决策过程时，可以使用强化学习（RL）方法（如生成流网络）来训练采样策略。离策略RL训练有助于发现多样化、高回报的候选解，但现有方法在高效探索方面仍面临挑战。我们提出使用自适应训练分布（教师）来指导主要摊销采样器（学生）的训练。教师是一个辅助行为模型，经过训练以采样学生的高损失区域，并能够在未探索的模式之间进行泛化，从而通过提供高效的训练课程来增强模式覆盖。我们在合成环境中验证了该方法的有效性。

    arXiv:2410.01432v3 Announce Type: replace  Abstract: Amortized inference is the task of training a parametric model, such as a neural network, to approximate a distribution with a given unnormalized density where exact sampling is intractable. When sampling is implemented as a sequential decision-making process, reinforcement learning (RL) methods, such as generative flow networks, can be used to train the sampling policy. Off-policy RL training facilitates the discovery of diverse, high-reward candidates, but existing methods still face challenges in efficient exploration. We propose to use an adaptive training distribution (the \teacher) to guide the training of the primary amortized sampler (the \student). The \teacher, an auxiliary behavior model, is trained to sample high-loss regions of the \student and can generalize across unexplored modes, thereby enhancing mode coverage by providing an efficient training curriculum. We validate the effectiveness of this approach in a syntheti
    
[^67]: 通过秩的概念理解深度学习

    Understanding Deep Learning via Notions of Rank

    [https://arxiv.org/abs/2408.02111](https://arxiv.org/abs/2408.02111)

    本论文以“秩”为核心概念构建深度学习理论，证明了梯度训练会对多种神经网络产生隐式低秩正则化从而可能解释对自然数据的泛化，并借助神经网络与张量分解的联系，用秩的概念刻画了图神经网络建模交互的能力。

    

    尽管深度学习在科学和工业界极为流行，但对其形式化理解仍然有限。本论文提出将秩的概念作为发展深度学习理论的关键，重点关注泛化性与表达能力这两个基本方面。具体而言，我们证明了基于梯度的训练可以对多种神经网络架构产生向低秩的隐式正则化，并通过实证表明这一现象可能有助于解释模型对自然数据（如音频、图像和文本）的泛化能力。随后，我们借助一种常用于量子物理中量化纠缠的秩的概念，刻画了图神经网络建模交互作用的能力。支撑这些结果的核心工具是神经网络与张量分解之间的联系。我们的理论对设计显式正则化方案具有实际指导意义。

    arXiv:2408.02111v4 Announce Type: replace-cross  Abstract: Despite the extreme popularity of deep learning in science and industry, its formal understanding is limited. This thesis puts forth notions of rank as key for developing a theory of deep learning, focusing on the fundamental aspects of generalization and expressiveness. In particular, we establish that gradient-based training can induce an implicit regularization towards low rank for several neural network architectures, and demonstrate empirically that this phenomenon may facilitate an explanation of generalization over natural data (e.g., audio, images, and text). Then, we characterize the ability of graph neural networks to model interactions via a notion of rank, which is commonly used for quantifying entanglement in quantum physics. A central tool underlying these results is a connection between neural networks and tensor factorizations. Practical implications of our theory for designing explicit regularization schemes an
    
[^68]: 函数空间中的自编码器

    Autoencoders in Function Space

    [https://arxiv.org/abs/2408.01362](https://arxiv.org/abs/2408.01362)

    本文提出并分析了函数空间中的自编码器（FAE）与变分自编码器（FVAE），解决了函数空间中VAE目标函数的良定义性难题，使算法能够在不同分辨率之间平滑运作。

    

    自编码器以其原始的确定性形式和变分形式（VAE）得到了广泛应用。在科学应用和图像处理中，人们通常希望将数据视为函数来处理；虽然（科学中出现的微分方程的）离散化或（图像的）像素化在实践中使问题成为有限维的，但若先构想作用于函数层面的算法，然后再进行离散化或像素化，则可以得到能够在不同分辨率之间平滑运作的更优算法。本文引入、分析并部署了函数空间版本的自编码器（FAE）和变分自编码器（FVAE）。控制VAE的目标函数的良定义性是一个微妙的问题，尤其是在函数空间中，这限制了其适用性。要使FVAE目标函数良定义，需要数据分布与所选择的（摘要在此处被截断）

    arXiv:2408.01362v4 Announce Type: replace-cross  Abstract: Autoencoders have found widespread application in both their original deterministic form and in their variational formulation (VAEs). In scientific applications and in image processing it is often of interest to consider data that are viewed as functions; while discretisation (of differential equations arising in the sciences) or pixellation (of images) renders problems finite dimensional in practice, conceiving first of algorithms that operate on functions, and only then discretising or pixellating, leads to better algorithms that smoothly operate between resolutions. In this paper function-space versions of the autoencoder (FAE) and variational autoencoder (FVAE) are introduced, analysed, and deployed. Well-definedness of the objective governing VAEs is a subtle issue, particularly in function space, limiting applicability. For the FVAE objective to be well defined requires compatibility of the data distribution with the chos
    
[^69]: 一维高斯混合模型的模型选择与参数估计

    Model Selection and Parameter Estimation of One-Dimensional Gaussian Mixture Models

    [https://arxiv.org/abs/2404.12613](https://arxiv.org/abs/2404.12613)

    本文针对一维高斯混合模型证明了阶数估计的样本复杂度基本下界，并提出一种基于傅里叶测量的估计算法，其样本复杂度与该下界匹配，从而以最优复杂度同时实现模型阶数和混合分布的估计。

    

    本文研究一维高斯混合模型的学习问题，重点在于从独立同分布样本中同时估计模型阶数和混合分布。本文建立了一维高斯混合模型阶数估计的最优采样复杂度。我们证明了以高概率正确识别分量个数所需样本数的基本下界，并表明该下界关键地依赖于各分量均值之间的间隔以及分量总数。随后，我们提出一种基于傅里叶的方法来同时估计模型阶数和混合分布。我们的算法利用由样本构造的傅里叶测量，分析表明其样本复杂度与所建立的下界相匹配，从而证实了该算法的最优性。

    arXiv:2404.12613v4 Announce Type: replace-cross  Abstract: In this paper, we study the problem of learning one-dimensional Gaussian mixture models (GMMs) with a specific focus on estimating both the model order and the mixing distribution from independent and identically distributed (i.i.d.) samples. This paper establishes the optimal sampling complexity for model order estimation in one-dimensional Gaussian mixture models. We prove a fundamental lower bound on the number of samples required to correctly identify the number of components with high probability, showing that this limit depends critically on the separation between component means and the total number of components.   We then propose a Fourier-based approach to estimate both the model order and the mixing distribution. Our algorithm utilizes Fourier measurements constructed from the samples, and our analysis demonstrates that its sample complexity matches the established lower bound, thereby confirming its optimality. Nume
    
[^70]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^71]: 研究浅层神经网络中的统计推断与协变量效应

    Investigating Statistical Inference and Covariate Effects in Shallow Neural Networks

    [https://arxiv.org/abs/2311.08139](https://arxiv.org/abs/2311.08139)

    该论文研究了如何在惩罚化浅层前馈神经网络中应用经典统计推断方法（如协变量级Wald检验和协变量效应可视化），使其从黑盒预测工具转变为具有可解释性的统计模型。

    

    前馈神经网络（FNNs）通常被视为纯粹的预测算法，其强大的预测性能使其在许多机器学习应用中得到了广泛使用。然而，这种灵活性伴随着可解释性上的折衷；因此，FNNs 在历史上一直不太受统计学家的青睐。尽管如此，对于适当简约的浅层 FNNs，经典统计理论（如显著性检验和不确定性量化）仍然可以提供有用的回归式分析摘要。通过为 FNNs 补充统计推断方法和协变量效应可视化，可以将关注点从黑盒预测转移开，使 FNNs 向传统统计模型靠拢。这可以支持更多的推断性分析，从而使 FNNs 在统计建模的背景下更易于被接受和使用。我们研究了惩罚化 FNNs 情境下的协变量级 Wald 检验，并还提出……

    arXiv:2311.08139v2 Announce Type: replace-cross  Abstract: Feedforward neural networks (FNNs) are typically viewed as pure prediction algorithms, and their strong predictive performance has led to their use in many machine-learning applications. However, their flexibility comes with an interpretability trade-off; thus, FNNs have been historically less popular among statisticians. Nevertheless, for suitably parsimonious shallow FNNs, classical statistical theory, such as significance testing and uncertainty quantification, may still provide useful regression-style summaries. Supplementing FNNs with methods of statistical inference, and covariate-effect visualisations, can shift the focus away from black-box prediction and move FNNs towards traditional statistical models. This can allow for more inferential analysis, and, hence, make FNNs more accessible within the statistical-modelling context. We investigate covariate-level Wald testing in the context of penalised FNNs, and also propos
    
[^72]: GFlowNets 与变分推断

    GFlowNets and variational inference

    [https://arxiv.org/abs/2210.00580](https://arxiv.org/abs/2210.00580)

    本文证明了变分推断与生成流网络在特定条件下学习目标的期望梯度等价，并指出 GFlowNets 得益于无需重要性采样的离策略训练能力，在捕捉多峰目标分布的多样性方面更具优势。

    

    本文在两类概率算法之间架起了桥梁：一类是（层次化的）变分推断（VI），通常用于对连续空间上的分布进行建模；另一类是生成流网络，已被用于对图等离散结构上的分布进行建模。我们证明，在某些情况下，VI 算法等价于 GFlowNets 的特例，即两者学习目标的期望梯度相等。随后，我们指出了这两类方法之间的差异，并通过实验展示了这些差异是如何产生的。值得注意的是，借鉴了强化学习思想的 GFlowNets 比 VI 更适合进行离策略训练，而无需承担重要性采样所带来的高梯度方差的代价。我们认为，GFlowNets 的这一特性能够为捕捉多峰目标分布中的多样性提供优势。

    arXiv:2210.00580v4 Announce Type: replace  Abstract: This paper builds bridges between two families of probabilistic algorithms: (hierarchical) variational inference (VI), which is typically used to model distributions over continuous spaces, and generative flow networks (GFlowNets), which have been used for distributions over discrete structures such as graphs. We demonstrate that, in certain cases, VI algorithms are equivalent to special cases of GFlowNets in the sense of equality of expected gradients of their learning objectives. We then point out the differences between the two families and show how these differences emerge experimentally. Notably, GFlowNets, which borrow ideas from reinforcement learning, are more amenable than VI to off-policy training without the cost of high gradient variance induced by importance sampling. We argue that this property of GFlowNets can provide advantages for capturing diversity in multimodal target distributions.
    
[^73]: PhyloGFN: 基于生成流网络的系统发育推断

    PhyloGFN: Phylogenetic inference with generative flow networks. (arXiv:2310.08774v1 [q-bio.PE])

    [http://arxiv.org/abs/2310.08774](http://arxiv.org/abs/2310.08774)

    PhyloGFN是一种基于生成流网络的系统发育推断方法，通过采样复杂的组合结构，能够产生多样且高质量的进化假设，并在边缘似然估计方面具有竞争力。

    

    系统发育学是计算生物学的一个分支，研究生物实体之间的进化关系。尽管有着悠久的历史和众多应用，但从序列数据推断系统发育树仍然具有挑战性：树空间的高复杂性对当前的组合和概率技术构成了重要障碍。在本文中，我们采用生成流网络（GFlowNets）的框架来解决系统发育学中的两个核心问题：基于最简原则的和贝叶斯的系统发育推断。由于GFlowNets适用于采样复杂的组合结构，它们是探索和采样树拓扑和进化距离的多模态后验分布的自然选择。我们证明了我们的摊还后验采样器PhyloGFN在真实基准数据集上产生多样且高质量的进化假设。PhyloGFN在边缘似然估计方面与之前的工作相比具有竞争力。

    Phylogenetics is a branch of computational biology that studies the evolutionary relationships among biological entities. Its long history and numerous applications notwithstanding, inference of phylogenetic trees from sequence data remains challenging: the high complexity of tree space poses a significant obstacle for the current combinatorial and probabilistic techniques. In this paper, we adopt the framework of generative flow networks (GFlowNets) to tackle two core problems in phylogenetics: parsimony-based and Bayesian phylogenetic inference. Because GFlowNets are well-suited for sampling complex combinatorial structures, they are a natural choice for exploring and sampling from the multimodal posterior distribution over tree topologies and evolutionary distances. We demonstrate that our amortized posterior sampler, PhyloGFN, produces diverse and high-quality evolutionary hypotheses on real benchmark datasets. PhyloGFN is competitive with prior works in marginal likelihood estimat
    
[^74]: Delta-AI: 稀疏图模型的摊还推理中的局部目标

    Delta-AI: Local objectives for amortized inference in sparse graphical models. (arXiv:2310.02423v1 [cs.LG])

    [http://arxiv.org/abs/2310.02423](http://arxiv.org/abs/2310.02423)

    Delta-AI算法提出了一种基于稀疏图模型的摊还推理方法，通过局部信用分配和离策略训练加快了训练速度。

    

    我们提出了一种新的算法，用于稀疏概率图模型（PGMs）的摊还推理，我们称之为Delta-AI。我们的方法基于这样的观察：当PGM中的变量采样被视为一个代理人采取的动作序列时，PGM的稀疏性使得代理人的策略学习目标能够进行局部信用分配。这导致了一个局部约束，可以转化为类似生成流网络（GFlowNets）中的局部损失，从而实现了离策略训练，但避免了每个参数更新需要实例化所有随机变量的需求，从而大大加快了训练速度。Delta-AI目标与一个可计算的学习采样器中的变量给定其马尔可夫毯子的条件分布相匹配，该采样器的结构类似于贝叶斯网络，在目标PGM下具有相同的条件分布。因此，训练后的采样器可以恢复感兴趣变量的边际分布和条件分布。

    We present a new algorithm for amortized inference in sparse probabilistic graphical models (PGMs), which we call $\Delta$-amortized inference ($\Delta$-AI). Our approach is based on the observation that when the sampling of variables in a PGM is seen as a sequence of actions taken by an agent, sparsity of the PGM enables local credit assignment in the agent's policy learning objective. This yields a local constraint that can be turned into a local loss in the style of generative flow networks (GFlowNets) that enables off-policy training but avoids the need to instantiate all the random variables for each parameter update, thus speeding up training considerably. The $\Delta$-AI objective matches the conditional distribution of a variable given its Markov blanket in a tractable learned sampler, which has the structure of a Bayesian network, with the same conditional distribution under the target PGM. As such, the trained sampler recovers marginals and conditional distributions of intere
    
[^75]: 量子经济的势能优势

    Potential Energy Advantage of Quantum Economy. (arXiv:2308.08025v1 [quant-ph])

    [http://arxiv.org/abs/2308.08025](http://arxiv.org/abs/2308.08025)

    量子计算在能源效率方面具有优势，并且能够在盈利和能源效率上超越经典计算。这使得量子计算成为计算行业更可持续的选择。

    

    随着大规模机器学习模型和语言模型的广泛部署，能源成本越来越关键。对于提供计算服务的公司来说，低能耗对于市场增长和政府法规来说都非常重要。本文研究了量子计算与经典计算之间的能源优势。我们在能源效率的背景下重新定义优势，与仅基于计算复杂性的传统量子优势不同。通过一个以能量使用为约束条件的Cournot竞争模型，我们证明量子计算公司在Nash均衡点上在盈利能力和能源效率方面都能超越经典对手。因此，量子计算可能代表计算行业更可持续的发展路径。此外，我们发现量子计算经济的能源利益取决于大规模计算。

    Energy cost is increasingly crucial in the modern computing industry with the wide deployment of large-scale machine learning models and language models. For the firms that provide computing services, low energy consumption is important both from the perspective of their own market growth and the government's regulations. In this paper, we study the energy benefits of quantum computing vis-a-vis classical computing. Deviating from the conventional notion of quantum advantage based solely on computational complexity, we redefine advantage in an energy efficiency context. Through a Cournot competition model constrained by energy usage, we demonstrate quantum computing firms can outperform classical counterparts in both profitability and energy efficiency at Nash equilibrium. Therefore quantum computing may represent a more sustainable pathway for the computing industry. Moreover, we discover that the energy benefits of quantum computing economies are contingent on large-scale computation
    
[^76]: 深度图核点过程

    Deep graph kernel point processes. (arXiv:2306.11313v1 [stat.ML])

    [http://arxiv.org/abs/2306.11313](http://arxiv.org/abs/2306.11313)

    本文提出了一种基于潜在图拓扑的图点过程方法，并开发了一种新颖的深度图核来描述事件之间的触发和抑制效应，该方法在合成和实际数据集上具有优越性。

    

    点过程模型广泛用于分析图中异步事件，反映不同类型事件之间的相互影响。预测未来事件的时间和类型是一项关键任务，并且图的大小和拓扑结构增加了问题的难度。最近的神经点过程模型揭示了捕捉复杂的事件类别之间依赖关系的可能性。然而，这些方法在每个目标事件类型的强度计算中使用了包括所有事件类别在内的未经滤波的事件记录。在本文中，我们提出了一种基于潜在图拓扑的图点过程方法。对应的无向图具有代表事件类别的节点和表示潜在贡献关系的边。然后，我们开发了一种新颖的深度图核来描述事件之间的触发和抑制效应。本质影响结构通过图神经网络-based的局部邻域信息聚合进行了融合。我们在合成和实际数据集上展示了我们提出的方法比最先进的模型更具优越性。

    Point process models are widely used to analyze asynchronous events occurring within a graph that reflect how different types of events influence one another. Predicting future events' times and types is a crucial task, and the size and topology of the graph add to the challenge of the problem. Recent neural point process models unveil the possibility of capturing intricate inter-event-category dependencies. However, such methods utilize an unfiltered history of events, including all event categories in the intensity computation for each target event type. In this work, we propose a graph point process method where event interactions occur based on a latent graph topology. The corresponding undirected graph has nodes representing event categories and edges indicating potential contribution relationships. We then develop a novel deep graph kernel to characterize the triggering and inhibiting effects between events. The intrinsic influence structures are incorporated via the graph neural
    

