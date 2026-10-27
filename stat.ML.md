# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Recertifying decisions via e-closure: uniform improvements of conformal risk control and randomized e-BH](https://arxiv.org/abs/2610.35821) | 提出“重新认证”这一通用范式，借助e-闭包原理利用风险控制程序有效性证书中的可计算松弛，统一改进了共形风险控制和随机化e-BH方法，构造出更少保守的决策。 |
| [^2] | [Model Selection for Off-Policy Evaluation via Pairwise Bellman Validation](https://arxiv.org/abs/2610.35748) | 该论文提出成对贝尔曼验证（PBV）方法，通过成对候选差异构建评判函数，实现离线策略评估中无需调参的模型选择，并在候选类误设情况下仍能获得有限样本Oracle保证。 |
| [^3] | [LUCID: Local Understanding of Covariate Importance with Conditional Density Estimation](https://arxiv.org/abs/2610.35740) | 提出LUCID，一种模型无关的局部变量重要性指标，通过从给定其余特征的完整条件分布中采样来扰动目标特征，从而尊重特征间依赖关系，避免在不真实区域评估模型。 |
| [^4] | [Multi-Objective Reinforcement Learning Using Routed Ensembles and Trajectory Attribution](https://arxiv.org/abs/2610.35733) | 本文提出目标冲突系数来量化多目标强化学习中目标冲突对学习效率的影响并给出近最优遗憾界，同时设计了路由集成策略学习算法，通过维护少量基础策略并动态路由轨迹来实现样本高效的多目标策略学习。 |
| [^5] | [Provably Efficient Q-learning for Robust Reinforcement Learning](https://arxiv.org/abs/2610.35656) | 本文提出了首个面向鲁棒强化学习的样本高效异步Q-learning算法，填补了无模型鲁棒RL方法（尤其是针对马尔可夫数据场景）的空白。 |
| [^6] | [Balanced Sampling in Stochastic Multi-objective Bandits](https://arxiv.org/abs/2610.35551) | 该论文提出用KL散度来衡量多目标老虎机中帕累托最优臂采样的不公平性，并基于其递归分解结构设计了结合采样计数与帕累托隶属经验证据的自适应前二采样算法，以实现帕累托最优臂之间的平衡采样。 |
| [^7] | [Depth Improves Training-Time Scaling for Difficult Tasks in Deep Linear Networks](https://arxiv.org/abs/2610.35514) | 本文通过对深度线性网络的分析证明，任务谱决定了深度何时有益：对于谱上困难的任务，增加网络深度能够改善训练时间缩放指数，而对于简单任务该指数与深度无关。 |
| [^8] | [Minimax Rates for Conditional Generative Learning under Wasserstein Loss](https://arxiv.org/abs/2610.35490) | 该论文确立了具有Hölder连续条件分布和无约束协变量分布的条件生成学习在Wasserstein损失下的极小极大最优速率，证明期望损失的速率由局部化偏差与无条件分布估计误差的平衡决定，而损失的高阶矩会因未观测协变量区域产生额外的 $n^{-1}$ 阶风险贡献。 |
| [^9] | [Prior Predictive Matching for Bayesian LoRA](https://arxiv.org/abs/2610.35457) | 该论文提出了一种闭式先验预测匹配估计器，可在微调之前为贝叶斯LoRA自动确定先验超参数，并从理论上证明了最优先验精度应随适配秩按 r^{-1/2} 缩放。 |
| [^10] | [A knowledge-driven nonconvex square-root regularization framework for joint group sparsity and structured smoothness](https://arxiv.org/abs/2610.35428) | 提出了一种知识驱动的非凸平方根正则化框架NS³R，通过将不依赖噪声水平的平方根损失与联合非凸ℓ₂,ₚ惩罚相结合，实现对组稀疏性与结构化平滑的联合建模，克服了传统平方损失对噪声敏感以及凸惩罚导致估计偏差和大幅值过度收缩的问题。 |
| [^11] | [On the asymptotic behaviour of $\alpha$-divergence generalized posteriors](https://arxiv.org/abs/2610.35330) | 该论文证明α-散度正则化的广义后验经重缩放后收敛于非高斯极限分布——α<1时为Student-t分布且收缩快于经典n^{-1/2}速率，α>1时为紧支撑Pearson II型分布且收缩更慢——而KL散度是α-散度族中唯一保持经典集中速率的成员。 |
| [^12] | [Beyond Wasserstein Proximity: Finding Better Transfer Sources](https://arxiv.org/abs/2610.35314) | 该论文证明Wasserstein距离等边缘邻近性指标并不一定能选出最有用的迁移源，提出基于领域适应理论的条件性选择准则，并在加密货币波动率预测实验中表明选择最远而非最近的源能显著提升预测技能，且源选择效用与最终预测模型的质量相互独立。 |
| [^13] | [Temporal-Difference Calibration of Black-Box Simulators](https://arxiv.org/abs/2610.35311) | 该论文提出了一个基于Bellman正交性的无导数时序差分校准框架，通过三种随机方向有限差分算法校准黑箱模拟器，并建立了涵盖收敛性、均方误差界和中心极限定理的完整理论保证，同时揭示了依赖于模拟器设定正确性的有限差分偏差现象。 |
| [^14] | [Identities, Not Pairs: Distribution-Free False-Accept Control for Embedding-Based Verification](https://arxiv.org/abs/2610.35277) | 该论文揭示了基于嵌入的验证中仅依赖校准集的错误接受阈值在存在“枢纽身份”时的最坏情况局限，并提出 COVE 方法，通过将试验配对置于由探针与参考扩充后的所有配对中进行排名，在无分布假设下实现 O(1/m²) 水平的错误接受率控制。 |
| [^15] | [Sequential versus fully conditional specification for tree-based synthesis of health data](https://arxiv.org/abs/2610.35163) | 该研究比较了基于决策树的条件分布模型在合成医疗数据时采用顺序规格与全条件规格两种方式的效果，结果显示在隐私风险与数据效用权衡上没有一种规格普遍更优。 |
| [^16] | [AutoEnergyDesign: An LLM-Driven Agent for Automatic Energy System Design Optimization](https://arxiv.org/abs/2610.35160) | 该论文提出AutoEnergyDesign框架，利用具备多类型显式记忆和git式自动回退机制的LLM智能体，通过仿真反馈进行上下文学习来自动优化建筑能源系统设计，从而大幅减少黑盒优化所需的仿真评估次数。 |
| [^17] | [Risk-Aware Output Control Should be the Focus of Machine Unlearning](https://arxiv.org/abs/2610.35052) | 本文主张机器遗忘应摒弃以参数空间相似性为目标的传统范式，转而聚焦于具有统计保证的风险感知输出控制，因为这才是操作上有意义、可扩展且可解释的遗忘概念。 |
| [^18] | [Macro Mondrian Conformal Risk Control](https://arxiv.org/abs/2610.35035) | 本文提出一种结合蒙德里安保形预测与宏观覆盖算法的新方法，通过对用户定义的分组进行平均风险控制，解决了标准保形风险控制在不平衡分布下对稀有对象风险失控的问题。 |
| [^19] | [Conformal Novelty Detection via Data-enhanced Localization](https://arxiv.org/abs/2610.34999) | 本文提出一种数据增强的局部化保形p值及相应的条件新颖性检测算法，利用数据驱动带宽选择在有限样本下保证错误发现率控制的同时提升了检测功效。 |
| [^20] | [The Noise-Transmission Burden of Diffusion Network: Why Image Prediction Can Outperform Noise Prediction](https://arxiv.org/abs/2610.34834) | 本文揭示了扩散模型噪声预测参数化带来的“噪声传递负担”——网络必须保留输入噪声信息以抵消污染，在低秩输入瓶颈下导致随噪声增长的去噪误差下界，而图像预测不受此限制，从而解释了为什么在有限网络中图像预测可以优于噪声预测。 |
| [^21] | [Designing Customer Feedback for Subscription Boxes](https://arxiv.org/abs/2610.34726) | 针对订阅盒服务的反馈问卷设计问题，本文提出用高斯信号在入选临界点处近似序数评分所携带的信息，以量化因未选中真正最优商品而产生的期望价值损失，从而指导平台决定应询问哪些候选商品以及采用何种评分形式。 |
| [^22] | [Intrinsic Information Rates of Summaries in Simulation-Based Inference](https://arxiv.org/abs/2610.34703) | 该论文将基于模拟推断中的摘要建模为有损信息信道，证明了所需的信息预算（互信息 $I(Y;S)$）会随后验分布集中而无界增长（即使存在固定维充分统计量），并通过量化后验与概率加权逆定理给出了所需信息速率的上下界。 |
| [^23] | [Valid Early Detection of Memorization in Diffusion Models](https://arxiv.org/abs/2610.34617) | 该论文提出将分割共形校准与序贯监测相结合的两种方法（逐时间步Bonferroni校准和轨迹级最大记忆化分数直接校准），实现了扩散模型记忆化的无提示自适应早期检测，同时保证轨迹层面的有效误报控制。 |
| [^24] | [How Much Graph Is Too Much? Graph Construction and Exposure in Spatio-Temporal Forecasting](https://arxiv.org/abs/2610.34611) | 该论文系统解耦了时空预测中“图构建”与“图暴露”两个因素，发现同一领域图在不同融合设计下既可能提升也可能损害预测精度，仅靠后置融合不足以获益，需要引入融合门控才能稳定发挥图信息的作用。 |
| [^25] | [HIFI: A Highly Efficient Framework for Statistical Inference with Foundation Models](https://arxiv.org/abs/2610.34550) | HIFI提出一种两阶段微调框架：第一阶段在大样本上仅对基础模型做一次微调以捕捉数据结构并控制估计偏差，第二阶段在小样本上并行拟合多个轻量级PEFT估计器并结合重采样进行统计推断，从而把原本需要数百次模型拟合的高昂计算开销降至可承受水平。 |
| [^26] | [Joint Independence Testing for Latent Position Random Graphs](https://arxiv.org/abs/2610.34544) | 本文提出基于核方法的潜在位置随机图联合独立性检验，利用格拉姆矩阵以 $O(n^{-1/2})$ 速率收敛的特性来度量图中节点潜在表示之间的独立性。 |
| [^27] | [Ambiguous Fits: The Rashomon Effect and Challenges in Scaling Law Estimation](https://arxiv.org/abs/2610.34514) | 本文揭示了神经缩放定律拟合中存在“罗生门效应”——多个本质不同的参数选择都能几乎同样好地拟合数据，但其外推结果在大尺度上差异巨大，并提出确定性分支定界算法来解决这一模糊性问题。 |
| [^28] | [The Price of Scale in Proactive Memory Recall](https://arxiv.org/abs/2610.34507) | 该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。 |
| [^29] | [Certifying Distributionally Robust Decisions from Projection Observations](https://arxiv.org/abs/2610.34493) | 该论文提出一种置信集证书方法，仅凭标量投影观测即可在不识别潜在协方差与各决策风险的情况下认证分布鲁棒决策，并证明了其健全性、必要弃权条件与固定检查点认证保证。 |
| [^30] | [Intrinsic Gradient Barriers in Pushforward Deep Generative Models](https://arxiv.org/abs/2610.34484) | 本文证明了深度生成模型在表示重尾和多峰分布时的困难源于传输问题本身的内在属性而非模型架构或训练方法，并对前推映射的梯度和雅可比矩阵建立了定量下界。 |
| [^31] | [Inference for Causal Path-Specific Effects in Longitudinal Studies with Repeated Treatments and Multiple Mediators](https://arxiv.org/abs/2610.34349) | 该论文针对重复处理后中介变量与时变协变量序列可变的纵向研究，发展了路径特定因果效应的半参数推断方法，通过边g-公式识别效应，并给出有效影响函数的两种参数化形式，其中基于序贯回归的方法避免了对中间变量密度的直接估计。 |
| [^32] | [Conformal Prediction with Stable Measurement Mechanisms](https://arxiv.org/abs/2610.34204) | 该论文提出在“测量机制稳定”的因果假设下，将保形预测区间从拥有参考结果的源数据适配到仅有传感器读数的目标总体，并分析了这一稳定性限制对预测覆盖率与区间长度的影响。 |
| [^33] | [MetaARIMA: Configuration-Level Transfer Learning for ARIMA Models](https://arxiv.org/abs/2610.34171) | MetaARIMA通过元学习在配置层面（而非模型层面）为ARIMA模型引入迁移学习，从历史实验中学习高性能配置模式，为新序列推荐合适的ARIMA配置，同时保留了经典方法的可解释性和局部估计能力。 |
| [^34] | [Synthetic Data Augmentation for Functional Data: Wasserstein Risk Bounds and Diffusion-Based Conditional Generation](https://arxiv.org/abs/2610.34120) | 该论文首次为函数型数据的合成数据增强建立了基于类条件扩散生成器的理论框架，在Wasserstein-1距离下给出训练风险的上下界，推导出最优合成样本数量以及合成数据优于真实数据的条件，并在类别不平衡场景下给出了无需样本分割的额外风险保证。 |
| [^35] | [Minimax Regret for Sobolev Kernel-Based Reinforcement Learning](https://arxiv.org/abs/2610.34045) | 该论文首次证明了连续状态-动作空间中基于索伯列夫核的强化学习的极小极大遗憾率，并提出了能达到该最优遗憾率的SOVI算法。 |
| [^36] | [Prior-Data Fitted Networks for In-Context Off-Policy Evaluation](https://arxiv.org/abs/2610.34035) | 提出InCOPE，一种基于先验数据拟合网络的上下文离线策略评估方法，通过在合成OPE任务上预训练学习共享的策略-价值预测规则，能够在无需任务特定参数更新的情况下直接预测新表格决策问题中目标策略的期望回报。 |
| [^37] | [The Information Geometry of Representation Formation](https://arxiv.org/abs/2610.34015) | 该论文将表征形成过程本身作为研究对象，用Bures几何与量子Fisher信息刻画训练中表征的几何轨迹，发现学习会将局部敏感性集中于任务相关因素，且当所学计算需要更高阶的输入交互时，表征轨迹会系统性地变得更不直接。 |
| [^38] | [Partial Modal Tensor Regression](https://arxiv.org/abs/2610.34008) | 提出部分模态张量回归方法（PMTR），通过构造融合条件模态信息的部分模态张量协方差来指导张量降维，为非对称、重尾或含异常值的响应数据提供稳健且可解释的张量回归建模方案。 |
| [^39] | [Constrained Gradient Equilibrium: Approachability under Locally Constrained Adversaries](https://arxiv.org/abs/2610.33985) | 该论文研究了带约束的梯度均衡问题，证明了其在简单约束下可能不可实现，进而针对局部约束对手建立了基于Blackwell可达性思想的充要刻画框架，并提出了在带裕量局部可行性条件下将无遗憾学习器转换为CGEQ算法的通用原则。 |
| [^40] | [A Nonconvex Framework for Joint Sparse Sample Weighting and Parameter Estimation](https://arxiv.org/abs/2610.33957) | 提出了一种联合稀疏样本加权与参数估计（SWAPE）的非凸框架，通过交替最小化和主化-最小化近似技术同时优化模型参数与每个样本的稀疏权重，并证明了算法的收敛性。 |
| [^41] | [When is early stopping beneficial under label noise? A phase diagram for kernel gradient flow](https://arxiv.org/abs/2610.33846) | 该论文通过在可解的核梯度流设定中推导高维学习曲线并构建相图，精确刻画了标签噪声下测试误差“后期恶化”现象消失的条件，从而解释了为什么深度网络即使完全记住噪声标签也无需早停止。 |
| [^42] | [Approximate Universal Prediction under Interaction](https://arxiv.org/abs/2610.33842) | 本文研究了通用预测器（如Solomonoff归纳）的近似保证在与环境交互反馈下能否保留，证明迁移系数可分解为策略似然比与任务证据的乘积，导致任务稀有性与动作分支稀有性复合，并在固定通用混合下存在无穷多可计算环境使预测误差达到 κη·2^(K(μ)) 的紧量级。 |
| [^43] | [Hessian-Similar Optimization with Capped Participation: Lower Bounds and Virtual Masters](https://arxiv.org/abs/2610.33835) | 本文在每轮严格限制参与客户端数量的设定下，为Hessian相似优化建立了轮次复杂度下界，并提出虚拟主节点SVRS算法以消除锚点依赖，实现近最优的收敛轮次复杂度。 |
| [^44] | [How Geometry Shapes Generative Model Evaluation on Riemannian Manifolds](https://arxiv.org/abs/2610.33731) | 该论文揭示黎曼流形上生成模型评估指标的可靠性取决于几何结构（曲率与维度），没有任何一种度量普遍适用，并据此提出了报告评估距离和根据样本几何校准度量的实用建议。 |
| [^45] | [Projection-Based Federated Learning on Compact Submanifolds with Geometry-Independent Differential Privacy](https://arxiv.org/abs/2610.33712) | 提出了 DP-RFedProj，一种面向紧致子流形的记录级差分隐私联邦学习方法，通过闭式投影进行聚合并在固定环境几何中隐私化梯度，实现了与流形和聚合方案无关的 f-DP 隐私保证。 |
| [^46] | [On the Error Analysis of Triangular Transport Maps for Conditional Generation via MMD-based Objectives](https://arxiv.org/abs/2610.33610) | 本文对基于MMD目标的三角传输映射条件生成进行了误差分析，证明最小化联合MMD通常无法一致地恢复条件分布，并精确刻画了联合误差与条件误差之间的差距。 |
| [^47] | [Affine-invariant Cubic Newton with Weak Learners](https://arxiv.org/abs/2610.33516) | 本文将仿射不变三次牛顿法（AICN）推广到带弱学习器的凸优化，通过平均余弦值和Hessian诱导范数下的方向改进量进行分析，克服了GRN提升算法对弱梯度边界参数的 γ¹² 重度依赖，建立了更优的全局收敛速率。 |
| [^48] | [Causal Inference in Metric Spaces via Distance Profiles](https://arxiv.org/abs/2610.33491) | 该论文提出了仅依赖距离度量即可定义的“轮廓因果效应”Δ(ω,t)，用以刻画处理在空间位置和尺度上如何移动概率质量（包括均值不变时的离散度、尾部与结构变化），并结合交叉拟合双重稳健得分与高维高斯近似实现统计推断。 |
| [^49] | [Online Covariance Estimation for Averaged Stochastic Gradient Descent on Riemannian Manifolds](https://arxiv.org/abs/2610.33419) | 本文通过向量传输技术构造了黎曼流形上平均化RSGD的在线协方差估计器，能够自适应处理迭代中变化的切空间，无需计算黎曼Hessian矩阵或预设终止样本量，且收敛速度与欧几里得情形一致。 |
| [^50] | [Faithful Federated Tabular Data Synthesis](https://arxiv.org/abs/2610.33403) | 提出FITS框架，通过基于局部线性回归的两步联邦条件累积分布函数估计结合概率链式法则来保证分布保真度，并采用无损连续化方法统一处理混合类型表格数据。 |
| [^51] | [When Are Nesterov Trajectories Eventually Locally Minimizing?](https://arxiv.org/abs/2610.33341) | 本论文将 Nesterov 轨迹的最终局部最小化问题等价转化为相关薛定谔算子的非负性问题，并证明对于幂函数目标，在所有指数 p≥2 中该性质当且仅当 p=4 成立，且这一特殊指数可由半线性椭圆方程的稳定性理论解释。 |
| [^52] | [Blind Random Search with Noisy Measurements: Almost Sure Convergence and Convergence Rates](https://arxiv.org/abs/2610.33302) | 本文将盲目随机搜索扩展至带噪声测量的场景，通过结合递减接受阈值与噪声损失测量值的滑动平均，证明了算法几乎必然收敛到全局最优解，并给出了低维情形下的渐近上界以及高维情形下的Weibull极限分布。 |
| [^53] | [The kernel mean embedding of a random probability distribution and its principal components](https://arxiv.org/abs/2610.33286) | 本文将随机概率分布的核均值嵌入作为再生核希尔伯特空间中的随机元素建立了严格的理论框架，并推导出核主成分分析的一般收敛性结果以及正态分布情形下协方差算子谱分解的显式表达式。 |
| [^54] | [Not All Mislabels Are Equal: Efficient Task-Aware Auditing with Confidence Sequences](https://arxiv.org/abs/2610.33262) | 该论文提出ATLAS框架，将任意时刻有效的置信序列推断与任务感知的自适应审计策略结合，根据标签不确定性及其对下游评分的影响优先审计数据点，从而以比随机抽样少11%-24%的审计量将标签错误率认证到±2个百分点的精度。 |
| [^55] | [Kernelised covariance matrices, dimension reduction and imbalanced spectral clustering](https://arxiv.org/abs/2610.33177) | 提出基于截断核的核化协方差矩阵作为PCA降维的替代方案，在高斯混合模型的不平衡聚类场景下实现了更有效的降维与谱聚类，并给出了定量的理论保证。 |
| [^56] | [A Measure-Geometric Theory of Robustness in Machine Learning](https://arxiv.org/abs/2610.33123) | 本文建立了机器学习鲁棒性的基础性测度几何理论，将鲁棒性重塑为逃逸的几何形成、组织、实现与持续过程，为超越标量准则的结构化鲁棒性分析奠定了基础。 |
| [^57] | [TGAge: A Phenotypic Aging Clock Based on Tabular Foundation Models and the Gompertz Law for Assisted Living Facility Residents](https://arxiv.org/abs/2610.32994) | TGAge是一种基于表格基础模型和冈珀茨定律的第二代表型衰老时钟，能够从常规临床测量数据中估计辅助生活机构居民的风险等效生物学年龄，在生存预测上优于现有方法，且其年龄加速指标与多种衰老相关合并症显著相关。 |
| [^58] | [GoCo: structure-guided conformal calibration for error-controlled Gene Ontology annotation release](https://arxiv.org/abs/2610.32990) | GoCo提出了一种结构引导的共形校准框架，通过为GO层次感知的基因水平错误发现比例（FDP）认证全局阈值并构建嵌套发布路径，在错误可控的前提下释放更多经参考支持的基因本体注释，且无需对独立数据集重新调参。 |
| [^59] | [Metric Conformal Prediction via the Conditional Expected Local Radius](https://arxiv.org/abs/2610.31336) | 该论文提出条件期望局部半径（CELR）这一新度量，构建了适用于一般度量空间中非欧几里得响应的保形预测方法，结合度量分布随机森林实现高效计算，并从理论上证明了估计量一致性、渐近条件有效性及有限样本边际覆盖率。 |
| [^60] | [Width and Depth Limits Commute During Training](https://arxiv.org/abs/2610.30729) | 本文将残差网络无限宽与无限深极限的可交换性从初始化阶段首次扩展到训练过程，证明了在Depth-μP参数化与梯度裁剪下，线性残差网络在任意有限训练步数内，网络输出沿宽度-深度联合增长路径按 $L^2$ 收敛且两极限可交换。 |
| [^61] | [Private Online Prediction from Experts with Small Losses](https://arxiv.org/abs/2610.29575) | 该论文提出了首个遗憾度达到 $\tilde{O}(\sqrt{L^\star})$ 的差分隐私专家在线预测算法，在低损失情形下显著优于现有 $\tilde{O}(\sqrt{T})$ 界限，并证明在忘却输入下隐私代价很小。 |
| [^62] | [A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners](https://arxiv.org/abs/2610.28747) | 该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。 |
| [^63] | [Ask to Learn: Query-Based Collaborative Reinforcement Learning](https://arxiv.org/abs/2610.27606) | 该论文提出了基于查询的协作强化学习新框架，使仅有离线数据的学习者无需直接与环境交互，即可通过自适应地向持有互补信息的协作者查询状态-时间对来突破离线数据覆盖范围的限制，并针对有限视界表格型问题给出了所需查询次数的实例依赖信息论下界。 |
| [^64] | [Hilbert-Scale Stability of Posterior Diffusion Priors for Ill-Posed Signal Recovery](https://arxiv.org/abs/2610.27030) | 该论文在希尔伯特尺度框架下为线性逆问题的后验扩散采样建立了稳定性理论，在正向算子平滑条件、真值源条件与分数误差弱范数控制下，证明了学习到的扩散先验在不适定信号恢复中能够给出将噪声、分数近似与反向时间效应分离的非渐近恢复保证。 |
| [^65] | [Announced Breaks Separate Conformal Reliability from Frequency Calibration](https://arxiv.org/abs/2610.23917) | 本文证明预先公告的结构性变点使“下一区间的保形可靠性”与“封闭窗口的频率校准”这两种重置审计目标可以分离，并提出在公告时点重启经验分位数与自适应递归的方法，前者达到1/√n的minimax可靠性速率、后者达到1/n的频率差距，且实验中覆盖率误差显著优于自适应基线（0.035对0.052）。 |
| [^66] | [StochLOB: Heavy-Tailed Score Matching for Robust Limit Order Book Trend Prediction](https://arxiv.org/abs/2610.21081) | 该论文提出StochLOB训练框架，用跳跃Lévy噪声和截断α稳定噪声等重尾非高斯扩散过程替代传统的布朗高斯噪声来加噪训练数据，以更好地捕捉限价订单簿订单流的重尾与跳跃特性，从而提升LOB趋势预测的稳健性，同时推理阶段保持确定性预测器。 |
| [^67] | [SNACK: A Sequential Notation Framework for Probabilistic Graph Generation](https://arxiv.org/abs/2610.19781) | SNACK提出了一种将图线性化为字符串序列的表示框架，通过序列与有序邻接矩阵的双射对应和无效logit屏蔽机制，使序列模型能够以100%有效率生成图，并在保持有竞争力样本质量的同时显著提升GFlowNet训练吞吐量。 |
| [^68] | [Adjoint-Based Calibration and Optimal Control of Stochastic Multiscale Bioprocess Digital Twins](https://arxiv.org/abs/2610.09505) | 本文提出了一个基于伴随敏感性分析的偏差感知数字孪生校准与最优控制框架，通过拟似然估计、矩展开和正反向伴随方法量化校准不确定性对策略性能的传播影响，实现了多尺度生物过程的不确定性感知策略优化与自适应实验设计。 |
| [^69] | [From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion](https://arxiv.org/abs/2609.01043) | 提出一种无需训练的承诺式揭示采样（CRS），将选定的词元作为持久上下文插入后续模型输入，使均匀离散扩散模型的并行预测能在序列级选择上保持一致。 |
| [^70] | [The Tamed Subgradient Unadjusted Langevin Algorithm beyond Convexity](https://arxiv.org/abs/2608.06283) | 提出 SG-TULA 算法，直接利用次梯度实现对非光滑、非凸且梯度超线性增长目标分布的稳定采样，给出了显式常数的 Wasserstein-2 非渐近收敛界与优化超额风险估计，并可应用于 GPT-2 类大模型的预训练势函数。 |
| [^71] | [Isotonic Conformal Inference](https://arxiv.org/abs/2607.16675) | 提出保序保形推断统一框架，利用保序预测器的水平集结构计算局部化保形分位数与p值，在预测和多重筛选任务中提供有限样本覆盖与错误率保证，同时避免反复的保序重拟合。 |
| [^72] | [Determinantal sampling on general spaces: fast approximation via diffusion geometry](https://arxiv.org/abs/2607.06644) | 该论文通过马尔可夫扩散算子的谱核，为流形、图和网络等一般空间上的行列式点过程采样建立了明确的速率保证，突破了传统方法仅适用于欧几里得空间的局限。 |
| [^73] | [Minimax PAC Bounds for Learning in Exogenous Contextual MDPs](https://arxiv.org/abs/2606.25170) | 该论文提出了一个在查询已知前后分配采样预算的新型PAC学习框架，并针对带外生上下文的折扣马尔可夫决策过程中的策略评估、最优值估计和最优策略提取任务，给出了极小极大最优的样本复杂度界。 |
| [^74] | [Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers](https://arxiv.org/abs/2606.11949) | 提出了一个针对已部署安全分类器的在线偏移检测与保形自适应框架，通过PCA降维解决高维密度比失效问题，并引入分数分歧“金丝雀”机制，以高检出率和低延迟实现对分布偏移、逃避攻击等失效模式的监控与自适应保护。 |
| [^75] | [Divide et Calibra: Multiclass Local Calibration via Vector Quantization](https://arxiv.org/abs/2605.21060) | 提出一种结合向量量化与狄利克雷浓度索引化参数化的组合式校准方法，通过共享因子构建区域特定的校准映射，即使在校准数据稀疏的潜在空间区域也能实现良好泛化的多分类局部校准。 |
| [^76] | [Continuous Diffusion Scales Competitively with Discrete Diffusion for Language](https://arxiv.org/abs/2605.18530) | 该研究通过将 Plaid 与现代离散扩散语言模型架构对齐构建了 RePlaid，建立了首个可媲美离散 DLM 的连续扩散语言模型缩放定律，证明连续扩散是语言建模中高度竞争且可扩展的方案。 |
| [^77] | [Unified Approach for Weakly Supervised Multicalibration](https://arxiv.org/abs/2605.09857) | 本文提出了一个统一框架，通过将污染矩阵风险重写与基于见证的校准约束相结合，为缺乏干净标签的弱监督学习场景（如正例-未标注学习等）提供多校准误差估计与事后校正方法，并具有有限样本保证。 |
| [^78] | [Linear-Core Surrogates: Smooth Loss Functions with Linear Rates for Classification and Structured Prediction](https://arxiv.org/abs/2604.27742) | 本文提出了线性核心替代损失函数，通过将线性核心与平滑尾部拼接，首次实现了处处可微且具有严格线性$H$-一致性界的凸损失函数，解决了分类理论中平滑性与统计效率之间的根本矛盾。 |
| [^79] | [Iterative Identification Closure: Amplifying Causal Identifiability in Linear SEMs](https://arxiv.org/abs/2604.09309) | 该论文提出迭代识别闭包（IIC）框架，通过将外部信息源识别出的初始已知边与迭代约化的HTC传播相结合，突破了传统半Trek准则节点级识别的局限，显著提升了含潜在混杂因子的线性结构方程模型中因果效应的可识别性。 |
| [^80] | [Lumbermark: Resistant Clustering by Chopping Up Mutual Reachability Minimum Spanning Trees](https://arxiv.org/abs/2604.07143) | 提出了Lumbermark，一种通过迭代切分互达性最小生成树来检测不同大小、密度和形状聚类的稳健分裂式聚类算法，可作为HDBSCAN的替代方案并支持生成用户指定大小的分区。 |
| [^81] | [Minimaxity and Admissibility of Bayesian Neural Networks](https://arxiv.org/abs/2604.04673) | 本文从统计决策理论出发，证明了固定先验尺度下深度ReLU贝叶斯神经网络诱导的贝叶斯决策规则不具有最小最大性，并通过在有效输出方差上引入超先验，构造出同时具有容许性与最小最大性的决策规则，且将结果扩展到了KL损失下的预测密度估计问题。 |
| [^82] | [Formulation and Learning Of The Forecasting Problem](https://arxiv.org/abs/2603.20359) | 本文为数据驱动的动力系统预测方法建立了理论基础，证明了定义预测问题的映射的存在性，并证明该映射可通过监督学习以任意精度逼近。 |
| [^83] | [Multi-Domain Empirical Bayes for Linearly-Mixed Causal Representations](https://arxiv.org/abs/2603.18404) | 该论文提出了一种基于经验贝叶斯 f-建模的多领域算法，通过因果结构化得分匹配并利用领域内与领域间的不变结构，从线性混合观测中更准确地估计因果潜变量。 |
| [^84] | [Adaptive Estimation and Inference in Conditional Moment Models via the Discrepancy Principle](https://arxiv.org/abs/2603.01337) | 本文提出了一种基于差异原理的自适应方法，无需知晓未知的源参数即可在条件矩模型的线性逆问题中自动选择正则化强度，使RDIV和TRAE等估计器达到与已知源参数时相同的误差收敛率。 |
| [^85] | [Decomposition of Spillover Effects Under Misspecification: Pseudo-true Estimands and a Local-Global Extension](https://arxiv.org/abs/2602.12023) | 该论文证明当溢出效应的概括度量存在误设时，研究者估计的是真实政策效应的最佳逼近，并进一步表明即使只对局部网络或全局均衡其中一个溢出渠道建模，对应估计量仍能准确恢复该渠道的溢出效应。 |
| [^86] | [Winner's Curse Drives False Promises in Data-Driven Decisions](https://arxiv.org/abs/2602.08892) | 本文通过对近十年《管理科学》55篇论文的实证审查与理论分析，揭示了基于模型的政策评估因赢者诅咒而系统性高估决策收益，且模型精度、随机处理分配、正确模型设定和样本分割等常见辩护理由均无法消除这一乐观偏差。 |
| [^87] | [On Generation in Metric Spaces](https://arxiv.org/abs/2602.07710) | 该论文将语言生成框架推广到可分度量空间，提出尺度敏感的 $(\varepsilon,\varepsilon')$-闭包维数来刻画可生成性，并揭示了倍增空间中生成性的稳定性与一般度量空间中生成性对尺度和度量的高度敏感性之间的鲜明对比。 |
| [^88] | [Provably Reliable Classifier Guidance via Cross-Entropy Control](https://arxiv.org/abs/2601.21200) | 本文证明了对分类器交叉熵损失的控制足以约束扩散模型中分类器引导的误差，从而为基于经验损失训练的分类器引导机制首次提供了可证明的可靠性保证。 |
| [^89] | [Reinforcement Learning with Multi-Step Lookahead Information Via Adaptive Batching](https://arxiv.org/abs/2601.10418) | 该论文针对多步前瞻强化学习中寻找最优策略为NP难的问题，提出自适应分批策略（ABP），推导出其最优Bellman方程，并设计了乐观的遗憾最小化学习算法。 |
| [^90] | [The Impact of Anisotropic Covariance Structure on the Training Dynamics and Generalization Error of Linear Networks](https://arxiv.org/abs/2601.06961) | 本研究揭示了数据各向异性（尖峰协方差结构）使两层线性网络的学习动力学分为两阶段——先由输入-输出相关性主导，后由数据主方向主导，并推导出泛化误差的解析表达式，量化了数据尖峰结构与任务对齐程度对性能的提升作用。 |
| [^91] | [Computationally efficient goodness-of-fit tests through kernelized Stein discrepancy](https://arxiv.org/abs/2512.20007) | 本文提出一种基于核化Stein差异的计算高效半参数拟合优度检验，并设计了无需重新拟合模型或从中采样的影响调整野自助法来确定检验的显著性水平。 |
| [^92] | [Wasserstein-Cram\'er-Rao Theory of Unbiased Estimation](https://arxiv.org/abs/2511.07414) | 该论文提出了基于Wasserstein几何的估计量“敏感度”概念，用以刻画数据受到无穷小扰动时估计量的不稳定性，并建立了与经典Cramér-Rao方差理论（基于Fisher-Rao几何）相平行的全新无偏估计理论框架。 |
| [^93] | [On the Hardness of Reinforcement Learning with Transition Look-Ahead](https://arxiv.org/abs/2510.19372) | 该论文通过新颖的线性规划方法证明带一步前瞻的强化学习最优规划可在多项式时间内求解，而两步及以上前瞻的规划问题是NP难的，从而精确划定了该问题的可解与不可解边界。 |
| [^94] | [AlignFlow: Improving Flow-based Generative Models with Semi-Discrete Optimal Transport](https://arxiv.org/abs/2510.15038) | AlignFlow提出利用半离散最优传输（SDOT）在噪声分布与数据点之间建立显式的、具有收敛保证的最优对齐，从而提升基于流的生成模型的训练效果与可扩展性。 |
| [^95] | [Sliding-Window Signatures as Features for Time Series Forecasting](https://arxiv.org/abs/2510.12337) | 提出了一种基于岭回归与滑动窗口签名特征的时间序列预测框架，能够无需先验假设地捕捉外生变量与目标序列之间的非线性延迟依赖关系，并给出了逼近普适性与平稳性的理论保证。 |
| [^96] | [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046) | 该论文通过对十种优化器在多个模型规模和数据比例下的系统性公平评估，指出超参数调优不对等和评估设置缺陷导致了许多新优化器声称的加速被夸大，并给出了公平比较优化器的方法论标准。 |
| [^97] | [A Unified Framework for Inference with General Missingness Patterns and Machine Learning Imputation](https://arxiv.org/abs/2508.15162) | 本文提出了一个在随机缺失假设和一般缺失模式下、结合机器学习插补的统一统计推断框架，通过按缺失模式分层并聚合观测与预测信息构造估计量，理论上保证效率优于加权完整案例分析。 |
| [^98] | [Diagonally-Weighted Generalized Method of Moments Estimation for Gaussian Mixture Modeling](https://arxiv.org/abs/2507.20459) | 本文提出对角加权广义矩估计方法（DGMM），通过对最优加权矩阵的结构化近似为每个矩阶数分配自适应权重，从而在低信噪比下以极低的计算与存储成本实现高斯混合模型的统计高效估计。 |
| [^99] | [Repulsive Ensembles for Bayesian Inference in Physics-informed Neural Networks](https://arxiv.org/abs/2505.17308) | 提出在物理信息神经网络损失函数中加入排斥项的集成方法（RE-PINN），用于微分方程逆问题的贝叶斯不确定性估计，其在集成成员趋于无穷时收敛于真实贝叶斯后验，并比标准集成产生更多样的预测。 |
| [^100] | [Simultaneous analysis of approximate leave-one-out cross-validation and mean-field inference](https://arxiv.org/abs/2501.02624) | 本文通过将ALO-CV与平均场推断进行联合分析，在高斯协变量与强凸性条件下首次证明了近似留一交叉验证对于不可微正则化子也能以可忽略的误差一致地逼近真实的留一泛化误差。 |
| [^101] | [Graph Neural Networks for Supply Chain Management: A Framework and Real World Benchmark](https://arxiv.org/abs/2411.08550) | 该论文提出了一个将供应链与图结构相连接以应用图神经网络的框架，并发布了来自孟加拉国领先快消品公司的真实世界供应链规划基准数据集。 |
| [^102] | [Highly Adaptive Ridge](https://arxiv.org/abs/2410.02680) | 本文提出高度自适应岭回归（HAR）方法，在适用于表格数据的大型非参数函数类中实现了维数无关的 n^{-1/3} L2 收敛速率，且在计算上具有良好的可扩展性。 |
| [^103] | [Sharp Generalization for Nonparametric Regression in Interpolation Space by Over-Parameterized Neural Networks Trained with Preconditioned Gradient Descent and Early Stopping](https://arxiv.org/abs/2407.11353) | 本文提出一种结合早停的预条件梯度下降算法，使过参数化两层神经网络在插值空间中的非参数回归达到锐利的回归速率。 |
| [^104] | [Cost-aware Bayesian Optimization via the Pandora's Box Gittins Index](https://arxiv.org/abs/2406.20062) | 本文建立了成本感知贝叶斯优化与经济学中潘多拉魔盒问题之间的新联系，提出将基廷斯指数作为采集函数，在中高维度问题上表现优异，且在无显式成本的经典贝叶斯优化中同样有效。 |
| [^105] | [Budgeted Prompt-Set Expansion and Refinement for Language Models via Simulation Optimization](https://arxiv.org/abs/2404.08164) | 本文提出了一种基于仿真优化的两阶段框架，在有限评估预算下将提示词表示为中等维度向量并构建得分代理模型，从而实现语言模型提示词集合的高效扩展与精炼以最大化任务得分。 |
| [^106] | [Causal Inference Under Unmeasured Confounding With Negative Controls: A Minimax Learning Approach](https://arxiv.org/abs/2103.14029) | 本文提出了一种无需完备性条件、仅假设桥函数存在的负控制变量识别策略，并基于极小极大学习框架构建了适用于再生核希尔伯特空间和神经网络等一般函数类的新型桥函数估计量。 |

# 详细

[^1]: 通过e-闭包重新认证决策：共形风险控制与随机化e-BH的一致改进

    Recertifying decisions via e-closure: uniform improvements of conformal risk control and randomized e-BH

    [https://arxiv.org/abs/2610.35821](https://arxiv.org/abs/2610.35821)

    提出“重新认证”这一通用范式，借助e-闭包原理利用风险控制程序有效性证书中的可计算松弛，统一改进了共形风险控制和随机化e-BH方法，构造出更少保守的决策。

    

    统计程序通常在规定水平上控制期望损失，同时留下部分风险预算未被利用。基于风险受控决策的e值表示以及多重检验中的e-闭包原则，我们提出了“重新认证”这一通用范式，通过利用有效性证书中可计算的松弛来改进统计程序。一个有效程序的损失按其风险预算归一化后，构成一个以状态为索引的e值集合；反之，归一化损失受该集合约束的决策将继承其风险保证。重新认证利用分布结构来量化松弛，在保持有效性的前提下扩大认证e值，从而构造出保守性更低的决策。我们开发了两个应用：对于已知损失族的共形风险控制（CRC），重新认证在每个假设增广样本的留出点之间重新分配未使用的风险预算……（原文摘要至此被截断）

    arXiv:2610.35821v1 Announce Type: cross  Abstract: Statistical procedures often control expected loss at a prescribed level while leaving part of their risk budget unused. Building on the e-value representation of risk-controlled decisions and the e-closure principle for multiple testing, we formulate recertification, a general paradigm for improving procedures by exploiting computable slack in their validity certificates. A valid procedure's losses normalized by their risk budgets form a state-indexed e-value collection; conversely, decisions whose normalized losses are bounded by this collection inherit its risk guarantees. Recertification uses distributional structure to quantify slack, enlarge the certifying e-values while preserving validity, and construct less conservative decisions.   We develop two applications. For conformal risk control (CRC) with a known loss family, recertification redistributes unused risk budget among the held-out points of each hypothetical augmented sam
    
[^2]: 基于成对贝尔曼验证的离线策略评估模型选择方法

    Model Selection for Off-Policy Evaluation via Pairwise Bellman Validation

    [https://arxiv.org/abs/2610.35748](https://arxiv.org/abs/2610.35748)

    该论文提出成对贝尔曼验证（PBV）方法，通过成对候选差异构建评判函数，实现离线策略评估中无需调参的模型选择，并在候选类误设情况下仍能获得有限样本Oracle保证。

    

    模型选择和超参数调整是离线策略评估（OPE）中的一个核心挑战，因为拟合的Q函数和占据率比估计的误差无法从留出的转移数据中直接评估。贝尔曼残差和极小极大验证方法通常需要辅助回归或在单独的评判函数类上进行优化，这增加了计算和调整成本。我们提出了成对贝尔曼验证（PBV），这是一种无需调参的极小极大程序，可利用留出的转移数据在有限的拟合Q函数类和占据率比估计器类中进行选择。PBV通过成对候选差异构建评判函数，只需要有限次经验矩计算，无需辅助验证模型或单独的评判函数优化。我们建立了相对于最佳可用候选者的有限样本Oracle不等式，即使在候选类存在误设的情况下也依然成立。我们的关键洞察是Q函数误差……

    arXiv:2610.35748v1 Announce Type: cross  Abstract: Model selection and hyperparameter tuning pose a central challenge in off-policy evaluation (OPE) because errors of fitted Q-functions and occupancy-ratio estimates cannot be directly evaluated from held-out transitions. Bellman-residual and minimax validation methods often require auxiliary regressions or optimization over separate critic classes, adding computational and tuning costs. We introduce pairwise Bellman validation (PBV), a tuning-free minimax procedure for selecting among finite classes of fitted Q-functions and occupancy-ratio estimators using held-out transitions. PBV constructs critics from pairwise candidate differences, requiring only finitely many empirical moment calculations without auxiliary validation models or separate critic optimization. We establish finite-sample oracle inequalities relative to the best available candidate, even under candidate-class misspecification. Our key insight is that Q-function errors
    
[^3]: LUCID：基于条件密度估计的协变量重要性局部理解

    LUCID: Local Understanding of Covariate Importance with Conditional Density Estimation

    [https://arxiv.org/abs/2610.35740](https://arxiv.org/abs/2610.35740)

    提出LUCID，一种模型无关的局部变量重要性指标，通过从给定其余特征的完整条件分布中采样来扰动目标特征，从而尊重特征间依赖关系，避免在不真实区域评估模型。

    

    计算局部层面的变量重要性（VI）对于理解机器学习模型至关重要。基于局部置换的VI指标提供了一种简单而强大的方法，通过衡量扰动模型输入时输出变化的大小来评估变量的重要性。然而，现有的扰动策略往往忽略特征之间的依赖关系，或仅通过受限的近似来考虑这种依赖。在不充分尊重这种依赖关系的情况下扰动特征，可能会产生不真实的反事实样本，从而在观测数据分布无法支撑的区域上评估模型。我们通过提出一种新的模型无关的局部VI指标和估计器来填补这一空白，即基于条件密度估计的协变量重要性局部理解方法（LUCID）。LUCID通过从给定其余已观测特征条件下完整条件分布中抽取的合理取值来扰动目标特征。我们建立了LUCID与现有方法之间的理论联系……

    arXiv:2610.35740v1 Announce Type: cross  Abstract: Computing local-level variable importance (VI) is critical for understanding machine learning models. Local permutation-based VI metrics offer a simple yet powerful approach, measuring how much a model's output changes when perturbing its inputs. However, existing perturbation strategies often ignore feature dependence or account for it only through restricted approximations. Perturbing a feature without fully respecting this dependence can generate unrealistic counterfactuals, thereby evaluating the model in regions unsupported by the observed data distribution. We address this gap by introducing a new model-agnostic local VI metric and estimator, Local Understanding of Covariate Importance with conditional Density estimation (LUCID). LUCID perturbs features of interest using plausible values drawn from the full conditional distribution given the remaining observed features. We establish theoretical connections between LUCID and exist
    
[^4]: 基于路由集成与轨迹归因的多目标强化学习

    Multi-Objective Reinforcement Learning Using Routed Ensembles and Trajectory Attribution

    [https://arxiv.org/abs/2610.35733](https://arxiv.org/abs/2610.35733)

    本文提出目标冲突系数来量化多目标强化学习中目标冲突对学习效率的影响并给出近最优遗憾界，同时设计了路由集成策略学习算法，通过维护少量基础策略并动态路由轨迹来实现样本高效的多目标策略学习。

    

    多目标强化学习是现代人工智能的一个基本问题，其应用涵盖后训练、推理模型和机器人控制等领域。一个核心挑战是学习能够平衡相互竞争目标的策略，例如安全与对齐中的有用性与诚实性之间的权衡，或机器人技术中的速度与能耗之间的权衡。最优策略会随目标权重组合的变化而急剧改变，而这种变化的程度关键取决于目标之间的冲突程度，这使得针对每种偏好单独学习策略的样本效率十分低下。在本文中，我们引入了目标冲突系数，并利用这一度量在在线设置下推导出接近最优的遗憾界，从而量化了目标间不一致对学习效率的影响。随后，我们设计了路由集成策略学习算法，该算法维护一小组基础策略，并动态地将轨迹路由到具有最高潜力的策略……

    arXiv:2610.35733v1 Announce Type: cross  Abstract: Multi-objective reinforcement learning is a fundamental problem for modern AI, with applications spanning post-training, reasoning models, and robotic control. A central challenge is to learn policies that balance competing objectives, such as helpfulness versus honesty in safety and alignment, or speed versus energy in robotics. The optimal policy can vary sharply with objective-weighting combinations, and the extent of this variation depends crucially on the degree of conflict among objectives, making per-preference policy learning sample-inefficient. In this paper, we introduce objective-conflict coefficients and derive near-optimal regret bounds for online settings using this measure, thereby quantifying the impact of misalignment on learning efficiency. Then, we design routed ensemble policy learning, an algorithm that maintains a small set of basis policies and dynamically routes trajectories to the policy with the highest potent
    
[^5]: 面向鲁棒强化学习的可证明高效Q-learning算法

    Provably Efficient Q-learning for Robust Reinforcement Learning

    [https://arxiv.org/abs/2610.35656](https://arxiv.org/abs/2610.35656)

    本文提出了首个面向鲁棒强化学习的样本高效异步Q-learning算法，填补了无模型鲁棒RL方法（尤其是针对马尔可夫数据场景）的空白。

    

    基于马尔可夫决策过程（MDP）的标准强化学习（RL）算法容易受到模拟到现实差距的影响。为解决这一问题，研究者提出了基于鲁棒马尔可夫决策过程（RMDP）的分布式鲁棒强化学习算法，以确保智能体在现实环境中的不可预测条件下仍能可靠地执行任务。尽管相关文献十分丰富，鲁棒强化学习仍然缺乏高效的无模型方法，尤其是针对马尔可夫数据的方法，而后者在实际应用中更为常见。在这项工作中，我们通过提出一种样本高效的（异步Q-learning）鲁棒强化学习算法，朝着解决这一问题迈出了一步。特别地，我们考虑一个具有状态空间 $\mathcal{S}$ 和动作空间 $\mathcal{A}$ 的 $\gamma$ 折扣无限时域RMDP，其中不确定性集由总变差距离刻画，不确定性水平为 $\sigma$。在假设能够获得单条轨迹（摘要在此处截断）

    arXiv:2610.35656v1 Announce Type: cross  Abstract: Standard reinforcement learning (RL) algorithms based on Markov Decision Processes (MDPs) are vulnerable to sim-to-real gaps. To address this issue, distributionally robust RL algorithms based on robust MDPs (RMDPs) have been proposed, for ensuring that agents can perform reliably to unpredictable conditions in real-world environments. Despite of the rich literature, \revision{an efficient model-free method is still lacking} for robust RL, particularly for Markovian data, which is more relevant in practical applications. In this work, we take a step towards settling this problem by proposing a sample efficient (asynchronous Q-learning) algorithm for robust RL. In particular, consider a $\gamma$-discounted infinite-horizon RMDP with state space $\mathcal{S}$ and action space $\mathcal{A}$, where the uncertainty set is specified by the total variation (TV) distance with uncertainty level $\sigma$. Assuming access to a single trajectory o
    
[^6]: 随机多目标老虎机中的平衡采样

    Balanced Sampling in Stochastic Multi-objective Bandits

    [https://arxiv.org/abs/2610.35551](https://arxiv.org/abs/2610.35551)

    该论文提出用KL散度来衡量多目标老虎机中帕累托最优臂采样的不公平性，并基于其递归分解结构设计了结合采样计数与帕累托隶属经验证据的自适应前二采样算法，以实现帕累托最优臂之间的平衡采样。

    

    arXiv:2610.35551v1 公告类型：cross 摘要：在随机多目标老虎机问题中，帕累托最优臂可能代表各目标之间不同的权衡关系。在实际场景（如在线广告）中，对这些臂进行平衡采样非常重要。然而，据我们所知，关于如何衡量公平性目前似乎缺乏共识。实现公平性也十分困难，因为帕累托最优臂的集合是未知的，而且早期的采样决策可能造成难以在事后消除的不平衡。我们通过从真实帕累托集上的均匀分布到由采样轨迹诱导的经验分布之间的KL散度来衡量不公平性。该度量的一大优点是允许精确分解，从而将采样次优臂的成本与帕累托最优臂之间的不平衡分离开来。利用不公平性的递归形式，我们提出了自适应前二采样，该方法结合采样计数与帕累托隶属关系的经验证据来定义一个增益……

    arXiv:2610.35551v1 Announce Type: cross  Abstract: In stochastic multi-objective bandits, Pareto optimal arms may represent different trade-offs among objectives. Balanced sampling across them is important in practical settings such as online advertising. Yet, there seems to be little consensus on how to measure fairness, to our best knowledge. Achieving fairness is also tough as the set of Pareto optimal arms is unknown, and early sampling decisions may create imbalances that are difficult to erase afterwards. We measure unfairness by the KL divergence from the uniform distribution over the true Pareto set to the empirical measure induced by the sampling trajectory. Nicely, this measure admits an exact decomposition that separates the cost of sampling suboptimal arms from the imbalance among Pareto optimal arms. With a recursive form for unfairness, we develop \emph{adaptive top-two sampling}, which combines sampling counts and empirical evidence of Pareto membership to define a gain 
    
[^7]: 深度可改善深度线性网络中困难任务的训练时间缩放

    Depth Improves Training-Time Scaling for Difficult Tasks in Deep Linear Networks

    [https://arxiv.org/abs/2610.35514](https://arxiv.org/abs/2610.35514)

    本文通过对深度线性网络的分析证明，任务谱决定了深度何时有益：对于谱上困难的任务，增加网络深度能够改善训练时间缩放指数，而对于简单任务该指数与深度无关。

    

    神经缩放定律描述了模型性能对训练时间、数据集大小和模型规模的的可预测幂律依赖关系，其中的指数量化了性能随资源变化的程度。近期的理论研究表明，缩放指数由数据分布和学习任务的结构共同塑造，但网络深度能否改善训练时间缩放一直悬而未决。通过对深度线性网络的精细化分析，我们阐明了任务谱决定了何时增加网络深度能够改善训练时间缩放。对于谱上困难（spectrally difficult）的任务，增加深度可改善渐近的训练时间缩放指数；而对于简单任务，该指数与深度无关，深度仅在两种机制的边界处影响一个对数修正项。值得注意的是，我们发现一种由预测器范数主导的时间重参数化为……（摘要在此处被截断）

    arXiv:2610.35514v1 Announce Type: new  Abstract: Neural scaling laws describe the predictable power-law dependence of model performance on training time, dataset size, and model scale, with the exponents quantifying how performance changes with resources. Recent theory has shown how scaling exponents are shaped by the data distribution and the structure of the learning task, but whether network depth can improve training-time scaling has remained unsettled. Through a refined analysis of deep linear networks, we clarify that the task spectrum determines when increasing network depth improves training-time scaling. For spectrally difficult tasks, increasing depth improves the asymptotic training-time scaling exponent, whereas for easy tasks, the exponent remains independent of depth, with depth affecting only a logarithmic correction at the boundary between these regimes. Remarkably, we find that a reparameterization of time governed by the predictor norm provides a unified description o
    
[^8]: Wasserstein损失下条件生成学习的极小极大速率

    Minimax Rates for Conditional Generative Learning under Wasserstein Loss

    [https://arxiv.org/abs/2610.35490](https://arxiv.org/abs/2610.35490)

    该论文确立了具有Hölder连续条件分布和无约束协变量分布的条件生成学习在Wasserstein损失下的极小极大最优速率，证明期望损失的速率由局部化偏差与无条件分布估计误差的平衡决定，而损失的高阶矩会因未观测协变量区域产生额外的 $n^{-1}$ 阶风险贡献。

    

    我们研究了在积分Wasserstein损失下、具有Hölder连续条件分布和无约束协变量分布的条件生成学习的极小极大速率。我们通过条件输出分布来评估无约束的生成器。对于期望Wasserstein损失，其极小极大速率与均匀协变量情形下的速率一致，可通过平衡局部化偏差与无条件分布估计误差得到。而损失的高阶矩可能被未观测到的协变量区域所主导，这迫使产生一个阶为 $n^{-1}$ 的额外风险贡献，其中 $n$ 为样本量。自适应局部化论证给出了相匹配的上界。相关结果涵盖了无约束的响应分布以及 $W_1$ 情形下可能趋于消失的Besov密度，并在均匀响应密度界和可积性条件下可推广至更高阶的Wasserstein。多尺度下界证明闭合了对数间隙。

    arXiv:2610.35490v1 Announce Type: cross  Abstract: We study minimax rates for conditional generative learning under integrated Wasserstein loss, with H\"older continuous conditional laws and an unrestricted covariate distribution. We evaluate unrestricted generators through their conditional output distributions. For expected Wasserstein loss, the minimax rate agrees with the rate under uniform covariates and follows from balancing localization bias against unconditional distribution estimation error. Higher moments of the loss can instead be dominated by unobserved covariate regions, which force an additional risk contribution of order $n^{-1}$, where $n$ is the sample size. An adaptive localization argument gives matching upper bounds. The results cover unrestricted response laws and possibly vanishing Besov densities for $W_1$, and extend to higher Wasserstein orders under uniform response density bounds and integrability conditions. A multiscale lower bound closes the logarithmic g
    
[^9]: 贝叶斯LoRA的先验预测匹配

    Prior Predictive Matching for Bayesian LoRA

    [https://arxiv.org/abs/2610.35457](https://arxiv.org/abs/2610.35457)

    该论文提出了一种闭式先验预测匹配估计器，可在微调之前为贝叶斯LoRA自动确定先验超参数，并从理论上证明了最优先验精度应随适配秩按 r^{-1/2} 缩放。

    

    低秩适配（LoRA）的贝叶斯变体在微调大型基础模型时能够提供有原则的不确定性估计，但需要为低秩权重更新指定先验分布。我们通过引入一种在微调之前即可运行的闭式先验预测估计器，来解决这一超参数选择问题。我们分析了网络的二阶展开，并量化了低秩先验如何驱动函数空间偏离预训练输出。我们还证明了这些先验预测量仅通过适配秩与先验尺度四次幂的乘积依赖于先验分布。由于训练目标共享同一次潜在权重更新的采样，我们可以利用样本残差之间的协方差推导出矩匹配估计器。此外，我们的理论分析表明，最优先验精度随秩 r 以 $r^{-1/2}$ 的比例缩放。

    arXiv:2610.35457v1 Announce Type: cross  Abstract: Bayesian variants of Low-Rank Adaptation (LoRA) provide principled uncertainty estimation when fine-tuning large foundation models, but require specifying the prior over the low-rank weight updates. We address this hyperparameter selection problem by introducing a closed-form prior-predictive estimator that operates before fine-tuning.   We analyze a second-order expansion of the network and quantify how the low-rank prior drives function-space deviations from the pretrained outputs. We also show that these prior-predictive quantities depend on the prior only through the product of the adaptation rank and the fourth power of the prior scale. Because the training targets share a single draw of the latent weight update, we can leverage the covariance between example residuals to derive a moment-matching estimator. Furthermore, our theoretical analysis establishes that the optimal prior precision scales as $r^{-1/2}$ with the rank $r$, al
    
[^10]: 一种知识驱动的非凸平方根正则化框架，用于联合组稀疏性与结构化平滑

    A knowledge-driven nonconvex square-root regularization framework for joint group sparsity and structured smoothness

    [https://arxiv.org/abs/2610.35428](https://arxiv.org/abs/2610.35428)

    提出了一种知识驱动的非凸平方根正则化框架NS³R，通过将不依赖噪声水平的平方根损失与联合非凸ℓ₂,ₚ惩罚相结合，实现对组稀疏性与结构化平滑的联合建模，克服了传统平方损失对噪声敏感以及凸惩罚导致估计偏差和大幅值过度收缩的问题。

    

    结构化稀疏学习是高维数据建模的主流范式，因为它能够将经过领域验证的结构先验严格地融入学习过程。在这些先验中，预先定义的分组结构和分段平滑性在现实应用中无处不在。然而，大多数现有的结构化正则化方法都建立在平方损失和凸惩罚之上，很少联合建模这两种互补的结构先验，且存在两个关键局限：由于使用平方损失，其最优调参对未知噪声水平 σ 高度敏感；同时，凸惩罚会引入固有的估计偏差，并对大幅值系数产生严重的过度收缩。为解决这些问题，我们提出了一种知识驱动的非凸结构化平滑调整平方根正则化方法（NS³R），该方法将不依赖 σ 的平方根损失与联合非凸 ℓ₂,ₚ 惩罚统一起来。

    arXiv:2610.35428v1 Announce Type: new  Abstract: Structured sparse learning is a leading paradigm for high-dimensional data modeling, as it enables rigorous integration of domain-validated structural priors into the learning process. Among these priors, pre-defined grouped structure and piecewise smoothness are ubiquitous in real-world applications. However, most existing structured regularization methods are built on squared loss and convex penalties, rarely jointly model these two complementary structural priors, and suffer from two key limitations. Their optimal tuning parameters are sensitive to the unknown noise level $\sigma$ due to the squared loss. Meanwhile, convex penalties induce inherent estimation bias and severe over-shrinkage of large magnitudes. To address these issues, we propose a knowledge-driven Nonconvex Structured Smooth Adjustment for Square-Root Regularization (NS$^3$R), which unifies the $\sigma$-free square-root loss with a joint nonconvex $\ell_{2,q}$ penalty
    
[^11]: 关于α-散度广义后验的渐近行为

    On the asymptotic behaviour of $\alpha$-divergence generalized posteriors

    [https://arxiv.org/abs/2610.35330](https://arxiv.org/abs/2610.35330)

    该论文证明α-散度正则化的广义后验经重缩放后收敛于非高斯极限分布——α<1时为Student-t分布且收缩快于经典n^{-1/2}速率，α>1时为紧支撑Pearson II型分布且收缩更慢——而KL散度是α-散度族中唯一保持经典集中速率的成员。

    

    我们研究了由α-散度正则化的无约束广义变分推断问题之解的渐近行为。在温和的条件下，极小化子在相差一个标量归一化常数的意义下是显式的，并且我们证明它在某个重新缩放下于L¹中收敛到一个固定的非高斯分布，其围绕经验损失极小化子的集中尺度偏离了经典Bernstein–von Mises定理中典型的n^{-1/2}。当α∈(0,1)且d<2/(1-α)时，极限分布为Student-t分布，且后验宽度比经验极小化子的采样误差收缩得更快；相反，当α>1时，极限分布为紧支撑的Pearson II型分布，且后验宽度比采样误差收缩得更慢。KL散度（α=1）是α-散度族中唯一一个其后验宽度…（原文在此处截断）

    arXiv:2610.35330v1 Announce Type: cross  Abstract: We study the asymptotic behaviour of the solutions to the unconstrained generalized variational inference problem regularized by an $\alpha$-divergence. Under mild conditions, the minimizer is explicit up to a scalar normalizing constant, and we show that it admits a rescaling under which it converges in $L^1$ to a fixed non-Gaussian law, at a concentration scale around the empirical loss minimizer that departs from the typical $n^{-1/2}$ in the classical Bernstein--von Mises theorem. For $\alpha\in(0,1)$ and $d<2/(1-\alpha)$ the limit is a Student-$t$ distribution and the posterior width shrinks faster than the sampling error of the empirical minimizer. Conversely, for $\alpha>1$ the limiting distribution is a compactly supported Pearson type-II distribution, and the posterior width shrinks more slowly than the sampling error. The KL divergence ($\alpha=1$) is the unique member of the $\alpha$-divergence family whose posterior width m
    
[^12]: 超越Wasserstein邻近性：寻找更好的迁移源

    Beyond Wasserstein Proximity: Finding Better Transfer Sources

    [https://arxiv.org/abs/2610.35314](https://arxiv.org/abs/2610.35314)

    该论文证明Wasserstein距离等边缘邻近性指标并不一定能选出最有用的迁移源，提出基于领域适应理论的条件性选择准则，并在加密货币波动率预测实验中表明选择最远而非最近的源能显著提升预测技能，且源选择效用与最终预测模型的质量相互独立。

    

    源选择通常偏爱相似的数据，然而边缘邻近性未必能识别出最有用的预训练源。我们通过领域适应理论和一个回归示例论证了一种条件性的替代方法：在该示例中，更广泛的源变化尽管增大了Wasserstein距离，却改善了估计效果。我们利用十种币安现货资产、六类预测器家族、三个随机种子以及统一的时间切分点，对五天加密货币波动率进行了研究。选择距离最远的而非最近的高斯-Wasserstein源，使MLP的平均持续性相对技能提升12.52个百分点，Transformer提升20.90个百分点，且事后配对日期块区间均为正。前瞻性试点选择在所有六个预测器家族中均优于随机迁移的平均技能。紧凑的循环神经网络模型实现了最低的平均绝对误差，这表明迁移源的选择效用与预测模型本身的质量是相互分离的。回顾性五候选筛选显示了额外的（优势）。

    arXiv:2610.35314v1 Announce Type: new  Abstract: Source selection often favors similar data, yet marginal proximity need not identify the most useful pretraining source. We motivate a conditional alternative through domain-adaptation theory and a regression example in which broader source variation improves estimation despite increasing Wasserstein distance. We examine five-day cryptocurrency volatility using ten Binance spot assets, six forecaster families, three seeds, and shared temporal cutoffs. Selecting the three farthest rather than nearest Gaussian-Wasserstein sources improves mean persistence-relative skill by 12.52 percentage points for MLP and 20.90 for Transformer, with positive post-hoc paired date-block intervals. Forward pilot selection improves mean skill over random transfer in all six families. Compact recurrent models achieve the lowest absolute mean errors, separating selection utility from forecasting quality. Retrospective five-candidate screening shows additional
    
[^13]: 黑箱模拟器的时序差分校准

    Temporal-Difference Calibration of Black-Box Simulators

    [https://arxiv.org/abs/2610.35311](https://arxiv.org/abs/2610.35311)

    该论文提出了一个基于Bellman正交性的无导数时序差分校准框架，通过三种随机方向有限差分算法校准黑箱模拟器，并建立了涵盖收敛性、均方误差界和中心极限定理的完整理论保证，同时揭示了依赖于模拟器设定正确性的有限差分偏差现象。

    

    我们开发了一个无导数的时序差分（TD）框架，用于利用来自目标系统的一步转移和模拟器生成的含噪延续值来校准黑箱模拟器。校准参数通过Bellman正交性来定义，当模拟器设定正确时能够恢复真实的物理参数，当设定存在偏差时则给出策略值的近最优L2逼近。我们提出了三种基于随机方向有限差分的TD算法，它们在查询成本和含噪模拟器评估结果的复用方式上各有不同。我们在四种查询噪声情形下建立了算法的几乎必然收敛性、最后一步迭代的均方误差界以及Polyak-Ruppert平均的中心极限定理。我们的分析揭示了一种依赖于模拟器设定的偏差现象：对于其中两种算法，在设定正确的条件下，TD估计方程的有限差分偏差在校准目标处消失。

    arXiv:2610.35311v1 Announce Type: new  Abstract: We develop a derivative-free temporal-difference (TD) framework for calibrating black-box simulators using one-step transitions from a target system and noisy simulator-generated continuation values. The calibration parameter is defined through Bellman orthogonality, recovering the physical parameter under correct simulator specification and yielding a near-optimal $L_2$ approximation of the policy value under misspecification. We propose three TD algorithms based on random-direction finite differences, distinguished by their query costs and reuse of noisy simulator evaluations. We establish almost-sure convergence, last-iterate mean-square error bounds, and central limit theorems for Polyak-Ruppert averages across four query-noise regimes. Our analysis reveals a specification-dependent bias phenomenon: for two algorithms, the finite-difference bias of the TD estimating equation vanishes at the calibration target under correct specificat
    
[^14]: 身份而非配对：基于嵌入的验证中无分布假设的错误接受控制

    Identities, Not Pairs: Distribution-Free False-Accept Control for Embedding-Based Verification

    [https://arxiv.org/abs/2610.35277](https://arxiv.org/abs/2610.35277)

    该论文揭示了基于嵌入的验证中仅依赖校准集的错误接受阈值在存在“枢纽身份”时的最坏情况局限，并提出 COVE 方法，通过将试验配对置于由探针与参考扩充后的所有配对中进行排名，在无分布假设下实现 O(1/m²) 水平的错误接受率控制。

    

    基于嵌入的验证系统通常从 m 个身份之间的所有冒名顶替配对中校准错误接受阈值，尽管这些配对共享身份。我们刻画了一个最坏情况下的极限：当“枢纽身份”（hub identities）对总体中比例为 ρ 的部分获得高分时，在所有身份可交换分布下边际错误接受率（FAR）至多为 α 的纯校准阈值，在无枢纽的校准集上达到枢纽得分的概率至多为 (m+2)α/(2ρ)。因此，要均匀地接受接近该得分的配对且水平远低于 ρ/m，就需要试验信息。我们提出 COVE 方法，它通过用探针和参考扩充校准单元，在形成的所有 N=C(m+2,2) 个配对中对试验配对进行排名。在身份可交换性下，其边际 FAR 至多为 ⌊αN⌋/N，首次可能的接受发生在 1/N=O(1/m²)。该规则在使用固定对称参照群时同样保持这一保证。

    arXiv:2610.35277v1 Announce Type: new  Abstract: Embedding-based verification often calibrates a false-accept threshold from all impostor pairs among $m$ identities, although these pairs share identities. We characterize a worst-case limit: when hub identities score highly against a fraction $\rho$ of the population, a calibration-only threshold with marginal false-accept rate (FAR) at most $\alpha$ under every identity-exchangeable law reaches the hub score on hub-free calibration sets with probability at most $(m+2)\alpha/(2\rho)$. Uniformly accepting near that score at levels well below $\rho/m$ therefore requires trial information. We propose COVE, which ranks the trial pair among all $N=\binom{m+2}{2}$ pairs formed by augmenting the calibration units with the probe and reference. Under identity exchangeability, its marginal FAR is at most $\lfloor\alpha N\rfloor/N$, with first possible acceptance at $1/N=O(1/m^2)$. The rule also retains this guarantee with a fixed symmetric cohort
    
[^15]: 树模型健康数据合成中的顺序规格与全条件规格比较

    Sequential versus fully conditional specification for tree-based synthesis of health data

    [https://arxiv.org/abs/2610.35163](https://arxiv.org/abs/2610.35163)

    该研究比较了基于决策树的条件分布模型在合成医疗数据时采用顺序规格与全条件规格两种方式的效果，结果显示在隐私风险与数据效用权衡上没有一种规格普遍更优。

    

    用合成数据代替原始数据进行传播，是在遵守开放科学原则的同时保障隐私的一种有前景的方法，尤其是在医疗健康领域。为了跟上研究人员收集此类数据的速度，需要快速且灵活的合成流程。此类流程中一个流行的方法类别是基于由决策树诱导的条件分布模型。这类模型可以通过两种主要方式组合成合成分布：顺序规格和全条件规格。然而，目前尚无关于应使用哪种规格来优化风险-效用权衡的建议。在这项工作中，我们使用这两种规格，通过组合基于树的条件分布模型来合成医疗数据集，并使用多种风险和效用指标比较了所生成合成数据的质量。我们的结果表明，没有普遍占优的选择。

    arXiv:2610.35163v1 Announce Type: new  Abstract: Disseminating synthetic data in place of original ones is a promising way to enforce privacy while complying with open science principles, especially in the healthcare domain. To keep pace with the rate at which researchers collect such data, there is a need for fast and flexible synthesis procedures. A popular class of such procedures is based on models of conditional distributions induced by decision trees. Such models can be combined into a synthetic distribution following two main approaches: sequential specification and fully conditional specification. However, there is currently no recommendation on which specification to use to optimize the risk-utility tradeoffs. In this work, we use both specifications to synthesize medical datasets by combining tree-based models of conditional distributions. We compare the quality of the resulting synthetic data using several metrics of risk and utility. Our results indicate no general superior
    
[^16]: AutoEnergyDesign：一种由大语言模型驱动的自动能源系统设计优化智能体

    AutoEnergyDesign: An LLM-Driven Agent for Automatic Energy System Design Optimization

    [https://arxiv.org/abs/2610.35160](https://arxiv.org/abs/2610.35160)

    该论文提出AutoEnergyDesign框架，利用具备多类型显式记忆和git式自动回退机制的LLM智能体，通过仿真反馈进行上下文学习来自动优化建筑能源系统设计，从而大幅减少黑盒优化所需的仿真评估次数。

    

    设计建筑能源系统是一个高维度、混合整数的搜索问题：当每个设计方案都需要通过长达一年的仿真来评估时，黑盒优化器通常需要数百次评估。本文提出了AutoEnergyDesign，这是一个由大语言模型（LLM）智能体驱动的框架，该智能体对先前的仿真结果进行推理、提出配置变更，并通过上下文从每次结果中学习，同时基于git的机制会自动回退不成功的变更。该智能体将推理模型与显式记忆组织（程序性、语义性、情景性和反思性记忆）、受限动作接口以及基于环境的保留/回退信号相结合。仿真器涵盖八种技术（光伏、微型风机、热电联产、太阳能光热、燃气锅炉、电池、热泵和逆变器），并计算25年净现值成本（NPC），其中包含资本成本、运营成本和残值。

    arXiv:2610.35160v1 Announce Type: new  Abstract: Designing building energy systems is a high-dimensional, mixed-integer search problem: when each design is evaluated by a year-long simulation, black-box optimizers typically need hundreds of evaluations. This paper presents AutoEnergyDesign, a framework in which a Large Language Model (LLM) agent reasons over previous simulation results, proposes configuration changes and learns in context from each outcome, while a git-based mechanism automatically reverts unsuccessful changes. The agent couples a reasoning model with an explicit memory organisation (procedural, semantic, episodic and reflective), a constrained action interface and an environment-grounded keep/revert signal. The simulator covers eight technologies (photovoltaics, micro-wind, combined heat and power, solar thermal, gas boilers, batteries, heat pumps and inverters) and computes the 25-year Net Present Cost (NPC), including capital and operating costs and salvage value. O
    
[^17]: 风险感知的输出控制应成为机器遗忘的核心焦点

    Risk-Aware Output Control Should be the Focus of Machine Unlearning

    [https://arxiv.org/abs/2610.35052](https://arxiv.org/abs/2610.35052)

    本文主张机器遗忘应摒弃以参数空间相似性为目标的传统范式，转而聚焦于具有统计保证的风险感知输出控制，因为这才是操作上有意义、可扩展且可解释的遗忘概念。

    

    机器学习（ML）系统中有据可查的失效模式，包括偏见、毒性、幻觉以及敏感信息泄露，推动了机器遗忘（MU）的兴起。然而，现有工作大多以参数空间的不可区分性为目标，追求遗忘后的模型与在保留数据上从头重新训练的模型尽可能接近。在本文中，我们认为这一目标与实际及用户相关的关注点并不一致：机器遗忘应优先考虑具有统计保证的输出空间控制，而非参数空间的相似性。我们主张，带有统计保证的输出控制才是操作上有意义的遗忘概念，因为它支持严格且稳健的目标，能够扩展到现代部署场景，并可提供可解释的保证。我们通过实证证据以及最近的理论结果来佐证这一立场，尽管这些结果最初并非以此形式提出。

    arXiv:2610.35052v1 Announce Type: cross  Abstract: Well-documented failure modes of machine learning (ML) systems, including bias, toxicity, hallucination, and leakage of sensitive information, has driven the rise of machine unlearning (MU). Most existing work, however, targets parameter-space indistinguishability, seeking unlearned models that are as close as possible to models retrained from scratch on retained data. In this paper, we argue that this objective is misaligned with practical and user-relevant concerns: MU should prioritize statistically guaranteed output-space control rather than parameter-space similarity. We contend that output control with statistical guarantees is the operationally meaningful notion of unlearning, as it supports rigorous and robust objectives, scales to modern deployment settings, and enables interpretable guarantees. We substantiate this position with empirical evidence and with recent theoretical results that, although not originally formulated in
    
[^18]: 宏观蒙德里安保形风险控制

    Macro Mondrian Conformal Risk Control

    [https://arxiv.org/abs/2610.35035](https://arxiv.org/abs/2610.35035)

    本文提出一种结合蒙德里安保形预测与宏观覆盖算法的新方法，通过对用户定义的分组进行平均风险控制，解决了标准保形风险控制在不平衡分布下对稀有对象风险失控的问题。

    

    保形风险控制将保形预测从覆盖设置扩展到对一般单调损失函数的控制，以无分布且与模型无关的方式进行。这使其非常适合具有高维目标的问题，例如本文所研究的语义分割任务。然而，保形风险控制仅提供边际保证，即风险仅在数据分布上平均受控。在不平衡分布下，即使边际保证成立，包含稀有对象的输入的风险可能仍然不受控制。我们提出了一种新方法，借鉴蒙德里安保形预测和宏观覆盖算法，在用户定义的域分组上对风险进行平均控制。该方法依赖于一个分类体系，即为每个数据点分配分组的函数，并根据每个点所属分组的概率，通过相应系数对其风险进行重加权，从而使各分组的风险得到有效控制。

    arXiv:2610.35035v1 Announce Type: new  Abstract: Conformal Risk Control (CRC) extends conformal prediction beyond the coverage setting to the control of general monotone loss functions, in a distribution-free and modelagnostic fashion. This makes it well-suited to problems with high-dimensional targets, such as the semantic segmentation task addressed in this work. However, CRC guarantees only marginal control, meaning that the risk is controlled on average over the data distribution. Under imbalanced distributions, the risk on inputs that contain rare objects may therefore remain uncontrolled, even when the marginal guarantee holds. We propose a new method that controls the risk on average over user-defined groups of the domain, drawing on Mondrian conformal prediction and macro-coverage algorithms. The method relies on a taxonomy, that is, a function that assigns a group to each data point, and reweights each point's risk by a coefficient that depends on its group probability, so tha
    
[^19]: 基于数据增强局部化的保形新颖性检测

    Conformal Novelty Detection via Data-enhanced Localization

    [https://arxiv.org/abs/2610.34999](https://arxiv.org/abs/2610.34999)

    本文提出一种数据增强的局部化保形p值及相应的条件新颖性检测算法，利用数据驱动带宽选择在有限样本下保证错误发现率控制的同时提升了检测功效。

    

    新颖性检测是统计学和机器学习领域的一个重要研究方向。本文关注条件新颖性检测问题，即新颖性产生于不同变量之间的关系。我们首先采用保形推断框架，提出了一种数据增强的局部化保形p值，该p值通过重新校准增强的条件分布估计器而构建。该估计器通过将校准数据和测试数据同时纳入核估计，高效地捕捉条件信息。我们证明了所提出的p值在有限样本下是有效的，并且能够提高检测效率。基于该p值，我们进一步开发了一种新颖的条件新颖性检测算法，并配套一种数据驱动的带宽选择方法，该方法在增强检测功效的同时确保有限样本下的错误发现率控制。模拟数据和真实数据实验均展示了我们方法的优越性。

    arXiv:2610.34999v1 Announce Type: cross  Abstract: Novelty detection is an important area of research in both statistics and machine learning. In this paper, we focus on the conditional novelty detection problem, where novelties arise from the relationship between different variables. We first adopt the conformal inference framework and propose a data-enhanced localized conformal p-value, constructed by recalibrating an enhanced conditional distribution estimator. This estimator efficiently captures conditional information by incorporating both calibration and test data into its kernel estimation. We show that our p-values are valid in finite samples and can improve detection efficiency. Based on this p-value, we then develop a novel conditional novelty detection algorithm, along with a data-driven bandwidth selection method that ensures finite-sample false discovery rate control while enhancing detection power. Both simulated and real data experiments demonstrate the advantages of our
    
[^20]: 扩散网络的噪声传递负担：为什么图像预测可以优于噪声预测

    The Noise-Transmission Burden of Diffusion Network: Why Image Prediction Can Outperform Noise Prediction

    [https://arxiv.org/abs/2610.34834](https://arxiv.org/abs/2610.34834)

    本文揭示了扩散模型噪声预测参数化带来的“噪声传递负担”——网络必须保留输入噪声信息以抵消污染，在低秩输入瓶颈下导致随噪声增长的去噪误差下界，而图像预测不受此限制，从而解释了为什么在有限网络中图像预测可以优于噪声预测。

    

    干净图像预测（$x$-prediction）与噪声预测在总体最优去噪器意义上是等价的，但在有限的扩散网络中却可能表现出很大差异。对于噪声标准差为 $\sigma_t$ 的含噪输入 $Y_t$，噪声预测器 $f_{\varepsilon,t}$ 所诱导的干净图像估计为 $\widehat X_1=Y_t-\sigma_t f_{\varepsilon,t}(Y_t)$。这揭示了一种“噪声传递负担”：预测器必须保留用于抵消输入污染的信息。在对 $d$ 维数据施加高斯污染的情况下，一个秩为 $r<d$ 的线性输入瓶颈即使在不限制下游计算的情况下，也会强加一个 $\sigma_t^2(d-r)$ 的去噪误差下界，而图像预测对应的最优风险则保持有界。EDM 在高噪声水平下会削弱这一瓶颈的影响；而精确的预处理（preconditioning）则能在每个噪声水平上完全消除它。在没有输入瓶颈的情况下，假设含噪图像块可以从模型中被精确仿射恢复……（原文摘要在此处截断）

    arXiv:2610.34834v1 Announce Type: new  Abstract: Clean-image prediction ($x$-prediction) and noise prediction yield equivalent population-optimal denoisers, yet can perform very differently in finite diffusion networks. For noisy input $Y_t$ with noise standard deviation $\sigma_t$, a noise predictor $f_{\varepsilon,t}$ induces the clean-image estimate $\widehat X_1=Y_t-\sigma_t f_{\varepsilon,t}(Y_t)$. This reveals a noise-transmission burden: the predictor must preserve information needed to cancel the input corruption. Under Gaussian corruption of $d$-dimensional data, a linear input bottleneck of rank $r<d$ imposes a denoising-error floor $\sigma_t^2(d-r)$ even with unrestricted downstream computation, while the corresponding optimal risk for image prediction remains bounded. EDM attenuates this bottleneck contribution at high noise; exact preconditioning cancels it at every noise level. Without an input bottleneck, assuming exact affine recovery of the noisy patch from the model r
    
[^21]: 订阅盒服务的客户反馈问卷设计

    Designing Customer Feedback for Subscription Boxes

    [https://arxiv.org/abs/2610.34726](https://arxiv.org/abs/2610.34726)

    针对订阅盒服务的反馈问卷设计问题，本文提出用高斯信号在入选临界点处近似序数评分所携带的信息，以量化因未选中真正最优商品而产生的期望价值损失，从而指导平台决定应询问哪些候选商品以及采用何种评分形式。

    

    问题定义：订阅盒服务通常在选择发送什么商品之前，先请客户对候选商品进行评分。客户回复数据稀缺，而更长的问卷和更细的评分量表会降低完成率。我们研究在平台随后需从候选中选出固定数量商品的情形下，问卷应询问哪些候选商品以及采用何种形式；设计的优劣以未选中真正最优商品所损失的期望价值来衡量。方法/结果：我们用一个高斯信号来近似每个序数评分，该信号在入选与未入选商品的临界点处携带相应信息。当每件商品的回复数量相同且增加时，该代理方法的损失与精确损失之比趋于1；在每件商品1至100条回复的范围内，两者相差约在7%以内。相比之下，若将信息量在先验分布上取平均，则可能使损失估计的偏差达约一半。对于远离临界点的商品，回复的价值起初增长缓慢、随后加速，因此我们的包络规则……（原文在此处截断）

    arXiv:2610.34726v1 Announce Type: cross  Abstract: Problem definition: Subscription box services often ask customers for ratings before choosing what to send. Responses are scarce, and longer questionnaires and finer scales lower completion. We study which candidate items a questionnaire should ask about, and in what format, when the platform then selects a fixed number of them. A design is judged by the expected value lost by not selecting the truly best items. Methodology/results: We approximate each ordinal rating by a Gaussian signal carrying its information at the cutoff between selected and unselected items. As equal numbers of responses per item grow, the ratio of this proxy's loss to the exact loss tends to one. With 1 to 100 responses per item, the two are within about 7 percent. Averaging the information over the prior instead can misstate the loss by about half. For an item far from the cutoff, the value of responses grows slowly at first and then faster, so our envelope rul
    
[^22]: 基于模拟的推断中摘要的内在信息速率

    Intrinsic Information Rates of Summaries in Simulation-Based Inference

    [https://arxiv.org/abs/2610.34703](https://arxiv.org/abs/2610.34703)

    该论文将基于模拟推断中的摘要建模为有损信息信道，证明了所需的信息预算（互信息 $I(Y;S)$）会随后验分布集中而无界增长（即使存在固定维充分统计量），并通过量化后验与概率加权逆定理给出了所需信息速率的上下界。

    

    基于模拟的推断通常将观测值 $Y$ 编码为固定维度的摘要 $S$，而后验分布仅以 $S$ 为条件。虽然完整保留信息通常需要无限的信息量，但任何此类摘要都会对推断施加一个有限的信息预算，而这个预算很少被显式地选择或度量。在本文中，我们通过将摘要建模为从 $Y$ 到 $S$ 的有损信道来使这一预算显式化，并提出两个问题：这一预算必须有多大，以及如何最有效地使用它？首先，我们证明对于参数模型，这一预算（以互信息 $I(Y;S)$ 度量）会随后验分布的集中而无界增长，即使存在固定维度的充分统计量也是如此。更一般地，所需预算由随机后验的分布决定，我们通过将后验量化给出其上界，并通过以每个后验状态的概率进行加权的逆定理给出其下界。其次，我们证明了……（摘要原文在此处截断）

    arXiv:2610.34703v1 Announce Type: new  Abstract: Simulation-based inference often encodes observations $Y$ into a fixed-dimensional summary $S$, with the posterior conditioning only on $S$. While exact preservation often requires infinite information, any such summary imposes a finite information budget on inference which is rarely an explicit choice or measurement. In this paper, we make the budget explicit by modeling the summary as a lossy channel from $Y$ to $S$, and ask two questions: how large must the budget be, and how best is it spent? First, we show for parametric models that this budget, measured by the mutual information $I(Y;S)$, grows unboundedly as the posterior concentrates, even when fixed-dimensional sufficient statistics exist. More generally, the required budget is set by the distribution of the random posterior, and we bound it from above by quantizing posteriors and from below by converses that weight each posterior state by its probability. Second, we show that a
    
[^23]: 扩散模型中记忆化的有效早期检测

    Valid Early Detection of Memorization in Diffusion Models

    [https://arxiv.org/abs/2610.34617](https://arxiv.org/abs/2610.34617)

    该论文提出将分割共形校准与序贯监测相结合的两种方法（逐时间步Bonferroni校准和轨迹级最大记忆化分数直接校准），实现了扩散模型记忆化的无提示自适应早期检测，同时保证轨迹层面的有效误报控制。

    

    扩散模型有时会复现其训练数据中的样本，这种现象被称为记忆化。现有的针对无条件扩散模型的记忆化检测方法通常在预先选定的时间步上运行，因此无法根据不断演化的去噪轨迹自适应地调整检测时机。序贯监测能够实现自适应检测，但在多个时间步上的重复测试使得轨迹层面的误报控制变得复杂。我们研究了将分割共形校准与序贯监测相结合以实现无提示记忆化检测的两种方法：逐时间步的Bonferroni校准，以及对最大记忆化分数的直接轨迹级校准。前者通过将误报预算分配到各时间步来控制重复测试，而后者则直接通过轨迹层面的最大值对“越过阈值”这一事件进行校准。我们在无条件的CIFAR-1（摘要在此处被截断）上评估了这一方法。

    arXiv:2610.34617v1 Announce Type: cross  Abstract: Diffusion models can sometimes reproduce examples from their training data, a phenomenon known as memorization. Existing memorization detection methods for unconditional diffusion models typically operate at a pre-selected timestep and therefore cannot adapt the detection time to the evolving denoising trajectory. Sequential monitoring enables adaptive detection but repeated testing across timesteps complicates trajectory-level false-alarm control. We study two ways of combining split conformal calibration with sequential monitoring for prompt-free memorization detection: per-step Bonferroni calibration and direct trajectory-level calibration of the maximum memorization score. The former controls repeated testing by splitting the false-alarm budget across timesteps, whereas the latter calibrates the event of ever crossing a threshold directly through the trajectory-level maximum. We evaluate this formulation on an unconditional CIFAR-1
    
[^24]: 图到底要用多少才算多？时空预测中的图构建与图暴露

    How Much Graph Is Too Much? Graph Construction and Exposure in Spatio-Temporal Forecasting

    [https://arxiv.org/abs/2610.34611](https://arxiv.org/abs/2610.34611)

    该论文系统解耦了时空预测中“图构建”与“图暴露”两个因素，发现同一领域图在不同融合设计下既可能提升也可能损害预测精度，仅靠后置融合不足以获益，需要引入融合门控才能稳定发挥图信息的作用。

    

    时空图神经网络将节点自身的历史信息与来自图邻居的信息相结合。其核心问题是：图邻居能否在节点自身历史之外提供额外的预测信息。预测精度既取决于邻居节点的选择（图构建），也取决于图信息进入时间模型的方式（图暴露）。本文在美国各州失业率、交通速度和电力需求数据上研究了这两种选择。诊断性设计采用两组对比：在固定领域图的前提下改变图暴露方式，以及在固定后置门控融合的前提下改变图构建方式。仅使用节点自身历史的GRU以及随机化图对照，有助于评估图信息带来的贡献。在统一的变化预测协议下，同一领域图在不同的融合设计下，相对于仅用自身历史的GRU，既可能改善误差也可能恶化误差。仅有后置融合是不够的，而加入融合门控则能带来（摘要在此处截断）。

    arXiv:2610.34611v1 Announce Type: new  Abstract: Spatio-temporal graph neural networks combine a node's history with information from graph neighbors. The central question is whether graph neighbors provide predictive information beyond each node's own history. Forecast accuracy depends on both the choice of neighbors and the way graph messages enter the temporal model. We study these choices as graph construction and graph exposure on U.S. state unemployment, traffic speed, and electricity demand. Our diagnostic design uses two comparisons: varying exposure while holding the domain graph fixed, and varying construction while holding late gated integration fixed. A GRU using only each node's own history and randomized graph controls help assess what the graph contributes. Under a common change forecasting protocol, the same domain graph can improve or worsen error relative to the GRU across integration designs. Late integration alone is insufficient, while adding a fusion gate yields l
    
[^25]: HIFI：一个基于基础模型的高效统计推断框架

    HIFI: A Highly Efficient Framework for Statistical Inference with Foundation Models

    [https://arxiv.org/abs/2610.34550](https://arxiv.org/abs/2610.34550)

    HIFI提出一种两阶段微调框架：第一阶段在大样本上仅对基础模型做一次微调以捕捉数据结构并控制估计偏差，第二阶段在小样本上并行拟合多个轻量级PEFT估计器并结合重采样进行统计推断，从而把原本需要数百次模型拟合的高昂计算开销降至可承受水平。

    

    基础模型为文本和图像等非结构化数据提供了强大的表示能力，而参数高效微调（PEFT）使基础模型向下游任务的适配日益可行。对于缺乏可解析采样分布的神经估计器，基于重采样的方法为统计推断提供了灵活的途径，但这类方法通常需要对模型进行数百次拟合，在基础模型规模下计算上难以承受。我们提出了HIFI，一个用于对条件均值等参数进行统计推断的两阶段微调框架。第一阶段在较大的样本上对基础模型进行一次适配，捕捉数据的复杂结构并通过单次拟合控制估计偏差。第二阶段在较小的独立样本上应用重采样推断程序，同时拟合多个轻量级PEFT估计器来拟合残差，从而无需重新学习即可量化估计的不确定性……（原文摘要在此处截断）

    arXiv:2610.34550v1 Announce Type: new  Abstract: Foundation models provide powerful representations for unstructured data, such as text and images, and parameter-efficient fine-tuning (PEFT) makes their adaptation to downstream tasks increasingly practical. For neural estimators that lack a tractable sampling distribution, resampling-based procedures offer a flexible route to statistical inference, but they often require hundreds of model fits, making them computationally prohibitive at foundation-model scale. We introduce HIFI, a two-stage fine-tuning framework for statistical inference on parameters such as conditional means. Stage 1 adapts a foundation model once on a larger sample, capturing the complex structure of the data and controlling estimation bias through a single fit. Stage 2 applies a resampling inference procedure to a smaller independent sample, concurrently fitting multiple lightweight PEFT estimators to the residual to quantify estimation variability without relearni
    
[^26]: 潜在位置随机图的联合独立性检验

    Joint Independence Testing for Latent Position Random Graphs

    [https://arxiv.org/abs/2610.34544](https://arxiv.org/abs/2610.34544)

    本文提出基于核方法的潜在位置随机图联合独立性检验，利用格拉姆矩阵以 $O(n^{-1/2})$ 速率收敛的特性来度量图中节点潜在表示之间的独立性。

    

    核方法是数据科学与统计学中最强大的方法之一。核方法之所以在众多应用中取得成功，其关键特性之一在于：在温和的条件下，与之对应的再生核希尔伯特空间具有足够的表达能力，既能无损地编码概率分布，又能捕捉 $M\ge 2$ 个随机变量的联合独立性。在计算方面，这些度量：(i) 依赖于所谓的格拉姆矩阵，其中包含样本点之间的两两相似度；(ii) 以 $O(n^{-1/2})$ 的速率收敛，其中 $n$ 表示样本量。我们专注于度量潜在位置随机图的独立性，该模型被广泛应用于神经科学、社会学和流行病学等领域。在潜在位置随机图中，每个节点都关联着一个潜在表示，这些表示之间的相似度（通过核来度量）决定了两个节点是否……

    arXiv:2610.34544v1 Announce Type: new  Abstract: Kernel techniques are among the most powerful methods in data science and statistics. One of the key properties of kernels contributing to their numerous successful applications is that the associated reproducing kernel Hilbert space is expressive enough to encode probability distributions without loss of information or to capture the joint independence of $M\ge 2$ random variables, under mild conditions. Computationally, these measures (i) rely on the so-called Gram matrix which contains the pairwise similarity of sample points, and (ii) converge at a rate $O \left(n^{-\frac{1}{2}}\right)$, where $n$ denotes the sample size. We focus on measuring the independence of latent position random graphs (LPRGs), a model widely-applied for instance in neuroscience, sociology, and epidemiology. In LPRGs a latent representation is associated to each node. The similarity of these representations (meant via a kernel) determines whether two nodes are
    
[^27]: 模糊的拟合：罗生门效应与缩放定律估计中的挑战

    Ambiguous Fits: The Rashomon Effect and Challenges in Scaling Law Estimation

    [https://arxiv.org/abs/2610.34514](https://arxiv.org/abs/2610.34514)

    本文揭示了神经缩放定律拟合中存在“罗生门效应”——多个本质不同的参数选择都能几乎同样好地拟合数据，但其外推结果在大尺度上差异巨大，并提出确定性分支定界算法来解决这一模糊性问题。

    

    估计神经缩放定律涉及一个端到端的流程：设计缩放方案、跨尺度训练模型、在目标指标上评估模型，以及部署一种算法将缩放定律拟合到所得数据上。这最后的拟合阶段通常被视为常规操作：标准流程使用从初始点网格出发的局部优化来最小化一个非凸拟合损失，并返回找到的最优解。然而，这种启发式搜索可能以两种截然不同的方式失败：它们无法保证返回近似最优的解，并且掩盖了本质上不同的参数选择是否能够几乎同样好地拟合数据。事实上，我们证明了标准的缩放定律拟合存在罗生门效应（Breiman, 2001）：近最小化器的次水平集对应于宽广的参数区域，这些区域在大尺度上的外推结果会出现实质性分歧。我们通过一种确定性的分支定界算法来解决这些问题。

    arXiv:2610.34514v1 Announce Type: cross  Abstract: Estimating neural scaling laws involves an end-to-end pipeline: designing a scaling recipe, training models across scales, evaluating models on a target metric, and deploying an algorithm to fit a scaling law to the resulting data. This final fitting stage is often treated as routine: standard procedures minimize a non-convex fitting loss using local optimization from a grid of initializations, and return the best solution found. Yet such heuristic searches can fail in two distinct ways: they offer no guarantee of returning a nearly optimal solution, and they obscure whether fundamentally different parameter choices fit the data nearly as well. Indeed, we show that standard scaling law fitting is subject to a Rashomon effect (Breiman, 2001): the sublevel sets of near-minimizers correspond to broad parameter regions whose extrapolations diverge substantially at scale. We resolve these issues with a deterministic branch-and-bound algorit
    
[^28]: 主动记忆召回中规模的代价

    The Price of Scale in Proactive Memory Recall

    [https://arxiv.org/abs/2610.34507](https://arxiv.org/abs/2610.34507)

    该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。

    

    一个无需被询问便能主动提起记忆的长期助手，必须在每轮交互中判断其记忆库中是否有内容值得打断用户。我们证明这一决策本质上是一个多重假设检验问题，其难度随记忆库规模 N 的增长而加剧。在 LongMemEval 与 LoCoMo 的流式回放实验中，一个在 N=8 时调校至 5% 误报率的相似度阈值，在 N=2048 时误报率飙升至 87–100%，该现象在各类嵌入模型以及显式与隐式线索中均成立；LLM 判别器虽起点更低，也呈现同样的增长趋势。我们证明，一个误报率为 α 的门控机制，其单条记忆的假阳性率约为 -ln(1-α)/N，这将把其命中率锁定在单次比较 ROC 曲线的最左端，且在正则变化条件下，命中率随 N 以幂次规律衰减并带有显式常数。由此得出的预测与逐 N oracle 阈值的命中率相差在一个百分点以内。该分析解释了为何词汇与语义通道……（原文摘要在此处截断）

    arXiv:2610.34507v1 Announce Type: cross  Abstract: A long-lived assistant that brings up memories without being asked must decide at every turn whether anything in its store is worth an interruption. We show that this decision is a multiple-testing problem whose difficulty grows with the store size $N$. On streaming replays of LongMemEval and LoCoMo, a similarity threshold tuned to 5% false alarms at $N=8$ reaches 87-100% at $N=2048$, across embedding models and for explicit and implicit cues; LLM judges show the same growth from a lower base. We prove that a gate with false-alarm rate $\alpha$ operates at a per-item false-positive rate of about $-\ln(1-\alpha)/N$, which pins its hit rate to the far left of the single-comparison ROC curve and, under regular variation, makes it decay as a power of $N$ with an explicit constant. The resulting prediction matches the hit rate of a per-$N$ oracle threshold to within one percentage point. The analysis explains why lexical and semantic channe
    
[^29]: 从投影观测中认证分布鲁棒决策

    Certifying Distributionally Robust Decisions from Projection Observations

    [https://arxiv.org/abs/2610.34493](https://arxiv.org/abs/2610.34493)

    该论文提出一种置信集证书方法，仅凭标量投影观测即可在不识别潜在协方差与各决策风险的情况下认证分布鲁棒决策，并证明了其健全性、必要弃权条件与固定检查点认证保证。

    

    标量测量能够在不识别潜在协方差或每个决策风险的情况下确定一个鲁棒决策。我们研究了具有固定“均值加标准差”目标的有限高斯投影实验。观测规律定义了一组不可区分的矩模型。我们展示了尽管风险值未被识别、但在该模型集合上保持一致的决策，并给出一种置信集证书，该证书返回一个决策或一个“未解决”的结果。我们证明了该方法的健全性、对于具有不相容最优解的不可区分模型的必要弃权，以及固定检查点的认证保证。一种标量轮廓方法将非线性风险比较简化为凸内问题。在七个合成实例中，联合校准的矩区域给出了比显式期望样本信息下界充足21至52倍的预算。单独测量的停止时间更低，并在1,900个模拟数据流中实现了零决策错误。这些结果……

    arXiv:2610.34493v1 Announce Type: cross  Abstract: A scalar measurement can determine a robust decision without identifying the latent covariance or every decision risk. We study finite Gaussian projection experiments with a fixed mean-plus-standard-deviation target. The observation laws define a set of indistinguishable moment models. We exhibit decisions that agree throughout that set despite unidentified risk values, and give a confidence-set certificate that returns a decision or an unresolved outcome. We prove soundness, necessary abstention for indistinguishable models with incompatible optima, and a fixed-checkpoint certification guarantee. A scalar profile reduces nonlinear risk comparisons to convex inner problems. Across seven synthetic instances, jointly calibrated moment regions give sufficient budgets 21 to 52 times explicit expected-sample information lower bounds. Separately measured stopping times are lower, with zero decision errors in 1,900 simulated streams. These co
    
[^30]: 前推深度生成模型中的内在梯度障碍

    Intrinsic Gradient Barriers in Pushforward Deep Generative Models

    [https://arxiv.org/abs/2610.34484](https://arxiv.org/abs/2610.34484)

    本文证明了深度生成模型在表示重尾和多峰分布时的困难源于传输问题本身的内在属性而非模型架构或训练方法，并对前推映射的梯度和雅可比矩阵建立了定量下界。

    

    许多类型的深度生成模型（DGMs）通过学习将简单的源分布（如标准高斯分布）映射到复杂的高维目标分布来运行。尽管取得了成功，深度生成模型在经验上往往难以准确表示重尾分布或峰间距较大的多峰分布。在这项工作中，我们证明了这些困难反映的是底层传输问题的内在属性，而非特定的神经网络架构或训练过程。我们对将对数凹源分布传输到此类目标分布的前推映射的梯度和雅可比矩阵建立了定量下界。对于重尾目标分布，所需的梯度幅值会随着目标分布矩的增大而增长。对于多峰目标分布，分离良好的模式之间的低概率区域同样会迫使传输映射中产生大梯度。我们讨论……

    arXiv:2610.34484v1 Announce Type: new  Abstract: Broad classes of deep generative models (DGMs) operate by learning to map a simple source distribution, such as a standard Gaussian, to a complex high-dimensional target distribution. Despite their success, DGMs often empirically struggle to accurately represent distributions with heavy tails or well-separated modes. In this work, we show that these difficulties reflect intrinsic properties of the underlying transport problem rather than particular neural network architectures or training procedures. We establish quantitative lower bounds on the gradients and Jacobians of pushforward mappings that transport log-concave source distributions to such targets. For heavy-tailed target distributions, the required gradient magnitude grows as the target develops increasingly large moments. For multimodal target distributions, the low-probability regions between well-separated modes similarly force large gradients in the transport map. We discuss
    
[^31]: 具有重复处理与多个中介变量的纵向研究中因果路径特定效应的推断

    Inference for Causal Path-Specific Effects in Longitudinal Studies with Repeated Treatments and Multiple Mediators

    [https://arxiv.org/abs/2610.34349](https://arxiv.org/abs/2610.34349)

    该论文针对重复处理后中介变量与时变协变量序列可变的纵向研究，发展了路径特定因果效应的半参数推断方法，通过边g-公式识别效应，并给出有效影响函数的两种参数化形式，其中基于序贯回归的方法避免了对中间变量密度的直接估计。

    

    在纵向研究中，重复施加的处理之后可能伴随不同的中介变量和时变协变量序列。我们在这一设定下发展了路径特定效应的半参数推断方法，允许中间变量的数量、类型和排序在不同的处理时点上有所变化。在一致性、正性、边一致性以及具有独立误差的非参数结构方程模型的跨世界约束条件下，路径特定均值可通过边g-公式得到识别。我们推导了识别泛函的有效影响函数的两种参数化形式：第一种遵循观测数据的因子分解，使用条件中间变量密度；第二种利用序贯回归替代相应的积分运算，并通过贝叶斯法则借助处理历史回归来表达密度比，从而避免了对中间变量密度的直接估计。

    arXiv:2610.34349v1 Announce Type: cross  Abstract: In longitudinal studies, repeated treatments may be followed by different sequences of mediators and time-varying covariates. We develop semiparametric inference for path-specific effects in this setting, allowing the number, type, and ordering of intermediate variables to vary across treatment occasions. Under consistency, positivity, edge consistency, and the cross-world restrictions of a nonparametric structural equation model with independent errors, path-specific means are identified by the edge g-formula. We derive two parameterizations of the efficient influence function of the identifying functional. The first follows the observed-data factorization and uses conditional intermediate-variable densities. The second uses sequential regressions to replace the corresponding integrations and Bayes' rule to express density ratios through treatment-history regressions, thereby avoiding direct estimation of intermediate-variable densiti
    
[^32]: 基于稳定测量机制的保形预测

    Conformal Prediction with Stable Measurement Mechanisms

    [https://arxiv.org/abs/2610.34204](https://arxiv.org/abs/2610.34204)

    该论文提出在“测量机制稳定”的因果假设下，将保形预测区间从拥有参考结果的源数据适配到仅有传感器读数的目标总体，并分析了这一稳定性限制对预测覆盖率与区间长度的影响。

    

    在空气质量监测等应用中，廉价的传感器读数可以帮助预测由独立参考仪器测量的污染物浓度。我们研究当源数据中可以获得参考结果，而目标数据中仅有传感器读数和上下文信息（如天气条件）时，如何将保形预测区间适配到新的总体。我们假设在给定结果和上下文的条件下，测量分布保持稳定，而上下文频率和结果分布可能发生变化。该模型是因果机制偏移的一种特殊形式，其统计限制已被作为稀疏联合偏移加以研究。利用已有的边缘匹配、预测调整和加权保形校准方法，我们考察了这一限制对连续结果的覆盖率和区间长度的作用。高斯分析展示了测量稳定性如何识别结果变化。

    arXiv:2610.34204v1 Announce Type: cross  Abstract: In applications such as air-quality monitoring, inexpensive sensor readings can help predict pollutant concentrations measured by a separate reference instrument. We study how to adapt conformal prediction intervals to a new population when reference outcomes are available in the source data, but only sensor readings and context, such as weather conditions, are available in the target. We assume that the measurement distribution at a given outcome and context remains stable, while context frequencies and outcome distributions may change. This model is a particular form of causal mechanism shift. Its statistical restriction has been studied as sparse joint shift. Using established marginal matching, predictive adjustment, and weighted conformal calibration, we examine what this restriction contributes to coverage and interval length for continuous outcomes. Gaussian analyses show how measurement stability can identify outcome changes an
    
[^33]: MetaARIMA：面向ARIMA模型的配置级迁移学习

    MetaARIMA: Configuration-Level Transfer Learning for ARIMA Models

    [https://arxiv.org/abs/2610.34171](https://arxiv.org/abs/2610.34171)

    MetaARIMA通过元学习在配置层面（而非模型层面）为ARIMA模型引入迁移学习，从历史实验中学习高性能配置模式，为新序列推荐合适的ARIMA配置，同时保留了经典方法的可解释性和局部估计能力。

    

    时间序列基础模型已经证明，在多样化数据集上进行预训练能够有效迁移到新的预测任务中。然而，像ARIMA这样的经典方法仍然需要对每个序列独立进行配置和训练，丢弃了以往实验积累的知识。因此，选择和训练ARIMA配置仍然具有挑战性，且通常依赖于基于样本内拟合优度指标的启发式方法。我们提出了MetaARIMA，这是一种基于元学习的预训练配置策略，将迁移学习引入经典统计预测中，但作用于配置层面而非模型层面。该策略从历史实验中学习哪些ARIMA配置能够达到顶尖性能水平，并为新序列推荐有前景的候选配置，同时保持可解释性和局部估计能力。在推理阶段，它通过最大边际相关性选择一组多样化的候选配置。

    arXiv:2610.34171v1 Announce Type: new  Abstract: Time Series Foundation Models have demonstrated that pretraining on diverse datasets enables effective transfer to new forecasting tasks. Classical methods like ARIMA, however, remain configured and trained independently for each series, discarding knowledge from previous experiments. Consequently, selecting and training an ARIMA configuration remains challenging and often relies on heuristic approaches based on in-sample goodnessof-fit measures. We propose MetaARIMA, a pretrained configuration policy based on metalearning that brings transfer learning to classical statistical forecasting at the configuration level rather than the model level. The policy learns from historical experiments which ARIMA configurations achieve top-quantile performance, and recommends promising candidates for new series while preserving interpretability and local estimation. At inference, it selects a diverse set of candidates via maximal marginal relevance a
    
[^34]: 函数型数据的合成数据增强：Wasserstein风险界与基于扩散模型的条件生成

    Synthetic Data Augmentation for Functional Data: Wasserstein Risk Bounds and Diffusion-Based Conditional Generation

    [https://arxiv.org/abs/2610.34120](https://arxiv.org/abs/2610.34120)

    该论文首次为函数型数据的合成数据增强建立了基于类条件扩散生成器的理论框架，在Wasserstein-1距离下给出训练风险的上下界，推导出最优合成样本数量以及合成数据优于真实数据的条件，并在类别不平衡场景下给出了无需样本分割的额外风险保证。

    

    函数型数据的采集成本高昂，样本通常仅有数十到数百条曲线。合成数据增强可以缓解这一问题，但现有理论仅针对有限维数据发展而来，且依赖于全变差距离——而全变差对函数数据而言不具信息量。我们研究基于类条件函数型扩散生成器的数据增强方法，并以Wasserstein-1距离给出理论保证。对于稳定的下游估计器，我们从上界和下界两个方向将基于合成曲线训练的风险界定为基于相同数量真实曲线训练的风险加上一个生成惩罚项，并由此推导出最佳合成曲线数量，以及合成数据优于真实数据的条件。针对类别不平衡情形下合成曲线的分配问题，我们在无需样本分割的情况下，对同时使用真实与合成曲线训练的额外风险给出了界定。在温度、高光谱和步态曲线的实际应用中，数据增强提升了分类和函数到标量（回归）的性能。

    arXiv:2610.34120v1 Announce Type: cross  Abstract: Functional data are costly to collect; samples of tens to hundreds of curves are common. Synthetic data augmentation can help, but its theory is developed for finite-dimensional data and relies on total variation, which is uninformative for functions. We study augmentation with a class-conditional functional diffusion generator and state guarantees in the Wasserstein-1 distance. For stable downstream estimators, we bound the risk of training on synthetic curves, from above and below, by the risk of training on as many real curves plus a generation penalty, and derive the best number of synthetic curves and a condition under which synthetic data outperform real data. For allocating synthetic curves under class imbalance, we bound the excess risk of training on real and synthetic curves together, without sample splitting. In applications to temperature, hyperspectral and gait curves, augmentation improved classification, function-to-scal
    
[^35]: 基于索伯列夫核的强化学习的极小极大遗憾

    Minimax Regret for Sobolev Kernel-Based Reinforcement Learning

    [https://arxiv.org/abs/2610.34045](https://arxiv.org/abs/2610.34045)

    该论文首次证明了连续状态-动作空间中基于索伯列夫核的强化学习的极小极大遗憾率，并提出了能达到该最优遗憾率的SOVI算法。

    

    我们研究了情景式核强化学习的极小极大遗憾率，其中期望奖励函数和转移密度属于索伯列夫再生核希尔伯特空间，状态空间和动作空间连续，且初始状态是对抗性的。对于索伯列夫光滑参数 $\nu>0$、状态维度 $d_S$ 和动作维度 $d_A$，我们证明了在 $T$ 个情景上的极小极大期望遗憾为 $\widetilde\Theta(T^{\max\{e_A,e_B\}})$，其中 $e_A=\frac{\nu+d_A}{2\nu+d_A}$，$e_B=\frac{\nu+d_A+3d_S/2}{2\nu+d_A+2d_S}$（其他模型参数固定）。这两个指数分别反映了动作选择和跨状态学习这两种不同的困难。我们提出了索伯列夫最优值迭代算法（SOVI），该算法能够从观测到的奖励和转移中达到这一遗憾率。在光滑度和参考核长度尺度固定的情况下，其遗憾界的预因子是关于状态-动作维度 $d=d_S+d_A$、视野 $H$ 等参数的多项式。

    arXiv:2610.34045v1 Announce Type: cross  Abstract: We study the minimax regret rate for episodic kernel-based reinforcement learning, where the expected reward and transition density belong to a Sobolev reproducing kernel Hilbert space, with continuous state and action spaces, and adversarial initial states. For Sobolev smoothness parameter $\nu>0$, state dimension $d_S$, and action dimension $d_A$, we prove that the minimax expected regret over $T$ episodes is $\widetilde\Theta(T^{\max\{e_A,e_B\}})$, where $e_A=\frac{\nu+d_A}{2\nu+d_A}$, $e_B=\frac{\nu+d_A+3d_S/2}{2\nu+d_A+2d_S}$ with the other model parameters fixed. The two exponents reflect the distinct difficulties of action selection and learning across states. We introduce Sobolev Optimal Value Iteration (SOVI), which attains this rate from observed rewards and transitions. For fixed smoothness and reference-kernel lengthscale, its regret bound has a prefactor polynomial in the state-action dimension $d=d_S+d_A$, horizon $H$, tr
    
[^36]: 用于上下文离线策略评估的先验数据拟合网络

    Prior-Data Fitted Networks for In-Context Off-Policy Evaluation

    [https://arxiv.org/abs/2610.34035](https://arxiv.org/abs/2610.34035)

    提出InCOPE，一种基于先验数据拟合网络的上下文离线策略评估方法，通过在合成OPE任务上预训练学习共享的策略-价值预测规则，能够在无需任务特定参数更新的情况下直接预测新表格决策问题中目标策略的期望回报。

    

    离线策略评估（OPE）对于在在线实验成本高昂或不安全的情况下、在部署前评估决策策略至关重要。当行为策略数据有限，且目标策略偏向于这些数据中很少出现的状态和动作时，准确估计仍然十分困难。我们提出了InCOPE，一种先验数据拟合网络，它在多个合成OPE任务上学习一个共享的策略-价值预测规则。我们在马尔可夫决策过程以及行为-目标策略对之上构建了OPE任务先验，生成用于预训练的行为策略数据集和精确的目标策略价值。对于一个新的表格决策问题，预训练网络以行为策略数据集、目标策略和已知任务设置作为上下文，直接预测该策略的期望回报，而无需进行任务特定的参数更新。在四个表格领域（包括一个半合成临床环境）上的实验证明了InCOPE的有效性。

    arXiv:2610.34035v1 Announce Type: new  Abstract: Off-policy evaluation (OPE) is essential for assessing decision policies before deployment when online experimentation is costly or unsafe. Accurate estimation remains difficult when behavior-policy data are limited and the target policy favors states and actions rarely observed in those data. We propose InCOPE, a prior-data fitted network that learns a shared policy-value prediction rule across synthetic OPE tasks. We construct an OPE task prior over Markov decision processes and behavior-target policy pairs, generating behavior-policy datasets and exact target-policy values for pretraining. For a new tabular decision problem, the pretrained network uses the behavior-policy dataset, target policy, and known task settings as context to predict the policy's expected return without task-specific parameter updates. Experiments across four tabular domains, including a semi-synthetic clinical setting, demonstrate InCOPE's effectiveness for po
    
[^37]: 表征形成的信息几何

    The Information Geometry of Representation Formation

    [https://arxiv.org/abs/2610.34015](https://arxiv.org/abs/2610.34015)

    该论文将表征形成过程本身作为研究对象，用Bures几何与量子Fisher信息刻画训练中表征的几何轨迹，发现学习会将局部敏感性集中于任务相关因素，且当所学计算需要更高阶的输入交互时，表征轨迹会系统性地变得更不直接。

    

    大多数对神经表征的分析都聚焦于训练完成后的最终状态，而内部结构形成的过程却相对缺乏测量。我们转而将表征形成过程本身作为研究对象，探究表征在训练中所走的路径是否揭示了其终点无法提供的信息。我们通过将表征学习视为一条几何轨迹来形式化这一想法，利用Bures几何来度量在表征空间中的移动，并用量子Fisher信息来刻画对输入因素的局部敏感性。我们发现，学习会使局部敏感性集中于与任务相关的因素，并且仅改变预测目标就会使这种几何结构围绕相应的视觉因素进行重组。更引人注目的是，随着所学计算需要输入因素之间的高阶交互，表征轨迹会系统性地变得不那么直接。该效应在（原文摘要在此处截断）……条件下依然持续存在。

    arXiv:2610.34015v1 Announce Type: cross  Abstract: Most analyses of neural representations focus on trained endpoints, leaving the process by which internal structure forms comparatively unmeasured. We instead treat representation formation itself as an object of study, asking whether the path a representation takes through training reveals information that its endpoint does not. We formalize this idea by viewing representation learning as a geometric trajectory, using Bures geometry to measure movement through representation space and quantum Fisher information to characterize local sensitivity to input factors. We find that learning concentrates local sensitivity on task-relevant factors, and that changing only the prediction target reorganizes this geometry around the corresponding visual factor. More strikingly, representation trajectories become systematically less direct as the learned computation requires higher-order interactions among input factors. This effect persists under 
    
[^38]: 部分模态张量回归

    Partial Modal Tensor Regression

    [https://arxiv.org/abs/2610.34008](https://arxiv.org/abs/2610.34008)

    提出部分模态张量回归方法（PMTR），通过构造融合条件模态信息的部分模态张量协方差来指导张量降维，为非对称、重尾或含异常值的响应数据提供稳健且可解释的张量回归建模方案。

    

    随着现代科学研究的快速发展，张量数据在众多领域中日益普遍。因此，如何在保留张量数据内在结构的同时，有效刻画张量预测变量与响应变量之间的关系，已成为高维统计分析中的一个重要问题。现有的张量回归方法主要关注条件均值或条件分位数。当响应变量的分布呈现非对称、重尾或受到异常观测污染时，条件模态提供了一种稳健且可解释的替代性刻画方式。在本文中，我们提出了一种部分模态张量回归（PMTR）方法。该方法构造了部分模态张量协方差，将条件模态信息纳入其中以指导张量降维，并在此基础上建立了模态张量回归模型……

    arXiv:2610.34008v1 Announce Type: new  Abstract: With the rapid development of modern scientific research, tensor data have become increasingly prevalent across a wide range of fields. How to effectively characterize the relationship between tensor predictors and a response while preserving the intrinsic structure of tensor data has therefore become an important problem in high-dimensional statistical analysis. Existing tensor regression methods mainly focus on conditional means or conditional quantiles. When the response distribution is asymmetric, heavy-tailed, or contaminated by outlying observations, the conditional modal provides a robust and interpretable alternative characterization. In this paper, we propose a partial modal tensor regression (PMTR) method. The proposed method constructs the partial modal tensor covariance, which incorporates information on the conditional modal to guide tensor dimension reduction, and subsequently establishes a modal tensor regression model in 
    
[^39]: 约束梯度均衡：局部约束对手下的可达性

    Constrained Gradient Equilibrium: Approachability under Locally Constrained Adversaries

    [https://arxiv.org/abs/2610.33985](https://arxiv.org/abs/2610.33985)

    该论文研究了带约束的梯度均衡问题，证明了其在简单约束下可能不可实现，进而针对局部约束对手建立了基于Blackwell可达性思想的充要刻画框架，并提出了在带裕量局部可行性条件下将无遗憾学习器转换为CGEQ算法的通用原则。

    

    梯度均衡（GEQ）是近期在线学习领域提出的一个框架，要求沿学习者轨迹的时间平均梯度趋近于零。我们研究学习者受约束情形下的GEQ问题，并探讨在对抗性环境中这种均衡何时仍然可以实现。我们首先证明了约束梯度均衡（CGEQ）即使在简单且缓慢变化的约束下也可能无法实现，这一结果促使我们对对手施加局部时间结构上的限制。随后，我们针对局部约束对手下的CGEQ建立了一个可达性框架，并以经典Blackwell可达性的思想为基础，获得了充要条件的刻画。除这些理论刻画外，我们还提出了一种算法原则，在带裕量的局部可行性条件下，能够将逐窗口的无遗憾学习器转换为CGEQ算法。在合成数据与真实数据上的实验表明，由此得到的算法表现出色。

    arXiv:2610.33985v1 Announce Type: cross  Abstract: Gradient equilibrium (GEQ) is a recent framework in online learning that requires the time-averaged gradients along the learner's trajectory to vanish. We study GEQ when the learner is constrained and ask when such an equilibrium remains achievable in an adversarial environment. We first show that constrained GEQ (CGEQ) can be impossible even under simple and slowly varying constraints, motivating imposing local temporal structure on the adversary. We then develop an approachability framework for CGEQ under locally constrained adversaries, and obtain necessary and sufficient characterizations in the spirit of classical Blackwell approachability. Beyond these theoretical characterizations, we provide an algorithmic principle that converts window-wise no-regret learners into CGEQ algorithms under a local feasibility condition with margin. Experiments on synthetic and real data demonstrate that the resulting algorithms perform strongly in
    
[^40]: 一种用于联合稀疏样本加权与参数估计的非凸框架

    A Nonconvex Framework for Joint Sparse Sample Weighting and Parameter Estimation

    [https://arxiv.org/abs/2610.33957](https://arxiv.org/abs/2610.33957)

    提出了一种联合稀疏样本加权与参数估计（SWAPE）的非凸框架，通过交替最小化和主化-最小化近似技术同时优化模型参数与每个样本的稀疏权重，并证明了算法的收敛性。

    

    我们提出了一种联合稀疏样本加权与参数估计（SWAPE）框架，该框架联合优化模型参数和样本权重，在训练模型的同时为每个训练数据点分配单独的权重。我们的框架适用于多种广泛使用的模型参数损失函数，包括凸、弱凸和凸差损失，以及用于控制样本权重复杂度的弱凸稀疏函数。为了求解由此产生的二元问题，我们提出了一种交替最小化方法，其中更新模型参数和权重的子问题各自使用主化-最小化技术进行近似。我们为本文考虑的损失函数提供了几种近似方法，并证明了整体算法的收敛性。在各种学习任务和数据异常设置上的数值实验证明了所提出框架的有效性。

    arXiv:2610.33957v1 Announce Type: cross  Abstract: We propose a joint sparse sample weighting and parameter estimation (SWAPE) framework that jointly optimizes model parameters and sample weights to train a model while simultaneously assigning an individual weight to each training data point. Our framework accommodates several widely used loss functions for the model parameters, including convex, weakly convex, and difference-of-convex loss, and weakly convex sparsity functions to control the complexity of the sample weights. To solve the resulting bivariate problem, we propose an alternating minimization method in which the subproblems for updating the model parameters and weights are each approximated using majorize-minimize techniques. We provide several approximation methods for the loss functions considered in this paper, and show the overall algorithm's convergence. Numerical experiments on various learning tasks and data anomaly settings demonstrate the effectiveness of the prop
    
[^41]: 标签噪声下早停止何时有益？核梯度流的相图

    When is early stopping beneficial under label noise? A phase diagram for kernel gradient flow

    [https://arxiv.org/abs/2610.33846](https://arxiv.org/abs/2610.33846)

    该论文通过在可解的核梯度流设定中推导高维学习曲线并构建相图，精确刻画了标签噪声下测试误差“后期恶化”现象消失的条件，从而解释了为什么深度网络即使完全记住噪声标签也无需早停止。

    

    早停止（early stopping）通常被用来防止对标签噪声的拟合损害泛化性能。这是否意味着，当训练标签中有一定比例本质上不可预测时，最佳测试误差必然出现在插值之前？在带有随机标签样本的CIFAR10分类任务上的实验激发了相反的可能性：尽管固定特征模型和浅层模型表现出测试误差在训练后期的恶化，但在更大、更深的卷积网络中，即使随机标签已全部被正确分类，对早停止的需求也基本消失了。我们在一个可求解的设定中确定了后期恶化完全消失的条件。对于合成噪声分类任务上的核梯度流，我们推导出精确的高维学习曲线，并得到了一个相图，该相图由学习输入与输出之间共享关系的线性核分量与其他分量之间的平衡所控制……

    arXiv:2610.33846v1 Announce Type: cross  Abstract: Early stopping is commonly used to prevent fitting label noise from degrading generalization. Does this mean that, when a positive fraction of the training labels is intrinsically unpredictable, the best test error must occur before interpolation? Experiments on a CIFAR10 classification task with randomly labeled examples motivate the opposite possibility: although fixed-feature and shallow models exhibit late-time deterioration of the test error, the need for early stopping essentially disappears in larger, deeper convolutional networks even after the random labels have all been classified correctly. We determine when late-time deterioration disappears entirely in a solvable setting. For kernel gradient flow on a synthetic noisy classification task, we derive sharp high-dimensional learning curves and obtain a phase diagram controlled by the balance between the linear kernel component that learns the shared relation between inputs and
    
[^42]: 交互作用下的近似通用预测

    Approximate Universal Prediction under Interaction

    [https://arxiv.org/abs/2610.33842](https://arxiv.org/abs/2610.33842)

    本文研究了通用预测器（如Solomonoff归纳）的近似保证在与环境交互反馈下能否保留，证明迁移系数可分解为策略似然比与任务证据的乘积，导致任务稀有性与动作分支稀有性复合，并在固定通用混合下存在无穷多可计算环境使预测误差达到 κη·2^(K(μ)) 的紧量级。

    

    贝叶斯混合，特别是Solomonoff归纳，提供了一种通过最小化总体对数损失来学习预测的理想化模型，并且是诸如AIXI等通用智能体的基础。一个使用此类预测器近似进行规划的智能体会决定该预测器在何处被评估，而规划器可能会被引向其模型表现乐观的分支上。我们研究了哪些近似保证能够在这种反馈作用下得以保留。对于归一化的有限时域因果混合，迁移系数可以分解为策略似然比和每次预测之前可获得的任务证据，因此任务稀有性和动作分支稀有性可能会复合叠加。在同一个固定的通用混合中，存在无穷多个可计算环境，其超出预测误差达到 κη·2^(K(μ)) 的量级，其中 η 是训练KL预算，κ 衡量策略覆盖率。这一量级是紧的，并且对于每个目标，都存在一个可计算预测器能够达到它。

    arXiv:2610.33842v1 Announce Type: cross  Abstract: Bayesian mixtures, and Solomonoff induction in particular, provide an idealized model of prediction learned by minimizing population log loss, and they underlie universal agents such as AIXI. An agent that plans with an approximation of such a predictor decides where that predictor is evaluated, and a planner can be drawn toward branches on which its model is optimistic. We study which approximation guarantees survive this feedback. For normalized finite-horizon causal mixtures, the transfer coefficient factors into a policy likelihood ratio and task evidence available before each prediction, so task rarity and action-branch rarity can compound. Within one fixed universal mixture, infinitely many computable environments admit excess prediction error of order $\kappa\eta\,2^{K(\mu)}$, where $\eta$ is the training KL budget and $\kappa$ measures policy coverage. This order is sharp, and for each target a computable predictor attains it a
    
[^43]: 参与受限的Hessian相似性优化：下界与虚拟主节点

    Hessian-Similar Optimization with Capped Participation: Lower Bounds and Virtual Masters

    [https://arxiv.org/abs/2610.33835](https://arxiv.org/abs/2610.33835)

    本文在每轮严格限制参与客户端数量的设定下，为Hessian相似优化建立了轮次复杂度下界，并提出虚拟主节点SVRS算法以消除锚点依赖，实现近最优的收敛轮次复杂度。

    

    当每轮最多只能联系m个客户端中的s个时，Hessian相似性优化的轮次复杂度是多少？已有的速率结果允许免费使用主节点或要求完全参与，并未刻画严格的参与上限。本文对每一次单向量查询与回复进行计数。通过隐藏客户端链的构造，本文为采用自适应队列与随机停止的“零尊重”局部算法给出了下界：在强凸二次函数上，当s ≤ (m-2)/2且处于高精度、高差异度情形时，所需轮次为 Ω((m/s)√(min{δ_a, δ̄/√m}/μ) log(1/ε))，其中μ为强凸性参数，δ_a与δ̄分别刻画锚点Hessian变化与平均Hessian变化的界。为消除对较差锚点的依赖，虚拟主节点SVRS算法采用相互独立的估计器队列与近端队列，并对近端回复取平均。结合Catalyst加速技术，该算法可在期望 Õ(m/s + (√(m/s) + m^{3/4}/s)√(δ̄/μ)) 的轮次内达到相对误差ε。

    arXiv:2610.33835v1 Announce Type: cross  Abstract: What is the round complexity of Hessian-similar optimization when each round can contact at most $s$ of $m$ clients? Prior rates permit a free master or full participation and do not characterize strict caps. We count every one-vector query and reply. A hidden-client chain gives a lower bound for zero-respecting local algorithms with adaptive cohorts and random stopping: on strongly convex quadratics, the required rounds are $\Omega((m/s)\sqrt{\min\{\delta_a,\bar\delta/\sqrt m\}/\mu}\,\log(1/\epsilon))$ for $s\leq(m-2)/2$ in the high-accuracy, high-dissimilarity regime. Here $\mu$ is strong convexity, while $\delta_a$ and $\bar\delta$ bound anchor and average Hessian variation. To remove dependence on a poor anchor, virtual-master SVRS uses independent estimator and proximal cohorts and averages the proximal replies. With Catalyst, it reaches relative gap $\epsilon$ in $\tilde O(m/s+(\sqrt{m/s}+m^{3/4}/s)\sqrt{\bar\delta/\mu})$ expecte
    
[^44]: 几何结构如何塑造黎曼流形上的生成模型评估

    How Geometry Shapes Generative Model Evaluation on Riemannian Manifolds

    [https://arxiv.org/abs/2610.33731](https://arxiv.org/abs/2610.33731)

    该论文揭示黎曼流形上生成模型评估指标的可靠性取决于几何结构（曲率与维度），没有任何一种度量普遍适用，并据此提出了报告评估距离和根据样本几何校准度量的实用建议。

    

    在黎曼流形上，生成模型的评估依赖于几何结构，因为曲率和维度共同决定了距离与体积增长。我们的框架系统地比较了Wasserstein和MMD在不同几何下的测地线版本、环境空间版本和切空间版本。通过受控噪声扰动和黎曼流匹配样本，我们研究了度量之间的一致性、噪声敏感性，以及它们与几何诱导的体积缩放之间的关系。我们发现没有任何一种度量是普遍可靠的：评估距离会影响所测得的差异和噪声敏感性，且这种影响随曲率和维度而变化。我们的分析为报告评估距离、比较度量类型以及根据样本几何结构校准度量提供了实用建议。

    arXiv:2610.33731v1 Announce Type: cross  Abstract: On Riemannian manifolds, generative-model evaluation is geometry-dependent since curvature and dimension jointly determine distances and volume growth. Our framework systematically compares geodesic, ambient, and tangent-space versions of Wasserstein and MMD across different geometries. Using controlled-noise perturbations and Riemannian Flow Matching samples, we study metric agreement, noise sensitivity, and their relation to geometry-induced volume scaling. We find that no metric is uniformly reliable: evaluation distance can affect both measured discrepancies and noise sensitivity, varying across curvature and dimension. Our analysis leads to practical recommendations for reporting evaluation distances, comparing metric types, and calibrating metrics to the geometry of the samples.
    
[^45]: 基于投影的紧致子流形联邦学习与几何无关的差分隐私

    Projection-Based Federated Learning on Compact Submanifolds with Geometry-Independent Differential Privacy

    [https://arxiv.org/abs/2610.33712](https://arxiv.org/abs/2610.33712)

    提出了 DP-RFedProj，一种面向紧致子流形的记录级差分隐私联邦学习方法，通过闭式投影进行聚合并在固定环境几何中隐私化梯度，实现了与流形和聚合方案无关的 f-DP 隐私保证。

    

    联邦学习可以在不交换客户端数据的情况下跨客户端训练共享模型，但客户端所传输的更新仍然会泄露关于个体记录的信息。当模型参数位于黎曼流形上时，实施差分隐私则更加困难。现有方法需要在每次新的迭代中根据该迭代所处的几何结构重新构建用于隐私保护的噪声，通过通常缺乏闭式表达的内蕴映射进行聚合，并且仅在受限的情形下才能证明收敛性。我们提出了 DP-RFedProj，这是一种针对紧致子流形的记录级差分隐私联邦学习方法，它通过闭式投影进行聚合，并在固定的环境几何中对梯度进行隐私化处理。每个客户端可以运行任意随机一阶优化器、执行多次本地更新步，并且仅间歇性地参与训练。该方法为整个训练过程提供了显式的记录级 f-DP 保证，且该保证与流形无关、与聚合方案无关……

    arXiv:2610.33712v1 Announce Type: new  Abstract: Federated learning trains a shared model across clients without exchanging their data, yet the updates they transmit still leak information about individual records. Enforcing differential privacy is harder still when the model parameters lie on a Riemannian manifold. Existing methods rebuild the noise added for privacy in the geometry of each new iterate, aggregate through intrinsic maps that often lack a closed form, and prove convergence only in restricted regimes. We introduce DP-RFedProj, a record-level differentially private federated learning method for compact submanifolds that aggregates through closed-form projections and privatises gradients in a fixed ambient geometry. Each client may run any stochastic first-order optimiser, take several local steps, and participate only intermittently. An explicit record-level $f$-DP guarantee holds for the entire training transcript, independently of the manifold, the aggregation scheme an
    
[^46]: 基于MMD目标的条件生成中三角传输映射的误差分析

    On the Error Analysis of Triangular Transport Maps for Conditional Generation via MMD-based Objectives

    [https://arxiv.org/abs/2610.33610](https://arxiv.org/abs/2610.33610)

    本文对基于MMD目标的三角传输映射条件生成进行了误差分析，证明最小化联合MMD通常无法一致地恢复条件分布，并精确刻画了联合误差与条件误差之间的差距。

    

    由三角传输映射参数化的条件生成模型被广泛用于条件采样，但对其所产生的条件测度质量的理论保证仍然十分匮乏。我们针对三角映射的最小散度估计器来研究这一问题。我们采用基于MMD的目标函数作为一个易于处理的框架，从而能够精确分析联合误差与条件误差之间的差距，并通过平均条件估计误差来度量性能，该误差定义为估计条件分布与真实条件分布之间的最大均值差异（MMD）的平方，并在条件变量上取平均。我们首先证明，最小化联合MMD这一常见做法可能无法恢复条件分布：通过将联合准则与条件准则表示为同一条件均值嵌入残差的二次能量形式，我们证明，对于参数类而言，联合最小化解通常是不一致的。

    arXiv:2610.33610v1 Announce Type: new  Abstract: Conditional generative models parameterized by triangular transport maps are widely used for conditional sampling, yet theoretical guarantees on the quality of the conditional measures they produce remain scarce. We address this question for minimum divergence estimators of triangular maps. We use MMD-based objectives as a tractable setting in which the gap between joint and conditional errors can be analyzed precisely, and we measure performance through the average conditional estimation error, defined as the squared maximum mean discrepancy (MMD) between the estimated and the true conditionals, averaged over the conditioning variable. We first show that the common practice of minimizing a joint MMD may fail to recover the conditionals: writing the joint and the conditional criteria as quadratic energies of the same conditional mean embedding residual, we prove that, for parametric classes, joint minimizers are generically inconsistent 
    
[^47]: 基于弱学习器的仿射不变三次牛顿法

    Affine-invariant Cubic Newton with Weak Learners

    [https://arxiv.org/abs/2610.33516](https://arxiv.org/abs/2610.33516)

    本文将仿射不变三次牛顿法（AICN）推广到带弱学习器的凸优化，通过平均余弦值和Hessian诱导范数下的方向改进量进行分析，克服了GRN提升算法对弱梯度边界参数的 γ¹² 重度依赖，建立了更优的全局收敛速率。

    

    现代梯度提升算法基于牛顿法，该方法在实践中表现出色，但缺乏全局收敛性保证。近期关于梯度正则化牛顿（GRN）提升的研究实现了全局 𝒪(1/γ¹²k²) 的收敛速率，但其分析未能利用自然的Hessian几何结构，且严重依赖于对最坏情况“弱梯度边界”γ 的 γ¹² 依赖。在本文中，我们通过将仿射不变三次牛顿法（AICN）方案推广到带弱学习器的凸优化问题，来解决这些理论缺陷。我们的分析基于优化路径上的平均余弦值 Θ̄ₖ 以及Hessian诱导范数下的方向改进量 δₖ。我们证明弱AICN更新可归结为一个阻尼弱牛顿步，其显式步长由 δₖ 确定。在半强自和谐条件下，我们建立了全局 𝒪(1/Θ̄ₖ（摘要在此处被截断）

    arXiv:2610.33516v1 Announce Type: new  Abstract: Modern gradient boosting algorithms are based on Newton's method, which excels empirically but lacks global convergence guarantees. While recent work on Gradient-Regularized Newton (GRN) boosting achieved a global $\mathcal{O}(1/\gamma^{12}k^2)$ rate, its analysis was not able to exploit the natural Hessian geometry and suffers from a heavy $\gamma^{12}$ dependence on a worst-case `weak gradient edge' $\gamma$. In this paper, we tackle these theoretical drawbacks by generalizing the Affine-Invariant Cubic Newton (AICN) scheme to convex optimization with weak learners. Our analysis is based on average cosine values $\overline{\Theta}_k$ along the optimization path and directional improvements $\delta_k$ in the Hessian-induced norm. We show that the weak-AICN update reduces to a damped weak-Newton step with an explicit stepsize determined by $\delta_k$. Under semi-strong self-concordance, we establish a global $\mathcal{O}(1/\overline{\The
    
[^48]: 基于距离轮廓的度量空间因果推断

    Causal Inference in Metric Spaces via Distance Profiles

    [https://arxiv.org/abs/2610.33491](https://arxiv.org/abs/2610.33491)

    该论文提出了仅依赖距离度量即可定义的“轮廓因果效应”Δ(ω,t)，用以刻画处理在空间位置和尺度上如何移动概率质量（包括均值不变时的离散度、尾部与结构变化），并结合交叉拟合双重稳健得分与高维高斯近似实现统计推断。

    

    现代干预措施日益产生对象值结果，例如分布、协方差矩阵、网络和树。现有的因果框架往往利用特殊几何结构，如一维 Wasserstein 分位数或测地 Fréchet 均值，而度量统计则在无混杂调整的情况下研究随机对象。我们提出了轮廓因果效应 Δ(ω,t)，它刻画了处理在何处以及以何种尺度移动概率质量，包括那些不改变均值层面汇总信息的离散度、尾部或结构上的变化。形式上，Δ(ω,t)=Pr{d(ω,Y(1))≤t}−Pr{d(ω,Y(0))≤t} 对比了以同一锚点为中心的度量球下潜在结果的概率，且仅需距离度量而无需其他结构假设。在由确定性或辅助设计固定的网格上，我们将交叉拟合的双重稳健得分与高维高斯近似以及多重……（原文摘要截断）

    arXiv:2610.33491v1 Announce Type: cross  Abstract: Modern interventions increasingly produce object-valued outcomes, such as distributions, covariance matrices, networks, and trees. Existing causal frameworks often exploit special geometry such as one-dimensional Wasserstein quantiles or geodesic Fr\'echet means, whereas metric statistics studies random objects without confounding adjustment. We propose the profile causal effect $\Delta(\omega,t)$, which describes where and at what scale a treatment shifts probability mass, including changes in dispersion, tails, or structure that leave mean-level summaries unchanged. Formally, $\Delta(\omega,t)=\Pr\{d(\omega,Y(1))\le t\}-\Pr\{d(\omega,Y(0))\le t\}$ contrasts the potential-outcome probabilities of the same anchor-centered metric ball and requires no structure beyond the metric. On grids fixed by a deterministic or auxiliary design, we combine cross-fitted doubly robust scores with high-dimensional Gaussian approximation and the multipl
    
[^49]: 黎曼流形上平均化随机梯度下降的在线协方差估计

    Online Covariance Estimation for Averaged Stochastic Gradient Descent on Riemannian Manifolds

    [https://arxiv.org/abs/2610.33419](https://arxiv.org/abs/2610.33419)

    本文通过向量传输技术构造了黎曼流形上平均化RSGD的在线协方差估计器，能够自适应处理迭代中变化的切空间，无需计算黎曼Hessian矩阵或预设终止样本量，且收敛速度与欧几里得情形一致。

    

    我们研究了当未知参数位于流形上时，平均化黎曼随机梯度下降（RSGD）的在线推断问题。尽管平均化RSGD迭代具有渐近正态性，但实际推断需要估计其极限分布中未知的协方差算子。现有的在线批均值方法无法直接应用，因为一般的黎曼流形缺乏加法运算，且包含黎曼梯度的切空间在迭代过程中会发生变化。通过向量传输，我们构造了重叠与非重叠协方差估计器，它们能够自适应地融合连续批次的信息，同时处理不断变化的切空间。所提出的估计器是在线的，既不需要计算黎曼Hessian矩阵，也不需要预先指定终止样本量，并且达到了与其欧几里得对应方法相同的收敛速度（Zhu et al., 2023）。结合渐近……

    arXiv:2610.33419v1 Announce Type: new  Abstract: We study online inference for averaged Riemannian stochastic gradient descent (RSGD) when the unknown parameter lies on a manifold. Although the averaged RSGD iterate is asymptotically normal, practical inference requires estimating the unknown covariance operator in its limiting distribution. Existing online batch-means methods cannot be applied directly because a general Riemannian manifold lacks an addition operation, and the tangent space containing the Riemannian gradient varies across iterations. Through vector transport, we construct overlapping and non-overlapping covariance estimators that adaptively combine information from successive batches while accounting for the changing tangent spaces. The proposed estimators are online, require neither Riemannian Hessian computations nor a prespecified terminal sample size, and attain the same convergence rates as their Euclidean counterparts (Zhu et al., 2023). Together with asymptotic 
    
[^50]: 高保真联邦表格数据合成

    Faithful Federated Tabular Data Synthesis

    [https://arxiv.org/abs/2610.33403](https://arxiv.org/abs/2610.33403)

    提出FITS框架，通过基于局部线性回归的两步联邦条件累积分布函数估计结合概率链式法则来保证分布保真度，并采用无损连续化方法统一处理混合类型表格数据。

    

    合成数据是人工智能时代的重要数据来源。本文研究了联邦环境下混合类型表格合成数据的生成问题，其中面临两大关键挑战：确保分布保真度以及有效处理混合类型数据。我们提出了FITS（联邦整合两步合成），一个用于混合类型表格数据合成的框架。其核心是一种两步联邦算法，利用局部线性回归来估计条件累积分布函数（CDF）。基于该算法，我们开发了一种基于概率链式法则的联邦合成程序，以确保分布保真度。为了处理混合类型数据，FITS引入了一种连续化方法，将离散变量提升到连续空间中，并在密度函数和累积分布函数两个层面上无损地恢复其离散分布。这一转换为混合类型数据提供了统一的连续表示……

    arXiv:2610.33403v1 Announce Type: cross  Abstract: Synthetic data is an important data source in the AI era. This paper studies the generation of mixed-type tabular synthetic data in federated settings, where two key challenges arise: ensuring distributional fidelity and effectively handling mixed-type data. We propose Federated Integrative Two-Step Synthesis (FITS), a framework for mixed-type tabular data synthesis. At its core is a two-step federated algorithm for estimating conditional cumulative distribution functions (CDFs) using local linear regression. Building on this algorithm, we develop a federated synthesis procedure based on the probability chain rule to ensure distributional fidelity. To handle mixed-type data, FITS introduces a continualization method that lifts discrete variables into a continuous space and recovers their discrete distributions without loss at both the density-function and CDF levels. This transformation provides a unified continuous representation of m
    
[^51]: Nesterov 轨迹何时具有最终局部最小化性质？

    When Are Nesterov Trajectories Eventually Locally Minimizing?

    [https://arxiv.org/abs/2610.33341](https://arxiv.org/abs/2610.33341)

    本论文将 Nesterov 轨迹的最终局部最小化问题等价转化为相关薛定谔算子的非负性问题，并证明对于幂函数目标，在所有指数 p≥2 中该性质当且仅当 p=4 成立，且这一特殊指数可由半线性椭圆方程的稳定性理论解释。

    

    我们重新审视了 Nesterov 著名方法的变分刻画，并研究了 Nesterov 轨迹在变分意义下何时具有最终局部最小化性质——这是一个我们精确定义的概念。我们证明，该问题等价于一个相关的薛定谔算子作用于轨迹变分时的非负性。对于形如 f(x)=（1/p)‖x‖^p 的幂函数目标（其中 ‖·‖ 表示欧几里得范数），我们给出了完整的刻画：在所有指数 p≥2 中，具有非零初始位置的 Nesterov 轨迹具有最终局部最小化性质当且仅当 p=4。我们进一步通过半线性椭圆方程的稳定性理论来解释这一特殊指数。

    arXiv:2610.33341v1 Announce Type: cross  Abstract: We revisit the variational characterization of Nesterov's celebrated method and study when Nesterov trajectories are eventually locally minimizing in a variational sense, a notion that we define precisely. We show that this question is equivalent to the nonnegativity of an associated Schr\"odinger operator acting on variations of the trajectory. For power objectives of the form \[ f(x)=\frac{1}{p}\|x\|^p \] (where $\| \cdot\|$ denotes the Euclidean norm), we give a complete characterization: Among all exponents $p\ge 2$, the Nesterov trajectory with non-zero initial position is eventually locally minimizing if and only if $p=4$. We further interpret this exceptional exponent through the stability theory of semilinear elliptic equations.
    
[^52]: 带噪声测量的盲目随机搜索：几乎必然收敛与收敛速率

    Blind Random Search with Noisy Measurements: Almost Sure Convergence and Convergence Rates

    [https://arxiv.org/abs/2610.33302](https://arxiv.org/abs/2610.33302)

    本文将盲目随机搜索扩展至带噪声测量的场景，通过结合递减接受阈值与噪声损失测量值的滑动平均，证明了算法几乎必然收敛到全局最优解，并给出了低维情形下的渐近上界以及高维情形下的Weibull极限分布。

    

    盲目随机搜索是一种简单的全局优化方法，它只需要损失函数的取值，而无需梯度信息或显式的损失模型。然而，其经典形式假设损失评估是无噪声的，这限制了它直接应用于存在噪声测量的实际问题。我们通过将递减的接受阈值与噪声损失测量值的滑动平均相结合，对盲目随机搜索进行了扩展。在所述假设条件下，我们证明了所提出的算法几乎必然收敛到全局最小值点。我们进一步分析了当前估计处的损失逼近最小损失的速度。对于低维情形，我们得到了两者之差的几乎必然渐近上界；对于高维情形，我们证明了该差值经过适当缩放后具有精确的Weibull极限分布。

    arXiv:2610.33302v1 Announce Type: cross  Abstract: Blind random search is a simple global optimization method that only requires the value of the loss function, without needing gradient information or an explicit loss model. However, its classical formulation assumes that the loss evaluation is noise-free, which limits its direct application to practical problems with noisy measurements. We extend blind random search by combining a vanishing acceptance threshold with the running average of noisy loss measurements. Under the stated assumptions, we prove that the proposed algorithm converges almost surely to the global minimizer. We further analyze the rate at which the loss at the current estimate approaches the minimum loss. For low-dimensional cases, we obtain an almost sure asymptotic upper bound on the difference between the two, and for high-dimensional cases, we prove that this difference, suitably scaled, has an exact Weibull limiting distribution.
    
[^53]: 随机概率分布的核均值嵌入及其主成分

    The kernel mean embedding of a random probability distribution and its principal components

    [https://arxiv.org/abs/2610.33286](https://arxiv.org/abs/2610.33286)

    本文将随机概率分布的核均值嵌入作为再生核希尔伯特空间中的随机元素建立了严格的理论框架，并推导出核主成分分析的一般收敛性结果以及正态分布情形下协方差算子谱分解的显式表达式。

    

    给定一个随机概率分布 \(F\)，其实现定义在 \(\mathbb R^d\) 上，以及一个核函数 \(k\)，我们的目标是分析 \(F\) 的核均值嵌入 \(\mu_F\) 作为相应再生核希尔伯特空间 \(\mathcal H\) 中随机元素的各项性质。这建立了一个严格的框架，使得针对分布数据的核统计技术可以在样本和总体两种设定下加以表述，从而能够正确地建立样本方法的渐近性质。作为该方法的一个示例，我们重点研究 \(F\) 的主成分。我们获得了一般性的收敛结果，以及期望服从高斯分布的正态概率分布的非中心化协方差算子谱分解的显式表达式。该核主成分方法的性能通过仿真实验和数据分析得到了验证。

    arXiv:2610.33286v1 Announce Type: cross  Abstract: Given a random probability distribution \(F\), whose realizations are defined on \(\mathbb R^d\), and a kernel \(k\), our aim is to analyze the properties of the kernel mean embedding, \(\mu_F\), of \(F\) as a random element of the corresponding reproducing kernel Hilbert space \(\mathcal H\). This establishes a rigorous framework in which kernel statistical techniques for distributional data can be stated both in the sample and population settings and, thus, the asymptotic properties of sample procedures can be properly established. As an illustration of this approach we focus on the principal components of \(F\). We obtain a general convergence result and the explicit expression of the spectral decomposition for the uncentered covariance operator of a normal probability distribution with a Gaussian distributed expectation. The performance of this kernel principal components procedure is illustrated with simulations and the analysis o
    
[^54]: 并非所有错误标签都同等重要：基于置信序列的高效任务感知审计

    Not All Mislabels Are Equal: Efficient Task-Aware Auditing with Confidence Sequences

    [https://arxiv.org/abs/2610.33262](https://arxiv.org/abs/2610.33262)

    该论文提出ATLAS框架，将任意时刻有效的置信序列推断与任务感知的自适应审计策略结合，根据标签不确定性及其对下游评分的影响优先审计数据点，从而以比随机抽样少11%-24%的审计量将标签错误率认证到±2个百分点的精度。

    

    arXiv:2610.33262v1 公告类型：交叉。摘要：我们研究在固定数据集中对类别标签进行高效、自适应的审计，其动机来自生物多样性信用项目：在这类项目中，专家验证物种记录的成本高昂，且下游评分（如付款金额）对某些标签上错误的敏感度高于其他标签。我们提出的ATLAS框架将任意时刻有效推断与任务感知的自适应审计策略相结合。通过混淆矩阵来表示标签修正，我们根据标签不确定性以及潜在标签修正对下游评分函数的影响，来决定接下来优先检查哪些数据点。对于误标率等线性评分，我们的优先级感知方法在肯尼亚生物多样性数据和广泛使用的CIFAR-10N基准数据集上，分别以比随机抽样少24%和11%的审计量，将整体标签错误率认证到±2个百分点的精度。对于希尔-香农多样性等非线性评分，我们探讨了（原文摘要至此截断）

    arXiv:2610.33262v1 Announce Type: cross  Abstract: We study efficient, adaptive auditing of categorical labels in a fixed dataset, motivated by biodiversity-credit programs in which expert verification of species records is costly and downstream scores (e.g., payment) are more sensitive to errors on certain labels than on others. Our proposed ATLAS framework couples anytime-valid inference with a task-aware adaptive auditing policy. Representing label corrections through a confusion matrix, we prioritize which data points to inspect next based on both label uncertainty and the influence of potential label corrections on the downstream score function. For linear scores such as the mislabel rate, our priority-aware approach certifies the overall label error rate to $\pm 2$ percentage points with 24% and 11% fewer audits than random sampling on biodiversity data from Kenya and a widely used CIFAR-10N benchmark, respectively. For nonlinear scores such as the Hill-Shannon diversity, we expl
    
[^55]: 核化协方差矩阵、降维与不平衡谱聚类

    Kernelised covariance matrices, dimension reduction and imbalanced spectral clustering

    [https://arxiv.org/abs/2610.33177](https://arxiv.org/abs/2610.33177)

    提出基于截断核的核化协方差矩阵作为PCA降维的替代方案，在高斯混合模型的不平衡聚类场景下实现了更有效的降维与谱聚类，并给出了定量的理论保证。

    

    高斯混合模型（GMM）是聚类数据的一种经典生成模型，基于PCA降维的谱聚类一直是处理该模型的有效框架。在本工作中，我们研究了核化协方差矩阵，将其作为降维与谱聚类的一种替代机制，重点考察通过合适的分离约束来选择数据点对的截断核。这些矩阵的结构设计受到排斥粒子系统统计力学思想的启发。我们的方法尤其适用于不平衡聚类的场景，即最大聚类与最小聚类的大小之比随问题规模而发散，此时现有的基于PCA的方法效果有限。我们通过定量界限证明了这些改进的协方差矩阵具有高效的降维与聚类性能。

    arXiv:2610.33177v1 Announce Type: new  Abstract: The Gaussian Mixture Model (abbrv. GMM) is a classic generative model for clustered data, for which spectral clustering based on dimension reduction via PCA has been an effective framework. In this work, we investigate kernelised covariance matrices as an alternative mechanism for dimension reduction and spectral clustering, focusing on truncation kernels that select pairs of data points with suitable separation constraints. The structure of these matrices is motivated by considerations from statistical mechanics of repulsive particle systems. Our approach is particularly relevant for the setting of imbalanced clusters, where the sizes of the largest and smallest clusters have a diverging ratio with the problem size, and existing PCA-based methods turn out to be of limited effectiveness.   We demonstrate efficient dimension reduction and clustering performance of these modified covariance matrices, substantiated by quantitative bounds. I
    
[^56]: 机器学习中鲁棒性的测度几何理论

    A Measure-Geometric Theory of Robustness in Machine Learning

    [https://arxiv.org/abs/2610.33123](https://arxiv.org/abs/2610.33123)

    本文建立了机器学习鲁棒性的基础性测度几何理论，将鲁棒性重塑为逃逸的几何形成、组织、实现与持续过程，为超越标量准则的结构化鲁棒性分析奠定了基础。

    

    本文发展了一个机器学习鲁棒性的基础性测度几何理论。鲁棒性被表述为相对于承载数据的流形上的参考测度、一个扰动度规以及一个容许实现机制而言，从保持的任务状态中的逃逸。该理论区分了逃逸在参考流形上的结构性可能性与它在机制相关的可观测读数下的环境实现，同时将二者纳入统一的几何解释框架之中。它引入了逃逸几何的介观组织结构，通过这一结构，脆弱质量、有限尺度形态、传递的进展以及分支一致性以显式定律的方式被刻画为相互关联但不完全相同的鲁棒性现象。通过将鲁棒性重塑为逃逸的几何形成、组织、实现与持续，该理论为超越标量准则的分析提供了结构性基础。

    arXiv:2610.33123v1 Announce Type: new  Abstract: This paper develops a foundational measure-geometric theory of robustness in machine learning. Robustness is formulated as escape from a retained task state relative to a reference measure on a data-bearing manifold, a perturbation gauge, and an admissible realization mechanism. The theory distinguishes the structural possibility of escape on the reference manifold from its ambient realization under mechanism-dependent observable readouts, while placing both within a unified geometric account. It introduces a mesoscopic organization of escape geometry, through which vulnerable mass, finite-scale morphology, delivered progress, and branch coherence are characterized by explicit laws as linked but non-identical robustness phenomena. By recasting robustness as the geometric formation, organization, realization, and persistence of escape, the theory provides a structural basis for analysis beyond scalar criteria.
    
[^57]: TGAge：基于表格基础模型与冈珀茨定律的辅助生活机构居民表型衰老时钟

    TGAge: A Phenotypic Aging Clock Based on Tabular Foundation Models and the Gompertz Law for Assisted Living Facility Residents

    [https://arxiv.org/abs/2610.32994](https://arxiv.org/abs/2610.32994)

    TGAge是一种基于表格基础模型和冈珀茨定律的第二代表型衰老时钟，能够从常规临床测量数据中估计辅助生活机构居民的风险等效生物学年龄，在生存预测上优于现有方法，且其年龄加速指标与多种衰老相关合并症显著相关。

    

    生物学年龄能够捕捉超越实际年龄的整体健康状况，然而由于表观遗传数据稀缺，现有的生物学衰老时钟难以应用于辅助生活机构（ALF）居民。我们提出了TGAge，这是一种第二代表型衰老时钟，可以从常规收集的临床测量数据中估计风险等效的生物学年龄。TGAge采用表格基础模型来预测离散的距死亡时间区间内的全因死亡风险，并通过冈珀茨定律将所得的死亡风险比转换为生物学年龄加速。我们在从一个专有的多州长期护理数据库中整理的大型辅助生活机构居民队列上开发并评估了TGAge。TGAge在生存预测方面优于临床指标和已建立的生物学衰老时钟。此外，TGAge得出的年龄加速与多种衰老相关合并症显著相关。

    arXiv:2610.32994v1 Announce Type: cross  Abstract: Biological age captures overall health status beyond chronological age, yet existing biological aging clocks are difficult to apply to assisted living facility (ALF) residents due to scarce epigenetic data. We propose TGAge, a second-generation phenotypic aging clock that estimates risk-equivalent biological age from routinely collected clinical measurements. TGAge employs a tabular foundation model to predict all-cause mortality risks over discrete time-to-death intervals and converts the resulting mortality-risk ratio into biological age acceleration via the Gompertz law. We develop and evaluate TGAge on a large cohort of ALF residents curated from a proprietary, multi-state long-term care database. TGAge outperforms clinical indices and established biological aging clocks in survival prediction. Moreover, TGAge-derived age acceleration is significantly associated with multiple aging-related comorbidities. These findings support TGAg
    
[^58]: GoCo：面向错误可控的基因本体（GO）注释发布的结构引导共形校准框架

    GoCo: structure-guided conformal calibration for error-controlled Gene Ontology annotation release

    [https://arxiv.org/abs/2610.32990](https://arxiv.org/abs/2610.32990)

    GoCo提出了一种结构引导的共形校准框架，通过为GO层次感知的基因水平错误发现比例（FDP）认证全局阈值并构建嵌套发布路径，在错误可控的前提下释放更多经参考支持的基因本体注释，且无需对独立数据集重新调参。

    

    计算基因功能预测能够以实验表征无法企及的规模生成注释，然而即便在最先进的方法中仍存在大量错误。针对基因本体（GO）预测的显式错误受控校准在很大程度上仍属空白。我们提出GoCo，一种结构引导的框架，用于校准如何将现有预测器的分数转换为最终发布的注释。GoCo为感知GO层次结构的基因水平错误发现比例（FDP）认证一个全局阈值，并在相同的渐近风险控制保证下构建嵌套的发布路径。在主要操作点（α=0.10，δ=0.50）上，基于Wainberg共必需性模块开发的GoCo-M在无需重新调参的情况下被应用于三项具有相同共享证据结构的独立研究，其在保留集上的平均基因水平FDP始终保持在目标水平或以下，同时比基线方法MultiL多发布6.3%–17.6%的经参考支持的GO条目。

    arXiv:2610.32990v1 Announce Type: cross  Abstract: Computational gene-function prediction generates annotations at a scale that experimental characterization cannot match, yet substantial errors remain even among leading methods. Explicit error-controlled calibration for Gene Ontology (GO) prediction remains largely undeveloped. We introduce GoCo, a structure-guided framework that calibrates how scores from an existing predictor are converted into released annotations. GoCo certifies a global threshold for a GO-hierarchy-aware gene-level false discovery proportion (FDP), then constructs a nested release path under the same asymptotic risk-control guarantee. At the primary operating point (alpha=0.10, delta=0.50), GoCo-M, developed on Wainberg co-essentiality modules and applied without retuning to three independent studies with the same shared-evidence structure, kept mean held-out gene-level FDP at or below target while releasing 6.3-17.6% more reference-supported GO terms than Multil
    
[^59]: 基于条件期望局部半径的度量保形预测

    Metric Conformal Prediction via the Conditional Expected Local Radius

    [https://arxiv.org/abs/2610.31336](https://arxiv.org/abs/2610.31336)

    该论文提出条件期望局部半径（CELR）这一新度量，构建了适用于一般度量空间中非欧几里得响应的保形预测方法，结合度量分布随机森林实现高效计算，并从理论上证明了估计量一致性、渐近条件有效性及有限样本边际覆盖率。

    

    我们提出了条件期望局部半径（CELR），并开发了一种适用于一般度量空间中非欧几里得响应变量的新型保形预测方法。CELR 度量局部条件概率的集中程度，使预测集能够适应异质性和多峰性，而无需依赖密度估计或向量空间结构。其条件分布产生了一个在不同协变量取值之间具有可比性的非一致性得分，并避免了距离轮廓之间代价高昂的传输比较。我们进一步改造了度量分布随机森林，使其能够处理多变量协变量和非线性交互作用，同时单次森林拟合即可提供所有得分组成部分，通过消除重复的局部拟合来降低计算成本。我们在适当的正则性条件下建立了估计量的一致性和渐近条件有效性，并在可交换性假设下获得了有限样本边际覆盖率。

    arXiv:2610.31336v1 Announce Type: cross  Abstract: We introduce the conditional expected local radius (CELR) and develop a new conformal prediction method for non-Euclidean responses in general metric spaces. CELR measures local conditional probability concentration, allowing prediction sets to accommodate heterogeneity and multimodality without requiring densities or a vector space structure. Its conditional distribution yields a nonconformity score that is comparable across covariate values and avoids costly transport comparisons between distance profiles. We further adapt a metric distributional random forest that handles multivariate covariates and nonlinear interactions, while a single forest fit supplies all score components, reducing computation by eliminating repeated local fitting. We establish uniform consistency of the estimators and asymptotic conditional validity under suitable regularity conditions, together with finite-sample marginal coverage under exchangeability. Nume
    
[^60]: 训练过程中宽度与深度极限的可交换性

    Width and Depth Limits Commute During Training

    [https://arxiv.org/abs/2610.30729](https://arxiv.org/abs/2610.30729)

    本文将残差网络无限宽与无限深极限的可交换性从初始化阶段首次扩展到训练过程，证明了在Depth-μP参数化与梯度裁剪下，线性残差网络在任意有限训练步数内，网络输出沿宽度-深度联合增长路径按 $L^2$ 收敛且两极限可交换。

    

    近期的研究工作证明了，当残差分支以 $1/\sqrt{深度}$ 进行缩放时，残差网络在初始化时的无限宽与无限深极限是可交换的。特别地，无论宽度与深度是依次趋于无穷还是同时趋于无穷，网络输出都收敛到相同的极限。我们将这一可交换性结果扩展到线性残差网络的训练过程中。在相同的 $1/\sqrt{深度}$ 残差分支缩放（Depth-$\mu$P）和 $\mu$P 宽度参数化下，我们证明：在使用梯度裁剪的条件下，对于任意固定数量的梯度下降步数，当宽度和深度同时趋于无穷时，网络输出沿着每条宽度-深度联合增长路径按 $L^2$ 收敛。因此，无限宽与无限深极限在任何有限训练范围内都是可交换的。在不使用梯度裁剪的情况下，相应的 $L^2$ 收敛一般不成立，但我们建立了较弱的依概率收敛。

    arXiv:2610.30729v1 Announce Type: new  Abstract: Recent work established that the infinite-width and infinite-depth limits of residual networks commute at initialization when residual branches are scaled by $1/\sqrt{depth}$. In particular, the network output converges to the same limit whether width and depth tend to infinity sequentially or jointly. We extend this commutativity result to training in linear residual networks. Under the same $1/\sqrt{depth}$ residual-branch scaling (Depth-$\mu$P) and the $\mu$P width parameterization, we prove that, for any fixed number of gradient-descent steps with gradient clipping, the network output converges in $L^2$ along every joint width--depth growth path as both width and depth tend to infinity. Consequently, the infinite-width and infinite-depth limits commute throughout any finite training horizon. Without gradient clipping, the corresponding $L^2$ convergence does not hold in general, but we establish the weaker convergence in probability.
    
[^61]: 小损失下的差分隐私专家在线预测

    Private Online Prediction from Experts with Small Losses

    [https://arxiv.org/abs/2610.29575](https://arxiv.org/abs/2610.29575)

    该论文提出了首个遗憾度达到 $\tilde{O}(\sqrt{L^\star})$ 的差分隐私专家在线预测算法，在低损失情形下显著优于现有 $\tilde{O}(\sqrt{T})$ 界限，并证明在忘却输入下隐私代价很小。

    

    我们研究机器学习中的一个基本问题——差分隐私约束下的专家在线预测。现有的隐私保护算法实现了接近最优的遗憾度 $\tilde{O}(\sqrt{T})$，其中 $T$ 为时间范围。然而，当最优专家的累计损失 $L^\star$ 远小于 $T$ 时，该界限变得次优。在这项工作中，我们提出了首个遗憾度为 $\tilde{O}(\sqrt{L^\star})$ 的差分隐私算法，在低损失情形下带来了显著的改进。对于常数隐私参数，我们的遗憾度界限在多对数因子范围内与非隐私下界相匹配，表明在此设置下隐私只需付出很小的代价。我们的结果假设输入序列是忘却的，即预先固定且独立于算法的输出。我们还证明了对于自适应生成的输入，$\tilde{\Omega}(\sqrt{T})$ 的遗憾度是不可避免的。

    arXiv:2610.29575v1 Announce Type: new  Abstract: We consider online prediction from experts, a fundamental problem in machine learning, under differential privacy constraints. Existing private algorithms achieve near-optimal regret $\tilde{O}(\sqrt{T})$, where $T$ is the time horizon. However, this bound becomes suboptimal when $L^\star$, the cumulative loss of the best expert, is significantly less than $T$. In this work, we present the first differentially private algorithm with regret $\tilde{O}(\sqrt{L^\star})$, offering a substantial improvement in the low-loss regime. Our regret bound matches the non-private lower bound up to poly-logarithmic factors for constant privacy parameters, demonstrating that privacy incurs only a small cost in this setting. Our result assumes that the input sequence is oblivious, i.e., it is fixed in advance and independent of the algorithm's outputs. We also prove that an $\tilde{\Omega}(\sqrt{T})$ regret is inevitable for adaptively generated inputs, 
    
[^62]: 逆向戈德定理：基于基底受限对比学习器的正数据语言识别

    A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners

    [https://arxiv.org/abs/2610.28747](https://arxiv.org/abs/2610.28747)

    该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。

    

    戈德定理表明，任何学习器都无法仅凭正例呈现来识别超有限类中的任意语言。我们证明了一个形态不同的正面结果。固定一个先天的“递归对比算子”$R$，它将任意字符串映射到其结构上受限的有限扰动集合；再考虑一个容量受限的判别式评分器，它通过在自我生成的扰动上进行InfoNCE式对比估计训练。在算子、评分器与目标语言之间满足忠实性条件的前提下，训练后的评分器能够仅凭正数据实现“PAC条件识别”：以高概率，它会把除$\varepsilon$比例之外的所有正例都排在其自身轨道中每个语言外扰动之上，且样本复杂度关于模型容量、轨道大小、分数界限和$1/\varepsilon$呈多项式级。负例由学习器自行构造，而非由外部提供……

    arXiv:2610.28747v1 Announce Type: cross  Abstract: Gold's theorem shows that no learner can identify an arbitrary language in a superfinite class from positive presentations alone. We prove a positive result of a different shape. Fix an innate \emph{recursive contrast operator} $R$ mapping any string to a finite set of structurally bounded perturbations of it, and a discriminative scorer of bounded capacity trained by InfoNCE-style contrastive estimation \citep{vandenoord2018,macollins2018} on self-generated perturbations. Under a faithfulness condition relating operator, scorer, and target, the trained scorer achieves \emph{PAC conditional identification} from positive data alone: with high probability it ranks all but an $\varepsilon$-fraction of positives above every out-of-language perturbation in their own orbit, with sample complexity polynomial in the capacity, the orbit size, the score bound, and $1/\varepsilon$. The negatives are constructed by the learner, not supplied by an 
    
[^63]: 问而学：基于查询的协作强化学习

    Ask to Learn: Query-Based Collaborative Reinforcement Learning

    [https://arxiv.org/abs/2610.27606](https://arxiv.org/abs/2610.27606)

    该论文提出了基于查询的协作强化学习新框架，使仅有离线数据的学习者无需直接与环境交互，即可通过自适应地向持有互补信息的协作者查询状态-时间对来突破离线数据覆盖范围的限制，并针对有限视界表格型问题给出了所需查询次数的实例依赖信息论下界。

    

    离线强化学习从根本上受限于可用数据的覆盖范围，而通过在线探索收集额外轨迹可能代价高昂、不安全或不可行。我们研究能否在不与底层环境直接交互的情况下克服这一局限。受协作者中介模拟等实际场景的启发——其中协作者可以提供与离线数据互补的信息——我们提出了基于查询的协作强化学习，以研究如何高效地获取这些信息。学习者从一个固定的离线数据集出发，自适应地查询状态-时间对；协作者根据一个固定的、可能是次优的策略采样动作，模拟器返回相应的奖励和下一状态。对于有限视界的表格型问题，我们推导了关于额外查询次数的实例依赖的信息论下界……

    arXiv:2610.27606v1 Announce Type: cross  Abstract: Offline reinforcement learning is fundamentally limited by the coverage of the available data, while collecting additional trajectories through online exploration may be costly, unsafe, or infeasible. We study whether this limitation can be overcome without directly interacting with the underlying environment. Motivated by practical settings such as collaborator-mediated simulation, where a collaborator can provide information complementary to the offline data, we introduce query-based collaborative reinforcement learning to study how to acquire this information efficiently. The learner starts with a fixed offline dataset and adaptively queries state-time pairs; the collaborator samples an action according to a fixed, possibly suboptimal policy, and the simulator returns the corresponding reward and next state. For finite-horizon tabular problems, we derive an instance-dependent information-theoretic lower bound on the additional queri
    
[^64]: 希尔伯特尺度上后验扩散先验用于不适定信号恢复的稳定性

    Hilbert-Scale Stability of Posterior Diffusion Priors for Ill-Posed Signal Recovery

    [https://arxiv.org/abs/2610.27030](https://arxiv.org/abs/2610.27030)

    该论文在希尔伯特尺度框架下为线性逆问题的后验扩散采样建立了稳定性理论，在正向算子平滑条件、真值源条件与分数误差弱范数控制下，证明了学习到的扩散先验在不适定信号恢复中能够给出将噪声、分数近似与反向时间效应分离的非渐近恢复保证。

    

    当正向算子会把未知信号平滑掉时，间接且带噪声的观测难以被反演。在这种情况下，不稳定的模态恰恰是数据揭示信息最少的模态，因此任何重构方法的成功都取决于先验信息与算子几何结构在不同正则性尺度上的相互作用方式。扩散先验在这类问题中已被广泛使用；然而，现有理论仍无法确定：在不适定正向算子作用下，学习到的分数函数何时仍能保持有效性。本文针对希尔伯特尺度上的线性逆问题研究后验扩散，并在一个显式的分数放大条件下推导出稳定性理论。在正向映射满足平滑条件、真值满足源条件、以及分数误差具有弱范数控制的假设下，我们推导出了希尔伯特空间 \(H^s\) 中的非渐近恢复界，该界将观测噪声、分数近似误差与反向时间（原文摘要在此处截断）的影响分离开来。

    arXiv:2610.27030v1 Announce Type: new  Abstract: Indirect and noisy observations are difficult to invert when the forward operator smooths out the unknown signal. In this case, the unstable modes are precisely the modes that the data reveals least, thus the success of any reconstruction method depends on how prior information interacts with the geometry of the operator across regularity scales. Diffusion priors have become ubiquitous for such problems; however, the current theory still does not determine when a learned score remains informative under ill-posed forward operators. In this paper, we study posterior diffusion for linear inverse problems on Hilbert scales and derive a stability theory under an explicit score-amplification condition. Under a smoothing condition on the forward map, a source condition on the truth, and weak-norm control of score error, we derive non-asymptotic recovery bounds in \(\Hs^s\) that separate observation noise, score approximation, and reverse-time d
    
[^65]: 预先公告的变点使保形可靠性与频率校准相分离

    Announced Breaks Separate Conformal Reliability from Frequency Calibration

    [https://arxiv.org/abs/2610.23917](https://arxiv.org/abs/2610.23917)

    本文证明预先公告的结构性变点使“下一区间的保形可靠性”与“封闭窗口的频率校准”这两种重置审计目标可以分离，并提出在公告时点重启经验分位数与自适应递归的方法，前者达到1/√n的minimax可靠性速率、后者达到1/n的频率差距，且实验中覆盖率误差显著优于自适应基线（0.035对0.052）。

    

    许多结构性变点会提前公告，然而自适应保形方法只能从自身的预测误差中获知它们。在公告日期进行重置只能服务于两种审计之一：下一区间的保形可靠性，或一个封闭窗口的频率校准，而这两者可能不一致。对于独立得分，在变点边界处重启经验分位数可以达到极小化极大（minimax）可靠性速率，其阶为断裂后样本数的平方根分之一（1/√n）。在得分有界且递归下限累计增量有界的条件下，重启自适应递归可以达到阶为断裂后样本数分之一（1/n）的频率差距，但该差距对下一区间没有任何约束作用。在一次较大的合成变点之后，重启分位数在第50次预测时的覆盖率误差为0.035，而五个自适应基线中最好者为0.052。在每次合成变点之后的前50次预测中，重启递归都具有最小的频率差距。真实变点大多是……（原文在此截断）

    arXiv:2610.23917v2 Announce Type: replace-cross  Abstract: Many structural breaks are announced in advance, yet adaptive conformal methods learn of them only from their own errors. A reset at the announced date serves one of two audits, conformal reliability of the next interval or frequency calibration of a closed window, and the two can disagree. For independent scores, restarting the empirical quantile at the boundary attains the minimax reliability rate, of order one over the square root of the post-break count. Restarting an adaptive recursion attains a frequency gap of order one over that count on bounded scores when its floor adds a bounded total. That gap bounds nothing about the next interval. After a large synthetic break, the restarted quantile's coverage error at the 50th forecast is 0.035, against 0.052 for the best of five adaptive baselines. The restarted recursion has the smallest frequency gap over the first 50 forecasts after every synthetic break. Real breaks are mos
    
[^66]: StochLOB：用于稳健限价订单簿趋势预测的重尾分数匹配

    StochLOB: Heavy-Tailed Score Matching for Robust Limit Order Book Trend Prediction

    [https://arxiv.org/abs/2610.21081](https://arxiv.org/abs/2610.21081)

    该论文提出StochLOB训练框架，用跳跃Lévy噪声和截断α稳定噪声等重尾非高斯扩散过程替代传统的布朗高斯噪声来加噪训练数据，以更好地捕捉限价订单簿订单流的重尾与跳跃特性，从而提升LOB趋势预测的稳健性，同时推理阶段保持确定性预测器。

    

    基于分数的扩散方法通常使用布朗高斯噪声对数据进行扰动，这对应于轻尾的前向随机微分方程（SDE）。对于市场时间序列而言，这种SDE可能是一种糟糕的抽象：限价订单簿（LOB）的订单流具有重尾性、突发性，且偶尔呈现跳跃状。我们研究了STOCHLOB，这是一个仅用于训练阶段的分数匹配框架，它改变了前向扩散的损坏规律，同时保持部署的预测器为确定性模型。该加噪过程不再仅使用布朗高斯损坏，而是采用两种受非高斯SDE启发的替代方案：通过复合泊松方差混合实现的跳跃Lévy噪声，以及通过正稳定高斯尺度混合实现的截断α稳定噪声。这些损坏机制在训练中注入了有限跳跃和重尾残差，而无需在推理阶段进行反向时间扩散采样。所得的分数目标具有一个有用的解释：其径向因子h（摘要在此处截断）

    arXiv:2610.21081v1 Announce Type: new  Abstract: Score-based diffusion methods usually perturb data with Brownian Gaussian noise, corresponding to a light-tailed forward stochastic differential equation (SDE). For market time series this SDE can be a poor abstraction: limit order book (LOB) order flow is heavy-tailed, bursty, and occasionally jump-like. We study STOCHLOB, a training-only score-matching framework that changes the forward diffusion corruption law while leaving the deployed predictor deterministic. Instead of Brownian-only Gaussian corruption, the noising process uses two non-Gaussian SDE-inspired alternatives: jump-L\'evy noise through a compound-Poisson variance mixture, and clipped $\alpha$-stable noise through a positive-stable Gaussian scale mixture. These corruptions inject finite jumps and heavy-tailed residuals into training without requiring reverse-time diffusion sampling at inference. The resulting score target has a useful interpretation: its radial factor $h_
    
[^67]: SNACK：一种用于概率图生成的序列表示框架

    SNACK: A Sequential Notation Framework for Probabilistic Graph Generation

    [https://arxiv.org/abs/2610.19781](https://arxiv.org/abs/2610.19781)

    SNACK提出了一种将图线性化为字符串序列的表示框架，通过序列与有序邻接矩阵的双射对应和无效logit屏蔽机制，使序列模型能够以100%有效率生成图，并在保持有竞争力样本质量的同时显著提升GFlowNet训练吞吐量。

    

    我们提出了SNACK，这是一种用于概率图生成的字符串表示方法，使序列模型能够生成并建模具有离散节点和边特征的图。SNACK通过编码下三角邻接矩阵的非零元素来实现图的线性化，在有效序列与有序邻接矩阵之间建立了双射对应关系。这种设计使得无效logit屏蔽成为可能，从而能够强制执行结构约束（如化学价规则和排序规则），保证生成的序列100%有效。SNACK还通过显式考虑节点排序对称性，实现了基于字符串的原则性GFlowNet训练。在标准图生成任务和分子分布学习基准测试中，SNACK取得了具有竞争力的样本质量，同时相对于基于图的生成方法大幅提升了GFlowNet的训练吞吐量。探测分析进一步表明，在SNACK上训练的模型编码了（原文摘要在此处被截断）

    arXiv:2610.19781v1 Announce Type: cross  Abstract: We introduce SNACK, a string representation for probabilistic graph generation that enables sequence models to generate and model graphs with discrete node and edge features. SNACK linearizes graphs by encoding the nonzero entries of the lower-triangular adjacency matrix, establishing a bijective correspondence between valid sequences and ordered adjacency matrices. This design enables invalid logit masking to enforce structural constraints--such as chemical valence and ordering rules--guaranteeing 100% valid generated sequences. SNACK also enables principled string-based GFlowNet training by explicitly accounting for node-ordering symmetries. Across standard graph-generation tasks and molecular distribution-learning benchmarks, SNACK achieves competitive sample quality, while substantially improving the throughput of GFlowNet training relative to graph-based generation. Probing analyses further show that models trained on SNACK encode
    
[^68]: 基于伴随方法的随机多尺度生物过程数字孪生校准与最优控制

    Adjoint-Based Calibration and Optimal Control of Stochastic Multiscale Bioprocess Digital Twins

    [https://arxiv.org/abs/2610.09505](https://arxiv.org/abs/2610.09505)

    本文提出了一个基于伴随敏感性分析的偏差感知数字孪生校准与最优控制框架，通过拟似然估计、矩展开和正反向伴随方法量化校准不确定性对策略性能的传播影响，实现了多尺度生物过程的不确定性感知策略优化与自适应实验设计。

    

    我们在生物系统体系（Bio-SoS）范式下，开发了一个面向多尺度生物过程模型的、具有偏差感知能力的数字孪生校准与控制框架。该数字孪生由随机微分方程（SDE）模型表示，并利用拟似然估计和伴随敏感性分析从稀疏的离散观测数据中进行校准。基于SDE生成算子的矩展开刻画了截断引起的参数偏差，而正反向伴随方法则量化了校准不确定性如何传播至价值函数和策略性能。由此得到的参数误差分布既支持面向策略的自适应实验设计，也支持通过二阶高斯平均目标实现的不确定性感知策略优化。我们刻画了所得探索准则的渐近行为，并推导了在优化策略下物理系统的性能表现。为了实现这些想法，我们开发了……（原文摘要在此处截断）

    arXiv:2610.09505v1 Announce Type: cross  Abstract: We develop a bias-aware digital-twin calibration and control framework for multiscale bioprocess models within a biological systems-of-systems (Bio-SoS) paradigm. The digital twin is represented by a stochastic differential equation (SDE) model and calibrated from sparse, discrete observations using quasi-likelihood estimation and adjoint sensitivity analysis. SDE generator-based moment expansions characterize truncation-induced parameter bias, while forward-backward adjoints quantify how calibration uncertainty propagates to value functions and policy performance. The resulting parameter-error distribution supports both policy-directed adaptive experimental design and uncertainty-aware policy optimization through a second-order Gaussian-averaged objective. We characterize the asymptotic behavior of the resulting exploration criterion and derive a physical-system performance under the optimized policy. To implement these ideas, we deve
    
[^69]: 从截断到承诺：均匀离散扩散中的持久上下文

    From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion

    [https://arxiv.org/abs/2609.01043](https://arxiv.org/abs/2609.01043)

    提出一种无需训练的承诺式揭示采样（CRS），将选定的词元作为持久上下文插入后续模型输入，使均匀离散扩散模型的并行预测能在序列级选择上保持一致。

    

    均匀状态离散扩散模型并行更新所有词元，同时保持每个位置都可被修改。即使常用的 top-p 规则在一个位置只留下一个候选，该选择也仅影响当前的反向步骤，并可在下一个采样步骤中被修改。我们探讨当被选中的假设转而成为后续预测的持久上下文时会发生什么变化。为此，我们提出了承诺式揭示采样，这是一种无需训练的采样器，它存储被选中的 argmax 词元，并将其插入后续的模型输入中。我们的分析为“更晚做出选择”和“保持被选词元可见”提供了理论依据：在精确的前向过程下，随着噪声降低，选择干净词元的贝叶斯误差不会增加；而在一个简单的潜变量模式模型中，保持被选词元可见有助于后续的并行预测在相同的序列级选择上达成一致。实证上，在 Duo-distilled 模型上的成对实验（摘要在此处截断）……

    arXiv:2609.01043v1 Announce Type: cross  Abstract: Uniform-state discrete diffusion models update all tokens in parallel while keeping every position revisable. Even when the commonly used top-$p$ rule leaves only one candidate at a position, that choice affects only the current reverse step and can be revised at the next sampling step. We ask what changes when selected hypotheses instead become persistent context for later predictions. We therefore propose committed reveal sampling (CRS), a training-free sampler that stores selected argmax tokens and inserts them into subsequent model inputs. Our analysis gives a rationale for selecting later and for keeping selected tokens visible. Under the exact forward process, the Bayes error of selecting a clean token cannot increase as noise decreases, while in a simple latent-mode model, keeping the selected token visible helps later parallel predictions agree on the same sequence-level choice. Empirically, paired experiments on Duo-distilled 
    
[^70]: 超越凸性情形的驯服次梯度未调整朗之万算法

    The Tamed Subgradient Unadjusted Langevin Algorithm beyond Convexity

    [https://arxiv.org/abs/2608.06283](https://arxiv.org/abs/2608.06283)

    提出 SG-TULA 算法，直接利用次梯度实现对非光滑、非凸且梯度超线性增长目标分布的稳定采样，给出了显式常数的 Wasserstein-2 非渐近收敛界与优化超额风险估计，并可应用于 GPT-2 类大模型的预训练势函数。

    

    我们研究从目标分布中进行采样的问�题，其中目标分布的势函数同时具有非光滑、梯度超线性增长以及非凸的特性。我们提出了次梯度驯服未调整朗之万算法（SG-TULA），这是朗之万扩散的一种离散化格式，直接作用于次梯度，无需依赖计算代价高昂的光滑化处理。为应对超线性增长的情形，我们采用驯服技术构建出稳定且显式的计算格式。我们在 Wasserstein-2 距离下推导出非渐近收敛界，所有常数均以维度和逆温度显式给出，改进了目前已知的基于次梯度的朗之万算法的收敛速率。我们还为相关优化问题提供了超额风险估计。我们以显式常数验证了相关假设对于 GPT-2 系大语言模型的正则化预训练势函数成立，并且……

    arXiv:2608.06283v2 Announce Type: replace-cross  Abstract: We study the problem of sampling from target distributions whose potentials are simultaneously non-smooth, subject to superlinear gradient growth, and non-convex. We introduce the Subgradient Tamed Unadjusted Langevin Algorithm (SG-TULA), a discretisation of the Langevin diffusion that operates directly on subgradients, without relying on computationally demanding smoothing procedures. To handle the superlinear regime, taming techniques are employed to produce a stable, explicit scheme. We derive non-asymptotic convergence bounds in Wasserstein-2 distance, with all constants tracked explicitly in terms of dimension and inverse temperature, improving upon the currently known rates for subgradient-based Langevin algorithms. We further provide excess risk estimates for the associated optimisation problem. We verify the assumptions, with explicit constants, for the regularized pretraining potential of a LLM in the GPT-2 lineage and
    
[^71]: 保序保形推断

    Isotonic Conformal Inference

    [https://arxiv.org/abs/2607.16675](https://arxiv.org/abs/2607.16675)

    提出保序保形推断统一框架，利用保序预测器的水平集结构计算局部化保形分位数与p值，在预测和多重筛选任务中提供有限样本覆盖与错误率保证，同时避免反复的保序重拟合。

    

    总体偏差较小的预测器，在以其预测值为条件时仍可能表现出系统性偏差，这限制了其在下游不确定性量化和候选对象筛选中的可靠性。我们提出保序保形推断，这是一个面向预测与多重检验的统一框架，它将校准与推断相分离，并充分利用保序预测器的水平集结构。这一结构使得我们可以计算局部化的保形分位数和 p 值，同时避免在候选响应之间反复进行保序重拟合。对于预测任务，我们提出了分割式保序保形预测和直推式保序保形预测，为这两种方法建立了预测条件覆盖保证，并刻画了它们的有限样本与渐近自校准性质。对于筛选任务，我们提出了保序保形筛选（ICS），它提供有限样本分层条件下……

    arXiv:2607.16675v2 Announce Type: replace  Abstract: A predictor with small overall bias can still exhibit systematic bias conditional on its predicted value, limiting its reliability for downstream uncertainty quantification and candidate screening. We propose Isotonic Conformal Inference, a unified framework for prediction and multiple testing that separates calibration from inference and exploits the level set structure of an isotonic predictor. This structure enables the computation of localized conformal quantiles and $p$-values while avoiding repeated isotonic refitting across candidate responses. For prediction, we introduce Split Isotonic Conformal Prediction (sICP) and Transductive Isotonic Conformal Prediction (tICP), establish prediction-conditional coverage guarantees for both methods, and characterize their finite-sample and asymptotic self-calibration properties. For selection, we introduce Isotonic Conformal Selection (ICS), which provides finite-sample stratum-condition
    
[^72]: 一般空间上的行列式采样：基于扩散几何的快速近似

    Determinantal sampling on general spaces: fast approximation via diffusion geometry

    [https://arxiv.org/abs/2607.06644](https://arxiv.org/abs/2607.06644)

    该论文通过马尔可夫扩散算子的谱核，为流形、图和网络等一般空间上的行列式点过程采样建立了明确的速率保证，突破了传统方法仅适用于欧几里得空间的局限。

    

    arXiv:2607.06644v2 公告类型：替换 摘要：行列式点过程（DPPs）近来已成为一种基于核的方法，作为标准独立采样的替代方案，用于构建高效的小批量数据、核心集以及大规模数据集的其他紧凑表示。基于DPP的采样器的关键优势之一是它们可以部署在非常一般的空间上，相比之下，独立同分布（i.i.d.）之外的更经典采样方法往往只针对特定的结构良好的设置，主要是欧几里得空间。在这项工作中，我们为远远超出已知欧几里得设置的空间中的行列式采样建立了明确的速率保证，重点关注由自然相关的马尔可夫扩散算子的特征空间所得到的谱核，特别包括流形上的拉普拉斯算子、数据点云上的最近邻类型图，以及大规模网络上的离散薛定谔算子。这特别涵盖了……（原文摘要至此截断）

    arXiv:2607.06644v2 Announce Type: replace  Abstract: Determinantal point processes (DPPs) have recently emerged as a kernel-based alternative to standard independent sampling for constructing efficient minibatches, coresets, and other compact representations of large-scale datasets. One of the key strengths of DPP based samplers is that they can be deployed over very general spaces, in contrast to more classical sampling methods beyond i.i.d. which tend to be tailored to particular well-structured settings, principally Euclidean spaces. In this work, we establish explicit rate guarantees for determinantal sampling in spaces that extend far beyond known Euclidean setups, focussing on spectral kernels obtained from eigenspaces of naturally associated Markov diffusion operators, including in particular the Laplacian on manifolds and nearest-neighbour type graphs on data point clouds and the discrete Schrodinger operator on large scale networks. This covers, in particular, the settings of 
    
[^73]: 外生上下文马尔可夫决策过程学习的极小极大PAC界

    Minimax PAC Bounds for Learning in Exogenous Contextual MDPs

    [https://arxiv.org/abs/2606.25170](https://arxiv.org/abs/2606.25170)

    该论文提出了一个在查询已知前后分配采样预算的新型PAC学习框架，并针对带外生上下文的折扣马尔可夫决策过程中的策略评估、最优值估计和最优策略提取任务，给出了极小极大最优的样本复杂度界。

    

    我们引入了一个PAC框架，其中学习者可以在决策之前和决策之时访问采样预言机。样本复杂度由一对 $(n,m)$ 来衡量，其中 $n$ 是在查询已知之前花费的学习预算，$m$ 是每个查询的额外采样预算。我们在带有外生独立同分布（i.i.d.）上下文的折扣马尔可夫决策过程中展示了该框架的相关性，这些上下文在行动之前被揭示。上下文可能影响奖励和转移，但不受智能体的控制。学习者可以对未知的上下文分布和转移核进行采样。我们研究了策略评估（PE）、最优值估计（BVE）和最优策略提取（BPE）三类任务。当奖励和转移已知时，一种方差缩减算法以样本复杂度 $(\widetilde O((1-\gamma)^{-3}\varepsilon^{-2}),0)$ 解决所有这三个任务，该复杂度在对数因子意义下是极小极大最优的。设 $\mathcal{X}$ 为受控状态……（摘要此处截断）

    arXiv:2606.25170v2 Announce Type: replace-cross  Abstract: We introduce a PAC framework in which the learner can access sampling oracles both before and at decision time. Sample complexity is measured by a pair $(n,m)$, where $n$ is the learning budget spent before a query is known and $m$ is the additional sampling budget per query. We demonstrate its relevance in discounted Markov decision processes with exogenous i.i.d.\ contexts revealed before acting. Contexts may affect both rewards and transitions but remain uncontrolled by the agent. The learner can sample the unknown context distribution and the transition kernel. We study policy evaluation (PE), best-value estimation (BVE), and best-policy extraction (BPE). When rewards and transitions are known, a variance-reduced algorithm solves all three tasks with sample complexity $\bigl(\widetilde O((1-\gamma)^{-3}\varepsilon^{-2}),0\bigr)$, which is minimax optimal up to logarithmic factors. Let $\mathcal{X}$ be the controlled state s
    
[^74]: 面向已部署安全分类器的在线偏移检测与保形自适应

    Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers

    [https://arxiv.org/abs/2606.11949](https://arxiv.org/abs/2606.11949)

    提出了一个针对已部署安全分类器的在线偏移检测与保形自适应框架，通过PCA降维解决高维密度比失效问题，并引入分数分歧“金丝雀”机制，以高检出率和低延迟实现对分布偏移、逃避攻击等失效模式的监控与自适应保护。

    

    我们研究已部署安全分类器的三种失效模式：分布偏移、针对性逃避攻击，以及来自推理模型监控器的无效分数。基于一个预注册因子实验（4个分类器 × 5种偏移 × 20个随机种子 × 2个窗口）的缩减版800单元子集，方法实现了86.8%的经对照验证的检测率（在偏移开始时或之后报警，同时在配对的无偏移数据流上保持安静；原始TPR为91.5%），在w=100时平均延迟为39.7步，并发现分类器与偏移类型之间存在交互效应（完整案例分析下η²=0.186；删失敏感性分析下为0.02–0.03），这一结果支持对每个分类器进行单独监控。由于偏移语料库大多为有害提示，该方法检测到的是构造出的分数变化，而非准确率损失。加权保形自适应中的密度比步骤在4个分类器中的3个上于高维嵌入中失效；将数据经PCA降至32维可提升时间偏移下的覆盖率（分别提升33和20.5个百分点）。一个分数分歧“金丝雀”（将一个非目标分类器与主分类器进行比较）可……（摘要原文在此处被截断）

    arXiv:2606.11949v5 Announce Type: replace-cross  Abstract: We study three failure modes of deployed safety classifiers: distribution shift, targeted evasion, and invalid scores from reasoning-model monitors. A reduced 800-cell subset of a pre-registered factorial (4 classifiers x 5 shifts x 20 seeds x 2 windows) gives 86.8% control-validated detection (alarm at or after onset, quiet paired no-shift stream; raw TPR 91.5%), mean latency 39.7 steps at w=100 and a classifier-by-shift crossover ($\eta^2 = 0.186$ complete-case; 0.02-0.03 under a censoring sensitivity analysis) that motivates per-classifier monitoring. Shifted corpora are mostly harmful prompts, so this detects a constructed score change, not accuracy loss. The density-ratio step of weighted conformal adaptation collapses for 3/4 classifiers in high-dimensional embeddings; PCA to 32 dimensions raises temporal-shift coverage (+33pp, +20.5pp).   A score-disagreement canary (an un-targeted classifier compared with the primary) f
    
[^75]: 分而校之：基于向量量化的多分类局部校准

    Divide et Calibra: Multiclass Local Calibration via Vector Quantization

    [https://arxiv.org/abs/2605.21060](https://arxiv.org/abs/2605.21060)

    提出一种结合向量量化与狄利克雷浓度索引化参数化的组合式校准方法，通过共享因子构建区域特定的校准映射，即使在校准数据稀疏的潜在空间区域也能实现良好泛化的多分类局部校准。

    

    在高风险场景中，准确且校准良好的机器学习模型是必需的，然而有效的多分类校准仍然具有挑战性：全局方法假设校准误差在整个潜在空间中是同质的，而局部方法通常依赖表示空间的降维，从而丢弃了与误校准相关的潜在方向。为了解决这些问题，我们提出了一种组合式的多分类校准方法，其中针对特定区域的校准映射由共享因子构建。我们通过向量量化（VQ）来实现这一想法——它在表示空间中诱导出结构化的划分，并结合狄利克雷浓度的索引化参数化，实现跨区域的参数共享。我们的方法能够学习异构的校准映射，即使对于潜在空间的稀疏区域也能良好泛化。在基准数据集上的实验显示出显著的改进。

    arXiv:2605.21060v2 Announce Type: replace-cross  Abstract: Accurate and well-calibrated Machine Learning (ML) models are mandatory in high-stakes settings, yet effective multiclass calibration remains challenging: global approaches assume calibration errors are homogeneous across the latent space, while local methods often rely on representation-space dimensionality reduction, discarding latent directions relevant to miscalibration. To address these issues, we propose a compositional approach to multiclass calibration, where region-specific calibration maps are constructed from shared factors. We instantiate this idea via Vector Quantization (VQ), which induces a structured partition of the representation space, and an indexed parameterization of Dirichlet concentrations that enables parameter sharing across regions. Our approach learns heterogeneous calibration maps that generalize well even to sparse regions of the latent space. Experiments on benchmark datasets show significant impr
    
[^76]: 连续扩散在语言建模中与离散扩散具有同等竞争力的扩展性

    Continuous Diffusion Scales Competitively with Discrete Diffusion for Language

    [https://arxiv.org/abs/2605.18530](https://arxiv.org/abs/2605.18530)

    该研究通过将 Plaid 与现代离散扩散语言模型架构对齐构建了 RePlaid，建立了首个可媲美离散 DLM 的连续扩散语言模型缩放定律，证明连续扩散是语言建模中高度竞争且可扩展的方案。

    

    尽管扩散模型近来在语言建模领域引起了广泛关注，但连续扩散的可扩展性似乎不如离散方法。为了挑战这一观点，我们重新审视了基于似然的连续扩散语言模型 Plaid，并通过将 Plaid 的架构与现代离散 DLM 对齐，构建了 RePlaid。在这一统一设置下，我们建立了首个可与离散 DLM 相媲美的连续 DLM 缩放定律：RePlaid 与自回归模型相比仅存在约 $20\times$ 的计算差距，在使用更少参数的情况下优于 Duo，并在过度训练的情形下优于 MDLM。我们将 RePlaid 与近期的连续 DLM 进行了基准测试：在 OpenWebText 数据集上，RePlaid 在连续 DLM 中实现了新的最先进困惑度上界 $22.1$，并具有更优的生成质量。这些结果表明，通过似然训练的连续扩散是一种极具竞争力且可扩展的替代方案。

    arXiv:2605.18530v2 Announce Type: replace  Abstract: While diffusion has drawn considerable recent attention from the language modeling community, continuous diffusion has appeared less scalable than discrete approaches. To challenge this belief we revisit Plaid, a likelihood-based continuous diffusion language model (DLM), and construct RePlaid by aligning the architecture of Plaid with modern discrete DLMs. In this unified setting, we establish the first scaling law for continuous DLMs that rivals discrete DLMs: RePlaid exhibits a compute gap of only $20\times$ compared to autoregressive models, outperforms Duo while using fewer parameters, and outperforms MDLM in the over-trained regime. We benchmark RePlaid against recent continuous DLMs: on OpenWebText, RePlaid achieves a new state-of-the-art PPL bound of $22.1$ among continuous DLMs and superior generation quality. These results suggest that continuous diffusion, when trained via likelihood, is a highly competitive and scalable a
    
[^77]: 弱监督多校准的统一方法

    Unified Approach for Weakly Supervised Multicalibration

    [https://arxiv.org/abs/2605.09857](https://arxiv.org/abs/2605.09857)

    本文提出了一个统一框架，通过将污染矩阵风险重写与基于见证的校准约束相结合，为缺乏干净标签的弱监督学习场景（如正例-未标注学习等）提供多校准误差估计与事后校正方法，并具有有限样本保证。

    

    多校准要求预测分数在丰富的子群族和分数相关测试中与标签概率保持一致，但现有方法需要干净的输入-标签对来进行评估和后处理。这一假设在弱监督学习（WSL）场景中并不成立——包括正例-未标注学习、未标注-未标注学习和正例-置信度学习——在这些场景中，干净标签的获取成本高昂或根本不可得，而可靠的不确定性估计却可能至关重要。我们通过为无法获得干净输入-标签对的弱监督设置开发多校准误差估计器和事后校正方法来填补这一空白。我们提出了一个在弱监督下估计和校正多校准的统一框架，该框架将污染矩阵风险重写与基于见证的校准约束相结合，从而得到具有有限样本保证的校正后多校准矩。我们进一步提出……（摘要原文此处截断）

    arXiv:2605.09857v2 Announce Type: replace  Abstract: Multicalibration requires predicted scores to agree with label probabilities across rich families of subgroups and score-dependent tests, but existing methods require clean input-label pairs for evaluation and post-processing. This assumption fails in weakly supervised learning (WSL) regimes -- including positive-unlabeled, unlabeled-unlabeled, and positive-confidence learning -- where clean labels are costly or unavailable even though reliable uncertainty estimates may be crucial. We address this gap by developing estimators of multicalibration error and post-hoc correction methods for WSL settings in which clean input-label pairs are unavailable. We propose a unified framework for estimating and correcting multicalibration under weak supervision by combining contamination-matrix risk rewrites with witness-based calibration constraints, yielding corrected multicalibration moments with finite-sample guarantees. We further propose wea
    
[^78]: 线性核心替代损失函数：具有线性速率的平滑损失函数用于分类与结构化预测

    Linear-Core Surrogates: Smooth Loss Functions with Linear Rates for Classification and Structured Prediction

    [https://arxiv.org/abs/2604.27742](https://arxiv.org/abs/2604.27742)

    本文提出了线性核心替代损失函数，通过将线性核心与平滑尾部拼接，首次实现了处处可微且具有严格线性$H$-一致性界的凸损失函数，解决了分类理论中平滑性与统计效率之间的根本矛盾。

    

    arXiv:2604.27742v2 公告类型：替换 摘要：分类理论中的一个基本二分法将平滑性与统计效率对立起来：平滑替代损失函数（如逻辑损失）能够实现快速的$O(1/T)$优化，但产生缓慢的平方根$H$-一致性界，而分段线性损失函数（如铰链损失）能达到最优的线性$H$-一致性速率，但不可微。我们引入了线性核心（LC）替代损失函数，这是首个能够明确解决这一矛盾的凸损失函数族。通过将线性核心与平滑尾部拼接，我们构建了处处可微（$C^1$，且在温和条件下甚至$C^2$）的替代损失函数，同时保持严格的线性$H$-一致性界，这是已知最强形式的一致性保证。我们在三种日益复杂的场景中建立了这些线性界：二元分类、多类分类和结构化预测。据我们所知，这是首个明确的凸损失函数族，能够同时实现可微性和线性一致性速率。

    arXiv:2604.27742v2 Announce Type: replace  Abstract: A fundamental dichotomy in the theory of classification sets smoothness against statistical efficiency: smooth surrogate losses such as the logistic loss enable fast $O(1/T)$ optimization but yield slow square-root $H$-consistency bounds, while piecewise-linear losses like the Hinge loss achieve optimal linear $H$-consistency rates but are non-differentiable. We introduce Linear-Core (LC) Surrogates, the first family of explicit convex loss functions that provably resolve this tension. By stitching a linear core to a smooth tail, we construct surrogates that are differentiable everywhere ($C^1$, and even $C^2$ under mild conditions) while retaining strict linear $H$-consistency bounds, the strongest known form of consistency guarantee. We establish these linear bounds across three increasingly complex settings: binary classification, multi-class classification, and structured prediction. To our knowledge, this is the first explicit c
    
[^79]: 迭代识别闭包：增强线性结构方程模型中的因果可识别性

    Iterative Identification Closure: Amplifying Causal Identifiability in Linear SEMs

    [https://arxiv.org/abs/2604.09309](https://arxiv.org/abs/2604.09309)

    该论文提出迭代识别闭包（IIC）框架，通过将外部信息源识别出的初始已知边与迭代约化的HTC传播相结合，突破了传统半Trek准则节点级识别的局限，显著提升了含潜在混杂因子的线性结构方程模型中因果效应的可识别性。

    

    半Trek准则（HTC）是判定含有潜在混杂因子的线性结构方程模型（SEMs）中因果效应系数一般可识别性的主要图论工具。然而，HTC本质上是节点级的：它同时解析某个节点的所有入边，从而留下一部分“无法判定”的因果效应空白（在中等规模图中占15-23%）。我们提出了迭代识别闭包（IIC），这是一个通用框架，将因果识别解耦为两个阶段：（1）种子函数S_0，可从任何外部信息源（工具变量、干预、非高斯性、先验知识等）识别出初始的边集合；（2）约化HTC传播，通过迭代代入已知系数来降低系统维度，从而能够识别标准HTC无法解析的边。其核心创新在于迭代识别传播：新识别出的边会反馈（原文摘要在此处截断）。

    arXiv:2604.09309v2 Announce Type: replace  Abstract: The Half-Trek Criterion (HTC) is the primary graphical tool for determining generic identifiability of causal effect coefficients in linear structural equation models (SEMs) with latent confounders. However, HTC is inherently node-wise: it simultaneously resolves all incoming edges of a node, leaving a gap of "inconclusive" causal effects (15-23% in moderate graphs). We introduce Iterative Identification Closure (IIC), a general framework that decouples causal identification into two phases: (1) a seed function S_0 that identifies an initial set of edges from any external source of information (instrumental variables, interventions, non-Gaussianity, prior knowledge, etc.); and (2) Reduced HTC propagation that iteratively substitutes known coefficients to reduce system dimension, enabling identification of edges that standard HTC cannot resolve. The core novelty is iterative identification propagation: newly identified edges feed back
    
[^80]: Lumbermark：通过切分互达性最小生成树实现的抗噪声聚类算法

    Lumbermark: Resistant Clustering by Chopping Up Mutual Reachability Minimum Spanning Trees

    [https://arxiv.org/abs/2604.07143](https://arxiv.org/abs/2604.07143)

    提出了Lumbermark，一种通过迭代切分互达性最小生成树来检测不同大小、密度和形状聚类的稳健分裂式聚类算法，可作为HDBSCAN的替代方案并支持生成用户指定大小的分区。

    

    我们提出了Lumbermark，一种稳健的分裂式聚类算法，能够检测不同大小、密度和形状的聚类。Lumbermark通过迭代地切掉数据集互达性最小生成树上由突出片段连接的大枝干来进行聚类。互达性距离的使用平滑了数据分布，降低了低密度对象（如聚类之间的噪声点或聚类边缘的离群点）的影响。该算法可被视为Genie的分裂式版本，也是HDBSCAN的替代方案，能够生成用户指定大小的分区。这种新方法的快速、易用的实现已在开源的Python和R软件包lumbermark中提供。我们展示了Lumbermark在基准数据上的良好表现，并希望它能对不同领域的数据科学家和从业者有所帮助。

    arXiv:2604.07143v2 Announce Type: replace-cross  Abstract: We introduce Lumbermark, a robust divisive clustering algorithm capable of detecting clusters of varying sizes, densities, and shapes. Lumbermark iteratively chops off large limbs connected by protruding segments of a dataset's mutual reachability minimum spanning tree. The use of mutual reachability distances smoothens the data distribution and decreases the influence of low-density objects, such as noise points between clusters or outliers at their peripheries. The algorithm can be viewed as a divisive version of Genie and an alternative to HDBSCAN that produces partitions with user-specified sizes. A fast, easy-to-use implementation of this new method is available in the open-source lumbermark package for Python and R. We show that Lumbermark performs well on benchmark data and hope it will prove useful to data scientists and practitioners across different fields.
    
[^81]: 贝叶斯神经网络的最小最大性与容许性

    Minimaxity and Admissibility of Bayesian Neural Networks

    [https://arxiv.org/abs/2604.04673](https://arxiv.org/abs/2604.04673)

    本文从统计决策理论出发，证明了固定先验尺度下深度ReLU贝叶斯神经网络诱导的贝叶斯决策规则不具有最小最大性，并通过在有效输出方差上引入超先验，构造出同时具有容许性与最小最大性的决策规则，且将结果扩展到了KL损失下的预测密度估计问题。

    

    贝叶斯神经网络（BNN）为深度学习模型中的推断提供了一种自然的概率化表述。尽管贝叶斯神经网络广受欢迎，但从统计决策理论的角度对其最优性的研究却十分有限。本文研究了在正态位置模型和平方损失下，由深度全连接前馈ReLU贝叶斯神经网络所诱导的决策规则。我们证明，在固定先验尺度的情况下，所诱导的贝叶斯决策规则不是最小最大的（minimax）。随后，我们在BNN先验的有效输出方差上提出一种超先验，该超先验可产生超调和的平方根边际密度，从而证明了由此得到的决策规则同时具有容许性和最小最大性。我们进一步将这些结果从平方损失设定扩展到基于Kullback-Leibler损失的预测密度估计问题。最后，我们通过数值模拟验证了这些理论发现。

    arXiv:2604.04673v2 Announce Type: replace-cross  Abstract: Bayesian neural networks (BNNs) offer a natural probabilistic formulation for inference in deep learning models. Despite their popularity, their optimality has received limited attention through the lens of statistical decision theory. In this paper, we study decision rules induced by deep, fully connected feedforward ReLU BNNs in the normal location model under quadratic loss. We show that, for fixed prior scales, the induced Bayes decision rule is not minimax. We then propose a hyperprior on the effective output variance of the BNN prior that yields a superharmonic square-root marginal density, establishing that the resulting decision rule is simultaneously admissible and minimax. We further extend these results from the quadratic loss setting to the predictive density estimation problem with Kullback--Leibler loss. Finally, we validate our theoretical findings numerically through simulation.
    
[^82]: 预测问题的构建与学习

    Formulation and Learning Of The Forecasting Problem

    [https://arxiv.org/abs/2603.20359](https://arxiv.org/abs/2603.20359)

    本文为数据驱动的动力系统预测方法建立了理论基础，证明了定义预测问题的映射的存在性，并证明该映射可通过监督学习以任意精度逼近。

    

    arXiv:2603.20359v3 公告类型： replace 摘要： 机器学习为纯粹数据驱动的动力系统预测开辟了新的前沿，由此产生的方法在一些应用领域展现出可观的前景。然而，与模型驱动算法相比，对这些数据驱动方法的理论分析发展尚不充分。本文针对这一问题，发展了一套理论来支撑求解预测问题的数据驱动方法。该理论框架包含两个关键部分：(i) 确立定义预测问题的映射的存在性；(ii) 证明该映射可以通过监督学习技术以任意精度进行逼近。我们在连续时间且无噪声的设定下开展工作，因为该设定提供了一个简单的规范化环境，便于建立基础理论；连续时间设定使得该问题具有算子化的表述形式，其中底层的预测映射定义在路径空间上。

    arXiv:2603.20359v3 Announce Type: replace  Abstract: Machine learning has opened new frontiers in purely data-driven forecasting of dynamical systems; the resulting methods are showing considerable promise in some application domains. However, in contrast to model-driven algorithms, analysis of these data-driven methods is poorly developed. In this paper we address this issue, developing a theory to underpin data-driven methods to solve forecasting problems. The theoretical framework comprises two key components: (i) establishing the existence of a mapping defining the forecasting problem; (ii) showing that it is possible to approximate this mapping by supervised learning techniques, to arbitrary accuracy. We work in the continuous-time and noise-free setting as this defines a simple canonical setting in which to establish an underpinning theory; the continuous-time setting leads to an operator formulation of the problem in which the underlying forecasting map is defined on pathspace. 
    
[^83]: 线性混合因果表示的多领域经验贝叶斯方法

    Multi-Domain Empirical Bayes for Linearly-Mixed Causal Representations

    [https://arxiv.org/abs/2603.18404](https://arxiv.org/abs/2603.18404)

    该论文提出了一种基于经验贝叶斯 f-建模的多领域算法，通过因果结构化得分匹配并利用领域内与领域间的不变结构，从线性混合观测中更准确地估计因果潜变量。

    

    因果表示学习（CRL）旨在从高维观测数据中学习低维因果潜变量。尽管因果表示学习的可识别性已被广泛研究，但其估计方法却较少被探索。我们使用经验贝叶斯（EB）方法从多个领域中估计因果表示，这些领域之间的差异反映了共享因果模型中的干预。多领域因果表示学习构成了一个适合经验贝叶斯的同时推断问题。在此，我们提出了一种经验贝叶斯 f-建模算法，通过利用领域内和领域间的不变结构来提高学习到的因果变量的质量。具体而言，我们考虑了具有干预先验的线性测量模型，该先验由具有已知图结构和干预目标的无环潜因果模型所诱导。我们开发了一种基于因果结构化得分匹配的 EM 风格算法。在合成数据和半合成单细胞数据的实验中，我们提出的方法实现了更准确的……

    arXiv:2603.18404v3 Announce Type: replace  Abstract: Causal representation learning (CRL) aims to learn low-dimensional causal latent variables from high-dimensional observations. While identifiability has been extensively studied for CRL, estimation has been less explored. We use empirical Bayes (EB) to estimate causal representations from multiple domains whose differences reflect interventions in a shared causal model. Multi-domain CRL poses a simultaneous inference problem suited to EB. Here, we propose an EB $f$-modeling algorithm that improves the quality of learned causal variables by exploiting invariant structure within and across domains. Specifically, we consider linear measurement models with interventional priors induced by an acyclic latent causal model with known graph and intervention targets. We develop an EM-style algorithm based on causally structured score matching. In experiments on synthetic and semi-synthetic single-cell data, our proposed method achieves more ac
    
[^84]: 基于差异原理的条件矩模型自适应估计与推断

    Adaptive Estimation and Inference in Conditional Moment Models via the Discrepancy Principle

    [https://arxiv.org/abs/2603.01337](https://arxiv.org/abs/2603.01337)

    本文提出了一种基于差异原理的自适应方法，无需知晓未知的源参数即可在条件矩模型的线性逆问题中自动选择正则化强度，使RDIV和TRAE等估计器达到与已知源参数时相同的误差收敛率。

    

    我们研究由条件矩限制定义的线性逆问题中的自适应估计与推断。这些问题通常是病态的，因此估计器需要正则化，然而最优正则化强度取决于一个几乎从不知晓的源参数。我们开发了一种差异原理，可以从经验损失和预先设定的统计误差界限中选择该参数，而无需知道源参数。该框架涵盖了正则化DeepIV（RDIV）和Tikhonov正则化对抗估计器（TRAE）。在所述可实现性条件以及可比的拟合和校准误差界限下，两种单步程序都获得了强误差和弱误差界限，在源参数不超过1的情况下与已知源参数时的收敛率相匹配。我们还将该框架扩展到在总体迭代可实现性条件下的迭代程序，拓宽了收敛率所能涵盖的源参数范围。

    arXiv:2603.01337v2 Announce Type: replace  Abstract: We study adaptive estimation and inference in linear inverse problems defined by conditional moment restrictions. These problems are typically ill-posed, so estimators require regularization, yet the optimal regularization strength depends on a source parameter that is almost never known. We develop a discrepancy principle that chooses this parameter from empirical losses and prescribed bounds on statistical error, without knowing the source parameter. The framework covers Regularized DeepIV (RDIV) and the Tikhonov Regularized Adversarial Estimator (TRAE). Under the stated realizability conditions and comparable fitting and calibration error bounds, both one-step procedures attain strong- and weak-error bounds that match known-source rates for source parameters up to one. We also extend the framework to iterative procedures under realizability of the population iterates, widening the range of source parameters covered by the rate com
    
[^85]: 误设情形下溢出效应的分解：伪真实估计量与局部-全局扩展

    Decomposition of Spillover Effects Under Misspecification: Pseudo-true Estimands and a Local-Global Extension

    [https://arxiv.org/abs/2602.12023](https://arxiv.org/abs/2602.12023)

    该论文证明当溢出效应的概括度量存在误设时，研究者估计的是真实政策效应的最佳逼近，并进一步表明即使只对局部网络或全局均衡其中一个溢出渠道建模，对应估计量仍能准确恢复该渠道的溢出效应。

    

    为了度量溢出效应，研究者通常用一个简单的度量来概括其他人是否受到处理，例如邻居中受处理者的比例。我们研究当这一概括度量存在误设时，研究者实际估计的是什么。我们证明，研究者估计的是在所选概括度量的所有泛函中对真实政策效应的最佳逼近，而通常的直接效应与溢出效应估计恰好构成这一逼近的组成部分。随后，我们将该框架应用于现金转移等应用中常见的情形：处理效应既通过市场均衡产生全局溢出，又通过网络外部性产生局部溢出，而研究者只对其中一个渠道进行建模。结果表明，即使忽略另一个渠道，网络估计量仍能恢复网络溢出效应，均衡估计量仍能恢复均衡溢出效应。我们通过与一项大型现金转移实验校准的模拟来展示这些结果。

    arXiv:2602.12023v5 Announce Type: replace  Abstract: To measure spillovers, researchers often summarize who else was treated with a simple measure, such as the share of treated neighbors. We study what the researcher estimates when that summary is misspecified. We show that the researcher estimates the best approximation to the true policy effect among all functionals of the chosen summary, and that the usual direct and spillover estimates are exactly the components of this approximation. We then specialize this framework to a setting common in applications such as cash transfers: treatment spills over both globally, through market equilibrium, and locally, through network externalities, while the researcher models only one channel. The network estimator still recovers the network spillover, and the equilibrium estimator the equilibrium spillover, even when the other channel is ignored. We illustrate with a simulation calibrated to a large cash-transfer experiment.
    
[^86]: 赢者诅咒导致数据驱动决策中的虚假承诺

    Winner's Curse Drives False Promises in Data-Driven Decisions

    [https://arxiv.org/abs/2602.08892](https://arxiv.org/abs/2602.08892)

    本文通过对近十年《管理科学》55篇论文的实证审查与理论分析，揭示了基于模型的政策评估因赢者诅咒而系统性高估决策收益，且模型精度、随机处理分配、正确模型设定和样本分割等常见辩护理由均无法消除这一乐观偏差。

    

    数据驱动决策中的一项重大挑战是准确的政策评估——即保证学习到的决策策略能够实现所承诺的收益。一种流行的策略是基于模型的政策评估，即从数据中估计一个模型来推断反事实结果。众所周知，由于赢者诅咒（winner's curse），这种策略会产生对真实收益的过度乐观估计。我们搜索了近期关于数据驱动决策的文献，识别出过去十年发表在《管理科学》（Management Science）上的55篇论文样本；其中除两篇外，其余全部依赖于这种有缺陷的方法。文献中通常给出以下几种辩护理由：(1) 估计出的模型准确、稳定且校准良好；(2) 历史数据采用随机处理分配；(3) 模型族设定正确；(4) 评估方法使用样本分割。遗憾的是，我们证明这些辩护理由的任何组合都无法消除这一问题。

    arXiv:2602.08892v2 Announce Type: replace-cross  Abstract: A major challenge in data-driven decision-making is accurate policy evaluation---i.e., guaranteeing that a learned decision-making policy achieves the promised benefits. A popular strategy is model-based policy evaluation, which estimates a model from data to infer counterfactual outcomes. This strategy is known to produce unwarrantedly optimistic estimates of the true benefit due to the winner's curse. We searched the recent literature on data-driven decision-making, identifying a sample of 55 papers published in Management Science in the past decade; all but two relied on this flawed methodology. Several common justifications are provided: (1) the estimated models are accurate, stable, and well-calibrated, (2) the historical data uses random treatment assignment, (3) the model family is well-specified, and (4) the evaluation methodology uses sample splitting. Unfortunately, we show that no combination of these justifications 
    
[^87]: 论度量空间中的生成问题

    On Generation in Metric Spaces

    [https://arxiv.org/abs/2602.07710](https://arxiv.org/abs/2602.07710)

    该论文将语言生成框架推广到可分度量空间，提出尺度敏感的 $(\varepsilon,\varepsilon')$-闭包维数来刻画可生成性，并揭示了倍增空间中生成性的稳定性与一般度量空间中生成性对尺度和度量的高度敏感性之间的鲜明对比。

    

    我们研究可分度量实例空间中的生成问题。我们通过度量分离来定义新颖性，并允许对抗方与生成方具有非对称的新颖性参数，从而将 Kleinberg 和 Mullainathan [2024] 的语言生成框架扩展到了可数域之外。我们引入了 $(\varepsilon,\varepsilon')$-闭包维数，这是闭包维数的一种尺度敏感的类比，它给出了均匀与非均匀可生成性的刻画，并提供了极限生成的一个充分条件。在此过程中，我们发现了一个鲜明的几何对比：在倍增空间（包括所有有限维赋范空间）中，可生成性在不同新颖性尺度上是稳定的，并且在等价度量下保持不变；然而，在一般度量空间中，可生成性可能高度依赖于尺度和具体的度量选择；即使在自然的无穷维希尔伯特空间 $\ell^2$ 中，所有的生成概念……（摘要在此处截断）

    arXiv:2602.07710v2 Announce Type: replace  Abstract: We study generation in separable metric instance spaces. We extend the language generation framework from Kleinberg and Mullainathan [2024] beyond countable domains by defining novelty through metric separation and allowing asymmetric novelty parameters for the adversary and the generator. We introduce the $(\varepsilon,\varepsilon')$-closure dimension, a scale-sensitive analogue of closure dimension, which yields characterizations of uniform and non-uniform generatability and a sufficient condition for generation in the limit. Along the way, we identify a sharp geometric contrast. Namely, in doubling spaces, including all finite-dimensional normed spaces, generatability is stable across novelty scales and invariant under equivalent metrics. In general metric spaces, however, generatability can be highly scale-sensitive and metric-dependent; even in the natural infinite-dimensional Hilbert space $\ell^2$, all notions of generation ma
    
[^88]: 通过交叉熵控制实现可证明可靠的分类器引导

    Provably Reliable Classifier Guidance via Cross-Entropy Control

    [https://arxiv.org/abs/2601.21200](https://arxiv.org/abs/2601.21200)

    本文证明了对分类器交叉熵损失的控制足以约束扩散模型中分类器引导的误差，从而为基于经验损失训练的分类器引导机制首次提供了可证明的可靠性保证。

    

    分类器引导的扩散模型通过将概率分类器所预测对数概率的梯度添加到逆时间得分上，来生成条件样本。在实践中，该分类器通常通过最小化经验损失函数获得。尽管现有的统计理论在样本量足够大时能够保证良好的泛化性能，但这种训练是否能够产生有效的引导机制仍不清楚。我们在交叉熵损失的框架下研究这一问题，交叉熵损失是分类器训练中广泛使用的损失函数。在对分类器施加温和的光滑性假设下，我们证明控制扩散模型每一步的交叉熵足以控制相应的引导误差。特别地，达到条件KL散度 ε² 的概率分类器所诱导的引导向量具有 Õ(dε) 的均方误差，最高至……

    arXiv:2601.21200v3 Announce Type: replace  Abstract: Classifier-guided diffusion models generate conditional samples by augmenting the reverse-time score with the gradient of the log-probability predicted by a probabilistic classifier. In practice, this classifier is usually obtained by minimizing an empirical loss function. While existing statistical theory guarantees good generalization performance when the sample size is sufficiently large, it remains unclear whether such training yields an effective guidance mechanism.   We study this question in the context of cross-entropy loss, which is widely used for classifier training. Under mild smoothness assumptions on the classifier, we show that controlling the cross-entropy at each diffusion model step is sufficient to control the corresponding guidance error. In particular, probabilistic classifiers achieving conditional KL divergence $\varepsilon^2$ induce guidance vectors with mean squared error $\widetilde O(d \varepsilon )$, up to
    
[^89]: 基于自适应分批的多步前瞻信息强化学习

    Reinforcement Learning with Multi-Step Lookahead Information Via Adaptive Batching

    [https://arxiv.org/abs/2601.10418](https://arxiv.org/abs/2601.10418)

    该论文针对多步前瞻强化学习中寻找最优策略为NP难的问题，提出自适应分批策略（ABP），推导出其最优Bellman方程，并设计了乐观的遗憾最小化学习算法。

    

    我们研究了具有多步前瞻信息的表格型有限时域强化学习问题。在采取行动之前，学习者能够观察到ℓ步的未来转移和奖励实现情况：即在任何可能的行动方案下，智能体将会到达的精确状态以及将获得的奖励。尽管已有研究表明此类信息可以大幅提升价值，但寻找最优策略是NP难问题，因此通常采用两种可处理的启发式方法之一：以预定义大小的分块处理前瞻信息（“固定分批策略”），或模型预测控制。我们首先阐明了这两种方法存在的问题，进而提出以自适应的（依赖状态的）批次来利用前瞻信息；我们将此类策略称为自适应分批策略（ABPs）。我们推导了这类策略的最优Bellman方程，并设计了一种乐观的遗憾最小化算法，从而实现学习……

    arXiv:2601.10418v2 Announce Type: replace-cross  Abstract: We study tabular finite-horizon reinforcement learning problems with multiple steps of lookahead information. Before acting, the learner observes $\ell$ steps of future transition and reward realizations: the exact state the agent would reach and the rewards it would collect under any possible course of action. While it has been shown that such information can drastically boost the value, finding the optimal policy is NP-hard, and it is common to apply one of two tractable heuristics: processing the lookahead in chunks of predefined sizes ('fixed batching policies'), and model predictive control. We first illustrate the problems with these two approaches and propose utilizing the lookahead in adaptive (state-dependent) batches; we refer to such policies as adaptive batching policies (ABPs). We derive the optimal Bellman equations for these strategies and design an optimistic regret-minimizing algorithm that enables learning the
    
[^90]: 各向异性协方差结构对线性网络训练动力学和泛化误差的影响

    The Impact of Anisotropic Covariance Structure on the Training Dynamics and Generalization Error of Linear Networks

    [https://arxiv.org/abs/2601.06961](https://arxiv.org/abs/2601.06961)

    本研究揭示了数据各向异性（尖峰协方差结构）使两层线性网络的学习动力学分为两阶段——先由输入-输出相关性主导，后由数据主方向主导，并推导出泛化误差的解析表达式，量化了数据尖峰结构与任务对齐程度对性能的提升作用。

    

    深度神经网络的成功在很大程度上依赖于训练数据的统计结构。尽管各向同性数据上的学习动力学和泛化性能已得到充分研究，但显著的各向异性对这些关键方面的影响尚未被完全理解。我们研究了数据各向异性的影响——以尖峰协方差结构这一经典且易于处理的模型来表示——在线性回归设定下对两层线性网络的学习动力学和泛化误差的作用。我们的分析表明，学习动力学分为两个明显的阶段：最初由输入-输出相关性主导，随后由数据结构的其他主方向主导。此外，我们推导出了泛化误差的解析表达式，量化了数据的尖峰结构与学习任务的对齐程度如何改善性能。我们的发现为这一领域提供了深入的理论洞见。

    arXiv:2601.06961v2 Announce Type: replace  Abstract: The success of deep neural networks largely depends on the statistical structure of the training data. While learning dynamics and generalization on isotropic data are well-established, the impact of pronounced anisotropy on these crucial aspects is not yet fully understood. We examine the impact of data anisotropy, represented by a spiked covariance structure, a canonical yet tractable model, on the learning dynamics and generalization error of a two-layer linear network in a linear regression setting. Our analysis reveals that the learning dynamics proceed in two distinct phases, governed initially by the input-output correlation and subsequently by other principal directions of the data structure. Furthermore, we derive an analytical expression for the generalization error, quantifying how the alignment of the spike structure of the data with the learning task improves performance. Our findings offer deep theoretical insights into
    
[^91]: 基于核化Stein差异的计算高效拟合优度检验

    Computationally efficient goodness-of-fit tests through kernelized Stein discrepancy

    [https://arxiv.org/abs/2512.20007](https://arxiv.org/abs/2512.20007)

    本文提出一种基于核化Stein差异的计算高效半参数拟合优度检验，并设计了无需重新拟合模型或从中采样的影响调整野自助法来确定检验的显著性水平。

    

    具有难解归一化常数的模型在统计学和机器学习中被广泛使用。评估此类模型的适用性面临重大挑战：从拟合后的模型中获取样本通常需要复杂的采样算法。此外，模型拟合有时需要迭代数值优化，这使得需要重复重新拟合的自助法程序在计算上代价高昂。在本文中，我们利用基于核的检验框架，开发了一种基于核化Stein差异的通用半参数拟合优度检验。我们在一般的干扰参数估计下建立了检验统计量的相合性和渐近零分布。为了构造水平为 $\alpha$ 的检验，我们提出了一种新颖的影响调整野自助法，该方法既不需要重新拟合模型，也不需要从模型中采样。我们证明了所提出的自助检验程序在零假设下的相合性，并……

    arXiv:2512.20007v3 Announce Type: replace-cross  Abstract: Models with intractable normalizing constants are widely used in statistics and machine learning. Assessing the adequacy of such models poses significant challenges: obtaining samples from the fitted model often requires sophisticated sampling algorithms. Moreover, model fitting sometimes requires iterative numerical optimization, making bootstrap procedures that require repeated refitting computationally expensive. In this paper, we leverage the kernel-based testing framework to develop a general semiparametric goodness-of-fit test based on the kernelized Stein discrepancy. We establish the consistency and the asymptotic null distribution of the test statistic under general nuisance estimation. To produce a level-$\alpha$ test, we propose a novel influence-adjusted wild bootstrap that requires neither refitting the model nor sampling from it. We prove the consistency of the proposed bootstrap test procedure under the null and 
    
[^92]: 无偏估计的Wasserstein-Cramér-Rao理论

    Wasserstein-Cram\'er-Rao Theory of Unbiased Estimation

    [https://arxiv.org/abs/2511.07414](https://arxiv.org/abs/2511.07414)

    该论文提出了基于Wasserstein几何的估计量“敏感度”概念，用以刻画数据受到无穷小扰动时估计量的不稳定性，并建立了与经典Cramér-Rao方差理论（基于Fisher-Rao几何）相平行的全新无偏估计理论框架。

    

    经典Cramér-Rao无偏估计理论所关注的量是方差（例如Cramér-Rao下界、该下界在指数族中的精确达到性、以及极大似然估计的渐近有效性），方差刻画的是当估计量的值与来自同一分布的独立采样数据集上的值进行比较时估计量的不稳定性。在本文中，我们关注的是另一个量，它刻画的是当估计量的值与原始数据集经过无穷小加性扰动后的值进行比较时估计量的不稳定性；我们将其称为估计量的“敏感度”。由此建立的敏感度理论基于Wasserstein几何，正如经典方差理论基于Fisher-Rao（等价地，Hellinger）几何一样，这一洞察使我们能够确定一系列与经典结果相类似的理论成果。

    arXiv:2511.07414v2 Announce Type: replace-cross  Abstract: The quantity of interest in the classical Cram\'er-Rao theory of unbiased estimation (e.g., the Cram\'er-Rao lower bound, its exact attainment for exponential families, and asymptotic efficiency of maximum likelihood estimation) is the variance, which represents the instability of an estimator when its value is compared to the value for an independently-sampled data set from the same distribution. In this paper we are interested in a quantity which represents the instability of an estimator when its value is compared to the value for an infinitesimal additive perturbation of the original data set; we refer to this as the "sensitivity" of an estimator. The resulting theory of sensitivity is based on the Wasserstein geometry in the same way that the classical theory of variance is based on the Fisher-Rao (equivalently, Hellinger) geometry, and this insight allows us to determine a collection of results which are analogous to the 
    
[^93]: 论带转移前瞻的强化学习的计算困难性

    On the Hardness of Reinforcement Learning with Transition Look-Ahead

    [https://arxiv.org/abs/2510.19372](https://arxiv.org/abs/2510.19372)

    该论文通过新颖的线性规划方法证明带一步前瞻的强化学习最优规划可在多项式时间内求解，而两步及以上前瞻的规划问题是NP难的，从而精确划定了该问题的可解与不可解边界。

    

    我们研究具有转移前瞻的强化学习（RL），其中智能体可以在决定其行动方案之前，观察到执行任意 ℓ 个动作序列后将会访问哪些状态。虽然这种预测信息能够大幅提升可实现的性能，但我们证明最优地利用这些信息可能伴随着极高的计算代价。具体而言，我们证明了带一步前瞻（ℓ=1）的最优规划问题可以通过一种新颖的线性规划表述在多项式时间内求解。相比之下，当 ℓ ≥ 2 时，该问题变为NP-难问题。我们的结果为强化学习中带转移前瞻的规划问题划定了可解与不可解情形之间的精确边界。

    arXiv:2510.19372v3 Announce Type: replace  Abstract: We study reinforcement learning (RL) with transition look-ahead, where the agent may observe which states would be visited upon playing any sequence of $\ell$ actions before deciding its course of action. While such predictive information can drastically improve the achievable performance, we show that using this information optimally comes at a potentially prohibitive computational cost. Specifically, we prove that optimal planning with one-step look-ahead ($\ell=1$) can be solved in polynomial time through a novel linear programming formulation. In contrast, for $\ell \geq 2$, the problem becomes NP-hard. Our results delineate a precise boundary between tractable and intractable cases for the problem of planning with transition look-ahead in reinforcement learning.
    
[^94]: AlignFlow：利用半离散最优传输改进基于流的生成模型

    AlignFlow: Improving Flow-based Generative Models with Semi-Discrete Optimal Transport

    [https://arxiv.org/abs/2510.15038](https://arxiv.org/abs/2510.15038)

    AlignFlow提出利用半离散最优传输（SDOT）在噪声分布与数据点之间建立显式的、具有收敛保证的最优对齐，从而提升基于流的生成模型的训练效果与可扩展性。

    

    基于流的生成模型（FGMs）能够有效地将噪声转化为复杂的数据分布。在FGM训练过程中引入最优传输（OT）来耦合噪声和数据，已被证明可以改善流轨迹的直线性，从而实现更有效的推理。然而，现有的基于OT的方法使用（小）批量的采样噪声和数据点来估计OT方案，这限制了它们在FGM中对大规模和高维数据集的可扩展性。本文提出了AlignFlow，这是一种新颖的方法，利用半离散最优传输（SDOT）来增强FGM的训练，通过在噪声分布和数据点之间建立显式的、具有收敛保证的最优对齐。SDOT通过将噪声空间划分为拉盖尔单元格来计算传输映射，每个单元格映射到相应的数据点。在FGM训练期间，独立同分布的噪声样本通过SDOT映射与数据点进行配对。

    arXiv:2510.15038v2 Announce Type: replace-cross  Abstract: Flow-based Generative Models (FGMs) effectively transform noise into complex data distributions. Incorporating Optimal Transport (OT) to couple noise and data during FGM training has been shown to improve the straightness of flow trajectories, enabling more effective inference. However, existing OT-based methods estimate the OT plan using (mini-)batches of sampled noise and data points, which limits their scalability to large and high-dimensional datasets in FGMs. This paper introduces AlignFlow, a novel approach that leverages Semi-Discrete Optimal Transport (SDOT) to enhance the training of FGMs by establishing an explicit, optimal alignment between noise distribution and data points with guaranteed convergence. SDOT computes a transport map by partitioning the noise space into Laguerre cells, each mapped to a corresponding data point. During FGM training, i.i.d. noise samples are paired with data points via the SDOT map. Ali
    
[^95]: 滑动窗口签名特征用于时间序列预测

    Sliding-Window Signatures as Features for Time Series Forecasting

    [https://arxiv.org/abs/2510.12337](https://arxiv.org/abs/2510.12337)

    提出了一种基于岭回归与滑动窗口签名特征的时间序列预测框架，能够无需先验假设地捕捉外生变量与目标序列之间的非线性延迟依赖关系，并给出了逼近普适性与平稳性的理论保证。

    

    时间序列的演变不仅受其过去值的影响，还常常受到外部协变量的影响，由此产生了带外生变量的预测问题。协变量与目标序列之间的非线性及延迟效应使这一任务尤其具有挑战性。我们提出了一种基于岭回归并结合滑动窗口签名特征的预测框架，将（半）参数模型的计算效率与非参数方法的灵活性相结合。这些特征能够捕捉目标序列与外生变量之间的复杂依赖关系，而无需对这些关系的形式或过去观测的影响做先验假设。我们在离散时间设定下建立了理论保证，包括滑动窗口变换的逼近普适性与平稳性。为便于实现，我们引入了一种新颖的顺序算法，该算法利用了窗口……（原文摘要此处截断）

    arXiv:2510.12337v2 Announce Type: replace-cross  Abstract: The evolution of a time series is often influenced not only by its past values but also by external covariates, giving rise to forecasting with exogenous variables. Nonlinear and delayed effects between covariates and the target series make this task particularly challenging. We propose a forecasting framework based on Ridge regression with sliding-window signature features, combining the computational efficiency of (semi-)parametric models with the flexibility of nonparametric approaches. These features capture complex dependencies between the target and the exogenous variables without relying on prior assumptions about the form of these relationships or the influence of past observations. We establish theoretical guarantees in the discrete-time setting, including universality of approximation and stationarity for the sliding-window transform. To ease implementation, we introduce a novel sequential algorithm that exploits wind
    
[^96]: 神奇的预训练优化器及其寻找之道

    Fantastic Pretraining Optimizers and Where to Find Them

    [https://arxiv.org/abs/2509.02046](https://arxiv.org/abs/2509.02046)

    该论文通过对十种优化器在多个模型规模和数据比例下的系统性公平评估，指出超参数调优不对等和评估设置缺陷导致了许多新优化器声称的加速被夸大，并给出了公平比较优化器的方法论标准。

    

    长期以来，AdamW一直是语言模型预训练中占主导地位的优化器，尽管有众多研究声称其他替代优化器能提供1.4至2倍的加速。我们认为，两个方法论上的缺陷掩盖了公平的比较并阻碍了这些优化器的实际应用：(i) 超参数调优不对等；(ii) 评估设置有限或具有误导性。为解决这两个问题，我们对十种深度学习优化器进行了系统性研究，涵盖四个模型规模（0.1B至1.2B参数）以及不同的数据与模型比例（Chinchilla最优值的1至8倍）。我们发现，公平且有信息量的比较需要严格的超参数调优，并需要在一系列模型规模和数据与模型比例上、在训练结束时进行评估。首先，对某一优化器最优的超参数对另一优化器可能是次优的，这使得盲目迁移超参数的做法有失公平。其次，许多新提出的优化器相对于经过充分调优的基线的实际加速……

    arXiv:2509.02046v3 Announce Type: replace-cross  Abstract: AdamW has long been the dominant optimizer in language model pretraining, despite numerous claims that alternative optimizers offer 1.4 to 2x speedup. We posit that two methodological shortcomings have obscured fair comparisons and hindered practical adoption: (i) unequal hyperparameter tuning and (ii) limited or misleading evaluation setups. To address these two issues, we conduct a systematic study of ten deep learning optimizers across four model scales (0.1B-1.2B parameters) and data-to-model ratios (1-8x the Chinchilla optimum). We find that fair and informative comparisons require rigorous hyperparameter tuning and evaluations across a range of model scales and data-to-model ratios, performed at the end of training. First, optimal hyperparameters for one optimizer may be suboptimal for another, making blind hyperparameter transfer unfair. Second, the actual speedup of many proposed optimizers over well-tuned baselines is 
    
[^97]: 一种适用于一般缺失模式与机器学习插补的统一统计推断框架

    A Unified Framework for Inference with General Missingness Patterns and Machine Learning Imputation

    [https://arxiv.org/abs/2508.15162](https://arxiv.org/abs/2508.15162)

    本文提出了一个在随机缺失假设和一般缺失模式下、结合机器学习插补的统一统计推断框架，通过按缺失模式分层并聚合观测与预测信息构造估计量，理论上保证效率优于加权完整案例分析。

    

    机器学习（ML）模型的预测结果越来越多地被用于插补缺失数据，但简单直接地使用这些预测存在导致有偏推断的风险。现有方法能够在不依赖预测质量的前提下利用机器学习插补实现有效推断以提升效率，但它们仅局限于简单的缺失数据结构或完全随机缺失（MCAR）假设。我们开发了一个新颖的框架，在随机缺失（MAR）假设和一般缺失模式下，利用机器学习插补对Z估计问题进行有效的统计推断。该方法按不同的缺失模式对数据进行分层，并通过聚合各模式特定的观测信息与预测信息来构造估计量。在理论方面，我们建立了该方法的渐近性质和全局最优聚合结果，从而保证其相对于加权完整案例分析（WCCA）具有效率优势。在实际应用中，该方法实现简单，可利用机器学习模型…

    arXiv:2508.15162v4 Announce Type: replace-cross  Abstract: Predictions from machine learning (ML) models are increasingly used to impute missing data, but their naive use risks biased inference. Existing methods provide valid inference with ML imputations regardless of prediction quality to enhance efficiency, while they are limited to either simple missing data structures or a missing-completely-at-random assumption. We develop a novel framework for valid statistical inference in Z-estimation problems using ML imputations under a missing-at-random assumption and for general missingness patterns. The method stratifies data by distinct missingness patterns and constructs an estimator by aggregating pattern-specific observed and prediction information. Theoretically, we establish the asymptotic properties and globally optimal aggregation that guarantees efficiency dominance over weighted complete-case analyses (WCCA). Practically, the method affords simple implementations by leveraging e
    
[^98]: 用于高斯混合建模的对角加权广义矩估计方法

    Diagonally-Weighted Generalized Method of Moments Estimation for Gaussian Mixture Modeling

    [https://arxiv.org/abs/2507.20459](https://arxiv.org/abs/2507.20459)

    本文提出对角加权广义矩估计方法（DGMM），通过对最优加权矩阵的结构化近似为每个矩阶数分配自适应权重，从而在低信噪比下以极低的计算与存储成本实现高斯混合模型的统计高效估计。

    

    arXiv:2507.20459v2 公告类型：replace-cross 摘要：在 d 维空间中直接计算 L 阶以内的矩时，标准的矩方法（MM）会产生 Ω(d^L) 量级的计算与存储成本；而标准的广义矩方法（GMM）在此之外还需要估计并求逆一个大小为 Ω(d^L)×Ω(d^L)、且常常病态的协方差矩阵以获得最优加权矩阵。Pereira、Kileel 和 Kolda [arXiv:2202.06930 (2022)] 通过隐式矩计算避免了 MM 的张量计算成本。尽管取得了这一进展，如何在保持计算效率的同时引入具有统计学依据的加权仍然是一个重要挑战，尤其是在低信噪比（SNR）的情形下，不同矩阶数的采样方差可能差异巨大。我们提出了对角加权广义矩方法（DGMM），该方法通过对最优加权矩阵的结构化近似，为每个矩阶数分配一个自适应权重……

    arXiv:2507.20459v2 Announce Type: replace-cross  Abstract: With direct computation of moments through order $L$ in $d$ dimensions, standard method of moments (MM) incurs $\Omega(d^L)$ computation and storage costs, and standard generalized method of moments (GMM) additionally requires estimating and inverting an often ill-conditioned covariance matrix of size $\Omega(d^L)$-by-$\Omega(d^L)$ to obtain the optimal weighting matrix. Pereira, Kileel, and Kolda [arXiv:2202.06930 (2022)] avoid the tensor computation cost of MM through implicit moment computation. Despite this progress, incorporating statistically principled weighting while preserving computational efficiency remains an important challenge, particularly at low signal-to-noise ratio (SNR), where sampling variances can differ substantially across moment orders. We introduce diagonally-weighted GMM (DGMM), which assigns one adaptive weight to each moment order through a structured approximation of the optimal weighting matrix whi
    
[^99]: 物理信息神经网络中用于贝叶斯推断的排斥集成方法

    Repulsive Ensembles for Bayesian Inference in Physics-informed Neural Networks

    [https://arxiv.org/abs/2505.17308](https://arxiv.org/abs/2505.17308)

    提出在物理信息神经网络损失函数中加入排斥项的集成方法（RE-PINN），用于微分方程逆问题的贝叶斯不确定性估计，其在集成成员趋于无穷时收敛于真实贝叶斯后验，并比标准集成产生更多样的预测。

    

    物理信息神经网络（PINNs）已被证明是求解微分方程的有效工具，尤其是在考虑逆问题时。当从数据中推断微分方程的解和参数时，不确定性估计比点估计更为可取，因为它们能够反映解的准确程度。在本工作中，我们针对若干基础微分方程的逆问题，采用物理信息神经网络的排斥集成方法（RE-PINN）来获得此类估计。排斥机制通过在损失函数中加入一个特定的排斥项来实现，该项具有这样的性质：当集成成员数量趋于无穷时，集成预测将对应于真实的贝叶斯后验分布。在可行的情况下，我们将集成预测与蒙特卡洛基线方法进行了比较。标准集成往往坍缩到最大后验解，而排斥集成则能产生显著更多样的预测。

    arXiv:2505.17308v2 Announce Type: replace  Abstract: Physics-informed neural networks (PINNs) have proven an effective tool for solving differential equations, in particular when considering the inverse problem. When inferring solutions and parameters of the differential equation from data, uncertainty estimates are preferable to point estimates, as they give an idea about the accuracy of the solution. In this work, we consider the inverse problem for a selection of basic differential equations and employ repulsive ensembles of PINNs (RE-PINN) to obtain such estimates. The repulsion is implemented by adding a particular repulsive term to the loss function, which has the property that the ensemble predictions correspond to the true Bayesian posterior in the limit of infinite ensemble members. Where available, we compare the ensemble predictions to Monte Carlo baselines. Whereas the standard ensemble tends to collapse to maximum-a-posteriori solutions, the repulsive ensemble produces sig
    
[^100]: 近似留一交叉验证与平均场推断的联合分析

    Simultaneous analysis of approximate leave-one-out cross-validation and mean-field inference

    [https://arxiv.org/abs/2501.02624](https://arxiv.org/abs/2501.02624)

    本文通过将ALO-CV与平均场推断进行联合分析，在高斯协变量与强凸性条件下首次证明了近似留一交叉验证对于不可微正则化子也能以可忽略的误差一致地逼近真实的留一泛化误差。

    

    近似留一交叉验证（ALO-CV）是一种被提出用于估计正则化估计量泛化误差的方法，适用于维度与样本量同阶的高维情形，即所谓的“成比例情形”。本文发展了一种新的分析方法，在高斯协变量与强凸性条件下，推导出了ALO-CV在不可微正则化子情形下的一致性。通过条件化论证，本文证明了ALO-CV权重与平均场推断中对应权重之间的差异很小。再结合平均场推断估计量与留一量之间的上界，本文证明了ALO-CV能够在误差项可忽略的范围内逼近真实的留一量。文中明确处理了带平方损失的线性模型、稳健线性回归以及单指标模型。

    arXiv:2501.02624v3 Announce Type: replace-cross  Abstract: Approximate Leave-One-Out Cross-Validation (ALO-CV) is a method that has been proposed to estimate the generalization error of a regularized estimator in the high-dimensional regime where dimension and sample size are of the same order, the so-called ``proportional regime''. A new analysis is developed to derive the consistency of ALO-CV for non-differentiable regularizers under Gaussian covariates and strong convexity. Using a conditioning argument, the difference between the ALO-CV weights and their counterparts in mean-field inference is shown to be small. Combined with upper bounds between the mean-field inference estimate and the leave-one-out quantity, this provides a proof that ALO-CV approximates the leave-one-out quantity up to negligible error terms. Linear models with square loss, robust linear regression and single-index models are explicitly treated.
    
[^101]: 用于供应链管理的图神经网络：一个框架与真实世界基准

    Graph Neural Networks for Supply Chain Management: A Framework and Real World Benchmark

    [https://arxiv.org/abs/2411.08550](https://arxiv.org/abs/2411.08550)

    该论文提出了一个将供应链与图结构相连接以应用图神经网络的框架，并发布了来自孟加拉国领先快消品公司的真实世界供应链规划基准数据集。

    

    图神经网络（GNN）近年来在交通、生物信息学、语言和图像处理领域获得了广泛关注，但其在供应链管理中的应用研究仍然有限。供应链本质上具有图结构特征，使其非常适合应用GNN方法，这类方法能够优化和解决复杂问题。其障碍包括缺乏适当的概念基础、对图在供应链管理（SCM）中应用的熟悉度不足，以及缺乏用于基于GNN的供应链研究的真实世界基准数据集。为解决这些问题，我们讨论并建立了供应链与图结构之间的联系，以实现有效的GNN应用，提供了详细的公式化表达、示例、数学定义和任务指南。此外，我们提供了一个来自孟加拉国一家领先快速消费品（FMCG）公司的多视角真实世界基准数据集，重点关注供应链规划。我们讨论了使用GNN的各种供应链任务，并对多种最先进的方法进行了基准测试。

    arXiv:2411.08550v3 Announce Type: replace-cross  Abstract: Graph Neural Networks (GNNs) have recently gained traction in transportation, bioinformatics, language and image processing, but research on their application to supply chain management remains limited. Supply chains are inherently graph-like, making them ideal for GNN methodologies, which can optimize and solve complex problems. The barriers include a lack of proper conceptual foundations, familiarity with graph applications in SCM, and real-world benchmark datasets for GNN-based supply chain research. To address this, we discuss and connect supply chains with graph structures for effective GNN application, providing detailed formulations, examples, mathematical definitions, and task guidelines. Additionally, we present a multi-perspective real-world benchmark dataset from a leading FMCG company in Bangladesh, focusing on supply chain planning. We discuss various supply chain tasks using GNNs and benchmark several state-of-the
    
[^102]: 高度自适应岭回归

    Highly Adaptive Ridge

    [https://arxiv.org/abs/2410.02680](https://arxiv.org/abs/2410.02680)

    本文提出高度自适应岭回归（HAR）方法，在适用于表格数据的大型非参数函数类中实现了维数无关的 n^{-1/3} L2 收敛速率，且在计算上具有良好的可扩展性。

    

    在本文中，我们提出了高度自适应岭回归：这是一种回归方法，在具有平方可积截面导数的右连续函数类中实现了 $n^{-1/3}$ 的维数无关的 $\mathscr L_2$ 收敛速率（最高至对数因子）。这是一个庞大的非参数函数类，特别适用于表格数据。对于平方误差损失，HAR 恰好是采用基于饱和零阶张量积样条基展开的特定数据自适应核的核岭回归，因此在计算上具有良好的可扩展性。我们证明了 HAR 与在支配混合导数的 Sobolev 空间中的核岭回归之间的密切联系，并强调了实际应用中的差异。我们通过模拟和真实数据验证了我们的理论。

    arXiv:2410.02680v2 Announce Type: replace  Abstract: In this paper we propose the Highly Adaptive Ridge (HAR): a regression method that achieves an $n^{-1/3}$ dimension-free $\mathscr L_2$ convergence rate (up to log factors) in the class of right-continuous functions with square-integrable sectional derivatives. This is a large nonparametric function class that is particularly appropriate for tabular data. For squared error loss, HAR is exactly kernel ridge regression with a specific data-adaptive kernel based on a saturated zero-order tensor-product spline basis expansion, and thus scales reasonably well, computationally. We prove close connections between HAR and kernel ridge regression in Sobolev spaces of dominating mixed derivatives, and highlight practical differences. We use simulation and real data to confirm our theory.
    
[^103]: 通过预条件梯度下降与早停训练的过参数化神经网络在插值空间中实现非参数回归的锐利泛化

    Sharp Generalization for Nonparametric Regression in Interpolation Space by Over-Parameterized Neural Networks Trained with Preconditioned Gradient Descent and Early Stopping

    [https://arxiv.org/abs/2407.11353](https://arxiv.org/abs/2407.11353)

    本文提出一种结合早停的预条件梯度下降算法，使过参数化两层神经网络在插值空间中的非参数回归达到锐利的回归速率。

    

    本文研究了利用具有算法保证训练的过参数化两层神经网络进行非参数回归的问题。我们考虑训练特征从$\RR^d$中的单位球面上均匀抽取的设定，且目标函数位于统计学习理论中常研究的插值空间内。我们证明，使用一种新颖的预条件梯度下降算法并结合早停技术来训练神经网络，可以达到锐利的回归速率$\cO(n^{-\frac{2\alpha s'}{2\alpha s'+1}})$，其中$\alpha$刻画了正则化神经正切核（NTK）特征值的多项式衰减$\lambda_j\asymp j^{-2\alpha}$，且在本文考虑的球面均匀设计下$2\alpha=d/(d-1)$。假设目标函数属于插值空间$\bth{\cH_K}^{s'}$，其中$s'\ge3$。该速率甚至比目前已知的近最优结果更为锐利。

    arXiv:2407.11353v5 Announce Type: replace  Abstract: We study nonparametric regression using an over-parameterized two-layer neural network trained with algorithmic guarantees in this paper. We consider the setting where the training features are drawn uniformly from the unit sphere in $\RR^d$, and the target function lies in an interpolation space commonly studied in statistical learning theory. We demonstrate that training the neural network with a novel Preconditioned Gradient Descent (PGD) algorithm, equipped with early stopping, achieves a sharp regression rate of $\cO(n^{-\frac{2\alpha s'}{2\alpha s'+1}})$, where $\alpha$ parametrizes the polynomial eigenvalue decay of the regular Neural Tangent Kernel (NTK), $\lambda_j\asymp j^{-2\alpha}$, and $2\alpha=d/(d-1)$ under the spherical uniform design considered here. The target function is assumed to belong to the interpolation space $\bth{\cH_K}^{s'}$ with $s'\ge3$. This rate is even sharper than the currently known nearly-optimal r
    
[^104]: 基于潘多拉魔盒基廷斯指数的成本感知贝叶斯优化

    Cost-aware Bayesian Optimization via the Pandora's Box Gittins Index

    [https://arxiv.org/abs/2406.20062](https://arxiv.org/abs/2406.20062)

    本文建立了成本感知贝叶斯优化与经济学中潘多拉魔盒问题之间的新联系，提出将基廷斯指数作为采集函数，在中高维度问题上表现优异，且在无显式成本的经典贝叶斯优化中同样有效。

    

    贝叶斯优化是一种以黑盒方式高效优化未知函数的技术。为了应对实际场景中数据收集需要消耗有限资源的情况，人们希望将函数评估成本显式地纳入贝叶斯优化策略中。为了理解如何做到这一点，我们建立了成本感知贝叶斯优化与潘多拉魔盒问题（一个来自经济学的决策问题）之间此前未被探索的联系。潘多拉魔盒问题存在基于基廷斯指数（Gittins index）表达的贝叶斯最优解，该指数可以被重新解释为一种采集函数。我们研究了将这种采集函数用于成本感知贝叶斯优化，并通过实证表明其表现良好，尤其在中高维度上表现出色。我们进一步表明，这种性能也延续到了没有显式评估成本的经典贝叶斯优化中。

    arXiv:2406.20062v4 Announce Type: replace-cross  Abstract: Bayesian optimization is a technique for efficiently optimizing unknown functions in a black-box manner. To handle practical settings where gathering data requires use of finite resources, it is desirable to explicitly incorporate function evaluation costs into Bayesian optimization policies. To understand how to do so, we develop a previously-unexplored connection between cost-aware Bayesian optimization and the Pandora's Box problem, a decision problem from economics. The Pandora's Box problem admits a Bayesian-optimal solution based on an expression called the Gittins index, which can be reinterpreted as an acquisition function. We study the use of this acquisition function for cost-aware Bayesian optimization, and demonstrate empirically that it performs well, particularly in medium-high dimensions. We further show that this performance carries over to classical Bayesian optimization without explicit evaluation costs. Our w
    
[^105]: 基于仿真优化的预算约束下语言模型提示词集合扩展与精炼

    Budgeted Prompt-Set Expansion and Refinement for Language Models via Simulation Optimization

    [https://arxiv.org/abs/2404.08164](https://arxiv.org/abs/2404.08164)

    本文提出了一种基于仿真优化的两阶段框架，在有限评估预算下将提示词表示为中等维度向量并构建得分代理模型，从而实现语言模型提示词集合的高效扩展与精炼以最大化任务得分。

    

    提示词是用户提供的指令或描述，用于引导生成式语言模型产生满足应用需求的输出。当这些需求发生变化时，现有提示词可能需要被调整和改进。评估候选提示词需要耗费时间和计算资源。因此，我们研究如何在有限的评估预算下，通过仿真优化方法对初始提示词集合进行扩展与精炼，目标是在固定语言模型上最大化预先定义的任务得分。具体而言，我们提出了一个两阶段框架：在第一阶段，将初始提示词集合扩展为有限的候选集合，其中每个提示词由一个中等维度的向量表示；在随后的评估与选择阶段，为表示提示词的中等维度向量构建得分的代理模型。我们提出基于该代理模型依次选择提示词进行评估（原文摘要在此处截断）。

    arXiv:2404.08164v3 Announce Type: replace-cross  Abstract: A prompt is an instruction or description provided by the user to guide a generative language model toward outputs that meet application requirements. When these requirements change, existing prompts may need to be adapted and improved. Evaluating candidate prompts requires time and computational resources. We therefore study how to expand and refine an initial set of prompts under a limited evaluation budget through simulation optimization, aiming to maximize a pre-defined task score for a fixed language model. Specifically, we propose a two-stage framework. In the first stage, we expand the initial set of prompts into a finite candidate set, where each prompt is represented by a moderate-dimensional vector. In the subsequent stage of evaluation and selection, we construct a surrogate model of the score for moderate-dimensional vectors representing the prompts. We propose sequentially selecting prompts for evaluation using the
    
[^106]: 未测量混杂因素下基于负控制变量的因果推断：一种极小极大学习方法

    Causal Inference Under Unmeasured Confounding With Negative Controls: A Minimax Learning Approach

    [https://arxiv.org/abs/2103.14029](https://arxiv.org/abs/2103.14029)

    本文提出了一种无需完备性条件、仅假设桥函数存在的负控制变量识别策略，并基于极小极大学习框架构建了适用于再生核希尔伯特空间和神经网络等一般函数类的新型桥函数估计量。

    

    我们研究了当并非所有混杂因素都被观测到、而负控制变量可用时因果参数的估计问题。最近的研究表明，通过两个所谓的桥函数可以实现因果效应的识别和有效估计。在本文中，我们解决了基于负控制变量进行因果推断的核心挑战：识别和估计这些桥函数。以往的工作通常依赖于完备性条件进行识别，施加唯一性假设进行估计，并专注于桥函数的参数化模型。与此不同，我们提供了一种替代的识别策略，该策略假设两个桥函数的存在，而不需要完备性条件。我们还基于极小极大学习形式为这些函数提出了新的估计量。这些估计量适用于一般的函数类，例如再生核希尔伯特空间和神经网络。

    arXiv:2103.14029v5 Announce Type: replace  Abstract: We study the estimation of causal parameters when not all confounders are observed and instead negative controls are available. Recent work has shown how these can enable identification and efficient estimation via two so-called bridge functions. In this paper, we tackle the primary challenges in causal inference with negative controls: identifying and estimating these bridge functions. Previous work has typically relied on completeness conditions for identification, imposed uniqueness assumptions for estimation, and focused on parametric models for the bridge functions. Instead, we provide an alternative identification strategy that assumes the existence of two bridge functions rather than completeness conditions. We also provide new estimators for these functions based on minimax learning formulations. These estimators accommodate general function classes such as Reproducing Kernel Hilbert Spaces and neural networks. We study finit
    

