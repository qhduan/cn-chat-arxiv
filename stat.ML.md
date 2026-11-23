# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Linear Regression Without a Variance](https://arxiv.org/abs/2611.27983) | 该论文提出一种基于经验残差剪枝的黑盒归约方法，在噪声仅有未知有限α矩（α∈(1,2)）且协变量仅有有限四阶矩的无穷方差情形下，可将任意有限方差回归算法自动转化为达到最优统计速率的弱矩回归算法，且无需知道矩指数、噪声尺度等任何先验信息。 |
| [^2] | [Communication-Efficient Federated Learning of Latent Patient Representations from Multi-Institutional EHRs](https://arxiv.org/abs/2611.27900) | 该论文提出一种通信高效的非迭代联邦谱方法，各机构只需共享边际频率和低秩谱摘要即可从多机构电子健康记录中协同学习共享的潜在患者表征，同时将患者级数据保留在本地以保护隐私。 |
| [^3] | [Autonomous quantum machine learning via Riemannian gradient-based optimization](https://arxiv.org/abs/2611.27815) | 提出了一种基于黎曼梯度优化的完全自主自训练量子机器学习框架，无需量子测量或经典反馈即可在量子域内完成优化，并给出了指数收敛保证。 |
| [^4] | [Energy-preserving Bayesian model reduction: closed-form structured posteriors and probabilistic stability certificates for quadratic operator inference](https://arxiv.org/abs/2611.27794) | 该论文提出在贝叶斯二次算子推断中通过线性等式约束精确施加能量保持性，在不失去闭式高斯后验形式的同时获得全局有界性的线性矩阵稳定条件。 |
| [^5] | [An Asymptotic One-Sample Surrogate for Non-Gaussian ERM with General Convex Regularization](https://arxiv.org/abs/2611.27754) | 该论文提出一种由单个高斯向量驱动的代理模型来近似非高斯特征下的凸经验风险最小化（ERM），证明该代理与原ERM最小化器在归一化均值和预测方差上渐近一致，并给出适用于一般一致强凸正则化及纯Lasso的预测与分类风险显式公式。 |
| [^6] | [Budgeted Subset Selection with a Transferable Learned Value Model](https://arxiv.org/abs/2611.27751) | 提出SALVES框架，将预算约束下的子集选择分解为可跨任务迁移的学习价值模型与基于模拟退火的子集搜索两个组件，摆脱了现有方法对特定应用场景的依赖。 |
| [^7] | [Solving the pre-image problem in kernel PCA: The reproducing property is all you need](https://arxiv.org/abs/2611.27718) | 本文提出仅利用再生性质即可直接求解核PCA的预像问题，无需非线性优化或额外的监督学习，同时在线性核情形下能自然退化为标准PCA的重构映射。 |
| [^8] | [Efficient inference for fairness-constrained sequential decision policies](https://arxiv.org/abs/2611.27714) | 该论文研究了带公平性约束的序贯决策策略的最优价值，证明了其在多阶段非马尔可夫数据下逐路径可微并给出了有效影响函数，进而构造了高效的双重稳健估计量。 |
| [^9] | [Can pretrained Transformers deliver semiparametrically efficient inference?](https://arxiv.org/abs/2611.27666) | 该论文证明单个预训练Transformer无需推导有效影响函数或估计干扰函数，即可通过一次前向传播对一族半参数待估量同时实现有效估计与具有渐近精确覆盖率的有效推断。 |
| [^10] | [PhiJEPA: Denoising Latent Predictions for Visual Representation Learning](https://arxiv.org/abs/2611.27516) | PhiJEPA提出了一种两阶段去噪潜在预测架构，由Transformer从损坏上下文预测掩码教师视图的隐藏token、再由逐token的MLP将其映射到干净token，在ImageNet-1K上预训练后在PASCAL VOC线性分割中达到69.3 mIoU，超越I-JEPA和DSeq-JEPA且训练计算量减少25%。 |
| [^11] | [Verifying and Strengthening Neural Tangent Kernel Theory](https://arxiv.org/abs/2611.27487) | 该论文首次用 Lean 4 对两层网络的神经正切核理论进行了端到端的机器验证形式化证明，证明了梯度流的存在唯一性与残差指数收敛定理，并在此过程中强化了标准理论。 |
| [^12] | [Semiparametric Efficiency for Entropic Optimal Transport under Structured Distribution Shift](https://arxiv.org/abs/2611.27261) | 本文通过精确投影恒等式揭示了结构约束下熵最优传输估计的方差增益何时为正、何时为零，并构造了达到该效率界的混合经验似然估计量。 |
| [^13] | [Distillation with Errors: Sparse Unbiased Backpropagation for Knowledge Distillation](https://arxiv.org/abs/2611.27131) | 本文提出带误差蒸馏（DWE），利用 logit 梯度的零和结构构建稀疏成对的无偏梯度估计，并结合几何感知配对与分层抽样降低方差，从而大幅削减全词表知识蒸馏中输出头反向传播的计算成本。 |
| [^14] | [Reliable Behavior Discovery via False Discovery Rate Control](https://arxiv.org/abs/2611.27104) | 提出了DISCO方法，利用粗化e-过程的迭代搜索，在温和条件下可证明地控制行为发现中的错误发现率，并通过适应数据依赖的假设数量实现了比以往eBH方法更高的检验效能。 |
| [^15] | [Spectral Estimation with Deformed Decompression](https://arxiv.org/abs/2611.26945) | 该论文提出了一种名为“形变解压缩”的谱估计新方法，旨在克服谱求逆的不稳定性以及自由解压缩仅适用于酉不变系综的局限，从而能够仅凭小规模矩阵的信息推断大规模样本协方差矩阵的特征谱。 |
| [^16] | [Large Reported CHB-MIT Adaptation Gains Disappear with Pretrained Encoders and Matched Probes](https://arxiv.org/abs/2611.26929) | 本文通过复现发现，Dani和Liebe声称的SSL微调在CHB-MIT癫痫检测上带来的36.2和29.5点ROC-AUC提升，实际上是配置错误（预训练权重未被正确加载、探测时dropout未关闭）造成的假象——在正确设置下适应增益几乎完全消失。 |
| [^17] | [Memorization and Generalization in Generative Models: A Perspective From Energy-based Modeling and Associative Memory](https://arxiv.org/abs/2611.26797) | 本文从能量建模与联想记忆的视角提出：扩散模型本质上是一种联想记忆，随着数据增长而涌现的“伪状态”吸引子正是生成能力的开端，因此记忆与泛化可以在同一检索过程中共存。 |
| [^18] | [Learning under Bounded Manipulation for Quadratic Incentive Design](https://arxiv.org/abs/2611.26736) | 本文研究了重复二次激励设计博弈中跟随者在操纵预算约束下策略性影响观测的问题，建立了近似理性跟随者的极小极大遗憾率 $\max\{\log T, \min\{C_T,T\}\}$ 及相应的信息论下界。 |
| [^19] | [On the loss landscape of the linear neural network: beyond critical points](https://arxiv.org/abs/2611.26671) | 本文提出了一种包含线性神经网络损失景观中所有临界点的中间几何结构——“临界谷线”，它提供了可二维可视化的损失景观全局描述，并表明梯度下降会被该结构吸引从而沿其持续降低损失至全局最小值。 |
| [^20] | [SDG: State-Dependent Gating at Chunk Boundaries for Linear Attention](https://arxiv.org/abs/2611.26637) | 提出SDG方法，通过在分块边界处依据记忆状态的变化计算逐元素门控，为线性注意力模型提供状态反馈的遗忘机制，同时保持块内并行计算的高效性。 |
| [^21] | [Cancellation-Free Constrained SquareCB for Safe Controller Selection](https://arxiv.org/abs/2611.21880) | 该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。 |
| [^22] | [Does Learning Order Explain Forgetting? The Lie Bracket as an Early-Warning Signal for Catastrophic Forgetting Without Old Data](https://arxiv.org/abs/2611.20087) | 本文实证验证了两个任务梯度场的李括号决定了训练顺序导致的权重差异并与灾难性遗忘相关，且仅凭旧任务训练结束时存储的梯度和前20个Hessian特征对，就能在无需旧数据的条件下早期预测遗忘，效果优于EWC惩罚项。 |
| [^23] | [Reinforcement Learning Under Runtime Confounding: Policy Learning and Off-Policy Evaluation](https://arxiv.org/abs/2610.40056) | 该论文针对强化学习中部署时关键变量不可观测的“运行时混淆”问题，提出了相应的策略学习算法和半参数高效的离线策略评估方法，以解决混淆随时间传播所导致的学习偏差。 |
| [^24] | [CoCoNet: Convex Conformal Network for Multi-Dimensional Uncertainty Regions](https://arxiv.org/abs/2610.24246) | 提出 CoCoNet，通过端到端学习依赖输入的凸一致性分数并直接最小化其子水平集大小，生成灵活数据自适应的凸预测区域，同时借助分割保形校准保留有限样本覆盖率保证。 |
| [^25] | [A Dynamical Theory of LoRA in Continual Learning](https://arxiv.org/abs/2609.39367) | 该论文在师生模型中建立了持续学习下LoRA的精确动力学理论，揭示低秩更新虽能减少对已学任务特征的干扰从而缓解灾难性遗忘，但其初始化会减慢对新任务的适应。 |
| [^26] | [AECSF: Adaptive Ensemble Conditional Score Filtering for High-Dimensional Nonlinear Data Assimilation](https://arxiv.org/abs/2609.32411) | 提出了一种免训练的自适应集成条件得分滤波器AECSF，利用条件Tweedie恒等式构建解析可处理的得分估计器，从而在高维非线性数据同化中同时避免粒子权重退化并捕捉非高斯后验结构。 |
| [^27] | [A General Framework for Metropolis-Adjusted Dikin Walks: Dimension-Square Mixing on Polytopes and Log-Det Walks on Spectrahedra](https://arxiv.org/abs/2608.25273) | 本文提出了一种统一框架，通过精确度量的Metropolis调整Dikin游走，分别在多胞体和谱面体上实现了维数平方混合，并利用提议比较参数将分析简化，显著提升了混合速度的上界。 |
| [^28] | [COMPACT: Spectral Adjustment Scores from an Exhaustive and Nonredundant Set of Invariant Relations](https://arxiv.org/abs/2608.10305) | 该论文提出COMPACT方法，通过穷举检验候选得分、处理与结局之间（允许潜在变量）的所有因果图，识别出在处理到结局的边被添加或移除时保持不变的无条件与条件依赖关系的穷举且非冗余集合，从而构建谱调整得分以辅助因果调整变量选择。 |
| [^29] | [A unified perspective of Gaussian process approximation for differential equations](https://arxiv.org/abs/2607.06292) | 该论文提出了一个统一的贝叶斯视角，通过导数匹配方式将微分方程约束纳入似然函数，把碎片化的高斯过程微分方程数值方法整合进一个共同的概率框架，并同时支持参数估计与解逼近。 |
| [^30] | [A Deep Zero-Inflated Model of North Atlantic Right Whale Presence To Support Blue Economy Management in the U.S. East Coast](https://arxiv.org/abs/2606.14403) | 本文提出了一种深度零膨胀伯努利模型，联合建模物种存在性与检测概率，有效处理被动声学监测数据中的零膨胀和复杂依赖，为濒危物种保护与蓝色经济管理提供新工具。 |
| [^31] | [Recovering Direct Price Effects of Environmental Amenities in Housing Markets: Regression and Causal Machine Learning Model Assessment with Empirical Monte Carlo Simulation](https://arxiv.org/abs/2606.02795) | 该研究利用基于纽约州百万房产交易真实数据生成过程的经验蒙特卡洛模拟，系统评估了回归与因果机器学习方法在估计环境舒适品直接价格效应上的表现，发现广义双重差分回归方法表现最为稳健。 |
| [^32] | [A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions](https://arxiv.org/abs/2605.18898) | 该论文提出用双参数Weibull分布作为统一参考框架来诊断Transformer权重幅值分布，发现跨七个模型系列的FFN权重形状参数高度稳定（约1.20），而查询/键投影则呈现与架构相关的系统性偏离。 |
| [^33] | [Fast Training of Mixture-of-Experts for Time Series Forecasting via Expert Loss Integration](https://arxiv.org/abs/2605.10330) | 提出一种融合专家特定损失与部分在线学习策略的自适应混合专家框架，解决了小门控权重导致的优化难题，在保持计算效率的同时显著提升了时间序列预测性能。 |
| [^34] | [Free Decompression with Algebraic Spectral Curves](https://arxiv.org/abs/2605.03634) | 本文利用代数谱曲线理论，为Stieltjes变换满足代数关系的谱密度建立了更通用的自由解压缩方法，放宽了原有方法的强假设，使跨矩阵规模的谱信息外推能够应用于更现实的机器学习模型。 |
| [^35] | [Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data](https://arxiv.org/abs/2604.26841) | 该论文证明均匀离散扩散语言模型本质上是联想记忆，其吸引盆可通过条件似然最大化而非显式能量函数形成，并揭示了由数据规模支配的从记忆到泛化的急剧转变，使其能够检索未见过的数据。 |
| [^36] | [Multi-User mmWave Beam and Rate Adaptation via Combinatorial Satisficing Bandits](https://arxiv.org/abs/2604.14908) | 本文提出SAT-CTS轻量级策略，将多用户毫米波系统中的波束与速率联合自适应建模为满意化目标的组合半老虎机问题，并首次给出了此类问题的有限时间遗憾界理论保证。 |
| [^37] | [Does This Gradient Spark Joy?](https://arxiv.org/abs/2603.20526) | 论文提出愉悦策略梯度（DG），在前向传播中用优势与惊讶度的乘积“愉悦度”评估样本的学习价值，并通过“近藤门控”只为值得的样本执行昂贵的反向传播，从而以极少的计算实现更快的学习。 |
| [^38] | [Delightful Policy Gradient under Distributed Friction](https://arxiv.org/abs/2603.20521) | 本文提出愉悦策略梯度（DG），通过用优势与惊讶度的乘积作为门控信号，在不依赖行为概率的情况下抑制分布式强化学习中罕见失败带来的噪声，同时保留罕见成功所携带的关键信息。 |
| [^39] | [Power-Law Spectrum of the Random Feature Model](https://arxiv.org/abs/2603.14578) | 该论文刻画了随机特征模型总体协方差的谱结构，证明了数据协方差的幂律谱在经过随机线性投影与非线性激活后仍以幂律形式被保留，从而为神经网络缩放定律背后的谱机制提供了理论解释。 |
| [^40] | [Surprisal-R\'enyi Free Energy](https://arxiv.org/abs/2603.03405) | 本文提出惊异-雷尼自由能（SRFE）这一似然比的归一化雷尼-切尔诺夫对数矩泛函，证明其在端点极限下恢复前向与反向KL散度，并通过局部展开揭示了对数似然比方差作为一阶修正所体现的均值-方差权衡机制。 |
| [^41] | [Efficient and Debiased Learning of Average Hazard Under Non-Proportional Hazards](https://arxiv.org/abs/2602.13475) | 该论文提出一个半参数双重稳健框架，实现了非比例风险情形下协变量调整平均风险（AH）的高效估计与有效的√n推断，为比例风险假设不成立时的因果效应估计提供了可靠的替代方案。 |
| [^42] | [Beyond Arrow: From Impossibility to Possibility in Multi-Criteria Benchmarking](https://arxiv.org/abs/2602.07593) | 该论文突破了阿罗不可能定理对多准则基准排序的限制，识别出单峰、群可分和距离受限三类偏好结构，在这些条件下多指标基准测试可以产生稳定且有意义的大模型排名。 |
| [^43] | [Convergence Rate of the Last Iterate of Stochastic Proximal Algorithms](https://arxiv.org/abs/2602.05489) | 本文在放松有界方差假设的条件下，证明了随机梯度邻近方法与随机增量邻近方法的最后迭代点在逐分量凸性与光滑性下达到对数因子意义下最优的 $\widetilde{O}(1/\sqrt{T})$ 收敛速率，并可直接应用于多任务与联邦学习中的图引导正则项。 |
| [^44] | [Efficient Subgroup Analysis via Optimal Trees with Global Parameter Fusion](https://arxiv.org/abs/2602.04077) | 提出一种基于混合整数优化的融合最优因果树方法，通过全局最优划分与参数融合约束，克服了传统树方法贪婪分裂导致的次优划分和过拟合问题，实现精确的差异性治疗效应亚组识别。 |
| [^45] | [Contractive kinetic Langevin samplers beyond global Lipschitz continuity](https://arxiv.org/abs/2509.12031) | 本文提出两种驯化离散化的动力学朗之万采样器（驯化指数欧拉格式与驯化OBABO格式），突破了全局Lipschitz连续性的限制，可处理梯度超线性增长的对数凹分布采样，并证明其在Wasserstein和全变差距离下可达接近最优的Õ(ε⁻¹)收敛速率。 |
| [^46] | [The Sample Complexity of Distributed Simple Binary Hypothesis Testing under Information Constraints](https://arxiv.org/abs/2506.13686) | 本文解决了信息约束下分布式简单二元假设检验的两个开放问题，证明顺序交互无法降低样本复杂度，并给出了通信约束下样本复杂度的最优紧致界。 |
| [^47] | [Deep Learning-based Approaches for State Space Models: A Selective Review](https://arxiv.org/abs/2412.11211) | 本文对基于深度学习的状态空间模型方法进行了选择性综述，为离散时间与连续时间（如潜在神经ODE和SDE）的深度状态空间模型提供了统一的理论视角。 |
| [^48] | [Average Causal Effect Estimation in DAGs with Hidden Variables: Beyond Back-Door and Front-Door Criteria](https://arxiv.org/abs/2409.03962) | 该论文针对含隐变量DAG中超越后门与前门准则的可识别因果效应，提出了基于序贯回归和贝叶斯法则重参数化的估计方法，从而避免了条件密度估计与数值积分带来的统计和计算难题。 |
| [^49] | [On the Global Convergence of Gradient Descent for multi-layer ResNets in the mean-field regime](https://arxiv.org/abs/2110.02926) | 该论文证明了在平均场机制下，当激活函数满足2-齐次或部分1-齐次条件时，多层ResNet训练所对应的梯度流偏微分方程能收敛到全局最小值，表明足够大的ResNet可通过一阶优化方法找到拟合训练数据的全局最优解。 |
| [^50] | [On the interplay between noise and curvature and its effect on optimization and generalization](https://arxiv.org/abs/1906.07774) | 该论文探讨了损失曲率与梯度噪声的相互作用如何影响优化速度以及泛化间隙的估计，并澄清了费舍尔矩阵、海森矩阵和梯度协方差矩阵之间的区别。 |

# 详细

[^1]: 无需方差假设的最优线性回归

    Optimal Linear Regression Without a Variance

    [https://arxiv.org/abs/2611.27983](https://arxiv.org/abs/2611.27983)

    该论文提出一种基于经验残差剪枝的黑盒归约方法，在噪声仅有未知有限α矩（α∈(1,2)）且协变量仅有有限四阶矩的无穷方差情形下，可将任意有限方差回归算法自动转化为达到最优统计速率的弱矩回归算法，且无需知道矩指数、噪声尺度等任何先验信息。

    

    我们研究了无穷方差情形下的高概率线性回归问题，其中回归噪声仅对某个未知的 α∈(1,2) 具有有限的条件 α-矩 ν，而协变量可能只具有直至给定阶数的有限矩（例如4阶矩）。我们提出了一种黑盒归约方法，能够将有限方差的回归程序转换为弱矩条件下的回归程序。该归约基于经验残差剪枝，并且对矩指数、噪声尺度、协方差以及设计矩都是不可知的（即无需预先知道这些量）。在协变量满足有限四阶矩的条件下，任何达到亚高斯速率的有限方差子程序，当样本数 n 超过某个适当定义的阈值时，均可得到如下结果：在不知道 α 或 ν 的情况下，我们以至少 1−δ 的概率有 ‖θ̂−θ*‖_Σ ≲_α ν^{1/α} ((d+log(1/δ))/n)^{…

    arXiv:2611.27983v1 Announce Type: cross  Abstract: We study high-probability linear regression in the infinite variance regime, wherecthe regression noise has only a finite conditional $\alpha$-moment $\nu$ for some unknown $\alpha\in(1,2)$, and the covariates may only have finite moments up to a given order (e.g., 4th moment). We give a black-box reduction that converts finite-variance regression procedures into weak-moment procedures. The reduction is based on empirical residual pruning and is agnostic to the moment exponent, the noise scale, the covariance, and the design moment. Under a finite fourth-moment condition on the covariates, any finite-variance subroutine attaining the sub-Gaussian rate yields the following when the number of samples $n$ is above a suitably defined threshold: Without knowing $\alpha$ or $\nu$, we have with probability at least $1-\delta$ that $ \|\widehat\theta-\theta^*\|_\Sigma   \lesssim_\alpha   \nu^{1/\alpha}   \big(\frac{d+\log(1/\delta)}{n}\big)^{\
    
[^2]: 多机构电子健康记录中潜在患者表征的通信高效联邦学习

    Communication-Efficient Federated Learning of Latent Patient Representations from Multi-Institutional EHRs

    [https://arxiv.org/abs/2611.27900](https://arxiv.org/abs/2611.27900)

    该论文提出一种通信高效的非迭代联邦谱方法，各机构只需共享边际频率和低秩谱摘要即可从多机构电子健康记录中协同学习共享的潜在患者表征，同时将患者级数据保留在本地以保护隐私。

    

    电子健康记录（EHR）包含丰富的健康信息，可用于揭示潜在的患者表征和具有临床意义的患者亚组。跨机构学习EHR数据非常有价值，因为在单一机构学习的表征可能反映本地数据模式且缺乏泛化能力，然而由于隐私限制和通信成本，跨机构汇集患者级记录通常难以实现。我们提出了一种通信高效、非迭代的联邦谱方法，用于从多机构EHR中学习共享的潜在表征组件。在该方法中，每个机构仅共享聚合的边际频率和低秩谱摘要，而将患者级记录保留在本地。中央服务器利用汇集的边际频率聚合这些摘要，从而恢复共享的低秩结构以及可解释表征所需的规范行归一化方向。

    arXiv:2611.27900v1 Announce Type: cross  Abstract: Electronic health records (EHRs) contain rich health information for uncovering latent patient representations and clinically meaningful subgroups. Learning from EHR data across institutions is valuable because representations learned at a single site may reflect local data patterns and lack generalizability, yet pooling patient-level records across institutions is often impractical because of privacy constraints and communication costs. We propose a communication-efficient, non-iterative federated spectral method for learning shared latent representation components from multi-institutional EHRs, in which each institution shares aggregate marginal frequencies and a low-rank spectral summary while keeping patient-level records local. A central server aggregates these summaries with the pooled marginal frequencies to recover both the shared low-rank structure and the canonical row-normalization direction required for interpretable repres
    
[^3]: 基于黎曼梯度优化的自主量子机器学习

    Autonomous quantum machine learning via Riemannian gradient-based optimization

    [https://arxiv.org/abs/2611.27815](https://arxiv.org/abs/2611.27815)

    提出了一种基于黎曼梯度优化的完全自主自训练量子机器学习框架，无需量子测量或经典反馈即可在量子域内完成优化，并给出了指数收敛保证。

    

    量子机器学习模型在训练过程中通常依赖于重复的量子测量和经典优化，这使得从量子到经典的信息提取过程成为可扩展性的根本瓶颈。我们提出了一种基于黎曼梯度优化的自主自训练量子机器学习框架，其中优化过程完全在量子域内进行，无需量子测量或经典反馈。我们推导了针对广泛类别学习任务的黎曼梯度更新规则，并确定了由此产生的训练动力学能够完全自主实现的充分条件。在此机制下，我们针对足够小的步长建立了训练迭代次数意义上的指数收敛保证，并进一步从电路深度增长的角度分析了训练成本。重要的是，我们进一步表明，训练后的模型……

    arXiv:2611.27815v1 Announce Type: cross  Abstract: Quantum machine learning models typically rely on repeated quantum measurements and classical optimization during training, making the quantum-to-classical information-extraction process a fundamental bottleneck for scalability. We propose a framework for autonomous self-training quantum machine learning, based on Riemannian gradient optimization, in which the optimization proceeds entirely within the quantum domain without requiring quantum measurements or classical feedback. We derive the Riemannian gradient update rules for a broad class of learning tasks and identify sufficient conditions under which the resulting training dynamics can be realized fully autonomously. Within this regime, we establish an exponential convergence guarantee, in the training iterations, for sufficiently small step sizes, and further analyze the training cost in terms of the growth of the circuit depth. Importantly, we further show that the trained model 
    
[^4]: 能量保持的贝叶斯模型降阶：面向二次算子推断的闭式结构化后验与概率稳定性证书

    Energy-preserving Bayesian model reduction: closed-form structured posteriors and probabilistic stability certificates for quadratic operator inference

    [https://arxiv.org/abs/2611.27794](https://arxiv.org/abs/2611.27794)

    该论文提出在贝叶斯二次算子推断中通过线性等式约束精确施加能量保持性，在不失去闭式高斯后验形式的同时获得全局有界性的线性矩阵稳定条件。

    

    贝叶斯算子推断从状态数据中学习具有多项式结构的降阶模型（ROM），并返回关于降阶算子的闭式高斯后验。该后验是无约束的，因此会给动力学无界的算子赋予正概率，而现有做法仅通过对少量样本进行有限时域的仿真测试来控制这一问题。我们证明，二次算子的能量保持性可以表示为一组线性等式约束，可以在贝叶斯框架内精确施加而不失去闭式形式。特别地，约束后验只需对无约束后验进行一次卡尔曼类型的更新即可得到，其样本也可通过对无约束抽样施加相同的修正而精确获得。在这种结构下，针对所有初始条件和有界输入的全局有界性问题可归结为线性算子对称部分上的一个线性矩阵条件。仿真……

    arXiv:2611.27794v1 Announce Type: cross  Abstract: Bayesian operator inference learns reduced-order models (ROMs) with polynomial structure from state data and returns a closed-form Gaussian posterior over the reduced operators. That posterior is unconstrained, so it gives positive probability to operators with unbounded dynamics, and current practice controls this only through finite-horizon simulation tests on a few samples. We show that energy preservation of the quadratic operator is a set of linear equality constraints, which can be imposed exactly inside the Bayesian formulation without losing the closed form. In particular, the constrained posterior follows from one Kalman-type update of the unconstrained one, and its samples are obtained exactly, by the same correction applied to unconstrained draws. Under this structure, global boundedness for all initial conditions and bounded inputs reduces to a linear matrix condition on the symmetric part of the linear operator. The simula
    
[^5]: 具有一般凸正则化的非高斯经验风险最小化的渐近单样本代理模型

    An Asymptotic One-Sample Surrogate for Non-Gaussian ERM with General Convex Regularization

    [https://arxiv.org/abs/2611.27754](https://arxiv.org/abs/2611.27754)

    该论文提出一种由单个高斯向量驱动的代理模型来近似非高斯特征下的凸经验风险最小化（ERM），证明该代理与原ERM最小化器在归一化均值和预测方差上渐近一致，并给出适用于一般一致强凸正则化及纯Lasso的预测与分类风险显式公式。

    

    我们通过由单个高斯向量驱动的代理模型来研究具有非高斯特征和非光滑惩罚项的凸经验风险最小化（ERM）。该代理模型保留了原始惩罚项，并利用特征-标签模型的实际联合分布来校准其系数。我们证明了这种总体校准的存在性和唯一性。当数据满足庞加莱（Poincaré）不等式时，我们证明了对新样本预测分数的高斯近似。结合损失函数的正则性条件，该近似使得ERM最小化器与其代理模型在归一化均值和预测方差上达成一致，并给出了预测风险和分类风险的公式。该结果涵盖了一致强凸惩罚项的一般情形，以及在投影反集中条件下的纯Lasso。核心证明依赖于腔方法的哈密顿-雅可比（Hamilton-Jacobi）论证，可在无需求导的情况下比较两个最小化器的主要统计量。

    arXiv:2611.27754v1 Announce Type: cross  Abstract: We study convex empirical risk minimization (ERM) with non-Gaussian features and nonsmooth penalties through a surrogate driven by a single Gaussian vector. The surrogate preserves the penalty and calibrates its coefficients using the actual joint law of the feature-label model. We prove that this population calibration exists and is unique. We prove a Gaussian approximation for fresh prediction scores when the data satisfy a Poincar\'e inequality.   Together with regularity conditions on the losses, this approximation yields agreement between the ERM minimizer and its surrogate in normalized means and prediction variances, as well as formulas for prediction and classification risks. The result covers general uniformly strongly convex penalties and pure Lasso under projection anti-concentration. The core proof relies on a cavity Hamilton-Jacobi argument that compares the main statistics of the two minimizers without differentiating the
    
[^6]: 基于可迁移学习价值模型的预算约束子集选择

    Budgeted Subset Selection with a Transferable Learned Value Model

    [https://arxiv.org/abs/2611.27751](https://arxiv.org/abs/2611.27751)

    提出SALVES框架，将预算约束下的子集选择分解为可跨任务迁移的学习价值模型与基于模拟退火的子集搜索两个组件，摆脱了现有方法对特定应用场景的依赖。

    

    子集选择是机器学习、统计学和组合优化中的一项基础任务，在主动学习、波段选择、输入选择和实验设计等领域有着广泛应用。子集的效用取决于只有在选择之后才能揭示的信息（如标签或响应），以及所选候选者之间的相互作用。现有方法以特定于应用的方式将价值估计与冗余控制相结合，这限制了它们在其他领域的使用。本文提出了SALVES（基于学习价值估计与搜索的子集获取），这是一个将子集选择分解为两个组件的框架。首先，在隐藏信息已知的任务上通过监督学习训练一个价值模型；在新任务上，该模型预测每个候选者能为当前选择所增加的效用。其次，选择步骤基于这些预测来选取子集，通常采用模拟退火方

    arXiv:2611.27751v1 Announce Type: cross  Abstract: Subset selection is a fundamental task in machine learning, statistics and combinatorial optimisation, with applications in active learning, band selection, input selection and experimental design. The utility of a subset depends on information revealed only after the selection, such as labels or responses, and on interactions among selected candidates. Existing approaches combine value estimation and redundancy control in application-specific ways, which limits their use in other domains. This paper introduces Subset Acquisition with Learned Value Estimates and Search (SALVES), a framework that decomposes subset selection into two components. First, a value model is trained by supervised learning on tasks where the hidden information is known. On a new task, it predicts the utility each candidate would add to the current selection. Second, a selection step chooses the subset from these predictions, typically by simulated annealing wit
    
[^7]: 解决核PCA中的预像问题：再生性质就足够了

    Solving the pre-image problem in kernel PCA: The reproducing property is all you need

    [https://arxiv.org/abs/2611.27718](https://arxiv.org/abs/2611.27718)

    本文提出仅利用再生性质即可直接求解核PCA的预像问题，无需非线性优化或额外的监督学习，同时在线性核情形下能自然退化为标准PCA的重构映射。

    

    核PCA通过核技巧将数据隐式映射到再生核希尔伯特空间（RKHS），从而将PCA扩展到非线性降维，其应用范围涵盖图像去噪到分布外检测。虽然其投影映射已被充分理解，但如何恰当地定义相应的重构映射却是一个长期存在的挑战，因为从RKHS投影回输入空间并非易事。这就是所谓的预像问题。现有方法解决该问题的途径要么依赖非线性优化，这容易产生数值不稳定性且对初始化敏感；要么通过求解一个额外的监督学习问题，带来额外的计算开销。此外，尽管当使用线性核时核PCA的投影映射会退化为标准PCA的投影映射，但大多数现有方法并不能保证在这种情况下恢复PCA的重构映射。

    arXiv:2611.27718v1 Announce Type: new  Abstract: Kernel PCA extends PCA to nonlinear dimensionality reduction by implicitly mapping data into a reproducing kernel Hilbert space (RKHS) via the kernel trick, with applications ranging from image denoising to out-of-distribution detection. While its projection map is well understood, appropriately defining the associated reconstruction map is a longstanding challenge since projecting back from the RKHS to the input space is non-trivial. This is known as the pre-image problem. Existing approaches have tackled this problem by either relying on nonlinear optimization, which is prone to numerical instabilities and sensitive to initialization, or by solving a separate supervised learning problem, incurring additional computational overhead. Furthermore, although the projection map of kernel PCA reduces to that of standard PCA when a linear kernel is used, most of these approaches are not guaranteed to recover PCA's reconstruction map in this ca
    
[^8]: 公平性约束下序贯决策策略的高效推断

    Efficient inference for fairness-constrained sequential decision policies

    [https://arxiv.org/abs/2611.27714](https://arxiv.org/abs/2611.27714)

    该论文研究了带公平性约束的序贯决策策略的最优价值，证明了其在多阶段非马尔可夫数据下逐路径可微并给出了有效影响函数，进而构造了高效的双重稳健估计量。

    

    教育与医疗领域的自适应决策系统需要依次选择干预措施，并且通常被要求不让受保护群体落后太多。我们研究了公平性约束下的最优价值：即在一个平滑参数化的策略类中，在满足约束条件（这些约束是群体特定反事实泛函的平滑函数，例如价值差距、价值底线、最大最小福利和预算）的情况下所能达到的最大总体价值。对于序贯可忽略性假设下的多阶段、非马尔可夫数据，对Karush-Kuhn-Tucker系统的敏感性分析表明，该最优价值是逐路径可微的，其有效影响函数等于各群体价值影响函数的拉格朗日组合。同一分析还给出了最优策略、乘数以及其带来的各群体价值的影响函数。尽管其可行集依赖于数据，交叉拟合的双重稳健估计量仍然是高效的，并可据此构建统计推断。

    arXiv:2611.27714v1 Announce Type: cross  Abstract: Adaptive decision systems in education and health choose interventions sequentially and are often required not to leave a protected group too far behind. We study the fairness-constrained optimal value: the largest population value attainable by a policy in a smooth parametric class subject to constraints that are smooth functions of group specific counterfactual functionals, such as value gaps, floors, max-min welfare and budgets. For multi-stage, non-Markov data under sequential ignorability, a sensitivity analysis of the Karush-Kuhn-Tucker system shows that this value is pathwise differentiable, with efficient influence function equal to the Lagrangian combination of the influence functions of the group values. The same analysis gives the influence functions of the optimal policy, the multiplier and the group values it delivers. A cross-fitted doubly robust estimator is efficient despite its data-dependent feasible set and yields co
    
[^9]: 预训练Transformer能否实现半参数有效推断？

    Can pretrained Transformers deliver semiparametrically efficient inference?

    [https://arxiv.org/abs/2611.27666](https://arxiv.org/abs/2611.27666)

    该论文证明单个预训练Transformer无需推导有效影响函数或估计干扰函数，即可通过一次前向传播对一族半参数待估量同时实现有效估计与具有渐近精确覆盖率的有效推断。

    

    半参数模型为统计推断提供了一个广泛的框架，然而有效估计通常依赖于针对每个待估量量身定制的影响函数校正，以及对每个样本重复进行的干扰函数估计。我们证明，单个预训练Transformer可以在一次前向传播中为一系列待估量提供有效估计和有效推断，而无需推导有效影响函数或估计干扰函数。通过在以真实参数作为唯一标签的模拟数据上进行预训练，该Transformer学习所查询待估量的后验均值和协方差，从而返回一个估计值和一个Wald型置信区域。理论上，我们证明，对于复杂度缓慢增长的局部平坦预训练先验，后验均值对每个查询的待估量都是有效的。该Transformer继承了这种有效性，并具有渐近精确的覆盖率，在评估总体上以概率成立。

    arXiv:2611.27666v1 Announce Type: cross  Abstract: Semiparametric models provide a broad framework for statistical inference, yet efficient estimation typically relies on influence function correction tailored to each estimand and on nuisance estimation repeated for every sample. We show that a single pretrained Transformer can deliver efficient estimation and valid inference for a family of estimands in one forward pass, without deriving efficient influence functions or estimating nuisance functions. Pretrained on simulated data with the true parameter as the only label, the Transformer learns the posterior mean and covariance of the queried estimand and hence returns an estimate and a Wald-type confidence region. Theoretically, we prove that, for locally flat pretraining priors of slowly growing complexity, the posterior mean is efficient for every queried estimand. The Transformer inherits this efficiency with asymptotically exact coverage, in probability over evaluation populations
    
[^10]: PhiJEPA：用于视觉表征学习的去噪潜在预测

    PhiJEPA: Denoising Latent Predictions for Visual Representation Learning

    [https://arxiv.org/abs/2611.27516](https://arxiv.org/abs/2611.27516)

    PhiJEPA提出了一种两阶段去噪潜在预测架构，由Transformer从损坏上下文预测掩码教师视图的隐藏token、再由逐token的MLP将其映射到干净token，在ImageNet-1K上预训练后在PASCAL VOC线性分割中达到69.3 mIoU，超越I-JEPA和DSeq-JEPA且训练计算量减少25%。

    

    一个被训练用于推断图像隐藏区域的预测器，能否在物体部分从视野中消失且可见上下文变得模糊时，仍然帮助世界模型进行规划？我们提出了PhiJEPA，这是一种联合嵌入预测架构，将PhiNet的两阶段预测方式适配到掩码Transformer中。从稀疏的、经过光度损坏的上下文中，一个Transformer块预测独立损坏的教师视图的隐藏token。随后，一个逐token的MLP将这些预测映射到干净的教师token。受控消融实验表明，该组合目标在密集特征方面优于所测试的单阶段替代方案。在ImageNet-1K上以5.6×10^19训练FLOPs进行预训练后，PhiJEPA在线性PASCAL VOC分割任务上达到69.3 mIoU，在分割精度上超过了所评估的I-JEPA和DSeq-JEPA检查点，且训练FLOPs比I-JEPA减少25%。它还在隐藏区域分割方面优于Simple I-JEPA（49.1 vs. 44…）。

    arXiv:2611.27516v1 Announce Type: cross  Abstract: Can a predictor trained to infer hidden image regions still help a world model plan when part of an object disappears from view and the visible context is blurred? We introduce PhiJEPA, a joint-embedding predictive architecture that adapts PhiNet's two-stage prediction to masked Transformers. From sparse, photometrically corrupted context, a Transformer block predicts hidden tokens of an independently corrupted teacher view. A token-wise MLP then maps these predictions to clean teacher tokens. Controlled ablations show that the combined objective improves dense features over the tested one-stage alternatives. Pre-trained on ImageNet-1K with $5.6\times10^{19}$ training FLOPs, PhiJEPA achieves 69.3 mIoU on linear PASCAL VOC segmentation, exceeding the evaluated I-JEPA and DSeq-JEPA checkpoints in segmentation accuracy, with 25\% fewer training FLOPs than I-JEPA. It also improves hidden-region segmentation over Simple I-JEPA (49.1 vs.\ 44
    
[^11]: 验证与强化神经正切核理论

    Verifying and Strengthening Neural Tangent Kernel Theory

    [https://arxiv.org/abs/2611.27487](https://arxiv.org/abs/2611.27487)

    该论文首次用 Lean 4 对两层网络的神经正切核理论进行了端到端的机器验证形式化证明，证明了梯度流的存在唯一性与残差指数收敛定理，并在此过程中强化了标准理论。

    

    神经正切核（NTK）理论解释了为什么通过梯度方法训练的超宽神经网络会在懒惰训练模式下收敛。其标准证明结合了矩阵集中不等式、随机 Gram 矩阵、连续时间动力学、谱扰动和自举论证，但一直缺乏机器可检验的验证。我们提出了针对两层网络的 Lean 4 形式化证明，证明了一个端到端定理：对于光滑有界激活函数（以 σ=tanh 实例化）、冻结偏置、成对 ±1 初始化以及显式的宽度条件，梯度流以至少 1−δ 的概率唯一存在，并且满足 ||r(t)||² ≤ ||y||² e^(−λ₀t/2)。唯一的数据假设是总体 NTK 非退化性，即 λ₀>0。初始强制性、核漂移、移动自举、残差 ODE 以及流的存在性与唯一性均得到了形式化证明。该形式化还强化了标准……

    arXiv:2611.27487v1 Announce Type: new  Abstract: Writing   Neural tangent kernel (NTK) theory explains why very wide neural networks trained by gradient methods converge in the lazy-training regime. Its standard proof combines matrix concentration, random Gram matrices, continuous-time dynamics, spectral perturbation, and a bootstrap argument, but has lacked machine-checked verification. We present a Lean\~4 formalisation for two-layer networks, proving an end-to-end theorem: for a smooth bounded activation (instantiated with $\sigma=\tanh$), frozen biases, paired $\pm 1$ initialisation, and explicit width conditions, gradient flow exists uniquely with probability at least $1-\delta$, and   $$ \|r(t)\|^2 \leq \|y\|^2 e^{-\lambda_0 t/2}. $$   The sole data assumption is population NTK non-degeneracy, $\lambda_0>0$. Initial coercivity, kernel drift, the movement bootstrap, the residual ODE, and flow existence and uniqueness are formally proved.   The formalisation also sharpens the stand
    
[^12]: 结构化分布偏移下熵正则最优传输的半参数有效性

    Semiparametric Efficiency for Entropic Optimal Transport under Structured Distribution Shift

    [https://arxiv.org/abs/2611.27261](https://arxiv.org/abs/2611.27261)

    本文通过精确投影恒等式揭示了结构约束下熵最优传输估计的方差增益何时为正、何时为零，并构造了达到该效率界的混合经验似然估计量。

    

    结构信息能否改进熵正则最优传输（EOT）的估计，以及何时增益为零？我们研究了一个固定维数的指数密度比模型（DRM），该模型具有非参数基线、紧支撑集和固定的正则化系数。在正的极限抽样比例下，一个精确的投影恒等式量化了来自限制约束的方差增益。混合经验似然估计量达到了受约束界的水平，并且当该界为正时支持渐近有效的置信区间。即使在模型正确设定的情形下，增益也可能为零，且随着倾斜空间的扩大而缩小。在局部错误设定下，同一投影残差决定了方向特定的一阶偏差。当两个样本同时增长且目标比例趋近于零时，倾斜函数的一阶导数为零会导致源样本波动和二次倾斜误差成为主导项。二者的平衡产生了相变现象。

    arXiv:2611.27261v1 Announce Type: cross  Abstract: Can structural information improve estimation of entropic optimal transport (EOT), and when is the gain zero? We study a fixed-dimensional exponential density ratio model (DRM) with a nonparametric baseline, compact support, and fixed positive regularization. At positive limiting sampling fractions, an exact projection identity quantifies the variance gain from the restriction. The pooled empirical-likelihood estimator attains the restricted bound and supports asymptotically valid confidence intervals when the bound is positive. Even under correct specification, the gain can be zero and can shrink as the tilt space expands. The same projection residual determines direction-specific first-order bias under local misspecification. When both samples grow and the target fraction vanishes, a zero first-order tilt derivative can leave source fluctuations and quadratic tilt error as the leading terms. Their balance gives a phase transition in 
    
[^13]: 带误差的蒸馏：面向知识蒸馏的稀疏无偏反向传播

    Distillation with Errors: Sparse Unbiased Backpropagation for Knowledge Distillation

    [https://arxiv.org/abs/2611.27131](https://arxiv.org/abs/2611.27131)

    本文提出带误差蒸馏（DWE），利用 logit 梯度的零和结构构建稀疏成对的无偏梯度估计，并结合几何感知配对与分层抽样降低方差，从而大幅削减全词表知识蒸馏中输出头反向传播的计算成本。

    

    知识蒸馏（KD）通过训练学生模型来匹配教师模型的输出分布。然而，全词表知识蒸馏需要对输出头进行代价高昂的反向传播。为降低这一成本，我们提出了带误差蒸馏（Distillation with Errors, DWE）。我们的关键洞察是：小批量训练本身已是对训练梯度的近似，因此逐位置的反向传播无需完全精确。为此，DWE 利用 logit 梯度的零和结构，构建稀疏的成对梯度估计，其期望与全词表梯度一致；随后通过几何感知配对和分层抽样进一步降低估计方差。我们证明了该估计器是无偏的，并评估了 DWE 的梯度估计精度与训练效率。在相同的抽样预算下，与重要性抽样相比，DWE 将隐藏状态和输出权重梯度的均方误差（MSE）分别降低了 63.5% 和 57.9%。在离策略（off-policy）知识蒸馏中，它实现了最高达 1.95 倍的加速提升。

    arXiv:2611.27131v1 Announce Type: new  Abstract: Knowledge distillation (KD) trains a student to match a teacher's output distribution. However, full-vocabulary KD requires costly output-head backpropagation. To reduce this cost, we introduce Distillation with Errors (DWE). Our key insight is that minibatch training already approximates the training gradient, so per-position backpropagation does not need to be exact. DWE therefore exploits the zero-sum structure of logit gradients to construct sparse paired gradient estimates whose expectation matches the full-vocabulary gradient. It then uses geometry-aware pairing and stratified sampling to further reduce estimation variance. We prove that the estimator is unbiased and evaluate DWE's gradient estimation accuracy and training efficiency. At the same sampling budget, DWE reduces hidden-state and output-weight gradient MSE by 63.5% and 57.9%, respectively, compared with importance sampling. In off-policy KD, it achieves up to a $1.95\ti
    
[^14]: 通过错误发现率控制实现可靠的行为发现

    Reliable Behavior Discovery via False Discovery Rate Control

    [https://arxiv.org/abs/2611.27104](https://arxiv.org/abs/2611.27104)

    提出了DISCO方法，利用粗化e-过程的迭代搜索，在温和条件下可证明地控制行为发现中的错误发现率，并通过适应数据依赖的假设数量实现了比以往eBH方法更高的检验效能。

    

    随着大型生成式模型变得日益复杂和强大，发现其中行为模式的可靠方法变得越来越重要。我们将行为发现形式化为一类值得关注的问题，并强调控制错误发现率（FDR）是保证行为发现可靠性的关键任务。基于多重假设检验领域的文献，我们提出了DISCO（Discovery through Iterative Search with COarsened e-processes，即通过粗化e-过程的迭代搜索进行发现），这是一种在温和条件下能够可证明地控制行为发现中FDR的方法。DISCO通过允许适应数据依赖的待检验假设数量，相较于之前基于eBH的相关方法提升了检验效能。我们在合成数据以及一项针对大型生成式模型同时作为发现者和发现对象的场景下FDR控制方法的新颖实证评估中，通过实验证明了DISCO能以比可比方法更高的效能控制FDR。

    arXiv:2611.27104v1 Announce Type: new  Abstract: Reliable methods for discovering behavior patterns in large generative models are increasingly essential as models become both more complex and capable. We formalize behavior discovery as a problem class of interest and highlight the task of controlling false discovery rate (FDR) as crucial for guaranteeing reliability in behavior discovery. Building upon the multiple hypothesis testing literature, we propose DISCO (Discovery through Iterative Search with COarsened $e$-processes), a method which provably controls FDR in behavior discovery under mild conditions. DISCO improves power over previous related eBH-based approaches by allowing for adaptivity to a data-dependent number of hypotheses to be tested. We demonstrate empirically that DISCO controls FDR with higher power than comparable approaches on both synthetic data and a novel empirical evaluation of FDR-controlling methods in settings where large generative models act as both disc
    
[^15]: 基于形变解压缩的谱估计

    Spectral Estimation with Deformed Decompression

    [https://arxiv.org/abs/2611.26945](https://arxiv.org/abs/2611.26945)

    该论文提出了一种名为“形变解压缩”的谱估计新方法，旨在克服谱求逆的不稳定性以及自由解压缩仅适用于酉不变系综的局限，从而能够仅凭小规模矩阵的信息推断大规模样本协方差矩阵的特征谱。

    

    样本协方差矩阵是机器学习和统计学中的基本对象，其特征值中编码了有价值的信息。在处理现代问题时，这些矩阵的规模可能会大到使可行计算变得不切实际。这就产生了对“谱解压缩”的关键需求：一种仅利用较小规模实现的信息来推断超大规模模型特征谱的方法。现有方法主要分为两类：一类依赖于谱求逆，另一类依赖于自由解压缩技术。一方面，谱求逆从根本上是一个病态问题，以不稳定性著称，在有限尺寸的情况下往往无法给出可靠的结果；另一方面，尽管最近提出的自由解压缩方法解决了这一规模问题，但其对Nica-Speicher框架的依赖将其应用限制在酉不变系综中，这是一种对称性条件……

    arXiv:2611.26945v1 Announce Type: cross  Abstract: Sample covariance matrices are fundamental objects in machine learning and statistics, with valuable information encoded into their eigenvalues. When treating modern problems, the scale of these matrices can become prohibitive for tractable computation. This creates a critical need for "spectral decompression": a method for inferring the eigenspectrum of a very large-scale model using only information from its smaller realizations. Existing methods fall into two main categories: those relying on spectral inversion; and those that rely on Free Decompression techniques. On the one hand, spectral inversion is a fundamentally ill-posed problem that is notoriously unstable, often failing to yield reliable results in finite-size settings. On the other hand, while Free Decompression was recently proposed to address this scaling, its reliance on the Nica-Speicher framework restricts its use to unitarily invariant ensembles, a symmetry conditio
    
[^16]: 使用预训练编码器与匹配探测方法后，文献中报告的CHB-MIT大幅适应增益消失

    Large Reported CHB-MIT Adaptation Gains Disappear with Pretrained Encoders and Matched Probes

    [https://arxiv.org/abs/2611.26929](https://arxiv.org/abs/2611.26929)

    本文通过复现发现，Dani和Liebe声称的SSL微调在CHB-MIT癫痫检测上带来的36.2和29.5点ROC-AUC提升，实际上是配置错误（预训练权重未被正确加载、探测时dropout未关闭）造成的假象——在正确设置下适应增益几乎完全消失。

    

    Dani 和 Liebe（2026）（简称DL）报告称，通过自监督学习（SSL）对BIOT和CBraMod的最后一个编码器块进行适应微调，相比线性探测可将CHB-MIT癫痫发作检测的ROC-AUC分别提升36.2和29.5个点。我们使用其发布的代码和一条独立流程对这些结果进行了复现研究。结果发现，该代码库只有在提供了一个原本未设置的配置键时才会为适应微调加载预训练权重，且其CBraMod探测头在dropout处于激活状态下读取重建头的输出。在正确进行预训练初始化、并在评估模式下对编码器输出施加探测的情况下，在DL的数据划分上，五个随机种子下的固定配方适应增益为：BIOT为-0.6个点（95%区间[-3.8, +2.6]），CBraMod为+0.01（[-0.02, +0.04]）。在DL的数据划分以及CHB-MIT和Siena头皮脑电数据集的五种受试者轮换设置下，使用同等调优的线性探测（30种配置、最多500轮、基于验证集的早停），平均适应效应范围为-1.1至……（原文摘要在此处截断）

    arXiv:2611.26929v1 Announce Type: new  Abstract: Dani and Liebe (2026) (DL) report that adapting the final encoder block of BIOT and CBraMod with self-supervised learning (SSL) improves CHB-MIT seizure-detection ROC-AUC by 36.2 and 29.5 points over linear probing. We investigate these results with their released code and an independent pipeline. The release loads pretrained weights for adaptation only when an otherwise unset configuration key is supplied, and its CBraMod probe reads the reconstruction head with dropout active. With pretrained initialisation and a probe on the encoder output in evaluation mode, the fixed-recipe adaptation gain on DL's split is -0.6 points for BIOT (95% interval [-3.8, +2.6]) and +0.01 for CBraMod ([-0.02, +0.04]) across five seeds. Across DL's split and five subject rotations on CHB-MIT and Siena Scalp EEG, with equally tuned linear probes (30 configurations, up to 500 epochs, validation-based early stopping), mean adaptation effects range from -1.1 to 
    
[^17]: 生成模型中的记忆与泛化：基于能量建模与联想记忆的视角

    Memorization and Generalization in Generative Models: A Perspective From Energy-based Modeling and Associative Memory

    [https://arxiv.org/abs/2611.26797](https://arxiv.org/abs/2611.26797)

    本文从能量建模与联想记忆的视角提出：扩散模型本质上是一种联想记忆，随着数据增长而涌现的“伪状态”吸引子正是生成能力的开端，因此记忆与泛化可以在同一检索过程中共存。

    

    生成模型如今已能编写可运行的代码、生成逼真的图像并进行长对话，但其智能背后的机制仍未被我们理解。本论文通过联想记忆（Associative Memory, AM）来探索这一谜题：在联想记忆中，模式以能量极小值的形式被存储，并通过吸引子动力学进行检索。除了存储容量之外，AM 还会形成一类被称为“伪状态”的新极小值，这长期以来被视为记忆的失败。我们提出，这种“失败”正是创造力的起点，记忆与泛化可以作为同一检索过程的两种结果而共存。首先，我们证明扩散模型表现为一种联想记忆。当数据很少时，扩散模型会在每个训练样本周围形成各自独立的吸引盆；随着数据量增长，这些吸引盆逐渐缩小，而伪状态作为融合了已存储样本却不匹配任何单个样本的吸引子开始涌现——它们正是生成能力的最初征兆。我们通过……刻画了这一转变（原文摘要在此处截断）

    arXiv:2611.26797v1 Announce Type: new  Abstract: Generative models now write working code, produce realistic images, and hold long conversations, yet the mechanisms behind their intelligence still evade our understanding. This thesis explores that puzzle through Associative Memory (AM), in which patterns are stored as energy minima and retrieved through attractor dynamics. Beyond their capacity, AMs form new minima known as spurious states, long regarded as a failure of memory. We argue that this failure is where creativity begins, and that memorization and generalization can coexist as outcomes of one retrieval process. First, we show that diffusion models behave as AMs. With little data, a diffusion model forms a distinct basin of attraction around each training sample. As the data grows, these basins shrink, and spurious states emerge as attractors that blend stored samples yet match none of them. They are the first signs of generative capability. We characterize this transition by 
    
[^18]: 有界操纵下的二次激励设计学习

    Learning under Bounded Manipulation for Quadratic Incentive Design

    [https://arxiv.org/abs/2611.26736](https://arxiv.org/abs/2611.26736)

    本文研究了重复二次激励设计博弈中跟随者在操纵预算约束下策略性影响观测的问题，建立了近似理性跟随者的极小极大遗憾率 $\max\{\log T, \min\{C_T,T\}\}$ 及相应的信息论下界。

    

    我们研究了一个重复的二次激励设计博弈，其中领导者从跟随者行动的噪声观测中学习跟随者的私人类型，并在仿射激励规则中使用其估计值。跟随者知道领导者的学习策略，该策略利用过去的观测来产生对跟随者类型的估计，并且跟随者可以偏离其短视最优响应以影响领导者未来的估计。我们探究跟随者操纵这些观测的能力如何影响领导者的性能。在操纵预算 $C_T$ 约束跟随者行动相对于其短视最优响应的期望累积平方偏差的条件下，我们为近似理性的跟随者建立了阶为 $\max\{\log T, \min\{C_T,T\}\}$ 的极小极大遗憾率。我们的信息论下界适用于领导者和跟随者的一般类因果估计与操纵策略。我们提供了一个简……

    arXiv:2611.26736v1 Announce Type: cross  Abstract: We study a repeated quadratic incentive design game in which a leader learns a follower's private type from noisy observations of the follower's actions and uses her estimates in an affine incentive rule. The follower knows the leader's learning policy, which incorporates past observations to produce estimates of the follower's type, and can deviate from his myopic best response to influence the leader's future estimates. We ask how the follower's ability to manipulate these observations affects the leader's performance. Under a manipulation budget $C_T$ constraining the expected cumulative squared deviation of the follower's actions from his myopic best response, we establish a minimax regret rate of order $\max\{\log T, \min\{C_T,T\}\}$ for approximately rational followers. Our information-theoretic lower bound applies to a general class of causal estimation and manipulation policies for the leader and the follower. We provide a simp
    
[^19]: 关于线性神经网络的损失景观：超越临界点

    On the loss landscape of the linear neural network: beyond critical points

    [https://arxiv.org/abs/2611.26671](https://arxiv.org/abs/2611.26671)

    本文提出了一种包含线性神经网络损失景观中所有临界点的中间几何结构——“临界谷线”，它提供了可二维可视化的损失景观全局描述，并表明梯度下降会被该结构吸引从而沿其持续降低损失至全局最小值。

    

    自Baldi和Hornik在20世纪80年代末的开创性工作以来，线性神经网络损失景观中的临界点已被广泛研究，但在中间尺度或全局尺度上对这一景观的描述仍然缺失。这种表征至关重要，尤其是因为大量文献表明梯度下降会绕过临界点。本文引入了一种中间结构，称为临界谷线，它包含所有临界点并展现出关键的几何性质。除了在比现有文献更简单的假设下为Baldi和Hornik的结果提供了几何证明之外，该框架还提供了损失景观的全局视角，并且可以在二维平面上轻松可视化。经验观察表明，梯度下降会被吸引向临界谷线，沿着该谷线，损失持续下降直至其全局最小值。

    arXiv:2611.26671v1 Announce Type: cross  Abstract: While the critical points of the loss landscape in linear neural networks have been extensively studied since the seminal work of Baldi \& Hornik in the late 1980s, a description of this landscape at an intermediate or global scale is still lacking. Such a characterization is crucial, particularly as a substantial body of literature indicates that gradient descent circumvents critical points. The present article introduces an intermediate structure, termed the critical thalweg, which encompasses all critical points and exhibits key geometric properties. Beyond providing a geometric proof of Baldi \& Hornik's results using simpler assumptions than the existing literature, this framework offers a global view of the loss landscape that can be easily visualized in a two-dimensional plane. Empirical observations suggest that gradient descent is attracted toward the critical thalweg, along which the loss continues to decrease to its global m
    
[^20]: SDG：面向线性注意力的分块边界状态依赖门控

    SDG: State-Dependent Gating at Chunk Boundaries for Linear Attention

    [https://arxiv.org/abs/2611.26637](https://arxiv.org/abs/2611.26637)

    提出SDG方法，通过在分块边界处依据记忆状态的变化计算逐元素门控，为线性注意力模型提供状态反馈的遗忘机制，同时保持块内并行计算的高效性。

    

    近期的门控线性注意力模型使用状态无关的门控机制，该门控由token表示计算得出，而不直接读取循环记忆。这种设计支持高效的分块计算：序列被划分为连续token组成的块，并在每个块内进行并行计算。然而，即使层中存储的信息不同，相同的token表示仍会产生相同的门控。我们提出了状态依赖门控（SDG），它根据记忆在一个块上的变化计算逐元素门控，并将其应用于块的边界。这在保持每个块内并行计算的同时，为遗忘机制提供了状态反馈。我们为Gated DeltaNet、Kimi Delta Attention、Gated DeltaNet-2和Mamba-3开发了融合内核。我们的理论分析在信号-噪声模型下量化了状态依赖的逐元素门控在重建方面的潜在收益。我们还表明，边界衰减（摘要在此处被截断）

    arXiv:2611.26637v1 Announce Type: cross  Abstract: Recent gated linear attention models use state-independent gates, computed from token representations without directly reading the recurrent memory. This design supports efficient chunkwise computation: sequences are divided into chunks of consecutive tokens, with parallel computation within each chunk. However, the same token representation produces the same gate, even when the layer stores different information. We introduce State-Dependent Gating (SDG), which computes element-wise gates from the change in memory over a chunk and applies them at the chunk boundary. This provides state feedback for forgetting while preserving parallel computation within each chunk. We develop fused kernels for Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2, and Mamba-3. Our theoretical analysis quantifies the potential reconstruction benefits of state-dependent, element-wise gating under a signal--noise model. We also show that boundary decay 
    
[^21]: 用于安全控制器选择的无抵消约束SquareCB算法

    Cancellation-Free Constrained SquareCB for Safe Controller Selection

    [https://arxiv.org/abs/2611.21880](https://arxiv.org/abs/2611.21880)

    该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。

    

    我们研究约束上下文老虎机问题，其中学习者在每一轮从 $K$ 个动作中选择一个，以最小化损失，同时使其动作的期望成本保持在预算以下。现有的到在线回归的归约方法使用从过去约束残差中学习到的拉格朗日乘子对成本进行加权，因此某些轮次的松弛量可能会抵消其他轮次的违约。我们转而在成本上施加修正权重来运行 Foster 等人（2020）提出的 SquareCB：当估计成本在预算之内时权重为零，否则权重恰好足够大以将成本拉回预算。在可实现性假设以及具有边际 $\delta$ 的 Slater 条件下，该算法达到遗憾 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 和无抵消违约 $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，其中 $\mathrm{Reg}_{\mathsf{Sq}}$ 是回归预言机的误差。当预算必须在（原文摘要在此处截断，后续内容缺失）……

    arXiv:2611.21880v1 Announce Type: cross  Abstract: We study constrained contextual bandits, in which a learner chooses one of $K$ actions in each round to minimize loss while keeping the expected cost of its play below a budget. Existing reductions to online regression weight the cost by a Lagrange multiplier learned from past constraint residuals, so slack in some rounds can offset violations in others. We instead run SquareCB proposed in \citep{foster2020beyond} with a rectified weight on the cost: it is zero when the estimated cost is within the budget, and otherwise just large enough to bring it back to the budget. Under realizability and Slater's condition with margin $\delta$, the algorithm attains regret $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ and cancellation-free violation $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$, where $\mathrm{Reg}_{\mathsf{Sq}}$ is the error of the regression oracle. The same bounds hold when the budget must be met in
    
[^22]: 学习顺序能否解释遗忘？李括号作为无需旧数据的灾难性遗忘早期预警信号

    Does Learning Order Explain Forgetting? The Lie Bracket as an Early-Warning Signal for Catastrophic Forgetting Without Old Data

    [https://arxiv.org/abs/2611.20087](https://arxiv.org/abs/2611.20087)

    本文实证验证了两个任务梯度场的李括号决定了训练顺序导致的权重差异并与灾难性遗忘相关，且仅凭旧任务训练结束时存储的梯度和前20个Hessian特征对，就能在无需旧数据的条件下早期预测遗忘，效果优于EWC惩罚项。

    

    先在任务A上训练、再在任务B上训练的网络，其最终权重与先训练B再训练A的网络不同。在主导阶上，这一差距由两个任务梯度场的李括号（Lie bracket）决定，Dherin (2023) 猜想非零的李括号可能与灾难性遗忘相关。本文对这一猜想进行了实证检验。在Split MNIST、Split Fashion-MNIST 和 Split CIFAR-10 上，任务A的遗忘几乎恰好等于两种训练顺序下A的损失之差（Spearman rho = 0.96-1.00）。随后，作者进一步探究李括号能否在任务B训练早期、且不使用A的任何数据的情况下预测遗忘——仅利用A训练结束时存储的小型摘要信息：其梯度及其前20个Hessian特征对。当旧任务已被充分学习且遗忘较慢（任务增量式学习）时，这个存储的李括号对遗忘的排序相关系数达到 rho = 0.70-0.81，而 Elastic Weight Consolidation（EWC）惩罚项仅为 0.33-0.58，并且匹配或超过（原文在此处截断）。

    arXiv:2611.20087v1 Announce Type: new  Abstract: A network trained on task A and then task B ends at different weights than one trained on B and then A. To leading order this gap is set by the Lie bracket of the two tasks' gradient fields, and Dherin (2023) conjectured that a non-vanishing bracket may be linked to catastrophic forgetting. I test this conjecture empirically. Across Split MNIST, Split Fashion-MNIST and Split CIFAR-10, the forgetting of A is almost exactly the difference in A's loss between the two training orders (Spearman rho = 0.96-1.00). I then ask whether the bracket can predict that forgetting early in training on B and without any of A's data, using only a small summary stored when A finishes: its gradient and its top 20 Hessian eigenpairs. When the old task is well learned and forgetting is slow (task-incremental), this stored bracket ranks forgetting with rho = 0.70-0.81, against 0.33-0.58 for the penalty of Elastic Weight Consolidation, and matches or exceeds re
    
[^23]: 运行时混淆下的强化学习：策略学习与离线策略评估

    Reinforcement Learning Under Runtime Confounding: Policy Learning and Off-Policy Evaluation

    [https://arxiv.org/abs/2610.40056](https://arxiv.org/abs/2610.40056)

    该论文针对强化学习中部署时关键变量不可观测的“运行时混淆”问题，提出了相应的策略学习算法和半参数高效的离线策略评估方法，以解决混淆随时间传播所导致的学习偏差。

    

    强化学习越来越多地被用于在医疗保健、电子商务和机器人等领域训练鲁棒、可适应的系统。在这些场景中，由于法律、伦理或后勤方面的限制，某些驱动决策的关键变量可能在部署时无法被收集。这就产生了一种被称为“运行时混淆”的设定：记录的数据具有完全可观测性，但部署时缺乏对某些变量（即运行时混淆变量）的了解。与单时间点的非自适应设定不同，强化学习下的运行时混淆有可能随时间传播，因为早期决策会影响未来的状态、观测和动作分配，从而导致有偏的学习。我们通过三项贡献来解决这些问题：(1) 在运行时混淆条件下开发策略学习算法，(2) 开发半参数有效的离线策略估计方法，(3) 提供诊断……

    arXiv:2610.40056v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) is increasingly used to train robust, adaptable systems across healthcare, e-commerce, and robotics. In these settings, legal, ethical, or logistical concerns may preclude certain key variables that drive decisions from being collected during deployment time. This gives rise to a setting known as runtime confounding: the logged data hold full observability, but deployment time lacks knowledge of certain variables, known as runtime confounders. Unlike single timepoint and non-adaptive settings, runtime confounding under reinforcement learning has the potential to propagate over time, as earlier decisions affect future states, observations, and action assignments, leading to biased learning. We address these problems with three contributions: we (1) develop policy learning algorithms under runtime-confounding, (2) develop semi-parametric efficient off-policy estimation procedures, and (3) provide diagn
    
[^24]: CoCoNet：用于多维不确定性区域的凸保形网络

    CoCoNet: Convex Conformal Network for Multi-Dimensional Uncertainty Regions

    [https://arxiv.org/abs/2610.24246](https://arxiv.org/abs/2610.24246)

    提出 CoCoNet，通过端到端学习依赖输入的凸一致性分数并直接最小化其子水平集大小，生成灵活数据自适应的凸预测区域，同时借助分割保形校准保留有限样本覆盖率保证。

    

    保形预测（CP）在可交换性假设下提供无分布的覆盖率保证，但其预测区域的效率在很大程度上取决于一致性分数的选择。这一点在多变量回归中尤为重要，因为输出维度之间的依赖关系会产生复杂的联合不确定性结构。本文提出了 CoCoNet（Convex Conformal Network，凸保形网络），用于学习能够产生灵活且数据自适应预测区域的一致性分数。其核心机制是一个端到端学习的、依赖输入的凸分数：CoCoNet 通过直接最小化该分数子水平集的大小，在留出数据上进行训练，同时强制其相对于残差保持凸性，从而使每个预测区域天然为凸。随后，CoCoNet 应用分割保形校准以保留有限样本覆盖率保证。论文提出了该方法的两个变体：1）CoCoNet-Spline，基于

    arXiv:2610.24246v3 Announce Type: replace-cross  Abstract: Conformal prediction (CP) provides distribution-free coverage guarantees under exchangeability, but the efficiency of its prediction regions depends critically on the choice of the conformity score. This is particularly important in multivariate regression, where dependencies across output dimensions induce complex joint uncertainty structures. This paper introduces CoCoNet (Convex Conformal Network) for learning conformity scores that induce flexible and data-adaptive prediction regions. The main mechanism is an input-dependent convex score learned end-to-end: CoCoNet trains the score on held-out data by directly minimizing the size of its sublevel sets, while enforcing convexity with respect to the residual so that every prediction region is inherently convex. CoCoNet then applies split-conformal calibration to retain finite-sample coverage guarantees. The paper presents two variants of the method: 1) CoCoNet-Spline, based on
    
[^25]: 持续学习中LoRA的动力学理论

    A Dynamical Theory of LoRA in Continual Learning

    [https://arxiv.org/abs/2609.39367](https://arxiv.org/abs/2609.39367)

    该论文在师生模型中建立了持续学习下LoRA的精确动力学理论，揭示低秩更新虽能减少对已学任务特征的干扰从而缓解灾难性遗忘，但其初始化会减慢对新任务的适应。

    

    尽管低秩适配被广泛使用，但人们对其在持续学习中的动力学行为，以及低秩更新影响灾难性遗忘的机制，仍然知之甚少。我们在一个可解的双任务师生模型中，为LoRA提供了渐近精确的动力学刻画。在高维在线学习极限下，我们针对有限个宏观序参量导出了一个封闭的常微分方程组，从而在整个初始任务1学习阶段以及随后在任务2上的LoRA微调过程中，都得到了泛化误差的精确表达式。该理论与有限维模拟定量吻合，并揭示了LoRA的两个特征效应：低秩适配减少了与任务1所学特征的干扰，但其初始化会减慢对任务2的适应。基于这一机理图景，我们进一步分析了一种状态相关的掩蔽方法……

    arXiv:2609.39367v1 Announce Type: new  Abstract: Despite the widespread use of Low-Rank Adaptation (LoRA), little is known about its dynamics in continual learning and the mechanisms by which low-rank updates affect catastrophic forgetting. We provide an asymptotically exact dynamical characterization of LoRA in a solvable two-task teacher-student model. In the high-dimensional online-learning limit, we derive a closed system of ordinary differential equations for a finite set of macroscopic order parameters, yielding exact expressions for the generalization errors throughout both the initial Task 1 learning phase and the subsequent LoRA fine-tuning on Task 2. The theory quantitatively matches finite-dimensional simulations and exposes two characteristic effects of LoRA: low-rank adaptation reduces interference with features learned on the first task, but its initialization slows adaptation to the second task. Building on this mechanistic picture, we analyze a state-dependent masking s
    
[^26]: AECSF：用于高维非线性数据同化的自适应集成条件得分滤波

    AECSF: Adaptive Ensemble Conditional Score Filtering for High-Dimensional Nonlinear Data Assimilation

    [https://arxiv.org/abs/2609.32411](https://arxiv.org/abs/2609.32411)

    提出了一种免训练的自适应集成条件得分滤波器AECSF，利用条件Tweedie恒等式构建解析可处理的得分估计器，从而在高维非线性数据同化中同时避免粒子权重退化并捕捉非高斯后验结构。

    

    高维非线性动力系统的贝叶斯状态估计面临着统计精度与计算可行性之间的根本性矛盾：粒子权重可能发生退化坍缩，而高斯集成更新则可能遗漏非高斯的后验结构。基于得分的扩散滤波器提供了一种基于采样的替代方案，但现有的免训练得分滤波器通常依赖启发式的似然修正，由于忽略了与每个含噪反向粒子相关联的系统状态的不确定性，这可能会损害后验估计的精度。为解决这些问题，我们提出了AECSF——一种免训练的自适应集成条件得分滤波器。AECSF基于条件Tweedie恒等式构建了一个解析可处理的得分估计器，将含噪后验得分估计问题转化为在给定含噪反向粒子与观测条件下估计系统状态的条件均值。为估计这些条件……（摘要原文在此处截断）

    arXiv:2609.32411v2 Announce Type: replace-cross  Abstract: Bayesian state estimation for high-dimensional nonlinear dynamical systems entails a fundamental tension between statistical fidelity and computational tractability, as particle weights can collapse, while Gaussian ensemble updates can miss non-Gaussian posterior structure. Score-based diffusion filters offer a sampling-based alternative, but existing training-free score filters often rely on heuristic likelihood corrections, which can compromise posterior accuracy by neglecting uncertainty about the system state associated with each noisy reverse particle. To address these issues, we propose AECSF, a training-free adaptive ensemble conditional score filter. AECSF constructs an analytically tractable score estimator from the conditional Tweedie identity, which recasts noisy posterior score estimation as estimating the conditional mean of the system state given a noisy reverse particle and the observation. To estimate these cond
    
[^27]: 一种用于Metropolis调整Dikin游走的一般框架：多胞体上的维数平方混合与谱面体上的对数行列式游走

    A General Framework for Metropolis-Adjusted Dikin Walks: Dimension-Square Mixing on Polytopes and Log-Det Walks on Spectrahedra

    [https://arxiv.org/abs/2608.25273](https://arxiv.org/abs/2608.25273)

    本文提出了一种统一框架，通过精确度量的Metropolis调整Dikin游走，分别在多胞体和谱面体上实现了维数平方混合，并利用提议比较参数将分析简化，显著提升了混合速度的上界。

    

    我们通过将提议行列式与逆二次型保持在一起，分析了精确度量的Metropolis调整Dikin游走。在完整的对数接受比率中，它们的前导非中心项相互抵消，留下可通过二阶工具控制的中心化波动。对于由$n$个不等式和凸$L$-Lipschitz势给出的多胞体，这为正则化Lee--Sidford游走提供了在$\widetilde O((d^{2}+dL^{2}R^{2})\log(w/\delta))$步内的暖启动混合。对于具有$n\times n$块的谱面体，对数行列式游走在$\widetilde O((\psi^\star nd+dL^{2}R^{2})\log(w/\delta))$步内混合，其中$\psi^\star$衡量矩阵杠杆。这两个分析共享一个从接受到混合的简化。一个提议比较参数将多胞体界限转移到由高精度Lewis权重计算的适当填充的$O(1/d)$-精确度量上。对于谱面体，给定$\widehat\psi\ge\psi^\star$，

    arXiv:2608.25273v1 Announce Type: cross  Abstract: We analyze exact-metric, Metropolis-adjusted Dikin walks by keeping the proposal determinant and reverse quadratic form together. Their leading uncentered terms cancel in the complete logarithmic acceptance ratio, leaving centered fluctuations that can be controlled with second-order tools. For a polytope given by $n$ inequalities and a convex $L$-Lipschitz potential, this yields warm-start mixing in $\widetilde O((d^{2}+dL^{2}R^{2})\log(w/\delta))$ steps for the regularized Lee--Sidford walk. For a spectrahedron with $n\times n$ blocks, the log-det walk mixes in $\widetilde O((\psi^\star nd+dL^{2}R^{2})\log(w/\delta))$ steps, where $\psi^\star$ measures matrix leverage.   The two analyses share an acceptance-to-mixing reduction. A proposal-comparison argument transfers the polytope bound to an appropriately padded $O(1/d)$-accurate metric computed from high-precision Lewis weights. For spectrahedra, given $\widehat\psi\ge\psi^\star$, 
    
[^28]: COMPACT：基于穷举且非冗余不变关系集合的谱调整得分

    COMPACT: Spectral Adjustment Scores from an Exhaustive and Nonredundant Set of Invariant Relations

    [https://arxiv.org/abs/2608.10305](https://arxiv.org/abs/2608.10305)

    该论文提出COMPACT方法，通过穷举检验候选得分、处理与结局之间（允许潜在变量）的所有因果图，识别出在处理到结局的边被添加或移除时保持不变的无条件与条件依赖关系的穷举且非冗余集合，从而构建谱调整得分以辅助因果调整变量选择。

    

    观测数据集通常包含大量基线变量，然而在估计因果效应时，研究者可能并不知道哪些变量应纳入调整集合。混杂信息也可能微弱地分布在多个变量之中。倾向得分可以通过在二值处理下将高维协变量缩减为一个标量来简化调整过程。尽管倾向得分是最粗的平衡得分，但这一平衡性质并不一定能够区分与调整相关的因果结构。我们转而检验候选得分、处理和结局之间的所有因果图，同时允许存在潜在变量。在忠实性假设下，我们识别出所有在添加或移除处理到结局的边时保持不变的无条件与条件依赖关系，并将处理效应估计留给下游分析。这些关系共同定义了最小兼容的……

    arXiv:2608.10305v2 Announce Type: replace-cross  Abstract: Observational datasets frequently contain many baseline variables, yet investigators estimating causal effects may not know which variables to include in the adjustment set. Confounding information may also be distributed weakly across many variables. Propensity scores can simplify adjustment by reducing high-dimensional covariates to a scalar with binary treatment. Although the propensity score is the coarsest balancing score, this balancing property does not necessarily distinguish among causal structures relevant to adjustment. We instead examine all causal graphs among a candidate score, treatment, and outcome while allowing latent variables. Under faithfulness, we identify all unconditional and conditional dependence relations that remain unchanged when the treatment-to-outcome edge is added or removed, leaving treatment-effect estimation to the downstream analysis. Together, these relations define the smallest compatible 
    
[^29]: 微分方程高斯过程逼近的统一视角

    A unified perspective of Gaussian process approximation for differential equations

    [https://arxiv.org/abs/2607.06292](https://arxiv.org/abs/2607.06292)

    该论文提出了一个统一的贝叶斯视角，通过导数匹配方式将微分方程约束纳入似然函数，把碎片化的高斯过程微分方程数值方法整合进一个共同的概率框架，并同时支持参数估计与解逼近。

    

    利用高斯过程逼近微分方程的应用正在迅速扩展，催生了数量不断增长、种类多样但较为碎片化的数值方法体系。我们提出了一个统一的贝叶斯视角，基于一种“导数匹配”的解释方式，将微分方程约束纳入似然函数，从而把这些技术置于一个共同的概率框架之中。这一统一视角同时支持参数估计和解的逼近，并展示了如何在该框架内理解一系列现有方法。本工作旨在整合当前的研究进展，并为未来的研究提供基础。

    arXiv:2607.06292v2 Announce Type: replace-cross  Abstract: The use of Gaussian processes for approximating differential equations has expanded rapidly, leading to a growing, diverse, and fragmented body of numerical methods. We present a unified Bayesian perspective that places these techniques within a common probabilistic framework, based on a derivative matching interpretation for incorporating differential equation constraints into likelihood. This unified perspective supports both parameter estimation and solution approximation, and shows how a range of existing methods can be understood within it. This work aims to consolidate current developments and provide a foundation for future research.
    
[^30]: 一种深度零膨胀模型用于美国东海岸北大西洋露脊鲸存在性建模以支持蓝色经济管理

    A Deep Zero-Inflated Model of North Atlantic Right Whale Presence To Support Blue Economy Management in the U.S. East Coast

    [https://arxiv.org/abs/2606.14403](https://arxiv.org/abs/2606.14403)

    本文提出了一种深度零膨胀伯努利模型，联合建模物种存在性与检测概率，有效处理被动声学监测数据中的零膨胀和复杂依赖，为濒危物种保护与蓝色经济管理提供新工具。

    

    arXiv:2606.14403v2 公告类型：替换-交叉  摘要：对濒危海洋哺乳动物物种（如北大西洋露脊鲸）的有效建模，对于在日益增长的蓝色经济中平衡海洋保护至关重要。由自主水下航行器收集的被动声学监测数据为局部海洋物种检测和海洋学感知提供了新机会，但也引入了复杂的统计挑战，如零膨胀、不完全检测和复杂依赖结构。为此，我们提出了深度零膨胀伯努利（DeepZIB）模型——一种深度统计方法，该方法联合建模潜在物种存在性和条件检测概率，同时从异构协变量信息中学习复杂的栖息地关系。我们建立了模型结构性质的理论结果，并进行了模拟实验以证明其恢复底层参数和潜在存在场的能力。应用...

    arXiv:2606.14403v2 Announce Type: replace-cross  Abstract: Effective modeling of endangered marine mammal species, such as the North Atlantic Right Whale, is critical for balancing marine conservation with the growing blue economy. Passive acoustic monitoring data collected by autonomous underwater vehicles provide new opportunities for localized marine species detection and oceanographic sensing, but introduce complex statistical challenges such as zero inflation, imperfect detection, and intricate dependence structures. In response, we propose the Deep Zero-Inflated Bernoulli (DeepZIB) model--a deep statistical method which jointly models latent species presence and conditional detection probabilities while learning complex habitat relationships from heterogeneous covariate information. We establish theoretical results on the model's structural properties and conduct simulation experiments to demonstrate its ability to recover underlying parameters and latent presence fields. Applica
    
[^31]: 恢复住房市场中环境舒适品的直接价格效应：基于经验蒙特卡洛模拟的回归与因果机器学习模型评估

    Recovering Direct Price Effects of Environmental Amenities in Housing Markets: Regression and Causal Machine Learning Model Assessment with Empirical Monte Carlo Simulation

    [https://arxiv.org/abs/2606.02795](https://arxiv.org/abs/2606.02795)

    该研究利用基于纽约州百万房产交易真实数据生成过程的经验蒙特卡洛模拟，系统评估了回归与因果机器学习方法在估计环境舒适品直接价格效应上的表现，发现广义双重差分回归方法表现最为稳健。

    

    特征价格（享乐价格）模型被广泛用于评估环境舒适品的价值，但关于如何估计直接且未受混淆的价格效应的实用指导仍然稀少。我们开展了一项经验蒙特卡洛模拟，以评估传统回归方法与因果机器学习方法在估计空间划定舒适品对受处理房产的直接非中介价格效应（DUET）方面的表现——DUET是福利变化的一个下界近似，可直接应用于成本效益分析。与以往依赖参数假设的模拟不同，我们保留了纽约州北部超过100万笔房产交易（1990–2024年）背后的真实数据生成过程，并在各次迭代中随机分配处理位置，从而建立一个真值基准，据此可以精确测量估计误差。结果表明，广义双重差分回归始终优于基线双重差分和双向固定效应等方法。

    arXiv:2606.02795v2 Announce Type: replace  Abstract: Hedonic price models are widely used to value environmental amenities, yet practical guidance for estimating direct unconfounded price effects remains sparse. We conduct an empirical Monte Carlo simulation to evaluate traditional regression and causal machine learning (CML) approaches for estimating the direct unmediated price effect of spatially delineated amenities on treated properties (DUET), a lower-bound approximation of welfare change with direct applications to benefit-cost analysis. Whereas previous simulations rely on parametric assumptions, we preserve the actual data-generating process underlying over 1 million property transactions from upstate New York (1990--2024) and randomly assign treatment locations across iterations, establishing a ground truth against which estimation error can be measured precisely. Generalized difference-in-differences (DID) regression consistently outperforms baseline DID and two-way fixed eff
    
[^32]: 用于诊断Transformer权重分布的双参数Weibull框架

    A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions

    [https://arxiv.org/abs/2605.18898](https://arxiv.org/abs/2605.18898)

    该论文提出用双参数Weibull分布作为统一参考框架来诊断Transformer权重幅值分布，发现跨七个模型系列的FFN权重形状参数高度稳定（约1.20），而查询/键投影则呈现与架构相关的系统性偏离。

    

    Transformer检查点包含数十亿个权重，但其逐元素分布如何随组件、模型和训练而变化仍难以刻画。我们在固定的中间80%协议下，用双参数Weibull参考模型拟合每个受分析权重矩阵的幅值分布。对于独立同分布的高斯初始化，对应的半正态目标经协议拟合得到的形状参数为 $k_0 \approx 1.20$。与gamma分布和对数正态分布的匹配比较表明，Weibull分布对大多数组件组更为适用；对于单独存储的MHA查询/键投影，支持度则不够明确，且没有哪个被测试的单一分布族能干净地描述带有次级权重群体的矩阵。在跨越七个系列的12个模型条目中，终端FFN形状参数中位数落在很窄的区间内，$k \in [1.186, 1.204]$（变异系数 CV = 0.51%），而查询/键投影则表现出与架构相关的偏离。这种稳定性仅在宏观层面成立：在五个Pyth（摘要在此处被截断）

    arXiv:2605.18898v2 Announce Type: replace  Abstract: Transformer checkpoints contain billions of weights, yet how their element-wise distributions vary across components, models, and training is hard to characterize. We fit the magnitude distribution of each analyzed weight matrix with a two-parameter Weibull reference model under a fixed middle-80% protocol. For i.i.d. Gaussian initialization, the corresponding half-Normal target has protocol-fitted shape $k_0 \approx 1.20$. Matched comparisons with gamma and lognormal favor Weibull for most component groups; support is less decisive for separately stored MHA query/key projections, and no tested single family cleanly describes matrices carrying a secondary weight population.   Across 12 model entries spanning seven families, terminal FFN shape medians fall in a narrow band, $k \in [1.186, 1.204]$ (CV $= 0.51\%$), whereas query/key projections show architecture-associated departures. This stability is macroscopic only: in the five Pyth
    
[^33]: 通过专家损失集成实现时间序列预测混合专家模型的快速训练

    Fast Training of Mixture-of-Experts for Time Series Forecasting via Expert Loss Integration

    [https://arxiv.org/abs/2605.10330](https://arxiv.org/abs/2605.10330)

    提出一种融合专家特定损失与部分在线学习策略的自适应混合专家框架，解决了小门控权重导致的优化难题，在保持计算效率的同时显著提升了时间序列预测性能。

    

    我们提出了一种新颖的自适应混合专家框架用于时间序列预测，该框架通过引入专家特定损失来解决由小门控权重引起的优化问题，专家特定损失为每个专家提供独立于门控分配权重的直接学习信号。具体而言，总体目标由基础预测损失和专家特定损失组成，使各个专家的预测误差能够与总体预测误差一起直接影响参数更新。该框架还鼓励不同的专家从数据的不同时间片段中学习。所提出的框架进一步与部分在线学习策略相结合，实现模型参数的高效增量更新。通过将专家级损失信息与部分在线优化相结合，所提出的方法在保持计算效率的同时提高了预测性能。

    arXiv:2605.10330v2 Announce Type: replace-cross  Abstract: We propose a novel adaptive Mixture-of-Experts (MoE) framework for time series forecasting that addresses the optimization problem arising from small gating weights by incorporating expert-specific losses, which provide each expert with a direct learning signal independent of the gate-assigned weight. Specifically, the overall objective comprises the base forecasting loss and expert-specific losses, allowing individual expert prediction errors to directly influence parameter updates alongside the aggregate forecasting error. The framework also encourages different experts to learn from different temporal segments of the data. The proposed framework is further combined with a partial online learning strategy that enables efficient incremental updates of model parameters. By integrating expert-level loss information with partial online optimization, the proposed method improves forecasting performance while retaining computationa
    
[^34]: 基于代数谱曲线的自由解压缩

    Free Decompression with Algebraic Spectral Curves

    [https://arxiv.org/abs/2605.03634](https://arxiv.org/abs/2605.03634)

    本文利用代数谱曲线理论，为Stieltjes变换满足代数关系的谱密度建立了更通用的自由解压缩方法，放宽了原有方法的强假设，使跨矩阵规模的谱信息外推能够应用于更现实的机器学习模型。

    

    arXiv:2605.03634v2 公告类型：replace-cross 摘要：来自随机矩阵理论的工具已成为深度学习理论的核心，它们利用谱信息为泛化、鲁棒性、缩放和失效模式等建模提供机制。尽管这些工具通常能够刻画经验行为，但实际计算受矩阵规模的限制，往往只能局限于过小而不具现实性的模型。这促使人们从较小模型的行为来推断较大模型的性质。自由解压缩（FD）是最近提出的一种跨矩阵规模外推谱信息的方法，但其效用目前受到强假设的约束，这些假设使其无法应用于更现实的机器学习（ML）模型。我们利用代数谱曲线理论，为Stieltjes变换满足代数关系的谱密度提供了一种通用的自由解压缩方法学——这一建模假设在实践中更有可能成立。

    arXiv:2605.03634v2 Announce Type: replace-cross  Abstract: Tools from random matrix theory have become central to deep learning theory, using spectral information to provide mechanisms for modeling generalization, robustness, scaling, and failure modes. While often capable of modeling empirical behavior, practical computations are limited by matrix size, often imposing a restriction to models that are too small to be realistic. This motivates the inference of properties of larger models from the behavior of smaller ones. Free decompression (FD) is a recently proposed method for extrapolating spectral information across matrix sizes, but its utility is currently limited by strong assumptions that preclude its implementation on more realistic machine learning (ML) models. We use algebraic spectral curve theory to provide a general FD methodology for spectral densities whose Stieltjes transform satisfies an algebraic relation, a modeling assumption that is more likely to hold in practice.
    
[^35]: 语言扩散模型是能够检索未见数据的联想记忆

    Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data

    [https://arxiv.org/abs/2604.26841](https://arxiv.org/abs/2604.26841)

    该论文证明均匀离散扩散语言模型本质上是联想记忆，其吸引盆可通过条件似然最大化而非显式能量函数形成，并揭示了由数据规模支配的从记忆到泛化的急剧转变，使其能够检索未见过的数据。

    

    语言扩散模型何时会记忆其训练数据，以及如何定量评估其真正的生成机制？我们通过证明基于均匀分布的离散扩散模型（UDDMs）在根本上表现为具有涌现创造能力的联想记忆（AMs）来回答这些问题。联想记忆的核心思想是通过在存储的数据点周围建立独特的吸引盆，从而可靠地将这些数据点作为“记忆”恢复出来。历史上，像Hopfield网络这样的模型使用显式的能量函数来保证这些稳定吸引子的存在。我们拓展了这一视角，利用了一个关键观察：能量并非严格必需，因为吸引盆也可以通过条件似然最大化来形成。通过评估模型对训练样本和测试样本的词元恢复能力，我们在UDDMs中识别出一个由训练规模大小所支配的、从记忆到泛化的急剧转变（摘要在此处截断）。

    arXiv:2604.26841v2 Announce Type: replace-cross  Abstract: When do language diffusion models memorize their training data, and how to quantitatively assess their true generative regime? We address these questions by showing that Uniform-based Discrete Diffusion Models (UDDMs) fundamentally behave as Associative Memories (AMs) $\textit{with emergent creative capabilities}$. The core idea of an AM is to reliably recover stored data points as $\textit{memories}$ by establishing distinct basins of attraction around them. Historically, models like Hopfield networks use an explicit energy function to guarantee these stable attractors. We broaden this perspective by leveraging the observation that energy is not strictly necessary, as basins of attraction can also be formed via conditional likelihood maximization. By evaluating token recovery of $\textit{training}$ and $\textit{test}$ examples, we identify in UDDMs a sharp memorization-to-generalization transition governed by the size of the t
    
[^36]: 基于组合满意化老虎机的多用户毫米波波束与速率自适应

    Multi-User mmWave Beam and Rate Adaptation via Combinatorial Satisficing Bandits

    [https://arxiv.org/abs/2604.14908](https://arxiv.org/abs/2604.14908)

    本文提出SAT-CTS轻量级策略，将多用户毫米波系统中的波束与速率联合自适应建模为满意化目标的组合半老虎机问题，并首次给出了此类问题的有限时间遗憾界理论保证。

    

    我们研究了多用户毫米波MISO系统中的下行链路波束与速率自适应问题，其中多个基站（BS）各自使用有限码本中的模拟波束赋形，以每个用户设备（UE）唯一的波束和离散的数据传输速率来服务多个单天线用户设备。基站基于ACK/NACK反馈学习传输是否成功。为了刻画服务目标，我们引入了满意化吞吐量阈值 $\tau_r$，并将波束与速率的联合自适应建模为波束-速率元组上的组合半老虎机问题。在该框架下，我们提出了SAT-CTS，这是一种轻量级、阈值感知的策略，它将保守的置信估计与后验采样相结合，引导学习朝着满足 $\tau_r$ 的方向发展，而非仅仅追求最大化。我们的主要理论贡献在于为具有满意化目标的组合半老虎机提供了首个有限时间遗憾界：当 $\tau_r$ 可实现时，我们对累积遗憾（摘要在此处截断）……

    arXiv:2604.14908v2 Announce Type: replace-cross  Abstract: We study downlink beam and rate adaptation in a multi-user mmWave MISO system where multiple base stations (BSs), each using analog beamforming from finite codebooks, serve multiple single-antenna user equipments (UEs) with a unique beam per UE and discrete data transmission rates. BSs learn about transmission success based on ACK/NACK feedback. To encode service goals, we introduce a satisficing throughput threshold $\tau_r$ and cast joint beam and rate adaptation as a combinatorial semi-bandit over beam-rate tuples. Within this framework, we propose SAT-CTS, a lightweight, threshold-aware policy that blends conservative confidence estimates with posterior sampling, steering learning toward meeting $\tau_r$ rather than merely maximizing. Our main theoretical contribution provides the first finite-time regret bounds for combinatorial semi-bandits with satisficing objective: when $\tau_r$ is realizable, we upper bound the cumula
    
[^37]: 这个梯度能带来喜悦吗？

    Does This Gradient Spark Joy?

    [https://arxiv.org/abs/2603.20526](https://arxiv.org/abs/2603.20526)

    论文提出愉悦策略梯度（DG），在前向传播中用优势与惊讶度的乘积“愉悦度”评估样本的学习价值，并通过“近藤门控”只为值得的样本执行昂贵的反向传播，从而以极少的计算实现更快的学习。

    

    策略梯度方法为每个样本都要执行一次反向传播，然而反向传播代价高昂，且大多数样本几乎没有学习价值。愉悦策略梯度在前向传播中就用“愉悦度”来评估样本的学习价值，愉悦度是优势与惊讶度（负对数概率）的乘积。我们提出了“近藤门控”，它将愉悦度与一个计算价格进行比较，只有当样本值得时才为其支付反向传播的开销；通过调节该价格可以刻画出质量与成本之间的权衡曲线。在MNIST上，即使反向传播完全免费，仅保留3%的梯度项也比标准策略梯度学习得更快，并且几乎能达到完整DG的效果。在老虎机任务中，零价格门控能够保持期望梯度方向并消除其垂直方向上的噪声；与价值和惊讶度的加性混合相比，愉悦度对样本的排序更加可靠；而具有高奖励方差的罕见动作则揭示了愉悦度失效的场景。在Transformer的词元反……（原文摘要在此截断）

    arXiv:2603.20526v2 Announce Type: replace-cross  Abstract: Policy gradient computes a backward pass for every sample, although the backward pass is expensive and most samples carry little learning value. The Delightful Policy Gradient (DG) scores learning value in the forward pass with \emph{delight}, the product of advantage and surprisal (negative log-probability). We introduce the \emph{Kondo gate}, which compares delight against a compute price and pays for a backward pass only when the sample is worth it; sweeping the price traces a quality--cost trade-off. On MNIST, keeping 3\% of gradient terms learns faster than standard policy gradient even when backward passes are free, and nearly matches full DG. In bandits, zero-price gating preserves the expected gradient direction and removes its perpendicular noise, delight ranks samples more reliably than additive mixes of value and surprise, and a rare action with high reward variance marks where delight fails. On transformer token rev
    
[^38]: 分布式摩擦下的愉悦策略梯度

    Delightful Policy Gradient under Distributed Friction

    [https://arxiv.org/abs/2603.20521](https://arxiv.org/abs/2603.20521)

    本文提出愉悦策略梯度（DG），通过用优势与惊讶度的乘积作为门控信号，在不依赖行为概率的情况下抑制分布式强化学习中罕见失败带来的噪声，同时保留罕见成功所携带的关键信息。

    

    分布式强化学习的训练数据来自过时、有缺陷或策略不匹配的执行器，这些执行器的动作在学习者策略下往往具有高惊讶度（即负对数概率）。监督微调能够从这类记录数据中稳定学习，而策略梯度方法通常不能。我们将这种差异追溯到来自惊讶数据的负学习效应：罕见的失败可能主导有限批次的更新，而罕见的成功则携带着当前策略本会错过的信息。愉悦策略梯度通过用“惊喜”（delight，即优势与惊讶度的乘积）对每一项进行门控来区分这两者，且无需行为概率。在一个受污染的老虎机问题中，DG 以一个随学习者能力提升而趋于消失的因子来抑制罕见失败的垂直噪声；任何仅基于学习者概率的门控都无法在不抑制罕见成功的情况下做到这一点。在带有过时执行器的 MNIST 任务上，DG 优于重要性加权的策略梯度方法（原文摘要在此处截断）。

    arXiv:2603.20521v3 Announce Type: replace-cross  Abstract: Distributed reinforcement learning trains on data from stale, buggy, or mismatched actors, whose actions often have high surprisal (negative log-probability) under the learner's policy. Supervised fine-tuning learns stably from such logged data, while policy gradient often does not. We trace the difference to negative learning from surprising data: rare failures can dominate finite-batch updates, while rare successes carry information the current policy would otherwise miss. The Delightful Policy Gradient (DG) separates the two by gating each term with delight, the product of advantage and surprisal, and needs no behavior probabilities. In a contaminated bandit, DG suppresses the perpendicular noise of rare failures by a factor that vanishes as the learner improves; no gate based only on learner probability can do this without also suppressing rare successes. On MNIST with stale actors, DG beats importance-weighted policy gradi
    
[^39]: 随机特征模型的幂律谱

    Power-Law Spectrum of the Random Feature Model

    [https://arxiv.org/abs/2603.14578](https://arxiv.org/abs/2603.14578)

    该论文刻画了随机特征模型总体协方差的谱结构，证明了数据协方差的幂律谱在经过随机线性投影与非线性激活后仍以幂律形式被保留，从而为神经网络缩放定律背后的谱机制提供了理论解释。

    

    神经网络的缩放定律——即损失随参数量、数据量和计算量呈幂律衰减——从根本上依赖于数据协方差的谱结构，其中幂律特征值衰减在视觉和语言任务中普遍存在。一个核心问题是：当数据经过神经网络的基本构建模块，即随机线性投影加非线性激活时，这种谱结构是被保留还是被破坏？我们针对随机特征模型研究了这一问题：给定数据 $x \sim N(0,H)\in \mathbb{R}^v$，其中 $H$ 具有 $\alpha$ 幂律谱（$\lambda_j(H) \asymp j^{-\alpha}$，$\alpha > 1$），一个高斯草图矩阵 $W \in \mathbb{R}^{v\times d}$，以及一个次数为 $p$ 的逐元素多项式 $f$，我们刻画了总体随机特征协方差 $\mathbb{E}_{x}[\frac{1}{d}f(W^\top x)^{\otimes 2}]$ 的特征值。我们证明了匹配的……

    arXiv:2603.14578v2 Announce Type: replace-cross  Abstract: Scaling laws for neural networks, in which the loss decays as a power-law in the number of parameters, data, and compute, depend fundamentally on the spectral structure of the data covariance, with power-law eigenvalue decay appearing ubiquitously in vision and language tasks. A central question is whether this spectral structure is preserved or destroyed when data passes through the basic building block of a neural network: a random linear projection followed by a nonlinear activation. We study this question for the random feature model: given data $x \sim N(0,H )\in \mathbb{R}^v$ where $H$ has $\alpha$-power-law spectrum ($\lambda_j( H ) \asymp j^{-\alpha}$, $\alpha > 1$), a Gaussian sketch matrix $W \in \mathbb{R}^{v\times d}$, and an entrywise polynomial $f$ of degree $p$, we characterize the eigenvalues of the population random-feature covariance $\mathbb{E}_{ x }[\frac{1}{d}f(. W ^\top x )^{\otimes 2}]$. We prove matching
    
[^40]: 惊异-雷尼自由能

    Surprisal-R\'enyi Free Energy

    [https://arxiv.org/abs/2603.03405](https://arxiv.org/abs/2603.03405)

    本文提出惊异-雷尼自由能（SRFE）这一似然比的归一化雷尼-切尔诺夫对数矩泛函，证明其在端点极限下恢复前向与反向KL散度，并通过局部展开揭示了对数似然比方差作为一阶修正所体现的均值-方差权衡机制。

    

    前向与反向Kullback-Leibler（KL）散度在学习与推理中作为极限目标而出现，然而它们会诱导出显著不同的归纳偏置，这一点无法仅凭期望项加以解释。我们提出了惊异-雷尼自由能，它是似然比的一个归一化雷尼-切尔诺夫对数矩泛函。尽管对于固定的阶参数取值，SRFE在代数上与雷尼散度相关联，但其端点对称的归一化提供了一个独特的自由能视角，可用于分析前向与反向KL行为如何在奇异极限附近涌现。我们证明SRFE在端点极限下分别恢复出前向与反向KL散度，并推导了在这两种情形附近的局部展开，其中对数似然比的方差作为偏离KL目标的一阶修正项出现。这些展开揭示了一个显式的均值-方差权衡，支配着对KL目标的偏离。

    arXiv:2603.03405v3 Announce Type: replace-cross  Abstract: The forward and reverse Kullback-Leibler (KL) divergences arise as limiting objectives in learning and inference, yet they induce markedly different inductive biases that cannot be explained by expectation terms alone. We introduce the Surprisal-R\'enyi Free Energy (SRFE), a normalized R\'enyi-Chernoff log-moment functional of the likelihood ratio. Although SRFE is algebraically related to R\'enyi divergence for fixed values of the order parameter, its endpoint-symmetric normalization yields a distinct free-energy viewpoint for analyzing how forward- and reverse KL behavior emerges near singular limits. We show that SRFE recovers the forward and reverse KL divergences as endpoint limits and derive local expansions around both regimes in which the variance of the log-likelihood ratio appears as a first-order correction away from the KL objective. These expansions reveal an explicit mean-variance tradeoff governing departures fro
    
[^41]: 非比例风险下平均风险的高效且无偏学习

    Efficient and Debiased Learning of Average Hazard Under Non-Proportional Hazards

    [https://arxiv.org/abs/2602.13475](https://arxiv.org/abs/2602.13475)

    该论文提出一个半参数双重稳健框架，实现了非比例风险情形下协变量调整平均风险（AH）的高效估计与有效的√n推断，为比例风险假设不成立时的因果效应估计提供了可靠的替代方案。

    

    Cox比例风险模型产生的风险比是汇总治疗效果的常用指标。然而，当风险非成比例时，风险比可能失去稳定的因果解释并变得依赖于具体研究，因为它实际上是用由随访时长和删失机制决定的权重来平均时变效应。我们考虑将平均风险（AH）作为替代的因果估计量：它是一个人群水平的人时事件率，在不假设比例风险的情况下仍然具有良好的定义和可解释性。尽管AH可以通过非参数方法进行估计，且已有回归式的调整方法被提出，但现有方法并未为灵活的高维冗余参数估计提供具有有效√n推断的通用框架。我们通过开发一个半参数、双重稳健的协变量调整AH框架来填补这一空白。我们在非参数模型中建立了AH的路径可微性……

    arXiv:2602.13475v2 Announce Type: replace-cross  Abstract: The hazard ratio from the Cox proportional hazards model is a ubiquitous summary of treatment effect. However, when hazards are non-proportional, the hazard ratio can lose a stable causal interpretation and become study-dependent because it effectively averages time-varying effects with weights determined by follow-up and censoring. We consider the average hazard (AH) as an alternative causal estimand: a population-level person-time event rate that remains well-defined and interpretable without assuming proportional hazards. Although AH can be estimated nonparametrically and regression-style adjustments have been proposed, existing approaches do not provide a general framework for flexible, high-dimensional nuisance estimation with valid sqrt{n} inference. We address this gap by developing a semiparametric, doubly robust framework for covariate-adjusted AH. We establish pathwise differentiability of AH in the nonparametric mode
    
[^42]: 超越阿罗：多准则基准测试中从不可能到可能

    Beyond Arrow: From Impossibility to Possibility in Multi-Criteria Benchmarking

    [https://arxiv.org/abs/2602.07593](https://arxiv.org/abs/2602.07593)

    该论文突破了阿罗不可能定理对多准则基准排序的限制，识别出单峰、群可分和距离受限三类偏好结构，在这些条件下多指标基准测试可以产生稳定且有意义的大模型排名。

    

    现代基准测试（如 HELM、MMLU）需要考虑准确率、鲁棒性和效率等多个指标。当试图将这些指标汇总为单一排名时，自然的聚合程序可能产生循环的两两比较，或对模型集合变化不稳定的排名。我们将这种聚合形式化为一个社会选择问题：每个指标在每个数据集上诱导出对模型的偏好排名，而基准测试算子则跨指标聚合这些投票。虽然先前的工作主要聚焦于阿罗不可能定理，我们识别出了不可能性消失的充分条件，从而使有意义的多准则基准测试成为可能。特别是，我们研究了排名组合上的三种限制，并证明在单峰、群可分和距离受限的偏好下，基准测试算子能够构建出性质良好的相关模型排名。

    arXiv:2602.07593v2 Announce Type: replace  Abstract: Modern benchmarks such as HELM MMLU account for multiple metrics like accuracy, robustness and efficiency. When trying to turn these metrics into a single ranking, natural aggregation procedures can produce cyclic pairwise comparisons or rankings that are unstable to changes in the model set. We formalize this aggregation as a social choice problem where each metric induces a preference ranking over models on each dataset, and a benchmark operator aggregates these votes across metrics. While prior work has focused on Arrow's impossibility result, we identify sufficient conditions under which the impossibility disappears and meaningful multi-criteria benchmarking becomes possible. In particular, we deal with three restrictions on the combinations of rankings and show that on single-peaked, group-separable and distance-restricted preferences, the benchmark operator allows for the construction of well-behaved rankings of the involved mo
    
[^43]: 随机邻近算法最后迭代点的收敛速率

    Convergence Rate of the Last Iterate of Stochastic Proximal Algorithms

    [https://arxiv.org/abs/2602.05489](https://arxiv.org/abs/2602.05489)

    本文在放松有界方差假设的条件下，证明了随机梯度邻近方法与随机增量邻近方法的最后迭代点在逐分量凸性与光滑性下达到对数因子意义下最优的 $\widetilde{O}(1/\sqrt{T})$ 收敛速率，并可直接应用于多任务与联邦学习中的图引导正则项。

    

    arXiv:2602.05489v2 公告类型： replace-cross 摘要： 我们分析了两种求解可加复合凸优化问题的经典算法，该问题的目标函数是光滑项与非光滑正则项之和。第一种算法是针对单一正则项的随机梯度邻近方法；第二种是随机增量邻近方法，当正则项表示为许多非光滑函数之和时，该方法使用随机选取函数的邻近算子。我们重点关注放松获得最后迭代点收敛速率时常见但苛刻的有界方差假设。我们在逐分量凸性与光滑性条件下，证明了两种算法最后迭代点的 $\widetilde{O}(1/\sqrt{T})$ 收敛速率，该速率在对数因子意义下是最优的。我们的结果可直接应用于多任务学习和联邦学习中出现的图引导正则项，其中正则项可分解为协作图上各边函数之和。

    arXiv:2602.05489v2 Announce Type: replace-cross  Abstract: We analyze two classical algorithms for solving additively composite convex optimization problems where the objective is the sum of a smooth term and a nonsmooth regularizer. The first algorithm is the proximal stochastic gradient method for a single regularizer; the second is the randomized incremental proximal method, which uses the proximal operator of a randomly selected function when the regularizer is given as the sum of many nonsmooth functions. We focus on relaxing the bounded variance assumption that is common, yet stringent, for getting last iterate convergence rates. We prove the $\widetilde{O}(1/\sqrt{T})$ rate of convergence for the last iterate of both algorithms under componentwise convexity and smoothness, which is optimal up to log terms. Our results apply directly to graph-guided regularizers that arise in multi-task and federated learning, where the regularizer decomposes as a sum over edges of a collaboratio
    
[^44]: 通过全局参数融合的最优树实现高效亚组分析

    Efficient Subgroup Analysis via Optimal Trees with Global Parameter Fusion

    [https://arxiv.org/abs/2602.04077](https://arxiv.org/abs/2602.04077)

    提出一种基于混合整数优化的融合最优因果树方法，通过全局最优划分与参数融合约束，克服了传统树方法贪婪分裂导致的次优划分和过拟合问题，实现精确的差异性治疗效应亚组识别。

    

    识别差异性治疗效应并进行统计推断（在临床研究中通常称为亚组分析）是精准健康的核心。亚组分析使从业者能够精准定位对治疗特别受益或受保护的人群，从而推动针对性干预措施的开展。基于树的递归划分方法因其良好的可解释性而被广泛应用于亚组分析。然而，这些方法存在显著的局限性，包括贪婪启发式算法导致的次优划分，以及局部估计分裂带来的过拟合问题，在样本量有限的情况下尤为突出。为了解决这些局限性，我们提出了一种融合最优因果树方法，利用混合整数优化（MIO）来实现精确的亚组识别。我们的方法确保全局最优划分，并引入参数融合约束以促进[统计推断]。……

    arXiv:2602.04077v2 Announce Type: replace-cross  Abstract: Identifying and making statistical inferences on differential treatment effects (commonly known as subgroup analysis in clinical research) is central to precision health. Subgroup analysis allows practitioners to pinpoint populations for whom a treatment is especially beneficial or protective, thereby advancing targeted interventions. Tree based recursive partitioning methods are widely used for subgroup analysis due to their interpretability. Nevertheless, these approaches encounter significant limitations, including suboptimal partitions induced by greedy heuristics and overfitting from locally estimated splits, especially under limited sample sizes. To address these limitations, we propose a fused optimal causal tree method that leverages mixed integer optimization (MIO) to facilitate precise subgroup identification. Our approach ensures globally optimal partitions and introduces a parameter fusion constraint to facilitate i
    
[^45]: 超越全局Lipschitz连续性的收缩动力学朗之万采样器

    Contractive kinetic Langevin samplers beyond global Lipschitz continuity

    [https://arxiv.org/abs/2509.12031](https://arxiv.org/abs/2509.12031)

    本文提出两种驯化离散化的动力学朗之万采样器（驯化指数欧拉格式与驯化OBABO格式），突破了全局Lipschitz连续性的限制，可处理梯度超线性增长的对数凹分布采样，并证明其在Wasserstein和全变差距离下可达接近最优的Õ(ε⁻¹)收敛速率。

    

    本文研究了在对数凹分布（其梯度可能呈超线性增长）上使用动力学（欠阻尼）朗之万算法进行采样的问题。通过精心设计的驯化方案，我们提出了动力学朗之万随机微分方程的两种新颖离散化方法：驯化指数欧拉格式和驯化OBABO格式，并证明二者均具有收缩性且满足对数Sobolev不等式。在此基础上，我们建立了每种算法所达到的分布与目标测度之间在2-Wasserstein距离和全变差距离下的一系列非渐近界。特别地，我们证明两种算法在Wasserstein距离和全变差距离下均可达到接近最优的Õ(ε⁻¹)收敛速率，且驯化OBABO格式在Hessian局部正则性增强的条件下可达到Õ(ε⁻¹ᐟ²)的收敛速率。

    arXiv:2509.12031v3 Announce Type: replace-cross  Abstract: In this paper, we examine the problem of sampling from log-concave distributions with (possibly) superlinear gradient growth under kinetic (underdamped) Langevin algorithms. Using a carefully tailored taming scheme, we propose two novel discretizations of the kinetic Langevin SDE, the tamed exponential euler scheme and tamed OBABO and we show that they are both contractive and satisfy a log-Sobolev inequality. Building on this, we establish a series of non-asymptotic bounds in $2$-Wasserstein and total variation distance between the law reached by each algorithm and the underlying target measure. In particular, we show that both algorithms can reach almost optimal $\tilde{\mathcal{O}}(\epsilon^{-1})$ convergence rate in Wasserstein and Total Variation distances, and $\tilde{\mathcal{O}}(\epsilon^{-\frac{1}{2}})$ under increased Hessian local regularity for the tamed OBABO scheme.
    
[^46]: 信息约束下分布式简单二元假设检验的样本复杂度

    The Sample Complexity of Distributed Simple Binary Hypothesis Testing under Information Constraints

    [https://arxiv.org/abs/2506.13686](https://arxiv.org/abs/2506.13686)

    本文解决了信息约束下分布式简单二元假设检验的两个开放问题，证明顺序交互无法降低样本复杂度，并给出了通信约束下样本复杂度的最优紧致界。

    

    本文解决了最近一篇论文（arXiv:2403.16981）中关于信息约束下分布式简单二元假设检验样本复杂度的两个开放问题。第一个开放问题询问交互是否能降低分布式简单二元假设检验的样本复杂度。在本文中，我们证明了顺序交互没有帮助。第二个问题建议收紧通信约束下简单二元假设检验的现有样本复杂度界。我们为该设置推导出了最优紧致的界，从而解决了这一问题。我们的主要技术贡献包括：(i) 简单二元假设检验中贝叶斯误差的一个单次（one-shot）下界，该下界满足一个关键的张量化性质；(ii) 对无约束情形下简单二元假设检验样本复杂度公式（最初由 arXiv:2403.16981 建立）的简化证明；(iii) 一个 rev

    arXiv:2506.13686v2 Announce Type: replace-cross  Abstract: This paper resolves two open problems from a recent paper, arXiv:2403.16981, concerning the sample complexity of distributed simple binary hypothesis testing under information constraints. The first open problem asks whether interaction reduces the sample complexity of distributed simple binary hypothesis testing. In this paper, we show that sequential interaction does not help. The second problem suggests tightening existing sample complexity bounds for communication-constrained simple binary hypothesis testing. We derive optimally tight bounds for this setting and resolve this problem. Our main technical contributions are: (i) a one-shot lower bound on the Bayes error in simple binary hypothesis testing that satisfies a crucial tensorisation property; (ii) a streamlined proof of the formula for the sample complexity of simple binary hypothesis testing without constraints, first established in arXiv:2403.16981; and (iii) a rev
    
[^47]: 基于深度学习的状态空间模型方法：选择性综述

    Deep Learning-based Approaches for State Space Models: A Selective Review

    [https://arxiv.org/abs/2412.11211](https://arxiv.org/abs/2412.11211)

    本文对基于深度学习的状态空间模型方法进行了选择性综述，为离散时间与连续时间（如潜在神经ODE和SDE）的深度状态空间模型提供了统一的理论视角。

    

    状态空间模型（SSMs）为动力系统分析提供了一个强大的框架，其中假设系统的时间动态通过潜在状态的演化来捕捉，而这些潜在状态决定着观测值。本文对基于深度神经网络的SSMs方法的最新进展进行了选择性综述，并为离散时间深度状态空间模型和连续时间模型（如潜在神经常微分方程与随机微分方程）提供了统一的视角。文章首先概述了基于经典最大似然方法的SSMs学习方式，回顾了变分自编码器作为存在潜在变量情况下基于神经网络方法的一般学习流程，并详细讨论了属于SSM框架的代表性深度学习模型。此外，还介绍了最新的研究进展，即SSMs被用作独立架构……

    arXiv:2412.11211v2 Announce Type: replace-cross  Abstract: State-space models (SSMs) offer a powerful framework for dynamical system analysis, wherein the temporal dynamics of the system are assumed to be captured through the evolution of the latent states, which govern the values of the observations. This paper provides a selective review of recent advancements in deep neural network-based approaches for SSMs, and presents a unified perspective for discrete-time deep state space models and continuous-time ones such as latent neural Ordinary Differential and Stochastic Differential Equations. It starts with an overview of the classical maximum likelihood based approach for learning SSMs, reviews the variational autoencoder as a general learning pipeline for neural network-based approaches in the presence of latent variables, and discusses in detail representative deep learning models that fall under the SSM framework. Very recent developments, where SSMs are used as standalone architec
    
[^48]: 含隐变量DAG中的平均因果效应估计：超越后门与前门准则

    Average Causal Effect Estimation in DAGs with Hidden Variables: Beyond Back-Door and Front-Door Criteria

    [https://arxiv.org/abs/2409.03962](https://arxiv.org/abs/2409.03962)

    该论文针对含隐变量DAG中超越后门与前门准则的可识别因果效应，提出了基于序贯回归和贝叶斯法则重参数化的估计方法，从而避免了条件密度估计与数值积分带来的统计和计算难题。

    

    具有隐变量的有向无环图（DAG）中因果效应的识别理论已经相当成熟，但超越g-公式的估计与推断方法仍发展不足。以往研究已针对广泛类别的可识别效应引入了基于影响函数的估计量。当通过观测数据密度分解来实现这些估计量时，往往需要进行条件密度估计和数值积分，这对于连续变量或多变量而言在统计上具有挑战性且计算代价高昂。此前关于相关因果泛函的研究采用了序贯回归和密度比的贝叶斯法则表示来避免这些需求。基于已有的识别理论与影响函数结果，我们将这些重参数化明确应用于在处理原始可固定性、Tian和Pearl的充要条件下所识别的效应……

    arXiv:2409.03962v3 Announce Type: replace-cross  Abstract: The identification theory for causal effects in directed acyclic graphs (DAGs) with hidden variables is well established, but methods for estimation and inference beyond the g-formula remain less developed. Previous studies have introduced influence-function-based estimators for broad classes of identifiable effects. When implemented through the observed-data density factorization, these estimators can require conditional density estimation and numerical integration, which can be statistically challenging and computationally demanding for continuous or multivariate variables. Prior work on related causal functionals has used sequential regressions and Bayes-rule representations of density ratios to avoid these requirements. Building on established identification and influence-function results, we make these reparameterizations explicit for effects identified under treatment primal fixability, Tian and Pearl's necessary and suff
    
[^49]: 关于平均场机制下多层ResNet梯度下降的全局收敛性研究

    On the Global Convergence of Gradient Descent for multi-layer ResNets in the mean-field regime

    [https://arxiv.org/abs/2110.02926](https://arxiv.org/abs/2110.02926)

    该论文证明了在平均场机制下，当激活函数满足2-齐次或部分1-齐次条件时，多层ResNet训练所对应的梯度流偏微分方程能收敛到全局最小值，表明足够大的ResNet可通过一阶优化方法找到拟合训练数据的全局最优解。

    

    在ResNet中寻找参数的最优配置是一个非凸最小化问题，但当网络过参数化且训练算法运行足够多次迭代时，一阶方法往往能找到全局最优解。我们在平均场机制（mean-field regime）中研究这一现象，在该机制下，网络可以近似地由一个常积分方程（OIE）描述，而ResNet的训练过程则变为一个梯度流偏微分方程（PDE）。在激活函数为$2$-齐次或部分$1$-齐次的条件下，我们证明该梯度流PDE收敛到全局最小值。这一结果表明，如果ResNet足够大，一阶优化方法同样可以找到拟合训练数据的全局最小化器。此外，通过控制泛化误差，我们证明了该梯度流PDE相对于其代价函数是稳定的。

    arXiv:2110.02926v3 Announce Type: replace  Abstract: Finding the optimal configuration of parameters in ResNet is a nonconvex minimization problem, but first-order methods often find the global optimum when the network is overparameterized and the training algorithm is run for sufficiently many iterations. We study this phenomenon in the mean-field regime, where the network can be described approximately by an ordinary integral equation (OIE) and the training process of ResNet becomes a gradient-flow partial differential equation (PDE). Under the condition that the activation function is $2$-homogeneous or partially $1$-homogeneous, we show that this gradient-flow PDE converges to the global minimum. This result suggests that if the ResNet is sufficiently large, first-order optimization methods can find global minimizers that fit the training data as well. Further, by controlling the generalization error, we prove that the gradient-flow PDE is stable with respect to its cost function, 
    
[^50]: 论噪声与曲率之间的相互作用及其对优化和泛化的影响

    On the interplay between noise and curvature and its effect on optimization and generalization

    [https://arxiv.org/abs/1906.07774](https://arxiv.org/abs/1906.07774)

    该论文探讨了损失曲率与梯度噪声的相互作用如何影响优化速度以及泛化间隙的估计，并澄清了费舍尔矩阵、海森矩阵和梯度协方差矩阵之间的区别。

    

    使用随机方法最小化期望损失的速度取决于两个性质：损失的曲率和梯度的方差。以往的大多数研究只关注其中某一个性质，而我们探讨了二者的相互作用如何影响优化速度。此外，由于最终目标是获得良好的泛化性能，我们阐明了曲率和噪声两者都与正确估计泛化间隙相关。我们意识到一些现有工作的局限性源于对这些矩阵的混淆，因此还澄清了费舍尔矩阵、海森矩阵和梯度协方差矩阵之间的区别。

    arXiv:1906.07774v3 Announce Type: replace  Abstract: The speed at which one can minimize an expected loss using stochastic methods depends on two properties: the curvature of the loss and the variance of the gradients. While most previous works focus on one or the other of these properties, we explore how their interaction affects optimization speed. Further, as the ultimate goal is good generalization performance, we clarify how both curvature and noise are relevant to properly estimate the generalization gap. Realizing that the limitations of some existing works stems from a confusion between these matrices, we also clarify the distinction between the Fisher matrix, the Hessian, and the covariance matrix of the gradients.
    

