# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sliding-Window Detection of Hidden Regimes under Stochastic Switching](https://arxiv.org/abs/2611.34905) | 该论文证明了对随机切换的隐藏状态过程，滑动窗口内发生状态切换时错误概率恰好为 1/2，由此得到解码错误的上界为单状态错误率与切换率贡献之和。 |
| [^2] | [The Look-Elsewhere Effect in Searches for New Physics: from Classical Methods to Machine Learning](https://arxiv.org/abs/2611.34888) | 本综述系统回顾了对撞机新物理搜索中的“观察别处效应”（LEE），涵盖基于Davies与Gross-Vitells的经典统计修正框架，并讨论了“别处”定义的模糊性以及机器学习搜索中隐含的后选择偏差等尚未解决的现代挑战。 |
| [^3] | [Provably Efficient Offline Discounted Reinforcement Learning under Recursive Entropic Risk](https://arxiv.org/abs/2611.34840) | 本文提出了首个在递归熵风险度量下离线强化学习的可证明高效算法 ERM-VI-LCB，并给出匹配的样本复杂度上下界，同时覆盖风险厌恶与风险追求两种情形。 |
| [^4] | [Conditional Independence Graph Learning from Dependent Data Via Time-Delay Embedding](https://arxiv.org/abs/2611.34616) | 提出一种基于时滞嵌入的时域方法，将相依时间序列的条件独立图估计转化为多属性随机向量图估计问题，突破了以往频域方法仅适用于短程相依数据的限制，可同时处理短程相依和长程相依的高维平稳多元高斯时间序列。 |
| [^5] | [Global Convergence of Second-Order Methods for Physics-Informed Neural Networks in the Overparameterized Limit](https://arxiv.org/abs/2611.34452) | 本文首次为物理信息神经网络训练中的正则化牛顿法和Gauss-牛顿法建立了全局收敛理论，证明了在过参数化极限下训练动力学以显式速率概率收敛到确定性极限方程的解。 |
| [^6] | [Change Point Detection with Generative Models](https://arxiv.org/abs/2611.34228) | 该论文提出利用生成模型对时间序列窗口进行数据增广，突破了基于窗口的变点检测中“样本量瓶颈”的根本局限，显著提升了变点检测的准确性与质量。 |
| [^7] | [Cell-Type-Specific Expression Is Hard to Predict from Bulk, but It Is Still Useful](https://arxiv.org/abs/2611.34166) | 该论文通过推导闭式精度界从理论上解释了为何从bulk RNA数据预测细胞类型比例容易而预测细胞类型特异性表达困难，并证明尽管CTSE难以估计，对其进行目标性反卷积在非小细胞肺癌总生存期预测中仍具有预后价值。 |
| [^8] | [Accelerated Proximal Gradient Methods for Multiobjective Optimization over Probability Measures via Hilbert-Space Lifting](https://arxiv.org/abs/2611.34073) | 本文提出基于希尔伯特空间提升的多目标Wasserstein近端梯度方法（MWPG）及其加速版本（MWAPG），通过隐式Wasserstein梯度步求解概率测度上KL散度的Pareto多目标优化问题，并提供严格的离散时间收敛保证。 |
| [^9] | [Asymptotically Efficient Finite Impulse Response System Identification without Noise Knowledge](https://arxiv.org/abs/2611.34039) | 提出了一类无需噪声分布先验知识的自适应估计器（eMLE与epMLE），通过核密度估计和一致性初始估计的残差来近似似然函数，在未知或重尾噪声下实现了有限脉冲响应系统辨识的渐近有效性。 |
| [^10] | [BReD: Block Replay Dithering for Stable Low-Bit EMA Optimizer States](https://arxiv.org/abs/2611.33951) | 提出 BReD 方法，通过将确定性种子重放与分块共享抖动值相结合的块重放抖动技术，减少 EMA 优化器状态迭代量化中的舍入偏差并控制舍入方差，从而解决误差累积问题、稳定大规模训练。 |
| [^11] | [A Graphical Criterion for Linear Causal Bounds](https://arxiv.org/abs/2611.33950) | 本文针对每个混杂组件最多含两个隐藏原因的因果图，给出了判断尖锐边界能否通过求解单个线性规划得到的充分必要条件，并实现了Python包partialid以计算高效的方式获得尖锐边界。 |
| [^12] | [CondUQ: Calibrated Conditional Uncertainty Quantification Using Distribution Matching](https://arxiv.org/abs/2611.33939) | 该论文提出 condUQ 方法，通过将误差模型条件化于可观测因素（如输入质量和运行条件）进行分布匹配，在不修改甚至黑盒的现有系统之上实现准确可靠的条件不确定性量化。 |
| [^13] | [Correlation-Aware Evidential Fusion for Trusted Multi-View Classification](https://arxiv.org/abs/2611.33900) | 提出了相关性感知证据融合框架CAFE，通过显式建模视图间的相关性，避免重复计算冗余证据、强化互补信息，从而提升可信多视图分类的性能和不确定性估计的校准质量。 |
| [^14] | [Certifying a Sufficient Gaussian Dependence Regime for Learned Weighted FDR Control](https://arxiv.org/abs/2611.33892) | 该论文提出利用参考观测认证单侧高斯检验的充分相依性条件，使学习得到的加权Benjamini-Hochberg程序在满足该条件时能严格控制错误发现率（FDR），并给出了匹配的极小极大样本量界。 |
| [^15] | [RHOPE: LLM-Augmented Adaptive Experimental Design with Complex Contextual Information](https://arxiv.org/abs/2611.33767) | 该论文提出RHOPE框架，利用大语言模型将复杂上下文信息转化为代理结果，将有限批次自适应实验设计形式化为后验马尔可夫决策过程，并通过样本增强与决策前预观察代理结果两种方式提升估计精度和分配决策效果。 |
| [^16] | [Understanding Double Descent through Universal Compression](https://arxiv.org/abs/2611.33759) | 该论文提出以通用压缩的全新理论视角解释深度学习中的双下降现象，将最大似然超额测试损失的上界分解为近似与偏移项、极小极大批次遗憾、先验代价和MLE与贝叶斯差距四个可解释的组成部分。 |
| [^17] | [Learning to Infer: Statistical Inference through Meta Learning](https://arxiv.org/abs/2611.33742) | 该论文将统计决策理论与元学习相结合，提出残差循环深度集合网络架构，使神经网络能从模拟任务中学习推断规则并无需参数更新地应用于新数据，同时为学习到的规则建立了渐近有效性、模型选择一致性和高维变量选择等频率学派统计保证。 |
| [^18] | [How Many Sources, Not Just How Many Problems: Planning the Precision of LLM Evaluations from Earlier Results](https://arxiv.org/abs/2611.33622) | 该论文提出利用试点数据或早期评估结果，结合经典设计效应与组内相关系数，在正式评估前预测当评估预算集中于更少来源时LLM评估分数标准差的增幅，从而事先规划评估的精度。 |
| [^19] | [Improved Sample Complexity of Robust Reinforcement Learning with KL Uncertainty Sets](https://arxiv.org/abs/2611.33592) | 本文针对基于KL散度不确定性集的鲁棒强化学习，给出了更紧致且完备的样本复杂度上下界，填补了以往结果中的差距，并将下界覆盖范围扩展到完整的不确定性水平区间。 |
| [^20] | [Amortizing Exact Orthogonality: Drift-Triggered Restoration for Orthogonal Fine-Tuning](https://arxiv.org/abs/2611.33589) | 该论文提出 RTOFT 方法，用一阶更新 W(I+S) 取代逐步构造正交变换，仅在漂移 ‖S‖₂ 超过容差 τ 时才通过极分解恢复精确正交性，从而以漂移触发的方式摊销正交微调的计算开销。 |
| [^21] | [Adversarial Bandit Convex Optimization with $d^{3/2}\sqrt{n}$ Regret](https://arxiv.org/abs/2611.33531) | 本文提出一种带高斯探索的在线牛顿步算法，将对抗性赌博机凸优化的遗憾上界从 $\widetilde O(d^{5/2}\sqrt n)$ 改进为 $\widetilde O(d^{3/2}\sqrt n)$，匹配了此前被猜想的最优速率，并将极小极大遗憾确定到 $d^{1/6}$ 因子以内。 |
| [^22] | [On the Impact of Extra Information in Bandit Learning: More is Not Always Merrier](https://arxiv.org/abs/2611.33479) | 该研究揭示了一个反直觉现象：在随机老虎机学习中，额外信息可能使UCB算法的遗憾变差而非改善，并提出了一种能为广泛类别老虎机算法可证明地保证“非退化性”的元算法。 |
| [^23] | [Differentially Private Policy Optimization in Linear MDPs with Adversarial Rewards](https://arxiv.org/abs/2611.33469) | 本文针对具有未知转移和自适应对抗性奖励的线性MDP，提出了一种结合批量化PPO与乐观策略评估的差分隐私策略优化算法，在联合差分隐私下同时保护轨迹与奖励数据，并实现了保留 $\widetilde{O}\left(K^{3/4}\right)$ 主导项的高概率遗憾界。 |
| [^24] | [PAC Learning to Sample: On Computational Aspects of Language Generation](https://arxiv.org/abs/2611.33464) | 该论文将PAC学习框架与极限语言生成模型相结合，首次为语言生成任务提供了关于样本复杂度和时间复杂度的定量保证。 |
| [^25] | [Centered Kernel Distance: Dimension-Free Error Bounds for Comparing Neural Representations](https://arxiv.org/abs/2611.33447) | 本文提出中心核距离（CKD）这一神经表征相似性度量，证明其存在无偏估计量且估计误差以 1/M + 1/N 的速度衰减，同时收敛速率与实验条件空间维度无关，从而克服了表征相似性估计中的维度灾难。 |
| [^26] | [Gaussian priors from nonlinear PDEs with the semiclassical approximation](https://arxiv.org/abs/2611.33400) | 本文提出利用半经典近似，通过在稳定解附近展开非线性PDE的变分能量或积分平方残差来构造高斯先验，从而将贝叶斯不确定性量化的概率数值方法拓展到非线性偏微分方程领域。 |
| [^27] | [Evaluating Ensemble Methods for Algorithmic Stability Under Data Perturbations](https://arxiv.org/abs/2611.33375) | 本文揭示了集成方法对分类器稳定性的双面效应：子装袋可显著提升不稳定分类器的稳定性，但会破坏本已稳定的分类器，其有限装袋引入的蒙特卡洛噪声可通过膨胀argmax方法缓解。 |
| [^28] | [Two Timescales of Forgetting in Fine-tuning of Language Models: Fast Behavioral Suppression and Slow Representational Drift](https://arxiv.org/abs/2611.33302) | 本研究通过在随机层次模型上预训练并用受限规则子集微调transformer，发现语言模型微调中的遗忘具有两个时间尺度——表面行为被快速抑制，而支撑该行为的潜在表征则以缓慢漂移的方式逐渐丢失。 |
| [^29] | [B-SLIM: BART Sparse Linear Interactions Model for Heterogeneous Treatment Effects](https://arxiv.org/abs/2611.33246) | 提出B-SLIM半参数因果推断模型，将灵活的BART预后建模与经全局-局部收缩先验正则化的稀疏线性治疗效应组件（含主效应和交互效应）相结合，兼顾模型灵活性、治疗效应可解释性以及对治疗效应调节因子的直接推断能力。 |
| [^30] | [A Causal Energy Prior for Learning Directed Acyclic Graphs](https://arxiv.org/abs/2611.33225) | 提出了局部-全局能量先验（LGE），通过结合局部结构差异（结构汉明距离）与因果差异（结构干预距离）构建DAG上的吉布斯分布，使贝叶斯DAG学习能够编码干预沿路径传播的因果信息，并量化了数据在先验部分错误时对先验的修正能力。 |
| [^31] | [The Symmetry Budget: How Much Group Structure Should a Network Architecture Assume?](https://arxiv.org/abs/2611.33169) | 该论文提出“对称性预算”概念，在一个精确可解模型中以闭式形式推导出有限样本超额风险，证明当数据仅具有近似对称性时，网络架构中的中间对称性绑定程度永远不会严格最优。 |
| [^32] | [Wavelet determinantal samplers: Optimal approximation rates adapting to regularity](https://arxiv.org/abs/2611.29992) | 该论文构造了基于小波的行列式点过程采样器，在流形与几何图上对线性统计量实现了 n^{1-2s/m} 阶的方差缩减，且该逼近速率能够自适应于观测量的 Hölder/Sobolev 正则性。 |
| [^33] | [Out-of-the-Box Tabular Foundation Models for Bayesian Optimization in Materials Discovery](https://arxiv.org/abs/2611.12491) | 该论文证明 TabPFN 和 TabICL 等表格基础模型可以作为开箱即用的代理模型，在材料发现的贝叶斯优化中媲美高斯过程等传统方法，无需核函数选择或任务特定训练。 |
| [^34] | [Compatible Likelihoods for Flow Matching on Manifolds](https://arxiv.org/abs/2610.31981) | 该论文针对黎曼流匹配中基于似然的目标可能产生推离数据分布的速度场这一失败模式，提出“相容性”概念刻画似然目标的有效性，并进一步设计了一种基于指数族的新型目标，既满足相容性又仅支持紧流形上可行的速度。 |
| [^35] | [Gaussian Approximation of Besov Functions for Logistic Classification on Manifolds](https://arxiv.org/abs/2610.27022) | 该论文针对紧致光滑流形上的正则化逻辑斯蒂分类，借助常数线性依赖预测器上确界范数的Bernstein型不等式得到无需有界贝叶斯假设的快速预言机不等式，并通过多带宽高斯核积分算子的带符号组合与实插值方法，构造出同时利用内在维度与目标正则性且无饱和现象的Besov函数高斯逼近。 |
| [^36] | [Federated Community Detection in Bipartite Networks](https://arxiv.org/abs/2610.25997) | 提出FLICKER与BiFLICKER方法，在原始数据不可共享的联邦环境下实现了二部网络的谱估计与社区检测，并提供了非渐近收敛与误聚类率的理论保证。 |
| [^37] | [Minimax estimation of the pseudo-spectral gap of continuous Markov chains, with applications to empirical concentration bounds](https://arxiv.org/abs/2610.20007) | 本文提出了紧致连续状态空间马尔可夫链伪谱隙的首个极小极大最优估计器，结合 Bernstein 不等式推导出经验集中不等式，并据此在无需伪谱隙先验知识的情况下，为马尔可夫奖励老虎机问题给出了具有接近独立同分布情形遗憾界的 UCB 算法。 |
| [^38] | [Extreme Binary Classification: Extreme Value Theory for Extreme Constraint on False Negative](https://arxiv.org/abs/2610.09984) | 本文提出“极端二分类”新问题，并基于极值理论设计了阈值自适应方法与基于置换检验的特征选择程序，使分类器的假负类率以快于 $1/N_1$ 的速率趋近于零，实验表现优于最先进方法。 |
| [^39] | [Conformal Prediction and Conditional Coverage for Tabular Foundation Models](https://arxiv.org/abs/2609.34887) | 该论文提出C-USIM方法，一种无需额外训练或推理、可处理多模态预测的轻量级最高预测密度分割共形预测技术，为表格基础模型提供有限样本边际覆盖有效性保证，并显著改善条件覆盖和分组覆盖精度。 |
| [^40] | [Online Non-Monotone DR-Submodular Maximization Matching the Offline $0.401$ Factor](https://arxiv.org/abs/2609.02145) | 该论文首次在对抗性在线设置下实现了非单调DR-次模最大化与非离线算法相同的0.401最优近似比，通过用加权在线学习器替代离线箱约束步骤并结合精确非对称平衡定理，在决策后全信息值预言机模型下达到次线性近似遗憾。 |
| [^41] | [Improved Quantum Algorithms for Reinforcement Learning Under a Generative Model](https://arxiv.org/abs/2608.02826) | 本文将标准值迭代与量子均值估计、量子最大值查找等量子子程序相结合，提出了求解有限时域和无限时域折扣MDP近似最优策略的新量子算法，其查询复杂度优于以往工作并接近已有的量子下界。 |
| [^42] | [Can Tabular In-Context Learners Generalize to Biomolecular Property Prediction?](https://arxiv.org/abs/2606.31126) | 本文评估了TabPFN和TabICL等表格基础模型能否作为小样本场景下的数据高效预测器泛化到生物分子性质预测任务，发现在ProteinGym上TabPFN3-ESMC600M与领先的监督方法具有竞争力并取得最低MSE，但Kermut在Spearman相关性上仍更强，且在更难的划分设置下性能明显下降。 |
| [^43] | [Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints](https://arxiv.org/abs/2605.22940) | 本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。 |
| [^44] | [Guiding Multi-Objective Genetic Programming with Description Length Improves Symbolic Regression Solutions](https://arxiv.org/abs/2605.22374) | 该论文提出将描述长度（DL）和分数贝叶斯因子（FBF）作为遗传编程符号回归中原则性的选择与搜索准则，以缓解过拟合和结构膨胀，从而提升解的泛化性能。 |
| [^45] | [Coreset-Induced Conditional Velocity Flow Matching](https://arxiv.org/abs/2605.12951) | 该论文提出CCVFM生成模型，通过熵Sinkhorn核心集构建闭式代理源替代分层整流流中内部流的高斯噪声源，使条件速度规律成为无需神经采样器即可采样的闭式高斯混合，并只需轻量级校正流即可完成训练。 |
| [^46] | [Reinforcement Learning from Human Feedback: A Statistical Perspective](https://arxiv.org/abs/2604.02507) | 本综述首次系统地从统计学视角梳理了RLHF框架，将其核心组成部分（监督微调、奖励建模与策略优化）与BTL模型、潜效用估计、主动学习、实验设计和不确定性量化等经典统计概念联系起来。 |
| [^47] | [BayesFlow 2: Multi-Backend Amortized Bayesian Inference in Python](https://arxiv.org/abs/2602.07098) | BayesFlow 2是一个支持多种深度学习后端的通用Python库，通过摊销贝叶斯推断实现对后验分布、似然和比率的快速估计，解决了传统贝叶斯方法在复杂模型和大数据集上的计算瓶颈。 |
| [^48] | [Concentration Inequalities for Exchangeable Tensors and Matrix-valued Data](https://arxiv.org/abs/2601.20152) | 该论文在可交换性条件下为张量内积与序贯矩阵和等结构化加权和建立了 Hoeffding 与 Bernstein 型集中不等式，以最优常数恢复了已知结果并改进了矩阵组合和的偏差项常数，为多因子响应模型估计与固定设计草图方法提供了新的分析工具。 |
| [^49] | [CoDeQ: Compression with a Dead-Zone Quantizer for Differentiable Joint Pruning--Quantization](https://arxiv.org/abs/2512.12981) | 该论文提出CoDeQ，通过可学习的死区量化器将剪枝与量化统一在同一训练框架中，以死区宽度控制逐层稀疏度并支持可学习的混合精度位宽，在单一全局压缩系数下联合优化权重与压缩参数，从而省去辅助分配流程和大量超参数调优。 |
| [^50] | [Debiased Inference via Posterior Corrections for High-Dimensional Regression](https://arxiv.org/abs/2512.09257) | 该论文提出一种新颖的去偏方法，通过校正整个贝叶斯后验分布的偏差并建立新的Bernstein-von Mises定理，使高维回归中基于稀疏先验的后验分布具备频率学派有效性，其可信集可作为有效置信集。 |
| [^51] | [Fixed Points and Envelope Dynamics on Credal Hyperspaces](https://arxiv.org/abs/2510.04769) | 本文证明了信度集超空间中包含单调且豪斯多夫连续的自映射在完备格的信度序区间上存在可由迭代求得的最小与最大不动点，并为随时间变化的更新族构造了界定所有容许轨迹的上下包络动力学。 |
| [^52] | [Statistical Inference for Differentially Private Stochastic Gradient Descent](https://arxiv.org/abs/2507.20560) | 本文首次建立了随机化子采样规则下SGD的渐近性质并将其扩展到DP-SGD，证明了其渐近方差可分解为统计、采样和隐私诱导三个分量，并提出了代入法和随机缩放法两种构建有效置信区间的方法。 |
| [^53] | [Stability Regularized Cross-Validation](https://arxiv.org/abs/2505.06927) | 该论文提出了一种嵌套 k 折交叉验证方案，通过最小化标准交叉验证误差与经验假设稳定性的加权和来选择超参数，并证明了由此得到的泛化界，显示在信噪比较低的场景下该方法对测试误差的改善最为显著。 |
| [^54] | [Structural transfer learning of non-Gaussian DAG.](http://arxiv.org/abs/2310.10239) | 本文提出了一种用于非高斯有向无环图的结构迁移学习方法，通过引入新颖的DAG结构相似度度量并利用不同相似性水平的辅助DAG的信息，能够显著提高目标研究中的DAG重构效果。 |
| [^55] | [Externally Valid Policy Choice.](http://arxiv.org/abs/2205.05561) | 本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。 |

# 详细

[^1]: 随机切换下隐藏状态的滑动窗口检测

    Sliding-Window Detection of Hidden Regimes under Stochastic Switching

    [https://arxiv.org/abs/2611.34905](https://arxiv.org/abs/2611.34905)

    该论文证明了对随机切换的隐藏状态过程，滑动窗口内发生状态切换时错误概率恰好为 1/2，由此得到解码错误的上界为单状态错误率与切换率贡献之和。

    

    滑动窗口平均是估计隐藏状态切换过程（如隐马尔可夫模型）当前状态的一种简单方法。较长的窗口能平均掉更多噪声，但更可能包含一次状态切换，问题在于这种切换的代价有多大。我们分析了将最近 m 个观测值的平均值归入最近状态均值的解码器。对于两个具有条件独立、随机有序观测的状态，窗口内发生的一次切换（对其位置和方向进行均匀平均后）恰好以 1/2 的概率导致分类错误。因此，对于任意平稳切换律，错误概率至多为 $\bar P(m)+\bar s(m-1)/2$，其中 $\bar P(m)$ 是平稳加权的单状态错误率，$\bar s$ 是切换率。在马尔可夫切换下，通过时间反演可以证明：只要窗口内最旧状态与当前状态不同，该窗口就会以恰好 1/2 的概率被错误分类，而与窗口内切换次数无关……

    arXiv:2611.34905v1 Announce Type: cross  Abstract: A sliding-window average is a simple estimator of the current state of a hidden regime-switching process, such as a hidden Markov model. A longer window averages out more noise but is more likely to contain a switch, and the question is what such a switch costs. We analyse the decoder that assigns the average of the last $m$ observations to the nearest regime mean. For two regimes with conditionally independent, stochastically ordered observations, a switch inside the window, averaged uniformly over its positions and directions, causes an error with probability exactly $1/2$. Consequently, for every stationary switching law, the error is at most $\bar P(m)+\bar s(m-1)/2$, where $\bar P(m)$ is the stationary-weighted single-regime error and $\bar s$ the switch rate. Under Markov switching, time reversal shows that a window whose oldest and current regimes differ is misclassified with probability exactly $1/2$, however many switches it c
    
[^2]: 新物理搜索中的“观察别处效应”：从经典方法到机器学习

    The Look-Elsewhere Effect in Searches for New Physics: from Classical Methods to Machine Learning

    [https://arxiv.org/abs/2611.34888](https://arxiv.org/abs/2611.34888)

    本综述系统回顾了对撞机新物理搜索中的“观察别处效应”（LEE），涵盖基于Davies与Gross-Vitells的经典统计修正框架，并讨论了“别处”定义的模糊性以及机器学习搜索中隐含的后选择偏差等尚未解决的现代挑战。

    

    当效应在参数空间中的位置未被事先指定时，就会产生“观察别处效应”：扫描许多位置会夸大最大超出量的显著性，因此必须将局部p值修正为全局p值。一个成熟的框架——根植于Davies的工作，并由Gross和Vitells在粒子物理中加以实际应用——为连续参数扫描提供了这种修正。然而，某些方面仍存在争议。“别处”的定义是模糊的，其范围可以从单一分布延伸到整个实验项目。此外，经典框架的解析捷径并不适用于基于机器学习（ML）的搜索，后者受到一种隐式的后选择偏差影响，该偏差取决于模型的分辨率和表达能力。本综述涵盖了对撞机新物理搜索中的LEE，从其统计基础到这些现代挑战。

    arXiv:2611.34888v1 Announce Type: cross  Abstract: The look-elsewhere effect (LEE) arises when the location of an effect in parameter space is not specified in advance: scanning many locations inflates the significance of the largest excess, and the local $p$-value must be corrected into a global one. A well-established framework, rooted in the work of Davies and operationalised for particle physics by Gross and Vitells, provides this correction for scans over continuous parameters. Some aspects, however, remain unsettled. The definition of "elsewhere" is ambiguous and ranges from a single distribution to an entire experimental programme. Moreover, the analytical shortcuts of the classical framework do not carry over to searches based on machine learning (ML), which are affected by an implicit post-selection bias that depends on the model's resolution and expressiveness. This review covers the LEE in searches for new physics at colliders, from its statistical foundations to these moder
    
[^3]: 递归熵风险下可证明高效的离线折扣强化学习

    Provably Efficient Offline Discounted Reinforcement Learning under Recursive Entropic Risk

    [https://arxiv.org/abs/2611.34840](https://arxiv.org/abs/2611.34840)

    本文提出了首个在递归熵风险度量下离线强化学习的可证明高效算法 ERM-VI-LCB，并给出匹配的样本复杂度上下界，同时覆盖风险厌恶与风险追求两种情形。

    

    我们研究了有限折扣无限时域马尔可夫决策过程（MDP）中基于递归熵风险度量（ERM）的风险敏感强化学习，该度量由参数 $\beta\neq 0$ 刻画。我们同时考虑了风险厌恶（$\beta>0$）和风险追求（$\beta<0$）两种设定。我们研究离线强化学习场景，即智能体获得一个固定数据集，并希望利用该数据集学习一个近似最优的策略。我们提出了一种基于模型的算法，称为 ERM 值迭代下置信界（ERM-VI-LCB），并推导出其样本复杂度的 PAC 型界。我们还建立了相应的下界，证明上界中所断言的对 $|\beta|/(1-\gamma)$ 的指数依赖在最坏情况下是不可避免的。所提出的界为递归 ERM 下的离线强化学习在风险厌恶和风险追求两种情形下提供了首个严格的样本复杂度保证。

    arXiv:2611.34840v1 Announce Type: new  Abstract: We study risk-sensitive reinforcement learning in finite discounted infinite-horizon Markov Decision Processes with recursive entropic risk measures (ERM), parameterized by $\beta\neq 0$. We consider both risk-averse ($\beta>0$) and risk-seeking ($\beta<0$) settings. We consider the offline RL setting, where the agent is provided with a fixed dataset and wishes to learn a near-optimal policy using this dataset. We introduce a model-based algorithm, called ERM Value Iteration Lower Confidence Bound (ERM-VI-LCB) and derive a PAC-type bound on its sample complexity. We also establish a corresponding lower bound, demonstrating that exponential dependence on $|\beta|/(1-\gamma)$ asserted by the upper bound is unavoidable in the worst case. The presented bounds provide the first rigorous sample complexity guarantees for offline RL under recursive ERM across both risk-averse and risk-seeking regimes.
    
[^4]: 基于时滞嵌入的相依数据条件独立图学习

    Conditional Independence Graph Learning from Dependent Data Via Time-Delay Embedding

    [https://arxiv.org/abs/2611.34616](https://arxiv.org/abs/2611.34616)

    提出一种基于时滞嵌入的时域方法，将相依时间序列的条件独立图估计转化为多属性随机向量图估计问题，突破了以往频域方法仅适用于短程相依数据的限制，可同时处理短程相依和长程相依的高维平稳多元高斯时间序列。

    

    我们研究高维平稳多元高斯时间序列的条件独立图（CIG）推断问题，该时间序列可以是短程相依（SRD）或长程相依（LRD）的。在时间序列图中，向量序列的每个分量由一个不同的节点表示，分量之间的关联由相应节点之间的边表示。我们考虑一种基于时滞嵌入的时域方法，将问题转化为随机向量的多属性图估计问题，其中图的每个节点都关联一个随机向量。对于嵌入维度 $d$，每个节点所关联的 $(d+1)$ 维随机向量由一个时间序列分量及其 $d$ 个延迟副本组成，延迟为不超过 $d \ge 1$ 的整数延迟。以往对相依数据的图估计研究多采用频域方法，但其分析仅限于短程相依数据。

    arXiv:2611.34616v1 Announce Type: cross  Abstract: We consider the problem of inferring the conditional independence graph (CIG) of a high-dimensional stationary multivariate Gaussian time series which could be short-range dependent (SRD) or long-range dependent (LRD). In a time series graph, each component of the vector series is represented by a distinct node, and associations between components are represented by edges between the corresponding nodes. We consider a time-domain approach based on time-delay embedding, casting the problem as one of multi-attribute graph estimation for random vectors where a vector is associated with each node of the graph. For an embedding dimension $d$, at each node the associated $(d+1)$-dimensional random vector consists of a time series component and its $d$ delayed copies, with integer delays up to $d \ge 1$. Graph estimation from dependent data has been investigated in the past using frequency-domain methods whose analysis is restricted to SRD da
    
[^5]: 过参数化极限下物理信息神经网络二阶优化方法的全局收敛性

    Global Convergence of Second-Order Methods for Physics-Informed Neural Networks in the Overparameterized Limit

    [https://arxiv.org/abs/2611.34452](https://arxiv.org/abs/2611.34452)

    本文首次为物理信息神经网络训练中的正则化牛顿法和Gauss-牛顿法建立了全局收敛理论，证明了在过参数化极限下训练动力学以显式速率概率收敛到确定性极限方程的解。

    

    求解高维偏微分方程（PDEs）是许多工程与科学领域的核心任务。物理信息神经网络（PINNs）和深度Galerkin方法通过最小化与PDE算子对应的残差以及外部约束，有效地利用机器学习模型来逼近已知PDE的解。尽管二阶优化器在训练神经网络求解PDE方面日益流行，并在速度和精度上都优于其一阶方法，但训练此类模型的收敛理论基本仅限于一阶方法。在本文中，我们填补了这一空白，为正则化牛顿法和Gauss-牛顿法建立了收敛性分析。当隐藏单元数量趋于无穷时，训练动力学以概率收敛（并具有显式收敛速率）到某个确定性极限方程的解。

    arXiv:2611.34452v1 Announce Type: cross  Abstract: Solving high-dimensional partial differential equations (PDEs) constitutes a core task in many engineering and scientific disciplines. Physics-informed neural networks and the deep Galerkin method effectively leverage machine learning models to approximate solutions of known PDEs by minimizing residuals corresponding to the PDE operator alongside external constraints. Although second-order optimizers are becoming increasingly popular for training neural networks (NNs) to solve PDEs and empirically outperform their first-order counterparts in both speed and accuracy, the convergence theory for training such models is essentially confined to first-order methods. In this paper, we close this gap and develop a convergence analysis for the regularized Newton and Gauss-Newton methods. As the number of hidden units tends to infinity, the training dynamics converge in probability, with explicit rates, to the solution of a deterministic limit e
    
[^6]: 基于生成模型的变点检测

    Change Point Detection with Generative Models

    [https://arxiv.org/abs/2611.34228](https://arxiv.org/abs/2611.34228)

    该论文提出利用生成模型对时间序列窗口进行数据增广，突破了基于窗口的变点检测中“样本量瓶颈”的根本局限，显著提升了变点检测的准确性与质量。

    

    变点检测（CPD）方法旨在识别时间序列属性中的突变，这是众多领域中的一个关键问题。一大类变点检测方法依赖于计算变点检测分数，即两个相邻样本（称为窗口）之间的差异或散度。基于窗口的变点检测方法存在一个根本性局限，即“样本量瓶颈”。为了确保时间精度，窗口必须很小；然而，小样本无法为目前用于计算变点检测分数的分类器和深度学习模型提供充分的统计证据，导致分数方差过大和频繁的误预测。我们的研究通过探索生成模型进行窗口增广来解决这一问题，有效地扩展了可用数据以提高变点检测质量。在合成数据和真实数据上的实验表明，所提出的方法显著提升了变点检测性能。

    arXiv:2611.34228v1 Announce Type: new  Abstract: Change point detection (CPD) methods aim to identify abrupt changes in the properties of time series, which is a key problem in numerous fields. A major class of CPD approaches relies on computing a CPD score, which is either a discrepancy or divergence between two subsequent samples, called windows. A fundamental limitation of window-based CPD is the 'sample size bottleneck.' To ensure temporal precision, windows must be small; however, small samples provide insufficient statistical evidence for the classifiers and deep-learning models currently used to compute CPD scores. This leads to high variance in the scores and frequent mispredictions. Our study tackles this issue by exploring generative models for window augmentation, effectively expanding available data to improve CPD quality. Experiments on synthetic and real data demonstrate that the proposed approach significantly enhances CPD. Specifically, we show that this approach allows
    
[^7]: 从bulk数据预测细胞类型特异性表达虽困难但仍然有用

    Cell-Type-Specific Expression Is Hard to Predict from Bulk, but It Is Still Useful

    [https://arxiv.org/abs/2611.34166](https://arxiv.org/abs/2611.34166)

    该论文通过推导闭式精度界从理论上解释了为何从bulk RNA数据预测细胞类型比例容易而预测细胞类型特异性表达困难，并证明尽管CTSE难以估计，对其进行目标性反卷积在非小细胞肺癌总生存期预测中仍具有预后价值。

    

    基于单细胞信息的反卷积方法能够为临床bulk RNA队列提供细胞类型分辨的特征。现有大多数方法预测K个组成细胞类型（CTs）的比例 π_k。较少的方法预测细胞类型 k 中基因 g 的细胞类型特异性表达（CTSE）μ_kg，它能捕捉比例所遗漏的细胞状态，但近期的基准测试发现这一任务要困难得多。我们用闭式精度界解释了这一差距。对于CTSE，我们推导了逐基因线性估计器的R²上限；对于比例，跨基因汇集证据的最优线性估计器具有闭式R²，且其可计算的界可预测哪些细胞类型是可学习的。我们研究了106个经生物学整理的（细胞类型，基因）对的目标反卷积，这些对在事后被发现具有更高的可学习性评分。在非小细胞肺癌的总生存期分析中，估计难度并不能预测这些目标位点的预后价值。然而，稀疏生存模型……

    arXiv:2611.34166v1 Announce Type: new  Abstract: Single-cell-informed deconvolution provides cell-type-resolved features for clinical bulk RNA cohorts. Most existing methods predict the proportions $\pi_k$ of the $K$ constituent cell types (CTs). Fewer predict cell-type-specific expression (CTSE) $\mu_{kg}$ of gene $g$ in CT $k$, which can capture cell states that proportions miss, but recent benchmarks find it much harder. We explain this gap with closed-form accuracy bounds. For CTSE, we derive an $R^2$ ceiling for per-gene linear estimators. For proportions, the best linear estimator, which pools evidence across genes, has a closed-form $R^2$, and a computable lower bound on it predicts which CTs are learnable. We study targeted deconvolution of 106 biologically curated (CT, gene) pairs, found post hoc to have higher learnability scores. On overall survival in non-small cell lung cancer, estimation difficulty does not predict prognostic value across these targets. Yet a sparse survi
    
[^8]: 通过希尔伯特空间提升的概率测度上多目标优化的加速近端梯度方法

    Accelerated Proximal Gradient Methods for Multiobjective Optimization over Probability Measures via Hilbert-Space Lifting

    [https://arxiv.org/abs/2611.34073](https://arxiv.org/abs/2611.34073)

    本文提出基于希尔伯特空间提升的多目标Wasserstein近端梯度方法（MWPG）及其加速版本（MWAPG），通过隐式Wasserstein梯度步求解概率测度上KL散度的Pareto多目标优化问题，并提供严格的离散时间收敛保证。

    

    许多采样和推断任务需要一个能够平衡多个目标的单一分布。我们将该任务表述为在概率测度上对Kullback-Leibler (KL)散度进行Pareto意义下的多目标优化。粒子方法MWGraD（Nguyen等，2025）使用显式Wasserstein梯度步，而其加速版本A-MWGraD（Nguyen等，2026）缺乏离散时间保证。我们提出了使用隐式Wasserstein梯度步的多目标Wasserstein近端梯度方法（MWPG），以及其加速版本——具有严格离散时间保证的多目标Wasserstein加速近端梯度方法（MWAPG）。我们的核心思想是希尔伯特空间提升：每个测度被表示为希尔伯特空间中一个映射对固定基础测度的推前。该问题由此转化为关于传输映射的约束多目标复合优化问题，从而可以应用加速近端梯度方法求解。

    arXiv:2611.34073v1 Announce Type: cross  Abstract: Many sampling and inference tasks require a single distribution that balances several targets. We formulate this task as multiobjective optimization of Kullback--Leibler (KL) divergences over probability measures, in the Pareto sense. The particle method MWGraD (Nguyen et al., 2025) uses explicit Wasserstein gradient steps, and its accelerated version A-MWGraD (Nguyen et al., 2026) lacks discrete-time guarantees. We propose the multiobjective Wasserstein proximal gradient (MWPG), which uses implicit Wasserstein gradient steps, and its accelerated version, multiobjective Wasserstein accelerated proximal gradient (MWAPG), with rigorous discrete-time guarantees. Our key idea is a Hilbert-space lifting: each measure is represented as the pushforward of a fixed base measure by a map in a Hilbert space. The problem then becomes a constrained multiobjective composite problem over transport maps, to which accelerated proximal gradient methods 
    
[^9]: 无需噪声知识的渐近有效有限脉冲响应系统辨识

    Asymptotically Efficient Finite Impulse Response System Identification without Noise Knowledge

    [https://arxiv.org/abs/2611.34039](https://arxiv.org/abs/2611.34039)

    提出了一类无需噪声分布先验知识的自适应估计器（eMLE与epMLE），通过核密度估计和一致性初始估计的残差来近似似然函数，在未知或重尾噪声下实现了有限脉冲响应系统辨识的渐近有效性。

    

    在未知或重尾噪声条件下对有限脉冲响应（FIR）系统进行高效辨识，一直是系统识别领域的一个难题。经典估计方法，包括最小二乘法（LS）和最小绝对偏差（LAD）等稳健替代方法，在噪声特性偏离其基本假设时，往往对异常值敏感或统计效率低下。为解决这一问题，我们提出了一类新颖的自适应估计器，无需噪声分布的先验知识即可实现渐近有效性。我们的方法利用核密度估计构建了两类非参数极大似然型估计器：估计极大似然估计器和估计轮廓似然极大似然估计器。这些估计器利用来自一致性初始估计的残差来近似似然函数，从而避免了对噪声的参数化假设。我们建立了理论保证，表明在……（摘要在此处被截断）

    arXiv:2611.34039v1 Announce Type: cross  Abstract: Efficient identification of finite impulse response systems under unknown or heavy-tailed noise remains a challenge in system identification. Classical estimators, including least squares (LS) and robust alternatives like least absolute deviations (LAD), are often sensitive to outliers or statistically inefficient when noise characteristics deviate from their underlying assumptions. To address this, we propose a novel class of adaptive estimators that achieves asymptotic efficiency without requiring prior knowledge of the noise distribution. Our approach leverages kernel density estimation to construct two nonparametric maximum likelihood-type estimators: the estimated maximum likelihood estimator (eMLE) and the estimated profiled MLE (epMLE). These estimators approximate the likelihood using residuals from consistent initial estimates, avoiding parametric assumptions on the noise. We establish theoretical guarantees showing that, unde
    
[^10]: BReD：用于稳定低比特 EMA 优化器状态的块重放抖动方法

    BReD: Block Replay Dithering for Stable Low-Bit EMA Optimizer States

    [https://arxiv.org/abs/2611.33951](https://arxiv.org/abs/2611.33951)

    提出 BReD 方法，通过将确定性种子重放与分块共享抖动值相结合的块重放抖动技术，减少 EMA 优化器状态迭代量化中的舍入偏差并控制舍入方差，从而解决误差累积问题、稳定大规模训练。

    

    优化器状态的低比特存储可以大幅减少大规模训练的内存占用。然而在实践中，这些状态并非只被量化一次：在每个训练步骤中，它们都会被反量化、更新、重新量化并写回。这种迭代式的量化流程可能会破坏训练的稳定性。我们识别出了导致这种不稳定性的一个失效模式：对于通过指数移动平均（EMA）更新的状态，量化误差会被递归地引入后续更新中。因此，有偏误差会随时间不断累积，而高方差误差则会被放大，尤其是当 EMA 衰减因子接近 1 时。为了解决这种误差累积问题，我们提出了 BReD（块重放抖动），一种在量化过程中减少舍入偏差并控制舍入方差的方法。BReD 通过将确定性种子重放与分块共享的抖动值相结合，将经典的减法抖动技术适配到了优化器状态量化中。

    arXiv:2611.33951v1 Announce Type: new  Abstract: Low-bit storage of optimizer states can substantially reduce the memory footprint of large-scale training. In practice, however, these states are not quantized once: at every training step, they are dequantized, updated, requantized, and written back. This iterative quantization pipeline can destabilize training. We identify a failure mode behind this instability: for states updated via exponential moving averages (EMAs), quantization errors are recursively fed into subsequent updates. As a result, biased errors accumulate over time while high-variance errors are amplified, particularly when the EMA decay factor is close to 1. To address this error accumulation, we propose BReD (Block Replay Dithering), a method that reduces rounding bias and controls rounding variance during quantization. BReD adapts classical subtractive dithering to optimizer-state quantization by combining deterministic seed replay with block-wise shared dither value
    
[^11]: 线性因果边界的图形判定准则

    A Graphical Criterion for Linear Causal Bounds

    [https://arxiv.org/abs/2611.33950](https://arxiv.org/abs/2611.33950)

    本文针对每个混杂组件最多含两个隐藏原因的因果图，给出了判断尖锐边界能否通过求解单个线性规划得到的充分必要条件，并实现了Python包partialid以计算高效的方式获得尖锐边界。

    

    在存在未观测混杂因素的情况下，因果效应和反事实的值通常只能通过观测数据进行部分识别。对于离散值变量，在特定条件下，可以通过求解单个线性规划来推导出尖锐边界，这使得许多因果问题在计算上是可处理的。本文针对每个混杂组件最多包含两个隐藏原因的因果图，给出了对于任意（正）输入概率表，尖锐边界等于单个线性规划最优解的充分必要条件。该算法已在一个名为"partialid"的新Python包中实现，允许用户组合假设、查询和数据，以计算高效的方式获得尖锐边界。

    arXiv:2611.33950v1 Announce Type: cross  Abstract: The values of causal effects and counterfactuals are typically only partially identified from observational data in the presence of unobserved confounding. With discretely-valued variables, under certain conditions, sharp bounds can be derived by solving one linear program. This is then computationally tractable for many causal questions. In this paper, for every diagram in which each confounded component has at most two hidden causes, we give necessary and sufficient conditions for the sharp bounds to be equal to the optimum of one linear program for every (positive) input probability table. This algorithm is implemented in a new Python package, called "partialid", that allows users to compose assumptions, queries, and data to obtain sharp bounds in a computationally tractable manner.
    
[^12]: CondUQ：基于分布匹配的校准条件不确定性量化

    CondUQ: Calibrated Conditional Uncertainty Quantification Using Distribution Matching

    [https://arxiv.org/abs/2611.33939](https://arxiv.org/abs/2611.33939)

    该论文提出 condUQ 方法，通过将误差模型条件化于可观测因素（如输入质量和运行条件）进行分布匹配，在不修改甚至黑盒的现有系统之上实现准确可靠的条件不确定性量化。

    

    许多应用，尤其是可靠性和安全性关键的应用，需要对基于模型或学习的测量、估计和预测系统的输出进行准确的不确定性量化。此类系统的误差往往随输入质量和运行条件等因素变化，因此无法用单一的 高斯模型准确描述。尽管高斯混合模型和校准分位数可以近似此类误差分布，但它们的无条件形式无法捕捉误差行为对这些因素的依赖关系。因此，我们提出了 condUQ，一种将误差模型条件化于可观测因素的方法，以实现准确可靠的不确定性量化。condUQ 构建于现有系统之上而无需修改该系统，即使系统是黑盒也能适用。首先，condUQ 将可观测因素映射到一个潜在的质量指数，使得共享相同 QI 的误差服从相同的分布……（摘要在此处被截断）

    arXiv:2611.33939v1 Announce Type: new  Abstract: Many applications, especially reliability- and safety-critical ones, require accurate uncertainty quantification for the outputs of model- or learning-based measurement, estimation, and prediction systems. The errors of such systems often vary with factors such as input quality and operating conditions, and thus cannot be accurately described by a single Gaussian model. Although Gaussian mixture models and calibrated quantiles can approximate such error distributions, their unconditional forms do not capture how the error behavior depends on these factors. Hence, we propose condUQ, an approach that conditions error models on observable factors for enabling accurate and reliable uncertainty quantification. condUQ is built on top of an existing system without modifying it and applies even when the system is a black box. First, condUQ maps the observable factors to a latent quality index (QI) such that errors sharing the same QI follow a co
    
[^13]: 面向可信多视图分类的相关性感知证据融合

    Correlation-Aware Evidential Fusion for Trusted Multi-View Classification

    [https://arxiv.org/abs/2611.33900](https://arxiv.org/abs/2611.33900)

    提出了相关性感知证据融合框架CAFE，通过显式建模视图间的相关性，避免重复计算冗余证据、强化互补信息，从而提升可信多视图分类的性能和不确定性估计的校准质量。

    

    可信多视图分类旨在通过整合来自多个视图的信息，做出准确且可靠的预测。现有方法通常为每个视图估计类别证据和不确定性，并通过聚合规则组合由此产生的观点。然而，这些方法通常将不同视图视为独立的信息源，忽略了视图之间的相关性。高度相关的视图往往会产生相似的证据，将它们视为独立信息源可能会重复计算一致的信息并赋予其不成比例的影响力，同时削弱提供额外信息的互补性证据。因此，这种不考虑相关性的融合不仅可能限制分类性能，还可能产生校准不佳的不确定性估计。为解决这一问题，我们提出了相关性感知证据融合（CAFE）框架，用于可信多视图分类。CAFE通过对……

    arXiv:2611.33900v1 Announce Type: cross  Abstract: Trusted multi-view classification aims to make accurate and reliable predictions by integrating information from multiple views. Existing methods typically estimate class evidence and uncertainty for each view and combine the resulting opinions through aggregation rules. However, they generally treat different views as independent information sources, overlooking the correlations among them. Highly correlated views tend to produce similar evidence, and treating them as independent sources may repeatedly count consistent information and give it disproportionate influence, while weakening complementary evidence that provides additional information. Consequently, such correlation-unaware fusion may not only limit classification performance but also produce miscalibrated uncertainty estimates. To address this issue, we propose Correlation-Aware Fusion of Evidence (CAFE), a framework for trusted multi-view classification. CAFE applies compl
    
[^14]: 为学习型加权FDR控制认证充分的高斯相依性条件

    Certifying a Sufficient Gaussian Dependence Regime for Learned Weighted FDR Control

    [https://arxiv.org/abs/2611.33892](https://arxiv.org/abs/2611.33892)

    该论文提出利用参考观测认证单侧高斯检验的充分相依性条件，使学习得到的加权Benjamini-Hochberg程序在满足该条件时能严格控制错误发现率（FDR），并给出了匹配的极小极大样本量界。

    

    从信息丰富的辅助数据中学习假设权重可以提升多重检验的功效。即使权重是独立于检验而学习的，加权Benjamini-Hochberg方法的有效性仍取决于权重与检验的联合分布。我们利用参考观测来认证单侧高斯检验的一个充分相依性条件。一个针对不确定相关符号的交集-并集检验决定是应用加权Benjamini-Hochberg方法，还是在调整后的水平上应用预先指定的后备方法。在认证与检验实现条件分离的前提下，该程序对每一个满足所述抽样假设和权重约束的高斯相关矩阵都能控制错误发现率（FDR）。我们还建立了统一认证的边界性障碍，并针对最坏情形配对集合和全配对问题给出了匹配的极小极大样本量界。

    arXiv:2611.33892v1 Announce Type: cross  Abstract: Learning hypothesis weights from informative auxiliary data can improve the power of multiple testing. Even when the weights are learned independently of the tests, the validity of weighted Benjamini-Hochberg depends on their joint distribution. We use reference observations to certify a sufficient dependence condition for one-sided Gaussian tests. An intersection-union test of uncertain correlation signs determines whether to apply weighted Benjamini-Hochberg or a prespecified fallback at an adjusted level. Under conditional separation of certification and testing, the procedure controls the false discovery rate (FDR) for every Gaussian correlation matrix satisfying the stated sampling assumptions and weight constraints. We establish a boundary obstruction to uniform certification and matching minimax sample-size bounds for worst-case pair sets and the all-pairs problem. In the parameter ranges of the matching bounds, the minimax samp
    
[^15]: RHOPE：利用大语言模型增强的复杂上下文信息自适应实验设计

    RHOPE: LLM-Augmented Adaptive Experimental Design with Complex Contextual Information

    [https://arxiv.org/abs/2611.33767](https://arxiv.org/abs/2611.33767)

    该论文提出RHOPE框架，利用大语言模型将复杂上下文信息转化为代理结果，将有限批次自适应实验设计形式化为后验马尔可夫决策过程，并通过样本增强与决策前预观察代理结果两种方式提升估计精度和分配决策效果。

    

    实验设计将有限的测量资源分配到候选处理方案上，以学习其期望结果。自适应设计会随着证据的积累不断优化分配策略，从而提升估计精度和处理方案的选择效果。上下文信息可以为这些决策提供依据，但如何利用非结构化或高维度的上下文信息仍然具有挑战性。我们使用大语言模型（LLM）将复杂的上下文信息转化为代理结果，并提出了面向预测驱动实验的残差视野优化方法（RHOPE）——一个有原则的贝叶斯自适应实验框架，它将真实结果与LLM生成的代理结果相结合，并将有限批次的设计问题形式化为后验马尔可夫决策过程（MDP）。我们的方法通过两种方式利用代理结果：样本增强可以提高估计精度，而在每次分配决策之前观察代理结果则能提供前瞻性信息。合成实验和半合成……

    arXiv:2611.33767v1 Announce Type: cross  Abstract: Experimental design allocates limited measurement effort across candidate treatments to learn their expected outcomes. Adaptive designs refine allocations as evidence accumulates, improving estimation and treatment selection. Contextual information can inform these decisions, but exploiting unstructured or high-dimensional context remains challenging. We use large language models (LLMs) to transform complex context into proxy outcomes and propose Residual Horizon Optimization for Prediction-Powered Experimentation (RHOPE), a principled Bayesian adaptive experimentation framework that combines true outcomes with LLM-generated proxy outcomes and formulates finite-batch design as a posterior Markov decision process (MDP). Our methodology uses proxy outcomes in two ways: sample augmentation improves estimation precision, while observing proxies before each allocation decision provides advance information. Synthetic experiments and semi-syn
    
[^16]: 通过通用压缩理解双下降现象

    Understanding Double Descent through Universal Compression

    [https://arxiv.org/abs/2611.33759](https://arxiv.org/abs/2611.33759)

    该论文提出以通用压缩的全新理论视角解释深度学习中的双下降现象，将最大似然超额测试损失的上界分解为近似与偏移项、极小极大批次遗憾、先验代价和MLE与贝叶斯差距四个可解释的组成部分。

    

    许多深度学习模型表现出双下降现象，即测试损失在插值阈值附近出现急剧峰值，然后随着模型规模的进一步增大而再次下降，这与经典偏差-方差权衡所预测的U形曲线相反。双下降现象已在简单的经典模型中被发现并得到解释，但尽管经过了大量实证研究，其在现代深度学习中仍未被充分理解。我们通过通用压缩这一全新视角来解释双下降现象。具体而言，我们将最大似然的超额测试损失的上界表示为四项之和：近似与偏移项、极小极大批次遗憾、先验代价，以及最大似然估计（MLE）与贝叶斯方法之间的差距。在精确后验采样条件下，我们进一步证明，在设定良好的情况下，预期的MLE与集成方法之间的差距可对MLE与贝叶斯方法之间的差距进行下界化。在我们的实验中，采用高斯先验的SGLD对贝叶斯预测器进行近似，与SGD不同，……

    arXiv:2611.33759v1 Announce Type: cross  Abstract: Many deep learning models exhibit a double-descent phenomenon, where test loss exhibits a sharp peak near the interpolation threshold and then descends a second time as model size grows further, contrary to the U-shape predicted by the classical bias--variance tradeoff. The double-descent phenomenon has been found and explained in toy classical models, but is not well-understood in modern deep learning despite extensive empirical study. We interpret double descent through the novel lens of universal compression. Specifically, we upper bound the maximum-likelihood excess test loss as a sum of four terms: an approximation-and-shift term, a minimax batch regret, a prior cost, and an MLE-vs-Bayes gap. Under exact posterior sampling, we further show that the expected MLE-vs-ensemble gap lower-bounds the MLE-vs-Bayes gap in the well-specified setting. In our experiments, a Gaussian-prior SGLD approximation to the Bayes predictor, unlike SGD,
    
[^17]: 学会推断：通过元学习实现统计推断

    Learning to Infer: Statistical Inference through Meta Learning

    [https://arxiv.org/abs/2611.33742](https://arxiv.org/abs/2611.33742)

    该论文将统计决策理论与元学习相结合，提出残差循环深度集合网络架构，使神经网络能从模拟任务中学习推断规则并无需参数更新地应用于新数据，同时为学习到的规则建立了渐近有效性、模型选择一致性和高维变量选择等频率学派统计保证。

    

    我们研究用于统计推断的元学习，其中神经网络从模拟任务中学习推断规则，并将其应用于新数据集而无需参数更新，从而提供了一种形式的上下文学习。我们借鉴统计决策理论来设计训练目标，使其总体最小化器为贝叶斯规则，并利用其与经典推断程序之间的联系，为学习到的规则建立频率学派的统计保证。经典推断程序的结构也启发了我们神经网络架构的设计，该架构称为残差循环深度集合，它以置换不变的方式反复聚合观测之间的信息，并通过残差更新来精炼其内部状态。在适当的正则性和训练条件下，我们建立了参数估计的渐近有效性、模型选择的一致性，以及高维情形下的变量选择保证。

    arXiv:2611.33742v1 Announce Type: cross  Abstract: We study meta learning for statistical inference, in which neural networks learn inference rules from simulated tasks and apply them to new datasets without parameter updates, providing a form of in-context learning. We draw on statistical decision theory to design training objectives whose population minimizers are Bayes rules and exploit their connections to classical inference procedures to establish frequentist guarantees for the learned rules. The structure of classical inference procedures also motivates the design of our neural network architecture, called Residual Recurrent Deep Sets (RRDS), which repeatedly aggregates information across observations in a permutation-invariant manner and refines its internal state through residual updates. Under suitable regularity and training conditions, we establish asymptotic efficiency in parameter estimation, consistency in model selection, and variable selection guarantees in high-dimens
    
[^18]: 不仅是问题的数量，还有来源的数量：从早期结果规划LLM评估的精度

    How Many Sources, Not Just How Many Problems: Planning the Precision of LLM Evaluations from Earlier Results

    [https://arxiv.org/abs/2611.33622](https://arxiv.org/abs/2611.33622)

    该论文提出利用试点数据或早期评估结果，结合经典设计效应与组内相关系数，在正式评估前预测当评估预算集中于更少来源时LLM评估分数标准差的增幅，从而事先规划评估的精度。

    

    当基准测试中的问题共享来源时（例如同一模板的变体或来自同一代码仓库的任务），评估预算必须同时固定来源的数量和问题的数量。我们探讨这样一个问题：试点数据或早期评估结果能否在正式评估运行之前，预测当相同的预算集中于更少来源时，分数标准差（SD）会增长多少。我们使用经典的设计效应（design effect）方法，其中组内相关系数在与用于检验预测的数据不相交的数据上进行估计。在GSM-Symbolic上，以10个模板×10个变体方式抽取的100个问题的标准差为9.2个百分点，而100个不同模板对应的标准差仅为2.7个百分点；在纳入实际抽样计划后，基于15个其他问题的试点数据预测出的比值为3.49，与在留出问题上实测的3.52非常接近（20次划分的中位数；单次划分的误差为-13%至+16%）。在SWE-bench Verified上，随机选取的10个公开提交预测出了中位数比值……（摘要在此处被截断）

    arXiv:2611.33622v1 Announce Type: cross  Abstract: When benchmark problems share sources, such as variants of one template or tasks from one repository, an evaluation budget must fix the number of sources as well as the number of problems. We ask whether pilot data or earlier evaluation results can predict, before an evaluation is run, how much the standard deviation (SD) of the score grows when the same budget is concentrated on fewer sources. We use the classical design effect with an intraclass correlation estimated on data disjoint from those used to check the prediction. On GSM-Symbolic, 100 problems drawn as 10 templates x 10 variants have an SD of 9.2 pp against 2.7 pp from 100 templates; pilot data on 15 other questions predict the ratio as 3.49 against 3.52 measured on held-out questions (medians over 20 splits; single splits err by -13% to +16%), once the actual sampling plan is included. On SWE-bench Verified, 10 randomly selected public submissions predict the median ratio 
    
[^19]: 基于KL不确定性集的鲁棒强化学习样本复杂度的改进

    Improved Sample Complexity of Robust Reinforcement Learning with KL Uncertainty Sets

    [https://arxiv.org/abs/2611.33592](https://arxiv.org/abs/2611.33592)

    本文针对基于KL散度不确定性集的鲁棒强化学习，给出了更紧致且完备的样本复杂度上下界，填补了以往结果中的差距，并将下界覆盖范围扩展到完整的不确定性水平区间。

    

    本文研究了采用Kullback-Leibler（KL）散度定义的不确定性集的分布鲁棒强化学习。目标是找到一个能够最大化鲁棒价值函数的策略，该鲁棒价值函数定义为不确定性集中所有转移核对应的最坏情况值。在假设可以使用生成模型的前提下，我们旨在理解寻找ε-最优鲁棒策略所需的样本复杂度。文献中已知的最佳样本复杂度结果表明，上界与下界之间存在大小为 $\mathcal{O}\{\max\{p_\wedge^{-1}(1-\gamma)^{-1},(1-\gamma)^{-2}\}\}$ 的非平凡差距，其中 $p_\wedge$ 表示标称转移核的最小正转移概率，$\gamma$ 为折扣因子。更重要的是，现有关于下界的结果仅覆盖了有限范围的不确定性水平。在本文中，我们为基于KL不确定性集的鲁棒强化学习开发出更紧致且完备的上下界。

    arXiv:2611.33592v1 Announce Type: cross  Abstract: In this paper, we study distributionally robust reinforcement learning with Kullback--Leibler (KL) divergence-defined uncertainty sets. The goal is to find a policy that maximizes the robust value function, defined as the worst-case value over all transition kernels in the uncertainty set. Assuming access to a generative model, we aim to understand the sample complexity of finding an $\epsilon$-optimal robust policy. The best-known sample complexity results in the literature show a non-trivial gap of $\mathcal{O}\{\max\{p_\wedge^{-1}(1-\gamma)^{-1},(1-\gamma)^{-2}\}\}$ between the upper and lower bounds, where $p_\wedge$ denotes the minimum positive transition probability of the nominal transition kernel, and $\gamma$ is the discount factor. More importantly, existing results on the lower bound only cover a limited range of the uncertainty level. In this paper, we develop tighter and complete upper and lower bounds for robust RL with K
    
[^20]: 摊销精确正交性：面向正交微调的漂移触发恢复方法

    Amortizing Exact Orthogonality: Drift-Triggered Restoration for Orthogonal Fine-Tuning

    [https://arxiv.org/abs/2611.33589](https://arxiv.org/abs/2611.33589)

    该论文提出 RTOFT 方法，用一阶更新 W(I+S) 取代逐步构造正交变换，仅在漂移 ‖S‖₂ 超过容差 τ 时才通过极分解恢复精确正交性，从而以漂移触发的方式摊销正交微调的计算开销。

    

    正交微调（OFT）通过正交变换来适配预训练权重，并保持其奇异值和行间角度，但现有方法在每次优化步骤中都需要构造精确或近似的正交变换。我们通过正交性缺陷 ‖AᵀA−I‖₂ 来衡量变换 A 偏离正交的程度。对于截断 Cayley-Neumann 级数的 OFTv2，我们证明截断阶数并不能决定缺陷大小：在默认阶数下，参考实现检查点中的分块缺陷跨越五个数量级。我们提出重启切线正交微调，它使用带斜对称矩阵 S 的权重 W 的一阶更新 W(I+S)，并且仅在漂移 ‖S‖₂ 达到容差 τ 时，才通过极分解恢复精确正交性。由于 I+S 的缺陷恰好为 ‖S‖₂²，因此只需在每次更新后读取漂移，并在……（摘要在此处被截断）

    arXiv:2611.33589v1 Announce Type: new  Abstract: Orthogonal finetuning (OFT) adapts a pretrained weight by an orthogonal transform and preserves its singular values and row angles, but existing methods construct an exact or approximate orthogonal transform at every optimization step. We measure how far a transform $A$ is from orthogonal by its orthogonality defect $\|A^\top A-I\|_2$. For OFTv2, which truncates a Cayley-Neumann series, we show that the truncation order does not determine the defect. At its default order, block defects in checkpoints from the reference implementation span five orders of magnitude. We propose Restarted Tangent Orthogonal Finetuning (RTOFT), which uses the first-order update $W(I+S)$ of a weight $W$ with a skew-symmetric $S$ and restores exact orthogonality by a polar decomposition only when the drift $\|S\|_2$ reaches a tolerance $\tau$. The defect of $I+S$ is exactly $\|S\|_2^2$. Therefore, reading the drift after every update and restoring before the ne
    
[^21]: 具有 $d^{3/2}\sqrt{n}$ 遗憾的对抗性赌博机凸优化

    Adversarial Bandit Convex Optimization with $d^{3/2}\sqrt{n}$ Regret

    [https://arxiv.org/abs/2611.33531](https://arxiv.org/abs/2611.33531)

    本文提出一种带高斯探索的在线牛顿步算法，将对抗性赌博机凸优化的遗憾上界从 $\widetilde O(d^{5/2}\sqrt n)$ 改进为 $\widetilde O(d^{3/2}\sqrt n)$，匹配了此前被猜想的最优速率，并将极小极大遗憾确定到 $d^{1/6}$ 因子以内。

    

    我们研究对抗性赌博机凸优化问题：在 $n$ 轮中的每一轮，学习者选择凸体 $\mathcal{K}\subset\mathbb{R}^d$ 中的一个点 $A_t$，并仅观测到该点处凸损失函数 $f_t:\mathcal{K}\to[0,1]$ 的含噪值。损失函数由对手选择，对手可以逐轮改变损失函数，并根据学习者过去的动作进行自适应调整。我们将该问题目前已知的最佳遗憾界从 $\widetilde O(d^{5/2}\sqrt n)$（Lattimore, 2020, 2026）改进为 $\widetilde O(d^{3/2}\sqrt n)$，这与 Bubeck 等人（2017）所猜测的速率相匹配。结合 Rajaraman 和 Han（2026）最近给出的 $\Omega(d^{4/3}\sqrt n)$ 下界，该结果将极小极大遗憾确定到 $d^{1/6}$ 因子和对数因子范围之内。该算法是带高斯探索的在线牛顿步方法。与以往同类方法不同，它使用高斯平滑后函数的梯度与 Hessian 矩阵的无偏估计……

    arXiv:2611.33531v1 Announce Type: cross  Abstract: We study adversarial bandit convex optimization, where in each of $n$ rounds a learner chooses a point $A_t$ in a convex body $\mathcal{K}\subset\mathbb{R}^d$ and observes only a noisy value of a convex loss $f_t:\mathcal{K}\to[0,1]$ at that point. The losses are chosen by an adversary, who may change them from round to round and adapt them to the learner's past actions. We improve the best known regret bound for this problem from $\widetilde O(d^{5/2}\sqrt n)$ (Lattimore, 2020, 2026) to $\widetilde O(d^{3/2}\sqrt n)$, which matches the rate conjectured by Bubeck et al. (2017). Together with the recent lower bound of $\Omega(d^{4/3}\sqrt n)$ due to Rajaraman and Han (2026), it determines the minimax regret up to a factor of $d^{1/6}$ and logarithmic factors. The algorithm is an online Newton step with Gaussian exploration. Unlike earlier methods of this kind, it uses unbiased estimates of the gradient and Hessian of the Gaussian smooth
    
[^22]: 论额外信息对老虎机学习的影响：更多并非总是更好

    On the Impact of Extra Information in Bandit Learning: More is Not Always Merrier

    [https://arxiv.org/abs/2611.33479](https://arxiv.org/abs/2611.33479)

    该研究揭示了一个反直觉现象：在随机老虎机学习中，额外信息可能使UCB算法的遗憾变差而非改善，并提出了一种能为广泛类别老虎机算法可证明地保证“非退化性”的元算法。

    

    通常认为更多的数据会提高学习性能，然而我们证明这一直觉在序列决策问题中可能出人意料地失效。在带有外生额外信息的随机老虎机设置下，我们探究免费且无偏的额外奖励样本是否必然能降低学习者的遗憾。我们引入了“额外信息下的非退化性”这一形式化概念，并证明广泛使用的置信上界算法（UCB）违反了这一性质，即存在某些良性的额外信息，反而使UCB产生严格高于无额外信息时的遗憾。我们将这种行为与汤普森采样和“先探索后承诺”算法进行对比，后者似乎能从相同的额外信息中受益。作为建设性贡献，我们提出了一种元算法，可证明地为一大类老虎机算法保证非退化性。我们还通过实证模拟来研究该元算法的表现。

    arXiv:2611.33479v1 Announce Type: cross  Abstract: More data is commonly assumed to improve learning performance, however, we show that this intuition can surprisngly fail in sequential decision-making. Considering a stochastic bandits setup with exogenous extra information, we ask if the free and unbiased extra reward samples are guaranteed to reduce the regret of the learner. We introduce a formal notion of `non-degradation under extra information' and demonstrate that the widely used Upper Confidence Bound algorithm (UCB) violates this property, i.e., there exist benign extra-information under which UCB incurs strictly higher regret than when run without extra information. We contrast this behaviour with Thompson Sampling and Explore-Then-Commit, which appear to benefit from the same extra information. Constructively, we propose a meta-algorithm that provably enforces non-degradation for a broad class of bandit algorithms. We also run empirical simulations to study our meta-algorith
    
[^23]: 具有对抗性奖励的线性马尔可夫决策过程中的差分隐私策略优化

    Differentially Private Policy Optimization in Linear MDPs with Adversarial Rewards

    [https://arxiv.org/abs/2611.33469](https://arxiv.org/abs/2611.33469)

    本文针对具有未知转移和自适应对抗性奖励的线性MDP，提出了一种结合批量化PPO与乐观策略评估的差分隐私策略优化算法，在联合差分隐私下同时保护轨迹与奖励数据，并实现了保留 $\widetilde{O}\left(K^{3/4}\right)$ 主导项的高概率遗憾界。

    

    个性化服务中的强化学习（RL）通常依赖于敏感的交互数据。出于保护这些数据的需求，我们研究了在全信息反馈下、具有未知转移和自适应对抗性线性奖励的分段式线性马尔可夫决策过程（MDP）中的差分隐私（DP）策略优化。这一设定带来了两个关键挑战：过去的轨迹会被重复用于策略评估，且奖励的选择可能依赖于先前发布的私有统计量。所提出的算法将批量化的近端策略优化（PPO）更新与乐观策略评估相结合，利用私有回归统计量和奖励估计，在联合差分隐私（JDP）下同时保护轨迹和奖励函数。在所有其他参数固定的情况下，我们的高概率遗憾界保留了关于回合数的主要 $\widetilde{O}\left(K^{3/4}\right)$ 依赖关系。

    arXiv:2611.33469v1 Announce Type: new  Abstract: Reinforcement learning (RL) in personalized services often relies on sensitive interaction data. Motivated by the need to protect these data, we study differentially private (DP) policy optimization in episodic linear Markov decision processes (MDPs) with unknown transitions and adaptive adversarial linear rewards under full-information feedback. This setting poses two key challenges: past trajectories are repeatedly reused for policy evaluation, and reward choices may depend on previously released private statistics. The proposed algorithm combines batched proximal policy optimization (PPO) updates with optimistic policy evaluation, using private regression statistics and reward estimates to protect both trajectories and reward functions under joint differential privacy (JDP). With all other parameters fixed, our high-probability regret bound retains the same leading $\widetilde{O}\left(K^{3/4}\right)$ dependence on the number of episod
    
[^24]: PAC学习采样：论语言生成的计算复杂性

    PAC Learning to Sample: On Computational Aspects of Language Generation

    [https://arxiv.org/abs/2611.33464](https://arxiv.org/abs/2611.33464)

    该论文将PAC学习框架与极限语言生成模型相结合，首次为语言生成任务提供了关于样本复杂度和时间复杂度的定量保证。

    

    语言模型的成功推动了从离散域分布中学习采样的理论研究。一个重要的抽象模型是由Kleinberg和Mullainathan（NeurIPS 2024）提出的“极限语言生成”模型，其中算法必须在经过有限但未指定数量的正例后从目标语言中进行生成。虽然该模型的积极结果出奇地具有普适性，但大多数保证都是定性的。在这项工作中，我们受统计学习理论和计算学习理论的启发，寻求关于样本复杂度和时间复杂度的定量保证。为此，我们研究了由De、Diakonikolas和Servedio（SODA 2015）在另一背景下提出的语言生成PAC模型，其中定义在{-1,1}^n上的布尔概念类C定义了目标语言：给定未知概念f ∈ C的均匀随机正例，学习者必须输出一个采样器，其输出……

    arXiv:2611.33464v1 Announce Type: cross  Abstract: The success of language models has motivated theoretical work on learning to sample from distributions on discrete domains. A prominent abstraction is the model of language generation in the limit, introduced by Kleinberg and Mullainathan (NeurIPS 2024), in which the algorithm must generate from the target language after some finite but unspecified number of positive examples. While its positive results are surprisingly general, most guarantees are qualitative. In this work, we seek quantitative guarantees on sample and time complexity, inspired by statistical and computational learning theory.   To this end, we study a PAC model of language generation, proposed (in a different context) by De, Diakonikolas, and Servedio (SODA 2015), in which a Boolean concept class $C$ over $\{-1,1\}^n$ defines the target languages: given uniformly random positive examples of an unknown concept $f \in C$, the learner must output a sampler whose output 
    
[^25]: 中心核距离：用于比较神经表征的无维度依赖误差界

    Centered Kernel Distance: Dimension-Free Error Bounds for Comparing Neural Representations

    [https://arxiv.org/abs/2611.33447](https://arxiv.org/abs/2611.33447)

    本文提出中心核距离（CKD）这一神经表征相似性度量，证明其存在无偏估计量且估计误差以 1/M + 1/N 的速度衰减，同时收敛速率与实验条件空间维度无关，从而克服了表征相似性估计中的维度灾难。

    

    中心核对齐（CKA）和表征相似性分析（RSA）等表征相似性度量的渐近误差已被充分刻画，但对于表征相似性的估计如何同时依赖于采样神经元数量（N）和实验条件数量（M），目前尚缺乏符合实际的误差界。我们研究了CKA和RSA的自然度量对应物，称之为中心核距离（CKD）。我们证明CKD存在无偏估计量，而包括CKA在内的几种流行的相似性度量则不存在无偏估计量。在温和的假设条件下，我们证明CKD的估计误差以 1/M + 1/N 的比例衰减。引人注目的是，该误差界中的收敛速率不依赖于实验条件空间的维度。因此，相似性估计不会受到维度灾难的影响，而维度灾难正困扰着高维统计中的类似问题。

    arXiv:2611.33447v1 Announce Type: cross  Abstract: The asymptotic error of representational similarity metrics such as centered kernel alignment (CKA) and representational similarity analysis (RSA) has been well-characterized, but there are no realistic bounds on how estimates of representational similarity $\textit{simultaneously}$ depend on the number of sampled neurons ($N$) and experimental conditions ($M$). We study the natural metric counterpart of CKA and RSA, which we call the Centered Kernel Distance (CKD). We show that CKD admits an unbiased estimator, whereas several popular similarity measures including CKA have no unbiased estimator. Under mild assumptions, we show that estimation error of CKD decays in proportion to $1/M + 1/N$. Strikingly, the rate in this bound does not depend on the dimensionality of the space of experimental conditions. Thus, similarity estimation does not suffer from the curse of dimensionality, which plagues similar problems in high-dimensional stat
    
[^26]: 基于半经典近似从非线性偏微分方程构造高斯先验

    Gaussian priors from nonlinear PDEs with the semiclassical approximation

    [https://arxiv.org/abs/2611.33400](https://arxiv.org/abs/2611.33400)

    本文提出利用半经典近似，通过在稳定解附近展开非线性PDE的变分能量或积分平方残差来构造高斯先验，从而将贝叶斯不确定性量化的概率数值方法拓展到非线性偏微分方程领域。

    

    arXiv:2611.33400v1 公告类型：cross 摘要：偏微分方程（PDE）的概率数值方法将控制方程的物理规律与观测数据相结合，把求解PDE视为一个贝叶斯逆问题。这使得人们能够在模型形式误差存在的情况下量化场重建中的不确定性。在大多数情况下，会给感兴趣的场分配一个高斯先验，并在无穷维设置下利用贝叶斯定理推导后验分布。虽然高斯分布已被证明非常适合线性PDE，但对非线性PDE的类似处理仍是一个悬而未决的问题。在本工作中，我们研究利用半经典近似，从变分能量或积分平方残差中构造用于概率数值方法的高斯先验。该先验通过在稳定解附近对PDE能量进行展开并揭示出高斯形式而得到。随后，我们分析了所得先验的不同特性，以及它们在……（摘要原文在此处截断）

    arXiv:2611.33400v1 Announce Type: cross  Abstract: Probabilistic numerical methods for partial differential equations (PDE) combines the governing physics with observations to treat solving the PDE as a Bayesian inverse problem. This allows techniques to quantify the uncertainty in the field reconstruction under model-form error. In the majority of cases, a Gaussian prior is assigned to the field of interest, and Bayes's theorem in the infinite-dimensional setting is used to derive the posterior. While Gaussians have been shown to be well-suited for linear PDEs, a similar treatment for nonlinear PDEs remains open-ended. In this work, study the use of the semiclassical approximation to construct Gaussian priors from variational energies or integrated square residuals for probabilistic numerics. The prior is derived by expanding the PDE energy about a stable solution and uncovering a Gaussian form. We then analyze the different characteristics of the resulting priors as they pertain to t
    
[^27]: 评估数据扰动下算法稳定性的集成方法

    Evaluating Ensemble Methods for Algorithmic Stability Under Data Perturbations

    [https://arxiv.org/abs/2611.33375](https://arxiv.org/abs/2611.33375)

    本文揭示了集成方法对分类器稳定性的双面效应：子装袋可显著提升不稳定分类器的稳定性，但会破坏本已稳定的分类器，其有限装袋引入的蒙特卡洛噪声可通过膨胀argmax方法缓解。

    

    标准分类算法通常在决策边界附近表现出敏感性，这可能加剧因对高影响样本子集过拟合而已经存在的不稳定性。集成学习常被用于稳定分类器，但我们证明它反而可能破坏那些本已稳定的分类器。成熟有效的子装袋方法可以带来稳定性方面的显著改进，我们在死因分类场景中验证了这一点。然而，其效果取决于扰动水平，并且对于稳定的分类器而言，有限数量的装袋会引入破坏稳定性的蒙特卡洛噪声，且该噪声随着装袋数量的增长而缩减缓慢。不过，这种不稳定性可以通过最近提出的膨胀argmax（inflated argmax）方法来缓解，该方法允许在模糊情况下进行多类预测。我们进一步研究了Rashomon集——即实现接近最优预测性能的不同模型的集合——在多大程度上……

    arXiv:2611.33375v1 Announce Type: cross  Abstract: Standard classification algorithms often exhibit sensitivity near decision boundaries, which can compound instability that already exists from overfitting to a subset of high-influence samples. Ensembling is commonly used to stabilize classifiers, but we show that it can destabilize those that are already stable. The well-established practice of subbagging can yield substantial improvements in stability, as we validate in the cause-of-death classification setting. However, its effect depends on the perturbation level, and for stable classifiers the finite number of bags introduces destabilizing Monte Carlo noise that shrinks only slowly as the number of bags grows. This instability can be mitigated, though, by the recently proposed inflated argmax, which allows for multi-class predictions in ambiguous cases. We further examine the extent to which the Rashomon set, the collection of distinct models that achieve near-optimal predictive p
    
[^28]: 语言模型微调中遗忘的两个时间尺度：快速的行为抑制与缓慢的表征漂移

    Two Timescales of Forgetting in Fine-tuning of Language Models: Fast Behavioral Suppression and Slow Representational Drift

    [https://arxiv.org/abs/2611.33302](https://arxiv.org/abs/2611.33302)

    本研究通过在随机层次模型上预训练并用受限规则子集微调transformer，发现语言模型微调中的遗忘具有两个时间尺度——表面行为被快速抑制，而支撑该行为的潜在表征则以缓慢漂移的方式逐渐丢失。

    

    当今，人工智能系统的核心流程是先在广泛的文本语料库上进行预训练，然后在特定任务或期望行为上进行微调。然而，对于在受限数据分布上进行微调如何改变预训练期间获得的表征，我们的理解仍然有限。尤其是，表面行为的抑制是否与支撑该行为的潜在表征的擦除同时发生，这一点仍不清楚。在这项工作中，我们研究了监督微调（SFT）：先在一个具有分层潜在结构的受控上下文无关文法——随机层次模型上预训练transformer模型，再使用其生成规则的一个受限子集进行微调。虽然在预训练期间，模型依据正确语法规则进行生成的能力与构建相应潜在变量的内部表征是同步发生的，但我们发现SFT表现出截然不同的动力学：(i)（摘要在此处被截断，后续内容缺失）

    arXiv:2611.33302v1 Announce Type: cross  Abstract: Today, a core pipeline of artificial intelligence systems is pretraining on a broad range of text corpora and then fine-tuning on specific tasks or desired behaviors. Yet we still have limited understanding of how fine-tuning on a restricted data distribution modifies representations acquired during pretraining. In particular, it remains unclear whether surface behavioral suppression happens together with the erasure of latent representations that support it. In this work, we study supervised fine-tuning (SFT) by pretraining transformers on a controlled context-free grammar with hierarchical latent structure, the Random Hierarchy Model (RHM), and fine-tuning with a restricted subset of its generative rules. While during pretraining the model's ability to generate with the correct grammar rules coincides with building the internal representations of the corresponding latent, we find that SFT exhibits a very different dynamics: (i) the g
    
[^29]: B-SLIM：用于异质性治疗效应的BART稀疏线性交互模型

    B-SLIM: BART Sparse Linear Interactions Model for Heterogeneous Treatment Effects

    [https://arxiv.org/abs/2611.33246](https://arxiv.org/abs/2611.33246)

    提出B-SLIM半参数因果推断模型，将灵活的BART预后建模与经全局-局部收缩先验正则化的稀疏线性治疗效应组件（含主效应和交互效应）相结合，兼顾模型灵活性、治疗效应可解释性以及对治疗效应调节因子的直接推断能力。

    

    估计依赖于协变量的治疗效应是个性化医疗中的核心挑战。现有方法通常要么采用完全参数化模型（存在模型误设的风险），要么使用灵活的机器学习方法（可能需要大样本才能检测到医疗场景中常见的微小治疗效应异质性）。我们提出了B-SLIM（BART稀疏线性交互模型），这是一种用于因果推断的半参数模型，它将灵活的贝叶斯加性回归树（BART）预后组件与稀疏线性治疗效应组件相结合。这种混合结构将灵活的预后建模与可解释的治疗效应相结合。治疗效应组件包含主效应和成对交互效应，并通过全局-局部收缩先验进行正则化。该模型允许协变量同时对两个组件作出贡献，并提供对治疗效应调节因子的直接推断以及条件均……（摘要在此处截断）

    arXiv:2611.33246v1 Announce Type: cross  Abstract: Estimating covariate-dependent treatment effects is a central challenge in personalized medicine. Existing approaches often emphasize either fully parametric models, risking misspecification, or flexible machine learning methods, which may require large samples to detect the small treatment effect heterogeneity common in medical settings. We propose B-SLIM (BART Sparse Linear Interaction Model), a semi-parametric model for causal inference that combines a flexible Bayesian Additive Regression Trees (BART) prognostic component with a sparse linear treatment effect component. This hybrid structure combines flexible prognosis modelling with interpretable treatment effects. The treatment effect component includes main effects and pairwise interactions, regularized through a global-local shrinkage prior. The model allows covariates to contribute to both components and provides direct inference on treatment effect modifiers, conditional mean
    
[^30]: 一种用于学习有向无环图的因果能量先验

    A Causal Energy Prior for Learning Directed Acyclic Graphs

    [https://arxiv.org/abs/2611.33225](https://arxiv.org/abs/2611.33225)

    提出了局部-全局能量先验（LGE），通过结合局部结构差异（结构汉明距离）与因果差异（结构干预距离）构建DAG上的吉布斯分布，使贝叶斯DAG学习能够编码干预沿路径传播的因果信息，并量化了数据在先验部分错误时对先验的修正能力。

    

    贝叶斯有向无环图（DAG）学习长期以来依赖于无信息或弱信息先验，其假设是有信息先验所需的信息很少可用。在遗传学和系统生物学中，注释数据库如今使得在从数据学习之前构建先验DAG成为可能。现有的有信息先验是局部的：它们逐条弧地对DAG进行评分，无法编码干预如何沿其路径传播。因此，我们提出了局部-全局能量先验（LGE），这是一种DAG上的吉布斯分布，其能量结合了与先验DAG的局部差异（结构汉明距离）和因果差异（结构干预距离）。我们推导了仅从先验DAG出发校准其衰减速度以及两种差异权重的规则。当先验DAG部分不正确时，我们量化了有多少先验概率质量保留在真实DAG上，以及需要多少数据才能推翻该先验。在仿真实验中，LGE……

    arXiv:2611.33225v1 Announce Type: cross  Abstract: Bayesian learning of directed acyclic graphs (DAGs) has long relied on non- or weakly informative priors, on the assumption that the information needed for an informative one is rarely available. In genetics and systems biology, annotation databases now make it possible to assemble a prior DAG before learning from data. Existing informative priors are local: they score a DAG arc by arc and cannot encode how interventions propagate along its paths. Thus, we propose the local-global energy prior (LGE), a Gibbs distribution over DAGs whose energy combines a local divergence from the prior DAG, the Structural Hamming Distance, and a causal one, the Structural Intervention Distance. We derive rules to calibrate its decay and the weight of the two divergences from the prior DAG alone. We quantify how much prior mass stays on the true DAG, and how much data overturn the prior, when the prior DAG is partially incorrect. In simulations, LGE imp
    
[^31]: 对称性预算：网络架构应该假设多少群结构？

    The Symmetry Budget: How Much Group Structure Should a Network Architecture Assume?

    [https://arxiv.org/abs/2611.33169](https://arxiv.org/abs/2611.33169)

    该论文提出“对称性预算”概念，在一个精确可解模型中以闭式形式推导出有限样本超额风险，证明当数据仅具有近似对称性时，网络架构中的中间对称性绑定程度永远不会严格最优。

    

    网络架构在很大程度上是一种对称性假设：卷积将权重在平移之间进行绑定，注意力机制将权重在置换之间进行绑定，而全连接层则不做任何绑定。当数据仅具有近似对称性时，这种绑定会在近似误差与估计误差之间进行权衡，而适当的绑定程度是数据集的属性，而非仅由架构本身决定。我们在一个精确可解的模型中研究这种权衡：循环群 $\mathbb{Z}_d$ 上信号的线性映射，其架构由 $\mathbb{Z}_d$ 的子群格索引，使得单个整数 $k \mid d$ 可以在无约束层（$k=1$，$d^2$ 个参数）和完全等变层（$k=d$，$d$ 个参数）之间进行插值。我们以闭式形式推导了有限样本超额风险，证明对于一般的对称性缺陷，主导阶风险与 $1/k$ 呈仿射关系，并证明此时中间架构永远不会严格最优：

    arXiv:2611.33169v1 Announce Type: cross  Abstract: A network architecture is, in large part, a symmetry assumption: convolution ties weights across translations, attention ties them across permutations, and a fully connected layer ties nothing. When the data are only approximately symmetric, tying trades approximation error against estimation error, and the right amount of tying is a property of the dataset, not of the architecture alone. We study this trade-off in an exactly solvable model: linear maps on signals over the cyclic group $\mathbb{Z}_d$, with architectures indexed by the subgroup lattice of $\mathbb{Z}_d$, so that a single integer $k \mid d$ interpolates between an unconstrained layer ($k=1$, $d^2$ parameters) and a fully equivariant one ($k=d$, $d$ parameters). We derive the finite-sample excess risk in closed form, show that for a generic symmetry defect the leading-order risk is affine in $1/k$, and prove that intermediate architectures are then never strictly optimal:
    
[^32]: 小波行列式采样器：自适应于正则性的最优逼近速率

    Wavelet determinantal samplers: Optimal approximation rates adapting to regularity

    [https://arxiv.org/abs/2611.29992](https://arxiv.org/abs/2611.29992)

    该论文构造了基于小波的行列式点过程采样器，在流形与几何图上对线性统计量实现了 n^{1-2s/m} 阶的方差缩减，且该逼近速率能够自适应于观测量的 Hölder/Sobolev 正则性。

    

    我们在流形与几何图上构造了基于小波的行列式采样器，其逼近速率能够自适应于观测量的正则性。在维度为 $m$ 的黎曼流形上，我们的小波行列式点过程（DPP）对于 $s$-Hölder 观测量和 $H^s$ 观测量的线性统计量实现了 $n^{1-2s/m}$ 阶的方差缩减，其中 $0 < s < m/2$。

    arXiv:2611.29992v1 Announce Type: cross  Abstract: We construct wavelet-based determinantal samplers on manifolds and geometric graphs whose approximation rates adapt to the regularity of the observable. On a Riemannian manifold of dimension $m$, our wavelet DPPs satisfy variance reduction of order $n^{1-2s/m}$ for linear statistics of both $s$-H\"older and $H^s$ observables, where $0
    
[^33]: 材料发现中贝叶斯优化的开箱即用表格基础模型

    Out-of-the-Box Tabular Foundation Models for Bayesian Optimization in Materials Discovery

    [https://arxiv.org/abs/2611.12491](https://arxiv.org/abs/2611.12491)

    该论文证明 TabPFN 和 TabICL 等表格基础模型可以作为开箱即用的代理模型，在材料发现的贝叶斯优化中媲美高斯过程等传统方法，无需核函数选择或任务特定训练。

    

    贝叶斯优化被广泛应用于材料科学的主动学习循环中。高斯过程是事实上的代理模型家族，但需要依赖具体任务的核函数选择，并且可能缺乏对复杂关系的建模能力。表格基础模型（TFMs）最近已成为结构化数据预测任务的最先进方法，无需定制核函数或超参数调优。我们评估了两个表格基础模型 TabPFN 和 TabICL，在两个标准的材料科学基准上作为开箱即用的代理模型，并与涵盖高斯过程、随机森林和 CatBoost 的大量基线方法进行比较。我们的研究结果表明，TFMs 在这两个基准上都是极具竞争力的代理模型，且无需针对主动学习的任务特定训练或 GP 核函数选择。我们的结果为将 TFMs 进一步发展为材料发现中主动学习的代理模型提供了动力。

    arXiv:2611.12491v3 Announce Type: replace  Abstract: Bayesian optimization is widely used in active learning loops for materials science. Gaussian processes serve as the de-facto surrogate model family, but require task-dependent kernel choices and can lack the modeling capacity for complex relationships. Tabular foundation models (TFMs) have recently emerged as the state-of-the-art for predictive tasks on structured data, without requiring bespoke kernels or hyperparameter tuning. We evaluate two TFMs, TabPFN and TabICL, as out-of-the-box surrogate models on two standard materials science benchmarks, comparing against a large field of baselines encompassing Gaussian processes, random forests, and CatBoost. Our findings indicate that TFMs are highly competitive surrogate models across the two benchmarks, without task-specific training for active learning or GP kernel selection. Our results motivate developing TFMs further as surrogates for active learning in materials discovery.
    
[^34]: 流形上流匹配的相容似然

    Compatible Likelihoods for Flow Matching on Manifolds

    [https://arxiv.org/abs/2610.31981](https://arxiv.org/abs/2610.31981)

    该论文针对黎曼流匹配中基于似然的目标可能产生推离数据分布的速度场这一失败模式，提出“相容性”概念刻画似然目标的有效性，并进一步设计了一种基于指数族的新型目标，既满足相容性又仅支持紧流形上可行的速度。

    

    在最简单的非欧几里得流形——一维平坦环面上，我们展示了一个分布，其中最近提出的黎曼流匹配扩展方法被一个将样本推离数据分布的速度场全局优化。我们将这种失败模式追溯到一个我们称之为“相容性”的性质：如果一个基于似然的流匹配目标的总体最小化器能够恢复由数据和耦合所诱导的边际速度场，则称该目标是相容的。我们给出了相容性的一般条件，将其应用于现有方法，并通过在相容性条件下消失的残差分量来界定最坏情况误差。此外，我们注意到现有的相容目标将概率密度置于紧流形上不可行的速度上。为此，我们提出了一个基于端点上指数族构建的目标，以条件速度作为充分统计量，该目标既满足相容性，又仅支持可行的速度。

    arXiv:2610.31981v1 Announce Type: cross  Abstract: On the simplest non-Euclidean manifold, the 1D flat torus, we exhibit a distribution where a recently proposed extension of Riemannian flow matching is globally optimized by a velocity field that pushes samples away from the data distribution. We trace this failure mode to a property we call compatibility; a likelihood-based flow matching objective is compatible if its population minimizer recovers the marginal velocity field induced by the data and coupling. We give a general condition for compatibility, apply it to existing methods, and bound the worst-case error in terms of a residual component vanishing under compatibility. Next, we note that existing compatible objectives put density on velocities that are infeasible on compact manifolds. We propose an objective built from an exponential family over endpoints, with the conditional velocity as sufficient statistic, which is both compatible and supported only on feasible velocities.
    
[^35]: 流形上逻辑斯蒂分类的Besov函数高斯逼近

    Gaussian Approximation of Besov Functions for Logistic Classification on Manifolds

    [https://arxiv.org/abs/2610.27022](https://arxiv.org/abs/2610.27022)

    该论文针对紧致光滑流形上的正则化逻辑斯蒂分类，借助常数线性依赖预测器上确界范数的Bernstein型不等式得到无需有界贝叶斯假设的快速预言机不等式，并通过多带宽高斯核积分算子的带符号组合与实插值方法，构造出同时利用内在维度与目标正则性且无饱和现象的Besov函数高斯逼近。

    

    我们研究了连通、紧致、光滑的嵌入流形 $X\subset\mathbb R^d$（无边界，内在维度为 $\varrho$）上的高斯核逼近与正则化逻辑斯蒂分类问题。两个主要挑战分别是：在不假设贝叶斯函数有界的前提下推导快速的预言机不等式，以及构造既能利用内在维度又能利用目标函数正则性、且不出现饱和现象的高斯逼近器。对于第一个挑战，我们通过一个Bernstein型不等式加以解决，该不等式中的常数与预测器的上确界范数呈线性依赖关系。结合局部化的复杂度估计，由此得到了可分赋范函数空间上的预言机不等式。针对第二个挑战，我们建立了一个一致的Sobolev算子展开，并利用不同带宽下高斯核积分算子的带符号组合来抵消其低阶偏差项。随后通过实插值方法，进而得到Besov函数的 $L_2(X)$ 逼近（摘要原文在此处截断）。

    arXiv:2610.27022v1 Announce Type: cross  Abstract: We study Gaussian kernel approximation and regularized logistic classification on a connected compact smooth embedded manifold $X\subset\mathbb R^d$ without boundary and of intrinsic dimension $\varrho$. Two main challenges are to derive a fast oracle inequality without assuming a bounded Bayes function and to construct Gaussian approximants that exploit both the intrinsic dimension and the target's regularity without saturation. We address the first challenge through a Bernstein-type inequality whose constant depends linearly on the supremum norm of the predictor. Combined with localized complexity estimates, this yields an oracle inequality over separable normed function spaces. To address the second challenge, we establish a uniform Sobolev operator expansion and cancel its lower-order bias terms using a signed combination of Gaussian kernel integral operators at different bandwidths. Real interpolation then gives an $L_2(X)$ approx
    
[^36]: 二部网络中的联邦社区检测

    Federated Community Detection in Bipartite Networks

    [https://arxiv.org/abs/2610.25997](https://arxiv.org/abs/2610.25997)

    提出FLICKER与BiFLICKER方法，在原始数据不可共享的联邦环境下实现了二部网络的谱估计与社区检测，并提供了非渐近收敛与误聚类率的理论保证。

    

    我们研究了当一个大矩阵被水平划分到无法共享原始数据的多台服务器上时的联邦谱估计问题。我们提出了FLICKER，一种针对特定目标的基于梯度的幂迭代方法，它仅通过在本地数据块上执行矩阵-向量乘积，即可恢复与预定谱目标相关联的奇异方向。不同的目标方向可以独立且并行地恢复。我们建立了非渐近收敛保证，并提供了一种用于谱目标选择的递归网格搜索程序。基于FLICKER，我们开发了用于二部网络联邦社区检测的BiFLICKER，并针对相近或重复的奇异方向提出了顺序投影变体。在稀疏二部随机块模型下，我们证明了来自参考服务器的经过适当缩放的主奇异值可以提供有效的全局目标，并且顺序BiFLICKER实现了趋近于零的误聚类率。

    arXiv:2610.25997v2 Announce Type: replace-cross  Abstract: We study federated spectral estimation when a large matrix is horizontally partitioned across servers that cannot share raw data. We propose FLICKER, a target-specific gradient-based power iteration that recovers a singular direction associated with a prescribed spectral target using only matrix-vector products on local blocks. Different target directions can be recovered independently and in parallel. We establish non-asymptotic convergence guarantees and provide a recursive grid-search procedure for spectral target selection. Building on FLICKER, we develop BiFLICKER for federated community detection in bipartite networks, together with a sequential projected variant for nearby or repeated singular directions. Under a sparse bipartite stochastic block model, we show that properly rescaled leading singular values from a reference server provide valid global targets, and that Sequential BiFLICKER achieves a vanishing miscluster
    
[^37]: 连续马尔可夫链伪谱隙的极小极大估计及其在经验集中界中的应用

    Minimax estimation of the pseudo-spectral gap of continuous Markov chains, with applications to empirical concentration bounds

    [https://arxiv.org/abs/2610.20007](https://arxiv.org/abs/2610.20007)

    本文提出了紧致连续状态空间马尔可夫链伪谱隙的首个极小极大最优估计器，结合 Bernstein 不等式推导出经验集中不等式，并据此在无需伪谱隙先验知识的情况下，为马尔可夫奖励老虎机问题给出了具有接近独立同分布情形遗憾界的 UCB 算法。

    

    我们研究了基于单条轨迹观测来对马尔可夫链伪谱隙 $\gamma_{ps}$ 进行极小极大估计的问题。通过扩展 Wolfer 和 Kontorovich (2024) 关于有限状态空间链的早期工作，我们针对紧致连续状态空间的情形提出了一个估计器。在转移算子具有 Lipschitz 密度的链类上，我们的估计器在相差对数因子的意义下似乎是极小极大最优的。将该估计器与 Paulin 版本的 Bernstein 不等式相结合，可以为马尔可夫链推导出经验集中不等式。这开启了许多应用，包括置信区间、主动学习和泛化界。特别地，对于具有马尔可夫奖励的静止老虎机问题，我们提出了一个 UCB 算法版本，其遗憾界与独立同分布情形类似，且无需对链的伪谱隙有任何先验知识。

    arXiv:2610.20007v1 Announce Type: cross  Abstract: We study the minimax estimation of the pseudo-spectral gap $\gamma_{ps}$ of a Markov chain from the observation of a single trajectory. Extending earlier works on finite-state space chains by Wolfer and Kontorovich (2024), we provide an estimator in the case of a compact, continuous state-space. Our estimator appears to be minimax-optimal on the class of chains with a transition operator with Lipschitz density, up to logarithmic terms. Our estimator, combined with Paulin's version of Bernstein inequality, allows to derive empirical concentration inequalities for Markov chains. This unlocks many applications, from confidence intervals, to active learning and generalization bounds. In particular, for rested bandits with Markov rewards, we provide a version of UCB with a regret bound similar to the one in the iid case, without assuming any knowledge on the pseudo-spectral gap of the chains.
    
[^38]: 极端二分类：利用极值理论实现对假负类的极端约束

    Extreme Binary Classification: Extreme Value Theory for Extreme Constraint on False Negative

    [https://arxiv.org/abs/2610.09984](https://arxiv.org/abs/2610.09984)

    本文提出“极端二分类”新问题，并基于极值理论设计了阈值自适应方法与基于置换检验的特征选择程序，使分类器的假负类率以快于 $1/N_1$ 的速率趋近于零，实验表现优于最先进方法。

    

    尽管二分类是机器学习中研究最为广泛的问题之一，但以学习一个假负类率几乎为零的分类器为目标的这一情形在很大程度上仍未被探索。在本文中，我们提出了“极端二分类”问题，其目标是学习一个假负类率 $\alpha$ 受 $\epsilon_{N_1}=o_{N_1\to\infty}(1/N_1)$ 约束的分类器，其中 $N_1$ 表示训练集中正例样本的数量。为了解决这一问题，我们提出了一种基于极值理论推导的理论保证的阈值自适应方法，并结合一种基于对样本最大值进行置换检验的特征选择程序。在四个不同规模的真实数据集上的实验结果表明，我们的方法优于当前最先进的方法。此外，我们还通过……展示了该方法的可解释性。

    arXiv:2610.09984v1 Announce Type: cross  Abstract: While binary classification is one of the most extensively studied problems in machine learning,   the regime in which the goal is to learn a classifier with an almost zero false negative rate remains largely unexplored.   In this paper, we introduce the Extreme Binary Classification problem, where the objective is to learn a classifier whose false negative rate $\alpha$ is constrained by $\epsilon_{N_1}=o_{N_1\to\infty}(1/N_1)$, with $N_1$ denoting the number of positive examples in the training set.   To address this problem, we propose a threshold adaptation method theoretically grounded in guarantees derived from Extreme Value Theory, together with a feature selection procedure based on a permutation test applied to sample maxima.   Experimental results on four real-world datasets of varying sizes demonstrate that our approach compares favorably with state-of-the-art methods.   In addition, we illustrate its interpretability throug
    
[^39]: 面向表格基础模型的共形预测与条件覆盖

    Conformal Prediction and Conditional Coverage for Tabular Foundation Models

    [https://arxiv.org/abs/2609.34887](https://arxiv.org/abs/2609.34887)

    该论文提出C-USIM方法，一种无需额外训练或推理、可处理多模态预测的轻量级最高预测密度分割共形预测技术，为表格基础模型提供有限样本边际覆盖有效性保证，并显著改善条件覆盖和分组覆盖精度。

    

    表格基础模型（TFMs）为回归任务提供预测分布，但即使点预测准确，其预测区间仍可能出现覆盖不足或覆盖过度的问题。我们提出了C-USIM（条件均匀化得分积分方法），这是最高预测密度分割共形预测的一种轻量级应用，能够适应多模态预测。在给定校准集和测试集输出的情况下，该方法无需额外的训练或模型推理。在我们的假设下，它提供了有限样本的边际有效性保证。我们利用分布估计误差和得分离散性对条件-边际覆盖差距进行了界定，并通过百分位秩-得分图考察了覆盖的异质性。在TabPFN和TabICL上的实验表明，该方法提高了边际覆盖的准确性，并降低了平均条件覆盖误差和分组覆盖误差。在固定数据预算下，将更多的观测数据分配给校准阶段可以减少（条件覆盖误差）……

    arXiv:2609.34887v2 Announce Type: replace-cross  Abstract: Tabular foundation models (TFMs) provide predictive distributions for regression, but their prediction regions can exhibit undercoverage or overcoverage even when point predictions are accurate. We introduce C-USIM (Conditionally-Uniformized Score Integration Method), a lightweight application of highest predictive density split conformal prediction that accommodates multimodal predictions. Given calibration and test outputs, it requires no additional training or model inference. It provides finite-sample marginal validity under our assumptions. We bound conditional-marginal coverage gaps using distribution-estimation error and score discreteness, and examine coverage heterogeneity through percentile rank-score plots. Experiments with TabPFN and TabICL show improved marginal coverage accuracy and lower average conditional and group coverage errors. Under a fixed data budget, allocating more observations to calibration can reduc
    
[^40]: 在线非单调DR-次模最大化：匹配离线0.401近似比

    Online Non-Monotone DR-Submodular Maximization Matching the Offline $0.401$ Factor

    [https://arxiv.org/abs/2609.02145](https://arxiv.org/abs/2609.02145)

    该论文首次在对抗性在线设置下实现了非单调DR-次模最大化与非离线算法相同的0.401最优近似比，通过用加权在线学习器替代离线箱约束步骤并结合精确非对称平衡定理，在决策后全信息值预言机模型下达到次线性近似遗憾。

    

    我们研究在 $d$ 维单位立方体的紧凸下闭子集上，非负、非单调DR-次模函数的在线最大化问题。在相应的元可解性假设下，目前已知最好的构造性离线近似比为 $0.401$，而可比的对抗性在线保证一直停留在 $1/e$。我们证明该 $0.401$ 近似比同样可以在线实现。在决策后全信息值预言机模型中，当预言机反馈条件无偏且有界时，我们的算法以次线性近似遗憾达到了 $0.401$ 的近似比。该在线算法并不在变化的目标函数上运行离线构造，而是用加权在线学习器替代离线算法中依赖于目标的箱约束步骤，以累积方式控制所需的残差项。一个精确的非对称平衡定理使得离线系数在对抗性变化下仍得以保持。直接实现（原文摘要在此处截断）……

    arXiv:2609.02145v1 Announce Type: cross  Abstract: We study online maximization of nonnegative, non-monotone DR-submodular functions over compact convex down-closed subsets of the $d$-dimensional unit cube. The best known constructive offline approximation factor is $0.401$ under the corresponding meta-solvability assumptions, whereas comparable adversarial online guarantees had remained at $1/e$. We show that this factor is also achievable online. In the post-decision full-information value-oracle model, our algorithm attains factor $0.401$ with sublinear approximate regret when oracle feedback is conditionally unbiased and bounded.   The online algorithm does not run the offline construction on a changing objective. Instead, it replaces the offline objective-dependent box step by a weighted online learner that controls the required residual terms cumulatively. An exact asymmetric balance theorem preserves the offline coefficients despite adversarial variation. The direct implementati
    
[^41]: 生成模型下强化学习的改进量子算法

    Improved Quantum Algorithms for Reinforcement Learning Under a Generative Model

    [https://arxiv.org/abs/2608.02826](https://arxiv.org/abs/2608.02826)

    本文将标准值迭代与量子均值估计、量子最大值查找等量子子程序相结合，提出了求解有限时域和无限时域折扣MDP近似最优策略的新量子算法，其查询复杂度优于以往工作并接近已有的量子下界。

    

    强化学习是机器学习的一个子领域，研究智能体如何与环境进行交互，从而获取尽可能大的奖励。研究这种交互的标准方法是通过马尔可夫决策过程（MDP）以及选择最优策略的任务——策略是一个告诉智能体应采取何种动作的函数。在这项工作中，我们研究了两种时间相关的MDP——有限时域MDP和无限时域折扣MDP，并提出了用于计算近似最优策略的新量子算法。我们的量子算法基于标准值迭代与量子子程序（如量子均值估计和量子最大值查找）的一种新组合，并整体上融合了样本最优经典算法的技术加以增强。我们得到的查询复杂度优于以往工作，从而接近已经建立的量子下界。

    arXiv:2608.02826v2 Announce Type: replace-cross  Abstract: Reinforcement learning is a subfield of machine learning that studies how an agent interacts with an environment in order to extract as large a reward as possible. A standard approach to study such interaction is through Markov Decision Processes (MDPs) and the task of choosing an optimal policy---a function that tells the agent which action to take. In this work, we study two types of time-dependent MDP---finite-horizon and infinite-horizon discounted---and propose new quantum algorithms for computing approximate optimal policies. Our quantum algorithms are based on a new combination of standard value iteration and quantum subroutines like quantum mean estimation and quantum maximum finding, overall enhanced with techniques from sample-optimal classical algorithms. Our resulting query complexities improve upon previous works, thus approaching already established quantum lower bounds.
    
[^42]: 表格上下文学习者能否泛化到生物分子性质预测？

    Can Tabular In-Context Learners Generalize to Biomolecular Property Prediction?

    [https://arxiv.org/abs/2606.31126](https://arxiv.org/abs/2606.31126)

    本文评估了TabPFN和TabICL等表格基础模型能否作为小样本场景下的数据高效预测器泛化到生物分子性质预测任务，发现在ProteinGym上TabPFN3-ESMC600M与领先的监督方法具有竞争力并取得最低MSE，但Kermut在Spearman相关性上仍更强，且在更难的划分设置下性能明显下降。

    

    从有限的标注数据预测生物分子性质是蛋白质工程和小分子设计中的一个核心瓶颈。随着强大的预训练编码器如今能够提供丰富的固定长度表示，难点已从表示学习转向为小样本场景构建数据高效的预测器。TabPFN和TabICL等表格基础模型在合成表格上完成预训练后，可通过上下文推理来执行此类预测。我们测试了它们能否作为在信息丰富的固定生物分子表示之上运作的灵活插值器。将每种方法视为预测器-表示对，我们在两个领域进行了评估。在ProteinGym上，TabPFN3-ESMC600M与领先的监督替换模型条目具有竞争力，并在所考虑的排行榜上下文行中取得了最低的MSE，而Kermut在Spearman相关性上仍更胜一筹。在模运算和连续留出设置下，性能显著降低……（原文摘要在此处截断）

    arXiv:2606.31126v3 Announce Type: replace  Abstract: Predicting biomolecular properties from limited labeled data is a central bottleneck in protein engineering and small-molecule design. As strong pretrained encoders now supply rich fixed-length representations, the difficulty has shifted from representation learning to building a data-efficient predictor for the few-shot regime. Tabular foundation models such as TabPFN and TabICL perform this prediction through in-context inference after pretraining on synthetic tables. We test whether they can act as flexible interpolators over informative fixed biomolecular representations. Treating each method as a predictor-representation pair, we evaluate across two domains. On ProteinGym, TabPFN3-ESMC600M is competitive with leading supervised-substitution entries and obtains the lowest MSE among the leaderboard-context rows considered, while Kermut remains stronger by Spearman. Performance is substantially lower under modulo and contiguous hol
    
[^43]: 以人为中心的学习力学：受保护风险约束下的可观测熵控制

    Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints

    [https://arxiv.org/abs/2605.22940](https://arxiv.org/abs/2605.22940)

    本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。

    

    熵降低并非一个安全准则：它可能使预测保持不变，或伴随着受保护群体风险的上升。我们研究在分别指定的任务、风险与干预约束下的熵调控学习。一个经噪声校准的对数行列式泛函给出了精确的谱力公式，以及将速度度量分解为可观测分量与预测零空间分量的方法。对于标量反馈，我们刻画了在保持任务下降比例并满足熵速率要求时系数的完整区间；而将干预限制在可观测方向上可能使该区间为空。对于多个受保护风险，我们在指定子空间与范数预算内推导出精确的极小极大可行性证书、针对不确定约束行的精确鲁棒扩展，以及保护后剩余熵削减的锥投影公式。这些公式采用标准的凸几何方法，但……

    arXiv:2605.22940v2 Announce Type: replace-cross  Abstract: Entropy reduction is not a safety criterion: it can leave predictions unchanged or accompany increasing risk for a protected group. We study entropy-regulated learning under separately specified task, risk, and intervention constraints. A noise-calibrated log-determinant functional yields an exact spectral force formula and a metric decomposition into observable and prediction-null velocities. For scalar feedback, we characterize the complete interval of coefficients preserving a task-descent fraction and meeting an entropy-rate request; restricting intervention to observable directions can make that interval empty. For several protected risks, we derive an exact minimax certificate of feasibility within a specified subspace and norm budget, an exact robust extension for uncertain constraint rows, and a cone-projection formula for the entropy reduction remaining after protection. These formulas use standard convex geometry but 
    
[^44]: 使用描述长度引导多目标遗传编程以改进符号回归解决方案

    Guiding Multi-Objective Genetic Programming with Description Length Improves Symbolic Regression Solutions

    [https://arxiv.org/abs/2605.22374](https://arxiv.org/abs/2605.22374)

    该论文提出将描述长度（DL）和分数贝叶斯因子（FBF）作为遗传编程符号回归中原则性的选择与搜索准则，以缓解过拟合和结构膨胀，从而提升解的泛化性能。

    

    基于遗传编程的符号回归（GPSR）可能遭受过拟合和结构膨胀问题，尤其是在存在噪声的情况下。本文评估了描述长度（DL）和分数贝叶斯因子（FBF）准则，作为启发式方法的原则性、数据高效替代方案，用于选择泛化能力良好的紧凑表达式。我们使用基于Fisher信息的参数编码实现了DL，并将其与赤池信息准则（AIC）和贝叶斯信息准则（BIC）在多个数据集上进行比较，包括含噪声的合成基准和真实世界的回归问题。我们研究了三种搜索/选择策略：(i) 先针对精度和程序长度进行多目标搜索，随后用DL/FBF进行选择；(ii) 直接将DL作为目标的多目标搜索；(iii) 以DL/FBF作为适应度的单目标优化。在各个数据集中，我们发现DL/FBF后选择改进了测试

    arXiv:2605.22374v2 Announce Type: replace-cross  Abstract: Symbolic regression with genetic programming (GPSR) may suffer from overfitting and structural bloat, especially when noise is present. In this paper we evaluate description length (DL) and fractional Bayes factor (FBF) criteria as principled, data-efficient alternatives to heuristics for selecting compact expressions that generalise well. We implement DL using a Fisher-information-based parameter encoding and compare it to the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC) across multiple datasets, including noisy synthetic benchmarks and real-world regression problems. We study three search/selection strategies: (i) multi-objective search for accuracy and program length followed by DL/FBF selection; (ii) multi-objective search using DL directly as an objective; and (iii) single-objective optimisation with DL/FBF as the fitness. Across datasets, we find that DL/FBF post-selection improves test 
    
[^45]: 核心集诱导的条件速度流匹配

    Coreset-Induced Conditional Velocity Flow Matching

    [https://arxiv.org/abs/2605.12951](https://arxiv.org/abs/2605.12951)

    该论文提出CCVFM生成模型，通过熵Sinkhorn核心集构建闭式代理源替代分层整流流中内部流的高斯噪声源，使条件速度规律成为无需神经采样器即可采样的闭式高斯混合，并只需轻量级校正流即可完成训练。

    

    我们提出了核心集诱导的条件速度流匹配（CCVFM），这是一种生成模型，它通过由数据信息构建的源分布来增强分层整流流。标准流匹配仅回归速度的条件均值；而分层流匹配则通过速度空间中的第二个内部流来建模完整的条件速度规律，但这个内部流需要从头开始将各向同性高斯噪声传输到多峰的目标速度分布。我们的关键观察是：这个内部源可以被一个由目标核心集构建的闭式代理分布所替代。CCVFM首先使用熵正则化的Sinkhorn核心集将目标压缩为加权原子，并将其提升为高斯混合。由此诱导的条件速度规律是一个闭式高斯混合，无需学习神经采样器即可进行采样。基于该代理源训练的轻量级校正流，t...

    arXiv:2605.12951v3 Announce Type: replace  Abstract: We propose Coreset-Induced Conditional Velocity Flow Matching (CCVFM), a generative model that augments hierarchical rectified flow with a data-informed source distribution. Standard flow matching regresses only the conditional mean of the velocity; hierarchical flow matching instead models the full conditional velocity law with a second, inner flow in velocity space, but this inner flow is asked to transport isotropic Gaussian noise to a multimodal target velocity distribution from scratch. Our key observation is that this inner source can be replaced by a closed-form surrogate built from a coreset of the target. CCVFM first compresses the target into weighted atoms using an entropic Sinkhorn coreset and lifts them to a Gaussian mixture. The induced conditional velocity law is then a closed-form Gaussian mixture that can be sampled without a learned neural sampler. A lightweight correction flow, trained from this surrogate source, t
    
[^46]: 基于人类反馈的强化学习：统计视角

    Reinforcement Learning from Human Feedback: A Statistical Perspective

    [https://arxiv.org/abs/2604.02507](https://arxiv.org/abs/2604.02507)

    本综述首次系统地从统计学视角梳理了RLHF框架，将其核心组成部分（监督微调、奖励建模与策略优化）与BTL模型、潜效用估计、主动学习、实验设计和不确定性量化等经典统计概念联系起来。

    

    基于人类反馈的强化学习（RLHF）已成为将大语言模型（LLM）与人类偏好对齐的核心框架。尽管在实践中取得了成功，RLHF由于依赖嘈杂、主观且异构的反馈来学习奖励模型并优化策略，因而引发了一些根本性的统计学问题。本综述从统计学的视角审视RLHF，主要聚焦于大语言模型的对齐。我们介绍了RLHF的主要组成部分，包括监督微调、奖励建模和策略优化，并将它们与人们熟知的统计概念联系起来，如Bradley-Terry-Luce（BTL）模型、潜效用估计、主动学习、实验设计和不确定性量化。我们回顾了从成对偏好数据中学习奖励函数的方法，以及通过两阶段RLHF流程和新兴的一阶段方法（如直接偏好优化）来优化策略的相关方法。

    arXiv:2604.02507v2 Announce Type: replace-cross  Abstract: Reinforcement learning from human feedback (RLHF) has emerged as a central framework for aligning large language models (LLMs) with human preferences. Despite its practical success, RLHF raises fundamental statistical questions because it relies on noisy, subjective, and heterogeneous feedback to learn reward models and optimize policies. This survey provides a statistical perspective on RLHF, focusing primarily on LLM alignment. We introduce the main components of RLHF, including supervised fine-tuning, reward modeling, and policy optimization, and relate them to familiar statistical concepts such as the Bradley-Terry-Luce (BTL) model, latent utility estimation, active learning, experimental design, and uncertainty quantification. We review methods for learning reward functions from pairwise preference data and for optimizing policies through both two-stage RLHF pipelines and emerging one-stage approaches such as direct prefer
    
[^47]: BayesFlow 2：Python中的多后端摊销贝叶斯推断

    BayesFlow 2: Multi-Backend Amortized Bayesian Inference in Python

    [https://arxiv.org/abs/2602.07098](https://arxiv.org/abs/2602.07098)

    BayesFlow 2是一个支持多种深度学习后端的通用Python库，通过摊销贝叶斯推断实现对后验分布、似然和比率的快速估计，解决了传统贝叶斯方法在复杂模型和大数据集上的计算瓶颈。

    

    现代贝叶斯推断涉及多种计算方法的组合，用于在系统性工作流程中对概率模型进行估计、验证并从中得出结论。许多贝叶斯方法的一个共同特征是速度相对较慢，在将复杂模型拟合到大型数据集时，这往往成为难以逾越的障碍。摊销贝叶斯推断为解决贝叶斯方法的计算挑战提供了一条途径。ABI通过在模型模拟数据上训练神经网络，使用户能够快速推断任何模型隐含的量，例如点估计、似然或完整的后验分布。在这项工作中，我们介绍了用于通用ABI的Python库BayesFlow第2版。除了直接的后验、似然和比率估计外，该软件还支持多种流行的深度学习后端，并提供丰富的用于采样和密度估计的生成网络集合。

    arXiv:2602.07098v3 Announce Type: replace-cross  Abstract: Modern Bayesian inference involves a mixture of computational methods for estimating, validating, and drawing conclusions from probabilistic models as part of principled workflows. An overarching motif of many Bayesian methods is that they are relatively slow, which often becomes prohibitive when fitting complex models to large data sets. Amortized Bayesian inference (ABI) offers a path to solving the computational challenges of Bayes. ABI trains neural networks on model simulations, rewarding users with rapid inference of any model-implied quantity, such as point estimates, likelihoods, or full posterior distributions. In this work, we present the Python library BayesFlow, Version 2, for general-purpose ABI. Along with direct posterior, likelihood, and ratio estimation, the software includes support for multiple popular deep learning backends, a rich collection of generative networks for sampling and density estimation, comple
    
[^48]: 可交换张量与矩阵值数据的集中不等式

    Concentration Inequalities for Exchangeable Tensors and Matrix-valued Data

    [https://arxiv.org/abs/2601.20152](https://arxiv.org/abs/2601.20152)

    该论文在可交换性条件下为张量内积与序贯矩阵和等结构化加权和建立了 Hoeffding 与 Bernstein 型集中不等式，以最优常数恢复了已知结果并改进了矩阵组合和的偏差项常数，为多因子响应模型估计与固定设计草图方法提供了新的分析工具。

    

    我们研究随机数据的结构化加权和的集中不等式，包括 张量内积以及 序贯矩阵和。我们关注在可交换性条件下这些结构化加权和的尾界与集中不等式，从而将研究范围拓展至超越经典独立项框架的情形。我们在结构相关的可交换性条件下建立了 Hoeffding 界和 Bernstein 界。在此过程中，我们以最优常数重新得到了关于可交换随机变量加权和以及独立同分布随机矩阵和的已知结果。我们还推导了矩阵阵列组合和的集中界，其偏差项中的常数相比可交换对方法有所改进。在应用方面，更丰富的结构为我们提供了新颖的分析工具，可用于估计多因子响应模型的平均效应以及研究固定设计草图方法。

    arXiv:2601.20152v4 Announce Type: replace-cross  Abstract: We study concentration inequalities for structured weighted sums of random data, including (i) tensor inner products and (ii) sequential matrix sums. We are interested in tail bounds and concentration inequalities for those structured weighted sums under exchangeability, extending beyond the classical framework of independent terms.   We develop Hoeffding and Bernstein bounds provided with structure-dependent exchangeability. Along the way, we recover known results in weighted sum of exchangeable random variables and i.i.d. sums of random matrices to the optimal constants. We also derive a concentration bound for combinatorial sums of matrix arrays with an improved constant in the bias term compared with that by the method of exchangeable pairs.   For applications, the richer structures provide us with novel analytical tools for estimating the average effect of multi-factor response models and studying fixed-design sketching me
    
[^49]: CoDeQ：使用死区量化器实现可微联合剪枝与量化的模型压缩

    CoDeQ: Compression with a Dead-Zone Quantizer for Differentiable Joint Pruning--Quantization

    [https://arxiv.org/abs/2512.12981](https://arxiv.org/abs/2512.12981)

    该论文提出CoDeQ，通过可学习的死区量化器将剪枝与量化统一在同一训练框架中，以死区宽度控制逐层稀疏度并支持可学习的混合精度位宽，在单一全局压缩系数下联合优化权重与压缩参数，从而省去辅助分配流程和大量超参数调优。

    

    联合剪枝与量化可以降低神经网络的推理成本，但确定各层的稀疏度与精度通常需要辅助的分配流程以及大量的超参数调优。我们提出CoDeQ，这是一种量化感知训练方法，通过一个可学习的死区量化器将剪枝与量化统一起来。其学习到的死区宽度可以控制各层的稀疏度，而位宽既可以保持固定，也可以在混合精度模式下进行学习。一个剪枝感知的缩放因子会在保留的范围内分配所有可用的非零重建电平。我们推导了在固定位宽和范围端点条件下，能够最小化该范围内最坏情况重建误差的间距。模型权重与压缩参数在一个全局压缩系数下联合优化，压缩参数共享同一学习率。在VGG7/CIFAR-10上，使用2位激活的CoDeQ实现了0.05%的相对……

    arXiv:2512.12981v2 Announce Type: replace  Abstract: Joint pruning and quantization can reduce the inference cost of neural-networks, but determining layerwise sparsity and precision often requires auxiliary allocation procedures and considerable hyperparameter tuning. We introduce CoDeQ, a quantization-aware training method that unifies pruning and quantization through a learnable dead-zone quantizer. Its learned dead-zone width controls layerwise sparsity, while bit widths can be held fixed or learned in mixed-precision mode. A pruning-aware scale allocates all available nonzero reconstruction levels over the retained range. We derive the spacing that minimizes worst-case reconstruction error over this range for a fixed bit width and range endpoint. Model weights and compression parameters are optimized jointly under one global compression coefficient, with a shared learning rate for the compression parameters. On VGG7/CIFAR-10, CoDeQ with 2-bit activations achieves 0.05% relative BO
    
[^50]: 通过后验校正实现高维回归的去偏推断

    Debiased Inference via Posterior Corrections for High-Dimensional Regression

    [https://arxiv.org/abs/2512.09257](https://arxiv.org/abs/2512.09257)

    该论文提出一种新颖的去偏方法，通过校正整个贝叶斯后验分布的偏差并建立新的Bernstein-von Mises定理，使高维回归中基于稀疏先验的后验分布具备频率学派有效性，其可信集可作为有效置信集。

    

    基于稀疏性诱导先验（如spike-and-slab先验和马蹄先验）的高维回归贝叶斯推断已取得显著进展。然而，由此产生的后验分布通常不具备理想的频率学派性质，因此其可信集即使在渐近意义下也无法作为有效的置信集。我们提出了一种新颖的去偏方法，对整个（可能是近似的）贝叶斯后验分布进行偏差校正。我们建立了新的Bernstein-von Mises定理，以保证去偏后分布的频率学派有效性。我们通过蒙特卡洛模拟和经济学中的两个实证应用展示了该方法在实际中的表现。

    arXiv:2512.09257v2 Announce Type: replace  Abstract: There has been significant progress in Bayesian inference based on sparsity-inducing priors, such as spike-and-slab and horseshoe priors, for high-dimensional regression. The resulting posteriors, however, in general do not possess desirable frequentist properties, and the credible sets thus cannot serve as valid confidence sets even asymptotically. We introduce a novel debiasing approach that corrects the bias for the entire (possibly approximate) Bayesian posterior distribution. We establish new Bernstein-von Mises theorems that guarantee the frequentist validity of the debiased law. We demonstrate the practical performance of our proposal through Monte Carlo simulations and two empirical applications in economics.
    
[^51]: 信度超空间上的不动点与包络动力学

    Fixed Points and Envelope Dynamics on Credal Hyperspaces

    [https://arxiv.org/abs/2510.04769](https://arxiv.org/abs/2510.04769)

    本文证明了信度集超空间中包含单调且豪斯多夫连续的自映射在完备格的信度序区间上存在可由迭代求得的最小与最大不动点，并为随时间变化的更新族构造了界定所有容许轨迹的上下包络动力学。

    

    设 $X$ 为紧度量空间，考虑由 $X$ 上博雷尔概率测度的非空弱闭凸集构成的超空间 $\mathscr C_X$，称为信度集。这些集合表示关于哪种概率规律支配所研究现象的不确定性。我们用由弱收敛度量诱导的豪斯多夫度量来装备 $\mathscr C_X$。信度序区间 $[\mathcal{A},\mathcal{B}] = \{ \mathcal{P}\in\mathscr C_X: \mathcal{A}\subseteq\mathcal{P}\subseteq\mathcal{B} \}$ 是完备格，且此类区间上的每个包含单调、豪斯多夫连续的自映射都具有最小和最大不动点，它们可分别作为从 $\mathcal{A}$ 和 $\mathcal{B}$ 出发迭代所得的极限而得到。对于可能随时间变化的一族包含单调更新，我们构造了下包络映射和上包络映射，用以界定每条容许轨迹。当这些包络映射为豪斯多夫连续时，每条轨迹与……的距离（摘要在此处被截断）

    arXiv:2510.04769v3 Announce Type: replace-cross  Abstract: Let $X$ be a compact metric space and consider the hyperspace $\mathscr C_X$ of nonempty weakly closed convex sets of Borel probability measures on $X$, called credal sets. These sets represent uncertainty about which probability law governs the phenomenon of interest. We equip $\mathscr C_X$ with the Hausdorff metric induced by a metric for weak convergence. Credal order intervals $[\mathcal{A},\mathcal{B}] = \{ \mathcal{P}\in\mathscr C_X: \mathcal{A}\subseteq\mathcal{P}\subseteq\mathcal{B} \}$ are complete lattices, and every inclusion-monotone, Hausdorff-continuous self map of such an interval has least and greatest fixed points obtained as limits of iteration from $\mathcal{A}$ and $\mathcal{B}$. For families of inclusion-monotone updates that may vary with time, we construct lower and upper envelope maps that bound every admissible trajectory. When these envelope maps are Hausdorff-continuous, the distance of every traject
    
[^52]: 差分隐私随机梯度下降的统计推断

    Statistical Inference for Differentially Private Stochastic Gradient Descent

    [https://arxiv.org/abs/2507.20560](https://arxiv.org/abs/2507.20560)

    本文首次建立了随机化子采样规则下SGD的渐近性质并将其扩展到DP-SGD，证明了其渐近方差可分解为统计、采样和隐私诱导三个分量，并提出了代入法和随机缩放法两种构建有效置信区间的方法。

    

    机器学习中的隐私保护，特别是通过差分隐私随机梯度下降（DP-SGD）实现的隐私保护，对于分析敏感数据至关重要。然而，现有的SGD统计推断方法主要集中于循环子采样，而DP-SGD需要随机化子采样。本文首次通过建立随机化规则下SGD的渐近性质并将这些结果扩展到DP-SGD来弥合这一差距。对于DP-SGD，我们证明了渐近方差可分解为统计分量、采样分量和隐私诱导分量三个部分。我们提出了两种构建有效置信区间的方法：代入法和随机缩放法。我们还进行了广泛的数值分析，结果表明所提出的置信区间在保持隐私的同时达到了名义覆盖率。

    arXiv:2507.20560v3 Announce Type: replace-cross  Abstract: Privacy preservation in machine learning, particularly through Differentially Private Stochastic Gradient Descent (DP-SGD), is crucial for analyzing sensitive data. However, existing statistical inference methods for SGD predominantly focus on cyclic subsampling, while DP-SGD requires randomized subsampling. This paper first bridges this gap by establishing the asymptotic properties of SGD under the randomized rule and extending these results to DP-SGD. For DP-SGD, we show that the asymptotic variance decomposes into statistical, sampling, and privacy-induced components. We propose two methods for constructing valid confidence intervals: the plug-in method and the random scaling method. We also perform extensive numerical analysis showing that the proposed confidence intervals achieve nominal coverage rates while maintaining privacy.
    
[^53]: 稳定性正则化交叉验证

    Stability Regularized Cross-Validation

    [https://arxiv.org/abs/2505.06927](https://arxiv.org/abs/2505.06927)

    该论文提出了一种嵌套 k 折交叉验证方案，通过最小化标准交叉验证误差与经验假设稳定性的加权和来选择超参数，并证明了由此得到的泛化界，显示在信噪比较低的场景下该方法对测试误差的改善最为显著。

    

    我们研究通过交叉验证进行超参数选择的问题，其动机源于这样的场景：所选模型的 k 折交叉验证误差会低估该模型的测试误差。我们提出了一种嵌套的 k 折交叉验证方案，该方案最小化标准交叉验证误差与经验假设稳定性的加权和，其中稳定性权重由外层交叉验证循环来选择。我们推导出了一个泛化界，利用交叉验证误差和经验假设稳定性（以及每个候选模型共有的修正项）来控制测试误差。在一个简化的岭回归设定中，我们进一步表明，稳定性权重能否提升期望测试集性能取决于信噪比，当信噪比较小时提升效果最为显著。在 13 个真实世界的回归数据集上，该程序相比 k 折交叉验证降低了平均样本外 MSE。

    arXiv:2505.06927v3 Announce Type: replace-cross  Abstract: We study hyperparameter selection via cross-validation, motivated by settings in which the selected $k$-fold cross-validation error underestimates the selected model's test error. We propose a nested $k$-fold cross-validation scheme that minimizes a weighted sum of the standard cross-validation error and empirical hypothesis stability, with the stability weight selected by an outer cross-validation loop. We derive a generalization bound that controls test error using the cross-validation error and empirical hypothesis stability, together with correction terms common to every candidate. In a stylized ridge regression setting, we then show that whether the stability weight improves expected test-set performance depends on the signal-to-noise ratio, with the greatest improvement when the signal-to-noise ratio is small. On $13$ real-world regression datasets, the procedure reduces the average out-of-sample MSE relative to $k$-fold 
    
[^54]: 非高斯有向无环图的结构迁移学习

    Structural transfer learning of non-Gaussian DAG. (arXiv:2310.10239v1 [stat.ML])

    [http://arxiv.org/abs/2310.10239](http://arxiv.org/abs/2310.10239)

    本文提出了一种用于非高斯有向无环图的结构迁移学习方法，通过引入新颖的DAG结构相似度度量并利用不同相似性水平的辅助DAG的信息，能够显著提高目标研究中的DAG重构效果。

    

    有向无环图(DAG)被广泛用于表示一组收集到的节点之间的方向关系。然而，一次单独研究中可用的数据往往有限，不足以准确地重构DAG，而异构数据可能会从多个相关研究中收集到。如何将异构数据汇集到目标研究中以更好地重构DAG结构仍然是一个未解决的问题。在本文中，我们首先引入了一套新颖的DAG结构相似度度量，然后提出了一个传输DAG学习框架，通过有效地利用不同相似性水平的辅助DAG的信息来提高目标研究中的DAG重构效果。我们的理论分析表明，在目标DAG没有整体与辅助DAG相似的情况下，我们的方法也能在DAG重构方面实现显著的改进，这与大多数现有的传输学习方法形成鲜明对比。我们的实验结果也支持了所提出的传输DAG学习的优势。

    Directed acyclic graph (DAG) has been widely employed to represent directional relationships among a set of collected nodes. Yet, the available data in one single study is often limited for accurate DAG reconstruction, whereas heterogeneous data may be collected from multiple relevant studies. It remains an open question how to pool the heterogeneous data together for better DAG structure reconstruction in the target study. In this paper, we first introduce a novel set of structural similarity measures for DAG and then present a transfer DAG learning framework by effectively leveraging information from auxiliary DAGs of different levels of similarities. Our theoretical analysis shows substantial improvement in terms of DAG reconstruction in the target study, even when no auxiliary DAG is overall similar to the target DAG, which is in sharp contrast to most existing transfer learning methods. The advantage of the proposed transfer DAG learning is also supported by extensive numerical ex
    
[^55]: 外部有效的策略选择

    Externally Valid Policy Choice. (arXiv:2205.05561v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.05561](http://arxiv.org/abs/2205.05561)

    本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。

    

    我们考虑学习个性化治疗策略的问题，这些策略是外部有效或广义化的：它们在除了实验（或训练）人群外的其他目标人群中表现良好。我们首先证明，对于实验人群而言，最大化福利的策略对于实验和目标人群之间的结果（但不是特征）分布变化具有鲁棒性。然后，我们开发了新的方法来学习对结果和特征变化具有鲁棒性的策略。在这样做时，我们强调了实验人群内的治疗效果异质性如何影响策略的普适性。我们的方法可以使用实验或观察数据（其中治疗是内生的）。我们的许多方法可以使用线性规划实现。

    We consider the problem of learning personalized treatment policies that are externally valid or generalizable: they perform well in other target populations besides the experimental (or training) population from which data are sampled. We first show that welfare-maximizing policies for the experimental population are robust to shifts in the distribution of outcomes (but not characteristics) between the experimental and target populations. We then develop new methods for learning policies that are robust to shifts in outcomes and characteristics. In doing so, we highlight how treatment effect heterogeneity within the experimental population affects the generalizability of policies. Our methods may be used with experimental or observational data (where treatment is endogenous). Many of our methods can be implemented with linear programming.
    

