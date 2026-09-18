# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prediction-Powered Smoothing and Validation for Disaggregated AI Evaluation](https://arxiv.org/abs/2609.20758) | 本文提出预测驱动平滑（PP-S）及其跨分类体系借力扩展（PP-TS），利用贝叶斯小区域估计方法为标签稀少领域的AI分解式评估提供精确的点估计和区间估计，并推导了新的近似无偏基于设计的交叉验证分数用于模型验证。 |
| [^2] | [Instance-Optimal Adaptive Location Estimation via Multiscale Mid-Summaries](https://arxiv.org/abs/2609.20749) | 该论文提出一种无需知道噪声分布形状的自适应位置估计器，利用多尺度中间摘要，在具有非递减危险率的对称单峰密度类上同时达到 Le Cam 两点速率给出的逐实例最优估计精度。 |
| [^3] | [Robust Multi-Task Learning for Principal Component Analysis](https://arxiv.org/abs/2609.20733) | 提出了鲁棒的多任务PCA方法，利用任务间的相似性结构改进特征空间估计，并能对异常任务实现估计误差的最优鲁棒性，达到极小极大最优收敛速率。 |
| [^4] | [Epidemiological Causal Graph Identification: Challenges, Identifiability and Algorithms](https://arxiv.org/abs/2609.20676) | 该论文证明了有序分布节点与指数族分布节点之间的因果边方向在一般参数值下是可识别的，将有序-泊松模型的可识别性结果推广到更广泛的指数族分布，并提出了基于分数的穷举搜索和掩码连续优化两种因果发现算法。 |
| [^5] | [TAP Accuracy Below the Fluctuation Scale and Universal Posterior Geometry in Spherical Linear Models](https://arxiv.org/abs/2609.20577) | 本文在Marchenko–Pastur谱正则性条件下，证明了球面线性模型中TAP自由能逼近精度可达 $O_P(p^{-1})$（低于自然波动尺度 $O_P(p^{-1/2})$），并在所有全局TAP最大化子上一致刻画了后验几何。 |
| [^6] | [Parallelism, critical windows, and separations among diffusion language models](https://arxiv.org/abs/2609.20539) | 本文首次对掩码扩散、均匀扩散与高斯扩散三类主流扩散语言模型的并行生成能力进行了细粒度理论比较，证明均匀扩散和高斯扩散同样能以前向传播次数与分布对偶总相关（可远小于上下文长度）成比例的方式完成采样，而此前仅掩码扩散具备这一性质。 |
| [^7] | [Sharp spectral norm concentration of sparse random tensors](https://arxiv.org/abs/2609.20520) | 本文证明了稀疏伯努利随机张量谱范数的锐利集中不等式，即当 $np\ge c\log n$ 时谱范数以高概率被 $C\sqrt{np}$ 控制，消除了以往结果中的对数因子，并将结果推广至非齐次采样情形及随机超图模型的无对数第二特征值界。 |
| [^8] | [Resolution limits for process comparison from event data](https://arxiv.org/abs/2609.20489) | 本文证明了基于事件日志随机语言的标准过程挖掘方法无法从数据中区分并发与顺序行为，刻画了这种分辨率极限，并指出可以通过活动起止时间或以对象为中心的记录等被随机语言丢弃的证据来恢复这种区分。 |
| [^9] | [Online Supervised Dimension Reduction with Random Features: Diagnostics and Computational Trade-offs](https://arxiv.org/abs/2609.20454) | 本文为基于随机特征的在线核监督主成分分析（OKSPCA）提供了一致性、集中性与扰动的理论诊断，并通过六个基准实验揭示：精确优化监督谱目标并不能保证得到精确的总体子空间或更优的预测表示，从而阐明了目标优化精度、子空间恢复精度与预测性能之间的计算权衡。 |
| [^10] | [The Bias of Nonlinear Two-Time-scale Stochastic Approximation under Constant Step-Sizes](https://arxiv.org/abs/2609.20409) | 本文首次对常数步长下的非线性双时间尺度随机逼近给出了紧的均方误差与偏差上界 $O(\alpha+\beta^2/\alpha^2)$，并通过分离各项误差来源阐明了 $\beta^2/\alpha^2$ 偏差项的产生机制。 |
| [^11] | [Model-based Bootstrap for Offline Policy Evaluation in Tabular Reinforcement Learning](https://arxiv.org/abs/2609.20389) | 该论文提出了一种基于模型的自助法框架，通过从估计的MDP重新生成轨迹来对表格型强化学习中的离线策略评估进行不确定性量化，从而克服了经典重采样方法在鲁棒性、可扩展性和有限样本有效性上的局限。 |
| [^12] | [Near-Optimal Pure Single-Loop Extragradient Method for Strongly Convex--Strongly Concave Minimax Optimization](https://arxiv.org/abs/2609.20327) | 提出了一种参数固定的纯单循环阻尼外梯度方法用于强凸-强凹极小极大优化，无需内层求解或重启策略，即可以 O(√(κ_x κ_y) log(1/ε)) 的近最优梯度查询复杂度实现末次迭代线性收敛。 |
| [^13] | [Counterexamples and Sufficient Conditions: Comments on "Optimally-Transported Generalized Method of Moments"](https://arxiv.org/abs/2609.20260) | 本文对Schennach & Starck提出的最优传输广义矩方法（OTGMM）估计量的核心定理给出反例，证明其定理2-6在原假设条件下不成立，并探讨使这些结论成立的充分条件。 |
| [^14] | [Foundations of Stochastic Lexical Calculus: Semantic Descent and Random Dynamics on Probability Simplices](https://arxiv.org/abs/2609.20207) | 本文建立了一个可观测的理论框架，给出了语言导出的概率能够唯一支持语义状态更新的充要条件，并在平均收缩条件下证明了概率单纯形上随机递归的存在性、唯一性与稳定性，从而构建了一种无需将内部演算归因于语言模型的随机词汇演算。 |
| [^15] | [When Does Retrieval Help Time-Series Forecasting?](https://arxiv.org/abs/2609.20193) | 该研究证明检索插件在时间序列预测中的收益并非源于其机制本身，而是取决于回看窗口长度与主导季节周期的关系，在该关系合适时简单的周期重复基线即可击败标准模型和检索插件。 |
| [^16] | [Equivalence Between Nested Gibbs Measures and Log-Linear Combinations of Gibbs Measures](https://arxiv.org/abs/2609.19988) | 本文证明了通过嵌套运算（改变参考测度）得到的吉布斯测度与通过归一化对数线性组合得到的吉布斯测度之间的等价性，从而统一了对吉布斯概率测度的这两种运算。 |
| [^17] | [Error bounds in Sobolev norms for approximations with norm constrained ReLU neural networks](https://arxiv.org/abs/2609.19937) | 该论文将路径范数约束ReLU神经网络的逼近理论从一致逼近推广到Sobolev范数逼近，分别针对浅层网络和深层网络给出了以$W^{1,p}$范数度量的逼近误差界，且深层网络的误差界无需对函数光滑性附加限制。 |
| [^18] | [Dual-Axis Policy Optimization for LLM Agents: Bayesian Feedback Attribution and Trajectory Mass Normalization](https://arxiv.org/abs/2609.19830) | 提出双轴策略优化框架BATON，通过贝叶斯反馈归因优化轨迹内反馈利用、通过轨迹质量归一化优化轨迹间目标聚合，在多个智能体基准测试中跨模型规模均取得最强性能。 |
| [^19] | [Learn Your Own Thoughts: Abstract Token Curriculum](https://arxiv.org/abs/2609.19717) | 提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。 |
| [^20] | [Improving Sample Efficiency in Peptide-HLA Binding Prediction with Hybrid Quantum-Classical Neural Networks](https://arxiv.org/abs/2609.19642) | 提出了一种混合量子-经典神经网络（HQNN），利用量子电路的归纳偏置优势，在训练数据极其有限的肽-HLA结合预测任务中，在所有数据规模下均全面优于参数量相同的经典CNN基线。 |
| [^21] | [Portfolio-Based Constrained Multi-Objective Bayesian Optimization for Materials Design](https://arxiv.org/abs/2609.19550) | 该论文提出将约束多目标贝叶斯优化中的采集函数选择转化为自适应策略选择问题，通过UCB多臂老虎机和大语言模型驱动的多智能体系统两种控制器，在材料设计中同时实现了发现可行候选材料与完善帕累托前沿的竞争性性能。 |
| [^22] | [Compressed Active Subspaces for Scalable Bayesian Inference](https://arxiv.org/abs/2609.19539) | 本文提出压缩主动子空间（CAS）方法，通过结构化等距嵌入先将模型参数映射到压缩空间再构建主动子空间，大幅降低内存开销，使大规模模型的可扩展贝叶斯推断成为可能。 |
| [^23] | [Next-token functional estimation](https://arxiv.org/abs/2609.19529) | 针对时间依赖数据，本文提出“留窗口法”估计量用于估计下一词元泛函（如惊喜概率、测试误差等），克服了留一法在时间依赖下不一致的缺陷，并在平稳 β-混合过程下以参数速率收敛。 |
| [^24] | [Null importance: Disentangling relevance for interpretable machine learning](https://arxiv.org/abs/2609.19511) | 本文提出基于“零重要性”的统一框架，厘清了可解释机器学习中特征重要性所隐含的多种本质上不同的相关性概念，并在算法公平性和基因组扰动建模中展示了这些区分的实际意义。 |
| [^25] | [Sharpness-Aware Minimization (SAM) Improves Classification Accuracy of Bacterial Raman Spectral Data Enabling Portable Diagnostics](https://arxiv.org/abs/2609.19453) | 本论文将锐度感知最小化（SAM）应用于细菌拉曼光谱分类任务，在无需复杂预处理的情况下将分类准确率提升高达10.5%，并增强了模型在有限数据集上的泛化能力，为便携式抗生素耐药性诊断奠定了基础。 |
| [^26] | [Stable Policy Learning](https://arxiv.org/abs/2609.19418) | 本文证明算法稳定性是刻画策略学习中期望福利与抽样风险之间权衡的核心因素，并提出“策略投票装袋”方法，通过对多个子样本的处理决策投票取平均来降低抽样风险。 |
| [^27] | [Learning Submanifolds for Subsequent Inference on Random Dot Product Graphs, Part 1: Theory](https://arxiv.org/abs/2609.19357) | 提出了一种半监督决策规则框架，利用辅助数据和Isomap流形学习来学习随机点积图潜在位置的未知低维支撑流形，并证明随着辅助数据量的增加，半监督规则的风险收敛于最优先知规则的风险。 |
| [^28] | [When a High Score Is an Illusion: Certifying Genuine versus Repackaged Forecasting Skill](https://arxiv.org/abs/2609.19223) | 该论文揭示了重复使用评估观测数据会虚增预测与结果秩对比之间的关联、使高分成为假象，并证明对每个映射对施加零加权参考重叠是认证真实预测技能在所有允许映射和分布下均匀成立的充要条件，同时给出无偏的三轨迹核估计方法。 |
| [^29] | [Not All Nodes Are Created Equal: Homophily-Aware Stratification for Stable GNN Evaluation](https://arxiv.org/abs/2609.19210) | 该论文指出，仅按类别分层的交叉验证不足以稳定图神经网络评估，因为数据划分间局部邻域同质性分布的差异会系统性影响消息传递行为并夸大评估方差，为此提出了同质性感知的分层划分方法以实现更可靠的GNN比较。 |
| [^30] | [Federated Soft Clustering via Generalized Total Variation Minimization](https://arxiv.org/abs/2609.19202) | 该论文提出基于广义全变差最小化的联邦软聚类框架，系统比较了三种模型差异度量方法（无需组件匹配的KL散度和闭式MMD，以及需要组件匹配的欧氏距离），并通过同步投影梯度优化，为基于MMD的实例提供了收敛到驻点的理论保证。 |
| [^31] | [Optimal Value Inference for Reinforcement Learning](https://arxiv.org/abs/2609.09981) | 该论文提出了一种基于Neyman正交性的去偏估计方法，通过softmax近似的自诱导贝尔曼方程构建冗余参数，实现了强化学习中最优价值的有效统计推断，且在视界发散和行为策略随时间变化的情况下依然保持渐近正态性。 |
| [^32] | [Robust Bayesian Inference for Unnormalized Models with Mixed-Domain Data](https://arxiv.org/abs/2609.01783) | 提出SME-BETEL半参数贝叶斯框架，将得分匹配估计方程与贝叶斯指数倾斜经验似然相结合，无需计算归一化常数和学习率校准即可对含混合域数据的非归一化模型进行稳健贝叶斯推断，并通过Bernstein-von Mises定理保证了模型误设下不确定性量化的渐近校准性。 |
| [^33] | [A Finite Sample Analysis for Quantile Temporal Difference Learning in Distributional Reinforcement Learning](https://arxiv.org/abs/2608.27313) | 本文首次为表格型分布强化学习中的同步分位数时序差分学习提供了全局有限样本保证，通过分离稳定性机制，证明了其收敛速率对分位数数量无多项式依赖。 |
| [^34] | [GRAS: Guided Reduced-Variance Proposals and Adaptive Selection for Training-Free Reward Alignment in Discrete Diffusion](https://arxiv.org/abs/2608.26585) | 本文提出GRAS方法，通过Rao-Blackwell化提议和自适应温度选择，在不增加降噪器成本的情况下，显著提升离散扩散模型免训练奖励对齐的效率和稳定性。 |
| [^35] | [Spending Scarce Confirmatory PET Measurements: Target-Aligned Validation in A4/LEARN](https://arxiv.org/abs/2608.22223) | 本文提出了一种目标对齐的PET验证策略，通过结合目标影响和残差不确定性来优化稀缺确认性测量的分配，避免在影响弱的受试者上浪费资源。 |
| [^36] | [A Quantum/Classical Example Oracle Separation for Making Things Up](https://arxiv.org/abs/2608.11648) | 本研究首次证明，在Oracle模型下，存在某些分布只能被量子示例学习者高效生成，而经典示例学习者无法做到，从而揭示了量子示例的独特优势。 |
| [^37] | [Rapid mixing for Gibbs measures in Riemannian manifolds](https://arxiv.org/abs/2606.13453) | 本文识别了黎曼流形上朗之万动力学快速混合到吉布斯测度的条件（涉及流形曲率、逆温度和鞍点逃逸方向），并证明满足这些条件时可实现与流形维度呈多项式关系的混合时间。 |
| [^38] | [Fast Training of Mixture-of-Experts for Time Series Forecasting via Expert Loss Integration](https://arxiv.org/abs/2605.10330) | 提出一种融合专家特定损失与部分在线学习策略的自适应混合专家框架，解决了小门控权重导致的优化难题，在保持计算效率的同时显著提升了时间序列预测性能。 |
| [^39] | [Domain Elastic Transform: Bayesian Function Registration for High-Dimensional Scientific Data](https://arxiv.org/abs/2603.21235) | 该论文提出域弹性变换（DET），一种无网格的贝叶斯概率框架，通过联合空间-函数似然引导的弹性变形建模，在完全无监督的条件下直接对齐不规则稀疏流形上高维科学数据（如空间转录组学基因表达）的几何与功能信号，无需分箱或体素化处理。 |
| [^40] | [Score-based diffusion models for severely ill-posed problems in diffuse optical tomography](https://arxiv.org/abs/2602.03449) | 本文针对扩散光学层析成像这一严重不适定的逆问题，提出了一种由学习成分与基于模型的成分相结合的混合分数正则化策略，以提升基于分数的扩散模型在真实实验测量条件下的重建质量。 |
| [^41] | [On the Inherent Privacy Amplification of Missing Data](https://arxiv.org/abs/2602.01928) | 该论文提出了一个将缺失数据融入差分隐私的新框架，从形式上证明并刻画了数据缺失对隐私保护的固有放大效应，即缺失特征本身天然地增强了个人隐私保护。 |
| [^42] | [Why $\beta_1 = \beta_2$ Is Dynamically Special in Adam](https://arxiv.org/abs/2601.21739) | 本文揭示了Adam优化器中 $\beta_1 = \beta_2$ 在动力学上特殊的具体机制：连续时间极限下，归一化更新中与两个记忆时间差成正比的幅度滞后项恰好在两参数相等时消失，使得对角线区域成为结构上不存在失配诱发响应的唯一情形。 |
| [^43] | [Spherical Cauchy Variational Autoencoders: Heavy Angular Tails and Exact KL Evaluation](https://arxiv.org/abs/2506.21278) | 提出球面柯西分布作为超球面变分自编码器的后验分布，兼具重角尾特性和精确的KL散度解析计算能力，克服了von Mises-Fisher分布需要拒绝采样和Power Spherical分布密度在对跖点归零的缺陷。 |
| [^44] | [Out-of-Sample Embedding with Proximity Data: Projection versus Restricted Reconstruction](https://arxiv.org/abs/2505.06756) | 本文综述了基于邻近性数据的样本外嵌入的各种核方法，证明它们均可归结为投影或受限重构这两种基本策略之一，其中受限重构策略可简化为只需一维搜索的非线性优化问题。 |
| [^45] | [When fairness metrics fail: A utility-based perspective on $\varepsilon$-fairness](https://arxiv.org/abs/2405.09360) | 该论文提出了一个将决策后果纳入公平性评估的基于效用的框架，并证明一个决策过程即使满足ε-公平性概率度量，在考虑结果效用时仍可能达到最大程度的不公平。 |
| [^46] | [Estimation of multiple mean vectors in high dimension](https://arxiv.org/abs/2403.15038) | 通过凸组合的方法估计高维空间中不同概率分布的多维均值，引入了两种权重确定策略：一种通过测试程序识别低方差的相邻均值，提出了封闭形式插补公式；另一种通过最小化二次风险的上置信界确定权重，通过理论分析得出方法对经验均值的二次风险改进，在维度渐近的角度上渐近地接近 Oracle（Minimax）改进。 |

# 详细

[^1]: 用于分解式AI评估的预测驱动平滑与验证

    Prediction-Powered Smoothing and Validation for Disaggregated AI Evaluation

    [https://arxiv.org/abs/2609.20758](https://arxiv.org/abs/2609.20758)

    本文提出预测驱动平滑（PP-S）及其跨分类体系借力扩展（PP-TS），利用贝叶斯小区域估计方法为标签稀少领域的AI分解式评估提供精确的点估计和区间估计，并推导了新的近似无偏基于设计的交叉验证分数用于模型验证。

    

    评估一个AI系统需要进行分解式评估，因为其性能在不同领域（如基准测试任务类型或已部署智能体的对话类型）之间存在差异。穷举测试成本高昂，因此评估依赖于一个带有标签的单元样本。我们将评估集视为有限总体，寻求对每个领域均值的精确点估计和区间估计。直接估计方法（包括预测驱动推断PPI）仅使用该领域自身的标签，在标签稀少的情况下精度不足。小区域估计（small area estimation）解决了这一问题，我们在其基础上开发了一个集成估计与验证的完整工作流程。在估计方面，我们提出了预测驱动平滑，这是一个拟合到每个领域预测驱动估计值的贝叶斯模型，并进一步扩展为可在报告分类体系间借力的版本（PP-TS）。在验证方面，我们推导了一种新的近似无偏的基于设计的交叉验证分数，用于选择……

    arXiv:2609.20758v1 Announce Type: cross  Abstract: Evaluating an AI system requires disaggregated assessment, as performance varies across domains such as benchmark task types or conversation types in deployed agents. Exhaustive testing is expensive, so evaluation rests on a sample of labeled units. We treat the evaluation set as a finite population and seek accurate point and interval estimates of each domain mean. Direct estimators, including prediction-powered inference (PPI), use only a domain's own labels and are imprecise where labels are few. Small area estimation addresses this problem, and we build on it to develop an integrated workflow for estimation and validation. For estimation, we propose prediction-powered smoothing (PP-S), a Bayesian model fit to each domain's prediction-powered estimate, with an extension that borrows strength across a reporting taxonomy (PP-TS). For validation, we derive a new, approximately unbiased design-based cross-validation score for choosing a
    
[^2]: 基于多尺度中间摘要的实例最优自适应位置估计

    Instance-Optimal Adaptive Location Estimation via Multiscale Mid-Summaries

    [https://arxiv.org/abs/2609.20749](https://arxiv.org/abs/2609.20749)

    该论文提出一种无需知道噪声分布形状的自适应位置估计器，利用多尺度中间摘要，在具有非递减危险率的对称单峰密度类上同时达到 Le Cam 两点速率给出的逐实例最优估计精度。

    

    位置估计在不同噪声分布下呈现出显著不同的有限样本行为：正则分布族通常给出根号 \(n\) 的收敛速率，而紧支撑分布则可能允许更快的、由边界驱动的速率。我们提出疑问：在不了解密度形状的前提下，单个估计器能否像知晓底层位置族的先知（oracle）那样，自适应地达到逐实例最优的估计速率。对于具有对称对数凹噪声密度 \(f\) 的已知位置族，样本量为 \(n\)、失败概率为 \(\delta\) 时的最优位置估计误差已被证明是 Le Cam 的两点速率：\[ \sup\left\{r>0:\mathsf{H}^2\left(f_0, f_{2r}\right)\lesssim \frac{\log(1/\delta)}{n}\right\}. \] 当位置族未知时，我们提出了一种与形状无关的估计器，它在所有具有非递减危险率的对称单峰密度类（一个严格更……）上同时达到这一先知基准。

    arXiv:2609.20749v1 Announce Type: cross  Abstract: Location estimation exhibits markedly different finite-sample behavior across noise distributions: regular families typically yield root-\(n\) rates, whereas compactly supported laws may admit faster, boundary-driven rates. We question whether a single estimator, without knowledge of the density's shape, can adapt to the instance-wise optimal estimation rate, as an oracle that knows the underlying location family can.   For a known location family with symmetric log-concave noise density \(f\), the optimal location estimation error with sample size \(n\) under failure probability \(\delta\) is known to be Le Cam's two-point rate: \[ \sup\left\{r>0:\mathsf{H}^2\left(f_0, f_{2r}\right)\lesssim \frac{\log(1/\delta)}{n}\right\}. \] When the location family is unknown, we propose a shape-agnostic estimator that attains this oracle benchmark simultaneously over all symmetric unimodal densities with non-decreasing hazard rates, a class strict
    
[^3]: 鲁棒的多任务主成分分析学习

    Robust Multi-Task Learning for Principal Component Analysis

    [https://arxiv.org/abs/2609.20733](https://arxiv.org/abs/2609.20733)

    提出了鲁棒的多任务PCA方法，利用任务间的相似性结构改进特征空间估计，并能对异常任务实现估计误差的最优鲁棒性，达到极小极大最优收敛速率。

    

    主成分分析（PCA）是从高维数据中学习低维结构的基本工具。当数据从多个来源收集时，潜在的任务分布可能表现出未知程度的相似性，其中一些任务甚至可能来自任意分布。我们提出了新的多任务PCA方法，利用任务间的相似性结构来改进特征空间估计，同时对异常任务保持鲁棒性。我们建立了非渐近收敛速率，并证明所提出的方法在一系列情形下达到了极小极大最优速率。其中一种方法基于Chen、Gao和Ren（2018）提出的矩阵深度概念，能够实现估计误差对异常任务比例的最优依赖关系，解决了鲁棒多任务学习中的一个关键挑战。大量的模拟实验和真实数据分析证明了所提方法的有效性。

    arXiv:2609.20733v1 Announce Type: cross  Abstract: Principal component analysis (PCA) is a fundamental tool for learning low-dimensional structure from high-dimensional data. When data are collected from multiple sources, the underlying task distributions may exhibit unknown degrees of similarity, with some tasks potentially arising from arbitrary distributions. We propose new multi-task PCA procedures that exploit similarity structure across tasks to improve eigenspace estimation while remaining robust to outlier tasks. We establish non-asymptotic convergence rates and show that the proposed procedures attain minimax optimal rates in a range of regimes. One of the procedures builds on the matrix-depth notion of Chen, Gao, and Ren (2018) and can achieve the optimal dependence of the estimation error on the proportion of outlier tasks, addressing a key challenge in robust multi-task learning. Extensive simulations and real-data analyses demonstrate the effectiveness of the proposed meth
    
[^4]: 流行病学因果图识别：挑战、可识别性与算法

    Epidemiological Causal Graph Identification: Challenges, Identifiability and Algorithms

    [https://arxiv.org/abs/2609.20676](https://arxiv.org/abs/2609.20676)

    该论文证明了有序分布节点与指数族分布节点之间的因果边方向在一般参数值下是可识别的，将有序-泊松模型的可识别性结果推广到更广泛的指数族分布，并提出了基于分数的穷举搜索和掩码连续优化两种因果发现算法。

    

    从观测数据中进行因果发现是统计学和机器学习的基础，然而在没有干预的情况下确定因果方向需要结构性假设。现有的可识别性研究主要集中于加性噪声模型下的连续变量，往往忽略了包含有序尺度、计数和连续测量的混合数据集。本文研究了有向无环图中节点遵循有序分布（通过有序logit模型）或正则单参数指数族分布的因果发现问题。我们证明了对于一般参数值，有序节点与指数族节点之间的边方向是分布可识别的。我们的发现将之前关于有序-泊松模型的结果推广到了更广泛的指数族分布。在计算方面，我们引入了基于分数的穷举搜索方法以及一种使用掩码的连续优化框架。

    arXiv:2609.20676v1 Announce Type: new  Abstract: Causal discovery from observational data is fundamental to statistics and machine learning, yet determining causal direction without interventions necessitates structural assumptions. Existing identifiability research primarily focuses on continuous variables under additive noise models, often neglecting mixed datasets containing ordinal scales, counts, and continuous measurements. This paper investigates causal discovery in Directed Acyclic Graphs (DAGs) where nodes follow either an ordinal distribution (via an ordered logit model) or a regular one-parameter exponential family distribution. We prove that the edge direction between an ordinal and an exponential family node is distributionally identifiable for generic parameter values. Our findings generalize previous Ordinal-Poisson results to the broader exponential family. Computationally, we introduce a score-based exhaustive search and a masked continuous optimization framework using
    
[^5]: 球面线性模型中低于波动尺度的TAP精度与普适后验几何

    TAP Accuracy Below the Fluctuation Scale and Universal Posterior Geometry in Spherical Linear Models

    [https://arxiv.org/abs/2609.20577](https://arxiv.org/abs/2609.20577)

    本文在Marchenko–Pastur谱正则性条件下，证明了球面线性模型中TAP自由能逼近精度可达 $O_P(p^{-1})$（低于自然波动尺度 $O_P(p^{-1/2})$），并在所有全局TAP最大化子上一致刻画了后验几何。

    

    我们研究了在环境维度与样本量成比例增长时的贝叶斯最优球面线性模型，并对设计矩阵施加了定量的Marchenko–Pastur谱正则性条件。该条件可被元素标准化且四阶矩有限的归一化独立同分布设计所满足，但不要求元素间相互独立，也无需对奇异向量施加任何条件。在此条件下，我们证明了定量的全温度TAP逼近，并刻画了后验几何。对于自然的有限纵横比TAP泛函，归一化球面自由能与TAP最优值之差为 $O_P(p^{-1})$。二者与其显式确定性等价形式之差均在 $O_P(p^{-1/2})$ 以内，且该波动尺度是紧致（sharp）的。在所有全局TAP最大化子上一致地，到球面后验均值的归一化平方欧氏距离为 $O_P(p^{-1})$。此外，我们还证明了数据依赖邻域（此处摘要原文截断）。

    arXiv:2609.20577v1 Announce Type: cross  Abstract: We study the Bayes-optimal spherical linear model as the ambient dimension and sample size grow proportionally, under a quantitative Marchenko--Pastur spectral-regularity condition on the design. This condition is satisfied by normalized i.i.d. designs with standardized entries of finite fourth moment, but does not require entrywise independence or impose conditions on the singular vectors. Under this condition, we prove a quantitative all-temperature TAP approximation and characterize the posterior geometry. For the natural finite-aspect-ratio TAP functional, the normalized spherical free energy and the TAP optimum differ by $O_P(p^{-1})$. Each is within $O_P(p^{-1/2})$ of its explicit deterministic equivalent, and this fluctuation scale is sharp. Uniformly over all global TAP maximizers, the normalized squared Euclidean distance to the spherical posterior mean is $O_P(p^{-1})$. We also prove that the posterior mass outside a data-dep
    
[^6]: 扩散语言模型中的并行性、临界窗口与分离性

    Parallelism, critical windows, and separations among diffusion language models

    [https://arxiv.org/abs/2609.20539](https://arxiv.org/abs/2609.20539)

    本文首次对掩码扩散、均匀扩散与高斯扩散三类主流扩散语言模型的并行生成能力进行了细粒度理论比较，证明均匀扩散和高斯扩散同样能以前向传播次数与分布对偶总相关（可远小于上下文长度）成比例的方式完成采样，而此前仅掩码扩散具备这一性质。

    

    扩散大语言模型的一个广受欢迎的卖点在于其并行性能力：即能够以远高于自回归模型的效率生成文本序列，后者每个 token 都需要一次前向传播。然而，在众多相互竞争的 dLLM 范式之中——从掩码扩散到均匀扩散再到高斯扩散——对于这些不同方案在并行性方面如何比较，原理性的理解仍然有限。在本工作中，我们对这三种主流方法的并行性能力开展了细粒度的比较研究，并证明了以下结果：均匀扩散和高斯扩散可以在前向传播次数随底层分布的对偶总相关缩放的情况下完成采样，对偶总相关是一种内在复杂度的度量，其数值可以远小于上下文长度；而在此之前，只有掩码扩散被证明能够实现这一点。此外，对于某一族随机经验测度，我们表明（摘要截断）……

    arXiv:2609.20539v1 Announce Type: new  Abstract: A popular selling point of diffusion large language models (dLLMs) is their capacity for parallelism: the ability to generate sequences of text far more efficiently than autoregressive models, which require one forward pass per token. Yet among the many competing paradigms for dLLMs, from masked to uniform to Gaussian diffusion, principled understanding of how these different proposals compare in parallelism remains limited. In this work, we initiate a fine-grained comparison of the capacity for parallelism among these three leading approaches and prove the following:   - Uniform and Gaussian diffusion can sample in a number of forward passes which scales with the dual total correlation of the underlying distribution, a measure of intrinsic complexity which can be much smaller than the context length. Previously, it was only known how to achieve this using masked diffusion.   - For a certain family of random empirical measures, we show t
    
[^7]: 稀疏随机张量谱范数的锐利集中性

    Sharp spectral norm concentration of sparse random tensors

    [https://arxiv.org/abs/2609.20520](https://arxiv.org/abs/2609.20520)

    本文证明了稀疏伯努利随机张量谱范数的锐利集中不等式，即当 $np\ge c\log n$ 时谱范数以高概率被 $C\sqrt{np}$ 控制，消除了以往结果中的对数因子，并将结果推广至非齐次采样情形及随机超图模型的无对数第二特征值界。

    

    我们证明了具有独立伯努利项的稀疏随机张量谱范数的锐利集中不等式。设 $T$ 是维度为 $n\times\cdots\times n$ 的 $k$ 阶张量，其各项独立服从伯努利$(p)$ 分布，其中 $k$ 固定。对于任意 $c,r>0$，我们证明当 $np\ge c\log n$ 时，$\|T-\mathbb E T\|\le C_{k,r,c}\sqrt{np}$ 以至少 $1-n^{-r}$ 的概率成立。我们将该界推广到具有确定性逐项权重的非齐次伯努利采样。这一结果消除了 Zhou 和 Zhu（2021）工作中的对数因子。证明遵循 Kahn–Szemerédi 轻-重分解方法，并对重元组部分给出了精细估计。我们还为 Friedman 和 Wigderson（1995）的随机超图模型得到了一个不含对数因子的第二特征值界。

    arXiv:2609.20520v1 Announce Type: cross  Abstract: We prove a sharp concentration inequality for the spectral norm of sparse random tensors with independent Bernoulli entries. Let $T$ be an order-$k$ tensor of dimension $n\times\cdots\times n$ with independent Bernoulli$(p)$ entries, where $k$ is fixed. For any $c,r>0$, we show that $\|T-\mathbb E T\|\le C_{k,r,c}\sqrt{np}$ with probability at least $1-n^{-r}$ whenever $np\ge c\log n$. We extend this bound to inhomogeneous Bernoulli sampling with deterministic entrywise weights. This removes the logarithmic factor in the work of Zhou and Zhu (2021). The proof follows the Kahn--Szemer\'edi light--heavy decomposition with a refined estimate on the heavy tuple part. We also obtain a log-free second eigenvalue bound for the random hypergraph model of Friedman and Wigderson (1995).
    
[^8]: 基于事件数据的过程比较分辨率极限

    Resolution limits for process comparison from event data

    [https://arxiv.org/abs/2609.20489](https://arxiv.org/abs/2609.20489)

    本文证明了基于事件日志随机语言的标准过程挖掘方法无法从数据中区分并发与顺序行为，刻画了这种分辨率极限，并指出可以通过活动起止时间或以对象为中心的记录等被随机语言丢弃的证据来恢复这种区分。

    

    一家医院同时进行血液检查和影像检查，另一家医院则按先后顺序进行，且两种顺序出现的频率相同。了解实际发生了什么，以及它是如何被记录在数据中的，对所有运营管理者至关重要。在过程挖掘中，标准方法是构建事件日志，并尝试以数据驱动的方式发现并发和顺序过程。我们证明了这种基于事件日志随机语言的标准方法，只能报告其发现算法的假设，因为每一个这样的日志都可以被一个完全没有并发性的模型同样好地解释。此外，在获取任何数据之前，我们刻画了数据在何时能够以及何时无法区分并发行为。当数据无法区分时，这种区分可以从随机语言所丢弃的证据中恢复，例如活动开始和结束的时间，或在单次执行中固定顺序的以对象为中心的记录。

    arXiv:2609.20489v1 Announce Type: cross  Abstract: One hospital runs bloods and imaging at the same time. Another runs them one after the other, in either order, equally often. Knowing which actually happened, and how it is recorded in data, is critical for all operational managers. In process mining, the standard approach is to construct an event log, and attempt to discover concurrent and sequential processes in a data-driven way. We show this standard approach, built on the stochastic language of an event log, reports only the assumptions of its discovery algorithm, because every such log is explained equally well by a model with no concurrency at all. Further, before any data is acquired, we characterise when data can and cannot distinguish concurrent behaviour. Where it cannot, the distinction is recoverable from evidence the stochastic language discards, such as the times at which activities start and end, or object-centric records that fix an order within an execution. The remed
    
[^9]: 基于随机特征的在线监督降维：诊断与计算权衡

    Online Supervised Dimension Reduction with Random Features: Diagnostics and Computational Trade-offs

    [https://arxiv.org/abs/2609.20454](https://arxiv.org/abs/2609.20454)

    本文为基于随机特征的在线核监督主成分分析（OKSPCA）提供了一致性、集中性与扰动的理论诊断，并通过六个基准实验揭示：精确优化监督谱目标并不能保证得到精确的总体子空间或更优的预测表示，从而阐明了目标优化精度、子空间恢复精度与预测性能之间的计算权衡。

    

    对监督谱目标的精确优化未必能产生精确的总体子空间或更好的预测表示。我们针对在线核监督主成分分析（OKSPCA）研究了这些区别，该方法将有限随机特征坐标中的中心化交叉矩与Adam式的正交基更新相结合，用于一个已确立的目标。固定映射一致性、集中性与扰动结果刻画了估计器及其精确子空间；随后通过同目标比较分别评估实际迭代过程。在六个预测基准上，性能取决于所声明的流水线：用精确的经验目标替换跟踪器后，两个回归缺陷基本保持不变。直接的分类秩模型平均而言几乎捕获了全部终端目标能量，但保存的中间状态表现出显著的几何偏差；一个受控的样本……

    arXiv:2609.20454v1 Announce Type: cross  Abstract: Accurate optimization of a supervised spectral objective need not produce an accurate population subspace or a better predictive representation. We investigate these distinctions for Online Kernel Supervised Principal Component Analysis (OKSPCA), which combines a centered cross-moment in finite random-feature coordinates with an Adam-style orthonormal basis update for an established objective. Fixed-map consistency, concentration and perturbation results describe the estimator and its exact subspace; same-target comparisons then assess the practical iterate separately. Across six predictive benchmarks, performance depends on the declared pipeline: replacing the tracker with the exact empirical target leaves the two regression deficits largely unchanged. Direct classification-rank models capture nearly all terminal objective energy on average, but a saved intermediate state exhibits substantial geometric deviation; a controlled sample-s
    
[^10]: 常数步长下非线性双时间尺度随机逼近的偏差

    The Bias of Nonlinear Two-Time-scale Stochastic Approximation under Constant Step-Sizes

    [https://arxiv.org/abs/2609.20409](https://arxiv.org/abs/2609.20409)

    本文首次对常数步长下的非线性双时间尺度随机逼近给出了紧的均方误差与偏差上界 $O(\alpha+\beta^2/\alpha^2)$，并通过分离各项误差来源阐明了 $\beta^2/\alpha^2$ 偏差项的产生机制。

    

    双时间尺度随机逼近（TTSA）是分析强化学习、优化和随机控制中耦合迭代算法的基础工具。然而，非线性双时间尺度方案的有限时间保证仍然难以获得，尤其是在常数步长的情形下。在本文中，我们研究了步长满足 $\alpha\gg\beta$ 的非线性TTSA。在标准的稳定性、正则性和马尔可夫噪声假设下，我们对两种迭代围绕其极限平衡点的均方误差和偏差给出了上界。我们的界为 $O(\alpha+\beta^2/\alpha^2)$，并证明了当 $\beta\le\alpha^{3/2}$ 时该界是紧的。我们的分析分离了初始条件、快时间尺度跟踪误差、马尔可夫依赖性以及时间尺度耦合各自的贡献，从而阐明了 $\beta^2/\alpha^2$ 项的来源。我们的结果揭示了其与先前研究的线性TTSA设定之间的定性差异。

    arXiv:2609.20409v1 Announce Type: new  Abstract: Two-timescale stochastic approximation (TTSA) is a fundamental tool for analyzing coupled iterative algorithms in reinforcement learning, optimization, and stochastic control. However, finite-time guarantees for nonlinear two-timescale schemes remain difficult to obtain, especially under constant step-sizes. In this paper, we study nonlinear TTSA with step-sizes $\alpha\gg\beta$. Under standard stability, regularity, and Markovian noise assumptions, we upper bound the mean-squared error and the bias of both iterates around their limiting equilibria. Our bounds scale as $O(\alpha+\beta^2/\alpha^2)$, which we prove to be tight when $\beta\le\alpha^{3/2}$. The analysis separates the contributions of initial conditions, fast-timescale tracking error, Markovian dependence, and timescale coupling, thereby clarifying the origin of the $\beta^2/\alpha^2$ term. Our results reveal qualitative differences from the linear TTSA setting previously stu
    
[^11]: 表格型强化学习中离线策略评估的基于模型的自助法（Bootstrap）框架

    Model-based Bootstrap for Offline Policy Evaluation in Tabular Reinforcement Learning

    [https://arxiv.org/abs/2609.20389](https://arxiv.org/abs/2609.20389)

    该论文提出了一种基于模型的自助法框架，通过从估计的MDP重新生成轨迹来对表格型强化学习中的离线策略评估进行不确定性量化，从而克服了经典重采样方法在鲁棒性、可扩展性和有限样本有效性上的局限。

    

    离线策略评估（OPE）在高风险强化学习应用中至关重要，在这些应用中，新策略必须在部署前得到可靠评估。在此类场景下，仅有点估计是不够的；有原则的不确定性量化，例如置信区间和方差估计，对于安全且具备风险意识的决策必不可少。统一这些任务的一种全面途径是估计评估误差的抽样分布。然而，现有方法往往在鲁棒性、可扩展性或有限样本有效性方面存在不足。本文提出了一种基于模型的自助法框架，用于有限时域、时变非齐次马尔可夫决策过程（MDP）中离线策略评估的不确定性量化。与依赖对完整回合进行重采样的经典自助法不同，所提方法从估计得到的MDP中重新生成轨迹，因此能够适用于范围更广的……（摘要在此处截断）

    arXiv:2609.20389v1 Announce Type: cross  Abstract: Offline policy evaluation (OPE) is crucial in high-stakes reinforcement learning applications, where new policies must be assessed reliably before deployment. In such settings, point estimates alone are insufficient; principled uncertainty quantification, such as confidence intervals and variance estimates, is essential for safe and risk-aware decision-making. A comprehensive way to unify these tasks is to estimate the sampling distribution of the evaluation error. Existing approaches, however, often suffer from limited robustness, scalability, or finite-sample validity. In this paper, we propose a model-based bootstrap framework for uncertainty quantification of OPE in finite-horizon, time-inhomogeneous Markov decision processes (MDPs). Unlike classical bootstrap methods that rely on resampling complete episodes, the proposed method regenerates trajectories from an estimated MDP and can therefore accommodate a much broader range of of
    
[^12]: 用于强凸-强凹极小极大优化的近最优纯单循环外梯度方法

    Near-Optimal Pure Single-Loop Extragradient Method for Strongly Convex--Strongly Concave Minimax Optimization

    [https://arxiv.org/abs/2609.20327](https://arxiv.org/abs/2609.20327)

    提出了一种参数固定的纯单循环阻尼外梯度方法用于强凸-强凹极小极大优化，无需内层求解或重启策略，即可以 O(√(κ_x κ_y) log(1/ε)) 的近最优梯度查询复杂度实现末次迭代线性收敛。

    

    我们研究确定性无约束设置下具有一般非线性耦合的光滑强凸-强凹极小极大优化问题。我们提出了一种纯单循环阻尼外梯度方法，其参数固定，在初始一次梯度查询后，每次迭代仅需两次新的全梯度评估。该方法使用辅助反馈递归，无需内层求解、精度调度或分阶段重启。我们建立了末次迭代的线性收敛性，并证明将到鞍点的欧氏距离平方减少到其初始值的ε倍需要 O(√(κ_x κ_y) log(2κ_x κ_y/ε)) 次全梯度查询，其中 κ_x = L/μ_x，κ_y = L/μ_y。该界通过固定的显式更新，在对数因子范围内达到了关于条件数的最优阶。数值实验证明了该方法的有效性。

    arXiv:2609.20327v1 Announce Type: cross  Abstract: We study smooth strongly convex--strongly concave minimax optimization with general nonlinear coupling in the deterministic unconstrained setting. We propose a pure single-loop damped extragradient method with fixed parameters and two new full-gradient evaluations per iteration after one initialization query. The method uses an auxiliary feedback recursion and requires no inner solves, accuracy schedules, or staged restarts. We establish last-iterate linear convergence and show that reducing the squared Euclidean distance to the saddle point to an $\varepsilon$ fraction of its initial value requires $O(\sqrt{\kappa_x\kappa_y}\log(2\kappa_x\kappa_y/\varepsilon))$ full-gradient queries, where $\kappa_x=L/\mu_x$ and $\kappa_y=L/\mu_y$. This bound attains the optimal condition-number order up to logarithmic factors through fixed explicit updates. Numerical experiments demonstrate the effectiveness of the method.
    
[^13]: 反例与充分条件：对《最优传输广义矩方法》一文的评论

    Counterexamples and Sufficient Conditions: Comments on "Optimally-Transported Generalized Method of Moments"

    [https://arxiv.org/abs/2609.20260](https://arxiv.org/abs/2609.20260)

    本文对Schennach & Starck提出的最优传输广义矩方法（OTGMM）估计量的核心定理给出反例，证明其定理2-6在原假设条件下不成立，并探讨使这些结论成立的充分条件。

    

    我们针对Schennach & Starck（2026a）提出的最优传输广义矩方法（OTGMM）估计量进行评论，并在其所述假设条件下给出针对定理2-6的反例。首先，小误差分析中所使用的假设不足以保证定理2的一致性和定理3的渐近正态性。其次，我们考察大误差分析，其中定理4声称OTGMM估计量等价于带修正矩的GMM估计量。我们证明，在标量模型中，定理4所选择的值不同于唯一的OTGMM最小化点，且违反了OTGMM样本矩约束。在一个满足定理5和定理6假设的过度识别模型中，拉格朗日乘子的第一个分量在OTGMM估计量和带修正矩的GMM估计量下具有不同的概率极限。在模型误设的情形下，OTGMM所选择的总体值取决于……（原文摘要至此截断）

    arXiv:2609.20260v1 Announce Type: cross  Abstract: We comment on the optimally-transported generalized method of moments (OTGMM) estimator proposed by Schennach & Starck (2026a) and give counterexamples to Theorems 2-6 under their stated assumptions. First, the assumptions used in the small-error analysis are insufficient for consistency in Theorem 2 and asymptotic normality in Theorem 3. Next, we consider the large-error analysis, in which Theorem 4 states that the OTGMM estimator is equivalent to a GMM estimator with modified moments. We show that in a scalar model, Theorem 4 selects a value that differs from the unique OTGMM minimizer and violates the OTGMM sample moment restriction. In an overidentified model satisfying the assumptions used in Theorems 5 and 6, the first component of the Lagrange multiplier has different probability limits under the OTGMM estimator and the GMM estimator with modified moments. Under misspecification, the population value selected by OTGMM depends on
    
[^14]: 随机词汇演算的基础：概率单纯形上的语义下降与随机动力学

    Foundations of Stochastic Lexical Calculus: Semantic Descent and Random Dynamics on Probability Simplices

    [https://arxiv.org/abs/2609.20207](https://arxiv.org/abs/2609.20207)

    本文建立了一个可观测的理论框架，给出了语言导出的概率能够唯一支持语义状态更新的充要条件，并在平均收缩条件下证明了概率单纯形上随机递归的存在性、唯一性与稳定性，从而构建了一种无需将内部演算归因于语言模型的随机词汇演算。

    

    大语言模型产生的是依赖于提示词的词汇概率，而科学系统需要对有意义状态的不确定性建模，并随证据的到来而不断更新。我们开发了一个可观测的框架，用以判定何时由语言导出的概率能够支持这种序贯状态表示。在理论上，我们定义了上下文语言的类型化可测变换，构建了一个最小的闭表示，并给出了语义更新唯一存在的充要条件。我们界定了不可消除的非闭合性与累积误差，并在平均收缩条件下证明了概率单纯形上外部随机递归的存在性、唯一性与稳定性。这些结果定义了一种随机词汇演算，而无需将某种内部演算归因于语言模型本身。在实证方面，冻结实验检验了该理论的可观测含义。原始的提示条件概率未能通过预先指定的（测试）……

    arXiv:2609.20207v1 Announce Type: new  Abstract: Large language models produce prompt-dependent probabilities over words, whereas scientific systems require uncertainty over meaningful states that can be updated as evidence arrives. We develop an observable framework for determining when language-derived probabilities support such a sequential state representation. Theoretically, we define typed measurable transformations of contextual language, construct a minimal closed representation, and give necessary and sufficient conditions for semantic updates to exist uniquely. We bound irreducible nonclosure and accumulated error, and under average contraction prove existence, uniqueness and stability of an external random recursion on a probability simplex. These results define a stochastic lexical calculus without attributing an internal calculus to the language model. Empirically, frozen experiments test the observable implications. Raw prompt-conditioned probabilities fail the prespecifi
    
[^15]: 检索何时有助于时间序列预测？

    When Does Retrieval Help Time-Series Forecasting?

    [https://arxiv.org/abs/2609.20193](https://arxiv.org/abs/2609.20193)

    该研究证明检索插件在时间序列预测中的收益并非源于其机制本身，而是取决于回看窗口长度与主导季节周期的关系，在该关系合适时简单的周期重复基线即可击败标准模型和检索插件。

    

    检索插件为深度预测器提供其回看窗口无法承载的信息。已发表的评估报告显示一致的性能提升，且各自将功劳归于自身的机制。我们证明这种收益实际上取决于运行点：即窗口长度S与主导季节周期L之间的关系，这是标准评估协议从未改变的一个维度。按该关系对评估进行分层即可揭示这一规律。在S=12时，一个简单地重复最后观测周期的简单对照方法，在七个基准中的四个上以8%到44%的MSE优势击败了六个标准骨干模型（总体而言）。它在ETTm1上击败了我们运行的最强检索插件，并在ECL上与其持平。而在三个训练集频谱缺乏集中共享周期的数据集上，它最多落后25%。对预测期、周期和窗口进行的受控合成扫描显示，收益边界与周期相关（相关性+0.71），而非与预测期相关（-0.23）。一项成对对照……

    arXiv:2609.20193v1 Announce Type: new  Abstract: Retrieval plug-ins supply a deep forecaster with information its lookback window cannot carry. Published evaluations report consistent gains, and each credits its own mechanism. We show that the benefit belongs instead to the operating point: the relation between window length $S$ and dominant seasonal period $L$, an axis the standard protocol never varies. Stratifying the evaluation by that relation exposes the regime. At $S{=}12$, a simple control that repeats the last observed period beats the six standard backbones, in aggregate, on four of seven benchmarks by $8\%$ to $44\%$ of MSE. It beats the strongest plug-in we run on ETTm1 and matches it on ECL. It is worse by up to $25\%$ on the three datasets whose training-split spectra lack a concentrated, shared period. A controlled synthetic sweep of horizon, period, and window shows the benefit boundary tracks the period (correlation $+0.71$), not the horizon ($-0.23$). A paired control
    
[^16]: 嵌套吉布斯测度与吉布斯测度对数线性组合之间的等价性

    Equivalence Between Nested Gibbs Measures and Log-Linear Combinations of Gibbs Measures

    [https://arxiv.org/abs/2609.19988](https://arxiv.org/abs/2609.19988)

    本文证明了通过嵌套运算（改变参考测度）得到的吉布斯测度与通过归一化对数线性组合得到的吉布斯测度之间的等价性，从而统一了对吉布斯概率测度的这两种运算。

    

    本文研究了吉布斯概率测度上的三种运算。第一种运算通常被称为重正化，它通过对一个吉布斯概率测度的密度的幂进行归一化来生成新的吉布斯测度。这种归一化具有双重效果：它既改变了正则化因子，又将支撑集集中在原始支撑集的一个子集内。有趣的是，这两种效果可以通过不同的参数进行独立控制。第二种运算是对若干个吉布斯概率测度的密度进行归一化的对数线性组合。第三种运算取两个吉布斯概率测度，并用前者替换后者的参考测度。因此，前者被称为“嵌套”于后者之中，从而产生一个新的吉布斯概率测度。由第二种和第三种运算所得到的测度同样是吉布斯概率测度，并且分别被证明满足……（摘要在此处截断）

    arXiv:2609.19988v1 Announce Type: cross  Abstract: In this paper, three operations on Gibbs probability measures are studied. The first operation, often referred to as renormalization, takes one Gibbs probability measure and generates a new Gibbs measure by normalizing a power of its density. This normalization has a twofold effect: it changes the regularization factor and concentrates the support within a subset of the original support. Interestingly, these effects can be independently controlled by different parameters. The second operation consists of a normalized log-linear combination of the densities of Gibbs probability measures. The third operation takes two Gibbs probability measures and changes the reference measure of the latter with the former. Hence, the former is said to be "nested" within the latter, yielding a new Gibbs probability measure. The resulting measures from the second and third operations are also Gibbs probability measures and are shown, respectively, to sol
    
[^17]: 范数约束ReLU神经网络逼近的Sobolev范数误差界

    Error bounds in Sobolev norms for approximations with norm constrained ReLU neural networks

    [https://arxiv.org/abs/2609.19937](https://arxiv.org/abs/2609.19937)

    该论文将路径范数约束ReLU神经网络的逼近理论从一致逼近推广到Sobolev范数逼近，分别针对浅层网络和深层网络给出了以$W^{1,p}$范数度量的逼近误差界，且深层网络的误差界无需对函数光滑性附加限制。

    

    最近的研究表明，光滑函数可以通过权重具有路径范数约束的ReLU神经网络得到良好的逼近。我们将这些结果从一致逼近扩展到Sobolev范数下的逼近。具体而言，我们分析了$W^{n,p}$空间中的Sobolev函数能被宽度为$W$、深度为$L$且路径范数以$K$为界的神经网络逼近的程度，其中逼近误差以$W^{1,p}$范数度量。对于深度$L=1$的浅层网络，当光滑性指标满足$n<s=(d+3)/2$且输入为$d$维时，我们推导出逼近误差界$\mathcal{O}(\max\{W^{-(n-1)/d}, K^{-(n-1)/(s-n)}\})$。对于深层网络，我们消除了对光滑性的限制，证明了当宽度$W$和深度$L$充分大时，逼近界$\mathcal{O}(K^{-(n-1)/(d+d/p+1)})$成立。

    arXiv:2609.19937v1 Announce Type: cross  Abstract: Recent studies have shown that smooth functions can be well approximated by ReLU neural networks with path norm constraint on the weights. We extend these results from uniform approximation to approximation in Sobolev norm. Specifically, we analyze how well Sobolev functions in $W^{n,p}$ can be approximated by neural networks with width $W$, depth $L$ and path norm bounded by $K$, when the approximation error is measured in the $W^{1,p}$-norm. For shallow networks with depth $L=1$, we derive the approximation error bound $\mathcal{O}(\max\{W^{-(n-1)/d}, K^{-(n-1)/(s-n)}\})$, when the smoothness index satisfies $n<s=(d+3)/2$ and the input is $d$-dimensional. For deep networks, we remove the restriction on the smoothness by showing that the approximation bound $\mathcal{O}(K^{-(n-1)/(d+d/p+1)})$ holds if the width $W$ and depth $L$ are sufficiently large.
    
[^18]: 面向LLM智能体的双轴策略优化：贝叶斯反馈归因与轨迹质量归一化

    Dual-Axis Policy Optimization for LLM Agents: Bayesian Feedback Attribution and Trajectory Mass Normalization

    [https://arxiv.org/abs/2609.19830](https://arxiv.org/abs/2609.19830)

    提出双轴策略优化框架BATON，通过贝叶斯反馈归因优化轨迹内反馈利用、通过轨迹质量归一化优化轨迹间目标聚合，在多个智能体基准测试中跨模型规模均取得最强性能。

    

    针对LLM智能体的强化学习涉及两个不同的优化维度：如何在单条轨迹内利用环境反馈，以及如何在批处理中聚合完整轨迹。我们将这两个维度形式化为“轨迹内反馈归因”和“轨迹间目标聚合”，并提出了BATON（贝叶斯归因与轨迹目标归一化），一个双轴策略优化框架。BATON的第一个轴通过贝叶斯反馈归因实现，该机制构建了以反馈为条件的选择动作后验分布；第二个轴通过轨迹质量归一化（TMN）实现，该机制为完整轨迹分配相等的优化权重。在ALFWorld、WebShop和SearchQA数据集上使用GRPO和GiGPO进行的实验表明，两个轴均能带来独立收益，且两者的结合在不同模型规模下始终取得最强的整体性能。

    arXiv:2609.19830v1 Announce Type: new  Abstract: Reinforcement learning for LLM agents involves two distinct optimization di- mensions: how environment feedback is exploited within a trajectory, and how complete trajectories are aggregated across a batch. We formulate these dimen- sions as Intra-Trajectory Feedback Attribution and Inter-Trajectory Objec- tive Aggregation, and introduce BATON (Bayesian Attribution and Trajectory Objective Normalization), a dual-axis policy optimization framework. BATON instantiates the first axis with Bayesian Feedback Attribution, which constructs a feedback-conditioned posterior over sampled actions, and the second with Trajec- tory Mass Normalization (TMN), which assigns equal optimization mass to com- plete trajectories. Experiments with GRPO and GiGPO on ALFWorld, WebShop, and SearchQA show that both axes provide independent gains and that their combi- nation consistently achieves the strongest overall performance across model scales.
    
[^19]: 学会自己的思考：抽象token课程学习

    Learn Your Own Thoughts: Abstract Token Curriculum

    [https://arxiv.org/abs/2609.19717](https://arxiv.org/abs/2609.19717)

    提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。

    

    大语言模型（LLMs）通过利用思维链（CoT）作为思考中间阶段的草稿板，已经获得了卓越的推理能力。然而，CoT技术需要对思考token进行显式监督，这需要丰富的、特定任务的数据。在这项工作中，我们提出了抽象token课程学习（Abstract Token Curriculum, ATC），这是一种新颖的课程学习框架，能够在没有直接监督或手动草稿板设计的情况下，引出有效的连续中间表示。ATC通过一系列分布逐渐增加问题复杂度，训练模型在连续表示空间中发展出内部的抽象“思维”。本文为ATC的优势及其相对于以往训练连续思维方法的长处提供了理论和实验证据。理论上，我们证明了使用ATC在单层softmax注意力机制下学习奇偶函数时……

    arXiv:2609.19717v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have achieved remarkable reasoning capabilities by utilizing chain-of-thought (CoT) as a scratchpad for intermediate stages of thinking. However, CoT techniques require explicit supervision on thinking tokens, which requires rich, task-specific data. In this work, we propose Abstract Token Curriculum (ATC), a novel curriculum learning framework that elicits effective continuous intermediate representations without direct supervision or manual scratchpad design. ATC gradually increases problem complexity through a sequence of distributions, training the model to develop internal abstract ``thoughts'' in the continuous representation space. This paper provides both theoretical and experimental evidence for the benefits of ATC and its advantages over previous methods for training continuous thoughts. Theoretically, we show that for learning parity functions with single-layer softmax attention using ATC, attent
    
[^20]: 使用混合量子-经典神经网络提高肽-HLA结合预测的样本效率

    Improving Sample Efficiency in Peptide-HLA Binding Prediction with Hybrid Quantum-Classical Neural Networks

    [https://arxiv.org/abs/2609.19642](https://arxiv.org/abs/2609.19642)

    提出了一种混合量子-经典神经网络（HQNN），利用量子电路的归纳偏置优势，在训练数据极其有限的肽-HLA结合预测任务中，在所有数据规模下均全面优于参数量相同的经典CNN基线。

    

    肽-HLA结合预测是个性化癌症免疫治疗中新抗原鉴定的关键步骤，具有重要的临床价值。然而，许多HLA等位基因可用的训练数据极其有限，这严重限制了传统方法在此任务上的表现。参数化量子电路被假设能够诱导有利于从小数据集学习的归纳偏置，但其在生物序列预测中的应用仍然探索不足。为解决这一问题，我们提出了一种专为肽-HLA结合预测设计的混合量子-经典神经网络（HQNN）。HQNN将多源生物特征编码与并行量子特征提取器及量子增强分类器相结合。在两个HLA等位基因（A*02:01和B*07:02）上，HQNN在所有训练数据规模下均优于参数量匹配的经典CNN基线，且随着训练数据的减少，性能差距进一步扩大。

    arXiv:2609.19642v1 Announce Type: cross  Abstract: Peptide-HLA binding prediction is a critical step in neoantigen identification for personalized cancer immunotherapy and holds significant clinical value. However, the training data available for many HLA alleles are extremely limited, which severely constrains the performance of conventional methods on this task. Parameterized quantum circuits are hypothesized to induce inductive biases beneficial for learning from small datasets, yet their application to biological sequence prediction remains underexplored. To address this, we propose a hybrid quantum-classical neural network (HQNN) specifically designed for peptide-HLA binding prediction. HQNN integrates multi-source biological feature encoding with parallel quantum feature extractors and a quantum-enhanced classifier. On two HLA alleles (A*02:01 and B*07:02), HQNN outperforms a parameter-matched classical CNN baseline across all training sizes, with the performance gap widening as 
    
[^21]: 基于组合策略的约束多目标贝叶斯优化材料设计

    Portfolio-Based Constrained Multi-Objective Bayesian Optimization for Materials Design

    [https://arxiv.org/abs/2609.19550](https://arxiv.org/abs/2609.19550)

    该论文提出将约束多目标贝叶斯优化中的采集函数选择转化为自适应策略选择问题，通过UCB多臂老虎机和大语言模型驱动的多智能体系统两种控制器，在材料设计中同时实现了发现可行候选材料与完善帕累托前沿的竞争性性能。

    

    材料发现与设计活动可以被表述为约束多目标贝叶斯优化（CMOBO）问题，其中每次实验决策都需要在两个相互耦合又彼此竞争的目标之间进行权衡：发现可行候选材料和完善潜在的帕累托前沿。本文将采集函数的选择重新构建为在常规采集函数与聚焦可行性的采集函数组合之上的自适应策略选择问题。这通过两个控制器实现：UCB-Bandit（一种改进的UCB多臂老虎机）和Agentic-Switch（一种由大语言模型（LLM）驱动的多智能体决策系统）。两者在五个合成基准函数和两个材料设计案例研究中与固定策略基线进行了计算机仿真对比评估。自适应策略在累积可行样本数量和可行超体积改进方面均表现出有竞争力的性能，而每个单独的采集函数……

    arXiv:2609.19550v1 Announce Type: cross  Abstract: Materials discovery and design campaigns can be formulated as constrained multi-objective Bayesian optimization (CMOBO) problems, within which each experimental decision negotiates between two coupled but competing goals: discovering feasible candidates and refining the underlying Pareto front. Here we recast acquisition-function choice as an adaptive policy-selection problem over a portfolio of conventional and feasibility-focused acquisition functions. This was done using two controllers: UCB-Bandit, a modified UCB multi-armed bandit, and Agentic-Switch, a multi-agent decision system driven by a large language model (LLM). Both were evaluated against fixed-policy baselines in silico across five synthetic benchmark functions and two materials design case studies. The adaptive policies performed competitively in terms of both cumulative feasibility count and feasible hypervolume improvement, while each individual acquisition function p
    
[^22]: 用于可扩展贝叶斯推断的压缩主动子空间

    Compressed Active Subspaces for Scalable Bayesian Inference

    [https://arxiv.org/abs/2609.19539](https://arxiv.org/abs/2609.19539)

    本文提出压缩主动子空间（CAS）方法，通过结构化等距嵌入先将模型参数映射到压缩空间再构建主动子空间，大幅降低内存开销，使大规模模型的可扩展贝叶斯推断成为可能。

    

    主动子空间方法通过识别对模型输出影响最大的参数方向并沿这些方向进行推断，为高维模型中的预测不确定性量化提供了一个框架。然而，主动子空间的构建需要存储大量全维度的模型梯度，随着模型规模的增大，这一开销变得难以承受。我们通过提出压缩主动子空间（CAS）来解决这一局限性，这是一种可扩展的方法，首先利用结构化等距嵌入将模型参数映射到压缩空间，然后在该降维后的参数化空间中构建主动子空间。我们的方法大幅降低了主动子空间构建所需的内存，使得标准主动子空间方法变得不切实际的大型模型的贝叶斯推断成为可能。我们在规模不断增大的神经网络上演示了CAS的可扩展性，同时保持了预测……

    arXiv:2609.19539v1 Announce Type: cross  Abstract: Active subspace methods provide a framework for quantifying predictive uncertainty in high-dimensional models by identifying and performing inference along parameter directions that have the greatest influence on the model output. However, the construction of active subspaces requires storing many full-dimensional model gradients, which becomes prohibitive as model size increases. We address this limitation by proposing Compressed Active Subspaces (CAS), a scalable approach that first maps the model parameters to a compressed space using a structured isometric embedding and then constructs the active subspace within this reduced parameterization. Our approach substantially reduces the memory required for active subspace construction and enables Bayesian inference for large models where standard active subspace methods become impractical. We demonstrate the scalability of CAS on neural networks of increasing size while maintaining predi
    
[^23]: 下一词元泛函估计

    Next-token functional estimation

    [https://arxiv.org/abs/2609.19529](https://arxiv.org/abs/2609.19529)

    针对时间依赖数据，本文提出“留窗口法”估计量用于估计下一词元泛函（如惊喜概率、测试误差等），克服了留一法在时间依赖下不一致的缺陷，并在平稳 β-混合过程下以参数速率收敛。

    

    假设我们观察到一个长度为 n+1 的随机变量序列的前 n 个点，并希望估计关于未观测的最后一个点与 n 个已观测训练点的经验测度的某个泛函。此类“下一词元泛函”包括：下一个词元是新词的概率（也称为惊喜概率）、下一个词元与训练点之间最小距离的尾部概率，以及在已观测点上训练的分类器的测试误差。所有这些量传统上都是通过留一法（leave-one-out）进行估计的，但该方法在时间依赖性下是不一致的。我们提出了一种“留窗口法”（leave-a-window-out）估计量，它在构造经验测度之前删除每个索引之后长度为 τ 的窗口，并在 τ=1 时退化为留一法。在自然假设下，我们证明该估计量的误差对于任何平稳 β-混合过程都以参数速率衰减。

    arXiv:2609.19529v1 Announce Type: cross  Abstract: Suppose we observe the first $n$ points of a sequence of random variables having length $n+1$, and wish to estimate a functional of the unobserved final point and the empirical measure of the $n$ observed training points. Such next-token functionals include the probability that the next token is novel (also known as the surprise probability), the tail probability of the minimum distance between the next token and training points, and the test error of a classifier trained on the observed points. All of these quantities are classically estimated by the leave-one-out method, which is inconsistent under temporal dependence. We propose a leave-a-window-out estimator, which deletes a window of length $\tau$ after each index before forming the empirical measure and reduces to leave-one-out at $\tau = 1$. Under natural assumptions, we show that the error of our estimator decays at a parametric rate for any stationary $\beta$-mixing process th
    
[^24]: 零重要性：解耦可解释机器学习中的相关性概念

    Null importance: Disentangling relevance for interpretable machine learning

    [https://arxiv.org/abs/2609.19511](https://arxiv.org/abs/2609.19511)

    本文提出基于“零重要性”的统一框架，厘清了可解释机器学习中特征重要性所隐含的多种本质上不同的相关性概念，并在算法公平性和基因组扰动建模中展示了这些区分的实际意义。

    

    特征重要性是可解释机器学习的核心，但“重要性”一词涵盖了多种本质上不同的相关性概念。我们基于零重要性发展了一个统一的视角：即在特定相关性概念下，对特征在总体层面何时无关的刻画。我们考虑了由边际和条件统计相关性、预测风险、函数不变性以及因果效应所产生的零重要性的标准概念，并展示了这些概念如何回答不同的科学问题。我们在两个这种区分尤为关键的应用中阐释了该框架：在算法公平性中，常见的公平性准则对应于不同的零重要性概念；在基因组扰动建模中，不同的相关性概念会导致关于预测模型学到了什么的不同结论。该框架连接了三个方面

    arXiv:2609.19511v1 Announce Type: cross  Abstract: Feature importance is central to interpretable machine learning, but the term "importance" encompasses several fundamentally different notions of relevance. We develop a unified perspective based on null importance: a population-level characterization of when a feature is irrelevant under a specified notion of relevance. We consider standard notions of null importance arising from marginal and conditional statistical relevance, predictive risk, functional invariance, and causal effects, and show how these notions answer different scientific questions. We illustrate the framework in two applications in which the distinction is particularly consequential: algorithmic fairness, where common fairness criteria correspond to different notions of null importance, and genomic perturbation modeling, where different notions of relevance lead to different conclusions about what a prediction model has learned. The framework connects three aspects 
    
[^25]: 锐度感知最小化（SAM）提高细菌拉曼光谱数据分类准确率，实现便携式诊断

    Sharpness-Aware Minimization (SAM) Improves Classification Accuracy of Bacterial Raman Spectral Data Enabling Portable Diagnostics

    [https://arxiv.org/abs/2609.19453](https://arxiv.org/abs/2609.19453)

    本论文将锐度感知最小化（SAM）应用于细菌拉曼光谱分类任务，在无需复杂预处理的情况下将分类准确率提升高达10.5%，并增强了模型在有限数据集上的泛化能力，为便携式抗生素耐药性诊断奠定了基础。

    

    预计到2050年，抗菌素耐药性每年将夺走1000万人的生命，而资源有限的地区受影响最为严重。拉曼光谱是一种新型的病原体诊断方法，有望在几小时内完成快速、便携的抗生素耐药性检测，而使用金标准方法则需要数天时间。然而，当前的拉曼光谱分析算法存在以下两个问题：1）在跨不同患者人群的有限数据集上无法很好地泛化；2）由于必须进行非平凡的预处理步骤（如特征提取，这对于缓解拉曼光谱数据的低质量特性至关重要），增加了算法的复杂性。在本工作中，我们使用锐度感知最小化（SAM）来解决这些局限性，在临床细菌分离株分类任务中增强模型在多种超参数设置下的泛化能力。我们证明SAM在单个（原文摘要在此处截断）上实现了高达10.5%的准确率提升。

    arXiv:2609.19453v1 Announce Type: new  Abstract: Antimicrobial resistance is expected to claim 10 million lives per year by 2050, and resource-limited regions are most affected. Raman spectroscopy is a novel pathogen diagnostic approach promising rapid and portable antibiotic resistance testing within a few hours, compared to days when using gold standard methods. However, current algorithms for Raman spectra analysis 1) are unable to generalize well on limited datasets across diverse patient populations and 2) require increased complexity due to the necessity of non-trivial pre-processing steps, such as feature extraction, which are essential to mitigate the low-quality nature of Raman spectral data. In this work, we address these limitations using Sharpness-Aware Minimization (SAM) to enhance model generalization across a diverse array of hyperparameters in clinical bacterial isolate classification tasks. We demonstrate that SAM achieves accuracy improvements of up to 10.5% on a sing
    
[^26]: 稳定策略学习

    Stable Policy Learning

    [https://arxiv.org/abs/2609.19418](https://arxiv.org/abs/2609.19418)

    本文证明算法稳定性是刻画策略学习中期望福利与抽样风险之间权衡的核心因素，并提出“策略投票装袋”方法，通过对多个子样本的处理决策投票取平均来降低抽样风险。

    

    在循证政策制定中，通常先观察一个实验样本，然后将学习到的政策建议大规模实施。从实验数据中学习到的政策在期望福利方面可能表现良好，但实验中的随机抽样可能产生福利结果较差的建议。在本文中，我们提出这样一个问题：策略学习算法应如何在期望福利与抽样风险之间取得平衡？我们的主要贡献是证明算法稳定性在刻画和应对这一权衡中起着核心作用。直观地说，如果一个策略学习算法在替换一个实验单元时其建议保持稳定，那么该算法的抽样风险有限。我们提出了一种名为“策略投票装袋”的策略学习方法，该方法在许多子样本上学习处理决策，然后将其投票平均为处理概率。相对于使用单个子样本，跨子样本平均……

    arXiv:2609.19418v1 Announce Type: cross  Abstract: In evidence-based policymaking, typically one experimental sample is observed, then a learned policy recommendation is implemented at scale. Policies learned from the experimental data can perform well in expected welfare, yet random sampling in the experiment can produce recommendations with poor welfare outcomes. In this paper, we ask: how should policy learning algorithms balance expected welfare against sampling risk? Our main contribution is to show that algorithmic stability plays a central role in characterizing and navigating the tradeoff. Intuitively, if a policy learning algorithm's recommendation remains stable when one experimental unit is replaced, then that algorithm has limited sampling risk. We propose a method for policy learning called policy-vote bagging, which learns treatment decisions on many subsamples then averages their votes into treatment probabilities. Relative to using one subsample, averaging across subsam
    
[^27]: 学习子流形以用于随机点积图的后续推断，第一部分：理论

    Learning Submanifolds for Subsequent Inference on Random Dot Product Graphs, Part 1: Theory

    [https://arxiv.org/abs/2609.19357](https://arxiv.org/abs/2609.19357)

    提出了一种半监督决策规则框架，利用辅助数据和Isomap流形学习来学习随机点积图潜在位置的未知低维支撑流形，并证明随着辅助数据量的增加，半监督规则的风险收敛于最优先知规则的风险。

    

    我们提出了一个针对随机点积图的受限推断框架，其中图的潜在位置位于一个未知的低维支撑流形上。对于一般的决策问题，我们提出了利用辅助数据来学习支撑流形的半监督决策规则。具体而言，我们的规则使用Isomap流形学习过程来构建观测图的低维欧几里得表示，在该空间中，一个等距不变的函数将点的配置映射为动作。我们研究了当从未知支撑流形采样的辅助数据量增加时所提出规则的行为。我们证明，随着辅助样本量的增加，半监督规则的风险收敛于一个先知规则的风险，该先知规则依赖于能够从支撑流形中提取出的最大低维欧几里得结构。示例、应用和模拟研究……

    arXiv:2609.19357v1 Announce Type: cross  Abstract: We propose a framework for restricted inference on random dot product graphs whose latent positions lie on an unknown low-dimensional support manifold. For general decision problems, we propose semisupervised decision rules that use auxiliary data to learn the support manifold. Specifically, our rules use the Isomap manifold learning procedure to construct a low-dimensional Euclidean representation of the observed graph, in which space an isometrically invariant function maps configurations of points to actions. We study the behavior of the proposed rules as the quantity of auxiliary data sampled from the unknown support manifold increases. We show that, as the auxiliary sample size increases, the risk of the semisupervised rule converges to the risk of an oracle rule that relies on the maximal amount of low-dimensional Euclidean structure that can be extracted from the support manifold. Examples, applications, and simulation studies a
    
[^28]: 当高分只是幻觉：认证真实的与重新包装的预测技能

    When a High Score Is an Illusion: Certifying Genuine versus Repackaged Forecasting Skill

    [https://arxiv.org/abs/2609.19223](https://arxiv.org/abs/2609.19223)

    该论文揭示了重复使用评估观测数据会虚增预测与结果秩对比之间的关联、使高分成为假象，并证明对每个映射对施加零加权参考重叠是认证真实预测技能在所有允许映射和分布下均匀成立的充要条件，同时给出无偏的三轨迹核估计方法。

    

    秩（排名）取决于用于比较的观测数据。重复使用这些观测数据可能会在预测与结果的秩对比之间引入关联，即使被评估的预测和结果本身保持不变。我们刻画了能够保持特定总体秩对比之间关联的分配方式，包括预测秩减去基线秩与结果秩减去基线秩之间的比较。在独立训练的条件下，整条轨迹从共同分布中独立采样，而每条轨迹内部允许任意形式的依赖。期望得分可分解为其目标值与映射对之间的显式交互项。在重新分配参考时，我们保持学习到的映射、参考分布和系数行和固定。结果表明，对每个映射对而言，零加权参考重叠是在所有允许的映射和分布上均匀保持关联的充要条件。一个无偏的三轨迹核被用于估计该交互项；……

    arXiv:2609.19223v1 Announce Type: cross  Abstract: Ranks depend on the observations used for comparison. Reusing those observations can add association between forecast and outcome rank contrasts even when the evaluated forecast and outcome stay fixed. We characterize assignments that preserve association between specified population-rank contrasts, including forecast rank minus baseline rank compared with outcome rank minus baseline rank. Conditional on independent training, whole trajectories are sampled independently from a common law, with unrestricted dependence within each trajectory. The expected score separates into its target and an explicit interaction between map pairs. When reassigning references, we keep the learned maps, reference law and coefficient row sums fixed. Zero weighted reference overlap for every map pair is necessary and sufficient for preservation uniformly over permitted maps and laws. An unbiased three-trajectory kernel estimates the interaction; independen
    
[^29]: 并非所有节点生而平等：面向稳定GNN评估的同质性感知分层方法

    Not All Nodes Are Created Equal: Homophily-Aware Stratification for Stable GNN Evaluation

    [https://arxiv.org/abs/2609.19210](https://arxiv.org/abs/2609.19210)

    该论文指出，仅按类别分层的交叉验证不足以稳定图神经网络评估，因为数据划分间局部邻域同质性分布的差异会系统性影响消息传递行为并夸大评估方差，为此提出了同质性感知的分层划分方法以实现更可靠的GNN比较。

    

    图神经网络被广泛用于直推式节点分类，其准确率通常在随机划分的训练/验证/测试集上进行测量。研究表明，同一数据集的不同随机划分会导致报告的准确率发生显著偏移，使得已发表的架构间比较变得不可靠。在非图场景中，经典的解决方法是分层k折交叉验证，它确保每个测试折都能反映数据集的完整类别分布。我们认为，仅凭类别分层对图数据而言是不够的：节点并非孤立而是相互连接的，局部邻域同质性分布不同的折会使模型暴露于系统性不同的关系条件中，这些条件直接影响消息传递行为。由此产生的跨折变化反映了每个划分的同质性构成，使报告的方差膨胀到超出模型行为本身所能解释的程度。

    arXiv:2609.19210v1 Announce Type: cross  Abstract: Graph neural networks are widely used for transductive node classification, with accuracy typically measured on randomly drawn train/validation/test splits. Reported accuracy has been shown to shift substantially across different random splits of the same dataset, making published comparisons between architectures unreliable. The classical remedy in non-graph settings is stratified $k$-fold cross-validation, which ensures each test fold reflects the full class distribution of the dataset. We argue that class stratification alone is insufficient for graphs: nodes are not isolated but connected, and folds that differ in their distribution of local neighbourhood homophily expose the model to systematically different relational conditions that directly affect message-passing behaviour. The resulting cross-fold variation reflects the homophily composition of each split, inflating reported variance beyond what model behaviour alone would pro
    
[^30]: 通过广义全变差最小化实现联邦软聚类

    Federated Soft Clustering via Generalized Total Variation Minimization

    [https://arxiv.org/abs/2609.19202](https://arxiv.org/abs/2609.19202)

    该论文提出基于广义全变差最小化的联邦软聚类框架，系统比较了三种模型差异度量方法（无需组件匹配的KL散度和闭式MMD，以及需要组件匹配的欧氏距离），并通过同步投影梯度优化，为基于MMD的实例提供了收敛到驻点的理论保证。

    

    我们研究了联邦学习（FL）网络中设备上的联邦软聚类问题，这些设备各自持有私有的本地数据集，并拟合个性化的高斯混合模型（GMM）。广义全变差最小化通过一个图正则化项来耦合局部最大似然问题，该正则化项惩罚相连节点模型之间的差异。差异度量方法的选择是一个关键的设计决策：我们比较了模型参数之间的平方欧氏距离（该方法需要进行组件匹配）与两种直接比较本地模型分布、因而无需匹配的度量方法：蒙特卡洛近似的Kullback-Leibler（KL）散度和具有闭式解的最大均值差异（MMD）。由此产生的三种GTVMin实例均通过同步投影梯度更新进行优化；对于平滑的MMD实例，我们提供了收敛到驻点的理论保证。我们对它们的计算（原文在此处截断）

    arXiv:2609.19202v1 Announce Type: cross  Abstract: We study federated soft clustering over federated learning (FL) networks of devices that each hold a private local dataset and fit a personalized Gaussian mixture model (GMM). Generalized total variation minimization (GTVMin) couples the local maximum likelihood problems through a graph regularizer that penalizes a discrepancy between the models of connected nodes. The choice of discrepancy measure is a key design decision: we compare a squared Euclidean distance between model parameters, which requires component matching, with two measures that compare the local model distributions directly and hence need no matching: a Monte-Carlo approximated Kullback-Leibler (KL) divergence and a closed-form maximum mean discrepancy (MMD). All three resulting GTVMin instances are optimized by synchronous projected gradient updates; for the smooth MMD instance we provide a convergence guarantee to stationary points. We characterize their computation
    
[^31]: 强化学习中的最优价值推断

    Optimal Value Inference for Reinforcement Learning

    [https://arxiv.org/abs/2609.09981](https://arxiv.org/abs/2609.09981)

    该论文提出了一种基于Neyman正交性的去偏估计方法，通过softmax近似的自诱导贝尔曼方程构建冗余参数，实现了强化学习中最优价值的有效统计推断，且在视界发散和行为策略随时间变化的情况下依然保持渐近正态性。

    

    我们研究强化学习中离线推断最优价值的问题。我们将两个新的冗余参数推导为自诱导贝尔曼方程的不动点，其中我们用最大贝尔曼算子的softmax对应形式进行近似。我们通过Neyman正交性提出了一个去偏估计量，并在视界发散的情况下建立了其渐近正态性，即使行为策略随时间变化，只要这些冗余参数达到许多机器学习方法所能实现的统计收敛速率即可。我们为这些冗余参数提供了具体的估计程序，并证明它们能够带来有效的统计推断。合成实验验证了我们推断方法的数值性能，我们还在现实决策问题中实现了该方法，包括自行车重新调配和AI智能体工具使用。

    arXiv:2609.09981v1 Announce Type: new  Abstract: We study offline inference for the optimal value in reinforcement learning. Two new nuisances are derived as fixed points of a self-induced Bellman equation, in which we approximate the maximum Bellman operator by its softmax correspondence. We propose a debiased estimator through the Neyman orthogonality and establish its asymptotic normality under diverging horizons even when the behavior policy changes with time, as long as the nuisances have the statistical rates that can be achieved by many machine learning methods. We provide a concrete estimating procedure for these nuisances and show they can lead to valid inference. Synthetic experiments validate the numerical performance of our inference method, and we implement it in real-life decision-making problems, including bike repositioning and AI agentic tool use.
    
[^32]: 混合域数据下非归一化模型的稳健贝叶斯推断

    Robust Bayesian Inference for Unnormalized Models with Mixed-Domain Data

    [https://arxiv.org/abs/2609.01783](https://arxiv.org/abs/2609.01783)

    提出SME-BETEL半参数贝叶斯框架，将得分匹配估计方程与贝叶斯指数倾斜经验似然相结合，无需计算归一化常数和学习率校准即可对含混合域数据的非归一化模型进行稳健贝叶斯推断，并通过Bernstein-von Mises定理保证了模型误设下不确定性量化的渐近校准性。

    

    许多统计模型涉及依赖于参数的归一化常数，这些常数在计算上难以处理，给标准贝叶斯推断造成了重大障碍。尽管现有的基于似然的算法通常可以绕过这些常数，但在模型误设的情况下，其不确定性量化可能校准不佳。为了应对这些挑战，我们提出了SME-BETEL，这是一种半参数贝叶斯框架，将得分匹配估计方程与贝叶斯指数倾斜经验似然相结合。所得的后验分布避免了归一化常数的计算，且不需要学习率校准。我们建立了得分匹配估计量的一致性和渐近正态性，并证明了SME-BETEL后验的Bernstein-von Mises定理。这些结果表明，SME-BETEL可信集在渐近意义上与得分匹配估计量的抽样变异性校准一致。

    arXiv:2609.01783v1 Announce Type: cross  Abstract: Many statistical models involve parameter-dependent normalizing constants that are computationally intractable, creating substantial obstacles to standard Bayesian inference. Although existing likelihood-based algorithms can often circumvent these constants, their uncertainty quantification may be poorly calibrated under model misspecification. To address these challenges, we propose SME-BETEL, a semiparametric Bayesian framework that combines score matching estimating equations with Bayesian exponentially tilted empirical likelihood. The resulting posterior avoids evaluation of normalizing constants and does not require learning-rate calibration. We establish consistency and asymptotic normality of the score matching estimator, and prove a Bernstein-von Mises theorem for the SME-BETEL posterior. These results show that SME-BETEL credible sets are asymptotically calibrated to the sampling variability of the score matching estimator, yi
    
[^33]: 分布强化学习中分位数时序差分学习的有限样本分析

    A Finite Sample Analysis for Quantile Temporal Difference Learning in Distributional Reinforcement Learning

    [https://arxiv.org/abs/2608.27313](https://arxiv.org/abs/2608.27313)

    本文首次为表格型分布强化学习中的同步分位数时序差分学习提供了全局有限样本保证，通过分离稳定性机制，证明了其收敛速率对分位数数量无多项式依赖。

    

    arXiv:2608.27313v1 公告类型：交叉 摘要：我们在表格型分布强化学习中，为同步分位数时序差分学习（QTD）建立了全局有限样本保证。证明分离了两种稳定性机制。一个基于奖励累积分布函数的顺序单调性和分布贝尔曼算子的$W_\infty$收缩的全局比较论证，将任意初始化的迭代带入一个局部邻域。在该邻域内，我们对QTD均值场进行线性化。其雅可比矩阵是一个非奇异$M$-矩阵，相关的正半群允许进行方差敏感的鞅分析。对于步长$\alpha_t=c(t+1)^{-a}$，其中$a\in(1/2,1)$，主导的最后迭代波动阶为$\widetilde O\bigl(T^{-a/2}/\sqrt{1-\gamma}\bigr)$，且对分位数数量没有多项式依赖。确定性瞬态和所需的预热时间仍可能依赖于最小的贝尔曼误差。

    arXiv:2608.27313v1 Announce Type: cross  Abstract: We establish a global finite-sample guarantee for synchronous quantile temporal-difference learning (QTD) in tabular distributional reinforcement learning. The proof separates two stability mechanisms. A global comparison argument, based on the order monotonicity of reward cumulative distribution functions and the $W_\infty$ contraction of the distributional Bellman operator, brings an arbitrarily initialized iterate into a local neighborhood. Inside that neighborhood, we linearize the QTD mean field. Its Jacobian is a nonsingular $M$-matrix, and the associated positive semigroup permits a variance-sensitive martingale analysis. For stepsizes $\alpha_t=c(t+1)^{-a}$ with $a\in(1/2,1)$, the leading last-iterate fluctuation is of order $\widetilde O\bigl(T^{-a/2}/\sqrt{1-\gamma}\bigr)$ and has no polynomial dependence on the number of quantiles. The deterministic transient and the required burn-in can still depend on the smallest Bellman-
    
[^34]: GRAS：用于离散扩散模型免训练奖励对齐的引导式低方差提议与自适应选择

    GRAS: Guided Reduced-Variance Proposals and Adaptive Selection for Training-Free Reward Alignment in Discrete Diffusion

    [https://arxiv.org/abs/2608.26585](https://arxiv.org/abs/2608.26585)

    本文提出GRAS方法，通过Rao-Blackwell化提议和自适应温度选择，在不增加降噪器成本的情况下，显著提升离散扩散模型免训练奖励对齐的效率和稳定性。

    

    arXiv:2608.26585v1 公告类型：新 摘要：离散扩散模型已成为序列数据生成器中强大且广泛采用的一类，而在推理时无需重新训练即可将其引导至下游奖励，正变得越来越重要。这种免训练引导通过梯度引导、搜索或两者结合来实现。我们研究了结合模式，并识别出通常运行方式中的两个弱点：引导提议从单个噪声样本估计梯度，而搜索则以固定温度重新采样粒子，忽略了奖励在每个去噪步骤中的分布。我们通过一组小改动解决了这两个问题，且不增加降噪器成本。对于提议，我们通过Rao-Blackwell化揭示来降低可微奖励的估计方差，并为不可微奖励使用留一法基线；对于搜索，我们将每步值标准化为组相对优势，并证明它会坍缩为单个活跃项。

    arXiv:2608.26585v1 Announce Type: new  Abstract: Discrete diffusion models have become a strong, widely adopted class of generators for sequence data, and steering them toward a downstream reward at inference time, without any retraining, is increasingly important. Such training-free steering is done by gradient guidance, by search, or by combining the two. We study the combined regime and identify two weaknesses in how it is usually run: the guided proposal estimates its gradient from a single noisy sample, and the search then resamples particles at a fixed temperature that ignores how rewards spread across each denoising step. We address both with a small set of changes that add no denoiser cost. For the proposal, we lower the estimator variance with a Rao-Blackwellized reveal for differentiable rewards and a leave-one-out baseline for non-differentiable ones; for the search, we standardize the per-step values into a group-relative advantage and prove it collapses to a single active 
    
[^35]: 稀缺确认性PET测量的合理分配：A4/LEARN中的目标对齐验证

    Spending Scarce Confirmatory PET Measurements: Target-Aligned Validation in A4/LEARN

    [https://arxiv.org/abs/2608.22223](https://arxiv.org/abs/2608.22223)

    本文提出了一种目标对齐的PET验证策略，通过结合目标影响和残差不确定性来优化稀缺确认性测量的分配，避免在影响弱的受试者上浪费资源。

    

    抗淀粉样蛋白疗法和血液生物标志物正在将阿尔茨海默病的诊疗流程转变为两阶段测量工作流：首先使用成本较低的信息进行广泛筛查，然后在能支持最终报告决策的关键环节使用稀缺的确认性淀粉样蛋白测量。淀粉样蛋白正电子发射断层扫描（PET）仍是此类用于评估淀粉样蛋白负担的协议测量手段之一，但PET机位、试验预算和面向支付方的证据包都是有限的。本文提出了一个明确的操作性问题：何时简单的透明PET验证足够，何时拟合残差不确定性评分值得增加复杂度？对于加权协议目标，验证受试者i的一阶价值是目标影响与残差协议不确定性的乘积。通用不确定性采样仅使用第二个因素，可能将PET测量分配给那些难以预测但对科学、临床或商业目标影响较弱的受试者。

    arXiv:2608.22223v1 Announce Type: cross  Abstract: Anti-amyloid therapies and blood-based biomarkers are changing Alzheimer disease workups into a two-stage measurement workflow: screen broadly with cheaper information, then spend scarce confirmatory amyloid measurements where they support the decision that will be reported. Amyloid positron-emission tomography (PET) remains one such protocol measurement for amyloid burden, but PET slots, trial budgets, and payer-facing evidence packages are finite. This paper asks a deliberately operational question: when is simple transparent PET validation enough, and when is a fitted residual-uncertainty score worth the added complexity? For a weighted protocol target, the first-order value of validating subject i is the product of target influence and residual protocol uncertainty. Generic uncertainty sampling uses only the second factor and can spend PET measurements on subjects that are hard to predict but weak for the scientific, clinical, or c
    
[^36]: 量子与经典示例的Oracle分离：关于“制造”内容的研究

    A Quantum/Classical Example Oracle Separation for Making Things Up

    [https://arxiv.org/abs/2608.11648](https://arxiv.org/abs/2608.11648)

    本研究首次证明，在Oracle模型下，存在某些分布只能被量子示例学习者高效生成，而经典示例学习者无法做到，从而揭示了量子示例的独特优势。

    

    我们研究了在PAC学习框架中，量子示例相对于经典示例的能力。这里，我们考虑两种学习算法，它们都能访问量子计算，但一种获得量子示例，而另一种获得经典示例。此前尚不清楚是否存在学习任务，其中前者能高效完成而后者不能。我们的主要结果是，相对于一个Oracle，存在一些分布，可以由访问量子示例的量子学习者高效生成，但无法由仅访问经典示例的量子学习者生成，这为肯定回答此问题取得了进展。

    arXiv:2608.11648v1 Announce Type: cross  Abstract: We study the power of quantum examples, as compared to classical examples, in the PAC learning framework. Here, we have two learning algorithms, both with access to quantum computation, but one gets quantum examples, whereas the other gets classical examples. It was previously unknown whether there were learning tasks that can be efficiently performed but not by the latter. Our primary result is to show that relative to an oracle, there are distributions that can be efficiently generated by a quantum learner with access to quantum examples, but not by a quantum learner with access to only classical examples, making progress to answering this question in the affirmative.
    
[^37]: 黎曼流形中吉布斯测度的快速混合

    Rapid mixing for Gibbs measures in Riemannian manifolds

    [https://arxiv.org/abs/2606.13453](https://arxiv.org/abs/2606.13453)

    本文识别了黎曼流形上朗之万动力学快速混合到吉布斯测度的条件（涉及流形曲率、逆温度和鞍点逃逸方向），并证明满足这些条件时可实现与流形维度呈多项式关系的混合时间。

    

    本文分析了黎曼流形上的朗之万动力学。研究确定了确保合适的对数Sobolev不等式存在（即快速混合到吉布斯测度）的条件。这些条件涉及流形的曲率、逆温度以及从鞍点逃逸的方向，并排除了贫瘠高原和虚假局部极小值。我们证明，当满足这些条件时，可以实现与流形维度呈多项式关系的混合时间。该结果是通过黎曼浸没的定义域和像空间中朗之万过程之间的关系获得的。这种关系本身可能具有独立的研究价值。

    arXiv:2606.13453v2 Announce Type: replace-cross  Abstract: Langevin dynamics on Riemannian manifolds is analyzed. Conditions ensuring the existence of a suitable logarithmic Sobolev inequality (rapid mixing to the Gibbs measure) are identified. These conditions involve the curvature of the manifold, the inverse temperature, escaping directions from saddle points, and exclude barren plateaus and spurious local minima. We show that when these conditions are met, mixing times polynomial in the dimension of the manifold are achievable. This result is obtained through a relation between Langevin processes in the domain and in the image of a Riemannian submersion. Such a relation can be of independent interest.
    
[^38]: 通过专家损失集成实现时间序列预测混合专家模型的快速训练

    Fast Training of Mixture-of-Experts for Time Series Forecasting via Expert Loss Integration

    [https://arxiv.org/abs/2605.10330](https://arxiv.org/abs/2605.10330)

    提出一种融合专家特定损失与部分在线学习策略的自适应混合专家框架，解决了小门控权重导致的优化难题，在保持计算效率的同时显著提升了时间序列预测性能。

    

    我们提出了一种新颖的自适应混合专家框架用于时间序列预测，该框架通过引入专家特定损失来解决由小门控权重引起的优化问题，专家特定损失为每个专家提供独立于门控分配权重的直接学习信号。具体而言，总体目标由基础预测损失和专家特定损失组成，使各个专家的预测误差能够与总体预测误差一起直接影响参数更新。该框架还鼓励不同的专家从数据的不同时间片段中学习。所提出的框架进一步与部分在线学习策略相结合，实现模型参数的高效增量更新。通过将专家级损失信息与部分在线优化相结合，所提出的方法在保持计算效率的同时提高了预测性能。

    arXiv:2605.10330v2 Announce Type: replace-cross  Abstract: We propose a novel adaptive Mixture-of-Experts (MoE) framework for time series forecasting that addresses the optimization problem arising from small gating weights by incorporating expert-specific losses, which provide each expert with a direct learning signal independent of the gate-assigned weight. Specifically, the overall objective comprises the base forecasting loss and expert-specific losses, allowing individual expert prediction errors to directly influence parameter updates alongside the aggregate forecasting error. The framework also encourages different experts to learn from different temporal segments of the data. The proposed framework is further combined with a partial online learning strategy that enables efficient incremental updates of model parameters. By integrating expert-level loss information with partial online optimization, the proposed method improves forecasting performance while retaining computationa
    
[^39]: 域弹性变换：面向高维科学数据的贝叶斯函数配准

    Domain Elastic Transform: Bayesian Function Registration for High-Dimensional Scientific Data

    [https://arxiv.org/abs/2603.21235](https://arxiv.org/abs/2603.21235)

    该论文提出域弹性变换（DET），一种无网格的贝叶斯概率框架，通过联合空间-函数似然引导的弹性变形建模，在完全无监督的条件下直接对齐不规则稀疏流形上高维科学数据（如空间转录组学基因表达）的几何与功能信号，无需分箱或体素化处理。

    

    非刚性配准传统上分为点集配准（对齐稀疏几何结构）和图像配准（对齐规则网格上的连续强度场）。这种二分法对于新兴科学数据（如空间转录组学）具有局限性，因为这类数据中，高维向量值函数（如基因表达）定义在不规则的稀疏流形上。因此，研究人员要么必须通过体素化牺牲单细胞分辨率，要么为了几何对齐而忽略功能信号。我们提出了域弹性变换（DET），这是一个无网格的概率框架，可以联合对齐几何与函数。通过将数据视为不规则域上的函数，DET无需分箱即可直接配准高维信号。在广义贝叶斯框架下，域变形被建模为由联合空间-函数似然引导的弹性运动。DET是完全无监督的。

    arXiv:2603.21235v2 Announce Type: replace  Abstract: Nonrigid registration is conventionally divided into point set registration, which aligns sparse geometries, and image registration, which aligns continuous intensity fields on regular grids. This dichotomy is limiting for emerging scientific data such as spatial transcriptomics, where high-dimensional vector-valued functions, e.g., gene expression, are defined on irregular sparse manifolds. Researchers must therefore either sacrifice single-cell resolution through voxelization or ignore functional signals in favor of geometric alignment.   We propose Domain Elastic Transform (DET), a grid-free probabilistic framework that jointly aligns geometry and function. By treating data as functions on irregular domains, DET registers high-dimensional signals directly without binning. Within a generalized Bayesian formulation, domain deformation is modeled as elastic motion guided by a joint spatial-functional likelihood. DET is fully unsuperv
    
[^40]: 面向扩散光学层析成像中严重不适定问题的基于分数的扩散模型

    Score-based diffusion models for severely ill-posed problems in diffuse optical tomography

    [https://arxiv.org/abs/2602.03449](https://arxiv.org/abs/2602.03449)

    本文针对扩散光学层析成像这一严重不适定的逆问题，提出了一种由学习成分与基于模型的成分相结合的混合分数正则化策略，以提升基于分数的扩散模型在真实实验测量条件下的重建质量。

    

    基于分数的扩散模型是近来发展的一种用于贝叶斯逆问题后验采样的框架，通过利用从经验数据中学习到的具有强表达能力的先验分布，能够在逆问题中实现高质量的重建。尽管此类模型在实证中表现优异，并日益受到机器学习界的关注，但它们在使用实验测量数据的现实且严重不适定的逆问题中的行为仍未得到充分探索。扩散光学层析成像（DOT）是一个逆边值问题，它利用近红外光的边界测量来恢复生物组织中空间变化的吸收和散射参数。该问题高度不适定，对测量噪声和建模误差尤其敏感。我们通过构建由学习成分和基于模型的成分组成的混合分数，引入了一种正则化策略。我们证明……

    arXiv:2602.03449v2 Announce Type: replace-cross  Abstract: Score-based diffusion models are a recently developed framework for posterior sampling in Bayesian inverse problems, enabling high-quality reconstructions in inverse problems by leveraging expressive prior distributions learned from empirical data. Despite their strong empirical performance and growing interest within the machine learning community, their behaviour in realistic, severely ill-posed inverse problems with experimental measurement data remains under-explored. Diffuse optical tomography (DOT) is an inverse boundary value problem that uses boundary measurements of near-infrared light to recover spatially varying absorption and scattering parameters in biological tissue. The problem is highly ill-posed and particularly sensitive to both measurement noise and modelling errors. We introduce a regularization strategy by constructing a mixed score consisting of a learned component and a model-based component. We show that
    
[^41]: 论缺失数据的固有隐私放大效应

    On the Inherent Privacy Amplification of Missing Data

    [https://arxiv.org/abs/2602.01928](https://arxiv.org/abs/2602.01928)

    该论文提出了一个将缺失数据融入差分隐私的新框架，从形式上证明并刻画了数据缺失对隐私保护的固有放大效应，即缺失特征本身天然地增强了个人隐私保护。

    

    隐私保护在医学和金融等许多高风险领域至关重要，在这些领域中，敏感数据必须在不损害个人机密性的前提下进行分析。与此同时，这些应用所涉及的数据集往往由于无响应或数据损坏等原因而天然存在缺失值。传统上，缺失数据是通过其对统计效率和模型性能的影响来研究的。事实上，缺失数据减少了分析者可获得的信息，并可能降低模型的最终效用。在这项工作中，我们采取了另一种方法，从隐私保护的角度来研究缺失数据。直观地说，当特征缺失时，关于个人的信息被揭示得更少，这表明数据缺失可能在本质上增强隐私。我们在一个将缺失数据融入差分隐私的新颖框架中形式化了这一直觉。本质上，我们的方法考虑了……

    arXiv:2602.01928v3 Announce Type: replace-cross  Abstract: Privacy preservation is critical in many high-stakes domains such as medicine and finance, where sensitive data must be analyzed without compromising individual confidentiality. At the same time, these applications often involve datasets with inherent missing values due to non-response or data corruption for example. Missing data is traditionally analyzed through its impact on statistical efficiency and model performance. In fact, it reduces the information available to analysts and can degrade the final utility of the model. In this work, we take an alternative approach and study missing data through the lens of privacy preservation. Intuitively, when features are missing, less information is revealed about individuals, suggesting that data missingness could inherently enhance privacy. We formalize this intuition within a novel framework that integrates missing data into differential privacy. In essence, our approach accounts 
    
[^42]: 为什么在Adam优化器中 $\beta_1 = \beta_2$ 在动力学上是特殊的

    Why $\beta_1 = \beta_2$ Is Dynamically Special in Adam

    [https://arxiv.org/abs/2601.21739](https://arxiv.org/abs/2601.21739)

    本文揭示了Adam优化器中 $\beta_1 = \beta_2$ 在动力学上特殊的具体机制：连续时间极限下，归一化更新中与两个记忆时间差成正比的幅度滞后项恰好在两参数相等时消失，使得对角线区域成为结构上不存在失配诱发响应的唯一情形。

    

    Adam优化器在大规模训练的核心地位已持续近十年，但其两个动量参数的作用仍然知之甚少。近期研究表明，将 $\beta_{1}=\beta_{2}$ 绑定取相同值时，即使把两个记忆尺度合并为一个，Adam依然能保持其强大性能，这引出了一个基本问题：当两个记忆被绑定时，动力学上究竟有什么变得特殊？我们识别出了一个具体的机制。在连续时间极限下，每个归一化更新坐标可以分解为一个符号分量、一个与两个记忆时间之差成正比的显式幅度滞后项，以及额外的过渡项、曲率项和非线性比率项。这一滞后通道恰好在 $\beta_{1}=\beta_{2}$ 时消失，使得对角线（两参数相等）成为这种失配所引起的响应在结构上不存在的唯一区域。在真实训练梯度上进行的全历史离散分解也恢复了这种组成上的变化：绑定后的更新以符……

    arXiv:2601.21739v3 Announce Type: replace-cross  Abstract: Adam has been at the core of large-scale training for almost a decade, yet the role of its two momentum parameters remains poorly understood. Recent work shows that tying $\beta_{1}=\beta_{2}$ can preserve Adam's strong performance despite collapsing two memory scales into one, raising a basic question: what becomes dynamically special when the memories are tied? We identify a concrete mechanism. In the continuous-time limit, each normalized-update coordinate decomposes into a sign component, an explicit magnitude-lag term proportional to the difference between the two memory times, and additional transition, curvature, and nonlinear ratio terms. This lag channel vanishes exactly when $\beta_{1}=\beta_{2}$, making the diagonal the unique regime in which this mismatch-induced response is structurally absent. A full-history discrete decomposition on real training gradients recovers this change in composition: tied updates are sig
    
[^43]: 球面柯西变分自编码器：重角尾与精确KL散度计算

    Spherical Cauchy Variational Autoencoders: Heavy Angular Tails and Exact KL Evaluation

    [https://arxiv.org/abs/2506.21278](https://arxiv.org/abs/2506.21278)

    提出球面柯西分布作为超球面变分自编码器的后验分布，兼具重角尾特性和精确的KL散度解析计算能力，克服了von Mises-Fisher分布需要拒绝采样和Power Spherical分布密度在对跖点归零的缺陷。

    

    重尾后验在欧氏变分自编码器中十分常见，其中Student族无需额外机制即可放宽高斯假设。然而，球面上一直缺乏可与之媲美的选择。von Mises-Fisher分布需要修正贝塞尔函数和拒绝采样器，而Power Spherical分布则通过强制密度在对跖点处归零来换取封闭形式的表达。我们开发了球面柯西分布作为一种超球面后验分布，无需做出上述任何一种妥协。通过球极投影，该分布可映射为多元Student分布；借助一个默比乌斯变换，可以将均匀球面采样转化为基于内积、范数和标量运算的精确后验样本。同一变换也解决了正则化项的计算问题。沿着采样映射评估密度，将相对于均匀先验的KL散度简化为一个标量期望，其展开式在每个偶数维环境空间中均能终止，仅剩下一个对数积分需要计算。

    arXiv:2506.21278v4 Announce Type: replace-cross  Abstract: Heavy-tailed posteriors are routine in Euclidean variational autoencoders, where the Student family relaxes the Gaussian without new machinery. The sphere has had no comparable option. Von Mises-Fisher distribution needs modified Bessel functions and a rejection sampler, and Power Spherical buys its closed forms by forcing the density to vanish at the antipode. We develop the spherical Cauchy distribution as a hyperspherical posterior that needs neither compromise. Stereographic projection carries it to a multivariate Student law, and a M\"obius transformation turns a uniform spherical draw into an exact posterior sample from inner products, norms, and scalar arithmetic. The same transformation settles the regularizer. Evaluating the density along the sampling map reduces the Kullback-Leibler (KL) divergence to the uniform prior to a scalar expectation whose expansion terminates in every even ambient dimension, leaving one loga
    
[^44]: 基于邻近性数据的样本外嵌入：投影与受限重构之比较

    Out-of-Sample Embedding with Proximity Data: Projection versus Restricted Reconstruction

    [https://arxiv.org/abs/2505.06756](https://arxiv.org/abs/2505.06756)

    本文综述了基于邻近性数据的样本外嵌入的各种核方法，证明它们均可归结为投影或受限重构这两种基本策略之一，其中受限重构策略可简化为只需一维搜索的非线性优化问题。

    

    利用邻近性（相似性或相异性）数据来实现“在向量图中添加一个点”的问题最早由 J.C. Gower 于1968年研究。此后，人们提出了许多方法——主要是核方法——来解决这一后来被称为*样本外嵌入*的问题。我们综述了我们所遇到的各种核方法，并证明其中每一种方法都可以从两种相互竞争的策略之一推导出来：*投影*或*受限重构*。投影可以类比为在主成分分析中添加一个点的著名公式。受限重构则提出了一个不同的挑战：如何在保持先前获得的向量图固定不变的情况下，最好地近似重新进行整个多变量分析。这一策略会产生一个非线性优化问题，该问题可以简化为一维搜索。

    arXiv:2505.06756v2 Announce Type: replace-cross  Abstract: The problem of using proximity (similarity or dissimilarity) data for the purpose of "adding a point to a vector diagram" was first studied by J.C. Gower in 1968. Since then, a number of methods -- mostly kernel methods -- have been proposed for solving what has come to be called the problem of *out-of-sample embedding*. We survey the various kernel methods that we have encountered and show that each can be derived from one or the other of two competing strategies: *projection* or *restricted reconstruction*. Projection can be analogized to a well-known formula for adding a point to a principal component analysis. Restricted reconstruction poses a different challenge: how to best approximate redoing the entire multivariate analysis while holding fixed the vector diagram that was previously obtained. This strategy results in a nonlinear optimization problem that can be simplified to a unidimensional search. Various circumstances
    
[^45]: 当公平性度量失效时：基于效用的ε-公平性视角

    When fairness metrics fail: A utility-based perspective on $\varepsilon$-fairness

    [https://arxiv.org/abs/2405.09360](https://arxiv.org/abs/2405.09360)

    该论文提出了一个将决策后果纳入公平性评估的基于效用的框架，并证明一个决策过程即使满足ε-公平性概率度量，在考虑结果效用时仍可能达到最大程度的不公平。

    

    决策过程中的公平性通常使用概率度量来量化。然而，这些度量未必能反映决策对受影响个体和群体所产生的后果。我们开发了一个基于效用的框架，将这些后果纳入公平性评估中。我们的主要结果表明，一个决策过程即使满足ε-公平性，一旦考虑其结果所关联的效用，仍可能在最大程度上不公平。为了应对假阴性信息不可用的应用场景，我们还提出了一种简化的设定，保留了基于效用的公平性评估的核心要素。我们通过两个应用来阐述该框架：大学录取和信用风险评估。在这两个案例中，概率度量可能将一个决策过程归类为近似公平，即使其对应的效用结果实际上极为不公平。

    arXiv:2405.09360v3 Announce Type: replace  Abstract: Fairness in decision-making processes is often quantified using probabilistic metrics. However, these metrics need not reflect the consequences of decisions for the affected individuals and groups. We develop a utility-based framework that incorporates these consequences into the assessment of fairness. Our main result shows that a decision-making process can satisfy $\varepsilon$-fairness while nevertheless being maximally unfair once the utilities associated with its outcomes are taken into account. To address applications in which information on false negatives is unavailable, we also formulate a reduced setting that retains the essential elements of the utility-based fairness assessment. We illustrate the framework through two applications: college admissions and credit-risk assessment. In both cases, probabilistic metrics may classify a decision-making process as approximately fair even though the corresponding utility outcomes 
    
[^46]: 高维情况下多个均值向量的估计

    Estimation of multiple mean vectors in high dimension

    [https://arxiv.org/abs/2403.15038](https://arxiv.org/abs/2403.15038)

    通过凸组合的方法估计高维空间中不同概率分布的多维均值，引入了两种权重确定策略：一种通过测试程序识别低方差的相邻均值，提出了封闭形式插补公式；另一种通过最小化二次风险的上置信界确定权重，通过理论分析得出方法对经验均值的二次风险改进，在维度渐近的角度上渐近地接近 Oracle（Minimax）改进。

    

    我们致力于基于独立样本在一个共同空间中估计来自不同概率分布的多维均值。我们的方法是通过对这些样本导出的经验均值进行凸组合来形成估计量。我们引入了两种策略来找到适当的依赖于数据的凸组合权重：第一种利用测试程序来识别具有低方差的相邻均值，从而产生了一个关于权重的封闭形式插补公式；第二种通过最小化二次风险的上置信区间来确定权重。通过理论分析，我们评估了我们的方法相对于经验均值提供的二次风险改进。我们的分析集中在维度渐近的角度上，显示我们的方法在数据的有效维度增加时渐近地接近于一个 Oracle（Minimax）改进。我们展示了通过提出的方法在均值估计中的应用。

    arXiv:2403.15038v1 Announce Type: cross  Abstract: We endeavour to estimate numerous multi-dimensional means of various probability distributions on a common space based on independent samples. Our approach involves forming estimators through convex combinations of empirical means derived from these samples. We introduce two strategies to find appropriate data-dependent convex combination weights: a first one employing a testing procedure to identify neighbouring means with low variance, which results in a closed-form plug-in formula for the weights, and a second one determining weights via minimization of an upper confidence bound on the quadratic risk.Through theoretical analysis, we evaluate the improvement in quadratic risk offered by our methods compared to the empirical means. Our analysis focuses on a dimensional asymptotics perspective, showing that our methods asymptotically approach an oracle (minimax) improvement as the effective dimension of the data increases.We demonstrat
    

