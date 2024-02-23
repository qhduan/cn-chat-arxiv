# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rao-Blackwellising Bayesian Causal Inference](https://arxiv.org/abs/2402.14781) | 本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。 |
| [^2] | [Causal Imputation for Counterfactual SCMs: Bridging Graphs and Latent Factor Models](https://arxiv.org/abs/2402.14777) | 介绍了一种新颖的基于SCM的模型类，用于因果插补任务，将结果表示为反事实，操作表示为对工具变量进行干预，环境基于初始定义。 |
| [^3] | [Batch and match: black-box variational inference with a score-based divergence](https://arxiv.org/abs/2402.14758) | BaM是一种基于分数的离散的BBVI替代方法，针对高方差梯度估计慢收敛问题，能够在高斯变分族中通过封闭形式的近端更新进行优化，在目标分布为高斯时，批处理大小趋于无穷时变分参数更新将指数快速收敛到目标均值和协方差，BaM在多种生成模型推断中表现出良好性能 |
| [^4] | [How Transformers Learn Causal Structure with Gradient Descent](https://arxiv.org/abs/2402.14735) | Transformers通过梯度下降学习因果结构的过程中，关键的证据是注意力矩阵的梯度编码了token之间的互信息 |
| [^5] | [Incorporating Expert Rules into Neural Networks in the Framework of Concept-Based Learning](https://arxiv.org/abs/2402.14726) | 本文提出了将专家规则融入神经网络的方法，通过形成约束和使用凸多面体来保证输出概率不违反专家规则，实现了归纳与演绎学习的结合。 |
| [^6] | [On the Curses of Future and History in Future-dependent Value Functions for Off-policy Evaluation](https://arxiv.org/abs/2402.14703) | 本文提出了针对POMDP结构的新颖覆盖假设，以解决未来依赖价值函数方法中的长度指数增长问题。 |
| [^7] | [Adaptive time series forecasting with markovian variance switching](https://arxiv.org/abs/2402.14684) | 本论文提出了一种基于马尔可夫方差切换的自适应时间序列预测方法，通过在线学习理论和专家聚合方法来学习方差，相比于传统方法在电量负荷预测问题中表现更优。 |
| [^8] | [Bayesian Off-Policy Evaluation and Learning for Large Action Spaces](https://arxiv.org/abs/2402.14664) | 该论文提出了一个统一的贝叶斯框架，通过结构化和信息丰富的先验捕捉动作之间的相关性，提出了一个适用于离策略评估和学习的通用贝叶斯方法sDM，并引入了能评估算法在多问题实例中平均表现的贝叶斯指标，分析了sDM在OPE和OPL中利用动作相关性的优势，并展示了其强大性能 |
| [^9] | [CoLoRA: Continuous low-rank adaptation for reduced implicit neural modeling of parameterized partial differential equations](https://arxiv.org/abs/2402.14646) | CoLoRA通过连续低秩自适应提供了一种快速预测参数化偏微分方程解演变的简化神经网络建模方法 |
| [^10] | [Sparse Linear Regression and Lattice Problems](https://arxiv.org/abs/2402.14645) | 本文提供了关于稀疏线性回归在所有高效算法的平均情况困难性的证据，假设格问题的最坏情况困难性。 |
| [^11] | [latrend: A Framework for Clustering Longitudinal Data](https://arxiv.org/abs/2402.14621) | latrend框架为纵向数据聚类提供了统一的方法应用框架，方便研究人员比较不同方法，实现快速原型设计。 |
| [^12] | [Bandits with Abstention under Expert Advice](https://arxiv.org/abs/2402.14585) | 我们提出了CBA算法，其利用放弃参与游戏的假设获得了可以显著改进经典Exp4算法的奖励界限，成为首个对一般置信评级预测器的预期累积奖励实现界限的研究者，并在专家案例中实现了一种新颖的奖励界限。 |
| [^13] | [Multivariate Online Linear Regression for Hierarchical Forecasting](https://arxiv.org/abs/2402.14578) | 提出了MultiVAW方法，将Vovk-Azoury-Warmuth算法扩展到多元设置，同时应用于在线层次预测问题，并且能够放宽传统分析所做的假设 |
| [^14] | [A Framework for Variational Inference of Lightweight Bayesian Neural Networks with Heteroscedastic Uncertainties](https://arxiv.org/abs/2402.14532) | 提出了一种新框架，通过将异方差Aleatoric和认知方差嵌入到学习BNN参数的方差中，改善了轻量级网络的预测性能。 |
| [^15] | [Spectral invariance and maximality properties of the frequency spectrum of quantum neural networks](https://arxiv.org/abs/2402.14515) | 量子神经网络研究了频谱的极大性质，证明了在一类模型中存在极大结果，以及在一些条件下存在保持频谱的光谱不变性，解释了文献中观察到的结果对称性。 |
| [^16] | [Imbalanced Data Clustering using Equilibrium K-Means](https://arxiv.org/abs/2402.14490) | Equilibrium K-Means（EKM）是一种新颖且简单的K均值类型算法，通过减少聚类中心在大类簇中心聚集的倾向，显著改善了不平衡数据的聚类结果。 |
| [^17] | [Reimagining Anomalies: What If Anomalies Were Normal?](https://arxiv.org/abs/2402.14469) | 方法提出了一种新颖的解释方法，生成多个反事实示例以捕获异常的多样概念，为用户提供对触发异常检测器机制的高级语义解释，允许探索“假设情景”。 |
| [^18] | [The Universe as a Learning System](https://arxiv.org/abs/2402.14423) | 量子系统在一般要求下遵循一种扰乱版本的梯度下降模型，学习过程受到量子系统自组织的影响。 |
| [^19] | [Global Safe Sequential Learning via Efficient Knowledge Transfer](https://arxiv.org/abs/2402.14402) | 提出了考虑转移安全的全局顺序学习方法，以加速安全学习，并通过预先计算源组件来减少额外的计算负载。 |
| [^20] | [WindDragon: Enhancing wind power forecasting with Automated Deep Learning](https://arxiv.org/abs/2402.14385) | 利用自动深度学习结合数值天气预报风速图，WindDragon系统在全国范围内实现了短期风力预测，为电网运营和系统平衡提供关键支持。 |
| [^21] | [HyperFast: Instant Classification for Tabular Data](https://arxiv.org/abs/2402.14335) | HyperFast是一个针对表格数据的即时分类方法，通过在单次前向传递中生成特定任务的神经网络，避免了需进行模型训练的必要性，并在实验中展现出高度竞争力。 |
| [^22] | [From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection](https://arxiv.org/abs/2402.14332) | 通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。 |
| [^23] | [Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation](https://arxiv.org/abs/2402.14264) | 采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性 |
| [^24] | [A hierarchical decomposition for explaining ML performance discrepancies](https://arxiv.org/abs/2402.14254) | 提出了一种详细的变量级分解方法，可以量化每个变量对性能差异的影响，为实现有针对性干预措施提供更深入的理解 |
| [^25] | [Estimating Unknown Population Sizes Using the Hypergeometric Distribution](https://arxiv.org/abs/2402.14220) | 提出了一种使用超几何似然解决估计离散分布挑战的新方法，即使存在严重的欠采样，也能实现，且在人口规模估计的准确性和学习能力方面优于其他方法。 |
| [^26] | [Multiply Robust Estimation for Local Distribution Shifts with Multiple Domains](https://arxiv.org/abs/2402.14145) | 提出了一种两阶段的乘幂稳健估计方法，用于改善表格数据分析中每个个体部分的模型性能，并建立了在测试风险上的理论保证。 |
| [^27] | [Computational-Statistical Gaps for Improper Learning in Sparse Linear Regression](https://arxiv.org/abs/2402.14103) | 该研究探讨了稀疏线性回归中的计算统计差距问题，为了高效地找到可以在样本上实现非平凡预测误差的潜在密集估计的回归向量，需要至少 $\Omega(k \log (d/k))$ 个样本。 |
| [^28] | [Social Environment Design](https://arxiv.org/abs/2402.14090) | 该论文提出了一种新的研究议程，介绍了社会环境设计作为一种用于自动化政策制定的AI通用框架，旨在捕捉一般经济环境，通过AI模拟系统分析政府和经济政策，并强调未来基于AI的政策制定研究中的关键挑战。 |
| [^29] | [Robust Learning of Noisy Time Series Collections Using Stochastic Process Models with Motion Codes](https://arxiv.org/abs/2402.14081) | 使用具有学习谱核的混合高斯过程的潜变量模型方法，针对嘈杂时间序列数据进行鲁棒学习。 |
| [^30] | [Efficient Normalized Conformal Prediction and Uncertainty Quantification for Anti-Cancer Drug Sensitivity Prediction with Deep Regression Forests](https://arxiv.org/abs/2402.14080) | 通过深度回归森林计算样本方差，提高了抗癌药物敏感性预测中的规范化置信预测效率和覆盖率 |
| [^31] | [Probability Tools for Sequential Random Projection](https://arxiv.org/abs/2402.14026) | 该论文提出了适用于顺序随机投影的概率框架，通过构建停止过程并采用混合方法，实现了对一系列相互连接的浓缩事件的分析，从而创造了对Johnson-Lindenstrauss引理的非平凡鞅扩展。 |
| [^32] | [Revisiting Convergence of AdaGrad with Relaxed Assumptions](https://arxiv.org/abs/2402.13794) | 重新审视了AdaGrad在非凸光滑优化问题上的收敛性，提出了通用噪声模型，得出了概率收敛速度，无需先验知识，且可以在噪声参数足够小时加速至更快的速度。 |
| [^33] | [A Method For Bounding Tail Probabilities](https://arxiv.org/abs/2402.13662) | 提出了一种界定连续随机变量右尾和左尾概率上下界的方法，通过设置特定的函数，得到了新的上下界限，并与马尔可夫不等式建立了联系 |
| [^34] | [Learning under Singularity: An Information Criterion improving WBIC and sBIC](https://arxiv.org/abs/2402.12762) | LS信息准则旨在增强WBIC和sBIC的功能，有效处理非正则情况，具有稳定性，为奇异情况下的信息准则提供了新的方法 |
| [^35] | [BlackJAX: Composable Bayesian inference in JAX](https://arxiv.org/abs/2402.10797) | BlackJAX是一个实现在JAX中组合式贝叶斯推断的库，采用函数式方法提高易用性、速度和模块化，适用于需要尖端方法、研究人员和想要了解工作原理的人。 |
| [^36] | [EduGym: An Environment and Notebook Suite for Reinforcement Learning Education](https://arxiv.org/abs/2311.10590) | EduGym是一套用于强化学习教育的环境和笔记本套件，旨在解决学生在转换理论和实践中遇到的困难。 |
| [^37] | [Uncertainty Quantification of Spatiotemporal Travel Demand with Probabilistic Graph Neural Networks](https://arxiv.org/abs/2303.04040) | 该研究提出了一种概率图神经网络（Prob-GNN）框架，用于量化出行需求的时空不确定性，实证应用表明概率假设对不确定性预测影响大于确定性假设。 |
| [^38] | [Promises and Pitfalls of Threshold-based Auto-labeling](https://arxiv.org/abs/2211.12620) | TBAL系统可以通过验证数据自动标注未标注数据，减少手动标注的依赖；研究结果展示了即使模型表现不佳也可以准确自动标记数据，并揭示了TBAL系统的潜在缺陷 |
| [^39] | [Controlling Multiple Errors Simultaneously with a PAC-Bayes Bound](https://arxiv.org/abs/2202.05560) | 该研究提出了一种PAC-Bayes界限，能够同时控制多个错误，并提供丰富的信息，适用于回归中测试损失分布或分类中不同错误分类的概率。 |
| [^40] | [FIGARO: Generating Symbolic Music with Fine-Grained Artistic Control](https://arxiv.org/abs/2201.10936) | 提出了一种自监督的描述-序列任务，实现了在全局水平上对生成音乐的细粒度可控，通过结合高级特征和领域知识，在符号音乐生成方面取得了最新的成果 |
| [^41] | [Adversarial Machine Learning: Bayesian Perspectives](https://arxiv.org/abs/2003.03546) | AML旨在保护机器学习系统免受安全威胁，贝叶斯视角为防御提供了新的好处 |
| [^42] | [On Feynman--Kac training of partial Bayesian neural networks.](http://arxiv.org/abs/2310.19608) | 本文提出了一种将部分贝叶斯神经网络训练转化为模拟费曼-卡克模型的高效采样训练策略，并通过各种数据集的实验证明其在预测性能方面优于现有技术。 |
| [^43] | [Flow-based Distributionally Robust Optimization.](http://arxiv.org/abs/2310.19253) | 这项研究提出了一种称为FlowDRO的计算高效框架，用于解决基于流的分布鲁棒优化问题，通过使用流模型和Wasserstein近端梯度流类型的算法，实现了对具有更大样本大小的问题的可扩展性和更好的泛化能力。 |
| [^44] | [Externally Valid Policy Evaluation Combining Trial and Observational Data.](http://arxiv.org/abs/2310.14763) | 这项研究提出了一种结合试验和观察数据的外部有效策略评估方法，利用试验数据对目标人群上的政策结果进行有效推断，并给出了可验证的评估结果。 |
| [^45] | [Improving Adaptive Online Learning Using Refined Discretization.](http://arxiv.org/abs/2309.16044) | 通过一种新颖的连续时间启发式算法，提高了自适应在线学习的效果，将梯度方差的依赖性从次优的$O(\sqrt{V_T\log V_T})$改进到最优速率$O(\sqrt{V_T})$，并可适用于未知Lipschitz常数的情况。 |
| [^46] | [Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?.](http://arxiv.org/abs/2307.14642) | 本文证明了带有控制变量的黑盒变分推断在完美变分族规范下以几何速度收敛，为BBVI提供了收敛性保证，同时提出了对熵梯度估计器的改进，对比了STL估计器，并给出了明确的非渐近复杂度保证。 |
| [^47] | [Bootstrap aggregation and confidence measures to improve time series causal discovery.](http://arxiv.org/abs/2306.08946) | 本文介绍了一种新的自助聚合和置信度度量方法，使得时间序列因果发现能够提供连接的置信度度量。在广泛的数值实验中，实验证明该方法提高了因果发现的性能。 |
| [^48] | [Domain-Agnostic Batch Bayesian Optimization with Diverse Constraints via Bayesian Quadrature.](http://arxiv.org/abs/2306.05843) | 本论文提出了cSOBER，一种处理多样化约束条件、离散和混合空间、未知约束以及查询拒绝问题的领域无关型贝叶斯优化算法。 |
| [^49] | [Meaningful Causal Aggregation and Paradoxical Confounding.](http://arxiv.org/abs/2304.11625) | 聚合变量上的因果性不确定性可能会使得原本不混淆的因果关系变得混淆，在实际应用中，我们需要接受宏观因果关系通常只与微观状态相关的事实。 |
| [^50] | [Imprecise Bayesian Neural Networks.](http://arxiv.org/abs/2302.09656) | 在机器学习和人工智能领域，该论文提出了一种新的算法——不精确的贝叶斯神经网络(IBNNs)。这种算法使用可信区间先验分布集合和似然分布集合进行训练，相比标准的BNNs，可以区分先验和后验的不确定性并量化。此外，IBNNs在贝叶斯灵敏度分析方面具有更强的鲁棒性，并且对分布变化也更加鲁棒。 |

# 详细

[^1]: Rao-Blackwellising Bayesian Causal Inference

    Rao-Blackwellising Bayesian Causal Inference

    [https://arxiv.org/abs/2402.14781](https://arxiv.org/abs/2402.14781)

    本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。

    

    贝叶斯因果推断，即推断用于下游因果推理任务中的因果模型的后验概率，构成了一个在文献中鲜有探讨的难解的计算推断问题。本文将基于顺序的MCMC结构学习技术与最近梯度图学习的进展相结合，构建了一个有效的贝叶斯因果推断框架。具体而言，我们将推断因果结构的问题分解为(i)推断变量之间的拓扑顺序以及(ii)推断每个变量的父节点集合。当限制每个变量的父节点数量时，我们可以在多项式时间内完全边缘化父节点集合。我们进一步使用高斯过程来建模未知的因果机制，从而允许其精确边缘化。这引入了一个Rao-Blackwell化方案，其中除了因果顺序之外，模型中的所有组件都被消除。

    arXiv:2402.14781v1 Announce Type: cross  Abstract: Bayesian causal inference, i.e., inferring a posterior over causal models for the use in downstream causal reasoning tasks, poses a hard computational inference problem that is little explored in literature. In this work, we combine techniques from order-based MCMC structure learning with recent advances in gradient-based graph learning into an effective Bayesian causal inference framework. Specifically, we decompose the problem of inferring the causal structure into (i) inferring a topological order over variables and (ii) inferring the parent sets for each variable. When limiting the number of parents per variable, we can exactly marginalise over the parent sets in polynomial time. We further use Gaussian processes to model the unknown causal mechanisms, which also allows their exact marginalisation. This introduces a Rao-Blackwellization scheme, where all components are eliminated from the model, except for the causal order, for whi
    
[^2]: 因果插补用于反事实结构方程模型：桥接图和潜在因子模型

    Causal Imputation for Counterfactual SCMs: Bridging Graphs and Latent Factor Models

    [https://arxiv.org/abs/2402.14777](https://arxiv.org/abs/2402.14777)

    介绍了一种新颖的基于SCM的模型类，用于因果插补任务，将结果表示为反事实，操作表示为对工具变量进行干预，环境基于初始定义。

    

    我们考虑因果插补任务，旨在预测一系列操作在各种可能环境下的结果。以预测不同药物如何影响不同细胞类型为运行示例。我们研究指标唯一的设置，其中操作和环境是具有有限可能值的分类变量。即使在这种简单设置中，由于通常只有很少可能的操作-环境对得到研究，因此存在实际挑战。模型必须对新颖的操作-环境对进行外推，这可以被构造为行由操作索引、列由环境索引、矩阵条目对应结果的一种矩阵补全形式。我们引入了一种新颖的基于SCM的模型类，其中结果表示为反事实，操作表示为对工具变量进行干预，环境基于初始定义。

    arXiv:2402.14777v1 Announce Type: cross  Abstract: We consider the task of causal imputation, where we aim to predict the outcomes of some set of actions across a wide range of possible contexts. As a running example, we consider predicting how different drugs affect cells from different cell types. We study the index-only setting, where the actions and contexts are categorical variables with a finite number of possible values. Even in this simple setting, a practical challenge arises, since often only a small subset of possible action-context pairs have been studied. Thus, models must extrapolate to novel action-context pairs, which can be framed as a form of matrix completion with rows indexed by actions, columns indexed by contexts, and matrix entries corresponding to outcomes. We introduce a novel SCM-based model class, where the outcome is expressed as a counterfactual, actions are expressed as interventions on an instrumental variable, and contexts are defined based on the initia
    
[^3]: 批处理和匹配：基于分数的离散的黑匣子变分推断

    Batch and match: black-box variational inference with a score-based divergence

    [https://arxiv.org/abs/2402.14758](https://arxiv.org/abs/2402.14758)

    BaM是一种基于分数的离散的BBVI替代方法，针对高方差梯度估计慢收敛问题，能够在高斯变分族中通过封闭形式的近端更新进行优化，在目标分布为高斯时，批处理大小趋于无穷时变分参数更新将指数快速收敛到目标均值和协方差，BaM在多种生成模型推断中表现出良好性能

    

    大多数主要的黑匣子变分推断（BBVI）实现都是基于优化随机证据下界（ELBO）。但是，这种BBVI方法通常由于其梯度估计的高方差而收敛缓慢。在本文中，我们提出了批处理和匹配（BaM），这是一种基于分数的离散的BBVI替代方法。值得注意的是，这种基于分数的离散可以通过对具有全协方差矩阵的高斯变分族使用封闭形式的近端更新进行优化。我们分析了当目标分布为高斯分布时BaM的收敛性，并证明在批量大小趋于无穷时变分参数更新会指数收敛到目标均值和协方差。我们还评估了BaM在源自层次和深度生成模型后验推断的高斯和非高斯目标分布上的性能。在这些实验中，我们发现BaM在...

    arXiv:2402.14758v1 Announce Type: cross  Abstract: Most leading implementations of black-box variational inference (BBVI) are based on optimizing a stochastic evidence lower bound (ELBO). But such approaches to BBVI often converge slowly due to the high variance of their gradient estimates. In this work, we propose batch and match (BaM), an alternative approach to BBVI based on a score-based divergence. Notably, this score-based divergence can be optimized by a closed-form proximal update for Gaussian variational families with full covariance matrices. We analyze the convergence of BaM when the target distribution is Gaussian, and we prove that in the limit of infinite batch size the variational parameter updates converge exponentially quickly to the target mean and covariance. We also evaluate the performance of BaM on Gaussian and non-Gaussian target distributions that arise from posterior inference in hierarchical and deep generative models. In these experiments, we find that BaM ty
    
[^4]: Transformers如何通过梯度下降学习因果结构

    How Transformers Learn Causal Structure with Gradient Descent

    [https://arxiv.org/abs/2402.14735](https://arxiv.org/abs/2402.14735)

    Transformers通过梯度下降学习因果结构的过程中，关键的证据是注意力矩阵的梯度编码了token之间的互信息

    

    Transformers在序列建模任务上取得了令人难以置信的成功，这在很大程度上归功于自注意机制，它允许信息在序列的不同部分之间传递。自注意机制使得transformers能够编码因果结构，从而使其特别适合序列建模。然而，transformers通过梯度训练算法学习这种因果结构的过程仍然不太清楚。为了更好地理解这个过程，我们引入了一个需要学习潜在因果结构的上下文学习任务。我们证明了简化的两层transformer上的梯度下降可以学会解决这个任务，通过在第一层注意力中编码潜在因果图来完成。我们证明的关键洞察是注意力矩阵的梯度编码了token之间的互信息。由于数据处理不等式的结果，注意力矩阵中最大的条目...

    arXiv:2402.14735v1 Announce Type: new  Abstract: The incredible success of transformers on sequence modeling tasks can be largely attributed to the self-attention mechanism, which allows information to be transferred between different parts of a sequence. Self-attention allows transformers to encode causal structure which makes them particularly suitable for sequence modeling. However, the process by which transformers learn such causal structure via gradient-based training algorithms remains poorly understood. To better understand this process, we introduce an in-context learning task that requires learning latent causal structure. We prove that gradient descent on a simplified two-layer transformer learns to solve this task by encoding the latent causal graph in the first attention layer. The key insight of our proof is that the gradient of the attention matrix encodes the mutual information between tokens. As a consequence of the data processing inequality, the largest entries of th
    
[^5]: 在概念学习框架中将专家规则融入神经网络

    Incorporating Expert Rules into Neural Networks in the Framework of Concept-Based Learning

    [https://arxiv.org/abs/2402.14726](https://arxiv.org/abs/2402.14726)

    本文提出了将专家规则融入神经网络的方法，通过形成约束和使用凸多面体来保证输出概率不违反专家规则，实现了归纳与演绎学习的结合。

    

    本文阐述了将专家规则融入机器学习模型中以扩展基于概念学习的问题。提出了如何将逻辑规则和预测概念概率的神经网络相结合。该组合背后的第一个想法是形成约束，以满足专家规则的所有概念值组合的联合概率分布。第二个想法是以凸多面体的形式表示概率分布的可行集，并使用其顶点或面。

    arXiv:2402.14726v1 Announce Type: cross  Abstract: A problem of incorporating the expert rules into machine learning models for extending the concept-based learning is formulated in the paper. It is proposed how to combine logical rules and neural networks predicting the concept probabilities. The first idea behind the combination is to form constraints for a joint probability distribution over all combinations of concept values to satisfy the expert rules. The second idea is to represent a feasible set of probability distributions in the form of a convex polytope and to use its vertices or faces. We provide several approaches for solving the stated problem and for training neural networks which guarantee that the output probabilities of concepts would not violate the expert rules. The solution of the problem can be viewed as a way for combining the inductive and deductive learning. Expert rules are used in a broader sense when any logical function that connects concepts and class labe
    
[^6]: 在未来依赖价值函数中探讨未来和历史的诅咒在离线评估中的应用

    On the Curses of Future and History in Future-dependent Value Functions for Off-policy Evaluation

    [https://arxiv.org/abs/2402.14703](https://arxiv.org/abs/2402.14703)

    本文提出了针对POMDP结构的新颖覆盖假设，以解决未来依赖价值函数方法中的长度指数增长问题。

    

    我们研究了在部分可观测环境中复杂观测的离线评估(OPE)，旨在开发能够避免对时间跨度指数依赖的估计器。最近，Uehara等人（2022年）提出了未来依赖价值函数作为解决这一问题的一个有前途的框架。然而，该框架也取决于未来依赖价值函数的有界性以及其他相关数量，我们发现这些数量可能会随着长度呈指数增长，从而抹去该方法的优势。在本文中，我们发现了针对POMDP结构的新颖覆盖假设。

    arXiv:2402.14703v1 Announce Type: cross  Abstract: We study off-policy evaluation (OPE) in partially observable environments with complex observations, with the goal of developing estimators whose guarantee avoids exponential dependence on the horizon. While such estimators exist for MDPs and POMDPs can be converted to history-based MDPs, their estimation errors depend on the state-density ratio for MDPs which becomes history ratios after conversion, an exponential object. Recently, Uehara et al. (2022) proposed future-dependent value functions as a promising framework to address this issue, where the guarantee for memoryless policies depends on the density ratio over the latent state space. However, it also depends on the boundedness of the future-dependent value function and other related quantities, which we show could be exponential-in-length and thus erasing the advantage of the method. In this paper, we discover novel coverage assumptions tailored to the structure of POMDPs, such
    
[^7]: 具有马尔可夫方差切换的自适应时间序列预测

    Adaptive time series forecasting with markovian variance switching

    [https://arxiv.org/abs/2402.14684](https://arxiv.org/abs/2402.14684)

    本论文提出了一种基于马尔可夫方差切换的自适应时间序列预测方法，通过在线学习理论和专家聚合方法来学习方差，相比于传统方法在电量负荷预测问题中表现更优。

    

    自适应时间序列预测对于在制度变化下进行预测是至关重要的。许多传统方法假设具有在时间上恒定的方差的线性高斯状态空间模型（LGSSM）。然而，许多现实世界的过程不能被这样的模型捕捉。我们考虑具有马尔可夫切换方差的状态空间模型。这样的动态系统通常是无法解决的，因为它们的计算复杂性随时间呈指数增长；变分贝叶斯（VB）技术已被应用于解决此问题。在本文中，我们提出了一种基于在线学习理论的新的估计方差的方法；我们调整专家聚合方法来随时间学习方差。我们将提出的方法应用于合成数据以及用于电量负荷预测的问题。我们展示了这种方法对于误差估计的稳健性，并优于传统的专家聚合方法。

    arXiv:2402.14684v1 Announce Type: cross  Abstract: Adaptive time series forecasting is essential for prediction under regime changes. Several classical methods assume linear Gaussian state space model (LGSSM) with variances constant in time. However, there are many real-world processes that cannot be captured by such models. We consider a state-space model with Markov switching variances. Such dynamical systems are usually intractable because of their computational complexity increasing exponentially with time; Variational Bayes (VB) techniques have been applied to this problem. In this paper, we propose a new way of estimating variances based on online learning theory; we adapt expert aggregation methods to learn the variances over time. We apply the proposed method to synthetic data and to the problem of electricity load forecasting. We show that this method is robust to misspecification and outperforms traditional expert aggregation.
    
[^8]: 大动作空间的贝叶斯离策略评估与学习

    Bayesian Off-Policy Evaluation and Learning for Large Action Spaces

    [https://arxiv.org/abs/2402.14664](https://arxiv.org/abs/2402.14664)

    该论文提出了一个统一的贝叶斯框架，通过结构化和信息丰富的先验捕捉动作之间的相关性，提出了一个适用于离策略评估和学习的通用贝叶斯方法sDM，并引入了能评估算法在多问题实例中平均表现的贝叶斯指标，分析了sDM在OPE和OPL中利用动作相关性的优势，并展示了其强大性能

    

    在交互式系统中，动作经常是相关的，这为大动作空间中更有效的离策略评估（OPE）和学习（OPL）提供了机会。我们引入了一个统一的贝叶斯框架，通过结构化和信息丰富的先验来捕捉这些相关性。在该框架中，我们提出了sDM，一个为OPE和OPL设计的通用贝叶斯方法，既有算法基础又有理论基础。值得注意的是，sDM利用动作相关性而不会影响计算效率。此外，受在线贝叶斯赌博机启发，我们引入了评估算法在多个问题实例中平均性能的贝叶斯指标，偏离传统的最坏情况评估。我们分析了sDM在OPE和OPL中的表现，凸显了利用动作相关性的好处。实证证据展示了sDM的强大性能。

    arXiv:2402.14664v1 Announce Type: cross  Abstract: In interactive systems, actions are often correlated, presenting an opportunity for more sample-efficient off-policy evaluation (OPE) and learning (OPL) in large action spaces. We introduce a unified Bayesian framework to capture these correlations through structured and informative priors. In this framework, we propose sDM, a generic Bayesian approach designed for OPE and OPL, grounded in both algorithmic and theoretical foundations. Notably, sDM leverages action correlations without compromising computational efficiency. Moreover, inspired by online Bayesian bandits, we introduce Bayesian metrics that assess the average performance of algorithms across multiple problem instances, deviating from the conventional worst-case assessments. We analyze sDM in OPE and OPL, highlighting the benefits of leveraging action correlations. Empirical evidence showcases the strong performance of sDM.
    
[^9]: CoLoRA:用于参数化偏微分方程简化隐式神经建模的连续低秩自适应

    CoLoRA: Continuous low-rank adaptation for reduced implicit neural modeling of parameterized partial differential equations

    [https://arxiv.org/abs/2402.14646](https://arxiv.org/abs/2402.14646)

    CoLoRA通过连续低秩自适应提供了一种快速预测参数化偏微分方程解演变的简化神经网络建模方法

    

    该工作介绍了一种基于连续低秩自适应（CoLoRA）的简化模型，它预先训练神经网络适用于给定的偏微分方程，然后在时间上连续地调整低秩权重，以快速预测新物理参数和新初始条件下解场的演变。自适应可以是纯粹数据驱动的，也可以通过一个方程驱动的变分方法，提供Galerkin最优的逼近。由于CoLoRA在时间上局部逼近解场，权重的秩可以保持较小，这意味着只需要离线训练几条轨迹，因此CoLoRA非常适用于数据稀缺的情况。与传统方法相比，CoLoRA的预测速度快上几个数量级，其准确度和参数效率也比其他神经网络方法更高。

    arXiv:2402.14646v1 Announce Type: new  Abstract: This work introduces reduced models based on Continuous Low Rank Adaptation (CoLoRA) that pre-train neural networks for a given partial differential equation and then continuously adapt low-rank weights in time to rapidly predict the evolution of solution fields at new physics parameters and new initial conditions. The adaptation can be either purely data-driven or via an equation-driven variational approach that provides Galerkin-optimal approximations. Because CoLoRA approximates solution fields locally in time, the rank of the weights can be kept small, which means that only few training trajectories are required offline so that CoLoRA is well suited for data-scarce regimes. Predictions with CoLoRA are orders of magnitude faster than with classical methods and their accuracy and parameter efficiency is higher compared to other neural network approaches.
    
[^10]: 稀疏线性回归和格问题

    Sparse Linear Regression and Lattice Problems

    [https://arxiv.org/abs/2402.14645](https://arxiv.org/abs/2402.14645)

    本文提供了关于稀疏线性回归在所有高效算法的平均情况困难性的证据，假设格问题的最坏情况困难性。

    

    稀疏线性回归（SLR）是统计学中一个研究良好的问题，其中给定设计矩阵 $X\in\mathbb{R}^{m\times n}$ 和响应向量 $y=X\theta^*+w$，其中 $\theta^*$ 是 $k$-稀疏向量（即，$\|\theta^*\|_0\leq k$），$w$ 是小的、任意的噪声，目标是找到一个 $k$-稀疏的 $\widehat{\theta} \in \mathbb{R}^n$，使得均方预测误差 $\frac{1}{m}\|X\widehat{\theta}-X\theta^*\|^2_2$ 最小化。虽然 $\ell_1$-松弛方法如基 Pursuit、Lasso 和 Dantzig 选择器在设计矩阵条件良好时解决了 SLR，但没有已知通用算法，也没有任何关于在所有高效算法的平均情况设置中的困难性的正式证据。

    arXiv:2402.14645v1 Announce Type: new  Abstract: Sparse linear regression (SLR) is a well-studied problem in statistics where one is given a design matrix $X\in\mathbb{R}^{m\times n}$ and a response vector $y=X\theta^*+w$ for a $k$-sparse vector $\theta^*$ (that is, $\|\theta^*\|_0\leq k$) and small, arbitrary noise $w$, and the goal is to find a $k$-sparse $\widehat{\theta} \in \mathbb{R}^n$ that minimizes the mean squared prediction error $\frac{1}{m}\|X\widehat{\theta}-X\theta^*\|^2_2$. While $\ell_1$-relaxation methods such as basis pursuit, Lasso, and the Dantzig selector solve SLR when the design matrix is well-conditioned, no general algorithm is known, nor is there any formal evidence of hardness in an average-case setting with respect to all efficient algorithms.   We give evidence of average-case hardness of SLR w.r.t. all efficient algorithms assuming the worst-case hardness of lattice problems. Specifically, we give an instance-by-instance reduction from a variant of the bo
    
[^11]: latrend: 用于聚类纵向数据的框架

    latrend: A Framework for Clustering Longitudinal Data

    [https://arxiv.org/abs/2402.14621](https://arxiv.org/abs/2402.14621)

    latrend框架为纵向数据聚类提供了统一的方法应用框架，方便研究人员比较不同方法，实现快速原型设计。

    

    纵向数据的聚类用于探索不同主题随时间变化的共同趋势，以数值测量为兴趣。多年来引入了各种R包，用于识别纵向模式的聚类，以一种或多种趋势来总结主题之间轨迹的变化。我们介绍了R包"latrend"作为纵向聚类方法的统一应用框架，使得可以在最小编码量情况下比较各种方法。该包还作为常用包"dtwclust"、"flexmix"、"kml"、"lcmm"、"mclust"、"mixAK"和"mixtools"的接口，这使得研究人员可以轻松比较不同方法、实现和方法规范。此外，研究人员还可以利用框架提供的标准工具来快速实现新的聚类方法，从而实现快速原型设计。

    arXiv:2402.14621v1 Announce Type: new  Abstract: Clustering of longitudinal data is used to explore common trends among subjects over time for a numeric measurement of interest. Various R packages have been introduced throughout the years for identifying clusters of longitudinal patterns, summarizing the variability in trajectories between subject in terms of one or more trends. We introduce the R package "latrend" as a framework for the unified application of methods for longitudinal clustering, enabling comparisons between methods with minimal coding. The package also serves as an interface to commonly used packages for clustering longitudinal data, including "dtwclust", "flexmix", "kml", "lcmm", "mclust", "mixAK", and "mixtools". This enables researchers to easily compare different approaches, implementations, and method specifications. Furthermore, researchers can build upon the standard tools provided by the framework to quickly implement new cluster methods, enabling rapid protot
    
[^12]: 具有弃权选项的专家建议下的赌徒问题

    Bandits with Abstention under Expert Advice

    [https://arxiv.org/abs/2402.14585](https://arxiv.org/abs/2402.14585)

    我们提出了CBA算法，其利用放弃参与游戏的假设获得了可以显著改进经典Exp4算法的奖励界限，成为首个对一般置信评级预测器的预期累积奖励实现界限的研究者，并在专家案例中实现了一种新颖的奖励界限。

    

    我们研究了在赌徒反馈下利用专家建议进行预测的经典问题。我们的模型假设一种行动，即学习者放弃参与游戏，在每次试验中都没有奖励或损失。我们提出了CBA算法，利用这一假设获得了可以显著改进经典Exp4算法的奖励界限。我们可以将我们的问题视为在学习者有放弃参与游戏选项时对置信评级预测器进行聚合。重要的是，我们是第一个对一般置信评级预测器的预期累积奖励实现界限的研究者。在专家案例中，我们实现了一种新颖的奖励界限，显著改进了之前在专家Exp（将弃权视为另一种行动）的边界。作为一个示例应用，我们讨论了在有限度量空间中学习球的并集。在这个上下文设置中，我们设计了CBA的有效实现，re

    arXiv:2402.14585v1 Announce Type: new  Abstract: We study the classic problem of prediction with expert advice under bandit feedback. Our model assumes that one action, corresponding to the learner's abstention from play, has no reward or loss on every trial. We propose the CBA algorithm, which exploits this assumption to obtain reward bounds that can significantly improve those of the classical Exp4 algorithm. We can view our problem as the aggregation of confidence-rated predictors when the learner has the option of abstention from play. Importantly, we are the first to achieve bounds on the expected cumulative reward for general confidence-rated predictors. In the special case of specialists we achieve a novel reward bound, significantly improving previous bounds of SpecialistExp (treating abstention as another action). As an example application, we discuss learning unions of balls in a finite metric space. In this contextual setting, we devise an efficient implementation of CBA, re
    
[^13]: 用于层次预测的多元在线线性回归

    Multivariate Online Linear Regression for Hierarchical Forecasting

    [https://arxiv.org/abs/2402.14578](https://arxiv.org/abs/2402.14578)

    提出了MultiVAW方法，将Vovk-Azoury-Warmuth算法扩展到多元设置，同时应用于在线层次预测问题，并且能够放宽传统分析所做的假设

    

    在本文中，我们考虑了一种确定性的在线多元线性回归模型，其中允许响应是多元的。为了解决这个问题，我们引入了MultiVAW，一种将著名的Vovk-Azoury-Warmuth算法扩展到多元设置的方法，并表明它在时间上也具有对数遗憾。我们将我们的结果应用于在线层次预测问题，并将这个文献中的一个算法作为一种特殊情况加以恢复，从而放宽了通常用于其分析的假设。

    arXiv:2402.14578v1 Announce Type: cross  Abstract: In this paper, we consider a deterministic online linear regression model where we allow the responses to be multivariate. To address this problem, we introduce MultiVAW, a method that extends the well-known Vovk-Azoury-Warmuth algorithm to the multivariate setting, and show that it also enjoys logarithmic regret in time. We apply our results to the online hierarchical forecasting problem and recover an algorithm from this literature as a special case, allowing us to relax the hypotheses usually made for its analysis.
    
[^14]: 一种用于具有异方差不确定性的轻量级贝叶斯神经网络变分推断的框架

    A Framework for Variational Inference of Lightweight Bayesian Neural Networks with Heteroscedastic Uncertainties

    [https://arxiv.org/abs/2402.14532](https://arxiv.org/abs/2402.14532)

    提出了一种新框架，通过将异方差Aleatoric和认知方差嵌入到学习BNN参数的方差中，改善了轻量级网络的预测性能。

    

    从贝叶斯神经网络（BNN）中获得异方差预测不确定性对许多应用至关重要。通常，除了预测均值外，异方差Aleatoric不确定性作为BNN的输出进行学习，然而这样做可能需要向网络中添加更多可学习参数。在这项工作中，我们展示了异方差Aleatoric和认知方差均可以嵌入到学习BNN参数的方差中，从而提高轻量级网络的预测性能。通过将这种方法与矩传播方法相结合，我们引入了一个适用于轻量级BNNs的无需取样的变分推断相对简单的框架。

    arXiv:2402.14532v1 Announce Type: new  Abstract: Obtaining heteroscedastic predictive uncertainties from a Bayesian Neural Network (BNN) is vital to many applications. Often, heteroscedastic aleatoric uncertainties are learned as outputs of the BNN in addition to the predictive means, however doing so may necessitate adding more learnable parameters to the network. In this work, we demonstrate that both the heteroscedastic aleatoric and epistemic variance can be embedded into the variances of learned BNN parameters, improving predictive performance for lightweight networks. By complementing this approach with a moment propagation approach to inference, we introduce a relatively simple framework for sampling-free variational inference suitable for lightweight BNNs.
    
[^15]: 量子神经网络频谱的光谱不变性和极大性质

    Spectral invariance and maximality properties of the frequency spectrum of quantum neural networks

    [https://arxiv.org/abs/2402.14515](https://arxiv.org/abs/2402.14515)

    量子神经网络研究了频谱的极大性质，证明了在一类模型中存在极大结果，以及在一些条件下存在保持频谱的光谱不变性，解释了文献中观察到的结果对称性。

    

    量子神经网络（QNNs）是量子机器学习领域的热门方法，由于其与变分量子电路的密切联系，使其成为在噪声中间尺度量子（NISQ）设备上进行实际应用的有前途的候选方法。QNN可以表示为有限傅里叶级数，其中频率集被称为频谱。我们分析了这个频谱并证明，对于一大类模型，存在各种极大性结果。此外，我们证明在一些温和条件下，存在一个保持频谱的具有相同面积$A = RL$的模型类之间的双射，其中$R$表示量子比特数量，$L$表示层数，我们因此称之为面积保持变换下的光谱不变性。通过这个，我们解释了文献中经常观察到的在结果中$R$和$L$的对称性，并展示了最大频谱的依赖性

    arXiv:2402.14515v1 Announce Type: cross  Abstract: Quantum Neural Networks (QNNs) are a popular approach in Quantum Machine Learning due to their close connection to Variational Quantum Circuits, making them a promising candidate for practical applications on Noisy Intermediate-Scale Quantum (NISQ) devices. A QNN can be expressed as a finite Fourier series, where the set of frequencies is called the frequency spectrum. We analyse this frequency spectrum and prove, for a large class of models, various maximality results. Furthermore, we prove that under some mild conditions there exists a bijection between classes of models with the same area $A = RL$ that preserves the frequency spectrum, where $R$ denotes the number of qubits and $L$ the number of layers, which we consequently call spectral invariance under area-preserving transformations. With this we explain the symmetry in $R$ and $L$ in the results often observed in the literature and show that the maximal frequency spectrum depen
    
[^16]: 使用Equilibrium K-Means进行不平衡数据聚类

    Imbalanced Data Clustering using Equilibrium K-Means

    [https://arxiv.org/abs/2402.14490](https://arxiv.org/abs/2402.14490)

    Equilibrium K-Means（EKM）是一种新颖且简单的K均值类型算法，通过减少聚类中心在大类簇中心聚集的倾向，显著改善了不平衡数据的聚类结果。

    

    不平衡数据指的是数据点在不同类别之间分布不均衡，这给传统的硬聚类算法和模糊聚类算法（如硬K均值（HKM，或者Lloyd算法）和模糊K均值（FKM，或者Bezdek算法））带来了挑战。本文介绍了一种新颖且简单的K均值类型算法——Equilibrium K-Means（EKM），它在两个步骤之间交替进行，显著改善了不平衡数据的聚类结果，减少了聚类中心向大类簇中心聚集的倾向。我们还提出了对HKM、FKM和EKM的统一视角，表明它们本质上是具有明确关系的牛顿方法的梯度下降算法。EKM具有与FKM相同的时间和空间复杂度，但对其成员定义提供了更清晰的物理意义。我们在两个合成数据集和十个真实数据集上展示了EKM的性能，并将其与各种聚类算法进行了比较。

    arXiv:2402.14490v1 Announce Type: new  Abstract: Imbalanced data, characterized by an unequal distribution of data points across different clusters, poses a challenge for traditional hard and fuzzy clustering algorithms, such as hard K-means (HKM, or Lloyd's algorithm) and fuzzy K-means (FKM, or Bezdek's algorithm). This paper introduces equilibrium K-means (EKM), a novel and simple K-means-type algorithm that alternates between just two steps, yielding significantly improved clustering results for imbalanced data by reducing the tendency of centroids to crowd together in the center of large clusters. We also present a unifying perspective for HKM, FKM, and EKM, showing they are essentially gradient descent algorithms with an explicit relationship to Newton's method. EKM has the same time and space complexity as FKM but offers a clearer physical meaning for its membership definition. We illustrate the performance of EKM on two synthetic and ten real datasets, comparing it to various cl
    
[^17]: 重新构想异常：如果异常是正常的呢？

    Reimagining Anomalies: What If Anomalies Were Normal?

    [https://arxiv.org/abs/2402.14469](https://arxiv.org/abs/2402.14469)

    方法提出了一种新颖的解释方法，生成多个反事实示例以捕获异常的多样概念，为用户提供对触发异常检测器机制的高级语义解释，允许探索“假设情景”。

    

    基于深度学习的方法在图像异常检测方面取得了突破，但其复杂性给理解为何实例被预测为异常带来了相当大的挑战。我们引入了一种新颖的解释方法，为每个异常生成多个反事实示例，捕获异常的多样概念。反事实示例是对异常的修改，被异常检测器视为正常。该方法提供了触发异常检测器机制的高级语义解释，允许用户探索“假设情景”。对不同图像数据集进行的定性和定量分析显示，该方法应用于最先进的异常检测器可以实现对检测器的高质量语义解释。

    arXiv:2402.14469v1 Announce Type: cross  Abstract: Deep learning-based methods have achieved a breakthrough in image anomaly detection, but their complexity introduces a considerable challenge to understanding why an instance is predicted to be anomalous. We introduce a novel explanation method that generates multiple counterfactual examples for each anomaly, capturing diverse concepts of anomalousness. A counterfactual example is a modification of the anomaly that is perceived as normal by the anomaly detector. The method provides a high-level semantic explanation of the mechanism that triggered the anomaly detector, allowing users to explore "what-if scenarios." Qualitative and quantitative analyses across various image datasets show that the method applied to state-of-the-art anomaly detectors can achieve high-quality semantic explanations of detectors.
    
[^18]: 宇宙作为一个学习系统

    The Universe as a Learning System

    [https://arxiv.org/abs/2402.14423](https://arxiv.org/abs/2402.14423)

    量子系统在一般要求下遵循一种扰乱版本的梯度下降模型，学习过程受到量子系统自组织的影响。

    

    在其微观水平上，宇宙遵循量子力学定律。通过关注从量子力学的流体力学表述中跟随的粒子的量子轨迹，我们提出在一般要求下，量子系统遵循一种扰乱版本的梯度下降模型，这是一种基本的机器学习算法，在其中学习由于量子系统的自组织过程而失真。当我们假设耗散即量子系统是开放的时，这样的学习过程才有可能。学习参数是过程的时间增量除以量子粒子的质量，一个摩擦参数确定了量子系统的非线性。然后我们提供了所提出模型的实证演示。

    arXiv:2402.14423v1 Announce Type: cross  Abstract: At its microscopic level, the universe follows the laws of quantum mechanics. Focusing on the quantum trajectories of particles as followed from the hydrodynamical formulation of quantum mechanics, we propose that under general requirements, quantum systems follow a disrupted version of the gradient descent model, a basic machine learning algorithm, where the learning is distorted due to the self-organizing process of the quantum system. Such a learning process is possible only when we assume dissipation, i.e., that the quantum system is open. The learning parameter is the time increment of the process over the mass of the quantum particle, and a friction parameter determines the nonlinearity of the quantum system. We then provide an empirical demonstration of the proposed model.
    
[^19]: 全局安全顺序学习通过高效知识转移

    Global Safe Sequential Learning via Efficient Knowledge Transfer

    [https://arxiv.org/abs/2402.14402](https://arxiv.org/abs/2402.14402)

    提出了考虑转移安全的全局顺序学习方法，以加速安全学习，并通过预先计算源组件来减少额外的计算负载。

    

    arXiv:2402.14402v1 公告类型: 新摘要: 顺序学习方法例如主动学习和贝叶斯优化选择最具信息量的数据来学习一个任务。在许多医学或工程应用中，数据选择受先验未知的安全条件限制。一条有前途的安全学习方法利用高斯过程（GPs）来建模安全概率，并在具有较高安全置信度的区域中进行数据选择。然而，准确的安全建模需要先验知识或消耗数据。此外，安全置信度集中在给定的观测值周围，导致局部探索。由于在安全关键实验中通常存在可转移的源知识，我们提出考虑转移安全顺序学习来加速安全学习。我们进一步考虑先计算源组件，以减少引入源数据带来的额外计算负载。

    arXiv:2402.14402v1 Announce Type: new  Abstract: Sequential learning methods such as active learning and Bayesian optimization select the most informative data to learn about a task. In many medical or engineering applications, the data selection is constrained by a priori unknown safety conditions. A promissing line of safe learning methods utilize Gaussian processes (GPs) to model the safety probability and perform data selection in areas with high safety confidence. However, accurate safety modeling requires prior knowledge or consumes data. In addition, the safety confidence centers around the given observations which leads to local exploration. As transferable source knowledge is often available in safety critical experiments, we propose to consider transfer safe sequential learning to accelerate the learning of safety. We further consider a pre-computation of source components to reduce the additional computational load that is introduced by incorporating source data. In this pap
    
[^20]: 使用自动深度学习改进风力发电预测的WindDragon系统

    WindDragon: Enhancing wind power forecasting with Automated Deep Learning

    [https://arxiv.org/abs/2402.14385](https://arxiv.org/abs/2402.14385)

    利用自动深度学习结合数值天气预报风速图，WindDragon系统在全国范围内实现了短期风力预测，为电网运营和系统平衡提供关键支持。

    

    实现到2050年零碳排放的目标需要将大量风力纳入电网中。这种能源由于其变化性和不确定性对系统运营商构成挑战。因此，准确预测风力发电对于电网运营和系统平衡至关重要。本文提出了一种在全国范围内进行短期（1至6小时）风力预测的创新方法。该方法利用了自动深度学习结合数值天气预报风速图来准确预测风力发电。

    arXiv:2402.14385v1 Announce Type: new  Abstract: Achieving net zero carbon emissions by 2050 requires the integration of increasing amounts of wind power into power grids. This energy source poses a challenge to system operators due to its variability and uncertainty. Therefore, accurate forecasting of wind power is critical for grid operation and system balancing. This paper presents an innovative approach to short-term (1 to 6 hour horizon) windpower forecasting at a national level. The method leverages Automated Deep Learning combined with Numerical Weather Predictions wind speed maps to accurately forecast wind power.
    
[^21]: 超快速：用于表格数据的即时分类

    HyperFast: Instant Classification for Tabular Data

    [https://arxiv.org/abs/2402.14335](https://arxiv.org/abs/2402.14335)

    HyperFast是一个针对表格数据的即时分类方法，通过在单次前向传递中生成特定任务的神经网络，避免了需进行模型训练的必要性，并在实验中展现出高度竞争力。

    

    训练深度学习模型和进行超参数调整可能需要大量计算资源和时间。与此同时，传统的梯度提升算法等机器学习方法仍然是大多数表格数据应用的首选，而神经网络方法要么需要进行大量的超参数调整，要么仅适用于在有限设置下的玩具数据集。本文介绍了HyperFast，一个为在单次前向传递中立即分类表格数据而设计的元训练的超网络。HyperFast生成一个针对未见数据集定制的特定任务神经网络，可直接用于分类推断，无需训练模型。我们使用OpenML和基因组数据进行了大量实验，将HyperFast与竞争性表格数据神经网络、传统ML方法、AutoML系统和提升机器进行了比较。HyperFast展现出极具竞争力的结果。

    arXiv:2402.14335v1 Announce Type: cross  Abstract: Training deep learning models and performing hyperparameter tuning can be computationally demanding and time-consuming. Meanwhile, traditional machine learning methods like gradient-boosting algorithms remain the preferred choice for most tabular data applications, while neural network alternatives require extensive hyperparameter tuning or work only in toy datasets under limited settings. In this paper, we introduce HyperFast, a meta-trained hypernetwork designed for instant classification of tabular data in a single forward pass. HyperFast generates a task-specific neural network tailored to an unseen dataset that can be directly used for classification inference, removing the need for training a model. We report extensive experiments with OpenML and genomic data, comparing HyperFast to competing tabular data neural networks, traditional ML methods, AutoML systems, and boosting machines. HyperFast shows highly competitive results, wh
    
[^22]: 从大规模到小规模数据集：用于聚类算法选择的尺寸泛化

    From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection

    [https://arxiv.org/abs/2402.14332](https://arxiv.org/abs/2402.14332)

    通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。

    

    在聚类算法选择中，我们会得到一个大规模数据集，并要有效地选择要使用的聚类算法。我们在半监督设置下研究了这个问题，其中有一个未知的基准聚类，我们只能通过昂贵的oracle查询来访问。理想情况下，聚类算法的输出将与基本事实结构上接近。我们通过引入一种聚类算法准确性的尺寸泛化概念来解决这个问题。我们确定在哪些条件下我们可以（1）对大规模聚类实例进行子采样，（2）在较小实例上评估一组候选算法，（3）保证在小实例上准确度最高的算法将在原始大实例上拥有最高的准确度。我们为三种经典聚类算法提供了理论尺寸泛化保证：单链接、k-means++和Gonzalez的k中心启发式（一种平滑的变种）。

    arXiv:2402.14332v1 Announce Type: new  Abstract: In clustering algorithm selection, we are given a massive dataset and must efficiently select which clustering algorithm to use. We study this problem in a semi-supervised setting, with an unknown ground-truth clustering that we can only access through expensive oracle queries. Ideally, the clustering algorithm's output will be structurally close to the ground truth. We approach this problem by introducing a notion of size generalization for clustering algorithm accuracy. We identify conditions under which we can (1) subsample the massive clustering instance, (2) evaluate a set of candidate algorithms on the smaller instance, and (3) guarantee that the algorithm with the best accuracy on the small instance will have the best accuracy on the original big instance. We provide theoretical size generalization guarantees for three classic clustering algorithms: single-linkage, k-means++, and (a smoothed variant of) Gonzalez's k-centers heuris
    
[^23]: 双稳健学习在处理效应估计中的结构不可知性最优性

    Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation

    [https://arxiv.org/abs/2402.14264](https://arxiv.org/abs/2402.14264)

    采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性

    

    平均处理效应估计是因果推断中最核心的问题，应用广泛。虽然文献中提出了许多估计策略，最近还纳入了通用的机器学习估计器，但这些方法的统计最优性仍然是一个开放的研究领域。本文采用最近引入的统计下界结构不可知框架，该框架对干扰函数没有结构性质假设，除了访问黑盒估计器以达到小误差；当只愿意考虑使用非参数回归和分类神谕作为黑盒子过程的估计策略时，这一点尤其吸引人。在这个框架内，我们证明了双稳健估计器对于平均处理效应（ATE）和平均处理效应的统计最优性。

    arXiv:2402.14264v1 Announce Type: cross  Abstract: Average treatment effect estimation is the most central problem in causal inference with application to numerous disciplines. While many estimation strategies have been proposed in the literature, recently also incorporating generic machine learning estimators, the statistical optimality of these methods has still remained an open area of investigation. In this paper, we adopt the recently introduced structure-agnostic framework of statistical lower bounds, which poses no structural properties on the nuisance functions other than access to black-box estimators that attain small errors; which is particularly appealing when one is only willing to consider estimation strategies that use non-parametric regression and classification oracles as a black-box sub-process. Within this framework, we prove the statistical optimality of the celebrated and widely used doubly robust estimators for both the Average Treatment Effect (ATE) and the Avera
    
[^24]: 解释机器学习性能差异的分层分解方法

    A hierarchical decomposition for explaining ML performance discrepancies

    [https://arxiv.org/abs/2402.14254](https://arxiv.org/abs/2402.14254)

    提出了一种详细的变量级分解方法，可以量化每个变量对性能差异的影响，为实现有针对性干预措施提供更深入的理解

    

    机器学习（ML）算法在不同领域的性能往往有所不同。了解它们的性能差异的原因对于确定何种干预措施（例如算法或运营）最有效以缩小性能差距至关重要。现有方法侧重于将总性能差异分解为特征分布$p(X)$变化的影响与结果条件分布$p(Y|X)$变化的影响的$\textit{汇总分解}$；然而，这样粗糙的解释只提供了很少的方法来缩小性能差距。$\textit{详细的变量级分解}$可以量化每个变量对汇总分解中每个项的重要性，从而提供更深入的理解，并提出更有针对性的干预措施。然而，现有方法假设有关全因果图的完整知识或进行强参数假设。

    arXiv:2402.14254v1 Announce Type: new  Abstract: Machine learning (ML) algorithms can often differ in performance across domains. Understanding $\textit{why}$ their performance differs is crucial for determining what types of interventions (e.g., algorithmic or operational) are most effective at closing the performance gaps. Existing methods focus on $\textit{aggregate decompositions}$ of the total performance gap into the impact of a shift in the distribution of features $p(X)$ versus the impact of a shift in the conditional distribution of the outcome $p(Y|X)$; however, such coarse explanations offer only a few options for how one can close the performance gap. $\textit{Detailed variable-level decompositions}$ that quantify the importance of each variable to each term in the aggregate decomposition can provide a much deeper understanding and suggest much more targeted interventions. However, existing methods assume knowledge of the full causal graph or make strong parametric assumpti
    
[^25]: 使用超几何分布估计未知人口规模

    Estimating Unknown Population Sizes Using the Hypergeometric Distribution

    [https://arxiv.org/abs/2402.14220](https://arxiv.org/abs/2402.14220)

    提出了一种使用超几何似然解决估计离散分布挑战的新方法，即使存在严重的欠采样，也能实现，且在人口规模估计的准确性和学习能力方面优于其他方法。

    

    多元超几何分布描述从划分为多个类别的离散元素总体中进行无放回抽样。在文献中存在的一个空白中，我们解决了估计离散分布的挑战，当总体规模和其构成类别的大小均未知时。在这里，我们提出了一种使用超几何似然解决这一估计挑战的新方法，即使存在严重的欠采样也能实现。我们开发了我们的方法，以解释一个数据生成过程，其中地面真实值是有条件的连续潜变量混合分布，比如协同过滤，使用变分自动编码器框架。实证数据模拟表明，我们的方法在人口规模估计的准确性和学习能力方面均优于其他用于建模计数数据的似然函数。

    arXiv:2402.14220v1 Announce Type: new  Abstract: The multivariate hypergeometric distribution describes sampling without replacement from a discrete population of elements divided into multiple categories. Addressing a gap in the literature, we tackle the challenge of estimating discrete distributions when both the total population size and the sizes of its constituent categories are unknown. Here, we propose a novel solution using the hypergeometric likelihood to solve this estimation challenge, even in the presence of severe under-sampling. We develop our approach to account for a data generating process where the ground-truth is a mixture of distributions conditional on a continuous latent variable, such as with collaborative filtering, using the variational autoencoder framework. Empirical data simulation demonstrates that our method outperforms other likelihood functions used to model count data, both in terms of accuracy of population size estimate and in its ability to learn an 
    
[^26]: 带有多个领域的本地分布偏移的乘幂稳健估计

    Multiply Robust Estimation for Local Distribution Shifts with Multiple Domains

    [https://arxiv.org/abs/2402.14145](https://arxiv.org/abs/2402.14145)

    提出了一种两阶段的乘幂稳健估计方法，用于改善表格数据分析中每个个体部分的模型性能，并建立了在测试风险上的理论保证。

    

    分布偏移在现实世界的机器学习应用中普遍存在，给在一个数据分布上训练的模型推广到另一个数据分布带来挑战。本文专注于数据分布随整个总体的多个部分变化的情形，并仅在每个部分内对训练与测试（部署）数据分布的差异进行局部假设。我们提出了一种两阶段的乘幂稳健估计方法，用于改善表格数据分析中每个个体部分的模型性能。该方法涉及拟合基于从多个部分的训练数据中学到的模型的线性组合，然后对每个部分进行细化。我们的方法旨在与常用的现成机器学习模型一起实施。我们在测试风险上建立了该方法泛化界限的理论保证。通过大量实验...

    arXiv:2402.14145v1 Announce Type: cross  Abstract: Distribution shifts are ubiquitous in real-world machine learning applications, posing a challenge to the generalization of models trained on one data distribution to another. We focus on scenarios where data distributions vary across multiple segments of the entire population and only make local assumptions about the differences between training and test (deployment) distributions within each segment. We propose a two-stage multiply robust estimation method to improve model performance on each individual segment for tabular data analysis. The method involves fitting a linear combination of the based models, learned using clusters of training data from multiple segments, followed by a refinement step for each segment. Our method is designed to be implemented with commonly used off-the-shelf machine learning models. We establish theoretical guarantees on the generalization bound of the method on the test risk. With extensive experiments
    
[^27]: 稀疏线性回归中不当学习的计算统计差距

    Computational-Statistical Gaps for Improper Learning in Sparse Linear Regression

    [https://arxiv.org/abs/2402.14103](https://arxiv.org/abs/2402.14103)

    该研究探讨了稀疏线性回归中的计算统计差距问题，为了高效地找到可以在样本上实现非平凡预测误差的潜在密集估计的回归向量，需要至少 $\Omega(k \log (d/k))$ 个样本。

    

    我们研究了稀疏线性回归中不当学习的计算统计差距。具体来说，给定来自维度为 $d$ 的 $k$-稀疏线性模型的 $n$ 个样本，我们询问了在时间多项式中的最小样本复杂度，以便高效地找到一个对这 $n$ 个样本达到非平凡预测误差的潜在密集估计的回归向量。信息理论上，这可以用 $\Theta(k \log (d/k))$ 个样本实现。然而，尽管在文献中很显著，但没有已知的多项式时间算法可以在不附加对模型的其他限制的情况下使用少于 $\Theta(d)$ 个样本达到相同的保证。类似地，现有的困难结果要么仅限于适当设置，在该设置中估计值也必须是稀疏的，要么仅适用于特定算法。

    arXiv:2402.14103v1 Announce Type: new  Abstract: We study computational-statistical gaps for improper learning in sparse linear regression. More specifically, given $n$ samples from a $k$-sparse linear model in dimension $d$, we ask what is the minimum sample complexity to efficiently (in time polynomial in $d$, $k$, and $n$) find a potentially dense estimate for the regression vector that achieves non-trivial prediction error on the $n$ samples. Information-theoretically this can be achieved using $\Theta(k \log (d/k))$ samples. Yet, despite its prominence in the literature, there is no polynomial-time algorithm known to achieve the same guarantees using less than $\Theta(d)$ samples without additional restrictions on the model. Similarly, existing hardness results are either restricted to the proper setting, in which the estimate must be sparse as well, or only apply to specific algorithms.   We give evidence that efficient algorithms for this task require at least (roughly) $\Omega(
    
[^28]: 社会环境设计

    Social Environment Design

    [https://arxiv.org/abs/2402.14090](https://arxiv.org/abs/2402.14090)

    该论文提出了一种新的研究议程，介绍了社会环境设计作为一种用于自动化政策制定的AI通用框架，旨在捕捉一般经济环境，通过AI模拟系统分析政府和经济政策，并强调未来基于AI的政策制定研究中的关键挑战。

    

    人工智能（AI）作为一种用于改善政府和经济政策制定的技术具有潜力。本文提出了一个新的研究议程，介绍了社会环境设计，这是一种用于自动化政策制定的AI通用框架，与强化学习、经济与计算社会选择社区相连接。该框架旨在捕捉一般经济环境，包括对政策目标的投票，并为通过AI模拟对政府和经济政策进行系统分析提供指导。我们强调了未来基于AI的政策制定研究中的关键开放问题。通过解决这些挑战，我们希望实现各种社会福利目标，从而促进更具道德和负责任的决策制定。

    arXiv:2402.14090v1 Announce Type: new  Abstract: Artificial Intelligence (AI) holds promise as a technology that can be used to improve government and economic policy-making. This paper proposes a new research agenda towards this end by introducing Social Environment Design, a general framework for the use of AI for automated policy-making that connects with the Reinforcement Learning, EconCS, and Computational Social Choice communities. The framework seeks to capture general economic environments, includes voting on policy objectives, and gives a direction for the systematic analysis of government and economic policy through AI simulation. We highlight key open problems for future research in AI-based policy-making. By solving these challenges, we hope to achieve various social welfare objectives, thereby promoting more ethical and responsible decision making.
    
[^29]: 使用具有运动代码的随机过程模型对嘈杂时间序列集合进行鲁棒学习

    Robust Learning of Noisy Time Series Collections Using Stochastic Process Models with Motion Codes

    [https://arxiv.org/abs/2402.14081](https://arxiv.org/abs/2402.14081)

    使用具有学习谱核的混合高斯过程的潜变量模型方法，针对嘈杂时间序列数据进行鲁棒学习。

    

    虽然时间序列分类和预测问题已经得到广泛研究，但具有任意时间序列长度的嘈杂时间序列数据的情况仍具挑战性。每个时间序列实例可以看作是嘈杂动态模型的一个样本实现，其特点是连续随机过程。对于许多应用，数据是混合的，由多个随机过程建模的几种类型的嘈杂时间序列序列组成，使得预测和分类任务变得更具挑战性。我们不是简单地将数据回归到每种时间序列类型，而是采用具有学习谱核的混合高斯过程的潜变量模型方法。更具体地说，我们为每种类型的嘈杂时间序列数据自动分配一个称为其运动代码的签名向量。然后，在每个分配的运动代码的条件下，我们推断出相关性的稀疏近似。

    arXiv:2402.14081v1 Announce Type: cross  Abstract: While time series classification and forecasting problems have been extensively studied, the cases of noisy time series data with arbitrary time sequence lengths have remained challenging. Each time series instance can be thought of as a sample realization of a noisy dynamical model, which is characterized by a continuous stochastic process. For many applications, the data are mixed and consist of several types of noisy time series sequences modeled by multiple stochastic processes, making the forecasting and classification tasks even more challenging. Instead of regressing data naively and individually to each time series type, we take a latent variable model approach using a mixtured Gaussian processes with learned spectral kernels. More specifically, we auto-assign each type of noisy time series data a signature vector called its motion code. Then, conditioned on each assigned motion code, we infer a sparse approximation of the corr
    
[^30]: 高效的规范化置信预测与不确定性量化：基于深度回归森林的抗癌药物敏感性预测

    Efficient Normalized Conformal Prediction and Uncertainty Quantification for Anti-Cancer Drug Sensitivity Prediction with Deep Regression Forests

    [https://arxiv.org/abs/2402.14080](https://arxiv.org/abs/2402.14080)

    通过深度回归森林计算样本方差，提高了抗癌药物敏感性预测中的规范化置信预测效率和覆盖率

    

    深度学习模型正在被应用于各种关键决策任务，然而它们被训练为提供点预测而没有提供信心度。如果与不确定性估计结合，深度学习模型的可信度可以得到提高。置信预测已经被证明是一种有希望的方法，可以将机器学习模型与预测区间配对，从而可以看到模型的不确定性。然而，常见的用于置信预测的不确定性估计方法未能提供对所有样本同样准确的异方差间隔。本文提出了一种方法，通过从深度回归森林获得的方差来估计每个样本的不确定性。我们展示了深度回归森林的方差如何提高药物反应预测任务上规范化诱导置信预测的效率和覆盖率。

    arXiv:2402.14080v1 Announce Type: cross  Abstract: Deep learning models are being adopted and applied on various critical decision-making tasks, yet they are trained to provide point predictions without providing degrees of confidence. The trustworthiness of deep learning models can be increased if paired with uncertainty estimations. Conformal Prediction has emerged as a promising method to pair machine learning models with prediction intervals, allowing for a view of the model's uncertainty. However, popular uncertainty estimation methods for conformal prediction fail to provide heteroskedastic intervals that are equally accurate for all samples. In this paper, we propose a method to estimate the uncertainty of each sample by calculating the variance obtained from a Deep Regression Forest. We show that the deep regression forest variance improves the efficiency and coverage of normalized inductive conformal prediction on a drug response prediction task.
    
[^31]: 用于顺序随机投影的概率工具

    Probability Tools for Sequential Random Projection

    [https://arxiv.org/abs/2402.14026](https://arxiv.org/abs/2402.14026)

    该论文提出了适用于顺序随机投影的概率框架，通过构建停止过程并采用混合方法，实现了对一系列相互连接的浓缩事件的分析，从而创造了对Johnson-Lindenstrauss引理的非平凡鞅扩展。

    

    我们引入了第一个专为顺序随机投影定制的概率框架，这种方法植根于面对不确定性的顺序决策的挑战。分析受到随机变量的顺序依赖和高维性质的影响，这是顺序决策过程中固有的自适应机制的副产品。我们的工作特点是构建了一个停止过程，便于分析一系列以顺序方式相互连接的浓缩事件。通过在从停止过程得出的自规范过程内采用混合方法，我们实现了所需的非渐近概率界限。该界限代表了对Johnson-Lindenstrauss（JL）引理的一个非平凡的鞅扩展，标志着对随机投影和顺序分析的文献做出了开创性贡献。

    arXiv:2402.14026v1 Announce Type: cross  Abstract: We introduce the first probabilistic framework tailored for sequential random projection, an approach rooted in the challenges of sequential decision-making under uncertainty. The analysis is complicated by the sequential dependence and high-dimensional nature of random variables, a byproduct of the adaptive mechanisms inherent in sequential decision processes. Our work features a novel construction of a stopped process, facilitating the analysis of a sequence of concentration events that are interconnected in a sequential manner. By employing the method of mixtures within a self-normalized process, derived from the stopped process, we achieve a desired non-asymptotic probability bound. This bound represents a non-trivial martingale extension of the Johnson-Lindenstrauss (JL) lemma, marking a pioneering contribution to the literature on random projection and sequential analysis.
    
[^32]: 重新审视AdaGrad在宽松假设下的收敛性

    Revisiting Convergence of AdaGrad with Relaxed Assumptions

    [https://arxiv.org/abs/2402.13794](https://arxiv.org/abs/2402.13794)

    重新审视了AdaGrad在非凸光滑优化问题上的收敛性，提出了通用噪声模型，得出了概率收敛速度，无需先验知识，且可以在噪声参数足够小时加速至更快的速度。

    

    在这项研究中，我们重新审视了AdaGrad在非凸光滑优化问题上的收敛性，包括AdaGrad作为一种特殊情况。我们考虑了一个通用的噪声模型，其中噪声的大小由函数值差和梯度大小控制。这个模型涵盖了广泛范围的噪声，包括有界噪声、次高斯噪声、仿射方差噪声和预期光滑度，并且在许多实际应用中被证明更加现实。我们的分析得出了一个概率收敛速度，根据通用噪声，可以达到( \tilde{\mathcal{O}}(1/\sqrt{T}))。这个速度不依赖于先前对问题参数的了解，当与函数值差和噪声水平相关的参数足够小时，它可以加速到(\tilde{\mathcal{O}}(1/T))，其中(T)表示总迭代次数。收敛速度因此匹配了下限速度。

    arXiv:2402.13794v1 Announce Type: cross  Abstract: In this study, we revisit the convergence of AdaGrad with momentum (covering AdaGrad as a special case) on non-convex smooth optimization problems. We consider a general noise model where the noise magnitude is controlled by the function value gap together with the gradient magnitude. This model encompasses a broad range of noises including bounded noise, sub-Gaussian noise, affine variance noise and the expected smoothness, and it has been shown to be more realistic in many practical applications. Our analysis yields a probabilistic convergence rate which, under the general noise, could reach at (\tilde{\mathcal{O}}(1/\sqrt{T})). This rate does not rely on prior knowledge of problem-parameters and could accelerate to (\tilde{\mathcal{O}}(1/T)) where (T) denotes the total number iterations, when the noise parameters related to the function value gap and noise level are sufficiently small. The convergence rate thus matches the lower rat
    
[^33]: 一种界定尾部概率的方法

    A Method For Bounding Tail Probabilities

    [https://arxiv.org/abs/2402.13662](https://arxiv.org/abs/2402.13662)

    提出了一种界定连续随机变量右尾和左尾概率上下界的方法，通过设置特定的函数，得到了新的上下界限，并与马尔可夫不等式建立了联系

    

    我们提出了一种方法，用于上下界定连续随机变量（RVs）的右尾和左尾概率。对于具有概率密度函数$f_X(x)$的RV $X$的右尾概率，该方法首先要求设置一个连续的、正的、严格递减的函数$g_X(x)$，使得$-f_X(x)/g'_X(x)$是一个递减且递增的函数，$\forall x>x_0$，分别给出形式为$-f_X(x) g_X(x)/g'_X(x)$的上界和下界，$\forall x>x_0$，其中$x_0$是某个点。类似地，对于$X$的左尾概率的上下界，该方法首先要求设置一个连续的、正的、严格递增的函数$g_X(x)$，使得$f_X(x)/g'_X(x)$是一个增加且递减的函数，$\forall x<x_0$。我们提供了一些函数$g_X(x)$的良好候选示例。我们还建立了新界限与马尔可夫不等式的联系。

    arXiv:2402.13662v1 Announce Type: cross  Abstract: We present a method for upper and lower bounding the right and the left tail probabilities of continuous random variables (RVs). For the right tail probability of RV $X$ with probability density function $f_X(x)$, this method requires first setting a continuous, positive, and strictly decreasing function $g_X(x)$ such that $-f_X(x)/g'_X(x)$ is a decreasing and increasing function, $\forall x>x_0$, which results in upper and lower bounds, respectively, given in the form $-f_X(x) g_X(x)/g'_X(x)$, $\forall x>x_0$, where $x_0$ is some point. Similarly, for the upper and lower bounds on the left tail probability of $X$, this method requires first setting a continuous, positive, and strictly increasing function $g_X(x)$ such that $f_X(x)/g'_X(x)$ is an increasing and decreasing function, $\forall x<x_0$. We provide some examples of good candidates for the function $g_X(x)$. We also establish connections between the new bounds and Markov's in
    
[^34]: 在奇异性下的学习：改进WBIC和sBIC的信息准则

    Learning under Singularity: An Information Criterion improving WBIC and sBIC

    [https://arxiv.org/abs/2402.12762](https://arxiv.org/abs/2402.12762)

    LS信息准则旨在增强WBIC和sBIC的功能，有效处理非正则情况，具有稳定性，为奇异情况下的信息准则提供了新的方法

    

    我们介绍了一种新颖的信息准则（IC），称为在奇异性下的学习（LS），旨在增强广泛适用的贝叶斯信息准则（WBIC）和奇异贝叶斯信息准则（sBIC）的功能。 LS在没有正则性约束的情况下是有效的，并表现出稳定性。Watanabe定义了一个统计模型或学习机器为正则，如果从参数到概率分布的映射是一对一的，并且其Fisher信息矩阵是正定的。相反，不符合这些条件的模型被称为奇异。 在过去的十年中，已经提出了几种奇异情况下的信息准则，包括WBIC和sBIC。 WBIC适用于非正则情况，但在样本量很大且已知学习系数估计冗余时面临挑战。 相反，sBIC在广泛应用方面存在限制，因为它依赖于最大似然估计。

    arXiv:2402.12762v1 Announce Type: cross  Abstract: We introduce a novel Information Criterion (IC), termed Learning under Singularity (LS), designed to enhance the functionality of the Widely Applicable Bayes Information Criterion (WBIC) and the Singular Bayesian Information Criterion (sBIC). LS is effective without regularity constraints and demonstrates stability. Watanabe defined a statistical model or a learning machine as regular if the mapping from a parameter to a probability distribution is one-to-one and its Fisher information matrix is positive definite. In contrast, models not meeting these conditions are termed singular. Over the past decade, several information criteria for singular cases have been proposed, including WBIC and sBIC. WBIC is applicable in non-regular scenarios but faces challenges with large sample sizes and redundant estimation of known learning coefficients. Conversely, sBIC is limited in its broader application due to its dependence on maximum likelihood
    
[^35]: BlackJAX: JAX中的组合式贝叶斯推断

    BlackJAX: Composable Bayesian inference in JAX

    [https://arxiv.org/abs/2402.10797](https://arxiv.org/abs/2402.10797)

    BlackJAX是一个实现在JAX中组合式贝叶斯推断的库，采用函数式方法提高易用性、速度和模块化，适用于需要尖端方法、研究人员和想要了解工作原理的人。

    

    BlackJAX是一个库，实现了在贝叶斯计算中常用的抽样和变分推断算法。它通过采用函数式方法实现算法，旨在提高易用性、速度和模块化。BlackJAX使用Python编写，利用JAX在CPU、GPU和TPU上编译和运行类似Numpy的抽样器和变分方法。该库通过直接处理（非正则化）目标对数密度函数，与概率编程语言很好地集成。BlackJAX旨在成为基本统计“基元”的低级可组合实现的集合，可组合执行定义良好的贝叶斯推断，同时还提供高级例程以提高易用性。它面向需要尖端方法的用户、希望创建复杂抽样方法的研究人员，以及想要了解这些方法工作原理的人。

    arXiv:2402.10797v1 Announce Type: cross  Abstract: BlackJAX is a library implementing sampling and variational inference algorithms commonly used in Bayesian computation. It is designed for ease of use, speed, and modularity by taking a functional approach to the algorithms' implementation. BlackJAX is written in Python, using JAX to compile and run NumpPy-like samplers and variational methods on CPUs, GPUs, and TPUs. The library integrates well with probabilistic programming languages by working directly with the (un-normalized) target log density function. BlackJAX is intended as a collection of low-level, composable implementations of basic statistical 'atoms' that can be combined to perform well-defined Bayesian inference, but also provides high-level routines for ease of use. It is designed for users who need cutting-edge methods, researchers who want to create complex sampling methods, and people who want to learn how these work.
    
[^36]: EduGym: 用于强化学习教育的环境和笔记本套件

    EduGym: An Environment and Notebook Suite for Reinforcement Learning Education

    [https://arxiv.org/abs/2311.10590](https://arxiv.org/abs/2311.10590)

    EduGym是一套用于强化学习教育的环境和笔记本套件，旨在解决学生在转换理论和实践中遇到的困难。

    

    由于强化学习的经验成功，越来越多的学生在学习这个课题。然而，根据我们的实际教学经验，我们发现学生在进入这个领域（本科生、硕士生和早期博士生）时常常遇到困难。一方面，教科书和（在线）讲座提供了基础知识，但学生发现很难在方程式和代码之间进行转换。另一方面，公共代码库提供了实际的例子，但实现的算法往往复杂，并且基础测试环境同时包含多个强化学习挑战。尽管这在研究角度上是现实的，但它经常阻碍了教育概念的理解。为了解决这个问题，我们推出了EduGym，这是一组专门针对教育的强化学习环境和相关交互式笔记本。

    arXiv:2311.10590v2 Announce Type: replace-cross  Abstract: Due to the empirical success of reinforcement learning, an increasing number of students study the subject. However, from our practical teaching experience, we see students entering the field (bachelor, master and early PhD) often struggle. On the one hand, textbooks and (online) lectures provide the fundamentals, but students find it hard to translate between equations and code. On the other hand, public codebases do provide practical examples, but the implemented algorithms tend to be complex, and the underlying test environments contain multiple reinforcement learning challenges at once. Although this is realistic from a research perspective, it often hinders educational conceptual understanding. To solve this issue we introduce EduGym, a set of educational reinforcement learning environments and associated interactive notebooks tailored for education. Each EduGym environment is specifically designed to illustrate a certain 
    
[^37]: 使用概率图神经网络对时空出行需求的不确定性建模

    Uncertainty Quantification of Spatiotemporal Travel Demand with Probabilistic Graph Neural Networks

    [https://arxiv.org/abs/2303.04040](https://arxiv.org/abs/2303.04040)

    该研究提出了一种概率图神经网络（Prob-GNN）框架，用于量化出行需求的时空不确定性，实证应用表明概率假设对不确定性预测影响大于确定性假设。

    

    近期的研究显著提高了使用图神经网络预测出行需求的准确性。然而，这些研究很大程度上忽略了出行需求预测中不可避免的不确定性。为了填补这一空白，本研究提出了一种概率图神经网络（Prob-GNN）框架，用于量化出行需求的时空不确定性。这个Prob-GNN框架基于确定性和概率假设，并在芝加哥市预测公共交通和拼车需求的任务上得到了实证应用。我们发现，概率假设（如分布尾部、支持）对不确定性预测的影响大于确定性假设（如深度模块、深度）。在Prob-GNN家族中，采用截断高斯和拉普拉斯分布的GNN在公共交通和拼车数据中表现最佳。即使在存在明显域偏移情况下，Prob-GNNs

    arXiv:2303.04040v2 Announce Type: replace  Abstract: Recent studies have significantly improved the prediction accuracy of travel demand using graph neural networks. However, these studies largely ignored uncertainty that inevitably exists in travel demand prediction. To fill this gap, this study proposes a framework of probabilistic graph neural networks (Prob-GNN) to quantify the spatiotemporal uncertainty of travel demand. This Prob-GNN framework is substantiated by deterministic and probabilistic assumptions, and empirically applied to the task of predicting the transit and ridesharing demand in Chicago. We found that the probabilistic assumptions (e.g. distribution tail, support) have a greater impact on uncertainty prediction than the deterministic ones (e.g. deep modules, depth). Among the family of Prob-GNNs, the GNNs with truncated Gaussian and Laplace distributions achieve the highest performance in transit and ridesharing data. Even under significant domain shifts, Prob-GNNs
    
[^38]: 基于阈值的自动标注的优势与局限性

    Promises and Pitfalls of Threshold-based Auto-labeling

    [https://arxiv.org/abs/2211.12620](https://arxiv.org/abs/2211.12620)

    TBAL系统可以通过验证数据自动标注未标注数据，减少手动标注的依赖；研究结果展示了即使模型表现不佳也可以准确自动标记数据，并揭示了TBAL系统的潜在缺陷

    

    创建大规模高质量标记数据集是监督机器学习工作流程中的一个主要瓶颈。阈值自动标注（TBAL）通过使用人类获取的验证数据来寻找一个置信阈值，高于该阈值的数据将由机器标记，从而减少了对手动注释的依赖。TBAL正逐渐成为实践中被广泛采用的解决方案。鉴于所得数据的长期有效性和多样化使用，理解这种自动标注系统获取的数据何时可以被依赖是至关重要的。这是第一项分析TBAL系统并推导需要保证机器标记数据质量的人工标记验证数据量样本复杂性界限的工作。我们的结果提供了两个关键见解。首先，表面上糟糕的模型可以自动、准确地标记合理数量的未标记数据。其次，TBAL系统的一个隐藏的缺点是潜在地

    arXiv:2211.12620v2 Announce Type: replace-cross  Abstract: Creating large-scale high-quality labeled datasets is a major bottleneck in supervised machine learning workflows. Threshold-based auto-labeling (TBAL), where validation data obtained from humans is used to find a confidence threshold above which the data is machine-labeled, reduces reliance on manual annotation. TBAL is emerging as a widely-used solution in practice. Given the long shelf-life and diverse usage of the resulting datasets, understanding when the data obtained by such auto-labeling systems can be relied on is crucial. This is the first work to analyze TBAL systems and derive sample complexity bounds on the amount of human-labeled validation data required for guaranteeing the quality of machine-labeled data. Our results provide two crucial insights. First, reasonable chunks of unlabeled data can be automatically and accurately labeled by seemingly bad models. Second, a hidden downside of TBAL systems is potentially
    
[^39]: 使用PAC-Bayes界限同时控制多个错误

    Controlling Multiple Errors Simultaneously with a PAC-Bayes Bound

    [https://arxiv.org/abs/2202.05560](https://arxiv.org/abs/2202.05560)

    该研究提出了一种PAC-Bayes界限，能够同时控制多个错误，并提供丰富的信息，适用于回归中测试损失分布或分类中不同错误分类的概率。

    

    当前的PAC-Bayes泛化界限仅限于性能的标量度量，如损失或错误率。我们提供了第一个能够提供丰富信息的PAC-Bayes界限，通过界定一组M种错误类型的经验概率与真实概率之间的Kullback-Leibler差异来控制可能结果的整个分布。

    arXiv:2202.05560v2 Announce Type: replace-cross  Abstract: Current PAC-Bayes generalisation bounds are restricted to scalar metrics of performance, such as the loss or error rate. However, one ideally wants more information-rich certificates that control the entire distribution of possible outcomes, such as the distribution of the test loss in regression, or the probabilities of different mis classifications. We provide the first PAC-Bayes bound capable of providing such rich information by bounding the Kullback-Leibler divergence between the empirical and true probabilities of a set of M error types, which can either be discretized loss values for regression, or the elements of the confusion matrix (or a partition thereof) for classification. We transform our bound into a differentiable training objective. Our bound is especially useful in cases where the severity of different mis-classifications may change over time; existing PAC-Bayes bounds can only bound a particular pre-decided w
    
[^40]: FIGARO：具有细粒度艺术控制的符号音乐生成

    FIGARO: Generating Symbolic Music with Fine-Grained Artistic Control

    [https://arxiv.org/abs/2201.10936](https://arxiv.org/abs/2201.10936)

    提出了一种自监督的描述-序列任务，实现了在全局水平上对生成音乐的细粒度可控，通过结合高级特征和领域知识，在符号音乐生成方面取得了最新的成果

    

    使用深度神经网络生成音乐近年来一直是一个活跃的研究领域。虽然生成样本的质量不断提高，但大多数方法只能对生成的序列施加最小的控制，甚至没有。我们提出了自监督的描述-序列任务，它允许在全局水平上进行细粒度可控生成。我们通过提取有关目标序列的高级特征，以及学习给定相应高级描述时序列的条件分布，来实现这一点。我们通过将描述-序列建模应用到符号音乐中来训练FIGARO（基于注意力的、鲁棒控制的细粒度音乐生成）。通过将学习到的高级特征与领域知识结合，作为强归纳偏差，该模型在可控符号音乐生成方面实现了最新的成果。

    arXiv:2201.10936v4 Announce Type: replace-cross  Abstract: Generating music with deep neural networks has been an area of active research in recent years. While the quality of generated samples has been steadily increasing, most methods are only able to exert minimal control over the generated sequence, if any. We propose the self-supervised description-to-sequence task, which allows for fine-grained controllable generation on a global level. We do so by extracting high-level features about the target sequence and learning the conditional distribution of sequences given the corresponding high-level description in a sequence-to-sequence modelling setup. We train FIGARO (FIne-grained music Generation via Attention-based, RObust control) by applying description-to-sequence modelling to symbolic music. By combining learned high level features with domain knowledge, which acts as a strong inductive bias, the model achieves state-of-the-art results in controllable symbolic music generation a
    
[^41]: 对抗机器学习：贝叶斯视角

    Adversarial Machine Learning: Bayesian Perspectives

    [https://arxiv.org/abs/2003.03546](https://arxiv.org/abs/2003.03546)

    AML旨在保护机器学习系统免受安全威胁，贝叶斯视角为防御提供了新的好处

    

    对抗机器学习(AML)正在成为一个重要的领域，旨在保护机器学习(ML)系统免受安全威胁：在某些情况下，可能存在敌对方积极操纵输入数据以欺骗学习系统。 这创造了一类新的安全漏洞，ML系统可能会面临，并引入了一种新的被称为敌对稳健性的可信操作所必需的性质。 大部分AML工作都建立在对抗学习系统和准备操纵输入数据的对手之间冲突的博弈论建模之上。 这假设每个代理都了解对手的兴趣和不确定性判断，从而促进基于Nash均衡的推理。 然而，在AML典型的安全方案中，这种共同知识假设并不现实。 在回顾了这种博弈论方法之后，我们讨论了贝叶斯视角在防御中提供的好处

    arXiv:2003.03546v2 Announce Type: replace  Abstract: Adversarial Machine Learning (AML) is emerging as a major field aimed at protecting machine learning (ML) systems against security threats: in certain scenarios there may be adversaries that actively manipulate input data to fool learning systems. This creates a new class of security vulnerabilities that ML systems may face, and a new desirable property called adversarial robustness essential to trust operations based on ML outputs. Most work in AML is built upon a game-theoretic modelling of the conflict between a learning system and an adversary, ready to manipulate input data. This assumes that each agent knows their opponent's interests and uncertainty judgments, facilitating inferences based on Nash equilibria. However, such common knowledge assumption is not realistic in the security scenarios typical of AML. After reviewing such game-theoretic approaches, we discuss the benefits that Bayesian perspectives provide when defendin
    
[^42]: 论费曼-卡克训练部分贝叶斯神经网络

    On Feynman--Kac training of partial Bayesian neural networks. (arXiv:2310.19608v1 [cs.LG])

    [http://arxiv.org/abs/2310.19608](http://arxiv.org/abs/2310.19608)

    本文提出了一种将部分贝叶斯神经网络训练转化为模拟费曼-卡克模型的高效采样训练策略，并通过各种数据集的实验证明其在预测性能方面优于现有技术。

    

    最近，部分贝叶斯神经网络(pBNNs)被证明与全贝叶斯神经网络具有竞争力，但pBNNs在潜变量空间中往往是多峰的，因此用参数模型来近似是具有挑战性的。为了解决这个问题，我们提出了一种高效的基于采样的训练策略，即将pBNN的训练转化为模拟费曼-卡克模型。我们还描述了序贯蒙特卡洛采样器的变种，使我们能够以可行的计算成本同时估计参数和该模型的潜在后验分布。我们在各种合成和真实世界的数据集上展示了我们提出的训练方案在预测性能方面优于现有技术。

    Recently, partial Bayesian neural networks (pBNNs), which only consider a subset of the parameters to be stochastic, were shown to perform competitively with full Bayesian neural networks. However, pBNNs are often multi-modal in the latent-variable space and thus challenging to approximate with parametric models. To address this problem, we propose an efficient sampling-based training strategy, wherein the training of a pBNN is formulated as simulating a Feynman--Kac model. We then describe variations of sequential Monte Carlo samplers that allow us to simultaneously estimate the parameters and the latent posterior distribution of this model at a tractable computational cost. We show on various synthetic and real-world datasets that our proposed training scheme outperforms the state of the art in terms of predictive performance.
    
[^43]: 基于流的分布鲁棒优化

    Flow-based Distributionally Robust Optimization. (arXiv:2310.19253v1 [cs.LG])

    [http://arxiv.org/abs/2310.19253](http://arxiv.org/abs/2310.19253)

    这项研究提出了一种称为FlowDRO的计算高效框架，用于解决基于流的分布鲁棒优化问题，通过使用流模型和Wasserstein近端梯度流类型的算法，实现了对具有更大样本大小的问题的可扩展性和更好的泛化能力。

    

    我们提出了一种称为FlowDRO的计算高效框架，用于解决基于流的分布鲁棒优化（DRO）问题，其中要求最坏情况分布（也称为最不利分布，LFD）是连续的，从而使得算法能够可扩展到具有更大样本大小的问题，并实现对诱导的鲁棒算法的更好泛化能力。为了解决计算上具有挑战性的无限维优化问题，我们利用基于流的模型，在数据分布和目标分布之间进行连续时间可逆传输映射，并开发了一种Wasserstein近端梯度流类型的算法。在实践中，我们通过梯度下降逐步训练块内的神经网络序列来参数化传输映射。我们的计算框架通用，能够处理高维数据和大样本大小，并可用于各种应用。

    We present a computationally efficient framework, called \texttt{FlowDRO}, for solving flow-based distributionally robust optimization (DRO) problems with Wasserstein uncertainty sets, when requiring the worst-case distribution (also called the Least Favorable Distribution, LFD) to be continuous so that the algorithm can be scalable to problems with larger sample sizes and achieve better generalization capability for the induced robust algorithms. To tackle the computationally challenging infinitely dimensional optimization problem, we leverage flow-based models, continuous-time invertible transport maps between the data distribution and the target distribution, and develop a Wasserstein proximal gradient flow type of algorithm. In practice, we parameterize the transport maps by a sequence of neural networks progressively trained in blocks by gradient descent. Our computational framework is general, can handle high-dimensional data with large sample sizes, and can be useful for various
    
[^44]: 外部验证策略评估结合试验和观察数据

    Externally Valid Policy Evaluation Combining Trial and Observational Data. (arXiv:2310.14763v1 [stat.ME])

    [http://arxiv.org/abs/2310.14763](http://arxiv.org/abs/2310.14763)

    这项研究提出了一种结合试验和观察数据的外部有效策略评估方法，利用试验数据对目标人群上的政策结果进行有效推断，并给出了可验证的评估结果。

    

    随机试验被广泛认为是评估决策策略影响的金 standard。然而，试验数据来自可能与目标人群不同的人群，这引发了外部效度（也称为泛化能力）的问题。在本文中，我们试图利用试验数据对目标人群上的政策结果进行有效推断。目标人群的额外协变量数据用于模拟试验研究中个体的抽样。我们开发了一种方法，在任何指定的模型未校准范围内产生可验证的基于试验的政策评估。该方法是非参数的，即使样本是有限的，有效性也得到保证。使用模拟和实际数据说明了认证的政策评估结果。

    Randomized trials are widely considered as the gold standard for evaluating the effects of decision policies. Trial data is, however, drawn from a population which may differ from the intended target population and this raises a problem of external validity (aka. generalizability). In this paper we seek to use trial data to draw valid inferences about the outcome of a policy on the target population. Additional covariate data from the target population is used to model the sampling of individuals in the trial study. We develop a method that yields certifiably valid trial-based policy evaluations under any specified range of model miscalibrations. The method is nonparametric and the validity is assured even with finite samples. The certified policy evaluations are illustrated using both simulated and real data.
    
[^45]: 改进的精细离散化方法提高自适应在线学习

    Improving Adaptive Online Learning Using Refined Discretization. (arXiv:2309.16044v1 [cs.LG])

    [http://arxiv.org/abs/2309.16044](http://arxiv.org/abs/2309.16044)

    通过一种新颖的连续时间启发式算法，提高了自适应在线学习的效果，将梯度方差的依赖性从次优的$O(\sqrt{V_T\log V_T})$改进到最优速率$O(\sqrt{V_T})$，并可适用于未知Lipschitz常数的情况。

    

    我们研究了具有Lipschitz损失的非约束在线线性优化问题。目标是同时达到（i）二阶梯度自适应性；和（ii）比较器范数自适应性，也被称为文献中的“参数自由性”。现有的遗憾界（Cutkosky和Orabona，2018；Mhammedi和Koolen，2020；Jacobsen和Cutkosky，2022）对于梯度方差$V_T$有次优的$O(\sqrt{V_T\log V_T})$依赖性，而本工作利用一种新颖的连续时间启发式算法将其改进为最优速率$O(\sqrt{V_T})$，而无需任何不切实际的加倍技巧。这一结果可以推广到未知Lipschitz常数的情况，消除了先前工作中的范围比率问题（Mhammedi和Koolen，2020）。具体来说，我们首先展示了在问题的连续时间类比中可以相当容易地实现目标的同时适应性，其中环境由任意连续半鞘式建模。然后，我们的关键创新是

    We study unconstrained Online Linear Optimization with Lipschitz losses. The goal is to simultaneously achieve ($i$) second order gradient adaptivity; and ($ii$) comparator norm adaptivity also known as "parameter freeness" in the literature. Existing regret bounds (Cutkosky and Orabona, 2018; Mhammedi and Koolen, 2020; Jacobsen and Cutkosky, 2022) have the suboptimal $O(\sqrt{V_T\log V_T})$ dependence on the gradient variance $V_T$, while the present work improves it to the optimal rate $O(\sqrt{V_T})$ using a novel continuous-time-inspired algorithm, without any impractical doubling trick. This result can be extended to the setting with unknown Lipschitz constant, eliminating the range ratio problem from prior works (Mhammedi and Koolen, 2020).  Concretely, we first show that the aimed simultaneous adaptivity can be achieved fairly easily in a continuous time analogue of the problem, where the environment is modeled by an arbitrary continuous semimartingale. Then, our key innovation 
    
[^46]: 黑盒变分推断的线性收敛性：我们应该坚持到底吗？

    Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?. (arXiv:2307.14642v1 [stat.ML])

    [http://arxiv.org/abs/2307.14642](http://arxiv.org/abs/2307.14642)

    本文证明了带有控制变量的黑盒变分推断在完美变分族规范下以几何速度收敛，为BBVI提供了收敛性保证，同时提出了对熵梯度估计器的改进，对比了STL估计器，并给出了明确的非渐近复杂度保证。

    

    我们证明了带有控制变量的黑盒变分推断（BBVI），特别是着陆稳定（STL）估计器，在完美变分族规范下收敛于几何（传统上称为“线性”）速度。特别地，我们证明了STL估计器的梯度方差的二次界限，该界限包括了误指定的变分族。结合先前关于二次方差条件的工作，这直接暗示了在使用投影随机梯度下降的情况下BBVI的收敛性。我们还改进了现有对于正常封闭形式熵梯度估计器的分析，这使得我们能够将其与STL估计器进行比较，并为两者提供明确的非渐进复杂度保证。

    We prove that black-box variational inference (BBVI) with control variates, particularly the sticking-the-landing (STL) estimator, converges at a geometric (traditionally called "linear") rate under perfect variational family specification. In particular, we prove a quadratic bound on the gradient variance of the STL estimator, one which encompasses misspecified variational families. Combined with previous works on the quadratic variance condition, this directly implies convergence of BBVI with the use of projected stochastic gradient descent. We also improve existing analysis on the regular closed-form entropy gradient estimators, which enables comparison against the STL estimator and provides explicit non-asymptotic complexity guarantees for both.
    
[^47]: 自助聚合和置信度度量方法用于改进时间序列因果发现

    Bootstrap aggregation and confidence measures to improve time series causal discovery. (arXiv:2306.08946v1 [stat.ME])

    [http://arxiv.org/abs/2306.08946](http://arxiv.org/abs/2306.08946)

    本文介绍了一种新的自助聚合和置信度度量方法，使得时间序列因果发现能够提供连接的置信度度量。在广泛的数值实验中，实验证明该方法提高了因果发现的性能。

    

    因果发现方法已经展示了识别表示动态系统的因果时间依赖结构的时序图的能力。然而，它们不包括对估计连接的置信度的测量。本文介绍了一种新的自助聚合（Bagging）和置信度度量方法，它与时间序列因果发现相结合。该方法允许通过在保留时间依赖性的情况下对原始时间序列数据集进行自助重采样来测量由因果发现方法计算出的时序图连接的置信度。除了置信度量，聚合引导图通过多数投票得出最终聚合输出图。在本文中，我们将我们的方法与最先进的基于条件独立性算法PCMCI+相结合。通过广泛的数值实验，我们实验性地展示了Bagged-PCMCI+除了提供连接的置信度度量外，还可以提高因果发现的性能。

    Causal discovery methods have demonstrated the ability to identify the time series graphs representing the causal temporal dependency structure of dynamical systems. However, they do not include a measure of the confidence of the estimated links. Here, we introduce a novel bootstrap aggregation (bagging) and confidence measure method that is combined with time series causal discovery. This new method allows measuring confidence for the links of the time series graphs calculated by causal discovery methods. This is done by bootstrapping the original times series data set while preserving temporal dependencies. Next to confidence measures, aggregating the bootstrapped graphs by majority voting yields a final aggregated output graph. In this work, we combine our approach with the state-of-the-art conditional-independence-based algorithm PCMCI+. With extensive numerical experiments we empirically demonstrate that, in addition to providing confidence measures for links, Bagged-PCMCI+ improv
    
[^48]: 无领域偏见批量贝叶斯优化，通过贝叶斯积分处理多种约束条件

    Domain-Agnostic Batch Bayesian Optimization with Diverse Constraints via Bayesian Quadrature. (arXiv:2306.05843v1 [cs.LG])

    [http://arxiv.org/abs/2306.05843](http://arxiv.org/abs/2306.05843)

    本论文提出了cSOBER，一种处理多样化约束条件、离散和混合空间、未知约束以及查询拒绝问题的领域无关型贝叶斯优化算法。

    

    现实世界的优化问题通常具有多样的约束条件、离散和混合空间、高度可并行化等特点。同时，当存在未知约束时，例如在药物发现和动物实验安全性等领域，必须确立未知约束之后才能查询目标函数。现有工作通常仅针对上述某些特征而并非综合考虑。本文提出了cSOBER，一种基于SOBER算法的领域无关型谨慎并行主动采样器，考虑到了未知约束情况下的集成误差的影响并提出了处理方法，处理多种约束条件和未知约束查询拒绝的问题。

    Real-world optimisation problems often feature complex combinations of (1) diverse constraints, (2) discrete and mixed spaces, and are (3) highly parallelisable. (4) There are also cases where the objective function cannot be queried if unknown constraints are not satisfied, e.g. in drug discovery, safety on animal experiments (unknown constraints) must be established before human clinical trials (querying objective function) may proceed. However, most existing works target each of the above three problems in isolation and do not consider (4) unknown constraints with query rejection. For problems with diverse constraints and/or unconventional input spaces, it is difficult to apply these techniques as they are often mutually incompatible. We propose cSOBER, a domain-agnostic prudent parallel active sampler for Bayesian optimisation, based on SOBER of Adachi et al. (2023). We consider infeasibility under unknown constraints as a type of integration error that we can estimate. We propose 
    
[^49]: 有意义的因果聚合和悖论性混淆

    Meaningful Causal Aggregation and Paradoxical Confounding. (arXiv:2304.11625v1 [cs.AI])

    [http://arxiv.org/abs/2304.11625](http://arxiv.org/abs/2304.11625)

    聚合变量上的因果性不确定性可能会使得原本不混淆的因果关系变得混淆，在实际应用中，我们需要接受宏观因果关系通常只与微观状态相关的事实。

    

    在聚合变量中，干预的影响通常是不确定的，因为相同的宏观干预的不同微观实现可能会导致下游宏观变量的不同变化。我们表明，对于聚合变量，因果性的不确定性可以使得原本不混淆的因果关系变得混淆，并且反之亦然，这一点取决于相应的微观实现。我们认为，只有在聚合因果系统没有这种不确定性的情况下，我们才可以实际应用这种方法。否则，我们需要接受一点，就是宏观因果关系通常只与微观状态相关。在积极方面，我们表明当宏观干预的分布与观测分布中微观状态的分布相同时，因果关系可以进行聚合，并讨论了此观察的概括。

    In aggregated variables the impact of interventions is typically ill-defined because different micro-realizations of the same macro-intervention can result in different changes of downstream macro-variables. We show that this ill-definedness of causality on aggregated variables can turn unconfounded causal relations into confounded ones and vice versa, depending on the respective micro-realization. We argue that it is practically infeasible to only use aggregated causal systems when we are free from this ill-definedness. Instead, we need to accept that macro causal relations are typically defined only with reference to the micro states. On the positive side, we show that cause-effect relations can be aggregated when the macro interventions are such that the distribution of micro states is the same as in the observational distribution and also discuss generalizations of this observation.
    
[^50]: 不精确的贝叶斯神经网络

    Imprecise Bayesian Neural Networks. (arXiv:2302.09656v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09656](http://arxiv.org/abs/2302.09656)

    在机器学习和人工智能领域，该论文提出了一种新的算法——不精确的贝叶斯神经网络(IBNNs)。这种算法使用可信区间先验分布集合和似然分布集合进行训练，相比标准的BNNs，可以区分先验和后验的不确定性并量化。此外，IBNNs在贝叶斯灵敏度分析方面具有更强的鲁棒性，并且对分布变化也更加鲁棒。

    

    在机器学习和人工智能中, 确定不确定性和鲁棒性是重要的目标。虽然贝叶斯神经网络使得预测中的不确定性能够被评估，不同来源的不确定性是无法区分的。我们提出了不精确的贝叶斯神经网络（IBNNs），它们可以概括和克服标准BNNs的某些缺点。标准BNNs使用单一的先验分布和似然分布进行训练，而IBNNs使用可信区间先验分布和似然分布进行训练。它们允许区分先验和后验不确定性，并对其进行量化。此外，IBNNs在贝叶斯灵敏度分析方面具有鲁棒性，并且对分布变化比标准BNNs更加鲁棒。它们还可以用于计算具有PAC样本复杂性的结果集。我们将IBNNs应用于两个案例研究：一个是为了人工胰腺控制模拟血糖和胰岛素动力学，另一个是运动规划。

    Uncertainty quantification and robustness to distribution shifts are important goals in machine learning and artificial intelligence. Although Bayesian neural networks (BNNs) allow for uncertainty in the predictions to be assessed, different sources of uncertainty are indistinguishable. We present imprecise Bayesian neural networks (IBNNs); they generalize and overcome some of the drawbacks of standard BNNs. These latter are trained using a single prior and likelihood distributions, whereas IBNNs are trained using credal prior and likelihood sets. They allow to distinguish between aleatoric and epistemic uncertainties, and to quantify them. In addition, IBNNs are robust in the sense of Bayesian sensitivity analysis, and are more robust than BNNs to distribution shift. They can also be used to compute sets of outcomes that enjoy PAC-like properties. We apply IBNNs to two case studies. One, to model blood glucose and insulin dynamics for artificial pancreas control, and two, for motion p
    

