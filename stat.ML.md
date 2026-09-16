# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Coupled Calibration and Learning: Mitigating Teacher Bias in LLM Distillation without Target-Domain Reward Feedback](https://arxiv.org/abs/2609.17474) | 提出耦合校准与学习（CCL）算法，通过token级分支将教师校准与学生更新相互耦合、仅利用源域奖励反馈，从而在无目标域奖励反馈的条件下缓解LLM蒸馏中的教师偏差迁移问题。 |
| [^2] | [Hybrid Variational Quantum Circuits for Multivariate Regression and High-Dimensional Data Reconstruction](https://arxiv.org/abs/2609.17358) | 该论文提出混合变分量子线路（HVQC），通过在量子线路后添加经典仿射层，实现了无需独立标量线路开销的向量值回归与高维数据重构，其性能媲美高斯过程回归并超越XGBoost和随机森林。 |
| [^3] | [Conformal Policy Learning with Distribution-Free Safety Guarantees](https://arxiv.org/abs/2609.17296) | 本文提出共形策略学习（CPL），通过将每个处理决策视为反事实伤害假设检验并利用共形p值阈值化分配处理，首次实现了控制“对会受伤害个体分配处理”概率的无分布安全保证。 |
| [^4] | [Personalized Federated Learning through Global Knowledge Distillation and Local Head Adaptation](https://arxiv.org/abs/2609.17284) | 提出pFedKDH方法，通过仅聚合共享骨干网络、保留客户端专属分类头并利用重新校准的全局分类头作为蒸馏教师，在标签偏斜的个性化联邦学习场景中显著提升准确率。 |
| [^5] | [Covariate Selection for Doubly Robust Double/debiased Machine Learning Estimators for Causal Inference](https://arxiv.org/abs/2609.17238) | 该论文提出将倾向得分模型与结局机器学习模型各自选定的协变量取并集来重新估计两个模型，以保障双重稳健性质的实际效用，从而更有效地减少混杂偏倚。 |
| [^6] | [Splitting the Difference: Interpretable Causal Forests for Treatment Effect Heterogeneity and Bias](https://arxiv.org/abs/2609.16971) | 本文提出一种基于决策树和随机森林的可解释因果森林算法，仅通过修改分裂准则即可准确估计个体处理效应并揭示其异质性原因，无需双重机器学习或正交化等额外复杂步骤。 |
| [^7] | [Learning Choice Model Trees for Feature-Based Multi-Product Pricing: Exact Optimization and Field Evidence](https://arxiv.org/abs/2609.16952) | 该论文提出具有多项Logit叶节点的最优选择模型树（OCMT-MNL），通过精确动态规划联合优化树结构与叶节点需求模型，利用闭式Fenchel下界剪枝避免冗余计算，实现高达7.15倍的加速，并为基于特征的多产品定价提供了实地验证。 |
| [^8] | [Causal Discovery via Transformed Low-Rank Quantile Surfaces](https://arxiv.org/abs/2609.16931) | 本文提出低秩分位数曲面（LRQS）二元因果模型，证明在因果方向上变换后的条件分位数曲面具有低秩结构从而保证因果方向的通用可识别性，并提出了一种交替进行秩约束近似与单调变换保序估计的非参数因果发现方法。 |
| [^9] | [HyCoSeq: Contextual Hyperbolic Representation Learning for Genomic Sequences](https://arxiv.org/abs/2609.16925) | HyCoSeq通过将加权洛伦兹残差聚合融入多曲率洛伦兹编码，并引入双向LSTM学习序列上下文关系，将局部双曲卷积编码扩展为序列级的上下文化基因组表示。 |
| [^10] | [On the disintegration of the stochastic majority vote: From PAC-Bayesian bounds to a self-bounding algorithm](https://arxiv.org/abs/2609.16803) | 本文提出了一种去随机化框架，将解积PAC-贝叶斯理论直接应用于多数投票权重向量空间，把随机多数投票的泛化保证转化为单一确定性多数投票的证书，并由此推导出两族高概率泛化界和一种自约束学习算法。 |
| [^11] | [Time-warping estimation via stationarity-based learning of the de-warped signal](https://arxiv.org/abs/2609.16796) | 本文提出可训练的时间扭曲估计模型TWET，将时间扭曲估计转化为小波域的平稳化问题，通过可微平稳性准则和分层膨胀卷积架构实现端到端优化，在提高形变重建精度的同时大幅减少计算时间。 |
| [^12] | [Characterizing Heterogeneous Rates in Finite Mixture Estimation via Partial Optimal Transport](https://arxiv.org/abs/2609.16622) | 本文提出基于Voronoi的部分最优输运（VPOT）框架，突破了传统Wasserstein距离分析只给出最坏情况速率的局限，能够精细刻画有限混合模型极大似然估计的局部与全局异质收敛速率。 |
| [^13] | [Stable by Construction: Variational Latent Markov Operators for Long-Horizon PDE Prediction](https://arxiv.org/abs/2609.16621) | 提出变分自编码马尔可夫算子（VAMO），通过函数空间上的潜马尔可夫动力学、谱几何结构与变分转移对齐来正则化自回归误差传播，实现稳定的长时程偏微分方程预测。 |
| [^14] | [Supervising the Chain Ladder](https://arxiv.org/abs/2609.16552) | 本文将链梯法准备金进展模式的选择建模为监督学习问题，通过在严格凸的目标函数上添加可解释的惩罚项与超参数（如数据衰减、权重幂、基准参考与平滑约束），把精算师的专业判断形式化，并可通过单一线性系统求解。 |
| [^15] | [Causal Path Analysis from Perturbational and Population-Scale Single-Cell Data with Multiscale Confounding and Measurement Error](https://arxiv.org/abs/2609.16510) | 该论文提出了一个整合单细胞扰动实验与群体规模单细胞数据的因果通路分析框架，利用多尺度替代变量方法和变量误差校正来处理混杂因素与测量误差，并为高维估计提供了理论保证。 |
| [^16] | [Certified Inference and Training for Deep Equilibrium Networks: A Continuation Framework with Polynomial Complexity Guarantees](https://arxiv.org/abs/2609.16485) | 本文提出了一种认证延拓框架，将深度均衡网络的训练表述为精度插值问题，使推理和训练都能在多项式复杂度预算下获得可认证的保证。 |
| [^17] | [Learned Look-Ahead Splitting Rule for CART](https://arxiv.org/abs/2609.16440) | 该论文提出一种通过在候选分裂点下方生长CART子树来评估分裂质量的前瞻分裂规则，并利用节点级特征学习的智能前瞻算法大幅降低计算成本，在保持决策树可解释性的同时显著改善层级或交互场景下的分裂选择。 |
| [^18] | [Bounded Adjustment with Reliability-Guided Embedding for Imbalanced Learning with Noisy Labels](https://arxiv.org/abs/2609.16380) | 该论文提出BARGE方法，将有界先验调整密度幂得分与可靠性引导嵌入结合为单阶段目标函数，在应对类别不平衡的同时约束噪声标签带来的分类风险扰动，并在模型与标签高度冲突时自动衰减梯度以抵抗标签噪声。 |
| [^19] | [Mini-batch Sampling Strategies for Long-Tailed Image Classification: An Empirical Study on CIFAR-100-LT](https://arxiv.org/abs/2609.16365) | 本文在统一的偏差-方差框架下系统比较了均匀实例采样、类平衡采样、平方根采样和渐进平衡采样四种小批量采样策略对长尾图像分类中梯度估计的影响，并在CIFAR-100-LT数据集上进行了实证评估。 |
| [^20] | [Compute-Optimal Pretrain--Fine-tune in Ridge Gradient Descent](https://arxiv.org/abs/2609.16262) | 本文在岭回归梯度下降的两阶段预训练—微调框架下，首次从理论上刻画了固定总优化预算时上游预训练与下游微调之间的最优计算分配，并揭示该分配由预测相关的谱分量和下游数据几何共同决定。 |
| [^21] | [Copula Adapted Directed Acyclic Graph for Cluster Representation of Biomedical Data](https://arxiv.org/abs/2609.16240) | 本文提出了一种融合Copula非高斯非线性依赖建模与基于有向无环图的集成因果结构发现方法的新型数据表示框架，用于无标签高维生物医学数据的聚类表示。 |
| [^22] | [Skeletal Prototypes on Iterative Nerve Expansions](https://arxiv.org/abs/2609.16170) | SPINE方法创新性地用嵌入的一维复形（骨架结构）而非传统点集来表示各类原型，通过类条件Mapper图构建初始边集并在分类目标下优化顶点位置，使骨架线段直接参与决策规则，在17个基准数据集上取得了最优的平均准确率和排名。 |
| [^23] | [Stochastic Gradient Descent over P2](https://arxiv.org/abs/2609.13343) | 该论文将经典欧氏空间中SGD的扩散（高斯）近似理论首次推广到Wasserstein空间P2上的优化问题，通过Lions可微性将问题提升至线性希尔伯特空间，并构造了与随机梯度矩信息相匹配的高斯随机场近似。 |
| [^24] | [ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing](https://arxiv.org/abs/2608.26083) | ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。 |
| [^25] | [Random Hazard Forests](https://arxiv.org/abs/2608.21597) | 随机风险森林通过非参数风险似然和连续时间树集成，直接处理不规则、多源临床数据，实现动态更新的个体化风险预测。 |
| [^26] | [CausalSmith: A Formally Grounded, Self-Improving Agentic Framework for Automated Research in Causal Inference](https://arxiv.org/abs/2607.22511) | CausalSmith通过结合Lean证明助手和自改进代理管道，解决了LLM评审员不可靠的问题，实现了因果推断领域自动化理论研究中可验证、可靠的结果生成与评估。 |
| [^27] | [Missing Data Imputation under Manifold Hypothesis](https://arxiv.org/abs/2607.03641) | 本文基于流形假设与混合变分自编码器，提出了一种通过SIR采样和潜空间扩散模型从条件分布中采样的缺失数据插补方法，在尊重数据几何结构的同时实现高质量插补并量化不确定性。 |
| [^28] | [Information-Theoretic Bounds for Sparse Covariance Estimation in the Vertical-Split Distributed Model](https://arxiv.org/abs/2606.07124) | 该论文首次证明在纵向切分分布式设置中，对互协方差矩阵施加稀疏性约束能够有效降低通信和样本复杂度，这与水平切分设置下稀疏性无法降低通信成本的结论形成鲜明对比。 |
| [^29] | [BASIS: Batchwise Advantage Estimation from Single-Rollout Information Sharing for LLM Reasoning](https://arxiv.org/abs/2605.27293) | BASIS通过在批次内共享单rollout的跨提示信息来改进价值函数估计，以显著更低的计算成本实现了接近多rollout方法的策略优化性能。 |
| [^30] | [Universal Feature Selection with Noisy Observations and Weak Symmetry Conditions](https://arxiv.org/abs/2605.09396) | 本文提出在弱球对称性条件下基于噪声数据的通用特征选择框架，通过典型相关依存矩阵的奇异值分解实现渐近最优误差指数，并证明精确的球对称性条件并非必要。 |
| [^31] | [Equivalence of approximation by networks of single- and multi-spike neurons](https://arxiv.org/abs/2603.13478) | 本文证明对于包括泄漏积分发放模型在内的一大类脉冲神经元模型，单脉冲网络与多脉冲网络在函数逼近能力上完全等价，两者之间仅需以神经元数量的线性倍数进行转换。 |
| [^32] | [Statistical Inference for Score Decompositions](https://arxiv.org/abs/2603.04275) | 该论文提出了基于预测线性重校准的评分分解统计推断方法，将预测评分分解为误校准、判别力与不确定性三个可解释成分，适用于非光滑评分函数并支持模型误设下的渐近推断。 |
| [^33] | [Nonnegative matrix factorizations and related compositional models: Equivalence, identifiability, and an application on the grain-size analysis of sediments](https://arxiv.org/abs/2512.22282) | 本文证明了来自社会科学、地质学和机器学习的五种模型（LBA、LCA、EMA、PLSA、NMF）在本质上等价，NMF的解唯一性定理可直接推广到其他四种模型，并将其应用于沉积物粒度分析。 |
| [^34] | [Training Energy-Based Models with Non-MCMC Samplers and Efficient Temperature Estimation](https://arxiv.org/abs/2512.02323) | 该论文提出了朗之万模拟分岔（LSB）快速并行玻尔兹曼采样器、条件期望匹配（CEM）高效温度估计方法以及采样器自适应学习（SAL）框架，从而无需MCMC即可高效训练能量基模型。 |
| [^35] | [A proximal augmented Lagrangian method for nonconvex optimization with equality and inequality constraints](https://arxiv.org/abs/2509.02894) | 本文提出一种具有罚参数与近端项自适应更新规则的非精确近端增广拉格朗日方法（P-ALM），通过证明增广拉格朗日函数沿迭代点的可控性，为非凸约束优化建立了新的收敛理论，并对经典ALM也得出了类似的收敛性质。 |
| [^36] | [Neural Stochastic Differential Equations on Compact State Spaces: Theory, Methods, and Application to Suicide Risk Modeling](https://arxiv.org/abs/2508.17090) | 本文提出了一类新型神经随机微分方程，其解可被严格证明限制在指定的紧凑多面体状态空间内，克服了现有SDE模型违反定义域约束和数值不稳定的问题，并成功应用于自杀风险建模。 |
| [^37] | [Kinetic Interacting Particle Langevin Monte Carlo](https://arxiv.org/abs/2407.05790) | 本文提出了动力学交互粒子朗之万蒙特卡洛（KIPLMC）方法，通过参数与潜变量联合演化的扩散过程实现潜变量模型的统计推断，并在强凹条件下获得了具有加速收敛速率和更优维度依赖性的Wasserstein-2距离非渐近收敛保证。 |

# 详细

[^1]: 耦合校准与学习：在无目标域奖励反馈的LLM蒸馏中缓解教师偏差

    Coupled Calibration and Learning: Mitigating Teacher Bias in LLM Distillation without Target-Domain Reward Feedback

    [https://arxiv.org/abs/2609.17474](https://arxiv.org/abs/2609.17474)

    提出耦合校准与学习（CCL）算法，通过token级分支将教师校准与学生更新相互耦合、仅利用源域奖励反馈，从而在无目标域奖励反馈的条件下缓解LLM蒸馏中的教师偏差迁移问题。

    

    大语言模型（LLM）蒸馏旨在将强大教师模型的能力迁移到更小的学生模型中。然而，直接模仿也会将教师的系统性偏差和错误一同迁移。这一挑战在协变量偏移情况下尤为突出，即教师对目标问题的可靠性不确定且目标域奖励反馈不可用时。我们提出了耦合校准与学习（CCL），这是一种LLM蒸馏算法，通过token级分支将教师校准与学生更新耦合起来，仅在源问题上使用奖励反馈。每次迭代中，先利用源域反馈校准教师，再用校准后的教师在目标问题上训练学生；更新后的学生反过来又为后续的校准提供信息。在自回归策略框架下，我们证明了输出的学生模型相对于oracle学生模型的期望平均Kullback-Leibler散度收敛

    arXiv:2609.17474v1 Announce Type: cross  Abstract: Large language model (LLM) distillation aims to transfer the capabilities of a powerful teacher to a smaller student. Direct imitation, however, can also transfer the teacher's systematic bias and errors. This challenge is particularly pronounced under covariate shift, when the teacher's reliability on target questions is uncertain and target-domain reward feedback is unavailable. We propose Coupled Calibration and Learning (CCL), an LLM distillation algorithm that couples teacher calibration with student updates through token-level branching, using reward feedback only on source questions. Each iteration calibrates the teacher using source feedback and then uses the calibrated teacher to train the student on target questions. The updated student, in turn, informs subsequent calibration. In an autoregressive policy framework, we prove that the output student's expected average Kullback-Leibler divergence to the oracle student converges
    
[^2]: 用于多元回归与高维数据重构的混合变分量子线路

    Hybrid Variational Quantum Circuits for Multivariate Regression and High-Dimensional Data Reconstruction

    [https://arxiv.org/abs/2609.17358](https://arxiv.org/abs/2609.17358)

    该论文提出混合变分量子线路（HVQC），通过在量子线路后添加经典仿射层，实现了无需独立标量线路开销的向量值回归与高维数据重构，其性能媲美高斯过程回归并超越XGBoost和随机森林。

    

    变分量子线路（VQC）是一类通过经典方法优化的参数化量子线路。我们提出了一种混合变分量子线路（HVQC），在VQC基础上增加了一个经典的测量后仿射层，从而实现向量值回归，避免了使用多个独立标量线路所带来的线性开销。在理论方面，我们证明了基本的单量子比特和双量子比特线路可以通过数据重复上传和纠缠来近似二次函数和乘积运算，为完整架构奠定了基础。在实验方面，在两个合成图像重构数据集和Friedman1基准数据集（40,568个测试样本）上，我们的HVQC达到了与高斯过程回归相当的性能，并优于XGBoost和随机森林。消融实验证实了量子组件和经典组件都不可或缺，结果还凸显了特征映射在混合量子-经典模型中的核心作用。

    arXiv:2609.17358v1 Announce Type: new  Abstract: Variational quantum circuits (VQCs) are parameterized quantum circuits optimized classically. We propose a hybrid variational quantum circuit (HVQC) extending VQCs with a classical affine post-measurement layer, enabling vector-valued regression without the linear overhead of independent scalar circuits. Theoretically, we show that elementary one-and two-qubit circuits can approximate quadratic functions and products via data re-uploading and entanglement, providing the foundations of the full architecture. Experimentally, on two synthetic image reconstruction datasets and the Friedman1 benchmark (40,568 test samples), our HVQC matches Gaussian Process Regression and outperforms XGBoost and Random Forest. An ablation study confirms that both quantum and classical components are essential, and results highlight the central role of the feature map in hybrid quantum-classical models.
    
[^3]: 带有无分布安全保证的共形策略学习

    Conformal Policy Learning with Distribution-Free Safety Guarantees

    [https://arxiv.org/abs/2609.17296](https://arxiv.org/abs/2609.17296)

    本文提出共形策略学习（CPL），通过将每个处理决策视为反事实伤害假设检验并利用共形p值阈值化分配处理，首次实现了控制“对会受伤害个体分配处理”概率的无分布安全保证。

    

    策略学习旨在基于个体特征决定谁应该接受处理。在医学和公共政策等以安全为核心关切的高风险场景中，仅仅改善平均结果可能是不够的：决策者还可能希望保护个体免受伤害，这符合“不伤害”的希波克拉底原则。本文提出了共形策略学习（CPL），这是一种带有新型无分布安全保证的策略学习程序，该保证控制了将处理分配给相对于对照组会受到伤害的个体的概率。CPL将每个处理决策视为对反事实伤害假设的检验，并通过共形p值的阈值化来分配处理。这些p值利用可观测的代理变量和选择性校准，解决了所比较的潜在结果永远不会同时被观测到的挑战。对于随机实验…

    arXiv:2609.17296v1 Announce Type: cross  Abstract: Policy learning aims to determine who should be treated based on individual characteristics. In high-stakes settings such as medicine and public policy where safety is a central concern, improving the average outcomes alone may not be sufficient: decision makers may also seek to protect individuals from harm, in line with the Hippocratic principle of ``do no harm.'' In this paper, we propose \textit{conformal policy learning} (CPL), a policy learning procedure with a new distribution-free safety guarantee that controls the probability of assigning treatment to an individual who would be harmed relative to control. CPL views each treatment decision as testing a hypothesis of counterfactual harm and assigns treatment by thresholding conformal p-values. These p-values use observable proxies and selective calibration to address the challenge that the potential outcomes under comparison are never simultaneously observed. For randomized expe
    
[^4]: 基于全局知识蒸馏与本地分类头自适应的个性化联邦学习

    Personalized Federated Learning through Global Knowledge Distillation and Local Head Adaptation

    [https://arxiv.org/abs/2609.17284](https://arxiv.org/abs/2609.17284)

    提出pFedKDH方法，通过仅聚合共享骨干网络、保留客户端专属分类头并利用重新校准的全局分类头作为蒸馏教师，在标签偏斜的个性化联邦学习场景中显著提升准确率。

    

    当单一全局分类器无法表示客户端特定的标签分布时，统计异质性会限制联邦学习的性能。在这项工作中，我们提出了带分类头自适应的个性化联邦知识蒸馏方法，该方法仅聚合共享的骨干网络，保留持久的客户端专属分类头，并在本地训练期间使用重新校准的全局分类头作为教师模型。在基于类别Dirichlet划分的MNIST、Fashion-MNIST、CIFAR10和CIFAR100数据集上，pFedKDH在大多数设置下取得了最佳准确率，相对于最弱基线的准确率差距高达37.67%，且在多次重复实验中始终保持较低的标准差。组件级诊断和收敛性结果验证了持久分类头以及蒸馏引导的局部优化在标签偏斜数据下的有效性。

    arXiv:2609.17284v1 Announce Type: new  Abstract: Statistical heterogeneity limits federated learning when a single global classifier cannot represent client-specific label distributions. In this work, we propose Personalized Federated Knowledge Distillation with Head Adaptation (pFedKDH), which aggregates only the shared backbone, keeps persistent client-specific heads, and uses a recalibrated global head as a teacher during local training. Across MNIST, Fashion-MNIST, CIFAR10, and CIFAR100 under class-wise Dirichlet partitions, pFedKDH obtains the best accuracy in most settings, with accuracy gaps up to 37.67\% over the weakest baseline and consistently low standard deviation across repetitions. Component-wise diagnostics and convergence results support the role of persistent heads and distillation-guided local optimization under label-skewed data.
    
[^5]: 因果推断中双重稳健双机器学习/去偏机器学习估计量的协变量选择

    Covariate Selection for Doubly Robust Double/debiased Machine Learning Estimators for Causal Inference

    [https://arxiv.org/abs/2609.17238](https://arxiv.org/abs/2609.17238)

    该论文提出将倾向得分模型与结局机器学习模型各自选定的协变量取并集来重新估计两个模型，以保障双重稳健性质的实际效用，从而更有效地减少混杂偏倚。

    

    高维数据给因果效应估计带来了挑战，因为识别出正确模型设定所需的协变量变得越来越困难。双机器学习/去偏机器学习（DML）通过缓解正则化偏差和过拟合偏差，促进了机器学习（ML）在因果推断中的应用，但对于某些DML估计量所具有的双重稳健（DR）性质，与之相关的协变量选择问题受到的关注相对较少。特别是，基于机器学习的协变量选择可能导致差异化协变量选择，或导致两个模型同时被错误设定，从而限制了DR性质的实际效用。为解决这些问题，我们提出使用倾向得分（PS）模型和结局机器学习模型各自选定的协变量的并集来重新估计这两个模型。仿真结果表明，使用该并集始终比分别使用各自选定的协变量能减少更多的混杂偏倚。

    arXiv:2609.17238v1 Announce Type: cross  Abstract: High-dimensional data create challenges for causal effect estimation because identifying the covariates needed for correct model specification becomes increasingly difficult. Double/debiased machine learning (DML) facilitates the use of machine learning (ML) for causal inference by mitigating regularization and overfitting bias, but comparatively less attention has been given to covariate selection in relation to the double robustness (DR) property possessed by some DML estimators. In particular, ML-based covariate selection may result in differential covariate selection or in misspecification of both models, thereby limiting the practical utility of the DR property. To address these issues, we propose using the union of the covariates selected by the propensity score (PS) and outcome ML models to re-estimate both models. Simulation results show that using the union consistently reduces more confounding bias than using separate selecte
    
[^6]: 巧妙分割：用于处理效应异质性与偏差的可解释因果森林

    Splitting the Difference: Interpretable Causal Forests for Treatment Effect Heterogeneity and Bias

    [https://arxiv.org/abs/2609.16971](https://arxiv.org/abs/2609.16971)

    本文提出一种基于决策树和随机森林的可解释因果森林算法，仅通过修改分裂准则即可准确估计个体处理效应并揭示其异质性原因，无需双重机器学习或正交化等额外复杂步骤。

    

    在医学和营销等各个领域，准确预测个体处理效应具有重要意义。然而，仅实现可靠的预测往往不足以做出明智的决策；同样重要的是理解为什么某些个体的处理效应高于其他个体。为了应对预测和解释这一双重挑战，我们提出了一种基于决策树和随机森林的算法，用于估计个体处理效应。我们的算法非常简单：它的运行方式与标准随机森林完全相同，只是采用了不同的分裂准则，并且不需要额外的变通方法，例如广义随机森林中使用的双重机器学习或正交化技术。该算法能够处理具有不同处理倾向的观察性研究，而无需单独估计完整的倾向函数。这是通过结合两种分裂……（原文在此处截断）

    arXiv:2609.16971v1 Announce Type: cross  Abstract: In various fields, such as medicine and marketing, accurately predicting individual treatment effects holds significant promise. However, achieving reliable predictions alone is often insufficient for making informed decisions; it is equally important to understand why the treatment effect is higher for some individuals than for others. To address this two-fold challenge of prediction and interpretation, we introduce an algorithm based on decision trees and random forests for estimating individual treatment effects. Our algorithm is simple: it operates exactly like a standard random forest, but with a different splitting criterion, and requires no additional workarounds such as double machine learning or orthogonalization as used in Generalized random forests. It handles observational studies with varying treatment propensities without requiring separate estimation of the full propensity function. This is achieved by combining two spli
    
[^7]: 面向基于特征的多产品定价的选择模型树学习：精确优化与实地证据

    Learning Choice Model Trees for Feature-Based Multi-Product Pricing: Exact Optimization and Field Evidence

    [https://arxiv.org/abs/2609.16952](https://arxiv.org/abs/2609.16952)

    该论文提出具有多项Logit叶节点的最优选择模型树（OCMT-MNL），通过精确动态规划联合优化树结构与叶节点需求模型，利用闭式Fenchel下界剪枝避免冗余计算，实现高达7.15倍的加速，并为基于特征的多产品定价提供了实地验证。

    

    基于特征的多产品定价利用客户特征来识别需求异质性，并据此针对不同产品制定差异化价格。选择模型树通过可解释的特征规则对客户进行细分，并在每个叶节点内拟合需求模型。现有方法通常以贪心方式构建这些树，每次仅选择一个短视的分裂点。我们提出了具有多项Logit（MNL）叶节点的最优选择模型树（OCMT-MNL），在给定深度约束下对树结构和叶节点模型进行联合优化。我们的精确动态规划方法在受约束的牛顿迭代过程中推导出闭式Fenchel下界，并将这些下界传播至嵌套且互不相交的客户子集上，从而避免新的拟合过程，并能在不重复已完成工作的情况下恢复未完成的拟合。在合成实验中，该方法将精确叶节点拟合次数减少了99.98%，叶节点评估次数减少了86.13%，相比未剪枝的动态规划实现了高达7.15倍的加速。一维查找表将学习到的客户细分转（化为可部署的定价策略……）

    arXiv:2609.16952v1 Announce Type: cross  Abstract: Feature-based multi-product pricing uses customer characteristics to identify demand heterogeneity and tailor prices across products. Choice model trees segment customers through interpretable feature rules and fit a demand model within each leaf. Existing methods typically construct these trees greedily, selecting one myopic split at a time. We develop optimal choice model trees with multinomial logit leaves (OCMT-MNL), jointly optimizing the tree and leaf models within a prescribed depth. Our exact dynamic program derives closed-form Fenchel lower bounds during constrained Newton iterations and propagates them across nested and disjoint customer subsets, avoiding new fits and resuming unfinished fits without repeating completed work. In synthetic experiments, it reduces exact leaf fits by 99.98% and leaf evaluations by 86.13%, achieving up to 7.15-fold speedups over unpruned dynamic programming. One-dimensional lookup tables translat
    
[^8]: 基于变换低秩分位数曲面的因果发现

    Causal Discovery via Transformed Low-Rank Quantile Surfaces

    [https://arxiv.org/abs/2609.16931](https://arxiv.org/abs/2609.16931)

    本文提出低秩分位数曲面（LRQS）二元因果模型，证明在因果方向上变换后的条件分位数曲面具有低秩结构从而保证因果方向的通用可识别性，并提出了一种交替进行秩约束近似与单调变换保序估计的非参数因果发现方法。

    

    我们提出了低秩分位数曲面（LRQS），这是一种二元因果模型，在因果方向上，条件分位数曲面的未知单调变换具有低秩函数分解形式。LRQS 涵盖了位置-尺度噪声模型和后非线性异方差噪声模型，同时允许多个分位数基函数来表示超出位置-尺度效应的分布变化。我们证明了 LRQS 的通用可识别性：变换后的分位数曲面在因果方向上是低秩的，而在相应约束下，反方向的可表示性仅出现在例外的、经过精细调整的原因边缘分布中。我们提供了一个简单而强大的因果评分方法，采用非参数拟合程序，在离散化分位数曲面的秩约束近似与未知单调变换的保序估计之间交替进行。在具有更高秩分布变化的合成机制上的实验表……（原文摘要在此处被截断）

    arXiv:2609.16931v1 Announce Type: cross  Abstract: We propose Low-Rank Quantile Surfaces (LRQS), a bivariate causal model in which, in the causal direction, an unknown monotone transformation of the conditional quantile surface admits a low-rank functional decomposition. LRQS subsumes location-scale noise models and post-nonlinear heteroscedastic noise models, while allowing multiple quantile bases to represent changes beyond location-scale effects. We prove generic identifiability of LRQS: the transformed quantile surface is low rank in the causal direction, whereas reverse representability under the corresponding constraints occurs only for exceptional, fine-tuned cause marginals. We provide a simple-yet-powerful causal score using a nonparametric fitting procedure that alternates between rank-constrained approximation of discretized quantile surfaces and isotonic estimation of the unknown monotone transformation. Experiments on synthetic mechanisms with higher-rank distributional sh
    
[^9]: HyCoSeq：面向基因组序列的上下文双曲表示学习

    HyCoSeq: Contextual Hyperbolic Representation Learning for Genomic Sequences

    [https://arxiv.org/abs/2609.16925](https://arxiv.org/abs/2609.16925)

    HyCoSeq通过将加权洛伦兹残差聚合融入多曲率洛伦兹编码，并引入双向LSTM学习序列上下文关系，将局部双曲卷积编码扩展为序列级的上下文化基因组表示。

    

    双曲几何为基因组表示学习提供了一种天然的归纳偏置，但现有的双曲基因组模型主要使用洛伦兹卷积来学习局部序列表示，而其残差通路并未直接聚合完整的洛伦兹表示。我们提出了HyCoSeq，一个面向基因组序列的上下文双曲表示学习框架。HyCoSeq将加权洛伦兹残差聚合融入多曲率洛伦兹编码中，使完整的洛伦兹表示能够直接参与几何一致的局部聚合。它进一步引入了双向长短期记忆网络，整合来自序列两个方向的信息，以学习基因组序列中不同位置的局部表示之间的上下文关系，从而将局部双曲卷积编码扩展为序列级别的上下文化表示。

    arXiv:2609.16925v1 Announce Type: new  Abstract: Hyperbolic geometry provides a natural inductive bias for genomic representation learning, but existing hyperbolic genomic models primarily use Lorentz convolutions to learn local sequence representations, while their residual pathways do not directly aggregate full Lorentz representations. We propose HyCoSeq, a contextual hyperbolic representation learning framework for genomic sequences. HyCoSeq incorporates weighted Lorentzian residual aggregation into multi-curvature Lorentz encoding, allowing full Lorentz representations to participate directly in geometry-consistent local aggregation. It further introduces a bidirectional long short-term memory network that integrates information from both sequence directions to learn contextual relationships among local representations at different positions within a genomic sequence, thereby extending local hyperbolic convolutional encoding to sequence-level contextualized representations. Extens
    
[^10]: 论随机多数投票的解体：从PAC-贝叶斯界到自约束算法

    On the disintegration of the stochastic majority vote: From PAC-Bayesian bounds to a self-bounding algorithm

    [https://arxiv.org/abs/2609.16803](https://arxiv.org/abs/2609.16803)

    本文提出了一种去随机化框架，将解积PAC-贝叶斯理论直接应用于多数投票权重向量空间，把随机多数投票的泛化保证转化为单一确定性多数投票的证书，并由此推导出两族高概率泛化界和一种自约束学习算法。

    

    加权多数投票是许多成功集成方法的核心。PAC-贝叶斯理论通过分析随机分类器的期望风险，为此类模型提供了紧密的泛化保证，而分析确定性多数投票的风险则依赖于替代界。为了避免这些替代方法，Zantedeschi等人（2021）引入了针对随机多数投票的保证，但由此产生的模型仍然是随机化的。在本文中，我们提出了一个针对随机多数投票的去随机化框架。为此，我们将解积PAC-贝叶斯理论的最新进展直接应用于多数投票权重向量空间，将随机保证转化为单一确定性多数投票的证书。我们推导出了两族高概率泛化界，同时涵盖了集成的数据无关构造和数据依赖构造，这自然地引出了一种自约束学习算法。

    arXiv:2609.16803v1 Announce Type: cross  Abstract: Weighted majority votes are central to many successful ensemble methods. PAC-Bayesian theory provides tight generalization guarantees for such models by analyzing the expected risk of stochastic classifiers, while analyzing the risk of deterministic majority votes relies on surrogate bounds. To avoid these surrogates, Zantedeschi et al. ( 2021) introduced guarantees for stochastic majority votes, but the resulting models remain randomized. In this paper, we propose a derandomization framework for stochastic majority votes. To do so, we apply recent advances in disintegrated PAC-Bayesian theory directly to the space of majority vote weight vectors, transforming stochastic guarantees into certificates for a single deterministic majority vote. We derive two families of high-probability generalization bounds, covering both data-independent and data-dependent constructions of the ensemble, which naturally lead to a self-bounding learning al
    
[^11]: 基于去时变信号平稳性学习的时间扭曲估计

    Time-warping estimation via stationarity-based learning of the de-warped signal

    [https://arxiv.org/abs/2609.16796](https://arxiv.org/abs/2609.16796)

    本文提出可训练的时间扭曲估计模型TWET，将时间扭曲估计转化为小波域的平稳化问题，通过可微平稳性准则和分层膨胀卷积架构实现端到端优化，在提高形变重建精度的同时大幅减少计算时间。

    

    时间扭曲估计是信号处理中的一个基本问题，在生物声学、雷达和生物医学分析等领域有广泛应用。本文介绍了一种可训练的时间扭曲估计模型（Time-Warping Estimation Trainable，TWET），用于从单次观测中估计时间扭曲函数。所提出的方法将时间扭曲估计表述为小波域中的平稳化问题，并利用分层膨胀卷积架构来估计时间扭曲函数。为实现端到端优化，本文引入了一种可微的平稳性准则。论文将TWET与现有方法进行了比较，实验结果表明，该方法的形变重建精度有所提高，同时计算时间显著减少，使该框架能够兼容低延迟应用。

    arXiv:2609.16796v1 Announce Type: cross  Abstract: Time-warping estimation is a fundamental problem in signal processing with applications in bioacoustics, radar, and biomedical analysis. This paper introduces a Time-Warping Estimation Trainable (TWET) model for estimating timewarping functions from a single observation. The proposed approach formulates time-warping estimation as a stationarization problem in the wavelet domain and leverages a hierarchical dilated convolutional architecture to estimate the time-warping functions. A differentiable stationarity criterion is introduced for end-to-end optimization. TWET is compared with existing approaches. Experimental results show improved deformation reconstruction accuracy together with significantly reduced computation time, making the framework compatible with low-latency applications.
    
[^12]: 通过部分最优输运刻画有限混合估计中的异质收敛速率

    Characterizing Heterogeneous Rates in Finite Mixture Estimation via Partial Optimal Transport

    [https://arxiv.org/abs/2609.16622](https://arxiv.org/abs/2609.16622)

    本文提出基于Voronoi的部分最优输运（VPOT）框架，突破了传统Wasserstein距离分析只给出最坏情况速率的局限，能够精细刻画有限混合模型极大似然估计的局部与全局异质收敛速率。

    

    有限混合模型中的参数估计可能表现出高度异质的收敛行为：局部孤立的成分的估计速度可能远快于相互竞争的成分组。现有的基于Wasserstein距离的分析通常仅刻画最坏情况下的收敛速率，因此无法完全捕捉这种局部异质性。在本文中，我们提出了一个基于Voronoi的部分最优输运（VPOT）框架，用于为混合测度的极大似然估计量获得精细的局部和全局收敛保证。关键的几何思想是将两个混合测度的比较局部化到扩展的Voronoi邻域中，并利用部分最优输运来适应其局部限制的不等质量。在每个邻域内，一阶POT差异被提升到由局部竞争原子数量所确定的幂次，使得所得的损失能够适应……

    arXiv:2609.16622v1 Announce Type: cross  Abstract: Parameter estimation in finite mixture models can exhibit highly heterogeneous convergence behavior: locally isolated components may be estimated substantially faster than groups of competing components. Existing analyses based on Wasserstein distances typically characterize only the worst-case rate and therefore do not fully capture this local heterogeneity. In this paper, we introduce a Voronoi-based partial optimal transport (VPOT) framework for obtaining refined local and global convergence guarantees for the maximum likelihood estimator of the mixing measure. The key geometric idea is to localize the comparison of two mixing measures to extended Voronoi neighborhoods and use partial optimal transport to accommodate the unequal masses of their local restrictions. Within each neighborhood, the first-order POT discrepancy is raised to a power determined by the number of locally competing atoms, allowing the resulting loss to adapt to
    
[^13]: 构造即稳定：用于长时程偏微分方程预测的变分潜马尔可夫算子

    Stable by Construction: Variational Latent Markov Operators for Long-Horizon PDE Prediction

    [https://arxiv.org/abs/2609.16621](https://arxiv.org/abs/2609.16621)

    提出变分自编码马尔可夫算子（VAMO），通过函数空间上的潜马尔可夫动力学、谱几何结构与变分转移对齐来正则化自回归误差传播，实现稳定的长时程偏微分方程预测。

    

    神经偏微分方程求解器为时变物理系统提供了高效的代理模型，但在长时程上的自回归预测仍然具有挑战性，因为局部误差会引起分布偏移，并在递归部署下不断累积。我们针对这一问题开发了一种变分方法，通过引入潜马尔可夫动力学，将物理状态表示为潜分布，并通过概率转移进行演化。该框架直接在函数空间上构建，并专门针对函数型高斯模型，其中结构化的潜扰动诱导出谱几何结构，变分转移对齐则对学习到的动力学进行正则化。我们进一步分析了这些机制如何影响自回归误差传播，为变分训练与长时程预测之间建立了理论联系。我们将该框架实例化为变分自编码马尔可夫算子（VAMO），

    arXiv:2609.16621v1 Announce Type: new  Abstract: Neural PDE solvers provide efficient surrogates for time-dependent physical systems, but autoregressive prediction over long horizons remains challenging because local errors can induce distribution shift and accumulate under recursive deployment. We develop a variational approach to this problem by introducing latent Markov dynamics in which physical states are represented by latent distributions and evolved through probabilistic transitions. The framework is formulated directly on function spaces and specialized to functional Gaussian models, where structured latent perturbations induce a spectral geometry and variational transition alignment regularizes the learned dynamics. We further analyze how these mechanisms affect autoregressive error propagation, providing a theoretical connection between variational training and long-horizon prediction. We instantiate the framework as the Variational Autoencoding Markov Operator (VAMO), which
    
[^14]: 监督链梯法

    Supervising the Chain Ladder

    [https://arxiv.org/abs/2609.16552](https://arxiv.org/abs/2609.16552)

    本文将链梯法准备金进展模式的选择建模为监督学习问题，通过在严格凸的目标函数上添加可解释的惩罚项与超参数（如数据衰减、权重幂、基准参考与平滑约束），把精算师的专业判断形式化，并可通过单一线性系统求解。

    

    链梯法的加权进展模式最小化一个显式的损失函数，但实务中很少直接照此入账。精算师通常会调整该模式并记录最终调整后的比率。本文将链梯法的进展模式选择视为一个监督学习问题。对模式调整的专业判断被转化为在链梯法损失函数上定义好的惩罚项和超参数框架，该损失函数在此被视为机器学习中的目标函数。数据权重通过引入衰减参数和幂参数进行推广，分别用于近期性和加权控制。基准形态和平滑性约束则通过参考惩罚项和Whittaker-Henderson平滑方法引入。所构建的目标函数是严格凸的，可通过求解一个线性系统得到最小值。每个超参数本身都是一种可解释的调整方式，可由专业判断予以声明，并归类为经验调整或前瞻性调整。经验调整可以被设置得更加客观。

    arXiv:2609.16552v1 Announce Type: cross  Abstract: The chain ladder's volume-weighted pattern minimises an explicit loss function, yet is rarely booked as such. Practitioners adjust the pattern and record the final adjusted ratios. This paper treats the chain ladder's pattern selection as a supervised-learning problem. Judgement on pattern adjustments becomes a framework of defined penalties and hyperparameters on the chain ladder's loss function, treated here as an objective function in machine learning. Data weights are generalised with a decay and a power parameter for recency and volume weighting. Benchmark shaping and smoothness enter through a reference penalty and Whittaker-Henderson smoothing. The assembled objective is strictly convex and minimised by a single linear system. Each hyperparameter becomes an interpretable adjustment in its own right, declarable by judgement and categorised as an experience or a prospective adjustment. Experience adjustments can be set more object
    
[^15]: 基于扰动实验与群体规模单细胞数据的多尺度混杂与测量误差下的因果通路分析

    Causal Path Analysis from Perturbational and Population-Scale Single-Cell Data with Multiscale Confounding and Measurement Error

    [https://arxiv.org/abs/2609.16510](https://arxiv.org/abs/2609.16510)

    该论文提出了一个整合单细胞扰动实验与群体规模单细胞数据的因果通路分析框架，利用多尺度替代变量方法和变量误差校正来处理混杂因素与测量误差，并为高维估计提供了理论保证。

    

    单细胞扰动实验提供了基因调控的因果信息，而群体规模的单细胞研究则刻画了人类群体中的基因表达与表型。我们开发了一个整合这两种互补数据源进行因果通路分析的框架。我们并不假设扰动实验得到的基因网络可以直接迁移到目标群体，而是利用外部学习到的祖先关系来约束网络拓扑结构，并从群体数据中重新估计网络的直接边及其效应。为解决多尺度单细胞测量中的潜在异质性和测量误差问题，我们开发了一种在细胞和受试者两个层面运行的替代变量方法，并结合变量误差校正用于网络回归和结局回归。我们为混杂因素的恢复以及网络和基因-结局效应的高维估计建立了理论保证。模拟实验

    arXiv:2609.16510v1 Announce Type: cross  Abstract: Single-cell perturbation experiments provide causal information on gene regulation, whereas population-scale single-cell studies characterize gene expression and phenotypes in human populations. We develop a framework that integrates these complementary data sources for causal path analysis. Rather than assuming that a perturbational gene network transfers directly to the target population, we use externally learned ancestral relationships to constrain the network topology and re-estimate its direct edges and effects from population data. To address latent heterogeneity and measurement error in multiscale single-cell measurements, we develop a surrogate-variable procedure operating at both the cell and subject levels, combined with errors-in-variables correction for network and outcome regressions. We establish theoretical guarantees for confounder recovery and high-dimensional estimation of network and gene-outcome effects. Simulation
    
[^16]: 深度均衡网络的认证推理与训练：具有多项式复杂度保证的延拓框架

    Certified Inference and Training for Deep Equilibrium Networks: A Continuation Framework with Polynomial Complexity Guarantees

    [https://arxiv.org/abs/2609.16485](https://arxiv.org/abs/2609.16485)

    本文提出了一种认证延拓框架，将深度均衡网络的训练表述为精度插值问题，使推理和训练都能在多项式复杂度预算下获得可认证的保证。

    

    我们为均衡计算和深度均衡网络（DEQ）训练开发了一个认证延拓框架，其中训练被表述为达到精度 $2^{-b}$ 的插值问题。对于推理，紧凑输入同伦从给定的起始根中选择唯一分支，并在经过认证的边界、条件数、导数和管道半径约束下，由舍入牛顿追踪器沿该分支进行追踪。对于训练，我们通过可编程的休眠双线性秩一通道来增强局部加低秩递归结构。加载的 Tikhonov 求解可以在无需谱分解的情况下诊断失败的插值过程；与该过程残差对齐的保持输出的修复机制提供了所需的方向。训练需要在每个过程区域上实现认证门控和列稳定性、适定的推理以及有限的更新误差预算。在多项式几何、编码、精度和完整后端预算约束下，认证推理和训练均具有（多项式复杂度）……

    arXiv:2609.16485v1 Announce Type: cross  Abstract: We develop a certified continuation framework for equilibrium computation and for training deep equilibrium networks (DEQs), with training formulated as interpolation to accuracy $2^{-b}$. For inference, compact input homotopy selects a unique branch from a supplied start root, and a rounded Newton tracker follows it under certified boundary, conditioning, derivative, and tube-radius bounds. For training, we augment local-plus-low-rank recurrence with programmable dormant bilinear rank-one channels. Loaded Tikhonov solves diagnose a failed interpolation pass without spectral decomposition; an output-preserving repair aligned with the pass residual supplies the required direction. Training requires certified gate realization and column stability on each pass region, well-posed inference, and finite-update error budgets. With polynomial geometric, encoding, precision, and complete backend budgets, both certified inference and training ha
    
[^17]: CART的学习型前瞻分裂规则

    Learned Look-Ahead Splitting Rule for CART

    [https://arxiv.org/abs/2609.16440](https://arxiv.org/abs/2609.16440)

    该论文提出一种通过在候选分裂点下方生长CART子树来评估分裂质量的前瞻分裂规则，并利用节点级特征学习的智能前瞻算法大幅降低计算成本，在保持决策树可解释性的同时显著改善层级或交互场景下的分裂选择。

    

    分类和回归树（CART）通常采用贪心分裂规则构建，即在每个节点处最大化预测误差的即时下降。尽管这种策略计算效率高，但它可能错过那些短期收益较小、却在进一步划分后能带来显著下游改进的分裂。我们提出了一种前瞻式建树方法，通过在该候选分裂点下方生长一个常规CART子树后所取得的预测误差下降来评估每个候选分裂。由于完整的前瞻过程计算代价高昂，我们还提出了一种智能前瞻算法，利用节点级特征来学习下游的分裂值。所提出的框架在保持递归划分可解释性的同时，改善了层级结构或交互效应主导场景下的分裂选择。我们开展了模拟研究，对常规方法、完整前瞻和智能前瞻方法进行比较。

    arXiv:2609.16440v1 Announce Type: cross  Abstract: Classification and regression trees are typically constructed using a greedy splitting rule that maximizes the immediate reduction in prediction error at each node. Although this strategy is computationally efficient, it can miss splits that yield small short-term gains but create substantial downstream improvements after further partitioning. We propose a look-ahead tree-building method that evaluates each candidate split by the prediction error reduction achieved after growing a conventional CART subtree below that split. Because the full look-ahead procedure can be computationally expensive, we also describe a smart look-ahead algorithm that learns downstream split values using node-level features. The proposed framework preserves the interpretability of recursive partitioning while improving split selection in hierarchical or interaction-driven settings. We conduct a simulation study comparing conventional, full look-ahead, and sma
    
[^18]: 基于有界调整与可靠性引导嵌入的带噪标签不平衡学习方法

    Bounded Adjustment with Reliability-Guided Embedding for Imbalanced Learning with Noisy Labels

    [https://arxiv.org/abs/2609.16380](https://arxiv.org/abs/2609.16380)

    该论文提出BARGE方法，将有界先验调整密度幂得分与可靠性引导嵌入结合为单阶段目标函数，在应对类别不平衡的同时约束噪声标签带来的分类风险扰动，并在模型与标签高度冲突时自动衰减梯度以抵抗标签噪声。

    

    类平衡学习与标签噪声会形成一种耦合的失效模式：频率校正可以防止多数类主导决策规则，但却可能放大被错误标注的少数类样本的影响。我们提出了BARGE（有界调整与可靠性引导嵌入，Bounded Adjustment with Reliability-Guided Embeddings），这是一种单阶段目标函数，将有界的、经先验调整的密度幂得分与可靠性引导的角度几何结构相结合。其分类得分在调整后的概率空间中是严格恰当的，并且在干净监督与真实类别先验条件下能够恢复平衡的贝叶斯排序。在标签污染的情况下，其有限的取值范围在固定预测器处约束了分类风险的扰动，而当模型高度自信地与给定标签相矛盾时，其logit梯度会重新衰减。调整后的目标概率还对类间均等的特征紧凑性进行加权，同时单侧分离项会抑制不同类别方向的对齐。BARGE既不需要噪声（原文摘要在此处截断）

    arXiv:2609.16380v1 Announce Type: new  Abstract: Class-balanced learning and label noise create a coupled failure mode: frequency correction prevents majority classes from dominating the decision rule, but can amplify incorrectly labeled minority examples. We introduce BARGE (Bounded Adjustment with Reliability-Guided Embeddings), a single-stage objective combining a bounded, prior-adjusted density-power score with reliability-guided angular geometry. Its classification score is strictly proper in the adjusted probability space and recovers balanced Bayes ordering under clean supervision and the true class prior. Under label contamination, its finite range bounds classification-risk perturbation at a fixed predictor, while its logit gradient redescends when the model confidently contradicts the supplied label. The adjusted target probability also weights class-equal feature compactness, and a one-sided separation term discourages aligned class directions. BARGE requires neither a noise
    
[^19]: 面向长尾图像分类的小批量采样策略：基于CIFAR-100-LT的实证研究

    Mini-batch Sampling Strategies for Long-Tailed Image Classification: An Empirical Study on CIFAR-100-LT

    [https://arxiv.org/abs/2609.16365](https://arxiv.org/abs/2609.16365)

    本文在统一的偏差-方差框架下系统比较了均匀实例采样、类平衡采样、平方根采样和渐进平衡采样四种小批量采样策略对长尾图像分类中梯度估计的影响，并在CIFAR-100-LT数据集上进行了实证评估。

    

    现实世界的数据集通常呈现长尾类别分布，少数头部类别包含大量训练样本，而大量尾部类别仅有极少样本。由采样策略决定的每个小批量的组成，决定了哪些类别参与随机梯度估计，从而影响整个类别范围内的收敛行为和泛化能力。我们对四种用于长尾图像分类的小批量采样策略进行了系统的理论与实证比较：均匀实例采样、类平衡采样、平方根采样和渐进平衡采样。我们将这四种策略置于统一的偏差-方差框架中，阐述它们对梯度估计的影响，揭示了经验损失无偏优化与稀有类别公平表示之间的矛盾。随后，我们在受控条件下对它们进行评估。

    arXiv:2609.16365v1 Announce Type: cross  Abstract: Real-world datasets often exhibit long-tailed class distributions, where a few head classes contain a large number of training samples while a large number of tail classes have only a few. The composition of each mini-batch, determined by the sampling strategy, governs which classes contribute to the stochastic gradient estimate, and therefore affects convergence behaviour and generalisation across the whole class spectrum. We provide a systematic theoretical and empirical comparison of four mini-batch sampling strategies for long-tailed image classification: uniform instance sampling, class-balanced sampling, square-root sampling, and progressively balanced sampling. We place all four in a unified bias-variance framework describing their effect on gradient estimation, which exposes the tension between unbiased optimisation of the empirical loss and fair representation of rare classes. We then evaluate them under controlled conditions 
    
[^20]: 岭梯度下降中的计算最优预训练—微调策略

    Compute-Optimal Pretrain--Fine-tune in Ridge Gradient Descent

    [https://arxiv.org/abs/2609.16262](https://arxiv.org/abs/2609.16262)

    本文在岭回归梯度下降的两阶段预训练—微调框架下，首次从理论上刻画了固定总优化预算时上游预训练与下游微调之间的最优计算分配，并揭示该分配由预测相关的谱分量和下游数据几何共同决定。

    

    预训练之后进行微调会引入一个计算分配问题：在固定训练预算下，用于提升上游目标的计算会减少可用于下游适配的计算。尽管这一权衡在实践中十分重要，但即使是在简单模型中，其理论理解仍然不足。本文将这一分配问题转化为一个在总优化预算固定的两阶段预训练—微调流程下的计算拆分问题，并以由梯度下降训练的正则化最小二乘（岭回归）作为一个可解析处理的设置。我们刻画了在由微调问题所诱导的数据相关评估几何下的最优拆分。结果表明，计算分配取决于预训练方向如何影响微调预测，以及微调偏移如何通过下游数据几何被观测。特别地，相关量由与预测相关的谱分量所决定。

    arXiv:2609.16262v1 Announce Type: cross  Abstract: Pretraining followed by fine-tuning introduces a compute-allocation problem: under a fixed training budget, compute spent improving the upstream objective reduces the compute available for downstream adaptation. Despite its practical importance, this trade-off is not yet well understood theoretically, even in simple models. In this paper, we cast this allocation as a compute-split problem under a two-stage pretrain--fine-tune procedure with fixed total optimisation budget, using regularised least squares trained by gradient descent as a tractable setting. We characterise the optimal split under data-dependent evaluation geometries induced by the fine-tuning problem. Our results show that the allocation depends on how pretraining directions affect fine-tuning predictions and how fine-tuning shifts are seen through downstream data geometry. In particular, the relevant quantities are determined by prediction-relevant spectral components o
    
[^21]: 用于生物医学数据聚类表示的Copula自适应有向无环图

    Copula Adapted Directed Acyclic Graph for Cluster Representation of Biomedical Data

    [https://arxiv.org/abs/2609.16240](https://arxiv.org/abs/2609.16240)

    本文提出了一种融合Copula非高斯非线性依赖建模与基于有向无环图的集成因果结构发现方法的新型数据表示框架，用于无标签高维生物医学数据的聚类表示。

    

    诊断错误和标签误标在生物医学领域十分常见，这损害了预测模型和数据驱动结果的可靠性。基于特征之间的复杂关系对无标签生物医学数据进行分层，能够消除对数据标签的需求，并克服监督学习的局限性。传统聚类方法假设数据分布具有较强的限制性，因此在捕捉高维生物医学数据中的复杂依赖关系方面表现欠佳。本文提出了一种新颖的面向聚类的数据表示框架，该框架将Copula模型的非高斯和非线性特征依赖建模与基于有向无环图（DAG）的集成因果结构发现（CSD）方法相结合。Copula通过放宽多元正态性、线性依赖和对称关系等假设来建模灵活的多元分布，而基于DAG的集成因果结构发现方法能够识别……

    arXiv:2609.16240v1 Announce Type: cross  Abstract: Diagnostic errors and mislabeling are common in biomedicine, which compromise the reliability of predictive models and data-driven outcomes. Stratifying unlabeled biomedical data based on complex relationships between features eliminates the need for data labels and overcomes the limitations of supervised learning. Traditional clustering methods assume restrictive data distributions, making them suboptimal for capturing complex dependencies in high-dimensional biomedical data. This paper introduces a novel cluster-friendly data presentation framework that integrates the non-Gaussian and non-linear feature dependence of copula models with an ensemble of causal structure discovery (CSD) methods based on Directed Acyclic Graphs (DAGs). While copulas model flexible multivariate distributions by relaxing assumptions related to multivariate normality, linear dependence, and symmetric relationships, an ensemble of DAG-based CSD methods identi
    
[^22]: 迭代神经扩张上的骨架原型

    Skeletal Prototypes on Iterative Nerve Expansions

    [https://arxiv.org/abs/2609.16170](https://arxiv.org/abs/2609.16170)

    SPINE方法创新性地用嵌入的一维复形（骨架结构）而非传统点集来表示各类原型，通过类条件Mapper图构建初始边集并在分类目标下优化顶点位置，使骨架线段直接参与决策规则，在17个基准数据集上取得了最优的平均准确率和排名。

    

    原型约简是用一个更小的表示来替换训练集，而现有方法返回的是一个有限的点集。我们提出了迭代神经扩张骨架原型方法（SPINE）。该方法中每个类别的模型是一个嵌入的一维复形，而非点集。其初始边集是一个类条件Mapper图，因此由数据本身决定哪些局部聚类被连接在一起。后续阶段在分类目标下对顶点进行拟合，并将观测样本分配给其复形距离最近的类别。因此，这些线段不仅参与拟合过程，还直接进入决策规则。我们在17个基准数据集上，采用分层10折交叉验证，在相同预算条件下与七种其他原型约简方法进行对比来评估SPINE。SPINE获得了最高的平均准确率和最佳的平均排名。在经Holm校正的Wilcoxon符号秩检验下，它显著优于七个竞争方法中的五个。

    arXiv:2609.16170v1 Announce Type: new  Abstract: Prototype reduction replaces a training set with a smaller representation, and the established methods return a finite set of points. We propose Skeletal Prototypes on Iterative Nerve Expansions (SPINE). The model for each class is an embedded 1-complex rather than a point set. Its initial edge set is a class-conditional Mapper graph, so the data decide which localized clusters are joined. Later phases fit the vertices under a classification objective, and an observation is assigned to the class whose complex is nearest. The segments therefore enter the decision rule and not only the fitting. We evaluate SPINE on seventeen benchmark datasets under stratified 10-fold cross validation, against seven other prototype reduction methods at a matched budget. SPINE attains the highest mean accuracy and the best average rank. It is significantly better than five of the seven competitors under Wilcoxon signed-rank tests with Holm correction. A bud
    
[^23]: P2空间（Wasserstein空间）上的随机梯度下降

    Stochastic Gradient Descent over P2

    [https://arxiv.org/abs/2609.13343](https://arxiv.org/abs/2609.13343)

    该论文将经典欧氏空间中SGD的扩散（高斯）近似理论首次推广到Wasserstein空间P2上的优化问题，通过Lions可微性将问题提升至线性希尔伯特空间，并构造了与随机梯度矩信息相匹配的高斯随机场近似。

    

    随机梯度下降（SGD）存在扩散近似方法，即用高斯噪声替代随机梯度中复杂的随机性，这为理解其动力学和长时间行为提供了强有力的工具。我们研究了类似的近似原理是否适用于概率测度空间上的优化问题，其目标函数是定义在Wasserstein空间P2上的泛函。P2的非线性几何结构和无穷维特性阻碍了经典欧几里得理论的直接推广。利用Lions可微性，我们将该问题提升到一个线性希尔伯特空间，从而可以进行高阶微分演算。随后，我们构造了一个高斯随机场近似，其速度场与原始随机梯度的均值和协方差相匹配。通过在高阶泰勒展开中利用这种矩匹配，我们证明了高斯近似能够捕捉原始动力学……（摘要原文在此处截断）

    arXiv:2609.13343v1 Announce Type: cross  Abstract: Stochastic gradient descent (SGD) admits diffusion approximations that replace the complicated randomness of stochastic gradients by Gaussian noise, providing a powerful tool for understanding its dynamics and long-time behavior. We investigate whether an analogous approximation principle holds for optimization over probability measures, where the objective is a functional defined on the Wasserstein space P2. The nonlinear geometry and infinite-dimensional nature of P2 prevent a direct extension of the classical Euclidean theory. Using Lions differentiability, we lift the problem to a linear Hilbert space, where higher-order differential calculus becomes available. We then construct a Gaussian random-field approximation whose velocity field matches the mean and covariance of the original stochastic gradient. By exploiting this moment matching through higher-order Taylor expansions, we show that the Gaussian approximation captures the S
    
[^24]: ICON分解：用于模型审计的深度表示多变量概念级解释

    ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing

    [https://arxiv.org/abs/2608.26083](https://arxiv.org/abs/2608.26083)

    ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。

    

    arXiv:2608.26083v1 公告类型：新 摘要：深度神经网络经常利用训练数据中的虚假关联，这种失败被称为捷径学习。基于概念的可解释性方法通过测试诸如患者性别或扫描仪设置等概念是否能从网络层中解码来筛选捷径。由于每个概念是单独评估的，这些方法可能会将概念之间的相关性误认为是模型使用它们的证据。我们引入了ICON分解，它转而量化每个概念在考虑所有其他概念和结果后所解释的层方差的比例。在具有已知真实标签的合成数据上，ICON比七种替代基线方法更准确地恢复了概念重要性。在皮肤病变和脑成像模型中，它隔离了模型真正依赖的概念，量化了任何提供的概念未解释的表示部分，并产生了我们验证过的稀疏解释。

    arXiv:2608.26083v1 Announce Type: new  Abstract: Deep neural networks often exploit spurious associations in their training data, a failure known as shortcut learning. Concept-based explainability methods screen for shortcuts by testing whether concepts such as a patient's sex or scanner settings can be decoded from a network layer. Because each concept is evaluated in isolation, these methods can mistake correlations between concepts as evidence that the model uses them. We introduce ICON decomposition, which instead quantifies how much of a layer's variance each concept explains after accounting for all other concepts and the outcome. On synthetic data with known ground truth, ICON recovers concept importance more accurately than seven alternative baseline methods. On skin-lesion and brain-imaging models, it isolates the concepts on which a model genuinely relies, quantifies the representation unexplained by any of the supplied concepts, and yields sparse explanations that we validat
    
[^25]: 随机风险森林

    Random Hazard Forests

    [https://arxiv.org/abs/2608.21597](https://arxiv.org/abs/2608.21597)

    随机风险森林通过非参数风险似然和连续时间树集成，直接处理不规则、多源临床数据，实现动态更新的个体化风险预测。

    

    arXiv:2608.21597v1 公告类型：新 摘要：临床数据源，如电子健康记录和可穿戴传感器，会在随访期间反复记录患者状态，通常时间不规律且不同测量有不同的时间表。这些数据为持续更新、个体化的风险预测创造了机会。然而，现有方法在建模前往往简化了时间结构。我们引入了随机风险森林（RHF），这是一种生存树集成方法，它学习当新测量值可用时，患者风险如何在连续时间内变化。RHF通过可预测协变量过程的非参数风险似然直接公式化估计问题。一个高效的工作模型指导树的构建，之后为每个终端节点估计灵活的时间变化风险。给定任何可预测的协变量路径，每棵树沿其终端节点随时间跟踪路径，并组装相应的节点级风险估计。

    arXiv:2608.21597v1 Announce Type: new  Abstract: Clinical data sources such as electronic health records and wearable sensors record patient status repeatedly over follow-up, often at irregular times and on different schedules for different measurements. These data create opportunities for continuously updated, individualized risk prediction. Existing approaches, however, often simplify the temporal structure before modeling it. We introduce Random Hazard Forests (RHF), a survival tree ensemble that learns how a patient's hazard changes in continuous time as new measurements become available. RHF formulates the estimation problem directly through a nonparametric hazard likelihood for predictable covariate processes. An efficient working model guides tree construction, after which flexible time-varying hazards are estimated for each terminal node. Given any predictable covariate path, each tree follows the path through its terminal nodes over time and assembles the corresponding node-le
    
[^26]: CausalSmith：一个形式化基础、自我改进的自动化因果推断研究代理框架

    CausalSmith: A Formally Grounded, Self-Improving Agentic Framework for Automated Research in Causal Inference

    [https://arxiv.org/abs/2607.22511](https://arxiv.org/abs/2607.22511)

    CausalSmith通过结合Lean证明助手和自改进代理管道，解决了LLM评审员不可靠的问题，实现了因果推断领域自动化理论研究中可验证、可靠的结果生成与评估。

    

    自动化理论研究不仅受限于候选结果的生成，还受限于其可靠评估。一种常见方法是使用大型语言模型（LLM）评审员来闭环研究过程。然而，此类评审员在经验上仍不可靠：他们可能接受伪造论文，并以接近随机水平的概率检测出这些论文（Bad Scientist，2025）。我们提出了CausalSmith，一个基于Lean证明助手的因果推断自动化理论研究框架。CausalSmith结合了Causalean（一个基础性的因果推断Lean库，包含7,035条机器检查的声明，在人类设计与审查下借助语言模型辅助开发）以及CausalSmith（一个自我改进的代理管道，用于选择研究主题、提出结果、形式化陈述、构造证明，并呈现最终产物供人类检查）。由于机器检查的证明……

    arXiv:2607.22511v3 Announce Type: replace-cross  Abstract: Automating theoretical research is constrained not only by the generation of candidate results, but also by their reliable evaluation. A common approach is to close the research loop with a large language model (LLM) reviewer. However, such reviewers remain empirically unreliable: they may accept fabricated papers and detect them at rates close to chance (Bad Scientist, 2025). We present CausalSmith, a framework for automated theoretical research in causal inference grounded in the Lean proof assistant. CausalSmith combines Causalean, a foundational Lean library for causal inference containing 7,035 machine-checked declarations developed with language-model assistance under human design and review, with CausalSmith, a self-improving agentic pipeline that selects research topics, proposes results, formalizes statements, constructs proofs, and presents the resulting artifacts for human inspection. Because a machine-checked proof 
    
[^27]: 流形假设下的缺失数据插补

    Missing Data Imputation under Manifold Hypothesis

    [https://arxiv.org/abs/2607.03641](https://arxiv.org/abs/2607.03641)

    本文基于流形假设与混合变分自编码器，提出了一种通过SIR采样和潜空间扩散模型从条件分布中采样的缺失数据插补方法，在尊重数据几何结构的同时实现高质量插补并量化不确定性。

    

    流形假设认为，高维数据集中在低维嵌入流形附近。混合变分自编码器（VAE）的最新进展为忠实提取这种潜在结构提供了强大的工具。所得的几何结构自然地引入了变量之间的局部和全局关系，从而为缺失数据插补提供了一种系统化的方法。我们提出了一种基于模型的插补方法，能够通过采样-重要性-重采样（SIR）程序从 \( p(\bm{x}_{\mathrm{mis}} \mid \bm{x}_{\mathrm{obs}}) \) 中进行采样，并且可以通过潜空间中的联合扩散模型进一步增强。我们的方法在尊重底层几何结构的同时对缺失数据进行插补，与最先进的方法相比取得了有竞争力的性能，能够量化插补结果的不确定性，并且是基于模型的方法，从而能够实现即时插补。

    arXiv:2607.03641v3 Announce Type: replace-cross  Abstract: The manifold hypothesis posits that high-dimensional data are concentrated near a low-dimensional embedded manifold. Recent advances in mixture variational autoencoders (VAEs) provide a powerful tool for extracting such underlying structure in a faithful manner. The resulting geometric structure naturally introduces local and global relationships among variables, thereby providing a systematic way of imputing missing data. We propose a model-based imputation method that enables sampling from \( p(\bm{x}_{\mathrm{mis}} \mid \bm{x}_{\mathrm{obs}}) \) via a sampling-importance-resampling (SIR) procedure, which can be further augmented with a joint diffusion model in the latent space. Our method imputes missing data while respecting the underlying geometry, achieves competitive performance compared to state-of-the-art procedures, quantifies uncertainty in the imputations, and is model-based, thereby enabling on-the-fly imputation w
    
[^28]: 纵向切分分布式模型中稀疏协方差估计的信息论界

    Information-Theoretic Bounds for Sparse Covariance Estimation in the Vertical-Split Distributed Model

    [https://arxiv.org/abs/2606.07124](https://arxiv.org/abs/2606.07124)

    该论文首次证明在纵向切分分布式设置中，对互协方差矩阵施加稀疏性约束能够有效降低通信和样本复杂度，这与水平切分设置下稀疏性无法降低通信成本的结论形成鲜明对比。

    

    我们研究纵向切分（特征切分）设置下分布式协方差矩阵估计的极小极大估计误差。在该设置中，两个智能体各自观测 m 个独立同分布次高斯样本的不同坐标，并向中央服务器传送有限数量的比特。尽管先前研究已为稠密（非结构化）互协方差矩阵建立了近乎紧致的界，我们研究的问题是：对互协方差矩阵 $C_{21}$ 施加逐元素 s-稀疏性约束能否降低所需的通信复杂度和样本复杂度。与水平切分设置形成鲜明对比的是——在该设置中已有研究表明稀疏性并不能降低均值估计的通信成本——我们证明在纵向切分下，稀疏性确实有助于互协方差估计。具体而言，对于足够大的 $d_1d_2/s'$ 以及 $0<\varepsilon<\sigma^2\sqrt{s'}/32$，任何在期望 Frobenius 范数误差上达到目标精度的方案……（摘要原文在此处截断）

    arXiv:2606.07124v2 Announce Type: replace-cross  Abstract: We study the minimax estimation error for distributed covariance matrix estimation in the vertical-split (feature-split) setting, where two agents each observe different coordinates of~$m$ i.i.d.\ sub-Gaussian samples and communicate a limited number of bits to a central server. While \cite{rahmani2025fundamental} established nearly tight bounds for dense (unstructured) cross-covariance matrices, we investigate whether imposing elementwise $s$-sparsity on the cross-covariance $C_{21}$ can reduce the required communication and sample complexity. In contrast to the horizontal-split setting, where \cite{braverman2016communication} showed that sparsity does \emph{not} reduce communication cost for mean estimation, we prove that sparsity \emph{does} help for cross-covariance estimation in the vertical split.   Specifically, for sufficiently large $d_1d_2/s'$ and $0<\varepsilon<\sigma^2\sqrt{s'}/32$, any scheme achieving expected Fro
    
[^29]: BASIS：基于单rollout信息共享的批量优势估计方法用于大语言模型推理

    BASIS: Batchwise Advantage Estimation from Single-Rollout Information Sharing for LLM Reasoning

    [https://arxiv.org/abs/2605.27293](https://arxiv.org/abs/2605.27293)

    BASIS通过在批次内共享单rollout的跨提示信息来改进价值函数估计，以显著更低的计算成本实现了接近多rollout方法的策略优化性能。

    

    带可验证奖励的强化学习已成为提升大语言模型推理能力的标准方法。现有算法在价值估计和策略学习中面临计算效率与样本效率之间的权衡。我们提出了BASIS，一种无需评论家（critic-free）的后训练算法，旨在解决这一权衡问题。在每个在线训练步骤中，BASIS仅对每个提示采样一个rollout，但利用整个批次中跨提示的丰富信息来改进价值函数估计。我们的实验表明，与REINFORCE++（一个代表性的单rollout基线）相比，BASIS将价值函数估计的均方误差（MSE）降低了69%，且仅用一个rollout就达到了比使用8个rollout的组均值估计器更低的MSE。这种价值估计的改进转化为更好的策略优化：BASIS在使用显著更少训练时间的情况下，实现了接近多rollout方法的性能。

    arXiv:2605.27293v2 Announce Type: replace  Abstract: Reinforcement learning with verifiable rewards has become a standard recipe for improving the reasoning abilities of large language models. Existing algorithms face a tradeoff between computational efficiency and sample efficiency in value estimation and policy learning. We introduce BASIS, a critic-free post-training algorithm designed to address this tradeoff. At each online training step, BASIS samples only one rollout per prompt, but leverages rich information across prompts in the entire batch to improve value function estimation. Our experiments demonstrate that BASIS reduces MSE in value function estimation by 69% compared to REINFORCE++, a representative single-rollout baseline, and achieves lower MSE with one rollout than group mean estimators with 8 rollouts. This improvement in value estimation translates to better policy optimization: using substantially less training time, BASIS achieves performance close to multi-rollou
    
[^30]: 基于噪声观测与弱对称性条件的通用特征选择

    Universal Feature Selection with Noisy Observations and Weak Symmetry Conditions

    [https://arxiv.org/abs/2605.09396](https://arxiv.org/abs/2605.09396)

    本文提出在弱球对称性条件下基于噪声数据的通用特征选择框架，通过典型相关依存矩阵的奇异值分解实现渐近最优误差指数，并证明精确的球对称性条件并非必要。

    

    本文放宽了文献[4]、[5]中所采用的严格对称性条件，并将其通用特征选择框架扩展至可处理含噪声观测以及可能表现出方向性偏好的属性结构。我们引入了弱球对称性的概念，通过二阶矩距离对其进行量化，从而允许对旋转不变性产生受控的偏离。在这一放宽的条件下，我们开发了一种基于由噪声数据计算得到的典型相关依存矩阵奇异值分解的通用特征选择框架。我们的主要结果表明，所选择的特征能够达到渐近最优的误差指数，仅存在一个取决于对称性偏差 $\delta$ 和噪声水平 $\eta_1, \eta_2$ 的残余项。当 $\delta, \eta_1, \eta_2$ 相对较小时，我们的结果可以恢复文献[5]的结果，从而证明了精确的球对称性并非必要条件。

    arXiv:2605.09396v2 Announce Type: replace-cross  Abstract: This paper relaxes the restrictive symmetry conditions adopted in [4], [5] and extends their universal feature selection framework to accommodate noisy observations as well as attribute structures that may exhibit directional preferences. We introduce the notion of weak spherical symmetry, quantified by second-moment distances, which allows controlled deviations from rotational invariance. Under this relaxed condition, we develop a universal feature selection framework based on the singular value decomposition of the canonical dependence matrix computed from noisy data. Our main result shows that the selected features achieve asymptotically optimal error exponents up to a residual term that depends on the symmetry deviation $\delta$ and the noise levels $\eta_1, \eta_2$. When $\delta, \eta_1, \eta_2$ are relatively small, our result recovers that of [5], thereby demonstrating that exact spherical symmetry is unnecessary. Overal
    
[^31]: 单脉冲与多脉冲神经元网络逼近的等价性

    Equivalence of approximation by networks of single- and multi-spike neurons

    [https://arxiv.org/abs/2603.13478](https://arxiv.org/abs/2603.13478)

    本文证明对于包括泄漏积分发放模型在内的一大类脉冲神经元模型，单脉冲网络与多脉冲网络在函数逼近能力上完全等价，两者之间仅需以神经元数量的线性倍数进行转换。

    

    在脉冲神经网络中，每个神经元至多发放一次脉冲是否就足够了？在最近的研究中，已经推导出了脉冲神经网络的逼近界，用以量化它们拟合目标函数的能力。然而，这些结果仅对至多发放一次脉冲的神经元有效，这通常被认为是一个很强的限制。本文证明，对于一大类脉冲神经元模型（包括常用的带减法重置的泄漏积分发放模型），情况恰恰相反：对于每一个对多脉冲神经网络集合成立的逼近界，都存在一个等价的单脉冲神经网络集合——其神经元数量相对于最大脉冲数量仅线性地更多（或更少）——该逼近界对其同样成立。反方向亦是如此。这表明，就一般机器学习任务中的逼近能力而言，单脉冲与多脉冲神经网络（是等价的）。

    arXiv:2603.13478v2 Announce Type: replace-cross  Abstract: In a spiking neural network, is it enough for each neuron to spike at most once? In recent work, approximation bounds for spiking neural networks have been derived, quantifying how well they can fit target functions. However, these results are only valid for neurons that spike at most once, which is commonly thought to be a strong limitation. Here, we show that the opposite is true for a large class of spiking neuron models, including the commonly used leaky integrate-and-fire model with subtractive reset: for every approximation bound that is valid for a set of multi-spike neural networks, there is an equivalent set of single-spike neural networks with only linearly more (or less) neurons, in the maximum number of spikes, for which the bound holds. The same is true for the reverse direction too, showing that regarding their approximation capabilities in general machine learning tasks, single-spike and multi-spike neural networ
    
[^32]: 分数分解的统计推断

    Statistical Inference for Score Decompositions

    [https://arxiv.org/abs/2603.04275](https://arxiv.org/abs/2603.04275)

    该论文提出了基于预测线性重校准的评分分解统计推断方法，将预测评分分解为误校准、判别力与不确定性三个可解释成分，适用于非光滑评分函数并支持模型误设下的渐近推断。

    

    我们提出了针对分数分解的推断方法，该方法将用于预测评估的评分函数分解为三个可解释的组成部分：误校准、判别力和不确定性。我们的估计与推断依赖于对预测的线性重校准，并且由于其对非光滑评分函数的有效性，可适用于一般的点预测，例如均值和分位数。该方法确保了有限样本中分解项的非负性，能够在模型误设下进行渐近推断，并与经典的 Mincer-Zarnowitz 回归建立了直接联系。由此产生的推断框架促进了对线性化预测校准或判别力相等性的新检验，这带来了三个关键优势：它们通过分解分数增强了预测能力检验的信息含量，能够在预测差异归因于特定成分的场景中提高检测能力……

    arXiv:2603.04275v2 Announce Type: replace  Abstract: We introduce inference methods for score decompositions, which partition scoring functions for predictive assessment into three interpretable components: miscalibration, discrimination, and uncertainty. Our estimation and inference relies on a linear recalibration of the forecasts and is applicable to general point forecasts such as means and quantiles due to its validity for non-smooth scoring functions. This approach ensures non-negative decomposition terms in finite samples, enables asymptotic inference under model misspecification, and establishes a direct connection to the classical Mincer-Zarnowitz regression. The resulting inference framework facilitates novel tests for equal linearized forecast calibration or discrimination, which yield three key advantages. They enhance the information content of predictive ability tests by decomposing scores, can improve detection power in scenarios where predictive differences are attribut
    
[^33]: 非负矩阵分解及相关成分模型：等价性、可辨识性及其在沉积物粒度分析中的应用

    Nonnegative matrix factorizations and related compositional models: Equivalence, identifiability, and an application on the grain-size analysis of sediments

    [https://arxiv.org/abs/2512.22282](https://arxiv.org/abs/2512.22282)

    本文证明了来自社会科学、地质学和机器学习的五种模型（LBA、LCA、EMA、PLSA、NMF）在本质上等价，NMF的解唯一性定理可直接推广到其他四种模型，并将其应用于沉积物粒度分析。

    

    在机器学习、社会科学和地质学等领域中，将非负矩阵分解为两个或三个矩阵乘积的模型受到了广泛关注，这些模型受非负约束或行和为1的约束。尽管这些模型在很大程度上相似甚至等价，但它们以不同的名称呈现，其相似性并不为人所熟知。本文重点阐述了五种模型之间的相似性，包括来自社会科学的潜在预算分析（LBA）和潜在类别分析（LCA）、来自地质学的端元分析（EMA），以及来自机器学习的概率潜在语义分析（PLSA）和非负矩阵分解（NMF）。我们聚焦于这些模型的可辨识性，证明了LBA、EMA、LCA、PLSA的解是唯一的当且仅当NMF的解是唯一的。因此，NMF现有的唯一性定理可直接应用于LBA、EMA、LCA、PLSA，反之亦然。

    arXiv:2512.22282v2 Announce Type: replace-cross  Abstract: Across fields such as machine learning, social science, and geology, considerable attention has been given to models that factorize a nonnegative matrix into the product of two or three matrices, subject to nonnegative or row-sum-to-1 constraints. Although these models are to a large extent similar or even equivalent, they are presented under different names, and their similarity is not well known. This paper highlights similarities among five models, latent budget analysis (LBA) and latent class analysis (LCA) from social science, end-member analysis (EMA) from geology, probabilistic latent semantic analysis (PLSA) and nonnegative matrix factorization (NMF) from machine learning. We focus on the identifiability of these models. We prove that the solution of LBA, EMA, LCA, PLSA is unique if and only if the solution of NMF is unique. Consequently, existing uniqueness theorems for NMF directly apply to LBA, EMA, LCA, PLSA, and vi
    
[^34]: 使用非MCMC采样器与高效温度估计训练能量基模型

    Training Energy-Based Models with Non-MCMC Samplers and Efficient Temperature Estimation

    [https://arxiv.org/abs/2512.02323](https://arxiv.org/abs/2512.02323)

    该论文提出了朗之万模拟分岔（LSB）快速并行玻尔兹曼采样器、条件期望匹配（CEM）高效温度估计方法以及采样器自适应学习（SAL）框架，从而无需MCMC即可高效训练能量基模型。

    

    从离散变量上的玻尔兹曼分布中高效采样是众多应用领域中的基础操作。虽然快速的非MCMC采样器近来已成为传统MCMC方法的有前景的替代方案，但由于难以估计生成样本的有效温度，它们在概率学习中的实际应用仍受到阻碍。在本工作中，我们首先介绍了朗之万模拟分岔（Langevin simulated bifurcation, LSB），这是一种玻尔兹曼采样器，能够实现快速并行采样，其精度可与顺序MCMC方法相媲美。为解决未知有效温度的难题，我们提出了条件期望匹配（conditional expectation matching, CEM），这是一种高效的估计方法，适用于具有可利用条件独立结构的能量基模型（EBM）。基于这些组件，我们进一步开发了一个名为采样器自适应学习（sampler adaptive learning, SAL）的学习框架，该框架能够自适应地调整……

    arXiv:2512.02323v2 Announce Type: replace  Abstract: Efficient sampling from Boltzmann distributions over discrete variables is a fundamental operation in a wide range of applications. While fast non-MCMC samplers have recently emerged as promising alternatives to conventional MCMC methods, their practical use for probabilistic learning remains hindered by the difficulty of estimating the effective temperature of the generated samples. In this work, we begin by introducing Langevin simulated bifurcation (LSB), a Boltzmann sampler that enables fast and parallel sampling with accuracy comparable to sequential MCMC methods. To address the challenge of unknown effective temperature, we propose conditional expectation matching (CEM), an efficient estimation method applicable to energy-based models (EBMs) with exploitable conditional independence structures. Building on these components, we further develop a learning framework, termed sampler adaptive learning (SAL), which adaptively adjusts
    
[^35]: 一种用于带等式与不等式约束的非凸优化的近端增广拉格朗日方法

    A proximal augmented Lagrangian method for nonconvex optimization with equality and inequality constraints

    [https://arxiv.org/abs/2509.02894](https://arxiv.org/abs/2509.02894)

    本文提出一种具有罚参数与近端项自适应更新规则的非精确近端增广拉格朗日方法（P-ALM），通过证明增广拉格朗日函数沿迭代点的可控性，为非凸约束优化建立了新的收敛理论，并对经典ALM也得出了类似的收敛性质。

    

    我们针对非凸结构化优化问题提出了一种非精确近端增广拉格朗日方法（P-ALM）。所提出的方法具有一个易于实现的规则，不仅可用于更新罚参数，还可用于自适应调节近端项。它允许罚参数在早期阶段快速增长以加速进程，同时改善后期迭代中的病态问题——这是传统线性增大罚参数方法的一个众所周知的缺点。我们分析的一个关键要素在于观察到：只要能获得一个初始可行点，增广拉格朗日函数就可以沿迭代点得到有效控制。我们的分析虽然简单，却为P-ALM提供了新的理论视角，并且作为副产品，其非近端变体——经典增广拉格朗日方法（ALM）——也得出了类似的收敛性质。数值实验……

    arXiv:2509.02894v2 Announce Type: replace-cross  Abstract: We propose an inexact proximal augmented Lagrangian method (P-ALM) for nonconvex structured optimization problems. The proposed method features an easily implementable rule not only for updating the penalty parameters, but also for adaptively tuning the proximal term. It allows the penalty parameter to grow rapidly in the early stages to speed up progress, while ameliorating the issue of ill-conditioning in later iterations, a well-known drawback of the traditional approach of linearly increasing the penalty parameters. A key element in our analysis lies in the observation that the augmented Lagrangian can be controlled effectively along the iterates, provided an initial feasible point is available. Our analysis, while simple, provides a new theoretical perspective about P-ALM and, as a by-product, results in similar convergence properties for its non-proximal variant, the classical augmented Lagrangian method (ALM). Numerical 
    
[^36]: 紧凑状态空间上的神经随机微分方程：理论、方法及其在自杀风险建模中的应用

    Neural Stochastic Differential Equations on Compact State Spaces: Theory, Methods, and Application to Suicide Risk Modeling

    [https://arxiv.org/abs/2508.17090](https://arxiv.org/abs/2508.17090)

    本文提出了一类新型神经随机微分方程，其解可被严格证明限制在指定的紧凑多面体状态空间内，克服了现有SDE模型违反定义域约束和数值不稳定的问题，并成功应用于自杀风险建模。

    

    生态瞬间评估（EMA）研究使得通过智能手机收集关于自杀想法和行为（STBs）的高频自我报告成为可能。潜在随机微分方程（SDEs）是建模EMA数据的一个有前景的模型类别，因为这类数据采样不规则、含噪声且部分可观测。但基于SDE的模型存在两个关键局限性：(a) 这些模型经常违反定义域约束，损害了模型的科学有效性和临床可信度；(b) 若不采用临时修复手段（如过度简化的动力学），训练在数值上是不稳定的，而这些修复手段并不适合高风险的应用场景。在本文中，我们开发了一类新颖的、具有强表达能力的SDE，其解可被严格证明被限制在规定的紧凑多面体状态空间内，从而与EMA数据的定义域相匹配。在这项工作中，（1）我们从理论和实证上展示了为什么基于链式法则在紧凑域上构建SDE的方法会失败；（2）我们推导了（摘要在此处被截断）……

    arXiv:2508.17090v5 Announce Type: replace-cross  Abstract: Ecological Momentary Assessment (EMA) studies enable the collection of high-frequency self-reports of suicidal thoughts and behaviors (STBs) via smartphones. Latent stochastic differential equations (SDEs) are a promising model class for EMA data, as it is irregularly sampled, noisy, and partially observed. But SDE-based models suffer from two key limitations. (a) These models often violate domain constraints, undermining scientific validity and clinical trust of the model. (b) Training is numerically unstable without ad hoc fixes (e.g. oversimplified dynamics) that are ill-suited for high-stakes applications. Here, we develop a novel class of expressive SDEs whose solutions are provably confined to a prescribed compact polyhedral state space, matching the domains of EMA data. In this work, (1) we show why chain-rule based constructions of SDEs on compact domains fail, theoretically and empirically; (2) we derive constraints on
    
[^37]: 动力学交互粒子朗之万蒙特卡洛

    Kinetic Interacting Particle Langevin Monte Carlo

    [https://arxiv.org/abs/2407.05790](https://arxiv.org/abs/2407.05790)

    本文提出了动力学交互粒子朗之万蒙特卡洛（KIPLMC）方法，通过参数与潜变量联合演化的扩散过程实现潜变量模型的统计推断，并在强凹条件下获得了具有加速收敛速率和更优维度依赖性的Wasserstein-2距离非渐近收敛保证。

    

    本文提出并分析了用于潜变量模型统计推断的交互式欠阻尼朗之万算法，称为动力学交互粒子朗之万蒙特卡洛（KIPLMC）方法。我们提出了一种在参数空间和潜变量空间中联合演化的扩散过程，并证明该扩散过程的平稳分布集中在参数的极大边际似然估计附近。随后，我们提供了该扩散过程的两种显式离散化方法，作为估计统计模型参数的实用算法。对于每种算法，在联合对数似然关于潜变量和参数强凹的情况下，我们获得了Wasserstein-2距离下的非渐近收敛速率。我们实现了加速的收敛速率，清楚地展示了对维度依赖性的改进。为了展示所引入方法的实用性

    arXiv:2407.05790v4 Announce Type: replace-cross  Abstract: This paper introduces and analyses interacting underdamped Langevin algorithms, termed Kinetic Interacting Particle Langevin Monte Carlo (KIPLMC) methods, for statistical inference in latent variable models. We propose a diffusion process that evolves jointly in the space of parameters and latent variables and show that the stationary distribution of this diffusion concentrates around the maximum marginal likelihood estimate of the parameters. We then provide two explicit discretisations of this diffusion as practical algorithms to estimate parameters of statistical models. For each algorithm, we obtain nonasymptotic rates of convergence in Wasserstein-2 distance for the case where the joint log-likelihood is strongly concave with respect to latent variables and parameters. We achieve accelerated convergence rates clearly demonstrating improvement in dimension dependence. To demonstrate the utility of the introduced methodology
    

