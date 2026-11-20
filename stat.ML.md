# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simple Private Noisy SGD is Minimax-Optimal under Population Strong Convexity](https://arxiv.org/abs/2611.26515) | 本文证明了在强凸损失（尤其是强凸广义线性模型，如隐私逻辑回归）上，采用固定步长的单次噪声SGD即可达到极小极大最优的估计误差率，其上界与基于Score Attack框架推导的隐私下界相匹配，并首次消除了对数因子。 |
| [^2] | [Bayesian optimization on amortized search space](https://arxiv.org/abs/2611.26459) | 该论文提出了一种利用机理模拟器的摊销式逆映射来约束搜索空间的贝叶斯优化方法，通过感知问题几何的高斯过程核高效利用问题的低内在维度，并给出了依赖内在维度的遗憾率理论保证，成功应用于帕金森病的药物剂量个性化。 |
| [^3] | [Exact Bayesian layers via characteristic functions](https://arxiv.org/abs/2611.26426) | 本文提出CF-Prop方法，利用特征函数无采样地精确计算贝叶斯ReLU层的输出分布（即使在非高斯情形下），并能对高斯、泊松、伽马等多种似然闭式计算证据下界，从而实现高效的变分推断算法CF-VI，在CIFAR-100上取得最低校准误差。 |
| [^4] | [Statistical Inference with Ill-Conditioned Random Weight Neural Network Sieves](https://arxiv.org/abs/2611.26410) | 本文证明了随机权重神经网络筛在维数增大时必然出现病态性（最小特征值消失），并在显式处理该病态性的基础上建立了相应的逼近速率、估计与统计推断理论。 |
| [^5] | [Flow Matching from Non-Gaussian Sources: A Theory of Efficient Sampling](https://arxiv.org/abs/2611.26407) | 本文为从非高斯源分布（如重尾和多峰分布）出发的流匹配建立了首个非渐近收敛理论，证明欧拉离散化采样器在温和条件下仅需 $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ 次迭代即可达到 $\varepsilon$ 级总变差误差。 |
| [^6] | [Conformal Selection for Hierarchical Streaming Data](https://arxiv.org/abs/2611.26328) | 该论文提出了面向分层数据流的无分布在线保形选择方法OHS及其带反馈变体OHSF，通过构建组间和组内的基于秩的e值并结合在线e-BH程序，将在线保形选择框架扩展到超越独立同分布/可交换数据假设的分层结构场景。 |
| [^7] | [A Statistical Framework for Training Data Auditing of Large Language Models](https://arxiv.org/abs/2611.26264) | 本文提出了一个统一的统计框架，用于大语言模型训练数据审计，涵盖文档级成员推断、目标数据集全局审计和可疑子集识别，能够有效聚合文档内局部成员信号并进行整体评估。 |
| [^8] | [Time Series Structure Learning with Local Causal Parent Discovery](https://arxiv.org/abs/2611.26232) | 该论文提出ITPD——一种可靠且完备的基于约束的因果结构学习方法，通过局部到全局的因果父节点发现来学习具有因果非平稳性的多变量时间序列结构，并借助时间结构先验与图划分将高阶条件独立性检验替换为低阶检验，从而降低计算与样本复杂度。 |
| [^9] | [Attention Head Specialization: Phase Transitions in Sample Complexity](https://arxiv.org/abs/2611.26228) | 该论文证明注意力头特化是样本复杂度中真正的相变，在可求解的多头注意力模型中表现为测试误差的非解析性，且随数据量增加可通过级联相变逐步恢复更弱的潜在特征，形成统计意义上的简单性偏置。 |
| [^10] | [Hellinger Decision Trees for Density Ratio Estimation: Finite-Sample Guarantees](https://arxiv.org/abs/2611.25936) | 提出了一种通过牛顿提升在轴对齐决策树集成上最小化平衡Hellinger风险来直接估计对数密度比的方法，并建立了有限样本保证，其收敛速率在对数因子范围内达到极小极大最优的L²速率。 |
| [^11] | [Closing Performance Gap for Diffusion Language Models under Low Inference Budgets](https://arxiv.org/abs/2611.25763) | 提出了无需训练的 CaRR 框架，通过 token 承诺重掩码、更新上下文回填和基于轨迹的恢复机制，在低推理预算下提升可编辑扩散语言模型的生成质量，从而缩小其性能差距。 |
| [^12] | [Differentially Private Score Estimation with Applications to Score-Based Generative Models](https://arxiv.org/abs/2611.25536) | 本文研究了差分隐私约束下的分数估计问题，针对具有局部 Hölder 连续分数的次高斯分布，提出了一种基于二次 B 样条和指数机制的采样式隐私分数估计器，并将其应用于隐私保护的基于分数的生成模型。 |
| [^13] | [Sequential Probability Assignment against Smoothed Adversaries with Unknown Base Measure](https://arxiv.org/abs/2611.25514) | 本文在基测度 μ 未知且无需 Lipschitz 条件的更一般平滑在线学习设定下，证明了针对自适应型与疏忽型平滑对手的序贯概率赋值遗憾上界，其中自适应界由经验 ℓ∞ Hellinger 熵控制。 |
| [^14] | [Faithful Generative Modeling for Censored Time-to-Event Data: From Distributional Fidelity to Statistical Inference](https://arxiv.org/abs/2611.25464) | 该论文提出TSGSurv——一个通用的生存数据两阶段生成框架，在事件状态条件下联合生成观测时间和协变量，不仅建立了分布收敛速度，还为Cox比例风险模型下下游统计推断的可靠性提供了理论保证条件。 |
| [^15] | [Instance-Optimal Moment Estimation in the Add-Remove Model of Differential Privacy](https://arxiv.org/abs/2611.25425) | 本文引入实例相关的误差度量并证明其严格下界，设计了在纯差分隐私加-删除模型下达到这些下界的矩估计机制，实现了比现有方法渐近更优的效用。 |
| [^16] | [Causal Discovery with False Positive Error Control](https://arxiv.org/abs/2611.25376) | 提出m-保守PC算法，在线性高斯设定和极弱忠实性假设下实现了因果发现中错误发现率的有限样本控制，同时惩罚虚假邻接与错误边方向。 |
| [^17] | [Fundamentals for Binary Classification Under Label Noise](https://arxiv.org/abs/2611.25372) | 本章系统综述了标签噪声下二分类学习的基础理论，在 PAC 框架下形式化了随机分类噪声、Massart 噪声和恶意分类噪声三种模型，并阐明了经验风险最小化的适用条件以及样本复杂度与噪声率之间的关系。 |
| [^18] | [Prescribed Bayesian Efficiency under Lifetime Adjustment Costs](https://arxiv.org/abs/2611.25311) | 该论文刻画了在终身调整成本约束下，贝叶斯学习者在时间范围 T=2^J 处同时实现累积期望超额损失 O(G(J)) 与有限期望终身费用的可达性准则，即跨动作距离尺度上带前缀约束的跟踪误差配额分配的有限性。 |
| [^19] | [Statistical, Machine Learning and Deep Learning Methods for Missing Covariates: An Empirical Evaluation with Applications to Alzheimer's Disease Research](https://arxiv.org/abs/2611.25219) | 本研究对八种处理缺失协变量的统计、机器学习和深度学习方法进行了系统性实证评估，在阿尔茨海默病研究背景下比较了它们在不同缺失机制下的表现。 |
| [^20] | [The low-degree sample complexity of clustering Gaussian mixtures with more groups than dimensions](https://arxiv.org/abs/2611.25163) | 该论文在组数大于维数的设定下，通过低度多项式的下界与匹配的上界，首次精确刻画了高斯混合聚类所需的最小间隔与样本复杂度，例如当 K=d^{p/2}（p>2）时证明需要量级为 K³d^((⌈p⌉−p)/2) 的样本量。 |
| [^21] | [HapBO: Harnessed $\alpha$-$\pi$ Bayesian Optimization via Expert Natural Language Priors](https://arxiv.org/abs/2611.25100) | 该论文提出HapBO方法，将从人类专家或大语言模型一次性引出的自然语言先验确定性地编译为不限于高斯族的可重用信念函数，从而提升贝叶斯优化的样本效率，并对可能误导的信息保持鲁棒。 |
| [^22] | [Empirical Variational Bayes Factorization for Longitudinal Matrix Data](https://arxiv.org/abs/2611.25065) | 提出了一种经验变分贝叶斯因子分解方法，能够高效处理具有时间相关性的高维纵向矩阵数据，自动确定所有超参数，并有效应对缺失数据和不规则时间点的挑战。 |
| [^23] | [Near-Quadratic Sample Compression for All VC Classes](https://arxiv.org/abs/2611.25011) | 本文通过构造大小为 O(d²log²d) 的样本压缩方案，证明了Floyd和Warmuth提出的弱样本压缩猜想，即每个VC维为 d 的二值概念类都存在多项式大小的样本压缩方案。 |
| [^24] | [A Model of Diverse Sampling from Language Models](https://arxiv.org/abs/2611.25003) | 该论文提出 PoolDPP 方法，将语言模型的多样性采样形式化为行列式点过程（DPP）下的子集采样，通过候选池上的 K-DPP 采样器实现了输出与目标分布间 KL 散度 O(1/N) 的收敛保证，并在实验中扩展了质量-多样性前沿。 |
| [^25] | [Variance-Adaptive Upper Confidence Bounds for Multi-Armed Bandits with Kaplan--Meier Survival Estimation](https://arxiv.org/abs/2611.24988) | 该论文提出了KM-UCB，一种基于Greenwood方差公式和经验Bernstein界限的方差自适应多臂老虎机策略，将连续时间生存分析与序贯决策相结合以处理右删失数据下的异方差问题。 |
| [^26] | [Risk-optimal active learning with informative abstention](https://arxiv.org/abs/2611.24986) | 该论文提出了一种在预言机可能弃权且响应概率与分类器共享参数情形下的主动学习查询获取准则，通过结合调整冗余参数后的分类器信息与多类零一风险的局部曲率，实现首阶渐近超额风险的最小化。 |
| [^27] | [DL101 The Representation Capability of Neural Network Models](https://arxiv.org/abs/2611.24980) | 本文系统梳理了通用近似定理这一神经网络表示能力的数学基石，指出非多项式激活函数是实现通用逼近的必要条件，并强调这些定理本身并不提供关于效率、训练收敛与泛化能力的保证。 |
| [^28] | [FedE2E: One-Shot Federated Learning for End-to-End Autonomous Driving with Closed-Loop Evaluation](https://arxiv.org/abs/2611.24254) | 提出首个面向端到端自动驾驶的一次性联邦学习方法FedE2E，通过多客户端校准、锚点特征对齐和词元级共识蒸馏从异构模型中蒸馏知识，在Bench2Drive闭环评估中一次性驾驶分数达55.85，超过先前工作4.07分。 |
| [^29] | [Prediction with Foundation Models under Deployment Budgets](https://arxiv.org/abs/2611.12466) | 该论文提出了在平均查询成本预算约束下提升基础模型预测性能的框架：固定查询模型的方法可达到极小化极大最优率，并证明多个单独欠佳的模型输出组合后可超越最佳单一模型；自适应方法则借助Q学习为不同输入动态选择查询模型并决定停止时机。 |
| [^30] | [Mixing Regularization for Average-Reward $Q$-Learning and $Q$-Planning](https://arxiv.org/abs/2610.14988) | 该论文提出通过向一个与奖励无关的 $m$ 步混合策略进行正则化来控制平均奖励 $Q$-学习的收缩与探索性质，并给出有限时间收敛保证，在单步情形下匹配最佳已知速率，在 $m$ 步混合情形下首次推导出新的速率。 |
| [^31] | [Pragmatic DML with AI-Learned Representations](https://arxiv.org/abs/2610.01935) | 该论文揭示了AI学习表示的误差会以结果回归误差与平衡权重误差的乘积形式影响因果参数估计，并证明了交叉拟合DML可为依赖表示的目标提供有效推断，且与按折表示学习/微调兼容，为基于AI表示的因果推断提供了实用框架。 |
| [^32] | [Arbitrary-Accuracy Neural Approximation with Optimal Neuron Count and Optimal Bit Complexity](https://arxiv.org/abs/2609.35628) | 本文通过显式构造激活函数与闭式网络，证明了在 $[0,1]^d$ 上对连续函数进行任意精度逼近时 $d+1$ 个隐神经元是精确最优的，并给出了结合取整与指数函数的更简单构造及最优编码复杂度。 |
| [^33] | [Agnostic Smoothed Online Regression with Adversarial Responses](https://arxiv.org/abs/2609.32478) | 本文提出 Hedge-Cover 算法，利用协变量平滑性界定专家先验中的分歧次数，为具有有界对抗响应的不可知平滑在线回归实现了 $\widetilde{O}(\sqrt{\text{Pdim}(\mathcal{F}) \textsf{C}_{\textsf{cov}} T})$ 的次线性遗憾，从而解决了极小极大最优自适应遗憾这一开放问题。 |
| [^34] | [Prior-Amortized In-Context Bayesian Inference for Generalized Linear Mixed-Effects Models](https://arxiv.org/abs/2609.24422) | 提出metabeta——一个预训练神经网络，在测试时接受任意先验族和超参数输入，利用集合变换器与条件归一化流对GLMM进行先验摊销的上下文贝叶斯推断，无需MCMC即可零样本泛化并获得校准的后验估计。 |
| [^35] | [Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time](https://arxiv.org/abs/2609.22226) | 该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。 |
| [^36] | [Omni-modal decomposition autoencoders learn full-stack wearable disentangled representations](https://arxiv.org/abs/2608.07385) | 提出全模态变分分解自编码器（OmniDecVAEs），利用单一共享编码器-解码器和多视角自监督分解损失，从任意多模态可穿戴时间序列中学习解耦表征，在人体活动识别数据集上实现最高 3.0 个百分点的 DCI 解耦度提升。 |
| [^37] | [A Human-in-the-Loop Bayesian Optimization Framework for Constraint-Aware Bioprocess Development](https://arxiv.org/abs/2606.19230) | 本文提出了一种扩展的人机协同贝叶斯优化框架，通过将约束满足概率和鲁棒性能作为帕累托目标，使专家能交互式选择最优候选，从而在生物过程开发中兼顾约束与不确定性。 |
| [^38] | [Adaptive Inference for Resource-Constrained Dynamic Pricing](https://arxiv.org/abs/2606.03736) | 本文提出一种推断感知的重新求解策略，在资源受限动态定价中平衡收益与需求推断，实现接近最优的遗憾和信息效率。 |
| [^39] | [An interpretable semiparametric Bayesian machine learning model for high-dimensional circulating epigenetic signatures of multiple myeloma](https://arxiv.org/abs/2605.20143) | 该论文提出了一种可解释的半参数贝叶斯机器学习模型spBART，将线性协变量成分与稀疏树集成相结合，从多发性骨髓瘤患者的循环游离DNA 5hmC数据中筛选出八个关键基因位点（包括与IL-1/IL-6信号通路相关的IL1RAP），模型AUC达到0.96。 |
| [^40] | [Survey-aware Machine Learning: A Guideline for Valid Population Health Inference based on Scoping Review](https://arxiv.org/abs/2605.08963) | 本文提出了一个九步骤的感知调查设计机器学习（SaML）指南，通过将调查设计元数据（如抽样权重、分层变量等）纳入机器学习全生命周期，解决基于复杂健康调查数据训练模型时估计偏差、不确定性低估和公平性评估失真的问题。 |
| [^41] | [Information Theory and Statistical Learning](https://arxiv.org/abs/2605.02989) | 该论文章节简明阐述了散度度量在统计学习模型训练中的核心作用，示例涵盖从线性回归、变分自编码器到扩散模型等多种模型。 |
| [^42] | [Gradient Regularized Newton Boosting Trees with Global Convergence](https://arxiv.org/abs/2605.00581) | 本文提出受限牛顿下降理论框架，证明了牛顿提升树在光滑强凸损失下的线性收敛速率，并通过梯度正则化扩展实现了对一般凸损失函数的全局收敛保证。 |
| [^43] | [Learning Curves and Prediction Persistence to Under-Regularization of Spectral Algorithms in Large Dimensions](https://arxiv.org/abs/2604.23212) | 该文在大维度体制（$n\asymp d^\gamma$）下，以常数因子精度刻画了谱算法随正则化从最优尺度减弱直至插值时条件超额风险的阶数，揭示了风险何时保持极小极大阶、何时出现多项式量级更大的欠正则化风险区间。 |
| [^44] | [The Value of Information in Resource-Constrained Pricing](https://arxiv.org/abs/2603.24974) | 本文研究了资源受限动态定价中的信息价值，在完全信息、无信息和知情定价三种信息环境下分别设计了算法：完全信息下仅求解一次流体问题的残差修正策略达到 O(log T) 后悔值，无信息下确定性探索加确定性等价策略达到 O(√T) 后悔值，知情定价下反馈控制器在斜率矩阵对称时无需估计斜率即可学习最优实施价格。 |
| [^45] | [Continuous-time learning of probability distributions: a case study in a digital trial of young children with type 1 diabetes](https://arxiv.org/abs/2603.24427) | 提出一个基于高斯混合模型与神经常微分方程的连续时间概率分布学习框架，用于刻画1型糖尿病儿童连续血糖监测数据中血糖分布随时间的动态演变。 |
| [^46] | [On the Complexity of Offline Reinforcement Learning with Q*-Approximation and Partial Coverage](https://arxiv.org/abs/2602.12107) | 该论文通过信息论下界证明仅凭 Q\* 可实现性与贝尔曼完整性不足以实现部分覆盖下样本高效的离线强化学习，并提出一个通用的决策估计框架，将离线强化学习的复杂度分解为决策复杂度与价值估计误差两部分。 |
| [^47] | [Wedge Sampling: Efficient Tensor Completion with Nearly-Linear Sample Complexity](https://arxiv.org/abs/2602.05869) | 本文提出“楔形采样”这一非自适应采样方案，通过对张量展开同列元素进行成对采样以提供谱初始化所需的行相关性，从而在固定秩条件下以近线性样本复杂度实现低秩张量的多项式时间恢复。 |
| [^48] | [Anytime-Valid Conformal Risk Control](https://arxiv.org/abs/2602.04364) | 本文提出了任意时刻有效的保形风险控制方法，将误差控制从固定大小校准数据集上的平均有效性扩展为随校准数据累积增长时在任意时刻均以高概率保持有效，并证明了该保证渐近紧致且修正速率不可再改进。 |
| [^49] | [Learning Multinomial Logits in $O(n \log n)$ time](https://arxiv.org/abs/2601.04423) | 本文针对多项Logit（MNL）模型的学习问题，提出了自适应与非自适应算法，能够以 $O(\frac{n\log n}{\varepsilon^2})$ 次查询使每个选择板上的诱导选择分布在总变差距离 $\varepsilon$ 内逼近真实分布。 |
| [^50] | [A Bayesian latent class reinforcement learning framework to capture adaptive, feedback-driven travel behaviour](https://arxiv.org/abs/2512.14713) | 本文提出了一种基于贝叶斯潜在类别强化学习的框架，能够捕捉出行行为中个体偏好随时间演变的异质性，并通过驾驶模拟数据识别出三种不同适应策略的旅行者类型。 |
| [^51] | [Sparse corruption in low-rank matrix inference: the PCA benchmark](https://arxiv.org/abs/2511.11927) | 该论文研究了稀疏噪声（建模为有限平均连通度的加权图邻接矩阵）损坏下低秩信号的PCA推断问题，利用复制方法与种群动力学解析计算了最大特征值、特征向量分量密度及信号重叠度，并识别出随图连通度变化的两个信号强度转变点。 |
| [^52] | [Survival of the fittest Cox model: Pivotal variable selection for time-to-event data](https://arxiv.org/abs/2510.19374) | 本文通过对Cox比例风险模型的偏似然函数进行平方根变换，使正则化参数选择具有渐近枢轴性（精确消除基线风险和删失率的影响），从而显著提升了生存分析中变量选择的准确性和可靠性。 |
| [^53] | [TabPFN: One Model to Rule Them All?](https://arxiv.org/abs/2505.20003) | 本文从统计学视角将TabPFN阐释为一种近似贝叶斯推断方法，并证明这一表格数据基础模型在半监督参数估计、协变量偏移预测和异质性处理效应等任务上有时能超越专门的先进方法。 |
| [^54] | [tempdisagg: A Python Framework for Temporal Disaggregation of Time Series Data](https://arxiv.org/abs/2503.22054) | tempdisagg是一个开源Python框架，实现了Chow-Lin、Denton等经典时间分解与基准调整方法，并通过基于三角分解的统一GLS计算引擎将似然评估的计算复杂度从O(n²n_l)降至O(n·n_l²)。 |
| [^55] | [Almost Sure Convergence Rates and Concentration of Stochastic Approximation and Reinforcement Learning with Markovian Noise](https://arxiv.org/abs/2411.13711) | 本文通过一种使用长度递减区间的平均ODE新颖离散化技术，首次为带马尔可夫噪声的压缩随机逼近算法建立了几乎必然收敛速率和亚Weibull尾部的最大集中性界，并首次应用于无基于计数学习率的Q学习及离策略时序差分学习。 |
| [^56] | [Designing Algorithmic Recommendations to Achieve Human-AI Complementarity](https://arxiv.org/abs/2405.01484) | 本文提出了一种基于因果推断潜在结果框架的算法设计方法，将算法推荐与工具变量相联系，无需对推荐效果做先验假设即可设计出实现人机互补的推荐算法。 |
| [^57] | [Active Learning in the Predict-then-Optimize Framework: A Margin-Based Approach.](http://arxiv.org/abs/2305.06584) | 本文提出了预测-优化框架中的首个主动学习方法，通过基于间隔的准则和SPO损失的可求解替代函数来决定是否请求样本标签，在提供理论泛化保证的同时有效降低标签复杂度。 |
| [^58] | [Variance Reduction for Matrix Computations with Applications to Gaussian Processes.](http://arxiv.org/abs/2106.14565) | 本文提出了一种通过矩阵分解来进行矩阵计算的方差缩减方法，并展示了在某些问题上可以实现任意更好的随机性能。此外，本文提出的矩阵乘积迹的分解估计器与对半正定矩阵行列式的对数估计器均有显著的效率提高，可以在实际的贝叶斯优化问题中得到应用。 |

# 详细

[^1]: 简单的隐私保护噪声SGD在总体强凸性下是极小极大最优的

    Simple Private Noisy SGD is Minimax-Optimal under Population Strong Convexity

    [https://arxiv.org/abs/2611.26515](https://arxiv.org/abs/2611.26515)

    本文证明了在强凸损失（尤其是强凸广义线性模型，如隐私逻辑回归）上，采用固定步长的单次噪声SGD即可达到极小极大最优的估计误差率，其上界与基于Score Attack框架推导的隐私下界相匹配，并首次消除了对数因子。

    

    随机梯度下降（SGD）是现代机器学习的核心，并且在许多回归场景中已被证明具有统计上的极小极大最优性，例如凸的广义线性模型（GLMs）。然而，尚不清楚其最简单的差分隐私版本——噪声SGD（Noisy SGD）——即使在强凸损失函数上是否也能享有同样良好的保证。我们对这一问题给出了肯定的答案。我们在紧凸域上的光滑、强凸且满足Lipschitz条件的损失函数下，在Rényi差分隐私框架内，针对采用固定步长调度和恒定批量大小的单次噪声SGD，给出了参数估计误差的上界。当专门应用于强凸GLMs时，我们的误差上界与我们基于Score Attack框架推导出的极小极大下界相匹配。特别地，这为隐私保护的逻辑回归中的噪声SGD提供了首个极小极大最优的收敛速率，避免了现有结果中出现的多项式对数因子项。我们的分析方法具有广泛的通用性……

    arXiv:2611.26515v1 Announce Type: cross  Abstract: Stochastic gradient descent (SGD) is at the center of modern machine learning and is known to be statistically minimax-optimal in many regression settings, e.g., for convex Generalized Linear Models (GLMs). Yet, it is unknown whether its simplest differentially private Noisy SGD counterpart fulfills similar favorable guarantees even on strongly convex losses. We resolve this question affirmatively. We upper bound the parameter estimation error for a single pass of Noisy SGD with a fixed stepsize schedule and constant batch sizes on smooth and strongly convex Lipschitz losses over a compact convex domain under Renyi DP. Specialized to strongly convex GLMs, our error bounds match minimax lower bounds we derive based on the Score Attack framework. In particular, this provides the first minimax-optimal rate for Noisy SGD in private logistic regression, avoiding poly-logarithmic terms present in existing results. Our analysis is versatile a
    
[^2]: 摊销式搜索空间上的贝叶斯优化

    Bayesian optimization on amortized search space

    [https://arxiv.org/abs/2611.26459](https://arxiv.org/abs/2611.26459)

    该论文提出了一种利用机理模拟器的摊销式逆映射来约束搜索空间的贝叶斯优化方法，通过感知问题几何的高斯过程核高效利用问题的低内在维度，并给出了依赖内在维度的遗憾率理论保证，成功应用于帕金森病的药物剂量个性化。

    

    在现实世界的黑盒优化问题中，通常存在一个机理模拟器，它能够以被优化参数的函数形式近似预测系统的行为。本文展示了如何利用此类模拟器的摊销式逆映射来限制贝叶斯优化的搜索空间，同时使用能够感知原始问题几何结构的高斯过程核。这使得我们能够高效地利用优化问题的低内在维度。我们刻画了该方法的近似偏差和信息增益，并在额外的几何与高斯过程假设下，推导出了依赖于内在维度的遗憾率。我们以帕金森病的药物剂量个性化作为应用实例来说明该方法。我们使用药物的药代动力学/药效动力学（PK/PD）模型作为模拟器，这是一个基于群体数据校准并量化了参数不确定性的机理模型。

    arXiv:2611.26459v1 Announce Type: cross  Abstract: In real-world black-box optimization problems, a mechanistic simulator is often available that approximately predicts the behavior of the system as a function of the parameters being optimized. Here, we show how an amortized inverse map of such a simulator can be used in Bayesian optimization to restrict the search space while using a Gaussian process kernel that is aware of the original problem geometry. This allows for efficiently leveraging the low intrinsic dimensionality of the optimization problem. We characterize its approximation bias and information gain, and derive an intrinsic-dimension-dependent regret rate under additional geometric and Gaussian Process assumptions. We illustrate the method with an application to drug dosing personalization in Parkinson's disease. We use, as a simulator, a pharmacokinetic/pharmacodynamic (PK/PD) model of a drug, a mechanistic model calibrated on population data with quantified parameter un
    
[^3]: 基于特征函数的精确贝叶斯层

    Exact Bayesian layers via characteristic functions

    [https://arxiv.org/abs/2611.26426](https://arxiv.org/abs/2611.26426)

    本文提出CF-Prop方法，利用特征函数无采样地精确计算贝叶斯ReLU层的输出分布（即使在非高斯情形下），并能对高斯、泊松、伽马等多种似然闭式计算证据下界，从而实现高效的变分推断算法CF-VI，在CIFAR-100上取得最低校准误差。

    

    贝叶斯神经网络能够提供有原则的不确定性估计，但使用蒙特卡洛采样进行训练的代价可能很高。无采样变分推断算法使训练贝叶斯神经网络变得可行，但现有方法假设输出分布为高斯分布，这在实际应用问题中可能并不准确。本文提出了CF-Prop，一种不确定性传播方法，它利用特征函数来精确计算贝叶斯ReLU层的输出分布，无需任何采样，即使在非高斯情形下同样适用。由于CF-Prop传播的是特征函数，证据下界——变分训练的目标函数——可以对一大类似然函数（包括高斯、泊松和伽马分布）以闭式形式计算。这使得可以实现一种高效的变分推断算法CF-VI。在CIFAR-100上，CF-VI在所有受测方法中取得了最低的校准误差。

    arXiv:2611.26426v1 Announce Type: cross  Abstract: Bayesian neural networks provide principled uncertainty estimates, but training them with Monte Carlo sampling can be expensive. Sampling-free variational inference algorithms make training Bayesian neural networks feasible, but existing approaches assume a Gaussian output distribution, which can be inaccurate in applied problems. Here, we introduce CF-Prop, an uncertainty propagation method that uses characteristic functions to compute the exact output distribution of a Bayesian ReLU layer, without any sampling, even in non-Gaussian settings. Because CF-Prop propagates characteristic functions, the evidence lower bound---the objective of variational training---can be computed in closed form for a broad class of likelihoods, including the Gaussian, Poisson, and Gamma. This allows to implement an efficient resulting variational inference algorithm, CF-VI. On CIFAR-100, CF-VI achieves the lowest calibration error of any tested method, im
    
[^4]: 基于病态随机权重神经网络筛的统计推断

    Statistical Inference with Ill-Conditioned Random Weight Neural Network Sieves

    [https://arxiv.org/abs/2611.26410](https://arxiv.org/abs/2611.26410)

    本文证明了随机权重神经网络筛在维数增大时必然出现病态性（最小特征值消失），并在显式处理该病态性的基础上建立了相应的逼近速率、估计与统计推断理论。

    

    随机权重神经网络（RWNN）是一类浅层网络，其隐藏层参数为随机生成并保持固定，仅对输出系数进行估计。在给定采样参数的条件下，RWNN回归可视为一个具有随机逼近空间的线性筛问题。我们证明，随着维数的增长，这类随机筛无法始终保持一致的良好条件性。对于满足多项式覆盖条件与Lipschitz条件的参数化特征族，我们证明了总体协方差矩阵的特征值具有由参数空间覆盖几何结构决定的多项式上界，从而导致最小特征值趋于消失。我们发展了显式处理这种病态性的逼近、估计与推断理论。在β-Hölder源条件下，我们同时获得了快于蒙特卡洛速率的L²逼近速率，以及基于……（原文摘要在此处截断）的新L^∞逼近界。

    arXiv:2611.26410v1 Announce Type: cross  Abstract: Random weight neural networks (RWNNs) are shallow networks whose hidden-layer parameters are randomly generated and then kept fixed, while only the output coefficients are estimated. Conditional on the sampled parameters, RWNN regression is a linear sieve problem with random approximation space. We show that such random sieves cannot remain uniformly well-conditioned as their dimension grows. For parametrized feature families satisfying polynomial covering and Lipschitz conditions, the eigenvalues of the population covariance matrix are proven to have polynomial upper bounds determined by the covering geometry of the parameter space, leading to a vanishing smallest eigenvalue. We develop approximation, estimation, and inference theory that explicitly treats the ill-conditioning. Under a $\beta$-H\"older source condition, we obtain both faster-than-Monte-Carlo $L^2$ approximation rates and new $L^\infty$ approximation bounds based on st
    
[^5]: 来自非高斯源的流匹配：高效采样理论

    Flow Matching from Non-Gaussian Sources: A Theory of Efficient Sampling

    [https://arxiv.org/abs/2611.26407](https://arxiv.org/abs/2611.26407)

    本文为从非高斯源分布（如重尾和多峰分布）出发的流匹配建立了首个非渐近收敛理论，证明欧拉离散化采样器在温和条件下仅需 $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ 次迭代即可达到 $\varepsilon$ 级总变差误差。

    

    流匹配是一种多功能的生成建模框架，它将一个源分布传输至目标数据分布。其关键优势之一在于源分布选择的灵活性，例如可以使用重尾分布和多峰分布作为源。然而，针对非高斯源的有限步采样保证仍然十分有限。本文为采用线性插值的流匹配的欧拉离散化建立了非渐近收敛理论。假设定义在 $\mathbb{R}^d$ 上的源分布具有处处为正、连续可微的密度函数、有限的总方差 $\mathsf{TotVar}_{\mathsf{base}}$，以及 $L$-Lipschitz 的得分函数。在使用总体速度场和适当的时间网格时，只要目标分布满足温和的二阶矩条件，该采样器即可在 $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ 次迭代内实现至多 $\varepsilon$ 的总变差误差。这些假设涵盖……（摘要原文在此处截断）

    arXiv:2611.26407v1 Announce Type: cross  Abstract: Flow matching is a versatile framework for generative modeling that transports a source distribution to a target data distribution. A key advantage is its flexibility in the choice of source distribution, such as heavy-tailed and multimodal sources. Yet finite-step sampling guarantees for non-Gaussian sources remain limited. We establish a nonasymptotic convergence theory for Euler discretization of flow matching with linear interpolation. Suppose the source distribution on $\mathbb{R}^d$ has an everywhere positive, continuously differentiable density, finite total variance $\mathsf{TotVar}_{\mathsf{base}}$, and an $L$-Lipschitz score. Using the population velocity field and a suitable time grid, this sampler achieves total variation error at most $\varepsilon$ in $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ iterations, provided the target distribution satisfies a mild second-moment condition. These assumptions cover,
    
[^6]: 分层数据流的保形选择

    Conformal Selection for Hierarchical Streaming Data

    [https://arxiv.org/abs/2611.26328](https://arxiv.org/abs/2611.26328)

    该论文提出了面向分层数据流的无分布在线保形选择方法OHS及其带反馈变体OHSF，通过构建组间和组内的基于秩的e值并结合在线e-BH程序，将在线保形选择框架扩展到超越独立同分布/可交换数据假设的分层结构场景。

    

    许多现实世界的任务需要在感兴趣的结果被揭示之前，对是否选择候选对象做出序贯决策。在这项工作中，我们研究了分层数据背景下的在线选择问题，其中候选对象依次来自不同的总体。这将现有的在线保形选择框架扩展到了标准的独立同分布/可交换数据假设之外。主要挑战在于如何在分层数据结构下为每个候选对象序贯地构建有效的统计量。为此，我们首先开发了在线分层选择（OHS），这是一种无分布方法，能够对分层数据流进行有原则的候选选择。具体而言，我们提出了为组间和组内样本构建基于秩的e值的方法，并将平均e值应用于在线e-BH程序。随后，我们开发了带反馈的OHS（OHSF），该方法利用反馈信息提升了选择的性能。

    arXiv:2611.26328v1 Announce Type: cross  Abstract: Many real-world tasks involve making sequential decisions about whether to select candidates before their responses of interest are revealed. In this work, we address the online selection problem in the context of hierarchical data, where candidates arrive sequentially from different populations. This extends existing online conformal selection frameworks beyond the standard i.i.d./exchangeable data assumption. The primary challenge lies in constructing valid statistics for each candidate sequentially under the hierarchical data structure. To this end, we first develop Online Hierarchical Selection (OHS), a distribution-free method that enables principled candidate selection for hierarchical data streams. Specifically, we introduce methods to construct rank-based e-values for both between-group and within-group samples, and apply averaged e-values to the online e-BH procedure. We then develop OHS with feedback (OHSF), which boosts the 
    
[^7]: 大语言模型训练数据审计的统计框架

    A Statistical Framework for Training Data Auditing of Large Language Models

    [https://arxiv.org/abs/2611.26264](https://arxiv.org/abs/2611.26264)

    本文提出了一个统一的统计框架，用于大语言模型训练数据审计，涵盖文档级成员推断、目标数据集全局审计和可疑子集识别，能够有效聚合文档内局部成员信号并进行整体评估。

    

    训练数据审计旨在判断感兴趣的数据是否被用于训练大语言模型（LLM），这一问题对于隐私审计、版权保护和基准污染检测至关重要。这些问题在预训练中使用受版权保护的书籍和科学论文，以及在微调中使用合成文本和医疗问答记录时尤为突出。在实践中，训练成员信号可能仅集中在长文档的少数段落中，因此在每个文档内聚合证据而不稀释这些局部信号至关重要。除了单个文档之外，审计目标数据集还需要基于文档级证据进行整体评估。在本文中，我们提出了一个统一的统计框架，用于文档级成员推断、目标数据集全局审计以及可疑子集识别。我们整合了互补的块级成员信号（摘要在此处被截断）

    arXiv:2611.26264v1 Announce Type: cross  Abstract: Training-data auditing asks whether data of interest were used to train a large language model (LLM), a question central to privacy auditing, copyright protection, and benchmark-contamination detection. These concerns are particularly relevant to the use of copyrighted books and scientific papers in pre-training, as well as synthetic text and medical question-answer records in fine-tuning. In practice, training membership signals may be concentrated in only a few passages of a long document, making it important to aggregate evidence within each document without diluting these localized signals. Beyond individual documents, auditing a target collection requires an overall assessment based on the document-level evidence. In this paper, we propose a unified statistical framework for document-level membership inference, target-collection global auditing, and suspicious-subset identification. We integrate complementary chunk-level membershi
    
[^8]: 基于局部因果父节点发现的时间序列结构学习

    Time Series Structure Learning with Local Causal Parent Discovery

    [https://arxiv.org/abs/2611.26232](https://arxiv.org/abs/2611.26232)

    该论文提出ITPD——一种可靠且完备的基于约束的因果结构学习方法，通过局部到全局的因果父节点发现来学习具有因果非平稳性的多变量时间序列结构，并借助时间结构先验与图划分将高阶条件独立性检验替换为低阶检验，从而降低计算与样本复杂度。

    

    时间序列上的学习与推断会带来重大的统计挑战，包括观测值的自相关性、随时间变化的分布偏移以及维数灾难。这些挑战使因果发现算法难以投入实际应用，而这类算法在一般情况下本已存在高计算复杂度与高样本复杂度的问题。为缓解这些问题，本工作提出了迭代时间父节点发现（ITPD）：一种针对具有因果非平稳性的多变量时间序列的可靠且完备的基于约束的因果结构学习程序。ITPD 采用局部到全局的学习方法，通过依次发现每个变量的父节点集合来构建全局图。ITPD 利用时间结构先验、图划分和局部马尔可夫条件，将高阶条件独立性检验替换为低阶替代检验。在精确神谕（oracle）下，ITPD 执行了最少次数的……

    arXiv:2611.26232v1 Announce Type: new  Abstract: Learning and inference on time series can pose significant statistical challenges, including autocorrelated observations, distribution shift over time, and the curse of dimensionality. These challenges complicate the practical application of causal discovery algorithms, which already suffer from high computational and sample complexity in the general case. To mitigate these issues, this work introduces Iterative Temporal Parent Discovery (ITPD): a sound and complete constraint-based procedure for causal structure learning on multivariate time series with causal nonstationarity. ITPD takes a local-to-global learning approach that discovers the global graph by sequentially discovering the parent set for each variable. ITPD exploits temporal structural priors, graph partitioning, and the local Markov condition to swap higher-order conditional independence tests for lower-order alternatives. With an exact oracle, ITPD performed the least num
    
[^9]: 注意力头特化：样本复杂度中的相变

    Attention Head Specialization: Phase Transitions in Sample Complexity

    [https://arxiv.org/abs/2611.26228](https://arxiv.org/abs/2611.26228)

    该论文证明注意力头特化是样本复杂度中真正的相变，在可求解的多头注意力模型中表现为测试误差的非解析性，且随数据量增加可通过级联相变逐步恢复更弱的潜在特征，形成统计意义上的简单性偏置。

    

    注意力头在训练过程中会特化为不同的功能，往往随模型行为的急剧变化而分阶段涌现。虽然这一现象大多被作为训练时间的函数来研究，我们则探究注意力头特化如何依赖于可用训练数据的数量。在实证方面，我们发现BERT中特化于句法关系的注意力头会随数据集规模的增大而涌现。随后，我们在一个可求解的多头注意力高维模型中研究这一现象，并证明在该模型中，注意力头特化是样本复杂度意义上真正的相变，对应于测试误差及其他可观测量中的非解析性。根据数据结构和正则化的不同，特化可以跟随无信息量的头分化而发生，或通过一系列级联相变逐步恢复较弱的潜在特征，从而提供了一种统计形式的简单性偏置。

    arXiv:2611.26228v1 Announce Type: new  Abstract: Attention heads specialize into distinct functions during training, often emerging in stages together with sharp changes in model behavior. While this phenomenon has mostly been studied as a function of training time, we ask how head specialization depends on the amount of available training data. Empirically, we find that heads specialized to syntactic relations in BERT emerge as the dataset size increases. We then study this phenomenon in a solvable high-dimensional model of multi-head attention and establish that, in this model, head specialization is a genuine phase transition in sample complexity, corresponding to a non-analyticity in the test error and other observables. Depending on data structure and regularization, specialization can follow uninformative head differentiation or occur through a cascade of phase transitions that progressively recover weaker latent features, providing a statistical form of simplicity bias. Finally,
    
[^10]: 用于密度比估计的Hellinger决策树：有限样本保证

    Hellinger Decision Trees for Density Ratio Estimation: Finite-Sample Guarantees

    [https://arxiv.org/abs/2611.25936](https://arxiv.org/abs/2611.25936)

    提出了一种通过牛顿提升在轴对齐决策树集成上最小化平衡Hellinger风险来直接估计对数密度比的方法，并建立了有限样本保证，其收敛速率在对数因子范围内达到极小极大最优的L²速率。

    

    密度比在双样本比较、分布偏移适应以及生成模型评估中起着核心作用。然而，通过对底层密度分别进行估计来估计密度比可能是不必要的困难。在本文中，我们开发了一种直接估计对数密度比的方法，即在轴对齐决策树的加性集成上最小化平衡的Hellinger风险，并使用牛顿提升进行训练。我们为该算法产生的估计器建立了有限样本保证，并推导出了显式的收敛速率。在适当的尾部条件和Hölder光滑性条件下，再加上关于每次提升迭代所取得进展的额外假设，我们的速率在忽略对数因子的情况下与函数估计的经典极小极大最优 $\mathscr{L}^{2}$ 速率相匹配。一个关键的技术贡献是利用特殊的指数结构进行的优化分析。

    arXiv:2611.25936v1 Announce Type: cross  Abstract: Density ratios play a central role in two-sample comparison, distribution-shift adaptation, and generative-model evaluation. However, estimating a density ratio through separate estimation of the underlying densities can be unnecessarily challenging. In this paper, we develop a direct approach to the log density ratio estimation by minimizing a balanced Hellinger risk over additive ensembles of axis-aligned decision trees, trained using Newton boosting. We establish finite-sample guarantees for the estimator produced by the algorithm and derive an explicit convergence rate. Under suitable tail and H\"{o}lder smoothness conditions, together with an additional assumption on the progress made at each boosting iteration, our rate matches, up to logarithmic factors, the classical minimax-optimal $ \mathscr{L}^{2} $ rate for function estimation. A key technical contribution is an optimization analysis that exploits the special exponential st
    
[^11]: 在低推理预算下缩小扩散语言模型的性能差距

    Closing Performance Gap for Diffusion Language Models under Low Inference Budgets

    [https://arxiv.org/abs/2611.25763](https://arxiv.org/abs/2611.25763)

    提出了无需训练的 CaRR 框架，通过 token 承诺重掩码、更新上下文回填和基于轨迹的恢复机制，在低推理预算下提升可编辑扩散语言模型的生成质量，从而缩小其性能差距。

    

    扩散语言模型通过并行更新多个 token，为低预算生成提供了一条架构层面的途径，与逐 token 的自回归解码相比，可以减少顺序解码步骤的数量。然而，在低预算下实现这一优势颇具挑战，因为当去噪和细化步骤的数量减少时，生成质量往往会下降。可编辑扩散模型通过草稿后细化来应对这种质量损失，但在最终提交之前，有用的中间结果可能仍然会丢失。我们提出了 CaRR（Commitment-aware Refinement and Recovery，承诺感知的细化与恢复），这是一个无需训练的框架，它在现有的前向调度内，通过更新上下文回填和基于轨迹的恢复，从固定的可编辑检查点提升提交质量。其中 Token Commitment Remasking（TCR，token 承诺重掩码）会对选定的原生纯替换进行重掩码，并在下一个预定的编辑前向中使用更新后的内容进行回填。

    arXiv:2611.25763v1 Announce Type: cross  Abstract: Diffusion language models offer an architectural route to low-budget generation by updating multiple tokens in parallel, which can reduce the number of sequential decoding steps relative to token-by-token autoregressive decoding. However, realizing this advantage at low budgets is challenging, as generation quality often declines when the number of denoising and refinement steps is reduced. Editable diffusion models address this quality loss through post-draft refinement, but useful intermediate results may still be lost before final submission. We introduce CaRR (Commitment-aware Refinement and Recovery), a training-free framework which improves submission quality from a fixed editable checkpoint through updated-context refill and trajectory-based recovery within its existing forward schedule. Token Commitment Remasking (TCR) remasks selected native pure replacements and refills them on the next scheduled edit forward using the update
    
[^12]: 差分隐私分数估计及其在基于分数的生成模型中的应用

    Differentially Private Score Estimation with Applications to Score-Based Generative Models

    [https://arxiv.org/abs/2611.25536](https://arxiv.org/abs/2611.25536)

    本文研究了差分隐私约束下的分数估计问题，针对具有局部 Hölder 连续分数的次高斯分布，提出了一种基于二次 B 样条和指数机制的采样式隐私分数估计器，并将其应用于隐私保护的基于分数的生成模型。

    

    估计一个分布的分数函数在统计学和机器学习中扮演着核心角色，其应用涵盖分布估计、采样以及基于分数的生成建模。然而，从敏感数据中学习分数会引发隐私问题。这些担忧在基于分数的生成建模中尤为突出，因为训练后的模型可能会记住个别训练样本，并通过成员推断和数据提取攻击泄露敏感信息。尽管非隐私的分数估计已被广泛研究，但其隐私版本（即隐私保护下的分数估计）的统计极限仍知之甚少。在这项工作中，我们研究了在 $(\alpha,\beta)$-差分隐私约束下，对具有局部 $\gamma$-Hölder 连续分数的 $d$ 维次高斯分布进行分数估计的问题。我们开发了一种基于采样的隐私分数估计器，该估计器使用了二次 B 样条和指数机制，并且我们实现了……

    arXiv:2611.25536v1 Announce Type: cross  Abstract: Estimating the score function of a distribution plays a central role in statistics and machine learning, with applications in distribution estimation, sampling, and score-based generative modeling. However, learning scores from sensitive data raises privacy concerns. These concerns are particularly pronounced in score-based generative modeling, where trained models may memorize individual training examples and expose sensitive information through membership inference and data extraction attacks.   Although non-private score estimation has been extensively studied, the statistical limits of its private counterpart remain less understood. In this work, we study score estimation under $(\alpha,\beta)$-differential privacy for $d$-dimensional sub-Gaussian distributions with locally $\gamma$-H\"older-continuous scores. We develop a sampling-based private score estimator using quadratic B-splines and the exponential mechanism, and we impleme
    
[^13]: 针对具有未知基测度的平滑对手的序贯概率赋值

    Sequential Probability Assignment against Smoothed Adversaries with Unknown Base Measure

    [https://arxiv.org/abs/2611.25514](https://arxiv.org/abs/2611.25514)

    本文在基测度 μ 未知且无需 Lipschitz 条件的更一般平滑在线学习设定下，证明了针对自适应型与疏忽型平滑对手的序贯概率赋值遗憾上界，其中自适应界由经验 ℓ∞ Hellinger 熵控制。

    

    平滑在线学习近年来被作为一种绕过完全对抗设定下难度结果的方法而受到研究，因为完全对抗设定可能过于悲观。在该框架中，对手被限制从相对于某个固定基测度 μ 具有有界密度的分布中生成上下文。先前的工作大多强假设学习者已知 μ，显著的例外包括 Block 等（2024）和 Blanchard（2025）。本文在 μ 未知的更一般设定下，研究使用平滑且良好指定数据的序贯概率赋值（即基于对数损失的在线学习），并且无需 Block 等（2024）与 Blanchard（2025）中所施加的 Lipschitz 条件。我们证明了针对自适应型与疏忽型平滑对手的遗憾上界：自适应情形的界由经验 ℓ∞ Hellinger 熵控制，而疏忽型情形的界改进了……

    arXiv:2611.25514v1 Announce Type: new  Abstract: Smoothed online learning has recently been studied as a way to bypass hardness results for the fully adversarial setting, which can be overly pessimistic. In this framework, the adversary is constrained to generate contexts from distributions having a bounded density with respect to some fixed base measure $\mu$. Prior work makes the strong assumption that $\mu$ is known to the learner, with notable exceptions including Block et al. (2024) and Blanchard (2025). In this paper, we study sequential probability assignment (a.k.a., online learning with log loss) with smooth, well-specified data in the more general setting where $\mu$ is \emph{unknown}, without the Lipschitz condition imposed in the work of Block et al. (2024) and Blanchard (2025). We prove regret upper bounds against both adaptive and oblivious smoothed adversaries: the adaptive bound is controlled by empirical $\ell_\infty$ Hellinger entropy, while the oblivious bound improv
    
[^14]: 面向删失生存时间数据的忠实生成建模：从分布保真度到统计推断

    Faithful Generative Modeling for Censored Time-to-Event Data: From Distributional Fidelity to Statistical Inference

    [https://arxiv.org/abs/2611.25464](https://arxiv.org/abs/2611.25464)

    该论文提出TSGSurv——一个通用的生存数据两阶段生成框架，在事件状态条件下联合生成观测时间和协变量，不仅建立了分布收敛速度，还为Cox比例风险模型下下游统计推断的可靠性提供了理论保证条件。

    

    合成数据已成为数据共享、临床试验模拟和数字孪生开发的有力工具，在生物医学研究和药物研发中的应用日益广泛。然而，针对生存时间数据的合成研究却鲜有关注，而这类数据正是临床试验和生存分析的核心。尤其是，合成生存数据分布保真度的理论保证在很大程度上仍属空白。更为重要的是，即使生成器达到了分布保真度，其下游统计分析是否可靠仍不清楚。为弥补这些空白，我们提出了一个通用的生存数据两阶段生成框架，该框架在事件状态的条件下联合生成观测时间和协变量，并建立了其分布收敛速度。聚焦于广泛使用的Cox比例风险模型，我们建立了生成器保真度（在此条件下……）（注：摘要原文在此处被截断）

    arXiv:2611.25464v1 Announce Type: cross  Abstract: Synthetic data has emerged as a powerful tool for data sharing, clinical trial simulation, and digital twin development, with growing applications in biomedical research and drug development. However, limited attention has been devoted to synthetic time-to-event data, which are central to clinical trials and survival analysis. In particular, theoretical guarantees for the distributional fidelity of synthetic survival data remain largely unexplored. More importantly, even when a generator achieves distributional fidelity, it remains unclear whether the downstream statistical analysis is reliable. To address these gaps, we propose a general two-stage generative framework for survival data (TSGSurv) that jointly generates observed times and covariates conditional on event status, and establish its distributional convergence rate. Focusing on the widely used Cox proportional hazards model, we establish conditions under which generator fide
    
[^15]: 差分隐私加-删除模型中的实例最优矩估计

    Instance-Optimal Moment Estimation in the Add-Remove Model of Differential Privacy

    [https://arxiv.org/abs/2611.25425](https://arxiv.org/abs/2611.25425)

    本文引入实例相关的误差度量并证明其严格下界，设计了在纯差分隐私加-删除模型下达到这些下界的矩估计机制，实现了比现有方法渐近更优的效用。

    

    在本文中，我们研究了在加-删除模型的纯差分隐私约束下估计经验均值、方差和协方差的问题。虽然现有机制能够达到最优的最坏情况误差，但它们对于单个数据集而言可能并非最优。为了衡量机制在单个数据集上的性能，我们引入了实例相关的误差度量，并证明了其严格下界。随后，我们设计了能够达到这些下界的机制，并证明与现有方法相比，这些机制获得了渐近更优的效用。

    arXiv:2611.25425v1 Announce Type: cross  Abstract: In this paper, we study the problem of estimating empirical means, variances, and covariances under pure differential privacy in the add-remove model. While existing mechanisms achieve optimal worst-case error, they may not be optimal for individual datasets. To measure the performance of mechanisms on individual datasets, we introduce instance-dependent error metrics and prove sharp lower bounds on them. Then, we design mechanisms that achieve these lower bounds and demonstrate that they achieve asymptotically better utility compared to existing methods.
    
[^16]: 具有假阳性错误控制的因果发现

    Causal Discovery with False Positive Error Control

    [https://arxiv.org/abs/2611.25376](https://arxiv.org/abs/2611.25376)

    提出m-保守PC算法，在线性高斯设定和极弱忠实性假设下实现了因果发现中错误发现率的有限样本控制，同时惩罚虚假邻接与错误边方向。

    

    因果发现方法产生关于因果关系的数据驱动假设，这些假设可能成为新的科学发现，或为下游干预决策提供依据。在此类场景中，假阳性的因果论断可能比假阴性造成更严重的后果。然而，鲜有方法针对因果发现中的假阳性错误控制，且现有的错误保证通常仅为渐近性的，并需要较强版本的忠实性假设。我们提出了 m-保守 PC 算法，并证明在线性高斯设定下，该算法在极弱版本的忠实性假设下即可控制错误发现率（FDR）。在我们的框架中，因果图的假阳性错误基于其所蕴含的条件独立性信息来定义，由此得到的度量能够同时惩罚虚假邻接关系和错误的边方向。参数 m 则量化了算法复杂度与保证强度之间的权衡。

    arXiv:2611.25376v1 Announce Type: cross  Abstract: Causal discovery methods produce data-driven hypotheses about causal relationships, which may become new scientific findings or inform decisions about downstream interventions. In such settings, false positive causal claims can be more consequential than false negatives. Yet few methods target false positive error control in causal discovery, and existing error guarantees are often asymptotic and require strong versions of the faithfulness assumption. We introduce the $m$-conservative PC algorithm and prove that it controls false discovery rate in the linear Gaussian setting under an extremely weak version of the faithfulness assumption. In our framework, false positive error for causal graphs is defined based on the conditional-independence information they entail, yielding a metric that penalizes both false adjacencies and incorrect edge orientations. The parameter $m$ quantifies a trade-off between algorithmic complexity and the str
    
[^17]: 标签噪声下二分类的基础理论

    Fundamentals for Binary Classification Under Label Noise

    [https://arxiv.org/abs/2611.25372](https://arxiv.org/abs/2611.25372)

    本章系统综述了标签噪声下二分类学习的基础理论，在 PAC 框架下形式化了随机分类噪声、Massart 噪声和恶意分类噪声三种模型，并阐明了经验风险最小化的适用条件以及样本复杂度与噪声率之间的关系。

    

    本章综述了标签噪声下二分类的基础性研究成果。标签噪声下的二分类是指在一部分类别标签不正确的数据集上训练模型，这会导致模型性能下降和误分类问题。我们区分了投毒攻击与噪声过程，并在可能近似正确（PAC）框架下形式化了随机分类噪声、Massart 噪声和恶意分类噪声这三种噪声模型，它们刻画了不同的对抗强度和数据生成场景。随后，我们重点介绍了一些核心思想与代表性结果。特别地，我们解释了为什么随机分类噪声能够保持假设之间的期望排序，从而支持基于经验风险最小化（ERM）原则的学习；样本复杂度如何依赖于噪声率；以及为什么学习算法通常需要对污染水平设定一个上界，还有如何成功地……（原文摘要在此处被截断）

    arXiv:2611.25372v1 Announce Type: cross  Abstract: This chapter surveys fundamental results for binary classification under label noise. Binary classification under label noise involves training models on datasets where class labels are partially incorrect, leading to degraded performance and misclassification. We distinguish poisoning attacks from noise processes and formalize random classification noise, Massart noise, and malicious classification noise within the probably approximately correct (PAC) framework. These models capture different adversarial strengths and data-generation realities. We then highlight some core ideas and representative results. In particular, we explain why random classification noise preserves the expected ordering of hypotheses and thus supports learning using the empirical risk minimization (ERM) principle, how sample complexity depends on the noise rate, and why learning algorithms often require an upper bound on the corruption level, as well as how suc
    
[^18]: 终身调整成本约束下的规定贝叶斯效率

    Prescribed Bayesian Efficiency under Lifetime Adjustment Costs

    [https://arxiv.org/abs/2611.25311](https://arxiv.org/abs/2611.25311)

    该论文刻画了在终身调整成本约束下，贝叶斯学习者在时间范围 T=2^J 处同时实现累积期望超额损失 O(G(J)) 与有限期望终身费用的可达性准则，即跨动作距离尺度上带前缀约束的跟踪误差配额分配的有限性。

    

    贝叶斯学习者可能在进行后验更新的同时保留已发出的动作，在预测损失与调整费用之间进行权衡。我们假设存在总体曲率、边界控制、关于最优动作的期望后验不确定性的 1/n 阶下界，以及一个可观测的无偏梯度见证，其平方范数一致可积。样本损失可以是无界的或非凸的。累积期望超额损失衡量相对于当前预测贝叶斯动作的损失增加。对于满足 c(0)=0 的有界非负非递减费用 c，以及满足 G(J)=o(J) 的非递减剖面 G: ℕ → [1,∞)，我们刻画了在时间范围 T=2^J 下累积期望超额损失 O(G(J)) 与有限期望终身费用的联合可达性。这里 J 是对数时间的索引。判据是跨动作距离尺度上带前缀约束的跟踪误差配额分配的有限性，等价地……（原文在此处截断）

    arXiv:2611.25311v1 Announce Type: new  Abstract: A Bayesian learner may retain an issued action while updating its posterior, trading predictive loss against adjustment charges. We assume population curvature, boundary control, an order-$1/n$ lower bound on expected posterior uncertainty about the optimal action, and an observable unbiased gradient witness whose squared norms are uniformly integrable. Sample losses may be unbounded or nonconvex. Cumulative expected excess measures the loss increase relative to current predictive Bayes actions. For bounded nonnegative nondecreasing charges $c$ with $c(0)=0$ and nondecreasing profiles $G: \mathbb{N} \to [1,\infty)$ with $G(J)=o(J)$, we characterize joint attainability of cumulative expected excess $O(G(J))$ at horizons $T=2^J$ and finite expected lifetime charge. Here $J$ indexes logarithmic time. The criterion is finiteness of a prefix-constrained allocation of tracking-error allowances across action-distance scales, equivalently its de
    
[^19]: 处理缺失协变量的统计学、机器学习与深度学习方法：在阿尔茨海默病研究中的实证评估

    Statistical, Machine Learning and Deep Learning Methods for Missing Covariates: An Empirical Evaluation with Applications to Alzheimer's Disease Research

    [https://arxiv.org/abs/2611.25219](https://arxiv.org/abs/2611.25219)

    本研究对八种处理缺失协变量的统计、机器学习和深度学习方法进行了系统性实证评估，在阿尔茨海默病研究背景下比较了它们在不同缺失机制下的表现。

    

    缺失数据在生物医学研究中十分常见，尤其是当生物标志物成本高昂或需要侵入性操作才能获取时，例如阿尔茨海默病研究中的脑脊液指标。尽管目前存在从传统统计方法到现代机器学习（ML）和深度学习（DL）算法的多种缺失数据处理方法，但它们在不同缺失机制下的比较性能仍缺乏充分了解。在本研究中，我们对八种处理缺失协变量的方法进行了实证评估，包括完整病例分析、多重插补、基于机器学习的技术，以及多层感知机、变分自编码器和生成对抗网络等深度学习框架。我们总结了这些方法的概念基础、实际应用考量、优势与局限性。通过大量的模拟实验，我们评估了各方法的估计偏差、效率、标准误准确性等性能表现。

    arXiv:2611.25219v1 Announce Type: cross  Abstract: Missing data are common in biomedical research, particularly when biomarkers are costly or require invasive procedures to obtain, such as cerebrospinal fluid measures in Alzheimer's disease. Although a wide range of missing data methods exists, from traditional statistical approaches to modern machine learning (ML) and deep learning (DL) algorithms, their comparative performance under different missingness mechanisms remains insufficiently understood. In this study, we provide an empirical evaluation of eight methods for handling missing covariates, including complete-case analysis, multiple imputation, ML-based techniques, and DL frameworks such as multilayer perceptrons, variational autoencoders, and generative adversarial networks. We summarize their conceptual foundations, practical implementation considerations, strengths, and limitations. Through extensive simulations, we assess estimation bias, efficiency, standard error accurac
    
[^20]: 组数多于维数的高斯混合聚类问题的低度样本复杂度

    The low-degree sample complexity of clustering Gaussian mixtures with more groups than dimensions

    [https://arxiv.org/abs/2611.25163](https://arxiv.org/abs/2611.25163)

    该论文在组数大于维数的设定下，通过低度多项式的下界与匹配的上界，首次精确刻画了高斯混合聚类所需的最小间隔与样本复杂度，例如当 K=d^{p/2}（p>2）时证明需要量级为 K³d^((⌈p⌉−p)/2) 的样本量。

    

    我们研究了当组数 $K$ 远大于维数 $d$ 时，对从 $\mathbb{R}^d$ 中 $K$ 个各向同性高斯分布的混合中抽取的 $n$ 个观测进行聚类的计算极限。在这一机制下，基于距离的方法和谱方法需要量级为 $\sqrt{d}$ 的平方间隔 $\Delta^2$；相比之下，基于张量的方法可以在多对数级间隔下完成聚类，但需要 $dK$ 的大多项式量级的样本量，而目前已知的针对随机中心的低度下界仅表明 $n$ 必须超过 $dK^2$。我们聚焦于正态分布的中心，通过一个低度下界以及在亚多项式因子意义下匹配的上界，刻画了在 $K>d$ 机制下使用低次多项式进行聚类所需的最小间隔。特别地，对于 $K=d^{p/2}$（$p>2$ 为实数），我们证明了恢复聚类结构需要量级为 $K^3 d^{(\lceil p\rceil-p)/2}$ 的样本量。

    arXiv:2611.25163v1 Announce Type: cross  Abstract: We study the computational limits of clustering $n$ observations drawn from a mixture of $K$ isotropic Gaussian distributions in $\mathbb R^d$, when the number $K$ of groups is much larger than the dimension $d$. In this regime, distance-based and spectral procedures require a squared separation $\Delta^2$ of order $\sqrt d$. In contrast, tensor-based procedures cluster at a polylogarithmic separation but need a sample size that is a large polynomial of $dK$, and the low-degree lower bounds known for random centers only show that $n$ must exceed $dK^2$.   Focusing on normally distributed centers, we characterize the minimal separation for clustering with low-degree polynomials in the regime $K>d$, through a low-degree lower bound and a matching upper bound, up to subpolynomial factors. In particular, for $K=d^{p/2}$ with $p> 2$ a real number, we show that a sample size of the order of $K^3 d^{(\lceil p\rceil-p)/2}$ is required to recov
    
[^21]: HapBO：通过专家自然语言先验实现的 α-π 贝叶斯优化

    HapBO: Harnessed $\alpha$-$\pi$ Bayesian Optimization via Expert Natural Language Priors

    [https://arxiv.org/abs/2611.25100](https://arxiv.org/abs/2611.25100)

    该论文提出HapBO方法，将从人类专家或大语言模型一次性引出的自然语言先验确定性地编译为不限于高斯族的可重用信念函数，从而提升贝叶斯优化的样本效率，并对可能误导的信息保持鲁棒。

    

    贝叶斯优化（BO）在获得关于问题的先验知识时可以显著提升样本效率。然而在实践中，这类知识往往只能以自然语言的形式获得，例如专家直觉或文档，而非先验引导的BO方法所要求的显式概率分布图景。有效利用这类知识既需要从语言到先验的富有表现力的映射，也需要对可能产生误导的信息保持鲁棒性。我们提出HapBO（Harnessed α-π 贝叶斯优化），它从人类专家、大语言模型（LLM）或两者中一次性引出信念，并将结构化规范确定性地编译为可重用的信念函数（π: X → [0,1]），其表示形式丰富，不限于高斯分布族。基于 α-πBO，HapBO 通过在任意非负[函数上]施加有界正乘数来应用这一信念函数……

    arXiv:2611.25100v1 Announce Type: new  Abstract: Bayesian Optimization (BO) can be substantially more sample-efficient when informed by prior knowledge about the problem. In practice, however, such knowledge is often available only in natural language, e.g., expert intuitions or documentation, rather than as the explicit probability distribution landscape that prior-guided BO methods may require. Leveraging it effectively demands both an expressive mapping from language to priors and robustness to information that may be misleading. We propose HapBO (Harnessed $\alpha$-$\pi$ Bayesian Optimization), which elicits beliefs once from a human expert, a large language model (LLM), or both, and deterministically compiles a structured specification into a reusable belief function ($\pi: \mathcal{X} \to[0,1]$), with a rich representation not restricted to a Gaussian family. Building on $\alpha$-$\pi$BO, HapBO applies this belief function through a bounded positive multiplier on any nonnegative 
    
[^22]: 纵向矩阵数据的经验变分贝叶斯因子分解

    Empirical Variational Bayes Factorization for Longitudinal Matrix Data

    [https://arxiv.org/abs/2611.25065](https://arxiv.org/abs/2611.25065)

    提出了一种经验变分贝叶斯因子分解方法，能够高效处理具有时间相关性的高维纵向矩阵数据，自动确定所有超参数，并有效应对缺失数据和不规则时间点的挑战。

    

    在生物医学研究中，为多个个体随时间收集高维数据正变得越来越普遍，尤其是在疾病进展研究或分子时间进程实验中。通常情况下，数据存在缺失，或不同个体之间的时间点间隔不规则。目前缺乏灵活的方法来探索随时间变化的潜在模式并处理缺失数据。我们提出了一种经验变分贝叶斯（EVB）方法来对高维纵向数据进行因子分解，其中潜在因子具有时间上的相关性。EVB方法计算高效，能够灵活适应随时间变化的协方差结构，并自动确定所有超参数（包括潜在秩，即因子数量）。模拟研究表明，该方法比其他策略更精确地捕捉潜在信号，并且在处理缺失数据方面表现良好。

    arXiv:2611.25065v1 Announce Type: cross  Abstract: Collecting high-dimensional data over time for multiple individuals is increasingly common in biomedical research, particularly in studies of disease progression or molecular time-course experiments. Often data are missing or the timepoints are irregularly spaced across individuals. There is a lack of flexible methodology to explore latent patterns of change over time and account for missing data. We propose an empirical variational Bayes (EVB) approach to factorize high-dimensional longitudinal data, in which the underlying factors have temporal correlation over time. The EVB approach is computationally efficient and accommodates flexible covariance structures over time with automatic determination of all hyper-parameters (including the underlying rank, i.e., number of factors). Simulations demonstrate that the approach captures the underlying signal more precisely than other strategies, and performs well with respect to missing data 
    
[^23]: 所有VC类的近二次样本压缩

    Near-Quadratic Sample Compression for All VC Classes

    [https://arxiv.org/abs/2611.25011](https://arxiv.org/abs/2611.25011)

    本文通过构造大小为 O(d²log²d) 的样本压缩方案，证明了Floyd和Warmuth提出的弱样本压缩猜想，即每个VC维为 d 的二值概念类都存在多项式大小的样本压缩方案。

    

    样本压缩猜想是学习理论中一个长期悬而未决的问题，起源于Littlestone和Warmuth在20世纪80年代的工作。其核心在于简化与泛化之间的联系：如果一个与训练样本一致的预测器仅通过该样本的一小部分来定义，那么它很可能具有泛化能力。样本压缩方案将这一思想形式化，许多自然的学习算法都可以通过这种方式进行分析。Floyd和Warmuth（1995）提出的样本压缩猜想断言，每个VC维为 $d$ 的二值概念类都存在大小为 $O(d)$ 的样本压缩方案；一个较弱的猜想只要求方案大小为 $d$ 的多项式。这两个猜想至今仍未解决：第一个仅依赖于 $d$ 的界由Moran和Yehudayoff（2016）给出，但它是 $d$ 的指数级。我们通过构造一个大小为 $O(d^2 \log^2 d)$ 的样本压缩方案，证明了较弱的猜想。

    arXiv:2611.25011v1 Announce Type: cross  Abstract: The sample compression conjecture is a longstanding open problem in learning theory, originating in the work of Littlestone and Warmuth in the 1980s. At its heart is a connection between simplification and generalization: if a predictor consistent with a training sample is defined using only a small part of that sample, then it is likely to generalize. Sample compression schemes formalize this idea, and many natural learning algorithms can be analyzed this way.   The sample compression conjecture of Floyd and Warmuth (1995) asserts that every binary concept class of VC dimension $d$ admits a sample compression scheme of size $O(d)$; a weaker conjecture asks only for size polynomial in $d$. Both have remained open: the first bound depending only on $d$, due to Moran and Yehudayoff (2016), is exponential in $d$. We prove the weaker conjecture by constructing a sample compression scheme of size $O(d^2 \log^2 d)$.
    
[^24]: 一个语言模型多样性采样模型

    A Model of Diverse Sampling from Language Models

    [https://arxiv.org/abs/2611.25003](https://arxiv.org/abs/2611.25003)

    该论文提出 PoolDPP 方法，将语言模型的多样性采样形式化为行列式点过程（DPP）下的子集采样，通过候选池上的 K-DPP 采样器实现了输出与目标分布间 KL 散度 O(1/N) 的收敛保证，并在实验中扩展了质量-多样性前沿。

    

    语言模型通常能产生高质量的个体样本，但其样本集合往往质量不佳，重复样本聚集在相同的语义模式附近。我们将多样性生成形式化为从行列式点过程（DPP）中采样大小为 $K$ 的完整字符串子集，其中似然编码条目质量，嵌入几何结构编码相似输出之间的排斥作用。由于对所有字符串进行精确推理是不可行的，我们提出 PoolDPP：从一个易于处理的提议分布中抽取有限的候选池，通过重要性加权使候选偏向目标质量，然后在诱导的池核上运行 K-DPP 采样器；贪心 MAP 变体则提供确定性的选择方式。我们证明了一个具有显式常数的有限样本界：采样器输出与目标 $K$-DPP 之间的 KL 散度随池大小 $N$ 以 O(1/N) 的速度衰减。实证结果表明，PoolDPP 在开放式生成任务上扩展了质量-多样性边界，并提高了精确率-召回率……（摘要截断）

    arXiv:2611.25003v1 Announce Type: cross  Abstract: Language models often produce high-quality individual samples but poor sets, with repeated samples clustering around the same semantic modes. We formalize diverse generation as sampling a size-$K$ subset of complete strings from a determinantal point process (DPP), where the likelihood encodes item quality, and the embedding geometry encodes repulsion between similar outputs. Exact inference over all strings is intractable, so we propose PoolDPP: draw a finite candidate pool from a tractable proposal, importance-weight the candidates toward the target quality, and run a $K$-DPP sampler on the induced pool kernel; a greedy MAP variant provides deterministic selection. We prove a finite-sample bound with an explicit constant: the KL divergence between the sampler's output and the target $K$-DPP decays as O(1/N) in the pool size. Empirically, PoolDPP extends the quality--diversity frontier on open-ended generation and the precision--recal
    
[^25]: 基于Kaplan–Meier生存估计的多臂老虎机方差自适应置信上界算法

    Variance-Adaptive Upper Confidence Bounds for Multi-Armed Bandits with Kaplan--Meier Survival Estimation

    [https://arxiv.org/abs/2611.24988](https://arxiv.org/abs/2611.24988)

    该论文提出了KM-UCB，一种基于Greenwood方差公式和经验Bernstein界限的方差自适应多臂老虎机策略，将连续时间生存分析与序贯决策相结合以处理右删失数据下的异方差问题。

    

    我们研究了随机多臂老虎机问题，其中与每个臂相关联的奖励是在预先指定时间范围内的生存概率，而观测到的结果是一个右删失的事件时间变量。在这种设置下，标准老虎机算法面临延迟和不完整的观测。先前针对Kaplan–Meier (KM) 估计器的改进方法依赖于Hoeffding类型的界限，忽略了连续生存过程固有的异方差性。为弥合这一差距，我们提出了KM-UCB，一种方差自适应策略，将连续时间生存分析与离散序贯决策相结合。通过利用鞅理论和连续影响函数，我们推导出了以Greenwood方差公式为参数的固定队列经验Bernstein界限。对于实时分配，我们保留Greenwood指数，并使用来自已完成记录的生存界限来审查所推荐的臂。在追赶式报告机制下……

    arXiv:2611.24988v1 Announce Type: cross  Abstract: We study the stochastic multi-armed bandit problem in which the reward associated with each arm is the survival probability at a pre-specified time horizon, and the observed outcome is a right-censored time-to-event variable. Standard bandit algorithms face delayed and partial observations in this setting. Prior adaptations for the Kaplan--Meier (KM) estimator rely on Hoeffding-type bounds that ignore the inherent heteroscedasticity of continuous survival processes. To bridge this gap, we propose KM-UCB, a variance-adaptive policy that unites continuous-time survival analysis with discrete sequential decision-making. By leveraging martingale theory and continuous influence functions, we derive a fixed-cohort empirical Bernstein bound parameterized by Greenwood's variance formula. For realtime allocation, we retain the Greenwood index and use a survival bound from completed records to review the recommended arm. Under catch-up reporting
    
[^26]: 具有信息性弃权的风险最优主动学习

    Risk-optimal active learning with informative abstention

    [https://arxiv.org/abs/2611.24986](https://arxiv.org/abs/2611.24986)

    该论文提出了一种在预言机可能弃权且响应概率与分类器共享参数情形下的主动学习查询获取准则，通过结合调整冗余参数后的分类器信息与多类零一风险的局部曲率，实现首阶渐近超额风险的最小化。

    

    我们研究了在预言机可能弃权且其响应概率与分类器共享参数的情形下的标签获取问题。此时，一次查询可以通过返回的标签和响应结果两条途径提供信息。在正确设定的参数模型下，我们推导出对响应模型冗余参数进行调整后的分类器信息。将该信息与多类零一风险的局部曲率相结合，得到了一个用于最小化首阶渐近超额风险的查询获取准则。我们刻画了固定查询预算下的最优分配方案，并展示了查询分配如何影响响应模型的可识别性。当响应概率不依赖于分类器参数时，该预言机准则退化为响应折扣式获取。仿真实验考察了由此得到的查询分配，并通过消融实验分离出响应结果项的贡献。此外还基于消费者投诉数据回放进行了实验验证。

    arXiv:2611.24986v1 Announce Type: cross  Abstract: We study label acquisition when an oracle may abstain and its response probability shares parameters with the classifier. A query can then provide information through both a returned label and the response outcome. Under a correctly specified parametric model, we derive the classifier information after adjustment for response-model nuisance parameters. Combining this information with the local curvature of multiclass zero-one risk gives an acquisition criterion for minimizing the leading asymptotic excess risk. We characterize the optimal allocation under a fixed query budget and show how query allocation affects response identifiability. The oracle criterion reduces to response-discounted acquisition when response probability does not depend on the classifier parameters. Simulations examine the resulting query allocations and isolate the contribution of the response-outcome term through an ablation. A Consumer Complaints replay retain
    
[^27]: DL101：神经网络模型的表示能力

    DL101 The Representation Capability of Neural Network Models

    [https://arxiv.org/abs/2611.24980](https://arxiv.org/abs/2611.24980)

    本文系统梳理了通用近似定理这一神经网络表示能力的数学基石，指出非多项式激活函数是实现通用逼近的必要条件，并强调这些定理本身并不提供关于效率、训练收敛与泛化能力的保证。

    

    arXiv:2611.24980v1 公告类型：cross 摘要：通用近似定理（Cybenko、Funahashi、Hornik 与 Leshno）为神经网络为何足够强大、能够建模复杂非线性现象提供了数学依据。浅层网络可以逼近紧集上的任意连续函数。Leshno 定理阐明了非多项式激活函数对于实现通用性是必要的，这解释了为什么 ReLU、逻辑斯蒂（sigmoid）、双曲正切以及其他常见激活函数是有效的（而纯多项式激活则不行）。然而，逼近定理并未提供效率、训练或泛化方面的任何保证。它们既没有说明为达到给定精度需要多少神经元，也没有说明梯度下降能否找到正确的参数，或者在有限数据上训练的网络是否能够良好地泛化。Barron 为特定函数类提供了定量的误差界（即对傅里叶范数有界的函数给出了逼近速率），而 Montúfar 则是第一个证明……（摘要到此截断）

    arXiv:2611.24980v1 Announce Type: cross  Abstract: Universal approximation theorems (Cybenko, Funahashi, Hornik & Leshno) provide the mathematical justification for why neural networks are powerful enough to model complex, nonlinear phenomena. Shallow networks can approximate any continuous function on compact sets. Leshno's theorem clarified that non-polynomial activations are necessary for universality, which explains why ReLU, logistic, hyperbolic tangent, and other common activations work (but pure polynomials do not).   However, approximation theorems provide no efficiency, training, nor generalization guarantees. They do not indicate how many neurons are needed for a given accuracy, whether gradient descent will find the right parameters, or whether networks trained on finite data will generalize well.   Barron provided quantitative error bounds for particular function classes (approximation rates for functions with bounded Fourier norm), whereas Mont\'ufar was the first to show 
    
[^28]: FedE2E：面向端到端自动驾驶的带闭环评估的一次性联邦学习

    FedE2E: One-Shot Federated Learning for End-to-End Autonomous Driving with Closed-Loop Evaluation

    [https://arxiv.org/abs/2611.24254](https://arxiv.org/abs/2611.24254)

    提出首个面向端到端自动驾驶的一次性联邦学习方法FedE2E，通过多客户端校准、锚点特征对齐和词元级共识蒸馏从异构模型中蒸馏知识，在Bench2Drive闭环评估中一次性驾驶分数达55.85，超过先前工作4.07分。

    

    联邦学习能够在不共享数据集的情况下实现协同模型训练。然而，现有的自动驾驶联邦学习方法通常需要多轮通信且缺乏闭环评估，而一次性联邦学习方法则主要专注于感知任务。为了解决这些局限性，我们提出了FedE2E，一种面向端到端自动驾驶的一次性联邦学习方法。该方法包含多客户端校准、基于锚点的特征对齐以及词元级共识蒸馏，以从异构模型中蒸馏知识。在Bench2Drive上的闭环评估证明了FedE2E在一次性和多轮联邦学习中的有效性。在一次性设置下，它取得了55.85的驾驶分数，比先前工作高出4.07分；在多轮设置下，经过五轮通信后达到60.35，是同等通信轮数下所有对比方法中的最佳性能。

    arXiv:2611.24254v1 Announce Type: new  Abstract: Federated learning enables collaborative model training without sharing datasets. However, existing federated learning methods for autonomous driving typically require multi-round communication and lack closed-loop evaluation, while one-shot methods mainly focus on perception tasks. To address these limitations, we propose \method, a one-shot federated learning method for end-to-end autonomous driving. It comprises multi-client calibration, anchor-based feature alignment, and token-wise consensus distillation to distill knowledge from heterogeneous models. Closed-loop evaluation on Bench2Drive demonstrates the effectiveness of \method in both one-shot and multi-round federated learning. In the one-shot setting, it achieves a Driving Score of 55.85, which is 4.07 points higher than prior work. In the multi-round setting, it reaches 60.35 after five communication rounds, the best performance among all compared methods under the same number
    
[^29]: 部署预算约束下基于基础模型的预测

    Prediction with Foundation Models under Deployment Budgets

    [https://arxiv.org/abs/2611.12466](https://arxiv.org/abs/2611.12466)

    该论文提出了在平均查询成本预算约束下提升基础模型预测性能的框架：固定查询模型的方法可达到极小化极大最优率，并证明多个单独欠佳的模型输出组合后可超越最佳单一模型；自适应方法则借助Q学习为不同输入动态选择查询模型并决定停止时机。

    

    基础模型能够为下游任务提供预测，这些预测借鉴了大规模预训练所积累的知识，但不同基础模型在任务特定准确率和查询成本方面存在差异。我们开发了一个框架，在每条输入的平均查询成本预算约束下改进下游任务的预测，利用训练数据来学习应查询哪些模型，以及如何将模型输出与无需查询即可获得的特征相结合。第一种方法对每条输入查询相同的模型；第二种方法则允许不同输入查询不同的模型，并使用Q学习来决定下一个查询哪个模型以及何时停止查询。对于第一种方法，我们证明了多个单独并不准确的输出组合起来可以优于最佳的单个输出。当回归函数仅通过预算内的模型输出而依赖于这些输出时，该方法可以达到极小化极大率；否则，预算约束下的任何方法都可能被迫只能达到更慢的收敛率，而第一种方法恰好能达到该较慢的速率。对于第二种方法……（摘要原文在此处截断）

    arXiv:2611.12466v2 Announce Type: replace-cross  Abstract: Foundation models can supply predictions for a downstream task that draw on knowledge from extensive pretraining, but they differ in task-specific accuracy and query cost. We develop a framework that improves prediction for the task under a budget on the average query cost per input, using training data to learn which models to query and how to combine their outputs with features available without querying. The first method queries the same models for every input; the second lets them vary across inputs and uses Q-learning to decide which model to query next and when to stop. For the first method, we show that several individually inaccurate outputs can jointly outperform the best single one. It attains the minimax rate when the regression function depends on the outputs only through models within the budget. Otherwise, any method within the budget can be forced to a slower rate, which the first method attains. For the second m
    
[^30]: 面向平均奖励 $Q$-学习与 $Q$-规划的混合正则化方法

    Mixing Regularization for Average-Reward $Q$-Learning and $Q$-Planning

    [https://arxiv.org/abs/2610.14988](https://arxiv.org/abs/2610.14988)

    该论文提出通过向一个与奖励无关的 $m$ 步混合策略进行正则化来控制平均奖励 $Q$-学习的收缩与探索性质，并给出有限时间收敛保证，在单步情形下匹配最佳已知速率，在 $m$ 步混合情形下首次推导出新的速率。

    

    强化学习的平均奖励设定具有挑战性，因为其贝尔曼算子仅在底层马尔可夫链具有混合性时才在跨度上收缩。现有工作通常假设存在收缩性质（例如通过环境混合性），而我们考虑这样的设定：给定一个与奖励无关的 $m$ 步混合策略，并通过向该混合策略正则化来控制所学策略的收缩性质与探索性质。我们对 $Q$-学习算法提供了有限时间分析，这些算法可以用多种正则化选择实例化，其保证相对于正则化后的不动点来度量。我们的框架揭示了新的、依赖于正则化子的对象，这些对象控制收缩、探索和算子偏差。在单步设定中，我们的速率匹配了 $Q$-迭代和同步 $Q$-学习的最佳已知速率；对于 $m$ 步混合情形，我们推导出了新的速率。

    arXiv:2610.14988v1 Announce Type: cross  Abstract: The average-reward setting for reinforcement learning is challenging because its Bellman operator is known to contract in span only when the underlying Markov chain is mixing. Whereas existing work typically assumes a contraction property, e.g., through environment mixing, we consider the setting where we are given a reward-agnostic $m$-step mixing policy, and we control the contraction and exploration properties of the learned policy by regularizing toward the mixing policy. We provide finite-time analysis of $Q$-learning algorithms that can be instantiated with multiple choices of regularization, with guarantees measured with respect to the regularized fixed point. Our framework exposes novel regularizer-dependent objects that control contraction, exploration, and operator bias. In the one-step setting, our rates match the best known rates for $Q$-iteration and synchronous $Q$-learning, and for $m$-step mixing we derive new rates. We
    
[^31]: 实用的双机器学习方法与AI学习表示

    Pragmatic DML with AI-Learned Representations

    [https://arxiv.org/abs/2610.01935](https://arxiv.org/abs/2610.01935)

    该论文揭示了AI学习表示的误差会以结果回归误差与平衡权重误差的乘积形式影响因果参数估计，并证明了交叉拟合DML可为依赖表示的目标提供有效推断，且与按折表示学习/微调兼容，为基于AI表示的因果推断提供了实用框架。

    

    文本、图像及其他丰富的协变量正日益被压缩为AI学习得到的表示，并被用作因果分析中的控制变量。我们研究了这一方法在何种情况下是有效的，并针对基于学习表示的因果推断开发了一个实用框架。对于广泛一类估计对象，不完美的表示会通过两个表示误差的乘积来扭曲目标因果参数：一个来自结果回归，另一个来自平衡权重（或Riesz表示元）。这带来了三项建设性成果。第一，交叉拟合的双机器学习（DML）为依赖表示的目标提供了有效的Wald推断。当表示误差较小时，同一置信区间能够覆盖因果参数，甚至可以达到半参数有效界。第二，按折进行的表示学习（或微调）与针对因果参数的DML推断是兼容的。为此，我们开发了凸-（原文在此处截断）

    arXiv:2610.01935v1 Announce Type: new  Abstract: Text, images, and other rich covariates are increasingly compressed into AI-learned representations and then used as controls in causal analysis. We study when this approach is valid and develop a practical framework for causal inference with learned representations. For a broad class of estimands, an imperfect representation distorts the target causal parameter by the product of two representation errors: one in the outcome regression and one in the balancing weight (or Riesz representer). This yields three constructive results. First, cross-fitted double machine learning (DML) provides valid Wald inference for the representation-dependent target. When representation errors are small, the same interval covers the causal parameter, and it can even attain the semiparametric efficiency bound. Second, fold-wise representation learning (or fine-tuning) is compatible with DML inference for the causal parameter. To this end, we develop convex-
    
[^32]: 具有最优神经元数量与最优比特复杂度的任意精度神经逼近

    Arbitrary-Accuracy Neural Approximation with Optimal Neuron Count and Optimal Bit Complexity

    [https://arxiv.org/abs/2609.35628](https://arxiv.org/abs/2609.35628)

    本文通过显式构造激活函数与闭式网络，证明了在 $[0,1]^d$ 上对连续函数进行任意精度逼近时 $d+1$ 个隐神经元是精确最优的，并给出了结合取整与指数函数的更简单构造及最优编码复杂度。

    

    我们研究了在 $[0,1]^d$ 上对连续函数进行任意精度逼近所需的最少隐神经元数量以及相关的编码复杂度。对于 $d\geq 2$，我们构造了两个固定的、显式定义的激活函数，并为每个激活函数给出了一个具有宽度为 $d$ 和 $1$ 的两个隐藏层的闭式网络，该网络在一致范数下可实现任意精度逼近。我们证明了 $d+1$ 是在具有局部可积激活函数和仿射输出层的标准前馈网络中，隐神经元总数的精确最小值。我们进一步给出了一种更简单的构造，仅使用一个结合了取整函数与指数函数的初等激活函数。该构造需要宽度为 $d$、$1$ 和 $2$ 的三个隐藏层；如果允许跳跃连接，则宽度为 $d$、$1$ 和 $1$ 即可满足要求。对于一个有界的 $\alpha$-Hölder 类（$\alpha\in(0,1]$），采用数字提取激活函数与初等激活函数的网络……

    arXiv:2609.35628v2 Announce Type: replace  Abstract: We study the minimum number of hidden neurons required for arbitrary-accuracy approximation of continuous functions on $[0,1]^d$ and the associated encoding complexity. For $d\geq 2$, we construct two fixed, explicitly defined activation functions and, for each, a closed-form network with two hidden layers of widths $d$ and $1$ that achieves arbitrary accuracy in the uniform norm. We prove that $d+1$ is the exact minimum total number of hidden neurons among standard feedforward networks with locally integrable activations and affine outputs. We further give a simpler construction using a single elementary activation that combines the floor and exponential functions. This construction requires three hidden layers of widths $d$, $1$, and $2$. If a skip connection is allowed, widths $d$, $1$, and $1$ suffice. For a bounded $\alpha$-H\"older class with $\alpha\in(0,1]$, the networks with the digit-extraction activation and with the eleme
    
[^33]: 具有对抗性响应的不可知平滑在线回归

    Agnostic Smoothed Online Regression with Adversarial Responses

    [https://arxiv.org/abs/2609.32478](https://arxiv.org/abs/2609.32478)

    本文提出 Hedge-Cover 算法，利用协变量平滑性界定专家先验中的分歧次数，为具有有界对抗响应的不可知平滑在线回归实现了 $\widetilde{O}(\sqrt{\text{Pdim}(\mathcal{F}) \textsf{C}_{\textsf{cov}} T})$ 的次线性遗憾，从而解决了极小极大最优自适应遗憾这一开放问题。

    

    我们研究具有有界对抗响应的平滑在线预测问题。这一被广泛研究的框架通过平滑参数 $\textsf{C}_{\textsf{cov}}$ 在独立同分布采样与对抗性协变量选择之间架起桥梁，该参数以固定但未知的基准测度为参照，约束条件协变量密度。我们提出了 Hedge-Cover，这是一种信息论算法，对于伪维度有界的函数类，能够实现次线性遗憾 $\widetilde{O}(\sqrt{\text{Pdim}(\mathcal{F}) \textsf{C}_{\textsf{cov}} T})$。该算法使用 Hedge 聚合一个精心构建的专家族，其先验将遗憾与一致性选择器与目标函数之间的分歧次数联系起来。我们通过利用协变量平滑性来界定这一分歧次数。这回答了先前文献中提出的关于平滑在线回归问题极小极大最优自适应遗憾的开放问题。

    arXiv:2609.32478v2 Announce Type: replace-cross  Abstract: We study smoothed online prediction with bounded adversarial responses. This widely studied framework bridges i.i.d. sampling and adversarial covariate selection through a smoothness parameter $\textsf{C}_{\textsf{cov}}$, which bounds conditional covariate densities relative to a fixed, unknown base measure. We propose Hedge-Cover, an information-theoretic algorithm that achieves sublinear regret $\widetilde{O}(\sqrt{\text{Pdim}(\mathcal{F}) \textsf{C}_{\textsf{cov}} T})$ for function classes with bounded pseudo-dimension. The algorithm aggregates a carefully constructed family of experts using Hedge, with a prior that links regret to the number of disagreements between a consistent selector and a target function. We bound this number by exploiting covariate smoothness. This answers an open problem posed in \cite{blanchard2025agnostic} on the minimax optimal adaptive regret of the smoothed online regression problem. We establis
    
[^34]: 面向广义线性混合效应模型的先验摊销上下文贝叶斯推断

    Prior-Amortized In-Context Bayesian Inference for Generalized Linear Mixed-Effects Models

    [https://arxiv.org/abs/2609.24422](https://arxiv.org/abs/2609.24422)

    提出metabeta——一个预训练神经网络，在测试时接受任意先验族和超参数输入，利用集合变换器与条件归一化流对GLMM进行先验摊销的上下文贝叶斯推断，无需MCMC即可零样本泛化并获得校准的后验估计。

    

    分层数据在实证科学中无处不在，通常使用广义线性混合效应模型（GLMM）进行分析。对GLMM进行贝叶斯推断能够提供校准的不确定性量化，但需要MCMC采样；No-U-Turn采样器（NUTS）是金标准，但速度缓慢，且每遇到新的数据集、模型和先验都必须从头开始。我们提出了metabeta，一个用于GLMM先验摊销上下文贝叶斯推断的预训练神经网络。与以往在训练时固定先验的神经后验估计器不同，metabeta在测试时接受先验族和超参数作为输入，从而实现零样本泛化。两个集合变换器与条件归一化流镜像了后验的两级结构（跨组共享的全局参数与每组的局部参数）。该模型在数百万个涵盖连续、二元和计数结果的真实感模拟数据集上进行训练。

    arXiv:2609.24422v1 Announce Type: new  Abstract: Hierarchical data is ubiquitous in the empirical sciences and is most commonly analyzed with generalized linear mixed-effects models (GLMMs). Bayesian inference for GLMMs yields calibrated uncertainty but requires MCMC; the No-U-Turn Sampler (NUTS) is the gold standard but is slow and must restart from scratch for every new dataset, model and prior. We introduce metabeta, a pretrained neural network for prior-amortized in-context Bayesian inference over GLMMs. Unlike previous neural posterior estimators that fix the prior at training time, metabeta accepts prior families and hyperparameters as inputs at test time, enabling zero-shot generalization. Two set transformers and conditional normalizing flows mirror the posterior's two-level structure (global parameters shared across groups, local parameters per group). The model is trained on millions of realistic simulated datasets spanning continuous, binary, and count outcomes. By default, 
    
[^35]: Swiss-Knife：解码时可重构的外部化多目标对齐框架

    Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time

    [https://arxiv.org/abs/2609.22226](https://arxiv.org/abs/2609.22226)

    该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。

    

    解码时对齐方法通过使用外部奖励对候选续写进行评分并选择得分最大者，来引导一个冻结的语言模型。我们认为这种共享设计只是更大空间中的一个退化点。我们提出Swiss-Knife，一个外部化多目标对齐框架，其中对齐规范是一等运行时对象：可热插拔的评分刀片、批归一化器、成对聚合算子和选择规则。六个公理刻画了可容许的聚合算子，并且我们证明了一个表示定理：每个满足这些公理的算子都具有形式 $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$，这是一个双参数族，其中包含probit和logistic比较规则以及逐点argmax作为命名的坐标。在该族中，成对聚合在对抗性奖励污染下是Lipschitz稳定的，而argmax则不是，且候选批归一化……（原文摘要在此处截断）

    arXiv:2609.22226v1 Announce Type: new  Abstract: Decode-time alignment methods steer a frozen language model by scoring candidate continuations with an external reward and selecting the maximiser. We argue that this shared design is a single degenerate point in a much larger space. We introduce Swiss-Knife, a framework for externalised multi-objective alignment in which the alignment specification is a first-class runtime object: hot-swappable scoring blades, a batch normaliser, a pairwise aggregation operator, and a selection rule. Six axioms characterise the admissible aggregation operators, and we prove a representation theorem: every operator satisfying them has the form $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$, a two-parameter family containing probit and logistic comparison rules and pointwise argmax as named coordinates. Within it, pairwise aggregation is Lipschitz-stable under adversarial reward contamination while argmax is not, and Candidate-Batch Norma
    
[^36]: 全模态分解自编码器学习全栈式可穿戴解耦表征

    Omni-modal decomposition autoencoders learn full-stack wearable disentangled representations

    [https://arxiv.org/abs/2608.07385](https://arxiv.org/abs/2608.07385)

    提出全模态变分分解自编码器（OmniDecVAEs），利用单一共享编码器-解码器和多视角自监督分解损失，从任意多模态可穿戴时间序列中学习解耦表征，在人体活动识别数据集上实现最高 3.0 个百分点的 DCI 解耦度提升。

    

    学习解耦表征是通用多模态可穿戴模型的关键，然而现有方法很少能在单一模型中同时实现异构多模态时间序列的分类、解耦、融合与生成。我们提出了全模态变分分解自编码器（OmniDecVAEs），这是一个通过单一共享编码器-解码器从任意数量模态中学习多用途表征的框架。OmniDecVAEs 在 DecVAEs 的基础上进行扩展，通过多视角自监督分解损失和逐分量条件解码器，为每个模态学习时频潜在子空间。在全模态人体活动识别（HAR）HARWE 数据集（最多三十个通道）以及公开的 UCI HAR 和 WESAD 数据集上，OmniDecVAEs 为未见过的受试者学习到了最解耦的表征（DCI 解耦指标最高提升 3.0 个百分点）。在 HARWE 数据集上，它们改善了受试者相关的（摘要在此处截断）

    arXiv:2608.07385v2 Announce Type: replace-cross  Abstract: Learning disentangled representations is key for general-purpose multi-modal wearable models, yet existing approaches rarely combine classification, disentanglement, fusion, and generation of heterogeneous multi-modal time series in a single model. We introduce Omni-modal Variational Decomposition Autoencoders (OmniDecVAEs), a framework that learns multi-purpose representations from arbitrarily many modalities with a single shared encoder-decoder. OmniDecVAEs extend DecVAEs by learning time-frequency latent subspaces per modality through a multi-view self-supervised decomposition loss and a component-wise conditional decoder. On the omni-modal human activity recognition (HAR) HARWE dataset (up to thirty channels) and the public UCI HAR and WESAD datasets, OmniDecVAEs learn the most disentangled representations for unseen subjects (DCI disentanglement up to 3.0 percentage points (pp) higher). On HARWE, they improve subject-depen
    
[^37]: 一种面向约束感知生物过程开发的人机协同贝叶斯优化框架

    A Human-in-the-Loop Bayesian Optimization Framework for Constraint-Aware Bioprocess Development

    [https://arxiv.org/abs/2606.19230](https://arxiv.org/abs/2606.19230)

    本文提出了一种扩展的人机协同贝叶斯优化框架，通过将约束满足概率和鲁棒性能作为帕累托目标，使专家能交互式选择最优候选，从而在生物过程开发中兼顾约束与不确定性。

    

    本文介绍了对帕累托前沿引导采样（PFGS）的扩展，这是一种人机协同（HitL）贝叶斯优化（BO）框架，其中高斯过程（GP）代理导出的量被重新表述为多目标优化问题的目标，所得帕累托前沿暴露给领域专家进行交互式候选选择，而非返回单一自动推荐。该框架在两个方向上进行了扩展：约束优化通过将满足输出规格限制的后验概率作为显式帕累托目标来处理，该概率从GP后验分布解析计算；鲁棒优化通过蒙特卡洛采样策略来处理，该策略估计在用户定义的输入扰动变异性下的期望下置信性能，捕捉在可能实现偏差下的性能退化。

    arXiv:2606.19230v2 Announce Type: replace-cross  Abstract: This work presents an extension to Pareto Front Guided Sampling (PFGS), a Human-in-the-Loop (HitL) Bayesian Optimization (BO) framework in which Gaussian process (GP) surrogate-derived quantities are reformulated as objectives of a multi-objective optimization problem, and the resulting Pareto front is exposed to a domain expert for interactive candidate selection rather than returning a single automated recommendation. The framework is extended in two directions: constrained optimization is addressed by incorporating the posterior probability of satisfying output specification limits as an explicit Pareto objective, computed analytically from the GP posterior distribution; robust optimization is addressed by a Monte Carlo sampling strategy that estimates expected lower-confidence performance over a user-defined variability of input perturbations, capturing performance degradation under likely implementation deviations. The res
    
[^38]: 资源受限动态定价的自适应推断

    Adaptive Inference for Resource-Constrained Dynamic Pricing

    [https://arxiv.org/abs/2606.03736](https://arxiv.org/abs/2606.03736)

    本文提出一种推断感知的重新求解策略，在资源受限动态定价中平衡收益与需求推断，实现接近最优的遗憾和信息效率。

    

    arXiv:2606.03736v2 公告类型：替换-交叉 摘要：我们研究了当卖方在销售季前设定一个固定价格并寻求收益及对该价格需求的有效推断时的资源受限动态定价问题。库存耗尽可能移除目标价格附近的每个可行价格，因此对剩余价格进行随机化不一定能保持识别性。我们提出了一种推断感知的重新求解策略，该策略在观察当前协变量之前检查目标支持，并使用记录的价格混合实现流体目标负荷。在仿射约束容量族中，目标质量$t^{-\gamma}$产生信息量$T^{1-\gamma}$，区间半径$T^{-(1-\gamma)/2}$，以及对初始流体最优值的遗憾$O(\log T+T^{1-\gamma})$。在同一仿射族中，学习到的重心重新求解保留了一个恒定质量的局部目标组件，并且通过指数大于1的多项式误差支出，实现了线性信息时钟和$O(\log T)$遗憾；松弛容量局部定价

    arXiv:2606.03736v2 Announce Type: replace-cross  Abstract: We study resource-constrained dynamic pricing when the seller seeks revenue and valid inference about demand at a price fixed before the selling season. Depletion can remove every feasible price near the target, so randomization over the remaining prices need not preserve identification. We propose an inference-aware re-solving policy that checks target support before observing the current covariates and implements the fluid target load with a logged pricing mixture. In an affine binding-capacity family, target mass $t^{-\gamma}$ yields information $T^{1-\gamma}$, interval radius $T^{-(1-\gamma)/2}$, and regret $O(\log T+T^{1-\gamma})$ against the initial fluid optimum. In the same affine family, learned barycentric re-solving retains a target-local component of constant mass and, with polynomial error spending of exponent greater than one, achieves a linear information clock and $O(\log T)$ regret; slack-capacity local pricing
    
[^39]: 一种用于多发性骨髓瘤高维循环表观遗传特征的可解释半参数贝叶斯机器学习模型

    An interpretable semiparametric Bayesian machine learning model for high-dimensional circulating epigenetic signatures of multiple myeloma

    [https://arxiv.org/abs/2605.20143](https://arxiv.org/abs/2605.20143)

    该论文提出了一种可解释的半参数贝叶斯机器学习模型spBART，将线性协变量成分与稀疏树集成相结合，从多发性骨髓瘤患者的循环游离DNA 5hmC数据中筛选出八个关键基因位点（包括与IL-1/IL-6信号通路相关的IL1RAP），模型AUC达到0.96。

    

    循环游离DNA（cfDNA）的全基因组5-羟甲基胞嘧啶（5hmC）谱可以揭示与多发性骨髓瘤（MM）相关的表观遗传变化，但其分析需要在调整已知风险因素的同时，从数千个基因中筛选出少量具有信息量的基因。我们提出了半参数probit贝叶斯加性回归树（spBART）模型，其中协变量进入具有可解释系数的线性成分，而基因则进入一个稀疏的树集成，通过其后验包含概率在协变量条件下进行基因选择。该分析筛选出八个候选基因位点：IL1RAP、ST5、HERC6、KL、MYO1E、ELK3、CAPN2和UBR4。其中IL1RAP的入选将循环表观遗传特征与多发性骨髓瘤相关的IL-1/IL-6信号通路联系起来。最终模型的受试者工作特征曲线下面积（AUC）达到0.96。

    arXiv:2605.20143v3 Announce Type: replace-cross  Abstract: Genome-wide 5-hydroxymethylcytosine (5hmC) profiles of circulating cell-free DNA (cfDNA) can reveal epigenetic changes associated with multiple myeloma (MM), but their analysis requires adjusting for known risk factors while selecting a few informative genes from thousands. We propose semiparametric probit Bayesian Additive Regression Tree (spBART), in which the covariates enter a linear component with interpretable coefficients and the genes enter a sparse tree ensemble whose posterior inclusion probabilities select genes conditional on the covariates. The analysis selects eight candidate loci, \textit{IL1RAP}, \textit{ST5}, \textit{HERC6}, \textit{KL}, \textit{MYO1E}, \textit{ELK3}, \textit{CAPN2} and \textit{UBR4}. Selection of \textit{IL1RAP} connects the circulating signature to the \textit{IL-1/IL-6} signaling axis implicated in MM. The final model achieves an area under the receiver operating characteristic curve of 0.96
    
[^40]: 感知调查设计的机器学习（SaML）：基于范围综述的人群健康有效推断指南

    Survey-aware Machine Learning: A Guideline for Valid Population Health Inference based on Scoping Review

    [https://arxiv.org/abs/2605.08963](https://arxiv.org/abs/2605.08963)

    本文提出了一个九步骤的感知调查设计机器学习（SaML）指南，通过将调查设计元数据（如抽样权重、分层变量等）纳入机器学习全生命周期，解决基于复杂健康调查数据训练模型时估计偏差、不确定性低估和公平性评估失真的问题。

    

    基于复杂健康调查（如国家健康与营养调查NHANES）训练的机器学习（ML）模型往往忽略初级抽样单元、分层变量和抽样权重。这种做法违反了标准评估方法的独立性假设，导致估计量产生偏差、不确定性被低估，且公平性评估无法反映人群层面的差异。我们提出了感知调查设计的机器学习，这是一个九步骤指南，将调查设计元数据纳入机器学习的整个生命周期。通过对16篇方法学论文的范围综述，我们总结了加权模型训练、基于设计的交叉验证以及调查调整性能评估方面的现有研究工作，并识别了超参数调优和模型部署方面的研究空白。我们还提供了针对特定任务的指导，阐明了不同分析目标所需的步骤。

    arXiv:2605.08963v2 Announce Type: replace-cross  Abstract: Machine Learning (ML) models trained on complex health surveys such as the National Health and Nutrition Examination Survey (NHANES) often ignore primary sampling units, stratification variables, and sampling weights. This practice violates the independence assumptions of standard evaluation methods. As a result, estimates become biased, uncertainty is underestimated, and fairness assessments fail to reflect population-level disparities. We propose Survey-aware Machine Learning (SaML), a nine-step guideline that incorporates survey design metadata across the ML lifecycle. Through a scoping review of 16 methodological papers, we summarize existing work on weighted model training, design-based cross-validation, and survey-adjusted performance evaluation. We also identify gaps in hyperparameter tuning and deployment. We provide task-specific guidance that clarifies which steps are required for different analytical objectives. SaML
    
[^41]: 信息论与统计学习

    Information Theory and Statistical Learning

    [https://arxiv.org/abs/2605.02989](https://arxiv.org/abs/2605.02989)

    该论文章节简明阐述了散度度量在统计学习模型训练中的核心作用，示例涵盖从线性回归、变分自编码器到扩散模型等多种模型。

    

    本手稿是即将出版的Cover和Thomas所著《信息论基础》第三版中一章的预印本，经Wiley出版社许可发布。如需反馈，请联系abbas@ee.stanford.edu。学习与信息论在模型训练和基本性能极限的刻画这两个方面均存在交集。本手稿对第一个交集进行了简明且易于理解的阐述，仅要求读者具备高年级本科生/一年级研究生水平的信息论和统计学基础。章末习题使得这些内容非常适合课堂教学和自学使用。本章重点讨论了散度度量在模型训练中的作用，示例涵盖从线性和逻辑回归到自回归模型、变分自编码器、扩散模型以及基于分数的模型。特别是，对生成扩散（模型的处理……）

    arXiv:2605.02989v3 Announce Type: replace-cross  Abstract: This manuscript contains preprint of a chapter in the forthcoming third edition of Cover and Thomas's Elements of Information Theory, posted with permission from Wiley. For feedback, please contact abbas@ee.stanford.edu.   Learning and information theory intersect in both model training and the characterization of fundamental performance limits. This manuscript provides a concise and accessible treatment of the first intersection, requiring only basic background in information theory and statistics at the senior undergraduate / first-year graduate level. End-of-chapter exercises make the material well suited for classroom use as well as self-study.   The chapter focuses on the role of divergence measures in model training, with examples ranging from linear and logistic regression to autoregressive models, variational autoencoders, diffusion models, and score-based models. In particular, the treatment of the generative diffusion
    
[^42]: 具有全局收敛性的梯度正则化牛顿提升树

    Gradient Regularized Newton Boosting Trees with Global Convergence

    [https://arxiv.org/abs/2605.00581](https://arxiv.org/abs/2605.00581)

    本文提出受限牛顿下降理论框架，证明了牛顿提升树在光滑强凸损失下的线性收敛速率，并通过梯度正则化扩展实现了对一般凸损失函数的全局收敛保证。

    

    梯度提升决策树（GBDT）在表格数据机器学习中占据主导地位，XGBoost、LightGBM 和 CatBoost 等现代实现均基于牛顿提升，即在决策树空间中的二阶下降步骤。尽管牛顿提升在经验上取得了成功，但与一阶提升相比，其全局收敛性却鲜有理论理解。在本文中，我们提出了受限牛顿下降，这是一个在希尔伯特空间上使用不精确迭代进行牛顿方法凸优化的框架，其建立在余弦角和弱梯度边的概念之上。在该框架内，基于 GBDT 的牛顿提升和经典有限维理论均可作为特例被涵盖。我们首先证明了原始牛顿提升对于满足海森矩阵占优条件的光滑强凸损失函数能够达到线性收敛速率。为了处理具有 Lipschitz 海森矩阵的一般凸损失函数，我们扩展了近期基于梯度的……

    arXiv:2605.00581v2 Announce Type: replace-cross  Abstract: Gradient Boosting Decision Trees (GBDTs) dominate tabular machine learning, with modern implementations like XGBoost, LightGBM, and CatBoost being based on Newton boosting: a second-order descent step in the space of decision trees. Despite its empirical success, the global convergence of Newton boosting is poorly understood compared to first-order boosting. In this paper, we introduce Restricted Newton Descent, a framework for convex optimization with Newton's method on Hilbert spaces with inexact iterates, based on the concepts of cosine angle and weak gradient edge. Within this framework, we recover Newton boosting with GBDTs and classical finite-dimensional theory as special cases. We first prove that vanilla Newton boosting achieves a linear rate of convergence for smooth, strongly convex losses that satisfy a Hessian-dominance condition. To handle general convex losses with Lipschitz Hessians, we extend a recent gradient 
    
[^43]: 大维度下谱算法的学习曲线与欠正则化时的预测持续性

    Learning Curves and Prediction Persistence to Under-Regularization of Spectral Algorithms in Large Dimensions

    [https://arxiv.org/abs/2604.23212](https://arxiv.org/abs/2604.23212)

    该文在大维度体制（$n\asymp d^\gamma$）下，以常数因子精度刻画了谱算法随正则化从最优尺度减弱直至插值时条件超额风险的阶数，揭示了风险何时保持极小极大阶、何时出现多项式量级更大的欠正则化风险区间。

    

    我们研究当样本量 $n$ 与维度 $d$ 满足 $n\asymp d^\gamma$（$\gamma>0$ 为固定常数）时，谱算法在正则化程度减弱过程中预测风险的变化规律。对于球面上的内积核，在光滑度 $s\geq 0$ 的源条件与信号非退化条件下，我们在每个固定的可容许尺度 $\lambda\asymp d^{-u}$（$u>0$）以及插值（interpolation）情形下，以常数因子精度刻画了条件超额风险。所得的风险阶数描述了源光滑度、滤波器资格以及谱块之间转换的影响。这些结果识别了在正则化从最优尺度减弱至插值的过程中，条件超额风险何时仍保持极小极大风险阶数，以及何时在插值之前的某个区间会出现按多项式量级增大的风险。我们还使用确定性正则化方法，将相同谱滤波器下的核风险与总体序列风险进行了比较……

    arXiv:2604.23212v2 Announce Type: replace-cross  Abstract: We study how prediction risk changes as regularization is reduced for spectral algorithms when the sample size $n$ and dimension $d$ satisfy $n\asymp d^\gamma$ for fixed $\gamma>0$. For inner-product kernels on the sphere, under a source condition with smoothness $s\geq0$ and a signal nondegeneracy condition, we characterize the conditional excess risk up to constant factors at each fixed admissible scale $\lambda\asymp d^{-u}$, $u>0$, and at interpolation. The resulting risk orders describe the effects of source smoothness, filter qualification, and transitions between spectral blocks. They identify when the conditional excess risk retains the minimax risk order as regularization is weakened from an optimal scale to interpolation, and when an interval before interpolation exhibits a polynomially larger risk. We also compare the kernel risk with a population sequence risk for the same spectral filter, using a deterministic regu
    
[^44]: 资源受限定价中信息的价值

    The Value of Information in Resource-Constrained Pricing

    [https://arxiv.org/abs/2603.24974](https://arxiv.org/abs/2603.24974)

    本文研究了资源受限动态定价中的信息价值，在完全信息、无信息和知情定价三种信息环境下分别设计了算法：完全信息下仅求解一次流体问题的残差修正策略达到 O(log T) 后悔值，无信息下确定性探索加确定性等价策略达到 O(√T) 后悔值，知情定价下反馈控制器在斜率矩阵对称时无需估计斜率即可学习最优实施价格。

    

    在资源约束下的动态定价中，需要在探索未知需求模式与利用已知信息之间进行权衡，以实现收益最大化。我们针对该问题在不同的信息环境下开发了相应的算法。产品共享易逝性资源，企业发布价格并观察实际到达的需求请求，这些请求可能超过剩余容量。在完全信息环境下，一种仅求解一次流体问题的残差修正策略，在有界价格和任意秩资源矩阵的条件下可达到 O(log T) 的后悔值。在无信息环境下，先进行确定性探索再采用确定性等价策略，当极限流体解非退化时可达到 O(√T) 的后悔值。在知情定价环境下，给定初始的价格-需求估计，若斜率矩阵是对称的，反馈控制器无需估计斜率矩阵即可学习实施价格……（原文摘要此处截断）

    arXiv:2603.24974v3 Announce Type: replace-cross  Abstract: Dynamic pricing with resource constraints requires balancing exploration of unknown demand patterns against exploitation of known information to maximize revenue. We develop algorithms for this problem across different information regimes. Products share perishable resources, and the firm posts prices and observes the realized requests, which may exceed the remaining capacity. In the full-information setting, a residual-correction policy that solves the fluid problem once attains $O(\log T)$ regret with bounded prices and arbitrary-rank resource matrices. In the no-information setting, deterministic exploration followed by a certainty-equivalent policy attains $O(\sqrt T)$ regret when the limiting fluid solution is nondegenerate. In the informed-price setting, given an initial price-demand estimate, a feedback controller learns the implementing prices without estimating the slope matrix if the slope matrix is symmetric, a set c
    
[^45]: 概率分布的连续时间学习：1型糖尿病幼儿数字试验中的案例研究

    Continuous-time learning of probability distributions: a case study in a digital trial of young children with type 1 diabetes

    [https://arxiv.org/abs/2603.24427](https://arxiv.org/abs/2603.24427)

    提出一个基于高斯混合模型与神经常微分方程的连续时间概率分布学习框架，用于刻画1型糖尿病儿童连续血糖监测数据中血糖分布随时间的动态演变。

    

    理解生物标志物分布如何随时间演变是数字健康和慢性病监测中的核心挑战。例如在糖尿病中，血糖测量值分布的变化可以揭示传统汇总指标所忽略的疾病进展与治疗反应模式。受一项为期13周、比较闭环胰岛素输注系统 t:slim X2 与标准疗法在1型糖尿病儿童中疗效的临床试验启发，我们提出了一个概率框架，利用每五分钟采集一次的连续血糖监测（CGM）数据来建模时间索引分布的连续时间演变。我们将血糖分布表示为高斯混合模型，其时变的混合权重由神经常微分方程（neural ODE）控制，并基于最大均值差异（MMD）的分布匹配准则来估计模型参数。由此得到的框架具有可解释性和计算高效性。

    arXiv:2603.24427v2 Announce Type: replace-cross  Abstract: Understanding how biomarker distributions evolve over time is a central challenge in digital health and chronic disease monitoring. For example, in diabetes, changes in the distribution of glucose measurements can reveal patterns of disease progression and treatment response that conventional summary measures miss. Motivated by a 13-week clinical trial comparing the closed-loop insulin delivery system t:slim X2 with standard therapy in children with type 1 diabetes, we propose a probabilistic framework to model the continuous-time evolution of time-indexed distributions using continuous glucose monitoring (CGM) data collected every five minutes. We represent the glucose distribution as a Gaussian mixture, with time-varying mixture weights governed by a neural ODE. We estimate the model parameters using a distribution-matching criterion based on the maximum mean discrepancy. The resulting framework is interpretable, computationa
    
[^46]: 关于具有 Q\* 近似与部分覆盖的离线强化学习的复杂性研究

    On the Complexity of Offline Reinforcement Learning with Q*-Approximation and Partial Coverage

    [https://arxiv.org/abs/2602.12107](https://arxiv.org/abs/2602.12107)

    该论文通过信息论下界证明仅凭 Q\* 可实现性与贝尔曼完整性不足以实现部分覆盖下样本高效的离线强化学习，并提出一个通用的决策估计框架，将离线强化学习的复杂度分解为决策复杂度与价值估计误差两部分。

    

    我们研究了在 Q\* 近似与部分覆盖条件下的离线强化学习，这一设定激发了许多实用算法（如保守 Q 学习，CQL；Kumar 等，2020），但其受到的理论关注十分有限。我们的工作受到以下开放问题的启发：“Q\* 可实现性与贝尔曼完整性是否足以支撑部分覆盖下样本高效的离线强化学习？”我们通过信息论下界对该问题给出了否定回答。为了识别能够使部分覆盖下样本高效离线强化学习成为可能的额外结构，我们受在线强化学习中无模型决策估计系数（DEC）（Foster 等，2023b；Liu 等，2025b）的启发，引入了一个通用的决策估计框架。该框架将离线强化学习的复杂度分解为决策复杂度和价值估计误差，从而可以对这两个子问题进行模块化研究。我们的结果不仅统一了已有的……

    arXiv:2602.12107v3 Announce Type: replace-cross  Abstract: We study offline reinforcement learning under $Q^\star$-approximation and partial coverage, a setting that motivates practical algorithms such as Conservative $Q$-Learning (CQL; Kumar et al., 2020) but has received limited theoretical attention. Our work is inspired by the following open question: "Are $Q^\star$-realizability and Bellman completeness sufficient for sample-efficient offline RL under partial coverage?"   We answer in the negative via an information-theoretic lower bound. To identify additional structure that enables sample-efficient offline RL under partial coverage, we introduce a general decision-estimation framework, inspired by model-free decision-estimation coefficients (DEC) for online RL (Foster et al., 2023b; Liu et al., 2025b). Our framework decomposes offline RL complexity into decision complexity and value estimation error. This allows modular study of both sub-problems. Our result not only unifies exi
    
[^47]: 楔形采样：具有近线性样本复杂度的高效张量补全

    Wedge Sampling: Efficient Tensor Completion with Nearly-Linear Sample Complexity

    [https://arxiv.org/abs/2602.05869](https://arxiv.org/abs/2602.05869)

    本文提出“楔形采样”这一非自适应采样方案，通过对张量展开同列元素进行成对采样以提供谱初始化所需的行相关性，从而在固定秩条件下以近线性样本复杂度实现低秩张量的多项式时间恢复。

    

    我们提出了楔形采样（wedge sampling），这是一种非自适应采样方案，用于从张量元素的结构化观测中恢复阶数为 k、边长为 n 的低秩张量。该方案在张量展开矩阵的同一列中对元素进行成对采样，从而在相关的二部采样图中形成长度为二的路径（即楔形）。在均匀采样产生楔形数量不足的情形下，成对元素的乘积为谱初始化提供了所需的行相关性。在秩、不相干性、张量阶数和条件数均为固定的情况下，楔形采样能够以关于 n 近似线性的样本复杂度实现多项式时间的恢复。该谱初始化器可与现有的精细化程序结合，仅需额外使用 Õ(n) 个均匀采样的元素。谱精细化过程可实现到任意固定相对 Frobenius 误差的弱恢复。对于具有不相干典型多分解（CP）因子的对称三阶张量，……

    arXiv:2602.05869v4 Announce Type: replace-cross  Abstract: We introduce wedge sampling, a non-adaptive sampling scheme for recovering an order-$k$ low-rank tensor of side length $n$ from structured observations of its entries. The scheme samples pairs of entries in the same column of a tensor unfolding, forming length-two paths (wedges) in the associated bipartite sampling graph. The products of the paired entries supply the row correlations needed for spectral initialization in regimes where uniform sampling yields too few wedges. For fixed rank, incoherence, tensor order, and condition number, wedge sampling enables polynomial-time recovery with nearly linear sample complexity in $n$. The spectral initializer can be combined with existing refinement procedures using only an additional $\tilde O(n)$ uniformly sampled entries. Spectral refinement achieves weak recovery to any fixed relative Frobenius error. For symmetric third-order tensors with incoherent canonical polyadic factors, g
    
[^48]: 任意时刻有效的保形风险控制

    Anytime-Valid Conformal Risk Control

    [https://arxiv.org/abs/2602.04364](https://arxiv.org/abs/2602.04364)

    本文提出了任意时刻有效的保形风险控制方法，将误差控制从固定大小校准数据集上的平均有效性扩展为随校准数据累积增长时在任意时刻均以高概率保持有效，并证明了该保证渐近紧致且修正速率不可再改进。

    

    预测集为预测任务中的不确定性量化提供了一种手段。利用留出的校准数据，保形预测和风险控制能够以计算高效的方式生成具有统计上有效误差控制的预测集。然而，在标准形式中，误差仅在固定大小的多个可能校准数据集上进行平均控制。在本文中，我们将这种控制扩展为：随着校准数据集的累积增长，在任意时间点上都能以高概率保持有效。我们利用基于分位数的论证推导出此类保证，并说明了所提出框架在涉及分布偏移的情境中的适用性。我们进一步证明了这些保证是渐近紧的，即风险从下方收敛到目标水平，并且我们的修正速率无法再被改进。最后，我们展示了该方法的实际性能。

    arXiv:2602.04364v2 Announce Type: replace-cross  Abstract: Prediction sets provide a means of quantifying the uncertainty in predictive tasks. Using held out calibration data, conformal prediction and risk control can produce prediction sets that exhibit statistically valid error control in a computationally efficient manner. However, in the standard formulations, the error is only controlled on average over many possible calibration datasets of fixed size. In this paper, we extend the control to remain valid with high probability over a cumulatively growing calibration dataset at any time point. We derive such guarantees using quantile-based arguments and illustrate the applicability of the proposed framework to settings involving distribution shift. We further show that our guarantees are asymptotically tight, in the sense that the risks converge to the target level from below, and that the rate of our correction cannot be improved. Finally, we demonstrate the practical performance o
    
[^49]: 在 $O(n\log n)$ 时间内学习多项Logit模型

    Learning Multinomial Logits in $O(n \log n)$ time

    [https://arxiv.org/abs/2601.04423](https://arxiv.org/abs/2601.04423)

    本文针对多项Logit（MNL）模型的学习问题，提出了自适应与非自适应算法，能够以 $O(\frac{n\log n}{\varepsilon^2})$ 次查询使每个选择板上的诱导选择分布在总变差距离 $\varepsilon$ 内逼近真实分布。

    

    多项Logit（MNL）模型由一个有限的物品全集 $[n]=\{1,\dots,n\}$ 组成，其中每个物品被赋予一个正权重。一次查询指定一个可接受的子集——称为一个“选择板”——模型以与权重成正比的概率从该选择板中选择一个物品。这种查询模型在文献中也被称为 Plackett–Luce 模型或条件采样预言机。尽管MNL模型已被广泛研究，但一个基本的计算问题仍未解决：在给定对选择板的查询访问的条件下，我们能以多高的效率学习各物品的权重，使得对于每一个选择板，模型诱导出的选择分布与真实分布之间的总变差距离不超过 $\varepsilon$？这一问题居于MNL学习的核心，并对现代推荐系统界面具有直接的影响。我们为该任务提供了自适应与非自适应两类算法。我们的第一个自适应算法进行 $O(\frac{n\log n}{\varepsilon^2})$ 次查询并精确学习……

    arXiv:2601.04423v2 Announce Type: replace-cross  Abstract: A Multinomial Logit (MNL) model is composed of a finite universe of items $[n]=\{1,\dots,n\}$, each assigned a positive weight. A query specifies an admissible subset---called a slate---and the model chooses one item from that slate with probability proportional to its weight. This query model is also known as the Plackett--Luce model or conditional sampling oracle in the literature. Although MNLs have been studied extensively, a basic computational question remains open: given query access to slates, how efficiently can we learn weights so that, for every slate, the induced choice distribution is within total variation distance $\varepsilon$ of the ground truth? This question is central to MNL learning and has direct implications for modern recommender system interfaces. We provide both adaptive and non-adaptive algorithms for this task. Our first adaptive algorithm makes $O(\frac{n\log n}{\varepsilon^2})$ queries and learns e
    
[^50]: 一种贝叶斯潜在类别强化学习框架，用于捕捉适应性、反馈驱动的出行行为

    A Bayesian latent class reinforcement learning framework to capture adaptive, feedback-driven travel behaviour

    [https://arxiv.org/abs/2512.14713](https://arxiv.org/abs/2512.14713)

    本文提出了一种基于贝叶斯潜在类别强化学习的框架，能够捕捉出行行为中个体偏好随时间演变的异质性，并通过驾驶模拟数据识别出三种不同适应策略的旅行者类型。

    

    许多出行决策涉及一定程度的经验形成，即个体随时间学习其偏好。同时，个体旅行者之间存在广泛的异质性，无论是在其潜在偏好还是这些偏好的演变方式上。本文提出了一种潜在类别强化学习（LCRL）模型，使分析人员能够捕捉这两种现象。我们将该模型应用于驾驶模拟器数据集，并通过变分贝叶斯方法估计参数。我们识别出三类不同的个体，它们在偏好适应方式上存在显著差异：第一类表现出情境依赖的偏好，并具有特定情境的开发倾向；第二类无论情境如何都遵循持续的开发策略；第三类则采用探索性策略并结合情境特定偏好。

    arXiv:2512.14713v1 Announce Type: cross  Abstract: Many travel decisions involve a degree of experience formation, where individuals learn their preferences over time. At the same time, there is extensive scope for heterogeneity across individual travellers, both in their underlying preferences and in how these evolve. The present paper puts forward a Latent Class Reinforcement Learning (LCRL) model that allows analysts to capture both of these phenomena. We apply the model to a driving simulator dataset and estimate the parameters through Variational Bayes. We identify three distinct classes of individuals that differ markedly in how they adapt their preferences: the first displays context-dependent preferences with context-specific exploitative tendencies; the second follows a persistent exploitative strategy regardless of context; and the third engages in an exploratory strategy combined with context-specific preferences.
    
[^51]: 低秩矩阵推断中的稀疏损坏：PCA基准测试

    Sparse corruption in low-rank matrix inference: the PCA benchmark

    [https://arxiv.org/abs/2511.11927](https://arxiv.org/abs/2511.11927)

    该论文研究了稀疏噪声（建模为有限平均连通度的加权图邻接矩阵）损坏下低秩信号的PCA推断问题，利用复制方法与种群动力学解析计算了最大特征值、特征向量分量密度及信号重叠度，并识别出随图连通度变化的两个信号强度转变点。

    

    主成分分析（PCA）是从含噪观测中提取低秩信号的标准工具。众所周知，当将PCA应用于被稠密、均匀噪声损坏的秩一信号时，在大矩阵尺寸极限下会出现BBP转变：离群特征值的涌现与相应特征向量的对齐发生在同一临界信号强度处。本文研究稀疏噪声损坏的情形。噪声矩阵被建模为具有有限平均连通度的加权无向图的邻接矩阵。利用复制方法，我们通过由种群动力学求解的递归分布方程，解析地计算了典型的最大特征值、最大特征向量分量密度以及与信号的重叠度。我们识别出两个作为图连通度函数的信号强度转变：θ_crit，标记通过最大特征向量实现信号恢复……

    arXiv:2511.11927v3 Announce Type: replace-cross  Abstract: Principal Component Analysis (PCA) is a standard tool for extracting a low-rank signal from noisy observations. It is known that when applying PCA to a rank-one signal corrupted by a dense, homogeneous noise, in the large matrix size limit, the BBP transition occurs, where the emergence of an outlying eigenvalue and the alignment of the corresponding eigenvector occur at the same critical signal strength. Here we study the case of sparse noise corruption. The noise matrix is modelled as the adjacency matrix of a weighted undirected graph with finite average connectivity. Using the replica method, we analytically compute the typical top eigenvalue, the top eigenvector component density, and the overlap with the signal, through recursive distributional equations solved by population dynamics. We identify two signal-strength transitions as functions of graph connectivity: $\theta_{\rm crit}$, marking signal recovery by the top eig
    
[^52]: 适者生存的Cox模型：针对时间-事件数据的枢轴变量选择方法

    Survival of the fittest Cox model: Pivotal variable selection for time-to-event data

    [https://arxiv.org/abs/2510.19374](https://arxiv.org/abs/2510.19374)

    本文通过对Cox比例风险模型的偏似然函数进行平方根变换，使正则化参数选择具有渐近枢轴性（精确消除基线风险和删失率的影响），从而显著提升了生存分析中变量选择的准确性和可靠性。

    

    我们重新审视了Cox比例风险模型，以改进生存分析中的变量选择。通过对偏似然函数进行平方根变换，使得正则化参数的选择在渐近意义下成为枢轴量：基线风险与删失率被精确地抵消，而删失在时间上的分布方式仅通过一个阶为 $1/\log n$ 的有界修正项产生影响。利用鞅残差，我们通过蒙特卡洛方法计算该惩罚水平，而无需对设计矩阵进行重新排序。所得的准则融合了BIC等信息准则与lasso等惩罚回归方法的优点，取两者之长。在模拟数据和真实数据上，我们的方法显著优于日常使用的最先进方法。

    arXiv:2510.19374v3 Announce Type: replace-cross  Abstract: We revisit Cox's proportional hazards model to improve variable selection in survival analysis. A square-root transformation of the partial likelihood renders the selection of the regularization parameter asymptotically pivotal: the baseline hazard and the censoring rate cancel exactly, and the way censoring is spread over time only enters through a bounded correction of order $1/\log n$. Using martingale residuals, we compute this penalty level by Monte Carlo without reordering the design matrix. The resulting criterion borrows from information criteria such as BIC and from penalized regression methods such as the lasso, taking the best of both. On simulated and real data, our method substantially improves upon state-of-the-art approaches used daily.
    
[^53]: TabPFN：一个模型统御一切？

    TabPFN: One Model to Rule Them All?

    [https://arxiv.org/abs/2505.20003](https://arxiv.org/abs/2505.20003)

    本文从统计学视角将TabPFN阐释为一种近似贝叶斯推断方法，并证明这一表格数据基础模型在半监督参数估计、协变量偏移预测和异质性处理效应等任务上有时能超越专门的先进方法。

    

    Hollmann等人（Nature 637 (2025) 319-326）最近提出了TabPFN，这是一种基于Transformer的深度学习模型，用于表格数据的回归与分类任务。他们声称该模型“在样本量不超过10,000的数据集上以显著优势超越所有先前的方法，且训练时间大幅减少”。此外，他们将TabPFN称为表格数据的“基础模型”，因为它能够支持“数据生成、密度估计、学习可复用的嵌入以及微调”。在本文中，我们面向统计学界对TabPFN的工作原理进行了针对性的阐释，重点强调其可理解为近似贝叶斯推断。随后，我们探讨了TabPFN对统计学领域的意义：我们展示了开箱即用的TabPFN在某些情况下能够超越专为半监督参数估计、协变量偏移下的预测以及异质性处理效应估计而设计的最先进方法。

    arXiv:2505.20003v3 Announce Type: replace  Abstract: Hollmann et al. (Nature 637 (2025) 319-326) recently introduced TabPFN, a transformer-based deep learning model for regression and classification on tabular data, which they claim "outperforms all previous methods on datasets with up to 10,000 samples by a wide margin, using substantially less training time." Furthermore, they have called TabPFN a "foundation model" for tabular data, as it can support "data generation, density estimation, learning reusable embeddings and fine-tuning". In this paper, we provide a tailored explanation of how TabPFN works for a statistics audience, by emphasizing its interpretation as approximate Bayesian inference. We then explore the significance of TabPFN to the field of statistics: We show that an out-of-the-box application of TabPFN can sometimes outperform specialized state-of-the-art methods for semi-supervised parameter estimation, prediction under covariate shift, and heterogeneous treatment ef
    
[^54]: tempdisagg：一个用于时间序列数据时间分解的Python框架

    tempdisagg: A Python Framework for Temporal Disaggregation of Time Series Data

    [https://arxiv.org/abs/2503.22054](https://arxiv.org/abs/2503.22054)

    tempdisagg是一个开源Python框架，实现了Chow-Lin、Denton等经典时间分解与基准调整方法，并通过基于三角分解的统一GLS计算引擎将似然评估的计算复杂度从O(n²n_l)降至O(n·n_l²)。

    

    tempdisagg是一个开源Python包，用于从低频基准数据以及可选的高频指标中估计高频时间序列，并确保估计值能够精确聚合到基准数据上。该包实现了基于回归的Chow-Lin、Fernández和Litterman估计器、动态Chow-Lin模型、Denton和Denton-Cholette基准调整程序以及均匀分布方法，其公式和默认设置与R语言包tempdisagg保持一致，同时将Proietti的自回归分布滞后模型作为实验性扩展纳入其中。所有广义最小二乘（GLS）估计器共享同一个计算引擎，其中残差协方差通过其三角分解因子来应用，使得一次似然函数评估的计算成本从O(n²n_l)降低到O(n·n_l²)。该包提供完整的统计推断功能，包括系数估计、标准误、基于似然的评价准则、预测方差以及自回归的完整轮廓分析。

    arXiv:2503.22054v2 Announce Type: replace  Abstract: tempdisagg is an open-source Python package that estimates a high-frequency time series from low-frequency benchmarks and, optionally, high-frequency indicators, such that the estimates aggregate exactly to the benchmarks. It implements the regression-based estimators of Chow-Lin, Fern\'andez and Litterman, the dynamic Chow-Lin model, the Denton and Denton-Cholette benchmarking procedures and uniform distribution, with the formulas and defaults of the R package tempdisagg, together with the autoregressive distributed-lag model of Proietti as an experimental extension. All generalised least squares estimators share one engine in which the residual covariance is applied through its triangular factor, so that one likelihood evaluation costs $O(n\,n_l^2)$ instead of $O(n^2 n_l)$. The package reports complete inference (coefficients, standard errors, likelihood-based criteria, prediction variances and a full profile of the autoregressive 
    
[^55]: 马尔可夫噪声下随机逼近与强化学习的几乎必然收敛速率与集中性

    Almost Sure Convergence Rates and Concentration of Stochastic Approximation and Reinforcement Learning with Markovian Noise

    [https://arxiv.org/abs/2411.13711](https://arxiv.org/abs/2411.13711)

    本文通过一种使用长度递减区间的平均ODE新颖离散化技术，首次为带马尔可夫噪声的压缩随机逼近算法建立了几乎必然收敛速率和亚Weibull尾部的最大集中性界，并首次应用于无基于计数学习率的Q学习及离策略时序差分学习。

    

    据我们所知，本文首次为具有马尔可夫噪声的一般压缩随机逼近算法建立了几乎必然收敛速率以及具有亚Weibull尾部的最大集中性界。作为推论，我们还得到了 $L^p$ 中的收敛速率。我们成功的关键在于对随机逼近算法的平均常微分方程（ODE）采用了一种新颖的离散化方法，即使用长度递减（而非恒定）的区间。在应用方面，我们首次得到了无需基于计数的 learning rates 的 $Q$-学习的几乎必然收敛速率，以及离策略时序差分学习（off-policy temporal difference learning）的首个最大集中性界，两者均基于马尔可夫样本。

    arXiv:2411.13711v2 Announce Type: replace  Abstract: This paper establishes, to our knowledge, the first almost sure convergence rate and the first maximal concentration bound with sub-Weibull tails for general contractive stochastic approximation algorithms with Markovian noise. As a corollary, we also obtain convergence rates in $L^p$. Key to our successes is a novel discretization of the mean ODE of stochastic approximation algorithms using intervals with diminishing (instead of constant) length. Applications yield the first almost sure convergence rate for $Q$-learning without count-based learning rates and the first maximal concentration bound for off-policy temporal difference learning, both with Markovian samples.
    
[^56]: 设计算法推荐以实现人机互补

    Designing Algorithmic Recommendations to Achieve Human-AI Complementarity

    [https://arxiv.org/abs/2405.01484](https://arxiv.org/abs/2405.01484)

    本文提出了一种基于因果推断潜在结果框架的算法设计方法，将算法推荐与工具变量相联系，无需对推荐效果做先验假设即可设计出实现人机互补的推荐算法。

    

    arXiv:2405.01484v3 公告类型：replace-cross 摘要：算法经常是辅助而非取代人类决策者。然而，算法的设计与分析通常聚焦于预测结果，并未明确建模其对人类决策的影响。鉴于实证证据一再表明算法辅助工具无法改善人类决策，算法助手的设计与其角色之间的这种差异尤其令人担忧。在本文中，我们对推荐算法的设计进行了形式化，使其能够辅助人类决策者，而无需对推荐如何影响决策做出限制性的先验假设。我们构建了一个算法设计问题，利用因果推断中的潜在结果框架来建模推荐对人类决策者二元处理选择的影响。在此模型中，我们建立了算法推荐与工具变量之间的联系。

    arXiv:2405.01484v3 Announce Type: replace-cross  Abstract: Algorithms frequently assist, rather than replace, human decision-makers. However, the design and analysis of algorithms often focus on predicting outcomes and do not explicitly model their effect on human decisions. This discrepancy between the design and role of algorithmic assistants becomes particularly concerning in light of empirical evidence that suggests that algorithmic assistants again and again fail to improve human decisions. In this article, we formalize the design of recommendation algorithms that assist human decision-makers without making restrictive ex-ante assumptions about how recommendations affect decisions. We formulate an algorithmic-design problem that leverages the potential-outcomes framework from causal inference to model the effect of recommendations on a human decision-maker's binary treatment choice. Within this model, we make a connection between algorithmic recommendations and instrumental variab
    
[^57]: 预测-优化框架中的主动学习：一种基于间隔的方法

    Active Learning in the Predict-then-Optimize Framework: A Margin-Based Approach. (arXiv:2305.06584v1 [cs.LG])

    [http://arxiv.org/abs/2305.06584](http://arxiv.org/abs/2305.06584)

    本文提出了预测-优化框架中的首个主动学习方法，通过基于间隔的准则和SPO损失的可求解替代函数来决定是否请求样本标签，在提供理论泛化保证的同时有效降低标签复杂度。

    

    我们在预测-优化框架中开发了首个主动学习方法。具体而言，我们开发了一种学习方法，该方法从无标签数据流中依次决定是否请求特征样本的“标签”，其中标签对应于用于决策优化的模型参数。我们的主动学习方法是首个由预测参数所导致的决策误差（即智能预测-优化损失，SPO损失）直接驱动的方法。受SPO损失结构的启发，我们的算法采用基于间隔的准则，利用退化距离的概念，并在收集的数据上最小化SPO损失的一个易于求解的替代函数。特别地，我们开发了一种高效的主动学习算法，包含硬拒绝和软拒绝两种变体，每种变体都具有理论上的超额风险（即泛化）保证。我们进一步推导了标签复杂度（即请求标签的样本数量）的界限。

    We develop the first active learning method in the predict-then-optimize framework. Specifically, we develop a learning method that sequentially decides whether to request the "labels" of feature samples from an unlabeled data stream, where the labels correspond to the parameters of an optimization model for decision-making. Our active learning method is the first to be directly informed by the decision error induced by the predicted parameters, which is referred to as the Smart Predict-then-Optimize (SPO) loss. Motivated by the structure of the SPO loss, our algorithm adopts a margin-based criterion utilizing the concept of distance to degeneracy and minimizes a tractable surrogate of the SPO loss on the collected data. In particular, we develop an efficient active learning algorithm with both hard and soft rejection variants, each with theoretical excess risk (i.e., generalization) guarantees. We further derive bounds on the label complexity, which refers to the number of samples who
    
[^58]: 矩阵计算的方差缩减及其在高斯过程中的应用

    Variance Reduction for Matrix Computations with Applications to Gaussian Processes. (arXiv:2106.14565v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2106.14565](http://arxiv.org/abs/2106.14565)

    本文提出了一种通过矩阵分解来进行矩阵计算的方差缩减方法，并展示了在某些问题上可以实现任意更好的随机性能。此外，本文提出的矩阵乘积迹的分解估计器与对半正定矩阵行列式的对数估计器均有显著的效率提高，可以在实际的贝叶斯优化问题中得到应用。

    

    随着计算速度和内存容量的不断提高，方法学上的进步也为随机模拟的性能带来了显著提高。本文聚焦于通过矩阵分解来进行矩阵计算的方差缩减。我们提供了现有方差缩减方法在估计大型矩阵元素时未能利用矩阵分解所带来的方差缩减，而如何通过计算矩阵的平方根分解，在某些情况下能够实现任意更好的随机性能。此外，我们提出了一种矩阵乘积迹的分解估计器，并在数值上证明，在某些高斯过程对数似然函数估计的问题上，该估计器的效率可以提高1000倍。此外，我们提供了一种对半正定矩阵行列式的对数估计器，其方差比文献中使用的标准估计器快速衰减。我们在实际的贝叶斯优化问题中说明了我们提出的方法的实用性。

    In addition to recent developments in computing speed and memory, methodological advances have contributed to significant gains in the performance of stochastic simulation. In this paper, we focus on variance reduction for matrix computations via matrix factorization. We provide insights into existing variance reduction methods for estimating the entries of large matrices. Popular methods do not exploit the reduction in variance that is possible when the matrix is factorized. We show how computing the square root factorization of the matrix can achieve in some important cases arbitrarily better stochastic performance. In addition, we propose a factorized estimator for the trace of a product of matrices and numerically demonstrate that the estimator can be up to 1,000 times more efficient on certain problems of estimating the log-likelihood of a Gaussian process. Additionally, we provide a new estimator of the log-determinant of a positive semi-definite matrix where the log-determinant 
    

