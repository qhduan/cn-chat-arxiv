# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Markov Chain Convergence in Kernel-Induced Geometry](https://arxiv.org/abs/2610.13916) | 该论文在核诱导几何的收缩性与谱隙假设下，建立了马尔可夫链经验占据测度收敛的非渐近高概率界，并将结果应用于单轨迹动力系统的一步预测，为非线性最小二乘估计器导出了快速学习速率。 |
| [^2] | [Hitting-Time Bounds for Asynchronous Q-Learning](https://arxiv.org/abs/2610.13901) | 该论文在仅要求状态-动作链不可约（允许周期性）的弱条件下，证明了零初始化、常数步长的异步Q学习只需 $\widetilde{O}(H^4/(\mu_{\min}\varepsilon^2)+\mathfrak h_\star)$ 次转移即可高概率达到 $\varepsilon$ 上确界范数精度，且初始化成本以反向链最大期望命中时间 $\mathfrak h_\star$ 的加性形式出现。 |
| [^3] | [SLICE: Sequential Likelihood-Free Inference via Conformal E-Values](https://arxiv.org/abs/2610.13855) | 提出SLICE方法，利用共形e值构建针对难解似然模拟模型的精确随时有效置信序列，在持续监测下保持有效，并通过摊销神经比率估计器近似理想似然比得分。 |
| [^4] | [Finite-Sample Guarantees for Fully Discrete Conditional Diffusion Models](https://arxiv.org/abs/2610.13851) | 本文为以有限去噪步数实现的条件扩散模型建立了首个有限样本理论保证，通过将分数训练目标与采样网格对齐来联合分析统计估计与数值离散化误差，获得了近优的KL散度与总变差误差收敛速率。 |
| [^5] | [Constrained Probabilistic Numerics: Gaussian Process Solvers for Linear PDEs under Linear Inequality Constraints](https://arxiv.org/abs/2610.13804) | 该论文提出了一种支持线性不等式约束（如非负性、单调性）的高斯过程PDE求解方法，证明等式与不等式条件化可以顺序进行，并通过截断高斯矩实现了非高斯后验的可计算性。 |
| [^6] | [From Pixels to Privacy-Sensitive Features: An Information-Theoretic Framework for Reconstruction Bounds](https://arxiv.org/abs/2610.13768) | 该论文提出了一个基于率失真理论的信息论框架，能在训练前计算出 DP-SGD 训练模型的重构误差下界，即使面对可观测训练过程、知晓其余训练数据和总体分布的最坏情况攻击者依然成立，并允许分析者选择面向隐私敏感特征的误差度量表示。 |
| [^7] | [How Well Do Prior-Data Fitted Networks Amortize Bayesian Inference?](https://arxiv.org/abs/2610.13764) | 本文将先验数据拟合网络（PFNs）的部署误差分解为先验、上下文和摊销误差，发现PFNs虽能很好地恢复目标后验预测分布，但在罕见设置下仍存在残余摊销误差，而贝叶斯模型平均方法可有效减少这种不确定性并提升预测性能。 |
| [^8] | [Calibration in One Go: Jointly Learning Accurate and Calibrated Neural Networks](https://arxiv.org/abs/2610.13693) | 该论文提出通过双层 Stackelberg 优化将校准器与主干网络端到端联合训练，无需单独的校准数据集或专用校准损失，仅引入校准器学习率这一个超参数，即可同时学习到既准确又校准良好的神经网络。 |
| [^9] | [Not All Dimensions Are Equal: Outlier-Aware Attention Linearization with Grouped Quadratic Kernels](https://arxiv.org/abs/2610.13674) | 该论文发现预训练注意力中少数离群维度主导了线性化的近似误差，并提出维度感知的OAL方法，通过Hadamard乘积表述将全局系数放宽为维度特定的线性项与跨维度二次交互，再以分组二次核在更少参数下实现高保真的注意力线性化。 |
| [^10] | [Scalable Sparse Autoregressive Variational Inference for Non-Gaussian Spatial Fields](https://arxiv.org/abs/2610.13656) | 该论文提出了一种可扩展的变分推断框架，通过在变分族中镜像稀疏最近邻自回归先验结构、使用归一化流条件分布和共享神经条件器摊销参数，并以双随机优化实现对数多项式迭代复杂度，从而高效解决大规模非高斯空间场的后验推断难题。 |
| [^11] | [A Critical Look at Power Sampling when Test Time Sharpening LLMs](https://arxiv.org/abs/2610.13653) | 该论文证明基于MCMC/SMC的幂采样方法在现实计算预算下无法收敛到目标幂分布，其收益可由简单的Best-of-N采样复现，表明其成功源于筛选高概率序列而非真正近似幂分布。 |
| [^12] | [Kernel-based External Validity of Experimental Studies](https://arxiv.org/abs/2610.13625) | 该论文提出用核方法（最大均值差异MMD）形式化定义实验研究的外部有效性，并建立了研究规模与MMD分位数之间的渐近幂律关系，从而能够从小规模初步研究估计出具有外部有效性的研究所需规模。 |
| [^13] | [Some remarks on the connection between generative diffusion and optimal transport](https://arxiv.org/abs/2610.13614) | 本文揭示了生成扩散与最优传输之间的三个简单联系：数据加噪的联合分布在每个时刻都是熵正则化最优传输问题的解（因此反向扩散等价于逐渐降低熵正则化的退火过程），且扩散与流匹配中分离数据模式的动力学不稳定性，在最优传输中表现为生成初期一个被熵正则化平滑的指数级薄边界层。 |
| [^14] | [CorrFlow: Exact and Permutation-Equivariant Flow Matching on the Correlation Elliptope](https://arxiv.org/abs/2610.13606) | CorrFlow通过广义Fisher变换将满秩相关矩阵双射映射到无约束坐标，再用置换等变的图流匹配网络学习其经验分布，从而无需投影即可精确保证输出位于相关椭圆面上，并对变量排序保持等变性。 |
| [^15] | [Beyond Discrepancy: Oracle Drift in Nonstationary Weighted Risk Minimization](https://arxiv.org/abs/2610.13598) | 该论文提出用模型类内最优预测器的移动（即“最优预测器漂移”）来直接刻画非平稳加权风险最小化中的漂移误差，证明传统差异型界过于保守，并在线性回归和非参数回归中得到了更精确的漂移界。 |
| [^16] | [Probabilistic Forecasting of Electric Vehicle Charging in a Time-varying Hierarchical Infrastructure](https://arxiv.org/abs/2610.13588) | 提出将分层预测与处理缺失数据的保形预测相结合，在持续演化的电动汽车充电基础设施分层结构中，为从单个充电点到国家级的每个聚合节点生成有效的预测区间。 |
| [^17] | [Defence against the dark arts of distillation: Unsupervised discovery of subliminal learning](https://arxiv.org/abs/2610.13536) | 提出了一种无需预先知晓目标特征的无监督防御方法，仅通过分析学生模型的参数更新即可发现、解释并抑制从共享基础参数的教师模型阈下继承的失对齐行为特征。 |
| [^18] | [Gaussian BROJA Partial Information Decomposition in Closed Form](https://arxiv.org/abs/2610.13464) | 本文推导出了高斯BROJA部分信息分解优化问题的闭式解，使一般多变量高斯情形下的分解可以直接计算，并将高维计算的速度提升超过两个数量级。 |
| [^19] | [Learning by Compression: Risk Estimates Attain the Law-of-Large-Numbers Rate](https://arxiv.org/abs/2610.13442) | 本文证明基于优先压缩的学习算法的风险界中并非固有存在对数因子，其风险估计对所有复杂度取值一致地达到 $\boldsymbol{k}/N + O(1/\sqrt{N})$ 的大数定律速率，与经验估计速率相同。 |
| [^20] | [One Thing Leads to Another: The Essentials of Time Series Causal Discovery](https://arxiv.org/abs/2610.13430) | 本教程论文首次为统计学、动力系统和机器学习等不同研究社区的时间序列因果发现方法提供了统一概述，系统介绍了其关键原理、局限性、假设、评估基准方法以及在农业、地球科学、医疗保健和人工智能系统等下游任务中的重要性。 |
| [^21] | [Model-free Bayesian Clustering and Multimodal Posterior Summarisation](https://arxiv.org/abs/2610.13397) | 本文提出了一种无模型、噪声感知的贝叶斯聚类框架，通过鞅后验避免MCMC采样，并利用Wasserstein型摘要方法将多模态后验高效压缩为少量加权代表，同时边缘化了密度水平的不确定性。 |
| [^22] | [BLOKS: Scaling Kernel Solvers with Sketched Block Preconditioning](https://arxiv.org/abs/2610.13382) | BLOKS提出了一种将块Jacobi近似与草图化相结合的新颖预条件子，在加速核岭回归近似求解收敛速度的同时有效控制了内存成本，兼顾了效率与模型表达能力。 |
| [^23] | [Fast Gradient-Based Approximation of Continuous Dynamic Time Warping](https://arxiv.org/abs/2610.13129) | 本文提出了 FastCDTW，一种高效的连续动态时间规整（CDTW）近似求解器，证明了其在温和条件下的可微性，并针对分段线性规整函数提供精确闭式解、针对平滑规整函数提供无偏蒙特卡洛估计两种计算策略，大幅降低了计算成本。 |
| [^24] | [Why Heavy-Tailed Weight Spectra Predict Model Quality](https://arxiv.org/abs/2610.13123) | 该论文通过概率框架推导出仅基于DNN最后一层权重的边际似然，首次从统计学角度解释了为什么训练权重矩阵中的重尾谱分布能够有效预测模型质量。 |
| [^25] | [Forecasting Corpus-Sampling Disagreement in Masked Text Reconstruction](https://arxiv.org/abs/2610.12987) | 该论文推导出一个一阶公式，可仅凭参考语料库预测两个独立语料库上训练的掩码文本重建预测器之间的期望分歧，并发现语料库扩大四倍仅使分歧降低5–40%，远低于1/n缩放律的预期，其根源在于罕见上下文的平滑效应。 |
| [^26] | [Scan-Parallel Variational Inference for event-driven Latent Jump Models](https://arxiv.org/abs/2610.12986) | 提出潜在跳跃模型（LJMs），通过泊松事件驱动的可学习瞬时跳跃来显式建模不规则时间序列中未观测冲击的时间与幅度，并配套开发了扫描并行变分推断方法用于高效训练。 |
| [^27] | [Spatial Prediction with Guided Diffusion](https://arxiv.org/abs/2610.12967) | 本文提出利用机器学习中的引导扩散模型，在似然函数已知且可微的条件下求解一大类空间预测问题，并可灵活应用于高斯与非高斯过程模型。 |
| [^28] | [From Imbalance to Evidence: Deep Nonparametric Testing for Complex Data under Severe Sample Imbalance](https://arxiv.org/abs/2610.12946) | 提出增强型不平衡感知深度分类器双样本检验，创新性地将严重样本不平衡从分类器训练的不利因素转化为可重复利用的检验证据来源，从而在高不平衡情形下实现对图等复杂数据更可靠的分布比较。 |
| [^29] | [Sparse Dictionary Learning for Measure-Valued Data via Linearized Optimal Transport: Efficient Algorithms and Sample Complexity](https://arxiv.org/abs/2610.12924) | 该论文提出一个基于线性化最优传输的稀疏字典学习框架，将概率测度嵌入希尔伯特空间并通过稀疏自编码器学习数据自适应字典，实现对高光谱图像、点云及大语言模型文档嵌入等测度值数据的高效可解释表示，同时给出以原子类度量复杂度刻画的泛化界和样本复杂度理论保证。 |
| [^30] | [Beyond Second-Order Information: Paid Logistic Regression with Noisy Features](https://arxiv.org/abs/2610.12919) | 该文提出带付费噪声特征的在线逻辑回归问题，证明与付费线性回归不同，仅凭协方差这类二阶信息已不足以刻画最优损失——相同协方差的噪声分布可产生不同损失、更低方差甚至导致更高损失，因此需要超越二阶信息的新结构刻画。 |
| [^31] | [Generalized Determinantal Point Processes for Attraction and Repulsion with Distinct Interaction Kernels](https://arxiv.org/abs/2610.12894) | 提出行列式比点过程（DRPP），利用两个对称正定核行列式之比分别建模排斥与吸引，在保持特征可解释性的同时，借助辅助变量表示实现高效MCMC采样与最大似然学习。 |
| [^32] | [Universal Bounds for Self-Normalized Kullback-Leibler Estimation](https://arxiv.org/abs/2610.12851) | 本文证明了基于归一化重要性权重的KL散度估计器存在通用有限样本界——其期望超过目标-提议KL散度至多 $\log 2$，且无需重要性权重有界或高阶矩有限，并将该结果应用于生成模型的Soft Best-of-$N$测试时对齐与相对熵编码。 |
| [^33] | [Two-way Guidance: Simultaneous Learning of Summary Statistics for Generative Bayes](https://arxiv.org/abs/2610.12850) | 提出“双向引导”机制，通过汇总统计量与后验分布的同步联合学习，使基于模拟的贝叶斯推断在观测规模不断增大时仍能有效恢复集中化的后验分布。 |
| [^34] | [Scalable Learning of Determinantal Point Processes with Kernel-Structured Interactions](https://arxiv.org/abs/2610.12829) | 该论文提出一种利用非线性核随机特征进行随机优化的可扩展行列式点过程学习方法，通过全秩交互建模与替代函数设计，突破了结构约束的局限，显著改善了表达能力与计算可扩展性之间的权衡，并以极高的参数效率获得了更好的泛化性能。 |
| [^35] | [Minimax Rates for Estimating Powered Sliced Wasserstein Functionals](https://arxiv.org/abs/2610.12740) | 本文建立了幂次切片Wasserstein泛函两样本估计的极小极大速率——当 $r<2p$ 时为 $N^{-(r-p)/r}$ 阶，当 $r>2p$ 时为 $N^{-1/2}$ 阶——并证明自然的代入估计量在两种情形下均达到速率最优。 |
| [^36] | [Kernelized Parameter Sensitivity Atlases for Amortized Conditional Interpretability](https://arxiv.org/abs/2610.12737) | 该论文提出 KPSA，通过经验神经正切核的分块分解构建参数组敏感性的核化图谱，用轻量级核函数摊销式地为任意新输入估计敏感性，从而以远低于逐样本反向传播的成本实现输入条件化的模型可解释性。 |
| [^37] | [A Distributional Framework for Behavioral Characterization and Risk Management of Large Language Models](https://arxiv.org/abs/2610.12718) | 该论文提出一个基于蒙特卡洛模拟构建经验响应分布的框架，通过语义干预量化分布变化，将大语言模型的随机行为刻画与模型风险管理、治理以及黑盒模型监控联系起来。 |
| [^38] | [Amortised Inference through One-Step Implicit Sampling](https://arxiv.org/abs/2610.12679) | 提出单步隐式采样器（OSIS），利用短时MCMC动力学定义局部目标使建模分布成为MCMC核的不动点，从而实现高维贝叶斯后验分布的单步摊销采样。 |
| [^39] | [Transformer-Based Masked Diffusion Models Provably Learn Distributions with Latent Structure](https://arxiv.org/abs/2610.12544) | 本文首次为基于Transformer的掩码扩散模型建立了端到端统计保证，证明其能以由潜在结构（隐马尔可夫树混合模型）决定的样本复杂度学习结构化离散分布，突破了以往理论中 $V^L$ 维度灾难的限制。 |
| [^40] | [Faithfulness Without Anchors: A Position Paper on the Structural Misalignments of Explainable AI Faithfulness Metrics for Feature Importance](https://arxiv.org/abs/2610.12493) | 本文是一篇立场论文，指出当前面向特征重要性的可解释AI忠实性度量存在结构性错位，并强调XAI实际上更多扮演调试工具的角色而非建立信任的手段。 |
| [^41] | [On the estimation and validity of AI time horizons---a statistical look at the METR plot](https://arxiv.org/abs/2610.12466) | 本文用样条函数和项目反应理论重新估计了METR的AI时间视野，发现人类时间与AI难度的关系并非全局线性（在2至30分钟区间近乎平坦），因此同样10倍的提升，从3分钟到30分钟远比从30分钟到5小时容易实现。 |
| [^42] | [Efficient quadratic entropy with distance sketches](https://arxiv.org/abs/2610.11976) | 本文提出了一种基于随机特征嵌入、投影和控制变量技术的可扩展二次熵近似方法，并在文献计量学应用中仅凭引用和文本特征揭示了论文、领域和机构的跨学科影响力。 |
| [^43] | [Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization](https://arxiv.org/abs/2610.08969) | 该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。 |
| [^44] | [Explicit Asymptotic Bounds for Sequential Calibration Beyond $T^{2/3}$](https://arxiv.org/abs/2610.07623) | 该论文提出新的两阶段递归标记策略并改进归约方法，首次为序贯校准问题建立了超越 $T^{2/3}$ 的显式渐近界 $O(T^{0.662942288})$。 |
| [^45] | [Sharp Statistical Rates for Asynchronous TD Learning with Markovian Data](https://arxiv.org/abs/2609.38880) | 该论文为基于单条马尔可夫轨迹的表格型TD学习建立了精确的末次迭代样本复杂度界 $\widetilde O\left(\frac{H^3}{\mu_{\min}\varepsilon^2}+\frac{t_{\operatorname{mix}}}{\mu_{\min}}\right)$，且该界对常数步长与递减步长调度均成立。 |
| [^46] | [Robust LassoNet: Enhancing Feature Selection in Neural Networks via Robust Loss Functions](https://arxiv.org/abs/2609.38263) | 本文提出Robust LassoNet，通过在LassoNet中引入Huber、Cauchy、Tukey双平方等鲁棒损失函数，在保留原始优化框架的同时，显著提升了神经网络在数据污染情况下的特征选择准确性与预测性能。 |
| [^47] | [Provable Parameter-Free Fixed-Point Algorithms with Linear Convergence Rates](https://arxiv.org/abs/2608.09022) | 本文提出了两种无需压缩因子先验知识的无参数Halpern不动点迭代算法，并证明了对不动点残差和到唯一不动点距离的显式线性收敛率。 |
| [^48] | [Particle-based Generalised Stochastic Optimisation](https://arxiv.org/abs/2608.02844) | 本文提出了针对不可解梯度损失函数的基于扩散的广义随机粒子优化框架，通过平均场动力学及其交互粒子近似统一了多种现有算法，证明了指数收敛性与非渐近误差界，并据此构建了高阶Langevin和动量优化器。 |
| [^49] | [Equality, Equity, and Causality in Fairness Research: A Commentary on Cheng (2026)](https://arxiv.org/abs/2607.17679) | 该评论在Cheng（2026）对心理测量学与AI/ML公平性系统比较的基础上，进一步探讨了平等与公平的概念区别以及因果性在公平性研究中的作用，为跨学科公平性研究的未来方向提供了指引。 |
| [^50] | [Unsupervised Identification and Removal of Spurious Correlations During Fine-Tuning](https://arxiv.org/abs/2605.27676) | 本文证明可以在无监督条件下从朴素LoRA微调的权重中识别出微调引入的伪相关潜在因素，并提出应消除伪相关本身而非潜在因素，以避免破坏预训练模型依赖该因素获得的有效任务信号。 |
| [^51] | [SURF: Steering the Scalarization Weight to Uniformly Traverse the Pareto Front](https://arxiv.org/abs/2605.20619) | 该论文提出SURF方法，通过将权重变化时解在帕累托前沿上的移动速度积分得到的累积弧长归一化为CDF，并利用其反函数调控标量化权重的采样，从而实现对帕累托前沿的均匀覆盖。 |
| [^52] | [Occam's Razor is Only as Sharp as Your ELBO](https://arxiv.org/abs/2604.25984) | 本文证明在贝叶斯线性回归中，基于ELBO的超参数学习会因高斯近似后验协方差结构的不同而产生系统性偏差：对角或低秩加对角协方差会低估信噪比导致欠拟合，而低秩协方差会高估信噪比导致过拟合。 |
| [^53] | [Locally Private Parametric Methods for Change-Point Detection](https://arxiv.org/abs/2602.13619) | 本文研究了本地差分隐私下的参数化变点检测问题，建立了以Chernoff信息为根本限制的有限样本极小极大上下界，并据此设计了分别适用于低隐私和高隐私机制的最优本地隐私变点检测算法，从而精确量化了隐私保护带来的统计代价。 |
| [^54] | [Federated Learning of Nonlinear Temporal Dynamics with Graph Attention-based Cross-Client Interpretability](https://arxiv.org/abs/2602.13485) | 本文提出一种垂直联邦学习框架，在原始数据不可共享、客户端观测异构且各客户端模型固定不可重训的约束下，利用图注意力机制实现跨客户端的非线性时间动态一步预测，并具备跨客户端可解释性。 |
| [^55] | [Fast Rates for Nonstationary Weighted Risk Minimization](https://arxiv.org/abs/2602.05742) | 本文将分布漂移下的超额风险分解为学习误差与漂移误差，对非平稳加权经验风险最小化建立了在任意权重类上一致成立的快速oracle界，并在结构突变与平滑漂移过程以及线性模型、基近似和神经网络等回归问题中刻画了最优权重。 |
| [^56] | [Robust Machine Learning for Regulatory Sequence Modeling under Biological and Technical Distribution Shifts](https://arxiv.org/abs/2601.14969) | 提出了一个结合机制仿真基准与MPRA真实数据的鲁棒性评估框架，用于量化深度学习模型在生物与技术分布偏移下进行调控序列预测时的性能退化、校准失效与可靠性。 |
| [^57] | [Bayesian dynamic scheduling of multipurpose batch processes under incomplete look-ahead information](https://arxiv.org/abs/2512.01093) | 该论文提出了一种基于从扰动概率分布中学习贝叶斯网络的动态调度方法，使多用途批处理过程能够在仅拥有不完整前瞻信息的现实条件下有效应对扰动，从而最小化长期成本和系统紧张性。 |
| [^58] | [Infinitesimal Operator World Models: Finite-Sample Guarantees for Offline Reinforcement Learning](https://arxiv.org/abs/2511.10383) | 该论文将强化学习与HJB方程相联系，提出基于受控扩散过程无穷小生成元（在再生核希尔伯特空间中学习）的算子理论世界模型，在生成元近似误差足够小时证明了值函数的全局收敛性，并为离线强化学习提供了有限样本保证。 |
| [^59] | [A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)](https://arxiv.org/abs/2510.22809) | 提出了一种基于信息论不确定性（惊奇度）的“无模型”学习框架，将学习视为对不确定性的测量，可直接从原始数据进行推理，支持数据的编辑与删除，并在异常检测等任务上达到最先进性能。 |
| [^60] | [Surrogate Graph Partitioning for Spatial Prediction](https://arxiv.org/abs/2510.07832) | 本文提出将代理图划分问题建模为混合整数二次规划，用于构建使分段内预测方差之和最小的空间分段，并针对其计算复杂度过高的问题开发了高效的近似求解方案。 |
| [^61] | [Approximation of differential entropy in Bayesian optimal experimental design](https://arxiv.org/abs/2510.00734) | 该论文针对贝叶斯最优实验设计提出了一种高效计算方法：当似然熵与设计无关或可显式计算时，期望信息增益的嵌套积分可简化为单积分，并通过蒙特卡洛或拟蒙特卡洛代理近似证据密度来高效估计微分熵。 |
| [^62] | [Optimizable Model Averaging for Ensembling Heterogeneous Model Outputs](https://arxiv.org/abs/2505.21857) | 提出可优化模型平均（OMA），一种黑盒集成方法，通过最小化正则化预测熵来学习非负组合权重以融合异构架构模型的预测概率，无需微调或参数合并即可改善性能与校准。 |
| [^63] | [Disentangling Total-Variance and Signal-to-Noise-Ratio Improves Diffusion Models](https://arxiv.org/abs/2502.08598) | 提出了一种总方差/信噪比解耦（TV/SNR）框架，使扩散模型噪声调度中的总方差和信噪比可被独立控制，并发现恒定总方差调度能够持续改进指数爆炸总方差的调度。 |
| [^64] | [Data-Driven Stochastic Optimal Control in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2407.16407) | 该论文提出一种完全数据驱动的方法，在动力学和成本函数均未知的情况下，通过在再生核希尔伯特空间中学习马尔可夫转移算子，并与随状态维度线性扩展的凸算子理论HJB递归相结合，实现非线性随机扩散系统的最优控制。 |
| [^65] | [Linearized Wasserstein dimensionality reduction with approximation guarantees](https://arxiv.org/abs/2302.07373) | 该论文提出LOT Wassmap算法，利用Sinkhorn距离和线性化最优传输等近似方案避免计算两两Wasserstein距离矩阵，从而高效揭示概率测度数据集在Wasserstein空间中的低维结构，并为此类近似下的嵌入质量提供了理论保证。 |

# 详细

[^1]: 核诱导几何中的马尔可夫链收敛性

    Markov Chain Convergence in Kernel-Induced Geometry

    [https://arxiv.org/abs/2610.13916](https://arxiv.org/abs/2610.13916)

    该论文在核诱导几何的收缩性与谱隙假设下，建立了马尔可夫链经验占据测度收敛的非渐近高概率界，并将结果应用于单轨迹动力系统的一步预测，为非线性最小二乘估计器导出了快速学习速率。

    

    我们研究了马尔可夫链在与正定核相关联的希尔伯特嵌入所诱导的几何（也称为最大均值差异，MMD）中的收敛性。在核诱导几何中的收缩性假设和标准谱隙假设下，我们建立了经验占据测度向不变测度收敛的非渐近高概率界。所得到的界分离了这两个条件的作用：谱隙决定了首项中的有效样本量，而核收缩仅通过线性修正项进入。作为应用，我们研究了从单条轨迹出发的动力系统一步预测问题，为非线性最小二乘估计器导出了快速学习速率。这些速率突出了核诱导几何中的混合性质与分析中所用的经典源条件和容量条件之间的相互作用。

    arXiv:2610.13916v1 Announce Type: cross  Abstract: We study the convergence of Markov chains in the geometry induced by Hilbertian embeddings associated with positive definite kernels, also called Maximum Mean Discrepancy (MMD). Under a contraction assumption in the kernel-induced geometry, and a standard spectral gap assumption, we establish non-asymptotic, high-probability bounds for the convergence of the empirical occupation measure to the invariant measure. The obtained bounds separate the roles of the two conditions: the spectral gap determines the effective sample size in the leading term, while the kernel contraction enters only through a linear correction. As an application, we study one-step prediction of dynamical systems from a single trajectory, deriving fast learning rates for a nonlinear least-squares estimator. The rates highlight the interplay between mixing properties in the kernel-induced geometry and the classical source and capacity conditions used in the analysis 
    
[^2]: 异步Q学习的命中时间界

    Hitting-Time Bounds for Asynchronous Q-Learning

    [https://arxiv.org/abs/2610.13901](https://arxiv.org/abs/2610.13901)

    该论文在仅要求状态-动作链不可约（允许周期性）的弱条件下，证明了零初始化、常数步长的异步Q学习只需 $\widetilde{O}(H^4/(\mu_{\min}\varepsilon^2)+\mathfrak h_\star)$ 次转移即可高概率达到 $\varepsilon$ 上确界范数精度，且初始化成本以反向链最大期望命中时间 $\mathfrak h_\star$ 的加性形式出现。

    

    我们研究了从有限马尔可夫决策过程的单条轨迹出发的标准异步Q学习的最后一次迭代。该轨迹遵循一个固定的行为策略，其状态-动作链是不可约的，且可能是周期的。对于折扣因子 $\gamma$，记 $H=(1-\gamma)^{-1}$，设 $\mu_{\min}$ 为最小的平稳状态-动作概率，$\mathfrak h_\star$ 为反向状态-动作链的最大期望命中时间。我们证明，对于 $0<\varepsilon\le1$，采用常数步长、零初始化的Q学习能够以高概率在 $\widetilde{O}(H^4/(\mu_{\min}\varepsilon^2)+\mathfrak h_\star)$ 次转移内实现上确界范数误差至多 $\varepsilon$。统计项保留了标准Q学习对有效视野的四次方依赖，而初始化成本以 $\mathfrak h_\star$ 的加性形式出现。对于非周期行为链，通过与混合时间的比较，可得到一个瞬态项 $\widehat{O}$……（摘要原文在此处被截断）

    arXiv:2610.13901v1 Announce Type: cross  Abstract: We study the last iterate of standard asynchronous Q-learning from a single trajectory of a finite Markov decision process. The trajectory follows a fixed behavior policy whose state--action chain is irreducible and may be periodic. For discount factor $\gamma$, write $H=(1-\gamma)^{-1}$, and let $\mu_{\min}$ be the minimum stationary state--action probability and $\mathfrak h_\star$ the largest expected hitting time of the reverse state--action chain. We prove that zero-initialized Q-learning with a constant step size achieves sup-norm error at most $\varepsilon$ with high probability using $\widetilde{O}(H^4/(\mu_{\min}\varepsilon^2)+\mathfrak h_\star)$ transitions, for $0<\varepsilon\le1$. The statistical term retains the quartic effective-horizon dependence of standard Q-learning, while initialization cost is additive in $\mathfrak h_\star$. For aperiodic behavior chains, a comparison with mixing time yields a transient term $\wide
    
[^3]: SLICE：基于共形e值的序贯无似然推断

    SLICE: Sequential Likelihood-Free Inference via Conformal E-Values

    [https://arxiv.org/abs/2610.13855](https://arxiv.org/abs/2610.13855)

    提出SLICE方法，利用共形e值构建针对难解似然模拟模型的精确随时有效置信序列，在持续监测下保持有效，并通过摊销神经比率估计器近似理想似然比得分。

    

    许多科学模型可以进行模拟但其似然函数难以处理，这使得频率学派的量化充满困难。我们为此类模型的参数构建了精确的、随时有效的置信序列。在每次观测时，利用来自每个候选原假设的全新模拟，将一个正的可预测得分共形化为条件e值；这些e值的乘积构成一个检验鞅，我们再对参数在该鞅上进行求逆。无论得分本身的准确性如何，有效性对任何此类得分都成立。我们证明了理想似然比得分能够最大化期望对数增长，这启发我们使用摊销式神经比率估计器来近似它。在合成基准测试、LLM API中模型路由的审计以及微透镜数据上，所得到的置信序列在持续监测下始终保持有效，而固定样本方法则无法做到这一点；同时，神经得分与精确似然比以及针对特定应用的得分相比仍具有竞争力。

    arXiv:2610.13855v1 Announce Type: cross  Abstract: Many scientific models can be simulated but have intractable likelihoods, making frequentist uncertainty quantification difficult. We construct exact anytime-valid confidence sequences for the parameters of such models. At each observation, fresh simulations from each candidate null conformalize a positive predictable score into a conditional e-value; their product is a test martingale, which we invert over the parameter. Validity holds for any such score, whatever its accuracy. We show that the oracle likelihood-ratio score maximizes expected log growth, which motivates approximating it with an amortized neural ratio estimator. On synthetic benchmarks, an audit of model routing in an LLM API, and microlensing data, the resulting confidence sequences stay valid under continuous monitoring, where fixed-sample methods do not, and neural scores remain competitive with exact likelihood ratios and application-specific scores.
    
[^4]: 全离散条件扩散模型的有限样本保证

    Finite-Sample Guarantees for Fully Discrete Conditional Diffusion Models

    [https://arxiv.org/abs/2610.13851](https://arxiv.org/abs/2610.13851)

    本文为以有限去噪步数实现的条件扩散模型建立了首个有限样本理论保证，通过将分数训练目标与采样网格对齐来联合分析统计估计与数值离散化误差，获得了近优的KL散度与总变差误差收敛速率。

    

    我们为从数据中学习并以有限个去噪步骤实现的条件扩散模型建立了有限样本理论保证。我们的方法将分数训练目标与采样网格对齐，从而能够对统计估计误差与数值离散化误差进行联合分析。在光滑条件密度模型下，我们推导出所生成分布误差的高概率上界，并量化了所需的分数评估次数。对于每一个光滑度指标 $\beta>0$，选择合适的网格后，无论是否采用提前停止，均可获得相对于原始目标的平均条件 Kullback–Leibler (KL) 散度 $\widetilde O_p(n^{-2\beta/(d_x+d_y+2\beta)})$。完整采样器与提前停止采样器均能达到平均目标总变差 (TV) 误差 $\widetilde O_p(n^{-\beta/(d_x+d_y+2\beta)})$。其中 $n$ 为样本量，$d_x,d_y$ 分别为数据维度与条件变量维度，误差在（原文此处截断）上取平均。

    arXiv:2610.13851v1 Announce Type: cross  Abstract: We establish finite-sample guarantees for conditional diffusion models learned from data and implemented with finitely many denoising steps. Our approach aligns the score-training objective with the sampling grid, allowing statistical estimation and numerical discretization errors to be analyzed together. Under a smooth conditional density model, we derive high-probability bounds on the generated distribution's error and quantify the required number of score evaluations. For every smoothness index $\beta>0$, a suitable grid yields averaged conditional Kullback--Leibler (KL) divergence $\widetilde O_p(n^{-2\beta/(d_x+d_y+2\beta)})$ relative to the original target with or without early stopping. Both the full and early-stopped samplers attain average target total-variation (TV) error $\widetilde O_p(n^{-\beta/(d_x+d_y+2\beta)})$. Here $n$ is the sample size, $d_x,d_y$ are the data and conditioning dimensions, and errors are averaged over
    
[^5]: 约束概率数值方法：线性不等式约束下线性偏微分方程的高斯过程求解器

    Constrained Probabilistic Numerics: Gaussian Process Solvers for Linear PDEs under Linear Inequality Constraints

    [https://arxiv.org/abs/2610.13804](https://arxiv.org/abs/2610.13804)

    该论文提出了一种支持线性不等式约束（如非负性、单调性）的高斯过程PDE求解方法，证明等式与不等式条件化可以顺序进行，并通过截断高斯矩实现了非高斯后验的可计算性。

    

    arXiv:2610.13804v1 公告类型： cross 摘要：线性偏微分方程（PDE）的高斯过程求解器返回的是后验分布而非点估计，但它们仅以等式信息为条件：PDE残差、边界数据和测量值。然而，许多感兴趣的解还满足不等式约束，包括非负性、单调性和无套利性，而这些约束无法通过等式条件来强制执行。我们证明，将高斯过程先验同时条件于线性等式观测和一个有限维不等式事件，等价于在解空间上以Borel测度的形式进行顺序条件化。因此，融合物理信息的后验可以被视为虚拟点处不等式更新的有效先验。组合后的分布虽然是非高斯的，但可以通过截断高斯矩进行计算，且PDE条件化和不等式更新均无需任何修改。我们在一个泊松基准问题、期权定价……上对该构造方法进行了评估。

    arXiv:2610.13804v1 Announce Type: cross  Abstract: Gaussian process solvers for linear partial differential equations (PDEs) return posteriors rather than point estimates, but they condition only on equality information: the PDE residual, boundary data, and measurements. Many solutions of interest also obey inequality constraints, including nonnegativity, monotonicity, and absence of arbitrage, which equality conditioning cannot enforce. We prove that conditioning a Gaussian process prior jointly on linear equality observations and a finite-dimensional inequality event is equivalent to conditioning sequentially in terms of Borel measures on the solution space. The physics-informed posterior is therefore a valid prior for the inequality update at virtual points. The combined law is non-Gaussian yet computable through truncated-Gaussian moments, and neither the PDE conditioning nor the inequality update requires modification. We evaluated the construction on a Poisson benchmark, option-p
    
[^6]: 从像素到隐私敏感特征：重构误差界的信息论框架

    From Pixels to Privacy-Sensitive Features: An Information-Theoretic Framework for Reconstruction Bounds

    [https://arxiv.org/abs/2610.13768](https://arxiv.org/abs/2610.13768)

    该论文提出了一个基于率失真理论的信息论框架，能在训练前计算出 DP-SGD 训练模型的重构误差下界，即使面对可观测训练过程、知晓其余训练数据和总体分布的最坏情况攻击者依然成立，并允许分析者选择面向隐私敏感特征的误差度量表示。

    

    arXiv:2610.13768v1 公告类型：cross 摘要：对于一个使用 DP-SGD 训练的模型，攻击者究竟能够恢复出什么？其 $(\varepsilon,\delta)$ 差分隐私保证并不能直接回答这个问题。我们推导出一个可以在训练之前计算的失真界，并且该界甚至适用于一类很强的攻击者：他们可以在训练过程中观察参数轨迹、知晓所有其他训练记录，以及训练记录所来自的总体分布。分析者可以自由选择度量误差所用的表示方式，使该误差界能够针对特定应用中真正涉及隐私敏感性的重构方面。我们的信息论框架将两项贡献分离开来：其一是通过条件互信息度量的、训练过程所泄露的关于某条记录的信息量；其二是通过信源熵度量的、数据分布中原本存在的不确定性。率失真理论将二者结合，得到对所有重构方法期望误差的一个下界。

    arXiv:2610.13768v1 Announce Type: cross  Abstract: What can an adversary recover from a model trained with DP-SGD? Its $(\varepsilon,\delta)$ guarantee does not answer this question directly. We derive a distortion bound that can be computed before training and applies even to an adversary who observes the parameter trajectory during training, knows every other training record and the population distribution from which the training records are drawn. The analyst may choose the representation in which error is measured, allowing the bound to target the aspects of reconstruction that are privacy-sensitive in a given application. Our information-theoretic framework separates two contributions: the information about a record revealed by training, measured through conditional mutual information, and the uncertainty already present in the data distribution, measured through source entropy. Rate-distortion theory combines them into a lower bound on the expected error of every reconstruction a
    
[^7]: 先验数据拟合网络能多好地摊销贝叶斯推断？

    How Well Do Prior-Data Fitted Networks Amortize Bayesian Inference?

    [https://arxiv.org/abs/2610.13764](https://arxiv.org/abs/2610.13764)

    本文将先验数据拟合网络（PFNs）的部署误差分解为先验、上下文和摊销误差，发现PFNs虽能很好地恢复目标后验预测分布，但在罕见设置下仍存在残余摊销误差，而贝叶斯模型平均方法可有效减少这种不确定性并提升预测性能。

    

    先验数据拟合网络（PFNs）被训练用于近似合成先验的贝叶斯后验预测分布（PPD），然而关于其近似质量的研究结果却出人意料地稀少。从经验风险最小化的视角出发，我们将PFNs的部署误差分解为先验误差、上下文误差和摊销误差，并在现实先验下相对于参考后验预测分布来度量摊销误差。结果表明，PFNs通常能很好地恢复目标后验预测分布。然而，我们发现部分残余摊销误差仍然存在，尤其是在先验预训练分布中的罕见设置下，即使扩大模型规模也无法消除，这表现为独立预训练的PFNs之间的分歧，并在推理时给预测器留下不确定性。作为补救措施，我们研究了多种类型的贝叶斯模型平均方法，发现这些方法能够持续减少这种不确定性并改善预测性能。

    arXiv:2610.13764v1 Announce Type: cross  Abstract: Prior-data fitted networks (PFNs) are trained to approximate the Bayesian posterior predictive distribution (PPD) of a synthetic prior, yet results about the approximation quality are surprisingly thin. Taking an empirical risk minimization perspective, we decompose the deployment error of PFNs into prior, context, and amortization error, and measure the latter against a reference PPD under a realistic prior. Results show that PFNs generally recover the targeted PPD very well. Nevertheless, we find that parts of the residual amortization error persist, especially in \textit{rare} settings under the prior pretraining distribution, even when scaling the model size, which manifests in disagreement between independently pretrained PFNs and leaves uncertainty about the predictor at inference time. As a remedy, we study multiple types of Bayesian model averages and find that these can consistently reduce this uncertainty and improve predicti
    
[^8]: 一步到位的校准：联合学习既准确又校准良好的神经网络

    Calibration in One Go: Jointly Learning Accurate and Calibrated Neural Networks

    [https://arxiv.org/abs/2610.13693](https://arxiv.org/abs/2610.13693)

    该论文提出通过双层 Stackelberg 优化将校准器与主干网络端到端联合训练，无需单独的校准数据集或专用校准损失，仅引入校准器学习率这一个超参数，即可同时学习到既准确又校准良好的神经网络。

    

    神经网络（NN）虽然具有高度的灵活性和通用性，但往往校准不佳。事后校准方法使用单独的校准数据集和额外的校准参数来调整已经训练好的模型的输出。这一过程增加了额外开销，需要对校准器进行单独优化以及后续的模型选择。在本文中，我们证明了标准的（事后）校准器可以通过双层 Stackelberg 公式与主干网络进行端到端联合训练。该公式通过使用交叉熵同时优化准确性和校准性能，消除了对单独校准集或专用校准损失函数的需求，仅引入了一个额外的超参数：校准器学习率。尽管我们的框架支持从温度缩放、矩阵缩放到神经网络层等各种校准器，但我们的结果表明，使用最终的分类层本身作为校准器即可得到既准确又校准良好的神经网络。

    arXiv:2610.13693v1 Announce Type: cross  Abstract: Neural networks (NNs), while highly flexible and versatile, are often poorly calibrated. Post-hoc calibration methods adjust the outputs of an already trained model using a separate calibration dataset and additional calibration parameters. This process adds overhead that requires separate optimization of the calibrator and subsequent model selection. In this paper, we show that standard (post-hoc) calibrators can be trained end-to-end with the backbone via a bilevel Stackelberg formulation. This formulation eliminates the need for a separate calibration set or a dedicated calibration loss by jointly optimizing accuracy and calibration using cross-entropy, introducing only one additional hyperparameter: the calibrator learning rate. Although our framework supports calibrators ranging from temperature and matrix scaling to NN layers, our results show that using the final classification layer itself as the calibrator yields accurate, wel
    
[^9]: 并非所有维度生而平等：基于分组二次核的离群值感知注意力线性化

    Not All Dimensions Are Equal: Outlier-Aware Attention Linearization with Grouped Quadratic Kernels

    [https://arxiv.org/abs/2610.13674](https://arxiv.org/abs/2610.13674)

    该论文发现预训练注意力中少数离群维度主导了线性化的近似误差，并提出维度感知的OAL方法，通过Hadamard乘积表述将全局系数放宽为维度特定的线性项与跨维度二次交互，再以分组二次核在更少参数下实现高保真的注意力线性化。

    

    线性化softmax注意力为高效的Transformer推理提供了一条途径，但如何在近似中保留预训练注意力仍然充满挑战。我们识别出一个被标准线性化方法所忽略的重要近似误差来源：一小部分离群维度对查询-键点积的贡献不成比例地大，并占据了近似误差的很大一部分。受这种异质性的启发，我们提出了离群值感知注意力线性化（Outlier-Aware Attention Linearization, OAL），这是一种用于构建二次注意力近似的维度感知方法。从二阶泰勒近似出发，我们引入了一种Hadamard乘积表述，使各个查询-键的独立贡献得以显式呈现。该表述放宽了全局共享系数的限制，以支持维度特定的线性项和跨维度的二次交互，同时保持可分离的计算方式。为了以更少的参数保留这种灵活性，OAL将维度进行分组（原文在此处截断）。

    arXiv:2610.13674v1 Announce Type: cross  Abstract: Linearizing softmax attention offers a route to efficient Transformer inference, but preserving pretrained attention remains challenging. We identify an important source of approximation error overlooked by standard linearization methods: a small number of outlier dimensions contribute disproportionately to query-key dot products and account for a large fraction of the approximation error. Motivated by this heterogeneity, we propose Outlier-Aware Attention Linearization (OAL), a dimension-aware method for constructing quadratic attention approximations. Starting from a second-order Taylor approximation, we introduce a Hadamard-product formulation that makes individual query-key contributions explicit. This formulation relaxes globally shared coefficients to support dimension-specific linear terms and cross-dimensional quadratic interactions while preserving separable evaluation. To retain this flexibility with fewer parameters, OAL gro
    
[^10]: 面向非高斯空间场的可扩展稀疏自回归变分推断

    Scalable Sparse Autoregressive Variational Inference for Non-Gaussian Spatial Fields

    [https://arxiv.org/abs/2610.13656](https://arxiv.org/abs/2610.13656)

    该论文提出了一种可扩展的变分推断框架，通过在变分族中镜像稀疏最近邻自回归先验结构、使用归一化流条件分布和共享神经条件器摊销参数，并以双随机优化实现对数多项式迭代复杂度，从而高效解决大规模非高斯空间场的后验推断难题。

    

    潜场模型是空间统计学中的重要工具，但即使采用高斯过程（GP）先验，只要似然函数是非高斯的，潜场的后验分布在解析上便是不可求解的。现有的近似方法对于大规模非高斯数据集而言，要么速度太慢，要么限制过多。我们提出了一种可扩展的变分推断框架，该框架在变分族中镜像了先验的稀疏最近邻自回归结构，使用归一化流条件分布以增强表达能力，并通过单个共享的神经条件器对所有变分参数进行摊销。结合缩减祖先集的双随机优化方法，实现了每次迭代的对数多项式复杂度。

    arXiv:2610.13656v1 Announce Type: cross  Abstract: Latent field models are essential for spatial statistics, but the posterior distribution of the latent field, even with a Gaussian process (GP) prior, is analytically intractable whenever the likelihood is non-Gaussian. Existing approximations are either too slow or too restrictive for large non-Gaussian datasets. We present a scalable variational inference framework that mirrors the sparse nearest-neighbor autoregressive structure of the prior in the variational family, uses normalizing-flow conditionals for expressiveness, and amortizes all variational parameters via a single shared neural conditioner. Doubly stochastic optimization with reduced ancestor sets achieves polylogarithmic complexity per iteration.
    
[^11]: 对测试时锐化大语言模型中幂采样的批判性审视

    A Critical Look at Power Sampling when Test Time Sharpening LLMs

    [https://arxiv.org/abs/2610.13653](https://arxiv.org/abs/2610.13653)

    该论文证明基于MCMC/SMC的幂采样方法在现实计算预算下无法收敛到目标幂分布，其收益可由简单的Best-of-N采样复现，表明其成功源于筛选高概率序列而非真正近似幂分布。

    

    带可验证奖励的强化学习（RLVR）已显著提升了大语言模型的推理能力，然而越来越多的证据表明，它主要进行的是分布锐化：将概率质量集中在基础模型本身已经能够产生的推理路径上。因此，近期研究试图在不需要奖励信号的情况下获得RLVR的收益，其方法是通过马尔可夫链蒙特卡洛（MCMC）或序列蒙特卡洛（SMC）从幂分布 $\pi_\alpha \propto \pi_{\text{base}}^\alpha$ 中进行采样。我们证明，在现实的计算预算下，这些采样器无法收敛到 $\pi_\alpha$，并且它们的收益可以通过简单的 Best-of-$N$ 采样（配合适当的评分）大体复现。这表明它们成功的原因在于选择了高概率序列，而非近似 $\pi_\alpha$，由此引出一个自然的问题：幂分布本身是否是一个值得追求的目标？我们通过微调基础模型使其匹配 $\pi_\alpha$ 来直接检验这一假设，从而绕开了采样器带来的复杂性。

    arXiv:2610.13653v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has substantially improved LLM reasoning, yet growing evidence suggests it mainly performs distribution sharpening: concentrating probability mass on reasoning paths the base model already produces. Recent work therefore seeks RLVR's gains without needing reward signals, by sampling from the power distribution $\pi_\alpha \propto \pi_{\text{base}}^\alpha$ via Markov chain Monte Carlo or sequential Monte Carlo. We show that these samplers fail to converge to $\pi_\alpha$ under realistic compute budgets, and that their gains are largely recreated via simple Best-of-$N$ with appropriate grading . This suggests they succeed by selecting high-probability sequences, not by approximating $\pi_\alpha$, raising a natural question: is the power distribution itself a worthwhile target? We test this hypothesis directly by fine-tuning the base model to match $\pi_\alpha$, sidestepping sampler co
    
[^12]: 基于核方法的实验研究外部有效性

    Kernel-based External Validity of Experimental Studies

    [https://arxiv.org/abs/2610.13625](https://arxiv.org/abs/2610.13625)

    该论文提出用核方法（最大均值差异MMD）形式化定义实验研究的外部有效性，并建立了研究规模与MMD分位数之间的渐近幂律关系，从而能够从小规模初步研究估计出具有外部有效性的研究所需规模。

    

    一项研究的外部有效性是指其研究结果能够泛化到未见过的实验条件（如数据集）的程度。我们将研究结果及其研究问题共同形式化，并将外部有效性定义为：该研究的两次独立且规模相同的重复实验所产生的结果之间的最大均值差异（MMD）低于给定容差的概率。这一方法以结果空间的选择以及适用于MMD的合适再生核为核心，足够灵活，可应用于多种不同类型的结果和研究问题。此外，我们建立了研究规模与MMD分位数之间的渐近幂律关系，并利用该关系从小规模的初步研究中估计出具有外部有效性的研究所需的规模。我们将该框架应用于带并列的排名问题，并讨论了适合不同研究问题的核函数。

    arXiv:2610.13625v1 Announce Type: cross  Abstract: The external validity of a study is the extent to which its findings generalize to unseen experimental conditions, such as datasets. We formalize the results of a study together with its research question, and define external validity as the probability that two independent and same-sized replications of the study produce results whose maximum mean discrepancy (MMD) is below a given tolerance. This approach, centered on the choice of a space of results and of an appropriate reproducing kernel for the MMD, is flexible enough to be deployed on multiple different types of results and research questions. Additionally, we establish an asymptotic power-law relation between the size of a study and the quantiles of the MMD, which we use to estimate the desired size of an externally valid study from a small-scale preliminary study. We deploy our framework to rankings with ties and discuss kernels suited to different research questions. Experime
    
[^13]: 关于生成扩散与最优传输之间联系的一些注记

    Some remarks on the connection between generative diffusion and optimal transport

    [https://arxiv.org/abs/2610.13614](https://arxiv.org/abs/2610.13614)

    本文揭示了生成扩散与最优传输之间的三个简单联系：数据加噪的联合分布在每个时刻都是熵正则化最优传输问题的解（因此反向扩散等价于逐渐降低熵正则化的退火过程），且扩散与流匹配中分离数据模式的动力学不稳定性，在最优传输中表现为生成初期一个被熵正则化平滑的指数级薄边界层。

    

    我们讨论了生成扩散与最优传输之间的三个简单联系。首先，数据与其 Ornstein-Uhlenbeck 加噪过程的联合分布在每一时刻都是某个熵正则化最优传输问题的解，其温度由噪声水平决定。因此，反向扩散可以被解读为一种逐渐降低熵正则化的退火过程。其次，我们研究了物种形成转变（speciation transition）——即数据分布的各个模式在生成过程中如何浮现——并比较了它在扩散模型、流匹配与最优传输中的发生方式。以高斯混合分布作为高维例子，我们表明扩散与流匹配借以分离各模式的动力学不稳定性，在最优传输情形下变为生成初始阶段一个指数级薄的边界层，而该边界层会被熵正则化所平滑。第三，考虑一个带有一般前向动力学的……（摘要原文在此处截断）

    arXiv:2610.13614v1 Announce Type: cross  Abstract: We discuss three simple connections between generative diffusion and optimal transport. First, the joint law of the data and of their Ornstein-Uhlenbeck noising is, at every time, the solution of an entropic optimal-transport problem, at a temperature set by the noise level. Backward diffusion can consequently be read as an annealing which gradually decreases the entropic regularization. Second, we study the speciation transition--how the modes of the data distribution emerge during generation--and compare how it happens in diffusion and flow matching with respect to optimal transport. Using Gaussian mixtures as high-dimensional example, we show that the dynamical instability through which diffusion and flow matching separate the modes becomes, for optimal transport, an exponentially thin boundary layer at the beginning of generation, which is rounded by entropic regularization. Third, considering a generic forward dynamics with a gene
    
[^14]: CorrFlow：相关椭圆面上的精确且置换等变的流匹配

    CorrFlow: Exact and Permutation-Equivariant Flow Matching on the Correlation Elliptope

    [https://arxiv.org/abs/2610.13606](https://arxiv.org/abs/2610.13606)

    CorrFlow通过广义Fisher变换将满秩相关矩阵双射映射到无约束坐标，再用置换等变的图流匹配网络学习其经验分布，从而无需投影即可精确保证输出位于相关椭圆面上，并对变量排序保持等变性。

    

    相关矩阵同时结合了非线性正定性约束与行列同时置换的对称性，因此其生成涉及两个截然不同的问题：第一，无论变量如何排序，每个输出都必须位于相关椭圆面上；第二，生成器必须学习经验数据在该有效空间内所占据的特定领域分布。现有的金融图像生成器将这两个问题混在一起：它们学习有序的矩阵元素，然后再事后修复无效的采样结果。我们提出了CorrFlow，将这两个问题分离开来。一个广义Fisher变换将满秩相关矩阵双射地映射到无约束坐标，从而无需投影即可保证输出的有效性。一个图流匹配网络在这些坐标中学习经验分布，并且具有置换等变性：对变量重新排序只会使输出相应地重新排序。我们比较了重新实现的条件CorrGAN……

    arXiv:2610.13606v1 Announce Type: cross  Abstract: Correlation matrices combine nonlinear positive-definiteness constraints with a simultaneous row-column permutation symmetry. Their generation therefore involves two distinct problems. First, every output must remain on the correlation elliptope, irrespective of how its variables are ordered. Second, the generator must learn the domain-specific distribution occupied by empirical data inside that valid space. Existing financial image generators mix these problems: they learn ordered matrix entries and repair invalid draws afterward. We introduce CorrFlow, which separates them. A generalized Fisher transform maps full-rank correlation matrices bijectively to unconstrained coordinates, guaranteeing valid outputs without projection. A graph flow-matching network learns the empirical distribution in those coordinates and is permutation equivariant: reordering the variables only reorders the output. We compare a conditional CorrGAN reimpleme
    
[^15]: 超越差异度量：非平稳加权风险最小化中的最优预测器漂移

    Beyond Discrepancy: Oracle Drift in Nonstationary Weighted Risk Minimization

    [https://arxiv.org/abs/2610.13598](https://arxiv.org/abs/2610.13598)

    该论文提出用模型类内最优预测器的移动（即“最优预测器漂移”）来直接刻画非平稳加权风险最小化中的漂移误差，证明传统差异型界过于保守，并在线性回归和非参数回归中得到了更精确的漂移界。

    

    我们通过将超额风险分解为估计、漂移和近似三个组成部分，来分析非平稳加权风险最小化中的分布漂移问题。漂移误差通常通过差异型量来界定，这类量控制预测器类别上风险的变化。与之不同，我们直接将漂移误差与模型类内最优预测器的移动联系起来，并分析其在参数化回归模型中的行为。我们的分析表明，差异型量可能过于保守：与我们的最优预测器漂移不同，它们可能以比相应的超额风险贡献更大的阶数增长，并且会对不影响最优预测器的风险面变化产生响应。对于线性回归，我们精确地刻画了这一差距，并将由此得到的漂移界与估计保证相结合。我们还进一步将分析扩展到非参数回归。

    arXiv:2610.13598v1 Announce Type: cross  Abstract: We analyze distributional drift in nonstationary weighted risk minimization by decomposing the excess risk into estimation, drift, and approximation components. The drift error is commonly bounded through discrepancy-type quantities that control changes of the risk over classes of predictors. We instead relate the drift error directly to the movement of the best predictor within the model class and analyze its behavior in parametrized regression models. Our analysis shows that discrepancy-type quantities can be overly conservative: unlike our oracle drift, they may scale at a larger order than the resulting excess-risk contribution and can respond to changes in the risk surface that do not affect the optimal predictor. For linear regression, we characterize this gap sharply and combine the resulting drift bounds with estimation guarantees. We further extend the analysis to nonparametric regression.
    
[^16]: 时变分层基础设施中电动汽车充电的概率预测

    Probabilistic Forecasting of Electric Vehicle Charging in a Time-varying Hierarchical Infrastructure

    [https://arxiv.org/abs/2610.13588](https://arxiv.org/abs/2610.13588)

    提出将分层预测与处理缺失数据的保形预测相结合，在持续演化的电动汽车充电基础设施分层结构中，为从单个充电点到国家级的每个聚合节点生成有效的预测区间。

    

    对电动汽车（EV）充电需求进行不确定性量化，对于将电动汽车普及给电力系统带来的日益增长的压力转化为灵活性来源至关重要。我们研究了苏格兰公共充电站的一个大规模数据集，其中需求呈现出分层结构，跨越多个聚合层级组织，从单个充电点到区域和国家级尺度。充电基础设施在不断演变，新站点不断部署，现有站点偶尔被移除，这就需要针对这种演变结构量身定制的预测和不确定性量化方法。为了应对这些挑战，我们将分层预测与处理缺失数据的保形预测最新进展相结合。所考虑的保形方法在层次结构的每个节点上都能产生有效的预测区间，其中CP-MNR-Nested算法成功地处理了时变的分层结构。

    arXiv:2610.13588v1 Announce Type: cross  Abstract: Uncertainty quantification for electric vehicle (EV) charging demand is essential to turn the increasing pressure of EV adoption on power systems into a source of flexibility. We study a large-scale dataset of public charging stations in Scotland, where demand exhibits a hierarchical structure, organized across multiple aggregation levels, from individual charging points to regional and national scales. The charging infrastructure is continuously evolving, with new stations deployed and existing ones occasionally removed, which calls for forecasting and uncertainty quantification methods tailored to this evolving structure. To address these challenges, we combine hierarchical forecasting with recent advances in conformal prediction with missing data. The conformal procedures considered produce valid prediction intervals at every node of the hierarchy and the algorithm CP-MNR-Nested successfully accounts for the time-varying hierarchica
    
[^17]: 防御蒸馏的黑魔法：阈下学习的无监督发现

    Defence against the dark arts of distillation: Unsupervised discovery of subliminal learning

    [https://arxiv.org/abs/2610.13536](https://arxiv.org/abs/2610.13536)

    提出了一种无需预先知晓目标特征的无监督防御方法，仅通过分析学生模型的参数更新即可发现、解释并抑制从共享基础参数的教师模型阈下继承的失对齐行为特征。

    

    当语言模型从共享其基础参数的教师模型蒸馏而来时，学生模型可能会继承教师模型的行为特征——包括失对齐——即使训练数据在语义上与该特征无关。现有的所有防御方法都需要预先知道该特征。我们移除了这一要求，并证明通过利用一个关键的不对称性——阈下传递的特征在教师模型生成的多样化数据集上保持一致表达，而任务特定能力和共享能力则各不相同——可以仅凭参数更新，在无监督条件下恢复该特征。由此得到的估计器会返回一个权重空间方向，放大该方向即可发现并解释该特征，或将其投影出去以抑制该特征。在阈下学习基准测试中，我们的方法在没有任何特征标签的情况下成功恢复了传递的动物偏好特征，并抑制了其表达。

    arXiv:2610.13536v1 Announce Type: cross  Abstract: When a language model is distilled from a teacher that shares its base parameters, the student can inherit behavioural traits of the teacher---including misalignment---even from data that is semantically unrelated to the trait (Cloud et al., 2026). All existing defences require knowing the trait. We remove this requirement and prove that a subliminally transferred trait can be recovered from the parameter updates alone, without supervision, by exploiting a single asymmetry: the trait is expressed consistently across diverse datasets generated by the teacher, whereas task-specific and shared capabilities vary. The resulting estimator returns a weight-space direction that can be amplified to discover and interpret the trait or projected out to suppress it. Applied to the subliminal-learning benchmarks of (Cloud et al., 2026), our method recovers the transmitted animal-preference traits without any trait labels and suppresses their expres
    
[^18]: 闭式解形式的高斯BROJA部分信息分解

    Gaussian BROJA Partial Information Decomposition in Closed Form

    [https://arxiv.org/abs/2610.13464](https://arxiv.org/abs/2610.13464)

    本文推导出了高斯BROJA部分信息分解优化问题的闭式解，使一般多变量高斯情形下的分解可以直接计算，并将高维计算的速度提升超过两个数量级。

    

    BROJA部分信息分解提供了一种具有操作性动机的方法，可将两个信源所携带的关于目标的信息分离为独有、冗余和协同三种成分。然而，对于连续变量和高维变量，计算BROJA分解需要求解一个困难的优化问题。先前的工作表明，当每个信源-目标对服从联合高斯分布时，该问题可以简化为有限维数值优化。在本工作中，我们推导出了该优化问题的闭式解，使得在一般的多变量高斯设置下能够直接计算BROJA分解。所得的闭式表达式使高维高斯BROJA PID的计算速度提升超过两个数量级。代码可在 https://github.com/brainsandmachines/Eigen_PID 获取。

    arXiv:2610.13464v1 Announce Type: cross  Abstract: The BROJA partial information decomposition provides an operationally motivated way to separate the information that two sources carry about a target into unique, redundant, and synergistic components. For continuous and high-dimensional variables, however, computing the BROJA decomposition requires solving a difficult optimization problem. Previous work showed that, when each source-target pair is jointly Gaussian, the problem can be reduced to finite-dimensional numerical optimization. Here, we derive a closed-form solution to this optimization problem, enabling direct computation of the BROJA decomposition in the general multivariate Gaussian setting. The resulting closed-form expression enables speedups exceeding two orders of magnitude for high-dimensional Gaussian BROJA PID. Code is available at https://github.com/brainsandmachines/Eigen_PID.
    
[^19]: 通过压缩学习：风险估计达到大数定律速率

    Learning by Compression: Risk Estimates Attain the Law-of-Large-Numbers Rate

    [https://arxiv.org/abs/2610.13442](https://arxiv.org/abs/2610.13442)

    本文证明基于优先压缩的学习算法的风险界中并非固有存在对数因子，其风险估计对所有复杂度取值一致地达到 $\boldsymbol{k}/N + O(1/\sqrt{N})$ 的大数定律速率，与经验估计速率相同。

    

    压缩方案为推导学习算法的数据相关风险界提供了一个通用框架。对于基于优先压缩的学习规则，现有结果以一个样本相关的复杂度 $\boldsymbol{k}$ 来上界风险，该复杂度定义为压缩的基数再加上重构假设不适用的训练样本数量。然而，这些结果给出的裕量为 $\boldsymbol{k}/N$ 之上 $O(\sqrt{\ln N/N})$ 的量级。本文证明该对数因子并非内在固有。我们证明，对于每个固定的置信水平 $1-\delta$，风险以至少 $1-\delta$ 的概率被上界为 $$ \frac{\boldsymbol{k}}{N} + O\left(\frac{1}{\sqrt{N}}\right), $$ 且该界对所有可能的 $\boldsymbol{k}$ 值一致成立。因此，基于优先压缩的学习算法的风险估计达到了与经验估计相同的大数定律速率。

    arXiv:2610.13442v1 Announce Type: cross  Abstract: Compression schemes provide a general framework for deriving data-dependent risk bounds for learning algorithms. For learning rules based on preferent compression, existing results upper-bound the risk in terms of a sample-dependent complexity $\boldsymbol{k}$, defined as the cardinality of the compression augmented by the training examples on which the reconstructed hypothesis is inappropriate. These results, however, yield a margin of order $O(\sqrt{\ln N/N})$ above $\boldsymbol{k}/N$. In this paper, we establish that the logarithmic factor is not intrinsic. We prove that, for every fixed confidence level $1-\delta$, the risk is bounded with probability at least $1-\delta$ by $$ \frac{\boldsymbol{k}}{N} + O\left(\frac{1}{\sqrt{N}}\right), $$ uniformly over all possible values of $\boldsymbol{k}$. Thus, risk estimates for learning algorithms based on preferent compression attain the same law-of-large-numbers rate as the empirical esti
    
[^20]: 一因导一果：时间序列因果发现的核心要点

    One Thing Leads to Another: The Essentials of Time Series Causal Discovery

    [https://arxiv.org/abs/2610.13430](https://arxiv.org/abs/2610.13430)

    本教程论文首次为统计学、动力系统和机器学习等不同研究社区的时间序列因果发现方法提供了统一概述，系统介绍了其关键原理、局限性、假设、评估基准方法以及在农业、地球科学、医疗保健和人工智能系统等下游任务中的重要性。

    

    时间序列因果发现——即检测时间序列之间因果关系的过程——起源于半个多世纪前的信号处理研究。此后，它一直受到积极而广泛的关注，并在统计学、动力系统和机器学习等领域演化出不同的形式。进入这个碎片化且不断发展的研究领域可能令人望而生畏。本教程为跨研究社区的时间序列因果发现提供了统一的概述。我们介绍了现代发现方法所采用的关键原理，并讨论了它们的局限性和假设。我们还讨论了如何对发现方法进行评估、基准测试和微调。我们进一步讨论了时间序列因果发现如何在农业与地球科学、医疗保健以及人工智能系统的下游任务中变得至关重要。我们相信本教程将作为一个有价值的……

    arXiv:2610.13430v1 Announce Type: cross  Abstract: Time series causal discovery --- the process of detecting causation between time series --- has its roots in signal processing research from over a half century ago. It has since received active and widespread interest, evolving into different forms across domains such as statistics, dynamical systems and machine learning. Entering this fragmented and constantly evolving research landscape can be daunting. This tutorial provides a unified overview of time series causal discovery across research communities. We introduce the key principles invoked by modern discovery methods, and we discuss their limitations and assumptions. We also discuss how discovery methods are evaluated, benchmarked and fine-tuned. We further discuss how causal discovery for time series is becoming crucial for downstream tasks in agriculture and Earth science, in healthcare, and in artificial intelligence systems. We believe that this tutorial will serve as a valu
    
[^21]: 无模型贝叶斯聚类与多模态后验摘要

    Model-free Bayesian Clustering and Multimodal Posterior Summarisation

    [https://arxiv.org/abs/2610.13397](https://arxiv.org/abs/2610.13397)

    本文提出了一种无模型、噪声感知的贝叶斯聚类框架，通过鞅后验避免MCMC采样，并利用Wasserstein型摘要方法将多模态后验高效压缩为少量加权代表，同时边缘化了密度水平的不确定性。

    

    我们开发了一个无模型、噪声感知的贝叶斯聚类框架，该框架从预测贝叶斯的视角构建聚类划分上的鞅后验。这一方法避免了显式的似然-先验模型设定、昂贵的MCMC采样以及寻求众数的变分近似，同时能够适应不规则的簇形状，并支持后验样本的并行生成。为了解释高维、噪声感知聚类上的后验分布，我们引入了一种Wasserstein类型的摘要方法。利用一种新颖的噪声感知损失，该方法将大量后验样本压缩为少量的加权代表，从而捕捉不同的簇结构与噪声分配情况。此外，我们对密度水平的不确定性进行边缘化处理，避免了对特定密度水平选择的依赖。我们建立了所提损失的理论性质，并为水平边缘化后验近似提供了理论保证。

    arXiv:2610.13397v1 Announce Type: cross  Abstract: We develop a model-free, noise-aware Bayesian clustering framework that constructs a martingale posterior over clusterings from a predictive Bayesian perspective. This avoids explicit likelihood-prior model specification, expensive MCMC sampling, and mode-seeking variational approximations, while accommodating irregular cluster shapes and enabling parallel generation of posterior draws. To interpret the posterior over high-dimensional, noise-aware clusterings, we introduce a Wasserstein-type summarization method. Using a novel noise-aware loss, this approach compresses the large collection of posterior draws into a small set of weighted representatives, capturing distinct cluster structures and noise assignments. Further, we marginalise the uncertainty in the density level, avoiding reliance on a specific choice. We establish theoretical properties of the proposed loss and guarantees for the level-marginal posterior approximation. Resu
    
[^22]: BLOKS：通过草图化块预条件子扩展核方法求解器

    BLOKS: Scaling Kernel Solvers with Sketched Block Preconditioning

    [https://arxiv.org/abs/2610.13382](https://arxiv.org/abs/2610.13382)

    BLOKS提出了一种将块Jacobi近似与草图化相结合的新颖预条件子，在加速核岭回归近似求解收敛速度的同时有效控制了内存成本，兼顾了效率与模型表达能力。

    

    核岭回归（KRR）是一种理论上严谨的非参数学习算法，也常被用于研究深度学习的性质。然而，精确的KRR公式会带来高昂的时间和内存成本，使其难以应用于大规模数据集，这推动了近似求解器的发展。这些算法依赖于：（i）迭代方法，以避免直接求解的计算负担；（ii）草图化或批处理技术，以减小问题规模；（iii）预条件技术，以加速收敛。尽管预条件对于快速收敛至关重要，但其带来的内存成本可以通过减少模型的自由参数数量来控制，不过这可能以牺牲模型的表达能力进而牺牲精度为代价。为应对这一挑战，我们提出了BLOKS，这是一种基于共轭梯度法的Nyström草图化KRR求解器，其核心是一种新颖的预条件子，该预条件子将块Jacobi近似与草图化技术相结合。

    arXiv:2610.13382v1 Announce Type: new  Abstract: Kernel Ridge Regression (KRR) is a theoretically sound algorithm for nonparametric learning, which is also often used to investigate properties of deep learning. However, the exact KRR formulation incurs time and memory costs preventing applicability to large datasets, motivating the development of approximate solvers. These algorithms rely on (i) iterative methods, to avoid the computational burden of direct solutions; (ii) sketching or batching, to reduce the problem size; and (iii) preconditioning, to accelerate convergence. While crucial for fast convergence, preconditioning incurs memory costs that can be managed by lowering the number of free parameters of the model, potentially at the expense of expressivity and hence accuracy. To address this challenge, we propose BLOKS, a Conjugate Gradient-based solver for Nystr\"om-sketched KRR built around a novel preconditioner that combines a block Jacobi approximation with sketching. The B
    
[^23]: 连续动态时间规整的快速梯度近似

    Fast Gradient-Based Approximation of Continuous Dynamic Time Warping

    [https://arxiv.org/abs/2610.13129](https://arxiv.org/abs/2610.13129)

    本文提出了 FastCDTW，一种高效的连续动态时间规整（CDTW）近似求解器，证明了其在温和条件下的可微性，并针对分段线性规整函数提供精确闭式解、针对平滑规整函数提供无偏蒙特卡洛估计两种计算策略，大幅降低了计算成本。

    

    动态时间规整（DTW）是度量时间序列之间差异的常用方法。它通过动态规划在时间戳之间寻找最优规整路径，但这一过程速度较慢且不可微分。因此，已有多种变体被提出。其中，Soft-DTW 提出了 DTW 的平滑形式，使其能够进行梯度计算；而连续动态时间规整（CDTW）则通过可参数化的规整函数引入了连续形式。然而，这两种方法的计算成本都很高，对于现代机器学习流水线中的大规模数据集而言并不实用。本文从多个方面重新审视了 CDTW。在证明了其在温和条件下的可微性之后，我们提出了 FastCDTW——一个高效的近似求解器，具有两种不同的计算策略：针对分段线性规整函数的精确闭式方案，以及针对平滑规整函数的无偏蒙特卡洛估计器……

    arXiv:2610.13129v1 Announce Type: cross  Abstract: Dynamic Time Warping (DTW) is a common choice for measuring the discrepancy between time series. It proceeds by finding an optimal warping between timestamps through dynamic programming, which can be slow and is not differentiable. Consequently, several variants have been proposed. Among them, Soft-DTW proposes a smooth formulation of DTW, amenable to gradient computation, while Continuous-DTW (CDTW) introduces a continuous formulation through parametrizable warping functions. Yet, computational costs remain high for both methods and impractical for large datasets in modern machine learning pipelines. This paper revisits CDTW in several ways. After establishing its differentiability under mild conditions, we introduce FastCDTW, which is an efficient approximate solver with two different computational strategies: an exact closed-form scheme for piecewise-linear warping functions, and an unbiased Monte Carlo estimator for smooth ones, su
    
[^24]: 为什么重尾权重谱能够预测模型质量

    Why Heavy-Tailed Weight Spectra Predict Model Quality

    [https://arxiv.org/abs/2610.13123](https://arxiv.org/abs/2610.13123)

    该论文通过概率框架推导出仅基于DNN最后一层权重的边际似然，首次从统计学角度解释了为什么训练权重矩阵中的重尾谱分布能够有效预测模型质量。

    

    对深度神经网络（DNN）预测成功的坚实理解仍然难以捉摸。例如，虽然存在许多试图预测DNN模型质量的数据相关指标，但这些指标在大型数据集上的计算成本可能很高，和/或在训练数据集未公开时可能根本无法计算。一个实用且有效的模型质量预测方法涉及分析DNN权重矩阵，因为先前的工作已经证明，训练后权重矩阵中的重尾谱分布能够对模型质量产生很强的预测能力。然而，这类指标缺乏严格的统计学论证。在这项工作中，我们旨在从统计学角度理解这些指标为何表现良好。为此，我们采用概率框架来推导DNN的边际似然，且仅使用最后一层权重进行评估。在大型卷积DNN和大型语言模型（LLM）上，我们发现这种边际似然（原文摘要在此处截断）

    arXiv:2610.13123v1 Announce Type: cross  Abstract: A solid understanding of the predictive success of deep neural networks (DNNs) remains elusive. For example, while many data-dependent metrics exist that seek to predict model quality of DNNs, these metrics may be expensive to compute for large datasets and/or may be impossible to compute when training datasets are not released. A practical and effective predictor of model quality involves analyzing DNN weight matrices, as prior work has established that heavy-tailed spectral distributions in trained weight matrices can be strongly predictive of model quality. However, such metrics lack a rigorous statistical justification. In this work, we aim to understand why these metrics perform well, from a statistical perspective. To do so, we employ a probabilistic framework to derive the marginal likelihood for a DNN, evaluated using only the final layer weights. Across large-scale convolutional DNNs and LLMs, we find that this marginal likeli
    
[^25]: 掩码文本重建中语料库采样分歧的预测

    Forecasting Corpus-Sampling Disagreement in Masked Text Reconstruction

    [https://arxiv.org/abs/2610.12987](https://arxiv.org/abs/2610.12987)

    该论文推导出一个一阶公式，可仅凭参考语料库预测两个独立语料库上训练的掩码文本重建预测器之间的期望分歧，并发现语料库扩大四倍仅使分歧降低5–40%，远低于1/n缩放律的预期，其根源在于罕见上下文的平滑效应。

    

    如果从同一总体中再次收集一个文本语料库，预测器的输出会发生多大变化？我们针对掩码文本重建任务研究了这一问题，使用了一种由邻近词对计数构建的平滑预测器，并在完整的50,257词元的GPT-2词表上进行评估。将每个源文档视为一个观测值，我们推导出了拟合于两个独立语料库的预测器之间期望分歧的一阶公式。该公式考虑了可变的文档长度、随机上下文计数以及跨词元偏移的相关性，并且可以在不访问任何目标语料库的情况下，仅凭一个单独的参考语料库进行计算。在TinyStories、WikiText-103和CNN/DailyMail数据集上，将每个语料库从512个块扩展到2,048个块（每块64个词元）仅使分歧降低了5–40%，远低于1/n缩放律所隐含的75%。该预测准确地追踪了这种缓慢的下降趋势，我们将其归因于对罕见上下文的平滑处理。

    arXiv:2610.12987v1 Announce Type: new  Abstract: If a second text corpus were collected from the same population, how much would a predictor's outputs change? We study this question for masked-text reconstruction using a smoothed predictor built from nearby token-pair counts and evaluated over the full 50,257-token GPT-2 vocabulary. Treating each source document as one observation, we derive a first-order formula for the expected disagreement between predictors fitted to two independent corpora. The formula accounts for variable document length, random context counts, and correlations across token offsets, and it can be evaluated from a separate reference corpus without access to either target corpus. On TinyStories, WikiText-103, and CNN/DailyMail, growing each corpus from 512 to 2,048 chunks of 64 tokens lowers disagreement by only 5--40\%, far short of the 75\% implied by $1/n$ scaling. The forecast tracks this slow decline, which we trace to smoothing of rarely seen contexts. Forec
    
[^26]: 面向事件驱动潜在跳跃模型的扫描并行变分推断

    Scan-Parallel Variational Inference for event-driven Latent Jump Models

    [https://arxiv.org/abs/2610.12986](https://arxiv.org/abs/2610.12986)

    提出潜在跳跃模型（LJMs），通过泊松事件驱动的可学习瞬时跳跃来显式建模不规则时间序列中未观测冲击的时间与幅度，并配套开发了扫描并行变分推断方法用于高效训练。

    

    连续时间潜在状态空间模型被广泛应用于不规则观测的时间序列，通常通过一个随机初始状态加平滑的常微分方程（ODE）动力学，或通过随机微分方程（SDE）累积的布朗扰动来表示不确定性。虽然这些构造对于平滑或扩散系统而言是自然的，但它们并不能直接编码一组数量有限的未观测冲击，而这些冲击的发生时间、幅度及其对记忆的影响本身都是不确定的。我们转而提出潜在跳跃模型（LJMs）：潜在状态在泊松驱动的各事件时刻之间遵循闭式确定性动力学，并在每个事件时刻发生一次学习得到的瞬时跳跃，其强度和标记分布依赖于状态。这定义了一个时间齐次的马尔可夫过程，其跳跃是内在的状态转移，而非数值积分步长。在训练方面，我们推导了一种可处理的事件增广（原文摘要至此截断）

    arXiv:2610.12986v1 Announce Type: cross  Abstract: Continuous-time latent state-space models are widely used for irregularly observed time series, typically representing uncertainty through a random initial state followed by smooth ordinary differential equation (ODE) dynamics, or Brownian perturbations accumulated by a stochastic differential equation (SDE). While these constructions are natural for smooth or diffusive systems, they do not directly encode a finite set of unobserved shocks whose timing, magnitude, and effect on memory are themselves uncertain. We instead introduce latent jump models (LJMs): the latent state follows closed-form deterministic dynamics between Poisson-driven event times and undergoes a learned instantaneous jump at each event, with state-dependent intensity and mark distributions. This defines a time-homogeneous Markov process whose jumps are intrinsic state transitions rather than numerical integration steps. For training, we derive a tractable event-aug
    
[^27]: 基于引导扩散的空间预测

    Spatial Prediction with Guided Diffusion

    [https://arxiv.org/abs/2610.12967](https://arxiv.org/abs/2610.12967)

    本文提出利用机器学习中的引导扩散模型，在似然函数已知且可微的条件下求解一大类空间预测问题，并可灵活应用于高斯与非高斯过程模型。

    

    空间预测是涉及空间或时空参考数据的分析中一项至关重要的任务。它通常通过分层模型来实现，该模型由两部分组成：描述感兴趣的潜在过程的过程模型，以及描述该过程如何被观测的数据模型。目标量是预测分布，即在给定观测数据条件下潜在空间过程的条件分布。遗憾的是，预测分布很少能以封闭形式获得，且往往难以从中采样。本文展示了机器学习中引导扩散建模的最新进展，如何解决当由数据模型导出的似然函数已知且可微（可能经过松弛处理）时的一大类空间预测问题。为展示该方法的多功能性，我们将其应用于高斯与非高斯过程模型以及多种……（原文摘要在此处截断）

    arXiv:2610.12967v1 Announce Type: cross  Abstract: Spatial prediction is a critically important task in data analyses that involve spatially or spatio-temporally-referenced data. It is commonly carried out using hierarchical models, which comprise a process model describing the latent process of interest, and a data model describing how the process is observed. The target quantity is the predictive distribution, which is the conditional distribution of the latent spatial process given the observed data. Unfortunately, the predictive distribution is rarely available in closed form and can be difficult to sample from. Here, we show how recent developments in guided diffusion modelling from machine learning can address a wide class of problems in spatial prediction when the likelihood function derived from the data model is known and differentiable, possibly after relaxation. To show the versatility of the approach, we apply it to both Gaussian and non-Gaussian process models and a variet
    
[^28]: 从不平衡到证据：严重样本不平衡下复杂数据的深度非参数检验

    From Imbalance to Evidence: Deep Nonparametric Testing for Complex Data under Severe Sample Imbalance

    [https://arxiv.org/abs/2610.12946](https://arxiv.org/abs/2610.12946)

    提出增强型不平衡感知深度分类器双样本检验，创新性地将严重样本不平衡从分类器训练的不利因素转化为可重复利用的检验证据来源，从而在高不平衡情形下实现对图等复杂数据更可靠的分布比较。

    

    在严重样本不平衡的情况下比较分布，需要有效利用数量占多数的丰富观测数据，同时对涉及较小样本的差异保持敏感性。这一挑战对于图等复杂的非欧几里得数据尤为突出，因为其结构特征需要灵活的检验方法。基于分类器的双样本检验提供了这种灵活性，但现有方法主要是针对平衡或规模相近的样本设计的。本文研究了高度不平衡情形下的C2ST，并表明严重的样本不平衡可能会显著降低检验性能。与传统的将样本不平衡仅视为分类器训练中不利因素的不平衡学习方法不同，我们将其转化为可重复检验证据的来源。基于这一见解，我们提出了一种增强型不平衡感知深度分类器双样本检验，这是一种分类……

    arXiv:2610.12946v1 Announce Type: cross  Abstract: Comparing distributions under severe sample imbalance requires effectively using abundant majority observations while retaining sensitivity to differences involving the smaller sample. This challenge is particularly relevant for complex, non-Euclidean data, such as graphs, whose structural characteristics call for flexible testing procedures. Classifier two-sample tests (C2STs) provide such flexibility, but existing procedures are primarily designed for balanced or comparably sized samples. In this paper, we study C2STs in a highly imbalanced regime, and show that severe sample imbalance may substantially degrade testing performance. Different from conventional imbalanced learning approaches that treat sample imbalance merely as a nuisance for classifier training, we turn it into a source of repeated testing evidence. Building on this insight, we propose an Enhanced Imbalance-Aware Deep Classifier Two-Sample Test (EIAD-C2ST), a classif
    
[^29]: 基于线性化最优传输的测度值数据稀疏字典学习：高效算法与样本复杂度

    Sparse Dictionary Learning for Measure-Valued Data via Linearized Optimal Transport: Efficient Algorithms and Sample Complexity

    [https://arxiv.org/abs/2610.12924](https://arxiv.org/abs/2610.12924)

    该论文提出一个基于线性化最优传输的稀疏字典学习框架，将概率测度嵌入希尔伯特空间并通过稀疏自编码器学习数据自适应字典，实现对高光谱图像、点云及大语言模型文档嵌入等测度值数据的高效可解释表示，同时给出以原子类度量复杂度刻画的泛化界和样本复杂度理论保证。

    

    我们通过使用源自最优传输几何的特征将概率测度嵌入到希尔伯特空间中，研究概率测度值数据的字典学习问题。我们通过稀疏自编码器参数化学习数据自适应的字典，将概率测度表示为快速、可扩展的稀疏有限维编码。所得框架将最优传输丰富的几何结构与稀疏表示学习的可解释性和高效性相结合。我们在多种数据模态上展示了该框架的实用性，包括高光谱图像、几何点云以及大语言模型潜在空间中的文档嵌入。我们还针对广泛的 $L^2$ 值测度嵌入类别，证明了稀疏字典学习目标的泛化界，该泛化界以原子类的度量复杂度来表示。对于最优传输和熵正则化传输映射字典，我们推导了……（原文摘要在此处截断）

    arXiv:2610.12924v1 Announce Type: cross  Abstract: We study dictionary learning for probability measure-valued data by embedding measures into Hilbert spaces using features derived from optimal transport geometry. We learn data-adapted dictionaries through sparse autoencoder parameterizations, producing fast, scalable representations of probability measures as sparse finite-dimensional codes. The resulting framework combines the rich, geometric structure of optimal transport with the interpretability and efficiency of sparse representation learning. We demonstrate its utility across diverse data modalities, including hyperspectral images, geometric point clouds, and document embeddings in large language model latent spaces. We also prove generalization bounds for sparse dictionary learning objectives on broad classes of $L^2$-valued measure embeddings, expressed in terms of the metric complexity of the atom class. For optimal transport and entropic transport map dictionaries, we derive
    
[^30]: 超越二阶信息：带噪声特征的付费逻辑回归

    Beyond Second-Order Information: Paid Logistic Regression with Noisy Features

    [https://arxiv.org/abs/2610.12919](https://arxiv.org/abs/2610.12919)

    该文提出带付费噪声特征的在线逻辑回归问题，证明与付费线性回归不同，仅凭协方差这类二阶信息已不足以刻画最优损失——相同协方差的噪声分布可产生不同损失、更低方差甚至导致更高损失，因此需要超越二阶信息的新结构刻画。

    

    我们研究具有付费随机特征的在线二分类预测问题，其中学习者可以花费资源来获得对潜在特征的更好观测。这一设定受医学检测等应用的启发：额外花费可以提供对患者特征更精确的测量，从而更好地预测二元结果。在每一轮中，一个潜在特征被独立同分布地抽取，响应通过逻辑（logistic）模型生成，学习者在观测到带噪声的特征版本之前选择支付多少费用。学习者的目标是对预测损失和特征获取成本进行联合控制。与以往关于带付费特征的在线线性回归的结果不同，仅有协方差信息已不再足够：两个具有相同协方差的噪声分布可能导致不同的最优逻辑损失，甚至更低的方差反而会带来更高的损失。在一种结构性排序下，更大的支付…（原文摘要在此处截断）

    arXiv:2610.12919v1 Announce Type: cross  Abstract: We consider online binary prediction with paid stochastic features, where the learner can spend resources to obtain better observations of the underlying features. This setting is motivated by applications such as medical testing, where additional cost can provide more precise measurements of patient's features for predicting a binary outcome. At each round, a latent feature is drawn i.i.d., the response is generated through a logistic model, and the learner chooses how much to pay before observing a noisy version of the feature. The goal of learner is to jointly control prediction loss and acquisition cost. In contrast to the former result on the online linear regression with paid features, covariance information alone is no longer sufficient: two noise distributions with the same covariance may induce different optimal logistic losses, and lower variance can even lead to higher loss. Under a structural ordering in which larger paymen
    
[^31]: 面向吸引与排斥、具有不同相互作用核的广义行列式点过程

    Generalized Determinantal Point Processes for Attraction and Repulsion with Distinct Interaction Kernels

    [https://arxiv.org/abs/2610.12894](https://arxiv.org/abs/2610.12894)

    提出行列式比点过程（DRPP），利用两个对称正定核行列式之比分别建模排斥与吸引，在保持特征可解释性的同时，借助辅助变量表示实现高效MCMC采样与最大似然学习。

    

    行列式点过程（DPPs）通过排斥作用对子集选择概率进行建模，但许多应用中还表现出由成对关系或共享潜在结构所产生的吸引作用。非对称DPPs（NDPPs）虽然能够同时捕捉这两种效应，但其核函数难以解释和设定。我们提出行列式比点过程（DRPP），它通过两个对称正定核的行列式之比来建模排斥与吸引。这一表述在保留标准DPPs基于特征的解释性的同时，允许对两种效应分别进行设定。此外，我们引入了一种辅助变量表示，可实现高效的MCMC采样，并给出包括归一化常数在内的关键统计量的可计算期望公式，从而共同支持最大似然学习。我们还从理论上刻画了适合DRPPs的相互作用结构：对于聚类（原文在此处截断）

    arXiv:2610.12894v1 Announce Type: cross  Abstract: Determinantal point processes (DPPs) model subset selection probability with repulsion, but many applications also exhibit attraction arising from pairwise relations or shared latent structures. Nonsymmetric DPPs (NDPPs) capture both effects, but their kernels are difficult to interpret and specify. We propose the determinant-ratio point process (DRPP), which models repulsion and attraction via a ratio of the determinants of two symmetric positive-definite kernels. This formulation preserves the feature-based interpretation of standard DPPs while allowing the two effects to be specified separately. Furthermore, we introduce an auxiliary-variable representation that enables efficient MCMC sampling and yields tractable expectation formulas for key statistics including the normalizing constant, which together facilitate maximum-likelihood learning. We theoretically characterize the interaction structures well suited to DRPPs: for clustere
    
[^32]: 自归一化Kullback-Leibler估计的通用界

    Universal Bounds for Self-Normalized Kullback-Leibler Estimation

    [https://arxiv.org/abs/2610.12851](https://arxiv.org/abs/2610.12851)

    本文证明了基于归一化重要性权重的KL散度估计器存在通用有限样本界——其期望超过目标-提议KL散度至多 $\log 2$，且无需重要性权重有界或高阶矩有限，并将该结果应用于生成模型的Soft Best-of-$N$测试时对齐与相对熵编码。

    

    本研究探讨了一种基于熵的Kullback-Leibler（KL）散度估计器，该估计器由归一化的重要性权重构造而成。我们证明了一个通用的有限样本界，表明对于任意样本量，该估计器的期望超过目标-提议分布间KL散度的幅度至多为 $\log 2$。随后，我们刻画了由重要性加权重采样所诱导的分布与提议分布之间的KL散度，证明其上界同时由基于熵的估计器的期望和目标-提议分布间的KL散度约束。重要的是，这些界既不要求重要性权重有界，也不要求高阶矩有限。我们将这些结果应用于通过Soft Best-of-$N$采样实现生成模型的测试时对齐（这是Verdun等人（2025）提出的一种基于重要性采样的扩展技术），以及应用于相对熵编码，得到了有限样本策略散度证书和无需矩条件的编码成本界。

    arXiv:2610.12851v1 Announce Type: cross  Abstract: This work studies an entropy-based estimator of Kullback-Leibler (KL) divergence constructed from normalized importance weights. We prove a universal finite-sample bound showing that its expectation can exceed the target-proposal KL divergence by at most $\log 2$ for any sample size. We then characterize the KL divergence between the distribution induced by importance-weighted resampling and the proposal, showing that it is upper bounded by both the expectation of the entropy-based estimator and the target-proposal KL divergence. Importantly, these bounds require neither bounded importance weights nor finite higher moments. We apply these results to test-time alignment of generative models via Soft Best-of-$N$ sampling, an importance-sampling-based scaling technique proposed by Verdun et al. (2025), and to relative entropy coding, obtaining finite-sample policy-divergence certificates and moment-free bounds on coding cost.
    
[^33]: 双向引导：生成式贝叶斯中汇总统计量的同步学习

    Two-way Guidance: Simultaneous Learning of Summary Statistics for Generative Bayes

    [https://arxiv.org/abs/2610.12850](https://arxiv.org/abs/2610.12850)

    提出“双向引导”机制，通过汇总统计量与后验分布的同步联合学习，使基于模拟的贝叶斯推断在观测规模不断增大时仍能有效恢复集中化的后验分布。

    

    贝叶斯基于模拟的推断（SBI）在观测规模增大时，可能难以恢复越来越集中的后验分布。我们考虑将充分汇总统计量与后验分布进行联合学习，以此作为在各种SBI方法中实现可扩展推断的通用原则。SBI中常见的做法是通过预先确定的汇总量进行降维，但这可能会部分丢弃后验信息。我们的方法则对后验学习器如何依赖于所学习的汇总量（例如亲和性）施加结构约束，并与后验同时学习该汇总量。我们识别出一种名为“双向引导”的重要机制：在汇总量受到结构限制的条件下，所学汇总量的后验充分性简化了后验学习，而合适的后验目标又引导所学汇总量趋向后验充分性。该机制解释了一个反直觉的发现，即结构限制……（摘要原文在此处截断）

    arXiv:2610.12850v1 Announce Type: cross  Abstract: Bayesian simulation-based inference (SBI) can struggle to recover increasingly concentrated posteriors as observation size grows. We consider learning sufficient summary statistics jointly with the posterior as a general principle for scalable inference in various SBI methods. A common measure taken in SBI is dimension reduction through predetermined summaries, which may partially discard posterior information. Our approach instead imposes structure on how the posterior learner depends on a learned summary (e.g., affinity), and learns the summary simultaneously with the posterior. We identify an important mechanism named two-way guidance: under a structural restriction on the summary, the posterior sufficiency of the learned summary simplifies posterior learning, whereas a suitable posterior objective guides the learned summary towards posterior sufficiency. This mechanism explains a counterintuitive finding that the structural restric
    
[^34]: 具有核结构交互的可扩展行列式点过程学习

    Scalable Learning of Determinantal Point Processes with Kernel-Structured Interactions

    [https://arxiv.org/abs/2610.12829](https://arxiv.org/abs/2610.12829)

    该论文提出一种利用非线性核随机特征进行随机优化的可扩展行列式点过程学习方法，通过全秩交互建模与替代函数设计，突破了结构约束的局限，显著改善了表达能力与计算可扩展性之间的权衡，并以极高的参数效率获得了更好的泛化性能。

    

    有限集合上的行列式点过程（DPP）是一种展现排斥性的概率模型，其中项目间的相似性由一个正定核矩阵编码。将DPP学习扩展到大规模数据集通常需要对核矩阵施加结构约束，例如低秩加对角形式。然而，这种结构性假设在表达能力与计算可扩展性之间引入了严重的权衡。我们的研究通过以下方式应对这一挑战：（a）通过非线性核函数和潜在特征对全秩交互进行建模；（b）为最大似然估计器设计替代函数；（c）采用基于非线性核随机特征的随机优化方法。在真实世界数据集上的数值实验表明，我们的方法显著改善了这一权衡，以卓越的参数效率实现了更好的泛化性能。

    arXiv:2610.12829v1 Announce Type: cross  Abstract: Determinantal point processes (DPPs) on a finite set are probability models that exhibit repulsion where inter-item similarities are encoded by a positive definite kernel matrix. Scaling DPP learning to large datasets typically requires imposing structural constraints on the kernel matrix, such as a low-rank-plus-diagonal form. However, structural assumptions introduce a severe trade-off between expressive power and computational scalability. Our study addresses this challenge by (a) modeling the full-rank interactions via a nonlinear kernel function and latent features, (b) designing a surrogate function for the maximum likelihood estimator, and (c) adopting stochastic optimization using random features of the nonlinear kernel. Numerical experiments on real-world datasets demonstrate that our method significantly improves this trade-off, achieving better generalization performance with remarkable parameter efficiency.
    
[^35]: 幂次切片Wasserstein泛函估计的极小极大速率

    Minimax Rates for Estimating Powered Sliced Wasserstein Functionals

    [https://arxiv.org/abs/2610.12740](https://arxiv.org/abs/2610.12740)

    本文建立了幂次切片Wasserstein泛函两样本估计的极小极大速率——当 $r<2p$ 时为 $N^{-(r-p)/r}$ 阶，当 $r>2p$ 时为 $N^{-1/2}$ 阶——并证明自然的代入估计量在两种情形下均达到速率最优。

    

    我们研究了在投影 $r$ 阶矩假设下幂次切片Wasserstein泛函的两样本估计问题。令 $N=n\wedge m$，其中 $n$ 和 $m$ 为两个样本量。我们建立了非临界情形下的极小极大速率：当 $r<2p$ 时，极小极大风险为 $N^{-(r-p)/r}$ 阶；而当 $r>2p$ 时，为 $N^{-1/2}$ 阶。在这两种情形下，自然的代入估计量均达到极小极大速率最优。在端点 $r=p$ 处，极小极大风险具有不趋于零的下界，因此一致相合估计是不可能实现的。在临界阶 $r=2p$ 处，我们得到了代入估计量的精确最坏情形速率以及极小极大风险的下界，但上下界之间存在一个对数差距，该临界情形下的精确极小极大速率仍是悬而未决的问题。

    arXiv:2610.12740v1 Announce Type: cross  Abstract: We study the two-sample estimation of the powered Sliced Wasserstein functional under projected $r$-th moment assumptions. Let $N=n\wedge m$, where $n$ and $m$ are two sample sizes. We establish the minimax rates in the noncritical regimes. When $p<2p$, the minimax risk is of order $N^{-(r-p)/r}$, while for $r>2p$ it is of order $N^{-1/2}$. In both cases, the natural plug-in estimator is minimax rate-optimal. At the endpoint $r=p$, the minimax risk is bounded away from zero, so uniformly consistent estimation is impossible. At the critical order $r=2p$, we obtain the exact worst-case rate of the plug-in estimator and a lower bound for the minimax risk, leaving a logarithmic gap between the lower and upper bounds. The exact minimax rate in this critical case remains open.
    
[^36]: 面向摊销式条件可解释性的核化参数敏感性图谱

    Kernelized Parameter Sensitivity Atlases for Amortized Conditional Interpretability

    [https://arxiv.org/abs/2610.12737](https://arxiv.org/abs/2610.12737)

    该论文提出 KPSA，通过经验神经正切核的分块分解构建参数组敏感性的核化图谱，用轻量级核函数摊销式地为任意新输入估计敏感性，从而以远低于逐样本反向传播的成本实现输入条件化的模型可解释性。

    

    基于梯度的归因方法能够提供关于模型参数的输入特定信息，但通过反向传播为大量样本计算归因代价高昂，而静态重要性得分则丢弃了对输入的依赖性。我们从经验神经正切核的分块分解出发推导出 KPSA（核化参数敏感性图谱）。它度量归一化的参数组敏感性，将这些敏感性来源嵌入一个显式的查询空间，并使用轻量级的核函数为未见过的输入估计敏感性。我们推导了区分有限来源误差与不可约表示误差的理论界，并给出了基于间隔（margin）的 top-k 恢复保证。本文研究了两个可解释性应用。其一，KPSA 在视觉、时序预测和蛋白质建模等多种设置中识别出对扰动高度敏感的参数组。在参数预算跨越五倍范围（0.02%–0.1%）的情况下，其选定的参数组可达到扰动效果的 39%–85%（原文在此处截断）。

    arXiv:2610.12737v1 Announce Type: new  Abstract: Gradient-based attribution provides input-specific information about model parameters, but computing it via backpropagation for many samples is expensive, while static importance scores discard input dependence. We derive KPSA (Kernelized Parameter Sensitivity Atlas) from a block decomposition of the empirical neural tangent kernel. It measures normalized parameter-group sensitivity, embeds those sources in an explicit query space, and uses a lightweight kernel to estimate sensitivities for unseen inputs. We derive bounds that separate finite-source from irreducible representation error and give margin-based top-$k$ recovery. This paper studies two interpretability applications. Firstly, KPSA identifies parameter groups with high susceptibility to perturbations across vision, forecasting, and protein-modeling setups. Across a fivefold range of parameter budgets ($0.02$-$0.1\%$), its selected groups attain $39$-$85\%$ of the perturbation 
    
[^37]: 大语言模型行为刻画与风险管理的分布框架

    A Distributional Framework for Behavioral Characterization and Risk Management of Large Language Models

    [https://arxiv.org/abs/2610.12718](https://arxiv.org/abs/2610.12718)

    该论文提出一个基于蒙特卡洛模拟构建经验响应分布的框架，通过语义干预量化分布变化，将大语言模型的随机行为刻画与模型风险管理、治理以及黑盒模型监控联系起来。

    

    大语言模型本质上是随机的，在相同实验条件下重复执行可能产生不同的响应。我们提出了一个可应用于模型全生命周期的框架，从任务设计到残差行为分析和持续监控。该方法通过蒙特卡洛模拟估计来构建经验响应分布，并引入语义干预来研究这些分布如何变化。我们用两个不同的任务，借款人分类和音乐专辑推荐，来演示该框架，并量化由此产生的分布变化。该框架通过测量鲁棒性和控制有效性、刻画残差行为，与模型风险管理和治理相联系。由于它仅依赖于观测到的响应，因此也可应用于通过商业API访问的黑盒模型。

    arXiv:2610.12718v1 Announce Type: cross  Abstract: Large language models are stochastic by nature. Repeated executions under the same experimental conditions may produce different responses. We propose a framework that can be applied across the model lifecycle, from task design to residual behavior and continuous monitoring. The method consists of building an empirical response distribution estimated through Monte Carlo simulation. Semantic interventions are introduced to study how these distributions change. We illustrate the framework with two different tasks, borrower classification and music album recommendation, and quantify the resulting distributional changes. The framework is linked to model risk management and governance by measuring robustness and control effectiveness, and characterizing residual behavior. Since it relies only on observed responses, it can also be applied to black-box models accessed through commercial APIs.
    
[^38]: 通过单步隐式采样实现摊销式推断

    Amortised Inference through One-Step Implicit Sampling

    [https://arxiv.org/abs/2610.12679](https://arxiv.org/abs/2610.12679)

    提出单步隐式采样器（OSIS），利用短时MCMC动力学定义局部目标使建模分布成为MCMC核的不动点，从而实现高维贝叶斯后验分布的单步摊销采样。

    

    尽管在理论和算法方面取得了进展，从高维贝叶斯后验分布中进行高效采样仍然是一个难题。通常需要多个步骤才能抽取无偏样本，例如MCMC算法的迭代，或基于扩散的摊销采样器中的积分步骤。然而，可扩展的单步摊销推断仍然是一个具有挑战性的开放问题。在本工作中，我们建立在数据驱动生成建模的最新进展之上，为单步摊销采样提供了一种可能的解决方案。我们的单步隐式采样器（OSIS）利用短时MCMC动力学为隐式建模分布的演化定义局部目标。在渐近极限下，所建模的分布成为MCMC核的不动点，因此其样本即来自目标分布。我们分析了OSIS的收敛性质，并展示了它与其他生成建模和采样算法之间的联系。我们研究了各种……

    arXiv:2610.12679v1 Announce Type: new  Abstract: Efficient sampling from Bayesian posterior distributions in high dimensions remains a hard problem, despite advances in theory and algorithms. Multiple steps are usually required to draw unbiased samples, for example, iterations of an MCMC algorithm or integration steps in a diffusion-based amortised sampler. However, scalable one-step amortised inference remains a challenging open problem. In this work, we build upon recent progress in data-driven generative modelling to offer a possible solution for one-step amortised sampling. Our one-step implicit sampler (OSIS) uses short-time MCMC dynamics to define a local target for the evolution of an implicitly modelled distribution. In the asymptotic limit, the modelled distribution is a fixed point of the MCMC kernel and thus samples from the target. We analyse the convergence properties of OSIS and show connections with other generative modelling and sampling algorithms. We study various des
    
[^39]: 基于Transformer的掩码扩散模型可证明地学习具有潜在结构的分布

    Transformer-Based Masked Diffusion Models Provably Learn Distributions with Latent Structure

    [https://arxiv.org/abs/2610.12544](https://arxiv.org/abs/2610.12544)

    本文首次为基于Transformer的掩码扩散模型建立了端到端统计保证，证明其能以由潜在结构（隐马尔可夫树混合模型）决定的样本复杂度学习结构化离散分布，突破了以往理论中 $V^L$ 维度灾难的限制。

    

    掩码扩散模型（MDMs）通过使用学习到的掩码预测器迭代地并行解除多个词元的掩码来生成离散数据。其输出包含两类误差：由并行采样导致的并行化误差，以及由有限样本学习掩码预测器所产生的估计误差。现有理论将这两类误差分开处理，且已有的统计界通常随词表大小为 $V$、长度为 $L$ 的序列总数 $V^L$ 而扩展，饱受维度灾难之苦。在本工作中，我们建立了端到端的统计保证，表明基于Transformer的掩码扩散模型能够学习结构化分布，其样本复杂度由数据的潜在结构决定。具体而言，我们通过隐马尔可夫树的混合模型来刻画这种结构，该模型能够捕捉组份内部的潜在依赖关系以及组份之间的异质性。当数据分布能够被此类混合模型很好地近似时，基于Transformer的……

    arXiv:2610.12544v1 Announce Type: new  Abstract: Masked diffusion models (MDMs) generate discrete data by iteratively unmasking multiple tokens in parallel using a learned mask predictor. Their output carries two errors: parallelization error due to parallel sampling, and estimation error from the mask predictor learned with finite samples. Existing theory treats these errors separately, and the available statistical bounds typically scale with the number $V^L$ of length-$L$ sequences over a vocabulary of size $V$, suffering from the curse of dimensionality. In this work, we establish end-to-end statistical guarantees showing that transformer-based MDMs learn structured distributions with a sample complexity governed by their latent structure. Specifically, we model this structure through mixtures of hidden Markov trees, which capture latent dependencies within components and heterogeneity across components. When the data distribution is well approximated by such a mixture, a transform
    
[^40]: 无锚点的忠实性：关于可解释AI特征重要性忠实性度量结构性错位的立场论文

    Faithfulness Without Anchors: A Position Paper on the Structural Misalignments of Explainable AI Faithfulness Metrics for Feature Importance

    [https://arxiv.org/abs/2610.12493](https://arxiv.org/abs/2610.12493)

    本文是一篇立场论文，指出当前面向特征重要性的可解释AI忠实性度量存在结构性错位，并强调XAI实际上更多扮演调试工具的角色而非建立信任的手段。

    

    可解释人工智能（XAI）工具通常被呈现为一种通过使机器学习模型算法的某些方面更易于人类理解来提升对模型信任的方法。然而，过去十年积累的证据表明，这一目标总体上未能实现，反而促进了对错误预测的过度依赖，进而可能导致算法厌恶（algorithmic aversion）。实际上，XAI似乎主要作为一种调试工具发挥作用，有助于获得关于模型所学到的特征等方面的经验性洞察。造成这一局限性的主要问题之一在于，XAI方法常常过度简化模型的内部工作机制，导致解释与模型本身之间出现错位——这种行为被称为不忠实性（unfaithfulness）或不忠（infidelity）。忠实性可以被理解为解释与模型之间的近似程度。以严格的方式计算忠实性……

    arXiv:2610.12493v1 Announce Type: new  Abstract: Explainable Artificial Intelligence (XAI) tools have often been presented as methods to increase trust in machine learning models by making aspects of their algorithms more human-understandable. However, evidence accumulated over the last decade points to an overall failure of this objective, instead promoting overreliance on incorrect predictions, which can subsequently lead to algorithmic aversion. Rather, XAI appears to function primarily as a debugging tool, useful for obtaining anecdotal insights into, for example, features learned by the model. One of the main issues underlying this limitation is that XAI methods often oversimplify the inner workings of the model, resulting in misalignment between the explanation and the model itself - a behavior referred to as unfaithfulness or infidelity. Faithfulness can be interpreted as the degree of approximation between an explanation and the model. Computing faithfulness in a rigorous manne
    
[^41]: 关于AI时间视野估计与有效性的研究——从统计学视角审视METR图表

    On the estimation and validity of AI time horizons---a statistical look at the METR plot

    [https://arxiv.org/abs/2610.12466](https://arxiv.org/abs/2610.12466)

    本文用样条函数和项目反应理论重新估计了METR的AI时间视野，发现人类时间与AI难度的关系并非全局线性（在2至30分钟区间近乎平坦），因此同样10倍的提升，从3分钟到30分钟远比从30分钟到5小时容易实现。

    

    METR的50%时间视野衡量的是AI以50%概率成功解决的软件任务所需的人类完成时间，从而能以可解释的单位表达AI能力。本文基于228个任务和26个AI，使用样条函数和项目反应理论重新计算时间视野，放宽了“任务对AI的难度与人类时间的对数呈线性关系”这一假设。我们拟合出的样条可被解释为将人类时间转换为AI难度的函数；它在2至30分钟的区间内近乎平坦，但在其他区域接近线性。因此，尽管倍数同为10倍，时间视野从3分钟跃升至30分钟远比从30分钟跃升至5小时容易。总体而言，我们贡献了在一组经交叉验证的适当评分规则下表现更优的时间视野点估计，以及用于评估时间视野构建效度的诊断图。我们建议时间视野应结合……（原文此处截断）

    arXiv:2610.12466v1 Announce Type: new  Abstract: METR's 50\% time horizon measures the human completion time of software tasks that an AI solves with 50\% probability, allowing AI capabilities to be expressed in interpretable units. On 228 tasks and 26 AIs, we recompute the time horizons using splines and item-response theory to relax the assumption that the AI difficulty of a task depends linearly on the log of human time. Our fitted spline can be interpreted as a function that \emph{converts} human time to AI difficulty; it is nearly flat in a region from 2--30 min but close to linear elsewhere. Hence, a time-horizon jump from 3 min to 30 min is much easier than one from 30 min to 5 hours despite the same multiplier of $10 \times$. Overall, we contribute time-horizon point estimates that perform better under a cross-validated suite of proper scoring rules, as well as diagnostic plots for assessing time horizons' construct validity. We suggest that time horizons be interpreted togethe
    
[^42]: 基于距离草图的高效二次熵计算

    Efficient quadratic entropy with distance sketches

    [https://arxiv.org/abs/2610.11976](https://arxiv.org/abs/2610.11976)

    本文提出了一种基于随机特征嵌入、投影和控制变量技术的可扩展二次熵近似方法，并在文献计量学应用中仅凭引用和文本特征揭示了论文、领域和机构的跨学科影响力。

    

    我们详细介绍了用于近似任意分布 $p$ 和负型常见距离 $d$ 下二次熵 $p^T d p$ 的可扩展方法。我们聚焦于欧几里得距离和球面测地距离两种情形，二者均在简单的框架内利用随机特征嵌入和投影来显著改善计算复杂度。通过摊销单次大型矩阵乘法以及控制变量技术，在 $d$ 保持不变而 $p$ 变化的场景下，该方法进一步实现了低内存占用和短运行时间的大规模计算。我们通过与直接配对采样方法的对比，以及在 Open Graph Benchmark 数据集上的文献计量学/科学计量学示例验证了该方法，仅凭引用和文本特征即可揭示跨学科影响力特别窄或特别宽的论文、领域和机构。

    arXiv:2610.11976v1 Announce Type: cross  Abstract: We detail scalable methods for approximating the quadratic entropy $p^T d p$ for arbitrary distributions $p$ and common distances $d$ of negative type. We focus on the Euclidean and spherical geodesic cases, which both use random feature embeddings and projections to dramatically improve computational complexity within a simple framework. Amortization of a single large matrix multiplication and control variates further enable computation at large scale with low memory and runtime in situations where $d$ is held constant while $p$ varies. We demonstrate this with a comparison against direct pair sampling and bibliometric/scientometric examples on Open Graph Benchmark datasets, revealing papers, fields, and institutions with both particularly narrow and broad interdisciplinary reach from their citations and text features alone.
    
[^43]: 趁它们沉睡时工作：利用评估延迟实现全贝叶斯优化

    Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization

    [https://arxiv.org/abs/2610.08969](https://arxiv.org/abs/2610.08969)

    该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。

    

    黑盒优化问题在科学与工程领域无处不在，通常涉及代价高昂的目标函数。这种目标评估延迟在优化过程中带来两个后果：（i）目标函数评估主导了整个执行时间；（ii）样本高效的算法对于加速开发、避免资源浪费至关重要。贝叶斯优化（BO）方法是规划器为推荐下一个尝试点的事实上的标准选择。标准的BO采用点估计来拟合代理模型的超参数；而完全贝叶斯方法则通过模型平均来考虑超参数的不确定性，从而获得更好的不确定性估计——这在BO中普遍存在的低数据场景下非常有用。然而，该方法往往计算代价过高，因此很少被使用。在这项工作中，我们提出ELF-BO，一种利用目标函数评估延迟来提前启动全贝叶斯代理模型计算的算法。

    arXiv:2610.08969v1 Announce Type: new  Abstract: Black-box optimization problems are ubiquitous across science and engineering, often dealing with expensive objective functions. This objective latency has two consequences during optimization: (i) the objective evaluation dominates execution time, and (ii) sample-efficient algorithms are crucial to accelerate development and avoid wasting resources. Bayesian optimization (BO) methods are the \textit{de facto} choice of planners for suggesting the next point to try. Standard BO fits the surrogate model's hyperparameters with a point estimate. Alternatively, a fully Bayesian approach uses model averaging to account for uncertainty over the hyperparameters, leading to better uncertainty estimates---useful in the low-data regime that is pervasive in BO. However, it is often prohibitively expensive and thus rarely used. In this work, we propose ELF-BO, an algorithm that uses the objective evaluation latency to headstart the computation of th
    
[^44]: 超越 $T^{2/3}$ 的序贯校准问题的显式渐近界

    Explicit Asymptotic Bounds for Sequential Calibration Beyond $T^{2/3}$

    [https://arxiv.org/abs/2610.07623](https://arxiv.org/abs/2610.07623)

    该论文提出新的两阶段递归标记策略并改进归约方法，首次为序贯校准问题建立了超越 $T^{2/3}$ 的显式渐近界 $O(T^{0.662942288})$。

    

    当预测概率与经验结果频率相匹配时，概率预测被称为校准的：在被赋予概率 $p$ 的事件中，我们希望正结果的占比接近 $p$。我们研究二元结果的序贯预测问题。Foster 和 Vohra 建立的关于期望累积 $\ell_1$ 校准误差的经典 $O(T^{2/3})$ 界保持了二十多年，直到 Dagan 等人将指数 $2/3$ 降低了一个未具体指明的常数。我们为“符号保持-复用”博弈建立了一种新的两阶段递归标记策略，对于所有空间和时间的选择都能得到 $O(n^{\alpha}t^\beta)$ 的界。随后，我们通过修改 Dagan 等人的等价性，使其仅使用 $O(\log T)$ 个“符号保持-复用”博弈实例，从而锐化了从符号保持上界到校准的归约。这使我们能够建立一个显式的界 $O(T^{0.662942288})$。

    arXiv:2610.07623v1 Announce Type: cross  Abstract: Probability forecasts are calibrated when predicted probabilities match empirical outcome frequencies: among events assigned a probability $p$, we'd hope that the fraction of positive outcomes is close to $p$. We study the problem of sequential forecasting of binary outcomes. The classical $O(T^{2/3})$ bound on expected cumulative $\ell_1$-calibration error established by Foster and Vohra stood for over two decades until Dagan et al. reduced the exponent $2/3$ by an unspecified constant.   We establish a new two-phase recursive labeling strategy for the sign-preservation-with-reuse game that yields the bound $O(n^{\alpha}t^\beta)$ for all choices of space and time. We then sharpen the reduction from upper bounds on sign preservation to calibration by modifying the equivalence of Dagan et al. to use only $O(\log T)$ instances of the sign-preservation-with-reuse game. This lets us establish an explicit bound of $O(T^{0.662942288})$, the 
    
[^45]: 马尔可夫数据下异步TD（时序差分）学习的精确统计速率

    Sharp Statistical Rates for Asynchronous TD Learning with Markovian Data

    [https://arxiv.org/abs/2609.38880](https://arxiv.org/abs/2609.38880)

    该论文为基于单条马尔可夫轨迹的表格型TD学习建立了精确的末次迭代样本复杂度界 $\widetilde O\left(\frac{H^3}{\mu_{\min}\varepsilon^2}+\frac{t_{\operatorname{mix}}}{\mu_{\min}}\right)$，且该界对常数步长与递减步长调度均成立。

    

    我们研究了基于有限马尔可夫奖励过程单条轨迹的标准表格型时序差分（TD）学习的末次迭代（last iterate）。对于折扣因子 $\gamma$，记 $H=(1-\gamma)^{-1}$，并令 $\mu_{\min}$ 和 $t_{\operatorname{mix}}$ 分别表示最小平稳概率和全变差混合时间。我们证明，对于 $0<\varepsilon\leq1$，末次迭代TD以高概率达到至多 $\varepsilon$ 的sup范数误差，所需转移步数为 $\widetilde O\left(\frac{H^3}{\mu_{\min}\varepsilon^2}+\frac{t_{\operatorname{mix}}}{\mu_{\min}}\right)$。该速率既适用于为目标精度所选的常数步长，也适用于与目标精度和终止时间无关的递减步长调度。后者在超过某个显式瞬态阈值之后的所有时刻上提供了一致的同时保证。统计项保留了同步TD中三次方（立方）的有效视界依赖关系，而附加的混合过渡项则是不可省略的。

    arXiv:2609.38880v1 Announce Type: new  Abstract: We study the last iterate of standard tabular temporal-difference (TD) learning from a single trajectory of a finite Markov reward process. For discount factor $\gamma$, write $H=(1-\gamma)^{-1}$, and let $\mu_{\min}$ and $t_{\operatorname{mix}}$ denote the minimum stationary probability and total-variation mixing time. We prove that last-iterate TD achieves sup-norm error at most $\varepsilon$ with high probability using$\widetilde O\left(   \frac{H^3}{\mu_{\min}\varepsilon^2}   +\frac{t_{\operatorname{mix}}}{\mu_{\min}} \right)$ transitions, for $0<\varepsilon\leq1$. This rate holds both for a constant step size selected for the target accuracy and for a decreasing schedule independent of the target accuracy and terminal time. The latter gives a simultaneous guarantee over all times beyond an explicit transient threshold. The statistical term retains the cubic effective-horizon dependence of synchronous TD, and the additive mixing tran
    
[^46]: 鲁棒LassoNet：通过鲁棒损失函数增强神经网络中的特征选择

    Robust LassoNet: Enhancing Feature Selection in Neural Networks via Robust Loss Functions

    [https://arxiv.org/abs/2609.38263](https://arxiv.org/abs/2609.38263)

    本文提出Robust LassoNet，通过在LassoNet中引入Huber、Cauchy、Tukey双平方等鲁棒损失函数，在保留原始优化框架的同时，显著提升了神经网络在数据污染情况下的特征选择准确性与预测性能。

    

    神经网络中的特征选择仍然是一个具有挑战性的问题，尤其是在存在噪声或被污染数据的情况下。LassoNet是近期提出的一种应对该问题的方法，它将神经网络与分层稀疏性约束相结合，能够同时进行预测和变量选择。然而，其标准形式依赖于均方误差（MSE）损失，而MSE损失对异常值高度敏感是众所周知的。在本文中，我们提出了Robust LassoNet，这是LassoNet的一种扩展，它引入了鲁棒损失函数，如Huber损失、Cauchy损失、Tukey双平方损失和非负Garrote损失，以减轻极端观测值的影响。所提出的方法在保留原始优化框架的同时，提高了数据被污染情况下的稳定性。通过在合成数据集和真实数据集上的实验，我们证明了鲁棒LassoNet在预测性能和特征选择准确性方面均有显著提升。

    arXiv:2609.38263v1 Announce Type: new  Abstract: Feature selection in neural networks remains a challenging problem, particularly in the presence of noisy or contaminated data. LassoNet is a recent approach that addresses this issue by combining neural networks with hierarchical sparsity constraints, enabling simultaneous prediction and variable selection. However, its standard formulation relies on the mean squared error (MSE) loss, which is known to be highly sensitive to outliers. In this paper, we present Robust LassoNet, an extension of LassoNet that incorporates robust loss functions, such as Huber, Cauchy, Tukey's bisquare, and Nonnegative Garrote, to mitigate the effect of extreme observations. The proposed approach preserves the original optimization framework while improving stability under data contamination. Through experiments on synthetic and real datasets, we show that robust LassoNet significantly improves both predictive performance and feature selection accuracy in th
    
[^47]: 具有线性收敛率的可证明无参数不动点算法

    Provable Parameter-Free Fixed-Point Algorithms with Linear Convergence Rates

    [https://arxiv.org/abs/2608.09022](https://arxiv.org/abs/2608.09022)

    本文提出了两种无需压缩因子先验知识的无参数Halpern不动点迭代算法，并证明了对不动点残差和到唯一不动点距离的显式线性收敛率。

    

    在本文中，我们为压缩映射开发了可证明的无参数不动点算法，重点在于在不需要压缩因子先验知识的情况下自动利用隐藏的压缩性。我们的第一种方法是无参数的Halpern不动点迭代变体，无需线搜索、二分法或压缩因子的先验估计，同时保持与经典不动点方法基本相同的每次迭代计算成本。我们为不动点残差和到唯一不动点的距离建立了显式的线性收敛率。第二种算法同样是无参数的Halpern方法，但它需要线搜索过程来寻找压缩因子的局部估计，这可能需要几次额外的算子评估。该方法同样具有显式的线性收敛保证。我们进一步在两个方向上扩展了这些想法。首先

    arXiv:2608.09022v2 Announce Type: replace-cross  Abstract: In this paper, we develop provable parameter-free fixed-point algorithms for contractive mappings, with an emphasis on automatically exploiting hidden contractivity without requiring prior knowledge of the contraction factor. Our first method is a parameter-free variant of the Halpern fixed-point iteration with no line-search, bisection, or prior estimate of the contraction factor, while retaining essentially the same per-iteration computational cost as classical fixed-point schemes. We establish explicit linear convergence rates for both the fixed-point residual and the distance to the unique fixed point. The second algorithm is also a parameter-free Halpern method, but it requires a line-search procedure to find a local estimate for the contraction factor, which may need a few additional operator evaluations. This method also enjoys explicit linear convergence guarantees. We further extend these ideas in two directions. First
    
[^48]: 基于粒子的广义随机优化

    Particle-based Generalised Stochastic Optimisation

    [https://arxiv.org/abs/2608.02844](https://arxiv.org/abs/2608.02844)

    本文提出了针对不可解梯度损失函数的基于扩散的广义随机粒子优化框架，通过平均场动力学及其交互粒子近似统一了多种现有算法，证明了指数收敛性与非渐近误差界，并据此构建了高阶Langevin和动量优化器。

    

    我们为具有难以求解梯度的损失函数开发了一类基于扩散的通用随机粒子优化方法。具体而言，我们考虑损失梯度是关于一个依赖于参数的分布的积分这一问题结构，该结构涵盖了训练生成模型、微调以及学习潜变量模型等场景。我们引入了平均场动力学及其交互粒子近似，其中包含多种现有算法作为特例，并为构建新方法提供了途径。在适定性与联合压缩性假设下，我们证明了指数收敛性，并表明连续时间粒子系统满足非渐近误差界。我们通过开发高阶Langevin和动量优化器，并在最大边际似然估计和基于能量的模型训练上进行评估，展示了该框架的应用。

    arXiv:2608.02844v2 Announce Type: replace-cross  Abstract: We develop a class of general diffusion-based stochastic particle optimisation methods for loss functions with intractable gradients. Specifically, we consider problems in which the loss gradient is an integral with respect to a parameter-dependent distribution, a structure that includes training generative models, fine-tuning, and learning latent-variable models. We introduce mean-field dynamics and its interacting-particle approximations, which contain several existing algorithms as special cases and provide a route to constructing new methods. Under well-posedness and joint contractivity assumptions, we prove exponential convergence and show that the continuous-time particle system admits a non-asymptotic error bound. We illustrate our framework by developing higher-order Langevin and momentum optimisers and evaluating them on maximum marginal-likelihood estimation and energy-based-model training.
    
[^49]: 公平性研究中的平等、公平与因果性：对Cheng（2026）的评论

    Equality, Equity, and Causality in Fairness Research: A Commentary on Cheng (2026)

    [https://arxiv.org/abs/2607.17679](https://arxiv.org/abs/2607.17679)

    该评论在Cheng（2026）对心理测量学与AI/ML公平性系统比较的基础上，进一步探讨了平等与公平的概念区别以及因果性在公平性研究中的作用，为跨学科公平性研究的未来方向提供了指引。

    

    这是一篇受邀撰写的评论，针对《心理测量学》重点文章《心理测量学与AI/ML中的公平性问题与评估：我们能从彼此领域学到什么？》（作者Ying Cheng，2026）。Cheng对长期存在的测验公平性与现代算法公平性进行了系统性比较。她将整个测验工作流程（而不仅仅是最终的选拔阶段）映射到AI/ML公平性范式上，这是对跨学科公平性研究的一项重要贡献。本评论通过探讨两个概念性问题来扩展她的讨论：平等与公平之间的区别，以及因果性在公平性研究中的作用。这篇重点文章和本评论共同为心理测量学界与AI/ML学界未来的公平性研究指明了方向。

    arXiv:2607.17679v2 Announce Type: replace-cross  Abstract: This is an invited commentary on the Psychometrika focus article "Fairness Issues and Evaluation in Psychometrics and AI/ML: What Can We Learn from Each Field?" by Ying Cheng (2026, doi:10.1017/psy.2026.10110). Cheng offers a systematic comparison between long-standing test fairness and modern algorithmic fairness. Her mapping of the entire testing workflow onto the AI/ML fairness paradigm, rather than only the final selection stage, is a crucial contribution to interdisciplinary fairness research. This commentary extends her discussion by examining two conceptual issues: the distinction between equality and equity, and the role of causality in fairness research. The focus article and this commentary point to directions for future fairness research across the psychometrics and AI/ML communities.
    
[^50]: 微调过程中伪相关的无监督识别与消除

    Unsupervised Identification and Removal of Spurious Correlations During Fine-Tuning

    [https://arxiv.org/abs/2605.27676](https://arxiv.org/abs/2605.27676)

    本文证明可以在无监督条件下从朴素LoRA微调的权重中识别出微调引入的伪相关潜在因素，并提出应消除伪相关本身而非潜在因素，以避免破坏预训练模型依赖该因素获得的有效任务信号。

    

    在经过人工筛选的数据集上对预训练语言模型进行微调时，可能会在微调任务与非预期的潜在因素之间产生伪相关——例如失调的人格倾向或政治立场偏向——这些因素是数据筛选过程与任务纠缠在一起的。模型可能会依赖这些伪相关，从而导致偏见和分布外泛化能力下降。我们证明，在对任务复杂度和伪相关性做出合理假设的前提下，这类潜在因素可以在无监督的情况下，从朴素LoRA微调的权重中被识别出来。现有的消除偏见的方法（如激活引导）是在推理阶段或训练过程中从残差流激活中移除已识别的因素。然而，我们认为目标应该是消除伪相关本身，而不是潜在因素本身，因为预训练模型可能依赖该因素来获取真实的任务信号。为此，我们提出……

    arXiv:2605.27676v2 Announce Type: replace-cross  Abstract: Fine-tuning a pretrained language model on a curated dataset can produce spurious correlations between the fine-tuning task and unintended latent factors---such as misaligned personas or political slant---that the curation procedure has entangled with the task. The model can latch onto these spurious correlations, leading to bias and reduced out-of-distribution generalisation. We prove that under reasonable assumptions on task complexity and the spurious correlation, such latent factors can be identified, without supervision, from the weights of a naive LoRA fine-tune. Existing approaches to removing bias, such as activation steering, remove identified factors from residual-stream activations, either at inference or during training. We argue, however, that the goal should be to remove the spurious correlation, not the latent factor itself, as the pretrained model may rely on it for genuine task signal. To enable this, we propos
    
[^51]: SURF：调控标量化权重以均匀遍历帕累托前沿

    SURF: Steering the Scalarization Weight to Uniformly Traverse the Pareto Front

    [https://arxiv.org/abs/2605.20619](https://arxiv.org/abs/2605.20619)

    该论文提出SURF方法，通过将权重变化时解在帕累托前沿上的移动速度积分得到的累积弧长归一化为CDF，并利用其反函数调控标量化权重的采样，从而实现对帕累托前沿的均匀覆盖。

    

    标量化方法因其简单性和可扩展性在多目标优化中被广泛使用。在许多应用中，目标是生成能够代表多样化用户偏好的解，理想情况下这些解应均匀地覆盖帕累托前沿（PF）。然而，对标量化权重进行均匀采样通常会导致帕累托前沿的非均匀覆盖。我们通过标量化路径的几何结构解释了这种不匹配现象：随着权重的变化，相应的解通常以不均匀的速度遍历帕累托前沿。对该速度进行积分可得累积弧长，经总弧长归一化后，它定义了一个关于权重的累积分布函数（CDF），其反函数可用于选取等弧长间隔的点。基于这一见解，我们提出了SURF（沿帕累托前沿均匀采样）方法。对于包括双目标老虎机在内的结构化问题，我们推导了该CDF映射的解析表达式以及由此产生的……（摘要被截断）

    arXiv:2605.20619v2 Announce Type: replace  Abstract: Scalarization is widely used in multi-objective optimization owing to its simplicity and scalability. In many applications, the goal is to generate solutions that represent diverse user preferences, ideally with uniform coverage of the Pareto front (PF). However, uniformly sampling scalarization weights usually induces non-uniform coverage of the PF. We explain this mismatch through the geometry of the scalarization path. As the weight varies, the corresponding solutions traverse the PF at a generally nonuniform speed. Integrating this speed gives cumulative arc length. After normalization by the total length, it defines a cumulative distribution function (CDF) on weights whose inverse selects points at equal arc-length intervals. Building on this insight, we propose SURF (Sampling Uniformly along the Pareto Front). For structured problems, including bi-objective bandits, we derive analytical expressions for this CDF map and the resu
    
[^52]: 奥卡姆剃刀的锋利程度取决于你的ELBO

    Occam's Razor is Only as Sharp as Your ELBO

    [https://arxiv.org/abs/2604.25984](https://arxiv.org/abs/2604.25984)

    本文证明在贝叶斯线性回归中，基于ELBO的超参数学习会因高斯近似后验协方差结构的不同而产生系统性偏差：对角或低秩加对角协方差会低估信噪比导致欠拟合，而低秩协方差会高估信噪比导致过拟合。

    

    边际似然，也被称为证据，被视为奥卡姆剃刀的数学体现，使超参数选择能够避免过拟合。来自变分推断的证据下界目标函数也被用于类似的目的。在本文中，我们针对贝叶斯线性回归定义了一个信噪比，该信噪比由控制信号方差和噪声方差的关键超参数表示。随后，我们研究了基于ELBO的超参数学习如何根据高斯近似后验中所假设的协方差矩阵结构而低估或高估信噪比。我们证明了假设对角协方差矩阵或低秩加对角协方差矩阵会低估信噪比，而低秩协方差矩阵会高估信噪比。低估信噪比会导致训练集上的均方误差高于最优值（欠拟合），而高估信噪比则会导致误差更低（过拟合）。

    arXiv:2604.25984v2 Announce Type: replace-cross  Abstract: The marginal likelihood, also known as the evidence, is regarded as a mathematical embodiment of Occam's razor, enabling hyperparameter selection that avoids overfitting. The evidence lower bound (ELBO) objective from variational inference has also been used for similar purposes. In this paper, we define a signal-to-noise ratio (SNR) for Bayesian linear regression in terms of key hyperparameters controlling signal and noise variance. We then study how ELBO-based hyperparameter learning can under or overestimate SNR depending on the assumed covariance matrix structure in a Gaussian approximate posterior. We prove that assuming a diagonal or low-rank plus diagonal covariance matrix underestimates the SNR, while a low-rank covariance matrix overestimates the SNR. Underestimating the SNR leads to higher than optimal mean squared error on the training set (underfitting), while overestimating SNR leads to lower error (overfitting). B
    
[^53]: 本地隐私保护的变点检测参数化方法

    Locally Private Parametric Methods for Change-Point Detection

    [https://arxiv.org/abs/2602.13619](https://arxiv.org/abs/2602.13619)

    本文研究了本地差分隐私下的参数化变点检测问题，建立了以Chernoff信息为根本限制的有限样本极小极大上下界，并据此设计了分别适用于低隐私和高隐私机制的最优本地隐私变点检测算法，从而精确量化了隐私保护带来的统计代价。

    

    我们研究本地差分隐私下的参数化变点检测问题，重点关注对独立观测序列中单一变点的离线定位。在无隐私保护的设定下，我们利用鞅方法为广义似然比估计量推导出改进的有限样本精度保证。我们在无隐私和本地隐私两种设定下均建立了有限样本极小极大下界，表明定位精度从根本上受变点前后分布之间Chernoff信息的限制。受此启发，我们设计了基于聚合随机响应机制和二元机制的本地隐私变点检测算法，这两种算法分别在低隐私和高隐私机制下最大化诱导输出分布之间的Chernoff信息，并推导出其有限样本保证。我们的上下界共同量化了隐私保护所带来的统计代价。

    arXiv:2602.13619v2 Announce Type: replace-cross  Abstract: We study parametric change-point detection under local differential privacy, focusing on the offline localization of a single change point in a sequence of independent observations.~In the non-private setting, we derive improved finite-sample accuracy guarantees for a generalized likelihood-ratio estimator using martingale methods.~We establish finite-sample minimax lower bounds in both the non-private and locally private settings, showing that localization accuracy is fundamentally limited by the Chernoff information between the pre- and post-change distributions.~Guided by this, we design locally private CPD algorithms based on aggregated randomized response and binary mechanisms, which maximize the Chernoff information between the induced output distributions in low- and high-privacy regimes, respectively, and derive their finite-sample guarantees. Together, our upper and lower bounds quantify the statistical cost of privacy
    
[^54]: 基于图注意力实现跨客户端可解释性的非线性时间动态联邦学习

    Federated Learning of Nonlinear Temporal Dynamics with Graph Attention-based Cross-Client Interpretability

    [https://arxiv.org/abs/2602.13485](https://arxiv.org/abs/2602.13485)

    本文提出一种垂直联邦学习框架，在原始数据不可共享、客户端观测异构且各客户端模型固定不可重训的约束下，利用图注意力机制实现跨客户端的非线性时间动态一步预测，并具备跨客户端可解释性。

    

    现代工业系统的网络日益由分布式传感器进行监测，每个系统由多个子系统组成，这些子系统生成高维时间序列。这些子系统之间往往相互依赖，因此理解在一个子系统中观察到的时间模式与其他子系统中观察到的时间模式之间的关系至关重要。这一问题在去中心化环境中尤其具有挑战性，因为原始测量数据无法共享，且客户端的观测数据是异构的。此外，在实际应用中，每个子系统（客户端）运行着一个固定的、专有的模型，该模型无法被修改或重新训练，这限制了现有方法的适用性。另外，非线性动态使得跨客户端的时间相互依赖关系难以刻画，因为它们隐式地嵌入在非线性状态转移函数之中。本文提出了一种垂直联邦学习框架，用于学习一步前瞻的跨客户端时间动态……

    arXiv:2602.13485v3 Announce Type: replace  Abstract: Networks of modern industrial systems are increasingly monitored by distributed sensors, where each system comprises multiple subsystems that generate high-dimensional time-series. These subsystems are often interdependent, making it important to understand how temporal patterns observed at one subsystem relate to those observed across others. This problem is particularly challenging in decentralized settings where raw measurements cannot be shared, and client observations are heterogeneous. Moreover, in practice, each subsystem (client) operates a fixed, proprietary model that cannot be modified or retrained, limiting the applicability of existing approaches. Separately, nonlinear dynamics make cross-client temporal interdependencies difficult to characterize, as they are embedded implicitly within nonlinear state-transition functions. This paper presents a vertical federated learning framework for learning one-step-ahead cross-clie
    
[^55]: 非平稳加权风险最小化的快速收敛率

    Fast Rates for Nonstationary Weighted Risk Minimization

    [https://arxiv.org/abs/2602.05742](https://arxiv.org/abs/2602.05742)

    本文将分布漂移下的超额风险分解为学习误差与漂移误差，对非平稳加权经验风险最小化建立了在任意权重类上一致成立的快速oracle界，并在结构突变与平滑漂移过程以及线性模型、基近似和神经网络等回归问题中刻画了最优权重。

    

    加权经验风险最小化是分布漂移下预测问题的一种常用方法。本文研究了非平稳条件下该方法的样本外预测误差。我们将超额风险给出了一个一般性的分解：学习项与分布漂移相关的误差项，并在混合条件下证明了学习误差的oracle不等式。该学习界对任意权重类一致成立，并考虑了由权重向量诱导的有效样本量、权重类与假设类的复杂度以及潜在的数据依赖性。我们还分析了平方损失设定下的漂移误差，并针对结构突变过程和平滑漂移过程给出了显式界。随后，我们将学习分析与漂移分析相结合，应用于线性模型、基函数近似和神经网络的（自）回归问题。在所有情形下，最优权重都得到了刻画。

    arXiv:2602.05742v3 Announce Type: replace-cross  Abstract: Weighted empirical risk minimization is a common approach to prediction under distribution drift. This article studies its out-of-sample prediction error under nonstationarity. We provide a general decomposition of the excess risk into a learning term and an error term associated with distribution drift, and prove oracle inequalities for the learning error under mixing conditions. The learning bound holds uniformly over arbitrary weight classes and accounts for the effective sample size induced by the weight vector, the complexity of the weight and hypothesis classes, and potential data dependence. We also analyze the drift error in the square loss setting and provide explicit bounds for structural-break and smoothly drifting processes. We then combine the learning and drift analysis in (auto-) regression problems with linear models, basis approximations, and neural networks. In all cases, the optimal weights are characterized 
    
[^56]: 生物与技术分布偏移下调控序列建模的鲁棒机器学习

    Robust Machine Learning for Regulatory Sequence Modeling under Biological and Technical Distribution Shifts

    [https://arxiv.org/abs/2601.14969](https://arxiv.org/abs/2601.14969)

    提出了一个结合机制仿真基准与MPRA真实数据的鲁棒性评估框架，用于量化深度学习模型在生物与技术分布偏移下进行调控序列预测时的性能退化、校准失效与可靠性。

    

    本研究探讨了在生物和技术因素引起的分布偏移下调控基因组学的鲁棒机器学习问题。基于深度卷积和注意力机制的模型在DNA调控序列预测任务上取得了较强的分布内性能，但通常是在独立同分布假设下进行评估的，而实际应用中会涉及细胞类型特异性程序、进化更替、实验方案变化和测序伪影。我们提出了一个鲁棒性评估框架，将机制仿真基准与大规模并行报告基因检测数据集的真实数据分析相结合，以量化性能退化、校准失效以及基于不确定性的可靠性。在仿真中，通过细胞类型特异性程序、PWM扰动、GC偏差、测序深度变化、批次效应和异方差噪声生成基序驱动的调控输出，并对CNN、BiLSTM和transformer模型进行了评估。

    arXiv:2601.14969v3 Announce Type: replace-cross  Abstract: Robust machine learning for regulatory genomics is studied under biologically and technically induced distribution shifts. Deep convolutional and attention based models achieve strong in distribution performance on DNA regulatory sequence prediction tasks but are usually evaluated under i.i.d. assumptions, even though real applications involve cell type specific programs, evolutionary turnover, assay protocol changes, and sequencing artifacts. We introduce a robustness framework that combines a mechanistic simulation benchmark with real data analysis on a massively parallel reporter assay (MPRA) dataset to quantify performance degradation, calibration failures, and uncertainty based reliability. In simulation, motif driven regulatory outputs are generated with cell type specific programs, PWM perturbations, GC bias, depth variation, batch effects, and heteroscedastic noise, and CNN, BiLSTM, and transformer models are evaluated.
    
[^57]: 不完整前瞻信息下多用途批处理过程的贝叶斯动态调度

    Bayesian dynamic scheduling of multipurpose batch processes under incomplete look-ahead information

    [https://arxiv.org/abs/2512.01093](https://arxiv.org/abs/2512.01093)

    该论文提出了一种基于从扰动概率分布中学习贝叶斯网络的动态调度方法，使多用途批处理过程能够在仅拥有不完整前瞻信息的现实条件下有效应对扰动，从而最小化长期成本和系统紧张性。

    

    多用途批处理过程因其能够适应低产量、高价值产品和不断变化的需求，在制造业中日益受到欢迎。这些过程通常在动态环境中运行，面临诸如加工延迟和需求变化等扰动。为了最小化长期成本和系统紧张性（即对调度产生的破坏性改变），调度员必须设计重调度策略以有效应对此类扰动。现有方法通常假设在整个调度范围内拥有完整的前瞻信息。这一假设与现实情况不符——实际中调度员只能获取不完整的前瞻信息。沿用现有方法可能导致次优的长期成本和高水平的系统紧张性。在本工作中，我们提出了一种贝叶斯动态调度方法。我们的方法依赖于从扰动的概率分布中学习贝叶斯网络。

    arXiv:2512.01093v2 Announce Type: replace  Abstract: Multipurpose batch processes become increasingly popular in manufacturing industries since they adapt to low-volume, high-value products and shifting demands. These processes often operate in a dynamic environment, which faces disturbances such as processing delays and demand changes. To minimise long-term cost and system nervousness (i.e., disruptive changes to schedules), schedulers must design rescheduling strategies to address such disturbances effectively. Existing methods often assume complete look-ahead information over the scheduling horizon. This assumption contrasts with realistic situations where schedulers can only access incomplete look-ahead information. Sticking with existing methods may lead to suboptimal long-term costs and high-level system nervousness. In this work we propose a Bayesian dynamic scheduling method. Our method relies on learning a Bayesian Network from the probability distribution of disturbances. Spe
    
[^58]: 无穷小算子世界模型：离线强化学习的有限样本保证

    Infinitesimal Operator World Models: Finite-Sample Guarantees for Offline Reinforcement Learning

    [https://arxiv.org/abs/2511.10383](https://arxiv.org/abs/2511.10383)

    该论文将强化学习与HJB方程相联系，提出基于受控扩散过程无穷小生成元（在再生核希尔伯特空间中学习）的算子理论世界模型，在生成元近似误差足够小时证明了值函数的全局收敛性，并为离线强化学习提供了有限样本保证。

    

    连续时间随机过程是许多自然系统与工程系统的基础。在医疗保健、自动驾驶和工业控制等领域，与环境的直接交互往往是不安全或不切实际的，这促使人们从历史数据中进行离线强化学习。然而，对于从离线数据集中学习策略所固有的近似误差，目前的统计学理解仍然有限。我们通过将强化学习与哈密顿-雅可比-贝尔曼（HJB）方程联系起来来解决这一问题，并提出了一种基于简单动态规划递归的算子理论算法。具体而言，我们以受控扩散过程的无穷小生成元来表示世界模型，并在再生核希尔伯特空间中对其进行学习。通过融合统计学习方法与算子理论，我们在生成元近似误差足够小的条件下建立了值函数的全局收敛性，并由此推导出有限样本保证……

    arXiv:2511.10383v2 Announce Type: replace-cross  Abstract: Continuous-time stochastic processes underlie many natural and engineered systems. In healthcare, autonomous driving, and industrial control, direct interaction with the environment is often unsafe or impractical, motivating offline reinforcement learning from historical data. However, there is limited statistical understanding of the approximation errors inherent in learning policies from offline datasets. We address this by linking reinforcement learning to the Hamilton-Jacobi-Bellman equation and proposing an operator-theoretic algorithm based on a simple dynamic programming recursion. Specifically, we represent our world model in terms of the infinitesimal generator of controlled diffusion processes learned in a reproducing kernel Hilbert space. By integrating statistical learning methods and operator theory, we establish global convergence of the value function under sufficiently small generator-approximation error and der
    
[^59]: 信息力学理论：通过不确定性测量实现泛化（学习即测量）

    A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)

    [https://arxiv.org/abs/2510.22809](https://arxiv.org/abs/2510.22809)

    提出了一种基于信息论不确定性（惊奇度）的“无模型”学习框架，将学习视为对不确定性的测量，可直接从原始数据进行推理，支持数据的编辑与删除，并在异常检测等任务上达到最先进性能。

    

    传统机器学习依赖显式模型和领域假设，这常常限制了灵活性和可解释性。我们引入了一种“无模型”的基于实例的学习框架，利用惊奇度（surprisal，即信息论中的不确定性）直接对原始数据进行分析和推理，从而消除了分布建模的需求，减少了偏差，并支持高效的更新操作，包括直接编辑和删除训练数据。通过以不确定性来量化相关性，该方法能够在多种任务上实现可泛化的推理，包括生成式推理、因果发现、异常检测和时间序列预测。我们的方法强调可追溯性、可解释性和数据驱动的决策，为机器学习提供了一个统一的、人类可理解的框架，几乎可以适用于所有机器学习任务，在所有任务上均表现出色，并在异常检测上达到了最先进的性能。

    arXiv:2510.22809v2 Announce Type: replace-cross  Abstract: Traditional machine learning relies on explicit models and domain assumptions, often limiting flexibility and interpretability. We introduce a "model-free" framework of instance-based learning using surprisal (information theoretic uncertainty) to directly analyze and perform inferences from raw data, eliminating distribution modeling, reducing bias, and enabling efficient updates including direct edits and deletion of training data. By quantifying relevance through uncertainty, the approach enables generalizable inference across tasks including generative inference, causal discovery, anomaly detection, and time series forecasting. Our approach emphasizes traceability, interpretability, and data-driven decision making, offering a unified, human-understandable framework for machine learning, can be adapted to virtually all machine learning tasks, achieving competency across all and state-of-the-art performance on anomaly detecti
    
[^60]: 面向空间预测的代理图划分

    Surrogate Graph Partitioning for Spatial Prediction

    [https://arxiv.org/abs/2510.07832](https://arxiv.org/abs/2510.07832)

    本文提出将代理图划分问题建模为混合整数二次规划，用于构建使分段内预测方差之和最小的空间分段，并针对其计算复杂度过高的问题开发了高效的近似求解方案。

    

    空间预测是指从空间分布的观测值中估计未观测到的值。尽管近期的进展提升了对多种观测类型建模的能力，但在需要可解释性的行业中，实际应用仍然有限。为弥合这一差距，能够解释黑盒预测器的代理模型为实现可解释决策提供了一条有前景的路径。在本研究中，我们提出一个图划分问题，用于构建空间分段，使各分段内个体预测值的方差之和最小。将数据点分配到分段的问题可以被表述为一个混合整数二次规划问题。虽然这种表述有可能识别出精确的分段，但随着数据点数量的增加，其计算复杂度会变得难以承受。受此挑战的启发，我们开发了一种利用该结构的近似求解方案……

    arXiv:2510.07832v2 Announce Type: replace-cross  Abstract: Spatial prediction refers to the estimation of unobserved values from spatially distributed observations. Although recent advances have improved the capacity to model diverse observation types, adoption in practice remains limited in industries that demand interpretability. To mitigate this gap, surrogate models that explain black-box predictors provide a promising path toward interpretable decision making. In this study, we propose a graph partitioning problem to construct spatial segments that minimize the sum of within-segment variances of individual predictions. The assignment of data points to segments can be formulated as a mixed-integer quadratic programming problem. While this formulation potentially enables the identification of exact segments, its computational complexity becomes prohibitive as the number of data points increases. Motivated by this challenge, we develop an approximation scheme that leverages the struc
    
[^61]: 贝叶斯最优实验设计中微分熵的近似

    Approximation of differential entropy in Bayesian optimal experimental design

    [https://arxiv.org/abs/2510.00734](https://arxiv.org/abs/2510.00734)

    该论文针对贝叶斯最优实验设计提出了一种高效计算方法：当似然熵与设计无关或可显式计算时，期望信息增益的嵌套积分可简化为单积分，并通过蒙特卡洛或拟蒙特卡洛代理近似证据密度来高效估计微分熵。

    

    贝叶斯最优实验设计为选择能够最大化所获信息的实验配置提供了一个原则性框架。在本工作中，我们考虑似然熵与设计无关或可被显式计算的情形，这适用于例如具有设计无关或可显式计算噪声熵的加性噪声模型。这将问题简化为最大熵估计，其中定义期望信息增益的嵌套积分塌缩为单个积分。我们的研究受到大规模推断问题（如逆问题）的启发，此类问题的计算成本主要由昂贵的似然评估所主导。在加性高斯噪声模型下，我们提出了一种计算方法：证据密度由蒙特卡洛或拟蒙特卡洛代理进行近似，而微分熵则使用标准方法进行评估，无需……

    arXiv:2510.00734v2 Announce Type: replace-cross  Abstract: Bayesian optimal experimental design provides a principled framework for selecting experimental configurations that maximize obtained information. In this work, we consider settings where the likelihood entropy is independent of the design or can be evaluated explicitly, which applies, e.g. to additive noise models with design-independent or explicitly computable noise entropy. This reduces the problem to maximum entropy estimation, where the nested integral defining the expected information gain collapses into a single integral.   Our study is motivated by large-scale inference problems, such as inverse problems, where the computational cost is dominated by expensive likelihood evaluations. Under additive Gaussian noise model, we propose a computational approach in which the evidence density is approximated by a Monte Carlo or quasi-Monte Carlo surrogate, while the differential entropy is evaluated using standard methods witho
    
[^62]: 用于集成异构模型输出的可优化模型平均方法

    Optimizable Model Averaging for Ensembling Heterogeneous Model Outputs

    [https://arxiv.org/abs/2505.21857](https://arxiv.org/abs/2505.21857)

    提出可优化模型平均（OMA），一种黑盒集成方法，通过最小化正则化预测熵来学习非负组合权重以融合异构架构模型的预测概率，无需微调或参数合并即可改善性能与校准。

    

    基础模型集成可以提高准确性和鲁棒性，但当候选模型具有不同的架构时，完全微调和参数空间合并通常代价高昂甚至无法实现。我们提出了可优化模型平均（OMA），这是一种黑盒方法，它通过学习非负的模型组合权重（每个模型对应一个标量系数）来平均候选模型的预测概率，同时保持所有模型参数冻结，并通过最小化正则化预测熵来实现。一个核心挑战是最小化预测熵可能偏向那些过度自信但错误的模型：置信度的提升并不必然意味着准确性或校准的改善。因此，我们研究了正则化作为控制适应性的一种机制，并通过实验评估了其对校准的影响，观察到在选定配置中期望校准误差更低，同时相比模型输出平均方法具有更好的性能-校准权衡。

    arXiv:2505.21857v2 Announce Type: replace  Abstract: Foundation model ensembles can improve accuracy and robustness, but full fine-tuning and parameter-space merging are often expensive or impossible when candidate models have different architectures. We propose Optimizable Model Averaging (OMA), a black-box approach that learns nonnegative model-combination weights, one scalar coefficient per model, to average the candidate models' predictive probabilities, leaving all model parameters frozen, by minimizing regularized predictive entropy. A central challenge is that minimizing predictive entropy can favour confidently incorrect models: increased confidence need not imply improved accuracy or calibration. We therefore examine regularization as a mechanism for controlling adaptation and empirically evaluate its effect on calibration, observing lower expected calibration error in selected configurations, alongside better performance-calibration trade-offs than model output averaging. As 
    
[^63]: 解耦总方差与信噪比以改进扩散模型

    Disentangling Total-Variance and Signal-to-Noise-Ratio Improves Diffusion Models

    [https://arxiv.org/abs/2502.08598](https://arxiv.org/abs/2502.08598)

    提出了一种总方差/信噪比解耦（TV/SNR）框架，使扩散模型噪声调度中的总方差和信噪比可被独立控制，并发现恒定总方差调度能够持续改进指数爆炸总方差的调度。

    

    arXiv:2502.08598v3 公告类型：替换 摘要：扩散模型作为一种强大的生成框架已被广泛应用于众多科学领域，但其固有的迭代采样过程仍然是一个显著的计算瓶颈。虽然通常可以通过减少反向扩散步数来缓解这一问题，但样本质量会对所选的噪声调度（即在每个步骤中引入噪声和降低信号的方式）变得高度敏感。尽管先前的工作在原始的方差保持和方差爆炸调度基础上有所改进，但这些方法只是被动地调整总方差，而无法对其进行直接的独立控制。在这项工作中，我们提出了一种新颖的总方差/信噪比解耦（TV/SNR）框架，使总方差和信噪比能够被独立控制。我们的方法揭示，通过采用恒定总方差调度并保持（摘要原文在此处截断）……

    arXiv:2502.08598v3 Announce Type: replace  Abstract: Diffusion models have been widely adopted as a powerful generative framework across numerous scientific domains, yet their inherently iterative sampling process remains a significant computational bottleneck. While this is typically mitigated by reducing the number of reverse diffusion steps, sample quality becomes highly sensitive to the chosen noise schedule, i.e. the manner in which noise is introduced and the signal is reduced at each step. Although prior work has improved on the original variance-preserving and variance-exploding schedules, these approaches passively adjust the total variance, without direct independent control over it. In this work, we propose a novel total-variance/signal-to-noise-ratio disentangled (TV/SNR) framework, where TV and SNR can be controlled independently. Our approach reveals that schedules with exponentially exploding TV can be consistently improved by adopting a constant TV schedule while preser
    
[^64]: 再生核希尔伯特空间中的数据驱动随机最优控制

    Data-Driven Stochastic Optimal Control in Reproducing Kernel Hilbert Spaces

    [https://arxiv.org/abs/2407.16407](https://arxiv.org/abs/2407.16407)

    该论文提出一种完全数据驱动的方法，在动力学和成本函数均未知的情况下，通过在再生核希尔伯特空间中学习马尔可夫转移算子，并与随状态维度线性扩展的凸算子理论HJB递归相结合，实现非线性随机扩散系统的最优控制。

    

    本文提出了一种完全数据驱动的方法，用于由随机扩散所描述的非线性仿射控制系统的最优控制。研究聚焦于非线性动力学和阶段成本函数均未知、仅提供控制惩罚函数和约束条件的场景。为此，我们将状态概率密度嵌入到再生核希尔伯特空间（RKHS）中，以利用算子回归领域的最新进展，从而识别与受控扩散过程相关联的马尔可夫转移算子。这种算子学习方法与随状态维度线性扩展的凸算子理论哈密顿-雅可比-贝尔曼（HJB）递归自然融合，能够有效求解广泛的非线性最优控制问题。数值结果证明了该方法处理多种非线性控制任务的能力，包括自主水下航行器的深度调节。

    arXiv:2407.16407v3 Announce Type: replace-cross  Abstract: This paper proposes a fully data-driven approach for optimal control of nonlinear control-affine systems represented by a stochastic diffusion. The focus is on the scenario where both the nonlinear dynamics and stage cost functions are unknown, while only a control penalty function and constraints are provided. To this end, we embed state probability densities into a reproducing kernel Hilbert space (RKHS) to leverage recent advances in operator regression, thereby identifying Markov transition operators associated with controlled diffusion processes. This operator learning approach integrates naturally with convex operator-theoretic Hamilton-Jacobi-Bellman recursions that scale linearly with state dimensionality, effectively solving a wide range of nonlinear optimal control problems. Numerical results demonstrate its ability to address diverse nonlinear control tasks, including the depth regulation of an autonomous underwater 
    
[^65]: 具有近似保证的线性化Wasserstein降维方法

    Linearized Wasserstein dimensionality reduction with approximation guarantees

    [https://arxiv.org/abs/2302.07373](https://arxiv.org/abs/2302.07373)

    该论文提出LOT Wassmap算法，利用Sinkhorn距离和线性化最优传输等近似方案避免计算两两Wasserstein距离矩阵，从而高效揭示概率测度数据集在Wasserstein空间中的低维结构，并为此类近似下的嵌入质量提供了理论保证。

    

    我们提出了LOT Wassmap，一种计算上可行的算法，用于揭示Wasserstein空间中的低维结构。该算法的动机来自这样一个观察：许多数据集天然地应被理解为概率测度而非$\mathbb{R}^n$中的点，而为此类数据集寻找低维描述需要在Wasserstein空间中使用流形学习算法。现有的大多数算法都依赖于计算两两Wasserstein距离矩阵，这对于高维大规模数据集而言在计算上极具挑战性。我们的算法利用Sinkhorn距离和线性化最优传输等近似方案来加速计算，尤其避免了计算两两距离矩阵。我们为在这种近似下的嵌入质量提供了理论保证，包括当概率测度的显式表达不可用、而必须处理从测度中采样得到的样本的情形。

    arXiv:2302.07373v2 Announce Type: replace  Abstract: We introduce LOT Wassmap, a computationally feasible algorithm to uncover low-dimensional structures in the Wasserstein space. The algorithm is motivated by the observation that many datasets are naturally interpreted as probability measures rather than points in $\mathbb{R}^n$, and that finding low-dimensional descriptions of such datasets requires manifold learning algorithms in the Wasserstein space. Most available algorithms are based on computing the pairwise Wasserstein distance matrix, which can be computationally challenging for large datasets in high dimensions. Our algorithm leverages approximation schemes such as Sinkhorn distances and linearized optimal transport to speed-up computations, and in particular, avoids computing a pairwise distance matrix. We provide guarantees on the embedding quality under such approximations, including when explicit descriptions of the probability measures are not available and one must dea
    

