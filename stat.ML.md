# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Ranking Approach for Measuring Calibration](https://arxiv.org/abs/2609.13100) | 本文提出了一种新的校准误差度量方法 rankECE，通过比较预测概率的邻近值来衡量校准误差，理论和实验证明它比常用的分箱近似方法能更好地逼近期望校准误差（ECE）。 |
| [^2] | [Benign Loss Landscapes Can Coexist with Worst-Case Hardness](https://arxiv.org/abs/2609.13057) | 本文研究了树张量网络，证明其损失景观虽然是条件良性的（易于优化），但仍包含梯度下降无法在多项式时间内学习的最坏情况目标，从而揭示了良性损失景观可以与最坏情况的学习难度共存。 |
| [^3] | [Transfer Learning for Evolving Domains](https://arxiv.org/abs/2609.13039) | 该论文提出了一种新的迁移学习问题——面向演化领域的迁移学习，将传统上按目标数据可用性假设相互分割的各子领域统一起来，把迁移学习建模为部署系统随时间逐步获得目标数据与标签的完整演化轨迹。 |
| [^4] | [A Full Adam Theorem for Spectral Heavy-Tail Onset](https://arxiv.org/abs/2609.12996) | 该论文在高斯Stein-Hermite师生状态演化模型中证明了谱重尾出现的完整Adam定理，给出了由首个尖峰-主体谱间隙决定的命中时间定律 τ_ε = Θ(Δ_1^{-γ} d^ρ log(Ψ_0/ε))。 |
| [^5] | [Dimension-Corrected Hitting Times for Heavy-Tailed Spectral Emergence in Neural Optimizer Dynamics](https://arxiv.org/abs/2609.12994) | 该论文将神经网络训练中重尾谱的涌现时间首次建模为右删失命中时间问题，并实证发现由谱隙与权重矩阵维度共同决定的修正幂律 τ_HT ≈ C·Δ₁^(-γ)·d^ρ 能够准确预测重尾涌现时间，显著优于仅依赖谱隙的模型。 |
| [^6] | [Convergence of Stochastic Gradient Methods under Heavy-Tailed Noise and H\"{o}lder Smoothness](https://arxiv.org/abs/2609.12785) | 本文在同时放宽Lipschitz光滑性与有限方差噪声这两个经典假设的条件下，证明了标准SGD和$\delta$-正则化梯度裁剪方法在Hölder光滑目标函数和重尾梯度噪声下的非凸收敛速率。 |
| [^7] | [A Splitting Method for SDE Terminal-Law Estimation](https://arxiv.org/abs/2609.12513) | 本文全面研究了通过分裂部分路径生成路径树来提高随机微分方程终端分布估计效率的方法，以Kolmogorov-Smirnov距离为精度度量确定了模拟预算趋于无穷时的极限误差，并刻画了由渐近优化问题驱动的最优分裂策略。 |
| [^8] | [Linear Exponential Quadratic Gaussian Covariance Steering](https://arxiv.org/abs/2609.12463) | 本文提出并分析了连续时间下的线性指数二次高斯（LEQG）协方差控制问题，证明最优线性状态反馈控制器由一个通过求解编码风险敏感度参数隐式依赖的代数方程的对称矩阵参数化，并将风险中性情形的现有结果显著推广。 |
| [^9] | [Exact Community Recovery in Bipartite Networks](https://arxiv.org/abs/2609.12445) | 本文证明了一个基于删除对角线Gram矩阵的简单谱聚类算法，在稀疏性、社区平衡性和聚类数量的温和条件下，能以高概率在二分网络中实现精确的社区恢复。 |
| [^10] | [Inference for Newton Methods with Accelerated Sketch-and-Project via Random Scaling](https://arxiv.org/abs/2609.12421) | 本文提出了一种基于广义加速草图投影（GAS）求解器的在线草图牛顿方法，并通过随机缩放建立了其平均迭代的渐近正态性，证明了其极限协方差通常能比未加速方法更快地收敛到极小极大最优协方差。 |
| [^11] | [A Multimodal Explainable Deep Learning Framework for Alzheimer's Disease Diagnosis using 3D Magnetic Resonance Imaging and Clinical Data](https://arxiv.org/abs/2609.12410) | 该研究开发了一个结合3D磁共振成像和临床数据的可解释多模态深度学习框架用于阿尔茨海默病诊断，并首次系统考察了其模型解释在不同模态、融合策略和队列间的一致性表现。 |
| [^12] | [Representation-based Masked Diffusion Model](https://arxiv.org/abs/2609.12382) | 提出基于表示的掩码扩散模型（RMDM），通过预训练编码器将文本表示归一化为高斯先验以显式编码全局语义，从而协调被掩码词元的并行更新，生成更连贯的文本。 |
| [^13] | [Membership Inference via Pairwise Likelihood Ratios](https://arxiv.org/abs/2609.12367) | 提出了PL-MIA，一种将高斯似然比统计量、总体校准与柯西组合检验相结合的统一成员推断攻击方法，能够有效汇总模型输出的统计信号并从理论上提升攻击能力。 |
| [^14] | [PDE-constrained inverse problems at the $\sqrt{n}$ rate via debiased physics-informed neural networks](https://arxiv.org/abs/2609.12301) | 本文提出一种结合神经网络非参数估计与影响函数偏差校正的两步去偏估计方法，使PDE约束逆问题的有限维参数估计达到√n一致性和渐近正态性，且无需对神经网络进行欠平滑处理。 |
| [^15] | [Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding](https://arxiv.org/abs/2609.12243) | 本文提出切薄-共识幂采样（CCPS），通过Chopthin重采样器限制最大与最小权重比值并保留不等权重，避免了等权重重采样对低权重推理路径的过度剪除，在保持SMC近似无偏的同时维持推理路径多样性并保证有效样本量下界。 |
| [^16] | [Rank-Efficient LoRA via Joint Tangent-Space Optimization under Isotropic Curvature](https://arxiv.org/abs/2609.12123) | 该论文发现LoRA的标称秩并不等于实际利用的表示能力，优化器会显著影响更新中的有效秩，并据此提出ISO-LoRA——一种通过对权重空间诱导切向扰动进行谱下降来联合耦合LoRA因子更新的优化器，从而更充分、更高效地利用秩容量。 |
| [^17] | [Score-based Outlier Generation via Controlling the Radon-Nikodym Derivative](https://arxiv.org/abs/2609.12113) | 该论文提出通过似然分布的Radon-Nikodym导数对扩散模型分数函数进行受控缩放，从而无需重新训练即可生成幅度可控的低似然离群值。 |
| [^18] | [Learning Interaction Kernels from Collective Steady States](https://arxiv.org/abs/2609.12004) | 该论文提出了一种仅需从集体稳态的单快照观测中学习相互作用粒子系统相互作用核的方法，通过基于观测构型经验分布的正则化策略解决了本质上不适定的逆问题，实现了对相互作用规律的稳定准确恢复以及对集体行为乃至其动力学过程的忠实重现。 |
| [^19] | [DCRA: Diffusion-Conditioned Representation Alignment for Robust Time-Series Learning](https://arxiv.org/abs/2609.11997) | 该论文提出DCRA训练框架，将扩散前向过程重新用作结构化损坏调度器，并通过特征级一致性目标在噪声水平间对齐表示且保持类别判别结构，从而在噪声和分布偏移下学习鲁棒的时间序列表示，尤其适用于EEG和ECG等临床信号分析。 |
| [^20] | [Semiparametric Inference for Conditional Shapley Feature Importance](https://arxiv.org/abs/2609.10313) | 本文针对条件Shapley特征重要性提出了一种带K折交叉拟合和U统计量修正的半参数一步估计器，消除了蒙特卡洛偏差，在双重稳健速率条件下实现√n一致性与渐近正态性，并提供覆盖率有保证的Wald置信区间。 |
| [^21] | [Transformers as In-Context Samplers: From Closed-Form Diffusion to Estimation-Free Sampling](https://arxiv.org/abs/2609.08981) | 本文证明冻结的Transformer可以通过上下文学习模拟迭代式生成采样器（如闭式扩散采样器），其中softmax注意力计算责任权重与加权经验均值、前馈层实现欧拉更新，从而将上下文学习能力从监督学习拓展到数据生成任务。 |
| [^22] | [Dynamic Latent Space Modeling of Inhomogeneous Poisson Network Processes with Applications to International Relations](https://arxiv.org/abs/2609.08813) | 该论文提出了一种针对非齐次泊松过程的动态潜在空间模型，利用B样条建模时变潜在距离，解决了模型可识别性与大规模计算的可扩展性问题，并成功应用于揭示1995至2022年间60个主要经济体国际合作模式的演变。 |
| [^23] | [High-resolution Calibrated Probabilistic Hourly Precipitation from a Deterministic Forecast](https://arxiv.org/abs/2608.12685) | 本文提出了一种基于注意力残差U-Net的神经网络方法，通过混合Gamma分布和气候学先验，实现了对确定性降水预报的高分辨率概率校准，提升了小时降水预测的准确性。 |
| [^24] | [The General Theory of Localization Methods](https://arxiv.org/abs/2605.20635) | 本文提出了一个以局部化核与局部均值为核心概念的通用机器学习理论框架——局部化方法，通过两个理论支柱（局部化模型构建与局部化技巧）统一解释了核方法、自注意力机制、Hopfield网络等多种现有机器学习模型背后的共同原理。 |
| [^25] | [A data-driven Fourier-mixture neural-network method for density estimation](https://arxiv.org/abs/2605.18019) | 该论文提出一种在傅里叶空间中直接训练、具有闭式特征函数的正定高斯-拉普拉斯混合神经网络密度估计方法，并针对独立同分布采样与基于重采样的伪采样两种情形分别给出了可分离傅里叶截断、训练、离散化与采样误差的期望 L₂ 密度误差理论界。 |
| [^26] | [Adapt or Forget: Provable Tradeoffs Between Adam and SGD in Nonstationary Optimization](https://arxiv.org/abs/2605.04269) | 本文首次对Adam在非平稳随机优化中的表现给出理论分析，将有限时间界清晰分解为初始化、目标漂移、一阶矩跟踪误差和预条件器扰动四个分量，并揭示了噪声与漂移之间可证明的权衡关系。 |
| [^27] | [Bias-Corrected Data Synthesis for Imbalanced Learning](https://arxiv.org/abs/2510.26046) | 该论文提出了一种偏差校正的数据合成方法，通过从留出的多数类数据估计生成器引起的损失偏差并将其转移至少数类，为不平衡学习建立了有限样本理论保证，同时刻画了SMOTE产生显著损失偏差的条件。 |
| [^28] | [An ab initio foundation model of wavefunctions that accurately describes chemical bond breaking](https://arxiv.org/abs/2506.19960) | 该研究提出了Orbformer——一个在22,000个平衡与解离结构上预训练的可迁移波函数基础模型，只需在未见分子上微调即可实现与经典多参考方法相媲美的精度-成本比，从而准确描述化学键断裂这一量子化学难题。 |
| [^29] | [Aligning Language Models with Observational Data: Opportunities and Risks from a Causal Perspective](https://arxiv.org/abs/2506.00152) | 本文从因果视角系统分析了利用历史观测数据微调大语言模型的机遇与风险，指出观测结果可作为A/B测试的低成本替代监督信号，但同时需警惕因果混淆带来的偏差。 |
| [^30] | [Scalable Krylov Subspace Methods for Generalized Mixed-Effects Models with Crossed Random Effects](https://arxiv.org/abs/2505.09552) | 该论文提出了基于Krylov子空间的新方法，解决了广义混合效应模型中高维交叉随机效应导致的计算瓶颈，在保持同等精度的同时实现了几个数量级的加速。 |
| [^31] | [Can SGD Select Good Fishermen? Local Convergence under Self-Selection Biases](https://arxiv.org/abs/2504.07133) | 本文提出了首个针对自选择偏差的局部收敛算法，通过将自选择问题归约为粗化估计问题，给出了运行时间为poly(d, k, 1/ε) + (k log k)^{O(k)}的更快算法，从而解决了CDIZ23提出的主要开放问题之一。 |
| [^32] | [A Generalized Tangent Approximation based Variational Inference Framework for Strongly Super-Gaussian Likelihoods](https://arxiv.org/abs/2504.05431) | 本文提出了一种基于广义切变换的变分推断框架，利用凸对偶性构建对数似然的切下界，使强超高斯似然类概率模型与高斯先验实现共轭，从而将该结构化变分方法的应用范围从逻辑回归扩展到更广泛的模型类别。 |
| [^33] | [Statistical Uncertainty Quantification for Aggregate Performance Metrics in Machine Learning Benchmarks](https://arxiv.org/abs/2501.04234) | 该论文展示了如何利用自助法和贝叶斯分层建模等统计方法，来量化机器学习基准测试中跨多个任务聚合的性能指标的不确定性。 |
| [^34] | [Satisficing Regret Minimization in Bandits: Constant Rate and Light-Tailed Distribution](https://arxiv.org/abs/2406.06802) | 本文提出SELECT算法模板，通过采样与下置信界检验，在赌博机满意遗憾最小化问题中实现了常数级别的期望满意遗憾，并同时具备标准遗憾保证。 |

# 详细

[^1]: 一种用于衡量校准的排序方法

    A Ranking Approach for Measuring Calibration

    [https://arxiv.org/abs/2609.13100](https://arxiv.org/abs/2609.13100)

    本文提出了一种新的校准误差度量方法 rankECE，通过比较预测概率的邻近值来衡量校准误差，理论和实验证明它比常用的分箱近似方法能更好地逼近期望校准误差（ECE）。

    

    当使用预测模型提供预测概率时，理想的模型应具有完美校准：结果的真实概率（即 $Y=1$ 的概率）应与预测概率 $f(X)$ 完全一致。在实践中，模型不可避免地会出现校准误差，因此能够测量这种校准误差以评估模型的可靠性非常重要。期望校准误差（ECE）是目前使用最广泛的校准误差度量，但已知在无假设的环境中无法保证对 ECE 进行准确估计。在这项工作中，我们提出了一种替代性度量方法 rankECE，该方法基于将数据点与预测概率 $f(X)$ 的邻近值进行比较。我们的理论保证和实证结果表明，与实践中最常用的分箱近似方法相比，rankECE 为 ECE 提供了更好的代理。

    arXiv:2609.13100v1 Announce Type: cross  Abstract: When providing forecasted probabilities with a predictive model, the ideal model offers perfect calibration: the true probability of the outcome (i.e., the probability that $Y=1$) exactly matches the forecasted probability $f(X)$. In practice, models inevitably exhibit calibration error, and it is therefore important to be able to measure this miscalibration to assess a model's reliability. The Expected Calibration Error (ECE) is the most widely used measure of miscalibration, but is known to be impossible to estimate the ECE with guaranteed accuracy in an assumption-free setting. In this work, we propose an alternative measure, the rankECE, that is based on comparing points with neighboring values of the predicted probability $f(X)$. Our theoretical guarantees and empirical results establish that rankECE provides a better proxy for ECE as compared to binned approximations to ECE, which are the most commonly-used approximations in prac
    
[^2]: 良性损失景观可以与最坏情况难度共存

    Benign Loss Landscapes Can Coexist with Worst-Case Hardness

    [https://arxiv.org/abs/2609.13057](https://arxiv.org/abs/2609.13057)

    本文研究了树张量网络，证明其损失景观虽然是条件良性的（易于优化），但仍包含梯度下降无法在多项式时间内学习的最坏情况目标，从而揭示了良性损失景观可以与最坏情况的学习难度共存。

    

    深度神经网络足够富有表现力，能够包含可以在多项式时间内评估、但无法通过梯度下降在多项式时间内学习的最坏情况目标。然而，对于实际任务，它们却学习得很好，这引发了一个问题：现实世界目标的何种非通用结构使得这一点成为可能。现有的代理模型无法提出这个问题，因为它们要么完全缺乏难以学习的目标（深度线性网络），要么无法高效地评估此类目标（核方法、无限宽度极限）。我们研究了树张量网络（TTN），这是一种泛化了深度线性网络和Tucker分解的模型类。我们证明它们可以嵌入任意的单次读取布尔公式，因此在与神经网络相同的机制下，包含无法被梯度下降在多项式时间内学习的多项式规模目标。尽管如此，我们证明了它们的损失景观对于每个可实现的目标都是条件良性的。

    arXiv:2609.13057v1 Announce Type: new  Abstract: Deep neural networks are expressive enough to contain worst-case targets that can be evaluated in polynomial time but cannot be learned in polynomial time by gradient descent. For practical tasks they nonetheless learn well, raising the question of what non-generic structure of real-world targets enables this. Existing surrogate models cannot pose this question because they either lack hard-to-learn targets entirely (deep linear networks) or cannot evaluate such targets efficiently (kernel methods, infinite-width limits). We study tree tensor networks (TTNs), a model class that generalizes deep linear networks and Tucker decompositions. We show they embed arbitrary read-once Boolean formulas, and thus contain polynomial-size targets that cannot be learned by gradient descent in polynomial time under the same mechanism as neural networks. Despite this, we prove that their loss landscapes are conditionally benign for every realizable targe
    
[^3]: 面向演化领域的迁移学习

    Transfer Learning for Evolving Domains

    [https://arxiv.org/abs/2609.13039](https://arxiv.org/abs/2609.13039)

    该论文提出了一种新的迁移学习问题——面向演化领域的迁移学习，将传统上按目标数据可用性假设相互分割的各子领域统一起来，把迁移学习建模为部署系统随时间逐步获得目标数据与标签的完整演化轨迹。

    

    迁移学习研究如何利用来自不同任务或领域（源）的知识，来提升相关任务或领域（目标）中的预测性能。通常，迁移学习研究被划分为若干相互孤立的子领域（如领域泛化、领域自适应或多领域学习），每个子领域对目标数据的可用性做出不同的假设，即训练时有多少数据以及多少标签可用。然而，在许多现实应用中，数据可用性并非固定不变，而是随着实例和标签从新领域逐步收集而随时间演化。每一种经典设定实际上仅描述了已部署系统必须完整经历的一条轨迹中的某个快照。我们将这条轨迹形式化为一个独立的迁移学习问题——面向演化领域的迁移学习，该问题由环境固定的数据可用性过程来定义。

    arXiv:2609.13039v1 Announce Type: new  Abstract: Transfer learning explores how to leverage knowledge from various tasks or domains (sources) to enhance predictive performance in related tasks or domains (targets). Typically, transfer learning research is segmented into several isolated sub-areas (such as domain generalisation, domain adaptation, or multi-domain learning), each making distinct assumptions about target data availability, namely how much data and how many labels are available at training time. However, in many real-world applications, data availability is not fixed but evolves over time, as instances and labels are progressively collected from a new domain. Each of the classical settings then describes only a snapshot of a trajectory that a deployed system must traverse in full. We formalise this trajectory as a transfer learning problem in its own right, Transfer Learning for Evolving Domains (TrED), specified by a data availability process fixed by the environment, a l
    
[^4]: 谱重尾出现的完整Adam定理

    A Full Adam Theorem for Spectral Heavy-Tail Onset

    [https://arxiv.org/abs/2609.12996](https://arxiv.org/abs/2609.12996)

    该论文在高斯Stein-Hermite师生状态演化模型中证明了谱重尾出现的完整Adam定理，给出了由首个尖峰-主体谱间隙决定的命中时间定律 τ_ε = Θ(Δ_1^{-γ} d^ρ log(Ψ_0/ε))。

    

    我们在一个闭合的高斯Stein-Hermite师生状态演化模型中，证明了关于谱重尾出现的完整Adam定理。该定理从实际的全批量Adam递推关系出发，通过Stein-Hermite微积分推导出总体梯度，证明了有限宽度协方差集中性，将多步Adam动量转换为精确的非中心高斯符号核，通过基同质化定理控制Adam的对角分母，从Hermite边缘转移定理推导出正则变化的投影更新响应，将该响应推过精确的Gram更新，并证明了具有匹配上下界命中结果的近似目标KL收缩。最终得到的定律为 τ_ε = Θ(Δ_1^{-γ} d^ρ log(Ψ_0/ε))，其中 Δ_1 是第一个尖峰-主体谱间隙。该结果是“完整”的，其精确含义为：从Adam的动量和分母到谱（原文在此处截断）……

    arXiv:2609.12996v1 Announce Type: new  Abstract: We prove a full Adam theorem for spectral heavy-tail onset in a closed Gaussian Stein-Hermite teacher-student state-evolution model. The theorem begins with the actual full-batch Adam recurrences, derives the population gradient by Stein-Hermite calculus, proves finite-width covariance concentration, converts multi-step Adam momentum into an exact non-centered Gaussian sign kernel, controls the diagonal Adam denominator by a basis-homogenization theorem, derives a regularly varying projected update response from a Hermite edge-transfer theorem, pushes the response through the exact Gram update, and proves approximate-target KL contraction with matching upper and lower hitting bounds. The final law is (\tau_\varepsilon=\Theta(\Delta_1^{-\gamma}d^\rho\log(\Psi_0/\varepsilon))), where (\Delta_1) is the first spike-bulk spectral gap. The result is full in the following precise sense: every step from Adam's momentum and denominator to the spe
    
[^5]: 神经网络优化器动力学中重尾谱涌现的维度修正命中时间

    Dimension-Corrected Hitting Times for Heavy-Tailed Spectral Emergence in Neural Optimizer Dynamics

    [https://arxiv.org/abs/2609.12994](https://arxiv.org/abs/2609.12994)

    该论文将神经网络训练中重尾谱的涌现时间首次建模为右删失命中时间问题，并实证发现由谱隙与权重矩阵维度共同决定的修正幂律 τ_HT ≈ C·Δ₁^(-γ)·d^ρ 能够准确预测重尾涌现时间，显著优于仅依赖谱隙的模型。

    

    神经网络权重矩阵的经验谱密度呈现重尾分布，这被广泛用作隐式自正则化的诊断指标，但重尾涌现所需的步数复杂度仍知之甚少。我们将谱重尾的形成形式化为一个右删失命中时间问题：在观测范围内未达到重尾诊断指标的运行被视为删失数据，而非被丢弃。在受控的全批量教师-学生动力学中，我们发现仅凭第一步的尖峰-体隙（spike-bulk gap）无法解释涌现时间。相反，有限涌现时间的回归结果支持一个维度修正的谱隙定律 τ_HT ≈ C·Δ₁^(-γ)·d^ρ，在330个完整运行中 R²=0.683，γ=0.626，ρ=0.772。右删失对数正态加速失效时间模型进一步表明，维度修正模型优于仅考虑谱隙的模型，将AIC从706.62改进至628.70。在理论层面，我们……（原文摘要截断）

    arXiv:2609.12994v1 Announce Type: new  Abstract: Heavy-tailed empirical spectral densities of neural-network weight matrices are widely used as diagnostics of implicit self-regularization, but the step complexity of heavy-tail emergence remains poorly understood. We formulate spectral heavy-tail formation as a right-censored hitting-time problem: a run that does not reach a heavy-tail diagnostic within the observation horizon is treated as censored rather than discarded. In controlled full-batch teacher--student dynamics, we find that the first-step spike--bulk gap alone does not explain onset time. Instead, finite-onset regression supports a dimension-corrected spectral-gap law, (\tau_{\mathrm{HT}}\approx C\Delta_1^{-\gamma}d^\rho), with (R^2=0.683), (\gamma=0.626), and (\rho=0.772) across 330 completed runs. Right-censored lognormal accelerated-failure-time models further favor the dimension-corrected model over a gap-only model, improving AIC from 706.62 to 628.70. Theoretically, we
    
[^6]: 重尾噪声与Hölder光滑性下随机梯度方法的收敛性

    Convergence of Stochastic Gradient Methods under Heavy-Tailed Noise and H\"{o}lder Smoothness

    [https://arxiv.org/abs/2609.12785](https://arxiv.org/abs/2609.12785)

    本文在同时放宽Lipschitz光滑性与有限方差噪声这两个经典假设的条件下，证明了标准SGD和$\delta$-正则化梯度裁剪方法在Hölder光滑目标函数和重尾梯度噪声下的非凸收敛速率。

    

    经典随机梯度方法的收敛保证通常假设目标函数是Lipschitz光滑的，且梯度噪声具有有限方差，而这两个假设在实践中经常被违反。相比之下，我们研究了在同时放宽这些假设条件下的非凸随机优化问题：目标函数具有$(L,s)$-Hölder连续梯度，其中$s\in(0,1]$，且梯度噪声仅满足有界的$\alpha$阶矩条件，其中$\alpha\in(1,2]$。我们建立了三个收敛结果。首先，当$\alpha\ge1+s$时，标准SGD以$O(T^{-s/(1+s)})$的速率收敛，这将经典的非凸SGD收敛速率同时扩展到了重尾噪声和Hölder光滑性的情形。其次，我们分析了$\delta$-正则化梯度裁剪（$\delta$-GClip）——一种已被证明能够有效训练宽深神经网络的方法——并在相同条件下建立了$O(T^{-2s(\alpha-1)/[(1+s)(2\alpha-1)]})$的平稳性收敛速率。第三，我们分析了标准梯度裁剪方法（摘要原文在此处截断）

    arXiv:2609.12785v1 Announce Type: new  Abstract: Classical convergence guarantees for stochastic gradient methods typically assume Lipschitz-smooth objectives and finite-variance gradient noise, both frequently violated in practice. In contrast, we study nonconvex stochastic optimization under the joint relaxation of these assumptions: objectives with $(L,s)$-H\"older continuous gradients, $s\in(0,1]$, and gradient noise satisfying only a bounded $\alpha$-th moment condition for $\alpha\in(1,2]$. We establish three convergence results. Firstly, that standard SGD converges at rate $O(T^{-s/(1+s)})$ whenever $\alpha\ge1+s$, extending the classical nonconvex SGD rate to heavy-tailed noise and H\"older smoothness simultaneously. Secondly, we analyze $\delta$-regularized gradient clipping ($\delta$-GClip), a provable trainer of wide and deep nets, and establish a stationarity rate of $O(T^{-2s(\alpha-1)/[(1+s)(2\alpha-1)]})$ under the same condition. Thirdly, we analyze standard gradient cl
    
[^7]: 一种用于随机微分方程终端分布律估计的分裂方法

    A Splitting Method for SDE Terminal-Law Estimation

    [https://arxiv.org/abs/2609.12513](https://arxiv.org/abs/2609.12513)

    本文全面研究了通过分裂部分路径生成路径树来提高随机微分方程终端分布估计效率的方法，以Kolmogorov-Smirnov距离为精度度量确定了模拟预算趋于无穷时的极限误差，并刻画了由渐近优化问题驱动的最优分裂策略。

    

    在许多涉及随机微分方程的场景中，包括基于扩散的生成式人工智能，我们的目标是准确地从终端分布中生成样本。通常，这是通过生成独立同分布的扩散路径样本来实现的。在给定固定模拟预算的情况下，一种提高效率的合理方法可能是通过适当分裂的部分路径来生成路径树。这暗示了性能的提升，但人们会担心由此引入的依赖性问题。在本文中，我们全面研究了这一问题。以Kolmogorov-Smirnov距离作为精度度量，我们确定了当模拟预算增至无穷时相关经验分布的极限误差。我们刻画了一种由相应渐近优化问题所启发的分裂策略。理论结果揭示了该问题优雅的内在结构。实际实现包含两个阶段，

    arXiv:2609.12513v1 Announce Type: cross  Abstract: In many settings involving stochastic differential equations, including in diffusion based generative AI, our aim is to accurately generate samples from a terminal distribution. Typically, this is done by generating i.i.d. samples of diffusion paths. Given a fixed simulation budget, a reasonable way to gain efficiency may be to instead generate a tree of paths through appropriately split partial paths. This suggests improved performance, but one worries about the injected dependence. In this paper, we study this issue comprehensively. With Kolmogorov-Smirnov distance as a measure of accuracy, we identify the limiting errors of the associated empirical distributions as the simulation budget increases to infinity. We characterize a splitting strategy motivated by a corresponding asymptotic optimization problem. The theoretical results bring out the elegant underlying structure in the problem. Practical implementation involves two phases,
    
[^8]: 线性指数二次高斯协方差控制

    Linear Exponential Quadratic Gaussian Covariance Steering

    [https://arxiv.org/abs/2609.12463](https://arxiv.org/abs/2609.12463)

    本文提出并分析了连续时间下的线性指数二次高斯（LEQG）协方差控制问题，证明最优线性状态反馈控制器由一个通过求解编码风险敏感度参数隐式依赖的代数方程的对称矩阵参数化，并将风险中性情形的现有结果显著推广。

    

    我们提出并分析了在给定截止期限（有限时间范围）内连续时间下的线性指数二次高斯（LEQG）协方差控制问题。该问题的解可以看作线性二次设定下高斯端点之间的风险敏感薛定谔桥。与风险中性情形不同，LEQG协方差控制控制器——仍然是线性状态反馈——不再能写成闭式解形式。我们证明了最优控制器由一个对称矩阵参数化，该矩阵通过求解一个编码了对风险敏感度参数隐式依赖的代数方程得到。我们解释了该最优控制器的结构如何显著推广了风险中性情形的现有结果。基于这些结果，对于噪声与输入通道匹配的情形，我们证明了在k邻域内LEQG协方差控制问题解的存在唯一性。

    arXiv:2609.12463v1 Announce Type: cross  Abstract: We formulate and analyze the linear exponential quadratic Gaussian (LEQG) covariance steering problem in continuous time over a given deadline (finite time horizon). The solution for this problem can be seen as a risk-sensitive Schr\"{o}dinger bridge between Gaussian endpoints in the linear quadratic setting. Unlike the risk-neutral case, the LEQG covariance steering controller--still a linear state feedback--can no longer be written in closed form. We show that the optimal controller is parameterized by a symmetric matrix solving an algebraic equation that encodes the implicit dependence on the risk-sensitivity parameter. We explain how the structure of this optimal controller significantly generalizes the existing results for the risk-neutral case. Building on these results, for the matched noise and input channel case, we prove the existence-uniqueness of solution for the LEQG covariance steering problem in the neighborhood of the k
    
[^9]: 二分网络中的精确社区恢复

    Exact Community Recovery in Bipartite Networks

    [https://arxiv.org/abs/2609.12445](https://arxiv.org/abs/2609.12445)

    本文证明了一个基于删除对角线Gram矩阵的简单谱聚类算法，在稀疏性、社区平衡性和聚类数量的温和条件下，能以高概率在二分网络中实现精确的社区恢复。

    

    二分网络中的社区检测是现代数据分析中的一个基本问题，在推荐系统、生物网络和社交网络分析等领域有着广泛的应用。与传统的单部图不同，二分网络由两种不同类型的节点组成，边只在两种类型之间连接，因此恢复潜在社区需要对两种节点类型的标签同时进行估计。随机协同块模型（stochastic co-blockmodel）是此类网络的一种经典概率框架，但在该设置下精确社区恢复的理论保证仍然有限，尤其是在社区数量增长、社区规模不平衡或度分布异质的情况下。在这项工作中，我们证明了一个基于删除对角线的Gram矩阵的简单谱聚类算法，在稀疏性、社区平衡性和聚类数量的温和条件下，能够以高概率实现精确的社区恢复。

    arXiv:2609.12445v1 Announce Type: cross  Abstract: Community detection in bipartite networks is a fundamental problem in modern data analysis, with applications in recommendation systems, biological networks, and social network analysis. Unlike conventional unipartite graphs, bipartite networks consist of two distinct types of nodes with edges only connecting across types, so recovering latent communities requires estimating labels on the two node types. The stochastic co-blockmodel is a classical probabilistic framework for such networks, yet theoretical guarantees for exact community recovery in this setting remain limited, especially when the number of communities grows, the community sizes are unbalanced, or the degrees are heterogeneous. In this work, we prove that a simple spectral clustering algorithm based on the diagonal-deleted Gram matrix achieves exact recovery with high probability under mild conditions on sparsity, community balance, and the number of clusters. We further
    
[^10]: 基于随机缩放的加速草图投影牛顿方法的统计推断

    Inference for Newton Methods with Accelerated Sketch-and-Project via Random Scaling

    [https://arxiv.org/abs/2609.12421](https://arxiv.org/abs/2609.12421)

    本文提出了一种基于广义加速草图投影（GAS）求解器的在线草图牛顿方法，并通过随机缩放建立了其平均迭代的渐近正态性，证明了其极限协方差通常能比未加速方法更快地收敛到极小极大最优协方差。

    

    我们研究了一种在线草图牛顿方法，该方法通过一种最先进的草图求解器（称为广义加速草图投影求解器，GAS）在每一步近似牛顿方向，从而缓解了经典二阶方法的计算瓶颈。GAS求解器通过Nesterov动量更新实现加速收敛，从而改进了原始的未加速草图投影求解器，并且支持灵活的投影度量，其适当选择可进一步降低计算成本。基于这一设计，我们建立了平均草图牛顿迭代序列的渐近正态性，并刻画了其极限协方差矩阵。所得的协方差在特定加速参数选择下可恢复未加速草图牛顿方法的协方差，通常情况下（以草图步数计）能更快地收敛到极小极大最优协方差，并且小于……（摘要在此处截断）

    arXiv:2609.12421v1 Announce Type: cross  Abstract: We study an online sketched Newton method that approximates the Newton direction at each step via a state-of-the-art sketching solver, called the generalized accelerated sketch-and-project solver (GAS), thereby mitigating the computational bottleneck of classical second-order methods. The GAS solver improves upon vanilla, unaccelerated sketch-and-project solvers by achieving accelerated convergence through Nesterov momentum updates, and accommodates a flexible projection metric whose proper choice further reduces computational cost. Building on this design, we establish asymptotic normality of the averaged sketched Newton iterates and characterize their limiting covariance matrix. The resulting covariance recovers that of the unaccelerated sketched Newton method under a specific choice of acceleration parameters, converges more rapidly (in the number of sketching steps) to the minimax-optimal covariance in general, and is smaller than 
    
[^11]: 一种使用三维磁共振成像和临床数据进行阿尔茨海默病诊断的多模态可解释深度学习框架

    A Multimodal Explainable Deep Learning Framework for Alzheimer's Disease Diagnosis using 3D Magnetic Resonance Imaging and Clinical Data

    [https://arxiv.org/abs/2609.12410](https://arxiv.org/abs/2609.12410)

    该研究开发了一个结合3D磁共振成像和临床数据的可解释多模态深度学习框架用于阿尔茨海默病诊断，并首次系统考察了其模型解释在不同模态、融合策略和队列间的一致性表现。

    

    痴呆症是一项重大且不断增长的全球健康负担，其中阿尔茨海默病（AD）占大多数病例。及时准确的诊断是应对这一负担的核心，且日益依赖于整合互补的临床和影像信息。多模态深度学习可以结合这些模态用于AD诊断，但其解释在不同模态、融合策略和队列之间的表现仍不清楚。我们开发了一个可解释的多模态框架，将用于T1加权MRI的3D CNN编码器与用于标准化临床和人口统计学数据的前馈网络相结合，使用来自ADNI的6,479条内部记录和来自OASIS-3的1,703条独立记录，在三分类和两两分类诊断任务上比较了不同的模型设置。在ADNI上，仅表格数据模型实现了最高的三分类AUC-ROC 0.879，并在区分认知正常（CN）与轻度认知障碍（MCI；0.903）方面表现最佳，而c（摘要原文在此处截断）

    arXiv:2609.12410v1 Announce Type: cross  Abstract: Dementia is a major and growing global health burden, with Alzheimer's disease (AD) accounting for most cases. Timely and accurate diagnosis is central to managing this burden and increasingly depends on integrating complementary clinical and imaging information. Multimodal deep learning can combine these modalities for AD diagnosis, but how its explanations behave across modalities, fusion strategies, and cohorts remains unclear. We developed an explainable multimodal framework pairing a 3D CNN encoder for T1-weighted MRI with a feedforward network for harmonized clinical and demographic data, comparing varied model setups on three-way and pairwise diagnostic tasks using 6,479 internal records from the ADNI and 1,703 independent records from the OASIS-3. On ADNI, the tabular-only model achieved the highest three-class AUC-ROC of 0.879 and best discriminated cognitively normal (CN) versus mild cognitive impairment (MCI; 0.903), while c
    
[^12]: 基于表示的掩码扩散模型

    Representation-based Masked Diffusion Model

    [https://arxiv.org/abs/2609.12382](https://arxiv.org/abs/2609.12382)

    提出基于表示的掩码扩散模型（RMDM），通过预训练编码器将文本表示归一化为高斯先验以显式编码全局语义，从而协调被掩码词元的并行更新，生成更连贯的文本。

    

    掩码扩散模型（MDMs）已成为语言建模中一种极具吸引力的范式，提供了高效并行文本生成的能力。然而，现有的并行采样方法通常独立地更新多个被掩码的词元，忽略了被掩码词元之间复杂的相互依赖关系。这种独立更新机制缺乏全局协调，可能导致输出不连贯。为了解决这一局限性，我们提出了基于表示的掩码扩散模型（RMDM），这是一个利用文本表示显式编码全局语义、从而帮助更精确地并行更新词元的框架。具体而言，我们首先使用预训练编码器将文本编码到连续语义空间中，并学习一个可逆变换，将表示分布归一化为高斯先验，从而促进生成过程中的高效采样。以该潜在语义表示为条件……（原文摘要至此截断）

    arXiv:2609.12382v1 Announce Type: new  Abstract: Masked Diffusion Models (MDMs) have emerged as a compelling paradigm for language modeling, offering the capability for efficient parallel text generation. However, existing parallel sampling methods typically update multiple masked tokens independently and ignore the complex mutual dependencies among the masked tokens. This independent updating mechanism lacks global coordination and might lead to incoherent outputs. To address this limitation, we propose Representation-based Masked Diffusion Model (RMDM), a framework that leverages the text representation to explicitly encode global semantics and help to parallel update tokens more precisely. Specifically, we first encode text into a continuous semantic space using a pretrained encoder and learn an invertible transformation that normalizes the representation distribution to a Gaussian prior, facilitating efficient sampling during generation. Conditioned on this latent semantic represen
    
[^13]: 基于成对似然比的成员推断攻击

    Membership Inference via Pairwise Likelihood Ratios

    [https://arxiv.org/abs/2609.12367](https://arxiv.org/abs/2609.12367)

    提出了PL-MIA，一种将高斯似然比统计量、总体校准与柯西组合检验相结合的统一成员推断攻击方法，能够有效汇总模型输出的统计信号并从理论上提升攻击能力。

    

    成员推断攻击（MIA）是审计机器学习模型隐私风险的标准工具。给定一个查询点，成员推断攻击旨在确定该点是否被用于训练目标模型。在实践中，这种推断必须依赖于模型输出所暴露的统计信号，例如置信度分数、logits以及中间特征表示。然而，现有方法往往无法有效地汇总和组合这些统计信号。为了解决这一局限，我们提出了成对似然成员推断攻击（PL-MIA），这是一种统一的方法，将高斯似然比（GLR）统计量与总体校准和柯西组合检验相结合。我们从理论上刻画了GLR如何保留方差收缩信号，并建立了总体校准和柯西组合能够提升攻击能力的条件。我们通过查询点与……之间的成对比较获得p值（原文在此处截断）。

    arXiv:2609.12367v1 Announce Type: cross  Abstract: Membership inference attacks (MIAs) are the standard tool for auditing the privacy risks of machine learning models. Given a query point, an MIA aims to determine whether that point was used to train the target model. In practice, such inference must rely on the statistical signals exposed by the model's outputs, such as confidence scores, logits, and intermediate feature representations. However, existing methods often fail to efficiently summarize and combine these statistical signals. To address this limitation, we propose Pairwise Likelihood MIA (PL-MIA), a unified method that combines a Gaussian likelihood-ratio (GLR) statistic with population calibration and the Cauchy combination test. We characterize theoretically how the GLR retains variance-contraction signals and establish conditions under which population calibration and Cauchy combination improve attack power. We obtain $p$-values from pairwise comparisons between the quer
    
[^14]: 通过去偏物理信息神经网络实现√n速率的偏微分方程约束逆问题求解

    PDE-constrained inverse problems at the $\sqrt{n}$ rate via debiased physics-informed neural networks

    [https://arxiv.org/abs/2609.12301](https://arxiv.org/abs/2609.12301)

    本文提出一种结合神经网络非参数估计与影响函数偏差校正的两步去偏估计方法，使PDE约束逆问题的有限维参数估计达到√n一致性和渐近正态性，且无需对神经网络进行欠平滑处理。

    

    我们研究从含噪观测中估计偏微分方程（PDE）约束逆问题中未知参数的问题，其中PDE的解采用物理信息神经网络（PINNs）进行近似。尽管PINNs在实证中取得了显著成功，但现有估计器往往继承了神经网络解的缓慢非参数收敛速率，导致对感兴趣的有限维参数的推断存在偏差且统计效率低下。为解决这一问题，我们提出了一种两步去偏估计方法，将基于神经网络的非参数估计与基于影响函数的偏差校正相结合。通过消除估计器对冗余函数误差的一阶敏感性，我们的方法获得了一个√n一致且渐近正态的估计器，而无需对神经网络组件进行欠平滑处理。我们进一步将该框架扩展到贝叶斯推断

    arXiv:2609.12301v1 Announce Type: cross  Abstract: We study the problem of estimating unknown parameters in PDE-constrained inverse problems from noisy observations, where the PDE solution is approximated using Physics-Informed Neural Networks (PINNs). While PINNs have demonstrated remarkable empirical success, existing estimators often inherit the slow nonparametric convergence rate of the neural-network solution, leading to biased and statistically inefficient inference for the finite-dimensional parameters of interest. To address this, we propose a two-step debiased estimation procedure that combines neural-network-based nonparametric estimation with an influence-function-based bias correction. By eliminating the first-order sensitivity of the estimator to errors in the nuisance function, our procedure yields a $\sqrt{n}$-consistent and asymptotically normal estimator without requiring undersmoothing of the neural network component. We further extend this framework to Bayesian infer
    
[^15]: 切薄-共识幂采样：一种保持多样性的大语言模型解码方法

    Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding

    [https://arxiv.org/abs/2609.12243](https://arxiv.org/abs/2609.12243)

    本文提出切薄-共识幂采样（CCPS），通过Chopthin重采样器限制最大与最小权重比值并保留不等权重，避免了等权重重采样对低权重推理路径的过度剪除，在保持SMC近似无偏的同时维持推理路径多样性并保证有效样本量下界。

    

    通过序贯蒙特卡洛（SMC）进行的推理时幂采样可以在无需后训练的情况下显著提升大语言模型（LLM）的推理能力。然而，许多现有的SMC方法依赖于等权重重采样，这可能会激进地剪除低权重轨迹，丢弃潜在正确的推理路径，并降低搜索空间的谱系多样性。为了解决这一问题，我们提出了切薄-共识幂采样（CCPS）。我们的方法将Chopthin重采样器应用于LLM解码：它不是均衡权重并强制进行不必要的粒子复制，而是对最大权重与最小权重之间的比值施加一个上界，并将不相等的权重向前传递。这种有针对性的干预保留了更加丰富的不同推理路径集合，使加权SMC近似在条件期望下保持不变，并保证了重采样后有效样本量的下界。

    arXiv:2609.12243v1 Announce Type: new  Abstract: Inference-time power sampling via Sequential Monte Carlo (SMC) can substantially improve large language model (LLM) reasoning without requiring post-training. However, many existing SMC approaches rely on equal-weight resampling, which can aggressively prune low-weight trajectories, discarding potentially correct reasoning paths and degrading the genealogical diversity of the search space. To address this, we introduce Chopthin-Consensus Power Sampling (CCPS). Our method applies the Chopthin resampler to LLM decoding: rather than equalizing weights and forcing unnecessary particle duplication, it enforces an upper bound on the ratio between the largest and smallest weights and carries the unequal weights forward. This targeted intervention preserves a richer set of distinct reasoning paths, keeps the weighted SMC approximation unchanged in conditional expectation, and guarantees a lower bound on the post-resampling effective sample size 
    
[^16]: 各向同性曲率下通过联合切空间优化实现秩高效的LoRA

    Rank-Efficient LoRA via Joint Tangent-Space Optimization under Isotropic Curvature

    [https://arxiv.org/abs/2609.12123](https://arxiv.org/abs/2609.12123)

    该论文发现LoRA的标称秩并不等于实际利用的表示能力，优化器会显著影响更新中的有效秩，并据此提出ISO-LoRA——一种通过对权重空间诱导切向扰动进行谱下降来联合耦合LoRA因子更新的优化器，从而更充分、更高效地利用秩容量。

    

    低秩适应（LoRA）是一种通过学习低秩权重更新来适配大型预训练模型的有效方法。在实践中，LoRA的秩被用来控制适配器的参数预算和表示能力。我们表明这种观点是不完整的：虽然标称秩决定了表示能力，但优化器决定了在由此产生的权重空间更新中实际使用了多少这种能力。在一项使用LoRA对GPT-2进行适配的案例研究中，我们观察到强烈的秩相关优化器效应。尽管使用相同的标称秩，AdamW通常产生奇异谱集中且有效秩较低的每步更新，而Muon则使用更丰富的方向集合，并且从增加LoRA秩中获益更加一致。这些观察结果促使我们提出ISO-LoRA，这是一种通过在权重空间中对诱导的切向扰动进行谱下降来耦合LoRA因子更新的优化器。ISO-LoRA促进……

    arXiv:2609.12123v1 Announce Type: new  Abstract: Low-Rank Adaptation (LoRA) is an effective approach for adapting large pretrained models by learning low-rank weight updates. In practice, the LoRA rank is used to control an adapter's parameter budget and representational capacity. We show that this view is incomplete: while the nominal rank determines the representational capacity, the optimizer shapes how much of that capacity is used in the induced weight-space updates. In a case study of GPT-2 adaptation with LoRA, we observe a strong rank-dependent optimizer effect. Despite using the same nominal rank, AdamW often produces per-step updates with concentrated singular spectra and low effective rank, whereas Muon uses a richer set of directions and benefits more consistently from increasing LoRA rank. These observations motivate ISO-LoRA, an optimizer that couples the LoRA factor updates through spectral descent on the induced tangent perturbation in weight space. ISO-LoRA promotes up
    
[^17]: 通过控制Radon-Nikodym导数实现基于分数的离群值生成

    Score-based Outlier Generation via Controlling the Radon-Nikodym Derivative

    [https://arxiv.org/abs/2609.12113](https://arxiv.org/abs/2609.12113)

    该论文提出通过似然分布的Radon-Nikodym导数对扩散模型分数函数进行受控缩放，从而无需重新训练即可生成幅度可控的低似然离群值。

    

    离群值对于压力测试算法以及理解系统在罕见条件下的行为非常重要。尽管离群值通常被描述为低似然事件，但现有的生成方法很少显式地控制似然。在这项工作中，我们基于对数似然值的分布引入了一种测度论意义上的离群值概念，该方法保证将更高的概率质量分配给幅度可指定的低似然事件。基于这一表述，我们推导出似然重加权如何修改扩散分数，并利用这一关系来启发对反向时间动力学的受控修改。特别地，似然重加权意味着分数函数的缩放，其控制项由似然分布的Radon-Nikodym导数导出。相应地，更新后的分数函数无需对扩散模型进行重新训练即可获得。我们利用Ornstein（原文摘要在此处截断）

    arXiv:2609.12113v1 Announce Type: new  Abstract: Outliers are important for stress-testing algorithms and understanding system behaviour under rare conditions. Despite being commonly described as low-likelihood events, existing generative approaches rarely control likelihood explicitly. In this work, we introduce a measure-theoretic notion of outliers based on the distribution of log-likelihood values, which is guaranteed to assign higher probability mass to low-likelihood events with a specifiable magnitude. Building on this formulation, we derive how likelihood reweighting modifies the diffusion score and use this relation to motivate a controlled modification of the reverse-time dynamics. In particular, likelihood reweighting implies a scaling of the score function with a control term derived from the Radon-Nikodym derivative of the likelihood distributions. Correspondingly, the updated score function can be obtained with no retraining of the diffusion model. We exploit the Ornstein
    
[^18]: 从集体稳态中学习相互作用核

    Learning Interaction Kernels from Collective Steady States

    [https://arxiv.org/abs/2609.12004](https://arxiv.org/abs/2609.12004)

    该论文提出了一种仅需从集体稳态的单快照观测中学习相互作用粒子系统相互作用核的方法，通过基于观测构型经验分布的正则化策略解决了本质上不适定的逆问题，实现了对相互作用规律的稳定准确恢复以及对集体行为乃至其动力学过程的忠实重现。

    

    我们提出了一种从集体行为的单快照观测中对相互作用粒子系统进行系统辨识的学习方法，这与依赖轨迹观测的现有方法不同。这一设定导致了一个本质上不适定的逆问题，我们通过一种基于观测构型经验分布的正则化策略来解决该问题，这些构型来自不同的、未被观测到的初始条件。我们在多种具有稳态和准稳态模式的代表性模型上测试了该学习程序，在这些模型中，集体行为编码了关于相互作用机制的隐含信息。结果表明，我们的方法能够稳定且准确地恢复潜在的相互作用规律，从而忠实地重现集体行为，在许多情况下甚至能够重现导致该集体行为的动力学过程。

    arXiv:2609.12004v1 Announce Type: cross  Abstract: We propose a learning procedure for system identification in interacting particle systems from single-snapshot observations of collective behaviors, unlike existing approaches that rely on observations of trajectories. This setting leads to a fundamentally ill-posed inverse problem, which we solve by using a regularization strategy based on the empirical distribution of observed configurations, drawn from different, unobserved initial conditions. We test our learning procedure on a variety of representative models with steady-state and quasi-stationary patterns, where collective behaviors encode implicit information about the interaction mechanisms, demonstrating that our approach enables stable and accurate recovery of the underlying interaction laws, leading to faithful reproduction of the collective behavior, and in many cases even of the dynamics leading up to it.
    
[^19]: DCRA：面向鲁棒时间序列学习的扩散条件化表示对齐

    DCRA: Diffusion-Conditioned Representation Alignment for Robust Time-Series Learning

    [https://arxiv.org/abs/2609.11997](https://arxiv.org/abs/2609.11997)

    该论文提出DCRA训练框架，将扩散前向过程重新用作结构化损坏调度器，并通过特征级一致性目标在噪声水平间对齐表示且保持类别判别结构，从而在噪声和分布偏移下学习鲁棒的时间序列表示，尤其适用于EEG和ECG等临床信号分析。

    

    arXiv:2609.11997v1 公告类型：新论文 摘要：在噪声和分布偏移条件下学习时间序列信号的鲁棒表示仍然具有挑战性，尤其是在脑电图（EEG）和心电图（ECG）分析等临床应用中。我们提出了扩散条件化表示对齐（DCRA），这是一种训练框架，它将前向扩散过程重新用作表示学习的结构化损坏调度器。与依赖独立采样扰动的传统数据增强和基于一致性的方法不同，DCRA通过扩散前向过程引入了结构化的损坏轨迹，从而实现了跨噪声水平的连续且可控的表示演化。我们引入了一个特征级一致性目标，在保持类别判别结构的同时跨噪声水平对齐表示。这一机制促进了保持结构的一致性，从而实现平滑且具有语义……

    arXiv:2609.11997v1 Announce Type: new  Abstract: Learning robust representations for time-series signals under noise and distribution shifts remains challenging, especially in clinical applications such as electroencephalogram (EEG) and electrocardiogram (ECG) analysis. We propose Diffusion-Conditioned Representation Alignment (DCRA), a training framework that repurposes the forward diffusion process as a structured corruption scheduler for representation learning. Different from conventional augmentation and consistency-based methods that rely on independently sampled perturbations, DCRA introduces a structured corruption trajectory via the diffusion forward process, which enables continuous and controlled representation evolution across noise levels. We introduce a feature-level consistency objective that aligns representations across noise levels while preserving class-discriminative structure. This mechanism promotes structure-preserving consistency, which enables smooth and semant
    
[^20]: 条件Shapley特征重要性的半参数推断

    Semiparametric Inference for Conditional Shapley Feature Importance

    [https://arxiv.org/abs/2609.10313](https://arxiv.org/abs/2609.10313)

    本文针对条件Shapley特征重要性提出了一种带K折交叉拟合和U统计量修正的半参数一步估计器，消除了蒙特卡洛偏差，在双重稳健速率条件下实现√n一致性与渐近正态性，并提供覆盖率有保证的Wald置信区间。

    

    Shapley值被广泛用于事后特征归因，但大多数估计器仅返回点估计量而无法量化不确定性，且流行的实现方法从边际分布中采样联盟外特征，这在特征相关时会导致重要性归因错误。本文研究了条件形式化方法，即联盟外特征在其真实条件分布下被积分出去。目标是一个全局的、基于损失的重要性度量，它将条件价值函数与SAGE风格的损失聚合相结合。我们提出了一种结合K折交叉拟合的一步估计器，并对平方损失进行U统计量修正，从而消除了朴素插入估计器的蒙特卡洛偏差；该估计器在双重稳健速率条件下具有√n一致性和渐近正态性，由此构造的Wald置信区间达到了名义覆盖率。此外，还给出了Pinsker型界以量化工作条件分布被误设时产生的偏差。

    arXiv:2609.10313v1 Announce Type: cross  Abstract: Shapley values are widely used for post-hoc feature attribution, but most estimators return point quantities and do not quantify uncertainty, and popular implementations sample out-of-coalition features from their marginal distribution, which misattributes importance when features are dependent. This paper studies the conditional formulation, in which out-of-coalition features are integrated out under their true conditional distribution. The target is a global, loss-based importance that pairs a conditional value function with a SAGE-style loss aggregation. We propose a one-step estimator with K-fold cross-fitting and a U-statistic correction of the squared loss that removes the Monte Carlo bias of the naive plug-in; it is $\sqrt{n}$-consistent and asymptotically normal under double-robust rate conditions, and the resulting Wald interval attains nominal coverage. A Pinsker-type bound quantifies the bias from misspecifying the working c
    
[^21]: 作为上下文采样器的Transformer：从闭式扩散到无需估计的采样

    Transformers as In-Context Samplers: From Closed-Form Diffusion to Estimation-Free Sampling

    [https://arxiv.org/abs/2609.08981](https://arxiv.org/abs/2609.08981)

    本文证明冻结的Transformer可以通过上下文学习模拟迭代式生成采样器（如闭式扩散采样器），其中softmax注意力计算责任权重与加权经验均值、前馈层实现欧拉更新，从而将上下文学习能力从监督学习拓展到数据生成任务。

    

    越来越多的研究证实，大型语言模型并非仅仅是统计记忆器，而是具备上下文学习能力：在测试时仅使用提示中提供的样本进行推理，无需任何参数更新。先前的理论工作已表明，这种能力可以扩展到线性回归等监督学习任务。我们证明上下文学习可以进一步扩展到数据生成领域：冻结的Transformer可以从上下文样本中模拟迭代式生成采样器。我们首先表明，Transformer可以实现闭式和平滑闭式扩散采样器。该构造为softmax注意力机制确定了一个具体的生成性角色：它负责计算责任权重和加权经验均值，而前馈层则实现欧拉更新。为了在实证层面将这些构造与预训练语言模型建立联系，我们研究了语义主题采样：通过提示……

    arXiv:2609.08981v1 Announce Type: cross  Abstract: A growing body of work establishes that large language models are not mere statistical memorizers, but are capable of in-context learning: performing inference at test time using only examples provided in the prompt, without any parameter updates. Prior theoretical work has shown that this capability extends to supervised learning tasks such as linear regression. We prove that in-context learning extends further to \emph{data generation}: frozen transformers can simulate iterative generative samplers from in-context samples. We first show that transformers can realize closed-form and smoothed closed-form diffusion samplers. The construction identifies a concrete generative role for softmax attention: it computes responsibility weights and weighted empirical averages, while feedforward layers implement Euler updates.   To empirically relate these constructions to pretrained language models, we study \emph{semantic-topic sampling}: promp
    
[^22]: 非齐次泊松网络过程的动态潜在空间建模及其在国际关系中的应用

    Dynamic Latent Space Modeling of Inhomogeneous Poisson Network Processes with Applications to International Relations

    [https://arxiv.org/abs/2609.08813](https://arxiv.org/abs/2609.08813)

    该论文提出了一种针对非齐次泊松过程的动态潜在空间模型，利用B样条建模时变潜在距离，解决了模型可识别性与大规模计算的可扩展性问题，并成功应用于揭示1995至2022年间60个主要经济体国际合作模式的演变。

    

    我们研究连续时间关系事件数据，其中带有时间戳的二元交互既反映了个体节点的活动倾向，也反映了不断演变的关系亲近度。我们提出了一种针对非齐次泊松过程的动态潜在空间模型，其中事件强度依赖于节点特定的活动参数以及通过灵活的B样条建模的时变潜在距离。我们通过将基线活动与潜在位置解耦来证明模型的可识别性，确保高交互量不会扭曲空间映射。为了提高可扩展性，我们开发了一种具有稳定初始化和几何锚定的小批量随机梯度算法，并提出了一种基于有效自由度的BIC准则来调节模型复杂度。仿真结果证实了该模型能够准确恢复参数并进行样本外预测。将该模型应用于60个主要经济体（1995-2022年）的合作性外交事件，模型揭示了国际合作模式的动态变迁。

    arXiv:2609.08813v2 Announce Type: replace-cross  Abstract: We study continuous-time relational event data, where time-stamped dyadic interactions reflect both individual node propensities and evolving relational proximity. We propose a dynamic latent space model for inhomogeneous Poisson processes, where event intensities depend on node-specific activity parameters and time-varying latent distances modeled via flexible B-splines. We prove model identifiability by decoupling baseline activity from latent position, ensuring high interaction volumes do not warp the spatial map. For scalability, we develop a minibatch stochastic gradient algorithm with stable initialization and geometric anchoring, alongside an effective-degrees-of-freedom BIC for tuning model complexity. Simulations confirm accurate parameter recovery and out-of-sample prediction. Applied to cooperative diplomatic events among 60 major economies (1995--2022), the model uncovers shifting patterns of international cooperati
    
[^23]: 基于确定性预报的高分辨率校准概率小时降水预测

    High-resolution Calibrated Probabilistic Hourly Precipitation from a Deterministic Forecast

    [https://arxiv.org/abs/2608.12685](https://arxiv.org/abs/2608.12685)

    本文提出了一种基于注意力残差U-Net的神经网络方法，通过混合Gamma分布和气候学先验，实现了对确定性降水预报的高分辨率概率校准，提升了小时降水预测的准确性。

    

    arXiv:2608.12685v1 公告类型：交叉 摘要：本文描述了一种“注意力残差U-Net”方法，用于概率定量降水预报（PQPF），该方法预测无降水的每小时概率，以及通过两个Gamma分布的加权混合得到的正降水量分布。神经网络在天气公司的高分辨率对流允许GRAF（全球高分辨率大气预报）模型的数值天气预报（NWP）每小时降水数据块上进行训练，并结合了来自美国国家海洋和大气管理局（NOAA）全球预报系统（GFS）的地形信息和柱平均相对湿度。目标数据是NOAA的多雷达、多传感器（MRMS）雨量计校正、质量控制雷达数据，采样到与GRAF数据相同的网格上。网络为每个模型网格点输出分布参数。训练使用负对数似然作为适当的评分规则，并包含气候学先验信息。

    arXiv:2608.12685v1 Announce Type: cross  Abstract: An ``Attention Residual U-Net'' method is described for probabilistic quantitative precipitation forecasting (PQPF) that predicts the hourly probability of no precipitation plus the distribution of positive precipitation from a weighted mixture of two Gamma distributions. The neural network is trained on patches of numerical weather prediction (NWP) hourly precipitation from The Weather Company's convection-permitting GRAF (Global high-Resolution Atmospheric Forecasting) model along with terrain information and column-average relative humidity from the National Oceanic and Atmospheric Administration's (NOAA's) Global Forecast System (GFS). The target data are NOAA's Multi-Radar, Multi-Sensor (MRMS) gauge-corrected, quality controlled radar data sampled to the same grid as the GRAF data. The network outputs distributional parameters for each model grid point. Training uses negative log-likelihood as a proper scoring rule, with climatolo
    
[^24]: 局部化方法的普适理论

    The General Theory of Localization Methods

    [https://arxiv.org/abs/2605.20635](https://arxiv.org/abs/2605.20635)

    本文提出了一个以局部化核与局部均值为核心概念的通用机器学习理论框架——局部化方法，通过两个理论支柱（局部化模型构建与局部化技巧）统一解释了核方法、自注意力机制、Hopfield网络等多种现有机器学习模型背后的共同原理。

    

    本文提出了一种称为局部化方法的通用机器学习框架，其根本建立在两个核心概念之上：局部化核与局部均值——这两个关键组件正是自注意力机制的基石。为了建立严格的理论基础，该框架通过两个基本支柱被正式定义：局部（化）模型的构建与局部化技巧。我们系统地研究了局部化方法与众多现有机器学习模型/方法之间的联系，包括（但不限于）核方法、惰性学习、MeanShift算法、松弛标注、Hopfield网络、局部线性嵌入（LLE）、模糊推理以及去噪自编码器（DAE）。通过剖析这些关系，我们阐明了局部化方法更广泛的理论意义，并证明了其在多样化机器学习场景中的实际适用性。

    arXiv:2605.20635v4 Announce Type: replace  Abstract: This paper proposes a general machine learning framework called the localization method, which is fundamentally built on two core concepts: localization kernels and local means -- key components that underpin the self-attention mechanism. To establish a rigorous theoretical foundation, the framework is formally defined through two essential pillars: the formulation of the local(-ized) model and the localization trick. We systematically investigate the connections between the localization method and a wide range of existing machine learning models/methods, including (but not limited to) kernel methods, lazy learning, the MeanShift algorithm, relaxation labeling, Hopfield networks, local linear embedding (LLE), fuzzy inference, and denoising autoencoders (DAEs). By dissecting these relationships, we clarify the broader theoretical significance of the localization method and demonstrate its practical applicability across diverse machine
    
[^25]: 一种用于密度估计的数据驱动傅里叶混合神经网络方法

    A data-driven Fourier-mixture neural-network method for density estimation

    [https://arxiv.org/abs/2605.18019](https://arxiv.org/abs/2605.18019)

    该论文提出一种在傅里叶空间中直接训练、具有闭式特征函数的正定高斯-拉普拉斯混合神经网络密度估计方法，并针对独立同分布采样与基于重采样的伪采样两种情形分别给出了可分离傅里叶截断、训练、离散化与采样误差的期望 L₂ 密度误差理论界。

    

    我们提出了一种数据驱动的傅里叶训练神经网络方法，用于从经验特征函数（CF）信息中估计固定期限的概率密度。该估计器是一个具有闭式特征函数的正定高斯-拉普拉斯混合模型，因此可以直接在傅里叶空间中进行训练，同时保持非负性和单位质量。我们考虑了两种采样设置。在直接独立同分布（i.i.d.）采样设置中，该方法针对由独立同分布样本构建的经验特征函数进行训练。在基于重采样的伪采样设置中，该方法针对由相关数据通过重采样构建的经验伪特征函数进行训练。对于直接独立同分布情形，我们推导了期望平方 L₂ 密度误差界，该界将傅里叶截断误差、经验训练误差、离散化误差和特征函数采样误差分离开来，并给出了期望 L₂ 范数的相应误差界。对于伪采样情形，我们得到了条件……

    arXiv:2605.18019v2 Announce Type: replace-cross  Abstract: We propose a data-driven Fourier-trained neural-network method for estimating fixed-horizon probability densities from empirical characteristic-function (CF) information. The estimator is a positive Gaussian--Laplace mixture with closed-form CF, so training can be performed directly in Fourier space while preserving nonnegativity and unit mass. We consider two sampling settings. In the direct i.i.d. sampling setting, the method is trained against an empirical CF constructed from i.i.d. samples. In the resampling-based pseudo-sampling setting, it is trained against an empirical pseudo-CF constructed from dependent data by resampling. For the direct i.i.d. case, we derive an expected squared $L_2$ density-error bound that separates Fourier truncation, empirical training error, discretization, and CF sampling error, together with a corresponding bound for the expected $L_2$ norm. For the pseudo-sampling case, we obtain conditional
    
[^26]: 适应还是遗忘：非平稳优化中Adam与SGD之间的可证明权衡

    Adapt or Forget: Provable Tradeoffs Between Adam and SGD in Nonstationary Optimization

    [https://arxiv.org/abs/2605.04269](https://arxiv.org/abs/2605.04269)

    本文首次对Adam在非平稳随机优化中的表现给出理论分析，将有限时间界清晰分解为初始化、目标漂移、一阶矩跟踪误差和预条件器扰动四个分量，并揭示了噪声与漂移之间可证明的权衡关系。

    

    我们在非平稳随机目标下对Adam进行了理论分析，区分了两种情形：一是在Adam预条件化平均梯度算子具有自适应强单调性下的欧氏跟踪情形，二是在一般 $L$-光滑目标下的高概率投影平稳性保证。在跟踪情形中，我们推导了有限时间的期望界和高概率界，这些界可以清晰地分解为四个分量：初始化、目标漂移、由 $\beta_1$ 控制的一阶矩跟踪误差，以及由 $\beta_2$ 控制的预条件器扰动。我们刻画了在常数步长和步长衰减调度下，瞬态项衰减到渐近跟踪界所需的预热时间。我们还证明了Adam在分布偏移下平均投影平稳性间隙的高概率界。在两种分析中，我们的界均揭示了一种噪声—漂移权衡：在噪声主导的情形中，一阶矩……（原文摘要在此处截断）

    arXiv:2605.04269v2 Announce Type: replace-cross  Abstract: We provide a theoretical analysis of Adam under non-stationary stochastic objectives, separating two regimes: Euclidean tracking under adaptive strong monotonicity of the Adam-preconditioned mean-gradient operator, and high-probability projected stationarity guarantees under general $L$-smooth objectives. In the tracking regime, we derive finite-time expected and high-probability bounds that decompose sharply into four components: initialization, objective drift, a first-moment tracking error governed by $\beta_1$, and a preconditioner perturbation governed by $\beta_2$. We characterize the burn-in time required for the transient terms to decay to the asymptotic tracking bound under constant and step-decay schedules. We also prove a high-probability bound on the average projected stationarity gap for Adam under distribution shift. Across both analyses, our bounds reveal a noise--drift tradeoff: in noise-dominated regimes, first
    
[^27]: 面向不平衡学习的偏差校正数据合成方法

    Bias-Corrected Data Synthesis for Imbalanced Learning

    [https://arxiv.org/abs/2510.26046](https://arxiv.org/abs/2510.26046)

    该论文提出了一种偏差校正的数据合成方法，通过从留出的多数类数据估计生成器引起的损失偏差并将其转移至少数类，为不平衡学习建立了有限样本理论保证，同时刻画了SMOTE产生显著损失偏差的条件。

    

    类别不平衡使概率分类变得复杂，因为标准训练目标侧重于多数类的性能。合成过采样可以缓解不平衡问题，但合成分布与目标少数类分布之间的差异可能会使拟合的分类器产生偏差，尤其是当合成样本依赖于观测数据时。我们提出了一种偏差校正程序，该程序从留出的多数类观测子集中估计由生成器引起的损失差异，并在统一的偏差转移条件下将该校正转移到少数类。我们为偏差转移以及所得经验风险最小化器的超额平衡风险建立了有限样本界，并刻画了SMOTE会产生不可忽略损失偏差的情形。该框架还可以应用于不平衡多任务学习和倾向得分估计，相关细节见补充材料。

    arXiv:2510.26046v3 Announce Type: replace-cross  Abstract: Class imbalance complicates probabilistic classification because standard training objectives emphasize majority-class performance. Synthetic oversampling can reduce imbalance, but discrepancies between the synthetic and target minority distributions may bias the fitted classifier, especially because synthetic samples depend on the observed data. We propose a bias-correction procedure that estimates the generator-induced loss discrepancy from a held-out subset of majority observations and transfers this correction to the minority class under a uniform bias-transfer condition. We establish finite-sample bounds for bias transfer and for the excess balanced risk of the resulting empirical risk minimizer, and characterize a regime in which SMOTE induces non-negligible loss bias. The framework can also be implemented in imbalanced multi-task learning and propensity-score estimation, with details provided in the Supplementary Materia
    
[^28]: 一个能准确描述化学键断裂的波函数从头算基础模型

    An ab initio foundation model of wavefunctions that accurately describes chemical bond breaking

    [https://arxiv.org/abs/2506.19960](https://arxiv.org/abs/2506.19960)

    该研究提出了Orbformer——一个在22,000个平衡与解离结构上预训练的可迁移波函数基础模型，只需在未见分子上微调即可实现与经典多参考方法相媲美的精度-成本比，从而准确描述化学键断裂这一量子化学难题。

    

    可靠地描述化学键断裂仍然是量子化学中的一项重大挑战，其难点在于解离物种中电子结构的多参考特性。多参考方法尤其受到巨大计算成本的困扰，而且在通常的范式下，每个体系都需要重新支付全额计算代价，完全忽略了不同分子间电子结构的共性。结合深度神经网络的量子蒙特卡洛方法独特地提供了通过预训练可迁移波函数模型来利用这些共性的可能，但迄今为止所有此类尝试的范围都很有限。在本工作中，我们通过Orbformer将这一范式变为现实——Orbformer是一个在22,000个平衡态和解离结构上预训练的可迁移波函数模型，可以在未见过的分子上进行微调，达到与经典多参考方法相媲美的精度-成本比。在既有基准测试以及更具挑战性的键解离任务上……

    arXiv:2506.19960v2 Announce Type: replace-cross  Abstract: Reliable description of bond breaking remains a major challenge for quantum chemistry due to the multireference character of the electronic structure in dissociating species. Multireference methods in particular suffer from large computational cost, which under the normal paradigm has to be paid anew for each system at a full price, ignoring commonalities in electronic structure across molecules. Quantum Monte Carlo with deep neural networks uniquely offers to exploit such commonalities by pretraining transferable wavefunction models, but all such attempts were so far limited in scope. Here, we bring this paradigm to fruition with Orbformer, a transferable wavefunction model pretrained on 22,000 equilibrium and dissociating structures that can be fine-tuned on unseen molecules reaching an accuracy-cost ratio rivalling classical multireference methods. On established benchmarks as well as more challenging bond dissociations and 
    
[^29]: 基于观测数据对齐语言模型：因果视角下的机遇与风险

    Aligning Language Models with Observational Data: Opportunities and Risks from a Causal Perspective

    [https://arxiv.org/abs/2506.00152](https://arxiv.org/abs/2506.00152)

    本文从因果视角系统分析了利用历史观测数据微调大语言模型的机遇与风险，指出观测结果可作为A/B测试的低成本替代监督信号，但同时需警惕因果混淆带来的偏差。

    

    大语言模型正被广泛应用于各行各业，生成直接贡献于关键性能指标的文本，例如患者消息传递中的用药依从性和内容生成中的转化率。然而，预训练模型在对齐人类偏好或优化业务目标方面往往表现不足。因此，使用高质量的标注数据进行微调对于引导模型生成更有效的内容至关重要。受控实验（如A/B测试）可以提供此类数据，但其成本高昂，且伴随着重大的工程、后勤和伦理挑战。与此同时，企业拥有大量尚未充分利用的历史（观测）数据。在这项工作中，我们研究了使用观测数据微调大语言模型的挑战与机遇。我们表明，虽然观测结果可以提供有价值的监督信号（摘要在此处截断）。

    arXiv:2506.00152v2 Announce Type: replace  Abstract: Large language models are being widely used across industries to generate text that contributes directly to key performance metrics, such as medication adherence in patient messaging and conversion rates in content generation. Pretrained models, however, often fall short when it comes to aligning with human preferences or optimizing for business objectives. As a result, fine-tuning with good-quality labeled data is essential to guide models to generate content that achieves better results. Controlled experiments, like A/B tests, can provide such data, but they are often expensive and come with significant engineering, logistical, and ethical challenges. Meanwhile, companies have access to a vast amount of historical (observational) data that remains underutilized. In this work, we study the challenges and opportunities of fine-tuning LLMs using observational data. We show that while observational outcomes can provide valuable supervi
    
[^30]: 针对具有交叉随机效应的广义混合效应模型的可扩展Krylov子空间方法

    Scalable Krylov Subspace Methods for Generalized Mixed-Effects Models with Crossed Random Effects

    [https://arxiv.org/abs/2505.09552](https://arxiv.org/abs/2505.09552)

    该论文提出了基于Krylov子空间的新方法，解决了广义混合效应模型中高维交叉随机效应导致的计算瓶颈，在保持同等精度的同时实现了几个数量级的加速。

    

    混合效应模型被广泛用于建模具有复杂分组结构和高基数分类预测变量的数据。然而，对于高维交叉随机效应，目前依赖Cholesky分解的标准计算方法可能变得极其缓慢。在这项工作中，我们提出了基于Krylov子空间的方法来解决现有的计算瓶颈，并从理论和实证两方面对其进行了分析。特别地，我们推导了预条件随机Lanczos求积法和共轭梯度法在混合效应模型中收敛性和准确性的新结果，并开发了用于计算预测方差的可扩展方法。在模拟数据和真实数据的实验中，所提出的方法实现了几个数量级的加速，在计算上比基于Cholesky分解的方法更加稳健，同时保持了基本相同的精度。

    arXiv:2505.09552v4 Announce Type: replace-cross  Abstract: Mixed-effects models are widely used to model data with complex grouping structures and high-cardinality categorical predictor variables. However, for high-dimensional crossed random effects, current standard computations relying on Cholesky decompositions can become prohibitively slow. In this work, we present Krylov subspace-based methods that address existing computational bottlenecks, and we analyze them both theoretically and empirically. In particular, we derive new results on the convergence and accuracy of the preconditioned stochastic Lanczos quadrature and conjugate gradient methods for mixed-effects models, and we develop scalable methods for calculating predictive variances. In experiments with simulated and real-world data, the proposed methods yield speedups of several orders of magnitude and are more computationally robust than Cholesky-based computations, while maintaining essentially the same accuracy.
    
[^31]: SGD能否选出好的“渔夫”？自选择偏差下的局部收敛性

    Can SGD Select Good Fishermen? Local Convergence under Self-Selection Biases

    [https://arxiv.org/abs/2504.07133](https://arxiv.org/abs/2504.07133)

    本文提出了首个针对自选择偏差的局部收敛算法，通过将自选择问题归约为粗化估计问题，给出了运行时间为poly(d, k, 1/ε) + (k log k)^{O(k)}的更快算法，从而解决了CDIZ23提出的主要开放问题之一。

    

    我们重新审视了由Cherapanamjeri、Daskalakis、Ilyas和Zampetakis [CDIZ23, STOC'23]提出的在d维空间中使用最大选择准则估计具有自选择偏差的k个线性回归器的问题。我们的主要结果是一个运行时间为poly(d, k, 1/ε) + (k log k)^{O(k)}的算法，该算法改进了Cherapanamjeri、Daskalakis、Ilyas和Zampetakis [CDIZ23]以及Gaitonde和Mossel [GM24, arXiv]所提出算法的运行时间。我们通过提供首个针对自选择的局部收敛算法来实现这一点，从而解决了Cherapanamjeri、Daskalakis、Ilyas和Zampetakis [CDIZ23]提出的主要开放问题之一。为获得该算法，我们将自选择问题归约为一个看似无关的统计问题——粗化下的估计 [FKKT21, COLT'21]。粗化是指人们无法观测到样本的确切值，而只能观测到某个集合的情形。

    arXiv:2504.07133v2 Announce Type: replace-cross  Abstract: We revisit the problem of estimating $k$ linear regressors with self-selection bias in $d$ dimensions with the maximum selection criterion, as introduced by Cherapanamjeri, Daskalakis, Ilyas, and Zampetakis [CDIZ23, STOC'23]. Our main result is a $\mathrm{poly}(d, k, 1/\varepsilon) + (k \log k)^{O(k)}$ time algorithm for this problem that improves upon the running time of the algorithms by Cherapanamjeri, Daskalakis, Ilyas, and Zampetakis [CDIZ23] and Gaitonde and Mossel [GM24, arXiv]. We achieve this by providing the first local convergence algorithm for self-selection, thus resolving one of the main open questions of Cherapanamjeri, Daskalakis, Ilyas, and Zampetakis [CDIZ23].   To obtain this algorithm, we reduce self-selection to a seemingly unrelated statistical problem called estimation under coarsening [FKKT21, COLT'21]. Coarsening occurs when one does not observe the exact value of the sample but only some set (from a pa
    
[^32]: 一种基于广义切近似的强超高斯似然变分推断框架

    A Generalized Tangent Approximation based Variational Inference Framework for Strongly Super-Gaussian Likelihoods

    [https://arxiv.org/abs/2504.05431](https://arxiv.org/abs/2504.05431)

    本文提出了一种基于广义切变换的变分推断框架，利用凸对偶性构建对数似然的切下界，使强超高斯似然类概率模型与高斯先验实现共轭，从而将该结构化变分方法的应用范围从逻辑回归扩展到更广泛的模型类别。

    

    变分推断作为马尔可夫链蒙特卡洛采样的替代方法，在实现复杂贝叶斯模型的可扩展计算方面发挥了变革性作用。然而，现有方法通常依赖于僵化的模型特定公式或随机黑盒优化程序。切近似是一类有原则的结构化变分方法，它利用了底层概率模型的几何特性。然而，其应用在很大程度上局限于逻辑回归及相关建模领域。在本文中，我们针对以强超高斯似然为特征的一类广泛概率模型，提出了一种基于切变换的新型变分框架。我们的方法利用凸对偶性来构建对数似然的切下界，从而在原本难以处理的设置中诱导出与模型参数高斯先验的共轭性。

    arXiv:2504.05431v4 Announce Type: replace-cross  Abstract: Variational inference, as an alternative to Markov chain Monte Carlo sampling, has played a transformative role in enabling scalable computation for complex Bayesian models. Nevertheless, existing approaches often depend on either rigid model-specific formulations or stochastic black-box optimization routines. Tangent approximation is a principled class of structured variational methods that exploits the geometry of the underlying probability model. However, its utility has largely been confined to logistic regression and related modeling regimes. In this article, we propose a novel variational framework based on tangent transformation for a broad class of probability models characterized by strongly super-Gaussian likelihoods. Our method leverages convex duality to construct tangent minorants of the log-likelihood, thereby inducing conjugacy with Gaussian priors over model parameters in an otherwise intractable setup. Under mi
    
[^33]: 机器学习基准测试中聚合性能指标的统计不确定性量化

    Statistical Uncertainty Quantification for Aggregate Performance Metrics in Machine Learning Benchmarks

    [https://arxiv.org/abs/2501.04234](https://arxiv.org/abs/2501.04234)

    该论文展示了如何利用自助法和贝叶斯分层建模等统计方法，来量化机器学习基准测试中跨多个任务聚合的性能指标的不确定性。

    

    现代人工智能由机器学习模型（如基础模型）支撑，这些模型在海量数据语料库上进行预训练，然后被适配以解决各种下游任务。为了总结跨多个任务的性能，评估指标通常被聚合为一个汇总指标，例如跨10个问答任务的平均准确率。在聚合评估指标时，将不确定性纳入聚合指标中是有益的，以便更真实地理解模型性能。我们在这项工作中的目标是展示如何运用统计方法来量化跨多个任务聚合的指标的不确定性。我们重点强调的方法包括自助法（bootstrap）、贝叶斯分层（即多层）建模，以及考虑标准误差的任务权重可视化。这些技术揭示了诸如某种任务占主导地位等洞见

    arXiv:2501.04234v2 Announce Type: replace-cross  Abstract: Modern artificial intelligence is supported by machine learning models (e.g., foundation models) that are pretrained on a massive data corpus and then adapted to solve a variety of downstream tasks. To summarize performance across multiple tasks, evaluation metrics are often aggregated into a summary metric, e.g., average accuracy across 10 question-answering tasks. When aggregating evaluation metrics, it is useful to incorporate uncertainty in the aggregate metric in order to gain a more realistic understanding of model performance. Our objective in this work is to demonstrate how statistical methodology can be used for quantifying uncertainty in metrics that have been aggregated across multiple tasks. The methods we emphasize are bootstrapping, Bayesian hierarchical (i.e., multilevel) modeling, and the visualization of task weightings that consider standard errors. These techniques reveal insights such as the dominance of a s
    
[^34]: 赌博机中的满意遗憾最小化：常数速率与轻尾分布

    Satisficing Regret Minimization in Bandits: Constant Rate and Light-Tailed Distribution

    [https://arxiv.org/abs/2406.06802](https://arxiv.org/abs/2406.06802)

    本文提出SELECT算法模板，通过采样与下置信界检验，在赌博机满意遗憾最小化问题中实现了常数级别的期望满意遗憾，并同时具备标准遗憾保证。

    

    受决策制定中“满意”概念的启发，我们研究了赌博机优化中满意遗憾最小化的问题。在该设定下，学习者的目标是尽可能频繁地选择满意臂（即平均奖励超过某个阈值的臂）。性能通过满意遗憾来衡量，即所选臂的平均奖励相对于阈值的累计不足量。我们提出了SELECT，这是一个通过采样和下置信界检验来实现满意遗憾最小化的通用算法模板，在可实现情形下（即存在满意臂时），该算法能够为多种赌博机优化问题实现常数级别的期望满意遗憾。作为补充，在不可实现情形下，SELECT也能享有与预言机相同的标准遗憾保证。为了进一步提高算法的稳定性，我们引入了SELECT-LITE，它实现了轻尾……

    arXiv:2406.06802v4 Announce Type: replace-cross  Abstract: Motivated by the concept of satisficing in decision-making, we consider the problem of satisficing regret minimization in bandit optimization. In this setting, the learner aims at selecting satisficing arms (arms with mean reward exceeding a certain threshold value) as frequently as possible. The performance is measured by satisficing regret, which is the cumulative deficit of the chosen arm's mean reward compared to the threshold. We propose SELECT, a general algorithmic template for Satisficing REgret Minimization via SampLing and LowEr Confidence bound Testing, that attains constant expected satisficing regret for a wide variety of bandit optimization problems in the realizable case (i.e., a satisficing arm exists). As a complement, SELECT also enjoys the same (standard) regret guarantee as the oracle in the non-realizable case. To further ensure stability of the algorithm, we introduce SELECT-LITE that achieves a light-tail
    

