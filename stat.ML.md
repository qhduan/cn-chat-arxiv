# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved Gradient Descent Lower Bounds Beyond Nesterov](https://arxiv.org/abs/2609.02855) | 本文证明了光滑凸优化中固定步长梯度下降的两个更强下界——非anytime的Ω(n^{-1.6342})与anytime的Ω(n^{-1.2408})，并借助silver调度可达的O(n^{-log_2(1+√2)})速率，严格分离了两种设定下可实现的收敛指数。 |
| [^2] | [Copula Transformations for Data-Consistent Inversion](https://arxiv.org/abs/2609.02832) | 本文利用Sklar定理将DCI更新分解为边际变换与copula相依性变换，证明iDCI收敛后的残余差异完全由观测与预测分布的copula刻画，并据此提出可恢复联合DCI解的copula变换iDCI方法。 |
| [^3] | [Full-Model Optimality for Tunable Linear Generative Priors in Compressed Sensing](https://arxiv.org/abs/2609.02790) | 本文针对压缩感知中通过奇异值分解相互关联的可调线性生成先验族建立了理论，证明在无噪声高斯压缩感知中，全维线性先验在整个先验族中达到最小的期望重建误差。 |
| [^4] | [Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency](https://arxiv.org/abs/2609.02728) | 该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。 |
| [^5] | [Neural operators approximate strongly continuous convex monotone semigroups](https://arxiv.org/abs/2609.02727) | 该论文提出Chernoff神经算子与包络神经算子，通过学习单步算子实现了对强连续凸单调半群的万能逼近并给出定量逼近速率，在非线性偏微分方程、随机最优控制和模型不确定性下的随机过程等数值例子中验证了方法的有效性。 |
| [^6] | [A computational approach to maximum likelihood thresholds for colored Gaussian graphical models](https://arxiv.org/abs/2609.02382) | 本文针对有色高斯图模型，通过几何表述建立统一理论框架并提出新的符号算法，解决了其最大似然阈值的计算问题。 |
| [^7] | [Bayes-Optimal BER and AUC: Estimation and Evaluation of Estimators](https://arxiv.org/abs/2609.02304) | 该论文提出了基于软标签来估计贝叶斯最优平衡错误率（BER）和AUC的新方法，并研究了如何评估这些估计量，从而在类别不平衡等准确率失效的场景中衡量模型性能的理论上限。 |
| [^8] | [From topology learning to graph generation: A unifying perspective](https://arxiv.org/abs/2609.02286) | 本综述提出统一框架，将图拓扑学习与图生成视为同一图数据生成过程的逆问题，从而连接了这两个长期平行发展的研究方向。 |
| [^9] | [Schr\"odinger Bridges on Lie Group Manifolds for Probabilistic Intrinsic Generation](https://arxiv.org/abs/2609.02196) | 该论文将薛定谔桥推广到李群流形上，实现了在弯曲几何空间中直接进行概率生成建模，允许仅约束部分可观测端点变量，并针对紧致阿贝尔群与非阿贝尔群分别提出了WKBC和RCCBM两种计算方法。 |
| [^10] | [Online Non-Monotone DR-Submodular Maximization Matching the Offline $0.401$ Factor](https://arxiv.org/abs/2609.02145) | 该论文首次在对抗性在线设置下实现了非单调DR-次模最大化与非离线算法相同的0.401最优近似比，通过用加权在线学习器替代离线箱约束步骤并结合精确非对称平衡定理，在决策后全信息值预言机模型下达到次线性近似遗憾。 |
| [^11] | [HyperMC: Multi-Fidelity Hyperparameter Tuning for Stochastic Gradient MCMC](https://arxiv.org/abs/2609.02138) | 提出了HyperMC框架，将Hyperband风格的资源分配与核Stein差异评估相结合，为缺乏Metropolis-Hastings接受率的SGMCMC方法实现高效的多保真度超参数调优，并通过全局网格初始化与精英引导局部细化增强了鲁棒性。 |
| [^12] | [What Would it Cost to End Extreme Poverty?](https://arxiv.org/abs/2609.02013) | 本文将直接转移支付扶贫问题框架化为统计学习问题，利用34个国家的家庭消费调查数据估算出将贫困率降至1%每年仅需2110亿美元（约合全球GDP的0.28%），远低于全民基本收入的成本。 |
| [^13] | [Posterior Tempering Explains Variance Inflation in Linear and Generalized Linear Thompson Sampling](https://arxiv.org/abs/2609.01999) | 该论文提出 α-TS 算法，通过用 α-后验（分数幂后验）替代标准后验来形式化方差膨胀思想，并给出了先验与奖励分布的一般正则性条件，使汤普森采样在广义线性老虎机中无需后验近似即可完成遗憾分析，且当 α ∝ d^{-1} 时达到了已知最优的 O(d^{3/2}√T log T) 遗憾界。 |
| [^14] | [Robust Bayesian Inference for Unnormalized Models with Mixed-Domain Data](https://arxiv.org/abs/2609.01783) | 提出SME-BETEL半参数贝叶斯框架，将得分匹配估计方程与贝叶斯指数倾斜经验似然相结合，无需计算归一化常数和学习率校准即可对含混合域数据的非归一化模型进行稳健贝叶斯推断，并通过Bernstein-von Mises定理保证了模型误设下不确定性量化的渐近校准性。 |
| [^15] | [Pooling and Drift in Delayed Bandits](https://arxiv.org/abs/2609.01761) | 该论文发现当延迟老虎机的反馈结果仅通过动作所产生的状态依赖于动作时，学习代价由有效维度（真正不同的状态数量）而非动作数量决定，并据此证明了 $\widetilde{O}(\sqrt{(d+1)V\log K})$ 等新的遗憾界，突破了以往随动作数增长的界限。 |
| [^16] | [Optimal Transport for Network Comparison: A Review with Machine Learning Applications](https://arxiv.org/abs/2608.27500) | 本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。 |
| [^17] | [Diagonal Multi-omics Integration of Heterogenous Datasets](https://arxiv.org/abs/2608.16968) | 本文提出了一种基于极值迹问题和梯度上升方法的新特征，利用最大值与最小值点差的范数来表征数据集异质性，用于异质数据集的对角多组学整合。 |
| [^18] | [Variation Spaces for Encoder--Decoder Neural Operators: Approximation and Generalization](https://arxiv.org/abs/2606.01244) | 该论文基于有界变差向量值测度构建了神经算子的变分空间理论，证明了在ReLU激活下该空间与Schatten-1算子类范数等价，并建立了编码器-解码器神经算子的逼近误差界与高概率泛化界。 |
| [^19] | [Connections between the F\"ollmer process and the denoising diffusion probabilistic model](https://arxiv.org/abs/2605.18040) | 本文阐明了离散化Föllmer过程与DDPM采样器之间的直接联系，证明其为DDPM采样器提供了自然的超参数设置，且能容纳比离散化反向SDE更广泛的方差调度，从而系统地恢复了最先进的DDPM采样误差界结果。 |
| [^20] | [Stabilizing Private LASSO under Heterogeneous Covariates via Anisotropic Objective Perturbation](https://arxiv.org/abs/2605.01492) | 该论文提出一种基于Gram矩阵的各向异性目标扰动“预失真”策略，通过抵消异质协变量结构引起的失真来稳定差分隐私下的高维LASSO估计，显著提升了收敛稳定性、统计效率和隐私性能。 |
| [^21] | [Quantum Maximum Likelihood Prediction via Hilbert Space Embeddings](https://arxiv.org/abs/2602.18364) | 本文通过将经验概率分布嵌入量子态并最小化量子相对熵，提出了一种量子最大似然预测方法，并为其在经典和量子大语言模型中的统一应用提供了非渐近性能保证。 |
| [^22] | [Cantelli Constrained Policy Optimization](https://arxiv.org/abs/2601.22993) | 本文提出风险厌恶方法Canary，利用Cantelli不等式基于成本回报的前两阶矩得到可处理的风险价值约束上界，并扩展CPO信赖域框架提供最坏情况保证，是所有测试环境中唯一能可靠满足风险价值约束的方法。 |
| [^23] | [What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/abs/2512.24497) | 本文将联合嵌入预测世界模型（JEPA-WM）类规划方法进行了系统化表征，通过对若干关键组件的全面研究，找出了在抽象表示空间中进行物理规划取得成功的关键技术选择。 |
| [^24] | [A Multivariate Bernoulli-Based Sampling Method for Multi-Label Data with Application to Meta-Research](https://arxiv.org/abs/2512.08371) | 提出了一种基于多元伯努利分布、考虑标签间依赖性的加权抽样算法，解决了多标签数据中稀有标签难以获得足够样本的问题，并成功应用于元研究领域。 |
| [^25] | [The Ensemble Kalman Inversion Race](https://arxiv.org/abs/2511.15853) | 该论文聚焦气候模型参数校准问题，指出随着混合物理-机器学习气候模型日益复杂，集合卡尔曼方法因无需导数、可扩展至高维且对统计观测噪声鲁棒，成为实现快速迭代、校准驱动的气候模型开发的自然选择。 |
| [^26] | [Multidimensional scaling of two-mode three-way asymmetric dissimilarities: finding archetypal profiles and clustering](https://arxiv.org/abs/2511.15813) | 本文将h-plot方法扩展至三向（含对称与不对称）邻近数据，提出一种基于特征向量解析解的多维尺度分析新方法，能够从三向不对称相异性数据中提取原型轮廓并实现聚类。 |
| [^27] | [Gradient Prediction with Control Variates in the Cheap-Forward Regime](https://arxiv.org/abs/2511.05187) | 该论文提出用降精度、推理风格的程序预测梯度，并通过控制变量将大量预测与少量精确梯度结合，使近似误差转化为方差而非偏差，从而在集群推理资源足够廉价时降低语言模型训练的成本。 |
| [^28] | [Neural Variational Cut Posteriors without Upstream Data](https://arxiv.org/abs/2510.10268) | 提出NeVI-Cut方法，一种无需访问上游数据和模型、仅利用上游后验样本即可模块化且可证明准确地近似切割后验的神经变分推断方法。 |
| [^29] | [DLM-One: Diffusion Language Models for One-Step Sequence Generation](https://arxiv.org/abs/2506.00290) | DLM-One提出了一种基于分数蒸馏的框架，将扩散语言模型的生成过程压缩为单步，实现采样步数约2000倍、推理时间约500倍的加速，同时保持有竞争力的文本生成性能。 |
| [^30] | [Adaptive Replication Strategies in Trust-Region-Based Bayesian Optimization of Stochastic Functions](https://arxiv.org/abs/2504.20527) | 该论文提出了 OGPIT 方法，在信赖域框架下将高斯过程局部建模与自适应重复评估（复制）策略相结合，通过改进采集函数和成本感知评估策略，在目标函数噪声大、需要大量采样的随机优化场景中显著提升计算效率。 |
| [^31] | [Online Multivariate Regularized Distributional Regression for High-dimensional Probabilistic Electricity Price Forecasting](https://arxiv.org/abs/2504.02518) | 本文提出了一种结合在线坐标下降与LASSO正则化的多变量分布回归在线算法，可高效建模日前电价的条件均值、方差与依赖结构，实现高维空间下快速准确且避免过拟合的概率电价预测。 |
| [^32] | [Robust Streaming PCA](https://arxiv.org/abs/1902.03223) | 该论文提出了协方差矩阵属于时变不确定集合的鲁棒流式主成分分析框架，给出了算法收敛的基本极限，并证明噪声幂法在此扰动设定下达到速率最优。 |
| [^33] | [Clustering Three-Way Data with Outliers.](http://arxiv.org/abs/2310.05288) | 这项研究提出了一种用于聚类矩阵形式数据的方法，可以处理其中的异常值。 |
| [^34] | [Generalized Regret Analysis of Thompson Sampling using Fractional Posteriors.](http://arxiv.org/abs/2309.06349) | 这项研究对使用分数后验概率的汤普森抽样算法进行了广义遗憾分析，获得了依赖于实例和实例独立的频率遗憾界。这对多臂赌博问题的解决有重要意义。 |

# 详细

[^1]: 超越Nesterov的改进梯度下降下界

    Improved Gradient Descent Lower Bounds Beyond Nesterov

    [https://arxiv.org/abs/2609.02855](https://arxiv.org/abs/2609.02855)

    本文证明了光滑凸优化中固定步长梯度下降的两个更强下界——非anytime的Ω(n^{-1.6342})与anytime的Ω(n^{-1.2408})，并借助silver调度可达的O(n^{-log_2(1+√2)})速率，严格分离了两种设定下可实现的收敛指数。

    

    我们研究了在光滑凸优化中，梯度下降（GD）通过预先设定的步长能够被加速到何种程度。在超越Nemirovsky和Yudin经典的Ω(n^{-2})一阶oracle下界的基础上，我们证明了Ω(n^{-1.6342})的非anytime下界以及Ω(n^{-1.2408})的anytime下界。这两个结果分别改进了Ma和Chen近期提出的Ω(n^{-1.932})非anytime下界，以及Tsai等人提出的Ω(n^{-4/3}) anytime下界。结合silver步长调度所达到的非anytime O(n^{-log_2(1+√2)})收敛速率，我们的anytime下界在这两种设定下可实现的收敛指数之间建立了严格的分离。

    arXiv:2609.02855v1 Announce Type: cross  Abstract: We study how far gradient descent (GD) can be accelerated by predetermined stepsizes in smooth convex optimization. Going beyond the classical $\Omega(n^{-2})$ first-order oracle lower bound of Nemirovsky and Yudin, we prove an $\Omega(n^{-1.6342})$ non-anytime lower bound and an $\Omega(n^{-1.2408})$ anytime lower bound. These improve the recent $\Omega(n^{-1.932})$ non-anytime lower bound of Ma and Chen and the $\Omega(n^{-4/3})$ anytime lower bound of Tsai et al., respectively. Together with the non-anytime $O(n^{-\log_2(1+\sqrt{2})})$ rate achieved by silver schedules, our anytime lower bound establishes a strict separation between the achievable convergence exponents in the two settings.
    
[^2]: 面向数据一致反演的Copula变换

    Copula Transformations for Data-Consistent Inversion

    [https://arxiv.org/abs/2609.02832](https://arxiv.org/abs/2609.02832)

    本文利用Sklar定理将DCI更新分解为边际变换与copula相依性变换，证明iDCI收敛后的残余差异完全由观测与预测分布的copula刻画，并据此提出可恢复联合DCI解的copula变换iDCI方法。

    

    数据一致反演（DCI）构造其前推分布与观测数据相符的概率测度；迭代数据一致反演（iDCI）则通过依次施加多个前推约束，将这一框架推广到广义随机逆问题。尽管iDCI避免了对高维联合密度的直接近似，但它与原始联合DCI解之间的关系一直不明确。在本工作中，我们借助copula理论建立了这一关系。利用Sklar定理，我们将DCI更新分解为相互独立的边际变换与相依结构变换，并证明iDCI算法收敛后残留的差异完全由与观测及预测联合分布相关联的copula所刻画。这一刻画启发了一种经copula变换的iDCI解，我们进一步证明，一个精确的copula变换能够……（摘要原文在此处截断）

    arXiv:2609.02832v1 Announce Type: new  Abstract: Data-consistent inversion (DCI) constructs probability measures whose push-forward distributions agree with observed data, while iterative data-consistent inversion (iDCI) extends this framework to generalized stochastic inverse problems by enforcing multiple push-forward constraints sequentially. Although iDCI avoids the direct approximation of high-dimensional joint densities, its relationship to the original joint DCI solution has remained unclear. In this work, we establish this relationship through copula theory. Using Sklar's theorem, we derive a factorization of the DCI update into separate marginal and dependence transformations and show that the discrepancy remaining after convergence of the iDCI algorithm is entirely characterized by the copulas associated with the observed and predicted joint distributions. This characterization motivates a copula-transformed iDCI solution, and we prove that an exact copula transformation reco
    
[^3]: 压缩感知中可调线性生成先验的全模型最优性

    Full-Model Optimality for Tunable Linear Generative Priors in Compressed Sensing

    [https://arxiv.org/abs/2609.02790](https://arxiv.org/abs/2609.02790)

    本文针对压缩感知中通过奇异值分解相互关联的可调线性生成先验族建立了理论，证明在无噪声高斯压缩感知中，全维线性先验在整个先验族中达到最小的期望重建误差。

    

    生成模型作为压缩感知等逆问题的先验，已在实验和理论层面得到广泛研究。Gunn 等人最近的工作研究了具有可调复杂度的生成先验的使用方法，即维护一个包含不同复杂度的生成先验族，并在重建阶段选择特定的复杂度。他们证明，通过适当地调整生成先验的复杂度，可以在多种逆问题中实验性地获得更低的重建误差。在本文中，我们针对通过奇异值分解自然关联的可调线性生成先验族的设定，为压缩感知建立了理论。我们证明，在无噪声高斯压缩感知中，全维线性先验在整个线性先验族上达到了最小的期望重建误差。因此，在这种理想化的线性无噪声环境中（摘要在此处截断）……

    arXiv:2609.02790v1 Announce Type: cross  Abstract: Generative models have been studied experimentally and theoretically as priors for inverse problems such as compressed sensing. Recent work by Gunn et al. studied the use of generative priors with tunable complexity, where a family of generative priors with varying complexity is maintained and a specific complexity can be selected at inversion time. They demonstrated that lower reconstruction errors can be experimentally attained for a variety of inverse problems by appropriately tuning the complexity of the generative prior. In the present paper, we establish theory for compressed sensing in the setting of a tunable family of linear generative priors naturally related through their singular value decompositions. We prove that in noiseless Gaussian compressed sensing, the full-dimensional linear prior attains the minimum expected reconstruction error over the entire family of linear priors. Thus, in this idealized linear noiseless sett
    
[^4]: 大批量训练中的动量：Polyak 动量扩大临界批大小，Nesterov 动量提升数据效率

    Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency

    [https://arxiv.org/abs/2609.02728](https://arxiv.org/abs/2609.02728)

    该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。

    

    我们在单遍（one-pass）训练机制下研究动量何时以及如何改善大批量训练，并以幂律核回归作为一个易于解析的设定。我们首先通过临界学习率刻画风险的稳定性，临界学习率定义为保证训练稳定的最大学习率，并得到 $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$、$\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$ 以及 $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$，其中 $B$ 为批大小，$\rho$ 为动量因子，$\beta>1$ 为容量指数。在该允许区域内，我们推导出完整风险动力学的标度律，刻画了训练从早期瞬态阶段、经幂律衰减、直至噪声底限的完整演化过程。随后，在固定数据预算下，我们在允许的学习率与动量因子范围内最小化最后一步风险，得到了一个包含三个区制的批大小相图，揭示了（原文摘要在此处截断）

    arXiv:2609.02728v1 Announce Type: cross  Abstract: We study when and how momentum improves large-batch training in the one-pass regime, using power-law kernel regression as a tractable setting. We first characterize risk stability through the critical learning rate, defined as the largest learning rate for stable training, and obtain $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$, $\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$, and $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$, where $B$ is the batch size, $\rho$ is the momentum factor, and $\beta>1$ is the capacity exponent. Within this admissible region, we derive scaling laws for the full risk dynamics, capturing the progression from an early transient, through power-law decay, to a noise floor. We then minimize the final-step risk over the admissible learning rates and momentum factors under a fixed data budget, yielding a three-regime batch-size phase diagram that reveals how the rol
    
[^5]: 神经算子逼近强连续凸单调半群

    Neural operators approximate strongly continuous convex monotone semigroups

    [https://arxiv.org/abs/2609.02727](https://arxiv.org/abs/2609.02727)

    该论文提出Chernoff神经算子与包络神经算子，通过学习单步算子实现了对强连续凸单调半群的万能逼近并给出定量逼近速率，在非线性偏微分方程、随机最优控制和模型不确定性下的随机过程等数值例子中验证了方法的有效性。

    

    我们通过用神经算子学习其Chernoff型单步算子来逼近强连续凸单调半群。首先，我们引入了所谓的Chernoff神经算子这一一般类，并通过万能逼近定理证明它们可以任意好地逼近Chernoff单步算子。通过利用加权Hölder空间之间的稳定性估计，单步逼近误差可以在迭代过程中传播，从而得到相应半群的万能逼近。其次，我们针对包络半群引入了更专门的包络神经算子类，这使我们能够推导出定量的逼近速率。最后，我们通过多个源自非线性偏微分方程、随机最优控制以及模型不确定性下随机过程的数值例子，展示了这些神经算子的有效性。

    arXiv:2609.02727v1 Announce Type: cross  Abstract: We approximate strongly continuous convex monotone semigroups by learning their Chernoff-type one-step operators with neural operators. First, we introduce the general class of so-called Chernoff-neural operators and show in a universal approximation theorem that they can approximate the Chernoff one-step operators arbitrarily well. By using stability estimates between weighted H\"older spaces, the one-step approximation error can be propagated through the iterations which yields universal approximation of the corresponding semigroup. Second, we introduce the more specialized class of envelope-neural operators for envelope semigroups which allows us to derive quantitative approximation rates. Finally, we illustrate the effectiveness of these neural operators in several numerical examples arising from non-linear partial differential equations, stochastic optimal control and stochastic processes under model uncertainty.
    
[^6]: 有色高斯图模型最大似然阈值的计算方法

    A computational approach to maximum likelihood thresholds for colored Gaussian graphical models

    [https://arxiv.org/abs/2609.02382](https://arxiv.org/abs/2609.02382)

    本文针对有色高斯图模型，通过几何表述建立统一理论框架并提出新的符号算法，解决了其最大似然阈值的计算问题。

    

    高斯图模型（GGMs）是可解释结构学习的重要工具。然而，在高维小样本的情形下，现有数据往往不足以保证最大似然估计量的存在。有色高斯图模型（CGGMs）通过图着色施加对称性约束来缓解这一限制，从而降低了所需的样本量。保证估计量几乎必然存在所需的最小观测数被定义为最大似然阈值（MLT）。本文通过关注MLT的几何表述来解决CGGMs的MLT计算问题：即求样本协方差矩阵的最小秩，使得其投影几乎必然位于充分统计量锥的内部。我们建立了一个统一的理论框架，将已有结果从无色模型推广到有色模型，并提出了新的符号算法。此外……

    arXiv:2609.02382v1 Announce Type: cross  Abstract: Gaussian graphical models (GGMs) are essential tools for interpretable structure learning. However, in high-dimensional, small-sample regimes, the available data is often insufficient for the maximum likelihood estimator to exist. Colored Gaussian graphical models (CGGMs) mitigate this limitation by imposing symmetry constraints through graph coloring, which reduces the required sample size. This minimal number of observations needed to guarantee that the estimator exists almost surely is defined as the maximum likelihood threshold (MLT). Here, we address the computation of the MLT for CGGMs by focusing on its geometric formulation: finding the minimum rank of a sample covariance matrix such that its projection lies almost surely within the interior of the cone of sufficient statistics. We establish a unified theoretical framework, extending results from uncolored to colored models and introducing new symbolic algorithms. Furthermore, 
    
[^7]: 贝叶斯最优BER与AUC：估计量的估计与评估

    Bayes-Optimal BER and AUC: Estimation and Evaluation of Estimators

    [https://arxiv.org/abs/2609.02304](https://arxiv.org/abs/2609.02304)

    该论文提出了基于软标签来估计贝叶斯最优平衡错误率（BER）和AUC的新方法，并研究了如何评估这些估计量，从而在类别不平衡等准确率失效的场景中衡量模型性能的理论上限。

    

    机器学习中的一个基本量是任何模型在给定任务上可达到的最优性能。估计这一量可以使我们将不可消除的误差部分与模型自身的缺陷区分开来，从而告诉我们还剩多大的改进空间。最近的研究表明，在二分类任务中，贝叶斯误差（或等价地，最优准确率）可以从软标签中估计出来。然而，在类别严重不平衡或标注存在噪声的场景下，准确率往往不能很好地概括模型性能，此时平衡错误率（BER）和ROC曲线下面积（AUC）等指标更为合适。我们通过两项互补的贡献来填补这一空白：（i）估计方面，我们提出了基于软标签的最优BER和AUC估计量；我们首先考虑真实软标签和类别先验均已知的干净设定，随后将估计量扩展到更贴近现实的设定中（摘要截断）。

    arXiv:2609.02304v1 Announce Type: new  Abstract: A fundamental quantity in machine learning is the optimal performance achievable by any model on a given task. Estimating this quantity allows us to distinguish the irreducible part of the error from a deficiency of the model, telling us how much room for improvement remains. Recent work has shown that the Bayes error, or equivalently the optimal accuracy, can be estimated from soft labels in binary classification. However, accuracy is often a poor summary of performance in settings with severe class imbalance or noisy annotations, where metrics such as the balanced error rate (BER) and the area under the ROC curve (AUC) are more appropriate. We address this gap with two complementary contributions. (i) Estimation. We propose soft-label-based estimators for the optimal BER and AUC. We first consider the clean setting in which true soft labels and the class prior are known, and then extend the estimators to a more realistic setting in whi
    
[^8]: 从拓扑学习到图生成：一个统一的视角

    From topology learning to graph generation: A unifying perspective

    [https://arxiv.org/abs/2609.02286](https://arxiv.org/abs/2609.02286)

    本综述提出统一框架，将图拓扑学习与图生成视为同一图数据生成过程的逆问题，从而连接了这两个长期平行发展的研究方向。

    

    从数据中学习图结构是一个基础性问题，涵盖广泛的信号处理和机器学习任务。尽管针对这一问题已有大量研究，但现有工作主要沿着两个平行的方向发展：第一个方向试图从支撑于图上的观测数据中推断单个图的拓扑结构，而第二个方向试图从观测到的图实例中学习生成分布，从而实现对新图的采样。本综述提出了一个统一的框架，通过将这两种建模方式视为图数据共同生成过程的逆问题来连接它们。我们回顾了该框架下的主要方法论，强调了它们之间的关系、优势和局限性，并指出了跨范式整合思想的机会。通过架起图拓扑学习与图生成之间的桥梁，本综述提供了更广泛的跨学科视角。

    arXiv:2609.02286v1 Announce Type: cross  Abstract: Learning graph structures from data is a fundamental problem that spans a wide range of signal processing and machine learning tasks. While significant effort has been made to tackle the problem, existing research has largely evolved along two parallel directions. The first seeks to infer the topology of an individual graph from observations supported on it, whereas the second seeks to learn a generative distribution from observed graph instances, enabling the sampling of new graphs. This review presents a unified framework that connects these formulations by viewing them as inverse problems of a common generation process for graph data. We review the major methodologies within this framework, highlight their relationships, strengths, and limitations, and identify opportunities for integrating ideas across paradigms. By bridging graph topology learning and graph generation, this review provides a broader cross-disciplinary perspective 
    
[^9]: 李群流形上的薛定谔桥用于概率性内在生成

    Schr\"odinger Bridges on Lie Group Manifolds for Probabilistic Intrinsic Generation

    [https://arxiv.org/abs/2609.02196](https://arxiv.org/abs/2609.02196)

    该论文将薛定谔桥推广到李群流形上，实现了在弯曲几何空间中直接进行概率生成建模，允许仅约束部分可观测端点变量，并针对紧致阿贝尔群与非阿贝尔群分别提出了WKBC和RCCBM两种计算方法。

    

    直接在几何流形上进行生成建模，可以避免将非欧几里得数据展平、反复向环境空间投影以及欧几里得表示中坐标不一致所带来的误差。薛定谔桥为一个在指定端点分布之间进行熵正则化输运的概率生成框架提供了理论基础。我们研究了李群流形上动力学系统的薛定谔桥问题，其状态为 X_t = (g_t, ξ_t) ∈ G × 𝔤，允许端点观测仅约束那些实际被测量的变量。特别地，熵投影确定了未观测到的端点速度的条件分布律。针对相同的观测端点桥问题，我们发展了两种计算实现方法：缠绕核桥校准使用紧致阿贝尔群上的显式周期化动力学核，而互惠条件控制桥匹配（RCCBM）则处理紧致非阿贝尔群的情形（原文摘要在此处被截断）。

    arXiv:2609.02196v1 Announce Type: cross  Abstract: Generative modeling directly on geometric manifolds can avoid errors introduced by flattening non-Euclidean data, repeated ambient projection, and coordinate inconsistency in Euclidean representations. Schrodinger bridges provide a probabilistic generative framework for entropy-regularized transport between prescribed endpoint distributions. We study Schrodinger bridges for kinetic dynamics on Lie group manifolds with state X_t = (g_t, xi_t) in G x g, allowing endpoint observations to constrain only the variables that are actually measured. In particular, the entropy projection determines the conditional law of the unobserved endpoint velocities.   For the same observed endpoint bridge, we develop two computational realizations: Wrapped-Kernel Bridge Calibration (WKBC) uses an explicit periodized kinetic kernel on compact Abelian groups, whereas Reciprocal Conditional-Control Bridge Matching (RCCBM) handles compact non-Abelian groups t
    
[^10]: 在线非单调DR-次模最大化：匹配离线0.401近似比

    Online Non-Monotone DR-Submodular Maximization Matching the Offline $0.401$ Factor

    [https://arxiv.org/abs/2609.02145](https://arxiv.org/abs/2609.02145)

    该论文首次在对抗性在线设置下实现了非单调DR-次模最大化与非离线算法相同的0.401最优近似比，通过用加权在线学习器替代离线箱约束步骤并结合精确非对称平衡定理，在决策后全信息值预言机模型下达到次线性近似遗憾。

    

    我们研究在 $d$ 维单位立方体的紧凸下闭子集上，非负、非单调DR-次模函数的在线最大化问题。在相应的元可解性假设下，目前已知最好的构造性离线近似比为 $0.401$，而可比的对抗性在线保证一直停留在 $1/e$。我们证明该 $0.401$ 近似比同样可以在线实现。在决策后全信息值预言机模型中，当预言机反馈条件无偏且有界时，我们的算法以次线性近似遗憾达到了 $0.401$ 的近似比。该在线算法并不在变化的目标函数上运行离线构造，而是用加权在线学习器替代离线算法中依赖于目标的箱约束步骤，以累积方式控制所需的残差项。一个精确的非对称平衡定理使得离线系数在对抗性变化下仍得以保持。直接实现（原文摘要在此处截断）……

    arXiv:2609.02145v1 Announce Type: cross  Abstract: We study online maximization of nonnegative, non-monotone DR-submodular functions over compact convex down-closed subsets of the $d$-dimensional unit cube. The best known constructive offline approximation factor is $0.401$ under the corresponding meta-solvability assumptions, whereas comparable adversarial online guarantees had remained at $1/e$. We show that this factor is also achievable online. In the post-decision full-information value-oracle model, our algorithm attains factor $0.401$ with sublinear approximate regret when oracle feedback is conditionally unbiased and bounded.   The online algorithm does not run the offline construction on a changing objective. Instead, it replaces the offline objective-dependent box step by a weighted online learner that controls the required residual terms cumulatively. An exact asymmetric balance theorem preserves the offline coefficients despite adversarial variation. The direct implementati
    
[^11]: HyperMC：面向随机梯度MCMC的多保真度超参数调优方法

    HyperMC: Multi-Fidelity Hyperparameter Tuning for Stochastic Gradient MCMC

    [https://arxiv.org/abs/2609.02138](https://arxiv.org/abs/2609.02138)

    提出了HyperMC框架，将Hyperband风格的资源分配与核Stein差异评估相结合，为缺乏Metropolis-Hastings接受率的SGMCMC方法实现高效的多保真度超参数调优，并通过全局网格初始化与精英引导局部细化增强了鲁棒性。

    

    随机梯度马尔可夫链蒙特卡罗（SGMCMC）方法能够实现可扩展的贝叶斯推断，但其性能强烈依赖于步长、小批量大小以及leapfrog步数等超参数。由于大多数SGMCMC算法缺乏Metropolis-Hastings接受率，标准的基于接受率的调优方法无法直接适用。我们提出了HyperMC，一个将Hyperband风格的资源分配与核Stein差异（KSD）评估相结合的多保真度调优框架。通过运行多个连续减半调度区间，HyperMC在固定计算预算下，平衡了对连续超参数空间的广泛探索与对有前景配置的日益精确的评估。我们进一步提出了Robust HyperMC，它采用全局网格初始化 followed by 精英引导的局部细化策略，以降低对随机候选生成和含噪声的有限预算评估的敏感性。

    arXiv:2609.02138v1 Announce Type: cross  Abstract: Stochastic gradient Markov chain Monte Carlo (SGMCMC) methods enable scalable Bayesian inference, but their performance depends strongly on hyperparameters such as the step size, mini-batch size, and number of leapfrog steps. Since most SGMCMC algorithms lack a Metropolis-Hastings acceptance rate, standard acceptance-based tuning methods are not directly applicable. We propose HyperMC, a multi-fidelity tuning framework that combines Hyperband-style resource allocation with kernel Stein discrepancy (KSD) evaluation. By running multiple successive-halving brackets, HyperMC balances broad exploration of a continuous hyperparameter space with increasingly accurate evaluation of promising configurations under a fixed computational budget. We further introduce Robust HyperMC, which uses global grid initialization followed by elite-guided local refinement to reduce sensitivity to random candidate generation and noisy finite-budget evaluations
    
[^12]: 终结极端贫困需要多少成本？

    What Would it Cost to End Extreme Poverty?

    [https://arxiv.org/abs/2609.02013](https://arxiv.org/abs/2609.02013)

    本文将直接转移支付扶贫问题框架化为统计学习问题，利用34个国家的家庭消费调查数据估算出将贫困率降至1%每年仅需2110亿美元（约合全球GDP的0.28%），远低于全民基本收入的成本。

    

    我们研究通过直接转移支付实现贫困最小化的问题，将其框架化为一个统计学习问题，同时保留了现实世界扶贫项目所面临的信息约束。利用来自34个国家（这些国家合计占世界贫困人口的76%）的全国代表性家庭消费调查数据，我们估计将贫困率从13%的基线降至1%，每年需要2110亿美元的名义支出。这一成本是相应贫困缺口总减少量的4.0倍，但仅为全民基本收入成本的19%。外推到全球范围，结果表明基本终结极端贫困的成本约为全球GDP的0.28%。

    arXiv:2609.02013v1 Announce Type: new  Abstract: We study poverty minimization via direct transfers, framing this as a statistical learning problem while retaining the information constraints faced by real-world programs. Using nationally representative household consumption surveys from 34 countries that together account for 76% of the world's poor, we estimate that reducing the poverty rate to 1% (from a baseline of 13%) would cost $211 B nominal per year. This is 4.0 times the corresponding reduction in the aggregate poverty gap, but only 19% of the cost of universal basic income. Extrapolated globally, the results imply a cost of 0.28% of global GDP to (approximately) end extreme poverty.
    
[^13]: 后验温度化解释了线性与广义线性汤普森采样中的方差膨胀

    Posterior Tempering Explains Variance Inflation in Linear and Generalized Linear Thompson Sampling

    [https://arxiv.org/abs/2609.01999](https://arxiv.org/abs/2609.01999)

    该论文提出 α-TS 算法，通过用 α-后验（分数幂后验）替代标准后验来形式化方差膨胀思想，并给出了先验与奖励分布的一般正则性条件，使汤普森采样在广义线性老虎机中无需后验近似即可完成遗憾分析，且当 α ∝ d^{-1} 时达到了已知最优的 O(d^{3/2}√T log T) 遗憾界。

    

    我们研究了一种汤普森采样（TS）算法的变体，称为 α-TS，用于解决随机广义线性老虎机问题。现有的 TS 分析方法需要膨胀后验方差才能推导出接近最优的遗憾界保证。我们通过引入 α-TS 来形式化方差膨胀的思想，该算法使用分数幂后验（α-后验）替代标准后验。我们的主要贡献是识别了关于先验分布和奖励分布的一般正则性条件，使得能够在不假设后验分布存在任何可处理近似的情况下对 α-TS 进行遗憾分析，这一点不同于以往的工作。对于 α ∝ d^{-1} 的特定选择，我们的一般遗憾界对指数族和次高斯族的奖励分布均给出了已知最优的遗憾界 O(d^{3/2}√T log T)。我们进一步提供了一个依赖于 α 的下界，表明遗憾常数 d（摘要在此处被截断）

    arXiv:2609.01999v1 Announce Type: cross  Abstract: We study a variant of the Thompson Sampling (TS) algorithm, called $\alpha$-TS, for solving stochastic generalized linear bandit problems. Existing analyses of TS require inflating the posterior variance to derive near-optimal regret guarantees. We formalize the idea of variance inflation by introducing $\alpha$-TS that uses a fractional or $\alpha$-posterior instead of the standard posterior. Our main contribution is to identify general regularity conditions on the prior and reward distributions that enable a regret analysis of $\alpha$-TS without assuming any tractable approximation of the posterior distribution, unlike previous works. For a specific choice of $\alpha \propto d^{-1}$, our general regret bound yields the best known regret bound of $O(d^{3/2}\sqrt{T}\log T)$ for both the exponential and sub-Gaussian families of reward distributions. We further provide an $\alpha$-dependent lower bound showing that the regret constant d
    
[^14]: 混合域数据下非归一化模型的稳健贝叶斯推断

    Robust Bayesian Inference for Unnormalized Models with Mixed-Domain Data

    [https://arxiv.org/abs/2609.01783](https://arxiv.org/abs/2609.01783)

    提出SME-BETEL半参数贝叶斯框架，将得分匹配估计方程与贝叶斯指数倾斜经验似然相结合，无需计算归一化常数和学习率校准即可对含混合域数据的非归一化模型进行稳健贝叶斯推断，并通过Bernstein-von Mises定理保证了模型误设下不确定性量化的渐近校准性。

    

    许多统计模型涉及依赖于参数的归一化常数，这些常数在计算上难以处理，给标准贝叶斯推断造成了重大障碍。尽管现有的基于似然的算法通常可以绕过这些常数，但在模型误设的情况下，其不确定性量化可能校准不佳。为了应对这些挑战，我们提出了SME-BETEL，这是一种半参数贝叶斯框架，将得分匹配估计方程与贝叶斯指数倾斜经验似然相结合。所得的后验分布避免了归一化常数的计算，且不需要学习率校准。我们建立了得分匹配估计量的一致性和渐近正态性，并证明了SME-BETEL后验的Bernstein-von Mises定理。这些结果表明，SME-BETEL可信集在渐近意义上与得分匹配估计量的抽样变异性校准一致。

    arXiv:2609.01783v1 Announce Type: cross  Abstract: Many statistical models involve parameter-dependent normalizing constants that are computationally intractable, creating substantial obstacles to standard Bayesian inference. Although existing likelihood-based algorithms can often circumvent these constants, their uncertainty quantification may be poorly calibrated under model misspecification. To address these challenges, we propose SME-BETEL, a semiparametric Bayesian framework that combines score matching estimating equations with Bayesian exponentially tilted empirical likelihood. The resulting posterior avoids evaluation of normalizing constants and does not require learning-rate calibration. We establish consistency and asymptotic normality of the score matching estimator, and prove a Bernstein-von Mises theorem for the SME-BETEL posterior. These results show that SME-BETEL credible sets are asymptotically calibrated to the sampling variability of the score matching estimator, yi
    
[^15]: 延迟老虎机中的池化与漂移

    Pooling and Drift in Delayed Bandits

    [https://arxiv.org/abs/2609.01761](https://arxiv.org/abs/2609.01761)

    该论文发现当延迟老虎机的反馈结果仅通过动作所产生的状态依赖于动作时，学习代价由有效维度（真正不同的状态数量）而非动作数量决定，并据此证明了 $\widetilde{O}(\sqrt{(d+1)V\log K})$ 等新的遗憾界，突破了以往随动作数增长的界限。

    

    系统常常不得不在得知行动是否奏效之前就采取行动：推荐系统几秒内就能观察到点击，而购买则要几天后才能看到。在 $K$ 个动作、延迟 $d$ 轮的设定下，$T$ 轮内已知的最优遗憾率为 $\widetilde{O}(\sqrt{(K+d)T})$，因此可选动作越多，学习代价就越高。但事实未必如此：如果结果仅通过动作所产生的状态依赖于动作，那么一个迟到的结果就能为所有可能产生该观测状态的动作提供信息，此时代价由动作产生的真正不同的状态数量决定，而非动作的数量。我们用介于 $1$ 与状态总数之间的有效维度 $v_t$ 来度量这一概念，并针对任意预先固定的预算证明：对于轮转算法，遗憾率为 $\widetilde{O}(\sqrt{(d+1)V\log K})$；对于实践中常用的单副本算法，遗憾率为 $\widetilde{O}(\sqrt{V^{-}}+\sqrt{dT})$；合并相似状态可以进一步降低这一代价。

    arXiv:2609.01761v1 Announce Type: cross  Abstract: A system often has to act long before it learns whether the act worked: a recommender sees a click in seconds and a purchase in days. With $K$ actions and a delay of $d$ rounds, the best rate known for this setting is $\widetilde{O}(\sqrt{(K+d)T})$ over $T$ rounds, so a longer menu is always more expensive to learn from. It need not be: if the outcome depends on the action only through the state it produced, then one late outcome informs every action that could have produced the observed state, and the price is set by how many genuinely different states the actions produce rather than by how many actions there are. We measure this using an effective dimension $v_t$ between $1$ and the number of states, and prove $\widetilde{O}(\sqrt{(d+1)V\log K})$ for a rotating algorithm and $\widetilde{O}(\sqrt{V^{-}}+\sqrt{dT})$ for the single-copy algorithm used in practice, for any budget fixed in advance; merging similar states lowers the price 
    
[^16]: 用于网络比较的最优传输：综述及其机器学习应用

    Optimal Transport for Network Comparison: A Review with Machine Learning Applications

    [https://arxiv.org/abs/2608.27500](https://arxiv.org/abs/2608.27500)

    本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。

    

    运用最优传输进行网络比较是网络科学中一个不断发展的研究领域。与标准的图度量不同，最优传输不仅计算网络间的相异性，还提供一个传输方案来解释一张图如何演变为另一张图。本文综述了如何利用三种主要距离——Wasserstein距离、Gromov-Wasserstein距离和Bures-Wasserstein距离——来比较无向无权图。我们考察了通过节点特征概率分布在一维情形下Wasserstein距离的闭式解，并展示了Wasserstein距离和Gromov-Wasserstein距离的传输方案如何捕捉图扰动后具体哪些节点影响了距离。对于Bures-Wasserstein距离，我们利用拉普拉斯谱推导出上界，从而避免了完整的谱分解。最后，我们使用合成网络数据集评估这些距离在聚类任务中的表现，并应用于真实世界的时间序列网络数据。

    arXiv:2608.27500v1 Announce Type: cross  Abstract: Network comparison using optimal transport is a growing area of research in network science. Unlike standard graph metrics, optimal transport computes both network dissimilarity and a transport plan that explains how one graph morphs into another. In this paper, we review how optimal transport compares undirected, unweighted graphs using three primary distances: the Wasserstein, Gromov-Wasserstein, and Bures-Wasserstein distances. We examine the closed form of the Wasserstein distance in one dimension via node feature probability distributions, and show how the transport plans of the Wasserstein and Gromov-Wasserstein distances capture which specific nodes influence the distance after graph perturbation. For the Bures-Wasserstein distance, we derive bounds using Laplacian spectra to bypass full spectral decompositions. Finally, we evaluate these distances using a synthetic network dataset for clustering and a real-world time series net
    
[^17]: 异质数据集的对角多组学整合

    Diagonal Multi-omics Integration of Heterogenous Datasets

    [https://arxiv.org/abs/2608.16968](https://arxiv.org/abs/2608.16968)

    本文提出了一种基于极值迹问题和梯度上升方法的新特征，利用最大值与最小值点差的范数来表征数据集异质性，用于异质数据集的对角多组学整合。

    

    本文考虑了异质数据集的对角多组学整合方法。我们分析并发展了多种处理生物异质性本质的方法，以更清晰地理解所产生的差异。具体而言，研究了嵌入复欧几里得空间中与Stiefel流形同胚的集合上耦合拉普拉斯算子的极值迹问题。最大化问题的梯度上升方法以泛函分析的经典术语进行了详细阐述，这本身具有重要研究意义。在此基础上，我们通过采用最大值与最小值点之间差的范数，引入了数据集异质性的一个新特征。

    arXiv:2608.16968v1 Announce Type: cross  Abstract: In this paper, we consider methods for the diagonal multi-omics integration of heterogeneous datasets. Several approaches to the nature of biological heterogeneity are analyzed and developed to comprehend more clearly the generated differences. Specifically, the extremal trace problems for the coupled Laplacian on sets homeomorphic to the Stiefel manifold embedded in the complex Euclidean space are investigated. The gradient ascent method for the maximization problem is elaborated in the classical terms of functional analysis, which is of significant interest in itself. On this basis, we introduce a novel characteristic of dataset heterogeneity by employing the norm of the difference between the maximum and minimum points.
    
[^18]: 编码器-解码器神经算子的变分空间：逼近与泛化

    Variation Spaces for Encoder--Decoder Neural Operators: Approximation and Generalization

    [https://arxiv.org/abs/2606.01244](https://arxiv.org/abs/2606.01244)

    该论文基于有界变差向量值测度构建了神经算子的变分空间理论，证明了在ReLU激活下该空间与Schatten-1算子类范数等价，并建立了编码器-解码器神经算子的逼近误差界与高概率泛化界。

    

    受神经网络函数空间理论的启发，我们构建并分析了希尔伯特空间之间非线性算子的一个变分空间，该空间通过具有有界变差的向量值Borel测度来定义。我们将该空间的单位球刻画为Bochner空间中向量值单神经元字典的闭凸包。对于ReLU激活函数，该空间中的有界线性算子恰好是Schatten-1算子，且两者的范数等价。对于该空间中的算子，我们在Bochner $L^q$范数下建立了编码器-解码器逼近界，其中误差分解为输入和输出编码误差以及一个阶为 $N^{-1/2}$ 的有限宽度项。在输入和噪声满足次高斯假设的条件下，我们进一步为路径范数约束的编码器-解码器网络上的经验最小二乘推导了高概率泛化界；有限样本对平方预测误差的贡献……

    arXiv:2606.01244v2 Announce Type: replace-cross  Abstract: Inspired by the function-space theory of neural networks, we formulate and analyze a variation space for nonlinear operators between Hilbert spaces, defined through vector-valued Borel measures of bounded variation. We characterize its unit ball as the closed convex hull of a vector-valued single-neuron dictionary in Bochner spaces. For the ReLU activation, the bounded linear operators in this space are precisely the Schatten-$1$ operators, with equivalent norms. For operators in this space, we establish encoder--decoder approximation bounds in the Bochner $L^q$-norm, where the error decomposes into input and output encoding errors and a finite-width term of order $N^{-1/2}$. Under sub-Gaussian assumptions on the input and noise, we further derive high-probability generalization bounds for empirical least squares over path-norm-constrained encoder--decoder networks; the finite-sample contribution to the squared prediction error
    
[^19]: Föllmer过程与去噪扩散概率模型之间的联系

    Connections between the F\"ollmer process and the denoising diffusion probabilistic model

    [https://arxiv.org/abs/2605.18040](https://arxiv.org/abs/2605.18040)

    本文阐明了离散化Föllmer过程与DDPM采样器之间的直接联系，证明其为DDPM采样器提供了自然的超参数设置，且能容纳比离散化反向SDE更广泛的方差调度，从而系统地恢复了最先进的DDPM采样误差界结果。

    

    Föllmer过程是一个在时刻1被条件化为具有预先指定分布的布朗运动。该过程可以被解释为对应于去噪扩散概率模型（DDPM）的反向随机微分方程（SDE）的一个“增广”时间压缩版本。虽然这一事实已被间接用于通过反向SDE的离散化来分析DDPM的采样误差，但Föllmer过程的直接离散化与DDPM采样器之间的联系尚未得到充分探索。本文在综述文献中相关结果的同时阐明了这一点。我们证明，离散化的Föllmer过程为DDPM采样器提供了自然的超参数设置，同时比离散化的反向SDE能够容纳更广泛一类的方差调度。此外，这使我们能够系统地恢复关于DDPM采样误差界的最先进结果，并得到略微……

    arXiv:2605.18040v2 Announce Type: replace-cross  Abstract: The F\"ollmer process is a Brownian motion conditioned to have a pre-specified distribution at time 1. This process can be interpreted as an ``augmented'' time-compressed version of the reverse stochastic differential equation (SDE) corresponding to the denoising diffusion probabilistic model (DDPM). While this fact has been indirectly used to analyze DDPM sampling errors via discretization of the reverse SDE, the connection between direct discretization of the F\"ollmer process and the DDPM sampler has not yet been fully explored. This paper clarifies this point while surveying relevant results from the literature. We show that discretized F\"ollmer processes give natural hyper-parameter settings of the DDPM sampler while accommodating a broader class of variance schedules than discretized reverse SDEs. Moreover, this allows us to systematically recover state-of-the-art results on DDPM sampling error bounds, along with slight 
    
[^20]: 通过各向异性目标扰动稳定异质协变量下的隐私保护LASSO

    Stabilizing Private LASSO under Heterogeneous Covariates via Anisotropic Objective Perturbation

    [https://arxiv.org/abs/2605.01492](https://arxiv.org/abs/2605.01492)

    该论文提出一种基于Gram矩阵的各向异性目标扰动“预失真”策略，通过抵消异质协变量结构引起的失真来稳定差分隐私下的高维LASSO估计，显著提升了收敛稳定性、统计效率和隐私性能。

    

    我们研究了在差分隐私下，针对具有异质协变量尺度的高维LASSO问题采用目标扰动方法。在实际场景中，协变量通常呈现不同的尺度；然而，在隐私约束下，标准预处理是有问题的，因为它会消耗额外的隐私预算。这种异质性通过协变量的逆Gram矩阵在目标扰动中引入了有效的各向异性，这会降低算法的稳定性和准确性。为解决这一问题，我们提出了一种基于Gram矩阵的各向异性目标扰动方法，这是一种“预失真”策略，通过抵消协变量结构带来的失真来恢复估计过程中的各向同性。利用近似消息传递（AMP）框架和状态演化分析，我们证明了与现有方法相比，我们提出的扰动方法显著稳定了收敛性，并提升了统计效率和隐私性能。

    arXiv:2605.01492v2 Announce Type: replace-cross  Abstract: We study high-dimensional LASSO under differential privacy via objective perturbation with heterogeneous covariate scales. In practical scenarios, covariates often exhibit diverse scales; however, standard preprocessing is problematic under privacy constraints, as it consumes additional privacy budget. This heterogeneity induces effective anisotropy in the objective perturbation via the inverse Gram matrix of covariates, which can degrade the stability and accuracy of algorithms. To address this, we propose a Gram-based anisotropic objective perturbation, a ``pre-distortion" strategy that counteracts the distortion from the covariate structure to restore isotropy in the estimation process. Using an Approximate Message Passing (AMP) framework and state evolution analysis, we demonstrate that our proposed perturbation significantly stabilizes convergence and improves both statistical efficiency and privacy performance compared to
    
[^21]: 基于希尔伯特空间嵌入的量子最大似然预测

    Quantum Maximum Likelihood Prediction via Hilbert Space Embeddings

    [https://arxiv.org/abs/2602.18364](https://arxiv.org/abs/2602.18364)

    本文通过将经验概率分布嵌入量子态并最小化量子相对熵，提出了一种量子最大似然预测方法，并为其在经典和量子大语言模型中的统一应用提供了非渐近性能保证。

    

    arXiv:2602.18364v3 公告类型: 替换-交叉 摘要：最大似然预测（MLP）是现代大型语言模型的核心任务。在此，我们首次针对由独立同分布样本构成的简化数据模型，研究该任务的量子版本。量子最大似然预测器（QMLP）通过将经验概率分布嵌入到量子态中，并在给定状态类上最小化量子相对熵来获得。我们推导了QMLP在迹范数和量子相对熵方面的非渐近性能保证，包括收敛速率和浓度不等式。我们的方法为在经典和量子大语言模型中处理MLP提供了一个统一框架。我们还考虑了量子信息投影的相关问题，并将著名的量子毕达哥拉斯定理推广到并非由自伴类生成的混合族。

    arXiv:2602.18364v3 Announce Type: replace-cross  Abstract: Maximum likelihood prediction (MLP) is a core task at the heart of modern large language models. Here, we study a quantum version of this task for a simplified data model consisting of independent and identically distributed samples, as a first step. The quantum maximum likelihood predictor (QMLP) is obtained by embedding of empirical probability distributions into quantum states and performing a minimization of quantum relative entropy over a given class of states. We derive non-asymptotic performance guarantees for QMLP in terms of convergence rates and concentration inequalities, both in trace norm and quantum relative entropy. Our approach provides a unified framework to handle MLP within both classical and quantum LLMs. We also consider the related problem of quantum information projection and generalize the well known quantum Pythagorean theorem to mixture families which are not necessarily generated by a self-adjoint cla
    
[^22]: 基于Cantelli不等式的约束策略优化

    Cantelli Constrained Policy Optimization

    [https://arxiv.org/abs/2601.22993](https://arxiv.org/abs/2601.22993)

    本文提出风险厌恶方法Canary，利用Cantelli不等式基于成本回报的前两阶矩得到可处理的风险价值约束上界，并扩展CPO信赖域框架提供最坏情况保证，是所有测试环境中唯一能可靠满足风险价值约束的方法。

    

    我们提出了Canary，这是一种风险厌恶型方法，旨在优化带有风险价值约束的强化学习问题。我们利用Cantelli不等式，基于成本回报的一阶矩和二阶矩，得到了一个可处理、保守且平滑的风险价值约束上界。由此产生的约束估计器在密集成本机制下，即使违反阈值设置得很严格也能保持稳定。在约束策略优化（CPO）方法的信赖域框架基础上进行扩展，我们进一步为训练过程中的策略改进和约束违反提供了最坏情况界。实证结果表明，在训练过程中，Canary是所有测试环境中唯一能够可靠满足风险价值约束的方法。

    arXiv:2601.22993v5 Announce Type: replace  Abstract: We introduce Canary, a risk-averse method designed to optimize Value-at-Risk (VaR) constrained reinforcement learning (RL) problems. We employ Cantelli's inequality to obtain a tractable, conservative and smooth bound on the VaR constraint based on the first two moments of the cost return. This yields a constraint estimator that remains stable with tight violation thresholds in dense cost regimes. Extending the trust-region framework of the Constrained Policy Optimization (CPO) method, we further provide worst-case bounds for both policy improvement and constraint violation during the training process. Empirically during training, Canary is the only method that reliably satisfies the VaR constraint in every environment tested.
    
[^23]: 什么驱动了基于联合嵌入预测世界模型的物理规划的成功？

    What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

    [https://arxiv.org/abs/2512.24497](https://arxiv.org/abs/2512.24497)

    本文将联合嵌入预测世界模型（JEPA-WM）类规划方法进行了系统化表征，通过对若干关键组件的全面研究，找出了在抽象表示空间中进行物理规划取得成功的关键技术选择。

    

    人工智能领域一个长期存在的挑战是开发能够解决广泛物理任务、并能泛化到新的未见任务和环境的智能体。近期一种流行的方法是从状态-动作轨迹中训练世界模型，随后将其与规划算法结合使用以解决新任务。规划通常在输入空间中进行，但最近有一类方法引入了在世界模型学习到的表示空间中进行优化的规划算法，其承诺是通过抽象掉无关细节来实现更高效的规划。在这项工作中，我们将这一类模型表征为JEPA-WMs（联合嵌入预测世界模型），并研究了使此类算法有效运作的技术选择。我们对若干关键组件进行了全面研究，目标是找出该类方法中的最优方案。我们使用模拟环境和真实世界机器人进行了实验。

    arXiv:2512.24497v4 Announce Type: replace  Abstract: A long-standing challenge in AI is to develop agents capable of solving a wide range of physical tasks and generalizing to new, unseen tasks and environments. A popular recent approach involves training a world model from state-action trajectories and subsequently use it with a planning algorithm to solve new tasks. Planning is commonly performed in the input space, but a recent family of methods has introduced planning algorithms that optimize in the learned representation space of the world model, with the promise that abstracting irrelevant details yields more efficient planning. In this work, we characterize models from this family as JEPA-WMs and investigate the technical choices that make algorithms from this class work. We propose a comprehensive study of several key components with the objective of finding the optimal approach within the family. We conducted experiments using both simulated environments and real-world robotic
    
[^24]: 基于多元伯努利分布的多标签数据抽样方法及其在元研究中的应用

    A Multivariate Bernoulli-Based Sampling Method for Multi-Label Data with Application to Meta-Research

    [https://arxiv.org/abs/2512.08371](https://arxiv.org/abs/2512.08371)

    提出了一种基于多元伯努利分布、考虑标签间依赖性的加权抽样算法，解决了多标签数据中稀有标签难以获得足够样本的问题，并成功应用于元研究领域。

    

    数据集可能包含具有多个标签的观测值。如果标签之间不互斥，且各标签的出现频率差异很大，那么要获得一个既包含足够多稀有标签观测值以便对这些标签进行推断、又以已知方式偏离总体频率的样本，将面临很大挑战。在本文中，我们将多元伯努利分布作为多标签问题的底层分布。我们提出了一种考虑标签依赖性的新型抽样算法。该算法利用观测到的标签频率来估计多元伯努利分布的参数，并为每个标签组合计算权重。这种方法确保加权抽样能够获得目标分布的特征，同时考虑到标签之间的依赖关系。我们将该方法应用于多种数据集，其中包括从Web of Science中抽取的带有标签的研究论文样本……

    arXiv:2512.08371v5 Announce Type: replace  Abstract: Datasets may contain observations with multiple labels. If the labels are not mutually exclusive, and if the labels vary greatly in frequency, obtaining a sample that includes sufficient observations with scarcer labels to make inferences about those labels, and which deviates from the population frequencies in a known manner, creates challenges. In this paper, we consider a multivariate Bernoulli distribution as our underlying distribution of a multi-label problem. We present a novel sampling algorithm that takes label dependencies into account. It uses observed label frequencies to estimate multivariate Bernoulli distribution parameters and calculates weights for each label combination. This approach ensures the weighted sampling acquires target distribution characteristics while accounting for label dependencies. We applied this approach to a variety of datasets, including a sample of research articles from Web of Science labeled 
    
[^25]: 集合卡尔曼反演竞赛

    The Ensemble Kalman Inversion Race

    [https://arxiv.org/abs/2511.15853](https://arxiv.org/abs/2511.15853)

    该论文聚焦气候模型参数校准问题，指出随着混合物理-机器学习气候模型日益复杂，集合卡尔曼方法因无需导数、可扩展至高维且对统计观测噪声鲁棒，成为实现快速迭代、校准驱动的气候模型开发的自然选择。

    

    集合卡尔曼方法最初是为解决海洋学中的非线性数据同化问题而开发的，但如今已在远超其原始应用场景的众多领域中广受欢迎。其中特别值得关注的是气候模型校准。随着物理-机器学习混合模型的不断发展，气候模型中参数的数量和参数化的复杂度将持续增长。为了充分发挥这些进展的潜力，我们必须从费时费力的人工调参转向以校准驱动的快速迭代模型开发模式。因此，对这些参数进行稳健校准正变得日益重要。我们聚焦于在理想化设定下，通过最小化模拟气候统计量与观测气候统计量之间的失配来学习气候模型参数。集合卡尔曼方法是解决该问题的自然选择，因为它们无需导数、可扩展至高维，并且对统计观测带来的噪声具有鲁棒性。

    arXiv:2511.15853v2 Announce Type: replace-cross  Abstract: Ensemble Kalman methods were initially developed to solve nonlinear data assimilation problems in oceanography but are now popular in applications far beyond their original use cases. Of particular interest is climate model calibration. As hybrid physics and machine-learning models advance, the number of parameters and complexity of parameterizations in climate models will continue to grow. To fully realize these advances, we must move from laborious hand-tuning to calibration-driven model development in rapid iteration cycles. Thus, robust calibration of these parameters plays an increasingly important role. We focus on learning climate model parameters by minimizing the misfit between modeled and observed climate statistics in an idealized setting. Ensemble Kalman methods are a natural choice for this problem because they are derivative-free, scalable to high dimensions, and robust to noise caused by statistical observations.
    
[^26]: 双模三向不对称相异性的多维尺度分析：寻找原型轮廓与聚类

    Multidimensional scaling of two-mode three-way asymmetric dissimilarities: finding archetypal profiles and clustering

    [https://arxiv.org/abs/2511.15813](https://arxiv.org/abs/2511.15813)

    本文将h-plot方法扩展至三向（含对称与不对称）邻近数据，提出一种基于特征向量解析解的多维尺度分析新方法，能够从三向不对称相异性数据中提取原型轮廓并实现聚类。

    

    多维尺度分析可以可视化对象之间的相异性并降低数据维度。虽然已有许多方法用于处理对称的邻近数据，但不对称的、尤其是三向邻近数据（用于捕捉跨多个场合的关系）仍然研究不足。h-plot方法通过将相异性嵌入欧几里得空间，能够分析不对称和非自反的关系，从而可以进一步应用原型分析等技术来识别具有代表性的极端轮廓。然而，目前尚无现有方法能够从三向不对称邻近数据中提取原型轮廓。这项工作将h-plot方法扩展到三向邻近数据，涵盖对称与不对称、条件与非条件框架。所提出的方法具有多项优势：通过统一的欧几里得表示实现直观的可解释性；基于特征向量的显式解析解，避免了局部极小值问题……

    arXiv:2511.15813v2 Announce Type: replace-cross  Abstract: Multidimensional scaling visualizes dissimilarities among objects and reduces data dimensionality. While many methods address symmetric proximity data, asymmetric and especially three-way proximity data (capturing relationships across multiple occasions) remain underexplored. The h-plot enables the analysis of asymmetric and non-reflexive relationships by embedding dissimilarities in a Euclidean space, allowing further techniques like archetypoid analysis to identify representative extreme profiles. However, no existing methods extract archetypal profiles from three-way asymmetric proximity data. This work extends the h-plot methodology to three-way proximity data under both symmetric and asymmetric, conditional and unconditional frameworks. The proposed approach offers several advantages: intuitive interpretability through a unified Euclidean representation; an explicit, eigenvector-based analytical solution free from local mi
    
[^27]: 廉价前向计算场景下基于控制变量的梯度预测

    Gradient Prediction with Control Variates in the Cheap-Forward Regime

    [https://arxiv.org/abs/2511.05187](https://arxiv.org/abs/2511.05187)

    该论文提出用降精度、推理风格的程序预测梯度，并通过控制变量将大量预测与少量精确梯度结合，使近似误差转化为方差而非偏差，从而在集群推理资源足够廉价时降低语言模型训练的成本。

    

    我们研究能否利用原本闲置的推理资源来降低训练中稀缺GPU的成本。我们的分析采用一种模拟计算账本，其中集群工作按稀缺GPU单次前向计算的一部分计费；所有实验均在常规GPU上运行。我们的算法通过一个降低精度、推理风格的反向模式程序来预测梯度，并通过控制变量将大量预测梯度与少量精确梯度相结合，使近似误差表现为方差而非偏差。在一个1.24亿参数的语言模型以及选定的短训练窗口上，当集群工作足够便宜时，该方法相对于所测试的基线能够降低模拟账本成本。跨越1000万至7.74亿参数规模的实验既显示了方法的有效迁移，也显示了失败案例。我们并未测试仅推理专用硬件、端到端的分布式延迟，或完整的按批次大小扫描的优化器基线。

    arXiv:2511.05187v2 Announce Type: replace  Abstract: We study whether otherwise-idle inference resources could reduce the scarce-GPU cost of training. Our analysis uses a simulated compute ledger in which fleet work is billed at a fraction of a scarce-GPU forward; all experiments run on a regular GPU. Our algorithm predicts gradients with a reduced-precision, inference-style reverse-mode program and combines many predictions with a few exact gradients through a control variate, so approximation error becomes variance rather than bias. On a 124M-parameter language model and selected short training windows, the method can lower simulated ledger cost relative to the tested baselines when fleet work is sufficiently cheap. Experiments spanning 10M-774M parameters show both transfers and failures. We do not test inference-only hardware, end-to-end distributed latency, or a full optimizer-by-batch-size baseline sweep.
    
[^28]: 无需上游数据的神经变分切割后验

    Neural Variational Cut Posteriors without Upstream Data

    [https://arxiv.org/abs/2510.10268](https://arxiv.org/abs/2510.10268)

    提出NeVI-Cut方法，一种无需访问上游数据和模型、仅利用上游后验样本即可模块化且可证明准确地近似切割后验的神经变分推断方法。

    

    在许多应用中，需要将来自先前（上游）分析的参数不确定性（以样本形式提供）传播到后续（下游）分析中，且不允许反馈。这一问题被称为“切断反馈”（cutting feedback）或 cut-Bayes，而切割后验作为保持信息流约束的最优后验已被充分刻画。然而，从切割后验中采样（例如通过嵌套MCMC）计算成本高昂，而现有的用于cut-Bayes的变分推断方法需要访问上游数据和模型，这在实际中往往不可得。我们提出了一种模块化且可证明准确的cut-Bayes方法，无需访问上游数据或模型。我们利用切割后验作为在上游后验期望下最小化下游条件Kullback-Leibler散度的刻画，并用上游样本的经验平均来替代期望。我们的方法NeVI-Cut（用于切割后验的神经变分推断）……

    arXiv:2510.10268v3 Announce Type: replace-cross  Abstract: In many applications, one must propagate parameter uncertainty from an earlier (upstream) analysis, available as samples, to subsequent (downstream) analyses without feedback. This problem is called cutting feedback or cut-Bayes, and the cut-posterior, the optimal posterior preserving information-flow constraints, is well characterized. However, sampling from it (e.g., via nested MCMC) is computationally intensive, while existing variational inference methods for cut-Bayes require access to upstream data and model, often unavailable. We propose a modular and provably accurate cut-Bayes approach requiring no access to upstream data or model. We leverage the characterization of the cut-posterior as the minimizer of the expected downstream conditional Kullback-Leibler divergence over the upstream posterior, replacing the expectation with the sample average over upstream draws. Our method, NeVI-Cut (neural variational inference for
    
[^29]: DLM-One：用于单步序列生成的扩散语言模型

    DLM-One: Diffusion Language Models for One-Step Sequence Generation

    [https://arxiv.org/abs/2506.00290](https://arxiv.org/abs/2506.00290)

    DLM-One提出了一种基于分数蒸馏的框架，将扩散语言模型的生成过程压缩为单步，实现采样步数约2000倍、推理时间约500倍的加速，同时保持有竞争力的文本生成性能。

    

    本文介绍了DLM-One，这是一个基于分数蒸馏的框架，可实现连续扩散语言模型（DLM）的单步序列生成。DLM-One通过将学生模型输出的分数与前向扩散噪声空间中预训练教师DLM的分数函数对齐，从而消除了迭代精炼过程。我们证明了该框架与具体架构无关，并在多种连续流形上具有鲁棒性，包括标准的词嵌入空间和logit单纯形空间。通过对多个代表性扩散语言模型的实验，我们展示了DLM-One在采样步数上实现了高达约2000倍的加速，在墙钟时间上实现了约500倍的加速，同时在基准文本生成任务上保持了有竞争力的性能。我们进一步分析了语言领域扩散蒸馏中的失败模式，并提出了一种对抗正则化的两阶段训练方案以防止学生模型退化。

    arXiv:2506.00290v2 Announce Type: replace  Abstract: This paper introduces DLM-One, a score-distillation-based framework for one-step sequence generation with continuous diffusion language models (DLMs). DLM-One eliminates iterative refinement by aligning the scores of a student model's outputs with the score function of a pretrained teacher DLM in the forward-diffused noisy space. We demonstrate that our framework is architecture-agnostic and robust across diverse continuous manifolds, including standard token embedding spaces and logit simplex spaces. Through experiments on multiple representative DLMs, we show that DLM-One achieves up to $\sim$2000$\times$ speedup in sampling steps and $\sim$500$\times$ in wall-clock time, while maintaining competitive performance on benchmark text generation tasks. We further analyze failure modes in language-domain diffusion distillation and propose an adversarially-regularized two-stage training scheme to prevent student degeneration. Our finding
    
[^30]: 基于信赖域的随机函数贝叶斯优化中的自适应复制策略

    Adaptive Replication Strategies in Trust-Region-Based Bayesian Optimization of Stochastic Functions

    [https://arxiv.org/abs/2504.20527](https://arxiv.org/abs/2504.20527)

    该论文提出了 OGPIT 方法，在信赖域框架下将高斯过程局部建模与自适应重复评估（复制）策略相结合，通过改进采集函数和成本感知评估策略，在目标函数噪声大、需要大量采样的随机优化场景中显著提升计算效率。

    

    我们开发并分析了一种基于高斯过程模型、在信赖域框架下的随机仿真优化方法。我们重点关注目标函数方差较大的场景，在这种情况下精确估计十分困难，往往需要大量的函数评估。为应对这一情形，我们将局部建模与自适应复制相结合，使方法能够在最有价值的地方分配重复评估。我们引入了多种促进并动态调整复制的机制，包括对采集函数的修改以及考虑成本的评估策略。这些组件使我们的方法在需要大量采样以降低噪声时能够有效扩展。我们将所得到的方法称为 OGPIT，即信赖域内基于高斯过程的优化。数值实验表明，自适应复制能够在保持解质量的同时显著提高计算效率。

    arXiv:2504.20527v3 Announce Type: replace-cross  Abstract: We develop and analyze a method for stochastic simulation optimization based on Gaussian process models within a trust-region framework. We focus on settings where the variance of the objective function is large, making accurate estimation challenging and often requiring many evaluations. To address this regime, we combine local modeling with adaptive replication, allowing the method to allocate repeated evaluations where they are most beneficial. We introduce several mechanisms to promote and adapt replication, including modifications to the acquisition function and cost-aware evaluation strategies. These components enable our approach to scale effectively when high levels of sampling are required to reduce noise. We refer to the resulting method as OGPIT, for Optimization by Gaussian Processes In Trust regions. Numerical experiments show that adaptive replication can substantially improve computational efficiency while preser
    
[^31]: 面向高维概率电价预测的在线多变量正则化分布回归

    Online Multivariate Regularized Distributional Regression for High-dimensional Probabilistic Electricity Price Forecasting

    [https://arxiv.org/abs/2504.02518](https://arxiv.org/abs/2504.02518)

    本文提出了一种结合在线坐标下降与LASSO正则化的多变量分布回归在线算法，可高效建模日前电价的条件均值、方差与依赖结构，实现高维空间下快速准确且避免过拟合的概率电价预测。

    

    概率电价预测（PEPF）对短期电力市场至关重要，然而日前价格的多变量特性——横跨24个连续小时——仍然未得到充分探索。与此同时，实时决策需要既准确又快速的方法。我们提出了一种针对多变量分布回归模型的在线算法，能够高效地对电价的条件均值、方差和依赖结构进行建模。该方法将多变量分布回归与在线坐标下降法以及LASSO型正则化（绝对收缩与选择算子）相结合，实现了高维协变量空间中的可扩展估计。此外，我们提出了一种在复杂度递增的依赖结构上的正则化估计路径，允许提前停止并避免过拟合。在使用德国历史数据的案例研究中……

    arXiv:2504.02518v4 Announce Type: replace-cross  Abstract: Probabilistic electricity price forecasting (PEPF) is vital for short-term electricity markets, yet the multivariate nature of day-ahead prices - spanning 24 consecutive hours - remains underexplored. At the same time, real-time decision-making requires methods that are both accurate and fast. We introduce an online algorithm for multivariate distributional regression models, allowing efficient modeling of the conditional means, variances, and dependence structures of electricity prices. The approach combines multivariate distributional regression with online coordinate descent and LASSO-type regularization (absolute shrinkage and selection operator), enabling scalable estimation in high-dimensional covariate spaces. Additionally, we propose a regularized estimation path over increasingly complex dependence structures, allowing for early stopping and avoiding overfitting. In a case study using historical data from the German da
    
[^32]: 鲁棒流式主成分分析

    Robust Streaming PCA

    [https://arxiv.org/abs/1902.03223](https://arxiv.org/abs/1902.03223)

    该论文提出了协方差矩阵属于时变不确定集合的鲁棒流式主成分分析框架，给出了算法收敛的基本极限，并证明噪声幂法在此扰动设定下达到速率最优。

    

    我们研究了当随机数据生成模型受到扰动时的流式主成分分析问题。现有模型假设协方差矩阵是固定的，而我们采用鲁棒的视角，即协方差矩阵属于一个随时间变化的不确定集合。在此设定下，我们给出了任何恢复主成分的算法在收敛性上的基本极限。我们分析了噪声幂法和Oja算法的收敛性（这两种算法此前都是针对平稳数据生成模型研究的），并论证了在我们的设定下噪声幂法在速率上是最优的。最后，我们通过在合成数据集和真实数据集上的数值实验验证了我们分析的有效性。

    arXiv:1902.03223v4 Announce Type: replace-cross  Abstract: We consider streaming principal component analysis when the stochastic data generating model is subject to perturbations. While existing models assume a fixed covariance, we adopt a robust perspective where the covariance matrix belongs to a temporal uncertainty set. Under this setting, we provide fundamental limits on convergence of any algorithm recovering principal components. We analyze the convergence of the noisy power method and Oja's algorithm, both studied for the stationary data generating model, and argue that the noisy power method is rate-optimal in our setting. Finally, we demonstrate the validity of our analysis through numerical experiments on synthetic and real-world datasets.
    
[^33]: 带有异常值的三元数据聚类

    Clustering Three-Way Data with Outliers. (arXiv:2310.05288v1 [stat.ML])

    [http://arxiv.org/abs/2310.05288](http://arxiv.org/abs/2310.05288)

    这项研究提出了一种用于聚类矩阵形式数据的方法，可以处理其中的异常值。

    

    矩阵变量分布是模型聚类领域的最新添加，从而可以分析具有复杂结构（如图像和时间序列）的矩阵形式数据。由于其最近的出现，关于矩阵变量数据的文献有限，对于处理这些模型中的异常值的文献更少。本文讨论了一种用于聚类矩阵变量正态数据的方法。该方法使用子集对数似然的分布，将OCLUST算法扩展到矩阵变量正态数据，并使用迭代方法检测和剪裁异常值。

    Matrix-variate distributions are a recent addition to the model-based clustering field, thereby making it possible to analyze data in matrix form with complex structure such as images and time series. Due to its recent appearance, there is limited literature on matrix-variate data, with even less on dealing with outliers in these models. An approach for clustering matrix-variate normal data with outliers is discussed. The approach, which uses the distribution of subset log-likelihoods, extends the OCLUST algorithm to matrix-variate normal data and uses an iterative approach to detect and trim outliers.
    
[^34]: 使用分数后验概率对汤普森抽样进行广义遗憾分析

    Generalized Regret Analysis of Thompson Sampling using Fractional Posteriors. (arXiv:2309.06349v1 [stat.ML])

    [http://arxiv.org/abs/2309.06349](http://arxiv.org/abs/2309.06349)

    这项研究对使用分数后验概率的汤普森抽样算法进行了广义遗憾分析，获得了依赖于实例和实例独立的频率遗憾界。这对多臂赌博问题的解决有重要意义。

    

    汤普森抽样（TS）是解决随机多臂赌博问题的最流行和最早的算法之一。我们考虑了TS的一个变种，称为α-TS，其中我们使用分数或α-后验（α∈（0,1））代替标准后验分布。为了计算α-后验，标准后验的定义中的似然函数被一个因子α搅拌。对于α-TS，我们在非常温和的先验和奖励分布条件下获得了既依赖于实例的Ο（∑_{k≠i^*}Δ_k（\frac{\log(T)}{C(α)Δ_k^2}+\frac{1}{2}））也依赖于实例独立的Ο（\sqrt{KT\log K}）频率遗憾界，其中Δ_k是第k个和最好的臂的真实均值奖励之间的差，而C(α)是已知的常数。子高斯和指数族模型都满足我们对奖励分布的一般条件。我们对先验的条件是...

    Thompson sampling (TS) is one of the most popular and earliest algorithms to solve stochastic multi-armed bandit problems. We consider a variant of TS, named $\alpha$-TS, where we use a fractional or $\alpha$-posterior ($\alpha\in(0,1)$) instead of the standard posterior distribution. To compute an $\alpha$-posterior, the likelihood in the definition of the standard posterior is tempered with a factor $\alpha$. For $\alpha$-TS we obtain both instance-dependent $\mathcal{O}\left(\sum_{k \neq i^*} \Delta_k\left(\frac{\log(T)}{C(\alpha)\Delta_k^2} + \frac{1}{2} \right)\right)$ and instance-independent $\mathcal{O}(\sqrt{KT\log K})$ frequentist regret bounds under very mild conditions on the prior and reward distributions, where $\Delta_k$ is the gap between the true mean rewards of the $k^{th}$ and the best arms, and $C(\alpha)$ is a known constant. Both the sub-Gaussian and exponential family models satisfy our general conditions on the reward distribution. Our conditions on the prior di
    

