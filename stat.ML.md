# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prediction-powered inference for time series across space](https://arxiv.org/abs/2610.08715) | 本文针对时空数据提出适用于时间依赖场景的预测驱动推断方法，利用短期标注数据与长期无标签协变量，在每个空间位置为未来期望标签值构建有效置信区间，解决了传统PPI独立同分布假设失效的问题。 |
| [^2] | [Steering Diffusion Models to Rare Events with Sequential Monte Carlo](https://arxiv.org/abs/2610.08652) | 本文提出DireSMC，一种序列蒙特卡洛方法，通过引导加权样本群体趋向扩散模型中的稀有事件，不仅能生成稀有事件样本，还能给出其概率的校准估计，并可轻松扩展到各类用户自定义稀有事件。 |
| [^3] | [Spectral Recovery of Point Clouds from Noisy Geometric Graphs](https://arxiv.org/abs/2610.08634) | 该论文证明在点数与维度均趋于无穷的高维信号+噪声图模型下，只要满足谱间隙条件，利用邻接矩阵的顶部特征向量和特征值即可在正交变换意义下近似恢复被高斯噪声扰动的低维点云。 |
| [^4] | [Feature Information Dynamics in Diffusion](https://arxiv.org/abs/2610.08626) | 提出了基于 I-MMSE 恒等式的信息论框架“特征信息动力学”，通过比较无条件与特征条件去噪损失之差来估计特征信息密度，从而精确定位各特征在扩散生成过程中出现的时间，并定量证实了谱自回归现象。 |
| [^5] | [Early Memory Selection for Balanced Adam](https://arxiv.org/abs/2610.08624) | 该论文提出一种通过短暂试点训练自动选择Adam共享记忆参数β的方法，利用三次记忆规则平衡采样波动与梯度平均延迟，在十一个视觉和语言任务上将平均相对验证差距降低40.7%以上。 |
| [^6] | [Classifications in modular restricted Boltzmann machines](https://arxiv.org/abs/2610.08612) | 该论文将霍普菲尔德模型与受限玻尔兹曼机之间的对偶性扩展到模块化框架，证明由赫布型模块内耦合与反赫布型模块间耦合构成的模块化联想网络等价于隐含层相互耦合的多个RBM，并可通过一步对比散度训练实现多标签分类。 |
| [^7] | [When does conformal calibration need censoring weights? Cause-of-failure prediction sets under competing risks](https://arxiv.org/abs/2610.08602) | 该论文研究竞争风险场景下保形预测的校准问题，发现右删失导致的完全案例校准会使总体覆盖率向任一方向偏离名义水平，并由此分析何时必须引入删失权重才能保证有效覆盖。 |
| [^8] | [Reinforcement Learning for Hierarchical Reasoning Rewards: Minimax-Optimal Rates with Transformers](https://arxiv.org/abs/2610.08561) | 本文将推理任务的奖励建模为响应空间上的分层函数，并证明一种基于Transformer的actor-critic强化学习算法在查询预算和正则化强度上达到极小极大最优速率，从理论上解释了在策略探索结合神经奖励模型的RL后训练为何有效。 |
| [^9] | [Information-Dense Synthesis for Molecular Discovery](https://arxiv.org/abs/2610.08495) | 提出信息密集型合成方法，通过设计、合成复杂分子混合物并池化测试后解卷积分子-活性映射，理论上可将寻找最优分子的实验次数从O(d)降至O(log d)或O(1)，比现有贝叶斯优化方法效率提升一个数量级。 |
| [^10] | [One-Shot Private Confidence Regions via Resampling](https://arxiv.org/abs/2610.08460) | 提出一个一次性构建差分隐私置信区域的简单框架，仅对最终重采样分位数加噪，使隐私代价在有放回（m-out-of-n）采样下仅为对数级、在无放回（子）采样下与重采样次数 B 无关，避免了以往方法中的 √B 因子，并给出了非渐近的高斯差分隐私与效用保证。 |
| [^11] | [Scalable Regularized Vector Multiplicative Error Models for Positive-valued Financial Time Series](https://arxiv.org/abs/2610.08443) | 本文提出一种基于分层滞后结构与分块坐标下降算法的正则化对数向量乘性误差模型，实现了高维正值金融时间序列的高效估计与预测。 |
| [^12] | [Symmetry-Aware Feature Learning: A Polynomial Separation for Multi-Index Models](https://arxiv.org/abs/2610.08420) | 该论文证明了对称感知与对称无关特征学习之间存在多项式级的样本复杂度分离：在具有循环对称轨道的增长秩多指标模型中，通过权重共享或全群数据增强利用对称性的学习器，仅需约 $d^{p-1}$ 个样本（对于信息指数 $p\ge3$ 的多项式链接函数）即可实现弱方向恢复，而无法利用对称性的学习器则代价更高。 |
| [^13] | [Network Intervention by Polling Strategic Agents](https://arxiv.org/abs/2610.08347) | 论文利用最优价格基于中心性的福利核分解，提出轮询算法Poll，使规划者能够通过局部询问高效设定最优非歧视性价格，同时应对智能体私人信息、谎报行为和计算可扩展性三大挑战。 |
| [^14] | [High-Dimensional Statistical Inference for Sparse Support Vector Machines](https://arxiv.org/abs/2610.08345) | 该论文通过将 $L_1$-惩罚支持向量机表示为线性规划并借助对偶变量识别铰链损失次梯度，突破了铰链损失非光滑性导致的去偏难题，首次在高维比例渐近机制下为稀疏SVM建立了计算可行的渐近高斯推断框架，实现了置信区间、假设检验和FDR受控的变量选择。 |
| [^15] | [Two-Sample Testing via Generative Processes](https://arxiv.org/abs/2610.08277) | 提出了一种基于随机插值与时间反射对称性的双样本检验方法，通过计算时间 t 和 1-t 处边缘分布的 Jensen-Shannon 散度来判断两个样本是否同分布，该方法无需学习任何参数、可精确控制有限样本检验水平，并能达到极小化极大分离速率。 |
| [^16] | [How Many Independent Samples Does a Satellite Image Contain? Generalization Bounds for Spatially Dependent Data](https://arxiv.org/abs/2610.08227) | 该论文证明了空间相关性持续 r 个像素的 n×n 卫星图像，其有效样本量仅为 Θ(n²/r²) 而非 n²，并通过匹配的上下界证明该速率是紧的且不可超越，从而为空间交叉验证提供了最优泛化保证的理论依据。 |
| [^17] | [Anytime-valid simulation-based hypothesis testing](https://arxiv.org/abs/2610.08210) | 本文针对只能获得模拟样本而无解析密度的假设检验问题，构造了 e 检验鞅，实现了任意时刻有效的一类错误控制、几何衰减的二类错误界和渐近满功效的序贯检验。 |
| [^18] | [Beyond Marginal Monitoring: Distributed Joint-Distribution Testing for Data Concept Drift in Large Scale E-Commerce Operations](https://arxiv.org/abs/2610.08132) | 该论文在亿级规模的电商真实数据上系统评估了五种多变量双样本漂移检测方法，证明基于 Apache Spark 的分布式最大均值差异（MMD）结合随机傅里叶特征的方法在检测概念漂移时具备稳健的可扩展性。 |
| [^19] | [ProximalFM: Amortized Proximal Causal Inference under Hidden Confounding](https://arxiv.org/abs/2610.08078) | 该论文提出ProximalFM，利用先验数据拟合网络（PFN）以摊销式贝叶斯方法在隐藏混杂下进行近端因果推断，通过先验正则化缓解了非参数近端估计中病态积分方程的数据饥渴、超参数敏感和优化不稳定等问题。 |
| [^20] | [Detecting a Shift Is Not Enough: Exact Minimax Limits of Linear Representation Repair](https://arxiv.org/abs/2610.08069) | 该论文将两个数据源间均值偏移的消除建模为统计决策问题，推导出线性表示修复的精确有限样本极小极大风险，并揭示了“检测-修复差距”：检测偏移仅需信噪比 κ 远大于 √d，而实际修复则需 κ 与维度 d 同阶。 |
| [^21] | [Spectra: Exact Component Transport for Test-Time Prior Adaptation in Simulation-Based Inference](https://arxiv.org/abs/2610.08021) | Spectra利用精确的得分传输恒等式，使冻结的扩散式SBI模型能够在测试时以闭式形式适应结构化的先验变化，无需额外的仿真或训练，在六个基准测试中于强先验偏移下实现了准确的后验推断。 |
| [^22] | [Stochastic Gradient Descent Ascent is Suboptimal for Nonconvex-PL Min-Max Games](https://arxiv.org/abs/2610.07814) | 该论文首次建立了非凸-PL极小极大博弈中固定时间尺度比双时间尺度SGDA的紧致复杂度下界，证明SGDA本质上次优——其下界与现有上界匹配、与Smoothed-AGDA形成复杂度分离，且当时间尺度比小于o(κ²)时甚至无法找到驻点。 |
| [^23] | [Adaptive Mean Estimation by In-Context Learning: A Gradient-Flow Analysis](https://arxiv.org/abs/2610.07804) | 该论文通过梯度流分析揭示了先验拟合网络（如TabPFN）如何在分布族未知的均值估计任务中通过上下文学习获得统计自适应性，自动选择与数据分布相匹配的最优估计策略并达到相应的理论收敛速率。 |
| [^24] | [Extending Pathwise Gradients to Discrete Random Variables via Finite-Order Relaxation](https://arxiv.org/abs/2610.07786) | 提出一个通用框架，通过有限阶松弛为泊松等常见离散变量构建精确的路径梯度估计器，该估计器保留硬前向采样、无需温度调节、实现简单，且在所有可行解中唯一并最小化权重方差。 |
| [^25] | [Trustworthy Method Comparison with AI Judges: Estimation and Design under Order, Batch, and Aggregation Effects](https://arxiv.org/abs/2610.07755) | 该论文提出用马尔可夫广义线性混合模型刻画LLM裁判的评估机制，证明了随机化取平均排名法的一致性条件及威廉姆斯方设计的效率优势，并揭示了组间比较中因模型非线性导致朴素平均可能得出错误结论的问题。 |
| [^26] | [Adversarially Trained Linear Transformers Are Optimal Robust In-Context Learners for Gaussian Mixtures](https://arxiv.org/abs/2610.07754) | 经过跨任务对抗预训练的线性Transformer无需额外训练，即可通过上下文学习将鲁棒性迁移到未见过的任务，并渐近达到高斯混合分类任务的最优鲁棒贝叶斯误差。 |
| [^27] | [High-dimensional online calibration from harmonic weights](https://arxiv.org/abs/2610.07740) | 本文提出了一个基于调和权重、对过去结果进行调和平滑的简单在线校准算法，首次在高维多结果预测中以 $d^{O(1/\varepsilon)}$ 轮实现 $\varepsilon$-校准，将此前结果的维度依赖性指数级降低。 |
| [^28] | [Nash Social Welfare for Multi Armed Bandits: Trajectory-wise Expected and High Probability Regret](https://arxiv.org/abs/2610.07737) | 该论文提出了一种新的“轨迹级纳什遗憾”度量，通过先对完整奖励样本路径取几何平均再求期望，弥补了现有度量忽略各轮奖励联合分布的缺陷，并借助詹森不等式证明其严格强于原有度量，从而更忠实地体现纳什社会福利的公平性目标。 |
| [^29] | [Exact Calibration and Sharp Risk Geometry for Volume-Sampled Ridge Regression](https://arxiv.org/abs/2610.07721) | 本文为体积抽样岭回归建立了精确的惩罚校准理论（当且仅当抽样行数超过目标有效维度时该惩罚存在），并通过严格的扇形不等式刻画了中心化协方差风险的尖锐上界以及所有最大化响应的结构。 |
| [^30] | [Stability of Measure-to-Measure Transformers on Sub-Gaussian Data](https://arxiv.org/abs/2610.07717) | 本文从数学上证明了Transformer将次高斯数据映射为次高斯输出且关于1-Wasserstein距离具有Hölder连续性，由此建立了经验近似下的误差传播估计，并揭示了交叉注意力机制均值场类似物的不同正则性与样本复杂度。 |
| [^31] | [Uniform Discrete Diffusion Models are Minimax Optimal for Estimating Distributions with Small Effective Support Size](https://arxiv.org/abs/2610.07655) | 该论文证明了均匀离散扩散模型在估计有效支撑较小的分布时达到极小极大最优，其统计误差由依赖于样本量的有效支撑大小而非环境空间规模决定。 |
| [^32] | [Asymptotic Analysis of Empirical Risk Minimization on Entry-wise i.i.d. Heavy-Tailed Data](https://arxiv.org/abs/2610.07637) | 本文通过引入函数序参量并运用复制方法，首次在比例高维极限下精确刻画了对称α-稳定重尾数据上线性回归经验风险最小化的泛化误差，并建立了重尾普适性定律与相应的标度律。 |
| [^33] | [Explicit Asymptotic Bounds for Sequential Calibration Beyond $T^{2/3}$](https://arxiv.org/abs/2610.07623) | 该论文提出新的两阶段递归标记策略并改进归约方法，首次为序贯校准问题建立了超越 $T^{2/3}$ 的显式渐近界 $O(T^{0.662942288})$。 |
| [^34] | [Vine Copula VAR:From Recursive Margins to Joint Forecast Inference](https://arxiv.org/abs/2610.07589) | 本文在稳定的藤Copula VAR框架下，推导出历史递归估计误差与终端预测的联合影响，并提出一个协方差估计量，可为固定单侧事件概率提供渐近有效的重复样本预测区间。 |
| [^35] | [Learning a Mixture of GFlowNets](https://arxiv.org/abs/2610.07562) | 提出了一个描述GFlowNets混合体的通用理论框架，将其细分为连续索引（CI）和离散索引（DI）两类：前者通过随机特征扩展与谱移位可证明地提升采样器的表达能力并降低学习不稳定性，后者统一了已有训练方法并支撑了新提出的分层条件化（SC）GFlowNets。 |
| [^36] | [Is $\sqrt{d}$ Separation Necessary for Gradient EM to Learn Gaussian Mixtures in High Dimensions?](https://arxiv.org/abs/2610.07551) | 本文证明在高维学习高斯混合模型时，梯度 EM 全局收敛所需的 $\Omega(\sqrt{d})$ 分离度条件是不可避免的，即这一对维度的依赖性无法被去除。 |
| [^37] | [Two-Sample Testing for Random Graphs without Vertex Correspondence](https://arxiv.org/abs/2610.07503) | 该论文首次建立了顶点无对应情形下图总体双样本检验的最优样本复杂度理论，证明对于保持度不变的两块结构差异，每组需要约 t^{-3} 张图，带符号三角形计数可达到该最优速率，且未对齐相比对齐情形需多付出 t^{-2} 阶的样本代价。 |
| [^38] | [Does Muon Need Fine-Grained Spectral Shaping?](https://arxiv.org/abs/2610.07497) | 本文提出 BulkBoost 双频段谱重加权框架，表明 Muon 并不需要细粒度的谱整形，只需将奇异谱粗略地划分为噪声主体和高增益尖峰两个频段并进行重加权即可提升优化效果。 |
| [^39] | [The interface of data assimilation and machine learning](https://arxiv.org/abs/2610.07496) | 本文综述了数据同化与机器学习这一新兴交叉领域的主要主题和方法，探讨了两者结合对混沌系统（如大气）进行最优状态估计的应用前景。 |
| [^40] | [Structure, Not Belief: Correlated Thompson Sampling from LLM-Derived Covariance in Combinatorial Semi-Bandits](https://arxiv.org/abs/2610.07470) | 提出一种对组合汤普森采样的最小改动方法，仅查询LLM一次将臂划分转化为相关协方差矩阵来引导探索，理论证明相比独立采样可获得有限时域内√(d/K)的遗憾改进，实验中遗憾降低19%。 |
| [^41] | [Active Feature Acquisition for Cost-Efficient Temporal Prediction with Reduced Participant Burden](https://arxiv.org/abs/2610.07452) | 该论文提出纵向主动特征获取（LAFA）方法，通过学习一种策略在每个时间点仅选择性地采集最优的条目动态子集，从而在降低参与者负担、减少无应答与流失风险的同时，保持对心理病理结果的准确预测能力。 |
| [^42] | [Bayesian Optimization on Function Spaces via Sparse RKHS Manifolds](https://arxiv.org/abs/2610.07417) | 提出L0MO方法，通过在RKHS中由核函数稀疏表示构成的流形子集上搜索，并同时优化核的位置与系数，从而实现函数空间中的贝叶斯优化，并为现有FBO方法提供了统一视角。 |
| [^43] | [A perspective note on likelihood approximation and inference for complex simulation models using a chain of aggregated normalizing flows](https://arxiv.org/abs/2610.07391) | 提出了一种基于n级聚合标准化流链的似然近似新方法，通过按顺序估计各组双射变换参数，为复杂仿真模型的大规模数据分析、假设检验和不确定性量化提供了可扩展且高效的解决方案。 |
| [^44] | [DeepAJM: Deep Association Joint Model for Irregularly Sampled data](https://arxiv.org/abs/2610.07388) | 提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。 |
| [^45] | [HyperNSDE: Personalized Neural SDEs for Joint Static-Longitudinal Clinical Data Generation](https://arxiv.org/abs/2610.07383) | HyperNSDE通过超网络将静态患者特征注入潜在神经SDE，首次联合生成异构静态协变量、不规则采样的纵向轨迹和观测时间三类紧密耦合的临床数据，为医疗AI提供真实且保护隐私的合成患者数据。 |
| [^46] | [Redundancy and synergy in multivariate Gaussians via the Blackwell order](https://arxiv.org/abs/2610.07360) | 本文基于Blackwell序为多变量高斯系统提出了部分信息分解的定义，证明了高斯信道对提取冗余与联合信息是最优的，并给出了两信源情形下的闭式表达式与高效数值算法。 |
| [^47] | [Assumption-lean logistic regression with missing covariates](https://arxiv.org/abs/2610.07292) | 该论文提出了一种在协变量分布未知（仅有界）的少假设设定下处理协变量缺失逻辑回归参数估计的随机近似方法，克服了传统方法在协变量分布未知时可能严重失效的问题。 |
| [^48] | [Empirical-Bayes spectral partial pooling across related tasks](https://arxiv.org/abs/2610.07284) | 该论文提出了一种可扩展的经验贝叶斯框架——层次谱收缩（HSS），通过在相关任务之间对任务特定的谱估计器进行部分池化，在样本量相对有限时既能获得比独立估计更稳定的结果，又能保留完全合并所无法体现的任务特定结构。 |
| [^49] | [Conditional Flow Matching for Transport Between Markov Processes](https://arxiv.org/abs/2610.07229) | 本文提出一种保持马尔可夫结构的条件流匹配算法，用于学习马尔可夫过程从源轨迹分布到目标轨迹分布的传输映射，并证明了总体一致性、有限样本误差界以及与混合时间相关的样本复杂度下界。 |
| [^50] | [How Inefficient Is Natural Gradient Descent? From Exact Optimality to \Theta ( \sqrt{ \log d } ) Divergence](https://arxiv.org/abs/2610.07228) | 本文提出“低效比”来量化自然梯度下降偏离最短 Fisher–Rao 路径的程度，并证明该比值在三类情形中变化：二次势或一维族精确最优（R=1）、有界偏度族具有与维度无关的界、而尺度族乘积（如高斯协方差和 Gamma 率）的低效比随维度以 Θ(√log d) 增长。 |
| [^51] | [Poisson empirical Bayes estimation of sums of random variables via minimum-distance methods](https://arxiv.org/abs/2610.07190) | 该论文提出了一种基于规则与粗化最小距离估计的非参数经验贝叶斯方法，用于估计泊松混合模型中可观测与不可观测变量函数之和，证明了大样本下代入估计与oracle贝叶斯估计渐近一致，且当混合分布支撑有限时可达接近参数化的收敛速度。 |
| [^52] | [A theory of platonic representations in language models](https://arxiv.org/abs/2610.07168) | 本文通过假设数据具有隐藏的层级结构（抽象层次跨语言共享、表面层次为语言或模态特定），并借助概率上下文无关文法与信念传播理论推导出分析性预测，首次从理论上解释了多语言模型中间层出现柏拉图式表示的现象及其随语言相近程度和模型质量增强的规律。 |
| [^53] | [Sample-Optimal Estimation of the Fr\'echet Inception Distance](https://arxiv.org/abs/2610.07114) | 该论文针对FID估计中的有限样本偏差问题，证明了插件估计器的紧致偏差与方差界并确立其平方级（d²）样本复杂度，同时将FID∞估计器推广到任意阶外推方法以实现去偏估计。 |
| [^54] | [A Query Is Not a Commitment: Learning to Correct Expert Answers in Online Deferral](https://arxiv.org/abs/2610.07084) | 提出ORUCB算法，利用累积响应学习误差的界来校准置信度加权风险回归与探索，在在线推迟学习中对不准确专家的答案进行纠正，实现了 $O(\sqrt T\log(T+1))$ 的高概率伪遗憾界。 |
| [^55] | [Learning Decision-Stump Thresholds in Context: Dynamics of Softmax Attention](https://arxiv.org/abs/2610.07074) | 本文证明了两参数softmax注意力模型通过基于梯度的预训练能够学习决策阈值估计，其误差为$\widetilde O((m\wedge n)^{-1}+N^{-1})$，并揭示了背后的机制是参数协调发散——注意力尺度以$t^{1/4}$增长、阈值误差以$t^{-1/4}$衰减。 |
| [^56] | [sHAIL-Causal: A Sequential Staircase Procedure for Invariant Causal Predictor Discovery](https://arxiv.org/abs/2610.07057) | 本文提出 sHAIL-Causal，一种以拟合优度饱和与跨环境不变性联合判据为门控的序列阶梯式学习程序，可避免被混杂预测因子诱导，并在 Richness 条件下可证明地停在真正的因果预测因子集合上，而仅依赖复杂度控制或朴素贪心搜索的方法均无法做到。 |
| [^57] | [Data Fusion for Errors-in-Variables](https://arxiv.org/abs/2610.07048) | 本文提出了一种数据融合估计方法，通过条件可迁移性假设利用外部研究的重复测量来识别目标研究中的条件测量误差分布，从而在源-目标异质性下解决变量含误差问题。 |
| [^58] | [Multi-Task Active Learning with Efficient Resource Allocation](https://arxiv.org/abs/2610.07045) | 该论文提出 ALCATRAs 统一框架，通过成本自适应的任务选择策略与代理学习，在预算约束下有选择地获取标注期的昂贵辅助信息，从而提升部署时辅助变量系统性缺失场景下的下游预测性能。 |
| [^59] | [The Premise Is the Problem: Exchangeability Failure in Self-Monitored Test-Time Adaptation](https://arxiv.org/abs/2610.07038) | 论文证明在自监控的测试时自适应中，由于监控与自适应共用同一反馈，预测目标重叠与误差依赖性会破坏可交换性假设，导致虚假警报、预测质量下降，且自适应会掩盖持续变化，而冻结的原始模型信号反而更清晰。 |
| [^60] | [Near-Optimal Sample Complexity for Recursive Entropic Risk Reinforcement Learning with a Generative Model](https://arxiv.org/abs/2610.06931) | 本文对基于模型的风险敏感 Q 值迭代（MB-RS-QVI）算法进行了精细分析，在生成模型假设下首次为递归熵风险强化学习建立了近最优的样本复杂度保证，其对有效视界的指数依赖性与现有下界相匹配，消除了理论差距。 |
| [^61] | [Low-Rank and Structured Sparse Tensor Decomposition for Anomaly Detection in Multivariate Functional Data](https://arxiv.org/abs/2610.06930) | 提出两种无监督稀疏张量分解方法（ES-CP与FG-Lasso），通过低秩CP分解结合逐元素与纤维方向的稀疏惩罚，在保留多模态结构的同时检测多元函数型数据中的局部异常与时间纤维集中型异常。 |
| [^62] | [Anchor Divergence for Semantic Geometry in Contrastive Learning](https://arxiv.org/abs/2610.06919) | 本文提出“锚点散度”方法，通过建立锚点概率分布与Bregman几何之间的对应关系，使固定表示上的语义几何能够适配特定上下文，突破了余弦相似度单一固定几何的局限。 |
| [^63] | [Memory Prediction Excess: A Probabilistic Quantity for Predictive Gain and Memory Length in Stochastic Processes](https://arxiv.org/abs/2610.06894) | 本文提出“记忆预测超额”（MPE）这一新的概率量，用以量化在离散时间有限状态随机过程中利用完整历史信息相对于仅用静态边际分布所带来的预测准确率平均提升，并证明了其非负性、上界条件及退化情形等基本性质。 |
| [^64] | [Consideration Circuits: Depth Separation and Universality Beyond a Single Softmax](https://arxiv.org/abs/2610.04143) | 论文提出由多项logit（MNL）单元构成的有向无环图所定义的“考虑电路”多阶段选择模型，并证明了尖锐的深度-范数分离定理：深度从2增至3时，逼近误差ε所需的品味向量范数从Θ(log(1/ε)/ε)降至Θ(log(1/ε))，而包括单一MNL在内的菜单无关随机效用模型在折中任务上误差存在不可消除的下界，从而在表达能力与结构上严格超越了单一softmax模型。 |
| [^65] | [Grand Canonical Generators](https://arxiv.org/abs/2610.00683) | 提出了巨正则生成器（GCG），将玻尔兹曼生成器扩展至巨正则系综，其分解式设计可复用现有正则生成器、解析编码化学势线性依赖，并提供可处理的似然以支持自归一化重要性采样，在流体和吸附问题上准确再现巨正则观测量。 |
| [^66] | [Schedule optimization for tau-leaping in masked discrete diffusion](https://arxiv.org/abs/2609.21960) | 该论文通过依赖密度ρ的精确积分表示来刻画掩码离散扩散中tau-leaping采样的因式分解误差，并推导有限步优化问题的递归平稳性方程，从而实现去噪调度的优化。 |
| [^67] | [When Is the Sharp Covariance Envelope Tight? Feature-Only Geometry for Volume-Sampled Least Squares](https://arxiv.org/abs/2608.26877) | 本文建立了体积采样最小二乘中心系数协方差的Loewner包络，并揭示了仅特征边际nu_A决定谱包络严格性的精确条件。 |
| [^68] | [A Critical Audit of Spatiotemporal Forecasting Benchmark Datasets and Baselines](https://arxiv.org/abs/2608.20980) | 本文通过经典时间序列方法分析常用时空基准数据集，揭示无空间感知的线性模型比以往报告更具竞争力，质疑了现有基准数据集的判别可靠性。 |
| [^69] | [QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction](https://arxiv.org/abs/2608.13966) | 本文提出QUASAR，一种在量化感知训练过程中持续进行轻量级损失感知重构的方法，以降低损失下限并提升低比特模型质量。 |
| [^70] | [The Geometry of Statistical Feature Learning in Mean-Field Langevin Dynamics](https://arxiv.org/abs/2606.31429) | 该论文通过基-纤维分解为统计特征学习建立了几何框架，证明球面平均场朗之万动力学的低温平稳分布在多指标模型中会集中于隐藏指标并形成多尖峰结构、以高概率实现参数恢复，且这一集中现象在温度约等于1处存在锐利的相变。 |
| [^71] | [Learning Probabilistic Filters with Strictly Proper Scoring Rules](https://arxiv.org/abs/2606.26497) | 本文提出PSEF方法，利用严格恰当评分规则训练基于Transformer的置换不变映射，仅通过合成数据实现贝叶斯滤波分布的逼近。 |
| [^72] | [Enhancing Spectral Embedding through Robust and Flexible Knowledge Transfer in Electronic Health Records](https://arxiv.org/abs/2606.11570) | 该论文提出一种基于谱方法的无监督表示学习框架，通过放宽一对一信号对齐假设并采用两步嵌入流程，从更广泛人群中稳健且灵活地迁移知识，为样本量有限的罕见病电子健康记录数据生成高质量的低维嵌入表示。 |
| [^73] | [Express Language Modeling](https://arxiv.org/abs/2606.10944) | Express 是一种将非因果注意力近似转换为具有匹配保证的因果近似的工具，与 Thinformer 结合后实现了已知最佳的因果注意力近似保证，并通过高效的 Triton 实现显著超越 FlashAttention 2，解决了语言建模中的长上下文预填充、KV 缓存压缩和长文本解码等四大资源瓶颈。 |
| [^74] | [Flow-Transformed Implicit Processes for Function-Space Variational Inference](https://arxiv.org/abs/2606.01954) | 提出流变换隐过程（FTIP），通过超越高斯组合权重分布的限制，使有限维函数空间近似能够灵活表示非对称、重尾或多峰的后验不确定性。 |
| [^75] | [Online Conformal Prediction for Non-Exchangeable Panel Data](https://arxiv.org/abs/2605.17705) | 提出了W-TQA方法，通过结合从单元历史学习的相似性权重与自适应误覆盖水平，解决了非可交换、部分观测面板数据中的在线保形预测问题，并证明了即使在反馈缺失情况下也能实现长期平均覆盖率保证。 |
| [^76] | [Concentration and Calibration in Predictive Bayesian Inference](https://arxiv.org/abs/2605.00455) | 本文证明了预测贝叶斯推断的后验会集中到一个明确定义的量上，且该量及其不确定性量化完全由所选的前向预测模型决定，从而揭示了PBI的可靠性与校准性本质上取决于前向预测模型的选择。 |
| [^77] | [Deep Time-Series Forecasting in 10 Years: A Survey](https://arxiv.org/abs/2603.19899) | 本文从自相关性建模的统一视角系统综述了十年来的深度时间序列预测研究，首次提出同时涵盖骨干架构与损失函数的分类体系，并据此剖析了现有文献的动机与洞见。 |
| [^78] | [Fast and Efficient Asynchronous Gossip Algorithm for Robust and Non-Smooth Convex Decentralized Learning](https://arxiv.org/abs/2601.20571) | 本文提出Goal-PD，一种异步Gossip原始-对偶算法，每个节点仅维护两个变量而与网络度数无关，实现了几乎必然收敛与线性收敛，并通过分布式均值估计中的成对平均特例与经典Gossip算法建立了直接联系。 |
| [^79] | [Intersectional Fairness via Mixed-Integer Optimization](https://arxiv.org/abs/2601.19595) | 本文提出一个基于混合整数优化（MIO）的统一框架，训练同时具备交叉公平性和内在可解释性的分类器，证明了两种交叉公平性度量（MSD 与 SPSF）在检测最不公平子群体上的等价性，并能将交叉偏见有效控制在可接受阈值以下。 |
| [^80] | [Computationally efficient goodness-of-fit tests through kernelized Stein discrepancy](https://arxiv.org/abs/2512.20007) | 本文提出一种基于核化Stein差异的计算高效半参数拟合优度检验，并设计了无需重新拟合模型或从中采样的影响调整野自助法来确定检验的显著性水平。 |
| [^81] | [Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models](https://arxiv.org/abs/2511.00053) | 该论文提出了一种新颖的二次型加权学习目标，通过加权矩阵的非对角元素捕捉未来步骤间的标签自相关效应，同时利用非均匀对角元素为不同预测步骤设置异构任务权重，从而同时解决传统均方误差目标的两个缺陷，提升多步时间序列预测模型的训练效果。 |
| [^82] | [Action-Driven Processes for Continuous-Time Control](https://arxiv.org/abs/2510.26672) | 本文通过动作驱动过程统一了随机过程与强化学习的视角，证明最小化策略驱动分布与奖励驱动分布之间的KL散度等价于最大熵强化学习，并将其应用于脉冲神经网络。 |
| [^83] | [Beyond the Semicircle: Free Diffusion Models with Prescribed Equilibria](https://arxiv.org/abs/2510.22778) | 该论文发现状态依赖的自由波动率能突破常系数自由扩散只能收敛到半圆律的固有局限，并为任意充分正则的紧支撑目标谱分布显式构造出具有指定平衡态的自由扩散模型。 |
| [^84] | [Improving Mixup Calibration with Wasserstein Distributionally Robust Optimization](https://arxiv.org/abs/2506.17874) | 本文提出DRO-Augment框架，将Wasserstein分布鲁棒优化与Mixup数据增强相结合，有效缓解了腐蚀鲁棒性与模型校准之间的权衡，在保持腐蚀准确率的同时显著降低了期望校准误差（ECE）。 |
| [^85] | [Identifiability Analysis of Linear ODE Systems with Hidden Confounders](https://arxiv.org/abs/2410.21917) | 本文系统分析了含隐藏混杂因素的线性常微分方程系统的可辨识性，分别研究了潜在混杂因素无因果关系但遵循特定函数形式（如时间多项式）演化，以及潜在混杂因素之间具有由有向无环图描述的因果依赖关系这两种情况，填补了该领域的空白。 |
| [^86] | [FreDF: Learning to Forecast in Frequency Domain](https://arxiv.org/abs/2402.02399) | FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。 |

# 详细

[^1]: 面向跨空间时间序列的预测驱动推断

    Prediction-powered inference for time series across space

    [https://arxiv.org/abs/2610.08715](https://arxiv.org/abs/2610.08715)

    本文针对时空数据提出适用于时间依赖场景的预测驱动推断方法，利用短期标注数据与长期无标签协变量，在每个空间位置为未来期望标签值构建有效置信区间，解决了传统PPI独立同分布假设失效的问题。

    

    以下情境在时空数据环境中十分常见：我们在一个相对较短的近期时间段内观测到协变量与标签的配对序列，同时可以获取更长时间段内的无标签协变量，且数据分布在许多空间位置上。例如，作物产量可能仅在最近几年于较大地理区域内被观测到，而天气数据（可用于预测作物产量）则可在更长的时间段内获得。研究目标是在每个空间位置估计未来的期望标签值（如作物产量），并给出该值的有效置信区间。然而，仅凭已观测的较短时间段无法做出可靠估计；用机器学习填补缺失标签则会引入显著偏差。预测驱动推断（PPI）能够纠正这种偏差，但它依赖于独立同分布假设，而该假设在时间序列依赖性下会被打破。此外，异方差性和自相关问题（摘要在此处截断）……

    arXiv:2610.08715v1 Announce Type: cross  Abstract: The following motif is common in spatiotemporal settings: we have a sequence of covariate and label pairs observed for a relatively short, recent time period. We have access to unlabeled covariates over a longer time period. Data is observed over many spatial locations. For instance, crop yield might be observed over a large geographical area for recent years, but weather data (which is informative about crop yield) is available for a much longer period. The goal is to estimate, at each spatial location, the expected label (e.g., crop yield) in the future and provide a valid confidence interval for this value. The observed time period alone is too short for reliable estimates. Imputing missing labels with machine learning can cause substantial bias. Prediction-powered inference (PPI) can correct for this bias, but it relies on an i.i.d. assumption that breaks under our expected temporal dependencies. Heteroskedasticity and autocorrelat
    
[^2]: 利用序列蒙特卡洛引导扩散模型生成稀有事件

    Steering Diffusion Models to Rare Events with Sequential Monte Carlo

    [https://arxiv.org/abs/2610.08652](https://arxiv.org/abs/2610.08652)

    本文提出DireSMC，一种序列蒙特卡洛方法，通过引导加权样本群体趋向扩散模型中的稀有事件，不仅能生成稀有事件样本，还能给出其概率的校准估计，并可轻松扩展到各类用户自定义稀有事件。

    

    扩散模型正日益被用作天气预报、分子动力学和材料设计等领域中昂贵模拟器的替代品。在这些模型中，计算事件 $E$ 的概率 $p_0[E]$ 十分困难，尤其当所关注的事件是稀有事件时。使用蒙特卡洛方法进行稳定估计在计算上将变得不可行，因为需要随稀有程度增加而不断增长的样本量（$\propto 1/p_0[E]$）来补偿。在本文中，我们提出了稀有事件的扩散重要性采样方法（Diffusion Importance Sampling of Rare Events，简称 DireSMC），这是一种序列蒙特卡洛方案，通过引导一组加权样本趋向稀有事件，不仅能获得样本，还能得到其概率的校准估计。我们通过对事件集合进行解析松弛来构建引导机制，使该方法能够轻松扩展到广泛的用户自定义稀有事件。我们在一个具有解析解的玩具问题以及一个基于分数的模型上验证了我们的方法。

    arXiv:2610.08652v1 Announce Type: cross  Abstract: Diffusion models are increasingly used as surrogates for expensive simulators in weather prediction, molecular dynamics, and materials design. In these models, computing the probability $p_0[E]$ of an event $E$ is difficult, especially when the event of interest is rare. A stable estimate using Monte Carlo becomes computationally intractable, requiring a growing sample size $\propto\!1/p_0[E]$ to compensate for an increasing rarity. In this paper, we present Diffusion Importance Sampling of Rare Events or DireSMC, a sequential Monte Carlo scheme that guides a population of weighted samples towards the rare event, giving access not only to samples but also to a calibrated estimate of its probability. We set up our guidance using an analytical relaxation of the event set, allowing the method to easily extend to a wide range of user-defined rare events. We validate our method on a toy problem with analytical solutions and on a score-based
    
[^3]: 从带噪声几何图中谱恢复点云

    Spectral Recovery of Point Clouds from Noisy Geometric Graphs

    [https://arxiv.org/abs/2610.08634](https://arxiv.org/abs/2610.08634)

    该论文证明在点数与维度均趋于无穷的高维信号+噪声图模型下，只要满足谱间隙条件，利用邻接矩阵的顶部特征向量和特征值即可在正交变换意义下近似恢复被高斯噪声扰动的低维点云。

    

    我们研究从由带噪声的高维数据生成的随机几何图中恢复低维潜在几何的问题。具体而言，我们分析谱嵌入算法在信号+噪声图模型上的性能，该模型中顶点与被高斯噪声扰动的点相关联，当点对的内积超过指定的对齐阈值时则连接边。在点数 $n$ 和环境维度 $d$ 都趋于无穷的高维情况下，我们证明在谱间隙条件下，图的邻接矩阵的顶部特征向量和特征值可用于在正交变换意义下近似恢复点云。我们在从嵌套球面和高维正弦曲线中采样的点云上展示了我们的结果。

    arXiv:2610.08634v1 Announce Type: cross  Abstract: We study the problem of recovering low-dimensional latent geometry from a random geometric graph generated by noisy, high-dimensional data. Specifically, we analyze the performance of a spectral embedding algorithm on the Signal+Noise Graph Model, in which vertices are associated to points perturbed by Gaussian noise, and edges are included for pairs whose inner product exceeds a specified alignment threshold. In the high-dimensional regime where the number $n$ of points and the ambient dimension $d$ both tend to infinity, we show that under a spectral gap condition, the top eigenvectors and eigenvalues of the graph's adjacency matrix can be used to approximately recover the point cloud up to an orthogonal transformation. We illustrate our results on point clouds sampled from nested spheres and high-dimensional sinusoid curves.
    
[^4]: 扩散模型中的特征信息动力学

    Feature Information Dynamics in Diffusion

    [https://arxiv.org/abs/2610.08626](https://arxiv.org/abs/2610.08626)

    提出了基于 I-MMSE 恒等式的信息论框架“特征信息动力学”，通过比较无条件与特征条件去噪损失之差来估计特征信息密度，从而精确定位各特征在扩散生成过程中出现的时间，并定量证实了谱自回归现象。

    

    扩散模型通过一系列连续的去噪问题来生成数据，并被广泛观察到先呈现粗略结构、后生成精细细节。然而，这一直觉大多停留在经验性和定性层面。我们提出了特征信息动力学，这是一个用于定位特征在扩散过程中何时被生成的信息论框架。利用 I-MMSE 恒等式，我们将特征互信息的变化率与最优无条件去噪损失和特征条件去噪损失之间的差距联系起来，从而得到特征信息密度的实用估计器。我们进一步开发了一种链式分解方法，可在特征层次结构中分离共享信息与增量信息。我们首先利用该框架定量证实了像素扩散中的谱自回归现象，随后将分析扩展到频率维度之外：在“类别 → 掩码 → Canny”条件链下，各特征的信息密度在像素空间上存在差异。

    arXiv:2610.08626v1 Announce Type: cross  Abstract: Diffusion models generate data through a continuum of denoising problems, and are widely observed to reveal coarse structure before fine detail. Yet, this intuition is mostly empirical and qualitative. We introduce feature information dynamics, an information-theoretic framework for localizing when a feature is generated during diffusion. Using the I-MMSE identity, we connect the rate of feature mutual information change to a gap between optimal unconditional and feature-conditional denoising losses, yielding practical estimators for feature information density. We further develop a chained decomposition that separates shared from incremental information in a feature hierarchy. We use this framework first to quantitatively confirm spectral autoregression in pixel diffusion, and then to extend the analysis beyond frequency: under a class $\to$ mask $\to$ Canny conditioning chain, the per-feature information densities differ across pixel
    
[^5]: 面向平衡Adam的早期记忆选择方法

    Early Memory Selection for Balanced Adam

    [https://arxiv.org/abs/2610.08624](https://arxiv.org/abs/2610.08624)

    该论文提出一种通过短暂试点训练自动选择Adam共享记忆参数β的方法，利用三次记忆规则平衡采样波动与梯度平均延迟，在十一个视觉和语言任务上将平均相对验证差距降低40.7%以上。

    

    我们提出了一种通过短暂的试点训练来选择Adam中共享记忆参数 $\beta_1=\beta_2=\beta$ 的方法。所选的 $\beta$ 在随后的完整训练过程中保持固定。通过对Adam归一化方向的局部建模，我们平衡了采样波动性与平均历史梯度所引入的延迟。这种平衡导出了一个三次记忆规则，其两个系数通过在少数试点检查点处的梯度探测来估计。该估计器联合使用分子和分母，从而保留了两者的协方差。在200步更新的试点训练中，于四个检查点各使用十六个探测梯度，在与随机种子匹配的回溯评估中，该方法在十一个视觉和语言工作负载上相比共享 $\beta=0.95$ 的网格代表值，将平均相对验证差距降低了40.7%，最差四分之一平均差距降低了44.3%。其平均差距还比从全部十一个工作负载中选出的最佳常数 $\beta$ 低32.3%。

    arXiv:2610.08624v1 Announce Type: cross  Abstract: We propose a method for choosing the shared memory parameter $\beta_1=\beta_2=\beta$ in Adam from a short pilot training. The selected $\beta$ remains fixed during the subsequent full training. A local model of Adam's normalized direction balances sampling variability against the delay introduced by averaging past gradients. This balance gives a cubic memory rule, whose two coefficients are estimated from gradient probes at a few pilot checkpoints. The estimator uses the numerator and denominator jointly, preserving their covariance. With a 200-update pilot and sixteen probe gradients at each of four checkpoints, a seed-matched retrospective evaluation on eleven vision and language workloads reduces mean relative validation gap by 40.7% and worst-quarter mean gap by 44.3% against the grid representative of shared $\beta=0.95$. The mean gap is also 32.3% lower than that of the best constant $\beta$ chosen across all eleven workloads.
    
[^6]: 模块化受限玻尔兹曼机中的分类

    Classifications in modular restricted Boltzmann machines

    [https://arxiv.org/abs/2610.08612](https://arxiv.org/abs/2610.08612)

    该论文将霍普菲尔德模型与受限玻尔兹曼机之间的对偶性扩展到模块化框架，证明由赫布型模块内耦合与反赫布型模块间耦合构成的模块化联想网络等价于隐含层相互耦合的多个RBM，并可通过一步对比散度训练实现多标签分类。

    

    我们考虑一个由 $L$ 个霍普菲尔德模型（HMs）组成的模块化联想神经网络，其耦合方式为：模块内相互作用是赫布型（Hebbian），模块间相互作用是反赫布型（anti-Hebbian）；这种竞争性耦合已被证明能赋予网络模式解耦的能力。该系统的积分表示与 $L$ 个隐含层相互耦合的受限玻尔兹曼机（RBMs）的集合相一致，从而将霍普菲尔德模型-RBM对偶性推广到了模块化设置中。随后，我们通过一步对比散度训练该模块化RBM来执行分类任务，其中编码在可见层上的查询被映射为从隐含层读出的 $L$ 元组标签。当查询由平均意义上相互正交的 $L$ 个模式组成时，我们证明了依照霍普菲尔德模型-RBM等价性的建议、作为训练数据集上的经验均值得到的RBM权重构成了该（动力学）的一个不动点……（摘要在此处截断）

    arXiv:2610.08612v1 Announce Type: cross  Abstract: We consider a modular associative neural network made of $L$ Hopfield models (HMs), coupled so that intra-module interactions are Hebbian and inter-module interactions are anti-Hebbian; this competitive coupling is known to endow the network with pattern-disentanglement capabilities. The integral representation of this system coincides with an assembly of $L$ restricted Boltzmann machines (RBMs) whose hidden layers are coupled, thereby extending the HM-RBM duality to the modular setting. We then train this modular RBM, via one-step contrastive divergence, to perform a classification task in which a query encoded on the visible layers is mapped onto an $L$-tuple of labels read off the hidden layers. When the query is composed of $L$ patterns that are mutually orthogonal on average, we prove that the RBM weights obtained as empirical means over the training dataset, as suggested by the HM-RBM equivalence, constitute a fixed point of the 
    
[^7]: 何时保形校准需要删失权重？竞争风险下的失效原因预测集

    When does conformal calibration need censoring weights? Cause-of-failure prediction sets under competing risks

    [https://arxiv.org/abs/2610.08602](https://arxiv.org/abs/2610.08602)

    该论文研究竞争风险场景下保形预测的校准问题，发现右删失导致的完全案例校准会使总体覆盖率向任一方向偏离名义水平，并由此分析何时必须引入删失权重才能保证有效覆盖。

    

    固定时间点竞争风险标签的分割保形预测集所需的校准标签，可能因右删失而无法被观测到。完全案例（complete-case）校准仅能保证标签完整子群体的覆盖率，其总体覆盖率可能向任一方向偏离名义水平，即使模型给出的是真实类别概率也是如此。我们研究了选择机制如何改变总体分位数附近的分数分布。在我们的主要模拟族中，采用独立抽样，当22%的受试者在给定时间点无事件时，完全案例校准在真实分数处的覆盖率为0.8723，低于名义水平0.900。在48个采用逐次抽样归一化的额外设计中，有4个设计将原因发生率估计为一减去原因特异性Nelson-Aalen累积风险的负指数（即1−exp(−累积风险)），并将无事件概率作为裁剪后重新归一化的剩余部分，这些设计的完全案例覆盖率比名义水平低至少四个标准误。在这些设计中……

    arXiv:2610.08602v1 Announce Type: cross  Abstract: Split conformal prediction sets for competing-risks labels at a fixed horizon require calibration labels that right censoring can leave unobserved. Complete-case calibration guarantees coverage for the label-complete subpopulation, but its population coverage can deviate in either direction, even at the true class probabilities. We study how selection changes the score distribution near the population quantile. At the true score in our main simulation family, with independent draws, complete-case calibration covers 0.8723 at a nominal 0.900 when 22% of subjects are event-free at the horizon. In 4 of 48 further designs using per-draw normalisation, estimating cause incidences as one minus the exponential of the negative of the cause-specific Nelson-Aalen cumulative hazards, with the event-free probability as the clipped and renormalised remainder, puts complete-case coverage at least four standard errors below nominal. In these designs,
    
[^8]: 面向分层推理奖励的强化学习：基于Transformer的极小极大最优速率

    Reinforcement Learning for Hierarchical Reasoning Rewards: Minimax-Optimal Rates with Transformers

    [https://arxiv.org/abs/2610.08561](https://arxiv.org/abs/2610.08561)

    本文将推理任务的奖励建模为响应空间上的分层函数，并证明一种基于Transformer的actor-critic强化学习算法在查询预算和正则化强度上达到极小极大最优速率，从理论上解释了在策略探索结合神经奖励模型的RL后训练为何有效。

    

    强化学习（RL）已成为在推理任务上对语言模型进行后训练的标准工具，其中策略在探索响应空间的同时通过奖励反馈进行更新。尽管其在实证上取得了成功，但对RL后训练的理论理解仍然有限，尤其是对于为什么在策略探索结合神经奖励模型能够有效这一问题的理解。在本文中，我们通过将奖励建模为响应空间上的分层函数来回答这一问题：奖励由无穷多个局部组件构成，每个组件只有在前面的组件被解决之后才会变得相关。我们证明了一种自然的基于Transformer的actor-critic算法——该算法在从当前KL正则化策略中采样、用观测到的奖励拟合Transformer评论家网络、以及更新策略这三个步骤之间交替进行——在查询预算和正则化强度方面达到了极小极大最优速率（最多相差对数因子）。

    arXiv:2610.08561v1 Announce Type: new  Abstract: Reinforcement learning (RL) has become a standard tool for post-training language models on reasoning tasks, where the policy is updated by reward feedback while exploring the space of responses. Despite its empirical success, theoretical understanding of RL post-training remains limited, in particular of why on-policy exploration combined with a neural reward model is effective. In this paper, we address this question by modeling the reward as a hierarchical function on the response space: the reward consists of infinitely many local components, each of which becomes relevant only after the preceding ones have been resolved. We show that a natural Transformer-based actor--critic algorithm, which alternates between sampling from the current KL-regularized policy, fitting a Transformer critic to the observed rewards, and updating the policy, achieves the minimax optimal rates in the query budget and in the regularization strength up to lo
    
[^9]: 面向分子发现的信息密集型合成

    Information-Dense Synthesis for Molecular Discovery

    [https://arxiv.org/abs/2610.08495](https://arxiv.org/abs/2610.08495)

    提出信息密集型合成方法，通过设计、合成复杂分子混合物并池化测试后解卷积分子-活性映射，理论上可将寻找最优分子的实验次数从O(d)降至O(log d)或O(1)，比现有贝叶斯优化方法效率提升一个数量级。

    

    机器学习可以通过设计分子和规划实验来加速分子发现。然而，许多科学挑战需要具有极为罕见性质的分子，在这种稀疏设定下，现有算法相比随机猜测几乎没有优势。我们提出了一种利用算法控制的随机合成来高效搜索大范围分子空间的方法。我们不逐一设计、合成并测试单个分子，而是设计和合成复杂的混合物，将其作为一个池进行测试，然后解卷积出分子-活性映射关系。我们对合成过程进行优化以编码最大信息量。从理论上讲，该方法可以将从 $d$ 个候选分子中找到最优分子所需的实验次数从 $\mathcal{O}(d)$ 降低到 $\mathcal{O}(\log d)$ 甚至 $\mathcal{O}(1)$。在基于估计的蛋白质适应度景观的模拟中，该方法找到活性分子所需的实验次数比现有贝叶斯

    arXiv:2610.08495v1 Announce Type: cross  Abstract: Machine learning can accelerate molecular discovery by designing molecules and planning experiments. However, many scientific challenges demand molecules with very rare properties, and in this sparse setting, existing algorithms offer little gain over random guessing. We propose a method to efficiently search large regions of molecular space using algorithmically controlled stochastic synthesis. Rather than design, make and test individual molecules, we design and make complex mixtures, test them as a pool, then deconvolute the molecule-activity map. We optimize synthesis to encode maximal information. Theoretically, this approach can reduce the number of experiments required to find the optimal molecule among $d$ candidates from $\mathcal{O}(d)$ to $\mathcal{O}(\log d)$ or $\mathcal{O}(1)$. In simulation, on estimated protein fitness landscapes, it finds active molecules with an order of magnitude fewer experiments than existing Bayes
    
[^10]: 通过重采样的一次性差分隐私置信区域

    One-Shot Private Confidence Regions via Resampling

    [https://arxiv.org/abs/2610.08460](https://arxiv.org/abs/2610.08460)

    提出一个一次性构建差分隐私置信区域的简单框架，仅对最终重采样分位数加噪，使隐私代价在有放回（m-out-of-n）采样下仅为对数级、在无放回（子）采样下与重采样次数 B 无关，避免了以往方法中的 √B 因子，并给出了非渐近的高斯差分隐私与效用保证。

    

    我们提出了一个简单的框架，用于“一次性”构建差分隐私置信区域，即仅对最终的重采样分位数添加噪声，而不是对每次重采样计算的估计器进行隐私化处理。我们的方法中隐私的代价在有放回采样（$m$-out-of-$n$ 采样）下仅与重采样次数 $B$ 呈对数关系，在无放回采样（子采样）下与 $B$ 无关，从而避免了以往工作中出现的 $\sqrt{B}$ 因子。我们为子采样和 $m$-out-of-$n$ 重采样提供了非渐近的高斯差分隐私（GDP）保证和效用保证，涵盖了具有较小全局敏感度的均值类估计器，以及具有可高效计算的光滑敏感度上界的估计器，包括分位数和退化U统计量。这使我们还能为退化U统计量获得隐私置信区域，其隐私误差要小得多。

    arXiv:2610.08460v1 Announce Type: new  Abstract: We propose a simple framework for constructing differentially private confidence regions \textit{in one shot}, i.e., by adding noise only to the final resampling quantile instead of privatizing the estimator computed on each resample. The cost of privacy of our procedure is only logarithmic in the number of resamples $B$ under with-replacement ($m$-out-of-$n$) sampling and independent of $B$ under without replacement sampling (subsampling), avoiding the $\sqrt{B}$ factor that arises in previous works. We provide nonasymptotic Gaussian Differential Privacy (GDP) and utility guarantees for both subsampling and $m$-out-of-$n$ resampling, covering mean-like estimators with small global sensitivity as well as estimators admitting efficiently computable smooth sensitivity bounds, including quantiles and degenerate U-statistics. This allows us to also obtain private confidence regions for degenerate U-statistics where the private error is much 
    
[^11]: 面向正值金融时间序列的可扩展正则化向量乘性误差模型

    Scalable Regularized Vector Multiplicative Error Models for Positive-valued Financial Time Series

    [https://arxiv.org/abs/2610.08443](https://arxiv.org/abs/2610.08443)

    本文提出一种基于分层滞后结构与分块坐标下降算法的正则化对数向量乘性误差模型，实现了高维正值金融时间序列的高效估计与预测。

    

    对数乘性误差模型在对数vMEM形式下已被广泛应用于多元正值金融时间序列的建模与预测。然而，随着系统维度和滞后阶数的增加，参数数量迅速增长，使得高维情形下的模型估计在计算上极具挑战性。本文针对采用Tsionas（2004）多元伽马误差分布的对数vMEM模型，提出了基于分层滞后结构（Nicholson et al., 2020）的正则化估计方法。参数估计采用带有Gauss-Seidel式更新方案（Wright, 2015）的分块坐标下降算法，相比传统的惩罚极大似然方法，该策略实现了更高效的计算。此外，论文通过组合三种分层滞后结构（分量级、元素级、自身-其他）与四种惩罚项（组lasso、自适应组lasso、组……），对各类竞争模型进行了比较评估。

    arXiv:2610.08443v1 Announce Type: cross  Abstract: The logarithmic multiplicative error model (log-vMEM) has been useful in modeling and forecasting multivariate positive-valued financial time series. The number of parameters grow rapidly with the dimension of the system and the lag order, making estimation computationally demanding in high-dimensional settings. This paper describes regularized estimation via hierarchical lag structures (Nicholson et al., 2020) for log-vMEM models with multivariate gamma error distribution of Tsionas (2004). The parameter estimation is performed using a blockwise coordinate descent algorithm with a Gauss-Seidel-style update scheme (Wright, 2015). This enables an efficient computation strategy compared to traditional penalized maximum likelihood approaches. The competing models are juxtaposed against each other by combining three hierarchical lag structures (componentwise, elementwise, own-other) and four penalties(group-lasso, adaptive group-lasso, gro
    
[^12]: 对称感知特征学习：多指标模型的多项式分离

    Symmetry-Aware Feature Learning: A Polynomial Separation for Multi-Index Models

    [https://arxiv.org/abs/2610.08420](https://arxiv.org/abs/2610.08420)

    该论文证明了对称感知与对称无关特征学习之间存在多项式级的样本复杂度分离：在具有循环对称轨道的增长秩多指标模型中，通过权重共享或全群数据增强利用对称性的学习器，仅需约 $d^{p-1}$ 个样本（对于信息指数 $p\ge3$ 的多项式链接函数）即可实现弱方向恢复，而无法利用对称性的学习器则代价更高。

    

    我们建立了对称感知（symmetry-aware）与对称无关（symmetry-agnostic）特征学习之间的多项式样本复杂度分离。我们研究了位于 $\mathbb{R}^d$ 中的高维高斯协变量下、秩随维度增长的多指标模型，其中 $r=\Theta(d^\delta)$ 个教师方向构成一个循环对称轨道，且 $0<\delta<1/2$。我们比较了利用这一结构的三种方式：架构层面的权重共享、在完整对称群上的数据增强，以及无法获取对称性时的学习。特别地，我们分析了对称绑定（权重共享）的卷积网络、非绑定网络，以及使用全群数据增强训练的同一非绑定网络，三者均采用带相关损失的球面在线SGD进行训练。对于一类信息指数 $p\ge3$ 的多项式链接函数，我们在对数因子内证明了相互匹配的样本复杂度界：绑定与增强的学习器在 $\widetilde{\Theta}(d^{p-1})$ 个样本内即可实现弱方向恢复，而……

    arXiv:2610.08420v1 Announce Type: new  Abstract: We establish a polynomial sample complexity separation between symmetry-aware and symmetry-agnostic feature learning. We study growing-rank multi-index models with high-dimensional Gaussian covariates in $\mathbb{R}^d$ and $r=\Theta(d^\delta)$ teacher directions forming a cyclic symmetry orbit, where $0<\delta<1/2$. We compare three ways of exploiting this structure: architectural weight sharing, data augmentation over the full symmetry group, and learning without access to the symmetry. In particular, we analyze a symmetry-tied convolutional network, an untied network, and the same untied network trained with full-group data augmentation, using spherical online SGD with correlation loss. For a class of polynomial links with information exponent $p\ge3$, we prove matching sample complexity bounds up to logarithmic factors: the tied and augmented learners achieve weak directional recovery in $\widetilde{\Theta}(d^{p-1})$ samples, whereas 
    
[^13]: 通过轮询策略性智能体实现网络干预

    Network Intervention by Polling Strategic Agents

    [https://arxiv.org/abs/2610.08347](https://arxiv.org/abs/2610.08347)

    论文利用最优价格基于中心性的福利核分解，提出轮询算法Poll，使规划者能够通过局部询问高效设定最优非歧视性价格，同时应对智能体私人信息、谎报行为和计算可扩展性三大挑战。

    

    策略性智能体网络中的规划者面临三个相互交织的挑战：最优解依赖于智能体的私人信息，被询问的智能体可能谎报以操纵结果，且精确计算无法扩展。我们在具有异构私有技术的多活动网络博弈中研究这些挑战，其中规划者设定非歧视性价格。我们证明，最优价格允许对福利核进行基于中心性的分解：每个智能体的贡献与其在一个按智能体跨活动偏好重新加权的网络中的中心性平方成正比。这一分解启发了Poll——一种轮询算法，其中规划者每轮抽取一个智能体，简要遍历该智能体的邻域，并根据局部报告更新价格。从同一分解中还衍生出三种形式的效率：在计算方面，Poll所用的操作次数显著少于精确计算和其他分布式算法。

    arXiv:2610.08347v1 Announce Type: cross  Abstract: A planner in a network of strategic agents faces three entangled challenges: the optimum depends on agents' private information, queried agents may misreport to steer the outcome, and exact computation does not scale. We study these challenges in multi-activity network games with heterogeneous private technologies, in which the planner sets non-discriminatory prices. We show that the optimal prices admit a centrality-based decomposition of the welfare kernel: each agent's contribution scales with its squared centrality in a network reweighted by agents' preferences across activities. This decomposition motivates Poll, a polling algorithm in which the planner samples one agent per round, walks briefly through the agent's neighborhood, and updates the price from a local report. From the same decomposition flow three forms of efficiency: computationally, Poll uses significantly fewer operations than exact computation and other distributed
    
[^14]: 稀疏支持向量机的高维统计推断

    High-Dimensional Statistical Inference for Sparse Support Vector Machines

    [https://arxiv.org/abs/2610.08345](https://arxiv.org/abs/2610.08345)

    该论文通过将 $L_1$-惩罚支持向量机表示为线性规划并借助对偶变量识别铰链损失次梯度，突破了铰链损失非光滑性导致的去偏难题，首次在高维比例渐近机制下为稀疏SVM建立了计算可行的渐近高斯推断框架，实现了置信区间、假设检验和FDR受控的变量选择。

    

    利用复制对称的高维刻画方法，我们在样本量与特征数成比例增长的情形下，为稀疏支持向量机建立了一个统计推断框架。主要挑战在于铰链损失的非光滑性，这阻碍了为光滑分类损失发展的去偏方法被直接应用。我们通过将 $L_1$-惩罚的支持向量机（SVM）表示为一个线性规划，并经由其对偶变量识别铰链损失的次梯度，从而克服了这一困难。由此得到一个计算上可行的去偏估计量，在比例渐近机制下其各坐标渐近服从高斯分布。所得的分布刻画为单个特征提供了置信区间与假设检验，并支持错误发现率（FDR）受控的变量选择。大量模拟实验检验了校准性、功效以及变量选择的性能。

    arXiv:2610.08345v1 Announce Type: cross  Abstract: Using a replica-symmetric high-dimensional characterization, we develop an inferential framework for sparse support vector machines when the sample size and number of features grow proportionally. The main challenge is the nonsmooth hinge loss, which prevents direct application of debiasing arguments developed for smooth classification losses. We overcome this difficulty by representing the $L_1$-penalized support vector machine (SVM) as a linear program and identifying the hinge-loss subgradient through its dual variables. This yields a computationally accessible debiased estimator whose coordinates are asymptotically Gaussian under the proportional asymptotic regime. The resulting distributional characterization provides confidence intervals and hypothesis tests for individual features and enables false-discovery-rate-controlled variable selection. Extensive simulations examine calibration, power, and variable-selection performance u
    
[^15]: 基于生成过程的双样本检验

    Two-Sample Testing via Generative Processes

    [https://arxiv.org/abs/2610.08277](https://arxiv.org/abs/2610.08277)

    提出了一种基于随机插值与时间反射对称性的双样本检验方法，通过计算时间 t 和 1-t 处边缘分布的 Jensen-Shannon 散度来判断两个样本是否同分布，该方法无需学习任何参数、可精确控制有限样本检验水平，并能达到极小化极大分离速率。

    

    判断两个样本是否来自同一分布是统计学中的一个经典问题，而生成式传输为解决这一问题提供了新的途径。我们直接在两个样本之间构建随机插值，并观察到：在对称调度下，只要两个分布相同，该插值的分布在时间反射 t ↦ 1-t 下保持不变。因此，我们通过计算时间 t 和 1-t 处边缘分布之间的 Jensen-Shannon 散度来检验它们是否一致。两个边缘分布都是所有观测交叉对上的显式混合，因此无需学习任何参数，并且通过置换校准可以获得精确的有限样本检验水平。对于高斯噪声，该散度等于一个时间积分，该积分将速度场和得分函数的反射缺陷配对，因此该检验比较的是传输动力学过程而不仅仅是端点。通过窄加宽的噪声设计，该检验达到了极小化极大分离速率 n^{-2s/(4s+d)}。

    arXiv:2610.08277v1 Announce Type: cross  Abstract: Deciding whether two samples come from the same distribution is a classical problem in statistics, and generative transport offers a new way to approach it. We build a stochastic interpolant directly between the two samples and observe that, for a symmetric schedule, its law is invariant under the time reflection $t \mapsto 1-t$ whenever the two distributions coincide. We therefore test whether the marginals at times t and 1-t agree by computing their Jensen--Shannon divergence. Both marginals are explicit mixtures over all cross-pairs of observations, so nothing is learned, and permutation calibration gives an exact finite-sample level. For Gaussian noise, this divergence equals a time integral that pairs the reflection defects of the velocity field and of the score, so the test compares transport dynamics rather than endpoints alone. With a narrow-plus-broad noise design, the test attains the minimax separation rate n^{-2s/(4s+d)} ov
    
[^16]: 一张卫星图像包含多少独立样本？空间相关数据的泛化界

    How Many Independent Samples Does a Satellite Image Contain? Generalization Bounds for Spatially Dependent Data

    [https://arxiv.org/abs/2610.08227](https://arxiv.org/abs/2610.08227)

    该论文证明了空间相关性持续 r 个像素的 n×n 卫星图像，其有效样本量仅为 Θ(n²/r²) 而非 n²，并通过匹配的上下界证明该速率是紧的且不可超越，从而为空间交叉验证提供了最优泛化保证的理论依据。

    

    arXiv:2610.08227v1 通告类型：交叉 摘要：用于遥感影像的机器学习分类器通常在评估时将每个像素视为独立样本。空间自相关违反了这一假设，因为相邻像素携带冗余信息，从而夸大了样本量。一张卫星图像实际上包含多少独立样本？对于一个空间相关性在 r 个像素范围内持续存在的 n×n 图像，其有效样本量为 Θ(n²/r²)，而非 n²。我们将其证明为空间相关数据上分类器的有限样本上界，并通过匹配的下界证明该速率是紧的，即任何算法都无法超越这一速率。我们将该结果扩展到具有方向性相关和空间变化相关结构的图像。我们的结果为空间交叉验证提供了理论依据，因为与相关范围成比例的分块留出方法能够达到最优的泛化保证，而随机留出（摘要在此处截断）……

    arXiv:2610.08227v1 Announce Type: cross  Abstract: Machine learning classifiers for remote sensing imagery are typically evaluated as though every pixel were an independent sample. Spatial autocorrelation violates this assumption, since neighboring pixels carry redundant information which inflates sample sizes. How many independent samples does a satellite image actually contain? For an $n \times n$ image whose spatial correlation persists over a range of $r$ pixels, the effective sample size is $\Theta(n^2/r^2)$, not $n^2$. We prove this as a finite-sample upper bound for classifiers on spatially correlated data, and show via a matching lower bound that the rate is tight, and no algorithm can do better. We extend the results to images with directional correlation and spatially varying correlation structure. Our result justifies spatial cross-validation since block holdout with separation proportional to the correlation range achieves optimal generalization guarantees, while random hol
    
[^17]: 任意时刻有效的基于模拟的假设检验

    Anytime-valid simulation-based hypothesis testing

    [https://arxiv.org/abs/2610.08210](https://arxiv.org/abs/2610.08210)

    本文针对只能获得模拟样本而无解析密度的假设检验问题，构造了 e 检验鞅，实现了任意时刻有效的一类错误控制、几何衰减的二类错误界和渐近满功效的序贯检验。

    

    对于给定的独立同分布数据 $(X_t)_{t \in \mathbb{N}} \sim Q$，我们研究如下假设检验问题：$H_0: Q = P_0$ 对 $H_1: Q = P_1$，其中 $P_0$ 和 $P_1$ 是两个不同的模型概率分布。与标准设定中给定解析密度函数 $p_0$ 和 $p_1$ 不同，本文考虑的是无密度设定，即我们只能获得独立同分布的模拟样本 $(Z^0_t)_{t \in \mathbb{N}} \sim P_0$ 和 $(Z^1_t)_{t \in \mathbb{N}} \sim P_1$。针对这种基于模拟的假设检验设定，我们构造了一个 e 检验鞅，由此得到的序贯检验具有任意时刻有效的一类错误保证、近似增长最优性、几何衰减的二类错误界以及渐近功效为 1。我们构造中使用的大多数要素都是众所周知概念的变体。本文的价值在于以紧凑的方式呈现了一种有效的、任意时刻有效的无密度模拟假设检验解决方案。

    arXiv:2610.08210v1 Announce Type: cross  Abstract: For a given data distribution $(X_t)_{t \in \mathbb{N}} \sim Q$ i.i.d., we investigate the hypothesis testing problem: $H_0: Q = P_0$ vs. $H_1: Q = P_1$, for two different model probability distributions $P_0$ and $P_1$. In contrast to the standard setting, where analytic densities $p_0$ and $p_1$ are given, here, we consider the density-free setting, where we only have access to i.i.d. simulations $(Z^0_t)_{t \in \mathbb{N}} \sim P_0$ and $(Z^1_t)_{t \in \mathbb{N}} \sim P_1$. For this simulation-based hypothesis testing setting, we construct an e-test martingale, resulting in a sequential test with anytime-valid type-I error guarantees, approximate growth optimality, geometrically decaying type-II error bounds, and asymptotic power one. Most ingredients used in our constructions are variants of well known concepts. The value of this paper lies in the compact presentation of an effective, anytime-valid solution for the density-free si
    
[^18]: 超越边际监测：面向大规模电商运营中数据概念漂移的分布式联合分布检验

    Beyond Marginal Monitoring: Distributed Joint-Distribution Testing for Data Concept Drift in Large Scale E-Commerce Operations

    [https://arxiv.org/abs/2610.08132](https://arxiv.org/abs/2610.08132)

    该论文在亿级规模的电商真实数据上系统评估了五种多变量双样本漂移检测方法，证明基于 Apache Spark 的分布式最大均值差异（MMD）结合随机傅里叶特征的方法在检测概念漂移时具备稳健的可扩展性。

    

    概念漂移威胁着生产环境的机器学习系统，然而多变量双样本漂移检测器在大规模场景下的实证表现仍缺乏充分的研究刻画。现有基准测试很少涉及工业运营数据集中典型的数亿行数据规模和高基数特征。我们在三个互补环境中评估了五种多列双样本检验方法（边际方法、基于投影的方法和核嵌入方法）：哈佛 Dataverse 数据集、经过验证的 Failing Loudly 复现实验（平均绝对误差介于 0.030 至 0.053 之间），以及基于 1.375 亿行 Trendyol 集合排序特征表构建的新型合成注入基准。通过在两种严重性-范围机制下对四种漂移类型进行测试，我们证明了基于 Apache Spark、结合随机傅里叶特征的分布式最大均值差异（MMD）方法能够稳健地扩展。在强机制下对四种漂移类型取平均并在校准阈值条件下，该方法达到了……

    arXiv:2610.08132v1 Announce Type: cross  Abstract: Concept drift threatens production machine learning, yet the empirical behavior of multivariate two-sample drift detectors at scale remains under-characterized. Existing benchmarks rarely address the hundreds of millions of rows and high-cardinality features typical of industrial-operational datasets. We evaluate five multi-column two-sample tests (marginal, projection-based, and kernel embedding methods) across three complementary environments: the Harvard Dataverse, a validated Failing Loudly reproduction (mean absolute error between 0.030 and 0.053), and a novel synthetic-injection benchmark on the 137.5-million-row Trendyol collection-ranking feature table. Testing four drift types across two severity-scope regimes, we demonstrate that distributed Maximum Mean Discrepancy with Random Fourier Features on Apache Spark scales robustly. Averaged over the four drift types in the strong regime and under a calibrated threshold, it achieve
    
[^19]: ProximalFM：隐藏混杂下的摊销式近端因果推断

    ProximalFM: Amortized Proximal Causal Inference under Hidden Confounding

    [https://arxiv.org/abs/2610.08078](https://arxiv.org/abs/2610.08078)

    该论文提出ProximalFM，利用先验数据拟合网络（PFN）以摊销式贝叶斯方法在隐藏混杂下进行近端因果推断，通过先验正则化缓解了非参数近端估计中病态积分方程的数据饥渴、超参数敏感和优化不稳定等问题。

    

    标准的因果识别方法通常假设不存在未观测的混杂因素，当相关混杂因素未被观测到时这些方法可能会失效。近端因果推断则转而使用代理变量在隐藏混杂存在的情况下识别因果效应。然而，非参数近端估计在实践中可能颇具挑战性：恢复条件平均处理效应（CATE）等因果估计量需要求解一个病态的积分方程，该方程对数据需求量大、对超参数敏感且优化过程不稳定。对此类模型进行贝叶斯推断提供了一种理想的替代方案，可通过先验进行正则化来缓解上述困难。然而，计算后验本身也具有挑战性，因为典型的似然函数中会包含潜变量。借鉴近期表格基础模型在后门、工具变量和前门等设定中取得的成功，我们提出先验数据拟合网络（PFN）在……方面具有独特优势（摘要在此处截断）

    arXiv:2610.08078v1 Announce Type: cross  Abstract: Standard causal identification methods often assume no unmeasured confounding and can fail when relevant confounders are unobserved. Proximal causal inference instead uses proxy variables to identify effects under hidden confounding. However, nonparametric proximal estimation can be challenging in practice: recovering causal estimands such as the conditional average treatment effect (CATE) requires solving an ill-posed integral equation that is data-hungry, hyperparameter-sensitive, and optimization-unstable. Bayesian inference for such models provides a desirable alternative, mitigating these difficulties by regularizing through the prior. However, computing a posterior is itself challenging, as a typical likelihood function will include latent variables. Following the recent success of tabular foundation models in backdoor, instrumental variable, and frontdoor settings, we propose that prior-data fitted networks (PFNs) are uniquely s
    
[^20]: 检测到偏移并不足够：线性表示修复的精确极小极大极限

    Detecting a Shift Is Not Enough: Exact Minimax Limits of Linear Representation Repair

    [https://arxiv.org/abs/2610.08069](https://arxiv.org/abs/2610.08069)

    该论文将两个数据源间均值偏移的消除建模为统计决策问题，推导出线性表示修复的精确有限样本极小极大风险，并揭示了“检测-修复差距”：检测偏移仅需信噪比 κ 远大于 √d，而实际修复则需 κ 与维度 d 同阶。

    

    两个数据源之间的均值偏移可能很容易检测，但若不大幅改变其表示，则难以消除。我们将该消除问题建模为一个统计决策问题：从 $\mathbb{R}^d$ 中成对校准测量的含噪差异中，学习一个线性映射，在硬失真预算约束下同时应用于两个数据源，使得在新数据上残留的偏移尽可能小。我们推导出了所有此类映射上的精确有限样本极小极大风险：$(d-k) \mathbb{E}[1/(d+2J)]$，其中 $J\sim\mathrm{Pois}(\kappa/2)$，预算允许删除 $k$ 个方向，$\kappa$ 为校准信噪比。在不了解 $\kappa$ 或噪声尺度的情况下，投影掉均值校准差异即可达到该风险。这揭示了一个“检测-修复差距”：检测偏移只需 $\kappa\gg\sqrt d$，而在恒定失真下去除其固定比例则需要 $\kappa\asymp d$，与估计其方向的要求相同。

    arXiv:2610.08069v1 Announce Type: new  Abstract: A mean shift between two data sources can be easy to detect but hard to remove without substantially changing their representations. We cast its removal as a statistical decision problem: from noisy differences between paired calibration measurements in $\mathbb{R}^d$, learn one linear map, applied to both sources under a hard distortion budget, that leaves as little of the shift as possible on fresh data. We derive the exact finite-sample minimax risk over all such maps, $(d-k) \mathbb{E}[1/(d+2J)]$ with $J\sim\mathrm{Pois}(\kappa/2)$, where the budget allows deleting $k$ directions and $\kappa$ is the calibration signal-to-noise ratio. Projecting out the mean calibration difference attains it without knowing $\kappa$ or the noise scale. This exposes a detection-repair gap: detecting the shift needs only $\kappa\gg\sqrt d$, whereas removing a fixed fraction of it at constant distortion needs $\kappa\asymp d$, as for estimating its direc
    
[^21]: Spectra：仿真推理中面向测试时先验自适应的精确成分传输方法

    Spectra: Exact Component Transport for Test-Time Prior Adaptation in Simulation-Based Inference

    [https://arxiv.org/abs/2610.08021](https://arxiv.org/abs/2610.08021)

    Spectra利用精确的得分传输恒等式，使冻结的扩散式SBI模型能够在测试时以闭式形式适应结构化的先验变化，无需额外的仿真或训练，在六个基准测试中于强先验偏移下实现了准确的后验推断。

    

    基于仿真的推理（SBI）已成为对似然函数难以评估或无法评估的复杂科学模型进行贝叶斯推断的一种强大方法。摊销式SBI从模拟数据中学习可复用的推理模型，从而能够对新观测进行快速的后验推断，而现代生成模型使这些模型的表达能力日益增强。然而，这种复用仅限于训练时所选择的先验分布，而科学分析往往需要在知识不断积累或检验不同假设时修改先验。我们提出了Spectra，这是一种面向基于扩散模型的SBI的测试时自适应方法。Spectra利用精确的得分传输恒等式，对于结构化的先验变化，能够以闭式形式从冻结的扩散模型中获得自适应后的得分，无需额外的仿真或训练。在六个SBI基准测试中，Spectra在强先验偏移下以较低的在线采样……实现了准确的自适应。

    arXiv:2610.08021v1 Announce Type: new  Abstract: Simulation-based inference (SBI) has become a powerful approach to Bayesian inference in complex scientific models whose likelihoods are difficult or impossible to evaluate. Amortized SBI learns reusable inference models from simulated data, enabling rapid posterior inference for new observations, and modern generative models have made these models increasingly expressive. However, this reuse is limited to the prior distribution chosen during training, whereas scientific analyses often need revised priors as knowledge accumulates or alternative assumptions are tested. We introduce Spectra, a test-time adaptation method for diffusion-based SBI. Spectra uses an exact score-transport identity to obtain the adapted score from a frozen diffusion model in closed form for structured prior changes, without additional simulation or training. Across six SBI benchmarks, Spectra achieves accurate adaptation under strong prior shifts at low online sa
    
[^22]: 随机梯度下降上升法在非凸-PL极小极大博弈中是次优的

    Stochastic Gradient Descent Ascent is Suboptimal for Nonconvex-PL Min-Max Games

    [https://arxiv.org/abs/2610.07814](https://arxiv.org/abs/2610.07814)

    该论文首次建立了非凸-PL极小极大博弈中固定时间尺度比双时间尺度SGDA的紧致复杂度下界，证明SGDA本质上次优——其下界与现有上界匹配、与Smoothed-AGDA形成复杂度分离，且当时间尺度比小于o(κ²)时甚至无法找到驻点。

    

    arXiv:2610.07814v1 公告类型： cross 摘要：在非凸极小极大博弈中，通过调整时间尺度比和步长，随机梯度下降上升法（SGDA）究竟能走多远？我们针对非凸-PL（NC-PL）博弈回答了这一问题，首次建立了固定时间尺度比和非递增步长下双时间尺度SGDA的紧致复杂度。对于满足内层μ-PL不等式的ℓ-光滑博弈，我们证明了复杂度下界Ω(κ²ℓε⁻²+κ⁴ℓσ²ε⁻⁴)，其中κ=ℓ/μ为条件数，σ²为梯度方差，ε度量外层梯度范数。该下界与现有的SGDA上界相匹配，并建立了与Smoothed-AGDA（Yang等人，22'）之间的复杂度分离。此外，我们证明当SGDA的时间尺度比小至o(κ²)时，它可能无法找到驻点。我们的负面结果凸显了SGDA在NC-PL博弈中的根本局限性，并……（原文摘要至此截断）

    arXiv:2610.07814v1 Announce Type: cross  Abstract: How far can stochastic gradient descent ascent (SGDA) go by tuning its timescale ratio and step sizes in nonconvex min-max games? We answer this question for nonconvex-PL (NC-PL) games by establishing the first tight complexity of two-timescale SGDA with a fixed timescale ratio and non-increasing step sizes. For $\ell$-smooth games with an inner $\mu$-PL inequality, we prove a complexity lower bound $\Omega(\kappa^2\ell\varepsilon^{-2}+\kappa^4\ell\sigma^2\varepsilon^{-4})$, where $\kappa=\ell/\mu$ is the condition number, $\sigma^2$ is the gradient variance, and $\varepsilon$ measures the outer gradient norm. This matches existing SGDA upper bounds and establishes a complexity separation from Smoothed-AGDA (Yang et al., 22'). In addition, we show that SGDA can fail to find a stationary point when its timescale ratio is as small as $o(\kappa^2)$. Our negative results highlight the fundamental limitation of SGDA in NC-PL games, and just
    
[^23]: 通过上下文学习实现自适应均值估计：梯度流分析

    Adaptive Mean Estimation by In-Context Learning: A Gradient-Flow Analysis

    [https://arxiv.org/abs/2610.07804](https://arxiv.org/abs/2610.07804)

    该论文通过梯度流分析揭示了先验拟合网络（如TabPFN）如何在分布族未知的均值估计任务中通过上下文学习获得统计自适应性，自动选择与数据分布相匹配的最优估计策略并达到相应的理论收敛速率。

    

    先验拟合网络（PFNs）如TabPFN如今在预测和估计任务上已能与成熟的统计方法相媲美。一个自然的解释是PFNs具有统计自适应性，即对于一组异构模型，它们的表现几乎与针对真实数据生成模型定制的方法一样好，而无需被告知数据来自哪个模型。我们在一个受控的位置估计问题中研究这种自适应性是如何被学习到的。每个任务是一个未标记样本，其分布族是隐藏的：高斯数据需要用平均法估计，误差阶为 $n^{-1}$；而均匀数据最好通过其极值来估计，达到更快的 $n^{-2}$ 速率。我们还给出了对称高斯混合的例子，其可达到的速率为 $\sigma^2_n/n$。在标量输入上，softmax注意力计算的是经验累积生成函数的导数。因此，单个基元即可同时提供……

    arXiv:2610.07804v1 Announce Type: new  Abstract: Prior Fitted Networks (PFNs) such as TabPFN now rival established statistical procedures across prediction and estimation tasks. A natural explanation is that PFNs have the property of statistical adaptivity, that is, they perform nearly as well as a method tailored to the true data-generating model for a heterogeneous set of models, while not being told which model the data comes from. We study how such adaptivity is learned in a controlled location-estimation problem. Each task is an unlabeled sample whose family is hidden: Gaussian data call for averaging, with error of order $n^{-1}$, whereas uniform data are best estimated from their extremes, at the faster rate $n^{-2}$. We also provide the example of a symmetric Gaussian mixture, for which a rate of $\sigma^2_n/n$ can be attained. On scalar inputs, softmax attention computes the derivative of the empirical cumulant-generating function. A single primitive therefore both supplies fe
    
[^24]: 通过有限阶松弛将路径梯度扩展到离散随机变量

    Extending Pathwise Gradients to Discrete Random Variables via Finite-Order Relaxation

    [https://arxiv.org/abs/2610.07786](https://arxiv.org/abs/2610.07786)

    提出一个通用框架，通过有限阶松弛为泊松等常见离散变量构建精确的路径梯度估计器，该估计器保留硬前向采样、无需温度调节、实现简单，且在所有可行解中唯一并最小化权重方差。

    

    路径梯度因其无偏、低方差且仅需单样本即可工作的特性，在连续随机变量中备受青睐。然而对于离散变量，路径恒等式通常无法对每个可微函数都精确成立。我们提出了一个通用框架，可为诸如泊松分布等一系列常见离散变量构建有限阶精确路径梯度估计器。该估计器是在所有对不超过某阶次的多项式均无偏的解中范数最小的解。所得估计器保留了硬前向采样，无需温度调节，且仅需几行代码即可实现。与其他可行解相比，我们的估计器是唯一的并能最小化权重方差；相比之下，先前的工作使用类别变量或增广表示来近似非类别变量，这会引入额外的方差和计算开销。为了理解逼近偏差……

    arXiv:2610.07786v1 Announce Type: new  Abstract: Pathwise gradients are preferred for continuous random variables because they are unbiased, low variance, and work with a single sample. For discrete variables, however, the pathwise identity cannot generally be exact for every differentiable function. We propose a general framework to construct finite-order exact pathwise gradient estimators for a range of common discrete variables such as Poisson. The estimator is the least-norm solution among all solutions that are unbiased for polynomials of degree at most. The resulting estimators preserve the hard forward sample, require no temperature tuning, and can be implemented in a few lines of codes. Against other admissible solutions, our estimator is unique and minimizes weight variance; in contrast, prior works use categorical variables or augmented representations to approximate non-categorical variables that induces excess variance and computations. To understand approximation bias for 
    
[^25]: 基于AI裁判的可信方法比较：顺序、批次与聚合效应下的估计与设计

    Trustworthy Method Comparison with AI Judges: Estimation and Design under Order, Batch, and Aggregation Effects

    [https://arxiv.org/abs/2610.07755](https://arxiv.org/abs/2610.07755)

    该论文提出用马尔可夫广义线性混合模型刻画LLM裁判的评估机制，证明了随机化取平均排名法的一致性条件及威廉姆斯方设计的效率优势，并揭示了组间比较中因模型非线性导致朴素平均可能得出错误结论的问题。

    

    大语言模型（LLM）正越来越多地被用作自动化AI评估的裁判。一种常见做法是将提示序列随机化并对所得分数取平均，但其统计有效性尚不明确。我们证明了LLM评估机制可以用一类马尔可夫广义线性混合模型（GLMM）来近似，这一结论得到了三个主要商业LLM样本外预测结果的支持。利用一阶马尔可夫GLMM，我们研究了排行榜排名和组间比较问题。对于排行榜排名，在温和的分离条件下，随机化后取平均的选择方法具有一致性；当被评估项质量接近时，采用威廉姆斯方设计可以提高效率。对于组间比较，由于响应模型的非线性，朴素平均方法可能对组级质量差异得出不一致的结论。实证结果进一步支持了所提出的基于模型的推断方法在一阶近似之外的有效性。

    arXiv:2610.07755v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as judges for automated AI evaluation. A common practice is to randomize prompt sequences and average the resulting scores, but its statistical validity remains unclear. We show that LLM evaluation mechanisms can be approximated by a class of Markov generalized linear mixed models (GLMMs), supported by out-of-sample predictions across three major commercial LLMs. Using a first-order Markov GLMM, we study leaderboard ranking and group comparison. For leaderboard ranking, randomize-and-average selection is consistent under a mild separation condition, and a Williams square design can improve efficiency when item qualities are close. For group comparison, naive averaging can yield inconsistent conclusions about differences in group-level quality because of the response model's nonlinearity. Empirical results further support the validity of the proposed model-based inference beyond the fir
    
[^26]: 对抗训练的线性Transformer是高斯混合分布的最优鲁棒上下文学习者

    Adversarially Trained Linear Transformers Are Optimal Robust In-Context Learners for Gaussian Mixtures

    [https://arxiv.org/abs/2610.07754](https://arxiv.org/abs/2610.07754)

    经过跨任务对抗预训练的线性Transformer无需额外训练，即可通过上下文学习将鲁棒性迁移到未见过的任务，并渐近达到高斯混合分类任务的最优鲁棒贝叶斯误差。

    

    对抗训练是对抗攻击最可靠的防御手段之一，但其高昂的计算成本通常需要针对每个任务重新付出。鲁棒基础模型提供了一种有前景的替代方案：只需对模型进行一次对抗性预训练，然后通过轻量级适配将其鲁棒性迁移到下游任务。然而，一个根本性的问题仍未解决：预训练中获得的鲁棒性能否在无需进一步对抗训练的情况下迁移到未见过的任务？在本研究中，我们对这一问题给出了肯定的答案。一个经过大规模对抗预训练的单一模型，无需额外的任务特定训练即可在新任务上实现最优鲁棒性。具体而言，我们证明，对于一类高斯混合分类任务，经过跨任务对抗训练的足够深的线性Transformer，可以通过对干净样本的上下文学习，渐近地达到在未见过的任务上的鲁棒贝叶斯误差。

    arXiv:2610.07754v1 Announce Type: new  Abstract: Adversarial training is one of the most reliable defenses against adversarial attacks, but its high computational cost must generally be paid anew for each task. Robust foundation models offer a promising alternative: adversarially pretrain a model once and then transfer its robustness to downstream tasks through lightweight adaptation. However, a fundamental question remains open: can robustness acquired during pretraining transfer to unseen tasks without further adversarial training? In this study, we answer this question affirmatively. A single model adversarially pretrained at scale can achieve optimal robustness on new tasks without additional task-specific training. Specifically, we show that, for a family of Gaussian-mixture classification tasks, a sufficiently deep linear transformer adversarially trained across tasks can asymptotically attain the robust Bayes error on previously unseen tasks through in-context learning from clea
    
[^27]: 基于调和权重的高维在线校准

    High-dimensional online calibration from harmonic weights

    [https://arxiv.org/abs/2610.07740](https://arxiv.org/abs/2610.07740)

    本文提出了一个基于调和权重、对过去结果进行调和平滑的简单在线校准算法，首次在高维多结果预测中以 $d^{O(1/\varepsilon)}$ 轮实现 $\varepsilon$-校准，将此前结果的维度依赖性指数级降低。

    

    我们研究了在任意凸集 $Y\subseteq\mathbb{R}^d$ 上、相对于任意误差范数 $\|\cdot\|_{L}$ 的多维预测在线校准问题。对于同时预测 $d$ 个二元结果（$Y=[0,1]^d$）的情形，我们给出了首个在每个固定精度下都能以关于 $d$ 为多项式的轮数实现 $\varepsilon$-校准的算法。该算法需要 $d^{O(1/\varepsilon)}$ 轮，相比此前界中的维度依赖性实现了指数级改进。对于多类别预测（$Y=\Delta_d$），我们获得了相同的 $d^{O(1/\varepsilon)}$ 速率，改进了 Peng 以及 Fishelson 等人给出的 $d^{\widetilde{O}(1/\varepsilon^2)}$ 界。我们的算法非常简单：在每一轮，它输出一个对过去结果进行调和平滑后的调和加权分布。同一个算法适用于所有预测集和范数。更一般地，它在 $\exp(O(\gamma(Y,L)/\varepsilon$……（原文摘要在此处截断）轮后即可实现 $\varepsilon$-校准。

    arXiv:2610.07740v1 Announce Type: cross  Abstract: We study the online calibration of multidimensional forecasts over an arbitrary convex set $Y\subseteq\mathbb{R}^d$ relative to an arbitrary error norm $\|\cdot\|_{L}$. For forecasting $d$ binary outcomes simultaneously ($Y=[0,1]^d$), we give the first algorithm that achieves $\varepsilon$-calibration in a number of rounds that is polynomial in $d$ for every fixed accuracy. It requires $d^{O(1/\varepsilon)}$ rounds, exponentially improving the dimension dependence of previous bounds. For multi-class forecasting ($Y=\Delta_d$), we obtain the same $d^{O(1/\varepsilon)}$ rate, improving the $d^{\widetilde{O}(1/\varepsilon^2)}$ bounds of Peng and Fishelson et al.   Our algorithm is simple: on each round, it outputs a harmonically weighted distribution over harmonically smoothed past outcomes. The same algorithm works for every forecast set and norm. More generally, it achieves $\varepsilon$-calibration after $\exp(O(\gamma(Y,L)/\varepsilon
    
[^28]: 多臂老虎机的纳什社会福利：轨迹级期望与高概率遗憾

    Nash Social Welfare for Multi Armed Bandits: Trajectory-wise Expected and High Probability Regret

    [https://arxiv.org/abs/2610.07737](https://arxiv.org/abs/2610.07737)

    该论文提出了一种新的“轨迹级纳什遗憾”度量，通过先对完整奖励样本路径取几何平均再求期望，弥补了现有度量忽略各轮奖励联合分布的缺陷，并借助詹森不等式证明其严格强于原有度量，从而更忠实地体现纳什社会福利的公平性目标。

    

    我们研究在纳什社会福利（NSW）目标下的公平多臂老虎机问题，该目标通过累积奖励的几何平均来衡量性能。现有工作将纳什遗憾定义为 $\mathrm{NR}_T = \mu^\star - (\prod_{t=1}^T \mathbb{E}\mu_{I_t})^{1/T}$，其中 $\mu_{I_t}$ 是推荐臂 $I_t$ 的平均奖励，$T$ 是时间范围。由于该定义将几何平均应用于每轮的边际期望，忽略了各轮奖励之间的联合分布，未能在轨迹层面体现NSW的公平性动机。我们提出轨迹级纳什遗憾 $\widetilde{\mathrm{NR}}_T = \mu^\star - \mathbb{E}[(\prod_{t=1}^T \mu_{I_t})^{1/T}]$，它在取期望之前先对完整样本路径计算几何平均，从而更忠实地刻画NSW的公平性。根据詹森不等式，$\widetilde{\mathrm{NR}}_T \geq \mathrm{NR}_T$，这使其成为一个严格更强的度量指标。我们还引入了（摘要在此处截断）

    arXiv:2610.07737v1 Announce Type: cross  Abstract: We study fair multi-armed bandits under the Nash Social Welfare (NSW) objective, which measures performance via the geometric mean of accumulated rewards. Existing work defines Nash regret as $\mathrm{NR}_T = \mu^\star - (\prod_{t=1}^T \mathbb{E}\mu_{I_t})^{1/T}$, where $\mu_{I_t}$ is the mean reward of the recommended arm $I_t$ and $T$ is the horizon. Since it applies the geometric mean to per-round marginal expectations, it ignores the joint distribution of rewards across rounds, leaving the NSW fairness motivation unaddressed at the trajectory level. We propose \emph{trajectory-wise Nash regret} $\widetilde{\mathrm{NR}}_T = \mu^\star - \mathbb{E}[(\prod_{t=1}^T \mu_{I_t})^{1/T}]$, which computes the geometric mean over complete sample paths before taking expectations, capturing NSW fairness more faithfully. By Jensen's inequality, $\widetilde{\mathrm{NR}}_T \geq \mathrm{NR}_T$, making it a strictly stronger metric. We also introduce
    
[^29]: 体积抽样岭回归的精确校准与尖锐风险几何

    Exact Calibration and Sharp Risk Geometry for Volume-Sampled Ridge Regression

    [https://arxiv.org/abs/2610.07721](https://arxiv.org/abs/2610.07721)

    本文为体积抽样岭回归建立了精确的惩罚校准理论（当且仅当抽样行数超过目标有效维度时该惩罚存在），并通过严格的扇形不等式刻画了中心化协方差风险的尖锐上界以及所有最大化响应的结构。

    

    我们研究从固定设计中恰好抽取 $s$ 个不同行进行岭回归的问题。响应是固定的，只有所抽取的子集是随机的。行列式法则与所选岭拟合共享同一个正定惩罚项。借助已建立的均值恒等式和指数族对偶性，我们给出了唯一的惩罚项，使其在期望意义上匹配一个指定的全数据岭拟合；该惩罚项当且仅当 $s$ 超过目标的有效维度时才存在。我们的主要结果关注以全数据惩罚损失归一化的中心化协方差风险。对于平衡的带符号坐标副本，一个严格的扇形不等式给出了从维度到行数减一之间的每一个预算水平下的尖锐风险以及所有最大化响应。这一结论对任何非零半正定查询均成立。当目标与查询固定时，最大化响应空间在这些预算水平之间保持不变。对于一般设计，我们刻画了留一包络的达到条件。对于现有的实等角……

    arXiv:2610.07721v1 Announce Type: cross  Abstract: We study ridge regression from exactly $s$ distinct rows of a fixed design. Responses are fixed, and only the subset is random. The determinant law and selected ridge fit share one positive definite penalty. Established mean identities and exponential-family duality give the unique penalty that matches a prescribed full-data ridge fit in expectation. It exists exactly when $s$ exceeds the target's effective dimension. Our main result concerns centered covariance risk normalized by full-data penalized loss. For balanced signed coordinate replicas, a strict sector inequality gives the sharp risk and all maximizing responses at every budget from the dimension to one below the row count. This holds for any nonzero positive semidefinite query. With the target and query fixed, the maximizing response space is unchanged across these budgets. For general designs, we characterize attainment of a leave-one-out envelope. For existing real equiang
    
[^30]: 次高斯数据上测度到测度Transformer的稳定性

    Stability of Measure-to-Measure Transformers on Sub-Gaussian Data

    [https://arxiv.org/abs/2610.07717](https://arxiv.org/abs/2610.07717)

    本文从数学上证明了Transformer将次高斯数据映射为次高斯输出且关于1-Wasserstein距离具有Hölder连续性，由此建立了经验近似下的误差传播估计，并揭示了交叉注意力机制均值场类似物的不同正则性与样本复杂度。

    

    Transformer在各个领域展现了令人瞩目的实证成功，但其理论基础仍相对欠缺。本工作对由Transformer定义的测度到测度算子进行了数学研究。我们证明Transformer将次高斯输入映射为次高斯输出，这保证了softmax算子任意长度复合的良定性。随后我们证明，在适当的次高斯输入空间上，Transformer关于1-Wasserstein距离具有Hölder连续性。这使我们能够建立关于Transformer在次高斯输入与其经验近似之间误差传播的估计。我们还研究了交叉注意力机制的均值场类似物，它是一个从概率测度对到单个概率测度的算子。我们证明交叉注意力表现出不同的Hölder正则性与样本复杂度。

    arXiv:2610.07717v1 Announce Type: cross  Abstract: Transformers have exhibited impressive empirical success across various domains, but their theoretical foundations remain less developed. This work constitutes a mathematical study of the measure-to-measure operators defined by transformers. We show that transformers map sub-Gaussian inputs to sub-Gaussian outputs; this ensures that taking arbitrary-length compositions of the softmax operator is well-defined. We then show that transformers are H\"older continuous with respect to the 1-Wasserstein distance on appropriate spaces of sub-Gaussian inputs. This allows us to establish estimates on the error propagation along a transformer between a sub-Gaussian input and its empirical approximation. We also study a mean-field analog of the cross-attention mechanism, which is an operator from a pair of probability measures to a single probability measure. We show that cross-attention exhibits different H\"older regularity and sample-complexity
    
[^31]: 均匀离散扩散模型对于估计小有效支撑大小的分布是极小极大最优的

    Uniform Discrete Diffusion Models are Minimax Optimal for Estimating Distributions with Small Effective Support Size

    [https://arxiv.org/abs/2610.07655](https://arxiv.org/abs/2610.07655)

    该论文证明了均匀离散扩散模型在估计有效支撑较小的分布时达到极小极大最优，其统计误差由依赖于样本量的有效支撑大小而非环境空间规模决定。

    

    离散扩散模型已成为在离散乘积空间上进行生成建模的一种在实践中非常成功的框架，但其统计泛化性质仍未被充分理解。文本或生物序列等离散的真实世界数据，由于语义或物理约束，往往集中在天文数字般庞大的环境空间中的一小部分上；然而现有的理论误差界无法捕捉这种分布结构，而是随环境空间的大小进行缩放，导致误差界几乎空洞无用。我们针对均匀离散扩散——与掩码扩散并列为两大主流离散扩散范式之一——填补了这一空白，推导出由有效支撑大小 $s_n(P_0)$ 决定的统计保证，该度量是一种依赖于样本量的分布复杂度刻画。给定来自 $[K]$ 上未知数据分布 $P_0$ 的 $n$ 个独立同分布样本……

    arXiv:2610.07655v1 Announce Type: cross  Abstract: Discrete diffusion models have emerged as a practically successful framework for generative modeling on discrete product spaces, yet their statistical generalization properties remain poorly understood. Discrete real-world data such as text or biological sequences often concentrate on a small fraction of the astronomically large ambient space because of semantic or physical constraints, but existing bounds fail to capture this distributional structure and instead scale with the size of the ambient space, giving rise to almost vacuous error bounds. We address this gap for uniform discrete diffusion, one of the two dominant discrete diffusion paradigms alongside masking diffusion, by deriving statistical guarantees governed by the effective support size $s_n(P_0)$, a sample-size-dependent measure of distributional complexity. Given $n$ independent and identically distributed (i.i.d.) samples from an unknown data distribution $P_0$ on $[K
    
[^32]: 逐元素独立同分布重尾数据上经验风险最小化的渐近分析

    Asymptotic Analysis of Empirical Risk Minimization on Entry-wise i.i.d. Heavy-Tailed Data

    [https://arxiv.org/abs/2610.07637](https://arxiv.org/abs/2610.07637)

    本文通过引入函数序参量并运用复制方法，首次在比例高维极限下精确刻画了对称α-稳定重尾数据上线性回归经验风险最小化的泛化误差，并建立了重尾普适性定律与相应的标度律。

    

    许多现实世界的数据集出现异常大值的频率远超高斯模型的预测。重尾分布能够刻画这种现象，但在重尾分布下评估学习性能仍然具有挑战性，因为稀有的大幅特征元素即使在高维情形下也保持着不可忽略的影响。即使在带有逐元素独立同分布对称 $\alpha$-稳定数据的线性回归经验风险最小化这一经典设定中，预测性能的精确渐近刻画也一直缺失。在这项工作中，我们引入了一个函数序参量来描述与每个系数相关的随机有效问题。利用复制方法，我们在样本量与特征维度以固定比例发散的比例高维极限下完全刻画了泛化误差。此外，该分析建立了一个重尾普适性定律，以及将典型误差与标度律相联系的规律（摘要原文在此处被截断）。

    arXiv:2610.07637v1 Announce Type: cross  Abstract: Many real-world datasets exhibit unusually large values far more frequently than predicted by Gaussian models. Heavy-tailed distributions capture this behavior, yet evaluating learning performance under them remains challenging because rare, large feature entries retain non-vanishing effects even in high dimensions. Even in the canonical setting of empirical risk minimization for linear regression with entry-wise i.i.d. symmetric $\alpha$-stable data, a precise asymptotic characterization of prediction has been lacking. In this work, we introduce a functional order parameter that describes the random effective problem associated with each coefficient. Using the replica method, we fully characterize the generalization error in the proportional high-dimensional limit where the sample size and feature dimension diverge at a fixed ratio. Additionally, this analysis establishes a heavy-tail universality law, scaling laws relating typical er
    
[^33]: 超越 $T^{2/3}$ 的序贯校准问题的显式渐近界

    Explicit Asymptotic Bounds for Sequential Calibration Beyond $T^{2/3}$

    [https://arxiv.org/abs/2610.07623](https://arxiv.org/abs/2610.07623)

    该论文提出新的两阶段递归标记策略并改进归约方法，首次为序贯校准问题建立了超越 $T^{2/3}$ 的显式渐近界 $O(T^{0.662942288})$。

    

    当预测概率与经验结果频率相匹配时，概率预测被称为校准的：在被赋予概率 $p$ 的事件中，我们希望正结果的占比接近 $p$。我们研究二元结果的序贯预测问题。Foster 和 Vohra 建立的关于期望累积 $\ell_1$ 校准误差的经典 $O(T^{2/3})$ 界保持了二十多年，直到 Dagan 等人将指数 $2/3$ 降低了一个未具体指明的常数。我们为“符号保持-复用”博弈建立了一种新的两阶段递归标记策略，对于所有空间和时间的选择都能得到 $O(n^{\alpha}t^\beta)$ 的界。随后，我们通过修改 Dagan 等人的等价性，使其仅使用 $O(\log T)$ 个“符号保持-复用”博弈实例，从而锐化了从符号保持上界到校准的归约。这使我们能够建立一个显式的界 $O(T^{0.662942288})$。

    arXiv:2610.07623v1 Announce Type: cross  Abstract: Probability forecasts are calibrated when predicted probabilities match empirical outcome frequencies: among events assigned a probability $p$, we'd hope that the fraction of positive outcomes is close to $p$. We study the problem of sequential forecasting of binary outcomes. The classical $O(T^{2/3})$ bound on expected cumulative $\ell_1$-calibration error established by Foster and Vohra stood for over two decades until Dagan et al. reduced the exponent $2/3$ by an unspecified constant.   We establish a new two-phase recursive labeling strategy for the sign-preservation-with-reuse game that yields the bound $O(n^{\alpha}t^\beta)$ for all choices of space and time. We then sharpen the reduction from upper bounds on sign preservation to calibration by modifying the equivalence of Dagan et al. to use only $O(\log T)$ instances of the sign-preservation-with-reuse game. This lets us establish an explicit bound of $O(T^{0.662942288})$, the 
    
[^34]: 藤Copula VAR：从递归边际到联合预测推断

    Vine Copula VAR:From Recursive Margins to Joint Forecast Inference

    [https://arxiv.org/abs/2610.07589](https://arxiv.org/abs/2610.07589)

    本文在稳定的藤Copula VAR框架下，推导出历史递归估计误差与终端预测的联合影响，并提出一个协方差估计量，可为固定单侧事件概率提供渐近有效的重复样本预测区间。

    

    联合事件预测通常将基于过去预测误差的依赖性估计与新估计的边际分布相结合。当每个历史误差保留其发布日期时可获得的边际拟合时，推断必须考虑一系列相互重叠的估计误差。我们在一个具有正态创新项边际、且采用固定的、正确设定的Gaussian或正Clayton藤结构的稳定Vine Copula VAR中，推导出这些历史误差与终端预测估计的联合影响。利用截距恒等式和稳定的VAR滤波器，历史修正被简化为调和加权的创新项矩，而终端斜率的不确定性仍然保留。由此得到的协方差估计量，能够在已实现的预测状态下，为固定的单侧事件概率提供渐近有效的重复样本区间。在Gaussian子模型中，相对于在同一……

    arXiv:2610.07589v1 Announce Type: new  Abstract: Joint-event forecasts often combine a dependence estimate based on past forecast errors with newly estimated marginal distributions. When each historical error retains the marginal fit available at its issue date, inference must account for an overlapping sequence of estimation errors. We derive their joint influence with the terminal forecast estimates in a stable Vine Copula VAR with normal innovation margins and a fixed, correctly specified Gaussian or positive Clayton vine. An intercept identity and the stable VAR filter reduce the historical correction to harmonically weighted innovation moments, while terminal slope uncertainty remains. The resulting covariance estimator gives asymptotically valid repeated-sample intervals for fixed one-sided event probabilities at the realized forecast state. In the Gaussian submodel, retaining issued transforms adds a positive semidefinite covariance term relative to refitting margins on the same
    
[^35]: 学习GFlowNets混合体

    Learning a Mixture of GFlowNets

    [https://arxiv.org/abs/2610.07562](https://arxiv.org/abs/2610.07562)

    提出了一个描述GFlowNets混合体的通用理论框架，将其细分为连续索引（CI）和离散索引（DI）两类：前者通过随机特征扩展与谱移位可证明地提升采样器的表达能力并降低学习不稳定性，后者统一了已有训练方法并支撑了新提出的分层条件化（SC）GFlowNets。

    

    学习一组GFlowNets集成以从离散目标分布中进行采样，已成为比单一采样器实现更好的状态空间探索和收敛性的常用方法。然而，这些方法通常会给基础模型带来较大的运行时开销，且它们之间的概念联系仍不明确。为解决这一问题，我们首先提出了一个用于描述GFlowNets混合体的通用理论框架，并将其细分为连续索引（CI）和离散索引（DI）两类集合。一方面，我们证明CI GFlowNets可以通过随机特征扩展的视角来解释，在图结构任务中可证明地提升采样器的表达能力，并通过谱移位降低学习的不稳定性。另一方面，我们证明DI GFlowNets涵盖了此前已有的GFlowNet训练方法，并为新提出的分层条件化（SC）GFlowNets奠定了基础。

    arXiv:2610.07562v1 Announce Type: cross  Abstract: Learning an ensemble of GFlowNets to sample from a discrete target distribution has become a common approach for achieving better state space exploration and convergence than that of a monolithic sampler. However, these methods often add a substantial runtime overhead to the base model, and their conceptual connection remains elusive. To address this, we first propose a general-purpose theoretical framework for describing a mixture of GFlowNets, which we specialize into continuously (CI) and discretely indexed (DI) collections. On the one hand, we show CI GFlowNets can be interpreted through the lens of a random features expansion, provably boosting the sampler's expressivity in graph-structured tasks and reducing learning instability via spectral shifting. On the other hand, we demonstrate DI GFlowNets encompass prior approaches for GFlowNet training and provide the foundation for the newly proposed Stratum-Conditioned (SC) GFlowNets.
    
[^36]: 高维学习高斯混合模型时，梯度 EM 是否必须要求 $\sqrt{d}$ 量级的分离度？

    Is $\sqrt{d}$ Separation Necessary for Gradient EM to Learn Gaussian Mixtures in High Dimensions?

    [https://arxiv.org/abs/2610.07551](https://arxiv.org/abs/2610.07551)

    本文证明在高维学习高斯混合模型时，梯度 EM 全局收敛所需的 $\Omega(\sqrt{d})$ 分离度条件是不可避免的，即这一对维度的依赖性无法被去除。

    

    使用期望最大化（EM）算法及其基于梯度的变体来学习高斯混合模型（GMM）是机器学习中的一个基本问题。众所周知，在精确参数化设置（即分量数量与真实 GMM 的分量数量相匹配）下，随机初始化的（梯度）EM 无法学习多分量 GMM。最近，在过参数化设置（即使用更多分量）下，只要真实分量之间分离良好，梯度 EM 的全局收敛性已经得到证明。特别地，真实分量之间的最小分离度需要达到 $\Omega(\sqrt{d})$ 的量级，其中 $d$ 为维度。在本文中，我们证明在高维设置下这种对维度的依赖是不可避免的。具体而言，我们考虑了一种混合 EM 算法，其对混合权重采用标准 EM 更新，对分量均值采用梯度 EM 更新。

    arXiv:2610.07551v1 Announce Type: cross  Abstract: Learning Gaussian mixture models (GMMs) using the Expectation-Maximization (EM) algorithm and its gradient-based variants is a fundamental problem in machine learning. It is known that randomly initialized (gradient) EM fails to learn multi-component GMMs in the exact-parameterized setting, where the number of components matches that of the ground-truth GMM. Recently, global convergence of gradient EM has been established in the over-parameterized setting, where more components are used, provided that the ground-truth components are well separated. In particular, the minimum separation between ground-truth components is required to scale as $\Omega(\sqrt{d})$, where $d$ is the dimension. In this paper, we show that this dimensional dependence is unavoidable in high-dimensional settings. Specifically, we consider a hybrid EM algorithm that uses standard EM updates for the mixing weights and gradient EM updates for the component means. F
    
[^37]: 无需顶点对应关系的随机图双样本检验

    Two-Sample Testing for Random Graphs without Vertex Correspondence

    [https://arxiv.org/abs/2610.07503](https://arxiv.org/abs/2610.07503)

    该论文首次建立了顶点无对应情形下图总体双样本检验的最优样本复杂度理论，证明对于保持度不变的两块结构差异，每组需要约 t^{-3} 张图，带符号三角形计数可达到该最优速率，且未对齐相比对齐情形需多付出 t^{-2} 阶的样本代价。

    

    两群图之间常常需要在顶点没有任何对应关系的情况下进行比较，例如当网络来自不同的社区时，或者在评估图生成模型与留出图时。我们研究了这种未对齐的双样本检验需要多少张图，以及哪些图统计量能够检测哪些类型的差异。对于 Erdős–Rényi 零假设以及保持每个期望度不变的植入式两块差异，我们证明当每张图的信噪比 t<1 时，每组 m≍t^{-3} 张图既是必要也是充分的。带符号三角形计数可以达到这一速率，且该下界对任意图规模都成立。当顶点对齐时，m≍t^{-1} 张图就足够了，因此未对齐会带来 t^{-2} 阶的代价因子。当三角形信号相互抵消时，速率变为 t^{-4}，此时需要借助 4-环。基于树构建的统计量在两种情形下具有完全相同的期望……

    arXiv:2610.07503v1 Announce Type: cross  Abstract: Two populations of graphs often have to be compared without any correspondence between their vertices, for instance when networks come from different communities, or when a graph generative model is evaluated against held-out graphs. We study how many graphs such an unaligned two-sample test needs, and which graph statistics can detect which differences. For an Erd\H{o}s--R\'enyi null and a planted two-block difference that leaves every expected degree unchanged, we show that $m\asymp t^{-3}$ graphs per group are necessary and sufficient when the per-graph signal-to-noise ratio is $t<1$. Signed triangle counts attain this rate, and the lower bound holds for every graph size. With aligned vertices $m\asymp t^{-1}$ graphs suffice, so misalignment costs a factor of order $t^{-2}$. When the triangle signal cancels, the rate becomes $t^{-4}$ and $4$-cycles are needed. Statistics built from trees have exactly the same expectation under both 
    
[^38]: Muon 需要细粒度的谱整形吗？

    Does Muon Need Fine-Grained Spectral Shaping?

    [https://arxiv.org/abs/2610.07497](https://arxiv.org/abs/2610.07497)

    本文提出 BulkBoost 双频段谱重加权框架，表明 Muon 并不需要细粒度的谱整形，只需将奇异谱粗略地划分为噪声主体和高增益尖峰两个频段并进行重加权即可提升优化效果。

    

    Muon 将当前梯度与历史梯度结合为矩阵动量。对于 M=UΣV^T，理想化的极分解更新 Q=UV^T 会给每个奇异方向赋予相同的权重，我们将其称为“平坦谱形”。近期一些优化器用细粒度的谱映射取代这种平坦谱形，为每个方向赋予各自的增益。我们探究 Muon 更新到底需要多少这样的谱细节。我们的谱诊断显示，约 94%–97% 的测量奇异模态位于估计的噪声边缘之下，但它们整体上与参考梯度呈正相关对齐。我们提出了 BulkBoost，一个双频段谱重加权框架，包含固定秩与噪声校准两种变体。后者利用拆分小批量梯度差异，为 Muon 的 Nesterov 输入校准一个 Marchenko–Pastur 参考边缘，从而将边缘之下的主体部分与边缘之上的尖峰部分分离开来。两种变体都通过一次……（增加主体部分的相对权重）［摘要在此处被截断］

    arXiv:2610.07497v1 Announce Type: new  Abstract: Muon combines current and past gradients into matrix momentum. For $M=U\Sigma V^\top$, the idealized polar update $Q=UV^\top$ gives every singular direction the same weight. We refer to this as the flat profile. Several recent optimizers replace this flat profile with fine-grained spectral maps that give each direction its own gain. We ask how much of this spectral detail a Muon update needs. Our spectral diagnostics show that approximately $94$--$97\%$ of measured singular modes lie below an estimated noise edge, yet collectively align positively with a reference gradient.   We introduce BulkBoost, a two-band spectral reweighting framework with fixed-rank and noise-calibrated variants. The latter uses split-minibatch gradient differences to calibrate a Marchenko--Pastur reference edge for Muon's Nesterov input, separating the bulk below the edge from the spikes above it. Both variants increase the bulk's relative weight through one shar
    
[^39]: 数据同化与机器学习的交汇

    The interface of data assimilation and machine learning

    [https://arxiv.org/abs/2610.07496](https://arxiv.org/abs/2610.07496)

    本文综述了数据同化与机器学习这一新兴交叉领域的主要主题和方法，探讨了两者结合对混沌系统（如大气）进行最优状态估计的应用前景。

    

    数据同化（DA）是将模型的预报结果与观测数据相结合，从而最优地估计系统状态的过程。这对于混沌系统（如大气）至关重要，因为如果不持续同化观测数据，模型将很快失去预测能力。世界各地的业务预报中心通常每6小时进行一次数据同化。本文讨论了机器学习（ML）与数据同化的交汇领域。这仍然是一个新兴且快速发展的领域，本文试图对其中一些主要主题和方法进行概述。

    arXiv:2610.07496v1 Announce Type: cross  Abstract: Data assimilation (DA) is the process of combining forecasts from a model with observations in order to optimally estimate the state of a system. This is critical for chaotic systems, such as the atmosphere, since if observations are not continually assimilated the model will quickly lose skill. DA is routinely performed (usually every 6 hours) at operational forecasting centres around the world.   In this article we discuss the interface of machine learning (ML) and DA. This is still an emerging and quickly developing field, and this article tries to give an overview of some of the main topics and methods.
    
[^40]: 结构而非信念：组合半老虎机中基于LLM衍生协方差的相关汤普森采样

    Structure, Not Belief: Correlated Thompson Sampling from LLM-Derived Covariance in Combinatorial Semi-Bandits

    [https://arxiv.org/abs/2610.07470](https://arxiv.org/abs/2610.07470)

    提出一种对组合汤普森采样的最小改动方法，仅查询LLM一次将臂划分转化为相关协方差矩阵来引导探索，理论证明相比独立采样可获得有限时域内√(d/K)的遗憾改进，实验中遗憾降低19%。

    

    组合汤普森采样（CTS）为每个臂独立抽取后验样本，因此其探索动态忽略了臂之间的任何关联。我们研究了对此动态的一个最小改动：仅向大语言模型（LLM）查询一次以获得臂的划分，该划分通过聚类秩上的RBF核转化为正定相关矩阵Σ，每轮后验样本以协方差Σ抽取，而Beta后验仅依据真实奖励进行更新，因此LLM塑造的是采样器的移动方式，而非其信念内容。我们为理想化的高斯采样器给出了一个自洽的贝叶斯遗憾界，其信息增益分解为来自K聚类结构的K log T项和增长至d log T的岭回归项：相比独立采样的√(d/K)遗憾改进是一种有限时域的瞬态效应，仅在簇内相关性趋于1时才精确成立。该相关采样器将遗憾降低了19%。

    arXiv:2610.07470v1 Announce Type: cross  Abstract: Combinatorial Thompson sampling (CTS) draws independent posterior samples for every arm, so its exploration dynamics ignore any relation among arms. We study a minimal change to those dynamics: an LLM is queried once for a partition of the arms, the partition becomes a positive-definite correlation matrix $\Sigma$ through an RBF kernel on cluster ranks, and the per-round posterior sample is drawn with covariance $\Sigma$ while the Beta posteriors are updated from real rewards only, so the LLM shapes how the sampler moves, not what it believes. We give a self-contained Bayesian regret bound for the idealized Gaussian sampler whose information gain splits into a $K\log T$ term from the $K$-cluster structure and a ridge term that grows to $d\log T$: the $\sqrt{d/K}$ improvement over independent sampling is a finite-horizon transient, exact only as the within-cluster correlation tends to one. The correlated sampler reduces regret by 19% ov
    
[^41]: 面向降低参与者负担的代价高效时序预测的主动特征获取

    Active Feature Acquisition for Cost-Efficient Temporal Prediction with Reduced Participant Burden

    [https://arxiv.org/abs/2610.07452](https://arxiv.org/abs/2610.07452)

    该论文提出纵向主动特征获取（LAFA）方法，通过学习一种策略在每个时间点仅选择性地采集最优的条目动态子集，从而在降低参与者负担、减少无应答与流失风险的同时，保持对心理病理结果的准确预测能力。

    

    arXiv:2610.07452v1 公告类型：cross 摘要：准确预测病理结果是心理学中的一个核心问题。为此，心理学家通常会收集密集的纵向数据。然而，在此类研究中，为了实现准确预测而获取大量变量的愿望，往往与最小化参与者负担的需求相冲突。每次测量获取更多变量可以带来更好的预测效果，但过多的测量会增加无应答和被试流失的风险。纵向主动特征获取（Longitudinal Active Feature Acquisition, LAFA）是一种解决这一难题的规范化方法。LAFA 不再要求参与者在每次测量时回答所有条目，而是生成一种策略，在每个时间点最优地选择需要获取的条目动态子集，同时保留对特定结果进行预测的能力。然而，现有的 LAFA 方法大多基于神经网络（NN），在实际应用中难以解释。在此……

    arXiv:2610.07452v1 Announce Type: cross  Abstract: Accurate forecasting of pathological outcomes is a central problem in psychology. To do so, psychologists often collect intensive longitudinal data. However, in such studies, the desire to acquire a large number of variables for the sake of accurate prediction is often counteracted by the need to minimize participant burden. Acquiring more variables per occasion can yield better predictions, but having too many acquisitions increase the risk of non-response and attrition. Longitudinal Active Feature Acquisition (LAFA) is a principled approach to resolve this conundrum. Instead of requiring responses to every item at every acquisition occasion, LAFA produces a policy that seeks to optimally select dynamic subsets of items to be acquired at each timepoint while preserving our ability to forecast a specific outcome. However, existing LAFA methods are mostly based on Neural Networks (NN) that are difficult to interpret in practice. In this
    
[^42]: 基于稀疏RKHS流形的函数空间贝叶斯优化

    Bayesian Optimization on Function Spaces via Sparse RKHS Manifolds

    [https://arxiv.org/abs/2610.07417](https://arxiv.org/abs/2610.07417)

    提出L0MO方法，通过在RKHS中由核函数稀疏表示构成的流形子集上搜索，并同时优化核的位置与系数，从而实现函数空间中的贝叶斯优化，并为现有FBO方法提供了统一视角。

    

    贝叶斯优化（BO）已成为最小化向量输入黑箱函数的成熟方法论。然而，这一参数向量往往源于对本质上是函数关系的离散化。近期有多篇文章研究了函数贝叶斯优化（FBO）的设定，其中待优化的变量不是有限维向量空间中的元素，而是无限维函数空间中的函数。在本工作中，我们提出 $L^0$ 流形优化（L0MO），这是一种简单的FBO方法，它在再生核希尔伯特空间（RKHS）中由具有核函数稀疏表示的函数构成的子集上进行搜索，同时优化核函数的位置及其系数。我们详细讨论了本方法与现有方法之间的关系，提供了一个统一的视角来审视先前的工作。为了将我们的方法与最先进的技术进行评估比较，

    arXiv:2610.07417v1 Announce Type: cross  Abstract: Bayesian Optimization (BO) has become an established methodology for minimizing black-box functions of a vector input. Often, however, this parameter vector arises from the discretization of an inherently functional relationship. Several recent articles have considered the Functional Bayesian Optimization (FBO) setting, in which the variable to be optimized is not a member of a finite dimensional vector space, but rather an infinite dimensional function space. In this work, we propose $L^0$ Manifold Optimization (L0MO), a simple approach to FBO which searches the subset of a Reproducing Kernel Hilbert Space (RKHS) consisting of functions with a sparse representation in the kernel functions, optimizing both the kernel locations and their coefficients. We discuss in detail the relationship between our method and existing ones, providing a unifying lens through which to view prior works. To assess our method against the state of the art, 
    
[^43]: 关于使用聚合标准化流链进行复杂仿真模型似然近似与推断的视角研究

    A perspective note on likelihood approximation and inference for complex simulation models using a chain of aggregated normalizing flows

    [https://arxiv.org/abs/2610.07391](https://arxiv.org/abs/2610.07391)

    提出了一种基于n级聚合标准化流链的似然近似新方法，通过按顺序估计各组双射变换参数，为复杂仿真模型的大规模数据分析、假设检验和不确定性量化提供了可扩展且高效的解决方案。

    

    我们在基于仿真的推断框架内，针对似然近似问题提出了一种新视角，该视角促进了面向大规模数据分析的可扩展、可控的仿真流程，支持在高维空间中进行高效的参数空间探索或平滑插值，从而为假设检验和不确定性量化提供有效的统计处理手段。特别地，我们考虑一种由 $n$ 个聚合标准化流组成的链式似然近似方案，其中一组来自前向复杂仿真模型的预先复制观测数据集首先通过第一组双射变换，随后再依次通过后续的多组双射变换。这里，我们假设对于任意 $k \in \{1,\,2, \ldots, n\}$，前 $k$ 组双射变换所对应的参数按某种最优性意义依次进行估计……

    arXiv:2610.07391v1 Announce Type: cross  Abstract: We present a new perspective on the problem of likelihood approximation within the framework of simulation-based inference that promotes scalable and controllable simulation routines for large-scale data analysis, allows efficient parameter space exploration or smooth interpolation in high-dimensions and, thus, supports valid statistical treatments of hypothesis testings as well as uncertainty quantification. In particular, we consider a chain of $n$-aggregated normalizing flows for likelihood approximation scheme, where a set of upfront replicated observation datasets from the forward complex simulation model pass through the first set of bijective transformations, and then subsequently pass to the other sets of bijective transformations. Here, we assume that, for any $k \in \{1,\,2, \ldots, n\}$, the parameters corresponding to the first $k$ sets of bijective transformations are estimated sequentially, in some sense of optimality, fo
    
[^44]: DeepAJM：面向不规则采样数据的深度关联联合模型

    DeepAJM: Deep Association Joint Model for Irregularly Sampled data

    [https://arxiv.org/abs/2610.07388](https://arxiv.org/abs/2610.07388)

    提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。

    

    联合模型同时建模纵向结局与生存结局，利用患者纵向轨迹中的模式来改进生存结局的预测。然而，经典的参数化联合模型依赖于固定的参数假设，在模型误设和样本量较小的情况下容易产生偏差。我们提出了一种深度联合模型 DeepAJM，它不需要任何参数假设，同时保留了部分可解释的、针对每个纵向结局的关联结构。该联合模型采用编码器-解码器（序列到序列）架构来学习患者时变协变量轨迹中的潜在结构。模型通过一个学习得到的可解释关联结构将纵向过程与生存过程联系起来，其中解码器输出的每个纵向结果在贡献于（生存模型的）风险评分之前，会先由基线协变量进行重新调制……

    arXiv:2610.07388v1 Announce Type: cross  Abstract: Joint Models simultaneously model longitudinal and survival outcomes, leveraging patterns in patients' longitudinal trajectory to improve the prediction of survival outcomes. The classical parametric joint models, however, rely on fixed parametric assumptions, making them susceptible to bias under model misspecification and smaller sample sizes. We propose a deep joint model, DeepAJM, that does not require any parametric assumptions, while retaining a partially interpretable, per-longitudinal-outcome association structure. The joint model uses an encoder-decoder (sequence-to-sequence) architecture to learn the latent structure in patients' time-varying covariate trajectories. The model links the longitudinal processes to the survival processes through a learned interpretable association structure, in which each longitudinal output from the decoder gets remodulated by baseline covariates before it contributes to the risk scores from the
    
[^45]: HyperNSDE：用于静态-纵向临床数据联合生成的个性化神经SDE

    HyperNSDE: Personalized Neural SDEs for Joint Static-Longitudinal Clinical Data Generation

    [https://arxiv.org/abs/2610.07383](https://arxiv.org/abs/2610.07383)

    HyperNSDE通过超网络将静态患者特征注入潜在神经SDE，首次联合生成异构静态协变量、不规则采样的纵向轨迹和观测时间三类紧密耦合的临床数据，为医疗AI提供真实且保护隐私的合成患者数据。

    

    合成患者数据生成是解决医疗机器学习中数据稀缺与隐私限制双重挑战的一种有前景的方案。要真实地合成患者级临床数据，需要联合建模异构的静态协变量、不规则采样的纵向轨迹以及具有信息量的观测时间——这三者在实践中紧密耦合，却很少被共同处理。我们提出HyperNSDE，一种连续时间生成模型，它通过超网络将潜在神经随机微分方程条件化于静态患者表示之上，使基线特征能够在初始条件之外塑造轨迹演化，而无需轨迹编码器，同时随机潜在动力学捕捉生成路径中的真实变异性。观测时间通过依赖于潜在状态的强度过程进行联合建模，不规则随机路径上的训练则通过确定性方法加以稳定……

    arXiv:2610.07383v1 Announce Type: cross  Abstract: Synthetic patient data generation is a promising solution to the dual challenge of data scarcity and privacy constraints in healthcare machine learning. Realistic synthesis of patient-level clinical data requires jointly modeling heterogeneous static covariates, irregularly sampled longitudinal trajectories, and informative observation times - three tightly coupled components in practice yet rarely addressed together. We propose HyperNSDE, a continuous-time generative model that conditions a latent Neural SDE on static patient representations through a hypernetwork, allowing baseline characteristics to shape trajectory evolution beyond the initial condition without requiring a trajectory encoder, while stochastic latent dynamics capture realistic variability in generated paths. Observation times are modeled jointly through a latent-state-dependent intensity process, and training on irregular stochastic paths is stabilized via a determi
    
[^46]: 基于Blackwell序的多变量高斯系统中的冗余与协同

    Redundancy and synergy in multivariate Gaussians via the Blackwell order

    [https://arxiv.org/abs/2610.07360](https://arxiv.org/abs/2610.07360)

    本文基于Blackwell序为多变量高斯系统提出了部分信息分解的定义，证明了高斯信道对提取冗余与联合信息是最优的，并给出了两信源情形下的闭式表达式与高效数值算法。

    

    部分信息分解（PID）的目标是量化多个信源关于目标所提供的冗余信息和协同信息。PID在机器学习、神经科学及其他领域有着广泛的应用，但如何为高维连续系统定义和计算PID仍然是一个具有挑战性的问题。本文基于Blackwell序（该概念形式化了何时一个信道比另一个信道信息量更大）为多变量高斯系统定义了一种PID。我们证明了高斯信道对于提取冗余信息和联合信息均是最优的，从而得到了直观的几何解释以及计算PID的高效数值算法。我们的联合信息和协同度量与著名的BROJA度量相一致，并且我们推导出了两信源情形下两者的闭式表达式。我们还论证了Blackwell冗余（与BROJA不同）是唯一满足特定公理的现有冗余度量。

    arXiv:2610.07360v1 Announce Type: cross  Abstract: The goal of the partial information decomposition (PID) is to quantify the redundant and synergistic information that multiple sources provide about a target. PID has many applications in machine learning, neuroscience, and other fields, but defining and computing it for high-dimensional continuous systems remains challenging. Here, we define a PID for multivariate Gaussian systems based on the Blackwell order, which formalizes when one channel is more informative than another. We prove that Gaussian channels are optimal for extracting both redundant and union information, yielding an intuitive geometric interpretation and an efficient numerical algorithm for the PID. Our union information and synergy coincide with the well-known BROJA measures, and we derive closed-form expressions for both in the case of two sources. We also argue that Blackwell redundancy (which differs from BROJA) is the only existing redundancy measure that satisf
    
[^47]: 协变量缺失情形下的少假设逻辑回归

    Assumption-lean logistic regression with missing covariates

    [https://arxiv.org/abs/2610.07292](https://arxiv.org/abs/2610.07292)

    该论文提出了一种在协变量分布未知（仅有界）的少假设设定下处理协变量缺失逻辑回归参数估计的随机近似方法，克服了传统方法在协变量分布未知时可能严重失效的问题。

    

    在监督学习问题中经常遇到协变量缺失的情况，使用此类数据进行估计的经典方法依赖于精心设计的缺失数据插补方案，或采用会导致非凸 M-估计问题的似然近似。这些方法及其相关变体适用于协变量分布已知的场景，更广泛地说，它们在线性模型中取得了巨大成功。但即使是在中等维度的逻辑回归这类基本的非线性问题中，当协变量分布未知时，这些方法也可能出现严重的失效模式。出于对可靠替代方法的需求，我们研究了协变量缺失情形下逻辑回归的参数估计问题。关键在于，我们在少假设的设定下开展工作，即协变量分布未知（但有界）。我们设计了一种基于 Z-的随机近似方法（摘要在此处截断）

    arXiv:2610.07292v1 Announce Type: cross  Abstract: Missing covariates are frequently encountered in supervised learning problems, and classical methods for estimation using such data use carefully chosen imputation schemes for missing data, or likelihood approximations that lead to nonconvex $M$-estimation problems. These methods and their relatives are suitable for scenarios in which the covariate distribution is known, and more broadly, have enjoyed tremendous success in linear models. But even in basic nonlinear problems such as logistic regression in moderate dimensions, such methods can experience drastic failure modes when the covariate distribution is unknown.   Motivated by the need for reliable alternatives, we consider the problem of parameter estimation in logistic regression with missing covariates. Crucially, we operate in the assumption-lean setting where the covariate distribution is unknown (but bounded). We design a stochastic approximation method that is based on $Z$-
    
[^48]: 跨相关任务的经验贝叶斯谱部分池化

    Empirical-Bayes spectral partial pooling across related tasks

    [https://arxiv.org/abs/2610.07284](https://arxiv.org/abs/2610.07284)

    该论文提出了一种可扩展的经验贝叶斯框架——层次谱收缩（HSS），通过在相关任务之间对任务特定的谱估计器进行部分池化，在样本量相对有限时既能获得比独立估计更稳定的结果，又能保留完全合并所无法体现的任务特定结构。

    

    谱方法是高维统计和机器学习的核心，是协方差估计、矩阵去噪、表示学习、聚类和潜变量建模等诸多方法的基础。在这项工作中，我们主要关注高维因子模型的谱估计器，其中利用数据矩阵的主奇异向量来估计潜在结构和协方差参数。然而，在许多现代应用中，数据是在相关但异质的任务、研究、领域或人群中收集的。当样本量相对于维度较为有限时，将此类估计器单独应用于每个任务可能导致不稳定的估计，而将数据完全合并则会掩盖有意义的任务特定结构。我们提出了层次谱收缩（Hierarchical Spectral Shrinkage, HSS），这是一个可扩展的经验贝叶斯框架，用于对任务特定的谱估计器进行部分池化。该方法对……（摘要在此处被截断）

    arXiv:2610.07284v1 Announce Type: cross  Abstract: Spectral methods are central to high-dimensional statistics and machine learning, underlying procedures for covariance estimation, matrix denoising, representation learning, clustering, and latent variable modeling. In this work, we focus primarily on spectral estimators for high-dimensional factor models, where leading singular vectors of the data matrix are used to estimate latent structure and covariance parameters. In many modern applications, however, data are collected across related but heterogeneous tasks, studies, domains, or populations. Applying such estimators separately to each task can lead to unstable estimates when sample sizes are limited relative to dimension, while completely pooling the data can obscure meaningful task-specific structure. We introduce Hierarchical Spectral Shrinkage (\texttt{HSS}), a scalable empirical-Bayes framework for partially pooling task-specific spectral estimators. The method regularizes th
    
[^49]: 马尔可夫过程之间传输的条件流匹配方法

    Conditional Flow Matching for Transport Between Markov Processes

    [https://arxiv.org/abs/2610.07229](https://arxiv.org/abs/2610.07229)

    本文提出一种保持马尔可夫结构的条件流匹配算法，用于学习马尔可夫过程从源轨迹分布到目标轨迹分布的传输映射，并证明了总体一致性、有限样本误差界以及与混合时间相关的样本复杂度下界。

    

    受时间序列领域自适应中序列到序列传输问题的启发，我们研究了马尔可夫过程轨迹之间的传输问题。在仅有来自源分布和目标分布的有限数量轨迹的情况下，我们提出了一种基于流匹配的算法，该算法学习从源轨迹分布到目标轨迹分布的传输映射，同时保持马尔可夫结构。我们证明了该算法在总体极限下是一致的，并在混合时间假设下推导了有限样本误差界，其分析遵循马尔可夫设定下经典统计问题的分析框架，包括回归（Nagaraj等，2020）、主成分分析（Kumar和Sarkar，2023）以及矩阵集中性（Neeman等，2024）。此外，我们还给出了一个下界构造，表明即使条件转移具有规则的高斯形式，依赖于混合时间的样本复杂度也是不可避免的。

    arXiv:2610.07229v1 Announce Type: new  Abstract: Motivated by sequence-to-sequence transport in the context time-series domain adaptation, we study the problem of transportation between trajectories of Markov processes. Given a limited number of trajectories from source distribution and the target distribution, we formulate a flow matching based algorithm which learns a transport map from the source to target trajectory distribution, while preserving the Markov structure. We show that this is consistent in the population limit and derive finite-sample error bounds under mixing time assumptions, following the analysis of classical statistical problems including regression (Nagaraj et al., 2020), principal component analysis (Kumar and Sarkar, 2023), and matrix concentration (Neeman et al., 2024) in the Markov setting. We complement that with a lower-bound construction showing that a mixing-time dependent sample complexity is unavoidable even with regular Gaussian conditional transitions
    
[^50]: 自然梯度下降有多低效？从精确最优到 Θ(√log d) 的偏差

    How Inefficient Is Natural Gradient Descent? From Exact Optimality to \Theta ( \sqrt{ \log d } ) Divergence

    [https://arxiv.org/abs/2610.07228](https://arxiv.org/abs/2610.07228)

    本文提出“低效比”来量化自然梯度下降偏离最短 Fisher–Rao 路径的程度，并证明该比值在三类情形中变化：二次势或一维族精确最优（R=1）、有界偏度族具有与维度无关的界、而尺度族乘积（如高斯协方差和 Gamma 率）的低效比随维度以 Θ(√log d) 增长。

    

    自然梯度下降（NGD）是机器学习中众多常用方法的基础。对于对偶平坦（dually flat）分布族，在正向 Kullback–Leibler 目标上的理想化 NGD 沿混合测地线行进，而该测地线往往比最短的 Fisher–Rao 路径更长。我们用“低效比” \(R \ge 1\) 来量化这一额外开销，即混合测地线的 Fisher 长度与 Fisher–Rao 距离之比，并给出该比值在所有端点对上的上确界随参数维度 \(d\) 变化的界。我们提出一个张量判据来刻画情形（I）的分布族，其处处满足 \(R=1\)：恰好是具有二次势函数或维度为一的族，例如固定协方差的高斯分布。对于非二次族，我们证明了另外两种情形：（II）有界的三阶偏度加上有限的 Fisher–Rao 直径可得到与维度无关的界；（III）对于尺度族的乘积——包括高斯协方差和 Gamma 率——\(R\) 以 \(\Theta(\sqrt{\log d})\) 增长，即随维度无界增长。

    arXiv:2610.07228v1 Announce Type: cross  Abstract: Natural gradient descent (NGD) underlies common methods in ML. For dually flat families, idealized NGD on the forward Kullback--Leibler objective follows the mixture geodesic which is often longer than the shortest Fisher--Rao path. We quantify this overhead by the inefficiency ratio \(R \ge 1\), the Fisher length of the mixture geodesic divided by the Fisher--Rao distance, and bound its supremum over endpoint pairs as a function of the parameter dimension \(d\). A tensor criterion identifies the regime (I) families, with \(R=1\) everywhere: exactly those with quadratic potential or dimension one, such as fixed-covariance Gaussians. For non-quadratic families, we prove two further regimes: (II) bounded third-order skewness plus finite Fisher--Rao diameter yields a dimension-independent bound; and (III) for products of scale families---including Gaussian covariances and Gamma rates---\(R\) grows as \(\Theta(\sqrt{\log d})\), unbounded i
    
[^51]: 基于最小距离方法的泊松经验贝叶斯随机变量和估计

    Poisson empirical Bayes estimation of sums of random variables via minimum-distance methods

    [https://arxiv.org/abs/2610.07190](https://arxiv.org/abs/2610.07190)

    该论文提出了一种基于规则与粗化最小距离估计的非参数经验贝叶斯方法，用于估计泊松混合模型中可观测与不可观测变量函数之和，证明了大样本下代入估计与oracle贝叶斯估计渐近一致，且当混合分布支撑有限时可达接近参数化的收敛速度。

    

    对可观测变量与不可观测变量的函数之和进行估计是统计学中一个长期存在的问题，在众多领域都有应用。我们在泊松混合模型中考虑这一问题，此类模型中经验贝叶斯提供了一个自然的框架，但其非参数理论仍然有限。我们基于对未知混合分布的规则最小距离估计与粗化最小距离估计，发展了一种非参数经验贝叶斯方法。对于广泛的一类此类和，我们建立了大样本保证，表明所得的代入估计与oracle贝叶斯估计渐近一致。特别地，当混合分布具有有限支撑时，我们获得了几乎达到参数化水平的收敛速度（仅相差一个对数因子）。随后，我们对两个代表性的和给出了有限样本分析：即观测计数不超过固定阈值的单元的总强度，以及观测……（原文在此处被截断）

    arXiv:2610.07190v1 Announce Type: cross  Abstract: The estimation of sums of functions of observable and unobservable variables is a long-standing problem in statistics, with applications in many domains. We consider this problem in Poisson mixture models, where empirical Bayes provides a natural framework but nonparametric theory remains limited. We develop a nonparametric empirical Bayes methodology based on regular and coarsened minimum-distance estimation of the unknown mixing distribution. For a broad class of such sums, we establish large-sample guarantees showing that the resulting plug-in estimates asymptotically merge with the oracle Bayes estimate. In particular, when the mixing distribution has finite support, we obtain a nearly parametric convergence rate, up to a logarithmic factor. We then provide a finite-sample analysis of two representative sums: the total intensity among units whose observed count does not exceed a fixed threshold, and the number of units whose observ
    
[^52]: 语言模型中柏拉图式表示的理论

    A theory of platonic representations in language models

    [https://arxiv.org/abs/2610.07168](https://arxiv.org/abs/2610.07168)

    本文通过假设数据具有隐藏的层级结构（抽象层次跨语言共享、表面层次为语言或模态特定），并借助概率上下文无关文法与信念传播理论推导出分析性预测，首次从理论上解释了多语言模型中间层出现柏拉图式表示的现象及其随语言相近程度和模型质量增强的规律。

    

    在多语言语言模型的内层中，翻译句子的表示是相似的——这一观察与柏拉图表示假说相关联，但在理论上尚未得到解释。我们基于以下假设提供了相关解释：数据具有隐藏的层级结构，其抽象层次在语言之间共享，而表面层次则是模态或语言特定的。具体而言，我们从概率上下文无关文法生成合成语言，这些文法共享上层产生式规则但不共享下层产生式规则。在此设定下，贝叶斯最优的下一词预测器是信念传播（BP）；将它的消息编码到连续的层中可以产生分析性预测，这些预测与在同一数据上训练的transformer高度吻合。该框架解释了为什么跨语言相似性在中间层达到峰值、与语言特定结构共存，并随语言相近程度、模型质量和数据而增强。

    arXiv:2610.07168v1 Announce Type: cross  Abstract: Representations of translated sentences are similar in the inner layers of multilingual language models -- an observation connected to the platonic representation hypothesis, yet unexplained theoretically. We provide an explanation based on the assumption that data have a hidden hierarchical structure whose abstract levels are shared across languages while surface levels are modality- or language-specific. Concretely, we generate synthetic languages from probabilistic context-free grammars sharing upper-level but not lower-level production rules. In this setting the Bayes-optimal next-token predictor is belief propagation (BP); encoding its messages in successive layers yields analytical predictions that agree well with transformers trained on the same data. The framework explains why cross-lingual similarity peaks in middle layers, coexists with language-specific structure, and strengthens with language proximity, model quality and da
    
[^53]: Fréchet Inception 距离的样本最优估计

    Sample-Optimal Estimation of the Fr\'echet Inception Distance

    [https://arxiv.org/abs/2610.07114](https://arxiv.org/abs/2610.07114)

    该论文针对FID估计中的有限样本偏差问题，证明了插件估计器的紧致偏差与方差界并确立其平方级（d²）样本复杂度，同时将FID∞估计器推广到任意阶外推方法以实现去偏估计。

    

    Fréchet Inception 距离（FID）被广泛用于评估生成模型，但其经验插件估计器存在有限样本偏差 [BSAG18, CF20]。我们研究了在一个分布已知的情况下，估计具有有界均值距离和协方差的 $d$ 维高斯分布之间的 FID 至误差 $\epsilon$ 所需的样本复杂度 $n$。我们的贡献有三点：(1) 我们为经验插件估计器建立了紧致的有限样本偏差界 $\Theta(\frac{d^2}{n})$ 和方差界 $\Theta(\frac{d}{n} + \frac{d^2}{n^2})$，从而确立了 $\gtrsim d^2$ 的样本复杂度。(2) 为了对经验插件估计器进行去偏，我们将 [CF20] 的 ${\rm FID}_\infty$ 估计器推广到任意阶数 $k$ 的外推方法，并进一步在我们的框架下证明了任意 $k$ 阶外推的紧致偏差界 $\Theta(\frac{d^{k+2}}{n^{k+1}})$ 和方差界 $\Theta(\frac{d}{n} + \frac{d^2}{n^2})$。(3) 我们引入

    arXiv:2610.07114v1 Announce Type: new  Abstract: The Fr\'echet Inception Distance (FID) is widely used to evaluate generative models, but its empirical plug-in estimator suffers from finite-sample bias [BSAG18, CF20]. We study the sample complexity $n$ of estimating FID to error $\epsilon$ between $d$-dimensional Gaussians with bounded mean distance and covariances, when one distribution is known. Our contributions are threefold. (1) We establish tight finite-sample $\Theta(\frac{d^2}{n})$ bias and $\Theta(\frac{d}{n} + \frac {d^2} {n^2})$ variance bounds for the empirical plug-in estimator, establishing a $\gtrsim d^2$ sample complexity. (2) To debias the empirical plug-in estimator, we generalize the ${\rm FID}_\infty$ estimator of [CF20] to extrapolation methods of arbitrary order $k$. We further prove tight bias and variance bounds of $\Theta(\frac{d^{k + 2}}{n^{k + 1}})$ and $\Theta(\frac d n + \frac{d^2}{n^2})$ for any order-$k$ extrapolation under our framework. (3) We introduce
    
[^54]: 一次查询并非承诺：在线推迟中学习纠正专家答案

    A Query Is Not a Commitment: Learning to Correct Expert Answers in Online Deferral

    [https://arxiv.org/abs/2610.07084](https://arxiv.org/abs/2610.07084)

    提出ORUCB算法，利用累积响应学习误差的界来校准置信度加权风险回归与探索，在在线推迟学习中对不准确专家的答案进行纠正，实现了 $O(\sqrt T\log(T+1))$ 的高概率伪遗憾界。

    

    arXiv:2610.07084v1 公告类型：cross 摘要：一个不准确的专家在经过纠正后仍然可以提供有用的信息。我们研究在线学习推迟问题，其中学习者选择一名专家，并在购买其答案之前确定一个纠正函数，然后将该函数应用于所收到的答案。困难在于，观察到的损失同时反映了专家质量和尚未完成的纠正过程：早期的错误可能会阻碍那些在学习后会非常有价值的查询。我们提出ORUCB算法，它汇集了共享响应和专家特定的多项式响应。累积响应学习误差的界用于校准置信度加权的风险回归和探索，使路由器在决定购买哪些答案时能够考虑这一误差。在有界残差和分歧、最优响应的固定可行模型、以及自由风险和最优查询风险的线性模型的假设下，校准后的算法在T轮中实现了高概率伪遗憾 $O(\sqrt T\log(T+1))$

    arXiv:2610.07084v1 Announce Type: cross  Abstract: An inaccurate expert can still provide useful information after correction. We study online learning to defer in which the learner chooses an expert and fixes a correction function before purchasing its answer, then applies that function to the answer received. The difficulty is that observed losses reflect both expert quality and an unfinished correction: early errors can discourage queries that would be valuable after learning. We propose ORUCB, which pools shared and expert-specific polynomial responses. A bound on cumulative response-learning error calibrates confidence-weighted risk regression and exploration, allowing the router to account for this error when deciding which answers to buy. Under bounded residuals and disagreements, a fixed feasible model of optimal responses, and linear models of free and optimal queried risk, the calibrated algorithm achieves high-probability pseudo-regret $O(\sqrt T\log(T+1))$ over $T$ rounds f
    
[^55]: 在上下文中学习决策树桩阈值：Softmax注意力的动力学

    Learning Decision-Stump Thresholds in Context: Dynamics of Softmax Attention

    [https://arxiv.org/abs/2610.07074](https://arxiv.org/abs/2610.07074)

    本文证明了两参数softmax注意力模型通过基于梯度的预训练能够学习决策阈值估计，其误差为$\widetilde O((m\wedge n)^{-1}+N^{-1})$，并揭示了背后的机制是参数协调发散——注意力尺度以$t^{1/4}$增长、阈值误差以$t^{-1/4}$衰减。

    

    估计决策阈值需要在未知边界附近定位观测值。我们研究了基于梯度的预训练如何在具有固定特征与不等号方向的两参数softmax注意力模型中学习这一统计规则。预训练使用带标签的上下文及其真实阈值；而新的阈值必须仅凭上下文推断。在大分辨率初始化下，对m个任务（每个任务含n个样本）进行恒定步长的梯度下降，会得到一个冻结的估计器，对于每个固定的内部阈值和任意新上下文规模N，其误差为$\widetilde O((m\wedge n)^{-1}+N^{-1})$。这两项将有限预训练的精度与新上下文的定位能力分离开来。其机制是协调的参数发散：总体训练先校准相对的标签分数与特征分数，随后使注意力尺度以$t^{1/4}$的速度增长，从而得到总体阈值误差$O(t^{-1/4})$。为了将这一机制（摘要在此处截断）

    arXiv:2610.07074v1 Announce Type: cross  Abstract: Estimating a decision threshold requires locating observations near an unknown boundary. We study how gradient-based pretraining learns this statistical rule in a two-parameter softmax-attention model with a fixed feature and inequality direction. Pretraining uses labeled contexts and their true thresholds; a fresh threshold must be inferred from context alone. Under a large-resolution initialization, constant-step gradient descent on $m$ tasks with $n$ examples each produces a frozen estimator with error $\widetilde O((m\wedge n)^{-1}+N^{-1})$ for each fixed interior threshold and every fresh-context size $N$. The two terms separate finite-pretraining accuracy from fresh-context localization. The mechanism is coordinated parameter divergence: population training calibrates the relative label and feature scores, then increases the attention scale as $t^{1/4}$, giving population threshold error $O(t^{-1/4})$. To transfer this mechanism 
    
[^56]: sHAIL-Causal：一种用于不变因果预测因子发现的序列阶梯式方法

    sHAIL-Causal: A Sequential Staircase Procedure for Invariant Causal Predictor Discovery

    [https://arxiv.org/abs/2610.07057](https://arxiv.org/abs/2610.07057)

    本文提出 sHAIL-Causal，一种以拟合优度饱和与跨环境不变性联合判据为门控的序列阶梯式学习程序，可避免被混杂预测因子诱导，并在 Richness 条件下可证明地停在真正的因果预测因子集合上，而仅依赖复杂度控制或朴素贪心搜索的方法均无法做到。

    

    我们提出了 sHAIL-Causal，这是饱和分层原子增量学习范式的因果特化版本：一种序列阶梯式程序，当饱和信号表明当前阶段的掌握程度已趋于平台期时，该程序会沿嵌套的假设类层次结构 H_0 < H_1 < ... < H_K 逐级上升。一般的 sHAIL 将饱和判据留待确定，而 sHAIL-Causal 用拟合优度饱和与跨环境不变性的联合判据将其具体化，以此取代结构风险最小化的复杂度控制。我们从理论上并通过仿真表明，仅基于复杂度的阶梯式方法会被混杂预测因子所诱导——这些预测因子虽能降低经验风险，却不反映稳定的因果结构；而在每变量 Richness 条件下，以不变性为门控的阶梯式方法可被证明恰好停在真正的因果预测因子集合上。我们进一步表明，即使在 Richness 条件下，朴素的贪心搜索也无法恢复因果集合……

    arXiv:2610.07057v1 Announce Type: cross  Abstract: We introduce sHAIL-Causal, the causal specialization of the Saturated Hierarchical Atomic Incremental Learning (sHAIL) paradigm: a sequential staircase procedure that ascends a nested hierarchy of hypothesis classes H_0 < H_1 < ... < H_K once a saturation signal indicates that mastery of the current stage has plateaued. Where general sHAIL leaves the saturation criterion open, sHAIL-Causal instantiates it with a joint criterion of goodness-of-fit saturation and cross-environment invariance, replacing the complexity control of Structural Risk Minimization. We show, theoretically and by simulation, that complexity-only staircases are seduced by confounded predictors that lower empirical risk without reflecting stable causal structure, whereas an invariance-gated staircase provably halts at the true causal predictor set under a per-variable Richness condition. We show that naive greedy search fails to recover the causal set even under Ric
    
[^57]: 变量含误差问题的数据融合方法

    Data Fusion for Errors-in-Variables

    [https://arxiv.org/abs/2610.07048](https://arxiv.org/abs/2610.07048)

    本文提出了一种数据融合估计方法，通过条件可迁移性假设利用外部研究的重复测量来识别目标研究中的条件测量误差分布，从而在源-目标异质性下解决变量含误差问题。

    

    我们研究变量含误差问题，其中目标研究仅包含未观测暴露变量的单个易出错替代测量，而外部源研究提供了来自不同人群的重复替代测量。该方法允许测量误差分布依赖于观测到的无误差变量，且无误差变量的分布本身在不同研究之间可能存在差异。我们引入了一个条件可迁移性假设，使得在源-目标异质性的情况下能够利用外部重复测量数据。结合额外的重复误差条件，该假设识别了目标研究的条件测量误差分布。基于这一识别结果，我们为一类广泛的目标泛函开发了数据融合估计量。该估计量结合了条件反卷积、灵活的干扰参数估计以及正交校正技术，降低了对干扰参数估计的一阶敏感性。

    arXiv:2610.07048v1 Announce Type: cross  Abstract: We study errors-in-variables problems in which a target study contains only a single error-prone surrogate of an unobserved exposure, while an external source study provides repeated surrogate measurements from a different population. The measurement error distribution is allowed to depend on the observed error-free variables, and the error-free variable distribution itself may differ between studies. We introduce a conditional transportability assumption that enables the use of external repeated measurements under source-target heterogeneity. Together with additional replicate-error conditions, it identifies the target conditional measurement-error distribution. Building on this identification result, we develop a data-fusion estimator for a broad class of target functionals. The estimator combines conditional deconvolution, flexible nuisance estimation, and orthogonal correction that reduces first-order sensitivity to nuisance estima
    
[^58]: 基于高效资源分配的多任务主动学习

    Multi-Task Active Learning with Efficient Resource Allocation

    [https://arxiv.org/abs/2610.07045](https://arxiv.org/abs/2610.07045)

    该论文提出 ALCATRAs 统一框架，通过成本自适应的任务选择策略与代理学习，在预算约束下有选择地获取标注期的昂贵辅助信息，从而提升部署时辅助变量系统性缺失场景下的下游预测性能。

    

    许多科学研究允许在数据标注阶段收集昂贵的辅助信息，但在部署阶段却无法获取。这类信息的例子包括诊断检测、实验室化验和专家评估。我们研究了这种部署不对称性下的预测问题：辅助变量在标注阶段于预算约束下被有选择地获取，但在预测时却系统性地不可用，由此形成了一个“设计性缺失”问题，该问题将数据获取、代理变量构建与预测耦合在一起。在本工作中，我们提出了具有成本自适应任务资源分配的主动学习框架（ALCATRAs），这是一个在资源约束下有选择地获取辅助信息，并利用这些信息改进下游预测的统一框架。ALCATRAs 包含两个主要组成部分：任务选择策略，用于策略性地为未标注数据选取一系列高性价比的任务来执行；以及代理学习策略...

    arXiv:2610.07045v1 Announce Type: cross  Abstract: Many scientific studies allow costly auxiliary information to be collected during data labeling but not at deployment. Examples include diagnostic tests, laboratory assays, and expert evaluations. We study prediction under this deployment asymmetry, where auxiliary variables are selectively acquired during labeling under a budget constraint but systematically unavailable at prediction time, creating a missing-by-design problem that couples data acquisition, surrogate construction, and prediction. In this work, we introduce Active Learning with Cost-Adaptive Task Resource Allocations (ALCATRAs), a unified framework for selectively acquiring auxiliary information under resource constraints and leveraging that information to improve downstream prediction. ALCATRAs consists of two main components: a task-selection policy which strategically selects a sequence of cost-effective tasks for unlabeled data to perform, and a surrogate learning p
    
[^59]: 前提即是问题：自监控测试时自适应中的可交换性失效

    The Premise Is the Problem: Exchangeability Failure in Self-Monitored Test-Time Adaptation

    [https://arxiv.org/abs/2610.07038](https://arxiv.org/abs/2610.07038)

    论文证明在自监控的测试时自适应中，由于监控与自适应共用同一反馈，预测目标重叠与误差依赖性会破坏可交换性假设，导致虚假警报、预测质量下降，且自适应会掩盖持续变化，而冻结的原始模型信号反而更清晰。

    

    现代预测模型通常在部署后进行更新，以应对不断变化的数据。但这些更新也可能使预测变差，因此实际系统需要一个可靠的监控器来检测有害变化并触发保护机制。一种自然的设计是监控引导更新的同一批预测误差。本文提出这样一个问题：当监控与自适应使用相同的反馈时，该监控器背后的统计保证是否仍然有效。我们在多步时间序列预测中研究了这一问题，结果表明预测目标的重叠和预测误差之间的依赖性会破坏该保证所需的一个关键假设。此时，即使没有发生有害变化，监控器也可能发出警报，而其触发的响应可能进一步损害预测质量。我们还发现，自适应过程可能会使其自身的监控器无法察觉持续发生的变化，而原始的冻结模型反而保留了更清晰的信号。这些结果揭示了一个基本的……

    arXiv:2610.07038v1 Announce Type: new  Abstract: Modern forecasting models are often updated after deployment so they can respond to changing data. These updates can also make predictions worse, so practical systems need a reliable monitor that can detect harmful changes and trigger protection. A natural design is to monitor the same prediction errors that guide the updates. This paper asks whether the statistical guarantee behind such a monitor remains valid when monitoring and adaptation use the same feedback. We study this question in multi-step time-series forecasting. We show that overlapping targets and dependence in forecast errors can break a key assumption required by the guarantee. The monitor may then raise alarms even when no harmful change has occurred, and its response can further damage prediction quality. We also find that adaptation can hide sustained changes from its own monitor, while the original frozen model retains a clearer signal. These results expose a basic fa
    
[^60]: 基于生成模型的递归熵风险强化学习的近最优样本复杂度

    Near-Optimal Sample Complexity for Recursive Entropic Risk Reinforcement Learning with a Generative Model

    [https://arxiv.org/abs/2610.06931](https://arxiv.org/abs/2610.06931)

    本文对基于模型的风险敏感 Q 值迭代（MB-RS-QVI）算法进行了精细分析，在生成模型假设下首次为递归熵风险强化学习建立了近最优的样本复杂度保证，其对有效视界的指数依赖性与现有下界相匹配，消除了理论差距。

    

    本文研究了在具有风险参数 β≠0 的递归熵风险偏好下，假设可以访问 MDP 的生成模型时，有限折扣马尔可夫决策过程（MDP）中价值学习和策略学习的样本复杂度。我们对基于模型的风险敏感 Q 值迭代（MB-RS-QVI）——一种先前工作中提出的插件式基于模型的方法——进行了精细分析，并针对学习最优 Q 值函数和 ε-最优策略分别推导出了 (ε,δ)-PAC 保证。与该设定下现有的最佳理论保证相比，我们的样本复杂度边界改进了对有效视界 1/(1-γ) 的指数依赖。特别地，在关于 |β|/(1-γ) 的指数依赖方面，以及在 S、A、ε 和 |β| 等参数方面（直至对数因子），我们的边界与现有下界相匹配。因此，我们的分析消除了……

    arXiv:2610.06931v1 Announce Type: new  Abstract: In this paper, we study the sample complexities of value and policy learning in finite discounted Markov decision processes (MDPs) under recursive entropic risk preferences with risk parameter \(\beta\neq 0\), assuming access to a generative model of the MDP. We provide a refined analysis of model-based risk-sensitive Q-value iteration (MB-RS-QVI), a plug-in model-based method introduced in prior work, and derive \((\varepsilon,\delta)\)-PAC guarantees for both learning the optimal \(Q\)-value function and an \(\varepsilon\)-optimal policy. Our bounds improve the exponential dependence on the effective horizon \(1/(1-\gamma)\) compared with the best existing guarantees for this setting. In particular, they match the existing lower bounds in their exponential dependence on \(|\beta|/(1-\gamma)\), as well as in \(S\), \(A\), \(\varepsilon\), and \(|\beta|\), up to logarithmic factors. Consequently, our analysis removes the exponential gap 
    
[^61]: 面向多元函数型数据异常检测的低秩与结构化稀疏张量分解

    Low-Rank and Structured Sparse Tensor Decomposition for Anomaly Detection in Multivariate Functional Data

    [https://arxiv.org/abs/2610.06930](https://arxiv.org/abs/2610.06930)

    提出两种无监督稀疏张量分解方法（ES-CP与FG-Lasso），通过低秩CP分解结合逐元素与纤维方向的稀疏惩罚，在保留多模态结构的同时检测多元函数型数据中的局部异常与时间纤维集中型异常。

    

    多元函数型数据广泛出现在许多现代制造系统中，其中多个传感器以密集采样方式记录过程轨迹。监测此类数据具有挑战性，因为正常（名义）变化在样本、传感器和时间之间具有很强的相关性，而故障可能表现为孤立的偏差，也可能表现为集中在少量特定传感器时间轨迹内的结构性偏移。我们提出两种能够保留这种多模态结构的无监督稀疏张量分解方法。逐元素稀疏CP分解（ES-CP）使用逐元素的 ℓ₁ 惩罚来识别局部异常，而纤维方向稀疏组Lasso CP分解（FG-Lasso）则结合逐元素与纤维方向的惩罚，既能检测局部偏差，也能检测集中在时间纤维内的异常。两种方法均通过低秩CP分解来表示名义过程行为，并采用交替（优化算法进行估计）……（原文摘要在此处截断）

    arXiv:2610.06930v1 Announce Type: cross  Abstract: Multivariate functional data arise in many modern manufacturing systems, where multiple sensors record densely sampled process trajectories. Monitoring such data is challenging because nominal variation is strongly correlated across samples, sensors, and time, while faults may appear either as isolated deviations or as structured departures concentrated within a limited number of sensor-specific temporal trajectories. We propose two unsupervised sparse tensor decomposition methods that preserve this multimode structure. Entrywise Sparse CP Decomposition (ES-CP) uses an entrywise \(\ell_1\) penalty to identify localized anomalies, whereas Fiberwise Sparse-Group Lasso CP Decomposition (FG-Lasso) combines entrywise and fiberwise penalties to detect both localized deviations and anomalies concentrated within temporal fibers. Both methods represent nominal process behavior through a low-rank CP decomposition and are estimated using alternat
    
[^62]: 对比学习中语义几何的锚点散度

    Anchor Divergence for Semantic Geometry in Contrastive Learning

    [https://arxiv.org/abs/2610.06919](https://arxiv.org/abs/2610.06919)

    本文提出“锚点散度”方法，通过建立锚点概率分布与Bregman几何之间的对应关系，使固定表示上的语义几何能够适配特定上下文，突破了余弦相似度单一固定几何的局限。

    

    本文研究语义上下文如何决定学习到的向量表示中的几何结构。相似度通常使用余弦相似度来衡量，这种方式提供了一种单一固定的几何。然而，语义相似度本质上依赖于上下文：两幅图像可能相似是因为它们描绘了同一物体、共享某种视觉风格，或与同一临床发现相关。我们证明对比表示天然地涵盖了一族几何结构，这些几何结构可以被专门化以匹配特定的语义结构。关键思想是利用对比学习、指数族分布和信息几何三者之间的相互作用，在“锚点”上的概率分布与表示空间上的Bregman几何之间建立对应关系。我们利用这种对应关系定义了“锚点散度”，这是一种在固定表示上指定特定于上下文的语义几何的方法。在这种对应关系下，模（原文摘要在此处截断）

    arXiv:2610.06919v1 Announce Type: new  Abstract: This paper concerns how semantic context determines geometry in learned vector representations. Similarity is typically measured using cosine similarity, which provides a single fixed geometry. Semantic similarity, however, is inherently context dependent: two images may be similar because they depict the same object, share a visual style, or are relevant to the same clinical finding. We show that contrastive representations naturally encompass a family of geometries that can be specialized to particular semantic structure. The key idea is to use an interplay between contrastive learning, exponential families, and information geometry to establish a correspondence between probability distributions over "anchors" and Bregman geometries on the representation space. We use this correspondence to define "Anchor Divergences", a method for specifying context-specific semantic geometries on fixed representations. Under this correspondence, mode
    
[^63]: 记忆预测超额：用于衡量随机过程预测增益与记忆长度的一个概率量

    Memory Prediction Excess: A Probabilistic Quantity for Predictive Gain and Memory Length in Stochastic Processes

    [https://arxiv.org/abs/2610.06894](https://arxiv.org/abs/2610.06894)

    本文提出“记忆预测超额”（MPE）这一新的概率量，用以量化在离散时间有限状态随机过程中利用完整历史信息相对于仅用静态边际分布所带来的预测准确率平均提升，并证明了其非负性、上界条件及退化情形等基本性质。

    

    随机过程预测中的一个核心问题是：过去的信息能够在多大程度上提高正确预测下一个状态的概率。我们引入“记忆预测超额”这一概念来定量地回答这一问题。在离散时间有限状态过程中，MPE 度量的是使用完整观测历史相对于仅使用静态边际分布所获得的预测准确率的平均提升。它被定义为期望最优条件预测准确率与最优静态预测准确率之差。本文考察了它的基本性质：MPE 始终非负；它存在一个依赖于静态准确率的上界，当且仅当未来几乎必然是过去的确定性函数时该上界被达到；此外还刻画了 MPE 为零的退化情形。文中还引入了一个取值于单位区间的归一化版本。

    arXiv:2610.06894v1 Announce Type: cross  Abstract: A central question in the prediction of stochastic processes is the extent to which past information can improve the probability of correctly predicting the next state. We introduce the Memory Prediction Excess (MPE) to address this question quantitatively. The MPE measures the average improvement in prediction accuracy obtained by using the entire observed history relative to using only the static marginal distribution, in discrete-time finite-state processes. It is defined as the difference between the expected optimal conditional prediction accuracy and the optimal static prediction accuracy. Its basic properties are examined: the MPE is always non-negative; it admits an upper bound depending on the static accuracy, attained if and only if the future is almost surely a deterministic function of the past; and degenerate cases in which the MPE vanishes are characterized. A normalized version, taking values in the unit interval, is int
    
[^64]: 考虑电路：超越单一Softmax的深度分离与普适性

    Consideration Circuits: Depth Separation and Universality Beyond a Single Softmax

    [https://arxiv.org/abs/2610.04143](https://arxiv.org/abs/2610.04143)

    论文提出由多项logit（MNL）单元构成的有向无环图所定义的“考虑电路”多阶段选择模型，并证明了尖锐的深度-范数分离定理：深度从2增至3时，逼近误差ε所需的品味向量范数从Θ(log(1/ε)/ε)降至Θ(log(1/ε))，而包括单一MNL在内的菜单无关随机效用模型在折中任务上误差存在不可消除的下界，从而在表达能力与结构上严格超越了单一softmax模型。

    

    大多数基于特征的选择模型——无论是经典的还是深度的——都是对物品打分后应用单一的softmax。我们提出了“考虑电路”，这是一类基于特征的多阶段选择模型，由多项logit（MNL）单元构成的有向无环图定义。源单元为菜单中的物品分配概率，内部单元则利用由其前驱的概率加权特征摘要计算出的MNL权重来组合前驱分布。在一个具有固定非共线特征的三物品折中任务上，与菜单无关的随机效用模型（RUM），包括单个MNL单元，其误差存在一个不趋于零的下界。相比之下，对于考虑电路，我们建立了一个尖锐的深度-范数分离结果：将深度从2增加到3，可将达到误差ε所需的最优最大品味向量范数从Θ(log(1/ε)/ε)降低到Θ(log(1/ε))。深度为2的下界对任意宽度和与菜单无关的路由偏置均成立，而一个fi……（摘要原文在此处被截断）

    arXiv:2610.04143v2 Announce Type: replace  Abstract: Most feature-based choice models, classical and deep, score items and apply a single softmax. We introduce consideration circuits (CC), feature-based models of multi-stage choice defined by directed acyclic graphs of multinomial logit (MNL) units. Source units assign probabilities to menu items, and internal units combine predecessor distributions using MNL weights computed from their probability-weighted feature summaries. On a three-item compromise task with fixed non-collinear features, menu-independent random-utility models (RUM), including a single MNL unit, suffer an error bounded away from zero. For CC, in contrast, we establish a sharp depth--norm separation: increasing depth from $2$ to $3$ reduces the optimal maximum taste-vector norm for error $\epsilon$ from $\Theta(\log(1/\epsilon)/\epsilon)$ to $\Theta(\log(1/\epsilon))$. The depth-$2$ lower bound holds for arbitrary width and menu-independent routing biases, while a fi
    
[^65]: 巨正则生成器

    Grand Canonical Generators

    [https://arxiv.org/abs/2610.00683](https://arxiv.org/abs/2610.00683)

    提出了巨正则生成器（GCG），将玻尔兹曼生成器扩展至巨正则系综，其分解式设计可复用现有正则生成器、解析编码化学势线性依赖，并提供可处理的似然以支持自归一化重要性采样，在流体和吸附问题上准确再现巨正则观测量。

    

    我们提出了巨正则生成器，这是一种将玻尔兹曼生成器扩展到巨正则系综的生成式框架。我们提出了两种设计方案：第一种以化学势为条件对可变尺寸的生成模型进行条件化，从而联合采样粒子数和构型；第二种将巨正则分布分解为粒子数分布和相应的正则玻尔兹曼密度。这种分解式设计可以对正则分量复用任何现有的玻尔兹曼生成器，以解析方式编码已知的化学势线性依赖关系，并产生易于处理的似然，从而支持自归一化重要性采样（SNIS）。实验结果表明，GCG在Lennard-Jones流体和沸石中甲烷吸附问题上准确再现了巨正则观测量，展示了跨化学势的泛化能力，并可通过SNIS和巨正则蒙特卡洛进行校正。

    arXiv:2610.00683v1 Announce Type: cross  Abstract: We introduce Grand Canonical Generators (GCG), a generative framework that extends Boltzmann generators to the grand canonical ensemble. We present two designs. The first conditions a variable-size generative model on the chemical potential, sampling particle number and configuration jointly. The second factorizes the grand canonical distribution into a particle-number distribution and the corresponding canonical Boltzmann density. This factorized formulation can use any existing Boltzmann generator for the canonical component, encodes the known linear chemical-potential dependence analytically, and yields a tractable likelihood that supports self-normalized importance sampling (SNIS). Empirically, GCG accurately reproduces grand canonical observables on a Lennard--Jones fluid and methane adsorption in a zeolite, demonstrating generalization across chemical potentials and correction via SNIS and grand canonical Monte Carlo.
    
[^66]: 掩码离散扩散中tau-leaping的调度优化

    Schedule optimization for tau-leaping in masked discrete diffusion

    [https://arxiv.org/abs/2609.21960](https://arxiv.org/abs/2609.21960)

    该论文通过依赖密度ρ的精确积分表示来刻画掩码离散扩散中tau-leaping采样的因式分解误差，并推导有限步优化问题的递归平稳性方程，从而实现去噪调度的优化。

    

    掩码离散扩散模型通常使用所谓的tau-leaping离散化方法来加速，该方法在每个采样步骤中并行揭示多个坐标。采样器用乘积分布替代每个被揭示块的联合条件分布，从而产生因式分解误差 $\varepsilon_\text{fact}$，即使预测器被完美学习，该误差依然存在。我们分析了在 $N$ 个坐标上进行 $K$ 步采样的标准采样器，其随机块大小取决于去噪调度。我们的分析使用 $\varepsilon_\text{fact}$ 的精确积分表示，该表示以依赖于分布的依赖密度 $\rho$ 来刻画，$\rho$ 记录了随着坐标被揭示比例的增长，条件依赖如何演化。我们为该依赖轮廓开发了估计器，并量化了估计误差如何影响调度选择。我们为有限 $K$ 的优化问题推导了递归平稳性方程，并……（原文在此处截断）

    arXiv:2609.21960v1 Announce Type: cross  Abstract: Masked discrete diffusion models are commonly accelerated using the so-called tau-leaping discretization method, which reveals several coordinates in parallel at each sampling step. The sampler replaces the joint conditional law of each revealed block by a product distribution, incurring a factorization error $\varepsilon_\text{fact}$ present even with perfectly learned predictors. We analyze the standard sampler on $N$ coordinates with $K$ sampling steps, whose random block sizes depend on a denoising schedule. Our analysis uses an exact integral representation of $\varepsilon_\text{fact}$ in terms of a distribution-dependent dependence density $\rho$, which records how conditional dependence evolves as the revealed fraction of coordinates grows. We develop estimators for this profile and quantify how estimation errors affect schedule selection. We derive recursive stationarity equations for the finite-$K$ optimization problem and, un
    
[^67]: 何时锐利协方差包络是紧的？体积采样最小二乘的仅特征几何

    When Is the Sharp Covariance Envelope Tight? Feature-Only Geometry for Volume-Sampled Least Squares

    [https://arxiv.org/abs/2608.26877](https://arxiv.org/abs/2608.26877)

    本文建立了体积采样最小二乘中心系数协方差的Loewner包络，并揭示了仅特征边际nu_A决定谱包络严格性的精确条件。

    

    arXiv:2608.26877v1 公告类型：新 摘要：Derezinski和Warmuth的先前分析建立了普通体积采样的全尺寸采样恒等式、选定OLS无偏性和逆矩，而他们精确的任意固定响应损失和预测协方差公式位于秩-尺寸端点s=d。我们针对每个满秩固定池、响应和合法预算d <= s <= m，在普通索引固定尺寸体积采样后进行选定未加权最小二乘的情况下，建立了中心系数协方差的Loewner包络；其系数在满秩类上全局锐利。全局锐利性并不决定在当前池上的可达性。在正损失、严格内部预算和无共环条件下，一个仅特征边际nu_A给出了精确的固定设计谱相位：nu_A > 0当且仅当归一化谱包络对每个兼容残差是严格的，而nu_A = 0当且仅当某个兼容残差是谱紧的。

    arXiv:2608.26877v1 Announce Type: new  Abstract: Prior analyses by Derezinski and Warmuth established all-size sampling identities, selected-OLS unbiasedness, and inverse moments for ordinary volume sampling, while their exact arbitrary-fixed-response loss and prediction-covariance formulas are at the rank-size endpoint s=d. We establish a Loewner envelope for centered coefficient covariance for every full-rank fixed pool, response, and legal budget d <= s <= m under ordinary indexed fixed-size volume sampling followed by selected unweighted least squares; its coefficient is globally sharp over the full-rank class. Global sharpness does not determine attainability on the pool in hand. Under positive loss, strict-interior budgets, and no coloops, a feature-only margin nu_A gives the exact fixed-design spectral phase: nu_A > 0 if and only if the normalized spectral envelope is strict for every compatible residual, whereas nu_A = 0 if and only if some compatible residual is spectrally tig
    
[^68]: 对时空预测基准数据集和基线的批判性审计

    A Critical Audit of Spatiotemporal Forecasting Benchmark Datasets and Baselines

    [https://arxiv.org/abs/2608.20980](https://arxiv.org/abs/2608.20980)

    本文通过经典时间序列方法分析常用时空基准数据集，揭示无空间感知的线性模型比以往报告更具竞争力，质疑了现有基准数据集的判别可靠性。

    

    arXiv:2608.20980v1 公告类型：新 摘要：图神经网络（GNNs）通常用于具有空间图结构的多元时间序列的短期预测。尽管存在许多替代数据集，但该领域的方法创新主要针对一组有限的基准数据集进行评估，最著名的是Chickenpox、PedalMe、WikiMaths、METR-LA和PEMS-BAY。评估协议包含从历史平均值到经典机器学习方法的基线。这些基线通常表现出与GNNs相当的性能。在本研究中，我们退一步，通过经典时间序列方法分析基准数据集，以揭示为什么无空间感知的线性模型比先前报道的更具竞争力，从而进一步质疑上述广泛采用的数据集的判别可靠性。我们的统计分析提供了一套工具集，用于识别显著的...

    arXiv:2608.20980v1 Announce Type: new  Abstract: Graph neural networks (GNNs) are routinely employed for short-range forecasting on multivariate time series with a spatial graph structure. Despite the availability of many alternative datasets, method innovations within this domain are predominantly assessed against a rather limited set of benchmark datasets, most notably Chickenpox, PedalMe, WikiMaths, METR-LA, and PEMS-BAY. The evaluation protocols contain baselines spanning from historical averages to classical machine learning approaches. These baselines often show competitive performance compared to GNNs. In the present work, we take a step back and analyse the benchmark datasets via classical time series methods to uncover why spatially-unaware linear models pose a stronger competitor than previously reported, casting further doubt on the discriminative reliability of the aforementioned widely adopted datasets. Our statistical analysis provides a toolset for identifying significan
    
[^69]: QUASAR：通过损失感知重构降低量化感知训练中的损失下限

    QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction

    [https://arxiv.org/abs/2608.13966](https://arxiv.org/abs/2608.13966)

    本文提出QUASAR，一种在量化感知训练过程中持续进行轻量级损失感知重构的方法，以降低损失下限并提升低比特模型质量。

    

    随着大型语言模型推理转向更低精度，训练后量化（PTQ）变得越来越脆弱，使得量化感知训练（QAT）对于保持模型质量至关重要。然而，QAT在计算损失和代理梯度时，使用的是潜在全精度权重的有损重构，而更新则应用于潜在权重本身。这种不匹配可能导致次优的训练轨迹和更高的损失下限。二阶PTQ方法通过最小化损失感知重构误差来缓解类似差距，但对冻结模型执行一次可能需要数小时；在整个QAT过程中，随着权重变化而重复此过程是不切实际的。我们引入了QUASAR，一种QAT方法，它在训练循环中持续执行轻量级的损失感知重构，以降低损失下限并改进最终的低比特模型。在每一步训练中，QUASAR使用平方的指数移动平均。

    arXiv:2608.13966v1 Announce Type: cross  Abstract: As large language model inference shifts toward lower precision, post-training quantization (PTQ) becomes increasingly brittle, making quantization-aware training (QAT) essential for preserving model quality. However, QAT computes the loss and surrogate gradients using a lossy reconstruction of latent full-precision weights, while applying updates to the latent weights themselves. This mismatch can lead to suboptimal training trajectories and a higher loss floor. Second-order PTQ methods mitigate a similar gap by minimizing loss-aware reconstruction error, but doing it once for a frozen model can take hours; repeating this process throughout QAT as the weights evolve is impractical. We introduce QUASAR, a QAT method that continuously performs lightweight, loss-aware reconstruction in the training loop to lower the loss floor and improve the resulting low-bit model. At each training step, QUASAR uses the exponential moving average of sq
    
[^70]: 平均场朗之万动力学中统计特征学习的几何学

    The Geometry of Statistical Feature Learning in Mean-Field Langevin Dynamics

    [https://arxiv.org/abs/2606.31429](https://arxiv.org/abs/2606.31429)

    该论文通过基-纤维分解为统计特征学习建立了几何框架，证明球面平均场朗之万动力学的低温平稳分布在多指标模型中会集中于隐藏指标并形成多尖峰结构、以高概率实现参数恢复，且这一集中现象在温度约等于1处存在锐利的相变。

    

    我们为监督回归引入了一种统计特征学习的几何表述。特征学习通过基-纤维分解来定义：基是训练过程产生的特征侧几何结构，纤维是执行估计所用的学习特征空间。我们针对球面平均场朗之万动力学证明了这一性质，该动力学被视为负熵正则化经验风险的Wasserstein梯度流。在高斯多指标模型中，低温平稳分布集中于隐藏指标附近，形成多尖峰结构，并以高概率实现参数恢复——尽管负熵正则化本身是惩罚集中现象的。这种集中现象在温度 λ ≍ 1 处存在一个急剧的相变。在高斯单指标模型中，平稳测度满足集中性质，其奇偶性决定该测度位于 S_2^{d-1} 上还是其他位置（摘要在此处截断）。

    arXiv:2606.31429v2 Announce Type: cross  Abstract: We introduce a geometric formulation of statistical feature learning for supervised regression. Feature learning is defined through a base--fiber decomposition: the base is the feature-side geometry produced by training, and the fiber is the learned feature space where estimation is performed. We prove this property for spherical mean-field Langevin dynamics, viewed as the Wasserstein gradient flow of a negative entropy-regularized empirical risk. In Gaussian multi-index models, the low-temperature stationary distribution concentrates near the hidden indices, forms a multi-spike structure, and yields parameter recovery with high probability, even though negative entropy regularization penalizes concentration. This concentration has a sharp transition at temperature $\lambda\asymp 1$. In Gaussian single-index models, the stationary measure satisfies a concentration property, with parity determining whether it lives on $S_2^{d-1}$ or $\m
    
[^71]: 基于严格恰当评分规则学习概率滤波器

    Learning Probabilistic Filters with Strictly Proper Scoring Rules

    [https://arxiv.org/abs/2606.26497](https://arxiv.org/abs/2606.26497)

    本文提出PSEF方法，利用严格恰当评分规则训练基于Transformer的置换不变映射，仅通过合成数据实现贝叶斯滤波分布的逼近。

    

    针对部分观测且含噪声的动态系统的贝叶斯滤波，旨在在线推断系统状态随观测演变的条件分布。该贝叶斯滤波分布是不确定性量化的自然对象，但很少能作为监督学习目标直接获得。然而，我们通常可以利用预测模型生成合成系统轨迹及合成观测数据。本文提出了恰当评分集成滤波器（PSEF），这是一种基于训练分析映射的集成数据同化方法，仅通过合成状态-观测轨迹来逼近滤波分布。分析步骤被表示为一种基于置换不变性、Transformer架构的映射，它接收预测集成和观测作为输入，生成分析集成。训练基于严格恰当的评分规则——其中使用了能量评分。

    arXiv:2606.26497v1 Announce Type: new  Abstract: Bayesian filtering of partially and noisily observed dynamical systems seeks to infer the evolving conditional distribution of the state of a dynamical system, given observations, in an online fashion. This Bayesian filtering distribution is the natural object for uncertainty quantification, but it is rarely available as a supervised learning target. However, one can often use the forecast model to generate synthetic system trajectories, along with synthetic observations. We introduce the proper scoring ensemble filter (PSEF), an ensemble data assimilation method based on training an analysis map to approximate the filtering distribution using only synthetic state--observation trajectories. The analysis step is represented as a permutation-invariant, transformer-based map that takes as input a forecast ensemble and observations, producing an analysis ensemble. Training is based on strictly proper scoring rules -- with the energy score us
    
[^72]: 通过电子健康记录中稳健且灵活的知识迁移增强谱嵌入

    Enhancing Spectral Embedding through Robust and Flexible Knowledge Transfer in Electronic Health Records

    [https://arxiv.org/abs/2606.11570](https://arxiv.org/abs/2606.11570)

    该论文提出一种基于谱方法的无监督表示学习框架，通过放宽一对一信号对齐假设并采用两步嵌入流程，从更广泛人群中稳健且灵活地迁移知识，为样本量有限的罕见病电子健康记录数据生成高质量的低维嵌入表示。

    

    我们提出了一种基于谱方法的无监督表示学习框架，用于从电子健康记录中为罕见病队列的临床概念和患者推导低维嵌入表示，此类数据具有高维特征但样本量有限。为克服这一挑战，我们引入了一个从更广泛人群中提取的知识矩阵，该矩阵与罕见病队列共享部分重叠的子空间。我们的方法与现有方法的不同之处在于，放宽了潜在数据矩阵与知识矩阵之间严格的一对一信号对齐假设，从而允许更灵活、更符合实际的结构化共享形式。我们提出了一种新颖的两步谱嵌入流程：首先，识别并移除知识矩阵中的无关成分；然后，应用基于投影的方法分别恢复共享成分与异质成分。仿真实验和对真实数据的分析（摘要在此处截断）……

    arXiv:2606.11570v2 Announce Type: replace-cross  Abstract: We propose a spectral-based, unsupervised representation learning framework to derive low-dimensional embeddings for clinical concepts and patients in rare disease cohorts from electronic health records, where data are high-dimensional but sample sizes are limited. To overcome this challenge, we incorporate a knowledge matrix extracted from a broader population that shares a partially overlapping subspace with the rare-disease cohort. Our method departs from existing approaches by relaxing restrictive one-to-one signal-alignment assumptions between the latent data matrix and knowledge matrix, allowing more flexible and realistic forms of structured sharing. We introduce a novel two-step spectral embedding procedure: first, we identify and remove irrelevant components from the knowledge matrix; then, we apply a projection-based method to separately recover shared and heterogeneous components. Simulations and an analysis of a rea
    
[^73]: Express 语言建模

    Express Language Modeling

    [https://arxiv.org/abs/2606.10944](https://arxiv.org/abs/2606.10944)

    Express 是一种将非因果注意力近似转换为具有匹配保证的因果近似的工具，与 Thinformer 结合后实现了已知最佳的因果注意力近似保证，并通过高效的 Triton 实现显著超越 FlashAttention 2，解决了语言建模中的长上下文预填充、KV 缓存压缩和长文本解码等四大资源瓶颈。

    

    我们介绍了一种新工具 Express，用于将非因果注意力近似转换为具有匹配近似保证的因果近似。当与最先进的 Thinformer 近似相结合时，Express 改进了已知最佳的因果注意力保证：对于长度为 n 的序列，在仅使用 O(s) 内存和 O(s² log²(n)) 压缩开销的情况下，实现了 log^{3/2}(n)/s 的近似误差。我们将这些进展与高效的 I/O 感知 Triton 实现相结合，展示了相对于 FlashAttention 2 的显著加速，并利用 Express 克服了语言建模流程中的四个资源瓶颈：长上下文预填充、KV 缓存压缩、长文本内存受限解码以及长文本计算受限解码。

    arXiv:2606.10944v2 Announce Type: replace  Abstract: We introduce a new tool, Express, for converting a non-causal attention approximation into a causal approximation with matching approximation guarantees. When combined with the state-of-the-art Thinformer approximation, Express improves upon the best known causal attention guarantees, delivering $\log^{3/2}(n)/s$ approximation error with only $O(s)$ memory and $O(s^2 \log^2(n))$ compression overhead for a sequence of length $n$. We pair these developments with an efficient I/O-aware Triton implementation, demonstrate substantial speedups over FlashAttention 2, and use Express to overcome four resource bottlenecks in the language modeling pipeline: long-context prefill, KV cache compression, long-form memory-constrained decoding, and long-form compute-constrained decoding.
    
[^74]: 面向函数空间变分推断的流变换隐过程

    Flow-Transformed Implicit Processes for Function-Space Variational Inference

    [https://arxiv.org/abs/2606.01954](https://arxiv.org/abs/2606.01954)

    提出流变换隐过程（FTIP），通过超越高斯组合权重分布的限制，使有限维函数空间近似能够灵活表示非对称、重尾或多峰的后验不确定性。

    

    隐过程先验通过灵活的生成机制来定义函数上的分布，这使其在贝叶斯函数空间建模中颇具吸引力。然而，使用此类先验进行后验推断具有挑战性，因为其诱导的函数空间分布通常不具备闭式形式。一种实用的策略是使用有限个采样函数的集合来近似先验，然后将后验函数表示为这些样本的学习组合。现有方法通常在组合权重上放置高斯变分分布。尽管这种方法易于处理，但它限制了所能表示的后验不确定性的形状，尤其是当真实后验呈现非对称、重尾或多峰特性时。我们提出了流变换隐过程（FTIP），这是一种变分推断方法，使这种有限维函数空间近似更加灵活。

    arXiv:2606.01954v2 Announce Type: replace-cross  Abstract: Implicit-process priors define distributions over functions through flexible generative mechanisms, making them attractive for Bayesian function-space modelling. However, performing posterior inference with such priors is challenging because their induced function-space distributions are typically not available in closed form. One practical strategy is to approximate the prior using a finite collection of sampled functions, and then represent posterior functions as learned combinations of these samples. Existing approaches commonly place a Gaussian variational distribution over the combination weights. While tractable, this choice limits the shapes of posterior uncertainty that can be represented, especially when the true posterior is asymmetric, heavy-tailed, or multimodal. We propose Flow-Transformed Implicit Processes (FTIP), a variational inference method that makes this finite-dimensional function-space approximation more 
    
[^75]: 面向非可交换面板数据的在线保形预测

    Online Conformal Prediction for Non-Exchangeable Panel Data

    [https://arxiv.org/abs/2605.17705](https://arxiv.org/abs/2605.17705)

    提出了W-TQA方法，通过结合从单元历史学习的相似性权重与自适应误覆盖水平，解决了非可交换、部分观测面板数据中的在线保形预测问题，并证明了即使在反馈缺失情况下也能实现长期平均覆盖率保证。

    

    我们研究了部分观测面板数据中的在线保形预测问题：在观测每个目标结果之前，会观察到新的横截面同行结果；目标反馈可能是间歇性出现或完全缺失的，且单元和轮次均无需满足可交换性。我们提出了加权时间分位数调整方法，该方法将基于单元历史学习到的相似性权重与自适应的目标特定误覆盖水平相结合。我们证明了目标与同行之间的失配以及未揭示轮次上的覆盖率都是不可识别的，因此关于跨单元相似性和反馈机制的假设是不可避免的。我们以这种失配为条件对过去条件误覆盖率进行了界定，在画像相似性假设下量化了学习权重的代价，并证明了所实现的算法——在必要时回退到最大同行分数——在完全随机缺失的反馈条件和可行性条件下能够实现长期平均覆盖率。

    arXiv:2605.17705v2 Announce Type: replace-cross  Abstract: We study online conformal prediction in a partially observed panel: a new cross-section of peer outcomes is observed before each target outcome, target feedback may be intermittent or absent, and neither units nor rounds need be exchangeable. We propose Weighted Temporal Quantile Adjustment (W-TQA), which combines similarity weights learned from unit histories with an adaptive target-specific miscoverage level. We prove that neither target-peer mismatch nor coverage on unrevealed rounds is identifiable, so assumptions on cross-unit similarity and on the feedback mechanism cannot be avoided. We bound the past-conditional miscoverage in terms of this mismatch, quantify the cost of learning the weights under a profile-similarity assumption, and show that the implemented procedure, which falls back on the largest peer score, attains long-run average coverage under missing-completely-at-random feedback and a feasibility condition on
    
[^76]: 预测贝叶斯推断中的集中性与校准性

    Concentration and Calibration in Predictive Bayesian Inference

    [https://arxiv.org/abs/2605.00455](https://arxiv.org/abs/2605.00455)

    本文证明了预测贝叶斯推断的后验会集中到一个明确定义的量上，且该量及其不确定性量化完全由所选的前向预测模型决定，从而揭示了PBI的可靠性与校准性本质上取决于前向预测模型的选择。

    

    预测贝叶斯推断（PBI）是一种不依赖模型和先验的标准贝叶斯推断替代方法，它允许用户仅需为未来未观测数据指定一个前向预测模型，即可对所关注的泛函进行不确定性量化。该框架的灵活性与通用性催生了大量实现此方法的新算法以及众多实证应用，然而由此产生的推断对于底层目标统计泛函的可靠性仍不明确。本文证明，当将PBI用于某个总体泛函时，所得后验会集中到一个明确定义的量上，该量显式依赖于实现该方法底层预测递归所使用的前向预测模型。此外，前向预测模型完全决定了PBI所产生的不确定性量化。因此，我们的结果表

    arXiv:2605.00455v2 Announce Type: replace-cross  Abstract: Predictive Bayesian inference (PBI) represents a model-and prior-agnostic approach to standard Bayesian inference which allows users to quantify uncertainty for a functional of interest only by specifying a forward predictive model for future unobserved data. The flexibility and generality of this framework have led to a host of novel algorithms for implementing this approach, and many empirical applications, yet the reliability of the resulting inferences for the underlying statistical functional of interest remains unclear. Herein, we demonstrate that when using PBI for a population functional of interest, the resulting posterior concentrates onto a well-defined quantity that explicitly depends on the forward predictive model used to implement the predictive recursion underlying the method. Furthermore, the forward predictive model entirely determines the uncertainty quantification produced in PBI. Consequently, our results s
    
[^77]: 十年深度时间序列预测研究综述

    Deep Time-Series Forecasting in 10 Years: A Survey

    [https://arxiv.org/abs/2603.19899](https://arxiv.org/abs/2603.19899)

    本文从自相关性建模的统一视角系统综述了十年来的深度时间序列预测研究，首次提出同时涵盖骨干架构与损失函数的分类体系，并据此剖析了现有文献的动机与洞见。

    

    自相关性是时间序列的一种普遍特性，即每个观测值都依赖于其前序观测值。在深度时间序列预测中，这带来了两个核心挑战：（1）设计骨干网络架构以建模历史序列中的自相关性；（2）设计损失函数以建模标签序列中的自相关性。近年来，相关研究在应对这些挑战方面取得了长足进展，但目前仍缺乏对这两个方面进行系统性考察的综述。为填补这一空白，本文从自相关性建模的视角对深度时间序列预测进行了综述，并做出了超越现有综述工作的两点贡献：其一，提出了一个同时涵盖骨干网络架构与损失函数的分类体系，而以往综述对损失函数的覆盖较为有限；其二，从统一的自相关性视角分析了所调研文献背后的动机与洞见，提供了一个整体性的（认识框架）。

    arXiv:2603.19899v2 Announce Type: replace-cross  Abstract: Autocorrelation is a common property of time-series, where each observation is dependent on its predecessors. In deep time-series forecasting, it raises two central challenges: (1) designing backbone architectures to model autocorrelation in history sequences, and (2) devising loss functions to model autocorrelation in label sequences. Recent studies have made strides in tackling these challenges, but a systematic survey examining both aspects remains lacking. To bridge this gap, this paper reviews deep time-series forecasting from an autocorrelation modeling perspective, offering two contributions beyond existing surveys. First, it introduces a taxonomy that jointly covers both backbone architectures and loss functions, whereas prior surveys provide limited coverage of the latter. Second, it analyzes the motivations and insights underlying the surveyed literature from a unified autocorrelation perspective, providing a holistic
    
[^78]: 面向鲁棒与非光滑凸分布式学习的快速高效异步Gossip算法

    Fast and Efficient Asynchronous Gossip Algorithm for Robust and Non-Smooth Convex Decentralized Learning

    [https://arxiv.org/abs/2601.20571](https://arxiv.org/abs/2601.20571)

    本文提出Goal-PD，一种异步Gossip原始-对偶算法，每个节点仅维护两个变量而与网络度数无关，实现了几乎必然收敛与线性收敛，并通过分布式均值估计中的成对平均特例与经典Gossip算法建立了直接联系。

    

    面向分布式非光滑凸优化的异步原始-对偶方法通常要求每个节点维护 $\mathcal{O}(d)$ 个辅助变量，其中 $d$ 为该节点的度数。这种对度数的依赖增加了内存需求，并可能放大过期信息的影响，尤其是在密集网络中。受分布式学习中节约内存管理这一挑战的启发，我们提出了 Goal-PD，一种基于异步Gossip的原始-对偶算法，无论节点度数如何，每个节点仅需维护两个变量。我们建立了Goal-PD几乎必然收敛到所研究优化问题最小化子的结论，并证明了当目标函数为分段线性二次函数时的线性收敛性。对于分布式均值估计，我们证明成对平均是Goal-PD的一个特例，这在所提出的原始-对偶框架与经典Gossip算法之间建立了直接联系。

    arXiv:2601.20571v3 Announce Type: replace-cross  Abstract: Asynchronous primal-dual methods for decentralized non-smooth convex optimization often require each node to maintain $\mathcal{O}(d)$ auxiliary variables, where $d$ is its degree. This dependence on degree increases memory requirements and can amplify the effects of stale information, especially in dense networks. Motivated by the challenge of frugal memory management in decentralized learning, we introduce Goal-PD, an asynchronous gossip-based primal-dual algorithm that maintains only two variables per node, regardless of the node's degree. We establish almost-sure convergence of Goal-PD to a minimizer of the underlying optimization problem, and prove linear convergence when the objective functions are piecewise linear-quadratic. For decentralized mean estimation, we show that pairwise averaging is a special case of Goal-PD, which establishes a direct link between the proposed primal-dual framework and classical gossip. Exper
    
[^79]: 基于混合整数优化的交叉性公平

    Intersectional Fairness via Mixed-Integer Optimization

    [https://arxiv.org/abs/2601.19595](https://arxiv.org/abs/2601.19595)

    本文提出一个基于混合整数优化（MIO）的统一框架，训练同时具备交叉公平性和内在可解释性的分类器，证明了两种交叉公平性度量（MSD 与 SPSF）在检测最不公平子群体上的等价性，并能将交叉偏见有效控制在可接受阈值以下。

    

    在金融和医疗等高风险领域部署人工智能，需要既公平又透明的模型。虽然包括欧盟《人工智能法案》在内的监管框架要求减轻偏见，但它们对偏见的定义故意保持模糊。与现有研究一致，我们认为真正的公平需要在受保护群体的交叉点上解决偏见问题。我们提出了一个统一框架，利用混合整数优化（MIO）来训练具有交叉公平性和内在可解释性的分类器。我们证明了两种交叉公平性度量（MSD 和 SPSF）在检测最不公平子群体方面的等价性，并通过实验证明我们基于 MIO 的算法在发现偏见方面提升了性能。我们训练了高性能、可解释的分类器，将交叉偏见限制在可接受的阈值以下，为监管合规提供了稳健的解决方案。

    arXiv:2601.19595v2 Announce Type: replace-cross  Abstract: The deployment of Artificial Intelligence in high-risk domains, such as finance and healthcare, necessitates models that are both fair and transparent. While regulatory frameworks, including the EU's AI Act, mandate bias mitigation, they are deliberately vague about the definition of bias. In line with existing research, we argue that true fairness requires addressing bias at the intersections of protected groups. We propose a unified framework that leverages Mixed-Integer Optimization (MIO) to train intersectionally fair and intrinsically interpretable classifiers. We prove the equivalence of two measures of intersectional fairness (MSD and SPSF) in detecting the most unfair subgroup and empirically demonstrate that our MIO-based algorithm improves performance in finding bias. We train high-performing, interpretable classifiers that bound intersectional bias below an acceptable threshold, offering a robust solution for regulat
    
[^80]: 基于核化Stein差异的计算高效拟合优度检验

    Computationally efficient goodness-of-fit tests through kernelized Stein discrepancy

    [https://arxiv.org/abs/2512.20007](https://arxiv.org/abs/2512.20007)

    本文提出一种基于核化Stein差异的计算高效半参数拟合优度检验，并设计了无需重新拟合模型或从中采样的影响调整野自助法来确定检验的显著性水平。

    

    具有难解归一化常数的模型在统计学和机器学习中被广泛使用。评估此类模型的适用性面临重大挑战：从拟合后的模型中获取样本通常需要复杂的采样算法。此外，模型拟合有时需要迭代数值优化，这使得需要重复重新拟合的自助法程序在计算上代价高昂。在本文中，我们利用基于核的检验框架，开发了一种基于核化Stein差异的通用半参数拟合优度检验。我们在一般的干扰参数估计下建立了检验统计量的相合性和渐近零分布。为了构造水平为 $\alpha$ 的检验，我们提出了一种新颖的影响调整野自助法，该方法既不需要重新拟合模型，也不需要从模型中采样。我们证明了所提出的自助检验程序在零假设下的相合性，并……

    arXiv:2512.20007v3 Announce Type: replace-cross  Abstract: Models with intractable normalizing constants are widely used in statistics and machine learning. Assessing the adequacy of such models poses significant challenges: obtaining samples from the fitted model often requires sophisticated sampling algorithms. Moreover, model fitting sometimes requires iterative numerical optimization, making bootstrap procedures that require repeated refitting computationally expensive. In this paper, we leverage the kernel-based testing framework to develop a general semiparametric goodness-of-fit test based on the kernelized Stein discrepancy. We establish the consistency and the asymptotic null distribution of the test statistic under general nuisance estimation. To produce a level-$\alpha$ test, we propose a novel influence-adjusted wild bootstrap that requires neither refitting the model nor sampling from it. We prove the consistency of the proposed bootstrap test procedure under the null and 
    
[^81]: 面向多步时间序列预测模型训练的二次型直接预测方法

    Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models

    [https://arxiv.org/abs/2511.00053](https://arxiv.org/abs/2511.00053)

    该论文提出了一种新颖的二次型加权学习目标，通过加权矩阵的非对角元素捕捉未来步骤间的标签自相关效应，同时利用非均匀对角元素为不同预测步骤设置异构任务权重，从而同时解决传统均方误差目标的两个缺陷，提升多步时间序列预测模型的训练效果。

    

    arXiv:2511.00053v2 公告类型： replace-cross 摘要：学习目标的设计是训练时间序列预测模型的核心。现有的学习目标（如均方误差）大多将每个未来预测步骤视为独立的、等权重的任务，这导致了以下两个挑战：（1）它们忽视了未来步骤之间的标签自相关效应，导致学习目标存在偏差；（2）它们未能为对应不同未来预测步骤的各项预测任务设置异构的任务权重，从而限制了预测性能。为填补这一空白，我们提出了一种新颖的二次型加权学习目标，能够同时解决上述两个问题。具体而言，加权矩阵的非对角元素用于刻画未来步骤之间的标签自相关效应，而非均匀的对角元素则用于匹配具有不同预测步骤的各项预测任务所偏好的权重。在此基础上，我们提出了二次型直接预测（Quadratic Direct Forecast）方法……

    arXiv:2511.00053v2 Announce Type: replace-cross  Abstract: The design of learning objectives is central to training time-series forecasting models. Existing learning objectives such as mean squared error mostly treat each future step as an independent, equally weighted task, which leads to the following two challenges: (1) they overlook the label autocorrelation effect among future steps, leading to biased learning objectives; (2) they fail to set heterogeneous task weights for different forecasting tasks corresponding to varying future steps, limiting the forecasting performance. To fill this gap, we propose a novel quadratic-form weighted learning objective, addressing both issues simultaneously. Specifically, the off-diagonal elements of the weighting matrix account for the label autocorrelation effect, whereas the non-uniform diagonals are expected to match the preferred weights of the forecasting tasks with varying future steps. On this basis, we propose a Quadratic Direct Forecas
    
[^82]: 动作驱动过程用于连续时间控制

    Action-Driven Processes for Continuous-Time Control

    [https://arxiv.org/abs/2510.26672](https://arxiv.org/abs/2510.26672)

    本文通过动作驱动过程统一了随机过程与强化学习的视角，证明最小化策略驱动分布与奖励驱动分布之间的KL散度等价于最大熵强化学习，并将其应用于脉冲神经网络。

    

    强化学习的核心在于动作——即针对环境观察所作出的决策。动作在随机过程建模中同样具有基础性地位，因为它们触发不连续的状态转移，并使信息能够在大型复杂系统中流动。本文通过动作驱动过程统一了随机过程与强化学习这两个视角，并展示了其在脉冲神经网络中的应用。借助“控制即推断”的思想，我们证明：对于适当定义的动作驱动过程，最小化策略驱动的真实分布与奖励驱动的模型分布之间的Kullback-Leibler散度，等价于最大熵强化学习。

    arXiv:2510.26672v3 Announce Type: replace-cross  Abstract: At the heart of reinforcement learning are actions -- decisions made in response to observations of the environment. Actions are equally fundamental in the modeling of stochastic processes, as they trigger discontinuous state transitions and enable the flow of information through large, complex systems. In this paper, we unify the perspectives of stochastic processes and reinforcement learning through action-driven processes, and illustrate their application to spiking neural networks. Leveraging ideas from control-as-inference, we show that minimizing the Kullback-Leibler divergence between a policy-driven true distribution and a reward-driven model distribution for a suitably defined action-driven process is equivalent to maximum entropy reinforcement learning.
    
[^83]: 超越半圆律：具有指定平衡态的自由扩散模型

    Beyond the Semicircle: Free Diffusion Models with Prescribed Equilibria

    [https://arxiv.org/abs/2510.22778](https://arxiv.org/abs/2510.22778)

    该论文发现状态依赖的自由波动率能突破常系数自由扩散只能收敛到半圆律的固有局限，并为任意充分正则的紧支撑目标谱分布显式构造出具有指定平衡态的自由扩散模型。

    

    一类日益增多的机器学习对象——协方差矩阵与Gram矩阵、核矩阵与注意力矩阵、MIMO信道矩阵、密度算子——本质上以谱（特征值分布）而非坐标向量的形式存在。为这类数据构建去噪扩散模型时，逐坐标地对特征值加噪不仅是形式上的不优雅：它会收敛到错误的极限，因为特征值之间存在排斥效应而非独立运动。自由概率论提供了正确的前向过程，用自由卷积取代经典卷积，用Voiculescu共轭变量取代得分函数，但常系数自由扩散自身存在一个隐蔽的局限：无论目标分布是什么，其唯一可能的平衡态都是半圆律。我们证明，状态依赖的自由波动率可以消除这一限制。对于任何充分正则的紧支撑目标分布，我们通过一个闭式……显式构造……

    arXiv:2510.22778v4 Announce Type: replace-cross  Abstract: A growing class of machine-learning objects -- covariance and Gram matrices, kernel and attention matrices, MIMO channel matrices, density operators -- are naturally spectra rather than coordinate vectors. Building a denoising diffusion model for such data by corrupting eigenvalues coordinatewise is not merely elegant: it converges to the wrong limit, because eigenvalues repel rather than move independently. Free probability theory supplies the correct forward process, with free convolution replacing classical convolution and Voiculescu's conjugate variable replacing the score, but constant-coefficient free diffusions have a hidden limitation of their own: their only possible equilibrium is the semicircular law, whatever the target distribution looks like. We show that state-dependent free volatility removes this restriction. For any sufficiently regular compactly supported target law, we explicitly construct, through a closed-
    
[^84]: 利用Wasserstein分布鲁棒优化改进Mixup校准

    Improving Mixup Calibration with Wasserstein Distributionally Robust Optimization

    [https://arxiv.org/abs/2506.17874](https://arxiv.org/abs/2506.17874)

    本文提出DRO-Augment框架，将Wasserstein分布鲁棒优化与Mixup数据增强相结合，有效缓解了腐蚀鲁棒性与模型校准之间的权衡，在保持腐蚀准确率的同时显著降低了期望校准误差（ECE）。

    

    在许多实际应用中，确保深度神经网络（DNN）的鲁棒性和稳定性至关重要，特别是对于面临各种输入扰动的图像分类任务。虽然基于Mixup的数据增强技术已被广泛采用，以增强训练模型的抗扰动能力，但我们的实验揭示了一个重要的腐蚀鲁棒性-校准权衡：更强的基于Mixup的增强可以提高对腐蚀数据的鲁棒性，但同时会显著增加期望校准误差（ECE）。为了解决这一挑战，我们提出了DRO-Augment框架，该框架将Wasserstein分布鲁棒优化（W-DRO）与各种基于Mixup的数据增强策略相结合，以缓解这种权衡。我们的方法在强Mixup增强下大幅降低了ECE，同时在CIFAR-10、CIFAR-100等数据集上基本保持了腐蚀准确率。

    arXiv:2506.17874v3 Announce Type: replace-cross  Abstract: In many real-world applications, ensuring the robustness and stability of deep neural networks (DNNs) is crucial, particularly for image classification tasks that encounter various input perturbations. While Mixup-based data augmentation techniques have been widely adopted to enhance the resilience of trained models against such perturbations, our experiments reveal an important corruption robustness-calibration trade-off: stronger Mixup-based augmentation can improve robustness against corrupted data while substantially increasing expected calibration error (ECE). To address this challenge, we introduce DRO-Augment, a framework that integrates Wasserstein Distributionally Robust Optimization (W-DRO) with various Mixup-based data augmentation strategies to mitigate this trade-off. Our method substantially reduces ECE under strong Mixup-based augmentation while largely preserving corruption accuracy across CIFAR-10, CIFAR-100, C
    
[^85]: 具有隐藏混杂因素的线性常微分方程系统的可辨识性分析

    Identifiability Analysis of Linear ODE Systems with Hidden Confounders

    [https://arxiv.org/abs/2410.21917](https://arxiv.org/abs/2410.21917)

    本文系统分析了含隐藏混杂因素的线性常微分方程系统的可辨识性，分别研究了潜在混杂因素无因果关系但遵循特定函数形式（如时间多项式）演化，以及潜在混杂因素之间具有由有向无环图描述的因果依赖关系这两种情况，填补了该领域的空白。

    

    arXiv:2410.21917v3 公告类型：replace-cross 摘要：对线性常微分方程（ODE）系统进行可辨识性分析是针对这些系统做出可靠因果推断的必要前提。尽管在系统完全可观测的场景下，可辨识性已得到充分研究，但当潜在变量与系统发生交互作用时，其可辨识性的条件仍未被探索。本文旨在通过系统性地分析包含隐藏混杂因素的线性ODE系统的可辨识性来填补这一空白。具体而言，我们研究了此类系统的两种情况。在第一种情况中，潜在混杂因素之间不存在因果关系，但其演化遵循特定的函数形式，例如时间 $t$ 的多项式函数。随后，我们将这一分析扩展到隐藏混杂因素之间存在因果依赖的场景，其中潜在变量的因果结构由有向无环图（DAG）描述。

    arXiv:2410.21917v3 Announce Type: replace-cross  Abstract: The identifiability analysis of linear Ordinary Differential Equation (ODE) systems is a necessary prerequisite for making reliable causal inferences about these systems. While identifiability has been well studied in scenarios where the system is fully observable, the conditions for identifiability remain unexplored when latent variables interact with the system. This paper aims to address this gap by presenting a systematic analysis of identifiability in linear ODE systems incorporating hidden confounders. Specifically, we investigate two cases of such systems. In the first case, latent confounders exhibit no causal relationships, yet their evolution adheres to specific functional forms, such as polynomial functions of time $t$. Subsequently, we extend this analysis to encompass scenarios where hidden confounders exhibit causal dependencies, with the causal structure of latent variables described by a Directed Acyclic Graph (
    
[^86]: FreDF: 在频域中学习预测

    FreDF: Learning to Forecast in Frequency Domain

    [https://arxiv.org/abs/2402.02399](https://arxiv.org/abs/2402.02399)

    FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。

    

    时间序列建模在历史序列和标签序列中都面临自相关的挑战。当前的研究主要集中在处理历史序列中的自相关问题，但往往忽视了标签序列中的自相关存在。具体来说，新兴的预测模型主要遵循直接预测（DF）范式，在标签序列中假设条件独立性下生成多步预测。这种假设忽视了标签序列中固有的自相关性，从而限制了基于DF的模型的性能。针对这一问题，我们引入了频域增强直接预测（FreDF），通过在频域中学习预测来避免标签自相关的复杂性。我们的实验证明，FreDF在性能上大大超过了包括iTransformer在内的现有最先进方法，并且与各种预测模型兼容。

    Time series modeling is uniquely challenged by the presence of autocorrelation in both historical and label sequences. Current research predominantly focuses on handling autocorrelation within the historical sequence but often neglects its presence in the label sequence. Specifically, emerging forecast models mainly conform to the direct forecast (DF) paradigm, generating multi-step forecasts under the assumption of conditional independence within the label sequence. This assumption disregards the inherent autocorrelation in the label sequence, thereby limiting the performance of DF-based models. In response to this gap, we introduce the Frequency-enhanced Direct Forecast (FreDF), which bypasses the complexity of label autocorrelation by learning to forecast in the frequency domain. Our experiments demonstrate that FreDF substantially outperforms existing state-of-the-art methods including iTransformer and is compatible with a variety of forecast models.
    

