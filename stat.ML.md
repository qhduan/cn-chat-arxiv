# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [When are likely answers right? On Sequence Probability and Correctness in LLMs](https://arxiv.org/abs/2606.27359) | 本文发现序列概率仅在固定数据集内能预测跨提示-答案对的正确性，但不能推广到跨解码方法或超参数调整的解码决策中。 |
| [^2] | [All you need is log](https://arxiv.org/abs/2606.27349) | 本文刻画了在数据处理下单调且在独立乘积上可加的多分布泛函的唯一形式，即通过多路重合散度在四层参数空间上的正积分来统一表示，解决了Rényi族向多分布泛化的开放问题。 |
| [^3] | [Fast algorithms for learning a Gaussian under halfspace truncation with optimal sample complexity](https://arxiv.org/abs/2606.27298) | 提出了一种在半空间截断下学习高斯分布的算法，其样本和时间复杂度均达到理论最优，且与无截断情况下的最优复杂度一致。 |
| [^4] | [Ribbon: Scalable Approximation and Robust Uncertainty Quantification](https://arxiv.org/abs/2606.27269) | Ribbon通过影响函数线性化替代重复重拟合，实现了对贝叶斯自助不确定性的高效近似，并引入可调集中参数实现校准。 |
| [^5] | [The Geometry of Updates: Fisher Alignment at Vocabulary Scale](https://arxiv.org/abs/2606.27242) | 本文提出FisherSketch方法，通过将头费舍尔对齐等价为联合激活-误差空间中核均值嵌入的余弦值，实现了在词汇规模下高效、可识别的无训练源选择，解决了传统表示度量不可识别而经典几何度量计算成本过高的问题。 |
| [^6] | [Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference](https://arxiv.org/abs/2606.27090) | 本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。 |
| [^7] | [Decision-Aligned Evaluation of Uncertainty Quantification](https://arxiv.org/abs/2606.26990) | 本文提出决策对齐标准，发现传统不确定性量化指标常与下游决策效用不一致，并设计先验加权效用指标以实现与决策效用的对齐，从而修正了现有评估协议的缺陷。 |
| [^8] | [XMSE-Aware Adaptive Empirical Bayes Estimation](https://arxiv.org/abs/2606.26975) | 本文通过将超额均方误差（XMSE）分析从诊断工具转化为设计原则，提出了一种在最大似然和经验贝叶斯之间自适应插值的混合估计器，并证明了其在二阶意义下不劣于两者。 |
| [^9] | [Asymptotically Optimal Learning for Parametric Prophet Inequalities](https://arxiv.org/abs/2606.26893) | 针对参数未知的指数型参数族先知不等式问题，提出了一种仅靠在线观测即可达到最优渐近竞争比的置信度动态规划策略，无需离线样本。 |
| [^10] | [Data-Driven Duration Management -- Term Structure Forecasting Using Machine Learning](https://arxiv.org/abs/2606.26815) | 本文提出将神经网络模型（尤其是受经典模型启发的架构）与宏观经济变量结合，用于美国和欧洲债券期限结构预测，并证明其在统计精度和交易策略经济价值上均优于传统模型。 |
| [^11] | [Escaping Iterative Parameter-Space Noise: Differentially Private Learning with a Hypernetwork](https://arxiv.org/abs/2606.26772) | 提出了一种基于超网络的新框架，通过仅一次向低维数据集表示注入隐私噪声，避免了迭代参数空间噪声，从而显著降低差分隐私学习中噪声的不利影响。 |
| [^12] | [Scalable Operator Learning via Nystr\"om Approximation With Denoising Applications](https://arxiv.org/abs/2606.26652) | 本文提出一种基于奈斯特龙子采样的高效算子学习算法，在广泛源条件下达到极小化最优收敛率，并成功应用于通用函数去噪问题。 |
| [^13] | [$\lambda$-PSD: Scalable Approximate SNR-Optimised Polynomial Stein Discrepancies](https://arxiv.org/abs/2606.26621) | 本文揭示了多项式斯坦差异中SNR²随阶数指数衰减的问题，并提出一种基于瑞利商最大化的可扩展近似加权方案λ-PSD，有效避免了这一失效模式。 |
| [^14] | [Learning Probabilistic Filters with Strictly Proper Scoring Rules](https://arxiv.org/abs/2606.26497) | 本文提出PSEF方法，利用严格恰当评分规则训练基于Transformer的置换不变映射，仅通过合成数据实现贝叶斯滤波分布的逼近。 |
| [^15] | [A probabilistic framework for online test-time adaptation](https://arxiv.org/abs/2606.26457) | 提出了一种基于状态空间模型的概率框架，用于在线测试时自适应，以应对训练与测试分布之间的偏移。 |
| [^16] | [Explainable Outlier Detection for Interval-valued Data](https://arxiv.org/abs/2606.26307) | 提出了一种基于沙普利值的区间数据异常检测可解释性方法，通过闭式表达式高效分解变量贡献，实现中心、范围及交叉项的精细分析。 |
| [^17] | [The Role of Input Dimensionality in the Emergence and Targeted Control of Adversarial Examples](https://arxiv.org/abs/2606.26207) | 通过实证研究揭示输入维度增加会使对抗样本更易构造，并发现真实图像类别的强经验局部化特性超出传统高维几何理论假设。 |
| [^18] | [Statistical and Structural Approaches to Algorithmic Fairness](https://arxiv.org/abs/2606.26200) | 本论文指出现代算法公平性方法的两大根本缺陷——依赖确定性点估计审计和将个体视为孤立实体，并提出改进方案。 |
| [^19] | [Representation Costs in Data Science: Foundations and the Quasi-Banach Spaces of Deep Neural Networks](https://arxiv.org/abs/2606.14954) | 本文提出了一个统一框架，通过参数空间正则化器分析数据科学中的表示成本，揭示了参数化方法与其原生函数空间之间的联系，并将核方法、小波和神经网络等经典方法统一为特例。 |
| [^20] | [Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning](https://arxiv.org/abs/2606.04574) | 本研究通过分层配对选择方法和专有执行模型，结合深度强化学习，显著提升了加密货币市场中配对交易的稳健性与收益表现。 |
| [^21] | [FoReco and FoRecoML: A Unified Toolbox for Forecast Reconciliation in R](https://arxiv.org/abs/2604.27696) | FoReco 与 FoRecoML 是 R 语言中首个统一涵盖截面、时间及跨时间预测协调的综合性工具箱，兼顾易用性与灵活性，填补了该领域软件空白。 |
| [^22] | [A unifying view of contrastive learning, importance sampling, and bridge sampling for energy-based models](https://arxiv.org/abs/2604.08116) | 本文提出了一个统一框架，将噪声对比估计、反向逻辑回归、多重重要性采样和桥接采样在基于能量的模型中联系起来，揭示了它们之间的等价关系，并促进了更高效估计量的开发。 |
| [^23] | [Quantum Maximum Likelihood Prediction via Hilbert Space Embeddings](https://arxiv.org/abs/2602.18364) | 本文通过将经验概率分布嵌入量子态并最小化量子相对熵，提出了一种量子最大似然预测方法，并为其在经典和量子大语言模型中的统一应用提供了非渐近性能保证。 |
| [^24] | [Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates](https://arxiv.org/abs/2602.17683) | 该论文提出了一种概率性预测框架，通过分离历史数据编码与未来协变量、引入时间距离加权损失函数，解决了稀疏不规则卫星观测下的田块级NDVI短期预测挑战。 |
| [^25] | [Causal Inference with the Napkin Graph](https://arxiv.org/abs/2512.19861) | 本文提出了一种针对“餐巾”图结构的因果效应识别与估计方法，通过非标准比率和基于影响函数的双重稳健估计量，在Verma约束下实现了半参数有效推断。 |
| [^26] | [Machine Learning-based Unfolding for Cross Section Measurements in the Presence of Nuisance Parameters](https://arxiv.org/abs/2512.07074) | 本文基于OmniFold算法，提出了Profile OmniFold算法，将机器学习展开方法扩展至包含干扰参数，从而更准确地处理探测器模拟中的不确定性。 |
| [^27] | [No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference](https://arxiv.org/abs/2505.20178) | 本文通过有限样本分析证明，PPI++方法并非总是优于仅使用黄金标准标签，其优势仅在伪标签与黄金标准标签的相关性高于特定阈值时成立。 |
| [^28] | [Communication-Efficient, 2D Parallel Stochastic Gradient Descent for Distributed-Memory Optimization](https://arxiv.org/abs/2501.07526) | 本文提出一种名为HybridSGD的二维并行随机梯度下降方法，通过整合一维s步SGD和联邦平均SGD的优势，在分布式内存集群中实现了通信效率与性能的连续权衡，并理论验证了其在收敛性、计算、通信和内存方面的综合优势。 |
| [^29] | [Learning from a Biased Sample](https://arxiv.org/abs/2209.01754) | 本文提出了一种新的条件Γ-有偏抽样模型来量化训练数据中的抽样偏差，并利用分布鲁棒优化框架开发了一种元方法，以在部署时仍能获得良好性能的决策规则。 |
| [^30] | [Theory of the Frequency Principle for General Deep Neural Networks](https://arxiv.org/abs/1906.09235) | 本文严格证明了通用深度神经网络在训练中从低频到高频学习的频率原理，并提供了针对不同训练阶段的理论定理，适用于多种激活函数、数据分布和损失函数。 |

# 详细

[^1]: 何时高概率答案是正确的？论大语言模型中序列概率与正确性的关系

    When are likely answers right? On Sequence Probability and Correctness in LLMs

    [https://arxiv.org/abs/2606.27359](https://arxiv.org/abs/2606.27359)

    本文发现序列概率仅在固定数据集内能预测跨提示-答案对的正确性，但不能推广到跨解码方法或超参数调整的解码决策中。

    

    arXiv:2606.27359v1 公告类型：跨领域 摘要：大语言模型的许多解码方法可以被理解为将概率质量向模型更可能输出的结果转移，这种转移既可以在词元级别局部进行，也可以在序列级别全局进行。因此，它们的成功取决于一个基本问题：序列概率，即给定提示后延续文本的条件概率，何时真正与正确性一致？在本文中，我们旨在跨解码方法、模型和基准，在四个层面上量化这种关系：跨解码方法、方法内跨超参数、数据集内跨提示-答案对、以及针对同一提示的重复响应。我们发现，在固定数据集中，更高的序列概率通常能预测跨提示-答案对的正确性。然而，这种关系通常不能推广到解码决策中：通过改变超参数来增加序列概率并不总能保证更高的正确性。

    arXiv:2606.27359v1 Announce Type: cross  Abstract: Many decoding methods for large language models can be understood as shifting probability mass toward outputs that are more likely under the model, either locally at the token level or globally at the sequence level. Therefore, their success depends on a fundamental question: when does sequence probability, that is, the conditional probability of a continuation given a prompt, actually align with correctness? In this paper, we set out to quantify this relationship across decoding methods, models, and benchmarks at four levels: across decoding methods, across hyperparameters within a method, across prompt-answer pairs within a dataset, and across repeated responses to the same prompt. We find that higher sequence probability is often predictive of correctness across prompt-answer pairs within a fixed dataset. However, this relationship does not generally transfer to decoding decisions: increasing sequence probability by changing hyperpa
    
[^2]: 你只需要对数

    All you need is log

    [https://arxiv.org/abs/2606.27349](https://arxiv.org/abs/2606.27349)

    本文刻画了在数据处理下单调且在独立乘积上可加的多分布泛函的唯一形式，即通过多路重合散度在四层参数空间上的正积分来统一表示，解决了Rényi族向多分布泛化的开放问题。

    

    arXiv:2606.27349v1 公告类型：交叉 摘要：比较两个概率分布是统计学和机器学习的基本构建块，而正确的族已被充分理解：阶数为α∈[0,∞]的Rényi散度是在数据处理下单调且在独立乘积上可加的唯一族。许多问题却需要同时比较两个以上的分布——多群体公平性、多先验PAC-Bayes界、多假设检验——而Rényi族的多分布泛化正确形式一直是一个开放问题。我们对此进行了刻画。每个在数据处理下单调且在独立乘积上可加的W元分布泛函，都可以表示为多路重合散度C_α(π_1,…,π_W) := -log∫ π_1^{α_1}…π_W^{α_W}（其中∑_k α_k = 1）在具有四个分层参数空间上的正积分：单纯形内部；混合符号指数锥。

    arXiv:2606.27349v1 Announce Type: cross  Abstract: Comparing two probability distributions is a basic building block of statistics and machine learning, and the right family is well understood: the R\'enyi divergences of order $\alpha\in[0,\infty]$ are the unique family monotone under data processing and additive on independent products. Many problems instead compare more than two distributions at once -- multi-population fairness, multi-prior PAC-Bayes bounds, multi-hypothesis testing -- and the right multi-distribution generalization of the R\'enyi family has been an open question.   We characterize it. Every functional of $W$-tuples of distributions that is monotone under data processing and additive on independent products is a positive integral of multi-way coincidence divergences $C_{\alpha}(\pi_1,\dots,\pi_W) := -\log\int \pi_1^{\alpha_1}\cdots\pi_W^{\alpha_W}$ (with $\sum_k \alpha_k = 1$) over a parameter space with four strata: the simplex interior; mixed-sign exponent cones (
    
[^3]: 在半空间截断下以最优样本复杂度学习高斯分布的快速算法

    Fast algorithms for learning a Gaussian under halfspace truncation with optimal sample complexity

    [https://arxiv.org/abs/2606.27298](https://arxiv.org/abs/2606.27298)

    提出了一种在半空间截断下学习高斯分布的算法，其样本和时间复杂度均达到理论最优，且与无截断情况下的最优复杂度一致。

    

    我们研究了将高维高斯分布截断到未知半空间这一基本学习问题。Lee、Mehrotra 和 Zampetakis（FOCS'24）最近首次提出了该问题的多项式时间算法，但其样本和时间复杂度界限并非最优。在非平凡截断下，对于任意目标精度 $\varepsilon > 0$ 和维度 $d$，我们给出了一种高效算法，该算法使用 $n = \tilde{O}(d^2/\varepsilon^2)$ 个样本，并以总变差距离误差 $\varepsilon$ 学习到原始高斯分布。我们的算法速度也很快：其运行时间主要由计算经验协方差矩阵的成本决定。我们的样本和时间复杂度在 $d$ 和 $\varepsilon$ 方面均达到最优，即使在没有截断的情况下也是如此：就此而言，我们可以在半空间截断下免费学习高斯分布。我们结果的关键要素是对截断后高斯分布低阶矩的一种新颖重新解释。

    arXiv:2606.27298v1 Announce Type: cross  Abstract: We study the fundamental problem of learning a high-dimensional Gaussian truncated to an unknown halfspace. Lee, Mehrotra and Zampetakis (FOCS'24) recently obtained the first polynomial time algorithm for this problem, but their resulting sample and time complexity bounds are not optimal. Under non-trivial truncation, for any target accuracy $\varepsilon > 0$ and dimension $d$ we give an efficient algorithm that uses $n = \tilde{O}(d^2/\varepsilon^2)$ samples and learns the underlying Gaussian to error $\varepsilon$ in total variation distance. Our algorithm is also fast: its runtime is dominated by the cost of computing the empirical covariance matrix. Both our sample and time complexity are optimal in terms of $d$ and $\varepsilon$ even without truncation: in this regard, we can learn a Gaussian under halfspace truncation for free.   The key ingredient behind our result is a novel reinterpretation of the low-degree moments of the tru
    
[^4]: Ribbon：可扩展的近似与稳健的不确定性量化

    Ribbon: Scalable Approximation and Robust Uncertainty Quantification

    [https://arxiv.org/abs/2606.27269](https://arxiv.org/abs/2606.27269)

    Ribbon通过影响函数线性化替代重复重拟合，实现了对贝叶斯自助不确定性的高效近似，并引入可调集中参数实现校准。

    

    arXiv:2606.27269v1 公告类型：交叉 摘要：对于复杂、高维或错误指定的模型，可靠地量化预测不确定性是困难的。完全贝叶斯方法和自助重采样方法都能提供原则性的不确定性估计，但对于现代机器学习模型而言，这些方法往往过于昂贵，因为它们需要后验采样或重复模型重拟合。我们提出了Ribbon，一种对狄利克雷加权自助不确定性的可扩展近似方法。Ribbon用围绕单个拟合模型的影响函数线性化替代了重复重拟合，保留了贝叶斯自助的一阶数据加权结构，同时仅需事后线性代数运算。Ribbon近似于贝叶斯自助或加权似然自助的重拟合目标。通过一个通用的集中参数，Ribbon提供了一个校准的狄利克雷加权族，其不确定性尺度可在验证数据上调整。我们证明了Ribbon在渐近意义上是等效的。

    arXiv:2606.27269v1 Announce Type: cross  Abstract: Reliably quantifying predictive uncertainty is difficult for complex, high-dimensional, or misspecified models. Both fully Bayesian and bootstrap resampling methods provide principled uncertainty estimates but are often too expensive for modern machine-learning models because they require posterior sampling or repeated model refitting. We introduce Ribbon, a scalable approximation to Dirichlet-reweighted bootstrap uncertainty. Ribbon replaces repeated refitting with an influence-function linearization around a single fitted model, preserving the first-order data-reweighting structure of the Bayesian bootstrap while requiring only post-hoc linear algebra. Ribbon approximates the Bayesian-bootstrap or weighted-likelihood-bootstrap refitting target. With a general concentration parameter, Ribbon gives a calibrated Dirichlet-reweighting family whose uncertainty scale can be tuned on validation data. We show that Ribbon is asymptotically eq
    
[^5]: 更新的几何：词汇规模下的费舍尔对齐

    The Geometry of Updates: Fisher Alignment at Vocabulary Scale

    [https://arxiv.org/abs/2606.27242](https://arxiv.org/abs/2606.27242)

    本文提出FisherSketch方法，通过将头费舍尔对齐等价为联合激活-误差空间中核均值嵌入的余弦值，实现了在词汇规模下高效、可识别的无训练源选择，解决了传统表示度量不可识别而经典几何度量计算成本过高的问题。

    

    arXiv:2606.27242v1 公告类型：交叉 摘要：在具有共享词汇表的大型语言模型家族中进行无训练源选择，出现在SMILES、蛋白质和基因组序列等科学字符串领域，其中候选语料库共享分词器但预测目标不同。这导致了一种“激活-暗区”现象：在缺乏关于标签条件误差几何假设的情况下，表示相似性度量可能无效，而经典的更新几何度量在词汇规模下计算成本过高。我们证明，在共享输出头的设定下，表示度量（如CKA）对于迁移是不可识别的；模型可以共享完全相同的表示，却具有正交的头更新。关键恒等式是：头费舍尔对齐恰好是联合激活-误差空间中核均值嵌入之间的余弦值，从而揭示了激活、误差和耦合因子，而无需实例化费舍尔矩阵。FisherSketch通过直接估计这一余弦值来工作。

    arXiv:2606.27242v1 Announce Type: cross  Abstract: Training-free source selection for LLM families with shared vocabularies arises in scientific string domains such as SMILES, protein, and genomic sequences, where candidate corpora share a tokenizer but differ in prediction targets. This creates an activation-dark regime: representation-similarity metrics can be uninformative without assumptions about label-conditioned error geometry, while classical update-geometry metrics are computationally prohibitive at vocabulary scale. We show that, in a shared-output head setting, representation metrics (e.g., CKA) are non-identifiable for transfer; models can share identical representations yet have orthogonal head updates. The key identity is that head Fisher alignment is exactly a cosine between kernel mean embeddings in the joint activation-error space, exposing activation, error, and coupling factors rather than requiring a materialized Fisher matrix. FisherSketch estimates this cosine dir
    
[^6]: 超越全局分歧：贝叶斯推理中的局部质量视角

    Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference

    [https://arxiv.org/abs/2606.27090](https://arxiv.org/abs/2606.27090)

    本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。

    

    摘要：arXiv:2606.27090v1 公告类型：交叉 摘要：全局目标函数，如KL散度和ELBO，在贝叶斯推理中被广泛用于度量分布差异。本文研究这些目标函数未能直接捕捉的局部质量行为。我们引入并使用了两种数学工具：（1）质量指数，用于记录局部质量的多项式和对数衰减尺度；（2）正则化扩展KL（RE-KL），一种在存在奇异成分时可公式化的局部化散度。质量指数有助于刻画贝叶斯更新如何改变局部质量：（1）幂对数似然因子显式地改变它；（2）参数依赖的支持域或其平滑软化，可能通过参数值附近剩余的质量量来改变局部尺度。利用局部RE-KL，我们证明了在两种KL方向下比较局部小球质量的绝对、相对和方向性不等式。这些结果共同为局部质量行为提供了理论依据。

    arXiv:2606.27090v1 Announce Type: cross  Abstract: Global objectives, such as KL divergence and ELBO, are widely used in Bayesian inference for measuring distributional discrepancy. This paper studies their local-mass behaviour that is not directly captured by such objectives. We introduce and use two mathematical tools: (1) Mass Index for recording the polynomial and logarithmic decay scales of local mass, and (2) regularised extended KL (RE-KL), a set-localised divergence that can be formulated in the presence of singular components. Mass Indices help characterise how Bayesian updating changes local mass: (1) power-log likelihood factors shift it explicitly, and (2) parameter-dependent supports, or their smooth softenings, may change the local scale through the amount of mass that remains near the parameter value. Using local RE-KL, we prove absolute, relative, and directional inequalities for comparing local small-ball masses under the two KL directions. Together, these results prov
    
[^7]: 不确定性量化的决策对齐评估

    Decision-Aligned Evaluation of Uncertainty Quantification

    [https://arxiv.org/abs/2606.26990](https://arxiv.org/abs/2606.26990)

    本文提出决策对齐标准，发现传统不确定性量化指标常与下游决策效用不一致，并设计先验加权效用指标以实现与决策效用的对齐，从而修正了现有评估协议的缺陷。

    

    arXiv:2606.26990v1 公告类型：交叉 摘要：机器学习中的不确定性估计通常使用通用指标（如负对数似然和期望校准误差）进行评估，然而在这些指标上表现良好并不一定意味着在下游决策中具有高实用性。我们引入了“决策对齐”这一标准，它揭示了哪些评估指标能够有意义地与下游效用对齐。应用这一框架，我们表明许多广泛使用的不确定性指标要么与常见决策问题不一致，要么编码了关于下游任务的病态先验信念。然后，我们提出了先验加权效用指标，这是一类特殊的适当评分规则，能够提供决策对齐的不确定性评估。在基准实验和实际案例研究中，我们的指标始终与实现的决策效用保持一致，而传统指标则不然。我们的结果揭示了当前不确定性量化评估协议中的缺陷，并提供了一种新的评估范式。

    arXiv:2606.26990v1 Announce Type: cross  Abstract: Uncertainty estimates in machine learning are typically evaluated using generic metrics such as the negative log-likelihood and expected calibration error, yet good performance on such metrics does not necessarily imply high utility in downstream decisions. We introduce decision-alignment, a criterion that reveals which evaluation metrics meaningfully align with downstream utilities. Applying this framework, we show that many widely used uncertainty metrics are either misaligned with common decision problems or encode pathological prior beliefs about the downstream task. We then propose prior-weighted utility metrics, a special class of proper scoring rules that provides decision-aligned uncertainty evaluation. Across benchmark experiments and real-world case studies, our metrics consistently align with realized decision utility, while conventional metrics do not. Our results surface flaws in the current UQ evaluation protocol and offe
    
[^8]: 面向超额均方误差的自适应经验贝叶斯估计

    XMSE-Aware Adaptive Empirical Bayes Estimation

    [https://arxiv.org/abs/2606.26975](https://arxiv.org/abs/2606.26975)

    本文通过将超额均方误差（XMSE）分析从诊断工具转化为设计原则，提出了一种在最大似然和经验贝叶斯之间自适应插值的混合估计器，并证明了其在二阶意义下不劣于两者。

    

    经验贝叶斯（EB）估计器能够在一阶渐近风险上与最大似然（ML）估计相匹配，但在二阶行为上存在显著差异：最新的超额均方误差（XMSE）分析表明，当核函数与真实参数对齐不佳时，基于核的经验贝叶斯估计可能比最大似然估计更差。本文将这一诊断转化为设计原则。我们提出了一种面向XMSE的混合估计器，它在ML估计和EB收缩之间进行插值。其固定权重的XMSE是一个标量二次形式，从而得到一个闭式的理想混合权重，该权重在XMSE尺度上不劣于ML估计和基础EB估计。基于有限样本XMSE近似的插件实现被证明是一致的，并且对于内部理想权重具有二阶遗憾率。我们进一步将遗憾界迁移到所选权重下的固定权重风险曲线、一个阈值边界规则以及相关扩展。

    arXiv:2606.26975v1 Announce Type: cross  Abstract: Empirical Bayes (EB) estimators can match the first-order asymptotic risk of maximum likelihood (ML) while behaving very differently at second order: recent excess mean squared error (XMSE) analysis shows that kernel-based EB estimation may be worse than ML when the kernel is poorly aligned with the true parameter. This paper turns that diagnostic into a design principle. We propose an XMSE-aware mixed estimator that interpolates between ML and EB shrinkage. Its fixed-weight XMSE is a scalar quadratic, yielding a closed-form oracle mixing weight that is no worse than both ML and the base EB estimator at the XMSE scale. A plug-in implementation based on finite-sample XMSE approximations is proved consistent, with a second-order oracle regret rate for an interior oracle weight. We further establish a transfer of the regret bound to the fixed-weight risk curve evaluated at the selected weight, a thresholded boundary rule, and extensions t
    
[^9]: 参数化先知不等式问题的渐近最优学习

    Asymptotically Optimal Learning for Parametric Prophet Inequalities

    [https://arxiv.org/abs/2606.26893](https://arxiv.org/abs/2606.26893)

    针对参数未知的指数型参数族先知不等式问题，提出了一种仅靠在线观测即可达到最优渐近竞争比的置信度动态规划策略，无需离线样本。

    

    我们研究了先知不等式中的学习问题，其中收益独立同分布，来自一个参数未知的指数型参数族，该族包括指数分布、帕累托分布和有界支撑幂族分布。我们首先刻画了该族的最优全信息渐近竞争比。在无界支撑情形下，该极限值为 ${\left({\theta}/({\theta-c_+})\right)^{c_+/\theta}}/ {\Gamma(1-c_+/\theta)}$；而在有界支撑情形下，极限值为 $1$。随后，我们提出了一种基于置信度的动态规划在线学习策略。通过利用显式的参数结构，该策略仅使用在线观测数据即可达到相同的最优渐近竞争比，无需外部离线样本。我们还针对典型例子推导了分布特定的收敛速率。最后，在合成实例上的数值实验展示了我们算法的性能。

    arXiv:2606.26893v1 Announce Type: new  Abstract: We study learning in prophet inequalities with i.i.d. rewards drawn from an exponential-type parametric family with an unknown parameter $\theta$, a class that includes exponential, Pareto, and bounded-support power-family distributions. We first characterize the optimal full-information asymptotic competitive ratio for this family. In the unbounded-support case, the limit is $ {\left({\theta}/({\theta-c_+})\right)^{c_+/\theta}}/ {\Gamma(1-c_+/\theta)},$ while in the bounded-support case, the limit is $1$. We then propose a confidence-based dynamic-programming policy for online learning. By exploiting the explicit parametric structure, the policy achieves the same optimal asymptotic competitive ratio using only online observations, without external offline samples. We further derive distribution-specific convergence rates for canonical examples. Finally, numerical experiments on synthetic instances illustrate the performance of our algor
    
[^10]: 数据驱动的久期管理——基于机器学习的期限结构预测

    Data-Driven Duration Management -- Term Structure Forecasting Using Machine Learning

    [https://arxiv.org/abs/2606.26815](https://arxiv.org/abs/2606.26815)

    本文提出将神经网络模型（尤其是受经典模型启发的架构）与宏观经济变量结合，用于美国和欧洲债券期限结构预测，并证明其在统计精度和交易策略经济价值上均优于传统模型。

    

    本文比较了利用传统计量经济学和机器学习方法预测美国和欧洲零息政府债券期限结构的不同方式。我们在美国国债市场和欧洲央行发行的债券上，对比了经典模型（如动态尼尔森-西格尔模型和主成分分析）与不同的神经网络架构（包括受经典模型启发的架构）。为提升预测性能，我们引入了宏观经济变量。两个市场的结果被分别分析并比较。为此，我们提出一个稳健的模型评估框架，该框架将统计精度指标（如均方根误差、平均绝对误差和方向准确率）与定量债券交易策略的经济相关性相结合。结果表明，神经网络在预测准确性和投资组合表现上始终优于传统模型。

    arXiv:2606.26815v1 Announce Type: new  Abstract: This paper compares different methods for forecasting the term structure of U.S. and European zero-coupon government bonds using both traditional econometric and Machine Learning (ML) approaches. We compare classical models (e.g., Dynamic Nelson-Siegel (DNS) and Principal Component Analysis (PCA)) with different Neural Network (NN) architectures, including those inspired by the classical models, on the U.S. Treasury market and bonds issued by the European Central Bank (ECB). To enhance predictive performance, macroeconomic variables are incorporated. The findings for both markets are separately analyzed and compared. To this end, we propose a robust model evaluation framework combining statistical accuracy metrics - such as RMSE, MAE, and directional accuracy - with the economic relevance of a quantitative bond trading strategy. Results show that NNs consistently outperform traditional models in both forecasting accuracy and portfolio pe
    
[^11]: 逃离迭代参数空间噪声：基于超网络的差分隐私学习

    Escaping Iterative Parameter-Space Noise: Differentially Private Learning with a Hypernetwork

    [https://arxiv.org/abs/2606.26772](https://arxiv.org/abs/2606.26772)

    提出了一种基于超网络的新框架，通过仅一次向低维数据集表示注入隐私噪声，避免了迭代参数空间噪声，从而显著降低差分隐私学习中噪声的不利影响。

    

    神经网络差分隐私（DP）训练常常受到基于梯度的方法（如DP-SGD）所需大量噪声的阻碍，这些方法在整个训练过程中反复向参数空间注入高维噪声。本文提出了一种新的DP学习框架，避免了参数空间中的迭代优化。我们不使用私有梯度更新目标模型，而是采用在公共数据集上训练的超网络，将私有数据集映射到目标模型的参数。具体来说，每个示例被嵌入到一个低维表示中，嵌入被聚合和扰动以获得DP数据集嵌入，超网络从该噪声嵌入生成目标模型参数。由于隐私噪声仅注入一次到低维数据集表示中，我们的方法可以显著降低噪声的不利影响。我们从理论上证明了该框架的隐私保证，并实验表明其在多个基准数据集上优于标准DP-SGD方法。

    arXiv:2606.26772v1 Announce Type: new  Abstract: Differentially private (DP) training of neural networks is often hindered by the large amount of noise required by gradient-based methods such as DP-SGD, which repeatedly inject high-dimensional noise in parameter space throughout training. In this paper, we propose a new framework for DP learning that avoids iterative optimization in parameter space. Instead of updating the target model using privatized gradients, we employ a hypernetwork trained on public datasets to map a private dataset to the parameters of the target model. Specifically, each example is embedded into a low-dimensional representation, the embeddings are aggregated and perturbed to obtain a DP dataset embedding, and the hypernetwork generates the target model parameters from this noisy embedding. Because privacy noise is injected only once into a low-dimensional dataset representation, our approach can significantly reduce the adverse effect of noise. We theoretically
    
[^12]: 基于奈斯特龙近似的可扩展算子学习及其在去噪中的应用

    Scalable Operator Learning via Nystr\"om Approximation With Denoising Applications

    [https://arxiv.org/abs/2606.26652](https://arxiv.org/abs/2606.26652)

    本文提出一种基于奈斯特龙子采样的高效算子学习算法，在广泛源条件下达到极小化最优收敛率，并成功应用于通用函数去噪问题。

    

    本文研究了向量值再生核希尔伯特空间中向量值回归的奈斯特龙子采样方法。标准核方法通常因构建和求逆大型核矩阵而面临高昂的计算成本，这限制了其在大规模数据集上的可扩展性。为克服这一瓶颈，我们提出了一种基于奈斯特龙子采样的高效算子学习算法，该算法能够处理函数型输出。在由索引函数刻画的一般源条件下——该条件超越了经典的赫尔德型和算子单调框架——我们为所提出的估计器建立了极小化最优收敛速率。作为所提框架的一个应用，我们考虑了函数去噪问题。与通常针对特定信号表示或噪声模型量身定制的经典去噪方法不同，我们的方法将去噪问题置于一个通用的算子学习框架中。

    arXiv:2606.26652v1 Announce Type: cross  Abstract: In this paper, we study Nystr\"om subsampling for vector-valued regression in vector-valued reproducing kernel Hilbert spaces. Standard kernel methods often suffer from prohibitive computational costs due to the construction and inversion of large kernel matrices, which limits their scalability to large datasets. To overcome this bottleneck, we propose an efficient operator learning algorithm based on Nystr\"om subsampling that accommodates functional outputs. Under general source conditions characterized by index functions-extending beyond the classical H\"older-type and operator-monotone frameworks-we establish minimax-optimal convergence rates for the proposed estimator. As an application of the proposed framework, we consider function denoising problems. Unlike classical denoising methods, which are typically tailored to specific signal representations or noise models, our approach formulates denoising within a general operator lea
    
[^13]: λ-多项式斯坦差异：可扩展的近似信噪比优化多项式斯坦差异

    $\lambda$-PSD: Scalable Approximate SNR-Optimised Polynomial Stein Discrepancies

    [https://arxiv.org/abs/2606.26621](https://arxiv.org/abs/2606.26621)

    本文揭示了多项式斯坦差异中SNR²随阶数指数衰减的问题，并提出一种基于瑞利商最大化的可扩展近似加权方案λ-PSD，有效避免了这一失效模式。

    

    多项式斯坦差异（PSD）为核斯坦方法提供了一种可扩展的替代方案，用于衡量样本质量和拟合优度检验，但其统计特性仍不为人所熟知。我们证明，增加多项式阶数主要放大信号，而未能充分控制方差，而非直接优化信噪比（SNR）。在适当假设下，这可能导致一种失效模式，即SNR²会随多项式阶数呈指数衰减。受此观察启发，我们将斯坦差异构造重新表述为一个明确的SNR²最大化问题，从而在斯坦特征上得到瑞利商。这一视角催生了λ-PSD，一种在低维子空间中定义的可扩展近似协方差感知加权方案。在高斯设置下，我们证明λ-PSD避免了指数级的SNR²衰减。

    arXiv:2606.26621v1 Announce Type: cross  Abstract: Polynomial Stein discrepancies (PSD) provide a scalable alternative to kernel Stein methods for measuring sample quality and goodness-of-fit testing, but their statistical properties remain poorly understood. We show that increasing polynomial degree primarily amplifies signal without adequately controlling variance, rather than directly optimising the signal-to-noise ratio (SNR). Under suitable assumptions, this might lead to a failure mode in which the $\text{SNR}^2$ can provably decay exponentially with polynomial degree. Motivated by this observation, we reformulate Stein discrepancy construction as an explicit $\text{SNR}^2$ maximisation problem, yielding a Rayleigh quotient over Stein features. This perspective motivates $\lambda$-PSD, an approximate scalable covariance-aware reweighting scheme defined in a low-dimensional subspace. Under Gaussian settings, we show that $\lambda$-PSD avoids the exponential $\text{SNR}^2$ collapse
    
[^14]: 基于严格恰当评分规则学习概率滤波器

    Learning Probabilistic Filters with Strictly Proper Scoring Rules

    [https://arxiv.org/abs/2606.26497](https://arxiv.org/abs/2606.26497)

    本文提出PSEF方法，利用严格恰当评分规则训练基于Transformer的置换不变映射，仅通过合成数据实现贝叶斯滤波分布的逼近。

    

    针对部分观测且含噪声的动态系统的贝叶斯滤波，旨在在线推断系统状态随观测演变的条件分布。该贝叶斯滤波分布是不确定性量化的自然对象，但很少能作为监督学习目标直接获得。然而，我们通常可以利用预测模型生成合成系统轨迹及合成观测数据。本文提出了恰当评分集成滤波器（PSEF），这是一种基于训练分析映射的集成数据同化方法，仅通过合成状态-观测轨迹来逼近滤波分布。分析步骤被表示为一种基于置换不变性、Transformer架构的映射，它接收预测集成和观测作为输入，生成分析集成。训练基于严格恰当的评分规则——其中使用了能量评分。

    arXiv:2606.26497v1 Announce Type: new  Abstract: Bayesian filtering of partially and noisily observed dynamical systems seeks to infer the evolving conditional distribution of the state of a dynamical system, given observations, in an online fashion. This Bayesian filtering distribution is the natural object for uncertainty quantification, but it is rarely available as a supervised learning target. However, one can often use the forecast model to generate synthetic system trajectories, along with synthetic observations. We introduce the proper scoring ensemble filter (PSEF), an ensemble data assimilation method based on training an analysis map to approximate the filtering distribution using only synthetic state--observation trajectories. The analysis step is represented as a permutation-invariant, transformer-based map that takes as input a forecast ensemble and observations, producing an analysis ensemble. Training is based on strictly proper scoring rules -- with the energy score us
    
[^15]: 在线测试时自适应的概率框架

    A probabilistic framework for online test-time adaptation

    [https://arxiv.org/abs/2606.26457](https://arxiv.org/abs/2606.26457)

    提出了一种基于状态空间模型的概率框架，用于在线测试时自适应，以应对训练与测试分布之间的偏移。

    

    本文提出了一种用于在线测试时自适应问题的概率框架。在该问题中，模型基于标注数据进行训练，但必须在测试时适应未标注数据，且假设训练分布与测试分布可能存在差异，即可能存在分布偏移。该框架基于状态空间建模架构，可对参数学习、参数时间演化、先验调优以及预测进行系统刻画。

    arXiv:2606.26457v1 Announce Type: cross  Abstract: This paper presents a probabilistic framework for online test-time adaptation problems. In them, a model is trained on labeled data but must adapt to unlabeled data at test time under the assumption that training and test distributions potentially differ, that is, there might have been a distributional shift. The framework is based on a state-space modelling architecture from which parameter learning, parameter time evolution, prior tuning, and prediction can be characterized.
    
[^16]: 面向区间数据的可解释性异常检测

    Explainable Outlier Detection for Interval-valued Data

    [https://arxiv.org/abs/2606.26307](https://arxiv.org/abs/2606.26307)

    提出了一种基于沙普利值的区间数据异常检测可解释性方法，通过闭式表达式高效分解变量贡献，实现中心、范围及交叉项的精细分析。

    

    可解释性正日益被视为异常检测的关键方面。然而，对于区间数据等复杂数据结构，这一领域仍基本未被探索。基于区间最小协方差行列式估计器的异常检测框架，我们提出了一种新方法，利用沙普利值的概念来解释区间观测值的异常程度。我们推导了鲁棒区间马氏距离平方的沙普利值的闭式表达式，从而能够高效计算变量贡献。这一公式实现了对异常值的精细解释，提供了对区间观测值中心、范围和交叉项贡献的详细分解。此外，沙普利值与逐单元异常的概念紧密相关，因为它有助于识别在多元层面可能不明显的特定变量异常。

    arXiv:2606.26307v1 Announce Type: cross  Abstract: Explainability is increasingly recognized as a key aspect of outlier detection. However, for complex data structures such as interval-valued data, it remains largely unexplored. Building on an outlier detection framework based on the Interval Minimum Covariance Determinant estimator, we propose a novel approach to explain the outlyingness of interval-valued observations using the concept of the Shapley value. We derive a closed-form expression for the Shapley value of the squared robust Interval-Mahalanobis distance, enabling efficient computation of variable contributions. This formulation allows for a fine-grained interpretation of outliers, providing a detailed decomposition into contributions from centers, ranges, and cross-terms of the interval-valued observations. Moreover, the Shapley value is closely connected to the concept of cellwise outliers, as it can help identify variable-specific outliers that may not be evident at mult
    
[^17]: 输入维度在对抗样本涌现与定向控制中的作用

    The Role of Input Dimensionality in the Emergence and Targeted Control of Adversarial Examples

    [https://arxiv.org/abs/2606.26207](https://arxiv.org/abs/2606.26207)

    通过实证研究揭示输入维度增加会使对抗样本更易构造，并发现真实图像类别的强经验局部化特性超出传统高维几何理论假设。

    

    arXiv:2606.26207v1 公告类型：交叉 摘要：多项理论研究试图通过高维几何性质解释深度神经网络的对抗脆弱性。然而，这些研究所依据的假设很少得到实证检验，系统性证据仍然有限。在本工作中，我们系统研究了输入维度在对抗样本涌现与定向控制中的作用。我们首先分析了基于测度集中现象的现有理论框架的适用范围与局限性，结果表明真实图像类别表现出强经验局部化特征，其程度超出此类理论通常假设的范围。随后，我们跨多个层次图像数据集（涵盖广泛输入维度范围与多样化神经架构）进行了广泛实证评估。结果一致表明，随着维度增加，对抗样本更易于构造。我们还研究了……

    arXiv:2606.26207v1 Announce Type: cross  Abstract: Several theoretical works have tried to explain the adversarial vulnerability of deep neural networks through properties of high-dimensional geometry. However, the assumptions underlying these works are rarely examined empirically, and systematic evidence remains limited. In this work, we present a systematic study of the role of input dimensionality in both the emergence and the targeted control of adversarial examples. We first analyse the scope and limitations of existing theoretical frameworks based on concentration of measure, showing that real image classes exhibit strong empirical localization, beyond what such theories typically assume. We then conduct an extensive empirical evaluation across hierarchical image datasets spanning a wide range of input dimensionalities and diverse neural architectures. Our results consistently show that adversarial examples become easier to construct as dimensionality increases. We also investiga
    
[^18]: 算法公平性的统计与结构方法

    Statistical and Structural Approaches to Algorithmic Fairness

    [https://arxiv.org/abs/2606.26200](https://arxiv.org/abs/2606.26200)

    本论文指出现代算法公平性方法的两大根本缺陷——依赖确定性点估计审计和将个体视为孤立实体，并提出改进方案。

    

    现代机器学习系统已超越其作为孤立预测构件的起源，演变为积极调节人类机遇的复杂社会技术架构。随着算法日益决定经济与社会机会的获取，人们普遍认识到这些系统深度嵌入了其所在环境的结构性不平等与偏见。算法公平性领域应运而生，以应对一个日益明确的认知：为预测准确性优化的模型可能系统性地边缘化弱势群体。然而，早期的缓解策略建立在脆弱的简化假设之上，这限制了其在复杂社会技术环境中的有效性。本论文识别并解决了当代公平性范式的两个根本局限性：依赖确定性点估计进行审计，以及将个体视为孤立实体。

    arXiv:2606.26200v1 Announce Type: cross  Abstract: Modern machine learning systems have outgrown their origins as isolated predictive constructs, evolving into complex socio-technical architectures that actively mediate human opportunity. As algorithms increasingly determine access to economic and social opportunities, it has become widely recognized that these systems are deeply embedded with the structural inequalities and prejudices of their environments. The field of algorithmic fairness emerged in response to the growing recognition that models optimized for predictive accuracy can systematically disadvantage marginalized groups. Early mitigation strategies, however, rested on fragile simplifications that limited their effectiveness in complex socio-technical environments. This thesis identifies and addresses two fundamental limitations of contemporary fairness paradigms: the reliance on deterministic point estimates for auditing and the treatment of individuals as isolated entiti
    
[^19]: 数据科学中的表示成本：深度神经网络的基础与拟巴拿赫空间

    Representation Costs in Data Science: Foundations and the Quasi-Banach Spaces of Deep Neural Networks

    [https://arxiv.org/abs/2606.14954](https://arxiv.org/abs/2606.14954)

    本文提出了一个统一框架，通过参数空间正则化器分析数据科学中的表示成本，揭示了参数化方法与其原生函数空间之间的联系，并将核方法、小波和神经网络等经典方法统一为特例。

    

    我们开发了一个通用框架，用于通过参数空间正则化器分析参数化数据拟合方法的表示成本。从这一抽象视角出发，我们定义了任意参数化模型的表示成本，并揭示了它们所诱导的（原生）函数空间。这统一了近期关于数据拟合方法的函数空间视角。我们还证明，在该抽象设定下许多自然结论成立，包括参数化方法在其原生空间上的表示定理。该框架还严格地将参数化方法与其在充分过参数化下的等价非参数描述联系起来。经典方法及其原生空间，如核方法/再生核希尔伯特空间、小波/贝索夫空间以及浅层神经网络/变分空间，均作为我们抽象框架的特例出现。将表示成本研究“公理化”是一个副产品。

    arXiv:2606.14954v3 Announce Type: replace-cross  Abstract: We develop a general framework for analyzing representation costs of parametric data-fitting methods through their parameter-space regularizers. From this abstract perspective, we define representation costs for arbitrary parametric models and reveal their induced (native) function spaces. This unifies recent function-space views of data-fitting methods. We also prove that many natural results hold in this abstract setting, including representer theorems for parametric methods on their native spaces. The framework also rigorously connects parametric methods with their equivalent nonparametric descriptions under sufficient overparameterization. Classical methods and their native spaces, such as kernel methods / reproducing kernel Hilbert spaces, wavelets / Besov spaces, and shallow neural networks / variation spaces emerge as special cases of our abstract framework. A byproduct of "axiomatizing" the study of representation costs
    
[^20]: 基于深度强化学习的加密货币市场动态多对交易策略

    Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning

    [https://arxiv.org/abs/2606.04574](https://arxiv.org/abs/2606.04574)

    本研究通过分层配对选择方法和专有执行模型，结合深度强化学习，显著提升了加密货币市场中配对交易的稳健性与收益表现。

    

    本研究旨在探讨深度强化学习作为专门执行覆盖层，能否增强高波动性加密货币市场中的配对交易。尽管经典配对交易策略在传统股票市场中已证明成功，但在高方差环境中常表现出僵化性，并面临严重的发散风险。为应对这一需求，本研究引入了新颖概念。为构建稳健系统，我们开发了分层的“筛选-排序”配对选择方法，以及专有的“固定风险、自适应均值”执行模型。该系统采用带有长短期记忆层的近端策略优化智能体，在严格的确定性风险管理边界内控制执行决策。基于币安USD-M期货市场1小时间隔数据的评估显示，优化后的强化学习策略在样本外测试中实现了...

    arXiv:2606.04574v2 Announce Type: replace  Abstract: This study aims to determine whether the application of Deep Reinforcement Learning (DRL) as a specialized execution overlay can enhance pair trading in highly volatile cryptocurrency markets. Although classical implementations of the strategy have proven successful in traditional equities, they frequently exhibit rigidity and suffer from severe divergence risks when applied to high-variance environments. To address this need, this research introduces novel concepts. To construct a robust system, we developed a hierarchical "Filter-then-Rank" pair selection methodology and a proprietary "Fixed Risk, Adaptive Mean" execution model. The system employs a Proximal Policy Optimization (PPO) agent with a Long Short-Term Memory (LSTM) layer to govern execution decisions within strict deterministic risk management boundaries. Evaluated on 1-hour interval data from the Binance USD-M Futures market, the optimized RL policy achieved an out-of-s
    
[^21]: FoReco 与 FoRecoML：R 语言中用于预测协调的统一工具箱

    FoReco and FoRecoML: A Unified Toolbox for Forecast Reconciliation in R

    [https://arxiv.org/abs/2604.27696](https://arxiv.org/abs/2604.27696)

    FoReco 与 FoRecoML 是 R 语言中首个统一涵盖截面、时间及跨时间预测协调的综合性工具箱，兼顾易用性与灵活性，填补了该领域软件空白。

    

    预测协调已成为提高线性约束多时间序列（如层次序列和分组序列）预测准确性与一致性的关键方法。然而，目前尚缺乏能同时涵盖截面、时间及跨时间协调的综合软件。R 语言包 FoReco 和 FoRecoML 通过提供统一框架填补了这一空白。这两个包分别实现了经典的和基于回归的线性协调方法，以及基于机器学习的非线性方法，适用于截面、时间及跨时间场景。为兼顾易用性与灵活性，这些包提供了合理的默认选项，使新用户能以最小工作量应用协调方法，同时允许专家用户通过自定义设置探索前沿扩展功能。凭借这一双重设计，FoReco 和 FoRecoML 为预测协调领域提供了强大的工具支持。

    arXiv:2604.27696v2 Announce Type: replace-cross  Abstract: Forecast reconciliation has become key to improving the accuracy and coherence of forecasts for linearly constrained multiple time series, such as hierarchical and grouped series. Yet, comprehensive software that jointly covers cross-sectional, temporal, and cross-temporal reconciliation has so far been lacking. The R packages FoReco and FoRecoML address this gap by offering a comprehensive and unified framework. The packages respectively implement classical and regression-based linear reconciliation approaches, and non-linear approaches based on machine learning for cross-sectional, temporal and cross-temporal frameworks. Designed for accessibility and flexibility, these packages provide sensible default options that allow new users to apply reconciliation methods with minimal effort, while still giving expert users full control to explore state-of-the-art extensions through customized settings. With this dual focus, FoReco an
    
[^22]: 对比学习、重要性采样和桥接采样在基于能量的模型中的统一视角

    A unifying view of contrastive learning, importance sampling, and bridge sampling for energy-based models

    [https://arxiv.org/abs/2604.08116](https://arxiv.org/abs/2604.08116)

    本文提出了一个统一框架，将噪声对比估计、反向逻辑回归、多重重要性采样和桥接采样在基于能量的模型中联系起来，揭示了它们之间的等价关系，并促进了更高效估计量的开发。

    

    在过去几十年中，基于能量的模型（EBMs）已成为一类重要的概率模型，其似然函数的一部分是难处理的，因此无法显式计算。因此，对于传统推断方法而言，EBMs中的参数估计具有挑战性。在这项工作中，我们提供了一个统一框架，将噪声对比估计（NCE）、反向逻辑回归（RLR）、多重重要性采样（MIS）和桥接采样在EBMs的背景下联系起来。我们进一步证明，这些方法在特定条件下是等价的。这种统一视角澄清了现有方法之间的关系，并使得新估计量的开发成为可能，有望提高统计和计算效率。此外，本研究有助于阐明NCE在灵活性和鲁棒性方面的成功，同时识别出其性能可能下降的场景。

    arXiv:2604.08116v2 Announce Type: replace-cross  Abstract: In the last decades, energy-based models (EBMs) have become an important class of probabilistic models in which a component of the likelihood is intractable and therefore cannot be evaluated explicitly. Consequently, parameter estimation in EBMs is challenging for conventional inference methods. In this work, we provide a unified framework that connects noise contrastive estimation (NCE), reverse logistic regression (RLR), multiple importance sampling (MIS), and bridge sampling within the context of EBMs. We further show that these methods are equivalent under specific conditions. This unified perspective clarifies relationships among existing methods and enables the development of new estimators, with the potential to improve statistical and computational efficiency. Furthermore, this study helps elucidate the success of NCE in terms of its flexibility and robustness, while also identifying scenarios in which its performance c
    
[^23]: 基于希尔伯特空间嵌入的量子最大似然预测

    Quantum Maximum Likelihood Prediction via Hilbert Space Embeddings

    [https://arxiv.org/abs/2602.18364](https://arxiv.org/abs/2602.18364)

    本文通过将经验概率分布嵌入量子态并最小化量子相对熵，提出了一种量子最大似然预测方法，并为其在经典和量子大语言模型中的统一应用提供了非渐近性能保证。

    

    arXiv:2602.18364v3 公告类型: 替换-交叉 摘要：最大似然预测（MLP）是现代大型语言模型的核心任务。在此，我们首次针对由独立同分布样本构成的简化数据模型，研究该任务的量子版本。量子最大似然预测器（QMLP）通过将经验概率分布嵌入到量子态中，并在给定状态类上最小化量子相对熵来获得。我们推导了QMLP在迹范数和量子相对熵方面的非渐近性能保证，包括收敛速率和浓度不等式。我们的方法为在经典和量子大语言模型中处理MLP提供了一个统一框架。我们还考虑了量子信息投影的相关问题，并将著名的量子毕达哥拉斯定理推广到并非由自伴类生成的混合族。

    arXiv:2602.18364v3 Announce Type: replace-cross  Abstract: Maximum likelihood prediction (MLP) is a core task at the heart of modern large language models. Here, we study a quantum version of this task for a simplified data model consisting of independent and identically distributed samples, as a first step. The quantum maximum likelihood predictor (QMLP) is obtained by embedding of empirical probability distributions into quantum states and performing a minimization of quantum relative entropy over a given class of states. We derive non-asymptotic performance guarantees for QMLP in terms of convergence rates and concentration inequalities, both in trace norm and quantum relative entropy. Our approach provides a unified framework to handle MLP within both classical and quantum LLMs. We also consider the related problem of quantum information projection and generalize the well known quantum Pythagorean theorem to mixture families which are not necessarily generated by a self-adjoint cla
    
[^24]: 基于稀疏卫星时间序列和天气协变量的概率性NDVI预测

    Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates

    [https://arxiv.org/abs/2602.17683](https://arxiv.org/abs/2602.17683)

    该论文提出了一种概率性预测框架，通过分离历史数据编码与未来协变量、引入时间距离加权损失函数，解决了稀疏不规则卫星观测下的田块级NDVI短期预测挑战。

    

    arXiv:2602.17683v3 公告类型：替换 摘要：植被动态的短期预测是实现精准农业中数据驱动决策支持的关键推动因素。然而，基于卫星观测的归一化植被指数（NDVI）预测仍具挑战性，原因在于云掩膜导致的稀疏和不规则采样，以及作物生长的异质性气候条件。在这项工作中，我们提出了一种概率性预测框架，用于在稀疏、不规则的晴空采集条件下进行田块级NDVI预测。该架构将历史NDVI和气象观测的编码与未来外生协变量分离，融合两种表示以进行多步分位数预测。为解决不规则重访模式和与预测时间跨度相关的不确定性，我们引入了一种时间距离加权分位数损失函数，使训练目标与有效预测时间跨度对齐。此外，我们纳入了累积...

    arXiv:2602.17683v3 Announce Type: replace  Abstract: Short-term forecasting of vegetation dynamics is a key enabler for data-driven decision support in precision agriculture. Normalized Difference Vegetation Index (NDVI) forecasting from satellite observations, however, remains challenging due to sparse and irregular sampling caused by cloud masking, as well as the heterogeneous climatic conditions under which crops evolve. In this work, we propose a probabilistic forecasting framework for field-level NDVI prediction under sparse, irregular clear-sky acquisitions. The architecture separates the encoding of historical NDVI and meteorological observations from future exogenous covariates, fusing both representations for multi-step quantile prediction. To address irregular revisit patterns and horizon-dependent uncertainty, we introduce a temporal-distance weighted quantile loss that aligns the training objective with the effective forecasting horizon. In addition, we incorporate cumulati
    
[^25]: 餐巾图上的因果推断

    Causal Inference with the Napkin Graph

    [https://arxiv.org/abs/2512.19861](https://arxiv.org/abs/2512.19861)

    本文提出了一种针对“餐巾”图结构的因果效应识别与估计方法，通过非标准比率和基于影响函数的双重稳健估计量，在Verma约束下实现了半参数有效推断。

    

    未测量的混杂因素可能使基于调整函数的识别策略失效。我们研究了“餐巾”图，这是一种因果结构，它包含了M偏倚、工具变量、经典后门和前门设置的特征，但通过两个g公式的非标准比率来识别平均处理效应。我们为这一泛函开发了基于影响函数的估计量，包括双重稳健的一步估计量和基于目标最小损失的估计量，这些估计量在使用机器学习进行慢于参数速率的干扰项估计时仍保持渐近线性。餐巾图的一个显著特征是它对观测数据分布施加了一种广义独立性约束，即Verma约束，而非通常的条件独立性约束。我们在对应于这一约束的矩限制下，发展了因果效应的半参数效率理论。

    arXiv:2512.19861v2 Announce Type: replace-cross  Abstract: Unmeasured confounding can render identification strategies based on adjustment functionals invalid. We study the "Napkin" graph, a causal structure that encapsulates features of M-bias, instrumental variables, and classical back-door and front-door settings, yet identifies the average treatment effect through a nonstandard ratio of two g-formulas. We develop influence-function-based estimators for this functional, including doubly-robust one-step and targeted minimum loss-based estimators that remain asymptotically linear under slower-than-parametric nuisance estimation using machine learning. A distinguishing feature of the Napkin graph is that it imposes a generalized independence restriction, known as a Verma constraint, rather than ordinary conditional independence restrictions, on the observed data distribution. We develop semiparametric efficiency theory for causal effects under a moment restriction corresponding to this
    
[^26]: 基于机器学习的含干扰参数截面测量展开方法

    Machine Learning-based Unfolding for Cross Section Measurements in the Presence of Nuisance Parameters

    [https://arxiv.org/abs/2512.07074](https://arxiv.org/abs/2512.07074)

    本文基于OmniFold算法，提出了Profile OmniFold算法，将机器学习展开方法扩展至包含干扰参数，从而更准确地处理探测器模拟中的不确定性。

    

    arXiv:2512.07074v3 公告类型：替换交叉 摘要：对测量截面进行探测器效应的统计校正是许多应用中的重要步骤。在粒子物理中，这一逆问题被称为展开。在仪器复杂的情况下，它们引入的畸变通常仅通过探测器模拟隐含地已知。现代机器学习已实现基于模拟的高维数据展开高效方法。其中，首个成功应用于实验数据的方法之一是OmniFold算法，这是一种基于分类器的期望最大化过程。然而，在实际中，前向模型仅近似指定，相应的不确定性通过干扰参数编码。基于已被充分研究的OmniFold算法，我们展示了如何将基于机器学习的展开扩展到包含干扰参数。我们的新算法称为Profile OmniFold，并通过实验进行了演示。

    arXiv:2512.07074v3 Announce Type: replace-cross  Abstract: Statistically correcting measured cross sections for detector effects is an important step across many applications. In particle physics, this inverse problem is known as unfolding. In cases with complex instruments, the distortions they introduce are often known only implicitly through simulations of the detector. Modern machine learning has enabled efficient simulation-based approaches for unfolding high-dimensional data. Among these, one of the first methods successfully deployed on experimental data is the OmniFold algorithm, a classifier-based Expectation-Maximization procedure. In practice, however, the forward model is only approximately specified, and the corresponding uncertainty is encoded through nuisance parameters. Building on the well-studied OmniFold algorithm, we show how to extend machine learning-based unfolding to incorporate nuisance parameters. Our new algorithm, called Profile OmniFold, is demonstrated usi
    
[^27]: 没有免费午餐：预测驱动推断的非渐近分析

    No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference

    [https://arxiv.org/abs/2505.20178](https://arxiv.org/abs/2505.20178)

    本文通过有限样本分析证明，PPI++方法并非总是优于仅使用黄金标准标签，其优势仅在伪标签与黄金标准标签的相关性高于特定阈值时成立。

    

    预测驱动推断（PPI）是一种将黄金标准标签与可能有噪声的伪标签相结合进行统计估计的流行策略。先前的研究表明，PPI++（PPI的一种自适应形式）存在渐近意义上的“免费午餐”，即PPI++的渐近方差总是小于或等于仅使用黄金标准标签所获得的方差。值得注意的是，这一结论无论伪标签的质量如何都成立。在本工作中，我们通过对均值估计问题中PPI++的估计误差进行精确的有限样本分析，揭示了这一结果背后的真相。我们给出了一个“没有免费午餐”的结论，刻画了在哪些设定（以及样本量）下，PPI++的估计误差会明确差于仅使用黄金标准标签。具体而言，PPI++能够表现更好的充分必要条件是伪标签与黄金标准标签之间的相关性超过某个依赖于样本量的阈值。

    arXiv:2505.20178v2 Announce Type: replace-cross  Abstract: Prediction-Powered Inference (PPI) is a popular strategy for combining gold-standard and possibly noisy pseudo-labels to perform statistical estimation. Prior work has shown an asymptotic \enquote{free lunch} for PPI++, an adaptive form of PPI, showing that the \textit{asymptotic} variance of PPI++ is always less than or equal to the variance obtained from using gold-standard labels alone. Notably, this result holds \textit{regardless of the quality of the pseudo-labels}. In this work, we demystify this result by conducting an exact finite-sample analysis of the estimation error of PPI++ on the mean estimation problem. We give a \enquote{no free lunch} result, characterizing the settings (and sample sizes) where PPI++ has provably worse estimation error than using gold-standard labels alone. Specifically, PPI++ will outperform if and only if the correlation between pseudo- and gold-standard is above a certain level that depends
    
[^28]: 面向分布式内存优化的通信高效二维并行随机梯度下降方法

    Communication-Efficient, 2D Parallel Stochastic Gradient Descent for Distributed-Memory Optimization

    [https://arxiv.org/abs/2501.07526](https://arxiv.org/abs/2501.07526)

    本文提出一种名为HybridSGD的二维并行随机梯度下降方法，通过整合一维s步SGD和联邦平均SGD的优势，在分布式内存集群中实现了通信效率与性能的连续权衡，并理论验证了其在收敛性、计算、通信和内存方面的综合优势。

    

    数值优化算法（如随机梯度下降SGD）的分布式内存实现需要在算法的每次迭代中进行处理器间通信。在现代分布式内存集群中，通信成本高于计算成本，因此这些算法的可扩展性和性能受限于通信开销。本研究将先前关于一维s步SGD和一维联邦平均SGD（FedAvg）的工作推广为一种二维并行SGD方法（HybridSGD），该方法在两种基线算法之间实现了连续的性能权衡。我们提供了理论分析，展示了s步SGD、FedAvg、二维并行SGD以及其他并行SGD变体在收敛性、计算、通信和内存之间的权衡。我们在C++和MPI中实现了所有算法，并在Cray EX超级计算系统上评估了它们的性能。实验结果表明，HybridSGD在通信效率和性能上具有显著优势。

    arXiv:2501.07526v2 Announce Type: replace-cross  Abstract: Distributed-memory implementations of numerical optimization algorithm, such as stochastic gradient descent (SGD), require interprocessor communication at every iteration of the algorithm. On modern distributed-memory clusters where communication is more expensive than computation, the scalability and performance of these algorithms are limited by communication cost. This work generalizes prior work on 1D $s$-step SGD and 1D Federated SGD with Averaging (FedAvg) to yield a 2D parallel SGD method (HybridSGD) which attains a continuous performance trade off between the two baseline algorithms. We present theoretical analysis which show the convergence, computation, communication, and memory trade offs between $s$-step SGD, FedAvg, 2D parallel SGD, and other parallel SGD variants. We implement all algorithms in C++ and MPI and evaluate their performance on a Cray EX supercomputing system. Our empirical results show that HybridSGD 
    
[^29]: 从有偏样本中学习

    Learning from a Biased Sample

    [https://arxiv.org/abs/2209.01754](https://arxiv.org/abs/2209.01754)

    本文提出了一种新的条件Γ-有偏抽样模型来量化训练数据中的抽样偏差，并利用分布鲁棒优化框架开发了一种元方法，以在部署时仍能获得良好性能的决策规则。

    

    基于经验风险最小化的数据驱动决策方法要求训练数据与决策规则部署时所面临的条件相同。然而，在许多场景中，我们可能担心训练样本存在偏差，即某些群体（基于可观测或不可观测特征）相对于总体可能被低估或高估；在这种情况下，基于训练集的经验风险最小化可能无法生成在部署时表现良好的决策规则。我们提出了一种称为条件Γ-有偏抽样的抽样偏差模型，其中观测协变量可以任意影响样本选择概率，但样本选择概率中未解释的变化量被常数因子所限制。应用分布鲁棒优化框架，我们提出了一种元方法。

    arXiv:2209.01754v5 Announce Type: replace-cross  Abstract: The empirical risk minimization approach to data-driven decision making requires access to training data drawn under the same conditions as those that will be faced when the decision rule is deployed. However, in a number of settings, we may be concerned that our training sample is biased in the sense that some groups (characterized by either observable or unobservable attributes) may be under- or over-represented relative to the general population; and in this setting empirical risk minimization over the training set may fail to yield rules that perform well at deployment. We propose a model of sampling bias called conditional $\Gamma$-biased sampling, where observed covariates can affect the probability of sample selection arbitrarily much but the amount of unexplained variation in the probability of sample selection is bounded by a constant factor. Applying the distributionally robust optimization framework, we propose a met
    
[^30]: 通用深度神经网络的频率原理理论

    Theory of the Frequency Principle for General Deep Neural Networks

    [https://arxiv.org/abs/1906.09235](https://arxiv.org/abs/1906.09235)

    本文严格证明了通用深度神经网络在训练中从低频到高频学习的频率原理，并提供了针对不同训练阶段的理论定理，适用于多种激活函数、数据分布和损失函数。

    

    随着深度神经网络在现实问题中的广泛应用，近期一些关于DNN的实证研究报道了一个普遍现象——频率原理：在训练过程中，DNN倾向于从低频到高频学习目标函数。频率原理在提供DNN的定性和定量理解方面非常有用。本文严格研究了通用DNN在三个训练阶段（初始阶段、中间阶段和最终阶段）的频率原理。针对每个阶段，我们通过表征频率原理的适当量提供了相应定理。我们的结果具有普适性，适用于具有一般激活函数的多层网络、数据分布密度以及一大类损失函数。本研究为频率原理奠定了理论基础，有助于更好地理解训练过程。

    arXiv:1906.09235v3 Announce Type: replace  Abstract: Along with fruitful applications of Deep Neural Networks (DNNs) to realistic problems, recently, some empirical studies of DNNs reported a universal phenomenon of Frequency Principle (F-Principle): a DNN tends to learn a target function from low to high frequencies during the training. The F-Principle has been very useful in providing both qualitative and quantitative understandings of DNNs. In this paper, we rigorously investigate the F-Principle for the training dynamics of a general DNN at three stages: initial stage, intermediate stage, and final stage. For each stage, a theorem is provided in terms of proper quantities characterizing the F-Principle. Our results are general in the sense that they work for multilayer networks with general activation functions, population densities of data, and a large class of loss functions. Our work lays a theoretical foundation of the F-Principle for a better understanding of the training proc
    

