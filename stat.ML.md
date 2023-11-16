# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Supervised and Penalized Baseline Correction.](http://arxiv.org/abs/2310.18306) | 本研究改进了受罚基线校正方法，通过利用先验分析物浓度来改善光谱预测性能，并在两个近红外数据集上进行了评估。 |
| [^2] | [On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers.](http://arxiv.org/abs/2310.14421) | 本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。 |
| [^3] | [Graph-based Neural Weather Prediction for Limited Area Modeling.](http://arxiv.org/abs/2309.17370) | 该论文提出了一种基于图像神经网络的有限区域天气预测方法，并通过使用北欧地区的本地模型进行了验证。 |
| [^4] | [Thompson Sampling for Real-Valued Combinatorial Pure Exploration of Multi-Armed Bandit.](http://arxiv.org/abs/2308.10238) | 这篇论文介绍了一种名为广义汤普森抽样探索算法，能够解决多臂老虎机实值组合纯探索问题中动作集合大小为指数级别的情况。 |
| [^5] | [Error Bounds for Learning with Vector-Valued Random Features.](http://arxiv.org/abs/2305.17170) | 本文提供了对向量值随机特征学习的完整误差分析，包括在模型错误说明下向量值RF估计器的强一致性和在良好规定的情况下极小化最优收敛速率。 |
| [^6] | [Is My Prediction Arbitrary? Measuring Self-Consistency in Fair Classification.](http://arxiv.org/abs/2301.11562) | 在公平分类中，模型的预测方差是一个重要但鲜为人知的误差来源问题。作者提出了一个自洽性标准来衡量测量和减少随意性。作者还开发了一个算法来处理随意性预测，并通过实证研究揭示了当前模型无法处理某些类型数据的问题。 |
| [^7] | [Optimal Approximation Rates for Deep ReLU Neural Networks on Sobolev and Besov Spaces.](http://arxiv.org/abs/2211.14400) | 该论文研究了在Sobolev和Besov空间中，使用ReLU激活函数的深度神经网络能够以怎样的参数效率逼近函数，包括$L_p(\Omega)$范数下的误差度量。我们提供了所有$1\leq p,q \leq \infty$和$s>0$的完整解决方案，并引入了一种新的位提取技术来获得尖锐的上界。 |

# 详细

[^1]: 监督和受罚基线校正

    Supervised and Penalized Baseline Correction. (arXiv:2310.18306v1 [stat.ML])

    [http://arxiv.org/abs/2310.18306](http://arxiv.org/abs/2310.18306)

    本研究改进了受罚基线校正方法，通过利用先验分析物浓度来改善光谱预测性能，并在两个近红外数据集上进行了评估。

    

    光谱测量可以显示由吸收和散射成分混合引起的扭曲光谱形状。这些扭曲（或基线）通常表现为非恒定偏移或低频振荡。因此，这些基线可能对分析和定量结果产生不利影响。基线校正是一个涵盖了预处理方法的总称，通过获取基线光谱（不需要的扭曲）并通过差异化去除扭曲。然而，当前最先进的基线校正方法即使可用分析物浓度或者它们对观察到的光谱变异有重要贡献，也没有利用它们。我们研究了一类最先进的方法（受罚基线校正）并对其进行修改，使其能够适应先验分析物浓度，从而提高预测性能。将在两个近红外数据集上评估性能，包括经典受罚方法。

    Spectroscopic measurements can show distorted spectra shapes arising from a mixture of absorbing and scattering contributions. These distortions (or baselines) often manifest themselves as non-constant offsets or low-frequency oscillations. As a result, these baselines can adversely affect analytical and quantitative results. Baseline correction is an umbrella term where one applies pre-processing methods to obtain baseline spectra (the unwanted distortions) and then remove the distortions by differencing. However, current state-of-the art baseline correction methods do not utilize analyte concentrations even if they are available, or even if they contribute significantly to the observed spectral variability. We examine a class of state-of-the-art methods (penalized baseline correction) and modify them such that they can accommodate a priori analyte concentration such that prediction can be enhanced. Performance will be access on two near infra-red data sets across both classical penal
    
[^2]: 对AI分类器的对抗鲁棒性度量的存在性，唯一性和可扩展性研究

    On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers. (arXiv:2310.14421v1 [stat.ML])

    [http://arxiv.org/abs/2310.14421](http://arxiv.org/abs/2310.14421)

    本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。

    

    本文提出并证明了针对（局部）唯一可逆分类器、广义线性模型（GLM）和熵AI（EAI）具有最小对抗路径（MAP）和最小对抗距离（MAD）的存在性、唯一性和明确的分析计算的简单可验证的数学条件。在常见的合成基准测试数据集上，针对神经网络、提升随机森林、GLM和EAI等各类AI工具进行MAP和MAD的实际计算、比较和解释，包括双卷状螺旋线及其扩展以及两个生物医学数据问题（用于健康保险理赔预测和心脏病发作致死率分类）。在生物医学应用中，展示了MAP如何在预定义的可访问控制变量子集中提供唯一的最小患者特定风险缓解干预措施。

    Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.
    
[^3]: 基于图像神经网络的有限区域天气预测

    Graph-based Neural Weather Prediction for Limited Area Modeling. (arXiv:2309.17370v1 [cs.LG])

    [http://arxiv.org/abs/2309.17370](http://arxiv.org/abs/2309.17370)

    该论文提出了一种基于图像神经网络的有限区域天气预测方法，并通过使用北欧地区的本地模型进行了验证。

    

    高精度的机器学习方法在天气预报领域的应用为模拟大气的可能性带来了新的变革。在气候变化时代，获取像这样的高分辨率预报模型的能力变得越来越重要。虽然大多数现有的神经网络天气预报方法都是针对全球预测，但如何将这些技术应用于有限区域建模是一个重要问题。本文将基于图像的神经网络天气预测方法应用于有限区域，并提出了多尺度分层模型扩展。通过使用北欧地区的本地模型进行实验证实了我们的方法的有效性。

    The rise of accurate machine learning methods for weather forecasting is creating radical new possibilities for modeling the atmosphere. In the time of climate change, having access to high-resolution forecasts from models like these is also becoming increasingly vital. While most existing Neural Weather Prediction (NeurWP) methods focus on global forecasting, an important question is how these techniques can be applied to limited area modeling. In this work we adapt the graph-based NeurWP approach to the limited area setting and propose a multi-scale hierarchical model extension. Our approach is validated by experiments with a local model for the Nordic region.
    
[^4]: Thompson Sampling用于多臂老虎机的实值组合纯探索

    Thompson Sampling for Real-Valued Combinatorial Pure Exploration of Multi-Armed Bandit. (arXiv:2308.10238v1 [cs.LG])

    [http://arxiv.org/abs/2308.10238](http://arxiv.org/abs/2308.10238)

    这篇论文介绍了一种名为广义汤普森抽样探索算法，能够解决多臂老虎机实值组合纯探索问题中动作集合大小为指数级别的情况。

    

    我们研究了多臂老虎机的实值组合纯探索（R-CPE-MAB）问题。在R-CPE-MAB中，玩家从给定的d个随机臂中选择一个，每个臂s的奖励遵循未知分布，其平均值为μs。在每个时间步骤中，玩家拉动一个臂并观察其奖励。玩家的目标是以尽可能少的臂拉动次数来确定最优动作π* = argmaxπ∈A μTπ，其中A是有限大小的实值动作集合。之前的方法假设动作集合A的大小在d的多项式级别上。我们引入了一种名为广义汤普森抽样探索（GenTS-Explore）算法，它是第一个可以解决动作集合大小在d的指数级别上的算法。同时，我们还引入了一种新的问题相关的样本复杂性。

    We study the real-valued combinatorial pure exploration of the multi-armed bandit (R-CPE-MAB) problem. In R-CPE-MAB, a player is given $d$ stochastic arms, and the reward of each arm $s\in\{1, \ldots, d\}$ follows an unknown distribution with mean $\mu_s$. In each time step, a player pulls a single arm and observes its reward. The player's goal is to identify the optimal \emph{action} $\boldsymbol{\pi}^{*} = \argmax_{\boldsymbol{\pi} \in \mathcal{A}} \boldsymbol{\mu}^{\top}\boldsymbol{\pi}$ from a finite-sized real-valued \emph{action set} $\mathcal{A}\subset \mathbb{R}^{d}$ with as few arm pulls as possible. Previous methods in the R-CPE-MAB assume that the size of the action set $\mathcal{A}$ is polynomial in $d$. We introduce an algorithm named the Generalized Thompson Sampling Explore (GenTS-Explore) algorithm, which is the first algorithm that can work even when the size of the action set is exponentially large in $d$. We also introduce a novel problem-dependent sample complexity 
    
[^5]: 向量值随机特征学习的误差界分析

    Error Bounds for Learning with Vector-Valued Random Features. (arXiv:2305.17170v1 [stat.ML])

    [http://arxiv.org/abs/2305.17170](http://arxiv.org/abs/2305.17170)

    本文提供了对向量值随机特征学习的完整误差分析，包括在模型错误说明下向量值RF估计器的强一致性和在良好规定的情况下极小化最优收敛速率。

    

    本文提供了对于向量值随机特征学习的完整误差分析。该理论是针对完全通用的无限维度输入-输出设定中的RF Ridge回归而开发的，但仍适用于并改进了现有的有限维度分析。与文献中其他类似的工作相比，本文提出的方法依赖于底层风险函数的直接分析，完全避免了基于随机矩阵的显式RF Ridge回归解决方案公式的使用。这消除了随机矩阵理论中的浓度结果或其对随机算子的推广的需求。本文建立的主要结果包括在模型错误说明下向量值RF估计器的强一致性和在良好规定的情况下极小化最优收敛速率。实现这些收敛速率所需的参数复杂度(随机特征数量)和样本复杂度(标记数据数量)与

    This paper provides a comprehensive error analysis of learning with vector-valued random features (RF). The theory is developed for RF ridge regression in a fully general infinite-dimensional input-output setting, but nonetheless applies to and improves existing finite-dimensional analyses. In contrast to comparable work in the literature, the approach proposed here relies on a direct analysis of the underlying risk functional and completely avoids the explicit RF ridge regression solution formula in terms of random matrices. This removes the need for concentration results in random matrix theory or their generalizations to random operators. The main results established in this paper include strong consistency of vector-valued RF estimators under model misspecification and minimax optimal convergence rates in the well-specified setting. The parameter complexity (number of random features) and sample complexity (number of labeled data) required to achieve such rates are comparable with 
    
[^6]: 预测是否随意？在公平分类中评估自洽性

    Is My Prediction Arbitrary? Measuring Self-Consistency in Fair Classification. (arXiv:2301.11562v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.11562](http://arxiv.org/abs/2301.11562)

    在公平分类中，模型的预测方差是一个重要但鲜为人知的误差来源问题。作者提出了一个自洽性标准来衡量测量和减少随意性。作者还开发了一个算法来处理随意性预测，并通过实证研究揭示了当前模型无法处理某些类型数据的问题。

    

    在公平分类中，不同经过训练的模型之间的预测方差是一个重要但鲜为人知的误差来源问题。 实证表明，某些情况下，预测的方差差异非常大，以至于决策实际上是随意的。 为了研究这个问题，我们进行了大规模的实证研究，并做出了四个总体贡献：我们1）定义了一种基于方差的度量标准，称为自洽性，在测量和减少随意性时使用； 2）开发了一种合理的算法，当预测无法做出决策时，可以放弃分类； 3）进行了迄今为止有关公平分类中方差（相对于自洽性和随意性）作用的最大规模实证研究； 4）推出了一个工具包，使美国住房抵押贷款披露法案（HMDA）数据集易于用于未来研究。 总的来说，我们的实证结果揭示了关于可重复性的令人震惊的见解。当考虑到方差和随意预测的可能性时，大多数公平分类基准接近公平。 但是，一小部分实例显示出极大的随意性水平，这表明当前的模型可能无法处理某些类型的数据。

    Variance in predictions across different trained models is a significant, under-explored source of error in fair classification. Empirically, the variance on some instances is so large that decisions can be effectively arbitrary. To study this problem, we perform a large-scale empirical study and make four overarching contributions: We 1) Define a metric called self-consistency, derived from variance, which we use as a proxy for measuring and reducing arbitrariness; 2) Develop an ensembling algorithm that abstains from classification when a prediction would be arbitrary; 3) Conduct the largest to-date empirical study of the role of variance (vis-a-vis self-consistency and arbitrariness) in fair classification; and, 4) Release a toolkit that makes the US Home Mortgage Disclosure Act (HMDA) datasets easily usable for future research. Altogether, our empirical results reveal shocking insights about reproducibility. Most fairness classification benchmarks are close-to-fair when taking into
    
[^7]: 在Sobolev和Besov空间上，关于深度ReLU神经网络的最佳逼近速率研究

    Optimal Approximation Rates for Deep ReLU Neural Networks on Sobolev and Besov Spaces. (arXiv:2211.14400v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.14400](http://arxiv.org/abs/2211.14400)

    该论文研究了在Sobolev和Besov空间中，使用ReLU激活函数的深度神经网络能够以怎样的参数效率逼近函数，包括$L_p(\Omega)$范数下的误差度量。我们提供了所有$1\leq p,q \leq \infty$和$s>0$的完整解决方案，并引入了一种新的位提取技术来获得尖锐的上界。

    

    本文研究了使用ReLU激活函数的深度神经网络在Sobolev空间$W^s(L_q(\Omega))$和Besov空间$B^s_r(L_q(\Omega))$中以$L_p(\Omega)$范数度量误差的参数效率问题。我们的研究对于在科学计算和信号处理等领域中应用神经网络非常重要，在过去只有当$p=q=\infty$时才完全解决。我们的贡献是提供了所有$1\leq p,q\leq \infty$和$s>0$的完整解决方案，包括渐近匹配的上下界。关键的技术工具是一种新的位提取技术，它提供了稀疏向量的最佳编码。这使我们能够在$p>q$的非线性区域获得尖锐的上界。我们还提供了一种基于的$L_p$逼近下界推导的新方法。

    Let $\Omega = [0,1]^d$ be the unit cube in $\mathbb{R}^d$. We study the problem of how efficiently, in terms of the number of parameters, deep neural networks with the ReLU activation function can approximate functions in the Sobolev spaces $W^s(L_q(\Omega))$ and Besov spaces $B^s_r(L_q(\Omega))$, with error measured in the $L_p(\Omega)$ norm. This problem is important when studying the application of neural networks in a variety of fields, including scientific computing and signal processing, and has previously been completely solved only when $p=q=\infty$. Our contribution is to provide a complete solution for all $1\leq p,q\leq \infty$ and $s > 0$, including asymptotically matching upper and lower bounds. The key technical tool is a novel bit-extraction technique which gives an optimal encoding of sparse vectors. This enables us to obtain sharp upper bounds in the non-linear regime where $p > q$. We also provide a novel method for deriving $L_p$-approximation lower bounds based upon
    

