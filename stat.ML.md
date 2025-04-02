# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty quantification for data-driven weather models](https://arxiv.org/abs/2403.13458) | 研究旨在系统比较不确定性量化方法，以生成概率性天气预测，超越传统基于物理的天气预测模型。 |
| [^2] | [Sharp Rates in Dependent Learning Theory: Avoiding Sample Size Deflation for the Square Loss](https://arxiv.org/abs/2402.05928) | 本文研究了依赖学习理论中的尖锐率，主要是为了避免样本大小缩减对方差产生影响。当假设类别的拓扑结构符合某些条件时，经验风险最小化者的性能与类别的复杂性和二阶统计量有关。 |
| [^3] | [Causal Dynamic Variational Autoencoder for Counterfactual Regression in Longitudinal Data.](http://arxiv.org/abs/2310.10559) | 本论文提出了一种因果动态变分自编码器（CDVAE）来解决纵向数据中的反事实回归问题。该方法假设存在未观察到的调整变量，并通过结合动态变分自编码器（DVAE）框架和使用倾向得分的加权策略来估计反事实响应。 |

# 详细

[^1]: 基于数据驱动的天气模型的不确定性量化

    Uncertainty quantification for data-driven weather models

    [https://arxiv.org/abs/2403.13458](https://arxiv.org/abs/2403.13458)

    研究旨在系统比较不确定性量化方法，以生成概率性天气预测，超越传统基于物理的天气预测模型。

    

    人工智能（AI）驱动的数据驱动天气预报模型在过去几年取得了快速进展。最近的研究使用再分析数据训练的模型取得了令人印象深刻的结果，并在一系列变量和评估指标上展示了明显改进，超越了现有的基于物理的数值天气预测模型。除了改进的预测外，数据驱动天气模型的主要优势是它们显著较低的计算成本和一旦模型被训练就能更快地生成预测。然而，大多数数据驱动天气预测的努力都局限于确定性的、点值预测，使得无法量化预测的不确定性，对于研究和应用中的最佳决策至关重要。我们的整体目标是系统地研究和比较不确定性量化方法，以生成概率性预测。

    arXiv:2403.13458v1 Announce Type: cross  Abstract: Artificial intelligence (AI)-based data-driven weather forecasting models have experienced rapid progress over the last years. Recent studies, with models trained on reanalysis data, achieve impressive results and demonstrate substantial improvements over state-of-the-art physics-based numerical weather prediction models across a range of variables and evaluation metrics. Beyond improved predictions, the main advantages of data-driven weather models are their substantially lower computational costs and the faster generation of forecasts, once a model has been trained. However, most efforts in data-driven weather forecasting have been limited to deterministic, point-valued predictions, making it impossible to quantify forecast uncertainties, which is crucial in research and for optimal decision making in applications. Our overarching aim is to systematically study and compare uncertainty quantification methods to generate probabilistic 
    
[^2]: 依赖学习理论中的尖锐率：避免样本大小缩减的平方损失

    Sharp Rates in Dependent Learning Theory: Avoiding Sample Size Deflation for the Square Loss

    [https://arxiv.org/abs/2402.05928](https://arxiv.org/abs/2402.05928)

    本文研究了依赖学习理论中的尖锐率，主要是为了避免样本大小缩减对方差产生影响。当假设类别的拓扑结构符合某些条件时，经验风险最小化者的性能与类别的复杂性和二阶统计量有关。

    

    本文研究了具有依赖性（β-混合）数据和平方损失的统计学习，在一个假设类别Φ_p的子集F中，其中Φ_p是范数∥f∥_Φ_p≡sup_m≥1 m^{-1/p}∥f∥_L^m，其中p∈[2，∞]。我们的研究动机是在具有依赖性数据的学习中寻找尖锐的噪声交互项或方差代理。在没有任何可实现性假设的情况下，典型的非渐近结果显示出方差代理通过底层协变量过程的混合时间进行了乘积缩减。我们证明，只要在我们的假设类别F上，L^2和Φ_p的拓扑是可比较的，即Φ_p是一个弱亚高斯类别：∥f∥_Φ_p≲∥f∥_L^2^η，其中η∈(0，1]，经验风险最小化者在其主导项中只实现了一种只依赖于类别复杂性和二阶统计量的速率。我们的结果适用于许多依赖性数据模型。

    In this work, we study statistical learning with dependent ($\beta$-mixing) data and square loss in a hypothesis class $\mathscr{F}\subset L_{\Psi_p}$ where $\Psi_p$ is the norm $\|f\|_{\Psi_p} \triangleq \sup_{m\geq 1} m^{-1/p} \|f\|_{L^m} $ for some $p\in [2,\infty]$. Our inquiry is motivated by the search for a sharp noise interaction term, or variance proxy, in learning with dependent data. Absent any realizability assumption, typical non-asymptotic results exhibit variance proxies that are deflated \emph{multiplicatively} by the mixing time of the underlying covariates process. We show that whenever the topologies of $L^2$ and $\Psi_p$ are comparable on our hypothesis class $\mathscr{F}$ -- that is, $\mathscr{F}$ is a weakly sub-Gaussian class: $\|f\|_{\Psi_p} \lesssim \|f\|_{L^2}^\eta$ for some $\eta\in (0,1]$ -- the empirical risk minimizer achieves a rate that only depends on the complexity of the class and second order statistics in its leading term. Our result holds whether t
    
[^3]: 因果动态变分自编码器用于纵向数据中的反事实回归

    Causal Dynamic Variational Autoencoder for Counterfactual Regression in Longitudinal Data. (arXiv:2310.10559v1 [stat.ML])

    [http://arxiv.org/abs/2310.10559](http://arxiv.org/abs/2310.10559)

    本论文提出了一种因果动态变分自编码器（CDVAE）来解决纵向数据中的反事实回归问题。该方法假设存在未观察到的调整变量，并通过结合动态变分自编码器（DVAE）框架和使用倾向得分的加权策略来估计反事实响应。

    

    在很多实际应用中，如精准医学、流行病学、经济和市场营销中，估计随时间变化的治疗效果是相关的。许多最先进的方法要么假设了所有混杂变量的观测结果，要么试图推断未观察到的混杂变量。我们采取了不同的观点，假设存在未观察到的风险因素，即仅影响结果序列的调整变量。在无混杂性的情况下，我们以未观测到的风险因素导致的治疗反应中的未知异质性为目标，估计个体治疗效果（ITE）。我们应对了时变效应和未观察到的调整变量所带来的挑战。在学习到的调整变量的有效性和治疗效果的一般化界限的理论结果指导下，我们设计了因果DVAE（CDVAE）。该模型将动态变分自编码器（DVAE）框架与使用倾向得分的加权策略相结合，用于估计反事实响应。

    Estimating treatment effects over time is relevant in many real-world applications, such as precision medicine, epidemiology, economy, and marketing. Many state-of-the-art methods either assume the observations of all confounders or seek to infer the unobserved ones. We take a different perspective by assuming unobserved risk factors, i.e., adjustment variables that affect only the sequence of outcomes. Under unconfoundedness, we target the Individual Treatment Effect (ITE) estimation with unobserved heterogeneity in the treatment response due to missing risk factors. We address the challenges posed by time-varying effects and unobserved adjustment variables. Led by theoretical results over the validity of the learned adjustment variables and generalization bounds over the treatment effect, we devise Causal DVAE (CDVAE). This model combines a Dynamic Variational Autoencoder (DVAE) framework with a weighting strategy using propensity scores to estimate counterfactual responses. The CDVA
    

