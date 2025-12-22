# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SCAFFLSA: Quantifying and Eliminating Heterogeneity Bias in Federated Linear Stochastic Approximation and Temporal Difference Learning](https://arxiv.org/abs/2402.04114) | 本文量化了联邦线性随机逼近算法中异质性偏差的影响，并提出SCAFFLSA作为一种改进方法来消除此偏差。在联邦时间差异学习中，该方法能够显著提高算法的复杂性。 |
| [^2] | [Differentially Private Bayesian Tests.](http://arxiv.org/abs/2401.15502) | 本文提出了一种差分隐私贝叶斯检验框架，利用规范化的数据生成机制来进行推断，并避免了对完整数据生成机制的建模需求。该框架具有可解释性，并在计算上具有实质性的优势。 |
| [^3] | [Unifying Distributionally Robust Optimization via Optimal Transport Theory.](http://arxiv.org/abs/2308.05414) | 本文通过最优输运理论将分布鲁棒优化的散度方法和Wasserstein方法统一到一个框架中，并且提出了可以同时扰动似然和结果的最优对抗分布。这个统一框架在实际应用中具有较强的可行性和实用性。 |

# 详细

[^1]: SCAFFLSA：量化和消除联邦线性随机逼近和时间差异学习中的异质性偏差

    SCAFFLSA: Quantifying and Eliminating Heterogeneity Bias in Federated Linear Stochastic Approximation and Temporal Difference Learning

    [https://arxiv.org/abs/2402.04114](https://arxiv.org/abs/2402.04114)

    本文量化了联邦线性随机逼近算法中异质性偏差的影响，并提出SCAFFLSA作为一种改进方法来消除此偏差。在联邦时间差异学习中，该方法能够显著提高算法的复杂性。

    

    本文对联邦线性随机逼近算法（FedLSA）进行了非渐进分析。我们明确量化了异质代理本地训练引入的偏差，并研究了该算法的样本复杂性。我们证明了FedLSA的通信复杂性与所需精度 $\epsilon$ 呈多项式关系，这限制了联邦的好处。为了克服这一问题，我们提出了SCAFFLSA，一种新型的FedLSA变体，它使用控制变量来校正本地训练的偏差，并在不对统计异质性做出任何假设的情况下证明了其收敛性。我们将所提出的方法应用于具有线性函数逼近的联邦时间差异学习，并分析了相应的复杂性改进。

    In this paper, we perform a non-asymptotic analysis of the federated linear stochastic approximation (FedLSA) algorithm. We explicitly quantify the bias introduced by local training with heterogeneous agents, and investigate the sample complexity of the algorithm. We show that the communication complexity of FedLSA scales polynomially with the desired precision $\epsilon$, which limits the benefits of federation. To overcome this, we propose SCAFFLSA, a novel variant of FedLSA, that uses control variates to correct the bias of local training, and prove its convergence without assumptions on statistical heterogeneity. We apply the proposed methodology to federated temporal difference learning with linear function approximation, and analyze the corresponding complexity improvements.
    
[^2]: 差分隐私贝叶斯检验

    Differentially Private Bayesian Tests. (arXiv:2401.15502v1 [stat.ML])

    [http://arxiv.org/abs/2401.15502](http://arxiv.org/abs/2401.15502)

    本文提出了一种差分隐私贝叶斯检验框架，利用规范化的数据生成机制来进行推断，并避免了对完整数据生成机制的建模需求。该框架具有可解释性，并在计算上具有实质性的优势。

    

    在利用机密数据进行科学假设检验的领域中，差分隐私已经成为一个重要的基石。在报告科学发现时，广泛采用贝叶斯检验，因为它们有效地避免了P值的主要批评，即缺乏可解释性和无法量化对竞争假设的支持证据。我们提出了一个新颖的差分隐私贝叶斯假设检验框架，该框架在基于规范化的数据生成机制基础上自然产生，从而保持了推断结果的可解释性。此外，通过专注于基于广泛使用的检验统计量的差分隐私贝叶斯因子，我们避免了对完整数据生成机制建模的需求，并确保了实质性的计算优势。我们还提供了一组充分条件，以在所提框架下确立贝叶斯因子一致性的结果。

    Differential privacy has emerged as an significant cornerstone in the realm of scientific hypothesis testing utilizing confidential data. In reporting scientific discoveries, Bayesian tests are widely adopted since they effectively circumnavigate the key criticisms of P-values, namely, lack of interpretability and inability to quantify evidence in support of the competing hypotheses. We present a novel differentially private Bayesian hypotheses testing framework that arise naturally under a principled data generative mechanism, inherently maintaining the interpretability of the resulting inferences. Furthermore, by focusing on differentially private Bayes factors based on widely used test statistics, we circumvent the need to model the complete data generative mechanism and ensure substantial computational benefits. We also provide a set of sufficient conditions to establish results on Bayes factor consistency under the proposed framework. The utility of the devised technology is showc
    
[^3]: 通过最优输运理论统一分布鲁棒优化

    Unifying Distributionally Robust Optimization via Optimal Transport Theory. (arXiv:2308.05414v1 [math.OC])

    [http://arxiv.org/abs/2308.05414](http://arxiv.org/abs/2308.05414)

    本文通过最优输运理论将分布鲁棒优化的散度方法和Wasserstein方法统一到一个框架中，并且提出了可以同时扰动似然和结果的最优对抗分布。这个统一框架在实际应用中具有较强的可行性和实用性。

    

    在过去几年中，对于分布鲁棒优化 (DRO) 有两种主要方法引起了相当大的关注：基于散度和基于Wasserstein的方法。散度方法使用似然比来建模错配，而后者使用实际结果的距离或成本来建模错配。在这些进展的基础上，本文引入了一种新的方法，将这些方法统一到一个基于最优输运 (OT) 和条件矩约束的框架中。例如，我们提出的方法可以使得最优对抗分布同时扰动似然和结果，并在基线模型和对抗模型之间产生一个最优 (从最优输运意义上) 的耦合。此外，本文还研究了几个对偶结果，并提出了可行的改进，增强了这个统一框架的实际适用性。

    In the past few years, there has been considerable interest in two prominent approaches for Distributionally Robust Optimization (DRO): Divergence-based and Wasserstein-based methods. The divergence approach models misspecification in terms of likelihood ratios, while the latter models it through a measure of distance or cost in actual outcomes. Building upon these advances, this paper introduces a novel approach that unifies these methods into a single framework based on optimal transport (OT) with conditional moment constraints. Our proposed approach, for example, makes it possible for optimal adversarial distributions to simultaneously perturb likelihood and outcomes, while producing an optimal (in an optimal transport sense) coupling between the baseline model and the adversarial model.Additionally, the paper investigates several duality results and presents tractable reformulations that enhance the practical applicability of this unified framework.
    

