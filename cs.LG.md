# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DualView: Data Attribution from the Dual Perspective](https://arxiv.org/abs/2402.12118) | 提出了DualView，一种基于替代建模的后期数据归因方法，具有高效计算和优质评估结果。 |
| [^2] | [SCAFFLSA: Quantifying and Eliminating Heterogeneity Bias in Federated Linear Stochastic Approximation and Temporal Difference Learning](https://arxiv.org/abs/2402.04114) | 本文量化了联邦线性随机逼近算法中异质性偏差的影响，并提出SCAFFLSA作为一种改进方法来消除此偏差。在联邦时间差异学习中，该方法能够显著提高算法的复杂性。 |
| [^3] | [Differentially Private Bayesian Tests.](http://arxiv.org/abs/2401.15502) | 本文提出了一种差分隐私贝叶斯检验框架，利用规范化的数据生成机制来进行推断，并避免了对完整数据生成机制的建模需求。该框架具有可解释性，并在计算上具有实质性的优势。 |

# 详细

[^1]: DualView：双重视角下的数据归因

    DualView: Data Attribution from the Dual Perspective

    [https://arxiv.org/abs/2402.12118](https://arxiv.org/abs/2402.12118)

    提出了DualView，一种基于替代建模的后期数据归因方法，具有高效计算和优质评估结果。

    

    本文提出了DualView，这是一种基于替代建模的后期数据归因方法，展示了高计算效率和良好的评估结果。我们专注于神经网络，在与文献相关的适当定量评估策略下评估了我们提出的技术，比较了与相关主要本地数据归因方法的性能。

    arXiv:2402.12118v1 Announce Type: cross  Abstract: Local data attribution (or influence estimation) techniques aim at estimating the impact that individual data points seen during training have on particular predictions of an already trained Machine Learning model during test time. Previous methods either do not perform well consistently across different evaluation criteria from literature, are characterized by a high computational demand, or suffer from both. In this work we present DualView, a novel method for post-hoc data attribution based on surrogate modelling, demonstrating both high computational efficiency, as well as good evaluation results. With a focus on neural networks, we evaluate our proposed technique using suitable quantitative evaluation strategies from the literature against related principal local data attribution methods. We find that DualView requires considerably lower computational resources than other methods, while demonstrating comparable performance to comp
    
[^2]: SCAFFLSA：量化和消除联邦线性随机逼近和时间差异学习中的异质性偏差

    SCAFFLSA: Quantifying and Eliminating Heterogeneity Bias in Federated Linear Stochastic Approximation and Temporal Difference Learning

    [https://arxiv.org/abs/2402.04114](https://arxiv.org/abs/2402.04114)

    本文量化了联邦线性随机逼近算法中异质性偏差的影响，并提出SCAFFLSA作为一种改进方法来消除此偏差。在联邦时间差异学习中，该方法能够显著提高算法的复杂性。

    

    本文对联邦线性随机逼近算法（FedLSA）进行了非渐进分析。我们明确量化了异质代理本地训练引入的偏差，并研究了该算法的样本复杂性。我们证明了FedLSA的通信复杂性与所需精度 $\epsilon$ 呈多项式关系，这限制了联邦的好处。为了克服这一问题，我们提出了SCAFFLSA，一种新型的FedLSA变体，它使用控制变量来校正本地训练的偏差，并在不对统计异质性做出任何假设的情况下证明了其收敛性。我们将所提出的方法应用于具有线性函数逼近的联邦时间差异学习，并分析了相应的复杂性改进。

    In this paper, we perform a non-asymptotic analysis of the federated linear stochastic approximation (FedLSA) algorithm. We explicitly quantify the bias introduced by local training with heterogeneous agents, and investigate the sample complexity of the algorithm. We show that the communication complexity of FedLSA scales polynomially with the desired precision $\epsilon$, which limits the benefits of federation. To overcome this, we propose SCAFFLSA, a novel variant of FedLSA, that uses control variates to correct the bias of local training, and prove its convergence without assumptions on statistical heterogeneity. We apply the proposed methodology to federated temporal difference learning with linear function approximation, and analyze the corresponding complexity improvements.
    
[^3]: 差分隐私贝叶斯检验

    Differentially Private Bayesian Tests. (arXiv:2401.15502v1 [stat.ML])

    [http://arxiv.org/abs/2401.15502](http://arxiv.org/abs/2401.15502)

    本文提出了一种差分隐私贝叶斯检验框架，利用规范化的数据生成机制来进行推断，并避免了对完整数据生成机制的建模需求。该框架具有可解释性，并在计算上具有实质性的优势。

    

    在利用机密数据进行科学假设检验的领域中，差分隐私已经成为一个重要的基石。在报告科学发现时，广泛采用贝叶斯检验，因为它们有效地避免了P值的主要批评，即缺乏可解释性和无法量化对竞争假设的支持证据。我们提出了一个新颖的差分隐私贝叶斯假设检验框架，该框架在基于规范化的数据生成机制基础上自然产生，从而保持了推断结果的可解释性。此外，通过专注于基于广泛使用的检验统计量的差分隐私贝叶斯因子，我们避免了对完整数据生成机制建模的需求，并确保了实质性的计算优势。我们还提供了一组充分条件，以在所提框架下确立贝叶斯因子一致性的结果。

    Differential privacy has emerged as an significant cornerstone in the realm of scientific hypothesis testing utilizing confidential data. In reporting scientific discoveries, Bayesian tests are widely adopted since they effectively circumnavigate the key criticisms of P-values, namely, lack of interpretability and inability to quantify evidence in support of the competing hypotheses. We present a novel differentially private Bayesian hypotheses testing framework that arise naturally under a principled data generative mechanism, inherently maintaining the interpretability of the resulting inferences. Furthermore, by focusing on differentially private Bayes factors based on widely used test statistics, we circumvent the need to model the complete data generative mechanism and ensure substantial computational benefits. We also provide a set of sufficient conditions to establish results on Bayes factor consistency under the proposed framework. The utility of the devised technology is showc
    

