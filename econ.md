# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Bi-level Sparse Group Regressions for Macroeconomic Forecasting](https://arxiv.org/abs/2404.02671) | 提出了基于贝叶斯双层稀疏组回归的机器学习方法，可以进行高维宏观经济预测，并且理论证明其具有最小极限速率的收缩性，能够恢复模型参数，支持集包含模型的支持集。 |
| [^2] | [Does AI help humans make better decisions? A methodological framework for experimental evaluation](https://arxiv.org/abs/2403.12108) | 引入一种新的实验框架用于评估人类是否通过使用AI可以做出更好的决策，在单盲实验设计中比较了三种决策系统的表现 |
| [^3] | [Generative Probabilistic Forecasting with Applications in Market Operations](https://arxiv.org/abs/2403.05743) | 提出了一种基于Wiener-Kallianpur创新表示的生成式概率预测方法，包括自编码器和新颖的深度学习算法，具有渐近最优性和结构收敛性质，适用于实时市场运营中的高动态和波动时间序列。 |

# 详细

[^1]: 基于贝叶斯双层稀疏组回归的宏观经济预测

    Bayesian Bi-level Sparse Group Regressions for Macroeconomic Forecasting

    [https://arxiv.org/abs/2404.02671](https://arxiv.org/abs/2404.02671)

    提出了基于贝叶斯双层稀疏组回归的机器学习方法，可以进行高维宏观经济预测，并且理论证明其具有最小极限速率的收缩性，能够恢复模型参数，支持集包含模型的支持集。

    

    我们提出了一种机器学习方法，在已知具有组结构的协变量的高维设置中进行最优宏观经济预测。我们的模型涵盖了许多时间序列、混合频率和未知非线性的预测设置。我们引入了时间序列计量经济学中的双层稀疏概念，即稀疏性在组水平和组内均成立，我们假设真实模型符合这一假设。我们提出了一种引起双层稀疏性的先验，相应的后验分布被证明以最小极限速率收缩，恢复模型参数，并且其支持集在渐近上包含模型的支持集。我们的理论允许组间相关性，而同一组中的预测变量可以通过强相关性以及共同特征和模式进行表征。通过全面展示有限样本的性能来说明。

    arXiv:2404.02671v1 Announce Type: new  Abstract: We propose a Machine Learning approach for optimal macroeconomic forecasting in a high-dimensional setting with covariates presenting a known group structure. Our model encompasses forecasting settings with many series, mixed frequencies, and unknown nonlinearities. We introduce in time-series econometrics the concept of bi-level sparsity, i.e. sparsity holds at both the group level and within groups, and we assume the true model satisfies this assumption. We propose a prior that induces bi-level sparsity, and the corresponding posterior distribution is demonstrated to contract at the minimax-optimal rate, recover the model parameters, and have a support that includes the support of the model asymptotically. Our theory allows for correlation between groups, while predictors in the same group can be characterized by strong covariation as well as common characteristics and patterns. Finite sample performance is illustrated through comprehe
    
[^2]: AI是否有助于人类做出更好的决策？一种用于实验评估的方法论框架

    Does AI help humans make better decisions? A methodological framework for experimental evaluation

    [https://arxiv.org/abs/2403.12108](https://arxiv.org/abs/2403.12108)

    引入一种新的实验框架用于评估人类是否通过使用AI可以做出更好的决策，在单盲实验设计中比较了三种决策系统的表现

    

    基于数据驱动算法的人工智能（AI）在当今社会变得无处不在。然而，在许多情况下，尤其是当利益高昂时，人类仍然作出最终决策。因此，关键问题是AI是否有助于人类比单独的人类或单独的AI做出更好的决策。我们引入了一种新的方法论框架，用于实验性地回答这个问题，而不需要额外的假设。我们使用基于基准潜在结果的标准分类指标测量决策者做出正确决策的能力。我们考虑了一个单盲实验设计，在这个设计中，提供AI生成的建议在不同案例中被随机分配给最终决策的人类。在这种实验设计下，我们展示了如何比较三种替代决策系统的性能--仅人类、人类与AI、仅AI。

    arXiv:2403.12108v1 Announce Type: new  Abstract: The use of Artificial Intelligence (AI) based on data-driven algorithms has become ubiquitous in today's society. Yet, in many cases and especially when stakes are high, humans still make final decisions. The critical question, therefore, is whether AI helps humans make better decisions as compared to a human alone or AI an alone. We introduce a new methodological framework that can be used to answer experimentally this question with no additional assumptions. We measure a decision maker's ability to make correct decisions using standard classification metrics based on the baseline potential outcome. We consider a single-blinded experimental design, in which the provision of AI-generated recommendations is randomized across cases with a human making final decisions. Under this experimental design, we show how to compare the performance of three alternative decision-making systems--human-alone, human-with-AI, and AI-alone. We apply the pr
    
[^3]: 具有市场运营应用的生成式概率预测

    Generative Probabilistic Forecasting with Applications in Market Operations

    [https://arxiv.org/abs/2403.05743](https://arxiv.org/abs/2403.05743)

    提出了一种基于Wiener-Kallianpur创新表示的生成式概率预测方法，包括自编码器和新颖的深度学习算法，具有渐近最优性和结构收敛性质，适用于实时市场运营中的高动态和波动时间序列。

    

    本文提出了一种新颖的生成式概率预测方法，该方法源自于非参数时间序列的Wiener-Kallianpur创新表示。在生成人工智能的范式下，所提出的预测架构包括一个自编码器，将非参数多变量随机过程转化为规范的创新序列，从中根据过去样本生成未来时间序列样本，条件是它们的概率分布取决于过去样本。提出了一种新的深度学习算法，将潜在过程限制为具有匹配自编码器输入-输出条件概率分布的独立同分布序列。建立了所提出的生成式预测方法的渐近最优性和结构收敛性质。该方法在实时市场运营中涉及高度动态和波动时间序列的三个应用方面。

    arXiv:2403.05743v1 Announce Type: cross  Abstract: This paper presents a novel generative probabilistic forecasting approach derived from the Wiener-Kallianpur innovation representation of nonparametric time series. Under the paradigm of generative artificial intelligence, the proposed forecasting architecture includes an autoencoder that transforms nonparametric multivariate random processes into canonical innovation sequences, from which future time series samples are generated according to their probability distributions conditioned on past samples. A novel deep-learning algorithm is proposed that constrains the latent process to be an independent and identically distributed sequence with matching autoencoder input-output conditional probability distributions. Asymptotic optimality and structural convergence properties of the proposed generative forecasting approach are established. Three applications involving highly dynamic and volatile time series in real-time market operations a
    

