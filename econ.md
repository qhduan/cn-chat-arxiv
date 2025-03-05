# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Heterogeneous Aggregate Valence Analysis (HAVAN) Model: A Flexible Approach to Modeling Unobserved Heterogeneity in Discrete Choice Analysis](https://arxiv.org/abs/2402.00184) | 本文引入了HAVAN模型，一种灵活的离散选择分析中未观测异质性建模方法。与传统模型不同，HAVAN模型直接刻画了备选项的异质偏好，并在模拟实验中表现出优于神经网络的预测性能。 |
| [^2] | [Bayesian Analysis of Linear Contracts.](http://arxiv.org/abs/2211.06850) | 本文在贝叶斯框架下为线性合同在实践中普遍存在的原因进行了解释和证明，并表明在线性合同中，当委托-代理环境中存在足够不确定性时，线性合同是近乎最优的。 |
| [^3] | [Smiles in Profiles: Improving Fairness and Efficiency Using Estimates of User Preferences in Online Marketplaces.](http://arxiv.org/abs/2209.01235) | 通过模拟不同特征的档案照片对出借人需求的影响，研究提出了实现线上市场公平和效率的方法。 |
| [^4] | [The Transfer Performance of Economic Models.](http://arxiv.org/abs/2202.04796) | 该论文提供了一个易于处理的“跨领域”预测问题，并根据模型在新领域中的表现定义了模型的迁移误差。该论文比较了经济模型和黑盒算法在预测确定等价值方面的迁移能力，发现经济模型在跨领域上的泛化能力更强。 |

# 详细

[^1]: HAVAN模型：一种灵活的离散选择分析中未观测异质性建模方法

    The Heterogeneous Aggregate Valence Analysis (HAVAN) Model: A Flexible Approach to Modeling Unobserved Heterogeneity in Discrete Choice Analysis

    [https://arxiv.org/abs/2402.00184](https://arxiv.org/abs/2402.00184)

    本文引入了HAVAN模型，一种灵活的离散选择分析中未观测异质性建模方法。与传统模型不同，HAVAN模型直接刻画了备选项的异质偏好，并在模拟实验中表现出优于神经网络的预测性能。

    

    本文介绍了一种新型的离散选择模型，名为HAVAN模型（Heterogeneous Aggregate Valence Analysis）。我们采用“valence”这个术语来表示用于建模消费者决策的任何潜在量（如效用、后悔等）。与传统的在各个产品属性之间参数化异质偏好的模型不同，HAVAN模型直接刻画了特定备选的异质偏好。这种对消费者异质性的创新视角提供了前所未有的灵活性，并显著减少了混合logit模型中常见的模拟负担。在一个模拟实验中，HAVAN模型显示出优于最先进的人工神经网络的预测性能。这一发现强调了HAVAN模型改进离散选择建模能力的潜力。

    This paper introduces the Heterogeneous Aggregate Valence Analysis (HAVAN) model, a novel class of discrete choice models. We adopt the term "valence'' to encompass any latent quantity used to model consumer decision-making (e.g., utility, regret, etc.). Diverging from traditional models that parameterize heterogeneous preferences across various product attributes, HAVAN models (pronounced "haven") instead directly characterize alternative-specific heterogeneous preferences. This innovative perspective on consumer heterogeneity affords unprecedented flexibility and significantly reduces simulation burdens commonly associated with mixed logit models. In a simulation experiment, the HAVAN model demonstrates superior predictive performance compared to state-of-the-art artificial neural networks. This finding underscores the potential for HAVAN models to improve discrete choice modeling capabilities.
    
[^2]: 线性合同的贝叶斯分析

    Bayesian Analysis of Linear Contracts. (arXiv:2211.06850v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2211.06850](http://arxiv.org/abs/2211.06850)

    本文在贝叶斯框架下为线性合同在实践中普遍存在的原因进行了解释和证明，并表明在线性合同中，当委托-代理环境中存在足够不确定性时，线性合同是近乎最优的。

    

    我们在贝叶斯框架下为实践中普遍存在的线性（佣金制）合同提供了合理性的证明。我们考虑了一个隐藏行动的委托-代理模型，在该模型中，不同行动需要不同的努力量，并且代理人的努力成本是私有的。我们展示了当在委托-代理环境中存在足够的不确定性时，线性合同是近乎最优的。

    We provide a justification for the prevalence of linear (commission-based) contracts in practice under the Bayesian framework. We consider a hidden-action principal-agent model, in which actions require different amounts of effort, and the agent's cost per-unit-of-effort is private. We show that linear contracts are near-optimal whenever there is sufficient uncertainty in the principal-agent setting.
    
[^3]: 档案中的微笑：利用用户对线上市场的偏好估计提高公平性和效率性

    Smiles in Profiles: Improving Fairness and Efficiency Using Estimates of User Preferences in Online Marketplaces. (arXiv:2209.01235v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2209.01235](http://arxiv.org/abs/2209.01235)

    通过模拟不同特征的档案照片对出借人需求的影响，研究提出了实现线上市场公平和效率的方法。

    

    在线平台经常面临公平（即非歧视性）和效率（即最大化收益）之间的挑战。使用计算机视觉算法和微型借贷市场的观察数据，我们发现，借款人在创建在线档案时所做的选择影响着这两个目标。我们进一步通过网络随机调查实验支持了这一结论。在实验中，我们使用生成对抗网络创建具有不同特征的档案图片，并估计其对出借人需求的影响。我们然后反事实地评估替代平台政策，并确定特定方法来影响可改变的档案照片特征，以缓解公平性与效率性之间的紧张关系。

    Online platforms often face challenges being both fair (i.e., non-discriminatory) and efficient (i.e., maximizing revenue). Using computer vision algorithms and observational data from a micro-lending marketplace, we find that choices made by borrowers creating online profiles impact both of these objectives. We further support this conclusion with a web-based randomized survey experiment. In the experiment, we create profile images using Generative Adversarial Networks that differ in a specific feature and estimate its impact on lender demand. We then counterfactually evaluate alternative platform policies and identify particular approaches to influencing the changeable profile photo features that can ameliorate the fairness-efficiency tension.
    
[^4]: 经济模型的迁移表现

    The Transfer Performance of Economic Models. (arXiv:2202.04796v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2202.04796](http://arxiv.org/abs/2202.04796)

    该论文提供了一个易于处理的“跨领域”预测问题，并根据模型在新领域中的表现定义了模型的迁移误差。该论文比较了经济模型和黑盒算法在预测确定等价值方面的迁移能力，发现经济模型在跨领域上的泛化能力更强。

    

    经济学家经常使用特定领域的数据来估计模型，例如在特定对象组中估计风险偏好或特定彩票类别上估计风险偏好。模型预测是否跨领域适用取决于估计模型是否捕捉到可推广结构。我们提供了一个易于处理的“跨领域”预测问题，并根据模型在新领域中的表现定义了模型的迁移误差。我们推导出有限样本预测区间，当领域独立同分布时，保证以用户选择的概率包含实现的迁移误差，并使用这些区间比较经济模型和黑盒算法在预测确定等价值方面的迁移能力。我们发现在这个应用程序中，我们考虑的黑盒算法在相同领域的数据上估计和测试时优于标准经济模型，但经济模型在跨领域上的泛化能力更强。

    Economists often estimate models using data from a particular domain, e.g. estimating risk preferences in a particular subject pool or for a specific class of lotteries. Whether a model's predictions extrapolate well across domains depends on whether the estimated model has captured generalizable structure. We provide a tractable formulation for this "out-of-domain" prediction problem and define the transfer error of a model based on how well it performs on data from a new domain. We derive finite-sample forecast intervals that are guaranteed to cover realized transfer errors with a user-selected probability when domains are iid, and use these intervals to compare the transferability of economic models and black box algorithms for predicting certainty equivalents. We find that in this application, the black box algorithms we consider outperform standard economic models when estimated and tested on data from the same domain, but the economic models generalize across domains better than 
    

