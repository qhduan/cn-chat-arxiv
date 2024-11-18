# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Bi-level Sparse Group Regressions for Macroeconomic Forecasting](https://arxiv.org/abs/2404.02671) | 提出了基于贝叶斯双层稀疏组回归的机器学习方法，可以进行高维宏观经济预测，并且理论证明其具有最小极限速率的收缩性，能够恢复模型参数，支持集包含模型的支持集。 |
| [^2] | [Limited substitutability, relative price changes and the uplifting of public natural capital values.](http://arxiv.org/abs/2308.04400) | 本研究通过全球元分析得出，生态系统服务相对价格每年约为2.2%，用于公共项目评估和环境经济会计中的调整。 |
| [^3] | [Time-Varying Parameters as Ridge Regressions.](http://arxiv.org/abs/2009.00401) | 该论文提出了一种实际上是基于岭回归的时变参数模型，这比传统的状态空间方法计算更快，调整更容易，有助于研究经济结构性变化。 |

# 详细

[^1]: 基于贝叶斯双层稀疏组回归的宏观经济预测

    Bayesian Bi-level Sparse Group Regressions for Macroeconomic Forecasting

    [https://arxiv.org/abs/2404.02671](https://arxiv.org/abs/2404.02671)

    提出了基于贝叶斯双层稀疏组回归的机器学习方法，可以进行高维宏观经济预测，并且理论证明其具有最小极限速率的收缩性，能够恢复模型参数，支持集包含模型的支持集。

    

    我们提出了一种机器学习方法，在已知具有组结构的协变量的高维设置中进行最优宏观经济预测。我们的模型涵盖了许多时间序列、混合频率和未知非线性的预测设置。我们引入了时间序列计量经济学中的双层稀疏概念，即稀疏性在组水平和组内均成立，我们假设真实模型符合这一假设。我们提出了一种引起双层稀疏性的先验，相应的后验分布被证明以最小极限速率收缩，恢复模型参数，并且其支持集在渐近上包含模型的支持集。我们的理论允许组间相关性，而同一组中的预测变量可以通过强相关性以及共同特征和模式进行表征。通过全面展示有限样本的性能来说明。

    arXiv:2404.02671v1 Announce Type: new  Abstract: We propose a Machine Learning approach for optimal macroeconomic forecasting in a high-dimensional setting with covariates presenting a known group structure. Our model encompasses forecasting settings with many series, mixed frequencies, and unknown nonlinearities. We introduce in time-series econometrics the concept of bi-level sparsity, i.e. sparsity holds at both the group level and within groups, and we assume the true model satisfies this assumption. We propose a prior that induces bi-level sparsity, and the corresponding posterior distribution is demonstrated to contract at the minimax-optimal rate, recover the model parameters, and have a support that includes the support of the model asymptotically. Our theory allows for correlation between groups, while predictors in the same group can be characterized by strong covariation as well as common characteristics and patterns. Finite sample performance is illustrated through comprehe
    
[^2]: 有限的替代性、相对价格变动与公共自然资本价值的提升

    Limited substitutability, relative price changes and the uplifting of public natural capital values. (arXiv:2308.04400v1 [econ.GN])

    [http://arxiv.org/abs/2308.04400](http://arxiv.org/abs/2308.04400)

    本研究通过全球元分析得出，生态系统服务相对价格每年约为2.2%，用于公共项目评估和环境经济会计中的调整。

    

    随着全球经济的不断增长，生态系统服务往往停滞或减少。经济学理论已经揭示了如何将这种相对稀缺性的转变反映到公共项目评估和环境经济会计中，但缺乏实证证据来将理论付诸实践。为了估计可用于进行此类调整的生态系统服务相对价格变化，我们对环境价值评估研究进行了全球元分析，以推导出意愿支付收入弹性作为有限替代性程度的代理。基于749个收入-意愿支付对，我们估计意愿支付收入弹性约为0.78（95-CI：0.6至1.0）。将这些结果与生态系统服务相对稀缺性变化的全球数据集结合起来，我们估计生态系统服务相对价格每年约为2.2％。在对非木材林生态系统的自然资本估值中应用了这些结果。

    As the global economy continues to grow, ecosystem services tend to stagnate or degrow. Economic theory has shown how such shifts in relative scarcities can be reflected in the appraisal of public projects and environmental-economic accounting, but empirical evidence has been lacking to put the theory into practice. To estimate the relative price change in ecosystem services that can be used to make such adjustments, we perform a global meta-analysis of environmental valuation studies to derive income elasticities of willingness to pay (WTP) for ecosystem services as a proxy for the degree of limited substitutability. Based on 749 income-WTP pairs, we estimate an income elasticity of WTP of around 0.78 (95-CI: 0.6 to 1.0). Combining these results with a global data set on shifts in the relative scarcity of ecosystem services, we estimate relative price change of ecosystem services of around 2.2 percent per year. In an application to natural capital valuation of non-timber forest ecosys
    
[^3]: 使用岭回归法的时变参数模型

    Time-Varying Parameters as Ridge Regressions. (arXiv:2009.00401v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2009.00401](http://arxiv.org/abs/2009.00401)

    该论文提出了一种实际上是基于岭回归的时变参数模型，这比传统的状态空间方法计算更快，调整更容易，有助于研究经济结构性变化。

    

    时变参数模型(TVPs)经常被用于经济学中来捕捉结构性变化。我强调了一个被忽视的事实——这些实际上是岭回归。这使得计算、调整和实现比状态空间范式更容易。在高维情况下，解决等价的双重岭问题的计算非常快,关键的“时间变化量”通常是由交叉验证来调整的。使用两步回归岭回归来处理不断变化的波动性。我考虑了基于稀疏性(算法选择哪些参数变化, 哪些不变)和降低秩约束的扩展(变化与因子模型相关联)。为了展示这种方法的有用性, 我使用它来研究加拿大货币政策的演变, 并使用大规模时变局部投影估计约4600个TVPs, 这一任务完全可以利用这种新方法完成。

    Time-varying parameters (TVPs) models are frequently used in economics to capture structural change. I highlight a rather underutilized fact -- that these are actually ridge regressions. Instantly, this makes computations, tuning, and implementation much easier than in the state-space paradigm. Among other things, solving the equivalent dual ridge problem is computationally very fast even in high dimensions, and the crucial "amount of time variation" is tuned by cross-validation. Evolving volatility is dealt with using a two-step ridge regression. I consider extensions that incorporate sparsity (the algorithm selects which parameters vary and which do not) and reduced-rank restrictions (variation is tied to a factor model). To demonstrate the usefulness of the approach, I use it to study the evolution of monetary policy in Canada using large time-varying local projections. The application requires the estimation of about 4600 TVPs, a task well within the reach of the new method.
    

