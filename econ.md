# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Strategic Model of Software Dependency Networks](https://arxiv.org/abs/2402.13375) | 通过研究软件包和库之间的依赖网络形成，发现程序员创建依赖时对其他程序员具有正的外部性作用，且倾向于链接到相同软件类型的热门包，但链接到其他类型的不太热门包。 |
| [^2] | [Threshold Regression in Heterogeneous Panel Data with Interactive Fixed Effects.](http://arxiv.org/abs/2308.04057) | 本文介绍了在面板数据阈值回归中引入了单位特定的异质性，并提出了一种新的参数估计方法，该方法可以更快地收敛。对费尔斯坦-宝里奥卡之谜的研究结果表明…… |
| [^3] | [A Robust Method for Microforecasting and Estimation of Random Effects.](http://arxiv.org/abs/2308.01596) | 我们提出了一种稳健的方法，用于在面板数据模型和增值模型中预测个体结果和估计随机效应，特别是在时间维度较短的情况下。该方法通过加权平均值结合了时序和汇总的预测/估计，并利用个体权重来提高预测的准确性。与现有方法相比，我们的方法不仅假设较少，而且可以在更弱的假设下获得良好的性能。该方法还避免了专制，通过让数据决定每个个体应借用多少力量。 |

# 详细

[^1]: 软件依赖网络的战略模型

    A Strategic Model of Software Dependency Networks

    [https://arxiv.org/abs/2402.13375](https://arxiv.org/abs/2402.13375)

    通过研究软件包和库之间的依赖网络形成，发现程序员创建依赖时对其他程序员具有正的外部性作用，且倾向于链接到相同软件类型的热门包，但链接到其他类型的不太热门包。

    

    现代软件开发涉及协作努力和重用现有代码，这降低了开发新软件的成本。然而，从现有包中重用代码会使程序员暴露于这些依赖关系的漏洞中。我们研究了软件包和库之间依赖网络的形成，该研究由一个具有可观察和不可观察异质性的网络形成结构模型指导。我们使用一种新颖的可扩展算法估算了 Rust 编程语言的 35,473 个代码库之间的 696,790 个有向依赖关系网络的成本、利益和链接外部性。我们发现，当程序员创建依赖关系时，会对其他程序员产生正的外部性作用。此外，我们展示了程序员倾向于链接到相同软件类型的更受欢迎的包，但链接到其他类型的不太受欢迎的包。我们采用传染病传播模型来衡量一个软件包的系统程度。

    arXiv:2402.13375v1 Announce Type: new  Abstract: Modern software development involves collaborative efforts and reuse of existing code, which reduces the cost of developing new software. However, reusing code from existing packages exposes coders to vulnerabilities in these dependencies. We study the formation of dependency networks among software packages and libraries, guided by a structural model of network formation with observable and unobservable heterogeneity. We estimate costs, benefits, and link externalities of the network of 696,790 directed dependencies between 35,473 repositories of the Rust programming language using a novel scalable algorithm. We find evidence of a positive externality exerted on other coders when coders create dependencies. Furthermore, we show that coders are likely to link to more popular packages of the same software type but less popular packages of other types. We adopt models for the spread of infectious diseases to measure a package's systemicnes
    
[^2]: 异质面板数据中的阈值回归与交互固定效应

    Threshold Regression in Heterogeneous Panel Data with Interactive Fixed Effects. (arXiv:2308.04057v1 [econ.EM])

    [http://arxiv.org/abs/2308.04057](http://arxiv.org/abs/2308.04057)

    本文介绍了在面板数据阈值回归中引入了单位特定的异质性，并提出了一种新的参数估计方法，该方法可以更快地收敛。对费尔斯坦-宝里奥卡之谜的研究结果表明……

    

    本文介绍了在面板数据阈值回归中引入了单位特定的异质性。斜率系数和阈值参数都允许因单位而异。异质阈值参数通过单位专有的经验分位数转换来表示，该转换由整个面板数据高效估计得出。在误差项中，面板数据的未观测异质性采用了交互固定效应的一般形式。这种新引入的参数异质性对模型的识别、估计、解释和渐近推断都有重要影响。假设阈值幅度收缩，现在意味着异质性的收缩，使得估计器的收敛速度比以前更快。我们推导了所提出估计器的渐近理论，蒙特卡洛模拟结果表明其在小样本中的有效性。我们应用新模型对费尔斯坦-宝里奥卡之谜进行了研究，发现……

    This paper introduces unit-specific heterogeneity in panel data threshold regression. Both slope coefficients and threshold parameters are allowed to vary by unit. The heterogeneous threshold parameters manifest via a unit-specific empirical quantile transformation of a common underlying threshold parameter which is estimated efficiently from the whole panel. In the errors, the unobserved heterogeneity of the panel takes the general form of interactive fixed effects. The newly introduced parameter heterogeneity has implications for model identification, estimation, interpretation, and asymptotic inference. The assumption of a shrinking threshold magnitude now implies shrinking heterogeneity and leads to faster estimator rates of convergence than previously encountered. The asymptotic theory for the proposed estimators is derived and Monte Carlo simulations demonstrate its usefulness in small samples. The new model is employed to examine the Feldstein-Horioka puzzle and it is found that
    
[^3]: 一种微观预测和随机效应估计的稳健方法

    A Robust Method for Microforecasting and Estimation of Random Effects. (arXiv:2308.01596v1 [econ.EM])

    [http://arxiv.org/abs/2308.01596](http://arxiv.org/abs/2308.01596)

    我们提出了一种稳健的方法，用于在面板数据模型和增值模型中预测个体结果和估计随机效应，特别是在时间维度较短的情况下。该方法通过加权平均值结合了时序和汇总的预测/估计，并利用个体权重来提高预测的准确性。与现有方法相比，我们的方法不仅假设较少，而且可以在更弱的假设下获得良好的性能。该方法还避免了专制，通过让数据决定每个个体应借用多少力量。

    

    我们提出了一种方法，在面板数据模型和增值模型中，对个体结果进行预测和估计随机效应，当面板数据的时间维度较短时。该方法稳健且易于实现，需要的假设很少。该方法的思想是，在时间序列信息的基础上，将时序和汇总的预测/估计的加权平均值，其中个体权重基于时间序列信息。我们展示了个体权重的预测最优性，无论是在最小化最大遗憾还是均方预测误差方面。然后，我们提供了可行的权重，以在弱于现有方法所需假设的情况下确保良好的性能。与现有的收缩方法不同，我们的方法利用了多数的优势，但避免了专制，通过针对个人（而不是群体）的准确性，并让数据决定每个个体应借用多少力量。与现有的经验贝叶斯方法不同，我们的频率主义方法不需要任何分布假设。

    We propose a method for forecasting individual outcomes and estimating random effects in linear panel data models and value-added models when the panel has a short time dimension. The method is robust, trivial to implement and requires minimal assumptions. The idea is to take a weighted average of time series- and pooled forecasts/estimators, with individual weights that are based on time series information. We show the forecast optimality of individual weights, both in terms of minimax-regret and of mean squared forecast error. We then provide feasible weights that ensure good performance under weaker assumptions than those required by existing approaches. Unlike existing shrinkage methods, our approach borrows the strength - but avoids the tyranny - of the majority, by targeting individual (instead of group) accuracy and letting the data decide how much strength each individual should borrow. Unlike existing empirical Bayesian methods, our frequentist approach requires no distributio
    

