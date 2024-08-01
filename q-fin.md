# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Convergence of the deep BSDE method for stochastic control problems formulated through the stochastic maximum principle](https://arxiv.org/abs/2401.17472) | 本文研究了基于深度BSDE方法和随机最大原则的随机控制问题，提供了该方法的收敛结果，并展示了在高维问题中相比其他方法具有卓越性能。 |
| [^2] | [Dynamic Risk Measurement by EVT based on Stochastic Volatility models via MCMC.](http://arxiv.org/abs/2201.09434) | 本文介绍了一种基于随机波动模型和极值理论的动态风险测量模型，结合重尾分布和杠杆效应，能够更有效地避免金融风险。 |

# 详细

[^1]: 通过随机最大原则，基于深度BSDE方法的随机控制问题的收敛性研究

    Convergence of the deep BSDE method for stochastic control problems formulated through the stochastic maximum principle

    [https://arxiv.org/abs/2401.17472](https://arxiv.org/abs/2401.17472)

    本文研究了基于深度BSDE方法和随机最大原则的随机控制问题，提供了该方法的收敛结果，并展示了在高维问题中相比其他方法具有卓越性能。

    

    众所周知，随机控制的决策问题可以通过前向后向随机微分方程（FBSDE）来表述。最近，Ji等人（2022）提出了一种基于随机最大原则（SMP）的高效深度学习算法。本文提供了该深度SMP-BSDE算法的收敛结果，并将其性能与其他现有方法进行比较。通过采用类似于Han和Long（2020）的策略，我们推导出后验误差估计，并展示了总近似误差可以由损失函数值和离散化误差的值来限制。我们在高维随机控制问题的数值例子中展示了该算法在漂移控制和扩散控制的情况下，相比现有算法表现出的卓越性能。

    It is well-known that decision-making problems from stochastic control can be formulated by means of forward-backward stochastic differential equation (FBSDE). Recently, the authors of Ji et al. 2022 proposed an efficient deep learning-based algorithm which was based on the stochastic maximum principle (SMP). In this paper, we provide a convergence result for this deep SMP-BSDE algorithm and compare its performance with other existing methods. In particular, by adopting a similar strategy as in Han and Long 2020, we derive a posteriori error estimate, and show that the total approximation error can be bounded by the value of the loss functional and the discretization error. We present numerical examples for high-dimensional stochastic control problems, both in case of drift- and diffusion control, which showcase superior performance compared to existing algorithms.
    
[^2]: 通过MCMC基于随机波动模型的EVT动态风险测量

    Dynamic Risk Measurement by EVT based on Stochastic Volatility models via MCMC. (arXiv:2201.09434v4 [stat.AP] UPDATED)

    [http://arxiv.org/abs/2201.09434](http://arxiv.org/abs/2201.09434)

    本文介绍了一种基于随机波动模型和极值理论的动态风险测量模型，结合重尾分布和杠杆效应，能够更有效地避免金融风险。

    

    本文旨在描述金融市场回报和波动性的典型事实特征，并解决资产回报的尾部特征未被充分考虑的问题，以更有效地避免风险和生产性地管理股票市场风险。因此，本文将重尾分布和杠杆效应引入SV模型。然后，通过MCMC估算模型参数。随后，全面描述了金融市场回报的重尾分布，并将其与极值理论相结合，以拟合标准残差的尾部分布。其后，建立了一种新的金融风险测量模型，称为SV-EVT-VaR动态模型。通过使用每日的标普500指数和模拟回报，达成了实证结果，揭示了SV-EVT模型在外样本回测中可以优于其他模型。

    This paper aims to characterize the typical factual characteristics of financial market returns and volatility and address the problem that the tail characteristics of asset returns have been not sufficiently considered, as an attempt to more effectively avoid risks and productively manage stock market risks. Thus, in this paper, the fat-tailed distribution and the leverage effect are introduced into the SV model. Next, the model parameters are estimated through MCMC. Subsequently, the fat-tailed distribution of financial market returns is comprehensively characterized and then incorporated with extreme value theory to fit the tail distribution of standard residuals. Afterward, a new financial risk measurement model is built, which is termed the SV-EVT-VaR-based dynamic model. With the use of daily S&P 500 index and simulated returns, the empirical results are achieved, which reveal that the SV-EVT-based models can outperform other models for out-of-sample data in backtesting and depic
    

