# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Reasoning for Physics Informed Neural Networks.](http://arxiv.org/abs/2308.13222) | 本文提出了一种基于贝叶斯推理的物理信息神经网络方法（PINN）。该方法采用贝叶斯神经网络框架，通过计算证据来优化模型并解决不确定性问题。 |
| [^2] | [Estimation of an Order Book Dependent Hawkes Process for Large Datasets.](http://arxiv.org/abs/2307.09077) | 本研究提出了一种用于高频交易事件到达的点过程，其中强度是Hawkes过程和委托簿派生的高维协变量函数的乘积。算法可以在存在数十亿数据点的情况下进行估计，并证明了其收敛性和一致性。样本外测试结果显示，捕捉委托簿信息的非线性特征对于高频交易的自激性特征有价值。 |

# 详细

[^1]: 用于物理信息神经网络的贝叶斯推理

    Bayesian Reasoning for Physics Informed Neural Networks. (arXiv:2308.13222v1 [physics.comp-ph])

    [http://arxiv.org/abs/2308.13222](http://arxiv.org/abs/2308.13222)

    本文提出了一种基于贝叶斯推理的物理信息神经网络方法（PINN）。该方法采用贝叶斯神经网络框架，通过计算证据来优化模型并解决不确定性问题。

    

    本文提出了一种基于贝叶斯公式的物理信息神经网络（PINN）方法。我们采用了MacKay在Neural Computation（1992年）中提出的贝叶斯神经网络框架。通过拉普拉斯近似法，得到后验密度。对于每个模型（拟合），计算所谓的证据。它是一种分类假设的度量。最优解具有最大的证据值。贝叶斯框架使我们能够控制边界对总损失的影响。事实上，贝叶斯算法通过微调损失组件的相对权重。我们解决了热力学、波动和Burger方程。所得结果与精确解基本一致。所有解都提供了在贝叶斯框架内计算的不确定性。

    Physics informed neural network (PINN) approach in Bayesian formulation is presented. We adopt the Bayesian neural network framework formulated by MacKay (Neural Computation 4 (3) (1992) 448). The posterior densities are obtained from Laplace approximation. For each model (fit), the so-called evidence is computed. It is a measure that classifies the hypothesis. The most optimal solution has the maximal value of the evidence. The Bayesian framework allows us to control the impact of the boundary contribution to the total loss. Indeed, the relative weights of loss components are fine-tuned by the Bayesian algorithm. We solve heat, wave, and Burger's equations. The obtained results are in good agreement with the exact solutions. All solutions are provided with the uncertainties computed within the Bayesian framework.
    
[^2]: 《大数据集上基于委托簿相关Hawkes过程的估计》

    Estimation of an Order Book Dependent Hawkes Process for Large Datasets. (arXiv:2307.09077v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.09077](http://arxiv.org/abs/2307.09077)

    本研究提出了一种用于高频交易事件到达的点过程，其中强度是Hawkes过程和委托簿派生的高维协变量函数的乘积。算法可以在存在数十亿数据点的情况下进行估计，并证明了其收敛性和一致性。样本外测试结果显示，捕捉委托簿信息的非线性特征对于高频交易的自激性特征有价值。

    

    本研究介绍了一种用于高频交易事件到达的点过程。强度是Hawkes过程和委托簿派生的高维协变量函数的乘积。讨论了该过程稳定性的条件。并提出了一种算法，即使在存在数十亿数据点的情况下，也可以进行模型估计，可能需要将协变量映射到高维空间。大样本量是常见于使用多个流动工具的高频数据应用中的情况。证明了算法的收敛性，建立了在弱条件下的一致性结果，并提出了一种测试统计量来评估不同模型规范的样本外表现。将该方法应用于纽约证券交易所（NYSE）上交易的四只股票的研究中。样本外测试过程表明，捕捉委托簿信息的非线性特征对于高频交易的自激性特征有价值。

    A point process for event arrivals in high frequency trading is presented. The intensity is the product of a Hawkes process and high dimensional functions of covariates derived from the order book. Conditions for stationarity of the process are stated. An algorithm is presented to estimate the model even in the presence of billions of data points, possibly mapping covariates into a high dimensional space. The large sample size can be common for high frequency data applications using multiple liquid instruments. Convergence of the algorithm is shown, consistency results under weak conditions is established, and a test statistic to assess out of sample performance of different model specifications is suggested. The methodology is applied to the study of four stocks that trade on the New York Stock Exchange (NYSE). The out of sample testing procedure suggests that capturing the nonlinearity of the order book information adds value to the self exciting nature of high frequency trading even
    

