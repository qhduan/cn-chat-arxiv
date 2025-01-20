# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic gradient descent for streaming linear and rectified linear systems with Massart noise](https://arxiv.org/abs/2403.01204) | 我们提出了一种针对具有Massart噪声的线性和ReLU回归问题的随机梯度下降方法，具有新颖的近乎线性收敛保证，首次在流式设置中为鲁棒ReLU回归提供了收敛保证，并展示了其相比于以前的方法有改进的收敛速率。 |
| [^2] | [Treatment Effect Estimation with Observational Network Data using Machine Learning.](http://arxiv.org/abs/2206.14591) | 该论文开发了增广逆概率加权（AIPW）方法，用于使用观测网络数据估计和推断具有溢出效应的治疗的直接效应。方法使用机器学习和样本分割，得到收敛速度较快且服从高斯分布的半参数治疗效果估计器。研究发现，在考虑学生社交网络的情况下，学习时间对考试成绩有影响。 |

# 详细

[^1]: 具有Massart噪声的流式线性和修正线性系统的随机梯度下降

    Stochastic gradient descent for streaming linear and rectified linear systems with Massart noise

    [https://arxiv.org/abs/2403.01204](https://arxiv.org/abs/2403.01204)

    我们提出了一种针对具有Massart噪声的线性和ReLU回归问题的随机梯度下降方法，具有新颖的近乎线性收敛保证，首次在流式设置中为鲁棒ReLU回归提供了收敛保证，并展示了其相比于以前的方法有改进的收敛速率。

    

    我们提出了SGD-exp，一种用于线性和ReLU回归的随机梯度下降方法，在Massart噪声（对抗性半随机破坏模型）下，完全流式设置下。我们展示了SGD-exp对真实参数的近乎线性收敛保证，最高可达50%的Massart破坏率，在对称无忧破坏情况下，任意破坏率也有保证。这是流式设置中鲁棒ReLU回归的第一个收敛保证结果，它显示了相比于以前的鲁棒方法对于L1线性回归具有改进的收敛速率，这是由于选择了指数衰减步长，这在实践中已被证明是有效的。我们的分析基于离散随机过程的漂移分析，这本身也可能是有趣的。

    arXiv:2403.01204v1 Announce Type: new  Abstract: We propose SGD-exp, a stochastic gradient descent approach for linear and ReLU regressions under Massart noise (adversarial semi-random corruption model) for the fully streaming setting. We show novel nearly linear convergence guarantees of SGD-exp to the true parameter with up to $50\%$ Massart corruption rate, and with any corruption rate in the case of symmetric oblivious corruptions. This is the first convergence guarantee result for robust ReLU regression in the streaming setting, and it shows the improved convergence rate over previous robust methods for $L_1$ linear regression due to a choice of an exponentially decaying step size, known for its efficiency in practice. Our analysis is based on the drift analysis of a discrete stochastic process, which could also be interesting on its own.
    
[^2]: 使用机器学习处理观测网络数据的治疗效果估计

    Treatment Effect Estimation with Observational Network Data using Machine Learning. (arXiv:2206.14591v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2206.14591](http://arxiv.org/abs/2206.14591)

    该论文开发了增广逆概率加权（AIPW）方法，用于使用观测网络数据估计和推断具有溢出效应的治疗的直接效应。方法使用机器学习和样本分割，得到收敛速度较快且服从高斯分布的半参数治疗效果估计器。研究发现，在考虑学生社交网络的情况下，学习时间对考试成绩有影响。

    

    因果推断方法通常假设独立单元来进行治疗效果估计。然而，这种假设经常是有问题的，因为单元之间可能会相互作用，导致单元之间的溢出效应。我们开发了增广逆概率加权（AIPW）方法，用于使用具有溢出效应的单个（社交）网络的观测数据对治疗的直接效应进行估计和推断。我们使用基于插件的机器学习和样本分割方法，得到一个半参数的治疗效果估计器，其渐近收敛于参数速率，并且在渐近情况下服从高斯分布。我们将AIPW方法应用于瑞士学生人生研究数据，以研究学习时间对考试成绩的影响，考虑到学生的社交网络。

    Causal inference methods for treatment effect estimation usually assume independent units. However, this assumption is often questionable because units may interact, resulting in spillover effects between units. We develop augmented inverse probability weighting (AIPW) for estimation and inference of the direct effect of the treatment with observational data from a single (social) network with spillover effects. We use plugin machine learning and sample splitting to obtain a semiparametric treatment effect estimator that converges at the parametric rate and asymptotically follows a Gaussian distribution. We apply our AIPW method to the Swiss StudentLife Study data to investigate the effect of hours spent studying on exam performance accounting for the students' social network.
    

