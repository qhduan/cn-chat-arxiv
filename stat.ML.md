# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PCL-Indexability and Whittle Index for Restless Bandits with General Observation Models.](http://arxiv.org/abs/2307.03034) | 本文研究了一种一般观测模型下的不安定多臂赌博机问题，提出了PCL-可索引性和Whittle索引的分析方法，并通过近似过程将问题转化为有限状态问题。数值实验表明算法表现优秀。 |
| [^2] | [Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series.](http://arxiv.org/abs/2304.03069) | 本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。 |

# 详细

[^1]: 带有一般观测模型的不安定赌博机问题的PCL-可索引性和Whittle索引

    PCL-Indexability and Whittle Index for Restless Bandits with General Observation Models. (arXiv:2307.03034v1 [stat.ML])

    [http://arxiv.org/abs/2307.03034](http://arxiv.org/abs/2307.03034)

    本文研究了一种一般观测模型下的不安定多臂赌博机问题，提出了PCL-可索引性和Whittle索引的分析方法，并通过近似过程将问题转化为有限状态问题。数值实验表明算法表现优秀。

    

    本文考虑了一种一般观测模型，用于不安定多臂赌博机问题。由于资源约束或环境或固有噪声，玩家操作需要基于某种有误差的反馈机制。通过建立反馈/观测动力学的一般概率模型，我们将问题表述为一个从任意初始信念（先验信息）开始的具有可数信念状态空间的不安定赌博机问题。我们利用具有部分守恒定律（PCL）的可实现区域方法，分析了无限状态问题的可索引性和优先级索引（Whittle索引）。最后，我们提出了一个近似过程，将问题转化为可以应用Niño-Mora和Bertsimas针对有限状态问题的AG算法的问题。数值实验表明，我们的算法具有出色的性能。

    In this paper, we consider a general observation model for restless multi-armed bandit problems. The operation of the player needs to be based on certain feedback mechanism that is error-prone due to resource constraints or environmental or intrinsic noises. By establishing a general probabilistic model for dynamics of feedback/observation, we formulate the problem as a restless bandit with a countable belief state space starting from an arbitrary initial belief (a priori information). We apply the achievable region method with partial conservation law (PCL) to the infinite-state problem and analyze its indexability and priority index (Whittle index). Finally, we propose an approximation process to transform the problem into which the AG algorithm of Ni\~no-Mora and Bertsimas for finite-state problems can be applied to. Numerical experiments show that our algorithm has an excellent performance.
    
[^2]: 自适应学生t分布与方法矩移动估计器用于非平稳时间序列

    Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series. (arXiv:2304.03069v1 [stat.ME])

    [http://arxiv.org/abs/2304.03069](http://arxiv.org/abs/2304.03069)

    本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。

    

    真实的时间序列通常是非平稳的，这带来了模型适应的难题。传统方法如GARCH假定任意类型的依赖性。为了避免这种偏差，我们将着眼于最近提出的不可知的移动估计器哲学：在时间$t$找到优化$F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$移动对数似然的参数，随时间演化。例如，它允许使用廉价的指数移动平均值（EMA）来估计参数，例如绝对中心矩$E[|x-\mu|^p]$随$p\in\mathbb{R}^+$的变化而演化$m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$。这种基于方法的一般自适应矩的应用将呈现在学生t分布上，尤其是在经济应用中流行，这里应用于DJIA公司的对数收益率。

    The real life time series are usually nonstationary, bringing a difficult question of model adaptation. Classical approaches like GARCH assume arbitrary type of dependence. To prevent such bias, we will focus on recently proposed agnostic philosophy of moving estimator: in time $t$ finding parameters optimizing e.g. $F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$ moving log-likelihood, evolving in time. It allows for example to estimate parameters using inexpensive exponential moving averages (EMA), like absolute central moments $E[|x-\mu|^p]$ evolving with $m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$ for one or multiple powers $p\in\mathbb{R}^+$. Application of such general adaptive methods of moments will be presented on Student's t-distribution, popular especially in economical applications, here applied to log-returns of DJIA companies.
    

