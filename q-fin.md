# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Functional Limit Theorems for Hawkes Processes.](http://arxiv.org/abs/2401.11495) | Hawkes过程的长期行为由子事件的平均数量和离散程度决定。对于亚临界过程，提供了FLLNs和FCLTs，具体形式取决于子事件的离散程度。对于具有弱离散子事件的临界Hawkes过程，不存在功能中心极限定理。通过缩放后的强度过程和缩放后的Hawkes过程的分布与极限过程之间的Wasserstein距离的上界，给出了收敛速度。具有高度离散子事件的临界Hawkes过程与亚临界过程共享许多属性，功能极限定理成立。 |
| [^2] | [Continuous-Time q-learning for McKean-Vlasov Control Problems.](http://arxiv.org/abs/2306.16208) | 本文研究了连续时间q-learning在熵正则化强化学习框架下用于McKean-Vlasov控制问题，并揭示了两种不同的q函数的存在及其积分表示。 |
| [^3] | [Common Idiosyncratic Quantile Risk.](http://arxiv.org/abs/2208.14267) | 该论文提出了一种新的风险类型，称为共同的特异性分位数风险，其揭示了投资者如何定价正向和负向风险的新见解，对总体市场回报也具有预测能力。 |

# 详细

[^1]: Hawkes过程的功能极限定理

    Functional Limit Theorems for Hawkes Processes. (arXiv:2401.11495v1 [math.PR])

    [http://arxiv.org/abs/2401.11495](http://arxiv.org/abs/2401.11495)

    Hawkes过程的长期行为由子事件的平均数量和离散程度决定。对于亚临界过程，提供了FLLNs和FCLTs，具体形式取决于子事件的离散程度。对于具有弱离散子事件的临界Hawkes过程，不存在功能中心极限定理。通过缩放后的强度过程和缩放后的Hawkes过程的分布与极限过程之间的Wasserstein距离的上界，给出了收敛速度。具有高度离散子事件的临界Hawkes过程与亚临界过程共享许多属性，功能极限定理成立。

    

    我们证明Hawkes过程的长期行为完全由子事件的平均数量和离散程度确定。对于亚临界过程，我们在过程的核函数的最小条件下提供了FLLNs和FCLTs，极限定理的具体形式严重取决于子事件的离散程度。对于具有弱离散子事件的临界Hawkes过程，功能中心极限定理不成立。相反，我们证明了经过缩放的强度过程和经过缩放的Hawkes过程分别行为类似于无均值回归的CIR过程和整合的CIR过程。通过建立缩放的Hawkes过程的分布与相应极限过程之间的Wasserstein距离的上界，我们给出了收敛速度。相反，具有高度离散子事件的临界Hawkes过程与亚临界过程共享许多属性。特别是，功能极限定理成立。然而，与亚临界过程不同的是，亚临界过程没有离散子事件，具有强离散子事件的临界Hawkes过程没有功能中心极限定理。

    We prove that the long-run behavior of Hawkes processes is fully determined by the average number and the dispersion of child events. For subcritical processes we provide FLLNs and FCLTs under minimal conditions on the kernel of the process with the precise form of the limit theorems depending strongly on the dispersion of child events. For a critical Hawkes process with weakly dispersed child events, functional central limit theorems do not hold. Instead, we prove that the rescaled intensity processes and rescaled Hawkes processes behave like CIR-processes without mean-reversion, respectively integrated CIR-processes. We provide the rate of convergence by establishing an upper bound on the Wasserstein distance between the distributions of rescaled Hawkes process and the corresponding limit process. By contrast, critical Hawkes process with heavily dispersed child events share many properties of subcritical ones. In particular, functional limit theorems hold. However, unlike subcritica
    
[^2]: 连续时间q-learning用于McKean-Vlasov控制问题

    Continuous-Time q-learning for McKean-Vlasov Control Problems. (arXiv:2306.16208v1 [cs.LG])

    [http://arxiv.org/abs/2306.16208](http://arxiv.org/abs/2306.16208)

    本文研究了连续时间q-learning在熵正则化强化学习框架下用于McKean-Vlasov控制问题，并揭示了两种不同的q函数的存在及其积分表示。

    

    本文研究了q-learning，在熵正则化强化学习框架下，用于连续时间的McKean-Vlasov控制问题。与Jia和Zhou（2022c）的单个代理控制问题不同，代理之间的均场相互作用使得q函数的定义更加复杂，我们揭示了自然产生两种不同q函数的情况：（i）被称为集成q函数（用$q$表示），作为Gu、Guo、Wei和Xu（2023）引入的集成Q函数的一阶近似，可以通过涉及测试策略的弱鞅条件进行学习；（ii）作为策略改进迭代中所使用的实质q函数（用$q_e$表示）。我们证明了这两个q函数在所有测试策略下通过积分表示相关联。基于集成q函数的弱鞅条件和我们提出的搜索方法，我们设计了算法来学习两个q函数以解决Mckean-Vlasov控制问题。

    This paper studies the q-learning, recently coined as the continuous-time counterpart of Q-learning by Jia and Zhou (2022c), for continuous time Mckean-Vlasov control problems in the setting of entropy-regularized reinforcement learning. In contrast to the single agent's control problem in Jia and Zhou (2022c), the mean-field interaction of agents render the definition of q-function more subtle, for which we reveal that two distinct q-functions naturally arise: (i) the integrated q-function (denoted by $q$) as the first-order approximation of the integrated Q-function introduced in Gu, Guo, Wei and Xu (2023) that can be learnt by a weak martingale condition involving test policies; and (ii) the essential q-function (denoted by $q_e$) that is employed in the policy improvement iterations. We show that two q-functions are related via an integral representation under all test policies. Based on the weak martingale condition of the integrated q-function and our proposed searching method of
    
[^3]: 共同的特异性分位数风险

    Common Idiosyncratic Quantile Risk. (arXiv:2208.14267v2 [q-fin.GN] UPDATED)

    [http://arxiv.org/abs/2208.14267](http://arxiv.org/abs/2208.14267)

    该论文提出了一种新的风险类型，称为共同的特异性分位数风险，其揭示了投资者如何定价正向和负向风险的新见解，对总体市场回报也具有预测能力。

    

    我们发现了一种新的风险类型，其特征在于资产回报的横截面分位数具有共性。我们提出的新型分位数风险因子与特定分位数的风险溢酬相关，并揭示了投资者如何定价正向和负向风险的新见解。与以往文献相比，我们在不做混淆假设或汇总可能的非线性信息的情况下恢复了横截面分位数的共同结构。我们讨论了新的分位数风险因子与流行的波动率和下行风险因子的不同之处，并确定了哪些分位数依赖性风险应该得到更大的补偿。分位数因子也具有对总体市场回报的预测能力。

    We identify a new type of risk that is characterised by commonalities in the quantiles of the cross-sectional distribution of asset returns. Our newly proposed quantile risk factor is associated with a quantile-specific risk premium and provides new insights into how upside and downside risks are priced by investors. In contrast to the previous literature, we recover the common structure in cross-sectional quantiles without making confounding assumptions or aggregating potentially non-linear information. We discuss how the new quantile-based risk factor differs from popular volatility and downside risk factors, and we identify where the quantile-dependent risks deserve greater compensation. Quantile factors also have predictive power for aggregate market returns.
    

