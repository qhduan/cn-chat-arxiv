# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bootstrapping Fisher Market Equilibrium and First-Price Pacing Equilibrium](https://arxiv.org/abs/2402.02303) | 本文引入和设计了几个对线性费舍尔市场（LFM）和一手价格竞价均衡（FPPE）进行统计有效自举推断的过程，其中最具挑战性的是对一般FPPE进行自举。通过使用环收敛理论的有力工具，在温和的退化条件下成功设计了FPPE的自举过程。 |
| [^2] | [Semidiscrete optimal transport with unknown costs.](http://arxiv.org/abs/2310.00786) | 本文研究了具有未知成本的半离散最优输运问题，提出了一种采用在线学习和随机逼近相结合的半局部算法，并证明其具有最优的收敛速度。 |
| [^3] | [Startup Acquisitions: Acquihires and Talent Hoarding.](http://arxiv.org/abs/2308.10046) | 该论文提出了一个创业公司收购模型，指出收购会导致低效的 "人才囤积"。研究发现，即使没有竞争效应，收购也可能是垄断行为的结果，导致人才分配低效，并对消费者剩余和被收购员工的工作稳定性产生负面影响。 |

# 详细

[^1]: 自举费舍尔市场均衡和一手价格竞价均衡

    Bootstrapping Fisher Market Equilibrium and First-Price Pacing Equilibrium

    [https://arxiv.org/abs/2402.02303](https://arxiv.org/abs/2402.02303)

    本文引入和设计了几个对线性费舍尔市场（LFM）和一手价格竞价均衡（FPPE）进行统计有效自举推断的过程，其中最具挑战性的是对一般FPPE进行自举。通过使用环收敛理论的有力工具，在温和的退化条件下成功设计了FPPE的自举过程。

    

    线性费舍尔市场（LFM）是经济学中的一个基本均衡模型，也在公平和高效的资源分配方面有应用。一手价格竞价均衡（FPPE）是一种捕捉第一价格拍卖中预算管理机制的模型。在某些实际情况下，如广告拍卖，在这些模型上进行统计推断具有一定的兴趣。一种广泛应用于一般统计推断的常用方法是自举过程。然而，对于LFM和FPPE，目前不存在有效应用自举程序的理论。在本文中，我们引入并设计了几个对LFM和FPPE进行统计有效自举推断的过程。最具挑战性的部分是对一般FPPE进行自举，这归结为自举约束M-估计量，这是一个很大程度上未开发的问题。我们通过使用环收敛理论的有力工具，在温和的退化条件下为FPPE设计了一个自举过程。通过合成和实际数据的实验证明了我们的方法的有效性。

    The linear Fisher market (LFM) is a basic equilibrium model from economics, which also has applications in fair and efficient resource allocation. First-price pacing equilibrium (FPPE) is a model capturing budget-management mechanisms in first-price auctions. In certain practical settings such as advertising auctions, there is an interest in performing statistical inference over these models. A popular methodology for general statistical inference is the bootstrap procedure. Yet, for LFM and FPPE there is no existing theory for the valid application of bootstrap procedures. In this paper, we introduce and devise several statistically valid bootstrap inference procedures for LFM and FPPE. The most challenging part is to bootstrap general FPPE, which reduces to bootstrapping constrained M-estimators, a largely unexplored problem. We devise a bootstrap procedure for FPPE under mild degeneracy conditions by using the powerful tool of epi-convergence theory. Experiments with synthetic and s
    
[^2]: 具有未知成本的半离散最优输运

    Semidiscrete optimal transport with unknown costs. (arXiv:2310.00786v1 [econ.EM])

    [http://arxiv.org/abs/2310.00786](http://arxiv.org/abs/2310.00786)

    本文研究了具有未知成本的半离散最优输运问题，提出了一种采用在线学习和随机逼近相结合的半局部算法，并证明其具有最优的收敛速度。

    

    半离散最优输运是线性规划中经典输运问题的一种有挑战性的推广。其目标是以固定边际分布的方式设计两个随机变量（一个连续，一个离散）的联合分布，以最小化期望成本。我们提出了这个问题的一个新型变体，其中成本函数是未知的，但可以通过噪声观测学习；然而，每次只能采样一个函数。我们开发了一种半局部算法，将在线学习与随机逼近相结合，并证明其实现了最优的收敛速度，尽管随机梯度的非光滑性和目标函数的缺乏强凹性。

    Semidiscrete optimal transport is a challenging generalization of the classical transportation problem in linear programming. The goal is to design a joint distribution for two random variables (one continuous, one discrete) with fixed marginals, in a way that minimizes expected cost. We formulate a novel variant of this problem in which the cost functions are unknown, but can be learned through noisy observations; however, only one function can be sampled at a time. We develop a semi-myopic algorithm that couples online learning with stochastic approximation, and prove that it achieves optimal convergence rates, despite the non-smoothness of the stochastic gradient and the lack of strong concavity in the objective function.
    
[^3]: 创业公司收购：人才抢购和人才囤积

    Startup Acquisitions: Acquihires and Talent Hoarding. (arXiv:2308.10046v1 [econ.GN])

    [http://arxiv.org/abs/2308.10046](http://arxiv.org/abs/2308.10046)

    该论文提出了一个创业公司收购模型，指出收购会导致低效的 "人才囤积"。研究发现，即使没有竞争效应，收购也可能是垄断行为的结果，导致人才分配低效，并对消费者剩余和被收购员工的工作稳定性产生负面影响。

    

    我们提出了一个创业公司收购模型，可能导致低效的 "人才囤积"。我们开发了一个有两个竞争公司的模型，这些公司可以收购和整合一个在不同领域运营的创业公司，这种收购改善了收购公司的竞争力。我们表明，即使没有经典的竞争效应，这种收购也可能不是良性的，而是垄断行为的结果，导致人才分配低效。此外，我们还表明，这种人才囤积可能会降低消费者剩余，并导致被收购员工的工作不稳定性增加。

    We present a model of startup acquisitions, which may give rise to inefficient "talent hoarding." We develop a model with two competing firms that can acquire and integrate (or "acquihire") a startup operating in an orthogonal market. Such an acquihire improves the competitiveness of the acquiring firm. We show that even absent the classical competition effects, acquihires need not be benign but can be the result of oligopolistic behavior, leading to an inefficient allocation of talent. Further, we show that such talent hoarding may reduce consumer surplus and lead to more job volatility for acquihired employees.
    

