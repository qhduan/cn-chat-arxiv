# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tactical Decision Making for Autonomous Trucks by Deep Reinforcement Learning with Total Cost of Operation Based Reward](https://arxiv.org/abs/2403.06524) | 通过基于全面运营成本的奖励函数，采用深度强化学习框架优化自主卡车的战术决策，将高级决策与低级控制分离，并采用不同技巧提升性能。 |
| [^2] | [On the Convergence of Differentially-Private Fine-tuning: To Linearly Probe or to Fully Fine-tune?](https://arxiv.org/abs/2402.18905) | 本文分析了差分隐私线性探测（LP）和完全微调（FT）的训练动态，探索了从线性探测过渡到完全微调（LP-FT）的顺序微调现象及其对测试损失的影响，提供了关于在超参数化神经网络中差分隐私微调收敛性的理论洞见和隐私预算分配的效用曲线。 |
| [^3] | [Synthetic Control Methods by Density Matching under Implicit Endogeneitiy.](http://arxiv.org/abs/2307.11127) | 本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。 |
| [^4] | [Non-stationary Online Convex Optimization with Arbitrary Delays.](http://arxiv.org/abs/2305.12131) | 本文研究了任意时延的非稳态在线凸优化，提出了一种简单的算法DOGD，并证明它能在最坏情况下获得$O(\sqrt{dT}(P_T+1))$的动态遗憾界，同时当延迟不改变梯度到达顺序时，自动将动态遗憾减少到$O(\sqrt{S}(1+P_T))$。 |

# 详细

[^1]: 基于全面运营成本奖励的深度强化学习自主卡车战术决策

    Tactical Decision Making for Autonomous Trucks by Deep Reinforcement Learning with Total Cost of Operation Based Reward

    [https://arxiv.org/abs/2403.06524](https://arxiv.org/abs/2403.06524)

    通过基于全面运营成本的奖励函数，采用深度强化学习框架优化自主卡车的战术决策，将高级决策与低级控制分离，并采用不同技巧提升性能。

    

    我们为自主卡车的战术决策制定了一个基于深度强化学习的框架，特别是针对高速公路场景中的自适应巡航控制（ACC）和变道动作。我们的研究结果表明，在强化学习代理和基于物理模型的低级控制器之间分离高级决策过程和低级控制动作是有益的。接下来，我们研究了通过在卡车的全面运营成本（TCOP）为基础的多目标奖励函数优化性能的不同方法；通过为奖励分量添加权重，通过对奖励分量进行归一化，以及使用课程学习技术。

    arXiv:2403.06524v1 Announce Type: cross  Abstract: We develop a deep reinforcement learning framework for tactical decision making in an autonomous truck, specifically for Adaptive Cruise Control (ACC) and lane change maneuvers in a highway scenario. Our results demonstrate that it is beneficial to separate high-level decision-making processes and low-level control actions between the reinforcement learning agent and the low-level controllers based on physical models. In the following, we study optimizing the performance with a realistic and multi-objective reward function based on Total Cost of Operation (TCOP) of the truck using different approaches; by adding weights to reward components, by normalizing the reward components and by using curriculum learning techniques.
    
[^2]: 论差分隐私微调的收敛性：应线性探测还是完全微调？

    On the Convergence of Differentially-Private Fine-tuning: To Linearly Probe or to Fully Fine-tune?

    [https://arxiv.org/abs/2402.18905](https://arxiv.org/abs/2402.18905)

    本文分析了差分隐私线性探测（LP）和完全微调（FT）的训练动态，探索了从线性探测过渡到完全微调（LP-FT）的顺序微调现象及其对测试损失的影响，提供了关于在超参数化神经网络中差分隐私微调收敛性的理论洞见和隐私预算分配的效用曲线。

    

    差分隐私（DP）机器学习流水线通常包括两个阶段的过程：在公共数据集上进行非私有预训练，然后使用DP优化技术在私有数据上进行微调。在DP设置中，已经观察到完全微调有时候并不总是产生最佳的测试准确度，即使对于分布内数据也是如此。本文（1）分析了DP线性探测（LP）和完全微调（FT）的训练动态，以及（2）探索了顺序微调的现象，从线性探测开始，过渡到完全微调（LP-FT），以及它对测试损失的影响。我们提供了有关DP微调在超参数化神经网络中的收敛性的理论洞见，并建立了一个确定隐私预算在线性探测和完全微调之间分配的效用曲线。理论结果得到了对各种基准和模型的经验评估支持。

    arXiv:2402.18905v1 Announce Type: cross  Abstract: Differentially private (DP) machine learning pipelines typically involve a two-phase process: non-private pre-training on a public dataset, followed by fine-tuning on private data using DP optimization techniques. In the DP setting, it has been observed that full fine-tuning may not always yield the best test accuracy, even for in-distribution data. This paper (1) analyzes the training dynamics of DP linear probing (LP) and full fine-tuning (FT), and (2) explores the phenomenon of sequential fine-tuning, starting with linear probing and transitioning to full fine-tuning (LP-FT), and its impact on test loss. We provide theoretical insights into the convergence of DP fine-tuning within an overparameterized neural network and establish a utility curve that determines the allocation of privacy budget between linear probing and full fine-tuning. The theoretical results are supported by empirical evaluations on various benchmarks and models.
    
[^3]: 通过密度匹配实现的合成对照方法下的隐式内生性问题

    Synthetic Control Methods by Density Matching under Implicit Endogeneitiy. (arXiv:2307.11127v1 [econ.EM])

    [http://arxiv.org/abs/2307.11127](http://arxiv.org/abs/2307.11127)

    本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。

    

    合成对照方法（SCMs）已成为比较案例研究中因果推断的重要工具。SCMs的基本思想是通过使用来自未处理单元的观测结果的加权和来估计经过处理单元的反事实结果。合成对照（SC）的准确性对于估计因果效应至关重要，因此，SC权重的估计成为了研究的焦点。在本文中，我们首先指出现有的SCMs存在一个隐式内生性问题，即未处理单元的结果与反事实结果模型中的误差项之间的相关性。我们展示了这个问题会对因果效应估计器产生偏差。然后，我们提出了一种基于密度匹配的新型SCM，假设经过处理单元的结果密度可以用未处理单元的密度的加权平均来近似（即混合模型）。基于这一假设，我们通过匹配来估计SC权重。

    Synthetic control methods (SCMs) have become a crucial tool for causal inference in comparative case studies. The fundamental idea of SCMs is to estimate counterfactual outcomes for a treated unit by using a weighted sum of observed outcomes from untreated units. The accuracy of the synthetic control (SC) is critical for estimating the causal effect, and hence, the estimation of SC weights has been the focus of much research. In this paper, we first point out that existing SCMs suffer from an implicit endogeneity problem, which is the correlation between the outcomes of untreated units and the error term in the model of a counterfactual outcome. We show that this problem yields a bias in the causal effect estimator. We then propose a novel SCM based on density matching, assuming that the density of outcomes of the treated unit can be approximated by a weighted average of the densities of untreated units (i.e., a mixture model). Based on this assumption, we estimate SC weights by matchi
    
[^4]: 任意时延的非稳态在线凸优化

    Non-stationary Online Convex Optimization with Arbitrary Delays. (arXiv:2305.12131v1 [cs.LG])

    [http://arxiv.org/abs/2305.12131](http://arxiv.org/abs/2305.12131)

    本文研究了任意时延的非稳态在线凸优化，提出了一种简单的算法DOGD，并证明它能在最坏情况下获得$O(\sqrt{dT}(P_T+1))$的动态遗憾界，同时当延迟不改变梯度到达顺序时，自动将动态遗憾减少到$O(\sqrt{S}(1+P_T))$。

    

    最近，以梯度或其他函数信息可以任意延迟为特点的在线凸优化（OCO）引起了越来越多的关注。与之前研究稳态环境的研究不同，本文研究了非稳态环境下的延迟OCO，并旨在最小化与任何比较器序列相关的动态遗憾。为此，我们首先提出了一个简单的算法，即DOGD，该算法根据其到达顺序为每个延迟梯度执行渐变下降步骤。尽管它很简单，但我们的新型分析表明，DOGD可以在最坏情况下获得$O(\sqrt{dT}(P_T+1))$的动态遗憾界，其中$d$是最大延迟，$T$是时间跨度，$P_T$是比较器的路径长度。更重要的是，在延迟不改变渐变的到达顺序的情况下，它可以自动将动态遗憾减少到$O(\sqrt{S}(1+P_T))$，其中$S$是延迟之和。此外，我们将DOGD扩展为更通用的算法，并证明它实现了与DOGD相同的遗憾界。广泛的模拟表明了所提出算法的有效性和效率。

    Online convex optimization (OCO) with arbitrary delays, in which gradients or other information of functions could be arbitrarily delayed, has received increasing attention recently. Different from previous studies that focus on stationary environments, this paper investigates the delayed OCO in non-stationary environments, and aims to minimize the dynamic regret with respect to any sequence of comparators. To this end, we first propose a simple algorithm, namely DOGD, which performs a gradient descent step for each delayed gradient according to their arrival order. Despite its simplicity, our novel analysis shows that DOGD can attain an $O(\sqrt{dT}(P_T+1)$ dynamic regret bound in the worst case, where $d$ is the maximum delay, $T$ is the time horizon, and $P_T$ is the path length of comparators. More importantly, in case delays do not change the arrival order of gradients, it can automatically reduce the dynamic regret to $O(\sqrt{S}(1+P_T))$, where $S$ is the sum of delays. Furtherm
    

