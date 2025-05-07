# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [q-Learning in Continuous Time.](http://arxiv.org/abs/2207.00713) | 本文研究了连续时间下的q-Learning，通过引入小q函数作为一阶近似，研究了q-learning理论，应用于设计不同的演员-评论家算法。 |

# 详细

[^1]: 连续时间下的q-Learning

    q-Learning in Continuous Time. (arXiv:2207.00713v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.00713](http://arxiv.org/abs/2207.00713)

    本文研究了连续时间下的q-Learning，通过引入小q函数作为一阶近似，研究了q-learning理论，应用于设计不同的演员-评论家算法。

    

    我们研究了基于熵正则化的探索性扩散过程的Q-learning在连续时间下的应用。我们引入了“小q函数”作为大Q函数的一阶近似，研究了q函数的q-learning理论，并应用于设计不同的演员-评论家算法。

    We study the continuous-time counterpart of Q-learning for reinforcement learning (RL) under the entropy-regularized, exploratory diffusion process formulation introduced by Wang et al. (2020). As the conventional (big) Q-function collapses in continuous time, we consider its first-order approximation and coin the term ``(little) q-function". This function is related to the instantaneous advantage rate function as well as the Hamiltonian. We develop a ``q-learning" theory around the q-function that is independent of time discretization. Given a stochastic policy, we jointly characterize the associated q-function and value function by martingale conditions of certain stochastic processes, in both on-policy and off-policy settings. We then apply the theory to devise different actor-critic algorithms for solving underlying RL problems, depending on whether or not the density function of the Gibbs measure generated from the q-function can be computed explicitly. One of our algorithms inter
    

