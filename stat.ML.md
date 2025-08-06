# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Discovering group dynamics in synchronous time series via hierarchical recurrent switching-state models.](http://arxiv.org/abs/2401.14973) | 通过分层的循环切换状态模型，我们可以无监督地同时解释系统级和个体级的动态，从而更好地建模同步时间序列中的群体动态。 |

# 详细

[^1]: 通过分层循环切换状态模型发现同步时间序列中的群体动态

    Discovering group dynamics in synchronous time series via hierarchical recurrent switching-state models. (arXiv:2401.14973v1 [stat.ML])

    [http://arxiv.org/abs/2401.14973](http://arxiv.org/abs/2401.14973)

    通过分层的循环切换状态模型，我们可以无监督地同时解释系统级和个体级的动态，从而更好地建模同步时间序列中的群体动态。

    

    我们致力于对同一时间段内多个实体相互作用而产生的时间序列集合进行建模。最近的研究集中在建模个体时间序列方面对我们的预期应用是不足够的，其中集体系统级行为影响着个体实体的轨迹。为了解决这类问题，我们提出了一种新的分层切换状态模型，可以以无监督的方式训练，同时解释系统级和个体级的动态。我们采用了一个隐含的系统级离散状态马尔可夫链，驱动着隐含的实体级链，进而控制每个观测时间序列的动态。观测结果在实体和系统级的链之间进行反馈，通过依赖于上下文的状态转换来提高灵活性。我们的分层切换循环动力学模型可以通过封闭形式的变分坐标上升更新来学习，其在个体数量上呈线性扩展。

    We seek to model a collection of time series arising from multiple entities interacting over the same time period. Recent work focused on modeling individual time series is inadequate for our intended applications, where collective system-level behavior influences the trajectories of individual entities. To address such problems, we present a new hierarchical switching-state model that can be trained in an unsupervised fashion to simultaneously explain both system-level and individual-level dynamics. We employ a latent system-level discrete state Markov chain that drives latent entity-level chains which in turn govern the dynamics of each observed time series. Feedback from the observations to the chains at both the entity and system levels improves flexibility via context-dependent state transitions. Our hierarchical switching recurrent dynamical models can be learned via closed-form variational coordinate ascent updates to all latent chains that scale linearly in the number of indivi
    

