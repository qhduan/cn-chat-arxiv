# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonlinear Filtering with Brenier Optimal Transport Maps](https://rss.arxiv.org/abs/2310.13886) | 本文提出了基于Brenier最优输运映射的非线性滤波方法，通过估计先验分布到后验分布的映射来避免权重退化问题，并利用神经网络建模复杂分布和随机优化算法提高可扩展性。 |
| [^2] | [Multi-Trigger Backdoor Attacks: More Triggers, More Threats.](http://arxiv.org/abs/2401.15295) | 本文主要研究了多触发后门攻击对深度神经网络的威胁。通过提出并研究了三种类型的多触发攻击，包括并行、顺序和混合攻击，文章揭示了不同触发器对同一数据集的共存、覆写和交叉激活效果。结果表明单触发攻击容易引起覆写问题。 |
| [^3] | [Model-free generalized fiducial inference.](http://arxiv.org/abs/2307.12472) | 本文提出了一种无模型的统计框架，用于不准确概率预测推理的不确定性量化，并考虑了精确概率近似模型无关的不准确推理框架的特性。 |

# 详细

[^1]: Brenier最优输运映射下的非线性滤波

    Nonlinear Filtering with Brenier Optimal Transport Maps

    [https://rss.arxiv.org/abs/2310.13886](https://rss.arxiv.org/abs/2310.13886)

    本文提出了基于Brenier最优输运映射的非线性滤波方法，通过估计先验分布到后验分布的映射来避免权重退化问题，并利用神经网络建模复杂分布和随机优化算法提高可扩展性。

    

    本论文研究非线性滤波问题，即在给定噪声部分观测历史的情况下计算随机动态系统状态的条件分布。传统的序列重要重采样（SIR）粒子滤波由于权重退化问题，在涉及退化似然或高维状态的情况下存在基本限制。在本文中，我们探索了一种基于估计从当前先验分布到下一个时间步的后验分布的Brenier最优输运（OT）映射的替代方法。与SIR粒子滤波不同，OT方法不需要似然的解析形式。此外，它允许我们利用神经网络的逼近能力来建模复杂的多模态分布，并使用随机优化算法来提高可扩展性。我们进行了大量的数字实验，比较了OT滤波器和SIR粒子滤波器的性能。

    This paper is concerned with the problem of nonlinear filtering, i.e., computing the conditional distribution of the state of a stochastic dynamical system given a history of noisy partial observations. Conventional sequential importance resampling (SIR) particle filters suffer from fundamental limitations, in scenarios involving degenerate likelihoods or high-dimensional states, due to the weight degeneracy issue. In this paper, we explore an alternative method, which is based on estimating the Brenier optimal transport (OT) map from the current prior distribution of the state to the posterior distribution at the next time step. Unlike SIR particle filters, the OT formulation does not require the analytical form of the likelihood. Moreover, it allows us to harness the approximation power of neural networks to model complex and multi-modal distributions and employ stochastic optimization algorithms to enhance scalability. Extensive numerical experiments are presented that compare the O
    
[^2]: 多触发后门攻击：更多触发器，更多威胁

    Multi-Trigger Backdoor Attacks: More Triggers, More Threats. (arXiv:2401.15295v1 [cs.LG])

    [http://arxiv.org/abs/2401.15295](http://arxiv.org/abs/2401.15295)

    本文主要研究了多触发后门攻击对深度神经网络的威胁。通过提出并研究了三种类型的多触发攻击，包括并行、顺序和混合攻击，文章揭示了不同触发器对同一数据集的共存、覆写和交叉激活效果。结果表明单触发攻击容易引起覆写问题。

    

    后门攻击已经成为深度神经网络（DNNs）的（预）训练和部署的主要威胁。尽管后门攻击在一些研究中已经得到了广泛的探讨，但其中大部分都集中在使用单个类型的触发器来污染数据集的单触发攻击上。可以说，在现实世界中，后门攻击可能更加复杂，例如，同一数据集可能存在多个对手，如果该数据集具有较高的价值。在这项工作中，我们研究了在多触发攻击设置下后门攻击的实际威胁，多个对手利用不同类型的触发器来污染同一数据集。通过提出和研究并行、顺序和混合攻击这三种类型的多触发攻击，我们提供了关于不同触发器对同一数据集的共存、覆写和交叉激活效果的重要认识。此外，我们还展示了单触发攻击往往容易引起覆写问题。

    Backdoor attacks have emerged as a primary threat to (pre-)training and deployment of deep neural networks (DNNs). While backdoor attacks have been extensively studied in a body of works, most of them were focused on single-trigger attacks that poison a dataset using a single type of trigger. Arguably, real-world backdoor attacks can be much more complex, e.g., the existence of multiple adversaries for the same dataset if it is of high value. In this work, we investigate the practical threat of backdoor attacks under the setting of \textbf{multi-trigger attacks} where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks, including parallel, sequential, and hybrid attacks, we provide a set of important understandings of the coexisting, overwriting, and cross-activating effects between different triggers on the same dataset. Moreover, we show that single-trigger attacks tend to cause over
    
[^3]: 无模型广义基准推理

    Model-free generalized fiducial inference. (arXiv:2307.12472v1 [stat.ML])

    [http://arxiv.org/abs/2307.12472](http://arxiv.org/abs/2307.12472)

    本文提出了一种无模型的统计框架，用于不准确概率预测推理的不确定性量化，并考虑了精确概率近似模型无关的不准确推理框架的特性。

    

    鉴于机器学习中不确定性量化方法的安全可靠性的需求，本文提出并发展了一种无模型的统计框架，用于不准确概率预测推理的不确定性量化。该框架通过提供预测集的形式，实现了对第一类错误的有限样本控制，这与一致性预测集具有相同的属性，但这种新方法还提供了更灵活的不准确概率推理工具。此外，本文提出并考虑了一种精确概率近似模型无关的不准确推理框架的理论和实证特性。通过将信念/可信度度量对近似为在可信区间中的[在某种意义上最优]概率度量，是扩大在统计和机器学习社区推广不准确概率推理方法所需的关键解决方案，目前在统计和

    Motivated by the need for the development of safe and reliable methods for uncertainty quantification in machine learning, I propose and develop ideas for a model-free statistical framework for imprecise probabilistic prediction inference. This framework facilitates uncertainty quantification in the form of prediction sets that offer finite sample control of type 1 errors, a property shared with conformal prediction sets, but this new approach also offers more versatile tools for imprecise probabilistic reasoning. Furthermore, I propose and consider the theoretical and empirical properties of a precise probabilistic approximation to the model-free imprecise framework. Approximating a belief/plausibility measure pair by an [optimal in some sense] probability measure in the credal set is a critical resolution needed for the broader adoption of imprecise probabilistic approaches to inference in statistical and machine learning communities. It is largely undetermined in the statistical and
    

