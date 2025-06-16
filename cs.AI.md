# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning](https://arxiv.org/abs/2402.05421) | DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。 |
| [^2] | [Evolution Guided Generative Flow Networks](https://arxiv.org/abs/2402.02186) | 本文提出了进化引导的生成流网络（EGFN），用于处理长时间跨度和稀疏奖励的挑战。通过使用进化算法训练一组代理参数，并将结果轨迹存储在优先回放缓冲区中，我们的方法在训练GFlowNets代理时展现出了很高的效果。 |
| [^3] | [Manipulating Feature Visualizations with Gradient Slingshots.](http://arxiv.org/abs/2401.06122) | 本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。 |

# 详细

[^1]: DiffTOP: 可微分轨迹优化在强化学习和模仿学习中的应用

    DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning

    [https://arxiv.org/abs/2402.05421](https://arxiv.org/abs/2402.05421)

    DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。

    

    本文介绍了DiffTOP，它利用可微分轨迹优化作为策略表示，为深度强化学习和模仿学习生成动作。轨迹优化是一种在控制领域中广泛使用的算法，由成本和动力学函数参数化。我们的方法的关键是利用了最近在可微分轨迹优化方面的进展，使得可以计算损失对于轨迹优化的参数的梯度。因此，轨迹优化的成本和动力学函数可以端到端地学习。DiffTOP解决了之前模型基于强化学习算法中的“目标不匹配”问题，因为DiffTOP中的动力学模型通过轨迹优化过程中的策略梯度损失直接最大化任务性能。我们还对DiffTOP在标准机器人操纵任务套件中进行了模仿学习性能基准测试。

    This paper introduces DiffTOP, which utilizes Differentiable Trajectory OPtimization as the policy representation to generate actions for deep reinforcement and imitation learning. Trajectory optimization is a powerful and widely used algorithm in control, parameterized by a cost and a dynamics function. The key to our approach is to leverage the recent progress in differentiable trajectory optimization, which enables computing the gradients of the loss with respect to the parameters of trajectory optimization. As a result, the cost and dynamics functions of trajectory optimization can be learned end-to-end. DiffTOP addresses the ``objective mismatch'' issue of prior model-based RL algorithms, as the dynamics model in DiffTOP is learned to directly maximize task performance by differentiating the policy gradient loss through the trajectory optimization process. We further benchmark DiffTOP for imitation learning on standard robotic manipulation task suites with high-dimensional sensory
    
[^2]: 进化引导的生成流网络

    Evolution Guided Generative Flow Networks

    [https://arxiv.org/abs/2402.02186](https://arxiv.org/abs/2402.02186)

    本文提出了进化引导的生成流网络（EGFN），用于处理长时间跨度和稀疏奖励的挑战。通过使用进化算法训练一组代理参数，并将结果轨迹存储在优先回放缓冲区中，我们的方法在训练GFlowNets代理时展现出了很高的效果。

    

    生成流网络（GFlowNets）是一类概率生成模型，用于学习按照奖励比例对组合对象进行采样。GFlowNets的一个重大挑战是在处理长时间跨度和稀疏奖励时有效训练它们。为了解决这个问题，我们提出了进化引导的生成流网络（EGFN），这是对GFlowNets训练的一种简单但强大的增强方法，使用进化算法（EA）进行训练。我们的方法可以在任何GFlowNets训练目标的基础上工作，通过使用EA训练一组代理参数，将结果轨迹存储在优先回放缓冲区中，并使用存储的轨迹训练GFlowNets代理。我们在广泛的玩具和真实世界基准任务上进行了深入研究，展示了我们方法在处理长轨迹和稀疏奖励方面的有效性。

    Generative Flow Networks (GFlowNets) are a family of probabilistic generative models that learn to sample compositional objects proportional to their rewards. One big challenge of GFlowNets is training them effectively when dealing with long time horizons and sparse rewards. To address this, we propose Evolution guided generative flow networks (EGFN), a simple but powerful augmentation to the GFlowNets training using Evolutionary algorithms (EA). Our method can work on top of any GFlowNets training objective, by training a set of agent parameters using EA, storing the resulting trajectories in the prioritized replay buffer, and training the GFlowNets agent using the stored trajectories. We present a thorough investigation over a wide range of toy and real-world benchmark tasks showing the effectiveness of our method in handling long trajectories and sparse rewards.
    
[^3]: 用梯度弹射操纵特征可视化

    Manipulating Feature Visualizations with Gradient Slingshots. (arXiv:2401.06122v1 [cs.LG])

    [http://arxiv.org/abs/2401.06122](http://arxiv.org/abs/2401.06122)

    本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。

    

    深度神经网络(DNNs)能够学习复杂而多样化的表示，然而，学习到的概念的语义性质仍然未知。解释DNNs学习到的概念的常用方法是激活最大化(AM)，它生成一个合成的输入信号，最大化激活网络中的特定神经元。在本文中，我们研究了这种方法对于对抗模型操作的脆弱性，并引入了一种新的方法来操纵特征可视化，而不改变模型结构或对模型的决策过程产生显著影响。我们评估了我们的方法对几个神经网络模型的效果，并展示了它隐藏特定神经元功能的能力，在模型审核过程中使用选择的目标解释屏蔽了原始解释。作为一种补救措施，我们提出了一种防止这种操纵的防护措施，并提供了定量证据，证明了它的有效性。

    Deep Neural Networks (DNNs) are capable of learning complex and versatile representations, however, the semantic nature of the learned concepts remains unknown. A common method used to explain the concepts learned by DNNs is Activation Maximization (AM), which generates a synthetic input signal that maximally activates a particular neuron in the network. In this paper, we investigate the vulnerability of this approach to adversarial model manipulations and introduce a novel method for manipulating feature visualization without altering the model architecture or significantly impacting the model's decision-making process. We evaluate the effectiveness of our method on several neural network models and demonstrate its capabilities to hide the functionality of specific neurons by masking the original explanations of neurons with chosen target explanations during model auditing. As a remedy, we propose a protective measure against such manipulations and provide quantitative evidence which 
    

