# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Right on Time: Revising Time Series Models by Constraining their Explanations](https://arxiv.org/abs/2402.12921) | 引入了准时到位（RioT）方法，通过使模型解释在时间和频率域之间交互，并利用反馈来约束模型，有效地解决了时间序列数据中的混杂因素问题。 |
| [^2] | [Where is the Truth? The Risk of Getting Confounded in a Continual World](https://arxiv.org/abs/2402.06434) | 这篇论文研究了在一个连续学习环境中遭遇混淆的问题，通过实验证明了传统的连续学习方法无法忽略混淆，需要更强大的方法来处理这个问题。 |
| [^3] | [DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning](https://arxiv.org/abs/2402.05421) | DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。 |
| [^4] | [Evolution Guided Generative Flow Networks](https://arxiv.org/abs/2402.02186) | 本文提出了进化引导的生成流网络（EGFN），用于处理长时间跨度和稀疏奖励的挑战。通过使用进化算法训练一组代理参数，并将结果轨迹存储在优先回放缓冲区中，我们的方法在训练GFlowNets代理时展现出了很高的效果。 |
| [^5] | [Analyzing Sharpness-aware Minimization under Overparameterization](https://arxiv.org/abs/2311.17539) | 本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。 |
| [^6] | [Manipulating Feature Visualizations with Gradient Slingshots.](http://arxiv.org/abs/2401.06122) | 本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。 |
| [^7] | [Searching for ribbons with machine learning.](http://arxiv.org/abs/2304.09304) | 用机器学习发现带状物，反驳四维平凡 Poincaré 猜想。 |

# 详细

[^1]: 准时到位：通过限制时间序列模型的解释来修订它们

    Right on Time: Revising Time Series Models by Constraining their Explanations

    [https://arxiv.org/abs/2402.12921](https://arxiv.org/abs/2402.12921)

    引入了准时到位（RioT）方法，通过使模型解释在时间和频率域之间交互，并利用反馈来约束模型，有效地解决了时间序列数据中的混杂因素问题。

    

    深度时间序列模型的可靠性经常会受到其依赖混杂因素的倾向的损害，这可能导致误导性结果。我们的新记录的、自然混杂的数据集P2S来自真实的机械生产线，强调了这一点。为了解决时间序列数据中的混杂因素的挑战性问题，我们引入了准时到位（RioT）。我们的方法使模型解释在时间和频率域之间进行交互。然后利用两个域内的解释反馈来约束模型，使其远离标注的混杂因素。在处理时间序列数据集中混杂因素方面，双域交互策略至关重要。我们凭经验证明，RioT能够有效地引导模型远离P2S以及流行的时间序列分类和预测数据集中的错误原因。

    arXiv:2402.12921v1 Announce Type: new  Abstract: The reliability of deep time series models is often compromised by their tendency to rely on confounding factors, which may lead to misleading results. Our newly recorded, naturally confounded dataset named P2S from a real mechanical production line emphasizes this. To tackle the challenging problem of mitigating confounders in time series data, we introduce Right on Time (RioT). Our method enables interactions with model explanations across both the time and frequency domain. Feedback on explanations in both domains is then used to constrain the model, steering it away from the annotated confounding factors. The dual-domain interaction strategy is crucial for effectively addressing confounders in time series datasets. We empirically demonstrate that RioT can effectively guide models away from the wrong reasons in P2S as well as popular time series classification and forecasting datasets.
    
[^2]: 真相在哪里？在连续的世界中遭遇混淆的风险

    Where is the Truth? The Risk of Getting Confounded in a Continual World

    [https://arxiv.org/abs/2402.06434](https://arxiv.org/abs/2402.06434)

    这篇论文研究了在一个连续学习环境中遭遇混淆的问题，通过实验证明了传统的连续学习方法无法忽略混淆，需要更强大的方法来处理这个问题。

    

    如果一个数据集通过一个虚假相关性来解决，而这种相关性无法泛化到新数据，该数据集就是混淆的。我们将展示，在一个连续学习的环境中，混淆因素可能随着任务的变化而变化，导致的挑战远远超过通常考虑的遗忘问题。具体来说，我们从数学上推导了这种混淆因素对一组混淆任务的有效联合解空间的影响。有趣的是，我们的理论预测，在许多这样的连续数据集中，当任务进行联合训练时，虚假相关性很容易被忽略，但是在顺序考虑任务时，避免混淆要困难得多。我们构建了这样一个数据集，并通过实验证明标准的连续学习方法无法忽略混淆，而同时对所有任务进行联合训练则是成功的。我们的连续混淆数据集ConCon基于CLEVR图像，证明了需要更强大的连续学习方法来处理混淆问题。

    A dataset is confounded if it is most easily solved via a spurious correlation which fails to generalize to new data. We will show that, in a continual learning setting where confounders may vary in time across tasks, the resulting challenge far exceeds the standard forgetting problem normally considered. In particular, we derive mathematically the effect of such confounders on the space of valid joint solutions to sets of confounded tasks. Interestingly, our theory predicts that for many such continual datasets, spurious correlations are easily ignored when the tasks are trained on jointly, but it is far harder to avoid confounding when they are considered sequentially. We construct such a dataset and demonstrate empirically that standard continual learning methods fail to ignore confounders, while training jointly on all tasks is successful. Our continually confounded dataset, ConCon, is based on CLEVR images and demonstrates the need for continual learning methods with more robust b
    
[^3]: DiffTOP: 可微分轨迹优化在强化学习和模仿学习中的应用

    DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning

    [https://arxiv.org/abs/2402.05421](https://arxiv.org/abs/2402.05421)

    DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。

    

    本文介绍了DiffTOP，它利用可微分轨迹优化作为策略表示，为深度强化学习和模仿学习生成动作。轨迹优化是一种在控制领域中广泛使用的算法，由成本和动力学函数参数化。我们的方法的关键是利用了最近在可微分轨迹优化方面的进展，使得可以计算损失对于轨迹优化的参数的梯度。因此，轨迹优化的成本和动力学函数可以端到端地学习。DiffTOP解决了之前模型基于强化学习算法中的“目标不匹配”问题，因为DiffTOP中的动力学模型通过轨迹优化过程中的策略梯度损失直接最大化任务性能。我们还对DiffTOP在标准机器人操纵任务套件中进行了模仿学习性能基准测试。

    This paper introduces DiffTOP, which utilizes Differentiable Trajectory OPtimization as the policy representation to generate actions for deep reinforcement and imitation learning. Trajectory optimization is a powerful and widely used algorithm in control, parameterized by a cost and a dynamics function. The key to our approach is to leverage the recent progress in differentiable trajectory optimization, which enables computing the gradients of the loss with respect to the parameters of trajectory optimization. As a result, the cost and dynamics functions of trajectory optimization can be learned end-to-end. DiffTOP addresses the ``objective mismatch'' issue of prior model-based RL algorithms, as the dynamics model in DiffTOP is learned to directly maximize task performance by differentiating the policy gradient loss through the trajectory optimization process. We further benchmark DiffTOP for imitation learning on standard robotic manipulation task suites with high-dimensional sensory
    
[^4]: 进化引导的生成流网络

    Evolution Guided Generative Flow Networks

    [https://arxiv.org/abs/2402.02186](https://arxiv.org/abs/2402.02186)

    本文提出了进化引导的生成流网络（EGFN），用于处理长时间跨度和稀疏奖励的挑战。通过使用进化算法训练一组代理参数，并将结果轨迹存储在优先回放缓冲区中，我们的方法在训练GFlowNets代理时展现出了很高的效果。

    

    生成流网络（GFlowNets）是一类概率生成模型，用于学习按照奖励比例对组合对象进行采样。GFlowNets的一个重大挑战是在处理长时间跨度和稀疏奖励时有效训练它们。为了解决这个问题，我们提出了进化引导的生成流网络（EGFN），这是对GFlowNets训练的一种简单但强大的增强方法，使用进化算法（EA）进行训练。我们的方法可以在任何GFlowNets训练目标的基础上工作，通过使用EA训练一组代理参数，将结果轨迹存储在优先回放缓冲区中，并使用存储的轨迹训练GFlowNets代理。我们在广泛的玩具和真实世界基准任务上进行了深入研究，展示了我们方法在处理长轨迹和稀疏奖励方面的有效性。

    Generative Flow Networks (GFlowNets) are a family of probabilistic generative models that learn to sample compositional objects proportional to their rewards. One big challenge of GFlowNets is training them effectively when dealing with long time horizons and sparse rewards. To address this, we propose Evolution guided generative flow networks (EGFN), a simple but powerful augmentation to the GFlowNets training using Evolutionary algorithms (EA). Our method can work on top of any GFlowNets training objective, by training a set of agent parameters using EA, storing the resulting trajectories in the prioritized replay buffer, and training the GFlowNets agent using the stored trajectories. We present a thorough investigation over a wide range of toy and real-world benchmark tasks showing the effectiveness of our method in handling long trajectories and sparse rewards.
    
[^5]: 在过参数化下分析锐度感知最小化

    Analyzing Sharpness-aware Minimization under Overparameterization

    [https://arxiv.org/abs/2311.17539](https://arxiv.org/abs/2311.17539)

    本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。

    

    在训练过参数化的神经网络时，尽管训练损失相同，但可以得到具有不同泛化能力的极小值。有证据表明，极小值的锐度与其泛化误差之间存在相关性，因此已经做出了更多努力开发一种优化方法，以显式地找到扁平极小值作为更具有泛化能力的解。然而，至今为止，关于过参数化对锐度感知最小化（SAM）策略的影响的研究还不多。在这项工作中，我们分析了在不同程度的过参数化下的SAM，并提出了实证和理论结果，表明过参数化对SAM具有重要影响。具体而言，我们进行了广泛的数值实验，涵盖了各个领域，并表明存在一种一致的趋势，即SAM在过参数化增加的情况下仍然受益。我们还发现了一些令人信服的案例，说明了过参数化的影响。

    Training an overparameterized neural network can yield minimizers of different generalization capabilities despite the same level of training loss. With evidence that suggests a correlation between sharpness of minima and their generalization errors, increasing efforts have been made to develop an optimization method to explicitly find flat minima as more generalizable solutions. However, this sharpness-aware minimization (SAM) strategy has not been studied much yet as to whether and how it is affected by overparameterization.   In this work, we analyze SAM under overparameterization of varying degrees and present both empirical and theoretical results that indicate a critical influence of overparameterization on SAM. Specifically, we conduct extensive numerical experiments across various domains, and show that there exists a consistent trend that SAM continues to benefit from increasing overparameterization. We also discover compelling cases where the effect of overparameterization is
    
[^6]: 用梯度弹射操纵特征可视化

    Manipulating Feature Visualizations with Gradient Slingshots. (arXiv:2401.06122v1 [cs.LG])

    [http://arxiv.org/abs/2401.06122](http://arxiv.org/abs/2401.06122)

    本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。

    

    深度神经网络(DNNs)能够学习复杂而多样化的表示，然而，学习到的概念的语义性质仍然未知。解释DNNs学习到的概念的常用方法是激活最大化(AM)，它生成一个合成的输入信号，最大化激活网络中的特定神经元。在本文中，我们研究了这种方法对于对抗模型操作的脆弱性，并引入了一种新的方法来操纵特征可视化，而不改变模型结构或对模型的决策过程产生显著影响。我们评估了我们的方法对几个神经网络模型的效果，并展示了它隐藏特定神经元功能的能力，在模型审核过程中使用选择的目标解释屏蔽了原始解释。作为一种补救措施，我们提出了一种防止这种操纵的防护措施，并提供了定量证据，证明了它的有效性。

    Deep Neural Networks (DNNs) are capable of learning complex and versatile representations, however, the semantic nature of the learned concepts remains unknown. A common method used to explain the concepts learned by DNNs is Activation Maximization (AM), which generates a synthetic input signal that maximally activates a particular neuron in the network. In this paper, we investigate the vulnerability of this approach to adversarial model manipulations and introduce a novel method for manipulating feature visualization without altering the model architecture or significantly impacting the model's decision-making process. We evaluate the effectiveness of our method on several neural network models and demonstrate its capabilities to hide the functionality of specific neurons by masking the original explanations of neurons with chosen target explanations during model auditing. As a remedy, we propose a protective measure against such manipulations and provide quantitative evidence which 
    
[^7]: 用机器学习搜索带状物

    Searching for ribbons with machine learning. (arXiv:2304.09304v1 [math.GT])

    [http://arxiv.org/abs/2304.09304](http://arxiv.org/abs/2304.09304)

    用机器学习发现带状物，反驳四维平凡 Poincaré 猜想。

    

    我们将贝叶斯优化和强化学习应用于拓扑学中的一个问题：如何确定一个结不能限定一个带状物。该问题在反驳四维平凡 Poincaré 猜想的方法中是相关的；利用我们的程序，我们排除了许多猜想的反例。我们还展示了这些程序成功检测了范围在70节点内的许多带状物。

    We apply Bayesian optimization and reinforcement learning to a problem in topology: the question of when a knot bounds a ribbon disk. This question is relevant in an approach to disproving the four-dimensional smooth Poincar\'e conjecture; using our programs, we rule out many potential counterexamples to the conjecture. We also show that the programs are successful in detecting many ribbon knots in the range of up to 70 crossings.
    

