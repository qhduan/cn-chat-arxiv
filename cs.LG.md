# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Defer in Content Moderation: The Human-AI Interplay](https://arxiv.org/abs/2402.12237) | 本文提出了一个模型，捕捉内容审核中人工智能的相互作用。 |
| [^2] | [Discrete Neural Algorithmic Reasoning](https://arxiv.org/abs/2402.11628) | 这项工作提出了一种强制神经推理器维护执行轨迹作为有限预定义状态组合的方法，通过对算法状态转换的监督训练，使模型能够与原始算法完美对齐，并在基准测试中取得了完美的测试成绩。 |
| [^3] | [Forecasting the steam mass flow in a powerplant using the parallel hybrid network.](http://arxiv.org/abs/2307.09483) | 这项研究使用并行混合神经网络结构来预测发电厂中的蒸汽质量流量，相比纯经典和纯量子模型，该混合模型在测试集上取得了更好的性能，平均平方误差降低了5.7倍和4.9倍，并且相对误差较小，最多提升了2倍。 |

# 详细

[^1]: 学习在内容审核中推迟：人工智能与人类协同作用

    Learning to Defer in Content Moderation: The Human-AI Interplay

    [https://arxiv.org/abs/2402.12237](https://arxiv.org/abs/2402.12237)

    本文提出了一个模型，捕捉内容审核中人工智能的相互作用。

    

    成功的在线平台内容审核依赖于人工智能协同方法。本文介绍了一个模型，捕捉内容审核中人工智能的相互作用。算法观察到即将发布的帖子的背景信息，做出分类和准入决策，并安排帖子进行人工审核。

    arXiv:2402.12237v1 Announce Type: cross  Abstract: Successful content moderation in online platforms relies on a human-AI collaboration approach. A typical heuristic estimates the expected harmfulness of a post and uses fixed thresholds to decide whether to remove it and whether to send it for human review. This disregards the prediction uncertainty, the time-varying element of human review capacity and post arrivals, and the selective sampling in the dataset (humans only review posts filtered by the admission algorithm).   In this paper, we introduce a model to capture the human-AI interplay in content moderation. The algorithm observes contextual information for incoming posts, makes classification and admission decisions, and schedules posts for human review. Only admitted posts receive human reviews on their harmfulness. These reviews help educate the machine-learning algorithms but are delayed due to congestion in the human review system. The classical learning-theoretic way to ca
    
[^2]: 离散神经算法推理

    Discrete Neural Algorithmic Reasoning

    [https://arxiv.org/abs/2402.11628](https://arxiv.org/abs/2402.11628)

    这项工作提出了一种强制神经推理器维护执行轨迹作为有限预定义状态组合的方法，通过对算法状态转换的监督训练，使模型能够与原始算法完美对齐，并在基准测试中取得了完美的测试成绩。

    

    神经算法推理旨在通过学习模仿经典算法的执行来捕捉神经网络中的计算。尽管常见的架构足够表达正确的模型在权重空间中，但当前的神经推理器在处理超出分布数据时面临泛化困难。另一方面，经典计算不受分布变化的影响，因为它们可以描述为离散计算状态之间的转换。在这项工作中，我们提出强制神经推理器将执行轨迹作为有限预定义状态的组合进行维护。通过对算法状态转换的监督训练，这种模型能够与原始算法完美对齐。为了证明这一点，我们在SALSA-CLRS基准测试上评估我们的方法，在那里我们为所有任务获得了完美的测试成绩。此外，所提出的架构选择使我们能够证明...

    arXiv:2402.11628v1 Announce Type: new  Abstract: Neural algorithmic reasoning aims to capture computations with neural networks via learning the models to imitate the execution of classical algorithms. While common architectures are expressive enough to contain the correct model in the weights space, current neural reasoners are struggling to generalize well on out-of-distribution data. On the other hand, classical computations are not affected by distribution shifts as they can be described as transitions between discrete computational states. In this work, we propose to force neural reasoners to maintain the execution trajectory as a combination of finite predefined states. Trained with supervision on the algorithm's state transitions, such models are able to perfectly align with the original algorithm. To show this, we evaluate our approach on the SALSA-CLRS benchmark, where we get perfect test scores for all tasks. Moreover, the proposed architectural choice allows us to prove the 
    
[^3]: 使用并行混合网络预测发电厂中的蒸汽质量流量

    Forecasting the steam mass flow in a powerplant using the parallel hybrid network. (arXiv:2307.09483v1 [cs.LG])

    [http://arxiv.org/abs/2307.09483](http://arxiv.org/abs/2307.09483)

    这项研究使用并行混合神经网络结构来预测发电厂中的蒸汽质量流量，相比纯经典和纯量子模型，该混合模型在测试集上取得了更好的性能，平均平方误差降低了5.7倍和4.9倍，并且相对误差较小，最多提升了2倍。

    

    高效可持续的发电是能源领域的一个关键问题。尤其是热电厂在准确预测蒸汽质量流量方面面临困难，这对于运营效率和成本降低至关重要。在本研究中，我们使用一个并行混合神经网络结构，该结构将参数化量子电路和传统的前馈神经网络相结合，特别设计用于工业环境中的时间序列预测，以提高对未来15分钟内蒸汽质量流量的预测能力。我们的结果表明，并行混合模型优于独立的经典和量子模型，在训练后的测试集上相对于纯经典模型和纯量子网络，平均平方误差（MSE）损失分别降低了5.7倍和4.9倍。此外，该混合模型在测试集上表现出相对误差较小，比纯经典模型更好，最多提升了2倍。

    Efficient and sustainable power generation is a crucial concern in the energy sector. In particular, thermal power plants grapple with accurately predicting steam mass flow, which is crucial for operational efficiency and cost reduction. In this study, we use a parallel hybrid neural network architecture that combines a parametrized quantum circuit and a conventional feed-forward neural network specifically designed for time-series prediction in industrial settings to enhance predictions of steam mass flow 15 minutes into the future. Our results show that the parallel hybrid model outperforms standalone classical and quantum models, achieving more than 5.7 and 4.9 times lower mean squared error (MSE) loss on the test set after training compared to pure classical and pure quantum networks, respectively. Furthermore, the hybrid model demonstrates smaller relative errors between the ground truth and the model predictions on the test set, up to 2 times better than the pure classical model.
    

