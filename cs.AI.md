# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluating Decision Optimality of Autonomous Driving via Metamorphic Testing](https://arxiv.org/abs/2402.18393) | 本文着重于评估自动驾驶系统的决策质量，提出了检测非最佳决策场景的方法，通过新颖的变形关系暴露最佳决策违规。 |
| [^2] | [Bandits with Deterministically Evolving States.](http://arxiv.org/abs/2307.11655) | 该论文提出了一种名为具有确定性演化状态的强盗模型，用于学习带有强盗反馈的推荐系统和在线广告。该模型考虑了状态演化的不同速率，能准确评估奖励与系统健康程度之间的关系。 |
| [^3] | [QuantumNAT: Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization.](http://arxiv.org/abs/2110.11331) | QuantumNAT是一个PQC特定框架，可以在训练和推断阶段执行噪声感知优化，提高鲁棒性，缓解量子噪声 |

# 详细

[^1]: 通过变形测试评估自动驾驶的决策最佳性

    Evaluating Decision Optimality of Autonomous Driving via Metamorphic Testing

    [https://arxiv.org/abs/2402.18393](https://arxiv.org/abs/2402.18393)

    本文着重于评估自动驾驶系统的决策质量，提出了检测非最佳决策场景的方法，通过新颖的变形关系暴露最佳决策违规。

    

    arXiv:2402.18393v1 公告类型：新摘要：自动驾驶系统（ADS）的测试在ADS开发中至关重要，目前主要关注的是安全性。然而，评估非安全关键性能，特别是ADS制定最佳决策并为自动车辆（AV）生成最佳路径的能力同样重要，以确保AV的智能性并降低风险。目前，鲜有工作致力于评估ADS的最佳决策性能，因为缺乏相应的预言和生成有非最佳决策的场景难度较大。本文侧重于评估ADS的决策质量，并提出首个用于检测非最佳决策场景（NoDSs）的方法，即ADS未计算AV的最佳路径的情况。首先，为解决预言问题，我们提出了一种旨在暴露最佳决策违规情况的新颖变形关系（MR）。这个MR确定了性能最佳决策的违规。

    arXiv:2402.18393v1 Announce Type: new  Abstract: Autonomous Driving System (ADS) testing is crucial in ADS development, with the current primary focus being on safety. However, the evaluation of non-safety-critical performance, particularly the ADS's ability to make optimal decisions and produce optimal paths for autonomous vehicles (AVs), is equally vital to ensure the intelligence and reduce risks of AVs. Currently, there is little work dedicated to assessing ADSs' optimal decision-making performance due to the lack of corresponding oracles and the difficulty in generating scenarios with non-optimal decisions. In this paper, we focus on evaluating the decision-making quality of an ADS and propose the first method for detecting non-optimal decision scenarios (NoDSs), where the ADS does not compute optimal paths for AVs. Firstly, to deal with the oracle problem, we propose a novel metamorphic relation (MR) aimed at exposing violations of optimal decisions. The MR identifies the propert
    
[^2]: 具有确定性演化状态的强盗模型

    Bandits with Deterministically Evolving States. (arXiv:2307.11655v1 [cs.LG])

    [http://arxiv.org/abs/2307.11655](http://arxiv.org/abs/2307.11655)

    该论文提出了一种名为具有确定性演化状态的强盗模型，用于学习带有强盗反馈的推荐系统和在线广告。该模型考虑了状态演化的不同速率，能准确评估奖励与系统健康程度之间的关系。

    

    我们提出了一种学习与强盗反馈结合的模型，同时考虑到确定性演化和不可观测的状态，我们称之为具有确定性演化状态的强盗模型。我们的模型主要应用于推荐系统和在线广告的学习。在这两种情况下，算法在每一轮获得的奖励是选择行动的短期奖励和系统的“健康”程度（即通过其状态测量）的函数。例如，在推荐系统中，平台从用户对特定类型内容的参与中获得的奖励不仅取决于具体内容的固有特征，还取决于用户与平台上其他类型内容互动后其偏好的演化。我们的通用模型考虑了状态演化的不同速率λ∈[0,1]（例如，用户的偏好因先前内容消费而快速变化）。

    We propose a model for learning with bandit feedback while accounting for deterministically evolving and unobservable states that we call Bandits with Deterministically Evolving States. The workhorse applications of our model are learning for recommendation systems and learning for online ads. In both cases, the reward that the algorithm obtains at each round is a function of the short-term reward of the action chosen and how ``healthy'' the system is (i.e., as measured by its state). For example, in recommendation systems, the reward that the platform obtains from a user's engagement with a particular type of content depends not only on the inherent features of the specific content, but also on how the user's preferences have evolved as a result of interacting with other types of content on the platform. Our general model accounts for the different rate $\lambda \in [0,1]$ at which the state evolves (e.g., how fast a user's preferences shift as a result of previous content consumption
    
[^3]: QuantumNAT：注重量子噪声的噪声注入、量化和归一化的量子训练

    QuantumNAT: Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization. (arXiv:2110.11331v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2110.11331](http://arxiv.org/abs/2110.11331)

    QuantumNAT是一个PQC特定框架，可以在训练和推断阶段执行噪声感知优化，提高鲁棒性，缓解量子噪声

    

    参数化量子电路是实现近期量子硬件优势的有希望方法。然而，由于存在较大的量子噪声（误差），在实际的量子设备上，PQC模型的性能会受到严重的降级。我们提出了QuantumNAT，一个可以在训练和推断阶段执行噪声感知优化的PQC特定框架，以提高其鲁棒性。通过实验我们发现，量子噪声对PQC测量结果的影响是从无噪声结果经过一个缩放和偏移因子得到的线性映射。基于此，我们提出了后测量归一化来缓解特征分布不一致的问题。

    Parameterized Quantum Circuits (PQC) are promising towards quantum advantage on near-term quantum hardware. However, due to the large quantum noises (errors), the performance of PQC models has a severe degradation on real quantum devices. Take Quantum Neural Network (QNN) as an example, the accuracy gap between noise-free simulation and noisy results on IBMQ-Yorktown for MNIST-4 classification is over 60%. Existing noise mitigation methods are general ones without leveraging unique characteristics of PQC; on the other hand, existing PQC work does not consider noise effect. To this end, we present QuantumNAT, a PQC-specific framework to perform noise-aware optimizations in both training and inference stages to improve robustness. We experimentally observe that the effect of quantum noise to PQC measurement outcome is a linear map from noise-free outcome with a scaling and a shift factor. Motivated by that, we propose post-measurement normalization to mitigate the feature distribution di
    

