# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multimodal Clinical Trial Outcome Prediction with Large Language Models](https://arxiv.org/abs/2402.06512) | 本研究提出了一种名为LIFTED的多模态临床试验结果预测方法，通过将不同模态数据转化为自然语言描述来统一数据，并构建统一的抗噪声编码器进行信息提取。 |
| [^2] | [Leveraging Continuously Differentiable Activation Functions for Learning in Quantized Noisy Environments](https://arxiv.org/abs/2402.02593) | 在量化噪声环境中，利用连续可微激活函数进行学习可以减轻模拟量化误差，为计算机视觉、信号处理等多个机器学习领域的硬件实现提供了指导。 |
| [^3] | [Bandits with Deterministically Evolving States.](http://arxiv.org/abs/2307.11655) | 该论文提出了一种名为具有确定性演化状态的强盗模型，用于学习带有强盗反馈的推荐系统和在线广告。该模型考虑了状态演化的不同速率，能准确评估奖励与系统健康程度之间的关系。 |
| [^4] | [Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning.](http://arxiv.org/abs/2307.05772) | 这篇论文提出了一种新的随机集合卷积神经网络（RS-CNN）用于分类，通过预测信念函数而不是概率矢量集合，以表示模型的置信度和认识不确定性。基于认识论深度学习方法，该模型能够估计由有限训练集引起的认识不确定性。 |
| [^5] | [Sources of Uncertainty in Machine Learning -- A Statisticians' View.](http://arxiv.org/abs/2305.16703) | 本文讨论了机器学习中不确定性的来源和类型，从统计学家的视角出发，分类别介绍了随机性和认知性不确定性的概念，证明了不确定性来源各异，不可简单归为两类。同时，与统计学概念进行类比，探讨不确定性在机器学习中的作用。 |
| [^6] | [QuantumNAT: Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization.](http://arxiv.org/abs/2110.11331) | QuantumNAT是一个PQC特定框架，可以在训练和推断阶段执行噪声感知优化，提高鲁棒性，缓解量子噪声 |

# 详细

[^1]: 使用大型语言模型的多模态临床试验结果预测

    Multimodal Clinical Trial Outcome Prediction with Large Language Models

    [https://arxiv.org/abs/2402.06512](https://arxiv.org/abs/2402.06512)

    本研究提出了一种名为LIFTED的多模态临床试验结果预测方法，通过将不同模态数据转化为自然语言描述来统一数据，并构建统一的抗噪声编码器进行信息提取。

    

    临床试验是一个关键且昂贵的过程，通常需要多年时间和大量财力资源。因此，开发临床试验结果预测模型旨在排除可能失败的药物，并具有显著的成本节约潜力。最近的数据驱动尝试利用深度学习方法整合多模态数据来预测临床试验结果。然而，这些方法依赖于手动设计的模态特定编码器，这限制了适应新模态的可扩展性和识别不同模态之间相似信息模式的能力。为了解决这些问题，我们提出了一种多模态专家混合（LIFTED）方法用于临床试验结果预测。具体而言，LIFTED通过将不同模态的数据转化为自然语言描述来统一不同模态数据。然后，LIFTED构建统一的抗噪声编码器，从模态特定的语言描述中提取信息。

    The clinical trial is a pivotal and costly process, often spanning multiple years and requiring substantial financial resources. Therefore, the development of clinical trial outcome prediction models aims to exclude drugs likely to fail and holds the potential for significant cost savings. Recent data-driven attempts leverage deep learning methods to integrate multimodal data for predicting clinical trial outcomes. However, these approaches rely on manually designed modal-specific encoders, which limits both the extensibility to adapt new modalities and the ability to discern similar information patterns across different modalities. To address these issues, we propose a multimodal mixture-of-experts (LIFTED) approach for clinical trial outcome prediction. Specifically, LIFTED unifies different modality data by transforming them into natural language descriptions. Then, LIFTED constructs unified noise-resilient encoders to extract information from modal-specific language descriptions. S
    
[^2]: 在量化噪声环境中利用连续可微激活函数进行学习的优化

    Leveraging Continuously Differentiable Activation Functions for Learning in Quantized Noisy Environments

    [https://arxiv.org/abs/2402.02593](https://arxiv.org/abs/2402.02593)

    在量化噪声环境中，利用连续可微激活函数进行学习可以减轻模拟量化误差，为计算机视觉、信号处理等多个机器学习领域的硬件实现提供了指导。

    

    实际世界中的模拟系统固有地受到噪声的影响，这可能会阻碍各种深度学习模型的收敛性和准确性。我们证明了像GELU和SiLU这样的可微激活函数可以稳健地传播梯度，有助于减轻普遍存在于所有模拟系统中的模拟量化误差。我们在量化噪声存在的情况下进行了卷积、线性和Transformer网络的分析和训练。我们能够证明，与传统的修正线性激活函数相比，连续可微激活函数在抗噪声方面具有显著优势。与ReLU相比，在接近零时梯度误差高出100倍。我们的研究结果为选择适当的激活函数提供了指导，以实现在计算机视觉、信号处理等多个机器学习领域中具有高性能和可靠性的硬件实现。

    Real-world analog systems intrinsically suffer from noise that can impede model convergence and accuracy on a variety of deep learning models. We demonstrate that differentiable activations like GELU and SiLU enable robust propagation of gradients which help to mitigate analog quantization error that is ubiquitous to all analog systems. We perform analysis and training of convolutional, linear, and transformer networks in the presence of quantized noise. Here, we are able to demonstrate that continuously differentiable activation functions are significantly more noise resilient over conventional rectified activations. As in the case of ReLU, the error in gradients are 100x higher than those in GELU near zero. Our findings provide guidance for selecting appropriate activations to realize performant and reliable hardware implementations across several machine learning domains such as computer vision, signal processing, and beyond.
    
[^3]: 具有确定性演化状态的强盗模型

    Bandits with Deterministically Evolving States. (arXiv:2307.11655v1 [cs.LG])

    [http://arxiv.org/abs/2307.11655](http://arxiv.org/abs/2307.11655)

    该论文提出了一种名为具有确定性演化状态的强盗模型，用于学习带有强盗反馈的推荐系统和在线广告。该模型考虑了状态演化的不同速率，能准确评估奖励与系统健康程度之间的关系。

    

    我们提出了一种学习与强盗反馈结合的模型，同时考虑到确定性演化和不可观测的状态，我们称之为具有确定性演化状态的强盗模型。我们的模型主要应用于推荐系统和在线广告的学习。在这两种情况下，算法在每一轮获得的奖励是选择行动的短期奖励和系统的“健康”程度（即通过其状态测量）的函数。例如，在推荐系统中，平台从用户对特定类型内容的参与中获得的奖励不仅取决于具体内容的固有特征，还取决于用户与平台上其他类型内容互动后其偏好的演化。我们的通用模型考虑了状态演化的不同速率λ∈[0,1]（例如，用户的偏好因先前内容消费而快速变化）。

    We propose a model for learning with bandit feedback while accounting for deterministically evolving and unobservable states that we call Bandits with Deterministically Evolving States. The workhorse applications of our model are learning for recommendation systems and learning for online ads. In both cases, the reward that the algorithm obtains at each round is a function of the short-term reward of the action chosen and how ``healthy'' the system is (i.e., as measured by its state). For example, in recommendation systems, the reward that the platform obtains from a user's engagement with a particular type of content depends not only on the inherent features of the specific content, but also on how the user's preferences have evolved as a result of interacting with other types of content on the platform. Our general model accounts for the different rate $\lambda \in [0,1]$ at which the state evolves (e.g., how fast a user's preferences shift as a result of previous content consumption
    
[^4]: 随机集合卷积神经网络（RS-CNN）用于认识论深度学习

    Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning. (arXiv:2307.05772v1 [cs.LG])

    [http://arxiv.org/abs/2307.05772](http://arxiv.org/abs/2307.05772)

    这篇论文提出了一种新的随机集合卷积神经网络（RS-CNN）用于分类，通过预测信念函数而不是概率矢量集合，以表示模型的置信度和认识不确定性。基于认识论深度学习方法，该模型能够估计由有限训练集引起的认识不确定性。

    

    机器学习越来越多地应用于安全关键领域，对抗攻击的鲁棒性至关重要，错误的预测可能导致潜在的灾难性后果。这突出了学习系统需要能够确定模型对其预测的置信度以及与之相关联的认识不确定性的手段，“知道一个模型不知道”。在本文中，我们提出了一种新颖的用于分类的随机集合卷积神经网络（RS-CNN），其预测信念函数而不是概率矢量集合，使用随机集合的数学，即对样本空间的幂集的分布。基于认识论深度学习方法，随机集模型能够表示机器学习中由有限训练集引起的“认识性”不确定性。我们通过近似预测信念函数相关联的置信集的大小来估计认识不确定性。

    Machine learning is increasingly deployed in safety-critical domains where robustness against adversarial attacks is crucial and erroneous predictions could lead to potentially catastrophic consequences. This highlights the need for learning systems to be equipped with the means to determine a model's confidence in its prediction and the epistemic uncertainty associated with it, 'to know when a model does not know'. In this paper, we propose a novel Random-Set Convolutional Neural Network (RS-CNN) for classification which predicts belief functions rather than probability vectors over the set of classes, using the mathematics of random sets, i.e., distributions over the power set of the sample space. Based on the epistemic deep learning approach, random-set models are capable of representing the 'epistemic' uncertainty induced in machine learning by limited training sets. We estimate epistemic uncertainty by approximating the size of credal sets associated with the predicted belief func
    
[^5]: 机器学习中的不确定性来源 -- 一个统计学家的视角

    Sources of Uncertainty in Machine Learning -- A Statisticians' View. (arXiv:2305.16703v1 [stat.ML])

    [http://arxiv.org/abs/2305.16703](http://arxiv.org/abs/2305.16703)

    本文讨论了机器学习中不确定性的来源和类型，从统计学家的视角出发，分类别介绍了随机性和认知性不确定性的概念，证明了不确定性来源各异，不可简单归为两类。同时，与统计学概念进行类比，探讨不确定性在机器学习中的作用。

    

    机器学习和深度学习已经取得了令人瞩目的成就，使我们能够回答几年前难以想象的问题。除了这些成功之外，越来越清晰的是，在纯预测之外，量化不确定性也是相关和必要的。虽然近年来已经出现了这方面的第一批概念和思想，但本文采用了一个概念性的视角，并探讨了可能的不确定性来源。通过采用统计学家的视角，我们讨论了与机器学习更常见相关的随机性和认知性不确定性的概念。本文旨在规范这两种类型的不确定性，并证明不确定性的来源各异，并且不总是可以分解为随机性和认知性。通过将统计概念与机器学习中的不确定性进行类比，我们也展示了统计学概念和机器学习中不确定性的作用。

    Machine Learning and Deep Learning have achieved an impressive standard today, enabling us to answer questions that were inconceivable a few years ago. Besides these successes, it becomes clear, that beyond pure prediction, which is the primary strength of most supervised machine learning algorithms, the quantification of uncertainty is relevant and necessary as well. While first concepts and ideas in this direction have emerged in recent years, this paper adopts a conceptual perspective and examines possible sources of uncertainty. By adopting the viewpoint of a statistician, we discuss the concepts of aleatoric and epistemic uncertainty, which are more commonly associated with machine learning. The paper aims to formalize the two types of uncertainty and demonstrates that sources of uncertainty are miscellaneous and can not always be decomposed into aleatoric and epistemic. Drawing parallels between statistical concepts and uncertainty in machine learning, we also demonstrate the rol
    
[^6]: QuantumNAT：注重量子噪声的噪声注入、量化和归一化的量子训练

    QuantumNAT: Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization. (arXiv:2110.11331v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2110.11331](http://arxiv.org/abs/2110.11331)

    QuantumNAT是一个PQC特定框架，可以在训练和推断阶段执行噪声感知优化，提高鲁棒性，缓解量子噪声

    

    参数化量子电路是实现近期量子硬件优势的有希望方法。然而，由于存在较大的量子噪声（误差），在实际的量子设备上，PQC模型的性能会受到严重的降级。我们提出了QuantumNAT，一个可以在训练和推断阶段执行噪声感知优化的PQC特定框架，以提高其鲁棒性。通过实验我们发现，量子噪声对PQC测量结果的影响是从无噪声结果经过一个缩放和偏移因子得到的线性映射。基于此，我们提出了后测量归一化来缓解特征分布不一致的问题。

    Parameterized Quantum Circuits (PQC) are promising towards quantum advantage on near-term quantum hardware. However, due to the large quantum noises (errors), the performance of PQC models has a severe degradation on real quantum devices. Take Quantum Neural Network (QNN) as an example, the accuracy gap between noise-free simulation and noisy results on IBMQ-Yorktown for MNIST-4 classification is over 60%. Existing noise mitigation methods are general ones without leveraging unique characteristics of PQC; on the other hand, existing PQC work does not consider noise effect. To this end, we present QuantumNAT, a PQC-specific framework to perform noise-aware optimizations in both training and inference stages to improve robustness. We experimentally observe that the effect of quantum noise to PQC measurement outcome is a linear map from noise-free outcome with a scaling and a shift factor. Motivated by that, we propose post-measurement normalization to mitigate the feature distribution di
    

