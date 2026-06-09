# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Are Classification Robustness and Explanation Robustness Really Strongly Correlated? An Analysis Through Input Loss Landscape](https://arxiv.org/abs/2403.06013) | 通过新颖的评估方法和训练方法，本研究发现增强解释鲁棒性并不能提高分类鲁棒性，这一发现挑战了传统观念。 |
| [^2] | [Investigating the Histogram Loss in Regression](https://arxiv.org/abs/2402.13425) | 学习整个分布在回归中的性能提升主要来自于优化的改进，而不是学习更好的表示。 |
| [^3] | [The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes](https://arxiv.org/abs/2402.08922) | 本文介绍和探讨了镜像影响假设，突出了训练和测试数据之间影响的相互性。具体而言，它指出，评估训练数据对测试预测的影响可以重新表述为一个等效但相反的问题：评估如果模型在特定的测试样本上进行训练，对训练样本的预测将如何改变。通过实证和理论验证，我们演示了这一假设的正确性。 |
| [^4] | [Generalization Error Curves for Analytic Spectral Algorithms under Power-law Decay.](http://arxiv.org/abs/2401.01599) | 本文研究了核回归方法的泛化误差曲线，对核梯度下降方法和其他分析谱算法在核回归中的泛化误差进行了全面特征化，从而提高了对训练宽神经网络泛化行为的理解，并提出了一种新的技术贡献-分析功能论证。 |
| [^5] | [Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook.](http://arxiv.org/abs/2310.10196) | 这篇综述探讨了大型模型在时间序列和时空数据中的应用。它们不仅带来了增强的模式识别和推理能力，还为人工通用智能打下了基础。 |
| [^6] | [Deep reinforcement learning for process design: Review and perspective.](http://arxiv.org/abs/2308.07822) | 深度强化学习在流程设计中的综述和展望。该研究调查了强化学习在流程设计中的最新研究，并探讨了潜在挑战和未来工作，以充分发挥其在化学工程中的潜力。 |
| [^7] | [TAMUNA: Doubly Accelerated Federated Learning with Local Training, Compression, and Partial Participation.](http://arxiv.org/abs/2302.09832) | TAMUNA是首个联合利用网络压缩和少量通信配合加速分布式梯度下降算法，并允许部分参与的算法。 |

# 详细

[^1]: 分类鲁棒性和解释鲁棒性是否真的强相关？通过输入损失景观的分析

    Are Classification Robustness and Explanation Robustness Really Strongly Correlated? An Analysis Through Input Loss Landscape

    [https://arxiv.org/abs/2403.06013](https://arxiv.org/abs/2403.06013)

    通过新颖的评估方法和训练方法，本研究发现增强解释鲁棒性并不能提高分类鲁棒性，这一发现挑战了传统观念。

    

    本文深入探讨了深度学习鲁棒性领域，挑战了传统观念，即图像分类系统中的分类鲁棒性和解释鲁棒性本质上是相关的。通过一种新颖的评估方法，利用聚类来有效评估解释鲁棒性，我们展示了增强解释鲁棒性并不一定会使输入损失景观相对于解释损失变平 - 与损失景观变平表示更好的分类鲁棒性相反。为了深入研究这一矛盾，提出了一种突破性的训练方法，旨在调整相对于解释损失的损失景观。通过这种新的训练方法，我们发现虽然这种调整可以影响解释的鲁棒性，但它们对分类的鲁棒性没有影响。这些发现不仅挑战了流行的观念

    arXiv:2403.06013v1 Announce Type: new  Abstract: This paper delves into the critical area of deep learning robustness, challenging the conventional belief that classification robustness and explanation robustness in image classification systems are inherently correlated. Through a novel evaluation approach leveraging clustering for efficient assessment of explanation robustness, we demonstrate that enhancing explanation robustness does not necessarily flatten the input loss landscape with respect to explanation loss - contrary to flattened loss landscapes indicating better classification robustness. To deeply investigate this contradiction, a groundbreaking training method designed to adjust the loss landscape with respect to explanation loss is proposed. Through the new training method, we uncover that although such adjustments can impact the robustness of explanations, they do not have an influence on the robustness of classification. These findings not only challenge the prevailing 
    
[^2]: 在回归中探讨直方图损失

    Investigating the Histogram Loss in Regression

    [https://arxiv.org/abs/2402.13425](https://arxiv.org/abs/2402.13425)

    学习整个分布在回归中的性能提升主要来自于优化的改进，而不是学习更好的表示。

    

    越来越常见的是，在回归中训练神经网络来建模整个分布，即使只需要均值来进行预测。 这种额外的建模通常会带来性能增益，但背后的原因尚不完全清楚。 本文研究了回归中的一种最新方法，即直方图损失，该方法通过最小化目标分布和灵活直方图预测之间的交叉熵来学习目标变量的条件分布。 我们设计了理论和实证分析，以确定为什么以及何时会出现性能增益，以及损失的不同组件如何为此做出贡献。 我们的结果表明，在这种设置中学习分布的好处来自于优化的改进，而不是学习更好的表示。 然后，我们展示了直方图损失在常见的深度学习应用中的可行性。

    arXiv:2402.13425v1 Announce Type: cross  Abstract: It is becoming increasingly common in regression to train neural networks that model the entire distribution even if only the mean is required for prediction. This additional modeling often comes with performance gain and the reasons behind the improvement are not fully known. This paper investigates a recent approach to regression, the Histogram Loss, which involves learning the conditional distribution of the target variable by minimizing the cross-entropy between a target distribution and a flexible histogram prediction. We design theoretical and empirical analyses to determine why and when this performance gain appears, and how different components of the loss contribute to it. Our results suggest that the benefits of learning distributions in this setup come from improvements in optimization rather than learning a better representation. We then demonstrate the viability of the Histogram Loss in common deep learning applications wi
    
[^3]: 镜像影响假设：通过利用前向传递实现高效的数据影响估计

    The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes

    [https://arxiv.org/abs/2402.08922](https://arxiv.org/abs/2402.08922)

    本文介绍和探讨了镜像影响假设，突出了训练和测试数据之间影响的相互性。具体而言，它指出，评估训练数据对测试预测的影响可以重新表述为一个等效但相反的问题：评估如果模型在特定的测试样本上进行训练，对训练样本的预测将如何改变。通过实证和理论验证，我们演示了这一假设的正确性。

    

    大规模黑盒模型已经在许多应用中变得无处不在。了解个别训练数据源对这些模型所做预测的影响对于改善其可信性至关重要。当前的影响评估技术涉及计算每个训练点的梯度或在不同子集上重复训练。当扩展到大规模数据集和模型时，这些方法面临明显的计算挑战。

    arXiv:2402.08922v1 Announce Type: new Abstract: Large-scale black-box models have become ubiquitous across numerous applications. Understanding the influence of individual training data sources on predictions made by these models is crucial for improving their trustworthiness. Current influence estimation techniques involve computing gradients for every training point or repeated training on different subsets. These approaches face obvious computational challenges when scaled up to large datasets and models.   In this paper, we introduce and explore the Mirrored Influence Hypothesis, highlighting a reciprocal nature of influence between training and test data. Specifically, it suggests that evaluating the influence of training data on test predictions can be reformulated as an equivalent, yet inverse problem: assessing how the predictions for training samples would be altered if the model were trained on specific test samples. Through both empirical and theoretical validations, we demo
    
[^4]: 分析谱算法在幂律衰减下的泛化误差曲线

    Generalization Error Curves for Analytic Spectral Algorithms under Power-law Decay. (arXiv:2401.01599v1 [cs.LG])

    [http://arxiv.org/abs/2401.01599](http://arxiv.org/abs/2401.01599)

    本文研究了核回归方法的泛化误差曲线，对核梯度下降方法和其他分析谱算法在核回归中的泛化误差进行了全面特征化，从而提高了对训练宽神经网络泛化行为的理解，并提出了一种新的技术贡献-分析功能论证。

    

    某些核回归方法的泛化误差曲线旨在确定在不同源条件、噪声水平和正则化参数选择下的泛化误差的确切顺序，而不是最小化率。在本文中，在温和的假设下，我们严格给出了核梯度下降方法（以及大类分析谱算法）在核回归中的泛化误差曲线的完整特征化。因此，我们可以提高核插值的近不一致性，并澄清具有更高资格的核回归算法的饱和效应，等等。由于神经切线核理论的帮助，这些结果极大地提高了我们对训练宽神经网络的泛化行为的理解。一种新颖的技术贡献，即分析功能论证，可能具有独立的兴趣。

    The generalization error curve of certain kernel regression method aims at determining the exact order of generalization error with various source condition, noise level and choice of the regularization parameter rather than the minimax rate. In this work, under mild assumptions, we rigorously provide a full characterization of the generalization error curves of the kernel gradient descent method (and a large class of analytic spectral algorithms) in kernel regression. Consequently, we could sharpen the near inconsistency of kernel interpolation and clarify the saturation effects of kernel regression algorithms with higher qualification, etc. Thanks to the neural tangent kernel theory, these results greatly improve our understanding of the generalization behavior of training the wide neural networks. A novel technical contribution, the analytic functional argument, might be of independent interest.
    
[^5]: 时间序列和时空数据的大型模型：综述与展望

    Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook. (arXiv:2310.10196v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.10196](http://arxiv.org/abs/2310.10196)

    这篇综述探讨了大型模型在时间序列和时空数据中的应用。它们不仅带来了增强的模式识别和推理能力，还为人工通用智能打下了基础。

    

    时间数据，特别是时间序列和时空数据，在现实世界的应用中非常普遍。它们捕捉动态系统的测量数据，并由物理和虚拟传感器大量产生。分析这些数据类型对于利用它们所包含的丰富信息以及受益于各种下游任务非常重要。近年来，大型语言和其他基础模型的突破推动了这些模型在时间序列和时空数据挖掘中的增加使用。这样的方法不仅能够实现跨不同领域的增强模式识别和推理，还为能够理解和处理常见时间数据的人工通用智能打下了基础。在此综述中，我们提供了针对时间序列和时空数据定制的大型模型的全面和最新综述，包括数据类型、模型类别、模型范围和应用领域/任务。我们的目标是

    Temporal data, notably time series and spatio-temporal data, are prevalent in real-world applications. They capture dynamic system measurements and are produced in vast quantities by both physical and virtual sensors. Analyzing these data types is vital to harnessing the rich information they encompass and thus benefits a wide range of downstream tasks. Recent advances in large language and other foundational models have spurred increased use of these models in time series and spatio-temporal data mining. Such methodologies not only enable enhanced pattern recognition and reasoning across diverse domains but also lay the groundwork for artificial general intelligence capable of comprehending and processing common temporal data. In this survey, we offer a comprehensive and up-to-date review of large models tailored (or adapted) for time series and spatio-temporal data, spanning four key facets: data types, model categories, model scopes, and application areas/tasks. Our objective is to 
    
[^6]: 深度强化学习在流程设计中的应用: 综述与展望

    Deep reinforcement learning for process design: Review and perspective. (arXiv:2308.07822v1 [cs.LG])

    [http://arxiv.org/abs/2308.07822](http://arxiv.org/abs/2308.07822)

    深度强化学习在流程设计中的综述和展望。该研究调查了强化学习在流程设计中的最新研究，并探讨了潜在挑战和未来工作，以充分发挥其在化学工程中的潜力。

    

    化学工业向可再生能源和原料供应的转型需要新的概念性流程设计方法。最近，人工智能方面取得的突破为加速这一转变提供了机会。具体而言，深度强化学习作为机器学习的一个子类，已经展示出解决复杂决策问题和帮助可持续流程设计的潜力。我们通过三个主要要素对流程设计中强化学习的最新研究进行了综述：（i）信息表示、（ii）代理架构，以及（iii）环境和奖励。此外，我们讨论了潜在挑战和有前景的未来工作，以充分发挥强化学习在化学工程的流程设计中的潜力。

    The transformation towards renewable energy and feedstock supply in the chemical industry requires new conceptual process design approaches. Recently, breakthroughs in artificial intelligence offer opportunities to accelerate this transition. Specifically, deep reinforcement learning, a subclass of machine learning, has shown the potential to solve complex decision-making problems and aid sustainable process design. We survey state-of-the-art research in reinforcement learning for process design through three major elements: (i) information representation, (ii) agent architecture, and (iii) environment and reward. Moreover, we discuss perspectives on underlying challenges and promising future works to unfold the full potential of reinforcement learning for process design in chemical engineering.
    
[^7]: TAMUNA: 带有局部训练、压缩和部分参与的双倍加速联邦学习

    TAMUNA: Doubly Accelerated Federated Learning with Local Training, Compression, and Partial Participation. (arXiv:2302.09832v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09832](http://arxiv.org/abs/2302.09832)

    TAMUNA是首个联合利用网络压缩和少量通信配合加速分布式梯度下降算法，并允许部分参与的算法。

    

    在联邦学习中，大量用户合作学习全局模型。他们交替进行本地计算和与远程服务器的通信。通信是该设置中的主要瓶颈，它可以慢且昂贵。为了减少通信负载并加速分布式梯度下降，使用两种策略很受欢迎：1）更少地通信，即在通信轮之间执行几个本地计算的迭代；2）传输压缩信息而不是完整维度的矢量。我们提出了TAMUNA，这是第一个分布式优化和联邦学习算法，它联合利用这两种策略，同时允许部分参与。TAMUNA以线性速度收敛到精确解决方案。

    In federated learning, a large number of users collaborate to learn a global model. They alternate local computations and communication with a distant server. Communication, which can be slow and costly, is the main bottleneck in this setting. In addition to communication-efficiency, a robust algorithm should allow for partial participation, the desirable feature that not all clients need to participate to every round of the training process. To reduce the communication load and therefore accelerate distributed gradient descent, two strategies are popular: 1) communicate less frequently; that is, perform several iterations of local computations between the communication rounds; and 2) communicate compressed information instead of full-dimensional vectors. We propose TAMUNA, the first algorithm for distributed optimization and federated learning, which harnesses these two strategies jointly and allows for partial participation. TAMUNA converges linearly to an exact solution in the stron
    

