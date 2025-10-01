# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bounding Reconstruction Attack Success of Adversaries Without Data Priors](https://arxiv.org/abs/2402.12861) | 本研究提供了差分隐私训练的机器学习模型在现实对抗设置下重建成功率的正式上限，并通过实证结果支持，有助于更明智地选择隐私参数。 |
| [^2] | [Complexity Reduction in Machine Learning-Based Wireless Positioning: Minimum Description Features](https://arxiv.org/abs/2402.09580) | 本文设计了一种定位神经网络（P-NN），通过最小描述特征降低了基于深度学习的无线定位中的复杂度，并开发了一种新的方法来自适应地选择特征空间的大小。 |
| [^3] | [Collective Counterfactual Explanations via Optimal Transport](https://arxiv.org/abs/2402.04579) | 本论文提出了一种集体方法来形成反事实解释，通过利用个体的当前密度来指导推荐的行动，解决了个体为中心的方法可能导致的新的竞争和意想不到的成本问题，并改进了经典反事实解释的期望。 |
| [^4] | [Unlearnable Algorithms for In-context Learning](https://arxiv.org/abs/2402.00751) | 本文提出了一种针对预先训练的大型语言模型的高效去学习方法，通过选择少量训练示例来实现任务适应训练数据的精确去学习，并与微调方法进行了比较和讨论。 |
| [^5] | [Bengali Document Layout Analysis -- A YOLOV8 Based Ensembling Approach.](http://arxiv.org/abs/2309.00848) | 本文提出了一种基于YOLOv8模型和创新的后处理技术的孟加拉文档布局分析方法，通过数据增强和两阶段预测策略实现了准确的元素分割。该方法优于单个基础架构，并解决了BaDLAD数据集中的问题，有助于提高OCR和文档理解能力。 |
| [^6] | [RobustNeuralNetworks.jl: a Package for Machine Learning and Data-Driven Control with Certified Robustness.](http://arxiv.org/abs/2306.12612) | RobustNeuralNetworks.jl是一个用Julia编写的机器学习和数据驱动控制包，它通过自然满足用户定义的鲁棒性约束条件，实现了神经网络模型的构建。 |
| [^7] | [Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance.](http://arxiv.org/abs/2304.11127) | 该论文介绍了一种广泛使用的贝叶斯优化方法 Tree-structured Parzen estimator (TPE)，并对其控制参数的作用和算法直觉进行了讨论和分析，提供了一组推荐设置并证明其能够提高TPE的性能表现。 |
| [^8] | [Efficient Utility Function Learning for Multi-Objective Parameter Optimization with Prior Knowledge.](http://arxiv.org/abs/2208.10300) | 该论文提出了一种利用偏好学习离线学习效用函数的方法，以应对真实世界问题中用专家知识定义效用函数困难且与专家反复互动昂贵的问题。使用效用函数空间的粗略信息，能够在使用很少结果时提高效用函数估计，并通过整个优化链中传递效用函数学习任务中出现的不确定性。 |

# 详细

[^1]: 在没有数据先验条件下限制对抗者重建攻击成功率

    Bounding Reconstruction Attack Success of Adversaries Without Data Priors

    [https://arxiv.org/abs/2402.12861](https://arxiv.org/abs/2402.12861)

    本研究提供了差分隐私训练的机器学习模型在现实对抗设置下重建成功率的正式上限，并通过实证结果支持，有助于更明智地选择隐私参数。

    

    机器学习模型的重建攻击存在泄漏敏感数据的风险。在特定情境下，对手可以使用模型的梯度几乎完美地重建训练数据样本。在使用差分隐私（DP）训练机器学习模型时，可以提供对这种重建攻击成功率的正式上限。迄今为止，这些上限是在可能不符合高度现实实用性的最坏情况假设下制定的。在本文中，我们针对差分隐私训练的机器学习模型提供了在现实对抗设置下的重建成功率正式上限，并通过实证结果支持这些上限。通过这一点，我们展示了在现实情境中，（a）预期的重建成功率可以在不同背景和不同度量下得到适当的限制，这（b）有助于更明智地选择隐私参数。

    arXiv:2402.12861v1 Announce Type: new  Abstract: Reconstruction attacks on machine learning (ML) models pose a strong risk of leakage of sensitive data. In specific contexts, an adversary can (almost) perfectly reconstruct training data samples from a trained model using the model's gradients. When training ML models with differential privacy (DP), formal upper bounds on the success of such reconstruction attacks can be provided. So far, these bounds have been formulated under worst-case assumptions that might not hold high realistic practicality. In this work, we provide formal upper bounds on reconstruction success under realistic adversarial settings against ML models trained with DP and support these bounds with empirical results. With this, we show that in realistic scenarios, (a) the expected reconstruction success can be bounded appropriately in different contexts and by different metrics, which (b) allows for a more educated choice of a privacy parameter.
    
[^2]: 基于机器学习的无线定位中的复杂度降低：最小描述特征

    Complexity Reduction in Machine Learning-Based Wireless Positioning: Minimum Description Features

    [https://arxiv.org/abs/2402.09580](https://arxiv.org/abs/2402.09580)

    本文设计了一种定位神经网络（P-NN），通过最小描述特征降低了基于深度学习的无线定位中的复杂度，并开发了一种新的方法来自适应地选择特征空间的大小。

    

    最近的一系列研究一直致力于基于深度学习的无线定位（WP）。尽管这些WP算法在不同信道条件下表现出了高精度和鲁棒性，但它们也存在一个主要缺点：它们需要处理高维特征，这对于移动应用来说可能是禁止的。在本工作中，我们设计了一个定位神经网络（P-NN），通过精心设计的最小描述特征，大大降低了基于深度学习的WP的复杂度。我们的特征选择基于最大功率测量及其时间位置，以传达进行WP所需的信息。我们还开发了一种新的方法来自适应地选择特征空间的大小，该方法通过在信号二进制选择上使用信息论度量，优化了期望有用信息量和分类能力之间的平衡。

    arXiv:2402.09580v1 Announce Type: new  Abstract: A recent line of research has been investigating deep learning approaches to wireless positioning (WP). Although these WP algorithms have demonstrated high accuracy and robust performance against diverse channel conditions, they also have a major drawback: they require processing high-dimensional features, which can be prohibitive for mobile applications. In this work, we design a positioning neural network (P-NN) that substantially reduces the complexity of deep learning-based WP through carefully crafted minimum description features. Our feature selection is based on maximum power measurements and their temporal locations to convey information needed to conduct WP. We also develop a novel methodology for adaptively selecting the size of feature space, which optimizes over balancing the expected amount of useful information and classification capability, quantified using information-theoretic measures on the signal bin selection. Numeri
    
[^3]: 通过最优传输实现集体反事实解释

    Collective Counterfactual Explanations via Optimal Transport

    [https://arxiv.org/abs/2402.04579](https://arxiv.org/abs/2402.04579)

    本论文提出了一种集体方法来形成反事实解释，通过利用个体的当前密度来指导推荐的行动，解决了个体为中心的方法可能导致的新的竞争和意想不到的成本问题，并改进了经典反事实解释的期望。

    

    反事实解释提供个体的成本最优行动，以改变标签为所需的类别。然而，如果大量实例寻求状态修改，这种个体为中心的方法可能导致新的竞争和意想不到的成本。此外，这些推荐忽视了基础数据分布，可能会建议用户认为是异常值的行动。为了解决这些问题，我们的工作提出了一种集体方法来形成反事实解释，重点是利用个体的当前密度来指导推荐的行动。我们的问题自然地转化为一个最优传输问题。借鉴最优传输的广泛文献，我们说明了这种集体方法如何改进经典反事实解释的期望。我们通过数值模拟支持我们的提议，展示了所提方法的有效性以及与经典方法的关系。

    Counterfactual explanations provide individuals with cost-optimal actions that can alter their labels to desired classes. However, if substantial instances seek state modification, such individual-centric methods can lead to new competitions and unanticipated costs. Furthermore, these recommendations, disregarding the underlying data distribution, may suggest actions that users perceive as outliers. To address these issues, our work proposes a collective approach for formulating counterfactual explanations, with an emphasis on utilizing the current density of the individuals to inform the recommended actions. Our problem naturally casts as an optimal transport problem. Leveraging the extensive literature on optimal transport, we illustrate how this collective method improves upon the desiderata of classical counterfactual explanations. We support our proposal with numerical simulations, illustrating the effectiveness of the proposed approach and its relation to classic methods.
    
[^4]: 无法学习的算法用于上下文学习

    Unlearnable Algorithms for In-context Learning

    [https://arxiv.org/abs/2402.00751](https://arxiv.org/abs/2402.00751)

    本文提出了一种针对预先训练的大型语言模型的高效去学习方法，通过选择少量训练示例来实现任务适应训练数据的精确去学习，并与微调方法进行了比较和讨论。

    

    随着模型被越来越多地部署在未知来源的数据上，机器去学习变得越来越受欢迎。然而，要实现精确的去学习——在没有使用要遗忘的数据的情况下获得与模型分布匹配的模型——是具有挑战性或低效的，通常需要大量的重新训练。在本文中，我们专注于预先训练的大型语言模型（LLM）的任务适应阶段的高效去学习方法。我们观察到LLM进行任务适应的上下文学习能力可以实现任务适应训练数据的高效精确去学习。我们提供了一种算法，用于选择少量训练示例加到LLM的提示前面（用于任务适应），名为ERASE，它的去学习操作成本与模型和数据集的大小无关，意味着它适用于大型模型和数据集。我们还将我们的方法与微调方法进行了比较，并讨论了两种方法之间的权衡。这使我们得到了以下结论：

    Machine unlearning is a desirable operation as models get increasingly deployed on data with unknown provenance. However, achieving exact unlearning -- obtaining a model that matches the model distribution when the data to be forgotten was never used -- is challenging or inefficient, often requiring significant retraining. In this paper, we focus on efficient unlearning methods for the task adaptation phase of a pretrained large language model (LLM). We observe that an LLM's ability to do in-context learning for task adaptation allows for efficient exact unlearning of task adaptation training data. We provide an algorithm for selecting few-shot training examples to prepend to the prompt given to an LLM (for task adaptation), ERASE, whose unlearning operation cost is independent of model and dataset size, meaning it scales to large models and datasets. We additionally compare our approach to fine-tuning approaches and discuss the trade-offs between the two approaches. This leads us to p
    
[^5]: 孟加拉文档布局分析-一种基于YOLOv8的集成方法

    Bengali Document Layout Analysis -- A YOLOV8 Based Ensembling Approach. (arXiv:2309.00848v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.00848](http://arxiv.org/abs/2309.00848)

    本文提出了一种基于YOLOv8模型和创新的后处理技术的孟加拉文档布局分析方法，通过数据增强和两阶段预测策略实现了准确的元素分割。该方法优于单个基础架构，并解决了BaDLAD数据集中的问题，有助于提高OCR和文档理解能力。

    

    本文侧重于利用YOLOv8模型和创新的后处理技术提升孟加拉文档布局分析（DLA）。我们通过数据增强以应对孟加拉复杂文字独特的挑战，经过严格的验证集评估，对完整数据集进行微调，实现准确的元素分割的两阶段预测策略。我们的集成模型结合后处理性能优于单个基础架构，解决了BaDLAD数据集中的问题。通过利用这种方法，我们旨在推动孟加拉文档分析的发展，提高OCR和文档理解能力，同时BaDLAD作为基础资源有助于未来的研究。此外，我们的实验为将新策略纳入现有解决方案提供了关键见解。

    This paper focuses on enhancing Bengali Document Layout Analysis (DLA) using the YOLOv8 model and innovative post-processing techniques. We tackle challenges unique to the complex Bengali script by employing data augmentation for model robustness. After meticulous validation set evaluation, we fine-tune our approach on the complete dataset, leading to a two-stage prediction strategy for accurate element segmentation. Our ensemble model, combined with post-processing, outperforms individual base architectures, addressing issues identified in the BaDLAD dataset. By leveraging this approach, we aim to advance Bengali document analysis, contributing to improved OCR and document comprehension and BaDLAD serves as a foundational resource for this endeavor, aiding future research in the field. Furthermore, our experiments provided key insights to incorporate new strategies into the established solution.
    
[^6]: RobustNeuralNetworks.jl：带有认证鲁棒性的机器学习和数据驱动控制包。

    RobustNeuralNetworks.jl: a Package for Machine Learning and Data-Driven Control with Certified Robustness. (arXiv:2306.12612v1 [cs.LG])

    [http://arxiv.org/abs/2306.12612](http://arxiv.org/abs/2306.12612)

    RobustNeuralNetworks.jl是一个用Julia编写的机器学习和数据驱动控制包，它通过自然满足用户定义的鲁棒性约束条件，实现了神经网络模型的构建。

    

    神经网络通常对于微小的输入扰动非常敏感，导致出现意外或脆弱的行为。本文介绍了RobustNeuralNetworks.jl：一个Julia包，用于构建神经网络模型，该模型自然地满足一组用户定义的鲁棒性约束条件。该包基于最近提出的Recurrent Equilibrium Network（REN）和Lipschitz-Bounded Deep Network（LBDN）模型类，并旨在直接与Julia最广泛使用的机器学习包Flux.jl接口。我们讨论了模型参数化背后的理论，概述了该包，并提供了一个教程，演示了其在图像分类、强化学习和非线性状态观测器设计中的应用。

    Neural networks are typically sensitive to small input perturbations, leading to unexpected or brittle behaviour. We present RobustNeuralNetworks.jl: a Julia package for neural network models that are constructed to naturally satisfy a set of user-defined robustness constraints. The package is based on the recently proposed Recurrent Equilibrium Network (REN) and Lipschitz-Bounded Deep Network (LBDN) model classes, and is designed to interface directly with Julia's most widely-used machine learning package, Flux.jl. We discuss the theory behind our model parameterization, give an overview of the package, and provide a tutorial demonstrating its use in image classification, reinforcement learning, and nonlinear state-observer design.
    
[^7]: 树状Parzen估计器：理解其算法组成部分及其在提高实证表现中的作用

    Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance. (arXiv:2304.11127v1 [cs.LG])

    [http://arxiv.org/abs/2304.11127](http://arxiv.org/abs/2304.11127)

    该论文介绍了一种广泛使用的贝叶斯优化方法 Tree-structured Parzen estimator (TPE)，并对其控制参数的作用和算法直觉进行了讨论和分析，提供了一组推荐设置并证明其能够提高TPE的性能表现。

    

    许多领域中最近的进展要求更加复杂的实验设计。这种复杂的实验通常有许多参数，需要参数调整。Tree-structured Parzen estimator (TPE) 是一种贝叶斯优化方法，在最近的参数调整框架中被广泛使用。尽管它很受欢迎，但控制参数的角色和算法直觉尚未得到讨论。在本教程中，我们将确定每个控制参数的作用以及它们对超参数优化的影响，使用多种基准测试。我们将从剖析研究中得出的推荐设置与基准方法进行比较，并证明我们的推荐设置提高了TPE的性能。我们的TPE实现可在https://github.com/nabenabe0928/tpe/tree/single-opt中获得。

    Recent advances in many domains require more and more complicated experiment design. Such complicated experiments often have many parameters, which necessitate parameter tuning. Tree-structured Parzen estimator (TPE), a Bayesian optimization method, is widely used in recent parameter tuning frameworks. Despite its popularity, the roles of each control parameter and the algorithm intuition have not been discussed so far. In this tutorial, we will identify the roles of each control parameter and their impacts on hyperparameter optimization using a diverse set of benchmarks. We compare our recommended setting drawn from the ablation study with baseline methods and demonstrate that our recommended setting improves the performance of TPE. Our TPE implementation is available at https://github.com/nabenabe0928/tpe/tree/single-opt.
    
[^8]: 多目标参数优化中的有效效用函数学习与先验知识

    Efficient Utility Function Learning for Multi-Objective Parameter Optimization with Prior Knowledge. (arXiv:2208.10300v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.10300](http://arxiv.org/abs/2208.10300)

    该论文提出了一种利用偏好学习离线学习效用函数的方法，以应对真实世界问题中用专家知识定义效用函数困难且与专家反复互动昂贵的问题。使用效用函数空间的粗略信息，能够在使用很少结果时提高效用函数估计，并通过整个优化链中传递效用函数学习任务中出现的不确定性。

    

    目前的多目标优化技术通常假定已有效用函数、通过互动学习效用函数或尝试确定完整的Pareto前沿来进行。然而，在真实世界的问题中，结果往往基于隐含和显性的专家知识，难以定义一个效用函数，而互动学习或后续启发式需要反复并且昂贵地专家参与。为了缓解这种情况，我们使用偏好学习离线学习效用函数，利用专家知识。与其他工作不同的是，我们不仅使用（成对的）结果偏好，而且使用效用函数空间的粗略信息。这使我们能够提高效用函数估计，特别是在使用很少的结果时。此外，我们对效用函数学习任务中出现的不确定性进行建模，并将其传递到整个优化链中。

    The current state-of-the-art in multi-objective optimization assumes either a given utility function, learns a utility function interactively or tries to determine the complete Pareto front, requiring a post elicitation of the preferred result. However, result elicitation in real world problems is often based on implicit and explicit expert knowledge, making it difficult to define a utility function, whereas interactive learning or post elicitation requires repeated and expensive expert involvement. To mitigate this, we learn a utility function offline, using expert knowledge by means of preference learning. In contrast to other works, we do not only use (pairwise) result preferences, but also coarse information about the utility function space. This enables us to improve the utility function estimate, especially when using very few results. Additionally, we model the occurring uncertainties in the utility function learning task and propagate them through the whole optimization chain. 
    

