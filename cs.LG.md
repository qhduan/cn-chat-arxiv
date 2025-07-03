# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty-aware Distributional Offline Reinforcement Learning](https://arxiv.org/abs/2403.17646) | 提出了一种不确定性感知的分布式离线强化学习方法，同时解决认知不确定性和环境随机性，在风险敏感和规避设置下进行了全面实验评估 |
| [^2] | [Co-Optimization of Environment and Policies for Decentralized Multi-Agent Navigation](https://arxiv.org/abs/2403.14583) | 将多智能体系统和周围环境视为共同演化的系统，提出智能体-环境协同优化问题并开发协调算法，以改进去中心化多智能体导航表现。 |
| [^3] | [Diffusion-based Iterative Counterfactual Explanations for Fetal Ultrasound Image Quality Assessment](https://arxiv.org/abs/2403.08700) | 提出了基于扩散的迭代反事实解释的方法，通过生成逼真的高质量标准平面，对提高临床医生的培训、改善图像质量以及提升下游诊断和监测具有潜在价值。 |
| [^4] | [Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation](https://arxiv.org/abs/2403.06759) | 提出一种平均L1校准误差（mL1-ACE）作为辅助损失函数，用于改善图像分割中的像素级校准，减少了校准误差并引入了数据集可靠性直方图以提高校准评估。 |
| [^5] | [Vehicle-group-based Crash Risk Formation and Propagation Analysis for Expressways](https://arxiv.org/abs/2402.12415) | 本研究基于车辆组作为分析对象，探讨了考虑车辆组和道路段特征的风险形成和传播机制，识别出影响碰撞风险的关键因素。 |
| [^6] | [SpikeNAS: A Fast Memory-Aware Neural Architecture Search Framework for Spiking Neural Network Systems](https://arxiv.org/abs/2402.11322) | SpikeNAS提出了一种快速内存感知神经架构搜索框架，旨在帮助脉冲神经网络系统快速找到在给定内存预算下高准确性的适当架构。 |
| [^7] | [EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge](https://arxiv.org/abs/2402.10787) | 本文提出了EdgeQAT，使用熵和分布引导的量化感知训练方法来优化轻量级LLMs，在边缘设备上实现推理加速。 |
| [^8] | [Role of Momentum in Smoothing Objective Function in Implicit Graduated Optimization](https://arxiv.org/abs/2402.02325) | 这篇论文揭示了具有动量的随机梯度下降算法平滑了目标函数，影响程度由多个超参数决定，同时提供了对动量改善泛化能力的理论解释和新见解。 |
| [^9] | [Dataset Distillation via the Wasserstein Metric](https://arxiv.org/abs/2311.18531) | 通过引入Wasserstein距离及其重心，我们提出一种有效的数据集精炼方法，利用先验知识提高分布匹配效果，实现了新的最先进性能。 |
| [^10] | [On the Lipschitz constant of random neural networks.](http://arxiv.org/abs/2311.01356) | 本文研究了随机ReLU神经网络的Lipschitz常数，对于浅层神经网络，我们得到了Lipschitz常数的精确刻画，对于足够宽度的深层神经网络，我们给出了上下界，并匹配一个依赖于深度的对数因子。 |
| [^11] | [Feature Reweighting for EEG-based Motor Imagery Classification.](http://arxiv.org/abs/2308.02515) | 本论文提出了一种特征重加权的方法，用于解决使用EEG信号进行运动想象分类时存在的低信噪比、非稳态性、非线性和复杂性等挑战，通过降低噪声和无关信息，提高分类性能。 |

# 详细

[^1]: 不确定性感知的分布式离线强化学习

    Uncertainty-aware Distributional Offline Reinforcement Learning

    [https://arxiv.org/abs/2403.17646](https://arxiv.org/abs/2403.17646)

    提出了一种不确定性感知的分布式离线强化学习方法，同时解决认知不确定性和环境随机性，在风险敏感和规避设置下进行了全面实验评估

    

    离线强化学习面临独特挑战，因其仅依赖于观测数据。在这一背景下中心关注点是通过量化与各种行动和环境随机性相关的不确定性，确保所学策略的安全性。传统方法主要强调通过学习风险规避策略来缓解认知不确定性，往往忽视环境随机性。在本研究中，我们提出了一种不确定性感知的分布式离线强化学习方法，以同时处理认知不确定性和环境随机性。我们提出了一种能够学习风险规避策略并表征折现累积奖励的整个分布的无模型离线强化学习算法，而不仅仅是最大化累积折现回报的期望值。我们的方法通过在风险敏感和风险规避设置下的全面实验得到严格评估。

    arXiv:2403.17646v1 Announce Type: new  Abstract: Offline reinforcement learning (RL) presents distinct challenges as it relies solely on observational data. A central concern in this context is ensuring the safety of the learned policy by quantifying uncertainties associated with various actions and environmental stochasticity. Traditional approaches primarily emphasize mitigating epistemic uncertainty by learning risk-averse policies, often overlooking environmental stochasticity. In this study, we propose an uncertainty-aware distributional offline RL method to simultaneously address both epistemic uncertainty and environmental stochasticity. We propose a model-free offline RL algorithm capable of learning risk-averse policies and characterizing the entire distribution of discounted cumulative rewards, as opposed to merely maximizing the expected value of accumulated discounted returns. Our method is rigorously evaluated through comprehensive experiments in both risk-sensitive and ri
    
[^2]: 为去中心化多智能体导航的环境和政策进行协同优化

    Co-Optimization of Environment and Policies for Decentralized Multi-Agent Navigation

    [https://arxiv.org/abs/2403.14583](https://arxiv.org/abs/2403.14583)

    将多智能体系统和周围环境视为共同演化的系统，提出智能体-环境协同优化问题并开发协调算法，以改进去中心化多智能体导航表现。

    

    这项工作将多智能体系统及其周围环境视为一个共同演化的系统，其中一个的行为会影响另一个。其目标是将智能体行为和环境配置都视为决策变量，并以协调的方式优化这两个组件，以改进某些感兴趣的度量。为此，我们考虑了在拥挤环境中的去中心化多智能体导航问题。通过引入多智能体导航和环境优化的两个子目标，提出了一个“智能体-环境协同优化”问题，并开发了一个“协调算法”，在这两个子目标之间交替以寻找智能体行为和障碍物环境配置的最佳综合；最终提高了导航性能。由于明确建模智能体、环境和性能之间关系的挑战，我们利用了

    arXiv:2403.14583v1 Announce Type: cross  Abstract: This work views the multi-agent system and its surrounding environment as a co-evolving system, where the behavior of one affects the other. The goal is to take both agent actions and environment configurations as decision variables, and optimize these two components in a coordinated manner to improve some measure of interest. Towards this end, we consider the problem of decentralized multi-agent navigation in cluttered environments. By introducing two sub-objectives of multi-agent navigation and environment optimization, we propose an $\textit{agent-environment co-optimization}$ problem and develop a $\textit{coordinated algorithm}$ that alternates between these sub-objectives to search for an optimal synthesis of agent actions and obstacle configurations in the environment; ultimately, improving the navigation performance. Due to the challenge of explicitly modeling the relation between agents, environment and performance, we leverag
    
[^3]: 基于扩散的迭代反事实解释用于胎儿超声图像质量评估

    Diffusion-based Iterative Counterfactual Explanations for Fetal Ultrasound Image Quality Assessment

    [https://arxiv.org/abs/2403.08700](https://arxiv.org/abs/2403.08700)

    提出了基于扩散的迭代反事实解释的方法，通过生成逼真的高质量标准平面，对提高临床医生的培训、改善图像质量以及提升下游诊断和监测具有潜在价值。

    

    怀孕期超声图像质量对准确诊断和监测胎儿健康至关重要。然而，生成高质量的标准平面很困难，受到超声波技术人员的专业知识以及像孕妇BMI或胎儿动态等因素的影响。在这项工作中，我们提出使用基于扩散的反事实可解释人工智能，从低质量的非标准平面生成逼真的高质量标准平面。通过定量和定性评估，我们证明了我们的方法在生成质量增加的可信反事实方面的有效性。这为通过提供视觉反馈加强临床医生培训以及改进图像质量，从而改善下游诊断和监测提供了未来的希望。

    arXiv:2403.08700v1 Announce Type: cross  Abstract: Obstetric ultrasound image quality is crucial for accurate diagnosis and monitoring of fetal health. However, producing high-quality standard planes is difficult, influenced by the sonographer's expertise and factors like the maternal BMI or the fetus dynamics. In this work, we propose using diffusion-based counterfactual explainable AI to generate realistic high-quality standard planes from low-quality non-standard ones. Through quantitative and qualitative evaluation, we demonstrate the effectiveness of our method in producing plausible counterfactuals of increased quality. This shows future promise both for enhancing training of clinicians by providing visual feedback, as well as for improving image quality and, consequently, downstream diagnosis and monitoring.
    
[^4]: 平均校准误差：一种可微损失函数，用于改善图像分割中的可靠性

    Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation

    [https://arxiv.org/abs/2403.06759](https://arxiv.org/abs/2403.06759)

    提出一种平均L1校准误差（mL1-ACE）作为辅助损失函数，用于改善图像分割中的像素级校准，减少了校准误差并引入了数据集可靠性直方图以提高校准评估。

    

    医学图像分割的深度神经网络经常产生与经验观察不一致的过于自信的结果，这种校准错误挑战着它们的临床应用。我们提出使用平均L1校准误差（mL1-ACE）作为一种新颖的辅助损失函数，以改善像素级校准而不会损害分割质量。我们展示了，尽管使用硬分箱，这种损失是直接可微的，避免了需要近似但可微的替代或软分箱方法的必要性。我们的工作还引入了数据集可靠性直方图的概念，这一概念推广了标准的可靠性图，用于在数据集级别聚合的语义分割中细化校准的视觉评估。使用mL1-ACE，我们将平均和最大校准误差分别降低了45%和55%，同时在BraTS 2021数据集上保持了87%的Dice分数。我们在这里分享我们的代码: https://github

    arXiv:2403.06759v1 Announce Type: cross  Abstract: Deep neural networks for medical image segmentation often produce overconfident results misaligned with empirical observations. Such miscalibration, challenges their clinical translation. We propose to use marginal L1 average calibration error (mL1-ACE) as a novel auxiliary loss function to improve pixel-wise calibration without compromising segmentation quality. We show that this loss, despite using hard binning, is directly differentiable, bypassing the need for approximate but differentiable surrogate or soft binning approaches. Our work also introduces the concept of dataset reliability histograms which generalises standard reliability diagrams for refined visual assessment of calibration in semantic segmentation aggregated at the dataset level. Using mL1-ACE, we reduce average and maximum calibration error by 45% and 55% respectively, maintaining a Dice score of 87% on the BraTS 2021 dataset. We share our code here: https://github
    
[^5]: 基于车辆组的高速公路碰撞风险形成和传播分析

    Vehicle-group-based Crash Risk Formation and Propagation Analysis for Expressways

    [https://arxiv.org/abs/2402.12415](https://arxiv.org/abs/2402.12415)

    本研究基于车辆组作为分析对象，探讨了考虑车辆组和道路段特征的风险形成和传播机制，识别出影响碰撞风险的关键因素。

    

    先前的研究主要将路段上的碰撞数量或可能性与交通参数或路段的几何特征联系起来，通常忽略了车辆连续运动和与附近车辆的互动对其影响。通信技术的进步赋予了从周围车辆收集驾驶信息的能力，使得研究基于车辆组的碰撞风险成为可能。基于高分辨率车辆轨迹数据，本研究以车辆组作为分析对象，探讨了考虑车辆组和道路段特征的风险形成和传播机制。确定了几个影响碰撞风险的关键因素，包括过去的高风险车辆组状态、复杂的车辆行为、大型车辆的高百分比、车辆组内频繁变道以及特定的道路几何形状。

    arXiv:2402.12415v1 Announce Type: new  Abstract: Previous studies in predicting crash risk primarily associated the number or likelihood of crashes on a road segment with traffic parameters or geometric characteristics of the segment, usually neglecting the impact of vehicles' continuous movement and interactions with nearby vehicles. Advancements in communication technologies have empowered driving information collected from surrounding vehicles, enabling the study of group-based crash risks. Based on high-resolution vehicle trajectory data, this research focused on vehicle groups as the subject of analysis and explored risk formation and propagation mechanisms considering features of vehicle groups and road segments. Several key factors contributing to crash risks were identified, including past high-risk vehicle-group states, complex vehicle behaviors, high percentage of large vehicles, frequent lane changes within a vehicle group, and specific road geometries. A multinomial logisti
    
[^6]: SpikeNAS: 一种面向脉冲神经网络系统的快速内存感知神经架构搜索框架

    SpikeNAS: A Fast Memory-Aware Neural Architecture Search Framework for Spiking Neural Network Systems

    [https://arxiv.org/abs/2402.11322](https://arxiv.org/abs/2402.11322)

    SpikeNAS提出了一种快速内存感知神经架构搜索框架，旨在帮助脉冲神经网络系统快速找到在给定内存预算下高准确性的适当架构。

    

    脉冲神经网络（SNN）为解决机器学习任务提供了实现超低功耗计算的有前途的解决方案。目前，大多数SNN架构都源自人工神经网络，其神经元的架构和操作与SNN不同，或者在不考虑来自底层处理硬件的内存预算的情况下开发。这些限制阻碍了SNN在准确性和效率方面充分发挥潜力。为此，我们提出了SpikeNAS，一种新颖的内存感知神经架构搜索（NAS）框架，可在给定内存预算下快速找到一个具有高准确性的适当SNN架构。为实现这一目标，我们的SpikeNAS采用了几个关键步骤：分析网络操作对准确性的影响，增强网络架构以提高学习质量，并开发快速内存感知搜索算法。

    arXiv:2402.11322v1 Announce Type: cross  Abstract: Spiking Neural Networks (SNNs) offer a promising solution to achieve ultra low-power/energy computation for solving machine learning tasks. Currently, most of the SNN architectures are derived from Artificial Neural Networks whose neurons' architectures and operations are different from SNNs, or developed without considering memory budgets from the underlying processing hardware. These limitations hinder the SNNs from reaching their full potential in accuracy and efficiency. Towards this, we propose SpikeNAS, a novel memory-aware neural architecture search (NAS) framework for SNNs that can quickly find an appropriate SNN architecture with high accuracy under the given memory budgets. To do this, our SpikeNAS employs several key steps: analyzing the impacts of network operations on the accuracy, enhancing the network architecture to improve the learning quality, and developing a fast memory-aware search algorithm. The experimental resul
    
[^7]: EdgeQAT: 熵和分布引导的量化感知训练，用于加速轻量级LLMs在边缘设备上的应用

    EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

    [https://arxiv.org/abs/2402.10787](https://arxiv.org/abs/2402.10787)

    本文提出了EdgeQAT，使用熵和分布引导的量化感知训练方法来优化轻量级LLMs，在边缘设备上实现推理加速。

    

    尽管大型语言模型（LLMs）在各个领域取得了显著进展，但由于其庞大的参数和计算量，LLMs在边缘设备上的广泛应用受到限制。为了解决这一问题，通常采用量化方法生成具有高效计算和快速推理的轻量级LLMs。然而，后训练量化（PTQ）方法在将权重、激活和KV缓存一起量化至8位以下时，质量会急剧下降。此外，许多量化感知训练（QAT）工作对模型权重进行量化，而激活未被触及，这不能充分发挥量化对边缘端推理加速的潜力。在本文中，我们提出了EdgeQAT，即熵和分布引导的QAT，用于优化轻量级LLMs以实现在边缘设备上的推理加速。我们首先确定量化性能下降主要源自信息

    arXiv:2402.10787v1 Announce Type: cross  Abstract: Despite the remarkable strides of Large Language Models (LLMs) in various fields, the wide applications of LLMs on edge devices are limited due to their massive parameters and computations. To address this, quantization is commonly adopted to generate lightweight LLMs with efficient computations and fast inference. However, Post-Training Quantization (PTQ) methods dramatically degrade in quality when quantizing weights, activations, and KV cache together to below 8 bits. Besides, many Quantization-Aware Training (QAT) works quantize model weights, leaving the activations untouched, which do not fully exploit the potential of quantization for inference acceleration on the edge. In this paper, we propose EdgeQAT, the Entropy and Distribution Guided QAT for the optimization of lightweight LLMs to achieve inference acceleration on Edge devices. We first identify that the performance drop of quantization primarily stems from the information
    
[^8]: 动量在隐式逐步优化中对目标函数的平滑作用的角色

    Role of Momentum in Smoothing Objective Function in Implicit Graduated Optimization

    [https://arxiv.org/abs/2402.02325](https://arxiv.org/abs/2402.02325)

    这篇论文揭示了具有动量的随机梯度下降算法平滑了目标函数，影响程度由多个超参数决定，同时提供了对动量改善泛化能力的理论解释和新见解。

    

    虽然具有动量的随机梯度下降（SGD）具有快速收敛和良好的泛化能力，但对此缺乏理论解释。本文展示了具有动量的SGD平滑了目标函数，其程度由学习率、批大小、动量因子、随机梯度的方差以及梯度范数的上界确定。这一理论发现揭示了为什么动量改善了泛化能力，并提供了关于动量因子等超参数作用的新见解。我们还提出了一种利用SGD动量平滑特性的隐式逐步优化算法，并提供了实验结果支持我们的观点，即SGD动量平滑了目标函数。

    While stochastic gradient descent (SGD) with momentum has fast convergence and excellent generalizability, a theoretical explanation for this is lacking. In this paper, we show that SGD with momentum smooths the objective function, the degree of which is determined by the learning rate, the batch size, the momentum factor, the variance of the stochastic gradient, and the upper bound of the gradient norm. This theoretical finding reveals why momentum improves generalizability and provides new insights into the role of the hyperparameters, including momentum factor. We also present an implicit graduated optimization algorithm that exploits the smoothing properties of SGD with momentum and provide experimental results supporting our assertion that SGD with momentum smooths the objective function.
    
[^9]: 通过Wasserstein度量进行数据集精炼

    Dataset Distillation via the Wasserstein Metric

    [https://arxiv.org/abs/2311.18531](https://arxiv.org/abs/2311.18531)

    通过引入Wasserstein距离及其重心，我们提出一种有效的数据集精炼方法，利用先验知识提高分布匹配效果，实现了新的最先进性能。

    

    数据集精炼（DD）作为一种强大的策略，将大型数据集的丰富信息封装为明显更小的合成等价物，从而在减少计算开销的同时保留模型性能。为实现这一目标，我们引入了Wasserstein距离，这是一种基于最优输运理论的度量，用于增强DD中的分布匹配。我们的方法利用Wasserstein重心提供了一种在量化分布差异和高效捕获分布集合中心的几何意义方法。通过在预训练分类模型的特征空间中嵌入合成数据，我们促进了有效的分布匹配，利用这些模型固有的先验知识。我们的方法不仅保持了基于分布匹配的技术的计算优势，而且在一系列任务中实现了新的最先进性能。

    arXiv:2311.18531v2 Announce Type: replace-cross  Abstract: Dataset Distillation (DD) emerges as a powerful strategy to encapsulate the expansive information of large datasets into significantly smaller, synthetic equivalents, thereby preserving model performance with reduced computational overhead. Pursuing this objective, we introduce the Wasserstein distance, a metric grounded in optimal transport theory, to enhance distribution matching in DD. Our approach employs the Wasserstein barycenter to provide a geometrically meaningful method for quantifying distribution differences and capturing the centroid of distribution sets efficiently. By embedding synthetic data in the feature spaces of pretrained classification models, we facilitate effective distribution matching that leverages prior knowledge inherent in these models. Our method not only maintains the computational advantages of distribution matching-based techniques but also achieves new state-of-the-art performance across a ran
    
[^10]: 关于随机神经网络的Lipschitz常数

    On the Lipschitz constant of random neural networks. (arXiv:2311.01356v1 [stat.ML])

    [http://arxiv.org/abs/2311.01356](http://arxiv.org/abs/2311.01356)

    本文研究了随机ReLU神经网络的Lipschitz常数，对于浅层神经网络，我们得到了Lipschitz常数的精确刻画，对于足够宽度的深层神经网络，我们给出了上下界，并匹配一个依赖于深度的对数因子。

    

    实证研究广泛证明神经网络对输入的微小对抗性扰动非常敏感。这些所谓的对抗性示例的最坏情况鲁棒性可以通过神经网络的Lipschitz常数来量化。然而，关于这个量的理论结果在文献中仅有少数。在本文中，我们开始研究随机ReLU神经网络的Lipschitz常数，即选择随机权重并采用ReLU激活函数的神经网络。对于浅层神经网络，我们将Lipschitz常数刻画到一个绝对数值常数。此外，我们将我们的分析扩展到足够宽度的深层神经网络，我们证明了Lipschitz常数的上下界。这些界匹配到一个依赖于深度的对数因子上。

    Empirical studies have widely demonstrated that neural networks are highly sensitive to small, adversarial perturbations of the input. The worst-case robustness against these so-called adversarial examples can be quantified by the Lipschitz constant of the neural network. However, only few theoretical results regarding this quantity exist in the literature. In this paper, we initiate the study of the Lipschitz constant of random ReLU neural networks, i.e., neural networks whose weights are chosen at random and which employ the ReLU activation function. For shallow neural networks, we characterize the Lipschitz constant up to an absolute numerical constant. Moreover, we extend our analysis to deep neural networks of sufficiently large width where we prove upper and lower bounds for the Lipschitz constant. These bounds match up to a logarithmic factor that depends on the depth.
    
[^11]: 基于EEG的运动想象分类的特征重加权

    Feature Reweighting for EEG-based Motor Imagery Classification. (arXiv:2308.02515v1 [cs.LG])

    [http://arxiv.org/abs/2308.02515](http://arxiv.org/abs/2308.02515)

    本论文提出了一种特征重加权的方法，用于解决使用EEG信号进行运动想象分类时存在的低信噪比、非稳态性、非线性和复杂性等挑战，通过降低噪声和无关信息，提高分类性能。

    

    利用非侵入性脑电图（EEG）信号进行运动想象（MI）分类是一个重要的目标，因为它用于预测主体肢体移动的意图。最近的研究中，基于卷积神经网络（CNN）的方法已被广泛应用于MI-EEG分类。训练神经网络进行MI-EEG信号分类的挑战包括信噪比低、非稳态性、非线性和EEG信号的复杂性。基于CNN的网络计算得到的MI-EEG信号特征包含无关信息。因此，由噪声和无关特征计算得到的CNN网络的特征图也包含无关信息。因此，许多无用的特征常常误导神经网络训练，降低分类性能。为解决这个问题，提出了一种新的特征重加权方法。

    Classification of motor imagery (MI) using non-invasive electroencephalographic (EEG) signals is a critical objective as it is used to predict the intention of limb movements of a subject. In recent research, convolutional neural network (CNN) based methods have been widely utilized for MI-EEG classification. The challenges of training neural networks for MI-EEG signals classification include low signal-to-noise ratio, non-stationarity, non-linearity, and high complexity of EEG signals. The features computed by CNN-based networks on the highly noisy MI-EEG signals contain irrelevant information. Subsequently, the feature maps of the CNN-based network computed from the noisy and irrelevant features contain irrelevant information. Thus, many non-contributing features often mislead the neural network training and degrade the classification performance. Hence, a novel feature reweighting approach is proposed to address this issue. The proposed method gives a noise reduction mechanism named
    

