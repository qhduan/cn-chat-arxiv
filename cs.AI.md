# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RASP: A Drone-based Reconfigurable Actuation and Sensing Platform Towards Ambient Intelligent Systems](https://arxiv.org/abs/2403.12853) | 提出了RASP，一个可在25秒内自主更换传感器和执行器的模块化和可重构传感和作动平台，使无人机能快速适应各种任务，同时引入了利用大规模语言和视觉语言模型的个人助理系统架构。 |
| [^2] | [Are Vision Language Models Texture or Shape Biased and Can We Steer Them?](https://arxiv.org/abs/2403.09193) | 本文研究了广泛应用的视觉语言模型中的纹理与形状偏见，发现这些模型通常比视觉编码器更偏向形状，暗示视觉偏见在一定程度上会受到文本的调节 |
| [^3] | [Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods](https://arxiv.org/abs/2403.08352) | 自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。 |
| [^4] | [On the Challenges and Opportunities in Generative AI](https://arxiv.org/abs/2403.00025) | 现代生成人工智能范例中存在关键的未解决挑战，如何解决这些挑战将进一步增强它们的能力、多功能性和可靠性，并为研究方向提供有价值的见解。 |
| [^5] | [GenCeption: Evaluate Multimodal LLMs with Unlabeled Unimodal Data](https://arxiv.org/abs/2402.14973) | 提出了一种名为GenCeption的新型MLLM评估框架，可以仅利用单模态数据评估跨模态语义一致性，并有效反映模型产生幻觉的倾向，具有较强的相关性和潜力于流行的MLLM基准结果。 |
| [^6] | [Which Frequencies do CNNs Need? Emergent Bottleneck Structure in Feature Learning](https://arxiv.org/abs/2402.08010) | 本文描述了CNN中卷积瓶颈（CBN）结构的出现，网络在前几层将输入表示转换为在少数频率和通道上受支持的表示，然后通过最后几层映射回输出。CBN秩定义了保留在瓶颈中的频率的数量和类型，并部分证明了参数范数与深度和CBN秩的比例成正比。此外，我们还展示了网络的参数范数依赖于函数的规则性。我们发现任何具有接近最优参数范数的网络都会展示出CBN结构，这解释了下采样的常见实践；我们还验证了CBN结构在下采样下仍然成立。最后，我们使用CBN结构来解释...（摘要完整内容请见正文） |
| [^7] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^8] | [Digital Divides in Scene Recognition: Uncovering Socioeconomic Biases in Deep Learning Systems.](http://arxiv.org/abs/2401.13097) | 该研究研究了深度学习系统中的社会经济偏见对场景识别的影响，发现了预训练的卷积神经网络在低社会经济地位的家庭照片中显示出更低的分类准确度和分类置信度，并更容易分配具有冒犯性的标签。 |
| [^9] | [Tutorial on amortized optimization.](http://arxiv.org/abs/2202.00665) | 该教程介绍了分摊优化的基础，并总结了其在变分推断、稀疏编码、元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。 |

# 详细

[^1]: 基于无人机的环境智能系统的可重构作动和传感平台RASP

    RASP: A Drone-based Reconfigurable Actuation and Sensing Platform Towards Ambient Intelligent Systems

    [https://arxiv.org/abs/2403.12853](https://arxiv.org/abs/2403.12853)

    提出了RASP，一个可在25秒内自主更换传感器和执行器的模块化和可重构传感和作动平台，使无人机能快速适应各种任务，同时引入了利用大规模语言和视觉语言模型的个人助理系统架构。

    

    实现消费级无人机与我们家中的吸尘机器人或日常生活中的个人智能手机一样有用，需要无人机能感知、驱动和响应可能出现的一般情况。为了实现这一愿景，我们提出了RASP，一个模块化和可重构的传感和作动平台，允许无人机在仅25秒内自主更换机载传感器和执行器，使单个无人机能够快速适应各种任务。RASP包括一个机械层，用于物理更换传感器模块，一个电气层，用于维护传感器/执行器的电源和通信线路，以及一个软件层，用于在无人机和我们平台上的任何传感器模块之间维护一个公共接口。利用最近在大型语言和视觉语言模型方面的进展，我们进一步介绍了一种利用RASP的个人助理系统的架构、实现和现实世界部署。

    arXiv:2403.12853v1 Announce Type: cross  Abstract: Realizing consumer-grade drones that are as useful as robot vacuums throughout our homes or personal smartphones in our daily lives requires drones to sense, actuate, and respond to general scenarios that may arise. Towards this vision, we propose RASP, a modular and reconfigurable sensing and actuation platform that allows drones to autonomously swap onboard sensors and actuators in only 25 seconds, allowing a single drone to quickly adapt to a diverse range of tasks. RASP consists of a mechanical layer to physically swap sensor modules, an electrical layer to maintain power and communication lines to the sensor/actuator, and a software layer to maintain a common interface between the drone and any sensor module in our platform. Leveraging recent advances in large language and visual language models, we further introduce the architecture, implementation, and real-world deployments of a personal assistant system utilizing RASP. We demo
    
[^2]: 视觉语言模型是纹理偏见还是形状偏见，我们可以引导它们吗？

    Are Vision Language Models Texture or Shape Biased and Can We Steer Them?

    [https://arxiv.org/abs/2403.09193](https://arxiv.org/abs/2403.09193)

    本文研究了广泛应用的视觉语言模型中的纹理与形状偏见，发现这些模型通常比视觉编码器更偏向形状，暗示视觉偏见在一定程度上会受到文本的调节

    

    arXiv:2403.09193v1 公告类型: 跨领域 摘要: 视觉语言模型（VLMs）在短短几年内彻底改变了计算机视觉模型的格局，开启了一系列新的应用，从零样本图像分类到图像字幕生成，再到视觉问答。与纯视觉模型不同，它们提供了通过语言提示访问视觉内容的直观方式。这种模型的广泛适用性引发我们思考它们是否也与人类视觉一致 - 具体来说，它们在多模态融合中有多大程度地采用了人类引导的视觉偏见，或者它们是否只是从纯视觉模型中继承了偏见。其中一个重要的视觉偏见是纹理与形状偏见，即局部信息的主导地位。在本文中，我们研究了一系列流行的VLMs中的这种偏见。有趣的是，我们发现VLMs通常比它们的视觉编码器更偏向于形状，这表明视觉偏见在一定程度上通过文本进行调节。

    arXiv:2403.09193v1 Announce Type: cross  Abstract: Vision language models (VLMs) have drastically changed the computer vision model landscape in only a few years, opening an exciting array of new applications from zero-shot image classification, over to image captioning, and visual question answering. Unlike pure vision models, they offer an intuitive way to access visual content through language prompting. The wide applicability of such models encourages us to ask whether they also align with human vision - specifically, how far they adopt human-induced visual biases through multimodal fusion, or whether they simply inherit biases from pure vision models. One important visual bias is the texture vs. shape bias, or the dominance of local over global information. In this paper, we study this bias in a wide range of popular VLMs. Interestingly, we find that VLMs are often more shape-biased than their vision encoders, indicating that visual biases are modulated to some extent through text
    
[^3]: 利用自动化机器学习的数据增强方法及与传统数据增强方法性能比较

    Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods

    [https://arxiv.org/abs/2403.08352](https://arxiv.org/abs/2403.08352)

    自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。

    

    数据增强被认为是常用于提高机器学习模型泛化性能的最重要的正则化技术。它主要涉及应用适当的数据转换操作，以创建具有所需属性的新数据样本。尽管其有效性，这一过程通常具有挑战性，因为手动创建和测试不同候选增强及其超参数需耗费大量时间。自动化数据增强方法旨在自动化这一过程。最先进的方法通常依赖于自动化机器学习（AutoML）原则。本研究提供了基于AutoML的数据增强技术的全面调查。我们讨论了使用AutoML实现数据增强的各种方法，包括数据操作、数据集成和数据合成技术。我们详细讨论了技术

    arXiv:2403.08352v1 Announce Type: cross  Abstract: Data augmentation is arguably the most important regularization technique commonly used to improve generalization performance of machine learning models. It primarily involves the application of appropriate data transformation operations to create new data samples with desired properties. Despite its effectiveness, the process is often challenging because of the time-consuming trial and error procedures for creating and testing different candidate augmentations and their hyperparameters manually. Automated data augmentation methods aim to automate the process. State-of-the-art approaches typically rely on automated machine learning (AutoML) principles. This work presents a comprehensive survey of AutoML-based data augmentation techniques. We discuss various approaches for accomplishing data augmentation with AutoML, including data manipulation, data integration and data synthesis techniques. We present extensive discussion of technique
    
[^4]: 关于生成人工智能中的挑战与机遇

    On the Challenges and Opportunities in Generative AI

    [https://arxiv.org/abs/2403.00025](https://arxiv.org/abs/2403.00025)

    现代生成人工智能范例中存在关键的未解决挑战，如何解决这些挑战将进一步增强它们的能力、多功能性和可靠性，并为研究方向提供有价值的见解。

    

    深度生成建模领域近年来增长迅速而稳定。随着海量训练数据的可用性以及可扩展的无监督学习范式的进步，最近的大规模生成模型展现出合成高分辨率图像和文本以及结构化数据（如视频和分子）的巨大潜力。然而，我们认为当前大规模生成人工智能模型没有充分解决若干基本问题，限制了它们在各个领域的广泛应用。在本工作中，我们旨在确定现代生成人工智能范例中的关键未解决挑战，以进一步增强它们的能力、多功能性和可靠性。通过识别这些挑战，我们旨在为研究人员提供有价值的见解，探索有益的研究方向，从而促进更加强大和可访问的生成人工智能的发展。

    arXiv:2403.00025v1 Announce Type: cross  Abstract: The field of deep generative modeling has grown rapidly and consistently over the years. With the availability of massive amounts of training data coupled with advances in scalable unsupervised learning paradigms, recent large-scale generative models show tremendous promise in synthesizing high-resolution images and text, as well as structured data such as videos and molecules. However, we argue that current large-scale generative AI models do not sufficiently address several fundamental issues that hinder their widespread adoption across domains. In this work, we aim to identify key unresolved challenges in modern generative AI paradigms that should be tackled to further enhance their capabilities, versatility, and reliability. By identifying these challenges, we aim to provide researchers with valuable insights for exploring fruitful research directions, thereby fostering the development of more robust and accessible generative AI so
    
[^5]: GenCeption：使用未标记的单模态数据评估多模态LLM

    GenCeption: Evaluate Multimodal LLMs with Unlabeled Unimodal Data

    [https://arxiv.org/abs/2402.14973](https://arxiv.org/abs/2402.14973)

    提出了一种名为GenCeption的新型MLLM评估框架，可以仅利用单模态数据评估跨模态语义一致性，并有效反映模型产生幻觉的倾向，具有较强的相关性和潜力于流行的MLLM基准结果。

    

    多模态大型语言模型（MLLMs）通常使用昂贵的带标注的多模态基准进行评估。然而，这些基准通常难以跟上MLLM评估的快速发展要求。我们提出了GenCeption，这是一个新颖的无需注释的MLLM评估框架，仅需要单模态数据来评估跨模态语义一致性，并反映出模型产生幻觉的倾向。类似于流行的DrawCeption游戏，GenCeption从一个非文本样本开始，并经历一系列迭代的描述和生成步骤。迭代之间的语义漂移使用GC@T指标进行量化。我们的实证发现验证了GenCeption的有效性，并显示出与流行的MLLM基准结果的强相关性。GenCeption可以通过利用普遍存在且以前未见的单模态数据来扩展，以减轻训练数据的污染。

    arXiv:2402.14973v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) are commonly evaluated using costly annotated multimodal benchmarks. However, these benchmarks often struggle to keep pace with the rapidly advancing requirements of MLLM evaluation. We propose GenCeption, a novel and annotation-free MLLM evaluation framework that merely requires unimodal data to assess inter-modality semantic coherence and inversely reflects the models' inclination to hallucinate. Analogous to the popular DrawCeption game, GenCeption initiates with a non-textual sample and undergoes a series of iterative description and generation steps. Semantic drift across iterations is quantified using the GC@T metric. Our empirical findings validate GenCeption's efficacy, showing strong correlations with popular MLLM benchmarking results. GenCeption may be extended to mitigate training data contamination by utilizing ubiquitous, previously unseen unimodal data.
    
[^6]: CNN需要哪些频率？特征学习中的紧急瓶颈结构的出现

    Which Frequencies do CNNs Need? Emergent Bottleneck Structure in Feature Learning

    [https://arxiv.org/abs/2402.08010](https://arxiv.org/abs/2402.08010)

    本文描述了CNN中卷积瓶颈（CBN）结构的出现，网络在前几层将输入表示转换为在少数频率和通道上受支持的表示，然后通过最后几层映射回输出。CBN秩定义了保留在瓶颈中的频率的数量和类型，并部分证明了参数范数与深度和CBN秩的比例成正比。此外，我们还展示了网络的参数范数依赖于函数的规则性。我们发现任何具有接近最优参数范数的网络都会展示出CBN结构，这解释了下采样的常见实践；我们还验证了CBN结构在下采样下仍然成立。最后，我们使用CBN结构来解释...（摘要完整内容请见正文）

    

    我们描述了CNN中卷积瓶颈（CBN）结构的出现，网络使用其前几层将输入表示转换为仅在几个频率和通道上受支持的表示，然后使用最后几层将其映射回输出。我们定义了CBN秩，描述了保留在瓶颈内的频率的数量和类型，并在一定程度上证明了表示函数$f$所需的参数范数按深度乘以CBN秩$f$的比例缩放。我们还展示了参数范数在下一阶中依赖于$f$的正则性。我们展示了任何具有近乎最优参数范数的网络都会在权重和（在网络对大学习率稳定的假设下）激活中表现出CBN结构，这促使了下采样的常见做法；并且我们验证了CBN结构在下采样下仍然成立。最后，我们使用CBN结构来解释...

    We describe the emergence of a Convolution Bottleneck (CBN) structure in CNNs, where the network uses its first few layers to transform the input representation into a representation that is supported only along a few frequencies and channels, before using the last few layers to map back to the outputs. We define the CBN rank, which describes the number and type of frequencies that are kept inside the bottleneck, and partially prove that the parameter norm required to represent a function $f$ scales as depth times the CBN rank $f$. We also show that the parameter norm depends at next order on the regularity of $f$. We show that any network with almost optimal parameter norm will exhibit a CBN structure in both the weights and - under the assumption that the network is stable under large learning rate - the activations, which motivates the common practice of down-sampling; and we verify that the CBN results still hold with down-sampling. Finally we use the CBN structure to interpret the
    
[^7]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^8]: 场景识别中的数字鸿沟：揭示深度学习系统中的社会经济偏见

    Digital Divides in Scene Recognition: Uncovering Socioeconomic Biases in Deep Learning Systems. (arXiv:2401.13097v1 [cs.CV])

    [http://arxiv.org/abs/2401.13097](http://arxiv.org/abs/2401.13097)

    该研究研究了深度学习系统中的社会经济偏见对场景识别的影响，发现了预训练的卷积神经网络在低社会经济地位的家庭照片中显示出更低的分类准确度和分类置信度，并更容易分配具有冒犯性的标签。

    

    基于计算机的场景理解影响了从城市规划到自动驾驶的领域，然而我们对这些技术在社会差异中的表现了解甚少。我们研究了深度卷积神经网络（dCNNs）在场景分类中的偏见，使用了来自全球和美国的近百万张图片，包括用户提交的家庭照片和Airbnb的房源照片。我们运用了统计模型，对家庭收入、人类发展指数（HDI）等社会经济指标以及公开数据来源（CIA和美国人口普查）的人口统计因素对dCNNs的表现影响进行了量化。我们的分析发现了显著的社会经济偏见，预训练的dCNNs表现出更低的分类准确度、更低的分类置信度，以及更高的倾向性在低社会经济地位的家庭（例如“废墟”，“贫民窟”）的图片中分配具有冒犯性的标签。这种趋势是持续的。

    Computer-based scene understanding has influenced fields ranging from urban planning to autonomous vehicle performance, yet little is known about how well these technologies work across social differences. We investigate the biases of deep convolutional neural networks (dCNNs) in scene classification, using nearly one million images from global and US sources, including user-submitted home photographs and Airbnb listings. We applied statistical models to quantify the impact of socioeconomic indicators such as family income, Human Development Index (HDI), and demographic factors from public data sources (CIA and US Census) on dCNN performance. Our analyses revealed significant socioeconomic bias, where pretrained dCNNs demonstrated lower classification accuracy, lower classification confidence, and a higher tendency to assign labels that could be offensive when applied to homes (e.g., "ruin", "slum"), especially in images from homes with lower socioeconomic status (SES). This trend is c
    
[^9]: 关于分摊优化的教程

    Tutorial on amortized optimization. (arXiv:2202.00665v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.00665](http://arxiv.org/abs/2202.00665)

    该教程介绍了分摊优化的基础，并总结了其在变分推断、稀疏编码、元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。

    

    优化是一种普遍的建模工具，经常在反复解决相同问题的情况下使用。分摊优化方法使用学习来预测这些设置中问题的解决方案，利用相似问题实例之间的共享结构。这些方法在变分推断和强化学习中至关重要，能够比不使用分摊的传统优化方法快几个数量级地解决优化问题。本次教程介绍了这些进步背后的分摊优化基础，并概述了它们在变分推断、稀疏编码、基于梯度的元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。本教程的源代码可在https://github.com/facebookresearch/amortized-optimization-tutorial上获得。

    Optimization is a ubiquitous modeling tool and is often deployed in settings which repeatedly solve similar instances of the same problem. Amortized optimization methods use learning to predict the solutions to problems in these settings, exploiting the shared structure between similar problem instances. These methods have been crucial in variational inference and reinforcement learning and are capable of solving optimization problems many orders of magnitudes times faster than traditional optimization methods that do not use amortization. This tutorial presents an introduction to the amortized optimization foundations behind these advancements and overviews their applications in variational inference, sparse coding, gradient-based meta-learning, control, reinforcement learning, convex optimization, optimal transport, and deep equilibrium networks. The source code for this tutorial is available at https://github.com/facebookresearch/amortized-optimization-tutorial.
    

