# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Initialisation and Topology Effects in Decentralised Federated Learning](https://arxiv.org/abs/2403.15855) | 分散式联邦学习的有效性受到连接设备网络拓扑结构的显著影响，我们提出了基于底层网络节点特征向量中心性分布的改进神经网络初始化策略，大大提高了训练效率。 |
| [^2] | [Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning](https://arxiv.org/abs/2403.10056) | 提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。 |
| [^3] | [CodeMind: A Framework to Challenge Large Language Models for Code Reasoning](https://arxiv.org/abs/2402.09664) | CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。 |
| [^4] | [Transformer-based deep imitation learning for dual-arm robot manipulation](https://arxiv.org/abs/2108.00385) | 使用Transformer的深度模仿学习结构成功解决了双臂机器人操作任务中神经网络性能不佳的问题 |
| [^5] | [Gaze-based dual resolution deep imitation learning for high-precision dexterous robot manipulation](https://arxiv.org/abs/2102.01295) | 基于人类基于凝视的双分辨率视觉运动控制系统的启发，提出了一种利用深度模仿学习解决高精度灵巧机器人操作任务的方法 |
| [^6] | [Fast Inference Through The Reuse Of Attention Maps In Diffusion Models.](http://arxiv.org/abs/2401.01008) | 本文提出了一种无需训练的方法，通过重用注意力映射来实现Text-to-image diffusion models中的快速推理，以提高效率。 |
| [^7] | [Diagnostic test accuracy (DTA) of artificial intelligence in digital pathology: a systematic review, meta-analysis and quality assessment.](http://arxiv.org/abs/2306.07999) | 本文进行了数字病理图像中应用人工智能的所有病理学领域的诊断准确度的系统综述和Meta分析。结果表明，人工智能在数字病理学中取得了高度的准确度，是可行的辅助诊断工具。 |
| [^8] | [Compressing neural network by tensor network with exponentially fewer variational parameters.](http://arxiv.org/abs/2305.06058) | 本文提出了一种通用的压缩方案，将神经网络的可变参数编码为多层张量网络，明显减少了可变参数的数量，并在多个神经网络和数据集上表现出了卓越的压缩性能，以VGG-16的测试精度提高为例。 |

# 详细

[^1]: 初始值和拓扑结构在分散式联邦学习中的影响

    Initialisation and Topology Effects in Decentralised Federated Learning

    [https://arxiv.org/abs/2403.15855](https://arxiv.org/abs/2403.15855)

    分散式联邦学习的有效性受到连接设备网络拓扑结构的显著影响，我们提出了基于底层网络节点特征向量中心性分布的改进神经网络初始化策略，大大提高了训练效率。

    

    具有完全分散式特征的联邦学习使得在网络上分布式设备上对个体机器学习模型进行协作训练，同时保持训练数据本地化。这种方法增强了数据隐私性，消除了单点故障和中央协调的必要性。我们的研究强调了分散式联邦学习的有效性受到连接设备的网络拓扑结构的显著影响。一个简化的数值模型用于研究这些系统的早期行为，使我们得出了一个利用底层网络节点的特征向量中心性分布的改进人工神经网络初始值策略，从而大大提高了训练效率。此外，我们的研究探讨了在我们提出的初始化策略下的比例行为和环境参数的选择。这项工作为更多研究打开了道路。

    arXiv:2403.15855v1 Announce Type: cross  Abstract: Fully decentralised federated learning enables collaborative training of individual machine learning models on distributed devices on a network while keeping the training data localised. This approach enhances data privacy and eliminates both the single point of failure and the necessity for central coordination. Our research highlights that the effectiveness of decentralised federated learning is significantly influenced by the network topology of connected devices. A simplified numerical model for studying the early behaviour of these systems leads us to an improved artificial neural network initialisation strategy, which leverages the distribution of eigenvector centralities of the nodes of the underlying network, leading to a radically improved training efficiency. Additionally, our study explores the scaling behaviour and choice of environmental parameters under our proposed initialisation strategy. This work paves the way for mor
    
[^2]: 不要半心半意：捕捉连续指导调整中的关键部分信息

    Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning

    [https://arxiv.org/abs/2403.10056](https://arxiv.org/abs/2403.10056)

    提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。

    

    arXiv:2403.10056v1 公告类型: 跨领域 摘要：大型语言模型（LLMs）的指导调整可以驱使它们在特定下游任务中产生符合人类目标的结果。然而，LLMs的连续指导调整（CIT）过程可能会带来灾难性遗忘（CF）问题，导致先前学到的能力退化。最近的方法尝试通过修改模型或重放数据来缓解CF问题，但这可能只记住指令的表面模式并在留存任务上感到困惑。在本文中，我们提出了一种基于关键部分信息增益（KPIG）的新型连续指导调整方法。我们的方法计算掩盖部分的信息增益，动态重放数据并优化训练目标，从而使LLMs能够捕捉与正确响应相关的任务感知信息，并减轻对指导中通用描述的过度拟合。此外，我们提出了两个指标，P分和V分，

    arXiv:2403.10056v1 Announce Type: cross  Abstract: Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score,
    
[^3]: CodeMind:一个用于挑战大型语言模型进行代码推理的框架

    CodeMind: A Framework to Challenge Large Language Models for Code Reasoning

    [https://arxiv.org/abs/2402.09664](https://arxiv.org/abs/2402.09664)

    CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    

    仅靠测试通过来评估大型语言模型（LLMs）的代码合成能力可能会导致不公正的评估或促进具有数据泄漏的模型，作为一种替代方案，我们介绍了CodeMind，这是一个旨在评估LLMs的代码推理能力的框架。CodeMind目前支持三种代码推理任务：独立执行推理（IER）、依赖执行推理（DER）和规范推理（SR）。前两者评估模型以预测任意代码的执行输出，或者模型能够正确合成的代码。第三个任务评估LLMs实现指定预期行为的程度。我们使用CodeMind对两种不同编程语言中的五个基准下的九个LLMs进行了广泛的评估，结果表明LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    arXiv:2402.09664v1 Announce Type: cross  Abstract: Solely relying on test passing to evaluate Large Language Models (LLMs) for code synthesis may result in unfair assessment or promoting models with data leakage. As an alternative, we introduce CodeMind, a framework designed to gauge the code reasoning abilities of LLMs. CodeMind currently supports three code reasoning tasks: Independent Execution Reasoning (IER), Dependent Execution Reasoning (DER), and Specification Reasoning (SR). The first two evaluate models to predict the execution output of an arbitrary code or code the model could correctly synthesize. The third one evaluates the extent to which LLMs implement the specified expected behavior. Our extensive evaluation of nine LLMs across five benchmarks in two different programming languages using CodeMind shows that LLMs fairly understand control flow constructs and, in general, are capable of reasoning how inputs evolve to output, specifically for simple programs and the ones 
    
[^4]: 基于Transformer的双臂机器人操作的深度模仿学习

    Transformer-based deep imitation learning for dual-arm robot manipulation

    [https://arxiv.org/abs/2108.00385](https://arxiv.org/abs/2108.00385)

    使用Transformer的深度模仿学习结构成功解决了双臂机器人操作任务中神经网络性能不佳的问题

    

    深度模仿学习对解决熟练操作任务具有潜力，因为它不需要环境模型和预编程的机器人行为。然而，将其应用于双臂操作任务仍具有挑战性。在双臂操作设置中，由于附加机器人操作器引起的状态维度增加，导致了神经网络性能不佳。我们通过使用一种自注意力机制来解决这个问题，该机制计算顺序输入中元素之间的依赖关系，并专注于重要元素。Transformer，作为自注意力架构的一种变体，被应用于深度模仿学习中，以解决真实世界中的双臂操作任务。所提出的方法已在真实机器人上的双臂操作任务上进行了测试。实验结果表明，基于Transformer的深度模仿学习架构可以进行关注

    arXiv:2108.00385v2 Announce Type: replace-cross  Abstract: Deep imitation learning is promising for solving dexterous manipulation tasks because it does not require an environment model and pre-programmed robot behavior. However, its application to dual-arm manipulation tasks remains challenging. In a dual-arm manipulation setup, the increased number of state dimensions caused by the additional robot manipulators causes distractions and results in poor performance of the neural networks. We address this issue using a self-attention mechanism that computes dependencies between elements in a sequential input and focuses on important elements. A Transformer, a variant of self-attention architecture, is applied to deep imitation learning to solve dual-arm manipulation tasks in the real world. The proposed method has been tested on dual-arm manipulation tasks using a real robot. The experimental results demonstrated that the Transformer-based deep imitation learning architecture can attend 
    
[^5]: 基于凝视的双分辨率深度模仿学习用于高精度灵巧机器人操作

    Gaze-based dual resolution deep imitation learning for high-precision dexterous robot manipulation

    [https://arxiv.org/abs/2102.01295](https://arxiv.org/abs/2102.01295)

    基于人类基于凝视的双分辨率视觉运动控制系统的启发，提出了一种利用深度模仿学习解决高精度灵巧机器人操作任务的方法

    

    一个高精度操纵任务，如穿针引线，是具有挑战性的。生理学研究提出了将低分辨率外围视觉和快速移动连接起来，将手传送到对象的附近，并使用高分辨率的凹陷视觉来实现手精确对准对象。本研究结果表明，受人类基于凝视的双分辨率视觉运动控制系统的启发，基于深度模仿学习的方法可以解决穿针引线任务。首先，我们记录了远程操作机器人的人类操作员的凝视运动。然后，在靠近目标时，我们仅使用围绕凝视点的高分辨率图像来精确控制线的位置。我们使用低分辨率的外围图像到达目标附近。本研究获得的实验结果表明，所提出的方法实现了精准的操纵

    arXiv:2102.01295v3 Announce Type: replace-cross  Abstract: A high-precision manipulation task, such as needle threading, is challenging. Physiological studies have proposed connecting low-resolution peripheral vision and fast movement to transport the hand into the vicinity of an object, and using high-resolution foveated vision to achieve the accurate homing of the hand to the object. The results of this study demonstrate that a deep imitation learning based method, inspired by the gaze-based dual resolution visuomotor control system in humans, can solve the needle threading task. First, we recorded the gaze movements of a human operator who was teleoperating a robot. Then, we used only a high-resolution image around the gaze to precisely control the thread position when it was close to the target. We used a low-resolution peripheral image to reach the vicinity of the target. The experimental results obtained in this study demonstrate that the proposed method enables precise manipulat
    
[^6]: Text-to-image diffusion models中通过重用注意力映射实现快速推理

    Fast Inference Through The Reuse Of Attention Maps In Diffusion Models. (arXiv:2401.01008v1 [cs.CV])

    [http://arxiv.org/abs/2401.01008](http://arxiv.org/abs/2401.01008)

    本文提出了一种无需训练的方法，通过重用注意力映射来实现Text-to-image diffusion models中的快速推理，以提高效率。

    

    文字到图像扩散模型在灵活和逼真的图像合成方面展示了前所未有的能力。然而，生成单个图像所需的迭代过程既昂贵又具有较高的延迟，促使研究人员进一步研究其效率。我们提出了一种无需调整采样步长的无需训练的方法。具体地说，我们发现重复计算注意力映射既耗时又冗余，因此我们建议在采样过程中结构化地重用注意力映射。我们的初步重用策略受到初级ODE理论的启发，该理论认为在采样过程的后期重用最合适。在注意到这种理论方法的一些局限性后，我们通过实验证明了一种更好的方法。

    Text-to-image diffusion models have demonstrated unprecedented abilities at flexible and realistic image synthesis. However, the iterative process required to produce a single image is costly and incurs a high latency, prompting researchers to further investigate its efficiency. Typically, improvements in latency have been achieved in two ways: (1) training smaller models through knowledge distillation (KD); and (2) adopting techniques from ODE-theory to facilitate larger step sizes. In contrast, we propose a training-free approach that does not alter the step-size of the sampler. Specifically, we find the repeated calculation of attention maps to be both costly and redundant; therefore, we propose a structured reuse of attention maps during sampling. Our initial reuse policy is motivated by rudimentary ODE-theory, which suggests that reuse is most suitable late in the sampling procedure. After noting a number of limitations in this theoretical approach, we empirically search for a bet
    
[^7]: 数字病理学中人工智能的诊断测试准确度：系统综述、Meta分析和质量评估

    Diagnostic test accuracy (DTA) of artificial intelligence in digital pathology: a systematic review, meta-analysis and quality assessment. (arXiv:2306.07999v1 [physics.med-ph])

    [http://arxiv.org/abs/2306.07999](http://arxiv.org/abs/2306.07999)

    本文进行了数字病理图像中应用人工智能的所有病理学领域的诊断准确度的系统综述和Meta分析。结果表明，人工智能在数字病理学中取得了高度的准确度，是可行的辅助诊断工具。

    

    确保临床使用之前AI模型的诊断表现是关键，以确保这些技术的安全和成功的采用。近年来，报道应用于数字病理学图像进行诊断目的的AI研究数量迅速增加。本研究旨在提供数字病理学中AI的诊断准确度的概述，涵盖了所有病理学领域。这项系统性综述和Meta分析包括使用任何类型的人工智能应用于任何疾病类型的WSI图像的诊断准确性研究。参考标准是通过组织病理学评估和/或免疫组化诊断。搜索在2022年6月在PubMed、EMBASE和CENTRAL中进行。在2976项研究中，有100项纳入综述，48项纳入完整的Meta分析。使用QUADAS-2工具评估了偏倚风险和适用性的关注点。数据提取由两个调查员进行，并进行了Meta分析。

    Ensuring diagnostic performance of AI models before clinical use is key to the safe and successful adoption of these technologies. Studies reporting AI applied to digital pathology images for diagnostic purposes have rapidly increased in number in recent years. The aim of this work is to provide an overview of the diagnostic accuracy of AI in digital pathology images from all areas of pathology. This systematic review and meta-analysis included diagnostic accuracy studies using any type of artificial intelligence applied to whole slide images (WSIs) in any disease type. The reference standard was diagnosis through histopathological assessment and / or immunohistochemistry. Searches were conducted in PubMed, EMBASE and CENTRAL in June 2022. We identified 2976 studies, of which 100 were included in the review and 48 in the full meta-analysis. Risk of bias and concerns of applicability were assessed using the QUADAS-2 tool. Data extraction was conducted by two investigators and meta-analy
    
[^8]: 使用指数级别的少量变分参数的张量网络压缩神经网络

    Compressing neural network by tensor network with exponentially fewer variational parameters. (arXiv:2305.06058v1 [cs.LG])

    [http://arxiv.org/abs/2305.06058](http://arxiv.org/abs/2305.06058)

    本文提出了一种通用的压缩方案，将神经网络的可变参数编码为多层张量网络，明显减少了可变参数的数量，并在多个神经网络和数据集上表现出了卓越的压缩性能，以VGG-16的测试精度提高为例。

    

    为了解决神经网络（NN）所包含的巨大可变的参数问题，本文提出了一种将这些参数 encoding 为多层张量网络（TN）的压缩方案。这种方案演示了出色的压缩性能，超过了以浅层张量网络为基础的现有最先进方法。例如，VGG-16中的3个卷积层的大约1000万参数被压缩到具有仅632个参数的TN中，而在CIFAR-10上的测试准确性令人惊喜地提高了81.14％。

    Neural network (NN) designed for challenging machine learning tasks is in general a highly nonlinear mapping that contains massive variational parameters. High complexity of NN, if unbounded or unconstrained, might unpredictably cause severe issues including over-fitting, loss of generalization power, and unbearable cost of hardware. In this work, we propose a general compression scheme that significantly reduces the variational parameters of NN by encoding them to multi-layer tensor networks (TN's) that contain exponentially-fewer free parameters. Superior compression performance of our scheme is demonstrated on several widely-recognized NN's (FC-2, LeNet-5, and VGG-16) and datasets (MNIST and CIFAR-10), surpassing the state-of-the-art method based on shallow tensor networks. For instance, about 10 million parameters in the three convolutional layers of VGG-16 are compressed in TN's with just $632$ parameters, while the testing accuracy on CIFAR-10 is surprisingly improved from $81.14
    

