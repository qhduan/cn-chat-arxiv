# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SliceIt! -- A Dual Simulator Framework for Learning Robot Food Slicing](https://arxiv.org/abs/2404.02569) | 提出了一个用于在模拟环境中安全高效地学习机器人食物切割任务的Dual Simulator框架。 |
| [^2] | [Threats, Attacks, and Defenses in Machine Unlearning: A Survey](https://arxiv.org/abs/2403.13682) | 机器遗忘（MU）通过知识去除过程来解决训练数据相关的人工智能治理问题，提高了AI系统的安全和负责任使用。 |
| [^3] | [AI-enhanced Collective Intelligence: The State of the Art and Prospects](https://arxiv.org/abs/2403.10433) | 人类和人工智能形成的多层次集体智能网络，可以实现超越任一单独实体的集体智能水平。 |
| [^4] | [Quantum Image Denoising with Machine Learning: A Novel Approach to Improve Quantum Image Processing Quality and Reliability](https://arxiv.org/abs/2402.11645) | 提出一种新方法，通过使用机器学习模型识别和校正量子处理图像中的噪声，以提高量子图像处理的质量和可靠性，实现与经典计算机类似的处理结果。 |
| [^5] | [An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Generative LLM Inference](https://arxiv.org/abs/2402.10712) | 通过实证研究，本文探讨了各种跨语言词汇适应方法对提高生成LLM推理效率的影响。 |
| [^6] | [Generative Adversarial Bayesian Optimization for Surrogate Objectives](https://arxiv.org/abs/2402.06532) | 提出了生成对抗贝叶斯优化（GABO）算法，通过使用自适应源批评家正则化，将优化轨迹限制在代理函数可靠的区域内，解决了离线模型基于策略优化中代理模型预测不准确的问题。在多个离线优化任务中，GABO表现优于现有基准方法。 |
| [^7] | [Fast Inference Through The Reuse Of Attention Maps In Diffusion Models.](http://arxiv.org/abs/2401.01008) | 本文提出了一种无需训练的方法，通过重用注意力映射来实现Text-to-image diffusion models中的快速推理，以提高效率。 |
| [^8] | [Learning Interactive Real-World Simulators.](http://arxiv.org/abs/2310.06114) | 通过生成建模学习交互体验的通用模拟器，以模拟人类、机器人和其他交互式代理人对真实世界中行为的响应。 |
| [^9] | [Serving Deep Learning Model in Relational Databases.](http://arxiv.org/abs/2310.04696) | 本文研究了在关系数据库中为深度学习模型提供服务的架构，并强调了三个关键范式：深度学习中心架构、UDF中心架构和关系中心架构。尽管每个架构都在特定的使用场景中有潜力，但还需要解决它们之间的集成问题和中间地带。 |
| [^10] | [Discrete, compositional, and symbolic representations through attractor dynamics.](http://arxiv.org/abs/2310.01807) | 这项工作探讨了如何通过模拟吸引子动力学来更加神经可行地实现离散化，从而将连续的表示空间划分为对应于符号序列的分区。通过引入符号空间结构，可以在丰富的感知输入的吸引子支持表示空间中实现组合性。 |
| [^11] | [Learning to Receive Help: Intervention-Aware Concept Embedding Models.](http://arxiv.org/abs/2309.16928) | 这项研究提出了一种干预感知的概念嵌入模型，用于提高神经架构对概念干预的响应性，并解决了概念干预顺序和模型架构的依赖性的问题。 |
| [^12] | [Alternative Telescopic Displacement: An Efficient Multimodal Alignment Method.](http://arxiv.org/abs/2306.16950) | 备选的变焦位移是一种高效的多模态对齐方法，通过交替移动和扩展特征信息来融合多模态数据，可以稳健地捕捉不同模态特征之间的高级交互作用，从而显著提高多模态学习的性能，并在多个任务上优于其他流行的多模态方案。 |
| [^13] | [Revolutionizing Agrifood Systems with Artificial Intelligence: A Survey.](http://arxiv.org/abs/2305.01899) | 这篇论文探讨了人工智能技术在农业食品系统中的应用，重点关注了农业、畜牧业和渔业等领域。人工智能技术在农业食品分类、生长监测、产量预测和品质评估等方面表现出强大的能力，同时也提出了未来研究方向以及潜在的挑战和限制。 |
| [^14] | [Bayesian Matrix Decomposition and Applications.](http://arxiv.org/abs/2302.11337) | 本书旨在介绍贝叶斯矩阵分解的概念和工具，并总结了贝叶斯矩阵分解方法在不同领域的应用。 |

# 详细

[^1]: SliceIt! -- 一个用于学习机器人食物切割的双重模拟器框架

    SliceIt! -- A Dual Simulator Framework for Learning Robot Food Slicing

    [https://arxiv.org/abs/2404.02569](https://arxiv.org/abs/2404.02569)

    提出了一个用于在模拟环境中安全高效地学习机器人食物切割任务的Dual Simulator框架。

    

    厨房机器人可以通过减轻日常烦琐任务的负担，提高家庭体验。然而，在处理危险工具（如厨房刀具）时，这些机器人必须在共享人类环境中灵巧且安全地执行任务。本研究旨在使机器人能够自主且安全地学习食物切割任务。具体来说，我们的目标是通过适应性控制，使用协作机器人或工业机器人手臂执行食物切割任务，以适应不同材料属性。我们的方法包括使用强化学习（RL）训练机器人以合规方式操作刀具，减少食物和切菜板施加的接触力。然而，在现实世界中训练机器人可能效率低下、危险，并导致大量食物浪费。因此，我们提出了SliceIt!，这是一个用于在模拟环境中安全高效地学习机器人食物切割任务的框架。

    arXiv:2404.02569v1 Announce Type: cross  Abstract: Cooking robots can enhance the home experience by reducing the burden of daily chores. However, these robots must perform their tasks dexterously and safely in shared human environments, especially when handling dangerous tools such as kitchen knives. This study focuses on enabling a robot to autonomously and safely learn food-cutting tasks. More specifically, our goal is to enable a collaborative robot or industrial robot arm to perform food-slicing tasks by adapting to varying material properties using compliance control. Our approach involves using Reinforcement Learning (RL) to train a robot to compliantly manipulate a knife, by reducing the contact forces exerted by the food items and by the cutting board. However, training the robot in the real world can be inefficient, and dangerous, and result in a lot of food waste. Therefore, we proposed SliceIt!, a framework for safely and efficiently learning robot food-slicing tasks in sim
    
[^2]: 机器学习中的威胁、攻击和防御：一项调查

    Threats, Attacks, and Defenses in Machine Unlearning: A Survey

    [https://arxiv.org/abs/2403.13682](https://arxiv.org/abs/2403.13682)

    机器遗忘（MU）通过知识去除过程来解决训练数据相关的人工智能治理问题，提高了AI系统的安全和负责任使用。

    

    机器遗忘（MU）最近引起了相当大的关注，因为它有潜力通过从训练的机器学习模型中消除特定数据的影响来实现安全人工智能。这个被称为知识去除的过程解决了与训练数据相关的人工智能治理问题，如数据质量、敏感性、版权限制和过时性。这种能力对于确保遵守诸如被遗忘权等隐私法规也至关重要。此外，有效的知识去除有助于减轻有害结果的风险，防范偏见、误导和未经授权的数据利用，从而增强了AI系统的安全和负责任使用。已经开展了设计高效的遗忘方法的工作，通过研究MU服务以与现有的机器学习作为服务集成，使用户能够提交请求从训练语料库中删除特定数据。

    arXiv:2403.13682v2 Announce Type: replace-cross  Abstract: Machine Unlearning (MU) has gained considerable attention recently for its potential to achieve Safe AI by removing the influence of specific data from trained machine learning models. This process, known as knowledge removal, addresses AI governance concerns of training data such as quality, sensitivity, copyright restrictions, and obsolescence. This capability is also crucial for ensuring compliance with privacy regulations such as the Right To Be Forgotten. Furthermore, effective knowledge removal mitigates the risk of harmful outcomes, safeguarding against biases, misinformation, and unauthorized data exploitation, thereby enhancing the safe and responsible use of AI systems. Efforts have been made to design efficient unlearning approaches, with MU services being examined for integration with existing machine learning as a service, allowing users to submit requests to remove specific data from the training corpus. However, 
    
[^3]: AI增强的集体智能：现状与展望

    AI-enhanced Collective Intelligence: The State of the Art and Prospects

    [https://arxiv.org/abs/2403.10433](https://arxiv.org/abs/2403.10433)

    人类和人工智能形成的多层次集体智能网络，可以实现超越任一单独实体的集体智能水平。

    

    目前的社会挑战超出了人类个体或集体努力的能力。随着人工智能的发展，其在人类集体中的角色将从辅助工具转变为参与式成员。人类和人工智能拥有互补的能力，当二者协同作用时，可以实现一种超越单独人类或人工智能集体能力的集体智能水平。然而，人工智能系统中的交互本质上是复杂的，涉及复杂的过程和相互依赖关系。本综述从网络科学的视角出发，构想了一个多层次的人工智能集体智能表示，包括认知层、物理层和信息层。在这个多层网络中，人类和人工智能代理展现出不同的特征；人类在多样性方面从表层到深层属性不同，而人工智能代理在程度上也有所区别。

    arXiv:2403.10433v1 Announce Type: cross  Abstract: The current societal challenges exceed the capacity of human individual or collective effort alone. As AI evolves, its role within human collectives is poised to vary from an assistive tool to a participatory member. Humans and AI possess complementary capabilities that, when synergized, can achieve a level of collective intelligence that surpasses the collective capabilities of either humans or AI in isolation. However, the interactions in human-AI systems are inherently complex, involving intricate processes and interdependencies. This review incorporates perspectives from network science to conceptualize a multilayer representation of human-AI collective intelligence, comprising a cognition layer, a physical layer, and an information layer. Within this multilayer network, humans and AI agents exhibit varying characteristics; humans differ in diversity from surface-level to deep-level attributes, while AI agents range in degrees of f
    
[^4]: 借助机器学习进行量子图像去噪：改进量子图像处理质量和可靠性的新方法

    Quantum Image Denoising with Machine Learning: A Novel Approach to Improve Quantum Image Processing Quality and Reliability

    [https://arxiv.org/abs/2402.11645](https://arxiv.org/abs/2402.11645)

    提出一种新方法，通过使用机器学习模型识别和校正量子处理图像中的噪声，以提高量子图像处理的质量和可靠性，实现与经典计算机类似的处理结果。

    

    Quantum Image Processing（QIP）是一个旨在利用量子计算优势来操作和分析图像的领域。然而，QIP面临两个挑战：量子比特的限制和量子机器中存在的噪声。在这项研究中，我们提出了一种新方法来解决QIP中的噪声问题。通过训练和使用一个机器学习模型，该模型能够识别并校正量子处理图像中的噪声，我们可以弥补机器造成的嘈杂，并以比经典计算机更高的效率检索出与之类似的处理结果。该模型通过学习包括现有处理图像和来自开放获取数据集的量子处理图像的数据集进行训练。该模型将能够为我们提供每个像素的置信水平及其潜在原始值。为了评估该模型在弥补Q处理中的损失和退相干方面的准确性

    arXiv:2402.11645v1 Announce Type: cross  Abstract: Quantum Image Processing (QIP) is a field that aims to utilize the benefits of quantum computing for manipulating and analyzing images. However, QIP faces two challenges: the limitation of qubits and the presence of noise in a quantum machine. In this research we propose a novel approach to address the issue of noise in QIP. By training and employing a machine learning model that identifies and corrects the noise in quantum processed images, we can compensate for the noisiness caused by the machine and retrieve a processing result similar to that performed by a classical computer with higher efficiency. The model is trained by learning a dataset consisting of both existing processed images and quantum processed images from open access datasets. This model will be capable of providing us with the confidence level for each pixel and its potential original value. To assess the model's accuracy in compensating for loss and decoherence in Q
    
[^5]: 一项关于跨语言词汇适应用于高效生成LLM推理的实证研究

    An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Generative LLM Inference

    [https://arxiv.org/abs/2402.10712](https://arxiv.org/abs/2402.10712)

    通过实证研究，本文探讨了各种跨语言词汇适应方法对提高生成LLM推理效率的影响。

    

    arXiv:2402.10712v1 通告类型: 跨领域 摘要: 最先进的生成大型语言模型(LLMs)的发展在很大程度上依赖于英语为中心的分词器、词汇和预训练数据。尽管一些LLMs具有多语言能力，但最近的研究表明，当生成英语以外的其他语言时，它们的推理效率会下降。这导致推理时间和成本增加。已经提出了跨语言词汇适应方法，用于将模型调整到目标语言，旨在提高下游性能。然而，这些方法对提高生成LLM推理效率的有效性尚未得到探究。在本文中，我们对五种生成LLMs（包括单语和多语模型）在四种语言类型多样且四种自然语言理解任务上进行了各种跨语言词汇适应方法的实证研究。

    arXiv:2402.10712v1 Announce Type: cross  Abstract: The development of state-of-the-art generative large language models (LLMs) disproportionately relies on English-centric tokenizers, vocabulary and pre-training data. Despite the fact that some LLMs have multilingual capabilities, recent studies have shown that their inference efficiency deteriorates when generating text in languages other than English. This results in increased inference time and costs. Cross-lingual vocabulary adaptation methods have been proposed for adapting models to a target language aiming to improve downstream performance. However, the effectiveness of these methods on increasing inference efficiency of generative LLMs has yet to be explored. In this paper, we perform an empirical study of various cross-lingual vocabulary adaptation methods on five generative LLMs (including monolingual and multilingual models) across four typologically-diverse languages and four natural language understanding tasks. We find th
    
[^6]: 生成对抗贝叶斯优化用于代理目标

    Generative Adversarial Bayesian Optimization for Surrogate Objectives

    [https://arxiv.org/abs/2402.06532](https://arxiv.org/abs/2402.06532)

    提出了生成对抗贝叶斯优化（GABO）算法，通过使用自适应源批评家正则化，将优化轨迹限制在代理函数可靠的区域内，解决了离线模型基于策略优化中代理模型预测不准确的问题。在多个离线优化任务中，GABO表现优于现有基准方法。

    

    离线基于模型的策略优化通过在优化过程中不查询真实的目标函数来优化学习到的代理目标函数。然而，在优化过程中经常遇到代理模型预测不准确的情况。为了解决这个问题，我们提出了使用自适应源批评家正则化的生成对抗贝叶斯优化（GABO），这是一个任务不可知的贝叶斯优化框架，采用了Lipschitz有界源批评家模型来约束优化轨迹，使其在代理函数可靠的区域内。我们证明，在连续输入空间先验的一定假设下，我们的算法动态调整源批评家正则化的强度。在各种科学领域的多个离线优化任务中，GABO优于现有基准方法。我们的代码可在https://github.com/michael-s-yao/gabo 查询。

    Offline model-based policy optimization seeks to optimize a learned surrogate objective function without querying the true oracle objective during optimization. However, inaccurate surrogate model predictions are frequently encountered along the optimization trajectory. To address this limitation, we propose generative adversarial Bayesian optimization (GABO) using adaptive source critic regularization, a task-agnostic framework for Bayesian optimization that employs a Lipschitz-bounded source critic model to constrain the optimization trajectory to regions where the surrogate function is reliable. We show that under certain assumptions for the continuous input space prior, our algorithm dynamically adjusts the strength of the source critic regularization. GABO outperforms existing baselines on a number of different offline optimization tasks across a variety of scientific domains. Our code is available at https://github.com/michael-s-yao/gabo
    
[^7]: Text-to-image diffusion models中通过重用注意力映射实现快速推理

    Fast Inference Through The Reuse Of Attention Maps In Diffusion Models. (arXiv:2401.01008v1 [cs.CV])

    [http://arxiv.org/abs/2401.01008](http://arxiv.org/abs/2401.01008)

    本文提出了一种无需训练的方法，通过重用注意力映射来实现Text-to-image diffusion models中的快速推理，以提高效率。

    

    文字到图像扩散模型在灵活和逼真的图像合成方面展示了前所未有的能力。然而，生成单个图像所需的迭代过程既昂贵又具有较高的延迟，促使研究人员进一步研究其效率。我们提出了一种无需调整采样步长的无需训练的方法。具体地说，我们发现重复计算注意力映射既耗时又冗余，因此我们建议在采样过程中结构化地重用注意力映射。我们的初步重用策略受到初级ODE理论的启发，该理论认为在采样过程的后期重用最合适。在注意到这种理论方法的一些局限性后，我们通过实验证明了一种更好的方法。

    Text-to-image diffusion models have demonstrated unprecedented abilities at flexible and realistic image synthesis. However, the iterative process required to produce a single image is costly and incurs a high latency, prompting researchers to further investigate its efficiency. Typically, improvements in latency have been achieved in two ways: (1) training smaller models through knowledge distillation (KD); and (2) adopting techniques from ODE-theory to facilitate larger step sizes. In contrast, we propose a training-free approach that does not alter the step-size of the sampler. Specifically, we find the repeated calculation of attention maps to be both costly and redundant; therefore, we propose a structured reuse of attention maps during sampling. Our initial reuse policy is motivated by rudimentary ODE-theory, which suggests that reuse is most suitable late in the sampling procedure. After noting a number of limitations in this theoretical approach, we empirically search for a bet
    
[^8]: 学习交互式现实世界模拟器

    Learning Interactive Real-World Simulators. (arXiv:2310.06114v1 [cs.AI])

    [http://arxiv.org/abs/2310.06114](http://arxiv.org/abs/2310.06114)

    通过生成建模学习交互体验的通用模拟器，以模拟人类、机器人和其他交互式代理人对真实世界中行为的响应。

    

    训练在互联网数据上的生成模型已经彻底改变了文本、图像和视频内容的创建方式。也许生成模型的下一个里程碑是在人类、机器人和其他交互式代理人采取行动时模拟真实的体验。实际应用范围从游戏和电影中的可控内容创建，到仅在模拟环境中训练可以直接部署在现实世界中的体验式代理人。我们探索了通过生成建模来学习现实世界交互的通用模拟器(UniSim)的可能性。我们首先重要地观察到，用于学习现实世界模拟器的自然数据集通常在不同的方面丰富多样（例如，图像数据中丰富的物体，机器人数据中密集采样的动作，导航数据中多样的移动）。通过精心协调各种数据集，每个数据集都提供整体体验的不同方面，UniSim可以模拟人类与环境的交互方式。

    Generative models trained on internet data have revolutionized how text, image, and video content can be created. Perhaps the next milestone for generative models is to simulate realistic experience in response to actions taken by humans, robots, and other interactive agents. Applications of a real-world simulator range from controllable content creation in games and movies, to training embodied agents purely in simulation that can be directly deployed in the real world. We explore the possibility of learning a universal simulator (UniSim) of real-world interaction through generative modeling. We first make the important observation that natural datasets available for learning a real-world simulator are often rich along different axes (e.g., abundant objects in image data, densely sampled actions in robotics data, and diverse movements in navigation data). With careful orchestration of diverse datasets, each providing a different aspect of the overall experience, UniSim can emulate how
    
[^9]: 在关系数据库中为深度学习模型提供服务

    Serving Deep Learning Model in Relational Databases. (arXiv:2310.04696v2 [cs.DB] UPDATED)

    [http://arxiv.org/abs/2310.04696](http://arxiv.org/abs/2310.04696)

    本文研究了在关系数据库中为深度学习模型提供服务的架构，并强调了三个关键范式：深度学习中心架构、UDF中心架构和关系中心架构。尽管每个架构都在特定的使用场景中有潜力，但还需要解决它们之间的集成问题和中间地带。

    

    在不同商业和科学领域中，在关系数据上为深度学习模型提供服务已经成为一个重要需求，并引发了最近日益增长的兴趣。本文通过全面探索代表性架构来满足这个需求。我们强调了三个关键范式：尖端的深度学习中心架构将深度学习计算转移到专用的深度学习框架上。潜在的UDF中心架构将一个或多个张量计算封装到数据库系统中的用户定义函数(UDFs)中。潜在的关系中心架构旨在通过关系运算来表示大规模的张量计算。虽然每个架构在特定的使用场景中都显示出了潜力，但我们确定了这些架构之间的无缝集成和中间地带之间的紧急需求。我们深入研究了妨碍集成的差距，并探索了创新的策略。

    Serving deep learning (DL) models on relational data has become a critical requirement across diverse commercial and scientific domains, sparking growing interest recently. In this visionary paper, we embark on a comprehensive exploration of representative architectures to address the requirement. We highlight three pivotal paradigms: The state-of-the-artDL-Centricarchitecture offloadsDL computations to dedicated DL frameworks. The potential UDF-Centric architecture encapsulates one or more tensor computations into User Defined Functions (UDFs) within the database system. The potentialRelation-Centricarchitecture aims to represent a large-scale tensor computation through relational operators. While each of these architectures demonstrates promise in specific use scenarios, we identify urgent requirements for seamless integration of these architectures and the middle ground between these architectures. We delve into the gaps that impede the integration and explore innovative strategies 
    
[^10]: 通过吸引子动力学实现离散、组合和符号表示

    Discrete, compositional, and symbolic representations through attractor dynamics. (arXiv:2310.01807v1 [cs.AI])

    [http://arxiv.org/abs/2310.01807](http://arxiv.org/abs/2310.01807)

    这项工作探讨了如何通过模拟吸引子动力学来更加神经可行地实现离散化，从而将连续的表示空间划分为对应于符号序列的分区。通过引入符号空间结构，可以在丰富的感知输入的吸引子支持表示空间中实现组合性。

    

    组合性是离散符号系统（如语言和程序）的重要特征，它使得这些系统尽管使用有限的符号集合，但仍具有无限的容量。它在认知科学和人工智能领域的推理中都具有很好的抽象性。然而，连续和符号处理之间的界面通常是通过算法级别上的量化或softmax采样步骤来实现的。在本研究中，我们通过模拟吸引子动力学将离散化实现得更加神经可行，这种方法将连续的表示空间划分为对应于符号序列的分区。在吸引子网络的基础上，引入了新的训练方法，我们展示了在丰富的感知输入的吸引子支持表示空间中引入符号空间结构可以产生组合性。最后，我们认为我们的模型展示了一种信息增长的过程。

    Compositionality is an important feature of discrete symbolic systems, such as language and programs, as it enables them to have infinite capacity despite a finite symbol set. It serves as a useful abstraction for reasoning in both cognitive science and in AI, yet the interface between continuous and symbolic processing is often imposed by fiat at the algorithmic level, such as by means of quantization or a softmax sampling step. In this work, we explore how discretization could be implemented in a more neurally plausible manner through the modeling of attractor dynamics that partition the continuous representation space into basins that correspond to sequences of symbols. Building on established work in attractor networks and introducing novel training methods, we show that imposing structure in the symbolic space can produce compositionality in the attractor-supported representation space of rich sensory inputs. Lastly, we argue that our model exhibits the process of an information b
    
[^11]: 学习接受帮助：干预感知的概念嵌入模型

    Learning to Receive Help: Intervention-Aware Concept Embedding Models. (arXiv:2309.16928v1 [cs.LG])

    [http://arxiv.org/abs/2309.16928](http://arxiv.org/abs/2309.16928)

    这项研究提出了一种干预感知的概念嵌入模型，用于提高神经架构对概念干预的响应性，并解决了概念干预顺序和模型架构的依赖性的问题。

    

    概念瓶颈模型（CBMs）通过使用一组高级概念构建和解释神经架构的预测，以解决其不透明性的问题。这些模型的一个特殊属性是它们允许概念干预，用户可以纠正被错误预测的概念，从而提高模型的性能。然而，最近的研究表明，干预有效性可能严重依赖于干预概念的顺序以及模型的架构和训练超参数。我们认为，这源于CBM在训练时缺乏模型适应概念干预的激励。为了解决这个问题，我们提出了干预感知的概念嵌入模型（IntCEMs），这是一种基于CBM的新型架构和训练范式，可以提高模型对测试时干预的响应性。我们的模型以端到端的方式学习了一个概念干预策略，从中可以采样有意义的干预轨迹。

    Concept Bottleneck Models (CBMs) tackle the opacity of neural architectures by constructing and explaining their predictions using a set of high-level concepts. A special property of these models is that they permit concept interventions, wherein users can correct mispredicted concepts and thus improve the model's performance. Recent work, however, has shown that intervention efficacy can be highly dependent on the order in which concepts are intervened on and on the model's architecture and training hyperparameters. We argue that this is rooted in a CBM's lack of train-time incentives for the model to be appropriately receptive to concept interventions. To address this, we propose Intervention-aware Concept Embedding models (IntCEMs), a novel CBM-based architecture and training paradigm that improves a model's receptiveness to test-time interventions. Our model learns a concept intervention policy in an end-to-end fashion from where it can sample meaningful intervention trajectories a
    
[^12]: 备选的变焦位移：一种高效的多模态对齐方法

    Alternative Telescopic Displacement: An Efficient Multimodal Alignment Method. (arXiv:2306.16950v1 [cs.CV])

    [http://arxiv.org/abs/2306.16950](http://arxiv.org/abs/2306.16950)

    备选的变焦位移是一种高效的多模态对齐方法，通过交替移动和扩展特征信息来融合多模态数据，可以稳健地捕捉不同模态特征之间的高级交互作用，从而显著提高多模态学习的性能，并在多个任务上优于其他流行的多模态方案。

    

    特征对齐是融合多模态数据的主要方式。我们提出了一种特征对齐方法，可以完全融合多模态信息，通过在特征空间中交替移动和扩展来实现不同模态之间的一致表示。所提出的方法能够稳健地捕捉不同模态特征之间的高级交互作用，从而显著提高多模态学习的性能。我们还表明，所提出的方法在多个任务上优于其他流行的多模态方案。对ETT和MIT-BIH-Arrhythmia数据集的实验评估表明，所提出的方法达到了最先进的性能。

    Feature alignment is the primary means of fusing multimodal data. We propose a feature alignment method that fully fuses multimodal information, which alternately shifts and expands feature information from different modalities to have a consistent representation in a feature space. The proposed method can robustly capture high-level interactions between features of different modalities, thus significantly improving the performance of multimodal learning. We also show that the proposed method outperforms other popular multimodal schemes on multiple tasks. Experimental evaluation of ETT and MIT-BIH-Arrhythmia, datasets shows that the proposed method achieves state of the art performance.
    
[^13]: 人工智能革命：农业食品系统的全面调查

    Revolutionizing Agrifood Systems with Artificial Intelligence: A Survey. (arXiv:2305.01899v1 [cs.AI])

    [http://arxiv.org/abs/2305.01899](http://arxiv.org/abs/2305.01899)

    这篇论文探讨了人工智能技术在农业食品系统中的应用，重点关注了农业、畜牧业和渔业等领域。人工智能技术在农业食品分类、生长监测、产量预测和品质评估等方面表现出强大的能力，同时也提出了未来研究方向以及潜在的挑战和限制。

    

    随着全球人口的迅速增长，转变农业食品系统，使其更具生产力、效率、安全和可持续性，是缓解潜在粮食短缺的关键。最近，深度学习等人工智能技术在语言、视觉、遥感和农业食品系统应用等各个领域均显示出了其强大的能力。然而，人工智能对农业食品系统的整体影响仍不清楚。本文全面回顾了人工智能技术如何改变农业食品系统，并为现代农业食品行业做出贡献。首先，我们总结了农业食品系统中的数据获取方法，包括获取、存储和处理技术。其次，我们详细介绍了人工智能技术在农业、畜牧业和渔业等领域中的进展情况，涵盖了农业食品分类、生长监测、产量预测和品质评估等主题。此外，我们还提出了未来研究方向，并讨论了在农业食品系统应用人工智能技术中潜在的挑战和限制。

    With the world population rapidly increasing, transforming our agrifood systems to be more productive, efficient, safe, and sustainable is crucial to mitigate potential food shortages. Recently, artificial intelligence (AI) techniques such as deep learning (DL) have demonstrated their strong abilities in various areas, including language, vision, remote sensing (RS), and agrifood systems applications. However, the overall impact of AI on agrifood systems remains unclear. In this paper, we thoroughly review how AI techniques can transform agrifood systems and contribute to the modern agrifood industry. Firstly, we summarize the data acquisition methods in agrifood systems, including acquisition, storage, and processing techniques. Secondly, we present a progress review of AI methods in agrifood systems, specifically in agriculture, animal husbandry, and fishery, covering topics such as agrifood classification, growth monitoring, yield prediction, and quality assessment. Furthermore, we 
    
[^14]: 贝叶斯矩阵分解及应用

    Bayesian Matrix Decomposition and Applications. (arXiv:2302.11337v2 [math.NA] UPDATED)

    [http://arxiv.org/abs/2302.11337](http://arxiv.org/abs/2302.11337)

    本书旨在介绍贝叶斯矩阵分解的概念和工具，并总结了贝叶斯矩阵分解方法在不同领域的应用。

    

    本书的唯一目的是为了给出贝叶斯矩阵分解概念和数学工具的自包含介绍，以便在后续章节中无缝引入矩阵分解技术及其应用。然而，我们清楚地意识到我们无法覆盖关于贝叶斯矩阵分解的所有有用和有趣的结果，并且由于讨论的范围有限，例如分析变分推理以进行优化的分离分析。我们将读者引导到贝叶斯分析领域的文献中，以便更详细地介绍相关领域。本书主要总结了重要的贝叶斯矩阵分解方法（例如实值分解、非负矩阵分解、贝叶斯插值分解）的目的和意义，以及这些方法的起源和复杂性对其应用提供的启示。数学先决条件是第一门课程。

    The sole aim of this book is to give a self-contained introduction to concepts and mathematical tools in Bayesian matrix decomposition in order to seamlessly introduce matrix decomposition techniques and their applications in subsequent sections. However, we clearly realize our inability to cover all the useful and interesting results concerning Bayesian matrix decomposition and given the paucity of scope to present this discussion, e.g., the separated analysis of variational inference for conducting the optimization. We refer the reader to literature in the field of Bayesian analysis for a more detailed introduction to the related fields.  This book is primarily a summary of purpose, significance of important Bayesian matrix decomposition methods, e.g., real-valued decomposition, nonnegative matrix factorization, Bayesian interpolative decomposition, and the origin and complexity of the methods which shed light on their applications. The mathematical prerequisite is a first course in 
    

