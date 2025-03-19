# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis](https://arxiv.org/abs/2403.13501) | 提出了一种名为VSTAR的方法，通过引入生成时序护理（GTN）的概念，自动生成视频梗概并改善对时序动态的控制，从而实现生成更长、更动态的视频 |
| [^2] | [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029) | 引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。 |
| [^3] | [CoTBal: Comprehensive Task Balancing for Multi-Task Visual Instruction Tuning](https://arxiv.org/abs/2403.04343) | 提出了一种全面任务平衡算法（CoTBal）用于大型多模态模型的多任务视觉指令调整，首次探索了视觉指令调整中的多任务优化。 |
| [^4] | [Semi-Supervised Semantic Segmentation Based on Pseudo-Labels: A Survey](https://arxiv.org/abs/2403.01909) | 这项综述提供了关于基于伪标签方法在半监督语义分割领域最新研究成果的全面且有组织的概述，探讨了伪标签技术在不同应用领域的具体方法，还研究了其在医学和遥感图像分割中的应用，提出了未来研究方向。 |
| [^5] | [Q-FOX Learning: Breaking Tradition in Reinforcement Learning](https://arxiv.org/abs/2402.16562) | Q-FOX学习是一种新颖的自动超参数调整方法，结合了FOX优化器和Q-learning算法，提出了使用新的目标函数来解决强化学习中超参数调整的问题。 |
| [^6] | [Hypergraph Neural Networks through the Lens of Message Passing: A Common Perspective to Homophily and Architecture Design](https://arxiv.org/abs/2310.07684) | 本文通过消息传递机制的视角，提出了一种新的对高阶网络同質性的概念化，并探索了一些处理高阶结构的策略，为超图神经网络的架构设计和性能提供了新的视野和方法。 |
| [^7] | [A RAG-based Question Answering System Proposal for Understanding Islam: MufassirQAS LLM.](http://arxiv.org/abs/2401.15378) | 基于RAG的MufassirQAS问答系统利用NLP技术建立联系并准确回答复杂问题，提高了LLMs的准确性和透明度，帮助理解伊斯兰教的复杂性和教义深度。 |
| [^8] | [Domain-Independent Dynamic Programming.](http://arxiv.org/abs/2401.13883) | 本文提出了一种领域无关的动态规划方法，并介绍了基于状态转移系统的动态规划描述语言。实验证明，该方法在许多组合优化问题上优于传统的混合整数规划和约束规划方法。 |
| [^9] | [StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling.](http://arxiv.org/abs/2310.17042) | StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。 |
| [^10] | [Efficient Last-iterate Convergence Algorithms in Solving Games.](http://arxiv.org/abs/2308.11256) | 该论文研究了求解博弈中高效收敛算法的问题，通过分析乐观梯度下降上升（OGDA）和乐观乘法权重更新（OMWU）算法，以及基于奖励转化（RT）框架的算法，提出了解决这些问题的方法。 |
| [^11] | [On the Transition from Neural Representation to Symbolic Knowledge.](http://arxiv.org/abs/2308.02000) | 该论文提出了一种神经-符号过渡字典学习框架，可以将神经网络与符号思维进行结合。通过学习过渡表示，并自监督地发现隐含的谓词结构，以及通过博弈和强化学习调整学习到的原型，该框架可以实现对高维信息的压缩和符号表示的学习。 |
| [^12] | [Emerging Synergies in Causality and Deep Generative Models: A Survey.](http://arxiv.org/abs/2301.12351) | 这项综述探讨了因果性和深度生成模型之间的新兴协同作用，阐明了将因果性原则融入DGM中的方法，以及在大规模生成模型中应用因果性的研究前沿。 |
| [^13] | [Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness.](http://arxiv.org/abs/2208.06648) | 本文研究了填补选择对不同群体的重建误差和下游预测的算法公平性属性的影响。 |

# 详细

[^1]: VSTAR：用于生成长动态视频合成的时间护理

    VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis

    [https://arxiv.org/abs/2403.13501](https://arxiv.org/abs/2403.13501)

    提出了一种名为VSTAR的方法，通过引入生成时序护理（GTN）的概念，自动生成视频梗概并改善对时序动态的控制，从而实现生成更长、更动态的视频

    

    尽管在文本到视频（T2V）合成领域取得了巨大进展，但开源的T2V扩散模型难以生成具有动态变化和不断进化内容的较长视频。它们往往合成准静态视频，忽略了文本提示中涉及的必要随时间变化的视觉变化。与此同时，将这些模型扩展到实现更长、更动态的视频合成往往在计算上难以处理。为了解决这一挑战，我们引入了生成时序护理（GTN）的概念，旨在在推理过程中即时改变生成过程，以改善对时序动态的控制，并实现生成更长的视频。我们提出了一种GTN方法，名为VSTAR，它包括两个关键要素：1）视频梗概提示（VSP）-基于原始单个提示自动生成视频梗概，利用LLMs提供准确的文本指导，以实现对时序动态的精确控制。

    arXiv:2403.13501v1 Announce Type: cross  Abstract: Despite tremendous progress in the field of text-to-video (T2V) synthesis, open-sourced T2V diffusion models struggle to generate longer videos with dynamically varying and evolving content. They tend to synthesize quasi-static videos, ignoring the necessary visual change-over-time implied in the text prompt. At the same time, scaling these models to enable longer, more dynamic video synthesis often remains computationally intractable. To address this challenge, we introduce the concept of Generative Temporal Nursing (GTN), where we aim to alter the generative process on the fly during inference to improve control over the temporal dynamics and enable generation of longer videos. We propose a method for GTN, dubbed VSTAR, which consists of two key ingredients: 1) Video Synopsis Prompting (VSP) - automatic generation of a video synopsis based on the original single prompt leveraging LLMs, which gives accurate textual guidance to differe
    
[^2]: 对齐与提炼：统一和改进领域自适应目标检测

    Align and Distill: Unifying and Improving Domain Adaptive Object Detection

    [https://arxiv.org/abs/2403.12029](https://arxiv.org/abs/2403.12029)

    引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。

    

    目标检测器通常表现不佳于与其训练集不同的数据。最近，领域自适应目标检测（DAOD）方法已经展示了在应对这一挑战上的强大结果。遗憾的是，我们发现了系统化的基准测试陷阱，这些陷阱对过去的结果提出质疑并阻碍了进一步的进展：（a）由于基线不足导致性能高估，（b）不一致的实现实践阻止了方法的透明比较，（c）由于过时的骨干和基准测试缺乏多样性，导致缺乏普遍性。我们通过引入以下问题来解决这些问题：（1）一个统一的基准测试和实现框架，Align and Distill（ALDI），支持DAOD方法的比较并支持未来发展，（2）一个公平且现代的DAOD训练和评估协议，解决了基准测试的陷阱，（3）一个新的DAOD基准数据集，CFC-DAOD，能够在多样化的真实环境中进行评估。

    arXiv:2403.12029v1 Announce Type: cross  Abstract: Object detectors often perform poorly on data that differs from their training set. Domain adaptive object detection (DAOD) methods have recently demonstrated strong results on addressing this challenge. Unfortunately, we identify systemic benchmarking pitfalls that call past results into question and hamper further progress: (a) Overestimation of performance due to underpowered baselines, (b) Inconsistent implementation practices preventing transparent comparisons of methods, and (c) Lack of generality due to outdated backbones and lack of diversity in benchmarks. We address these problems by introducing: (1) A unified benchmarking and implementation framework, Align and Distill (ALDI), enabling comparison of DAOD methods and supporting future development, (2) A fair and modern training and evaluation protocol for DAOD that addresses benchmarking pitfalls, (3) A new DAOD benchmark dataset, CFC-DAOD, enabling evaluation on diverse real
    
[^3]: CoTBal: 多任务视觉指令调整的全面任务平衡

    CoTBal: Comprehensive Task Balancing for Multi-Task Visual Instruction Tuning

    [https://arxiv.org/abs/2403.04343](https://arxiv.org/abs/2403.04343)

    提出了一种全面任务平衡算法（CoTBal）用于大型多模态模型的多任务视觉指令调整，首次探索了视觉指令调整中的多任务优化。

    

    arXiv:2403.04343v1 公告类型: 新   摘要: 视觉指令调整是大型多模态模型（LMMs）的关键训练阶段。然而，无差别混合来自各种任务的指令跟随数据的普遍做法可能导致由于任务之间的指令格式和知识领域不同而导致整体性能不佳。为了缓解这个问题，我们提出了一种新颖的全面任务平衡（CoTBal）算法，用于LMMs的多任务视觉指令调整。据我们所知，这是第一项探索视觉指令调整中多任务优化的工作。具体地，我们考虑任务平衡的两个关键维度:（1）任务间贡献，即学习一个任务可能增强其他任务的性能的现象，归因于重叠的知识领域，以及（2）任务内难度，指的是单个任务内的学习难度。通过用基于性能的方法量化这两个维度

    arXiv:2403.04343v1 Announce Type: new  Abstract: Visual instruction tuning is a key training stage of large multimodal models (LMMs). Nevertheless, the common practice of indiscriminately mixing instruction-following data from various tasks may result in suboptimal overall performance due to different instruction formats and knowledge domains across tasks. To mitigate this issue, we propose a novel Comprehensive Task Balancing (CoTBal) algorithm for multi-task visual instruction tuning of LMMs. To our knowledge, this is the first work that explores multi-task optimization in visual instruction tuning. Specifically, we consider two key dimensions for task balancing: (1) Inter-Task Contribution, the phenomenon where learning one task potentially enhances the performance in other tasks, attributable to the overlapping knowledge domains, and (2) Intra-Task Difficulty, which refers to the learning difficulty within a single task. By quantifying these two dimensions with performance-based me
    
[^4]: 基于伪标签的半监督语义分割：综述

    Semi-Supervised Semantic Segmentation Based on Pseudo-Labels: A Survey

    [https://arxiv.org/abs/2403.01909](https://arxiv.org/abs/2403.01909)

    这项综述提供了关于基于伪标签方法在半监督语义分割领域最新研究成果的全面且有组织的概述，探讨了伪标签技术在不同应用领域的具体方法，还研究了其在医学和遥感图像分割中的应用，提出了未来研究方向。

    

    语义分割是计算机视觉中一个重要且热门的研究领域，侧重于基于语义对图像中的像素进行分类。然而，监督学习需要大量数据来训练模型，而逐像素标记图像的过程耗时且繁琐。本综述旨在提供半监督语义分割领域中伪标签方法的最新研究成果的首次综合和有组织的概述，我们从不同角度对其进行分类，并提出了针对特定应用领域的具体方法。此外，我们还探讨了伪标签技术在医学和遥感图像分割中的应用。最后，我们还提出了一些可行的未来研究方向，以解决现有挑战。

    arXiv:2403.01909v1 Announce Type: cross  Abstract: Semantic segmentation is an important and popular research area in computer vision that focuses on classifying pixels in an image based on their semantics. However, supervised deep learning requires large amounts of data to train models and the process of labeling images pixel by pixel is time-consuming and laborious. This review aims to provide a first comprehensive and organized overview of the state-of-the-art research results on pseudo-label methods in the field of semi-supervised semantic segmentation, which we categorize from different perspectives and present specific methods for specific application areas. In addition, we explore the application of pseudo-label technology in medical and remote-sensing image segmentation. Finally, we also propose some feasible future research directions to address the existing challenges.
    
[^5]: Q-FOX学习：颠覆传统的强化学习

    Q-FOX Learning: Breaking Tradition in Reinforcement Learning

    [https://arxiv.org/abs/2402.16562](https://arxiv.org/abs/2402.16562)

    Q-FOX学习是一种新颖的自动超参数调整方法，结合了FOX优化器和Q-learning算法，提出了使用新的目标函数来解决强化学习中超参数调整的问题。

    

    强化学习（RL）是人工智能（AI）的一个子集，代理通过与环境的交互来学习最佳动作，因此适用于不需要标记数据或直接监督的任务。 本文提出了一种名为Q-FOX的新颖自动调参方法，该方法使用了FOX优化器和常用的易于实现的RL Q-learning算法解决了调参的问题。此外，还提出了一个新的目标函数，该函数将奖励放在均方误差（MSE）和学习时间之上。

    arXiv:2402.16562v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) is a subset of artificial intelligence (AI) where agents learn the best action by interacting with the environment, making it suitable for tasks that do not require labeled data or direct supervision. Hyperparameters (HP) tuning refers to choosing the best parameter that leads to optimal solutions in RL algorithms. Manual or random tuning of the HP may be a crucial process because variations in this parameter lead to changes in the overall learning aspects and different rewards. In this paper, a novel and automatic HP-tuning method called Q-FOX is proposed. This uses both the FOX optimizer, a new optimization method inspired by nature that mimics red foxes' hunting behavior, and the commonly used, easy-to-implement RL Q-learning algorithm to solve the problem of HP tuning. Moreover, a new objective function is proposed which prioritizes the reward over the mean squared error (MSE) and learning time (
    
[^6]: 透过消息传递的视角看超图神经网络：同質性与架构设计的共同视野

    Hypergraph Neural Networks through the Lens of Message Passing: A Common Perspective to Homophily and Architecture Design

    [https://arxiv.org/abs/2310.07684](https://arxiv.org/abs/2310.07684)

    本文通过消息传递机制的视角，提出了一种新的对高阶网络同質性的概念化，并探索了一些处理高阶结构的策略，为超图神经网络的架构设计和性能提供了新的视野和方法。

    

    当前大部分的超图学习方法和基准数据集都是通过从图的类比中提升过来的，忽略了超图的特殊性。本文尝试解决一些相关的问题：Q1 同質性在超图神经网络中是否起到了关键作用？Q2 是否可以通过细致处理高阶网络的特征来改善当前的超图神经网络架构？Q3 现有数据集是否对超图神经网络提供了有意义的基准？为了解决这些问题，我们首先引入了基于消息传递机制的高阶网络同質性的新概念化，统一了高阶网络的分析和建模。此外，我们还研究了在超图神经网络中处理高阶结构的一些自然但大部分未被探索的策略，比如保留超边依赖的节点表示，或是以节点和超边共同编码的方式进行处理。

    Most of the current hypergraph learning methodologies and benchmarking datasets in the hypergraph realm are obtained by lifting procedures from their graph analogs, leading to overshadowing specific characteristics of hypergraphs. This paper attempts to confront some pending questions in that regard: Q1 Can the concept of homophily play a crucial role in Hypergraph Neural Networks (HNNs)? Q2 Is there room for improving current HNN architectures by carefully addressing specific characteristics of higher-order networks? Q3 Do existing datasets provide a meaningful benchmark for HNNs? To address them, we first introduce a novel conceptualization of homophily in higher-order networks based on a Message Passing (MP) scheme, unifying both the analytical examination and the modeling of higher-order networks. Further, we investigate some natural, yet mostly unexplored, strategies for processing higher-order structures within HNNs such as keeping hyperedge-dependent node representations, or per
    
[^7]: 基于RAG的理解伊斯兰教问题回答系统提案：MufassirQAS LLM

    A RAG-based Question Answering System Proposal for Understanding Islam: MufassirQAS LLM. (arXiv:2401.15378v1 [cs.CL])

    [http://arxiv.org/abs/2401.15378](http://arxiv.org/abs/2401.15378)

    基于RAG的MufassirQAS问答系统利用NLP技术建立联系并准确回答复杂问题，提高了LLMs的准确性和透明度，帮助理解伊斯兰教的复杂性和教义深度。

    

    学习和理解宗教存在复杂性和教义深度的挑战。问答机器人作为解决这些挑战的问题回答系统，可以帮助。LLM聊天机器人利用自然语言处理技术建立主题之间的联系，准确回答复杂问题。这些能力使其成为用于宗教启蒙的问题回答聊天机器人的理想选择。然而，LLM也有生成虚假信息的倾向，称为幻觉。聊天机器人的回答可能包含侮辱个人宗教信仰、跨宗派冲突和有争议或敏感的话题的内容。它需要避免这种情况，而不会宣扬仇恨言论或冒犯某些群体的人或他们的信仰。本研究使用基于向量数据库的检索增强生成（RAG）方法来提高LLMs的准确性和透明度。我们的问答系统称为"MufassirQAS"。我们创建了一个模型来评估该系统并证明其在解决宗教行业问题中的效果。

    There exist challenges in learning and understanding religions as the presence of complexity and depth of religious doctrines and teachings. Chatbots as question-answering systems can help in solving these challenges. LLM chatbots use NLP techniques to establish connections between topics and accurately respond to complex questions. These capabilities make it perfect to be used in enlightenment on religion as a question answering chatbot. However, LLMs also have a tendency to generate false information, known as hallucination. The responses of the chatbots can include content that insults personal religious beliefs, interfaith conflicts, and controversial or sensitive topics. It needs to avoid such cases without promoting hate speech or offending certain groups of people or their beliefs. This study uses a vector database-based Retrieval Augmented Generation (RAG) approach to enhance the accuracy and transparency of LLMs. Our question-answering system is called as "MufassirQAS". We cre
    
[^8]: 领域无关的动态规划方法

    Domain-Independent Dynamic Programming. (arXiv:2401.13883v1 [cs.AI])

    [http://arxiv.org/abs/2401.13883](http://arxiv.org/abs/2401.13883)

    本文提出了一种领域无关的动态规划方法，并介绍了基于状态转移系统的动态规划描述语言。实验证明，该方法在许多组合优化问题上优于传统的混合整数规划和约束规划方法。

    

    对于组合优化问题，基于模型的范例如混合整数规划 (MIP) 和约束规划 (CP) 旨在解耦问题的建模和求解过程，这是声明性问题求解的“圣杯”。我们提出了领域无关的动态规划（DIDP），这是一种基于动态规划 (DP) 的新的基于模型的方法。虽然DP并不新鲜，但通常它被作为一种特定问题的方法来实现。我们引入了动态规划描述语言 (DyPDL)，一种基于状态转移系统的形式化语言，灵感来自于AI规划。我们展示了启发式搜索算法可以用来求解DyPDL模型，并提出了七种DIDP求解器。我们在常见的11个组合优化问题类别的基准实例上，将我们的DIDP求解器与商业MIP和CP求解器进行了实验比较（分别求解MIP和CP模型）。结果显示DIDP在九个问题类别中优于MIP，也优于CP在九个问题类别中。

    For combinatorial optimization problems, model-based paradigms such as mixed-integer programming (MIP) and constraint programming (CP) aim to decouple modeling and solving a problem: the `holy grail' of declarative problem solving. We propose domain-independent dynamic programming (DIDP), a new model-based paradigm based on dynamic programming (DP). While DP is not new, it has typically been implemented as a problem-specific method. We introduce Dynamic Programming Description Language (DyPDL), a formalism to define DP models based on a state transition system, inspired by AI planning. We show that heuristic search algorithms can be used to solve DyPDL models and propose seven DIDP solvers. We experimentally compare our DIDP solvers with commercial MIP and CP solvers (solving MIP and CP models, respectively) on common benchmark instances of eleven combinatorial optimization problem classes. We show that DIDP outperforms MIP in nine problem classes, CP also in nine problem classes, and 
    
[^9]: StochGradAdam: 利用随机梯度抽样加速神经网络训练

    StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling. (arXiv:2310.17042v1 [cs.LG])

    [http://arxiv.org/abs/2310.17042](http://arxiv.org/abs/2310.17042)

    StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。

    

    在深度学习优化领域中，本文介绍了StochGradAdam优化器，这是对广受赞誉的Adam算法的新颖改进。StochGradAdam的核心是其梯度抽样技术。该方法不仅确保稳定收敛，而且利用选择性梯度考虑的优势，通过减轻噪声或异常数据的影响和增强损失函数空间的探索，提升了鲁棒训练。在图像分类和分割任务中，StochGradAdam表现出优于传统Adam优化器的性能。通过在每次迭代中精心选择一部分梯度进行抽样，该优化器能够有效应对复杂模型的管理。本文从数学基础到偏差校正策略全面探讨了StochGradAdam的方法，展示了深度学习训练技术的可期进展。

    In the rapidly advancing domain of deep learning optimization, this paper unveils the StochGradAdam optimizer, a novel adaptation of the well-regarded Adam algorithm. Central to StochGradAdam is its gradient sampling technique. This method not only ensures stable convergence but also leverages the advantages of selective gradient consideration, fostering robust training by potentially mitigating the effects of noisy or outlier data and enhancing the exploration of the loss landscape for more dependable convergence. In both image classification and segmentation tasks, StochGradAdam has demonstrated superior performance compared to the traditional Adam optimizer. By judiciously sampling a subset of gradients at each iteration, the optimizer is optimized for managing intricate models. The paper provides a comprehensive exploration of StochGradAdam's methodology, from its mathematical foundations to bias correction strategies, heralding a promising advancement in deep learning training tec
    
[^10]: 在求解博弈中的高效收敛算法

    Efficient Last-iterate Convergence Algorithms in Solving Games. (arXiv:2308.11256v1 [cs.GT])

    [http://arxiv.org/abs/2308.11256](http://arxiv.org/abs/2308.11256)

    该论文研究了求解博弈中高效收敛算法的问题，通过分析乐观梯度下降上升（OGDA）和乐观乘法权重更新（OMWU）算法，以及基于奖励转化（RT）框架的算法，提出了解决这些问题的方法。

    

    无悔算法在学习两人零和标准型游戏和扩展型游戏的纳什均衡中很受欢迎。最近的许多研究考虑了最后一次迭代收敛的无悔算法。其中，最有名的两个算法是乐观梯度下降上升（OGDA）和乐观乘法权重更新（OMWU）。然而，OGDA的每次迭代复杂度很高。OMWU具有较低的每次迭代复杂度，但实验性能较差，并且它的收敛仅在纳什均衡唯一时成立。最近的研究提出了一种基于奖励转化（RT）框架用于MWU，它消除了唯一性条件，并且在与OMWU相同迭代次数的情况下实现了有竞争力的性能。不幸的是，基于RT的算法在相同迭代次数下表现不如OGDA，并且它们的收敛保证基于连续时间反馈假设，这在大多数情况下不成立。为了解决这些问题，我们对RT框架进行了更详细的分析。

    No-regret algorithms are popular for learning Nash equilibrium (NE) in two-player zero-sum normal-form games (NFGs) and extensive-form games (EFGs). Many recent works consider the last-iterate convergence no-regret algorithms. Among them, the two most famous algorithms are Optimistic Gradient Descent Ascent (OGDA) and Optimistic Multiplicative Weight Update (OMWU). However, OGDA has high per-iteration complexity. OMWU exhibits a lower per-iteration complexity but poorer empirical performance, and its convergence holds only when NE is unique. Recent works propose a Reward Transformation (RT) framework for MWU, which removes the uniqueness condition and achieves competitive performance with OMWU. Unfortunately, RT-based algorithms perform worse than OGDA under the same number of iterations, and their convergence guarantee is based on the continuous-time feedback assumption, which does not hold in most scenarios. To address these issues, we provide a closer analysis of the RT framework, w
    
[^11]: 从神经表示到符号知识的过渡

    On the Transition from Neural Representation to Symbolic Knowledge. (arXiv:2308.02000v1 [cs.AI])

    [http://arxiv.org/abs/2308.02000](http://arxiv.org/abs/2308.02000)

    该论文提出了一种神经-符号过渡字典学习框架，可以将神经网络与符号思维进行结合。通过学习过渡表示，并自监督地发现隐含的谓词结构，以及通过博弈和强化学习调整学习到的原型，该框架可以实现对高维信息的压缩和符号表示的学习。

    

    弥合神经表示与符号表示之间的巨大差距可能使符号思维从本质上融入神经网络。受人类如何逐渐从通过知觉和环境交互学习到的原型符号构建复杂的符号表示的启发，我们提出了一种神经-符号过渡字典学习（TDL）框架，该框架使用EM算法学习数据的过渡表示，将输入的高维视觉部分信息压缩到一组张量作为神经变量，并自监督地发现隐含的谓词结构。我们通过将输入分解视为合作博弈来实现框架，使用扩散模型学习谓词，并通过RL基于扩散模型的马尔可夫性质进一步调整学习到的原型，以融入主观因素。

    Bridging the huge disparity between neural and symbolic representation can potentially enable the incorporation of symbolic thinking into neural networks from essence. Motivated by how human gradually builds complex symbolic representation from the prototype symbols that are learned through perception and environmental interactions. We propose a Neural-Symbolic Transitional Dictionary Learning (TDL) framework that employs an EM algorithm to learn a transitional representation of data that compresses high-dimension information of visual parts of an input into a set of tensors as neural variables and discover the implicit predicate structure in a self-supervised way. We implement the framework with a diffusion model by regarding the decomposition of input as a cooperative game, then learn predicates by prototype clustering. We additionally use RL enabled by the Markovian of diffusion models to further tune the learned prototypes by incorporating subjective factors. Extensive experiments 
    
[^12]: 因果性和深度生成模型中的新兴协同作用：一项综述

    Emerging Synergies in Causality and Deep Generative Models: A Survey. (arXiv:2301.12351v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12351](http://arxiv.org/abs/2301.12351)

    这项综述探讨了因果性和深度生成模型之间的新兴协同作用，阐明了将因果性原则融入DGM中的方法，以及在大规模生成模型中应用因果性的研究前沿。

    

    在人工智能领域，了解和建模数据生成过程（DGP）的追求至关重要。深度生成模型（DGM）在捕捉复杂数据分布方面表现出色，但通常在泛化能力和可解释性方面表现不足。而因果性则提供了一种结构化的方法来理解驱动数据生成的机制，并突显了这些过程中固有的因果效应动力学。虽然因果性在可解释性和外推能力方面表现出色，但却面临着高维空间中的复杂性。意识到它们之间的协同潜力，我们深入探讨了因果性和DGM的交汇点。我们阐明了因果性原则在DGM中的整合，探讨了使用DGM进行因果识别的方法，并对因果性在大规模生成模型中的新兴研究前沿，尤其是大型语言模型（LLM）中的生成性问题提供了见解。我们介绍了方法论，突出了开放的挑战和机会。

    In the field of artificial intelligence (AI), the quest to understand and model data-generating processes (DGPs) is of paramount importance. Deep generative models (DGMs) have proven adept in capturing complex data distributions but often fall short in generalization and interpretability. On the other hand, causality offers a structured lens to comprehend the mechanisms driving data generation and highlights the causal-effect dynamics inherent in these processes. While causality excels in interpretability and the ability to extrapolate, it grapples with intricacies of high-dimensional spaces. Recognizing the synergistic potential, we delve into the confluence of causality and DGMs. We elucidate the integration of causal principles within DGMs, investigate causal identification using DGMs, and navigate an emerging research frontier of causality in large-scale generative models, particularly generative large language models (LLMs). We offer insights into methodologies, highlight open cha
    
[^13]: 在临床存在下的填补策略：对算法公平性的影响

    Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness. (arXiv:2208.06648v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2208.06648](http://arxiv.org/abs/2208.06648)

    本文研究了填补选择对不同群体的重建误差和下游预测的算法公平性属性的影响。

    

    机器学习可能会强化数据中的偏见，而我们在这个工作中提出，数据中缺失的内容也会产生偏见。在医疗领域，偏见已经在医疗历史上留下了深深的烙印，导致边缘化群体受到不平等的护理。缺失数据中的模式通常反映了这些群体的差异，但是特定群体缺失的算法公平性影响还不太清楚。尽管其潜在影响巨大，但填补往往被忽视为一个预处理步骤，而关注点放在了重建误差的减少和整体性能上，忽略了填补如何对不同群体产生影响。我们的工作研究了填补选择对不同群体的重建误差和下游预测的算法公平性属性的影响。

    Machine learning risks reinforcing biases present in data, and, as we argue in this work, in what is absent from data. In healthcare, biases have marked medical history, leading to unequal care affecting marginalised groups. Patterns in missing data often reflect these group discrepancies, but the algorithmic fairness implications of group-specific missingness are not well understood. Despite its potential impact, imputation is often an overlooked preprocessing step, with attention placed on the reduction of reconstruction error and overall performance, ignoring how imputation can affect groups differently. Our work studies how imputation choices affect reconstruction errors across groups and algorithmic fairness properties of downstream predictions.
    

