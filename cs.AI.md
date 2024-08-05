# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Reinforcement Learning for Traveling Purchaser Problems](https://arxiv.org/abs/2404.02476) | 提出了一种基于深度强化学习的方法，该方法分别解决了旅行购买者问题中的路由构建和购买规划问题，并从全局角度评估和优化解决方案。 |
| [^2] | [DASA: Delay-Adaptive Multi-Agent Stochastic Approximation](https://arxiv.org/abs/2403.17247) | DASA算法是第一个收敛速度仅依赖于混合时间和平均延迟的算法，同时在马尔科夫采样下实现N倍的收敛加速。 |
| [^3] | [Automated System-level Testing of Unmanned Aerial Systems](https://arxiv.org/abs/2403.15857) | 本文提出了一种利用模型测试和人工智能技术自动生成、执行和评估无人机系统级测试的新颖方法。 |
| [^4] | [An Optimization Framework to Enforce Multi-View Consistency for Texturing 3D Meshes Using Pre-Trained Text-to-Image Models](https://arxiv.org/abs/2403.15559) | 该论文介绍了一个四阶段的优化框架，通过MV一致的扩散过程、半定编程问题解决、非刚性对齐和MRF问题解决等步骤来实现对3D网格进行纹理贴图的多视图一致性。 |
| [^5] | [Multi-criteria approach for selecting an explanation from the set of counterfactuals produced by an ensemble of explainers](https://arxiv.org/abs/2403.13940) | 本文提出了一种多阶段集成方法，通过多标准分析选择单个反事实，避免了用户测试多种不同解释方法和分析冲突解决方案的困难，提供了一个在多个质量度量上得分很高的妥协方案。 |
| [^6] | [A Tutorial on the Pretrain-Finetune Paradigm for Natural Language Processing](https://arxiv.org/abs/2403.02504) | 预训练-微调范式在自然语言处理中展现了显著的效率，尤其对社会科学研究中数据有限的情况下具有益处。 |
| [^7] | [Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration.](http://arxiv.org/abs/2401.13979) | 本研究提出了Leeroo编排器的架构，通过集成多个训练过的LLMs模型，实现了一个新的最先进模型。该编排器在性能上与Mixtral模型相当，并且成本只有其三分之二。当允许更高的成本时，Leeroo编排器的准确性超过了Mixtral模型，并且当集成GPT4时进一步提升。 |
| [^8] | [Adaptive Self-training Framework for Fine-grained Scene Graph Generation.](http://arxiv.org/abs/2401.09786) | 本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。 |
| [^9] | [MERA: A Comprehensive LLM Evaluation in Russian.](http://arxiv.org/abs/2401.04531) | 这项研究提出了MERA，一个多模态俄语基础模型评估指标。该指标包括21个评估任务，涵盖了11个技能领域中生成模型的评估。研究还提出了一种在零样本和少样本固定指令设置下评估FM和LM的方法。 |
| [^10] | [SynthoGestures: A Novel Framework for Synthetic Dynamic Hand Gesture Generation for Driving Scenarios.](http://arxiv.org/abs/2309.04421) | SynthoGestures是一种使用虚幻引擎合成逼真手势的新框架，可以用于驾驶场景下的动态人机界面。该框架通过生成多种变体和模拟不同摄像机类型，提高了手势识别的准确性并节省了数据集创建的时间和精力。 |
| [^11] | [Arithmetic with Language Models: from Memorization to Computation.](http://arxiv.org/abs/2308.01154) | 本研究探索了使用语言模型进行算术计算的能力，发现语言模型可以通过内部的值空间进行计算，并取得了成功的实验结果。 |
| [^12] | [Urban Spatiotemporal Data Synthesis via Neural Disaggregation.](http://arxiv.org/abs/2306.07292) | 本研究提出了一种基于神经网络的城市时空数据合成方法，旨在通过分解粗糙的低分辨率地理单元的聚合城市数据来合成细粒度，高分辨率的城市数据，以增加高度聚合的城市数据的可用性和实现价值。 |
| [^13] | [Answering Questions by Meta-Reasoning over Multiple Chains of Thought.](http://arxiv.org/abs/2304.13007) | 本论文提出了基于元推理的Multi-Chain Reasoning (MCR)方法，该方法检查多个推理链，混合它们之间的信息并选择最相关的事实，从而超越多链思维，解决多跳QA问题。 实验结果表明MCR胜过多个强基线，解释质量高。 |

# 详细

[^1]: 用于旅行购买者问题的深度强化学习

    Deep Reinforcement Learning for Traveling Purchaser Problems

    [https://arxiv.org/abs/2404.02476](https://arxiv.org/abs/2404.02476)

    提出了一种基于深度强化学习的方法，该方法分别解决了旅行购买者问题中的路由构建和购买规划问题，并从全局角度评估和优化解决方案。

    

    旅行购买者问题（TPP）是一种具有广泛应用的重要组合优化问题。本文提出了一种基于深度强化学习（DRL）的新方法，该方法分别解决了路由构建和购买规划问题，同时从全局角度评估和优化解决方案。我们的方法的关键组成部分包括用于捕捉市场-产品关系的TPP的二部图表示，以及从二部图中提取信息并将其用于顺序构建路由的策略网络。

    arXiv:2404.02476v1 Announce Type: cross  Abstract: The traveling purchaser problem (TPP) is an important combinatorial optimization problem with broad applications. Due to the coupling between routing and purchasing, existing works on TPPs commonly address route construction and purchase planning simultaneously, which, however, leads to exact methods with high computational cost and heuristics with sophisticated design but limited performance. In sharp contrast, we propose a novel approach based on deep reinforcement learning (DRL), which addresses route construction and purchase planning separately, while evaluating and optimizing the solution from a global perspective. The key components of our approach include a bipartite graph representation for TPPs to capture the market-product relations, and a policy network that extracts information from the bipartite graph and uses it to sequentially construct the route. One significant benefit of our framework is that we can efficiently const
    
[^2]: DASA: 延迟自适应多智能体随机逼近

    DASA: Delay-Adaptive Multi-Agent Stochastic Approximation

    [https://arxiv.org/abs/2403.17247](https://arxiv.org/abs/2403.17247)

    DASA算法是第一个收敛速度仅依赖于混合时间和平均延迟的算法，同时在马尔科夫采样下实现N倍的收敛加速。

    

    我们考虑一种设置，其中$N$个智能体旨在通过并行操作并与中央服务器通信来加速一个常见的随机逼近（SA）问题。我们假定上行传输到服务器的传输受到异步和潜在无界时变延迟的影响。为了减轻延迟和落后者的影响，同时又能获得分布式计算的好处，我们提出了一种名为DASA的延迟自适应多智能体随机逼近算法。我们对DASA进行了有限时间分析，假设智能体的随机观测过程是独立马尔科夫链。与现有结果相比，DASA是第一个其收敛速度仅取决于混合时间$tmix$和平均延迟$\tau_{avg}$，同时在马尔科夫采样下实现N倍的收敛加速的算法。我们的工作对于各种SA应用是相关的。

    arXiv:2403.17247v1 Announce Type: new  Abstract: We consider a setting in which $N$ agents aim to speedup a common Stochastic Approximation (SA) problem by acting in parallel and communicating with a central server. We assume that the up-link transmissions to the server are subject to asynchronous and potentially unbounded time-varying delays. To mitigate the effect of delays and stragglers while reaping the benefits of distributed computation, we propose \texttt{DASA}, a Delay-Adaptive algorithm for multi-agent Stochastic Approximation. We provide a finite-time analysis of \texttt{DASA} assuming that the agents' stochastic observation processes are independent Markov chains. Significantly advancing existing results, \texttt{DASA} is the first algorithm whose convergence rate depends only on the mixing time $\tmix$ and on the average delay $\tau_{avg}$ while jointly achieving an $N$-fold convergence speedup under Markovian sampling. Our work is relevant for various SA applications, inc
    
[^3]: 无人机系统级测试的自动化系统

    Automated System-level Testing of Unmanned Aerial Systems

    [https://arxiv.org/abs/2403.15857](https://arxiv.org/abs/2403.15857)

    本文提出了一种利用模型测试和人工智能技术自动生成、执行和评估无人机系统级测试的新颖方法。

    

    无人机系统依赖于各种安全关键和任务关键的航空电子系统。国际安全标准的主要要求之一是对航空电子软件系统进行严格的系统级测试。当前工业实践是手动创建测试方案，使用模拟器手动/自动执行这些方案，并手动评估结果。本文提出了一种新颖的方法来自动化无人机系统级测试。所提出的方法(AITester)利用基于模型的测试和人工智能(AI)技术，自动生成、执行和评估各种测试方案。

    arXiv:2403.15857v1 Announce Type: cross  Abstract: Unmanned aerial systems (UAS) rely on various avionics systems that are safety-critical and mission-critical. A major requirement of international safety standards is to perform rigorous system-level testing of avionics software systems. The current industrial practice is to manually create test scenarios, manually/automatically execute these scenarios using simulators, and manually evaluate outcomes. The test scenarios typically consist of setting certain flight or environment conditions and testing the system under test in these settings. The state-of-the-art approaches for this purpose also require manual test scenario development and evaluation. In this paper, we propose a novel approach to automate the system-level testing of the UAS. The proposed approach (AITester) utilizes model-based testing and artificial intelligence (AI) techniques to automatically generate, execute, and evaluate various test scenarios. The test scenarios a
    
[^4]: 一个优化框架，利用预训练的文本到图像模型强制实现对3D网格进行纹理贴图的多视图一致性

    An Optimization Framework to Enforce Multi-View Consistency for Texturing 3D Meshes Using Pre-Trained Text-to-Image Models

    [https://arxiv.org/abs/2403.15559](https://arxiv.org/abs/2403.15559)

    该论文介绍了一个四阶段的优化框架，通过MV一致的扩散过程、半定编程问题解决、非刚性对齐和MRF问题解决等步骤来实现对3D网格进行纹理贴图的多视图一致性。

    

    在使用预训练的文本到图像模型对3D网格进行纹理贴图时，确保多视图一致性是一个基本问题。本文介绍了一个优化框架，通过四个阶段实现多视图一致性。具体而言，第一阶段使用MV一致的扩散过程从预定义的视点集生成2D纹理的过完备集。第二阶段通过解决半定编程问题选择相互一致且覆盖基础3D模型的视图子集。第三阶段执行非刚性对齐，使选定的视图在重叠区域对齐。第四阶段解决MRF问题以关联...

    arXiv:2403.15559v1 Announce Type: cross  Abstract: A fundamental problem in the texturing of 3D meshes using pre-trained text-to-image models is to ensure multi-view consistency. State-of-the-art approaches typically use diffusion models to aggregate multi-view inputs, where common issues are the blurriness caused by the averaging operation in the aggregation step or inconsistencies in local features. This paper introduces an optimization framework that proceeds in four stages to achieve multi-view consistency. Specifically, the first stage generates an over-complete set of 2D textures from a predefined set of viewpoints using an MV-consistent diffusion process. The second stage selects a subset of views that are mutually consistent while covering the underlying 3D model. We show how to achieve this goal by solving semi-definite programs. The third stage performs non-rigid alignment to align the selected views across overlapping regions. The fourth stage solves an MRF problem to associ
    
[^5]: 从解释器集合中选择反事实解释的多标准方法

    Multi-criteria approach for selecting an explanation from the set of counterfactuals produced by an ensemble of explainers

    [https://arxiv.org/abs/2403.13940](https://arxiv.org/abs/2403.13940)

    本文提出了一种多阶段集成方法，通过多标准分析选择单个反事实，避免了用户测试多种不同解释方法和分析冲突解决方案的困难，提供了一个在多个质量度量上得分很高的妥协方案。

    

    反事实被广泛用于解释机器学习模型的预测，提供获取更理想预测的替代场景。它们可以由多种方法生成，这些方法优化不同、有时是冲突的质量度量，并产生完全不同的解决方案。然而，选择最合适的解释方法和生成的反事实之一并不是一件容易的事情。本文提出使用多阶段集成方法，基于多标准分析来选择单个反事实，而不是强迫用户测试许多不同的解释方法并分析冲突的解决方案。它提供了一个妥协方案，在几个流行的质量度量上得分较高。这种方法利用支配关系和理想点决策辅助方法，从帕累托前沿中选择一个反事实。进行的实验证明了这种方法的有效性。

    arXiv:2403.13940v1 Announce Type: cross  Abstract: Counterfactuals are widely used to explain ML model predictions by providing alternative scenarios for obtaining the more desired predictions. They can be generated by a variety of methods that optimize different, sometimes conflicting, quality measures and produce quite different solutions. However, choosing the most appropriate explanation method and one of the generated counterfactuals is not an easy task. Instead of forcing the user to test many different explanation methods and analysing conflicting solutions, in this paper, we propose to use a multi-stage ensemble approach that will select single counterfactual based on the multiple-criteria analysis. It offers a compromise solution that scores well on several popular quality measures. This approach exploits the dominance relation and the ideal point decision aid method, which selects one counterfactual from the Pareto front. The conducted experiments demonstrated that the propos
    
[^6]: 自然语言处理中的预训练-微调范式教程

    A Tutorial on the Pretrain-Finetune Paradigm for Natural Language Processing

    [https://arxiv.org/abs/2403.02504](https://arxiv.org/abs/2403.02504)

    预训练-微调范式在自然语言处理中展现了显著的效率，尤其对社会科学研究中数据有限的情况下具有益处。

    

    预训练-微调范式代表了自然语言处理中的一种变革性方法。该范式通过使用大型预训练语言模型区别于众，展示了在微调任务中即使训练数据有限也具有显著的效率。这种效率对社会科学研究特别有益，因为注释样本的数量通常非常有限。我们的教程全面介绍了预训练-微调范式。我们首先深入探讨了预训练和微调的基本概念，然后进行了实际应用的案例练习。我们展示了该范式在各种任务中的应用，包括多类别分类和回归。强调其高效性和用户友好性，该教程旨在鼓励更广泛地采纳这种范式。为此，我们提供了所有代码和数据集的开放访问。

    arXiv:2403.02504v1 Announce Type: cross  Abstract: The pretrain-finetune paradigm represents a transformative approach in natural language processing (NLP). This paradigm distinguishes itself through the use of large pretrained language models, demonstrating remarkable efficiency in finetuning tasks, even with limited training data. This efficiency is especially beneficial for research in social sciences, where the number of annotated samples is often quite limited. Our tutorial offers a comprehensive introduction to the pretrain-finetune paradigm. We first delve into the fundamental concepts of pretraining and finetuning, followed by practical exercises using real-world applications. We demonstrate the application of the paradigm across various tasks, including multi-class classification and regression. Emphasizing its efficacy and user-friendliness, the tutorial aims to encourage broader adoption of this paradigm. To this end, we have provided open access to all our code and datasets
    
[^7]: Leeroo Orchestrator: 通过模型集成提高LLMs的性能

    Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration. (arXiv:2401.13979v1 [cs.CL])

    [http://arxiv.org/abs/2401.13979](http://arxiv.org/abs/2401.13979)

    本研究提出了Leeroo编排器的架构，通过集成多个训练过的LLMs模型，实现了一个新的最先进模型。该编排器在性能上与Mixtral模型相当，并且成本只有其三分之二。当允许更高的成本时，Leeroo编排器的准确性超过了Mixtral模型，并且当集成GPT4时进一步提升。

    

    本文提出了一种架构，利用多个训练过的LLMs的集体知识，创建一个新的最先进模型。该框架的核心是一个基于LLM的编排器，能够选择最佳的底层LLM专家进行任务执行。受到强化学习中的自我对弈的启发，我们创建了一个查询生成、编排和评估的循环，为编排器生成训练数据。我们的评估主要针对MMLU基准，在Hugging Face上使用了具有7B、13B和34B参数的模型。结果显示我们的Leeroo编排器实现了与Mixtral模型相当的性能，但只产生了其成本的三分之二。此外，增加允许的成本超过了Mixtral的准确性，达到了75.9%的准确性。当将GPT4集成到底层模型池中时，进一步提升也得到了观察。

    In this paper, we propose an architecture to harness the collective knowledge of multiple trained LLMs to create a new state-of-the-art. At the core of this framework is a LLM-based orchestrator that is adept at picking the right underlying LLM experts for optimal task execution. Inspired by self-play in reinforcement learning, we created a loop of query generation, orchestration, and evaluation to generate training data for the orchestrator. Our evaluation focused on the MMLU benchmark, employing models with 7B, 13B, and 34B parameters available on Hugging Face. The results demonstrate new state-of-the-art open-source models: Our Leeroo orchestrator achieves performance on par with the Mixtral model while incurring only two-thirds of its cost. Moreover, increasing the allowed cost surpasses Mixtral's accuracy by over 5% at the same cost level, reaching an accuracy of 75.9%. Further enhancements were observed when integrating GPT4 into the underlying model pool. The Leeroo orchestrator
    
[^8]: 自适应自训练框架用于细粒度场景图生成

    Adaptive Self-training Framework for Fine-grained Scene Graph Generation. (arXiv:2401.09786v1 [cs.CV])

    [http://arxiv.org/abs/2401.09786](http://arxiv.org/abs/2401.09786)

    本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。

    

    场景图生成（SGG）模型在基准数据集中存在长尾谓词分布和缺失注释问题。本研究旨在通过利用未标注的三元组缓解SGG的长尾问题。为此，我们引入了一种称为自训练SGG（ST-SGG）的框架，该框架基于未标注的三元组为其分配伪标签以训练SGG模型。虽然在图像识别方面的自训练取得了显著进展，但设计适用于SGG任务的自训练框架更具挑战，因为其固有特性，如语义歧义和长尾分布的谓词类别。因此，我们提出了一种新颖的SGG伪标签技术，称为具有动量的类别自适应阈值化（CATM），它是一种独立于模型的框架，可应用于任何已有的SGG模型。此外，我们设计了一个图结构学习器（GSL），从中获益。

    Scene graph generation (SGG) models have suffered from inherent problems regarding the benchmark datasets such as the long-tailed predicate distribution and missing annotation problems. In this work, we aim to alleviate the long-tailed problem of SGG by utilizing unannotated triplets. To this end, we introduce a Self-Training framework for SGG (ST-SGG) that assigns pseudo-labels for unannotated triplets based on which the SGG models are trained. While there has been significant progress in self-training for image recognition, designing a self-training framework for the SGG task is more challenging due to its inherent nature such as the semantic ambiguity and the long-tailed distribution of predicate classes. Hence, we propose a novel pseudo-labeling technique for SGG, called Class-specific Adaptive Thresholding with Momentum (CATM), which is a model-agnostic framework that can be applied to any existing SGG models. Furthermore, we devise a graph structure learner (GSL) that is benefici
    
[^9]: MERA: 俄语LLM综合评估的研究

    MERA: A Comprehensive LLM Evaluation in Russian. (arXiv:2401.04531v1 [cs.CL])

    [http://arxiv.org/abs/2401.04531](http://arxiv.org/abs/2401.04531)

    这项研究提出了MERA，一个多模态俄语基础模型评估指标。该指标包括21个评估任务，涵盖了11个技能领域中生成模型的评估。研究还提出了一种在零样本和少样本固定指令设置下评估FM和LM的方法。

    

    在过去几年中，人工智能研究中最显著的进展之一是基础模型（FM）的发展，其中语言模型（LM）的崛起引人注目。随着模型的规模增大，LM在可衡量的方面展示了提升，并且发展出了新的定性特征。然而，尽管研究人员的关注和LM应用的快速增长，LM的能力、限制和相关风险仍需更好地理解。为了解决这些问题，我们介绍了一种开放的俄语多模态架构评估（MERA）指导基准，用于评估以俄语为导向的基础模型。该基准涵盖了11个技能领域中生成模型的21个评估任务，并被设计为黑盒测试，以确保排除数据泄漏。论文介绍了一种在零样本和少样本固定指令设置下评估FM和LM的方法，并可扩展到其他模态。

    Over the past few years, one of the most notable advancements in AI research has been in foundation models (FMs), headlined by the rise of language models (LMs). As the models' size increases, LMs demonstrate enhancements in measurable aspects and the development of new qualitative features. However, despite researchers' attention and the rapid growth in LM application, the capabilities, limitations, and associated risks still need to be better understood. To address these issues, we introduce an open Multimodal Evaluation of Russian-language Architectures (MERA), a new instruction benchmark for evaluating foundation models oriented towards the Russian language. The benchmark encompasses 21 evaluation tasks for generative models in 11 skill domains and is designed as a black-box test to ensure the exclusion of data leakage. The paper introduces a methodology to evaluate FMs and LMs in zeroand few-shot fixed instruction settings that can be extended to other modalities. We propose an 
    
[^10]: SynthoGestures：一种用于驾驶场景的合成动态手势生成的新框架

    SynthoGestures: A Novel Framework for Synthetic Dynamic Hand Gesture Generation for Driving Scenarios. (arXiv:2309.04421v1 [cs.CV])

    [http://arxiv.org/abs/2309.04421](http://arxiv.org/abs/2309.04421)

    SynthoGestures是一种使用虚幻引擎合成逼真手势的新框架，可以用于驾驶场景下的动态人机界面。该框架通过生成多种变体和模拟不同摄像机类型，提高了手势识别的准确性并节省了数据集创建的时间和精力。

    

    在汽车领域中，为动态人机界面创建多样化和全面的手势数据集可能具有挑战性且耗时。为了克服这一挑战，我们提出使用虚拟3D模型生成合成手势数据集。我们的框架利用虚幻引擎合成逼真的手势，提供定制选项并降低过拟合风险。生成多种变体，包括手势速度、性能和手形，以提高泛化能力。此外，我们模拟不同的摄像机位置和类型，如RGB、红外和深度摄像机，而无需额外的时间和费用获取这些摄像机。实验结果表明，我们的提议框架SynthoGestures提高了手势识别准确率，可以替代或增强真手数据集。通过节省数据集创建的时间和精力，我们的工具促进了研究的进展。

    Creating a diverse and comprehensive dataset of hand gestures for dynamic human-machine interfaces in the automotive domain can be challenging and time-consuming. To overcome this challenge, we propose using synthetic gesture datasets generated by virtual 3D models. Our framework utilizes Unreal Engine to synthesize realistic hand gestures, offering customization options and reducing the risk of overfitting. Multiple variants, including gesture speed, performance, and hand shape, are generated to improve generalizability. In addition, we simulate different camera locations and types, such as RGB, infrared, and depth cameras, without incurring additional time and cost to obtain these cameras. Experimental results demonstrate that our proposed framework, SynthoGestures\footnote{\url{https://github.com/amrgomaaelhady/SynthoGestures}}, improves gesture recognition accuracy and can replace or augment real-hand datasets. By saving time and effort in the creation of the data set, our tool acc
    
[^11]: 使用语言模型进行算术运算：从记忆到计算

    Arithmetic with Language Models: from Memorization to Computation. (arXiv:2308.01154v1 [cs.AI])

    [http://arxiv.org/abs/2308.01154](http://arxiv.org/abs/2308.01154)

    本研究探索了使用语言模型进行算术计算的能力，发现语言模型可以通过内部的值空间进行计算，并取得了成功的实验结果。

    

    更好地理解最近的大型语言模型的出现性计算和问题解决能力对于进一步改进它们并拓宽其适用性至关重要。本研究探讨了一个训练用于预测下一个标记的语言模型如何在训练数据之外执行算术计算。二进制加法和乘法是一个很好的测试基础，因为它们需要一个非常小的词汇表，并且在输入/输出上展示了相关的不连续性，使得对新数据进行平滑的输入插值无效。我们成功地训练了一个轻量级的语言模型来学习这些任务，并进行了一系列实验证明其外推能力和内部信息处理。我们的研究结果支持这样一个假设，即语言模型作为一个编码-回归-解码机器，一旦将输入标记表示映射到合适的内部值空间，计算就在值空间中进行。

    A better understanding of the emergent computation and problem-solving capabilities of recent large language models is of paramount importance to further improve them and broaden their applicability. This work investigates how a language model, trained to predict the next token, can perform arithmetic computations generalizing beyond training data. Binary addition and multiplication constitute a good testbed for this purpose, since they require a very small vocabulary and exhibit relevant input/output discontinuities making smooth input interpolation ineffective for novel data. We successfully trained a light language model to learn these tasks and ran a number of experiments to investigate the extrapolation capabilities and internal information processing. Our findings support the hypotheses that the language model works as an Encoding-Regression-Decoding machine where the computation takes place in the value space once the input token representation is mapped to an appropriate intern
    
[^12]: 基于神经网络的城市时空数据合成方法

    Urban Spatiotemporal Data Synthesis via Neural Disaggregation. (arXiv:2306.07292v1 [cs.LG])

    [http://arxiv.org/abs/2306.07292](http://arxiv.org/abs/2306.07292)

    本研究提出了一种基于神经网络的城市时空数据合成方法，旨在通过分解粗糙的低分辨率地理单元的聚合城市数据来合成细粒度，高分辨率的城市数据，以增加高度聚合的城市数据的可用性和实现价值。

    

    开放数据的细节级别常常与其所能提供的实际效益发生冲突。较不细化的数据可以保护个人隐私，但在一定程度上牺牲了开放数据促进透明度和协助研究的承诺。类似于城市环境中，高层次地理单元的聚合城市数据可能会掩盖城市动态的底层特征，低级别地理单元的变化可能更为明显。本研究旨在通过分解粗糙的低分辨率地理单元的聚合城市数据，合成细粒度，高分辨率的城市数据，以增加高度聚合的城市数据的可用性和实现价值。为了解决一些传统分解方法的简单性问题-1) 我们尝试了许多神经网络模型，这些模型能够建模特征之间复杂的非线性关系。神经方法也可以同时利用空间和时间信息。我们展示了这些神经网络方法的优点。

    The level of granularity of open data often conflicts the benefits it can provide. Less granular data can protect individual privacy, but to certain degrees, sabotage the promise of open data to promote transparency and assist research. Similar in the urban setting, aggregated urban data at high-level geographic units can mask out the underline particularities of city dynamics that may vary at lower areal levels. In this work, we aim to synthesize fine-grained, high resolution urban data, by breaking down aggregated urban data at coarse, low resolution geographic units. The goal is to increase the usability and realize the values as much as possible of highly aggregated urban data. To address the issue of simplicity of some traditional disaggregation methods -- 1) we experimented with numerous neural-based models that are capable of modeling intricate non-linear relationships among features. Neural methods can also leverage both spatial and temporal information concurrently. We showed 
    
[^13]: 超越多链思维：基于元推理的问题解答方法

    Answering Questions by Meta-Reasoning over Multiple Chains of Thought. (arXiv:2304.13007v1 [cs.CL])

    [http://arxiv.org/abs/2304.13007](http://arxiv.org/abs/2304.13007)

    本论文提出了基于元推理的Multi-Chain Reasoning (MCR)方法，该方法检查多个推理链，混合它们之间的信息并选择最相关的事实，从而超越多链思维，解决多跳QA问题。 实验结果表明MCR胜过多个强基线，解释质量高。

    

    现代多跳问题解答（QA）系统通常将问题分解为一系列思考步骤（CoT），然后才得出最终答案。通常来说，多个链条被抽样并通过最终答案的投票机制进行聚合，但中间步骤本身被丢弃。虽然这种方法提高了性能，但它们并不考虑链之间的中间步骤之间的关系，并且不提供预测答案的统一解释。我们引入了基于元推理的 Multi-Chain Reasoning (MCR) 方法，该方法利用大型语言模型来超越多个思考链，而不是聚合回答。MCR检查不同的推理链，混合它们之间的信息并选择在生成解释和预测答案时最相关的事实。MCR在7个多跳QA数据集上胜过强基线。此外，我们的分析表明MCR的解释具有高质量。

    Modern systems for multi-hop question answering (QA) typically break questions into a sequence of reasoning steps, termed chain-of-thought (CoT), before arriving at a final answer. Often, multiple chains are sampled and aggregated through a voting mechanism over the final answers, but the intermediate steps themselves are discarded. While such approaches improve performance, they do not consider the relations between intermediate steps across chains and do not provide a unified explanation for the predicted answer. We introduce Multi-Chain Reasoning (MCR), an approach which prompts large language models to meta-reason over multiple chains of thought, rather than aggregating their answers. MCR examines different reasoning chains, mixes information between them and selects the most relevant facts in generating an explanation and predicting the answer. MCR outperforms strong baselines on 7 multi-hop QA datasets. Moreover, our analysis reveals that MCR explanations exhibit high quality, en
    

