# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Systematic AI Approach for AGI: Addressing Alignment, Energy, and AGI Grand Challenges.](http://arxiv.org/abs/2310.15274) | 本论文讨论了面临能源、对齐和从狭义人工智能到AGI的三大挑战的系统化人工智能方法。现有的人工智能方法在能源消耗、系统设计和对齐问题上存在不足，而系统设计在解决对齐、能源和AGI大挑战中是至关重要的。 |
| [^2] | [RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark.](http://arxiv.org/abs/2306.17100) | RL4CO是一个用于组合优化的广泛强化学习基准测试，着重于可扩展性和泛化能力的评估，并展示了一些最新方法在样本效率和适应不同数据分布方面的表现相对较差，强调了对神经CO求解器性能的平衡评估的重要性。 |
| [^3] | [On the Design Fundamentals of Diffusion Models: A Survey.](http://arxiv.org/abs/2306.04542) | 本文综述了扩散模型的设计基础，即其三个关键组件：正向过程、逆向过程和采样过程，为未来的研究提供了有益的细粒度透视。 |
| [^4] | [Rethinking AI Explainability and Plausibility.](http://arxiv.org/abs/2303.17707) | 本文研究了XAI评估中最普遍的人为概念——解释合理性。虽然一直被制定为AI可解释性任务的重要评估目标，但是评估XAI的合理性有时是有害的，且无法达到模型可理解性、透明度和可信度的目的。 |

# 详细

[^1]: 系统化的人工智能方法用于AGI：解决对齐、能源和AGI大挑战

    Systematic AI Approach for AGI: Addressing Alignment, Energy, and AGI Grand Challenges. (arXiv:2310.15274v1 [cs.AI])

    [http://arxiv.org/abs/2310.15274](http://arxiv.org/abs/2310.15274)

    本论文讨论了面临能源、对齐和从狭义人工智能到AGI的三大挑战的系统化人工智能方法。现有的人工智能方法在能源消耗、系统设计和对齐问题上存在不足，而系统设计在解决对齐、能源和AGI大挑战中是至关重要的。

    

    人工智能面临着三大挑战：能源壁垒、对齐问题和从狭义人工智能到AGI的飞跃。当代人工智能解决方案在模型训练和日常运行过程中消耗着不可持续的能源。更糟糕的是，自2020年以来，每个新的人工智能模型所需的计算量每两个月就翻倍，直接导致能源消耗的增加。从人工智能到AGI的飞跃需要多个功能子系统以平衡的方式运作，这需要一个系统架构。然而，当前的人工智能方法缺乏系统设计；即使系统特征在人脑中扮演着重要角色，从它处理信息的方式到它做出决策的方式。同样，当前的对齐和人工智能伦理方法在很大程度上忽视了系统设计，然而研究表明，大脑的系统架构在健康的道德决策中起着关键作用。在本文中，我们认为系统设计在解决对齐、能源和AGI大挑战中至关重要。

    AI faces a trifecta of grand challenges the Energy Wall, the Alignment Problem and the Leap from Narrow AI to AGI. Contemporary AI solutions consume unsustainable amounts of energy during model training and daily operations.Making things worse, the amount of computation required to train each new AI model has been doubling every 2 months since 2020, directly translating to increases in energy consumption.The leap from AI to AGI requires multiple functional subsystems operating in a balanced manner, which requires a system architecture. However, the current approach to artificial intelligence lacks system design; even though system characteristics play a key role in the human brain from the way it processes information to how it makes decisions. Similarly, current alignment and AI ethics approaches largely ignore system design, yet studies show that the brains system architecture plays a critical role in healthy moral decisions.In this paper, we argue that system design is critically im
    
[^2]: RL4CO: 用于组合优化的广泛强化学习基准测试

    RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark. (arXiv:2306.17100v1 [cs.LG])

    [http://arxiv.org/abs/2306.17100](http://arxiv.org/abs/2306.17100)

    RL4CO是一个用于组合优化的广泛强化学习基准测试，着重于可扩展性和泛化能力的评估，并展示了一些最新方法在样本效率和适应不同数据分布方面的表现相对较差，强调了对神经CO求解器性能的平衡评估的重要性。

    

    我们引入了RL4CO，这是一个广泛的强化学习（RL）用于组合优化（CO）的基准测试。RL4CO采用最先进的软件库和最佳实践，如模块化和配置管理，以便研究人员可以轻松修改神经网络架构、环境和算法。与现有的专注于特定任务（如旅行推销员问题）进行性能评估的方法不同，我们强调可扩展性和泛化能力对于各种优化任务的重要性。我们还系统地评估了各种模型在样本效率、零-shot泛化和适应不同数据分布方面的表现。我们的实验结果表明，一些最新的最先进方法在使用这些新指标进行评估时落后于之前的方法，这表明有必要更加平衡地评估神经CO求解器的性能。我们希望RL4CO能够为研究人员提供一个综合性的基准测试工具，以进一步推动强化学习在组合优化领域的研究。

    We introduce RL4CO, an extensive reinforcement learning (RL) for combinatorial optimization (CO) benchmark. RL4CO employs state-of-the-art software libraries as well as best practices in implementation, such as modularity and configuration management, to be efficient and easily modifiable by researchers for adaptations of neural network architecture, environments, and algorithms. Contrary to the existing focus on specific tasks like the traveling salesman problem (TSP) for performance assessment, we underline the importance of scalability and generalization capabilities for diverse optimization tasks. We also systematically benchmark sample efficiency, zero-shot generalization, and adaptability to changes in data distributions of various models. Our experiments show that some recent state-of-the-art methods fall behind their predecessors when evaluated using these new metrics, suggesting the necessity for a more balanced view of the performance of neural CO solvers. We hope RL4CO will 
    
[^3]: 关于扩散模型的设计基础：综述

    On the Design Fundamentals of Diffusion Models: A Survey. (arXiv:2306.04542v1 [cs.LG])

    [http://arxiv.org/abs/2306.04542](http://arxiv.org/abs/2306.04542)

    本文综述了扩散模型的设计基础，即其三个关键组件：正向过程、逆向过程和采样过程，为未来的研究提供了有益的细粒度透视。

    

    扩散模型是一种生成模型，通过逐渐添加和删除噪声来学习训练数据的潜在分布以生成数据。扩散模型的组成部分已经受到了广泛的关注，许多设计选择被提出。现有的评论主要关注高层次的解决方案，对组件的设计基础覆盖较少。本研究旨在通过提供一个全面而连贯的综述，针对扩散模型的组件设计选择进行分析。具体来说，我们将这个综述按照三个关键组件进行组织，即正向过程、逆向过程和采样过程。这使得我们可以提供扩散模型的细粒度透视，有助于未来研究分析个体组件、设计选择的适用性以及扩散模型的实现。

    Diffusion models are generative models, which gradually add and remove noise to learn the underlying distribution of training data for data generation. The components of diffusion models have gained significant attention with many design choices proposed. Existing reviews have primarily focused on higher-level solutions, thereby covering less on the design fundamentals of components. This study seeks to address this gap by providing a comprehensive and coherent review on component-wise design choices in diffusion models. Specifically, we organize this review according to their three key components, namely the forward process, the reverse process, and the sampling procedure. This allows us to provide a fine-grained perspective of diffusion models, benefiting future studies in the analysis of individual components, the applicability of design choices, and the implementation of diffusion models.
    
[^4]: 重新思考人工智能可解释性与合理性

    Rethinking AI Explainability and Plausibility. (arXiv:2303.17707v1 [cs.AI])

    [http://arxiv.org/abs/2303.17707](http://arxiv.org/abs/2303.17707)

    本文研究了XAI评估中最普遍的人为概念——解释合理性。虽然一直被制定为AI可解释性任务的重要评估目标，但是评估XAI的合理性有时是有害的，且无法达到模型可理解性、透明度和可信度的目的。

    

    为了使可解释人工智能（XAI）算法符合人类交流规范，支持人类推理过程，并满足人类对于AI解释的需求，设定适当的评估目标至关重要。在本文中，我们研究了解释合理性，这是XAI评估中最普遍的人为概念。合理性衡量机器解释与人类解释相比的合理程度。合理性一直被传统地制定为AI可解释性任务的重要评估目标。我们反对这个想法，并展示了如何优化和评估XAI的合理性有时是有害的，且无法达到模型可理解性、透明度和可信度的目的。具体来说，评估XAI算法的合理性会规范机器解释，以表达与人类解释完全相同的内容，这偏离了人类解释的基本动机：表达自己的理解。

    Setting proper evaluation objectives for explainable artificial intelligence (XAI) is vital for making XAI algorithms follow human communication norms, support human reasoning processes, and fulfill human needs for AI explanations. In this article, we examine explanation plausibility, which is the most pervasive human-grounded concept in XAI evaluation. Plausibility measures how reasonable the machine explanation is compared to the human explanation. Plausibility has been conventionally formulated as an important evaluation objective for AI explainability tasks. We argue against this idea, and show how optimizing and evaluating XAI for plausibility is sometimes harmful, and always ineffective to achieve model understandability, transparency, and trustworthiness. Specifically, evaluating XAI algorithms for plausibility regularizes the machine explanation to express exactly the same content as human explanation, which deviates from the fundamental motivation for humans to explain: expres
    

