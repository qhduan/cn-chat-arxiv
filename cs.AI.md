# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping.](http://arxiv.org/abs/2305.10721) | 本文证明了线性映射在长期时间序列预测中的重要性，提出了RevIN和CI的方法来提高预测性能，同时发现线性映射可以有效地捕捉时间序列的周期特征。 |
| [^2] | [A Machine with Short-Term, Episodic, and Semantic Memory Systems.](http://arxiv.org/abs/2212.02098) | 本文研究了一个具有短期、情节和语义内存系统的机器代理模型，通过基于知识图谱的建模，在强化学习环境中实现了短期记忆的管理和存储，实验证明这种人类记忆系统结构的代理比没有该结构的代理表现更好。 |

# 详细

[^1]: 重新审视长期时间序列预测：线性映射的探究

    Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping. (arXiv:2305.10721v1 [cs.LG])

    [http://arxiv.org/abs/2305.10721](http://arxiv.org/abs/2305.10721)

    本文证明了线性映射在长期时间序列预测中的重要性，提出了RevIN和CI的方法来提高预测性能，同时发现线性映射可以有效地捕捉时间序列的周期特征。

    

    近年来，长期时间序列预测受到了越来越多的关注。虽然有各种专门设计来捕捉时间依赖性的方法，但是先前的研究表明，与其他复杂的架构相比，单个线性层可以实现竞争性的预测性能。本文彻底研究了最近方法的内在有效性，并得出了三个主要结论：1）线性映射对于先前的长期时间序列预测至关重要；2）RevIN（可逆规范化）和CI（通道独立）在提高总体预测性能方面发挥重要作用；3）当增加输入视野时，线性映射能够有效捕捉时间序列的周期特征，并具有对不同通道不同周期的鲁棒性。我们提供了理论和实验解释来支持我们的发现，并讨论了局限性和未来工作。我们框架的代码可在\url{https://git}中获得。

    Long-term time series forecasting has gained significant attention in recent years. While there are various specialized designs for capturing temporal dependency, previous studies have demonstrated that a single linear layer can achieve competitive forecasting performance compared to other complex architectures. In this paper, we thoroughly investigate the intrinsic effectiveness of recent approaches and make three key observations: 1) linear mapping is critical to prior long-term time series forecasting efforts; 2) RevIN (reversible normalization) and CI (Channel Independent) play a vital role in improving overall forecasting performance; and 3) linear mapping can effectively capture periodic features in time series and has robustness for different periods across channels when increasing the input horizon. We provide theoretical and experimental explanations to support our findings and also discuss the limitations and future works. Our framework's code is available at \url{https://git
    
[^2]: 一个具有短期、情节和语义内存系统的机器

    A Machine with Short-Term, Episodic, and Semantic Memory Systems. (arXiv:2212.02098v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2212.02098](http://arxiv.org/abs/2212.02098)

    本文研究了一个具有短期、情节和语义内存系统的机器代理模型，通过基于知识图谱的建模，在强化学习环境中实现了短期记忆的管理和存储，实验证明这种人类记忆系统结构的代理比没有该结构的代理表现更好。

    

    受认知科学理论中显性人类记忆系统的启发，我们建立了一个具有短期、情节和语义记忆系统的代理模型，每个记忆系统都用知识图谱建模。为了评估该系统并分析该代理的行为，我们设计并发布了我们自己的强化学习代理环境“房间”，在这个环境中，代理必须学习如何编码、存储和检索记忆，通过回答问题来最大化回报。我们证明了我们基于深度Q学习的代理成功学习了短期记忆是否应该被遗忘，还是应该存储在情节或语义记忆系统中。我们的实验表明，具有类人记忆系统的代理在环境中表现优于没有这种记忆结构的代理。

    Inspired by the cognitive science theory of the explicit human memory systems, we have modeled an agent with short-term, episodic, and semantic memory systems, each of which is modeled with a knowledge graph. To evaluate this system and analyze the behavior of this agent, we designed and released our own reinforcement learning agent environment, "the Room", where an agent has to learn how to encode, store, and retrieve memories to maximize its return by answering questions. We show that our deep Q-learning based agent successfully learns whether a short-term memory should be forgotten, or rather be stored in the episodic or semantic memory systems. Our experiments indicate that an agent with human-like memory systems can outperform an agent without this memory structure in the environment.
    

