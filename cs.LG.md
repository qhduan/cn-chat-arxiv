# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Randomized Confidence Bounds for Stochastic Partial Monitoring](https://arxiv.org/abs/2402.05002) | 本文研究了随机部分监控中基于随机化的偏置界策略，该策略扩展了现有随机策略无法应用的情境和非情境设置，实验证明该策略优于最先进的方法。 |
| [^2] | [HiQA: A Hierarchical Contextual Augmentation RAG for Massive Documents QA](https://arxiv.org/abs/2402.01767) | HiQA是一个先进的多文档问答框架，使用分层的上下文增强和多路径检索机制，解决了大规模文档问答中的检索准确性问题，并在多文档环境中展示了最先进的性能。 |
| [^3] | [Federated Topic Model and Model Pruning Based on Variational Autoencoder.](http://arxiv.org/abs/2311.00314) | 本论文提出了一种基于变分自编码器的联邦主题模型和模型剪枝方法，用于解决跨多个方参与交叉分析时的数据隐私问题，并通过神经网络模型剪枝加速模型。两种不同的方法被提出来确定模型剪枝率。 |
| [^4] | [The Impact of Equal Opportunity on Statistical Discrimination.](http://arxiv.org/abs/2310.04585) | 本文通过修改统计性歧视模型，考虑了由机器学习生成的可合同化信念，给监管者提供了一种超过肯定行动的工具，通过要求公司选取一个平衡不同群体真正阳性率的决策策略，实现机会平等来消除统计性歧视。 |

# 详细

[^1]: 随机偏置界对随机部分监控的应用

    Randomized Confidence Bounds for Stochastic Partial Monitoring

    [https://arxiv.org/abs/2402.05002](https://arxiv.org/abs/2402.05002)

    本文研究了随机部分监控中基于随机化的偏置界策略，该策略扩展了现有随机策略无法应用的情境和非情境设置，实验证明该策略优于最先进的方法。

    

    部分监控 (PM) 框架提供了通过不完整的反馈进行顺序学习问题的理论表述。在每个回合中，学习代理选择一个动作，而环境同时选择一个结果。然后代理观察到一个仅部分提供信息关于（未观察到的）结果的反馈信号。代理利用接收到的反馈信号选择能够最小化（未观察到的）累计损失的动作。在情境 PM 中，结果依赖于代理在每轮选择动作之前可观察到的某些附加信息。在本文中，我们考虑了具有随机结果的情境和非情境的 PM 设置。我们引入了一种基于确定性置信界的随机化策略的新类方法，将遗憾保证扩展到现有的随机策略不适用的设置中。我们的实验表明，所提出的 RandCBP 和 RandCBPside* 策略改进了最先进的方法。

    The partial monitoring (PM) framework provides a theoretical formulation of sequential learning problems with incomplete feedback. On each round, a learning agent plays an action while the environment simultaneously chooses an outcome. The agent then observes a feedback signal that is only partially informative about the (unobserved) outcome. The agent leverages the received feedback signals to select actions that minimize the (unobserved) cumulative loss. In contextual PM, the outcomes depend on some side information that is observable by the agent before selecting the action on each round. In this paper, we consider the contextual and non-contextual PM settings with stochastic outcomes. We introduce a new class of strategies based on the randomization of deterministic confidence bounds, that extend regret guarantees to settings where existing stochastic strategies are not applicable. Our experiments show that the proposed RandCBP and RandCBPside* strategies improve state-of-the-art b
    
[^2]: HiQA：一种用于大规模文档问答的分层上下文增强的RAG模型

    HiQA: A Hierarchical Contextual Augmentation RAG for Massive Documents QA

    [https://arxiv.org/abs/2402.01767](https://arxiv.org/abs/2402.01767)

    HiQA是一个先进的多文档问答框架，使用分层的上下文增强和多路径检索机制，解决了大规模文档问答中的检索准确性问题，并在多文档环境中展示了最先进的性能。

    

    随着利用外部工具的语言模型代理迅速发展，使用补充文档和检索增强生成（RAG）方法的问答（QA）方法学取得了重要进展。这种进步提高了语言模型的回答质量，并减轻了幻觉的出现。然而，当面临大量无法区分的文档时，这些方法在检索准确性方面表现有限，给实际应用带来了显著挑战。针对这些新兴的挑战，我们提出了HiQA，这是一个先进的多文档问答（MDQA）框架，将级联的元数据整合到内容中，同时具备多路径检索机制。我们还发布了一个名为MasQA的基准来评估和研究MDQA。最后，HiQA在多文档环境中展示了最先进的性能。

    As language model agents leveraging external tools rapidly evolve, significant progress has been made in question-answering(QA) methodologies utilizing supplementary documents and the Retrieval-Augmented Generation (RAG) approach. This advancement has improved the response quality of language models and alleviates the appearance of hallucination. However, these methods exhibit limited retrieval accuracy when faced with massive indistinguishable documents, presenting notable challenges in their practical application. In response to these emerging challenges, we present HiQA, an advanced framework for multi-document question-answering (MDQA) that integrates cascading metadata into content as well as a multi-route retrieval mechanism. We also release a benchmark called MasQA to evaluate and research in MDQA. Finally, HiQA demonstrates the state-of-the-art performance in multi-document environments.
    
[^3]: 基于变分自编码器的联邦主题模型和模型剪枝

    Federated Topic Model and Model Pruning Based on Variational Autoencoder. (arXiv:2311.00314v1 [cs.LG])

    [http://arxiv.org/abs/2311.00314](http://arxiv.org/abs/2311.00314)

    本论文提出了一种基于变分自编码器的联邦主题模型和模型剪枝方法，用于解决跨多个方参与交叉分析时的数据隐私问题，并通过神经网络模型剪枝加速模型。两种不同的方法被提出来确定模型剪枝率。

    

    主题建模已经成为在大规模文档集合中发现模式和主题的有价值工具。然而，当跨多个方参与交叉分析时，数据隐私成为一个关键问题。联邦主题建模已经被开发出来解决这个问题，允许多个参与方在保护隐私的同时共同训练模型。然而，在联邦场景中存在通信和性能挑战。为了解决上述问题，本文提出了一种建立联邦主题模型并确保每个节点隐私的方法，并使用神经网络模型剪枝加速模型，其中客户端定期将模型神经元累积梯度和模型权重发送给服务器，服务器对模型进行剪枝。为了满足不同的要求，提出了两种确定模型剪枝率的不同方法。

    Topic modeling has emerged as a valuable tool for discovering patterns and topics within large collections of documents. However, when cross-analysis involves multiple parties, data privacy becomes a critical concern. Federated topic modeling has been developed to address this issue, allowing multiple parties to jointly train models while protecting pri-vacy. However, there are communication and performance challenges in the federated sce-nario. In order to solve the above problems, this paper proposes a method to establish a federated topic model while ensuring the privacy of each node, and use neural network model pruning to accelerate the model, where the client periodically sends the model neu-ron cumulative gradients and model weights to the server, and the server prunes the model. To address different requirements, two different methods are proposed to determine the model pruning rate. The first method involves slow pruning throughout the entire model training process, which has 
    
[^4]: 机会平等对统计性歧视的影响

    The Impact of Equal Opportunity on Statistical Discrimination. (arXiv:2310.04585v1 [econ.TH])

    [http://arxiv.org/abs/2310.04585](http://arxiv.org/abs/2310.04585)

    本文通过修改统计性歧视模型，考虑了由机器学习生成的可合同化信念，给监管者提供了一种超过肯定行动的工具，通过要求公司选取一个平衡不同群体真正阳性率的决策策略，实现机会平等来消除统计性歧视。

    

    本文修改了Coate和Loury（1993）的经典统计性歧视模型，假设公司对个体未观察到的类别的信念是由机器学习生成的，因此是可合同化的。这扩展了监管者的工具箱，超出了像肯定行动这样的无信念规定。可合同化的信念使得要求公司选择一个决策策略，使得不同群体之间的真正阳性率相等（算法公平文献中所称的机会平等）成为可能。尽管肯定行动不一定能消除统计性歧视，但本文表明实施机会平等可以做到。

    I modify the canonical statistical discrimination model of Coate and Loury (1993) by assuming the firm's belief about an individual's unobserved class is machine learning-generated and, therefore, contractible. This expands the toolkit of a regulator beyond belief-free regulations like affirmative action. Contractible beliefs make it feasible to require the firm to select a decision policy that equalizes true positive rates across groups -- what the algorithmic fairness literature calls equal opportunity. While affirmative action does not necessarily end statistical discrimination, I show that imposing equal opportunity does.
    

