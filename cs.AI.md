# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FLawN-T5: An Empirical Examination of Effective Instruction-Tuning Data Mixtures for Legal Reasoning](https://arxiv.org/abs/2404.02127) | 本研究提出了一个名为LawInstruct的大型法律指导数据集，证明了领域特定的预训练和指导调整可以改善在LegalBench上的性能，为在法律领域开发具有更强信息处理和决策能力的模型提供了一个资源。 |
| [^2] | [Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off](https://arxiv.org/abs/2402.14648) | 重新审视了基于表示的不变性正则化方法，提出了Asymmetrically Representation-regularized Adversarial Training (AR-AT)来解决“梯度冲突”和混合分布问题，改善鲁棒性-准确性权衡。 |
| [^3] | [Counter-intuitive: Large Language Models Can Better Understand Knowledge Graphs Than We Thought](https://arxiv.org/abs/2402.11541) | 本文通过对KG知识注入方法进行全面比较，探索为LLMs提供知识图谱知识的最佳方法，以增强它们的理解能力。 |
| [^4] | [ProtChatGPT: Towards Understanding Proteins with Large Language Models](https://arxiv.org/abs/2402.09649) | ProtChatGPT是一个基于大型语言模型的系统，通过自然语言学习和理解蛋白质结构，为用户提供上传蛋白质、提问和交互式对话等功能，有助于进一步理解蛋白质的结构与功能关系。 |
| [^5] | [Avoiding Catastrophe in Continuous Spaces by Asking for Help](https://arxiv.org/abs/2402.08062) | 在连续空间中，通过寻求帮助来避免灾难。引入了一种上下文多臂赌博问题的变体，目标是最小化灾难发生的概率。提出了一种算法，在连续1D状态空间和相对简单的回报函数下，遗憾和向导师查询率都趋近于0。 |
| [^6] | [Explicitly Disentangled Representations in Object-Centric Learning.](http://arxiv.org/abs/2401.10148) | 这篇论文提出了一种在物体中心化学习中明确解开形状和纹理成分的方法，通过将潜在空间划分为两个不重叠的子集，使得模型更加稳定和有效。 |
| [^7] | [The opaque law of artificial intelligence.](http://arxiv.org/abs/2310.13192) | 本文分析了算法的不透明性，重点关注人工智能在因果责任领域中的应用。通过对目前最好的生成式人工智能模型（Chat-GPT）的评估，可以了解其目前的性能以及可能的法律规制形式。 |
| [^8] | [Brain-Inspired Computational Intelligence via Predictive Coding.](http://arxiv.org/abs/2308.07870) | 这项研究介绍了一种通过预测编码的脑启发式计算智能方法，它可以解决现有人工智能方法的一些重要限制，并具有在机器学习领域有希望的应用潜力。 |
| [^9] | [One Transformer for All Time Series: Representing and Training with Time-Dependent Heterogeneous Tabular Data.](http://arxiv.org/abs/2302.06375) | 本研究提出了一种Transformer架构，用于表示具有时间相关的异构表格数据，通过使用一组频率函数来表示数值特征，并采用唯一的损失函数进行统一训练。 |

# 详细

[^1]: FLawN-T5: 有效指导调整数据混合在法律推理中的实证研究

    FLawN-T5: An Empirical Examination of Effective Instruction-Tuning Data Mixtures for Legal Reasoning

    [https://arxiv.org/abs/2404.02127](https://arxiv.org/abs/2404.02127)

    本研究提出了一个名为LawInstruct的大型法律指导数据集，证明了领域特定的预训练和指导调整可以改善在LegalBench上的性能，为在法律领域开发具有更强信息处理和决策能力的模型提供了一个资源。

    

    arXiv:2404.02127v1  公告类型: 跨领域  摘要: 指导调整是使语言模型对直接用户交互有效的重要步骤。然而，许多法律任务仍然超出了大多数开放式LLMs的范围，而且目前该领域还没有任何大规模的数据集。这严重限制了该应用领域的研究。在这项工作中，我们策划了一个名为LawInstruct的大型法律指导数据集，涵盖了17个司法管辖区、24种语言，总计1200万个示例。我们呈现证据表明，领域特定的预训练和指导调整能够改善在LegalBench上的性能，包括将Flan-T5 XL在基准线上提高8个点或16%。然而，该效应并不适用于所有任务、训练模式、模型大小和其他因素。LawInstruct是一个资源，可以加速在法律领域开发具有更强信息处理和决策能力的模型。

    arXiv:2404.02127v1 Announce Type: cross  Abstract: Instruction tuning is an important step in making language models useful for direct user interaction. However, many legal tasks remain out of reach for most open LLMs and there do not yet exist any large scale instruction datasets for the domain. This critically limits research in this application area. In this work, we curate LawInstruct, a large legal instruction dataset, covering 17 jurisdictions, 24 languages and a total of 12M examples. We present evidence that domain-specific pretraining and instruction tuning improve performance on LegalBench, including improving Flan-T5 XL by 8 points or 16\% over the baseline. However, the effect does not generalize across all tasks, training regimes, model sizes, and other factors. LawInstruct is a resource for accelerating the development of models with stronger information processing and decision making capabilities in the legal domain.
    
[^2]: 在对抗训练中重新思考不变性正则化以改善鲁棒性-准确性权衡

    Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off

    [https://arxiv.org/abs/2402.14648](https://arxiv.org/abs/2402.14648)

    重新审视了基于表示的不变性正则化方法，提出了Asymmetrically Representation-regularized Adversarial Training (AR-AT)来解决“梯度冲突”和混合分布问题，改善鲁棒性-准确性权衡。

    

    尽管对抗训练一直是抵抗对抗性样本（AEs）的最先进方法，但它们存在鲁棒性-准确性权衡问题。在这项研究中，我们重新审视基于表示的不变性正则化，学习具有辨别性却对抗性不变的表示，旨在缓解这种权衡。我们在经验上确定了妨碍不变性正则化的两个关键问题：（1）不变性损失和分类目标之间的“梯度冲突”，表明存在“崩溃解”，以及（2）由于干净和对抗性输入的分布发散而出现的混合分布问题。为了解决这些问题，我们提出了一种不对称表示正则化的对抗训练（AR-AT），该方法结合了一个停止梯度操作和一个预测器来避免“崩溃解”，灵感来自最近的非对比自监督学习。

    arXiv:2402.14648v1 Announce Type: cross  Abstract: Although adversarial training has been the state-of-the-art approach to defend against adversarial examples (AEs), they suffer from a robustness-accuracy trade-off. In this work, we revisit representation-based invariance regularization to learn discriminative yet adversarially invariant representations, aiming to mitigate this trade-off. We empirically identify two key issues hindering invariance regularization: (1) a "gradient conflict" between invariance loss and classification objectives, indicating the existence of "collapsing solutions," and (2) the mixture distribution problem arising from diverged distributions of clean and adversarial inputs. To address these issues, we propose Asymmetrically Representation-regularized Adversarial Training (AR-AT), which incorporates a stop-gradient operation and a pre-dictor in the invariance loss to avoid "collapsing solutions," inspired by a recent non-contrastive self-supervised learning a
    
[^3]: 逆向认知：大型语言模型比我们想象的更擅长理解知识图谱

    Counter-intuitive: Large Language Models Can Better Understand Knowledge Graphs Than We Thought

    [https://arxiv.org/abs/2402.11541](https://arxiv.org/abs/2402.11541)

    本文通过对KG知识注入方法进行全面比较，探索为LLMs提供知识图谱知识的最佳方法，以增强它们的理解能力。

    

    虽然通过使用知识图谱（KGs）来增强大型语言模型（LLMs）的推理能力并减少它们的幻觉的方法受到了广泛关注，但目前对如何使LLMs能够即时整合KGs中的结构化知识的探索还不足。本文采用复杂问题回答（CQA）作为一项任务，评估LLM理解KG知识的能力。我们对KG知识注入方法进行了全面比较（从三元组到自然语言文本），旨在探索为LLMs提供KG知识的最佳提示方法，从而增强它们的理解能力。

    arXiv:2402.11541v1 Announce Type: cross  Abstract: Although the method of enhancing large language models' (LLMs') reasoning ability and reducing their hallucinations through the use of knowledge graphs (KGs) has received widespread attention, the exploration of how to enable LLMs to integrate the structured knowledge in KGs on-the-fly remains inadequate. Researchers often co-train KG embeddings and LLM parameters to equip LLMs with the ability of comprehending KG knowledge. However, this resource-hungry training paradigm significantly increases the model learning cost and is also unsuitable for non-open-source, black-box LLMs. In this paper, we employ complex question answering (CQA) as a task to assess the LLM's ability of comprehending KG knowledge. We conducted a comprehensive comparison of KG knowledge injection methods (from triples to natural language text), aiming to explore the optimal prompting method for supplying KG knowledge to LLMs, thereby enhancing their comprehension o
    
[^4]: ProtChatGPT：用于理解大规模语言模型的蛋白质

    ProtChatGPT: Towards Understanding Proteins with Large Language Models

    [https://arxiv.org/abs/2402.09649](https://arxiv.org/abs/2402.09649)

    ProtChatGPT是一个基于大型语言模型的系统，通过自然语言学习和理解蛋白质结构，为用户提供上传蛋白质、提问和交互式对话等功能，有助于进一步理解蛋白质的结构与功能关系。

    

    蛋白质研究在各个基础学科中至关重要，但理解其复杂的结构与功能关系仍然具有挑战性。最近的大型语言模型（LLMs）在理解特定任务的知识方面取得了重大进展，这表明了用于蛋白质的ChatGPT-like系统在促进基础研究方面的潜力。在这项工作中，我们介绍了ProtChatGPT，旨在通过自然语言学习和理解蛋白质结构。ProtChatGPT使用户可以上传蛋白质、提问并进行交互式对话以产生全面的回答。该系统包括蛋白编码器、蛋白语言相关转换器（PLP-former）、投影适配器和LLM。蛋白质首先通过蛋白编码器和PLP-former进行编码以产生蛋白质嵌入，然后通过适配器将其投射到与LLM相符合。最后，LLM将用户的问题与蛋白质嵌入进行综合处理。

    arXiv:2402.09649v1 Announce Type: cross  Abstract: Protein research is crucial in various fundamental disciplines, but understanding their intricate structure-function relationships remains challenging. Recent Large Language Models (LLMs) have made significant strides in comprehending task-specific knowledge, suggesting the potential for ChatGPT-like systems specialized in protein to facilitate basic research. In this work, we introduce ProtChatGPT, which aims at learning and understanding protein structures via natural languages. ProtChatGPT enables users to upload proteins, ask questions, and engage in interactive conversations to produce comprehensive answers. The system comprises protein encoders, a Protein-Language Pertaining Transformer (PLP-former), a projection adapter, and an LLM. The protein first undergoes protein encoders and PLP-former to produce protein embeddings, which are then projected by the adapter to conform with the LLM. The LLM finally combines user questions wit
    
[^5]: 避免连续空间中的灾难：通过寻求帮助

    Avoiding Catastrophe in Continuous Spaces by Asking for Help

    [https://arxiv.org/abs/2402.08062](https://arxiv.org/abs/2402.08062)

    在连续空间中，通过寻求帮助来避免灾难。引入了一种上下文多臂赌博问题的变体，目标是最小化灾难发生的概率。提出了一种算法，在连续1D状态空间和相对简单的回报函数下，遗憾和向导师查询率都趋近于0。

    

    大多数具有正式遗憾保证的强化学习算法假设所有错误都是可逆的，并依赖于尝试所有可能的选项。当一些错误是无法修复甚至是灾难性的时，这种方法会导致糟糕的结果。我们提出了一种上下文多臂赌博问题的变体，在这个问题中，目标是最小化发生灾难的概率。具体而言，我们假设每轮的回报代表了在该轮避免灾难的概率，并尝试最大化回报的乘积（总体避免灾难的概率）。为了给 agent 一些成功的机会，我们允许有限次向导师提问，并假设回报函数为 Lipschitz 连续的。我们提出了一种算法，当时间跨度增长时，它的遗憾和向导师查询率都趋近于 0，假设是一个连续的 1D 状态空间和相对"简单"的回报函数。我们还提供了一个匹配的下界：在没有简单性假设的情况下，任何算法要么不断查询异常的行为，要么每次查询完全相同的行为。

    Most reinforcement learning algorithms with formal regret guarantees assume all mistakes are reversible and rely on essentially trying all possible options. This approach leads to poor outcomes when some mistakes are irreparable or even catastrophic. We propose a variant of the contextual bandit problem where the goal is to minimize the chance of catastrophe. Specifically, we assume that the payoff each round represents the chance of avoiding catastrophe that round, and try to maximize the product of payoffs (the overall chance of avoiding catastrophe). To give the agent some chance of success, we allow a limited number of queries to a mentor and assume a Lipschitz continuous payoff function. We present an algorithm whose regret and rate of querying the mentor both approach 0 as the time horizon grows, assuming a continuous 1D state space and a relatively "simple" payoff function. We also provide a matching lower bound: without the simplicity assumption: any algorithm either constantly
    
[^6]: 在物体中心化学习中明确解开的表示

    Explicitly Disentangled Representations in Object-Centric Learning. (arXiv:2401.10148v1 [cs.CV])

    [http://arxiv.org/abs/2401.10148](http://arxiv.org/abs/2401.10148)

    这篇论文提出了一种在物体中心化学习中明确解开形状和纹理成分的方法，通过将潜在空间划分为两个不重叠的子集，使得模型更加稳定和有效。

    

    从原始视觉数据中提取结构化表示是机器学习中一个重要且长期存在的挑战。最近，无监督学习物体中心化表示的技术引起了越来越多的关注。在这个背景下，增强潜在特征的稳定性可以提高下游任务训练的效率和效果。在这个方向上一个有希望的步骤是解开导致数据变化的因素。先前，不变卡槽注意实现了从其他特征中解开位置、尺度和方向。扩展这一方法，我们着重于分离形状和纹理组成部分。特别地，我们提出了一种新颖的架构，将物体中心化模型中的形状和纹理成分偏置为潜在空间维度的两个不重叠子集。这些子集是先验已知的，因此在训练过程之前。在一系列物体中心化测试中进行的实验揭示了...

    Extracting structured representations from raw visual data is an important and long-standing challenge in machine learning. Recently, techniques for unsupervised learning of object-centric representations have raised growing interest. In this context, enhancing the robustness of the latent features can improve the efficiency and effectiveness of the training of downstream tasks. A promising step in this direction is to disentangle the factors that cause variation in the data. Previously, Invariant Slot Attention disentangled position, scale, and orientation from the remaining features. Extending this approach, we focus on separating the shape and texture components. In particular, we propose a novel architecture that biases object-centric models toward disentangling shape and texture components into two non-overlapping subsets of the latent space dimensions. These subsets are known a priori, hence before the training process. Experiments on a range of object-centric benchmarks reveal t
    
[^7]: 人工智能的不透明法律

    The opaque law of artificial intelligence. (arXiv:2310.13192v1 [cs.AI])

    [http://arxiv.org/abs/2310.13192](http://arxiv.org/abs/2310.13192)

    本文分析了算法的不透明性，重点关注人工智能在因果责任领域中的应用。通过对目前最好的生成式人工智能模型（Chat-GPT）的评估，可以了解其目前的性能以及可能的法律规制形式。

    

    本文旨在分析算法的不透明性，并将其置于人工智能因果责任的公开辩论背景下进行讨论；通过应用图灵测试中提出的对话方法，我们希望评估现有最好的生成式人工智能模型（Chat-GPT）的性能，以确定其目前的能力和可能的法律规制形式。问题分析将基于对传统法律范畴（如因果关系、意图和过失）的评论，以理解人工智能使用中的问题，特别关注人机交互。从计算机科学角度来看，文中还将提出一种针对Chat-GPT进行实际询问的方法，以找到人工智能运行过程中的一些关键问题。文章的结尾将集中讨论一些现有的立法措施。

    The purpose of this paper is to analyse the opacity of algorithms, contextualized in the open debate on responsibility for artificial intelligence causation; with an experimental approach by which, applying the proposed conversational methodology of the Turing Test, we expect to evaluate the performance of one of the best existing NLP model of generative AI (Chat-GPT) to see how far it can go right now and how the shape of a legal regulation of it could be. The analysis of the problem will be supported by a comment of Italian classical law categories such as causality, intent and fault to understand the problem of the usage of AI, focusing in particular on the human-machine interaction. On the computer science side, for a technical point of view of the logic used to craft these algorithms, in the second chapter will be proposed a practical interrogation of Chat-GPT aimed at finding some critical points of the functioning of AI. The end of the paper will concentrate on some existing leg
    
[^8]: 通过预测编码实现脑启发式计算智能

    Brain-Inspired Computational Intelligence via Predictive Coding. (arXiv:2308.07870v1 [cs.AI])

    [http://arxiv.org/abs/2308.07870](http://arxiv.org/abs/2308.07870)

    这项研究介绍了一种通过预测编码的脑启发式计算智能方法，它可以解决现有人工智能方法的一些重要限制，并具有在机器学习领域有希望的应用潜力。

    

    人工智能（AI）正在迅速成为本世纪的关键技术之一。到目前为止，在AI领域取得的大部分成果都是使用误差反向传播学习算法训练的深度神经网络所实现的。然而，这种方法的普及应用已经凸显出了一些重要的局限性，例如计算成本高、难以量化不确定性、缺乏鲁棒性、不可靠性和生物学上的不合理性。解决这些限制可能需要受到神经科学理论的启发和指导的方案。其中一种理论称为预测编码（PC），在机器智能任务中表现出有希望的性能，具有令人兴奋的特性，使其在机器学习社区中具有潜在的价值：PC可以模拟不同脑区的信息处理，可以用于认知控制和机器人技术，并在变分推理方面具有坚实的数学基础，提供了一个强大的工具。

    Artificial intelligence (AI) is rapidly becoming one of the key technologies of this century. The majority of results in AI thus far have been achieved using deep neural networks trained with the error backpropagation learning algorithm. However, the ubiquitous adoption of this approach has highlighted some important limitations such as substantial computational cost, difficulty in quantifying uncertainty, lack of robustness, unreliability, and biological implausibility. It is possible that addressing these limitations may require schemes that are inspired and guided by neuroscience theories. One such theory, called predictive coding (PC), has shown promising performance in machine intelligence tasks, exhibiting exciting properties that make it potentially valuable for the machine learning community: PC can model information processing in different brain areas, can be used in cognitive control and robotics, and has a solid mathematical grounding in variational inference, offering a pow
    
[^9]: 一种适用于所有时间序列的Transformer：表示和训练具有时间相关的异构表格数据

    One Transformer for All Time Series: Representing and Training with Time-Dependent Heterogeneous Tabular Data. (arXiv:2302.06375v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.06375](http://arxiv.org/abs/2302.06375)

    本研究提出了一种Transformer架构，用于表示具有时间相关的异构表格数据，通过使用一组频率函数来表示数值特征，并采用唯一的损失函数进行统一训练。

    

    近年来，将深度学习技术应用于表格数据的兴趣日益增长，以复制其他人工智能领域在这一结构化领域的成功。特别有趣的是，表格数据具有时间依赖性，例如金融交易。然而，表格值的异质性，其中类别元素与数值项混合，使得这种适应变得困难。在本文中，我们提出了一种Transformer架构来表示异构的时间相关的表格数据，数值特征使用一组频率函数表示，并且整个网络使用唯一的损失函数进行统一训练。

    There is a recent growing interest in applying Deep Learning techniques to tabular data, in order to replicate the success of other Artificial Intelligence areas in this structured domain. Specifically interesting is the case in which tabular data have a time dependence, such as, for instance financial transactions. However, the heterogeneity of the tabular values, in which categorical elements are mixed with numerical items, makes this adaptation difficult. In this paper we propose a Transformer architecture to represent heterogeneous time-dependent tabular data, in which numerical features are represented using a set of frequency functions and the whole network is uniformly trained with a unique loss function.
    

