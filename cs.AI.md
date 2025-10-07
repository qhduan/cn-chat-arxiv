# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [K-Link: Knowledge-Link Graph from LLMs for Enhanced Representation Learning in Multivariate Time-Series Data](https://arxiv.org/abs/2403.03645) | 提出了一种名为K-Link的框架，利用大型语言模型编码通用知识，提取了Knowledge-Link图以捕获传感器之间的广泛语义知识和联系。 |
| [^2] | [PolyNet: Learning Diverse Solution Strategies for Neural Combinatorial Optimization](https://arxiv.org/abs/2402.14048) | PolyNet通过学习互补解决策略来改善解空间探索，避免了人为规则导致解决方案质量下降的问题。 |
| [^3] | [Jailbreaking Proprietary Large Language Models using Word Substitution Cipher](https://arxiv.org/abs/2402.10601) | 本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。 |
| [^4] | [Synthesizing Sentiment-Controlled Feedback For Multimodal Text and Image Data](https://arxiv.org/abs/2402.07640) | 该论文提出了一个可控的多模态反馈合成系统，能够根据文本和图像输入生成具有特定情感（积极或消极）的反馈，有着广泛的应用价值。 |
| [^5] | [Tabular Data: Is Attention All You Need?](https://arxiv.org/abs/2402.03970) | 本文引入了一项大规模实证研究，比较了神经网络和梯度提升决策树在表格数据上的表现，还比较了基于Transformer的架构和传统的多层感知器（MLP）与残差连接的架构。实证结果显示，神经网络在决策树方面具有竞争力，而基于Transformer的架构在表格数据集上并没有超过传统MLP架构的简化变体。 |
| [^6] | [Sum-of-Parts Models: Faithful Attributions for Groups of Features.](http://arxiv.org/abs/2310.16316) | Sum-of-Parts模型通过构造保证特征组归因的忠实性，将预测分解为可解释的分数之和，帮助天体物理学家发现了关于星系形成的新知识。 |
| [^7] | [AgentBench: Evaluating LLMs as Agents.](http://arxiv.org/abs/2308.03688) | AgentBench是一个用于评估LLMs作为代理人的多维度基准，发现在复杂环境中，商业LLMs在充当代理人方面表现强劲，但与开源竞争对手相比，存在显著性能差距。该研究揭示了LLMs在长期推理、决策和指令遵循能力上的瓶颈。 |
| [^8] | [Tutorial on amortized optimization.](http://arxiv.org/abs/2202.00665) | 该教程介绍了分摊优化的基础，并总结了其在变分推断、稀疏编码、元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。 |

# 详细

[^1]: K-Link：基于LLMs的知识链接图在多元时间序列数据增强表示学习中的应用

    K-Link: Knowledge-Link Graph from LLMs for Enhanced Representation Learning in Multivariate Time-Series Data

    [https://arxiv.org/abs/2403.03645](https://arxiv.org/abs/2403.03645)

    提出了一种名为K-Link的框架，利用大型语言模型编码通用知识，提取了Knowledge-Link图以捕获传感器之间的广泛语义知识和联系。

    

    从各种传感器采集并按时间顺序组织的多元时间序列（MTS）数据涉及关键的时空依赖性，如传感器之间的相关性。为了捕捉这些依赖关系，图神经网络（GNNs）已经成为强大的工具，但它们的有效性受到从MTS数据构建图的质量限制。通常，现有方法仅从MTS信号构建图，这可能会由于小训练数据集而引入偏差，并可能无法准确表示底层依赖关系。为了解决这一挑战，我们提出了一个名为K-Link的新框架，利用大型语言模型（LLMs）来编码广泛的通用知识，从而提供有效的解决方案以减少偏差。利用LLMs中嵌入的知识，例如物理原理，我们提取了一个Knowledge-Link图，捕获了传感器的广泛语义知识和传感器之间的链接。

    arXiv:2403.03645v1 Announce Type: new  Abstract: Sourced from various sensors and organized chronologically, Multivariate Time-Series (MTS) data involves crucial spatial-temporal dependencies, e.g., correlations among sensors. To capture these dependencies, Graph Neural Networks (GNNs) have emerged as powerful tools, yet their effectiveness is restricted by the quality of graph construction from MTS data. Typically, existing approaches construct graphs solely from MTS signals, which may introduce bias due to a small training dataset and may not accurately represent underlying dependencies. To address this challenge, we propose a novel framework named K-Link, leveraging Large Language Models (LLMs) to encode extensive general knowledge and thereby providing effective solutions to reduce the bias. Leveraging the knowledge embedded in LLMs, such as physical principles, we extract a \textit{Knowledge-Link graph}, capturing vast semantic knowledge of sensors and the linkage of the sensor-le
    
[^2]: PolyNet：学习神经组合优化的多样化解决策略

    PolyNet: Learning Diverse Solution Strategies for Neural Combinatorial Optimization

    [https://arxiv.org/abs/2402.14048](https://arxiv.org/abs/2402.14048)

    PolyNet通过学习互补解决策略来改善解空间探索，避免了人为规则导致解决方案质量下降的问题。

    

    强化学习方法用于构建组合优化问题解决方案，迅速接近人类设计的算法性能。为了进一步缩小差距，基于学习的方法在搜索过程中必须高效地探索解空间。最近的方法通过强制实施多样化解生成来人为增加探索，然而，这些规则可能损害解决方案质量，并且难以为更复杂的问题设计。本文介绍了PolyNet，一种通过学习互补解决策略来改善解空间探索的方法。与其他作品不同，PolyNet仅使用单个解码器，并且训练图式不通过人为规则强制实施多样化解生成。我们在四个组合优化问题上评估PolyNet，并观察到隐式多样性机制允许P

    arXiv:2402.14048v1 Announce Type: cross  Abstract: Reinforcement learning-based methods for constructing solutions to combinatorial optimization problems are rapidly approaching the performance of human-designed algorithms. To further narrow the gap, learning-based approaches must efficiently explore the solution space during the search process. Recent approaches artificially increase exploration by enforcing diverse solution generation through handcrafted rules, however, these rules can impair solution quality and are difficult to design for more complex problems. In this paper, we introduce PolyNet, an approach for improving exploration of the solution space by learning complementary solution strategies. In contrast to other works, PolyNet uses only a single-decoder and a training schema that does not enforce diverse solution generation through handcrafted rules. We evaluate PolyNet on four combinatorial optimization problems and observe that the implicit diversity mechanism allows P
    
[^3]: 使用单词替换密码来越狱专有的大型语言模型

    Jailbreaking Proprietary Large Language Models using Word Substitution Cipher

    [https://arxiv.org/abs/2402.10601](https://arxiv.org/abs/2402.10601)

    本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。

    

    大型语言模型（LLMs）遵循道德和伦理准则，但仍然容易受到名为Jailbreak的创意提示的影响，这些提示可以绕过对齐过程。然而，大多数越狱提示包含自然语言（主要是英语）中的有害问题，可以被LLMs自身检测到。本文提出了使用密码技术编码的越狱提示。我们首先在最先进的LLM，GPT-4上进行了一个试点研究，解码了使用各种密码技术加密的几个安全句子，发现简单的单词替换密码可以被最有效地解码。受此结果启发，我们使用这种编码技术来编写越狱提示。我们提供了将不安全单词映射到安全单词，并使用这些映射的单词提出不安全问题的映射。实验结果显示，我们提出的越狱攻击成功率（高达59.42%）。

    arXiv:2402.10601v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbrea
    
[^4]: 合成对多模态文本和图片数据的情感控制反馈

    Synthesizing Sentiment-Controlled Feedback For Multimodal Text and Image Data

    [https://arxiv.org/abs/2402.07640](https://arxiv.org/abs/2402.07640)

    该论文提出了一个可控的多模态反馈合成系统，能够根据文本和图像输入生成具有特定情感（积极或消极）的反馈，有着广泛的应用价值。

    

    生成对多模态输入（包括文本和图片）的情感控制反馈能够弥补人机交互领域的一个关键差距，使系统能够提供具有同理心、准确性和引人入胜的回应。这种能力在医疗、营销和教育等领域有着深远的应用。为此，我们构建了一个大规模的可控多模态反馈合成（CMFeed）数据集，并提出了一个可控的反馈合成系统。所提出的系统包括一个编码器、解码器和控制性模块，用于处理文本和视觉输入。它使用Transformer和Faster R-CNN网络提取文本和视觉特征，并将它们结合起来生成反馈。CMFeed数据集包含图片、文本、对帖子的反应、带有相关性评分的人类评论以及对评论的反应。对帖子和评论的反应被用来训练提出的模型以产生具有特定（积极或消极）情感的反馈。

    The ability to generate sentiment-controlled feedback in response to multimodal inputs, comprising both text and images, addresses a critical gap in human-computer interaction by enabling systems to provide empathetic, accurate, and engaging responses. This capability has profound applications in healthcare, marketing, and education. To this end, we construct a large-scale Controllable Multimodal Feedback Synthesis (CMFeed) dataset and propose a controllable feedback synthesis system. The proposed system includes an encoder, decoder, and controllability block for textual and visual inputs. It extracts textual and visual features using a transformer and Faster R-CNN networks and combines them to generate feedback. The CMFeed dataset encompasses images, text, reactions to the post, human comments with relevance scores, and reactions to the comments. The reactions to the post and comments are utilized to train the proposed model to produce feedback with a particular (positive or negative)
    
[^5]: 表格数据：注意力是唯一需要的吗？

    Tabular Data: Is Attention All You Need?

    [https://arxiv.org/abs/2402.03970](https://arxiv.org/abs/2402.03970)

    本文引入了一项大规模实证研究，比较了神经网络和梯度提升决策树在表格数据上的表现，还比较了基于Transformer的架构和传统的多层感知器（MLP）与残差连接的架构。实证结果显示，神经网络在决策树方面具有竞争力，而基于Transformer的架构在表格数据集上并没有超过传统MLP架构的简化变体。

    

    深度学习彻底改变了人工智能领域，并在涉及图像和文本数据的应用中取得了令人瞩目的成就。遗憾的是，关于神经网络在结构化表格数据上的优势存在着不一致的证据。本文引入了一项大规模实证研究，比较了神经网络和梯度提升决策树在表格数据上的表现，还比较了基于Transformer的架构和传统的多层感知器（MLP）与残差连接的架构。与之前的研究相比，我们的实证发现表明神经网络在决策树方面具有竞争力。此外，我们还评估了基于Transformer的架构在表格数据集上并没有超过传统MLP架构的简化变体。因此，本文帮助研究和实践社区在未来的表格数据应用中做出明智的选择。

    Deep Learning has revolutionized the field of AI and led to remarkable achievements in applications involving image and text data. Unfortunately, there is inconclusive evidence on the merits of neural networks for structured tabular data. In this paper, we introduce a large-scale empirical study comparing neural networks against gradient-boosted decision trees on tabular data, but also transformer-based architectures against traditional multi-layer perceptrons (MLP) with residual connections. In contrast to prior work, our empirical findings indicate that neural networks are competitive against decision trees. Furthermore, we assess that transformer-based architectures do not outperform simpler variants of traditional MLP architectures on tabular datasets. As a result, this paper helps the research and practitioner communities make informed choices on deploying neural networks on future tabular data applications.
    
[^6]: Sum-of-Parts模型：对特征组的忠实归因

    Sum-of-Parts Models: Faithful Attributions for Groups of Features. (arXiv:2310.16316v1 [cs.LG])

    [http://arxiv.org/abs/2310.16316](http://arxiv.org/abs/2310.16316)

    Sum-of-Parts模型通过构造保证特征组归因的忠实性，将预测分解为可解释的分数之和，帮助天体物理学家发现了关于星系形成的新知识。

    

    如果机器学习模型的解释准确反映了其决策过程，则被认为是“忠实”的解释。然而，例如深度学习的特征归因等解释并不能保证忠实，有可能产生具有误导性的解释。在这项工作中，我们开发了Sum-of-Parts（SOP）模型，它是一类模型，其预测具有通过构造保证忠实的特征组归因。该模型将预测分解为可解释的分数之和，每个分数直接归因于一组稀疏特征。我们使用标准可解释性指标对SOP进行评估，并在一个案例研究中，利用SOP提供的忠实解释帮助天体物理学家发现了关于星系形成的新知识。

    An explanation of a machine learning model is considered "faithful" if it accurately reflects the model's decision-making process. However, explanations such as feature attributions for deep learning are not guaranteed to be faithful, and can produce potentially misleading interpretations. In this work, we develop Sum-of-Parts (SOP), a class of models whose predictions come with grouped feature attributions that are faithful-by-construction. This model decomposes a prediction into an interpretable sum of scores, each of which is directly attributable to a sparse group of features. We evaluate SOP on benchmarks with standard interpretability metrics, and in a case study, we use the faithful explanations from SOP to help astrophysicists discover new knowledge about galaxy formation.
    
[^7]: AgentBench: 评估LLMs作为代理人

    AgentBench: Evaluating LLMs as Agents. (arXiv:2308.03688v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2308.03688](http://arxiv.org/abs/2308.03688)

    AgentBench是一个用于评估LLMs作为代理人的多维度基准，发现在复杂环境中，商业LLMs在充当代理人方面表现强劲，但与开源竞争对手相比，存在显著性能差距。该研究揭示了LLMs在长期推理、决策和指令遵循能力上的瓶颈。

    

    大型语言模型(LLMs)变得越来越智能和自主，针对传统的NLP任务之外的现实世界实际任务。因此，迫切需要在互动环境中评估LLMs作为代理人在具有挑战性的任务上的推理和决策能力。我们提出了AgentBench，一个多维度演变的基准，目前包括8个不同的环境，以评估LLM作为代理人在多轮开放式生成设置中的推理和决策能力。我们在27个基于API和开源的LLM上进行了广泛的测试，结果表明，虽然顶级商业LLM在复杂环境中表现出良好的代理人能力，但它们与开源竞争对手之间的性能差距很大。我们找出了环境和LLM中失败的典型原因，表明长期推理、决策和遵循指示能力不佳是开发可用LLM代理人的主要障碍。通过对代码和高质量进行训练

    Large Language Models (LLMs) are becoming increasingly smart and autonomous, targeting real-world pragmatic missions beyond traditional NLP tasks. As a result, there has been an urgent need to evaluate LLMs as agents on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional evolving benchmark that currently consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities in a multi-turn open-ended generation setting. Our extensive test over 27 API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and OSS competitors. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Training on code and high quality 
    
[^8]: 关于分摊优化的教程

    Tutorial on amortized optimization. (arXiv:2202.00665v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.00665](http://arxiv.org/abs/2202.00665)

    该教程介绍了分摊优化的基础，并总结了其在变分推断、稀疏编码、元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。

    

    优化是一种普遍的建模工具，经常在反复解决相同问题的情况下使用。分摊优化方法使用学习来预测这些设置中问题的解决方案，利用相似问题实例之间的共享结构。这些方法在变分推断和强化学习中至关重要，能够比不使用分摊的传统优化方法快几个数量级地解决优化问题。本次教程介绍了这些进步背后的分摊优化基础，并概述了它们在变分推断、稀疏编码、基于梯度的元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。本教程的源代码可在https://github.com/facebookresearch/amortized-optimization-tutorial上获得。

    Optimization is a ubiquitous modeling tool and is often deployed in settings which repeatedly solve similar instances of the same problem. Amortized optimization methods use learning to predict the solutions to problems in these settings, exploiting the shared structure between similar problem instances. These methods have been crucial in variational inference and reinforcement learning and are capable of solving optimization problems many orders of magnitudes times faster than traditional optimization methods that do not use amortization. This tutorial presents an introduction to the amortized optimization foundations behind these advancements and overviews their applications in variational inference, sparse coding, gradient-based meta-learning, control, reinforcement learning, convex optimization, optimal transport, and deep equilibrium networks. The source code for this tutorial is available at https://github.com/facebookresearch/amortized-optimization-tutorial.
    

