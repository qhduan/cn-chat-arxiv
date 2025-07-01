# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Support Vectors](https://arxiv.org/abs/2403.17329) | 该论文探索了深度学习模型中的深度支持向量（DSVs）的概念，介绍了DeepKKT条件，通过实证研究发现DSVs与SVM中的支持向量类似，为解释模型决策标准提供了方法，同时证明了可以有效地使用DSVs重构模型。 |
| [^2] | [Brain-inspired and Self-based Artificial Intelligence](https://arxiv.org/abs/2402.18784) | 该论文介绍了一种名为Brain-inspired and Self-based Artificial Intelligence（BriSe AI）的新人工智能范式，通过自我组织的方式协调各种认知功能和学习策略，以构建人类水平的AI模型和机器人应用。 |
| [^3] | [An Autonomous Large Language Model Agent for Chemical Literature Data Mining](https://arxiv.org/abs/2402.12993) | 介绍了一个端到端的人工智能代理框架，利用大型语言模型实现从化学文献中高保真提取信息，充当化学助手的角色，自动化数据收集和分析，从而提高工作效率。 |
| [^4] | [ResQuNNs:Towards Enabling Deep Learning in Quantum Convolution Neural Networks](https://arxiv.org/abs/2402.09146) | 本文介绍了一种增强量子卷积神经网络性能的新框架ResQuNNs，在quanvolutional层中引入可训练性，通过残差学习的概念解决了跨层梯度访问的问题。 |
| [^5] | [Benchmarking Spiking Neural Network Learning Methods with Varying Locality](https://arxiv.org/abs/2402.01782) | 本研究使用不同局部性对脉冲神经网络学习方法进行基准测试，并发现这些方法在性能和生物学合理性之间存在权衡。此外，研究还探讨了SNN的隐式循环特性。 |
| [^6] | [Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning.](http://arxiv.org/abs/2310.11594) | 本文研究了联邦学习中对抗性训练和后门攻击的交叉点，引入了Adversarial Robustness Unhardening（ARU），通过有意介入分散式训练过程中破坏模型的鲁棒性，使模型更容易受到更广泛的逃避攻击。 |
| [^7] | [Scaling Data-Constrained Language Models.](http://arxiv.org/abs/2305.16264) | 研究人员研究了在数据受限制的情况下缩放语言模型，并提出了一个计算最优性的缩放定律，考虑到重复令牌和过量参数的价值递减。 |

# 详细

[^1]: 深度支持向量

    Deep Support Vectors

    [https://arxiv.org/abs/2403.17329](https://arxiv.org/abs/2403.17329)

    该论文探索了深度学习模型中的深度支持向量（DSVs）的概念，介绍了DeepKKT条件，通过实证研究发现DSVs与SVM中的支持向量类似，为解释模型决策标准提供了方法，同时证明了可以有效地使用DSVs重构模型。

    

    尽管深度学习的成功通常被归因于其与支持向量机（SVM）在理论上的等价性，但这种关系的实际影响尚未得到全面探讨。本文在这一领域开展了一项探索，重点关注深度学习模型中深度支持向量（DSVs）的识别。我们引入了DeepKKT条件的概念，这是一种专为深度学习量身定制的传统Karush-Kuhn-Tucker（KKT）条件的调整版本。通过实证研究，我们阐明了DSVs与SVM中的支持向量之间存在相似性，提供了一种解释模型决策标准的切实方法。此外，我们的研究结果表明，可以有效地使用DSVs重构模型，类似于SVM中的过程。代码将会公开。

    arXiv:2403.17329v1 Announce Type: cross  Abstract: While the success of deep learning is commonly attributed to its theoretical equivalence with Support Vector Machines (SVM), the practical implications of this relationship have not been thoroughly explored. This paper pioneers an exploration in this domain, specifically focusing on the identification of Deep Support Vectors (DSVs) within deep learning models. We introduce the concept of DeepKKT conditions, an adaptation of the traditional Karush-Kuhn-Tucker (KKT) conditions tailored for deep learning. Through empirical investigations, we illustrate that DSVs exhibit similarities to support vectors in SVM, offering a tangible method to interpret the decision-making criteria of models. Additionally, our findings demonstrate that models can be effectively reconstructed using DSVs, resembling the process in SVM. The code will be available.
    
[^2]: 大脑启发和基于自我的人工智能

    Brain-inspired and Self-based Artificial Intelligence

    [https://arxiv.org/abs/2402.18784](https://arxiv.org/abs/2402.18784)

    该论文介绍了一种名为Brain-inspired and Self-based Artificial Intelligence（BriSe AI）的新人工智能范式，通过自我组织的方式协调各种认知功能和学习策略，以构建人类水平的AI模型和机器人应用。

    

    这篇论文挑战了当前AI的支持者所谓 "思考机器" 的观念，因为它们缺乏自我意识。该论文提出了一种名为Brain-inspired and Self-based Artificial Intelligence（BriSe AI）的新人工智能范式，旨在通过自我组织的方式协调各种认知功能和学习策略，构建人类水平的AI模型和机器人应用。

    arXiv:2402.18784v1 Announce Type: new  Abstract: The question "Can machines think?" and the Turing Test to assess whether machines could achieve human-level intelligence is one of the roots of AI. With the philosophical argument "I think, therefore I am", this paper challenge the idea of a "thinking machine" supported by current AIs since there is no sense of self in them. Current artificial intelligence is only seemingly intelligent information processing and does not truly understand or be subjectively aware of oneself and perceive the world with the self as human intelligence does. In this paper, we introduce a Brain-inspired and Self-based Artificial Intelligence (BriSe AI) paradigm. This BriSe AI paradigm is dedicated to coordinating various cognitive functions and learning strategies in a self-organized manner to build human-level AI models and robotic applications. Specifically, BriSe AI emphasizes the crucial role of the Self in shaping the future AI, rooted with a practical hi
    
[^3]: 用于化学文献数据挖掘的自主大型语言模型代理

    An Autonomous Large Language Model Agent for Chemical Literature Data Mining

    [https://arxiv.org/abs/2402.12993](https://arxiv.org/abs/2402.12993)

    介绍了一个端到端的人工智能代理框架，利用大型语言模型实现从化学文献中高保真提取信息，充当化学助手的角色，自动化数据收集和分析，从而提高工作效率。

    

    化学合成对于推动材料合成和药物发现至关重要，影响着包括环境科学和医疗保健在内的各个领域。化学领域的技术上升使得产生了大量的化学数据，挑战研究人员去识别模式并细化合成过程。人工智能通过分析数据来优化合成并提高产量。然而，人工智能在处理文献数据方面面临着挑战，因为化学文献的结构不规整，写作风格多样。为了克服这些困难，我们引入了一个端到端的人工智能代理框架，能够从广泛的化学文献中高保真地提取信息。这个人工智能代理采用大型语言模型（LLMs）进行快速生成和迭代优化。它充当化学助手的角色，自动化数据收集和分析，从而节省人力并提高性能。

    arXiv:2402.12993v1 Announce Type: cross  Abstract: Chemical synthesis, which is crucial for advancing material synthesis and drug discovery, impacts various sectors including environmental science and healthcare. The rise of technology in chemistry has generated extensive chemical data, challenging researchers to discern patterns and refine synthesis processes. Artificial intelligence (AI) helps by analyzing data to optimize synthesis and increase yields. However, AI faces challenges in processing literature data due to the unstructured format and diverse writing style of chemical literature. To overcome these difficulties, we introduce an end-to-end AI agent framework capable of high-fidelity extraction from extensive chemical literature. This AI agent employs large language models (LLMs) for prompt generation and iterative optimization. It functions as a chemistry assistant, automating data collection and analysis, thereby saving manpower and enhancing performance. Our framework's ef
    
[^4]: ResQuNNs: 实现量子卷积神经网络中深度学习的新框架

    ResQuNNs:Towards Enabling Deep Learning in Quantum Convolution Neural Networks

    [https://arxiv.org/abs/2402.09146](https://arxiv.org/abs/2402.09146)

    本文介绍了一种增强量子卷积神经网络性能的新框架ResQuNNs，在quanvolutional层中引入可训练性，通过残差学习的概念解决了跨层梯度访问的问题。

    

    本文提出了一种增强量子卷积神经网络（QuNNs）性能的新框架，通过引入可训练的quanvolutional层并解决与其相关的关键挑战。传统的quanvolutional层虽然有助于特征提取，但往往是静态的，适应性有限。与最先进的研究不同，我们的研究通过在这些层内部进行训练，显著提高了QuNNs的灵活性和潜力。然而，多个可训练的quanvolutional层的引入给基于梯度的优化带来了复杂性，主要是由于难以在这些层之间访问梯度。为了解决这个问题，我们提出了一种新的架构，Residual Quanvolutional Neural Networks (ResQuNNs)，利用残差学习的概念，在这些层之间添加跳过连接以促进梯度的流动。

    arXiv:2402.09146v1 Announce Type: new Abstract: In this paper, we present a novel framework for enhancing the performance of Quanvolutional Neural Networks (QuNNs) by introducing trainable quanvolutional layers and addressing the critical challenges associated with them. Traditional quanvolutional layers, although beneficial for feature extraction, have largely been static, offering limited adaptability. Unlike state-of-the-art, our research overcomes this limitation by enabling training within these layers, significantly increasing the flexibility and potential of QuNNs. However, the introduction of multiple trainable quanvolutional layers induces complexities in gradient-based optimization, primarily due to the difficulty in accessing gradients across these layers. To resolve this, we propose a novel architecture, Residual Quanvolutional Neural Networks (ResQuNNs), leveraging the concept of residual learning, which facilitates the flow of gradients by adding skip connections between 
    
[^5]: 使用不同局部性对脉冲神经网络学习方法进行基准测试

    Benchmarking Spiking Neural Network Learning Methods with Varying Locality

    [https://arxiv.org/abs/2402.01782](https://arxiv.org/abs/2402.01782)

    本研究使用不同局部性对脉冲神经网络学习方法进行基准测试，并发现这些方法在性能和生物学合理性之间存在权衡。此外，研究还探讨了SNN的隐式循环特性。

    

    脉冲神经网络（SNN）提供更真实的神经动力学，在多个机器学习任务中已经显示出与人工神经网络（ANN）相当的性能。信息在SNN中以脉冲形式进行处理，采用事件驱动机制，显著降低了能源消耗。然而，由于脉冲机制的非可微性，训练SNN具有挑战性。传统方法如时间反向传播（BPTT）已经显示出一定的效果，但在计算和存储成本方面存在问题，并且在生物学上不可行。相反，最近的研究提出了具有不同局部性的替代学习方法，在分类任务中取得了成功。本文表明，这些方法在训练过程中有相似之处，同时在生物学合理性和性能之间存在权衡。此外，本研究还探讨了SNN的隐式循环特性，并进行了调查。

    Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but comes with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, this research examines the implicitly recurrent nature of SNNs and investigat
    
[^6]: Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning. (arXiv:2310.11594v1 [cs.LG])

    Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning. (arXiv:2310.11594v1 [cs.LG])

    [http://arxiv.org/abs/2310.11594](http://arxiv.org/abs/2310.11594)

    本文研究了联邦学习中对抗性训练和后门攻击的交叉点，引入了Adversarial Robustness Unhardening（ARU），通过有意介入分散式训练过程中破坏模型的鲁棒性，使模型更容易受到更广泛的逃避攻击。

    

    在当今的数据驱动环境中，维护用户隐私和释放数据潜力之间微妙的平衡成为一个重要关注点。联邦学习是一种以隐私为中心的解决方案，它实现了协作模型训练而无需共享数据。这种分散式方法带来了安全挑战，特别是恶意实体注入损坏数据的中毒和后门攻击。我们的研究最初受到测试时间逃避攻击的启发，探讨了联邦学习中对抗性训练和后门攻击的交叉点，引入了Adversarial Robustness Unhardening（ARU）。ARU被一部分对手使用，以有意介入分散式训练过程中破坏模型的鲁棒性，使模型更容易受到更广泛的逃避攻击。我们进行了广泛的实证实验，评估了ARU对对抗性训练和现有的鲁棒聚合防御策略对中毒和后门攻击的影响。

    In today's data-driven landscape, the delicate equilibrium between safeguarding user privacy and unleashing data potential stands as a paramount concern. Federated learning, which enables collaborative model training without necessitating data sharing, has emerged as a privacy-centric solution. This decentralized approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data. Our research, initially spurred by test-time evasion attacks, investigates the intersection of adversarial training and backdoor attacks within federated learning, introducing Adversarial Robustness Unhardening (ARU). ARU is employed by a subset of adversaries to intentionally undermine model robustness during decentralized training, rendering models susceptible to a broader range of evasion attacks. We present extensive empirical experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning a
    
[^7]: 缩放数据受限的语言模型

    Scaling Data-Constrained Language Models. (arXiv:2305.16264v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.16264](http://arxiv.org/abs/2305.16264)

    研究人员研究了在数据受限制的情况下缩放语言模型，并提出了一个计算最优性的缩放定律，考虑到重复令牌和过量参数的价值递减。

    

    现在扩展语言模型的趋势涉及增加参数计数和训练数据集大小。推断这个趋势表明，训练数据集大小可能很快就会受到互联网上可用文本数据的限制。出于此限制的动机，我们研究在数据受限制的情况下缩放语言模型。具体而言，我们运行了大量的实验，变化数据重复程度和计算预算，范围达到了9000亿个训练令牌和9亿参数模型。我们发现，在有限的数据的情况下，使用高达4次重复数据的训练与使用唯一数据相比对损失的贡献微不足道。然而，使用更多的重复数据，添加计算的价值最终会衰减为零。我们提出并经验证了一个计算最优性的缩放定律，考虑到重复令牌和过量参数的价值递减。最后，我们尝试了缓解数据稀缺的方法。

    The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the amount of text data available on the internet. Motivated by this limit, we investigate scaling language models in data-constrained regimes. Specifically, we run a large set of experiments varying the extent of data repetition and compute budget, ranging up to 900 billion training tokens and 9 billion parameter models. We find that with constrained data for a fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero. We propose and empirically validate a scaling law for compute optimality that accounts for the decreasing value of repeated tokens and excess parameters. Finally, we experiment with approaches mitigating data scarcity,
    

