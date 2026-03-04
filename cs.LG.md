# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Topic-based Watermarks for LLM-Generated Text](https://arxiv.org/abs/2404.02138) | 提出了一种新的基于主题的水印算法，旨在解决当前水印方案的局限性，为区分LLM生成的文本和人类生成的文本提供了新的思路。 |
| [^2] | [Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews](https://arxiv.org/abs/2403.07183) | 该研究提出了一种估计大语料库中被大语言模型大幅修改的文本比例的方法，并在AI会议的同行评审中进行了实证分析，发现6.5%至16.9%的文本可能被LLMs大幅修改，揭示了用户行为的一些见解。 |
| [^3] | [Dynamic deep-reinforcement-learning algorithm in Partially Observed Markov Decision Processes.](http://arxiv.org/abs/2307.15931) | 本研究研究了在部分可观测马尔可夫决策过程中解决的动作序列的好处，并提出了几种扩展深度强化学习算法的结构和方法。 |
| [^4] | [Convergence of a Normal Map-based Prox-SGD Method under the KL Inequality.](http://arxiv.org/abs/2305.05828) | 本文提出了一种新的随机正态映射算法用于非凸复合型优化问题，并证明其收敛性质。该方法扩展了基本Proximal随机梯度法的更有限的收敛保证。 |
| [^5] | [Beyond Accuracy: A Critical Review of Fairness in Machine Learning for Mobile and Wearable Computing.](http://arxiv.org/abs/2303.15585) | 本文通过对IMWUT期刊上过去五年发表的论文进行系统回顾，发现UbiComp社区在算法公平方面的进展滞后，存在敏感属性偏差导致的歧视性结果，需要探索报告数据集的信息以解决这些偏差。 |
| [^6] | [SPARLING: Learning Latent Representations with Extremely Sparse Activations.](http://arxiv.org/abs/2302.01976) | 本论文介绍了一种名为Sparling的技术，通过使用极度稀疏激活，在没有中间状态监督的情况下，从端到端标记示例中学习模型。该技术利用了一种新型的信息瓶颈来实现极度稀疏激活，达到了良好的中间状态建模精度，并且在不同领域的实验中取得了较高的准确性。 |

# 详细

[^1]: 基于主题的LLM生成文本的水印

    Topic-based Watermarks for LLM-Generated Text

    [https://arxiv.org/abs/2404.02138](https://arxiv.org/abs/2404.02138)

    提出了一种新的基于主题的水印算法，旨在解决当前水印方案的局限性，为区分LLM生成的文本和人类生成的文本提供了新的思路。

    

    大型语言模型（LLMs）的最新进展导致了生成的文本输出与人类生成的文本相似度难以分辨。水印算法是潜在工具，通过在LLM生成的输出中嵌入可检测的签名，可以区分LLM生成的文本和人类生成的文本。然而，当前的水印方案在已知攻击下缺乏健壮性。此外，考虑到LLM每天生成数万个文本输出，水印算法需要记忆每个输出才能让检测正常工作，这是不切实际的。本文针对当前水印方案的局限性，提出了针对LLMs的“基于主题的水印算法”概念。

    arXiv:2404.02138v1 Announce Type: cross  Abstract: Recent advancements of large language models (LLMs) have resulted in indistinguishable text outputs comparable to human-generated text. Watermarking algorithms are potential tools that offer a way to differentiate between LLM- and human-generated text by embedding detectable signatures within LLM-generated output. However, current watermarking schemes lack robustness against known attacks against watermarking algorithms. In addition, they are impractical considering an LLM generates tens of thousands of text outputs per day and the watermarking algorithm needs to memorize each output it generates for the detection to work. In this work, focusing on the limitations of current watermarking schemes, we propose the concept of a "topic-based watermarking algorithm" for LLMs. The proposed algorithm determines how to generate tokens for the watermarked LLM output based on extracted topics of an input prompt or the output of a non-watermarked 
    
[^2]: 在规模上监测AI修改的内容：AI会议同行评审中ChatGPT影响的案例研究

    Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews

    [https://arxiv.org/abs/2403.07183](https://arxiv.org/abs/2403.07183)

    该研究提出了一种估计大语料库中被大语言模型大幅修改的文本比例的方法，并在AI会议的同行评审中进行了实证分析，发现6.5%至16.9%的文本可能被LLMs大幅修改，揭示了用户行为的一些见解。

    

    我们提出了一种估计大语料库中文本可能被大语言模型（LLM）大幅修改或生成的部分比例的方法。我们的最大似然模型利用专家撰写和AI生成的参考文本，准确高效地检查语料库级别上真实世界LLM使用。我们将这种方法应用于AI会议上科学同行评审的案例研究，该研究发生在ChatGPT发布之后，包括ICLR 2024、NeurIPS 2023、CoRL 2023和EMNLP 2023。我们的研究结果表明，在这些会议提交的同行评审中，6.5%至16.9%的文本可能是由LLMs大幅修改的，即超出拼写检查或小幅更新的范围。生成文本出现的情况为用户行为提供了见解：在报告信心较低、在截止日期前提交的评论以及从评论公司

    arXiv:2403.07183v1 Announce Type: cross  Abstract: We present an approach for estimating the fraction of text in a large corpus which is likely to be substantially modified or produced by a large language model (LLM). Our maximum likelihood model leverages expert-written and AI-generated reference texts to accurately and efficiently examine real-world LLM-use at the corpus level. We apply this approach to a case study of scientific peer review in AI conferences that took place after the release of ChatGPT: ICLR 2024, NeurIPS 2023, CoRL 2023 and EMNLP 2023. Our results suggest that between 6.5% and 16.9% of text submitted as peer reviews to these conferences could have been substantially modified by LLMs, i.e. beyond spell-checking or minor writing updates. The circumstances in which generated text occurs offer insight into user behavior: the estimated fraction of LLM-generated text is higher in reviews which report lower confidence, were submitted close to the deadline, and from review
    
[^3]: 部分可观测马尔可夫决策过程中的动态深度强化学习算法

    Dynamic deep-reinforcement-learning algorithm in Partially Observed Markov Decision Processes. (arXiv:2307.15931v1 [cs.LG])

    [http://arxiv.org/abs/2307.15931](http://arxiv.org/abs/2307.15931)

    本研究研究了在部分可观测马尔可夫决策过程中解决的动作序列的好处，并提出了几种扩展深度强化学习算法的结构和方法。

    

    在最近的研究中，强化学习取得了很大的进步，并且在实际应用中引起了越来越多的兴趣。在许多情况下，由于非静态干扰，使得智能体难以保持性能。这种干扰产生了被称为部分可观测马尔可夫决策过程的环境。在实践中，部分可观测马尔可夫决策过程通过引入额外的估计器或在强化学习的上下文中使用递归神经网络来处理。这两种情况都需要处理轨迹上的序列信息。然而，目前只有很少有研究探讨要考虑的信息的影响以及处理它们的网络结构。本研究展示了在解决部分可观测马尔可夫决策过程时包含动作序列的好处，并提出了几种结构和方法来扩展最新的深度强化学习算法。

    Reinforcement learning has been greatly improved in recent studies and an increased interest in real-world implementation has emerged in recent years. In many cases, due to the non-static disturbances, it becomes challenging for the agent to keep the performance. The disturbance results in the environment called Partially Observable Markov Decision Process. In common practice, Partially Observable Markov Decision Process is handled by introducing an additional estimator, or Recurrent Neural Network is utilized in the context of reinforcement learning. Both of the cases require to process sequential information on the trajectory. However, there are only a few studies investigating the effect of information to consider and the network structure to handle them. This study shows the benefit of action sequence inclusion in order to solve Partially Observable Markov Decision Process. Several structures and approaches are proposed to extend one of the latest deep reinforcement learning algori
    
[^4]: 基于正态映射的Prox-SGD方法在KL不等式下的收敛性

    Convergence of a Normal Map-based Prox-SGD Method under the KL Inequality. (arXiv:2305.05828v1 [math.OC])

    [http://arxiv.org/abs/2305.05828](http://arxiv.org/abs/2305.05828)

    本文提出了一种新的随机正态映射算法用于非凸复合型优化问题，并证明其收敛性质。该方法扩展了基本Proximal随机梯度法的更有限的收敛保证。

    

    本文提出了一种新颖的随机正态映射算法（$\mathsf{norM}\text{-}\mathsf{SGD}$）用于非凸复合型优化问题，并讨论了其收敛性质。使用基于时间窗口的策略，首先分析了$\mathsf{norM}\text{-}\mathsf{SGD}$的全局收敛行为，并证明了所生成的迭代序列$\{\boldsymbol{x}^k\}_k$的每个累积点几乎确定地和期望上都对应于一个稳定点。所得结果在标准假设下成立，并扩展了基本Proximal随机梯度法的更有限的收敛保证。此外，基于著名的Kurdyka-{\L}ojasiewicz（KL）分析框架，我们为迭代序列$\{\boldsymbol{x}^k\}_k$提供了新的逐点收敛结果，并得出了取决于基础KL指数$\boldsymbol{\theta}$和步长动态$\{\alpha_k\}_k$的收敛速率。

    In this paper, we present a novel stochastic normal map-based algorithm ($\mathsf{norM}\text{-}\mathsf{SGD}$) for nonconvex composite-type optimization problems and discuss its convergence properties. Using a time window-based strategy, we first analyze the global convergence behavior of $\mathsf{norM}\text{-}\mathsf{SGD}$ and it is shown that every accumulation point of the generated sequence of iterates $\{\boldsymbol{x}^k\}_k$ corresponds to a stationary point almost surely and in an expectation sense. The obtained results hold under standard assumptions and extend the more limited convergence guarantees of the basic proximal stochastic gradient method. In addition, based on the well-known Kurdyka-{\L}ojasiewicz (KL) analysis framework, we provide novel point-wise convergence results for the iterates $\{\boldsymbol{x}^k\}_k$ and derive convergence rates that depend on the underlying KL exponent $\boldsymbol{\theta}$ and the step size dynamics $\{\alpha_k\}_k$. Specifically, for the 
    
[^5]: 机器学习中公平性的关键回顾：超越准确性在移动和可穿戴计算中的应用

    Beyond Accuracy: A Critical Review of Fairness in Machine Learning for Mobile and Wearable Computing. (arXiv:2303.15585v1 [cs.CY])

    [http://arxiv.org/abs/2303.15585](http://arxiv.org/abs/2303.15585)

    本文通过对IMWUT期刊上过去五年发表的论文进行系统回顾，发现UbiComp社区在算法公平方面的进展滞后，存在敏感属性偏差导致的歧视性结果，需要探索报告数据集的信息以解决这些偏差。

    

    移动、可穿戴和普及计算领域正在经历着机器学习的革命性整合。设备现在可以诊断疾病、预测心脏不规则动，发掘人类认知的全部潜力。然而，相关算法在敏感属性（如性别、种族等）方面可能存在偏差，导致歧视性结果。近期，人机交互（HCI）和人工智能伦理学（AI-Ethics）研究社区开始探索报告数据集的信息以揭示并最终对抗这些偏差。本文旨在探讨在这些报告方面UbiComp社区所采纳的程度，并强调潜在不足之处。通过对过去五年（2018-2022）在ACM交互、移动、可穿戴和普适技术（IMWUT）期刊上发表的论文进行系统回顾，我们发现UbiComp社区在算法公平方面的进展滞后。

    The field of mobile, wearable, and ubiquitous computing (UbiComp) is undergoing a revolutionary integration of machine learning. Devices can now diagnose diseases, predict heart irregularities, and unlock the full potential of human cognition. However, the underlying algorithms are not immune to biases with respect to sensitive attributes (e.g., gender, race), leading to discriminatory outcomes. The research communities of HCI and AI-Ethics have recently started to explore ways of reporting information about datasets to surface and, eventually, counter those biases. The goal of this work is to explore the extent to which the UbiComp community has adopted such ways of reporting and highlight potential shortcomings. Through a systematic review of papers published in the Proceedings of the ACM Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT) journal over the past 5 years (2018-2022), we found that progress on algorithmic fairness within the UbiComp community lags behind. 
    
[^6]: SPARLING：使用极度稀疏激活进行学习潜在表示

    SPARLING: Learning Latent Representations with Extremely Sparse Activations. (arXiv:2302.01976v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.01976](http://arxiv.org/abs/2302.01976)

    本论文介绍了一种名为Sparling的技术，通过使用极度稀疏激活，在没有中间状态监督的情况下，从端到端标记示例中学习模型。该技术利用了一种新型的信息瓶颈来实现极度稀疏激活，达到了良好的中间状态建模精度，并且在不同领域的实验中取得了较高的准确性。

    

    现实世界的过程常常包含可以被建模为极度稀疏张量的中间状态。我们引入了Sparling，一种允许您从仅具有端到端标记示例（即无中间状态的监督）中学习与该状态相匹配的模型的技术。Sparling使用一种新型的信息瓶颈，通过强制激活稀疏程度来实现其他技术无法达到的水平。我们发现，极度稀疏性是实现良好的中间状态建模所必需的。在我们的合成DigitCircle领域以及LaTeX-OCR和Audio-MNIST-Sequence领域中，即使我们仅进行端到端训练，我们也能以超过90％的准确性精确定位中间状态，即使存在特征置换的情况。

    Real-world processes often contain intermediate state that can be modeled as an extremely sparse tensor. We introduce Sparling, a technique that allows you to learn models with intermediate layers that match this state from only end-to-end labeled examples (i.e., no supervision on the intermediate state). Sparling uses a new kind of informational bottleneck that enforces levels of activation sparsity unachievable using other techniques. We find that extreme sparsity is necessary to achieve good intermediate state modeling. On our synthetic DigitCircle domain as well as the LaTeX-OCR and Audio-MNIST-Sequence domains, we are able to precisely localize the intermediate states up to feature permutation with > 90% accuracy, even though we only train end-to-end.
    

