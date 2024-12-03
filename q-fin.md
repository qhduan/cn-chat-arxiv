# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [QFNN-FFD: Quantum Federated Neural Network for Financial Fraud Detection](https://arxiv.org/abs/2404.02595) | 介绍了将量子机器学习和量子计算技术与联邦学习相结合的Quantum Federated Neural Network for Financial Fraud Detection (QFNN-FFD)框架，提出了一种安全、高效的欺诈交易识别方法，显著改进了欺诈检测并确保了数据机密性。 |
| [^2] | [Spanning Multi-Asset Payoffs With ReLUs](https://arxiv.org/abs/2403.14231) | 提出了一种用ReLU解决跨度多资产回报问题的方法，通过前馈神经网络提供了更好的离散跨度避险结果。 |

# 详细

[^1]: QFNN-FFD：用于金融欺诈检测的量子联邦神经网络

    QFNN-FFD: Quantum Federated Neural Network for Financial Fraud Detection

    [https://arxiv.org/abs/2404.02595](https://arxiv.org/abs/2404.02595)

    介绍了将量子机器学习和量子计算技术与联邦学习相结合的Quantum Federated Neural Network for Financial Fraud Detection (QFNN-FFD)框架，提出了一种安全、高效的欺诈交易识别方法，显著改进了欺诈检测并确保了数据机密性。

    

    这项研究介绍了Quantum Federated Neural Network for Financial Fraud Detection (QFNN-FFD)，这是一个融合了量子机器学习（QML）和量子计算技术与联邦学习（FL）的前沿框架，用于创新金融欺诈检测。利用量子技术的计算能力和FL的数据隐私，QFNN-FFD提出了一种安全、高效的识别欺诈交易的方法。在分布式客户端实施双阶段训练模型超越了现有的性能方法。QFNN-FFD显著改进了欺诈检测并确保了数据机密性，标志着金融科技解决方案的重大进步，并为以隐私为重点的欺诈检测建立了新标准。

    arXiv:2404.02595v1 Announce Type: cross  Abstract: This study introduces the Quantum Federated Neural Network for Financial Fraud Detection (QFNN-FFD), a cutting-edge framework merging Quantum Machine Learning (QML) and quantum computing with Federated Learning (FL) to innovate financial fraud detection. Using quantum technologies' computational power and FL's data privacy, QFNN-FFD presents a secure, efficient method for identifying fraudulent transactions. Implementing a dual-phase training model across distributed clients surpasses existing methods in performance. QFNN-FFD significantly improves fraud detection and ensures data confidentiality, marking a significant advancement in fintech solutions and establishing a new standard for privacy-focused fraud detection.
    
[^2]: 用ReLU跨度多资产回报

    Spanning Multi-Asset Payoffs With ReLUs

    [https://arxiv.org/abs/2403.14231](https://arxiv.org/abs/2403.14231)

    提出了一种用ReLU解决跨度多资产回报问题的方法，通过前馈神经网络提供了更好的离散跨度避险结果。

    

    我们提出了利用香草篮子期权的分布式形式来解决多资产回报的跨度问题。我们发现，只有当回报函数为偶函数且绝对齐次函数时，此问题才有唯一解，并且我们建立了一个基于傅立叶的公式来计算解决方案。金融回报通常是分段线性的，导致可能可以明确推导出解决方案，但在数值上可能难以利用。相比于基于单资产香草避险的行业偏爱方法，单隐藏层前馈神经网络为离散跨度提供了一种自然而高效的数值替代方法。我们测试了这种方法用于一些典型回报，并发现与单资产香草避险的产业偏好方法相比，利用香草篮子期权获得了更好的避险结果。

    arXiv:2403.14231v1 Announce Type: new  Abstract: We propose a distributional formulation of the spanning problem of a multi-asset payoff by vanilla basket options. This problem is shown to have a unique solution if and only if the payoff function is even and absolutely homogeneous, and we establish a Fourier-based formula to calculate the solution. Financial payoffs are typically piecewise linear, resulting in a solution that may be derived explicitly, yet may also be hard to numerically exploit. One-hidden-layer feedforward neural networks instead provide a natural and efficient numerical alternative for discrete spanning. We test this approach for a selection of archetypal payoffs and obtain better hedging results with vanilla basket options compared to industry-favored approaches based on single-asset vanilla hedges.
    

