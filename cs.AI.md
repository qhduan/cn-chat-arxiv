# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarial Machine Learning in Latent Representations of Neural Networks.](http://arxiv.org/abs/2309.17401) | 这项研究通过分析分布式深度神经网络对抗性行为的韧性填补了现有研究空白，并发现潜在特征在相同信息失真水平下比输入表示更加韧性，并且对抗性韧性由特征维度和神经网络的泛化能力共同决定。 |

# 详细

[^1]: 神经网络潜在表示中的对抗性机器学习

    Adversarial Machine Learning in Latent Representations of Neural Networks. (arXiv:2309.17401v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.17401](http://arxiv.org/abs/2309.17401)

    这项研究通过分析分布式深度神经网络对抗性行为的韧性填补了现有研究空白，并发现潜在特征在相同信息失真水平下比输入表示更加韧性，并且对抗性韧性由特征维度和神经网络的泛化能力共同决定。

    

    分布式深度神经网络已被证明可以减轻移动设备的计算负担，并降低边缘计算场景中的端到端推理延迟。尽管已经对分布式深度神经网络进行了研究，但据我们所知，分布式深度神经网络对于对抗性行为的韧性仍然是一个开放问题。在本文中，我们通过严格分析分布式深度神经网络对抗性行为的韧性来填补现有的研究空白。我们将这个问题置于信息论的背景下，并引入了两个新的衡量指标来衡量失真和韧性。我们的理论发现表明：（i）在假设具有相同信息失真水平的情况下，潜在特征始终比输入表示更加韧性；（ii）对抗性韧性同时由特征维度和深度神经网络的泛化能力决定。为了验证我们的理论发现，我们进行了广泛的实验分析，考虑了6种不同的深度神经网络架构。

    Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge the resilience of distributed DNNs to adversarial action still remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and introduce two new measurements for distortion and robustness. Our theoretical findings indicate that (i) assuming the same level of information distortion, latent features are always more robust than input representations; (ii) the adversarial robustness is jointly determined by the feature dimension and the generalization capability of the DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN arc
    

