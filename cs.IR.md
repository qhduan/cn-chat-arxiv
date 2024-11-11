# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Probabilistic Deep Supervision Network: A Noise-Resilient Approach for QoS Prediction.](http://arxiv.org/abs/2308.02580) | PDS-Net is a novel framework for QoS prediction that effectively reduces errors resulting from noise data by utilizing a probabilistic space and a condition-based multitasking loss function. |

# 详细

[^1]: Probabilistic Deep Supervision Network: 一种抗噪声的QoS预测方法

    Probabilistic Deep Supervision Network: A Noise-Resilient Approach for QoS Prediction. (arXiv:2308.02580v1 [cs.SE])

    [http://arxiv.org/abs/2308.02580](http://arxiv.org/abs/2308.02580)

    PDS-Net is a novel framework for QoS prediction that effectively reduces errors resulting from noise data by utilizing a probabilistic space and a condition-based multitasking loss function.

    

    在推荐系统中，QoS（服务质量）的预测是一项重要任务，准确预测未知的QoS值可以提高用户满意度。然而，现有的QoS预测技术在存在噪声数据（如虚假位置信息或虚拟网关）时可能表现不佳。在本文中，我们提出了一种新颖的QoS预测框架——概率深度监督网络（PDS-Net），以解决这个问题。PDS-Net利用基于高斯的概率空间监督中间层，并学习已知特征和真实标签的概率空间。此外，PDS-Net采用基于条件的多任务损失函数来识别具有噪声数据的对象，并通过优化这些对象的概率空间与真实标签概率空间之间的Kullback-Leibler距离，直接对从概率空间中采样的深度特征进行监督。因此，PDS-Net有效减少了因传播引起的错误。

    Quality of Service (QoS) prediction is an essential task in recommendation systems, where accurately predicting unknown QoS values can improve user satisfaction. However, existing QoS prediction techniques may perform poorly in the presence of noise data, such as fake location information or virtual gateways. In this paper, we propose the Probabilistic Deep Supervision Network (PDS-Net), a novel framework for QoS prediction that addresses this issue. PDS-Net utilizes a Gaussian-based probabilistic space to supervise intermediate layers and learns probability spaces for both known features and true labels. Moreover, PDS-Net employs a condition-based multitasking loss function to identify objects with noise data and applies supervision directly to deep features sampled from the probability space by optimizing the Kullback-Leibler distance between the probability space of these objects and the real-label probability space. Thus, PDS-Net effectively reduces errors resulting from the propag
    

