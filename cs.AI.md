# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative Semantic Communication: Diffusion Models Beyond Bit Recovery.](http://arxiv.org/abs/2306.04321) | 本文提出了一个新的生成扩散引导框架，通过利用扩散模型在合成多媒体内容同时保留语义特征方面的优势，以发送高度压缩的语义信息来降低带宽使用，从而超越现有方法在重建质量和语义保留方面的限制。 |
| [^2] | [Feed-Forward Optimization With Delayed Feedback for Neural Networks.](http://arxiv.org/abs/2304.13372) | 本文提出了一种延迟反馈的前馈神经网络优化方法F^3，使用延迟的误差信息来缩放梯度从而提高生物可行性和计算效率，具有较高的预测性能，为低能量训练和并行化提供了新思路。 |

# 详细

[^1]: 生成语义通讯：超越比特恢复的扩散模型

    Generative Semantic Communication: Diffusion Models Beyond Bit Recovery. (arXiv:2306.04321v1 [cs.AI])

    [http://arxiv.org/abs/2306.04321](http://arxiv.org/abs/2306.04321)

    本文提出了一个新的生成扩散引导框架，通过利用扩散模型在合成多媒体内容同时保留语义特征方面的优势，以发送高度压缩的语义信息来降低带宽使用，从而超越现有方法在重建质量和语义保留方面的限制。

    

    语义通讯被认为是下一代基于人工智能的通讯中的核心之一。语义通讯的一个可能性是，在不必恢复传输比特序列的情况下，在目标端重建与传输的图像或视频语义等效的副本。当前的解决方案仍然缺乏从接收到的部分信息中构建复杂场景的能力。本文旨在通过提出一种新的生成扩散引导框架来弥补这一差距，该框架利用了扩散模型在合成多媒体内容同时保留语义特征方面的优势。我们通过仅发送高度压缩的语义信息来降低带宽使用。然后接收端的扩散模型生成高质量的多媒体内容。实验结果表明，我们提出的框架在重建质量和语义保留方面优于现有方法。

    Semantic communication is expected to be one of the cores of next-generation AI-based communications. One of the possibilities offered by semantic communication is the capability to regenerate, at the destination side, images or videos semantically equivalent to the transmitted ones, without necessarily recovering the transmitted sequence of bits. The current solutions still lack the ability to build complex scenes from the received partial information. Clearly, there is an unmet need to balance the effectiveness of generation methods and the complexity of the transmitted information, possibly taking into account the goal of communication. In this paper, we aim to bridge this gap by proposing a novel generative diffusion-guided framework for semantic communication that leverages the strong abilities of diffusion models in synthesizing multimedia content while preserving semantic features. We reduce bandwidth usage by sending highly-compressed semantic information only. Then, the diffus
    
[^2]: 延迟反馈的前馈优化神经网络

    Feed-Forward Optimization With Delayed Feedback for Neural Networks. (arXiv:2304.13372v1 [cs.LG])

    [http://arxiv.org/abs/2304.13372](http://arxiv.org/abs/2304.13372)

    本文提出了一种延迟反馈的前馈神经网络优化方法F^3，使用延迟的误差信息来缩放梯度从而提高生物可行性和计算效率，具有较高的预测性能，为低能量训练和并行化提供了新思路。

    

    反向传播长期以来一直受到生物学上的批评，因为它依赖于自然学习过程中不可行的概念。本文提出了一种替代方法来解决两个核心问题，即权重传输和更新锁定，以实现生物可行性和计算效率。我们引入了延迟反馈的前馈（F^3），通过利用延迟的误差信息作为样本级缩放因子来更准确地近似梯度，改进了先前的工作。我们发现，F^3将生物可行性训练算法和反向传播之间的预测性能差距缩小了高达96％。这证明了生物可行性训练的适用性，并为低能量训练和并行化开辟了有 promising 的新方向。

    Backpropagation has long been criticized for being biologically implausible, relying on concepts that are not viable in natural learning processes. This paper proposes an alternative approach to solve two core issues, i.e., weight transport and update locking, for biological plausibility and computational efficiency. We introduce Feed-Forward with delayed Feedback (F$^3$), which improves upon prior work by utilizing delayed error information as a sample-wise scaling factor to approximate gradients more accurately. We find that F$^3$ reduces the gap in predictive performance between biologically plausible training algorithms and backpropagation by up to 96%. This demonstrates the applicability of biologically plausible training and opens up promising new avenues for low-energy training and parallelization.
    

