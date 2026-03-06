# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TTA-Nav: Test-time Adaptive Reconstruction for Point-Goal Navigation under Visual Corruptions](https://arxiv.org/abs/2403.01977) | TTA-Nav提出了一种测试时自适应方法，通过引入自顶向下解码器，从损坏图像中重建出更清晰的图像，显著增强了点目标导航性能。 |
| [^2] | [Deep Learning Meets Mechanism Design: Key Results and Some Novel Applications.](http://arxiv.org/abs/2401.05683) | 本文介绍了深度学习与机制设计的结合，探讨了使用深度学习方法在无法同时满足所有期望特性的情况下，学习近似满足特性要求的机制。 |

# 详细

[^1]: TTA-Nav: 测试时自适应重建用于视觉损坏下的点目标导航

    TTA-Nav: Test-time Adaptive Reconstruction for Point-Goal Navigation under Visual Corruptions

    [https://arxiv.org/abs/2403.01977](https://arxiv.org/abs/2403.01977)

    TTA-Nav提出了一种测试时自适应方法，通过引入自顶向下解码器，从损坏图像中重建出更清晰的图像，显著增强了点目标导航性能。

    

    arXiv:2403.01977v1 公告类型: 跨  摘要: 在视觉损坏下的机器人导航是一个巨大的挑战。为了解决这一问题，我们提出了一种名为TTA-Nav的测试时自适应（TTA）方法，用于在视觉损坏下的点目标导航。我们的“即插即用”方法将自顶向下的解码器与预训练的导航模型相结合。首先，预训练的导航模型接收一个损坏的图像并提取特征。其次，自顶向下的解码器根据预训练模型提取的高级特征生成重建图像。然后，将损坏图像的重建图像馈送回预训练模型。最后，预训练模型再次进行前向传播以输出动作。尽管仅在清晰图像上训练，自顶向下的解码器可以从损坏图像中重建出更清晰的图像，无需基于梯度的自适应。具有我们自顶向下解码器的预训练导航模型显著提高了导航性能。

    arXiv:2403.01977v1 Announce Type: cross  Abstract: Robot navigation under visual corruption presents a formidable challenge. To address this, we propose a Test-time Adaptation (TTA) method, named as TTA-Nav, for point-goal navigation under visual corruptions. Our "plug-and-play" method incorporates a top-down decoder to a pre-trained navigation model. Firstly, the pre-trained navigation model gets a corrupted image and extracts features. Secondly, the top-down decoder produces the reconstruction given the high-level features extracted by the pre-trained model. Then, it feeds the reconstruction of a corrupted image back to the pre-trained model. Finally, the pre-trained model does forward pass again to output action. Despite being trained solely on clean images, the top-down decoder can reconstruct cleaner images from corrupted ones without the need for gradient-based adaptation. The pre-trained navigation model with our top-down decoder significantly enhances navigation performance acr
    
[^2]: 深度学习与机制设计：关键结果和一些新的应用

    Deep Learning Meets Mechanism Design: Key Results and Some Novel Applications. (arXiv:2401.05683v1 [cs.GT])

    [http://arxiv.org/abs/2401.05683](http://arxiv.org/abs/2401.05683)

    本文介绍了深度学习与机制设计的结合，探讨了使用深度学习方法在无法同时满足所有期望特性的情况下，学习近似满足特性要求的机制。

    

    机制设计本质上是对游戏的逆向工程，涉及在博弈中诱导一种方式，使得诱导的博弈在博弈均衡中满足一组期望的特性。机制的期望特性包括激励兼容性、个体合理性、福利最大化、收入最大化（或成本最小化）、分配公平等。根据机制设计理论，只有某些严格的子集可以同时被任何给定的机制完全满足。在现实世界应用中，通常所需的机制可能需要一些在理论上无法同时满足的特性子集。在这种情况下，一个显著的近期方法是使用基于深度学习的方法，通过最小化适当定义的损失函数来学习一个近似满足所需特性的机制。在本文中，我们从相关文献中介绍了技术细节。

    Mechanism design is essentially reverse engineering of games and involves inducing a game among strategic agents in a way that the induced game satisfies a set of desired properties in an equilibrium of the game. Desirable properties for a mechanism include incentive compatibility, individual rationality, welfare maximisation, revenue maximisation (or cost minimisation), fairness of allocation, etc. It is known from mechanism design theory that only certain strict subsets of these properties can be simultaneously satisfied exactly by any given mechanism. Often, the mechanisms required by real-world applications may need a subset of these properties that are theoretically impossible to be simultaneously satisfied. In such cases, a prominent recent approach is to use a deep learning based approach to learn a mechanism that approximately satisfies the required properties by minimizing a suitably defined loss function. In this paper, we present, from relevant literature, technical details 
    

